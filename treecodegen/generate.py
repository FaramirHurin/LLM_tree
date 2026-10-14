"""Greedy generation: prompt -> tree."""
import random

import torch

from .tree import Node
from .vocab import END, PAD, ROOT, UNK


@torch.no_grad()
def generate(enc, dec, tokenizer, vocab, prompt, cfg, rng=None, max_nodes=64, max_line=64, max_entries=1024,
             device="cpu"):
    """Each step: the most likely (father, line or new line, token), or STOP.

    Each node keeps: chain (its ID path: self, father, ...), rank (the child numbers on that path),
    father (node index), content (its tokens so far), n_children, and last (index of its latest entry, whose hidden state represents it in the heads).
    STOP ends generation, so it is produced at most once.
    """
    rng = rng or random.Random(0)
    ids, mask = tokenizer([prompt])
    memory, mem_pad = enc(ids.to(device), mask.to(device)), (mask == 0).to(device)
    n_ids, max_anc = cfg["n_ids"], cfg["max_anc"]
    banned = [vocab.id(PAD), vocab.id(UNK), vocab.id(ROOT), vocab.stoi.get(END, 0)]

    # Same start as training: entry 0 is the root with a random ID.
    root_id = rng.randrange(n_ids)
    E = dict(self_id=[root_id], fa_id=[n_ids], anc=[[]], rank=[[-1]], tok=[vocab.id(ROOT)], pos=[0])
    nodes = [dict(chain=[root_id], rank=[-1], father=-1, content=[], last=0, n_children=0)]

    while len(E["tok"]) < max_entries:
        # Run the decoder on all entries so far (no cache) and take the last hidden state.
        L, N = len(E["tok"]), len(nodes)
        anc = torch.full((1, L, max_anc), -1, dtype=torch.long)
        rank = torch.full((1, L, 2 + max_anc), -1, dtype=torch.long)
        for i, (a, r) in enumerate(zip(E["anc"], E["rank"])):
            if a:
                anc[0, i, : len(a)] = torch.tensor(a)
            rank[0, i, : len(r)] = torch.tensor(r)
        b = {k: torch.tensor([E[k]], device=device) for k in ("self_id", "fa_id", "tok", "pos")}
        b["anc"], b["rank"] = anc.to(device), rank.to(device)
        b["pad"] = torch.zeros(1, L, dtype=torch.bool, device=device)
        H = dec(b, memory, mem_pad)
        h = H[0, -1:]
        node_ids = torch.tensor([[n["chain"][0] for n in nodes]], device=device)

        # score every (father, line) pair with its best token and keep the best triple
        P_fa = dec.father_logp(h[None], node_ids, torch.ones(1, 1, N, dtype=torch.bool, device=device))[0, 0]
        u = dec.with_father(h.expand(N, -1), H[0, [n["last"] for n in nodes]])
        # allowed[f, j]: node j is a child of f whose line can still grow
        allowed = torch.tensor([[n["father"] == f and len(n["content"]) < max_line for n in nodes] for f in range(N)],
                               device=device)
        P_id = dec.id_logp(u, node_ids.expand(N, -1), allowed)                  # [father, line or new]
        if N > max_nodes:
            P_id[:, N] = float("-inf")
        fs, js = (P_id > float("-inf")).nonzero(as_tuple=True)
        h_line = H[0, [0 if j == N else nodes[j]["last"] for j in js.tolist()]]
        lt = dec.token_logits(u[fs], h_line, js < N).log_softmax(-1)
        lt[:, banned] = float("-inf")
        best_tok, tok_ids = lt.max(-1)
        joint = P_fa[fs] + P_id[fs, js] + best_tok
        k = int(joint.argmax())
        if P_fa[N] >= joint[k]:                       # STOP
            break
        f, j, tok = int(fs[k]), int(js[k]), int(tok_ids[k])
        new = j == N

        # A new line gets any unused ID; then the program fills in the entry (self, father, ancestors, position).
        if new:
            used = {n["chain"][0] for n in nodes}
            new_id = rng.choice([i for i in range(n_ids) if i not in used])
            k = nodes[f]["n_children"]                # the next child number under this father
            nodes[f]["n_children"] += 1
            nodes.append(dict(chain=[new_id] + nodes[f]["chain"], rank=[k] + nodes[f]["rank"], father=f,
                              content=[], last=None, n_children=0))
        node = nodes[j]
        ch = node["chain"]
        E["self_id"].append(ch[0]); E["fa_id"].append(ch[1]); E["anc"].append(ch[2:][:max_anc])
        E["rank"].append(node["rank"][: 2 + max_anc])
        E["tok"].append(tok); E["pos"].append(len(node["content"]))
        node["content"].append(tok)
        node["last"] = L

    tree = [Node([vocab.itos[t] for t in n["content"]]) for n in nodes]
    for i, n in enumerate(nodes[1:], 1):          # siblings in the order they were started (= child number)
        tree[n["father"]].children.append(tree[i])
    return tree[0]
