"""Generation: prompt -> tree, by sampling moves (temperature > 0) or taking the most likely one (temperature 0).

The pieces are reused by the completion evaluation (evaluate.py): TreeState holds a partial tree with its decoder
entries, score_moves() scores every possible next move, choose_move() picks one, TreeState.apply() writes it.
"""
import random

import torch

from .sequences import neighbour_ids
from .tree import Node
from .vocab import END, PAD, ROOT, UNK

MAX_MOVES = 200         # hard cap on the moves of free generation (from empty, or on top of a partial tree)


class TreeState:
    """A partial tree and its decoder entries E.

    Each node keeps: chain (its ID path: self, father, ...), father (node index), kids (its children, in their
    current order), content (its tokens so far), last (index of its latest entry, whose hidden state represents
    it in the heads), and closed (its <end> is written). Node 0 is the root.
    """

    def __init__(self, vocab, cfg, root_id):
        self.n_ids, self.max_anc, self.end_id = cfg["n_ids"], cfg["max_anc"], vocab.id(END)
        # Same start as training: entry 0 is the root with a random ID.
        self.E = dict(self_id=[root_id], fa_id=[self.n_ids], anc=[[]], nb=[[-1, -1, -1]], tok=[vocab.id(ROOT)],
                      pos=[0], depth=[0])
        self.nodes = [dict(chain=[root_id], father=-1, kids=[], content=[], last=0, closed=True)]  # not a line
        self.cache = {}                           # the decoder's attention cache (see TreeDecoder.extend)

    def apply(self, f, j, pl, tok, new_id=None):
        """Write one move: token tok on line j under father f; j == len(nodes) starts a new line (ID new_id),
        inserted among f's children right after pl (or first, pl = -1). The program fills in the entry."""
        nodes, E = self.nodes, self.E
        if j == len(nodes):
            kids = nodes[f]["kids"]
            kids.insert(kids.index(pl) + 1 if pl >= 0 else 0, j)
            nodes.append(dict(chain=[new_id] + nodes[f]["chain"], father=f, kids=[], content=[], last=None,
                              closed=False))
        node = nodes[j]
        ch, kids = node["chain"], nodes[node["father"]]["kids"]
        E["self_id"].append(ch[0]); E["fa_id"].append(ch[1]); E["anc"].append(ch[2:][:self.max_anc])
        E["nb"].append(neighbour_ids(kids, kids.index(j), lambda k: nodes[k]["chain"][0], self.n_ids + 1))
        E["tok"].append(tok); E["pos"].append(len(node["content"])); E["depth"].append(len(ch) - 1)
        node["content"].append(tok)
        node["last"] = len(E["tok"]) - 1
        node["closed"] = tok == self.end_id

    def unused_id(self, rng):
        used = {n["chain"][0] for n in self.nodes}
        return rng.choice([i for i in range(self.n_ids) if i not in used])

    def tree(self, vocab):
        tree = [Node([vocab.itos[t] for t in n["content"] if t != self.end_id]) for n in self.nodes]
        for i, n in enumerate(self.nodes):            # siblings in the order the insertions have built
            tree[i].children = [tree[k] for k in n["kids"]]
        return tree[0]


def entries_batch(E, start, max_anc, device):
    """Decoder input for entries start.. of E, as a batch of one."""
    L = len(E["tok"])
    anc = torch.full((1, L - start, max_anc), -1, dtype=torch.long)
    for i, a in enumerate(E["anc"][start:]):
        if a:
            anc[0, i, : len(a)] = torch.tensor(a)
    b = {k: torch.tensor([E[k][start:]], device=device) for k in ("self_id", "fa_id", "nb", "tok", "pos", "depth")}
    b["anc"] = anc.to(device)
    b["pad"] = torch.zeros(1, L - start, dtype=torch.bool, device=device)
    return b


@torch.no_grad()
def score_moves(dec, state, memory, mem_pad, vocab, device, max_nodes=None, max_line=None, use_cache=True,
                allow_unk=False):
    """Score every possible next move of a partial tree.

    Returns (cand, cand_lp, lt, stop_lp, can_stop): cand lists the candidates (f, j, place) — (f, open line, -2) or
    (f, new line N, after child / -1 first); log P(move with token t) = cand_lp[k] + lt[k, t]; stop_lp = log P(STOP),
    allowed only when can_stop (no line open). With use_cache, only the entries added since the last call go
    through the decoder (the state keeps the attention cache); without, it reruns on all entries (same result).
    <unk> can't be written unless allow_unk (the evaluation allows it: there it is the reference's own token for a
    rare one, and banning it would leave the model no right move, so it would start new lines instead).
    """
    E, nodes = state.E, state.nodes
    L, N = len(E["tok"]), len(nodes)
    is_open = lambda n: not n["closed"] and (max_line is None or len(n["content"]) < max_line)
    if use_cache:
        start = state.cache["H"].shape[1] if state.cache else 0
        H = dec.extend(entries_batch(E, start, state.max_anc, device), memory, mem_pad, state.cache)
    else:
        H = dec(entries_batch(E, 0, state.max_anc, device), memory, mem_pad)
    h = H[0, -1:]
    node_ids = torch.tensor([[n["chain"][0] for n in nodes]], device=device)

    P_fa = dec.father_logp(h[None], node_ids, torch.ones(1, 1, N, dtype=torch.bool, device=device))[0, 0]
    u = dec.with_father(h.expand(N, -1), H[0, [n["last"] for n in nodes]])
    # child[f, j]: node j is a child of f; its line can be extended only while it is open
    fathers = torch.tensor([n["father"] for n in nodes], device=device)
    child = fathers[None, :] == torch.arange(N, device=device)[:, None]
    is_open_t = torch.tensor([is_open(n) for n in nodes], device=device)
    P_id = dec.id_logp(u, node_ids.expand(N, -1), child & is_open_t)             # [father, line or new]
    P_pl = dec.place_logp(u, node_ids.expand(N, -1), child)                     # [father, after child or first]
    cand = [(f, j, -2) for f, j in child.logical_and(is_open_t).nonzero().tolist()]
    if max_nodes is None or N <= max_nodes:       # new lines: after any child of the father, or first
        cand += [(f, N, pl) for f in range(N) for pl in nodes[f]["kids"] + [-1]]
    can_stop = not any(is_open(n) for n in nodes)  # STOP only when every line is closed
    if not cand:
        return cand, None, None, P_fa[N], can_stop
    fs, js, pls = torch.tensor(cand, device=device).T
    new = js == N
    h_line = H[0, [0 if j == N else nodes[j]["last"] for j in js.tolist()]]
    h_place = H[0, [nodes[p]["last"] if p >= 0 else 0 for p in pls.tolist()]]
    lt = dec.token_logits(u[fs], h_line, ~new, h_place, pls >= 0).log_softmax(-1)
    lt[:, [vocab.id(PAD), vocab.id(ROOT)] + ([] if allow_unk else [vocab.id(UNK)])] = float("-inf")
    lt[new, state.end_id] = float("-inf")         # a new line can't be empty
    cand_lp = P_fa[fs] + P_id[fs, js] + torch.where(new, P_pl[fs, torch.where(pls >= 0, pls, N)], 0.0)
    return cand, cand_lp, lt, P_fa[N], can_stop


def choose_move(cand_lp, lt, stop_lp, can_stop, temperature, gen):
    """Pick (k, token) — candidate index and token — or None for STOP.

    temperature > 0: sampled from P(move)^(1/temperature), over all candidates and all tokens (and STOP if allowed).
    temperature 0: the most likely move (greedy), each candidate taking its best token.
    """
    if cand_lp is None:
        return None
    if temperature > 0:   # in two draws, the same distribution: the candidate (summed over its tokens), then the token
        scores = (cand_lp[:, None] + lt) / temperature                # [candidates, vocab]
        row = torch.logsumexp(scores, -1)
        if can_stop:
            row = torch.cat([row, stop_lp[None] / temperature])
        k = int(torch.multinomial(row.softmax(-1), 1, generator=gen))
        if can_stop and k == len(row) - 1:
            return None
        return k, int(torch.multinomial(scores[k].softmax(-1), 1, generator=gen))
    best_tok, tok_ids = lt.max(-1)
    joint = cand_lp + best_tok
    if can_stop and stop_lp >= joint.max():
        return None
    k = int(joint.argmax())
    return k, int(tok_ids[k])


@torch.no_grad()
def generate(enc, dec, tokenizer, vocab, prompt, cfg, rng=None, max_nodes=64, max_line=64, max_entries=MAX_MOVES + 1,
             device="cpu", temperature=0.2):
    """From an empty tree: at each step a move (father, line, token) or (father, new line, place, token), or STOP.

    A line is open until its <end> (or until it reaches max_line tokens). STOP is allowed only when no line is open,
    and ends generation, so it is produced at most once.
    """
    rng = rng or random.Random(0)
    gen = torch.Generator(device).manual_seed(rng.randrange(2 ** 31))   # for sampling, reproducible from rng
    ids, mask = tokenizer([prompt])
    memory, mem_pad = enc(ids.to(device), mask.to(device)), (mask == 0).to(device)
    state = TreeState(vocab, cfg, rng.randrange(cfg["n_ids"]))
    while len(state.E["tok"]) < max_entries:
        cand, cand_lp, lt, stop_lp, can_stop = score_moves(dec, state, memory, mem_pad, vocab, device,
                                                           max_nodes, max_line)
        move = choose_move(cand_lp, lt, stop_lp, can_stop, temperature, gen)
        if move is None:                          # STOP, or nothing can be written any more
            break
        k, tok = move
        f, j, pl = cand[k]
        state.apply(f, j, pl, tok, state.unused_id(rng) if j == len(state.nodes) else None)
    return state.tree(vocab)
