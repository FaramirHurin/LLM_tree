"""Tree -> decoder entries and frontier targets, plus batching.

Every step writes one token. The model picks a father, then an ID (an existing child of that father,
to extend that line, or a new line), then the token, which goes at the end of that line.
Each entry is [self ID, father ID, older ancestors, child numbers, token, in-line position]; the program fills
it in. The child numbers (one per path level: self, father, ...) say where each node sits among its siblings,
so the entries encode sibling order too.

With order="random" a sequence is built in three parts:
  1. a random partial tree (the mask): some lines, each holding a prefix of its tokens;
  2. that partial tree written out in a random valid order; its last entry is the "real point";
  3. the rest of the tree, in a random valid order.
After each entry, the valid next moves are recorded: extend any unfinished line, start the next unwritten
child of any existing node (siblings keep their real order), or STOP once the tree is complete.
"""
import torch

from .vocab import ROOT


def tree_arrays(root, vocab):
    """Flatten a Node tree into (contents, parent, children) lists. Index 0 is the root."""
    contents, parent, children = [], [], []

    def visit(n, p):
        i = len(contents)
        contents.append([vocab.id(t) for t in n.content]); parent.append(p); children.append([])
        if p >= 0:
            children[p].append(i)
        for c in n.children:
            visit(c, i)

    visit(root, -1)
    return contents, parent, children


def sample_mask(contents, children, rng):
    """Random partial tree: tokens to write per line. The first k children of each existing node exist.

    Returns n_toks[node]: how many of its tokens are written before the real point (0 = not started).
    Starting at the root, each existing node keeps a random number of its first children, and each kept child
    a random non-empty prefix of its line. Keeping siblings as a prefix makes the mask reachable by valid moves.
    """
    n_toks, stack = [0] * len(contents), [0]
    while stack:
        p = stack.pop()
        for c in children[p][: rng.randint(0, len(children[p]))]:
            n_toks[c] = rng.randint(1, len(contents[c]))
            stack.append(c)
    return n_toks


def flatten(arrays, vocab, rng, n_ids, order="random", max_anc=4, max_moves=4):
    """Build one training sequence with fresh random IDs.

    order="random": the mask, then random valid moves (lines interleave freely).
    order="dfs":    a fixed order: whole lines, depth-first. No real point.

    Besides the entries, returns per node its ID, father and first entry, and:
      moves:  (t, father node, node or -1 for a new line, token, father's last entry, line's last entry or -1),
              the valid moves after entry t; the move actually taken plus up to max_moves-1 others
      stop_t: the entry after which STOP is valid (the last one)
      real_t: the real point (the end of the masked part), or -1
    """
    contents, parent, children = arrays
    N = len(contents)
    # Bookkeeping per node: written = tokens written so far; first / last = its first / latest entry index
    # (the latest entry's hidden state stands for the node in the heads); chain = its ID path.
    ids = rng.sample(range(n_ids), N)                     # fresh random IDs per example
    chain = [[ids[0]]] + [None] * (N - 1)                 # node -> [self, father, grandfather, ...] IDs
    rank = [[-1]] + [None] * (N - 1)                      # same levels, child numbers; the root has none (-1)
    written, first, last = [0] * N, [0] + [-1] * (N - 1), [0] * N
    started, clock = [0] * N, 0                           # when each node was started (for dfs)
    front = list(children[0][:1])                         # nodes whose next token can be written now
    mask = sample_mask(contents, children, rng) if order == "random" else None
    real_t = -1

    # Entry 0 is the root: its own ID, father "none" (n_ids), the <root> token. It gives the model a start.
    E = {"self_id": [ids[0]], "fa_id": [n_ids], "anc": [[]], "rank": [[-1]], "tok": [vocab.id(ROOT)], "pos": [0]}
    moves = []
    while front:                                          # each loop: record the targets after entry t, then write entry t+1
        t = len(E["tok"]) - 1
        if order == "dfs":   # finish the open line, else start a child of the latest started node
            n = max(front, key=lambda m: (written[m] > 0, started[parent[m]]))
        else:
            inside = [m for m in front if written[m] < mask[m]]   # moves that stay inside the mask
            if not inside and real_t < 0:                 # the mask is fully written: this step is the real point
                real_t = t
            n = rng.choice(inside or front)

        # Targets: every move in front is correct here. Keep the one taken plus a few others (to save compute).
        for m in [n] + rng.sample([m for m in front if m != n], min(max_moves - 1, len(front) - 1)):
            new = written[m] == 0
            moves.append((t, parent[m], -1 if new else m, contents[m][written[m]], last[parent[m]],
                          -1 if new else last[m]))

        p = parent[n]
        if written[n] == 0:                               # a new line: its next sibling and first child open up
            k = children[p].index(n)                      # its child number: siblings start in order
            chain[n], rank[n] = [ids[n]] + chain[p], [k] + rank[p]
            first[n], clock = t + 1, clock + 1
            started[n] = clock
            if k + 1 < len(children[p]):
                front.append(children[p][k + 1])
            if children[n]:
                front.append(children[n][0])
        E["self_id"].append(ids[n]); E["fa_id"].append(ids[p]); E["anc"].append(chain[n][2:][:max_anc])
        E["rank"].append(rank[n][: 2 + max_anc])
        E["tok"].append(contents[n][written[n]]); E["pos"].append(written[n])
        written[n] += 1
        last[n] = t + 1
        if written[n] == len(contents[n]):                # line complete: nothing more to add to it
            front.remove(n)

    # front is empty: the whole tree is written, so STOP is the (only) valid move after the last entry
    return dict(E, ids=ids, parent=parent, first=first, moves=moves, stop_t=len(E["tok"]) - 1, real_t=real_t)


def collate(items, prompts, tokenizer, max_anc):
    """Pad a list of flatten() outputs into tensors; moves get their batch index.

    Entries are [B,L] (anc [B,L,max_anc], rank [B,L,2+max_anc], -1 = none); per-node tensors (node_ids, node_parent, node_first) are [B,N];
    moves is [M,7] = (b, t, father node, line node or -1, token, father's last entry, line's last entry or -1).
    """
    B, L = len(items), max(len(it["tok"]) for it in items)
    N = max(len(it["ids"]) for it in items)
    t = lambda fill, n=L: torch.full((B, n), fill, dtype=torch.long)
    self_id, fa_id, tok, pos = t(0), t(0), t(0), t(0)
    anc = torch.full((B, L, max_anc), -1, dtype=torch.long)
    rank = torch.full((B, L, 2 + max_anc), -1, dtype=torch.long)
    pad = torch.ones(B, L, dtype=torch.bool)
    node_ids, node_parent, node_first = t(0, N), t(-1, N), t(L, N)    # padded nodes never exist
    moves, stop, real = [], [], []
    for b, it in enumerate(items):
        n, k = len(it["tok"]), len(it["ids"])
        self_id[b, :n] = torch.tensor(it["self_id"]); fa_id[b, :n] = torch.tensor(it["fa_id"])
        tok[b, :n] = torch.tensor(it["tok"]); pos[b, :n] = torch.tensor(it["pos"]); pad[b, :n] = False
        for i, (a, r) in enumerate(zip(it["anc"], it["rank"])):
            if a:
                anc[b, i, : len(a)] = torch.tensor(a)
            rank[b, i, : len(r)] = torch.tensor(r)
        node_ids[b, :k] = torch.tensor(it["ids"]); node_parent[b, :k] = torch.tensor(it["parent"])
        node_first[b, :k] = torch.tensor(it["first"])
        moves += [(b,) + m for m in it["moves"]]
        stop.append((b, it["stop_t"]))
        if it["real_t"] >= 0:
            real.append((b, it["real_t"]))
    T = lambda x, w: torch.tensor(x, dtype=torch.long).view(-1, w)
    p_ids, p_mask = tokenizer(prompts)
    return dict(self_id=self_id, fa_id=fa_id, anc=anc, rank=rank, tok=tok, pos=pos, pad=pad, node_ids=node_ids,
                node_parent=node_parent, node_first=node_first, moves=T(moves, 7), stop=T(stop, 2),
                real=T(real, 2), p_ids=p_ids, p_mask=p_mask)
