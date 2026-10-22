"""Tree -> decoder entries and frontier targets, plus batching.

Every step writes one token. The model picks a father, then an ID (an existing child of that father,
to extend that line, or a new line), for a new line its place among the siblings (right after which existing
child, or first), then the token, which goes at the end of that line.
Each entry is [self ID, father ID, older ancestors, neighbours, token, in-line position, depth]; the program fills
it in. The neighbours are the line's two left and one right siblings when the entry is written (EDGE past the
first or last sibling): a snapshot, so a later insertion never changes an entry already written.
Every line ends with <end>, written like any other token; after it the line is closed (it can't be extended,
but it can still get children). STOP, once, ends the whole tree.

With order="random" a sequence is built in three parts:
  1. a random partial tree (the mask): some lines, each holding a prefix of its tokens, of one of three kinds (mask=):
     "subset": from the root down, each started node keeps a random subset of its children (small trees: the
               median mask holds 9% of the file);
     "holes":  the file minus holes (whole subtrees or runs of siblings), a share of lines uniform between 0 and
               100%; every other line complete except at most one cut short;
     "deep":   holes only below the top one or two levels: the skeleton (imports, class and def headers) is complete
               and bodies have holes;
     "mix":    "deep" with probability deep_share, "holes" with probability holes_share, else "subset" (one draw per
               sequence);
  2. that partial tree written out in a random valid order; its last entry is the "real point";
  3. the rest of the tree, in a random valid order.
Each sequence draws a stickiness s in [0, 1]: at each step it continues the line it just wrote with probability s,
else it picks any valid move. So the orders range from fully interleaved (s = 0) to line by line (s = 1),
which is how greedy generation tends to write.
A share line_share of sequences are "line-first": after the real point, while any line is open, only continuing an
open line is valid (and recorded as a target); a new line can start only once every started line is finished.
After each entry, the valid next moves are recorded: extend any unfinished line, start any unwritten child of
any started node (siblings in any order, each at its place among those already started), or STOP once the tree
is complete.
"""
import bisect

import torch

from .vocab import END, ROOT


def tree_arrays(root, vocab):
    """Flatten a Node tree into (contents, parent, children) lists. Index 0 is the root.

    Every line's content ends with <end>; the root has no content.
    """
    contents, parent, children = [], [], []

    def visit(n, p):
        i = len(contents)
        contents.append([vocab.id(t) for t in n.content] + ([vocab.id(END)] if p >= 0 else []))
        parent.append(p); children.append([])
        if p >= 0:
            children[p].append(i)
        for c in n.children:
            visit(c, i)

    visit(root, -1)
    return contents, parent, children


def neighbour_ids(sibs, i, id_of, edge):
    """[left 1, left 2, right 1] IDs of the node at index i of sibs (its father's started children, in order).

    Past the first or last sibling the slot is `edge`; a left slot beyond an edge is -1 (none).
    """
    left = [id_of(s) for s in sibs[max(0, i - 2):i][::-1]]           # nearest first
    if len(left) < 2:
        left.append(edge)
    left += [-1] * (2 - len(left))
    right = id_of(sibs[i + 1]) if i + 1 < len(sibs) else edge
    return left + [right]


def sample_mask(contents, children, rng):
    """Random partial tree: tokens to write per line.

    Returns n_toks[node]: how many of its tokens are written before the real point (0 = not started).
    Starting at the root, each existing node keeps a random subset of its children (any of them: siblings can
    start in any order), and each kept child a random non-empty prefix of its line.
    """
    n_toks, stack = [0] * len(contents), [0]
    while stack:
        p = stack.pop()
        for c in rng.sample(children[p], rng.randint(0, len(children[p]))):
            n_toks[c] = rng.randint(1, len(contents[c]))
            stack.append(c)
    return n_toks


def holes_mask(contents, parent, children, rng, deep=False):
    """Partial tree with holes: tokens to write per line (as sample_mask).

    The share of lines missing is uniform between 0 and 100% (at least one line). It is removed as holes: each a
    random line and up to 2 of its next siblings, with their subtrees, only as big as what is left to remove (so
    the share comes out exact). Every other line is complete, except that half the time one is cut to a random
    non-empty prefix (its <end> not written).

    deep=True: only lines deeper than d are removed or cut, d drawn from {1, 2} (top-level lines are depth 1), so the
    skeleton (imports, class and def headers, ...) is complete and only bodies have holes; the share is of those deep
    lines. A file with no line deeper than 2 tries d = 1; one with no line deeper than 1 gets plain holes.
    """
    N = len(contents)
    n_toks = [0] + [len(c) for c in contents[1:]]
    if N < 2:
        return n_toks
    size = [1] * N                                    # lines still present in each subtree
    for n in range(N - 1, 0, -1):                     # tree_arrays numbers each father before its children
        size[parent[n]] += size[n]
    eligible = [True] * N                             # lines that may be removed or cut
    if deep:
        depth = [0] * N
        for n in range(1, N):
            depth[n] = depth[parent[n]] + 1
        for d in (rng.choice([1, 2]), 1):
            if any(x > d for x in depth):
                eligible = [x > d for x in depth]
                break
    pool = sum(eligible[1:])
    budget = max(1, round(rng.random() * pool))      # lines still to remove
    while budget:
        n = rng.choice([m for m in range(1, N) if n_toks[m] and eligible[m] and size[m] <= budget])
        sibs = children[parent[n]]
        i, run = sibs.index(n), []
        for c in sibs[i:i + rng.randint(1, 3)]:
            if n_toks[c] and sum(size[r] for r in run) + size[c] <= budget:
                run.append(c)
        for c in run:
            s, stack, a = size[c], [c], parent[c]
            budget -= s
            while a >= 0:
                size[a] -= s
                a = parent[a]
            while stack:
                d = stack.pop()
                n_toks[d], size[d] = 0, 0
                stack.extend(children[d])
    kept = [n for n in range(1, N) if n_toks[n] > 1 and eligible[n]]
    if kept and rng.random() < 0.5:
        n = rng.choice(kept)
        n_toks[n] = rng.randint(1, n_toks[n] - 1)
    return n_toks


def flatten(arrays, vocab, rng, n_ids, order="random", max_anc=4, max_moves=4, mask="subset", holes_share=0.7,
            line_share=0.0, deep_share=0.0):
    """Build one training sequence with fresh random IDs.

    order="random": the mask, then random valid moves (lines interleave freely, more or less: see stickiness).
    order="dfs":    a fixed order: whole lines, depth-first, siblings in their real order. No real point.
    mask:           the kind of mask with order="random": "subset", "holes", "deep" or "mix" (see the top of the file).
    line_share:     with order="random", the chance that the sequence is line-first (see the top of the file).

    Besides the entries, returns per node its ID, father and first entry, and:
      moves:  (t, father node, node or -1 for a new line, token, father's last entry, line's last entry or -1,
              place, place's last entry or -1), the valid moves after entry t: the move actually taken plus up
              to max_moves-1 others, each once. place is, for a new line, the started sibling it goes right
              after (-1: before all of them); -2 when extending a line
      n_valid: per entry t, how many distinct valid moves there are after it (scored or not)
      end:    per node, the entry holding its <end> (after which the line is closed); len(entries) for the root
      stop_t: the entry after which STOP is valid (the last one)
      real_t: the real point (the end of the masked part), or -1
    """
    contents, parent, children = arrays
    N = len(contents)
    edge = n_ids + 1                                      # ID-code row for "past the first / last sibling"
    rank = [0] * N                                        # each node's real place among its siblings
    for kids in children:
        for k, c in enumerate(kids):
            rank[c] = k
    # Bookkeeping per node: written = tokens written so far; first / last = its first / latest entry index
    # (the latest entry's hidden state stands for the node in the heads); chain = its ID path;
    # placed = its started children, in their real order (= the order the insertions have built).
    ids = rng.sample(range(n_ids), N)                     # fresh random IDs per example
    chain = [[ids[0]]] + [None] * (N - 1)                 # node -> [self, father, grandfather, ...] IDs
    written, first, last, end = [0] * N, [0] + [-1] * (N - 1), [0] * N, [-1] * N
    placed = [[] for _ in range(N)]
    started, clock = [0] * N, 0                           # when each node was started (for dfs)
    front = list(children[0])                             # nodes whose next token can be written now
    if order != "random":
        mask = None
    else:
        if mask == "mix":                                 # one draw: deep, else holes, else subset
            r = rng.random()
            mask = "deep" if r < deep_share else "holes" if r < deep_share + holes_share else "subset"
        if mask in ("holes", "deep"):
            mask = holes_mask(contents, parent, children, rng, deep=mask == "deep")
        else:
            mask = sample_mask(contents, children, rng)
    stick, prev = rng.random(), -1                        # chance of continuing the line just written
    line_first = order == "random" and line_share > 0 and rng.random() < line_share
    real_t = -1

    def place_of(m):
        """The started sibling a new line m goes right after (-1: before all of them)."""
        sibs = placed[parent[m]]
        i = bisect.bisect_left([rank[s] for s in sibs], rank[m])
        return sibs[i - 1] if i else -1

    # Entry 0 is the root: its own ID, father "none" (n_ids), the <root> token. It gives the model a start.
    E = {"self_id": [ids[0]], "fa_id": [n_ids], "anc": [[]], "nb": [[-1, -1, -1]], "tok": [vocab.id(ROOT)],
         "pos": [0], "depth": [0]}
    moves, n_valid = [], []                               # n_valid[t]: distinct valid moves after entry t
    while front:                                          # each loop: record the targets after entry t, then write entry t+1
        t = len(E["tok"]) - 1
        if order == "dfs":   # finish the open line, else the first unstarted child of the latest started node
            n = max(front, key=lambda m: (written[m] > 0, started[parent[m]], -rank[m]))
        else:
            inside = [m for m in front if written[m] < mask[m]]   # moves that stay inside the mask
            if not inside and real_t < 0:                 # the mask is fully written: this step is the real point
                real_t = t
            pool = inside or front
            n = prev if prev in pool and rng.random() < stick else rng.choice(pool)
        targets = front
        if line_first and real_t >= 0:                    # past the real point: open lines first, if any
            targets = [m for m in front if written[m]] or front
            if n not in targets:
                n = prev if prev in targets and rng.random() < stick else rng.choice(targets)
        prev = n

        # Targets: every move in targets is correct here (all of front, except in line-first sequences). Keep the
        # one taken plus a few others (to save compute).
        # Two unstarted siblings with the same first token and the same place are the same move: keep it once.
        # All distinct valid moves are counted (n_valid), for the floor-adjusted loss in the log.
        seen, valid = set(), set()
        for m in [n] + rng.sample([m for m in targets if m != n], len(targets) - 1):
            p = parent[m]
            if written[m] == 0:
                pl = place_of(m)
                move = (t, p, -1, contents[m][0], last[p], -1, pl, last[pl] if pl >= 0 else -1)
            else:
                move = (t, p, m, contents[m][written[m]], last[p], last[m], -2, -1)
            key = move[1:4] + move[6:7]
            valid.add(key)
            if key not in seen and len(seen) < max_moves:
                seen.add(key)
                moves.append(move)
        n_valid.append(len(valid))

        p = parent[n]
        if written[n] == 0:                               # a new line: insert it among its siblings; its children open up
            sibs = placed[p]
            sibs.insert(bisect.bisect_left([rank[s] for s in sibs], rank[n]), n)
            chain[n] = [ids[n]] + chain[p]
            first[n], clock = t + 1, clock + 1
            started[n] = clock
            front.extend(children[n])
        E["self_id"].append(ids[n]); E["fa_id"].append(ids[p]); E["anc"].append(chain[n][2:][:max_anc])
        E["nb"].append(neighbour_ids(placed[p], placed[p].index(n), ids.__getitem__, edge))
        E["tok"].append(contents[n][written[n]]); E["pos"].append(written[n]); E["depth"].append(len(chain[n]) - 1)
        written[n] += 1
        last[n] = t + 1
        if written[n] == len(contents[n]):                # its <end> is written: the line is closed
            front.remove(n)
            end[n] = t + 1

    # front is empty: the whole tree is written, so STOP is the (only) valid move after the last entry
    end[0] = len(E["tok"])                                # the root is never closed
    n_valid.append(1)                                     # after the last entry only STOP is valid
    return dict(E, ids=ids, parent=parent, first=first, end=end, moves=moves, n_valid=n_valid,
                stop_t=len(E["tok"]) - 1, real_t=real_t)


def collate(items, prompts, tokenizer, max_anc):
    """Pad a list of flatten() outputs into tensors; moves get their batch index.

    Entries are [B,L] (anc [B,L,max_anc], nb [B,L,3], -1 = none); per-node tensors (node_ids, node_parent,
    node_first, node_end) are [B,N]; moves is [M,9] = (b, t, father node, line node or -1, token, father's last
    entry, line's last entry or -1, place (-1 first, -2 extending), place's last entry or -1).
    """
    B, L = len(items), max(len(it["tok"]) for it in items)
    N = max(len(it["ids"]) for it in items)
    t = lambda fill, n=L: torch.full((B, n), fill, dtype=torch.long)
    self_id, fa_id, tok, pos, depth, n_valid = t(0), t(0), t(0), t(0), t(0), t(1)
    anc = torch.full((B, L, max_anc), -1, dtype=torch.long)
    nb = torch.full((B, L, 3), -1, dtype=torch.long)
    pad = torch.ones(B, L, dtype=torch.bool)
    node_ids, node_parent, node_first, node_end = t(0, N), t(-1, N), t(L, N), t(L, N)  # padded nodes never exist
    moves, stop, real = [], [], []
    for b, it in enumerate(items):
        n, k = len(it["tok"]), len(it["ids"])
        self_id[b, :n] = torch.tensor(it["self_id"]); fa_id[b, :n] = torch.tensor(it["fa_id"])
        tok[b, :n] = torch.tensor(it["tok"]); pos[b, :n] = torch.tensor(it["pos"]); pad[b, :n] = False
        depth[b, :n] = torch.tensor(it["depth"]); nb[b, :n] = torch.tensor(it["nb"])
        n_valid[b, :n] = torch.tensor(it["n_valid"])
        for i, a in enumerate(it["anc"]):
            if a:
                anc[b, i, : len(a)] = torch.tensor(a)
        node_ids[b, :k] = torch.tensor(it["ids"]); node_parent[b, :k] = torch.tensor(it["parent"])
        node_first[b, :k] = torch.tensor(it["first"]); node_end[b, :k] = torch.tensor(it["end"])
        moves += [(b,) + m for m in it["moves"]]
        stop.append((b, it["stop_t"]))
        if it["real_t"] >= 0:
            real.append((b, it["real_t"]))
    T = lambda x, w: torch.tensor(x, dtype=torch.long).view(-1, w)
    p_ids, p_mask = tokenizer(prompts)
    return dict(self_id=self_id, fa_id=fa_id, anc=anc, nb=nb, tok=tok, pos=pos, depth=depth, pad=pad, n_valid=n_valid,
                node_ids=node_ids, node_parent=node_parent, node_first=node_first, node_end=node_end,
                moves=T(moves, 9), stop=T(stop, 2), real=T(real, 2), p_ids=p_ids, p_mask=p_mask)
