"""Held-out evaluation that matches training: complete a partial tree, plus the loss on held-out files.

Completion: a held-out tree gets a random mask written in a random valid order, exactly as in training, up to the
real point. From there the model makes every remaining move. A move is correct if it is one of the valid moves at
that point (any open line's next token, any missing line at its place, or STOP once the tree is complete). After a
wrong move, the valid move the model scores highest is applied instead, so every required move is judged. The number
of required moves is fixed: the tokens still to write (each line's <end> included) plus one STOP.
Score = correct moves / required moves. The model may write <unk>: a rare token is <unk> in the reference too, so
that is the right move there. The mask kind (mask=, see sequences.py) sets the partial trees: "subset"
gives small ones, "holes" whole lines with blocks missing (0-100% of the lines).
"""
import json
import random

import torch

from .generate import MAX_MOVES, TreeState, choose_move, score_moves
from .loss import compute_loss
from .sequences import collate, flatten, tree_arrays
from .tree import code_to_tree, nodes_to_tree, tree_to_code


def fits(arrays, cfg):
    """The training size filters: no line longer than max_pos (<end> included), at most n_ids nodes."""
    contents = arrays[0]
    return max(len(c) for c in contents) <= cfg["max_pos"] and len(contents) <= cfg["n_ids"]


def replay(x, arrays, vocab, cfg, upto):
    """The partial tree after entries 0..upto of a flatten() output, rebuilt with TreeState.apply.

    Returns (state, gi, written): gi[true node] = its index in state.nodes (-1 if not started), written[true node] =
    its tokens written so far.
    """
    contents, parent, children = arrays
    node = {i: n for n, i in enumerate(x["ids"])}
    state = TreeState(vocab, cfg, x["ids"][0])
    gi, written = [0] + [-1] * (len(contents) - 1), [0] * len(contents)
    for e in range(1, upto + 1):
        n = node[x["self_id"][e]]
        p = parent[n]
        if written[n]:
            state.apply(gi[p], gi[n], -2, x["tok"][e])
        else:
            pl = _place(n, parent, children, written)
            gi[n] = len(state.nodes)
            state.apply(gi[p], gi[n], gi[pl] if pl >= 0 else -1, x["tok"][e], x["ids"][n])
        written[n] += 1
    return state, gi, written


def _place(n, parent, children, written):
    """The started older sibling a new line n goes right after (-1: first)."""
    older = [c for c in children[parent[n]][: children[parent[n]].index(n)] if written[c]]
    return older[-1] if older else -1


@torch.no_grad()
def complete(enc, dec, tok, vocab, cfg, prompt, tree, rng, device="cpu", temperature=0.0, gen=None, mask="subset"):
    """Complete one held-out tree from a training-style partial tree. Returns (correct moves, required moves)."""
    arrays = tree_arrays(tree, vocab)
    contents, parent, children = arrays
    x = flatten(arrays, vocab, rng, cfg["n_ids"], "random", cfg["max_anc"], mask=mask)
    upto = x["real_t"] if x["real_t"] >= 0 else len(x["tok"]) - 1   # -1: the mask is the whole tree
    state, gi, written = replay(x, arrays, vocab, cfg, upto)
    ids, mask = tok([prompt])
    memory, mem_pad = enc(ids.to(device), mask.to(device)), (mask == 0).to(device)
    correct = required = 0
    while True:
        new_j = len(state.nodes)
        valid = {}                                # (father, line or new_j, place, token) -> true node
        for n in range(1, len(contents)):
            p = parent[n]
            if gi[p] < 0 or written[n] == len(contents[n]):
                continue
            if written[n]:
                valid[(gi[p], gi[n], -2, contents[n][written[n]])] = n
            else:                                 # identical starts (same token, same place) are one move
                pl = _place(n, parent, children, written)
                valid.setdefault((gi[p], new_j, gi[pl] if pl >= 0 else -1, contents[n][0]), n)
        cand, cand_lp, lt, stop_lp, can_stop = score_moves(dec, state, memory, mem_pad, vocab, device, allow_unk=True)
        move = choose_move(cand_lp, lt, stop_lp, can_stop, temperature, gen)
        required += 1
        if not valid:                             # the tree is complete: STOP is the required move
            correct += move is None
            return correct, required
        chosen = None if move is None else cand[move[0]] + (move[1],)
        if chosen in valid:
            correct += 1
        else:                                     # wrong: apply the valid move the model scores highest
            index = {c: k for k, c in enumerate(cand)}
            chosen = max(valid, key=lambda m: float(cand_lp[index[m[:3]]] + lt[index[m[:3]], m[3]]))
        f, j, pl, t = chosen
        n = valid[chosen]
        state.apply(f, j, pl, t, x["ids"][n] if j == new_j else None)
        if j == new_j:
            gi[n] = j
        written[n] += 1


@torch.no_grad()
def val_loss(enc, dec, tok, vocab, cfg, rows, device="cpu", batch=8, seed=0, max_moves=4, mask="subset"):
    """The training loss (random orders, random masks, seeded) averaged over held-out (prompt, tree) rows,
    and its floor-adjusted version (see compute_loss)."""
    rng = random.Random(seed)
    rows = [(p, a) for p, a in ((p, tree_arrays(t, vocab)) for p, t in rows) if fits(a, cfg)]
    total = total_norm = n = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        items = [flatten(a, vocab, rng, cfg["n_ids"], "random", cfg["max_anc"], max_moves, mask) for _, a in chunk]
        b = {k: v.to(device) for k, v in collate(items, [p for p, _ in chunk], tok, cfg["max_anc"]).items()}
        H = dec(b, enc(b["p_ids"], b["p_mask"]), b["p_mask"] == 0)
        loss, _, norm = compute_loss(dec, H, b)
        total, total_norm, n = total + loss.item() * len(chunk), total_norm + norm * len(chunk), n + len(chunk)
    return total / max(1, n), total_norm / max(1, n)


@torch.no_grad()
def free_completion(enc, dec, tok, vocab, cfg, prompt, tree, rng, device="cpu", temperature=0.0,
                    max_moves=MAX_MOVES, mask="subset"):
    """An example: the training-style partial tree of a held-out tree, completed by the model alone (no corrections),
    for at most max_moves moves (<unk> allowed, as in complete()).

    Returns (partial code, open lines marked with "…"; the model's completion; the reference code).
    """
    arrays = tree_arrays(tree, vocab)
    x = flatten(arrays, vocab, rng, cfg["n_ids"], "random", cfg["max_anc"], mask=mask)
    upto = x["real_t"] if x["real_t"] >= 0 else len(x["tok"]) - 1
    state, _, _ = replay(x, arrays, vocab, cfg, upto)
    shown = state.tree(vocab)
    stack = [(shown, state.nodes[0])]             # mark the lines still open in the partial tree
    while stack:
        t, n = stack.pop()
        if n["father"] >= 0 and not n["closed"]:
            t.content = t.content + ["…"]
        stack.extend(zip(t.children, (state.nodes[k] for k in n["kids"])))
    partial = tree_to_code(shown)
    ids, mask = tok([prompt])
    memory, mem_pad = enc(ids.to(device), mask.to(device)), (mask == 0).to(device)
    gen = torch.Generator(device).manual_seed(rng.randrange(2 ** 31))
    for _ in range(max_moves):
        cand, cand_lp, lt, stop_lp, can_stop = score_moves(dec, state, memory, mem_pad, vocab, device,
                                                           max_nodes=cfg["n_ids"] - 1, max_line=cfg["max_pos"],
                                                           allow_unk=True)
        move = choose_move(cand_lp, lt, stop_lp, can_stop, temperature, gen)
        if move is None:
            break
        k, t = move
        f, j, pl = cand[k]
        state.apply(f, j, pl, t, state.unused_id(rng) if j == len(state.nodes) else None)
    return partial, tree_to_code(state.tree(vocab)), tree_to_code(tree)


def evaluate(enc, dec, tok, vocab, cfg, rows, device="cpu", temperature=0.0, seed=1, max_entries=None,
             progress=False, mask="subset"):
    """Completion score and validation loss on held-out (prompt, tree) rows (those passing the training size filters).

    Returns dict(moves=correct / required moves, files=share completed with no wrong move, loss=validation loss,
    loss_norm=its floor-adjusted version, n=trees used). max_entries skips bigger trees (each step reruns the decoder on all entries, so big trees are slow).
    """
    enc.eval(); dec.eval()
    rng = random.Random(seed)
    gen = torch.Generator(device).manual_seed(seed)
    used, correct, required, perfect = [], 0, 0, 0
    for prompt, tree in rows:
        arrays = tree_arrays(tree, vocab)
        if not fits(arrays, cfg) or (max_entries and sum(len(c) for c in arrays[0]) + 1 > max_entries):
            continue
        c, r = complete(enc, dec, tok, vocab, cfg, prompt, tree, rng, device, temperature, gen, mask)
        correct, required, perfect = correct + c, required + r, perfect + (c == r)
        used.append((prompt, tree))
        if progress:
            print(f"  file {len(used)}: {c}/{r} moves correct (running total {correct / required:.1%})", flush=True)
    loss, loss_norm = val_loss(enc, dec, tok, vocab, cfg, used, device, mask=mask)
    enc.train(); dec.train()
    return dict(moves=correct / max(1, required), files=perfect / max(1, len(used)), loss=loss, loss_norm=loss_norm,
                n=len(used))


def heldout_rows(path, n, seed=0, val_frac=0.05):
    """The first n held-out (prompt, tree) rows of a JSONL file, as train.py splits it (shuffle with `seed`, first
    val_frac).

    Only those n lines are parsed. Assumes every line loads (true for files written by load_datasets.py), so the
    shuffle is the same permutation train.py applies.
    """
    with open(path) as f:
        n_lines = sum(1 for _ in f)
    order = list(range(n_lines))
    random.Random(seed).shuffle(order)
    want = order[: max(1, int(n_lines * val_frac))][:n]
    pos, rows = {w: k for k, w in enumerate(want)}, [None] * len(want)
    with open(path) as f:
        for i, line in enumerate(f):
            if i in pos:
                r = json.loads(line)
                stored = r.get("tree", r.get("code_tree"))
                rows[pos[i]] = (r.get("prompt", ""), nodes_to_tree(stored) if stored is not None else code_to_tree(r["code"]))
    return rows
