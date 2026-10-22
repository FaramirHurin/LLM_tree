"""Run with: python -m pytest tests   (or: python tests/test_treecodegen.py)"""
import ast
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treecodegen.sequences import flatten, tree_arrays
from treecodegen.toy import toy_example
from treecodegen.tree import canonical, code_to_tree, tree_to_code
from treecodegen.vocab import END, Vocab

VARIED = '''
@dec(1)
def g(a, *args, k=2, **kw):
    """doc
    more"""
    x = f"{a}-{k}"
    try:
        y = 1 / a
    except (ZeroDivisionError, ValueError) as e:
        y = 0
    else:
        y += 1
    finally:
        print("done")
    for i in range(3):
        if i == 1:
            continue
        elif i == 2:
            break
    else:
        pass
    while x:
        x = x[1:]
    with open("f") as fh, open("g") as gh:
        data = [l for l in fh]
    class C(Base):
        z: int = 3
    return lambda q: q ** 2
'''

MATCH = '''
def kind(cmd):
    match cmd.split():
        case ["go", direction] if direction in DIRS:
            return direction
        case [("quit" | "exit")]:
            raise SystemExit
        case _:
            pass
'''

DUPLICATES ="def f(a):\n    x = 1\n    x = 1\n    if a:\n        x = 1\n    return x"


def test_roundtrip():
    """code -> tree -> code gives the same program."""
    for src in (VARIED, MATCH):
        tree = code_to_tree(src)
        rebuilt = tree_to_code(tree)
        assert canonical(code_to_tree(rebuilt)) == canonical(tree)
        assert ast.unparse(ast.parse(rebuilt)) == ast.unparse(ast.parse(src))


def test_stdlib_roundtrip():
    """Every parseable standard-library file survives code -> tree -> code."""
    import glob, os, warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    files = glob.glob(os.path.join(os.path.dirname(ast.__file__), "*.py"))
    for p in files:
        src = open(p, encoding="utf-8", errors="replace").read()
        try:
            tree = code_to_tree(src)
        except (SyntaxError, ValueError):
            continue
        assert ast.unparse(ast.parse(tree_to_code(tree))) == ast.unparse(ast.parse(src)), p


def test_json_trees():
    """tree -> node list -> JSON -> tree is lossless, also through load_jsonl."""
    import json, os, tempfile
    from treecodegen.train import load_jsonl
    from treecodegen.tree import nodes_to_tree, tree_to_nodes
    for src in (VARIED, MATCH, DUPLICATES):
        tree = code_to_tree(src)
        back = nodes_to_tree(json.loads(json.dumps(tree_to_nodes(tree))))
        assert canonical(back) == canonical(tree)
        assert tree_to_code(back) == tree_to_code(tree)       # sibling order kept too
    path = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    with open(path, "w") as f:
        f.write(json.dumps({"prompt": "", "tree": tree_to_nodes(code_to_tree(MATCH))}) + "\n")
        f.write(json.dumps({"prompt": "", "code_tree": tree_to_nodes(code_to_tree(VARIED))}) + "\n")
        f.write(json.dumps({"prompt": "p", "code": DUPLICATES}) + "\n")
    rows = load_jsonl(path)
    assert [canonical(t) for _, t in rows] == [canonical(code_to_tree(s)) for s in (MATCH, VARIED, DUPLICATES)]


def _check_targets(arr, vocab, order, seed, mask="subset", line_share=0.0):
    """Replay a sequence: every entry is a valid move with the right neighbours, every recorded move is valid and
    recorded once, the insertions rebuild the real sibling order, STOP only at the end."""
    contents, parent, children = arr
    n_ids, EDGE = 64, 65
    x = flatten(arr, vocab, random.Random(seed), n_ids, order, max_moves=3, mask=mask, line_share=line_share)
    node = {i: n for n, i in enumerate(x["ids"])}
    written = [0] * len(contents)

    def place(n):   # a new line goes right after its nearest started older sibling, or first (-1)
        older = [c for c in children[parent[n]][: children[parent[n]].index(n)] if written[c]]
        return older[-1] if older else -1

    def valid():    # brute force: the next token of any line whose father exists (siblings in any order)
        out = set()
        for n in range(1, len(contents)):
            p = parent[n]
            if (p == 0 or written[p]) and written[n] < len(contents[n]):
                out.add((p, -1, contents[n][0], place(n)) if not written[n] else (p, n, contents[n][written[n]], -2))
        return out

    def neighbours(n):   # [left 1, left 2, right 1] among the started siblings, in real order
        sibs = [c for c in children[parent[n]] if written[c] or c == n]
        i, ids = sibs.index(n), x["ids"]
        l1 = ids[sibs[i - 1]] if i >= 1 else EDGE
        l2 = ids[sibs[i - 2]] if i >= 2 else EDGE if i == 1 else -1
        return [l1, l2, ids[sibs[i + 1]] if i + 1 < len(sibs) else EDGE]

    # every line ends with <end>, once; x["end"] is the entry holding it, after which the line is closed
    end_id = vocab.id(END)
    assert contents[0] == [] and all(c[-1] == end_id and end_id not in c[:-1] for c in contents[1:])
    for n in range(1, len(contents)):
        e = x["end"][n]
        assert x["tok"][e] == end_id and x["self_id"][e] == x["ids"][n]
    assert x["end"][0] == len(x["tok"])

    moves = {}
    for t, f, j, tok, fe, je, pl, ple in x["moves"]:
        m = (f, j, tok, pl)
        assert m not in moves.setdefault(t, set())        # each move recorded once
        moves[t].add(m)
        assert x["self_id"][fe] == x["ids"][f] and (j < 0 or x["self_id"][je] == x["ids"][j])
        assert fe <= t and je <= t and ple <= t
        assert j < 0 or x["end"][j] > t                   # a closed line is never extended
        assert (pl == -2) == (j >= 0) and (ple == -1 if pl < 0 else x["self_id"][ple] == x["ids"][pl])
    kids = [[] for _ in contents]                         # the sibling order the insertions build
    L = len(x["tok"])
    assert x["nb"][0] == [-1, -1, -1]
    assert len(x["n_valid"]) == L
    line_first = line_share == 1.0 and order == "random"
    for t in range(L):
        v = valid()
        if line_first and 0 <= x["real_t"] <= t and any(m[1] >= 0 for m in v):
            v = {m for m in v if m[1] >= 0}               # line-first, past the real point: open lines only
        assert moves.get(t, set()) <= v and x["n_valid"][t] == max(1, len(v))   # all distinct valid moves (STOP: 1)
        if t == L - 1:
            assert not v and x["stop_t"] == t
            break
        n = node[x["self_id"][t + 1]]
        p = parent[n]
        taken = (p, -1, x["tok"][t + 1], place(n)) if not written[n] else (p, n, x["tok"][t + 1], -2)
        assert taken in moves[t]
        if not written[n]:
            pl = taken[3]
            kids[p].insert(kids[p].index(pl) + 1 if pl >= 0 else 0, n)
        assert x["pos"][t + 1] == written[n] and x["fa_id"][t + 1] == x["ids"][p]
        assert x["nb"][t + 1] == neighbours(n)
        d, m = 0, n
        while m:
            d, m = d + 1, parent[m]
        assert x["depth"][t + 1] == d
        written[n] += 1
    assert all(w == len(c) for w, c in zip(written[1:], contents[1:]))
    assert kids == children                               # the insertions rebuild the real order
    assert (order == "dfs") == (x["real_t"] < 0) and x["real_t"] < L


def test_frontier_targets():
    r = random.Random(0)
    trees = [code_to_tree(toy_example(r)[1]) for _ in range(200)]
    trees += [code_to_tree(DUPLICATES), code_to_tree(VARIED)]
    vocab = Vocab.build(trees)
    for seed, tree in enumerate(trees):
        for order, mask, line in (("random", "subset", 0.0), ("random", "holes", 0.0), ("random", "mix", 0.0),
                                  ("random", "deep", 0.0), ("random", "subset", 1.0), ("random", "holes", 1.0),
                                  ("random", "deep", 1.0), ("dfs", "subset", 0.0)):
            _check_targets(tree_arrays(tree, vocab), vocab, order, seed, mask, line)


def test_holes_mask():
    """A holes mask is a valid partial tree: whole subtrees missing (at least one line), every other line complete
    except at most one cut short (never to nothing); the share of lines missing spreads over 0-100%."""
    from treecodegen.sequences import holes_mask
    r = random.Random(0)
    trees = [code_to_tree(toy_example(r)[1]) for _ in range(200)] + [code_to_tree(VARIED)]
    vocab = Vocab.build(trees)
    shares = []
    for seed, tree in enumerate(trees):
        contents, parent, children = tree_arrays(tree, vocab)
        assert all(parent[n] < n for n in range(1, len(contents)))   # holes_mask relies on this numbering
        m = holes_mask(contents, parent, children, random.Random(seed))
        lines = range(1, len(contents))
        shares.append(sum(m[n] == 0 for n in lines) / len(lines))
        assert any(m[n] == 0 for n in lines)
        assert all(m[n] == 0 or parent[n] == 0 or m[parent[n]] > 0 for n in lines)   # a started line's father is started
        assert all(m[c] == 0 for n in lines if m[n] == 0 for c in children[n])        # holes take whole subtrees
        assert sum(0 < m[n] < len(contents[n]) for n in lines) <= 1
    assert all(0.15 < sum(lo <= x < lo + 0.5 for x in shares) / len(shares) < 0.85 for lo in (0, 0.5))


def test_deep_mask():
    """A deep mask leaves the top one or two levels complete (holes and the cut line only below), in files that
    have deeper lines; otherwise it is a plain holes mask."""
    from treecodegen.sequences import holes_mask
    r = random.Random(0)
    trees = [code_to_tree(toy_example(r)[1]) for _ in range(200)] + [code_to_tree(VARIED), code_to_tree(DUPLICATES)]
    vocab = Vocab.build(trees)
    deep_ones = 0
    for seed, tree in enumerate(trees):
        contents, parent, children = tree_arrays(tree, vocab)
        depth = [0] * len(contents)
        for n in range(1, len(contents)):
            depth[n] = depth[parent[n]] + 1
        m = holes_mask(contents, parent, children, random.Random(seed), deep=True)
        lines = range(1, len(contents))
        assert any(m[n] == 0 for n in lines)
        assert all(m[n] == 0 or parent[n] == 0 or m[parent[n]] > 0 for n in lines)
        assert sum(0 < m[n] < len(contents[n]) for n in lines) <= 1
        if max(depth) > 1:                                  # some d in {1, 2}: every line down to depth d complete
            assert any(all(m[n] == len(contents[n]) for n in lines if depth[n] <= d) for d in (1, 2) if max(depth) > d)
            deep_ones += 1
    assert deep_ones > 50


def test_completion_eval():
    """The evaluation rebuilds the training prefix exactly, and needs one move per remaining entry plus STOP."""
    import torch
    from treecodegen.evaluate import complete, replay
    from treecodegen.model import build_models
    from treecodegen.vocab import PromptTokenizer
    trees = [code_to_tree(s) for s in (VARIED, DUPLICATES, MATCH)]
    vocab, tok = Vocab.build(trees), PromptTokenizer(texts=[""])
    cfg = dict(bert=None, enc_hidden=32, enc_layers=1, d_model=672, layers=1, heads=2, dropout=0.0,
               n_ids=64, k_slot=64, max_anc=4, max_pos=128, max_depth=16, k_depth=32)
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    for seed, tree in enumerate(trees):
        for mask in ("subset", "holes"):
            arr = tree_arrays(tree, vocab)
            x = flatten(arr, vocab, random.Random(seed), 64, "random", 4, mask=mask)
            upto = x["real_t"] if x["real_t"] >= 0 else len(x["tok"]) - 1
            state, _, _ = replay(x, arr, vocab, cfg, upto)
            for k in ("self_id", "fa_id", "anc", "nb", "tok", "pos", "depth"):
                assert state.E[k] == x[k][: upto + 1], k
            correct, required = complete(enc.eval(), dec.eval(), tok, vocab, cfg, "", tree, random.Random(seed),
                                         mask=mask)
            assert required == len(x["tok"]) - upto and 0 <= correct <= required


def test_attention_cache():
    """Scoring moves with the attention cache gives the same scores as rerunning the decoder on every entry,
    at every step of a completion (insertions included)."""
    import torch
    from treecodegen.evaluate import replay
    from treecodegen.generate import choose_move, score_moves
    from treecodegen.model import build_models
    from treecodegen.vocab import PromptTokenizer
    torch.manual_seed(0)
    tree = code_to_tree(VARIED)
    vocab, tok = Vocab.build([tree]), PromptTokenizer(texts=["add two numbers"])
    cfg = dict(bert=None, enc_hidden=32, enc_layers=1, d_model=672, layers=2, heads=4, dropout=0.1,
               n_ids=64, k_slot=64, max_anc=4, max_pos=128, max_depth=16, k_depth=32)
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    enc.eval(); dec.eval()
    arr = tree_arrays(tree, vocab)
    x = flatten(arr, vocab, random.Random(3), 64, "random", 4)
    state, _, _ = replay(x, arr, vocab, cfg, x["real_t"])
    ids, mask = tok(["add two numbers"])
    memory, mem_pad = enc(ids, mask), mask == 0
    rng, inserted = random.Random(0), 0
    for _ in range(40):
        c1, lp1, lt1, s1, _ = score_moves(dec, state, memory, mem_pad, vocab, "cpu", use_cache=True)
        c2, lp2, lt2, s2, _ = score_moves(dec, state, memory, mem_pad, vocab, "cpu", use_cache=False)
        assert c1 == c2 and torch.allclose(lp1, lp2, atol=1e-4) and torch.allclose(lt1, lt2, atol=1e-4)
        assert torch.allclose(s1, s2, atol=1e-4)
        k = rng.randrange(len(c1))                # any move, to cover insertions anywhere
        f, j, pl = c1[k]
        inserted += j == len(state.nodes) and pl >= 0
        state.apply(f, j, pl, int(lt1[k].argmax()), state.unused_id(rng) if j == len(state.nodes) else None)
    assert inserted > 0


def test_vocab_cap():
    """max_size keeps the most frequent tokens (specials included in the count); the rest become <unk>."""
    tree = code_to_tree(DUPLICATES)             # 'x' and '1' are the most frequent tokens
    full, capped = Vocab.build([tree]), Vocab.build([tree], max_size=6)
    assert len(capped) == 6 and capped.itos[:4] == full.itos[:4]
    assert {"x", "1"} <= set(capped.itos) and capped.id("def") == capped.id("<unk>")
    assert full.coverage == 1.0 and 0 < capped.coverage < 1
    assert len(Vocab.build([tree], max_size=10**6)) == len(full)


def test_ids_unique_and_redrawn():
    tree = code_to_tree(VARIED)
    vocab = Vocab.build([tree])
    arr = tree_arrays(tree, vocab)
    a = flatten(arr, vocab, random.Random(1), 512)
    b = flatten(arr, vocab, random.Random(2), 512)
    assert len(set(a["ids"])) == len(a["ids"])
    assert a["ids"] != b["ids"]


def test_train_step_and_generate():
    """One loss + backward on a small batch, then greedy generation, without errors."""
    import torch
    from treecodegen.generate import generate
    from treecodegen.loss import compute_loss
    from treecodegen.model import build_models
    from treecodegen.sequences import collate
    from treecodegen.vocab import PromptTokenizer
    r = random.Random(0)
    rows = [toy_example(r) for _ in range(8)]
    trees = [code_to_tree(c) for _, c in rows]
    vocab, tok = Vocab.build(trees), PromptTokenizer(texts=[p for p, _ in rows])
    cfg = dict(bert=None, enc_hidden=32, enc_layers=1, d_model=672, layers=1, heads=2, dropout=0.0,
               n_ids=64, k_slot=64, max_anc=4, max_pos=128, max_depth=16, k_depth=32)
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    items = [flatten(tree_arrays(t, vocab), vocab, r, 64) for t in trees]
    b = collate(items, [p for p, _ in rows], tok, 4)
    H = dec(b, enc(b["p_ids"], b["p_mask"]), b["p_mask"] == 0)
    loss, _, _ = compute_loss(dec, H, b, real_weight=2.0)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in dec.parameters() if p.grad is not None)
    generate(enc.eval(), dec.eval(), tok, vocab, rows[0][0], cfg, max_nodes=5, max_entries=40)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)


def test_eval_allows_unk():
    """The evaluation may write <unk> (the reference's token for a rare one); generation may not."""
    import torch
    from treecodegen.evaluate import replay
    from treecodegen.generate import score_moves
    from treecodegen.model import build_models
    from treecodegen.vocab import UNK, PromptTokenizer
    tree = code_to_tree(VARIED)
    vocab, tok = Vocab.build([tree]), PromptTokenizer(texts=[""])
    cfg = dict(bert=None, enc_hidden=32, enc_layers=1, d_model=672, layers=1, heads=2, dropout=0.0,
               n_ids=64, k_slot=64, max_anc=4, max_pos=128, max_depth=16, k_depth=32)
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    arr = tree_arrays(tree, vocab)
    x = flatten(arr, vocab, random.Random(0), 64, "random", 4)
    state, _, _ = replay(x, arr, vocab, cfg, x["real_t"])
    ids, mask = tok([""])
    memory = enc.eval()(ids, mask)
    for allow in (False, True):
        _, _, lt, _, _ = score_moves(dec.eval(), state, memory, mask == 0, vocab, "cpu", use_cache=False, allow_unk=allow)
        assert torch.isfinite(lt[:, vocab.id(UNK)]).any() == allow
