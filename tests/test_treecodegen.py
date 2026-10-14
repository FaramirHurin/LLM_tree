"""Run with: python -m pytest tests   (or: python tests/test_treecodegen.py)"""
import ast
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treecodegen.sequences import flatten, tree_arrays
from treecodegen.toy import toy_example
from treecodegen.tree import canonical, code_to_tree, tree_to_code
from treecodegen.vocab import Vocab

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


def _check_targets(arr, vocab, order, seed):
    """Replay a sequence: every entry is a valid move, every recorded move is valid, STOP only at the end."""
    contents, parent, children = arr
    x = flatten(arr, vocab, random.Random(seed), 64, order, max_moves=3)
    node = {i: n for n, i in enumerate(x["ids"])}
    written = [0] * len(contents)

    def valid():   # brute force: the next token of any line whose father exists and whose older siblings exist
        out = set()
        for n in range(1, len(contents)):
            p, k = parent[n], children[parent[n]].index(n)
            if (p == 0 or written[p]) and (k == 0 or written[children[p][k - 1]]) and written[n] < len(contents[n]):
                out.add((p, n if written[n] else -1, contents[n][written[n]]))
        return out

    moves = {}
    for t, f, j, tok, fe, je in x["moves"]:
        moves.setdefault(t, set()).add((f, j, tok))
        assert x["self_id"][fe] == x["ids"][f] and (j < 0 or x["self_id"][je] == x["ids"][j])
        assert fe <= t and je <= t
    L = len(x["tok"])
    for t in range(L):
        v = valid()
        assert moves.get(t, set()) <= v
        if t == L - 1:
            assert not v and x["stop_t"] == t
            break
        n = node[x["self_id"][t + 1]]
        assert (parent[n], n if written[n] else -1, x["tok"][t + 1]) in moves[t]
        assert x["pos"][t + 1] == written[n] and x["fa_id"][t + 1] == x["ids"][parent[n]]
        assert x["rank"][t + 1][:2] == [children[parent[n]].index(n),
                                        children[parent[parent[n]]].index(parent[n]) if parent[n] else -1]
        written[n] += 1
    assert all(w == len(c) for w, c in zip(written[1:], contents[1:]))
    assert (order == "dfs") == (x["real_t"] < 0) and x["real_t"] < L


def test_frontier_targets():
    r = random.Random(0)
    trees = [code_to_tree(toy_example(r)[1]) for _ in range(200)]
    trees += [code_to_tree(DUPLICATES), code_to_tree(VARIED)]
    vocab = Vocab.build(trees)
    for seed, tree in enumerate(trees):
        for order in ("random", "dfs"):
            _check_targets(tree_arrays(tree, vocab), vocab, order, seed)


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
    cfg = dict(bert=None, enc_hidden=32, enc_layers=1, d_model=448, layers=1, heads=2, dropout=0.0,
               n_ids=64, k_slot=64, max_anc=4, max_pos=128)
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    items = [flatten(tree_arrays(t, vocab), vocab, r, 64) for t in trees]
    b = collate(items, [p for p, _ in rows], tok, 4)
    H = dec(b, enc(b["p_ids"], b["p_mask"]), b["p_mask"] == 0)
    loss, _ = compute_loss(dec, H, b, real_weight=2.0)
    loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in dec.parameters() if p.grad is not None)
    generate(enc.eval(), dec.eval(), tok, vocab, rows[0][0], cfg, max_nodes=5, max_entries=40)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok ", name)
