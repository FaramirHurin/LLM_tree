"""Build the code-only pretraining file from codeparrot-clean.

Streams the dataset, turns each Python file into a line-level tree and writes one row
per file to OUT: {"prompt": "", "code_tree": [...]}, with the node list from tree_to_nodes.
"""
import ast
import json
import os
import warnings
from treecodegen.tree import tree_to_nodes
from treecodegen.tree import code_to_tree

warnings.filterwarnings("ignore", category=SyntaxWarning)   # "\d" in non-raw strings: harmless

OUT = "Datasets/pretrain.jsonl"
MAX_ENTRIES = 2048        # size cap: all tokens plus one per node (so at most MAX_ENTRIES / 2 nodes)
MIN_NODES = 3             # skip empty or near-empty files (e.g. a bare __init__.py)
LIMIT = 1_000_000            # files to keep before stopping (1_000_000 for the full run)


def is_docstring(stmt):
    """True for a bare string-literal statement."""
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str))


def strip_docstrings(module):
    """Remove module, class and function docstrings in place.

    Each docstring would otherwise become one unique, useless vocabulary token.
    A function or class left with an empty body gets `pass` so it stays valid.
    """
    for n in ast.walk(module):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            if is_docstring(n.body[0]):
                rest = n.body[1:]
                n.body = rest if (rest or isinstance(n, ast.Module)) else [ast.Pass()]
    return module


def tree_size(tree):
    """(number of nodes, size) for a tree from code_to_tree; size = all tokens plus one per node.

    The virtual root is not counted.
    """
    nodes = entries = 0
    stack = list(tree.children)
    while stack:
        x = stack.pop()
        nodes += 1
        entries += len(x.content) + 1
        stack.extend(x.children)
    return nodes, entries


def convert(rows, out_path, limit):
    """Write up to `limit` usable files from `rows` to `out_path`, one tree per line."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    kept = seen = 0
    with open(out_path, "w") as f:
        for r in rows:
            seen += 1
            # Skip files that don't parse, use unsupported statements (code_to_tree raises
            # ValueError), or are too deeply nested / too large to process.
            try:
                code = ast.unparse(strip_docstrings(ast.parse(r["content"])))
                tree = code_to_tree(code)
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                continue
            nodes, entries = tree_size(tree)
            if nodes < MIN_NODES or entries > MAX_ENTRIES:
                continue
            f.write(json.dumps({"prompt": "", "code_tree": tree_to_nodes(tree)}) + "\n")
            kept += 1
            if kept % 10_000 == 0:
                print(f"kept {kept} of {seen} files", flush=True)
            if kept >= limit:
                break
    print(f"done: kept {kept} of {seen} files -> {out_path}")


if __name__ == "__main__":
    from datasets import load_dataset
    # Streaming: rows are fetched as they are read, nothing is downloaded up front.
    ds = load_dataset("codeparrot/codeparrot-clean", split="train", streaming=True)
    print(next(iter(ds)).keys())          # peek at the first row (ds[0] doesn't work when streaming)
    convert(ds, OUT, LIMIT)
