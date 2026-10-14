"""Command line: python -m treecodegen {toy-data, show, train, generate} ..."""
import argparse
import random

from .generate import generate
from .sequences import flatten, tree_arrays
from .toy import write_toy
from .train import load_checkpoint, pick_device, train
from .tree import code_to_tree, print_tree, tree_to_code
from .vocab import Vocab


def cmd_show(a):
    """Print the tree and the decoder entries for a piece of code."""
    code = a.code.encode().decode("unicode_escape") if "\\n" in a.code else a.code
    tree = code_to_tree(code)
    print_tree(tree)
    vocab = Vocab.build([tree])
    x = flatten(tree_arrays(tree, vocab), vocab, random.Random(a.seed), 64, "random")
    print(f"\n{'entry':>5}  {'self':>4} {'father':>6}  {'ancestors':<12} {'child #':<14} {'pos':>3}  token")
    for i in range(len(x["tok"])):
        fa = "-" if x["fa_id"][i] == 64 else x["fa_id"][i]
        mark = "   <- real point (end of the masked part)" if i == x["real_t"] else ""
        print(f"{i:5d}  {x['self_id'][i]:4d} {fa:>6}  {str(x['anc'][i]):<12} {str(x['rank'][i]):<14} {x['pos'][i]:3d}  "
              f"{vocab.itos[x['tok'][i]]!r}{mark}")
    print("\nrebuilt:\n" + tree_to_code(tree))


def cmd_generate(a):
    device = pick_device(a.device)
    enc, dec, tok, vocab, cfg = load_checkpoint(a.ckpt, device)
    tree = generate(enc, dec, tok, vocab, a.prompt, cfg, random.Random(a.seed), device=device)
    print_tree(tree)
    print("\n" + tree_to_code(tree))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="treecodegen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("toy-data", help="write a synthetic dataset")
    s.add_argument("--out", default="toy.jsonl"); s.add_argument("--n", type=int, default=6000)
    s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("show", help="print the tree and decoder entries for some code")
    s.add_argument("--code", required=True); s.add_argument("--seed", type=int, default=0)

    s = sub.add_parser("generate", help="generate code from a prompt with a trained checkpoint")
    s.add_argument("--ckpt", required=True); s.add_argument("--prompt", required=True)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None, help="cpu, cuda or mps; default: the best available")

    s = sub.add_parser("train", help="train on a JSONL file of {prompt, code}")
    s.add_argument("--data", required=True); s.add_argument("--out", default="model.pt")
    s.add_argument("--bert", default=None, help="pretrained BERT name or path; omit to train a small encoder from scratch")
    s.add_argument("--enc-hidden", type=int, default=128); s.add_argument("--enc-layers", type=int, default=2)
    s.add_argument("--d-model", type=int, default=576); s.add_argument("--layers", type=int, default=4)
    s.add_argument("--heads", type=int, default=4); s.add_argument("--dropout", type=float, default=0.1)
    s.add_argument("--n-ids", type=int, default=512, help="size of the random ID pool (>= max nodes per tree)")
    s.add_argument("--k-slot", type=int, default=64, help="size of an ID code; the path takes (2 + max-anc)x this")
    s.add_argument("--max-anc", type=int, default=4, help="ancestor levels kept, nearest first, one slot each"); s.add_argument("--max-pos", type=int, default=128)
    s.add_argument("--order", choices=["random", "dfs"], default="random",
                   help="random: a masked partial tree, then random valid moves; dfs: whole lines, depth-first")
    s.add_argument("--max-moves", type=int, default=4, help="valid moves scored per step (the one taken + others)")
    s.add_argument("--real-weight", type=float, default=1.0, help="loss weight of the real point of each sequence")
    s.add_argument("--min-freq", type=int, default=1)
    s.add_argument("--steps", type=int, default=3000); s.add_argument("--batch", type=int, default=32)
    s.add_argument("--lr", type=float, default=5e-4); s.add_argument("--enc-lr", type=float, default=None)
    s.add_argument("--warmup", type=int, default=200)
    s.add_argument("--freeze-encoder-steps", type=int, default=0)
    s.add_argument("--init", default=None, help="checkpoint to take the decoder (and its vocabulary) from, e.g. a pretrained one")
    s.add_argument("--val-frac", type=float, default=0.05)
    s.add_argument("--log-every", type=int, default=100); s.add_argument("--eval-every", type=int, default=1000)
    s.add_argument("--eval-n", type=int, default=100, help="held-out rows to generate at each evaluation; 0 = none"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--device", default=None, help="cpu, cuda or mps; default: the best available")

    a = ap.parse_args(argv)
    if a.cmd == "toy-data":
        write_toy(a.out, a.n, a.seed)
    elif a.cmd == "show":
        cmd_show(a)
    elif a.cmd == "train":
        train(a)
    else:
        cmd_generate(a)
