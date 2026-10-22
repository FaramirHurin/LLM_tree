"""Command line: python -m treecodegen {toy-data, show, train, generate, evaluate} ..."""
import argparse
import datetime
import random
import re

import torch

from .evaluate import evaluate, fits, free_completion, heldout_rows
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
    print(f"\n{'entry':>5}  {'self':>4} {'father':>6}  {'ancestors':<12} {'L1 L2 R1':<14} {'pos':>3} {'depth':>5}  token")
    for i in range(len(x["tok"])):
        fa = "-" if x["fa_id"][i] == 64 else x["fa_id"][i]
        mark = "   <- real point (end of the masked part)" if i == x["real_t"] else ""
        nb = " ".join("|" if v == 65 else "-" if v < 0 else str(v) for v in x["nb"][i])   # | = edge, - = none
        print(f"{i:5d}  {x['self_id'][i]:4d} {fa:>6}  {str(x['anc'][i]):<12} {nb:<14} {x['pos'][i]:3d} {x['depth'][i]:5d}  "
              f"{vocab.itos[x['tok'][i]]!r}{mark}")
    print("\nrebuilt:\n" + tree_to_code(tree))


def cmd_generate(a):
    device = pick_device(a.device)
    enc, dec, tok, vocab, cfg = load_checkpoint(a.ckpt, device)
    tree = generate(enc, dec, tok, vocab, a.prompt, cfg, random.Random(a.seed), device=device,
                    temperature=a.temperature)
    print_tree(tree)
    print("\n" + tree_to_code(tree))


def cmd_evaluate(a):
    """Completion score and validation loss on held-out files, how far training has got, and examples."""
    out = open(a.report, "w") if a.report else None

    def say(text=""):
        print(text, flush=True)
        if out:
            out.write(text + "\n"); out.flush()

    device = pick_device(a.device)
    ck = torch.load(a.ckpt, map_location="cpu")
    enc, dec, tok, vocab, cfg = load_checkpoint(a.ckpt, device)
    step, batch, n_train = ck.get("step"), ck.get("batch", a.batch), ck.get("n_train", a.train_files)
    say(f"Evaluation of {a.ckpt} ({datetime.datetime.now():%Y-%m-%d %H:%M})")
    say(f"  step {step}" + (f" of {ck['steps']}" if ck.get("steps") else ""))
    if ck.get("resumed_from"):
        say(f"  resumed from {ck['resumed_from']}; trained with {cfg.get('mask', 'subset')} masks"
            + (f" (holes share {cfg.get('holes_share')}, deep share {cfg.get('deep_share', 0)})"
               if cfg.get("mask") == "mix" else "")
            + (f", {cfg['line_share']:.0%} line-first" if cfg.get("line_share") else ""))
    if step and batch and n_train:
        seen = ck.get("prior_sequences", 0) + step * batch
        say(f"  {seen:,} training sequences seen (batch {batch}"
            + (f", {ck['prior_sequences']:,} of them before resuming" if ck.get("prior_sequences") else "")
            + f") out of {n_train:,} training files: {seen / n_train:.2f} epochs")
    else:
        say("  epochs: unknown (pass --batch and --train-files for checkpoints that don't record them)")
    if a.train_log and step:                      # the training loss just before this checkpoint, to compare
        losses = [float(m.group(2)) for m in (re.match(r"step\s+(\d+)\s+loss ([\d.]+)", l) for l in open(a.train_log))
                  if m and step - 1000 < int(m.group(1)) <= step]
        if losses:
            say(f"  training loss over the last 1000 steps before it: {sum(losses) / len(losses):.3f}")
    rows = heldout_rows(a.data, a.n, a.seed, a.val_frac)
    say(f"\nCompleting {len(rows)} held-out files (skipping those over {a.max_entries} entries)"
        if a.max_entries else f"\nCompleting {len(rows)} held-out files")
    r = evaluate(enc, dec, tok, vocab, cfg, rows, device, a.temperature, max_entries=a.max_entries, progress=True,
                 mask=a.mask)
    say(f"\nResult on {r['n']} held-out files ({a.mask} masks):")
    say(f"  moves correct:            {r['moves']:.1%}  (correct moves / moves required, STOP included)")
    say(f"  files with no wrong move: {r['files']:.1%}")
    say(f"  validation loss:          {r['loss']:.3f}  (the training loss on these held-out files)")
    say(f"  floor-adjusted:           {r['loss_norm']:.3f}  (minus the floor from scoring only some valid moves; "
        f"a perfect model gets about 0)")

    # one example: a small held-out file, its partial tree, and the model's own completion (no corrections)
    rng = random.Random(a.seed)
    small = [(p, t) for p, t in rows if fits(tree_arrays(t, vocab), cfg) and len(tree_to_code(t).split("\n")) <= 30]
    if small:
        prompt, tree = small[0]
        partial, done, ref = free_completion(enc, dec, tok, vocab, cfg, prompt, tree, rng, device, a.temperature,
                                             mask=a.mask)
        indent = lambda code: "\n".join("    | " + l for l in code.split("\n"))
        say("\nExample completion (a held-out file; open lines end in …):")
        say("  partial tree given to the model:\n" + indent(partial))
        say(f"  the model's completion (temperature {a.temperature}, no corrections):\n" + indent(done))
        say("  the real file:\n" + indent(ref))
    sample = tree_to_code(generate(enc, dec, tok, vocab, "", cfg, random.Random(a.seed), device=device,
                                   temperature=a.sample_temperature))
    say(f"\nFree sample from an empty tree (temperature {a.sample_temperature}):\n"
        + "\n".join("    | " + l for l in sample.split("\n")))
    if out:
        out.close()


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
    s.add_argument("--temperature", type=float, default=0.2, help="sampling temperature; 0 = greedy")
    s.add_argument("--device", default=None, help="cpu, cuda or mps; default: the best available")

    s = sub.add_parser("evaluate", help="completion score and validation loss of a checkpoint on held-out files")
    s.add_argument("--ckpt", required=True)
    s.add_argument("--data", required=True, help="the JSONL file the checkpoint was trained on")
    s.add_argument("--n", type=int, default=100, help="held-out files to use (the first n of the validation split)")
    s.add_argument("--seed", type=int, default=0, help="the training run's --seed (it sets the split)")
    s.add_argument("--val-frac", type=float, default=0.05, help="the training run's --val-frac")
    s.add_argument("--temperature", type=float, default=0.0, help="how the model picks its move; 0 = greedy")
    s.add_argument("--max-entries", type=int, default=None, help="skip bigger files (generation reruns the "
                   "decoder on all entries at every step, so big files are slow, especially on CPU)")
    s.add_argument("--report", default=None, help="also write the report to this file")
    s.add_argument("--train-log", default=None, help="the run's log, to show the training loss at that step")
    s.add_argument("--batch", type=int, default=None, help="the run's batch, if the checkpoint doesn't record it")
    s.add_argument("--train-files", type=int, default=None,
                   help="the run's number of training files, if the checkpoint doesn't record it")
    s.add_argument("--sample-temperature", type=float, default=0.2, help="temperature of the free sample")
    s.add_argument("--mask", choices=["subset", "holes", "deep"], default="subset",
                   help="partial trees to complete: subset (small trees, as in pass 1), holes (whole lines with blocks missing, "
                        "0-100%% of the lines) or deep (holes only inside bodies: the top one or two levels are complete)")
    s.add_argument("--device", default=None, help="cpu, cuda or mps; default: the best available")

    s = sub.add_parser("train", help="train on a JSONL file of {prompt, code}")
    s.add_argument("--data", required=True); s.add_argument("--out", default="model.pt")
    s.add_argument("--bert", default=None, help="pretrained BERT name or path; omit to train a small encoder from scratch")
    s.add_argument("--enc-hidden", type=int, default=128); s.add_argument("--enc-layers", type=int, default=2)
    s.add_argument("--d-model", type=int, default=896); s.add_argument("--layers", type=int, default=4)
    s.add_argument("--heads", type=int, default=4); s.add_argument("--dropout", type=float, default=0.1)
    s.add_argument("--n-ids", type=int, default=512, help="size of the random ID pool (>= max nodes per tree)")
    s.add_argument("--k-slot", type=int, default=64, help="size of an ID code; the path takes (2 + max-anc + 3)x this + k-depth")
    s.add_argument("--max-anc", type=int, default=4, help="ancestor levels kept, nearest first, one slot each"); s.add_argument("--max-pos", type=int, default=128)
    s.add_argument("--max-depth", type=int, default=16, help="depth embedding rows; deeper nodes share the last one")
    s.add_argument("--k-depth", type=int, default=32, help="size of the depth slot in the path")
    s.add_argument("--order", choices=["random", "dfs"], default="random",
                   help="random: a masked partial tree, then random valid moves; dfs: whole lines, depth-first")
    s.add_argument("--max-moves", type=int, default=4, help="valid moves scored per step (the one taken + others)")
    s.add_argument("--real-weight", type=float, default=1.0, help="loss weight of the real point of each sequence")
    s.add_argument("--min-freq", type=int, default=1)
    s.add_argument("--max-vocab", type=int, default=None,
                   help="keep only the most frequent code tokens, up to this vocabulary size; the rest become <unk>")
    s.add_argument("--steps", type=int, default=3000); s.add_argument("--batch", type=int, default=32)
    s.add_argument("--accum", type=int, default=1, help="batches added up into one step (gradient accumulation): "
                   "acts like a batch this many times bigger, with the memory of one batch")
    s.add_argument("--lr", type=float, default=5e-4); s.add_argument("--enc-lr", type=float, default=None)
    s.add_argument("--warmup", type=int, default=200)
    s.add_argument("--freeze-encoder-steps", type=int, default=0)
    s.add_argument("--vocab-from", default=None,
                   help="checkpoint to take only the code vocabulary from (not its weights), e.g. a previous run's")
    s.add_argument("--init", default=None, help="checkpoint to take the decoder (and its vocabulary) from, e.g. a pretrained one")
    s.add_argument("--resume", default=None,
                   help="checkpoint to continue: encoder, decoder, vocabulary and shapes (a new optimizer and schedule)")
    s.add_argument("--mask", choices=["subset", "holes", "deep", "mix"], default="subset",
                   help="partial trees at the start of each sequence: subset (small), holes (whole lines with blocks "
                        "missing) or mix (holes with probability --holes-share)")
    s.add_argument("--holes-share", type=float, default=0.7, help="with --mask mix: share of sequences with holes masks")
    s.add_argument("--deep-share", type=float, default=0.0,
                   help="with --mask mix: share of sequences with deep masks (holes only inside bodies); the rest of "
                        "the mix is holes (--holes-share) and subset")
    s.add_argument("--line-share", type=float, default=0.0,
                   help="share of line-first sequences: after the mask, while a line is open only continuing a line is "
                        "a valid move (a new line starts only once every started line is finished)")
    s.add_argument("--val-frac", type=float, default=0.05)
    s.add_argument("--log-every", type=int, default=100); s.add_argument("--eval-every", type=int, default=1000, help="steps between evaluations and checkpoint saves")
    s.add_argument("--eval-n", type=int, default=100, help="held-out rows to complete at each evaluation; 0 = none"); s.add_argument("--seed", type=int, default=0)
    s.add_argument("--temperature", type=float, default=0.2, help="sampling temperature of the free sample shown at each evaluation; 0 = greedy")
    s.add_argument("--device", default=None, help="cpu, cuda or mps; default: the best available")

    a = ap.parse_args(argv)
    if a.cmd == "toy-data":
        write_toy(a.out, a.n, a.seed)
    elif a.cmd == "show":
        cmd_show(a)
    elif a.cmd == "train":
        train(a)
    elif a.cmd == "evaluate":
        cmd_evaluate(a)
    else:
        cmd_generate(a)
