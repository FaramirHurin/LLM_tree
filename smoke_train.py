"""Smoke test: a few training steps on a tiny slice of pretrain.jsonl, on CPU, with a memory check first.

    python smoke_train.py --data Datasets/pretrain.jsonl

Reads only the first --rows lines, keeps small trees, and aborts before the forward pass if the batch's
estimated activation memory is over --budget-gb.
"""
import argparse
import itertools
import json
import random
import resource
import sys
import time

import torch

from treecodegen.loss import compute_loss
from treecodegen.model import build_models
from treecodegen.sequences import collate, flatten, tree_arrays
from treecodegen.tree import nodes_to_tree
from treecodegen.vocab import PromptTokenizer, Vocab


def estimate_bytes(b, cfg, vocab_size):
    """Rough peak of the tensors kept for backward (float32), from the batch's real shapes."""
    B, L = b["tok"].shape
    N, M = b["node_ids"].shape[1], len(b["moves"])
    tok_logits = M * vocab_size * 3                          # logits, log_softmax, grad
    attention = cfg["layers"] * cfg["heads"] * B * L * L * 3  # scores, softmax, dropout, per layer
    father = B * L * (N + 1) * 3
    acts = cfg["layers"] * B * L * cfg["d_model"] * 20       # projections, FFN (4x), norms, residuals
    return 4 * (tok_logits + attention + father + acts), dict(B=B, L=L, N=N, M=M)


def peak_rss_gb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20   # bytes on macOS, KB on Linux


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="Datasets/pretrain.jsonl")
    ap.add_argument("--rows", type=int, default=20, help="lines read from the top of the file")
    ap.add_argument("--max-nodes", type=int, default=64, help="drop trees with more nodes (also the ID pool size)")
    ap.add_argument("--max-tokens", type=int, default=512, help="drop trees with more tokens (= sequence length)")
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--budget-gb", type=float, default=1.5, help="abort if a batch is estimated above this")
    a = ap.parse_args()
    torch.manual_seed(0); rng = random.Random(0)
    torch.set_num_threads(2)                                 # keep the laptop responsive

    cfg = dict(bert=None, enc_hidden=64, enc_layers=1, d_model=192, layers=2, heads=2, dropout=0.1,
               n_ids=a.max_nodes, k_slot=16, max_anc=2, max_pos=128, max_depth=16, k_depth=16, order="random")

    trees = []
    with open(a.data) as f:
        for line in itertools.islice(f, a.rows):             # never reads past --rows lines
            nodes = json.loads(line)["code_tree"]
            n_tok = sum(len(n["content"]) for n in nodes)
            longest = max(len(n["content"]) for n in nodes)
            # entries: the root, then every token plus one <end> per line; a line's length includes its <end>
            if len(nodes) <= a.max_nodes and n_tok + len(nodes) <= a.max_tokens and longest + 1 <= cfg["max_pos"]:
                trees.append(nodes_to_tree(nodes))
    print(f"kept {len(trees)} of {a.rows} rows")
    if len(trees) < a.batch:
        sys.exit("not enough small trees; raise --rows or the caps")

    vocab = Vocab.build(trees)
    tok = PromptTokenizer(None, texts=[""])
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    print(f"code vocab {len(vocab)}  params enc {sum(p.numel() for p in enc.parameters())/1e6:.2f}M "
          f"dec {sum(p.numel() for p in dec.parameters())/1e6:.2f}M")
    arrays = [tree_arrays(t, vocab) for t in trees]
    opt = torch.optim.AdamW(dec.parameters(), lr=5e-4)       # prompts are empty: the encoder isn't trained

    for step in range(a.steps):
        chunk = rng.sample(arrays, a.batch)
        items = [flatten(x, vocab, rng, cfg["n_ids"], cfg["order"], cfg["max_anc"]) for x in chunk]
        b = collate(items, [""] * a.batch, tok, cfg["max_anc"])
        est, shapes = estimate_bytes(b, cfg, len(vocab))
        print(f"step {step}: {shapes}  estimated {est / 2**30:.2f}GB", flush=True)
        if est > a.budget_gb * 2**30:
            sys.exit(f"over budget ({a.budget_gb}GB): lower --max-tokens or --batch")

        t0 = time.time()
        with torch.no_grad():
            memory = enc(b["p_ids"], b["p_mask"])
        H = dec(b, memory, b["p_mask"] == 0)
        loss, real, _ = compute_loss(dec, H, b)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 1.0)
        opt.step()
        print(f"        loss {loss.item():.3f} (real point {real:.3f})  {time.time() - t0:.2f}s  "
              f"peak RSS {peak_rss_gb():.2f}GB", flush=True)


if __name__ == "__main__":
    main()
