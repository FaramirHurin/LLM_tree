"""Data loading, training loop, evaluation, checkpoints."""
import ast
import json
import math
import random
import time

import torch

from .generate import generate
from .loss import compute_loss
from .model import build_models
from .sequences import collate, flatten, tree_arrays
from .tree import code_to_tree, nodes_to_tree, tree_to_code, tree_to_nodes
from .vocab import PromptTokenizer, Vocab


def load_jsonl(path):
    """Rows of {"prompt", "code"} or {"prompt", "tree"} -> [(prompt, tree)].

    "tree" is the flat node list from tree_to_nodes ("code_tree" is accepted too).
    Rows that don't parse are skipped.
    """
    rows, skipped = [], 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            try:
                stored = r.get("tree", r.get("code_tree"))
                tree = nodes_to_tree(stored) if stored is not None else code_to_tree(r["code"])
                rows.append((r.get("prompt", ""), tree))
            except (SyntaxError, ValueError, KeyError):
                skipped += 1
    if skipped:
        print(f"skipped {skipped} rows that did not parse or used unsupported statements")
    return rows


def evaluate(enc, dec, tok, vocab, rows, cfg, device, n=100, show=3):
    """Greedy generation on held-out rows; returns (exact tree match, sibling order included; fraction that parses)."""
    enc.eval(); dec.eval()
    exact = parses = 0
    rng = random.Random(1)
    for i, (prompt, tree) in enumerate(rows[:n]):
        gen = generate(enc, dec, tok, vocab, prompt, cfg, rng, device=device)
        exact += tree_to_nodes(gen) == tree_to_nodes(tree)
        code = tree_to_code(gen)
        try:
            ast.parse(code); parses += 1
        except SyntaxError:
            pass
        if i < show:
            print(f"  prompt: {prompt}\n" + "\n".join("    | " + l for l in code.split("\n")))
    enc.train(); dec.train()
    m = min(n, len(rows))
    return exact / m, parses / m


def pick_device(name=None):
    """The given device, else CUDA, else Apple GPU (mps), else CPU."""
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


MPS_CACHE_LIMIT = 30 * 2**30   # bytes of Apple-GPU memory above which the allocator cache is emptied


# Settings that fix the decoder's weight shapes; --init takes them from the checkpoint.
DECODER_KEYS = ("d_model", "layers", "heads", "n_ids", "k_slot", "max_anc", "max_pos")


def train(a):
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    t_start = time.time()
    say = lambda msg: print(f"[{time.time() - t_start:6.0f}s] {msg}", flush=True)
    say(f"loading {a.data}")
    rows = load_jsonl(a.data)
    rng.shuffle(rows)
    n_val = max(1, int(len(rows) * a.val_frac))
    val, train_rows = rows[:n_val], rows[n_val:]
    say(f"loaded {len(rows)} rows ({len(train_rows)} train, {len(val)} val)")
    init = None
    if a.init:
        say(f"loading checkpoint {a.init}")
        init = torch.load(a.init, map_location="cpu")
    # Fine-tuning a pretrained decoder keeps its vocabulary, so token IDs mean the same thing.
    say("taking the vocabulary from the checkpoint" if init else f"building the code vocabulary (min-freq {a.min_freq})")
    vocab = Vocab(init["vocab"]) if init else Vocab.build([t for _, t in train_rows], a.min_freq)
    say(f"code vocabulary: {len(vocab)} tokens")
    say("building the prompt tokenizer")
    tok = PromptTokenizer(a.bert, texts=[p for p, _ in train_rows])
    cfg = dict(bert=a.bert, enc_hidden=a.enc_hidden, enc_layers=a.enc_layers, d_model=a.d_model, layers=a.layers,
               heads=a.heads, dropout=a.dropout, n_ids=a.n_ids, k_slot=a.k_slot,
               max_anc=a.max_anc, max_pos=a.max_pos, order=a.order)
    if init:   # decoder shape comes from the checkpoint; the encoder is new, built from this run's flags
        cfg.update({k: init["cfg"][k] for k in DECODER_KEYS})
    say("building the models")
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    if init:
        # mem_proj maps the encoder output into the decoder: its shape depends on the encoder, and in
        # pretraining it only saw empty prompts, so it starts fresh.
        dec.load_state_dict({k: v for k, v in init["dec"].items() if not k.startswith("mem_proj.")}, strict=False)
        print(f"decoder initialized from {a.init} ({', '.join(f'{k}={cfg[k]}' for k in DECODER_KEYS)})")
    device = pick_device(a.device)
    say(f"moving the models to {device}")
    enc.to(device); dec.to(device)

    say("converting trees to arrays")
    arrays = [(p, tree_arrays(t, vocab)) for p, t in train_rows]
    arrays = [(p, x) for p, x in arrays if max(len(c) for c in x[0]) <= cfg["max_pos"] and len(x[0]) <= cfg["n_ids"]]
    no_prompts = not any(p for p, _ in train_rows)    # code-only pretraining: the encoder has nothing to learn
    say(f"kept {len(arrays)} of {len(train_rows)} training trees (dropped: a line longer than max-pos "
        f"or more nodes than n-ids)")
    print(f"train {len(arrays)}  val {len(val)}  code vocab {len(vocab)}  prompt vocab {tok.vocab_size}  "
          f"params enc {sum(p.numel() for p in enc.parameters())/1e6:.1f}M "
          f"dec {sum(p.numel() for p in dec.parameters())/1e6:.1f}M")

    enc_lr = a.enc_lr if a.enc_lr is not None else (3e-5 if a.bert else a.lr)
    opt = torch.optim.AdamW([{"params": dec.parameters(), "lr": a.lr},
                             {"params": enc.parameters(), "lr": enc_lr}], weight_decay=0.01)
    base = [a.lr, enc_lr]
    say(f"starting training: {a.steps} steps, batch {a.batch}")
    t0, step = time.time(), 0
    while step < a.steps:
        rng.shuffle(arrays)
        for i in range(0, len(arrays) - a.batch + 1, a.batch):
            chunk = arrays[i:i + a.batch]
            t_step = time.time()
            if step == 0:
                say("first batch: building sequences")
            items = [flatten(x, vocab, rng, cfg["n_ids"], a.order, cfg["max_anc"], a.max_moves) for _, x in chunk]
            b = {k: v.to(device) for k, v in collate(items, [p for p, _ in chunk], tok, cfg["max_anc"]).items()}

            scale = min(1, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
            for g, lr in zip(opt.param_groups, base):
                g["lr"] = lr * scale
            frozen = no_prompts or step < a.freeze_encoder_steps
            with torch.set_grad_enabled(not frozen):
                memory = enc(b["p_ids"], b["p_mask"])
            H = dec(b, memory, b["p_mask"] == 0)
            loss, real = compute_loss(dec, H, b, a.real_weight)
            if step == 0:
                say(f"first batch: forward done ({b['tok'].shape[1]} entries, {len(b['moves'])} moves), "
                    f"running backward")

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), 1.0)
            opt.step(); step += 1
            if device == "mps" and torch.mps.driver_allocated_memory() > MPS_CACHE_LIMIT:
                # Batch shapes change every step, and the mps allocator caches each shape's blocks; without this
                # the cache grows until the system swaps. Emptying it waits for the GPU, so only do it when it's big.
                torch.mps.empty_cache()
            if step <= 10:
                if device == "mps":
                    torch.mps.synchronize()     # GPU work is queued; wait for it so the time is real
                say(f"step {step} done ({time.time() - t_step:.1f}s)")

            if step % a.log_every == 0:
                mem = f"  gpu {torch.mps.driver_allocated_memory() / 2**30:.1f}GB" if device == "mps" else ""
                print(f"step {step:5d}  loss {loss.item():.3f}  (real point {real:.3f})  "
                      f"{'enc frozen  ' if frozen else ''}{time.time()-t0:.0f}s{mem}", flush=True)
            if a.eval_n > 0 and (step % a.eval_every == 0 or step == a.steps):
                say(f"evaluating on {min(a.eval_n, len(val))} held-out rows")
                em, ok = evaluate(enc, dec, tok, vocab, val, cfg, device, a.eval_n)
                print(f"== step {step}: held-out tree exact match {em:.1%}, parses {ok:.1%}", flush=True)
            if step >= a.steps:
                break

    say(f"saving {a.out}")
    torch.save(dict(cfg=cfg, enc=enc.state_dict(), dec=dec.state_dict(), vocab=vocab.itos,
                    prompt_itos=None if a.bert else tok.itos), a.out)
    print(f"saved {a.out}")


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg, vocab = ck["cfg"], Vocab(ck["vocab"])
    tok = PromptTokenizer(cfg["bert"], itos=ck["prompt_itos"])
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
    return enc.to(device).eval(), dec.to(device).eval(), tok, vocab, cfg
