"""Data loading, training loop, evaluation, checkpoints."""
import ast
import json
import math
import os
import random
import time

import torch

from .evaluate import evaluate, fits, free_completion
from .generate import generate
from .loss import compute_loss
from .model import build_models
from .sequences import collate, flatten, tree_arrays
from .tree import code_to_tree, nodes_to_tree, tree_to_code
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


EVAL_MAX_ENTRIES = 1000   # evaluation: held-out files up to this many entries, example completions up to this many moves


def show_example(enc, dec, tok, vocab, rows, cfg, device, temperature, mask="subset"):
    """Print one example: a small held-out file's partial tree (built like a training sequence), the model's own
    completion of it (no corrections, at most EVAL_MAX_ENTRIES moves), and the real file."""
    small = [(p, t) for p, t in rows if fits(tree_arrays(t, vocab), cfg) and len(tree_to_code(t).split("\n")) <= 30]
    if not small:
        return
    enc.eval(); dec.eval()
    partial, done, ref = free_completion(enc, dec, tok, vocab, cfg, *small[0], random.Random(1), device, temperature,
                                         max_moves=EVAL_MAX_ENTRIES, mask=mask)
    enc.train(); dec.train()
    try:
        ast.parse(done); ok = "parses"
    except SyntaxError:
        ok = "doesn't parse"
    indent = lambda code: "\n".join("    | " + l for l in code.split("\n"))
    print(f"  example ({mask} mask): partial tree given (open lines end in …):\n" + indent(partial))
    print(f"  the model's completion (temperature {temperature}, {ok}):\n" + indent(done))
    print("  the real file:\n" + indent(ref), flush=True)


def pick_device(name=None):
    """The given device, else CUDA, else Apple GPU (mps), else CPU."""
    if name:
        return name
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


MPS_CACHE_LIMIT = 30 * 2**30   # bytes of Apple-GPU memory above which the allocator cache is emptied


# Settings that fix the decoder's weight shapes; --init takes them from the checkpoint.
DECODER_KEYS = ("d_model", "layers", "heads", "n_ids", "k_slot", "max_anc", "max_pos", "max_depth", "k_depth")


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
    if a.init and a.resume:
        raise SystemExit("--init and --resume are exclusive")
    init = None
    if a.init or a.resume:
        say(f"loading checkpoint {a.init or a.resume}")
        init = torch.load(a.init or a.resume, map_location="cpu")
    # Fine-tuning a pretrained decoder keeps its vocabulary, so token IDs mean the same thing.
    # --vocab-from takes only the vocabulary of another checkpoint (not its weights), e.g. to keep a previous run's.
    if init or a.vocab_from:
        src = a.init or a.resume or a.vocab_from
        say(f"taking the vocabulary from {src}")
        vocab = Vocab(init["vocab"] if init else torch.load(src, map_location="cpu")["vocab"])
    else:
        say(f"building the code vocabulary (min-freq {a.min_freq}, max-vocab {a.max_vocab})")
        vocab = Vocab.build([t for _, t in train_rows], a.min_freq, a.max_vocab)
    say(f"code vocabulary: {len(vocab)} tokens" +
        (f", covering {vocab.coverage:.1%} of training tokens (the rest are <unk>)" if hasattr(vocab, "coverage") else ""))
    say("building the prompt tokenizer")
    cfg = dict(bert=a.bert, enc_hidden=a.enc_hidden, enc_layers=a.enc_layers, d_model=a.d_model, layers=a.layers,
               heads=a.heads, dropout=a.dropout, n_ids=a.n_ids, k_slot=a.k_slot,
               max_anc=a.max_anc, max_pos=a.max_pos, max_depth=a.max_depth, k_depth=a.k_depth, order=a.order)
    if a.resume:   # the same models, continued: every shape, the encoder and the prompt tokenizer come from the checkpoint
        cfg.update({k: v for k, v in init["cfg"].items() if k in DECODER_KEYS + ("bert", "enc_hidden", "enc_layers")})
        tok = PromptTokenizer(cfg["bert"], itos=init["prompt_itos"])
    else:
        tok = PromptTokenizer(a.bert, texts=[p for p, _ in train_rows])
    if a.init:   # decoder shape comes from the checkpoint; the encoder is new, built from this run's flags
        cfg.update({k: init["cfg"].get(k, cfg[k]) for k in DECODER_KEYS})   # .get: checkpoints from before max_depth
    cfg.update(mask=a.mask, holes_share=a.holes_share, line_share=a.line_share, deep_share=a.deep_share)
    say("building the models")
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    if a.resume:
        enc.load_state_dict(init["enc"]); dec.load_state_dict(init["dec"])
        print(f"models resumed from {a.resume} (its step {init.get('step')}); new optimizer, new learning-rate schedule")
    elif init:
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
    # training sequences seen by the checkpoint this run resumes from (and the ones before it), for epoch counts
    prior = init.get("prior_sequences", 0) + init["step"] * init["batch"] if a.resume else 0
    def save(step):
        """Write the checkpoint to a temporary file, then rename it: a crash mid-save keeps the previous one."""
        torch.save(dict(cfg=cfg, enc=enc.state_dict(), dec=dec.state_dict(), vocab=vocab.itos,
                        prompt_itos=None if cfg["bert"] else tok.itos, step=step, batch=a.batch * a.accum,
                        n_train=len(arrays), steps=a.steps, resumed_from=a.resume, prior_sequences=prior),
                   a.out + ".tmp")
        os.replace(a.out + ".tmp", a.out)

    say(f"starting training: {a.steps} steps, batch {a.batch}" +
        (f" x {a.accum} accumulated = {a.batch * a.accum} sequences per step" if a.accum > 1 else ""))
    t0, step, micro = time.time(), 0, 0
    while step < a.steps:
        rng.shuffle(arrays)
        for i in range(0, len(arrays) - a.batch + 1, a.batch):
            chunk = arrays[i:i + a.batch]
            if micro == 0:
                t_step, losses, reals, norms = time.time(), [], [], []   # this step's batch losses (for the log)
            if step == 0 and micro == 0:
                say("first batch: building sequences")
            items = [flatten(x, vocab, rng, cfg["n_ids"], a.order, cfg["max_anc"], a.max_moves, mask=a.mask,
                             holes_share=a.holes_share, line_share=a.line_share, deep_share=a.deep_share)
                     for _, x in chunk]
            b = {k: v.to(device) for k, v in collate(items, [p for p, _ in chunk], tok, cfg["max_anc"]).items()}

            if micro == 0:
                scale = min(1, (step + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
                for g, lr in zip(opt.param_groups, base):
                    g["lr"] = lr * scale
                opt.zero_grad()
            frozen = no_prompts or step < a.freeze_encoder_steps
            with torch.set_grad_enabled(not frozen):
                memory = enc(b["p_ids"], b["p_mask"])
            H = dec(b, memory, b["p_mask"] == 0)
            loss, real, norm = compute_loss(dec, H, b, a.real_weight)
            if step == 0 and micro == 0:
                say(f"first batch: forward done ({b['tok'].shape[1]} entries, {len(b['moves'])} moves), "
                    f"running backward")

            # --accum batches are added up into one step: same as a batch accum times bigger, in batch-sized memory
            (loss / a.accum).backward()
            losses.append(loss.item()); norms.append(norm)
            if real == real:                      # not NaN (a batch with no real point)
                reals.append(real)
            micro += 1
            if micro < a.accum:
                continue
            micro = 0
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
                real = sum(reals) / len(reals) if reals else float("nan")
                print(f"step {step:5d}  loss {sum(losses) / len(losses):.3f}  (floor-adjusted {sum(norms) / len(norms):.3f}, "
                      f"real point {real:.3f})  "
                      f"{'enc frozen  ' if frozen else ''}{time.time()-t0:.0f}s{mem}", flush=True)
            if a.eval_n > 0 and (step % a.eval_every == 0 or step == a.steps):
                say(f"evaluating on {min(a.eval_n, len(val))} held-out rows")
                # a run with holes masks is evaluated with each kind, to see a gain on one and any loss on another
                kinds = ["subset"] if a.mask == "subset" else ["subset", "holes"]
                if a.mask == "deep" or (a.mask == "mix" and a.deep_share > 0):
                    kinds.append("deep")
                for mask in kinds:
                    r = evaluate(enc, dec, tok, vocab, cfg, val[:a.eval_n], device, max_entries=EVAL_MAX_ENTRIES,
                                 mask=mask)
                    print(f"== step {step}{'' if a.mask == 'subset' else f' ({mask} masks)'}: held-out completion "
                          f"{r['moves']:.1%} of moves correct, {r['files']:.0%} of "
                          f"files with no wrong move, validation loss {r['loss']:.3f} (floor-adjusted {r['loss_norm']:.3f}) "
                          f"({r['n']} files)", flush=True)
                show_example(enc, dec, tok, vocab, val[:a.eval_n], cfg, device, a.temperature,
                             "subset" if a.mask == "subset" else "holes")
            if step % a.eval_every == 0 and step < a.steps:   # a checkpoint along the way, so a crash loses little
                save(step)
                say(f"checkpoint saved to {a.out} (step {step})")
            if step >= a.steps:
                break

    say(f"saving {a.out}")
    save(step)
    print(f"saved {a.out}")


def load_checkpoint(path, device="cpu"):
    ck = torch.load(path, map_location=device)
    cfg, vocab = ck["cfg"], Vocab(ck["vocab"])
    tok = PromptTokenizer(cfg["bert"], itos=ck["prompt_itos"])
    enc, dec = build_models(cfg, len(vocab), tok.vocab_size)
    enc.load_state_dict(ck["enc"]); dec.load_state_dict(ck["dec"])
    return enc.to(device).eval(), dec.to(device).eval(), tok, vocab, cfg
