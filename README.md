# treecodegen

Prompt → line-level Python syntax tree, with an encoder-decoder transformer.

For a step-by-step walk-through with a worked example, see [EXPLANATION.md](EXPLANATION.md).

## How it works

**Output.** Each statement is one tree node; its content is the line of code (or the header line of a block).
The statements inside a block are its children; a virtual root holds the top-level statements.

**Model.**

| Part | What it does |
|---|---|
| Prompt encoder | A BERT (pretrained with `--bert`, or small and trained from scratch); its output is the memory. |
| Node embedding | Turns each written token (an *entry*) into one vector (see below). |
| Decoder | Causal transformer layers (4 by default, 7 in the pretraining run below) with cross-attention to the prompt. No sequence position encoding. |
| Four heads | Pick the next move: a father, a line under it, for a new line its place among the siblings, a token (see below). |

**Entries.** Every token written is one decoder entry: self ID, father ID, older ancestors, neighbours, token,
position in the line, depth in the tree. IDs are random symbols from a pool of `--n-ids`, redrawn for every training
example; they only say "same node" or "different node". Each entry's vector (`--d-model`, 896 by default) has two parts:

- the path: one ID-code slot each for self, father, the `--max-anc` (4) nearest older ancestors, and the line's
  neighbours among its siblings: left 1, left 2, right 1. ID codes are fixed random vectors from one shared table
  (never trained), so a node has the same code in every role; EDGE (past the first or last sibling) and "no node"
  are two extra rows. The neighbours give the sibling order, which the random IDs can't. They are a snapshot of the
  moment the entry is written: inserting a line later never changes entries already written (so a cache of the
  decoder's states would stay valid). Missing slots are zeros, and deeper ancestors are left out. Last comes a
  learned depth slot (`--k-depth`, 32; the root is 0, depths from `--max-depth` − 1 = 15 up share one vector), since
  the ID slots can't tell depths apart once they are all full. `(2 + max-anc + 3) * --k-slot + --k-depth` =
  9 × 64 + 32 = 608 by default;
- the content: token + position in the line, the remaining 288.

If you change `--k-slot`, `--max-anc` or `--k-depth`, change `--d-model` with them.

**Moves.** Each step writes one token, through up to four choices, each seeing the earlier ones:

1. a father: any existing node (its ID code scored against a query), or STOP;
2. a line under it: an open child, to extend it, or a new line (which gets an unused ID);
3. for a new line, its place: right after one of the father's children, or first;
4. the token, added at the end of that line.

Lines can be written in any interleaving, and siblings in any order: a line can be inserted before lines already
written (a missing import at the top, `total = 0` before the loop that uses it). Every line ends with an `<end>`
token, written like any other; after it the line is closed (it can't be extended, but it can still get children).
STOP, once, ends the whole tree. The program, not the model, fills in each entry's IDs, neighbours, position and depth.

**Training.** Each sequence first writes a random partial tree (the mask) in a random valid order, then completes
the tree in a random valid order. The mask is `subset` (`--mask`, the default: small partial trees, median 9% of
the file) or `holes` (whole lines with blocks missing; the share missing is uniform from 0 to 100%), or a `mix` of both.
`subset` alone almost never shows an almost complete file with a block missing (see EXPLANATION.md, 6.1). The step right after the mask is the "real point"; `--real-weight` sets its
weight in the loss. At each step the loss is −log of the total probability of the valid moves (the frontier):
extending any open line (its last token being `<end>`), starting any unwritten child of any started node (at its
place among the siblings already written), or STOP when the tree is complete. Up to `--max-moves` of them are scored
per step; since the model can't know which ones, the loss can't go below log(valid / scored) at each step (about 2.0
on real files), so the log also shows a floor-adjusted loss (minus that floor; about 0 for a perfect model). How
interleaved the order is varies per sequence, from fully interleaved to one whole line after another,
which is how greedy generation tends to write.

**Generation.** At each step, a move (father, line, token) or (father, new line, place, token), or STOP, sampled
with `--temperature` (0.2 by default; 0 takes the most likely move every time).
STOP is allowed only once every line is closed (by `<end>`, or by reaching the length cap).

**Evaluation.** Matches training: each held-out tree gets a partial tree built like a training sequence (a random
mask, written in a random valid order up to the real point), and the model completes it. The score is the share of
correct moves among the moves required (the remaining tokens, each line's `<end>` included, plus STOP). A move is
correct if it is valid there (any order is fine); after a wrong one, the valid move the model scores highest is
applied, so every required move is judged. The model may write `<unk>` in the evaluation, since rare tokens are
`<unk>` in the reference too (scores from before this, up to the end of pass 2, are lower and not comparable). Also reported: the share of files completed with no wrong move, and the
validation loss (the training loss on held-out files). Training evaluates every `--eval-every` steps on `--eval-n`
held-out files of up to 1,000 entries, and prints one example: a held-out partial tree, the model's own completion
(no corrections, at most 1,000 moves) and the real file. `evaluate` does the same for a saved checkpoint, and with
`--report` writes it to a file. Generation keeps an attention cache: each step runs only the new entry through the
decoder (entries never change once written, so the cached keys and values stay valid).

## Layout

| File | What it does |
|---|---|
| `treecodegen/tree.py` | Python source ↔ tree (`code_to_tree`, `tree_to_code`), tree ↔ JSON node list (`tree_to_nodes`, `nodes_to_tree`) |
| `treecodegen/vocab.py` | Code-token vocabulary and prompt tokenizer |
| `treecodegen/sequences.py` | Tree → decoder entries with random IDs and neighbours, the mask, frontier targets, batching |
| `treecodegen/model.py` | `NodeEmbedding`, `PromptEncoder` (BERT), `TreeDecoder` |
| `treecodegen/loss.py` | Frontier loss |
| `treecodegen/evaluate.py` | Held-out evaluation: completing training-style partial trees, validation loss |
| `treecodegen/generate.py` | Greedy generation, prompt → tree |
| `treecodegen/train.py` | Data loading, training loop, evaluation, checkpoints |
| `treecodegen/toy.py` | Small synthetic dataset |
| `treecodegen/cli.py` | Command line |
| `load_datasets.py` | Builds a code-only pretraining file from codeparrot-clean |
| `EXPLANATION.md` | Step-by-step explanation of the whole system |
| `tests/test_treecodegen.py` | Round trips (incl. the whole standard library), JSON trees, frontier targets, IDs, a train step |

Data flows `tree → sequences → model → loss` during training, and `model → generate → tree` at inference.

## Requirements

The package imports only `torch` and `transformers`. `load_datasets.py` also needs `datasets`.
`requirements.txt` pins the versions this was tested with. `--bert` downloads a model from the
Hugging Face Hub on first use.

## Usage

```
pip install -r requirements.txt

# 1. Download the pretraining data (code only) -> Datasets/pretrain.jsonl
python load_datasets.py

# 2. Pretraining, pass 1: subset masks, one pass over the data (about 30 hours on one 48 GB GPU)
python -m treecodegen train --data Datasets/pretrain.jsonl --out pretrain.pt --n-ids 1024 --max-vocab 40000 \
    --layers 7 --heads 8 --batch 8 --accum 2 --steps 55000 --lr 3e-4 --warmup 1000 --eval-every 5000 --eval-n 20
#    quick check first: the same with --steps 200 --eval-every 100 --eval-n 5 --out pretrain_test.pt

# 3. Pretraining, pass 2: continue pass 1 with mostly holes masks (70%) and 30% line-first sequences, another pass
cp pretrain.pt pretrain_pass1.pt
python -m treecodegen train --data Datasets/pretrain.jsonl --out pretrain_pass2.pt --resume pretrain_pass1.pt \
    --mask mix --holes-share 0.7 --line-share 0.3 --batch 8 --accum 2 --steps 55000 --lr 1.5e-4 --warmup 1000 \
    --eval-every 5000 --eval-n 50

# 4. Pretraining, pass 3: continue pass 2, with 20% deep masks (holes only inside bodies), 55% holes, 25% subset
python -m treecodegen train --data Datasets/pretrain.jsonl --out pretrain_pass3.pt --resume pretrain_pass2.pt \
    --mask mix --deep-share 0.2 --holes-share 0.55 --line-share 0.3 --batch 8 --accum 2 --steps 55000 --lr 1.5e-4 \
    --warmup 1000 --eval-every 5000 --eval-n 50

# Look at a tree, evaluate or generate with a checkpoint, run the tests
python -m treecodegen show     --code "def f(w, h):\n    return w * h"
python -m treecodegen evaluate --ckpt pretrain.pt --data Datasets/pretrain.jsonl --n 100 [--mask holes]
python -m treecodegen generate --ckpt model.pt --prompt "write a function add that returns the sum of a and b"
python tests/test_treecodegen.py
```

Step 1 streams the data, so nothing is downloaded up front; it stops after `LIMIT` files (1,000,000 for the full
run, about 7.4 GB and 5–6 hours; 10,000 for a quick check).

Step 2 trains a 7-layer, 8-head decoder of width 896 (path 608 + content 288) with a 40,000-token vocabulary:
about 140M parameters, of which 47M are the token embedding and the token head. Measured on one 48 GB GPU, batch 8
peaks at 36 GB and takes about 0.57 s per step; batch 16 runs out of memory, so `--accum 2` makes the 16 (about 2 s
per step in the pass-1 run, 43 GB). With 883,932 training files left after the size filters, 55,000 steps of 16 is
about one pass over the data. The pass-1 run actually took its vocabulary with `--vocab-from pretrain_v2_step20000.pt` (an
earlier run's 40,000-token vocabulary) instead of `--max-vocab 40000`; reuse that file to reproduce it exactly. The checkpoint is saved at every
`--eval-every` steps as well as at the end, so a crash loses at most that many steps. For a bigger effective batch
without the memory, `--accum 2` adds up two batches of 8 into one step (like batch 16, about twice the time per step).

Step 3 (`--resume`) continues the same models: encoder, decoder, vocabulary and shapes come from the checkpoint, with
a new optimizer and a new learning-rate schedule (warmup, then cosine decay; half pass 1's peak, so as not to undo
it). Its sequences are 70% `holes`, 30% `subset`, so it learns files with whole blocks missing without forgetting the small
partial trees; each evaluation reports both kinds. In 30% of its sequences (`--line-share 0.3`), past the mask, a
line that is open must be finished before a new one starts: random orders start lines early and finish them late, and
free completions after pass 1 open line after line without finishing them. Keep `--seed` as in pass 1 (0 by default): it sets the
held-out split. `evaluate` counts epochs across both passes.

The token head scores every move against the whole code vocabulary, so the vocabulary sets most of the GPU memory:
keep it small with `--min-freq` (drop rare tokens) and `--max-vocab` (keep only the N most frequent).
On many files, `--max-vocab` is the one that matters; the log reports the share of tokens it covers
(40,000 tokens covered 90.5% of the first 40,000 files).
`--n-ids 1024` covers every file step 1 keeps: its size check counts every token plus one per node,
so a file has at most `MAX_ENTRIES / 2` nodes.

To train on prompt–code pairs, pass a JSONL with prompts (see below), optionally with `--bert bert-base-uncased
--freeze-encoder-steps 500`. Add `--init pretrain.pt` to start from the pretrained decoder: its vocabulary and
decoder settings (`--d-model`, `--layers`, `--heads`, `--n-ids`, `--k-slot`, `--max-anc`, `--max-pos`, `--max-depth`,
`--k-depth`)
come from the checkpoint, while the encoder and `mem_proj` start fresh. When every prompt is empty, the encoder is frozen. `--order dfs` trains on a fixed order instead (whole lines,
depth-first, no mask), for comparison.

## Data format

JSONL, one example per line, with the target given either as code or as a stored tree:

```json
{"prompt": "return the sum of a and b", "code": "def add(a, b):\n    return a + b"}
{"prompt": "", "tree": [{"id": 0, "father": null, "content": []},
                        {"id": 1, "father": 0, "content": ["import", "os"]}]}
```

- An empty prompt means code only, used to pretrain the decoder.
- `tree` (or `code_tree`) is the node list from `tree_to_nodes`: parents first, the root has `father: null`
  and empty content. The stored IDs are only for saving; training draws fresh random IDs every time.
- Rows that don't parse, or use unsupported statements, are skipped and counted.

## Pretraining data

`load_datasets.py` (usage step 1) reads [codeparrot/codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean)
(about 5.4M Python files) and writes one row per whole file to `Datasets/pretrain.jsonl`. Settings at the top of the script:

- `MAX_ENTRIES` (2048) and `MIN_NODES` (3): files with a longer decoder sequence, or fewer nodes, are skipped;
- `LIMIT` (10,000): number of files to keep (1,000,000 for the full run).

Docstrings are removed, since each would otherwise become a unique, useless vocabulary token.
If you raise `MAX_ENTRIES`, raise `--n-ids` to at least half of it.

For prompt–code pairs, [CodeSearchNet](https://huggingface.co/datasets/code-search-net/code_search_net) (docstring as
prompt, with the docstring removed from the code) is the largest fit. Keep
[MBPP](https://huggingface.co/datasets/google-research-datasets/mbpp) (500 test problems with unit tests) for evaluation.

## Known limits

- A line is finished only when the model writes `<end>`; until it does, the line stays open, and generation can't
  STOP. A line that never gets `<end>` runs until the length cap.
- Generation only adds: tokens go at the end of a line, lines can't be moved (e.g. wrapped in a new `try:`) or deleted.
- Checkpoints saved before the neighbour slots and `<end>` were added don't load.
- With `--init`, code tokens missing from the pretraining vocabulary become `<unk>`.
- Identifiers outside the training vocabulary become `<unk>` (no copy mechanism yet).
- Each string literal is one token, so long strings bloat the vocabulary.
- `try*` (exception groups) is not supported; `match` is.
