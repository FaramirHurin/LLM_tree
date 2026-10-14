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
| Decoder | 4 causal transformer layers with cross-attention to the prompt. No sequence position encoding. |
| Three heads | Pick the next move: a father, a line under it, a token (see below). |

**Entries.** Every token written is one decoder entry: self ID, father ID, older ancestors, child numbers, token,
position in the line. IDs are random symbols from a pool of `--n-ids`, redrawn for every training example; they only say
"same node" or "different node". Each entry's vector (`--d-model`, 576 by default) has two parts:

- the path: self, father, then the `--max-anc` (4) nearest older ancestors, one slot each. Each slot is the node's
  ID code plus a learned vector for its child number (its place among its siblings; one table per level).
  ID codes are fixed random vectors from one shared table (never trained), so a node has the same code as self,
  father or ancestor. The child numbers give the sibling order, which the random IDs can't; with them the entries
  describe exactly one tree. Missing ancestors are zero slots, the root has no child number, and deeper ancestors
  are left out. `(2 + max-anc) * --k-slot` = 6 × 64 = 384 by default;
- the content: token + position in the line, the remaining 192.

If you change `--k-slot` or `--max-anc`, change `--d-model` with them.

**Moves.** Each step writes one token, through three choices, each seeing the earlier ones:

1. a father: any existing node (its ID code scored against a query), or STOP;
2. a line under it: an existing child, to extend it, or a new line (which gets an unused ID and the next child
   number);
3. the token, added at the end of that line.

Lines can be written in any interleaving. There is no end marker per line, only STOP, once, for the whole tree.
The one fixed rule: under the same father, lines are started in their real order (any father can be chosen at any
step, and started lines can be extended in any order). The program, not the model, fills in each entry's IDs,
child numbers and position.

**Training.** Each sequence first writes a random partial tree (the mask) in a random valid order, then completes
the tree in a random valid order. The step right after the mask is the "real point"; `--real-weight` sets its
weight in the loss. At each step the loss is −log of the total probability of the valid moves (the frontier):
extending any unfinished line, starting the next child of any started node, or STOP when the tree is complete.
Up to `--max-moves` of them are scored per step.

**Generation.** At each step, the most likely (father, line, token) triple, or STOP if it scores at least as high.

**Evaluation.** Exact tree match (sibling order included) and the fraction of outputs that parse as Python.

## Layout

| File | What it does |
|---|---|
| `treecodegen/tree.py` | Python source ↔ tree (`code_to_tree`, `tree_to_code`), tree ↔ JSON node list (`tree_to_nodes`, `nodes_to_tree`) |
| `treecodegen/vocab.py` | Code-token vocabulary and prompt tokenizer |
| `treecodegen/sequences.py` | Tree → decoder entries with random IDs and child numbers, the mask, frontier targets, batching |
| `treecodegen/model.py` | `NodeEmbedding`, `PromptEncoder` (BERT), `TreeDecoder` |
| `treecodegen/loss.py` | Frontier loss |
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

# 2. Pretrain the decoder on it
python -m treecodegen train --data Datasets/pretrain.jsonl --out pretrain.pt --n-ids 1024

# Look at a tree, generate, run the tests
python -m treecodegen show     --code "def f(w, h):\n    return w * h"
python -m treecodegen generate --ckpt model.pt --prompt "write a function add that returns the sum of a and b"
python tests/test_treecodegen.py
```

Step 1 streams the data, so nothing is downloaded up front; it stops after `LIMIT` files (10,000 by default; set 1,000,000 for the full run).
`--n-ids 1024` covers every file step 1 keeps: its size check counts every token plus one per node,
so a file has at most `MAX_ENTRIES / 2` nodes.

To train on prompt–code pairs, pass a JSONL with prompts (see below), optionally with `--bert bert-base-uncased
--freeze-encoder-steps 500`. Add `--init pretrain.pt` to start from the pretrained decoder: its vocabulary and
decoder settings (`--d-model`, `--layers`, `--heads`, `--n-ids`, `--k-slot`, `--max-anc`, `--max-pos`)
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

- With no end marker per line, the model must tell from a line's content whether it is finished; the most common
  error is stopping while a line is still incomplete.
- Siblings under the same father must be started in their real order; only lines, not sibling starts, are free.
- Checkpoints saved before child numbers were added don't load (the embedding has new weights).
- With `--init`, code tokens missing from the pretraining vocabulary become `<unk>`.
- Identifiers outside the training vocabulary become `<unk>` (no copy mechanism yet).
- Each string literal is one token, so long strings bloat the vocabulary.
- Generation recomputes the whole sequence at every step (no cache).
- `try*` (exception groups) is not supported; `match` is.
