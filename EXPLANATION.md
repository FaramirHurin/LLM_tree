# How treecodegen works

This file explains the whole system step by step, with one running example. The README covers how to use it.

**The idea in short.** A prompt goes in, Python comes out. The output is not written as a flat string, but as a
tree of lines. At each step the model picks *where* to write (a line in the tree) and *what* to write (one token).
It may work on several lines in any order: start a function header, jump into its body, come back to finish the header.
The one rule: under the same father, lines are *started* in their real order (section 6.2).

---

## 1. From code to tree

Each statement is one node. Its content is the line of code, split into tokens. A block (`def`, `if`, `for`, `class`,
`try`, ...) keeps only its header line; the statements inside become its children. `else`, `except` and `finally`
become children of the block they belong to. A virtual root with no content holds the top-level statements.

```python
def add(a, b):
    c = a + b
    return c
```

```
<root>
  def add ( a , b ) :        <- node "def", child of the root
    c = a + b                <- node "c",   first child of "def"
    return c                 <- node "ret", second child of "def"
```

Code: [tree.py](treecodegen/tree.py) (`code_to_tree`, `tree_to_code`).

---

## 2. What the decoder reads: one entry per token

Every token written is one **entry**. An entry has six fields:

| Field | Meaning |
|---|---|
| self ID | the ID of the line this token belongs to |
| father ID | the ID of that line's father |
| ancestors | the IDs of the grandfather, great-grandfather, ... (up to 4, nearest first) |
| child numbers | for self, father and each ancestor: its place among its siblings (0 = first child) |
| token | the token itself |
| pos | its position inside the line (0, 1, 2, ...) |

**IDs are random symbols.** Every node gets a random number from a pool (`--n-ids`, 512 by default), drawn fresh for
every training example. An ID means nothing by itself; it only says "same node" or "different node". This way the
model can't memorize that "node 7 is always a `return`".

**Child numbers give the order the IDs can't.** `c = a + b` and `return c` have the same father, random IDs,
and positions that both start at 0. Without child numbers, swapping the two lines would give the same entries.
With them, `c` is child 0 and `return` is child 1, so the entries describe exactly one tree, sibling order included.
The root is no one's child, so it has no child number (–1, stored as zeros).

The model only chooses the token and the place (father + line). The program fills in the rest of the entry:
the IDs, the ancestors, the child numbers and the position all follow from the place. A new line gets the next
child number under its father.

Here is one real training sequence for the example (from `python -m treecodegen show --code ...`):

| entry | self | father | ancestors | child # (self, father, ...) | pos | token | line |
|---:|---:|---:|---|---|---:|---|---|
| 0 | 49 | – | [] | [–] | 0 | `<root>` | root |
| 1 | 53 | 49 | [] | [0, –] | 0 | `def` | def |
| 2 | 53 | 49 | [] | [0, –] | 1 | `add` | def |
| 3 | 53 | 49 | [] | [0, –] | 2 | `(` | def |
| 4 | 5 | 53 | [49] | [0, 0, –] | 0 | `c` | c |
| 5 | 5 | 53 | [49] | [0, 0, –] | 1 | `=` | c |
| 6 | 5 | 53 | [49] | [0, 0, –] | 2 | `a` | c |
| 7 | 5 | 53 | [49] | [0, 0, –] | 3 | `+` | c |
| 8 | 53 | 49 | [] | [0, –] | 3 | `a` | def |
| 9 | 53 | 49 | [] | [0, –] | 4 | `,` | def |
| 10 | 53 | 49 | [] | [0, –] | 5 | `b` | def |
| 11 | 53 | 49 | [] | [0, –] | 6 | `)` | def ← **real point** |
| 12 | 33 | 53 | [49] | [1, 0, –] | 0 | `return` | ret |
| 13 | 53 | 49 | [] | [0, –] | 7 | `:` | def |
| 14 | 5 | 53 | [49] | [0, 0, –] | 4 | `b` | c |
| 15 | 33 | 53 | [49] | [1, 0, –] | 1 | `c` | ret |

Read `[1, 0, –]` as: this line is child 1 of its father, the father is child 0 of its own father, which is the root.

The lines are interleaved: the header `def` is unfinished when the body starts. The IDs put every token back in
its line, the positions put it in place inside the line, and the child numbers put the lines in order. So the
tree is the same however the entries are ordered.

---

## 3. Inside an entry: the 576 numbers

Each entry becomes one vector of `--d-model` = 576 numbers, in two parts:

| Part | Size | Made from |
|---|---:|---|
| self slot | 64 | ID code of self + child-number vector of self |
| father slot | 64 | ID code of father + child-number vector of father |
| grandfather slot | 64 | same (zeros if there is none) |
| great-grandfather slot | 64 | same (zeros if there is none) |
| level 5 slot | 64 | same (zeros if there is none) |
| level 6 slot | 64 | same (zeros if there is none) |
| **path total** | **384** | then a LayerNorm |
| content | 192 | token embedding + position embedding, then a LayerNorm |

- **The ID table** holds one fixed random unit vector of 64 numbers per ID. It is never trained, and it is the same
  table for every role: node 53 has the same code whether it appears as self, father or ancestor.
  The root's father is a special extra row, "no node".
- **The child-number vectors are learned**, one table per level: "child 1" as self has a different vector than
  "child 1" as father. They are added to the ID code in the same slot, so each slot says *who* the node is (ID)
  and *where* it sits among its siblings (child number). The root has no child number, so nothing is added.
- **Each ancestor level has its own slot**, so the slot tells the level. Ancestors deeper than 4 levels are left out.
- **There is no sequence position.** The decoder doesn't know "this is entry 12". Order comes from the child
  numbers (between lines) and the position inside each line, which is why any interleaving of the lines means the
  same tree.

This is the standard way to give a transformer positions in a tree: each node's place is its path of child
numbers from the root (as in Shiv & Quirk, *Novel positional encodings to enable tree-based transformers*, 2019).
Here the path also carries the random IDs, so the model can point back at a line to continue it.

Code: `NodeEmbedding` in [model.py](treecodegen/model.py).

---

## 4. The model, layer by layer

```
prompt ──► encoder (BERT) ──► memory ──────────────┐
                                                   ▼ (cross-attention)
entries ──► NodeEmbedding ──► 4 decoder layers ──► H ──► three heads ──► next move
```

| Layer | What it does |
|---|---|
| **Prompt encoder** | A BERT. Pretrained (`--bert bert-base-uncased`) or a small one trained from scratch (128 wide, 2 layers). Turns the prompt into one vector per prompt token: the *memory*. |
| **mem_proj** | A linear layer that maps the memory to the decoder's width (576). |
| **NodeEmbedding** | Turns each entry into its 576 numbers (section 3). |
| **Decoder layer** ×4 | Each layer: self-attention over the earlier entries (causal: entry t sees only 0..t), then cross-attention to the prompt memory, then a feed-forward block (576 → 2304 → 576). 4 heads, pre-norm. |
| **Final LayerNorm** | Gives H: one hidden state per entry. H[t] is used to predict the move after entry t. |
| **Father head** | `fa_q` turns H[t] into a 64-number query. Each existing node's score = query · its ID code. One extra score (`stop`) for STOP. Softmax over all of them. |
| **fa_cond** | Adds the chosen father's state to H[t]: u = H[t] + fa_cond(state of the father's latest entry). |
| **Line head** | `id_q` turns u into a query. Each child of the father gets query · its ID code. One extra score (`new`) for a new line. Softmax. |
| **line_cond / new_line** | Adds the chosen line's state (its latest entry) to u, or a learned "new line" vector. |
| **Token head** | LayerNorm + linear layer over the code vocabulary. |

The father and line heads compare against the **same ID table** the input uses. The model doesn't learn
"choose node 53"; it learns "choose the node whose code looks like this", which works for any random ID.

With default settings the decoder has about 22M parameters.

---

## 5. One step: three choices

After entry t, the next move is chosen in three steps. Each step sees the earlier choices:

| # | Choice | Options | Masked out |
|---|---|---|---|
| 1 | **Father** | any node that exists, or STOP | nodes not written yet |
| 2 | **Line** | an existing child of that father (extend its line), or a new line | nodes that aren't children of that father |
| 3 | **Token** | any code token | – |

The token goes **at the end** of the chosen line. The probability of a move is the product of the three:

```
P(move) = P(father) × P(line | father) × P(token | father, line)
```

What the choice means:

- **existing line** → the token is appended to that line;
- **new line** → a new child of the father, with a fresh unused ID and the next child number; the token is its
  first one;
- **STOP** → the tree is done.

Because of the masks, impossible moves can't be chosen. A line that exists under a different father can't be picked.
A grandfather is never chosen at all: it follows from the father.

---

## 6. Training

### 6.1 Building a training sequence

For each tree, one training sequence is built in three parts ([sequences.py](treecodegen/sequences.py), `flatten`):

1. **The mask: a random partial tree.** Starting at the root, each node keeps a random number of its first
   children, and each kept child keeps a random non-empty start of its line. In the example the mask is
   `def add ( a , b )` (7 of 8 tokens), `c = a +` (4 of 5) and nothing of `return c`.
2. **The mask, written in a random valid order** (entries 1–11). The step after its last entry is the
   **real point** (entry 11 → the move to entry 12).
3. **The rest of the tree, in a random valid order** (entries 12–15).

The real point matters because it matches a situation the model can really find itself in: a half-written tree,
reached by any route. `--real-weight` gives it more weight in the loss.

### 6.2 Which moves are valid

At any moment several moves are correct. The valid moves are:

- **extend** any line that is started but not finished, with its next token;
- **start** the next unwritten child of any node that is already started (siblings start in their real order:
  the second child can start only once the first one has started, so a new line's child number is always
  the next one);
- **STOP**, but only once the whole tree is written.

At the real point of the example (after entry 11):

| Valid move | Father | Line | Token |
|---|---|---|---|
| finish the header | root (49) | def (53) | `:` |
| finish `c = a + b` | def (53) | c (5) | `b` |
| start the `return` line | def (53) | new | `return` |

STOP is not valid here: the tree is not complete.

So the freedom is partial:

| | Free? |
|---|---|
| Which father to work under at each step (any existing node) | yes |
| Extending any started line, in any order | yes |
| Interleaving lines, going deep or wide, coming back later | yes |
| Moving to another branch before finishing this one | yes |
| **Which sibling to start next under the same father** | **no**, always the next one in real order |

Lifting the last rule would need the model to also choose *where* among the siblings a new line goes, for
example by storing its left and right neighbors when it is created. That is not implemented.

### 6.3 The loss

The model shouldn't be punished for picking one correct move over another. So the loss at each step is

```
loss = −log( P(move 1) + P(move 2) + ... )      over the valid moves
```

It is small when the model puts most of its probability on *any* valid move. For example, with invented numbers:

| Valid move | P(father) | P(line \| father) | P(token \| father, line) | P(move) |
|---|---:|---:|---:|---:|
| root → def → `:` | 0.30 | 0.95 | 0.90 | 0.257 |
| def → c → `b` | 0.60 | 0.50 | 0.80 | 0.240 |
| def → new → `return` | 0.60 | 0.45 | 0.90 | 0.243 |
| **total** | | | | **0.740** |

Loss at this step = −log(0.740) = 0.30. The remaining 0.26 went to invalid moves (STOP, wrong tokens, ...).

To save compute, up to `--max-moves` (4) valid moves are scored per step: the one actually taken plus random
others. The loss is averaged over all steps of all sequences in the batch.

### 6.4 STOP happens once per tree

STOP is a valid move exactly once: after the last entry. At every other step it is wrong, so the loss pushes it
down at every step and up only at the end.

### 6.5 Putting it together

For each batch: build one sequence per tree with fresh IDs and a fresh mask, run the encoder on the prompts and the
decoder on the entries, compute the loss, update the weights ([train.py](treecodegen/train.py),
[loss.py](treecodegen/loss.py)). Every time a tree is seen again it gets new IDs, a new mask and a new order.

`--order dfs` trains on one fixed order instead (whole lines, depth-first, no mask), for comparison.

---

## 7. Generation

[generate.py](treecodegen/generate.py) builds the tree one token at a time:

1. Start with the root entry (random ID, `<root>` token).
2. Run the decoder on all entries so far.
3. Score every move: each existing node as father × each of its children or a new line × the best
   token for that pair. Keep the best of these moves.
4. If STOP scores at least as high as that move, stop. Otherwise write it: for a new line pick an unused ID,
   and fill in the entry (self, father, ancestors, child numbers, position).
5. Go back to 2.

Example, the start of a generation:

| Step | Best move | Result |
|---|---|---|
| 1 | root → new → `def` | new node A, child 0 of the root, line `def` |
| 2 | root → A → `add` | `def add` |
| 3 | A → new → `c` | new node B, child 0 of A, line `c` |
| 4 | root → A → `(` | `def add (` |
| 5 | A → new → `return` | new node C, child 1 of A, line `return` |
| ... | ... | ... |
| n | STOP | done |

At the end, each node's tokens are its line, and siblings are placed by child number (= the order they were started).
`tree_to_code` prints it as Python.

---

## 8. Evaluation

On held-out examples, the model generates a tree from the prompt and two numbers are reported:

- **exact match**: the generated tree equals the reference tree (same lines, same nesting, same sibling order);
- **parses**: the generated code is valid Python.

On the small toy dataset (2,850 training examples, 3,000 steps, measured before child numbers were added):
3% exact, 56% parse. Given a correct history,
the model's best move is valid 99% of the time; generating on its own, small errors add up over ~100 steps.
The most common error is STOP before a line is finished (e.g. `else` with no `:`).

---

## 9. Settings that shape the model

| Flag | Default | What it sets |
|---|---:|---|
| `--d-model` | 576 | entry width = path (384) + content (192) |
| `--k-slot` | 64 | size of one ID code |
| `--max-anc` | 4 | ancestor levels in the path; path = (2 + max-anc) × k-slot |
| `--n-ids` | 512 | size of the ID pool, and of each child-number table; must be ≥ the number of nodes in a tree |
| `--max-pos` | 128 | longest line, in tokens |
| `--layers`, `--heads` | 4, 4 | decoder depth and attention heads |
| `--order` | random | `random` (mask + random order) or `dfs` (fixed order) |
| `--max-moves` | 4 | valid moves scored per step |
| `--real-weight` | 1.0 | loss weight of the real point |

---

## 10. Words used in the code

| Word | Meaning |
|---|---|
| node / line | one statement; its tokens are its content |
| entry | one written token, with its IDs, child numbers and position |
| ID / ID code | the random symbol of a node / its fixed 64-number vector |
| chain | a node's IDs: self, father, grandfather, ... |
| rank / child number | a node's place among its siblings (0 = first); `rank` in the code holds one per path level |
| front | the lines whose next token can be written now |
| move | (father, line or new, token), or STOP |
| mask | the random partial tree at the start of a training sequence |
| real point | the step right after the mask |
| `last` | a node's latest entry; its hidden state stands for the node in the heads |
