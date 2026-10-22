# How treecodegen works

This file explains the whole system step by step, with one running example. The README covers how to use it.

**The idea in short.** A prompt goes in, Python comes out. The output is not written as a flat string, but as a
tree of lines. At each step the model picks *where* to write (a line in the tree) and *what* to write (one token).
It may work on several lines in any order: start a function header, jump into its body, come back to finish the header,
or add a line *above* lines already written (a missing import, a variable set up before the loop that uses it).

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

Every token written is one **entry**. An entry has seven fields:

| Field | Meaning |
|---|---|
| self ID | the ID of the line this token belongs to |
| father ID | the ID of that line's father |
| ancestors | the IDs of the grandfather, great-grandfather, ... (up to 4, nearest first) |
| neighbours | the IDs of the line's two left siblings and its right sibling, as they are when the entry is written |
| token | the token itself (the last token of every line is `<end>`) |
| pos | its position inside the line (0, 1, 2, ...) |
| depth | how deep its line is in the tree (root 0, its children 1, ...) |

**IDs are random symbols.** Every node gets a random number from a pool (`--n-ids`, 512 by default), drawn fresh for
every training example. An ID means nothing by itself; it only says "same node" or "different node". This way the
model can't memorize that "node 7 is always a `return`".

**Neighbours give the order the IDs can't.** `c = a + b` and `return c` have the same father, random IDs,
and positions that both start at 0. Without more, swapping the two lines would give the same entries. So each entry
also carries its line's neighbours among the siblings written so far: left 1, left 2 and right 1 (EDGE past the
first or last one, written `|` below; `–` = none). `return c` alone has neighbours `| – |`; once `c = a + b` is put
before it, `c`'s entries say `| – 33`: nothing on the left, `return` (ID 33) on the right.

The neighbours are a **snapshot**: they describe the line when that entry is written. When a line is later inserted
next to it, the line's *next* entries show the new neighbour, but entries already written never change. So the
decoder's earlier states stay valid (a cache of them would never need recomputing), and the final order is simply
the result of all the insertions.

The model only chooses the token and the place (father + line, and for a new line where it goes among its
siblings). The program fills in the rest of the entry: the IDs, the ancestors, the neighbours, the position and the
depth all follow from the place.

**Every line ends with `<end>`.** It is written like any other token, as the line's last entry. After it the line
is **closed**: it can't be extended, but it can still be a father and get children. So the model can see which
lines are finished, instead of guessing from their content. The end of the whole tree is a separate move, STOP.

Here is one real training sequence for the example (from `python -m treecodegen show --code ...`):

| entry | self | father | ancestors | L1 L2 R1 | pos | depth | token | line |
|---:|---:|---:|---|---|---:|---:|---|---|
| 0 | 49 | – | [] | – – – | 0 | 0 | `<root>` | root |
| 1 | 53 | 49 | [] | \| – \| | 0 | 1 | `def` | def |
| 2 | 33 | 53 | [49] | \| – \| | 0 | 2 | `return` | ret (the only line under def so far) |
| 3 | 33 | 53 | [49] | \| – \| | 1 | 2 | `c` | ret |
| 4 | 53 | 49 | [] | \| – \| | 1 | 1 | `add` | def |
| 5 | 33 | 53 | [49] | \| – \| | 2 | 2 | `<end>` | ret, now closed |
| 6 | 53 | 49 | [] | \| – \| | 2 | 1 | `(` | def |
| 7 | 53 | 49 | [] | \| – \| | 3 | 1 | `a` | def |
| 8 | 53 | 49 | [] | \| – \| | 4 | 1 | `,` | def ← **real point** |
| 9 | 53 | 49 | [] | \| – \| | 5 | 1 | `b` | def |
| 10 | 53 | 49 | [] | \| – \| | 6 | 1 | `)` | def |
| 11 | 53 | 49 | [] | \| – \| | 7 | 1 | `:` | def |
| 12 | 5 | 53 | [49] | \| – 33 | 0 | 2 | `c` | c, **inserted before** ret |
| 13 | 5 | 53 | [49] | \| – 33 | 1 | 2 | `=` | c |
| 14 | 53 | 49 | [] | \| – \| | 8 | 1 | `<end>` | def, now closed |
| 15 | 5 | 53 | [49] | \| – 33 | 2 | 2 | `a` | c |
| 16 | 5 | 53 | [49] | \| – 33 | 3 | 2 | `+` | c |
| 17 | 5 | 53 | [49] | \| – 33 | 4 | 2 | `b` | c |
| 18 | 5 | 53 | [49] | \| – 33 | 5 | 2 | `<end>` | c, now closed |

The lines are interleaved, and written out of order: `return c` is written and closed first (entries 2, 3, 5);
`c = a + b` is started later (entry 12) and placed *before* it, so its entries show `return` (33) as their
right neighbour. The header `def` is unfinished when the body starts, and it gets its `<end>` (entry 14) while
`c` is still open. The IDs put every token back in its line, the positions put it in place inside the line,
and the insertions put the lines in order. So the tree is the same however the entries are ordered.

---

## 3. Inside an entry: the 896 numbers

Each entry becomes one vector of `--d-model` = 896 numbers, in two parts:

| Part | Size | Made from |
|---|---:|---|
| self slot | 64 | ID code of self |
| father slot | 64 | ID code of father |
| 4 ancestor slots | 4 × 64 | ID codes of grandfather, great-grandfather, ... (zeros if there is none) |
| left 1, left 2 slots | 2 × 64 | ID codes of the two nearest left siblings (EDGE past the first one, then zeros) |
| right 1 slot | 64 | ID code of the right sibling (EDGE past the last one) |
| depth slot | 32 | depth embedding (`--k-depth`) |
| **path total** | **608** | then a LayerNorm |
| content | 288 | token embedding + position embedding, then a LayerNorm |

- **The ID table** holds one fixed random unit vector of 64 numbers per ID. It is never trained, and it is the same
  table for every role: node 53 has the same code whether it appears as self, father, ancestor or neighbour.
  Two special extra rows: "no node" (the root's father) and EDGE (past the first or last sibling).
- **Each ancestor level has its own slot**, so the slot tells the level. Ancestors deeper than 4 levels are left out.
- **The neighbour slots give the sibling order.** Two left and one right: the left ones say what comes before (the
  local order), the right one what the line was put in front of, which is often the reason it was inserted.
- **The depth has its own slot in the path**, learned, one vector per depth. The ID slots show the depth only while
  they reach the root (depth ≤ 5); deeper, all slots are full and depth 6 looks like depth 20. The depth slot keeps
  that information. Depths from `--max-depth` − 1 (15) up share one vector. It is part of the path because it says
  where the node is, like the other slots, not what is written there.
- **There is no sequence position.** The decoder doesn't know "this is entry 12". Order comes from the neighbours
  (between lines) and the position inside each line, which is why any interleaving of the lines means the same tree.

A common way to give a transformer positions in a tree is each node's path of child numbers from the root (as in
Shiv & Quirk, *Novel positional encodings to enable tree-based transformers*, 2019). Child numbers would change when
a line is inserted before others, so here the path holds IDs only: ancestors for the nesting, neighbours for the
order. The IDs also let the model point back at a line to continue it, or at a sibling to insert next to it.

Code: `NodeEmbedding` in [model.py](treecodegen/model.py).

---

## 4. The model, layer by layer

```
prompt ──► encoder (BERT) ──► memory ──────────────┐
                                                   ▼ (cross-attention)
entries ──► NodeEmbedding ──► 4 decoder layers ──► H ──► four heads ──► next move
```

| Layer | What it does |
|---|---|
| **Prompt encoder** | A BERT. Pretrained (`--bert bert-base-uncased`) or a small one trained from scratch (128 wide, 2 layers). Turns the prompt into one vector per prompt token: the *memory*. |
| **mem_proj** | A linear layer that maps the memory to the decoder's width (896). |
| **NodeEmbedding** | Turns each entry into its 896 numbers (section 3). |
| **Decoder layer** ×4 | Each layer: self-attention over the earlier entries (causal: entry t sees only 0..t), then cross-attention to the prompt memory, then a feed-forward block (896 → 3584 → 896). 4 heads, pre-norm. |
| **Final LayerNorm** | Gives H: one hidden state per entry. H[t] is used to predict the move after entry t. |
| **Father head** | `fa_q` turns H[t] into a 64-number query. Each existing node's score = query · its ID code. One extra score (`stop`) for STOP. Softmax over all of them. |
| **fa_cond** | Adds the chosen father's state to H[t]: u = H[t] + fa_cond(state of the father's latest entry). |
| **Line head** | `id_q` turns u into a query. Each child of the father gets query · its ID code. One extra score (`new`) for a new line. Softmax. |
| **line_cond / new_line** | Adds the chosen line's state (its latest entry) to u, or a learned "new line" vector. |
| **Place head** | For a new line only. `place_q` turns u + new_line into a query. Each existing child of the father gets query · its ID code ("go right after it"). One extra score (`first`) for "go before all of them". Softmax. |
| **place_cond / first_line** | For a new line, adds the state of the child it goes after (its latest entry), or a learned "first" vector. |
| **Token head** | LayerNorm + linear layer over the code vocabulary. |

The father, line and place heads compare against the **same ID table** the input uses. The model doesn't learn
"choose node 53"; it learns "choose the node whose code looks like this", which works for any random ID.

With default settings (4 layers) and a 40,000-token vocabulary the decoder has about 89M parameters; the
pretraining run in the README (7 layers, 8 heads) has about 140M.

---

## 5. One step: up to four choices

After entry t, the next move is chosen in up to four steps. Each step sees the earlier choices:

| # | Choice | Options | Masked out |
|---|---|---|---|
| 1 | **Father** | any node that exists, or STOP | nodes not written yet |
| 2 | **Line** | an open child of that father (extend its line), or a new line | nodes that aren't children of that father, closed lines |
| 3 | **Place** (new line only) | right after one of the father's children, or first | nodes that aren't children of that father |
| 4 | **Token** | any code token | – |

The token goes **at the end** of the chosen line. The probability of a move is the product of the choices:

```
P(move) = P(father) × P(line | father) × [new line: P(place | father)] × P(token | father, line, place)
```

What the choice means:

- **existing line** → the token is appended to that line; if it is `<end>`, the line is closed;
- **new line** → a new child of the father, with a fresh unused ID, put right after the chosen sibling (or first);
  the token is its first one. Lines already written keep their entries; only their *next* entries will show the
  new neighbour;
- **STOP** → the tree is done.

Because of the masks, impossible moves can't be chosen. A line that exists under a different father can't be picked.
A grandfather is never chosen at all: it follows from the father.

---

## 6. Training

### 6.1 Building a training sequence

For each tree, one training sequence is built in three parts ([sequences.py](treecodegen/sequences.py), `flatten`):

1. **The mask: a random partial tree.** Starting at the root, each node keeps a random subset of its children
   (any of them, not only the first ones), and each kept child keeps a random non-empty start of its line. In the
   example the mask is `def add ( a ,` (5 of 9 tokens), all of `return c <end>` and nothing of `c = a + b`.
   This is the `subset` mask (`--mask subset`, the default). Its partial trees are small: on real files the median
   one holds 9% of the file, and fewer than 1 in 50 hold more than half. The `holes` mask (`--mask holes`) removes
   a share of the lines drawn uniformly between 0 and 100% (at least one line), as holes: each a random line and up
   to 2 of its next siblings, with their subtrees, only as big as what is left to remove. Every other line is
   complete, except, half the time, one cut short. So it ranges from a file with one line missing to a few lines
   left, always as whole lines with blocks missing. The `deep` mask (`--mask deep`) is a holes mask that removes
   (and cuts) only lines deeper than d, with d = 1 or 2 drawn per sequence: imports, class and def headers (and,
   for d = 2, the lines right under them) are complete, and only bodies have holes. That is the usual completion
   case (write this method, finish this loop), and there a new top-level line is never right, which `holes`
   masks don't teach: they remove top-level lines too, so "add another import" is often valid. A file with no
   line deeper than the top level gets a plain holes mask. `--mask mix` draws `deep` for a share `--deep-share`
   of the sequences, `holes` for a share `--holes-share`, else `subset`.
2. **The mask, written in a random valid order** (entries 1–8). The step after its last entry is the
   **real point** (entry 8 → the move to entry 9).
3. **The rest of the tree, in a random valid order** (entries 9–18).

The real point matters because it matches a situation the model can really find itself in: a half-written tree,
reached by any route. `--real-weight` gives it more weight in the loss.

**Why the holes mask is needed.** Every sequence is written to the end and every step is in the loss, so large
partial trees are trained too, in the later steps. But they have a particular shape: in a random order, lines
are started early and finished late. Halfway through a typical file every line already exists, and what is
missing is the ends of about 14 lines spread around (measured on 300 held-out files: at 75% written, median 0
missing lines and 14 half-written ones). A file with whole lines complete and a block missing (a function body, a
few statements, an import) almost never appears. The `holes` mask puts exactly that state at the real point.

**How interleaved the order is** varies per sequence. Each sequence draws a stickiness s between 0 and 1: at each
step it continues the line it has just written with probability s, and otherwise picks any valid move. So the
orders range from fully interleaved (s = 0) to whole lines one after the other (s = 1). The second kind matters:
greedy generation tends to finish a line before starting the next, and a model trained only on interleaved orders
never sees those states (it then re-starts lines that already exist).

**Line-first sequences** (`--line-share`, 0 by default). Stickiness changes the order, not what counts as valid:
starting a new line stays valid while other lines are open, and since random orders start lines early and finish
them late, "start another line" is a likely move almost everywhere. Free completion shows it: the model opens line
after line and finishes almost none. In a line-first sequence, from the real point on, while any line is open only
continuing an open line is valid (and only those moves are targets); a new line can start only once every started
line is finished. The mask itself is unchanged (it needs lines left open).

### 6.2 Which moves are valid

At any moment several moves are correct. The valid moves are:

- **extend** any open line (started, `<end>` not written yet) with its next token, which is `<end>` once all its
  real tokens are written;
- **start** any unwritten child of any node that is already started, in any order. Its place is right after its
  nearest older sibling already written, or first if none is. Two unwritten siblings that would start with the
  same token at the same place are the same move, counted once;
- **STOP**, but only once the whole tree is written.

At the real point of the example (after entry 8):

| Valid move | Father | Line | Place | Token |
|---|---|---|---|---|
| continue the header | root (49) | def (53) | – | `b` |
| start the `c` line | def (53) | new | first (before `return`) | `c` |

STOP is not valid here: the tree is not complete.

So the freedom is broad:

| | Free? |
|---|---|
| Which father to work under at each step (any existing node) | yes |
| Extending any open line, in any order | yes |
| Interleaving lines, going deep or wide, coming back later | yes |
| Moving to another branch before finishing this one | yes |
| Starting siblings in any order, inserting a line before lines already written | yes |
| Inserting tokens in the middle of a line, moving or deleting lines | no (only appending, only adding) |

### 6.3 The loss

The model shouldn't be punished for picking one correct move over another. So the loss at each step is

```
loss = −log( P(move 1) + P(move 2) + ... )      over the valid moves
```

It is small when the model puts most of its probability on *any* valid move. For example, with invented numbers:

| Valid move | P(father) | P(line \| father) | P(place \| father) | P(token \| ...) | P(move) |
|---|---:|---:|---:|---:|---:|
| root → def → `b` | 0.40 | 0.95 | – | 0.90 | 0.342 |
| def → new, first → `c` | 0.55 | 0.90 | 0.80 | 0.85 | 0.337 |
| **total** | | | | | **0.679** |

Loss at this step = −log(0.679) = 0.39. The remaining 0.26 went to invalid moves (STOP, wrong tokens, ...).

To save compute, up to `--max-moves` (4) valid moves are scored per step: the one actually taken plus random
others. The loss is averaged over all steps of all sequences in the batch.

**The loss has a floor.** A step has many valid moves (median about 33 on real files), and only some are scored.
The model isn't told which, so even a perfect model spreads its probability over all of them, and a step with k
valid moves of which m are scored can't go below log(k / m): about 2.0 on average. So the log also shows a
**floor-adjusted** loss, each step's loss minus its log(k / m): a perfect model brings it to about 0. Training still
uses the plain loss; the adjusted one is only for reading the curve.

### 6.4 STOP happens once per tree

STOP is a valid move exactly once: after the last entry, when every line has its `<end>`. At every other step it
is wrong, so the loss pushes it down at every step and up only at the end. Ending a single line is a different,
much more frequent decision: writing `<end>`, once per line.

### 6.5 Putting it together

For each batch: build one sequence per tree with fresh IDs and a fresh mask, run the encoder on the prompts and the
decoder on the entries, compute the loss, update the weights ([train.py](treecodegen/train.py),
[loss.py](treecodegen/loss.py)). Every time a tree is seen again it gets new IDs, a new mask and a new order.

`--order dfs` trains on one fixed order instead (whole lines, depth-first, no mask), for comparison.

---

## 7. Generation

[generate.py](treecodegen/generate.py) builds the tree one token at a time:

1. Start with the root entry (random ID, `<root>` token).
2. Run the decoder on the new entry. An attention cache keeps every earlier entry's keys and values (they never
   change once written: section 2), so each step costs one entry, not the whole sequence.
3. Score every move: each existing node as father × (each of its open children, or a new line × each place: after
   any of its children, or first) × each token (a new line can't start with `<end>`), plus STOP once every line
   is closed.
4. Pick one: sampled from P(move)^(1/T) with temperature T (`--temperature`, 0.2 by default), or the most likely
   one when T = 0. STOP ends generation. Otherwise write the move: for a new line pick an unused ID and insert it
   among its father's children at the chosen place, then fill in the entry (self, father, ancestors, neighbours,
   position, depth). Writing `<end>` closes the line.
5. Go back to 2.

Example, the start of a generation:

| Step | Chosen move | Result |
|---|---|---|
| 1 | root → new, first → `def` | new node A under the root, line `def` |
| 2 | root → A → `add` | `def add` |
| 3 | A → new, first → `return` | new node C under A, line `return` |
| 4 | root → A → `(` | `def add (` |
| 5 | A → new, first → `c` | new node B under A, put **before** C, line `c` |
| ... | ... | ... |
| | root → A → `<end>` | `def add ( a , b ) :`, closed |
| ... | ... | ... |
| n | STOP | done |

At the end, each node's tokens (without `<end>`) are its line, and siblings are in the order the insertions built.
`tree_to_code` prints it as Python.

---

## 8. Evaluation

The evaluation does what training does: complete a partial tree ([evaluate.py](treecodegen/evaluate.py)).
Generating a whole file from nothing can't be checked against a reference when the prompt is empty (any good
file is as right as the held-out one); completing a given partial tree can.

For each held-out tree:

1. Build a partial tree exactly like a training sequence: a random mask, written in a random valid order, up to
   the real point. The mask kind is `--mask` of `evaluate` (`subset` or `holes`; section 6.1).
2. From there the model makes every remaining move itself, taking its most likely move (`--temperature 0`).
3. Each move is **correct** if it is one of the valid moves at that point (section 6.2): any order is fine.
   After a **wrong** move, the valid move the model scores highest is applied instead, and the completion goes
   on, so one early mistake doesn't decide the rest.
   The model may write `<unk>` here (generation from a prompt can't): a token outside the vocabulary is `<unk>` in
   the reference too, so `<unk>` is the right move there. Before this was allowed (up to the end of pass 2), about
   10% of the required moves could never be right, and the model, unable to write the name it expected, tended to
   start new lines instead of finishing the current one. Scores from before and after are not comparable.

The number of moves required is fixed by the tree: the tokens still to write (each line's `<end>` included)
plus one STOP. Reported:

- **moves correct**: correct moves / required moves, over all files;
- **files with no wrong move**: the share of files completed perfectly;
- **validation loss**: the training loss on the same held-out files (random masks and orders, seeded), directly
  comparable with the training loss; a gap means overfitting.

At every `--eval-every` steps training reports these on `--eval-n` held-out files (up to 1,000 entries each); a
run with `--mask holes` or `mix` reports them twice, with `subset` and with `holes` masks, so a gain on one and a
loss on the other both show. Plus one example (with a `holes` mask in such a run): a held-out partial tree, the model's own completion of it with no corrections (at most 1,000 moves),
and the real file. `python -m treecodegen evaluate` does the
same for a saved checkpoint (it finds the held-out files the way training split them).

Earlier versions reported exact match and parse rate of trees generated from an empty start (on the small toy
dataset: 3% exact, 56% parse). Those numbers predate `<end>`, free sibling order and this evaluation.

---

## 9. Settings that shape the model

| Flag | Default | What it sets |
|---|---:|---|
| `--d-model` | 896 | entry width = path (608) + content (288) |
| `--k-slot` | 64 | size of one ID code |
| `--max-anc` | 4 | ancestor levels in the path; path = (2 + max-anc + 3) × k-slot + k-depth |
| `--n-ids` | 512 | size of the ID pool; must be ≥ the number of nodes in a tree |
| `--max-pos` | 128 | longest line, in tokens |
| `--max-depth` | 16 | depth embedding rows; deeper lines share the last one |
| `--k-depth` | 32 | size of the depth slot in the path |
| `--min-freq`, `--max-vocab` | 1, none | code tokens kept: seen at least min-freq times, and at most max-vocab of the most frequent; the rest become `<unk>` |
| `--layers`, `--heads` | 4, 4 | decoder depth and attention heads |
| `--order` | random | `random` (mask + random order) or `dfs` (fixed order) |
| `--line-share` | 0 | share of line-first sequences: past the real point, open lines must be finished before a new one starts |
| `--mask`, `--holes-share`, `--deep-share` | subset, 0.7, 0 | the mask kind: `subset` (small partial trees), `holes` (whole lines with blocks missing, 0–100% of the lines), `deep` (holes only inside bodies) or `mix` (`deep` for the deep share of sequences, `holes` for the holes share, else `subset`) |
| `--max-moves` | 4 | valid moves scored per step |
| `--batch`, `--accum` | 32, 1 | sequences per batch; batches added up into one step (gradient accumulation) |
| `--real-weight` | 1.0 | loss weight of the real point |

---

## 10. Words used in the code

| Word | Meaning |
|---|---|
| node / line | one statement; its tokens are its content |
| entry | one written token, with its IDs, neighbours, position and depth |
| `<end>` | the last token of every line; after it the line is closed |
| ID / ID code | the random symbol of a node / its fixed 64-number vector |
| chain | a node's IDs: self, father, grandfather, ... |
| neighbours (`nb`) | a line's left 1, left 2 and right 1 siblings when an entry is written; EDGE past the ends |
| place | where a new line goes: right after one of its father's children, or first |
| `rank` (in `flatten`) | a node's real place among its siblings, used only to build training targets |
| front | the lines whose next token can be written now |
| move | (father, line, token), (father, new line, place, token), or STOP |
| mask | the random partial tree at the start of a training sequence |
| real point | the step right after the mask |
| `last` | a node's latest entry; its hidden state stands for the node in the heads |
