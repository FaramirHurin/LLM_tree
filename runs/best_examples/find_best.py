# Deep-mask evaluation of a pass-3 checkpoint, as in training (same files, seeds and masks), recording each file's
# score; then the best files completed by the model alone (greedy, no corrections) from the same partial tree.
import random, sys, json
import torch
torch.set_num_threads(16)
from treecodegen.train import load_checkpoint, EVAL_MAX_ENTRIES
from treecodegen.evaluate import heldout_rows, fits, complete, free_completion
from treecodegen.sequences import tree_arrays
from treecodegen.tree import tree_to_code
ckpt, out = sys.argv[1], sys.argv[2]
enc, dec, tok, vocab, cfg = load_checkpoint(ckpt, "cpu")
rows = heldout_rows("Datasets/pretrain.jsonl", 50)
rng, gen = random.Random(1), torch.Generator("cpu").manual_seed(1)
res = []
for i, (prompt, tree) in enumerate(rows):
    arrays = tree_arrays(tree, vocab)
    if not fits(arrays, cfg) or sum(len(c) for c in arrays[0]) + 1 > EVAL_MAX_ENTRIES:
        continue
    state = rng.getstate()
    c, r = complete(enc, dec, tok, vocab, cfg, prompt, tree, rng, "cpu", 0.0, gen, "deep")
    res.append(dict(i=i, correct=c, required=r, rng=state))
    print(f"file {i}: {c}/{r}", flush=True)
tot_c, tot_r = sum(x["correct"] for x in res), sum(x["required"] for x in res)
print(f"overall {tot_c/tot_r:.1%} of moves correct on {len(res)} files; perfect: {[x['i'] for x in res if x['correct'] == x['required']]}")
best = sorted(res, key=lambda x: (x["correct"] / x["required"], x["required"]), reverse=True)[:4]
examples = []
for x in best:
    r = random.Random(); r.setstate(x["rng"])
    partial, done, ref = free_completion(enc, dec, tok, vocab, cfg, *rows[x["i"]], r, "cpu", 0.0, max_moves=200, mask="deep")
    examples.append(dict(i=x["i"], correct=x["correct"], required=x["required"], partial=partial, completion=done, real=ref))
    print(f"\n######## file {x['i']}: {x['correct']}/{x['required']} moves correct with corrections")
    for name, code in (("partial tree given", partial), ("model alone (greedy, no corrections)", done), ("real file", ref)):
        print(f"--- {name}\n{code}")
json.dump(examples, open(out, "w"), indent=1)
