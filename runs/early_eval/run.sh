#!/bin/sh
# Fixed evaluation (<unk> allowed) run early, on CPU, with the code in runs/fix1_staging (run from inside it, so
# python -m picks it up); the repo and the GPU run are untouched.
R=/home/daniele/LLM_tree
cd $R/runs/fix1_staging || exit 1
export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16
$R/.venv/bin/python -c "
import treecodegen, inspect, treecodegen.generate as g
assert 'fix1_staging' in treecodegen.__file__ and 'allow_unk' in inspect.signature(g.score_moves).parameters
print('code:', treecodegen.__file__)" || exit 1
for c in pass1:$R/pretrain_pass1.pt:$R/pretrain.log pass2_step25000:$R/runs/early_eval/pass2_step25000.pt:$R/pretrain_pass2.log; do
    name=${c%%:*}; rest=${c#*:}; ckpt=${rest%%:*}; log=${rest#*:}
    for m in subset holes; do
        echo "[$(date '+%T')] $name, $m masks"
        $R/.venv/bin/python -m treecodegen evaluate --ckpt $ckpt --data $R/Datasets/pretrain.jsonl --n 50 \
            --max-entries 1000 --mask $m --device cpu --train-log $log --report $R/runs/early_eval/${name}_$m.txt \
            > $R/runs/early_eval/${name}_$m.progress 2>&1
        grep "moves correct\|no wrong move" $R/runs/early_eval/${name}_$m.txt
    done
done
echo "[$(date '+%T')] done"
