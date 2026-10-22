#!/bin/sh
# Pass 2, queued during pass 1 (2026-10-16). Waits for pass 1 to finish, keeps its checkpoint, applies the pass-2
# code (runs/pass2_staging, tested), evaluates pass 1 with both mask kinds as the baseline, then starts pass 2.
# Nothing in the repo changes while pass 1 runs. Pass 1's exact code is in runs/pass1_code.tar.gz.
cd /home/daniele/LLM_tree || exit 1
say() { echo "[$(date '+%F %T')] $*"; }
P=.venv/bin/python
say "waiting for pass 1 (PID 3405046)"
while kill -0 3405046 2>/dev/null; do sleep 60; done
if ! grep -q "^saved pretrain.pt$" pretrain.log; then
    say "pass 1 did not end with its final save; not starting pass 2"; tail -5 pretrain.log; exit 1
fi
say "pass 1 finished: $(grep '^== step' pretrain.log | tail -1)"
cp pretrain.pt pretrain_pass1.pt && say "kept pass 1's checkpoint as pretrain_pass1.pt"
cp runs/pass2_staging/treecodegen/*.py treecodegen/ && cp runs/pass2_staging/tests/test_treecodegen.py tests/ \
    && cp runs/pass2_staging/README.md runs/pass2_staging/EXPLANATION.md . && say "applied the pass-2 code and docs"
say "running the tests"
if ! CUDA_VISIBLE_DEVICES= $P -c "
import tests.test_treecodegen as T
for k in dir(T):
    if k.startswith('test_'): getattr(T, k)(); print('ok', k, flush=True)
"; then say "tests failed; not starting pass 2"; exit 1; fi
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for m in subset holes; do
    say "evaluating pass 1 with $m masks -> runs/pass1_eval_$m.txt"
    $P -m treecodegen evaluate --ckpt pretrain_pass1.pt --data Datasets/pretrain.jsonl --n 50 --max-entries 1000 \
        --mask $m --train-log pretrain.log --report runs/pass1_eval_$m.txt > /dev/null 2>&1 || say "evaluation failed"
done
say "starting pass 2 -> pretrain_pass2.log"
setsid nohup $P -m treecodegen train --data Datasets/pretrain.jsonl --out pretrain_pass2.pt --resume pretrain_pass1.pt \
    --mask mix --holes-share 0.7 --line-share 0.3 --batch 8 --accum 2 --steps 55000 --lr 1.5e-4 --warmup 1000 \
    --eval-every 5000 --eval-n 50 > pretrain_pass2.log 2>&1 < /dev/null &
say "pass 2 started (PID $!)"
