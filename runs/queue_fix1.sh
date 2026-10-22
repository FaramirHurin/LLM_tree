#!/bin/sh
# Fix 1 (evaluation may write <unk>), queued during pass 2 (2026-10-17). Waits for pass 2 to finish, snapshots the
# pass-2 code, applies runs/fix1_staging (tested), then re-scores pass 1 and pass 2 with the fixed evaluation.
cd /home/daniele/LLM_tree || exit 1
say() { echo "[$(date '+%F %T')] $*"; }
P=.venv/bin/python
say "waiting for pass 2 (PID 3039015)"
while kill -0 3039015 2>/dev/null; do sleep 60; done
if ! grep -q "^saved pretrain_pass2.pt$" pretrain_pass2.log; then
    say "pass 2 did not end with its final save; not applying the fix"; tail -5 pretrain_pass2.log; exit 1
fi
say "pass 2 finished: $(grep '^== step' pretrain_pass2.log | tail -2 | tr '\n' ' ')"
tar czf runs/pass2_code.tar.gz treecodegen/*.py tests/*.py smoke_train.py README.md EXPLANATION.md \
    && say "saved pass 2's code as runs/pass2_code.tar.gz"
cp runs/fix1_staging/treecodegen/*.py treecodegen/ && cp runs/fix1_staging/tests/test_treecodegen.py tests/ \
    && cp runs/fix1_staging/README.md runs/fix1_staging/EXPLANATION.md . && say "applied fix 1"
say "running the tests"
if ! CUDA_VISIBLE_DEVICES= $P -c "
import tests.test_treecodegen as T
for k in dir(T):
    if k.startswith('test_'): getattr(T, k)(); print('ok', k, flush=True)
"; then say "tests failed; not evaluating"; exit 1; fi
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for c in pass1:pretrain.log pass2:pretrain_pass2.log; do
    name=${c%%:*}; log=${c#*:}
    for m in subset holes; do
        say "evaluating $name with $m masks -> runs/eval_fixed_${name}_$m.txt"
        $P -m treecodegen evaluate --ckpt pretrain_$name.pt --data Datasets/pretrain.jsonl --n 50 --max-entries 1000 \
            --mask $m --train-log $log --report runs/eval_fixed_${name}_$m.txt > /dev/null 2>&1 || say "evaluation failed"
        say "  $(grep 'moves correct' runs/eval_fixed_${name}_$m.txt)"
    done
done
say "done"
