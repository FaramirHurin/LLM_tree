#!/bin/sh
# Pass 3 (2026-10-18): baseline of pass 2 with deep masks, then pass 3. Code: runs/pass3_code.tar.gz.
cd /home/daniele/LLM_tree || exit 1
say() { echo "[$(date '+%F %T')] $*"; }
export CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=.venv/bin/python
say "evaluating pass 2 with deep masks -> runs/eval_fixed_pass2_deep.txt"
$P -m treecodegen evaluate --ckpt pretrain_pass2.pt --data Datasets/pretrain.jsonl --n 50 --max-entries 1000 \
    --mask deep --train-log pretrain_pass2.log --report runs/eval_fixed_pass2_deep.txt > /dev/null 2>&1 || say "evaluation failed"
say "  $(grep 'moves correct' runs/eval_fixed_pass2_deep.txt)"
say "starting pass 3 -> pretrain_pass3.log"
setsid nohup $P -m treecodegen train --data Datasets/pretrain.jsonl --out pretrain_pass3.pt --resume pretrain_pass2.pt \
    --mask mix --deep-share 0.2 --holes-share 0.55 --line-share 0.3 --batch 8 --accum 2 --steps 55000 --lr 1.5e-4 \
    --warmup 1000 --eval-every 5000 --eval-n 50 > pretrain_pass3.log 2>&1 < /dev/null &
say "pass 3 started (PID $!)"
