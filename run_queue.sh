#!/bin/bash
# Waits for the dataset download (load_datasets.py) to finish, then starts the full pretraining run on GPU 1.
cd /home/daniele/LLM_tree
echo "[$(date +%T)] waiting for the download to finish"
while pgrep -f "python load_datasets.py" > /dev/null; do sleep 60; done
if ! grep -q "^done: kept" nohup.out; then
    echo "[$(date +%T)] download stopped without its 'done' line: not starting the run"; exit 1
fi
echo "[$(date +%T)] download done: $(tail -1 nohup.out)"
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -gt 2000 ]; do
    echo "[$(date +%T)] GPU 1 busy, waiting"; sleep 60
done
echo "[$(date +%T)] starting the run"
CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python -m treecodegen train \
  --data Datasets/pretrain.jsonl --out pretrain.pt --n-ids 1024 --max-vocab 40000 \
  --layers 7 --heads 8 --batch 8 --steps 110000 --lr 3e-4 --warmup 1000 \
  --eval-every 10000 --eval-n 10 > pretrain.log 2>&1
echo "[$(date +%T)] run ended with exit code $?"
