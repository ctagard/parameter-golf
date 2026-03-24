#!/bin/bash
# H100 setup: download dataset + install deps
# Usage: cd /workspace/parameter-golf && bash experiments/setup_h100.sh
set -e
pip install wandb zstandard huggingface-hub
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
echo "Setup complete. Dataset + deps ready."
