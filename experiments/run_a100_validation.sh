#!/bin/bash
# 8×H100 Submission Run — Parameter Golf
#
# Usage:
#   cd /workspace/parameter-golf
#   bash experiments/run_a100_validation.sh

set -e

echo "=========================================="
echo "Step 0: Setup"
echo "=========================================="
pip install wandb zstandard huggingface-hub 2>/dev/null
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
echo "Dataset download complete."

COMMON="MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=1024 \
NUM_HEADS=16 NUM_KV_HEADS=4 MLP_TYPE=swiglu MLP_MULT=3 \
LORA_RANK=32 MHC_STREAMS=4 \
BIGRAM_VOCAB_SIZE=4096 MUON_WD=0.04 ADAM_WD=0.04 \
TRAIN_BATCH_TOKENS=524288 MATRIX_LR=0.035 SCALAR_LR=0.035 TIED_EMBED_LR=0.02 \
WARMDOWN_ITERS=800 ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=0 TRAIN_LOG_EVERY=200 \
QUANT_BITS=6 SWA_ENABLED=1 \
EVAL_STRIDE=64 TTT_LORA_RANK=4 TTT_STEPS=3 \
WANDB_PROJECT=parameter-golf"

echo ""
echo "=========================================="
echo "Seed 1/3: 1337"
echo "=========================================="
env $COMMON RUN_ID=h100_d1024_seed1337 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=========================================="
echo "Seed 2/3: 42"
echo "=========================================="
env $COMMON RUN_ID=h100_d1024_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=========================================="
echo "Seed 3/3: 7"
echo "=========================================="
env $COMMON RUN_ID=h100_d1024_seed7 SEED=7 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=========================================="
echo "All 3 seeds complete. Check logs/ for:"
echo "  - final int6+zstd roundtrip"
echo "  - sliding_window_eval"
echo "  - ttt_eval"
echo "=========================================="
