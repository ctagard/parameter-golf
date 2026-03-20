#!/bin/bash
# A100 Validation Runs for Parameter Golf
# Run on a single A100 80GB on RunPod
# Cost estimate: ~$2-3 total for both runs
#
# Prerequisites:
#   cd /workspace
#   git clone <your-fork> parameter-golf && cd parameter-golf
#   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 10
#   pip install wandb  # optional

set -e

echo "=========================================="
echo "A100 Validation: Looped + SwiGLU + LoRA + mHC"
echo "=========================================="

COMMON="MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=512 \
NUM_HEADS=8 NUM_KV_HEADS=4 MLP_TYPE=swiglu LORA_RANK=32 MHC_STREAMS=4 \
ITERATIONS=500 MAX_WALLCLOCK_SECONDS=0 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=10 \
WARMUP_STEPS=20 WANDB_PROJECT=parameter-golf"

echo ""
echo "=== Run 1/2: batch=65536 lr=0.08 ==="
env $COMMON \
  TRAIN_BATCH_TOKENS=65536 MATRIX_LR=0.08 \
  RUN_ID=a100_val_batch65k_lr008 SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=== Run 2/2: batch=131072 lr=0.10 ==="
env $COMMON \
  TRAIN_BATCH_TOKENS=131072 MATRIX_LR=0.10 \
  RUN_ID=a100_val_batch131k_lr010 SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "Both runs complete. Check logs/ for results."
echo "Look for: val_bpb, final_int8_zlib_roundtrip, compressed size"
echo "=========================================="
