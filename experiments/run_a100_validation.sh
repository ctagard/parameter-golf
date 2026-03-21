#!/bin/bash
# 8×H100 Submission Run — Parameter Golf
#
# Prerequisites:
#   cd /workspace/parameter-golf && git pull
#   pip install wandb zstandard

set -e

COMMON="MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=768 \
NUM_HEADS=12 NUM_KV_HEADS=4 MLP_TYPE=swiglu MLP_MULT=3 \
LORA_RANK=32 MHC_STREAMS=4 \
BIGRAM_VOCAB_SIZE=4096 MUON_WD=0.04 ADAM_WD=0.04 \
TRAIN_BATCH_TOKENS=524288 MATRIX_LR=0.10 \
WARMDOWN_ITERS=500 ITERATIONS=20000 MAX_WALLCLOCK_SECONDS=600 \
QUANT_BITS=6 EVAL_STRIDE=64 TTT_LORA_RANK=4 TTT_STEPS=3 \
WANDB_PROJECT=parameter-golf"

echo "=========================================="
echo "Seed 1/3: 1337"
echo "=========================================="
env $COMMON RUN_ID=h100_seed1337 SEED=1337 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=========================================="
echo "Seed 2/3: 42"
echo "=========================================="
env $COMMON RUN_ID=h100_seed42 SEED=42 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=========================================="
echo "Seed 3/3: 7"
echo "=========================================="
env $COMMON RUN_ID=h100_seed7 SEED=7 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py

echo ""
echo "=========================================="
echo "All 3 seeds complete. Check logs/ for:"
echo "  - final_int8_zlib_roundtrip (or int6+zstd)"
echo "  - sliding_window_eval"
echo "  - ttt_eval"
echo "=========================================="
