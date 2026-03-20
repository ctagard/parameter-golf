#!/bin/bash
# A100 Hypernetwork Ablation: delta-from-origin vs EMA
#
# Prerequisites:
#   cd /workspace/parameter-golf && git pull
#   pip install wandb zstandard

set -e

COMMON="MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=768 \
NUM_HEADS=12 NUM_KV_HEADS=4 MLP_TYPE=swiglu \
LORA_RANK=32 MHC_STREAMS=4 \
BIGRAM_VOCAB_SIZE=4096 MUON_WD=0.04 ADAM_WD=0.04 \
TRAIN_BATCH_TOKENS=131072 MATRIX_LR=0.10 \
ITERATIONS=300 VAL_LOSS_EVERY=100 TRAIN_LOG_EVERY=10 \
MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=20 \
WANDB_PROJECT=parameter-golf"

echo "=========================================="
echo "Run 1/2: Hypernetwork coarse (delta-from-origin)"
echo "=========================================="
env $COMMON \
  HYPERNET_VARIANT=coarse \
  RUN_ID=hyper_coarse_delta SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "Run 2/2: Hypernetwork EMA (trajectory across loops)"
echo "=========================================="
env $COMMON \
  HYPERNET_VARIANT=ema \
  RUN_ID=hyper_ema_trajectory SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "Both runs complete. Compare val_bpb at steps 100/200/300"
echo "against no_hyper_tablestakes baseline (1.8140 @ step 300)."
echo "=========================================="
