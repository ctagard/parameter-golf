#!/bin/bash
# A100 4-Way LoRA Ablation + Baseline Comparison
# 1000 steps each, dim=768, all bells and whistles
#
# Usage:
#   cd /workspace/parameter-golf && git pull
#   pip install wandb zstandard huggingface-hub
#   python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
#   bash experiments/run_a100_validation.sh

set -e

COMMON="MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=768 \
NUM_HEADS=12 NUM_KV_HEADS=4 MLP_TYPE=swiglu MLP_MULT=3 \
MHC_STREAMS=4 \
BIGRAM_VOCAB_SIZE=4096 MUON_WD=0.04 ADAM_WD=0.04 \
TRAIN_BATCH_TOKENS=131072 MATRIX_LR=0.035 SCALAR_LR=0.035 TIED_EMBED_LR=0.02 \
ITERATIONS=1000 VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=20 \
WANDB_PROJECT=parameter-golf"

echo "=========================================="
echo "Run 1/4: TRUE BASELINE (original GPT, no mods)"
echo "=========================================="
MODEL_FAMILY=baseline NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 \
  MLP_MULT=2 MLP_TYPE=relu2 \
  TRAIN_BATCH_TOKENS=131072 MATRIX_LR=0.04 SCALAR_LR=0.04 TIED_EMBED_LR=0.05 \
  MUON_WD=0.0 ADAM_WD=0.0 BIGRAM_VOCAB_SIZE=0 \
  ITERATIONS=1000 VAL_LOSS_EVERY=200 TRAIN_LOG_EVERY=50 \
  MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=20 \
  WANDB_PROJECT=parameter-golf \
  RUN_ID=a100_true_baseline SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "Run 2/4: Our arch, NO LoRA (table-stakes only)"
echo "=========================================="
env $COMMON LORA_RANK=0 \
  RUN_ID=a100_no_lora SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "Run 3/4: LoRA rank=32 standard init (A freeze + B low LR)"
echo "=========================================="
env $COMMON LORA_RANK=32 LORA_WARMUP_STEPS=50 \
  RUN_ID=a100_lora32 SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "Run 4/4: LoRA rank=32 ortho SVD init (A freeze + B low LR)"
echo "=========================================="
env $COMMON LORA_RANK=32 ORTHO_LORA=1 LORA_WARMUP_STEPS=50 \
  RUN_ID=a100_lora32_ortho SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "All 4 runs complete. Compare val_bpb at steps 200/400/600/800/1000."
echo "=========================================="
