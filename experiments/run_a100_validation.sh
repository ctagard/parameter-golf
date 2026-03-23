#!/bin/bash
# A100 LoRA Ablation — runs 3 and 4 only (1 and 2 already done)
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
echo "Runs 3-4 complete. Compare with existing logs:"
echo "  logs/a100_true_baseline.txt"
echo "  logs/a100_no_lora.txt"
echo "  logs/a100_lora32.txt"
echo "  logs/a100_lora32_ortho.txt"
echo "=========================================="
