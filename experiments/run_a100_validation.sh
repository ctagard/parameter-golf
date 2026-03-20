#!/bin/bash
# A100 Hypernetwork Ablation + Table-Stakes Validation
# Run on a single A100 80GB on RunPod
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
echo "Run 1/2: No hypernetwork (table-stakes baseline)"
echo "=========================================="
env $COMMON \
  RUN_ID=no_hyper_tablestakes SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "Run 2/2: Hypernetwork Variant C (coarse stride=4)"
echo "=========================================="
env $COMMON \
  HYPERNET_VARIANT=coarse \
  RUN_ID=hyper_coarse_tablestakes SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py

echo ""
echo "=========================================="
echo "Both runs complete. Check logs/ and WandB for results."
echo "Compare val_bpb at steps 100/200/300 between the two runs."
echo "=========================================="
