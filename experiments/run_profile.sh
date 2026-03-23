#!/bin/bash
# Profile graph breaks + CUDA performance for looped + LoRA
#
# Usage:
#   cd /workspace/parameter-golf && git pull
#   bash experiments/run_profile.sh

set -e

COMMON="MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=768 \
NUM_HEADS=12 NUM_KV_HEADS=4 MLP_TYPE=swiglu MLP_MULT=3 \
LORA_RANK=32 MHC_STREAMS=4 \
BIGRAM_VOCAB_SIZE=4096 MUON_WD=0.04 ADAM_WD=0.04 \
TRAIN_BATCH_TOKENS=65536 MATRIX_LR=0.035 \
MAX_WALLCLOCK_SECONDS=0 WARMUP_STEPS=5"

echo "=========================================="
echo "Step 1: Graph break analysis (5 iterations)"
echo "=========================================="
TORCH_LOGS=graph_breaks \
env $COMMON ITERATIONS=5 TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=0 \
  RUN_ID=profile_graph_breaks SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/graph_breaks.log

echo ""
echo "=========================================="
echo "Step 2: CUDA profiler (single step after warmup)"
echo "=========================================="
env $COMMON ITERATIONS=10 TRAIN_LOG_EVERY=1 VAL_LOSS_EVERY=0 \
  PROFILE_STEP=0 \
  RUN_ID=profile_cuda SEED=1337 \
  torchrun --standalone --nproc_per_node=1 train_gpt.py 2>&1 | tee logs/profile_cuda.log

echo ""
echo ""
echo "=========================================="
echo "Step 3: Detailed per-component profiling"
echo "=========================================="
torchrun --standalone --nproc_per_node=1 experiments/detailed_profile.py 2>&1 | tee logs/detailed_profile.log

echo ""
echo "=========================================="
echo "Profile complete. Files:"
echo "  logs/graph_breaks.log      — compile breaks"
echo "  logs/profile_cuda.log      — CUDA key averages"
echo "  profile_trace.json         — chrome://tracing"
echo "  logs/detailed_profile.log  — per-component timing"
echo "=========================================="
