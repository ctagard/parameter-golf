#!/bin/bash
set -e
SEED=${SEED:-1337}
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

echo "Running LoopedGPT mHC-lite 3x2 TTT submission, seed=$SEED"
cd "$REPO_ROOT"

RUN_ID="eval_seed${SEED}" \
SEED=$SEED \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 \
  "$SCRIPT_DIR/train_gpt.py" 2>&1 | tee "$SCRIPT_DIR/logs/eval_seed${SEED}.log"

echo "Done. Check logs/eval_seed${SEED}.log for final_ttt_lora val_bpb"
