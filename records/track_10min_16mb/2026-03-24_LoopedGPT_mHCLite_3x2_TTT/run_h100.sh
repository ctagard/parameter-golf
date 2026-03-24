#!/bin/bash
# Full H100 submission: setup + 3 seeds. ~90 min total.
# Usage: cd /workspace && git clone ... && cd parameter-golf && bash records/track_10min_16mb/2026-03-24_LoopedGPT_mHCLite_3x2_TTT/run_h100.sh
set -e

SUBDIR="records/track_10min_16mb/2026-03-24_LoopedGPT_mHCLite_3x2_TTT"
SCRIPT="$SUBDIR/train_gpt.py"
mkdir -p "$SUBDIR/logs"

echo "=== SETUP ==="
pip install wandb zstandard huggingface-hub -q
wandb login
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
echo "Setup done."

for SEED in 1337 42 2024; do
  echo ""
  echo "=========================================="
  echo "SEED $SEED — starting $(date)"
  echo "=========================================="
  RUN_ID="h100_seed${SEED}" \
  SEED=$SEED \
  WANDB_PROJECT=parameter-golf \
  torchrun --standalone --nproc_per_node=8 "$SCRIPT" \
    2>&1 | tee "$SUBDIR/logs/train_seed${SEED}.log"
  echo "SEED $SEED done $(date)"
done

echo ""
echo "=========================================="
echo "ALL 3 SEEDS COMPLETE"
echo "=========================================="
echo "Extract results:"
for SEED in 1337 42 2024; do
  echo "--- Seed $SEED ---"
  grep "final_quant_roundtrip\|final_ttt_lora\|sliding_window\|Total submission size\|stopping_early\|ttt_gain" "$SUBDIR/logs/train_seed${SEED}.log" | tail -6
done
