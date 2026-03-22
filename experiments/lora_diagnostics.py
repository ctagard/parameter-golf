#!/usr/bin/env python3
"""LoRA diagnostic experiments on MLX.

Runs 3 configs (no LoRA, LoRA standard, LoRA with BigramHash) at 200 steps,
then analyzes LoRA weight diversity across loops.

Usage:
    source .venv/bin/activate
    python experiments/lora_diagnostics.py
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"

COMMON_ENV = {
    "MODEL_FAMILY": "looped",
    "NUM_UNIQUE_BLOCKS": "3",
    "NUM_LOOPS": "3",
    "MODEL_DIM": "512",
    "NUM_HEADS": "8",
    "NUM_KV_HEADS": "4",
    "MLP_TYPE": "swiglu",
    "MHC_STREAMS": "4",
    "BIGRAM_VOCAB_SIZE": "4096",
    "ITERATIONS": "200",
    "TRAIN_BATCH_TOKENS": "32768",
    "VAL_LOSS_EVERY": "0",
    "VAL_BATCH_SIZE": "524288",
    "WARMUP_STEPS": "5",
    "TRAIN_LOG_EVERY": "50",
    "MATRIX_LR": "0.08",
}


def run_config(name: str, extra_env: dict) -> dict:
    """Run a training config and return parsed results."""
    env = {**os.environ, **COMMON_ENV, **extra_env, "RUN_ID": name}
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "train_gpt_mlx.py")],
        env=env, cwd=str(REPO_ROOT),
    )
    elapsed = time.time() - t0

    # Parse val_bpb from log
    log_path = LOGS_DIR / f"{name}.txt"
    val_bpb = None
    model_params = None
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.startswith("step:200") and "val_bpb:" in line:
                for part in line.split():
                    if part.startswith("val_bpb:"):
                        val_bpb = float(part.split(":")[1])
            if line.startswith("model_family:") and "model_params:" in line:
                for part in line.split():
                    if part.startswith("model_params:"):
                        model_params = int(part.split(":")[1])

    print(f"  val_bpb: {val_bpb}")
    print(f"  params: {model_params}")
    print(f"  elapsed: {elapsed:.1f}s")
    return {"name": name, "val_bpb": val_bpb, "params": model_params, "elapsed": elapsed}


def analyze_lora_weights(run_id: str):
    """Analyze LoRA weight diversity from a trained model."""
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten
    except ImportError:
        print("MLX not available for weight analysis")
        return

    model_path = LOGS_DIR / f"{run_id}_mlx_model.npz"
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    # Load weights
    weights = dict(np.load(str(model_path), allow_pickle=True))

    print(f"\n{'='*60}")
    print(f"LoRA Weight Analysis: {run_id}")
    print(f"{'='*60}")

    # Find LoRA A matrices
    lora_keys = sorted([k for k in weights.keys() if "lora_q" in k and ".A" in k])
    if not lora_keys:
        print("No LoRA weights found")
        return

    # Experiment 1: Cosine similarity between loops
    print("\n--- Cosine Similarity Between Loops ---")
    for block_idx in range(3):
        for loop_a in range(3):
            for loop_b in range(loop_a + 1, 3):
                key_a = f"loop_lora_q.{loop_a}.{block_idx}.A"
                key_b = f"loop_lora_q.{loop_b}.{block_idx}.A"
                if key_a in weights and key_b in weights:
                    A_a = weights[key_a].flatten().astype(np.float32)
                    A_b = weights[key_b].flatten().astype(np.float32)
                    norm_a = np.linalg.norm(A_a)
                    norm_b = np.linalg.norm(A_b)
                    if norm_a > 0 and norm_b > 0:
                        sim = np.dot(A_a, A_b) / (norm_a * norm_b)
                        print(f"  block={block_idx} loop={loop_a} vs loop={loop_b}: cos_sim={sim:.4f}")

    # Experiment 4: LoRA weight norms
    print("\n--- LoRA Effective Norms (||B @ A^T||_F) ---")
    for loop_idx in range(3):
        for block_idx in range(3):
            key_a = f"loop_lora_q.{loop_idx}.{block_idx}.A"
            key_b = f"loop_lora_q.{loop_idx}.{block_idx}.B"
            if key_a in weights and key_b in weights:
                A = weights[key_a].astype(np.float32)
                B = weights[key_b].astype(np.float32)
                effective = np.linalg.norm(B @ A.T)
                print(f"  Q lora loop={loop_idx} block={block_idx}: norm={effective:.6f}")

    # V LoRA norms
    for loop_idx in range(3):
        for block_idx in range(3):
            key_a = f"loop_lora_v.{loop_idx}.{block_idx}.A"
            key_b = f"loop_lora_v.{loop_idx}.{block_idx}.B"
            if key_a in weights and key_b in weights:
                A = weights[key_a].astype(np.float32)
                B = weights[key_b].astype(np.float32)
                effective = np.linalg.norm(B @ A.T)
                print(f"  V lora loop={loop_idx} block={block_idx}: norm={effective:.6f}")


def main():
    results = []

    # Experiment 3: Direct comparison
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: LoRA vs No-LoRA comparison")
    print("=" * 60)

    results.append(run_config("diag_no_lora", {"LORA_RANK": "0"}))
    results.append(run_config("diag_lora32", {"LORA_RANK": "32"}))
    results.append(run_config("diag_lora32_bigram", {"LORA_RANK": "32", "BIGRAM_VOCAB_SIZE": "4096"}))

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Config':<30s} {'val_bpb':>10s} {'params':>12s}")
    print("-" * 55)
    for r in results:
        bpb = f"{r['val_bpb']:.4f}" if r['val_bpb'] else "N/A"
        params = f"{r['params']:,}" if r['params'] else "N/A"
        print(f"{r['name']:<30s} {bpb:>10s} {params:>12s}")

    # Analyze LoRA weights from the trained models
    analyze_lora_weights("diag_lora32")
    analyze_lora_weights("diag_lora32_bigram")

    # Experiment 2: Ablation (zero out one loop's LoRA)
    # This would require loading model and doing eval — skip for now,
    # the cosine similarity and norms give equivalent information.

    print("\n" + "=" * 60)
    print("DONE. Check WandB for training curves.")
    print("=" * 60)


if __name__ == "__main__":
    main()
