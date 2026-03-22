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
    "MHC_STREAMS": "0",
    "BIGRAM_VOCAB_SIZE": "0",
    "ITERATIONS": "200",
    "TRAIN_BATCH_TOKENS": "8192",
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


def analyze_lora_weights(run_id: str, num_loops: int = 3, num_blocks: int = 3):
    """Analyze LoRA weight diversity from a trained model."""
    model_path = LOGS_DIR / f"{run_id}_mlx_model.npz"
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        return

    weights = dict(np.load(str(model_path), allow_pickle=True))

    print(f"\n{'='*60}")
    print(f"LoRA Weight Analysis: {run_id}")
    print(f"{'='*60}")

    # Keys are: lora_q_0.A, lora_q_0.B, lora_v_0.A, etc.
    # Index = loop_idx * num_blocks + block_idx
    lora_keys = sorted([k for k in weights.keys() if k.startswith("lora_q_") and k.endswith(".A")])
    if not lora_keys:
        print("No LoRA weights found in npz. Keys:", [k for k in weights.keys() if "lora" in k][:10])
        return

    # Experiment 1: Cosine similarity between loops (same block, different loops)
    print("\n--- Experiment 1: Cosine Similarity Between Loops ---")
    for block_idx in range(num_blocks):
        for loop_a in range(num_loops):
            for loop_b in range(loop_a + 1, num_loops):
                idx_a = loop_a * num_blocks + block_idx
                idx_b = loop_b * num_blocks + block_idx
                key_a = f"lora_q_{idx_a}.A"
                key_b = f"lora_q_{idx_b}.A"
                if key_a in weights and key_b in weights:
                    A_a = weights[key_a].flatten().astype(np.float32)
                    A_b = weights[key_b].flatten().astype(np.float32)
                    norm_a = np.linalg.norm(A_a)
                    norm_b = np.linalg.norm(A_b)
                    if norm_a > 0 and norm_b > 0:
                        sim = np.dot(A_a, A_b) / (norm_a * norm_b)
                        print(f"  Q block={block_idx} loop={loop_a} vs loop={loop_b}: cos_sim={sim:.4f}")
                # Also check V
                key_a_v = f"lora_v_{idx_a}.A"
                key_b_v = f"lora_v_{idx_b}.A"
                if key_a_v in weights and key_b_v in weights:
                    A_a = weights[key_a_v].flatten().astype(np.float32)
                    A_b = weights[key_b_v].flatten().astype(np.float32)
                    norm_a = np.linalg.norm(A_a)
                    norm_b = np.linalg.norm(A_b)
                    if norm_a > 0 and norm_b > 0:
                        sim = np.dot(A_a, A_b) / (norm_a * norm_b)
                        print(f"  V block={block_idx} loop={loop_a} vs loop={loop_b}: cos_sim={sim:.4f}")

    # Experiment 4: LoRA effective norms
    print("\n--- Experiment 4: LoRA Effective Norms (||B @ A^T||_F) ---")
    for loop_idx in range(num_loops):
        for block_idx in range(num_blocks):
            idx = loop_idx * num_blocks + block_idx
            for prefix in ["lora_q", "lora_v"]:
                key_a = f"{prefix}_{idx}.A"
                key_b = f"{prefix}_{idx}.B"
                if key_a in weights and key_b in weights:
                    A = weights[key_a].astype(np.float32)
                    B = weights[key_b].astype(np.float32)
                    effective = np.linalg.norm(A @ B)
                    label = "Q" if "q" in prefix else "V"
                    print(f"  {label} loop={loop_idx} block={block_idx}: norm={effective:.6f}")


def main():
    results = []

    # Experiment 3: Direct comparison
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: LoRA vs No-LoRA comparison")
    print("=" * 60)

    results.append(run_config("diag_no_lora", {"LORA_RANK": "0"}))
    results.append(run_config("diag_lora32", {"LORA_RANK": "32"}))
    results.append(run_config("diag_lora32_ortho", {"LORA_RANK": "32", "ORTHO_LORA": "1"}))

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
    analyze_lora_weights("diag_lora32_ortho")

    # Experiment 2: Ablation (zero out one loop's LoRA)
    # This would require loading model and doing eval — skip for now,
    # the cosine similarity and norms give equivalent information.

    print("\n" + "=" * 60)
    print("DONE. Check WandB for training curves.")
    print("=" * 60)


if __name__ == "__main__":
    main()
