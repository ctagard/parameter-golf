#!/usr/bin/env python3
"""Experiment runner for parameter-golf architectural search.

Usage:
    python experiments/run.py <config_name> [--dry-run]
    python experiments/run.py list
    python experiments/run.py compare <run_id1> <run_id2> ...

Configs are JSON files in experiments/configs/.
Results are logged to logs/ and summarized in experiments/results.jsonl.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "experiments" / "configs"
RESULTS_FILE = REPO_ROOT / "experiments" / "results.jsonl"
LOGS_DIR = REPO_ROOT / "logs"


def load_config(name: str) -> dict:
    path = CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    return json.loads(path.read_text())


def list_configs():
    if not CONFIGS_DIR.exists():
        print("No configs directory found.")
        return
    for p in sorted(CONFIGS_DIR.glob("*.json")):
        cfg = json.loads(p.read_text())
        desc = cfg.get("description", "")
        print(f"  {p.stem:30s} {desc}")


def run_experiment(config_name: str, dry_run: bool = False):
    cfg = load_config(config_name)
    env_vars = cfg.get("env", {})
    description = cfg.get("description", "")
    script = cfg.get("script", "train_gpt_mlx.py")

    # Build run ID from config name + timestamp
    run_id = f"{config_name}_{int(time.time())}"
    env_vars["RUN_ID"] = run_id

    # Set defaults for local smoke testing if not specified
    env_vars.setdefault("ITERATIONS", "200")
    env_vars.setdefault("TRAIN_BATCH_TOKENS", "8192")
    env_vars.setdefault("VAL_LOSS_EVERY", "0")
    env_vars.setdefault("VAL_BATCH_SIZE", "524288")
    env_vars.setdefault("TRAIN_LOG_EVERY", "10")

    env = {**os.environ, **{k: str(v) for k, v in env_vars.items()}}

    print(f"{'[DRY RUN] ' if dry_run else ''}Running: {config_name}")
    print(f"  Description: {description}")
    print(f"  Script: {script}")
    print(f"  Run ID: {run_id}")
    print(f"  Key env vars:")
    for k, v in sorted(env_vars.items()):
        if k != "RUN_ID":
            print(f"    {k}={v}")

    if dry_run:
        return

    cmd = [sys.executable, str(REPO_ROOT / script)]
    t0 = time.time()
    result = subprocess.run(cmd, env=env, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0

    # Parse results from log file
    log_path = LOGS_DIR / f"{run_id}.txt"
    val_bpb = None
    val_loss = None
    model_params = None
    compressed_bytes = None

    if log_path.exists():
        for line in log_path.read_text().splitlines():
            if line.startswith("step:") and "val_bpb:" in line:
                for part in line.split():
                    if part.startswith("val_bpb:"):
                        val_bpb = float(part.split(":")[1])
                    if part.startswith("val_loss:"):
                        val_loss = float(part.split(":")[1])
            if line.startswith("model_params:"):
                model_params = int(line.split()[0].split(":")[1])
            if line.startswith("final_int8_zlib_roundtrip"):
                for part in line.split():
                    if part.startswith("val_bpb:"):
                        val_bpb = float(part.split(":")[1])
            if line.startswith("serialized_model_int8_zlib:"):
                compressed_bytes = int(line.split()[0].split(":")[1])

    record = {
        "config": config_name,
        "run_id": run_id,
        "description": description,
        "val_bpb": val_bpb,
        "val_loss": val_loss,
        "model_params": model_params,
        "compressed_bytes": compressed_bytes,
        "elapsed_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "env": env_vars,
    }
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\n{'='*60}")
    print(f"  Config:      {config_name}")
    print(f"  val_bpb:     {val_bpb}")
    print(f"  val_loss:    {val_loss}")
    print(f"  params:      {model_params}")
    print(f"  compressed:  {compressed_bytes}")
    print(f"  elapsed:     {elapsed:.1f}s")
    print(f"  exit_code:   {result.returncode}")
    print(f"{'='*60}")
    return record


def compare_results(run_ids: list[str]):
    if not RESULTS_FILE.exists():
        print("No results file found.")
        return
    records = [json.loads(line) for line in RESULTS_FILE.read_text().splitlines() if line.strip()]
    matched = [r for r in records if any(rid in r["run_id"] or rid == r["config"] for rid in run_ids)]
    if not matched:
        # Show all results
        matched = records

    print(f"{'Config':<30s} {'val_bpb':>10s} {'params':>12s} {'compressed':>12s} {'time':>8s}")
    print("-" * 75)
    for r in sorted(matched, key=lambda x: x.get("val_bpb") or 99):
        bpb = f"{r['val_bpb']:.4f}" if r.get("val_bpb") else "N/A"
        params = f"{r['model_params']:,}" if r.get("model_params") else "N/A"
        comp = f"{r['compressed_bytes']:,}" if r.get("compressed_bytes") else "N/A"
        print(f"{r['config']:<30s} {bpb:>10s} {params:>12s} {comp:>12s} {r['elapsed_s']:>7.1f}s")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "list":
        list_configs()
    elif cmd == "compare":
        compare_results(sys.argv[2:])
    else:
        dry_run = "--dry-run" in sys.argv
        run_experiment(cmd, dry_run=dry_run)
