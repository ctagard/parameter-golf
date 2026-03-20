# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

OpenAI Parameter Golf: train the best language model fitting in a 16MB artifact (compressed weights + code) within 10 minutes on 8xH100 GPUs. Scored by bits-per-byte (BPB) on FineWeb validation set. Lower is better.

**Key formula:** `BPB = val_loss_nats / log(2) * tokens_per_byte`

**Constraints:** 16,000,000 bytes (decimal, not MiB), no external downloads at eval time, self-contained artifact.

## Commands

### Local Development (Apple Silicon / MLX)

```bash
# Setup
python3 -m venv .venv && source .venv/bin/activate
pip install mlx numpy sentencepiece huggingface-hub datasets tqdm

# Download data (1 shard for smoke tests, up to 80 for full)
python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1

# Run a specific model family
MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=768 NUM_HEADS=12 \
  RUN_ID=test ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=524288 \
  python3 train_gpt_mlx.py

# Run via experiment config
python experiments/run.py <config_name>
python experiments/run.py list          # list available configs
python experiments/run.py compare       # compare all results
```

### GPU Training (CUDA / H100)

```bash
# Single GPU
MODEL_FAMILY=looped torchrun --standalone --nproc_per_node=1 train_gpt.py

# 8xH100 (leaderboard submission)
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

All hyperparameters are configured via environment variables (see `Hyperparameters` class). Key ones: `MODEL_FAMILY`, `RUN_ID`, `ITERATIONS`, `TRAIN_BATCH_TOKENS`, `VAL_LOSS_EVERY`, `MODEL_DIM`, `NUM_LAYERS`, `VOCAB_SIZE`.

## Architecture

### Two Parallel Training Scripts

- **`train_gpt.py`** — PyTorch/CUDA for submission. Supports distributed training via `torchrun`.
- **`train_gpt_mlx.py`** — MLX for local Apple Silicon iteration. NOT submitted.

**Hard limit: neither script may exceed 1500 lines.** Both are self-contained single files.

### Model Families (`MODEL_FAMILY` env var)

All families use `build_model(args)` factory. Each inherits `BaseModel` which provides shared logit/loss computation with tied embeddings and logit soft-capping.

| Family | Class | Key env vars | Description |
|---|---|---|---|
| `baseline` | `GPT` | `NUM_LAYERS` | U-Net skip connections, encoder-decoder split |
| `looped` | `LoopedGPT` | `NUM_UNIQUE_BLOCKS`, `NUM_LOOPS`, `LORA_RANK` | Depth recurrence with optional per-loop LoRA |
| `mpk` | `MPKGPT` | `NUM_LAYERS`, `MPK_K_STRIDE`, `MPK_M_STRIDE` | Multi-Pass Keys: shared weights at 3 resolutions |
| `mpk_looped` | `LoopedMPKGPT` | `NUM_UNIQUE_BLOCKS`, `NUM_LOOPS`, `MPK_K_STRIDE`, `MPK_M_STRIDE` | MPK + depth recurrence |

### Shared Components

- **`Block`** — transformer block with `resid_mix` (blends `x` with initial embedding `x0`), `attn_scale`, `mlp_scale`
- **`StreamBlock`** — simplified block without `resid_mix`/`x0`, used inside MPKBlock
- **`MPKBlock`** — runs shared `StreamBlock` at full, medium (`k_stride`), and coarse (`m_stride`) resolution; K stream gates P+M outputs via learned projections
- **`MLP_TYPE`** env var: `relu2` (default) or `swiglu`
- **`CastedLinear`** — stores weights in fp32, casts to compute dtype at forward time
- **GQA** (grouped query attention), **RoPE**, **RMSNorm**, **logit soft-capping** via tanh

### Optimizer

`SplitOptimizers` routes parameters automatically:
- `tok_emb.weight` → Adam (tied_embed_lr)
- 2D non-control params → Muon (Newton-Schulz orthogonalization)
- Everything else (scalars, control tensors, LoRA, loop_emb, skip_weights) → Adam (scalar_lr)

Control tensor patterns (kept fp32, routed to Adam): `attn_scale`, `mlp_scale`, `resid_mix`, `q_gain`, `skip_weight`, `fusion_scale`.

### Compression

Int8 quantization + zlib. Per-row scaling for 2D matrices, per-tensor for vectors. Small tensors (<65536 elements) kept as fp16. Training logs `final_int8_zlib_roundtrip` with compressed size. Target: stay under 14MB to leave headroom.

## Experiment Infrastructure

```
experiments/
  run.py              # Experiment runner
  configs/            # JSON experiment configs
  results.jsonl       # Accumulated results (gitignored)
```

Configs specify `MODEL_FAMILY` + architecture env vars. Runner handles RUN_ID, defaults for local testing, and result parsing.

## Submission Format

Records go in `records/track_10min_16mb/YYYY-MM-DD_Name/` containing:
- `README.md`, `submission.json`, `train_gpt.py`, train logs (3 seeds, p < 0.01)

New SOTA must beat existing record by >= 0.005 nats with statistical significance.

## Our Strategy

Phase 1 (validated): Depth recurrence beats baseline — `loop_3x3_768` (12.6M params) outperforms 9-block baseline (17.1M) at 200 steps.

Current focus:
1. **Phase 3**: Per-loop LoRA adapters + SwiGLU on looped architectures
2. **Phase 3b**: MPK replication + composition with depth recurrence
3. **Phase 4**: H100 validation of best config
4. **Phase 5**: Final submission targeting val_bpb < 1.00

Current SOTA in repo: 1.1748 BPB (Muon WD + Overtone Init + Sliding Window, 10L dim=512)
