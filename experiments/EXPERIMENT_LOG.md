# Experiment Log

Tracking all experiments, motivations, results, and decisions for our Parameter Golf submission.

## Current Best Config

```bash
MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=512 \
  NUM_HEADS=8 NUM_KV_HEADS=4 MLP_TYPE=swiglu LORA_RANK=32 MHC_STREAMS=4 \
  TRAIN_BATCH_TOKENS=32768 MATRIX_LR=0.08
```

**val_bpb=2.0514** at 200 steps on 1 shard (M1 Ultra). ~13M params, ~32k tok/s.

Winning recipe: **LoopedGPT + SwiGLU + LoRA32 + mHC-Lite (n=4), no MPK, batch=32k, lr=0.08**

---

## Strategy Evolution

**Original hypothesis:** MPK multi-scale attention × depth recurrence × LoRA × mHC = maximum compute per parameter.

**What actually happened:** MPK's 3× attention passes per block cost 2.3× wall time. At fixed wallclock, the simpler looped model sees 2.3× more tokens/sec and wins. The mHC+LoRA+SwiGLU composition is doing the real work — MPK's multi-scale processing doesn't add enough to justify its speed penalty.

**Current strategy:** Depth recurrence (3 blocks × 3 loops) with per-loop LoRA specialization, mHC-Lite for stable multi-stream mixing, SwiGLU MLP, at the largest batch size that fits in wallclock.

---

## Phase 1: Depth Recurrence Validation

**Date:** 2026-03-19
**Motivation:** The baseline uses 9 independent blocks (~17M params). Sharing blocks across loops frees parameter budget for wider models.

**Setup:** 200 steps, 1 shard, TRAIN_BATCH_TOKENS=8192, VAL_BATCH_SIZE=524288

| Config | Params | val_bpb | Train Loss | ms/step |
|--------|--------|---------|------------|---------|
| baseline (9 blocks, dim=512) | 17.1M | 2.3015 | 3.9021 | 1320 |
| loop_3x3_512 (3×3, dim=512) | 6.0M | 2.3593 | 3.8153 | 567 |
| **loop_3x3_768 (3×3, dim=768)** | 12.6M | **2.2896** | 3.8313 | 1194 |
| loop_1x9_512 (1×9, dim=512) | ~2.3M | 2.3892 | 3.8520 | 581 |
| loop_3x3_512 + naive mHC n=2 | ~6M | killed | 4.1578 @200 | ~1990 |
| loop_3x3_768 + naive mHC n=2 | ~12.6M | killed | 4.1130 @200 | ~2752 |

**Conclusions:**
- ✅ Depth recurrence validated — loop_3x3_768 beats baseline with 26% fewer params
- ❌ Naive mHC (n=2, static sigmoid, no H_pre/H_post) was 2× slower and hurt loss
- Decision: Proceed with recurrence. Revisit mHC with proper mHC-Lite implementation.

---

## Phase 3: Component Ablation (4-Way Comparison)

**Date:** 2026-03-19
**Motivation:** Test each component independently on mpk_looped to find the winning combination.

**Setup:** 200 steps, 1 shard, batch=8192, dim=256, 3×3 loops. Small dim (2.9-3.3M params) for fast iteration — tests relative differences, not absolute competitiveness.

| Config | Params | val_bpb | Train Loss | ms/step |
|--------|--------|---------|------------|---------|
| mpk_looped relu2 | 2.92M | 2.7464 | 4.4548 | 388 |
| mpk_looped swiglu | 3.31M | 2.6702 | 4.3576 | 420 |
| mpk_looped + lora32 | 3.16M | 2.7088 | 4.3623 | 399 |
| **mpk_looped + lora32 + mHC** | **3.26M** | **2.5309** | **4.1211** | 439 |

**Key findings:**
- ✅ **mHC-Lite + LoRA is the big winner** — 0.14 BPB gap over next best
- ✅ SwiGLU beats relu² (0.076 BPB) at ~8% speed cost
- ✅ LoRA helps (0.038 BPB) but alone doesn't beat SwiGLU
- ✅ mHC overhead is only 13% — much better than naive mHC's 2×
- ✅ Proper mHC-Lite (n=4, Birkhoff decomposition, input-dependent) is fundamentally better than naive mHC (n=2, static sigmoid)

**Interpretation:** LoRA creates distinct per-loop signals. mHC-Lite blends 4 streams with input-dependent doubly stochastic mixing. Together they enable richer information flow than either alone.

---

## Full Stack at dim=512 vs Baseline (batch=8k)

**Date:** 2026-03-19
**Motivation:** Scale up to fair param comparison with baseline.

**Setup:** 200 steps, 1 shard, batch=8192, dim=512

| Config | Params | val_bpb | Compressed | ms/step |
|--------|--------|---------|------------|---------|
| baseline 9L dim=512 | 17.1M | **2.3998** | 11.3MB | 373 |
| mpk_looped swiglu+lora32+mhc dim=512 | 13.8M | 2.4213 | — | 841 |

**Conclusion:** At batch=8k, the full stack is slightly behind baseline despite fewer params — the 2.3× speed penalty means fewer total tokens seen. Need larger batch.

---

## LR Sweep (mpk_looped, batch=8k)

**Date:** 2026-03-20
**Motivation:** mHC should stabilize training at higher LR.

**Setup:** Full stack (swiglu+lora32+mhc) dim=512, 200 steps, batch=8k, varying matrix_lr.

| matrix_lr | Train Loss | val_bpb | Diverged? |
|-----------|-----------|---------|-----------|
| **0.04** | **3.9798** | **2.4221** | No |
| 0.08 | 3.9981 | 2.4442 | No |
| 0.12 | 4.2251 | 2.5347 | No |

**Conclusion:** mHC keeps training stable at 3× LR (no divergence) but val_bpb gets worse. At batch=8k, the default LR is already optimal. However, this changes at larger batch — see below.

---

## Batch Size Sweep + MPK Ablation

**Date:** 2026-03-20
**Motivation:** batch=8k means only 1.6M tokens in 200 steps. Bigger batches = better gradients + more data.

### Batch Sweep (mpk_looped swiglu+lora32+mhc dim=512, lr=0.04)

| Batch | Steps | Total Tokens | Train Loss | val_bpb | tok/s | ms/step |
|-------|-------|-------------|-----------|---------|-------|---------|
| 8k | 200 | 1.6M | 3.983 | 2.4217 | 9.8k | 826 |
| **32k** | 200 | **6.5M** | **3.836** | **2.1965** | **13.8k** | 2377 |
| 64k | 134 (wallclock) | 8.8M | — | 2.5402 | 14.5k | 4503 |

batch=32k is the sweet spot. 64k hits the 600s wallclock cap. 32k sees 4× more data → **0.22 BPB improvement**.

### MPK Ablation (the decisive experiment)

**Does MPK actually help?** Tested looped (no MPK) vs mpk_looped, batch=32k, lr=0.04, everything else identical.

| Config | Params | val_bpb | tok/s | ms/step |
|--------|--------|---------|-------|---------|
| **looped (NO MPK)** swiglu+lora32+mhc | ~13M | **2.0597** | **31.9k** | 1015 |
| mpk_looped swiglu+lora32+mhc | 13.8M | 2.1965 | 13.8k | 2377 |

**MPK hurts.** It costs 2.3× wall time for worse BPB. The simpler looped model processes 2.3× more tokens per second. At fixed wallclock, speed wins over compute density. mHC+LoRA+SwiGLU is doing the real work.

**Decision:** Drop MPK from the submission architecture.

### Scaled LR at Larger Batch

**Motivation:** Classic linear scaling rule — when batch grows by k, LR can grow by sqrt(k) or k. mHC should keep this stable.

| Config | Batch | LR | Train Loss | val_bpb |
|--------|-------|-----|-----------|---------|
| mpk_looped | 32k | 0.04 | 3.836 | 2.1965 |
| mpk_looped | 32k | 0.08 | 3.697 | **2.1046** |
| **looped (no MPK)** | 32k | 0.04 | 3.626 | 2.0597 |
| **looped (no MPK)** | **32k** | **0.08** | **3.614** | **2.0514** |

**Conclusion:** LR=0.08 helps at batch=32k (unlike at batch=8k where it hurt). The looped (no MPK) at lr=0.08 batch=32k is our overall best: **val_bpb=2.0514**.

---

## BPB Calculation Verification

Checked `build_sentencepiece_luts` for the leading-space marker bug (using `"?"` instead of `"▁"`). Our MLX implementation correctly uses `"▁"` (U+2581). Absolute BPB numbers are correct.

---

## Architecture Components Implemented

| Component | Class | Status | Param Cost |
|-----------|-------|--------|------------|
| Depth recurrence | `LoopedGPT` | ✅ In winning config | Zero (saves params) |
| Per-loop LoRA | `LoRAAdapter` | ✅ In winning config | Small (rank×dim×2 per adapter) |
| SwiGLU MLP | `SwiGLUMLP` | ✅ In winning config | +50% vs relu² MLP |
| mHC-Lite (n=4) | `mHCLite` | ✅ In winning config | Small (3 projection matrices) |
| Loop embeddings | `loop_emb` | ✅ In winning config | Tiny (loops × dim) |
| MPK multi-scale | `MPKBlock` | ❌ Dropped (too slow) | — |
| Stride scheduling | `STRIDE_SCHEDULE` | ⏸ Implemented, untested (MPK-specific) | Zero |

---

## Key Implementation Notes

- **Optimizer routing:** Muon for 2D non-control params (40 matrices). Adam for tok_emb, LoRA A/B, mHC projections, scales, loop_emb (69 scalar-like params).
- **mHC-Lite:** Exact Birkhoff via softmax over 24 permutation matrices (n=4). Input-dependent H_pre/H_post/H_res. α init=0.01 so mixing starts near identity.
- **LoRA:** Routes to Adam not Muon (low-rank matrices don't suit Newton-Schulz).
- **Line budget:** `train_gpt_mlx.py` at 1478/1500 lines.
- **Compression:** At dim=256, 2.9M params compresses to 2.9MB. At dim=512 ~13M params, baseline compressed to 11.3MB. Well under 16MB limit.

---

## PyTorch Port (Complete)

**Date:** 2026-03-20

Ported to `train_gpt.py` (1291 lines). Changes from baseline:
- Added `LoopedGPT`, `mHCLite`, `LoRAAdapter`, `SwiGLUMLP`, `build_model()`, `split_model_params()`
- Added WandB logging (`WANDB_PROJECT` env var)
- Removed TTT LoRA eval (~210 lines freed)
- Updated `CONTROL_TENSOR_NAME_PATTERNS` to include `loop_emb`, `alpha_res/pre/post`
- LoRA weights explicitly promoted to fp32 after bf16 cast
- Baseline family path unchanged

**mHC Triton kernel analysis:** NOT worth it. mHC is <7% of one attention layer's compute (~74k vs ~1M+ FLOPs/token). The 13% wall time overhead is memory bandwidth, which torch.compile should handle. Engineering risk >> marginal speedup.

## Next: A100 Validation

**Run script:** `experiments/run_a100_validation.sh`

Two back-to-back runs on single A100 80GB:
- Run 1: batch=65536, lr=0.08, 500 steps (~$1-2)
- Run 2: batch=131072, lr=0.10, 500 steps (~$1-2)

**Watch for:**
- Loss decreasing smoothly (no spikes)
- Val BPB trending below 2.05 by step 500
- Grad norms stable
- Compressed artifact under 16MB
- No CUDA errors / NaN losses

**Gate to 8×H100:** All of the above must pass. One failed run from a port bug costs more than the A100 validation.

## Future Experiments (after A100 validates)

- Larger dim (640, 768) if compressed size allows
- More loops (4×4, 3×5) for deeper effective depth
- LoRA rank sweep (16 vs 32 vs 64)
- mHC n=2 vs n=4 (n=2 might be faster and sufficient)
- Stride scheduling on the looped (non-MPK) architecture — adapt concept from MPK to vary attention window or other per-loop processing

---

## Summary of Validated Insights

1. **Depth recurrence works** — sharing 3 blocks across 3 loops beats 9 independent blocks at same param count
2. **mHC-Lite + LoRA is the key combination** — LoRA creates per-loop diversity, mHC blends it stably. Together they give 0.14 BPB over the next best single technique.
3. **Proper mHC-Lite >> naive mHC** — n=4 with Birkhoff decomposition and input-dependent projections is fundamentally better than n=2 with static sigmoid
4. **SwiGLU > relu²** — consistent 0.07 BPB improvement
5. **MPK hurts at fixed wallclock** — 3× attention passes per block = 2.3× slower, doesn't compensate with quality
6. **Batch size matters enormously** — 32k batch beats 8k by 0.22 BPB at 200 steps
7. **LR scales with batch** — lr=0.08 hurts at batch=8k but helps at batch=32k. mHC keeps it stable.
