# Experiment Log

Tracking all experiments, motivations, results, and decisions for our Parameter Golf submission.

## Core Strategy

**Goal:** Beat 1.0156 BPB (current SOTA: MPK 8×384) by composing:
- **MPK** (multi-scale attention) — 3 effective attention passes per block for 1 set of weights
- **Depth recurrence** — loop shared blocks for effective depth without parameter cost
- **Per-loop LoRA** — cheap specialization so each loop iteration has a distinct role
- **mHC-Lite** — doubly stochastic stream mixing to stabilize deep recurrence
- **Stride scheduling** — vary resolution per loop (coarse→fine) for scale diversity across depth

Combined: 3 unique MPKBlocks × 3 loops × 3 scales per block = **27 effective attention passes** for 3 weight sets.

---

## Phase 1: Depth Recurrence Validation

**Date:** 2026-03-19
**Motivation:** The baseline uses 9 independent blocks (~17M params). Most of the 16MB budget goes to block weights. If we share blocks across loops, we free parameter budget for wider models.

**Setup:** 200 steps, 1 shard, TRAIN_BATCH_TOKENS=8192, VAL_BATCH_SIZE=524288

| Config | Params | val_bpb | Train Loss | ms/step | Notes |
|--------|--------|---------|------------|---------|-------|
| baseline (9 blocks, dim=512) | 17.1M | **2.3015** | 3.9021 | 1320 | Reference |
| loop_3x3_512 (3×3, dim=512) | 6.0M | 2.3593 | 3.8153 | 567 | 1/3 params, competitive |
| **loop_3x3_768** (3×3, dim=768) | 12.6M | **2.2896** | 3.8313 | 1194 | Beats baseline, 26% fewer params |
| loop_1x9_512 (1×9, dim=512) | ~2.3M | 2.3892 | 3.8520 | 581 | Extreme recurrence, still OK |
| loop_3x3_512 + mHC n=2 | ~6M | killed early | 4.1578 @200 | ~1990 | 2x slower, higher loss |
| loop_3x3_768 + mHC n=2 | ~12.6M | killed early | 4.1130 @200 | ~2752 | mHC hurt at these depths |

**Conclusions:**
- ✅ Depth recurrence validated — wider looped models beat independent blocks
- ✅ loop_3x3_768 is the best config (beats baseline with fewer params)
- ❌ Naive mHC (n=2, simple sigmoid mixing) hurt performance — too slow, not expressive enough
- The n=2 mHC was a simplistic implementation (just H_res, no H_pre/H_post). Proper mHC-Lite with n=4 and input-dependent projections should be different.

**Decision:** Proceed with depth recurrence. Revisit mHC after implementing the proper mHC-Lite (Birkhoff decomposition, n=4, input-dependent).

---

## Phase 3/3b: MPK + Looped Composition

**Date:** 2026-03-19
**Motivation:** MPK (Multi-Pass Keys) is the current SOTA at 1.0156 BPB. It runs one shared StreamBlock at 3 resolutions (full/k=2/m=4) per MPKBlock. Composing with depth recurrence multiplies compute efficiency: 3 blocks × 3 loops × 3 scales = 27 effective attention passes for 3 weight sets.

### Smoke Tests (3 steps, verifying all model families work)

| Family | Config | Params | Status |
|--------|--------|--------|--------|
| baseline (GPT) | 9L dim=512 | 17.1M | ✅ Works |
| looped (LoopedGPT) | 2×2 dim=512 | 4.2M | ✅ Works |
| looped + LoRA | 2×2 dim=512 rank=8 | 4.3M | ✅ Works |
| looped + SwiGLU | 2×2 dim=512 swiglu | 5.2M | ✅ Works |
| mpk (MPKGPT) | 2L dim=128 | 591K | ✅ Works |
| mpk_looped (LoopedMPKGPT) | 2×2 dim=128 | 591K | ✅ Works |
| mpk_looped + LoRA | 2×2 dim=128 rank=8 | 606K | ✅ Works |
| mpk_looped + mHC-Lite n=4 | 2×2 dim=128 | 624K | ✅ Works |

### 200-Step Comparisons (dim=256, 3×3 loops)

**Setup:** 200 steps, 1 shard, TRAIN_BATCH_TOKENS=8192, VAL_BATCH_SIZE=524288

⚠️ **Note:** dim=256 gives only 2.9M params — much smaller than baseline's 17M. These results test *relative* differences between variants, not absolute competitiveness vs baseline. Need to scale up dim for fair baseline comparison.

| Config | Params | val_bpb | int8 bpb | Compressed | ms/step |
|--------|--------|---------|----------|------------|---------|
| mpk_looped relu2 | 2.92M | 2.7464 | 2.7458 | 2.9MB | 388 |
| mpk_looped swiglu | 3.31M | **2.6702** | 2.6709 | 3.2MB | 420 |
| mpk_looped + lora32 | 3.16M | 2.7088 | — | — | 399 |
| **mpk_looped + lora32 + mHC** | **3.26M** | **2.5309** | — | — | 439 |

**Key findings:**
- ✅ **mHC-Lite + LoRA is the clear winner** — 2.5309 vs 2.6702 for next best (swiglu), a 0.14 BPB gap
- ✅ SwiGLU beats relu² (2.6702 vs 2.7464) at ~8% speed cost
- ✅ LoRA helps (2.7088 vs 2.7464) but alone doesn't beat SwiGLU
- ✅ mHC overhead is only 13% (439 vs 388 ms/step) — much better than naive mHC's 2× from Phase 1
- ⚠️ SwiGLU is absent from the mHC+LoRA winning run — combining all three is the obvious next experiment
- ⚠️ dim=256 gives only 2.9-3.3M params vs baseline's 17.1M — not a fair absolute comparison

**Interpretation:** LoRA creates distinct per-loop signals (each loop specializes its attention via rank-32 adapters). mHC-Lite blends these 4 streams with input-dependent doubly stochastic mixing. The combination enables information flow patterns that neither component achieves alone — LoRA provides diversity, mHC provides stable cross-stream integration.

**Decision:** Run SwiGLU + LoRA32 + mHC at dim=512 for fair baseline comparison.

---

## Running: Full Stack at dim=512 vs Baseline

**Date:** 2026-03-19
**Motivation:** The 4-way comparison showed mHC+LoRA dominates at dim=256, but that's only 3.3M params vs baseline's 17M. Need a fair comparison at dim=512. Also combining SwiGLU (best MLP) + LoRA32 (best specialization) + mHC (best mixing) for the first time.

**Setup:** 200 steps, 1 shard, TRAIN_BATCH_TOKENS=8192, VAL_BATCH_SIZE=524288

| Config | Expected Params | Status |
|--------|----------------|--------|
| mpk_looped swiglu+lora32+mhc dim=512 | ~12-15M | 🔄 Running |
| baseline 9L dim=512 | 17.1M | ⏳ Queued (runs after) |

**Results:**

| Config | Params | val_bpb | int8 bpb | Compressed | ms/step |
|--------|--------|---------|----------|------------|---------|
| baseline 9L dim=512 | 17.1M | **2.3998** | 2.4007 | 11.3MB | 373 |
| full stack mpk_looped swiglu+lora32+mhc dim=512 | 13.8M | 2.4213 | — | — | 841 |

**Analysis:** Full stack is slightly behind baseline (0.02 BPB) with 19% fewer params. But it's 2.3x slower per step — at 200 steps with 8192 tokens/batch, both models see the same number of tokens but the full stack takes 168s vs 75s. The loss curve is still descending steeply at step 200, suggesting it would overtake with more iterations.

**Key concern:** The per-step speed penalty means on H100 with a 10-minute cap, the full stack gets fewer total steps than baseline. Need to verify this doesn't eliminate the param efficiency advantage.

**Next experiment:** Bump learning rate. mHC's doubly stochastic constraint should allow more aggressive optimization without divergence — test matrix_lr=0.08 and 0.12.

### LR Sweep Results

**Setup:** Full stack (swiglu+lora32+mhc) at dim=512, 200 steps, varying matrix_lr only.

| matrix_lr | Train Loss | val_bpb | Divergence? |
|-----------|-----------|---------|-------------|
| **0.04** | **3.9798** | **2.4221** | No |
| 0.08 | 3.9981 | 2.4442 | No |
| 0.12 | 4.2251 | 2.5347 | No, but worse |

**Conclusion:** mHC keeps training stable at 3× base LR (no divergence), but val_bpb degrades. The default matrix_lr=0.04 is already well-tuned. Higher LR doesn't help — the bottleneck is not optimization speed but something else (possibly the small batch size / few tokens at 200 steps, or model capacity at this dim).

---

## Batch Size Sweep + MPK Ablation

**Date:** 2026-03-20
**Motivation:** Our prior runs used batch=8192 tokens — only 1.6M total tokens in 200 steps. Scaling batch size gives better gradient estimates and more data per run.

### Batch Sweep (mpk_looped swiglu+lora32+mhc dim=512)

| Batch | Steps | Total Tokens | Train Loss | val_bpb | tok/s | ms/step |
|-------|-------|-------------|-----------|---------|-------|---------|
| 8k | 200 | 1.6M | 3.9831 | 2.4217 | 9.8k | 826 |
| **32k** | 200 | **6.5M** | **3.8363** | **2.1965** | **13.8k** | 2377 |
| 64k | 134 (wallclock cap) | 8.8M | — | 2.5402 | 14.5k | 4503 |

**Key finding:** batch=32k is the sweet spot. 64k hits the 600s wallclock cap at only 134 steps. 32k sees 4× more data and gets **0.22 BPB improvement** over 8k.

### MPK Ablation (the critical question)

**Does MPK add value?** Compared looped (no MPK) vs mpk_looped at batch=32k, same everything else.

| Config | Params | val_bpb | tok/s | ms/step |
|--------|--------|---------|-------|---------|
| **looped (NO MPK)** swiglu+lora32+mhc | ~13M | **2.0597** | **31.9k** | 1015 |
| mpk_looped swiglu+lora32+mhc | 13.8M | 2.1965 | 13.8k | 2377 |

**MPK HURTS at this config.** The 3× attention passes per block cost 2.3× wall time, and the simpler looped model gets 2.3× more tokens/sec. At fixed wallclock, speed > compute density. The mHC+LoRA+SwiGLU composition is doing the heavy lifting — MPK's multi-scale processing doesn't add enough to justify its speed penalty.

**Decision:** Drop MPK. Focus on `looped` family with swiglu+lora32+mhc.

### Scaled LR at Larger Batch

| Config | Batch | LR | Train Loss | val_bpb | Status |
|--------|-------|-----|-----------|---------|--------|
| mpk_looped batch=32k lr=0.04 | 32k | 0.04 | 3.8363 | 2.1965 | Done |
| mpk_looped batch=32k lr=0.08 | 32k | 0.08 | 3.6973 | 🔄 pending | Running |

LR=0.08 at batch=32k: train loss improved (3.70 vs 3.84), no divergence. Waiting for val_bpb.

**Next:** Run `looped` (no MPK) at batch=32k with lr=0.08 — combining the speed advantage of no-MPK with scaled LR.

---

## Planned: Stride Schedule Ablation

**Motivation:** Fixed strides across all loops means every loop sees the same 3 scales. Varying strides per loop adds scale diversity across depth at zero parameter cost:
- Early loops → coarse strides → global context (document structure, entities)
- Late loops → fine strides → local predictions (syntax, next-token)

This is the hourglass / coarse-to-fine pattern applied to the stride schedule rather than block weights.

**Configs ready (waiting for 4-way comparison to finish):**

| Config | Loop strides (k,m) | Hypothesis |
|--------|-------------------|------------|
| uniform | (2,4), (2,4), (2,4) | Baseline — same as current mpk_looped |
| coarse_to_fine | (4,8), (2,4), (1,2) | Global first, local last |
| fine_to_coarse | (1,2), (2,4), (4,8) | Local first, global last |
| hourglass | (4,8), (2,4), (4,8) | Global sandwich |

**Expected outcome:** coarse_to_fine should win — it matches the natural hierarchy of language understanding (comprehend context → refine details → predict token).

---

## Architecture Components Implemented

| Component | Class | Key Innovation | Param Cost |
|-----------|-------|---------------|------------|
| Depth recurrence | `LoopedGPT`, `LoopedMPKGPT` | Share blocks across loops | Zero (saves params) |
| Per-loop LoRA | `LoRAAdapter` | Rank-r Q/V adapters per loop iteration | Small (rank×dim×2 per adapter) |
| SwiGLU MLP | `SwiGLUMLP` | Gate+up+down projections | +50% vs relu² MLP |
| MPK multi-scale | `MPKBlock` | Shared StreamBlock at 3 resolutions, K gates P+M | Small (stem + gate + fusion projs) |
| mHC-Lite (n=4) | `mHCLite` | Exact Birkhoff decomposition, input-dependent mixing | Small (3 projection matrices) |
| Stride scheduling | `STRIDE_SCHEDULE` env var | Per-loop (k,m) stride overrides | Zero |
| Loop embeddings | `loop_emb` | Learned per-loop positional signal | Tiny (loops × dim) |

---

## Key Implementation Notes

- **LoRA routing:** LoRA A/B matrices are 2D but routed to Adam (not Muon) via `"lora" not in k.lower()` check in `SplitOptimizers`
- **mHC-Lite:** Uses softmax over 24 permutation matrices (n=4) for exact doubly stochastic H_res. α scalars init at 0.01 so mixing starts near identity.
- **Line budget:** `train_gpt_mlx.py` is at 1478/1500 lines. Any new features need to be concise.
- **Compression:** Looped models compress better than independent-block models. The mpk_looped relu2 at 2.9M params compresses to 2.9MB (well under 16MB).

---

## Open Questions

1. **Optimal dim for mpk_looped?** — At dim=256 we have 2.9M params and val_bpb=2.75. Baseline has 17M params and val_bpb=2.30. What dim gives ~15M params under MPK+looped to make a fair comparison?
2. **Does mHC-Lite actually help with LoRA?** — The hypothesis is that LoRA creates distinct per-loop signals that mHC can meaningfully blend. The naive mHC (n=2) hurt in Phase 1, but mHC-Lite (n=4, input-dependent, exact Birkhoff) is a fundamentally different implementation.
3. **Stride schedule vs uniform?** — Zero-cost experiment. If coarse-to-fine wins, it's a free improvement to every mpk_looped config.
4. **When to port to PyTorch?** — Need to port all new model families to `train_gpt.py` before H100 validation. Should do this once we've settled on the best MLX config.
