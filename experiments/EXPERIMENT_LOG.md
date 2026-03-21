# Experiment Log

Tracking all experiments, motivations, results, and decisions for our Parameter Golf submission.

## Current Best Config

```bash
MODEL_FAMILY=looped NUM_UNIQUE_BLOCKS=3 NUM_LOOPS=3 MODEL_DIM=768 \
  NUM_HEADS=12 NUM_KV_HEADS=4 MLP_TYPE=swiglu LORA_RANK=32 MHC_STREAMS=4 \
  BIGRAM_VOCAB_SIZE=4096 MUON_WD=0.04 ADAM_WD=0.04 \
  TRAIN_BATCH_TOKENS=131072 MATRIX_LR=0.10
```

**val_bpb=1.8140** at 300 steps on 1×A100 SXM 80GB (with table-stakes mods).
**val_bpb=1.4309** at 1500 steps on 2×A100 SXM 80GB (batch=524k, no table-stakes).

Winning recipe: **LoopedGPT + SwiGLU + LoRA32 + mHC-Lite (n=4) + SmearGate + BigramHash + OrthoInit + WD 0.04**

---

## Strategy Evolution

**Original hypothesis:** MPK multi-scale attention × depth recurrence × LoRA × mHC = maximum compute per parameter.

**What actually happened:** MPK's 3× attention passes per block cost 2.3× wall time. At fixed wallclock, the simpler looped model sees 2.3× more tokens/sec and wins. The mHC+LoRA+SwiGLU composition is doing the real work.

**Current strategy:** Depth recurrence (3 blocks × 3 loops) with per-loop LoRA specialization, mHC-Lite for stable multi-stream mixing, SwiGLU MLP, table-stakes mods (SmearGate, BigramHash, OrthoInit, WD), at the largest batch size that fits. Exploring hypernetwork conditioning to make loops content-aware.

---

## Phase 1: Depth Recurrence Validation (MLX, M1 Ultra)

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

## Phase 3: Component Ablation — 4-Way Comparison (MLX)

**Date:** 2026-03-19
**Motivation:** Test each component independently on mpk_looped to find the winning combination.

**Setup:** 200 steps, 1 shard, batch=8192, dim=256, 3×3 loops. Small dim (2.9-3.3M params) for fast iteration.

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
- ✅ mHC overhead only 13% — much better than naive mHC's 2×

**Interpretation:** LoRA creates distinct per-loop signals. mHC-Lite blends 4 streams with input-dependent doubly stochastic mixing. Together they enable richer information flow than either alone.

---

## Batch Size Sweep + MPK Ablation (MLX)

**Date:** 2026-03-20
**Motivation:** batch=8k means only 1.6M tokens in 200 steps. Bigger batches = better gradients + more data.

### Batch Sweep (mpk_looped swiglu+lora32+mhc dim=512, lr=0.04)

| Batch | Steps | Total Tokens | val_bpb | tok/s |
|-------|-------|-------------|---------|-------|
| 8k | 200 | 1.6M | 2.4217 | 9.8k |
| **32k** | 200 | **6.5M** | **2.1965** | **13.8k** |
| 64k | 134 (wallclock) | 8.8M | 2.5402 | 14.5k |

### MPK Ablation — the decisive experiment

| Config | Params | val_bpb | tok/s |
|--------|--------|---------|-------|
| **looped (NO MPK)** swiglu+lora32+mhc | ~13M | **2.0597** | **31.9k** |
| mpk_looped swiglu+lora32+mhc | 13.8M | 2.1965 | 13.8k |

**MPK hurts.** 2.3× slower for worse BPB. Simpler looped model processes 2.3× more tokens/sec. **Decision:** Drop MPK.

### Scaled LR at Larger Batch

| Config | Batch | LR | val_bpb |
|--------|-------|-----|---------|
| looped (no MPK) | 32k | 0.04 | 2.0597 |
| **looped (no MPK)** | **32k** | **0.08** | **2.0514** |

LR=0.08 helps at batch=32k. Best MLX result: **val_bpb=2.0514**.

---

## A100 Validation Results

**Date:** 2026-03-20

### 1×A100 SXM 80GB — dim=768, batch=131k, lr=0.10, 500 steps (no table-stakes)

| Step | val_bpb | Train Loss |
|------|---------|-----------|
| 100 | 2.1990 | 3.878 |
| 200 | 1.9598 | 3.330 |
| 300 | 1.8520 | 3.114 |
| 400 | 1.7402 | 2.977 |
| 500 | **1.6794** | 2.861 |

- 17.2M params, compressed to **12.0MB** (int8+zlib, under 16MB limit)
- 828ms/step, 158k tok/s. Peak memory: 8.9GB / 80GB
- Loss still dropping steeply at step 500

### 2×A100 SXM 80GB — dim=768, batch=524k, lr=0.10, 1500 steps (no table-stakes)

| Step | Tokens Seen | val_bpb | Train Loss |
|------|------------|---------|-----------|
| 500 | 262M | 1.8670 | 3.106 |
| 1000 | 524M | 1.5118 | 2.552 |
| 1500 | 786M | **1.4309** | 2.450 |

- 1350ms/step, 391k tok/s (2.47× throughput vs 1×A100 — super-linear scaling)
- Loss still dropping at 1500 steps — no plateau

---

## Table-Stakes Mods — A100 Validation

**Date:** 2026-03-20
**Motivation:** Every top submission uses SmearGate, BigramHash, OrthoInit, WD. Implement all before 8×H100 run.

**Mods implemented (all opt-in via env vars):**
- **SmearGate** — blend each token with previous via learned gate. Always on in LoopedGPT.
- **BigramHash** — hash bigram pairs into learned embedding table (`BIGRAM_VOCAB_SIZE=4096`)
- **OrthoInit + muP scaling** — orthogonal weight init, output proj scaled by 1/sqrt(2*layers)
- **AdamW weight decay** — `ADAM_WD=0.04`, `MUON_WD=0.04`
- **Int6+zstd** — `QUANT_BITS=6` for 6-bit quantization + zstd-22 compression (not enabled yet)
- **SWA** — `SWA_ENABLED=1` for warmdown averaging (not tested yet)

### 1×A100 — dim=768, batch=131k, lr=0.10, 300 steps, WITH table-stakes

| Config | Step 100 | Step 200 | Step 300 | int8 roundtrip | Compressed |
|--------|----------|----------|----------|---------------|------------|
| **No hypernet (table-stakes)** | 2.1432 | 1.9131 | **1.8140** | 1.8197 | 10.7MB |
| Coarse hypernet (.mean()) | 2.1340 | 1.9124 | 1.8188 | 1.8242 | 10.9MB |

**Comparison to no-table-stakes:** At 300 steps, table-stakes gives 1.8140 vs ~1.8520 without (interpolating from the 500-step run). That's a ~0.04 BPB improvement from SmearGate + BigramHash + OrthoInit + WD alone.

**Hypernetwork with .mean(): no gain.** The coarse hypernetwork using `x[:, ::stride, :].mean()` showed essentially no improvement — within noise (0.005 BPB). The `.mean()` problem is real: averaging across all positions throws away the structural information that would actually be useful for loop conditioning. Two sequences with different content can have nearly identical means.

**Decision:** Fix the hypernetwork to read meaningful signals, not lossy averages.

---

## Hypernetwork Investigation

**Date:** 2026-03-20
**Motivation:** The `.mean()` hypernetwork showed no gain. We need the hypernetwork to read information that actually helps condition loop iterations.

### The .mean() Problem

`x.mean(dim=1)` produces a vector representing the "average token" — but the information needed for loop conditioning is the *structure*, not the average. Which topics are present, what entities have been introduced, how has the representation evolved since the last loop. Two completely different documents can have nearly identical means.

### Three Approaches (all with learned attention pooling — no .mean() anywhere)

All variants now use `_AttnPool`: a learned query that attends over T positions to produce a single summary vector. This replaces the uniform `.mean()` that killed the first hypernetwork attempt.

**Variant 1: Delta-from-origin (`HYPERNET_VARIANT=coarse`)**
- Reads `_AttnPool(x - x0)` — learned pooling over what the loops have changed
- x0 is constant across loops, so the delta isolates what looping contributes
- Simplest variant: no cross-loop state, just reads current delta

**Variant 2: EMA trajectory (`HYPERNET_VARIANT=ema`)**
- `_AttnPool(x)` → per-dim learned decay EMA across loop iterations
- Encodes trajectory / direction, not just current state
- `alpha_logit=0` init → sigmoid(0)=0.5, per-dim learned decay rates
- `.detach()` on EMA prevents O(loops²) backward graphs

**Variant 3: SSM + attention pooling (`HYPERNET_VARIANT=ssm`)**
- Learned attention query pools over T positions → rank-r SSM input
- Content-dependent forget/update gates (dt, B, C projected from input)
- Unlike EMA's fixed alpha, the transition depends on what the content looks like
- ~27K extra params at rank=32, d_state=16

| Property | Delta | EMA | SSM |
|----------|-------|-----|-----|
| Pooling | Learned (attn) | Learned (attn) | Learned (attn) |
| Cross-loop memory | None | Yes (per-dim EMA) | Yes (content-dependent) |
| What it reads | x - x0 | x | x |
| Temporal structure | None | Fixed decay | Content-dependent |
| Conceptual fit | "What changed?" | "Where are we going?" | "What should I remember?" |

### A100 Hypernetwork Ablation (running)

**Setup:** 1×A100 SXM 80GB, dim=768, batch=131k, lr=0.10, 300 steps each. All with table-stakes (BigramHash, WD, OrthoInit, SmearGate).

| Config | Step 100 | Step 200 | Step 300 | int8 roundtrip | ms/step |
|--------|----------|----------|----------|---------------|---------|
| **No hypernet** | **2.1432** | **1.9131** | **1.8140** | **1.8197** | 744 |
| .mean() hypernet | 2.1340 | 1.9124 | 1.8188 | 1.8242 | 755 |
| Delta + attn pool | 2.1448 | 1.9231 | 1.8303 | 1.8358 | 767 |
| EMA + attn pool | 2.1488 | 1.9225 | 1.8311 | 1.8365 | 764 |
| SSM + attn pool | 2.1576 | 1.9451 | 1.8618 | 1.8668 | 767 |

**Result: All hypernetwork variants are worse than no hypernet.** SSM is the worst (0.048 BPB degradation). The simpler variants (delta, EMA) are less bad but still negative. The ~3% speed overhead is negligible — the problem is that the extra parameters dilute the optimizer at 300 steps.

**Decision: Drop all hypernetworks from the submission.** The base architecture is already strong. Focus on scaling up training (more steps, larger batch, int6+zstd for artifact headroom).

---

## 8×H100 Projection

- H100 SXM ~2-3× faster than A100 SXM for bf16
- `grad_accum_steps = 8//8 = 1`, each GPU processes 65536 tokens in one shot
- Estimated: ~200-300ms/step → 600s / 250ms ≈ 2,400 steps
- 2,400 steps × 524k = 1.26T tokens
- From the 2×A100 curve: 1.43 BPB at 786M tokens, still dropping fast
- **Projected val_bpb ~1.25-1.35** at 1.26T tokens (without table-stakes)
- With table-stakes + hypernetwork + int6+zstd → **target sub-1.15 BPB**
- Current repo SOTA: 1.1748 BPB

---

## Architecture Components

| Component | Status | Impact |
|-----------|--------|--------|
| Depth recurrence (3×3) | ✅ Core architecture | Frees param budget for wider model |
| Per-loop LoRA (rank=32) | ✅ Core architecture | Per-loop specialization |
| mHC-Lite (n=4) | ✅ Core architecture | 0.14 BPB win with LoRA |
| SwiGLU MLP | ✅ Core architecture | 0.07 BPB over relu² |
| SmearGate | ✅ Table-stakes | Bigram context in x0 |
| BigramHash (4096) | ✅ Table-stakes | Bigram frequency embedding |
| OrthoInit + muP | ✅ Table-stakes | Better-conditioned init |
| AdamW WD 0.04 | ✅ Table-stakes | Smaller weights → better quantization |
| Int6+zstd | ✅ Implemented, not yet tested | ~35% smaller artifacts |
| SWA | ✅ Implemented, not yet tested | Better generalization in warmdown |
| Hypernetwork (delta/EMA/SSM) | ❌ Dropped | All variants worse than no hypernet |
| MPK multi-scale | ❌ Dropped | Too slow — 2.3× wall time for worse BPB |

---

## Key Implementation Notes

- **train_gpt.py:** 1463 lines (under 1500 limit). All features opt-in via env vars.
- **Optimizer routing:** Muon for 2D non-control params. Adam for tok_emb, LoRA A/B, mHC projections, hypernet, scales, loop_emb, smear gate, bigram.
- **mHC-Lite:** Exact Birkhoff via softmax over 24 permutation matrices (n=4). α init=0.01.
- **LoRA:** fp32 weights explicitly restored after bf16 cast.
- **Compression:** 17.2M params at dim=768 compresses to 10.7-10.9MB (int8+zlib). Int6+zstd should bring this to ~7MB, freeing headroom for larger model or MLP_MULT=3.
- **Triton kernel for mHC:** NOT worth it. mHC is <1% of total FLOPs.

---

## Summary of Validated Insights

1. **Depth recurrence works** — sharing 3 blocks across 3 loops beats 9 independent blocks
2. **mHC-Lite + LoRA is the key combination** — 0.14 BPB over next best single technique
3. **Proper mHC-Lite >> naive mHC** — n=4 Birkhoff with input-dependent projections vs n=2 static sigmoid
4. **SwiGLU > relu²** — consistent 0.07 BPB improvement
5. **MPK hurts at fixed wallclock** — speed > compute density
6. **Batch size matters enormously** — 32k beats 8k by 0.22 BPB
7. **LR scales with batch** — lr=0.08-0.10 at batch=131k+
8. **Table-stakes mods give ~0.04 BPB** — SmearGate + BigramHash + OrthoInit + WD
9. **Hypernetworks don't help at 300 steps** — tested .mean(), delta-from-origin, EMA, SSM+attention pooling. All variants worse than no hypernet. Extra params dilute optimizer at this training scale.
10. **Simpler is better** — every "clever" addition (MPK, hypernetworks) hurt. The winning recipe is clean depth recurrence + LoRA + mHC + table-stakes.
