# LoopedGPT + mHC-lite (3x2 loops) + Single-Epoch TTT

**val_bpb: TBD** (mean of 3 seeds, post int6+zstd quantization roundtrip + single-epoch TTT)

## Run Command

```bash
# Setup (once)
bash prepare.sh

# Train + evaluate (default seed=1337)
bash eval/eval.sh

# With specific seed
SEED=42 bash eval/eval.sh
```

All hyperparameters are set as defaults in `train_gpt.py`. No env vars needed beyond `SEED`.

## 3-Seed Results

| Seed | val_bpb (TTT) | artifact_bytes | train_time_s | valid |
|------|--------------|----------------|--------------|-------|
| 1337 | TBD | TBD | TBD | TBD |
| 42   | TBD | TBD | TBD | TBD |
| 2024 | TBD | TBD | TBD | TBD |
| **Mean** | **TBD** | | | |
| **Std**  | **TBD** | | | |

## Key Techniques

- **LoopedGPT**: 3 unique transformer blocks x 2 loops = 6 effective layers with weight tying
- **mHC-lite multi-stream routing**: 3 persistent streams with doubly stochastic mixing via Birkhoff decomposition
- **Per-loop LoRA adapters**: rank=16, A on Muon, B on Adam at 0.1x LR, warmup freeze
- **Single-epoch backward-looking TTT**: LoRA rank=8, score-then-train per chunk
- **int6 bit packing**: 4 values in 3 bytes (25% size reduction vs int8)
- **zstd-22 compression**
- **EMA**: decay=0.999, applied before serialization
- **SWA**: weight averaging during warmdown
- **BigramHash embedding**: vocab=4096, dim=128
- **SmearGate**: bigram context blending

## Architecture

- 3 unique blocks x 2 loops, model_dim=768
- 12 heads, 4 KV heads (GQA)
- SwiGLU MLP, mult=2 (hidden=1536)
- mHC-lite: 3 streams, 6 permutation matrices (3!), doubly stochastic H_res
- SmearGate token blending
- BigramHashEmbedding(4096, 128)
- Tied embeddings, logit softcap=30
- RoPE base=50000

## Training Hyperparameters

- Muon optimizer: matrix_lr=0.02, momentum=0.99, WD=0.04
- AdamW embeddings/scalars: scalar_lr=0.02, WD=0.04
- Batch: 786,432 tokens/step
- Warmup: 20 steps, warmdown: 3000 steps
- Max wallclock: 600s on 8xH100 SXM
- grad_clip=0.3, EMA decay=0.999

## TTT Configuration

- Single epoch (TTT_EPOCHS=1)
- LoRA rank=8, lr=0.01, chunk_size=256
- Backward-looking: score chunk i before training on chunk i
- Reset LoRA between documents
- Skip docs < 512 tokens (scored directly)

## Novel Contributions

1. mHC-lite multi-stream routing in looped transformer (first application)
2. Stream symmetry breaking via learned offsets for doubly stochastic routing
3. Per-loop LoRA adapters with Muon-on-A, Adam-on-B optimizer split
4. Alpha warmup schedule for LoRA cold-start protection
