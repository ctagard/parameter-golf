#!/usr/bin/env python3
"""Detailed profiling of looped + LoRA + mHC architecture.

Measures per-component timing: attention, MLP, mHC mixing, LoRA deltas,
SmearGate, BigramHash. Also reports memory breakdown and graph break status.

Usage:
    torchrun --standalone --nproc_per_node=1 experiments/detailed_profile.py
"""
import math
import os
import sys
import time

# Set defaults before importing train_gpt (class-level env reads)
os.environ.setdefault("MODEL_FAMILY", "looped")
os.environ.setdefault("NUM_UNIQUE_BLOCKS", "3")
os.environ.setdefault("NUM_LOOPS", "3")
os.environ.setdefault("MODEL_DIM", "768")
os.environ.setdefault("NUM_HEADS", "12")
os.environ.setdefault("NUM_KV_HEADS", "4")
os.environ.setdefault("MLP_TYPE", "swiglu")
os.environ.setdefault("MLP_MULT", "3")
os.environ.setdefault("LORA_RANK", "32")
os.environ.setdefault("MHC_STREAMS", "4")
os.environ.setdefault("BIGRAM_VOCAB_SIZE", "4096")
os.environ.setdefault("TRAIN_BATCH_TOKENS", "65536")
os.environ.setdefault("ITERATIONS", "1")

import torch
import torch.nn.functional as F

# Import model building from train_gpt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_gpt import Hyperparameters, build_model, LoRAAdapter, mHCLite, SmearGate, BigramHashEmbedding


def cuda_timer(fn, warmup=3, repeat=10, label=""):
    """Time a function on CUDA with proper synchronization."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    avg = sum(times) / len(times)
    std = (sum((t - avg)**2 for t in times) / len(times)) ** 0.5
    print(f"  {label:40s} {avg:8.3f} ms ± {std:.3f}")
    return avg


def main():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    args = Hyperparameters()
    print(f"Model: {args.model_family} dim={args.model_dim} heads={args.num_heads} "
          f"loops={args.num_loops} blocks={args.num_unique_blocks} "
          f"lora_rank={args.lora_rank} mhc={args.mhc_streams}")
    print(f"Batch: {args.train_batch_tokens} tokens, seq_len={args.train_seq_len}")

    # Build model
    model = build_model(args).to(device).bfloat16()
    for m in model.modules():
        if hasattr(m, 'weight') and hasattr(m.weight, 'data') and m.weight.dtype == torch.float32:
            pass  # CastedLinear keeps fp32 weights

    n_params = sum(p.numel() for p in model.parameters())
    lora_params = sum(p.numel() for n, p in model.named_parameters() if "lora" in n.lower())
    mhc_params = sum(p.numel() for n, p in model.named_parameters() if "mhc" in n.lower() or "alpha" in n.lower())
    print(f"Params: {n_params:,} total, {lora_params:,} LoRA, {mhc_params:,} mHC")

    # Create dummy batch
    grad_accum = 8
    local_tokens = args.train_batch_tokens // grad_accum
    batch_seqs = local_tokens // args.train_seq_len
    x = torch.randint(0, args.vocab_size, (batch_seqs, args.train_seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (batch_seqs, args.train_seq_len), device=device)

    print(f"Input shape: {x.shape} ({batch_seqs} seqs × {args.train_seq_len} tokens)")
    print()

    # =========================================================================
    # 1. Memory breakdown
    # =========================================================================
    print("=" * 60)
    print("MEMORY BREAKDOWN")
    print("=" * 60)
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(x, y)
    loss.backward()
    torch.cuda.synchronize()

    peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    print(f"  Model parameters:  {model_mb:8.1f} MB")
    print(f"  Peak allocated:    {peak_mb:8.1f} MB")
    print(f"  Activations+grads: {peak_mb - model_mb:8.1f} MB (estimated)")
    print()

    # =========================================================================
    # 2. Per-component timing (uncompiled, isolates each piece)
    # =========================================================================
    print("=" * 60)
    print("PER-COMPONENT TIMING (uncompiled, 1 microbatch)")
    print("=" * 60)

    model.zero_grad(set_to_none=True)

    # Full forward+backward
    def full_step():
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l = model(x, y)
        l.backward()

    total_ms = cuda_timer(full_step, label="Full forward+backward")

    # Forward only
    def fwd_only():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            model(x, y)
    cuda_timer(fwd_only, label="Forward only (no grad)")

    # Isolate SmearGate
    x_emb = model.tok_emb(x)
    x_norm = F.rms_norm(x_emb.bfloat16(), (x_emb.size(-1),))
    def smear_only():
        model.smear(x_norm)
    cuda_timer(smear_only, label="SmearGate")

    # Isolate BigramHash
    if model.bigram is not None:
        def bigram_only():
            model.bigram(x)
        cuda_timer(bigram_only, label="BigramHash")

    # Isolate a single Block
    x0 = x_norm.clone()
    def single_block():
        model.blocks[0](x_norm, x0)
    cuda_timer(single_block, label="Single Block (attn+MLP)")

    # Isolate attention within a block
    n_in = model.blocks[0].attn_norm(x_norm)
    def attn_only():
        model.blocks[0].attn(n_in)
    cuda_timer(attn_only, label="  Attention only")

    # Isolate MLP
    def mlp_only():
        model.blocks[0].mlp(model.blocks[0].mlp_norm(x_norm))
    cuda_timer(mlp_only, label="  MLP only")

    # Isolate LoRA (if present)
    if args.lora_rank > 0:
        lora_q = model.loop_lora_q[0][0]
        def lora_only():
            lora_q(n_in)
        cuda_timer(lora_only, label="Single LoRA adapter")
        print(f"  {'LoRA total (9 Q + 9 V per step)':40s} ≈ {cuda_timer(lora_only, warmup=1, repeat=1, label='(single)') * 18:.3f} ms")

    # Isolate mHC (if present)
    if args.mhc_streams == 4:
        B, T, D = x_norm.shape
        x_s = x_norm.unsqueeze(2).expand(B, T, 4, D).contiguous()
        block_out = x_norm.clone()
        def mhc_mix():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.mhc[0](x_s, block_out)
        cuda_timer(mhc_mix, label="mHC mix (forward)")
        def mhc_pool():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                model.mhc[0].mix_to_one(x_s)
        cuda_timer(mhc_pool, label="mHC mix_to_one")
        print(f"  {'mHC total (3 blocks × 3 loops × 2 ops)':40s} ≈ estimated from above")

    print()

    # =========================================================================
    # 3. Compiled vs uncompiled
    # =========================================================================
    print("=" * 60)
    print("COMPILED vs UNCOMPILED (full forward+backward)")
    print("=" * 60)

    uncompiled_ms = total_ms

    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)
    def compiled_step():
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            l = compiled_model(x, y)
        l.backward()

    # Warmup compile
    print("  Compiling (first run)...")
    try:
        compiled_step()
        compiled_step()
        torch.cuda.synchronize()
        compiled_ms = cuda_timer(compiled_step, label="Compiled forward+backward")
        speedup = uncompiled_ms / compiled_ms
        print(f"  {'Speedup':40s} {speedup:.2f}x")
    except Exception as e:
        print(f"  COMPILE FAILED: {e}")
        compiled_ms = None

    print()

    # =========================================================================
    # 4. Throughput summary
    # =========================================================================
    print("=" * 60)
    print("THROUGHPUT SUMMARY")
    print("=" * 60)
    tokens_per_step = args.train_batch_tokens
    best_ms = compiled_ms if compiled_ms else uncompiled_ms
    tok_per_sec = tokens_per_step / (best_ms / 1000)
    print(f"  Tokens/step:     {tokens_per_step:,}")
    print(f"  Best ms/step:    {best_ms:.1f} ms (1 microbatch, × {grad_accum} for full batch)")
    print(f"  Est full step:   {best_ms * grad_accum:.1f} ms")
    print(f"  Est tok/sec:     {tok_per_sec:,.0f}")
    if compiled_ms:
        h100_est = best_ms * 0.4  # rough H100/A100 ratio
        h100_steps_10min = 600_000 / (h100_est * grad_accum)
        print(f"  H100 est ms/step: {h100_est * grad_accum:.1f} ms (×{grad_accum} accum)")
        print(f"  H100 est steps/10min: {h100_steps_10min:.0f}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
