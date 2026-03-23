"""Benchmark: kernel vs PyTorch baseline for each operation."""
import torch
import time
from itertools import permutations


def bench(fn, *args, warmup=10, iters=100, label=""):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000 / iters
    print(f"  {label:40s} {ms:.3f} ms")
    return ms


if __name__ == "__main__":
    B, T, n, D = 1, 1024, 4, 768
    n_perms = 24
    device = "cuda"

    x_streams = torch.randn(B, T, n, D, device=device, dtype=torch.bfloat16)
    res_weights = torch.softmax(torch.randn(B, T, n_perms, device=device), dim=-1)

    perms = list(permutations(range(n)))
    perm_idx = torch.tensor(perms, dtype=torch.int32, device=device)
    perm_mat = torch.zeros(n_perms, n, n, device=device)
    for i, p in enumerate(perms):
        for r, c in enumerate(p):
            perm_mat[i, r, c] = 1.0

    print("=== Permute Mix ===")
    def pytorch_permute():
        H = torch.einsum("btp,pij->btij", res_weights, perm_mat)
        return torch.einsum("btij,btjd->btid", H, x_streams.float()).to(x_streams.dtype)

    t_base = bench(pytorch_permute, label="PyTorch einsum")

    try:
        from looped_gpt_kernels.kernels.mhc_permute import FusedPermuteMix
        t_kern = bench(lambda: FusedPermuteMix.apply(x_streams, res_weights, perm_idx),
                       label="Triton fused")
        print(f"  Speedup: {t_base / t_kern:.2f}x")
        print(f"  Per-step savings (9 calls): {9 * (t_base - t_kern):.1f}ms")
    except Exception as e:
        print(f"  Triton kernel failed: {e}")

    print("\n=== RMSNorm Streams ===")
    import torch.nn.functional as F

    def pytorch_norm():
        return F.rms_norm(x_streams.reshape(B, T, n * D), (n * D,))

    t_base = bench(pytorch_norm, label="PyTorch reshape+rms_norm")

    try:
        from looped_gpt_kernels.kernels.mhc_norm import fused_rms_norm_streams
        t_kern = bench(lambda: fused_rms_norm_streams(x_streams), label="Triton fused")
        print(f"  Speedup: {t_base / t_kern:.2f}x")
        print(f"  Per-step savings (18 calls): {18 * (t_base - t_kern):.1f}ms")
    except Exception as e:
        print(f"  Triton kernel failed: {e}")

    print("\n=== LoRA Forward ===")
    x_2d = torch.randn(B, T, D, device=device, dtype=torch.bfloat16)
    A = torch.randn(D, 32, device=device, dtype=torch.float32) * 0.01
    Bm = torch.randn(32, D, device=device, dtype=torch.float32)

    def pytorch_lora():
        return ((x_2d @ A.to(x_2d.dtype)) @ Bm.to(x_2d.dtype)) * 0.177

    t_base = bench(pytorch_lora, label="PyTorch matmul chain")

    from looped_gpt_kernels.kernels.lora_forward import fused_lora_forward
    t_kern = bench(lambda: fused_lora_forward(x_2d, A, Bm, 0.177), label="Fused (placeholder)")
    print(f"  Per-step (18 calls): {18 * t_base:.1f}ms")
