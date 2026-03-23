"""Correctness tests: verify kernel outputs match PyTorch reference."""
import torch
import torch.nn.functional as F
from itertools import permutations


def _make_perm_data(n=4, device="cuda"):
    perms = list(permutations(range(n)))
    perm_idx = torch.tensor(perms, dtype=torch.int32, device=device)
    perm_mat = torch.zeros(len(perms), n, n, device=device)
    for i, p in enumerate(perms):
        for r, c in enumerate(p):
            perm_mat[i, r, c] = 1.0
    return perm_idx, perm_mat


def test_permute_mix():
    from looped_gpt_kernels.kernels.mhc_permute import FusedPermuteMix, HAS_TRITON
    if not HAS_TRITON:
        print("permute_mix: SKIPPED (no Triton)")
        return

    B, T, n, D = 2, 128, 4, 768
    x = torch.randn(B, T, n, D, device="cuda", dtype=torch.bfloat16)
    w = torch.softmax(torch.randn(B, T, 24, device="cuda"), dim=-1)
    perm_idx, perm_mat = _make_perm_data(n)

    ref = torch.einsum("btij,btjd->btid",
                       torch.einsum("btp,pij->btij", w, perm_mat), x.float()).to(x.dtype)
    out = FusedPermuteMix.apply(x, w, perm_idx)
    diff = (ref.float() - out.float()).abs().max().item()
    assert diff < 0.05, f"permute_mix max diff: {diff}"
    print(f"permute_mix: PASSED (max diff: {diff:.6f})")


def test_rms_norm_streams():
    from looped_gpt_kernels.kernels.mhc_norm import fused_rms_norm_streams, HAS_TRITON
    if not HAS_TRITON:
        print("rms_norm_streams: SKIPPED (no Triton)")
        return

    B, T, n, D = 2, 128, 4, 768
    x = torch.randn(B, T, n, D, device="cuda", dtype=torch.bfloat16)
    ref = F.rms_norm(x.reshape(B, T, n * D), (n * D,))
    out = fused_rms_norm_streams(x)
    diff = (ref.float() - out.float()).abs().max().item()
    assert diff < 0.05, f"rms_norm_streams max diff: {diff}"
    print(f"rms_norm_streams: PASSED (max diff: {diff:.6f})")


def test_lora_forward():
    from looped_gpt_kernels.kernels.lora_forward import fused_lora_forward
    B, T, dim, rank, out_dim = 2, 128, 768, 32, 768
    x = torch.randn(B, T, dim, device="cuda", dtype=torch.bfloat16)
    A = torch.randn(dim, rank, device="cuda", dtype=torch.float32) * 0.01
    Bm = torch.randn(rank, out_dim, device="cuda", dtype=torch.float32)
    scale = 0.177

    ref = ((x @ A.to(x.dtype)) @ Bm.to(x.dtype)) * scale
    out = fused_lora_forward(x, A, Bm, scale)
    diff = (ref.float() - out.float()).abs().max().item()
    assert diff < 0.01, f"lora_forward max diff: {diff}"
    print(f"lora_forward: PASSED (max diff: {diff:.6f})")


if __name__ == "__main__":
    test_permute_mix()
    test_rms_norm_streams()
    test_lora_forward()
    print("\nAll tests passed!")
