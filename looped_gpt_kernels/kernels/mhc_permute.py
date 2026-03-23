"""Kernel 1: Fused permutation routing for mHC-Lite.

Replaces two einsum ops with a single fused kernel that does
weighted gather across permutations — no intermediate H_res tensor.
"""
import torch
from torch import Tensor
from itertools import permutations

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def get_perm_indices(n_streams: int) -> Tensor:
    """Precompute permutation index table: (n_perms, n_streams) int32."""
    perms = list(permutations(range(n_streams)))
    return torch.tensor(perms, dtype=torch.int32)


if HAS_TRITON:
    @triton.jit
    def _mhc_permute_fwd(
        x_ptr, w_ptr, perm_ptr, out_ptr,
        n: tl.constexpr, D: tl.constexpr, n_perms: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bt = tl.program_id(0)
        i = tl.program_id(1)
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        x_bt_base = bt * n * D
        for p in range(n_perms):
            w = tl.load(w_ptr + bt * n_perms + p)
            j = tl.load(perm_ptr + p * n + i)
            x = tl.load(x_ptr + x_bt_base + j * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
            acc += w * x
        tl.store(out_ptr + x_bt_base + i * D + d_offs, acc.to(tl.bfloat16), mask=d_mask)

    @triton.jit
    def _mhc_permute_bwd_x(
        grad_out_ptr, w_ptr, perm_ptr, grad_x_ptr,
        n: tl.constexpr, D: tl.constexpr, n_perms: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        bt = tl.program_id(0)
        j = tl.program_id(1)
        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        g_bt_base = bt * n * D
        for p in range(n_perms):
            w = tl.load(w_ptr + bt * n_perms + p)
            for i in range(n):
                perm_j = tl.load(perm_ptr + p * n + i)
                if perm_j == j:
                    g = tl.load(grad_out_ptr + g_bt_base + i * D + d_offs, mask=d_mask, other=0.0).to(tl.float32)
                    acc += w * g
        tl.store(grad_x_ptr + g_bt_base + j * D + d_offs, acc.to(tl.bfloat16), mask=d_mask)


class FusedPermuteMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_streams, res_weights, perm_indices):
        B, T, n, D = x_streams.shape
        n_perms = res_weights.shape[-1]
        x_flat = x_streams.contiguous().reshape(B * T, n, D)
        w_flat = res_weights.contiguous().reshape(B * T, n_perms)
        out_flat = torch.empty_like(x_flat)
        BLOCK_D = triton.next_power_of_2(D)
        _mhc_permute_fwd[(B * T, n)](
            x_flat, w_flat, perm_indices, out_flat,
            n, D, n_perms, BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(x_streams, res_weights, perm_indices)
        return out_flat.reshape(B, T, n, D)

    @staticmethod
    def backward(ctx, grad_out):
        x_streams, res_weights, perm_indices = ctx.saved_tensors
        B, T, n, D = x_streams.shape
        n_perms = res_weights.shape[-1]
        BLOCK_D = triton.next_power_of_2(D)
        g_flat = grad_out.contiguous().reshape(B * T, n, D)
        w_flat = res_weights.contiguous().reshape(B * T, n_perms)
        grad_x_flat = torch.zeros(B * T, n, D, device=g_flat.device, dtype=g_flat.dtype)
        _mhc_permute_bwd_x[(B * T, n)](
            g_flat, w_flat, perm_indices, grad_x_flat,
            n, D, n_perms, BLOCK_D=BLOCK_D,
        )
        # grad w.r.t. res_weights: PyTorch fallback
        grad_w = torch.zeros_like(res_weights)
        perm_np = perm_indices.cpu()
        for p in range(n_perms):
            gathered = x_streams[:, :, perm_np[p].long(), :]
            grad_w[:, :, p] = (grad_out * gathered).sum(dim=(-1, -2))
        return grad_x_flat.reshape(B, T, n, D), grad_w, None
