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
        B, T: tl.constexpr, n: tl.constexpr, D: tl.constexpr,
        n_perms: tl.constexpr,
        stride_x_bt, stride_x_n, stride_x_d,
        stride_w_bt, stride_w_p,
        stride_o_bt, stride_o_n, stride_o_d,
        BLOCK_D: tl.constexpr,
    ):
        bt = tl.program_id(0)
        i = tl.program_id(1)

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D

        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for p in range(n_perms):
            w = tl.load(w_ptr + bt * stride_w_bt + p * stride_w_p)
            j = tl.load(perm_ptr + p * n + i)
            x = tl.load(x_ptr + bt * stride_x_bt + j * stride_x_n + d_offs * stride_x_d,
                        mask=d_mask, other=0.0).to(tl.float32)
            acc += w * x

        tl.store(out_ptr + bt * stride_o_bt + i * stride_o_n + d_offs * stride_o_d,
                 acc.to(tl.bfloat16), mask=d_mask)

    @triton.jit
    def _mhc_permute_bwd_x(
        grad_out_ptr, w_ptr, perm_ptr, grad_x_ptr,
        B, T: tl.constexpr, n: tl.constexpr, D: tl.constexpr,
        n_perms: tl.constexpr,
        stride_g_bt, stride_g_n, stride_g_d,
        stride_w_bt, stride_w_p,
        stride_gx_bt, stride_gx_n, stride_gx_d,
        BLOCK_D: tl.constexpr,
    ):
        """Backward w.r.t. x_streams: grad_x[b,t,j,d] = sum_p w[b,t,p] * sum_i(perm[p,i]==j) * grad_out[b,t,i,d]"""
        bt = tl.program_id(0)
        j = tl.program_id(1)  # source stream index

        d_offs = tl.arange(0, BLOCK_D)
        d_mask = d_offs < D
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for p in range(n_perms):
            w = tl.load(w_ptr + bt * stride_w_bt + p * stride_w_p)
            # Find which output stream i this source j maps to under perm p
            for i in range(n):
                perm_j = tl.load(perm_ptr + p * n + i)
                if perm_j == j:
                    g = tl.load(grad_out_ptr + bt * stride_g_bt + i * stride_g_n + d_offs * stride_g_d,
                                mask=d_mask, other=0.0).to(tl.float32)
                    acc += w * g

        tl.store(grad_x_ptr + bt * stride_gx_bt + j * stride_gx_n + d_offs * stride_gx_d,
                 acc.to(tl.bfloat16), mask=d_mask)


class FusedPermuteMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_streams, res_weights, perm_indices):
        B, T, n, D = x_streams.shape
        n_perms = res_weights.shape[-1]
        out = torch.empty_like(x_streams)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B * T, n)

        _mhc_permute_fwd[grid](
            x_streams, res_weights, perm_indices, out,
            B, T, n, D, n_perms,
            x_streams.stride(0) * x_streams.stride(1) // max(x_streams.stride(1), 1),
            *x_streams.stride()[1:],
            *res_weights.stride(),
            *out.stride()[1:],
            BLOCK_D=BLOCK_D,
        )
        ctx.save_for_backward(x_streams, res_weights, perm_indices)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x_streams, res_weights, perm_indices = ctx.saved_tensors
        B, T, n, D = x_streams.shape
        n_perms = res_weights.shape[-1]
        BLOCK_D = triton.next_power_of_2(D)

        # grad w.r.t. x_streams
        grad_x = torch.zeros_like(x_streams)
        _mhc_permute_bwd_x[(B * T, n)](
            grad_out, res_weights, perm_indices, grad_x,
            B, T, n, D, n_perms,
            *grad_out.stride()[1:],
            *res_weights.stride(),
            *grad_x.stride()[1:],
            BLOCK_D=BLOCK_D,
        )

        # grad w.r.t. res_weights: simple PyTorch for now
        # grad_w[b,t,p] = sum_i sum_d grad_out[b,t,i,d] * x_streams[b,t,perm[p,i],d]
        grad_w = torch.zeros_like(res_weights)
        perm_np = perm_indices.cpu()
        for p in range(n_perms):
            gathered = x_streams[:, :, perm_np[p].long(), :]  # (B,T,n,D)
            grad_w[:, :, p] = (grad_out * gathered).sum(dim=(-1, -2))

        return grad_x, grad_w, None
