"""Kernel 2: Fused RMSNorm + stream reshape.

Replaces reshape + F.rms_norm with a single kernel that reads
(B,T,n,D) and writes (B,T,n*D) normalized — no intermediate tensor.
"""
import torch
from torch import Tensor

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def _rms_norm_streams_kernel(
        x_ptr, out_ptr,
        nD,
        stride_bt,
        eps: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        bt = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < nD
        x = tl.load(x_ptr + bt * stride_bt + offs, mask=mask, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=0) / nD
        rrms = 1.0 / tl.sqrt(ms + eps)
        tl.store(out_ptr + bt * stride_bt + offs, (x * rrms).to(tl.bfloat16), mask=mask)


def fused_rms_norm_streams(x_streams: Tensor, eps: float = 1e-5) -> Tensor:
    """(B,T,n,D) → (B,T,n*D) with RMSNorm, single kernel."""
    B, T, n, D = x_streams.shape
    nD = n * D
    x_flat = x_streams.reshape(B * T, nD).contiguous()
    out = torch.empty_like(x_flat)
    BLOCK = triton.next_power_of_2(nD)
    _rms_norm_streams_kernel[(B * T,)](
        x_flat, out, nD, x_flat.stride(0), eps=eps, BLOCK=BLOCK,
    )
    return out.reshape(B, T, nD)
