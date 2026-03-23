"""Kernel 3: Fused mix_to_one (RMSNorm → linear → sigmoid → weighted sum).

Currently a placeholder — uses PyTorch ops. The full fusion is complex
because it combines normalization, a linear projection, and a reduction.
Implement after Kernel 1 and 2 are validated.
"""
import torch
from torch import Tensor
import torch.nn.functional as F


def fused_mix_to_one(x_streams: Tensor, W_pre: Tensor, alpha_pre: Tensor) -> Tensor:
    """PyTorch implementation — replace with Triton kernel later."""
    B, T, n, D = x_streams.shape
    x_flat = F.rms_norm(x_streams.reshape(B, T, n * D), (n * D,))
    H_pre = torch.sigmoid(alpha_pre * F.linear(x_flat, W_pre))
    return (H_pre.unsqueeze(-1) * x_streams).sum(dim=2)
