"""Kernel 4: Fused LoRA forward (x @ A @ B * scale in one pass).

Currently a placeholder — uses PyTorch ops. Rank=32 is small enough
that the intermediate fits in L1 cache, making this a good fusion target.
Implement after Kernels 1-3 are validated.
"""
import torch
from torch import Tensor


def fused_lora_forward(x: Tensor, A: Tensor, B: Tensor, scale: float) -> Tensor:
    """PyTorch implementation — replace with Triton kernel later."""
    return ((x @ A.to(x.dtype)) @ B.to(x.dtype)) * scale
