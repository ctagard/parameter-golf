"""Public API with graceful PyTorch fallbacks.

Usage in train_gpt.py:
    try:
        from looped_gpt_kernels import ops as lgk
        HAS_LGK = True
    except ImportError:
        HAS_LGK = False
"""
import torch
from torch import Tensor
import torch.nn.functional as F

try:
    from .kernels.mhc_permute import FusedPermuteMix, get_perm_indices, HAS_TRITON as _HAS_PERM
    from .kernels.mhc_norm import fused_rms_norm_streams, HAS_TRITON as _HAS_NORM
    from .kernels.mix_to_one import fused_mix_to_one
    from .kernels.lora_forward import fused_lora_forward
    TRITON_AVAILABLE = _HAS_PERM and _HAS_NORM
except ImportError:
    TRITON_AVAILABLE = False


def permute_mix(x_streams: Tensor, res_weights: Tensor, perm_indices: Tensor,
                perm_matrices: Tensor | None = None) -> Tensor:
    """Fused permutation routing. Falls back to einsum if Triton unavailable."""
    if TRITON_AVAILABLE:
        return FusedPermuteMix.apply(x_streams, res_weights, perm_indices)
    if perm_matrices is None:
        raise ValueError("perm_matrices required for PyTorch fallback")
    H_res = torch.einsum("btp,pij->btij", res_weights, perm_matrices)
    return torch.einsum("btij,btjd->btid", H_res, x_streams)


def rms_norm_streams(x_streams: Tensor, eps: float = 1e-5) -> Tensor:
    """Fused RMSNorm over streams. Falls back to reshape+rms_norm."""
    if TRITON_AVAILABLE:
        return fused_rms_norm_streams(x_streams, eps)
    B, T, n, D = x_streams.shape
    return F.rms_norm(x_streams.reshape(B, T, n * D), (n * D,))


def mix_to_one(x_streams: Tensor, W_pre: Tensor, alpha_pre: Tensor) -> Tensor:
    """Fused stream aggregation."""
    return fused_mix_to_one(x_streams, W_pre, alpha_pre)


def lora_forward(x: Tensor, A: Tensor, B: Tensor, scale: float) -> Tensor:
    """Fused LoRA matmul."""
    return fused_lora_forward(x, A, B, scale)
