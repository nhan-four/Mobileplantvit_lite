from __future__ import annotations

import torch

Tensor = torch.Tensor


def trunc_normal_(tensor: Tensor, std: float = 0.02) -> Tensor:
    """Truncated normal N(0,std) clipped at ±2σ (simple, deterministic enough)."""
    with torch.no_grad():
        size = tensor.shape
        tmp = tensor.new_empty(size + (4,)).normal_()
        valid = (tmp < 2) & (tmp > -2)
        idx = valid.max(-1, keepdim=True)[1]
        tensor.copy_(tmp.gather(-1, idx).squeeze(-1))
        tensor.mul_(std)
    return tensor


def make_divisible(value: int, divisor: int = 8) -> int:
    """Round up channels so they are divisible by *divisor* (hardware-friendly)."""
    return int((value + divisor - 1) // divisor * divisor)
