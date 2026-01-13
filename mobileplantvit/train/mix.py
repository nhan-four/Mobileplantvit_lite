from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

Tensor = torch.Tensor


def mixup_data(x: Tensor, y: Tensor, alpha: float, device: torch.device) -> Tuple[Tensor, Tensor, Tensor, float]:
    """Classic MixUp: x' = lam*x + (1-lam)*x[perm]."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=device)
    mixed = lam * x + (1.0 - lam) * x[index]
    return mixed, y, y[index], lam


def cutmix_data(x: Tensor, y: Tensor, alpha: float, device: torch.device) -> Tuple[Tensor, Tensor, Tensor, float]:
    """CutMix: replace a random rectangle with a permuted sample."""
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=device)

    _, _, height, width = x.shape
    cut_ratio = np.sqrt(1.0 - lam)
    cut_w = int(width * cut_ratio)
    cut_h = int(height * cut_ratio)

    cx = np.random.randint(width)
    cy = np.random.randint(height)

    x1 = int(np.clip(cx - cut_w // 2, 0, width))
    y1 = int(np.clip(cy - cut_h // 2, 0, height))
    x2 = int(np.clip(cx + cut_w // 2, 0, width))
    y2 = int(np.clip(cy + cut_h // 2, 0, height))

    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]

    lam = 1.0 - ((x2 - x1) * (y2 - y1) / (width * height))
    return mixed, y, y[index], float(lam)


def mix_criterion(criterion: nn.Module, logits: Tensor, y_a: Tensor, y_b: Tensor, lam: float) -> Tensor:
    """Compute MixUp/CutMix loss."""
    return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
