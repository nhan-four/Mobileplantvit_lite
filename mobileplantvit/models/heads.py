from __future__ import annotations

import torch
import torch.nn as nn

Tensor = torch.Tensor


class ClassificationHead(nn.Module):
    """Token-average pooling (GAP over tokens) -> Dropout -> Linear."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        x = x.mean(dim=1)
        x = self.drop(x)
        return self.fc(x)


class AttentionPoolingHead(nn.Module):
    """Attention pooling over tokens -> Dropout -> Linear."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.score = nn.Linear(embed_dim, 1, bias=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        w = torch.softmax(self.score(x), dim=1)  # [B, L, 1]
        x = (w * x).sum(dim=1)                   # [B, D]
        x = self.drop(x)
        return self.fc(x)
