from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class LinearSelfAttention(nn.Module):
    """MobileViTv2-style linear self-attention (baseline)."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.qkv = nn.Linear(embed_dim, 1 + 2 * embed_dim, bias=True)
        self.out = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        b, l, d = x.shape
        qkv = self.qkv(x)
        q = qkv[..., :1]
        k = qkv[..., 1 : 1 + d]
        v = qkv[..., 1 + d :]
        w = torch.softmax(q, dim=1)
        context = torch.sum(w * k, dim=1, keepdim=True)
        fused = F.gelu(v) * context
        return self.out(fused)


class FactorizedLinearSelfAttention(nn.Module):
    """Lite++ attention: factorized (low-rank) K/V projection, scalar-q attention per token."""

    def __init__(self, embed_dim: int, attn_rank: int = 64, out_rank: int = 0, dropout: float = 0.0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attn_rank = max(1, int(attn_rank))

        self.q = nn.Linear(embed_dim, 1, bias=True)
        self.kv_down = nn.Linear(embed_dim, self.attn_rank, bias=True)
        self.kv_up = nn.Linear(self.attn_rank, 2 * embed_dim, bias=True)

        out_rank = int(out_rank)
        if out_rank and 0 < out_rank < embed_dim:
            self.out = nn.Sequential(
                nn.Linear(embed_dim, out_rank, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(out_rank, embed_dim, bias=True),
            )
        else:
            self.out = nn.Linear(embed_dim, embed_dim, bias=True)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, l, d = x.shape
        if d != self.embed_dim:
            raise RuntimeError(f"Expected D={self.embed_dim}, got D={d}")

        q = self.q(x)                          # [B, L, 1]
        w = torch.softmax(q, dim=1)            # [B, L, 1]

        kv = self.kv_up(self.kv_down(x))       # [B, L, 2D]
        k, v = kv[..., :d], kv[..., d:]        # [B, L, D], [B, L, D]

        context = torch.sum(w * k, dim=1, keepdim=True)  # [B, 1, D]
        fused = F.gelu(v) * context                      # [B, L, D]

        y = self.out(fused)
        y = self.drop(y)
        return y
