from __future__ import annotations

from typing import Tuple, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from mobileplantvit.models.attention import FactorizedLinearSelfAttention, LinearSelfAttention

Tensor = torch.Tensor

# Type hints for ablation switches
AttnType = Literal["factorized", "linear"]
FFNType = Literal["tokenconv", "mlp"]


class EncoderBlock(nn.Module):
    """Baseline transformer-ish encoder: linear attention + MLP FFN."""

    def __init__(self, embed_dim: int, ffn_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.attn = LinearSelfAttention(embed_dim)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x))
        return x


class MLPFFN(nn.Module):
    """Standard MLP FFN: Linear -> GELU -> Dropout -> Linear -> Dropout.
    
    For ablation: compare against TokenConvFFN.
    """

    def __init__(self, embed_dim: int, expand_ratio: float = 2.0, dropout: float = 0.2) -> None:
        super().__init__()
        hidden = max(8, int(embed_dim * float(expand_ratio)))
        self.fc1 = nn.Linear(embed_dim, hidden)
        self.fc2 = nn.Linear(hidden, embed_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, hw: Tuple[int, int]) -> Tensor:
        # hw is ignored for MLP (no spatial structure needed)
        _ = hw
        y = self.fc1(x)
        y = self.act(y)
        y = self.drop(y)
        y = self.fc2(y)
        y = self.drop(y)
        return y


class TokenConvFFN(nn.Module):
    """Lite++ FFN: inverted bottleneck + depthwise conv in token grid space."""

    def __init__(self, embed_dim: int, expand_ratio: float = 2.0, kernel_size: int = 3, dropout: float = 0.2) -> None:
        super().__init__()
        hidden = max(8, int(embed_dim * float(expand_ratio)))
        pad = (kernel_size - 1) // 2

        self.pw1 = nn.Conv2d(embed_dim, hidden, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.dw = nn.Conv2d(hidden, hidden, kernel_size=kernel_size, padding=pad, groups=hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pw2 = nn.Conv2d(hidden, embed_dim, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim)

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, hw: Tuple[int, int]) -> Tensor:
        b, l, d = x.shape
        h, w = hw
        if h * w != l:
            raise RuntimeError(f"TokenConvFFN expects H*W=L, got H*W={h*w}, L={l}")

        x_map = x.transpose(1, 2).contiguous().view(b, d, h, w)

        y = self.pw1(x_map)
        y = self.bn1(y)
        y = self.act(y)
        y = self.drop(y)

        y = self.dw(y)
        y = self.bn2(y)
        y = self.act(y)
        y = self.drop(y)

        y = self.pw2(y)
        y = self.bn3(y)
        y = self.drop(y)

        y = y.view(b, d, h * w).transpose(1, 2).contiguous()
        return y


class TokenPool2x2(nn.Module):
    """Optional: reduce tokens by 4x by pooling 2x2 token neighborhoods."""

    def __init__(self, pool_type: str = "avg") -> None:
        super().__init__()
        pool_type = pool_type.lower().strip()
        if pool_type not in ("avg", "max"):
            raise ValueError("pool_type must be 'avg' or 'max'")
        self.pool_type = pool_type

    def forward(self, x: Tensor, hw: Tuple[int, int]) -> Tuple[Tensor, Tuple[int, int]]:
        b, l, d = x.shape
        h, w = hw
        if h * w != l:
            raise RuntimeError(f"TokenPool2x2 expects H*W=L, got H*W={h*w}, L={l}")
        if h % 2 != 0 or w % 2 != 0:
            raise RuntimeError(f"TokenPool2x2 requires even H,W. Got H={h}, W={w}")

        x_map = x.transpose(1, 2).contiguous().view(b, d, h, w)

        if self.pool_type == "avg":
            pooled = F.avg_pool2d(x_map, kernel_size=2, stride=2)
        else:
            pooled = F.max_pool2d(x_map, kernel_size=2, stride=2)

        h2, w2 = h // 2, w // 2
        pooled_tokens = pooled.view(b, d, h2 * w2).transpose(1, 2).contiguous()
        return pooled_tokens, (h2, w2)


class EncoderBlockLitePP(nn.Module):
    """Lite++ encoder with configurable attention and FFN types for ablation.
    
    Args:
        embed_dim: Embedding dimension
        attn_rank: Rank for factorized attention (only used when attn_type='factorized')
        attn_out_rank: Output rank for attention
        ffn_expand: FFN expansion ratio
        ffn_kernel: Kernel size for TokenConvFFN (only used when ffn_type='tokenconv')
        dropout: Dropout rate
        attn_type: 'factorized' (default, Lite++) or 'linear' (baseline-style)
        ffn_type: 'tokenconv' (default, Lite++) or 'mlp' (baseline-style)
    """

    def __init__(
        self,
        embed_dim: int,
        attn_rank: int = 64,
        attn_out_rank: int = 0,
        ffn_expand: float = 2.0,
        ffn_kernel: int = 3,
        dropout: float = 0.2,
        attn_type: AttnType = "factorized",
        ffn_type: FFNType = "tokenconv",
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Attention selection for ablation
        attn_type = (attn_type or "factorized").lower().strip()
        if attn_type == "factorized":
            self.attn = FactorizedLinearSelfAttention(
                embed_dim, attn_rank=attn_rank, out_rank=attn_out_rank, dropout=dropout
            )
        elif attn_type == "linear":
            self.attn = LinearSelfAttention(embed_dim)
        else:
            raise ValueError(f"attn_type must be 'factorized' or 'linear', got '{attn_type}'")
        
        # FFN selection for ablation
        ffn_type = (ffn_type or "tokenconv").lower().strip()
        if ffn_type == "tokenconv":
            self.ffn = TokenConvFFN(embed_dim, expand_ratio=ffn_expand, kernel_size=ffn_kernel, dropout=dropout)
        elif ffn_type == "mlp":
            self.ffn = MLPFFN(embed_dim, expand_ratio=ffn_expand, dropout=dropout)
        else:
            raise ValueError(f"ffn_type must be 'tokenconv' or 'mlp', got '{ffn_type}'")

    def forward(self, x: Tensor, hw: Tuple[int, int]) -> Tensor:
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ffn(x, hw))
        return x
