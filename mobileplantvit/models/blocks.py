from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mobileplantvit.utils.init import trunc_normal_

Tensor = torch.Tensor


class DepthConvBlock(nn.Module):
    """Depthwise-separable conv: DWConv -> PWConv -> BN -> GELU (+ optional residual)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_residual: bool = False,
    ) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.use_residual = bool(use_residual and stride == 1 and in_channels == out_channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_residual:
            x = x + identity
        return x


class GroupConvBlock(nn.Module):
    """Group conv + BN + GELU (+ residual when stride=1)."""

    def __init__(self, channels: int, groups: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        groups = max(int(groups), 1)
        groups = min(groups, channels)
        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(channels)
        self.act = nn.GELU()
        self.use_residual = stride == 1

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        if self.use_residual:
            x = x + identity
        return x


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
        )

    def forward(self, x: Tensor) -> Tensor:
        b, c, _, _ = x.shape
        avg = F.adaptive_avg_pool2d(x, 1).view(b, c)
        mx = F.adaptive_max_pool2d(x, 1).view(b, c)
        attn = torch.sigmoid(self.mlp(avg) + self.mlp(mx)).view(b, c, 1, 1)
        return x * attn


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel + spatial)."""

    def __init__(self, channels: int, reduction: int = 16, sa_kernel: int = 7) -> None:
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(sa_kernel)

    def forward(self, x: Tensor) -> Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int,
        feature_map_size: Tuple[int, int],
        use_cbam: bool = True,
    ) -> None:
        super().__init__()
        fh, fw = feature_map_size
        
        # Xử lý patch_size=0 -> 1
        if patch_size == 0:
            patch_size = 1
            
        if fh % patch_size != 0 or fw % patch_size != 0:
            raise ValueError(f"feature_map_size={feature_map_size} must be divisible by patch_size={patch_size}")

        self.proj = DepthConvBlock(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
            use_residual=False,
        )
        self.cbam = CBAM(embed_dim) if use_cbam else nn.Identity()

        # --- THAY ĐỔI Ở ĐÂY: DÙNG CPE ---
        # Thay vì dùng self.pos = nn.Parameter(...) khổng lồ
        # Ta dùng 1 lớp Conv 3x3 Depthwise (PEG) siêu nhẹ (~2.5K tham số)
        self.peg = nn.Conv2d(
            embed_dim, 
            embed_dim, 
            kernel_size=3, 
            padding=1, 
            groups=embed_dim, 
            bias=True
        )
        # Khởi tạo weights cho PEG
        nn.init.normal_(self.peg.weight, std=0.02)
        nn.init.constant_(self.peg.bias, 0)
        # --------------------------------

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)
        x = self.cbam(x) # Output shape: [B, Embed_Dim, H, W]

        # --- ÁP DỤNG CPE (PEG) ---
        # Cộng thông tin vị trí ngay trên không gian 2D trước khi flatten
        x = x + self.peg(x) 
        # -------------------------

        b, d, h, w = x.shape
        tokens = x.view(b, d, h * w).transpose(1, 2)  # [B, L, D]

        # Không cần cộng self.pos ở đây nữa vì đã cộng self.peg(x) ở trên rồi
        return tokens