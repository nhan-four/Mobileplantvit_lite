from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from mobileplantvit.models.blocks import CBAM, DepthConvBlock, GroupConvBlock, PatchEmbedding
from mobileplantvit.models.encoder import EncoderBlock, EncoderBlockLitePP, TokenPool2x2
from mobileplantvit.models.heads import AttentionPoolingHead, ClassificationHead
from mobileplantvit.utils.init import make_divisible

Tensor = torch.Tensor


class MobilePlantViT(nn.Module):
    """Baseline MobilePlantViT (your current backbone + patchify + linear-attention encoder)."""

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        ffn_dim: int = 512,
        patch_size: int = 4,
        encoder_depth: int = 1,
        encoder_dropout: float = 0.2,
        classifier_dropout: float = 0.2,
        width_mult: float = 1.0,
        head_type: str = "attn",
    ) -> None:
        super().__init__()
        
        # --- SỬA LỖI: Xử lý patch_size=0 ---
        if patch_size == 0:
            print(f"[MobilePlantViT] Lưu ý: Patch Size = 0 được chuyển thành 1 (Pointwise) theo logic của paper.")
            patch_size = 1
        # ------------------------------------
        
        self.image_size = image_size

        c0 = make_divisible(int(32 * width_mult))
        c1 = make_divisible(int(64 * width_mult))
        c2 = make_divisible(int(128 * width_mult))

        g0 = max(c0 // 2, 1)
        g1 = max(c1 // 2, 1)
        g2 = max(c2 // 2, 1)

        self.stem = DepthConvBlock(in_channels, c0, kernel_size=3, stride=1, padding=1)

        self.stage1 = nn.Sequential(
            GroupConvBlock(c0, groups=g0),
            CBAM(c0),
            DepthConvBlock(c0, c1, kernel_size=3, stride=2, padding=1),
        )

        self.stage2 = nn.Sequential(
            GroupConvBlock(c1, groups=g1),
            GroupConvBlock(c1, groups=g1),
            CBAM(c1),
            DepthConvBlock(c1, c2, kernel_size=3, stride=2, padding=1),
        )

        self.stage3 = nn.Sequential(
            GroupConvBlock(c2, groups=g2),
            GroupConvBlock(c2, groups=g2),
            GroupConvBlock(c2, groups=g2),
            GroupConvBlock(c2, groups=g2),
            CBAM(c2),
        )

        feat_h = image_size[0] // 4
        feat_w = image_size[1] // 4
        self.patch_embedding = PatchEmbedding(
            in_channels=c2,
            embed_dim=embed_dim,
            patch_size=patch_size,
            feature_map_size=(feat_h, feat_w),
            use_cbam=True,
        )

        self.encoder = nn.Sequential(
            *[EncoderBlock(embed_dim=embed_dim, ffn_dim=ffn_dim, dropout=encoder_dropout) for _ in range(encoder_depth)]
        )

        self.classifier = _make_head(head_type, embed_dim, num_classes, classifier_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.patch_embedding(x)  # [B, L, D]
        x = self.encoder(x)
        return self.classifier(x)


class MobilePlantViT_LitePP(nn.Module):
    """Lite++: Factorized Linear SA + TokenConv-FFN + optional token pooling.
    
    With ablation switches:
        - attn_type: 'factorized' (default) or 'linear' (for ablation)
        - ffn_type: 'tokenconv' (default) or 'mlp' (for ablation)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 256,
        patch_size: int = 4,
        encoder_depth: int = 2,
        encoder_dropout: float = 0.2,
        classifier_dropout: float = 0.2,
        width_mult: float = 1.0,
        head_type: str = "attn",
        attn_rank: int = 64,
        attn_out_rank: int = 0,
        ffn_expand: float = 2.0,
        ffn_kernel: int = 3,
        use_token_pool: bool = False,
        pool_at: int = 1,
        pool_type: str = "avg",
        # Ablation switches
        attn_type: str = "factorized",
        ffn_type: str = "tokenconv",
    ) -> None:
        super().__init__()
        
        # --- SỬA LỖI: Xử lý patch_size=0 ---
        if patch_size == 0:
            print(f"[MobilePlantViT_LitePP] Lưu ý: Patch Size = 0 được chuyển thành 1 (Pointwise) theo logic của paper.")
            patch_size = 1
        # ------------------------------------
        
        self.image_size = image_size

        c0 = make_divisible(int(32 * width_mult))
        c1 = make_divisible(int(64 * width_mult))
        c2 = make_divisible(int(128 * width_mult))

        g0 = max(c0 // 2, 1)
        g1 = max(c1 // 2, 1)
        g2 = max(c2 // 2, 1)

        self.stem = DepthConvBlock(in_channels, c0, kernel_size=3, stride=1, padding=1)

        self.stage1 = nn.Sequential(
            GroupConvBlock(c0, groups=g0),
            CBAM(c0),
            DepthConvBlock(c0, c1, kernel_size=3, stride=2, padding=1),
        )

        self.stage2 = nn.Sequential(
            GroupConvBlock(c1, groups=g1),
            GroupConvBlock(c1, groups=g1),
            CBAM(c1),
            DepthConvBlock(c1, c2, kernel_size=3, stride=2, padding=1),
        )

        self.stage3 = nn.Sequential(
            GroupConvBlock(c2, groups=g2),
            GroupConvBlock(c2, groups=g2),
            GroupConvBlock(c2, groups=g2),
            GroupConvBlock(c2, groups=g2),
            CBAM(c2),
        )

        feat_h = image_size[0] // 4
        feat_w = image_size[1] // 4
        self.patch_embedding = PatchEmbedding(
            in_channels=c2,
            embed_dim=embed_dim,
            patch_size=patch_size,
            feature_map_size=(feat_h, feat_w),
            use_cbam=True,
        )

        tokens_h = feat_h // patch_size
        tokens_w = feat_w // patch_size
        self.initial_hw = (tokens_h, tokens_w)

        depth = int(encoder_depth)
        if depth <= 0:
            raise ValueError("encoder_depth must be >= 1")

        pool_at = int(pool_at) if bool(use_token_pool) else 0

        self.encoder_layers = nn.ModuleList()
        for block_index in range(1, depth + 1):
            self.encoder_layers.append(
                EncoderBlockLitePP(
                    embed_dim=embed_dim,
                    attn_rank=attn_rank,
                    attn_out_rank=attn_out_rank,
                    ffn_expand=ffn_expand,
                    ffn_kernel=ffn_kernel,
                    dropout=encoder_dropout,
                    attn_type=attn_type,
                    ffn_type=ffn_type,
                )
            )
            if pool_at > 0 and block_index == pool_at:
                self.encoder_layers.append(TokenPool2x2(pool_type=pool_type))

        self.classifier = _make_head(head_type, embed_dim, num_classes, classifier_dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        x = self.patch_embedding(x)  # [B, L, D]
        hw = self.initial_hw

        for layer in self.encoder_layers:
            if isinstance(layer, TokenPool2x2):
                x, hw = layer(x, hw)
            else:
                x = layer(x, hw)

        return self.classifier(x)


def _make_head(head_type: str, embed_dim: int, num_classes: int, dropout: float) -> nn.Module:
    head_type = (head_type or "attn").lower().strip()
    if head_type == "gap":
        return ClassificationHead(embed_dim, num_classes, dropout=dropout)
    if head_type == "attn":
        return AttentionPoolingHead(embed_dim, num_classes, dropout=dropout)
    raise ValueError("head_type must be 'gap' or 'attn'")


def build_model(args, num_classes: int) -> nn.Module:
    """Model factory to keep main/pretrain code DRY."""
    common = dict(
        image_size=(args.img_size, args.img_size),
        in_channels=3,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        encoder_depth=args.encoder_depth,
        encoder_dropout=args.encoder_dropout,
        classifier_dropout=args.classifier_dropout,
        width_mult=args.width_mult,
        head_type=args.head_type,
    )

    if args.model == "baseline":
        return MobilePlantViT(**common, ffn_dim=args.ffn_dim)

    return MobilePlantViT_LitePP(
        **common,
        attn_rank=args.attn_rank,
        attn_out_rank=args.attn_out_rank,
        ffn_expand=args.ffn_expand,
        ffn_kernel=args.ffn_kernel,
        use_token_pool=bool(args.use_token_pool and args.pool_at > 0),
        pool_at=args.pool_at,
        pool_type=args.pool_type,
        # Ablation switches
        attn_type=getattr(args, "attn_type", "factorized"),
        ffn_type=getattr(args, "ffn_type", "tokenconv"),
    )
