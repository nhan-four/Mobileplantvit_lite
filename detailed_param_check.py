#!/usr/bin/env python3
"""Kiểm tra chi tiết số lượng params từng module."""

import torch.nn as nn
from mobileplantvit.models.encoder import EncoderBlockLitePP, MLPFFN
from mobileplantvit.models.attention import LinearSelfAttention

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

# Tạo các module
embed_dim = 256
ffn_expand = 2.0
ffn_dim = int(embed_dim * ffn_expand)

# LinearSelfAttention
attn = LinearSelfAttention(embed_dim)
print("LinearSelfAttention:")
print(f"  qkv: {count_params(attn.qkv):,}")
print(f"    - weight: {attn.qkv.weight.numel():,}")
print(f"    - bias: {attn.qkv.bias.numel() if attn.qkv.bias is not None else 0:,}")
print(f"  out: {count_params(attn.out):,}")
print(f"    - weight: {attn.out.weight.numel():,}")
print(f"    - bias: {attn.out.bias.numel() if attn.out.bias is not None else 0:,}")
print(f"  Total attn: {count_params(attn):,}")

# MLPFFN
ffn = MLPFFN(embed_dim, expand_ratio=ffn_expand)
print("\nMLPFFN:")
print(f"  fc1: {count_params(ffn.fc1):,}")
print(f"    - weight: {ffn.fc1.weight.numel():,}")
print(f"    - bias: {ffn.fc1.bias.numel() if ffn.fc1.bias is not None else 0:,}")
print(f"  fc2: {count_params(ffn.fc2):,}")
print(f"    - weight: {ffn.fc2.weight.numel():,}")
print(f"    - bias: {ffn.fc2.bias.numel() if ffn.fc2.bias is not None else 0:,}")
print(f"  Total ffn: {count_params(ffn):,}")

# LayerNorm
norm1 = nn.LayerNorm(embed_dim)
norm2 = nn.LayerNorm(embed_dim)
print("\nLayerNorm (2x):")
print(f"  norm1: {count_params(norm1):,}")
print(f"  norm2: {count_params(norm2):,}")
print(f"  Total norms: {count_params(norm1) + count_params(norm2):,}")

# EncoderBlock
block = EncoderBlockLitePP(
    embed_dim=embed_dim,
    attn_type="linear",
    ffn_type="mlp",
    ffn_expand=ffn_expand
)
print("\nEncoderBlockLitePP:")
print(f"  Total: {count_params(block):,}")
print(f"  Breakdown:")
print(f"    - attn: {count_params(block.attn):,}")
print(f"    - norm1: {count_params(block.norm1):,}")
print(f"    - norm2: {count_params(block.norm2):,}")
print(f"    - ffn: {count_params(block.ffn):,}")

# Tính toán thủ công
qkv_params = embed_dim * (1 + 2*embed_dim) + (1 + 2*embed_dim)  # weight + bias
out_params = embed_dim * embed_dim + embed_dim  # weight + bias
attn_total = qkv_params + out_params

fc1_params = embed_dim * ffn_dim + ffn_dim  # weight + bias
fc2_params = ffn_dim * embed_dim + embed_dim  # weight + bias
ffn_total = fc1_params + fc2_params

norm_total = embed_dim * 2 * 2  # 2 norms, each has weight + bias

block_total = attn_total + norm_total + ffn_total

print("\n" + "="*60)
print("TÍNH TOÁN THỦ CÔNG (có bias):")
print("="*60)
print(f"  qkv: {qkv_params:,} (weight: {embed_dim * (1 + 2*embed_dim):,}, bias: {1 + 2*embed_dim:,})")
print(f"  out: {out_params:,} (weight: {embed_dim * embed_dim:,}, bias: {embed_dim:,})")
print(f"  attn_total: {attn_total:,}")
print(f"  fc1: {fc1_params:,} (weight: {embed_dim * ffn_dim:,}, bias: {ffn_dim:,})")
print(f"  fc2: {fc2_params:,} (weight: {ffn_dim * embed_dim:,}, bias: {embed_dim:,})")
print(f"  ffn_total: {ffn_total:,}")
print(f"  norms: {norm_total:,}")
print(f"  block_total: {block_total:,}")
print(f"  Actual: {count_params(block):,}")
print(f"  Difference: {count_params(block) - block_total:,}")

