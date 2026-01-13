#!/usr/bin/env python3
"""In chi tiết cấu trúc module và breakdown tham số."""

import torch.nn as nn
from mobileplantvit.models.mobileplantvit import build_model
from mobileplantvit.utils.summary import summarize_params_by_module

def count_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

class Args:
    model = "litepp"
    img_size = 224
    embed_dim = 256
    patch_size = 4
    encoder_depth = 1
    encoder_dropout = 0.2
    classifier_dropout = 0.2
    width_mult = 1.0
    head_type = "gap"
    attn_rank = 64
    attn_out_rank = 0
    ffn_expand = 2.0
    ffn_kernel = 3
    use_token_pool = False
    pool_at = 1
    pool_type = "avg"
    attn_type = "linear"
    ffn_type = "mlp"

args = Args()
model = build_model(args, num_classes=10)

print("=" * 80)
print("CẤU TRÚC MODULE VÀ BREAKDOWN THAM SỐ")
print("=" * 80)

# In breakdown từ summary
groups = summarize_params_by_module(model)
print("\nBreakdown theo module:")
for g in groups:
    print(f"  {g.name:30s} {g.trainable_params:>10,} params")

print("\n" + "=" * 80)
print("CHI TIẾT ENCODER BLOCK")
print("=" * 80)

if hasattr(model, 'encoder_layers'):
    for i, block in enumerate(model.encoder_layers):
        print(f"\nEncoderBlock {i+1}:")
        print(f"  Total: {count_params(block):,} params")
        print(f"\n  Components:")
        print(f"    norm1: {count_params(block.norm1):,} params")
        print(f"    norm2: {count_params(block.norm2):,} params")
        print(f"    attn ({type(block.attn).__name__}): {count_params(block.attn):,} params")
        print(f"    ffn ({type(block.ffn).__name__}): {count_params(block.ffn):,} params")
        
        # Chi tiết attention
        print(f"\n    Attention details:")
        if hasattr(block.attn, 'qkv'):
            print(f"      qkv: {count_params(block.attn.qkv):,} params")
        if hasattr(block.attn, 'out'):
            print(f"      out: {count_params(block.attn.out):,} params")
        if hasattr(block.attn, 'q'):
            print(f"      q: {count_params(block.attn.q):,} params")
        if hasattr(block.attn, 'kv_down'):
            print(f"      kv_down: {count_params(block.attn.kv_down):,} params")
        if hasattr(block.attn, 'kv_up'):
            print(f"      kv_up: {count_params(block.attn.kv_up):,} params")
        
        # Chi tiết FFN
        print(f"\n    FFN details:")
        if hasattr(block.ffn, 'fc1'):
            print(f"      fc1: {count_params(block.ffn.fc1):,} params")
        if hasattr(block.ffn, 'fc2'):
            print(f"      fc2: {count_params(block.ffn.fc2):,} params")
        if hasattr(block.ffn, 'pw1'):
            print(f"      pw1: {count_params(block.ffn.pw1):,} params")
        if hasattr(block.ffn, 'dw'):
            print(f"      dw: {count_params(block.ffn.dw):,} params")
        if hasattr(block.ffn, 'pw2'):
            print(f"      pw2: {count_params(block.ffn.pw2):,} params")

print("\n" + "=" * 80)
print("XÁC NHẬN")
print("=" * 80)

if hasattr(model, 'encoder_layers') and len(model.encoder_layers) > 0:
    block = model.encoder_layers[0]
    
    # Kiểm tra attention
    from mobileplantvit.models.attention import LinearSelfAttention, FactorizedLinearSelfAttention
    is_linear = isinstance(block.attn, LinearSelfAttention)
    is_factorized = isinstance(block.attn, FactorizedLinearSelfAttention)
    
    print(f"\nAttention Type:")
    print(f"  Expected: LinearSelfAttention")
    print(f"  Actual: {type(block.attn).__name__}")
    print(f"  Status: {'✅ PASS' if is_linear else '❌ FAIL'}")
    
    if is_linear:
        has_conv = any(isinstance(m, nn.Conv2d) for m in block.attn.modules())
        has_low_rank = hasattr(block.attn, 'kv_down') or hasattr(block.attn, 'kv_up')
        print(f"  - No Conv layers: {'✅' if not has_conv else '❌'}")
        print(f"  - No low-rank matrices: {'✅' if not has_low_rank else '❌'}")
    
    # Kiểm tra FFN
    from mobileplantvit.models.encoder import MLPFFN, TokenConvFFN
    is_mlp = isinstance(block.ffn, MLPFFN)
    is_tokenconv = isinstance(block.ffn, TokenConvFFN)
    
    print(f"\nFFN Type:")
    print(f"  Expected: MLPFFN")
    print(f"  Actual: {type(block.ffn).__name__}")
    print(f"  Status: {'✅ PASS' if is_mlp else '❌ FAIL'}")
    
    if is_mlp:
        has_conv = any(isinstance(m, nn.Conv2d) for m in block.ffn.modules())
        linear_count = len([m for m in block.ffn.modules() if isinstance(m, nn.Linear)])
        print(f"  - No Conv layers: {'✅' if not has_conv else '❌'}")
        print(f"  - Has Linear layers: {'✅' if linear_count >= 2 else '❌'} ({linear_count} layers)")

print()

