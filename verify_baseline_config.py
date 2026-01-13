#!/usr/bin/env python3
"""
Script để xác minh cấu hình baseline (linear attention + MLP FFN)
cho MobilePlantViT-Lite++ để đảm bảo tính công bằng và đúng đắn.
"""

import torch
import torch.nn as nn
from mobileplantvit.models.mobileplantvit import build_model
from mobileplantvit.models.encoder import EncoderBlockLitePP, MLPFFN, TokenConvFFN
from mobileplantvit.models.attention import LinearSelfAttention, FactorizedLinearSelfAttention
from mobileplantvit.utils.summary import summarize_params_by_module
import argparse
import json


def count_params(module: nn.Module) -> int:
    """Đếm số lượng tham số trainable của một module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_module_tree(module: nn.Module, prefix: str = "", max_depth: int = 3, current_depth: int = 0):
    """In cây module để kiểm tra cấu trúc."""
    if current_depth >= max_depth:
        return
    
    name = module.__class__.__name__
    params = count_params(module)
    print(f"{prefix}{name} ({params:,} params)")
    
    if hasattr(module, 'children'):
        for child_name, child in module.named_children():
            print_module_tree(child, prefix + "  ", max_depth, current_depth + 1)


def verify_attention_type(encoder_block: EncoderBlockLitePP, expected_type: str) -> tuple[bool, str]:
    """Xác minh loại attention được sử dụng."""
    attn = encoder_block.attn
    
    if expected_type == "linear":
        is_linear = isinstance(attn, LinearSelfAttention)
        is_factorized = isinstance(attn, FactorizedLinearSelfAttention)
        
        if is_linear:
            # Kiểm tra không có low-rank matrices
            has_low_rank = hasattr(attn, 'kv_down') or hasattr(attn, 'kv_up')
            if has_low_rank:
                return False, f"FAIL: LinearSelfAttention không nên có kv_down/kv_up, nhưng tìm thấy: {dir(attn)}"
            
            # Kiểm tra có qkv projection đầy đủ
            has_qkv = hasattr(attn, 'qkv')
            if not has_qkv:
                return False, f"FAIL: LinearSelfAttention thiếu qkv projection"
            
            return True, "PASS: Sử dụng LinearSelfAttention (không có low-rank matrices)"
        
        elif is_factorized:
            return False, f"FAIL: Đang sử dụng FactorizedLinearSelfAttention thay vì LinearSelfAttention"
        else:
            return False, f"FAIL: Loại attention không xác định: {type(attn)}"
    
    elif expected_type == "factorized":
        is_factorized = isinstance(attn, FactorizedLinearSelfAttention)
        if is_factorized:
            return True, "PASS: Sử dụng FactorizedLinearSelfAttention"
        else:
            return False, f"FAIL: Đang sử dụng {type(attn)} thay vì FactorizedLinearSelfAttention"
    
    return False, f"FAIL: Loại attention không hợp lệ: {expected_type}"


def verify_ffn_type(encoder_block: EncoderBlockLitePP, expected_type: str) -> tuple[bool, str]:
    """Xác minh loại FFN được sử dụng."""
    ffn = encoder_block.ffn
    
    if expected_type == "mlp":
        is_mlp = isinstance(ffn, MLPFFN)
        is_tokenconv = isinstance(ffn, TokenConvFFN)
        
        if is_mlp:
            # Kiểm tra không có Conv layers
            has_conv = any(isinstance(m, nn.Conv2d) for m in ffn.modules())
            if has_conv:
                return False, f"FAIL: MLPFFN không nên có Conv layers, nhưng tìm thấy: {[type(m).__name__ for m in ffn.modules() if isinstance(m, nn.Conv2d)]}"
            
            # Kiểm tra có Linear layers
            linear_layers = [m for m in ffn.modules() if isinstance(m, nn.Linear)]
            if len(linear_layers) < 2:
                return False, f"FAIL: MLPFFN cần ít nhất 2 Linear layers, tìm thấy: {len(linear_layers)}"
            
            return True, f"PASS: Sử dụng MLPFFN với {len(linear_layers)} Linear layers (không có Conv)"
        
        elif is_tokenconv:
            return False, f"FAIL: Đang sử dụng TokenConvFFN thay vì MLPFFN"
        else:
            return False, f"FAIL: Loại FFN không xác định: {type(ffn)}"
    
    elif expected_type == "tokenconv":
        is_tokenconv = isinstance(ffn, TokenConvFFN)
        if is_tokenconv:
            return True, "PASS: Sử dụng TokenConvFFN"
        else:
            return False, f"FAIL: Đang sử dụng {type(ffn)} thay vì TokenConvFFN"
    
    return False, f"FAIL: Loại FFN không hợp lệ: {expected_type}"


def calculate_expected_params(embed_dim: int = 256, ffn_expand: float = 2.0, encoder_depth: int = 1) -> dict:
    """Tính toán số lượng tham số dự kiến cho linear attention + MLP FFN."""
    ffn_dim = int(embed_dim * ffn_expand)
    
    # LinearSelfAttention params
    # qkv: embed_dim * (1 + 2*embed_dim) = 256 * (1 + 512) = 131,328
    qkv_params = embed_dim * (1 + 2 * embed_dim)
    # out: embed_dim * embed_dim = 65,536
    out_params = embed_dim * embed_dim
    attn_params_per_layer = qkv_params + out_params
    
    # LayerNorm params (2 per encoder block)
    # norm1: embed_dim * 2 (weight + bias) = 512
    # norm2: embed_dim * 2 = 512
    norm_params_per_layer = embed_dim * 2 * 2
    
    # MLPFFN params
    # fc1: embed_dim * ffn_dim = 256 * 512 = 131,072
    # fc2: ffn_dim * embed_dim = 512 * 256 = 131,072
    ffn_params_per_layer = embed_dim * ffn_dim + ffn_dim * embed_dim
    
    # Total per encoder block
    block_params = attn_params_per_layer + norm_params_per_layer + ffn_params_per_layer
    
    # Total for all encoder layers
    total_encoder_params = block_params * encoder_depth
    
    return {
        "attn_params_per_layer": attn_params_per_layer,
        "norm_params_per_layer": norm_params_per_layer,
        "ffn_params_per_layer": ffn_params_per_layer,
        "block_params": block_params,
        "total_encoder_params": total_encoder_params,
        "breakdown": {
            "qkv": qkv_params,
            "out": out_params,
            "norms": norm_params_per_layer,
            "fc1": embed_dim * ffn_dim,
            "fc2": ffn_dim * embed_dim,
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Xác minh cấu hình baseline")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--encoder_depth", type=int, default=1)
    parser.add_argument("--ffn_expand", type=float, default=2.0)
    parser.add_argument("--attn_type", type=str, default="linear", choices=["linear", "factorized"])
    parser.add_argument("--ffn_type", type=str, default="mlp", choices=["mlp", "tokenconv"])
    parser.add_argument("--output", type=str, default="verification_report.json")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KIỂM TRA CẤU HÌNH BASELINE: Linear Attention + MLP FFN")
    print("=" * 80)
    print(f"\nCấu hình:")
    print(f"  - embed_dim: {args.embed_dim}")
    print(f"  - encoder_depth: {args.encoder_depth}")
    print(f"  - ffn_expand: {args.ffn_expand}")
    print(f"  - attn_type: {args.attn_type}")
    print(f"  - ffn_type: {args.ffn_type}")
    print()
    
    # Tạo model với cấu hình baseline
    class Args:
        model = "litepp"
        img_size = 224
        embed_dim = args.embed_dim
        patch_size = 4
        encoder_depth = args.encoder_depth
        encoder_dropout = 0.2
        classifier_dropout = 0.2
        width_mult = 1.0
        head_type = "gap"
        attn_rank = 64
        attn_out_rank = 0
        ffn_expand = args.ffn_expand
        ffn_kernel = 3
        use_token_pool = False
        pool_at = 1
        pool_type = "avg"
        attn_type = args.attn_type
        ffn_type = args.ffn_type
    
    fake_args = Args()
    
    # Tạo model với số classes giả định
    model = build_model(fake_args, num_classes=10)
    
    print("=" * 80)
    print("1. KIỂM TRA CẤU TRÚC MODULE")
    print("=" * 80)
    print("\nCây module (encoder_layers):")
    if hasattr(model, 'encoder_layers'):
        for i, block in enumerate(model.encoder_layers):
            print(f"\nEncoderBlock {i+1}:")
            print_module_tree(block, max_depth=4)
    
    print("\n" + "=" * 80)
    print("2. XÁC MINH MODULE SWITCHING")
    print("=" * 80)
    
    results = {
        "config": vars(args),
        "verification": {},
        "parameter_check": {},
        "overall_status": "UNKNOWN"
    }
    
    all_pass = True
    
    if hasattr(model, 'encoder_layers') and len(model.encoder_layers) > 0:
        encoder_block = model.encoder_layers[0]
        
        # Kiểm tra attention type
        attn_ok, attn_msg = verify_attention_type(encoder_block, args.attn_type)
        print(f"\n[Attention Type] {attn_msg}")
        results["verification"]["attention"] = {
            "expected": args.attn_type,
            "status": "PASS" if attn_ok else "FAIL",
            "message": attn_msg
        }
        if not attn_ok:
            all_pass = False
        
        # Kiểm tra FFN type
        ffn_ok, ffn_msg = verify_ffn_type(encoder_block, args.ffn_type)
        print(f"[FFN Type] {ffn_msg}")
        results["verification"]["ffn"] = {
            "expected": args.ffn_type,
            "status": "PASS" if ffn_ok else "FAIL",
            "message": ffn_msg
        }
        if not ffn_ok:
            all_pass = False
    
    print("\n" + "=" * 80)
    print("3. KIỂM TRA SỐ LƯỢNG THAM SỐ")
    print("=" * 80)
    
    # Tính toán dự kiến
    expected = calculate_expected_params(
        embed_dim=args.embed_dim,
        ffn_expand=args.ffn_expand,
        encoder_depth=args.encoder_depth
    )
    
    print(f"\nSố lượng tham số dự kiến (per encoder block):")
    print(f"  - Attention: {expected['attn_params_per_layer']:,}")
    print(f"    + qkv: {expected['breakdown']['qkv']:,}")
    print(f"    + out: {expected['breakdown']['out']:,}")
    print(f"  - LayerNorm (2x): {expected['norm_params_per_layer']:,}")
    print(f"  - MLP FFN: {expected['ffn_params_per_layer']:,}")
    print(f"    + fc1: {expected['breakdown']['fc1']:,}")
    print(f"    + fc2: {expected['breakdown']['fc2']:,}")
    print(f"  - Tổng per block: {expected['block_params']:,}")
    print(f"  - Tổng encoder_layers ({args.encoder_depth} blocks): {expected['total_encoder_params']:,}")
    
    # Đếm thực tế
    if hasattr(model, 'encoder_layers'):
        actual_encoder_params = count_params(model.encoder_layers)
        print(f"\nSố lượng tham số thực tế (encoder_layers): {actual_encoder_params:,}")
        
        diff = actual_encoder_params - expected['total_encoder_params']
        diff_pct = (diff / expected['total_encoder_params']) * 100 if expected['total_encoder_params'] > 0 else 0
        
        print(f"  - Chênh lệch: {diff:+,} ({diff_pct:+.2f}%)")
        
        # So sánh với Lite++ reference (386k cho factorized + tokenconv)
        litepp_ref = 386113
        increase = actual_encoder_params - litepp_ref
        increase_pct = (increase / litepp_ref) * 100 if litepp_ref > 0 else 0
        
        print(f"\nSo sánh với Lite++ reference (386k):")
        print(f"  - Tăng: {increase:+,} ({increase_pct:+.2f}%)")
        
        # Kiểm tra điều kiện: tăng 70-100k so với Lite++
        expected_increase_min = 70000
        expected_increase_max = 100000
        
        if expected_increase_min <= increase <= expected_increase_max:
            param_check_msg = f"PASS: Tăng {increase:,} params nằm trong khoảng dự kiến ({expected_increase_min:,} - {expected_increase_max:,})"
            param_check_ok = True
        elif increase < expected_increase_min:
            param_check_msg = f"FAIL: Tăng {increase:,} params quá thấp (dự kiến >= {expected_increase_min:,})"
            param_check_ok = False
        else:
            param_check_msg = f"WARNING: Tăng {increase:,} params cao hơn dự kiến (dự kiến <= {expected_increase_max:,})"
            param_check_ok = True  # Vẫn OK nhưng cần kiểm tra
        
        print(f"  - {param_check_msg}")
        
        results["parameter_check"] = {
            "expected": expected,
            "actual": actual_encoder_params,
            "difference": diff,
            "difference_percent": diff_pct,
            "litepp_reference": litepp_ref,
            "increase_vs_litepp": increase,
            "increase_percent": increase_pct,
            "status": "PASS" if param_check_ok else "FAIL",
            "message": param_check_msg
        }
        
        if not param_check_ok:
            all_pass = False
    
    # Tổng số params
    total_params = count_params(model)
    print(f"\nTổng số tham số model: {total_params:,} ({total_params/1e6:.3f}M)")
    
    expected_total_min = 580000
    if total_params >= expected_total_min:
        print(f"PASS: Tổng params >= {expected_total_min:,} (yêu cầu)")
    else:
        print(f"FAIL: Tổng params < {expected_total_min:,} (yêu cầu)")
        all_pass = False
    
    results["parameter_check"]["total_params"] = total_params
    results["parameter_check"]["total_params_check"] = total_params >= expected_total_min
    
    print("\n" + "=" * 80)
    print("4. KẾT LUẬN")
    print("=" * 80)
    
    if all_pass:
        print("\n✅ PASS: Tất cả các kiểm tra đều PASS")
        results["overall_status"] = "PASS"
    else:
        print("\n❌ FAIL: Một hoặc nhiều kiểm tra FAIL")
        results["overall_status"] = "FAIL"
    
    # Lưu báo cáo
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nBáo cáo đã được lưu vào: {args.output}")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    exit(main())

