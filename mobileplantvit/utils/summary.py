from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ParamGroup:
    name: str
    trainable_params: int
    total_params: int


@dataclass(frozen=True)
class ModelComplexity:
    """Model complexity metrics."""
    trainable_params: int
    total_params: int
    flops: int  # FLOPs (Floating Point Operations)
    macs: int   # MACs (Multiply-Accumulate Operations), typically FLOPs/2


def count_params(module: nn.Module) -> Tuple[int, int]:
    """Return (trainable_params, total_params) for a module."""
    total = sum(p.numel() for p in module.parameters())
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    return trainable, total


def format_int(value: int) -> str:
    return f"{value:,}"


def format_millions(value: int) -> str:
    return f"{value/1e6:.3f}M"


def format_gflops(flops: int) -> str:
    """Format FLOPs to GFLOPs string."""
    return f"{flops/1e9:.3f}G"


def compute_flops(model: nn.Module, img_size: int = 224, device: torch.device = None) -> Tuple[int, int]:
    """
    Compute FLOPs and MACs for a model using thop library.
    
    Args:
        model: The model to analyze
        img_size: Input image size (default: 224)
        device: Device to run on (default: same as model)
    
    Returns:
        Tuple of (flops, macs)
    """
    try:
        from thop import profile, clever_format
    except ImportError:
        print("Warning: thop not installed. Install with: pip install thop")
        print("Returning 0 for FLOPs/MACs.")
        return 0, 0
    
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Create dummy input
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # Save training state and set to eval
    was_training = model.training
    model.eval()
    
    try:
        with torch.no_grad():
            macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        flops = macs * 2  # FLOPs = 2 * MACs (multiply + add)
    except Exception as e:
        print(f"Warning: Failed to compute FLOPs: {e}")
        flops, macs = 0, 0
    finally:
        # Restore training state
        if was_training:
            model.train()
    
    return int(flops), int(macs)


def get_model_complexity(model: nn.Module, img_size: int = 224, device: torch.device = None) -> ModelComplexity:
    """
    Get complete model complexity metrics.
    
    Args:
        model: The model to analyze
        img_size: Input image size (default: 224)
        device: Device to run on
    
    Returns:
        ModelComplexity dataclass with params and FLOPs
    """
    trainable, total = count_params(model)
    flops, macs = compute_flops(model, img_size, device)
    return ModelComplexity(
        trainable_params=trainable,
        total_params=total,
        flops=flops,
        macs=macs,
    )


def print_model_complexity(complexity: ModelComplexity, model_name: str = "Model") -> None:
    """Print model complexity in a nice format."""
    print(f"\n{'='*60}")
    print(f"{model_name} Complexity:")
    print(f"{'='*60}")
    print(f"  Trainable params: {format_int(complexity.trainable_params):>15} ({format_millions(complexity.trainable_params):>10})")
    print(f"  Total params:     {format_int(complexity.total_params):>15} ({format_millions(complexity.total_params):>10})")
    print(f"  MACs:             {format_int(complexity.macs):>15} ({format_gflops(complexity.macs):>10})")
    print(f"  FLOPs:            {format_int(complexity.flops):>15} ({format_gflops(complexity.flops):>10})")
    print(f"{'='*60}")


def summarize_params_by_module(model: nn.Module) -> List[ParamGroup]:
    """A pragmatic grouping for MobilePlantViT models (baseline + Lite++)."""
    groups: List[ParamGroup] = []

    def add(name: str, mod: nn.Module) -> None:
        trainable, total = count_params(mod)
        groups.append(ParamGroup(name=name, trainable_params=trainable, total_params=total))

    # Backbone blocks (present in both)
    for name in ("stem", "stage1", "stage2", "stage3"):
        if hasattr(model, name):
            add(f"backbone.{name}", getattr(model, name))

    if hasattr(model, "patch_embedding"):
        add("patch_embedding", getattr(model, "patch_embedding"))

    # Encoders differ
    if hasattr(model, "encoder"):
        add("encoder", getattr(model, "encoder"))
    if hasattr(model, "encoder_layers"):
        add("encoder_layers", getattr(model, "encoder_layers"))

    if hasattr(model, "classifier"):
        add("classifier", getattr(model, "classifier"))

    # Total
    trainable_total, total_total = count_params(model)
    groups.append(ParamGroup(name="TOTAL", trainable_params=trainable_total, total_params=total_total))
    return groups


def print_param_summary(groups: List[ParamGroup]) -> None:
    """Pretty console output, no external deps."""
    name_w = max(len(g.name) for g in groups) if groups else 20
    print("\nParameter summary (trainable / total):")
    print("-" * (name_w + 40))
    for g in groups:
        print(
            f"{g.name:<{name_w}}  "
            f"{format_int(g.trainable_params):>12} ({format_millions(g.trainable_params):>8})  /  "
            f"{format_int(g.total_params):>12} ({format_millions(g.total_params):>8})"
        )
    print("-" * (name_w + 40))
