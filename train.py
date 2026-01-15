from __future__ import annotations

import argparse
import os
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn

from mobileplantvit.data.datasets import create_dataloaders
from mobileplantvit.data.transforms import get_transforms
from mobileplantvit.models.mobileplantvit import build_model
from mobileplantvit.train.runner import run_phase
from mobileplantvit.utils.checkpoint import raw_model
from mobileplantvit.utils.io import save_json
from mobileplantvit.utils.summary import (
    print_param_summary,
    summarize_params_by_module,
    get_model_complexity,
    print_model_complexity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MobilePlantViT (baseline + Lite++) trainer")

    # data
    parser.add_argument("--pretrain_data_root", type=str, default="", help="Root folder for pretrain dataset.")
    parser.add_argument("--main_data_root", type=str, required=True, help="Root folder for main dataset.")
    parser.add_argument("--output_root", type=str, default="runs_mobileplantvit", help="Where to save runs.")

    # phases
    parser.add_argument("--use_pretrain", action="store_true", help="Run pretrain phase then finetune.")
    parser.add_argument("--pretrain_checkpoint", type=str, default="", help="Path to a pretrain checkpoint to load (skip pretrain).")
    parser.add_argument("--resume_pretrain_checkpoint", type=str, default="", help="Resume pretrain training from checkpoint.")
    parser.add_argument("--resume_main_checkpoint", type=str, default="", help="Resume finetune training from checkpoint.")

    # epochs
    parser.add_argument("--num_epochs_pretrain", type=int, default=100)
    parser.add_argument("--num_epochs_main", type=int, default=100)

    # variant
    parser.add_argument("--model", type=str, choices=["baseline", "litepp"], default="litepp")

    # model (shared)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--patch_size", type=int, default=0)  # giữ nguyên patch_size=4 (giá trị trong paper bị ghi "00" không rõ nghĩa)
    parser.add_argument("--encoder_depth", type=int, default=1)
    parser.add_argument("--encoder_dropout", type=float, default=0.2)
    parser.add_argument("--classifier_dropout", type=float, default=0.2)
    parser.add_argument("--width_mult", type=float, default=1.0)
    parser.add_argument("--head_type", type=str, choices=["gap", "attn"], default="attn")

    # baseline-only knobs
    parser.add_argument("--ffn_dim", type=int, default=512)

    # Lite++ knobs
    parser.add_argument("--attn_rank", type=int, default=64)
    parser.add_argument("--attn_out_rank", type=int, default=0)
    parser.add_argument("--ffn_expand", type=float, default=2.0)
    parser.add_argument("--ffn_kernel", type=int, default=3)
    parser.add_argument("--use_token_pool", action="store_true")
    parser.add_argument("--pool_at", type=int, default=1)
    parser.add_argument("--pool_type", type=str, choices=["avg", "max"], default="avg")
    
    # Ablation switches for Lite++ components
    parser.add_argument("--attn_type", type=str, choices=["factorized", "linear"], default="factorized",
                        help="Attention type: 'factorized' (Lite++ default) or 'linear' (baseline-style)")
    parser.add_argument("--ffn_type", type=str, choices=["tokenconv", "mlp"], default="tokenconv",
                        help="FFN type: 'tokenconv' (Lite++ default) or 'mlp' (baseline-style)")

    # training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument("--initial_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--lr_patience", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=50)
    parser.add_argument("--label_smoothing", type=float, default=0.0)

    # augmentation
    parser.add_argument("--augment_level", type=str, choices=["none", "light", "medium", "strong", "paper"], default="paper")
    parser.add_argument("--use_mixup", action="store_true")
    parser.add_argument("--use_cutmix", action="store_true")
    parser.add_argument("--mixup_alpha", type=float, default=0.4)
    parser.add_argument("--cutmix_alpha", type=float, default=1.0)
    parser.add_argument("--mix_prob", type=float, default=0.5)

    return parser.parse_args()


def save_param_summary(out_dir: str, model: nn.Module, img_size: int = 224, model_name: str = "Model") -> None:
    os.makedirs(out_dir, exist_ok=True)
    
    # Param breakdown by module
    groups = summarize_params_by_module(model)
    print_param_summary(groups)
    save_json(os.path.join(out_dir, "param_breakdown.json"), [g.__dict__ for g in groups])

    # also write CSV for tables
    csv_path = os.path.join(out_dir, "param_breakdown.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["module", "trainable_params", "total_params"])
        for g in groups:
            w.writerow([g.name, int(g.trainable_params), int(g.total_params)])
    
    # Compute and save FLOPs
    complexity = get_model_complexity(model, img_size=img_size)
    print_model_complexity(complexity, model_name=model_name)
    
    # Save complexity to JSON
    complexity_data = {
        "model_name": model_name,
        "img_size": img_size,
        "trainable_params": complexity.trainable_params,
        "total_params": complexity.total_params,
        "macs": complexity.macs,
        "flops": complexity.flops,
        "macs_G": complexity.macs / 1e9,
        "flops_G": complexity.flops / 1e9,
        "params_M": complexity.total_params / 1e6,
    }
    save_json(os.path.join(out_dir, "model_complexity.json"), complexity_data)
    
    # Save complexity to CSV for easy comparison
    complexity_csv_path = os.path.join(out_dir, "model_complexity.csv")
    with open(complexity_csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["metric", "value", "formatted"])
        w.writerow(["trainable_params", complexity.trainable_params, f"{complexity.trainable_params/1e6:.3f}M"])
        w.writerow(["total_params", complexity.total_params, f"{complexity.total_params/1e6:.3f}M"])
        w.writerow(["macs", complexity.macs, f"{complexity.macs/1e9:.3f}G"])
        w.writerow(["flops", complexity.flops, f"{complexity.flops/1e9:.3f}G"])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def maybe_data_parallel(model: nn.Module) -> nn.Module:
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        return nn.DataParallel(model)
    return model


def load_backbone_from_checkpoint(path: str, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
    return {k: v for k, v in state.items() if not k.startswith("classifier.")}


def apply_backbone(main_model: nn.Module, backbone_state: Dict[str, torch.Tensor]) -> int:
    model_state = main_model.state_dict()
    loaded = 0
    for key, value in backbone_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded += 1
    main_model.load_state_dict(model_state)
    return loaded


def main() -> None:
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.output_root, run_name)
    os.makedirs(out_dir, exist_ok=True)

    save_json(os.path.join(out_dir, "run_config.json"), vars(args))
    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    # ---------------------------------------------------------------------
    # Pretrain: either (A) load checkpoint, or (B) run pretrain, or (C) skip
    # ---------------------------------------------------------------------
    pretrained_backbone: Optional[Dict[str, torch.Tensor]] = None

    if args.pretrain_checkpoint:
        print(f"Loading pretrain checkpoint (skip pretrain): {args.pretrain_checkpoint}")
        pretrained_backbone = load_backbone_from_checkpoint(args.pretrain_checkpoint, device)

    elif args.use_pretrain:
        if not args.pretrain_data_root:
            raise ValueError("--use_pretrain requires --pretrain_data_root")

        pt_train, pt_val, pt_test, pt_classes = create_dataloaders(
            data_root=args.pretrain_data_root,
            img_size=args.img_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            augment_level=args.augment_level,
            transforms_factory=get_transforms,
            return_paths_for_test=True,
        )
        print(f"Pretrain classes ({len(pt_classes)}): {pt_classes}")

        pre_model = build_model(args, num_classes=len(pt_classes)).to(device)
        print(f"Pretrain trainable params: {count_parameters(pre_model):,}")
        save_param_summary(
            os.path.join(out_dir, "pretrain"),
            pre_model,
            img_size=args.img_size,
            model_name=f"MobilePlantViT-{args.model} (pretrain)",
        )
        pre_model = maybe_data_parallel(pre_model)

        best_state = run_phase(
            phase_name="pretrain",
            model=pre_model,
            train_loader=pt_train,
            val_loader=pt_val,
            test_loader=pt_test,
            device=device,
            out_dir=out_dir,
            checkpoint_prefix="mobileplantvit_pretrain",
            num_epochs=args.num_epochs_pretrain,
            initial_lr=args.initial_lr,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            lr_patience=args.lr_patience,
            early_stopping_patience=args.early_stopping_patience,
            label_smoothing=args.label_smoothing,
            class_names=pt_classes,
            resume_checkpoint=args.resume_pretrain_checkpoint or None,
            use_mixup=args.use_mixup,
            use_cutmix=args.use_cutmix,
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            mix_prob=args.mix_prob,
        )
        pretrained_backbone = {k: v for k, v in best_state.items() if not k.startswith("classifier.")}

    # ---------------------------------------------------------------------
    # Main train / finetune
    # ---------------------------------------------------------------------
    train_loader, val_loader, test_loader, classes = create_dataloaders(
        data_root=args.main_data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_level=args.augment_level,
        transforms_factory=get_transforms,
        return_paths_for_test=True,
    )
    print(f"Main classes ({len(classes)}): {classes}")

    main_model = build_model(args, num_classes=len(classes)).to(device)
    print(f"Main trainable params: {count_parameters(main_model):,}")
    save_param_summary(
        os.path.join(out_dir, "main_model"),
        main_model,
        img_size=args.img_size,
        model_name=f"MobilePlantViT-{args.model}",
    )

    if pretrained_backbone is not None:
        loaded = apply_backbone(main_model, pretrained_backbone)
        print(f"Loaded pretrained backbone tensors: {loaded}")

    main_model = maybe_data_parallel(main_model)

    run_phase(
        phase_name="finetune" if (args.use_pretrain or args.pretrain_checkpoint) else "train",
        model=main_model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        out_dir=out_dir,
        checkpoint_prefix="mobileplantvit_main",
        num_epochs=args.num_epochs_main,
        initial_lr=args.initial_lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        lr_patience=args.lr_patience,
        early_stopping_patience=args.early_stopping_patience,
        label_smoothing=args.label_smoothing,
        class_names=classes,
        resume_checkpoint=args.resume_main_checkpoint or None,
        use_mixup=args.use_mixup,
        use_cutmix=args.use_cutmix,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mix_prob,
    )

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
    print(f"Saved to: {out_dir}")


if __name__ == "__main__":
    main()


"""

python train.py --main_data_root /home/nhannv02/Hello/plantvit_lite/dataset/Dataset_for_Crop_Pest_and_Disease_Detection/Data_split/Tomato/seed_42

"""