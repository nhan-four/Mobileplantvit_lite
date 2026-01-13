from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mobileplantvit.train.engine import evaluate, predict, train_one_epoch
from mobileplantvit.utils.checkpoint import load_checkpoint, raw_model, save_checkpoint
from mobileplantvit.utils.io import append_csv, save_json
from mobileplantvit.utils.metrics import (
    compute_all_metrics,
    save_confusion_matrix_csv,
    save_confusion_matrix_png,
    save_per_class_metrics_csv,
)

Tensor = torch.Tensor


def run_phase(
    phase_name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    out_dir: str,
    checkpoint_prefix: str,
    num_epochs: int,
    initial_lr: float,
    min_lr: float,
    weight_decay: float,
    lr_patience: int,
    early_stopping_patience: int,
    label_smoothing: float,
    class_names: List[str],
    resume_checkpoint: Optional[str],
    use_mixup: bool,
    use_cutmix: bool,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
) -> Dict[str, Tensor]:
    """Train a phase (pretrain or finetune) and export paper-friendly artifacts."""
    phase_dir = os.path.join(out_dir, phase_name)
    ckpt_dir = os.path.join(phase_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=float(label_smoothing))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(initial_lr), weight_decay=float(weight_decay))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=int(lr_patience),
        min_lr=float(min_lr),
    )
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    start_epoch = 1
    best_val_acc = 0.0
    best_epoch = 0
    best_state = copy.deepcopy(raw_model(model).state_dict())
    bad_epochs = 0

    history_csv = os.path.join(phase_dir, "history.csv")
    history_json = os.path.join(phase_dir, "history.json")
    history: List[Dict[str, Any]] = []

    # Resume training (optional)
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"[{phase_name}] Resume from: {resume_checkpoint}")
        ckpt = load_checkpoint(resume_checkpoint, model, optimizer, scheduler, scaler, device=device)
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_acc = float(ckpt.get("best_val_acc", 0.0))
        best_epoch = int(ckpt.get("best_epoch", 0))

        best_path = os.path.join(ckpt_dir, f"{checkpoint_prefix}_best.pth")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location=device, weights_only=False)
            best_state = best_ckpt.get("model_state_dict", best_state)

    # Save phase configuration (good for reproducibility in paper)
    save_json(
        os.path.join(phase_dir, "phase_config.json"),
        {
            "phase_name": phase_name,
            "num_epochs": int(num_epochs),
            "initial_lr": float(initial_lr),
            "min_lr": float(min_lr),
            "weight_decay": float(weight_decay),
            "lr_patience": int(lr_patience),
            "early_stopping_patience": int(early_stopping_patience),
            "label_smoothing": float(label_smoothing),
            "num_classes": len(class_names),
            "class_names": class_names,
            "augmentation": {
                "use_mixup": bool(use_mixup),
                "use_cutmix": bool(use_cutmix),
                "mixup_alpha": float(mixup_alpha),
                "cutmix_alpha": float(cutmix_alpha),
                "mix_prob": float(mix_prob),
            },
        },
    )

    fields = ["epoch", "lr", "train_loss", "train_acc", "val_loss", "val_acc", "time_sec"]

    print(f"\n========== {phase_name.upper()} ({num_epochs} epochs) ==========")
    for epoch in range(start_epoch, int(num_epochs) + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            scaler=scaler,
            use_mixup=use_mixup,
            use_cutmix=use_cutmix,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            mix_prob=mix_prob,
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_acc)
        lr = float(optimizer.param_groups[0]["lr"])
        dt = float(time.time() - t0)

        print(
            f"[{phase_name}] Epoch {epoch:03d} | LR {lr:.6f} | "
            f"train {train_loss:.4f}/{train_acc:.4f} | val {val_loss:.4f}/{val_acc:.4f} | {dt:.1f}s"
        )

        row = {
            "epoch": int(epoch),
            "lr": lr,
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
            "time_sec": dt,
        }
        history.append(row)
        append_csv(history_csv, fields, row)
        save_json(history_json, history)

        # Save "last" checkpoint every epoch
        last_path = os.path.join(ckpt_dir, f"{checkpoint_prefix}_last.pth")
        save_checkpoint(last_path, model, optimizer, scheduler, scaler, epoch, best_val_acc, best_epoch)

        # Save "best" checkpoint on improvement
        if val_acc > best_val_acc + 1e-6:
            best_val_acc = float(val_acc)
            best_epoch = int(epoch)
            best_state = copy.deepcopy(raw_model(model).state_dict())

            best_path = os.path.join(ckpt_dir, f"{checkpoint_prefix}_best.pth")
            save_checkpoint(best_path, model, optimizer, scheduler, scaler, epoch, best_val_acc, best_epoch)
            bad_epochs = 0
        else:
            bad_epochs += 1

        # Early stopping
        if int(early_stopping_patience) > 0 and bad_epochs >= int(early_stopping_patience):
            print(f"[{phase_name}] Early stopping at epoch {epoch} (best={best_val_acc:.4f} @ {best_epoch})")
            break

    # ------------------------------------------------------------
    # Final evaluation: load best weights and export paper artifacts
    # ------------------------------------------------------------
    raw_model(model).load_state_dict(best_state)

    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(
        f"[{phase_name}] BEST val_acc={best_val_acc:.4f} @ epoch {best_epoch} | "
        f"test_acc={test_acc:.4f} | test_loss={test_loss:.4f}"
    )

    # Export: predictions + confusion matrix + per-class metrics + aggregated metrics
    logits, labels, paths = predict(model, test_loader, device)
    if logits.numel() > 0:
        y_true = labels.numpy().astype(np.int64)
        y_pred = logits.argmax(dim=1).numpy().astype(np.int64)

        cm, per_cls, agg = compute_all_metrics(y_true, y_pred, class_names=class_names)

        save_json(
            os.path.join(phase_dir, "test_metrics.json"),
            {
                "test_loss": float(test_loss),
                "top1_accuracy": float(agg.accuracy),
                "macro_precision": float(agg.macro_precision),
                "macro_recall": float(agg.macro_recall),
                "macro_f1": float(agg.macro_f1),
                "weighted_precision": float(agg.weighted_precision),
                "weighted_recall": float(agg.weighted_recall),
                "weighted_f1": float(agg.weighted_f1),
                "best_val_acc": float(best_val_acc),
                "best_epoch": int(best_epoch),
            },
        )

        save_confusion_matrix_csv(os.path.join(phase_dir, "confusion_matrix.csv"), cm, class_names=class_names)
        save_confusion_matrix_png(
            os.path.join(phase_dir, "confusion_matrix.png"),
            cm,
            class_names=class_names,
            normalize=True,
            title=f"{phase_name} Confusion Matrix (normalized)",
        )
        save_per_class_metrics_csv(os.path.join(phase_dir, "per_class_metrics.csv"), per_cls, class_names=class_names)

        # Save raw predictions (useful for error analysis & qualitative figures)
        pred_rows = []
        for i in range(len(y_true)):
            item = {
                "index": int(i),
                "true_id": int(y_true[i]),
                "true_name": class_names[int(y_true[i])],
                "pred_id": int(y_pred[i]),
                "pred_name": class_names[int(y_pred[i])],
            }
            if paths:
                item["path"] = paths[i]
            pred_rows.append(item)
        save_json(os.path.join(phase_dir, "test_predictions.json"), pred_rows)

    # Keep backward-compatible final_metrics.json
    save_json(
        os.path.join(phase_dir, "final_metrics.json"),
        {
            "best_val_acc": float(best_val_acc),
            "best_epoch": int(best_epoch),
            "test_loss": float(test_loss),
            "test_acc": float(test_acc),
        },
    )

    return best_state
