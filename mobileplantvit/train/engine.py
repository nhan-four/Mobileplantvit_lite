from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mobileplantvit.train.mix import cutmix_data, mix_criterion, mixup_data

Tensor = torch.Tensor


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    """Return (avg_loss, accuracy)."""
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    use_amp = device.type == "cuda"

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            images, labels, _paths = batch
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
        else:
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += int(batch_size)

    denom = max(total_samples, 1)
    return total_loss / denom, total_correct / denom


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler],
    use_mixup: bool,
    use_cutmix: bool,
    mixup_alpha: float,
    cutmix_alpha: float,
    mix_prob: float,
) -> Tuple[float, float]:
    """One epoch training. Returns (avg_loss, approx_accuracy)."""
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            images, labels, _paths = batch
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        do_mix = (use_mixup or use_cutmix) and (np.random.rand() < float(mix_prob))
        if do_mix:
            do_cutmix = use_cutmix and (not use_mixup or (np.random.rand() < 0.5))
            if do_cutmix:
                mixed, y_a, y_b, lam = cutmix_data(images, labels, cutmix_alpha, device)
            else:
                mixed, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha, device)

            if scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    logits = model(mixed)
                    loss = mix_criterion(criterion, logits, y_a, y_b, lam)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(mixed)
                loss = mix_criterion(criterion, logits, y_a, y_b, lam)
                loss.backward()
                optimizer.step()

            preds = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += float(loss.item()) * batch_size
            total_correct += int((preds == y_a).sum().item())  # monitoring-only
            total_samples += int(batch_size)
            continue

        # normal training
        if scaler is not None:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((preds == labels).sum().item())
        total_samples += int(batch_size)

    denom = max(total_samples, 1)
    return total_loss / denom, total_correct / denom


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, list]:
    """Collect logits, labels, and (optional) paths for the whole loader."""
    model.eval()

    all_logits = []
    all_labels = []
    all_paths = []

    use_amp = device.type == "cuda"

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 3:
            images, labels, paths = batch
            all_paths.extend(list(paths))
        else:
            images, labels = batch

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if use_amp:
            with torch.amp.autocast(device_type="cuda"):
                logits = model(images)
        else:
            logits = model(images)

        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    if not all_logits:
        return torch.empty(0), torch.empty(0, dtype=torch.long), all_paths

    return torch.cat(all_logits, dim=0), torch.cat(all_labels, dim=0), all_paths
