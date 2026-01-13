from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    weighted_precision: float
    weighted_recall: float
    weighted_f1: float


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def per_class_prf(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Compute per-class precision/recall/f1/support from confusion matrix."""
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=(predicted > 0))
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=(support > 0))
    f1 = np.divide(2 * precision * recall, (precision + recall), out=np.zeros_like(tp), where=((precision + recall) > 0))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def aggregate_metrics(per_cls: Dict[str, np.ndarray]) -> ClassificationMetrics:
    precision = per_cls["precision"]
    recall = per_cls["recall"]
    f1 = per_cls["f1"]
    support = per_cls["support"]

    macro_precision = float(np.mean(precision))
    macro_recall = float(np.mean(recall))
    macro_f1 = float(np.mean(f1))

    total = float(np.sum(support)) if float(np.sum(support)) > 0 else 1.0
    weights = support / total

    weighted_precision = float(np.sum(weights * precision))
    weighted_recall = float(np.sum(weights * recall))
    weighted_f1 = float(np.sum(weights * f1))

    return ClassificationMetrics(
        accuracy=0.0,  # filled by caller
        macro_precision=macro_precision,
        macro_recall=macro_recall,
        macro_f1=macro_f1,
        weighted_precision=weighted_precision,
        weighted_recall=weighted_recall,
        weighted_f1=weighted_f1,
    )


def save_confusion_matrix_csv(path: str, cm: np.ndarray, class_names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + list(class_names))
        for i, name in enumerate(class_names):
            writer.writerow([name] + [int(x) for x in cm[i].tolist()])


def save_per_class_metrics_csv(path: str, per_cls: Dict[str, np.ndarray], class_names: Sequence[str]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "support", "precision", "recall", "f1"])
        for i, name in enumerate(class_names):
            writer.writerow(
                [
                    name,
                    int(per_cls["support"][i]),
                    float(per_cls["precision"][i]),
                    float(per_cls["recall"][i]),
                    float(per_cls["f1"][i]),
                ]
            )


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], ClassificationMetrics]:
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_cls = per_class_prf(cm)

    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    agg = aggregate_metrics(per_cls)
    agg = ClassificationMetrics(
        accuracy=acc,
        macro_precision=agg.macro_precision,
        macro_recall=agg.macro_recall,
        macro_f1=agg.macro_f1,
        weighted_precision=agg.weighted_precision,
        weighted_recall=agg.weighted_recall,
        weighted_f1=agg.weighted_f1,
    )
    return cm, per_cls, agg


def save_confusion_matrix_png(
    path: str,
    cm: np.ndarray,
    class_names: Sequence[str],
    normalize: bool = True,
    title: str = "Confusion Matrix",
) -> None:
    """Save a clean confusion matrix figure (matplotlib only)."""
    import matplotlib.pyplot as plt  # local import to keep core deps light

    os.makedirs(os.path.dirname(path), exist_ok=True)

    mat = cm.astype(np.float64)
    if normalize:
        row_sum = mat.sum(axis=1, keepdims=True)
        mat = np.divide(mat, row_sum, out=np.zeros_like(mat), where=(row_sum > 0))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    im = ax.imshow(mat, aspect="auto")
    fig.colorbar(im, ax=ax)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
