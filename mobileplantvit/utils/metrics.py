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
    # Additional metrics for paper
    balanced_accuracy: float = 0.0
    macro_auc: float = 0.0
    weighted_auc: float = 0.0
    top3_accuracy: float = 0.0
    top5_accuracy: float = 0.0
    cohens_kappa: float = 0.0


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


def compute_auc_scores(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    num_classes: int,
) -> Tuple[float, float]:
    """Compute macro and weighted AUC-ROC scores.
    
    Args:
        y_true: True labels, shape (N,)
        y_probs: Predicted probabilities, shape (N, num_classes)
        num_classes: Number of classes
    
    Returns:
        (macro_auc, weighted_auc)
    """
    try:
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import label_binarize
    except ImportError:
        print("Warning: sklearn not available, AUC scores will be 0")
        return 0.0, 0.0
    
    # Binarize labels for one-vs-rest
    y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
    
    # Macro AUC
    try:
        macro_auc = float(roc_auc_score(y_true_bin, y_probs, average='macro', multi_class='ovr'))
    except Exception:
        macro_auc = 0.0
    
    # Weighted AUC
    try:
        weighted_auc = float(roc_auc_score(y_true_bin, y_probs, average='weighted', multi_class='ovr'))
    except Exception:
        weighted_auc = 0.0
    
    return macro_auc, weighted_auc


def compute_balanced_accuracy(cm: np.ndarray) -> float:
    """Compute balanced accuracy from confusion matrix.
    
    Balanced accuracy = average of recall for each class
    """
    per_cls = per_class_prf(cm)
    recall = per_cls["recall"]
    return float(np.mean(recall))


def compute_topk_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int) -> float:
    """Compute top-k accuracy.
    
    Args:
        y_true: True labels, shape (N,)
        y_probs: Predicted probabilities, shape (N, num_classes)
        k: k for top-k
    
    Returns:
        Top-k accuracy
    """
    if k >= y_probs.shape[1]:
        return 1.0
    
    # Get top-k predictions
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]
    
    # Check if true label is in top-k
    correct = np.any(topk_preds == y_true[:, None], axis=1)
    return float(np.mean(correct))


def compute_cohens_kappa(cm: np.ndarray) -> float:
    """Compute Cohen's Kappa coefficient from confusion matrix."""
    n = np.sum(cm)
    if n == 0:
        return 0.0
    
    # Observed agreement
    po = np.trace(cm) / n
    
    # Expected agreement
    row_sums = np.sum(cm, axis=1)
    col_sums = np.sum(cm, axis=0)
    pe = np.sum(row_sums * col_sums) / (n * n)
    
    if pe == 1.0:
        return 0.0
    
    kappa = (po - pe) / (1 - pe)
    return float(kappa)


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


def save_per_class_metrics_csv(
    path: str,
    per_cls: Dict[str, np.ndarray],
    class_names: Sequence[str],
    cm: Optional[np.ndarray] = None,
) -> None:
    """Save per-class metrics to CSV with additional specificity metric.
    
    Args:
        path: Output CSV path
        per_cls: Per-class metrics dict
        class_names: List of class names
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Compute specificity if confusion matrix data available
    specificity = np.zeros(len(class_names))
    if "support" in per_cls:
        # Can't compute specificity from per_cls alone, need confusion matrix
        pass
    
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




def save_classification_report(
    path: str,
    per_cls: Dict[str, np.ndarray],
    agg: ClassificationMetrics,
    class_names: Sequence[str],
) -> None:
    """Save a comprehensive classification report for paper."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Classification Report"])
        writer.writerow([])
        
        # Per-class metrics
        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
        for i, name in enumerate(class_names):
            writer.writerow([
                name,
                f"{per_cls['precision'][i]:.4f}",
                f"{per_cls['recall'][i]:.4f}",
                f"{per_cls['f1'][i]:.4f}",
                int(per_cls['support'][i]),
            ])
        
        writer.writerow([])
        
        # Aggregated metrics
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Accuracy", f"{agg.accuracy:.4f}"])
        writer.writerow(["Balanced Accuracy", f"{agg.balanced_accuracy:.4f}"])
        writer.writerow([])
        writer.writerow(["Macro Precision", f"{agg.macro_precision:.4f}"])
        writer.writerow(["Macro Recall", f"{agg.macro_recall:.4f}"])
        writer.writerow(["Macro F1-Score", f"{agg.macro_f1:.4f}"])
        writer.writerow([])
        writer.writerow(["Weighted Precision", f"{agg.weighted_precision:.4f}"])
        writer.writerow(["Weighted Recall", f"{agg.weighted_recall:.4f}"])
        writer.writerow(["Weighted F1-Score", f"{agg.weighted_f1:.4f}"])
        writer.writerow([])
        writer.writerow(["Macro AUC-ROC", f"{agg.macro_auc:.4f}"])
        writer.writerow(["Weighted AUC-ROC", f"{agg.weighted_auc:.4f}"])
        writer.writerow([])
        writer.writerow(["Top-3 Accuracy", f"{agg.top3_accuracy:.4f}"])
        writer.writerow(["Top-5 Accuracy", f"{agg.top5_accuracy:.4f}"])
        writer.writerow([])
        writer.writerow(["Cohen's Kappa", f"{agg.cohens_kappa:.4f}"])


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Sequence[str],
    y_probs: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], ClassificationMetrics]:
    """Compute all classification metrics including AUC, top-k accuracy, etc.
    
    Args:
        y_true: True labels, shape (N,)
        y_pred: Predicted labels, shape (N,)
        class_names: List of class names
        y_probs: Predicted probabilities, shape (N, num_classes) - optional for AUC
    
    Returns:
        (confusion_matrix, per_class_metrics, aggregated_metrics)
    """
    num_classes = len(class_names)
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    per_cls = per_class_prf(cm)

    acc = float(np.mean(y_true == y_pred)) if y_true.size else 0.0
    balanced_acc = compute_balanced_accuracy(cm)
    cohens_kappa = compute_cohens_kappa(cm)
    
    # AUC scores (if probabilities provided)
    macro_auc = 0.0
    weighted_auc = 0.0
    if y_probs is not None and y_probs.ndim == 2:
        macro_auc, weighted_auc = compute_auc_scores(y_true, y_probs, num_classes)
    
    # Top-k accuracy (if probabilities provided)
    top3_acc = 0.0
    top5_acc = 0.0
    if y_probs is not None and y_probs.ndim == 2:
        top3_acc = compute_topk_accuracy(y_true, y_probs, k=3)
        if num_classes >= 5:
            top5_acc = compute_topk_accuracy(y_true, y_probs, k=5)
    
    agg = aggregate_metrics(per_cls)
    agg = ClassificationMetrics(
        accuracy=acc,
        macro_precision=agg.macro_precision,
        macro_recall=agg.macro_recall,
        macro_f1=agg.macro_f1,
        weighted_precision=agg.weighted_precision,
        weighted_recall=agg.weighted_recall,
        weighted_f1=agg.weighted_f1,
        balanced_accuracy=balanced_acc,
        macro_auc=macro_auc,
        weighted_auc=weighted_auc,
        top3_accuracy=top3_acc,
        top5_accuracy=top5_acc,
        cohens_kappa=cohens_kappa,
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
