# Evaluation Metrics Guide

Hướng dẫn về các chỉ số đánh giá (metrics) được lưu sau khi training.

## Tổng quan

Sau mỗi lần training, hệ thống sẽ tự động tính toán và lưu các metrics chi tiết cho việc viết paper. Tất cả kết quả được lưu trong thư mục `runs_mobileplantvit/<timestamp>/train/`.

## File Outputs

### 1. `test_metrics.json`
File JSON chứa tất cả các metrics tổng hợp:

```json
{
    "test_loss": 0.5234,
    "test_acc": 0.9512,
    "top1_accuracy": 0.9512,
    "balanced_accuracy": 0.9489,
    "macro_precision": 0.9523,
    "macro_recall": 0.9489,
    "macro_f1": 0.9505,
    "weighted_precision": 0.9515,
    "weighted_recall": 0.9512,
    "weighted_f1": 0.9513,
    "macro_auc": 0.9934,
    "weighted_auc": 0.9936,
    "top3_accuracy": 0.9945,
    "top5_accuracy": 0.9989,
    "cohens_kappa": 0.9487,
    "best_val_acc": 0.9534,
    "best_epoch": 45
}
```

### 2. `classification_report.csv`
Classification report đầy đủ theo format chuẩn, phù hợp để copy vào paper:

```csv
Classification Report

Class,Precision,Recall,F1-Score,Support
Apple___Apple_scab,0.9876,0.9823,0.9849,234
Apple___Black_rot,0.9654,0.9701,0.9677,187
...

Metric,Value
Accuracy,0.9512
Balanced Accuracy,0.9489

Macro Precision,0.9523
Macro Recall,0.9489
Macro F1-Score,0.9505

Weighted Precision,0.9515
Weighted Recall,0.9512
Weighted F1-Score,0.9513

Macro AUC-ROC,0.9934
Weighted AUC-ROC,0.9936

Top-3 Accuracy,0.9945
Top-5 Accuracy,0.9989

Cohen's Kappa,0.9487
```

### 3. `confusion_matrix.csv`
Confusion matrix dạng CSV để import vào Excel/LaTeX:

```csv
true\pred,Class_0,Class_1,Class_2,...
Class_0,234,3,1,...
Class_1,2,187,5,...
...
```

### 4. `confusion_matrix.png`
Confusion matrix đã được normalize và vẽ dạng heatmap (DPI=200).

### 5. `per_class_metrics.csv`
Metrics chi tiết cho từng class:

```csv
class,support,precision,recall,f1
Apple___Apple_scab,234,0.9876,0.9823,0.9849
Apple___Black_rot,187,0.9654,0.9701,0.9677
...
```

### 6. `test_predictions.json`
Raw predictions cho tất cả các samples (để phân tích lỗi):

```json
[
    {
        "index": 0,
        "true_id": 5,
        "true_name": "Apple___healthy",
        "pred_id": 5,
        "pred_name": "Apple___healthy",
        "path": "/path/to/image.jpg"
    },
    ...
]
```

## Giải thích các Metrics

### Standard Metrics

1. **Accuracy (Top-1)**: Tỷ lệ dự đoán đúng
   - Công thức: `(TP + TN) / (TP + TN + FP + FN)`

2. **Balanced Accuracy**: Trung bình recall của tất cả các class
   - Hữu ích cho dataset không cân bằng
   - Công thức: `mean(recall_per_class)`

3. **Precision (Macro/Weighted)**: Độ chính xác của dự đoán positive
   - Macro: Trung bình không trọng số
   - Weighted: Trung bình có trọng số theo số lượng mẫu

4. **Recall (Macro/Weighted)**: Tỷ lệ tìm được positive thực sự
   - Macro: Trung bình không trọng số
   - Weighted: Trung bình có trọng số theo số lượng mẫu

5. **F1-Score (Macro/Weighted)**: Trung bình điều hòa của Precision và Recall
   - Công thức: `2 * (precision * recall) / (precision + recall)`

### Advanced Metrics

6. **AUC-ROC (Macro/Weighted)**: Area Under ROC Curve
   - Đo khả năng phân biệt giữa các class
   - Giá trị từ 0.5 (random) đến 1.0 (perfect)
   - Macro: One-vs-Rest, trung bình không trọng số
   - Weighted: One-vs-Rest, trung bình có trọng số

7. **Top-3 Accuracy**: Tỷ lệ label đúng nằm trong 3 dự đoán hàng đầu
   - Hữu ích cho các ứng dụng gợi ý

8. **Top-5 Accuracy**: Tỷ lệ label đúng nằm trong 5 dự đoán hàng đầu

9. **Cohen's Kappa**: Đo sự đồng thuận có tính đến xác suất ngẫu nhiên
   - Giá trị từ -1 đến 1
   - Kappa > 0.8: excellent agreement
   - Kappa > 0.6: substantial agreement
   - Công thức: `(observed_agreement - expected_agreement) / (1 - expected_agreement)`

## Sử dụng trong Paper

### Table 1: Overall Performance

| Model | Accuracy | Balanced Acc | Macro F1 | Weighted F1 | Macro AUC |
|-------|----------|--------------|----------|-------------|-----------|
| Lite++ | 95.12% | 94.89% | 95.05% | 95.13% | 99.34% |
| Baseline | ... | ... | ... | ... | ... |

### Table 2: Top-k Accuracy

| Model | Top-1 | Top-3 | Top-5 | Kappa |
|-------|-------|-------|-------|-------|
| Lite++ | 95.12% | 99.45% | 99.89% | 0.949 |
| Baseline | ... | ... | ... | ... |

### Figure: Confusion Matrix

Sử dụng file `confusion_matrix.png` (đã normalize, DPI=200)

### Table 3: Per-Class Performance

Sử dụng dữ liệu từ `per_class_metrics.csv` để tạo bảng chi tiết cho từng loại bệnh.

## Dependencies

Metrics yêu cầu các thư viện sau:
- `numpy`: Tính toán cơ bản
- `scikit-learn`: Tính AUC-ROC scores (optional, nếu không có AUC sẽ = 0)
- `matplotlib`: Vẽ confusion matrix

Tất cả đã được thêm vào `requirements.txt`.

## Code Example

```python
# Load metrics
import json
with open("runs_mobileplantvit/20260113_151004/train/test_metrics.json") as f:
    metrics = json.load(f)

print(f"Test Accuracy: {metrics['test_acc']:.4f}")
print(f"Macro AUC: {metrics['macro_auc']:.4f}")
print(f"Cohen's Kappa: {metrics['cohens_kappa']:.4f}")
```

## References

1. Confusion Matrix: [Wikipedia](https://en.wikipedia.org/wiki/Confusion_matrix)
2. ROC AUC: [Scikit-learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics)
3. Cohen's Kappa: [Wikipedia](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
4. Balanced Accuracy: [Scikit-learn Docs](https://scikit-learn.org/stable/modules/model_evaluation.html#balanced-accuracy-score)

