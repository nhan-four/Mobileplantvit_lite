# BÁO CÁO KIỂM TRA CẤU HÌNH TRAIN

## Tổng quan
Báo cáo này so sánh cấu hình train hiện tại với cấu hình yêu cầu từ EXPERIMENTAL SETUP AND DATA AUGMENTATION SUMMARY.

## Kết quả kiểm tra chi tiết

| Thuộc tính | Giá trị yêu cầu | Giá trị hiện tại | Trạng thái | Vị trí trong code |
|------------|----------------|------------------|------------|-------------------|
| **InputShape** | 3×224×224 | 3×224×224 | ✅ ĐÚNG | `train.py:47` (img_size=224) |
| **BatchSize** | 64 | 32 | ❌ SAI | `train.py:75` (batch_size=32) |
| **EmbedDimension** | 256 | 256 | ✅ ĐÚNG | `train.py:48` (embed_dim=256) |
| **FFNDimension** | 512 | 512 | ✅ ĐÚNG | `train.py:57` (ffn_dim=512) |
| **PatchSize** | 00 | 4 | ❓ CẦN XÁC NHẬN | `train.py:49` (patch_size=4) |
| **InitialLearningRate** | 1e-3 | 3e-4 | ❌ SAI | `train.py:77` (initial_lr=3e-4) |
| **LearningRateReductionRate** | 50% | 50% | ✅ ĐÚNG | `runner.py:61` (factor=0.5) |
| **MinimumLearningRate** | 1e-7 | 1e-6 | ❌ SAI | `train.py:78` (min_lr=1e-6) |
| **LearningRateReductionPatience** | 10 | 5 | ❌ SAI | `train.py:80` (lr_patience=5) |
| **ValidationAccuracyThreshold** | 1e-5 | 1e-6 | ❌ SAI | `runner.py:162` (1e-6) |
| **EarlyStoppingPatience** | 50 | 20 | ❌ SAI | `train.py:81` (early_stopping_patience=20) |
| **WeightDecay** | 1e-3 | 0 | ❌ SAI | `train.py:79` (weight_decay=0) |
| **EncoderDropoutRate** | 30% | 20% | ❌ SAI | `train.py:51` (encoder_dropout=0.2) |
| **ClassifierDropoutRate** | 20% | 20% | ✅ ĐÚNG | `train.py:52` (classifier_dropout=0.2) |
| **Optimizer** | Adam | AdamW | ❌ SAI | `runner.py:57` (AdamW) |
| **Loss** | CategoricalCrossentropy | CrossEntropyLoss | ✅ ĐÚNG | `runner.py:56` (CrossEntropyLoss) |
| **HorizontalFlip** | 50% | 50% | ✅ ĐÚNG | `transforms.py:30` (p=0.5) |
| **RandomRotate90°** | 50% | 50% | ✅ ĐÚNG | `transforms.py:33` (p=0.5) |
| **Shift,Scale,Rotate** | 50% (0.05,0.05,30°) | 50% (0.05,0.05,30°) | ✅ ĐÚNG | `transforms.py:36-42` |
| **RandomGamma** | 20% | 20% | ✅ ĐÚNG | `transforms.py:45` (p=0.2) |
| **RandomBrightness/Contrast** | 30% | 30% | ✅ ĐÚNG | `transforms.py:48` (p=0.3) |
| **RGBShift** | 30% (limits:15) | 30% (limits:15) | ✅ ĐÚNG | `transforms.py:51` (p=0.3, shift_limit=15) |
| **CLAHE** | 30% (cliplimit:4.0) | 30% (cliplimit:4.0) | ✅ ĐÚNG | `transforms.py:54` (p=0.3, clip_limit=4.0) |

## Tóm tắt

### ✅ Các thông số ĐÚNG (11/21):
- InputShape, EmbedDimension, FFNDimension, ClassifierDropoutRate
- LearningRateReductionRate, Loss
- Tất cả các data augmentation transforms (7/7)

### ❌ Các thông số SAI cần sửa (9/21):
1. **BatchSize**: 32 → 64
2. **InitialLearningRate**: 3e-4 → 1e-3
3. **MinimumLearningRate**: 1e-6 → 1e-7
4. **LearningRateReductionPatience**: 5 → 10
5. **ValidationAccuracyThreshold**: 1e-6 → 1e-5
6. **EarlyStoppingPatience**: 20 → 50
7. **WeightDecay**: 0 → 1e-3
8. **EncoderDropoutRate**: 0.2 (20%) → 0.3 (30%)
9. **Optimizer**: AdamW → Adam

### ❓ Cần xác nhận (1/21):
- **PatchSize**: Yêu cầu là "00" - có thể là:
  - Patch size = 0 (không patchify)
  - Hoặc là lỗi đánh máy và cần giữ nguyên patch_size=4
  - Cần xác nhận với người yêu cầu

## Các file cần chỉnh sửa

### 1. `train.py` - Các tham số mặc định:
```python
# Dòng 75: BatchSize
--batch_size: 32 → 64

# Dòng 77: InitialLearningRate  
--initial_lr: 3e-4 → 1e-3

# Dòng 78: MinimumLearningRate
--min_lr: 1e-6 → 1e-7

# Dòng 80: LearningRateReductionPatience
--lr_patience: 5 → 10

# Dòng 81: EarlyStoppingPatience
--early_stopping_patience: 20 → 50

# Dòng 79: WeightDecay
--weight_decay: 0 → 1e-3

# Dòng 51: EncoderDropoutRate
--encoder_dropout: 0.2 → 0.3

# Dòng 49: PatchSize (cần xác nhận)
--patch_size: 4 → 0 (nếu yêu cầu là 0)
```

### 2. `mobileplantvit/train/runner.py` - Optimizer và threshold:
```python
# Dòng 57: Optimizer
torch.optim.AdamW → torch.optim.Adam

# Dòng 162: ValidationAccuracyThreshold
val_acc > best_val_acc + 1e-6 → val_acc > best_val_acc + 1e-5
```

## Khuyến nghị

1. **Ưu tiên cao**: Sửa các thông số training quan trọng:
   - BatchSize, LearningRate, WeightDecay, Dropout rates
   - Early stopping và LR scheduler patience

2. **Ưu tiên trung bình**: 
   - Đổi Optimizer từ AdamW sang Adam
   - Điều chỉnh validation accuracy threshold

3. **Cần xác nhận**: 
   - PatchSize: Xác nhận với người yêu cầu xem "00" có nghĩa là 0 hay là lỗi đánh máy

4. **Lưu ý**: 
   - Data augmentation đã đúng hoàn toàn với yêu cầu
   - Loss function (CrossEntropyLoss) tương đương với CategoricalCrossentropy trong PyTorch

