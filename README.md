# MobilePlantViT (baseline + Lite++)

## Model Complexity (224x224 input, 27 classes)

| Model | Params | MACs | FLOPs |
|-------|--------|------|-------|
| Baseline | 1.052M | 0.374G | 0.749G |
| Lite++ | 0.901M | 0.346G | 0.692G |

**Lưu ý:** FLOPs và model complexity được tự động tính và lưu vào `runs_mobileplantvit/<run_name>/main_model/model_complexity.json` và `model_complexity.csv` khi training.

## Folder structure (dataset)

Dataset cần có cấu trúc:
```
root/
  train/<class_name>/*.jpg|png
  val/<class_name>/*.jpg|png
  test/<class_name>/*.jpg|png
```

## Split dataset

Nếu dataset chưa được chia (các class folders nằm trực tiếp trong root), sử dụng script `split_dataset.py` để chia theo tỷ lệ 80-10-10:

```bash
python split_dataset.py \
  --source_dir /home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete \
  --output_dir /home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete_split \
  --seed 42
```

**Tùy chọn:**
- `--train_ratio`: Tỷ lệ train (default: 0.8)
- `--val_ratio`: Tỷ lệ val (default: 0.1)
- `--seed`: Random seed (default: 42)

## Run

**Baseline:**
```bash
python train.py --model baseline --main_data_root /home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete_split
```

**Lite++ (recommended):**
```bash
python train.py --model litepp --main_data_root /home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete_split \
  --encoder_depth 1 --embed_dim 256 --attn_rank 64 --ffn_expand 2.0 --head_type gap

python train.py --model litepp --main_data_root /home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete_split_70_15_15   --encoder_depth 1 --embed_dim 256 --attn_rank 64 --ffn_expand 2.0 --head_type gap
```

**Lite++ + token pooling (ablation):**
```bash
python train.py --model litepp --main_data_root /home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete_split \
  --encoder_depth 2 --use_token_pool --pool_at 1 --pool_type avg
```

**Lite++ lightweight variant:**
```bash
python train.py --model litepp --main_data_root /home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete_split_70_15_15 \
  --encoder_depth 2 --embed_dim 192 --attn_rank 32 --ffn_expand 1.5 --head_type gap
```
## Ablation Studies

### Ablation Switches

Lite++ hỗ trợ 2 switch để ablation từng module:

| Switch | Options | Default | Description |
|--------|---------|---------|-------------|
| `--attn_type` | `factorized`, `linear` | `factorized` | Factorized LSA vs Linear SA (baseline-style) |
| `--ffn_type` | `tokenconv`, `mlp` | `tokenconv` | TokenConv-FFN vs MLP-FFN (baseline-style) |

**Model complexity với các combinations (encoder_depth=1, embed_dim=256):**

| attn_type | ffn_type | Params | MACs |
|-----------|----------|--------|------|
| factorized | tokenconv | 0.515M | 0.270G |
| factorized | mlp | 0.509M | 0.268G |
| linear | tokenconv | 0.597M | 0.286G |
| linear | mlp | 0.590M | 0.284G |

### Ablation #1: Factorized Attention (attn_rank)

```bash
# Rank nhỏ (rẻ hơn)
python train.py --model litepp --main_data_root /path/dataset \
  --encoder_depth 1 --embed_dim 256 --attn_rank 32 --ffn_expand 2.0 --head_type gap

# Rank lớn (đắt hơn)
python train.py --model litepp --main_data_root /path/dataset \
  --encoder_depth 1 --embed_dim 256 --attn_rank 96 --ffn_expand 2.0 --head_type gap
```

### Ablation #2: TokenConv-FFN vs MLP-FFN

```bash
# TokenConv-FFN (default)
python train.py --model litepp --main_data_root /path/dataset \
  --encoder_depth 1 --embed_dim 256 --attn_rank 64 --ffn_expand 2.0 --ffn_type tokenconv

# MLP-FFN (baseline-style)
python train.py --model litepp --main_data_root /path/dataset \
  --encoder_depth 1 --embed_dim 256 --attn_rank 64 --ffn_expand 2.0 --ffn_type mlp
```

### Ablation #3: Factorized vs Linear Attention

```bash
# Factorized LSA (default)
python train.py --model litepp --main_data_root /path/dataset \
  --encoder_depth 1 --embed_dim 256 --attn_rank 64 --attn_type factorized

# Linear SA (baseline-style, không factorize)
python train.py --model litepp --main_data_root /path/dataset \
  --encoder_depth 1 --embed_dim 256 --attn_type linear
```

### Ablation #4: Head - GAP vs Attention Pooling

```bash
python train.py --model litepp --main_data_root /path/dataset \
  --encoder_depth 1 --embed_dim 256 --head_type gap   # GAP head

python train.py --model litepp --main_data_root /path/dataset \
  --encoder_depth 1 --embed_dim 256 --head_type attn  # Attention pooling head
```



## Features

### Training Features
- **Mixed Precision Training (AMP)**: Tự động sử dụng FP16/BF16 trên CUDA để tăng tốc training và tiết kiệm bộ nhớ
- **Best Model Saving**: Tự động lưu best model checkpoint dựa trên validation accuracy
- **Early Stopping**: Dựa trên validation accuracy với patience mặc định 15 epochs
- **Model Complexity Analysis**: Tự động tính và lưu FLOPs, MACs, và parameters

### Data Augmentation

Default augmentation là `paper` (theo paper specifications):

| Augmentation | Probability |
|-------------|-------------|
| HorizontalFlip | 50% |
| RandomRotate90 | 50% |
| ShiftScaleRotate (0.05, 0.05, 30°) | 50% |
| RandomGamma | 20% |
| RandomBrightnessContrast | 30% |
| RGBShift (limits=15) | 30% |
| CLAHE (clip_limit=4.0) | 30% |

**Các chế độ augmentation:**
- `paper`: Augmentation theo paper specifications **(default, recommended)**
- `none`: Không augmentation (chỉ resize + normalize)
- `light`: Flip + brightness/contrast
- `medium`: Moderate augmentation
- `strong`: Heavy augmentation

```bash
# Sử dụng augmentation khác
python train.py --main_data_root /path/dataset --augment_level none
```

### Default Settings
- **Augmentation**: `paper` (có thể thay đổi bằng `--augment_level [none|light|medium|strong|paper]`)
- **Image size**: 224x224
- **Batch size**: 32
- **Learning rate**: 3e-4 với ReduceLROnPlateau scheduler

### Output Files
Mỗi run sẽ tạo folder trong `runs_mobileplantvit/<timestamp>/` chứa:
- `run_config.json`: Cấu hình training
- `main_model/param_breakdown.json`: Phân tích parameters theo module
- `main_model/model_complexity.json`: FLOPs, MACs, parameters
- `train/history.csv`: Training history
- `train/checkpoints/`: Model checkpoints (best và last)
- `train/test_metrics.json`: Test metrics và confusion matrix

## Requirements

```bash
pip install -r requirements.txt
```

Dependencies: torch, torchvision, albumentations, opencv-python, numpy, Pillow, matplotlib, thop
