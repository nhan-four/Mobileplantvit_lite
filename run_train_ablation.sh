#!/bin/bash
# Script để chạy training với 2 variants: attn_rank 96 và 32

set -e  # Exit on error

echo "=========================================="
echo "Starting training ablation: attn_rank variants"
echo "=========================================="
echo ""

# Activate conda environment
source /home/nhannv02/miniconda3/etc/profile.d/conda.sh
conda activate edgevit

# Change to project directory
cd /home/nhannv02/Hello/plantvit_lite/mobileplantvit_litepp_project_v2

# Dataset path
DATA_ROOT="/home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete_split_70_15_15"

echo "=========================================="
echo "Run 1/2: attn_rank=96"
echo "=========================================="
python train.py \
    --model litepp \
    --main_data_root "$DATA_ROOT" \
    --encoder_depth 1 \
    --embed_dim 256 \
    --attn_rank 96 \
    --ffn_expand 2.0 \
    --head_type gap

echo ""
echo "=========================================="
echo "Run 1/2 completed: attn_rank=96"
echo "=========================================="
echo ""
echo "Waiting 5 seconds before next run..."
sleep 5
echo ""

echo "=========================================="
echo "Run 2/2: attn_rank=32"
echo "=========================================="
python train.py \
    --model litepp \
    --main_data_root "$DATA_ROOT" \
    --encoder_depth 1 \
    --embed_dim 256 \
    --attn_rank 32 \
    --ffn_expand 2.0 \
    --head_type gap

echo ""
echo "=========================================="
echo "Run 2/2 completed: attn_rank=32"
echo "=========================================="
echo ""
echo "=========================================="
echo "All training runs completed!"
echo "=========================================="

