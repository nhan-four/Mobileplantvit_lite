#!/bin/bash
# Script để chạy training với 2 variants: ffn_expand 1.5 và 2.5
# Ablation #2: Lite-FFN / TokenConv-FFN

set -e  # Exit on error

echo "=========================================="
echo "Starting training ablation: ffn_expand variants"
echo "Ablation #2: Lite-FFN / TokenConv-FFN"
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
echo "Run 1/2: ffn_expand=1.5"
echo "=========================================="
python train.py \
    --model litepp \
    --main_data_root "$DATA_ROOT" \
    --encoder_depth 1 \
    --embed_dim 256 \
    --attn_rank 64 \
    --ffn_expand 1.5 \
    --head_type gap

echo ""
echo "=========================================="
echo "Run 1/2 completed: ffn_expand=1.5"
echo "=========================================="
echo ""
echo "Waiting 5 seconds before next run..."
sleep 5
echo ""

echo "=========================================="
echo "Run 2/2: ffn_expand=2.5"
echo "=========================================="
python train.py \
    --model litepp \
    --main_data_root "$DATA_ROOT" \
    --encoder_depth 1 \
    --embed_dim 256 \
    --attn_rank 64 \
    --ffn_expand 2.5 \
    --head_type gap

echo ""
echo "=========================================="
echo "Run 2/2 completed: ffn_expand=2.5"
echo "=========================================="
echo ""
echo "=========================================="
echo "All training runs completed!"
echo "=========================================="

