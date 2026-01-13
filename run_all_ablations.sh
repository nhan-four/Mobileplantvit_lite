#!/bin/bash
# Script để chạy tất cả ablation experiments lần lượt
# Ablation studies cho MobilePlantViT Lite++

set -e  # Exit on error

echo "=========================================="
echo "Starting All Ablation Experiments"
echo "=========================================="
echo ""

# Activate conda environment
source /home/nhannv02/miniconda3/etc/profile.d/conda.sh
conda activate edgevit

# Change to project directory
cd /home/nhannv02/Hello/plantvit_lite/mobileplantvit_litepp_project_v2.1

# Dataset path
DATA_ROOT="/home/nhannv02/Hello/plantvit_lite/dataset/PlantVillage_origin_nonDelete_split_70_15_15"

# Common arguments
COMMON_ARGS="--model litepp --main_data_root $DATA_ROOT --encoder_depth 1 --embed_dim 256 --head_type gap"

# Counter
RUN_COUNT=0
TOTAL_RUNS=6

# Function to run training
run_training() {
    local run_name=$1
    local extra_args=$2
    RUN_COUNT=$((RUN_COUNT + 1))
    
    echo ""
    echo "=========================================="
    echo "Run $RUN_COUNT/$TOTAL_RUNS: $run_name"
    echo "=========================================="
    echo "Command: python train.py $COMMON_ARGS $extra_args"
    echo ""
    
    python train.py $COMMON_ARGS $extra_args
    
    echo ""
    echo "=========================================="
    echo "Run $RUN_COUNT/$TOTAL_RUNS completed: $run_name"
    echo "=========================================="
    
    if [ $RUN_COUNT -lt $TOTAL_RUNS ]; then
        echo "Waiting 10 seconds before next run..."
        sleep 10
        echo ""
    fi
}

# ============================================
# Ablation #1: Factorized Attention (attn_rank)
# ============================================
# echo "=========================================="
# echo "Ablation #1: Factorized Attention (attn_rank)"
# echo "=========================================="

# run_training "attn_rank=32 (small)" "--attn_rank 32 --ffn_expand 2.0 --attn_type factorized --ffn_type tokenconv"
# run_training "attn_rank=96 (large)" "--attn_rank 96 --ffn_expand 2.0 --attn_type factorized --ffn_type tokenconv"

# ============================================
# Ablation #2: TokenConv-FFN vs MLP-FFN
# ============================================
echo ""
echo "=========================================="
echo "Ablation #2: TokenConv-FFN vs MLP-FFN"
echo "=========================================="

run_training "MLP-FFN (baseline)" "--attn_rank 64 --ffn_expand 2.0 --attn_type factorized --ffn_type mlp"

# ============================================
# Ablation #3: Factorized vs Linear Attention
# ============================================
echo ""
echo "=========================================="
echo "Ablation #3: Factorized vs Linear Attention"
echo "=========================================="

run_training "Linear SA (baseline)" "--attn_rank 64 --ffn_expand 2.0 --attn_type linear --ffn_type tokenconv"

# ============================================
# Ablation #4: Head - GAP vs Attention Pooling
# ============================================
echo ""
echo "=========================================="
echo "Ablation #4: Head - GAP vs Attention Pooling"
echo "=========================================="

run_training "Attention pooling head" "--attn_rank 64 --ffn_expand 2.0 --attn_type factorized --ffn_type tokenconv --head_type attn"

# ============================================
# Baseline comparison: Linear + MLP (full baseline-style)
# ============================================
# echo ""
# echo "=========================================="
# echo "Baseline Comparison: Linear + MLP"
# echo "=========================================="

# run_training "Linear SA + MLP-FFN (full baseline)" "--attn_rank 64 --ffn_expand 2.0 --attn_type linear --ffn_type mlp"

# ============================================
# Summary
# ============================================
echo ""
echo "=========================================="
echo "All Ablation Experiments Completed!"
echo "=========================================="
echo "Total runs: $TOTAL_RUNS"
echo "Results saved in: runs_mobileplantvit/"
echo "=========================================="

