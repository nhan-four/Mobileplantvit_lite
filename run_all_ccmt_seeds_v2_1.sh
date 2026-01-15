#!/usr/bin/env bash

set -e

# Thư mục gốc project v2.1 (chứa train.py)
PROJ_ROOT="/home/nhannv/Hello/ICN/Mobilevit/Mobileplantvit_lite"
cd "$PROJ_ROOT"

DATA_ROOT_BASE="/home/nhannv/Hello/ICN/Mobilevit/Data_split_cleaned"

CROPS=(Cashew Cassava Maize Tomato)
# SEEDS=(42 123 999)
SEEDS=(100 200 300 400 500)
for CROP in "${CROPS[@]}"; do
  for SEED in "${SEEDS[@]}"; do
    DATA_ROOT="${DATA_ROOT_BASE}/${CROP}/seed_${SEED}"
    echo "=================================================="
    echo "Running v2.1: CROP=${CROP}, SEED=${SEED}"
    echo "Data root: ${DATA_ROOT}"
    echo "Start time: $(date)"

    python train.py \
      --model litepp \
      --main_data_root "${DATA_ROOT}" \
      --encoder_depth 1 \
      --embed_dim 256 \
      --attn_rank 96 \
      --ffn_expand 2.0 \
      --head_type gap \
      --num_epochs_main 100 \
      # --batch_size 32 \
      
    echo "Finished v2.1: CROP=${CROP}, SEED=${SEED} at $(date)"
    echo
  done
done

echo "All runs for v2.1 completed."
