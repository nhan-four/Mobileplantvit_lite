#!/usr/bin/env bash

set -e

# Thư mục gốc project v2.1 (chứa train.py)
PROJ_ROOT="/home/nhannv02/Hello/plantvit_lite/mobileplantvit_litepp_project_v2.1"
cd "$PROJ_ROOT"

DATA_ROOT_BASE="/home/nhannv02/Hello/plantvit_lite/dataset/Dataset_for_Crop_Pest_and_Disease_Detection/Data_split"

# Cashew đã chạy xong với 3 seed (42, 123, 999)
CROPS=(Tomato Maize Cassava)
SEEDS=(42 123 999)

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
      --num_epochs_main 80 \
      --batch_size 64 \
      --augment_level paper

    echo "Finished v2.1: CROP=${CROP}, SEED=${SEED} at $(date)"
    echo
  done
done

echo "All runs for v2.1 completed."
