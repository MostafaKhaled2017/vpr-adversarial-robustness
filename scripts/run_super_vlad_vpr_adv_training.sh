#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

COMMON_ARGS=(
  --model=supervlad
  --eval_datasets_folder=datasets
  --gsv_cities_base_path=datasets/gsv_cities
  --eval_dataset_name=msls
  --resume=checkpoints/SuperVLAD_base.pth
  --resume_model_only
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth
  --backbone=dino
  --supervlad_clusters=4
  --crossimage_encoder
  --lr_plateau_patience=5
  --num_epochs=150
  --patience=15
  --batch_size=16
  --batches_per_epoch=400
  --mixed_precision
  --randomize_attack
  --adv_negatives=5
  --keep_every=3
  --val_batches=200
)

ATTACKS_ITER5=(
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)"
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)"
  --attack "StAdvAttack(model, num_iterations=5)"
  --attack "ReColorAdvAttack(model, num_iterations=5)"
)

ATTACKS_ITER7=(
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=7)"
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=7)"
  --attack "StAdvAttack(model, num_iterations=7)"
  --attack "ReColorAdvAttack(model, num_iterations=7)"
)

# Run 1: baseline configuration.
python3 train.py \
  "${COMMON_ARGS[@]}" \
  "${ATTACKS_ITER5[@]}" \
  --freeze_te=7 \
  --lr=0.000005 \

# Run 2: same as Run 1 but with a shuffled dataset ordering.
python3 train.py \
  "${COMMON_ARGS[@]}" \
  "${ATTACKS_ITER5[@]}" \
  --freeze_te=7 \
  --lr=0.000005 \
  --shuffle \

# Run 3
python3 train.py \
  "${COMMON_ARGS[@]}" \
  "${ATTACKS_ITER7[@]}" \
  --freeze_te=7 \
  --lr=0.000005 \