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
  --patience=12
  --batch_size=16
  --batches_per_epoch=400
  --randomize_attack
  --adv_negatives=5
  --keep_every=6
  --val_batches=200
  # --mixed_precision
)

ATTACKS_ITER5=(
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)"
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)"
)

ATTACKS_ITER7=(
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=7)"
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=7)"
)

ATTACKS_ITER10=(
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=10)"
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=10)"
)

# Compare between this run and next one to see effect of changing learning rate
# If this lead to improvement, try 2e-5
# python3 train.py \
#   "${COMMON_ARGS[@]}" \
#   "${ATTACKS_ITER5[@]}" \
#   --lr=1e-5  \
#   --freeze_te=8 \

# python3 train.py \
#   "${COMMON_ARGS[@]}" \
#   "${ATTACKS_ITER5[@]}" \
#   --lr=0.000005 \
#   --freeze_te=8 \

# Comparing this run with the previous one shows the effect of relying on robust loss only
python3 train.py \
  "${COMMON_ARGS[@]}" \
  "${ATTACKS_ITER5[@]}" \
  --lr=0.000005 \
  --freeze_te=8 \
  --selection_robust_weight=1.0 \

# Compare this run with the run that had similar arguments except for attacks generation
python3 train.py \
  "${COMMON_ARGS[@]}" \
  "${ATTACKS_ITER10[@]}" \
  --lr=0.000005 \
  --freeze_te=8 \

# Compare this run with the one that has similar arguments except for number of frozen layers
python3 train.py \
  "${COMMON_ARGS[@]}" \
  "${ATTACKS_ITER5[@]}" \
  --lr=0.000005 \
  --freeze_te=9 \