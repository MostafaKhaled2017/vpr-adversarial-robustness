#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

exec python3 train.py \
  --model=supervlad \
  --eval_datasets_folder=datasets \
  --gsv_cities_base_path=datasets/gsv_cities \
  --eval_dataset_name=msls \
  --resume=checkpoints/SuperVLAD_base.pth \
  --resume_model_only \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --lr_plateau_patience=5 \
  --num_epochs=100 \
  --patience=12 \
  --batch_size=16 \
  --batches_per_epoch=400 \
  --mixed_precision \
  --randomize_attack \
  --adv_negatives=5 \
  --keep_every=6 \
  --val_batches=200 \
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)" \
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)" \
  --lr=1e-5 \
  --freeze_te=8