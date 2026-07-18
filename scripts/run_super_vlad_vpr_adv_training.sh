#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# One virtual epoch is 400 batches (~1/10 of the 3908-batch full GSV-Cities
# epoch at batch_size=16), so every epoch-denominated flag below (num_epochs,
# patience, lr_plateau_patience, adv_warmup_epochs, keep_every) is in
# virtual-epoch units.
exec python3 train.py \
  --model=supervlad \
  --eval_datasets_folder=datasets \
  --gsv_cities_base_path=datasets/gsv_cities \
  --eval_dataset_name=msls \
  --resume=checkpoints/SuperVLAD.pth \
  --resume_model_only \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --freeze_te=7 \
  --lr=0.000005 \
  --lr_plateau_patience=5 \
  --num_epochs=150 \
  --patience=15 \
  --batch_size=16 \
  --batches_per_epoch=400 \
  --mixed_precision \
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)" \
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)" \
  --attack "LinfAttack(model, num_iterations=10)" \
  --attack "StAdvAttack(model, num_iterations=10)" \
  --attack "ReColorAdvAttack(model, num_iterations=10)" \
  --randomize_attack \
  --adv_loss_weight=0.25 \
  --adv_align_weight=0.2 \
  --adv_negatives=5 \
  --adv_warmup_epochs=5 \
  --keep_every=3 \
  --val_batches=40
