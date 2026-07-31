#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

BATCH_SIZE="${BATCH_SIZE:-120}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-0}"
ADV_WARMUP_EPOCHS="${ADV_WARMUP_EPOCHS:-2}"
VAL_BATCHES="${VAL_BATCHES:-250}"
EXTRA_FLAGS=()
if [[ "${MIXED_PRECISION:-0}" == "1" ]]; then
  EXTRA_FLAGS+=(--mixed_precision)
fi
# Omitted by default: one epoch is then exactly one full pass over GSV-Cities
# (62,514 places with >=4 images). Set to shorten epochs for quick runs.
if [[ -n "${BATCHES_PER_EPOCH:-}" ]]; then
  EXTRA_FLAGS+=(--batches_per_epoch="${BATCHES_PER_EPOCH}")
fi

exec python3 train.py \
  --model=supervlad \
  --eval_datasets_folder=datasets \
  --gsv_cities_base_path=datasets/gsv_cities \
  --eval_dataset_name=msls \
  --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth \
  --backbone=dino \
  --supervlad_clusters=4 \
  --crossimage_encoder \
  --freeze_te=8 \
  --lr=0.00005 \
  --num_epochs="${EPOCHS}" \
  --patience=3 \
  --lr_plateau_patience=3 \
  --lr_plateau_factor=0.5 \
  --batch_size="${BATCH_SIZE}" \
  --seed="${SEED}" \
  --shuffle \
  --randomize_attack \
  --adv_negatives=5 \
  --adv_warmup_epochs="${ADV_WARMUP_EPOCHS}" \
  --keep_every=6 \
  --val_batches="${VAL_BATCHES}" \
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)" \
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)" \
  "${EXTRA_FLAGS[@]}"
