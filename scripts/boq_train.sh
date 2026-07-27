#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

BOQ_CHECKPOINT="${BOQ_CHECKPOINT:-checkpoints/boq_dinov2_12288.pth}"
MODEL_WEIGHTS_ARGS=()
if [[ -f "${BOQ_CHECKPOINT}" ]]; then
  MODEL_WEIGHTS_ARGS=(--resume="${BOQ_CHECKPOINT}" --resume_model_only)
elif [[ "${BOQ_DOWNLOAD_WEIGHTS:-0}" == "1" ]]; then
  MODEL_WEIGHTS_ARGS=(--download_pretrained)
else
  echo "BoQ weights were not found at ${BOQ_CHECKPOINT}." >&2
  echo "Set BOQ_CHECKPOINT to a local checkpoint or set BOQ_DOWNLOAD_WEIGHTS=1." >&2
  exit 2
fi

exec python3 train.py \
  --model=boq \
  "${MODEL_WEIGHTS_ARGS[@]}" \
  --eval_datasets_folder=datasets \
  --gsv_cities_base_path=datasets/gsv_cities \
  --eval_dataset_name=msls \
  --boq_backbone=Dinov2 \
  --boq_descriptors_dimension=12288 \
  --train_resize 224 224 \
  --resize 322 322 \
  --optim=adamw \
  --weight_decay=0.0001 \
  --lr=0.000005 \
  --num_epochs=100 \
  --batch_size=16 \
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)" \
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=3)" \
  --adv_loss_weight=0.25 \
  --adv_align_weight=0.2 \
  --adv_negatives=5 \
  --val_batches=100 \
  --save_dir=boq_perceptual_adv_training
