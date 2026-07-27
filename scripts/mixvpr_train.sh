#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

MIXVPR_DESCRIPTORS_DIMENSION="${MIXVPR_DESCRIPTORS_DIMENSION:-4096}"
case "${MIXVPR_DESCRIPTORS_DIMENSION}" in
  128)
    MIXVPR_CHECKPOINT_FILENAME="resnet50_MixVPR_128_channels(64)_rows(2)"
    ;;
  512)
    MIXVPR_CHECKPOINT_FILENAME="resnet50_MixVPR_512_channels(256)_rows(2)"
    ;;
  4096)
    MIXVPR_CHECKPOINT_FILENAME="resnet50_MixVPR_4096_channels(1024)_rows(4)"
    ;;
  *)
    echo "Unsupported MIXVPR_DESCRIPTORS_DIMENSION=${MIXVPR_DESCRIPTORS_DIMENSION}. Use 128, 512, or 4096." >&2
    exit 2
    ;;
esac

DEFAULT_MIXVPR_CHECKPOINT="third_party/VPR-methods-evaluation/trained_models/mixvpr/${MIXVPR_CHECKPOINT_FILENAME}"
MIXVPR_CHECKPOINT="${MIXVPR_CHECKPOINT:-${DEFAULT_MIXVPR_CHECKPOINT}}"

MODEL_WEIGHTS_ARGS=()
if [[ -f "${MIXVPR_CHECKPOINT}" ]]; then
  MODEL_WEIGHTS_ARGS=(--resume="${MIXVPR_CHECKPOINT}" --resume_model_only)
elif [[ "${MIXVPR_DOWNLOAD_WEIGHTS:-0}" == "1" ]]; then
  MODEL_WEIGHTS_ARGS=(--download_pretrained)
else
  echo "MixVPR weights were not found at ${MIXVPR_CHECKPOINT}." >&2
  echo "Set MIXVPR_CHECKPOINT to a local checkpoint or set MIXVPR_DOWNLOAD_WEIGHTS=1." >&2
  exit 2
fi

FREEZE_BACKBONE_ARGS=()
case "${MIXVPR_FREEZE_BACKBONE:-0}" in
  0)
    ;;
  1)
    FREEZE_BACKBONE_ARGS=(--mixvpr_freeze_backbone)
    ;;
  *)
    echo "MIXVPR_FREEZE_BACKBONE must be 0 or 1." >&2
    exit 2
    ;;
esac

exec python3 train.py \
  --model=mixvpr \
  "${MODEL_WEIGHTS_ARGS[@]}" \
  "${FREEZE_BACKBONE_ARGS[@]}" \
  --eval_datasets_folder=datasets \
  --gsv_cities_base_path=datasets/gsv_cities \
  --eval_dataset_name=msls \
  --mixvpr_descriptors_dimension="${MIXVPR_DESCRIPTORS_DIMENSION}" \
  --train_resize 320 320 \
  --resize 320 320 \
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
  --save_dir=mixvpr_perceptual_adv_training
