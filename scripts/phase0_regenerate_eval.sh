#!/usr/bin/env bash
# Re-run the evaluation conditions behind the current paper through the Phase 0
# corrected reporting pipeline: per-query rank export, dual CC-ASR, and
# distributional rank displacement. No retraining and no attack changes — the
# checkpoints and attack flags below are recovered verbatim from the run metadata
# of the original evaluations, so only the metrics and artifacts differ.
#
# Usage:
#   scripts/phase0_regenerate_eval.sh sped            # cheap slice (~0.6 h, all archs)
#   scripts/phase0_regenerate_eval.sh msls sped       # full grid (~36 h)
#
# Environment overrides:
#   PHASE0_ARCHS          architectures to run (default: "supervlad boq mixvpr")
#   PHASE0_EPSILONS       attack budgets (default: "0.01 0.1")
#   PHASE0_OUTPUT_ROOT    output root (default: output/phase0_regen)
#   SUPERVLAD_BASE_PATH   SuperVLAD pretrained checkpoint
#   SUPERVLAD_ADV_PATH    SuperVLAD adversarially fine-tuned checkpoint
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

export PYTHONPATH="${PWD}:${PWD}/third_party/SuperVLAD:${PYTHONPATH:-}"

PYTHON=${PYTHON:-venv/bin/python3}
DATASETS=("$@")
if [ ${#DATASETS[@]} -eq 0 ]; then
  echo "usage: $0 <dataset> [dataset ...]" >&2
  exit 2
fi

ARCHS=${PHASE0_ARCHS:-"supervlad boq mixvpr"}
EPSILONS=${PHASE0_EPSILONS:-"0.01 0.1"}
OUTPUT_ROOT=${PHASE0_OUTPUT_ROOT:-output/phase0_regen}

# BoQ and MixVPR paths are recovered verbatim from the original run metadata.
#
# The SuperVLAD pair is NOT: the checkpoints the original run used
# (checkpoints/SuperVLAD.pth, checkpoints/perceptual_adv_checkpoint.pth) are absent from
# this working tree. The pair below is the current SuperVLAD base release plus the
# adversarially fine-tuned checkpoint from this project, which is what we carry forward.
# SuperVLAD numbers from this script are therefore NOT comparable to the SuperVLAD row of
# the paper's existing table.
SUPERVLAD_BASE_PATH=${SUPERVLAD_BASE_PATH:-checkpoints/SuperVLAD_base.pth}
SUPERVLAD_ADV_PATH=${SUPERVLAD_ADV_PATH:-checkpoints/supervlad_adv_trained.pth}
BOQ_BASE_PATH=checkpoints/boq_dinov2_12288.pth
BOQ_ADV_PATH=logs/boq_perceptual_adv_training/2026-07-13_07-23-11/best_model.pth
MIXVPR_BASE_PATH='third_party/VPR-methods-evaluation/trained_models/mixvpr/resnet50_MixVPR_4096_channels(1024)_rows(4)'
MIXVPR_ADV_PATH=logs/mixvpr_perceptual_adv_training/2026-07-12_23-35-14/best_model.pth

run_condition() {
  local arch=$1 dataset=$2
  local run_dir="${OUTPUT_ROOT}/${arch}_${dataset}"
  local -a model_args

  case "$arch" in
    supervlad)
      if [ ! -e "$SUPERVLAD_BASE_PATH" ] || [ ! -e "$SUPERVLAD_ADV_PATH" ]; then
        echo "SKIP ${arch}/${dataset}: missing ${SUPERVLAD_BASE_PATH} or ${SUPERVLAD_ADV_PATH}." >&2
        echo "     Set SUPERVLAD_BASE_PATH / SUPERVLAD_ADV_PATH to regenerate these conditions." >&2
        return 0
      fi
      model_args=(
        --model_type=supervlad
        --model_paths "$SUPERVLAD_BASE_PATH" "$SUPERVLAD_ADV_PATH"
        --model_tags base best
        --foundation_model_path=checkpoints/dinov2_vitb14_pretrain.pth
        --backbone=dino
        --supervlad_clusters=4
        --crossimage_encoder
        --freeze_te=8
        # The original run used batch 16 on a larger GPU; SuperVLAD's attack phase does not
        # fit in 10 GB at that size. Batch size and checkpointing affect memory, not metrics.
        --infer_batch_size=8
        --grad_checkpointing
      )
      ;;
    boq)
      model_args=(
        --model_type=boq
        --model_paths "$BOQ_BASE_PATH" "$BOQ_ADV_PATH"
        --model_tags base best
        --boq_backbone=Dinov2
        --boq_descriptors_dimension=12288
        --infer_batch_size=8
      )
      ;;
    mixvpr)
      model_args=(
        --model_type=mixvpr
        --model_paths "$MIXVPR_BASE_PATH" "$MIXVPR_ADV_PATH"
        --model_tags base best
        --mixvpr_descriptors_dimension=4096
        --infer_batch_size=8
      )
      ;;
    *)
      echo "unknown architecture: ${arch}" >&2
      return 1
      ;;
  esac

  echo "=== ${arch} / ${dataset} / eps ${EPSILONS} -> ${run_dir}"
  # shellcheck disable=SC2086
  "$PYTHON" eval.py \
    --eval_datasets_folder=datasets \
    --datasets "$dataset" \
    "${model_args[@]}" \
    --test_method=hard_resize \
    --rank_attack=rank_pgd_linf \
    --rank_steps=20 \
    --rank_restarts=1 \
    --epsilons ${EPSILONS} \
    --output_json="${run_dir}/rank_eval_results.json" \
    --output_csv="${run_dir}/rank_eval_results.csv"
}

for dataset in "${DATASETS[@]}"; do
  for arch in ${ARCHS}; do
    run_condition "$arch" "$dataset"
  done
done

echo "Done. Per-query rank CSVs and paired CC-ASR inputs are under ${OUTPUT_ROOT}/."
echo "Compute paired CC-ASR with:"
echo "  ${PYTHON} -m src.paired_analysis --base_csv <run_dir>/per_query_ranks.csv \\"
echo "      --base_model_tag base --trained_model_tag best"
