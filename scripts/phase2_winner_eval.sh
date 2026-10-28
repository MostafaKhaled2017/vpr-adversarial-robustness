#!/usr/bin/env bash
# Evaluate the selected Phase 2 seed-0 winner on full test benchmarks.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

PYTHON=${PYTHON:-python3}
WINNER=${PHASE2_WINNER_CHECKPOINT:-logs/phase2_supervlad_pilot_tau0.05_k1_pool0/2026-09-16_22-09-17/best_model.pth}
TAG=${PHASE2_WINNER_TAG:-mplc_pilot_tau0.05_k1_pool0_s0}
DATASETS=${PHASE2_WINNER_DATASETS:-"msls sped"}
EPSILONS=${PHASE2_WINNER_EPSILONS:-"0.01 0.1"}
OUTPUT_ROOT=${PHASE2_WINNER_OUTPUT_ROOT:-"output/phase2/winner_${TAG}"}

[ -f "${WINNER}" ] || { echo "Winner checkpoint not found: ${WINNER}" >&2; exit 1; }

for dataset in ${DATASETS}; do
  "${PYTHON}" eval.py \
    --eval_datasets_folder=datasets --datasets "${dataset}" --dataset_split=test \
    "${SUPERVLAD_EVAL_MODEL_FLAGS[@]}" \
    --model_paths "${WINNER}" --model_tags "${TAG}" \
    --test_method=hard_resize \
    --rank_attack=rank_pgd_linf --rank_steps=20 --rank_restarts=1 \
    --epsilons ${EPSILONS} \
    --output_json="${OUTPUT_ROOT}/supervlad_${dataset}/rank_eval_results.json" \
    --output_csv="${OUTPUT_ROOT}/supervlad_${dataset}/rank_eval_results.csv"
done
