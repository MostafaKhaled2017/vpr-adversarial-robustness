#!/usr/bin/env bash
# Screen Phase 2 standalone pilots on one fixed MSLS validation threat model.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

PYTHON=${PYTHON:-python3}
OUTPUT_ROOT=${PHASE2_PILOT_SCREEN_OUTPUT_ROOT:-output/phase2/pilot_screen}
DATASET=${PHASE2_PILOT_SCREEN_DATASET:-msls}
EPSILONS=${PHASE2_PILOT_SCREEN_EPSILONS:-"0.01 0.1"}

latest_finished_run() {
  local name=$1 newest="" candidate
  for candidate in "logs/${name}"/*/; do
    [ -f "${candidate}best_model.pth" ] && newest="${candidate%/}"
  done
  echo "${newest}"
}

add_model() {
  local path=$1 tag=$2
  MODEL_PATHS+=("${path}")
  MODEL_TAGS+=("${tag}")
}

MODEL_PATHS=()
MODEL_TAGS=()

for tau in 0.01 0.05; do
  for k in 1 5; do
    for pool in 0 4096; do
      label="tau${tau}_k${k}_pool${pool}"
      run_dir="$(latest_finished_run "phase2_supervlad_pilot_${label}")"
      [ -n "${run_dir}" ] || { echo "WARNING: missing pilot ${label}; skipping." >&2; continue; }
      if cmp -s "${run_dir}/best_model.pth" "${run_dir}/initial_validation_model.pth"; then
        echo "=== EXCLUDE ${label}: selected checkpoint is initial_validation_model.pth"
        continue
      fi
      add_model "${run_dir}/best_model.pth" "${label}"
    done
  done
done

[ "${#MODEL_PATHS[@]}" -gt 0 ] || { echo "No trained pilot checkpoint is eligible for screening." >&2; exit 1; }

exec "${PYTHON}" eval.py \
  --eval_datasets_folder=datasets --datasets "${DATASET}" --dataset_split=val \
  "${SUPERVLAD_EVAL_MODEL_FLAGS[@]}" \
  --model_paths "${MODEL_PATHS[@]}" \
  --model_tags "${MODEL_TAGS[@]}" \
  --test_method=hard_resize \
  --rank_attack=rank_pgd_linf --rank_steps=20 --rank_restarts=1 \
  --epsilons ${EPSILONS} \
  --output_json="${OUTPUT_ROOT}/${DATASET}_val/rank_eval_results.json" \
  --output_csv="${OUTPUT_ROOT}/${DATASET}_val/rank_eval_results.csv"
