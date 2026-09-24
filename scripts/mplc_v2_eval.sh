#!/usr/bin/env bash
# Evaluate the MPLC v2 pretrained baseline plus every finished clean_ft/mplc checkpoint
# (scripts/mplc_v2_train.sh) under the batch-independent rank-PGD threat model, one
# eval.py call per dataset.
#
# Usage:
#   scripts/mplc_v2_eval.sh
#
# Environment overrides:
#   MPLC_V2_SEEDS        seeds to look for checkpoints under  (default: "0 1")
#   MPLC_V2_DATASETS     evaluation datasets                  (default: "msls sped nordland")
#   MPLC_V2_EPSILONS     attack budgets                       (default: "0.01 0.1")
#   MPLC_V2_OUTPUT_ROOT  evaluation output root                (default: output/mplc_v2)
#   MPLC_V2_MODELS       "tag=path tag=path ..." replaces the discovered model list
#   MPLC_V2_DRY_RUN=1    print the commands without running them
#   PYTHON               python interpreter                    (default: python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

PYTHON=${PYTHON:-python}
SEEDS=${MPLC_V2_SEEDS:-"0 1"}
DATASETS=${MPLC_V2_DATASETS:-"msls sped nordland"}
EPSILONS=${MPLC_V2_EPSILONS:-"0.01 0.1"}
OUTPUT_ROOT=${MPLC_V2_OUTPUT_ROOT:-output/mplc_v2}
DRY_RUN=${MPLC_V2_DRY_RUN:-0}

run() {
  echo "+ $*"
  if [ "${DRY_RUN}" != "1" ]; then
    "$@"
  fi
}

latest_finished_run() {
  local name=$1 newest="" candidate
  for candidate in "logs/${name}"/*/; do
    [ -f "${candidate}best_model.pth" ] && newest="${candidate%/}"
  done
  echo "${newest}"
}

MODEL_PATHS=()
MODEL_TAGS=()

if [ -n "${MPLC_V2_MODELS:-}" ]; then
  for entry in ${MPLC_V2_MODELS}; do
    MODEL_TAGS+=("${entry%%=*}")
    MODEL_PATHS+=("${entry#*=}")
  done
else
  if [ -e checkpoints/SuperVLAD_base.pth ]; then
    MODEL_PATHS+=("checkpoints/SuperVLAD_base.pth")
    MODEL_TAGS+=("pretrained")
  fi

  for seed in ${SEEDS}; do
    for arm in clean_ft mplc; do
      name="mplc_v2_supervlad_${arm}_s${seed}"
      run_dir="$(latest_finished_run "${name}")"
      if [ -n "${run_dir}" ]; then
        MODEL_PATHS+=("${run_dir}/best_model.pth")
        MODEL_TAGS+=("${arm}_s${seed}")
      else
        echo "WARNING: no finished checkpoint for ${name}" >&2
      fi
    done
  done
fi

[ "${#MODEL_PATHS[@]}" -gt 0 ] || { echo "No model resolves to evaluate." >&2; exit 1; }

for dataset in ${DATASETS}; do
  run "${PYTHON}" eval.py \
    --eval_datasets_folder=datasets --datasets "${dataset}" --dataset_split=test \
    "${SUPERVLAD_EVAL_MODEL_FLAGS[@]}" \
    --model_paths "${MODEL_PATHS[@]}" \
    --model_tags "${MODEL_TAGS[@]}" \
    --test_method=hard_resize \
    --rank_attack=rank_pgd_linf --rank_steps=20 --rank_restarts=1 \
    --epsilons ${EPSILONS} \
    --output_json="${OUTPUT_ROOT}/supervlad_${dataset}/rank_eval_results.json" \
    --output_csv="${OUTPUT_ROOT}/supervlad_${dataset}/rank_eval_results.csv"
done
