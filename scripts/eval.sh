#!/usr/bin/env bash
# Evaluate one model on one test set under one attack, into
# ${OUT_ROOT}/MODEL/ATTACK/DATASET/TAG/.
#
# Usage:
#   scripts/eval.sh MODEL ATTACK DATASET TAG CHECKPOINT EPSILON...
#
#   MODEL       supervlad | boq
#   ATTACK      apgd       rank-APGD L-inf, 100 steps x 1 restart
#               apgd500    rank-APGD L-inf, 500 steps x 1 restart
#               apgd_r3    rank-APGD L-inf, 100 steps x 3 restarts
#               pgd        rank-PGD L-inf, 20 steps x 1 restart
#   CHECKPOINT  checkpoint file (the pretrained release or a run's *.pth)
#   EPSILON     normalised L-inf budgets; 0.0685 is 4/255
#
# Attacked R@1 uses at most 1000 evenly spaced queries; clean R@1 uses all queries. A rerun
# skips a piece whose DONE file points at complete results.
#
# Environment:
#   OUT_ROOT  results root                   (default: output)
#   DRY_RUN=1 print the eval.py command instead of running it
#   PYTHON    python interpreter             (default: python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH="${PWD}:${PWD}/third_party/SuperVLAD:${PYTHONPATH:-}"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

[ "$#" -ge 6 ] || { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0" >&2; exit 2; }
MODEL=$1 ATTACK=$2 DATASET=$3 TAG=$4 CHECKPOINT=$5
shift 5
EPSILONS=("$@")
PYTHON=${PYTHON:-python}
DIR="${OUT_ROOT:-output}/${MODEL}/${ATTACK}/${DATASET}/${TAG}"

case "${MODEL}" in
  supervlad) MODEL_FLAGS=("${SUPERVLAD_EVAL_FLAGS[@]}") ;;
  boq) MODEL_FLAGS=("${BOQ_EVAL_FLAGS[@]}") ;;
  *) echo "unknown model: ${MODEL}" >&2; exit 2 ;;
esac
case "${ATTACK}" in
  apgd) ATTACK_FLAGS=(--rank_attack=rank_apgd_linf --rank_steps=100 --rank_restarts=1) ;;
  apgd500) ATTACK_FLAGS=(--rank_attack=rank_apgd_linf --rank_steps=500 --rank_restarts=1) ;;
  apgd_r3) ATTACK_FLAGS=(--rank_attack=rank_apgd_linf --rank_steps=100 --rank_restarts=3) ;;
  pgd) ATTACK_FLAGS=(--rank_attack=rank_pgd_linf --rank_steps=20 --rank_restarts=1) ;;
  *) echo "unknown attack: ${ATTACK}" >&2; exit 2 ;;
esac

# Complete results: the DONE file names a directory holding all three result files.
done_dir() {
  local result
  [ -f "${DIR}/DONE" ] || return 1
  result="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1]))["result_dir"])' "${DIR}/DONE")" || return 1
  for file in rank_eval_results.json rank_eval_results.csv per_query_ranks.csv; do
    [ -f "${result}/${file}" ] || return 1
  done
  echo "${result}"
}

if result="$(done_dir)"; then
  echo "=== SKIP ${MODEL}/${ATTACK}/${DATASET}/${TAG} (${result})"
  exit 0
fi
[ -f "${CHECKPOINT}" ] || [ "${DRY_RUN:-0}" = "1" ] || { echo "checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }
echo "=== EVAL ${MODEL}/${ATTACK}/${DATASET}/${TAG}"

COMMAND=(
  "${PYTHON}" eval.py
  --eval_datasets_folder=datasets --datasets "${DATASET}" --dataset_split=test
  "${MODEL_FLAGS[@]}"
  --model_paths "${CHECKPOINT}"
  --model_tags "${TAG}"
  --test_method=hard_resize
  "${ATTACK_FLAGS[@]}"
  --max_queries=1000
  --epsilons "${EPSILONS[@]}"
  --output_json="${DIR}/rank_eval_results.json"
  --output_csv="${DIR}/rank_eval_results.csv"
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '+'; printf ' %q' "${COMMAND[@]}"; printf '\n'
  exit 0
fi

mkdir -p "${DIR}"
"${COMMAND[@]}"
# eval.py writes into a new timestamped directory below DIR; record the newest one.
result="$(ls -d "${DIR}"/*/ | sort | tail -n 1)"
result="${result%/}"
for file in rank_eval_results.json rank_eval_results.csv per_query_ranks.csv; do
  [ -f "${result}/${file}" ] || { echo "incomplete results in ${result}: no ${file}" >&2; exit 1; }
done
printf '{"result_dir": "%s"}\n' "${result}" > "${DIR}/DONE"
echo "=== DONE ${MODEL}/${ATTACK}/${DATASET}/${TAG} (${result})"
