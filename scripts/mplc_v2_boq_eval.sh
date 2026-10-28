#!/usr/bin/env bash
# Resumable BoQ rank-attack evaluation, one model/dataset piece per process.
#
# Every piece writes eval.py's timestamped results below:
#   <output-root>/boq_<dataset>_<attack>_steps<N>[_r<N>][_q<N>]/<model-tag>/
# A piece is complete only when DONE points to a directory containing JSON,
# CSV, and per-query results. Rerunning skips only those complete pieces.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

PYTHON=${PYTHON:-python}
DATASETS=${MPLC_V2_BOQ_DATASETS:-"msls pitts30k sped nordland"}
EPSILONS=${MPLC_V2_BOQ_EPSILONS:-"0.0342 0.0685"}
OUTPUT_ROOT=${MPLC_V2_BOQ_OUTPUT_ROOT:-output/mplc_v2_step4/boq/r1_apgd/all_checkpoints}
ATTACK=${MPLC_V2_BOQ_ATTACK:-rank_apgd_linf}
STEPS=${MPLC_V2_BOQ_STEPS:-100}
RESTARTS=${MPLC_V2_BOQ_RESTARTS:-1}
MAX_QUERIES=${MPLC_V2_BOQ_MAX_QUERIES:-1000}
NUM_WORKERS=${MPLC_V2_BOQ_NUM_WORKERS:-2}
INFER_BATCH_SIZE=${MPLC_V2_BOQ_INFER_BATCH_SIZE:-4}
GOAL=${MPLC_V2_BOQ_GOAL:-untargeted}
MODE=run
MODELS=()

usage() {
  sed -n '2,/^set -euo pipefail$/{/^set -euo pipefail$/d;s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"
  cat <<'EOF'

Usage:
  scripts/mplc_v2_boq_eval.sh [options]

Options:
  --model TAG=PATH       model checkpoint; repeatable
  --datasets "A B"       datasets to evaluate
  --epsilons "A B"       normalized L-inf budgets
  --output-root DIR      resumable piece root
  --attack NAME          rank_apgd_linf (default)
  --steps N              attack steps (default: 100)
  --restarts N           attack restarts (default: 1)
  --max-queries N        attacked-query cap (default: 1000)
  --num-workers N        DataLoader workers (default: 2)
  --infer-batch-size N   inference batch size (default: 4)
  --goal NAME            untargeted or targeted (default: untargeted)
  --dry-run              list pending pieces without evaluating
  --summary-only         report piece status without evaluating
  -h, --help             show this help
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --model)
      [ "$#" -ge 2 ] || { echo "--model requires TAG=PATH" >&2; exit 2; }
      MODELS+=("$2")
      shift 2
      ;;
    --datasets)
      [ "$#" -ge 2 ] || { echo "--datasets requires a space-separated value" >&2; exit 2; }
      DATASETS=$2
      shift 2
      ;;
    --epsilons)
      [ "$#" -ge 2 ] || { echo "--epsilons requires a space-separated value" >&2; exit 2; }
      EPSILONS=$2
      shift 2
      ;;
    --output-root)
      [ "$#" -ge 2 ] || { echo "--output-root requires a directory" >&2; exit 2; }
      OUTPUT_ROOT=$2
      shift 2
      ;;
    --attack)
      [ "$#" -ge 2 ] || { echo "--attack requires a value" >&2; exit 2; }
      ATTACK=$2
      shift 2
      ;;
    --steps)
      [ "$#" -ge 2 ] || { echo "--steps requires a value" >&2; exit 2; }
      STEPS=$2
      shift 2
      ;;
    --restarts)
      [ "$#" -ge 2 ] || { echo "--restarts requires a value" >&2; exit 2; }
      RESTARTS=$2
      shift 2
      ;;
    --max-queries)
      [ "$#" -ge 2 ] || { echo "--max-queries requires a value" >&2; exit 2; }
      MAX_QUERIES=$2
      shift 2
      ;;
    --num-workers)
      [ "$#" -ge 2 ] || { echo "--num-workers requires a value" >&2; exit 2; }
      NUM_WORKERS=$2
      shift 2
      ;;
    --infer-batch-size)
      [ "$#" -ge 2 ] || { echo "--infer-batch-size requires a value" >&2; exit 2; }
      INFER_BATCH_SIZE=$2
      shift 2
      ;;
    --goal)
      [ "$#" -ge 2 ] || { echo "--goal requires a value" >&2; exit 2; }
      GOAL=$2
      shift 2
      ;;
    --dry-run) MODE=dry; shift ;;
    --summary-only) MODE=summary; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

[ "${#MODELS[@]}" -gt 0 ] || { echo "at least one --model TAG=PATH is required" >&2; exit 2; }

read -r -a DATASET_LIST <<< "${DATASETS}"
read -r -a EPSILON_LIST <<< "${EPSILONS}"
[ "${#DATASET_LIST[@]}" -gt 0 ] || { echo "no datasets specified" >&2; exit 2; }
[ "${#EPSILON_LIST[@]}" -gt 0 ] || { echo "no epsilons specified" >&2; exit 2; }

MODEL_TAGS=()
MODEL_PATHS=()
for entry in "${MODELS[@]}"; do
  [[ "${entry}" == *=* ]] || { echo "malformed model (expected TAG=PATH): ${entry}" >&2; exit 2; }
  tag=${entry%%=*}
  path=${entry#*=}
  [[ "${tag}" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "invalid model tag: ${tag}" >&2; exit 2; }
  [ -f "${path}" ] || { echo "checkpoint not found: ${path}" >&2; exit 2; }
  MODEL_TAGS+=("${tag}")
  MODEL_PATHS+=("${path}")
done

OUTPUT_SUFFIX=""
[ "${ATTACK}" = "rank_pgd_linf" ] || OUTPUT_SUFFIX+="_${ATTACK}"
[ "${GOAL}" = "untargeted" ] || OUTPUT_SUFFIX+="_${GOAL}"
[ "${STEPS}" = "20" ] || OUTPUT_SUFFIX+="_steps${STEPS}"
[ "${RESTARTS}" = "1" ] || OUTPUT_SUFFIX+="_r${RESTARTS}"
[ -z "${MAX_QUERIES}" ] || OUTPUT_SUFFIX+="_q${MAX_QUERIES}"

piece_dir() {
  local dataset=$1 tag=$2
  printf '%s/boq_%s%s/%s\n' "${OUTPUT_ROOT}" "${dataset}" "${OUTPUT_SUFFIX}" "${tag}"
}

piece_result_dir() {
  local dir=$1 marker="${dir}/DONE" result_dir
  [ -f "${marker}" ] || return 1
  result_dir="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["result_dir"])' "${marker}" 2>/dev/null)" || return 1
  [ -d "${result_dir}" ] || return 1
  local name
  for name in rank_eval_results.json rank_eval_results.csv per_query_ranks.csv; do
    [ -f "${result_dir}/${name}" ] || return 1
  done
  printf '%s\n' "${result_dir}"
}

record_piece() {
  local dir=$1 result_dir="" candidate
  for candidate in "${dir}"/*/; do
    [ -d "${candidate}" ] || continue
    if [ -f "${candidate}rank_eval_results.json" ] && \
       [ -f "${candidate}rank_eval_results.csv" ] && \
       [ -f "${candidate}per_query_ranks.csv" ]; then
      result_dir=${candidate%/}
    fi
  done
  [ -n "${result_dir}" ] || {
    echo "evaluation exited successfully but produced incomplete results in ${dir}" >&2
    return 1
  }
  python3 - "${dir}/DONE" "${result_dir}" <<'PY'
import json
import sys
from pathlib import Path

Path(sys.argv[1]).write_text(
    json.dumps({"result_dir": sys.argv[2]}, indent=2) + "\n",
    encoding="utf-8",
)
PY
  printf '%s\n' "${result_dir}"
}

mkdir -p "${OUTPUT_ROOT}"
exec 9>"${OUTPUT_ROOT}/.lock"
flock -n 9 || { echo "another BoQ evaluation driver is using ${OUTPUT_ROOT}" >&2; exit 1; }

for dataset in "${DATASET_LIST[@]}"; do
  for index in "${!MODEL_TAGS[@]}"; do
    tag=${MODEL_TAGS[${index}]}
    path=${MODEL_PATHS[${index}]}
    dir=$(piece_dir "${dataset}" "${tag}")
    if result_dir=$(piece_result_dir "${dir}"); then
      echo "SKIP ${dataset}/${tag} (${result_dir})"
      continue
    fi
    if [ "${MODE}" = "summary" ]; then
      echo "PENDING ${dataset}/${tag} (${dir})"
      continue
    fi
    if [ "${MODE}" = "dry" ]; then
      echo "RUN ${dataset}/${tag} -> ${dir}"
      eval_command=(
        "${PYTHON}" eval.py
        --eval_datasets_folder=datasets
        --datasets "${dataset}"
        --dataset_split=test
        --model_type=boq
        --boq_backbone=Dinov2
        --boq_descriptors_dimension=12288
        --infer_batch_size="${INFER_BATCH_SIZE}"
        --num_workers="${NUM_WORKERS}"
        --grad_checkpointing
        --model_paths "${path}"
        --model_tags "${tag}"
        --test_method=hard_resize
        --rank_attack="${ATTACK}"
        --rank_steps="${STEPS}"
        --rank_restarts="${RESTARTS}"
        --rank_attack_goal="${GOAL}"
        --max_queries="${MAX_QUERIES}"
        --epsilons "${EPSILON_LIST[@]}"
        --output_json="${dir}/rank_eval_results.json"
        --output_csv="${dir}/rank_eval_results.csv"
      )
      printf '+ '
      printf '%q ' "${eval_command[@]}"
      printf '\n'
      continue
    fi

    mkdir -p "${dir}"
    echo "RUN ${dataset}/${tag} -> ${dir}"
    eval_command=(
      "${PYTHON}" eval.py
      --eval_datasets_folder=datasets
      --datasets "${dataset}"
      --dataset_split=test
      --model_type=boq
      --boq_backbone=Dinov2
      --boq_descriptors_dimension=12288
      --infer_batch_size="${INFER_BATCH_SIZE}"
      --num_workers="${NUM_WORKERS}"
      --grad_checkpointing
      --model_paths "${path}"
      --model_tags "${tag}"
      --test_method=hard_resize
      --rank_attack="${ATTACK}"
      --rank_steps="${STEPS}"
      --rank_restarts="${RESTARTS}"
      --rank_attack_goal="${GOAL}"
      --max_queries="${MAX_QUERIES}"
      --epsilons "${EPSILON_LIST[@]}"
      --output_json="${dir}/rank_eval_results.json"
      --output_csv="${dir}/rank_eval_results.csv"
    )
    "${eval_command[@]}"
    result_dir=$(record_piece "${dir}")
    echo "DONE ${dataset}/${tag} (${result_dir})"
  done
done
