#!/usr/bin/env bash
# Evaluate the MPLC v2 pretrained baseline plus finished clean_ft/mplc checkpoints
# (scripts/mplc_v2_train.sh) under the batch-independent rank-PGD threat model, one
# eval.py call per dataset.
#
# Usage:
#   scripts/mplc_v2_eval.sh [options]
#
# Options (each overrides the matching environment variable; lists are space-separated,
# so quote them):
#   --datasets "msls sped nordland"  datasets to evaluate on       (MPLC_V2_DATASETS)
#   --seeds "0 1"                    seeds to discover runs for     (MPLC_V2_SEEDS)
#   --arms "clean_ft mplc"           arms to discover runs for      (MPLC_V2_ARMS)
#   --no-pretrained                  do not evaluate checkpoints/SuperVLAD_base.pth
#   --model TAG=PATH                 evaluate PATH as TAG; repeatable. When given, only
#                                    these models run (no discovery, no pretrained)
#                                    unless --with-discovered is also passed
#   --with-discovered                add discovered runs (and pretrained) to --model ones
#   --epsilons "0.01712 0.03425 0.0685 0.137"  normalized L-inf budgets, <=1/2/4/8 per 255 (MPLC_V2_EPSILONS)
#   --output-root DIR                evaluation output root         (MPLC_V2_OUTPUT_ROOT)
#   --attack NAME                    rank_pgd_linf | rank_apgd_linf | rank_pgd_l2 (MPLC_V2_ATTACK)
#   --steps N / --restarts N         attack steps / restarts (20 / 1)  (MPLC_V2_STEPS/_RESTARTS)
#   --goal untargeted|targeted       query attack goal              (MPLC_V2_GOAL)
#   --checkpoint FILE                run checkpoint to evaluate, e.g. best_model_budget3.pth
#                                    (default best_model.pth)       (MPLC_V2_CHECKPOINT)
#   --shared-attacks                 craft attacks on the first model and replay them on the
#                                    others (transfer check)        (MPLC_V2_SHARED_ATTACKS=1)
#   --dry-run                        print the commands only        (MPLC_V2_DRY_RUN=1)
#   -h, --help                       show this help
#
# Other environment overrides:
#   MPLC_V2_MODELS       "tag=path tag=path ..." same as repeated --model
#   PYTHON               python interpreter                    (default: python)
#
# Examples:
#   scripts/mplc_v2_eval.sh --datasets sped --seeds 0
#   scripts/mplc_v2_eval.sh --datasets "sped nordland" --arms mplc --no-pretrained
#   scripts/mplc_v2_eval.sh --model mplc_tau0.01=logs/mplc_v2_supervlad_mplc_tau0.01_s0/<ts>/best_model.pth
#
# Only checkpoints whose run_status.json records a terminal state (completed,
# early_stopped, collapse_aborted) are auto-discovered; the newest such run wins.
# best_model.pth alone is not a reliable "finished" marker (it also exists right after
# the initial validation, i.e. the untrained/pretrained checkpoint, and in interrupted
# runs), so a still-training or crashed run is skipped with a WARNING rather than
# evaluated. A run trained with scripts/mplc_v2_train.sh overrides (MPLC_V2_TAU/_K/
# _POOL/_RAMP_EPOCHS/_ABORT_KNN) gets a save_dir with a distinct name and will not be
# auto-discovered here; pass it explicitly via MPLC_V2_MODELS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

PYTHON=${PYTHON:-python}
SEEDS=${MPLC_V2_SEEDS:-"0 1"}
ARMS=${MPLC_V2_ARMS:-"clean_ft mplc"}
DATASETS=${MPLC_V2_DATASETS:-"msls sped nordland"}
EPSILONS=${MPLC_V2_EPSILONS:-"0.01712 0.03425 0.0685 0.137"}
OUTPUT_ROOT=${MPLC_V2_OUTPUT_ROOT:-output/mplc_v2}
ATTACK=${MPLC_V2_ATTACK:-rank_pgd_linf}
STEPS=${MPLC_V2_STEPS:-20}
RESTARTS=${MPLC_V2_RESTARTS:-1}
GOAL=${MPLC_V2_GOAL:-untargeted}
CHECKPOINT=${MPLC_V2_CHECKPOINT:-best_model.pth}
SHARED_ATTACKS=${MPLC_V2_SHARED_ATTACKS:-0}
DRY_RUN=${MPLC_V2_DRY_RUN:-0}
EXPLICIT_MODELS=${MPLC_V2_MODELS:-}
INCLUDE_PRETRAINED=1
WITH_DISCOVERED=0

usage() {
  sed -n '2,/^$/{s/^# \{0,1\}//;p}' "${BASH_SOURCE[0]}"
}

while [ $# -gt 0 ]; do
  case "$1" in
    --datasets | --seeds | --arms | --model | --epsilons | --output-root | --attack | --steps | --restarts | --goal | --checkpoint)
      [ $# -ge 2 ] || { echo "$1 requires a value" >&2; exit 2; }
      case "$1" in
        --datasets) DATASETS=$2 ;;
        --seeds) SEEDS=$2 ;;
        --arms) ARMS=$2 ;;
        --model) EXPLICIT_MODELS="${EXPLICIT_MODELS:+${EXPLICIT_MODELS} }$2" ;;
        --epsilons) EPSILONS=$2 ;;
        --output-root) OUTPUT_ROOT=$2 ;;
        --attack) ATTACK=$2 ;;
        --steps) STEPS=$2 ;;
        --restarts) RESTARTS=$2 ;;
        --goal) GOAL=$2 ;;
        --checkpoint) CHECKPOINT=$2 ;;
      esac
      shift 2
      ;;
    --no-pretrained) INCLUDE_PRETRAINED=0; shift ;;
    --with-discovered) WITH_DISCOVERED=1; shift ;;
    --shared-attacks) SHARED_ATTACKS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h | --help) usage; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
done

for arm in ${ARMS}; do
  case "${arm}" in
    clean_ft | mplc) ;;
    *) echo "unknown arm: ${arm} (expected clean_ft or mplc)" >&2; exit 2 ;;
  esac
done
for dataset in ${DATASETS}; do
  [ -d "datasets/${dataset}/images/test" ] || echo "WARNING: datasets/${dataset}/images/test not found" >&2
done

case "${GOAL}" in
  untargeted | targeted) ;;
  *) echo "unknown goal: ${GOAL} (expected untargeted or targeted)" >&2; exit 2 ;;
esac

# Output dir: ${OUTPUT_ROOT}/supervlad_<dataset>[_<attack>][_<goal>][_<checkpoint>][_steps<N>][_r<N>][_shared],
# each part only when it differs from the default, so non-default runs never overwrite each other.
OUTPUT_SUFFIX=""
[ "${ATTACK}" = "rank_pgd_linf" ] || OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_${ATTACK}"
[ "${GOAL}" = "untargeted" ] || OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_${GOAL}"
[ "${CHECKPOINT}" = "best_model.pth" ] || OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_${CHECKPOINT%.pth}"
[ "${STEPS}" = "20" ] || OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_steps${STEPS}"
[ "${RESTARTS}" = "1" ] || OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_r${RESTARTS}"
EXTRA_EVAL_FLAGS=()
if [ "${SHARED_ATTACKS}" = "1" ]; then
  OUTPUT_SUFFIX="${OUTPUT_SUFFIX}_shared"
  EXTRA_EVAL_FLAGS+=(--shared_attacks)
fi

run() {
  echo "+ $*"
  if [ "${DRY_RUN}" != "1" ]; then
    "$@"
  fi
}

# A run is finished, and safe to evaluate, only when its run_status.json records a
# terminal state; the newest such run wins. best_model.pth alone is not a reliable
# finished marker (see header). Same state check as finished_run_dir in
# scripts/mplc_v2_train.sh.
latest_finished_run() {
  local name=$1 dir state newest=""
  for dir in "logs/${name}"/*/; do
    [ -f "${dir}run_status.json" ] || continue
    state="$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('state', ''))" "${dir}run_status.json")"
    case "${state}" in
      early_stopped | completed | collapse_aborted)
        newest="${dir%/}"
        ;;
    esac
  done
  echo "${newest}"
}

MODEL_PATHS=()
MODEL_TAGS=()

for entry in ${EXPLICIT_MODELS}; do
  case "${entry}" in
    *=*) ;;
    *) echo "model must be TAG=PATH, got: ${entry}" >&2; exit 2 ;;
  esac
  MODEL_TAGS+=("${entry%%=*}")
  MODEL_PATHS+=("${entry#*=}")
done

if [ -z "${EXPLICIT_MODELS}" ] || [ "${WITH_DISCOVERED}" = "1" ]; then
  if [ "${INCLUDE_PRETRAINED}" = "1" ] && [ -e checkpoints/SuperVLAD_base.pth ]; then
    MODEL_PATHS+=("checkpoints/SuperVLAD_base.pth")
    MODEL_TAGS+=("pretrained")
  fi

  for seed in ${SEEDS}; do
    for arm in ${ARMS}; do
      name="mplc_v2_supervlad_${arm}_s${seed}"
      run_dir="$(latest_finished_run "${name}")"
      if [ -n "${run_dir}" ]; then
        MODEL_PATHS+=("${run_dir}/${CHECKPOINT}")
        tag="${arm}_s${seed}"
        [ "${CHECKPOINT}" = "best_model.pth" ] || tag="${tag}_${CHECKPOINT%.pth}"
        MODEL_TAGS+=("${tag}")
      else
        echo "WARNING: no finished checkpoint for ${name}" >&2
      fi
    done
  done
fi

for path in "${MODEL_PATHS[@]+"${MODEL_PATHS[@]}"}"; do
  [ -f "${path}" ] || [ "${DRY_RUN}" = "1" ] || { echo "checkpoint not found: ${path}" >&2; exit 1; }
done
echo "Models: ${MODEL_TAGS[*]+"${MODEL_TAGS[*]}"} | datasets: ${DATASETS} | epsilons: ${EPSILONS}" >&2

[ "${#MODEL_PATHS[@]}" -gt 0 ] || { echo "No model resolves to evaluate." >&2; exit 1; }

for dataset in ${DATASETS}; do
  run "${PYTHON}" eval.py \
    --eval_datasets_folder=datasets --datasets "${dataset}" --dataset_split=test \
    "${SUPERVLAD_EVAL_MODEL_FLAGS[@]}" \
    --model_paths "${MODEL_PATHS[@]}" \
    --model_tags "${MODEL_TAGS[@]}" \
    --test_method=hard_resize \
    --rank_attack="${ATTACK}" --rank_steps="${STEPS}" --rank_restarts="${RESTARTS}" \
    --rank_attack_goal="${GOAL}" "${EXTRA_EVAL_FLAGS[@]+"${EXTRA_EVAL_FLAGS[@]}"}" \
    --epsilons ${EPSILONS} \
    --output_json="${OUTPUT_ROOT}/supervlad_${dataset}${OUTPUT_SUFFIX}/rank_eval_results.json" \
    --output_csv="${OUTPUT_ROOT}/supervlad_${dataset}${OUTPUT_SUFFIX}/rank_eval_results.csv"
done
