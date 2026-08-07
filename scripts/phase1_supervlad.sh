#!/usr/bin/env bash
# Phase 1 — Matched Controls and Seeds, SuperVLAD only.
#
# Removes the central confound in the current paper: the pretrained-vs-PAT comparison cannot
# tell whether the robustness gain came from *adversarial* training or from fine-tuning on
# GSV-Cities at all. This script runs the matched clean-only control alongside a re-run of
# the current PAT method across seeds {0, 1}, evaluates every checkpoint plus the pretrained
# base through the Phase 0 corrected pipeline, and emits the three-way comparison table.
#
#   Task 1.1  matched clean-only configuration (see scripts/lib/supervlad_common.sh)
#   Task 1.2  training-run matrix over seeds, evaluation, three-way table
#   Task 1.3  checkpoint-selection rule, recorded per run in training_config.yaml
#
# PHASE 1 IS A DECISION GATE. Review the resulting table with the supervisor before
# committing full compute to Phase 2's run matrix: if clean-FT reproduces part of the
# "robustness" gain, Phase 2's claims must be scoped accordingly.
#
# Usage:
#   scripts/phase1_supervlad.sh                 # all stages: train -> eval -> table
#   scripts/phase1_supervlad.sh train
#   scripts/phase1_supervlad.sh eval
#   scripts/phase1_supervlad.sh table
#
# Environment overrides:
#   PHASE1_SEEDS        seeds to train (default: "0 1")
#   PHASE1_ARMS         arms to train (default: "clean_ft pat")
#   PHASE1_DATASETS     evaluation datasets (default: "msls sped")
#   PHASE1_EPSILONS     attack budgets (default: "0.01 0.1")
#   PHASE1_OUTPUT_ROOT  evaluation output root (default: output/phase1)
#   PHASE1_BASE_PATH    pretrained SuperVLAD reference checkpoint
#   PHASE1_DRY_RUN=1    print the commands without running them
#   PYTHON              python interpreter (default: venv/bin/python3)
#
# Cost: 4 training runs of up to 50 epochs each, plus an evaluation grid of
# 5 checkpoints x 2 datasets x 2 epsilons. Both stages are restartable — a training run
# whose log directory already holds best_model.pth is skipped.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

PYTHON=${PYTHON:-venv/bin/python3}
SEEDS=${PHASE1_SEEDS:-"0 1"}
ARMS=${PHASE1_ARMS:-"clean_ft pat"}
DATASETS=${PHASE1_DATASETS:-"msls sped"}
EPSILONS=${PHASE1_EPSILONS:-"0.01 0.1"}
OUTPUT_ROOT=${PHASE1_OUTPUT_ROOT:-output/phase1}
DRY_RUN=${PHASE1_DRY_RUN:-0}

# The SuperVLAD runs behind the paper start from the DINOv2 foundation weights with a fresh
# VLAD head (the recovered metadata has resume=null), so the "pretrained" reference arm is
# the published SuperVLAD release rather than a checkpoint this project produced.
BASE_PATH=${PHASE1_BASE_PATH:-checkpoints/SuperVLAD_base.pth}

STAGE=${1:-all}

run() {
  echo "+ $*"
  if [ "${DRY_RUN}" != "1" ]; then
    "$@"
  fi
}

save_dir_name() {
  echo "phase1_supervlad_${1}_s${2}"
}

# train.py appends a timestamp to --save_dir, so the run directory is only knowable after
# the fact. Take the newest one that actually finished.
latest_finished_run() {
  local name=$1
  local newest=""
  local candidate
  for candidate in "logs/${name}"/*/; do
    [ -f "${candidate}best_model.pth" ] && newest="${candidate%/}"
  done
  echo "${newest}"
}

train_arm() {
  local arm=$1 seed=$2
  local name
  name="$(save_dir_name "${arm}" "${seed}")"

  local existing
  existing="$(latest_finished_run "${name}")"
  if [ -n "${existing}" ]; then
    echo "=== SKIP ${arm} seed ${seed}: already finished at ${existing}"
    return 0
  fi

  local -a attack_flags=()
  if [ "${arm}" = "pat" ]; then
    attack_flags=("${SUPERVLAD_ATTACK_FLAGS[@]}")
  fi

  echo "=== TRAIN ${arm} seed ${seed} -> logs/${name}/<timestamp>"
  run "${PYTHON}" train.py \
    "${SUPERVLAD_BASE_FLAGS[@]}" \
    "${attack_flags[@]}" \
    --seed="${seed}" \
    --save_dir="${name}"
}

stage_train() {
  for arm in ${ARMS}; do
    case "${arm}" in
      clean_ft | pat) ;;
      *)
        echo "unknown arm: ${arm} (expected clean_ft or pat)" >&2
        return 1
        ;;
    esac
    for seed in ${SEEDS}; do
      train_arm "${arm}" "${seed}"
    done
  done
}

# Collect the checkpoint paths and tags for every arm that finished training, prefixed by
# the pretrained reference. Tags follow the <arm>_s<seed> convention src/phase1_summary.py
# parses.
collect_models() {
  MODEL_PATHS=()
  MODEL_TAGS=()

  if [ -e "${BASE_PATH}" ]; then
    MODEL_PATHS+=("${BASE_PATH}")
    MODEL_TAGS+=("pretrained")
  else
    echo "WARNING: pretrained reference ${BASE_PATH} is missing; the table will have no deltas." >&2
  fi

  for arm in ${ARMS}; do
    for seed in ${SEEDS}; do
      local name run_dir
      name="$(save_dir_name "${arm}" "${seed}")"
      run_dir="$(latest_finished_run "${name}")"
      if [ -z "${run_dir}" ]; then
        echo "WARNING: no finished run for ${arm} seed ${seed}; skipping it in the evaluation." >&2
        continue
      fi
      MODEL_PATHS+=("${run_dir}/best_model.pth")
      MODEL_TAGS+=("${arm}_s${seed}")
    done
  done
}

stage_eval() {
  collect_models
  if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "Nothing to evaluate: no checkpoints found. Run the train stage first." >&2
    return 1
  fi

  for dataset in ${DATASETS}; do
    local run_dir="${OUTPUT_ROOT}/supervlad_${dataset}"
    echo "=== EVAL ${dataset} / eps ${EPSILONS} / ${#MODEL_PATHS[@]} checkpoints -> ${run_dir}"
    # shellcheck disable=SC2086
    run "${PYTHON}" eval.py \
      --eval_datasets_folder=datasets \
      --datasets "${dataset}" \
      "${SUPERVLAD_EVAL_MODEL_FLAGS[@]}" \
      --model_paths "${MODEL_PATHS[@]}" \
      --model_tags "${MODEL_TAGS[@]}" \
      --test_method=hard_resize \
      --rank_attack=rank_pgd_linf \
      --rank_steps=20 \
      --rank_restarts=1 \
      --epsilons ${EPSILONS} \
      --output_json="${run_dir}/rank_eval_results.json" \
      --output_csv="${run_dir}/rank_eval_results.csv"
  done
}

# Newest timestamped result file under a given eval run directory.
latest_results_json() {
  local root=$1
  local newest=""
  local candidate
  for candidate in "${root}"/*/rank_eval_results.json; do
    [ -f "${candidate}" ] && newest="${candidate}"
  done
  echo "${newest}"
}

stage_table() {
  local -a results=()
  for dataset in ${DATASETS}; do
    local found
    found="$(latest_results_json "${OUTPUT_ROOT}/supervlad_${dataset}")"
    if [ -n "${found}" ]; then
      results+=("${found}")
    else
      echo "WARNING: no evaluation results for ${dataset}." >&2
    fi
  done

  if [ ${#results[@]} -eq 0 ]; then
    echo "Nothing to summarize: run the eval stage first." >&2
    return 1
  fi

  echo "=== THREE-WAY TABLE (pretrained vs clean-FT vs PAT, mean ± range over seeds)"
  run "${PYTHON}" -m src.phase1_summary \
    --results_json "${results[@]}" \
    --output_markdown "${OUTPUT_ROOT}/phase1_three_way_table.md" \
    --output_csv "${OUTPUT_ROOT}/phase1_three_way_table.csv"

  # Intersection CC-ASR is defined over queries both models retrieve correctly when clean,
  # so it needs the per-query CSVs and a base/trained pair rather than either summary alone.
  echo
  echo "Intersection CC-ASR (feedback lines 22-27) — run per arm against the pretrained base:"
  for dataset in ${DATASETS}; do
    local json_path per_query
    json_path="$(latest_results_json "${OUTPUT_ROOT}/supervlad_${dataset}")"
    [ -z "${json_path}" ] && continue
    per_query="$(dirname "${json_path}")/per_query_ranks.csv"
    for arm in ${ARMS}; do
      for seed in ${SEEDS}; do
        echo "  ${PYTHON} -m src.paired_analysis --base_csv ${per_query} \\"
        echo "      --base_model_tag pretrained --trained_model_tag ${arm}_s${seed}"
      done
    done
  done
}

case "${STAGE}" in
  train) stage_train ;;
  eval) stage_eval ;;
  table) stage_table ;;
  all)
    stage_train
    stage_eval
    stage_table
    ;;
  *)
    echo "usage: $0 [train|eval|table|all]" >&2
    exit 2
    ;;
esac

echo
echo "Phase 1 stage '${STAGE}' complete."
echo "Decision gate: review ${OUTPUT_ROOT}/phase1_three_way_table.md with the supervisor"
echo "before committing compute to Phase 2 (scripts/phase2_supervlad.sh)."
