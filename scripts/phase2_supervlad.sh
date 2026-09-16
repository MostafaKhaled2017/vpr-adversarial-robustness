#!/usr/bin/env bash
# Phase 2 — Multi-Positive, Listwise, Collapse-Aware Adversarial Training. SuperVLAD only.
#
# This is the contribution that carries the paper. Every component is a separately
# switchable flag, so Phase 4's ablations fall out of the same launcher:
#
#   Task 2.1  --multi_positive        target every positive, not just the hardest
#   Task 2.2  --negative_pool_size    cross-batch FIFO negative pool
#   Task 2.3  --defense_loss=listwise smooth top-K recall surrogate (+ --listwise_tau/_k)
#   Task 2.4  --collapse_abort_knn    collapse / neighborhood-distortion monitoring
#   Task 2.5  --attack_ramp_epochs    attack-hardness curriculum
#   Task 2.6  pilot grid -> select -> final seeds -> evaluate
#
# REQUIRES PHASE 1. The Task 2.6 acceptance criterion is stated relative to Phase 1's arms
# (clean R@1 within PHASE2_MAX_CLEAN_DROP of clean-FT, attacked R@1 at least PAT's), so the
# 'select' stage needs those two baselines. Pass them explicitly, or let the script read
# them from output/phase1/phase1_three_way_table.csv. The code stages (pilot training) will
# run without Phase 1, but you cannot judge the result.
#
# Usage:
#   scripts/phase2_supervlad.sh pilot [--run-root DIR]  # new grid, or resume DIR
#   scripts/phase2_supervlad.sh select --run-root DIR   # rank this pilot against Phase 1
#   scripts/phase2_supervlad.sh final [--run-root DIR]  # train selected config on seeds {0, 1}
#   scripts/phase2_supervlad.sh eval      # evaluate final checkpoints + Phase 1 arms
#   scripts/phase2_supervlad.sh all
#
# Environment overrides:
#   PHASE2_TAUS               listwise temperatures  (default: "0.01 0.05")
#   PHASE2_KS                 listwise top-K targets (default: "1 5")
#   PHASE2_POOLS              negative pool sizes    (default: "0 4096")
#   PHASE2_SEEDS              final-run seeds        (default: "0 1")
#   PHASE2_RAMP_EPOCHS        attack-hardness ramp   (default: 5)
#   PHASE2_ABORT_KNN          collapse abort threshold (default: 0.15)
#   PHASE2_WINNER             skip 'select' and force a config label
#   PHASE2_CLEAN_FT_CLEAN_R1  Phase 1 clean-FT clean R@1 baseline
#   PHASE2_PAT_ATTACKED_R1    Phase 1 PAT attacked R@1 baseline
#   PHASE2_MAX_CLEAN_DROP     acceptable clean R@1 loss (default: 2.0)
#   PHASE2_DATASETS           evaluation datasets (default: "msls sped")
#   PHASE2_EPSILONS           attack budgets (default: "0.01 0.1")
#   PHASE2_OUTPUT_ROOT        evaluation output root (default: output/phase2)
#   PHASE2_PILOT_ROOT_BASE    parent for managed pilot roots (default: logs/phase2_supervlad_pilots)
#   PHASE2_DRY_RUN=1          print the commands without running them
#
# Cost: 8 pilot runs + 2 final runs, plus an evaluation grid. Managed pilot runs resume at
# epoch boundaries and use run_status.json—not best_model.pth—as the terminal marker.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

PYTHON=${PYTHON:-python}
TAUS_EXPLICIT=${PHASE2_TAUS+x}
KS_EXPLICIT=${PHASE2_KS+x}
POOLS_EXPLICIT=${PHASE2_POOLS+x}
RAMP_EPOCHS_EXPLICIT=${PHASE2_RAMP_EPOCHS+x}
ABORT_KNN_EXPLICIT=${PHASE2_ABORT_KNN+x}
BASE_PATH_EXPLICIT=${PHASE1_BASE_PATH+x}${SUPERVLAD_BASE_CHECKPOINT+x}
FROM_SCRATCH_EXPLICIT=${SUPERVLAD_FROM_SCRATCH+x}
TAUS=${PHASE2_TAUS:-"0.01 0.05"}
KS=${PHASE2_KS:-"1 5"}
POOLS=${PHASE2_POOLS:-"0 4096"}
SEEDS=${PHASE2_SEEDS:-"0 1"}
RAMP_EPOCHS=${PHASE2_RAMP_EPOCHS:-5}
ABORT_KNN=${PHASE2_ABORT_KNN:-0.15}
MAX_CLEAN_DROP=${PHASE2_MAX_CLEAN_DROP:-2.0}
DATASETS=${PHASE2_DATASETS:-"msls sped"}
EPSILONS=${PHASE2_EPSILONS:-"0.01 0.1"}
OUTPUT_ROOT=${PHASE2_OUTPUT_ROOT:-output/phase2}
PHASE1_OUTPUT_ROOT=${PHASE1_OUTPUT_ROOT:-output/phase1}
DRY_RUN=${PHASE2_DRY_RUN:-0}
BASE_PATH=${PHASE1_BASE_PATH:-${SUPERVLAD_BASE_CHECKPOINT:-checkpoints/SuperVLAD_base.pth}}
FROM_SCRATCH=${SUPERVLAD_FROM_SCRATCH:-0}
PILOT_ROOT_BASE=${PHASE2_PILOT_ROOT_BASE:-logs/phase2_supervlad_pilots}

WINNER_FILE="${OUTPUT_ROOT}/pilot_winner.txt"
STAGE=${1:-all}
if [ $# -gt 0 ]; then
  shift
fi
RUN_ROOT=""
while [ $# -gt 0 ]; do
  case "$1" in
    --run-root)
      [ $# -ge 2 ] || { echo "--run-root requires a directory" >&2; exit 2; }
      RUN_ROOT=$2
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

run() {
  echo "+ $*"
  if [ "${DRY_RUN}" != "1" ]; then
    "$@"
  fi
}

# Everything the new method turns on that is *not* part of the pilot sweep. Held fixed
# across the grid so the sweep isolates the three hyper-parameters it is varying.
method_flags() {
  local tau=$1 k=$2 pool=$3
  echo "--multi_positive" \
    "--defense_loss=listwise" \
    "--listwise_tau=${tau}" \
    "--listwise_k=${k}" \
    "--negative_pool_size=${pool}" \
    "--attack_ramp_epochs=${RAMP_EPOCHS}" \
    "--collapse_abort_knn=${ABORT_KNN}"
}

config_label() {
  echo "tau${1}_k${2}_pool${3}"
}

ensure_pilot_root() {
  if [ -n "${RUN_ROOT}" ]; then
    local manifest_values manifest_taus manifest_ks manifest_pools manifest_seed manifest_ramp manifest_abort manifest_base manifest_from_scratch
    manifest_values="$("${PYTHON}" -m src.phase2_run_group values --run-root "${RUN_ROOT}")"
    IFS=$'\t' read -r manifest_taus manifest_ks manifest_pools manifest_seed manifest_ramp manifest_abort manifest_base manifest_from_scratch <<<"${manifest_values}"
    if [ -n "${TAUS_EXPLICIT}" ] && [ "${TAUS}" != "${manifest_taus}" ]; then
      echo "PHASE2_TAUS does not match ${RUN_ROOT}/pilot_manifest.yaml" >&2
      return 2
    fi
    if [ -n "${KS_EXPLICIT}" ] && [ "${KS}" != "${manifest_ks}" ]; then
      echo "PHASE2_KS does not match ${RUN_ROOT}/pilot_manifest.yaml" >&2
      return 2
    fi
    if [ -n "${POOLS_EXPLICIT}" ] && [ "${POOLS}" != "${manifest_pools}" ]; then
      echo "PHASE2_POOLS does not match ${RUN_ROOT}/pilot_manifest.yaml" >&2
      return 2
    fi
    if [ -n "${RAMP_EPOCHS_EXPLICIT}" ] && [ "${RAMP_EPOCHS}" != "${manifest_ramp}" ]; then
      echo "PHASE2_RAMP_EPOCHS does not match ${RUN_ROOT}/pilot_manifest.yaml" >&2
      return 2
    fi
    if [ -n "${ABORT_KNN_EXPLICIT}" ] && [ "${ABORT_KNN}" != "${manifest_abort}" ]; then
      echo "PHASE2_ABORT_KNN does not match ${RUN_ROOT}/pilot_manifest.yaml" >&2
      return 2
    fi
    if [ -n "${BASE_PATH_EXPLICIT}" ] && [ "${BASE_PATH}" != "${manifest_base}" ]; then
      echo "Base-checkpoint override does not match ${RUN_ROOT}/pilot_manifest.yaml" >&2
      return 2
    fi
    if [ -n "${FROM_SCRATCH_EXPLICIT}" ] && [ "${FROM_SCRATCH}" != "${manifest_from_scratch}" ]; then
      echo "SUPERVLAD_FROM_SCRATCH does not match ${RUN_ROOT}/pilot_manifest.yaml" >&2
      return 2
    fi
    TAUS=${manifest_taus}
    KS=${manifest_ks}
    POOLS=${manifest_pools}
    RAMP_EPOCHS=${manifest_ramp}
    ABORT_KNN=${manifest_abort}
    BASE_PATH=${manifest_base}
    FROM_SCRATCH=${manifest_from_scratch}
    SUPERVLAD_FROM_SCRATCH=${manifest_from_scratch}
  else
    local -a create_args=(
      --base-dir "${PILOT_ROOT_BASE}"
      --taus ${TAUS} --ks ${KS} --pools ${POOLS}
      --seed 0 --ramp-epochs "${RAMP_EPOCHS}" --abort-knn "${ABORT_KNN}"
      --base-checkpoint "${BASE_PATH}"
    )
    if [ "${FROM_SCRATCH}" = "1" ]; then
      create_args+=(--from-scratch)
    fi
    # shellcheck disable=SC2086
    RUN_ROOT="$("${PYTHON}" -m src.phase2_run_group create "${create_args[@]}")"
  fi
  echo "Pilot run root: ${RUN_ROOT}"
}

acquire_pilot_lock() {
  exec 9>"${RUN_ROOT}/.pilot.lock"
  if ! flock -n 9; then
    echo "Pilot run root is already active: ${RUN_ROOT}" >&2
    return 1
  fi
}

latest_finished_run() {
  local name=$1
  local newest="" candidate
  for candidate in "logs/${name}"/*/; do
    [ -f "${candidate}best_model.pth" ] && newest="${candidate%/}"
  done
  echo "${newest}"
}

train_config() {
  local name=$1 seed=$2 tau=$3 k=$4 pool=$5 managed_dir=${6:-}

  if [ -n "${managed_dir}" ]; then
    local classification state checkpoint
    classification="$("${PYTHON}" -m src.phase2_run_group classify --config-dir "${managed_dir}")"
    IFS=$'\t' read -r state checkpoint <<<"${classification}"
    case "${state}" in
      completed | early_stopped | collapse_aborted)
        echo "=== SKIP ${name}: ${state} at ${managed_dir}"
        return 0
        ;;
      invalid)
        echo "Cannot resume ${name}: artifacts exist but no valid checkpoint was found in ${managed_dir}" >&2
        return 1
        ;;
    esac

    local -a launch_flags
    if [ "${state}" = "resumable" ]; then
      launch_flags=("${SUPERVLAD_RECIPE_FLAGS[@]}" --resume="${checkpoint}" --continue)
      echo "=== RESUME ${name} from ${checkpoint}"
    else
      launch_flags=("${SUPERVLAD_RECIPE_FLAGS[@]}")
      if [ "${FROM_SCRATCH}" != "1" ]; then
        launch_flags+=(--resume="${BASE_PATH}" --resume_model_only)
      fi
      echo "=== TRAIN ${name} (tau=${tau}, k=${k}, pool=${pool}, seed=${seed})"
    fi

    # shellcheck disable=SC2046
    run "${PYTHON}" train.py \
      "${launch_flags[@]}" \
      "${SUPERVLAD_ATTACK_FLAGS[@]}" \
      $(method_flags "${tau}" "${k}" "${pool}") \
      --seed="${seed}" \
      --save_dir="${name}" \
      --run_dir="${managed_dir}"
    return
  fi

  local existing
  existing="$(latest_finished_run "${name}")"
  if [ -n "${existing}" ]; then
    echo "=== SKIP ${name}: already finished at ${existing}"
    return 0
  fi

  echo "=== TRAIN ${name} (tau=${tau}, k=${k}, pool=${pool}, seed=${seed})"
  local -a standalone_flags=("${SUPERVLAD_RECIPE_FLAGS[@]}")
  if [ "${FROM_SCRATCH}" != "1" ]; then
    standalone_flags+=(--resume="${BASE_PATH}" --resume_model_only)
  fi
  # shellcheck disable=SC2046
  run "${PYTHON}" train.py \
    "${standalone_flags[@]}" \
    "${SUPERVLAD_ATTACK_FLAGS[@]}" \
    $(method_flags "${tau}" "${k}" "${pool}") \
    --seed="${seed}" \
    --save_dir="${name}"
}

stage_pilot() {
  ensure_pilot_root
  acquire_pilot_lock
  echo "Pilot grid on SuperVLAD seed 0: $(echo ${TAUS} | wc -w) taus x $(echo ${KS} | wc -w) ks x $(echo ${POOLS} | wc -w) pools"
  for tau in ${TAUS}; do
    for k in ${KS}; do
      for pool in ${POOLS}; do
        local label
        label="$(config_label "${tau}" "${k}" "${pool}")"
        train_config "${label}" 0 "${tau}" "${k}" "${pool}" "${RUN_ROOT}/${label}"
      done
    done
  done
}

# Read the Phase 1 baselines out of the three-way table when they were not passed in.
resolve_phase1_baselines() {
  CLEAN_FT_CLEAN_R1=${PHASE2_CLEAN_FT_CLEAN_R1:-}
  PAT_ATTACKED_R1=${PHASE2_PAT_ATTACKED_R1:-}

  local table="${PHASE1_OUTPUT_ROOT}/phase1_three_way_table.csv"
  if { [ -z "${CLEAN_FT_CLEAN_R1}" ] || [ -z "${PAT_ATTACKED_R1}" ]; } && [ -f "${table}" ]; then
    local parsed
    parsed="$("${PYTHON}" - "${table}" <<'PYTHON'
import csv, sys
from collections import defaultdict

clean_ft, pat = [], []
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row["arm"] == "clean_ft" and row.get("clean_r1_mean"):
            clean_ft.append(float(row["clean_r1_mean"]))
        if row["arm"] == "pat" and row.get("attacked_r1_mean"):
            pat.append(float(row["attacked_r1_mean"]))

# Average across datasets and epsilons: a single scalar gate for the pilot sweep. The
# reported comparison is always per-condition; this is only for picking a winner.
print(sum(clean_ft) / len(clean_ft) if clean_ft else "", sum(pat) / len(pat) if pat else "")
PYTHON
)"
    [ -z "${CLEAN_FT_CLEAN_R1}" ] && CLEAN_FT_CLEAN_R1="$(echo "${parsed}" | cut -d' ' -f1)"
    [ -z "${PAT_ATTACKED_R1}" ] && PAT_ATTACKED_R1="$(echo "${parsed}" | cut -d' ' -f2)"
  fi

  if [ -z "${CLEAN_FT_CLEAN_R1}" ] || [ -z "${PAT_ATTACKED_R1}" ]; then
    echo "WARNING: Phase 1 baselines unavailable — acceptance will be reported as unknown." >&2
    echo "         Run scripts/phase1_supervlad.sh, or set PHASE2_CLEAN_FT_CLEAN_R1 and" >&2
    echo "         PHASE2_PAT_ATTACKED_R1 explicitly." >&2
  fi
}

stage_select() {
  if [ -z "${RUN_ROOT}" ]; then
    echo "select requires --run-root so pilot attempts cannot be mixed" >&2
    return 2
  fi
  ensure_pilot_root
  acquire_pilot_lock
  resolve_phase1_baselines

  local -a run_args=()
  for tau in ${TAUS}; do
    for k in ${KS}; do
      for pool in ${POOLS}; do
        local label run_dir classification state checkpoint
        label="$(config_label "${tau}" "${k}" "${pool}")"
        run_dir="${RUN_ROOT}/${label}"
        classification="$("${PYTHON}" -m src.phase2_run_group classify --config-dir "${run_dir}")"
        IFS=$'\t' read -r state checkpoint <<<"${classification}"
        case "${state}" in
          completed | early_stopped)
            run_args+=(--run "${label}=${run_dir}")
            ;;
          collapse_aborted)
            echo "=== EXCLUDE ${label}: collapse_aborted at ${run_dir}"
            ;;
          *)
            echo "Cannot select: ${label} is not terminal (${state})" >&2
            return 1
            ;;
        esac
      done
    done
  done

  if [ ${#run_args[@]} -eq 0 ]; then
    echo "No finished pilot runs. Run the pilot stage first." >&2
    return 1
  fi

  local -a baseline_args=()
  [ -n "${CLEAN_FT_CLEAN_R1}" ] && baseline_args+=(--clean_ft_clean_r1 "${CLEAN_FT_CLEAN_R1}")
  [ -n "${PAT_ATTACKED_R1}" ] && baseline_args+=(--pat_attacked_r1 "${PAT_ATTACKED_R1}")

  echo "=== PILOT RANKING (criterion: clean R@1 within ${MAX_CLEAN_DROP} of clean-FT, attacked R@1 >= PAT)"
  local ranking_markdown="${RUN_ROOT}/pilot_ranking.md"
  local ranking_json="${RUN_ROOT}/pilot_ranking.json"
  WINNER_FILE="${RUN_ROOT}/pilot_winner.txt"
  run "${PYTHON}" -m src.phase2_pilot_summary \
    "${run_args[@]}" \
    "${baseline_args[@]}" \
    --max_clean_drop "${MAX_CLEAN_DROP}" \
    --output_markdown "${ranking_markdown}" \
    --output_json "${ranking_json}"

  if [ "${DRY_RUN}" = "1" ]; then
    return 0
  fi

  "${PYTHON}" - "${ranking_json}" "${WINNER_FILE}" <<'PYTHON'
import json, pathlib, sys

rows = json.loads(pathlib.Path(sys.argv[1]).read_text())
if not rows:
    raise SystemExit("no pilot rows to select from")

best = rows[0]
if best.get("accepted") is False:
    print("No pilot configuration met the Task 2.6 criterion.")
    print("Per the plan: stop and reassess with the supervisor before the full run matrix.")
    raise SystemExit(3)

pathlib.Path(sys.argv[2]).write_text(best["label"] + "\n")
print(f"Selected {best['label']} -> {sys.argv[2]}")
PYTHON
}

resolve_winner() {
  WINNER=${PHASE2_WINNER:-}
  if [ -z "${WINNER}" ] && [ -f "${WINNER_FILE}" ]; then
    WINNER="$(tr -d '[:space:]' <"${WINNER_FILE}")"
  fi
  if [ -z "${WINNER}" ]; then
    echo "No winning configuration. Run the select stage, or set PHASE2_WINNER." >&2
    return 1
  fi

  # Labels are tau<TAU>_k<K>_pool<POOL>.
  WINNER_TAU="${WINNER#tau}"; WINNER_TAU="${WINNER_TAU%%_*}"
  WINNER_K="${WINNER#*_k}"; WINNER_K="${WINNER_K%%_*}"
  WINNER_POOL="${WINNER##*_pool}"
  echo "Winning configuration: ${WINNER} (tau=${WINNER_TAU}, k=${WINNER_K}, pool=${WINNER_POOL})"
}

stage_final() {
  if [ -n "${RUN_ROOT}" ]; then
    # Validate the root and reload the settings that produced its selected winner.
    ensure_pilot_root
    WINNER_FILE="${RUN_ROOT}/pilot_winner.txt"
  fi
  resolve_winner
  for seed in ${SEEDS}; do
    train_config "phase2_supervlad_mplc_s${seed}" "${seed}" "${WINNER_TAU}" "${WINNER_K}" "${WINNER_POOL}"
  done
}

# Evaluate the new method beside Phase 1's arms so the comparison is like-for-like: same
# datasets, same epsilons, same attack protocol, one eval run.
collect_models() {
  MODEL_PATHS=()
  MODEL_TAGS=()

  if [ -e "${BASE_PATH}" ]; then
    MODEL_PATHS+=("${BASE_PATH}")
    MODEL_TAGS+=("pretrained")
  fi

  local arm seed name run_dir
  for arm in clean_ft pat; do
    for seed in ${SEEDS}; do
      run_dir="$(latest_finished_run "phase1_supervlad_${arm}_s${seed}")"
      if [ -n "${run_dir}" ]; then
        MODEL_PATHS+=("${run_dir}/best_model.pth")
        MODEL_TAGS+=("${arm}_s${seed}")
      else
        echo "WARNING: Phase 1 ${arm} seed ${seed} is missing from the comparison." >&2
      fi
    done
  done

  for seed in ${SEEDS}; do
    run_dir="$(latest_finished_run "phase2_supervlad_mplc_s${seed}")"
    if [ -n "${run_dir}" ]; then
      MODEL_PATHS+=("${run_dir}/best_model.pth")
      MODEL_TAGS+=("mplc_s${seed}")
    else
      echo "WARNING: no finished Phase 2 run for seed ${seed}." >&2
    fi
  done
}

stage_eval() {
  collect_models
  if [ ${#MODEL_PATHS[@]} -eq 0 ]; then
    echo "Nothing to evaluate. Run the final stage first." >&2
    return 1
  fi

  local -a results=()
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

    local newest="" candidate
    for candidate in "${run_dir}"/*/rank_eval_results.json; do
      [ -f "${candidate}" ] && newest="${candidate}"
    done
    [ -n "${newest}" ] && results+=("${newest}")
  done

  if [ ${#results[@]} -gt 0 ]; then
    echo "=== COMPARISON TABLE (pretrained vs clean-FT vs PAT vs the new method)"
    run "${PYTHON}" -m src.phase1_summary \
      --results_json "${results[@]}" \
      --output_markdown "${OUTPUT_ROOT}/phase2_comparison_table.md" \
      --output_csv "${OUTPUT_ROOT}/phase2_comparison_table.csv"
  fi
}

case "${STAGE}" in
  pilot) stage_pilot ;;
  select) stage_select ;;
  final) stage_final ;;
  eval) stage_eval ;;
  all)
    stage_pilot
    stage_select
    stage_final
    stage_eval
    ;;
  *)
    echo "usage: $0 [pilot|select|final|eval|all] [--run-root DIR]" >&2
    exit 2
    ;;
esac

echo
echo "Phase 2 stage '${STAGE}' complete."
echo "Headline acceptance: the new method must Pareto-dominate or clearly improve on both"
echo "PAT and clean-FT in the clean-vs-attacked trade-off. See ${OUTPUT_ROOT}/phase2_comparison_table.md."
