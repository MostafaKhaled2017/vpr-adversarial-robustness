#!/usr/bin/env bash
# Step 4 (runbook §4): ARAFT and its baselines on a second architecture, one model per call
# (STEP4_MODEL=boq, mixvpr, dinomix or qaa). `train` trains the runs, `eval` evaluates the trained ones.
#
# train: runs (src/step4_plan.py), each one arm of scripts/mplc_v2_train.sh at the follow-up
# runs' settings (validate every 3 epochs, patience 6, lr-plateau patience 3, up to 100 epochs,
# lr 1e-5; freeze_te 0 for BoQ, DINOMix and QAA, the whole network for MixVPR). Default keys, in order:
#   araft     ARAFT (single positive, rank L-inf only)  mplc_v2_<model>_mplc_mixlinf_sp[_fte0]_s0
#   plain_at  plain_at baseline                          mplc_v2_<model>_plain_at[_fte0]_s0
#   fare      FARE baseline, lambda_anchor 1.0           mplc_v2_<model>_fare[_fte0]_s0
# Optional keys (STEP4_RUNS): twin (clean twin), fare_eps2 (FARE at 2/255), screen_lr1e-5 and screen_lr3e-6
# (9-epoch ARAFT lr screens). Each run lives in STEP4_ROOT/<name>: stop at any point and rerun
# to resume; finished runs are skipped and a stopped run continues from its last epoch.
#
# eval: per trained row, the follow-up evaluation's checkpoints (Table A budget-5, Table B most
# robust; one file when the same epoch), on 1000 queries per dataset (SPED: all 607):
#   r1_apgd      rank-APGD-100 at 2/255 and 4/255 (headline)   pretrained, araft, plain_at, fare[, fare_eps2]
#   r2_pgd_grid  rank-PGD-20 over the eps grid                 the same models
#   twin_clean   the twin's clean recall, once it is trained (rank-PGD-20 at 1/255 rides along)
# A piece is one (run, model, dataset) eval.py call writing to
# STEP4_EVAL_ROOT/<model>/<run>/<tag>/; it is marked <dataset>.done once eval.py exits 0, so
# rerun to resume. A row whose run is not trained yet is skipped; rerun after it is.
#
# Sessions: a real run needs STEP4_STOP_AT, the time by which every process must be gone (the
# machine's clock). Same rules as STEP2_STOP_AT (scripts/lib/session.sh): Ctrl-C to every
# process 10 min before it, SIGKILL 5 min before. Before that, train.py pauses cleanly rather
# than start an epoch that could end after STEP4_STOP_AT - 15 min, and no new run or
# evaluation piece starts with less than 45 min left before then.
#
# Usage:
#   STEP4_MODEL=boq STEP4_STOP_AT="2026-10-25 08:30" scripts/mplc_v2_step4.sh train   # train or resume
#   STEP4_MODEL=boq STEP4_STOP_AT="2026-10-25 08:30" scripts/mplc_v2_step4.sh eval    # evaluate or resume
#   STEP4_MODEL=boq scripts/mplc_v2_step4.sh train --dry-run       # print the pending runs' commands only
#   STEP4_MODEL=boq scripts/mplc_v2_step4.sh train --summary-only  # rebuild the tables, train nothing
#   STEP4_MODEL=boq scripts/mplc_v2_step4.sh eval --dry-run        # print the pending pieces' commands only
#
# Environment overrides:
#   STEP4_MODEL          boq | mixvpr | dinomix | qaa (required)
#   STEP4_STOP_AT        end of this session, any `date -d` time (required for a real run)
#   STEP4_ROOT           training runs                       (default: logs/mplc_v2_step4)
#   STEP4_RUNS           run keys to train, in this order    (default: araft plain_at fare)
#   STEP4_CLEAN_BUDGETS  clean budgets of the reporting checkpoints, points; keep 5, the
#                        tables' budget                      (default: "1 3 5 7 10")
#   STEP4_PIPE_LOADER    1 = data-loader batches through a pipe, not /dev/shm (default: 0)
#   STEP4_EVAL_ROOT      evaluation outputs                  (default: output/mplc_v2_step4)
#   STEP4_EVAL_DATASETS  datasets to evaluate on             (default: "msls pitts30k sped nordland")
#   BOQ_BASE_CHECKPOINT / MIXVPR_BASE_CHECKPOINT / DINOMIX_BASE_CHECKPOINT / QAA_BASE_CHECKPOINT
#                        pretrained releases (lib/supervlad_common.sh)
#   PYTHON               python interpreter                  (default: python)
# Other MPLC_V2_* variables are ignored: the driver sets every launcher setting itself.
#
# Exit codes: 0 done; 1 a run or piece failed or did not finish (see its info.log, rerun to
# resume) or another driver holds the lock; 2 bad option, STEP4_MODEL, STEP4_STOP_AT or run
# key; 3 (eval) every piece of the trained rows is done but some default rows are not trained
# yet; 5 session over (rerun in the next session); 124 or 137 killed by the session's timeout.
#
# Outputs: STEP4_ROOT/summary/<model>/runs.md and runs.csv (train, one row per key, rebuilt after every
# run); STEP4_EVAL_ROOT/<model>/<run>/<tag>/<model>_<dataset>_<suffix>/rank_eval_results.{json,csv}
# and per_query_ranks.csv (eval).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/session.sh
source "${SCRIPT_DIR}/lib/session.sh"
# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

# A stray exported launcher setting would rename the runs the planner looks for.
unset "${!MPLC_V2_@}"

PYTHON=${PYTHON:-python}
MODEL=${STEP4_MODEL:-}
ROOT=${STEP4_ROOT:-logs/mplc_v2_step4}
EVAL_ROOT=${STEP4_EVAL_ROOT:-output/mplc_v2_step4}
DATASETS=${STEP4_EVAL_DATASETS:-"msls pitts30k sped nordland"}
MAX_QUERIES=1000
ARGS=("$@")
MODE=run

case "${1:-}" in
  -h | --help)
    sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'
    exit 0
    ;;
  train | eval) ACTION=$1; shift ;;
  *) echo "expected train or eval (see --help), got: ${1:-nothing}" >&2; exit 2 ;;
esac
if [ $# -gt 1 ]; then
  echo "expected at most one option, got: $*" >&2
  exit 2
fi
case "${ACTION}:${1:-}" in
  *:) ;;
  *:--dry-run) MODE=dry ;;
  train:--summary-only) MODE=summary ;;
  *) echo "unknown option for ${ACTION}: $1" >&2; exit 2 ;;
esac
case "${MODEL}" in
  boq) PRETRAINED=${BOQ_BASE_CHECKPOINT} ;;
  mixvpr) PRETRAINED=${MIXVPR_BASE_CHECKPOINT} ;;
  dinomix) PRETRAINED=${DINOMIX_BASE_CHECKPOINT} ;;
  qaa) PRETRAINED=${QAA_BASE_CHECKPOINT} ;;
  *) echo "Set STEP4_MODEL to boq, mixvpr, dinomix or qaa, got: ${MODEL:-nothing}" >&2; exit 2 ;;
esac

plan() {
  local command=$1
  shift
  "${PYTHON}" -m src.step4_plan "${command}" --model "${MODEL}" "$@"
}

# Time left check before starting a run or piece that cannot pause on its own.
too_late() {
  [ $(($(date +%s) + 45 * 60)) -gt "${SESSION_STOP_BY}" ]
}

# --------------------------------------------------------------------------------- train

train() {
  local run_flags=() launch_env assignments pending name previous="" status
  # shellcheck disable=SC2206 # STEP4_RUNS is a space-separated list by design.
  [ -z "${STEP4_RUNS:-}" ] || run_flags=(--runs ${STEP4_RUNS})

  summarize() {
    plan summary --root "${ROOT}" >/dev/null ||
      echo "=== Could not rebuild ${ROOT}/summary/${MODEL} (see above); the runs are unaffected." >&2
  }
  run_name() {
    "${PYTHON}" -c 'import sys; from src.step4_plan import run_name; print(run_name(dict(a.split("=", 1) for a in sys.argv[1:])))' "$@"
  }

  # Unknown run keys exit 2 here, before a session starts.
  plan list "${run_flags[@]}" >/dev/null
  if [ "${MODE}" = "summary" ]; then
    plan summary --root "${ROOT}"
    exit 0
  fi

  # Passed on as MPLC_V2_*: the unset above clears every MPLC_V2_* variable.
  launch_env=(
    MPLC_V2_RUN_ROOT="${ROOT}"
    MPLC_V2_CLEAN_BUDGETS="${STEP4_CLEAN_BUDGETS:-$("${PYTHON}" -c 'from src.followup_plan import CLEAN_BUDGETS; print(CLEAN_BUDGETS)')}"
    MPLC_V2_PIPE_LOADER="${STEP4_PIPE_LOADER:-0}"
    PYTHON="${PYTHON}"
  )

  if [ "${MODE}" = "dry" ]; then
    pending="$(plan next --root "${ROOT}" --all "${run_flags[@]}")"
    if [ "${pending}" = "DONE" ]; then
      echo "=== Every selected Step 4 run is finished: ${ROOT}"
      exit 0
    fi
    echo "=== Pending Step 4 runs (${MODEL}), in order:"
    while read -r assignments; do
      # shellcheck disable=SC2086 # the assignments are separate VAR=value words
      env ${assignments} "${launch_env[@]}" MPLC_V2_DRY_RUN=1 scripts/mplc_v2_train.sh
    done <<<"${pending}"
    exit 0
  fi

  session_begin STEP4_STOP_AT "${SCRIPT_DIR}/mplc_v2_step4.sh" "${ARGS[@]}"
  session_lock "${ROOT}" "Step 4 training driver"

  while true; do
    assignments="$(plan next --root "${ROOT}" "${run_flags[@]}")"
    if [ "${assignments}" = "DONE" ]; then
      summarize
      echo "=== Step 4 training (${MODEL}) done. Tables: ${ROOT}/summary/${MODEL}/runs.md"
      exit 0
    fi
    # shellcheck disable=SC2086 # the assignments are separate VAR=value words
    name="$(run_name ${assignments})"
    if [ "${assignments}" = "${previous}" ]; then
      echo "Run ${name} ran but did not finish; see ${ROOT}/${name}/info.log and run_status.json." >&2
      exit 1
    fi
    previous="${assignments}"
    if [ ! -d "${ROOT}/${name}" ] && too_late; then
      echo "=== Session over: under 45 min left before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
      exit 5
    fi
    echo "=== Step 4 run: ${name}"
    echo "    ${assignments}"
    status=0
    # shellcheck disable=SC2086 # the assignments are separate VAR=value words
    env ${assignments} "${launch_env[@]}" MPLC_V2_STOP_BY="${SESSION_STOP_BY}" scripts/mplc_v2_train.sh || status=$?
    if [ "${status}" -eq 5 ]; then
      summarize
      echo "=== Session over: the run paused before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
      exit 5
    fi
    [ "${status}" -eq 0 ] || exit "${status}"
    summarize
  done
}

# ---------------------------------------------------------------------------------- eval

# The eval launcher's flags for one evaluation run, after --model-type, --datasets and --model.
piece_flags() {
  case "$1" in
    r1_apgd) flags=(--attack rank_apgd_linf --steps 100 --epsilons "0.03425 0.0685") ;;
    r2_pgd_grid) flags=() ;;
    twin_clean) flags=(--epsilons 0.01712) ;;
  esac
}

evaluate() {
  local lines group rest main=() twin=() pending=() piece dataset entry tag path dir marker flags entries started=0
  lines="$(plan models --root "${ROOT}" --pretrained "${PRETRAINED}")"
  while read -r group rest; do
    case "${group}" in
      main) main+=("${rest}") ;;
      twin) twin+=("${rest}") ;;
      pending) pending+=("${rest}") ;;
    esac
  done <<<"${lines}"

  if [ "${MODE}" != "dry" ]; then
    session_begin STEP4_STOP_AT "${SCRIPT_DIR}/mplc_v2_step4.sh" "${ARGS[@]}"
    session_lock "${EVAL_ROOT}/${MODEL}" "Step 4 evaluation driver"
  fi
  echo "=== Step 4 evaluation (${MODEL}): ${main[*]%%=*} ${twin[*]%%=*}"
  [ "${#pending[@]}" -eq 0 ] || echo "    not trained yet: ${pending[*]}"

  for piece in r1_apgd r2_pgd_grid twin_clean; do
    entries=("${main[@]}")
    [ "${piece}" != "twin_clean" ] || entries=("${twin[@]+"${twin[@]}"}")
    piece_flags "${piece}"
    for dataset in ${DATASETS}; do
      for entry in "${entries[@]+"${entries[@]}"}"; do
        tag=${entry%%=*}
        path=${entry#*=}
        dir="${EVAL_ROOT}/${MODEL}/${piece}/${tag}"
        marker="${dir}/${dataset}.done"
        # Done only for the same checkpoint: a run retrained under the same name reruns.
        if [ -f "${marker}" ] && [ "$(cat "${marker}")" = "${path}" ]; then
          continue
        fi
        cmd=(scripts/mplc_v2_eval.sh --model-type "${MODEL}" --datasets "${dataset}" --model "${tag}=${path}"
             --max-queries "${MAX_QUERIES}" "${flags[@]+"${flags[@]}"}" --output-root "${dir}")
        if [ "${MODE}" = "dry" ]; then
          PYTHON="${PYTHON}" MPLC_V2_DRY_RUN=1 "${cmd[@]}"
          continue
        fi
        # ponytail: no per-piece time estimate (one model on one dataset, ~0.5-1 h for APGD);
        # a piece cut by the session's timeout reruns from scratch next session.
        if too_late; then
          echo "=== Session over: under 45 min left before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
          exit 5
        fi
        echo "=== Piece ${piece} / ${tag} / ${dataset}"
        if ! PYTHON="${PYTHON}" "${cmd[@]}"; then
          echo "Piece ${piece} / ${tag} / ${dataset} failed; see the output above. Rerun to retry it." >&2
          exit 1
        fi
        printf '%s\n' "${path}" >"${marker}"
        started=1
      done
    done
  done

  [ "${MODE}" != "dry" ] || exit 0
  if [ "${#pending[@]}" -gt 0 ]; then
    echo "=== Every piece of the trained rows is done; not trained yet: ${pending[*]}. Rerun after training them." >&2
    exit 3
  fi
  [ "${started}" = "1" ] || echo "=== Nothing left to evaluate."
  echo "=== Step 4 evaluation done (${MODEL}): ${EVAL_ROOT}/${MODEL}/"
}

if [ "${ACTION}" = "train" ]; then
  train
else
  evaluate
fi
