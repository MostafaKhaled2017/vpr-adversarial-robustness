#!/usr/bin/env bash
# Step 3 of the sprint 5 runbook: evaluation on the test sets, after Step 2 and the plain_at
# full run (scripts/mplc_v2_plain_at_full.sh).
#
# src/step3_plan.py resolves the models from the finished Step 2 runs and lists the work as
# pieces, one eval.py call per (run, dataset, model). Models (seed 0): pretrained, plain_at,
# MPLC* and the MPLC* ablations. Two tables: Table A at each run's budget-5 checkpoint, Table B
# at its best_model.pth; a checkpoint shared by both is evaluated once, and one that is the
# initial model is reported as pretrained. Runs, in order (runbook §3.2): P targeted pilot on
# SPED; 1 rank-APGD-100 at 4/255 (headline); 3 targeted rank-PGD-20; 4 embedding-shift PGD-20;
# 2 rank-PGD-20 over the eps grid; 7 transfer from pretrained (MSLS); 8 32/255 (MSLS); 9 and 9b
# APGD with 500 steps / 3 restarts (MPLC*, MSLS); 6 L2 rank-PGD-20; A runs 1 and 2 for the
# ablations on MSLS and Nordland. The worst case over runs 1, 3 and 4 (run 5) is computed in
# the summary.
#
# Every piece writes to a fixed directory under STEP3_ROOT and is marked DONE once eval.py
# exits 0, so the script can be stopped at any point and rerun to resume: done pieces are
# skipped and a piece killed mid-way runs again. The models are recorded in
# STEP3_ROOT/models.json at the first start; a later start whose checkpoints differ stops.
#
# Targeted pilot: it fails when every main model's targeted success rate is <= 5 % or every one
# is >= 95 %. Then run 3 waits and every other run continues; set STEP3_TARGET_RANK (a farther
# target when everything is near 100 %, a closer one near 0 %) for the next session, and the
# pilot runs again first.
#
# Sessions: a real run needs STEP3_STOP_AT, the time by which every process must be gone (the
# server's clock). Same rules as STEP2_STOP_AT (scripts/lib/session.sh): Ctrl-C to every
# process 10 min before it, SIGKILL 5 min before. Before that, a piece starts only if its
# estimated time ends by STEP3_STOP_AT - 15 min: 1.2 x the longest measured time per model of
# the same (run, dataset), or the runbook's upper bound before one is measured
# (STEP3_ROOT/durations.csv). A shorter piece further down the list runs first when the next
# one does not fit.
#
# Usage:
#   STEP3_STOP_AT="2026-10-14 08:30" scripts/mplc_v2_step3.sh   # run or resume Step 3
#   scripts/mplc_v2_step3.sh --dry-run        # list the pending pieces and their commands
#   scripts/mplc_v2_step3.sh --summary-only   # rebuild the tables, evaluate nothing
#
# Environment overrides:
#   STEP2_ROOT          Step 2 run directory              (default: logs/mplc_v2_step2)
#   STEP3_ROOT          Step 3 output directory           (default: output/mplc_v2_step3)
#   STEP3_STOP_AT       end of this session, any `date -d` time (required for a real run)
#   STEP3_TARGET_RANK   targeted attack's target rank     (default: 10)
#   PYTHON              python interpreter                (default: python)
#
# Exit codes: 0 done; 1 a piece failed (see its directory's info.log, rerun to resume) or
# another driver holds the lock; 2 bad option or STEP3_STOP_AT; 3 stopped: a Step 2 run or the
# plain_at run is missing or unfinished, the checkpoints changed since the first start, or (at
# the end) run 3 still waits for a pilot that passes; 5 session over (rerun in the next
# session); 124 or 137 killed by the session's timeout.
#
# Outputs in ${STEP3_ROOT}/summary/, rebuilt after every piece: status.md, table_budget5 and
# table_best (.md, .csv), ablations_budget5 and ablations_best, curves_pgd.csv, curves_l2.csv,
# worst_case.csv, paired.csv (MPLC* vs plain_at, exact McNemar test) and validity.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"
# shellcheck source=scripts/lib/session.sh
source "${SCRIPT_DIR}/lib/session.sh"

PYTHON=${PYTHON:-python}
STEP2=${STEP2_ROOT:-logs/mplc_v2_step2}
ROOT=${STEP3_ROOT:-output/mplc_v2_step3}
TARGET_RANK=${STEP3_TARGET_RANK:-10}
MODE=run

case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry ;;
  --summary-only) MODE=summary ;;
  -h | --help)
    sed -n '2,60p' "$0"
    exit 0
    ;;
  *)
    echo "unknown option: $1" >&2
    exit 2
    ;;
esac

PLAN_FLAGS=(--root "${ROOT}" --step2-root "${STEP2}" --target-rank "${TARGET_RANK}")
plan() {
  "${PYTHON}" -m src.step3_plan "$@" "${PLAN_FLAGS[@]}"
}

# Flags every eval.py call shares; each piece adds its models, dataset, attack and outputs.
EVAL_FLAGS=(--eval_datasets_folder=datasets --dataset_split=test "${SUPERVLAD_EVAL_MODEL_FLAGS[@]}" --test_method=hard_resize)

case "${MODE}" in
  summary)
    plan summarize
    exit 0
    ;;
  dry)
    echo "=== Every piece runs: ${PYTHON} eval.py ${EVAL_FLAGS[*]} <piece flags>"
    plan list
    exit 0
    ;;
esac

session_begin STEP3_STOP_AT "${SCRIPT_DIR}/mplc_v2_step3.sh" "$@"
session_lock "${ROOT}" "Step 3 driver"

previous=""
while true; do
  mapfile -t next < <(plan next --time-left "$((SESSION_STOP_BY - $(date +%s)))")
  case "${next[0]:-}" in
    DONE)
      plan summarize
      echo "=== Step 3 done. Tables: ${ROOT}/summary/"
      exit 0
      ;;
    STOP*)
      plan summarize >/dev/null || true
      echo "=== Step 3 stopped: ${next[0]#STOP }" >&2
      echo "    Status: ${ROOT}/summary/status.md" >&2
      exit 3
      ;;
    WAIT*)
      echo "=== Session over: ${next[0]#WAIT } before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
      exit 5
      ;;
    PIECE*) ;;
    *)
      echo "Unexpected planner output: ${next[*]}" >&2
      exit 1
      ;;
  esac

  piece="${next[0]#PIECE }"
  estimate="${next[1]}"
  piece_dir="${next[2]}"
  if [ "${piece}" = "${previous}" ]; then
    echo "Piece ${piece} ran but is not marked done; see ${piece_dir}." >&2
    exit 1
  fi
  previous="${piece}"
  mkdir -p "${piece_dir}"
  echo "=== Step 3 piece ${piece} (estimate $((estimate / 60)) min) at $(date '+%F %T')"
  start=$(date +%s)
  status=0
  "${PYTHON}" eval.py "${EVAL_FLAGS[@]}" "${next[@]:3}" || status=$?
  if [ "${status}" -ne 0 ]; then
    echo "=== Piece ${piece} failed with exit code ${status}; see ${piece_dir}/<timestamp>/info.log. Rerun to resume." >&2
    exit 1
  fi
  plan record "${piece}" --seconds "$(($(date +%s) - start))" >/dev/null
done
