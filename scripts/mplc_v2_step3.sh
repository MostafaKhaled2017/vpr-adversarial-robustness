#!/usr/bin/env bash
# Step 3 of the sprint 5 runbook: evaluation on the test sets of the checkpoints given with
# --model. Session commands: docs/reports/sprint5/step3_session_commands.md.
#
# Models (seed 0): the pretrained model, always, and every --model KEY:TABLES:PATH.
#   KEY     the table row: plain_at, mplc (main models), no_anchor, hinge, single_pos
#           (ablations). MPLC* is its own "no perceptual attacks" ablation.
#   TABLES  A: Table A, the budget-5 checkpoint (most robust within 5 clean R@1 points);
#           B: Table B, best_model.pth (most robust overall); AB: one checkpoint for both,
#           evaluated once.
#   PATH    the checkpoint file, relative to the repository root, or `pretrained` for a run
#           whose checkpoint for that table is the initial model (the row then shows the
#           pretrained model).
# Nothing is read from Step 2: the checkpoints are chosen by the caller. Every --model is
# recorded in STEP3_ROOT/models.json and the work is planned from everything recorded there,
# so a later start adds its models to the earlier ones (Step 3a, then Step 3b) and may leave
# the recorded ones out. A recorded KEY and table given another path stops the driver.
#
# src/step3_plan.py lists the work as pieces, one eval.py call per (run, dataset, model).
# Runs, in order (runbook §3.2): P targeted pilot on SPED; 1 rank-APGD-100 at 4/255
# (headline); 3 targeted rank-PGD-20; 4 embedding-shift PGD-20; 2 rank-PGD-20 over the eps
# grid; 7 transfer from pretrained (MSLS; waits until both mplc and plain_at are given); 8
# 32/255 (MSLS); 9 and 9b APGD with 500 steps / 3 restarts (MPLC*, MSLS); 6 L2 rank-PGD-20; A
# runs 1 and 2 for the ablations on MSLS and Nordland. The worst case over runs 1, 3 and 4
# (run 5) is computed in the summary.
#
# Every piece writes to a fixed directory under STEP3_ROOT and is marked DONE once eval.py
# exits 0, so the script can be stopped at any point and rerun to resume: done pieces are
# skipped and a piece killed mid-way runs again.
#
# Targeted pilot: it fails when every main model's targeted success rate is <= 5 % or every one
# is >= 95 %. Then run 3 waits and every other run continues; set STEP3_TARGET_RANK (a farther
# target when everything is near 100 %, a closer one near 0 %) for the next session, and the
# pilot runs again first. Once run 3 has finished a piece, its target rank is fixed.
#
# Sessions: a real run needs STEP3_STOP_AT, the time by which every process must be gone (the
# machine's clock). Same rules as STEP2_STOP_AT (scripts/lib/session.sh): Ctrl-C to every
# process 10 min before it, SIGKILL 5 min before. Before that, a piece starts only if its
# estimated time ends by STEP3_STOP_AT - 15 min: 1.2 x the longest measured time per model of
# the same (run, dataset), or the runbook's upper bound before one is measured
# (STEP3_ROOT/durations.csv). A shorter piece further down the list runs first when the next
# one does not fit.
#
# Usage:
#   STEP3_STOP_AT="2026-10-14 08:30" scripts/mplc_v2_step3.sh --model KEY:TABLES:PATH ...
#                                             # run or resume Step 3
#   scripts/mplc_v2_step3.sh --dry-run [--model ...]   # list the pending pieces and commands
#   scripts/mplc_v2_step3.sh --summary-only   # rebuild the tables, evaluate nothing
# e.g. --model mplc:AB:logs/mplc_v2_step2/mplc_v2_supervlad_mplc_mixlinf_fte0_s0/best_model.pth
#      --model no_anchor:A:pretrained
#
# Environment overrides:
#   STEP3_ROOT          Step 3 output directory           (default: output/mplc_v2_step3)
#   STEP3_STOP_AT       end of this session, any `date -d` time (required for a real run)
#   STEP3_TARGET_RANK   targeted attack's target rank     (default: 10)
#   STEP3_PRETRAINED    pretrained SuperVLAD checkpoint   (default: checkpoints/SuperVLAD_base.pth)
#   PYTHON              python interpreter                (default: python)
#
# Exit codes: 0 done (every piece of the recorded models); 1 a piece failed (see its
# directory's info.log, rerun to resume) or another driver holds the lock; 2 bad option or
# STEP3_STOP_AT, a malformed --model or missing checkpoint file, or no model given; 3 stopped:
# a --model conflicts with a recorded one, a recorded checkpoint file is missing,
# STEP3_TARGET_RANK differs from run 3's, or (at the end) run 3 still waits for a pilot that
# passes; 5 session over (rerun in the next session); 124 or 137 killed by the session's
# timeout.
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
ROOT=${STEP3_ROOT:-output/mplc_v2_step3}
TARGET_RANK=${STEP3_TARGET_RANK:-10}
PRETRAINED=${STEP3_PRETRAINED:-checkpoints/SuperVLAD_base.pth}
MODE=run
MODELS=()

ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE=dry ;;
    --summary-only) MODE=summary ;;
    --model)
      if [ $# -lt 2 ]; then
        echo "--model needs KEY:TABLES:PATH" >&2
        exit 2
      fi
      MODELS+=(--model "$2")
      shift
      ;;
    --model=*) MODELS+=(--model "${1#--model=}") ;;
    -h | --help)
      sed -n '2,69p' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

PLAN_FLAGS=(--root "${ROOT}" --target-rank "${TARGET_RANK}" --pretrained "${PRETRAINED}")
plan() {
  "${PYTHON}" -m src.step3_plan "$@" "${PLAN_FLAGS[@]}"
}

# Flags every eval.py call shares; each piece adds its models, dataset, attack and outputs.
EVAL_FLAGS=(--eval_datasets_folder=datasets --dataset_split=test "${SUPERVLAD_EVAL_MODEL_FLAGS[@]}" --test_method=hard_resize)

case "${MODE}" in
  summary)
    if [ "${#MODELS[@]}" -gt 0 ]; then
      echo "--summary-only uses the recorded models; it takes no --model" >&2
      exit 2
    fi
    plan summarize
    exit 0
    ;;
  dry)
    echo "=== Every piece runs: ${PYTHON} eval.py ${EVAL_FLAGS[*]} <piece flags>"
    plan list "${MODELS[@]}"
    exit 0
    ;;
esac

session_begin STEP3_STOP_AT "${SCRIPT_DIR}/mplc_v2_step3.sh" "${ARGS[@]}"
session_lock "${ROOT}" "Step 3 driver"
plan register "${MODELS[@]}"

previous=""
while true; do
  mapfile -t next < <(plan next --time-left "$((SESSION_STOP_BY - $(date +%s)))")
  case "${next[0]:-}" in
    DONE)
      plan summarize
      echo "=== Step 3 done. Tables: ${ROOT}/summary/"
      if [ -n "${next[1]:-}" ]; then
        echo "    ${next[1]}"
      fi
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
