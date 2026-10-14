#!/usr/bin/env bash
# Follow-up training after Step 2: seven full-length runs, trained one after another.
#
# Runs (src/followup_plan.py), in default order, each one arm of scripts/mplc_v2_train.sh at
# Step 2's full-run settings (validate every 3 epochs, patience 6, lr-plateau patience 3, up
# to 100 epochs), freeze_te 0 and lr 1e-5. The sp_* runs are the single-positive MPLC
# (rank L-inf attacks only, hardest positive):
#   sp_s1          single positive, seed 1           mplc_v2_supervlad_mplc_mixlinf_sp_fte0_s1
#   sp_hinge       single positive, hinge loss       mplc_v2_supervlad_mplc_mixlinf_hinge_sp_fte0_s0
#   sp_aw2         single positive, lambda_anchor 2  mplc_v2_supervlad_mplc_aw2_mixlinf_sp_fte0_s0
#   fare_eps2      fare, training eps 0.0342         mplc_v2_supervlad_fare_eps0.0342_fte0_s0
#   plain_at_eps2  plain_at, training eps 0.0342     mplc_v2_supervlad_plain_at_eps0.0342_fte0_s0
#   sp_eps2        single positive, eps 0.0342       mplc_v2_supervlad_mplc_eps0.0342_mixlinf_sp_fte0_s0
#   sp_aw3         single positive, lambda_anchor 3  mplc_v2_supervlad_mplc_aw3_mixlinf_sp_fte0_s0
# Every run validates at eps 0.0342 and 0.0685 whatever its training eps. Each run lives in
# FOLLOWUP_ROOT/<name>, so the script can be stopped at any point and rerun to resume:
# finished runs are skipped and a stopped run continues from its last epoch. It does not
# depend on Step 2 (another directory) and does not wait for any other driver: do not start
# it while another training driver uses the GPU.
#
# Sessions: a real run needs FOLLOWUP_TRAIN_STOP_AT, the time by which every process must be
# gone (the server's clock). Same rules as STEP2_STOP_AT (scripts/lib/session.sh): Ctrl-C to
# every process 10 min before it, SIGKILL 5 min before; before that train.py pauses cleanly
# rather than start an epoch that, with the final test, could end after
# FOLLOWUP_TRAIN_STOP_AT - 15 min, and no new run starts with less than 45 min left before
# then. Rerun with the next session's FOLLOWUP_TRAIN_STOP_AT to resume.
#
# Usage:
#   FOLLOWUP_TRAIN_STOP_AT="2026-10-15 08:30" scripts/mplc_v2_followup_train.sh   # train or resume
#   scripts/mplc_v2_followup_train.sh --dry-run       # print the pending runs' commands only
#   scripts/mplc_v2_followup_train.sh --summary-only  # rebuild the tables, train nothing
#
# Environment overrides:
#   FOLLOWUP_TRAIN_STOP_AT  end of this session, any `date -d` time (required for a real run)
#   FOLLOWUP_ROOT           run directory                       (default: logs/mplc_v2_followup)
#   FOLLOWUP_RUNS           run keys to train, in this order    (default: all seven, as above)
#   FOLLOWUP_CLEAN_BUDGETS  clean budgets of the reporting checkpoints, points; keep 5, the
#                           tables' budget                      (default: "1 3 5 7 10")
#   FOLLOWUP_PIPE_LOADER    1 = data-loader batches through a pipe, not /dev/shm (default: 0)
#   PYTHON                  python interpreter                  (default: python)
# Other MPLC_V2_* variables are ignored: the driver sets every launcher setting itself.
#
# Exit codes: 0 done; 1 a run failed or did not finish (see its info.log, rerun to resume) or
# another driver holds the lock; 2 bad option, FOLLOWUP_TRAIN_STOP_AT or run key; 5 session
# over (rerun in the next session); 124 or 137 killed by the session's timeout.
#
# Outputs in ${FOLLOWUP_ROOT}/summary/, rebuilt after every run and at the end: runs.md and
# runs.csv, one row per run (all seven, whatever FOLLOWUP_RUNS selects) with its budget-5 and
# best checkpoints.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/session.sh
source "${SCRIPT_DIR}/lib/session.sh"

# A stray exported launcher setting would rename the runs the planner looks for.
unset "${!MPLC_V2_@}"

PYTHON=${PYTHON:-python}
ROOT=${FOLLOWUP_ROOT:-logs/mplc_v2_followup}
MODE=run

if [ $# -gt 1 ]; then
  echo "expected at most one option, got: $*" >&2
  exit 2
fi
case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry ;;
  --summary-only) MODE=summary ;;
  -h | --help)
    sed -n '2,/^set -euo pipefail$/p' "$0" | sed '$d'
    exit 0
    ;;
  *)
    echo "unknown option: $1" >&2
    exit 2
    ;;
esac

RUN_FLAGS=()
# shellcheck disable=SC2206 # FOLLOWUP_RUNS is a space-separated list by design.
[ -z "${FOLLOWUP_RUNS:-}" ] || RUN_FLAGS=(--runs ${FOLLOWUP_RUNS})

plan() {
  "${PYTHON}" -m src.followup_plan "$@"
}

# The run directory's name for one line of `plan next` (VAR=value words).
run_name() {
  "${PYTHON}" -c 'import sys; from src.followup_plan import run_name; print(run_name(dict(a.split("=", 1) for a in sys.argv[1:])))' "$@"
}

# The tables cover all seven runs, so a session that trains a subset hides none.
summarize() {
  plan summary --root "${ROOT}" >/dev/null ||
    echo "=== Could not rebuild ${ROOT}/summary (see above); the runs are unaffected." >&2
}

# Unknown run keys exit 2 here, before a session starts.
plan list "${RUN_FLAGS[@]}" >/dev/null

if [ "${MODE}" = "summary" ]; then
  plan summary --root "${ROOT}"
  exit 0
fi

CLEAN_BUDGETS=${FOLLOWUP_CLEAN_BUDGETS:-$("${PYTHON}" -c 'from src.followup_plan import CLEAN_BUDGETS; print(CLEAN_BUDGETS)')}
# Passed on as MPLC_V2_*: the unset above clears every MPLC_V2_* variable.
LAUNCH_ENV=(
  MPLC_V2_RUN_ROOT="${ROOT}"
  MPLC_V2_CLEAN_BUDGETS="${CLEAN_BUDGETS}"
  MPLC_V2_PIPE_LOADER="${FOLLOWUP_PIPE_LOADER:-0}"
  PYTHON="${PYTHON}"
)

if [ "${MODE}" = "dry" ]; then
  pending="$(plan next --root "${ROOT}" --all "${RUN_FLAGS[@]}")"
  if [ "${pending}" = "DONE" ]; then
    echo "=== Every follow-up run is finished: ${ROOT}"
    exit 0
  fi
  echo "=== Pending follow-up runs, in order:"
  while read -r assignments; do
    # shellcheck disable=SC2086 # the assignments are separate VAR=value words
    env ${assignments} "${LAUNCH_ENV[@]}" MPLC_V2_DRY_RUN=1 scripts/mplc_v2_train.sh
  done <<<"${pending}"
  exit 0
fi

session_begin FOLLOWUP_TRAIN_STOP_AT "${SCRIPT_DIR}/mplc_v2_followup_train.sh" "$@"
session_lock "${ROOT}" "follow-up training driver"

previous=""
while true; do
  assignments="$(plan next --root "${ROOT}" "${RUN_FLAGS[@]}")"
  if [ "${assignments}" = "DONE" ]; then
    summarize
    echo "=== Follow-up training done. Tables: ${ROOT}/summary/runs.md"
    exit 0
  fi
  # shellcheck disable=SC2086 # the assignments are separate VAR=value words
  name="$(run_name ${assignments})"
  if [ "${assignments}" = "${previous}" ]; then
    echo "Run ${name} ran but did not finish; see ${ROOT}/${name}/info.log and run_status.json." >&2
    exit 1
  fi
  previous="${assignments}"
  if [ ! -d "${ROOT}/${name}" ] && [ $(($(date +%s) + 45 * 60)) -gt "${SESSION_STOP_BY}" ]; then
    echo "=== Session over: under 45 min left before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
    exit 5
  fi
  echo "=== Follow-up run: ${name}"
  echo "    ${assignments}"
  status=0
  # shellcheck disable=SC2086 # the assignments are separate VAR=value words
  env ${assignments} "${LAUNCH_ENV[@]}" MPLC_V2_STOP_BY="${SESSION_STOP_BY}" scripts/mplc_v2_train.sh || status=$?
  if [ "${status}" -eq 5 ]; then
    summarize
    echo "=== Session over: the run paused before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
    exit 5
  fi
  [ "${status}" -eq 0 ] || exit "${status}"
  summarize
done
