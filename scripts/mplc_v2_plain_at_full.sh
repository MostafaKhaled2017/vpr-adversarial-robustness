#!/usr/bin/env bash
# The plain_at baseline's full run, trained after Step 2 for the Step 3 evaluation.
#
# Step 2's planner skips this run because both plain_at screens were losses: neither had a
# trained checkpoint within the 5-point clean budget. Step 3 also evaluates each defense at
# its unconstrained best checkpoint, so plain_at needs a run as long as MPLC*'s. Its settings
# come from src/step3_plan.py (plain_at_env): Step 2's full-run settings (validate every 3
# epochs, patience 6, lr-plateau patience 3, up to 100 epochs), MPLC*'s freeze_te, the
# preferred lr 1e-5 and seed 0. The run lives in STEP2_ROOT next to the Step 2 runs, as
# mplc_v2_supervlad_plain_at_fte<N>_s0.
#
# It starts only once Step 2 is done (its planner prints DONE) and holds Step 2's lock, so it
# never trains alongside the Step 2 driver. Rerun to resume: a stopped run continues from its
# last epoch and a finished one is skipped.
#
# Sessions: a real run needs PLAIN_AT_STOP_AT, the time by which every process must be gone
# (the server's clock). Same rules as STEP2_STOP_AT (scripts/lib/session.sh): Ctrl-C to every
# process 10 min before it, SIGKILL 5 min before; before that train.py pauses cleanly rather
# than start an epoch that, with the final test, could end after PLAIN_AT_STOP_AT - 15 min,
# and no run starts with less than 45 min left before then.
#
# Usage:
#   PLAIN_AT_STOP_AT="2026-10-12 08:30" scripts/mplc_v2_plain_at_full.sh   # train or resume
#   scripts/mplc_v2_plain_at_full.sh --dry-run   # print the command only
#
# Environment overrides:
#   STEP1_ROOT            finished Step 1 sweep                (default: logs/mplc_v2_step1)
#   STEP2_ROOT            Step 2 run directory                 (default: logs/mplc_v2_step2)
#   PLAIN_AT_STOP_AT      end of this session, any `date -d` time (required for a real run)
#   PLAIN_AT_PIPE_LOADER  1 = data-loader batches through a pipe, not /dev/shm (default: 0)
#   PYTHON                python interpreter                   (default: python)
# Other MPLC_V2_* variables are ignored.
#
# Exit codes: 0 done (or already finished); 1 the run failed (see its info.log, rerun to
# resume) or another driver holds the lock; 2 bad option or PLAIN_AT_STOP_AT; 3 Step 2 is not
# done; 5 session over (rerun in the next session); 124 or 137 killed by the session's timeout.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"
# shellcheck source=scripts/lib/session.sh
source "${SCRIPT_DIR}/lib/session.sh"

# A stray exported launcher setting would rename the run.
unset "${!MPLC_V2_@}"

PYTHON=${PYTHON:-python}
STEP1=${STEP1_ROOT:-logs/mplc_v2_step1}
ROOT=${STEP2_ROOT:-logs/mplc_v2_step2}
MODE=run

case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry ;;
  -h | --help)
    sed -n '2,40p' "$0"
    exit 0
    ;;
  *)
    echo "unknown option: $1" >&2
    exit 2
    ;;
esac

if [ "${MODE}" = "run" ]; then
  session_begin PLAIN_AT_STOP_AT "${SCRIPT_DIR}/mplc_v2_plain_at_full.sh" "$@"
  session_lock "${ROOT}" "Step 2 or plain_at driver"
fi

# Step 2 must be done: a pending Step 2 run must not be overtaken by this one.
step2_next="$("${PYTHON}" -m src.step2_plan next --root "${ROOT}" --step1-root "${STEP1}" \
  --batch-size "${SUPERVLAD_TRAIN_BATCH_SIZE}" --batches-per-epoch "${SUPERVLAD_BATCHES_PER_EPOCH}" --no-freeze)"
if [ "${step2_next}" != "DONE" ]; then
  echo "=== Step 2 is not done; its next runs:" >&2
  sed 's/^/    /' <<<"${step2_next}" >&2
  if [ "${MODE}" = "run" ]; then
    echo "    Finish Step 2 (scripts/mplc_v2_step2.sh) first." >&2
    exit 3
  fi
fi

plain_at="$("${PYTHON}" -m src.step3_plan plain-at-env --step2-root "${ROOT}")"
assignments="$(head -n 1 <<<"${plain_at}")"
name="$(tail -n 1 <<<"${plain_at}")"
LAUNCH_ENV=(MPLC_V2_RUN_ROOT="${ROOT}" MPLC_V2_PIPE_LOADER="${PLAIN_AT_PIPE_LOADER:-0}" PYTHON="${PYTHON}")

if [ "${MODE}" = "dry" ]; then
  echo "=== plain_at full run: ${ROOT}/${name}"
  # shellcheck disable=SC2086 # the assignments are separate VAR=value words
  env ${assignments} "${LAUNCH_ENV[@]}" MPLC_V2_DRY_RUN=1 scripts/mplc_v2_train.sh
  exit 0
fi

if [ ! -d "${ROOT}/${name}" ] && [ $(($(date +%s) + 45 * 60)) -gt "${SESSION_STOP_BY}" ]; then
  echo "=== Session over: under 45 min left before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
  exit 5
fi
echo "=== plain_at full run: ${assignments}"
status=0
# shellcheck disable=SC2086 # the assignments are separate VAR=value words
env ${assignments} "${LAUNCH_ENV[@]}" MPLC_V2_STOP_BY="${SESSION_STOP_BY}" scripts/mplc_v2_train.sh || status=$?
if [ "${status}" -eq 5 ]; then
  echo "=== Session over: the run paused before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
  exit 5
fi
[ "${status}" -eq 0 ] || exit "${status}"
echo "=== plain_at full run done: ${ROOT}/${name}"
