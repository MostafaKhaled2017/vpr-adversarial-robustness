#!/usr/bin/env bash
# Step 2 of the sprint 5 runbook: full training of MPLC*, its clean twin, the two baseline
# defenses and the ablations.
#
# MPLC* is recomputed from the finished Step 1 sweep in STEP1_ROOT (src/step2_plan.py).
# Runs, in order:
#   2.0  MPLC* screens, seeds 1 and 2 (Step 1 length): with MPLC*'s Step 1 screen they give
#        the noise the baseline decisions use (Step 1a's noise was measured where every
#        seed scored 0)
#   2.1  clean twin and MPLC*, seeds 0 and 1, full length
#   2.2  plain_at (lr) and fare (λ_anchor × lr, plus one grid-edge screen) screens at the
#        Step 1 screen length, scored like Step 1; then each baseline's winner at full length
#   2.3  the MPLC* ablations that MPLC* does not already cover, seed 0, full length
# Every run lives in a fixed directory under STEP2_ROOT, so the script can be stopped at
# any point and rerun to resume: finished runs are skipped, a stopped run continues from
# its last epoch, and baseline winners are recomputed from the finished screens.
#
# Stages: a finished run holds ~7 GB of checkpoints at freeze_te=0, so Step 2 does not fit
# on disk at once. A new run starts only while STEP2_ROOT plus RUN_GB stays within
# STEP2_MAX_GB (a run already on disk always continues, so a stopped run resumes); otherwise
# the driver exits 4. Then zip the finished runs' *.pth files into STEP2_ROOT/_download/,
# download the zips from JupyterLab, delete them from the server and rerun to start the next
# stage (docs/reports/sprint5/step2_stage_commands.md). Zips left in _download/ count toward
# STEP2_MAX_GB. Keep each run directory and its other files: the planner reads
# run_status.json and validation_recalls.jsonl, and a missing directory is trained again.
# Never zip a checkpoint of an unfinished run: it resumes from last_model.pth. The driver
# itself never removes a file.
#
# Sessions: a real run needs STEP2_STOP_AT, the time by which every process must be gone
# (the server's clock). The driver reruns itself under `timeout`, which sends Ctrl-C to it,
# train.py and the data-loader workers 10 min before STEP2_STOP_AT and SIGKILL 5 min before.
# That is the guarantee; before it, train.py pauses cleanly rather than start an epoch that,
# with the final test, could end after STEP2_STOP_AT - 15 min, and the driver starts no run
# with less than 45 min left before then. Rerun with the next session's STEP2_STOP_AT to
# resume; a run killed or paused mid-way continues from its last epoch.
#
# Usage:
#   STEP2_STOP_AT="2026-10-01 07:30" scripts/mplc_v2_step2.sh   # run or resume Step 2
#   scripts/mplc_v2_step2.sh --dry-run       # print the pending runs' commands only
#   scripts/mplc_v2_step2.sh --summary-only  # rebuild the tables, train nothing
#
# Environment overrides:
#   STEP1_ROOT            finished Step 1 sweep              (default: logs/mplc_v2_step1)
#   STEP2_ROOT            Step 2 run directory               (default: logs/mplc_v2_step2)
#   STEP2_STOP_AT         end of this session, any `date -d` time, e.g. "2026-10-01 07:30"
#                         (required for a real run; see Sessions)
#   STEP2_BASELINE_SEEDS  seeds of the baselines' full runs  (default: 0; "0 1" adds seed 1)
#   STEP2_MAX_GB          disk limit of one stage, GB        (default: 25, i.e. 3 runs)
#   STEP2_NUM_WORKERS     data-loader processes per run; leave unset: another training count
#                         changes the augmentation draws     (default: unset = 4 train, 8 val)
#   STEP2_PIPE_LOADER     1 = data-loader batches through a pipe, not /dev/shm (small /dev/shm)
#                                                            (default: 0)
#   PYTHON               python interpreter                 (default: python)
# Other MPLC_V2_* variables are ignored: Step 2 sets every launcher setting itself.
#
# Exit codes: 0 done; 1 a run failed (see its info.log, rerun to resume); 2 bad option or
# STEP2_STOP_AT; 3 stopped: Step 1 is not finished or changed since Step 2 started, or a 2.0
# noise screen is a loss (reported after every other run has trained); 4 stage full (see
# Stages); 5 session over (see Sessions); 124 or 137 killed by the session's timeout.
#
# Outputs in ${STEP2_ROOT}/summary/, rebuilt before every run and at the end: runs.md
# (decisions and one row per run) and runs.csv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

# A stray exported launcher setting would rename the runs the planner looks for.
unset "${!MPLC_V2_@}"

PYTHON=${PYTHON:-python}
STEP1=${STEP1_ROOT:-logs/mplc_v2_step1}
ROOT=${STEP2_ROOT:-logs/mplc_v2_step2}
# Passed on as MPLC_V2_*: the unset above clears every MPLC_V2_* variable.
WORKERS_ENV=(MPLC_V2_PIPE_LOADER="${STEP2_PIPE_LOADER:-0}")
[ -z "${STEP2_NUM_WORKERS:-}" ] || WORKERS_ENV+=(MPLC_V2_NUM_WORKERS="${STEP2_NUM_WORKERS}")
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

# shellcheck disable=SC2206 # STEP2_BASELINE_SEEDS is a space-separated list by design.
PLAN_FLAGS=(
  --root "${ROOT}"
  --step1-root "${STEP1}"
  --batch-size "${SUPERVLAD_TRAIN_BATCH_SIZE}"
  --batches-per-epoch "${SUPERVLAD_BATCHES_PER_EPOCH}"
  --baseline-seeds ${STEP2_BASELINE_SEEDS:-0}
)
# A dry run must not freeze the Step 1 result for the real one.
[ "${MODE}" = "dry" ] && PLAN_FLAGS+=(--no-freeze)

plan() {
  "${PYTHON}" -m src.step2_plan "$1" "${PLAN_FLAGS[@]}"
}

# The run directory's name for one line of `plan next` (VAR=value words).
run_name() {
  "${PYTHON}" -c 'import sys; from src.step2_plan import run_name; print(run_name(dict(a.split("=", 1) for a in sys.argv[1:])))' "$@"
}

# Room one new run needs, GB: five 1.17 GB checkpoints, the two initial models and the
# temporary file of a checkpoint being saved.
RUN_GB=8
MAX_GB=${STEP2_MAX_GB:-25}

# Session deadline (see Sessions), all in Unix seconds.
RUN_START_SECONDS=$((45 * 60))
if [ "${MODE}" = "run" ]; then
  if [ -z "${STEP2_STOP_AT:-}" ]; then
    echo "Set STEP2_STOP_AT to the time this session must end, e.g. STEP2_STOP_AT=\"2026-10-01 07:30\"." >&2
    exit 2
  fi
  if ! stop_at=$(date -d "${STEP2_STOP_AT}" +%s 2>/dev/null); then
    echo "STEP2_STOP_AT is not a time: ${STEP2_STOP_AT}" >&2
    exit 2
  fi
  interrupt_at=$((stop_at - 10 * 60))
  stop_by=$((stop_at - 15 * 60))
  if [ -z "${_STEP2_UNDER_TIMEOUT:-}" ]; then
    now=$(date +%s)
    if [ $((stop_at - now)) -lt 3600 ]; then
      echo "STEP2_STOP_AT $(date -d "@${stop_at}" '+%F %T') is less than 1 h away: nothing would train." >&2
      exit 2
    fi
    echo "=== Session ends at $(date -d "@${stop_at}" '+%F %T %Z') (STEP2_STOP_AT=\"${STEP2_STOP_AT}\")"
    echo "    no run starts after       $(date -d "@$((stop_by - RUN_START_SECONDS))" '+%F %T')"
    echo "    no epoch ends after       $(date -d "@${stop_by}" '+%F %T') (then the final test)"
    echo "    Ctrl-C to every process   $(date -d "@${interrupt_at}" '+%F %T')"
    echo "    SIGKILL to every process  $(date -d "@$((stop_at - 5 * 60))" '+%F %T')"
    # timeout signals its whole process group: this driver, train.py and its workers. The
    # resolved time is passed on, so a relative STEP2_STOP_AT ("+11 hours") is not re-read.
    STEP2_STOP_AT="@${stop_at}" _STEP2_UNDER_TIMEOUT=1 exec timeout -s INT -k 5m "$((interrupt_at - now))s" "${SCRIPT_DIR}/mplc_v2_step2.sh" "$@"
  fi
fi

if [ "${MODE}" = "summary" ]; then
  plan summarize
  exit 0
fi

if [ "${MODE}" = "run" ]; then
  mkdir -p "${ROOT}"
  # One driver per directory: a second copy would train the same run twice.
  exec 9>"${ROOT}/.lock"
  if ! flock -n 9; then
    echo "Another Step 2 driver is running on ${ROOT}." >&2
    exit 1
  fi
fi

previous=""
while true; do
  next="$(plan next)"
  case "${next}" in
    DONE)
      echo "=== Step 2 done. Tables: ${ROOT}/summary/runs.md"
      exit 0
      ;;
    STOP*)
      echo "=== Step 2 stopped: ${next#STOP }" >&2
      echo "    Tables so far: ${ROOT}/summary/" >&2
      exit 3
      ;;
  esac

  if [ "${MODE}" = "dry" ]; then
    echo "=== Pending runs, in order:"
    while read -r assignments; do
      # shellcheck disable=SC2086 # the assignments are separate VAR=value words
      env ${assignments} "${WORKERS_ENV[@]}" MPLC_V2_RUN_ROOT="${ROOT}" MPLC_V2_DRY_RUN=1 PYTHON="${PYTHON}" scripts/mplc_v2_train.sh
    done <<<"${next}"
    exit 0
  fi

  assignments="$(head -n 1 <<<"${next}")"
  if [ "${assignments}" = "${previous}" ]; then
    echo "Run ${assignments} ran but did not finish; see its info.log and run_status.json." >&2
    exit 1
  fi
  previous="${assignments}"
  # shellcheck disable=SC2086 # the assignments are separate VAR=value words
  if [ ! -d "${ROOT}/$(run_name ${assignments})" ]; then
    used_gb=$(($(du -sB1 "${ROOT}" | cut -f1) / 1000000000))
    if [ $((used_gb + RUN_GB)) -gt "${MAX_GB}" ]; then
      echo "=== Stage full: ${ROOT} holds ${used_gb} GB; a new run needs ${RUN_GB} GB of the ${MAX_GB} GB limit." >&2
      echo "    Zip the finished runs' *.pth files, download and delete the zips (keep the run directories), then rerun." >&2
      exit 4
    fi
  fi
  if [ $(($(date +%s) + RUN_START_SECONDS)) -gt "${stop_by}" ]; then
    echo "=== Session over: under 45 min left before $(date -d "@${stop_by}" '+%F %T'). Rerun in the next session." >&2
    exit 5
  fi
  echo "=== Step 2 run: ${assignments}"
  status=0
  # shellcheck disable=SC2086 # the assignments are separate VAR=value words
  env ${assignments} "${WORKERS_ENV[@]}" MPLC_V2_RUN_ROOT="${ROOT}" MPLC_V2_STOP_BY="${stop_by}" PYTHON="${PYTHON}" \
    scripts/mplc_v2_train.sh || status=$?
  if [ "${status}" -eq 5 ]; then
    echo "=== Session over: the run paused before $(date -d "@${stop_by}" '+%F %T'). Rerun in the next session." >&2
    exit 5
  fi
  [ "${status}" -eq 0 ] || exit "${status}"
done
