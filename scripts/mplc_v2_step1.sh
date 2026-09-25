#!/usr/bin/env bash
# Step 1 of the sprint 5 runbook: the short-run hyperparameter sweep for the mplc arm.
#
# Runs stages 1a (noise) -> 1b (attack mix) -> 1c (freeze_te x lr) -> 1d (anchor weight) ->
# 1e (mix re-check), carrying each stage's winner into the next. Every decision is
# recomputed from the finished screens (src/step1_sweep.py), and every screen lives in a
# fixed directory under STEP1_ROOT, so the script can be stopped at any point and rerun to
# resume: finished screens are skipped, a stopped screen continues from its last epoch.
#
# Usage:
#   scripts/mplc_v2_step1.sh                 # run or resume the sweep
#   scripts/mplc_v2_step1.sh --dry-run       # print the next screen commands only
#   scripts/mplc_v2_step1.sh --summary-only  # rebuild tables and figures, train nothing
#
# Environment overrides:
#   STEP1_ROOT           sweep directory                          (default: logs/mplc_v2_step1)
#   STEP1_EPOCHS         epochs per screen; fixed once the sweep starts (default: 9)
#   STEP1_ACCEPT_NOISE=1 continue past 1a even if the noise is above 3 points
#   STEP1_PRUNE=1        after the sweep, delete the .pth files of every screen but MPLC*
#   PYTHON               python interpreter                       (default: python)
#
# Outputs in ${STEP1_ROOT}/summary/, rebuilt before every screen and at the end: screens.csv / screens.md,
# decisions.md, mplc_star.env (MPLC*'s settings for Step 2) and fig_*.pdf / fig_*.png.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

PYTHON=${PYTHON:-python}
ROOT=${STEP1_ROOT:-logs/mplc_v2_step1}
EPOCHS=${STEP1_EPOCHS:-9}
MODE=run

case "${1:-}" in
  "") ;;
  --dry-run) MODE=dry ;;
  --summary-only) MODE=summary ;;
  -h | --help)
    sed -n '2,24p' "$0"
    exit 0
    ;;
  *)
    echo "unknown option: $1" >&2
    exit 2
    ;;
esac

SWEEP_FLAGS=(--root "${ROOT}" --epochs "${EPOCHS}")
[ "${STEP1_ACCEPT_NOISE:-0}" = "1" ] && SWEEP_FLAGS+=(--accept-noise)
# A dry run must not freeze the sweep settings for the real one.
[ "${MODE}" = "dry" ] && SWEEP_FLAGS+=(--no-freeze)

sweep() {
  "${PYTHON}" -m src.step1_sweep "$1" "${SWEEP_FLAGS[@]}"
}

if [ "${MODE}" = "summary" ]; then
  sweep summarize
  exit 0
fi

if [ "${MODE}" = "run" ]; then
  mkdir -p "${ROOT}"
  # One sweep per directory: a second copy would train the same screen twice.
  exec 9>"${ROOT}/.lock"
  if ! flock -n 9; then
    echo "Another Step 1 sweep is running on ${ROOT}." >&2
    exit 1
  fi
fi

previous=""
while true; do
  next="$(sweep next)"
  case "${next}" in
    DONE)
      echo "=== Step 1 done. MPLC* settings: $(cat "${ROOT}/summary/mplc_star.env")"
      echo "    Tables and figures: ${ROOT}/summary/"
      if [ "${STEP1_PRUNE:-0}" = "1" ] && [ "${MODE}" = "run" ]; then
        sweep prune
      fi
      exit 0
      ;;
    STOP*)
      echo "=== Step 1 stopped: ${next#STOP }" >&2
      echo "    Tables so far: ${ROOT}/summary/" >&2
      exit 3
      ;;
  esac

  if [ "${MODE}" = "dry" ]; then
    echo "=== Next screens (current stage):"
    while read -r assignments; do
      # shellcheck disable=SC2086 # the assignments are separate VAR=value words
      env ${assignments} MPLC_V2_ARMS=mplc MPLC_V2_RUN_ROOT="${ROOT}" MPLC_V2_DRY_RUN=1 \
        PYTHON="${PYTHON}" scripts/mplc_v2_train.sh
    done <<<"${next}"
    exit 0
  fi

  assignments="$(head -n 1 <<<"${next}")"
  if [ "${assignments}" = "${previous}" ]; then
    echo "Screen ${assignments} ran but did not finish; see its info.log and run_status.json." >&2
    exit 1
  fi
  previous="${assignments}"
  echo "=== Step 1 screen: ${assignments}"
  # shellcheck disable=SC2086 # the assignments are separate VAR=value words
  env ${assignments} MPLC_V2_ARMS=mplc MPLC_V2_RUN_ROOT="${ROOT}" PYTHON="${PYTHON}" scripts/mplc_v2_train.sh
done
