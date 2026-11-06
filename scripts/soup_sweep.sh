#!/usr/bin/env bash
# Validate the weight average of two checkpoints at each mixing weight, one validation-only
# screen per weight, into ${RUN_ROOT}/KEY_a<ALPHA>/.
#
# Usage:
#   scripts/soup_sweep.sh MODEL KEY A B ALPHA... [STOP_AT="YYYY-MM-DD HH:MM"]
#
#   MODEL   supervlad | boq
#   KEY     screen name; mixing weight ALPHA gives the screen KEY_a<ALPHA>
#   A, B    checkpoints; the soup is (1 - ALPHA) * A + ALPHA * B (scripts/soup.py).
#           A = B with ALPHA 0 validates a single checkpoint.
#   ALPHA   mixing weights of B
#
# Each screen is scored as a 12-epoch training screen (scripts/validate_only.py), so
# scripts/screen_summary.py lists both kinds in one table. The clean drop is measured from
# the pretrained release's clean R@1 on the validation sample: 88.10 for SuperVLAD, 91.35 for
# BoQ. One screen takes up to 12 min (SuperVLAD) or 13 min (BoQ) on an RTX 4090. A rerun skips
# a mixing weight whose screen has finished.
#
# Environment:
#   RUN_ROOT   screen directory root                       (default: logs/sprint8_screens)
#   SOUP_ROOT  where the soups are written                 (default: logs/soups)
#   KEEP_SOUP=1 keep each soup after its screen            (default: deleted)
#   STOP_AT    "YYYY-MM-DD HH:MM": exit 5 rather than start a screen that could end after
#              this time; rerun to continue. A STOP_AT= argument wins.
#   DRY_RUN=1  print the commands instead of running them
#   PYTHON     python interpreter                          (default: python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH="${PWD}:${PWD}/third_party/SuperVLAD:${PYTHONPATH:-}"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

[ "$#" -ge 5 ] || { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0" >&2; exit 2; }
MODEL=$1 KEY=$2 A=$3 B=$4
shift 4
fail() { echo "$*" >&2; exit 2; }

STOP_AT="${STOP_AT:-}"
ALPHAS=()
for argument in "$@"; do
  case "${argument}" in
    STOP_AT=*) STOP_AT=${argument#STOP_AT=} ;;
    *)
      [[ "${argument}" =~ ^-?[0-9]*\.?[0-9]+$ ]] || fail "not a mixing weight: ${argument}"
      ALPHAS+=("${argument}")
      ;;
  esac
done
[ "${#ALPHAS[@]}" -gt 0 ] || fail "no mixing weight given"

case "${MODEL}" in
  supervlad) MODEL_FLAGS=("${SUPERVLAD_TRAIN_FLAGS[@]}") INFER_BATCH=16 INITIAL_CLEAN_R1=88.10 SCREEN_MINUTES=12 ;;
  boq) MODEL_FLAGS=("${BOQ_TRAIN_FLAGS[@]}") INFER_BATCH=8 INITIAL_CLEAN_R1=91.35 SCREEN_MINUTES=13 ;;
  *) fail "unknown model: ${MODEL}" ;;
esac

if [ -n "${STOP_AT}" ]; then
  STOP_BY=$(date -d "${STOP_AT}" +%s 2>/dev/null) || fail "STOP_AT is not a time: ${STOP_AT}"
  [ "${STOP_BY}" -gt "$(date +%s)" ] || fail "STOP_AT ${STOP_AT} has passed"
fi
PYTHON=${PYTHON:-python}
RUN_ROOT=${RUN_ROOT:-logs/sprint8_screens}
SOUP_ROOT=${SOUP_ROOT:-logs/soups}

run() {
  if [ "${DRY_RUN:-0}" = "1" ]; then
    printf '+'; printf ' %q' "$@"; printf '\n'
  else
    "$@"
  fi
}

for index in "${!ALPHAS[@]}"; do
  ALPHA=${ALPHAS[${index}]}
  NAME="${KEY}_a${ALPHA}"
  RUN_DIR="${RUN_ROOT}/${NAME}"
  SOUP="${SOUP_ROOT}/${NAME}.pth"

  state=""
  [ ! -f "${RUN_DIR}/run_status.json" ] || \
    state="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' "${RUN_DIR}/run_status.json")"
  case "${state}" in
    completed | early_stopped | collapse_aborted)
      echo "=== SKIP ${NAME}: ${state}"
      continue
      ;;
  esac

  # Checked only between screens: a screen that has started always finishes.
  if [ -n "${STOP_AT}" ] && [ "$(( $(date +%s) + SCREEN_MINUTES * 60 ))" -gt "${STOP_BY}" ]; then
    echo "=== STOP before ${NAME}: a screen could end after ${STOP_AT}; left: ${ALPHAS[*]:${index}}"
    exit 5
  fi

  echo "=== SCREEN ${NAME}"
  # The attack is never run; it makes validation measure the rank-PGD budgets as in training.
  run "${PYTHON}" scripts/soup.py "${A}" "${B}" "${ALPHA}" "${SOUP}"
  run "${PYTHON}" scripts/validate_only.py "${NAME}" --initial_clean_r1 "${INITIAL_CLEAN_R1}" \
    "${MODEL_FLAGS[@]}" "${TRAIN_RECIPE_FLAGS[@]}" --infer_batch_size="${INFER_BATCH}" \
    --freeze_te=0 --lr=1e-5 --num_epochs=1 --patience=6 --seed=0 \
    --run_dir="${RUN_DIR}" --save_dir="${NAME}" --resume="${SOUP}" --resume_model_only \
    --attack "RankLinfAttack(model, epsilon=0.0685, steps=5)" --defense_loss=listwise
  [ "${KEEP_SOUP:-0}" = "1" ] || run rm -f "${SOUP}"
done
