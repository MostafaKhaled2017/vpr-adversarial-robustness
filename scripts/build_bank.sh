#!/usr/bin/env bash
# Encode one image per GSV-Cities training place with the pretrained model into
# ${BANK_ROOT}/MODEL_f0.pt, the file train.py --negative_bank loads.
#
# Usage:
#   scripts/build_bank.sh MODEL
#
#   MODEL   supervlad | boq
#
# An existing bank file is kept and the build skipped.
#
# Environment:
#   BANK_ROOT bank directory              (default: logs/banks)
#   DRY_RUN=1 print the python command instead of running it
#   PYTHON    python interpreter          (default: python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH="${PWD}:${PWD}/third_party/SuperVLAD:${PYTHONPATH:-}"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

[ "$#" -eq 1 ] || { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0" >&2; exit 2; }
MODEL=$1
PYTHON=${PYTHON:-python}
OUT="${BANK_ROOT:-logs/banks}/${MODEL}_f0.pt"

case "${MODEL}" in
  supervlad) MODEL_FLAGS=("${SUPERVLAD_TRAIN_FLAGS[@]}") PRETRAINED=${SUPERVLAD_PRETRAINED} ;;
  boq) MODEL_FLAGS=("${BOQ_TRAIN_FLAGS[@]}") PRETRAINED=${BOQ_PRETRAINED} ;;
  *) echo "unknown model: ${MODEL}" >&2; exit 2 ;;
esac

if [ -f "${OUT}" ]; then
  echo "=== SKIP ${OUT}: exists"
  exit 0
fi

COMMAND=(
  "${PYTHON}" -m src.negative_bank
  "${MODEL_FLAGS[@]}"
  "${TRAIN_RECIPE_FLAGS[@]}"
  --resume="${PRETRAINED}"
  --out="${OUT}"
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '+'; printf ' %q' "${COMMAND[@]}"; printf '\n'
else
  "${COMMAND[@]}"
fi
