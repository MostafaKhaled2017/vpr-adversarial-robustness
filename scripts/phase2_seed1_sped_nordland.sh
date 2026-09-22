#!/usr/bin/env bash
# Evaluate the completed Phase 2 seed-1 winner on SPED and Nordland.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${PHASE2_SEED1_OUTPUT_ROOT:-output/phase2/winner_mplc_s1}
EPSILONS=${PHASE2_SEED1_EPSILONS:-"0.01 0.1"}

checkpoint=""
for candidate in logs/phase2_supervlad_mplc_s1/*/best_model.pth; do
  [ -f "${candidate}" ] && checkpoint="${candidate}"
done
[ -n "${checkpoint}" ] || {
  echo "No completed Phase 2 seed-1 checkpoint found. Run scripts/phase2_seed1_train_msls.sh first." >&2
  exit 1
}

PHASE2_WINNER_CHECKPOINT="${checkpoint}" \
PHASE2_WINNER_TAG=mplc_s1 \
PHASE2_WINNER_DATASETS="sped nordland" \
PHASE2_WINNER_EPSILONS="${EPSILONS}" \
PHASE2_WINNER_OUTPUT_ROOT="${OUTPUT_ROOT}" \
PYTHON="${PYTHON}" \
"${SCRIPT_DIR}/phase2_winner_eval.sh"
