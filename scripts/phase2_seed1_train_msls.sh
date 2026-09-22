#!/usr/bin/env bash
# Train the winning Phase 2 configuration with seed 1, then evaluate it on MSLS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON=${PYTHON:-python}
OUTPUT_ROOT=${PHASE2_SEED1_OUTPUT_ROOT:-output/phase2/winner_mplc_s1}
EPSILONS=${PHASE2_SEED1_EPSILONS:-"0.01 0.1"}

PHASE2_WINNER=tau0.05_k1_pool0 \
PHASE2_SEEDS=1 \
PYTHON="${PYTHON}" \
"${SCRIPT_DIR}/phase2_supervlad.sh" final

checkpoint=""
for candidate in logs/phase2_supervlad_mplc_s1/*/best_model.pth; do
  [ -f "${candidate}" ] && checkpoint="${candidate}"
done
[ -n "${checkpoint}" ] || { echo "No completed Phase 2 seed-1 checkpoint found." >&2; exit 1; }

PHASE2_WINNER_CHECKPOINT="${checkpoint}" \
PHASE2_WINNER_TAG=mplc_s1 \
PHASE2_WINNER_DATASETS=msls \
PHASE2_WINNER_EPSILONS="${EPSILONS}" \
PHASE2_WINNER_OUTPUT_ROOT="${OUTPUT_ROOT}" \
PYTHON="${PYTHON}" \
"${SCRIPT_DIR}/phase2_winner_eval.sh"
