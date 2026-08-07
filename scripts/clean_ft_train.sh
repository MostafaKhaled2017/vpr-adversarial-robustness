#!/usr/bin/env bash
# Task 1.1 — matched clean-only fine-tuning control for SuperVLAD.
#
# Identical to scripts/super_vlad_train.sh in every respect except that no --attack is
# configured, so no adversarial examples are generated and the rank/align terms are zero.
# With an empty attack list src/cli.py resolves --selection_robust_weight to 0.0 and records
# checkpoint_selection_rule=clean_recall in training_config.yaml (there is no adversarial
# validation signal to select on).
#
# Usage:
#   scripts/clean_ft_train.sh --seed 0 --save_dir supervlad_clean_ft_seed0
#   scripts/clean_ft_train.sh                     # defaults from src/cli.py
#
# Extra arguments are forwarded to train.py verbatim.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

PYTHON=${PYTHON:-python3}

exec "${PYTHON}" train.py "${SUPERVLAD_BASE_FLAGS[@]}" "$@"
