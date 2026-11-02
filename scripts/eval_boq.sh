#!/usr/bin/env bash
# Evaluate an ARAFT boq checkpoint on all four test sets. CHECKPOINT overrides the default
# run checkpoint, e.g. with a downloaded one.
set -euo pipefail
E="$(dirname "$0")/eval.sh"
CHECKPOINT=${CHECKPOINT:-${RUN_ROOT:-logs}/boq_araft_s0/best_model.pth}
EPS_HEADLINE=(0.03425 0.0685)            # 2/255, 4/255
EPS_GRID=(0.01712 0.03425 0.0685 0.137)  # 1, 2, 4, 8 /255

for ds in msls pitts30k sped nordland; do
  "$E" boq apgd "$ds" araft "$CHECKPOINT" "${EPS_HEADLINE[@]}"
  "$E" boq pgd "$ds" araft "$CHECKPOINT" "${EPS_GRID[@]}"
done
