#!/usr/bin/env bash
# Evaluate a boq checkpoint on all four test sets into the TAG results (default: araft).
# CHECKPOINT overrides the run checkpoint (default: the boq_<TAG>_s0 run), e.g. with a downloaded one.
set -euo pipefail
E="$(dirname "$0")/eval.sh"
TAG=${TAG:-araft}
CHECKPOINT=${CHECKPOINT:-${RUN_ROOT:-logs}/boq_${TAG}_s0/best_model.pth}
EPS_HEADLINE=(0.03425 0.0685)            # 2/255, 4/255
EPS_GRID=(0.01712 0.03425 0.0685 0.137)  # 1, 2, 4, 8 /255

for ds in msls pitts30k sped nordland; do
  "$E" boq apgd "$ds" "$TAG" "$CHECKPOINT" "${EPS_HEADLINE[@]}"
  "$E" boq pgd "$ds" "$TAG" "$CHECKPOINT" "${EPS_GRID[@]}"
done
