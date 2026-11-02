#!/usr/bin/env bash
set -euo pipefail
E="$(dirname "$0")/eval.sh"
R=${RUN_ROOT:-logs}
PRETRAINED=checkpoints/boq_dinov2_12288.pth
EPS_HEADLINE=(0.0342 0.0685)
EPS_GRID=(0.01712 0.03425 0.0685 0.137)

for ds in msls pitts30k sped nordland; do
  "$E" boq apgd "$ds" pretrained "$PRETRAINED" "${EPS_HEADLINE[@]}"
  "$E" boq apgd "$ds" araft "$R/boq_araft_s0/best_model.pth" "${EPS_HEADLINE[@]}"
  "$E" boq apgd "$ds" plain_at "$R/boq_plain_at_s0/best_model.pth" "${EPS_HEADLINE[@]}"
  "$E" boq pgd "$ds" pretrained "$PRETRAINED" "${EPS_GRID[@]}"
  "$E" boq pgd "$ds" araft "$R/boq_araft_s0/best_model.pth" "${EPS_GRID[@]}"
  "$E" boq pgd "$ds" plain_at "$R/boq_plain_at_s0/best_model.pth" "${EPS_GRID[@]}"
done
