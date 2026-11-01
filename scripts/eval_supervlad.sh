#!/usr/bin/env bash
# SuperVLAD test-set evaluation (Tables 2-4 and 6, Fig. 3, Sect. 5.4). Needs the runs of
# scripts/train_supervlad.sh. Each model is evaluated at its most robust checkpoint
# (best_model.pth); ARAFT's is also its budget-5 checkpoint.
set -euo pipefail
E="$(dirname "$0")/eval.sh"
R=${RUN_ROOT:-logs/paper}
PRETRAINED=checkpoints/SuperVLAD_base.pth
EPS_HEADLINE=(0.03425 0.0685)            # 2/255, 4/255
EPS_GRID=(0.01712 0.03425 0.0685 0.137)  # 1, 2, 4, 8 /255

for ds in msls pitts30k sped nordland; do
  # Headline attack (Tables 2, 3) and robustness curves
  "$E" supervlad apgd "$ds" pretrained "$PRETRAINED" "${EPS_HEADLINE[@]}"
  "$E" supervlad apgd "$ds" araft "$R/supervlad_araft_s0/best_model.pth" "${EPS_HEADLINE[@]}"
  "$E" supervlad apgd "$ds" plain_at "$R/supervlad_plain_at_s0/best_model.pth" "${EPS_HEADLINE[@]}"
  "$E" supervlad pgd "$ds" pretrained "$PRETRAINED" "${EPS_GRID[@]}"
  "$E" supervlad pgd "$ds" araft "$R/supervlad_araft_s0/best_model.pth" "${EPS_GRID[@]}"
  "$E" supervlad pgd "$ds" plain_at "$R/supervlad_plain_at_s0/best_model.pth" "${EPS_GRID[@]}"
  # Closest-positive ablation on every test set (Table 4, paired tests)
  "$E" supervlad apgd "$ds" araft_closest "$R/supervlad_araft_closest_s0/best_model.pth" 0.0685
  # Clean twin at its last checkpoint; only its clean R@1 is used (Table 6, Fig. 3)
  "$E" supervlad pgd "$ds" clean_twin "$R/supervlad_clean_s0/last_model.pth" 0.01712
done

# Other ablations (Table 4)
for ds in msls nordland; do
  "$E" supervlad apgd "$ds" araft_closest_hinge "$R/supervlad_araft_closest_hinge_s0/best_model.pth" 0.0685
  "$E" supervlad apgd "$ds" araft_closest_aw0 "$R/supervlad_araft_closest_aw0_s0/best_model.pth" 0.0685
  "$E" supervlad apgd "$ds" araft_aw2 "$R/supervlad_araft_aw2_s0/best_model.pth" 0.0685
done

# Gradient-masking checks (Sect. 5.4): more steps, restarts, and a ~32/255 budget
"$E" supervlad apgd500 msls araft "$R/supervlad_araft_s0/best_model.pth" 0.0685
"$E" supervlad apgd_r3 msls araft "$R/supervlad_araft_s0/best_model.pth" 0.0685
"$E" supervlad pgd msls araft_eps32 "$R/supervlad_araft_s0/best_model.pth" 0.548
