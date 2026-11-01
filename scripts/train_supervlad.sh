#!/usr/bin/env bash
# Full-length SuperVLAD runs: ARAFT, plain adversarial training, the clean twin and the
# ablations of Table 4 and Fig. 2.
set -euo pipefail
T="$(dirname "$0")/train.sh"

"$T" supervlad_araft_s0 supervlad araft 0
"$T" supervlad_plain_at_s0 supervlad plain_at 0
"$T" supervlad_clean_s0 supervlad clean 0
"$T" supervlad_araft_s1 supervlad araft 1
# Ablations
"$T" supervlad_araft_closest_s0 supervlad araft 0 POSITIVE=closest
"$T" supervlad_araft_closest_s1 supervlad araft 1 POSITIVE=closest
"$T" supervlad_araft_closest_hinge_s0 supervlad araft 0 POSITIVE=closest LOSS=hinge
"$T" supervlad_araft_closest_aw0_s0 supervlad araft 0 POSITIVE=closest AW=0
"$T" supervlad_araft_aw2_s0 supervlad araft 0 AW=2
