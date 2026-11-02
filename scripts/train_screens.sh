#!/usr/bin/env bash
set -euo pipefail
T="$(dirname "$0")/train.sh"
S=(POSITIVE=closest SCREEN=1)

# Attack set, 8 frozen blocks
"$T" screen_supervlad_araft_closest_mixall_fte8_s0 supervlad araft 0 "${S[@]}" MIX=all FTE=8
"$T" screen_supervlad_araft_closest_fte8_s0 supervlad araft 0 "${S[@]}" FTE=8
# Depth x learning rate
"$T" screen_supervlad_araft_closest_fte8_lr3e-6_s0 supervlad araft 0 "${S[@]}" FTE=8 LR=3e-6
"$T" screen_supervlad_araft_closest_fte4_s0 supervlad araft 0 "${S[@]}" FTE=4
"$T" screen_supervlad_araft_closest_fte4_lr3e-6_s0 supervlad araft 0 "${S[@]}" FTE=4 LR=3e-6
"$T" screen_supervlad_araft_closest_s0 supervlad araft 0 "${S[@]}"
"$T" screen_supervlad_araft_closest_lr3e-6_s0 supervlad araft 0 "${S[@]}" LR=3e-6
# Anchor weight (1.0 is the screen above)
"$T" screen_supervlad_araft_closest_aw0_s0 supervlad araft 0 "${S[@]}" AW=0
"$T" screen_supervlad_araft_closest_aw0.1_s0 supervlad araft 0 "${S[@]}" AW=0.1
"$T" screen_supervlad_araft_closest_aw10_s0 supervlad araft 0 "${S[@]}" AW=10
# Re-check of the attack set at the final depth and lr
"$T" screen_supervlad_araft_closest_mixall_s0 supervlad araft 0 "${S[@]}" MIX=all
# Baseline learning rate
"$T" screen_supervlad_plain_at_s0 supervlad plain_at 0 SCREEN=1
"$T" screen_supervlad_plain_at_lr3e-6_s0 supervlad plain_at 0 SCREEN=1 LR=3e-6
