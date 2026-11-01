#!/usr/bin/env bash
# Full-length BoQ runs with the settings selected on SuperVLAD.
set -euo pipefail
T="$(dirname "$0")/train.sh"

"$T" boq_araft_s0 boq araft 0
"$T" boq_plain_at_s0 boq plain_at 0
