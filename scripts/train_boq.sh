#!/usr/bin/env bash
set -euo pipefail
T="$(dirname "$0")/train.sh"

"$T" boq_araft_s0 boq araft 0
"$T" boq_fare_s0 boq fare 0
