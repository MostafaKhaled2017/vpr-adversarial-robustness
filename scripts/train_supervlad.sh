#!/usr/bin/env bash
set -euo pipefail
T="$(dirname "$0")/train.sh"

"$T" supervlad_araft_s0 supervlad araft 0
