#!/usr/bin/env bash
# MPLC v2: train the clean fine-tuned twin and the multi-positive, listwise,
# collapse-aware adversarially trained model for one seed. Checkpoint selection uses
# the batch-independent rank-PGD validation protocol (Task 6); the MPLC arm's training
# attacks include the rank-margin attack (Task 5) alongside the two perceptual attacks.
#
# Usage:
#   scripts/mplc_v2_train.sh                # seed 0: clean_ft, then mplc
#   MPLC_V2_SEED=1 scripts/mplc_v2_train.sh  # seed 1: clean_ft, then mplc
#
# Environment overrides:
#   MPLC_V2_SEED          training seed                          (default: 0)
#   MPLC_V2_ARMS          arms to train, space separated         (default: "clean_ft mplc")
#   MPLC_V2_TAU           listwise temperature (mplc arm)        (default: 0.05)
#   MPLC_V2_K             listwise top-K target (mplc arm)       (default: 1)
#   MPLC_V2_POOL          negative pool size (mplc arm)          (default: 0)
#   MPLC_V2_RAMP_EPOCHS   attack-hardness ramp (mplc arm)        (default: 5)
#   MPLC_V2_ABORT_KNN     collapse-abort threshold (mplc arm)    (default: 0.15)
#   MPLC_V2_DRY_RUN=1     print the commands without running them
#   PYTHON                python interpreter                    (default: python)
#
# When any of MPLC_V2_TAU/_K/_POOL/_RAMP_EPOCHS/_ABORT_KNN differs from its default, the
# mplc arm's save_dir gets a suffix naming the non-default values, e.g.
# mplc_v2_supervlad_mplc_tau0.01_s<seed>, so a differently configured rerun neither
# collides with, nor gets SKIPped by, a finished default run. With all defaults the
# save_dir is the plain mplc_v2_supervlad_mplc_s<seed> that scripts/mplc_v2_eval.sh
# auto-discovers; override runs must be evaluated explicitly via MPLC_V2_MODELS.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"

PYTHON=${PYTHON:-python}
SEED=${MPLC_V2_SEED:-0}
ARMS=${MPLC_V2_ARMS:-"clean_ft mplc"}
TAU=${MPLC_V2_TAU:-0.05}
K=${MPLC_V2_K:-1}
POOL=${MPLC_V2_POOL:-0}
RAMP_EPOCHS=${MPLC_V2_RAMP_EPOCHS:-5}
ABORT_KNN=${MPLC_V2_ABORT_KNN:-0.15}
DRY_RUN=${MPLC_V2_DRY_RUN:-0}

run() {
  echo "+ $*"
  if [ "${DRY_RUN}" != "1" ]; then
    "$@"
  fi
}

# The MPLC arm's training attacks: the two perceptual attacks shared with the Phase 1/2
# adversarial arm, plus the rank-margin attack (Task 5) that ascends the ranking
# objective directly against all positives.
MPLC_ATTACK_FLAGS=(
  --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)"
  --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)"
  --attack "RankLinfAttack(model, epsilon=0.1, steps=5)"
)

# A run is finished, and should be skipped, when its run_status.json records a terminal
# state. best_model.pth alone is not a reliable finished marker (it also exists mid-run).
finished_run_dir() {
  local name=$1 dir state
  for dir in "logs/${name}"/*/; do
    [ -f "${dir}run_status.json" ] || continue
    state="$("${PYTHON}" -c "import json,sys; print(json.load(open(sys.argv[1])).get('state', ''))" "${dir}run_status.json")"
    case "${state}" in
      early_stopped | completed | collapse_aborted)
        echo "${dir%/}"
        return 0
        ;;
    esac
  done
  return 1
}

# Non-default mplc-arm hyperparameters get named in the save_dir so a rerun with a
# different tau/k/pool/ramp/abort doesn't share a name with (and get SKIPped by, or
# silently evaluated as) a finished default run.
mplc_override_suffix() {
  local suffix=""
  [ "${TAU}" = "0.05" ] || suffix="${suffix}_tau${TAU}"
  [ "${K}" = "1" ] || suffix="${suffix}_k${K}"
  [ "${POOL}" = "0" ] || suffix="${suffix}_pool${POOL}"
  [ "${RAMP_EPOCHS}" = "5" ] || suffix="${suffix}_ramp${RAMP_EPOCHS}"
  [ "${ABORT_KNN}" = "0.15" ] || suffix="${suffix}_abort${ABORT_KNN}"
  echo "${suffix}"
}

for arm in ${ARMS}; do
  name="mplc_v2_supervlad_${arm}_s${SEED}"
  if [ "${arm}" = "mplc" ]; then
    name="mplc_v2_supervlad_mplc$(mplc_override_suffix)_s${SEED}"
  fi

  arm_flags=()
  case "${arm}" in
    clean_ft)
      arm_flags=()
      ;;
    mplc)
      arm_flags=(
        "${MPLC_ATTACK_FLAGS[@]}"
        --multi_positive
        --defense_loss=listwise
        --listwise_tau="${TAU}"
        --listwise_k="${K}"
        --negative_pool_size="${POOL}"
        --attack_ramp_epochs="${RAMP_EPOCHS}"
        --collapse_abort_knn="${ABORT_KNN}"
      )
      ;;
    *)
      echo "unknown arm: ${arm}" >&2
      exit 2
      ;;
  esac

  finished_dir=""
  if finished_dir="$(finished_run_dir "${name}")"; then
    echo "=== SKIP ${name}: finished at ${finished_dir}"
    continue
  fi

  echo "=== TRAIN ${name}"
  run "${PYTHON}" train.py \
    "${SUPERVLAD_RECIPE_FLAGS[@]}" \
    "${SUPERVLAD_INIT_FLAGS[@]}" \
    --validation_protocol=rank_pgd \
    --seed="${SEED}" \
    --save_dir="${name}" \
    "${arm_flags[@]}"
done
