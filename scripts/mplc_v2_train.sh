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
#   MPLC_V2_ALIGN_WEIGHT  anchor weight λ_anchor (mplc arm)     (default: 1.0)
#   MPLC_V2_TRAIN_EPS     rank-attack L-inf budget, normalized  (default: 0.0685, <=4/255)
#   MPLC_V2_FREEZE_TE     frozen DINOv2 blocks, both arms       (default: 8; <8 adds
#                         --grad_checkpointing)
#   MPLC_V2_CLEAN_BUDGETS clean-drop reporting budgets, points  (default: "1 3 5")
#   MPLC_V2_DRY_RUN=1     print the commands without running them
#   PYTHON                python interpreter                    (default: python)
#
# The mplc arm anchors the attacked descriptor to the frozen initial model
# (--align_target=initial, spec 2026-09-25 D1) and both arms select the most robust epoch,
# keeping best_model_budget<b>.pth per clean budget (D2). Non-default mplc hyperparameters
# (TAU/K/POOL/RAMP_EPOCHS/ABORT_KNN/ALIGN_WEIGHT/TRAIN_EPS) and a non-default FREEZE_TE (both
# arms, so each depth has a matched clean twin) are named in the save_dir, e.g.
# mplc_v2_supervlad_mplc_aw10_fte4_s<seed>. Only all-default runs are auto-discovered by
# scripts/mplc_v2_eval.sh; evaluate others via --model.
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
ALIGN_WEIGHT=${MPLC_V2_ALIGN_WEIGHT:-1.0}
TRAIN_EPS=${MPLC_V2_TRAIN_EPS:-0.0685}
FREEZE_TE=${MPLC_V2_FREEZE_TE:-8}
CLEAN_BUDGETS=${MPLC_V2_CLEAN_BUDGETS:-"1 3 5"}
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
  --attack "RankLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)"
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
  [ "${ALIGN_WEIGHT}" = "1.0" ] || suffix="${suffix}_aw${ALIGN_WEIGHT}"
  [ "${TRAIN_EPS}" = "0.0685" ] || suffix="${suffix}_eps${TRAIN_EPS}"
  echo "${suffix}"
}

# Backbone depth applies to both arms, so each freeze_te gets its own matched clean twin.
fte_suffix() {
  [ "${FREEZE_TE}" = "8" ] || echo "_fte${FREEZE_TE}"
}

# shellcheck disable=SC2206 # CLEAN_BUDGETS is a space-separated list by design.
COMMON_FLAGS=(
  --freeze_te="${FREEZE_TE}"
  --selection_clean_budgets ${CLEAN_BUDGETS}
  --val_rank_epsilons 0.0342 0.0685
)
if [ "${FREEZE_TE}" -lt 8 ]; then
  COMMON_FLAGS+=(--grad_checkpointing)
fi

for arm in ${ARMS}; do
  name="mplc_v2_supervlad_${arm}$(fte_suffix)_s${SEED}"
  if [ "${arm}" = "mplc" ]; then
    name="mplc_v2_supervlad_mplc$(mplc_override_suffix)$(fte_suffix)_s${SEED}"
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
        --align_target=initial
        --adv_align_weight="${ALIGN_WEIGHT}"
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
    "${COMMON_FLAGS[@]}" \
    --seed="${SEED}" \
    --save_dir="${name}" \
    "${arm_flags[@]}"
done
