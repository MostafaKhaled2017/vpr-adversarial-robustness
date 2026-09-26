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
#   MPLC_V2_ARMS          arms to train: clean_ft mplc plain_at fare (default: "clean_ft mplc")
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
#   MPLC_V2_LR            learning rate, both arms              (default: 1e-5)
#   MPLC_V2_NUM_EPOCHS    maximum epochs, both arms             (default: 100; 12 for screens)
#   MPLC_V2_VAL_EVERY     validate every N epochs and after the last, both arms
#                                                               (default: 1; 3 for screens)
#   MPLC_V2_ATTACK_MIX    training attacks (mplc arm): "all" = two perceptual attacks +
#                         rank L-inf, one sampled per step; "linf" = rank L-inf only
#                                                               (default: all)
#   MPLC_V2_DEFENSE_LOSS  defense objective (mplc arm): listwise | hinge (default: listwise)
#   MPLC_V2_MULTI_POSITIVE  1 = target every positive, 0 = hardest one only (mplc arm)
#                                                               (default: 1)
#   MPLC_V2_RUN_ROOT      resumable mode: each run lives in the fixed directory
#                         <root>/<save_dir name>; a stopped run resumes from its last
#                         checkpoint, a finished one is skipped   (default: unset)
#   MPLC_V2_DRY_RUN=1     print the commands without running them
#   PYTHON                python interpreter                    (default: python)
#
# The mplc arm anchors the attacked descriptor to the frozen initial model
# (--align_target=initial, spec 2026-09-25 D1) and both arms select the most robust epoch,
# keeping best_model_budget<b>.pth per clean budget (D2). Non-default mplc hyperparameters
# (TAU/K/POOL/RAMP_EPOCHS/ABORT_KNN/ALIGN_WEIGHT/TRAIN_EPS/ATTACK_MIX/
# DEFENSE_LOSS/MULTI_POSITIVE) and a non-default
# FREEZE_TE/LR/NUM_EPOCHS (both arms, so each setting has a matched clean twin) are named in
# the save_dir, e.g. mplc_v2_supervlad_mplc_aw10_mixlinf_fte4_lr3e-6_ep9_s<seed>. Only all-default runs are auto-discovered by
# scripts/mplc_v2_eval.sh; evaluate others via --model.
#
# Baselines (runbook §2.2): plain_at = hinge rank loss against rank L-inf only, single
# positive, no anchor; fare = anchor loss only against an L-inf attack that pushes the
# descriptor away from the frozen initial model's (FARE). Both share the MPLC arm's
# training eps, ramp and collapse abort.
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
LR=${MPLC_V2_LR:-1e-5}
NUM_EPOCHS=${MPLC_V2_NUM_EPOCHS:-100}
VAL_EVERY=${MPLC_V2_VAL_EVERY:-1}
ATTACK_MIX=${MPLC_V2_ATTACK_MIX:-all}
DEFENSE_LOSS=${MPLC_V2_DEFENSE_LOSS:-listwise}
MULTI_POSITIVE=${MPLC_V2_MULTI_POSITIVE:-1}
RUN_ROOT=${MPLC_V2_RUN_ROOT:-}
DRY_RUN=${MPLC_V2_DRY_RUN:-0}

run() {
  echo "+ $*"
  if [ "${DRY_RUN}" != "1" ]; then
    "$@"
  fi
}

# The MPLC arm's training attacks: the rank-margin attack (Task 5) that ascends the
# ranking objective directly against all positives, plus (mix "all") the two perceptual
# attacks shared with the Phase 1/2 adversarial arm.
case "${ATTACK_MIX}" in
  all)
    MPLC_ATTACK_FLAGS=(
      --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)"
      --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)"
      --attack "RankLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)"
    )
    ;;
  linf)
    MPLC_ATTACK_FLAGS=(--attack "RankLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)")
    ;;
  *)
    echo "unknown MPLC_V2_ATTACK_MIX: ${ATTACK_MIX} (expected all or linf)" >&2
    exit 2
    ;;
esac
case "${DEFENSE_LOSS}" in
  listwise | hinge) ;;
  *) echo "unknown MPLC_V2_DEFENSE_LOSS: ${DEFENSE_LOSS} (expected listwise or hinge)" >&2; exit 2 ;;
esac
case "${MULTI_POSITIVE}" in
  0 | 1) ;;
  *) echo "MPLC_V2_MULTI_POSITIVE must be 0 or 1, got: ${MULTI_POSITIVE}" >&2; exit 2 ;;
esac

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

# A run trained at another batch size or epoch length must never be skipped as finished or
# resumed: both are fixed once for all runs (SUPERVLAD_TRAIN_BATCH_SIZE and
# SUPERVLAD_BATCHES_PER_EPOCH in lib/supervlad_common.sh).
check_batch_size() {
  local config="$1/training_config.yaml" key wanted found
  [ -f "${config}" ] || return 0
  for key in "train_batch_size ${SUPERVLAD_TRAIN_BATCH_SIZE}" "batches_per_epoch ${SUPERVLAD_BATCHES_PER_EPOCH}"; do
    wanted="${key#* }"
    key="${key%% *}"
    found="$(sed -n "s/^${key}: //p" "${config}")"
    if [ -n "${found}" ] && [ "${found}" != "${wanted}" ]; then
      echo "$1 was trained with ${key} ${found}, but lib/supervlad_common.sh sets ${wanted}." >&2
      echo "Move that run aside or restore the setting; runs at different sizes must not be mixed." >&2
      exit 1
    fi
  done
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
  [ "${ATTACK_MIX}" = "all" ] || suffix="${suffix}_mix${ATTACK_MIX}"
  [ "${DEFENSE_LOSS}" = "listwise" ] || suffix="${suffix}_${DEFENSE_LOSS}"
  [ "${MULTI_POSITIVE}" = "1" ] || suffix="${suffix}_sp"
  echo "${suffix}"
}

# Backbone depth, lr and epochs apply to both arms, so each setting gets its own matched
# clean twin.
shared_suffix() {
  local suffix=""
  [ "${FREEZE_TE}" = "8" ] || suffix="${suffix}_fte${FREEZE_TE}"
  [ "${LR}" = "1e-5" ] || suffix="${suffix}_lr${LR}"
  [ "${NUM_EPOCHS}" = "100" ] || suffix="${suffix}_ep${NUM_EPOCHS}"
  echo "${suffix}"
}

# Baselines read only the attack budget, ramp and abort threshold (fare also λ_anchor), so
# only those are named: an MPLC-only setting in $CFG never renames a baseline run.
baseline_suffix() {
  local suffix=""
  [ "${RAMP_EPOCHS}" = "5" ] || suffix="${suffix}_ramp${RAMP_EPOCHS}"
  [ "${ABORT_KNN}" = "0.15" ] || suffix="${suffix}_abort${ABORT_KNN}"
  [ "$1" != "fare" ] || [ "${ALIGN_WEIGHT}" = "1.0" ] || suffix="${suffix}_aw${ALIGN_WEIGHT}"
  [ "${TRAIN_EPS}" = "0.0685" ] || suffix="${suffix}_eps${TRAIN_EPS}"
  echo "${suffix}"
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
# These override the recipe's --lr / --num_epochs (argparse keeps the last value).
[ "${LR}" = "1e-5" ] || COMMON_FLAGS+=(--lr="${LR}")
[ "${NUM_EPOCHS}" = "100" ] || COMMON_FLAGS+=(--num_epochs="${NUM_EPOCHS}")
[ "${VAL_EVERY}" = "1" ] || COMMON_FLAGS+=(--val_every="${VAL_EVERY}")

for arm in ${ARMS}; do
  case "${arm}" in
    mplc) name="mplc_v2_supervlad_mplc$(mplc_override_suffix)$(shared_suffix)_s${SEED}" ;;
    plain_at | fare) name="mplc_v2_supervlad_${arm}$(baseline_suffix "${arm}")$(shared_suffix)_s${SEED}" ;;
    *) name="mplc_v2_supervlad_${arm}$(shared_suffix)_s${SEED}" ;;
  esac

  arm_flags=()
  case "${arm}" in
    clean_ft)
      arm_flags=()
      ;;
    mplc)
      arm_flags=(
        "${MPLC_ATTACK_FLAGS[@]}"
        --defense_loss="${DEFENSE_LOSS}"
        --listwise_tau="${TAU}"
        --listwise_k="${K}"
        --negative_pool_size="${POOL}"
        --attack_ramp_epochs="${RAMP_EPOCHS}"
        --collapse_abort_knn="${ABORT_KNN}"
        --align_target=initial
        --adv_align_weight="${ALIGN_WEIGHT}"
      )
      [ "${MULTI_POSITIVE}" = "0" ] || arm_flags+=(--multi_positive)
      ;;
    plain_at)
      arm_flags=(
        --attack "RankLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)"
        --defense_loss=hinge
        --adv_align_weight=0
        --attack_ramp_epochs="${RAMP_EPOCHS}"
        --collapse_abort_knn="${ABORT_KNN}"
      )
      ;;
    fare)
      arm_flags=(
        --attack "EmbeddingShiftLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)"
        --adv_loss_weight=0
        --align_target=initial
        --adv_align_weight="${ALIGN_WEIGHT}"
        --attack_ramp_epochs="${RAMP_EPOCHS}"
        --collapse_abort_knn="${ABORT_KNN}"
      )
      ;;
    *)
      echo "unknown arm: ${arm}" >&2
      exit 2
      ;;
  esac

  init_flags=("${SUPERVLAD_INIT_FLAGS[@]}")
  if [ -n "${RUN_ROOT}" ]; then
    # Resumable mode: a fixed run directory, classified by what it already holds.
    managed_dir="${RUN_ROOT}/${name}"
    state="pending"
    checkpoint=""
    if [ -d "${managed_dir}" ]; then
      IFS=$'\t' read -r state checkpoint \
        <<<"$("${PYTHON}" -m src.phase2_run_group classify --config-dir "${managed_dir}")"
    fi
    case "${state}" in
      completed | early_stopped | collapse_aborted)
        check_batch_size "${managed_dir}"
        echo "=== SKIP ${name}: ${state} at ${managed_dir}"
        continue
        ;;
      resumable)
        check_batch_size "${managed_dir}"
        echo "=== RESUME ${name} from ${checkpoint}"
        init_flags=(--resume="${checkpoint}" --continue)
        ;;
      invalid)
        # Stopped before its first checkpoint (e.g. during the initial validation): keep
        # the partial directory for inspection and start over.
        stale="${managed_dir}.stale-$(date +%Y%m%d-%H%M%S)"
        echo "=== RESTART ${name}: no checkpoint in ${managed_dir}, moved to ${stale}"
        [ "${DRY_RUN}" = "1" ] || mv "${managed_dir}" "${stale}"
        ;;
      pending)
        echo "=== TRAIN ${name}"
        ;;
      *)
        echo "unexpected run state for ${managed_dir}: ${state}" >&2
        exit 1
        ;;
    esac
    init_flags+=(--run_dir="${managed_dir}")
  else
    finished_dir=""
    if finished_dir="$(finished_run_dir "${name}")"; then
      check_batch_size "${finished_dir}"
      echo "=== SKIP ${name}: finished at ${finished_dir}"
      continue
    fi
    echo "=== TRAIN ${name}"
  fi

  run "${PYTHON}" train.py \
    "${SUPERVLAD_RECIPE_FLAGS[@]}" \
    "${init_flags[@]}" \
    --validation_protocol=rank_pgd \
    "${COMMON_FLAGS[@]}" \
    --seed="${SEED}" \
    --save_dir="${name}" \
    "${arm_flags[@]}"
done
