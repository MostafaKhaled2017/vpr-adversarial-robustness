#!/usr/bin/env bash
# Train one run into ${RUN_ROOT}/NAME, or resume it.
#
# Usage:
#   scripts/train.sh NAME MODEL METHOD SEED [KEY=VALUE ...]
#
#   MODEL   supervlad | boq
#   METHOD  araft     ARAFT: listwise rank loss + anchor to the pretrained model
#           plain_at  plain adversarial training: hinge rank loss, no anchor
#           clean     clean fine-tuning without attacks (the clean twin)
#   KEY=VALUE overrides, ARAFT ablations and screens only:
#     POSITIVE=hardest|closest  protected positive              (default: hardest)
#     LOSS=listwise|hinge       rank loss                       (default: listwise)
#     AW=<float>                anchor weight lambda_anc        (default: 1.0)
#     MIX=linf|all              training attacks; "all" adds two perceptual attacks
#                                                               (default: linf)
#     FTE=<int>                 frozen DINOv2 blocks            (default: 0)
#     LR=<float>                learning rate                   (default: 1e-5)
#     SCREEN=1                  12-epoch screen(default: 0, full run)
#
# A rerun skips a finished run (terminal state in run_status.json) and resumes a stopped one
# from its last checkpoint.
#
# Environment:
#   RUN_ROOT  run directory root          (default: logs/paper)
#   DRY_RUN=1 print the train.py command instead of running it
#   PYTHON    python interpreter          (default: python)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
export PYTHONPATH="${PWD}:${PWD}/third_party/SuperVLAD:${PYTHONPATH:-}"
# shellcheck source=scripts/common.sh
source "${SCRIPT_DIR}/common.sh"

[ "$#" -ge 4 ] || { sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p}' "$0" >&2; exit 2; }
NAME=$1 MODEL=$2 METHOD=$3 SEED=$4
shift 4
POSITIVE=hardest LOSS=listwise AW=1.0 MIX=linf FTE=0 LR=1e-5 SCREEN=0
for setting in "$@"; do
  case "${setting}" in
    POSITIVE=* | LOSS=* | AW=* | MIX=* | FTE=* | LR=* | SCREEN=*) declare "${setting}" ;;
    *) echo "unknown setting: ${setting}" >&2; exit 2 ;;
  esac
done
PYTHON=${PYTHON:-python}
RUN_DIR="${RUN_ROOT:-logs/paper}/${NAME}"
TRAIN_EPS=0.0685  # 4/255

case "${MODEL}" in
  supervlad) MODEL_FLAGS=("${SUPERVLAD_TRAIN_FLAGS[@]}") PRETRAINED=${SUPERVLAD_PRETRAINED} ;;
  boq) MODEL_FLAGS=("${BOQ_TRAIN_FLAGS[@]}") PRETRAINED=${BOQ_PRETRAINED} ;;
  *) echo "unknown model: ${MODEL}" >&2; exit 2 ;;
esac
MODEL_FLAGS+=(--freeze_te="${FTE}")
[ "${FTE}" -ge 8 ] || MODEL_FLAGS+=(--grad_checkpointing)

# Screens keep the recipe's patience; with 4 validations it never stops a screen early.
if [ "${SCREEN}" = "1" ]; then
  SCHEDULE_FLAGS=(--num_epochs=12 --patience=8 --lr_plateau_patience=5)
else
  SCHEDULE_FLAGS=(--num_epochs=100 --patience=6 --lr_plateau_patience=3)
fi

RANK_ATTACK="RankLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)"
case "${METHOD}" in
  araft)
    case "${MIX}" in
      linf) METHOD_FLAGS=(--attack "${RANK_ATTACK}") ;;
      all)
        METHOD_FLAGS=(
          --attack "FastLagrangePerceptualAttack(model, bound=0.1, num_iterations=5)"
          --attack "PerceptualPGDAttack(model, bound=0.1, num_iterations=5)"
          --attack "${RANK_ATTACK}"
        )
        ;;
      *) echo "unknown MIX: ${MIX}" >&2; exit 2 ;;
    esac
    METHOD_FLAGS+=(
      --defense_loss="${LOSS}"
      --listwise_tau=0.05
      --listwise_k=1
      --attack_ramp_epochs=5
      --collapse_abort_knn=0.15
      --align_target=initial
      --adv_align_weight="${AW}"
    )
    case "${POSITIVE}" in
      hardest) ;;
      closest) METHOD_FLAGS+=(--multi_positive) ;;
      *) echo "unknown POSITIVE: ${POSITIVE}" >&2; exit 2 ;;
    esac
    ;;
  plain_at)
    METHOD_FLAGS=(
      --attack "${RANK_ATTACK}"
      --defense_loss=hinge
      --adv_align_weight=0
      --attack_ramp_epochs=5
      --collapse_abort_knn=0.15
    )
    ;;
  clean) METHOD_FLAGS=() ;;
  *) echo "unknown method: ${METHOD}" >&2; exit 2 ;;
esac

INIT_FLAGS=(--resume="${PRETRAINED}" --resume_model_only)
if [ -d "${RUN_DIR}" ]; then
  state=""
  [ ! -f "${RUN_DIR}/run_status.json" ] || \
    state="$("${PYTHON}" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("state", ""))' "${RUN_DIR}/run_status.json")"
  case "${state}" in
    completed | early_stopped | collapse_aborted)
      echo "=== SKIP ${NAME}: ${state}"
      exit 0
      ;;
  esac
  if [ -f "${RUN_DIR}/last_model.pth" ]; then
    echo "=== RESUME ${NAME} from ${RUN_DIR}/last_model.pth"
    INIT_FLAGS=(--resume="${RUN_DIR}/last_model.pth" --continue)
  elif [ -f "${RUN_DIR}/initial_validation_model.pth" ]; then
    echo "=== RESUME ${NAME} from ${RUN_DIR}/initial_validation_model.pth"
    INIT_FLAGS=(--resume="${RUN_DIR}/initial_validation_model.pth" --continue)
  elif [ -n "$(ls -A "${RUN_DIR}")" ]; then
    # Stopped before its first checkpoint: keep the partial directory and start over.
    stale="${RUN_DIR}.stale-$(date +%Y%m%d-%H%M%S)"
    echo "=== RESTART ${NAME}: no checkpoint, moved to ${stale}"
    [ "${DRY_RUN:-0}" = "1" ] || mv "${RUN_DIR}" "${stale}"
  fi
else
  echo "=== TRAIN ${NAME}"
fi

COMMAND=(
  "${PYTHON}" train.py
  "${MODEL_FLAGS[@]}"
  "${TRAIN_RECIPE_FLAGS[@]}"
  --lr="${LR}"
  "${SCHEDULE_FLAGS[@]}"
  "${INIT_FLAGS[@]}"
  --run_dir="${RUN_DIR}"
  --seed="${SEED}"
  --save_dir="${NAME}"
  "${METHOD_FLAGS[@]+"${METHOD_FLAGS[@]}"}"
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '+'; printf ' %q' "${COMMAND[@]}"; printf '\n'
else
  "${COMMAND[@]}"
fi
