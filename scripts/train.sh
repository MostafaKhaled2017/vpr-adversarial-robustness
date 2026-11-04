#!/usr/bin/env bash
# Train one run into ${RUN_ROOT}/NAME, or resume it.
#
# Usage:
#   scripts/train.sh NAME MODEL METHOD SEED [KEY=VALUE ...]
#
#   MODEL   supervlad | boq
#   METHOD  araft     ARAFT: listwise rank loss + anchor to the pretrained model
#           fare      FARE: clean loss + anchor loss, no rank loss, against an attack pushing
#                     the descriptor away from the pretrained model's
#           plain_at  listwise rank loss against the rank attack, no anchor
#   KEY=VALUE overrides:
#     EPOCHS=<int>                training epochs, all methods                   (default: 100)
#     FTE=<int>                   frozen DINOv2 blocks, all methods              (default: 0)
#     LR=<float>                  learning rate, all methods                     (default: 1e-5)
#     STOP_AT="YYYY-MM-DD HH:MM"  pause (exit 5) rather than start an epoch that, with the
#                                 final test, could end after this time; rerun to resume.
#                                 A session's first epoch always starts. All methods
#                                                                                (default: none)
#     POSITIVE=hardest|closest    protected positive, araft and plain_at         (default: hardest)
#     LOSS=listwise|hinge         rank loss, araft and plain_at                  (default: listwise)
#     LRW=<float>                 rank loss weight, araft and plain_at           (default: 1.0)
#     BANK=0|1                    negatives from ${BANK_ROOT}/MODEL_f0.pt
#                                 (scripts/build_bank.sh), araft and plain_at    (default: 0)
#     NEG_BANK=<int>              bank negatives per query, needs BANK=1         (default: 5)
#     BANK_DEVICE=cuda|cpu        device holding the bank, needs BANK=1          (default: cuda)
#     ATTACK=rank|shift|combined  training attack, araft only                    (default: rank)
#     BETA=<float>                rank weight of ATTACK=combined, araft only     (default: 1.0)
#     ANCHOR=iso|rel|both         anchor kind, araft only; rel and both
#                                 need BANK=1                                    (default: iso)
#     REL_M=<int>                 relational anchor neighbours, needs
#                                 ANCHOR=rel|both                                (default: 32)
#     REL_T=<float>               relational anchor temperature, needs
#                                 ANCHOR=rel|both                                (default: 0.1)
#     DRIFT=<float>               clean-drift weight, araft and fare             (default: 0)
#     AW=<float>                  anchor weight lambda_anc, araft and fare       (default: 1.0)
#
# A rerun skips a finished run (terminal state in run_status.json) and resumes a stopped one
# from its last checkpoint.
#
# Environment:
#   RUN_ROOT  run directory root          (default: logs)
#   BANK_ROOT bank directory              (default: logs/banks)
#   STOP_AT   as the setting; a STOP_AT= argument wins
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
POSITIVE=hardest LOSS=listwise AW=1.0 FTE=0 LR=1e-5 EPOCHS=100 ATTACK=rank BANK=0
STOP_AT="${STOP_AT:-}" LRW="" BETA="" DRIFT="" NEG_BANK="" BANK_DEVICE="" ANCHOR="" REL_M="" REL_T=""
# Exit unless METHOD is one of the arguments.
only() {
  local methods="$*"
  case " ${methods} " in
    *" ${METHOD} "*) ;;
    *) echo "${setting%%=*} applies to ${methods// / and } only" >&2; exit 2 ;;
  esac
}
for setting in "$@"; do
  case "${setting}" in
    EPOCHS=* | FTE=* | LR=* | STOP_AT=*) ;;
    POSITIVE=* | LOSS=* | LRW=* | BANK=* | NEG_BANK=* | BANK_DEVICE=*) only araft plain_at ;;
    ATTACK=* | BETA=* | ANCHOR=* | REL_M=* | REL_T=*) only araft ;;
    DRIFT=* | AW=*) only araft fare ;;
    *) echo "unknown setting: ${setting}" >&2; exit 2 ;;
  esac
  declare "${setting}"
done
fail() { echo "$*" >&2; exit 2; }
case "${BANK}" in 0 | 1) ;; *) fail "BANK must be 0 or 1" ;; esac
[ "${BANK}" = 1 ] || [ -z "${NEG_BANK}${BANK_DEVICE}" ] || fail "NEG_BANK and BANK_DEVICE need BANK=1"
case "${ANCHOR}" in
  "" | iso) ;;
  rel | both) [ "${BANK}" = 1 ] || fail "ANCHOR=${ANCHOR} needs BANK=1" ;;
  *) fail "unknown ANCHOR: ${ANCHOR}" ;;
esac
case "${ANCHOR}" in rel | both) ;; *) [ -z "${REL_M}${REL_T}" ] || fail "REL_M and REL_T need ANCHOR=rel or both" ;; esac
[ "${ATTACK}" = combined ] || [ -z "${BETA}" ] || fail "BETA needs ATTACK=combined"
if [ -n "${STOP_AT}" ]; then
  STOP_BY=$(date -d "${STOP_AT}" +%s 2>/dev/null) || fail "STOP_AT is not a time: ${STOP_AT}"
  [ "${STOP_BY}" -gt "$(date +%s)" ] || fail "STOP_AT ${STOP_AT} has passed"
fi
PYTHON=${PYTHON:-python}
RUN_DIR="${RUN_ROOT:-logs}/${NAME}"
TRAIN_EPS=0.0685  # 4/255

case "${MODEL}" in
  supervlad) MODEL_FLAGS=("${SUPERVLAD_TRAIN_FLAGS[@]}") PRETRAINED=${SUPERVLAD_PRETRAINED} ;;
  boq) MODEL_FLAGS=("${BOQ_TRAIN_FLAGS[@]}") PRETRAINED=${BOQ_PRETRAINED} ;;
  *) echo "unknown model: ${MODEL}" >&2; exit 2 ;;
esac
MODEL_FLAGS+=(--freeze_te="${FTE}")
[ "${FTE}" -ge 8 ] || MODEL_FLAGS+=(--grad_checkpointing)

case "${ATTACK}" in
  rank) ATTACK_STRING="RankLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)" ;;
  shift) ATTACK_STRING="EmbeddingShiftLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)" ;;
  combined) ATTACK_STRING="CombinedShiftRankLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5, rank_weight=${BETA:-1.0})" ;;
  *) fail "unknown ATTACK: ${ATTACK}" ;;
esac

case "${METHOD}" in
  araft | plain_at)
    METHOD_FLAGS=(
      --attack "${ATTACK_STRING}"
      --defense_loss="${LOSS}"
      --listwise_tau=0.05
      --listwise_k=1
      --attack_ramp_epochs=5
      --collapse_abort_knn=0.15
    )
    if [ "${METHOD}" = araft ]; then
      METHOD_FLAGS+=(--align_target=initial --adv_align_weight="${AW}")
    else
      METHOD_FLAGS+=(--adv_align_weight=0)
    fi
    case "${POSITIVE}" in
      hardest) ;;
      closest) METHOD_FLAGS+=(--multi_positive) ;;
      *) echo "unknown POSITIVE: ${POSITIVE}" >&2; exit 2 ;;
    esac
    ;;
  fare)
    METHOD_FLAGS=(
      --attack "EmbeddingShiftLinfAttack(model, epsilon=${TRAIN_EPS}, steps=5)"
      --adv_loss_weight=0
      --align_target=initial
      --adv_align_weight="${AW}"
      --attack_ramp_epochs=5
      --collapse_abort_knn=0.15
    )
    ;;
  *) echo "unknown method: ${METHOD}" >&2; exit 2 ;;
esac

# Flags of settings that were given; none by default.
EXTRA_FLAGS=()
[ -z "${LRW}" ] || EXTRA_FLAGS+=(--adv_loss_weight="${LRW}")
[ -z "${STOP_AT}" ] || EXTRA_FLAGS+=(--stop_by="${STOP_BY}")
[ -z "${DRIFT}" ] || EXTRA_FLAGS+=(--clean_drift_weight="${DRIFT}")
[ "${BANK}" = 0 ] || EXTRA_FLAGS+=(--negative_bank="${BANK_ROOT:-logs/banks}/${MODEL}_f0.pt")
[ -z "${NEG_BANK}" ] || EXTRA_FLAGS+=(--bank_negatives="${NEG_BANK}")
[ -z "${BANK_DEVICE}" ] || EXTRA_FLAGS+=(--bank_device="${BANK_DEVICE}")
case "${ANCHOR}" in
  iso) EXTRA_FLAGS+=(--anchor_kind=isotropic) ;;
  rel) EXTRA_FLAGS+=(--anchor_kind=relational) ;;
  both) EXTRA_FLAGS+=(--anchor_kind=both) ;;
esac
[ -z "${REL_M}" ] || EXTRA_FLAGS+=(--relational_neighbours="${REL_M}")
[ -z "${REL_T}" ] || EXTRA_FLAGS+=(--relational_temperature="${REL_T}")

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
  --num_epochs="${EPOCHS}"
  --patience=6
  --lr_plateau_patience=3
  "${INIT_FLAGS[@]}"
  --run_dir="${RUN_DIR}"
  --seed="${SEED}"
  --save_dir="${NAME}"
  "${METHOD_FLAGS[@]}"
  "${EXTRA_FLAGS[@]}"
)
if [ "${DRY_RUN:-0}" = "1" ]; then
  printf '+'; printf ' %q' "${COMMAND[@]}"; printf '\n'
else
  "${COMMAND[@]}"
fi
