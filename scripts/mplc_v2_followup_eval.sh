#!/usr/bin/env bash
# Follow-up evaluation: Step 3's protocol (scripts/mplc_v2_step3.sh) on the test sets, for the
# follow-up models trained by scripts/mplc_v2_followup_train.sh and three Step 2 runs.
#
# Models (src/followup_eval_plan.py, MODELS): the pretrained model, always, and these rows.
#   KEY            row                                group     run (FOLLOWUP_ROOT unless noted)
#   sp             ARAFT (hardest positive), seed 0   main      FOLLOWUP_STEP2_ROOT/mplc_v2_supervlad_mplc_mixlinf_sp_fte0_s0
#   sp_s1          ARAFT, seed 1                      main      mplc_v2_supervlad_mplc_mixlinf_sp_fte0_s1
#   sp_eps2        ARAFT, trained at 2/255            main      mplc_v2_supervlad_mplc_eps0.0342_mixlinf_sp_fte0_s0
#   fare_eps2      FARE, trained at 2/255             main      mplc_v2_supervlad_fare_eps0.0342_fte0_s0
#   plain_at_eps2  plain_at, trained at 2/255         main      mplc_v2_supervlad_plain_at_eps0.0342_fte0_s0
#   sp_hinge       hinge loss (hardest positive)      ablation  mplc_v2_supervlad_mplc_mixlinf_hinge_sp_fte0_s0
#   sp_aw2         anchor weight 2                    ablation  mplc_v2_supervlad_mplc_aw2_mixlinf_sp_fte0_s0
#   sp_aw3         anchor weight 3                    ablation  mplc_v2_supervlad_mplc_aw3_mixlinf_sp_fte0_s0
#   twin           clean twin (last_model.pth)        clean     FOLLOWUP_STEP2_ROOT/mplc_v2_supervlad_clean_ft_fte0_s0
# A row is evaluated once its run's run_status.json records a terminal state; until then it is
# "not trained yet" and skipped. Its checkpoints come from the run's validation_recalls.jsonl:
# Table A, the most robust trained epoch within 5 clean R@1 points (none: the pretrained model);
# Table B, the most robust trained epoch. One checkpoint (best_model.pth) when they are the same
# epoch, else best_model_budget5.pth for A and best_model.pth for B. The twin: last_model.pth,
# clean recall only. --model KEY:TABLES:PATH (Step 3's syntax) overrides a row's choice.
# Every checkpoint is recorded in FOLLOWUP_EVAL_ROOT/models.json as in Step 3: a later start adds
# the rows trained since, a recorded row keeps its checkpoints, and the same row and table with
# another path stops the driver. A row whose run finishes training while the driver runs is added
# once every recorded piece is done.
#
# Runs, in order (Step 3's, no targeted pilot): 1 rank-APGD-100 at 2/255 and 4/255 (headline);
# 3 targeted rank-PGD-20 at 4/255; 4 embedding-shift PGD-20 at 4/255; 2 rank-PGD-20 over the eps
# grid (main models, 4 datasets); T clean twin on the 4 datasets (eval.py has no clean-only mode:
# rank-PGD-20 at 1/255 on 1000 queries, for the clean recall on all queries that comes with it);
# 7 transfer from pretrained (MSLS; waits until every main row is recorded); 8 32/255 (MSLS);
# 9 and 9b APGD with 500 steps / 3 restarts (sp, MSLS); 6 L2 rank-PGD-20; A runs 1 and 2 for the
# ablations on MSLS and Nordland. Every piece writes to a fixed directory under FOLLOWUP_EVAL_ROOT
# and is marked DONE once eval.py exits 0: stop at any point and rerun to resume.
#
# Sessions: a real run needs FOLLOWUP_EVAL_STOP_AT, the time by which every process must be gone
# (the machine's clock). Same rules as STEP3_STOP_AT (scripts/lib/session.sh): Ctrl-C to every
# process 10 min before it, SIGKILL 5 min before. Before that, a piece starts only if its
# estimated time ends by FOLLOWUP_EVAL_STOP_AT - 15 min (FOLLOWUP_EVAL_ROOT/durations.csv); a
# shorter piece further down the list runs first when the next one does not fit.
#
# Usage:
#   FOLLOWUP_EVAL_STOP_AT="2026-10-20 08:30" scripts/mplc_v2_followup_eval.sh [--model KEY:TABLES:PATH ...]
#                                                     # run or resume the evaluation
#   scripts/mplc_v2_followup_eval.sh --dry-run [--model ...]   # list the pending pieces and commands
#   scripts/mplc_v2_followup_eval.sh --summary-only   # rebuild the tables, evaluate nothing
#   scripts/mplc_v2_followup_eval.sh -h               # this help
# e.g. --model sp_aw2:AB:logs/mplc_v2_followup/mplc_v2_supervlad_mplc_aw2_mixlinf_sp_fte0_s0/last_model.pth
#
# Environment overrides:
#   FOLLOWUP_EVAL_STOP_AT     end of this session, any `date -d` time (required for a real run)
#   FOLLOWUP_EVAL_ROOT        evaluation output directory   (default: output/mplc_v2_followup)
#   FOLLOWUP_ROOT             follow-up training runs       (default: logs/mplc_v2_followup)
#   FOLLOWUP_STEP2_ROOT       Step 2 runs                   (default: logs/mplc_v2_step2)
#   FOLLOWUP_EVAL_MODELS      rows to add, space-separated keys (default: all); recorded rows are
#                             always evaluated
#   FOLLOWUP_EVAL_TARGET_RANK targeted attack's target rank (default: 10)
#   FOLLOWUP_EVAL_PRETRAINED  pretrained SuperVLAD checkpoint (default: checkpoints/SuperVLAD_base.pth)
#   PYTHON                    python interpreter            (default: python)
#
# Exit codes: 0 done (every piece of the recorded rows, and every requested row is trained);
# 1 a piece failed (see its directory's info.log, rerun to resume) or another driver holds the
# lock; 2 bad option or FOLLOWUP_EVAL_STOP_AT, a malformed --model or missing checkpoint file, an
# unknown FOLLOWUP_EVAL_MODELS key or a bad FOLLOWUP_EVAL_TARGET_RANK; 3 stopped: a --model
# conflicts with a recorded one, a trained run's or a recorded checkpoint file is missing,
# FOLLOWUP_EVAL_TARGET_RANK differs from run 3's, or (at the end) every piece is done but some
# requested rows are not trained yet (rerun after scripts/mplc_v2_followup_train.sh finishes
# them); 5 session over (rerun in the next session); 124 or 137 killed by the session's timeout.
#
# Outputs in ${FOLLOWUP_EVAL_ROOT}/summary/, rebuilt after every piece: status.md, table_budget5 and
# table_best (.md, .csv; Step 3's columns plus APGD R@1 at 2/255), ablations_budget5 and
# ablations_best, clean_twin (clean R@1 / R@5 of the twin, pretrained and sp), curves_pgd.csv,
# curves_l2.csv, worst_case.csv, paired.csv (sp vs fare_eps2, plain_at_eps2 and sp_s1 on run 1's
# queries, exact McNemar test) and validity.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/third_party/SuperVLAD:${PYTHONPATH:-}"

# shellcheck source=scripts/lib/supervlad_common.sh
source "${SCRIPT_DIR}/lib/supervlad_common.sh"
# shellcheck source=scripts/lib/session.sh
source "${SCRIPT_DIR}/lib/session.sh"

PYTHON=${PYTHON:-python}
ROOT=${FOLLOWUP_EVAL_ROOT:-output/mplc_v2_followup}
TARGET_RANK=${FOLLOWUP_EVAL_TARGET_RANK:-10}
PRETRAINED=${FOLLOWUP_EVAL_PRETRAINED:-checkpoints/SuperVLAD_base.pth}
MODE=run
MODELS=()

ARGS=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) MODE=dry ;;
    --summary-only) MODE=summary ;;
    --model)
      if [ $# -lt 2 ]; then
        echo "--model needs KEY:TABLES:PATH" >&2
        exit 2
      fi
      MODELS+=(--model "$2")
      shift
      ;;
    --model=*) MODELS+=(--model "${1#--model=}") ;;
    -h | --help)
      sed -n '2,/^set -euo pipefail$/{/^set /d;s/^# \{0,1\}//;p}' "$0"
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

PLAN_FLAGS=(--root "${ROOT}" --target-rank "${TARGET_RANK}" --pretrained "${PRETRAINED}"
  --followup-root "${FOLLOWUP_ROOT:-logs/mplc_v2_followup}" --step2-root "${FOLLOWUP_STEP2_ROOT:-logs/mplc_v2_step2}"
  --keys "${FOLLOWUP_EVAL_MODELS:-}")
plan() {
  "${PYTHON}" -m src.followup_eval_plan "$@" "${PLAN_FLAGS[@]}"
}

# Flags every eval.py call shares; each piece adds its models, dataset, attack and outputs.
EVAL_FLAGS=(--eval_datasets_folder=datasets --dataset_split=test "${SUPERVLAD_EVAL_MODEL_FLAGS[@]}" --test_method=hard_resize)

case "${MODE}" in
  summary)
    if [ "${#MODELS[@]}" -gt 0 ]; then
      echo "--summary-only uses the recorded models; it takes no --model" >&2
      exit 2
    fi
    plan summarize
    exit 0
    ;;
  dry)
    echo "=== Every piece runs: ${PYTHON} eval.py ${EVAL_FLAGS[*]} <piece flags>"
    plan list "${MODELS[@]}"
    exit 0
    ;;
esac

session_begin FOLLOWUP_EVAL_STOP_AT "${SCRIPT_DIR}/mplc_v2_followup_eval.sh" "${ARGS[@]}"
session_lock "${ROOT}" "follow-up evaluation driver"
plan register "${MODELS[@]}"

previous=""
while true; do
  mapfile -t next < <(plan next --time-left "$((SESSION_STOP_BY - $(date +%s)))")
  case "${next[0]:-}" in
    DONE)
      plan summarize
      echo "=== Follow-up evaluation done. Tables: ${ROOT}/summary/"
      if [ -n "${next[1]:-}" ]; then
        echo "    ${next[1]}"
      fi
      exit 0
      ;;
    REGISTER*)
      if [ "${next[0]}" = "${previous}" ]; then
        echo "Recording ${next[0]#REGISTER } did not add them; see ${ROOT}/models.json." >&2
        exit 1
      fi
      previous="${next[0]}"
      echo "=== Trained since this session started: ${next[0]#REGISTER }"
      plan register
      continue
      ;;
    UNTRAINED*)
      plan summarize >/dev/null || true
      echo "=== Every piece of the recorded models is done, but these models are not trained yet: ${next[0]#UNTRAINED }" >&2
      printf '    %s\n' "${next[@]:1}" >&2
      echo "    Rerun after scripts/mplc_v2_followup_train.sh finishes them. Tables so far: ${ROOT}/summary/" >&2
      exit 3
      ;;
    STOP*)
      plan summarize >/dev/null || true
      echo "=== Follow-up evaluation stopped: ${next[0]#STOP }" >&2
      echo "    Status: ${ROOT}/summary/status.md" >&2
      exit 3
      ;;
    WAIT*)
      echo "=== Session over: ${next[0]#WAIT } before $(date -d "@${SESSION_STOP_BY}" '+%F %T'). Rerun in the next session." >&2
      exit 5
      ;;
    PIECE*) ;;
    *)
      echo "Unexpected planner output: ${next[*]}" >&2
      exit 1
      ;;
  esac

  piece="${next[0]#PIECE }"
  estimate="${next[1]}"
  piece_dir="${next[2]}"
  if [ "${piece}" = "${previous}" ]; then
    echo "Piece ${piece} ran but is not marked done; see ${piece_dir}." >&2
    exit 1
  fi
  previous="${piece}"
  mkdir -p "${piece_dir}"
  echo "=== Follow-up evaluation piece ${piece} (estimate $((estimate / 60)) min) at $(date '+%F %T')"
  start=$(date +%s)
  status=0
  "${PYTHON}" eval.py "${EVAL_FLAGS[@]}" "${next[@]:3}" || status=$?
  if [ "${status}" -ne 0 ]; then
    echo "=== Piece ${piece} failed with exit code ${status}; see ${piece_dir}/<timestamp>/info.log. Rerun to resume." >&2
    exit 1
  fi
  plan record "${piece}" --seconds "$(($(date +%s) - start))" >/dev/null
done
