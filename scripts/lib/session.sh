#!/usr/bin/env bash
# Session deadline and lock shared by the drivers that run after Step 2
# (scripts/mplc_v2_plain_at_full.sh, scripts/mplc_v2_step3.sh). Same rules as the Step 2
# driver (scripts/mplc_v2_step2.sh, "Sessions"), which keeps its own copy.
#
# A real run needs <PREFIX>_STOP_AT, the time by which every process must be gone (the
# server's clock). session_begin reruns the calling script under `timeout`, which sends
# Ctrl-C to it and every child 10 min before that time and SIGKILL 5 min before. That is the
# guarantee; before it, the driver itself starts no work that could end after
# SESSION_STOP_BY = <PREFIX>_STOP_AT - 15 min.

# session_begin VAR SCRIPT [ARGS...]
#   VAR: name of the deadline variable, e.g. STEP3_STOP_AT. SCRIPT and ARGS rerun the caller.
#   Exits 2 when VAR is unset, not a time, or less than 1 h away. Sets SESSION_STOP_AT,
#   SESSION_STOP_BY and SESSION_INTERRUPT_AT (Unix seconds).
session_begin() {
  local var=$1 script=$2 value now
  shift 2
  value=${!var:-}
  if [ -z "${value}" ]; then
    echo "Set ${var} to the time this session must end, e.g. ${var}=\"2026-10-01 07:30\"." >&2
    exit 2
  fi
  if ! SESSION_STOP_AT=$(date -d "${value}" +%s 2>/dev/null); then
    echo "${var} is not a time: ${value}" >&2
    exit 2
  fi
  SESSION_INTERRUPT_AT=$((SESSION_STOP_AT - 10 * 60))
  SESSION_STOP_BY=$((SESSION_STOP_AT - 15 * 60))
  if [ -n "${_SESSION_UNDER_TIMEOUT:-}" ]; then
    return 0
  fi
  now=$(date +%s)
  if [ $((SESSION_STOP_AT - now)) -lt 3600 ]; then
    echo "${var} $(date -d "@${SESSION_STOP_AT}" '+%F %T') is less than 1 h away: nothing would run." >&2
    exit 2
  fi
  echo "=== Session ends at $(date -d "@${SESSION_STOP_AT}" '+%F %T %Z') (${var}=\"${value}\")"
  echo "    no work ends after        $(date -d "@${SESSION_STOP_BY}" '+%F %T')"
  echo "    Ctrl-C to every process   $(date -d "@${SESSION_INTERRUPT_AT}" '+%F %T')"
  echo "    SIGKILL to every process  $(date -d "@$((SESSION_STOP_AT - 5 * 60))" '+%F %T')"
  # timeout signals its whole process group: the driver and everything it started. The
  # resolved time is passed on, so a relative value ("+11 hours") is not re-read.
  exec env "${var}=@${SESSION_STOP_AT}" _SESSION_UNDER_TIMEOUT=1 \
    timeout -s INT -k 5m "$((SESSION_INTERRUPT_AT - now))s" "${script}" "$@"
}

# session_lock DIR WHAT
#   One driver per directory: exits 1 if another process holds DIR/.lock.
session_lock() {
  mkdir -p "$1"
  exec 9>"$1/.lock"
  if ! flock -n 9; then
    echo "Another $2 is running on $1." >&2
    exit 1
  fi
}
