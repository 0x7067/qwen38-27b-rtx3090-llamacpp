#!/bin/bash
# gpulock.sh — a directory-free file lock for a single shared GPU, shared by
# every bench agent that touches it. Not a generic lockfile: it is built
# around three failures we actually hit during a multi-agent benchmark
# campaign on a single-GPU host.
#
#   1. A background heartbeat refresher that outlived its holder (the bench
#      process died, the refresher kept ticking, the lock looked "live"
#      forever). Fix: the refresher is a child process started with `&`,
#      and a trap on the PARENT kills it on any exit path. It cannot outlive
#      the shell that acquired the lock.
#   2. "Steal" logic that used the owner's PID to judge staleness. A holder
#      can legitimately go quiet for minutes (model load, ppl corpus churn)
#      while very much alive. Staleness here is judged ONLY by the GPU
#      itself being idle (nvidia-smi --query-compute-apps) for N minutes,
#      never by process liveness or heartbeat age alone.
#   3. `release` removing a lock it did not create, because a second agent
#      raced into an empty lock dir. Fix: release refuses unless the owner
#      field in the lock file matches the caller's declared name.
#
# Usage:
#   gpulock.sh acquire <lock-dir> <owner-name> <task-description>
#   gpulock.sh release <lock-dir> <owner-name>
#   gpulock.sh status  <lock-dir>
#   gpulock.sh steal    <lock-dir> <owner-name> <task-description> [idle-minutes]
#
# <lock-dir> is a plain directory path, e.g. /tmp/gpu.lock -- NOT the
# production /tmp/gpu.lock unless you mean it. Pass a throwaway path when
# testing this script.
#
# Lock directory layout:
#   <lock-dir>/owner        # "owner=<name>\npid=<shell-pid>\nstarted=<iso8601>\ntask=<text>"
#   <lock-dir>/heartbeat    # updated every HEARTBEAT_INTERVAL seconds by the child refresher
#   <lock-dir>/refresher.pid
set -u

HEARTBEAT_INTERVAL="${GPULOCK_HEARTBEAT_INTERVAL:-30}"
IDLE_STALE_MINUTES_DEFAULT=10

usage() { echo "usage: $0 {acquire|release|status|steal} <lock-dir> [owner] [task] [idle-minutes]" >&2; exit 2; }

now_iso() { date -u +%Y-%m-%dT%H:%M:%S%z; }

# Is the GPU genuinely idle (no compute-apps rows) right now? Sampling once
# is a snapshot, not a verdict -- callers that need staleness must poll this
# across the requested window themselves (see gpu_idle_for_minutes).
gpu_has_compute_apps() {
    local rows
    rows=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || true)
    [ "${rows:-0}" -gt 0 ]
}

# True only if the GPU shows zero compute-apps for the WHOLE requested
# window, sampled once a minute. This is deliberately slow and deliberately
# never looks at the lock's owner PID -- a stale lock is defined by GPU
# idleness, never by whether some unrelated PID still exists.
gpu_idle_for_minutes() {
    local minutes="$1" i
    for i in $(seq 1 "$minutes"); do
        if gpu_has_compute_apps; then
            return 1
        fi
        [ "$i" -lt "$minutes" ] && sleep 60
    done
    return 0
}

start_heartbeat_child() {
    local lock_dir="$1"
    (
        # This subshell IS the heartbeat. It has no reason to live past its
        # parent, so trap every exit path and stop touching the file the
        # instant the parent is gone. Do not let this become a daemon.
        trap 'exit 0' TERM INT
        while :; do
            date -u +%Y-%m-%dT%H:%M:%S%z > "$lock_dir/heartbeat" 2>/dev/null
            sleep "$HEARTBEAT_INTERVAL" &
            wait $!
        done
    ) &
    echo $! > "$lock_dir/refresher.pid"
    disown $! 2>/dev/null || true
}

stop_heartbeat_child() {
    local lock_dir="$1"
    if [ -f "$lock_dir/refresher.pid" ]; then
        local pid
        pid=$(cat "$lock_dir/refresher.pid" 2>/dev/null)
        if [ -n "${pid:-}" ]; then
            kill -TERM "$pid" 2>/dev/null
            sleep 0.2
            kill -9 "$pid" 2>/dev/null
        fi
        rm -f "$lock_dir/refresher.pid"
    fi
}

lock_owner() {
    local lock_dir="$1"
    [ -f "$lock_dir/owner" ] || return 1
    grep '^owner=' "$lock_dir/owner" | head -1 | cut -d= -f2-
}

cmd_acquire() {
    local lock_dir="$1" owner="$2" task="${3:-}"
    if ! mkdir "$lock_dir" 2>/dev/null; then
        echo "LOCKED: $(cmd_status "$lock_dir")" >&2
        return 1
    fi
    {
        echo "owner=$owner"
        echo "pid=$$"
        echo "started=$(now_iso)"
        echo "task=$task"
    } > "$lock_dir/owner"
    start_heartbeat_child "$lock_dir"
    # DIE-WITH-HOLDER: if this acquiring shell exits for any reason, the
    # heartbeat child must not keep ticking and making a dead lock look
    # alive. This trap is the actual enforcement point of that guarantee.
    trap 'stop_heartbeat_child "'"$lock_dir"'"' EXIT
    echo "acquired $lock_dir as $owner"
}

cmd_release() {
    local lock_dir="$1" owner="$2"
    local current
    current=$(lock_owner "$lock_dir")
    if [ -z "${current:-}" ]; then
        echo "no lock at $lock_dir" >&2
        return 1
    fi
    if [ "$current" != "$owner" ]; then
        echo "REFUSED: lock is held by '$current', not '$owner'. Never remove a lock you didn't create." >&2
        return 1
    fi
    stop_heartbeat_child "$lock_dir"
    rm -rf "$lock_dir"
    echo "released $lock_dir (was $owner)"
}

cmd_status() {
    local lock_dir="$1"
    if [ ! -d "$lock_dir" ]; then
        echo "FREE"
        return 0
    fi
    if [ ! -f "$lock_dir/owner" ]; then
        echo "HELD by <unknown: owner file missing>"
        return 0
    fi
    local owner heartbeat
    owner=$(lock_owner "$lock_dir")
    heartbeat=$(cat "$lock_dir/heartbeat" 2>/dev/null || echo "never")
    echo "HELD by $owner (heartbeat=$heartbeat)"
}

# Forcibly take over a lock -- only after GPU idleness is proven for the
# whole window, never on PID or heartbeat-age grounds alone.
cmd_steal() {
    local lock_dir="$1" owner="$2" task="${3:-}" idle_minutes="${4:-$IDLE_STALE_MINUTES_DEFAULT}"
    if [ ! -d "$lock_dir" ]; then
        cmd_acquire "$lock_dir" "$owner" "$task"
        return $?
    fi
    echo "checking GPU idle for ${idle_minutes}m before stealing $lock_dir ..." >&2
    if ! gpu_idle_for_minutes "$idle_minutes"; then
        echo "REFUSED: GPU shows compute-apps activity during the idle window -- lock is real, not stale." >&2
        return 1
    fi
    local prior
    prior=$(lock_owner "$lock_dir")
    stop_heartbeat_child "$lock_dir"
    rm -rf "$lock_dir"
    mkdir "$lock_dir"
    {
        echo "owner=$owner"
        echo "pid=$$"
        echo "started=$(now_iso)"
        echo "task=$task"
        echo "stole_from=$prior"
    } > "$lock_dir/owner"
    start_heartbeat_child "$lock_dir"
    trap 'stop_heartbeat_child "'"$lock_dir"'"' EXIT
    echo "stole $lock_dir from '$prior' as '$owner' after ${idle_minutes}m proven GPU idle"
}

main() {
    [ $# -ge 2 ] || usage
    local action="$1" lock_dir="$2"
    shift 2
    case "$action" in
        acquire) cmd_acquire "$lock_dir" "${1:?owner name required}" "${2:-}" ;;
        release) cmd_release "$lock_dir" "${1:?owner name required}" ;;
        status)  cmd_status  "$lock_dir" ;;
        steal)   cmd_steal   "$lock_dir" "${1:?owner name required}" "${2:-}" "${3:-}" ;;
        *) usage ;;
    esac
}

main "$@"
