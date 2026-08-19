#!/bin/bash
# container-health.sh — "did this bench container actually die, or is it
# just not registered with docker yet" check.
#
# Right after `docker run -d`, a container can be legitimately absent from
# `docker inspect` output for a brief window before the daemon finishes
# registering it. Treating "not found yet" as "it died" produced a false
# failure early in this campaign. The fix is a grace period: absence is
# only a genuine problem if it persists past REGISTRATION_GRACE_SECONDS
# from when this check first started polling for that name.
#
# Usage:
#   container-health.sh wait <container-name> [grace-seconds] [poll-interval]
#     Blocks until the container is running, or reports a genuine exit
#     (captures docker logs) once past the grace period. Exit 0 if running,
#     1 if it exited (logs captured to stdout), 2 if never appeared within
#     grace and still absent.
set -u

GRACE_DEFAULT=30
POLL_DEFAULT=2

usage() { echo "usage: $0 wait <container-name> [grace-seconds] [poll-interval]" >&2; exit 2; }

cmd_wait() {
    local name="$1" grace="${2:-$GRACE_DEFAULT}" poll="${3:-$POLL_DEFAULT}"
    local elapsed=0 state exit_code

    while :; do
        state=$(docker inspect -f '{{.State.Status}}' "$name" 2>/dev/null)

        if [ -z "$state" ]; then
            # Not registered yet. Only a problem once we're past the grace
            # window -- absence-at-startup is normal, absence-forever is not.
            if [ "$elapsed" -ge "$grace" ]; then
                echo "NEVER REGISTERED: '$name' absent from docker inspect after ${grace}s grace" >&2
                return 2
            fi
            sleep "$poll"
            elapsed=$((elapsed + poll))
            continue
        fi

        case "$state" in
            running)
                echo "RUNNING: $name"
                return 0
                ;;
            exited|dead)
                exit_code=$(docker inspect -f '{{.State.ExitCode}}' "$name" 2>/dev/null)
                echo "EXITED: $name (exit_code=$exit_code) -- capturing logs" >&2
                echo "--- docker logs $name (last 100 lines) ---"
                docker logs "$name" 2>&1 | tail -100
                return 1
                ;;
            *)
                # created/restarting/paused etc -- keep polling regardless of
                # the grace window, these are legitimate transient states.
                sleep "$poll"
                elapsed=$((elapsed + poll))
                ;;
        esac
    done
}

main() {
    [ $# -ge 2 ] || usage
    local action="$1"; shift
    case "$action" in
        wait) cmd_wait "$@" ;;
        *) usage ;;
    esac
}

main "$@"
