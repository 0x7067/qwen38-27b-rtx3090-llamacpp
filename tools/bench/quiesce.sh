#!/bin/bash
# quiesce.sh — "is the host quiet enough to trust a CPU-side benchmark
# number" gate.
#
# Do NOT use loadavg for this. loadavg counts uninterruptible sleep
# (iowait) the same as runnable CPU work, so a host doing heavy disk I/O
# (a corpus regen, a docker image pull) reads as "busy" even though the CPU
# itself is free -- and conversely a host pegged on CPU right after an
# iowait spike can look artificially calm for the first sample. We measure:
#
#   1. Non-idle, non-iowait CPU% (`us+sy+ni+irq+softirq+steal` from
#      /proc/stat), thresholded at <= 25%.
#   2. Runnable task count (procs_running from /proc/stat, NOT loadavg),
#      thresholded at <= 3.
#
# Both are sampled across the WHOLE measurement window (default 3s
# interval, N samples) rather than once, because a single instantaneous
# read can catch a scheduler blip in either direction.
#
# Usage:
#   quiesce.sh check [samples] [interval-seconds]
#     Exits 0 if quiet, 1 if not. Prints load_mean/load_max for both metrics.
#   quiesce.sh mark-contaminated <results-file> <row-id> <reason>
#     Appends a "contaminated" annotation to a results file. This NEVER
#     deletes or skips the row and never aborts a sweep -- a contaminated
#     regression must be annotated and re-measured, not silently dropped,
#     because dropping is how a real regression gets averaged away.
set -u

CPU_PCT_MAX="${QUIESCE_CPU_PCT_MAX:-25}"
RUNNABLE_MAX="${QUIESCE_RUNNABLE_MAX:-3}"

# Non-idle, non-iowait CPU% since last read of /proc/stat's aggregate "cpu " line.
# Returns "busy_pct runnable" as two space-separated integers.
sample_once() {
    local line1 line2 runnable
    line1=$(grep '^cpu ' /proc/stat)
    runnable=$(awk '{print $4}' /proc/stat 2>/dev/null) # placeholder, replaced below
    sleep 1
    line2=$(grep '^cpu ' /proc/stat)
    runnable=$(grep '^procs_running' /proc/stat | awk '{print $2}')

    awk -v l1="$line1" -v l2="$line2" -v run="$runnable" '
        function fields(s, arr,    n) { n = split(s, arr, " "); return n }
        BEGIN {
            n1 = fields(l1, a); n2 = fields(l2, b)
            # a[1]=user a[2]=nice a[3]=system a[4]=idle a[5]=iowait a[6]=irq a[7]=softirq a[8]=steal
            idle1 = a[4] + a[5]; idle2 = b[4] + b[5]
            total1 = 0; total2 = 0
            for (i = 1; i <= 8; i++) { total1 += a[i]; total2 += b[i] }
            dtotal = total2 - total1
            didle  = idle2 - idle1
            busy_pct = (dtotal > 0) ? (100.0 * (dtotal - didle) / dtotal) : 0
            printf "%.1f %s\n", busy_pct, run
        }'
}

cmd_check() {
    local samples="${1:-5}" interval="${2:-3}"
    local i busy_sum=0 busy_max=0 run_sum=0 run_max=0 busy run
    for i in $(seq 1 "$samples"); do
        read -r busy run < <(sample_once)
        busy_sum=$(awk -v a="$busy_sum" -v b="$busy" 'BEGIN{print a+b}')
        run_sum=$((run_sum + run))
        awk -v a="$busy" -v b="$busy_max" 'BEGIN{exit !(a>b)}' && busy_max="$busy"
        [ "$run" -gt "$run_max" ] && run_max="$run"
        [ "$i" -lt "$samples" ] && sleep "$interval"
    done
    local busy_mean run_mean
    busy_mean=$(awk -v s="$busy_sum" -v n="$samples" 'BEGIN{printf "%.1f", s/n}')
    run_mean=$(awk -v s="$run_sum" -v n="$samples" 'BEGIN{printf "%.1f", s/n}')

    echo "cpu_busy_pct_mean=$busy_mean cpu_busy_pct_max=$busy_max load_mean=$run_mean load_max=$run_max"

    local quiet=0
    awk -v m="$busy_max" -v t="$CPU_PCT_MAX" 'BEGIN{exit !(m<=t)}' || quiet=1
    [ "$run_max" -le "$RUNNABLE_MAX" ] || quiet=1
    if [ "$quiet" -eq 0 ]; then
        echo "QUIET"
        return 0
    else
        echo "NOT QUIET (thresholds: cpu<=${CPU_PCT_MAX}% runnable<=${RUNNABLE_MAX})"
        return 1
    fi
}

# Append a contamination annotation. Format is one line per call, append-only,
# so a re-measurement later just adds another line rather than editing history.
cmd_mark_contaminated() {
    local results_file="$1" row_id="$2" reason="$3"
    : "${results_file:?results file required}"
    printf 'CONTAMINATED row=%s reason="%s" ts=%s\n' \
        "$row_id" "$reason" "$(date -u +%Y-%m-%dT%H:%M:%S%z)" >> "$results_file"
    echo "annotated (not dropped): $row_id -- re-measure this row, do not exclude it"
}

usage() { echo "usage: $0 {check [samples] [interval]|mark-contaminated <file> <row-id> <reason>}" >&2; exit 2; }

main() {
    [ $# -ge 1 ] || usage
    local action="$1"; shift
    case "$action" in
        check)             cmd_check "$@" ;;
        mark-contaminated) cmd_mark_contaminated "$@" ;;
        *) usage ;;
    esac
}

main "$@"
