#!/bin/bash
# bench-post.sh — POST a /completion or /v1/chat/completions payload to a
# llama-server, reading the JSON body FROM A FILE.
#
# Do not pass the payload as a curl argv string. A long prompt (tens of
# thousands of tokens, easily hundreds of KB once JSON-escaped) blew past
# ARG_MAX during this campaign and killed the arm with E2BIG -- silently
# enough that it looked like a server crash rather than a shell-level limit.
# `curl -d @file` reads the body from disk and has no such ceiling.
#
# Conventions used throughout this campaign's bench runs (documented here so
# results stay comparable across arms):
#   - warmup: run the payload once and DISCARD the result before the reps
#     that count. First-request latency includes cache/allocator warmup that
#     does not recur.
#   - reps: 3 non-warmup repetitions per (payload, arm) pair is the default;
#     report the median, not the mean (one slow rep from an unrelated host
#     hiccup should not move the number).
#   - seed: pin --seed / "seed" in the payload for determinism arms. Do NOT
#     reuse a prompt across reps with cache_prompt enabled (see below).
#   - cache_prompt: leave OFF (false) unless the arm is specifically testing
#     prefix-cache behavior. Repeating an IDENTICAL prompt with cache_prompt
#     on lets the server serve tokens straight from its own KV cache instead
#     of recomputing them -- this is "self-memorization" and inflated one
#     arm's measured throughput by roughly 2x before it was caught. If you
#     must test cache_prompt, use a fresh, never-before-seen prompt per rep.
#   - ngram-mod: turn OFF (unset --spec-ngram-mod-n-match / omit "ngram-mod"
#     from --spec-type) for any determinism/acceptance-rate arm. It is a
#     legitimate speed feature but it makes accepted-token counts
#     input-order-dependent, which breaks apples-to-apples comparison.
#   - restart between arms: recycle the server container between arms that
#     are supposed to be independent (different quant, different draft-n,
#     etc). A live server carries KV cache and RNG state across requests
#     that can quietly leak into the next arm's numbers.
#
# Usage:
#   bench-post.sh <url> <payload-file> [reps] [warmup(0|1)]
#
# Prints one JSON line per rep to stdout (the raw response body) and a
# one-line timing summary to stderr. Non-zero exit if any non-warmup rep
# fails.
set -u

URL="${1:?url required, e.g. http://127.0.0.1:5899/completion}"
PAYLOAD_FILE="${2:?payload file required (curl reads it with -d @file, never as an argv string)}"
REPS="${3:-3}"
WARMUP="${4:-1}"

[ -f "$PAYLOAD_FILE" ] || { echo "no such payload file: $PAYLOAD_FILE" >&2; exit 1; }

do_one() {
    local t0 t1 body rc
    t0=$(date +%s.%N)
    body=$(curl -sf -X POST "$URL" -H 'Content-Type: application/json' -d @"$PAYLOAD_FILE")
    rc=$?
    t1=$(date +%s.%N)
    if [ "$rc" -ne 0 ]; then
        echo "request failed (curl rc=$rc)" >&2
        return 1
    fi
    printf '%s\n' "$body"
    awk -v a="$t0" -v b="$t1" 'BEGIN{printf "elapsed_s=%.3f\n", b-a}' >&2
    return 0
}

if [ "$WARMUP" = "1" ]; then
    echo "warmup rep (discarded) ..." >&2
    do_one >/dev/null || { echo "warmup rep failed, aborting" >&2; exit 1; }
fi

fail=0
for i in $(seq 1 "$REPS"); do
    echo "rep $i/$REPS ..." >&2
    do_one || fail=1
done

exit "$fail"
