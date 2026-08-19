# tools/bench

Small, self-contained helpers that encode lessons from a multi-agent
benchmark campaign on a single shared GPU. Each script is independently
useful; none of them assume the others are installed.

- `gpulock.sh` — acquire/release/status/steal for a single-GPU lock,
  heartbeat-refreshed, dies with its holder, never removes a lock it didn't
  create.
- `quiesce.sh` — "is the host quiet enough to trust a CPU-side number"
  gate, measured properly (not loadavg).
- `bench-post.sh` — POST a payload to `/completion` or `/v1/chat/completions`
  from a file, with the warmup/reps/seed conventions used throughout.
- `container-health.sh` — "started vs. genuinely exited" check for a bench
  container, with a startup registration grace period.

All pass `bash -n`.

## Methodology checklist

Lessons that cost real time to learn, kept here so the next sweep does not
re-learn them:

- **Background long GPU jobs, with a blocking watcher.** A foreground
  `timeout` on a long-running container kills the wrapping process but
  leaves the container (and its VRAM) alive, orphaned. Start it detached,
  then block on a watcher (`container-health.sh wait`, or a `docker wait`)
  that you control and can Ctrl-C without hurting the container.
- **Interleave and counterbalance arms.** Run A,B,A,B (or A,B,C,A,B,C for
  3+ arms), not A,A,A,B,B,B. Thermal drift, background daemons, and clock
  skew all correlate with wall-clock position, not with which arm is under
  test; interleaving is what separates "arm B is slower" from "the room got
  hotter."
- **Same-block controls.** Repeat the control arm within the same
  measurement block, not just once at the start. A control that drifts
  within its own block tells you the measurement itself is unstable before
  you trust any delta.
- **Report per-step-ms and acceptance-normalized numbers, not just raw
  tok/s.** Two arms with the same aggregate tok/s can have very different
  per-token cost and speculative-acceptance rate; the aggregate number alone
  hides which one actually improved.
- **Like-for-like prompt slices only.** Truncating a long payload to "the
  first N tokens" of a *different* prompt than the one another arm used is
  not the same measurement. A slice-dependent throughput difference produced
  a fake **-16% "thermal decay"** result in this campaign — it was the slice
  changing, not the card.
- **Determinism arms: ngram-mod OFF, restart between arms, never repeat a
  prompt with `cache_prompt` on.** `ngram-mod` makes accepted-token counts
  input-order-dependent. A live server carries KV-cache/RNG state across
  requests into the next arm. Repeating an identical prompt with
  `cache_prompt` enabled lets the server serve straight from its own cache —
  this "self-memorization" inflated one measured arm by roughly **2x**
  before it was caught. See `bench-post.sh` for the full convention list.
- **CUDA container output: parse from the first `[`.** The NVIDIA container
  license banner prints to stdout before your program's JSON does. Anything
  that parses container stdout as JSON must skip to the first `[` (or `{`),
  not assume line 1 is the payload.
- **`llama-perplexity` ignores `-t`.** It spawns `nproc-1` worker threads
  regardless of `-t`/`--threads`. If you need to bound its CPU usage,
  constrain the container/cgroup, not the flag.
- **KLD base-file size sanity check.** Expected base file size in bytes is
  approximately `n_chunk * (n_ctx/2 - 1) * 2 * (n_vocab + pad)` — use this to
  catch a truncated or wrong-corpus base file before spending GPU time on
  arms that compare against it.
- **Monitor comparisons must target the identity line, not the whole owner
  file.** A lock's heartbeat line changes every refresh interval; a monitor
  that diffs the entire owner file (rather than just the `owner=`/`pid=`
  identity fields) will see "change" every tick and never settle. Match the
  identity line specifically — see `gpulock.sh`'s `lock_owner()`.
- **`llama-swap`'s `/upstream/*` route is proxied, not authoritative for
  liveness.** It forwards to the upstream server and will report whatever
  that server says, including through a restart-in-progress. Use `/running`
  to ask llama-swap directly which model it currently has up, rather than
  inferring liveness from `/upstream/*` responses.
- **`pkill -f` self-matches.** A pattern-based `pkill -f <name>` matches
  against the full command line, which usually includes the invoking
  script's own argv (e.g. a wrapper shell that embeds the target name in its
  own invocation). This misfired twice in this campaign, killing the tooling
  instead of (or in addition to) the target. Use `pkill -x` (exact `comm`
  match) or a cgroup-scoped kill instead; reserve `-f` for cases where you
  have verified the pattern cannot match the caller.
