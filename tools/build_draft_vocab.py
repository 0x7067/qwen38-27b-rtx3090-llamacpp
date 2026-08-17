"""Build a vocab-truncated draft head id set for Qwen3.8 MTP speculative decoding.

Ported from syv-ai/qwen38-27b-rtx3090's build_draft_vocab.py, retargeted from
their Danish/English/code mix to ours (English technical + agentic + source code
heavy, Brazilian Portuguese light) and from vLLM/W4A16 to llama.cpp/GGUF.

What this produces is only the id set and the draft-row -> target-id map. Slicing
the actual lm_head rows is checkpoint-format specific and lives downstream.

Differences from the upstream script, all deliberate:

  * Upstream's texts_from() yields one string per plain file, and the holdout
    split is `j % 10 == 0` over that per-file index, so every .txt/.py file lands
    in the holdout counter and contributes nothing to the counts. Our corpus is
    almost entirely plain files, so we split at unit granularity instead:
    hash(unit key) % holdout_mod. A unit is a whole file (source trees) or a
    whole document (the downloaded dumps) -- never a chunk, because chunks of one
    file share its identifiers and would inflate held-out coverage.
  * Per-slice counters, blended by weight at the end, so the corpus mix is a
    weighting decision rather than a sampling decision and the per-slice coverage
    table is free.
  * Force-include is wider: upstream forces only all_special_ids plus a named
    chat-token list. We add the whole 256-token byte-level alphabet (ids 0..255;
    this tokenizer has no <0xNN> byte-fallback class, the byte alphabet *is* the
    first 256 ids) and every pure-whitespace token, which carries code indentation
    runs. Digits need no special case: the pre-tokenizer isolates \\p{N}, so every
    digit is a single-char token already inside 0..255.

Usage:
  python build_draft_vocab.py --sizes 32768 40960 49152
  python build_draft_vocab.py --weights code=0.45 eng=0.25 agentic=0.20 ptbr=0.10
"""
import argparse, collections, hashlib, json, os, re, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
TOKENIZER = "/data/buttercup_6tb/k3s/vllm-trial/models/Qwen3.8-27B-W4A16-AutoRound/tokenizer.json"
TARGET_VOCAB = 248320  # text_config.vocab_size in the checkpoint; see notes below

CODE_EXT = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".cuh", ".cu", ".py", ".rs", ".go",
    ".ts", ".tsx", ".js", ".jsx", ".mjs", ".sh", ".bash", ".yaml", ".yml", ".toml",
    ".json", ".sql", ".proto", ".Dockerfile", ".metal", ".m", ".java", ".rb",
}
DOC_EXT = {".md", ".mdx", ".rst", ".txt"}

# Per source: slice, root, extensions, byte cap for the whole root, per-file cap.
# Caps keep any one tree (llama.cpp's 18 MB of C++, prettier's test fixtures) from
# flooding the shortlist with project-local identifiers that do not generalize.
SOURCES = [
    dict(slice="code", root="/data/buttercup_6tb/k3s/llama-models/upstream-pr/llama.cpp",
         exts=CODE_EXT, cap=24_000_000),
    dict(slice="code", root=os.path.join(ROOT, "repos"), exts=CODE_EXT, cap=40_000_000,
         per_root_dir_cap=6_000_000),
    dict(slice="code", root="/data/docker-services", exts=CODE_EXT, cap=4_000_000),
    dict(slice="code", root=os.path.join(ROOT, "corpus/diffs"), exts={".txt"}, cap=7_000_000,
         docs=True),
    dict(slice="eng", root="/data/buttercup_6tb/k3s/llama-models/upstream-pr/llama.cpp",
         exts=DOC_EXT, cap=2_000_000),
    dict(slice="eng", root="/data/docker-services", exts=DOC_EXT, cap=14_000_000),
    dict(slice="eng", root=os.path.join(ROOT, "corpus/eng"), exts={".txt"}, cap=24_000_000,
         docs=True),
    dict(slice="agentic", root=os.path.join(ROOT, "corpus/agentic"), exts={".txt"},
         cap=26_000_000, docs=True),
    dict(slice="ptbr", root=os.path.join(ROOT, "corpus/ptbr"), exts={".txt"}, cap=22_000_000,
         docs=True),
]

SKIP_DIR = re.compile(
    r"(^|/)(\.git|node_modules|vendor|third_party|dist|build|target|\.venv|venv|"
    r"__pycache__|\.cache|site-packages|testdata|fixtures|__fixtures__|i18n)(/|$)")
SKIP_FILE = re.compile(r"(\.min\.|\.map$|-lock\.json$|\.lock$|\.svg$|\.pb\.go$|_pb2\.py$)")
PER_FILE_CAP = 262_144
MEAN_LINE_MAX = 200  # catches base64 blobs, generated tables, minified leftovers


def looks_generated(text):
    lines = text.count("\n") + 1
    return len(text) / lines > MEAN_LINE_MAX


def iter_units(src):
    """Yield (unit_key, text). One unit per file, or per document for dumps."""
    root, exts, cap = src["root"], src["exts"], src["cap"]
    dir_cap = src.get("per_root_dir_cap")
    total, per_dir = 0, collections.Counter()
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = os.path.relpath(dirpath, root)
        if SKIP_DIR.search("/" + rel_dir.replace(os.sep, "/")):
            dirnames[:] = []
            continue
        dirnames.sort()
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1] not in exts or SKIP_FILE.search(fn):
                continue
            path = os.path.join(dirpath, fn)
            rel = os.path.relpath(path, root)
            top = rel.split(os.sep)[0]
            if dir_cap and per_dir[top] >= dir_cap:
                continue
            if total >= cap:
                return
            try:
                with open(path, encoding="utf-8", errors="ignore") as fh:
                    text = fh.read(None if src.get("docs") else PER_FILE_CAP)
            except OSError:
                continue
            if not text.strip():
                continue
            if src.get("docs"):
                # Dumps are one document per blank-line-separated record; documents
                # are independent, so a per-document holdout is safe here.
                for i, doc in enumerate(text.split("\n\n")):
                    if not doc.strip():
                        continue
                    if total >= cap:
                        return
                    total += len(doc)
                    per_dir[top] += len(doc)
                    yield f"{root}::{rel}::{i}", doc
            else:
                if looks_generated(text):
                    continue
                total += len(text)
                per_dir[top] += len(text)
                yield f"{root}::{rel}", text


def is_held(key, mod):
    return int(hashlib.blake2b(key.encode(), digest_size=8).hexdigest(), 16) % mod == 0


def count_corpus(tok, sources, holdout_mod, batch=512):
    train = collections.defaultdict(collections.Counter)
    held = collections.defaultdict(collections.Counter)
    bytes_seen = collections.Counter()
    for src in sources:
        sl = src["slice"]
        if not os.path.exists(src["root"]):
            print(f"  ! missing root {src['root']}, skipped", file=sys.stderr)
            continue
        buf_t, buf_h = [], []
        n0 = sum(train[sl].values()) + sum(held[sl].values())
        for key, text in iter_units(src):
            bytes_seen[sl] += len(text)
            (buf_h if is_held(key, holdout_mod) else buf_t).append(text)
            for buf, sink in ((buf_t, train), (buf_h, held)):
                if len(buf) >= batch:
                    for enc in tok.encode_batch_fast(buf, add_special_tokens=False):
                        sink[sl].update(enc.ids)
                    buf.clear()
        for buf, sink in ((buf_t, train), (buf_h, held)):
            if buf:
                for enc in tok.encode_batch_fast(buf, add_special_tokens=False):
                    sink[sl].update(enc.ids)
        n = sum(train[sl].values()) + sum(held[sl].values()) - n0
        print(f"  {sl:8s} {os.path.basename(src['root'].rstrip('/')):24s} {n/1e6:8.2f} M tokens")
    return train, held, bytes_seen


def forced_ids(tok, vocab_n, tokenizer_path):
    """Byte alphabet + all added/control tokens + every pure-whitespace token."""
    byte_alphabet = set(range(256))
    assert all(len(tok.id_to_token(i)) == 1 for i in range(256)), "ids 0..255 are not the byte alphabet"
    added = {a["id"] for a in json.load(open(tokenizer_path))["added_tokens"]}
    whitespace = set()
    for i in range(vocab_n):
        s = tok.decode([i])
        if s and not s.strip():
            whitespace.add(i)
    return byte_alphabet | added | whitespace, dict(
        byte_alphabet=len(byte_alphabet), control=len(added), whitespace=len(whitespace))


def blended_rank(train, weights):
    """Rank ids by summed per-slice frequency share, each scaled to its weight."""
    score = collections.Counter()
    for sl, c in train.items():
        tot = sum(c.values())
        if not tot or sl not in weights:
            continue
        w = weights[sl] / tot
        for t, n in c.items():
            score[t] += n * w
    return score


def coverage(counter, idset):
    tot = sum(counter.values())
    if not tot:
        return float("nan")
    return sum(n for t, n in counter.items() if t in idset) / tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default=TOKENIZER)
    ap.add_argument("--sizes", type=int, nargs="+", default=[32768, 40960, 49152])
    ap.add_argument("--weights", nargs="+",
                    default=["code=0.45", "eng=0.25", "agentic=0.20", "ptbr=0.10"])
    ap.add_argument("--holdout-mod", type=int, default=10)
    ap.add_argument("--out", default=ROOT)
    args = ap.parse_args()
    weights = {k: float(v) for k, v in (w.split("=") for w in args.weights)}

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(args.tokenizer)
    vocab_n = tok.get_vocab_size(with_added_tokens=True)
    print(f"tokenizer ids: {vocab_n} (checkpoint vocab_size {TARGET_VOCAB}, "
          f"{TARGET_VOCAB - vocab_n} pad rows the tokenizer can never emit)")

    print("counting:")
    train, held, bytes_seen = count_corpus(tok, SOURCES, args.holdout_mod)
    for sl in sorted(set(list(train) + list(held))):
        print(f"  {sl:8s} {bytes_seen[sl]/1e6:7.1f} MB  train {sum(train[sl].values())/1e6:7.2f} M  "
              f"held {sum(held[sl].values())/1e6:6.2f} M  distinct {len(train[sl]) :6d}")

    force, force_break = forced_ids(tok, vocab_n, args.tokenizer)
    print(f"forced: {len(force)} ids {force_break}")
    score = blended_rank(train, weights)
    ranked = [t for t, _ in score.most_common() if t not in force]

    raw_total = collections.Counter()
    for c in train.values():
        raw_total.update(c)
    tot_blend = sum(score.values())
    with open(os.path.join(args.out, "token_freq.tsv"), "w", encoding="utf-8") as fh:
        fh.write("rank\ttoken_id\tpiece\tblended_share\tcum_coverage\traw_count\tforced\n")
        cum = 0.0
        order = sorted(score.items(), key=lambda kv: -kv[1])
        for r, (t, s) in enumerate(order):
            cum += s / tot_blend
            piece = (tok.id_to_token(t) or "").replace("\t", "\\t").replace("\n", "\\n")
            fh.write(f"{r}\t{t}\t{piece}\t{s/tot_blend:.10f}\t{cum:.8f}\t{raw_total[t]}\t"
                     f"{int(t in force)}\n")
        # forced ids that never appeared in the corpus still belong in the head
        for t in sorted(force - set(score)):
            piece = (tok.id_to_token(t) or "").replace("\t", "\\t").replace("\n", "\\n")
            fh.write(f"-1\t{t}\t{piece}\t0\t{cum:.8f}\t0\t1\n")
    print(f"wrote token_freq.tsv ({len(score)} observed ids)")

    rows = []
    for n in sorted(args.sizes):
        ids = sorted(force | set(ranked[: n - len(force)]))
        assert len(ids) == n, (len(ids), n)
        assert max(ids) < TARGET_VOCAB
        idset = set(ids)
        row = dict(n=n, overall=sum(weights[s] * coverage(held[s], idset) for s in weights
                                    if sum(held[s].values())),
                   per_slice={s: coverage(held[s], idset) for s in sorted(held)},
                   train_overall=sum(weights[s] * coverage(train[s], idset) for s in weights
                                     if sum(train[s].values())),
                   train_per_slice={s: coverage(train[s], idset) for s in sorted(train)})
        rows.append(row)
        out = dict(
            n=n, target_vocab_size=TARGET_VOCAB, tokenizer_vocab_size=vocab_n,
            weights=weights, forced=force_break, forced_count=len(force),
            held_out_coverage=row["overall"], per_slice_held_out_coverage=row["per_slice"],
            ids=ids, d2t=ids)
        p = os.path.join(args.out, f"draft_vocab_{n//1024}k.json")
        json.dump(out, open(p, "w"))
        print(f"wrote {os.path.basename(p)}: {n} ids, held-out coverage {row['overall']*100:.2f}%")

    print("\nN        overall   " + "  ".join(f"{s:>8s}" for s in sorted(held)) + "     p^2     p^4")
    for r in rows:
        p = r["overall"]
        print(f"{r['n']:<8d} {p*100:6.2f}%  " +
              "  ".join(f"{r['per_slice'][s]*100:7.2f}%" for s in sorted(held)) +
              f"  {p**2*100:6.2f}%  {p**4*100:6.2f}%")
    json.dump(rows, open(os.path.join(args.out, "coverage.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
