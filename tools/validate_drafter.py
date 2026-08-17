#!/usr/bin/env python3
"""Validate a truncated drafter GGUF against its source.

    validate.py <src.gguf> <out.gguf> [keepset.json]

Checks structure, the llama.cpp offset invariant, metadata identity, byte
identity of untouched tensors, and row-content identity of sampled kept rows.
Exits non-zero if any check fails.
"""

import hashlib
import json
import os
import random
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggufio import GGUF, GGML_TYPE_I64, pad_to  # noqa: E402

HEAD = "output.weight"
D2T = "d2t"
SAMPLE_TENSORS = 5
SAMPLE_ROWS = 10
SEED = 20260817

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print("%-4s %-46s %s" % ("PASS" if ok else "FAIL", name, detail))
    return ok


def sha256_range(path, offset, length, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        remain = length
        while remain:
            buf = f.read(min(chunk, remain))
            if not buf:
                raise IOError("short read at %d" % offset)
            h.update(buf)
            remain -= len(buf)
    return h.hexdigest()


def read_range(path, offset, length):
    with open(path, "rb") as f:
        f.seek(offset)
        return f.read(length)


def main(argv):
    if len(argv) not in (3, 4):
        raise SystemExit(__doc__)
    src_path, out_path = argv[1], argv[2]
    keep_path = argv[3] if len(argv) == 4 else None

    src = GGUF(src_path)
    out = GGUF(out_path)
    print("source %s\noutput %s\n" % (src_path, out_path))

    # --- structure -----------------------------------------------------------
    check("header parses, GGUF v3", True,
          "arch=%s alignment=%d data_start=%d" % (out.kv.get("general.architecture"),
                                                  out.alignment, out.data_start))
    check("tensor count == 19", len(out.tensors) == 19, "got %d" % len(out.tensors))

    head_s = src.tensor(HEAD)
    head_o = out.tensor(HEAD)
    d2t = out.tensor(D2T)
    n = d2t.dims[0]

    check("%s dims == [5120, n]" % HEAD,
          head_o.dims == [5120, n] and head_o.type_id == head_s.type_id,
          "%s %s (source %s %s)" % (head_o.type_name, head_o.dims,
                                    head_s.type_name, head_s.dims))
    check("d2t dtype I64", d2t.type_id == GGML_TYPE_I64,
          "type=%s (%d)" % (d2t.type_name, d2t.type_id))
    check("d2t len == head rows", d2t.dims[0] == head_o.dims[1], "n=%d" % n)
    check("d2t is 1-D", len(d2t.dims) == 1, "dims=%s" % d2t.dims)
    check("d2t is the last tensor", out.tensors[-1].name == D2T,
          "last=%s" % out.tensors[-1].name)

    # --- offset invariant (ggml/src/gguf.cpp:779) ----------------------------
    errs, data_size = out.check_offsets()
    check("tensor offsets sequential (gguf.cpp:779)", not errs, str(errs[:3]) if errs else "ok")
    check("every offset aligned to %d" % out.alignment,
          all(t.offset % out.alignment == 0 for t in out.tensors))
    size = os.path.getsize(out_path)
    check("file size == data_start + data_size",
          size == out.data_start + data_size,
          "%d B (%.3f GB)" % (size, size / 1e9))

    # --- d2t contents --------------------------------------------------------
    raw = read_range(out_path, out.data_start + d2t.offset, d2t.nbytes)
    ids = list(struct.unpack("<%dq" % n, raw))
    n_tokens = len(out.kv["tokenizer.ggml.tokens"])
    check("d2t sorted ascending", ids == sorted(ids))
    check("d2t unique", len(set(ids)) == n)
    check("d2t in range [0, %d)" % n_tokens,
          ids[0] >= 0 and ids[-1] < n_tokens, "min=%d max=%d" % (ids[0], ids[-1]))
    if keep_path:
        kj = json.load(open(keep_path))
        want = kj["d2t"] if "d2t" in kj else kj["ids"]
        check("d2t == keep-set json", ids == list(want), "n=%d" % n)

    # --- metadata identity ---------------------------------------------------
    check("KV block byte-identical to source",
          out.kv_raw == src.kv_raw and out.n_kv == src.n_kv,
          "%d KVs, %d B" % (out.n_kv, len(out.kv_raw)))
    check("tokenizer.ggml.tokens count unchanged",
          len(out.kv["tokenizer.ggml.tokens"]) == len(src.kv["tokenizer.ggml.tokens"]) == 248320,
          "%d entries" % len(out.kv["tokenizer.ggml.tokens"]))
    check("tokenizer.ggml.token_type count unchanged",
          len(out.kv["tokenizer.ggml.token_type"]) == len(src.kv["tokenizer.ggml.token_type"]),
          "%d entries" % len(out.kv["tokenizer.ggml.token_type"]))

    # --- byte identity of untouched tensors ---------------------------------
    rng = random.Random(SEED)
    untouched = [t.name for t in out.tensors if t.name not in (HEAD, D2T)]
    sampled = rng.sample(untouched, SAMPLE_TENSORS)
    if "token_embd.weight" not in sampled:
        sampled[0] = "token_embd.weight"
    for name in sampled:
        s, o = src.tensor(name), out.tensor(name)
        same_shape = s.dims == o.dims and s.type_id == o.type_id
        hs = sha256_range(src_path, src.data_start + s.offset, s.nbytes)
        ho = sha256_range(out_path, out.data_start + o.offset, o.nbytes)
        check("byte-identical: %s" % name, same_shape and hs == ho,
              "%s %s %d B sha=%s" % (o.type_name, o.dims, o.nbytes, ho[:16]))

    # --- row-content identity of kept rows ----------------------------------
    row_bytes = head_s.nbytes // head_s.dims[1]
    check("Q4_0 row is %d B (160 blocks x 18 B)" % row_bytes, row_bytes == 2880)
    picks = sorted(rng.sample(range(n), SAMPLE_ROWS - 3)) + [0, n // 2, n - 1]
    ok_rows = 0
    for i in sorted(set(picks))[:SAMPLE_ROWS]:
        tgt = ids[i]
        a = read_range(src_path, src.data_start + head_s.offset + tgt * row_bytes, row_bytes)
        b = read_range(out_path, out.data_start + head_o.offset + i * row_bytes, row_bytes)
        if a == b:
            ok_rows += 1
        else:
            print("     row mismatch: draft %d -> target %d" % (i, tgt))
    check("row identity: out[i] == src[d2t[i]] for %d rows" % SAMPLE_ROWS,
          ok_rows == len(sorted(set(picks))[:SAMPLE_ROWS]),
          "%d/%d rows matched" % (ok_rows, len(sorted(set(picks))[:SAMPLE_ROWS])))

    # A truncated head must NOT match the source rows at the same index, or the
    # row selection silently did nothing.
    probe = n - 1
    a = read_range(src_path, src.data_start + head_s.offset + probe * row_bytes, row_bytes)
    b = read_range(out_path, out.data_start + head_o.offset + probe * row_bytes, row_bytes)
    check("head is genuinely remapped (not a prefix copy)",
          ids[probe] != probe and a != b,
          "draft row %d -> target %d" % (probe, ids[probe]))

    # --- source untouched ----------------------------------------------------
    check("source file size unchanged", os.path.getsize(src_path) == 1680271648,
          "%d B" % os.path.getsize(src_path))

    failed = [r for r in results if not r[1]]
    print("\n%d checks, %d passed, %d failed" % (len(results), len(results) - len(failed), len(failed)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
