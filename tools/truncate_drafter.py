#!/usr/bin/env python3
"""Truncate the draft-MTP drafter's LM head to a reduced vocabulary.

    truncate_drafter.py <src.gguf> <keepset.json> <out.gguf>

Rewrites `output.weight` to hold only the rows named by the keep set, in keep-set
order, and appends a `d2t` I64 tensor mapping draft row i -> absolute target
token id. Every other tensor and the entire metadata block copy verbatim;
`token_embd.weight` in particular stays at full vocab, because the MTP input
embedding gathers by real target token id (src/models/qwen35.cpp:521-524).

Q4_0 rows copy losslessly: 5120 elements = 160 blocks x 18 B = 2880 B exactly,
so no requantization is involved.

Validation is deliberately loud and non-optional: an out-of-range or duplicate
d2t entry is an out-of-bounds write in ggml_set_rows at inference time.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ggufio import GGUF, GGML_TYPE_I64, Tensor, header_bytes, layout, pad_to  # noqa: E402

import struct

HEAD = "output.weight"
D2T = "d2t"
COPY_CHUNK = 8 << 20

# GGUF token types (llama_token_attr / LLAMA_TOKEN_TYPE_*)
TT_NORMAL, TT_UNKNOWN, TT_CONTROL, TT_USER_DEFINED, TT_UNUSED, TT_BYTE = 1, 2, 3, 4, 5, 6


def gpt2_byte_alphabet():
    """The 256 single-character pieces a gpt2-BPE vocab uses for raw bytes."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(0xA1, 0xAC + 1)) +
          list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for c in cs]


def load_keepset(path):
    obj = json.load(open(path))
    if isinstance(obj, list):
        return list(obj), {}
    for key in ("d2t", "ids"):
        if key in obj:
            return list(obj[key]), obj
    raise SystemExit("%s: no 'd2t' or 'ids' array found" % path)


def fail(msg):
    raise SystemExit("FATAL: " + msg)


def validate_keepset(ids, src):
    """Fail loudly on anything that would corrupt inference or waste the wave."""
    tokens = src.kv["tokenizer.ggml.tokens"]
    types = src.kv["tokenizer.ggml.token_type"]
    n_tokens = len(tokens)
    head = src.tensor(HEAD)
    n_rows = head.dims[1]

    if len(ids) != len(set(ids)):
        fail("keep set contains duplicate ids")
    if ids != sorted(ids):
        fail("keep set is not sorted ascending")
    if ids[0] < 0 or ids[-1] >= n_tokens:
        fail("keep set id out of range [0, %d): min=%d max=%d" % (n_tokens, ids[0], ids[-1]))
    if ids[-1] >= n_rows:
        fail("keep set id %d exceeds %s rows (%d)" % (ids[-1], HEAD, n_rows))

    keep = set(ids)
    report = {"n": len(ids), "n_tokens": n_tokens}

    # Every token the tokenizer marks special must survive: if the drafter cannot
    # propose the end-of-turn token, every turn ends in a guaranteed rejection.
    special = [i for i, t in enumerate(types)
               if t in (TT_UNKNOWN, TT_CONTROL, TT_USER_DEFINED, TT_BYTE)]
    missing_special = [i for i in special if i not in keep]
    report["special_total"] = len(special)
    report["special_missing"] = [(i, tokens[i]) for i in missing_special]

    # Named single-token ids from the tokenizer metadata.
    named = {}
    for kv_key, label in (("tokenizer.ggml.bos_token_id", "BOS"),
                          ("tokenizer.ggml.eos_token_id", "EOS"),
                          ("tokenizer.ggml.padding_token_id", "PAD")):
        if kv_key in src.kv:
            tid = src.kv[kv_key]
            named[label] = (tid, tokens[tid], tid in keep)
    report["named"] = named

    # Byte-fallback coverage: the 256 gpt2 byte-alphabet pieces.
    piece_to_id = {}
    for i, t in enumerate(tokens):
        piece_to_id.setdefault(t, i)
    missing_bytes = []
    for ch in gpt2_byte_alphabet():
        tid = piece_to_id.get(ch)
        if tid is None or tid not in keep:
            missing_bytes.append((ch, tid))
    report["byte_missing"] = missing_bytes

    problems = []
    if missing_special:
        problems.append("%d special/control tokens missing" % len(missing_special))
    if missing_bytes:
        problems.append("%d byte-alphabet tokens missing" % len(missing_bytes))
    for label, (tid, piece, present) in named.items():
        if not present:
            problems.append("%s token %d (%r) missing" % (label, tid, piece))
    report["problems"] = problems
    return report


def main(argv):
    if len(argv) != 4:
        raise SystemExit(__doc__)
    src_path, keep_path, out_path = argv[1], argv[2], argv[3]
    if os.path.abspath(src_path) == os.path.abspath(out_path):
        fail("refusing to write over the source model")

    src = GGUF(src_path)
    print("source: %s" % src_path)
    print("  arch=%s tensors=%d kv=%d alignment=%d data_start=%d"
          % (src.kv.get("general.architecture"), len(src.tensors), src.n_kv,
             src.alignment, src.data_start))

    errs, total = src.check_offsets()
    if errs:
        fail("source offsets are not sequential: %s" % errs[:3])

    head = src.tensor(HEAD)
    n_embd, n_rows = head.dims
    row_bytes = head.nbytes // n_rows
    print("  %s: %s %dx%d, %d B/row, %d B total"
          % (HEAD, head.type_name, n_embd, n_rows, row_bytes, head.nbytes))

    ids, meta = load_keepset(keep_path)
    print("keep set: %s -> n=%d" % (keep_path, len(ids)))

    rep = validate_keepset(ids, src)
    for label, (tid, piece, present) in sorted(rep["named"].items()):
        print("  %-4s id=%-7d %-16r present=%s" % (label, tid, piece, present))
    print("  special/control/byte-typed tokens: %d total, %d missing"
          % (rep["special_total"], len(rep["special_missing"])))
    if rep["special_missing"]:
        kept_types = {}
        for i, piece in rep["special_missing"][:5]:
            kept_types[i] = piece
        print("  first missing special: %s" % kept_types)
    print("  gpt2 byte-alphabet: %d/256 present" % (256 - len(rep["byte_missing"])))
    if rep["problems"]:
        fail("keep set is missing required tokens: " + "; ".join(rep["problems"]))
    print("  keep set OK: sorted, unique, in [0, %d)" % rep["n_tokens"])

    n = len(ids)

    # Build the new tensor table. d2t goes LAST in the info list, because
    # gguf.cpp:779 requires data offsets to match the running sum in info order.
    out_tensors = []
    for t in src.tensors:
        dims = [n_embd, n] if t.name == HEAD else list(t.dims)
        out_tensors.append(Tensor(t.name, dims, t.type_id, 0))
    out_tensors.append(Tensor(D2T, [n], GGML_TYPE_I64, 0))

    data_size = layout(out_tensors, src.alignment)
    hdr = header_bytes(len(out_tensors), src.kv_raw, src.n_kv, out_tensors)
    data_start = pad_to(len(hdr), src.alignment)
    expect_size = data_start + data_size
    print("output: %s" % out_path)
    print("  tensors=%d header=%d B data_start=%d expected_size=%d B (%.3f GB)"
          % (len(out_tensors), len(hdr), data_start, expect_size, expect_size / 1e9))

    tmp_path = out_path + ".partial"
    src_head_abs = src.data_start + head.offset

    with open(src_path, "rb") as fi, open(tmp_path, "wb") as fo:
        fo.write(hdr)
        fo.write(b"\x00" * (data_start - len(hdr)))

        for t in out_tensors:
            if fo.tell() != data_start + t.offset:
                fail("writer desync at %s: at %d, expected %d"
                     % (t.name, fo.tell(), data_start + t.offset))

            if t.name == D2T:
                fo.write(struct.pack("<%dq" % n, *ids))
            elif t.name == HEAD:
                written = 0
                # Coalesce consecutive ids into single reads; the low-id region
                # of a frequency-ranked keep set is largely contiguous.
                i = 0
                while i < n:
                    j = i
                    while j + 1 < n and ids[j + 1] == ids[j] + 1:
                        j += 1
                    fi.seek(src_head_abs + ids[i] * row_bytes)
                    buf = fi.read((j - i + 1) * row_bytes)
                    if len(buf) != (j - i + 1) * row_bytes:
                        fail("short read on %s rows %d..%d" % (HEAD, ids[i], ids[j]))
                    fo.write(buf)
                    written += len(buf)
                    i = j + 1
                if written != n * row_bytes:
                    fail("wrote %d B of head, expected %d B" % (written, n * row_bytes))
            else:
                s = src.tensor(t.name)
                fi.seek(src.data_start + s.offset)
                remain = s.nbytes
                while remain:
                    buf = fi.read(min(COPY_CHUNK, remain))
                    if not buf:
                        fail("short read on %s" % t.name)
                    fo.write(buf)
                    remain -= len(buf)

            padding = pad_to(t.nbytes, src.alignment) - t.nbytes
            if padding:
                fo.write(b"\x00" * padding)

        actual = fo.tell()

    if actual != expect_size:
        os.unlink(tmp_path)
        fail("wrote %d B, expected %d B" % (actual, expect_size))

    os.replace(tmp_path, out_path)
    print("  wrote %d B, matches expectation" % actual)
    print("  head %d -> %d rows (%.1f MB -> %.1f MB), d2t %d B"
          % (n_rows, n, head.nbytes / 1e6, n * row_bytes / 1e6, n * 8))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
