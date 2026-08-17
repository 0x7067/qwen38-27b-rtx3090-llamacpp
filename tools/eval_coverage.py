"""Coverage of the built draft-vocab sets on data that was never counted.

The holdout inside build_draft_vocab.py is 1/10 of the same sources. This is the
stricter check: repos and dump shards that the counting pass never opened, which
is what tells us whether the shortlist learned our workload or just llama.cpp's
and prettier's local identifiers.
"""
import argparse, collections, json, os, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from build_draft_vocab import CODE_EXT, DOC_EXT, SKIP_DIR, SKIP_FILE, PER_FILE_CAP, TOKENIZER


def read_tree(root, exts, cap):
    out, total = [], 0
    for dirpath, dirnames, filenames in os.walk(root):
        if SKIP_DIR.search("/" + os.path.relpath(dirpath, root).replace(os.sep, "/")):
            dirnames[:] = []
            continue
        for fn in sorted(filenames):
            if os.path.splitext(fn)[1] not in exts or SKIP_FILE.search(fn):
                continue
            if total >= cap:
                return out
            try:
                t = open(os.path.join(dirpath, fn), encoding="utf-8", errors="ignore").read(PER_FILE_CAP)
            except OSError:
                continue
            out.append(t)
            total += len(t)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["draft_vocab_32k.json", "draft_vocab_40k.json",
                                                  "draft_vocab_48k.json"])
    ap.add_argument("--eval-dir", default=os.path.join(ROOT, "ood"))
    ap.add_argument("--cap", type=int, default=8_000_000)
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER)

    groups = {}
    for name in sorted(os.listdir(args.eval_dir)):
        p = os.path.join(args.eval_dir, name)
        texts = read_tree(p, CODE_EXT | DOC_EXT, args.cap) if os.path.isdir(p) else \
            [open(p, encoding="utf-8", errors="ignore").read(args.cap)]
        c = collections.Counter()
        for enc in tok.encode_batch_fast(texts, add_special_tokens=False):
            c.update(enc.ids)
        groups[name] = c
        print(f"{name:28s} {sum(c.values())/1e6:6.2f} M tokens, {len(c)} distinct")

    print("\n" + " " * 28 + "".join(f"{os.path.basename(s).split('_')[-1][:-5]:>10s}" for s in args.sets))
    rows = {}
    for name, c in groups.items():
        line, tot = f"{name:28s}", sum(c.values())
        for s in args.sets:
            ids = set(json.load(open(os.path.join(ROOT, s)))["ids"])
            cov = sum(n for t, n in c.items() if t in ids) / tot
            rows.setdefault(os.path.basename(s), {})[name] = cov
            line += f"{cov*100:9.2f}%"
        print(line)
    json.dump(rows, open(os.path.join(ROOT, "ood_coverage.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
