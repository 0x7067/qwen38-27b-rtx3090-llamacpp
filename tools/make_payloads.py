#!/usr/bin/env python3
"""Build the four bench payloads (short / mid1k / prose7k / agentic54k).

    QWEN_TOKENIZER_JSON=/path/to/tokenizer.json \
    CORPUS_DIR=/path/to/any/large/code+docs/tree \
    python3 make_payloads.py [outdir]

CORPUS_DIR can be any sizeable source tree (a llama.cpp checkout works well:
~50k+ tokens of .cpp/.md needed for the 54k payload). Avoid chat transcripts:
at temp 0 they can make the model emit EOS at position 0, which is why the
54k payload sets ignore_eos. Requires `pip install tokenizers`.
"""
import json, os, sys, glob
from tokenizers import Tokenizer

tok = Tokenizer.from_file(os.environ["QWEN_TOKENIZER_JSON"])
corpus_dir = os.environ["CORPUS_DIR"]
outdir = sys.argv[1] if len(sys.argv) > 1 else "."

def read_all(patterns, cap_files=None):
    files = []
    for pat in patterns:
        files += sorted(glob.glob(os.path.join(corpus_dir, "**", pat), recursive=True))
    if cap_files:
        files = files[:cap_files]
    return "\n\n".join(open(f, errors="ignore").read() for f in files)

def take_tokens(text, n):
    return tok.decode(tok.encode(text, add_special_tokens=False).ids[:n])

short = ("You are a senior engineer. Explain, step by step, how a single-node k3s cluster "
         "reconciles workloads from a git repository using Flux v2: the controllers involved, "
         "the reconciliation loop, how a Kustomization differs from a HelmRelease, what happens "
         "when a manifest is invalid, and how secrets encrypted with SOPS and age are decrypted "
         "at apply time. Then write a small example kustomization.yaml for a service folder and "
         "explain each field. Be precise and complete.")

code = read_all(["*.cpp", "*.cu", "*.py"])
docs = read_all(["*.md"])
if not code or not docs:
    sys.exit(f"CORPUS_DIR={corpus_dir} must contain .cpp/.cu/.py and .md files")

mid1k = take_tokens(code, 850) + "\n\nSummarize what this code does, then list its key functions and their roles. Answer:"
prose7k = take_tokens(docs * 8, 6800) + "\n\nWrite a detailed executive summary of the material above. Answer:"
agentic54k = take_tokens((code + docs) * 3, 53400) + \
    "\n\nBased on the code and documentation above, list the 10 most important technical decisions made and briefly justify each. Answer:"

for name, prompt, npred, extra in [
        ("short", short, 256, {}),
        ("mid1k", mid1k, 256, {}),
        ("prose7k", prose7k, 256, {}),
        ("agentic54k", agentic54k, 384, {"ignore_eos": True})]:
    n_tok = len(tok.encode(prompt, add_special_tokens=False).ids)
    payload = {"prompt": prompt, "n_predict": npred, "temperature": 0, "cache_prompt": False, **extra}
    with open(os.path.join(outdir, f"{name}.json"), "w") as f:
        json.dump(payload, f)
    print(name, "prompt_tokens=", n_tok)
