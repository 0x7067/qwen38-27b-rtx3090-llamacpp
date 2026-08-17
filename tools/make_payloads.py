import json, os, glob

tok = __import__("tokenizers").Tokenizer.from_file("/data/buttercup_6tb/k3s/vllm-trial/models/Qwen3.8-27B-W4A16-AutoRound/tokenizer.json")

def take_tokens(text, n):
    ids = tok.encode(text, add_special_tokens=False).ids[:n]
    return tok.decode(ids)

short = ("You are a senior engineer. Explain, step by step, how a single-node k3s cluster "
         "reconciles workloads from a git repository using Flux v2: the controllers involved, "
         "the reconciliation loop, how a Kustomization differs from a HelmRelease, what happens "
         "when a manifest is invalid, and how secrets encrypted with SOPS and age are decrypted "
         "at apply time. Then write a small example kustomization.yaml for a service folder and "
         "explain each field. Be precise and complete.")

# mid1k: code+prose mix ~1000 tokens
src = open("/data/buttercup_6tb/k3s/llama-models/upstream-pr/llama.cpp/src/models/eagle3.cpp").read()
mid1k = take_tokens(src, 850) + "\n\nSummarize what this file does, then list its key functions and their roles. Answer:"

# prose7k: markdown docs
docs = ""
for f in ["/data/docker-services/k8s/MIGRATION_LOG.md", "/data/docker-services/k8s/workloads/apps/llama/PERFORMANCE.md", "/data/docker-services/CLAUDE.md"]:
    docs += open(f).read() + "\n\n"
prose7k = take_tokens(docs*4, 6800) + "\n\nWrite a detailed executive summary of the infrastructure described above. Answer:"

# agentic54k: transcript + code, ~54k tokens
big = ""
tfiles = glob.glob("/data/docker-services/2026-*.txt")
for f in tfiles: big += open(f, errors="ignore").read() + "\n\n"
for f in sorted(glob.glob("/data/buttercup_6tb/k3s/llama-models/upstream-pr/llama.cpp/src/*.cpp"))[:8]:
    big += open(f, errors="ignore").read() + "\n\n"
big += docs
agentic54k = take_tokens(big*3, 53500) + "\n\nBased on everything above, list the 10 most important technical decisions made and briefly justify each. Answer:"

for name, prompt, npred in [("short", short, 256), ("mid1k", mid1k, 256), ("prose7k", prose7k, 256), ("agentic54k", agentic54k, 384)]:
    n_tok = len(tok.encode(prompt, add_special_tokens=False).ids)
    with open(f"bench/{name}.json", "w") as f:
        json.dump({"prompt": prompt, "n_predict": npred, "temperature": 0, "cache_prompt": False}, f)
    print(name, "prompt_tokens=", n_tok)
