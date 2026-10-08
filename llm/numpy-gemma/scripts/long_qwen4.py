#!/usr/bin/env python3
"""A long-context test of Qwen3.8-Flash-Next: a prompt of about 128K tokens
of the source of this project, then questions with a long answer.

The prompt holds the files of the project (np_gemma/*.py, the C and CUDA
sources, the scripts, the plans), up to --tokens tokens, then the questions.
The answer streams to the terminal. At the end the script gives the time of
the prompt and the rate of the decode (and writes them with the answer to
--out).

    python scripts/long_qwen4.py                       # 128K on the GPU, up to 4096 new tokens
    python scripts/long_qwen4.py --tokens 32768 --gen 1024
    python scripts/long_qwen4.py --backend cpu --tokens 16384
    python scripts/long_qwen4.py --question "What does csrc/moe.c do?" --gen 512
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

MODEL = ROOT + "/models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf"
FILES = ["np_gemma/qwen4.py", "np_gemma/qwen4_gpu.py", "np_gemma/qwen_gpu.py", "np_gemma/program.py",
         "np_gemma/gguf.py", "np_gemma/st_qwen4.py", "np_gemma/csrc/kquants.c",
         "np_gemma/csrc/moe.c", "np_gemma/csrc/qsa.c", "np_gemma/csrc/hyperconn.c",
         "np_gemma/csrc/deltanet.c", "np_gemma/gpu.py", "np_gemma/cops.py", "np_gemma/qwen.py",
         "np_gemma/csrc/gpu.cu", "scripts/*.py", "*.md"]
QUESTION = """Questions about the source code above. Answer all of them in full, with the names of the files, functions, and records.

1. Write a technical report on this runtime for a new engineer. For each of these parts, explain what it does, how the data flows through it, and which functions implement it: the GGUF reader and the NVFP4 converter; the program of records and its CPU interpreter; the MoE layer on the CPU; the GPU path (hot experts, mixed groups, the plan, the copies); the QSA attention and its indexer; the hyper-connections and the n-gram (PLE) layer; MTP.
2. Explain the KQ_NVX layout of the NVFP4 experts byte by byte, and why it suits both the CPU kernel and the GPU tensor cores.
3. List ten concrete risks or bugs you can see in the code, each with the file, the function, and a proposed fix.
4. Propose a plan of five steps to make the prompt pass faster on a GPU with 8 GB of free memory, with the expected gain of each step."""


def sample(logits, temp, top_p, top_k, rng):
    """A token from logits: greedy for temp 0, else top-k, then top-p."""
    if temp <= 0:
        return int(np.argmax(logits))
    x = logits.astype(np.float64) / temp
    idx = np.argpartition(-x, top_k)[:top_k] if top_k and top_k < x.size else np.arange(x.size)
    v = x[idx]
    order = np.argsort(-v)
    idx, v = idx[order], v[order]
    p = np.exp(v - v.max())
    p /= p.sum()
    keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
    idx, p = idx[:keep], p[:keep] / p[:keep].sum()
    return int(rng.choice(idx, p=p))


def build_prompt(tok, budget, files, question):
    head = "<|im_start|>user\nHere is the source code of a project.\n\n"
    tail = "\n\n" + question + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    left = budget - len(tok.encode(head)) - len(tok.encode(tail))
    parts, used = [], []
    paths = []
    for pat in files:
        for p in sorted(glob.glob(os.path.join(ROOT, pat))):
            p = os.path.relpath(p, ROOT)
            if p not in paths and os.path.isfile(os.path.join(ROOT, p)):
                paths.append(p)
    for p in paths:
        if left <= 0:
            break
        try:
            text = "=== %s ===\n%s\n" % (p, open(os.path.join(ROOT, p), encoding="utf-8").read())
        except UnicodeDecodeError:
            continue
        ids = tok.encode(text)
        if len(ids) > left:
            ids = ids[:left]
            text = tok.decode(ids)
        parts.append(text)
        used.append((p, len(ids)))
        left -= len(ids)
    ids = tok.encode(head + "".join(parts) + tail)
    return ids, used


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-m", "--model", default=MODEL)
    ap.add_argument("--tok", default=None, help="tokenizer.json (default: next to the model)")
    ap.add_argument("--tokens", type=int, default=131072, help="the tokens of the prompt")
    ap.add_argument("--gen", type=int, default=4096, help="the most new tokens")
    ap.add_argument("--backend", choices=("gpu", "cpu"), default="gpu")
    ap.add_argument("--hot-gb", type=float, default=None,
                    help="gpu: GB of hot experts (default: the free memory after the cache)")
    ap.add_argument("--question", default=QUESTION)
    ap.add_argument("--files", nargs="*", default=FILES,
                    help="glob patterns of the files, from the root of the project")
    ap.add_argument("--temp", type=float, default=0.7, help="0: greedy")
    ap.add_argument("--top-p", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16384, help="the tokens of each prompt part")
    ap.add_argument("--out", default="long_qwen4_out.txt", help="the answer and the rates")
    args = ap.parse_args()

    tok = QwenTokenizer(args.tok or os.path.join(os.path.dirname(args.model), "tokenizer.json"))
    t0 = time.time()
    ids, used = build_prompt(tok, args.tokens, args.files, args.question)
    print("prompt: %d tokens from %d files (%.0f s to tokenize)" % (len(ids), len(used), time.time() - t0))
    stop = set(tok.stop_ids)

    t0 = time.time()
    m = Qwen4CPU(args.model)
    cache = Qwen4Cache(m.cfg, len(ids) + args.gen + 16)
    g = None
    if args.backend == "gpu":
        from np_gemma.qwen4_gpu import Qwen4GPU
        g = Qwen4GPU(m, hot_gb=args.hot_gb, ctx=len(ids) + args.gen + 16)
        g.attach(cache)
    print("model: %.0f s (dense %s%s)" % (time.time() - t0, m.dense,
                                           ", %d hot experts in each layer" % g.n_slots if g else ""))

    # the prompt, in parts (the rate of each part)
    t_prompt = time.time()
    h = None
    for c0 in range(0, len(ids), args.chunk):
        part = ids[c0:c0 + args.chunk]
        t1 = time.time()
        if g is not None:
            g.prefill(part, pos=c0)
        else:
            h = m.forward(part, cache, start_pos=c0)
        dt = time.time() - t1
        print("  prompt %6d .. %6d: %.1f s (%.0f tok/s)" % (c0, c0 + len(part), dt, len(part) / dt),
              flush=True)
    t_prompt = time.time() - t_prompt
    logits = g.logits() if g is not None else m.logits(h[-1:])[0]
    logits = np.asarray(logits).reshape(-1)
    print("prompt: %d tokens in %.1f s (%.0f tok/s)\n%s" % (len(ids), t_prompt, len(ids) / t_prompt,
                                                         "-" * 72), flush=True)

    rng = np.random.default_rng(args.seed)
    out, pos, shown = [], len(ids), ""
    nxt = sample(logits, args.temp, args.top_p, args.top_k, rng)
    t_dec = time.time()
    while True:
        out.append(nxt)
        if nxt in stop or len(out) >= args.gen:
            break
        if len(out) % 8 == 0:
            text = tok.decode(out, skip_special=True)
            print(text[len(shown):], end="", flush=True)
            shown = text
        if g is not None:
            g.step(nxt, pos)
            logits = g.logits()
        else:
            logits = m.logits(m.forward([nxt], cache, start_pos=pos))[0]
        pos += 1
        nxt = sample(np.asarray(logits).reshape(-1), args.temp, args.top_p, args.top_k, rng)
    t_dec = time.time() - t_dec
    text = tok.decode(out, skip_special=True)
    print(text[len(shown):], flush=True)
    n = len(out)
    stats = ("%s\nprompt: %d tokens in %.1f s (%.0f tok/s); decode: %d tokens in %.1f s (%.2f tok/s)%s"
             % ("-" * 72, len(ids), t_prompt, len(ids) / t_prompt, n, t_dec, n / t_dec,
                "; stopped at the end token" if out[-1] in stop else "; stopped at --gen"))
    print(stats)
    with open(args.out, "w") as f:
        f.write("files: %s\n\n%s\n%s\n" % (", ".join("%s (%d)" % u for u in used), text, stats))
    print("written to %s" % args.out)
    if g is not None:
        g.close()
    return 0


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
