#!/usr/bin/env python3
"""Check the groups of Qwen3.6 (GGUF) on the GPU.

The checks: small groups, the MTP verify group with commit, and the
prompt pass.

QWEN_PLAN.md, phase 4. HotCache is off (NP_GEMMA_GPU_HOT_DYN=0) for the
first checks. A hot expert runs on the GPU (float32 x) and a cold one on
the CPU (int8 x), so the bits depend on the slots.

1. a small group of 4 tokens gives the bits of 4 steps;
2. verify of 4 tokens, commit(2), then 2 steps: the bits of 4 steps (the
   rows, and the state of the linear layers);
3. the prompt pass on the GPU and on the CPU: the top tokens of the next
   token, and the rate for some lengths. The GPU uses
   large groups (the experts copied) and split groups.

    OPENBLAS_NUM_THREADS=1 python scripts/check_qwen_gpu_groups.py
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

os.environ.setdefault("NP_GEMMA_GPU_HOT_DYN", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import ops  # noqa: E402
from np_gemma.qwen import QwenCache, QwenGGUFProgram  # noqa: E402
from np_gemma.qwen_gpu import QwenGPU  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
TOK = "models/Qwen3.6-35B-A3B-OptiQ-4bit/tokenizer.json"


def text_ids(tok, n):
    """About n tokens of the files of this repository, as a chat prompt."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    body = open(os.path.join(root, "QWEN_PLAN.md")).read() + open(os.path.join(root, "README.md")).read()
    ids = tok.encode(body)[:n]
    head = tok.encode("<|im_start|>user\nSummarize this text:\n")
    tail = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return head + ids[:max(0, n - len(head) - len(tail))] + tail


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--tok", default=TOK)
    ap.add_argument("--lengths", default="100,600,1500")
    args = ap.parse_args()
    tok = QwenTokenizer(args.tok)
    m = QwenGGUFProgram(args.path)
    cfg = m.cfg
    g = QwenGPU(m)
    print("%d hot experts in each layer" % g.n_slots)
    ok = True

    ids = tok.encode("<|im_start|>user\nName three prime numbers.<|im_end|>\n<|im_start|>assistant\n")
    n0 = len(ids)
    base = QwenCache(cfg, 4096)
    m.forward(ids, base)
    more = [760, 1156, 3383, 1043]

    def steps(cache, toks, pos):
        rows = []
        for j, x in enumerate(toks):
            g.step(x, pos + j)
            g.g.download("xn")
            rows.append(g.prog.names["xn"].copy())
        return np.concatenate(rows)

    c1 = copy.deepcopy(base)
    g.attach(c1)
    r_steps = steps(c1, more, n0)
    g.detach(c1)

    c2 = copy.deepcopy(base)
    g.attach(c2)
    r_group = g.group(more, n0)
    g.detach(c2)
    same = np.array_equal(r_group, r_steps)
    lin = [i for i in range(cfg.num_hidden_layers) if cfg.layer_types[i] != "full_attention"]
    same_s = all(np.array_equal(c1.state[i], c2.state[i]) for i in lin)
    print("group of 4 = 4 steps: rows %s, states %s (max diff %.1e)" % (
        same, same_s, np.abs(r_group - r_steps).max()))
    ok &= same and same_s

    c3 = copy.deepcopy(base)
    g.attach(c3)
    r_ver = g.verify(more, n0)
    g.commit(2)
    r_rest = steps(c3, more[2:], n0 + 2)
    g.detach(c3)
    same_v = np.array_equal(r_ver, r_steps) and np.array_equal(r_rest, r_steps[2:])
    same_c = all(np.array_equal(c1.state[i], c3.state[i]) for i in lin) and \
        all(np.array_equal(c1.conv[i], c3.conv[i]) for i in lin)
    print("verify of 4, commit(2), 2 steps = 4 steps: rows %s, states %s" % (same_v, same_c))
    ok &= same_v and same_c

    for n in [int(x) for x in args.lengths.split(",")]:
        pids = text_ids(tok, n)
        cc = QwenCache(cfg, len(pids) + 1100)
        t0 = time.time()
        hc = m.forward(pids, cc)
        tc = time.time() - t0
        lc = m.logits(hc[-1:])[0]
        cg = QwenCache(cfg, len(pids) + 1100)
        g.attach(cg)
        g.prefill(pids)                 # the first run builds the programs
        g.detach(cg)
        cg = QwenCache(cfg, len(pids) + 1100)
        g.attach(cg)
        t0 = time.time()
        g.prefill(pids)
        lg = g.logits()
        tg = time.time() - t0
        g.detach(cg)
        top_c, top_g = np.argsort(-lc)[:5], np.argsort(-lg)[:5]
        both = len(set(top_c) & set(top_g))
        print("prompt of %d: GPU %.2f s (%.0f tok/s), CPU %.2f s (%.0f tok/s); top token %s; "
              "top 5 shared %d" % (len(pids), tg, len(pids) / tg, tc, len(pids) / tc,
                                   top_c[0] == top_g[0], both))
        ok &= both >= 3
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
