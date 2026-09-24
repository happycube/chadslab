"""Check that a sliding layer keeps about one window of keys.

The output must not change. The code drops only the rows that the window hides
for every query of the block.

Run:  PYTHONPATH=. python scripts/check_kv_window.py [n ...]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import Model, Session, Tokenizer
from np_gemma.chat import render_chat
from np_gemma.config import Config
from np_gemma.gguf import GGUF

P = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"


def cache_bytes(cache, i):
    n = 0
    for a in (cache.k[i], cache.v[i], cache.kq[i], cache.ks[i],
              cache.vq[i], cache.vs[i]):
        if a is not None:
            n += a.nbytes
    return n


def main():
    want = [int(a) for a in sys.argv[1:]] or [2048, 4096, 8192]
    g = GGUF(P)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    prompt = tok.encode(render_chat(
        [{"role": "user", "content": "Here is a README. Summarise it."}],
        add_generation_prompt=True, enable_thinking=False))
    w = cfg.sliding_window
    for n in want:
        ids = (prompt * (n // len(prompt) + 1))[:n]
        s = Session(model, max_len=n + 8)
        t0 = time.perf_counter()
        s.prefill(ids)
        dt = time.perf_counter() - t0
        slide, glob, total = [], [], 0
        for i in range(cfg.num_hidden_layers):
            if s.cache.k[i] is None:
                continue
            rows = s.cache.end[i] - s.cache.base[i]
            (slide if cfg.plan[i].is_sliding else glob).append(rows)
            total += cache_bytes(s.cache, i)
        out = list(s.generate(ids, max_new_tokens=6))[len(ids):]
        print("n=%-6d %6.1fs  cache %6.2f GB  sliding rows %d..%d of window %d"
              "  global rows %d" % (n, dt, total / 1e9, min(slide), max(slide), w,
                                    max(glob)), flush=True)
        print("        greedy %s" % out, flush=True)
        del s


main()
