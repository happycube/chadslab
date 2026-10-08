"""Measure the prompt pass against the prefill block size.

A larger block makes the mixture-of-experts GEMM see more tokens for each
expert. The block also makes each attention call wider. This script separates
the two effects on the real model.

Run:  PYTHONPATH=. python scripts/bench_prefill_chunk.py [n]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import Model, Session, Tokenizer
from np_gemma.chat import render_chat
from np_gemma.config import Config
from np_gemma.gguf import GGUF

P = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
    g = GGUF(P)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    prompt = tok.encode(render_chat(
        [{"role": "user", "content": "Here is a README. Summarise it."}],
        add_generation_prompt=True, enable_thinking=False))
    ids = (prompt * (n // len(prompt) + 1))[:n]
    os.environ["NP_GEMMA_FLASH"] = "slide"
    print("n=%d, per-expert tokens at block B = B*8/128 = B/16" % n)
    print("%-8s %10s %8s %10s %8s" % ("block", "time", "tok/s", "expert M", "top1"))
    chunks = [int(a) for a in sys.argv[2:]] or [128, 256, 512, 1024, 2048]
    for chunk in chunks:
        model.prefill_chunk = chunk
        s = Session(model, max_len=n + 8)
        t0 = time.perf_counter()
        s.prefill(ids)
        dt = time.perf_counter() - t0
        lg = model.logits(s._x[-1:])[0]
        print("%-8d %9.2fs %8.1f %10d %8d"
              % (chunk, dt, n / dt, chunk * 8 // 128, int(np.argmax(lg))), flush=True)
        del s
    model.prefill_chunk = 256


main()
