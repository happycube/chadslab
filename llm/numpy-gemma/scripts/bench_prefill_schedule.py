"""Compare the token-major and layer-major prompt schedules.

Token-major runs every layer for one block, then the next block. Layer-major
runs one layer for the whole prompt, then the next layer.

Run:  PYTHONPATH=. python scripts/bench_prefill_schedule.py [n]
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

P = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"
MODES = [
    ("token", 256, "slide"),
    ("layer", 256, "slide"),
    ("token", 256, "1"),
    ("layer", 256, "1"),
    ("layer", 0, "1"),
]


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
    print("n=%d" % n)
    print("%-7s %-7s %-7s %9s %8s %8s" % ("schedule", "block", "flash", "time", "tok/s", "top1"))
    ref = None
    for sched, block, flash in MODES:
        os.environ["NP_GEMMA_FLASH"] = flash
        s = Session(model, max_len=n + 8)
        t0 = time.perf_counter()
        if sched == "token":
            model.prefill_chunk = block
            s.prefill(ids)
        else:
            s._x = model.prefill_layer_major(ids, s.cache, 0, None, chunk=block)
            s.ids = list(ids)
        dt = time.perf_counter() - t0
        lg = model.logits(s._x[-1:])[0]
        top = int(np.argmax(lg))
        if ref is None:
            ref = top
        print("%-7s %-7s %-7s %8.2fs %8.1f %8d%s"
              % (sched, block if block else "all", flash, dt, n / dt, top,
                 "" if top == ref else "  DIFF"), flush=True)
        del s
    model.prefill_chunk = 256


main()
