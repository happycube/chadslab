"""Measure what the session cache saves on a multi-turn conversation.

The first turn reads the whole prompt. A later turn reuses the key and value
cache of the shared prefix, so it reads only the new tokens.

Run:  PYTHONPATH=. python scripts/bench_prefix.py
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

P = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def main():
    g = GGUF(P)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    para = ("The quick brown fox jumps over the lazy dog near the river bank. "
            "A second sentence follows it to make the paragraph longer. ")
    msgs = [{"role": "user",
             "content": "Read this report.\n\n" + para * 150 + "\n\nName the animal in it."}]
    s = Session(model, max_len=8192)
    total = 0.0
    cold = None
    print("%-6s %10s %10s %10s" % ("turn", "prompt", "prefill", "cumulative"))
    for turn in range(1, 5):
        ids = tok.encode(render_chat(msgs, add_generation_prompt=True, enable_thinking=False))
        t0 = time.perf_counter()
        s.prefill(ids)
        dt = time.perf_counter() - t0
        total += dt
        if cold is None:
            cold = dt
        print("%-6d %10d %9.2fs %9.2fs" % (turn, len(ids), dt, total), flush=True)
        reply = "The animal is the fox."
        msgs = msgs + [{"role": "assistant", "content": reply},
                       {"role": "user", "content": "Say it again."}]
    print("total %.2fs; four cold prefills would cost about %.2fs  (reuse saves %.0f%%)"
          % (total, cold * 4, 100.0 * (1 - total / (cold * 4))))


main()
