"""Check the sliding-window key slice.

The plain attention path keeps only the keys that a query in the block can
see. A key outside the range is hidden for every query, so the softmax gives
it zero and the result does not change. This compares the path with the slice
against the path without it: the last-token logits and the greedy tokens must
agree. It reports the time too.

The slice is exact in the real numbers. It is not bit exact: the value matmul
sums over fewer keys, so the rounding changes. The router then picks its
experts with an argmax and a rounding change can move the choice. The two
paths therefore drift apart, and the test is the greedy token.

Run:  PYTHONPATH=. python scripts/check_slide.py [n ...]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import Model, Tokenizer, Session
from np_gemma.chat import render_chat
from np_gemma.config import Config
from np_gemma.gguf import GGUF

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
NTOK = 8


def run(model, ids, slide):
    os.environ["NP_GEMMA_SLIDE"] = str(slide)
    os.environ["NP_GEMMA_FLASH"] = "0"
    s = Session(model, max_len=len(ids) + NTOK + 8)
    t0 = time.perf_counter()
    out = s.generate(ids, max_new_tokens=NTOK)
    dt = time.perf_counter() - t0
    logits = model.logits(s._x)[0].copy()
    return dt, out[len(ids):], logits, s


def main():
    want = [int(a) for a in sys.argv[1:]] or [2048, 4096]
    g = GGUF(GGUF_PATH)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    wins = [cfg.plan[i].window for i in range(cfg.num_hidden_layers)]
    print("windows:", wins[:6], "...  global layers:",
          [i for i, w in enumerate(wins) if not w])
    base = tok.encode(render_chat(
        [{"role": "user", "content": "Here is a README. Summarise it."}],
        add_generation_prompt=True, enable_thinking=False))
    ok = True
    for n in want:
        ids = (base * (n // len(base) + 1))[:n]
        dt0, g0, l0, s0 = run(model, ids, 0)
        dt1, g1, l1, s1 = run(model, ids, 1)
        dl = float(np.abs(l0 - l1).max())
        same = g0 == g1
        print("n=%-6d no-slide %7.2fs  slide %7.2fs  speedup %.2fx  logit maxdiff %.3e"
              % (n, dt0, dt1, dt0 / dt1, dl))
        print("        greedy no-slide %s" % g0)
        print("        greedy slide    %s" % g1)
        if not same:
            print("        first split at token", next(i for i, (a, b) in enumerate(zip(g0, g1)) if a != b))
        ok = ok and same
        del s0, s1
    print("RESULT", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
