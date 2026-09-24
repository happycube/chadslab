"""Compare the plain, flash, and slide attention paths on the model.

slide uses the C kernel for a sliding layer and the batched matmul for a global
layer. Run:  PYTHONPATH=. python scripts/bench_flash_model.py [n ...]
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from np_gemma import Model, Session, Tokenizer, cops
from np_gemma.chat import render_chat
from np_gemma.config import Config
from np_gemma.gguf import GGUF

P = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"
MODES = [("plain", "0"), ("flash", "1"), ("slide", "slide")]


def run(model, ids, mode):
    os.environ["NP_GEMMA_FLASH"] = mode
    cops.attn_prefill_impl(0)
    s = Session(model, max_len=len(ids) + 8)
    t0 = time.perf_counter()
    s.prefill(ids)
    dt = time.perf_counter() - t0
    logits = model.logits(s._x[-1:])[0].copy()
    out = list(s.generate(ids, max_new_tokens=6))
    del s
    return dt, logits, out[len(ids):]


def main():
    want = [int(a) for a in sys.argv[1:]] or [2048, 4096]
    g = GGUF(P)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    prompt = tok.encode(render_chat(
        [{"role": "user", "content": "Here is a README. Summarise it."}],
        add_generation_prompt=True, enable_thinking=False))
    for n in want:
        ids = (prompt * (n // len(prompt) + 1))[:n]
        base_dt = None
        base_tok = None
        base_l = None
        for name, mode in MODES:
            dt, logits, toks = run(model, ids, mode)
            if base_dt is None:
                base_dt, base_tok, base_l = dt, toks, logits
                extra = ""
            else:
                extra = "  speedup %5.2fx  logit maxdiff %.3e  greedy %s" % (
                    base_dt / dt, float(np.abs(logits - base_l).max()),
                    "SAME" if toks == base_tok else "DIFF")
            print("n=%-6d %-6s %7.2fs%s" % (n, name, dt, extra), flush=True)
        print("        tokens %s" % base_tok, flush=True)
    os.environ["NP_GEMMA_FLASH"] = "0"


main()
