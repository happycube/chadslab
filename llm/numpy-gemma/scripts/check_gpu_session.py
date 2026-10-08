#!/usr/bin/env python3
"""Check a chat Session of the 26B on the GPU when a turn cuts the history.

A client can send a history that differs from the cache before its end (the
Gemma 4 template drops the thought part of an earlier answer). The Session
then cuts the cache back to the common prefix. On the GPU, a layer with a
window drops its oldest rows (GPUKV.prepare). A cut to n needs the window
rows before n; the GPU once kept its base while the end went back, and the
next prompt pass wrote before the buffer (an illegal memory access).

The script runs a long prompt and some steps in a Session. It then sends
a history cut back by each count of --cuts tokens, with a new question, and
gives the Session --gen more tokens. A reference Session makes the same rows
in the same way with no cut: the logits must have the same bits. A cut that
leaves a window without its rows must start again from an empty cache.

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/check_gpu_session.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NP_GEMMA_GPU"] = "1"
# A fixed hot set and no mixed groups: the place of an expert (the GPU or the
# CPU) then does not follow the text, so the two Sessions compute the same
# rows the same way.
os.environ.setdefault("NP_GEMMA_GPU_HOT_DYN", "0")
os.environ.setdefault("NP_GEMMA_GPU_MIX", "0")

from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.model import Model, Session  # noqa: E402
from np_gemma.tokenizer import Tokenizer  # noqa: E402
from np_gemma import gpu  # noqa: E402

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def feed(m, s, ids, forced, gen=None):
    """Put ids in the Session s, then run gen more tokens: the tokens of
    forced, or the greedy tokens. Return the logits of the last prompt row
    and of each step."""
    gen = gen or FEED_GEN[0]
    s.prefill(ids)
    x = s._x
    out = [m.logits(x[-1:])[0]]
    pos = len(ids)
    for j in range(gen - 1):
        t = forced[j] if forced is not None else int(np.argmax(out[-1]))
        x = m.forward([t], cache=s.cache, start_pos=pos)
        s.ids.append(t)
        pos += 1
        out.append(m.logits(x[-1:])[0])
    return out


FEED_GEN = [24]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--chars", type=int, default=30000, help="the characters of README.md")
    ap.add_argument("--steps", type=int, default=48, help="the steps of the first turn")
    ap.add_argument("--gen", type=int, default=24, help="the steps of each later turn")
    ap.add_argument("--cuts", default="1300,300,40",
                    help="cut the history back by these counts of tokens")
    ap.add_argument("--hot-gb", type=float, default=1.5)
    ap.add_argument("--tol", type=float, default=1e-3, help="the most |d| of the logits")
    args = ap.parse_args()
    FEED_GEN[0] = args.gen
    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    m = Model(g, cfg).load_all(dtype="int4")
    gpu.offload(m, args.hot_gb)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "README.md")).read()[:args.chars]
    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text + "\n\nSummarize the text above."}],
        add_generation_prompt=True, thinking=False))
    max_len = len(ids) + args.steps + 4096
    new = tok.encode("\n\nNow give one sentence on the GPU work.")
    ok = True
    for cut in (int(c) for c in args.cuts.split(",") if c.strip()):
        # Turn 1 in a new Session. The cache of the Session is then on the
        # GPU at the cut.
        s = Session(m, max_len=max_len)
        s.generate(ids, args.steps)
        kv = m._gpu.kv
        hist = list(s.ids)
        n = len(hist) - cut
        ids2 = hist[:n] + list(new)
        bases = [kv.base[i] for i in range(cfg.num_hidden_layers) if cfg.plan[i].is_sliding]
        la = feed(m, s, ids2, None)
        toks = [int(np.argmax(v)) for v in la]
        reused = len(ids2) - s.prefilled
        del s
        # The reference makes the rows before n as the Session did: when it
        # reused the cache, the same prompt pass and then steps up to n (no
        # cut); when it started again, a fresh prompt pass. Then the same
        # tokens. A right cut gives the same bits.
        r = Session(m, max_len=max_len)
        if reused:
            r.prefill(ids[:n])
            for p in range(len(ids), n):
                m.forward([hist[p]], cache=r.cache, start_pos=p)
                r.ids.append(hist[p])
        lb = feed(m, r, ids2, toks)
        del r
        ref = toks
        same = sum(int(np.argmax(u)) == t for u, t in zip(la, ref))
        d = max(float(np.abs(u - v).max()) for u, v in zip(la, lb))
        good = same >= len(ref) - 1 and d < args.tol
        ok &= good
        print("history %d, cut %5d: to %d, GPU window base %d..%d, reused %d, the same top "
              "token %d/%d, logits max |d| %.3f %s" % (
                  len(hist), cut, n, min(bases), max(bases), reused, same, len(ref), d,
                  "ok" if good else "DIFFERENT"), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
