"""Measure how much of a prefill runs inside the C kernels.

    python scripts/measure_prefill_glue.py --gguf PATH [--tokens 256]

The script wraps every ctypes kernel that the prefill calls and adds the wall
time. The rest of the time is the NumPy and Python glue. It also wraps the
decoder layer and the NumPy parts of the attention, so the glue is easy to
place.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import KVCache, Model, cops, ops  # noqa: E402
from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.tokenizer import Tokenizer  # noqa: E402

KERNELS = [
    "quantize_q8_t", "int4_q8_tile", "quantize_q8_t_moe", "int4_q8_moe",
    "gelu_mul", "moe_scatter", "int4_multi4", "attn_decode", "router",
    "qkv_norm", "rope_apply", "rms_norm", "gelu", "linear_q6k",
]
TIME = {}
COUNT = {}
ACC = {}


def acc(name, dt):
    a = ACC.setdefault(name, [0.0, 0])
    a[0] += dt
    a[1] += 1


def wrap_kernels():
    for name in KERNELS:
        fn = getattr(cops, name, None)
        if fn is None:
            continue
        def make(fn, name):
            def w(*a, **k):
                t0 = time.perf_counter()
                r = fn(*a, **k)
                acc(name, time.perf_counter() - t0)
                return r
            return w
        setattr(cops, name, make(fn, name))


def wrap_call(holder, name, label):
    orig = getattr(holder, name)

    def w(*a, **k):
        t0 = time.perf_counter()
        r = orig(*a, **k)
        acc(label, time.perf_counter() - t0)
        return r
    setattr(holder, name, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    args = ap.parse_args()
    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    ids = tok.encode("The capital of France is")
    ids = (ids * (args.tokens // len(ids) + 1))[:args.tokens]
    cache = KVCache(cfg, max_len=len(ids) + 8)
    wrap_kernels()
    wrap_call(ops, "moe_int4_q8", "py:moe_int4_q8")
    wrap_call(ops, "softmax", "np:softmax")
    wrap_call(ops, "gelu_tanh", "np:gelu_tanh")
    wrap_call(np, "matmul", "np:matmul")
    wrap_call(Model, "_decoder_layer", "py:_decoder_layer")
    wrap_call(Model, "_attention", "py:_attention")
    wrap_call(Model, "_moe", "py:_moe")
    wrap_call(Model, "linear", "py:linear")
    wrap_call(KVCache, "write", "py:kv_write")
    t0 = time.perf_counter()
    model.prefill(ids, cache)
    total = time.perf_counter() - t0
    print("prefill %d tokens %.3f s" % (len(ids), total))
    csum = 0.0
    usum = 0.0
    for name, (t, c) in sorted(ACC.items(), key=lambda kv: -kv[1][0]):
        if name.startswith("py:") or name.startswith("np:"):
            usum += t
        else:
            csum += t
        print("%-22s %9.1f %8d %7.1f%%" % (name, t * 1e3, c, 100.0 * t / total))
    print("%-22s %9.1f %8s %7.1f%%" % ("SUM in C", csum * 1e3, "", 100.0 * csum / total))
    print("%-22s %9.1f %8s %7.1f%%" % ("wrapped py/np", usum * 1e3, "", 100.0 * usum / total))
    print("%-22s %9.1f %8s %7.1f%%" % ("other/untimed", (total - csum - usum) * 1e3, "",
                                       100.0 * (total - csum - usum) / total))
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
