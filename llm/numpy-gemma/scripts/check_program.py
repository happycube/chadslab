#!/usr/bin/env python3
"""Check the program of a decode step against the Python path, bit for bit.

PERF_PLAN.md, phase 2. For each layer of --layers, the script runs the
Python layer (Model._decoder_layer) and the program of the same layer. The
two start from the same input and the same cache. The script compares the
hidden state and the new cache row. It runs the program in C and with the Python interpreter. When a
result differs, it runs the records one prefix at a time and names the first
record whose output differs.

It then runs all the layers as one program and compares the result with the
Python loop over the layers. It prints the time of each. Last, it runs eight
decode steps through Model.forward with the program off and on.

Run it with NP_GEMMA_ATTN=0 as well, for the float cache.

    OPENBLAS_NUM_THREADS=1 OMP_WAIT_POLICY=ACTIVE PYTHONPATH=. \\
        python scripts/check_program.py
"""
from __future__ import annotations

import os
# The records of this check read the int4 matrices of the model: the parts
# split them (np_gemma/parts.py), and the Python interpreter of the program
# has no KQ_Q4X. The KQ_Q4X copies stay off (NP_GEMMA_Q4X, np_gemma/ops.py).
os.environ["NP_GEMMA_Q4X"] = "0"
import argparse
import time

import numpy as np

import np_gemma.model as model_mod
from np_gemma import KVCache, Model, ops
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.program import bind_step, compile_layers, format_form, layer_form, ready
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"
PARTS = ("kq", "ks", "vq", "vs")


def snap(cache, i):
    s = {n: None if getattr(cache, n)[i] is None else getattr(cache, n)[i].copy()
         for n in PARTS}
    s.update(base=cache.base[i], end=cache.end[i])
    return s


def restore(cache, i, s):
    for n in PARTS:
        getattr(cache, n)[i] = None if s[n] is None else s[n].copy()
    cache.base[i], cache.end[i] = s["base"], s["end"]


def rows(cache, i, pos):
    r = pos - cache.base[i]
    return [None if getattr(cache, n)[i] is None else getattr(cache, n)[i][r].copy()
            for n in PARTS]


def same_rows(a, b):
    return all((x is None and y is None) or np.array_equal(x, y) for x, y in zip(a, b))


def python_layer(model, cache, i, x, pos):
    plan = model.cfg.plan[i]
    positions = np.array([pos])
    cos, sin, ca, sa = model._rope(plan, positions)
    return model._decoder_layer(x.copy(), model.load_layer(i), plan, cos, sin,
                                positions, i, None, cache, ca, sa)


def program_layer(prog, model, cache, x, pos, python=False, limit=-1):
    bind_step(prog, model, cache, pos)
    prog.names["x"][:] = x
    if python:
        prog.run_py(limit)
    else:
        prog.run(limit)
    return prog.names["x"].copy()


def bisect(prog, model, cache, i, x, pos, saved):
    """Name the first record after which C and Python differ."""
    bufs = [a for a in prog.keep if a.dtype != np.uint8]
    for k in range(1, len(prog.recs) + 1):
        out = []
        for python in (False, True):
            restore(cache, i, saved)
            for a in bufs:
                a[...] = 0
            program_layer(prog, model, cache, x, pos, python=python, limit=k)
            out.append([a.copy() for a in bufs] + rows(cache, i, pos))
        if not all(np.array_equal(a, b) for a, b in zip(*out)):
            return k - 1
    return None


def check_e4b(g, tok, args):
    """Run steps of 1 to 8 tokens through E4B.forward, with the program off
    and on. Compare the hidden states and the logits."""
    import np_gemma.e4b as e4b_mod
    from np_gemma.e4b import E4B, E4BCache, E4BConfig
    cfg = E4BConfig({"text_config": g.text_config()})
    model = E4B(g, cfg, mode="int4")
    ids = tok.encode(open("README.md").read())
    ok = True
    sizes = [1, 3, 1, 2, 4, 1, 8, 1]
    for n in args.contexts:
        res = {}
        for on in (False, True):
            e4b_mod._PROGRAM = on
            cache = E4BCache(cfg, max_len=n + 64)
            model.forward(ids[:n], cache=cache)
            xs, ts = [], {}
            pos = n
            for rep in range(2):
                for t in sizes:
                    t0 = time.perf_counter()
                    x = model.forward(ids[pos:pos + t], cache=cache, start_pos=pos)
                    lg = model.logits(x)
                    ts.setdefault(t, []).append(time.perf_counter() - t0)
                    xs.append((x, lg))
                    pos += t
            res[on] = (xs, ts)
        same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                   for a, b in zip(res[False][0], res[True][0]))
        ok = ok and same
        print("E4B context %d, steps of %s tokens (two rounds): %s" % (
            n, sizes, "same" if same else "DIFFERENT"))
        for t in sorted(res[True][1]):
            print("  %d tokens, with logits: Python %.1f ms, program %.1f ms (last round)" % (
                t, 1000 * res[False][1][t][-1], 1000 * res[True][1][t][-1]))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--contexts", type=int, nargs="+", default=[200, 1100])
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--show", action="store_true", help="Print the form and the program.")
    ap.add_argument("--e4b", action="store_true",
                    help="The GGUF file is an E4B model. Check only the steps.")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    if args.e4b:
        return check_e4b(g, tok, args)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")
    ids = tok.encode(open("README.md").read())
    first_global = next(i for i, p in enumerate(cfg.plan) if not p.is_sliding)
    layers = args.layers or [0, first_global, cfg.num_hidden_layers - 1]
    if args.show:
        print(format_form(layer_form(model, layers[0])))
        print(compile_layers(model, [layers[0]]).dump())

    # the mode of the form of the cache (KVCache keeps only quantized rows)
    attn = ready(model, KVCache(cfg, max_len=16))
    print("attention mode:", attn)
    ok = True
    progs = {i: compile_layers(model, [i], attn) for i in layers}
    for n in args.contexts:
        cache = KVCache(cfg, max_len=n + 16)
        model.prefill(ids[:n], cache)
        pos = n
        for i in layers:
            x = model.forward([ids[pos]], cache=cache, start_pos=pos, max_layers=i)
            saved = snap(cache, i)
            x_py = python_layer(model, cache, i, x, pos)
            r_py = rows(cache, i, pos)
            res = []
            for python in (False, True):
                restore(cache, i, saved)
                x_p = program_layer(progs[i], model, cache, x, pos, python=python)
                res.append(np.array_equal(x_p, x_py) and same_rows(rows(cache, i, pos), r_py))
            ok = ok and all(res)
            kind = "sliding" if cfg.plan[i].is_sliding else "global"
            print("context %5d layer %2d (%s): C %s, Python interpreter %s, %d records"
                  % (n, i, kind, "same" if res[0] else "DIFFERENT",
                     "same" if res[1] else "DIFFERENT", len(progs[i].recs)))
            if not res[0]:
                k = bisect(progs[i], model, cache, i, x, pos, saved)
                if k is not None:
                    print("  first difference after record %d:" % k)
                    print("  " + progs[i].dump().split("code:\n")[1].split("\n")[k])
            restore(cache, i, saved)

    # All the layers as one program, against the Python loop.
    n = args.contexts[-1]
    cache = KVCache(cfg, max_len=n + 64)
    model.prefill(ids[:n], cache)
    allp = compile_layers(model, range(cfg.num_hidden_layers), attn)
    x0 = model.embed([ids[n]])
    saved = {i: snap(cache, i) for i in range(cfg.num_hidden_layers)}
    t_py, t_c = [], []
    for rep in range(5):
        for i in range(cfg.num_hidden_layers):
            restore(cache, i, saved[i])
        t0 = time.perf_counter()
        x = x0.copy()
        for i in range(cfg.num_hidden_layers):
            x = python_layer(model, cache, i, x, n)
        t_py.append(time.perf_counter() - t0)
        r_py = [rows(cache, i, n) for i in range(cfg.num_hidden_layers)]
        for i in range(cfg.num_hidden_layers):
            restore(cache, i, saved[i])
        t0 = time.perf_counter()
        bind_step(allp, model, cache, n)
        allp.names["x"][:] = x0
        allp.run()
        t_c.append(time.perf_counter() - t0)
        same = (np.array_equal(allp.names["x"], x)
                and all(same_rows(rows(cache, i, n), r_py[i])
                        for i in range(cfg.num_hidden_layers)))
    ok = ok and same
    print("all %d layers, %d records: %s. Python loop %.1f ms, program %.1f ms (best of 5)"
          % (cfg.num_hidden_layers, len(allp.recs), "same" if same else "DIFFERENT",
             1000 * min(t_py), 1000 * min(t_c)))

    # Steps through Model.forward, with the program off and on. A step of
    # more than one token is the verify group of MTP.
    sizes = [1, 3, 1, 2, 4, 1, 8, 1]
    res = {}
    for on in (False, True):
        model_mod._PROGRAM = on
        cache = KVCache(cfg, max_len=n + 64)
        model.prefill(ids[:n], cache)
        xs, ts = [], {}
        pos = n
        for t in sizes:
            t0 = time.perf_counter()
            x = model.forward(ids[pos:pos + t], cache=cache, start_pos=pos)
            lg = model.logits(x)
            ts.setdefault(t, []).append(time.perf_counter() - t0)
            xs.append((x, lg))
            pos += t
        res[on] = (xs, ts)
    same = all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
               for a, b in zip(res[False][0], res[True][0]))
    ok = ok and same
    print("steps of %s tokens through Model.forward: %s" % (sizes, "same" if same else "DIFFERENT"))
    for t in sorted(res[True][1]):
        print("  %d tokens, with logits: Python %.1f ms, program %.1f ms" % (
            t, 1000 * np.median(res[False][1][t]), 1000 * np.median(res[True][1][t])))
    print("PASS" if ok else "FAIL")
    g.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
