#!/usr/bin/env python3
"""Compare the logits of decode steps with the Hugging Face reference.

The reference is Gemma4ForCausalLM of transformers with the weights of the
GGUF file, in float32 (scripts/hf_gguf_reference.py builds it the same way).
Both sides then use the same weights, so a difference comes from the
arithmetic of the forward pass.

Two steps, so that the two models never hold memory at the same time:

    PYTHONPATH=. python scripts/check_hf_decode.py hf --out hf.npz
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/check_hf_decode.py np --ref hf.npz

The hf step runs the whole prompt in one pass and keeps the logits of the
last rows. The np step runs the prompt pass on the first tokens. Then it runs
one decode step for each remaining token, with the true token as input. It
compares four settings of this runtime:

1. The int8 cache with the program of a decode step.
2. The int8 cache with the Python loop.
3. The float cache with the C attention kernel and the program.
4. The float cache with the NumPy attention of the old code.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import time

import numpy as np

GGUF_PATH = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"


def prompt_ids(tok, n):
    return tok.encode(open(os.path.join(os.path.dirname(__file__), "..", "README.md")).read())[:n]


def run_hf(args):
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

    from np_gemma.gguf import GGUF
    from np_gemma.tokenizer import Tokenizer

    here = os.path.dirname(__file__)
    spec = importlib.util.spec_from_file_location(
        "hf_gguf_reference", os.path.join(here, "hf_gguf_reference.py"))
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    ids = prompt_ids(tok, args.tokens)
    known = set(g.names())
    torch.set_default_dtype(torch.float32)
    model = Gemma4ForCausalLM(ref.make_config(g.text_config())).eval()
    t0 = time.perf_counter()
    count = 0
    with torch.no_grad():
        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            if not n.startswith("model."):
                continue
            rt = "model.language_model." + n[len("model."):]
            if rt in known:
                p.copy_(torch.from_numpy(g.get(rt)))
                count += 1
    print("loaded %d tensors in %.0f s" % (count, time.perf_counter() - t0), flush=True)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
    logits = out.logits[0].float().numpy()
    print("forward of %d tokens in %.0f s" % (len(ids), time.perf_counter() - t0))
    n0 = len(ids) - args.steps
    np.savez(args.out, ids=np.array(ids), n0=n0, logits=logits[n0 - 1:])
    print("saved rows %d to %d of the logits to %s" % (n0 - 1, len(ids) - 1, args.out))


def run_np(args):
    import np_gemma.model as model_mod
    from np_gemma import KVCache, Model
    from np_gemma.config import Config
    from np_gemma.gguf import GGUF

    r = np.load(args.ref)
    ids = [int(x) for x in r["ids"]]
    n0 = int(r["n0"])
    ref = r["logits"]
    g = GGUF(args.gguf)
    cfg = Config({"text_config": g.text_config()})
    model = Model(g, cfg).load_all(dtype="int4")

    settings = [
        ("int8 cache, program", "1", True, True),
        ("int8 cache, Python loop", "1", True, False),
        ("float cache, C kernel, program", "0", True, True),
        ("float cache, NumPy (old)", "0", False, False),
    ]
    outs = {}
    for name, attn, f32c, prog in settings:
        os.environ["NP_GEMMA_ATTN"] = attn
        model_mod._F32_ATTN_C = f32c
        model_mod._PROGRAM = prog
        cache = KVCache(cfg, max_len=len(ids) + 8)
        x = model.prefill(ids[:n0], cache)
        rows = [model.logits(x[-1:])[0]]
        for k in range(len(ids) - n0):
            x = model.forward([ids[n0 + k]], cache=cache, start_pos=n0 + k)
            rows.append(model.logits(x)[0])
        outs[name] = np.stack(rows)

    scale = float(np.abs(ref).max())
    print("reference: %d rows, max |logit| %.2f" % (ref.shape[0], scale))
    print("%-32s %10s %10s %10s %8s" % ("setting", "max |d|", "mean |d|", "max rel", "top-1"))
    for name, _a, _f, _p in settings:
        d = np.abs(outs[name] - ref)
        top = np.mean(outs[name].argmax(-1) == ref.argmax(-1))
        print("%-32s %10.4f %10.5f %10.2e %7.0f%%" % (
            name, d.max(), d.mean(), d.max() / scale, 100 * top))
    a = outs["float cache, C kernel, program"]
    b = outs["float cache, NumPy (old)"]
    print("float cache, C kernel against NumPy: max |d| %.2e" % np.abs(a - b).max())
    c = outs["int8 cache, program"]
    e = outs["int8 cache, Python loop"]
    print("int8 cache, program against Python loop: %s" % (
        "same bits" if np.array_equal(c, e) else "max |d| %.2e" % np.abs(c - e).max()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("mode", choices=("hf", "np"))
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--tokens", type=int, default=208)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--out", default="hf_decode.npz")
    ap.add_argument("--ref", default="hf_decode.npz")
    args = ap.parse_args()
    if args.mode == "hf":
        run_hf(args)
    else:
        run_np(args)


if __name__ == "__main__":
    main()
