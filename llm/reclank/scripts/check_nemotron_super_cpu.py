#!/usr/bin/env python3
"""The CPU engine of Nemotron Super (reclank/models/nemotron/kq.py on Super's formats: FP8 in KQ_MXF8
rows with the tensor scale after, the NVFP4 experts in KQ_NV4 rows) against the NumPy reference
(models/nemotron/model.py), block by block: the program of the first N blocks (compile_nemotron_step
upto) on a real prompt, its output (the residual after them, normed by norm_f) against the
reference's at the same place. The experts of the blocks before N only are packed (1.6 GB a MoE
layer), so it runs beside other work.

    taskset -c 22,23,46,47 python scripts/check_nemotron_super_cpu.py \\
        /home/cpage/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --upto 2,8 --tokens 16
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.nemotron.engine import NemotronTok  # noqa: E402
from reclank.models.nemotron.kq import NemotronKQ, compile_nemotron_step  # noqa: E402
from reclank.models.nemotron.model import BB, NemotronCPU, rms_norm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--upto", default="2,8", help="the block counts to check (comma list)")
    ap.add_argument("--tokens", type=int, default=16)
    a = ap.parse_args()
    P = a.ckpt
    tok = NemotronTok(os.path.join(P, "tokenizer.json"), os.path.join(P, "chat_template.jinja"))
    readme = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "README.md")
    para = next(p for p in open(readme).read().split("\n\n")
                if len(p) > 200 and not p.startswith(("#", "|", "`", "    ")))
    ids = tok.encode(para)[:a.tokens]
    ups = sorted(int(v) for v in a.upto.split(","))
    ref = NemotronCPU(P, keep_dense=False)
    eps = ref.cfg.eps
    seen = {}
    ref.hook = lambda i, x: seen.__setitem__(i, x.copy()) if i + 1 in ups else None
    t0 = time.time()
    # the reference up to the last block checked (the blocks after it are not run)
    cache = ref.cache()
    x = ref.embed(ids)
    for i, kind in enumerate(ref.cfg.kinds[:ups[-1]]):
        h = rms_norm(x, ref._vec(BB + "layers.%d.norm.weight" % i), eps)
        x = x + (ref.mamba(i, h, cache) if kind == "mamba" else
                 ref.attention(i, h, cache) if kind == "attention" else ref.moe(i, h))
        ref.hook(i, x)
    print("reference: %d blocks in %.0f s" % (ups[-1], time.time() - t0), flush=True)
    kq = NemotronKQ(P, weightkeeper=False)
    nf = ref._vec(BB + "norm_f.weight")
    ok = True
    for n in ups:
        t1 = time.time()
        prog = compile_nemotron_step(kq, len(ids), upto=n)
        c = kq.cache(256)
        kq._run(prog, ids, c, 0)
        got = prog.names["xn"].copy()
        want = rms_norm(seen[n - 1], nf, eps)
        err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-9)
        cos = np.sum(got * want, -1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1) + 1e-30)
        print("upto %d (%s): max relative error %.2e, cosine min %.6f mean %.6f; %.0f s" % (
            n, ",".join(sorted(set(kq.cfg.kinds[:n]))), err, cos.min(), cos.mean(), time.time() - t1), flush=True)
        ok = ok and cos.min() > 0.999
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
