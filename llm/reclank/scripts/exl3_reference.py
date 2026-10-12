#!/usr/bin/env python3
"""Reference tensors of ExLlamaV3 for scripts/check_exl3.py (EXL3_PLAN.md,
phase 0). Run it with the Python of a virtual environment with ExLlamaV3
(torch, CUDA); it does not import reclank.

For each codebook (3inst, mcg, mul1) and each K: a matrix of random tiles
(random trellis words; suh, svh of the sizes of a real file) as a
LinearEXL3 of ExLlamaV3, and its tensors: the float16 weights in the
rotated basis (get_inner_weight_tensor: the reconstruct kernel), the
weights (get_weight_tensor), and the products for 1, 4, and 300 rows of x
(forward: the GEMV kernels, and for 300 rows the reconstruct and the GEMM).

    VENV/bin/python scripts/exl3_reference.py OUT.npz
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from exllamav3.modules.quant.exl3 import LinearEXL3


def main():
    out = sys.argv[1]
    rng = np.random.default_rng(7)
    dev = torch.device("cuda:0")
    res = {}
    for cb in ("3inst", "mcg", "mul1"):
        for K in range(1, 9):
            k, n = 512, 384
            tr = rng.integers(0, 1 << 16, size=(k // 16, n // 16, 16 * K), dtype=np.uint32)
            tr = tr.astype(np.uint16).view(np.int16)
            suh = (rng.choice([-1.0, 1.0], k) * rng.uniform(0.5, 1.5, k) / 1.24371088).astype(np.float16)
            svh = (rng.uniform(0.5, 1.5, n) * 0.01).astype(np.float16)
            marker = torch.tensor(0, dtype=torch.int32, device=dev)
            lin = LinearEXL3(None, k, n, suh=torch.from_numpy(suh).to(dev), svh=torch.from_numpy(svh).to(dev),
                             trellis=torch.from_numpy(tr).to(dev),
                             mcg=marker if cb == "mcg" else None, mul1=marker if cb == "mul1" else None,
                             key="ref.%s.%d" % (cb, K))
            p = "%s_%d_" % (cb, K)
            res[p + "trellis"] = tr
            res[p + "suh"] = suh
            res[p + "svh"] = svh
            res[p + "inner"] = lin.get_inner_weight_tensor().float().cpu().numpy()
            res[p + "weight"] = lin.get_weight_tensor().float().cpu().numpy()
            for rows in (1, 4, 300):
                x = rng.standard_normal((rows, k)).astype(np.float16)
                y = lin.forward(torch.from_numpy(x).to(dev), {}, out_dtype=torch.float)
                res[p + "x%d" % rows] = x
                res[p + "y%d" % rows] = y.float().cpu().numpy()
            print(cb, K, "done", flush=True)
    np.savez(out, **res)


if __name__ == "__main__":
    main()
