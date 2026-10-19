#!/usr/bin/env python3
"""The GPU's products of KQ_NV4 rows for 1-4 tokens (k_nv4_gemv, kernels/quant/nv4.cu: FP4-Local's MTP
head) against float64 sums of the rows as the CPU decodes them (cops.kq_rows): each output within 2e-6
of its row's rms (float32 sums in another order).

    python scripts/check_nv4.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.ir import program as P  # noqa: E402
from reclank.backends.cpu import cops  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402
from reclank.formats.fp4local_st import mtp_head_nvfp4  # noqa: E402

KQ_NV4 = 51


def main():
    rng = np.random.default_rng(7)
    mirror = Mirror()
    bad = 0
    for rows, cols in ((4099, 2560), (1000, 1024)):
        wf = (rng.standard_normal((rows, cols)) * 0.02).astype(np.float32)
        codes, scales, alpha = mtp_head_nvfp4(wf)
        w = np.empty(rows * cops.kq_nv4_row_bytes(cols), np.uint8)
        cops.kq_nv4_pack([codes], [scales], [alpha], rows, cols, w)
        deq = cops.kq_rows(np.ascontiguousarray(w).reshape(-1), KQ_NV4, cols, np.arange(rows)).astype(np.float64)
        for t in (1, 2, 3, 4):
            x = rng.standard_normal((t, cols)).astype(np.float32)
            out = np.zeros((t, rows), np.float32)
            prog = P.Program()
            prog.emit(P.KQ_LINEAR, None, None, None, x, w, KQ_NV4, rows, cols, t, out)
            prog.names["out"] = out
            g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
            g.run()
            g.download("out")
            g.close()
            ref = x.astype(np.float64) @ deq.T
            rms = np.sqrt((ref * ref).mean(-1, keepdims=True))
            err = float((np.abs(out - ref) / rms).max())
            ok = err < 2e-6
            bad += not ok
            print("%5d x %4d, t %d: the largest difference %.1e of the row's rms %s" % (
                rows, cols, t, err, "ok" if ok else "BAD"))
    print("PASS" if bad == 0 else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
