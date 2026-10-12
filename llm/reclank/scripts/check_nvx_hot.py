#!/usr/bin/env python3
"""The experts of a decode step and of a small group: the hot experts on the
GPU (GP_KQ_HOT_MOE) against the cold experts on the CPU (GP_KQ_MOE, the small
body of a step or a verify group), bit for bit. KQ_NVX (NVFP4) by default;
--q6 the types of the MIX-BF12 file (gate and up Q6_K, down Q8_0).

Experts of the shapes of Qwen3.8 (hidden 2560, inner 640) with random NVFP4
codes and scales (cops.kq_nvx_pack); one routed expert a token with the
weight 1.0, so the output of the CPU is the down product of that expert
(the GPU's de). The GPU of a token alone and in a group of 3 must agree
too (an MTP verify group). x as int16 (two int8 planes) by default; with
RECLANK_MOE_X16=0 RECLANK_GPU_KQH_X=i8, int8.

    python scripts/check_nvx_hot.py [--q6] [--split] [--rot]

--split: the output of 4 experts a token with GP_MOE_COMBINE when the GPU
holds all of them and when it holds half (the others the rows of the CPU,
KQ_MOE act bit 64): the same bits whichever experts the GPU holds. --rot:
the act rotated in each 32 values before its down product (the RQ8_0 /
RQ6_K experts of the MIX-BF12 file: tq6_rot32 on the CPU, tq6_rot_warp on
the GPU). --real: the NVFP4 experts of layer 5 of the file NVX_GGUF (512
experts, 10 a token) in place of random ones (with --split).
    RECLANK_MOE_X16=0 RECLANK_GPU_KQH_X=i8 python scripts/check_nvx_hot.py
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from reclank.backends.cpu import cops  # noqa: E402
from reclank.ir import program as P  # noqa: E402

KQ_NVX, KQ_BF12 = 53, 57
GT, DT = KQ_NVX, KQ_NVX       # the types of the experts (main: --q6)


def nvx_experts(rng, E, rows, cols):
    """E random NVFP4 matrices of rows x cols as KQ_NVX (one array)."""
    rb = cops.kq_nvx_row_bytes(cols)
    out = np.zeros(E * rows * rb, np.uint8)
    ws = [rng.integers(0, 256, (rows, cols // 2), dtype=np.uint8) for _ in range(E)]
    # E4M3 scales of about 0.25 to 4 (exponent codes 5 to 9), no NaN
    ss = [(rng.integers(5, 10, (rows, cols // 16)) << 3 | rng.integers(0, 8, (rows, cols // 16)))
          .astype(np.uint8) for _ in range(E)]
    gs = rng.uniform(0.002, 0.004, E).astype(np.float32)
    cops.kq_nvx_pack(ws, ss, gs, rows, cols, out)
    return out


def q_experts(rng, E, rows, cols, type_):
    """E random matrices of rows x cols as Q6_K (14) or Q8_0 (8)."""
    f = cops.kq_to_q6_k if type_ == 14 else cops.kq_to_q8_0
    return np.concatenate([f((rng.standard_normal((rows, cols)) * 0.05).astype(np.float32), cols).reshape(-1)
                           for _ in range(E)])


def run_gpu(gate, up, down, shared, x, ids, E, hidden, inner, mirror):
    """GP_KQ_HOT_MOE with every expert hot (slot e = expert e): the de of
    each routed pair (t x hidden)."""
    from reclank.backends.cuda.runtime import GPUProgram
    t, k = ids.shape
    pairs = t * k + t
    prog = P.Program()
    de = np.zeros((pairs, hidden), np.float32)
    prog.emit(P.KQ_HOT_MOE, x, np.ones((t, k), np.float32), ids.astype(np.int32),
              np.arange(E, dtype=np.int32), gate, up, down,
              np.zeros((pairs, 2 * inner), np.float32), np.zeros((pairs, inner), np.float32), de,
              np.zeros((t, hidden), np.float32), k, inner, hidden, GT, DT, t,
              shared[0], shared[1], shared[2], KQ_BF12, np.zeros(t, np.float32))
    prog.names["de"] = de
    prog = prog.finish()
    g = GPUProgram(prog, graph=False, mirror=mirror)
    g.run()
    g.download("de")
    g.close()
    return de[:t * k].reshape(t, k, hidden)[:, 0].copy()


def run_cpu(gate, up, down, x, ids, E, hidden, inner):
    """GP_KQ_MOE (the small body, the flags of a GPU step) with every expert
    cold: the output (t x hidden)."""
    t, k = ids.shape
    out = np.zeros((t, hidden), np.float32)
    if t == 1:
        cold = np.zeros(2 * k + 1, np.int32)
        cold[:k] = ids[0]
        cold[k] = k
        kcount = cold[k:]
    else:
        cold, kcount = ids.astype(np.int32), None
    mats = cops.kq_moe_mats((gate, GT), (up, GT), (down, DT), None)
    q = P.Program()
    q.emit(P.KQ_MOE, np.zeros((t, hidden), np.int8), np.zeros((t, hidden // 32), np.float32),
           np.zeros((t, hidden // 16), np.float32), cold, np.ones((t, k), np.float32), t, k, E, mats,
           None, hidden, inner, cops.kq_moe_scratch(t, k, E, hidden, inner), out, kcount,
           8 | (32 if os.environ.get("RECLANK_MOE_X16", "1") == "1" else 0), x, None, None, None)
    q.keep.append(mats)
    q = q.finish()
    assert cops.gp_run_task(q.buf) == 0
    return out


def run_split(gate, up, down, shared, x, ids, val, E, hidden, inner, hot, mirror):
    """The output of the experts ids (t x k, weights val) of x: GP_KQ_HOT_MOE
    with the experts of hot on the GPU, the others the rows of GP_KQ_MOE
    (act bit 64), then GP_MOE_COMBINE."""
    from reclank.backends.cuda.runtime import GPUProgram
    t, k = ids.shape
    pairs = t * k + t
    slots = np.array([e if e in hot else -1 for e in range(E)], np.int32)
    coldm = np.array([[e if e not in hot else -1 for e in r] for r in ids], np.int32)
    rows = np.zeros((t * k, hidden), np.float32)
    if (coldm >= 0).any():
        if t == 1:
            cl = [e for e in ids[0] if e not in hot]
            cold = np.zeros(2 * k + 1, np.int32)
            cold[:len(cl)] = cl
            cold[k] = len(cl)
            kc, cids = cold[k:], cold
            cval = np.array([[v for e, v in zip(ids[0], val[0]) if e not in hot] + [0.] * (k - len(cl))],
                            np.float32)
        else:
            kc, cids, cval = None, coldm, val
        mats = cops.kq_moe_mats((gate, GT), (up, GT), (down, DT), None)
        q = P.Program()
        q.emit(P.KQ_MOE, np.zeros((t, hidden), np.int8), np.zeros((t, hidden // 32), np.float32),
               np.zeros((t, hidden // 16), np.float32), cids, cval, t, k, E, mats, None, hidden, inner,
               cops.kq_moe_scratch(t, k, E, hidden, inner), rows, kc,
               8 | (32 if os.environ.get("RECLANK_MOE_X16", "1") == "1" else 0) | 64, x, None, None, None)
        q.keep.append(mats)
        assert cops.gp_run_task(q.finish().buf) == 0
    prog = P.Program()
    de = np.zeros((pairs, hidden), np.float32)
    out = np.zeros((t, hidden), np.float32)
    slog = np.zeros(t, np.float32)
    prog.emit(P.KQ_HOT_MOE, x, val, ids.astype(np.int32), slots, gate, up, down,
              np.zeros((pairs, 2 * inner), np.float32), np.zeros((pairs, inner), np.float32), de,
              None, k, inner, hidden, GT, DT, t, shared[0], shared[1], shared[2], KQ_BF12, slog)
    prog.emit(P.MOE_COMBINE, val, ids.astype(np.int32), slots, None, de, rows, out, k, hidden, t, slog,
              None if t == 1 else coldm)
    prog.names["out"] = out
    g = GPUProgram(prog.finish(), graph=False, mirror=mirror)
    g.run()
    g.download("out")
    g.close()
    return out.copy()


def main():
    from reclank.backends.cuda.runtime import Mirror
    rng = np.random.default_rng(3)
    E, hidden, inner = 8, 2560, 640
    global GT, DT
    if "--q6" in sys.argv:
        GT, DT = 14, 8
        gate, up, down = (q_experts(rng, E, inner, hidden, 14), q_experts(rng, E, inner, hidden, 14),
                          q_experts(rng, E, hidden, inner, 8))
    else:
        gate = nvx_experts(rng, E, inner, hidden)
        up = nvx_experts(rng, E, inner, hidden)
        down = nvx_experts(rng, E, hidden, inner)
    zero = lambda r, c: cops.kq_bf16_to_bf12(np.zeros((r, c), np.uint16), c)[0].reshape(-1)  # noqa: E731
    shared = (zero(inner, hidden), zero(inner, hidden), zero(hidden, inner))
    mirror = Mirror()
    ok = True
    if "--rot" in sys.argv:
        from reclank.backends.cuda.runtime import lib
        cops._lib.rc_kq_set_moe_rot(1)
        lib().rc_gpu_set_moe_rot(1)
    if "--split" in sys.argv:
        kk, hots = 4, ({0, 1, 2, 3}, {1, 3, 5, 7}, set())
        if "--real" in sys.argv:
            from reclank.formats.gguf import GGUF
            g = GGUF(os.environ.get("NVX_GGUF", "/mnt/pmem/Qwen3.8-Flash-Next-MIX-BF12-GGUF/"
                                    "Qwen3.8-Flash-Next-nvfp4-experts.gguf"), stage=False)
            gate, up, down = (np.array(g.raw("blk.5.ffn_%s_exps.weight" % n)[0])
                              for n in ("gate", "up", "down"))
            E, kk = 512, 10
            hots = (set(rng.choice(E, E // 2, replace=False).tolist()), set(range(0, E, 3)), set())
        for t in (1, 3):
            x = (rng.standard_normal((t, hidden)) * 0.7).astype(np.float32)
            # (--shared: the tokens choose among the same few experts, as a verify group does)
            pool = rng.choice(E, kk + 5, replace=False) if "--shared" in sys.argv else np.arange(E)
            ids = np.stack([rng.choice(pool, kk, replace=False) for _ in range(t)]).astype(np.int32)
            val = rng.uniform(0.05, 0.4, (t, kk)).astype(np.float32)
            a = run_split(gate, up, down, shared, x, ids, val, E, hidden, inner, set(range(E)), mirror)
            for hot in hots:
                b = run_split(gate, up, down, shared, x, ids, val, E, hidden, inner, hot, mirror)
                same = np.array_equal(a.view(np.uint32), b.view(np.uint32))
                ok &= same
                print("t %d, the GPU holds %-14s %s the output of all hot (max |diff| %.3g)" % (
                    t, (str(sorted(hot)) if len(hot) <= 8 else "%d experts" % len(hot)) + ":",
                    "the same bits as" if same else "DIFFERS from",
                    np.abs(a - b).max()))
        print("check_nvx_hot --split: %s" % ("OK" if ok else "FAILED"))
        return 0 if ok else 1
    x3 = (rng.standard_normal((3, hidden)) * 0.7).astype(np.float32)
    ids3 = np.array([[2], [5], [2]], np.int32)
    gpu3 = run_gpu(gate, up, down, shared, x3, ids3, E, hidden, inner, mirror)
    cpu3 = run_cpu(gate, up, down, x3, ids3, E, hidden, inner)
    for j in range(3):
        g1 = run_gpu(gate, up, down, shared, x3[j:j + 1], ids3[j:j + 1], E, hidden, inner, mirror)[0]
        c1 = run_cpu(gate, up, down, x3[j:j + 1], ids3[j:j + 1], E, hidden, inner)[0]
        ref = np.abs(c1).max()
        for name, a, b in (("GPU step / CPU step", g1, c1), ("GPU group / CPU group", gpu3[j], cpu3[j]),
                           ("GPU step / GPU group", g1, gpu3[j]), ("CPU step / CPU group", c1, cpu3[j])):
            same = np.array_equal(a.view(np.uint32), b.view(np.uint32))
            ok &= same
            print("token %d, %-22s %s (max |diff| %.3g of %.3g)" % (
                j, name + ":", "the same bits" if same else "DIFFER", np.abs(a - b).max(), ref))
    print("check_nvx_hot: %s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
