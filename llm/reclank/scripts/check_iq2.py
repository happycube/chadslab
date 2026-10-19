#!/usr/bin/env python3
"""Check the formats of Nemotron 3.5 Bolt on the CPU: IQ2_XXS (ggml type 16, the experts) and
KQ_NV4 (the Mamba projections, NVFP4), each tier in its own process (RECLANK_CPU), on shipped rows
and real activations (the latent of real tokens: fc1_latent_proj of their normed embeddings; the
shared expert and in_proj on the normed embeddings):

- cops.kq_rows (the C decoder) against reclank/formats/iq2.py (IQ2_XXS) and model.nvfp4 (the
  NVFP4 of the file): the same bits;
- cops.kq_linear against the float product of the decoded rows and x as the kernels see it (int8
  x and its scales): a relative error near the rounding of float32;
- the integer tiers (avx2, avx2_vnni, avx512, avx512_vnni) give the same bits as each other;
- the rows of KQ_IQ2R (reclank/formats/iq2.py to_rows, type 67) give the same bits as the blocks
  (kq_linear, and nm_linear2 on one token and on a group).

    python scripts/check_iq2.py MODEL_DIR [--tiers scalar,avx2,avx2_vnni,avx512,avx512_vnni]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
KQ_IQ2_XXS, KQ_NV4, KQ_IQ2R = 16, 51, 67
TEXT = "The lanes share the contexts in host memory: the context on a lane's device is that lane's."


def inputs(model):
    """(name, raw blocks [rows, cols / 256 * 66], x float32 [t, cols]) on real data."""
    from reclank.models.nemotron.model import BB, NemotronCPU, rms_norm
    from reclank.models.qwen.tokenizer import QwenTokenizer
    m = NemotronCPU(model, keep_dense=False)
    ids = QwenTokenizer(os.path.join(model, "tokenizer.json")).encode(TEXT)[:5]
    h = rms_norm(m.embed(ids), m._vec(BB + "layers.1.norm.weight"), m.cfg.eps)
    lat = h @ m._mat(BB + "layers.1.mixer.fc1_latent_proj").T
    out = []
    for e in (0, 57, 211):
        name = BB + "layers.1.mixer.experts.%d.up_proj.weight" % e
        out.append((name, m.ck.raw(name), lat))
    name = BB + "layers.1.mixer.shared_experts.up_proj.weight"
    out.append((name, m.ck.raw(name), h))
    p = BB + "layers.0.mixer.in_proj."
    out.append((p + "weight", (m.ck.raw(p + "weight")[:512], m.ck.raw(p + "weight_scale")[:512],
                               float(np.asarray(m.ck.f32(p + "weight_scale_2")).reshape(-1)[0])), h))
    return out


def run_tier(model, out):
    from reclank.backends.cpu import cops
    from reclank.formats import iq2
    from reclank.models.nemotron.model import nvfp4
    res = {"tier": np.array(cops.TIER)}
    for k, (name, b, x) in enumerate(inputs(model)):
        if isinstance(b, tuple):                    # NVFP4: (codes, scales, g) -> KQ_NV4 rows
            cw, cs, g = b
            rows, cols, type_ = cw.shape[0], cw.shape[1] * 2, KQ_NV4
            w = np.empty(rows * cops.kq_nv4_row_bytes(cols), np.uint8)
            cops.kq_nv4_pack([np.ascontiguousarray(cw)], [np.ascontiguousarray(cs)], [g], rows, cols, w)
            want = nvfp4(cw, cs, g)
        else:
            rows, cols, type_ = b.shape[0], b.shape[1] * 256, KQ_IQ2_XXS
            w = np.ascontiguousarray(b).reshape(rows, -1)
            want = iq2.dequant(b)
        x = np.ascontiguousarray(x, np.float32)
        t = x.shape[0]
        xq = np.empty((t, cols), np.int8)
        xs = np.empty((t, cols // 32), np.float32)
        xm = np.empty((t, cols // 16), np.float32)
        cops.kq_quant_x(x, xq, xs, xm)
        o = np.empty((t, rows), np.float32)
        cops.kq_linear(w, type_, rows, cols, xq, xs, xm, x, t, o)
        wf = cops.kq_rows(w, type_, cols, np.arange(rows))
        res["rows_same%d" % k] = np.array(np.array_equal(wf, want))
        res["nv4%d" % k] = np.array(type_ == KQ_NV4)
        xd = xq.astype(np.float64) * np.repeat(xs, 32, axis=1)
        res["ref%d" % k] = (xd @ wf.astype(np.float64).T).astype(np.float32)
        res["out%d" % k] = o
        if type_ == KQ_IQ2_XXS and cops.TIER != "scalar":
            r = iq2.to_rows(b)
            o2 = np.empty((t, rows), np.float32)
            cops.kq_linear(r, KQ_IQ2R, rows, cols, xq, xs, xm, x, t, o2)
            same = np.array_equal(o2, o) and np.array_equal(cops.kq_rows(r, KQ_IQ2R, cols, np.arange(rows)), wf)
            for tt in (1, t):
                same &= np.array_equal(cops.nm_linear2(w, KQ_IQ2_XXS, rows, cols, x[:tt]),
                                       cops.nm_linear2(r, KQ_IQ2R, rows, cols, x[:tt]))
            res["rows_same%d" % k] = np.array(bool(res["rows_same%d" % k]) and same)
    np.savez(out, **res)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model")
    ap.add_argument("--tiers", default="scalar,avx2,avx2_vnni,avx512,avx512_vnni")
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.one:
        run_tier(args.model, args.one)
        return 0
    flags = open("/proc/cpuinfo").read()
    have = {"scalar": True, "avx2": " avx2 " in flags,
            "avx2_vnni": " avx_vnni " in flags or " avx512_vnni " in flags,
            "avx512": " avx512bw " in flags, "avx512_vnni": " avx512_vnni " in flags}
    got, ok = {}, True
    with tempfile.TemporaryDirectory() as d:
        for t in args.tiers.split(","):
            if not have.get(t):
                print("%-12s skipped (the CPU does not run it)" % t)
                continue
            env = dict(os.environ, RECLANK_CPU=t)
            if t == "avx2_vnni" and " avx_vnni " not in flags:
                env["RECLANK_CPU_TEST_EVEX"] = "1"
            p = os.path.join(d, t + ".npz")
            r = subprocess.run([sys.executable, __file__, args.model, "--one", p], env=env,
                               capture_output=True, text=True)
            if r.returncode:
                print("%-12s FAIL to run:\n%s" % (t, r.stderr[-2000:]))
                ok = False
                continue
            g = got[t] = dict(np.load(p))
            n = sum(1 for k in g if k.startswith("out"))
            errs, same = [], all(bool(g["rows_same%d" % k]) for k in range(n))
            for k in range(n):
                e = float(np.abs(g["out%d" % k] - g["ref%d" % k]).max() / np.abs(g["ref%d" % k]).max())
                errs.append("%.1e" % e)
                ok &= e < 2e-5
            ok &= same
            print("%-12s rows %s; against the reference: %s" % (t, "the same bits" if same else "DIFFERENT",
                                                               ", ".join(errs)))
    # the same bits: IQ2_XXS on every integer tier (one kernel, iq2.c); NVFP4 on the tiers of
    # dot2/avx2.c (avx512_vnni has its own kernel, dots.c kq_dot_nv4: close, not the same bits)
    ints = [t for t in ("avx2", "avx2_vnni", "avx512", "avx512_vnni") if t in got]
    for t in ints[1:]:
        keys = [k for k in got[t] if k.startswith("out") and (t != "avx512_vnni" or not got[t]["nv4" + k[3:]])]
        s = all(np.array_equal(got[ints[0]][k], got[t][k]) for k in keys)
        print("%s and %s: %s" % (t, ints[0], "the same bits" if s else "DIFFERENT"))
        ok &= s
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
