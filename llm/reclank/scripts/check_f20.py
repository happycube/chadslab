"""F20 (formats.gguf F20; encoders.runtime.to_f20) on real float32 matrices: the C encoder against
the format's rule in NumPy (the bits of every value), KQ_F20X16 against the rows, and ENC_LINEAR of
F20 W against ENC_LINEAR of the decoded float32 W (the same bits: the CPU's KQ_F20X16 and
KQ_F32X16, the GPU's k_enc_sgemm and k_enc_gemm, the Python stepper).

    check_f20.py UNET_DIR REF.npz [--gpu]

The GPU's k_enc_gemm (the float32 path's fallback) with RECLANK_GPU_ENC_SGEMM=0.

UNET_DIR: a float32 safetensors directory (SD's unet); REF.npz: real activations for x (sd_reference.py's
dump: the text embeddings and latents)."""
import argparse
import glob
import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.encoders.runtime import _EncCompiler, _Plain, to_f20
from reclank.formats.safetensors import SafeTensors


def f20_spec(w):
    """The float32 values of F20 by the format's rule, in NumPy."""
    u = np.ascontiguousarray(w, np.float32).view(np.uint32).astype(np.uint64)
    u = ((u + 0x7F + ((u >> 8) & 1)) & 0xFFFFFF00).astype(np.uint32)
    g = u.reshape(-1, 32)
    e = ((g >> 23) & 255).astype(np.int64)
    gap = e.max(-1, keepdims=True) - e
    sgn, m = g >> 31, (g >> 8) & 0x7FFF
    zero = (gap > 15) | ((gap == 15) & (sgn == 1) & (m == 0))
    return np.where(zero, 0, g).astype(np.uint32).reshape(w.shape).view(np.float32)


def program(lin, n, k, gpu):
    c = _EncCompiler(1e-5, pack=not gpu)
    m = lin.f20.rows if getattr(lin, "f20", None) is not None else lin.w.shape[0]
    c.env.update(x=np.zeros((n, k), np.float32), y=np.zeros((n, m), np.float32))
    c.compile(("seq", ("set", "y", ("enc_linear", lin, "x"))))
    p = c.p.finish()
    if not gpu:
        p.threads = p.buf[3] = 12
    return p


def run(lin, x, gpu, py=False):
    p = program(lin, x.shape[0], x.shape[1], gpu)
    p.names["x"][...] = x
    if py:
        p.run_py()
    elif gpu:
        from reclank.backends.cuda.runtime import GPUProgram
        g = GPUProgram(p)
        g.upload("x")
        g.run()
        g.download("y")
    else:
        p.run()
    return p.names["y"].copy()


ap = argparse.ArgumentParser()
ap.add_argument("unet")
ap.add_argument("ref")
ap.add_argument("--gpu", action="store_true")
a = ap.parse_args()
W = SafeTensors(sorted(glob.glob(os.path.join(a.unet, "*.safetensors")))[0])
z = np.load(a.ref)
acts = np.concatenate([z[k].astype(np.float32).ravel() for k in ("emb", "uemb", "x0", "e0", "x5", "e5")])
names = ["down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
         "down_blocks.1.resnets.0.conv1.weight",
         "mid_block.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
         "up_blocks.1.resnets.0.conv1.weight",
         "down_blocks.0.attentions.0.transformer_blocks.0.attn2.to_k.weight"]
bad = 0
for name in names:
    w = np.ascontiguousarray(np.asarray(W.get(name), np.float32).reshape(W.get(name).shape[0], -1))
    for rows in (slice(None), slice(0, 40)):
        wr = np.ascontiguousarray(w[rows])
        m, k = wr.shape
        data, rep = cops.kq_f32_to_f20(wr, k)
        dec = cops.kq_f20_to_f32(data, m, k)
        spec = f20_spec(wr)
        x16 = cops.kq_f20_to_f32(cops.kq_pack_f20x16(data, m, k), m, k, x16=True)
        same = np.array_equal(dec.view(np.uint32), spec.view(np.uint32))
        same16 = np.array_equal(x16.view(np.uint32), dec.view(np.uint32))
        rel = np.linalg.norm((dec - wr).astype(np.float64)) / np.linalg.norm(wr.astype(np.float64))
        print("%s %d x %d: zeroed %d, collisions %d; the rule's bits %s, x16 %s; norm error %.2e" % (
            name.split(".weight")[0], m, k, rep["zeroed"], rep["collisions"], "same" if same else "DIFFER",
            "same" if same16 else "DIFFER", rel))
        bad += (not same) + (not same16)
        b = np.ascontiguousarray(np.asarray(W.get(name.replace(".weight", ".bias")), np.float32)[rows]) \
            if name.replace(".weight", ".bias") in W.header else None
        for n in (1, 3, 77, 300):
            x = np.resize(acts, n * k).reshape(n, k).astype(np.float32)
            for dev in (("cpu", False),) + ((("gpu", True),) if a.gpu else ()):
                ref = _Plain(dec.copy())
                ref.b = b
                f = _Plain(wr.copy())
                f.b = b
                assert to_f20(f, dev[1])
                y0, y1 = run(ref, x, dev[1]), run(f, x, dev[1])
                ok = np.array_equal(y0.view(np.uint32), y1.view(np.uint32))
                line = "  n %3d %s: F20 against float32 of its values %s" % (n, dev[0], "the same bits" if ok else
                                                                         "DIFFER %.2e" % np.abs(y0 - y1).max())
                bad += not ok
                if not dev[1] and n == 3:
                    yp = run(f, x, False, py=True)
                    err = np.abs(yp - y0).max() / max(np.abs(y0).max(), 1e-30)
                    line += "; the Python stepper %.1e" % err
                    bad += err > 1e-5
                print(line)
print("FAIL %d" % bad if bad else "PASS")
