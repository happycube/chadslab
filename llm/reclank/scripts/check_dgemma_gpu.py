#!/usr/bin/env python3
"""DiffusionGemma's GPU records (models/dgemma/gpu.py) against their NumPy cases (Program.run_py), on
the model's real data: DG_KV_PUT and DG_ATTN (rows of the BF12 token table as q, k and v; 256
queries, 16 heads over 8 KV heads of 256 and over 2 of 512, a cache of 1756 rows, the decoder's
ranges and the encoder's causal ones), DG_MOE (layer 7's experts, the router's choice on rows of
the table; also against the CPU's KQ_MOE), and with --logits (an .npz with a (256, vocab) array
"l1": transformers' canvas logits) DG_SAMPLE.

    python scripts/check_dgemma_gpu.py MODELS/diffusiongemma-26B-A4B-it-BF12-GGUF [--logits hf_short.npz]

Prints PASS or FAIL (exit 1)."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.backends.cuda.runtime import GPUProgram, Mirror  # noqa: E402
from reclank.core import ops  # noqa: E402
from reclank.encoders.runtime import _EncCompiler  # noqa: E402
from reclank.formats.gguf import GGUF  # noqa: E402
from reclank.ir import program as P  # noqa: E402
from reclank.models.dgemma.gpu import _gguf_file  # noqa: E402

rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731


def run_both(prog, names, outs):
    """The program in NumPy (run_py), then on the GPU from the same inputs: (numpy, gpu) outs."""
    base = {k: prog.names[k].copy() for k in prog.names}
    prog.run_py()
    ref = {k: prog.names[k].copy() for k in outs}
    for k in base:
        prog.names[k][...] = base[k]
    gp = GPUProgram(prog, graph=False, mirror=Mirror())
    for k in names:
        gp.upload(k)
    gp.run()
    for k in outs:
        gp.download(k)
    got = {k: prog.names[k].copy() for k in outs}
    gp.close()
    return ref, got


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--logits")
    a = ap.parse_args()
    g = GGUF(_gguf_file(a.model)[0])
    rng = np.random.default_rng(0)
    tab = g.dequant("token_embd.weight", rows=rng.integers(0, 262144, 6000))
    ok = True

    def rows(n, width):
        k = -(-width // tab.shape[1])
        x = np.concatenate([tab[rng.integers(0, len(tab), n)] for _ in range(k)], 1)[:, :width]
        return np.ascontiguousarray(x * np.float32(4.0), dtype=np.float32)

    for kvh, hd in ((8, 256), (2, 512)):
        for causal in (False, True):
            heads, t, n = 16, 256, 1500
            c = _EncCompiler(1e-6)
            c.env.update(q=rows(t, heads * hd), k=rows(t, kvh * hd), v=rows(t, kvh * hd),
                         K=rows(n + t, kvh * hd), V=rows(n + t, kvh * hd), out=np.zeros((t, heads * hd), np.float32),
                         lo=np.zeros(t, np.int32), hi=np.zeros(t, np.int32), row=np.array([n], np.int32))
            e = c.env
            P.k_dg_kv_put(c, e["k"], e["v"], e["K"], e["V"], e["row"])
            P.k_dg_attn(c, e["q"], e["K"], e["V"], e["lo"], e["hi"], heads, kvh, hd, out=e["out"])
            prog = c.p.finish()
            i = np.arange(t)
            win = kvh == 8
            if causal:
                prog.names["lo"][:] = np.maximum(0, n + i - 1023) if win else 0
                prog.names["hi"][:] = n + i + 1
            else:
                prog.names["lo"][:] = max(0, n - 1023) if win else 0
                prog.names["hi"][:] = n + t
            ref, got = run_both(prog, ("q", "k", "v", "K", "V", "lo", "hi", "row"), ("out", "K", "V"))
            r = rel(got["out"], ref["out"])
            same = np.array_equal(got["K"], ref["K"]) and np.array_equal(got["V"], ref["V"])
            good = r < 1e-5 and same
            ok &= good
            print("DG_ATTN %d KV heads of %d, %s ranges: %.2e; DG_KV_PUT rows the same: %s%s" % (
                kvh, hd, "causal" if causal else "canvas", r, same, "" if good else "  FAIL"))

    # DG_MOE: layer 7, the router on rows of the table (normed as a layer's input)
    H, E, K, I, t = 2816, 128, 8, 704, 256
    p = "blk.7."
    gu, dn = g.raw(p + "ffn_gate_up_exps.weight")[0], g.raw(p + "ffn_down_exps.weight")[0]
    x = np.ascontiguousarray(ops.rms_norm(tab[rng.integers(0, len(tab), t)] * np.float32(H ** 0.5), None, 1e-6),
                             np.float32)
    proj, sc = g.dequant(p + "ffn_gate_inp.weight"), g.dequant(p + "ffn_gate_inp.scale").reshape(-1)
    lg = (x * sc * H ** -0.5) @ proj.T
    pr = np.exp(lg - lg.max(-1, keepdims=True))
    pr /= pr.sum(-1, keepdims=True)
    idx = np.ascontiguousarray(np.argsort(-pr, -1)[:, :K], np.int32)
    val = np.take_along_axis(pr, idx, -1).astype(np.float32)
    val /= val.sum(-1, keepdims=True)
    c = _EncCompiler(1e-6)
    c.env.update(h=x.copy(), idx=idx, val=val, out=np.zeros((t, H), np.float32))
    P.k_dg_moe(c, c.env["h"], c.env["idx"], c.env["val"], gu, dn, E, I, out=c.env["out"])
    ref, got = run_both(c.p.finish(), ("h", "idx", "val"), ("out",))
    cpu = np.zeros((t, H), np.float32)
    cops.kq_moe_act(np.zeros((t, H), np.int8), np.zeros((t, H // 32), np.float32),
                    np.zeros((t, H // 16), np.float32), idx, val, E,
                    cops.kq_moe_mats((gu, 57), None, (dn, 57), None), None, H, I,
                    cops.kq_moe_scratch(t, K, E, H, I), cpu, gelu=True, hf=x, flags=128)
    r1, r2 = rel(got["out"], ref["out"]), rel(got["out"], cpu)
    good = r1 < 1e-5 and r2 < 1e-5
    ok &= good
    print("DG_MOE (%d experts used): %.2e against NumPy, %.2e against the CPU's KQ_MOE%s" % (
        len(np.unique(idx)), r1, r2, "" if good else "  FAIL"))
    # the split: the even experts packed as the GPU's slots, the others the CPU's (cidx); the two
    # parts' sum against the whole
    hot = np.arange(0, E, 2)
    gmap = np.full(E, -1, np.int32)
    gmap[hot] = np.arange(len(hot), dtype=np.int32)
    gu_h = np.ascontiguousarray(gu.reshape(E, -1)[hot]).reshape(-1)
    dn_h = np.ascontiguousarray(dn.reshape(E, -1)[hot]).reshape(-1)
    c = _EncCompiler(1e-6)
    c.env.update(h=x.copy(), idx=idx, val=val, out=np.zeros((t, H), np.float32), map=gmap,
                 cidx=np.zeros((t, K), np.int32))
    P.k_dg_moe(c, c.env["h"], c.env["idx"], c.env["val"], gu_h, dn_h, len(hot), I, out=c.env["out"],
               map=c.env["map"], cidx=c.env["cidx"])
    ref, got = run_both(c.p.finish(), ("h", "idx", "val", "map"), ("out", "cidx"))
    cold = np.zeros((t, H), np.float32)
    cops.kq_moe_act(np.zeros((t, H), np.int8), np.zeros((t, H // 32), np.float32),
                    np.zeros((t, H // 16), np.float32), got["cidx"], val, E,
                    cops.kq_moe_mats((gu, 57), None, (dn, 57), None), None, H, I,
                    cops.kq_moe_scratch(t, K, E, H, I), cold, gelu=True, hf=x, flags=128)
    want_cidx = np.where(gmap[idx] >= 0, -1, idx)
    r1, r2 = rel(got["out"], ref["out"]), rel(got["out"] + cold, cpu)
    same = np.array_equal(got["cidx"], want_cidx)
    good = r1 < 1e-5 and r2 < 1e-5 and same
    ok &= good
    print("DG_MOE split (%d of %d experts on the GPU): %.2e against NumPy; with the CPU's part %.2e against "
          "the whole; the CPU's ids right: %s%s" % (len(hot), E, r1, r2, same, "" if good else "  FAIL"))

    if a.logits:
        z0 = np.ascontiguousarray(np.load(a.logits)["l1"], np.float32)
        t, vocab = z0.shape
        # the raw logits' log-probabilities in float64 (z0 itself is the program's z: overwritten)
        zl = z0.astype(np.float64)
        zl -= zl.max(-1, keepdims=True)
        zl -= np.log(np.exp(zl).sum(-1, keepdims=True))
        c = _EncCompiler(1e-6)
        c.env.update(z=z0, probs=np.zeros((t, vocab), np.float32), temp=np.array([0.8], np.float32),
                     ban=(np.arange(t) < 100).astype(np.int32), eos=np.array([1, 106, 50], np.int32),
                     u=rng.random(t).astype(np.float32), ent=np.zeros(t, np.float32),
                     am=np.zeros(t, np.int32), draw=np.zeros(t, np.int32), pmax=np.zeros(t, np.float32),
                     praw=np.zeros(t, np.float32))
        e = c.env
        P.k_dg_sample(c, e["z"], e["probs"], e["temp"], e["ban"], e["eos"], e["u"], e["ent"], e["am"], e["draw"],
                      pmax=e["pmax"], praw=e["praw"])
        outs = ("z", "probs", "ent", "am", "draw", "pmax", "praw")
        ref, got = run_both(c.p.finish(), tuple(c.env), outs)
        want_raw = np.exp(zl[np.arange(t), got["am"]])
        fin = np.isfinite(ref["z"])
        zs = np.array_equal(np.where(fin, got["z"], 0), np.where(fin, ref["z"], 0))
        good = (zs and rel(got["probs"], ref["probs"]) < 1e-5 and np.array_equal(got["am"], ref["am"])
                and (got["draw"] == ref["draw"]).mean() > 0.98 and np.abs(got["ent"] - ref["ent"]).max() < 1e-4
                and not np.isin(got["draw"][:100], [1, 106, 50]).any()
                and rel(got["pmax"], ref["pmax"]) < 1e-5
                and np.allclose(got["pmax"], got["probs"][np.arange(t), got["am"]], rtol=1e-5)
                and rel(got["praw"], ref["praw"]) < 1e-5 and rel(got["praw"], want_raw) < 1e-5)
        ok &= good
        print("DG_SAMPLE: z the same %s; probs %.2e; entropy %.1e; the argmax's probability %.1e, raw %.1e "
              "(%.1e against float64); argmax %d/%d; samples %d/%d%s" % (
            zs, rel(got["probs"], ref["probs"]), np.abs(got["ent"] - ref["ent"]).max(), rel(got["pmax"], ref["pmax"]),
            rel(got["praw"], ref["praw"]), rel(got["praw"], want_raw),
            (got["am"] == ref["am"]).sum(), t, (got["draw"] == ref["draw"]).sum(), t, "" if good else "  FAIL"))
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
