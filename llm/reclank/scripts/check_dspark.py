#!/usr/bin/env python3
"""Check the DSpark drafter of Qwen3.8-27B on the GPU (reclank/models/qwen/dspark.py) beside the target's
QwenGPU (a GGUF or an EXL3 directory), on real text:

1. ref: a prompt in groups of 256 (the target's taps of each group captured), the context of the NumPy reference
   (DSpark, float32) from those taps against the GPU's rings, then a draft at the prompt's end: the hidden
   states of the block, the drafts and their confidences against the reference's (the reference's logits: the
   target's head dequantized, float32 x).
2. decode: greedy replies to real prompts with the drafter (speculative.stream) and plain: the tokens must be
   the same; the drafts kept, the tokens a round and the tokens a second, for each stop (RECLANK_DSPARK_STOP).

    python scripts/check_dspark.py TARGET DRAFTER.gguf [--parts ref,decode] [--tokens 256] [--stops 0.3,0.5]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PROMPTS = {
    "btree": "Explain how a B-tree keeps itself balanced on insertion, then write a short Python function that "
             "inserts a key into a B-tree node with splitting.",
    "story": "Write a short story about a village by a river that floods, told by the miller's daughter who "
             "saves the ledgers.",
    "summary": "Summarize this document in a few paragraphs for a new user:\n\n" +
               open(os.path.join(R, "docs", "FLASHVSR.md")).read(),
}


def rel(a, b):
    return float(np.abs(np.asarray(a, np.float64) - b).max() / (np.abs(b).max() + 1e-30))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("target")
    ap.add_argument("drafter")
    ap.add_argument("--parts", default="ref,decode")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--stops", default="0.3")
    ap.add_argument("--ctx", type=int, default=8192)
    a = ap.parse_args()
    parts = a.parts.split(",")
    from reclank.backends.cuda.runtime import _check, lib
    from reclank.core import speculative as spec
    from reclank.models.qwen.dflash2 import TMAX, DFlash2Drafter
    from reclank.models.qwen.dspark import DSpark, DSparkGPU
    from reclank.models.qwen.gpu35 import QwenGPU
    from reclank.models.qwen.qwen35 import QwenCache, QwenEXL3Program, QwenGGUFProgram
    from reclank.models.qwen.tokenizer import QwenTokenizer
    tdir = a.target if os.path.isdir(a.target) else os.path.dirname(os.path.abspath(a.target))
    tok = QwenTokenizer(os.path.join(tdir, "tokenizer.json"))
    m = QwenEXL3Program(a.target) if os.path.isdir(a.target) else QwenGGUFProgram(a.target)
    t0 = time.time()
    dr = DSparkGPU(a.drafter, m)
    if "ref" in parts:
        dr.capture = [np.zeros((TMAX, dr.hidden), np.float32) for _ in dr.taps]
    d = QwenGPU(m, ctx=a.ctx, mtp=False, dflash=dr)
    dr.bind(d)
    print("target and drafter loaded in %.0f s (taps %s, fc type %s)" % (time.time() - t0, dr.taps, dr.fc_parts()[0][1]),
          flush=True)
    bad = 0

    if "ref" in parts:
        ids = tok.encode(open(os.path.join(R, "docs", "QWEN_PLAN.md")).read())[:600]
        c = QwenCache(m.cfg, max_len=a.ctx)
        d.attach(c)
        rows = []
        for c0 in range(0, len(ids), 256):
            n = min(256, len(ids) - c0)
            d.group(ids[c0:c0 + n], c0, 256)
            caps = []
            for cap in dr.capture:
                _check(lib().rc_gpu_d2h(cap.ctypes.data, d.g.mirror.buffer_of(cap).ptr, n * dr.hidden * 4))
                caps.append(cap[:n].copy())
            rows.append(np.concatenate(caps, 1))
        nxt = int(np.argmax(np.asarray(d.logits()).reshape(-1)))
        rows = np.concatenate(rows)
        # the reference: its context from the same taps, its head the target's (dequantized)
        assert hasattr(m, "g"), "the reference's head: a GGUF target (--parts decode for an EXL3 one)"
        W = m.g.dequant(m.gname("lm_head"))
        ref = DSpark(a.drafter, embed=m.embed, head=lambda h: h @ W.T)
        rc = ref.cache()
        ref.inject(rc, rows, np.arange(len(ids)))
        for i in (0, ref.layers - 1):
            K = np.zeros_like(dr.ring_k[i])
            _check(lib().rc_gpu_d2h(K.ctypes.data, d.g.mirror.buffer_of(dr.ring_k[i]).ptr, K.nbytes))
            V = np.zeros_like(dr.ring_v[i])
            _check(lib().rc_gpu_d2h(V.ctypes.data, d.g.mirror.buffer_of(dr.ring_v[i]).ptr, V.nbytes))
            ek, ev = rel(K[:len(ids)], rc["K"][i][:len(ids)]), rel(V[:len(ids)], rc["V"][i][:len(ids)])
            ok = ek < 2e-2 and ev < 2e-2
            bad += not ok
            print("context layer %d: K rel err %.2e, V %.2e %s" % (i, ek, ev, "ok" if ok else "BAD"), flush=True)
        pos0 = len(ids)
        dg, cfg = dr.drafts_gpu(nxt, pos0)
        hg = dr.hidden_gpu()
        hr = ref.block_forward(rc, nxt, pos0)
        drf, cfr = ref.markov(hr, nxt)
        eh = rel(hg, hr)
        ec = float(np.abs(np.asarray(cfg, np.float64) - cfr).max())
        same = 0
        while same < len(dg) and dg[same] == drf[same]:
            same += 1
        ok = eh < 3e-2 and ec < 0.2 and same >= 4
        bad += not ok
        print("draft at %d after %r: hidden rel err %.2e; confidence logits GPU %s, reference %s (max diff %.3f); "
              "drafts the same for the first %d of %d: GPU %r, reference %r %s" % (
                  pos0, tok.decode([nxt]), eh, np.round(cfg, 2).tolist(), np.round(cfr, 2).tolist(), ec, same,
                  len(dg), tok.decode(dg), tok.decode(drf), "ok" if ok else "BAD"), flush=True)
        d.detach(c)
        del W

    if "decode" in parts:
        N = a.tokens

        def plain(ids):
            c = QwenCache(m.cfg, max_len=a.ctx)
            d.attach(c)
            d.prefill(ids)
            t = int(np.argmax(np.asarray(d.logits()).reshape(-1)))
            out, pos = [t], len(ids)
            t1 = time.time()
            for _ in range(N - 1):
                d.step(t, pos)
                t = int(np.argmax(np.asarray(d.logits()).reshape(-1)))
                out.append(t)
                pos += 1
            dt = time.time() - t1
            d.detach(c)
            return out, (N - 1) / dt

        def spec_run(ids, stop):
            dr.stop = stop
            c = QwenCache(m.cfg, max_len=a.ctx)
            d.attach(c)
            d.prefill(ids)
            nxt = int(np.argmax(np.asarray(d.logits()).reshape(-1)))
            st = {}
            target = spec.QwenTarget(d)
            t1 = time.time()
            out = list(spec.stream(target, DFlash2Drafter(dr), nxt, None, len(ids), dr.drafts,
                                   spec.greedy_pick, eos_ids=(), max_new_tokens=N, stats=st))
            dt = time.time() - t1
            d.detach(c)
            return out, st, dt

        stops = [float(v) for v in a.stops.split(",")]
        res = {}
        for name, text in PROMPTS.items():
            ids = tok.encode("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n" % text)
            ref_out, rate = plain(ids)
            res.setdefault("plain", []).append(rate)
            print("%-8s (%d prompt tokens) plain %.1f tok/s" % (name, len(ids), rate), flush=True)
            for stop in stops:
                for rep in range(2):        # (the first builds the verify groups' programs)
                    out, st, dt = spec_run(ids, stop)
                r = (len(out) - 1) / dt
                res.setdefault(stop, []).append(r)
                same = out == ref_out
                bad += not same
                print("%-8s stop %.2f: %.1f tok/s, %d of %d drafts kept, %.2f tokens a round (draft %.1f ms, "
                      "verify %.1f ms a round)%s" % (
                          name, stop, r, st["accepted"], st["drafts"], len(out) / max(st["steps"], 1),
                          1e3 * st["draft_s"] / max(st["steps"], 1), 1e3 * st["verify_s"] / max(st["steps"], 1),
                          "" if same else ", DIFFERS from plain"), flush=True)
        print("mean over the prompts:")
        for k, v in res.items():
            print("  %-10s %.1f tok/s" % (k if k == "plain" else "stop %.2f" % k, float(np.mean(v))))
    print("PASS" if bad == 0 else "FAIL")
    dr.release()
    d.close()
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
