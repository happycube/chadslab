#!/usr/bin/env python3
"""HotCache's C path (backends/cpu/csrc/common/hotcache.c: rc_hot_score, rc_hot_replace) against its
NumPy path, on the selections of a real MTP decode (Qwen3.8-Flash-Next EXL3 5.05, recorded by
--record: the tables after the prompt, then each score_group call). Each call runs both
paths from the same tables: the scores, uses, totals, runs, held, incoming, slot tables, pool owners
and the copies must be the same, bit for bit. The copies then land at once (their slots hot), so the
tables go on as a decode's would.

    python scripts/check_hotcache.py [recording.npz]   (default ~/reclank-golden/hotcache_mtp.npz)
    python scripts/check_hotcache.py --record MODEL recording.npz   (the GPU: MTP decode, 3 drafts,
        256 tokens of two chat prompts; OMP_NUM_THREADS=1)
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.backends.cuda import runtime as RT  # noqa: E402

TABLES = ("score", "uses", "total", "run", "held", "incoming")


class _Pool:
    def __init__(self, owner):
        self.owner = owner


def _cache(z):
    """A HotCache with the recorded tables (no GPU: the copies are only recorded)."""
    hc = RT.HotCache.__new__(RT.HotCache)
    for nm in TABLES:
        setattr(hc, nm, z[nm].copy())
    m, E = hc.score.shape
    hc.layers = [{"slots": z["slots"][r].copy(), "parts": []} for r in range(m)]
    hc.pool = _Pool(z["owner"].copy()) if z["owner"].size > 1 else None
    hc.pool_rows = z["pool_rows"].copy()
    hc.decay, hc.admit, hc.max_ins = float(z["decay"]), int(z["admit"]), int(z["max_ins"])
    hc.run_mode = bool(z["run_mode"])
    hc.pending, hc.skipped, hc.copies = [], 0, 0
    hc.got = []
    hc._start_copies = lambda items, changed: hc.got.append((list(items), sorted(changed)))
    return hc


def _state(hc):
    d = {nm: getattr(hc, nm).copy() for nm in TABLES}
    d["slots"] = np.stack([e["slots"] for e in hc.layers])
    d["owner"] = hc.pool.owner.copy() if hc.pool is not None else np.zeros(1)
    return d


def _set(hc, d):
    for nm in TABLES:
        getattr(hc, nm)[...] = d[nm]
    for r, e in enumerate(hc.layers):
        e["slots"][...] = d["slots"][r]
    if hc.pool is not None:
        hc.pool.owner[...] = d["owner"]


def record(model, out):
    """The tables of HotCache after the prompt and each score_group call of an MTP decode."""
    from reclank.models.qwen.qwen38 import Qwen4Cache, Qwen4CPU
    from reclank.models.qwen.gpu38 import Qwen4GPU, generate_mtp_gpu
    from reclank.models.qwen.tokenizer import QwenTokenizer
    tok = QwenTokenizer(os.path.join(model, "tokenizer.json"))
    m = Qwen4CPU(model)
    dev = Qwen4GPU(m, ctx=4096)
    hc = dev.hot_cache
    rec, calls = {}, []
    orig = RT.HotCache.score_group

    def sg(rows, sels):
        if not rec:
            rec.update(_state(hc))
            rec.update(pool_rows=hc.pool_rows.copy(), decay=hc.decay, admit=hc.admit,
                       max_ins=hc.max_ins, run_mode=hc.run_mode)
        calls.append((np.asarray(rows, np.int64).copy(), np.asarray(sels, np.int32).copy()))
        return orig(hc, rows, sels)
    hc.score_group = sg
    for q in ("Write a detailed explanation of how a CPU cache hierarchy works, with L1, L2 and L3, "
              "and why it matters for performance.",
              "Write a Python function that parses an ISO 8601 date string without using datetime, with tests."):
        ids = tok.encode("<|im_start|>user\n" + q + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        dev.attach(Qwen4Cache(m.cfg, len(ids) + 320))
        generate_mtp_gpu(dev, ids, 256, draft=3, stats={})
    for c, (r, s_) in enumerate(calls):
        rec["rows%d" % c], rec["sels%d" % c] = r, s_
    rec["n_calls"] = len(calls)
    np.savez_compressed(out, **rec)
    dev.close()
    print("%d calls recorded to %s" % (len(calls), out))
    return 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--record":
        return record(sys.argv[2], sys.argv[3])
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/reclank-golden/hotcache_mtp.npz")
    if not os.path.exists(path):
        print("SKIP: no recording at %s (--record makes one)" % path)
        return 0
    if RT._hot_lib() is None:
        print("FAIL: the CPU library has no rc_hot_score")
        return 1
    z = np.load(path)
    hc = _cache(z)
    calls, bad, copies = int(z["n_calls"]), 0, 0
    for c in range(calls):
        rows, sels = z["rows%d" % c], z["sels%d" % c]
        s0 = _state(hc)
        out = {}
        for c_on in (True, False):
            _set(hc, s0)
            hc.got = []
            RT.HOT_C = c_on
            RT.HotCache.score_group(hc, rows, sels)
            out[c_on] = (_state(hc), hc.got)
        (sc, gc), (sp, gp) = out[True], out[False]
        diff = [k for k in sp if not np.array_equal(sc[k], sp[k])]
        if diff or gc != gp:
            bad += 1
            if bad <= 3:
                print("call %d: the C path differs in %s%s" % (c, diff, "" if gc == gp else " and the copies"))
        # the NumPy result goes on; its copies land at once
        _set(hc, sp)
        for items, _changed in gp:
            copies += len(items)
            for r, x, slot in items:
                hc.layers[r]["slots"][x] = slot
                hc.held[r, x] = True
                hc.incoming[r, x] = False
    RT.HOT_C = True
    print("%d calls of score_group (%d copies): %s" % (calls, copies, "PASS" if bad == 0 else "FAIL: %d differ" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
