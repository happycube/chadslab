#!/usr/bin/env python3
"""Checks of the checkpoint nvidia/Qwen3.8-Flash-Next-FP4-Local (docs/FP4LOCAL_PLAN.md).

--scan (P0): every tensor of the shards in the order of the files.

- The CSF block scales of the routed experts (73,728 matrices): rc_csf_decode
  (csrc/kernels/kq/convert.c) against a NumPy transcription of the oracle
  (flashinfer scripts/convert_csf_blocks.py _logical_source) on every matrix, and
  the checks of the format: the row bases <= 240, the exceptions in order, in range
  and outside their row's window; the scale codes after decoding (zero, E4M3
  subnormal: the W4A16 packer drops a nonzero scale below 1/64, NaN 0x7F).
- weight_scale_2 of gate and up equal (the loader requires it); the input scales.
- The plain E4M3 scales of the MTP experts, the vision W4A16 linears and the
  n-gram table: the same code checks.
- The E8M0 scales of the MXFP8 matrices: the range (255 is NaN).

--source LAYERS (P1): reclank/formats/fp4local_st.py against the formats' definitions,
straight from the shards: the KQ_NVX stacks of the routed experts of those layers
(sampled experts: the rows of the stack equal E2M1 x the oracle's E4M3 scales x
weight_scale_2), the MTP experts, the MXFP8 matrices (exact in bfloat16; the float32
ones), the n-gram rows, the norms (bfloat16, + 1), the config.

    python scripts/check_fp4l_source.py models/Qwen3.8-Flash-Next-FP4-Local --scan --out scan.json
    python scripts/check_fp4l_source.py models/Qwen3.8-Flash-Next-FP4-Local --source 0,3,47
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.safetensors import SafeTensors  # noqa: E402

FIX, EXC = ".weight_scale.nvfp4_csf_fixed", ".weight_scale.nvfp4_csf_exceptions"


def oracle(fixed, records, rows, columns, bits=4):
    """convert_csf_blocks._logical_source, as it is."""
    row_bytes = (columns * bits + 7) // 8
    codes = fixed[:, 16:].reshape(rows, row_bytes)
    bit_columns = np.arange(columns) * bits
    result = fixed[:, :16].reshape(rows, 1) + (
        (codes[:, bit_columns // 8] >> (bit_columns % 8)) & ((1 << bits) - 1))
    result = np.ascontiguousarray(result, dtype=np.uint8)
    result.ravel()[records & 0xFFFFFF] = records >> 24
    return result


def code_stats(s):
    """Counts of the E4M3 scale codes of s: zero, subnormal, NaN, negative; min / max of the rest."""
    s = s.reshape(-1)
    mag = s & 0x7F
    pos = mag[(mag & 0x78) != 0]
    return dict(zero=int(np.count_nonzero(mag == 0)),
                subnormal=int(np.count_nonzero(((mag & 0x78) == 0) & (mag != 0))),
                nan=int(np.count_nonzero(mag == 0x7F)), negative=int(np.count_nonzero(s & 0x80)),
                lo=int(pos.min()) if pos.size else None, hi=int(pos.max()) if pos.size else None)


class Agg:
    def __init__(self):
        self.d = defaultdict(int)
        self.lo, self.hi = 255, 0

    def add(self, st):
        for k, v in st.items():
            if k == "lo" and v is not None:
                self.lo = min(self.lo, v)
            elif k == "hi" and v is not None:
                self.hi = max(self.hi, v)
            elif k == "max_base":
                self.d[k] = max(self.d[k], v)
            elif k not in ("lo", "hi"):
                self.d[k] += v

    def out(self):
        return dict(self.d, code_lo=self.lo, code_hi=self.hi)


def scan(path, out):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    by_file = defaultdict(list)
    for n, f in idx.items():
        by_file[f].append(n)
    res = dict(csf=Agg(), csf_matrices=0, csf_mismatch=0, csf_exceptions=0, csf_scales=0,
               mtp=Agg(), vision=Agg(), ple=Agg(), mxfp8=defaultdict(int), ws2_unequal=0,
               input_scale=[np.inf, 0.0], ws2=[np.inf, 0.0])
    per = defaultdict(Agg)
    ws2 = {}
    t0 = time.time()
    for fi, f in enumerate(sorted(by_file)):
        st = SafeTensors(os.path.join(path, f))
        names = sorted(by_file[f], key=lambda n: st.header[n]["data_offsets"][0])
        for n in names:
            if n.endswith(FIX):
                base = n[:-len(FIX)]
                if base + EXC not in idx:
                    raise SystemExit("no exceptions for " + n)
                fixed = st.get(n, dtype=None)
                exc = (st if idx[base + EXC] == f else SafeTensors(os.path.join(path, idx[base + EXC]))).get(
                    base + EXC, dtype=None).astype(np.uint32)
                w = base + ".weight"
                rows, half = (st if idx[w] == f else SafeTensors(os.path.join(path, idx[w]))).shape(w)
                c = 2 * half // 16
                assert fixed.shape == (rows // 16, 16 * (1 + c // 2)), (n, fixed.shape, rows, c)
                got, chk = cops.csf_decode(fixed, exc, rows, c)
                want = oracle(np.asarray(fixed), exc, rows, c)
                res["csf_mismatch"] += int(not np.array_equal(got, want))
                chk.update(code_stats(got))
                res["csf"].add(chk)
                proj = base.rsplit(".", 1)[1]
                per[proj].add(chk)
                res["csf_matrices"] += 1
                res["csf_exceptions"] += int(exc.size)
                res["csf_scales"] += rows * c
            elif n.endswith(".weight_scale_2"):
                v = float(np.asarray(st.get(n)).reshape(-1)[0])
                res["ws2"] = [min(res["ws2"][0], v), max(res["ws2"][1], v)]
                p = n[:-len(".weight_scale_2")]
                for a, b in ((".gate_proj", ".up_proj"),):
                    if p.endswith(a) or p.endswith(b):
                        key = p[:-len(a)] if p.endswith(a) else p[:-len(b)]
                        if key in ws2:
                            res["ws2_unequal"] += int(ws2.pop(key) != v)
                        else:
                            ws2[key] = v
            elif n.endswith(".input_scale"):
                v = float(np.asarray(st.get(n)).reshape(-1)[0])
                res["input_scale"] = [min(res["input_scale"][0], v), max(res["input_scale"][1], v)]
            elif n.endswith(".weight_scale") and st.dtype(n) == "F8_E4M3":
                s = st.get(n, dtype=None)
                kind = "mtp" if n.startswith("mtp.") else "vision" if ".visual." in n else \
                    "ple" if ".ngram_embedding." in n else None
                if kind is None:
                    raise SystemExit("an E4M3 scale of an unknown part: " + n)
                res[kind].add(code_stats(np.asarray(s)))
            elif n.endswith(".weight_scale") and st.dtype(n) == "U8":
                s = np.asarray(st.get(n, dtype=None)).reshape(-1)
                h = np.bincount(s, minlength=256)
                for b in np.nonzero(h)[0]:
                    res["mxfp8"][int(b)] += int(h[b])
        st.release_pages()
        st.close()
        print("%d/%d %s: %d CSF matrices so far (%.0f s)" % (fi + 1, len(by_file), f, res["csf_matrices"],
                                                             time.time() - t0), flush=True)
    m = res.pop("mxfp8")
    out_d = {k: (v.out() if isinstance(v, Agg) else v) for k, v in res.items()}
    out_d["csf_by_projection"] = {k: v.out() for k, v in per.items()}
    out_d["mxfp8_scale_bytes"] = dict(lo=min(m), hi=max(m), n255=m.get(255, 0), n0=m.get(0, 0),
                                      count=sum(m.values()))
    out_d["unpaired_ws2"] = len(ws2)
    print(json.dumps(out_d, indent=1))
    if out:
        json.dump(out_d, open(out, "w"), indent=1)
    ok = (out_d["csf_mismatch"] == 0 and out_d["csf"]["bad_base"] == 0 and out_d["csf"]["bad_exc"] == 0
          and out_d["ws2_unequal"] == 0 and out_d["csf"]["nan"] == 0 and out_d["mxfp8_scale_bytes"]["n255"] == 0)
    print("check_fp4l_source --scan: %s" % ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def nvx_diff(got, codes, scales, g):
    """The values of a dequantized KQ_NVX matrix that differ from the NVFP4 format:
    (E2M1 x e4m3) x g, the exact product rounded once."""
    from reclank.formats.fp4local_st import nvfp4_rows
    return int(np.count_nonzero(got != nvfp4_rows(codes, scales, g)))


def source(path, layers):
    from reclank.formats.fp4local_st import FP4LSource, bf16r, mxfp8_dequant, nvfp4_rows
    from reclank.formats.nvfp4_st import MAIN
    src = FP4LSource(path)
    cfg = src.config()
    rng = np.random.default_rng(0)
    fails = []

    def check(what, ok, detail=""):
        print("%-60s %s %s" % (what, "ok" if ok else "FAILED", detail), flush=True)
        if not ok:
            fails.append(what)

    raw = lambda n: src._file(n).get(n, dtype=None)  # noqa: E731
    for i in layers:
        m = MAIN + "layers.%d.mlp." % i
        for x in ("gate", "up", "down"):
            g = "blk.%d.ffn_%s_exps.weight" % (i, x)
            es = sorted(rng.choice(src.E, 6, replace=False).tolist() + [0, src.E - 1])
            got = src.dequant(g, rows=es)
            worst = 0
            for j, e in enumerate(es):
                p = m + "experts.%d.%s_proj" % (e, x)
                rows, half = src._shape(p + ".weight")
                sc = oracle(np.asarray(raw(p + FIX)), raw(p + EXC).astype(np.uint32), rows, half // 8)
                g2 = float(np.asarray(src._get(p + ".weight_scale_2")).reshape(-1)[0])
                worst = max(worst, nvx_diff(got[j], raw(p + ".weight"), sc, g2))
            check("layer %d %s experts %s (KQ_NVX = the format)" % (i, x, es[:3]), worst == 0,
                  "(%d values differ; CSF %s)" % (worst, src.csf_stats))
        for gname, hf in (("blk.%d.ffn_gate_shexp.weight" % i, m + "shared_expert.gate_proj.weight"),
                          ("blk.%d.ffn_down_shexp.weight" % i, m + "shared_expert.down_proj.weight")):
            w = mxfp8_dequant(raw(hf), raw(hf[:-7] + ".weight_scale"))
            b = src.raw(gname)[0]
            back = (np.asarray(b).astype(np.uint32) << 16).view(np.float32).reshape(w.shape)
            check("%s (MXFP8 as KQ_BF16, exact)" % gname, np.array_equal(back, w))
        t = cfg.layer_types[i]
        hf = MAIN + "layers.%d.%s" % (i, "linear_attn.in_proj_a.weight" if t == "linear_attention"
                                          else "self_attn.indexer.index_qk_proj.weight")
        gname = "blk.%d.%s" % (i, "ssm_alpha.weight" if t == "linear_attention" else "indexer.q_proj.weight")
        w = mxfp8_dequant(raw(hf), raw(hf[:-7] + ".weight_scale"))
        got = src.raw(gname)[0].reshape(-1)
        check("%s (MXFP8 as float32)" % gname, np.array_equal(got, w[:got.size // w.shape[1]].reshape(-1)))
        hc = MAIN + "layers.%d.attn_hyper_connection.hc_norm.weight" % i
        got = src.raw("blk.%d.hc_attn_norm.weight" % i)[0]
        check("blk.%d.hc_attn_norm (float32 -> bfloat16, + 1)" % i,
              np.array_equal(got, bf16r(np.asarray(src._file(hc).get(hc))) + np.float32(1)))
    L = src.L
    g = "blk.%d.ffn_up_exps.weight" % L
    if g in src.tensors:
        got = src.dequant(g, rows=[5])[0]
        p = "mtp.layers.0.mlp.experts.5.up_proj"
        g2 = float(np.asarray(src._get(p + ".weight_scale_2")).reshape(-1)[0])
        check("MTP experts (plain E4M3 scales, KQ_NVX)", nvx_diff(got, raw(p + ".weight"), raw(p + ".weight_scale"), g2) == 0)
    rows = np.concatenate([rng.integers(0, int(src._ple_start[-1]), 64), [0, int(src._ple_start[-1]) - 1]])
    got = src.ple_rows(rows)
    want = []
    for r in rows:
        k = int(np.searchsorted(src._ple_start, r, side="right") - 1)
        n = src._ple_shards[k]
        rr = int(r - src._ple_start[k])
        want.append(nvfp4_rows(raw(n)[rr:rr + 1], raw(n + "_scale")[rr:rr + 1], src._ple_scale())[0])
    check("n-gram rows (NVFP4 rows of %d)" % got.shape[1], np.array_equal(got, np.array(want)))
    check("config: ple_row %d, ple_bf16, experts %d top %d" % (cfg.ple_row, cfg.num_experts, cfg.top_k),
          cfg.ple_row == 160 and cfg.ple_bf16 and cfg.num_experts == 512 and cfg.top_k == 10)
    print("check_fp4l_source --source: %s" % ("OK" if not fails else "FAILED: %s" % fails))
    return 0 if not fails else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--source", default=None, help="layers, comma separated")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rc = 0
    if a.scan:
        rc |= scan(a.path, a.out)
    if a.source:
        rc |= source(a.path, [int(x) for x in a.source.split(",")])
    return rc


if __name__ == "__main__":
    sys.exit(main())
