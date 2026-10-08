#!/usr/bin/env python3
"""Convert the original bfloat16 checkpoint of Qwen3.8-Flash-Next
(Qwen/Qwen3.8-Flash-Next, "ORIG") to one GGUF file for this runtime: the
experts (routed, MTP, shared) in rotated Q8_0 (RQ8_0, type 55; --experts q8:
plain Q8_0, the shared experts bfloat16) and the n-gram table in Q8_0.

RQ8_EXPERTS_PLAN.md, section 6. Against ORIG, Q8_0 is about -45.2 dB on the
routed experts and RQ8_0 -45.45 (the worst matrix 1 dB better; the shared
experts, with heavy tails, -43.5 and -45.7), in place of -21.8 dB for the
NVFP4 of ModelOpt and -31.5 dB for its FP8 MTP experts; the n-gram table
-45.5 dB (FP8: -31.5). RQ8_0: the blocks of Q8_0 of each row after the TQ6
rotation of each 32 values (np_gemma.gguf.RQ8_0), so the runtime rotates
the x of the expert products (and their act) the same way.

- The names, the metadata, the MTP layer as blk.48 with nextn.*, and the
  value heads of the DeltaNet in the tiled order: as convert_nvfp4_gguf.py
  (the same NVFP4Source map of the names; ORIG has the names of the ModelOpt
  checkpoint but for the experts).
- The experts: ORIG has them fused, experts.gate_up_proj (512, 1280, 2560)
  with gate in rows 0-639, and experts.down_proj (512, 2560, 640). The file
  has the stacks of llama.cpp (blk.N.ffn_gate_exps.weight, ...: dims (cols,
  rows, 512)), written one expert at a time from the map of ORIG.
- The n-gram table: the bfloat16 shards of ORIG as Q8_0 rows (170 bytes for
  a row of 160); no per_layer_token_embd.scale.
- The dense matrices: bfloat16 (--dense bf16, the default), Q8_0, or BF12
  (--dense bf12: the bfloat16 values in 12.25 bits, plan-scripts/
  BF12_PLAN.md; a matrix that BF12 cannot hold, a zeroed value above 2^-8 of
  its RMS or an Inf or NaN, stays bfloat16; the report of each matrix in
  <out>.bf12.json).
- --experts-from FILE: the routed experts as they are in FILE (the experts
  file of --experts-only of the same --experts and --map): no quantization,
  no read of their part of ORIG.
- np_gemma.quant_error.<part>: the mean error in dB against ORIG of a
  sample of each quantized part (measured before the write).

The source is read ahead in the order of the output by a thread (the HDD
reads about 220 MB/s; the quantization and the write keep up), so the
conversion takes about the time of one read of ORIG (336 GB).

    python scripts/convert_q8_gguf.py /space/models/Qwen3.8-Flash-Next \\
        /mnt/pmem/Qwen3.8-Flash-Next-Q8-GGUF/Qwen3.8-Flash-Next-Q8-bf16.gguf
"""
from __future__ import annotations

import argparse
import re
import os
import shutil
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from convert_nvfp4_gguf import metadata, reorder  # noqa: E402
from np_gemma import cops  # noqa: E402
from np_gemma.gguf import (BF12, GGUF, Q8_0, RQ6_K, RQ8_0, rotation_meta, tensor_bytes,  # noqa: E402
                           write_gguf)
from np_gemma.st_qwen4 import NVFP4Source  # noqa: E402


def to_q8(w, rot):
    """Q8_0 blocks of bfloat16 rows (uint16), or RQ8_0 (rot: the rows rotated
    in each 32 values first)."""
    if not rot:
        return cops.kq_to_q8_0(w, w.shape[-1])
    return cops.kq_to_q8_0(cops.tq6_rotate(bf16f(w)), w.shape[-1])


def to_type(w, t):
    """The blocks of bfloat16 rows in type t: Q8_0, RQ8_0, or RQ6_K (Q6_K of
    the rotated rows, as llama.cpp quantizes Q6_K)."""
    if t == RQ6_K:
        return cops.kq_to_q6_k(cops.tq6_rotate(bf16f(w)), w.shape[-1])
    return to_q8(w, t == RQ8_0)


def back_type(q, cols, t):
    """The values of blocks of rows of cols values in type t."""
    kt = {RQ6_K: 14}.get(t, Q8_0)
    rb = cols // 256 * 210 if kt == 14 else cols // 32 * 34
    v = cops.kq_rows(q, kt, cols, np.arange(q.size // rb))
    return cops.tq6_rotate(v, inverse=True) if t in (RQ8_0, RQ6_K) else v


def ud_map(path, L):
    """The types of the routed experts of RQ6_MIX_PLAN.md from the choices of
    a GGUF of Unsloth (UD-Q4_K_XL): {(layer, "gate"|"up"|"down"): type}.
    Where it paid for Q8_0, RQ8_0; else RQ6_K if the rows are whole blocks
    of 256 (gate and up: 2560 inputs), else RQ8_0 (down: 640 inputs, M1)."""
    from np_gemma.gguf import open_gguf
    g = open_gguf(path)
    out = {}
    for i in range(L):
        for x in ("gate", "up", "down"):
            dims, t, _o = g.tensors["blk.%d.ffn_%s_exps.weight" % (i, x)]
            out[(i, x)] = RQ8_0 if t == Q8_0 or int(dims[0]) % 256 else RQ6_K
    g.close()
    return out


class OrigSource(NVFP4Source):
    """The GGUF view of ORIG: the map of NVFP4Source, with the fused bfloat16
    experts and the bfloat16 n-gram table in Q8_0. rot: the experts (and the
    shared experts) in RQ8_0."""

    def __init__(self, path, headers=None, rot=False, xtypes=None):
        """xtypes: {(layer, "gate"|"up"|"down"): type} of the routed experts
        (ud_map; the others RQ8_0 with rot, else Q8_0)."""
        self.rot = rot
        self.xtypes = xtypes or {}
        super().__init__(path, headers=headers)

    def _experts(self, b, m, mtp):
        gu, dn = m + "experts.gate_up_proj", m + "experts.down_proj"
        E, rows2, cols = self._shape(gu)
        inter = rows2 // 2
        assert E == self.E and tuple(self._shape(dn)) == (E, cols, inter), (gu, self._shape(dn))
        i = int(b.split(".")[1])
        t0 = RQ8_0 if self.rot else Q8_0
        tg, tu, td = (self.xtypes.get((i, x), t0) for x in ("gate", "up", "down"))
        self._add(b + "ffn_gate_exps.weight", "q8exps", (gu, 0, inter, tg), (cols, inter, E), tg)
        self._add(b + "ffn_up_exps.weight", "q8exps", (gu, inter, rows2, tu), (cols, inter, E), tu)
        self._add(b + "ffn_down_exps.weight", "q8exps", (dn, 0, cols, td), (inter, cols, E), td)

    def _build(self):
        super()._build()
        g = "per_layer_token_embd.weight"
        dims, _t, _ = self.tensors[g]
        self._add(g, "q8ple", None, dims, Q8_0)
        if self.rot:
            # the shared experts too: the MoE then rotates its x for all its
            # experts
            for n, (kind, hf) in list(self._map.items()):
                if kind == "dense" and n.endswith(("ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                                   "ffn_down_shexp.weight")):
                    self._add(n, "rq8dense", hf, self.tensors[n][0], RQ8_0)

    def _make(self, gname):
        kind, src = self._map[gname]
        if kind == "q8exps":
            hf, lo, hi, t = src
            return self._q8_experts(hf, lo, hi, t)
        if kind == "q8ple":
            return self._q8_ple()
        if kind == "rq8dense":
            return to_q8(np.asarray(self._bf16(src)), True)
        return super()._make(gname)

    def _q8_experts(self, hf, lo, hi, t):
        """Rows lo to hi of each expert of a fused stack, one expert at a
        time, in type t."""
        w = self._bf16(hf)
        for e in range(self.E):
            yield to_type(w[e, lo:hi], t)

    def _q8_ple(self):
        for n in self._ple_shards:
            w = self._bf16(n)
            yield cops.kq_to_q8_0(w, w.shape[1])

    def sources(self, gname):
        """The safetensors names that gname reads."""
        kind, src = self._map[gname]
        if kind == "f32":
            return [src[0]]
        if kind in ("bf16", "dense", "rq8dense"):
            return [src]
        if kind == "ehproj":
            return ["mtp.fc_embedding.weight", "mtp.fc_hidden.weight"]
        if kind == "q8exps":
            return [src[0]]  # (hf, lo, hi, type)
        if kind == "q8ple":
            return list(self._ple_shards)
        raise KeyError(gname)

    def span(self, hf):
        """(path, start, end) of the bytes of a safetensors tensor."""
        st = self._file(hf)
        a, b = st.header[hf]["data_offsets"]
        return os.path.join(self.path, self.where[hf]), st._base + a, st._base + b


def db(w, q):
    w = np.asarray(w, np.float64)
    return 10 * np.log10(((np.asarray(q, np.float64) - w) ** 2).mean() / (w ** 2).mean())


def bf16f(a):
    return (np.asarray(a, np.uint16).astype(np.uint32) << 16).view(np.float32)


def back(q, cols, rot):
    """The values of Q8_0 (or RQ8_0) blocks of rows of cols values."""
    v = cops.kq_rows(q, Q8_0, cols, np.arange(q.size // (cols // 32 * 34)))
    return cops.tq6_rotate(v, inverse=True) if rot else v


def errors(src):
    """The error in dB of Q8_0 (RQ8_0) against ORIG on a sample of each part."""
    out, by = {}, {}
    L = src.L
    rot = src.rot
    for part, layers in (("experts", [0, 12, 24, 36, L - 1]), ("mtp_experts", [L])):
        v = []
        for i in layers:
            if "blk.%d.ffn_gate_exps.weight" % i not in src._map:
                continue
            for x in ("gate", "up", "down"):
                hf, lo, hi, t = src._map["blk.%d.ffn_%s_exps.weight" % (i, x)][1]
                w = src._bf16(hf)
                for e in (0, 171, 342, 511):
                    a = np.ascontiguousarray(w[e, lo:hi])
                    d = db(bf16f(a), back_type(to_type(a, t), a.shape[1], t))
                    v.append(d)
                    by.setdefault("%s_%s" % (part, {RQ6_K: "rq6_k", RQ8_0: "rq8_0"}.get(t, "q8_0")),
                                  []).append(d)
        if v:
            out[part] = float(np.mean(v))
    for k, v in by.items():
        out[k] = float(np.mean(v))
    if rot:
        v = []
        for i in (0, 12, 24, 36, L - 1, L):
            for x in ("gate", "up", "down"):
                n = "blk.%d.ffn_%s_shexp.weight" % (i, x)
                if n in src._map and src._map[n][0] == "rq8dense":
                    a = np.asarray(src._bf16(src._map[n][1]))
                    v.append(db(bf16f(a), back(to_q8(a, True), a.shape[1], True)))
        if v:
            out["shared_experts"] = float(np.mean(v))
    if "per_layer_token_embd.weight" in src._map:
        v = []
        for n in src._ple_shards[::26]:
            w = src._bf16(n)
            r0 = w.shape[0] // 2            # a block of rows (random rows: a seek each)
            a = np.ascontiguousarray(w[r0:r0 + 20000])
            q = cops.kq_rows(cops.kq_to_q8_0(a, a.shape[1]), Q8_0, a.shape[1], np.arange(a.shape[0]))
            v.append(db(bf16f(a), q))
        out["ngram"] = float(np.mean(v))
    return out


class Prefetch(threading.Thread):
    """Read the source bytes of the tensors in the order of the output, at
    most ahead bytes past the tensor being written (the page cache then has
    them when the writer maps them)."""

    def __init__(self, spans, ahead=12e9):
        super().__init__(daemon=True)
        self.spans = spans          # for each output tensor: [(path, start, end)]
        self.ahead = ahead
        self.cv = threading.Condition()
        self.done = 0               # the output tensors written
        self.fds = {}

    def wrote(self):
        with self.cv:
            self.done += 1
            self.cv.notify_all()

    def run(self):
        seen = set()
        pos = 0
        ends = np.cumsum([sum(e - s for _p, s, e in sp) for sp in self.spans])
        for k, sp in enumerate(self.spans):
            with self.cv:
                while pos - (ends[self.done - 1] if self.done else 0) > self.ahead:
                    self.cv.wait(1.0)
            for path, s, e in sp:
                pos += e - s
                if (path, s) in seen:
                    continue
                seen.add((path, s))
                fd = self.fds.get(path)
                if fd is None:
                    fd = self.fds[path] = os.open(path, os.O_RDONLY)
                o = s
                while o < e:
                    n = min(32 << 20, e - o)
                    os.pread(fd, n, o)
                    o += n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="the ORIG checkpoint directory")
    ap.add_argument("out", help="the GGUF file to write")
    ap.add_argument("--dense", choices=("bf16", "q8", "bf12"), default="bf16")
    ap.add_argument("--experts-from", default=None,
                    help="the routed experts as they are in this GGUF (an --experts-only file of "
                         "the same --experts and --map)")
    ap.add_argument("--experts", choices=("rq8", "q8", "mix"), default="rq8",
                    help="rq8: the routed, MTP, and shared experts in RQ8_0; q8: the routed and "
                         "MTP experts in Q8_0, the shared experts as --dense; mix: as rq8, but "
                         "the routed experts in RQ6_K or RQ8_0 by --map (RQ6_MIX_PLAN.md)")
    ap.add_argument("--map", default="models2/Qwen3.8-Flash-Next-GGUF/UD-Q4_K_XL/"
                    "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf",
                    help="mix: the GGUF whose types choose: Q8_0 there -> RQ8_0, else RQ6_K "
                         "(whole blocks of 256) or RQ8_0")
    ap.add_argument("--experts-only", action="store_true",
                    help="write only the routed experts (a file that a model of the same "
                         "rotation takes over its own experts: Qwen4CPU experts=)")
    ap.add_argument("--no-errors", action="store_true", help="skip the sample of the errors")
    ap.add_argument("--only", default=None, help="a test: only the tensors whose names match "
                    "this regular expression")
    args = ap.parse_args()
    xtypes = None
    if args.experts == "mix":
        import json as _json
        L0 = _json.load(open(os.path.join(args.src, "config.json")))
        L0 = int(L0.get("text_config", L0)["num_hidden_layers"])
        xtypes = ud_map(args.map, L0)
        n6 = sum(t == RQ6_K for t in xtypes.values())
        print("the map of %s: %d of %d expert matrices RQ6_K, the others RQ8_0" % (
            os.path.basename(args.map), n6, len(xtypes)), flush=True)
    src = OrigSource(args.src, rot=args.experts in ("rq8", "mix"), xtypes=xtypes)
    cfg = src.config()
    hk, hv, dk, dv = cfg.lin_k_heads, cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim
    rep = hv // hk
    qk = 2 * hk * dk

    def tiled(gname, a):
        """The value heads of a DeltaNet tensor in the tiled order."""
        if gname.endswith(("attn_qkv.weight", "ssm_conv1d.weight")):
            return np.concatenate([a[:qk], reorder(a[qk:], 0, hk, rep, dv)])
        if gname.endswith("attn_gate.weight"):
            return reorder(a, 0, hk, rep, dv)
        if gname.endswith(("ssm_alpha.weight", "ssm_beta.weight", "ssm_a", "ssm_dt.bias")):
            return reorder(a, 0, hk, rep, 1)
        if gname.endswith("ssm_out.weight"):
            return reorder(a, 1, hk, rep, dv)
        return a

    def bf12_ok(gname):
        """A matrix that --dense bf12 makes BF12 (the rule of the runtime's
        dense modes: two dims, more than one row, cols a multiple of 32, not
        the indexer nor the embeddings)."""
        kind, _s = src._map[gname]
        dims = src.tensors[gname][0]
        return (args.dense == "bf12" and kind in ("dense", "ehproj") and len(dims) == 2
                and dims[1] > 1 and dims[0] % 32 == 0 and ".indexer." not in gname
                and gname != "token_embd.weight")

    def bf16_of(gname):
        """The bfloat16 rows (uint16) of a dense matrix in the order of the file."""
        kind, s = src._map[gname]
        if kind == "dense" and gname.endswith(("attn_qkv.weight", "attn_gate.weight",
                                               "ssm_out.weight")):
            return np.ascontiguousarray(tiled(gname, np.asarray(src._bf16(s))))
        return np.ascontiguousarray(np.asarray(src._make(gname)))

    def maker(gname):
        kind, s = src._map[gname]
        if gname in bf12_rows:
            return lambda: bf12_rows.pop(gname)
        if xfile is not None and kind == "q8exps" and int(gname.split(".")[1]) < src.L:
            return lambda: xfile.raw(gname)[0].view(np.uint8)
        if kind == "dense" and gname.endswith(("attn_qkv.weight", "attn_gate.weight",
                                               "ssm_out.weight")):
            def make():
                h = tiled(gname, np.asarray(src._bf16(s)))
                return cops.kq_to_q8_0(h, h.shape[1]) if args.dense == "q8" else h
            return make
        if kind == "f32":
            return lambda: tiled(gname, src._make(gname))
        if kind in ("dense", "ehproj") and args.dense == "q8":
            def make_q8():
                h = np.asarray(src._make(gname))
                return cops.kq_to_q8_0(h, h.shape[1])
            return make_q8
        return lambda: src._make(gname)

    # --experts-from: the routed experts as they are in that file
    xfile = GGUF(args.experts_from, stage=False) if args.experts_from else None
    # --dense bf12: each matrix first (its type depends on its values): the
    # rows kept for the write, the report
    bf12_rows, report, kept = {}, {}, []
    if args.dense == "bf12":
        t0 = time.time()
        names = [n for n in src._map if bf12_ok(n) and not (args.only and not re.search(args.only, n))]
        for gname in names:
            a = bf16_of(gname)
            rows, rep_ = cops.kq_bf16_to_bf12(a.view(np.uint16), a.shape[1])
            rms = rep_["rms"]
            bad = rep_["nonfinite"] > 0 or rep_["largest"] > rms * 2.0 ** -8
            report[gname] = dict(rep_, rows=a.shape[0], cols=a.shape[1], kept_bf16=bool(bad))
            if bad:
                kept.append(gname)
            else:
                bf12_rows[gname] = rows.reshape(-1)
        nz = sum(r["zeroed"] for r in report.values())
        nv = sum(r["rows"] * r["cols"] for r in report.values())
        print("BF12: %d matrices (%.2f G values): %d zeroed values (%.4f%%), %d collisions, the largest "
              "zeroed %.3g (%.2e of its RMS, %s); %d kept bfloat16 %s (%.0f s)" % (
                  len(report), nv / 1e9, nz, 100.0 * nz / max(nv, 1),
                  sum(r["collisions"] for r in report.values()),
                  max((r["largest"] for r in report.values()), default=0.0),
                  max((r["largest"] / max(r["rms"], 1e-30) for r in report.values()), default=0.0),
                  max(report, key=lambda n: report[n]["largest"] / max(report[n]["rms"], 1e-30))
                  if report else "-", len(kept), kept, time.time() - t0), flush=True)
        for gname in kept:
            r = report[gname]
            print("  kept bfloat16: %s: %d zeroed, the largest %.3g (RMS %.3g), %d Inf/NaN; worst %s"
                  % (gname, r["zeroed"], r["largest"], r["rms"], r["nonfinite"], r["worst"][:5]),
                  flush=True)
    tensors = []
    for gname, (kind, _s) in src._map.items():
        if args.only and not re.search(args.only, gname):
            continue
        if args.experts_only and not (kind == "q8exps" and int(gname.split(".")[1]) < src.L):
            continue
        dims, t, _ = src.tensors[gname]
        if kind in ("dense", "ehproj") and args.dense == "q8":
            t = Q8_0
        if gname in bf12_rows:
            t = BF12
        if xfile is not None and kind == "q8exps" and int(gname.split(".")[1]) < src.L:
            xd, xt, _o = xfile.tensors[gname]
            assert tuple(xd) == tuple(dims) and xt == t, (gname, xd, dims, xt, t)
        tensors.append((gname, dims, t, maker(gname)))
    total = sum(tensor_bytes(d, t) for _n, d, t, _m in tensors)
    print("%d tensors, %.1f GB (experts %s, n-gram table Q8_0, dense %s) -> %s"
          % (len(tensors), total / 1e9, args.experts.upper(), args.dense, args.out), flush=True)

    meta = dict(metadata(cfg, src))
    meta["general.name"] = "Qwen3.8-Flash-Next %s (np_gemma)" % (
        "experts RQ6_K/RQ8_0 (%s)" % os.path.basename(args.map) if args.experts == "mix"
        else "RQ8_0" if src.rot else "Q8_0")
    if args.experts_only:
        meta["general.name"] += ", the routed experts only"
        meta["np_gemma.experts_only"] = True
    meta["general.source"] = "Qwen/Qwen3.8-Flash-Next (original bfloat16)"
    if args.dense == "bf12":
        meta["general.name"] += ", dense BF12"
        meta["np_gemma.bf12.kept_bf16"] = ",".join(kept)
        meta["np_gemma.bf12.zeroed"] = int(sum(r["zeroed"] for r in report.values()))
    if src.rot:
        meta.update(rotation_meta())
    if not args.no_errors:
        t0 = time.time()
        err = errors(src)
        for k, v in err.items():
            meta["np_gemma.quant_error.%s" % k] = v
        print("%s against ORIG (a sample): %s (%.0f s)" % (args.experts.upper(),
            ", ".join("%s %.2f dB" % kv for kv in err.items()), time.time() - t0), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    def spans(n):
        # (the experts of --experts-from and the BF12 rows read nothing more)
        if n in bf12_rows or (xfile is not None and src._map[n][0] == "q8exps"
                              and int(n.split(".")[1]) < src.L):
            return []
        return [src.span(h) for h in src.sources(n)]
    pre = Prefetch([spans(n) for n, _d, _t, _m in tensors])
    pre.start()
    t0 = time.time()
    done = [0, 0.0]

    def progress(name, n):
        pre.wrote()
        done[0] += n
        if done[0] - done[1] >= 5e9 or done[0] == total:
            done[1] = done[0]
            dt = time.time() - t0
            print("  %6.1f of %.1f GB, %5.0f s, %.0f MB/s (%s)" % (
                done[0] / 1e9, total / 1e9, dt, done[0] / dt / 1e6, name), flush=True)

    tmp = args.out + ".part"
    write_gguf(tmp, list(meta.items()), tensors, progress=progress)
    os.replace(tmp, args.out)
    if report:
        import json as _json
        _json.dump(report, open(args.out + ".bf12.json", "w"), indent=1)
    for f in ("tokenizer.json", "chat_template.jinja"):
        if os.path.exists(os.path.join(args.src, f)):
            shutil.copy(os.path.join(args.src, f), os.path.join(os.path.dirname(os.path.abspath(args.out)), f))
    print("done: %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
