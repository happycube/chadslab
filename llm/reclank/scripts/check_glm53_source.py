#!/usr/bin/env python3
"""GLM53Source (reclank/formats/exl3_glm53.py) against the original FP8 checkpoint (P1 of
docs/GLM53_PLAN.md):

- every tensor of the EXL3 directory (the index and mtp.safetensors; the vision tower aside) is the
  source of a GGUF name, and the shapes are those of the config;
- the order of the fused qkv_proj (q | k | v) and the layout of kv_b (head h: 256 rows of keys, then
  256 of values): the decoded EXL3 matrices and the float16 split against the original q/k/v_proj
  and kv_b_proj (bfloat16), each order of the parts tried;
- the error of a few EXL3 matrices against their FP8 / bfloat16 originals (dB: 10 log10 of the
  error energy over the weight energy);
- the float16 / float32 tensors against the bfloat16 originals (max relative difference).

    python scripts/check_glm53_source.py /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
        /space/models/GLM-5.3-Flash
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from reclank.formats.exl3_glm53 import MAIN, GLM53Source  # noqa: E402


class Orig:
    """The tensors of the original checkpoint, float32 (FP8 times its 128 x 128 block scales)."""

    def __init__(self, path):
        where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.t = {}
        for f in sorted(set(where.values())):
            p = os.path.join(path, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            for k, v in h.items():
                s, e = v["data_offsets"]
                self.t[k] = (p, v["dtype"], tuple(v["shape"]), 8 + n + s, e - s)

    def get(self, name):
        p, dt, shape, off, n = self.t[name]
        with open(p, "rb") as fh:
            fh.seek(off)
            b = fh.read(n)
        if dt == "F32":
            return np.frombuffer(b, np.float32).reshape(shape).copy()
        if dt == "BF16":
            return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        assert dt == "F8_E4M3", (name, dt)
        from reclank.formats.dsv41_st import E4M3
        w = E4M3[np.frombuffer(b, np.uint8)].reshape(shape)
        s = self.get(name + "_scale_inv")
        return w * s.repeat(128, 0).repeat(128, 1)[:shape[0], :shape[1]]


def db(got, want):
    e = np.sum((got.astype(np.float64) - want) ** 2)
    return 10 * np.log10(e / np.sum(want.astype(np.float64) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exl3")
    ap.add_argument("orig")
    a = ap.parse_args()
    src = GLM53Source(a.exl3)
    cfg = src.config()
    org = Orig(a.orig)
    bad = []

    # 1. every tensor mapped
    used = set()
    for kind, s in src._map.values():
        if kind in ("e3",):
            used.update(s + x for x in (".trellis", ".suh", ".svh", ".mul1", ".weight"))
        elif kind == "plain":
            used.add(s[0])
        elif kind in ("bf16",):
            used.add(s)
        elif kind == "kvb":
            used.add(s[0])
        elif kind == "e3exps":
            m, x = s
            for e in range(src.E):
                used.update(m + "experts.%d.%s_proj%s" % (e, x, y) for y in (".trellis", ".suh", ".svh", ".mul1"))
    left = sorted(n for n in src.d.where if n not in used and ".visual." not in n and not n.startswith("model.visual"))
    print("tensors of the directory: %d, mapped to %d GGUF names; unmapped: %d %s" % (
        len(src.d.where), len(src.tensors), len(left), left[:8]))
    if left:
        bad.append("unmapped tensors")
    n_kda = sum(1 for g in src.tensors if g.endswith("attn_qkv.weight"))
    n_dsa = sum(1 for g in src.tensors if g.endswith("attn_k_b.weight"))
    n_moe = sum(1 for g in src.tensors if g.endswith("ffn_up_exps.weight"))
    print("layers: KDA %d, DSA %d (with MTP), MoE %d (with MTP); config: %s / %s, MTP %d" % (
        n_kda, n_dsa, n_moe, cfg.layer_types.count("kda"), cfg.layer_types.count("dsa"), cfg.mtp_layers))
    for g, want in (("blk.0.attn_qkv.weight", (4096, 3 * 8192)), ("blk.3.attn_k_b.weight", (512, 256, 64)),
                    ("blk.3.ffn_gate_exps.weight", (4096, 2048, 288)), ("blk.3.idx_ape", (128, 4)),
                    ("blk.0.ssm_conv1d.weight", (4, 24576)), ("output.weight", (4096, 154880))):
        got = src.tensors[g][0]
        print("  %-28s %s %s" % (g, got, "ok" if got == want else "WANT %s" % (want,)))
        if got != want:
            bad.append(g)

    # 2. the order of qkv_proj and the layout of kv_b
    for i in (0, 44):
        w = src.dequant("blk.%d.attn_qkv.weight" % i)                    # (24576, 4096)
        parts = {c: org.get(MAIN + "layers.%d.self_attn.%s_proj.weight" % (i, c)) for c in "qkv"}
        res = []
        for order in itertools.permutations("qkv"):
            want = np.concatenate([parts[c] for c in order])
            res.append((db(w, want), "".join(order)))
        res.sort()
        print("layer %d qkv_proj against the original (dB): %s" % (i, ", ".join("%s %.1f" % (o, d) for d, o in res)))
        if res[0][1] != "qkv":
            bad.append("qkv order of layer %d" % i)
    for i in (3, 43):
        kvb = org.get(MAIN + "layers.%d.self_attn.kv_b_proj.weight" % i).reshape(64, 512, 512)
        kb, vb = src.dequant("blk.%d.attn_k_b.weight" % i), src.dequant("blk.%d.attn_v_b.weight" % i)
        dk, dv = np.abs(kb - kvb[:, :256]).max(), np.abs(vb - kvb[:, 256:]).max()
        rel = max(dk, dv) / np.abs(kvb).max()
        print("layer %d kv_b: attn_k_b / attn_v_b against the original's heads: max diff %.2e / %.2e (rel %.1e)" % (
            i, dk, dv, rel))
        if rel > 1e-3:
            bad.append("kv_b layout of layer %d" % i)

    # 3. EXL3 matrices against the originals
    for g, o in (("blk.0.attn_output.weight", "layers.0.self_attn.o_proj.weight"),
                 ("blk.0.ffn_gate.weight", "layers.0.mlp.gate_proj.weight"),
                 ("blk.3.attn_q_a.weight", "layers.3.self_attn.q_a_proj.weight"),
                 ("blk.3.attn_q_b.weight", "layers.3.self_attn.q_b_proj.weight"),
                 ("blk.3.attn_kv_a_mqa.weight", "layers.3.self_attn.kv_a_proj_with_mqa.weight"),
                 ("blk.3.attn_output.weight", "layers.3.self_attn.o_proj.weight"),
                 ("blk.3.idx_q_b.weight", "layers.3.self_attn.indexer.wq_b.weight"),
                 ("blk.3.ffn_up_shexp.weight", "layers.3.mlp.shared_experts.up_proj.weight"),
                 ("blk.45.nextn.eh_proj.weight", "layers.45.eh_proj.weight")):
        d = db(src.dequant(g), org.get(MAIN + o))
        print("  %-28s %s K=%d: %.1f dB" % (g, src.tensors[g][0], src.tensors[g][1] - 70, d))
        # (the MTP's eh_proj: -18 dB at K=5, each half against its own half; the halves
        # swapped give +1 and +7 dB: the order is right, the quant coarse)
        if d > (-15 if ".nextn." in g else -20):
            bad.append(g)
    for i, e, x in ((3, 0, "gate"), (20, 141, "down"), (44, 287, "up")):
        w = src.dequant("blk.%d.ffn_%s_exps.weight" % (i, x), rows=[e])[0]
        d = db(w, org.get(MAIN + "layers.%d.mlp.experts.%d.%s_proj.weight" % (i, e, x)))
        print("  blk.%d.ffn_%s_exps expert %d: %.1f dB" % (i, x, e, d))
        if d > -20:
            bad.append("expert %d.%d.%s" % (i, e, x))

    # 4. the float16 / float32 tensors
    for g, o in (("blk.3.ffn_gate_inp.weight", "layers.3.mlp.gate.weight"),
                 ("blk.3.exp_probs_b.bias", "layers.3.mlp.gate.e_score_correction_bias"),
                 ("blk.0.hc_attn_fn", "layers.0.hc_attn_fn"), ("blk.10.hc_ffn_base", "layers.10.hc_ffn_base"),
                 ("blk.0.ssm_f_b.weight", "layers.0.self_attn.f_b_proj.weight"),
                 ("blk.0.ssm_a_log", "layers.0.self_attn.A_log"), ("blk.0.ssm_dt.bias", "layers.0.self_attn.dt_bias"),
                 ("blk.3.idx_k.weight", "layers.3.self_attn.indexer.wk.weight"),
                 ("blk.3.idx_ape", "layers.3.self_attn.indexer.index_kpool_compress_ape"),
                 ("blk.3.attn_norm.weight", "layers.3.input_layernorm.weight"),
                 ("output_norm.weight", "norm.weight")):
        got, want = src.dequant(g), org.get(MAIN + o)
        rel = np.abs(got - want).max() / max(np.abs(want).max(), 1e-30)
        print("  %-28s %-8s max rel diff %.1e" % (g, src.d.dtype(src._map[g][1][0]), rel))
        if rel > 1e-3:
            bad.append(g)
    conv = src.dequant("blk.0.ssm_conv1d.weight")
    want = np.concatenate([org.get(MAIN + "layers.0.self_attn.%s_conv1d.weight" % c) for c in "qkv"])[:, 0]
    print("  blk.0.ssm_conv1d.weight      max diff %.1e (q | k | v conv)" % np.abs(conv - want).max())
    if np.abs(conv - want).max() > 1e-6:
        bad.append("conv")
    emb = src.embed_rows([0, 1000, 154879])
    want = org.get(MAIN + "embed_tokens.weight")[[0, 1000, 154879]]
    print("  token_embd rows              equal: %s" % np.array_equal(emb, want))
    if not np.array_equal(emb, want):
        bad.append("embed")

    print("check_glm53_source: %s" % ("OK" if not bad else "FAILED: %s" % bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
