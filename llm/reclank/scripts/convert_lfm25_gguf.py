#!/usr/bin/env python3
"""Convert the bfloat16 checkpoint of LiquidAI/LFM2.5-24B-A2B to one GGUF file
for this runtime: the matrices in BF12 (the bfloat16 values in 12.25 bits,
plan-scripts/BF12_PLAN.md), the norms, the router, the expert bias, and the
convolution weights float32 (docs/LFM25_24B_PLAN.md, phase 1).

The names and the metadata are those of llama.cpp's lfm2moe (the official
LiquidAI/LFM2.5-24B-A2B-GGUF files):

    token_embd.weight              the embeddings, and the head (tied)
    token_embd_norm.weight         the final norm (embedding_norm of the
                                   checkpoint: it runs after the last layer)
    blk.N.attn_norm / ffn_norm     operator_norm / ffn_norm
    blk.N.shortconv.in_proj, .conv, .out_proj      the conv layers
    blk.N.attn_q, _k, _v, _output, attn_q_norm, attn_k_norm
    blk.N.ffn_gate, _up, _down     the dense layers 0 and 1
    blk.N.ffn_gate_inp             the router (float32)
    blk.N.exp_probs_b.bias         the expert bias (float32; the choice only)
    blk.N.ffn_gate_exps, _up_exps, _down_exps      the stacks of the 64
                                   experts (w1, w3, w2 of each expert)

lfm2moe.attention.head_count_kv is a list with 0 for the conv layers. A
matrix (or an expert stack) that BF12 cannot hold (a zeroed value above 2^-8
of its RMS, or an Inf or NaN) stays bfloat16, as convert_q8_gguf.py does.

--experts rq8: the routed experts in RQ8_0 (type 55: Q8_0 of the rows after
the TQ6 rotation of each 32 values, gguf.RQ8_0; the runtime rotates the x of
the expert products and their act, compile_qwen_step and kq_set_moe_rot):
24.4 GB in place of 34.4, so the GPU holds more of them. The dense matrices
stay BF12. reclank.quant_error.experts: the error of a sample of experts in
dB against bfloat16.

--experts rq6: the routed experts in RQ6_K (type 56: Q6_K of the rotated
rows, 6.5625 bits a value; 18.8 GB): more of them fit on the GPU, and the
copies of a prompt are smaller.

--experts rq5 / rq4: RQ5_K / RQ4_K (Q5_K / Q4_K of the rotated rows, 5.5 /
4.5 bits a value; 15.8 / 12.9 GB), with the better encoder of convert.c
(cops.kq_to_qk_opt: the same blocks, -0.21 dB for Q4_K and -0.43 for Q5_K
against the ggml encoder; --ggml-encoder takes ggml's).

--dense rq8: the dense matrices (the conv in_proj and out_proj, the
attention, the dense MLPs) in RQ8_0 too (the runtime rotates their x:
compile_qwen_step dense_rot), and the head as its own RQ8_0 matrix
(output.weight; the embeddings stay BF12, the rows the CPU reads).
tokenizer.json, tokenizer_config.json, and chat_template.jinja are copied
next to the file (the engine reads them there).

    python scripts/convert_lfm25_gguf.py ~/models/LFM2.5-24B-A2B \\
        ~/models/LFM2.5-24B-A2B-BF12-GGUF/LFM2.5-24B-A2B-bf12.gguf [--experts rq8]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.gguf import (BF12, BF16, F32, RQ4_K, RQ5_K, RQ6_K, RQ8_0, rotation_meta,  # noqa: E402
                                  tensor_bytes, write_gguf)


class Shards:
    """The tensors of the safetensors shards: name -> (path, dtype, shape,
    start, end); reads drop their pages from the page cache."""

    def __init__(self, path):
        where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.t = {}
        self.fds = {}
        for f in sorted(set(where.values())):
            p = os.path.join(path, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            hdr.pop("__metadata__", None)
            for k, v in hdr.items():
                s, e = v["data_offsets"]
                self.t[k] = (p, v["dtype"], tuple(v["shape"]), 8 + n + s, 8 + n + e)

    def read(self, name):
        p, dt, shape, s, e = self.t[name]
        if p not in self.fds:
            self.fds[p] = os.open(p, os.O_RDONLY)
        fd = self.fds[p]
        buf = bytearray(e - s)
        mv = memoryview(buf)
        got = 0
        while got < len(buf):
            got += os.preadv(fd, [mv[got:]], s + got)
        os.posix_fadvise(fd, s, e - s, os.POSIX_FADV_DONTNEED)
        a = np.frombuffer(buf, {"BF16": np.uint16, "F32": np.float32}[dt]).reshape(shape)
        return a, dt

    def f32(self, name):
        a, dt = self.read(name)
        return (a.astype(np.uint32) << 16).view(np.float32) if dt == "BF16" else a.astype(np.float32)

    def bf16(self, name):
        a, dt = self.read(name)
        assert dt == "BF16", (name, dt)
        return a


def rq8(a):
    """RQ8_0 blocks of bfloat16 rows a (uint16): Q8_0 of the rows rotated in
    each 32 values."""
    f = (a.astype(np.uint32) << 16).view(np.float32)
    return cops.kq_to_q8_0(cops.tq6_rotate(f), a.shape[-1])


def rq6(a):
    """RQ6_K blocks of bfloat16 rows a (uint16): Q6_K of the rows rotated in
    each 32 values (cols a multiple of 256)."""
    f = (a.astype(np.uint32) << 16).view(np.float32)
    return cops.kq_to_q6_k(cops.tq6_rotate(f), a.shape[-1])


GGML_ENCODER = False        # --ggml-encoder


def rqk(a, bits):
    """RQ4_K / RQ5_K blocks of bfloat16 rows a (uint16): Q4_K / Q5_K of the
    rows rotated in each 32 values (cols a multiple of 256)."""
    f = (a.astype(np.uint32) << 16).view(np.float32)
    x = cops.tq6_rotate(f)
    return cops.kq_to_qk(x, a.shape[-1], bits) if GGML_ENCODER else cops.kq_to_qk_opt(x, a.shape[-1], bits)


# the encoder and the ggml base type of each rotated expert type
XENC = {RQ8_0: (rq8, 8), RQ6_K: (rq6, 14), RQ5_K: (lambda a: rqk(a, 5), 13),
        RQ4_K: (lambda a: rqk(a, 4), 12)}


def rq8_db(a, t=RQ8_0):
    """The error of the rotated type t (RQ8_0, RQ6_K, RQ5_K, RQ4_K) of the rows
    a (uint16) in dB against them."""
    f = (a.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    enc, base = XENC[t]
    back = cops.tq6_rotate(cops.kq_rows(enc(a), base, a.shape[-1], np.arange(a.shape[0])),
                           inverse=True)
    return 10 * np.log10(((back - f) ** 2).sum() / (f ** 2).sum())


def bf12_or_none(a):
    """The BF12 rows of bfloat16 rows a (uint16, rows x cols), and the
    report; None for the rows when BF12 cannot hold them."""
    rows, rep = cops.kq_bf16_to_bf12(a, a.shape[-1])
    bad = rep["nonfinite"] > 0 or rep["largest"] > rep["rms"] * 2.0 ** -8
    return (None if bad else rows.reshape(-1)), rep


def tokenizer_meta(src):
    """The tokenizer.ggml keys of llama.cpp (gpt2 BPE, pre lfm2) from
    tokenizer.json, padded to the rows of the embeddings."""
    tj = json.load(open(os.path.join(src, "tokenizer.json")))
    cfg = json.load(open(os.path.join(src, "config.json")))
    n = int(cfg["vocab_size"])
    tokens = ["[PAD%d]" % i for i in range(n)]
    types = [5] * n                       # unused
    for t, i in tj["model"]["vocab"].items():
        tokens[i], types[i] = t, 1
    for a in tj["added_tokens"]:
        tokens[a["id"]], types[a["id"]] = a["content"], 3 if a.get("special") else 4
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    return [("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "lfm2"),
            ("tokenizer.ggml.tokens", tokens), ("tokenizer.ggml.token_type", types),
            ("tokenizer.ggml.merges", merges),
            ("tokenizer.ggml.bos_token_id", int(cfg["bos_token_id"])),
            ("tokenizer.ggml.eos_token_id", int(cfg["eos_token_id"])),
            ("tokenizer.ggml.padding_token_id", int(cfg["pad_token_id"])),
            ("tokenizer.ggml.add_bos_token", False),
            ("tokenizer.chat_template", open(os.path.join(src, "chat_template.jinja")).read())]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="the checkpoint directory")
    ap.add_argument("out", help="the GGUF file to write")
    ap.add_argument("--only", default=None, help="a test: only the tensors whose GGUF names match this "
                    "regular expression")
    ap.add_argument("--experts", choices=("bf12", "rq8", "rq6", "rq5", "rq4"), default="bf12",
                    help="the type of the routed experts")
    ap.add_argument("--ggml-encoder", action="store_true",
                    help="rq4/rq5 experts with ggml's encoder (not the better one of convert.c)")
    ap.add_argument("--dense", choices=("bf12", "rq8"), default="bf12",
                    help="the type of the dense matrices and of the head (the embeddings stay BF12)")
    args = ap.parse_args()
    global GGML_ENCODER
    GGML_ENCODER = args.ggml_encoder
    c = json.load(open(os.path.join(args.src, "config.json")))
    sh = Shards(args.src)
    L, E = c["num_hidden_layers"], c["num_experts"]
    H, I, dense = c["hidden_size"], c["moe_intermediate_size"], c["num_dense_layers"]
    types = c["layer_types"]
    P = "model."

    # (gguf name, checkpoint name or list of expert names, kind): kind "mat"
    # (BF12 / bf16), "f32", "conv" (float32, (H, 1, K) -> (H, K)), "exps" (a stack)
    plan = [("token_embd.weight", P + "embed_tokens.weight", "mat"),
            ("token_embd_norm.weight", P + "embedding_norm.weight", "f32")]
    if args.dense == "rq8":
        # the head as its own matrix (RQ8_0), the embeddings BF12
        plan.append(("output.weight", P + "embed_tokens.weight", "mat"))
    for i in range(L):
        b, p = "blk.%d." % i, P + "layers.%d." % i
        plan += [(b + "attn_norm.weight", p + "operator_norm.weight", "f32"),
                 (b + "ffn_norm.weight", p + "ffn_norm.weight", "f32")]
        if types[i] == "full_attention":
            a = p + "self_attn."
            plan += [(b + "attn_q.weight", a + "q_proj.weight", "mat"),
                     (b + "attn_k.weight", a + "k_proj.weight", "mat"),
                     (b + "attn_v.weight", a + "v_proj.weight", "mat"),
                     (b + "attn_output.weight", a + "out_proj.weight", "mat"),
                     (b + "attn_q_norm.weight", a + "q_layernorm.weight", "f32"),
                     (b + "attn_k_norm.weight", a + "k_layernorm.weight", "f32")]
        else:
            plan += [(b + "shortconv.in_proj.weight", p + "conv.in_proj.weight", "mat"),
                     (b + "shortconv.conv.weight", p + "conv.conv.weight", "conv"),
                     (b + "shortconv.out_proj.weight", p + "conv.out_proj.weight", "mat")]
        f = p + "feed_forward."
        if i < dense:
            plan += [(b + "ffn_gate.weight", f + "w1.weight", "mat"),
                     (b + "ffn_up.weight", f + "w3.weight", "mat"),
                     (b + "ffn_down.weight", f + "w2.weight", "mat")]
        else:
            plan += [(b + "ffn_gate_inp.weight", f + "gate.weight", "f32"),
                     (b + "exp_probs_b.bias", f + "expert_bias", "f32")]
            for g, w in (("gate", "w1"), ("up", "w3"), ("down", "w2")):
                plan.append((b + "ffn_%s_exps.weight" % g,
                             [f + "experts.%d.%s.weight" % (e, w) for e in range(E)], "exps"))
    if args.only:
        plan = [x for x in plan if re.search(args.only, x[0])]

    # each matrix first: its type depends on its values (BF12, or bfloat16
    # when BF12 cannot hold it); the rows are made again at the write
    t0 = time.time()
    kept, zeroed, nvals, worst = [], 0, 0, (0.0, "-")
    tensors = []
    xerr, derr = [], []
    XTYPE = {"rq8": RQ8_0, "rq6": RQ6_K, "rq5": RQ5_K, "rq4": RQ4_K}
    for gname, src, kind in plan:
        if kind == "exps" and args.experts != "bf12":
            rows, cols = sh.t[src[0]][2]
            dims = [cols, rows, len(src)]
            t = XTYPE[args.experts]
            if len(xerr) < 24 and int(gname.split(".")[1]) % 8 == 2:
                xerr.append(rq8_db(sh.bf16(src[len(xerr) * 7 % len(src)]), t))
        elif kind == "exps":
            p0, _dt, shape, _s, _e = sh.t[src[0]]
            rows, cols = shape
            dims = [cols, rows, len(src)]
            ok = True
            for nm in src:
                r, rep = bf12_or_none(sh.bf16(nm))
                zeroed += rep["zeroed"]
                nvals += rows * cols
                if rep["rms"] > 0 and rep["largest"] / rep["rms"] > worst[0]:
                    worst = (rep["largest"] / rep["rms"], nm)
                ok = ok and r is not None
            t = BF12 if ok else BF16
        elif kind == "mat" and args.dense == "rq8" and gname != "token_embd.weight":
            a = sh.bf16(src)
            dims = [a.shape[1], a.shape[0]]
            t = RQ8_0
            if len(derr) < 12 and (gname.startswith("blk.2.") or gname.startswith("blk.3.") or
                                   gname == "output.weight"):
                derr.append(rq8_db(a[:4096]))
        elif kind == "mat":
            a = sh.bf16(src)
            dims = [a.shape[1], a.shape[0]]
            r, rep = bf12_or_none(a)
            zeroed += rep["zeroed"]
            nvals += a.size
            if rep["rms"] > 0 and rep["largest"] / rep["rms"] > worst[0]:
                worst = (rep["largest"] / rep["rms"], src)
            t = BF12 if r is not None else BF16
        elif kind == "conv":
            shape = sh.t[src][2]                     # (H, 1, K)
            dims = [shape[2], shape[0]]
            t = F32
        else:
            shape = sh.t[src][2]
            dims = list(reversed(shape))
            t = F32
        if t == BF16:
            kept.append(gname)
        tensors.append((gname, dims, t, src, kind))
    print("BF12: %.2f G values, %d zeroed (%.5f%%), the largest zeroed %.2e of its RMS (%s); %d kept "
          "bfloat16 %s (%.0f s)" % (nvals / 1e9, zeroed, 100.0 * zeroed / max(nvals, 1), worst[0], worst[1],
                                     len(kept), kept, time.time() - t0), flush=True)

    def maker(src, kind, t):
        if kind == "exps" and t in XENC:
            enc = XENC[t][0]
            return lambda: (enc(sh.bf16(nm)) for nm in src)
        if kind == "exps":
            def make():
                for nm in src:
                    a = sh.bf16(nm)
                    yield bf12_or_none(a)[0] if t == BF12 else a
            return make
        if kind == "mat" and t == RQ8_0:
            return lambda: rq8(sh.bf16(src))
        if kind == "mat":
            return lambda: bf12_or_none(sh.bf16(src))[0] if t == BF12 else sh.bf16(src)
        if kind == "conv":
            return lambda: np.ascontiguousarray(sh.f32(src).reshape(sh.t[src][2][0], -1))
        return lambda: np.ascontiguousarray(sh.f32(src))

    out = [(g, d, t, maker(s, k, t)) for g, d, t, s, k in tensors]
    total = sum(tensor_bytes(d, t) for _g, d, t, _m in out)
    a = "lfm2moe."
    meta = [("general.architecture", "lfm2moe"),
            ("general.name", "LFM2.5-24B-A2B BF12%s%s (reclank)" % (
                {"rq8": ", experts RQ8_0", "rq6": ", experts RQ6_K", "rq5": ", experts RQ5_K",
                 "rq4": ", experts RQ4_K"}.get(args.experts, ""),
                ", dense and head RQ8_0" if args.dense == "rq8" else "")),
            ("general.source", "LiquidAI/LFM2.5-24B-A2B (original bfloat16)"),
            ("general.sampling.top_k", 50), ("general.sampling.temp", 0.1),
            ("general.sampling.penalty_repeat", 1.05),
            (a + "block_count", L),
            (a + "context_length", int(c["max_position_embeddings"])),
            (a + "embedding_length", H),
            (a + "feed_forward_length", int(c["intermediate_size"])),
            (a + "attention.head_count", int(c["num_attention_heads"])),
            (a + "attention.head_count_kv", [int(c["num_key_value_heads"]) if x == "full_attention" else 0
                                             for x in types]),
            (a + "rope.freq_base", float(c["rope_parameters"]["rope_theta"])),
            (a + "attention.layer_norm_rms_epsilon", float(c["norm_eps"])),
            (a + "expert_count", E), (a + "expert_used_count", int(c["num_experts_per_tok"])),
            (a + "expert_feed_forward_length", I),
            (a + "leading_dense_block_count", dense),
            (a + "expert_gating_func", 2),               # sigmoid (llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID)
            (a + "expert_weights_norm", bool(c["norm_topk_prob"])),
            (a + "expert_weights_scale", float(c["routed_scaling_factor"])),
            (a + "vocab_size", int(c["vocab_size"])),
            (a + "shortconv.l_cache", int(c["conv_L_cache"])),
            ("reclank.bf12.kept_bf16", ",".join(kept)), ("reclank.bf12.zeroed", int(zeroed))]
    if args.dense == "rq8":
        meta.append(("reclank.quant_error.dense", float(np.mean(derr))))
        print("RQ8_0 dense and head: %.2f dB against bfloat16 (a sample of %d)" % (np.mean(derr), len(derr)),
              flush=True)
    if args.experts != "bf12" or args.dense == "rq8":
        meta += list(rotation_meta().items())
    if args.experts != "bf12":
        meta.append(("reclank.quant_error.experts", float(np.mean(xerr))))
        print("%s experts: %.2f dB against bfloat16 (a sample of %d)" % (
            args.experts.upper(), np.mean(xerr), len(xerr)), flush=True)
    meta += tokenizer_meta(args.src)
    print("%d tensors, %.1f GB -> %s" % (len(out), total / 1e9, args.out), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    done = [0]

    def progress(name, n):
        done[0] += n
        if name.endswith("ffn_down_exps.weight") or name == out[-1][0]:
            el = time.time() - t0
            print("  %s: %.1f of %.1f GB, %.0f s" % (name, done[0] / 1e9, total / 1e9, el), flush=True)

    write_gguf(args.out + ".part", meta, out, progress=progress)
    os.replace(args.out + ".part", args.out)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json",
              "generation_config.json"):
        shutil.copy(os.path.join(args.src, f), os.path.join(os.path.dirname(os.path.abspath(args.out)), f))
    print("written %s in %.0f s" % (args.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
