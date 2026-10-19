#!/usr/bin/env python3
"""Convert the bfloat16 checkpoint of a dense model of the Qwen3.5
architecture (qwen3_5: Qwen3.8-27B) to one GGUF file for this runtime: the
matrices in BF12 (the bfloat16 values in 12.25 bits, plan-scripts/BF12_PLAN.md),
the norms, the DeltaNet vectors and the convolution weights float32. The text
model only (no vision tower, no MTP layer).

The names, the metadata, and the forms of the tensors are those of
llama.cpp's qwen35 (its converter, as the qwen35moe files of Qwen3.6 show:
scripts/convert_q36_dense.py checks the same rules against such a file):

- the RMSNorm weights (attn_norm, post_attention_norm, output_norm,
  attn_q_norm, attn_k_norm) are 1 + w of the checkpoint (transformers adds
  the 1 at run time); ssm_norm (the gated norm of the DeltaNet) is w;
- ssm_a is -exp(A_log);
- the value heads of the DeltaNet in the tiled order (value head j reads
  key head j % kh; the checkpoint groups them, v_order): the rows of
  attn_gate, ssm_alpha, ssm_beta, the value rows of attn_qkv and of
  ssm_conv1d, the values of ssm_a and ssm_dt.bias, and the columns of
  ssm_out;
- ssm_conv1d is (channels, kernel); attn_q holds the query and the output
  gate of each head, as q_proj does.

    blk.N.attn_norm, post_attention_norm             the two norms of a layer
    blk.N.attn_qkv, attn_gate, ssm_alpha, ssm_beta,  the DeltaNet layers
        ssm_a, ssm_dt.bias, ssm_conv1d, ssm_norm, ssm_out
    blk.N.attn_q, _k, _v, _output, attn_q_norm, attn_k_norm   the full layers
    blk.N.ffn_gate, ffn_up, ffn_down                 the SwiGLU MLP

A matrix that BF12 cannot hold (a zeroed value above 2^-8 of its RMS, or an
Inf or NaN) stays bfloat16, as convert_q8_gguf.py does. tokenizer.json,
tokenizer_config.json, chat_template.jinja, config.json, and
generation_config.json are copied next to the file.

    python scripts/convert_qwen35_dense_gguf.py ~/models/Qwen3.8-27B \\
        ~/models/Qwen3.8-27B-BF12-GGUF/Qwen3.8-27B-bf12.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from convert_lfm25_gguf import Shards, bf12_or_none  # noqa: E402
from reclank.formats.gguf import BF12, BF16, F32, tensor_bytes, write_gguf  # noqa: E402


def v_order(tc):
    """The checkpoint value head of each value head of the GGUF (the tiled
    order), and the value head dim."""
    vh, kh = tc["linear_num_value_heads"], tc["linear_num_key_heads"]
    rep = vh // kh
    return np.array([(j % kh) * rep + j // kh for j in range(vh)]), tc["linear_value_head_dim"]


def tokenizer_meta(src, n):
    """The tokenizer.ggml keys of llama.cpp (gpt2 BPE, pre qwen35) from
    tokenizer.json, padded to the n rows of the embeddings."""
    tj = json.load(open(os.path.join(src, "tokenizer.json")))
    tokens = ["[PAD%d]" % i for i in range(n)]
    types = [5] * n                       # unused
    for t, i in tj["model"]["vocab"].items():
        tokens[i], types[i] = t, 1
    for a in tj["added_tokens"]:
        tokens[a["id"]], types[a["id"]] = a["content"], 3 if a.get("special") else 4
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    sp = {a["content"]: a["id"] for a in tj["added_tokens"]}
    return [("tokenizer.ggml.model", "gpt2"), ("tokenizer.ggml.pre", "qwen35"),
            ("tokenizer.ggml.tokens", tokens), ("tokenizer.ggml.token_type", types),
            ("tokenizer.ggml.merges", merges),
            ("tokenizer.ggml.eos_token_id", int(sp["<|im_end|>"])),
            ("tokenizer.ggml.padding_token_id", int(sp["<|endoftext|>"])),
            ("tokenizer.ggml.add_bos_token", False),
            ("tokenizer.chat_template", open(os.path.join(src, "chat_template.jinja")).read())]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="the checkpoint directory")
    ap.add_argument("out", help="the GGUF file to write")
    ap.add_argument("--only", default=None, help="a test: only the tensors whose GGUF names match this "
                    "regular expression")
    args = ap.parse_args()
    c = json.load(open(os.path.join(args.src, "config.json")))
    tc = c.get("text_config", c)
    sh = Shards(args.src)
    L = tc["num_hidden_layers"]
    types = tc["layer_types"]
    perm, vd = v_order(tc)
    vh = len(perm)
    P = "model.language_model."

    def vrows(w):
        """The rows of whole value heads in the tiled order."""
        return w.reshape(vh, -1, *w.shape[1:])[perm].reshape(w.shape)

    def qkv(w):
        v0 = w.shape[0] - vh * vd
        return np.concatenate([w[:v0], vrows(w[v0:])])

    def vcols(w):
        return w.reshape(w.shape[0], vh, vd)[:, perm].reshape(w.shape)

    plus1 = lambda w: w + 1.0  # noqa: E731
    # (gguf name, checkpoint name, kind, transform): kind "mat" (BF12 /
    # bfloat16; the transform on the uint16 rows) or "f32" (on float32)
    plan = [("token_embd.weight", P + "embed_tokens.weight", "mat", None),
            ("output_norm.weight", P + "norm.weight", "f32", plus1),
            ("output.weight", "lm_head.weight", "mat", None)]
    for i in range(L):
        b, p = "blk.%d." % i, P + "layers.%d." % i
        plan += [(b + "attn_norm.weight", p + "input_layernorm.weight", "f32", plus1),
                 (b + "post_attention_norm.weight", p + "post_attention_layernorm.weight", "f32", plus1)]
        if types[i] == "full_attention":
            a = p + "self_attn."
            plan += [(b + "attn_q.weight", a + "q_proj.weight", "mat", None),
                     (b + "attn_k.weight", a + "k_proj.weight", "mat", None),
                     (b + "attn_v.weight", a + "v_proj.weight", "mat", None),
                     (b + "attn_output.weight", a + "o_proj.weight", "mat", None),
                     (b + "attn_q_norm.weight", a + "q_norm.weight", "f32", plus1),
                     (b + "attn_k_norm.weight", a + "k_norm.weight", "f32", plus1)]
        else:
            a = p + "linear_attn."
            plan += [(b + "attn_qkv.weight", a + "in_proj_qkv.weight", "mat", qkv),
                     (b + "attn_gate.weight", a + "in_proj_z.weight", "mat", vrows),
                     (b + "ssm_alpha.weight", a + "in_proj_a.weight", "f32", vrows),
                     (b + "ssm_beta.weight", a + "in_proj_b.weight", "f32", vrows),
                     (b + "ssm_a", a + "A_log", "f32", lambda w: -np.exp(w)[perm]),
                     (b + "ssm_dt.bias", a + "dt_bias", "f32", lambda w: w[perm]),
                     (b + "ssm_conv1d.weight", a + "conv1d.weight", "f32",
                      lambda w: qkv(w.reshape(w.shape[0], -1))),
                     (b + "ssm_norm.weight", a + "norm.weight", "f32", None),
                     (b + "ssm_out.weight", a + "out_proj.weight", "mat", vcols)]
        m = p + "mlp."
        plan += [(b + "ffn_gate.weight", m + "gate_proj.weight", "mat", None),
                 (b + "ffn_up.weight", m + "up_proj.weight", "mat", None),
                 (b + "ffn_down.weight", m + "down_proj.weight", "mat", None)]
    left = set(sh.t) - {x[1] for x in plan}
    left = [n for n in left if not n.startswith(("mtp.", "model.visual."))]
    assert not left, left[:5]
    if args.only:
        plan = [x for x in plan if re.search(args.only, x[0])]

    # each matrix first: its type depends on its values (BF12, or bfloat16
    # when BF12 cannot hold it); the rows are made again at the write
    t0 = time.time()
    kept, zeroed, nvals, worst = [], 0, 0, (0.0, "-")
    tensors = []
    for gname, src, kind, f in plan:
        shape = sh.t[src][2]
        if kind == "mat":
            a = sh.bf16(src)
            r, rep = bf12_or_none(a)
            zeroed += rep["zeroed"]
            nvals += a.size
            if rep["rms"] > 0 and rep["largest"] / rep["rms"] > worst[0]:
                worst = (rep["largest"] / rep["rms"], src)
            t = BF12 if r is not None else BF16
            dims = [shape[1], shape[0]]
        else:
            dims = [shape[2], shape[0]] if len(shape) == 3 else list(reversed(shape))
            t = F32
        if t == BF16:
            kept.append(gname)
        tensors.append((gname, dims, t, src, kind, f))
    print("BF12: %.2f G values, %d zeroed (%.5f%%), the largest zeroed %.2e of its RMS (%s); %d kept "
          "bfloat16 %s (%.0f s)" % (nvals / 1e9, zeroed, 100.0 * zeroed / max(nvals, 1), worst[0], worst[1],
                                     len(kept), kept, time.time() - t0), flush=True)

    def maker(src, kind, t, f):
        f = f or (lambda w: w)
        if kind == "mat":
            def make():
                a = np.ascontiguousarray(f(sh.bf16(src)))
                return bf12_or_none(a)[0] if t == BF12 else a
            return make
        return lambda: np.ascontiguousarray(f(sh.f32(src)), dtype=np.float32)

    out = [(g, d, t, maker(s, k, t, f)) for g, d, t, s, k, f in tensors]
    total = sum(tensor_bytes(d, t) for _g, d, t, _m in out)
    rp = tc["rope_parameters"]
    hd = int(tc["head_dim"])
    every = int(tc["full_attention_interval"])
    assert all((x == "full_attention") == ((i + 1) % every == 0) for i, x in enumerate(types))
    a = "qwen35."
    name = os.path.basename(os.path.normpath(args.src))
    gen = json.load(open(os.path.join(args.src, "generation_config.json")))
    meta = [("general.architecture", "qwen35"),
            ("general.name", "%s BF12 (reclank)" % name),
            ("general.source", "Qwen/%s (original bfloat16)" % name),
            ("general.sampling.top_k", int(gen.get("top_k", 20))),
            ("general.sampling.top_p", float(gen.get("top_p", 0.95))),
            ("general.sampling.temp", float(gen.get("temperature", 1.0))),
            (a + "block_count", L),
            (a + "context_length", int(tc["max_position_embeddings"])),
            (a + "embedding_length", int(tc["hidden_size"])),
            (a + "feed_forward_length", int(tc["intermediate_size"])),
            (a + "attention.head_count", int(tc["num_attention_heads"])),
            (a + "attention.head_count_kv", int(tc["num_key_value_heads"])),
            (a + "rope.dimension_sections", [int(v) for v in rp["mrope_section"]] + [0]),
            (a + "rope.freq_base", float(rp["rope_theta"])),
            (a + "attention.layer_norm_rms_epsilon", float(tc["rms_norm_eps"])),
            (a + "attention.key_length", hd),
            (a + "attention.value_length", hd),
            (a + "ssm.conv_kernel", int(tc["linear_conv_kernel_dim"])),
            (a + "ssm.state_size", int(tc["linear_key_head_dim"])),
            (a + "ssm.group_count", int(tc["linear_num_key_heads"])),
            (a + "ssm.time_step_rank", vh),
            (a + "ssm.inner_size", vh * vd),
            (a + "full_attention_interval", every),
            (a + "rope.dimension_count", int(hd * rp["partial_rotary_factor"])),
            ("reclank.bf12.kept_bf16", ",".join(kept)), ("reclank.bf12.zeroed", int(zeroed))]
    meta += tokenizer_meta(args.src, int(tc["vocab_size"]))
    print("%d tensors, %.1f GB -> %s" % (len(out), total / 1e9, args.out), flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    done = [0]

    def progress(gname, n):
        done[0] += n
        if gname.endswith("ffn_down.weight") and int(gname.split(".")[1]) % 8 == 7 or gname == out[-1][0]:
            print("  %s: %.1f of %.1f GB, %.0f s" % (gname, done[0] / 1e9, total / 1e9, time.time() - t0),
                  flush=True)

    write_gguf(args.out + ".part", meta, out, progress=progress)
    os.replace(args.out + ".part", args.out)
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json",
              "generation_config.json"):
        shutil.copy(os.path.join(args.src, f), os.path.join(os.path.dirname(os.path.abspath(args.out)), f))
    print("written %s in %.0f s" % (args.out, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
