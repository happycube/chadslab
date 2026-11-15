#!/usr/bin/env python3
"""Convert google/diffusiongemma-26B-A4B-it (the bfloat16 safetensors) to one GGUF file with the
matrices in BF12 (the bfloat16 values in 12.25 bits, plan-scripts/BF12_PLAN.md):
docs/DIFFUSIONGEMMA_PLAN.md section 4.

- The names and the metadata are llama.cpp's diffusion-gemma (as in unsloth's GGUFs of the same
  checkpoint): the metadata, the tokenizer and rope_freqs.weight are copied from --template (a
  GGUF of the checkpoint, only its header and that one small tensor are read).
- The matrices (attention, the shared MLP, the experts as their stacks, the self-conditioning MLP,
  the token table) are BF12; a matrix that BF12 cannot hold (a zeroed value above 2^-8 of its RMS,
  or an Inf or NaN) stays bfloat16. The norms, the router, the scales and the layer scalars (the
  decoder's layer_output_scale, the encoder's enc_layer_output_scale) are float32.
- Two passes: the BF12 report of every matrix first (the types go in the header), then the write,
  converting each tensor again when the writer comes to it (one tensor in memory).
- The vision tower is not in the file.

    python scripts/convert_dgemma_bf12.py MODELS/diffusiongemma-26B-A4B-it \\
        --template MODELS/diffusiongemma-26B-A4B-it-GGUF/diffusiongemma-26B-A4B-it-Q4_K_M.gguf \\
        MODELS/diffusiongemma-26B-A4B-it-BF12-GGUF/diffusiongemma-26B-A4B-it-BF12.gguf
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.gguf import BF12, BF16, F32, GGUF, tensor_bytes, write_gguf  # noqa: E402
from reclank.formats.safetensors import Shards  # noqa: E402

DEC = "model.decoder."
ENC = "model.encoder.language_model."
# GGUF suffix -> (the checkpoint's name under model.decoder.layers.N., a matrix)
LAYER = {
    "attn_norm.weight": ("input_layernorm.weight", False),
    "attn_q.weight": ("self_attn.q_proj.weight", True),
    "attn_k.weight": ("self_attn.k_proj.weight", True),
    "attn_v.weight": ("self_attn.v_proj.weight", True),
    "attn_output.weight": ("self_attn.o_proj.weight", True),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", False),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", False),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", False),
    "ffn_norm.weight": ("pre_feedforward_layernorm.weight", False),
    "ffn_gate.weight": ("mlp.gate_proj.weight", True),
    "ffn_up.weight": ("mlp.up_proj.weight", True),
    "ffn_down.weight": ("mlp.down_proj.weight", True),
    "post_ffw_norm.weight": ("post_feedforward_layernorm.weight", False),
    "post_ffw_norm_1.weight": ("post_feedforward_layernorm_1.weight", False),
    "post_ffw_norm_2.weight": ("post_feedforward_layernorm_2.weight", False),
    "pre_ffw_norm_2.weight": ("pre_feedforward_layernorm_2.weight", False),
    "ffn_gate_inp.weight": ("router.proj.weight", False),
    "ffn_gate_inp.scale": ("router.scale", False),
    "ffn_down_exps.scale": ("router.per_expert_scale", False),
    "ffn_gate_up_exps.weight": ("experts.gate_up_proj", True),
    "ffn_down_exps.weight": ("experts.down_proj", True),
    "layer_output_scale.weight": ("layer_scalar", False),
}
GLOBAL = {
    "token_embd.weight": (DEC + "embed_tokens.weight", True),
    "output_norm.weight": (DEC + "norm.weight", False),
    "self_cond_pre_norm.weight": (DEC + "self_conditioning.pre_norm.weight", False),
    "self_cond_gate.weight": (DEC + "self_conditioning.gate_proj.weight", True),
    "self_cond_up.weight": (DEC + "self_conditioning.up_proj.weight", True),
    "self_cond_down.weight": (DEC + "self_conditioning.down_proj.weight", True),
}


def plain(v):
    """A metadata value for write_gguf (its arrays as lists)."""
    if isinstance(v, np.ndarray):
        return [float(x) for x in v] if v.dtype.kind == "f" else [int(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--template", required=True, help="a GGUF of the checkpoint (its metadata)")
    a = ap.parse_args()
    st = Shards(a.src)
    tpl = GGUF(a.template)
    L = json.load(open(os.path.join(a.src, "config.json")))["text_config"]["num_hidden_layers"]

    # the tensors: GGUF name -> (checkpoint name, matrix)
    names = {}
    for g, (n, m) in GLOBAL.items():
        names[g] = (n, m)
    for i in range(L):
        for suf, (n, m) in LAYER.items():
            src = DEC + "layers.%d.%s" % (i, n)
            if src in st.where:
                names["blk.%d.%s" % (i, suf)] = (src, m)
        names["blk.%d.enc_layer_output_scale.weight" % i] = (ENC + "layers.%d.layer_scalar" % i, False)
    missing = [g for g in tpl.tensors if g not in names and g != "rope_freqs.weight"]
    assert not missing, "the template has tensors this converter does not make: %s" % missing[:8]

    def bits(n):
        """The bfloat16 bits of a checkpoint tensor as rows (uint16, rows x cols)."""
        a = st.get_bf16(n)
        return np.ascontiguousarray(a).view(np.uint16).reshape(-1, a.shape[-1])

    # pass 1: the BF12 report of each matrix
    t0 = time.time()
    kind, report = {}, {}
    for g, (n, m) in names.items():
        if not m:
            kind[g] = F32
            continue
        b = bits(n)
        _rows, r = cops.kq_bf16_to_bf12(b, b.shape[1])
        report[g] = r
        bad = r["nonfinite"] > 0 or r["largest"] > r["rms"] * 2.0 ** -8
        kind[g] = BF16 if bad else BF12
    kept = [g for g in kind if kind[g] == BF16]
    zeroed = sum(r["zeroed"] for r in report.values())
    vals = sum(np.prod(st.shape(names[g][0])) for g in report)
    print("pass 1: %d matrices (%.2f G values) %.0f s: %d zeroed values (%.5f%%), %d kept bfloat16 %s"
          % (len(report), vals / 1e9, time.time() - t0, zeroed, 100.0 * zeroed / vals, len(kept), kept),
          flush=True)

    def dims_of(g):
        if g == "rope_freqs.weight":
            return tpl.tensors[g][0]
        return tuple(reversed(st.shape(names[g][0])))

    def maker(g):
        if g == "rope_freqs.weight":
            return lambda: tpl.dequant(g).astype(np.float32)
        n, m = names[g]
        if kind[g] == F32:
            return lambda: (ops_f32(st, n)).reshape(-1)
        if kind[g] == BF16:
            return lambda: bits(n).reshape(-1)
        return lambda: cops.kq_bf16_to_bf12(bits(n), st.shape(n)[-1])[0].reshape(-1)

    order = list(tpl.tensors)          # the template's order (all of its names are made)
    tensors = [(g, dims_of(g), kind.get(g, F32), maker(g)) for g in order]
    total = sum(tensor_bytes(d, t) for _g, d, t, _m in tensors)
    print("%d tensors, %.2f GB -> %s" % (len(tensors), total / 1e9, a.out), flush=True)

    meta = {}
    for k, v in tpl.meta.items():
        if k in ("general.alignment", "general.file_type", "general.quantization_version"):
            continue
        if isinstance(v, dict) and "__array__" in v:
            v = tpl.meta_strings(k)
        meta[k] = plain(v)
    meta["general.name"] = "DiffusionGemma 26B-A4B it, BF12 (reclank)"
    meta["general.source"] = "google/diffusiongemma-26B-A4B-it (bfloat16 safetensors)"
    meta["reclank.bf12.kept_bf16"] = ",".join(kept)
    meta["reclank.bf12.zeroed"] = int(zeroed)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    tmp = a.out + ".part"
    done = [0, time.time()]

    def progress(name, n):
        done[0] += 1
        if done[0] % 100 == 0:
            print("  %d/%d tensors, %.0f s" % (done[0], len(tensors), time.time() - done[1]), flush=True)

    write_gguf(tmp, list(meta.items()), tensors, progress=progress)
    os.replace(tmp, a.out)
    json.dump({g: {k: v for k, v in r.items() if k != "worst"} for g, r in report.items()},
              open(a.out + ".bf12.json", "w"), indent=0)
    # the small files of the checkpoint beside it (the runtime reads them)
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "chat_template.jinja"):
        shutil.copy(os.path.join(a.src, f), os.path.join(os.path.dirname(os.path.abspath(a.out)), f))
    print("done %.0f s" % (time.time() - t0), flush=True)


def ops_f32(st, n):
    if st.dtype(n) == "BF16":
        return (st.get_bf16(n).astype(np.uint32) << 16).view(np.float32)
    return st.get(n).astype(np.float32)


if __name__ == "__main__":
    main()
