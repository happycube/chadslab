#!/usr/bin/env python3
"""Convert a DSpark drafter of Qwen3.8-27B (speculators' DSparkDraftModel: RedHatAI/Qwen3.8-27B-speculator.dspark,
a directory of config.json and model.safetensors in bfloat16) to the GGUF that reclank/models/qwen/dspark.py
reads: the same tensor names, its config as dspark.* metadata, the matrices of the layers and fc in the type
asked (BF12: the bfloat16 values in 12.25 bits, a matrix BF12 cannot hold stays bfloat16; Q4_K or Q6_K: ggml's
own quantizer, ggml_quantize_chunk of a libggml-base the --ggml path names, without an importance matrix), the
Markov head's tables in bfloat16 and the vectors in float32. One tensor in memory at a time.

    python scripts/convert_dspark_gguf.py MODELS/Qwen3.8-27B-speculator.dspark \\
        MODELS/Qwen3.8-27B-speculator.dspark/Qwen3.8-27B-DSpark-Q4_K.gguf --type q4_k --ggml LIB/libggml-base.so
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import struct
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.formats.gguf import BF12, BF16, F32, tensor_bytes, write_gguf  # noqa: E402

Q4_K, Q6_K = 12, 14
TYPES = {"bf12": BF12, "q4_k": Q4_K, "q6_k": Q6_K}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the drafter's directory (config.json, model.safetensors)")
    ap.add_argument("out")
    ap.add_argument("--type", default="q4_k", choices=sorted(TYPES))
    ap.add_argument("--ggml", default=None, help="libggml-base.so (Q4_K, Q6_K)")
    a = ap.parse_args()
    t0 = time.time()
    c = json.load(open(os.path.join(a.src, "config.json")))
    tc = c["transformer_layer_config"]
    assert c.get("speculators_model_type") == "dspark", "not a speculators DSpark drafter"
    p = os.path.join(a.src, "model.safetensors")
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    mm = np.memmap(p, np.uint8, "r")

    def bits(name):
        h = hdr[name]
        s, e = h["data_offsets"]
        assert h["dtype"] == "BF16", name
        return np.asarray(mm[8 + n + s:8 + n + e]).view(np.uint16).reshape(h["shape"])

    want = TYPES[a.type]
    if want in (Q4_K, Q6_K):
        assert a.ggml, "--ggml: the libggml-base.so whose quantizer makes %s" % a.type
        L = ctypes.CDLL(a.ggml)
        L.ggml_quantize_chunk.restype = ctypes.c_size_t
        L.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p]

        def quant(x):
            rows, cols = x.shape
            out = np.zeros(tensor_bytes((cols, rows), want), np.uint8)
            got = L.ggml_quantize_chunk(want, np.ascontiguousarray(x, np.float32).ctypes.data, out.ctypes.data,
                                        0, rows, cols, None)
            assert got == out.nbytes, (got, out.nbytes)
            return out

    def kind(name):
        shape = hdr[name]["shape"]
        if len(shape) < 2 or shape[0] == 1:
            return F32
        if name.startswith("markov_head."):
            return BF16
        return want

    tensors, kept = [], []
    for name in sorted(hdr):
        shape = hdr[name]["shape"]
        t = kind(name)
        if t == BF12:
            r = cops.kq_bf16_to_bf12(np.ascontiguousarray(bits(name)), int(shape[-1]))[1]
            if r["nonfinite"] > 0 or r["largest"] > r["rms"] * 2.0 ** -8:
                t = BF16
                kept.append(name)
        dims = tuple(int(d) for d in reversed(shape)) if len(shape) >= 2 and shape[0] != 1 else (int(np.prod(shape)),)

        def make(name=name, t=t, shape=shape):
            b = np.ascontiguousarray(bits(name))
            if t == F32:
                return (b.astype(np.uint32) << 16).view(np.float32).reshape(-1)
            if t == BF16:
                return b.reshape(-1)
            if t == BF12:
                return cops.kq_bf16_to_bf12(b, int(shape[-1]))[0].reshape(-1)
            return quant((b.astype(np.uint32) << 16).view(np.float32).reshape(shape))

        tensors.append((name, dims, t, make))
    meta = [
        ("general.architecture", "dspark"),
        ("general.name", "Qwen3.8-27B DSpark (%s), %s (reclank)" % (os.path.basename(os.path.abspath(a.src)), a.type)),
        ("general.source", os.path.basename(os.path.abspath(a.src))),
        ("dspark.verifier", str(c["speculators_config"]["verifier"]["name_or_path"])),
        ("dspark.target_layers", [int(v) for v in c["aux_hidden_state_layer_ids"]]),
        ("dspark.block_size", int(c["block_size"])),
        ("dspark.mask_token_id", int(c["mask_token_id"])),
        ("dspark.vocab_size", int(c["draft_vocab_size"])),
        ("dspark.markov_rank", int(c["markov_rank"])),
        ("dspark.embedding_length", int(tc["hidden_size"])),
        ("dspark.block_count", int(tc["num_hidden_layers"])),
        ("dspark.feed_forward_length", int(tc["intermediate_size"])),
        ("dspark.attention.head_count", int(tc["num_attention_heads"])),
        ("dspark.attention.head_count_kv", int(tc["num_key_value_heads"])),
        ("dspark.attention.key_length", int(tc["head_dim"])),
        ("dspark.attention.layer_norm_rms_epsilon", float(tc["rms_norm_eps"])),
        ("dspark.attention.sliding_window", int(tc["sliding_window"])),
        ("dspark.rope.freq_base", float(tc["rope_parameters"]["rope_theta"])),
        ("reclank.dspark.kept_bf16", ",".join(kept)),
    ]
    assert c["markov_head_type"] == "vanilla" and c["enable_confidence_head"] and c["confidence_head_with_markov"]
    assert c["sample_from_anchor"] and not c["sliding_window_non_causal"]
    assert float(tc["rope_parameters"].get("partial_rotary_factor", 1.0)) == 1.0
    total = sum(tensor_bytes(d, t) for _n, d, t, _m in tensors)
    print("%d tensors, %.2f GB -> %s (bfloat16 kept: %s)" % (len(tensors), total / 1e9, a.out, kept), flush=True)
    tmp = a.out + ".part"
    write_gguf(tmp, meta, tensors)
    os.replace(tmp, a.out)
    print("done %.0f s" % (time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
