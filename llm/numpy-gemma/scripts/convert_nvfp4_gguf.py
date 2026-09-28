#!/usr/bin/env python3
"""Convert the NVFP4 checkpoint of Qwen3.8-Flash-Next (NVIDIA ModelOpt) to one
GGUF file for this runtime.

The runtime can read the checkpoint itself (np_gemma/st_qwen4.py), but then
it repacks the experts at each start into 68 GB of memory. This file holds
the same forms, so the runtime maps it and the pages come and go:

- the names and the metadata of the GGUF files of llama.cpp (qwen4exp), with
  the MTP layer as blk.48; the value heads of the DeltaNet in the tiled
  order of llama.cpp (as the converter of llama.cpp does);
- the routed experts in groups of 16 rows (KQ_NVX, type 53: NVFP4 as it
  is; --experts nv4: the rows of KQ_NV4, type 51);
- the n-gram table as rows of E4M3 codes (type 52) and the tensor
  per_layer_token_embd.scale (FP8 as it is);
- the bfloat16 matrices as bfloat16 (--dense bf16, the default: the runtime
  can requantize them for a small GPU), or as Q8_0 (--dense q8);
- the MTP experts (FP8 blocks) as Q8_0; the small matrices and the norms as
  float32 (with the 1 of the norms).

The types 51, 52, and 53 are of this runtime only: llama.cpp cannot read the file.

    python scripts/convert_nvfp4_gguf.py models/Qwen3.8-Flash-Next-NVFP4 \\
        models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import cops  # noqa: E402
from np_gemma.gguf import BF16, E4M3_ROWS, F32, Q8_0, tensor_bytes, write_gguf  # noqa: E402
from np_gemma.st_qwen4 import NVFP4Source  # noqa: E402


def reorder(a, dim, hk, rep, hd):
    """The heads of axis dim from grouped by key head to the tiled order of
    llama.cpp (_reorder_v_heads of its converter)."""
    a = np.moveaxis(a, dim, 0)
    sh = a.shape
    a = a.reshape(hk, rep, hd, *sh[1:]).swapaxes(0, 1).reshape(sh)
    return np.ascontiguousarray(np.moveaxis(a, 0, dim))


def metadata(cfg, src):
    a = "qwen4exp."
    m = [("general.architecture", "qwen4exp"),
         ("general.name", "Qwen3.8-Flash-Next NVFP4 (np_gemma)"),
         ("general.source", "nvidia/Qwen3.8-Flash-Next-NVFP4"),
         (a + "block_count", cfg.num_hidden_layers),
         (a + "nextn_predict_layers", 1 if "blk.%d.nextn.eh_proj.weight" % cfg.num_hidden_layers
          in src.tensors else 0),
         (a + "full_attention_interval", int(src.hf.get("full_attention_interval", 4))),
         (a + "embedding_length", cfg.hidden_size),
         (a + "attention.layer_norm_rms_epsilon", float(cfg.rms_norm_eps)),
         (a + "attention.head_count", cfg.num_heads),
         (a + "attention.head_count_kv", cfg.num_kv_heads),
         (a + "attention.key_length", cfg.head_dim),
         (a + "attention.value_length", cfg.head_dim),
         (a + "rope.freq_base", float(cfg.rope_theta)),
         (a + "rope.dimension_count", cfg.rotary_dim),
         (a + "ssm.group_count", cfg.lin_k_heads),
         (a + "ssm.time_step_rank", cfg.lin_v_heads),
         (a + "ssm.state_size", cfg.lin_k_dim),
         (a + "ssm.inner_size", cfg.lin_v_heads * cfg.lin_v_dim),
         (a + "ssm.conv_kernel", cfg.conv_kernel),
         (a + "expert_count", cfg.num_experts),
         (a + "expert_used_count", cfg.top_k),
         (a + "expert_feed_forward_length", cfg.moe_inter),
         (a + "expert_shared_feed_forward_length", cfg.shared_inter),
         (a + "hyper_connection.count", cfg.hc_count),
         (a + "hyper_connection.low_rank", cfg.hc_lowrank),
         (a + "ple.layers", list(cfg.ple_layers)),
         (a + "ple.ngram_size", cfg.ple_ngram),
         (a + "ple.heads_per_ngram", cfg.ple_heads_per_ngram),
         (a + "ple.conv_kernel", cfg.ple_conv_kernel),
         (a + "ple.eos_token_id", cfg.ple_eos),
         (a + "embedding_length_per_layer_input", cfg.ple_row),
         (a + "ple.layer_multipliers", list(cfg.ple_multipliers)),
         (a + "ple.head_offsets", list(cfg.ple_offsets)),
         (a + "ple.head_vocab_sizes", list(cfg.ple_sizes)),
         (a + "attention.indexer.head_count", cfg.indexer_heads),
         (a + "attention.indexer.key_length", cfg.indexer_dim),
         (a + "attention.indexer.top_k", cfg.indexer_top_k),
         (a + "attention.compress_ratios", list(cfg.compress_ratios))]
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="the checkpoint directory")
    ap.add_argument("out", help="the GGUF file to write")
    ap.add_argument("--dense", choices=("bf16", "q8"), default="bf16")
    ap.add_argument("--experts", choices=("nvx", "nv4"), default="nvx")
    args = ap.parse_args()
    src = NVFP4Source(args.src, experts=args.experts)
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

    def maker(gname):
        kind, s = src._map[gname]
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

    tensors = []
    for gname, (kind, s) in src._map.items():
        dims, t, _ = src.tensors[gname]
        if kind in ("dense", "ehproj") and args.dense == "q8":
            t = Q8_0
        if kind == "ple":
            shards = src._ple_shards
            tensors.append((gname, dims, E4M3_ROWS,
                            lambda shards=shards: [src._get(n, dtype=None) for n in shards]))
            tensors.append((gname.replace(".weight", ".scale"), (1,), F32,
                            lambda: np.array([src._ple_scale()], np.float32)))
            continue
        tensors.append((gname, dims, t, maker(gname)))
    total = sum(tensor_bytes(d, t) for _n, d, t, _m in tensors)
    print("%d tensors, %.1f GB, dense %s -> %s" % (len(tensors), total / 1e9, args.dense, args.out))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    done = [0, 0.0]

    def progress(name, n):
        done[0] += n
        if done[0] - done[1] >= 5e9 or done[0] == total:
            done[1] = done[0]
            dt = time.time() - t0
            print("  %6.1f of %.1f GB, %5.0f s, %.0f MB/s (%s)" % (
                done[0] / 1e9, total / 1e9, dt, done[0] / dt / 1e6, name), flush=True)

    tmp = args.out + ".part"
    write_gguf(tmp, metadata(cfg, src), tensors, progress=progress)
    os.replace(tmp, args.out)
    tok = os.path.join(args.src, "tokenizer.json")
    if os.path.exists(tok):
        shutil.copy(tok, os.path.join(os.path.dirname(os.path.abspath(args.out)), "tokenizer.json"))
    print("done: %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
