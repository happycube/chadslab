#!/usr/bin/env python3
"""Write DiffusionGemma's vision tower as an mmproj GGUF (--mmproj of the dgemma engine).

DiffusionGemma's checkpoint holds its own vision tower (model.encoder.vision_tower,
model.encoder.embed_vision). The tower is the Gemma 4 26B's, but the projection into the text
model (embed_vision.embedding_projection) is its own: against the Gemma 4 26B mmproj's
mm.input_projection it has 1.53 x the norm and 20% of it left after the best scale of each row.
So the Gemma 4 mmproj gives other soft rows. This script takes the keys, the tensor names, the
types and the layouts of a Gemma 4 mmproj (--like) and every tensor's values from the checkpoint
(bfloat16 as stored; float32 tensors of --like get the bfloat16 values exactly):

    PYTHONPATH=../llama.cpp/gguf-py python scripts/dgemma_mmproj.py \\
        --checkpoint models/diffusiongemma-26B-A4B-it \\
        --like models/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-it-mmproj.gguf \\
        --out models/diffusiongemma-26B-A4B-it-BF12-GGUF/diffusiongemma-26B-A4B-it-mmproj.gguf

Each tensor is compared with --like's as a check of the names and the layouts: the tower's must
be within 1e-3 (relative), mm.input_projection is the one that differs.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct

import gguf
import numpy as np

T = "model.encoder.vision_tower."
LAYER = {
    "ln1": "input_layernorm", "attn_post_norm": "post_attention_layernorm",
    "ln2": "pre_feedforward_layernorm", "ffn_post_norm": "post_feedforward_layernorm",
    "attn_q": "self_attn.q_proj.linear", "attn_k": "self_attn.k_proj.linear",
    "attn_v": "self_attn.v_proj.linear", "attn_out": "self_attn.o_proj.linear",
    "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
    "ffn_gate": "mlp.gate_proj.linear", "ffn_up": "mlp.up_proj.linear", "ffn_down": "mlp.down_proj.linear",
}
OTHER = {
    "mm.input_projection.weight": "model.encoder.embed_vision.embedding_projection.weight",
    "v.patch_embd.weight": T + "patch_embedder.input_proj.weight",
    "v.position_embd.weight": T + "patch_embedder.position_embedding_table",
    "v.std_bias": T + "std_bias",
    "v.std_scale": T + "std_scale",
}


class Checkpoint:
    """The bfloat16 tensors of a sharded safetensors checkpoint, as uint16 bits."""

    def __init__(self, d):
        self.d = d
        self.map = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
        self.heads = {}

    def bits(self, name):
        path = os.path.join(self.d, self.map[name])
        if path not in self.heads:
            with open(path, "rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                self.heads[path] = (n, json.loads(f.read(n)))
        n, head = self.heads[path]
        e = head[name]
        assert e["dtype"] == "BF16", (name, e["dtype"])
        a, b = e["data_offsets"]
        with open(path, "rb") as f:
            f.seek(8 + n + a)
            return np.frombuffer(f.read(b - a), np.uint16).reshape(e["shape"])


def source(name):
    """The checkpoint's name of an mmproj tensor."""
    if name in OTHER:
        return OTHER[name]
    m = re.fullmatch(r"v\.blk\.(\d+)\.(\w+)\.weight", name)
    assert m and m.group(2) in LAYER, name
    return "%sencoder.layers.%s.%s.weight" % (T, m.group(1), LAYER[m.group(2)])


def values(ck, name, shape):
    """The bits of the checkpoint's tensor in the mmproj's layout (numpy order of shape)."""
    x = ck.bits(source(name))
    if name == "v.patch_embd.weight":
        # (out, 16 x 16 x 3: the pixels' rows, columns, channels) -> (out, 3, 16, 16)
        x = x.reshape(shape[0], shape[2], shape[3], shape[1]).transpose(0, 3, 1, 2)
    assert x.size == int(np.prod(shape)), (name, x.shape, shape)
    return np.ascontiguousarray(x).reshape(shape)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", required=True, help="DiffusionGemma's safetensors directory")
    ap.add_argument("--like", required=True, help="a Gemma 4 26B mmproj GGUF: the keys, names, types, layouts")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    ck = Checkpoint(args.checkpoint)
    like = gguf.GGUFReader(args.like)
    BF16, F32 = gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32
    out, worst = [], 0.0
    for t in like.tensors:
        shape = tuple(int(v) for v in reversed(t.shape))
        bits = values(ck, t.name, shape)
        f = (bits.astype(np.uint32) << 16).view(np.float32)
        if t.tensor_type == BF16:
            old = (np.ascontiguousarray(t.data).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            data = bits.view(np.uint8).reshape(t.data.shape)
        else:
            assert t.tensor_type == F32, (t.name, t.tensor_type)
            old = np.asarray(t.data, np.float32)
            data = f.reshape(t.data.shape)
        rel = float(np.linalg.norm(f.reshape(-1) - old.reshape(-1)) / max(np.linalg.norm(old), 1e-30))
        if t.name == "mm.input_projection.weight":
            print("%s: %.3f against --like (its own)" % (t.name, rel))
        else:
            if rel > 1e-3:
                raise SystemExit("%s: %.3g against --like: not the tower's tensor" % (t.name, rel))
            worst = max(worst, rel)
        out.append((t, data))
    print("the tower's %d tensors within %.2g of --like's" % (len(out) - 1, worst))

    arch = like.fields[gguf.Keys.General.ARCHITECTURE].contents()
    w = gguf.GGUFWriter(args.out, arch=arch, endianess=like.endianess)
    for f in like.fields.values():
        if f.name.startswith("GGUF.") or f.name == gguf.Keys.General.ARCHITECTURE:
            continue
        typ = f.types[0]
        sub = f.types[-1] if typ == gguf.GGUFValueType.ARRAY else None
        value = f.contents()
        if f.name == gguf.Keys.General.NAME:
            value = "DiffusionGemma 26B A4B It"
        w.add_key_value(f.name, value, typ, sub_type=sub)
    for t, data in out:
        w.add_tensor_info(t.name, data.shape, data.dtype, data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for _t, data in out:
        w.write_tensor_data(data, tensor_endianess=like.endianess)
    w.close()
    print("wrote %s: %d tensors" % (args.out, len(out)))


if __name__ == "__main__":
    main()
