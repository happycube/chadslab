#!/usr/bin/env python3
"""Write the mmproj GGUF of a Gemma 4 E2B / E4B checkpoint folder (its vision and audio towers).

The E2B and E4B checkpoints (google/gemma-4-E4B-it and its fine-tunes, such as NVIDIA's Nemotron 3.5
Bolt E4B) hold a gemma4v vision tower and a gemma4a audio tower (a Conformer), with the clamps of
their clippable linears. This script writes them as llama.cpp's converter does
(convert_hf_to_gguf.py, Gemma4VisionAudioModel): the names and the keys of clip, the matrices in
bfloat16 as stored, the norms, the clamps, the convs, the position table and the patch embedding in
float32. Its own changes of the values are llama.cpp's: the patch linear as a conv (width, 3, 16,
16), the depthwise conv's (1024, 1, 5) as (1024, 5), and per_dim_scale through softplus. One
difference: llama.cpp takes the softplus in bfloat16 (the tensor's type), this script in float32
(as transformers computes it at run time).

    PYTHONPATH=../llama.cpp/gguf-py python scripts/gemma4_mmproj.py MODELS/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16 \\
        MODELS/NVIDIA-Nemotron-3.5-Bolt-E4B-BF16/nemotron-3.5-bolt-e4b-mmproj.gguf [--check REF.gguf]

--check compares the file written with another (llama.cpp's converter on the same checkpoint): the
same keys and tensors, each tensor's type, shape and values (per_dim_scale within one bfloat16
step, the softplus above). The engine reads the file with reclank.models.gemma4.encoders.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import gguf
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.formats.safetensors import Shards  # noqa: E402

CLAMPS = ("input_min", "input_max", "output_min", "output_max")
# a layer's tensors: the checkpoint's name in the layer -> the GGUF's (after "v.blk.N." / "a.blk.N.")
VISION = {
    "input_layernorm": "ln1", "post_attention_layernorm": "attn_post_norm",
    "pre_feedforward_layernorm": "ln2", "post_feedforward_layernorm": "ffn_post_norm",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_out", "self_attn.q_norm": "attn_q_norm", "self_attn.k_norm": "attn_k_norm",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
}
AUDIO = {
    "feed_forward1.pre_layer_norm": "ffn_norm", "feed_forward1.ffw_layer_1": "ffn_up",
    "feed_forward1.ffw_layer_2": "ffn_down", "feed_forward1.post_layer_norm": "ffn_post_norm",
    "feed_forward2.pre_layer_norm": "ffn_norm_1", "feed_forward2.ffw_layer_1": "ffn_up_1",
    "feed_forward2.ffw_layer_2": "ffn_down_1", "feed_forward2.post_layer_norm": "ffn_post_norm_1",
    "norm_pre_attn": "attn_pre_norm", "norm_post_attn": "attn_post_norm", "norm_out": "ln2",
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.post": "attn_out", "self_attn.relative_k_proj": "attn_k_rel",
    "self_attn.per_dim_scale": "per_dim_scale",
    "lconv1d.pre_layer_norm": "conv_norm", "lconv1d.conv_norm": "norm_conv",
    "lconv1d.linear_start": "conv_pw1", "lconv1d.linear_end": "conv_pw2",
    "lconv1d.depthwise_conv1d": "conv_dw",
}
# the tensors outside the layers
OTHER = {
    "model.vision_tower.patch_embedder.input_proj": "v.patch_embd",
    "model.vision_tower.patch_embedder.position_embedding_table": "v.position_embd",
    "model.embed_vision.embedding_projection": "mm.input_projection",
    "model.audio_tower.subsample_conv_projection.layer0.conv": "a.conv1d.0",
    "model.audio_tower.subsample_conv_projection.layer0.norm": "a.conv1d.0.norm",
    "model.audio_tower.subsample_conv_projection.layer1.conv": "a.conv1d.1",
    "model.audio_tower.subsample_conv_projection.layer1.norm": "a.conv1d.1.norm",
    "model.audio_tower.subsample_conv_projection.input_proj_linear": "a.input_projection",
    "model.audio_tower.output_proj": "a.pre_encode.out",
    "model.embed_audio.embedding_projection": "mm.a.input_projection",
}


def gguf_name(name):
    """The GGUF name of a checkpoint tensor of the towers, or None (the text model's)."""
    for pre, tab, out in (("model.vision_tower.encoder.layers.", VISION, "v.blk."),
                          ("model.audio_tower.layers.", AUDIO, "a.blk.")):
        if name.startswith(pre):
            i, rest = name[len(pre):].split(".", 1)
            for k, v in tab.items():
                if rest.startswith(k + "."):
                    suffix = rest[len(k) + 1:]
                    suffix = "weight" if suffix == "linear.weight" else suffix
                    return "%s%s.%s.%s" % (out, i, v, suffix)
                if rest == k:                       # (per_dim_scale: a bare parameter)
                    return "%s%s.%s.weight" % (out, i, v)
            raise KeyError("no GGUF name for %s" % name)
    for k, v in OTHER.items():
        if name == k:
            return v + ".weight"
        if name.startswith(k + "."):
            suffix = name[len(k) + 1:]
            return v + "." + ("weight" if suffix == "linear.weight" else suffix)
    return None


def f32_of(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def tensor(st, name, gname):
    """(GGUF type, numpy array in the GGUF's numpy order) of a checkpoint tensor."""
    bits = st.get_bf16(name) if st.dtype(name) == "BF16" else None
    x = f32_of(bits) if bits is not None else st.get(name, dtype=np.float32)
    if x.ndim == 0:
        x = x.reshape(1)
    if gname.endswith("per_dim_scale.weight"):
        return gguf.GGMLQuantizationType.F32, np.logaddexp(0.0, x.astype(np.float64)).astype(np.float32)
    if gname == "v.patch_embd.weight":
        # (width, 16 x 16 x 3: the pixels' rows, columns, channels) -> (width, 3, 16, 16)
        p = int(round((x.shape[1] // 3) ** 0.5))
        x = x.reshape(x.shape[0], p, p, 3).transpose(0, 3, 1, 2)
        return gguf.GGMLQuantizationType.F32, np.ascontiguousarray(x)
    if gname.endswith("conv_dw.weight"):
        return gguf.GGMLQuantizationType.F32, np.ascontiguousarray(x.reshape(x.shape[0], x.shape[2]))
    # llama.cpp's rule (tensor_force_quant): an audio tensor whose name has ".conv", or "_conv" and
    # ".weight" (the subsample convs and their input_proj_linear), in float32
    audio_conv = "audio_tower" in name and (".conv" in name or ("_conv" in name and ".weight" in name))
    if x.ndim != 2 or "position_embd" in gname or audio_conv:
        return gguf.GGMLQuantizationType.F32, np.ascontiguousarray(x, dtype=np.float32)
    if bits is None:
        return gguf.GGMLQuantizationType.F32, np.ascontiguousarray(x, dtype=np.float32)
    return gguf.GGMLQuantizationType.BF16, np.ascontiguousarray(bits)


def keys(cfg):
    """The clip keys of llama.cpp's mmproj of a gemma4v + gemma4a model."""
    v, a = cfg["vision_config"], cfg.get("audio_config")
    text = cfg.get("text_config", cfg)
    out = [("general.architecture", "clip", gguf.GGUFValueType.STRING),
           ("clip.has_vision_encoder", True, gguf.GGUFValueType.BOOL),
           ("clip.vision.projection_dim", text["hidden_size"], gguf.GGUFValueType.UINT32),
           ("clip.vision.image_size", 224, gguf.GGUFValueType.UINT32),
           ("clip.vision.patch_size", v["patch_size"], gguf.GGUFValueType.UINT32),
           ("clip.vision.embedding_length", v["hidden_size"], gguf.GGUFValueType.UINT32),
           ("clip.vision.feed_forward_length", v["intermediate_size"], gguf.GGUFValueType.UINT32),
           ("clip.vision.block_count", v["num_hidden_layers"], gguf.GGUFValueType.UINT32),
           ("clip.vision.attention.head_count", v["num_attention_heads"], gguf.GGUFValueType.UINT32),
           ("clip.vision.image_mean", [0.0, 0.0, 0.0], gguf.GGUFValueType.ARRAY),
           ("clip.vision.image_std", [1.0, 1.0, 1.0], gguf.GGUFValueType.ARRAY),
           ("clip.vision.projector_type", "gemma4v", gguf.GGUFValueType.STRING),
           ("clip.vision.attention.layer_norm_epsilon", float(v.get("layer_norm_eps", 1e-6)),
            gguf.GGUFValueType.FLOAT32)]
    if a:
        out += [("clip.has_audio_encoder", True, gguf.GGUFValueType.BOOL),
                ("clip.audio.projection_dim", text["hidden_size"], gguf.GGUFValueType.UINT32),
                ("clip.audio.embedding_length", a["hidden_size"], gguf.GGUFValueType.UINT32),
                ("clip.audio.feed_forward_length", 4 * a["hidden_size"], gguf.GGUFValueType.UINT32),
                ("clip.audio.block_count", a["num_hidden_layers"], gguf.GGUFValueType.UINT32),
                ("clip.audio.attention.head_count", a["num_attention_heads"], gguf.GGUFValueType.UINT32),
                ("clip.audio.projector_type", "gemma4a", gguf.GGUFValueType.STRING),
                ("clip.audio.num_mel_bins", a.get("input_feat_size", 128), gguf.GGUFValueType.UINT32),
                ("clip.audio.attention.layer_norm_epsilon", 1e-5, gguf.GGUFValueType.FLOAT32)]
    return out


def check(path, ref):
    """The file against ref (llama.cpp's): the same keys and tensors, each the same type, shape and
    values (per_dim_scale within one bfloat16 step). Return the list of differences."""
    a, b = gguf.GGUFReader(path), gguf.GGUFReader(ref)
    bad = []
    ka = {f.name: f.contents() for f in a.fields.values() if f.name.startswith("clip.")}
    kb = {f.name: f.contents() for f in b.fields.values() if f.name.startswith("clip.")}
    for k in sorted(set(ka) | set(kb)):
        x, y = ka.get(k), kb.get(k)
        same = (np.allclose(x, y, rtol=1e-6) if isinstance(x, (float, list)) and isinstance(y, (float, list))
                else x == y)
        if not same:
            bad.append("key %s: %r against %r" % (k, x, y))
    ta, tb = {t.name: t for t in a.tensors}, {t.name: t for t in b.tensors}
    for n in sorted(set(ta) | set(tb)):
        if n not in ta or n not in tb:
            bad.append("tensor %s only in %s" % (n, "the file" if n in ta else "ref"))
            continue
        x, y = ta[n], tb[n]
        if x.tensor_type != y.tensor_type or tuple(x.shape) != tuple(y.shape):
            bad.append("%s: %s %s against %s %s" % (n, x.tensor_type.name, tuple(x.shape), y.tensor_type.name,
                                                    tuple(y.shape)))
            continue
        if n.endswith("per_dim_scale.weight"):
            xv, yv = np.asarray(x.data, np.float32), np.asarray(y.data, np.float32)
            if np.abs(xv - yv).max() > np.abs(yv).max() * 2.0 ** -7:
                bad.append("%s: %.3g" % (n, np.abs(xv - yv).max()))
        elif not np.array_equal(np.asarray(x.data).view(np.uint8), np.asarray(y.data).view(np.uint8)):
            bad.append("%s: other values" % n)
    return bad, len(ta)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("checkpoint")
    ap.add_argument("out")
    ap.add_argument("--check", metavar="REF", help="compare with this mmproj (llama.cpp's converter's)")
    a = ap.parse_args()
    cfg = json.load(open(os.path.join(a.checkpoint, "config.json")))
    st = Shards(a.checkpoint)
    w = gguf.GGUFWriter(a.out, arch="clip")
    for k, v, t in keys(cfg):
        if k == "general.architecture":
            continue
        if t == gguf.GGUFValueType.ARRAY:
            w.add_array(k, v)
        else:
            w.add_key_value(k, v, t)
    w.add_name(os.path.basename(os.path.normpath(a.checkpoint)))
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    out = []
    for name in sorted(st.names()):
        g = gguf_name(name)
        if g is None:
            continue
        typ, data = tensor(st, name, g)
        out.append((g, typ, data))
    for g, typ, data in out:
        if typ == gguf.GGMLQuantizationType.BF16:
            w.add_tensor(g, data.view(np.uint8).reshape(*data.shape[:-1], data.shape[-1] * 2), raw_dtype=typ)
        else:
            w.add_tensor(g, data)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print("wrote %s: %d tensors (vision %d, audio %d)" % (
        a.out, len(out), sum(g.startswith(("v.", "mm.input")) for g, _, _ in out),
        sum(g.startswith(("a.", "mm.a.")) for g, _, _ in out)))
    if a.check:
        bad, n = check(a.out, a.check)
        print("against %s: %d tensors, %s" % (a.check, n, "the same" if not bad else "%d differences" % len(bad)))
        for b in bad[:20]:
            print("  " + b)
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
