#!/usr/bin/env python3
"""The image path of Nemotron 3.5 Super VL in NumPy (reclank/models/nemotron/radio.py) against the
checkpoint's own torch code (image_processing_nemotron_h_omni.py, modeling_radio.py, the vision
projector of modeling_nemotron_h_omni.py), both in float32 on the CPU, on real images
(NEMOTRON_SUPER_PLAN.md S8).

The torch side runs in a venv with torch and transformers (a subprocess of this script with
--torch): it loads only vision_model.* and mlp1.* (and the final layer norm), runs the
processor and the tower, and writes the pixels, the features and the rows to an .npz; it also
checks the prompt layout of radio.py (render_content, expand_images, image_span) against the
chat template and the processor's input_ids. The NumPy side then runs here and reports, for
the pixels, the features and the rows: the max abs diff, the relative error (|d| / |ref|, all
values) and the cosine of each row (min and mean).

    CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 taskset -c 22,23,46,47 \\
        /home/cpage/npg-run/.venv/bin/python scripts/check_nemotron_radio.py \\
        --image /home/cpage/aitests/ds41-2/prodigy-poster.png
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))

# (the defaults of --model, --torch-python, --work)
MODEL = os.environ.get("NEMOTRON_SUPER", "/home/cpage/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
TORCH_PY = os.environ.get("TORCH_PYTHON", "/home/cpage/npg-run/minicpm-ref-venv/bin/python")
WORK = os.environ.get("RADIO_WORK", "/home/cpage/npg-run/work/super/vis")
VM = "vision_model.radio_model."
PARTS = [{"type": "image"}, {"type": "text", "text": "What does this chart show?"}]


# ---- the torch side (the checkpoint's code)

def _ref_package(model):
    """The checkpoint's .py files as the package nemo_ref (their relative imports work)."""
    import importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location("nemo_ref", os.path.join(model, "__init__.py"),
                                                  submodule_search_locations=[model])
    pkg = importlib.util.module_from_spec(spec)
    sys.modules["nemo_ref"] = pkg
    spec.loader.exec_module(pkg)
    return (importlib.import_module("nemo_ref.modeling_radio"),
            importlib.import_module("nemo_ref.configuration_radio"),
            importlib.import_module("nemo_ref.modeling_nemotron_h_omni"),
            importlib.import_module("nemo_ref.image_processing_nemotron_h_omni"),
            importlib.import_module("nemo_ref.processing_nemotron_h_omni"))


def _tensors(model, names):
    """The tensors by name as float32 torch tensors, one safe_open read each."""
    import torch
    from safetensors import safe_open
    where = json.load(open(os.path.join(model, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for f in sorted(set(where[n] for n in names)):
        with safe_open(os.path.join(model, f), framework="pt") as st:
            for n in names:
                if where[n] == f:
                    out[n] = st.get_tensor(n).to(torch.float32)
    return out


def torch_side(model, image, out):
    import types
    import torch
    from PIL import Image
    from transformers import AutoTokenizer
    from reclank.models.nemotron import radio
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
    mr, cr, mo, ip, pp = _ref_package(model)
    cfg = json.load(open(os.path.join(model, "config.json")))
    pre = json.load(open(os.path.join(model, "preprocessor_config.json")))
    t0 = time.time()

    # the processor (pixels, num_tokens) and the prompt
    keys = ("norm_mean", "norm_std", "patch_size", "downsample_ratio", "min_num_patches",
            "max_num_patches", "max_model_len", "video_target_num_patches", "video_maintain_aspect_ratio")
    proc_img = ip.NemotronH_Omni_Reasoning_V3ImageProcessor(**{k: pre[k] for k in keys})
    pil = Image.open(image)
    feat = proc_img(images=[pil], return_tensors="pt")
    pv = feat["pixel_values"]
    if isinstance(pv, list):
        pv = pv[0][None]
    pv = pv.to(torch.float32)
    n_ref = int(feat["num_tokens"][0])
    tok = AutoTokenizer.from_pretrained(model)
    proc = pp.NemotronH_Omni_Reasoning_V3Processor(image_processor=proc_img, tokenizer=tok)
    msgs = [{"role": "user", "content": PARTS}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids_ref = proc(images=[pil], text=[text], return_tensors="np")["input_ids"][0].tolist()
    hp, wp = pv.shape[-2] // 16, pv.shape[-1] // 16
    n_mine = radio.image_tokens(hp, wp)
    ids_mine = tok(radio.expand_images(text, [n_mine]), add_special_tokens=False)["input_ids"]
    i = ids_ref.index(radio.IMG_START)
    prompt = dict(content_in_template=radio.render_content(PARTS) in text,
                  n_ref=n_ref, n_mine=n_mine, ids_equal=ids_mine == ids_ref,
                  span_ok=ids_ref[i:i + n_ref + 2] == radio.image_span(n_ref),
                  prompt_len=len(ids_ref), template=text)

    # the tower and the projector, the weights of vision_model, mlp1 and the final layer norm
    vc = dict(cfg["vision_config"])
    for k in ("args", "model_type", "dtype"):
        vc.pop(k, None)
    rc = cr.RadioConfig(**vc)
    rc._attn_implementation = "sdpa"
    vm = mr.RadioModel(rc).float().eval()
    vm.make_preprocessor_external()
    g = VM + "model.patch_generator."
    names = [g + "embedder.weight", g + "pos_embed", g + "cls_token.token",
             "mlp1.0.weight", "mlp1.1.weight", "mlp1.3.weight",
             "vision_projector.vision_final_layernorm.weight", "vision_projector.vision_final_layernorm.bias"]
    layer = []
    for li in range(rc.num_hidden_layers):
        b = "%smodel.blocks.%d." % (VM, li)
        layer.append([b + k for k in ("norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
                                      "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
                                      "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias")])
    w = _tensors(model, names + [n for ls in layer for n in ls])
    sd = {"embeddings.patch_projection.weight": w[g + "embedder.weight"],
          "embeddings.position_embedding": w[g + "pos_embed"],
          "embeddings.cls_register_token": w[g + "cls_token.token"]}
    for li in range(rc.num_hidden_layers):
        b, d = "%smodel.blocks.%d." % (VM, li), "encoder.layer.%d." % li
        q, k, v = w[b + "attn.qkv.weight"].chunk(3, dim=0)
        qb, kb, vb = w[b + "attn.qkv.bias"].chunk(3, dim=0)
        sd.update({d + "norm1.weight": w[b + "norm1.weight"], d + "norm1.bias": w[b + "norm1.bias"],
                   d + "attention.attention.query.weight": q, d + "attention.attention.query.bias": qb,
                   d + "attention.attention.key.weight": k, d + "attention.attention.key.bias": kb,
                   d + "attention.attention.value.weight": v, d + "attention.attention.value.bias": vb,
                   d + "attention.output.dense.weight": w[b + "attn.proj.weight"],
                   d + "attention.output.dense.bias": w[b + "attn.proj.bias"],
                   d + "norm2.weight": w[b + "norm2.weight"], d + "norm2.bias": w[b + "norm2.bias"],
                   d + "mlp.fc1.weight": w[b + "mlp.fc1.weight"], d + "mlp.fc1.bias": w[b + "mlp.fc1.bias"],
                   d + "mlp.fc2.weight": w[b + "mlp.fc2.weight"], d + "mlp.fc2.bias": w[b + "mlp.fc2.bias"]})
    missing, unexpected = vm.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "layer_scale" not in m and "video_patch_projection" not in m
               and m != "summary_idxs"]
    assert not missing and not unexpected, (missing, unexpected)
    ns = types.SimpleNamespace(
        downsample_ratio=cfg["downsample_ratio"], video_temporal_patch_size=cfg["video_temporal_patch_size"],
        ps_version=cfg["ps_version"], vit_hidden_size=cfg["vit_hidden_size"],
        projector_hidden_size=cfg["projector_hidden_size"],
        llm_config=types.SimpleNamespace(hidden_size=cfg["llm_config"]["hidden_size"],
                                         num_nextn_predict_layers=cfg["llm_config"]["num_nextn_predict_layers"]),
        vision_config=types.SimpleNamespace(layer_norm_eps=cfg["vision_config"]["layer_norm_eps"]))
    pj = mo.NemotronH_Omni_Reasoning_V3VisionProjector(ns).float().eval()
    pj.mlp1.norm.weight.data.copy_(w["mlp1.0.weight"])
    pj.mlp1.linear1.weight.data.copy_(w["mlp1.1.weight"])
    pj.mlp1.linear2.weight.data.copy_(w["mlp1.3.weight"])
    pj.vision_final_layernorm.weight.data.copy_(w["vision_projector.vision_final_layernorm.weight"])
    pj.vision_final_layernorm.bias.data.copy_(w["vision_projector.vision_final_layernorm.bias"])
    del w, sd
    t1 = time.time()
    with torch.no_grad():
        feats = vm(pv).features
        rows = pj._project(feats, hp, wp)
    t2 = time.time()
    prompt.update(load_s=round(t1 - t0, 1), run_s=round(t2 - t1, 1))
    np.savez(out, pixels=pv[0].numpy(), features=feats[0].numpy(), rows=rows[0].numpy(),
             prompt=json.dumps(prompt))


# ---- the NumPy side and the comparison

def compare(name, ref, mine):
    ref, mine = ref.astype(np.float64), mine.astype(np.float64)
    if ref.shape != mine.shape:
        print("%-9s shape %s vs %s" % (name, ref.shape, mine.shape))
        return
    d = mine - ref
    r2, m2 = ref.reshape(ref.shape[0], -1), mine.reshape(mine.shape[0], -1)
    cos = (r2 * m2).sum(1) / (np.linalg.norm(r2, axis=1) * np.linalg.norm(m2, axis=1) + 1e-30)
    print("%-9s %-16s max|d| %.3e  max|ref| %.3e  rel %.3e  cos min %.9f mean %.9f" % (
        name, "x".join(map(str, ref.shape)), np.abs(d).max(), np.abs(ref).max(),
        np.linalg.norm(d) / np.linalg.norm(ref), cos.min(), cos.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--image", default=os.path.join(MODEL, "harness_comparisons.png"))
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--torch-python", default=TORCH_PY, help="a python with torch and transformers (CPU)")
    ap.add_argument("--torch", action="store_true", help="the torch side (in the torch venv)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--reuse", action="store_true", help="keep an existing torch .npz")
    ap.add_argument("--split", action="store_true", help="also the encoder on the reference's pixels")
    a = ap.parse_args()
    out = a.out or os.path.join(a.work, "radio_%s.npz" % os.path.splitext(os.path.basename(a.image))[0])
    if a.torch:
        torch_side(a.model, a.image, out)
        return
    os.makedirs(a.work, exist_ok=True)
    if not (a.reuse and os.path.exists(out)):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        t = time.time()
        subprocess.run([a.torch_python, os.path.abspath(__file__), "--torch", "--model", a.model,
                        "--image", a.image, "--out", out], check=True, env=env)
        print("torch side %.0f s" % (time.time() - t))
    ref = np.load(out)
    p = json.loads(str(ref["prompt"]))
    print("prompt: n tokens ref %d mine %d, ids equal %s, span ok %s, content in template %s, len %d" % (
        p["n_ref"], p["n_mine"], p["ids_equal"], p["span_ok"], p["content_in_template"], p["prompt_len"]))
    print("torch: load %.0f s, run %.0f s" % (p["load_s"], p["run_s"]))

    from reclank.models.nemotron.radio import RadioVision
    t = time.time()
    v = RadioVision(a.model)
    t1 = time.time()
    px, hp, wp = v.pixels(a.image)
    feats = v.features(px)
    rows = v.project(feats, hp, wp)
    t2 = time.time()
    print("numpy: load %.0f s, run %.0f s; image %s -> %d x %d patches, %d rows" % (
        t1 - t, t2 - t1, os.path.basename(a.image), hp, wp, rows.shape[0]))
    compare("pixels", ref["pixels"], px)
    compare("features", ref["features"], feats)
    compare("rows", ref["rows"], rows)
    # the tower and projector on the reference's own pixels: the encoder alone
    if a.split and px.shape == ref["pixels"].shape:
        f2 = v.features(ref["pixels"].astype(np.float32))
        compare("feat@ref", ref["features"], f2)
        compare("rows@ref", ref["rows"], v.project(f2, hp, wp))


if __name__ == "__main__":
    main()
