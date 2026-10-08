#!/usr/bin/env python3
"""Check the image and audio embedders of the Gemma 4 12B against transformers.

np_gemma/unified.py reads the mmproj GGUF. The reference is the
Gemma4UnifiedVisionEmbedder and the audio Gemma4UnifiedMultimodalEmbedder of
transformers, in float32, with the weights of the unquantized QAT
safetensors, and the processors of transformers. The script checks:

1. The image processor: the size, the count of soft tokens, the positions,
   and the pixels of each patch (PIL bicubic against torchvision bicubic).
2. The soft rows of the same pixels (the HF pixels, in the order of the
   GGUF): the embedder alone. Pass: max relative difference 1e-4.
3. The soft rows of the whole path (our pixels against the HF pixels).
   This is not a pass test. PIL and torchvision round the bicubic resize
   of 8-bit pixels in other ways (at most 2/255 for each value). The
   LayerNorm of a patch divides by the spread of its values, so a flat
   patch makes such a difference large (up to 0.14 of the largest value).
4. The audio frames and their soft rows. Pass: 1e-4.

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/check_mm_unified.py
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import unified as U  # noqa: E402

MMPROJ = "models/gemma-4-12B-qat-q4_0/mmproj-gemma-4-12b-it-qat-q4_0.gguf"
HUB = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub"
REPO = "models--google--gemma-4-12B-it-qat-q4_0-unquantized"
MEDIA = "../llama.cpp"
IMAGES = ["tools/mtmd/test-1.jpeg", "media/llama1-logo.png", "media/matmul.png",
          "tools/mtmd/tests/test-1-positive.png"]
AUDIO = ["tools/mtmd/test-2.mp3"]


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def hf_modules(snap):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.gemma4_unified import modeling_gemma4_unified as M
    cfg = AutoConfig.from_pretrained(snap)
    st = safe_open(os.path.join(snap, "model.safetensors"), "pt")

    def T(name):
        return st.get_tensor(name).float()

    ve = M.Gemma4UnifiedVisionEmbedder(cfg.vision_config, cfg.text_config).float().eval()
    p = "model.vision_embedder."
    ve.load_state_dict({
        "patch_ln1.weight": T(p + "patch_ln1.weight"), "patch_ln1.bias": T(p + "patch_ln1.bias"),
        "patch_dense.weight": T(p + "patch_dense.weight"),
        "patch_dense.bias": T(p + "patch_dense.bias"),
        "patch_ln2.weight": T(p + "patch_ln2.weight"), "patch_ln2.bias": T(p + "patch_ln2.bias"),
        "pos_embedding": T(p + "pos_embedding"),
        "pos_norm.weight": T(p + "pos_norm.weight"), "pos_norm.bias": T(p + "pos_norm.bias"),
        "multimodal_embedder.embedding_projection.weight":
            T("model.embed_vision.embedding_projection.weight"),
    })
    ae = M.Gemma4UnifiedMultimodalEmbedder(cfg.audio_config, cfg.text_config).float().eval()
    ae.load_state_dict({"embedding_projection.weight":
                        T("model.embed_audio.embedding_projection.weight")})
    return torch, ve, ae


def hwc_to_gguf(patches):
    """HF patches (n, 48*48*3) in the order row, column, channel -> the order
    of the GGUF (channel, row, column)."""
    n = patches.shape[0]
    return patches.reshape(n, U.PATCH, U.PATCH, 3).transpose(0, 3, 1, 2).reshape(n, -1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mmproj", default=MMPROJ)
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--budgets", default="70,280,1120")
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()
    snap = args.snapshot or sorted(glob.glob(os.path.join(HUB, REPO, "snapshots", "*")))[-1]
    torch, ve, ae = hf_modules(snap)
    from transformers.models.gemma4_unified.image_processing_gemma4_unified import (
        Gemma4UnifiedImageProcessor)
    from transformers.models.gemma4_unified.feature_extraction_gemma4_unified import (
        Gemma4UnifiedAudioFeatureExtractor)
    emb = U.UnifiedEmbedder(args.mmproj)
    ok = True

    print("image              budget  size       tokens  pixels max|d| mean|d|  "
          "embedder rel  whole path rel")
    for name in IMAGES:
        path = os.path.join(MEDIA, name)
        for budget in (int(b) for b in args.budgets.split(",")):
            proc = Gemma4UnifiedImageProcessor(max_soft_tokens=budget)
            img = U.load_image(path)
            out = proc(images=[img], return_tensors="pt")
            n = int(out["num_soft_tokens_per_image"][0])
            hf_pix = out["pixel_values"][0, :n].float().numpy()
            hf_pos = out["image_position_ids"][0, :n].numpy()
            pixels = U.image_pixels(img, budget)
            patches, pos, grid = U.image_patches(pixels)
            same_grid = patches.shape[0] == n and np.array_equal(pos, hf_pos)
            d = np.abs(hwc_to_gguf(hf_pix) - patches) if same_grid else None
            with torch.no_grad():
                ref = ve(out["pixel_values"][:, :n].float(),
                         out["image_position_ids"][:, :n]).pooler_output[0].numpy()
            mine_same = emb.image_rows(hwc_to_gguf(hf_pix), hf_pos)
            r1 = rel(mine_same, ref)
            r2 = rel(emb.image_rows(patches, pos), ref) if same_grid else float("nan")
            good = same_grid and r1 < args.tol
            ok &= good
            print("%-18s %6d  %4dx%-4d  %6d  %9.4f %8.5f  %12.2e  %14.2e %s" % (
                os.path.basename(name)[:18], budget, grid[1] * U.PATCH, grid[0] * U.PATCH, n,
                d.max() if d is not None else float("nan"),
                d.mean() if d is not None else float("nan"), r1, r2,
                "ok" if good else "FAIL"))

    fe = Gemma4UnifiedAudioFeatureExtractor()
    for name in AUDIO:
        samples = U.load_audio(os.path.join(MEDIA, name))
        feats = fe(samples, sampling_rate=U.AUDIO_RATE, return_tensors="np")
        hf_frames = np.asarray(feats["input_features"][0], np.float32)
        frames = U.audio_frames(samples)
        same = hf_frames.shape == frames.shape and np.array_equal(hf_frames, frames)
        with torch.no_grad():
            ref = ae(torch.from_numpy(hf_frames)[None]).numpy()[0]
        r = rel(emb.audio_rows(frames), ref)
        good = same and r < args.tol
        ok &= good
        print("audio %-12s %.1f s, %d tokens, frames equal %s, rel %.2e %s" % (
            os.path.basename(name), len(samples) / U.AUDIO_RATE, frames.shape[0], same, r,
            "ok" if good else "FAIL"))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
