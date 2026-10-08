#!/usr/bin/env python3
"""Check the gemma4v and gemma4a encoders (np_gemma/gemma4_encoders.py).

The reference is transformers (models/gemma4): Gemma4VisionModel,
Gemma4AudioModel, and Gemma4MultimodalEmbedder in float32, with the weights of
the unquantized QAT safetensors of the model (strict loading). Our encoders
read the mmproj GGUF of google. The script checks:

1. The weights: each GGUF tensor against its safetensors tensor, after the
   changes of the converter are undone (the name map of the module).
2. The image processor: the patches and positions of Gemma4ImageProcessor.
3. The soft rows of the same patches. Pass: relative difference --tol.
4. The mel of Gemma4AudioFeatureExtractor, and the soft rows of the same
   mel (--tol).

    OPENBLAS_NUM_THREADS=16 PYTHONPATH=. python scripts/check_mm_gemma4.py \\
        --mmproj models2/gemma-4-E4B-qat-gguf/gemma-4-E4B-it-mmproj.gguf \\
        --snapshot models2/gemma-4-E4B-qat
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import gemma4_encoders as G  # noqa: E402
from np_gemma import unified as U  # noqa: E402

MEDIA = "../llama.cpp"
IMAGES = ["tools/mtmd/test-1.jpeg", "media/matmul.png"]
AUDIO = ["tools/mtmd/test-2.mp3"]


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def mean_rel(a, b):
    return float(np.abs(a - b).mean() / max(np.abs(b).mean(), 1e-30))


class HFTensors:
    """The tensors of a safetensors file or of the shards of a directory."""

    def __init__(self, snap):
        import glob
        import json
        from safetensors import safe_open
        idx = os.path.join(snap, "model.safetensors.index.json")
        self.files = {}
        if os.path.exists(idx):
            for k, f in json.load(open(idx))["weight_map"].items():
                if os.path.exists(os.path.join(snap, f)):
                    self.files[k] = os.path.join(snap, f)
        for f in glob.glob(os.path.join(snap, "*.safetensors")):
            with safe_open(f, "pt") as h:
                for k in h.keys():
                    self.files.setdefault(k, f)
        self._open = {}

    def keys(self, prefix):
        return [k for k in self.files if k.startswith(prefix)]

    def get(self, k):
        from safetensors import safe_open
        f = self.files[k]
        if f not in self._open:
            self._open[f] = safe_open(f, "pt")
        return self._open[f].get_tensor(k).float()


def hf_models(snap, T):
    import torch
    from transformers import AutoConfig
    from transformers.models.gemma4 import modeling_gemma4 as M
    cfg = AutoConfig.from_pretrained(snap)
    # eager for the vision tower (its mask is all valid). The audio tower
    # needs sdpa (the default): with eager the mask is additive (0 for a key
    # that a query sees), and Gemma4AudioAttention, which masks where
    # attention_mask.logical_not() is True, then hides exactly those keys.
    for c in (cfg, cfg.vision_config, cfg.text_config):
        if c is not None:
            c._attn_implementation = "eager"
    if cfg.audio_config is not None:
        cfg.audio_config._attn_implementation = "sdpa"

    def load(module, prefix):
        sd = {k[len(prefix):]: T.get(k) for k in T.keys(prefix)}
        missing, unexpected = module.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "inv_freq" not in m]
        if missing or unexpected:
            raise SystemExit("%s: missing %s, unexpected %s" % (prefix, missing[:8], unexpected[:8]))
        return module.float().eval()

    vis = load(M.Gemma4VisionModel(cfg.vision_config), "model.vision_tower.")
    evis = load(M.Gemma4MultimodalEmbedder(cfg.vision_config, cfg.text_config), "model.embed_vision.")
    aud = aud_e = None
    if cfg.audio_config is not None and T.keys("model.audio_tower."):
        aud = load(M.Gemma4AudioModel(cfg.audio_config), "model.audio_tower.")
        aud_e = load(M.Gemma4MultimodalEmbedder(cfg.audio_config, cfg.text_config), "model.embed_audio.")
    return torch, cfg, vis, evis, aud, aud_e


def check_weights(emb, T):
    """Our weights (from the GGUF) against the safetensors. Return the worst
    relative difference and the count of compared tensors."""
    worst, n = 0.0, 0

    def cmp(ours, key):
        nonlocal worst, n
        if key not in T.files:
            raise SystemExit("no tensor %s in the safetensors" % key)
        ref = T.get(key).numpy().reshape(-1)
        a = np.asarray(ours)
        if a.dtype == np.uint16:
            a = (a.astype(np.uint32) << 16).view(np.float32)
        a = a.reshape(-1).astype(np.float32)
        if a.shape != ref.shape:
            raise SystemExit("%s: shape %s against %s" % (key, a.shape, ref.shape))
        worst = max(worst, rel(a, ref))
        n += 1

    v = emb.vision
    p = "model.vision_tower."
    cmp(v.patch_w, p + "patch_embedder.input_proj.weight")
    cmp(v.pos, p + "patch_embedder.position_embedding_table")
    if v.std is not None:
        cmp(v.std[0], p + "std_bias")
        cmp(v.std[1], p + "std_scale")
    names = [("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"), ("v", "self_attn.v_proj"),
             ("o", "self_attn.o_proj"), ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
             ("down", "mlp.down_proj")]
    for i, b in enumerate(v.blk):
        q = p + "encoder.layers.%d." % i
        for ours, hf in names:
            lin = b[ours]
            cmp(lin.w, q + hf + ".linear.weight")
            for s in ("imin", "imax", "omin", "omax"):
                if getattr(lin, s) is not None:
                    key = q + hf + "." + {"imin": "input_min", "imax": "input_max",
                                          "omin": "output_min", "omax": "output_max"}[s]
                    cmp([getattr(lin, s)], key)
        for ours, hf in (("ln1", "input_layernorm"), ("post_attn", "post_attention_layernorm"),
                         ("ln2", "pre_feedforward_layernorm"),
                         ("post_ffn", "post_feedforward_layernorm"),
                         ("qn", "self_attn.q_norm"), ("kn", "self_attn.k_norm")):
            cmp(b[ours], q + hf + ".weight")
    cmp(v.proj.w, "model.embed_vision.embedding_projection.weight")
    a = emb.audio_enc
    if a is not None:
        p = "model.audio_tower."
        for i, (w, nw) in enumerate(a.conv):
            cmp(w, p + "subsample_conv_projection.layer%d.conv.weight" % i)
            cmp(nw, p + "subsample_conv_projection.layer%d.norm.weight" % i)
        cmp(a.in_proj.w, p + "subsample_conv_projection.input_proj_linear.weight")
        for i, b in enumerate(a.blk):
            q = p + "layers.%d." % i
            for key, ff in (("ff1", "feed_forward1"), ("ff2", "feed_forward2")):
                pre, up, down, post = b[key]
                cmp(pre, q + ff + ".pre_layer_norm.weight")
                cmp(up.w, q + ff + ".ffw_layer_1.linear.weight")
                cmp(down.w, q + ff + ".ffw_layer_2.linear.weight")
                cmp(post, q + ff + ".post_layer_norm.weight")
            for ours, hf in (("q", "self_attn.q_proj.linear"), ("k", "self_attn.k_proj.linear"),
                             ("v", "self_attn.v_proj.linear"), ("o", "self_attn.post.linear"),
                             ("rel", "self_attn.relative_k_proj"),
                             ("pw1", "lconv1d.linear_start.linear"),
                             ("pw2", "lconv1d.linear_end.linear")):
                cmp(b[ours].w, q + hf + ".weight")
            import torch
            sp = torch.nn.functional.softplus(T.get(q + "self_attn.per_dim_scale")).numpy()
            worst_pds = rel(b["pds"], sp)
            worst = max(worst, worst_pds)
            n += 1
            cmp(b["dw"], q + "lconv1d.depthwise_conv1d.weight")
            cmp(b["conv_pre"], q + "lconv1d.pre_layer_norm.weight")
            cmp(b["conv_post"], q + "lconv1d.conv_norm.weight")
            cmp(b["pre_attn"], q + "norm_pre_attn.weight")
            cmp(b["post_attn"], q + "norm_post_attn.weight")
            cmp(b["out"], q + "norm_out.weight")
        cmp(a.out_proj.w, p + "output_proj.weight")
        cmp(a.out_proj.b, p + "output_proj.bias")
        cmp(a.proj.w, "model.embed_audio.embedding_projection.weight")
    return worst, n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--budget", type=int, default=280)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--gpu", action="store_true", help="the encoder programs on the GPU")
    ap.add_argument("--q8", action="store_true", help="Q8_0 weights and int8 activations")
    args = ap.parse_args()
    if args.q8 and args.tol == 1e-3:
        args.tol = 0.2
    if args.gpu and os.environ.get("NP_GEMMA_GPU_ENC_TC", "1") != "0" and args.tol == 1e-3:
        # The linears of the GPU on the tensor cores round x to float16: the
        # rows differ by a few percent at most (mean 1e-3), and the logits of
        # the E4B do not change (check_mm_prompt: KL 0.0064 against 0.0070).
        args.tol = 0.1
    emb = G.Gemma4Embedder(args.mmproj, gpu=args.gpu, q8=args.q8)
    T = HFTensors(args.snapshot)
    worst, n = check_weights(emb, T)
    ok = worst < 1e-6
    print("weights: %d tensors, worst relative difference %.2e %s" % (n, worst, "ok" if ok else "FAIL"),
          flush=True)
    torch, cfg, vis, evis, aud, aud_e = hf_models(args.snapshot, T)
    from transformers.models.gemma4.image_processing_gemma4 import Gemma4ImageProcessor
    for name in IMAGES:
        path = os.path.join(MEDIA, name)
        proc = Gemma4ImageProcessor(max_soft_tokens=args.budget)
        out = proc(images=[U.load_image(path)], return_tensors="pt")
        pv, pp = out["pixel_values"], out["image_position_ids"]
        valid = (pp[0] != -1).all(-1).numpy()
        with torch.no_grad():
            ref = evis(vis(pv.float(), pp).last_hidden_state).numpy()
        ref = ref.reshape(-1, ref.shape[-1])
        mine = emb.vision.encode(pv[0].numpy()[valid], pp[0].numpy()[valid])
        pix = U.image_pixels(U.load_image(path), args.budget)
        patches, pos = G.image_patches16(pix)
        same_pos = np.array_equal(pos, pp[0].numpy()[valid])
        r = rel(mine, ref) if mine.shape == ref.shape else float("inf")
        whole = rel(emb.vision.encode(patches, pos), ref) if same_pos else float("nan")
        good = r < args.tol and same_pos
        ok &= good
        print("image %-14s patches %d -> %d rows, positions equal %s, rel %.2e mean %.2e "
              "(whole path %.2e) %s" % (os.path.basename(name), int(valid.sum()), mine.shape[0],
                                        same_pos, r, mean_rel(mine, ref), whole,
                                        "ok" if good else "FAIL"), flush=True)
    if aud is not None:
        from transformers.models.gemma4.feature_extraction_gemma4 import Gemma4AudioFeatureExtractor
        fe = Gemma4AudioFeatureExtractor()
        for name in AUDIO:
            x = U.load_audio(os.path.join(MEDIA, name))
            f = fe([x], sampling_rate=16000, return_tensors="pt")
            mel_ref, m_ref = f["input_features"][0].numpy(), f["input_features_mask"][0].numpy()
            mel, valid = G.audio_features(x)
            rm = rel(mel, mel_ref) if mel.shape == mel_ref.shape else float("inf")
            with torch.no_grad():
                o = aud(f["input_features"].float(), f["input_features_mask"])
                ref = aud_e(o.last_hidden_state)[0]
                ref = ref[o.attention_mask[0].bool()].numpy()
            mine = emb.audio_enc.encode(mel_ref, m_ref.astype(bool))
            r = rel(mine, ref) if mine.shape == ref.shape else float("inf")
            good = rm < 1e-4 and np.array_equal(valid, m_ref.astype(bool)) and r < args.tol
            ok &= good
            print("audio %-14s mel rel %.2e, mask equal %s; %d tokens, rel %.2e %s" % (
                os.path.basename(name), rm, np.array_equal(valid, m_ref.astype(bool)),
                mine.shape[0], r, "ok" if good else "FAIL"), flush=True)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
