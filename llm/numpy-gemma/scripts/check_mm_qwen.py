#!/usr/bin/env python3
"""Check the Qwen3.6 image encoder (np_gemma/vision_qwen.py) and M-RoPE.

The reference is transformers (models/qwen3_5_moe): Qwen3_5MoeVisionModel in
float32 with the BF16 weights of Qwen/Qwen3.6-35B-A3B (the tensors
model.visual.*, range-read into models2/qwen36-vision). Our encoder reads the
mmproj GGUF. The script checks:

1. The weights: each GGUF tensor against its safetensors tensor.
2. The image processor: the patches and the grid of Qwen2VLImageProcessor
   against image_input (the resize differs: PIL against torchvision).
3. The soft rows of the same patches (the pixel_values of HF). Pass: the
   mean relative difference is below --tol.
4. M-RoPE: the cos and sin of Qwen.rope for the positions of get_rope_index
   against Qwen3_5MoeTextRotaryEmbedding.
5. Video (--video, Qwen3.6): against Qwen3VLVideoProcessor (with
   cap_pixels_per_frame, max_video_tokens = --video-budget, max_frames =
   --video-frames) and Qwen3VLProcessor: the sampled frames and their
   times, the pixels, the rows of each pair of frames, the prompt ids, and
   the M-RoPE positions of get_rope_index.

    OPENBLAS_NUM_THREADS=16 PYTHONPATH=. python scripts/check_mm_qwen.py [--gpu] [--q8]

--model 38 checks the encoder of Qwen3.8-Flash-Next instead: the weights of
the NVFP4 checkpoint (model.visual.* are BF16 there) for both, and
Qwen4ExpVisionModel (transformers models/qwen4_exp) as the reference. The
weights need no check (the same file), and M-RoPE is that of Qwen3.6.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import vision_qwen as VQ  # noqa: E402
from np_gemma.gemma4_encoders import _Weights  # noqa: E402

MMPROJ = "models/Qwen3.6-35B-A3B-GGUF/mmproj-BF16.gguf"
SNAP = "models2/qwen36-vision"
SNAP38 = "models/Qwen3.8-Flash-Next-NVFP4"
MEDIA = "../llama.cpp"
IMAGES = ["tools/mtmd/test-1.jpeg", "media/matmul.png"]


def rel(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-30))


def mean_rel(a, b):
    return float(np.abs(a - b).mean() / max(np.abs(b).mean(), 1e-30))


def hf_tensors(snap):
    from safetensors.numpy import load_file
    raw = load_file(os.path.join(snap, "vision_bf16u16.safetensors"))
    return {k[len("model.visual."):]: (v.astype(np.uint32) << 16).view(np.float32)
            for k, v in raw.items()}


def check_weights(W, hf):
    """The GGUF tensors against the HF tensors (the name map of llama.cpp)."""
    g = W.g
    worst = 0.0
    pairs = [("v.patch_embd.bias", "patch_embed.proj.bias"),
             ("v.position_embd.weight", "pos_embed.weight"),
             ("v.post_ln.weight", "merger.norm.weight"), ("v.post_ln.bias", "merger.norm.bias"),
             ("mm.0.weight", "merger.linear_fc1.weight"), ("mm.0.bias", "merger.linear_fc1.bias"),
             ("mm.2.weight", "merger.linear_fc2.weight"), ("mm.2.bias", "merger.linear_fc2.bias")]
    for i in range(int(W.meta["clip.vision.block_count"])):
        for a, b in (("ln1", "norm1"), ("ln2", "norm2"), ("attn_qkv", "attn.qkv"),
                     ("attn_out", "attn.proj"), ("ffn_up", "mlp.linear_fc1"),
                     ("ffn_down", "mlp.linear_fc2")):
            for s in ("weight", "bias"):
                pairs.append(("v.blk.%d.%s.%s" % (i, a, s), "blocks.%d.%s.%s" % (i, b, s)))
    for a, b in pairs:
        x = np.asarray(g.dequant(a), np.float32).reshape(-1)
        y = hf[b].reshape(-1)
        worst = max(worst, rel(x, y))
    pk = hf["patch_embed.proj.weight"]              # (1152, 3, 2, 16, 16)
    for t, name in ((0, "v.patch_embd.weight"), (1, "v.patch_embd.weight.1")):
        x = np.asarray(g.dequant(name), np.float32).reshape(-1)
        worst = max(worst, rel(x, pk[:, :, t].reshape(-1)))
    unused = set(hf) - {b for _, b in pairs} - {"patch_embed.proj.weight"}
    print("weights: %d tensors, worst relative difference %.2e; HF tensors not compared: %s"
          % (len(pairs) + 2, worst, sorted(unused) or "none"))
    return worst == 0.0


def hf_model(hf, snap=SNAP, q38=False):
    import torch
    if q38:
        from transformers.models.qwen4_exp.configuration_qwen4_exp import (
            Qwen4ExpVisionConfig as Config)
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel as Model
    else:
        from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
            Qwen3_5MoeVisionConfig as Config)
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeVisionModel as Model)
    vc = dict(json.load(open(os.path.join(snap, "config.json")))["vision_config"])
    vc.pop("model_type", None)
    vc.pop("dtype", None)
    cfg = Config(**vc)
    cfg._attn_implementation = "sdpa"
    m = Model(cfg)
    sd = {k: torch.from_numpy(v.copy()) for k, v in hf.items()}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "inv_freq" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    return m.float().eval()


def processor(budget, snap=SNAP):
    from transformers import AutoImageProcessor
    p = AutoImageProcessor.from_pretrained(snap)
    p.max_pixels = budget * VQ.FACTOR * VQ.FACTOR
    p.size = {"shortest_edge": VQ.MIN_TOKENS * VQ.FACTOR * VQ.FACTOR,
              "longest_edge": budget * VQ.FACTOR * VQ.FACTOR}
    return p


def check_mrope(W):
    """Qwen.rope of (3, t) positions against transformers."""
    import torch
    from np_gemma.qwen import Qwen, QwenConfig
    from np_gemma.media import mrope_positions, Span
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding
    tc = json.load(open(os.path.join(SNAP, "config.json")))["text_config"]
    tc.pop("model_type", None)
    hcfg = Qwen3_5MoeTextConfig(**tc)
    rot = Qwen3_5MoeTextRotaryEmbedding(hcfg)
    cfg = QwenConfig.__new__(QwenConfig)
    cfg.rope_theta = tc["rope_parameters"]["rope_theta"]
    cfg.rotary_dim = int(tc["head_dim"] * tc["rope_parameters"]["partial_rotary_factor"])
    cfg.mrope_section = tc["rope_parameters"]["mrope_section"]
    m = Qwen.__new__(Qwen)
    m.cfg = cfg
    # text 5, an image of 3 x 4 tokens, text 3, an image of 5 x 2, text 2
    n = 5 + 12 + 3 + 10 + 2
    spans = [Span(5, np.zeros((12, 1), np.float32), False, "image", "a", grid=(3, 4)),
             Span(20, np.zeros((10, 1), np.float32), False, "image", "b", grid=(5, 2))]
    pos = mrope_positions(n, spans)
    # get_rope_index of transformers on the same prompt (image grids in
    # patches: 2 x the tokens)
    import types
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel
    d = types.SimpleNamespace(config=types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    d.get_vision_position_ids = types.MethodType(Qwen3_5MoeModel.get_vision_position_ids, d)
    kinds = np.zeros(n, np.int64)
    for sp in spans:
        kinds[sp.start:sp.end] = 1
    ids = torch.where(torch.from_numpy(kinds) == 1, 248056, 11)[None]
    grid = torch.tensor([[1, 2 * sp.grid[0], 2 * sp.grid[1]] for sp in spans])
    want, _ = Qwen3_5MoeModel.get_rope_index(d, ids, torch.from_numpy(kinds)[None],
                                             image_grid_thw=grid)
    ok = np.array_equal(pos, want[:, 0].numpy())
    cos, sin = m.rope(pos)
    with torch.no_grad():
        hc, hs = rot(torch.zeros(1, dtype=torch.float32), torch.from_numpy(pos[:, None, :]))
    d = max(rel(cos, hc[0].numpy()), rel(sin, hs[0].numpy()))
    c1, s1 = m.rope(np.arange(7, 19))
    c3, s3 = m.rope(np.tile(np.arange(7, 19), (3, 1)))
    same = np.array_equal(c1, c3) and np.array_equal(s1, s3)
    print("M-RoPE: positions %s; cos/sin against HF %.2e; text (3, t) == (t,): %s"
          % ("same" if ok else "DIFFER", d, same))
    return ok and d < 1e-5 and same


VIDEO = "models2/mm-refs/slideshow.mp4"


def check_video(V, model, path, budget, max_frames):
    """Our video path against transformers (section 5 of the module text)."""
    import av
    import torch
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    from transformers.video_utils import VideoMetadata
    from np_gemma import media as MD
    from np_gemma.qwen_tok import QwenTokenizer
    with av.open(path) as c:
        st = c.streams.video[0]
        vfps = float(st.average_rate)
        allf = np.stack([fr.to_ndarray(format="rgb24") for fr in c.decode(video=0)])
    meta = VideoMetadata(total_num_frames=len(allf), fps=vfps, width=allf.shape[2],
                         height=allf.shape[1], duration=len(allf) / vfps)
    vp = Qwen3VLVideoProcessor.from_pretrained(SNAP)
    # the arguments of a call do not reach resize(): set them on the processor
    vp.cap_pixels_per_frame, vp.max_video_tokens, vp.max_frames = True, budget, max_frames
    kw = dict(do_sample_frames=True)
    out = vp(videos=[allf], video_metadata=[meta], return_metadata=True, return_tensors="pt", **kw)
    pv, thw = out["pixel_values_videos"], out["video_grid_thw"]
    hidx = list(out["video_metadata"][0].frames_indices)
    frames, idx, _ = VQ.video_frames(path, max_frames=max_frames)
    ours, grid, times = VQ.video_input(path, budget, max_frames)
    t_, gh, gw = (int(v) for v in thw[0])
    same_idx = list(idx) == [int(i) for i in hidx]
    pd = mean_rel(ours, pv.numpy()) if ours.shape == tuple(pv.shape) else float("nan")
    with torch.no_grad():
        ref = model(pv, grid_thw=thw).pooler_output.numpy()
    t0 = time.time()
    rows = np.concatenate(V.encode_video(np.ascontiguousarray(pv.numpy()), (t_, gh, gw)))
    dt = time.time() - t0
    e = mean_rel(rows, ref)
    print("video %s: %d frames (the same indices: %s), grid %s (ours %s), %d tokens; pixels mean "
          "rel %.4f; rows mean rel %.4f max rel %.4f; %.2f s"
          % (os.path.basename(path), len(idx), same_idx, (t_, gh, gw), grid, rows.shape[0], pd, e,
             rel(rows, ref), dt))
    # the prompt: Qwen3VLProcessor against video_text and expand_qwen
    tok = AutoTokenizer.from_pretrained(SNAP)
    proc = Qwen3VLProcessor(image_processor=AutoImageProcessor.from_pretrained(SNAP), tokenizer=tok,
                            video_processor=vp)
    text = ("<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>What happens?<|im_end|>\n"
            "<|im_start|>assistant\n")
    hp = proc(text=[text], videos=[allf], video_metadata=[meta], return_tensors="pt",
              videos_kwargs=dict(return_metadata=True, **kw))
    hids = hp["input_ids"][0].tolist()
    qt = QwenTokenizer(os.path.join(SNAP, "tokenizer.json"))
    n = rows.shape[0] // t_
    groups = [(tm, rows[i * n:(i + 1) * n], (gh // 2, gw // 2)) for i, tm in enumerate(times)]
    ids = qt.encode(text.replace("<|video_pad|>", VQ.video_text(groups)))
    ids, spans = MD.expand_qwen(ids, [MD.Media("video", r, "v%d" % i, grid=g)
                                      for i, (_t, r, g) in enumerate(groups)])
    same_ids = ids == hids
    import types
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeModel
    d = types.SimpleNamespace(config=types.SimpleNamespace(
        vision_config=types.SimpleNamespace(spatial_merge_size=2)))
    d.get_vision_position_ids = types.MethodType(Qwen3_5MoeModel.get_vision_position_ids, d)
    want, _ = Qwen3_5MoeModel.get_rope_index(d, hp["input_ids"], hp["mm_token_type_ids"],
                                             video_grid_thw=hp["video_grid_thw"])
    pos = MD.mrope_positions(len(ids), spans)
    same_pos = same_ids and np.array_equal(pos, want[:, 0].numpy())
    print("video prompt: %d tokens, the same ids as Qwen3VLProcessor: %s; M-RoPE positions equal: %s;"
          " times %s" % (len(ids), same_ids, same_pos, " ".join("%.1f" % t for t in times)))
    if not same_ids:
        k = next(i for i in range(min(len(ids), len(hids))) if ids[i] != hids[i])
        print("   first difference at %d: ours %r, HF %r" % (k, qt.decode(ids[k:k + 8]),
                                                            qt.decode(hids[k:k + 8])))
    return same_idx and e < 0.01 and same_ids and same_pos and (pd < 0.01 or pd != pd)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=("36", "38"), default="36")
    ap.add_argument("--mmproj", default=None, help="default: that of --model")
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--q8", action="store_true")
    ap.add_argument("--py", action="store_true", help="also the NumPy layers (slow)")
    ap.add_argument("--video", nargs="?", const=VIDEO, default=None,
                    help="also check a video (default %s)" % VIDEO)
    ap.add_argument("--video-budget", type=int, default=128)
    ap.add_argument("--video-frames", type=int, default=32)
    args = ap.parse_args()
    import torch
    torch.set_num_threads(16)
    q38 = args.model == "38"
    snap = SNAP38 if q38 else SNAP
    mmproj = args.mmproj or (SNAP38 if q38 else MMPROJ)
    if q38:
        W = VQ.vision_weights(mmproj)
        hf = {k: W.tensor(k) for k in W.where}
        good = check_mrope(None)
    else:
        W = _Weights(mmproj)
        hf = hf_tensors(SNAP)
        good = check_weights(W, hf)
        good &= check_mrope(W)
    emb = VQ.QwenEmbedder(mmproj, gpu=args.gpu, q8=args.q8)
    V = emb.vision
    model = hf_model(hf, snap, q38)
    proc = processor(args.budget, snap)
    for name in IMAGES:
        path = os.path.join(MEDIA, name)
        from np_gemma.unified import load_image
        img = load_image(path)
        out = proc(images=[img], return_tensors="pt")
        pv, thw = out["pixel_values"], out["image_grid_thw"]
        t_, gh, gw = (int(v) for v in thw[0])
        ours_p, grid = VQ.image_input(path, args.budget)
        # HF patches: (C, T, P, P), two equal frames
        hp = pv.numpy().reshape(-1, 3, 2, VQ.PATCH, VQ.PATCH)
        same_t = float(np.abs(hp[:, :, 0] - hp[:, :, 1]).max())
        hp0 = np.ascontiguousarray(hp[:, :, 0].reshape(hp.shape[0], -1))
        pd = mean_rel(ours_p, hp0) if ours_p.shape == hp0.shape else float("nan")
        with torch.no_grad():
            t0 = time.time()
            ref = model(pv, grid_thw=thw).pooler_output.numpy()
            t_hf = time.time() - t0
        t0 = time.time()
        rows = V.encode(hp0, (gh, gw))
        t1 = time.time()
        rows = V.encode(hp0, (gh, gw))
        t_ours = time.time() - t1
        e = mean_rel(rows, ref)
        print("%s: grid %dx%d (ours %dx%d; frames equal %.0e; patches mean rel %.4f); %d tokens; "
              "rows mean rel %.4f max rel %.4f; ours %.2f s (first %.2f s), HF %.1f s"
              % (name, gh, gw, grid[0], grid[1], same_t, pd, rows.shape[0], e, rel(rows, ref),
                 t_ours, t1 - t0, t_hf))
        good &= e < args.tol
        if args.py:
            import np_gemma.vision_qwen as M
            M._ENC_PY = True
            r2 = V.encode(hp0, (gh, gw))
            M._ENC_PY = False
            print("   NumPy layers: mean rel %.4f; program vs NumPy %.4f"
                  % (mean_rel(r2, ref), mean_rel(rows, r2)))
    if args.video:
        good &= check_video(V, model, args.video, args.video_budget, args.video_frames)
    print("RESULT", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    raise SystemExit(main())
