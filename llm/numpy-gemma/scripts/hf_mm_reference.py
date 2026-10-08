#!/usr/bin/env python3
"""Make the transformers reference of a chat prompt with an image or audio.

The model is Gemma4UnifiedForConditionalGeneration (the Gemma 4 12B) from the
unquantized QAT safetensors, in float32 on the CPU (about 48 GB of RAM). The
processor of the snapshot makes the prompt (the chat template and the soft
tokens). The script generates a greedy answer, then runs the prompt and the
answer again, and writes an .npz file:

    ids          the prompt and the answer (the soft tokens expanded)
    n_prompt     the count of prompt ids
    soft         the soft rows in the order of their tokens (n, 3840)
    mm_type      the media type of each id (0 text, 1 image, 3 audio)
    logits_idx   the top 64 token ids of each position from n_prompt - 1
    logits_val   their logits
    logz         the log of the sum of exp over the whole vocabulary
    fwd_idx, fwd_val, fwd_logz
                 the same from one forward pass of the prompt and the answer

The docstring of create_masks_for_vision_model (the mask of generate())
says that the tokens of an image see each other only in the layers with a
window. But the logits of generate() and of forward() both agree bit for bit
with a runtime that applies the mask in every layer (check_mm_prompt.py), as
llama.cpp does. The main set is that of generate().

    PYTHONPATH=. python scripts/hf_mm_reference.py --image ../llama.cpp/tools/mtmd/test-1.jpeg \\
        --text "What is in this image? Answer in two sentences." --out models2/mm-refs/12b-image.npz
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HUB = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub"
REPO = "models--google--gemma-4-12B-it-qat-q4_0-unquantized"
TOP = 64


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--image", action="append", default=[])
    ap.add_argument("--audio", action="append", default=[])
    ap.add_argument("--video", action="append", default=[])
    ap.add_argument("--text", required=True)
    ap.add_argument("--budget", type=int, default=280)
    ap.add_argument("--gen", type=int, default=48)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    import torch
    from transformers import AutoConfig, AutoProcessor
    from np_gemma import unified as U
    torch.set_num_threads(args.threads)
    snap = args.snapshot or sorted(glob.glob(os.path.join(HUB, REPO, "snapshots", "*")))[-1]
    proc = AutoProcessor.from_pretrained(snap)
    proc.image_processor.max_soft_tokens = args.budget
    content = []
    for path in args.video:
        # transformers decodes a path with torchcodec only; its PyAV reader
        # gives all the frames and their metadata, and the processor samples
        # them (32 frames) as it does for a path.
        content.append({"type": "video"})
    for path in args.image:
        content.append({"type": "image", "image": U.load_image(path)})
    content.append({"type": "text", "text": args.text})
    for path in args.audio:
        content.append({"type": "audio", "audio": U.load_audio(path)})
    messages = [{"role": "user", "content": content}]
    if args.video:
        from transformers.video_utils import load_video
        vids, metas = [], []
        for path in args.video:
            frames, meta = load_video(path, backend="pyav")
            vids.append(frames)
            metas.append(meta)
        text = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        kw = {}
        if args.image:
            kw["images"] = [U.load_image(p) for p in args.image]
        if args.audio:
            kw["audio"] = [U.load_audio(p) for p in args.audio]
        inputs = proc(text=text, videos=vids, video_metadata=metas, return_tensors="pt", **kw)
    else:
        inputs = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt")
    ids = inputs["input_ids"]
    print("prompt ids %d: %s" % (ids.shape[1], {k: tuple(v.shape) for k, v in inputs.items()
                                                 if hasattr(v, "shape")}), flush=True)
    t0 = time.time()
    kind = AutoConfig.from_pretrained(snap).model_type
    if kind == "gemma4_unified":
        from transformers import Gemma4UnifiedForConditionalGeneration as Cls
    else:
        from transformers import Gemma4ForConditionalGeneration as Cls
    # The default attention (sdpa): with eager the audio tower of Gemma 4
    # hides the keys that a query sees (scripts/check_mm_gemma4.py).
    model = Cls.from_pretrained(snap, dtype=getattr(torch, args.dtype)).eval()
    print("load %.0f s" % (time.time() - t0), flush=True)
    feats = {k: v for k, v in inputs.items()
             if k not in ("num_soft_tokens_per_image", "num_soft_tokens_per_video")}
    if args.video:
        print("prompt text around the video:", repr(proc.tokenizer.decode(ids[0][:60])), flush=True)
    with torch.no_grad():
        t0 = time.time()
        out = model.generate(**feats, max_new_tokens=args.gen, do_sample=False,
                             return_dict_in_generate=True, output_logits=True)
        print("generate %.0f s" % (time.time() - t0), flush=True)
        full = out.sequences[:1]
        gen_logits = torch.stack([lg[0] for lg in out.logits]).float()
        n_prompt = ids.shape[1]
        answer = full[0, n_prompt:].tolist()
        print("answer:", repr(proc.tokenizer.decode(answer)), flush=True)
        # The prompt and the answer in one pass: the logits of each answer
        # position (the last prompt position gives the first answer token).
        f2 = dict(feats)
        f2["input_ids"] = full
        f2["attention_mask"] = torch.ones_like(full)
        if "mm_token_type_ids" in f2:
            pad = torch.zeros((1, full.shape[1] - n_prompt), dtype=f2["mm_token_type_ids"].dtype)
            f2["mm_token_type_ids"] = torch.cat([f2["mm_token_type_ids"], pad], dim=1)
        t0 = time.time()
        logits = model(**f2).logits[0, n_prompt - 1:].float()
        print("forward %.0f s" % (time.time() - t0), flush=True)
        fwd_logz = torch.logsumexp(logits, dim=-1).numpy()
        fwd_val, fwd_idx = torch.topk(logits, TOP, dim=-1)
        logz = torch.logsumexp(gen_logits, dim=-1).numpy()
        val, idx = torch.topk(gen_logits, TOP, dim=-1)
        soft = []
        if "pixel_values" in feats:
            imf = model.model.get_image_features(feats["pixel_values"],
                                                 feats["image_position_ids"]).pooler_output
            soft += [t.float().numpy() for t in imf]
        if "pixel_values_videos" in feats:
            vf = model.model.get_video_features(feats["pixel_values_videos"],
                                                feats["video_position_ids"]).pooler_output
            soft += [t.float().numpy() for t in vf]
        if "input_features" in feats:
            ao = model.model.get_audio_features(feats["input_features"],
                                                feats.get("input_features_mask"))
            af = ao.pooler_output
            # The mask of the output tokens (after the subsample convs of the
            # E4B; the frames of the 12B).
            mask = getattr(ao, "attention_mask", None)
            for j in range(af.shape[0]):
                rows = af[j] if mask is None else af[j][mask[j].bool()]
                soft.append(rows.float().numpy())
    mm = inputs.get("mm_token_type_ids")
    np.savez(args.out, ids=full[0].numpy(), n_prompt=n_prompt,
             soft=np.concatenate(soft, axis=0) if soft else np.zeros((0, 3840), np.float32),
             mm_type=(mm[0].numpy() if mm is not None else np.zeros(n_prompt, np.int64)),
             logits_idx=idx.numpy(), logits_val=val.numpy(), logz=logz,
             fwd_idx=fwd_idx.numpy(), fwd_val=fwd_val.numpy(), fwd_logz=fwd_logz,
             text=args.text, images=np.array(args.image), audio=np.array(args.audio),
             videos=np.array(args.video),
             budget=args.budget, dtype=args.dtype)
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
