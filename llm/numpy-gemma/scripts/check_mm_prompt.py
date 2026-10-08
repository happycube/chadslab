#!/usr/bin/env python3
"""Check a prompt with an image or audio against the transformers reference.

The reference (.npz of scripts/hf_mm_reference.py) has the ids of a chat
prompt with its soft tokens, the soft rows, a greedy answer, and the top 64
logits of each answer position. This script runs the same ids in the
runtime with the soft rows as media spans (np_gemma/media.py), forces the
tokens of the answer, and compares the logits of each position:

- KL over the top 64 of the reference (the reference probabilities
  against ours on the same ids);
- the top token (ours against the reference);
- max |d| of the logits over the top 64.

--source st runs the float32 weights of the same safetensors (the same
weights as the reference); --source gguf runs the int4 GGUF. --soft ours
makes the soft rows with np_gemma/unified.py from the media files (not the
rows of the reference).

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/check_mm_prompt.py models2/mm-refs/12b-image.npz

The video test clip (9 s: three images of llama.cpp, 3 s each):

    S='scale=640:480:force_original_aspect_ratio=decrease,pad=640:480:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=24'
    ffmpeg -loop 1 -t 3 -i ../llama.cpp/tools/mtmd/test-1.jpeg \
        -loop 1 -t 3 -i ../llama.cpp/media/matmul.png \
        -loop 1 -t 3 -i ../llama.cpp/media/llama1-logo.png \
        -filter_complex "[0]$S[a];[1]$S[b];[2]$S[c];[a][b][c]concat=n=3:v=1:a=0,format=yuv420p" \
        -c:v libx264 -crf 20 models2/mm-refs/slideshow.mp4
    python scripts/hf_mm_reference.py --video models2/mm-refs/slideshow.mp4 \
        --text "Describe what this video shows, in order." --gen 64 \
        --out models2/mm-refs/12b-video.npz

The results (test-1.jpeg at 280 tokens, an answer of 40):
float32 from the safetensors, the mask of an image in every layer: KL 0,
logits max |d| 0.000, against generate() and forward(). The mask only in
the layers with a window: KL 0.008. No mask: KL 0.77. The int4 GGUF with
our soft rows: KL 0.013, 40/40 (the int4 weights on a text prompt: 0.014).
That was the file of Google, which is not on the grid of the QAT release. The
Unsloth file (the default now) gives KL 0.00055, 40/40.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if "--gpu" in sys.argv:
    # before np_gemma.model reads it
    os.environ["NP_GEMMA_GPU"] = "1"

from np_gemma import media as MD  # noqa: E402
from np_gemma import unified as U  # noqa: E402

GGUF_12B = "models2/gemma-4-12B-unsloth-UD-Q4_K_XL/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf"
MMPROJ_12B = "models/gemma-4-12B-qat-q4_0/mmproj-gemma-4-12b-it-qat-q4_0.gguf"
MODELS = {
    "12b": (GGUF_12B, MMPROJ_12B),
    "e4b": ("models2/gemma-4-E4B-unsloth-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf",
            "models2/gemma-4-E4B-qat-gguf/gemma-4-E4B-it-mmproj.gguf"),
    "26b": ("models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf",
            "models2/gemma-4-26B-qat-gguf/gemma-4-26B-it-mmproj.gguf"),
}
HUB = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub"


def is_e4b(path):
    """True for an E2B or E4B GGUF (inputs for each layer)."""
    from np_gemma.gguf import GGUF
    return any(".per_layer_" in n or "per_layer_token_embd" in n for n in GGUF(path).tensors)
REPO = "models--google--gemma-4-12B-it-qat-q4_0-unquantized"


def load_model(source, dtype, model_kind="12b", gguf_path=None):
    if model_kind == "e4b":
        from np_gemma.e4b import E4B, E4BConfig
        from np_gemma.gguf import GGUF
        g = GGUF(gguf_path)
        m = E4B(g, E4BConfig({"text_config": g.text_config()}), mode="int4")
        return m, m.cfg
    if source == "gguf":
        from np_gemma.config import Config
        from np_gemma.gguf import GGUF
        from np_gemma.model import Model
        g = GGUF(gguf_path or GGUF_12B)
        cfg = Config({"text_config": g.text_config()})
        return Model(g, cfg).load_all(dtype=dtype), cfg
    from np_gemma import Config, Model, SafeTensors
    snap = sorted(glob.glob(os.path.join(HUB, REPO, "snapshots", "*")))[-1]
    cfg = Config.load(os.path.join(snap, "config.json"))
    st = SafeTensors(os.path.join(snap, "model.safetensors"))
    return Model(st, cfg).load_all(dtype=dtype), cfg


def spans_of(ids, soft, mm_type):
    """The runs of soft tokens in ids, with the rows of soft in order."""
    spans, j, i = [], 0, 0
    ids = list(ids)
    while i < len(ids):
        if ids[i] in (MD.IMAGE_TOKEN, MD.AUDIO_TOKEN, MD.VIDEO_TOKEN):
            k = i
            while k < len(ids) and ids[k] == ids[i]:
                k += 1
            kind = {MD.IMAGE_TOKEN: "image", MD.AUDIO_TOKEN: "audio"}.get(ids[i], "video")
            spans.append(MD.Span(i, soft[j:j + (k - i)], kind != "audio", kind))
            j += k - i
            i = k
        else:
            i += 1
    if j != soft.shape[0]:
        raise SystemExit("the reference has %d soft rows, the ids %d soft tokens"
                         % (soft.shape[0], j))
    return spans


def our_prompt(ref, gguf_path=GGUF_12B, mmproj=MMPROJ_12B, gpu=False, q8=False):
    """Return the PromptIds of the reference prompt as the server makes it."""
    import base64
    import mimetypes
    from np_gemma.gguf import GGUF
    from np_gemma.server import Backend
    from np_gemma.tokenizer import Tokenizer

    def uri(path):
        mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
        return "data:%s;base64,%s" % (mime, base64.b64encode(open(path, "rb").read()).decode())

    from np_gemma.gemma4_encoders import load_embedder
    tok = Tokenizer.from_gguf(GGUF(gguf_path))
    b = Backend(None, tok, empty_thought_block=not is_e4b(gguf_path))
    b.embedder = load_embedder(mmproj, gpu=gpu, q8=q8)
    b.image_budget = int(ref["budget"])
    content = []
    for path in (ref["videos"] if "videos" in ref.files else []):
        content.append({"type": "video_url", "video_url": {"url": uri(str(path))}})
    for path in ref["images"]:
        content.append({"type": "image_url", "image_url": {"url": uri(str(path))}})
    content.append({"type": "text", "text": str(ref["text"])})
    for path in ref["audio"]:
        data = base64.b64encode(open(str(path), "rb").read()).decode()
        content.append({"type": "input_audio", "input_audio": {"data": data}})
    return b.prompt_ids([{"role": "user", "content": content}], thinking=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ref")
    ap.add_argument("--source", choices=("st", "gguf"), default="st")
    ap.add_argument("--model", choices=("12b", "e4b", "26b"), default="12b")
    ap.add_argument("--gguf", default=None, help="the text GGUF (the default of --model)")
    ap.add_argument("--mmproj", default=None, help="the mmproj GGUF (the default of --model)")
    ap.add_argument("--dtype", default=None, help="f32 for st, int4 for gguf (the defaults)")
    ap.add_argument("--soft", choices=("ref", "ours"), default="ref")
    ap.add_argument("--kl", type=float, default=None,
                    help="the most mean KL (default: 1e-4 for st, 0.03 for gguf; the int4 "
                         "weights alone give about 0.014 on a text prompt)")
    ap.add_argument("--agree", type=float, default=None,
                    help="the least top-token agreement (default: 0.99 for st, 0.9 for gguf)")
    ap.add_argument("--set", choices=("generate", "forward"), default="generate",
                    help="the logits of HF generate() or of one forward() pass")
    ap.add_argument("--gpu", action="store_true",
                    help="the model on the GPU (the dense weights): the prompt pass with "
                         "the media on the GPU, then the answer in groups of 3 (as MTP)")
    ap.add_argument("--q8", action="store_true", help="Q8_0 weights in the encoders (--soft ours)")
    ap.add_argument("--enc-cpu", action="store_true", help="the encoders on the CPU with --gpu")
    ap.add_argument("--window-only", action="store_true",
                    help="the mask of an image only in the layers with a window "
                         "(NP_GEMMA_BIDIR_ALL=0)")
    ap.add_argument("--no-bidir", action="store_true",
                    help="causal image tokens (to see the effect of the mask)")
    args = ap.parse_args()
    gguf_path = args.gguf or MODELS[args.model][0]
    mmproj = args.mmproj or MODELS[args.model][1]
    if args.model != "12b":
        args.source = "gguf"
    ref = np.load(args.ref, allow_pickle=False)
    ids = [int(x) for x in ref["ids"]]
    n_prompt = int(ref["n_prompt"])
    soft = ref["soft"].astype(np.float32)
    spans = spans_of(ids, soft, ref["mm_type"])
    if args.soft == "ours":
        # The prompt of the server: the chat template, the media parts as
        # data URIs, the embedder, and the soft tokens (Backend.prompt_ids).
        prompt = our_prompt(ref, gguf_path, mmproj, args.gpu and not args.enc_cpu, args.q8)
        same = list(prompt) == ids[:n_prompt]
        mine = np.concatenate([sp.rows for sp in prompt.spans], axis=0)
        print("our prompt ids equal the reference: %s (%d and %d); soft rows rel %.2e" % (
            same, len(prompt), n_prompt,
            np.abs(mine - soft).max() / np.abs(soft).max() if mine.shape == soft.shape
            else float("nan")))
        if not same:
            raise SystemExit("FAIL: the prompt ids differ")
        spans = prompt.spans
    if args.no_bidir:
        for sp in spans:
            sp.bidir = False
    if args.window_only:
        from np_gemma import model as M
        M._BIDIR_ALL = False
    dtype = args.dtype or ("f32" if args.source == "st" else "int4")
    t0 = time.time()
    model, cfg = load_model(args.source, dtype, args.model, gguf_path)
    if args.gpu:
        from np_gemma import gpu
        gpu.offload(model, 0.0)
    print("load %s %s: %.0f s; prompt %d ids, %d spans (%s), answer %d" % (
        args.source, dtype, time.time() - t0, n_prompt, len(spans),
        ", ".join("%s %d at %d" % (sp.kind, sp.rows.shape[0], sp.start) for sp in spans),
        len(ids) - n_prompt), flush=True)
    from np_gemma.model import KVCache
    make = getattr(model, "new_cache", None)
    cache = make(len(ids) + 16) if make else KVCache(cfg, max_len=len(ids) + 16)
    t0 = time.time()
    if args.gpu:
        model.prefill(ids[:n_prompt], cache, 0, media=spans)
        t1 = time.time()
        rows = [model.logits(model._gpu_xn[-1:])[0]]
        # The answer in groups of 3, as the verify groups of MTP (a small
        # group on the GPU), and the rest in steps.
        j = n_prompt
        while j < len(ids) - 1:
            n = 3 if len(ids) - 1 - j >= 3 else 1
            x = model.forward(ids[j:j + n], cache=cache, start_pos=j)
            rows.extend(model.logits(x[-n:]))
            j += n
        lg = np.stack(rows).astype(np.float64)
        print("prompt %.2f s, answer %.2f s" % (t1 - t0, time.time() - t1), flush=True)
    else:
        model.prefill(ids[:n_prompt - 1], cache, 0, media=spans)
        x = model.forward(ids[n_prompt - 1:], cache=cache, start_pos=n_prompt - 1)
        print("prompt and answer: %.1f s" % (time.time() - t0), flush=True)
        lg = model.logits(x).astype(np.float64)
    if args.set == "forward":
        idx, val, logz = ref["fwd_idx"], ref["fwd_val"].astype(np.float64), ref["fwd_logz"]
    else:
        idx, val, logz = ref["logits_idx"], ref["logits_val"].astype(np.float64), ref["logz"]
    n = min(lg.shape[0], idx.shape[0])
    kls, agree, worst = [], 0, 0.0
    for j in range(n):
        p = np.exp(val[j] - logz[j])
        m = lg[j].max()
        lq = lg[j] - (m + np.log(np.exp(lg[j] - m).sum()))
        kls.append(float((p * (val[j] - logz[j] - lq[idx[j]])).sum()))
        agree += int(np.argmax(lg[j]) == idx[j][0])
        worst = max(worst, float(np.abs(lg[j][idx[j]] - val[j]).max()))
    kl = float(np.mean(kls))
    rate = agree / n
    kl_max = args.kl if args.kl is not None else (1e-4 if args.source == "st" else 0.03)
    agree_min = args.agree if args.agree is not None else (0.99 if args.source == "st" else 0.9)
    ok = kl <= kl_max and rate >= agree_min
    print("positions %d: mean KL (top 64) %.5f, max %.5f; top token %d/%d (%.1f%%); "
          "logits max |d| %.3f" % (n, kl, max(kls), agree, n, 100 * rate, worst))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
