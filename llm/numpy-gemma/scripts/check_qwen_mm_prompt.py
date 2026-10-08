#!/usr/bin/env python3
"""Check a chat prompt with an image on Qwen3.6 or Qwen3.8, CPU against GPU.

The encoder has its own check (scripts/check_mm_qwen.py, against
transformers). This script checks the text side of an image prompt: the
rows of the image in place of the embeddings of <|image_pad|>, and the
M-RoPE positions (QwenCache.set_rope). The same prompt runs:

1. on the CPU program (QwenGGUFProgram or Qwen4CPU): the logits of the last
   token of the prompt;
2. on the GPU (QwenGPU or Qwen4GPU): the logits of the same token. Pass: the
   KL of the two below --tol, and the same top token;
3. the greedy answer on the GPU, and the answer to a second question after
   it (the text after the image has positions shifted by the image).

    OMP_NUM_THREADS=18 OPENBLAS_NUM_THREADS=1 PYTHONPATH=. \\
        python scripts/check_qwen_mm_prompt.py --model 36 [--image PATH] [--tokens 60]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import media as MD  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402
from np_gemma.vision_qwen import QwenEmbedder  # noqa: E402

MODELS = {
    "36": dict(gguf="models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
               vision="models/Qwen3.6-35B-A3B-GGUF/mmproj-BF16.gguf",
               tok="models/Qwen3.6-35B-A3B-GGUF/tokenizer.json"),
    "38": dict(gguf="models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf",
               vision="models/Qwen3.8-Flash-Next-NVFP4",
               tok="models2/Qwen3.8-Flash-Next-NVFP4-GGUF/tokenizer.json"),
}
IMAGE = "../llama.cpp/tools/mtmd/test-1.jpeg"
USER = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
EOS = (248044, 248046)


def kl(p_logits, q_logits):
    """KL(p || q) of two logit rows."""
    def logp(x):
        x = x.astype(np.float64)
        x = x - x.max()
        return x - np.log(np.exp(x).sum())
    lp, lq = logp(p_logits), logp(q_logits)
    return float((np.exp(lp) * (lp - lq)).sum())


def load(which, gpu):
    m = MODELS[which]
    if which == "36":
        from np_gemma.qwen import QwenCache, QwenGGUFProgram
        model, cache_cls = QwenGGUFProgram(m["gguf"]), QwenCache
        dev = None
        if gpu:
            from np_gemma.qwen_gpu import QwenGPU
            dev = QwenGPU(model)
    else:
        from np_gemma.qwen4 import Qwen4Cache, Qwen4CPU
        model, cache_cls = Qwen4CPU(m["gguf"]), Qwen4Cache
        dev = None
        if gpu:
            from np_gemma.qwen4_gpu import Qwen4GPU
            dev = Qwen4GPU(model)
    return model, cache_cls, dev


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=tuple(MODELS), default="36")
    ap.add_argument("--image", default=IMAGE)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--no-cpu", action="store_true", help="skip the CPU pass")
    args = ap.parse_args()
    m = MODELS[args.model]
    tok = QwenTokenizer(m["tok"])
    model, cache_cls, dev = load(args.model, gpu=True)
    emb = QwenEmbedder(m["vision"], gpu=True)
    t0 = time.time()
    rows, grid = emb.image(args.image, args.budget)
    rows, grid = emb.image(args.image, args.budget)
    print("image: %dx%d tokens, rows %s, encoder %.2f s" % (grid[0], grid[1], rows.shape,
                                                          (time.time() - t0) / 2))
    text = USER % ("<|vision_start|><|image_pad|><|vision_end|>"
                   "What is in this image? Answer in two sentences.")
    ids, spans = MD.expand_qwen(tok.encode(text), [MD.Media("image", rows, "img", grid=grid)])
    n = len(ids)
    rpos = MD.mrope_positions(n, spans)
    print("prompt: %d tokens; the text after the image starts at rope position %d (row %d)"
          % (n, rpos[0, spans[0].end], spans[0].end))
    good = True
    cpu_logits = None
    if not args.no_cpu:
        cache = cache_cls(model.cfg, n + 8)
        cache.set_rope(rpos)
        t0 = time.time()
        h = model.forward(ids, cache, 0, media=spans)
        cpu_logits = model.logits(h[-1:])[0]
        print("CPU prompt %.1f s; top %d %r" % (time.time() - t0, int(cpu_logits.argmax()),
                                                tok.decode([int(cpu_logits.argmax())])))
    cache = cache_cls(model.cfg, n + 2 * args.tokens + 64)
    dev.attach(cache)
    cache.set_rope(rpos)
    t0 = time.time()
    dev.prefill(ids, 0, media=spans)
    logits = np.asarray(dev.logits()).reshape(-1)
    print("GPU prompt %.2f s; top %d %r" % (time.time() - t0, int(logits.argmax()),
                                            tok.decode([int(logits.argmax())])))
    if cpu_logits is not None:
        d = kl(cpu_logits, logits)
        same = int(cpu_logits.argmax()) == int(logits.argmax())
        print("KL(CPU || GPU) %.4f; same top token: %s" % (d, same))
        good &= d < args.tol and same

    def answer(logits, pos, limit):
        out = []
        t0 = time.time()
        while len(out) < limit:
            t = int(np.argmax(logits))
            out.append(t)
            if t in EOS:
                break
            dev.step(t, pos)
            logits = np.asarray(dev.logits()).reshape(-1)
            pos += 1
        return out, pos, time.time() - t0

    out, pos, dt = answer(logits, n, args.tokens)
    print("answer (%d tokens, %.1f tok/s): %r" % (len(out), len(out) / dt, tok.decode(out)))
    # A second turn after the answer: the rows of the cache stay, and the new
    # text continues from the positions after the image.
    follow = [248046] if out[-1] not in EOS else []
    follow += tok.encode("\n" + USER % "What year is it about? Answer with the year only.")
    ids2 = ids + out + follow
    cache.set_rope(MD.mrope_positions(len(ids2), spans))
    dev.prefill(follow, pos, media=None)
    logits = np.asarray(dev.logits()).reshape(-1)
    out2, _pos, _dt = answer(logits, pos + len(follow), 16)
    print("second turn: %r" % tok.decode(out2))
    good &= "1969" in tok.decode(out2) or not args.image.endswith("test-1.jpeg")
    print("RESULT", "PASS" if good else "FAIL")
    return 0 if good else 1


if __name__ == "__main__":
    np.seterr(over="ignore")
    raise SystemExit(main())
