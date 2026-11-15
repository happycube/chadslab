#!/usr/bin/env python3
"""DiffusionGemma's image prompt on the GPU (BF12) against transformers' float32.

Each reference (an .npz: transformers' ids and mm_token_type_ids of one image prompt, its
pixel_values and image_position_ids, its soft rows of get_image_features, the canvas c1 and the
decoder's logits for it after the encoder over the prompt with the vision mask of generate(), l1,
and with the plain causal mask, l1_causal) and its image budget:

- the engine's prompt (DGemmaEngine.prompt_ids: the chat template, the image's soft tokens) has
  transformers' ids, and its spans are the image tokens of mm_token_type_ids;
- the soft rows of --mmproj's tower on transformers' pixels against transformers' soft rows, and
  on the engine's own pixels (the resize: PIL's bicubic, as transformers' PIL processor;
  transformers' default torchvision one puts a few hundred values one level apart);
- DGemmaGPU.encode with the spans, then the decoder's pass on c1 (no self-conditioning,
  temperature 1: the processed logits are the logits): the KL over the vocabulary against l1 and
  against l1_causal, and the argmax agreement; with the rows of transformers' pixels, then with
  the engine's own.

    python scripts/check_dgemma_image.py MODELS/diffusiongemma-26B-A4B-it-BF12-GGUF \\
        --mmproj MODELS/diffusiongemma-26B-A4B-it-BF12-GGUF/diffusiongemma-26B-A4B-it-mmproj.gguf \\
        --image chart.jpg hf_image_140.npz:140 hf_image_280.npz:280 hf_image_560.npz:560

PASS: the same ids and spans; on transformers' pixels the soft rows within 1e-4 and the KL against
l1 below 1e-3 and below the KL against l1_causal; the engine's own soft rows within 5e-3. Prints
PASS or FAIL (exit 1)."""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.dgemma import sampler as S  # noqa: E402
from reclank.models.dgemma.engine import DGemmaEngine  # noqa: E402
from reclank.models.dgemma.gpu import DGemmaGPU  # noqa: E402
from reclank.models.gemma4.encoders import load_embedder  # noqa: E402
from reclank.models.gemma4.tokenizer import Tokenizer  # noqa: E402
from server.media import MediaSource  # noqa: E402

PROMPT = "Describe this image in one sentence."


def kl(ref, got):
    """The mean KL(ref || got) of rows of logits (float64), and the argmax agreement."""
    a, b = ref.astype(np.float64), got.astype(np.float64)
    la = a - a.max(-1, keepdims=True)
    la -= np.log(np.exp(la).sum(-1, keepdims=True))
    lb = b - b.max(-1, keepdims=True)
    lb -= np.log(np.exp(lb).sum(-1, keepdims=True))
    return float((np.exp(la) * (la - lb)).sum(-1).mean()), float((a.argmax(-1) == b.argmax(-1)).mean())


def rel_rows(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b)) if a.shape == b.shape else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--mmproj", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("refs", nargs="+", help="REF.npz:BUDGET")
    a = ap.parse_args()
    d = a.model if os.path.isdir(a.model) else os.path.dirname(a.model)
    t0 = time.time()
    m = DGemmaGPU(a.model)
    tok = Tokenizer(os.path.join(d, "tokenizer.json"))
    s = S.Settings.from_config(json.load(open(os.path.join(d, "generation_config.json"))),
                               json.load(open(os.path.join(d, "config.json"))))
    eng = DGemmaEngine(m, tok, s, m.max_len)
    eng.embedder = load_embedder(a.mmproj, gpu=False)
    print("load %.0f s" % (time.time() - t0), flush=True)
    with open(a.image, "rb") as f:
        uri = "data:image/jpeg;base64," + base64.b64encode(f.read()).decode()
    ok = True
    for spec in a.refs:
        path, budget = spec.rsplit(":", 1)
        h = np.load(path)
        eng.image_budget = int(budget)
        msgs = [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": uri}},
                                             {"type": "text", "text": PROMPT}]}]
        p = eng.prompt_ids(msgs, False, source=MediaSource())
        ids = np.asarray(p, np.int64)
        same_ids = ids.shape == h["ids"].shape and np.array_equal(ids, h["ids"])
        img = np.flatnonzero(h["mm"] == 1)
        sp = p.spans
        same_span = len(sp) == 1 and sp[0].start == img[0] and sp[0].end == img[-1] + 1 and \
            len(img) == sp[0].end - sp[0].start
        own = np.concatenate([x.rows for x in sp]) if sp else np.zeros((0, m.H), np.float32)
        r_own = rel_rows(own, h["soft"])
        # the tower on transformers' pixels (the patches past the image's are padding)
        keep = h["image_position_ids"][0][:, 0] >= 0
        tower = eng.embedder.vision.encode(np.ascontiguousarray(h["pixel_values"][0][keep], np.float32),
                                           np.ascontiguousarray(h["image_position_ids"][0][keep], np.int64))
        r_tow = rel_rows(tower, h["soft"])
        res = {}
        for tag, rows in (("theirs", tower), ("own", own)):
            if not same_span or rows.shape != h["soft"].shape:
                res[tag] = (float("inf"),) * 4
                continue
            sp[0].rows = np.ascontiguousarray(rows, np.float32)
            m.reset()
            m.encode(ids, spans=sp)
            st = m.steps()
            st(h["c1"], 1.0, np.zeros(m.C, bool), np.full(m.C, 0.5, np.float32))
            m.mirror.buffer_of(m.z).download(m.z)
            res[tag] = kl(h["l1"], m.z) + kl(h["l1_causal"], m.z)
        k1, t1, kc, tc = res["theirs"]
        good = same_ids and same_span and r_tow < 1e-4 and r_own < 5e-3 and k1 < 1e-3 and k1 < kc
        ok &= good
        big = any(x.end - x.start > m.C for x in sp)
        print("budget %s: %d tokens, an image of %d rows (blocks of %d); ids the same %s, span the same %s%s" % (
            budget, len(ids), len(img), m.P if big else m.C, same_ids, same_span, "" if good else "  FAIL"))
        print("  on transformers' pixels: soft rows %.1e; l1 KL %.4f top1 %.4f; against the causal mask's KL "
              "%.4f top1 %.4f" % (r_tow, k1, t1, kc, tc))
        print("  on the engine's pixels:  soft rows %.1e; l1 KL %.4f top1 %.4f; against the causal mask's KL "
              "%.4f top1 %.4f" % ((r_own,) + res["own"]), flush=True)
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
