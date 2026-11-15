#!/usr/bin/env python3
"""Check reclank's EmbeddingGemma 2 against the reference of
scripts/embedgemma2_reference.py: the token ids of each input (the reclank
tokenizer, the prompt), the token rows before the mean, and the vector
(cosine with that of sentence-transformers). An input whose ids differ is
checked on the reference ids too. An audio input runs on the samples of the
reference: its ids, its audio rows (embed_audio), and its vector; an image
input the same (embed_vision).

    python scripts/check_embedgemma2.py --model models/embeddinggemma-2 \\
        --inputs texts.jsonl --refs refs/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.models.embedgemma2.model import EmbeddingGemma2  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--min-cos", type=float, default=0.99999)
    ap.add_argument("--gpu", action="store_true", help="the program on the GPU (BF12)")
    ap.add_argument("--audio", default=None, help="the device of the audio tower (gpu, cpu)")
    ap.add_argument("--vision", default=None, help="the device of the vision tower (gpu, cpu)")
    a = ap.parse_args()
    m = EmbeddingGemma2(a.model, gpu=a.gpu, audio=a.audio, vision=a.vision)
    print("devices:", m.devices, flush=True)
    bad = 0
    for line in open(a.inputs):
        item = json.loads(line)
        r = np.load(os.path.join(a.refs, item["name"] + ".npz"))
        if "audio" in item or "image" in item:
            t0 = time.time()
            if "audio" in item:
                ids, soft = m.audio_input(r["samples"])
            else:
                ids, soft = m.image_input(item["image"], int(item.get("budget", 280)))
            y, tok = m.embed_ids([ids], tokens=True, soft=[soft])
            dt = time.time() - t0
            same = list(r["ids"]) == ids
            cos = float(y[0] @ r["emb"])
            sc = (soft * r["soft"]).sum(1) / np.linalg.norm(soft, axis=1) / np.linalg.norm(r["soft"], axis=1) \
                if soft.shape == r["soft"].shape else np.array([np.nan])
            ok = same and cos >= a.min_cos
            bad += not ok
            print("%-12s %5d tokens  ids %s  cos %.7f  media rows min cos %.6f  %.2f s  %s" % (
                item["name"], len(r["ids"]), "same" if same else "DIFFER (%d vs %d)" % (len(ids), len(r["ids"])),
                cos, sc.min(), dt, "ok" if ok else "FAIL"), flush=True)
            continue
        ids = m.tokens(item["text"], item.get("prompt"))
        same = list(r["ids"]) == ids
        t0 = time.time()
        y, tok = m.embed_ids([[int(i) for i in r["ids"]]], tokens=True)
        dt = time.time() - t0
        cos = float(y[0] @ r["emb"])
        tc = (tok * r["tok"]).sum(1) / np.linalg.norm(tok, axis=1) / np.linalg.norm(r["tok"], axis=1)
        ok = same and cos >= a.min_cos
        bad += not ok
        print("%-12s %5d tokens  ids %s  cos %.7f  token rows min cos %.6f  max |d| %.2e  %.2f s  %s" % (
            item["name"], len(r["ids"]), "same" if same else "DIFFER (%d vs %d)" % (len(ids), len(r["ids"])),
            cos, tc.min(), np.abs(tok - r["tok"]).max(), dt, "ok" if ok else "FAIL"), flush=True)
    print("FAIL %d" % bad if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
