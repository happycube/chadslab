#!/usr/bin/env python3
"""Check reclank's Nemotron 3.5 Embed (1B, 8B) against the reference of
scripts/nemotron_embed_reference.py: the token ids of each input (the
template), the token rows, and the vector (cosine with the reference's). An
image input runs twice: on the reference's pixel values (the model alone:
its rows of the [IMG] tokens and the vector) and on reclank's own pixels (PIL's
resize: the whole path). --bench then times the texts in one packed pass and
each image alone.

    python scripts/check_nemotron_embed.py --model DIR --inputs embed.jsonl --refs refs/ [--gpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.models.nemotron_embed.model import NemotronRetriever, load_image  # noqa: E402


def cos_rows(a, b):
    return (a * b).sum(1) / np.linalg.norm(a, axis=1) / np.linalg.norm(b, axis=1)


def inputs_of(m, item, image_dir=None):
    """(ids, soft rows or None, seconds of the image) of an input with reclank's own pixels."""
    t0 = time.time()
    rows, grid = None, None
    if item.get("image"):
        path = item["image"] if image_dir is None else os.path.join(image_dir, os.path.basename(item["image"]))
        rows, grid = m.image_rows(load_image(path), item.get("longest_edge"))
    dt = time.time() - t0
    if m.kind == "rerank":
        return m.pair_ids(item["query"], item.get("text"), grid), rows, dt
    if item.get("kind") == "query":
        return m.query_ids(item["text"]), rows, dt
    return m.document_ids(item.get("text"), grid), rows, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--min-cos", type=float, default=0.99999, help="the vector on the reference's ids and pixels")
    ap.add_argument("--min-cos-own", type=float, default=0.9999, help="an image's vector with reclank's pixels")
    ap.add_argument("--gpu", action="store_true", help="the text model on the GPU (BF12)")
    ap.add_argument("--vision", default=None, help="the device of the vision tower (gpu, cpu)")
    ap.add_argument("--cpu-threads", type=int, default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--only", default=None, help="names, comma-separated")
    ap.add_argument("--bench", action="store_true", help="then the times of a packed pass and of each image")
    a = ap.parse_args()
    t0 = time.time()
    m = NemotronRetriever(a.model, gpu=a.gpu, vision=a.vision, cpu_threads=a.cpu_threads)
    print("%s: %s, devices %s, loaded in %.1f s" % (os.path.basename(a.model.rstrip("/")), m.kind, m.devices,
                                                    time.time() - t0), flush=True)
    only = set(a.only.split(",")) if a.only else None
    items = [json.loads(line) for line in open(a.inputs)]
    items = [it for it in items if not only or it["name"] in only]
    bad = 0
    worst = {}
    for item in items:
        r = np.load(os.path.join(a.refs, item["name"] + ".npz"))
        ref_ids = [int(i) for i in r["ids"]]
        kind = "image" if item.get("image") and item.get("text") is None else \
            "image+text" if item.get("image") else item.get("kind", "text")
        soft = None
        t1 = time.time()
        if "pixels" in r:
            soft, _ = m.pixel_rows(r["pixels"])
        y, tok = m.pooled([ref_ids], [soft], tokens=True)
        dt = time.time() - t1
        y = y[0]
        cos = float(y @ r["emb"] / np.linalg.norm(y))
        tc = cos_rows(tok, r["tok"])
        ok = cos >= a.min_cos
        extra = ""
        ids, own_soft, dti = inputs_of(m, item, a.image_dir)
        same = ids == ref_ids
        ok = ok and same
        if soft is not None:
            sc = cos_rows(soft, r["soft"])
            y2 = m.pooled([ids], [own_soft])[0]
            cos2 = float(y2 @ r["emb"] / np.linalg.norm(y2))
            ok = ok and cos2 >= a.min_cos_own
            extra = "  image rows min cos %.6f  own pixels: cos %.7f (image %.2f s)" % (sc.min(), cos2, dti)
            worst[kind + " own"] = min(worst.get(kind + " own", 1.0), cos2)
        worst[kind] = min(worst.get(kind, 1.0), cos)
        bad += not ok
        print("%-14s %5d tokens  ids %s  cos %.7f  token rows min cos %.6f  %.2f s%s  %s" % (
            item["name"], len(ref_ids), "same" if same else "DIFFER (%d vs %d)" % (len(ids), len(ref_ids)),
            cos, tc.min(), dt, extra, "ok" if ok else "FAIL"), flush=True)
    print("worst cos by input type: " + ", ".join("%s %.7f" % kv for kv in sorted(worst.items())))
    if a.bench:
        bench(m, items, a.image_dir)
    print("FAIL %d" % bad if bad else "PASS")
    return 1 if bad else 0


def bench(m, items, image_dir, batch=8192):
    """The texts by length in passes of at most batch tokens, as the server (twice: the first builds
    the programs), then each image."""
    seqs = sorted((inputs_of(m, it)[0] for it in items if not it.get("image")), key=len)
    passes, cur = [], []
    for s in seqs:
        if cur and sum(map(len, cur)) + len(s) > batch:
            passes.append(cur)
            cur = []
        cur.append(s)
    passes.append(cur)
    n = sum(len(s) for s in seqs)
    for k in range(2):
        t0 = time.time()
        for p in passes:
            m.pooled(p)
        dt = time.time() - t0
    print("bench: %d texts, %d tokens in %d passes: %.2f s, %.0f tokens/s" % (
        len(seqs), n, len(passes), dt, n / dt), flush=True)
    for it in items:
        if not it.get("image"):
            continue
        for k in range(2):
            t0 = time.time()
            ids, rows, dti = inputs_of(m, it, image_dir)
            t1 = time.time()
            m.pooled([ids], [rows])
            t2 = time.time()
        print("bench: %-14s %5d tokens: image %.2f s, text model %.2f s, total %.2f s" % (
            it["name"], len(ids), t1 - t0, t2 - t1, t2 - t0), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
