#!/usr/bin/env python3
"""Check reclank's Nemotron 3.5 Rerank against the reference of
scripts/nemotron_embed_reference.py --rerank (the checkpoint's model.py): the
token ids of each pair (the template), the pooled row (cosine), and the score
(the logit: its difference). A pair with an image runs on the reference's
pixel values and on reclank's own (PIL's resize). Then, for each query (the
names <query>_<document>), the order of its documents by score against the
reference's: Kendall's tau and the same first document.

    python scripts/check_nemotron_rerank.py --model DIR --inputs rerank.jsonl --refs refs/ [--gpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from check_nemotron_embed import inputs_of  # noqa: E402
from reclank.models.nemotron_embed.model import NemotronRetriever  # noqa: E402


def kendall(a, b):
    """Kendall's tau of two lists of scores (no ties)."""
    n, s = len(a), 0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return s / (n * (n - 1) / 2) if n > 1 else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--max-diff", type=float, default=2e-3, help="the score on the reference's ids and pixels")
    ap.add_argument("--max-diff-own", type=float, default=2e-2, help="a score with reclank's pixels")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--vision", default=None)
    ap.add_argument("--cpu-threads", type=int, default=None)
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    t0 = time.time()
    m = NemotronRetriever(a.model, gpu=a.gpu, vision=a.vision, cpu_threads=a.cpu_threads)
    assert m.kind == "rerank", "not a reranker: %s" % a.model
    print("%s: devices %s, loaded in %.1f s" % (os.path.basename(a.model.rstrip("/")), m.devices, time.time() - t0),
          flush=True)
    only = set(a.only.split(",")) if a.only else None
    items = [json.loads(line) for line in open(a.inputs)]
    items = [it for it in items if not only or it["name"] in only]
    bad = 0
    by_query = {}
    worst = {}
    for item in items:
        r = np.load(os.path.join(a.refs, item["name"] + ".npz"))
        ref_ids = [int(i) for i in r["ids"]]
        kind = "image" if item.get("image") and item.get("text") is None else \
            "image+text" if item.get("image") else "text"
        soft = None
        t1 = time.time()
        if "pixels" in r:
            soft, _ = m.pixel_rows(r["pixels"])
        pooled = m.pooled([ref_ids], [soft])[0]
        score = float(pooled @ m.score_w)
        dt = time.time() - t1
        ref = float(r["score"])
        pc = float(pooled @ r["pooled"] / np.linalg.norm(pooled) / np.linalg.norm(r["pooled"]))
        ids, own_soft, dti = inputs_of(m, item, a.image_dir)
        same = ids == ref_ids
        ok = same and abs(score - ref) <= a.max_diff
        own = score
        extra = ""
        if soft is not None:
            own = float(m.score_ids([ids], [own_soft])[0])
            ok = ok and abs(own - ref) <= a.max_diff_own
            extra = "  own pixels: score %.6f (d %+.2e)" % (own, own - ref)
            worst[kind + " own"] = max(worst.get(kind + " own", 0.0), abs(own - ref))
        worst[kind] = max(worst.get(kind, 0.0), abs(score - ref))
        bad += not ok
        q = item["name"].split("_", 1)[0]
        by_query.setdefault(q, []).append((item["name"], ref, score, own))
        print("%-12s %5d tokens  ids %s  score %.6f ref %.6f (d %+.2e)  pooled cos %.7f  %.2f s%s  %s" % (
            item["name"], len(ref_ids), "same" if same else "DIFFER (%d vs %d)" % (len(ids), len(ref_ids)),
            score, ref, score - ref, pc, dt, extra, "ok" if ok else "FAIL"), flush=True)
    print("worst |score - ref| by document type: " + ", ".join("%s %.2e" % kv for kv in sorted(worst.items())))
    for q, rows in by_query.items():
        ref = [x[1] for x in rows]
        for k, label in ((2, "reference ids and pixels"), (3, "own pixels")):
            mine = [x[k] for x in rows]
            tau = kendall(ref, mine)
            top = int(np.argmax(ref)) == int(np.argmax(mine))
            print("query %-5s %2d documents, %s: Kendall tau %.3f, first %s (%s)" % (
                q, len(rows), label, tau, "same" if top else "DIFFERS", rows[int(np.argmax(mine))][0]))
    print("FAIL %d" % bad if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
