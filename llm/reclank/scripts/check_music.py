#!/usr/bin/env python3
"""Check reclank's music embeddings (docs/MUSIC_EMBED_PLAN.md) against the
references of scripts/music_reference.py: for each text, the token ids of
each tower, the rows of its last layer, and its vector (cosine with the
reference's); for each clip of audio, MuQ's mel and the conformer's last
hidden state, CLAP's mel, and each tower's vector.

    python scripts/check_music.py --clap models/music/clap \\
        --muq models/music/muq-mulan --inputs music.jsonl --refs refs/ [--gpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.models.music.clap import ClapAudio, mel as clap_mel  # noqa: E402
from reclank.models.music.muq import MuqAudio, mel as muq_mel  # noqa: E402
from reclank.models.music.text import ClapText, MuqText  # noqa: E402


def cos_rows(a, b):
    return (a * b).sum(-1) / np.linalg.norm(a, axis=-1) / np.linalg.norm(b, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clap", required=True)
    ap.add_argument("--muq", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--min-cos", type=float, default=0.99999)
    ap.add_argument("--gpu", action="store_true", help="the towers on the GPU")
    ap.add_argument("--batch", type=int, default=1, help="texts packed in one run (the vectors must not change)")
    ap.add_argument("--towers", default="clap-text,muq-text,muq-audio,clap-audio")
    a = ap.parse_args()
    want = set(a.towers.split(","))
    t0 = time.time()
    towers = {}
    if "clap-text" in want:
        towers["clap"] = ClapText(a.clap, gpu=a.gpu)
    if "muq-text" in want:
        towers["muq"] = MuqText(a.muq, gpu=a.gpu)
    print("loaded in %.1f s (%s)" % (time.time() - t0, "gpu" if a.gpu else "cpu"), flush=True)
    every = [json.loads(line) for line in open(a.inputs)]
    items = [it for it in every if "text" in it]
    refs = {it["name"]: np.load(os.path.join(a.refs, it["name"] + ".npz")) for it in items}
    bad = 0
    worst = {k: 1.0 for k in towers}
    for name, m in towers.items():
        t0 = time.time()
        for i0 in range(0, len(items), a.batch):
            group = items[i0:i0 + a.batch]
            ids = [m.tokens(it["text"]) for it in group]
            y, rows = m.embed_ids([[int(t) for t in refs[it["name"]][name + "_ids"]] for it in group], tokens=True)
            at = 0
            for j, it in enumerate(group):
                r = refs[it["name"]]
                same = ids[j] == [int(t) for t in r[name + "_ids"]]
                n = len(r[name + "_ids"])
                tc = cos_rows(rows[at:at + n], r[name + "_tok"]).min()
                at += n
                cos = float(y[j] @ r[name + "_emb"])
                worst[name] = min(worst[name], cos)
                ok = same and cos >= a.min_cos
                bad += not ok
                if not ok or a.batch == 1 and i0 < 3:
                    print("%-5s %-9s %3d tokens  ids %s  cos %.8f  rows min cos %.7f  %s" % (
                        name, it["name"], n, "same" if same else "DIFFER", cos, tc, "ok" if ok else "FAIL"),
                        flush=True)
        print("%-5s %d texts, worst cos %.8f (1 - cos %.2e), %.2f s" % (
            name, len(items), worst[name], 1 - worst[name], time.time() - t0), flush=True)
    clips = [it for it in every if "audio" in it]
    if "muq-audio" in want and clips:
        A = MuqAudio(a.muq, gpu=a.gpu)
        worst = 1.0
        t0 = time.time()
        for it in clips:
            r = np.load(os.path.join(a.refs, it["name"] + ".npz"))
            dm = np.abs(muq_mel(r["s24"]) - r["muq_mel"]).max()
            t1 = time.time()
            y, hs = A.embed([r["s24"]], hidden=True)
            dt = time.time() - t1
            hc = cos_rows(hs[0], r["muq_hid"]).min()
            cos = float(y[0] @ r["muq_emb"])
            worst = min(worst, cos)
            ok = cos >= a.min_cos and dm < 1e-3
            bad += not ok
            print("muq-audio %-7s mel max |d| %.1e  hidden rows min cos %.7f  cos %.8f  %.2f s  %s" % (
                it["name"], dm, hc, cos, dt, "ok" if ok else "FAIL"), flush=True)
        print("muq-audio %d clips, worst cos %.8f (1 - cos %.2e), %.1f s" % (
            len(clips), worst, 1 - worst, time.time() - t0), flush=True)
    if "clap-audio" in want and clips:
        A = ClapAudio(a.clap, gpu=a.gpu)
        worst = 1.0
        t0 = time.time()
        for it in clips:
            r = np.load(os.path.join(a.refs, it["name"] + ".npz"))
            dm = np.abs(clap_mel(r["s48"]) - r["clap_mel"]).max()
            t1 = time.time()
            y = A.embed([r["s48"]])
            dt = time.time() - t1
            cos = float(y[0] @ r["clap_emb"])
            worst = min(worst, cos)
            ok = cos >= a.min_cos and dm < 1e-3
            bad += not ok
            print("clap-audio %-7s mel max |d| %.1e  cos %.8f  %.2f s  %s" % (
                it["name"], dm, cos, dt, "ok" if ok else "FAIL"), flush=True)
        print("clap-audio %d clips, worst cos %.8f (1 - cos %.2e), %.1f s" % (
            len(clips), worst, 1 - worst, time.time() - t0), flush=True)
    print("FAIL %d" % bad if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
