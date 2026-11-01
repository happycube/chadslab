#!/usr/bin/env python3
"""The quality of EmbeddingGemma 2 with its matrices in a quantized form:
the vectors of the real inputs of scripts/embedgemma2_reference.py (texts,
audio clips, images) against those of sentence-transformers (float32), and
against reclank's own bfloat16 run.

Each form puts the true values of its matrices (quantized, then decoded,
in float32) in every matrix that a GPU program would read: the layers of
the text model (and its PLE), of the audio tower and of the vision tower.
The table of the token embeddings, the projection to 768 and the patch
linear stay bfloat16. The model then runs its NumPy layers
(RECLANK_ENC_PY=1: forward_numpy, the NumPy encoders), so the numbers are
those of the weights alone, with float32 activations.

    bf16     the checkpoint
    bf12     BF12 rows (encoders.runtime.to_bf12: what --gpu runs)
    q8_0     Q8_0 (an fp16 scale for each 32 values)
    rq8_0    RQ8_0: Q8_0 of the rows rotated in each 32 values (the TQ6
             rotation, formats/tq6.py; the runtime rotates x instead)

    RECLANK_ENC_PY=1 python scripts/study_embedgemma2_quant.py \\
        --model models/embeddinggemma-2 --inputs all.jsonl --refs refs/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ["RECLANK_ENC_PY"] = "1"      # before reclank: the NumPy layers

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.backends.cpu import cops  # noqa: E402
from reclank.core import ops  # noqa: E402
from reclank.formats.gguf import BF12  # noqa: E402
from reclank.models.embedgemma2.model import EmbeddingGemma2  # noqa: E402

FORMS = ("bf16", "bf12", "q8_0", "rq8_0")


def decode(w16, form):
    """The float32 true values (rows, cols) of bfloat16 rows w16 in a form,
    and the error in dB against them."""
    f = ops.bf16_to_f32(w16)
    rows, cols = f.shape
    ids = np.arange(rows)
    if form == "bf16":
        back = f
    elif form == "bf12":
        data, rep = cops.kq_bf16_to_bf12(np.ascontiguousarray(w16), cols)
        back = cops.kq_rows(data.reshape(-1), BF12, cols, ids)
    elif form == "q8_0":
        back = cops.kq_rows(cops.kq_to_q8_0(f, cols), cops.KQ_Q8_0, cols, ids)
    elif form == "rq8_0":
        q = cops.kq_to_q8_0(cops.tq6_rotate(f), cols)
        back = cops.tq6_rotate(cops.kq_rows(q, cops.KQ_Q8_0, cols, ids), inverse=True)
    else:
        raise ValueError(form)
    d = back.astype(np.float64) - f
    db = 10 * np.log10(max((d * d).sum() / (f.astype(np.float64) ** 2).sum(), 1e-30))
    return np.ascontiguousarray(back, dtype=np.float32), db


def apply(m, form):
    """Put the true values of form in the matrices; return the mean error in
    dB of each tower."""
    errs = {"text": [], "audio": [], "vision": []}
    text = list(m.ple_lin) + [b[k] for b in m.blk
                              for k in ("q", "k", "v", "o", "gate", "up", "down", "ple_gate", "ple_proj")]
    for lin in text:
        w16 = lin.bf16 if hasattr(lin, "bf16") else lin.w
        lin.bf16 = w16
        back, db = decode(w16, form)
        lin.f32t = np.ascontiguousarray(back.T)       # forward_numpy reads this
        errs["text"].append(db)
    A, V = m.audio, m.vision
    alin = [A.in_proj, A.out_proj, A.proj]
    for b in A.blk:
        alin += [b[k] for k in ("q", "k", "v", "o", "rel", "pw1", "pw2")]
        alin += [b["ff1"][1], b["ff1"][2], b["ff2"][1], b["ff2"][2]]
    vlin = [b[k] for b in V.blk for k in ("q", "k", "v", "o", "gate", "up", "down")] + [V.proj]
    for tower, lins in (("audio", alin), ("vision", vlin)):
        for lin in lins:
            w16 = lin.bf16 if hasattr(lin, "bf16") else lin.w
            lin.bf16 = w16
            back, db = decode(w16, form)
            lin.w = back                                # _Linear: x @ w.T for float32 w
            errs[tower].append(db)
    return {k: float(np.mean(v)) for k, v in errs.items()}


def run(m, item, r):
    if "audio" in item:
        ids, soft = m.audio_input(r["samples"])
        return m.embed_ids([ids], soft=[soft])[0]
    if "image" in item:
        ids, soft = m.image_input(item["image"], int(item.get("budget", 280)))
        return m.embed_ids([ids], soft=[soft])[0]
    return m.embed_ids([[int(i) for i in r["ids"]]])[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--forms", default=",".join(FORMS))
    a = ap.parse_args()
    m = EmbeddingGemma2(a.model, gpu=False, audio="cpu", vision="cpu")
    items = [json.loads(line) for line in open(a.inputs)]
    refs = {it["name"]: np.load(os.path.join(a.refs, it["name"] + ".npz")) for it in items}
    vec = {}
    for form in a.forms.split(","):
        t0 = time.time()
        err = apply(m, form)
        vec[form] = {it["name"]: run(m, it, refs[it["name"]]) for it in items}
        print("%-6s weights: text %.1f dB, audio %.1f dB, vision %.1f dB (%.0f s)" % (
            form, err["text"], err["audio"], err["vision"], time.time() - t0), flush=True)
    kinds = lambda it: "audio" if "audio" in it else "image" if "image" in it else "text"  # noqa: E731
    print("\n1 - cos, in millionths (ppm): against sentence-transformers / against reclank bf16")
    print("%-13s %-6s" % ("input", "kind") + "".join("%18s" % f for f in vec))
    worst = {f: {} for f in vec}
    for it in items:
        n = it["name"]
        row = "%-13s %-6s" % (n, kinds(it))
        for f in vec:
            ref_c = 1 - float(vec[f][n] @ refs[n]["emb"])
            own_c = 1 - float(vec[f][n] @ vec["bf16"][n]) if "bf16" in vec else float("nan")
            row += "%11.1f /%5.1f" % (ref_c * 1e6, own_c * 1e6)
            w = worst[f].setdefault(kinds(it), [0.0, 0.0])
            w[0], w[1] = max(w[0], ref_c), max(w[1], own_c)
        print(row)
    print("\nthe worst of each kind (ppm, against sentence-transformers / against bf16):")
    for f in vec:
        print("  %-6s " % f + "   ".join("%s %.1f / %.1f" % (k, v[0] * 1e6, v[1] * 1e6)
                                         for k, v in sorted(worst[f].items())))


if __name__ == "__main__":
    main()
