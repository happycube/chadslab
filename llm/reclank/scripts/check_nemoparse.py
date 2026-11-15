"""Check reclank's Nemotron 3.5 Parse (reclank/models/nemoparse) against the checkpoint's own
transformers code in float32 on the CPU (the sandbox driver nparse_hf.py: a .npz, _logits.npy and
.json a page) on real document pages.

    check_nemoparse.py MODEL_DIR REF_DIR PAGES_DIR [--gpu] [--threads N] [--pages a,b] [--no-generate]

A page (REF_DIR/<name>.npz, PAGES_DIR/<name>.png):
  - pixels: the processor's packed patches against ours (model.page_pixels), the worst difference;
  - encoder: the visual rows of the reference's patches, the cosine (the whole and the worst row);
  - decoder: the reference's tokens a step at a time after the reference's visual rows: KL(reference ||
    reclank) of each step's softmax (mean and worst), the steps whose most likely token differs;
  - generate: greedy from the page file (our pixels, our encoder), the tokens and the text against
    the reference's, and the speed (the encoder's seconds, the decoder's tokens/s, pages/s).
The last line: PASS when every greedy output is the reference's, else FAIL."""
import argparse
import glob
import json
import os
import time

import numpy as np

from reclank.models.nemoparse.engine import ParseEngine
from reclank.models.nemoparse.model import page_pixels

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("refs")
ap.add_argument("pages")
ap.add_argument("--gpu", action="store_true")
ap.add_argument("--threads", type=int, default=None)
ap.add_argument("--pages-only", dest="only", default=None, help="the names to check (a,b)")
ap.add_argument("--no-generate", action="store_true")
a = ap.parse_args()
names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(a.refs, "*.npz")))
if a.only:
    names = [n for n in names if n in a.only.split(",")]
t0 = time.time()
e = ParseEngine(a.model, gpu=a.gpu, cpu_threads=a.threads)
print("loaded in %.1f s (%s)" % (time.time() - t0, "GPU" if a.gpu else "CPU"), flush=True)


def kl(ref, x):
    """KL(softmax(ref) || softmax(x)) of each row, in float64."""
    ref, x = ref.astype(np.float64), x.astype(np.float64)
    lp = ref - ref.max(-1, keepdims=True)
    lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
    lq = x - x.max(-1, keepdims=True)
    lq -= np.log(np.exp(lq).sum(-1, keepdims=True))
    return (np.exp(lp) * (lp - lq)).sum(-1)


def cos(x, y):
    x, y = x.astype(np.float64), y.astype(np.float64)
    whole = float((x * y).sum() / np.linalg.norm(x) / np.linalg.norm(y))
    rows = (x * y).sum(-1) / np.linalg.norm(x, axis=-1) / np.linalg.norm(y, axis=-1)
    return whole, float(rows.min())


same_all, kls, tops, steps, enc_s, dec_s, dec_t, pages_s = True, [], 0, 0, 0.0, 0.0, 0, 0.0
worst_cos = 1.0
for name in names:
    d = np.load(os.path.join(a.refs, name + ".npz"))
    ref = json.load(open(os.path.join(a.refs, name + ".json")))
    L = np.load(os.path.join(a.refs, name + "_logits.npy"), mmap_mode="r")
    page = os.path.join(a.pages, name + ".png")
    patches, hw, content = page_pixels(page, e.cfg)
    assert list(hw) == list(d["image_sizes"][0]) and list(content) == list(d["content_sizes"][0]), (hw, content)
    px = float(np.abs(patches - d["pixel_values"]).max())
    vis = e.encode_patches(d["pixel_values"], hw, content)
    c_all, c_row = cos(vis, d["vision"])
    worst_cos = min(worst_cos, c_row)
    prompt = [int(t) for t in d["input_ids"]]
    gold = [int(t) for t in d["output_ids"]]
    logs = e.generate(d["vision"], prompt, forced=gold)[1][:len(gold)]
    k = np.concatenate([kl(np.asarray(L[i:i + 64]), np.stack(logs[i:i + 64])) for i in range(0, len(gold), 64)])
    top = int(sum(int(np.argmax(L[i])) == int(np.argmax(logs[i])) for i in range(len(gold))))
    kls.append(k)
    tops += top
    steps += len(gold)
    line = "%-10s %4d x %4d, %4d rows: pixels %.1e | encoder cos %.9f (worst row %.9f) | " \
        "KL mean %.2e max %.2e, top-1 %d/%d" % (name, hw[1], hw[0], vis.shape[0], px, c_all, c_row,
                                                  k.mean(), k.max(), top, len(gold))
    if not a.no_generate:
        r = e.parse(page)
        out = r["ids"]
        same = out == gold
        same_all &= same
        n0 = next((i for i, (x, y) in enumerate(zip(out, gold)) if x != y), min(len(out), len(gold)))
        text = r["text"] == ref["raw"]
        enc_s += r["encode_s"]
        dec_s += r["decode_s"]
        dec_t += r["tokens"]
        pages_s += r["encode_s"] + r["decode_s"]
        line += " | greedy %d tokens %s%s, text %s | encode %.2f s, %d tokens in %.2f s (%.1f tok/s, %d steps)" % (
            len(out), "same" if same else "DIFFERENT", "" if same else " from token %d of %d" % (n0, len(gold)),
            "same" if text else "different", r["encode_s"], r["tokens"], r["decode_s"], r["tokens"] / r["decode_s"],
            r["steps"])
    print(line, flush=True)
k = np.concatenate(kls)
print("all: %d pages, %d steps: worst encoder row cos %.9f, KL mean %.2e max %.2e, top-1 %.4f" % (
    len(names), steps, worst_cos, k.mean(), k.max(), tops / steps))
if not a.no_generate:
    print("speed: encoder %.2f s a page, decoder %.1f tokens/s, %.3f pages/s (%d tokens a page)" % (
        enc_s / len(names), dec_t / dec_s, len(names) / pages_s, dec_t // len(names)))
    print("PASS" if same_all else "FAIL")
