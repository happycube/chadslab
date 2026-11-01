"""Check reclank's umT5 encoder (models/t5) against transformers' (scripts/umt5_reference.py) on
the reference's prompts.

    check_umt5.py TEXT_ENCODER_DIR ref.npz [--threads 12] [--study study.npz] [--speed]

The tokenizer: the ids of each prompt and of each tokenizer text of the reference. The relative
position buckets of every distance. Each prompt alone: the embeddings and the hidden state after
each block (1 - cos of each token row, the largest difference relative to the largest value),
then the pipeline's prompt_embeds (the token rows and the zero rows after them). The prompts as
one program: the rows of each prompt against its program alone (and encode's groups), and a program of max_len rows (the
pipeline's) against the rows of the longest prompt. --study: an npz of prompt_embeds of the same
prompts ("fox", "waves", "neg": an earlier dump of diffusers' encode_prompt) against reclank's.
--speed: two prompts of max_len tokens (the long prompt repeated; full rows), and a prompt and
the negative prompt as the pipeline's CFG pair; the memory (RSS) after the load and its peak."""
import argparse
import resource
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("ref")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--study")
ap.add_argument("--speed", action="store_true")
a = ap.parse_args()


def rss():
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            return int(line.split()[1]) / 1e6
    return 0.0


def cmp(name, x, ref):
    """1 - cos of each row, and the largest difference relative to the reference's largest value."""
    x, ref = np.asarray(x, np.float64).reshape(len(x), -1), np.asarray(ref, np.float64).reshape(len(ref), -1)
    rel = np.abs(x - ref).max() / max(np.abs(ref).max(), 1e-30)
    nx, nr = np.linalg.norm(x, axis=-1), np.linalg.norm(ref, axis=-1)
    live = nr > 0
    d = 1 - (x * ref).sum(-1)[live] / np.maximum(nx[live] * nr[live], 1e-300)
    zeros = "" if live.all() else ", zero rows %s" % ("same" if np.all(x[~live] == 0) else "DIFFER")
    print("  %-10s 1-cos worst %.2e (row %d) mean %.2e  rel %.2e%s" % (
        name, d.max(), np.flatnonzero(live)[d.argmax()], d.mean(), rel, zeros), flush=True)
    return d.max(), rel


r = dict(np.load(a.ref))
from reclank.models.t5.encoder import UMT5Encoder, buckets  # noqa: E402
from reclank.models.t5.tokenizer import prompt_clean  # noqa: E402

t0 = time.time()
enc = UMT5Encoder(a.model, cpu_threads=a.threads)
print("loaded in %.1f s, RSS %.2f GB" % (time.time() - t0, rss()), flush=True)
max_len = len(r["ids0"])
prompts = [str(r["prompt%d" % i]) for i in range(len([k for k in r if k.startswith("prompt")]))]

print("tokenizer")
bad = 0
for k in sorted((k for k in r if k.startswith("toktext")), key=lambda k: int(k[7:])):
    ids = np.zeros(max_len, np.int32)
    t = enc.tok.encode(prompt_clean(str(r[k])), max_len)
    ids[:len(t)] = t
    same = np.array_equal(ids, r["tok" + k[7:]])
    bad += not same
    print("  text %-3s %3d tokens %s" % (k[7:], len(t), "same" if same else "DIFFER"))
b = buckets(np.arange(-(max_len - 1), max_len), enc.nb, enc.maxd)
print("  buckets of %d distances %s" % (len(b), "same" if np.array_equal(b, r["buckets"]) else "DIFFER"))

worst = {}
singles = []
for i, p in enumerate(prompts):
    ids = enc.tokenize([p], max_len)[0]
    n = len(ids)
    ref_ids = r["ids%d" % i][:int(r["mask%d" % i].sum())]
    print("prompt %d (%d tokens): ids %s, %r" % (i, n, "same" if np.array_equal(ids, ref_ids) else "DIFFER", p[:60]))
    h = r["h%d" % i]
    cmp("embed", enc.embed(ids), h[0])
    t1 = time.time()
    y, taps = enc.run([ids], taps=True)
    dt = time.time() - t1
    for j in range(len(enc.layers)):
        if j in (0, 5, 11, 17, len(enc.layers) - 1):
            cmp("block %d" % j, taps[j, 0, :n], h[j + 1])
    emb, lens = enc.encode([p], max_len)
    singles.append(emb[0])
    worst[i] = cmp("emb", emb[0], r["emb%d" % i])
    print("  %.2f s (%d rows, with taps)" % (dt, y.shape[1]), flush=True)

ids = enc.tokenize(prompts, max_len)
rows = -(-max(len(t) for t in ids) // 16) * 16
print("the %d prompts in one program of %d rows each" % (len(prompts), rows))
t1 = time.time()
y = enc.run(ids, rows)
print("  %.2f s" % (time.time() - t1))
for i, t in enumerate(ids):
    x = y[i, :len(t)]
    same = np.array_equal(x, singles[i][:len(t)])
    print("  prompt %d: %s" % (i, "the bits of its program alone" if same else "differs from alone"), end="")
    if not same:
        print(" by %.2e (rel)" % (np.abs(x - singles[i][:len(t)]).max() / np.abs(singles[i]).max()), end="")
    print()
t1 = time.time()
emb, lens = enc.encode(prompts, max_len)
print("  encode (the prompts of the same rows together) %.2f s: %s" % (
    time.time() - t1, "the bits of each alone" if np.array_equal(emb, np.stack(singles)) else "DIFFERS from alone"))
t1 = time.time()
full, _ = enc.encode(prompts[:2], max_len, full=True)
print("  full %d rows (prompts 0, 1): %.2f s" % (max_len, time.time() - t1))
for i in range(2):
    cmp("full %d" % i, full[i], r["emb%d" % i])
    print("    against the trimmed program: max diff %.2e" % np.abs(full[i] - singles[i]).max())

if a.study:
    s = dict(np.load(a.study))
    from_ref = {"fox": 0, "waves": 1, "neg": 2}
    print("study", a.study)
    for k, i in from_ref.items():
        if k in s and i < len(prompts):
            cmp(k, singles[i], s[k])
            cmp(k + " ref", r["emb%d" % i], s[k])

if a.speed:
    long_ids = [enc.tok.encode(" ".join([prompt_clean(prompts[-1])] * 4), max_len)] * 2
    assert all(len(t) == max_len for t in long_ids)
    for rep in range(3):
        t1 = time.time()
        enc.run(long_ids, max_len)
        print("speed: 2 prompts of %d tokens %.2f s" % (max_len, time.time() - t1), flush=True)
    pair = [prompts[0], prompts[2]] if len(prompts) > 2 else prompts[:2]
    for rep in range(3):
        t1 = time.time()
        enc.encode(pair, max_len)
        print("speed: the prompt and the negative prompt (trimmed rows) %.2f s" % (time.time() - t1), flush=True)
    t1 = time.time()
    enc.encode(pair, max_len, full=True)
    print("speed: the same pair in %d rows each %.2f s" % (max_len, time.time() - t1), flush=True)
print("RSS %.2f GB, peak %.2f GB" % (rss(), resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))
print("worst prompt_embeds 1-cos %.2e, rel %.2e; tokenizer texts differing: %d" % (
    max(w[0] for w in worst.values()), max(w[1] for w in worst.values()), bad))
