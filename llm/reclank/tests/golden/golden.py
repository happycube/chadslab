#!/usr/bin/env python3
"""Golden dumps: the tokens and the logits of each model family on each
backend, to check that a change of the code changes no numbers.

    python tests/golden/golden.py run --family g26 --backend gpu --out DIR
    python tests/golden/golden.py compare DIR_A DIR_B

A run loads one model and runs the prompt of prompt.txt (a chat turn, no
think part). It runs two passes:

- rec: the plain decode of STEPS tokens, the greedy pick on the whole
  logits of each step on the host. For each step it keeps the sha256 of
  the logits, the best 64 logits and their ids, and the whole logits of
  the first and the last step. The end tokens do not stop it.
- fast: the path of the servers (the GPU argmax of Session, or the
  generate of the Qwen engine), with MTP where the file has a drafter
  (the 26B and Qwen3.8 on the GPU), with the times of the prompt and of
  the decode. Its tokens are compared with those of rec.

The paths of the model files come from DIR/models.json, or from the file
that --models names (models.example.json shows the keys). The files of a
run are DIR/<family>-<backend>.npz and .json. compare checks the tokens and
the hashes of each pair of runs, and for a run that differs gives the first
step that differs, the largest difference of the top logits, and the KL of
the saved whole rows.

The thread count and the CPU set change the bits of some products: keep
them fixed between the runs that are compared (run_golden.sh does).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

STEPS = 64
TOPK = 64
FAMILIES = ("g26", "g12", "e4b", "q36", "q38")
QWEN_CTX = 8192
GEMMA_LEN = 4096


def prompt_messages():
    with open(os.path.join(HERE, "prompt.txt"), encoding="utf-8") as f:
        return [{"role": "user", "content": f.read()}]


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


class Rec:
    """A pick for RowPicker (and the Qwen loop) that takes the whole logits
    of each row, keeps their hash and top values, and returns the argmax.
    It is not greedy for RowPicker, so the logits come to the host."""

    greedy = False
    temperature = 0.0

    def __init__(self):
        self.hashes, self.top_ids, self.top_vals = [], [], []
        self.first = self.last = None
        self.dtype = None

    def reset(self, ids=()):
        pass

    def fixed(self, t):
        return t

    def accept(self, t):
        pass

    def __call__(self, logits):
        a = np.asarray(logits).reshape(-1)
        self.dtype = str(a.dtype)
        self.hashes.append(digest(a))
        f = a.astype(np.float32)
        idx = np.argpartition(-f, TOPK)[:TOPK]
        idx = idx[np.lexsort((idx, -f[idx]))]
        self.top_ids.append(idx.astype(np.int32))
        self.top_vals.append(f[idx])
        if self.first is None:
            self.first = f.copy()
        self.last = f.copy()
        return int(np.argmax(a))


def load_paths(args):
    p = args.models or os.path.join(args.out, "models.json")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------- Gemma 4

def run_gemma(fam, backend, paths, res):
    from reclank import Model, Tokenizer
    from reclank.models.gemma4.config import Config
    from reclank.formats.gguf import GGUF
    from reclank.models.gemma4.model import Session
    from reclank.core.sampling import Sampler

    m = paths[fam]
    t0 = time.time()
    g = GGUF(m["gguf"])
    tok = Tokenizer.from_gguf(g)
    tc = g.text_config()
    e4b = bool(tc.get("hidden_size_per_layer_input"))
    if e4b:
        from reclank.models.gemma4.e4b import E4B, E4BConfig
        model = E4B(g, E4BConfig({"text_config": tc}), mode="int4")
    else:
        model = Model(g, Config({"text_config": tc})).load_all(dtype="int4")
    if backend.startswith("gpu"):
        from reclank.models.gemma4 import gpu as gpu
        gpu.offload(model, None, resident=backend == "gpu-resident")
    res["load_s"] = time.time() - t0
    text = tok.apply_chat_template(prompt_messages(), add_generation_prompt=True,
                                   thinking=False, empty_thought_block=not e4b)
    ids = tok.encode(text)
    res["prompt_ids"] = ids

    rec = Rec()
    s = Session(model, max_len=GEMMA_LEN)
    toks = list(s.generate_stream(ids, STEPS, eos_ids=(), sampler=rec))
    res["rec"] = (toks, rec)

    drafter = None
    if backend.startswith("gpu") and m.get("mtp"):
        os.environ["RECLANK_MTP"] = "1"
        from reclank.models.gemma4.gpu import GPUDrafter
        drafter = GPUDrafter(m["mtp"], model)
    s2 = Session(model, max_len=GEMMA_LEN, drafter=drafter, n_draft=2)
    t1 = time.time()
    it = s2.generate_stream(ids, STEPS, eos_ids=(), sampler=Sampler(temperature=0.0))
    fast = [next(it)]
    t2 = time.time()
    fast += list(it)
    t3 = time.time()
    res["fast"] = fast
    res["times"] = {"prompt_s": t2 - t1, "decode_s": t3 - t2, "prompt_n": len(ids),
                    "decode_n": len(fast) - 1, "mtp": drafter is not None}
    res["mtp_stats"] = dict(getattr(s2, "mtp_stats", {}) or {})


# ---------------------------------------------------------------- Qwen

def run_qwen(fam, backend, paths, res):
    from reclank.encoders import media as MD
    from reclank.formats.gguf import GGUF
    from reclank.core.sampling import Sampler

    from reclank.models.qwen import chat as QC
    from reclank.models.qwen.engine import QwenEngine
    np.seterr(over="ignore")
    m = paths[fam]
    path = m["gguf"]
    t0 = time.time()
    meta = GGUF(path, stage=False).meta
    arch = meta.get("general.architecture")
    d = os.path.dirname(path)
    template = m.get("template") or os.path.join(d, "chat_template.jinja")
    text = None
    if not os.path.exists(template):
        text = meta["tokenizer.chat_template"]
    tok = QC.QwenTok(m.get("tokenizer") or os.path.join(d, "tokenizer.json"), template, text)
    dev = None
    if arch == "qwen35moe":
        from reclank.models.qwen.qwen35 import QwenCache, QwenGGUFProgram
        model, cache_cls = QwenGGUFProgram(path), QwenCache
        if backend == "gpu":
            from reclank.models.qwen.gpu35 import QwenGPU
            dev = QwenGPU(model, hot_gb=None)
    else:
        from reclank.models.qwen.qwen38 import Qwen4Cache, Qwen4CPU
        model, cache_cls = Qwen4CPU(path, experts=m.get("experts")), Qwen4Cache
        if backend == "gpu":
            from reclank.models.qwen.gpu38 import Qwen4GPU
            dev = Qwen4GPU(model, hot_gb=None, ctx=QWEN_CTX)
    res["load_s"] = time.time() - t0
    ids = tok.encode(tok.apply_chat_template(prompt_messages(), add_generation_prompt=True,
                                             thinking=False))
    res["prompt_ids"] = ids
    n = len(ids)

    # rec: the plain loop on the model or the device
    cache = cache_cls(model.cfg, QWEN_CTX)
    if dev is not None:
        dev.attach(cache)
    cache.set_rope(MD.mrope_positions(n, []))
    if dev is not None:
        dev.prefill(ids, pos=0)
        logits = np.asarray(dev.logits()).reshape(-1)
    else:
        h = model.forward(ids, cache, start_pos=0)
        logits = model.logits(h[-1:])[0]
    rec = Rec()
    toks, pos = [], n
    for i in range(STEPS):
        t = rec(logits)
        toks.append(t)
        if i + 1 == STEPS:
            break
        if dev is not None:
            dev.step(t, pos)
            logits = np.asarray(dev.logits()).reshape(-1)
        else:
            logits = model.logits(model.forward([t], cache, start_pos=pos))[0]
        pos += 1
    res["rec"] = (toks, rec)

    # fast: the path of the server
    be = QwenEngine(model, dev, tok, QWEN_CTX, cache_cls)
    be.think_budget = be.wrap_left = 0
    be.think_force = tok.encode(QC.THINK_CLOSE)
    be.mtp = 3 if getattr(dev, "has_mtp", False) else 0
    be._fresh()
    t1 = time.time()
    it = be.generate(list(ids), STEPS, Sampler(temperature=0.0), ())
    fast = [next(it)]
    t2 = time.time()
    fast += list(it)
    t3 = time.time()
    res["fast"] = fast
    res["times"] = {"prompt_s": t2 - t1, "decode_s": t3 - t2, "prompt_n": n,
                    "decode_n": len(fast) - 1, "mtp": be.mtp}
    res["mtp_stats"] = {"drafts": be.mtp_stats[0], "accepted": be.mtp_stats[1]}


# ---------------------------------------------------------------- run, compare

def cmd_run(args):
    paths = load_paths(args)
    if args.backend.startswith("gpu") and not args.allow_busy:
        import subprocess
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True).stdout.split()
        if used and int(used[0]) > 1000:
            sys.exit("the GPU holds %s MiB: another program runs (--allow-busy to go on)" % used[0])
    os.makedirs(args.out, exist_ok=True)
    res = {}
    t0 = time.time()
    (run_gemma if args.family.startswith(("g", "e")) else run_qwen)(args.family, args.backend, paths, res)
    toks, rec = res["rec"]
    from reclank.backends.cpu import cops
    meta = {
        "family": args.family, "backend": args.backend, "tag": args.tag,
        "cpu_arch": "vnni" if cops.VNNI else ("avx512" if cops.AVX512 else "avx2"),
        "omp_threads": os.environ.get("OMP_NUM_THREADS"),
        "cpus": sorted(os.sched_getaffinity(0)),
        "steps": STEPS, "prompt_n": len(res["prompt_ids"]), "tokens": toks,
        "fast": res["fast"], "fast_same": res["fast"] == toks[:len(res["fast"])],
        "times": res["times"], "load_s": res["load_s"], "total_s": time.time() - t0,
        "mtp_stats": res.get("mtp_stats"), "logits_dtype": rec.dtype, "hashes": rec.hashes,
    }
    base = os.path.join(args.out, "%s-%s" % (args.family, args.backend))
    np.savez(base + ".npz", prompt_ids=np.asarray(res["prompt_ids"], np.int32),
             tokens=np.asarray(toks, np.int32), fast=np.asarray(res["fast"], np.int32),
             top_ids=np.stack(rec.top_ids), top_vals=np.stack(rec.top_vals),
             first=rec.first, last=rec.last)
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=1)
    t = res["times"]
    print("%s %s: prompt %d tokens in %.2f s (%.1f tok/s); decode %d in %.2f s (%.2f tok/s)%s; "
          "fast tokens %s rec" % (
              args.family, args.backend, t["prompt_n"], t["prompt_s"], t["prompt_n"] / t["prompt_s"],
              t["decode_n"], t["decode_s"], t["decode_n"] / max(t["decode_s"], 1e-9),
              ", MTP %s" % res.get("mtp_stats") if t["mtp"] else "",
              "==" if meta["fast_same"] else "!="), flush=True)


def _kl(p_logits, q_logits):
    p = p_logits.astype(np.float64)
    q = q_logits.astype(np.float64)
    p = np.exp(p - p.max())
    p /= p.sum()
    lq = q - q.max()
    lq -= np.log(np.exp(lq).sum())
    lp = np.log(np.maximum(p, 1e-300))
    return float((p * (lp - lq)).sum())


def cmd_compare(args):
    bad = 0
    names = sorted(os.path.relpath(os.path.join(d, f[:-5]), args.a)
                   for d, _, fs in os.walk(args.a) for f in fs
                   if f.endswith(".json") and f != "models.json")
    for name in names:
        fa, fb = os.path.join(args.a, name), os.path.join(args.b, name)
        if not os.path.exists(fb + ".json"):
            print("%-18s missing in %s" % (name, args.b))
            continue
        ja, jb = json.load(open(fa + ".json")), json.load(open(fb + ".json"))
        same_ids = ja["tokens"] == jb["tokens"]
        diff = [i for i, (x, y) in enumerate(zip(ja["hashes"], jb["hashes"])) if x != y]
        same_fast = ja["fast"] == jb["fast"]
        ok = same_ids and not diff and same_fast and \
            np.array_equal(np.load(fa + ".npz")["prompt_ids"], np.load(fb + ".npz")["prompt_ids"])
        ta, tb = ja["times"], jb["times"]
        speed = "decode %.2f -> %.2f tok/s, prompt %.1f -> %.1f tok/s" % (
            ta["decode_n"] / max(ta["decode_s"], 1e-9), tb["decode_n"] / max(tb["decode_s"], 1e-9),
            ta["prompt_n"] / ta["prompt_s"], tb["prompt_n"] / tb["prompt_s"])
        if ok:
            print("%-18s same bits (%d steps, fast path too); %s" % (name, len(ja["hashes"]), speed))
            continue
        bad += 1
        na, nb = np.load(fa + ".npz"), np.load(fb + ".npz")
        msg = ["%-18s DIFFERS:" % name]
        if diff:
            i = diff[0]
            msg.append("logits differ from step %d (%d of %d steps)" % (i, len(diff), len(ja["hashes"])))
            msg.append("top-64 max |d| at that step %.3g" % float(
                np.abs(na["top_vals"][i] - nb["top_vals"][i]).max()))
        if not same_ids:
            j = next((k for k, (x, y) in enumerate(zip(ja["tokens"], jb["tokens"])) if x != y), None)
            msg.append("tokens differ from step %s" % j)
        if not same_fast:
            msg.append("the fast path tokens differ")
        msg.append("KL first %.3g last %.3g" % (_kl(na["first"], nb["first"]), _kl(na["last"], nb["last"])))
        msg.append(speed)
        print(" ".join(msg))
    print("%d of %d runs differ" % (bad, len(names)))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--family", choices=FAMILIES, required=True)
    r.add_argument("--backend", choices=("cpu", "gpu", "gpu-resident"), required=True,
                   help="gpu-resident: every expert on the GPU (Gemma; the 26B on a 24 GB card)")
    r.add_argument("--out", required=True)
    r.add_argument("--models", default=None, help="the JSON of the model paths (default OUT/models.json)")
    r.add_argument("--tag", default="")
    r.add_argument("--allow-busy", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_run(args)
        return 0
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
