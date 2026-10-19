#!/usr/bin/env python3
"""The reference of Qwen3.8-27B (the dense model of the Qwen3.5 architecture:
qwen3_5, the DeltaNet and full attention 3:1 as Qwen3.6 35B A3B, a SwiGLU
MLP in place of the experts) on the CPU.

The transformers code (models/qwen3_5, Qwen3_5ForCausalLM: the text model
alone, no vision tower, no MTP layer) runs as it is, in float32 on the CPU
(108 GB). The weights come from the shards of the checkpoint
(model.language_model.* -> model.*). Reads of the shards drop their pages
(POSIX_FADV_DONTNEED).

Each prompt runs as one prompt pass, then --steps greedy decode steps (the
DeltaNet state and the attention cache of transformers). The model thinks by
default. For each prompt it writes <out>/<name>.npz:

    ids, prompt          the prompt (the model's template, the HF tokenizer)
    gen                  the greedy tokens (to EOS or --steps)
    hs                   [65, T + len(gen) - 1, 5120] float32: hs[0] the
                         embeddings, hs[i + 1] the output of layer i (the
                         last one after the final norm, as transformers
                         gives it)
    top_ids, top_vals    [T + len(gen) - 1, 64]: the top 64 logits of each row
    lse, argmax          [T + len(gen) - 1]: the log-sum-exp and the argmax
    step_logits          [1 + min(16, len(gen) - 1), vocab]: the full logits
                         of the last prompt row and of the first 16 decode rows

Row r of top_ids / lse / argmax is the prediction for token r + 1 of
ids + gen. The rows of gen run as decode steps, the rows of ids as a prompt.

    OMP_NUM_THREADS=1 taskset -c 24-43 python -I scripts/qwen38_27b_reference.py \\
        ~/models/Qwen3.8-27B --out ~/reclank-golden/qwen38_27b --threads 20

--forced FILE.npz (ids, gen: the tokens of another run, e.g. a free-running
generation of this runtime): the reference teacher-forced on ids + gen, in
chunks of 256 rows; <out>/<name of FILE>_ref.npz gets top_ids, top_vals,
lse, argmax of every row (row r predicts token r + 1).
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N_TOP = 64
N_FULL = 16


def log(msg):
    print("ref: " + msg, file=sys.stderr, flush=True)


def prompts():
    """(name, messages): real text of about 200 tokens, and a short question."""
    doc = open(os.path.join(ROOT, "docs", "MOE_PLAN.md")).read()[:600]
    doc = doc[:doc.rindex("\n")]
    u = lambda t: [{"role": "user", "content": t}]  # noqa: E731
    return [
        ("doc", u("Here is the start of a design document:\n\n" + doc + "\n\nSummarize it in two sentences.")),
        ("think", u("List the first ten prime numbers, separated by commas.")),
    ]


class Shards:
    """name -> (path, dtype, shape, start, end) of the shards; reads that
    drop their pages from the page cache."""

    def __init__(self, path):
        where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.t = {}
        self.fds = {}
        for f in sorted(set(where.values())):
            p = os.path.join(path, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            hdr.pop("__metadata__", None)
            for k, v in hdr.items():
                s, e = v["data_offsets"]
                self.t[k] = (p, v["dtype"], tuple(v["shape"]), 8 + n + s, 8 + n + e)
        self.read_bytes = 0

    def raw(self, p, off, n):
        if p not in self.fds:
            self.fds[p] = os.open(p, os.O_RDONLY)
        fd = self.fds[p]
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            got += os.preadv(fd, [mv[got:]], off + got)
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
        self.read_bytes += n
        return buf

    def f32(self, name):
        p, dt, shape, s, e = self.t[name]
        b = self.raw(p, s, e - s)
        if dt == "F32":
            return np.frombuffer(b, np.float32).reshape(shape).copy()
        assert dt == "BF16", (name, dt)
        return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)


def build(model_dir, sh):
    """Qwen3_5ForCausalLM of the text config on the meta device, then every
    weight from the shards (float32)."""
    import torch
    from torch import nn
    from transformers import AutoConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM, Qwen3_5TextRotaryEmbedding

    cfg = AutoConfig.from_pretrained(model_dir).text_config
    cfg.dtype = torch.float32
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM._from_config(cfg, dtype=torch.float32)
    model.eval()
    t0 = time.time()
    used = set()
    for n, _ in list(model.named_parameters()):
        src = n.replace("model.", "model.language_model.", 1) if n.startswith("model.") else n
        assert src in sh.t, (n, src)
        path, leaf = n.rsplit(".", 1)
        model.get_submodule(path)._parameters[leaf] = nn.Parameter(torch.from_numpy(sh.f32(src)),
                                                                   requires_grad=False)
        used.add(src)
    left = [n for n in sh.t if n not in used and not n.startswith(("mtp.", "model.visual."))]
    assert not left, left[:5]
    log("weights in %.0f s (%.1f GB read)" % (time.time() - t0, sh.read_bytes / 1e9))
    # the buffers made at init (rope) are on meta: make them again on the CPU
    model.model.rotary_emb = Qwen3_5TextRotaryEmbedding(config=cfg)
    still = [n for n, b in model.named_buffers() if b.is_meta] + [n for n, p in model.named_parameters()
                                                                    if p.is_meta]
    assert not still, still[:5]
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--only", default=None, help="names of prompts, comma separated")
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--forced", default=None, help="an npz of ids and gen: the logits of those tokens")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    sh = Shards(a.model)
    model = build(a.model, sh)
    cfg = model.config

    if a.forced:
        z = np.load(a.forced)
        toks = z["ids"].tolist() + z["gen"].tolist()
        cache = DynamicCache(config=cfg)
        rec = {k: [] for k in ("top_ids", "top_vals", "lse", "argmax")}
        t0 = time.time()
        for c0 in range(0, len(toks), 256):
            ch = toks[c0:c0 + 256]
            lg = model(input_ids=torch.tensor([ch]), past_key_values=cache, use_cache=True,
                       position_ids=torch.arange(c0, c0 + len(ch)).view(1, -1)).logits[0].float()
            tv, ti = torch.topk(lg, N_TOP, dim=-1)
            rec["top_ids"].append(ti.numpy().astype(np.int32))
            rec["top_vals"].append(tv.numpy())
            rec["lse"].append(torch.logsumexp(lg, dim=-1).numpy())
            rec["argmax"].append(lg.argmax(-1).numpy().astype(np.int32))
            log("forced: %d of %d rows (%.0f s)" % (c0 + len(ch), len(toks), time.time() - t0))
        out = os.path.join(a.out, os.path.basename(a.forced)[:-4] + "_ref.npz")
        np.savez(out, ids=z["ids"], gen=z["gen"], **{k: np.concatenate(v) for k, v in rec.items()})
        log("written %s" % out)
        return

    for name, msgs in prompts():
        if a.only and name not in a.only.split(","):
            continue
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        log("%s: %d tokens" % (name, len(ids)))
        cache = DynamicCache(config=cfg)
        rec = {k: [] for k in ("hs", "top_ids", "top_vals", "lse", "argmax")}
        full = []
        gen = []

        def run(x, pos):
            out = model(input_ids=torch.tensor([x]), past_key_values=cache, use_cache=True,
                        position_ids=torch.arange(pos, pos + len(x)).view(1, -1),
                        output_hidden_states=True)
            rec["hs"].append(np.stack([h[0].numpy() for h in out.hidden_states]))
            lg = out.logits[0].float()
            tv, ti = torch.topk(lg, N_TOP, dim=-1)
            rec["top_ids"].append(ti.numpy().astype(np.int32))
            rec["top_vals"].append(tv.numpy())
            rec["lse"].append(torch.logsumexp(lg, dim=-1).numpy())
            rec["argmax"].append(lg.argmax(-1).numpy().astype(np.int32))
            return lg[-1]

        t0 = time.time()
        last = run(ids, 0)
        full.append(last.numpy().copy())
        gen.append(int(last.argmax()))
        log("%s: prompt in %.1f s" % (name, time.time() - t0))
        t0 = time.time()
        while gen[-1] not in eos and len(gen) < a.steps:
            last = run([gen[-1]], len(ids) + len(gen) - 1)
            if len(full) <= N_FULL:
                full.append(last.numpy().copy())
            gen.append(int(last.argmax()))
            if len(gen) % 25 == 0:
                log("%s: %d tokens, %.2f s a step" % (name, len(gen), (time.time() - t0) / (len(gen) - 1)))
        out = tok.decode(gen)
        log("%s: %d tokens: %r" % (name, len(gen), out[-300:]))
        # the last token of gen has no row of its own (it was not run)
        np.savez(os.path.join(a.out, name + ".npz"), ids=np.array(ids, np.int32), gen=np.array(gen, np.int32),
                 prompt=np.array(text), text=np.array(out), hs=np.concatenate(rec["hs"], axis=1),
                 top_ids=np.concatenate(rec["top_ids"]), top_vals=np.concatenate(rec["top_vals"]),
                 lse=np.concatenate(rec["lse"]), argmax=np.concatenate(rec["argmax"]),
                 step_logits=np.stack(full))
    log("written to %s" % a.out)


if __name__ == "__main__":
    main()
