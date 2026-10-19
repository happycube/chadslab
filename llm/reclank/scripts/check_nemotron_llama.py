#!/usr/bin/env python3
"""Nemotron 3.5 Bolt against llama.cpp (the reference test of docs/NEMOTRON_BOLT_PLAN.md): the
per-token distributions of llama-perplexity --kl-divergence-base (the log-softmax of positions
n_ctx/2 .. n_ctx-2 of each chunk, uint16 with a scale and a floor) against the engine's on the same
tokens (each chunk from a fresh cache; the GGUF adds no BOS).

    llama-perplexity -m NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB.gguf -f TEXT -c 512 --chunks 4 \\
        --kl-divergence-base base.kld          # (CUDA_VISIBLE_DEVICES= : the CPU backend)
    python scripts/check_nemotron_llama.py MODEL_DIR base.kld [cpu | gpu HOT_GB]

The GGUF differs from the safetensors checkpoint in its embedding (Q8_0, the checkpoint's bfloat16)
and both quantize activations their own way: the KL is small, not zero.
"""
from __future__ import annotations

import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def read_base(path):
    f = open(path, "rb")
    assert f.read(8) == b"_logits_", "not a llama-perplexity logits file"
    n_ctx, = struct.unpack("<i", f.read(4))
    V, nch = struct.unpack("<ii", f.read(8))
    toks = np.frombuffer(f.read(4 * nch * n_ctx), np.int32).reshape(nch, n_ctx)
    nv, npos = 2 * ((V + 1) // 2) + 4, n_ctx - 1 - n_ctx // 2

    def chunk():
        raw = np.frombuffer(f.read(2 * nv * npos), np.uint16).reshape(npos, nv)
        sc = raw[:, :4].copy().view(np.float32)              # scale, the floor's log-prob
        return sc[:, 1:2] + sc[:, 0:1] * raw[:, 4:4 + V].astype(np.float32)
    return n_ctx, toks, chunk


def main():
    model, base = sys.argv[1], sys.argv[2]
    which = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    n_ctx, toks, chunk = read_base(base)
    first, npos = n_ctx // 2, n_ctx - 1 - n_ctx // 2
    from reclank.models.nemotron.kq import NemotronKQ
    m = NemotronKQ(model)
    if which == "gpu":
        from reclank.models.nemotron import gpu as G
        g = G.NemotronGPU(m, ctx=n_ctx + 64, hot_gb=float(sys.argv[4]), counts=G.load_counts(model))

        def run(ids):
            g.reset()
            return np.stack([g.prefill(list(ids[:1]))] + [g.step(t) for t in ids[1:]])
    else:
        def run(ids):
            return m.forward(ids, m.cache(n_ctx + 8))
    kls, nll_l, nll_m, agree = [], [], [], []
    for c in range(toks.shape[0]):
        lp_ref = chunk()
        lg = run(list(toks[c]))[first:first + npos].astype(np.float64)
        lp = lg - lg.max(-1, keepdims=True)
        lp -= np.log(np.exp(lp).sum(-1, keepdims=True))
        # KL as llama-perplexity --kl-divergence takes it: the base tokens above -16 only (the rest
        # sit on the floor of the file's quantization), not renormalized
        live = lp_ref > -16.0
        kls.append(np.sum(np.where(live, np.exp(lp_ref) * (lp_ref - lp), 0.0), -1))
        nxt = toks[c][first + 1:first + 1 + npos]
        nll_l.append(-lp_ref[np.arange(npos), nxt])
        nll_m.append(-lp[np.arange(npos), nxt])
        agree.append(lp_ref.argmax(-1) == lp.argmax(-1))
        print("chunk %d: KL mean %.4f, max %.3f; top-1 agree %.1f%%" % (
            c, kls[-1].mean(), kls[-1].max(), 100 * agree[-1].mean()), flush=True)
    kl = np.concatenate(kls)
    print("llama.cpp PPL %.4f, the engine (%s) PPL %.4f; KL mean %.4f median %.4f p99 %.4f; top-1 agree %.1f%%" % (
        np.exp(np.concatenate(nll_l).mean()), which, np.exp(np.concatenate(nll_m).mean()), kl.mean(),
        np.median(kl), np.percentile(kl, 99), 100 * np.concatenate(agree).mean()))


if __name__ == "__main__":
    main()
