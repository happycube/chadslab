#!/usr/bin/env python3
"""The reference of the DSpark drafter of GLM-5.3-Flash (RedHatAI/GLM-5.3-Flash-speculator.dspark-
preview; P7 of docs/GLM53_PLAN.md), replayed on a trace of the reference (scripts/glm53_reference.py
--trace-prefill N --trace-decode M: the streams after each layer, h<i> [rows, 4, 4096]).

The drafter as speculators' DSpark runs it (as scripts/minicpm_dspark_reference.py), in float32 torch
on the CPU, with its own embed_tokens and lm_head:

- the context: fc (6 x 4096 -> 4096) of the hidden states of the target layers 20, 28, 32, 36, 40,
  44 of each position, concatenated in that order, then hidden_norm; for each of the 5 layers k_proj
  and v_proj of it, k_norm of each head, the RoPE (neox, theta 1e4, the 64 dims) at the context
  positions (a sliding window of 2048: the last 2048 positions);
- the block: [the anchor token, 7 x mask_token_id] at the 8 positions from the anchor's, each layer
  a Qwen3 decoder layer (64 heads of 64, q/k norm) whose attention reads the context K/V and the
  block's own (no causal mask); then norm and lm_head;
- the Markov head, in order: logits[i] += markov_w2 @ markov_w1[out[i]], out[i + 1] = argmax (out[0]
  the anchor); the confidence sigmoid(proj(cat(hidden, markov_w1[out[i]]))).

What the code leaves open, each measured (the variant the drafter was trained on accepts far more):

    layer    "out": the hidden state of target layer L is the streams after it (h<L>); "in": before
             it (h<L - 1>)
    streams  "mean": the 4 streams collapsed by their mean (the sum gives the same: hidden_norm
             follows fc); "s0": stream 0

The anchor at index p of S = ids + gen (the greedy chain of the reference) has the context of the
positions 0 .. p - 1; its 8 drafts count while they match S[p + 1 ..]. Reported: the accepted drafts
of every anchor in gen, the tokens per verify step of a greedy run (anchor += accepted + 1), the
first-draft rate, and the Brier score of the confidence.

    taskset -c 40-47 python -I scripts/glm53_dspark_reference.py \\
        ~/reclank-golden/glm53/trace_256_256.npz --out ~/reclank-golden/glm53/dspark
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

DRAFT = "/home/cpage/models/GLM-5.3-Flash-speculator.dspark-preview"


def log(msg):
    print("dspark: " + msg, file=sys.stderr, flush=True)


def st_header(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    h.pop("__metadata__", None)
    return n, h


def st_read(p, n, h, k):
    import torch
    s, e = h[k]["data_offsets"]
    assert h[k]["dtype"] == "BF16"
    with open(p, "rb") as f:
        f.seek(8 + n + s)
        u = np.frombuffer(f.read(e - s), np.uint16)
    return torch.from_numpy((u.astype(np.uint32) << 16).view(np.float32).reshape(h[k]["shape"]).copy())


class Drafter:
    def __init__(self, path=DRAFT):
        import torch
        self.c = json.load(open(os.path.join(path, "config.json")))
        tc = self.c["transformer_layer_config"]
        self.targets = self.c["aux_hidden_state_layer_ids"]
        self.mask = self.c["mask_token_id"]
        self.bs = self.c["block_size"]
        self.nh, self.nkv, self.hd = tc["num_attention_heads"], tc["num_key_value_heads"], tc["head_dim"]
        self.eps = tc["rms_norm_eps"]
        self.L = tc["num_hidden_layers"]
        self.window = tc["sliding_window"]
        p = os.path.join(path, "model.safetensors")
        n, h = st_header(p)
        self.w = {k: st_read(p, n, h, k) for k in h}
        theta = tc["rope_parameters"]["rope_theta"]
        self.inv = 1.0 / (theta ** (torch.arange(0, self.hd, 2, dtype=torch.float64) / self.hd))

    def rms(self, x, w):
        import torch
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * w

    def rope(self, x, pos):
        """x [n, heads, hd], neox (rotate_half)."""
        import torch
        f = pos.double()[:, None] * self.inv[None, :]
        cos = torch.cat([f.cos(), f.cos()], -1).float()[:, None, :]
        sin = torch.cat([f.sin(), f.sin()], -1).float()[:, None, :]
        x1, x2 = x[..., :self.hd // 2], x[..., self.hd // 2:]
        return x * cos + torch.cat([-x2, x1], -1) * sin

    def context(self, hid, pos):
        """hid [n, 6 x 4096] (the target layers concatenated), pos [n]: the K and V of each layer."""
        w = self.w
        ctx = self.rms(hid @ w["fc.weight"].T, w["hidden_norm.weight"])
        out = []
        for i in range(self.L):
            pre = "layers.%d.self_attn." % i
            k = (ctx @ w[pre + "k_proj.weight"].T).view(-1, self.nkv, self.hd)
            v = (ctx @ w[pre + "v_proj.weight"].T).view(-1, self.nkv, self.hd)
            k = self.rope(self.rms(k, w[pre + "k_norm.weight"]), pos)
            out.append((k, v))
        return out

    def draft(self, anchor, kv, n_ctx, pos):
        """The 8 drafts from the anchor with the context K/V of the positions before n_ctx (the last
        `window` of them); pos [8] the block's positions. Return (drafts, confidence)."""
        import torch
        w = self.w
        ids = torch.full((self.bs,), self.mask, dtype=torch.long)
        ids[0] = anchor
        x = w["embed_tokens.weight"][ids]
        g = self.nh // self.nkv
        c0 = max(0, n_ctx - self.window)
        for i in range(self.L):
            pre = "layers.%d." % i
            a = self.rms(x, w[pre + "input_layernorm.weight"])
            q = (a @ w[pre + "self_attn.q_proj.weight"].T).view(-1, self.nh, self.hd)
            k = (a @ w[pre + "self_attn.k_proj.weight"].T).view(-1, self.nkv, self.hd)
            v = (a @ w[pre + "self_attn.v_proj.weight"].T).view(-1, self.nkv, self.hd)
            q = self.rope(self.rms(q, w[pre + "self_attn.q_norm.weight"]), pos)
            k = self.rope(self.rms(k, w[pre + "self_attn.k_norm.weight"]), pos)
            ck, cv = kv[i]
            K = torch.cat([ck[c0:n_ctx], k], 0).repeat_interleave(g, dim=1)
            V = torch.cat([cv[c0:n_ctx], v], 0).repeat_interleave(g, dim=1)
            s = torch.einsum("qhd,khd->hqk", q, K) / self.hd ** 0.5
            o = torch.einsum("hqk,khd->qhd", s.softmax(-1), V).reshape(-1, self.nh * self.hd)
            x = x + o @ w[pre + "self_attn.o_proj.weight"].T
            a = self.rms(x, w[pre + "post_attention_layernorm.weight"])
            m = torch.nn.functional.silu(a @ w[pre + "mlp.gate_proj.weight"].T) * (a @ w[pre + "mlp.up_proj.weight"].T)
            x = x + m @ w[pre + "mlp.down_proj.weight"].T
        hn = self.rms(x, w["norm.weight"])
        logits = hn @ w["lm_head.weight"].T
        out = [int(anchor)]
        emb = []
        for i in range(self.bs):
            e = w["markov_head.markov_w1.weight"][out[i]]
            emb.append(e)
            out.append(int((logits[i] + w["markov_head.markov_w2.weight"] @ e).argmax()))
        emb = torch.stack(emb)
        cw, cb = w["confidence_head.proj.weight"], w["confidence_head.proj.bias"]
        conf = torch.sigmoid(torch.cat([hn, emb], -1) @ cw.T + cb)[:, 0]
        return out[1:], conf.numpy()


def accepted(d, ref):
    n = 0
    while n < len(d) and n < len(ref) and d[n] == ref[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--variants", default="out/mean,in/mean,out/s0")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    t0 = time.time()
    dr = Drafter()
    log("weights in %.0f s" % (time.time() - t0))
    z = np.load(a.trace)
    ids, gen = z["ids"], z["gen"]
    T = len(ids)
    S = np.concatenate([ids, gen])
    rows = z["h0"].shape[0]             # positions with hidden states: S[:rows]
    report = {}
    for var in a.variants.split(","):
        lay, st = var.split("/")
        taps = []
        for L in dr.targets:
            hs = z["h%d" % L] if lay == "out" else (z["h%d" % (L - 1)] if L > 0 else np.repeat(z["embed"][:, None], 4, 1))
            taps.append(hs.mean(axis=1) if st == "mean" else hs[:, 0])
        hid = torch.from_numpy(np.concatenate(taps, -1).astype(np.float32))
        kv = dr.context(hid, torch.arange(rows))
        acc, conf = {}, {}
        t1 = time.time()
        for p in range(T, min(rows, len(S) - 1) + 1):
            if p >= len(S):
                break
            ref = S[p + 1:p + 1 + dr.bs].tolist()
            d, cf = dr.draft(int(S[p]), kv, p, torch.arange(p, p + dr.bs))
            acc[p], conf[p] = accepted(d, ref), cf
        steps, p = [], T
        while p in acc:
            steps.append(acc[p] + 1)
            p += acc[p] + 1
        full = [v for q, v in acc.items() if q + dr.bs < len(S)]
        pts = np.array([(float(conf[q][i]), i < v) for q, v in acc.items() if q + dr.bs < len(S)
                        for i in range(min(v + 1, dr.bs))])
        res = dict(tokens_per_step=float(np.mean(steps)) if steps else None, steps=len(steps),
                   mean_accepted=float(np.mean(full)) if full else None, anchors=len(full),
                   first_token=float(np.mean([v > 0 for v in full])) if full else None,
                   per_position=[float(np.mean([v > i for v in full])) for i in range(dr.bs)] if full else None,
                   brier=float(np.mean((pts[:, 0] - pts[:, 1]) ** 2)) if len(pts) else None)
        log("%s: %s (%.0f s)" % (var, res, time.time() - t1))
        report[var] = res
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump(report, open(os.path.join(a.out, "report.json"), "w"), indent=1)
        log("written to %s" % a.out)


if __name__ == "__main__":
    main()
