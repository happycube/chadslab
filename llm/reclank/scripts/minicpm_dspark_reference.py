#!/usr/bin/env python3
"""The reference of the DSpark drafter of MiniCPM-V 4.7 35B-A3B
(openbmb/MiniCPM-V-4.7-35B-A3B-DSpark; docs/MINICPM_V47_PLAN.md section 2),
on the dumps of scripts/minicpm_reference.py.

The drafter as vLLM runs it (qwen3_dflash.py, qwen3_dspark.py), in float32
torch on the CPU.

Ported from vLLM (vllm/model_executor/models/qwen3_dflash.py, qwen3_dspark.py;
Apache License 2.0; Copyright contributors to the vLLM project). See
THIRD_PARTY_NOTICES.md.


- the context: fc (10240 -> 2048) of the hidden states of the target layers
  3, 11, 19, 27, 35 of each position, concatenated in that order; then for
  each of the 5 layers: k_proj and v_proj of hidden_norm(context), k_norm of
  each head, the rope (neox, theta 1e7, all 256 dims) at the context
  positions;
- the block: [the anchor token, 14 x mask_token_id] at the 15 positions from
  the anchor's, the embeddings of the main model; each layer a Qwen3 decoder
  layer (RMSNorm x * w, q/k norm, GQA 8 / 2 heads) whose attention reads the
  context K/V and the block's own K/V with no causal mask; then norm and the
  main model's lm_head;
- the Markov head, in order: logits[i] += markov_w2 @ markov_w1[out[i]],
  out[i + 1] = argmax (out[0] the anchor); the confidence:
  sigmoid(proj(cat(hidden, markov_w1[out[i]]))).

The anchor at index p of S = ids + gen (a greedy chain of the main model)
has the context of the positions 0 .. p - 1; the drafts d1 .. d15 are
accepted while they match S[p + 1 ..]. The script measures, for each variant
of the two points the drafter code leaves open:

    layer   "out": the target hidden state of layer L is its output
            (hs[L + 1] of the dump); "in": its input (hs[L])
    pos     "text": the positions of the drafter are the indices of the
            sequence; "mrope": the t row of the canvas M-RoPE (differs only
            after an image)

the accepted drafts of every anchor in gen, and the tokens per verify step
of a greedy run (from the first token of gen, anchor += accepted + 1). The
variant the drafter was trained on accepts far more.

    python scripts/minicpm_dspark_reference.py DUMP_DIR [--out DUMP_DIR/dspark]
    (MINICPM_DIR, MINICPM_DRAFT: the model's and the drafter's directories)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np

MAIN = os.environ.get("MINICPM_DIR", "MiniCPM-V-4.7-35B-A3B")             # the model and its drafter
DRAFT = os.environ.get("MINICPM_DRAFT", MAIN + "-DSpark")


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
    def __init__(self):
        import torch
        self.c = json.load(open(os.path.join(DRAFT, "config.json")))
        dc = self.c["dflash_config"]
        self.targets = dc["target_layer_ids"]
        self.mask = dc["mask_token_id"]
        self.bs = self.c["block_size"]
        self.nh, self.nkv, self.hd = self.c["num_attention_heads"], self.c["num_key_value_heads"], self.c["head_dim"]
        self.eps = self.c["rms_norm_eps"]
        self.L = self.c["num_hidden_layers"]
        p = os.path.join(DRAFT, "model.safetensors")
        n, h = st_header(p)
        self.w = {k: st_read(p, n, h, k) for k in h}
        # the main model's embeddings and head
        where = json.load(open(os.path.join(MAIN, "model.safetensors.index.json")))["weight_map"]
        for k, name in (("embed", "model.language_model.embed_tokens.weight"), ("head", "lm_head.weight")):
            fp = os.path.join(MAIN, where[name])
            fn, fh = st_header(fp)
            self.w[k] = st_read(fp, fn, fh, name)
        theta = self.c["rope_parameters"]["rope_theta"]
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
        """hid [n, 5 * 2048] (the target layers concatenated), pos [n]: the
        K and V of each layer [n, nkv, hd]."""
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
        """The 15 drafts from the anchor token with the first n_ctx rows of
        the context K/V; pos [15] the block's positions. Return (drafts,
        confidence (pre-norm hidden), confidence (post-norm hidden))."""
        import torch
        w = self.w
        ids = torch.full((self.bs,), self.mask, dtype=torch.long)
        ids[0] = anchor
        x = w["embed"][ids]
        g = self.nh // self.nkv
        for i in range(self.L):
            pre = "layers.%d." % i
            a = self.rms(x, w[pre + "input_layernorm.weight"])
            q = (a @ w[pre + "self_attn.q_proj.weight"].T).view(-1, self.nh, self.hd)
            k = (a @ w[pre + "self_attn.k_proj.weight"].T).view(-1, self.nkv, self.hd)
            v = (a @ w[pre + "self_attn.v_proj.weight"].T).view(-1, self.nkv, self.hd)
            q = self.rope(self.rms(q, w[pre + "self_attn.q_norm.weight"]), pos)
            k = self.rope(self.rms(k, w[pre + "self_attn.k_norm.weight"]), pos)
            ck, cv = kv[i]
            K = torch.cat([ck[:n_ctx], k], 0).repeat_interleave(g, dim=1)      # [m, nh, hd]
            V = torch.cat([cv[:n_ctx], v], 0).repeat_interleave(g, dim=1)
            s = torch.einsum("qhd,khd->hqk", q, K) / self.hd ** 0.5
            o = torch.einsum("hqk,khd->qhd", s.softmax(-1), V).reshape(-1, self.nh * self.hd)
            x = x + o @ w[pre + "self_attn.o_proj.weight"].T
            a = self.rms(x, w[pre + "post_attention_layernorm.weight"])
            m = torch.nn.functional.silu(a @ w[pre + "mlp.gate_proj.weight"].T) * (a @ w[pre + "mlp.up_proj.weight"].T)
            x = x + m @ w[pre + "mlp.down_proj.weight"].T
        hn = self.rms(x, w["norm.weight"])
        logits = hn @ w["head"].T
        out = [int(anchor)]
        emb = []
        for i in range(self.bs):
            e = w["markov_head.markov_w1.weight"][out[i]]
            emb.append(e)
            out.append(int((logits[i] + w["markov_head.markov_w2.weight"] @ e).argmax()))
        emb = torch.stack(emb)
        cw, cb = w["confidence_head.proj.weight"], w["confidence_head.proj.bias"]
        c_pre = torch.sigmoid(torch.cat([x, emb], -1) @ cw.T + cb)[:, 0]
        c_post = torch.sigmoid(torch.cat([hn, emb], -1) @ cw.T + cb)[:, 0]
        return out[1:], c_pre.numpy(), c_post.numpy()


def accepted(d, ref):
    n = 0
    while n < len(d) and n < len(ref) and d[n] == ref[n]:
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    out_dir = a.out or os.path.join(a.dumps, "dspark")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    dr = Drafter()
    log("weights in %.0f s" % (time.time() - t0))
    report = {}
    for path in sorted(glob.glob(os.path.join(a.dumps, "*.npz"))):
        name = os.path.basename(path)[:-4]
        if a.only and name not in a.only.split(","):
            continue
        z = np.load(path)
        ids, gen, hs, P = z["ids"], z["gen"], z["hs"], z["positions"]
        S = np.concatenate([ids, gen])
        T = len(ids)
        rows = hs.shape[1]            # positions with hidden states: S[:rows]
        variants = [("out", "text"), ("in", "text")]
        if (P[1] != P[0]).any():
            variants += [("out", "mrope"), ("in", "mrope")]
        res = {}
        for lay, pk in variants:
            sel = [L + 1 if lay == "out" else L for L in dr.targets]
            hid = torch.from_numpy(np.concatenate([hs[j] for j in sel], -1))
            prow = torch.from_numpy(P[0] if pk == "text" else P[1]).long()
            kv = dr.context(hid, prow[:rows])
            acc, conf_pre, conf_post = {}, {}, {}
            # every anchor in gen with the full 15 tokens of reference after it
            for p in range(T, len(S) - 1):
                if p > rows:
                    break
                ref = S[p + 1:p + 1 + dr.bs].tolist()
                last = int(prow[p - 1]) if pk == "mrope" else p - 1
                bpos = torch.arange(last + 1, last + 1 + dr.bs)
                d, cpre, cpost = dr.draft(int(S[p]), kv, p, bpos)
                acc[p] = accepted(d, ref)
                conf_pre[p], conf_post[p] = cpre, cpost
            # a greedy run: from the first token of gen
            steps, p = [], T
            while p in acc:
                steps.append(acc[p] + 1)
                p += acc[p] + 1
            full = [v for q, v in acc.items() if q + dr.bs < len(S)]
            # confidence calibration: P(draft i accepted | the drafts before accepted)
            cal = {}
            for nm, cf in (("pre", conf_pre), ("post", conf_post)):
                pts = []
                for q, v in acc.items():
                    if q + dr.bs >= len(S):
                        continue
                    for i in range(min(v + 1, dr.bs)):
                        pts.append((float(cf[q][i]), i < v))
                pts = np.array(pts)
                if len(pts):
                    cal[nm] = float(np.mean((pts[:, 0] - pts[:, 1]) ** 2))
            key = "%s/%s" % (lay, pk)
            res[key] = dict(tokens_per_step=float(np.mean(steps)) if steps else None, steps=len(steps),
                            mean_accepted=float(np.mean(full)) if full else None, anchors=len(full),
                            first_token=float(np.mean([v > 0 for v in full])) if full else None,
                            brier_pre=cal.get("pre"), brier_post=cal.get("post"))
            log("%s %s: %s" % (name, key, res[key]))
        report[name] = res
    json.dump(report, open(os.path.join(out_dir, "report.json"), "w"), indent=1)
    log("written to %s" % out_dir)


if __name__ == "__main__":
    main()
