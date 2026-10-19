#!/usr/bin/env python3
"""The reference of nvidia/Qwen3.8-Flash-Next-FP4-Local on the CPU (docs/FP4LOCAL_PLAN.md, P2).

scripts/qwen38_flash_reference.py (the transformers model of Qwen3.8-Flash-Next in float32, the
routed experts loaded as the router picks them, the n-gram rows on demand) on this checkpoint,
with the contract of the plan's section 2, written from the formats and the serving source and
not from reclank's code (the CSF decode is the NumPy oracle of flashinfer's
scripts/convert_csf_blocks.py):

- the MXFP8 matrices (attention, DeltaNet projections, indexer, shared experts) dequantized
  exactly (e4m3 x 2^(s - 127)); their inputs rounded to bfloat16 (the a16 mode, section 2.3);
- the MXFP8 outputs rounded to bfloat16 (the a16 kernel's out = bf16(acc));
- the routed experts with the arithmetic of b12x (work/fp4local/P0_FACTS.md 1, 2), W = E2M1 x
  e4m3(scale) exact: decode rows A16: g, u = bf16(alpha1 (x_bf16 . W)), h = bf16(bf16(g sigma(g))
  u), each route y = bf16(bf16(h . W2) bf16(w alpha2)), out = bf16(the float32 sum of the routes
  in the router's order); prompt rows A4 (--a16-prompt: A16): x and h quantized to NVFP4 with
  the threshold quantizer of b12x (gs1 = min over experts of 1 / max(input_scale gate, up), gs2 =
  1 / input_scale down), g, u = bf16(acc alpha1), alpha = ws2 / gs, y = bf16(acc2 alpha2), out =
  bf16(sum of w y in float32);
- the float32 norms rounded to bfloat16 (vLLM's bfloat16 parameters), then (1 + w) in float32;
- the n-gram rows: E2M1 x e4m3 x weight_scale_2, rounded to bfloat16;
- the cache of the attention (the keys after their norm and RoPE, the values) in E4M3 (vLLM's
  --kv-cache-dtype fp8 with the scales 1: to nearest even, saturating; --kv-f32: float32);
- everything else float32, as transformers runs it (reclank's D2).

Outputs as qwen38_flash_reference.py (<out>/<name>.npz: ids, gen, top_ids, top_vals, lse, argmax,
step_logits), --forced FILE.npz (teacher-forced logits of another run's tokens, e.g. the float
reference of the original checkpoint), and --trace: <out>/<name>_trace.npz with each layer's
router picks (topk<i>, topw<i>), the experts' input (mlp_in<i>, float32) and output (moe_out<i>)
of every row, and the final streams (h).

    OMP_NUM_THREADS=1 taskset -c 24-46 <torch venv>/bin/python -I scripts/qfn_fp4l_reference.py \\
        models/Qwen3.8-Flash-Next-FP4-Local --out <dir> --threads 23
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from qwen38_flash_reference import MAIN, N_FULL, N_TOP, Shards, log, prompts  # noqa: E402

E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
FIX, EXC = ".weight_scale.nvfp4_csf_fixed", ".weight_scale.nvfp4_csf_exceptions"


def e4m3_table():
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m / 512.0, (1 + m / 8) * 2.0 ** (e - 7))
    v = np.where((b & 127) == 127, np.nan, v)
    return np.where(b & 128, -v, v).astype(np.float32)


E4M3 = e4m3_table()


def bf16r(a):
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32).reshape(np.shape(a))


def csf_oracle(fixed, records, rows, columns, bits=4):
    """flashinfer scripts/convert_csf_blocks.py _logical_source, as it is."""
    row_bytes = (columns * bits + 7) // 8
    codes = fixed[:, 16:].reshape(rows, row_bytes)
    bit_columns = np.arange(columns) * bits
    result = fixed[:, :16].reshape(rows, 1) + (
        (codes[:, bit_columns // 8] >> (bit_columns % 8)) & ((1 << bits) - 1))
    result = np.ascontiguousarray(result, dtype=np.uint8)
    result.ravel()[records & 0xFFFFFF] = records >> 24
    return result


def nvfp4_w(codes, scales):
    """E2M1 x e4m3(scale) of NVFP4 rows (exact in float32), without the global scale."""
    n, h = codes.shape
    q = np.empty((n, 2 * h), np.uint8)
    q[:, 0::2], q[:, 1::2] = codes & 15, codes >> 4
    return (E2M1[q].reshape(n, -1, 16) * E4M3[scales][:, :, None]).reshape(n, 2 * h)


class FP4Shards(Shards):
    """Shards with the forms of FP4-Local: MXFP8 matrices dequantized, the float32 norms rounded
    to bfloat16, raw bytes for the rest."""

    def bytes_of(self, name):
        p, dt, shape, s, e = self.t[name]
        return np.frombuffer(self.raw(p, s, e - s), {"U8": np.uint8, "F8_E4M3": np.uint8, "U32": np.uint32,
                                                     "F32": np.float32, "I64": np.int64}[dt]).reshape(shape)

    def is_mxfp8(self, name):
        return name.endswith(".weight") and self.t[name][1] == "F8_E4M3" and \
            self.t.get(name[:-7] + ".weight_scale", (0, ""))[1] == "U8"

    def f32(self, name):
        if self.is_mxfp8(name):
            w = E4M3[self.bytes_of(name)]
            s = np.ldexp(np.float32(1), self.bytes_of(name[:-7] + ".weight_scale").astype(np.int32) - 127)
            n, k = w.shape
            return (w.reshape(n, k // 32, 32) * s[:, :, None].astype(np.float32)).reshape(n, k)
        a = super().f32(name)
        if self.t[name][1] == "F32" and "norm" in name:
            a = bf16r(a)
        return a

    def expert(self, pre, e):
        """(Wg, Wu, Wd, alpha1, alpha2, input scales (g, u, d)) of expert e of the layer prefix pre
        (pre: ...mlp.experts.), W = E2M1 x e4m3 float32."""
        out = []
        for x in ("gate", "up", "down"):
            p = pre + "%d.%s_proj" % (e, x)
            codes = self.bytes_of(p + ".weight")
            rows = codes.shape[0]
            if p + FIX in self.t:
                sc = csf_oracle(self.bytes_of(p + FIX), self.bytes_of(p + EXC).astype(np.uint32), rows,
                                codes.shape[1] // 8)
            else:
                sc = self.bytes_of(p + ".weight_scale")
            g2 = float(self.bytes_of(p + ".weight_scale_2").reshape(-1)[0])
            isc = float(self.bytes_of(p + ".input_scale").reshape(-1)[0]) if p + ".input_scale" in self.t else None
            out.append((nvfp4_w(codes, sc), g2, isc))
        (wg, ag, ig), (wu, au, iu), (wd, ad, idn) = out
        assert ag == au, (pre, e, ag, au)       # (nvfp4_csf_utils.py:164-165)
        return wg, wu, wd, ag, ad, (ig, iu, idn)


# ---- the contract of the routed experts (P0_FACTS.md 1, 2) ----

def e4m3_round(v):
    """float32 to the nearest E4M3 value (round to nearest even, subnormals, saturating at 448):
    cvt.rn.satfinite.e4m3x2.f32."""
    pos = E4M3[:127]
    a = np.minimum(np.abs(np.asarray(v, np.float32)), np.float32(448.0))
    i = np.clip(np.searchsorted(pos, a), 1, 126)
    lo, hi = pos[i - 1], pos[i]
    pick_hi = (a - lo > hi - a) | ((a - lo == hi - a) & ((i % 2) == 0))
    return np.copysign(np.where(pick_hi, hi, lo), v).astype(np.float32)


_TH = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], np.float32)


def nvfp4_act(x, gs):
    """x (t x k float32) quantized as b12x's quantize_block_fp4 (fast_math off): for each 16, amax,
    sf = e4m3(min(amax * gs / 6, 448)), the code of |v| from the thresholds (sf / gs) x {0.25, 0.75,
    1.25, 1.75, 2.5, 3.5, 5} (RNE ties on the unscaled |v|, clamped at 6). Returns (q, sf): the E2M1
    values and the scale of each 16 (the product's operand is q x sf; x ~ q x sf / gs)."""
    t, k = x.shape
    gs = np.float32(gs)
    v = np.asarray(x, np.float32).reshape(t, k // 16, 16)
    amax = np.abs(v).max(axis=2)
    sf = e4m3_round(np.minimum((amax * gs) / np.float32(6.0), np.float32(448.0)))
    vs = (sf / gs).astype(np.float32)[:, :, None]
    th = (vs * _TH).astype(np.float32)                       # each an f32 multiply
    a = np.abs(v)
    q = np.zeros_like(a)
    q = np.where((a > th[..., 0:1]) & (a < th[..., 1:2]), 0.5, q)
    q = np.where((a >= th[..., 1:2]) & (a <= th[..., 2:3]), 1.0, q)
    q = np.where((a > th[..., 2:3]) & (a < th[..., 3:4]), 1.5, q)
    q = np.where((a >= th[..., 3:4]) & (a <= th[..., 4:5]), 2.0, q)
    q = np.where((a > th[..., 4:5]) & (a < th[..., 5:6]), 3.0, q)
    q = np.where((a >= th[..., 5:6]) & (a <= th[..., 6:7]), 4.0, q)
    q = np.where(a > th[..., 6:7], 6.0, q)
    q = np.where(sf[:, :, None] == 0, 0.0, q)
    q = np.where(v < 0, -q, q).astype(np.float32)
    return q.reshape(t, k), sf


def a4_operand(x, gs):
    """The A4 operand of the product: q x sf (exact in float32)."""
    q, sf = nvfp4_act(x, gs)
    t, k = x.shape
    return (q.reshape(t, k // 16, 16) * sf[:, :, None]).reshape(t, k).astype(np.float32)


def silu_bf16(g, u):
    """h = bf16(bf16(g sigma(g)) u) (the W4A16 / A4 epilogue)."""
    sig = (np.float32(1) / (np.float32(1) + np.exp(-g, dtype=np.float32))).astype(np.float32)
    return bf16r(bf16r(g * sig) * u)


class Experts:
    """The routed experts: an LRU of (Wg, Wu, Wd, ...) float32, the contract's forward."""

    def __init__(self, sh, cfg, cache_gb, args):
        self.sh, self.cfg, self.a = sh, cfg, args
        self.lru = OrderedDict()
        self.cap = int(cache_gb * 1e9 / (3 * cfg.moe_intermediate_size * cfg.hidden_size * 4))
        self.loads = self.hits = 0
        self.g13 = {}
        self.prompt = False                     # the rows that run are prompt rows (A4)
        self.trace = None

    def get(self, layer, e):
        key = (layer, e)
        w = self.lru.pop(key, None)
        if w is None:
            w = self.sh.expert(MAIN + "layers.%d.mlp.experts." % layer, e)
            self.loads += 1
        else:
            self.hits += 1
        self.lru[key] = w
        while len(self.lru) > self.cap:
            self.lru.popitem(last=False)
        return w

    def fc1_scale(self, layer):
        """The A4 global scale of FC1 for the layer: min over experts of 1 / max(input_scale_gate,
        input_scale_up) (b12x A4 README; nvfp4_csf.py:330-343)."""
        g = self.g13.get(layer)
        if g is None:
            pre = MAIN + "layers.%d.mlp.experts." % layer
            best = np.inf
            for e in range(self.cfg.num_experts):
                ig = float(self.sh.bytes_of(pre + "%d.gate_proj.input_scale" % e).reshape(-1)[0])
                iu = float(self.sh.bytes_of(pre + "%d.up_proj.input_scale" % e).reshape(-1)[0])
                best = min(best, np.float32(1.0) / np.float32(max(ig, iu)))
            g = self.g13[layer] = float(best)
        return g

    def forward(self, layer, x, idx, wts):
        """x (t x H float32), idx / wts (t x k): the routed experts' output (t x H float32)."""
        import torch
        a = self.a
        x = x.numpy() if hasattr(x, "numpy") else x
        idx, wts = idx.numpy(), wts.numpy().astype(np.float32)
        t, H = x.shape
        a4 = self.prompt and not a.a16_prompt
        xb = bf16r(x)
        gs1 = np.float32(self.fc1_scale(layer)) if a4 else None
        xin = a4_operand(xb, gs1) if a4 else xb
        ys = np.zeros((t, idx.shape[1], H), np.float32)          # each route's row (router order)
        mm = lambda A, W: (torch.from_numpy(np.ascontiguousarray(A)) @ torch.from_numpy(W).T).numpy()  # noqa: E731
        for e in np.unique(idx):
            tok, slot = np.nonzero(idx == e)
            wg, wu, wd, ws2g, ws2d, isc = self.get(layer, int(e))
            if a4:
                a1 = np.float32(np.float32(ws2g) / gs1)
                g, u = bf16r(mm(xin[tok], wg) * a1), bf16r(mm(xin[tok], wu) * a1)
                h = silu_bf16(g, u)
                gs2 = np.float32(np.float32(1) / np.float32(isc[2]))
                a2 = np.float32(np.float32(ws2d) / gs2)
                ys[tok, slot] = bf16r(mm(a4_operand(h, gs2), wd) * a2)
            else:
                g = bf16r(mm(xin[tok], wg) * np.float32(ws2g))
                u = bf16r(mm(xin[tok], wu) * np.float32(ws2g))
                h = silu_bf16(g, u)
                d2 = bf16r(mm(h, wd))
                ys[tok, slot] = bf16r(d2 * bf16r(wts[tok, slot] * np.float32(ws2d))[:, None])
        acc = np.zeros((t, H), np.float32)
        for k in range(idx.shape[1]):                              # the float32 sum in the router's order
            acc = (acc + ys[:, k] * wts[:, k:k + 1]).astype(np.float32) if a4 else (acc + ys[:, k]).astype(np.float32)
        out = bf16r(acc)
        if self.trace is not None:
            tr = self.trace
            tr.setdefault("mlp_in%d" % layer, []).append(x.astype(np.float32))
            tr.setdefault("topk%d" % layer, []).append(idx.astype(np.int16))
            tr.setdefault("topw%d" % layer, []).append(wts)
            tr.setdefault("moe_out%d" % layer, []).append(out)
        return torch.from_numpy(out)


def row_table(sh, prefix):
    """The n-gram table (NVFP4 rows of 160): forward(ids) decodes those rows, rounded to bfloat16."""
    import torch
    from torch import nn
    names = sorted((n for n in sh.t if n.startswith(prefix + ".shard_") and n.endswith(".weight")),
                   key=lambda n: int(n[len(prefix) + 7:].split(".")[0]))
    rows = [sh.t[n][2][0] for n in names]
    per, half = rows[0], sh.t[names[0]][2][1]
    g = float(sh.bytes_of(prefix + ".weight_scale_2").reshape(-1)[0])

    class RowTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.empty(0)
            self.num_rows = sum(rows)

        def forward(self, ids):
            flat = ids.reshape(-1).tolist()
            out = np.empty((len(flat), 2 * half), np.float32)
            for j, r in enumerate(flat):
                n = names[r // per]
                p, _dt, _sh, s, _e = sh.t[n]
                c = np.frombuffer(sh.raw(p, s + half * (r % per), half), np.uint8).reshape(1, half)
                p2, _dt, _sh, s2, _e = sh.t[n + "_scale"]
                sc = np.frombuffer(sh.raw(p2, s2 + half // 8 * (r % per), half // 8), np.uint8).reshape(1, -1)
                out[j] = bf16r(nvfp4_w(c, sc) * np.float32(g))[0]
            return torch.from_numpy(out).reshape(*ids.shape, 2 * half)
    return RowTable()


def build(model_dir, sh, ex_args, layers=None):
    import torch
    from torch import nn
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding

    cfg = AutoConfig.from_pretrained(model_dir).text_config
    cfg.dtype = torch.float32
    if layers:
        cfg.num_hidden_layers = layers
        cfg.layer_types = cfg.layer_types[:layers]
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM._from_config(cfg, dtype=torch.float32)
    model.eval()
    t0 = time.time()
    lm = model.model
    tables = {}
    for i, layer in enumerate(lm.layers):
        if layer.ple is not None:
            tables[i] = row_table(sh, MAIN + "layers.%d.ple.ple_embedding.ngram_embedding" % i)
            layer.ple.ple_embedding.ngram_embedding = tables[i]
    used, mx = set(), []
    for n, _ in list(model.named_parameters()):
        if ".mlp.experts." in n:
            continue
        src = n.replace("model.", MAIN, 1) if n.startswith("model.") else n
        assert src in sh.t, (n, src)
        path, leaf = n.rsplit(".", 1)
        model.get_submodule(path)._parameters[leaf] = nn.Parameter(torch.from_numpy(sh.f32(src)), requires_grad=False)
        used.add(src)
        if sh.is_mxfp8(src):
            used.add(src[:-7] + ".weight_scale")
            mx.append(path)
    for n, b in list(model.named_buffers()):
        src = n.replace("model.", MAIN, 1)
        if src in sh.t:
            path, leaf = n.rsplit(".", 1)
            model.get_submodule(path)._buffers[leaf] = torch.from_numpy(sh.f32(src))
            used.add(src)
    left = [n for n in sh.t if n not in used and not n.startswith(("mtp.", "model.visual."))
            and ".mlp.experts." not in n and ".ngram_embedding." not in n]
    assert layers or not left, left[:5]
    # the a16 contract of the MXFP8 matrices: their input in bfloat16
    for path in mx:
        mod = model.get_submodule(path)
        mod.register_forward_pre_hook(lambda mod, args: (torch.from_numpy(bf16r(args[0].numpy())),) + tuple(args[1:]))
        mod.register_forward_hook(lambda mod, args, out: torch.from_numpy(bf16r(out.numpy())))
    log("dense weights in %.0f s (%.1f GB read; %d MXFP8 matrices, bfloat16 inputs and outputs)" % (
        time.time() - t0, sh.read_bytes / 1e9, len(mx)))
    ex = Experts(sh, cfg, ex_args.expert_cache_gb, ex_args)
    for i, layer in enumerate(lm.layers):
        layer.mlp.experts.forward = lambda x, idx, w, i=i: ex.forward(i, x, idx, w)
        layer.mlp.experts._parameters.clear()
    lm.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=cfg)
    still = [n for n, b in model.named_buffers() if b.is_meta] + [n for n, p in model.named_parameters() if p.is_meta]
    assert not still, still[:5]
    return model, ex, tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--only", default=None)
    ap.add_argument("--threads", type=int, default=23)
    ap.add_argument("--expert-cache-gb", type=float, default=24.0)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--forced", default=None, help="an npz of ids and gen: the logits of those tokens")
    ap.add_argument("--a16-prompt", action="store_true", help="the prompt rows' experts A16 too (not A4)")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--kv-f32", action="store_true", help="the attention cache in float32 (not E4M3)")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    sh = FP4Shards(a.model)
    model, ex, tables = build(a.model, sh, a, a.layers)
    cfg = model.config
    hcap = {}
    model.model.norm.register_forward_pre_hook(lambda mod, args: hcap.__setitem__("h", args[0].numpy().copy())) \
        if hasattr(model.model, "norm") else None

    def fp8_cache(cache):
        """The keys and values that the attention writes to the cache rounded to E4M3."""
        if a.kv_f32:
            return cache
        upd = cache.update

        def update(k, v, layer_idx, *args, **kw):
            k = torch.from_numpy(e4m3_round(k.numpy()).reshape(k.shape))
            v = torch.from_numpy(e4m3_round(v.numpy()).reshape(v.shape))
            return upd(k, v, layer_idx, *args, **kw)
        cache.update = update
        return cache

    def run(x, pos, cache, prompt):
        ex.prompt = prompt
        out = model(input_ids=torch.tensor([x]), past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(pos, pos + len(x)).view(1, -1))
        return out.logits[0].float()

    def rows(lg, rec):
        tv, ti = torch.topk(lg, N_TOP, dim=-1)
        rec["top_ids"].append(ti.numpy().astype(np.int32))
        rec["top_vals"].append(tv.numpy())
        rec["lse"].append(torch.logsumexp(lg, dim=-1).numpy())
        rec["argmax"].append(lg.argmax(-1).numpy().astype(np.int32))

    if a.forced:
        z = np.load(a.forced)
        ids, gen = z["ids"].tolist(), z["gen"].tolist()
        cache = fp8_cache(DynamicCache(config=cfg))
        rec = {k: [] for k in ("top_ids", "top_vals", "lse", "argmax")}
        t0 = time.time()
        rows(run(ids, 0, cache, True), rec)                 # the prompt as prompt rows (A4)
        for j, y in enumerate(gen[:-1]):                    # the rest as decode steps (A16)
            rows(run([y], len(ids) + j, cache, False), rec)
            if j % 20 == 0:
                log("forced: %d of %d steps (%.0f s)" % (j + 1, len(gen) - 1, time.time() - t0))
        out = os.path.join(a.out, os.path.basename(a.forced)[:-4] + "_fp4l.npz")
        np.savez(out, ids=z["ids"], gen=z["gen"], **{k: np.concatenate(v) for k, v in rec.items()})
        log("written %s" % out)
        return

    for name, msgs in prompts():
        if a.only and name not in a.only.split(","):
            continue
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        log("%s: %d tokens" % (name, len(ids)))
        cache = fp8_cache(DynamicCache(config=cfg))
        rec = {k: [] for k in ("top_ids", "top_vals", "lse", "argmax")}
        ex.trace = {} if a.trace else None
        hs = []
        full, gen = [], []
        t0 = time.time()
        lg = run(ids, 0, cache, True)
        hs.append(hcap.get("h"))
        rows(lg, rec)
        full.append(lg[-1].numpy().copy())
        gen.append(int(lg[-1].argmax()))
        log("%s: prompt in %.0f s (%.1f GB read; experts %d read, %d from memory)" % (
            name, time.time() - t0, sh.read_bytes / 1e9, ex.loads, ex.hits))
        t0 = time.time()
        while gen[-1] not in eos and len(gen) < a.steps:
            lg = run([gen[-1]], len(ids) + len(gen) - 1, cache, False)
            hs.append(hcap.get("h"))
            rows(lg, rec)
            if len(full) <= N_FULL:
                full.append(lg[-1].numpy().copy())
            gen.append(int(lg[-1].argmax()))
            if len(gen) % 10 == 0:
                log("%s: %d tokens, %.1f s a step (experts %d read, %d from memory)" % (
                    name, len(gen), (time.time() - t0) / (len(gen) - 1), ex.loads, ex.hits))
        out = tok.decode(gen)
        log("%s: %d tokens: %r" % (name, len(gen), out[-300:]))
        np.savez(os.path.join(a.out, name + ".npz"), ids=np.array(ids, np.int32), gen=np.array(gen, np.int32),
                 prompt=np.array(text), text=np.array(out),
                 top_ids=np.concatenate(rec["top_ids"]), top_vals=np.concatenate(rec["top_vals"]),
                 lse=np.concatenate(rec["lse"]), argmax=np.concatenate(rec["argmax"]),
                 step_logits=np.stack(full))
        if ex.trace is not None:
            tr = {k: np.concatenate(v) for k, v in ex.trace.items()}
            if hs and hs[0] is not None:
                tr["h"] = np.concatenate([h.reshape(-1, h.shape[-1]) for h in hs])
            np.savez(os.path.join(a.out, name + "_trace.npz"), ids=np.array(ids, np.int32),
                     gen=np.array(gen, np.int32), **tr)
    log("written to %s" % a.out)


if __name__ == "__main__":
    main()
