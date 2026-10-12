"""DeepSeek V4.1 Flash on the CPU in NumPy (DEEPSEEK_V41_PLAN.md Phase 2): the
reference's forward (inference/model.py) op for op, with the products of
this runtime (csrc/kernels/kq/mxx.c).

Two modes (DEEPSEEK_V41_PLAN.md 3):

- near (the default): the activations in float32; the bfloat16 matrices as
  BF12 (their values decoded); the FP8 matrices with float32 x; the MXFP4
  experts with int16 x (KQ_MXX); the roundings that are part of the model
  as the reference does them (the window KV to E4M3, the compressed KV to
  FP4 with E4M3 scales for each 16, the index keys and queries to FP4 with
  E8M0 scales for each 32).
- exact (exact=True): the activations rounded to bfloat16 where the
  reference holds them in bfloat16; the bfloat16 matrices as bfloat16; the
  input of every FP8 and FP4 product rounded as act_quant rounds it.

The weights: the dense ones from the shards (maps of the files), the routed
experts from the zygote's segments when one runs (else from the shards),
packed to KQ_MXX for each use. Text only, one sequence: a prompt at position
0 in one chunk, then steps of one token (as the reference).

    m = DSV41CPU("/home/cpage/models/DeepSeek-V4.1-Flash", exact=False)
    logits, layers = m.forward(ids, 0, keep=True)     # logits [T, vocab], layers [T, 40, 4, 5120]
    logits, _ = m.forward([tok], len(ids))
"""
from __future__ import annotations

import math
import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats import dsv41_st as D
from reclank.models.deepseek41.engram import Hashes


def bf16r(x):
    """Round float32 to bfloat16 (nearest, ties to even), as float32."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    r = ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32)
    r = np.where(np.isnan(x), u | 0x400000, r).astype(np.uint32)
    return r.view(np.float32)


def bits_f32(b):
    return (np.asarray(b, np.uint16).astype(np.uint32) << 16).view(np.float32)


def _rope_freqs(dim, seqlen, original, base, factor, beta_fast, beta_slow):
    """precompute_freqs_cis of the reference: (cos, sin) [seqlen, dim / 2] float32."""
    freqs = (1.0 / (np.float32(base) ** (np.arange(0, dim, 2, dtype=np.float32) / np.float32(dim)))).astype(np.float32)
    if original > 0:
        def corr(rot):
            return dim * math.log(original / (rot * 2 * math.pi)) / (2 * math.log(base))
        low = max(math.floor(corr(beta_fast)), 0)
        high = min(math.ceil(corr(beta_slow)), dim - 1)
        ramp = np.clip((np.arange(dim // 2, dtype=np.float32) - low) / max(high - low, 1e-3), 0, 1).astype(np.float32)
        smooth = (1 - ramp).astype(np.float32)
        freqs = (freqs / np.float32(factor) * (1 - smooth) + freqs * smooth).astype(np.float32)
    ang = np.outer(np.arange(seqlen, dtype=np.float32), freqs).astype(np.float32)
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def rope(x, cs, inverse=False):
    """Rotate the last dim of x (pairs of adjacent values) by cs = (cos, sin)
    [..., d / 2] broadcast over the heads."""
    c, s = cs
    if inverse:
        s = -s
    a, b = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = a * s + b * c
    return out.astype(np.float32)


def _e2m1_round(v):
    grid = D.E2M1[:8]
    a = np.abs(v)
    hi = np.searchsorted(grid, a).clip(0, 7)
    lo = (hi - 1).clip(0)
    dlo, dhi = a - grid[lo], grid[hi] - a
    pick = np.where(dlo < dhi, lo, np.where(dhi < dlo, hi, np.where(lo % 2 == 0, lo, hi)))
    out = grid[pick]
    return np.where(v < 0, -out, out).astype(np.float32)


def _pow2_ceil(v):
    m, e = np.frexp(v.astype(np.float32))
    e = np.where(m == 0.5, e - 1, e)
    return np.ldexp(np.float32(1.0), e).astype(np.float32)


def fp4_quant(x, block, e4m3_scale):
    """fp4_act_quant of the reference, in place form: the dequantized values."""
    shp = x.shape
    g = x.astype(np.float32).reshape(-1, block)
    amax = np.abs(g).max(-1, keepdims=True)
    if e4m3_scale:
        amax = np.maximum(amax, np.float32(6 * 2.0 ** -9))
        s = _e4m3_round((amax / np.float32(6.0)).astype(np.float32))
    else:
        amax = np.maximum(amax, np.float32(6 * 2.0 ** -126))
        s = _pow2_ceil((amax * np.float32(1.0 / 6.0)).astype(np.float32))
    y = _e2m1_round(np.clip(g / s, -6, 6).astype(np.float32))
    return (y * s).astype(np.float32).reshape(shp)


_E4M3_POS = np.unique(D.E4M3[np.isfinite(D.E4M3) & (D.E4M3 >= 0)])


def _e4m3_round(v):
    a = np.abs(v)
    i = np.searchsorted(_E4M3_POS, a).clip(1, len(_E4M3_POS) - 1)
    lo, hi = _E4M3_POS[i - 1], _E4M3_POS[i]
    pick = np.where(a - lo < hi - a, lo, np.where(hi - a < a - lo, hi, np.where((i - 1) % 2 == 0, lo, hi)))
    pick = np.where(a >= _E4M3_POS[-1], _E4M3_POS[-1], pick)
    return np.where(v < 0, -pick, pick).astype(np.float32)


def act_quant(x):
    """act_quant of the reference (E4M3, power-of-two scales for each 32), in
    place form."""
    shp = x.shape
    return cops.dsv41_act_quant(np.ascontiguousarray(x, np.float32).reshape(-1, shp[-1])).reshape(shp)


def softmax(x, axis=-1):
    m = x.max(axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis, keepdims=True)


class Weights:
    """The dense weights (maps of the shards; small float forms cached) and the
    routed experts (the zygote's segments, or the shards)."""

    def __init__(self, path, exact, zygote=True):
        self.src = D.DSV41Source(path)
        self.exact = exact
        self.cache = {}
        self.z = None
        if zygote:
            try:
                from reclank.runtime import zygote as Z
                self.z = Z.attach()
                self.z.check(path)
            except (OSError, ValueError):
                self.z = None
        self.xcache = {}

    def has(self, name):
        return name in self.src.t

    def fp8(self, name):
        key = ("fp8", name)
        if key not in self.cache:
            codes, dt, shape = self.src.raw(name, copy=False)
            sc, _sdt, sshape = self.src.raw(name[:-len("weight")] + "scale")
            self.cache[key] = (np.asarray(codes).reshape(shape), sc.reshape(sshape))
        return self.cache[key]

    def dense_f32(self, name):
        """A bfloat16 matrix as float32: BF12 values (near) or bfloat16 (exact)."""
        key = ("f32", name)
        if key not in self.cache:
            b = self.src.bf16(name)
            if not self.exact and b.ndim == 2 and b.shape[1] % 32 == 0 and b.shape[0] > 1:
                rows, _rep = cops.kq_bf16_to_bf12(b, b.shape[1])
                b = cops.kq_bf12_to_bf16(rows, b.shape[0], b.shape[1])
            self.cache[key] = bits_f32(b)
        return self.cache[key]

    def vec(self, name):
        key = ("vec", name)
        if key not in self.cache:
            self.cache[key] = self.src.f32(name)
        return self.cache[key]

    def embed_rows(self, ids):
        b, _dt, shape = self.src.raw("embed.weight", copy=False)
        rows = np.asarray(b).view(np.uint16).reshape(shape)[np.asarray(ids)]
        if not self.exact:
            r, _rep = cops.kq_bf16_to_bf12(np.ascontiguousarray(rows), shape[1])
            rows = cops.kq_bf12_to_bf16(r, rows.shape[0], shape[1])
        return bits_f32(rows)

    def expert(self, layer, e):
        """[(gate+up part, the row range of inner), ...], [(down part, rows of hidden), ...]:
        the KQ_MXX parts of an expert (a part for each node of the zygote, or
        one for all the rows)."""
        key = (layer, e)
        if key in self.xcache:
            return self.xcache[key]
        H, I = self.src.hidden, self.src.inner
        gu, dn = [], []
        if self.z is not None and self.z.index["layout"] != "dsv41-expert-planes-v1":
            ix = self.z.index
            for part in range(len(ix["segments"])):
                ms = self.z.mats(layer, part)
                (g0, ng), (d0, nd) = ix["gu"][part], ix["dn"][part]
                gu.append((ms["w1"][e], ms["w3"][e], g0, ng))
                dn.append((ms["w2"][e], d0, nd))
        elif self.z is not None:
            ix = self.z.index
            for part in range(len(ix["segments"])):
                pl = self.z.planes(layer, part)
                (g0, ng), (d0, nd) = ix["gu"][part], ix["dn"][part]
                w1, _ = cops.kq_mxx_pack(pl["w1.weight"][e], pl["w1.scale"][e], ng, H)
                w3, _ = cops.kq_mxx_pack(pl["w3.weight"][e], pl["w3.scale"][e], ng, H)
                w2, _ = cops.kq_mxx_pack(pl["w2.weight"][e], pl["w2.scale"][e], nd, I)
                gu.append((w1, w3, g0, ng))
                dn.append((w2, d0, nd))
        else:
            ex = self.src.expert(layer, e)
            w1, _ = cops.kq_mxx_pack(*ex["w1"], I, H)
            w3, _ = cops.kq_mxx_pack(*ex["w3"], I, H)
            w2, _ = cops.kq_mxx_pack(*ex["w2"], H, I)
            gu.append((w1, w3, 0, I))
            dn.append((w2, 0, H))
        if len(self.xcache) > 256:              # 18.8 MB each
            self.xcache.clear()
        self.xcache[key] = (gu, dn)
        return gu, dn


class DSV41CPU:
    def __init__(self, path, exact=False, max_len=None, zygote=True):
        """max_len: the most positions of a sequence (default the model's,
        1M: the float32 caches here are reserved, not touched, until used)."""
        self.path = path
        self.exact = exact
        self.w = Weights(path, exact, zygote)
        c = self.w.src.cfg
        self.c = c
        self.L = int(c["num_hidden_layers"])
        self.dim = int(c["hidden_size"])
        self.hc = int(c["hc_mult"])
        self.eps = float(c["rms_norm_eps"])
        self.hc_eps = float(c["hc_eps"])
        self.sink_iters = int(c["hc_sinkhorn_iters"])
        self.heads = int(c["num_attention_heads"])
        self.head_dim = int(c["head_dim"])
        self.rd = int(c["qk_rope_head_dim"])
        self.win = int(c["sliding_window"])
        self.groups = int(c["o_groups"])
        self.o_rank = int(c["o_lora_rank"])
        self.ratios = list(c["compress_ratios"])
        self.kv_sources = set(c["kv_source_layer_ids"])
        self.index_sources = set(c["index_source_layer_ids"])
        self.cand_src = int(c["candidate_source_layer_id"])
        self.cand_topk, self.cand_bs = int(c["candidate_topk_blocks"]), int(c["candidate_block_size"])
        self.ix_heads, self.ix_dim, self.ix_topk = int(c["index_n_heads"]), int(c["index_head_dim"]), int(c["index_topk"])
        self.topk = int(c["num_experts_per_tok"])
        self.route_scale = float(c["routed_scaling_factor"])
        self.limit = float(c["swiglu_limit"])
        self.engram_layers = list(c["engram_layer_ids"])
        self.max_len = max_len = int(max_len or c.get("max_position_embeddings", 4096))
        rs = c["rope_scaling"]
        self.freq_win = _rope_freqs(self.rd, max_len, 0, c["rope_theta"], rs["factor"], rs["beta_fast"], rs["beta_slow"])
        self.freq_cmp = _rope_freqs(self.rd, max_len, rs["original_max_position_embeddings"], c["compress_rope_theta"],
                                    rs["factor"], rs["beta_fast"], rs["beta_slow"])
        self.hashes = Hashes(c, os.path.join(path, "tokenizer.json"), max_len)
        self.reset()

    # ---- the state of one sequence
    def reset(self):
        hd = self.head_dim
        self.window = [np.zeros((self.win, hd), np.float32) for _ in range(self.L)]
        self.ckv = {i: np.zeros((self.max_len // self.ratios[i], hd), np.float32) for i in self.kv_sources}
        self.kstate = {i: np.zeros((self.ratios[i], hd), np.float32) for i in self.kv_sources if self.ratios[i] > 1}
        self.sstate = {i: np.full((self.ratios[i], hd), -np.inf, np.float32) for i in self.kv_sources
                       if self.ratios[i] > 1}
        self.kcache = {i: np.zeros((self.max_len // self.ratios[i], self.ix_dim), np.float32) for i in self.kv_sources}
        self.shared = {}

    # ---- helpers
    def R(self, x):
        return bf16r(x) if self.exact else np.asarray(x, np.float32)

    def norm(self, x, name):
        x = x.astype(np.float32)
        y = x / np.sqrt((x * x).mean(-1, keepdims=True) + np.float32(self.eps))
        return self.R(self.w.vec(name) * y)

    def lin_fp8(self, x, name):
        codes, sc = self.w.fp8(name)
        shp = x.shape
        y = cops.kq_fp8b32_linear(codes, sc, codes.shape[0], codes.shape[1], x.reshape(-1, shp[-1]),
                                  exact=self.exact)
        return self.R(y.reshape(shp[:-1] + (codes.shape[0],)))

    def lin_bf16(self, x, name, out_f32=False):
        w = self.w.dense_f32(name)
        y = (self.R(x).astype(np.float32) @ w.T).astype(np.float32)
        return y if out_f32 else self.R(y)

    def mxx(self, x, w, rows, cols):
        y = cops.kq_mxx_linear(w, rows, cols, x, exact=self.exact)
        return self.R(y)

    # ---- hyper-connections
    def hc_mixes(self, x, pre):
        T = x.shape[0]
        f = x.reshape(T, -1).astype(np.float32)
        rs = 1.0 / np.sqrt((f * f).mean(-1, keepdims=True) + np.float32(self.eps))
        mixes = (f @ self.w.vec(pre + "_fn").T) * rs
        sc, base = self.w.vec(pre + "_scale"), self.w.vec(pre + "_base")
        hc, eps = self.hc, np.float32(self.hc_eps)
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        p = sig(mixes[:, :hc] * sc[0] + base[:hc]) + eps
        q = 2 * sig(mixes[:, hc:2 * hc] * sc[1] + base[hc:2 * hc])
        comb = (mixes[:, 2 * hc:] * sc[2] + base[2 * hc:]).reshape(T, hc, hc)
        comb = softmax(comb, -1) + eps
        comb = comb / (comb.sum(-2, keepdims=True) + eps)
        for _ in range(self.sink_iters - 1):
            comb = comb / (comb.sum(-1, keepdims=True) + eps)
            comb = comb / (comb.sum(-2, keepdims=True) + eps)
        return p.astype(np.float32), q.astype(np.float32), comb.astype(np.float32)

    def hc_pre(self, x, mix):
        return self.R((mix[..., None] * x).sum(1))

    def hc_post(self, y, res, post, comb):
        out = post[..., None] * y[:, None, :] + (comb[..., None] * res[:, :, None, :]).sum(1)
        return self.R(out)

    # ---- engram
    def engram(self, i, h, ids):
        pre = "layers.%d.engram." % i
        b, _dt, shape = self.w.src.raw(pre + "embed.weight", copy=False)
        s, _sdt, sshape = self.w.src.raw(pre + "embed.scale", copy=False)
        table = np.asarray(b).reshape(shape)
        scales = np.asarray(s).reshape(sshape)
        flat = ids.reshape(-1)
        vals = D.E4M3[table[flat]].reshape(-1, shape[1] // 32, 32) * D.E8M0[scales[flat]][..., None]
        emb = self.R(vals.reshape(ids.shape[0], -1))                  # [T, 24 * 256]
        kv = self.lin_fp8(emb, pre + "wkv.weight")                    # [T, 5 * dim]
        key = kv[:, :self.hc * self.dim].reshape(-1, self.hc, self.dim).astype(np.float32)
        value = kv[:, self.hc * self.dim:].astype(np.float32)
        wq = self.w.vec(pre + "q_weight") * self.w.vec(pre + "k_weight")
        hf = h.astype(np.float32)
        eps = np.float32(self.eps)
        rstd = 1.0 / np.sqrt((hf * hf).mean(-1) + eps) / np.sqrt((key * key).mean(-1) + eps)
        dot = (hf * wq * key).sum(-1) * rstd * np.float32(self.dim ** -0.5)
        gate = 1.0 / (1.0 + np.exp(-np.copysign(np.sqrt(np.maximum(np.abs(dot), np.float32(1e-6))), dot)))
        return self.R(hf + gate[..., None] * value[:, None, :])

    # ---- attention
    def attention(self, i, x, start):
        T = x.shape[0]
        pre = "layers.%d.attn." % i
        ratio = self.ratios[i]
        fr = self.freq_cmp if ratio else self.freq_win
        cs = (fr[0][start:start + T], fr[1][start:start + T])
        rd = self.rd
        qr = self.norm(self.lin_fp8(x, pre + "wq_a.weight"), pre + "q_norm.weight")
        q = self.lin_fp8(qr, pre + "wq_b.weight").reshape(T, self.heads, self.head_dim)
        q = q.copy()
        q[..., -rd:] = self.R(rope(q[..., -rd:], (cs[0][:, None], cs[1][:, None])))
        # the window
        kv = self.norm(self.lin_fp8(x, pre + "wkv.weight"), pre + "kv_norm.weight")
        kv = kv.copy()
        kv[:, -rd:] = self.R(rope(kv[:, -rd:], cs))
        kv = act_quant(kv)
        ring = self.window[i]
        if start == 0:
            for t in range(max(0, T - self.win), T):
                ring[t % self.win] = kv[t]
            win_kv = kv
            end = np.arange(T)[:, None]
            idx = np.clip(end - self.win + 1, 0, None) + np.arange(min(T, self.win))[None]
            win_idx = np.where(idx > end, -1, idx)
        else:
            assert T == 1
            ring[start % self.win] = kv[0]
            win_kv = ring
            oldest = start % self.win + 1
            idx = np.concatenate([np.arange(oldest, self.win), np.arange(oldest)])
            win_idx = np.where(idx > start, -1, idx)[None]
        kvs, idxs = win_kv, win_idx
        if ratio:
            ckv, cidx = self.compressed(i, x, qr, start, win_kv.shape[0])
            kvs = np.concatenate([win_kv, ckv], 0)
            idxs = np.concatenate([win_idx, cidx], 1)
        o = self.sparse_attn(q, kvs, self.w.vec(pre + "attn_sink"), idxs)
        o = o.copy()
        o[..., -rd:] = self.R(rope(o[..., -rd:], (cs[0][:, None], cs[1][:, None]), inverse=True))
        # wo_a: block diagonal over the groups (FP8, the reference's bfloat16 values: exact)
        codes, sc = self.w.fp8(pre + "wo_a.weight")
        og = o.reshape(T, self.groups, -1)
        outs = []
        for g in range(self.groups):
            rows = slice(g * self.o_rank, (g + 1) * self.o_rank)
            wg = D.dequant_fp8(codes[rows], sc[g * self.o_rank // 32:(g + 1) * self.o_rank // 32])
            outs.append(self.R(og[:, g].astype(np.float32)) @ wg.T)
        oa = self.R(np.concatenate(outs, -1).astype(np.float32))
        return self.lin_fp8(oa, pre + "wo_b.weight")

    def sparse_attn(self, q, kv, sink, idxs):
        T, H, d = q.shape
        scale = np.float32(d ** -0.5)
        out = np.zeros_like(q)
        for t in range(T):
            ii = idxs[t][idxs[t] >= 0]
            if ii.size == 0:
                continue
            k = self.R(kv[ii])
            s = (self.R(q[t]) @ k.T) * scale                     # [H, n]
            mx = s.max(-1, keepdims=True)
            p = np.exp(s - mx)
            den = p.sum(-1, keepdims=True) + np.exp(sink[:, None] - mx)
            out[t] = self.R((self.R(p) @ k) / den)
        return out

    def compressed(self, i, x, qr, start, offset):
        """The compressed KV this layer reads and the positions each query takes."""
        T = x.shape[0]
        ratio = self.ratios[i]
        clen = (start + T) // ratio
        latent = None
        if i in self.kv_sources:
            latent = self.compressor(i, x, start)
            self.shared["ckv"] = self.ckv[i]
            self.shared["ckv_owner"] = i
        idxs = self.index(i, x, qr, latent, start, offset, clen)
        if latent is not None:
            fr = self.freq_cmp
            pos = np.arange(0, T - T % ratio, ratio) if start == 0 else np.array([start + 1 - ratio])
            lat = latent.copy()
            lat[:, -self.rd:] = self.R(rope(lat[:, -self.rd:], (fr[0][pos], fr[1][pos])))
            lat = fp4_quant(lat, 16, True)
            self.ckv[i][start // ratio:start // ratio + lat.shape[0]] = lat
        return self.shared["ckv"][:clen], idxs

    def compressor(self, i, x, start):
        pre = "layers.%d.attn.compressor." % i
        ratio = self.ratios[i]
        T = x.shape[0]
        if ratio == 1:
            return self.norm(self.lin_bf16(x, pre + "wkv.weight"), pre + "norm.weight")
        xf = x.astype(np.float32)
        kv = xf @ self.w.dense_f32(pre + "wkv.weight").T
        sc = xf @ self.w.dense_f32(pre + "wgate.weight").T
        if start == 0:
            rem = T % ratio
            cut = T - rem
            if rem:
                self.kstate[i][:rem], self.sstate[i][:rem] = kv[cut:], sc[cut:]
            if cut < ratio:
                return None
            kvg = kv[:cut].reshape(-1, ratio, kv.shape[-1])
            scg = sc[:cut].reshape(-1, ratio, sc.shape[-1])
            pooled = (kvg * softmax(scg, 1)).sum(1)
        else:
            slot = start % ratio
            self.kstate[i][slot], self.sstate[i][slot] = kv[0], sc[0]
            if (start + 1) % ratio:
                return None
            pooled = (self.kstate[i] * softmax(self.sstate[i], 0)).sum(0, keepdims=True)
        return self.norm(self.R(pooled), pre + "norm.weight")

    def index(self, i, x, qr, latent, start, offset, clen):
        if i not in self.index_sources:
            return self.shared["idxs"]
        T = x.shape[0]
        if clen == 0:
            self.shared["idxs"] = np.zeros((T, 0), np.int64)
            return self.shared["idxs"]
        pre = "layers.%d.attn.indexer." % i
        ratio = self.ratios[i]
        fr = self.freq_cmp
        rd = self.rd
        end = start + T
        if i in self.kv_sources and latent is not None:
            pos = np.arange(0, T - T % ratio, ratio) if start == 0 else np.array([start + 1 - ratio])
            k = self.norm(self.lin_bf16(latent, pre + "wk.weight"), pre + "k_norm.weight").copy()
            k[:, -rd:] = self.R(rope(k[:, -rd:], (fr[0][pos], fr[1][pos])))
            k = fp4_quant(k, 32, False)
            self.kcache[i][start // ratio:start // ratio + k.shape[0]] = k
        if i in self.kv_sources:
            self.shared["index_k"] = self.kcache[i]
        q = self.lin_fp8(qr, pre + "wq_b.weight").reshape(T, self.ix_heads, self.ix_dim).copy()
        q[..., -rd:] = self.R(rope(q[..., -rd:], (fr[0][start:end][:, None], fr[1][start:end][:, None])))
        q = fp4_quant(q, 32, False)
        ik = self.shared["index_k"][:end // ratio]
        wts = self.R(self.lin_bf16(x, pre + "weights_proj.weight") *
                     np.float32(self.ix_dim ** -0.5 * self.ix_heads ** -0.5))
        sc = self.R(np.einsum("shd,td->sht", q, ik))
        sc = self.R((np.maximum(sc, 0) * wts[..., None]).sum(1))      # [T, n]
        if start == 0:
            clens = (np.arange(1, T + 1) // ratio)[:, None]
            sc = np.where(np.arange(T // ratio)[None] >= clens, -np.inf, sc)
        else:
            clens = end // ratio
        if i == self.cand_src:
            self.shared["cand"] = self.candidates(sc, clens)
        elif 0 <= self.cand_src < i:
            sc = np.where(self.shared["cand"], sc, -np.inf)
        k = min(self.ix_topk, end // ratio)
        top = np.sort(np.argsort(-sc, -1, kind="stable")[:, :k], -1)
        idxs = np.where(top < clens, top + offset, -1)
        self.shared["idxs"] = idxs
        return idxs

    def candidates(self, logits, clens):
        width = logits.shape[-1]
        bs = self.cand_bs
        pad = -width % bs
        s = np.concatenate([logits, np.full(logits.shape[:-1] + (pad,), -np.inf)], -1)
        s = s.reshape(logits.shape[:-1] + (-1, bs)).max(-1)
        nb = s.shape[-1]
        last = (np.asarray(clens) - 1) // bs
        s = np.where(np.arange(nb)[None] == np.broadcast_to(last, (s.shape[0], 1)), np.inf, s)
        k = min(self.cand_topk, nb)
        top = np.argsort(-s, -1, kind="stable")[:, :k]
        keep = np.zeros_like(s, dtype=bool)
        np.put_along_axis(keep, top, np.take_along_axis(s, top, -1) > -np.inf, -1)
        return np.repeat(keep, bs, -1)[:, :width]

    # ---- MoE
    def moe(self, i, x):
        pre = "layers.%d.ffn." % i
        T = x.shape[0]
        sc = x.astype(np.float32) @ self.w.dense_f32(pre + "gate.weight").T
        sc = np.sqrt(np.logaddexp(0, sc)).astype(np.float32)          # sqrt(softplus)
        sel = sc + self.w.vec(pre + "gate.bias")
        idx = np.argsort(-sel, -1, kind="stable")[:, :self.topk]
        if getattr(self, "force_routes", None) is not None:
            idx = np.asarray(self.force_routes[:, i], np.int64)     # a check: the reference's choices
        wts = np.take_along_axis(sc, idx, -1)
        wts = wts / (wts.sum(-1, keepdims=True) + np.float32(1e-20)) * np.float32(self.route_scale)
        y = np.zeros((T, self.dim), np.float32)
        H, I = self.dim, self.w.src.inner
        for e in np.unique(idx):
            rows, cols = np.nonzero(idx == e)
            xe = x[rows]
            gu, dn = self.w.expert(i, int(e))
            g = np.zeros((len(rows), I), np.float32)
            u = np.zeros((len(rows), I), np.float32)
            for w1, w3, g0, ng in gu:
                g[:, g0:g0 + ng] = self.mxx(xe, w1, ng, H)
                u[:, g0:g0 + ng] = self.mxx(xe, w3, ng, H)
            u = np.clip(u, -self.limit, self.limit)
            g = np.minimum(g, self.limit)
            act = (g / (1 + np.exp(-g))) * u * wts[rows, cols][:, None]
            act = self.R(act)
            out = np.zeros((len(rows), H), np.float32)
            for w2, d0, nd in dn:
                out[:, d0:d0 + nd] = self.mxx(act, w2, nd, I)
            y[rows] += out
        sx = pre + "shared_experts."
        g = self.lin_fp8(x, sx + "w1.weight").astype(np.float32)
        u = self.lin_fp8(x, sx + "w3.weight").astype(np.float32)
        u = np.clip(u, -self.limit, self.limit)
        g = np.minimum(g, self.limit)
        y += self.lin_fp8(self.R((g / (1 + np.exp(-g))) * u), sx + "w2.weight")
        self.last_route = idx
        return self.R(y)

    # ---- the model
    def forward(self, ids, start, keep=False):
        ids = np.asarray(ids, np.int64)
        T = len(ids)
        hashes = self.hashes.ids(ids, start)
        h = np.repeat(self.w.embed_rows(ids)[:, None, :], self.hc, 1)
        pre_mix = np.zeros((T, self.hc), np.float32)
        pre_mix[:, 0] = 1
        outs, routes = [], []
        for i in range(self.L):
            p = "layers.%d." % i
            if i in self.engram_layers:
                h = self.engram(i, h, hashes[:, self.engram_layers.index(i)])
            res = h
            ap, apost, acomb = self.hc_mixes(h, p + "hc_attn")
            x = self.norm(self.hc_pre(h, pre_mix), p + "attn_norm.weight")
            x = self.attention(i, x, start)
            h = self.hc_post(x, res, apost, acomb)
            res = h
            fp, fpost, fcomb = self.hc_mixes(h, p + "hc_ffn")
            x = self.norm(self.hc_pre(h, ap), p + "ffn_norm.weight")
            x = self.moe(i, x)
            h = self.hc_post(x, res, fpost, fcomb)
            pre_mix = fp
            routes.append(self.last_route)
            if keep:
                outs.append(h.astype(np.float32))
        x = self.norm(self.hc_pre(h, pre_mix), "norm.weight")
        logits = x.astype(np.float32) @ self.w.dense_f32("head.weight").T
        self.routes = np.stack(routes, 1)
        return logits, (np.stack(outs, 1) if keep else None)
