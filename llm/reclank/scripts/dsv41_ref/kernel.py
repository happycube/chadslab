"""The ops of the reference's kernel.py (DeepSeek V4.1 Flash, inference/kernel.py)
in plain torch on the CPU, for scripts/dsv41_reference.py (DEEPSEEK_V41_PLAN.md
Phase 0). The model code imports this module in place of the tilelang one.

The rounding is the kernels': act_quant takes the amax of each group (at
least 1e-4), the scale amax * (1/448) or, with scale_fmt, the power of two
above it (fast_round_scale: the exponent of the float32 bits, plus one when
any mantissa bit is set), x / s clamped to +-448 and cast to E4M3 (nearest,
ties to even); fp4_act_quant the same with 6 and E2M1 (E8M0 scales: amax at
least 6 * 2^-126; E4M3 scales of the compressed KV: amax at least 6 * 2^-9,
s = E4M3(amax / 6)). The GEMMs are float32 products of the dequantized
values (the kernels sum each 32 of K in float32, scale, and add: the same
values, other roundings of the float32 sums). sparse_attn: the scores in
float32, the softmax with the sink, the probabilities rounded to bfloat16
before the product with V (acc_s_cast of the kernel), float32 sums.
"""
from __future__ import annotations

import torch

FP8_MAX = 448.0
FP4_MAX = 6.0
_INV448 = torch.tensor(1.0 / 448.0, dtype=torch.float32)
_INV6 = torch.tensor(1.0 / 6.0, dtype=torch.float32)

# E2M1 codes 0..15 (convert.py FP4_TABLE)
FP4_TABLE = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                          -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=torch.float32)


def _pow2_ceil_log2(v):
    """fast_round_scale: 2^ceil(log2(v)) from the float32 bits of v (v > 0, normal)."""
    bits = v.float().contiguous().view(torch.int32)
    e = ((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()
    return ((e + 127) << 23).view(torch.float32)


def _groups(x, block):
    return x.float().unflatten(-1, (-1, block))


def act_quant(x, block_size=128, scale_fmt=None, scale_dtype=torch.float32, inplace=False):
    g = _groups(x, block_size)
    amax = g.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    s = _pow2_ceil_log2(amax * _INV448) if scale_fmt is not None else amax * _INV448
    y = (g / s).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    if inplace:
        x.copy_((y.float() * s).flatten(-2).to(x.dtype))
        return x
    return y.flatten(-2), s.squeeze(-1).to(scale_dtype)


def _to_e2m1(v):
    """float32 (already clamped to +-6) to the nearest E2M1 value, ties to the
    even code (cvt.rn.satfinite.e2m1x2.f32)."""
    a = v.abs()
    grid = FP4_TABLE[:8]
    # the two neighbours on the grid, and the tie to the even code
    hi = torch.searchsorted(grid, a.contiguous(), right=False).clamp(max=7)
    lo = (hi - 1).clamp(min=0)
    dlo, dhi = a - grid[lo], grid[hi] - a
    pick = torch.where(dlo < dhi, lo, torch.where(dhi < dlo, hi, torch.where(lo % 2 == 0, lo, hi)))
    out = grid[pick]
    return torch.where(v < 0, -out, out)


def fp4_act_quant(x, block_size=32, inplace=False, scale_dtype=torch.float8_e8m0fnu):
    assert inplace, "the model only quantizes in place"
    g = _groups(x, block_size)
    amax = g.abs().amax(-1, keepdim=True)
    if scale_dtype == torch.float8_e4m3fn:
        amax = amax.clamp_min(6 * 2.0 ** -9)
        s = (amax / FP4_MAX).to(torch.float8_e4m3fn).float()
    else:
        amax = amax.clamp_min(6 * 2.0 ** -126)
        s = _pow2_ceil_log2(amax * _INV6)
    y = _to_e2m1((g / s).clamp(-FP4_MAX, FP4_MAX))
    x.copy_((y * s).flatten(-2).to(x.dtype))
    return x


def _dequant_blocks(b, b_s, block):
    """An FP8 matrix (N, K) with scales of each block x block."""
    w = b.float()
    s = b_s.float().repeat_interleave(block, 0).repeat_interleave(block, 1)
    return w * s[:w.size(0), :w.size(1)]


def _dequant_act(a, a_s, block):
    return (a.float().unflatten(-1, (-1, block)) * a_s.float().unsqueeze(-1)).flatten(-2)


def fp8_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, block_size=128):
    x = _dequant_act(a, a_s, block_size)
    w = _dequant_blocks(b, b_s, block_size)
    return (x @ w.t()).to(torch.get_default_dtype())


def fp4_dequant(b, b_s):
    """An MXFP4 matrix (N, K / 2 bytes, low nibble first) with its E8M0 scale of
    each 32 values."""
    u = b.view(torch.uint8)
    v = torch.stack([FP4_TABLE[(u & 15).long()], FP4_TABLE[(u >> 4).long()]], dim=-1).flatten(-2)
    return (v.unflatten(-1, (-1, 32)) * b_s.float().unsqueeze(-1)).flatten(-2)


def fp4_gemm(a, a_s, b, b_s, scale_dtype=torch.float32, act_block_size=128):
    x = _dequant_act(a, a_s, act_block_size)
    return (x @ fp4_dequant(b, b_s).t()).to(torch.get_default_dtype())


def sparse_attn(q, kv, attn_sink, topk_idxs, softmax_scale):
    """q [b, m, h, d], kv [b, n, d], topk_idxs [b, m, k] (-1: none)."""
    b, m, h, d = q.shape
    out = torch.zeros_like(q)
    for bi in range(b):
        for mi in range(m):
            idx = topk_idxs[bi, mi].long()
            idx = idx[idx >= 0]
            if idx.numel() == 0:
                continue
            k = kv[bi, idx].to(torch.bfloat16)
            qq = q[bi, mi].to(torch.bfloat16)
            s = (qq.float() @ k.float().t()) * softmax_scale            # h x k
            mx = s.amax(-1, keepdim=True)
            p = torch.exp(s - mx)
            den = p.sum(-1, keepdim=True) + torch.exp(attn_sink.float().unsqueeze(-1) - mx)
            o = p.to(torch.bfloat16).float() @ k.float()
            out[bi, mi] = (o / den).to(q.dtype)
    return out


def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult=4, sinkhorn_iters=20, eps=1e-6):
    hc = hc_mult
    m = mixes.float()
    pre = torch.sigmoid(m[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(m[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (m[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).unflatten(-1, (hc, hc))
    comb = torch.softmax(comb, dim=-1) + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(sinkhorn_iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb
