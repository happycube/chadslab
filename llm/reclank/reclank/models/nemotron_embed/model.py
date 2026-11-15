"""Nemotron 3.5 Embed (nvidia/Nemotron-3.5-Embed-1B-BF16, -8B-BF16) and
Nemotron 3.5 Rerank (nvidia/Nemotron-3.5-Rerank-8B-BF16): a Pixtral vision
tower and a bidirectional Ministral3 text model (transformers' Mistral3Model).
An embedding is the mean of the last hidden state over every token of the
input, L2-normalized (2048 floats for the 1B, 4096 for the 8B; Matryoshka: the
first d values, normalized again). A rerank score is the logit of the
checkpoint's model.py: the same mean, times the score row (no activation).

Ported from transformers (src/transformers/models/ministral3/modeling_ministral3.py,
models/mistral3/modeling_mistral3.py, models/pixtral/modeling_pixtral.py and
image_processing_pixtral.py; Apache License 2.0; Copyright 2024-2025 Mistral AI
and the HuggingFace Inc. team). See THIRD_PARTY_NOTICES.md.

The text model (Ministral3Model with the config's is_causal false):

    the token embeddings (no scale); the rows of the image placeholders [IMG]
        take the projector's rows
    each layer: RMS norm, q k v (heads of 128, KV heads shared: 1B 24/8, 8B
        32/8), RoPE (NEOX halves; YaRN of factor 16 over 16384 positions,
        theta 10^6, attention factor 1), q times 1 + 0.1 log(1 + floor(pos /
        16384)) (Llama 4's scale: 1 below 16384), attention over every token
        of the input (scale 128^-0.5), o, + x; RMS norm, the gated FFN with
        SiLU, + x
    the final norm

The vision tower (PixtralVisionModel, 24 layers of width 1024, 16 heads of 64)
and the projector (Mistral3MultiModalProjector):

    patches of 14 x 14 (channel, row, column) -> the patch linear (no bias)
    -> RMS norm (ln_pre) -> layers (RMS norm, q k v, the 2D RoPE of theta 10^4:
    angle j of a head the row times the even frequencies, then the column
    times the odd ones, on the NEOX halves of 64; attention over the patches
    of the image (scale 1/8), o, + x; RMS norm, the gated FFN with SiLU, + x)
    -> RMS norm (eps 1e-5) -> each 2 x 2 block of patches one row of 4096
    (the patch merger) -> linear to 1024 -> linear to the text width -> GELU
    (erf) -> linear

The image (PixtralImageProcessor): the longest edge at most 1120 (the
processor's default; up to 1540) with the aspect kept, each side up to a
multiple of 28; bicubic (PIL's; transformers resizes the uint8 values with
torchvision's antialiased bicubic) -> (x - 255 mean) / (255 std). An image of
h x w blocks of 28 is h rows of w [IMG] and one [IMG_BREAK], the last
[IMG_BREAK] an [IMG_END].

The inputs follow the checkpoints' chat templates (no BOS, no EOS):

    embed:  "query: " + text, or "passage: " + text; a document with an image
            "passage: " + the image + " " + text (or the image alone)
    rerank: "query: " + query + "\\n\\npassage: " + document; with an image
            the image + " query: ..." (the whole stripped)

A text of a user is plain text: "[IMG]" in it is the characters, not the
token (the templates mark "[" and "<" with a zero-width non-joiner, and the
tokenizer's normalizer takes the marks out after the split at the special
tokens).

The weights: the BF12 file of scripts/convert_nemotron_embed_bf12.py
(BF12_FILE: each matrix BF12, the bfloat16 values in 12.25 bits; the vectors
float32), else the bfloat16 safetensors of the download (the reference's
weights). Both towers run as programs (reclank/ir/program.py, the records
ENC_*, as EmbeddingGemma 2's): in C on the CPU (KQ_BF12X16 of the BF12 file,
or the bfloat16 matrices in groups of 16 rows; float32 products), or as one
CUDA graph each on the GPU (the matrices in BF12). The RoPE is ENC_ROPE_TAB at the
cos and sin of each row; the q tables hold the attention scale (and Llama 4's
scale), so the attention (ENC_ATTN_RANGE) has scale 1. The inputs of a pass
are packed: each row's keys are those of its input. The token table stays
on the host in bfloat16. RECLANK_ENC_PY=1 runs the NumPy layers of
forward_numpy instead (the reference of the programs). All the arithmetic is
float32 (on the GPU the BF12 matrices), as the float32 reference
(scripts/nemotron_embed_reference.py).
"""
from __future__ import annotations

import io
import json
import math
import os

import numpy as np

from reclank.core import ops
from reclank.encoders.runtime import _ENC_PY, _EncCompiler, _Plain, _Runner, _shape, to_bf12
from reclank.formats.gguf import BF12
from reclank.formats.safetensors import Shards, drop_pages
from reclank.models.embedgemma2.model import _bucket
from reclank.models.qwen.tokenizer import QwenTokenizer
from reclank.runtime.threads import team_threads

PATCH = 14
MERGE = 2
LONGEST_EDGE = 1120     # the processor's default
MAX_EDGE = 1540         # the model card: up to 1540
MAX_TOKENS = 32768      # the context of the model card
BF12_FILE = "model-bf12.gguf"   # scripts/convert_nemotron_embed_bf12.py
PATCH_CONV = "vision_tower.patch_conv.weight"
MERGER = "multi_modal_projector.patch_merger.merging_layer.weight"


def program_layout(name, x):
    """A tensor of the checkpoint (bfloat16 bits or float32) in the layout of the programs (the
    values the same): the patch conv (out, 3, 14, 14) as the linear (out, 608) (the 588 values of a
    patch, then zeros: columns in multiples of 32 for BF12); the patch merger's columns from the
    unfold's (channel, block position) to ENC_GATHER's (block position, channel)."""
    if name.endswith(PATCH_CONV):
        x = x.reshape(x.shape[0], -1)
        pad = -x.shape[1] % 32
        return np.ascontiguousarray(np.pad(x, ((0, 0), (0, pad)))) if pad else np.ascontiguousarray(x)
    if name.endswith(MERGER):
        m = MERGE * MERGE
        return np.ascontiguousarray(x.reshape(x.shape[0], -1, m).transpose(0, 2, 1).reshape(x.shape[0], -1))
    return x


class _Weights:
    """The tensors of a directory in the programs' layout (program_layout): the converted file
    (BF12_FILE: each matrix BF12, the vectors float32) when it is there, else the bfloat16
    safetensors of the download. vec(name) float32 values; lin(name) a _Plain of bfloat16 rows
    (lin.w, uint16) or of BF12 rows (lin.b12: uint8, rows x bf12_row_bytes; lin.w None);
    rows(name, ids) float32 rows of a table."""

    def __init__(self, path):
        f = os.path.join(path, BF12_FILE)
        if os.path.exists(f):
            from reclank.formats.gguf import GGUF
            self.g, self.st = GGUF(f, stage=False), None
            self.kind = "BF12"
        else:
            self.g, self.st = None, Shards(path)
            self.kind = "bfloat16"

    def has(self, name):
        return name in (self.g.tensors if self.g is not None else self.st.where)

    def vec(self, name):
        if self.g is not None:
            return np.ascontiguousarray(self.g.dequant(name), dtype=np.float32)
        return np.ascontiguousarray(self.st.get(name), dtype=np.float32)

    def _b12(self, name):
        """(BF12 rows, rows, cols) of a matrix of the BF12 file, else None."""
        raw, dims, t = self.g.raw(name)
        if t != BF12:
            return None
        cols, rows = int(dims[0]), int(dims[1])
        return raw.reshape(rows, -1), rows, cols

    def lin(self, name):
        if self.g is None:
            return _Plain(program_layout(name, np.ascontiguousarray(self.st.get_bf16(name))))
        q = self._b12(name)
        if q is None:       # (a matrix that BF12 could not hold: bfloat16 as stored)
            raw, dims, _t = self.g.raw(name)
            return _Plain(np.asarray(raw).view(np.uint16).reshape(int(dims[1]), int(dims[0])))
        lin = _Plain(None)
        lin.b12, lin.b12_shape = q[0], (q[1], q[2])
        return lin

    def rows(self, name, ids):
        if self.g is None:
            return ops.bf16_to_f32(self.st.get_bf16(name)[ids])
        from reclank.backends.cpu import cops
        q = self._b12(name)
        return ops.bf16_to_f32(cops.kq_bf12_to_bf16(q[0][ids], len(ids), q[2]))

    def width(self, name):
        """The columns of a table."""
        if self.g is None:
            return self.st.shape(name)[-1]
        return int(self.g.tensors[name][0][0])


def to_device(lin, gpu):
    """A linear's W for a program: BF12 rows (lin.b12) as the GPU's KQ_LINEAR rows (lin.kq) or the
    CPU's KQ_BF12X16 (lin.bf12; float32 products of the BF12 values), the map's pages of the CPU's
    dropped; bfloat16 rows (lin.w): BF12 on the GPU (to_bf12; a matrix that its rule refuses stays
    bfloat16), on the CPU as they are (the compiler packs them in groups of 16 rows). Return False
    when W stays bfloat16 on the GPU."""
    from reclank.models.gemma4.e4b import _KQ
    b12 = getattr(lin, "b12", None)
    if b12 is None:
        if not gpu:
            return True
        w = lin.w
        if to_bf12(lin):
            drop_pages(w)
            return True
        return False
    rows, cols = lin.b12_shape
    if gpu:
        lin.kq = _KQ(b12, BF12, rows, cols)
    else:
        from reclank.backends.cpu import cops
        lin.bf12 = _KQ(cops.kq_pack_bf12x16(b12, rows, cols), cops.KQ_BF12X16, rows, cols)
        drop_pages(b12)
    return True


def yarn_inv_freq(rp, hd):
    """(inverse frequencies (hd / 2) float32, attention factor) of transformers'
    _compute_yarn_parameters, op for op in float32."""
    base, factor = float(rp["rope_theta"]), float(rp["factor"])
    orig = int(rp["original_max_position_embeddings"])
    ms, msa = rp.get("mscale"), rp.get("mscale_all_dim")

    def get_mscale(scale, m=1.0):
        return 1.0 if scale <= 1 else 0.1 * m * math.log(scale) + 1.0
    att = rp.get("attention_factor")
    if att is None:
        att = float(get_mscale(factor, ms) / get_mscale(factor, msa)) if ms and msa else get_mscale(factor)
    fast, slow = rp.get("beta_fast") or 32, rp.get("beta_slow") or 1

    def corr(rot):
        return hd * math.log(orig / (rot * 2 * math.pi)) / (2 * math.log(base))
    low, high = corr(fast), corr(slow)
    if rp.get("truncate", True):
        low, high = math.floor(low), math.ceil(high)
    low, high = max(low, 0), min(high, hd - 1)
    if low == high:
        high += 0.001
    pos = (np.float32(base) ** (np.arange(0, hd, 2, dtype=np.float32) / np.float32(hd))).astype(np.float32)
    extra = np.float32(1) / pos
    inter = np.float32(1) / (np.float32(factor) * pos)
    ramp = np.clip((np.arange(hd // 2, dtype=np.float32) - np.float32(low)) / np.float32(high - low), 0, 1)
    ef = np.float32(1) - ramp
    return (inter * (np.float32(1) - ef) + extra * ef).astype(np.float32), float(att)


def _cos_sin(ang):
    """cos and sin (float32) of float32 angles, rounded from float64."""
    a = ang.astype(np.float64)
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


def load_image(src):
    """A PIL image in RGB of a path, the bytes of a file, or a PIL image."""
    from PIL import Image
    if isinstance(src, (bytes, bytearray)):
        src = Image.open(io.BytesIO(src))
    elif isinstance(src, (str, os.PathLike)):
        src = Image.open(src)
    return src.convert("RGB") if src.mode != "RGB" else src


def image_size(h, w, edge=LONGEST_EDGE):
    """(height, width) of the resized image (get_resize_output_image_size of
    the processor): the longest edge at most edge, each side up to a multiple
    of 28."""
    ratio = max(h / edge, w / edge)
    if ratio > 1:
        h, w = int(math.floor(h / ratio)), int(math.floor(w / ratio))
    step = PATCH * MERGE
    return ((h - 1) // step + 1) * step, ((w - 1) // step + 1) * step


def image_pixels(src, edge=LONGEST_EDGE, mean=None, std=None):
    """The pixel values [3, H, W] float32 of an image (the processor's)."""
    from PIL import Image
    im = load_image(src)
    H, W = image_size(im.height, im.width, edge)
    if (H, W) != (im.height, im.width):
        im = im.resize((W, H), Image.BICUBIC)
    x = np.asarray(im, dtype=np.uint8).transpose(2, 0, 1).astype(np.float32)
    # the fused rescale and normalize of transformers: (x - 255 mean) / (255 std)
    m = (np.asarray(mean, np.float32) * np.float32(255.0)).reshape(3, 1, 1)
    s = (np.asarray(std, np.float32) * np.float32(255.0)).reshape(3, 1, 1)
    return np.ascontiguousarray((x - m) / s, dtype=np.float32)


class _Vision:
    """The Pixtral tower and the projector: pixels to the rows of the [IMG]
    tokens (one program a count of patches, _bucket)."""

    def __init__(self, W, p, cfg, text_eps, owner):
        vc = cfg["vision_config"]
        self.width = vc["hidden_size"]
        self.heads = vc["num_attention_heads"]
        self.hd = self.width // self.heads
        self.layers = vc["num_hidden_layers"]
        self.eps = 1e-5                 # PixtralRMSNorm(eps=1e-5) of every norm of the tower
        self.text_eps = text_eps        # the projector's norm
        self.gpu = False
        f32 = lambda n: W.vec(p + n)  # noqa: E731
        bf = lambda n: W.lin(p + n)  # noqa: E731
        v = "vision_tower."
        self.patch = bf(PATCH_CONV)        # (1024, 3 x 14 x 14 and zeros to 608)
        self.ln_pre = f32(v + "ln_pre.weight")
        self.blk = []
        for i in range(self.layers):
            q = v + "transformer.layers.%d." % i
            self.blk.append(dict(
                ln_attn=f32(q + "attention_norm.weight"),
                q=bf(q + "attention.q_proj.weight"), k=bf(q + "attention.k_proj.weight"),
                v=bf(q + "attention.v_proj.weight"), o=bf(q + "attention.o_proj.weight"),
                ln_ffn=f32(q + "ffn_norm.weight"),
                gate=bf(q + "feed_forward.gate_proj.weight"), up=bf(q + "feed_forward.up_proj.weight"),
                down=bf(q + "feed_forward.down_proj.weight")))
        m = "multi_modal_projector."
        self.norm = f32(m + "norm.weight")
        self.merge = bf(MERGER)            # (its columns in the order of ENC_GATHER)
        self.lin1 = bf(m + "linear_1.weight")
        self.lin2 = bf(m + "linear_2.weight")
        theta = float((vc.get("rope_parameters") or {}).get("rope_theta", vc.get("rope_theta", 10000.0)))
        inv = (np.float32(1) / (np.float32(theta) ** (np.arange(0, self.hd, 2, dtype=np.float32)
                                                      / np.float32(self.hd)))).astype(np.float32)
        self.inv_h, self.inv_w = inv[0::2].copy(), inv[1::2].copy()
        self.runner = _Runner(owner._team(self.program), keep=4, keep_gpu=2)

    def lins(self):
        return [self.patch, self.merge, self.lin1, self.lin2] + \
            [b[k] for b in self.blk for k in ("q", "k", "v", "o", "gate", "up", "down")]

    def tables(self, h, w, n):
        """The cos and sin (n, 32) of the patches of an h x w grid (row-major;
        rows past h w: angle 0): the q tables times the attention scale."""
        r = np.repeat(np.arange(h, dtype=np.float32), w)
        c = np.tile(np.arange(w, dtype=np.float32), h)
        ang = np.zeros((n, self.hd // 2), np.float32)
        ang[:h * w] = np.concatenate([r[:, None] * self.inv_h[None], c[:, None] * self.inv_w[None]], axis=1)
        cos, sin = _cos_sin(ang)
        s = np.float32(self.hd ** -0.5)
        return cos * s, sin * s, cos, sin

    def program(self, n, gpu=False):
        """The program of n patches (n a multiple of 4): inputs "pix" (n, 588),
        "cq", "sq", "ck", "sk" (n, 32), "lo", "hi" (the keys of each row),
        "gidx" (n / 4, 4) (the patches of each 2 x 2 block, -1: none); output
        "soft" (n / 4, text width)."""
        w, H, hd = self.width, self.heads, self.hd
        inter = _shape(self.blk[0]["gate"])[0]
        tw = _shape(self.lin2)[0]
        kp = _shape(self.patch)[1]
        kmax = max(kp, w * MERGE * MERGE, inter, tw)
        c = _EncCompiler(self.eps, pack=not gpu, rows=n, kmax=kmax)
        c.attn_f32 = True       # (a GPU program: ENC_ATTN_TILE, float32)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        k4 = n // (MERGE * MERGE)
        c.env.update(pix=z(n, kp), x=z(n, w), h=z(n, w), t=z(n, w), q=z(n, w), k=z(n, w), v=z(n, w), o=z(n, w),
                     g=z(n, inter), u=z(n, inter), cq=z(n, hd // 2), sq=z(n, hd // 2), ck=z(n, hd // 2),
                     sk=z(n, hd // 2), lo=np.zeros(n, np.int32), hi=np.zeros(n, np.int32),
                     gidx=np.full((k4, MERGE * MERGE), -1, np.int32), mg=z(k4, w * MERGE * MERGE),
                     m1=z(k4, w), y1=z(k4, tw), soft=z(k4, tw), _scratch=np.zeros(n * kmax, np.float32))
        forms = [("set", "x", ("enc_linear", self.patch, "pix")),
                 ("set", "x", ("enc_rms", "x", self.ln_pre, w))]
        for b in self.blk:
            forms += [
                ("set", "h", ("enc_rms", "x", b["ln_attn"], w)),
                ("set", "q", ("enc_linear", b["q"], "h")),
                ("set", "k", ("enc_linear", b["k"], "h")),
                ("set", "v", ("enc_linear", b["v"], "h")),
                ("enc_rope_tab", "q", "cq", "sq", H, hd, True),
                ("enc_rope_tab", "k", "ck", "sk", H, hd, True),
                ("set", "o", ("enc_attn_range_x2", "q", "k", "v", "lo", "hi", H, hd)),
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", b["ln_ffn"], w)),
                ("set", "g", ("enc_linear", b["gate"], "h")),
                ("set", "u", ("enc_linear", b["up"], "h")),
                ("set", "g", ("enc_silu_mul", "g", "u")),
                ("set", "t", ("enc_linear", b["down"], "g")),
                ("enc_add", "x", "t"),
            ]
        forms += [("set", "h", ("enc_rms", "x", self.norm, w, self.text_eps)),
                  ("set", "mg", ("enc_gather", "h", "gidx", w)),
                  ("set", "m1", ("enc_linear", self.merge, "mg")),
                  ("set", "y1", ("enc_linear", self.lin1, "m1")),
                  ("set", "y1", ("enc_gelu", "y1", True)),
                  ("set", "soft", ("enc_linear", self.lin2, "y1"))]
        c.compile(("seq", *forms))
        return c.p.finish()

    def inputs(self, pixels, n):
        """The inputs of the program of n patches for pixels [3, H, W]."""
        _, Hp, Wp = pixels.shape
        h, w = Hp // PATCH, Wp // PATCH
        m = h * w
        pix = np.zeros((n, _shape(self.patch)[1]), np.float32)     # (the columns past 588: zeros)
        pix[:m, :3 * PATCH * PATCH] = pixels.reshape(3, h, PATCH, w, PATCH).transpose(1, 3, 0, 2, 4).reshape(m, -1)
        cq, sq, ck, sk = self.tables(h, w, n)
        r = np.arange(n, dtype=np.int32)
        lo, hi = r.copy(), r + 1
        lo[:m], hi[:m] = 0, m
        # block (i, j) of the 2 x 2 blocks: the patches (2i + a, 2j + b), a b in raster order
        h2, w2 = h // MERGE, w // MERGE
        bi, bj = np.meshgrid(np.arange(h2), np.arange(w2), indexing="ij")
        sub = [(a, b) for a in range(MERGE) for b in range(MERGE)]
        gidx = np.full((n // (MERGE * MERGE), MERGE * MERGE), -1, np.int32)
        gidx[:h2 * w2] = np.stack([((MERGE * bi + a) * w + MERGE * bj + b).reshape(-1) for a, b in sub], axis=1)
        return dict(pix=pix, cq=cq, sq=sq, ck=ck, sk=sk, lo=lo, hi=hi, gidx=gidx)

    def rows(self, pixels):
        """The rows (blocks, text width) of pixels [3, H, W] (H, W multiples of 28)."""
        m = (pixels.shape[1] // PATCH) * (pixels.shape[2] // PATCH)
        if _ENC_PY:
            return self.forward_numpy(pixels)
        n = _bucket(m)
        return self.runner.run(n, self.gpu, self.inputs(pixels, n), "soft")[:m // (MERGE * MERGE)]

    def forward_numpy(self, pixels):
        """The NumPy layers (the reference of the program)."""
        m = (pixels.shape[1] // PATCH) * (pixels.shape[2] // PATCH)
        d = self.inputs(pixels, m)
        lin = lambda x, l: np.matmul(x, _wt(l))  # noqa: E731
        rms = lambda x, w, eps=self.eps: ops.rms_norm(x, w, eps)  # noqa: E731
        H, hd = self.heads, self.hd
        x = rms(lin(d["pix"], self.patch), self.ln_pre)
        for b in self.blk:
            h = rms(x, b["ln_attn"])
            q = _rope_np(lin(h, b["q"]).reshape(m, H, hd), d["cq"], d["sq"])
            k = _rope_np(lin(h, b["k"]).reshape(m, H, hd), d["ck"], d["sk"])
            v = lin(h, b["v"]).reshape(m, H, hd)
            x = x + lin(_attention_np(q, k, v, d["lo"], d["hi"]).reshape(m, -1), b["o"])
            h = rms(x, b["ln_ffn"])
            g = lin(h, b["gate"])
            x = x + lin(g / (1 + np.exp(-g)) * lin(h, b["up"]), b["down"])
        h = rms(x, self.norm, self.text_eps)
        g = d["gidx"][:m // (MERGE * MERGE)]
        y = lin(lin(h[g].reshape(len(g), -1), self.merge), self.lin1)
        return lin(_gelu_erf(y), self.lin2)


def _gelu_erf(x):
    """gelu(x), the erf form (math.erf: forward_numpy alone)."""
    from math import erf
    e = np.frompyfunc(erf, 1, 1)(x.astype(np.float64) / math.sqrt(2)).astype(np.float64)
    return (0.5 * x * (1 + e)).astype(np.float32)


def _wt(lin):
    """The float32 transpose (in, out) of a matrix, for forward_numpy."""
    t = getattr(lin, "f32t", None)
    if t is None:
        w = lin.w
        if w is None and getattr(lin, "b12", None) is not None:
            from reclank.backends.cpu import cops
            w = cops.kq_bf12_to_bf16(lin.b12, *lin.b12_shape)
        if w is None:
            raise ValueError("the matrices are BF12 on the GPU; forward_numpy needs gpu=False")
        t = lin.f32t = np.ascontiguousarray(ops.bf16_to_f32(w).T)
    return t


def _rope_np(x, cos, sin):
    """x (n, heads, hd): NEOX halves at the table (n, hd / 2) of each row."""
    h = x.shape[-1] // 2
    c, s = cos[:, None, :], sin[:, None, :]
    a, b = x[..., :h].copy(), x[..., h:].copy()
    x[..., :h] = a * c - b * s
    x[..., h:] = b * c + a * s
    return x


def _attention_np(q, k, v, lo, hi, block=512):
    """Query i over the keys [lo[i], hi[i]) (scale 1; KV heads shared), with
    NumPy in blocks of queries."""
    n, H, hd = q.shape
    kv = k.shape[1]
    rep = H // kv
    out = np.empty_like(q)
    qs = q.reshape(n, kv, rep, hd).transpose(1, 2, 0, 3)
    ks, vs = k.transpose(1, 2, 0), v.transpose(1, 0, 2)
    for i0 in range(0, n, block):
        i1 = min(n, i0 + block)
        j0, j1 = int(lo[i0:i1].min()), int(hi[i0:i1].max())
        s = np.matmul(np.ascontiguousarray(qs[:, :, i0:i1]).reshape(kv, rep * (i1 - i0), hd),
                      np.ascontiguousarray(ks[:, :, j0:j1]))
        s = s.reshape(kv, rep, i1 - i0, j1 - j0)
        j = np.arange(j0, j1)[None, :]
        s[:, :, (j < lo[i0:i1, None]) | (j >= hi[i0:i1, None])] = -np.inf
        s -= s.max(-1, keepdims=True)
        np.exp(s, out=s)
        s /= s.sum(-1, keepdims=True)
        o = np.matmul(s.reshape(kv, rep * (i1 - i0), j1 - j0), np.ascontiguousarray(vs[:, j0:j1]))
        out[i0:i1] = o.reshape(kv, rep, i1 - i0, hd).transpose(2, 0, 1, 3).reshape(i1 - i0, H, hd)
    return out


class NemotronRetriever:
    """A Nemotron 3.5 Embed or Rerank directory (kind "embed" or "rerank").
    embed: embed_ids(seqs) the vectors; rerank: score_ids(seqs) the logits.
    gpu: the text model on the GPU (BF12); vision: "gpu", "cpu" (default that
    of the text model) or "off"; cpu_threads: the team of a program on the CPU
    (default the CPUs of the process)."""

    def __init__(self, path, gpu=False, vision=None, cpu_threads=None):
        self.path = path
        self.gpu = bool(gpu)
        dev = "gpu" if self.gpu else "cpu"
        self.devices = {"text": dev, "vision": vision or dev}
        if self.devices["vision"] not in ("gpu", "cpu", "off"):
            raise ValueError("the device of the vision tower must be gpu, cpu or off, not %r" % vision)
        self.cpu_threads = int(cpu_threads or team_threads())
        cfg = json.load(open(os.path.join(path, "config.json")))
        self.kind = "rerank" if "Mistral3ForSequenceClassification" in (cfg.get("architectures") or ()) \
            else "embed"
        tc = cfg["text_config"]
        self.width = tc["hidden_size"]
        self.layers = tc["num_hidden_layers"]
        self.heads = tc["num_attention_heads"]
        self.kv = tc["num_key_value_heads"]
        self.hd = int(tc.get("head_dim") or self.width // self.heads)
        self.eps = float(tc["rms_norm_eps"])
        self.dim = self.width
        rp = tc["rope_parameters"]
        self.inv, self.att = yarn_inv_freq(rp, self.hd)
        self.l4_beta = float(rp.get("llama_4_scaling_beta") or 0.0)
        self.l4_orig = int(rp["original_max_position_embeddings"])
        self.max_tokens = MAX_TOKENS
        pc = json.load(open(os.path.join(path, "processor_config.json")))["image_processor"]
        self.image_mean, self.image_std = pc["image_mean"], pc["image_std"]
        self.edge = int((pc.get("size") or {}).get("longest_edge", LONGEST_EDGE))

        self.tok = QwenTokenizer(os.path.join(path, "tokenizer.json"))
        sp = self.tok.special
        self.image_id = int(cfg.get("image_token_index", sp["[IMG]"]))
        self.break_id, self.end_id = sp["[IMG_BREAK]"], sp["[IMG_END]"]

        # the BF12 file of scripts/convert_nemotron_embed_bf12.py, else the bfloat16 safetensors
        W = self.weights = _Weights(path)
        p = "model." if self.kind == "rerank" else ""
        f32 = lambda n: W.vec(p + "language_model." + n)  # noqa: E731
        bf = lambda n: W.lin(p + "language_model." + n)  # noqa: E731
        self.table = p + "language_model.embed_tokens.weight"      # (its rows: W.rows)
        self.norm = f32("norm.weight")
        self.blk = []
        for i in range(self.layers):
            q = "layers.%d." % i
            self.blk.append(dict(
                ln_in=f32(q + "input_layernorm.weight"),
                q=bf(q + "self_attn.q_proj.weight"), k=bf(q + "self_attn.k_proj.weight"),
                v=bf(q + "self_attn.v_proj.weight"), o=bf(q + "self_attn.o_proj.weight"),
                ln_post=f32(q + "post_attention_layernorm.weight"),
                gate=bf(q + "mlp.gate_proj.weight"), up=bf(q + "mlp.up_proj.weight"),
                down=bf(q + "mlp.down_proj.weight")))
        # the reranker's score row (model.py: linear(pooled.float(), score.weight.float()))
        self.score_w = W.vec("language_model.score.weight").reshape(-1) if self.kind == "rerank" else None
        self._vision = None
        self._vision_args = (W, p, cfg)
        self._to_device(self.lins(), self.gpu, "text")
        self._pages = None if self.gpu or W.kind == "BF12" else \
            [b[k].w for b in self.blk for k in ("q", "k", "v", "o", "gate", "up", "down")]
        self.runner = _Runner(self._team(self.program), keep=6, keep_gpu=3)

    def lins(self):
        return [b[k] for b in self.blk for k in ("q", "k", "v", "o", "gate", "up", "down")]

    def _to_device(self, lins, gpu, part):
        """The matrices for the programs of a device (to_device)."""
        kept = sum(not to_device(lin, gpu) for lin in lins)
        if kept:
            print("%s %s: %d of %d matrices stay bfloat16 (not BF12)" % (
                os.path.basename(self.path.rstrip("/")), part, kept, len(lins)), flush=True)

    def _team(self, build):
        """build(n, gpu) that gives a CPU program the team of cpu_threads."""
        def make(n, gpu):
            p = build(n, gpu)
            if not gpu:
                p.threads = self.cpu_threads
                p.buf[3] = self.cpu_threads     # word 3 of the header: the team
            return p
        return make

    # ---- the tokens -------------------------------------------------------------

    def plain(self, text):
        """The ids of text with no special tokens (a user's text: the
        templates' marks)."""
        tok, out = self.tok, []
        for w in tok._split(text):
            ids = tok._words.get(w)
            if ids is None:
                ids = tok._bpe(w.encode("utf-8").decode("latin-1").translate(tok._bytes_tr))
            out += ids
        return out

    def image_ids(self, h2, w2):
        """The ids of an image of h2 x w2 blocks."""
        row = [self.image_id] * w2 + [self.break_id]
        ids = row * h2
        ids[-1] = self.end_id
        return ids

    def query_ids(self, text):
        """The ids of a query (embed)."""
        return self.plain("query: " + text)

    def document_ids(self, text=None, image=None):
        """The ids of a document (embed): image (h2, w2), its blocks."""
        if image is None:
            return self.plain("passage: " + (text or ""))
        ids = self.plain("passage: ") + self.image_ids(*image)
        return ids + self.plain(" " + text) if text is not None else ids

    def pair_ids(self, query, text=None, image=None):
        """The ids of a query and a document (rerank)."""
        s = "query: " + query + "\n\npassage: " + (text or "")
        if image is None:
            return self.plain(s)
        return self.image_ids(*image) + self.plain((" " + s).rstrip())

    # ---- images ---------------------------------------------------------------------

    @property
    def vision(self):
        """The vision tower and the projector, loaded at the first use."""
        if self._vision is None:
            if self.devices["vision"] == "off":
                raise ValueError("the vision tower is off (devices)")
            V = _Vision(*self._vision_args, self.eps, self)
            V.gpu = self.devices["vision"] == "gpu"
            self._to_device(V.lins(), V.gpu, "vision")
            self._vision = V
        return self._vision

    def image_rows(self, src, edge=None):
        """(rows (blocks, width), (h2, w2)) of an image (a path, the bytes of a
        file, or a PIL image) at a longest edge (default the processor's)."""
        edge = int(edge or self.edge)
        if not PATCH * MERGE <= edge <= MAX_EDGE:
            raise ValueError("the longest edge of an image must be from %d to %d" % (PATCH * MERGE, MAX_EDGE))
        pix = image_pixels(src, edge, self.image_mean, self.image_std)
        return self.pixel_rows(pix)

    def pixel_rows(self, pix):
        """(rows, (h2, w2)) of pixel values [3, H, W] (the processor's)."""
        h2, w2 = pix.shape[1] // (PATCH * MERGE), pix.shape[2] // (PATCH * MERGE)
        return self.vision.rows(pix), (h2, w2)

    # ---- the model ---------------------------------------------------------------

    def tables(self, pos):
        """The cos and sin (n, hd / 2) of the positions: the q tables times
        the attention scale and Llama 4's."""
        ang = pos.astype(np.float32)[:, None] * self.inv[None, :]
        cos, sin = _cos_sin(ang)
        att = np.float32(self.att)
        if att != 1:
            cos, sin = cos * att, sin * att
        l4 = (np.float32(1) + np.float32(self.l4_beta) * np.log1p(np.floor(pos / self.l4_orig)).astype(np.float32))
        s = (l4 * np.float32(self.hd ** -0.5)).astype(np.float32)[:, None]
        return cos * s, sin * s, cos, sin

    def program(self, n, gpu=False):
        """The program of the layers for n rows: inputs "e" (the embeddings),
        "cq", "sq", "ck", "sk" (the RoPE tables), "lo", "hi" (the keys of each
        row); output "xn" (n, width), the rows of the final norm."""
        w, H, KV, hd = self.width, self.heads, self.kv, self.hd
        inter = _shape(self.blk[0]["gate"])[0]
        kmax = max(w, inter, H * hd)
        c = _EncCompiler(self.eps, pack=not gpu, rows=n, kmax=kmax)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(e=z(n, w), x=z(n, w), h=z(n, w), t=z(n, w), q=z(n, H * hd), k=z(n, KV * hd),
                     v=z(n, KV * hd), o=z(n, H * hd), g=z(n, inter), u=z(n, inter), xn=z(n, w),
                     cq=z(n, hd // 2), sq=z(n, hd // 2), ck=z(n, hd // 2), sk=z(n, hd // 2),
                     lo=np.zeros(n, np.int32), hi=np.zeros(n, np.int32),
                     _scratch=np.zeros(n * kmax, np.float32))
        if gpu:
            # the GPU's attention: each KV head copied to the query heads that read it (ENC_GATHER of
            # the rows of hd values), then ENC_ATTN_TILE (float32 tiles; ENC_ATTN_RANGE with shared
            # heads took 3.3x its time: 1B, 1320 tokens, 9 ms a layer)
            c.attn_f32 = True
            kx = np.arange(n, dtype=np.int32)[:, None] * KV + np.arange(H, dtype=np.int32)[None, :] // (H // KV)
            c.env.update(kx=np.ascontiguousarray(kx, dtype=np.int32), kh=z(n, H * hd), vh=z(n, H * hd))
            attn = [("set", "kh", ("enc_gather", "k", "kx", hd)),
                    ("set", "vh", ("enc_gather", "v", "kx", hd)),
                    ("set", "o", ("enc_attn_range_x2", "q", "kh", "vh", "lo", "hi", H, hd))]
        else:
            attn = [("set", "o", ("enc_attn_range", "q", "k", "v", "lo", "hi", H, KV, hd))]
        forms = [("set", "x", ("copy", "e"))]
        for b in self.blk:
            forms += [
                ("set", "h", ("enc_rms", "x", b["ln_in"], w)),
                ("set", "q", ("enc_linear", b["q"], "h")),
                ("set", "k", ("enc_linear", b["k"], "h")),
                ("set", "v", ("enc_linear", b["v"], "h")),
                ("enc_rope_tab", "q", "cq", "sq", H, hd, True),
                ("enc_rope_tab", "k", "ck", "sk", KV, hd, True),
                *attn,
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", b["ln_post"], w)),
                ("set", "g", ("enc_linear", b["gate"], "h")),
                ("set", "u", ("enc_linear", b["up"], "h")),
                ("set", "g", ("enc_silu_mul", "g", "u")),
                ("set", "t", ("enc_linear", b["down"], "g")),
                ("enc_add", "x", "t"),
            ]
        forms.append(("set", "xn", ("enc_rms", "x", self.norm, w)))
        c.compile(("seq", *forms))
        p = c.p.finish()
        if not gpu and getattr(self, "_pages", None):
            # (the CPU's matrices are in groups of 16 rows now: the pages of the map go)
            for a in self._pages:
                drop_pages(a)
            self._pages = None
        return p

    def _inputs(self, ids, spans, soft, n):
        """The inputs of the program of n rows (n at least the ids)."""
        m = len(ids)
        ids = np.asarray(ids, dtype=np.int64)
        e = np.zeros((n, self.width), np.float32)
        e[:m] = self.weights.rows(self.table, ids)
        where = np.flatnonzero(ids == self.image_id)
        if len(where) or soft is not None:
            soft = np.zeros((0, self.width), np.float32) if soft is None else soft
            if len(where) != len(soft):
                raise ValueError("%d image placeholders, %d image rows" % (len(where), len(soft)))
            e[where] = soft
        # the rows past the inputs: each over itself alone
        r = np.arange(n, dtype=np.int32)
        pos, lo, hi = np.zeros(n, np.int64), r.copy(), r + 1
        for a, b in spans:
            pos[a:b] = r[a:b] - a
            lo[a:b], hi[a:b] = a, b
        cq, sq, ck, sk = self.tables(pos)
        return dict(e=e, cq=cq, sq=sq, ck=ck, sk=sk, lo=lo, hi=hi)

    def forward_numpy(self, d, m):
        """The NumPy layers (the reference of the program): the rows of the
        final norm for the first m rows of the inputs d."""
        H, KV, hd, eps = self.heads, self.kv, self.hd, self.eps
        lin = lambda x, w: np.matmul(x, _wt(w))  # noqa: E731
        rms = lambda x, w: ops.rms_norm(x, w, eps)  # noqa: E731
        x = d["e"][:m].copy()
        cq, sq, ck, sk = (d[k][:m] for k in ("cq", "sq", "ck", "sk"))
        for b in self.blk:
            h = rms(x, b["ln_in"])
            q = _rope_np(lin(h, b["q"]).reshape(m, H, hd), cq, sq)
            k = _rope_np(lin(h, b["k"]).reshape(m, KV, hd), ck, sk)
            v = lin(h, b["v"]).reshape(m, KV, hd)
            x = x + lin(_attention_np(q, k, v, d["lo"][:m], d["hi"][:m]).reshape(m, -1), b["o"])
            h = rms(x, b["ln_post"])
            g = lin(h, b["gate"])
            x = x + lin(g / (1 + np.exp(-g)) * lin(h, b["up"]), b["down"])
        return rms(x, self.norm)

    def forward(self, ids, spans, soft=None):
        """The rows of the final norm (sum of tokens, width) for the packed ids
        of the inputs (spans: their (start, end)). soft: the rows of the [IMG]
        placeholders of the ids, in their order."""
        m = len(ids)
        if _ENC_PY:
            return self.forward_numpy(self._inputs(ids, spans, soft, m), m)
        n = _bucket(m)
        return self.runner.run(n, self.gpu, self._inputs(ids, spans, soft, n), "xn")[:m]

    def pooled(self, seqs, soft=None, tokens=False):
        """The mean of the last hidden state of each sequence (len(seqs),
        width); soft: for each sequence the rows of its image, or None.
        tokens: also the token rows (a check against the reference)."""
        spans, a = [], 0
        for s in seqs:
            spans.append((a, a + len(s)))
            a += len(s)
        rows = None
        if soft is not None and any(r is not None for r in soft):
            rows = np.concatenate([r for r in soft if r is not None])
        xn = self.forward([t for s in seqs for t in s], spans, rows)
        means = np.stack([xn[a:b].mean(0) for a, b in spans])
        return (means, xn) if tokens else means

    def embed_ids(self, seqs, dim=None, soft=None):
        """The vectors (len(seqs), dim) of lists of ids, L2-normalized (the
        first dim values of the mean, Matryoshka)."""
        dim = dim or self.dim
        if not 1 <= dim <= self.dim:
            raise ValueError("dimensions must be from 1 to %d" % self.dim)
        y = self.pooled(seqs, soft)[:, :dim]
        return y / np.linalg.norm(y, axis=1, keepdims=True)

    def score_ids(self, seqs, soft=None):
        """The rerank logits (len(seqs),) of lists of ids."""
        if self.score_w is None:
            raise ValueError("%s is not a reranker" % self.path)
        return self.pooled(seqs, soft) @ self.score_w

    def embed(self, texts, kind="query", dim=None):
        """The vectors of texts, as queries or documents."""
        f = self.query_ids if kind == "query" else self.document_ids
        return self.embed_ids([f(t) for t in texts], dim)
