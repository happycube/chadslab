"""The image input of DeepSeek V4.1 Flash (DEEPSEEK_V41_PLAN.md, Phase 6): the
checkpoint's inference/image_processor.py and vision.py.

    the image (RGB), padded (ImageOps.pad, gray 127) to the largest size of
    patches of 14 x 14 whose tokens fit max_image_tokens (1024; at least
    min_pixels) -> values (x / 255 - 0.5) / 0.5 rounded to bfloat16 ->
    patches (channel, row, column) in raster order -> the patch linear (bias)
    -> 32 blocks (RMS norm; wqkv with bias; the 2D RoPE, theta 1e4: the pairs
    (j, j + 32) of each head, the first 16 take the row of the patch, the next
    16 its column; attention of all the patches, scale 1 / 8; wo with bias;
    RMS norm; w2(silu(gate) up), no biases) -> RMS norm -> the aligner: each 3
    x 3 block of patches (the grid padded with zeros) as one row of 9 x 1024
    (F.unfold: channel, row, column) -> w1 (bias), gelu (erf), w2 (bias): a
    row of the model's width for each block.

The model sees an image as the span

    <start> + (<image> x columns + <newline>) x rows + <end>

with image_token_id at every position; the delimiters take the learned rows
image_start, image_newline, image_end, the <image> positions the aligner's
rows in reading order (model.py merge_image_embeddings; the rows rounded to
bfloat16, the reference's h). In the span the router takes bias_vl, and
engram adds nothing (DSV41GPU._inputs).

The ViT runs as a program of the encoder records (reclank/encoders/runtime.py,
the ViT of Qwen: ENC_LINEAR on the tensor cores, ENC_ROPE2D, ENC_ATTN, and
ENC_SILU_MUL), on the GPU; the rows of q and k of each head in the order of
ENC_ROPE2D (quarters 0, 2, 1, 3; models/qwen/vision.py rope_order). The
matrices are BF12 (GP_KQ_LINEAR, as the bfloat16 matrices of the model; the
patch linear stays bfloat16: 588 cols), or bfloat16 with
RECLANK_DS_VISION_BF12=0. The aligner is a second program;
the host moves the rows of each 3 x 3 block together between the two (the
columns of the aligner's w1 are in that order: row, column, channel).
"""
from __future__ import annotations

import io
import math
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Runner, to_bf12
from reclank.models.qwen.vision import _Lin, rope_order

# the token types of a span (image_processor.py)
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)


def num_image_tokens(n_llm_h, n_llm_w):
    return n_llm_h * (n_llm_w + 1) + 2


def llm_grid(best_height, best_width, patch, ratio):
    return math.ceil((best_height // patch) / ratio), math.ceil((best_width // patch) / ratio)


def solve_resize_ratio(height, width, patch, ratio, max_n_token):
    """The largest pixel size of the aspect whose tokens fit max_n_token."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch * ratio
    if max_w_float < 1.0:
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return math.floor(height * beta / patch) * patch, math.floor(width * beta / patch) * patch


def safe_resize(height, width, best_height, best_width, patch, ratio, max_n_token):
    n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch, ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = solve_resize_ratio(height, width, patch, ratio, max_n_token)
        n_llm_h, n_llm_w = llm_grid(best_height, best_width, patch, ratio)
        assert num_image_tokens(n_llm_h, n_llm_w) <= max_n_token
    return n_llm_h, n_llm_w, best_height, best_width


def bf16_round(x):
    """x (float32) rounded to bfloat16 (to nearest even), as float32."""
    u = np.ascontiguousarray(x, np.float32).view(np.uint32)
    u = (u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return u.view(np.float32)


class DSVision:
    """The ViT and the aligner of the checkpoint (src: DSV41Source; cfg: its
    config.json) on the GPU: image(data) gives the rows of the span."""

    def __init__(self, src, cfg, gpu=True):
        vc = cfg["vision_config"]
        self.patch = int(vc["patch_size"])
        self.ratio = int(vc["downsample_ratio"])
        self.max_tokens = int(vc["max_image_tokens"])
        self.min_pixels = int(vc["min_pixels"])
        self.max_wh = vc.get("max_wh_ratio")
        self.width = w = int(vc["hidden_size"])
        self.heads = int(vc["num_attention_heads"])
        self.hd = hd = w // self.heads
        self.layers = int(vc["num_hidden_layers"])
        self.inter = int(vc["intermediate_size"])
        self.dim = int(cfg["text_config"]["hidden_size"])
        self.gpu = gpu
        self.image_token_id = int(cfg["image_token_id"])
        f32 = src.f32
        bf = lambda n: np.ascontiguousarray(src.bf16(n))  # noqa: E731
        p = "vision."
        self.patch_lin = _Lin(bf(p + "patch_embed.proj.weight"), f32(p + "patch_embed.proj.bias"))
        order = np.concatenate([h * hd + rope_order(hd) for h in range(self.heads)])
        self.blk = []
        for i in range(self.layers):
            b = "%sblocks.%d." % (p, i)
            qkv, qb = bf(b + "attn.wqkv.weight"), f32(b + "attn.wqkv.bias").reshape(-1)
            w1 = bf(b + "mlp.w1.weight")
            self.blk.append(dict(
                n1=f32(b + "norm1.weight").reshape(-1),
                q=_Lin(np.ascontiguousarray(qkv[:w][order]), qb[:w][order]),
                k=_Lin(np.ascontiguousarray(qkv[w:2 * w][order]), qb[w:2 * w][order]),
                v=_Lin(np.ascontiguousarray(qkv[2 * w:]), qb[2 * w:]),
                o=_Lin(bf(b + "attn.wo.weight"), f32(b + "attn.wo.bias")),
                n2=f32(b + "norm2.weight").reshape(-1),
                gate=_Lin(np.ascontiguousarray(w1[:self.inter]), None),
                up=_Lin(np.ascontiguousarray(w1[self.inter:]), None),
                down=_Lin(bf(b + "mlp.w2.weight"), None)))
        self.norm = f32(p + "norm.weight").reshape(-1)
        # the aligner's w1 with its columns in the order (row, column, channel)
        # of a 3 x 3 block (F.unfold gives channel, row, column)
        r2 = self.ratio * self.ratio
        perm = (np.arange(w)[None, :] * r2 + np.arange(r2)[:, None]).reshape(-1)
        self.a1 = _Lin(np.ascontiguousarray(bf("aligner.w1.weight")[:, perm]), f32("aligner.w1.bias"))
        self.a2 = _Lin(bf("aligner.w2.weight"), f32("aligner.w2.bias"))
        self.a1_cols = self.a1.w.shape[1]
        self.start, self.end, self.newline = (bf16_round(f32(n).reshape(-1)) for n in
                                              ("image_start", "image_end", "image_newline"))
        half = hd // 2
        self.inv = (1.0 / (float(vc["rope_theta"]) ** (np.arange(0, half, 2, dtype=np.float32) / half))
                    ).astype(np.float32)
        self.scale = np.full(w, 1.0 / math.sqrt(hd), np.float32)
        if gpu and os.environ.get("RECLANK_DS_VISION_BF12", "1") != "0":
            for lin in [self.patch_lin, self.a1, self.a2] + [b[k] for b in self.blk for k in
                                                            ("q", "k", "v", "o", "gate", "up", "down")]:
                to_bf12(lin)
        self.vit = _Runner(self._vit_program, keep_gpu=1)
        self.align = _Runner(self._align_program, keep_gpu=1)

    # ---- the pixels (image_processor.py)

    def plan(self, width, height, max_tokens):
        """(n_llm_h, n_llm_w, best_height, best_width) of an image (plan_image_grid)."""
        p = self.patch
        if self.max_wh is not None and width > height * self.max_wh:
            width = height * self.max_wh
        if 0 < width * height < self.min_pixels:
            ratio = (self.min_pixels / (width * height)) ** 0.5
            width = int(width * ratio)
            height = int(height * ratio)
        best_width = math.ceil(width / p) * p
        best_height = math.ceil(height / p) * p
        return safe_resize(height, width, best_height, best_width, p, self.ratio, max_tokens)

    def patches(self, data, max_tokens=None):
        """The patches of an image (bytes, or a PIL image): (patches [n, 588],
        (n_vit_h, n_vit_w), (n_llm_h, n_llm_w)) (load_image)."""
        from PIL import Image, ImageOps
        if isinstance(data, (bytes, bytearray)):
            with Image.open(io.BytesIO(data)) as source:
                image = source.convert("RGB")
        else:
            image = data.convert("RGB")
        p = self.patch
        n_llm_h, n_llm_w, bh, bw = self.plan(image.width, image.height,
                                             min(max_tokens or self.max_tokens, self.max_tokens))
        nh, nw = bh // p, bw // p
        if self.max_wh is not None and image.width >= self.max_wh * image.height:
            image = image.resize((bw, bh))
        else:
            image = ImageOps.pad(image, (bw, bh), color=(127, 127, 127))
        x = np.asarray(image, dtype=np.float32) / np.float32(255)
        x = bf16_round((x - np.float32(0.5)) / np.float32(0.5))
        x = x.transpose(2, 0, 1).reshape(3, nh, p, nw, p).transpose(1, 3, 0, 2, 4).reshape(nh * nw, 3 * p * p)
        return np.ascontiguousarray(x), (nh, nw), (n_llm_h, n_llm_w)

    # ---- the programs

    def _vit_program(self, n, gpu=False):
        """The ViT of n patches: inputs "p" [n, 588], "pos" [n, 2] (row,
        column); output "h" [n, 1024]."""
        w, H, hd, inter = self.width, self.heads, self.hd, self.inter
        kp = self.patch_lin.w.shape[1]
        c = _EncCompiler(1e-6, pack=not gpu, rows=n, kmax=max(kp, w, inter))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(p=z(n, kp), pos=np.zeros((n, 2), np.int32), x=z(n, w), h=z(n, w), q=z(n, w),
                     k=z(n, w), v=z(n, w), o=z(n, w), t=z(n, w), g=z(n, inter), u=z(n, inter),
                     _scratch=np.zeros(n * max(kp, w, inter), np.float32))
        forms = [("set", "x", ("enc_linear", self.patch_lin, "p"))]
        for b in self.blk:
            forms += [
                ("set", "h", ("enc_rms", "x", b["n1"], w)),
                ("set", "q", ("enc_linear", b["q"], "h")),
                ("set", "k", ("enc_linear", b["k"], "h")),
                ("set", "v", ("enc_linear", b["v"], "h")),
                ("enc_rope2d", "q", "pos", self.inv, H, hd),
                ("enc_rope2d", "k", "pos", self.inv, H, hd),
                ("set", "q", ("enc_mul_vec", "q", self.scale)),
                ("set", "o", ("enc_attn", "q", "k", "v", H, hd)),
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", b["n2"], w)),
                ("set", "g", ("enc_linear", b["gate"], "h")),
                ("set", "u", ("enc_linear", b["up"], "h")),
                ("set", "g", ("enc_silu_mul", "g", "u")),
                ("set", "t", ("enc_linear", b["down"], "g")),
                ("enc_add", "x", "t"),
            ]
        forms.append(("set", "h", ("enc_rms", "x", self.norm, w)))
        c.compile(("seq", *forms))
        return c.p.finish()

    def _align_program(self, m, gpu=False):
        """The aligner of m blocks: input "b" [m, 9 x 1024] (each block's rows
        in the order row, column); output "y" [m, dim]."""
        kb = self.a1_cols
        c = _EncCompiler(1e-6, pack=not gpu, rows=m, kmax=max(kb, self.dim))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(b=z(m, kb), a=z(m, self.dim), y=z(m, self.dim))
        c.compile(("seq", ("set", "a", ("enc_linear", self.a1, "b")), ("set", "a", ("enc_gelu", "a", True)),
                   ("set", "y", ("enc_linear", self.a2, "a"))))
        return c.p.finish()

    # ---- an image

    def encode(self, patches, grid):
        """The aligner's rows [n_llm_h n_llm_w, dim] of the patches of a grid
        (n_vit_h, n_vit_w) of patches."""
        nh, nw = grid
        n = nh * nw
        pos = np.stack([np.arange(n) // nw, np.arange(n) % nw], 1).astype(np.int32)
        h = self.vit.run(n, self.gpu, {"p": patches, "pos": pos}, "h")
        r = self.ratio
        H3, W3 = -(-nh // r), -(-nw // r)
        y = np.zeros((H3 * r, W3 * r, self.width), np.float32)
        y[:nh, :nw] = h.reshape(nh, nw, self.width)
        b = y.reshape(H3, r, W3, r, self.width).transpose(0, 2, 1, 3, 4).reshape(H3 * W3, -1)
        return self.align.run(H3 * W3, self.gpu, {"b": np.ascontiguousarray(b)}, "y")

    def span_rows(self, rows, grid):
        """The rows of the span of an image [num_image_tokens, dim] from its
        aligner rows and its grid of tokens (n_llm_h, n_llm_w), rounded to
        bfloat16 (the reference's h)."""
        gh, gw = grid
        out = np.empty((num_image_tokens(gh, gw), self.dim), np.float32)
        out[0] = self.start
        body = out[1:-1].reshape(gh, gw + 1, self.dim)
        body[:, :gw] = bf16_round(rows).reshape(gh, gw, self.dim)
        body[:, gw] = self.newline
        out[-1] = self.end
        return out

    def image(self, data, max_tokens=None):
        """The rows of the span of an image (bytes): [num_image_tokens, dim],
        and its grid of tokens (n_llm_h, n_llm_w)."""
        patches, grid, lgrid = self.patches(data, max_tokens)
        rows = self.encode(patches, grid)
        assert rows.shape[0] == lgrid[0] * lgrid[1], (rows.shape, lgrid)
        return self.span_rows(rows, lgrid), lgrid

    def release(self):
        """Free the GPU memory of the programs and the weights (the next image
        uploads them again)."""
        self.vit.release()
        self.align.release()
