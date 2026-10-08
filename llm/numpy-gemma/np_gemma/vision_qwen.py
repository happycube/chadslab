"""The image encoder of Qwen3.6 and Qwen3.8 (the Qwen3-VL ViT, projector
qwen3vl_merger).

MULTIMODAL_PLAN.md, section 3.4. The weights come from the mmproj GGUF of
llama.cpp (Qwen3.6), or from the tensors model.visual.* of the safetensors
checkpoint (Qwen3.8-Flash-Next: the same ViT, with an output of 2560; the
vision tensors are BF16 in the NVFP4 checkpoint too). The graph follows
transformers (models/qwen3_5_moe and models/qwen4_exp:
Qwen3_5MoeVisionModel, and Qwen2VLImageProcessor for the pixels):

    the image, resized (bicubic) to sides that are multiples of 32 ->
    patches of 16 x 16 in the order of the 2 x 2 merge blocks, values
    2 (x / 255) - 1 -> the patch linear (the Conv3d of two equal frames is
    the sum of its two kernels) + bias -> + the position table (48 x 48,
    bilinear with align corners to the grid of patches) -> 27 layers
    (LayerNorm; q, k, v with bias; the 2D RoPE, theta 1e4: the pairs of the
    first quarter of a head take the row, the second quarter the column;
    attention of all the patches, scale 1 / sqrt(72); the output linear;
    LayerNorm; the FFN with gelu tanh and biases) -> LayerNorm -> each 2 x 2
    block as one row of 4 x 1152 -> mm.0, gelu (erf), mm.2 -> 2048 values.

The model has no deepstack layers (deepstack_visual_indexes is empty).

The program reuses the records of gemma4v (ENC_LINEAR, ENC_ROPE2D, ENC_ATTN,
...) and adds ENC_LNORM and ENC_GELU. ENC_ROPE2D rotates the pairs (j, j +
hd / 4) of each half of a head (gemma4v). The RoPE of Qwen rotates the
pairs (j, j + hd / 2) of the whole head. The rows of q and k of each head
are put in the order of ENC_ROPE2D (quarters 0, 2, 1, 3); the scores do not
change, because q and k take the same order.
"""
from __future__ import annotations

import math

import numpy as np

from .gemma4_encoders import (_ENC_PY, _EncCompiler, _Runner, _Weights, _gelu_tanh, _softmax,
                              quantize_q8)

PATCH = 16
MERGE = 2
FACTOR = PATCH * MERGE          # the sides of the image are multiples of 32
MIN_TOKENS = 64                 # shortest_edge of preprocessor_config.json: 65536 pixels


class _STWeights:
    """The tensors model.visual.* of a safetensors checkpoint (a directory
    with config.json and model.safetensors.index.json) with the names and
    the metadata of an mmproj GGUF, for QwenVision."""

    NAMES = {"ln1": "norm1", "ln2": "norm2", "attn_qkv": "attn.qkv", "attn_out": "attn.proj",
             "ffn_up": "mlp.linear_fc1", "ffn_down": "mlp.linear_fc2"}

    def __init__(self, path):
        import json
        import os
        from .st import SafeTensors
        cfg = json.load(open(os.path.join(path, "config.json")))
        vc = cfg["vision_config"]
        wm = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.files = {}
        self.where = {}
        for k, f in wm.items():
            if k.startswith("model.visual."):
                if f not in self.files:
                    self.files[f] = SafeTensors(os.path.join(path, f))
                self.where[k[len("model.visual."):]] = (self.files[f], k)
        self.meta = {
            "clip.projector_type": "qwen3vl_merger",
            "clip.vision.embedding_length": vc["hidden_size"],
            "clip.vision.block_count": vc["depth"],
            "clip.vision.attention.head_count": vc["num_heads"],
            "clip.vision.attention.layer_norm_epsilon": 1e-6,
            "clip.vision.spatial_merge_size": vc["spatial_merge_size"],
            "clip.vision.patch_size": vc["patch_size"],
            "clip.vision.is_deepstack_layers": [1 if i in vc.get("deepstack_visual_indexes", [])
                                                else 0 for i in range(vc["depth"])],
        }

    def _hf(self, name):
        """The HF name of a GGUF name, and the frame of the patch kernel."""
        if name.startswith("v.patch_embd.weight"):
            return "patch_embed.proj.weight", 1 if name.endswith(".1") else 0
        fixed = {"v.patch_embd.bias": "patch_embed.proj.bias",
                 "v.position_embd.weight": "pos_embed.weight",
                 "v.post_ln.weight": "merger.norm.weight", "v.post_ln.bias": "merger.norm.bias",
                 "mm.0.weight": "merger.linear_fc1.weight", "mm.0.bias": "merger.linear_fc1.bias",
                 "mm.2.weight": "merger.linear_fc2.weight", "mm.2.bias": "merger.linear_fc2.bias"}
        if name in fixed:
            return fixed[name], None
        _, blk, i, part, kind = name.split(".")
        return "blocks.%s.%s.%s" % (i, self.NAMES[part], kind), None

    def has(self, name):
        return self._hf(name)[0] in self.where

    def tensor(self, hf):
        """The float32 tensor of an HF name (without model.visual.)."""
        f, full = self.where[hf]
        return f.get(full, np.float32)

    def f32(self, name):
        hf, frame = self._hf(name)
        a = self.tensor(hf)
        if frame is not None:
            a = a[:, :, frame]
        return np.ascontiguousarray(a, np.float32)

    def matrix(self, name):
        f, full = self.where[self._hf(name)[0]]
        return np.ascontiguousarray(f.get_bf16(full))


def vision_weights(path):
    """The weights of the encoder: an mmproj GGUF, or a checkpoint directory."""
    import os
    return _STWeights(path) if os.path.isdir(path) else _Weights(path)


class _Lin:
    """A linear of the mmproj with a bias (W bfloat16 or float32)."""

    def __init__(self, w, b):
        self.w = w
        self.b = None if b is None else np.ascontiguousarray(b, np.float32).reshape(-1)
        self.imin = self.imax = self.omin = self.omax = None

    def __call__(self, x):
        from . import ops
        x = np.ascontiguousarray(x, dtype=np.float32)
        y = ops.linear_bf16(x, self.w) if self.w.dtype == np.uint16 else x @ self.w.T
        y = y.astype(np.float32, copy=False)
        return y + self.b if self.b is not None else y


def _layer_norm(x, w, b, eps):
    x = x.astype(np.float32, copy=False)
    m = x.mean(axis=-1, keepdims=True)
    d = x - m
    return d / np.sqrt((d * d).mean(axis=-1, keepdims=True) + eps) * w + b


def _gelu_erf(x):
    from math import erf
    return (0.5 * x * (1.0 + np.vectorize(erf, otypes=[np.float64])(x / math.sqrt(2.0)))
            ).astype(np.float32)


def rope_order(hd):
    """The order of the values of a head for ENC_ROPE2D: quarters 0, 2, 1,
    3. Row g of the new q (or k) is row order[g] of the old one."""
    q = hd // 4
    idx = np.arange(hd)
    return np.concatenate([idx[:q], idx[2 * q:3 * q], idx[q:2 * q], idx[3 * q:]])


class QwenVision:
    """The qwen3vl_merger encoder of an mmproj GGUF."""

    def __init__(self, W):
        m = W.meta
        kind = m.get("clip.projector_type", m.get("clip.vision.projector_type"))
        if kind != "qwen3vl_merger":
            raise ValueError("not a qwen3vl_merger mmproj (projector %r)" % kind)
        ds = m.get("clip.vision.is_deepstack_layers")
        if ds is not None and any(int(v) for v in ds):
            raise ValueError("the mmproj has deepstack layers; this encoder has no deepstack input")
        self.width = int(m["clip.vision.embedding_length"])
        self.layers = int(m["clip.vision.block_count"])
        self.heads = int(m["clip.vision.attention.head_count"])
        self.hd = self.width // self.heads
        self.eps = float(m.get("clip.vision.attention.layer_norm_epsilon", 1e-6))
        self.merge = int(m.get("clip.vision.spatial_merge_size", MERGE))
        assert int(m["clip.vision.patch_size"]) == PATCH and self.merge == MERGE
        w = self.width
        # The patch linear: the two frames of the Conv3d are the same image,
        # so the kernel is the sum of the two (width, channel, row, column).
        k0 = W.f32("v.patch_embd.weight").reshape(w, 3 * PATCH * PATCH)
        k1 = W.f32("v.patch_embd.weight.1").reshape(w, 3 * PATCH * PATCH)
        self.patch_lin = _Lin(np.ascontiguousarray(k0 + k1), W.f32("v.patch_embd.bias"))
        # The patch linear of video: two frames, the values (channel, frame,
        # row, column) as the Conv3d of transformers.
        k01 = np.stack([k0.reshape(w, 3, -1), k1.reshape(w, 3, -1)], axis=2).reshape(w, -1)
        self.patch_lin2 = _Lin(np.ascontiguousarray(k01), W.f32("v.patch_embd.bias"))
        self.pos = W.f32("v.position_embd.weight").reshape(-1, w)
        self.side = int(round(math.sqrt(self.pos.shape[0])))
        assert self.side * self.side == self.pos.shape[0]
        order = np.concatenate([h * self.hd + rope_order(self.hd) for h in range(self.heads)])
        self.blk = []
        for i in range(self.layers):
            p = "v.blk.%d." % i
            qkv = W.matrix(p + "attn_qkv.weight")
            qb = W.f32(p + "attn_qkv.bias").reshape(-1)
            self.blk.append(dict(
                ln1=(W.f32(p + "ln1.weight").reshape(-1), W.f32(p + "ln1.bias").reshape(-1)),
                q=_Lin(np.ascontiguousarray(qkv[:w][order]), qb[:w][order]),
                k=_Lin(np.ascontiguousarray(qkv[w:2 * w][order]), qb[w:2 * w][order]),
                v=_Lin(np.ascontiguousarray(qkv[2 * w:]), qb[2 * w:]),
                o=_Lin(W.matrix(p + "attn_out.weight"), W.f32(p + "attn_out.bias")),
                ln2=(W.f32(p + "ln2.weight").reshape(-1), W.f32(p + "ln2.bias").reshape(-1)),
                up=_Lin(W.matrix(p + "ffn_up.weight"), W.f32(p + "ffn_up.bias")),
                down=_Lin(W.matrix(p + "ffn_down.weight"), W.f32(p + "ffn_down.bias"))))
        self.post_ln = (W.f32("v.post_ln.weight").reshape(-1), W.f32("v.post_ln.bias").reshape(-1))
        self.mm0 = _Lin(W.matrix("mm.0.weight"), W.f32("mm.0.bias"))
        self.mm2 = _Lin(W.matrix("mm.2.weight"), W.f32("mm.2.bias"))
        self.hidden = self.mm2.w.shape[0]
        # The rope of each half: hd / 4 frequencies (Qwen3_5MoeVisionRotaryEmbedding,
        # theta 1e4; the frequencies of hd / 2 values).
        half = self.hd // 2
        self.inv = (1.0 / (10000.0 ** (np.arange(0, half, 2, dtype=np.float32) / half))).astype(np.float32)
        self.scale = np.full(self.width, 1.0 / math.sqrt(self.hd), np.float32)
        self.runner = _Runner(self.program)
        self.gpu = False
        self.q8 = False

    # ---- the inputs ----

    def pos_rows(self, gh, gw):
        """The rows of the position table for a grid of gh x gw patches, in
        the order of the merge blocks: the bilinear interpolation with
        align corners (get_vision_interpolation_indices_and_weights)."""
        s = self.side

        def taps(n):
            src = np.arange(n, dtype=np.float32) * np.float32(s - 1) / np.float32(max(n - 1, 1))
            f = np.floor(src)
            lo = np.clip(f.astype(np.int64), 0, s - 1)
            hi = np.clip(f.astype(np.int64) + 1, 0, s - 1)
            d = src - f
            return lo, hi, (1.0 - d).astype(np.float32), d.astype(np.float32)

        r, c = merge_order(gh, gw)
        rl, rh, rwl, rwh = (a[r] for a in taps(gh))
        cl, ch, cwl, cwh = (a[c] for a in taps(gw))
        t = self.pos
        out = (t[rl * s + cl] * (rwl * cwl)[:, None] + t[rl * s + ch] * (rwl * cwh)[:, None]
               + t[rh * s + cl] * (rwh * cwl)[:, None] + t[rh * s + ch] * (rwh * cwh)[:, None])
        return np.ascontiguousarray(out, np.float32)

    # ---- the encoder ----

    def encode(self, patches, grid):
        """patches (n, 768): values in [-1, 1] in the order (channel, row,
        column), the patches in the order of the merge blocks; grid (gh,
        gw) of patches. Or patches (n, 1536) of two frames of a video
        (channel, frame, row, column). Return the rows (n / 4, hidden)."""
        gh, gw = grid
        n = gh * gw
        video = patches.shape[1] == 2 * 3 * PATCH * PATCH
        assert patches.shape[0] == n and (video or patches.shape[1] == 3 * PATCH * PATCH)
        pe = self.pos_rows(gh, gw)
        r, c = merge_order(gh, gw)
        pos = np.stack([r, c], axis=1).astype(np.int32)
        if _ENC_PY:
            return self._layers(patches, pe, pos)
        return self.runner.run((n, video) if video else n, self.gpu,
                               {"p": np.asarray(patches, np.float32), "pe": pe, "pos": pos}, "y")

    def encode_video(self, patches, grid):
        """patches (t n, 1536) of a video (video_input), grid (t, gh, gw).
        Each pair of frames is one group: its patches see only each other
        (the cu_seqlens of transformers). Return the rows of each group, a
        list of t arrays (n / 4, hidden)."""
        t, gh, gw = grid
        n = gh * gw
        return [self.encode(patches[i * n:(i + 1) * n], (gh, gw)) for i in range(t)]

    def program(self, n, gpu=False):
        """The program of the encoder for n patches: inputs "p", "pe", "pos"
        (row, column); output "y" (n / 4, hidden). n = (count, True): the
        patches of two frames of a video."""
        video = isinstance(n, tuple)
        if video:
            n = n[0]
        patch_lin = self.patch_lin2 if video else self.patch_lin
        w, H, hd = self.width, self.heads, self.hd
        inter = self.blk[0]["up"].w.shape[0]
        kp = patch_lin.w.shape[1]
        m4 = w * MERGE * MERGE
        n4 = n // (MERGE * MERGE)
        c = _EncCompiler(self.eps, pack=not gpu, q8=self.q8, rows=n, kmax=max(kp, w, inter, m4))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(p=z(n, kp), pe=z(n, w), pos=np.zeros((n, 2), np.int32),
                     x=z(n, w), h=z(n, w), q=z(n, w), k=z(n, w), v=z(n, w), o=z(n, w),
                     t=z(n, w), u=z(n, inter), hm=z(n4, m4), g=z(n4, m4), y=z(n4, self.hidden),
                     _scratch=np.zeros(n * max(kp, w, inter), np.float32))
        forms = [("set", "x", ("enc_linear", patch_lin, "p")), ("enc_add", "x", "pe")]
        for b in self.blk:
            forms += [
                ("set", "h", ("enc_lnorm", "x", b["ln1"][0], b["ln1"][1], w)),
                ("set", "q", ("enc_linear", b["q"], "h")),
                ("set", "k", ("enc_linear", b["k"], "h")),
                ("set", "v", ("enc_linear", b["v"], "h")),
                ("enc_rope2d", "q", "pos", self.inv, H, hd),
                ("enc_rope2d", "k", "pos", self.inv, H, hd),
                ("set", "q", ("enc_mul_vec", "q", self.scale)),
                ("set", "o", ("enc_attn", "q", "k", "v", H, hd)),
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", b["ln2"][0], b["ln2"][1], w)),
                ("set", "u", ("enc_linear", b["up"], "h")),
                ("set", "u", ("enc_gelu", "u")),
                ("set", "t", ("enc_linear", b["down"], "u")),
                ("enc_add", "x", "t"),
            ]
        forms += [
            # The rows of one merge block are next to each other: the norm of
            # the n rows is the n / 4 rows of 4 x width.
            ("set", "hm", ("enc_lnorm", "x", self.post_ln[0], self.post_ln[1], w)),
            ("set", "g", ("enc_linear", self.mm0, "hm")),
            ("set", "g", ("enc_gelu", "g", True)),
            ("set", "y", ("enc_linear", self.mm2, "g")),
        ]
        c.compile(("seq", *forms))
        return c.p.finish()

    def _layers(self, patches, pe, pos):
        """The encoder in NumPy (the reference of the program)."""
        n = patches.shape[0]
        H, hd = self.heads, self.hd
        lin = self.patch_lin2 if patches.shape[1] == self.patch_lin2.w.shape[1] else self.patch_lin
        x = lin(patches) + pe
        q4 = hd // 4
        ang = [pos[:, i:i + 1].astype(np.float32) * self.inv[None, :] for i in range(2)]
        cos = [np.cos(a)[:, None, :] for a in ang]
        sin = [np.sin(a)[:, None, :] for a in ang]

        def rope(x):
            # The order of ENC_ROPE2D: each half of the head, pairs (j, j + hd / 4).
            x = x.reshape(n, H, hd).copy()
            for part in range(2):
                p = x[..., part * 2 * q4:(part + 1) * 2 * q4]
                a0, b0 = p[..., :q4].copy(), p[..., q4:].copy()
                p[..., :q4] = a0 * cos[part] - b0 * sin[part]
                p[..., q4:] = b0 * cos[part] + a0 * sin[part]
            return x

        for b in self.blk:
            h = _layer_norm(x, *b["ln1"], self.eps)
            q = rope(b["q"](h)) * self.scale.reshape(H, hd)
            k = rope(b["k"](h))
            v = b["v"](h).reshape(n, H, hd)
            s = np.einsum("qhd,khd->hqk", q, k)
            a = np.einsum("hqk,khd->qhd", _softmax(s), v).reshape(n, H * hd)
            x = x + b["o"](a)
            h = _layer_norm(x, *b["ln2"], self.eps)
            x = x + b["down"](_gelu_tanh(b["up"](h)))
        h = _layer_norm(x, *self.post_ln, self.eps).reshape(n // 4, -1)
        return self.mm2(_gelu_erf(self.mm0(h)))


def merge_order(gh, gw):
    """The (row, column) of each patch in the order of the merge blocks:
    the blocks of 2 x 2 in raster order, and the 4 patches of a block in
    raster order (get_vision_position_ids)."""
    br, bc, ir, ic = np.meshgrid(np.arange(gh // MERGE), np.arange(gw // MERGE),
                                 np.arange(MERGE), np.arange(MERGE), indexing="ij")
    return (br * MERGE + ir).reshape(-1), (bc * MERGE + ic).reshape(-1)


def smart_resize(h, w, factor=FACTOR, min_pixels=MIN_TOKENS * FACTOR * FACTOR,
                 max_pixels=16384 * FACTOR * FACTOR):
    """The size of the resized image (Qwen2VLImageProcessor smart_resize)."""
    if max(h, w) / min(h, w) > 200:
        raise ValueError("the aspect ratio of the image is more than 200")
    hb = round(h / factor) * factor
    wb = round(w / factor) * factor
    if hb * wb > max_pixels:
        beta = math.sqrt(h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif hb * wb < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def image_input(src, budget=1024):
    """The patches of an image for QwenVision.encode: (patches, (gh, gw)).
    budget is the most tokens (merge blocks) of the image."""
    from PIL import Image
    from .unified import load_image
    img = load_image(src)
    th, tw = smart_resize(img.height, img.width, max_pixels=max(budget, MIN_TOKENS) * FACTOR * FACTOR)
    if (img.height, img.width) != (th, tw):
        img = img.resize((tw, th), Image.BICUBIC)
    x = np.asarray(img, dtype=np.float32) * np.float32(2.0 / 255.0) - np.float32(1.0)
    return patches_of(x)


def patches_of(x):
    """x (h, w, 3) normalized pixels -> the patches (n, 768) in the order
    of the merge blocks, values (channel, row, column), and the grid."""
    h, w, _ = x.shape
    gh, gw = h // PATCH, w // PATCH
    p = x[:gh * PATCH, :gw * PATCH].reshape(gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH, 3)
    # (block row, block col, in row, in col, channel, y, x)
    p = p.transpose(0, 3, 1, 4, 6, 2, 5).reshape(gh * gw, 3 * PATCH * PATCH)
    return np.ascontiguousarray(p), (gh, gw)


# ---- video (Qwen3VLVideoProcessor) ----

VIDEO_FPS = 2.0                 # the frames of a second (the fps of the processor)
MIN_FRAMES, MAX_FRAMES = 4, 768
VIDEO_SHORTEST = 4096           # size.shortest_edge of video_preprocessor_config.json
VIDEO_LONGEST = 25165824        # size.longest_edge


def video_frames(src, fps=VIDEO_FPS, max_frames=MAX_FRAMES):
    """Decode the sampled frames of a video (a path or bytes) with PyAV, as
    sample_frames of Qwen3VLVideoProcessor: fps frames for each second
    (at least 4, at most max_frames), np.linspace over the frames, rounded.
    Only the sampled frames are kept. Return (frames: RGB uint8 arrays,
    their indices, the fps of the video)."""
    import io
    import av
    f = io.BytesIO(src) if isinstance(src, (bytes, bytearray)) else src
    with av.open(f) as container:
        stream = container.streams.video[0]
        vfps = float(stream.average_rate) if stream.average_rate else 24.0
        total = int(stream.frames or 0)
        if total <= 0:
            # no frame count in the file: decode all, then pick
            allf = [fr.to_ndarray(format="rgb24") for fr in container.decode(video=0)]
            total = len(allf)
        else:
            allf = None
        if total == 0:
            raise ValueError("the video has no frames")
        nf = int(total / vfps * fps)
        nf = min(max(nf, MIN_FRAMES), int(max_frames), total)
        idx = np.linspace(0, total - 1, nf).round().astype(np.int64)
        if allf is None:
            want = set(int(i) for i in idx)
            got = {}
            for j, fr in enumerate(container.decode(video=0)):
                if j in want:
                    got[j] = fr.to_ndarray(format="rgb24")
                if j >= idx[-1]:
                    break
            if not got:
                raise ValueError("the video has no frames")
            last = max(got)
            idx = np.minimum(idx, last)     # a count in the header that is too large
            frames = [got.get(int(i), got[last]) for i in idx]
        else:
            frames = [allf[int(i)] for i in idx]
    return frames, idx, vfps


def video_size(nf, h, w, budget, factor=FACTOR, temporal=2):
    """The size of the frames (smart_resize of Qwen3VLVideoProcessor, with
    cap_pixels_per_frame and max_video_tokens = budget: the most tokens of
    a pair of frames)."""
    ppf = max(min(budget * factor * factor, VIDEO_LONGEST // nf), int(VIDEO_SHORTEST * 1.05))
    max_pixels = ppf * nf
    if h < factor or w < factor:
        sc = max(factor / h, factor / w)
        h, w = int(h * sc), int(w * sc)
    if max(h, w) / min(h, w) > 200:
        raise ValueError("the aspect ratio of the video is more than 200")
    hb = round(h / factor) * factor
    wb = round(w / factor) * factor
    tb = round(nf / temporal) * temporal
    if tb * hb * wb > max_pixels:
        beta = math.sqrt(nf * h * w / max_pixels)
        hb = max(factor, math.floor(h / beta / factor) * factor)
        wb = max(factor, math.floor(w / beta / factor) * factor)
    elif tb * hb * wb < VIDEO_SHORTEST:
        beta = math.sqrt(VIDEO_SHORTEST / (nf * h * w))
        hb = math.ceil(h * beta / factor) * factor
        wb = math.ceil(w * beta / factor) * factor
    return hb, wb


def video_input(src, budget=128, max_frames=MAX_FRAMES, fps=VIDEO_FPS):
    """The patches of a video for QwenVision.encode_video: (patches (t n,
    1536), (t, gh, gw), the time of each pair of frames in seconds)."""
    from PIL import Image
    frames, idx, vfps = video_frames(src, fps, max_frames)
    nf = len(frames)
    h, w = frames[0].shape[:2]
    th, tw = video_size(nf, h, w, budget)
    xs = []
    for fr in frames:
        img = Image.fromarray(fr)
        if (img.height, img.width) != (th, tw):
            img = img.resize((tw, th), Image.BICUBIC)
        xs.append(np.asarray(img, dtype=np.float32) * np.float32(2.0 / 255.0) - np.float32(1.0))
    idx = list(idx)
    if len(xs) % 2:                     # an even count: the last frame again
        xs.append(xs[-1])
        idx.append(idx[-1])
    x = np.stack(xs)
    gt, gh, gw = len(xs) // 2, th // PATCH, tw // PATCH
    # (pair, frame, block row, row, y, block col, col, x, channel) ->
    # (pair, block row, block col, row, col, channel, frame, y, x)
    p = x.reshape(gt, 2, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH, 3)
    p = p.transpose(0, 2, 5, 3, 6, 8, 1, 4, 7).reshape(gt * gh * gw, 2 * 3 * PATCH * PATCH)
    t = [i / vfps for i in idx]
    times = [(t[2 * i] + t[2 * i + 1]) / 2 for i in range(gt)]
    return np.ascontiguousarray(p), (gt, gh, gw), times


def video_text(groups):
    """The text of a video in the prompt (replace_video_token of
    Qwen3VLProcessor), with one <|video_pad|> for each pair of frames:
    <T seconds><|vision_start|><|video_pad|><|vision_end|> ... expand_qwen
    then gives each pad the rows of its pair."""
    return "".join("<%.1f seconds><|vision_start|><|video_pad|><|vision_end|>" % t
                   for t, _rows, _grid in groups)


class QwenEmbedder:
    """The media embedder of Qwen3.6: images. The interface of
    gemma4_encoders.Gemma4Embedder (image() gives the rows and the grid of
    tokens)."""

    kind = "qwen"
    bidir = False           # the tokens of an image are causal in Qwen

    def __init__(self, path, gpu=False, q8=False):
        W = vision_weights(path)
        self.path = path
        self.vision = QwenVision(W)
        self.vision.gpu = gpu
        self.has_audio = False
        self.gpu = gpu
        self.q8 = q8
        if q8:
            V = self.vision
            lins = [V.mm0, V.mm2] + [b[k] for b in V.blk for k in ("q", "k", "v", "o", "up", "down")]
            for lin in lins:
                quantize_q8(lin)
            V.q8 = True
        self.hidden = self.vision.hidden
        # keep_gpu False: free the GPU memory of the encoder after each image
        # or video (Qwen3.8 fills the GPU; the weights take 0.9 GB)
        self.keep_gpu = True
        # reserve_gpu: the most tokens of an image, the bytes of the encoder
        # at that size, the bytes it holds now, and the buffer of the rest
        self.budget_max = None
        self.lent = False       # lend_gpu: the model holds the room between media
        self.room_bytes = 0
        self.held = 0
        self.room = None

    def reserve_gpu(self, budget):
        """Hold the GPU memory of the largest image of budget tokens from now
        on, before a model takes the free memory: the encoder runs its
        weights and the program of that image now, and keeps them. A smaller
        image later has a smaller program; a buffer then keeps the rest of
        the room between the media requests, so the buffers of the model
        cannot take it (image and video free it first). image() takes at
        most budget tokens. Return the bytes held."""
        from PIL import Image
        from .gpumm import mem_info
        assert self.gpu
        self.keep_gpu = True
        self.vision.runner.keep_gpu = 1     # one size at a time (_Runner.get frees first)
        self.budget_max = budget
        f0 = mem_info()[0]
        self.image(Image.new("RGB", (2048, 2048), (128, 128, 128)), budget)
        self.room_bytes = self.held = f0 - mem_info()[0]
        return self.room_bytes

    def lend_gpu(self, budget):
        """reserve_gpu, then give the room back: the weights and the program
        of the encoder leave the GPU after each image or video (keep_gpu
        False), and between them the model holds that room (QwenGPU.lend_warm:
        warm slots of its experts; the caller frees it before an image).
        image() still takes at most budget tokens. Return the bytes."""
        n = self.reserve_gpu(budget)
        self.vision.runner.release()
        self.keep_gpu = False
        self.lent = True
        return n

    def _give_room(self):
        from .gpumm import mem_info
        if self.room is not None:
            self.room.free()
            self.room = None
        self._f0 = mem_info()[0]

    def _take_room(self):
        from .gpumm import Buffer, mem_info
        self.held += self._f0 - mem_info()[0]
        rest = self.room_bytes - self.held
        if rest > 0:
            self.room = Buffer(rest, "lend", "encoder")

    def warm(self):
        """Encode a small gray image: the weights go to the GPU now (before
        the model takes the free memory for its buffers)."""
        from PIL import Image
        self.image(Image.new("RGB", (256, 256), (128, 128, 128)), MIN_TOKENS)

    def _done(self):
        if self.gpu and not self.keep_gpu:
            self.vision.runner.release()
        if self.budget_max is not None and not self.lent:
            self._take_room()

    def image(self, src, budget=1024):
        """Return the rows (tokens, hidden) and the grid of tokens (rows,
        columns)."""
        if self.budget_max is not None:
            budget = min(budget, self.budget_max)
        patches, (gh, gw) = image_input(src, budget)
        if self.budget_max is not None and not self.lent:
            self._give_room()
        try:
            rows = self.vision.encode(patches, (gh, gw))
        finally:
            self._done()
        return rows, (gh // MERGE, gw // MERGE)

    def audio(self, src):
        raise ValueError("Qwen3.6 has no audio encoder")

    def video(self, src, budget=128, num_frames=32):
        """Return the pairs of frames of a video: a list of (the time in
        seconds, the rows (tokens, hidden), the grid of tokens). budget is
        the most tokens of a pair; num_frames the most frames (2 for each
        second up to it)."""
        patches, (t, gh, gw), times = video_input(src, budget, num_frames)
        if self.budget_max is not None and not self.lent:
            self._give_room()
        try:
            rows = self.vision.encode_video(patches, (t, gh, gw))
        finally:
            self._done()
        return [(tm, r, (gh // MERGE, gw // MERGE)) for tm, r in zip(times, rows)]

