"""The pieces the depth models share: a program builder with pooled buffers,
DINOv2's blocks, the 3 x 3 convs and transposed convs of the heads, and the
host's preprocessing (OpenCV's and torch's resizes, the positions of a patch
grid).

The DINOv2 blocks ported from DINOv2 (as vendored in the authors' repositories;
Apache License 2.0; Copyright (c) Meta Platforms, Inc. and affiliates.). See
THIRD_PARTY_NOTICES.md.

The rows of a program's buffers are pixels or tokens (row major, frame after
frame), the columns channels. A 3 x 3 conv is the im2col of its 9 neighbours
by ENC_GATHER (zeros or the replicated edge past the border) into a buffer
of TILE rows, then ENC_LINEAR; a transposed conv of kernel = stride k is
ENC_LINEAR to k k C values a pixel, the pixel shuffle by ENC_GATHER, the
bias by ENC_BIAS_CLAMP; bilinear upsampling is ENC_RESIZE. Weights float32 as
stored, on the CPU and the GPU (attention float32: ENC_ATTN_TILE).
"""
from __future__ import annotations

import math

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain

TILE = 32768        # rows of an im2col at a time
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def lin(w, b=None):
    """A linear of W (out, in) float32 and its bias."""
    out = _Plain(np.ascontiguousarray(w, dtype=np.float32))
    out.b = None if b is None else np.ascontiguousarray(b, dtype=np.float32)
    out.shape = out.w.shape
    return out


def conv_lin(get, name, bias=True):
    """A conv (out, in, k, k) of a state dict (get(name)) as the linear of its
    im2col, W (out, k k in): the columns (ky, kx, channel)."""
    w = np.asarray(get(name + ".weight"), np.float32)
    if w.ndim == 4:
        w = w.transpose(0, 2, 3, 1).reshape(w.shape[0], -1)
    b = get(name + ".bias") if bias else None
    return lin(w, None if b is None else np.asarray(b, np.float32))


def conv_index(h, w, frames=1, pad="zeros", stride=1):
    """The im2col index (rows, 9) of a 3 x 3 conv with padding 1 over frames
    images of h x w pixels (frame after frame): -1 past the edge (zeros), or
    the nearest pixel (replicate); stride 2 takes every other pixel."""
    H, W = (h - 1) // stride + 1, (w - 1) // stride + 1
    y = stride * np.arange(H)[:, None, None, None] + np.arange(-1, 2)[None, None, :, None]
    x = stride * np.arange(W)[None, :, None, None] + np.arange(-1, 2)[None, None, None, :]
    y, x = np.broadcast_arrays(y, x)
    if pad == "replicate":
        idx = np.clip(y, 0, h - 1) * w + np.clip(x, 0, w - 1)
    else:
        idx = np.where((y >= 0) & (y < h) & (x >= 0) & (x < w), y * w + x, -1)
    idx = idx.reshape(H * W, 9)
    if frames > 1:
        idx = np.concatenate([np.where(idx >= 0, idx + f * h * w, -1) for f in range(frames)])
    return np.ascontiguousarray(idx, dtype=np.int32)


class Builder:
    """The forms of one program and its buffers. buf() takes a buffer from a
    pool of free arrays (the smallest that fits) and free() gives it back, so
    a head's levels share their memory (a straight-line program reads each
    buffer only between its set and its free)."""

    def __init__(self, gpu, kmax=9 * 1024):
        self.gpu = bool(gpu)
        self.c = _EncCompiler(1e-6, pack=not gpu, rows=TILE, kmax=kmax)
        self.c.attn_f32 = True                  # ENC_ATTN_TILE: float32 attention on the GPU
        self.env = self.c.env
        self.forms = []
        self.n = 0
        self.free_list = []                     # (size, array)
        self.base = {}                          # a pooled name: its base array
        self.col = None
        self.idx = {}                           # index arrays by key: shared by the forms

    def name(self, arr):
        self.n += 1
        nm = "_b%d" % self.n
        self.env[nm] = arr
        return nm

    def array(self, name, arr):
        """A named array of the program (an input, an output, a constant)."""
        self.env[name] = arr
        return name

    def buf(self, rows, cols):
        need = rows * cols
        best = None
        for i, (size, arr) in enumerate(self.free_list):
            if size >= need and (best is None or size < self.free_list[best][0]):
                best = i
        if best is None:
            base = np.zeros(need, np.float32)
        else:
            base = self.free_list.pop(best)[1]
        nm = self.name(base[:need].reshape(rows, cols))
        self.base[nm] = base
        return nm

    def free(self, *names):
        for nm in names:
            base = self.base.pop(nm, None)
            if base is not None:
                self.free_list.append((base.size, base))

    def view(self, nm, r0, r1):
        """a name of rows r0:r1 of a buffer"""
        return self.name(self.env[nm][r0:r1])

    def f(self, *form):
        self.forms.append(form)

    def shape(self, nm):
        return self.env[nm].shape

    def linear(self, l_, x, out=None):
        out = self.buf(self.shape(x)[0], l_.shape[0]) if out is None else out
        self.f("set", out, ("enc_linear", l_, x))
        return out

    def index(self, key, make):
        if key not in self.idx:
            self.idx[key] = make()
        return self.idx[key]

    def conv3(self, x, idx, l_, out=None):
        """a 3 x 3 conv of x (rows: the image(s) of idx) by im2col tiles"""
        cin = self.shape(x)[1]
        rows = idx.shape[0]
        out = self.buf(rows, l_.shape[0]) if out is None else out
        need = min(rows, TILE) * 9 * cin
        if self.col is None or self.col.size < need:
            self.col = np.zeros(max(need, TILE * 9 * 256), np.float32)
        for r0 in range(0, rows, TILE):
            r1 = min(rows, r0 + TILE)
            col = self.name(self.col[:(r1 - r0) * 9 * cin].reshape(r1 - r0, 9 * cin))
            self.f("set", col, ("enc_gather", x, idx[r0:r1] if r0 or r1 < rows else idx, cin))
            self.f("set", self.view(out, r0, r1), ("enc_linear", l_, col))
        return out

    def relu(self, x, out=None):
        out = self.buf(*self.shape(x)) if out is None else out
        self.f("set", out, ("enc_clamp", x, 0.0, float("inf")))
        return out

    def add(self, x, y):
        self.f("enc_add", x, y)
        return x

    def resize(self, x, n, h, w, H, W, align, out=None):
        """n images of h x w (rows of C) to H x W, bilinear"""
        C = self.shape(x)[1]
        out = self.buf(n * H * W, C) if out is None else out
        self.f("enc_resize", x, out, n, h, w, H, W, C, 1 if align else 0)
        return out

    def convT(self, x, n, h, w, wt, bias, k):
        """ConvTranspose2d of kernel = stride k (wt (in, out, k, k)) of n
        images of h x w"""
        cin, cout = wt.shape[:2]
        y = self.linear(lin(wt.transpose(2, 3, 1, 0).reshape(k * k * cout, cin)), x)
        F_, Y, X = np.meshgrid(np.arange(n), np.arange(h * k), np.arange(w * k), indexing="ij")
        idx = self.index(("convT", n, h, w, k), lambda: np.ascontiguousarray(
            ((F_ * h * w + (Y // k) * w + X // k) * k * k + (Y % k) * k + X % k).reshape(-1, 1), dtype=np.int32))
        out = self.buf(n * h * w * k * k, cout)
        self.f("set", out, ("enc_gather", self.name(self.env[y].reshape(-1, cout)), idx, cout))
        self.free(y)
        self.f("enc_bias", out, np.asarray(bias, np.float32))
        return out

    def finish(self, threads=None):
        self.c.compile(("seq", *self.forms))
        p = self.c.p.finish()
        if not self.gpu:
            from reclank.runtime.threads import team_threads
            p.threads = p.buf[3] = threads or team_threads()
        return p


# ---- DINOv2 -----------------------------------------------------------------

def bicubic_matrix(n_in, n_out, scale_factor):
    """(n_out, n_in) of torch's bicubic (A = -0.75, align_corners False) given
    a scale factor (the source coordinate (d + 0.5) / scale_factor - 0.5), the
    edge clamped."""
    A = -0.75
    M = np.zeros((n_out, n_in), np.float64)
    for d in range(n_out):
        src = (d + 0.5) / scale_factor - 0.5
        i = math.floor(src)
        t = src - i
        c = [((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A,
             ((A + 2) * t - (A + 3)) * t * t + 1,
             ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1,
             ((A * (2 - t) - 5 * A) * (2 - t) + 8 * A) * (2 - t) - 4 * A]
        for k in range(4):
            M[d, min(max(i - 1 + k, 0), n_in - 1)] += c[k]
    return M


def pos_embed(pe, rows, cols):
    """DINOv2's interpolate_pos_encoding (bicubic by the scale factors (n +
    0.1) / M) of pos_embed (1, 1 + M M, C): (1 + rows cols, C) float32."""
    pe = np.asarray(pe, np.float32).reshape(-1, pe.shape[-1])
    N, C = pe.shape[0] - 1, pe.shape[1]
    M = int(round(math.sqrt(N)))
    if rows * cols == N and rows == cols:
        return pe.copy()
    g = pe[1:].reshape(M, M, C).astype(np.float64)
    my = bicubic_matrix(M, rows, (rows + 0.1) / M)
    mx = bicubic_matrix(M, cols, (cols + 0.1) / M)
    t = (my @ g.reshape(M, M * C)).reshape(rows, M, C)              # the rows, then the columns
    p = np.matmul(mx[None], t).astype(np.float32).reshape(rows * cols, C)
    return np.concatenate([pe[:1], p])


class ViT:
    """DINOv2 ViT/14 blocks of a state dict (get(prefix + name)): LayerScale,
    erf GELU MLP, the block norms eps 1e-6, the final norm final_eps."""

    def __init__(self, get, prefix, heads, final_eps=1e-6):
        self.get, self.p, self.heads, self.final_eps = get, prefix, heads, final_eps
        g = lambda n: np.asarray(get(prefix + n), np.float32)  # noqa: E731
        self.width = g("cls_token").shape[-1]
        self.pe = g("pos_embed")
        self.cls = g("cls_token").reshape(-1)
        wp = g("patch_embed.proj.weight")
        self.patch = lin(wp.reshape(wp.shape[0], -1), g("patch_embed.proj.bias"))
        C = self.width
        hd = C // heads
        s = np.float32(hd ** -0.5)
        self.blocks = []
        i = 0
        while True:
            b = "blocks.%d." % i
            try:
                qw, qb = g(b + "attn.qkv.weight"), g(b + "attn.qkv.bias")
            except KeyError:
                break
            self.blocks.append(dict(
                n1=(g(b + "norm1.weight"), g(b + "norm1.bias")), n2=(g(b + "norm2.weight"), g(b + "norm2.bias")),
                q=lin(qw[:C] * s, qb[:C] * s), k=lin(qw[C:2 * C], qb[C:2 * C]), v=lin(qw[2 * C:], qb[2 * C:]),
                o=lin(g(b + "attn.proj.weight"), g(b + "attn.proj.bias")), ls1=g(b + "ls1.gamma"),
                fc1=lin(g(b + "mlp.fc1.weight"), g(b + "mlp.fc1.bias")),
                fc2=lin(g(b + "mlp.fc2.weight"), g(b + "mlp.fc2.bias")), ls2=g(b + "ls2.gamma")))
            i += 1
        self.final = (g("norm.weight"), g("norm.bias"))

    def forms(self, B, patches, T, rows, cols, take):
        """The blocks over T images of rows x cols patches: patches the name of
        their (T rows cols, 588) input. The final-normed rows of the layers in
        take (T (1 + rows cols), C), cls rows included."""
        C, N = self.width, rows * cols
        P = N + 1
        pos = pos_embed(self.pe, rows, cols)
        src = B.name(np.zeros((T * N + 1, C), np.float32))     # the patch rows, then the cls row (a constant)
        B.env[src][T * N] = self.cls + pos[0]
        posn = B.name(np.ascontiguousarray(pos[1:]))
        B.f("set", B.view(src, 0, T * N), ("enc_linear", self.patch, patches))
        for f in range(T):
            B.add(B.view(src, f * N, (f + 1) * N), posn)
        idx = np.empty((T * P, 1), np.int32)
        for f in range(T):
            idx[f * P, 0] = T * N
            idx[f * P + 1:(f + 1) * P, 0] = np.arange(f * N, (f + 1) * N)
        x = B.buf(T * P, C)
        B.f("set", x, ("enc_gather", src, idx, C))
        win = np.repeat(np.arange(T), P)
        lo = B.name((win * P).astype(np.int32))
        hi = B.name((win * P + P).astype(np.int32))
        h, q, k, v, o, t = (B.buf(T * P, C) for _ in range(6))
        u = B.buf(T * P, 4 * C)
        H = self.heads
        outs = []
        for i, L in enumerate(self.blocks):
            B.forms += [
                ("set", h, ("enc_lnorm", x, L["n1"][0], L["n1"][1], C, 1e-6)),
                ("set", q, ("enc_linear", L["q"], h)), ("set", k, ("enc_linear", L["k"], h)),
                ("set", v, ("enc_linear", L["v"], h)),
                ("set", o, ("enc_attn_range_x2", q, k, v, lo, hi, H, C // H)),
                ("set", t, ("enc_linear", L["o"], o)),
                ("set", t, ("enc_mul_vec", t, L["ls1"])),
                ("enc_add", x, t),
                ("set", h, ("enc_lnorm", x, L["n2"][0], L["n2"][1], C, 1e-6)),
                ("set", u, ("enc_linear", L["fc1"], h)),
                ("set", u, ("enc_gelu", u, True)),
                ("set", t, ("enc_linear", L["fc2"], u)),
                ("set", t, ("enc_mul_vec", t, L["ls2"])),
                ("enc_add", x, t),
            ]
            if i in take:
                y = B.buf(T * P, C)
                B.f("set", y, ("enc_lnorm", x, self.final[0], self.final[1], C, self.final_eps))
                outs.append(y)
        B.free(x, h, q, k, v, o, t, u)
        return outs


def patch_rows(img, rows, cols):
    """(rows cols, 3 14 14) of a normalized image (3, 14 rows, 14 cols): each
    patch's values in the conv's order (channel, ky, kx)."""
    x = img.reshape(3, rows, 14, cols, 14).transpose(1, 3, 0, 2, 4)
    return np.ascontiguousarray(x.reshape(rows * cols, 3 * 196), dtype=np.float32)


def patch_views(B, x, T, rows, cols):
    """the names of each frame's patch rows (no cls row) of a ViT output"""
    P = rows * cols + 1
    return [B.view(x, f * P + 1, (f + 1) * P) for f in range(T)]


# ---- the host's resizes -----------------------------------------------------

def _area_matrix(n_in, n_out):
    """(n_out, n_in): OpenCV's INTER_AREA (the overlap of each output pixel's
    span with each input pixel, normalized), a downscale"""
    s = n_in / n_out
    M = np.zeros((n_out, n_in))
    for d in range(n_out):
        a, b = d * s, (d + 1) * s
        for i in range(int(math.floor(a)), min(int(math.ceil(b)), n_in)):
            M[d, i] = min(b, i + 1) - max(a, i)
    return M / M.sum(1, keepdims=True)


def _cubic_matrix_cv(n_in, n_out):
    """(n_out, n_in): OpenCV's INTER_CUBIC (A = -0.75, half-pixel centres, the
    edge replicated)"""
    A = -0.75
    s = n_in / n_out
    M = np.zeros((n_out, n_in))
    for d in range(n_out):
        src = (d + 0.5) * s - 0.5
        i = math.floor(src)
        t = src - i
        c0 = ((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A
        c1 = ((A + 2) * t - (A + 3)) * t * t + 1
        c2 = ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1
        for k, c in enumerate((c0, c1, c2, 1 - c0 - c1 - c2)):
            M[d, min(max(i - 1 + k, 0), n_in - 1)] += c
    return M


def _linear_matrix_cv(n_in, n_out):
    """(n_out, n_in): OpenCV's INTER_LINEAR"""
    s = n_in / n_out
    M = np.zeros((n_out, n_in))
    for d in range(n_out):
        src = (d + 0.5) * s - 0.5
        i = math.floor(src)
        t = src - i
        if i < 0:
            i, t = 0, 0.0
        if i >= n_in - 1:
            i, t = n_in - 1, 0.0
        M[d, i] += 1 - t
        M[d, min(i + 1, n_in - 1)] += t
    return M


def taps(M):
    """the nonzero columns and weights of each row of an interpolation matrix:
    (idx (n_out, k), w (n_out, k)), padded with weight 0"""
    k = max(1, int((M != 0).sum(1).max()))
    idx = np.zeros((M.shape[0], k), np.int64)
    w = np.zeros((M.shape[0], k), np.float64)
    for d in range(M.shape[0]):
        nz = np.nonzero(M[d])[0]
        idx[d, :len(nz)] = nz
        w[d, :len(nz)] = M[d, nz]
    return idx, w


_TAPS = {}


def _resize_taps(x, my, mx):
    """x (h, w, C) resized by the row matrices my (H, h) and mx (W, w), as
    sums of their taps (float32)"""
    ky = _TAPS.get(id(my)) or taps(my)
    kx = _TAPS.get(id(mx)) or taps(mx)
    x = np.asarray(x, np.float32)
    iy, wy = ky
    y = np.zeros((iy.shape[0],) + x.shape[1:], np.float32)
    for k in range(iy.shape[1]):
        y += wy[:, k, None, None].astype(np.float32) * x[iy[:, k]]
    ix, wx = kx
    z = np.zeros((y.shape[0], ix.shape[0]) + x.shape[2:], np.float32)
    for k in range(ix.shape[1]):
        z += wx[None, :, k, None].astype(np.float32) * y[:, ix[:, k]]
    return z


_MATS = {}


def _matrix(kind, n_in, n_out):
    key = (kind, n_in, n_out)
    if key not in _MATS:
        mk = {"area": _area_matrix, "cubic": _cubic_matrix_cv, "linear": _linear_matrix_cv,
              "aa": aa_bilinear_matrix}[kind]
        M = mk(n_in, n_out)
        _MATS[key] = (M, taps(M))
    return _MATS[key]


def cv_resize(img, width, height, kind):
    """cv2.resize of an (H, W, 3) image (area, cubic, linear) as sums of the
    filters' taps in float32: uint8 in, uint8 out (rounded); float32 in,
    float32 out. OpenCV's own uint8 cubic and linear are fixed point, so a
    level may differ."""
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return img.copy()
    y = _resize_by(img, kind, h, w, height, width)
    if img.dtype == np.uint8:
        return np.clip(np.rint(y), 0, 255).astype(np.uint8)
    return y


def _resize_by(img, kind, h, w, H, W):
    (my, ty), (mx, tx) = _matrix(kind, h, H), _matrix(kind, w, W)
    _TAPS[id(my)], _TAPS[id(mx)] = ty, tx
    return _resize_taps(img, my, mx)


def aa_bilinear_matrix(n_in, n_out):
    """(n_out, n_in): torch's interpolate(bilinear, antialias=True,
    align_corners=False) of one axis (the triangle filter widened by the
    downscale)"""
    scale = n_in / n_out
    support = scale if scale >= 1 else 1.0
    inv = 1.0 / scale if scale >= 1 else 1.0
    M = np.zeros((n_out, n_in))
    for d in range(n_out):
        center = scale * (d + 0.5)
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), n_in)
        ws = np.array([max(0.0, 1.0 - abs((j + xmin - center + 0.5) * inv)) for j in range(xmax - xmin)])
        tot = ws.sum()
        M[d, xmin:xmax] = ws / tot if tot else ws
    return M


def torch_resize_aa(img, height, width):
    """an (H, W, 3) float image resized as torch's antialiased bilinear"""
    h, w = img.shape[:2]
    return _resize_by(img, "aa", h, w, height, width)


def normalize(img):
    """(3, H, W) float32 of an (H, W, 3) image in [0, 1] (float) or uint8"""
    x = np.asarray(img, np.float32)
    if img.dtype == np.uint8:
        x = x / np.float32(255)
    return np.ascontiguousarray(((x - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)


def run(runner, key, gpu, inputs, outputs):
    """Write the inputs (name: array) of the program of key, run it, and
    return copies of the outputs (names)."""
    prog, g = runner.get(key, gpu)
    for name, a in inputs.items():
        prog.names[name][...] = a
        if g is not None:
            g.upload(name)
    if g is not None:
        g.run()
        for name in outputs:
            g.download(name)
    else:
        prog.run()
    return [prog.names[name].copy() for name in outputs]


def resize_taps_of(kind, h, w, H, W):
    """(iy, wy, ix, wx) of a resize (area, cubic, linear: OpenCV's; aa: torch's
    antialiased bilinear) for ENC_RESIZE_TAPS: int32 indices, float32 weights"""
    (_, (iy, wy)), (_, (ix, wx)) = _matrix(kind, h, H), _matrix(kind, w, W)
    return (iy.astype(np.int32), wy.astype(np.float32), ix.astype(np.int32), wx.astype(np.float32))


def prep_forms(B, frame, h, w, nh, nw, kind="cubic"):
    """forms of a model's input in a program: frame (h w, 3) in [0, 1] resized
    to nh x nw (ENC_RESIZE_TAPS), normalized ((x - mean) / std as x + (-mean)
    times 1 / std), its patch rows (rows cols, 588) by a gather. Returns the
    name of the patch rows."""
    iy, wy, ix, wx = resize_taps_of(kind, h, w, nh, nw)
    img = B.name(np.zeros((nh * nw, 3), np.float32))
    B.f("enc_resize_taps", frame, img, 1, h, w, nh, nw, 3, iy, wy, ix, wx)
    B.f("enc_bias", img, -MEAN)
    B.f("set", img, ("enc_mul_vec", img, (np.float32(1) / STD).astype(np.float32)))
    rows, cols = nh // 14, nw // 14
    py, px, c, ky, kx = np.meshgrid(np.arange(rows), np.arange(cols), np.arange(3), np.arange(14), np.arange(14),
                                    indexing="ij")
    idx = (((py * 14 + ky) * nw + px * 14 + kx) * 3 + c).reshape(-1, 1).astype(np.int32)
    out = B.name(np.zeros((rows * cols, 588), np.float32))
    B.f("set", out, ("enc_gather", B.name(B.env[img].reshape(-1, 1)), idx, 1))
    return out
