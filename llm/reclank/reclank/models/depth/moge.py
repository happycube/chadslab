"""MoGe-2 (Ruicheng/moge-2-vitl, -vitl-normal): metric geometry of one image.

Ported from MoGe (microsoft/MoGe, moge/model/v2.py; MIT License; Copyright (c)
Microsoft Corporation.). See THIRD_PARTY_NOTICES.md.

    the image, resized bilinear (antialiased) to the token grid x 14 of
    num_tokens (1200..3600 by resolution_level 0..9; 9: 3600), ImageNet's mean
    and std -> DINOv2 ViT-L/14, layers 5, 11, 17, 23 (the final norm eps 1e-6),
    each 1 x 1 projected to 1024 and the four summed
    -> the neck: a ConvStack of 5 levels (1024 at the grid, 256, 128, 64, 32 at
       2x .. 16x), the view-plane UV as 2 more input channels at each level
       (a 1 x 1 input conv of them: a constant of the size here), residual
       blocks of two 3 x 3 convs (replicate padding, no norms), resamplers of
       a ConvTranspose 2x2 s2 (the last bilinear 2x) and a 3 x 3 conv
    -> heads of the same form on the neck's levels: points (3), mask (1),
       normal (3, -normal)
    -> the scale MLP (1024, 1024, 1024, 1) of the last layer's cls row: the
       metric scale, exp.

On the host, as the authors' infer(): the head outputs bilinear to the image
size, points remapped (z = exp, x y times z), the mask's sigmoid > 0.5, the
focal and the z shift by Levenberg-Marquardt (SciPy) on a 64 x 64 nearest
subsample, depth = (z + shift) x the metric scale, metres. The stereo path
takes the depth at the heads' size (16x the token grid) and the focal; the
full size only when the depth is asked for.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.depth.common import (Builder, ViT, conv_index, conv_lin, lin, normalize, patch_rows, run,
                                         torch_resize_aa)
from reclank.models.depth.ref import lerp_src


def uv_plane(width, height, aspect):
    """MoGe's normalized_view_plane_uv: (height, width, 2) float32"""
    sx = aspect / (1 + aspect ** 2) ** 0.5
    sy = 1 / (1 + aspect ** 2) ** 0.5
    u = np.linspace(-sx * (width - 1) / width, sx * (width - 1) / width, width, dtype=np.float32)
    v = np.linspace(-sy * (height - 1) / height, sy * (height - 1) / height, height, dtype=np.float32)
    return np.stack(np.meshgrid(u, v, indexing="xy"), -1).astype(np.float32)


class ConvStack:
    """MoGe's ConvStack (no norms) of a state dict prefix."""

    def __init__(self, get, p, cfg):
        self.cfg = cfg
        self.dims = cfg["dim_res_blocks"]
        n = len(self.dims)
        cv = lambda m: conv_lin(lambda k: get(p + k), m)  # noqa: E731
        dim_in = cfg["dim_in"] if isinstance(cfg["dim_in"], list) else [cfg["dim_in"]] * n
        dim_out = cfg["dim_out"] if isinstance(cfg["dim_out"], list) else [cfg["dim_out"]] * n
        nres = cfg["num_res_blocks"] if isinstance(cfg["num_res_blocks"], list) else [cfg["num_res_blocks"]] * n
        self.inp = [cv("input_blocks.%d" % i) if dim_in[i] is not None else None for i in range(n)]
        self.res = [[(cv("res_blocks.%d.%d.layers.2" % (i, j)), cv("res_blocks.%d.%d.layers.5" % (i, j)))
                     for j in range(nres[i])] for i in range(n)]
        self.out = [cv("output_blocks.%d" % i) if dim_out[i] is not None else None for i in range(n)]
        self.rs = []
        for i in range(n - 1):
            kind = cfg["resamplers"][i] if isinstance(cfg["resamplers"], list) else cfg["resamplers"]
            r = "resamplers.%d." % i
            if kind == "conv_transpose":
                self.rs.append(("ct", np.asarray(get(p + r + "0.weight"), np.float32),
                                np.asarray(get(p + r + "0.bias"), np.float32), cv(r + "1")))
            else:
                assert kind == "bilinear", kind
                self.rs.append(("bl", None, None, cv(r + "1")))

    def forms(self, B, ins, sizes, keep_levels=False, out_level=None):
        """ins[i]: the name of level i's input (None: none) or a constant table
        (added after its input conv, which the caller folded in). Returns the
        level outputs (keep_levels) or the output of out_level."""
        outs = []
        x = None
        n = len(self.dims)
        for i in range(n):
            h, w = sizes[i]
            kind, src = ins[i]
            if kind == "conv":
                f = B.linear(self.inp[i], src)
            elif kind == "table":
                f = src
            else:
                f = None
            if x is None:
                x = f
            elif f is not None:
                B.add(x, f)
                if kind == "conv":
                    B.free(f)
            idx = B.index(("rep", h, w), lambda: conv_index(h, w, pad="replicate"))
            for c1, c2 in self.res[i]:
                a = B.relu(x)
                b = B.conv3(a, idx, c1)
                B.free(a)
                B.f("set", b, ("enc_clamp", b, 0.0, float("inf")))
                c = B.conv3(b, idx, c2)
                B.free(b)
                B.add(c, x)
                B.free(x)
                x = c
            if keep_levels:
                outs.append(x)
            elif i == out_level:
                y = B.linear(self.out[i], x)
                B.free(x)
                return y
            if i < n - 1:
                kind_r, wt, bias, cv3 = self.rs[i]
                H, W = sizes[i + 1]
                if kind_r == "ct":
                    up = B.convT(x, 1, h, w, wt, bias, 2)
                else:
                    up = B.resize(x, 1, h, w, H, W, False)
                if not keep_levels:
                    B.free(x)
                x2 = B.conv3(up, B.index(("rep", H, W), lambda: conv_index(H, W, pad="replicate")), cv3)
                B.free(up)
                x = x2
        return outs


def converted(path, name):
    """the safetensors of a checkpoint converted by scripts/pt_to_safetensors.py"""
    f = os.path.join(path, name + ".safetensors")
    if not os.path.exists(f):
        raise FileNotFoundError("%s: convert the checkpoint once: scripts/pt_to_safetensors.py %s" % (
            f, os.path.join(path, name + (".pt" if name == "model" else ".pth"))))
    return SafeTensors(f)


class MoGe2:
    """MoGe-2 of a directory with model.safetensors and model.json (its
    model.pt converted by scripts/pt_to_safetensors.py)."""

    def __init__(self, path, gpu=False, cpu_threads=None, level=9):
        self.W = W = converted(path, "model")
        cfg = json.load(open(os.path.join(path, "model.json")))["model_config"]
        get = lambda n: np.asarray(W.get(n), np.float32)  # noqa: E731
        self.cfg, self.gpu, self.cpu_threads, self.level = cfg, bool(gpu), cpu_threads, level
        assert cfg.get("remap_output") == "exp"
        self.take = list(cfg["encoder"]["intermediate_layers"])
        self.vit = ViT(get, "encoder.backbone.", 16, final_eps=1e-6)
        self.proj = [conv_lin(get, "encoder.output_projections.%d" % i) for i in range(len(self.take))]
        self.neck = ConvStack(get, "neck.", cfg["neck"])
        self.heads = {h: ConvStack(get, h + ".", cfg[h]) for h in ("points_head", "mask_head", "normal_head") if h in cfg}
        self.scale = [lin(get("scale_head.%d.weight" % i), get("scale_head.%d.bias" % i)) for i in (0, 2, 4)]
        # the neck's level-0 input conv over [features, uv]: the features' part a linear, the uv's
        # part and the bias a table of the size
        w0 = self.neck.inp[0]
        C = self.vit.width
        self.neck0 = lin(w0.w[:, :C])
        self.neck0_uv = (w0.w[:, C:], w0.b)
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)
        self.runner.label = "moge2"
        self.tokens_range = cfg.get("num_tokens_range", [1200, 3600])

    def grid(self, h, w, num_tokens=None):
        if num_tokens is None:
            lo, hi = self.tokens_range
            num_tokens = int(lo + (self.level / 9) * (hi - lo))
        ar = w / h
        return round((num_tokens / ar) ** 0.5), round((num_tokens * ar) ** 0.5)

    def _build(self, key, gpu):
        bh, bw, ar = key
        B = Builder(gpu)
        P = bh * bw
        B.array("patches", np.zeros((P, 588), np.float32))
        feats = self.vit.forms(B, "patches", 1, bh, bw, self.take)
        x = None
        for i, f in enumerate(feats):
            y = B.linear(self.proj[i], B.view(f, 1, P + 1))
            if x is None:
                x = y
            else:
                B.add(x, y)
                B.free(y)
        cls = B.array("cls", np.zeros((1, self.vit.width), np.float32))
        B.f("set", cls, ("enc_clamp", B.view(feats[-1], 0, 1), float("-inf"), float("inf")))
        B.free(*feats)
        sizes = [(bh << i, bw << i) for i in range(5)]
        ins = []
        for lv in range(5):
            h, w = sizes[lv]
            uv = uv_plane(w, h, ar).reshape(-1, 2)
            if lv == 0:
                wu, b = self.neck0_uv
                t = (uv @ wu.T + b).astype(np.float32)
                y = B.linear(self.neck0, x)
                B.free(x)
                B.add(y, B.name(t))
                ins.append(("table", y))
            else:
                l_ = self.neck.inp[lv]
                ins.append(("table", B.name((uv @ l_.w.T + l_.b).astype(np.float32))))
        levels = self.neck.forms(B, ins, sizes, keep_levels=True)
        for h, st in self.heads.items():
            o = st.forms(B, [("conv", lv) for lv in levels], sizes, out_level=4)
            B.array(h, B.env[o])
        B.free(*levels)
        return B.finish(self.cpu_threads)

    def raw(self, img, num_tokens=None):
        """The heads' outputs at 16x the grid ({head: (16 bh, 16 bw, C)}) and the
        metric scale of an uint8 image."""
        h, w = img.shape[:2]
        bh, bw = self.grid(h, w, num_tokens)
        x = torch_resize_aa(img.astype(np.float32) / np.float32(255), bh * 14, bw * 14)
        names = sorted(self.heads) + ["cls"]
        outs = run(self.runner, (bh, bw, w / h), self.gpu, {"patches": patch_rows(normalize(x), bh, bw)}, names)
        r = {n: o.reshape(16 * bh, 16 * bw, -1) for n, o in zip(names[:-1], outs[:-1])}
        s = outs[-1].reshape(1, -1).astype(np.float32)
        for i, l_ in enumerate(self.scale):
            s = s @ l_.w.T + l_.b
            if i < 2:
                s = np.maximum(s, 0)
        r["metric_scale"] = float(np.exp(np.float32(s[0, 0])))
        r["grid"] = (bh, bw)
        return r

    def geometry(self, img, num_tokens=None, full=False):
        """dict(depth (metres; inf where the mask is off), valid, fx (in image
        widths), hfov (degrees), shift, metric_scale, normal (-normal)) at the
        heads' size, or at the image's (full)."""
        h, w = img.shape[:2]
        r = self.raw(img, num_tokens)
        pts, msk = r["points_head"], r["mask_head"][..., 0]
        # the focal and shift, as the reference: the points bilinear at the image size, then 64 x 64
        # nearest, so only those samples are made
        ys = np.minimum(np.floor(np.arange(64, dtype=np.float32) * np.float32(h / 64)).astype(np.int64), h - 1)
        xs = np.minimum(np.floor(np.arange(64, dtype=np.float32) * np.float32(w / 64)).astype(np.int64), w - 1)
        ps = sample_bilinear(pts, ys, xs, h, w)
        ms = 1 / (1 + np.exp(-sample_bilinear(msk[..., None], ys, xs, h, w)[..., 0].astype(np.float64))) > 0.5
        xyz = remap(ps)
        uv = uv_plane(w, h, w / h)[ys][:, xs]
        shift, focal = solve_focal_shift(uv[ms], xyz[ms])
        ar = w / h
        fx = focal / 2 * (1 + ar ** 2) ** 0.5 / ar
        if full:
            P = sample_bilinear(pts, np.arange(h), np.arange(w), h, w)
            M = sample_bilinear(msk[..., None], np.arange(h), np.arange(w), h, w)[..., 0]
        else:
            P, M = pts, msk
        z = remap(P)[..., 2] + np.float32(shift)
        valid = (1 / (1 + np.exp(-M.astype(np.float64))) > 0.5) & (z > 0)
        depth = np.where(valid, z * np.float32(r["metric_scale"]), np.inf).astype(np.float32)
        out = dict(depth=depth, valid=valid, fx=float(fx), hfov=float(2 * np.degrees(np.arctan(0.5 / fx))),
                   shift=float(shift), metric_scale=r["metric_scale"], grid=r["grid"])
        if "normal_head" in r:
            nr = r["normal_head"]
            if full:
                nr = sample_bilinear(nr, np.arange(h), np.arange(w), h, w)
            out["normal"] = nr / np.maximum(np.linalg.norm(nr, axis=-1, keepdims=True), 1e-12)
        return out


def sample_bilinear(m, ys, xs, H, W):
    """m (h, w, C) resized bilinear (align_corners False) to H x W, at the rows
    ys and columns xs only: (len ys, len xs, C)"""
    h, w = m.shape[:2]
    y0, y1, ly0, ly1 = lerp_src(h, H, 0)
    x0, x1, lx0, lx1 = lerp_src(w, W, 0)
    y0, y1, ly0, ly1 = y0[ys], y1[ys], ly0[ys][:, None, None], ly1[ys][:, None, None]
    x0, x1, lx0, lx1 = x0[xs], x1[xs], lx0[xs][None, :, None], lx1[xs][None, :, None]
    top = lx0 * m[y0][:, x0] + lx1 * m[y0][:, x1]
    bot = lx0 * m[y1][:, x0] + lx1 * m[y1][:, x1]
    return (ly0 * top + ly1 * bot).astype(np.float32)


def remap(p):
    """MoGe's 'exp' remap: z = exp(z), x y times z"""
    z = np.exp(p[..., 2:3].astype(np.float32))
    return np.concatenate([p[..., :2] * z, z], -1).astype(np.float32)


def solve_focal_shift(uv, xyz):
    """min |focal xy / (z + shift) - uv| over the shift (the focal in closed
    form): SciPy's Levenberg-Marquardt from 0, ftol 1e-3, as MoGe's
    utils.geometry_numpy.solve_optimal_focal_shift"""
    from scipy.optimize import least_squares
    uv, xy, z = uv.reshape(-1, 2), xyz[..., :2].reshape(-1, 2), xyz[..., 2].reshape(-1)
    if len(z) < 2:
        return np.float32(0), 1.0

    def fn(s):
        p = xy / (z + s)[:, None]
        f = (p * uv).sum() / np.square(p).sum()
        return (f * p - uv).ravel()
    s = least_squares(fn, x0=0, ftol=1e-3, method="lm")["x"].squeeze().astype(np.float32)
    p = xy / (z + s)[:, None]
    return s, float((p * uv).sum() / np.square(p).sum())
