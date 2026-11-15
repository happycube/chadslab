"""Depth references for scripts/check_stereo.py: a torch-only float32 reference of the depth
models on the CPU, written from the authors' code (read, not run: ByteDance-Seed/Depth-Anything-3,
DepthAnything/Video-Depth-Anything, microsoft/MoGe), its DINOv2 checked against transformers'
Dinov2Model (the same outputs at 518 x 518). No autocast (the authors' API runs bfloat16 / float16
on CUDA).

Written from Depth Anything 3 (ByteDance-Seed/Depth-Anything-3, src/depth_anything_3; Apache
License 2.0; Copyright (c) 2025 ByteDance Ltd. and/or its affiliates), Video Depth Anything
(DepthAnything/Video-Depth-Anything, video_depth_anything; Apache License 2.0; Copyright (2025)
Bytedance Ltd. and/or its affiliates), MoGe (microsoft/MoGe, moge; MIT License; Copyright (c)
Microsoft Corporation.) and DINOv2 (as vendored in the authors' repositories; Apache License 2.0;
Copyright (c) Meta Platforms, Inc. and affiliates.). See THIRD_PARTY_NOTICES.md.

    stereo_reference.py da3mono MODEL_DIR IMAGE out.npz [--res 504]
    stereo_reference.py moge MODEL_DIR IMAGE out.npz [--tokens 3600]
    stereo_reference.py vdavideo MODEL_DIR FRAMES_DIR out.npz [--n 96]

da3mono (DA3MONO-LARGE): "image" the processed image (upper_bound_resize, OpenCV), "tokens" the
ViT's input rows (patch embedding + positions), "feat0".."feat3" the four normed outputs (patch
rows), "logits" the depth head's output (depth = exp), "depth", "sky" (the sky branch, ReLU).
moge (moge-2-vitl[-normal]): "points_head", "mask_head" (and "normal" at the image size) the
heads' outputs at 16x the token grid, "points" (the image size, remapped), "mask" (sigmoid),
"metric_scale", "tokens", the four ViT outputs, and as MoGe's infer(): "shift", "fx" (in widths),
"depth" (metres; inf off the mask), "valid". vdavideo (Video-Depth-Anything-Small): the authors'
infer_video_depth of the first n frames (PNG files, sorted) at the frames' size, fp32: "depth"
(n, H, W). Run in a venv with torch, opencv-python-headless and scipy."""
import argparse
import glob
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

torch.set_grad_enabled(False)
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_st(path):
    from safetensors.numpy import load_file
    return {k: torch.from_numpy(v) for k, v in load_file(path).items()}


# ---- DINOv2 -----------------------------------------------------------------

def pos_embed(pe, rows, cols):
    """DINOv2's interpolate_pos_encoding: bicubic by scale factors with the 0.1 offset."""
    N = pe.shape[1] - 1
    M = int(math.sqrt(N))
    if N == rows * cols and rows == cols:
        return pe
    p = pe[:, 1:].reshape(1, M, M, -1).permute(0, 3, 1, 2)
    p = F.interpolate(p, scale_factor=((rows + 0.1) / M, (cols + 0.1) / M), mode="bicubic", antialias=False)
    assert p.shape[-2:] == (rows, cols)
    p = p.permute(0, 2, 3, 1).reshape(1, rows * cols, -1)
    return torch.cat([pe[:, :1], p], 1)


def rope2d(t, pos, freq=100.0):
    """DA3's RotaryPositionEmbedding2D: half the head by the row, half by the column."""
    d = t.shape[-1] // 2
    ex = torch.arange(0, d, 2).float() / d
    inv = 1.0 / (freq ** ex)
    n = int(pos.max()) + 1
    ang = torch.einsum("i,j->ij", torch.arange(n, dtype=inv.dtype), inv)
    ang = torch.cat([ang, ang], -1)
    cos, sin = ang.cos(), ang.sin()

    def one(x, p):
        c, s = cos[p][:, None], sin[p][:, None]           # (B, 1, N, d)
        x1, x2 = x[..., :d // 2], x[..., d // 2:]
        return x * c + torch.cat([-x2, x1], -1) * s
    v, h = t.chunk(2, -1)
    return torch.cat([one(v, pos[..., 0]), one(h, pos[..., 1])], -1)


class ViT:
    """DINOv2 ViT of a state dict's prefix p: blocks (LayerScale, MLP), qk-norm
    and RoPE from layer qk0 / rope0 (DA3), heads of 64."""

    def __init__(self, W, p, heads, eps=1e-6, final_eps=1e-6, qk0=-1, rope0=-1):
        self.W, self.p, self.H = W, p, heads
        self.eps, self.final_eps, self.qk0, self.rope0 = eps, final_eps, qk0, rope0
        self.n = 1 + max(int(k[len(p) + 7:].split(".")[0]) for k in W if k.startswith(p + "blocks."))

    def w(self, n):
        return self.W[self.p + n]

    def tokens(self, img):
        x = F.conv2d(img, self.w("patch_embed.proj.weight"), self.w("patch_embed.proj.bias"), stride=14)
        rows, cols = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat([self.w("cls_token").expand(x.shape[0], -1, -1), x], 1)
        return x + pos_embed(self.w("pos_embed"), rows, cols), rows, cols

    def block(self, i, x, pos=None):
        b = "blocks.%d." % i
        C = x.shape[-1]
        B, N = x.shape[:2]
        h = F.layer_norm(x, (C,), self.w(b + "norm1.weight"), self.w(b + "norm1.bias"), self.eps)
        qkv = F.linear(h, self.w(b + "attn.qkv.weight"), self.w(b + "attn.qkv.bias"))
        q, k, v = qkv.reshape(B, N, 3, self.H, C // self.H).permute(2, 0, 3, 1, 4)
        if self.qk0 >= 0 and i >= self.qk0:
            hd = C // self.H
            q = F.layer_norm(q, (hd,), self.w(b + "attn.q_norm.weight"), self.w(b + "attn.q_norm.bias"), 1e-5)
            k = F.layer_norm(k, (hd,), self.w(b + "attn.k_norm.weight"), self.w(b + "attn.k_norm.bias"), 1e-5)
        if pos is not None:
            q, k = rope2d(q, pos), rope2d(k, pos)
        a = F.scaled_dot_product_attention(q, k, v)
        a = F.linear(a.transpose(1, 2).reshape(B, N, C), self.w(b + "attn.proj.weight"), self.w(b + "attn.proj.bias"))
        x = x + a * self.w(b + "ls1.gamma")
        h = F.layer_norm(x, (C,), self.w(b + "norm2.weight"), self.w(b + "norm2.bias"), self.eps)
        h = F.linear(F.gelu(F.linear(h, self.w(b + "mlp.fc1.weight"), self.w(b + "mlp.fc1.bias"))),
                     self.w(b + "mlp.fc2.weight"), self.w(b + "mlp.fc2.bias"))
        return x + h * self.w(b + "ls2.gamma")

    def final_norm(self, x):
        return F.layer_norm(x, (x.shape[-1],), self.w("norm.weight"), self.w("norm.bias"), self.final_eps)

    def layers(self, img, take):
        """The final-normed outputs of the layers in take (patch rows only), and the cls rows."""
        x, rows, cols = self.tokens(img)
        self.x0 = x
        out, cls = [], []
        for i in range(self.n):
            x = self.block(i, x)
            if i in take:
                y = self.final_norm(x)
                out.append(y[:, 1:])
                cls.append(y[:, 0])
        return out, cls, rows, cols

    def layers_da3(self, img, take, alt0):
        """DA3-LARGE's backbone for one view (S = 1): the camera token in the cls
        row at alt0, from alt0 the odd layers 'global' (RoPE of no position
        difference: every patch at (1, 1)) and the even ones local (RoPE of the
        grid + 1), each output the [last local output, normed output] pair."""
        x, rows, cols = self.tokens(img)
        B, N = x.shape[:2]
        yy, xx = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
        grid = torch.stack([yy, xx], -1).reshape(1, -1, 2) + 1
        special = torch.zeros(1, 1, 2, dtype=grid.dtype)
        pos_l = torch.cat([special, grid], 1)
        pos_g = torch.cat([special, torch.ones_like(grid)], 1)
        out, local_x = [], None
        for i in range(self.n):
            if i == alt0:
                x = x.clone()
                x[:, 0] = self.w("camera_token")[:, 0]
            if i < self.rope0:
                x = self.block(i, x)
                local_x = x
            elif i % 2 == 1:
                x = self.block(i, x, pos_g)
            else:
                x = self.block(i, x, pos_l)
                local_x = x
            if i in take:
                out.append(torch.cat([local_x, self.final_norm(x)], -1)[:, 1:])
        return out, rows, cols


# ---- DPT pieces -------------------------------------------------------------

def conv(W, n, x, pad=None, stride=1):
    w = W[n + ".weight"]
    b = W.get(n + ".bias")
    if pad == "replicate":
        k = w.shape[-1] // 2
        return F.conv2d(F.pad(x, (k, k, k, k), mode="replicate"), w, b, stride=stride)
    return F.conv2d(x, w, b, stride=stride, padding=w.shape[-1] // 2)


def rcu(W, n, x):
    """ResidualConvUnit: relu, conv, relu, conv, + x"""
    o = conv(W, n + ".conv1", F.relu(x))
    o = conv(W, n + ".conv2", F.relu(o))
    return o + x


def fusion(W, n, x, lat=None, size=None):
    if lat is not None:
        x = x + rcu(W, n + ".resConfUnit1", lat)
    x = rcu(W, n + ".resConfUnit2", x)
    if size is None:
        size = (x.shape[-2] * 2, x.shape[-1] * 2)
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=True)
    return conv(W, n + ".out_conv", x)


def uv_embed(C, ph, pw, W_, H_, ratio=0.1):
    """DA3's _add_pos_embed table (C, ph, pw)."""
    ar = W_ / H_
    diag = (ar ** 2 + 1.0) ** 0.5
    sx, sy = ar / diag, 1.0 / diag
    xs = torch.linspace(-sx * (pw - 1) / pw, sx * (pw - 1) / pw, pw, dtype=torch.float32)
    ys = torch.linspace(-sy * (ph - 1) / ph, sy * (ph - 1) / ph, ph, dtype=torch.float32)
    uu, vv = torch.meshgrid(xs, ys, indexing="xy")
    pos = torch.stack([uu, vv], -1).reshape(-1, 2)

    def sincos(d, p):
        om = torch.arange(d // 2, dtype=torch.float32)
        om /= d / 2.0
        om = 1.0 / 100 ** om
        o = torch.einsum("m,d->md", p.reshape(-1), om)
        return torch.cat([torch.sin(o), torch.cos(o)], 1)
    e = torch.cat([sincos(C // 2, pos[:, 0]), sincos(C // 2, pos[:, 1])], -1)
    return (e.view(ph, pw, C) * ratio).permute(2, 0, 1)[None]


def dpt_da3(W, h, feats, ph, pw, H_, W_, norm=False, pe=False):
    """DA3's DPT / DualDPT main path: the logits (1, out, 14 ph, 14 pw)."""
    rs = []
    for i, x in enumerate(feats):
        if norm:
            x = F.layer_norm(x, (x.shape[-1],), W[h + "norm.weight"], W[h + "norm.bias"], 1e-5)
        x = x.permute(0, 2, 1).reshape(x.shape[0], -1, ph, pw)
        x = conv(W, h + "projects.%d" % i, x)
        if pe:
            x = x + uv_embed(x.shape[1], x.shape[2], x.shape[3], W_, H_)
        if i == 0:
            x = F.conv_transpose2d(x, W[h + "resize_layers.0.weight"], W[h + "resize_layers.0.bias"], stride=4)
        elif i == 1:
            x = F.conv_transpose2d(x, W[h + "resize_layers.1.weight"], W[h + "resize_layers.1.bias"], stride=2)
        elif i == 3:
            x = conv(W, h + "resize_layers.3", x, stride=2)
        rs.append(x)
    s = h + "scratch."
    l1, l2, l3, l4 = [F.conv2d(r, W[s + "layer%d_rn.weight" % (i + 1)], None, padding=1) for i, r in enumerate(rs)]
    o = fusion(W, s + "refinenet4", l4, size=l3.shape[2:])
    o = fusion(W, s + "refinenet3", o, l3, size=l2.shape[2:])
    o = fusion(W, s + "refinenet2", o, l2, size=l1.shape[2:])
    o = fusion(W, s + "refinenet1", o, l1)
    feat = conv(W, s + "output_conv1", o)
    feat = F.interpolate(feat, (ph * 14, pw * 14), mode="bilinear", align_corners=True)
    if pe:
        feat = feat + uv_embed(feat.shape[1], feat.shape[2], feat.shape[3], W_, H_)
    y = conv(W, s + "output_conv2.2", F.relu(conv(W, s + "output_conv2.0", feat)))
    return y, feat


def da3_preprocess(img, res):
    """upper_bound_resize to res, then each side to the nearest multiple of 14
    (cv2 INTER_AREA down, INTER_CUBIC up), normalized; (1, 3, H, W)."""
    import cv2
    h, w = img.shape[:2]
    s = res / max(h, w)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    a = img if max(h, w) == res else cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
    def near(x):
        down = (x // 14) * 14
        return down + 14 if abs(down + 14 - x) <= abs(x - down) else down
    mw, mh = max(1, near(nw)), max(1, near(nh))
    if (mw, mh) != (nw, nh):
        up = mw > nw or mh > nh
        a = cv2.resize(a, (mw, mh), interpolation=cv2.INTER_CUBIC if up else cv2.INTER_AREA)
    x = (a.astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())[None], a


def run_da3mono(path, img, res):
    W = load_st(os.path.join(path, "model.safetensors"))
    W = {k[len("model."):]: v for k, v in W.items()}
    vit = ViT(W, "backbone.pretrained.", 16, eps=1e-6, final_eps=1e-5)
    x, a = da3_preprocess(img, res)
    H_, W_ = x.shape[-2:]
    t = time.time()
    feats, _, ph, pw = vit.layers(x, [4, 11, 17, 23])
    t1 = time.time()
    y, feat = dpt_da3(W, "head.", feats, ph, pw, H_, W_)
    depth = torch.exp(y)[0, 0]
    sky = F.relu(conv(W, "head.scratch.sky_output_conv2.2",
                      F.relu(conv(W, "head.scratch.sky_output_conv2.0", feat))))[0, 0]
    t2 = time.time()
    return dict(depth=depth.numpy(), sky=sky.numpy(), image=a, feats=[f[0].numpy() for f in feats],
                tokens=vit.x0[0].numpy(), logits=y[0, 0].numpy(), t_vit=t1 - t, t_head=t2 - t1)


def run_da3(path, img, res):
    W = load_st(os.path.join(path, "model.safetensors"))
    W = {k[len("model."):]: v for k, v in W.items()}
    vit = ViT(W, "backbone.pretrained.", 16, eps=1e-6, final_eps=1e-5, qk0=8, rope0=8)
    x, a = da3_preprocess(img, res)
    H_, W_ = x.shape[-2:]
    t = time.time()
    feats, ph, pw = vit.layers_da3(x, [11, 15, 19, 23], 8)
    t1 = time.time()
    y, _ = dpt_da3(W, "head.", feats, ph, pw, H_, W_, norm=True, pe=True)
    t2 = time.time()
    return dict(depth=torch.exp(y[0, 0]).numpy(), conf=(torch.exp(y[0, 1]) + 1).numpy(), image=a,
                feats=[f[0].numpy() for f in feats], t_vit=t1 - t, t_head=t2 - t1)


# ---- MoGe-2 -----------------------------------------------------------------

def moge_uv(w, h, ar):
    sx = ar / (1 + ar ** 2) ** 0.5
    sy = 1 / (1 + ar ** 2) ** 0.5
    u = torch.linspace(-sx * (w - 1) / w, sx * (w - 1) / w, w)
    v = torch.linspace(-sy * (h - 1) / h, sy * (h - 1) / h, h)
    u, v = torch.meshgrid(u, v, indexing="xy")
    return torch.stack([u, v], -1)


def moge_stack(W, p, cfg, feats):
    """ConvStack of MoGe (no norms: res_block_in_norm / hidden_norm 'none'),
    3 x 3 convs with replicate padding; the outputs of each level."""
    outs, x = [], None
    n = len(cfg["dim_res_blocks"])
    for i in range(n):
        f = feats[i]
        if cfg["dim_in"][i] is not None:
            f = conv(W, p + "input_blocks.%d" % i, f)
        x = f if i == 0 else (x + f if f is not None else x)
        for j in range(cfg["num_res_blocks"][i]):
            r = p + "res_blocks.%d.%d.layers." % (i, j)
            h = conv(W, r + "2", F.relu(x), pad="replicate")
            h = conv(W, r + "5", F.relu(h), pad="replicate")
            x = x + h
        dout = cfg["dim_out"][i] if isinstance(cfg["dim_out"], list) else cfg["dim_out"]
        outs.append(conv(W, p + "output_blocks.%d" % i, x) if dout is not None else x)
        if i < n - 1:
            r = p + "resamplers.%d." % i
            if cfg["resamplers"][i] == "conv_transpose":
                x = F.conv_transpose2d(x, W[r + "0.weight"], W[r + "0.bias"], stride=2)
                x = conv(W, r + "1", x, pad="replicate")
            else:
                x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
                x = conv(W, r + "1", x, pad="replicate")
    return outs


def run_moge(path, img, tokens):
    ck = torch.load(os.path.join(path, "model.pt"), map_location="cpu", weights_only=True)
    cfg, W = ck["model_config"], ck["model"]
    vit = ViT(W, "encoder.backbone.", 16, eps=1e-6, final_eps=1e-6)
    H, Wd = img.shape[:2]
    ar = Wd / H
    bh, bw = round((tokens / ar) ** 0.5), round((tokens * ar) ** 0.5)
    im = torch.from_numpy(img.astype(np.float32).transpose(2, 0, 1).copy())[None] / 255.0
    t = time.time()
    x14 = F.interpolate(im, (bh * 14, bw * 14), mode="bilinear", align_corners=False, antialias=True)
    x14 = (x14 - W["encoder.image_mean"]) / W["encoder.image_std"]
    feats, cls, _, _ = vit.layers(x14, cfg["encoder"]["intermediate_layers"])
    x = sum(conv(W, "encoder.output_projections.%d" % i, f.permute(0, 2, 1).reshape(1, -1, bh, bw))
            for i, f in enumerate(feats))
    t1 = time.time()
    levels = [x, None, None, None, None]
    for lv in range(5):
        uv = moge_uv(bw * 2 ** lv, bh * 2 ** lv, ar).permute(2, 0, 1)[None]
        levels[lv] = uv if levels[lv] is None else torch.cat([levels[lv], uv], 1)
    nk = moge_stack(W, "neck.", cfg["neck"], levels)
    out = dict(image=img, tokens=(bh, bw))
    for head in ("points_head", "normal_head", "mask_head"):
        if head in cfg:
            out[head] = moge_stack(W, head + ".", cfg[head], nk)[-1]
    s = cls[-1]
    for i in (0, 2):
        s = F.relu(F.linear(s, W["scale_head.%d.weight" % i], W["scale_head.%d.bias" % i]))
    s = F.linear(s, W["scale_head.4.weight"], W["scale_head.4.bias"])
    t2 = time.time()
    pts = F.interpolate(out["points_head"], (H, Wd), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
    xy, z = pts.split([2, 1], -1)
    z = torch.exp(z)
    out["points"] = torch.cat([xy * z, z], -1).numpy()
    out["mask"] = torch.sigmoid(F.interpolate(out["mask_head"], (H, Wd), mode="bilinear", align_corners=False)[0, 0]).numpy()
    if "normal_head" in out:
        nrm = F.interpolate(out["normal_head"], (H, Wd), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
        out["normal"] = F.normalize(nrm, dim=-1).numpy()
        del out["normal_head"]
    out["metric_scale"] = float(torch.exp(s[0, 0]))
    for k in ("points_head", "mask_head"):
        out[k] = out[k][0].numpy()
    out["feats"] = [f[0].numpy() for f in feats]
    out["t_vit"], out["t_head"] = t1 - t, t2 - t1
    return out


# ---- Video Depth Anything ---------------------------------------------------

def vda_attn_module(W, p, x, T):
    """TemporalModule over x (B T, C, h, w): GroupNorm 32, proj_in, one block of
    two temporal self-attentions (sinusoidal positions, 8 heads) and a GEGLU FFN,
    proj_out, + x."""
    BT, C, h, w = x.shape
    t = p + "temporal_transformer."
    r = x
    y = F.group_norm(x, 32, W[t + "norm.weight"], W[t + "norm.bias"], 1e-6)
    y = y.permute(0, 2, 3, 1).reshape(BT, h * w, C)
    y = F.linear(y, W[t + "proj_in.weight"], W[t + "proj_in.bias"])
    b = t + "transformer_blocks.0."
    for i in range(2):
        a = b + "attention_blocks.%d." % i
        pe = W[a + "pos_encoder.pe"]
        hn = F.layer_norm(y, (C,), W[b + "norms.%d.weight" % i], W[b + "norms.%d.bias" % i], 1e-5)
        hs = hn.reshape(1, T, h * w, C).permute(0, 2, 1, 3).reshape(h * w, T, C)       # (b d) f c
        hs = hs + pe[:, :T]
        q = F.linear(hs, W[a + "to_q.weight"])
        k = F.linear(hs, W[a + "to_k.weight"])
        v = F.linear(hs, W[a + "to_v.weight"])
        H = 8
        q, k, v = [z.reshape(h * w, T, H, C // H).transpose(1, 2) for z in (q, k, v)]
        o = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(h * w, T, C)
        o = F.linear(o, W[a + "to_out.0.weight"], W[a + "to_out.0.bias"])
        o = o.reshape(1, h * w, T, C).permute(0, 2, 1, 3).reshape(T, h * w, C)
        y = y + o
    hn = F.layer_norm(y, (C,), W[b + "ff_norm.weight"], W[b + "ff_norm.bias"], 1e-5)
    g, gate = F.linear(hn, W[b + "ff.net.0.proj.weight"], W[b + "ff.net.0.proj.bias"]).chunk(2, -1)
    y = y + F.linear(g * F.gelu(gate), W[b + "ff.net.2.weight"], W[b + "ff.net.2.bias"])
    y = F.linear(y, W[t + "proj_out.weight"], W[t + "proj_out.bias"])
    return y.reshape(BT, h, w, C).permute(0, 3, 1, 2) + r


def vda_forward(W, vit, x, T):
    """VideoDepthAnything.forward for one window (T, 3, H, W): relu depth (T, H, W)."""
    H_, W_ = x.shape[-2:]
    feats, _, ph, pw = vit.layers(x, [2, 5, 8, 11])
    h = "head."
    rs = []
    for i, f in enumerate(feats):
        f = f.permute(0, 2, 1).reshape(T, -1, ph, pw)
        f = conv(W, h + "projects.%d" % i, f)
        if i == 0:
            f = F.conv_transpose2d(f, W[h + "resize_layers.0.weight"], W[h + "resize_layers.0.bias"], stride=4)
        elif i == 1:
            f = F.conv_transpose2d(f, W[h + "resize_layers.1.weight"], W[h + "resize_layers.1.bias"], stride=2)
        elif i == 3:
            f = conv(W, h + "resize_layers.3", f, stride=2)
        rs.append(f)
    l1, l2, l3, l4 = rs
    l3 = vda_attn_module(W, h + "motion_modules.0.", l3, T)
    l4 = vda_attn_module(W, h + "motion_modules.1.", l4, T)
    s = h + "scratch."
    l1, l2, l3, l4 = [F.conv2d(r, W[s + "layer%d_rn.weight" % (i + 1)], None, padding=1) for i, r in enumerate((l1, l2, l3, l4))]
    p4 = fusion(W, s + "refinenet4", l4, size=l3.shape[2:])
    p4 = vda_attn_module(W, h + "motion_modules.2.", p4, T)
    p3 = fusion(W, s + "refinenet3", p4, l3, size=l2.shape[2:])
    p3 = vda_attn_module(W, h + "motion_modules.3.", p3, T)
    p2 = fusion(W, s + "refinenet2", p3, l2, size=l1.shape[2:])
    p1 = fusion(W, s + "refinenet1", p2, l1)
    o = conv(W, s + "output_conv1", p1)
    o = F.interpolate(o, (ph * 14, pw * 14), mode="bilinear", align_corners=True)
    o = F.relu(conv(W, s + "output_conv2.2", F.relu(conv(W, s + "output_conv2.0", o))))
    o = F.interpolate(o, (H_, W_), mode="bilinear", align_corners=True)
    return F.relu(o)[:, 0]


def vda_preprocess(img, size=518):
    """Resize(lower_bound, keep aspect, multiple of 14, INTER_CUBIC) of DAv2's transform."""
    import cv2
    h, w = img.shape[:2]
    ratio = max(h, w) / min(h, w)
    if ratio > 1.78:
        size = round(int(size * 1.777 / ratio) / 14) * 14
    sh, sw = size / h, size / w
    s = max(sh, sw)                                        # lower_bound: scale so both >= size

    def mult(x, minv):
        y = int(np.round(x / 14) * 14)
        if y < minv:
            y = int(np.ceil(x / 14) * 14)
        return y
    nh, nw = mult(s * h, size), mult(s * w, size)
    a = cv2.resize(img.astype(np.float32) / 255.0, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = (a - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1).copy())


def run_vda(path, frames, n):
    W = torch.load(os.path.join(path, "video_depth_anything_vits.pth"), map_location="cpu", weights_only=True)
    vit = ViT(W, "pretrained.", 6, eps=1e-6, final_eps=1e-6)
    x = torch.stack([vda_preprocess(f) for f in frames[:n]])
    t = time.time()
    d = vda_forward(W, vit, x, x.shape[0])
    return dict(depth=d.numpy(), t_all=time.time() - t, shape=tuple(x.shape), tokens=vit.x0.numpy())


def run_vda_video(path, frames):
    """infer_video_depth of the authors (fp32): windows of 32, step 22, the
    keyframes, every window's depth bilinear (align_corners) to the frame size,
    the scale/shift fit and the crossfade at that size."""
    W = torch.load(os.path.join(path, "video_depth_anything_vits.pth"), map_location="cpu", weights_only=True)
    vit = ViT(W, "pretrained.", 6, eps=1e-6, final_eps=1e-6)
    INFER_LEN, OVERLAP, INTERP_LEN = 32, 10, 8
    KEYFRAMES = [0, 12, 24, 25, 26, 27, 28, 29, 30, 31]
    fh, fw = frames[0].shape[:2]
    frame_list = list(frames)
    step = INFER_LEN - OVERLAP
    org = len(frame_list)
    app = (step - (org % step)) % step + (INFER_LEN - step)
    frame_list = frame_list + [frame_list[-1].copy()] * app
    depth_list, pre_input = [], None
    t = time.time()
    for fid in range(0, org, step):
        cur = torch.stack([vda_preprocess(frame_list[fid + i]) for i in range(INFER_LEN)])[None]
        if pre_input is not None:
            cur[:, :OVERLAP] = pre_input[:, KEYFRAMES]
        d = vda_forward(W, vit, cur[0], INFER_LEN)[:, None]
        d = F.interpolate(d, size=(fh, fw), mode="bilinear", align_corners=True)
        depth_list += [d[i][0].numpy() for i in range(d.shape[0])]
        pre_input = cur
    al_len = OVERLAP - INTERP_LEN
    kf_align = KEYFRAMES[:al_len]
    aligned, ref_align = [], []

    def fit(prediction, target):
        prediction, target = prediction.astype(np.float32), target.astype(np.float32)
        mask = np.ones_like(prediction)
        a_00, a_01, a_11 = np.sum(mask * prediction * prediction), np.sum(mask * prediction), np.sum(mask)
        b_0, b_1 = np.sum(mask * prediction * target), np.sum(mask * target)
        det = a_00 * a_11 - a_01 * a_01
        if det == 0:
            return 1, 0
        return (a_11 * b_0 - a_01 * b_1) / det, (-a_01 * b_0 + a_00 * b_1) / det
    for fid in range(0, len(depth_list), INFER_LEN):
        if not aligned:
            aligned += depth_list[:INFER_LEN]
            for kf in kf_align:
                ref_align.append(depth_list[fid + kf])
        else:
            curr = [depth_list[fid + i] for i in range(len(kf_align))]
            scale, shift = fit(np.concatenate(curr), np.concatenate(ref_align))
            pre = aligned[-INTERP_LEN:]
            post = depth_list[fid + al_len:fid + OVERLAP]
            for i in range(len(post)):
                post[i] = post[i] * scale + shift
                post[i][post[i] < 0] = 0
            wl = [0.0] + [i * (1.0 / (INTERP_LEN - 1)) for i in range(1, INTERP_LEN - 1)] + [1.0]
            aligned[-INTERP_LEN:] = [pre[i] * (1 - wl[i]) + post[i] * wl[i] for i in range(INTERP_LEN)]
            for i in range(OVERLAP, INFER_LEN):
                nd = depth_list[fid + i] * scale + shift
                nd[nd < 0] = 0
                aligned.append(nd)
            ref_align = ref_align[:1]
            for kf in kf_align[1:]:
                nd = depth_list[fid + kf] * scale + shift
                nd[nd < 0] = 0
                ref_align.append(nd)
    return dict(depth=np.stack(aligned[:org]), t_all=time.time() - t)


def uv_grid(w, h, ar=None):
    ar = w / h if ar is None else ar
    sx, sy = ar / (1 + ar ** 2) ** 0.5, 1 / (1 + ar ** 2) ** 0.5
    u = np.linspace(-sx * (w - 1) / w, sx * (w - 1) / w, w, dtype=np.float32)
    v = np.linspace(-sy * (h - 1) / h, sy * (h - 1) / h, h, dtype=np.float32)
    return np.stack(np.meshgrid(u, v, indexing="xy"), -1)

def nearest(a, size):
    """torch F.interpolate(mode='nearest') index rule: floor(dst * in / out)."""
    H, W = a.shape[:2]
    ys = np.minimum(np.floor(np.arange(size[0], dtype=np.float32) * np.float32(H / size[0])).astype(np.int64), H - 1)
    xs = np.minimum(np.floor(np.arange(size[1], dtype=np.float32) * np.float32(W / size[1])).astype(np.int64), W - 1)
    return a[ys][:, xs]

def post(z):
    from scipy.optimize import least_squares
    pts, mask = z["points"].astype(np.float32), z["mask"] > 0.5
    H, W = pts.shape[:2]
    ar = W / H
    p_lr, uv_lr, m_lr = nearest(pts, (64, 64)), nearest(uv_grid(W, H), (64, 64)), nearest(mask, (64, 64))
    xyz, uv = p_lr[m_lr], uv_lr[m_lr]
    xy, zz = xyz[:, :2], xyz[:, 2]
    def fn(s):
        p = xy / (zz + s)[:, None]
        f = (p * uv).sum() / np.square(p).sum()
        return (f * p - uv).ravel()
    s = least_squares(fn, x0=0, ftol=1e-3, method="lm")["x"].squeeze().astype(np.float32)
    p = xy / (zz + s)[:, None]
    focal = (p * uv).sum() / np.square(p).sum()
    fx = focal / 2 * (1 + ar ** 2) ** 0.5 / ar          # in widths
    depth = (pts[..., 2] + s) * z["metric_scale"]
    mask &= (pts[..., 2] + s) > 0
    depth = np.where(mask, depth, np.inf).astype(np.float32)
    return depth, mask, float(s), float(fx), float(2 * np.degrees(np.arctan(0.5 / fx)))


def read(p):
    import cv2
    return cv2.cvtColor(cv2.imread(p, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("path")
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--res", type=int, default=504)
    ap.add_argument("--tokens", type=int, default=3600)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("REF_THREADS", "10")))
    if a.model == "da3mono":
        r = run_da3mono(a.path, read(a.inp), a.res)
    elif a.model == "da3":
        r = run_da3(a.path, read(a.inp), a.res)
    elif a.model == "moge":
        r = run_moge(a.path, read(a.inp), a.tokens)
        d, m, s, fx, _fov = post(r)
        r.update(depth=d, valid=m, shift=s, fx=fx)
    elif a.model == "vdavideo":
        fs = sorted(glob.glob(os.path.join(a.inp, "*.png")))[:a.n]
        r = run_vda_video(a.path, [read(f) for f in fs])
    elif a.model == "vda":
        fs = sorted(glob.glob(os.path.join(a.inp, "*.png")))
        r = run_vda(a.path, [read(f) for f in fs], a.n)
    feats = r.pop("feats", None)
    if feats is not None:
        for i, f in enumerate(feats):
            r["feat%d" % i] = f
    np.savez(a.out, **{k: np.asarray(v) for k, v in r.items()})
    print({k: (v if np.isscalar(v) or isinstance(v, tuple) else np.asarray(v).shape) for k, v in r.items()})
