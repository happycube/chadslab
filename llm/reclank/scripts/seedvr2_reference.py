"""SeedVR2-3B references for scripts/check_seedvr2.py: the official one-step restoration
(ByteDance-Seed/SeedVR, projects/inference_seedvr2_3b.py with configs_3b/main.yaml: cfg 1, one
step, sp_size 1) written out in torch float32 on the CPU from its code (Apache 2.0), with no
flash-attn, no apex and none of the official code imported or run. Saves each part on a real image
or video.

Written from SeedVR (ByteDance-Seed/SeedVR; Apache License 2.0; Copyright (c) 2025 Bytedance Ltd.
and/or its affiliates; its VAE also Copyright (c) 2023 HuggingFace Team). See
THIRD_PARTY_NOTICES.md.

    seedvr2_reference.py MODEL_DIR INPUT out.npz [--scale 2 | --res HxW] [--frames N] [--seed 0]
        [--threads 10] [--png out.png] [--taps taps.npz] [--parts all|vae]

MODEL_DIR has seedvr2_ema_3b.pth (or with --variant 7b seedvr2_ema_7b.pth), ema_vae.pth and pos_emb.pt
(--text: elsewhere; the 7B's repository has none), loaded with weights_only=True.
INPUT: an image (PIL) or a video (ffmpeg's rgb24 frames; --frames: the first N).

The official pipeline, and how it is written here:
    the frames in [0, 1] (t, 3, h, w); NaResize(sqrt(res_h res_w), "area", downsample_only False):
        scale = sqrt(res_h res_w / (h w)), the sides round(h scale), round(w scale), resized by
        torchvision's tensor resize (bicubic, antialias True: F.interpolate's antialiased bicubic,
        the same kernel); clamp to [0, 1]; DivisibleCrop 16 (center_crop's offsets
        int(round((h - ch) / 2))); Normalize 0.5: x 2 - 1. A video padded to 4k + 1 frames by its
        last frame (cut_videos).
    the VAE (video_vae_v3, s8_c16_t4_inflation_sd3): 3 x 3 x 3 causal convs whose time padding is
        2 copies of the first frame (extend_head), GroupNorm of 32 per frame, the mid attention per
        frame (diffusers' Attention of one head of 512, its GroupNorm, bias, residual), SD's
        (0, 1, 0, 1) zero padding before the stride-2 downsamplers (down block 0 in space, 1 and 2
        in space and time), the MAGViT upsampler (a 1 x 1 x 1 conv to 8 C or 4 C, the rearrange
        "b (x y z c) f h w -> b c (f z) (h x) (w y)", remove_head of the doubled first frame, a
        causal conv). The whole clip at once (the official slices are causal: the same values).
        The posterior's sample: mean + exp(logvar / 2) noise (logvar clamped to [-30, 20]); the
        latent (t, h, w, 16) times 0.9152.
    the DiT (models/dit_v2 NaDiT; 32 layers, 2560 wide, 20 heads of 128; layers 0-9 their own
        text weights, 10-31 shared; SwiGLU 6912): one call at t = 1000 on [noise, latent, 1]
        (33 channels, 1 x 2 x 2 patches (h w c)) and pos_emb (58 rows) through txt_in; AdaSingle:
        emb (timestep embedding, laid out (d l g)) plus the block's own vectors, x (scale) +
        shift, y gate; the windows of window.py (720pwin_by_size_bysize in even layers, the
        shifted 720pswin in odd ones; (4, 3, 3)); each window a sequence of its video rows and a
        copy of the text rows (SDPA in float32 in place of flash-attn's bfloat16), the text's
        output the mean over the windows; mmrope3d (rotary_embedding_torch's "lang" freqs of the
        checkpoint, interleaved pairs, 126 of 128 values): a video row at (58 + its window frame,
        its window row, its window column), a text row l at (l, l, l). The last layer: the text
        neither modulated nor its MLP (vid_only). The output: RMS norm (affine) times
        (emb's attn scale + out_scale) plus (emb's attn shift + out_shift): the official's Cache
        hands vid_out_ada the key "emb_repeat_0_vid" the first block stored (the attention's
        vectors), then vid_out. x0 = noise - v (v_lerp at t = T). (The official last layer's
        text rows come out as 2 x their attention output: its vid_only modules pass them through
        to the residual twice; here they stop at the attention. They reach nothing.) Checked
        against the official code (sandboxed, float32 on the CPU, its causal VAE slicing and
        memory limits on) on bridge_lq: x the same bits, the posterior rel 3e-7, v 1 - cos 1.6e-11
        (worst token 9.7e-9), the layers' rows rel 1e-7 to 1e-5, the decode 97.4 dB.
    the decode of x0 / 0.9152; the color fix (wavelet_reconstruction: 5 levels of the 3 x 3
        binomial blur, replicate padding, dilations 1 2 4 8 16; the output's high frequencies plus
        the input's low ones; written from its description), clip, uint8 by round.

out.npz: "lq" the input frames (t, h, w, 3) uint8 as decoded; "x" the transformed frames (3, t,
H, W) in [-1, 1] before the padding, "xp" after it; "mean", "logvar" (16, t', h, w) the
posterior; "pnoise" its noise; "latent" (t', h, w, 16) scaled; "noise" the DiT's x_T; "txt" the
text rows (58, 5120, float32 of the bfloat16 file); "v" the DiT's output; "x0"; "dec" the
decoder's frames (t, 3, H, W) in [-1, 1] (cut to the input's frames); "fixed" after the color
fix; "out" (t, H, W, 3) uint8. --taps: the RoPE tables and window rows of both layer kinds, the
rows after vid_in and txt_in, each layer's video and text rows (layers 0, 1, 10, 11, 31), the
output norm's rows. --parts vae: the VAE only (decode of the posterior's mean)."""
import argparse
import math
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("input")
ap.add_argument("out")
ap.add_argument("--scale", type=float, default=2.0)
ap.add_argument("--res", default=None, help="HxW: the official res_h x res_w (its area is used)")
ap.add_argument("--frames", type=int, default=None)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=10)
ap.add_argument("--png")
ap.add_argument("--taps")
ap.add_argument("--parts", default="all")
ap.add_argument("--variant", default="3b", help="3b, or 7b (the official models/dit: dit7b)")
ap.add_argument("--text", default=None, help="pos_emb.pt when MODEL_DIR has none")
a = None            # the arguments (main); the classes and functions import without them


def load(name):
    p = os.path.join(a.model, name)
    sd = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
    return sd


# ---- input --------------------------------------------------------------------------------------

def read_frames(path, frames):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
        from PIL import Image
        return np.asarray(Image.open(path).convert("RGB"))[None].copy()
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height", "-of", "csv=p=0", path], capture_output=True, text=True)
    w, h = map(int, probe.stdout.strip().split(",")[:2])
    cmd = ["ffmpeg", "-v", "error", "-i", path] + (["-frames:v", str(frames)] if frames else []) + \
          ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3).copy()


def transform(lq, res_h, res_w):
    """NaResize (area), clamp, DivisibleCrop 16, Normalize 0.5, (t c h w) -> (c t h w)."""
    x = torch.from_numpy(lq).permute(0, 3, 1, 2).float() / 255.0
    h, w = x.shape[-2:]
    resolution = (res_h * res_w) ** 0.5
    scale = math.sqrt(resolution ** 2 / (h * w))
    rh, rw = round(h * scale), round(w * scale)
    x = F.interpolate(x, size=(rh, rw), mode="bicubic", align_corners=False, antialias=True)
    x = x.clamp(0.0, 1.0)
    ch, cw = rh - rh % 16, rw - rw % 16
    top, left = int(round((rh - ch) / 2.0)), int(round((rw - cw) / 2.0))
    x = x[..., top:top + ch, left:left + cw]
    x = (x - 0.5) / 0.5
    return x.permute(1, 0, 2, 3).contiguous()


def cut_videos(x):
    t = x.size(1)
    if t == 1:
        return x
    if t <= 4:
        return torch.cat([x] + [x[:, -1:]] * (4 - t + 1), 1)
    if (t - 1) % 4 == 0:
        return x
    return torch.cat([x] + [x[:, -1:]] * (4 - (t - 1) % 4), 1)


# ---- the VAE ------------------------------------------------------------------------------------

class VAE:
    def __init__(self, sd):
        self.sd = sd

    def p(self, n):
        return self.sd[n].float()

    def conv(self, x, name, stride=(1, 1, 1), space_pad=True):
        """InflatedCausalConv3d: the time padding (k - 1) / 2 as 2 (k - 1) / 2 copies of the first
        frame, the space padding of the kernel in zeros (none for the downsamplers: their (0, 1,
        0, 1) is padded before)."""
        w, b = self.p(name + ".weight"), self.p(name + ".bias")
        kt, kh, kw = w.shape[2:]
        tp = (kt - 1) // 2
        if tp:
            x = torch.cat([x[:, :, :1]] * (2 * tp) + [x], 2)
        sp = (0, (kh - 1) // 2, (kw - 1) // 2) if space_pad else (0, 0, 0)
        return F.conv3d(x, w, b, stride=stride, padding=sp)

    def gnorm(self, x, name, silu=True):
        t = x.size(2)
        y = x.permute(0, 2, 1, 3, 4).flatten(0, 1)                           # (b t) c h w
        y = F.group_norm(y, 32, self.p(name + ".weight"), self.p(name + ".bias"), 1e-6)
        y = y.unflatten(0, (x.size(0), t)).permute(0, 2, 1, 3, 4)
        return F.silu(y) if silu else y

    def resnet(self, x, p):
        h = self.conv(self.gnorm(x, p + "norm1"), p + "conv1")
        h = self.conv(self.gnorm(h, p + "norm2"), p + "conv2")
        if p + "conv_shortcut.weight" in self.sd:
            x = self.conv(x, p + "conv_shortcut")
        return x + h

    def attn(self, x, p):
        b, c, t, h, w = x.shape
        res = x
        y = x.permute(0, 2, 1, 3, 4).flatten(0, 1)                           # (b t) c h w
        y = F.group_norm(y, 32, self.p(p + "group_norm.weight"), self.p(p + "group_norm.bias"), 1e-6)
        y = y.flatten(2).transpose(1, 2)                                     # (b t) (h w) c
        q = F.linear(y, self.p(p + "to_q.weight"), self.p(p + "to_q.bias"))
        k = F.linear(y, self.p(p + "to_k.weight"), self.p(p + "to_k.bias"))
        v = F.linear(y, self.p(p + "to_v.weight"), self.p(p + "to_v.bias"))
        o = F.scaled_dot_product_attention(q[:, None], k[:, None], v[:, None])[:, 0]
        o = F.linear(o, self.p(p + "to_out.0.weight"), self.p(p + "to_out.0.bias"))
        o = o.transpose(1, 2).reshape(b * t, c, h, w).unflatten(0, (b, t)).permute(0, 2, 1, 3, 4)
        return o + res

    def encode(self, x):
        """x (1, 3, T, H, W) -> (mean, logvar) (1, 16, T', H/8, W/8)."""
        e = "encoder."
        x = self.conv(x, e + "conv_in")
        for i in range(4):
            p = e + "down_blocks.%d." % i
            for j in range(2):
                x = self.resnet(x, p + "resnets.%d." % j)
            if i < 3:
                x = F.pad(x, (0, 1, 0, 1))
                x = self.conv(x, p + "downsamplers.0.conv", stride=(2 if i >= 1 else 1, 2, 2), space_pad=False)
        m = e + "mid_block."
        x = self.resnet(x, m + "resnets.0.")
        x = self.attn(x, m + "attentions.0.")
        x = self.resnet(x, m + "resnets.1.")
        x = self.conv(self.gnorm(x, e + "conv_norm_out"), e + "conv_out")
        mean, logvar = x.chunk(2, dim=1)
        return mean, logvar.clamp(-30.0, 20.0)

    def decode(self, z):
        """z (1, 16, T', h, w) -> (1, 3, T, 8h, 8w)."""
        d = "decoder."
        x = self.conv(z, d + "conv_in")
        m = d + "mid_block."
        x = self.resnet(x, m + "resnets.0.")
        x = self.attn(x, m + "attentions.0.")
        x = self.resnet(x, m + "resnets.1.")
        for i in range(4):
            p = d + "up_blocks.%d." % i
            for j in range(3):
                x = self.resnet(x, p + "resnets.%d." % j)
            if i < 3:
                temporal = i < 2
                u = p + "upsamplers.0."
                x = F.conv3d(x, self.p(u + "upscale_conv.weight"), self.p(u + "upscale_conv.bias"))
                z_ = 2 if temporal else 1
                b, cz, f, h, w = x.shape
                c = cz // (4 * z_)
                x = x.reshape(b, 2, 2, z_, c, f, h, w).permute(0, 4, 5, 3, 6, 1, 7, 2)
                x = x.reshape(b, c, f * z_, h * 2, w * 2)
                if temporal:
                    x = torch.cat([x[:, :, :1], x[:, :, 2:]], 2)            # remove_head
                x = self.conv(x, u + "conv")
        return self.conv(self.gnorm(x, d + "conv_norm_out"), d + "conv_out")


# ---- the DiT ------------------------------------------------------------------------------------

D, HEADS, HD, TXT_LAYERS_MM, NL = 2560, 20, 128, 10, 32


def window_slices(t, h, w, shifted, nwin=(4, 3, 3)):
    """make_720Pwindows_bysize / make_shifted_720Pwindows_bysize of window.py."""
    rnt, rnh, rnw = nwin
    scale = math.sqrt((45 * 80) / (h * w))
    rh, rw = round(h * scale), round(w * scale)
    wh, ww = math.ceil(rh / rnh), math.ceil(rw / rnw)
    wt = math.ceil(min(t, 30) / rnt)
    out = []
    if not shifted:
        nt, nh, nw = math.ceil(t / wt), math.ceil(h / wh), math.ceil(w / ww)
        for iw in range(nw):
            if min((iw + 1) * ww, w) <= iw * ww:
                continue
            for ih in range(nh):
                if min((ih + 1) * wh, h) <= ih * wh:
                    continue
                for it in range(nt):
                    if min((it + 1) * wt, t) <= it * wt:
                        continue
                    out.append((slice(it * wt, min((it + 1) * wt, t)), slice(ih * wh, min((ih + 1) * wh, h)),
                                slice(iw * ww, min((iw + 1) * ww, w))))
        return out
    st, sh, sw = (0.5 if wt < t else 0, 0.5 if wh < h else 0, 0.5 if ww < w else 0)
    nt, nh, nw = math.ceil((t - st) / wt), math.ceil((h - sh) / wh), math.ceil((w - sw) / ww)
    nt, nh, nw = (nt + 1 if st > 0 else 1, nh + 1 if sh > 0 else 1, nw + 1 if sw > 0 else 1)

    def sl(i, s, n, size):
        return slice(max(int((i - s) * n), 0), min(int((i - s + 1) * n), size))
    for iw in range(nw):
        if sl(iw, sw, ww, w).stop <= sl(iw, sw, ww, w).start:
            continue
        for ih in range(nh):
            if sl(ih, sh, wh, h).stop <= sl(ih, sh, wh, h).start:
                continue
            for it in range(nt):
                if sl(it, st, wt, t).stop <= sl(it, st, wt, t).start:
                    continue
                out.append((sl(it, st, wt, t), sl(ih, sh, wh, h), sl(iw, sw, ww, w)))
    return out


def rotate_half(x):
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(-1)
    return torch.stack((-x2, x1), -1).flatten(-2)


def axial(freqs, pos):
    """rotary_embedding_torch's freqs of positions pos (float32): pos x freqs, each repeated
    twice (interleaved)."""
    f = pos.float()[..., None] * freqs
    return f.repeat_interleave(2, -1)


def rope(x, ang):
    """apply_rotary_emb: x (rows, heads, 128), ang (rows, 126) in float32."""
    r = ang.shape[-1]
    c, s = ang.cos()[:, None], ang.sin()[:, None]
    mid = x[..., :r]
    return torch.cat([mid * c + rotate_half(mid) * s, x[..., r:]], -1)


def window_angles(freqs, shp, txt_len):
    """(rows of the window (f h w), 126): frames from txt_len, rows and columns from 0."""
    f, h, w = shp
    gf, gh, gw = torch.meshgrid(torch.arange(f) + txt_len, torch.arange(h), torch.arange(w), indexing="ij")
    return torch.cat([axial(freqs, gf.reshape(-1)), axial(freqs, gh.reshape(-1)), axial(freqs, gw.reshape(-1))], -1)


def timestep_embedding(t, dim=256):
    half = dim // 2
    exponent = -math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half
    e = t[:, None].float() * torch.exp(exponent)[None]
    return torch.cat([torch.sin(e), torch.cos(e)], -1)


def dit(sd, vid, txt, taps=None):
    """vid (t, h, w, 33), txt (L, 5120) -> v (t, h, w, 16)."""
    P = lambda n: sd[n].float()  # noqa: E731
    T, H, W, C = vid.shape
    hh, ww = H // 2, W // 2
    x = vid.reshape(T, hh, 2, ww, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(T * hh * ww, 4 * C)
    x = F.linear(x, P("vid_in.proj.weight"), P("vid_in.proj.bias"))
    y = F.linear(txt, P("txt_in.weight"), P("txt_in.bias"))
    L = y.shape[0]
    N = x.shape[0]
    e = timestep_embedding(torch.tensor([1000.0]))
    e = F.linear(e, P("emb_in.proj_in.weight"), P("emb_in.proj_in.bias"))
    e = F.linear(F.silu(e), P("emb_in.proj_hid.weight"), P("emb_in.proj_hid.bias"))
    e = F.linear(F.silu(e), P("emb_in.proj_out.weight"), P("emb_in.proj_out.bias"))
    emb = e.reshape(D, 2, 3)                      # (d l g): [:, layer, (shift, scale, gate)]
    if taps is not None:
        taps["x_in"], taps["txt_in"], taps["emb"] = x.numpy().copy(), y.numpy().copy(), emb.numpy().copy()
    grid = torch.arange(N).reshape(T, hh, ww)
    wins = [window_slices(T, hh, ww, s) for s in (False, True)]
    for i in range(NL):
        st = "all" if i >= TXT_LAYERS_MM else None
        last = i == NL - 1
        pre = "blocks.%d." % i

        def nm(stream, rest):
            mod, field = rest
            return pre + "%s.%s.%s" % (mod, st or stream, field)

        def ada(hid, stream, layer, mode):
            li = 0 if layer == "attn" else 1
            if mode == "in":
                sc = emb[:, li, 1] + P(nm(stream, ("ada", layer + "_scale")))
                sh = emb[:, li, 0] + P(nm(stream, ("ada", layer + "_shift")))
                return hid * sc + sh
            return hid * (emb[:, li, 2] + P(nm(stream, ("ada", layer + "_gate"))))

        hv = F.rms_norm(x, (D,), None, 1e-5)
        ht = F.rms_norm(y, (D,), None, 1e-5)
        hv = ada(hv, "vid", "attn", "in")
        if not last:
            ht = ada(ht, "txt", "attn", "in")
        qkv_v = F.linear(hv, P(pre + "attn.proj_qkv.%s.weight" % (st or "vid")))
        qkv_t = F.linear(ht, P(pre + "attn.proj_qkv.%s.weight" % (st or "txt")))
        qv, kv, vv = qkv_v.reshape(N, 3, HEADS, HD).unbind(1)
        qt, kt, vt = qkv_t.reshape(L, 3, HEADS, HD).unbind(1)
        nq = lambda s: P(pre + "attn.norm_q.%s.weight" % (st or s))  # noqa: E731
        nk = lambda s: P(pre + "attn.norm_k.%s.weight" % (st or s))  # noqa: E731
        qv, kv = F.rms_norm(qv, (HD,), nq("vid"), 1e-5), F.rms_norm(kv, (HD,), nk("vid"), 1e-5)
        qt, kt = F.rms_norm(qt, (HD,), nq("txt"), 1e-5), F.rms_norm(kt, (HD,), nk("txt"), 1e-5)
        freqs = P(pre + "attn.rope.rope.freqs")
        tang = axial(freqs, torch.arange(L)).repeat(1, 3)
        qt_r, kt_r = rope(qt, tang), rope(kt, tang)
        ov = torch.zeros(N, HEADS, HD)
        ot = torch.zeros(L, HEADS, HD)
        ws = wins[i % 2]
        for s in ws:
            idx = grid[s].reshape(-1)
            ang = window_angles(freqs, grid[s].shape, L)
            q = torch.cat([rope(qv[idx], ang), qt_r]).transpose(0, 1)
            k = torch.cat([rope(kv[idx], ang), kt_r]).transpose(0, 1)
            v = torch.cat([vv[idx], vt]).transpose(0, 1)
            o = F.scaled_dot_product_attention(q, k, v).transpose(0, 1)
            ov[idx] = o[:idx.numel()]
            ot += o[idx.numel():]
        ot = ot / len(ws)
        if taps is not None and i == 0:
            taps["tang"] = tang.numpy().copy()
        av = F.linear(ov.reshape(N, D), P(pre + "attn.proj_out.%s.weight" % (st or "vid")),
                      P(pre + "attn.proj_out.%s.bias" % (st or "vid")))
        at = F.linear(ot.reshape(L, D), P(pre + "attn.proj_out.%s.weight" % (st or "txt")),
                      P(pre + "attn.proj_out.%s.bias" % (st or "txt")))
        x = ada(av, "vid", "attn", "out") + x
        y = (at if last else ada(at, "txt", "attn", "out")) + y

        def mlp(hid, stream):
            s = st or stream
            g = F.linear(hid, P(pre + "mlp.%s.proj_in_gate.weight" % s))
            u = F.linear(hid, P(pre + "mlp.%s.proj_in.weight" % s))
            return F.linear(F.silu(g) * u, P(pre + "mlp.%s.proj_out.weight" % s))

        hv = ada(F.rms_norm(x, (D,), None, 1e-5), "vid", "mlp", "in")
        x = ada(mlp(hv, "vid"), "vid", "mlp", "out") + x
        if not last:
            ht = ada(F.rms_norm(y, (D,), None, 1e-5), "txt", "mlp", "in")
            y = ada(mlp(ht, "txt"), "txt", "mlp", "out") + y
        if taps is not None and i in (0, 1, 10, 11, 31):
            taps["vid%d" % i], taps["txt%d" % i] = x.numpy().copy(), y.numpy().copy()
    hv = F.rms_norm(x, (D,), P("vid_out_norm.weight"), 1e-5)
    hv = hv * (emb[:, 0, 1] + P("vid_out_ada.out_scale")) + (emb[:, 0, 0] + P("vid_out_ada.out_shift"))
    if taps is not None:
        taps["out_norm"] = hv.numpy().copy()
    o = F.linear(hv, P("vid_out.proj.weight"), P("vid_out.proj.bias"))
    return o.reshape(T, hh, ww, 2, 2, 16).permute(0, 1, 3, 2, 4, 5).reshape(T, H, W, 16)


D7, HEADS7, NL7 = 3072, 24, 36


def dit7b(sd, vid, txt):
    """The 7B (official models/dit NaDiT: 36 layers, 3072 wide, 24 heads of 128, every layer its own
    text weights, GELU (tanh) MLP with biases, the "pixel" RoPE of the video rows only:
    rotary_embedding_torch with linspace(-1, 1, n) positions along each axis of the window and the
    checkpoint's 10 freqs an axis, 60 of the 128 values; no output norm; its Cache off, so no
    output ada): vid (t, h, w, 33), txt (L, 5120) -> v (t, h, w, 16)."""
    P = lambda n: sd[n].float()  # noqa: E731
    T, H, W, C = vid.shape
    hh, ww = H // 2, W // 2
    x = vid.reshape(T, hh, 2, ww, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(T * hh * ww, 4 * C)
    x = F.linear(x, P("vid_in.proj.weight"), P("vid_in.proj.bias"))
    y = F.linear(txt, P("txt_in.weight"), P("txt_in.bias"))
    L, N = y.shape[0], x.shape[0]
    e = timestep_embedding(torch.tensor([1000.0]))
    e = F.linear(e, P("emb_in.proj_in.weight"), P("emb_in.proj_in.bias"))
    e = F.linear(F.silu(e), P("emb_in.proj_hid.weight"), P("emb_in.proj_hid.bias"))
    e = F.linear(F.silu(e), P("emb_in.proj_out.weight"), P("emb_in.proj_out.bias"))
    emb = e.reshape(D7, 2, 3)
    grid = torch.arange(N).reshape(T, hh, ww)
    wins = [window_slices(T, hh, ww, s) for s in (False, True)]
    freqs = P("blocks.0.attn.rope.rope.freqs")

    def angles(shp):
        g = torch.meshgrid(*[torch.linspace(-1, 1, steps=n) for n in shp], indexing="ij")
        return torch.cat([axial(freqs, gi.reshape(-1)) for gi in g], -1)
    for i in range(NL7):
        pre = "blocks.%d." % i

        def ada(hid, st, layer, mode):
            li = 0 if layer == "attn" else 1
            if mode == "in":
                return hid * (emb[:, li, 1] + P(pre + "ada.%s.%s_scale" % (st, layer))) + \
                    (emb[:, li, 0] + P(pre + "ada.%s.%s_shift" % (st, layer)))
            return hid * (emb[:, li, 2] + P(pre + "ada.%s.%s_gate" % (st, layer)))
        hv = ada(F.rms_norm(x, (D7,), None, 1e-5), "vid", "attn", "in")
        ht = ada(F.rms_norm(y, (D7,), None, 1e-5), "txt", "attn", "in")
        qv, kv, vv = F.linear(hv, P(pre + "attn.proj_qkv.vid.weight")).reshape(N, 3, HEADS7, HD).unbind(1)
        qt, kt, vt = F.linear(ht, P(pre + "attn.proj_qkv.txt.weight")).reshape(L, 3, HEADS7, HD).unbind(1)
        qv = F.rms_norm(qv, (HD,), P(pre + "attn.norm_q.vid.weight"), 1e-5)
        kv = F.rms_norm(kv, (HD,), P(pre + "attn.norm_k.vid.weight"), 1e-5)
        qt = F.rms_norm(qt, (HD,), P(pre + "attn.norm_q.txt.weight"), 1e-5)
        kt = F.rms_norm(kt, (HD,), P(pre + "attn.norm_k.txt.weight"), 1e-5)
        ov, ot = torch.zeros(N, HEADS7, HD), torch.zeros(L, HEADS7, HD)
        ws = wins[i % 2]
        for s in ws:
            idx = grid[s].reshape(-1)
            ang = angles(grid[s].shape)
            q = torch.cat([rope(qv[idx], ang), qt]).transpose(0, 1)
            k = torch.cat([rope(kv[idx], ang), kt]).transpose(0, 1)
            v = torch.cat([vv[idx], vt]).transpose(0, 1)
            o = F.scaled_dot_product_attention(q, k, v).transpose(0, 1)
            ov[idx] = o[:idx.numel()]
            ot += o[idx.numel():]
        ot = ot / len(ws)
        av = F.linear(ov.reshape(N, D7), P(pre + "attn.proj_out.vid.weight"), P(pre + "attn.proj_out.vid.bias"))
        at = F.linear(ot.reshape(L, D7), P(pre + "attn.proj_out.txt.weight"), P(pre + "attn.proj_out.txt.bias"))
        x = ada(av, "vid", "attn", "out") + x
        y = ada(at, "txt", "attn", "out") + y

        def mlp(hid, st):
            h = F.linear(hid, P(pre + "mlp.%s.proj_in.weight" % st), P(pre + "mlp.%s.proj_in.bias" % st))
            return F.linear(F.gelu(h, approximate="tanh"), P(pre + "mlp.%s.proj_out.weight" % st),
                            P(pre + "mlp.%s.proj_out.bias" % st))
        x = ada(mlp(ada(F.rms_norm(x, (D7,), None, 1e-5), "vid", "mlp", "in"), "vid"), "vid", "mlp", "out") + x
        y = ada(mlp(ada(F.rms_norm(y, (D7,), None, 1e-5), "txt", "mlp", "in"), "txt"), "txt", "mlp", "out") + y
    o = F.linear(x, P("vid_out.proj.weight"), P("vid_out.proj.bias"))
    return o.reshape(T, hh, ww, 2, 2, 16).permute(0, 1, 3, 2, 4, 5).reshape(T, H, W, 16)


# ---- the color fix ------------------------------------------------------------------------------

def wavelet_blur(img, radius):
    k = torch.tensor([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]], dtype=img.dtype)
    k = k[None, None].repeat(3, 1, 1, 1)
    img = F.pad(img, (radius, radius, radius, radius), mode="replicate")
    return F.conv2d(img, k, groups=3, dilation=radius)


def wavelet_decomposition(img, levels=5):
    high = torch.zeros_like(img)
    for i in range(levels):
        low = wavelet_blur(img, 2 ** i)
        high += img - low
        img = low
    return high, low


def wavelet_reconstruction(content, style):
    ch, _ = wavelet_decomposition(content)
    _, sl = wavelet_decomposition(style)
    return ch + sl


# ---- the pipeline -------------------------------------------------------------------------------

def main():
    global a
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    res = {}
    taps = {} if a.taps else None
    lq = read_frames(a.input, a.frames)
    t_in = lq.shape[0]
    if a.res:
        rh, rw = map(int, a.res.lower().split("x"))
    else:
        rh, rw = round(lq.shape[1] * a.scale), round(lq.shape[2] * a.scale)
    res["lq"], res["res"] = lq, np.array([rh, rw])
    with torch.no_grad():
        x = transform(lq, rh, rw)
        xp = cut_videos(x)
        res["x"], res["xp"] = x.numpy(), xp.numpy()
        print("input %s -> %s (padded %d frames)" % (lq.shape, tuple(x.shape), xp.shape[1]), flush=True)
        vae = VAE(load("ema_vae.pth"))
        gen = torch.Generator().manual_seed(a.seed)
        t0 = time.time()
        mean, logvar = vae.encode(xp[None])
        print("encode %.1f s" % (time.time() - t0), flush=True)
        pnoise = torch.randn(mean.shape, generator=gen)
        z = mean + torch.exp(0.5 * logvar) * pnoise
        res["mean"], res["logvar"], res["pnoise"] = mean[0].numpy(), logvar[0].numpy(), pnoise[0].numpy()
        latent = z[0].permute(1, 2, 3, 0) * 0.9152                             # (t', h, w, 16)
        res["latent"] = latent.numpy()
        if a.parts == "vae":
            t0 = time.time()
            dec = vae.decode(mean)
            print("decode %.1f s" % (time.time() - t0), flush=True)
            res["dec_mean"] = dec[0].permute(1, 0, 2, 3)[:t_in].numpy()
            np.savez(a.out, **res)
            return
        noise = torch.randn(latent.shape, generator=gen)
        res["noise"] = noise.numpy()
        cond = torch.cat([latent, torch.ones(latent.shape[:-1] + (1,))], -1)
        sd = load("seedvr2_ema_%s.pth" % a.variant)
        txt = torch.load(a.text or os.path.join(a.model, "pos_emb.pt"), map_location="cpu", weights_only=True).float()
        res["txt"] = txt.numpy()
        t0 = time.time()
        v = (dit if a.variant == "3b" else lambda s_, v_, t_, _taps: dit7b(s_, v_, t_))(
            sd, torch.cat([noise, cond], -1), txt, taps)
        print("dit %.1f s" % (time.time() - t0), flush=True)
        x0 = noise - v
        res["v"], res["x0"] = v.numpy(), x0.numpy()
        t0 = time.time()
        dec = vae.decode((x0 / 0.9152).permute(3, 0, 1, 2)[None])
        print("decode %.1f s" % (time.time() - t0), flush=True)
        sample = dec[0].permute(1, 0, 2, 3)[:t_in]                                # (t, 3, H, W)
        res["dec"] = sample.numpy()
        inp = x.permute(1, 0, 2, 3)[:sample.size(0)]
        fixed = wavelet_reconstruction(sample, inp)
        res["fixed"] = fixed.numpy()
        out = fixed.clip(-1, 1).mul(0.5).add(0.5).mul(255).round().to(torch.uint8)
        res["out"] = out.permute(0, 2, 3, 1).numpy()
    np.savez(a.out, **res)
    if taps is not None:
        for kind in (0, 1):
            T_, hh, ww = latent.shape[0], latent.shape[1] // 2, latent.shape[2] // 2
            ws = window_slices(T_, hh, ww, bool(kind))
            grid = np.arange(T_ * hh * ww).reshape(T_, hh, ww)
            taps["win%d_rows" % kind] = np.concatenate([grid[s].reshape(-1) for s in ws])
            taps["win%d_shapes" % kind] = np.array([grid[s].shape for s in ws])
        np.savez(a.taps, **taps)
    if a.png:
        from PIL import Image
        Image.fromarray(res["out"][0]).save(a.png)
    print("saved", a.out)


if __name__ == "__main__":
    main()
