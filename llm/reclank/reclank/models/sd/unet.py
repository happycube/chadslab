"""Stable Diffusion's UNet (diffusers' UNet2DConditionModel of SD 1.5 and 2.0:
blocks of 320 640 1280 1280 channels, 2 layers a down block, cross attention
in the first three) as a program of one step: the model's output of the
latents of a batch (classifier-free guidance: the negative prompt's row and
the prompt's) at a timestep.

Ported from diffusers (src/diffusers/models/unets/unet_2d_condition.py; Apache License 2.0;
Copyright 2026 The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    the step's vectors, on the host in float32: the timestep's sinusoid (320,
        [cos, sin]), time_embedding (linear, SiLU, linear: temb), and each
        ResNet block's time_emb_proj(SiLU(temb))
    conv_in (3 x 3, 4 -> 320)
    the down blocks: ResNet blocks (GroupNorm and SiLU, conv, + the step's
        vector, GroupNorm and SiLU, conv, + x or a 1 x 1 conv of x), each with
        a Transformer2D after it in the first three; a 3 x 3 conv of stride 2
        after the first three; each output kept for the up blocks
    the mid block: ResNet, Transformer2D, ResNet
    the up blocks: 3 ResNet blocks of the input and a kept output side by
        side (torch.cat of the channels), each with a Transformer2D after it
        in the last three; a 2 x nearest upsampling and a 3 x 3 conv after the
        first three
    GroupNorm and SiLU, conv_out (3 x 3, 320 -> 4)

A Transformer2D: GroupNorm (eps 1e-6), proj_in (a 1 x 1 conv in SD 1.5, a
linear in SD 2.0: the same product), a BasicTransformerBlock (LayerNorm, the
attention of the image's pixels over its pixels; LayerNorm, the attention of
its pixels over its prompt's 77 rows (cross_attention_dim); LayerNorm, GEGLU
(the halves of ff.net.0.proj: hidden and gate, hidden gelu(gate), erf),
ff.net.2; each + x), proj_out, + x. The heads: SD 1.5's attention_head_dim 8
is the heads (of 40, 80, 160), SD 2.0's [5, 10, 20, 20] the heads of 64.

The rows are pixels (row major) of each image of the batch, the columns
channels. A GroupNorm is one an image (ENC_GNORM of its rows); the attention
of an image's rows over its own (ENC_ATTN_RANGE: the ranges of the rows; on
the GPU ENC_ATTN_X2 for heads of at most 128; the cross attention's queries
and keys in buffers of the same rows); q times hd^-1/2. A 3 x 3 conv is the im2col of its 9
neighbours by ENC_GATHER (the stride 2 and the upsampling in the index), in
tiles of TILE rows, then ENC_LINEAR of W (out, 3, 3, in). The side by side of
the up blocks: the input written at the start of a stage buffer, the kept
output copied after it, an ENC_GATHER of their rows of 320 channels. The
matrices float32 as stored (ENC_LINEAR), on the CPU and the GPU.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.encoders.runtime import _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.flux2.dit import timestep_sinusoid
from reclank.models.flux2.vae import conv_index
from reclank.runtime.threads import team_threads

TILE = 8192         # rows of an im2col at a time
TEXT = 77


def _silu(x):
    return (x / (1.0 + np.exp(-x))).astype(np.float32)


def down_index(h, w):
    """The im2col index (h/2 w/2, 9) of a 3 x 3 conv of stride 2 with padding
    1 over an image of h x w pixels; -1 past the edge."""
    H, W = h // 2, w // 2
    y = 2 * np.arange(H)[:, None, None, None] + np.arange(-1, 2)[None, None, :, None]
    x = 2 * np.arange(W)[None, :, None, None] + np.arange(-1, 2)[None, None, None, :]
    ok = (y >= 0) & (y < h) & (x >= 0) & (x < w)
    return np.where(ok, y * w + x, -1).reshape(H * W, 9).astype(np.int32)


def batch_index(idx, B, rows_in):
    """An index of one image for B images (the rows of image b after b
    rows_in of the input)."""
    return np.concatenate([np.where(idx >= 0, idx + b * rows_in, -1) for b in range(B)]).astype(np.int32)


class SDUNet:
    """The UNet of a unet directory (config.json and its safetensors).
    output(x, t, ctx) runs a step."""

    def __init__(self, path, gpu=False, cpu_threads=None, bf12=False, f20=False):
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert cfg["down_block_types"] == ["CrossAttnDownBlock2D"] * 3 + ["DownBlock2D"]
        assert cfg["up_block_types"] == ["UpBlock2D"] + ["CrossAttnUpBlock2D"] * 3
        assert cfg.get("flip_sin_to_cos", True) and cfg.get("freq_shift", 0) == 0
        self.W = W = SafeTensors(os.path.join(path, "diffusion_pytorch_model.safetensors"))
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.lins = []                      # every conv and linear of the programs (bf12, f20)
        # f20: each conv and linear to F20 as it is made (its float32 W and the checkpoint's pages
        # of it dropped then): the float32 W of one matrix at a time in memory
        if f20:
            from reclank.encoders.runtime import F20Lins
            self.f20 = F20Lins(self.gpu, "sd unet")
        else:
            self.f20 = None
        self.chans = list(cfg["block_out_channels"])
        self.lpb = cfg["layers_per_block"]
        self.groups = cfg["norm_num_groups"]
        self.eps = float(cfg["norm_eps"])
        self.in_ch, self.out_ch = cfg["in_channels"], cfg["out_channels"]
        self.ctx_dim = cfg["cross_attention_dim"]
        heads = cfg.get("num_attention_heads") or cfg["attention_head_dim"]
        self.heads = list(heads) if isinstance(heads, (list, tuple)) else [heads] * len(self.chans)
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        self.f = f
        self.t1, self.t1b = f("time_embedding.linear_1.weight"), f("time_embedding.linear_1.bias")
        self.t2, self.t2b = f("time_embedding.linear_2.weight"), f("time_embedding.linear_2.bias")
        self.tdim = self.t1.shape[1]
        self.cin = self.conv("conv_in")
        self.temb = []                      # each ResNet block's (W, b) of time_emb_proj, in program order
        self.down = []
        for i, ch in enumerate(self.chans):
            p = "down_blocks.%d." % i
            attn = i < len(self.chans) - 1
            self.down.append(dict(
                res=[self.resnet(p + "resnets.%d." % j) for j in range(self.lpb)],
                attn=[self.transformer(p + "attentions.%d." % j) for j in range(self.lpb)] if attn else None,
                down=self.conv(p + "downsamplers.0.conv") if attn else None))
        m = "mid_block."
        self.mid = dict(r0=self.resnet(m + "resnets.0."), a=self.transformer(m + "attentions.0."),
                        r1=self.resnet(m + "resnets.1."))
        self.up = []
        for i in range(len(self.chans)):
            p = "up_blocks.%d." % i
            attn = i > 0
            ups = p + "upsamplers.0.conv"
            self.up.append(dict(
                res=[self.resnet(p + "resnets.%d." % j) for j in range(self.lpb + 1)],
                attn=[self.transformer(p + "attentions.%d." % j) for j in range(self.lpb + 1)] if attn else None,
                up=self.conv(ups) if ups + ".weight" in W.header else None))
        self.nout = (f("conv_norm_out.weight"), f("conv_norm_out.bias"))
        self.cout = self.conv("conv_out")
        if self.gpu and bf12:
            # BF12 rows of the float32 W rounded to bfloat16 (a change of the weights: by choice,
            # measured in check_sd.py --bf12); a matrix the rule refuses stays float32
            from reclank.models.music.text import gpu_bf12
            gpu_bf12(self.lins, "sd unet")
        elif f20:
            # F20: the float32 W to 16 significant bits (a change of the weights, by choice; 63% of
            # their memory, on the GPU and the CPU); a matrix the rule refuses stays float32
            self.f20.report()
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)

    def conv(self, name, bias=True, rows=None):
        """A conv (out, in, k, k) or a linear (out, in) as a linear of its
        im2col: W (out, k k in) float32, the bias. rows: a slice of W's rows
        (and of the bias)."""
        raw = src = self.W.get(name + ".weight")
        if raw.ndim == 4:
            raw = raw.transpose(0, 2, 3, 1).reshape(raw.shape[0], -1)
        rows = slice(None) if rows is None else rows
        lin = _Plain(np.ascontiguousarray(raw[rows], dtype=np.float32))
        lin.shape = lin.w.shape
        lin.b = np.ascontiguousarray(self.W.get(name + ".bias")[rows], dtype=np.float32) \
            if bias and name + ".bias" in self.W.header else None
        del raw
        if self.f20 is not None:
            self.f20.add(lin, src)
        self.lins.append(lin)
        return lin

    def resnet(self, p):
        f = self.f
        self.temb.append((f(p + "time_emb_proj.weight"), f(p + "time_emb_proj.bias")))
        return dict(n1=(f(p + "norm1.weight"), f(p + "norm1.bias")), c1=self.conv(p + "conv1"),
                    n2=(f(p + "norm2.weight"), f(p + "norm2.bias")), c2=self.conv(p + "conv2"),
                    sc=self.conv(p + "conv_shortcut") if p + "conv_shortcut.weight" in self.W.header else None,
                    tv="_tv%d" % (len(self.temb) - 1))

    def transformer(self, p):
        f = self.f
        b = p + "transformer_blocks.0."
        assert p + "transformer_blocks.1.norm1.weight" not in self.W.header
        inner = self.W.shape(b + "ff.net.0.proj.weight")[0] // 2
        return dict(gn=(f(p + "norm.weight"), f(p + "norm.bias")), pin=self.conv(p + "proj_in"),
                    pout=self.conv(p + "proj_out"),
                    ln1=(f(b + "norm1.weight"), f(b + "norm1.bias")),
                    ln2=(f(b + "norm2.weight"), f(b + "norm2.bias")),
                    ln3=(f(b + "norm3.weight"), f(b + "norm3.bias")),
                    q1=self.conv(b + "attn1.to_q"), k1=self.conv(b + "attn1.to_k"), v1=self.conv(b + "attn1.to_v"),
                    o1=self.conv(b + "attn1.to_out.0"),
                    q2=self.conv(b + "attn2.to_q"), k2=self.conv(b + "attn2.to_k"), v2=self.conv(b + "attn2.to_v"),
                    o2=self.conv(b + "attn2.to_out.0"),
                    fh=self.conv(b + "ff.net.0.proj", rows=slice(0, inner)),
                    fg=self.conv(b + "ff.net.0.proj", rows=slice(inner, 2 * inner)),
                    fo=self.conv(b + "ff.net.2"), inner=inner)

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    # ---- the step's vectors (host) ----------------------------------------

    def vectors(self, t):
        """The ResNet blocks' vectors (name: (channels,)) of timestep t."""
        e = self.t1 @ timestep_sinusoid(t, self.tdim) + self.t1b
        temb = (self.t2 @ _silu(e.astype(np.float32)) + self.t2b).astype(np.float32)
        s = _silu(temb)
        return {"_tv%d" % i: (w @ s + b).astype(np.float32) for i, (w, b) in enumerate(self.temb)}

    # ---- the program -------------------------------------------------------

    def program(self, B, h, w, gpu=False):
        """The program of a step of B images of latents h x w: inputs "x" (B h
        w, in channels), "ctx" (B 77, cross_attention_dim), the vectors of
        vectors(); output "out" (B h w, out channels)."""
        assert h % 8 == 0 and w % 8 == 0
        c = _EncCompiler(self.eps, pack=not gpu, rows=TILE, kmax=9 * 2 * max(self.chans))
        env = c.env
        G = self.groups
        nlev = len(self.chans)
        levels = [(h >> i, w >> i) for i in range(nlev)]
        npix = [hh * ww for hh, ww in levels]
        env["x"] = np.zeros((B * npix[0], self.in_ch), np.float32)
        env["ctx"] = np.zeros((B * TEXT, self.ctx_dim), np.float32)
        env["out"] = np.zeros((B * npix[0], self.out_ch), np.float32)
        for k_, v_ in self.vectors(1.0).items():
            env[k_] = np.zeros_like(v_)
        for li, n in enumerate(npix):
            # the ranges of an image's rows over its own rows, and over its prompt's
            env["_slo%d" % li] = np.repeat(np.arange(B) * n, n).astype(np.int32)
            env["_shi%d" % li] = env["_slo%d" % li] + np.int32(n)
            # (as many as the cross attention's rows: the last image's past its queries)
            lo = np.repeat(np.arange(B) * TEXT, n).astype(np.int32)
            env["_clo%d" % li] = np.concatenate([lo, np.full(max(0, B * (TEXT - n)), lo[-1], np.int32)])
            env["_chi%d" % li] = env["_clo%d" % li] + np.int32(TEXT)
        uses = {}           # name: (pool key, offset, rows, cols)
        n_names = [0]

        def name():
            n_names[0] += 1
            return "_v%d" % n_names[0]

        def buf(li, role, cols, rows=None, off=0):
            """a view (rows of level li, cols) at off of the pool (li, role)"""
            nm = name()
            uses[nm] = ((li, role), off, B * npix[li] if rows is None else rows, cols)
            return nm

        def sub(of, r0, r1):
            """a view of the rows [r0, r1) of a name"""
            nm = name()
            if of in uses:
                key, off, rows, cols = uses[of]
                uses[nm] = (key, off + r0 * cols, r1 - r0, cols)
            else:
                env[nm] = env[of][r0:r1]
            return nm

        def cols_of(nm):
            return uses[nm][3] if nm in uses else env[nm].shape[1]

        forms = []

        def gnorm(x, li, wb, eps, silu, out):
            """GroupNorm of each image's rows"""
            n = npix[li]
            for b in range(B):
                forms.append(("set", sub(out, b * n, (b + 1) * n),
                              ("enc_gnorm", sub(x, b * n, (b + 1) * n), wb[0], wb[1], G, eps, silu)))

        def conv3(x, cin, idx, lin, out):
            """a 3 x 3 conv of x (cin channels) by the index (rows, 9) into out"""
            rows = idx.shape[0]
            for r0 in range(0, rows, TILE):
                r1 = min(rows, r0 + TILE)
                col = buf(-1, "col", 9 * cin, rows=r1 - r0)
                forms.extend([("set", col, ("enc_gather", x, np.ascontiguousarray(idx[r0:r1]), cin)),
                              ("set", sub(out, r0, r1), ("enc_linear", lin, col))])

        def conv_same(x, li, cin, lin, out):
            hh, ww = levels[li]
            conv3(x, cin, batch_index(conv_index(hh, ww), B, npix[li]), lin, out)

        def resnet(x, li, cin, cout, R, free, out_role):
            """a ResNet block of x (level li); free: two roles to use"""
            ra, rb = free
            n1 = buf(li, ra, cin)
            gnorm(x, li, R["n1"], self.eps, True, n1)
            h1 = buf(li, rb, cout)
            conv_same(n1, li, cin, R["c1"], h1)
            forms.append(("enc_bias", h1, R["tv"]))
            n2 = buf(li, ra, cout)
            gnorm(h1, li, R["n2"], self.eps, True, n2)
            h2 = buf(li, out_role, cout)
            conv_same(n2, li, cout, R["c2"], h2)
            if R["sc"] is not None:
                s = buf(li, ra, cout)
                forms.append(("set", s, ("enc_linear", R["sc"], x)))
                forms.append(("enc_add", h2, s))
            else:
                forms.append(("enc_add", h2, x))
            return h2

        def attention(q, k, v, li, heads, hd, lo, hi, out):
            forms.append(("set", out, ("enc_attn_range_x2", q, k, v, lo, hi, heads, hd)))

        def transformer(x, li, C, A, free, out_role):
            """a Transformer2D of x (level li, C channels); free: two roles"""
            nonlocal forms
            ra, rb = free
            heads = A["heads"]
            hd = C // heads
            R = B * npix[li]
            sc = "_sc%d" % C
            env[sc] = np.full(C, hd ** -0.5, np.float32)
            g = buf(li, ra, C)
            gnorm(x, li, A["gn"], 1e-6, False, g)
            hs = buf(li, rb, C)
            forms.append(("set", hs, ("enc_linear", A["pin"], g)))
            n = buf(li, ra, C)
            q, k, v, o, t = (buf(li, r, C) for r in ("q", "k", "v", "o", "t"))
            forms += [("set", n, ("enc_lnorm", hs, A["ln1"][0], A["ln1"][1], C, 1e-5)),
                      ("set", q, ("enc_linear", A["q1"], n)),
                      ("set", q, ("enc_mul_vec", q, sc)),
                      ("set", k, ("enc_linear", A["k1"], n)),
                      ("set", v, ("enc_linear", A["v1"], n))]
            attention(q, k, v, li, heads, hd, "_slo%d" % li, "_shi%d" % li, o)
            forms += [("set", t, ("enc_linear", A["o1"], o)),
                      ("enc_add", hs, t),
                      ("set", n, ("enc_lnorm", hs, A["ln2"][0], A["ln2"][1], C, 1e-5))]
            # the cross attention's queries and keys in buffers of the same rows, the more of the
            # two (ENC_ATTN_X2 converts as many rows of k as of q, ENC_ATTN_RANGE on the CPU reads as
            # many keys as queries); the rows past the queries' are zeros and not read
            kr = max(R, B * TEXT)
            qc, oc = buf(li, "q", C, rows=kr), buf(li, "o", C, rows=kr)
            kc, vc = buf(li, "kc", C, rows=kr), buf(li, "vc", C, rows=kr)
            q, o = sub(qc, 0, R), sub(oc, 0, R)
            forms += [("set", q, ("enc_linear", A["q2"], n)),
                      ("set", q, ("enc_mul_vec", q, sc)),
                      ("set", sub(kc, 0, B * TEXT), ("enc_linear", A["k2"], "ctx")),
                      ("set", sub(vc, 0, B * TEXT), ("enc_linear", A["v2"], "ctx"))]
            attention(qc, kc, vc, li, heads, hd, "_clo%d" % li, "_chi%d" % li, oc)
            fh, fg = buf(li, "fh", A["inner"]), buf(li, "fg", A["inner"])
            forms += [("set", t, ("enc_linear", A["o2"], o)),
                      ("enc_add", hs, t),
                      ("set", n, ("enc_lnorm", hs, A["ln3"][0], A["ln3"][1], C, 1e-5)),
                      ("set", fh, ("enc_linear", A["fh"], n)),
                      ("set", fg, ("enc_linear", A["fg"], n)),
                      ("set", fh, ("enc_gelu_mul", fg, fh, True)),
                      ("set", t, ("enc_linear", A["fo"], fh)),
                      ("enc_add", hs, t)]
            out = buf(li, out_role, C)
            forms += [("set", out, ("enc_linear", A["pout"], hs)),
                      ("enc_add", out, x)]
            return out

        for i, D in enumerate(self.down):
            if D["attn"] is not None:
                for A in D["attn"]:
                    A["heads"] = self.heads[i]
        self.mid["a"]["heads"] = self.heads[-1]
        rheads = list(reversed(self.heads))
        for i, U in enumerate(self.up):
            if U["attn"] is not None:
                for A in U["attn"]:
                    A["heads"] = rheads[i]

        # conv_in, the down blocks (each output kept: the roles "s<j>")
        skips = []
        C = self.chans[0]
        x = buf(0, "s0", C)
        conv_same("x", 0, self.in_ch, self.cin, x)
        skips.append((x, 0, C))
        cin = C
        for i, D in enumerate(self.down):
            cout = self.chans[i]
            for j in range(self.lpb):
                last = D["attn"] is None
                role = "s%d" % len(skips)
                x = resnet(x, i, cin, cout, D["res"][j], ("a", "b"), role if last else "c")
                if not last:
                    x = transformer(x, i, cout, D["attn"][j], ("a", "b"), role)
                skips.append((x, i, cout))
                cin = cout
            if D["down"] is not None:
                hh, ww = levels[i]
                y = buf(i + 1, "s%d" % len(skips), cout)
                conv3(x, cout, batch_index(down_index(hh, ww), B, npix[i]), D["down"], y)
                x = y
                skips.append((x, i + 1, cout))
        # the mid block
        li = nlev - 1
        M = self.mid
        x = resnet(x, li, cin, cin, M["r0"], ("a", "b"), "c")
        x = transformer(x, li, cin, M["a"], ("a", "b"), "d")
        x = resnet(x, li, cin, cin, M["r1"], ("a", "b"), "c")
        # the up blocks: x at the start of the stage, the kept output after it, side by side by a gather
        unit = math.gcd(*self.chans)
        rchans = list(reversed(self.chans))
        for i, U in enumerate(self.up):
            li = nlev - 1 - i
            cout = rchans[i]
            n = B * npix[li]
            for j in range(self.lpb + 1):
                s, sl, cs = skips.pop()
                assert sl == li
                cx = cols_of(x)
                st = buf(li, "stage", cx)
                forms.append(("set", st, ("copy", x)))
                forms.append(("set", buf(li, "stage", cs, off=n * cx), ("copy", s)))
                stv = buf(li, "stage", unit, rows=n * (cx + cs) // unit)
                bx, bs = cx // unit, cs // unit
                p = np.arange(n)[:, None]
                idx = np.concatenate([p * bx + np.arange(bx)[None], n * bx + p * bs + np.arange(bs)[None]], 1)
                cat = buf(li, "cat", cx + cs)
                forms.append(("set", cat, ("enc_gather", stv, idx.astype(np.int32), unit)))
                x = resnet(cat, li, cx + cs, cout, U["res"][j], ("a", "b"), "c")
                if U["attn"] is not None:
                    x = transformer(x, li, cout, U["attn"][j], ("a", "b"), "d")
            if U["up"] is not None:
                hh, ww = levels[li]
                y = buf(li - 1, "c", cout)
                conv3(x, cout, batch_index(conv_index(hh, ww, up=True), B, npix[li]), U["up"], y)
                x = y
        assert not skips
        n_ = buf(0, "a", self.chans[0])
        gnorm(x, 0, self.nout, self.eps, True, n_)
        conv_same(n_, 0, self.chans[0], self.cout, "out")
        # the pools, each the size of its largest use, and the views
        pools = {}
        for key, off, rows, cols in uses.values():
            pools[key] = max(pools.get(key, 0), off + rows * cols)
        pools = {k: np.zeros(v, np.float32) for k, v in pools.items()}
        for nm, (key, off, rows, cols) in uses.items():
            env[nm] = pools[key][off:off + rows * cols].reshape(rows, cols)
        c.compile(("seq", *forms))
        return c.p.finish()

    def output(self, x, t, ctx):
        """The UNet's output (B, out channels, h, w) of latents x (B, in
        channels, h, w) at timestep t, the prompts' rows ctx (B, 77, dim)."""
        B, _, h, w = x.shape
        prog, g = self.runner.get((B, h, w), self.gpu)
        inputs = dict(x=x.reshape(B, self.in_ch, h * w).transpose(0, 2, 1).reshape(B * h * w, -1),
                      ctx=ctx.reshape(B * TEXT, -1))
        inputs.update(self.vectors(t))
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            g.download("out")
        else:
            prog.run()
        out = prog.names["out"].reshape(B, h * w, self.out_ch).transpose(0, 2, 1)
        return np.ascontiguousarray(out.reshape(B, self.out_ch, h, w))
