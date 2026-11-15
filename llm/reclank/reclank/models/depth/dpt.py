"""The DPT head of Depth Anything (DA3's mono head; Video Depth Anything's,
with its temporal modules) as forms of a program.

Ported from Depth Anything 3 (ByteDance-Seed/Depth-Anything-3,
src/depth_anything_3/model/dpt.py; Apache License 2.0; Copyright (c) 2025
ByteDance Ltd. and/or its affiliates) and Video Depth Anything
(DepthAnything/Video-Depth-Anything, video_depth_anything/dpt.py,
dpt_temporal.py and motion_module, the last also Copyright 2022 The HuggingFace
Team; Apache License 2.0; Copyright (2025) Bytedance Ltd. and/or its
affiliates). See THIRD_PARTY_NOTICES.md.

    projects (1 x 1) of the four ViT outputs -> ConvTranspose 4 (x4), 2 (x2),
    identity, a 3 x 3 stride-2 conv; the "rn" 3 x 3 convs (no bias) to the
    features; refinenet4..1 (ResidualConvUnits: ReLU, conv, ReLU, conv, + x;
    bilinear align_corners to the next level's size, a 1 x 1 out conv);
    output_conv1 (3 x 3, to features / 2) at 8x the patch grid, bilinear to
    14x, output_conv2 (3 x 3 to 32, ReLU, 1 x 1 to 1).

VDA's temporal modules (at layer 3, layer 4, after refinenet4 and 3): a
GroupNorm of each frame (32 groups), a linear in, two self-attentions of each
pixel over the window's frames (8 heads, the stored sinusoidal frame
positions added to the normed rows), a GEGLU FFN, a linear out, + x.
"""
from __future__ import annotations

import numpy as np

from reclank.models.depth.common import conv_index, conv_lin, lin


class DPT:
    """A DPT head of a state dict (get(prefix + name)). temporal: VDA's motion
    modules; sky: DA3's sky branch; relu_out: VDA's ReLU on the output."""

    def __init__(self, get, prefix, temporal=False, sky=False, relu_out=False):
        self.get, self.p = get, prefix
        g = lambda n: np.asarray(get(prefix + n), np.float32)  # noqa: E731
        self.g = g
        cv = lambda n, bias=True: conv_lin(lambda m: get(prefix + m), n, bias)  # noqa: E731
        self.projects = [cv("projects.%d" % i) for i in range(4)]
        self.rs0 = (g("resize_layers.0.weight"), g("resize_layers.0.bias"))
        self.rs1 = (g("resize_layers.1.weight"), g("resize_layers.1.bias"))
        self.rs3 = cv("resize_layers.3")
        s = "scratch."
        self.rn = [cv(s + "layer%d_rn" % (i + 1), bias=False) for i in range(4)]
        self.features = self.rn[0].shape[0]

        def rcu(n):
            return (cv(n + ".conv1"), cv(n + ".conv2"))

        self.ref = {}
        for i in range(1, 5):
            n = s + "refinenet%d" % i
            has1 = True
            try:
                get(prefix + n + ".resConfUnit1.conv1.weight")
            except KeyError:
                has1 = False
            self.ref[i] = dict(r1=rcu(n + ".resConfUnit1") if has1 and i < 4 else None,
                               r2=rcu(n + ".resConfUnit2"), out=cv(n + ".out_conv"))
        self.oc1 = cv(s + "output_conv1")
        self.oc2 = (cv(s + "output_conv2.0"), cv(s + "output_conv2.2"))
        self.sky = (cv(s + "sky_output_conv2.0"), cv(s + "sky_output_conv2.2")) if sky else None
        self.relu_out = relu_out
        self.tm = [self._temporal("motion_modules.%d." % i) for i in range(4)] if temporal else None

    def _temporal(self, p):
        g = self.g
        t = p + "temporal_transformer."
        b = t + "transformer_blocks.0."
        C = g(t + "proj_in.weight").shape[0]
        att = []
        sc = np.float32((C // 8) ** -0.5)       # the attention's scale in q (ENC_ATTN_RANGE takes none)
        for i in range(2):
            a = b + "attention_blocks.%d." % i
            att.append(dict(norm=(g(b + "norms.%d.weight" % i), g(b + "norms.%d.bias" % i)),
                            pe=g(a + "pos_encoder.pe").reshape(-1, C),
                            q=lin(g(a + "to_q.weight") * sc), k=lin(g(a + "to_k.weight")), v=lin(g(a + "to_v.weight")),
                            o=lin(g(a + "to_out.0.weight"), g(a + "to_out.0.bias"))))
        w0, b0 = g(b + "ff.net.0.proj.weight"), g(b + "ff.net.0.proj.bias")
        inner = w0.shape[0] // 2
        return dict(C=C, gn=(g(t + "norm.weight"), g(t + "norm.bias")),
                    pin=lin(g(t + "proj_in.weight"), g(t + "proj_in.bias")),
                    pout=lin(g(t + "proj_out.weight"), g(t + "proj_out.bias")), att=att,
                    ffn=(g(b + "ff_norm.weight"), g(b + "ff_norm.bias")),
                    ffv=lin(w0[:inner], b0[:inner]), ffg=lin(w0[inner:], b0[inner:]),
                    ff2=lin(g(b + "ff.net.2.weight"), g(b + "ff.net.2.bias")))

    # ---- forms ----

    def rcu(self, B, x, idx, w):
        a = B.relu(x)
        b = B.conv3(a, idx, w[0])
        B.free(a)
        B.f("set", b, ("enc_clamp", b, 0.0, float("inf")))
        c = B.conv3(b, idx, w[1])
        B.free(b)
        return B.add(c, x)

    def fusion(self, B, i, x, lat, n, h, w, H, W):
        """refinenet i of x (n images of h x w; lat its lateral input or None)
        to n of H x W; frees x and lat"""
        R = self.ref[i]
        idx = B.index(("c3", n, h, w), lambda: conv_index(h, w, n))
        if lat is not None:
            r = self.rcu(B, lat, idx, R["r1"])
            B.add(x, r)
            B.free(r, lat)
        y = self.rcu(B, x, idx, R["r2"])
        B.free(x)
        z = B.resize(y, n, h, w, H, W, True)
        B.free(y)
        o = B.linear(R["out"], z)
        B.free(z)
        return o

    def temporal(self, B, k, x, T, h, w):
        """motion module k over x (T frames of h x w, frame-major): a new buffer"""
        M = self.tm[k]
        C, P = M["C"], h * w
        n = B.buf(T * P, C)
        for f in range(T):
            B.f("set", B.view(n, f * P, (f + 1) * P), ("enc_gnorm", B.view(x, f * P, (f + 1) * P),
                                                       M["gn"][0], M["gn"][1], 32, 1e-6, False))
        y = B.linear(M["pin"], n)
        B.free(n)
        # (frame, pixel) <-> (pixel, frame) and each pixel's range of T rows
        to_pf = B.index(("pf", T, P), lambda: np.ascontiguousarray(
            (np.arange(T)[None, :] * P + np.arange(P)[:, None]).reshape(-1, 1), dtype=np.int32))
        to_fp = B.index(("fp", T, P), lambda: np.ascontiguousarray(
            (np.arange(P)[None, :] * T + np.arange(T)[:, None]).reshape(-1, 1), dtype=np.int32))
        lo = B.index(("lo", T, P), lambda: B.name(np.repeat(np.arange(P) * T, T).astype(np.int32)))
        hi = B.index(("hi", T, P), lambda: B.name(np.repeat(np.arange(P) * T + T, T).astype(np.int32)))
        H = 8
        for a in M["att"]:
            hn = B.buf(T * P, C)
            B.f("set", hn, ("enc_lnorm", y, a["norm"][0], a["norm"][1], C, 1e-5))
            hp = B.buf(T * P, C)
            B.f("set", hp, ("enc_gather", hn, to_pf, C))
            B.free(hn)
            pe = B.index(("pe", id(a["pe"]), T, P), lambda: B.name(np.ascontiguousarray(np.tile(a["pe"][:T], (P, 1)))))
            B.add(hp, pe)
            q, kk, v = B.linear(a["q"], hp), B.linear(a["k"], hp), B.linear(a["v"], hp)
            B.free(hp)
            o = B.buf(T * P, C)
            B.f("set", o, ("enc_attn_range_x2", q, kk, v, lo, hi, H, C // H))
            B.free(q, kk, v)
            oo = B.linear(a["o"], o)
            B.free(o)
            back = B.buf(T * P, C)
            B.f("set", back, ("enc_gather", oo, to_fp, C))
            B.free(oo)
            B.add(y, back)
            B.free(back)
        hn = B.buf(T * P, C)
        B.f("set", hn, ("enc_lnorm", y, M["ffn"][0], M["ffn"][1], C, 1e-5))
        gv, gg = B.linear(M["ffv"], hn), B.linear(M["ffg"], hn)
        B.free(hn)
        B.f("set", gv, ("enc_gelu_mul", gg, gv, True))
        B.free(gg)
        f2 = B.linear(M["ff2"], gv)
        B.free(gv)
        B.add(y, f2)
        B.free(f2)
        out = B.linear(M["pout"], y)
        B.free(y)
        return B.add(out, x)

    def forms(self, B, feats, T, ph, pw, micro=4):
        """The head over T frames: feats the four ViT outputs (T (1 + ph pw), C
        rows, cls rows included). Returns the names of the logits (T 14 ph 14
        pw, 1) and of the sky logits (or None)."""
        P = ph * pw
        sizes = [(4 * ph, 4 * pw), (2 * ph, 2 * pw), (ph, pw), ((ph - 1) // 2 + 1, (pw - 1) // 2 + 1)]
        idx = np.ascontiguousarray(np.concatenate(
            [np.arange(f * (P + 1) + 1, (f + 1) * (P + 1)) for f in range(T)]).reshape(-1, 1), dtype=np.int32)
        rs = []
        for i, x in enumerate(feats):
            C = B.shape(x)[1]
            if T == 1:
                pr = B.view(x, 1, P + 1)
            else:
                pr = B.buf(T * P, C)
                B.f("set", pr, ("enc_gather", x, idx, C))
            p = B.linear(self.projects[i], pr)
            B.free(pr, x)
            if i == 0:
                q = B.convT(p, T, ph, pw, *self.rs0, 4)
                B.free(p)
            elif i == 1:
                q = B.convT(p, T, ph, pw, *self.rs1, 2)
                B.free(p)
            elif i == 3:
                q = B.conv3(p, B.index(("s2", T, ph, pw), lambda: conv_index(ph, pw, T, stride=2)), self.rs3)
                B.free(p)
            else:
                q = p
            rs.append(q)
        if self.tm is not None:
            for i, k in ((2, 0), (3, 1)):
                h, w = sizes[i]
                t = self.temporal(B, k, rs[i], T, h, w)
                B.free(rs[i])
                rs[i] = t
        ls = []
        for i, r in enumerate(rs):
            h, w = sizes[i]
            ls.append(B.conv3(r, B.index(("c3", T, h, w), lambda: conv_index(h, w, T)), self.rn[i]))
            B.free(r)
        o = self.fusion(B, 4, ls[3], None, T, *sizes[3], *sizes[2])
        if self.tm is not None:
            t = self.temporal(B, 2, o, T, *sizes[2])
            B.free(o)
            o = t
        o = self.fusion(B, 3, o, ls[2], T, *sizes[2], *sizes[1])
        if self.tm is not None:
            t = self.temporal(B, 3, o, T, *sizes[1])
            B.free(o)
            o = t
        # the rest a micro-batch of frames at a time (VDA's 4; the same rows)
        m = T if T <= micro or T % micro else micro
        H1, W1 = sizes[0]
        HO, WO = 14 * ph, 14 * pw
        logits = B.name(np.zeros((T * HO * WO, 1), np.float32))
        sky = B.name(np.zeros((T * HO * WO, 1), np.float32)) if self.sky is not None else None
        S2 = sizes[1][0] * sizes[1][1]
        S1 = H1 * W1
        for f0 in range(0, T, m):
            ob = B.view(o, f0 * S2, (f0 + m) * S2)
            l1 = B.view(ls[1], f0 * S2, (f0 + m) * S2)
            l0 = B.view(ls[0], f0 * S1, (f0 + m) * S1)
            x = self.fusion(B, 2, ob, l1, m, *sizes[1], H1, W1)
            x = self.fusion(B, 1, x, l0, m, H1, W1, 2 * H1, 2 * W1)
            y = B.conv3(x, B.index(("c3", m, 2 * H1, 2 * W1), lambda: conv_index(2 * H1, 2 * W1, m)), self.oc1)
            B.free(x)
            z = B.resize(y, m, 2 * H1, 2 * W1, HO, WO, True)
            B.free(y)
            ci = B.index(("c3", m, HO, WO), lambda: conv_index(HO, WO, m))
            for heads, dst in ((self.oc2, logits), (self.sky, sky)):
                if heads is None:
                    continue
                a = B.conv3(z, ci, heads[0])
                B.f("set", a, ("enc_clamp", a, 0.0, float("inf")))
                out = B.view(dst, f0 * HO * WO, (f0 + m) * HO * WO)
                B.linear(heads[1], a, out)
                B.free(a)
                if self.relu_out or heads is self.sky:
                    B.f("set", out, ("enc_clamp", out, 0.0, float("inf")))
            B.free(z)
        B.free(o, *ls)
        return logits, sky
