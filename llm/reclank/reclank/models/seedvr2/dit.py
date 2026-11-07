"""SeedVR2's DiT (the official NaDiT) as one program a latent size: the one call at t = 1000 of the
one-step model. Two variants:

Ported from SeedVR (ByteDance-Seed/SeedVR, models/dit_v2 and models/dit; Apache License 2.0;
Copyright (c) 2025 Bytedance Ltd. and/or its affiliates) and diffusers
(src/diffusers/models/embeddings.py; Apache License 2.0; Copyright 2026 The HuggingFace Team). See
THIRD_PARTY_NOTICES.md.

    3B (models/dit_v2): 32 layers, width 2560, 20 heads of 128; layers 0-9 with their own text
        weights (MMDiT), 10-31 shared; SwiGLU 6912 (no biases); mmrope3d; an output norm
    7B (models/dit): 36 layers, width 3072, 24 heads of 128; every layer its own text weights;
        GELU (tanh) MLP 12288 with biases; the "pixel" RoPE of the video rows; no output norm

    the rows: the video tokens (frame, row, col: the latent / 2; patches 1 x 2 x 2 of 33 channels:
        the noise, the LQ latent and a mask of 1) then the 58 text rows (pos_emb through txt_in)
    a layer: RMS norm (no weights) times scale plus shift, q k v, the RMS norms of q and k of each
        head (q's weight times 128^-1/2: the attention's scale), RoPE, the window attention, the
        output (bias), times gate, + x; RMS norm, times scale plus shift, the MLP, times gate, + x.
        The scales, shifts and gates are constants of the checkpoint (AdaSingle: the timestep
        embedding of t = 1000, laid out (d l g), plus the layer's own vectors), folded into the
        norms' weights (ENC_RMS), ENC_BIAS_CLAMP and ENC_MUL_VEC. The 3B's last layer: the text
        rows' attention input not modulated (vid_only); the last layer's text MLP is not run (its
        output reaches nothing).
    the windows (window.py's 720pwin_by_size_bysize in even layers, the shifted 720pswin in odd
        ones, (4, 3, 3)): each window a sequence of its video rows and a copy of the text rows.
        q, k, v are gathered (ENC_GATHER) into that layout, the attention (ENC_ATTN_TILE on the
        GPU in float32, ENC_ATTN_RANGE on the CPU) runs over each row's window (its lo, hi), the
        video rows gathered back, the text rows the mean of their copies (ENC_ADD, then times
        1 / windows).
    RoPE, a table a layer kind (ENC_ROPE_TAB; interleaved pairs; the angle a float32 product as
        rotary_embedding_torch's, its cos and sin of float64 rounded to float32: torch's own
        float32 cos is within an ulp of these):
        3B mmrope3d ("lang" freqs of the checkpoint, 21 an axis, 63 of the 64 pairs): a video row
            at (58 + its frame in its window, its row in it, its column), a text row l at (l, l, l)
        7B pixel (the checkpoint's 10 freqs an axis, 30 of the 64 pairs): a video row at
            torch.linspace(-1, 1, n) of its frame, row and column in its window (n the window's
            extent on that axis); the text rows not turned
    the output: the 3B's RMS norm (affine) times (the embedding's attention scale + out_scale)
        plus (its attention shift + out_shift) (the official vid_out_ada reads the cached vectors
        of the first block's attention); vid_out (64: (y, x, channel) of a patch)

The matrices: on the GPU F20 (encoders.runtime.F20Lins), on the CPU the checkpoint's float32 as
stored (or F20 by f20). vid_in (132 columns) stays float32. The biases, norms and vectors float32.
The 7B's blocks (20.9 GB of F20) are streamed on the GPU (stream; the default for the 7B), as Wan
A14B's (models/wan/dit.py): each block's matrices in one registered host array, the programs'
linears of block j reading slot j % 2 of two slots on the GPU, GP_FETCH j after FETCH_DONE of
block j - 2; vid_in, txt_in and vid_out stay resident.
"""
from __future__ import annotations

import ctypes
import os
import math
import mmap

import numpy as np

from reclank.encoders.runtime import F20Lins, _EncCompiler, _Plain, _Runner, lin_q
from reclank.ir import program as P
from reclank.runtime.kept import Kept
from reclank.runtime.threads import team_threads

# the variants: width, heads, layers, the layers with their own text weights, the MLP and its width,
# the RoPE, the output norm, the last layer's text unmodulated (vid_only)
VARIANTS = {
    "3b": dict(D=2560, heads=20, layers=32, mm=10, mlp="swiglu", ff=6912, rope="mm", out_norm=True, vid_only=True),
    "7b": dict(D=3072, heads=24, layers=36, mm=36, mlp="gelu", ff=12288, rope="pixel", out_norm=False,
               vid_only=False),
}
HD = 128
EPS = 1e-5
ALIGN = 4096
_REGISTERED = set()     # (address, bytes) of the host memory registered with CUDA here
MLP_ROWS = 4096     # the rows of the MLP's products at a time (its buffers)


def window_slices(t, h, w, shifted, nwin=(4, 3, 3)):
    """The windows (frame, row, col slices) of a token grid t x h x w, in the official order
    (window.py: make_720Pwindows_bysize, make_shifted_720Pwindows_bysize)."""
    rnt, rnh, rnw = nwin
    scale = math.sqrt((45 * 80) / (h * w))
    rh, rw = round(h * scale), round(w * scale)
    wh, ww = math.ceil(rh / rnh), math.ceil(rw / rnw)
    wt = math.ceil(min(t, 30) / rnt)
    if not shifted:
        nt, nh, nw = math.ceil(t / wt), math.ceil(h / wh), math.ceil(w / ww)
        sl = lambda i, n, size: slice(i * n, min((i + 1) * n, size))  # noqa: E731
        cand = [(sl(it, wt, t), sl(ih, wh, h), sl(iw, ww, w))
                for iw in range(nw) for ih in range(nh) for it in range(nt)]
    else:
        st, sh, sw = (0.5 if wt < t else 0, 0.5 if wh < h else 0, 0.5 if ww < w else 0)
        nt, nh, nw = math.ceil((t - st) / wt), math.ceil((h - sh) / wh), math.ceil((w - sw) / ww)
        nt, nh, nw = (nt + 1 if st > 0 else 1, nh + 1 if sh > 0 else 1, nw + 1 if sw > 0 else 1)
        sl = lambda i, s, n, size: slice(max(int((i - s) * n), 0), min(int((i - s + 1) * n), size))  # noqa: E731
        cand = [(sl(it, st, wt, t), sl(ih, sh, wh, h), sl(iw, sw, ww, w))
                for iw in range(nw) for ih in range(nh) for it in range(nt)]
    return [s for s in cand if all(x.stop > x.start for x in s)]


def patchify(lat):
    """The DiT's input rows (t h/2 w/2, 4 C) of (t, h, w, C): each patch's (y, x, channel)."""
    t, h, w, C = lat.shape
    x = lat.reshape(t, h // 2, 2, w // 2, 2, C).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(x.reshape(t * (h // 2) * (w // 2), 4 * C), dtype=np.float32)


def unpatchify(rows, t, hh, ww, C=16):
    x = rows.reshape(t, hh, ww, 2, 2, C).transpose(0, 1, 3, 2, 4, 5)
    return np.ascontiguousarray(x.reshape(t, 2 * hh, 2 * ww, C), dtype=np.float32)


def timestep_embedding(t, dim=256):
    """diffusers' get_timestep_embedding (flip_sin_to_cos False, shift 0) in float32: [sin, cos]."""
    half = dim // 2
    ex = np.exp(np.float32(-math.log(10000)) * np.arange(half, dtype=np.float32) / np.float32(half)).astype(np.float32)
    e = (np.float32(t) * ex).astype(np.float32)
    return np.concatenate([np.sin(e), np.cos(e)]).astype(np.float32)


def _silu(x):
    return (x / (np.float32(1) + np.exp(-x))).astype(np.float32)


def torch_linspace(start, end, n):
    """torch.linspace(start, end, n) in float32 as torch's CPU kernel: start + step i for the first
    half, end - step (n - 1 - i) for the rest (models/zimage/pipeline.py's)."""
    from reclank.models.zimage.pipeline import torch_linspace as tl
    return tl(start, end, n)


# the DiT's F20 matrices on the GPU's tensor cores: x and W in bfloat16 (the official DiT's autocast), float32
# sums (ENC_LINEAR wbf 5); 0: the float32 products of the F20 values
DIT_TC = os.environ.get("RECLANK_SEEDVR2_DIT_TC", "1") != "0"
# the matrices' form on the GPU (RECLANK_SEEDVR2_W16, vae_video's): "bf16" F20 (the products above), "f16" W16H
# (float16 rows with a scale of each), "i16" W16I (int16 with a scale of each group of 64): encoders.runtime.to_w16,
# 16 bits a value, on the tensor cores
W16 = os.environ.get("RECLANK_SEEDVR2_W16", "f16")


class SeedVR2DiT:
    """The DiT of a SeedVR2 checkpoint (a SafeTensors or a dict of arrays; variant "3b" or "7b")
    with the text rows txt (58, 5120: pos_emb as float32). velocity(rows of (t, h, w, 33)) runs the
    call. stream: the blocks streamed (None: the 7B on the GPU)."""

    def __init__(self, W, txt, gpu=False, cpu_threads=None, f20=None, variant="3b", stream=None):
        self.W = W
        self.v = V = VARIANTS[variant]
        self.variant = variant
        self.D, self.heads, self.ff = V["D"], V["heads"], V["ff"]
        D = self.D
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.stream = bool(self.gpu and variant == "7b" if stream is None else stream)
        assert not self.stream or self.gpu, "the streamed blocks are a GPU mode"
        f20 = self.gpu if f20 is None else bool(f20)
        assert f20 or not self.gpu, "the GPU's DiT is F20"
        fmt = W16 if self.gpu and W16 in ("f16", "i16") else "f20"
        self.f20 = F20Lins(self.gpu, "seedvr2 %s dit" % variant, fmt=fmt) if f20 else None
        self.txt = np.ascontiguousarray(txt, dtype=np.float32)
        self.L = self.txt.shape[0]
        # the arrays and linears from the weightkeeper when it holds them (runtime/kept.py), else made;
        # not for the streamed blocks (_pack copies them) nor a dict of arrays (no file)
        path = getattr(W, "path", None)
        self.kept = k = Kept("seedvr2-%s-dit" % variant, path, dict(gpu=self.gpu, f20=f20, fmt=fmt), code=[__name__]) \
            if path and not self.stream else Kept.off()
        f = lambda n: k.array(n, lambda: np.ascontiguousarray(W.get(n), dtype=np.float32))  # noqa: E731
        # the timestep embedding of t = 1000: (D, 2, 3) = (d, layer attn/mlp, shift/scale/gate)
        e = f("emb_in.proj_in.weight") @ timestep_embedding(1000.0) + f("emb_in.proj_in.bias")
        e = f("emb_in.proj_hid.weight") @ _silu(e.astype(np.float32)) + f("emb_in.proj_hid.bias")
        e = f("emb_in.proj_out.weight") @ _silu(e.astype(np.float32)) + f("emb_in.proj_out.bias")
        emb = e.astype(np.float32).reshape(D, 2, 3)
        self.emb = emb
        L = self.linear
        self.vid_in = L("vid_in.proj", f20=False)
        self.txt_in = L("txt_in")
        self.blocks, self.packed, self.offs, self.block_bytes = [], {}, None, 0
        for i in range(V["layers"]):
            p = "blocks.%d." % i
            streams = ("vid", "txt") if i < V["mm"] else ("all",)
            B = dict(streams={})
            for st in streams:
                q = p + "attn.proj_qkv.%s" % st
                S = dict(q=L(q, rows=(0, D)), k=L(q, rows=(D, 2 * D)), v=L(q, rows=(2 * D, 3 * D)),
                         o=L(p + "attn.proj_out.%s" % st),
                         nq=(f(p + "attn.norm_q.%s.weight" % st) * np.float32(HD ** -0.5)).astype(np.float32),
                         nk=f(p + "attn.norm_k.%s.weight" % st),
                         up=L(p + "mlp.%s.proj_in" % st), down=L(p + "mlp.%s.proj_out" % st))
                if V["mlp"] == "swiglu":
                    S["gate"] = L(p + "mlp.%s.proj_in_gate" % st)
                for li, layer in enumerate(("attn", "mlp")):
                    S[layer + "_w"] = (emb[:, li, 1] + f(p + "ada.%s.%s_scale" % (st, layer))).astype(np.float32)
                    S[layer + "_b"] = (emb[:, li, 0] + f(p + "ada.%s.%s_shift" % (st, layer))).astype(np.float32)
                    S[layer + "_g"] = (emb[:, li, 2] + f(p + "ada.%s.%s_gate" % (st, layer))).astype(np.float32)
                B["streams"][st] = S
            B["freqs"] = f(p + "attn.rope.rope.freqs")
            self.blocks.append(B)
            if self.stream:
                self._pack(i)
        if V["out_norm"]:
            self.out_w = (f("vid_out_norm.weight") * (emb[:, 0, 1] + f("vid_out_ada.out_scale"))).astype(np.float32)
            self.out_b = (emb[:, 0, 0] + f("vid_out_ada.out_shift")).astype(np.float32)
        self.vid_out = L("vid_out.proj")
        if self.f20 is not None and k.mode != "read":
            self.f20.report()
        k.done()
        if self.stream:
            print("seedvr2 %s dit: %d blocks streamed (%.1f MB a block, %.2f GB in host memory)" % (
                variant, len(self.packed), self.block_bytes / 1e6, sum(v[1] for v in self.packed.values()) / 1e9),
                flush=True)
        freqs = [B["freqs"] for B in self.blocks]
        assert all(np.array_equal(freqs[0], x) for x in freqs), "the layers' RoPE freqs differ"
        self.freqs = freqs[0]
        self.slots = None
        self.runner = _Runner(self._build, keep=2, keep_gpu=1)

    def linear(self, name, rows=None, f20=True):
        lin = self.kept.linear(name if rows is None else "%s@%d:%d" % (name, rows[0], rows[1]),
                               lambda: self._linear(name, rows, f20))
        if self.gpu and DIT_TC and getattr(lin, "f20", None) is not None:
            lin.tc = True       # (ENC_LINEAR's F20 rows on the tensor cores: k_enc_gemm_f20tc)
        return lin

    def _linear(self, name, rows, f20):
        src = self.W.get(name + ".weight")
        w = src if rows is None else src[rows[0]:rows[1]]
        lin = _Plain(np.ascontiguousarray(w, dtype=np.float32))
        lin.shape = lin.w.shape
        b = name + ".bias"
        lin.b = None
        if b in self.W.header:
            bias = self.W.get(b)
            lin.b = np.ascontiguousarray(bias if rows is None else bias[rows[0]:rows[1]], dtype=np.float32)
        if self.f20 is not None and f20:
            # (a row slice drops only its own pages: the other slices of the matrix come next)
            self.f20.add(lin, src if rows is None else w)
        return lin

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    # ---- the streamed blocks (the 7B) ------------------------------------------------------------

    def _mats(self, i):
        """Block i's matrices: [(stream, name, linear)] in a fixed order."""
        out = []
        for st, S in sorted(self.blocks[i]["streams"].items()):
            for m in ("q", "k", "v", "o", "gate", "up", "down"):
                if m in S:
                    out.append((st, m, S[m]))
        return out

    def _pack(self, i):
        """Block i's F20 matrices in one host array (its layout self.offs, each at a multiple of
        ALIGN), to be registered with CUDA: self.packed[i] = (address, bytes, array)."""
        offs, tot = {}, 0
        for st, m, lin in self._mats(i):
            assert lin_q(lin)[1] is not None, "a streamed matrix that F20 refused: %d %s %s" % (i, st, m)
            offs[(st, m)] = tot
            tot += (lin_q(lin)[1].data.nbytes + ALIGN - 1) // ALIGN * ALIGN
        assert self.offs in (None, offs)
        self.offs, self.block_bytes = offs, tot
        m_ = mmap.mmap(-1, tot, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        buf = np.frombuffer(m_, np.uint8)
        for st, m, lin in self._mats(i):
            q = lin_q(lin)[1]
            d = q.data
            v = buf[offs[(st, m)]:offs[(st, m)] + d.nbytes]
            v[...] = d.reshape(-1)
            q.data = v.reshape(d.shape)
        self.packed[i] = (buf.ctypes.data, tot, buf)

    def _register(self):
        from reclank.backends.cuda.runtime import lib
        for a, n, _b in self.packed.values():
            if (a, n) not in _REGISTERED:
                if lib().rc_gpu_host_register_rw(ctypes.c_void_p(a), n) == 0:
                    raise RuntimeError("cudaHostRegister of %.2f GB: %s" % (n / 1e9, lib().rc_gpu_last_error().decode()))
                _REGISTERED.add((a, n))

    def _slots(self):
        if self.slots is None:
            self.slots = [np.frombuffer(mmap.mmap(-1, self.block_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS),
                                        np.uint8) for _ in range(2)]
        return self.slots

    def _proxy(self, j, slot):
        """Block j with its matrices' F20 rows read from slot."""
        from reclank.models.gemma4.e4b import _KQ
        B = dict(self.blocks[j])
        B["streams"] = {}
        for st, S in self.blocks[j]["streams"].items():
            S2 = dict(S)
            for m in ("q", "k", "v", "o", "gate", "up", "down"):
                if m in S:
                    src = S[m]
                    p = _Plain(None)
                    p.b, p.shape = src.b, src.shape
                    a, q = lin_q(src)
                    setattr(p, a, _KQ(slot[self.offs[(st, m)]:self.offs[(st, m)] + q.data.nbytes], q.type, q.rows,
                                      q.cols))
                    p.tc = getattr(src, "tc", False)
                    S2[m] = p
            B["streams"][st] = S2
        return B

    def _fill(self, prog):
        """The device addresses of a streamed program's ranges: its slots' device copies."""
        if not getattr(prog, "ranges", None):
            return
        mir = self.runner.mirror
        dst = [mir.translate(a.ctypes.data) for a in self.slots]
        for arr, s, keys in prog.ranges:
            for k, key in enumerate(keys):
                arr[k, 1] = dst[s] + self.offs[key]

    # ---- the windows and RoPE (host) -------------------------------------------------------------

    def layout(self, t, h, w, shifted):
        """The window layout of a layer kind: idx (R, 1) of the rows of [video N; text L] in it, the
        lo, hi of each of its rows, the place of each video row in it (N, 1), the start of each
        window's text copy, and the windows' (frames, rows, cols)."""
        N, L = t * h * w, self.L
        grid = np.arange(N).reshape(t, h, w)
        idx, lo, hi, tstart, shapes = [], [], [], [], []
        inv = np.zeros(N, np.int32)
        r = 0
        for s in window_slices(t, h, w, shifted):
            v = grid[s].reshape(-1)
            n = v.size + L
            inv[v] = r + np.arange(v.size)
            idx.append(np.concatenate([v, np.arange(N, N + L)]))
            lo.append(np.full(n, r))
            hi.append(np.full(n, r + n))
            tstart.append(r + v.size)
            shapes.append(grid[s].shape)
            r += n
        return (np.concatenate(idx).astype(np.int32)[:, None], np.concatenate(lo).astype(np.int32),
                np.concatenate(hi).astype(np.int32), inv[:, None], tstart, shapes)

    def angles(self, t, h, w, shifted):
        """The RoPE angles (N + L, 3 x freqs) of a layer kind: float32 products as
        rotary_embedding_torch's (the 7B's text rows 0: not turned)."""
        N, L = t * h * w, self.L
        fr = self.freqs
        pos = np.zeros((N + L, 3), np.float32)
        grid = np.arange(N).reshape(t, h, w)
        for s in window_slices(t, h, w, shifted):
            g = grid[s]
            if self.v["rope"] == "mm":
                axes = [np.arange(g.shape[0]) + L, np.arange(g.shape[1]), np.arange(g.shape[2])]
            else:
                axes = [torch_linspace(-1.0, 1.0, n) for n in g.shape]
            pf, py, px = np.meshgrid(*axes, indexing="ij")
            pos[g.reshape(-1)] = np.stack([pf, py, px], -1).reshape(-1, 3)
        if self.v["rope"] == "mm":
            pos[N:] = np.arange(L, dtype=np.float32)[:, None]
        ang = (pos[:, :, None] * fr[None, None, :]).astype(np.float32).reshape(N + L, 3 * fr.size)
        if self.v["rope"] != "mm":
            ang[N:] = 0
        return ang

    def rope_table(self, t, h, w, shifted):
        ang = self.angles(t, h, w, shifted).astype(np.float64)
        pad = HD // 2 - ang.shape[1]
        cos = np.concatenate([np.cos(ang), np.ones((ang.shape[0], pad))], 1)
        sin = np.concatenate([np.sin(ang), np.zeros((ang.shape[0], pad))], 1)
        return cos.astype(np.float32), sin.astype(np.float32)

    # ---- the program ------------------------------------------------------------------------------

    def program(self, t, h, w, gpu=False):
        """The call of t x h x w tokens: input "vid" (N, 132) (patchify of the noise and the
        condition), output "vel" (N, 64)."""
        D, HEADS, FF, V = self.D, self.heads, self.ff, self.v
        N, L = t * h * w, self.L
        NL = N + L
        c = _EncCompiler(EPS, pack=not gpu)
        c.attn_f32 = True
        env = c.env
        z = lambda *s: np.zeros(s, np.float32)  # noqa: E731
        lays = [self.layout(t, h, w, k) for k in (False, True)]
        R = max(Lk[0].shape[0] for Lk in lays)
        env.update(vid=z(N, 132), txt=self.txt.copy(), vel=z(N, 64), x=z(NL, D), hh=z(NL, D), tmp=z(NL, D),
                   q=z(NL, D), k=z(NL, D), v=z(NL, D), o=z(NL, D),
                   qw=z(R, D), kw=z(R, D), vw=z(R, D), ow=z(R, D),
                   g=z(min(NL, MLP_ROWS), FF))
        if V["mlp"] == "swiglu":
            env["u"] = z(min(NL, MLP_ROWS), FF)
        for kind, (idx, lo, hi, inv, tstart, shapes) in enumerate(lays):
            n = idx.shape[0]
            lo_, hi_ = np.full(R, n, np.int32), np.full(R, n, np.int32)   # rows past the layout: no keys
            lo_[:n], hi_[:n] = lo, hi
            env["idx%d" % kind], env["lo%d" % kind], env["hi%d" % kind], env["inv%d" % kind] = idx, lo_, hi_, inv
            env["cos%d" % kind], env["sin%d" % kind] = self.rope_table(t, h, w, bool(kind))
            env["tw%d" % kind] = np.full(D, 1.0 / len(tstart), np.float32)
            for j, r in enumerate(tstart):
                env["owt%d_%d" % (kind, j)] = env["ow"][r:r + L]
        for nm in ("x", "hh", "tmp", "q", "k", "v", "o"):
            env[nm + "_v"], env[nm + "_t"] = env[nm][:N], env[nm][N:]
        parts = {}

        def mlp_rows(r_lo, r_hi):
            out = []
            for r0 in range(r_lo, r_hi, MLP_ROWS):
                r1 = min(r_hi, r0 + MLP_ROWS)
                s = "_m%d_%d" % (r0, r1)
                if s not in parts:
                    parts[s] = True
                    env["hh" + s], env["tmp" + s] = env["hh"][r0:r1], env["tmp"][r0:r1]
                    env["g" + s] = env["g"][:r1 - r0]
                    if "u" in env:
                        env["u" + s] = env["u"][:r1 - r0]
                out.append(s)
            return out

        def block(i, B):
            kind = i % 2
            last = i == V["layers"] - 1
            shared = "all" in B["streams"]
            forms = []
            # the streams' rows: (stream weights, suffix, MLP row range: None for the last layer's text)
            if shared and not last:
                streams = [(B["streams"]["all"], "", (0, NL))]
            elif shared:
                streams = [(B["streams"]["all"], "_v", (0, N)), (B["streams"]["all"], "_t", None)]
            else:
                streams = [(B["streams"]["vid"], "_v", (0, N)), (B["streams"]["txt"], "_t", None if last else (N, NL))]
            for S, sfx, rr in streams:
                mod = not (last and sfx == "_t" and V["vid_only"])
                forms.append(("set", "hh" + sfx, ("enc_rms", "x" + sfx, S["attn_w"] if mod else None, D)))
                if mod:
                    forms.append(("enc_bias", "hh" + sfx, S["attn_b"]))
                for m in "qkv":
                    forms.append(("set", m + sfx, ("enc_linear", S[m], "hh" + sfx)))
                forms += [("set", "q" + sfx, ("enc_rms", "q" + sfx, S["nq"], HD)),
                          ("set", "k" + sfx, ("enc_rms", "k" + sfx, S["nk"], HD))]
            nwin = len(lays[kind][4])
            forms += [("enc_rope_tab", "q", "cos%d" % kind, "sin%d" % kind, HEADS, HD),
                      ("enc_rope_tab", "k", "cos%d" % kind, "sin%d" % kind, HEADS, HD),
                      ("set", "qw", ("enc_gather", "q", "idx%d" % kind, D)),
                      ("set", "kw", ("enc_gather", "k", "idx%d" % kind, D)),
                      ("set", "vw", ("enc_gather", "v", "idx%d" % kind, D)),
                      ("set", "ow", ("enc_attn_range_x2", "qw", "kw", "vw", "lo%d" % kind, "hi%d" % kind, HEADS, HD)),
                      ("set", "o_v", ("enc_gather", "ow", "inv%d" % kind, D))]
            if not last:
                forms.append(("set", "o_t", ("copy", "owt%d_0" % kind)))
                forms += [("enc_add", "o_t", "owt%d_%d" % (kind, j)) for j in range(1, nwin)]
                forms.append(("set", "o_t", ("enc_mul_vec", "o_t", "tw%d" % kind)))
            for S, sfx, rr in streams:
                if rr is None:
                    continue            # the last layer's text: its output is not used
                forms += [("set", "tmp" + sfx, ("enc_linear", S["o"], "o" + sfx)),
                          ("set", "tmp" + sfx, ("enc_mul_vec", "tmp" + sfx, S["attn_g"])),
                          ("enc_add", "x" + sfx, "tmp" + sfx),
                          ("set", "hh" + sfx, ("enc_rms", "x" + sfx, S["mlp_w"], D)),
                          ("enc_bias", "hh" + sfx, S["mlp_b"])]
                for s in mlp_rows(*rr):
                    if V["mlp"] == "swiglu":
                        forms += [("set", "g" + s, ("enc_linear", S["gate"], "hh" + s)),
                                  ("set", "u" + s, ("enc_linear", S["up"], "hh" + s)),
                                  ("set", "g" + s, ("enc_silu_mul", "g" + s, "u" + s))]
                    else:
                        forms += [("set", "g" + s, ("enc_linear", S["up"], "hh" + s)),
                                  ("set", "g" + s, ("enc_gelu", "g" + s))]
                    forms.append(("set", "tmp" + s, ("enc_linear", S["down"], "g" + s)))
                forms += [("set", "tmp" + sfx, ("enc_mul_vec", "tmp" + sfx, S["mlp_g"])),
                          ("enc_add", "x" + sfx, "tmp" + sfx)]
            return forms
        c.compile(("seq", ("set", "x_v", ("enc_linear", self.vid_in, "vid")),
                   ("set", "x_t", ("enc_linear", self.txt_in, "txt"))))
        ranges = []
        n = len(self.blocks)
        if not self.stream:
            for i, B in enumerate(self.blocks):
                c.compile(("seq", *block(i, B)))
        else:
            slots = self._slots()
            keys = [(st, m) for st, m, _ in self._mats(0)]
            rng = []
            for j in range(n):
                r = np.zeros((len(keys), 3), np.int64)
                for k, (st, m) in enumerate(keys):
                    r[k, 0] = self.packed[j][0] + self.offs[(st, m)]
                    r[k, 2] = lin_q(self.blocks[j]["streams"][st][m])[1].data.nbytes
                rng.append(r)
            for j in range(min(2, n)):
                c.p.emit(P.FETCH, rng[j], len(keys), j, j % 2)
            for j in range(n):
                s = j % 2
                c.p.emit(P.FETCH_WAIT, j)
                c.compile(("seq", *block(j, self._proxy(j, slots[s]))))
                c.p.emit(P.FETCH_DONE, s)
                if j + 2 < n:
                    c.p.emit(P.FETCH, rng[j + 2], len(keys), j + 2, s)
            ranges = [(rng[j], j % 2, keys) for j in range(n)]
        if V["out_norm"]:
            c.compile(("seq", ("set", "hh_v", ("enc_rms", "x_v", self.out_w, D)), ("enc_bias", "hh_v", self.out_b),
                       ("set", "vel", ("enc_linear", self.vid_out, "hh_v"))))
        else:
            c.compile(("set", "vel", ("enc_linear", self.vid_out, "x_v")))
        prog = c.p.finish()
        prog.ranges = ranges
        return prog

    def velocity(self, vid):
        """The DiT's output v (t, h, w, 16) of its input (t, h, w, 33): [noise, latent, 1]."""
        t, H, W_, _ = vid.shape
        hh, ww = H // 2, W_ // 2
        if self.stream and not getattr(self, "_registered", False):
            self._register()
            self._registered = True
        prog, g = self.runner.get((t, hh, ww), self.gpu)
        if g is not None:
            self._fill(prog)
        prog.names["vid"][...] = patchify(vid)
        if g is not None:
            g.upload("vid")
            g.run()
            g.download("vel")
        else:
            prog.run()
        return unpatchify(prog.names["vel"], t, hh, ww)

    def release(self):
        self.runner.release(keep_host=True)

    def close(self):
        """Free the GPU memory and, when streamed, unregister the blocks' host memory from CUDA (before
        it is freed: a range left registered after its unmap fails the copies of the memory mapped
        there next)."""
        self.runner.release(keep_host=False)
        if self.packed and getattr(self, "_registered", False):
            from reclank.backends.cuda.gpumm import mem
            from reclank.backends.cuda.runtime import lib
            mem().poll(wait=True)           # (the runs queued before are done with the blocks)
            for a, n, _b in self.packed.values():
                if (a, n) in _REGISTERED:
                    lib().rc_gpu_host_unregister(ctypes.c_void_p(a))
                    _REGISTERED.discard((a, n))
            self._registered = False
