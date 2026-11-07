"""FlashVSR's LQ projector (Causal_LQ4x_Proj, layer_num 1: the condition of the DiT's first block) as programs,
float32 (its bfloat16 weights as float32), on the CPU and the GPU, streamed a clip of 4 LQ frames at a time.

Ported from FlashVSR (OpenImagingLab/FlashVSR, examples/WanVSR/utils/utils.py; Apache License 2.0; no copyright
line stated: OpenImagingLab). See THIRD_PARTY_NOTICES.md.

    a clip's frames (3, 4, H, W) in [-1, 1] pixel-shuffled (1, 16, 16: channel (c 16 + y) 16 + x) to (768, 4, H/16,
        W/16); CausalConv3d 768 -> 2048 (4 x 3 x 3, stride 2 in time; the clip after the last 2 frames of the clip
        before, replicate-padded by 1 in space): 2 frames; RMS_norm (x / |x| sqrt(C) gamma over the channels),
        SiLU; CausalConv3d 2048 -> 3072 (4 x 3 x 3, stride 2) over the last clip's 2 and these 2: 1 frame; RMS_norm,
        SiLU; linear 3072 -> 1536: the condition (H/16 W/16, 1536) of one latent frame
    the first clip (the video's first frame alone): that frame 3 more times before it, and the causal padding of 2
        more (replicate): conv1 of 6 copies of it, its 2 frames the state of conv2, no output

The programs: ("first", h, w) and ("next", h, w) (h, w the tokens' grid) of "clip" (4 h w, 768) over the states x6
(conv1's 6 frames: the last clip's 2, then this one's 4) and y4 (conv2's 4: the last clip's 2 conv1 frames, then
this one's), kept on the device; next's output "lat" (h w, 1536).
"""
from __future__ import annotations

import numpy as np

from reclank.encoders.runtime import _Plain, _Runner
from reclank.models.realbasicvsr.model import _B

DIM = 1536
H1, H2 = 2048, 3072
OUTS = 6            # push_rows' outputs on the GPU: each one's array valid until OUTS more pushes
EPS = 1e-24         # (F.normalize's clamp of the norm at 1e-12, as an eps of the mean square)


def pixel_shuffle16(frames):
    """(n, h w, 768) of frames (3, n, H, W): PixelShuffle3d(1, 16, 16), the rows (y, x)"""
    c, n, H, W = frames.shape
    x = frames.reshape(c, n, H // 16, 16, W // 16, 16).transpose(1, 2, 4, 0, 3, 5)
    return np.ascontiguousarray(x.reshape(n, (H // 16) * (W // 16), c * 256), dtype=np.float32)


def causal_index(h, w, frames, t_out, stride=2):
    """the im2col (t_out h w, 36) of a 4 x 3 x 3 conv of stride 2 in time over `frames` frames of h x w rows (frame
    after frame), output frame j reading frames stride j .. stride j + 3, the space replicate-padded by 1"""
    y = np.arange(h)[:, None, None, None, None] + np.arange(-1, 2)[None, None, None, :, None]
    x = np.arange(w)[None, :, None, None, None] + np.arange(-1, 2)[None, None, None, None, :]
    sp = np.clip(y, 0, h - 1) * w + np.clip(x, 0, w - 1)                    # (h, w, 1, 3, 3)
    out = []
    for j in range(t_out):
        fr = (stride * j + np.arange(4))[None, None, :, None, None]
        assert stride * j + 3 < frames
        out.append((fr * h * w + sp).reshape(h * w, 36))
    return np.ascontiguousarray(np.concatenate(out), dtype=np.int32)


class LQProj:
    """The projector of LQ_proj_in.safetensors' tensors W (name: float32). reset() before a video; push(clip) of
    each clip: None for the first, else its condition (h w, 1536)."""

    def __init__(self, W, gpu=False, cpu_threads=None, tc=False):
        self.W, self.gpu, self.cpu_threads = W, bool(gpu), cpu_threads
        self.tc = bool(tc) and self.gpu
        self.lins = {}
        self.runner = _Runner(self._build, keep=4, keep_gpu=4)
        self.states = {}
        self.outs = {}                  # (h, w): the GPU's outputs' arrays
        self.pools16 = {}               # (h, w): the scratch of the convs' float16 inputs (tc)
        self.n = 0

    def lin(self, name):
        """the conv (its im2col's columns (t, y, x, c)) or linear name, one for all the programs; tc: its W16H rows"""
        if name in self.lins:
            return self.lins[name]
        w = self.W[name + ".weight"]
        W = w.transpose(0, 2, 3, 4, 1).reshape(w.shape[0], -1) if w.ndim == 5 else w
        out = _Plain(np.ascontiguousarray(W, np.float32))
        out.b = np.ascontiguousarray(self.W[name + ".bias"], np.float32)
        out.shape = out.w.shape
        if self.tc:
            from reclank.encoders.runtime import to_w16
            to_w16(out, "f16")
        self.lins[name] = out
        return out

    def _state(self, h, w):
        if (h, w) not in self.states:
            P = h * w
            z = lambda r, c: np.zeros((r, c), np.float32)  # noqa: E731
            self.states[(h, w)] = {"clip": z(4 * P, 768), "x6": z(6 * P, 768), "y4": z(4 * P, H1),
                                   "lat": z(P, DIM)}
        return self.states[(h, w)]

    def _build(self, key, gpu):
        kind, h, w = key
        P = h * w
        if gpu and self.tc and (h, w) not in self.pools16:
            self.pools16[(h, w)] = np.zeros(8192 * P, np.uint16)    # (conv2's y4: the most of a conv's input)
        B = _B(gpu, pool16=self.pools16.get((h, w)) if gpu and self.tc else None)
        B.c.eps = EPS
        st = self._state(h, w)
        B.env.update(st)
        v = lambda a, r0, r1: B.name(st[a][r0:r1])  # noqa: E731
        if kind == "first":
            B.f("set", v("x6", 0, 2 * P), ("copy", B.name(st["clip"][:2 * P])))
        B.f("set", v("x6", 2 * P, 6 * P), ("copy", "clip"))
        y = v("y4", 2 * P, 4 * P)
        c1 = B.conv("x6", 768, self.lin("conv1"), causal_index(h, w, 6, 2))
        g1 = B.name(np.ascontiguousarray(self.W["norm1.gamma"].reshape(-1), np.float32))
        B.f("set", y, ("enc_rms", c1, g1, H1))
        B.f("set", y, ("enc_silu", y))
        if kind == "next":
            c2 = B.conv("y4", H1, self.lin("conv2"), causal_index(h, w, 4, 1))
            g2 = B.name(np.ascontiguousarray(self.W["norm2.gamma"].reshape(-1), np.float32))
            z = B.buf(P, H2)
            B.f("set", z, ("enc_rms", c2, g2, H2))
            B.f("set", z, ("enc_silu", z))
            B.f("set", "lat", ("enc_linear", self.lin("linear_layers.0"), z))
        B.f("set", v("x6", 0, 2 * P), ("copy", v("x6", 4 * P, 6 * P)))     # (the states: the last 2 frames)
        B.f("set", v("y4", 0, 2 * P), ("copy", y))
        return B.finish(self.cpu_threads)

    def reset(self):
        self.n = 0

    @staticmethod
    def rows(clip):
        """push_rows' input of a clip (3, n, H, W) in [-1, 1] (a host step, on any thread): its pixel shuffle
        (n, h, w, 768)"""
        c, n, H, W = clip.shape
        return pixel_shuffle16(clip).reshape(n, H // 16, W // 16, c * 256)

    def push(self, clip):
        """the condition (h w, 1536) of a clip (3, n, H, W) in [-1, 1] (the first: n 1, None)"""
        return self.push_rows(self.rows(clip))

    def push_rows(self, x):
        """push of a clip's rows(clip) (on the GPU the output's array valid until OUTS more pushes)"""
        n, h, w, _ = x.shape
        P = h * w
        x = x.reshape(n, P, 768)
        first = self.n == 0
        if first:
            assert n == 1
            x = np.repeat(x, 4, 0)
        assert x.shape[0] == 4
        prog, g = self.runner.get(("first" if first else "next", h, w), self.gpu)
        self.n += 1
        if g is None:
            prog.names["clip"][...] = x.reshape(4 * P, 768)
            prog.run()
            return None if first else prog.names["lat"].copy()
        # (the rows uploaded and the output downloaded from and to arrays of their own: no host copies)
        g.mirror.buffer_of(prog.names["clip"]).upload(np.ascontiguousarray(x.reshape(4 * P, 768), np.float32))
        g.run()
        if first:
            return None
        # (into a ring of OUTS arrays whose pages are there: a new array's take their faults in the copy, 1.3 GB/s)
        ring = self.outs.setdefault((h, w), [])
        if len(ring) < OUTS:
            ring.append(np.zeros(prog.names["lat"].shape, np.float32))
        out = ring[(self.n - 2) % OUTS]
        g.mirror.buffer_of(prog.names["lat"]).download(out)
        return out

    def release(self):
        if self.gpu:
            self.runner.release(keep_host=True)
