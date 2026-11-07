"""FlashVSR's tiny conditional decoder (TCDecoder: TAEHV's decoder, widened to 512, 256, 128, 128 channels, a 3 x 3
identity-initialized conv + ReLU after each top-level ReLU, the LQ frames as its condition) as programs, float32 as
stored, on the CPU and the GPU.

Ported from FlashVSR (OpenImagingLab/FlashVSR, examples/WanVSR/utils/TCDecoder.py; Apache License 2.0; no
copyright line stated: OpenImagingLab). Its decoder is TAEHV's (madebyollin/taehv; MIT License; Copyright (c) 2025
Ollin Boer Bohan). See THIRD_PARTY_NOTICES.md.

    a latent frame's input (h w, 784): the LQ frames of its 4 output frames pixel-shuffled (4, 8, 8: channel
        ((c 4 + f) 8 + y) 8 + x; the chunk's first frame repeated before them to a multiple of 4), then its 16
        latent channels; Clamp (tanh(x / 3) 3) on the host
    conv 784 -> 512 + ReLU, conv 512 -> 512 (no bias) + ReLU, 3 MemBlocks (ReLU(conv(ReLU(conv(ReLU(conv([x,
        past])))) + x)), past = the block's input of the frame before (zeros for the first), upsample x 2 (nearest),
        TGrow (1 x 1, stride 1), conv 512 -> 256 (no bias), 3 MemBlocks; upsample, TGrow (stride 2: two frames),
        conv 256 -> 128, 3 MemBlocks; upsample, TGrow (stride 2), conv 128 -> 128 + ReLU, conv 128 -> 128 + ReLU,
        conv 128 -> 3: 4 frames (8h x 8w, ~[0, 1]) a latent frame; the video's first 3 trimmed
    in the official's sequential order (parallel=False): a TGrow's first frame through every later layer before its
        second; each MemBlock's past its last frame's input (the state, carried from chunk to chunk)

The programs (one _Runner; rows pixels, columns channels): ("A", h, w) of a latent frame's "x" (h w, 784) to the
second TGrow's two frames (tg2: (4 h w, 2 x 256)); ("B", h, w, s) of frame s of those to the third TGrow's two
(tg3: (16 h w, 2 x 128)); ("C", h, w, s) of frame s of those to "y<s>" (64 h w, 3). A MemBlock's input lives in the rows
after its state (the frame before) in one buffer, so its first conv reads [x, past] through an index of 18 taps (9
of x, 9 of the state; the weight's columns in that order), and the state is the input's copy after the block. A
nearest upsampling and a TGrow's frames are read through the next conv's index (the TGrow at the lower
resolution). C's three convs in strips of STRIP output rows (the same values). The programs' scratch from one pool,
their index arrays and linears shared (one device copy each). The convs as models/realbasicvsr's (ENC_CONV_MM on the
GPU, im2col tiles on the CPU; tc: the convs of a multiple of 32 channels a tap in float16 on the tensor cores).
"""
from __future__ import annotations

import numpy as np

from reclank.encoders.runtime import _Plain, _Runner
from reclank.models.realbasicvsr.model import _B, conv_index

CH = (512, 256, 128, 128)
STRIP = 96          # the output rows of a strip of the last level's convs
LATENT = 16
COND = 768


def cond_rows(cond):
    """(T, h w, 768) of the LQ frames cond (3, F, H, W): PixelShuffle3d(4, 8, 8) (the first frame repeated before
    them to a multiple of 4), each latent frame's rows (y, x)"""
    c, F, H, W = cond.shape
    if F % 4:
        cond = np.concatenate([np.repeat(cond[:, :1], 4 - F % 4, 1), cond], 1)
        F = cond.shape[1]
    x = cond.reshape(c, F // 4, 4, H // 8, 8, W // 8, 8).transpose(1, 3, 5, 0, 2, 4, 6)
    return np.ascontiguousarray(x.reshape(F // 4, (H // 8) * (W // 8), c * 4 * 64), dtype=np.float32)


def clamp(x):
    """TAEHV's Clamp: tanh(x / 3) 3"""
    return (np.tanh(x / np.float32(3)) * np.float32(3)).astype(np.float32)


class TCDecoder:
    """The decoder of TCDecoder.safetensors' tensors W (name: float32). reset() before a video; decode(x0, cond) of
    a chunk."""

    def __init__(self, W, gpu=False, cpu_threads=None, tc=False):
        self.W, self.gpu, self.cpu_threads = W, bool(gpu), cpu_threads
        self.tc = bool(tc) and self.gpu
        self.lins = {}
        self.runner = _Runner(self._build, keep=8, keep_gpu=8)
        self.states = {}            # (h, w): the shared arrays of the programs of that size
        self.pools, self.idxs = {}, {}  # (h, w): their scratch (C's two full-resolution buffers) and index arrays
        self.ybufs = {}                 # (h, w): the host's arrays of the frames downloaded
        self.pools16 = {}               # (h, w): the scratch of the convs' float16 inputs (tc)
        self.first = True

    # ---- weights -------------------------------------------------------------------------------------

    def lin(self, name, mem=False):
        """the conv decoder.name as the linear of its im2col (columns (ky, kx, c)); mem: a MemBlock's first conv,
        its columns [x's 9 taps, past's 9 taps]; tc: its W16H rows (a tap's channels a multiple of 32: all but the
        first conv's 784). One for all the programs (one device copy)."""
        if (name, mem) in self.lins:
            return self.lins[(name, mem)]
        w = self.W["decoder." + name + ".weight"]
        o = w.shape[0]
        if mem:
            W = w.reshape(o, 2, w.shape[1] // 2, *w.shape[2:]).transpose(0, 1, 3, 4, 2).reshape(o, -1)
        else:
            W = w.transpose(0, 2, 3, 1).reshape(o, -1)
        b = self.W.get("decoder." + name + ".bias")
        out = _Plain(np.ascontiguousarray(W, np.float32))
        out.b = np.ascontiguousarray(b if b is not None else np.zeros(o), np.float32)
        out.shape = out.w.shape
        cin = w.shape[1] // (2 if mem else 1)
        if self.tc and cin % 32 == 0:
            from reclank.encoders.runtime import to_w16
            to_w16(out, "f16")
        self.lins[(name, mem)] = out
        return out

    # ---- programs ------------------------------------------------------------------------------------

    def _state(self, h, w):
        """the arrays the programs of an h x w latent share: the MemBlocks' [state, x] buffers, the TGrows'
        outputs, the input and the output"""
        if (h, w) not in self.states:
            P = [h * w, 4 * h * w, 16 * h * w, 64 * h * w]
            z = lambda r, c: np.zeros((r, c), np.float32)  # noqa: E731
            st = {"x": z(P[0], COND + LATENT), "y0": z(P[3], 3), "y1": z(P[3], 3), "tg2": z(P[1], 2 * CH[1]),
                  "tg3": z(P[2], 2 * CH[2])}
            for k, lvl in zip(range(9), (0, 0, 0, 1, 1, 1, 2, 2, 2)):
                st["mem%d" % k] = z(2 * P[lvl], CH[lvl])
            self.states[(h, w)] = st
        return self.states[(h, w)]

    def _build(self, key, gpu):
        kind, h, w = key[:3]
        if (h, w) not in self.pools:
            # (the most of a program's: B's 3 buffers of the third level; A's 5 of the first and 3 of the second;
            # C's 2 strips)
            need = max(3 * 16 * h * w * CH[2], 5 * h * w * CH[0] + 3 * 4 * h * w * CH[1],
                       2 * (STRIP + 4) * 8 * w * CH[3])
            self.pools[(h, w)] = np.zeros(need + 64 * 16, np.float32)
            self.idxs[(h, w)] = {}
            # (the convs' inputs as float16: the most of one, the third TGrow's or a third-level MemBlock's)
            self.pools16[(h, w)] = np.zeros(4096 * h * w, np.uint16) if gpu and self.tc else None
        B = _B(gpu, pool=self.pools[(h, w)], idx=self.idxs[(h, w)], pool16=self.pools16[(h, w)])
        st = self._state(h, w)
        B.env.update(st)
        getattr(self, "_prog_" + kind)(B, st, h, w, *key[3:])
        return B.finish(self.cpu_threads)

    def _memblocks(self, B, st, first, names, P, hh, ww, out):
        """the 3 MemBlocks of decoder.names (their input already in the rows P.. of mem<first>): the last's output
        into out (or the next MemBlock's input rows)"""
        C = st["mem%d" % first].shape[1]
        i3 = B.index(("c3", hh, ww), lambda: conv_index(hh, ww))
        i18 = B.index(("m18", hh, ww), lambda: np.ascontiguousarray(
            np.concatenate([np.where(i3 >= 0, i3 + P, -1), i3], 1), dtype=np.int32))
        t1, t2 = B.buf(P, C), B.buf(P, C)
        for j, nm in enumerate(names):
            mb = "mem%d" % (first + j)
            x = B.name(st[mb][P:])
            dst = B.name(st["mem%d" % (first + j + 1)][P:]) if j < 2 else out
            B.conv(mb, C, self.lin(nm + ".conv.0", mem=True), i18, out=t1, slope=0.0)
            B.conv(t1, C, self.lin(nm + ".conv.2"), i3, out=t2, slope=0.0)
            B.conv(t2, C, self.lin(nm + ".conv.4"), i3, out=dst, res=x, slope=0.0)
            B.f("set", B.name(st[mb][:P]), ("copy", x))           # (the state: this frame's input)

    def _up_index(self, h, w, stride, s):
        """the 3 x 3 im2col at 2h x 2w of a TGrow's frame s (rows (pixel stride + s) of its h x w output), nearest-
        upsampled"""
        return conv_index(2 * h, 2 * w, src=lambda y, x: ((y // 2) * w + x // 2) * stride + s)

    def _prog_A(self, B, st, h, w):
        P0, P1 = h * w, 4 * h * w
        i3 = B.index(("c3", h, w), lambda: conv_index(h, w))
        c1 = B.conv("x", COND + LATENT, self.lin("1"), i3, slope=0.0)
        B.conv(c1, CH[0], self.lin("3"), i3, out=B.name(st["mem0"][P0:]), slope=0.0)
        u0 = B.buf(P0, CH[0])
        self._memblocks(B, st, 0, ("5", "6", "7"), P0, h, w, u0)
        i1 = B.index(("c1", h, w), lambda: np.arange(P0, dtype=np.int32).reshape(P0, 1))
        tg1 = B.conv(u0, CH[0], self.lin("9.conv"), i1)
        B.conv(tg1, CH[0], self.lin("10"), B.index(("u1", h, w), lambda: self._up_index(h, w, 1, 0)),
               out=B.name(st["mem3"][P1:]))
        u1 = B.buf(P1, CH[1])
        self._memblocks(B, st, 3, ("11", "12", "13"), P1, 2 * h, 2 * w, u1)
        i1b = B.index(("c1", 2 * h, 2 * w), lambda: np.arange(P1, dtype=np.int32).reshape(P1, 1))
        B.conv(u1, CH[1], self.lin("15.conv"), i1b, out="tg2")

    def _prog_B(self, B, st, h, w, s):
        P2 = 16 * h * w
        tg = B.name(st["tg2"].reshape(-1, CH[1]))
        B.conv(tg, CH[1], self.lin("16"), B.index(("u2", s), lambda: self._up_index(2 * h, 2 * w, 2, s)),
               out=B.name(st["mem6"][P2:]))
        u2 = B.buf(P2, CH[2])
        self._memblocks(B, st, 6, ("17", "18", "19"), P2, 4 * h, 4 * w, u2)
        i1 = B.index(("c1", 4 * h, 4 * w), lambda: np.arange(P2, dtype=np.int32).reshape(P2, 1))
        B.conv(u2, CH[2], self.lin("21.conv"), i1, out="tg3")

    def _prog_C(self, B, st, h, w, s):
        """in strips of STRIP output rows: conv 22 over the strip's rows and 2 more each side, conv 24 over 1 more
        each side (their indexes the image's, into the strip's rows), conv 26 (the same values as the whole
        image's)"""
        H, W_ = 8 * h, 8 * w
        tg = B.name(st["tg3"].reshape(-1, CH[2]))
        u3 = B.index(("u3", s), lambda: self._up_index(4 * h, 4 * w, 2, s))
        i3 = B.index(("c3", H, W_), lambda: conv_index(H, W_))
        sa, sb = B.buf((STRIP + 4) * W_, CH[3]), B.buf((STRIP + 2) * W_, CH[3])
        l22, l24, l26 = self.lin("22"), self.lin("24"), self.lin("26")
        for y0 in range(0, H, STRIP):
            y1 = min(H, y0 + STRIP)
            ya0, ya1, yb0, yb1 = max(0, y0 - 2), min(H, y1 + 2), max(0, y0 - 1), min(H, y1 + 1)

            def local(idx, r0, r1, base):
                return np.ascontiguousarray(np.where(idx[r0 * W_:r1 * W_] >= 0, idx[r0 * W_:r1 * W_] - base * W_, -1),
                                            dtype=np.int32)
            a = B.name(B.env[sa][:(ya1 - ya0) * W_])
            b = B.name(B.env[sb][:(yb1 - yb0) * W_])
            B.conv(tg, CH[2], l22, B.index(("s22", s, y0), lambda: np.ascontiguousarray(u3[ya0 * W_:ya1 * W_])),
                   out=a, slope=0.0)
            B.conv(a, CH[3], l24, B.index(("s24", y0), lambda: local(i3, yb0, yb1, ya0)), out=b, slope=0.0)
            B.conv(b, CH[3], l26, B.index(("s26", y0), lambda: local(i3, y0, y1, yb0)),
                   out=B.name(st["y%d" % s][y0 * W_:y1 * W_]))

    # ---- running -------------------------------------------------------------------------------------

    def _run(self, key, inputs=(), outputs=()):
        prog, g = self.runner.get(key, self.gpu)
        for nm, a in inputs:
            prog.names[nm][...] = a
            if g is not None:
                g.upload(nm)
        if g is not None:
            g.run()
            for nm in outputs:
                g.download(nm)
        else:
            prog.run()
        return [prog.names[nm] for nm in outputs]

    def reset(self, h, w):
        """a new video of h x w latents: the MemBlocks' states zeros, the first chunk trimmed"""
        st = self._state(h, w)
        for k in range(9):
            st["mem%d" % k][...] = 0
            if self.gpu:            # (through a program that reads it: A levels 0 and 1, B level 2)
                prog, g = self.runner.get(("A", h, w) if k < 6 else ("B", h, w, 0), self.gpu)
                g.upload("mem%d" % k)
        self.first = True

    @staticmethod
    def cond_rows(cond):
        """decode's cs of the LQ frames cond (3, F, 8 h, 8 w) (a host step, on any thread): each latent frame's
        pixel shuffle clamped (T, h w, 768)"""
        from concurrent.futures import ThreadPoolExecutor

        from reclank.models.media import widen
        c, F, H, W = cond.shape
        if F % 4:
            cond = np.concatenate([np.repeat(cond[:, :1], 4 - F % 4, 1), cond], 1)
        T = cond.shape[1] // 4
        cs = np.empty((T, (H // 8) * (W // 8), c * 4 * 64), np.float32)

        def one(t):                     # (a latent frame's: cond_rows of its 4 frames)
            cs[t] = clamp(cond_rows(cond[:, 4 * t:4 * t + 4])[0])
        with ThreadPoolExecutor(min(T, 8), initializer=widen) as pool:
            list(pool.map(one, range(T)))
        return cs

    def decode(self, x0, cond=None, cs=None, stack=True):
        """The frames (4 T, 3, 8 h, 8 w) in [-1, 1] (x 2 - 1 of the decoder's; the video's first 3 trimmed) of a
        chunk's latents x0 (16, T, h, w) and its LQ frames cond (3, F, 8 h, 8 w) in [-1, 1] (or cs: their
        cond_rows). Each latent frame's input (its pixel shuffle, the latent, the clamp: the same values) made on
        threads while the programs of the ones before run, each frame's output on threads after its program; on the
        GPU each program queued before the one before's output is read. stack False: a list of the frames."""
        C, T, h, w = x0.shape
        if cs is None:
            cs = self.cond_rows(cond)
        assert cs.shape == (T, h * w, COND), (cs.shape, T)

        def prep(t):
            x = np.empty((h * w, COND + LATENT), np.float32)
            x[:, :COND] = cs[t]
            x[:, COND:] = clamp(x0[:, t]).reshape(LATENT, -1).T
            return x

        def frame(y):
            return y.T.reshape(3, 8 * h, 8 * w) * np.float32(2) - np.float32(1)
        from concurrent.futures import ThreadPoolExecutor

        from reclank.models.media import widen
        out = []
        with ThreadPoolExecutor(min(T, 6), initializer=widen) as pool:
            xs = [pool.submit(prep, t) for t in range(T)]
            if not self.gpu:
                for t in range(T):
                    self._run(("A", h, w), [("x", xs[t].result())])
                    for s in (0, 1):
                        self._run(("B", h, w, s))
                        for s2 in (0, 1):
                            out.append(pool.submit(frame, self._run(("C", h, w, s2), outputs=("y%d" % s2,))[0].copy()))
            else:
                # each program queued before the output of the one before is downloaded (C's frames into y0 and
                # y1 in turn: the next program never writes the one being read), into two host arrays in turn
                # whose pages are there (a new array's take their faults in the copy), each free again when
                # its frame is made
                ys = self.ybufs.setdefault((h, w), [np.zeros((64 * h * w, 3), np.float32) for _ in range(2)])
                busy = [None, None]

                def fetch(p):
                    prog_, i, g_ = p
                    if busy[i] is not None:
                        busy[i].result()
                    g_.mirror.buffer_of(prog_.names["y%d" % i]).download(ys[i])
                    busy[i] = pool.submit(frame, ys[i])
                    out.append(busy[i])
                pend = None
                for t in range(T):
                    prog, g = self.runner.get(("A", h, w), True)
                    g.mirror.buffer_of(prog.names["x"]).upload(xs[t].result())
                    xs[t] = None
                    for key in [("A", h, w)] + [k for s in (0, 1) for k in
                                                [("B", h, w, s), ("C", h, w, 0), ("C", h, w, 1)]]:
                        prog, g = self.runner.get(key, True)
                        g.run()
                        if pend is not None:
                            fetch(pend)
                        pend = (prog, key[3], g) if key[0] == "C" else None
                fetch(pend)
            out = [f.result() for f in out]
        if self.first:
            out = out[3:]
            self.first = False
        return out if not stack else np.stack(out)

    def release(self):
        if self.gpu:
            self.runner.release(keep_host=True)
