"""FlashVSR v1.1 (OpenImagingLab/FlashVSR, Apache 2.0): one-step streaming diffusion video super-resolution, its
Tiny-Long pipeline (examples/WanVSR/infer_flashvsr_v1.1_tiny_long_video.py, diffsynth/pipelines/
flashvsr_tiny_long.py) on the CPU and the GPU: the LQ projector (lq_proj.py), the DiT (dit.py), the TCDecoder
(tcdecoder.py).

Ported from FlashVSR (OpenImagingLab/FlashVSR; Apache License 2.0; no copyright line stated: OpenImagingLab). See
THIRD_PARTY_NOTICES.md.

    the LQ video: each frame upscaled to the output's H x W (multiples of 128) by PIL's BICUBIC (the script's
        upscale_then_center_crop when H x W is its scaled size, no crop), / 255 x 2 - 1; the last frame repeated to
        F = 8 n + 1 frames, at least the frames + 4 (the script: at most the frames + 4, its last ones dropped) and 25
    the noise (16, (F - 1) / 4, H / 8, W / 8): each latent frame's own (NumPy's normal of the seed and its index), or
        given
    the chunks (the pipeline's loop, (F - 1) / 8 - 2 of them): the first of 6 latent frames, its LQ condition the
        projector's of the clips [0, 1), [1, 5) .. [21, 25) (the first gives none); chunk c of 2, its condition of
        the clips [8 c + 17, 8 c + 21) and [8 c + 21, 8 c + 25); x0 = noise - the DiT's output; the TCDecoder of x0
        and the LQ frames [0, 21) (the first), then [8 c + 13, 8 c + 21); the colour fix (AdaIN: each frame's
        channels the mean and std (biased, + 1e-5) of its LQ frame's), clipped to [-1, 1]
    the output: (x + 1) 127.5 clipped to [0, 255], truncated to uint8 (tensor2video), its first frames as many as
        the input's

On the GPU the DiT's matrices and self-attention and the LQ projector's and TCDecoder's convs in float16 on the
tensor cores by default (tc: x, q and the attention's weights float16, float32 sums; the official runs all in
bfloat16), its keys and values float16; RECLANK_FLASHVSR_TC=0: float32 products (the DiT's matrices F20). The host's
steps (the next chunk's LQ frames and pixel shuffles, the colour fix) on threads while the GPU runs.
"""
from __future__ import annotations

import os
import threading
import time

import numpy as np

from reclank.encoders.runtime import RESIDENCY
from reclank.formats.safetensors import SafeTensors
from reclank.models.flashvsr.dit import FlashDiT
from reclank.models.flashvsr.lq_proj import LQProj
from reclank.models.flashvsr.tcdecoder import TCDecoder


def lq_video(frames, H, W):
    """(3, t, H, W) float32 in [-1, 1] of frames (t, h, w, 3) uint8: PIL's BICUBIC to W x H, / 255 x 2 - 1"""
    from PIL import Image
    out = np.empty((3, len(frames), H, W), np.float32)
    for i, f in enumerate(frames):
        img = Image.fromarray(np.asarray(f, np.uint8))
        if img.size != (W, H):
            img = img.resize((W, H), Image.BICUBIC)
        x = np.asarray(img, np.uint8).astype(np.float32)
        out[:, i] = (x / np.float32(255.0) * np.float32(2.0) - np.float32(1.0)).transpose(2, 0, 1)
    return out


class LQFrames:
    """The LQ video of frames (t, h, w, 3) uint8 at H x W (lq_video), its frames made as they are read ([:, a:b] of
    the padded F frames: the last repeated) and kept while the chunks may read them again (at most 40 before the
    last read; the reads from any thread)"""

    def __init__(self, frames, F, H, W):
        self.frames, self.F, self.H, self.W = frames, F, H, W
        self.shape = (3, F, H, W)
        self.done = {}
        self.lock = threading.Lock()

    def __getitem__(self, key):
        _, sl = key
        a, b, _ = sl.indices(self.F)
        with self.lock:
            todo = [i for i in range(a, b) if i not in self.done]
            if todo:                    # (PIL lets the GIL go in its resize: on threads)
                from concurrent.futures import ThreadPoolExecutor

                from reclank.models.media import widen
                with ThreadPoolExecutor(min(len(todo), 16), initializer=widen) as pool:
                    made = list(pool.map(lambda i: lq_video(self.frames[min(i, len(self.frames) - 1)][None], self.H,
                                                            self.W)[:, 0], todo))
                self.done.update(zip(todo, made))
            for i in [i for i in self.done if i < b - 40]:
                del self.done[i]
            got = [self.done[i] for i in range(a, b)]
        return np.stack(got, 1)


def to_uint8(video):
    """(n, H, W, 3) uint8 of frames (3, n, H, W) in [-1, 1]: tensor2video's ((x + 1) 127.5 clipped to [0, 255],
    truncated), a frame a thread"""
    from concurrent.futures import ThreadPoolExecutor

    from reclank.models.media import widen
    out = np.empty((video.shape[1],) + video.shape[2:] + (3,), np.uint8)

    def one(i):
        out[i] = ((np.clip(video[:, i], -1, 1) + np.float32(1)) * np.float32(127.5)).clip(0, 255).astype(
            np.uint8).transpose(1, 2, 0)
    with ThreadPoolExecutor(min(16, max(1, video.shape[1])), initializer=widen) as pool:
        list(pool.map(one, range(video.shape[1])))
    return out


def padded_frames(t):
    """F: the frames of the LQ video of t frames (8 n + 1, at least t + 4 and 25)"""
    return max(25, -(-(t + 4 - 1) // 8) * 8 + 1)


COLOR_THREADS = int(os.environ.get("RECLANK_FLASHVSR_COLOR_THREADS", "16"))


def adain(content, style, eps=1e-5):
    """_adain of frames (3, f, H, W): each frame's channels content's normalized to style's mean and std (var biased,
    + eps); the frames on COLOR_THREADS threads (NumPy lets the GIL go; each on the CPUs of the start: the main
    thread is bound to one), the same values"""
    f = content.shape[1]
    if f > 1 and COLOR_THREADS > 1:
        from concurrent.futures import ThreadPoolExecutor

        from reclank.models.media import widen
        out = np.empty(content.shape, np.float32)

        def one(i):
            out[:, i] = adain(content[:, i:i + 1], style[:, i:i + 1], eps)[:, 0]
        with ThreadPoolExecutor(min(f, COLOR_THREADS), initializer=widen) as pool:
            list(pool.map(one, range(f)))
        return out
    c = content.reshape(3, content.shape[1], -1).astype(np.float32)
    s = style.reshape(3, style.shape[1], -1).astype(np.float32)
    cm, sm = c.mean(-1, keepdims=True), s.mean(-1, keepdims=True)
    cs = np.sqrt(c.var(-1, keepdims=True) + np.float32(eps))
    ss = np.sqrt(s.var(-1, keepdims=True) + np.float32(eps))
    return ((c - cm) / cs * ss + sm).reshape(content.shape).astype(np.float32)


class FlashVSR:
    """A FlashVSR-v1.1 directory (the DMD checkpoint, LQ_proj_in, TCDecoder and posi_prompt as safetensors:
    scripts/pt_to_safetensors.py of the .ckpt and .pth). upscale_video(frames, size) gives the output."""

    def __init__(self, path, gpu=False, cpu_threads=None, tc=None):
        self.gpu = bool(gpu)
        # tc (the default; RECLANK_FLASHVSR_TC=0: float32 products): the DiT's matrices ("dit") and self-attention
        # ("attn": q and its weights float16) and the LQ projector's ("lq") and TCDecoder's ("dec") convs in float16 on
        # the tensor cores (W16H rows, x float16, float32 sums; the official's are all bfloat16). Some of them: a list
        # of those (or RECLANK_FLASHVSR_TC=dit,dec)
        if tc is None:
            tc = os.environ.get("RECLANK_FLASHVSR_TC", "1")
        parts = {"dit", "attn", "lq", "dec"} if tc in (True, "1") else set() if tc in (False, None, "0", "") else \
            set(tc.split(",") if isinstance(tc, str) else tc)
        assert parts <= {"dit", "lq", "dec", "attn"}, parts
        self.tc = parts if self.gpu else set()
        self.dit = FlashDiT(path, gpu=gpu, cpu_threads=cpu_threads, tc="dit" in self.tc, attn_tc="attn" in self.tc)
        rd = lambda n: (lambda S: {k: np.array(S.get(k), np.float32) for k in S.names()})(  # noqa: E731
            SafeTensors(os.path.join(path, n)))
        self.proj = LQProj(rd("LQ_proj_in.safetensors"), gpu=gpu, cpu_threads=cpu_threads, tc="lq" in self.tc)
        self.dec = TCDecoder(rd("TCDecoder.safetensors"), gpu=gpu, cpu_threads=cpu_threads, tc="dec" in self.tc)

    def runners(self):
        return [self.dit.runner, self.proj.runner, self.dec.runner]

    def upscale_video(self, frames, size, seed=0, noise=None, lq=None, color_fix=True, progress=None, parts=None):
        """The output (t, H, W, 3) uint8 of frames (t, h, w, 3) uint8 at size (H, W) (multiples of 128). noise: the
        noise (16, (F - 1) / 4, H / 8, W / 8) in place of the seed's; lq: the LQ video (3, F, H, W) in place of the
        frames' (then frames is None). parts: a dict that gets the float frames before the uint8 (video, (3, t, H,
        W)). self.times: the stages' seconds."""
        with RESIDENCY.use(*self.runners()):
            return self._upscale(frames, size, seed, noise, lq, color_fix, progress, parts)

    def _upscale(self, frames, size, seed, noise, lq, color_fix, progress, parts):
        H, W = size
        assert H % 128 == 0 and W % 128 == 0, "FlashVSR: an output of multiples of 128"
        times = {}
        clock = [time.time()]

        def lap(stage):
            now = time.time()
            times[stage] = times.get(stage, 0.0) + now - clock[0]
            clock[0] = now
        if lq is None:
            t = len(frames)
            F = padded_frames(t)
            lq = LQFrames(frames, F, H, W)
        else:
            F = lq.shape[1]
            t = F - 4
        assert (F - 1) % 8 == 0 and F >= 25
        n_lat = (F - 1) // 4
        hh, ww = H // 16, W // 16
        if noise is None:
            noise = np.stack([np.random.default_rng([seed, j]).standard_normal((16, H // 8, W // 8)).astype(np.float32)
                              for j in range(n_lat)], 1)
        lap("prep")
        chunks = (F - 1) // 8 - 2

        from concurrent.futures import ThreadPoolExecutor

        from reclank.models.media import widen

        def plan(c):
            """chunk c's LQ clips, its latents' noise, its LQ frames [pre, cur)"""
            if c == 0:
                return [(0, 1)] + [(4 * i - 3, 4 * i + 1) for i in range(1, 7)], noise[:, :6], 0, 21
            return [(8 * c + 17 + 4 * i, 8 * c + 21 + 4 * i) for i in range(2)], noise[:, 4 + 2 * c:6 + 2 * c], \
                8 * c + 13, 8 * c + 21

        def inputs(c):
            """(on a thread) the host's inputs of chunk c: its clips' rows (the projector's), the DiT's, its LQ
            frames' (the decoder's), on threads"""
            clips, lat, pre, cur = plan(c)
            with ThreadPoolExecutor(4, initializer=widen) as pool:
                rows = [pool.submit(lambda a, b: self.proj.rows(lq[:, a:b]), a, b) for a, b in clips]
                dpre = pool.submit(self.dit.inputs, lat, c)
                cs = self.dec.cond_rows(lq[:, pre:cur])
                return [r.result() for r in rows], dpre.result(), cs

        def finish(fr, pre, cur):
            """(on a thread) a chunk's frames (a list of (3, H, W)): the colour fix, the clip or the uint8"""
            fr = np.stack(fr, 1)                                                    # (3, n, H, W)
            if color_fix:
                fr = adain(fr, lq[:, pre:cur])
            return np.clip(fr, -1, 1) if parts is not None else to_uint8(fr)
        # the host's steps on threads while the GPU runs: the next chunk's inputs, the chunks' colour fix
        with ThreadPoolExecutor(1, initializer=widen) as ahead, ThreadPoolExecutor(1, initializer=widen) as behind:
            self.dit.start(hh, ww, out_pixels=H * W)
            nxt, out = ahead.submit(inputs, 0), []
            self.proj.reset()
            self.dec.reset(H // 8, W // 8)
            for c in range(chunks):
                _, lat, pre, cur = plan(c)
                rows, dpre, cs = nxt.result()
                if c + 1 < chunks:
                    nxt = ahead.submit(inputs, c + 1)
                lap("inputs")
                conds = [self.proj.push_rows(x) for x in rows]
                cond = [x for x in conds if x is not None]                         # (each latent frame's)
                lap("lq_proj")
                x0 = (lat - self.dit.chunk(lat, cond, pre=dpre)).astype(np.float32)
                lap("dit")
                out.append(behind.submit(finish, self.dec.decode(x0, cs=cs, stack=False), pre, cur))
                lap("decode")
                if progress is not None:
                    progress("chunk", c + 1, chunks)
            out = [f.result() for f in out]
            lap("color")
        self.times = times
        if parts is None:
            return np.concatenate(out)[:t]
        video = np.concatenate(out, 1)[:, :t]
        parts["video"] = video
        return to_uint8(video)

    def release(self):
        for m in (self.dit, self.proj, self.dec):
            m.release()
