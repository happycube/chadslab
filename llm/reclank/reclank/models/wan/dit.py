"""Wan 2.2's DiT (diffusers' WanTransformer3DModel; TI2V-5B: 30 blocks, width 3072, 24 heads of
128, the FFN 14336, patches of 1 x 2 x 2 of 48 latent channels) as programs: the text's keys and
values once a prompt, and a step of the video's rows.

Ported from diffusers (src/diffusers/models/transformers/transformer_wan.py; Apache License 2.0;
Copyright 2025 The Wan Team and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    the step's vectors, on the host in float32: the timestep's sinusoid (256, [cos, sin]), the
        time embedder (linear, SiLU, linear: temb), time_proj of SiLU(temb) (6 x 3072), and each
        block's six (its scale_shift_table plus those): the shift, 1 + scale and gate of the
        self-attention and of the FFN; the last norm's shift and 1 + scale (temb plus the model's
        scale_shift_table). The pipeline's expand_timesteps gives each token its timestep, and
        text to video gives them all the same: one set of vectors for all rows.
    the text (prefix program, the prompt's and the negative prompt's 512 rows each, zeros past
        the prompt's tokens as the pipeline pads): the text embedder (linear, tanh GELU,
        linear), then each block's cross-attention keys (to_k, the RMS norm of all heads) and
        values (to_v): the caches kc, vc (blocks, rows, width)
    the video rows (step program): the patch embedding (Conv3d 1 x 2 x 2 as a 3072 x 192
        linear of each patch's (channel, y, x) values, patched on the host), then a block: LN
        (no weights) times 1 + scale plus shift (ENC_LNORM with the step's vectors as its
        weights), q k v, the RMS norms of q and k across the heads (q's weight times hd^-1/2),
        the 3D RoPE, the attention of all rows, the output, times the gate, + x; LN with its
        weights (norm2, cross_attn_norm), q (the norm, the scale), the attention over the
        prompt's 512 keys and values (copied from the cache into the key buffers), the output,
        + x; LN, modulation, the FFN (linear, tanh GELU, linear), times the gate, + x; then LN,
        the last modulation, proj_out (192: (y, x, channel) of a patch), unpatched on the host

Classifier-free guidance runs the step program twice, once a cache (which: 0 the prompt's, 1
the negative's): two programs that share their buffers (on the GPU one device copy). The
attention: float32 on the GPU (ENC_ATTN_TILE; x2, RECLANK_WAN_X2=1: ENC_ATTN_X2, q, k, v as two
bfloat16 planes), ENC_ATTN_RANGE on the CPU; the cross attention's queries and keys in buffers of
max(rows, 512) rows (the record converts as many keys as queries), the keys past 512 not read.

RoPE (WanRotaryPosEmbed): the pairs of a head (interleaved) at the angles of three axes, frames,
height and width, of 22, 21 and 21 pairs (44, 42, 42 values; theta 10000) of the token's (f, y,
x); float64 angles and their cos and sin rounded to float32, as diffusers' tables.

The matrices: on the GPU F20 (encoders.runtime.F20Lins: each matrix as it is made, its float32 W
and the checkpoint's pages dropped; 16 significant bits a value, float32 products); on the CPU
the checkpoint's float32 W as stored, or F20 (f20). The biases, the norms' weights, the
scale_shift_tables and the time embedder float32 as stored.

T2V-A14B (two experts of 40 blocks, width 5120, 40 heads, FFN 13824, 16 latent channels; a
scalar timestep: the same vectors): an expert's F20 is 35.7 GB, more than the card, so its blocks
are streamed (stream; by default when the blocks' F20 passes STREAM_GB): each block's 10 F20
matrices in one registered host array (cudaHostRegister: the copy worker's pull kernel reads it,
~12 GB/s), and the programs' linears of block j read slot j % 2 of two slots on the GPU (a host
placeholder of a block's bytes whose device copy is the slot); GP_FETCH j copies block j's
matrices (the step program's 8, the text program's k2 and v2) into its slot after FETCH_DONE of
block j - 2, so block j + 1 comes in while block j runs (a call of 30 GB of copies in 2.5 s
beside 23 s of compute at 480 x 832 x 17). The patch, text and output linears and the small
tensors stay resident. A second expert (share) joins the first's runner: the same activations,
text caches and slots (one device copy), its own programs (keys by expert). keeper: the F20 rows
and the small tensors held by a weightkeeper (models/wan/keep.py), mapped, not loaded: the same
bytes as a conversion here.
"""
from __future__ import annotations

import ctypes
import json
import mmap
import os
import re
import threading

import numpy as np

from reclank.encoders.runtime import F20Lins, _EncCompiler, _Plain, _Runner
from reclank.formats.safetensors import Shards
from reclank.ir import program as P
from reclank.models.flux2.dit import rope_table, timestep_sinusoid
from reclank.runtime.threads import team_threads

TEXT = 512          # the pipeline's text rows (max_sequence_length, zeros past the prompt)
MLP_ROWS = 4096     # the rows of the FFN's products at a time (its buffers)
# the matrices of a block; those a step reads and those the text program reads (streamed)
MATS = ("q", "k", "v", "o", "q2", "k2", "v2", "o2", "up", "down")
STEP_MATS = ("q", "k", "v", "o", "q2", "o2", "up", "down")
TEXT_MATS = ("k2", "v2")
# stream the blocks when their F20 passes this (GB; the 5B's 12.1 resident, the A14B's 35.6 not);
# RECLANK_WAN_STREAM=1 / 0 forces it
STREAM_GB = 16.0
ALIGN = 4096
F20_BYTES = 81 / 32     # (a value's bytes in F20: 32 values in 81)
_REGISTERED = set()     # (address, bytes) of the host memory registered with CUDA here


def _silu(x):
    return (x / (np.float32(1) + np.exp(-x))).astype(np.float32)


def patchify(lat):
    """The DiT's input rows (f h/2 w/2, 4 C) of latents (C, f, h, w): each patch's (channel, y,
    x) values (the patch embedding's Conv3d of kernel 1 x 2 x 2)."""
    C, f, h, w = lat.shape
    x = lat.reshape(C, f, h // 2, 2, w // 2, 2).transpose(1, 2, 4, 0, 3, 5)
    return np.ascontiguousarray(x.reshape(f * (h // 2) * (w // 2), C * 4), dtype=np.float32)


def unpatchify(rows, C, f, hh, ww):
    """The latents (C, f, 2 hh, 2 ww) of the DiT's output rows (f hh ww, 4 C: (y, x, channel))."""
    x = rows.reshape(f, hh, ww, 2, 2, C).transpose(5, 0, 1, 3, 2, 4)
    return np.ascontiguousarray(x.reshape(C, f, 2 * hh, 2 * ww), dtype=np.float32)


class _Prefetch:
    """A thread that reads the shards in the order of their blocks (sequential reads of 64 MB, at
    most PREFETCH_AHEAD blocks ahead of the block being made: done(i)), so that the map's faults
    find the page cache (an HDD's sequential rate, not its rate under the faults: an A14B expert
    918 -> 663 s from /space); a shard's pages dropped (POSIX_FADV_DONTNEED) once its blocks are
    made."""

    AHEAD = 6

    def __init__(self, W, path):
        first, last = {}, {}
        for n, st in W.where.items():
            m = re.match(r"blocks\.(\d+)\.", n)
            b = int(m.group(1)) if m else -1
            first[st.path] = min(first.get(st.path, 1 << 30), b)
            last[st.path] = max(last.get(st.path, -1), b)
        self.first, self.last = first, last
        self.files = sorted(first, key=lambda p: (first[p], p))
        self.cur, self.stop = -1, False
        self.cv = threading.Condition()
        self.t = threading.Thread(target=self._run, name="wan prefetch", daemon=True)
        self.t.start()

    def _run(self):
        buf = bytearray(64 << 20)
        for p in self.files:
            with self.cv:
                self.cv.wait_for(lambda: self.stop or self.cur >= self.first[p] - self.AHEAD)
                if self.stop:
                    return
            with open(p, "rb", buffering=0) as fh:
                while not self.stop and fh.readinto(buf):
                    pass

    @staticmethod
    def _drop(p):
        fd = os.open(p, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)

    def done(self, i):
        with self.cv:
            self.cur = i
            self.cv.notify_all()
        for p in self.files:
            if self.last[p] == i:
                self._drop(p)

    def close(self):
        with self.cv:
            self.stop = True
            self.cv.notify_all()
        self.t.join()
        for p in self.files:
            self._drop(p)


class WanDiT:
    """The DiT of a transformer directory (config.json and its safetensors). prefix(emb, nemb)
    runs the text rows into the caches; velocity(latents, t, which) runs a step."""

    def __init__(self, path, gpu=False, cpu_threads=None, f20=None, x2=None, stream=None, share=None,
                 keeper=None, log=print):
        """stream: the blocks streamed (None: when their F20 passes STREAM_GB on the GPU); share:
        the first expert of a pair (its runner, activations, caches, slots); keeper: a
        weightkeeper's Weights holding this transformer (keep.KeptExpert), else the shards."""
        cfg = json.load(open(os.path.join(path, "config.json")))
        assert list(cfg["patch_size"]) == [1, 2, 2] and cfg["qk_norm"] == "rms_norm_across_heads"
        assert cfg["cross_attn_norm"] and cfg.get("image_dim") is None and cfg.get("added_kv_proj_dim") is None
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        w_, n_ = cfg["num_attention_heads"] * cfg["attention_head_dim"], cfg["num_layers"]
        est = n_ * (8 * w_ * w_ + 2 * w_ * cfg["ffn_dim"]) * F20_BYTES / 1e9
        env_s = os.environ.get("RECLANK_WAN_STREAM")
        if stream is None:
            stream = env_s == "1" if env_s in ("0", "1") else self.gpu and est > STREAM_GB
        self.stream = bool(stream)
        assert not self.stream or self.gpu, "the streamed blocks are a GPU mode"
        self.keeper = keeper
        if keeper is None:
            self.W = W = Shards(path, "diffusion_pytorch_model.safetensors.index.json")
            if self.stream:
                W.prefetch = _Prefetch(W, path)
        else:
            self.W = W = keeper
        f20 = self.gpu if f20 is None else bool(f20)
        assert f20 or not self.gpu, "the GPU's DiT is F20 (float32 W: 20 GB)"
        self.f20 = F20Lins(self.gpu, "wan dit") if f20 and keeper is None else None
        # the attention on the GPU: float32 (ENC_ATTN_TILE), or ENC_ATTN_X2 by x2 (None: RECLANK_WAN_X2=1)
        self.x2 = os.environ.get("RECLANK_WAN_X2", "0") == "1" if x2 is None else bool(x2)
        self.heads, self.hd = cfg["num_attention_heads"], cfg["attention_head_dim"]
        self.width = w = self.heads * self.hd
        self.inter = cfg["ffn_dim"]
        self.eps = float(cfg["eps"])
        self.in_ch, self.out_ch = cfg["in_channels"], cfg["out_channels"]
        self.text_dim, self.freq = cfg["text_dim"], cfg["freq_dim"]
        hd = self.hd
        hw = 2 * (hd // 6)
        self.axes = [hd - 2 * hw, hw, hw]       # frames, height, width (44, 42, 42)
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        self.f = f
        sq = np.float32(hd ** -0.5)
        e = "condition_embedder."
        self.t1, self.t1b = f(e + "time_embedder.linear_1.weight"), f(e + "time_embedder.linear_1.bias")
        self.t2, self.t2b = f(e + "time_embedder.linear_2.weight"), f(e + "time_embedder.linear_2.bias")
        self.tp, self.tpb = f(e + "time_proj.weight"), f(e + "time_proj.bias")
        self.sst_out = f("scale_shift_table").reshape(2, w)
        L = self.linear
        self.patch = L("patch_embedding")
        self.te1, self.te2 = L(e + "text_embedder.linear_1"), L(e + "text_embedder.linear_2")
        self.proj = L("proj_out")
        self.blocks, self.packed, self.offs, self.block_bytes = [], {}, None, 0
        pf = getattr(W, "prefetch", None)
        for i in range(cfg["num_layers"]):
            p = "blocks.%d." % i
            a1, a2 = p + "attn1.", p + "attn2."
            self.blocks.append(dict(
                q=L(a1 + "to_q"), k=L(a1 + "to_k"), v=L(a1 + "to_v"), o=L(a1 + "to_out.0"),
                nq=(f(a1 + "norm_q.weight") * sq).astype(np.float32), nk=f(a1 + "norm_k.weight"),
                q2=L(a2 + "to_q"), k2=L(a2 + "to_k"), v2=L(a2 + "to_v"), o2=L(a2 + "to_out.0"),
                nq2=(f(a2 + "norm_q.weight") * sq).astype(np.float32), nk2=f(a2 + "norm_k.weight"),
                n2w=f(p + "norm2.weight"), n2b=f(p + "norm2.bias"),
                up=L(p + "ffn.net.0.proj"), down=L(p + "ffn.net.2"),
                sst=f(p + "scale_shift_table").reshape(6, w)))
            if self.stream:
                self._pack(i)
            if pf is not None:
                pf.done(i)
        if pf is not None:
            pf.close()
        if self.f20 is not None:
            self.f20.report()
        if share is None:
            self.group = [self]
            # the text and the steps of both caches (of each expert of a pair: the older go first)
            self.runner = _Runner(self._build_any, keep=6, keep_gpu=3)
            shape = (len(self.blocks), 2 * TEXT, w)
            self.kc, self.vc = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
            self.acts = {}          # (f, hh, ww): the buffers of the steps of both caches
            self.slots = None
        else:
            assert share.gpu == self.gpu and share.stream == self.stream and len(share.blocks) == len(self.blocks)
            assert share.offs == self.offs and share.width == self.width
            self.group = share.group
            self.group.append(self)
            self.runner, self.kc, self.vc, self.acts = share.runner, share.kc, share.vc, share.acts
        self.eid = len(self.group) - 1
        if self.stream:
            log("wan dit: %d blocks streamed (%.1f MB a block, %.2f GB in host memory%s)" % (
                len(self.packed), self.block_bytes / 1e6, sum(v[1] for v in self.packed.values()) / 1e9,
                ", the weightkeeper's" if keeper is not None else ""))

    def _pack(self, i):
        """Block i's matrices in one host array (its layout self.offs: 10 F20 matrices, each at a
        multiple of ALIGN), to be registered with CUDA: self.packed[i] = (address, bytes, array).
        From the keeper, its own rows (a block's matrices side by side there: the same layout)."""
        B = self.blocks[i]
        offs, tot = {}, 0
        for m in MATS:
            assert getattr(B[m], "f20", None) is not None, "a streamed matrix that F20 refused: %d %s" % (i, m)
            offs[m] = tot
            tot += (B[m].f20.data.nbytes + ALIGN - 1) // ALIGN * ALIGN
        assert self.offs in (None, offs)
        self.offs, self.block_bytes = offs, tot
        if self.keeper is not None:
            a0 = B[MATS[0]].f20.data.ctypes.data
            assert all(B[m].f20.data.ctypes.data == a0 + offs[m] for m in MATS), "the keeper's block layout"
            self.packed[i] = (a0, tot, None)
            return
        m_ = mmap.mmap(-1, tot, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
        buf = np.frombuffer(m_, np.uint8)
        for m in MATS:
            d = B[m].f20.data
            v = buf[offs[m]:offs[m] + d.nbytes]
            v[...] = d.reshape(-1)
            B[m].f20.data = v.reshape(d.shape)
        self.packed[i] = (buf.ctypes.data, tot, buf)

    def _register(self):
        """The streamed blocks' host memory registered with CUDA (once a process): the packed
        arrays, or the keeper's segments."""
        from reclank.backends.cuda.runtime import lib
        ranges = self.keeper.segments() if self.keeper is not None else \
            [(a, n) for a, n, _b in self.packed.values()]
        for a, n in ranges:
            if (a, n) not in _REGISTERED:
                if lib().rc_gpu_host_register_rw(ctypes.c_void_p(a), n) == 0:
                    raise RuntimeError("cudaHostRegister of %.2f GB: %s" % (n / 1e9, lib().rc_gpu_last_error().decode()))
                _REGISTERED.add((a, n))

    def _build_any(self, key, gpu):
        return self.group[key[0]]._build(key[1:], gpu)

    def linear(self, name):
        """A linear (out, in) of the checkpoint (a conv as its (out, in k) matrix) with its bias;
        in F20 when f20 (its float32 W and the checkpoint's pages dropped then); the keeper's F20
        rows (or float32 W) when it holds them."""
        if self.keeper is not None:
            return self.keeper.linear(name)
        src = self.W.get(name + ".weight")
        lin = _Plain(np.ascontiguousarray(src.reshape(src.shape[0], -1), dtype=np.float32))
        lin.shape = lin.w.shape
        lin.b = np.ascontiguousarray(self.W.get(name + ".bias"), dtype=np.float32)
        if self.f20 is not None:
            self.f20.add(lin, src)
        return lin

    def _build(self, key, gpu):
        p = self.program(*key, gpu=gpu)
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    # ---- the step's vectors (host) ----------------------------------------

    def vectors(self, t):
        """The modulation vectors (name: (width,)) of timestep t (0 to 1000): each block's shift,
        1 + scale and gate of the self-attention (a<i>b, a<i>w, a<i>g) and of the FFN (f<i>b,
        f<i>w, f<i>g); the last norm's shift and 1 + scale (ob, ow)."""
        w = self.width
        e = (self.t1 @ timestep_sinusoid(t, self.freq) + self.t1b).astype(np.float32)
        temb = (self.t2 @ _silu(e) + self.t2b).astype(np.float32)
        tp = (self.tp @ _silu(temb) + self.tpb).astype(np.float32).reshape(6, w)
        out = {}
        one = np.float32(1)
        for i, B in enumerate(self.blocks):
            m = (B["sst"] + tp).astype(np.float32)
            out.update({"a%db" % i: m[0], "a%dw" % i: one + m[1], "a%dg" % i: m[2],
                        "f%db" % i: m[3], "f%dw" % i: one + m[4], "f%dg" % i: m[5]})
        sh, sc = (self.sst_out + temb).astype(np.float32)
        out.update(ob=sh, ow=one + sc)
        return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}

    def rope(self, f, hh, ww):
        """cos and sin (f hh ww, hd / 2) of the tokens (frame, y, x)."""
        pos = np.stack(np.meshgrid(np.arange(f), np.arange(hh), np.arange(ww), indexing="ij"), -1).reshape(-1, 3)
        return rope_table(pos, self.axes, 10000.0)

    # ---- the programs ------------------------------------------------------

    def program(self, kind, *key, gpu=False):
        """kind "text": the prefix program: input "txt" (2 x 512, text dim: the prompt's rows,
        then the negative prompt's), the caches kc, vc. kind "step" (key (which, f, hh, ww)): a
        step of f x hh x ww tokens over the cache of which: input "img" (tokens, 4 in channels),
        the vectors of vectors(); output "vel" (tokens, 4 out channels)."""
        w, H, hd, inter = self.width, self.heads, self.hd, self.inter
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        if kind == "text":
            R = 2 * TEXT
            c = _EncCompiler(self.eps, pack=not gpu)
            env = c.env
            env.update(txt=z(R, self.text_dim), ctx=z(R, w), tmp=z(R, w))
            c.compile(("seq", ("set", "tmp", ("enc_linear", self.te1, "txt")),
                       ("set", "tmp", ("enc_gelu", "tmp")),
                       ("set", "ctx", ("enc_linear", self.te2, "tmp"))))

            def block(i, B):
                env["kc%d" % i], env["vc%d" % i] = self.kc[i], self.vc[i]
                return [("set", "tmp", ("enc_linear", B["k2"], "ctx")),
                        ("set", "kc%d" % i, ("enc_rms", "tmp", B["nk2"], w)),
                        ("set", "vc%d" % i, ("enc_linear", B["v2"], "ctx"))]
            rng = self._blocks(c, TEXT_MATS, block)
            prog = c.p.finish()
            prog.ranges = rng
            return prog
        which, f, hh, ww = key
        N = f * hh * ww
        K = max(N, TEXT)
        c = _EncCompiler(self.eps, pack=not gpu)
        c.attn_f32 = not self.x2
        env = c.env
        acts = self.acts.get((f, hh, ww))
        if acts is None:
            while len(self.acts) >= 2:
                # (a program kept of an older size keeps its own)
                self.acts.pop(next(iter(self.acts)))
            cos, sin = self.rope(f, hh, ww)
            acts = self.acts[(f, hh, ww)] = dict(
                img=z(N, 4 * self.in_ch), vel=z(N, 4 * self.out_ch), x=z(N, w), h=z(N, w), tmp=z(N, w),
                q=z(K, w), kk=z(K, w), vv=z(K, w), o=z(K, w),
                g=z(min(N, MLP_ROWS), inter), cos=cos, sin=sin,
                alo=np.zeros(N, np.int32), ahi=np.full(N, N, np.int32),
                clo=np.zeros(K, np.int32), chi=np.full(K, TEXT, np.int32),
                **{k_: np.zeros_like(v_) for k_, v_ in self.vectors(999.0).items()})
        env.update(acts)
        env["qi"], env["ki"], env["vi"], env["oi"] = (env[n][:N] for n in ("q", "kk", "vv", "o"))
        env["kt"], env["vt"] = env["kk"][:TEXT], env["vv"][:TEXT]
        for r0 in range(0, N, MLP_ROWS):
            r1 = min(N, r0 + MLP_ROWS)
            n = "_%d_%d" % (r0, r1)
            env["h" + n], env["tmp" + n], env["g" + n] = env["h"][r0:r1], env["tmp"][r0:r1], env["g"][:r1 - r0]
        rows = slice(which * TEXT, (which + 1) * TEXT)
        c.compile(("set", "x", ("enc_linear", self.patch, "img")))

        def block(i, B):
            env["kc%d" % i], env["vc%d" % i] = self.kc[i][rows], self.vc[i][rows]
            a, m = "a%d" % i, "f%d" % i
            forms = [("set", "h", ("enc_lnorm", "x", a + "w", a + "b", w)),
                      ("set", "tmp", ("enc_linear", B["q"], "h")),
                      ("set", "qi", ("enc_rms", "tmp", B["nq"], w)),
                      ("set", "tmp", ("enc_linear", B["k"], "h")),
                      ("set", "ki", ("enc_rms", "tmp", B["nk"], w)),
                      ("set", "vi", ("enc_linear", B["v"], "h")),
                      ("enc_rope_tab", "qi", "cos", "sin", H, hd),
                      ("enc_rope_tab", "ki", "cos", "sin", H, hd),
                      ("set", "oi", ("enc_attn_range_x2", "qi", "ki", "vi", "alo", "ahi", H, hd)),
                      ("set", "tmp", ("enc_linear", B["o"], "oi")),
                      ("set", "tmp", ("enc_mul_vec", "tmp", a + "g")),
                      ("enc_add", "x", "tmp"),
                      # the cross attention: the prompt's keys and values in the first rows
                      ("set", "h", ("enc_lnorm", "x", B["n2w"], B["n2b"], w)),
                      ("set", "tmp", ("enc_linear", B["q2"], "h")),
                      ("set", "qi", ("enc_rms", "tmp", B["nq2"], w)),
                      ("set", "kt", ("copy", "kc%d" % i)),
                      ("set", "vt", ("copy", "vc%d" % i)),
                      ("set", "o", ("enc_attn_range_x2", "q", "kk", "vv", "clo", "chi", H, hd)),
                      ("set", "tmp", ("enc_linear", B["o2"], "oi")),
                      ("enc_add", "x", "tmp"),
                      ("set", "h", ("enc_lnorm", "x", m + "w", m + "b", w))]
            for r0 in range(0, N, MLP_ROWS):
                n = "_%d_%d" % (r0, min(N, r0 + MLP_ROWS))
                forms += [("set", "g" + n, ("enc_linear", B["up"], "h" + n)),
                          ("set", "g" + n, ("enc_gelu", "g" + n)),
                          ("set", "tmp" + n, ("enc_linear", B["down"], "g" + n))]
            forms += [("set", "tmp", ("enc_mul_vec", "tmp", m + "g")),
                      ("enc_add", "x", "tmp")]
            return forms
        rng = self._blocks(c, STEP_MATS, block)
        c.compile(("seq", ("set", "h", ("enc_lnorm", "x", "ow", "ob", w)),
                   ("set", "vel", ("enc_linear", self.proj, "h"))))
        prog = c.p.finish()
        prog.ranges = rng
        return prog

    def _blocks(self, c, mats, block):
        """Compile each block's forms (block(i, B)). Streamed: block j's matrices mats from slot
        j % 2, GP_FETCH j of them (the first two at the start, block j + 2 after FETCH_DONE of
        block j) and FETCH_WAIT j before the block; the ranges [(array (len(mats), 3): host,
        device, bytes), block, mats], their devices filled at each run (_fill)."""
        if not self.stream:
            for i, B in enumerate(self.blocks):
                c.compile(("seq", *block(i, B)))
            return []
        n = len(self.blocks)
        slots = self._slots()
        rng = []
        for j in range(n):
            r = np.zeros((len(mats), 3), np.int64)
            for k, m in enumerate(mats):
                r[k, 0] = self.packed[j][0] + self.offs[m]
                r[k, 2] = self.blocks[j][m].f20.data.nbytes
            rng.append(r)
        for j in range(min(2, n)):
            c.p.emit(P.FETCH, rng[j], len(mats), j, j % 2)
        for j in range(n):
            s = j % 2
            c.p.emit(P.FETCH_WAIT, j)
            c.compile(("seq", *block(j, self._proxy(j, slots[s]))))
            c.p.emit(P.FETCH_DONE, s)
            if j + 2 < n:
                c.p.emit(P.FETCH, rng[j + 2], len(mats), j + 2, s)
        return [(rng[j], j % 2, mats) for j in range(n)]

    def _slots(self):
        """The two slots' host placeholders (a block's bytes; their device copies are the slots),
        of the pair."""
        g0 = self.group[0]
        if g0.slots is None:
            g0.slots = [np.frombuffer(mmap.mmap(-1, self.block_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS),
                                      np.uint8) for _ in range(2)]
        return g0.slots

    def _proxy(self, j, slot):
        """Block j with its matrices' F20 rows read from slot."""
        from reclank.models.gemma4.e4b import _KQ
        B = dict(self.blocks[j])
        for m in MATS:
            src = B[m]
            p = _Plain(None)
            p.b, p.shape = src.b, src.shape
            p.f20 = _KQ(slot[self.offs[m]:self.offs[m] + src.f20.data.nbytes], src.f20.type, src.f20.rows,
                        src.f20.cols)
            B[m] = p
        return B

    def _fill(self, prog):
        """The device addresses of the ranges of a streamed program: its slots' device copies (in the
        runner's mirror; new after a release)."""
        if not prog.ranges:
            return
        mir = self.runner.mirror
        dst = [mir.translate(a.ctypes.data) for a in self.group[0].slots]
        for arr, s, mats in prog.ranges:
            for k, m in enumerate(mats):
                arr[k, 1] = dst[s] + self.offs[m]

    def _get(self, key):
        """The runner's program of this expert (and its ranges' devices)."""
        if self.stream and not getattr(self, "_registered", False):
            self._register()
            self._registered = True
        prog, g = self.runner.get((self.eid,) + key, self.gpu)
        if g is not None:
            self._fill(prog)
        return prog, g

    def prefix(self, emb, nemb):
        """Run the prefix program of the prompt's rows emb and the negative prompt's nemb (512,
        text dim each, the pipeline's prompt_embeds): the caches of which 0 and 1. nemb None: the
        prompt's only (no guidance; cache 1 zeros)."""
        prog, g = self._get(("text",))
        txt = prog.names["txt"]
        txt[:TEXT] = emb
        txt[TEXT:] = 0 if nemb is None else nemb
        if g is not None:
            g.upload("txt")
            g.run()
        else:
            prog.run()

    def velocity(self, lat, t, which=0):
        """The DiT's output (C, f, h, w) of latents lat (C, f, h, w) at timestep t (0 to 1000)
        over the text of cache which (0 the prompt's, 1 the negative prompt's)."""
        C, f, h, w = lat.shape
        hh, ww = h // 2, w // 2
        prog, g = self._get(("step", which, f, hh, ww))
        inputs = dict(img=patchify(lat))
        inputs.update(self.vectors(t))
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            g.download("vel")
        else:
            prog.run()
        return unpatchify(prog.names["vel"], self.out_ch, f, hh, ww)
