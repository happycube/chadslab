"""FlashVSR's DiT (Wan 2.1 T2V-1.3B as FlashVSR v1.1's streaming DMD checkpoint: 30 blocks, width 1536, 12 heads of
128, the FFN 8960, patches of 1 x 2 x 2 of 16 latent channels; one step at t = 1000) as programs, on the CPU and the
GPU, a chunk of latent frames at a time over the keys and values of the chunks before.

Ported from FlashVSR (OpenImagingLab/FlashVSR, diffsynth/models/wan_video_dit.py, diffsynth/pipelines/
flashvsr_tiny_long.py; Apache License 2.0; no copyright line stated: OpenImagingLab), itself from DiffSynth-Studio
and Wan 2.1 (Apache License 2.0). See THIRD_PARTY_NOTICES.md.

    the vectors (host, float32): t = time_embedding(sinusoid(1000)) (linear, SiLU, linear), t_mod =
        time_projection(t) (SiLU, linear: 6 x 1536); each block's modulation + t_mod (the shift, 1 + scale and gate
        of the self-attention and of the FFN); the head's modulation + t (shift, 1 + scale)
    the text (once): the fixed prompt (posi_prompt: 512 x 4096) through text_embedding (linear, tanh GELU, linear),
        each block's cross-attention keys (k, the RMS norm across the heads) and values (v)
    a chunk (model_fn_wan_video's streaming form: the first of 6 latent frames, then 2 at a time; the RoPE frames
        0..5, then 4 + 2 c, 5 + 2 c): the patch embedding (a linear of each patch's (channel, y, x)), + the LQ
        projector's rows (the first block's condition); a block: LN (no weights) x (1 + scale) + shift, q k v, the RMS
        norms of q and k across the heads (q's weight times hd^-1/2: attention of scale 1), the 3D RoPE (44, 42, 42
        values of the frame, y, x; theta 10000, interleaved pairs); the self-attention block-sparse (below), its
        output x gate + x; LN with norm3's weights, the cross attention over the text's 512 keys, + x; LN x (1 +
        scale) + shift, the FFN (linear, tanh GELU, linear), x gate + x; the head: LN x (1 + scale) + shift, a
        linear to (y, x, channel) of a patch

The self-attention (SelfAttention.forward, generate_draft_block_mask, Block-Sparse-Attention): the rows in windows
of 2 x 8 x 8 tokens (frame pairs: a "slice"; a window a block of 128 rows; the rows of a chunk in the order (slice,
block y, block x, frame, y, x), so a block is 128 consecutive rows); the keys and values of the chunk's slices and
of the 3 slices before (kv_ratio 3: a ring of 4 slots of a slice each, float16 on the GPU (kv16), float32 on the
CPU). On the GPU each layer's ring lives in pinned host memory (28 GB at 2560 x 1920: more than the card), and two
device pairs of 4 slots take the layers in turn: GP_FETCH copies layer i + 2's cached slots into the pair of layer i
once that layer is done with it (FETCH_DONE), while the layers between run, and GP_TO_HOST its new slots back (the same
values as rings on the device). The mask of each block of queries (ENC_BLOCK_MASK; draft_mask the host's form): the
scores of the blocks' mean queries and mean keys (per head; each layer's mean keys of the ring's slots kept on the
device), -inf outside the local window of 11 x 11 blocks (local_range, not clamped at the edges), the softmax over
the key blocks, the top-k of each head's and query slice's map (topk = int(nb^2 ratio) - 1, nb the blocks of a
slice, ratio = sparse_ratio 768 1280 / (H W) of the output's pixels): the blocks above its (topk + 1)-th largest.
Each block of queries attends to its blocks' keys (ENC_ATTN_BLOCKS) in that order (the cache's oldest first).

The programs (one _Runner; the arrays of a chunk's size shared): ("text",); ("chunk", f, hh, ww, slot, keep): the
30 blocks of a chunk of f latent frames whose new slices go to slot (f = 6: slots 0-2), each its q, k, v, its new
keys and values, its mask, then its attention and the rest (the last: the head); keep: each block's lists kept
(parts' masks). No host step between the blocks. Each block in tiles of ROWS rows (RECLANK_FLASHVSR_ROWS, 8192):
every operation but the attention is a row's own, and a tile's query blocks read the keys of their lists
(ENC_ATTN_BLOCKS' qb0), so only the residual stream and the queries are kept for all the rows (the same values as
one tile). The matrices on the GPU F20 (encoders.runtime.F20Lins; 16 significant bits a value, float32 products), or
tc: W16H on the tensor cores (x float16); on the CPU float32 as stored. The self-attention float32, or attn_tc: on the
tensor cores (q and the weights float16, the sums float32). The rest float32.
"""
from __future__ import annotations

import ctypes
import os
import threading

import numpy as np

from reclank.encoders.runtime import F20Lins, _EncCompiler, _Plain, _Runner
from reclank.ir import program as P
from reclank.formats.safetensors import SafeTensors
from reclank.models.flux2.dit import rope_table
from reclank.runtime.threads import team_threads

WIDTH, HEADS, HD, INTER, LAYERS = 1536, 12, 128, 8960, 30
TEXT, TEXT_DIM, FREQ = 512, 4096, 256
EPS = 1e-6
MLP_ROWS = 4096
ROWS = int(os.environ.get("RECLANK_FLASHVSR_ROWS", "8192"))     # the rows of a tile of the programs (of 128)
WIN = (2, 8, 8)
KV_SLICES = 3           # kv_ratio
LOCAL = 11              # local_range
SPARSE = 2.0            # sparse_ratio (topk_ratio = SPARSE 768 1280 / (H W))
DIT_FILE = "diffusion_pytorch_model_streaming_dmd.safetensors"


def _silu(x):
    return (x / (np.float32(1) + np.exp(-x))).astype(np.float32)


def sinusoid(t, dim=FREQ):
    """Wan's sinusoidal_embedding_1d of one timestep: [cos, sin] of t 10000^(-i / (dim / 2)), float64 then float32"""
    half = dim // 2
    s = np.float64(t) * np.power(10000.0, -np.arange(half, dtype=np.float64) / half)
    return np.concatenate([np.cos(s), np.sin(s)]).astype(np.float32)


def patchify(lat):
    """the rows (f hh ww, 64) of latents (16, f, 2 hh, 2 ww): each patch's (channel, y, x)"""
    C, f, h, w = lat.shape
    x = lat.reshape(C, f, h // 2, 2, w // 2, 2).transpose(1, 2, 4, 0, 3, 5)
    return np.ascontiguousarray(x.reshape(f * (h // 2) * (w // 2), C * 4), dtype=np.float32)


def unpatchify(rows, f, hh, ww):
    """the latents (16, f, 2 hh, 2 ww) of the head's rows (f hh ww, 64: (y, x, channel))"""
    x = rows.reshape(f, hh, ww, 2, 2, 16).transpose(5, 0, 1, 3, 2, 4)
    return np.ascontiguousarray(x.reshape(16, f, 2 * hh, 2 * ww), dtype=np.float32)


def window_order(f, hh, ww):
    """the token (f, y, x order) of each row of the windowed order (slice, block y, block x, frame, y, x)"""
    t = np.arange(f * hh * ww).reshape(f // 2, 2, hh // 8, 8, ww // 8, 8)
    return np.ascontiguousarray(t.transpose(0, 2, 4, 1, 3, 5).reshape(-1))


def windowed(x, f, hh, ww, threads=8):
    """x (f hh ww, C: the order (f, y, x), or a list of each frame's (hh ww, C)) in the window order (window_order's:
    x[window_order(f, hh, ww)]), the copies on threads (NumPy lets the GIL go)"""
    if not isinstance(x, (list, tuple)):
        x = [x[i * hh * ww:(i + 1) * hh * ww] for i in range(f)]
    assert len(x) == f
    C = x[0].shape[1]
    out = np.empty((f * hh * ww, C), np.float32)
    ov = out.reshape(f // 2, hh // 8, ww // 8, 2, 8, 8, C)

    def one(j):
        s_, by = divmod(j, hh // 8)
        for fi in (0, 1):
            ov[s_, by, :, fi] = x[2 * s_ + fi].reshape(hh // 8, 8, ww // 8, 8, C)[by].transpose(1, 0, 2, 3)
    from concurrent.futures import ThreadPoolExecutor

    from reclank.models.media import widen
    with ThreadPoolExecutor(threads, initializer=widen) as pool:
        list(pool.map(one, range(f // 2 * (hh // 8))))
    return out


def local_mask(nby, nbx, r=LOCAL):
    """build_local_block_mask_shifted_vec_normal_slide: block (y, x) to the blocks of the r x r window from (y - r //
    2, x - r // 2), not clamped (nby nbx, nby nbx) bool"""
    y, x = np.divmod(np.arange(nby * nbx), nbx)
    sy, sx = y - r // 2, x - r // 2
    return (y[None, :] >= sy[:, None]) & (y[None, :] <= sy[:, None] + r - 1) & \
        (x[None, :] >= sx[:, None]) & (x[None, :] <= sx[:, None] + r - 1)


def draft_mask(qm, km, nb, local, topk, heads=HEADS, hd=HD):
    """generate_draft_block_mask of the blocks' mean queries qm (nq, 1536) and mean keys km (nk, 1536), their q already
    times hd^-1/2: (heads, nq, nk) bool (the host's form; ENC_BLOCK_MASK the programs')"""
    nq, nk = qm.shape[0], km.shape[0]
    qh = np.ascontiguousarray(qm.reshape(nq, heads, hd).transpose(1, 0, 2), dtype=np.float32)
    kh = np.ascontiguousarray(km.reshape(nk, heads, hd).transpose(1, 2, 0), dtype=np.float32)
    s = np.matmul(qh, kh).astype(np.float32)
    lm = np.tile(local, (nq // nb, nk // nb))
    s = np.where(lm[None], s, -np.inf).astype(np.float32)
    s = np.exp(s - s.max(-1, keepdims=True)).astype(np.float32)
    p = (s / s.sum(-1, keepdims=True)).astype(np.float32)
    flat = p.reshape(heads * (nq // nb), nb * nk)
    n = flat.shape[1]
    k = min(n - 1, topk)
    thr = np.partition(flat, n - 1 - k, axis=1)[:, n - 1 - k]
    return (flat > thr[:, None]).reshape(heads, nq, nk)


def block_lists(mask, ids):
    """(heads, nq, nk) int32: the ids (nk,) of each (head, query block)'s key blocks in mask, in order, -1 after"""
    order = np.argsort(~mask, axis=-1, kind="stable")
    out = np.asarray(ids, np.int32)[order]
    out[np.take_along_axis(~mask, order, -1)] = -1
    return np.ascontiguousarray(out, dtype=np.int32)


class FlashDiT:
    """The DiT of a FlashVSR-v1.1 directory (the DMD checkpoint, posi_prompt.safetensors). start(hh, ww) before a
    video of hh x ww tokens; chunk(lat, lq) of each chunk."""

    def __init__(self, path, gpu=False, cpu_threads=None, f20=None, kv16=None, tc=False, attn_tc=False):
        self.gpu, self.cpu_threads = bool(gpu), cpu_threads
        self.W = W = SafeTensors(os.path.join(path, DIT_FILE))
        f20 = self.gpu if f20 is None else bool(f20)
        self.kv16 = self.gpu if kv16 is None else bool(kv16)
        assert not self.kv16 or self.gpu, "the float16 keys and values are a GPU form"
        # (attn_tc: the self-attention on the tensor cores, q and the weights float16 (ENC_ATTN_BLOCKS' tc))
        self.attn_tc = bool(attn_tc) and self.kv16
        # (tc: the matrices' W16H rows on the tensor cores, x float16 as it is read; else F20, float32 products)
        self.f20 = F20Lins(self.gpu, "flashvsr dit", fmt="f16" if tc and self.gpu else "f20") if f20 else None
        f = lambda n: np.ascontiguousarray(W.get(n), dtype=np.float32)  # noqa: E731
        sq = np.float32(HD ** -0.5)
        self.te0, self.te2 = self.linear("text_embedding.0"), self.linear("text_embedding.2")
        self.patch, self.head = self.linear("patch_embedding"), self.linear("head.head")
        self.blocks = []
        for i in range(LAYERS):
            p = "blocks.%d." % i
            s, c = p + "self_attn.", p + "cross_attn."
            self.blocks.append(dict(
                q=self.linear(s + "q"), k=self.linear(s + "k"), v=self.linear(s + "v"), o=self.linear(s + "o"),
                nq=(f(s + "norm_q.weight") * sq).astype(np.float32), nk=f(s + "norm_k.weight"),
                q2=self.linear(c + "q"), k2=self.linear(c + "k"), v2=self.linear(c + "v"), o2=self.linear(c + "o"),
                nq2=(f(c + "norm_q.weight") * sq).astype(np.float32), nk2=f(c + "norm_k.weight"),
                n3w=f(p + "norm3.weight"), n3b=f(p + "norm3.bias"),
                up=self.linear(p + "ffn.0"), down=self.linear(p + "ffn.2"),
                mod=f(p + "modulation").reshape(6, WIDTH)))
        if self.f20 is not None:
            self.f20.report()
        self.vec = self.vectors(f)
        P = SafeTensors(os.path.join(path, "posi_prompt.safetensors"))
        self.prompt = np.ascontiguousarray(P.get(P.names()[0]), dtype=np.float32).reshape(TEXT, TEXT_DIM)
        P.close()
        self.kc = np.zeros((LAYERS, TEXT, WIDTH), np.float32)
        self.vc = np.zeros((LAYERS, TEXT, WIDTH), np.float32)
        self.runner = _Runner(self._build, keep=256, keep_gpu=256)
        self.acts, self.rings = {}, {}
        self.ring_lock = threading.Lock()
        self.text_done = False

    def linear(self, name):
        src = self.W.get(name + ".weight")
        lin = _Plain(np.ascontiguousarray(src.reshape(src.shape[0], -1), dtype=np.float32))
        lin.shape = lin.w.shape
        lin.b = np.ascontiguousarray(self.W.get(name + ".bias"), dtype=np.float32)
        if self.f20 is not None:
            self.f20.add(lin, src)
        return lin

    def vectors(self, f):
        """the modulation vectors (name: (1536,)) at t = 1000"""
        e = (f("time_embedding.0.weight") @ sinusoid(1000.0) + f("time_embedding.0.bias")).astype(np.float32)
        t = (f("time_embedding.2.weight") @ _silu(e) + f("time_embedding.2.bias")).astype(np.float32)
        tm = (f("time_projection.1.weight") @ _silu(t) + f("time_projection.1.bias")).astype(np.float32)
        tm = tm.reshape(6, WIDTH)
        one = np.float32(1)
        out = {}
        for i, B in enumerate(self.blocks):
            m = (B["mod"] + tm).astype(np.float32)
            out.update({"a%db" % i: m[0], "a%dw" % i: one + m[1], "a%dg" % i: m[2],
                        "f%db" % i: m[3], "f%dw" % i: one + m[4], "f%dg" % i: m[5]})
        sh, sc = (f("head.modulation").reshape(2, WIDTH) + t).astype(np.float32)
        out.update(ob=sh, ow=one + sc)
        return {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in out.items()}

    # ---- programs ------------------------------------------------------------------------------------

    def _acts(self, f, hh, ww):
        """the arrays of a chunk of f frames: prefixes of those of the first chunk's (6 frames: the most rows), so
        the chunks of both sizes share their device copies; the tiles' buffers (ROWS rows) shared by every tile; each
        layer's mean keys of the ring's slots (kms, kept on the device: the masks of the chunks after)"""
        nb = hh * ww // 64
        if (hh, ww) not in self.acts:
            z = lambda *s: np.zeros(s, np.float32)  # noqa: E731
            N = 6 * hh * ww
            T = min(N, ROWS)
            K = max(T, TEXT)
            R = KV_SLICES + 1
            base = dict(
                img=z(N, 64), vel=z(N, 64), x=z(N, WIDTH), q=z(N, WIDTH), cos=z(N, HD // 2), sin=z(N, HD // 2),
                qm=z(N // 128, WIDTH), kms=z(LAYERS, R * nb, WIDTH), ids=np.zeros(R * nb, np.int32),
                p=z(HEADS * max(9, R) * nb * nb), thr=z(HEADS * 3), bl=np.full(HEADS * 3 * nb * 3 * nb, -1, np.int32),
                h=z(T, WIDTH), tmp=z(T, WIDTH), k1=z(T, WIDTH), v1=z(T, WIDTH), oa=z(T, WIDTH), qc=z(K, WIDTH),
                o=z(K, WIDTH), g=z(min(T, MLP_ROWS), INTER), clo=np.zeros(K, np.int32), chi=np.full(K, TEXT, np.int32),
                local=np.ascontiguousarray(local_mask(hh // 8, ww // 8), dtype=np.uint8))
            if not self.gpu:            # (the CPU's ENC_ATTN_RANGE: the text's keys in buffers of the queries' rows)
                base.update(kk=z(K, WIDTH), vv=z(K, WIDTH))
            self.acts[(hh, ww)] = (base, {k: v.copy() for k, v in self.vec.items()})
        base, vec = self.acts[(hh, ww)]
        N = f * hh * ww
        nkb = (f // 2 if f > 2 else KV_SLICES + 1) * nb
        full = ("img", "vel", "x", "q", "cos", "sin")
        out = {k: (base[k][:N] if k in full else base[k]) for k in base if k != "bl"}
        out["qm"], out["ids"] = base["qm"][:N // 128], base["ids"][:nkb]
        out["bl"] = base["bl"][:HEADS * (N // 128) * nkb].reshape(HEADS, N // 128, nkb)
        out.update(vec)
        return out

    def _ring(self, hh, ww):
        """each layer's keys and values of the ring's 4 slots (on the host: the GPU's in pinned memory, each layer's
        copied to one of the two device pairs dev for its forms) and the device pairs (GPU); start pins them on a
        thread of its own (28 GB at 2560 x 1920) while the first chunk's inputs are made"""
        with self.ring_lock:
            return self._ring_make(hh, ww)

    def _ring_make(self, hh, ww):
        if (hh, ww) not in self.rings:
            S = 2 * hh * ww
            dt = np.uint16 if self.kv16 else np.float32
            R = (KV_SLICES + 1) * S
            ring = [(np.zeros((R, WIDTH), dt), np.zeros((R, WIDTH), dt)) for _ in range(LAYERS)]
            dev = [(np.zeros((R, WIDTH), dt), np.zeros((R, WIDTH), dt)) for _ in range(2)] if self.gpu else None
            if self.gpu:
                # (the pages written first, on threads: cudaHostRegister of pages not yet there holds the map's lock
                # while it makes them, and the other threads' allocations wait; 0.65 s a GB, 0.13 present)
                from concurrent.futures import ThreadPoolExecutor

                from reclank.backends.cuda.runtime import lib
                from reclank.models.media import widen
                with ThreadPoolExecutor(8, initializer=widen) as pool:
                    list(pool.map(lambda a: a.fill(0), [a for kv in ring for a in kv]))
                for kv in ring:
                    for a in kv:
                        if lib().rc_gpu_host_register_rw(ctypes.c_void_p(a.ctypes.data), a.nbytes) == 0:
                            raise RuntimeError("cudaHostRegister of %.2f GB: %s" % (
                                a.nbytes / 1e9, lib().rc_gpu_last_error().decode()))
            # each layer's GP_FETCH ranges (host, device, bytes) of its 3 cached slots' keys and values
            fetch = [np.zeros((2 * KV_SLICES, 3), np.int64) for _ in range(LAYERS)]
            self.rings[(hh, ww)] = (ring, dev, fetch)
        return self.rings[(hh, ww)]

    def _build(self, key, gpu):
        c = _EncCompiler(EPS, pack=not gpu)
        c.attn_f32 = True
        env = c.env
        if key[0] == "text":
            env.update(txt=self.prompt.copy(), ctx=np.zeros((TEXT, WIDTH), np.float32),
                       tmp=np.zeros((TEXT, WIDTH), np.float32))
            forms = [("set", "tmp", ("enc_linear", self.te0, "txt")), ("set", "tmp", ("enc_gelu", "tmp")),
                     ("set", "ctx", ("enc_linear", self.te2, "tmp"))]
            for i, B in enumerate(self.blocks):
                env["kc%d" % i], env["vc%d" % i] = self.kc[i], self.vc[i]
                forms += [("set", "tmp", ("enc_linear", B["k2"], "ctx")),
                          ("set", "kc%d" % i, ("enc_rms", "tmp", B["nk2"], WIDTH)),
                          ("set", "vc%d" % i, ("enc_linear", B["v2"], "ctx"))]
            c.compile(("seq", *forms))
        else:
            _, f, hh, ww, slot, keep = key
            env.update(self._acts(f, hh, ww))
            N = f * hh * ww
            S = 2 * hh * ww
            nb = hh * ww // 64
            first = f > 2
            new = list(range(f // 2)) if first else [slot]
            ring, dev, fetch = self._ring(hh, ww)
            topk = int(nb * nb * (SPARSE * 768 * 1280 / (256 * hh * ww))) - 1
            if not gpu:
                env["kt"], env["vt"] = env["kk"][:TEXT], env["vv"][:TEXT]
            if gpu and not first:
                for j in range(min(2, LAYERS)):
                    c.p.emit(P.FETCH, fetch[j], len(fetch[j]), j, j % 2)
            for i in range(LAYERS):
                head, tail = self._layer(env, i, N, S, nb, slot, new, ring, dev, topk, keep, gpu)
                c.compile(("seq", *head))
                if gpu and not first:
                    c.p.emit(P.FETCH_WAIT, i)       # (layer i's cached slots in its pair)
                c.compile(("seq", *tail))
                if gpu:
                    b = i % 2
                    # (the new slots back to the host; then the pair free for the copy of layer i + 2's)
                    cp = [(dev[b][kv][s_ * S:(s_ + 1) * S], ring[i][kv][s_ * S:(s_ + 1) * S])
                          for s_ in new for kv in (0, 1)]
                    for g0 in range(0, len(cp), 3):
                        ops = []
                        for src, dst in cp[g0:g0 + 3]:
                            ops += [src, dst, dst.nbytes]
                        ops += [0, 0, 0] * (3 - len(cp[g0:g0 + 3]))
                        c.p.emit(P.TO_HOST, *ops, 0)
                    if not first:
                        c.p.emit(P.FETCH_DONE, b)
                        if i + 2 < LAYERS:
                            c.p.emit(P.FETCH, fetch[i + 2], len(fetch[i + 2]), i + 2, b)
        p = c.p.finish()
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def _layer(self, env, i, N, S, nb, slot, new, ring, dev, topk, keep, gpu):
        """the forms of block i of a chunk of N rows, in tiles of ROWS rows: (its q, k, v (its new keys and values into
        slot of its pair; the first chunk: slots 0-2) and its mask, its attention and the rest)"""
        Kr, Vr = dev[i % 2] if gpu else ring[i]
        kn, vn = "K%d" % i, "V%d" % i
        env[kn], env[vn] = Kr, Vr
        kms = "kms%d" % i
        env[kms] = env["kms"][i]
        bl = env["bl"]
        if keep:                        # (each layer's lists of its own: parts' masks)
            bl = env["bl%d" % i] = np.full(env["bl"].shape, -1, np.int32)
        B = self.blocks[i]
        a, m = "a%d" % i, "f%d" % i
        v = lambda nm, arr: (env.__setitem__(nm, arr), nm)[1]  # noqa: E731
        pre, post = [], [] if gpu else [("set", "kt", ("copy", "kc%d" % i)), ("set", "vt", ("copy", "vc%d" % i))]
        env["kc%d" % i], env["vc%d" % i] = self.kc[i], self.vc[i]
        for t0 in range(0, N, ROWS):        # (the row tiles: a row's operations its own, but the attention)
            t1 = min(N, t0 + ROWS)
            n_ = t1 - t0
            sfx = "_%d_%d" % (i, t0)
            x, q = v("x" + sfx, env["x"][t0:t1]), v("q" + sfx, env["q"][t0:t1])
            h, tmp = v("h" + sfx, env["h"][:n_]), v("tmp" + sfx, env["tmp"][:n_])
            k1, v1 = v("k1" + sfx, env["k1"][:n_]), v("v1" + sfx, env["v1"][:n_])
            cs, sn = v("cos" + sfx, env["cos"][t0:t1]), v("sin" + sfx, env["sin"][t0:t1])
            if i == 0:                      # (the LQ rows uploaded into x: x = lq + the patches')
                pre += [("set", tmp, ("enc_linear", self.patch, v("img" + sfx, env["img"][t0:t1]))),
                        ("enc_add", x, tmp)]
            km0 = slot * nb + t0 // 128
            pre += [("set", h, ("enc_lnorm", x, a + "w", a + "b", WIDTH)),
                    ("set", tmp, ("enc_linear", B["q"], h)),
                    ("set", q, ("enc_rms", tmp, B["nq"], WIDTH)),
                    ("set", tmp, ("enc_linear", B["k"], h)),
                    ("set", k1, ("enc_rms", tmp, B["nk"], WIDTH)),
                    ("set", v1, ("enc_linear", B["v"], h)),
                    ("enc_rope_tab", q, cs, sn, HEADS, HD),
                    ("enc_rope_tab", k1, cs, sn, HEADS, HD),
                    ("enc_resize", q, v("qm" + sfx, env["qm"][t0 // 128:t1 // 128]), n_ // 128, 128, 1, 1, 1, WIDTH, 2),
                    ("enc_resize", k1, v("km" + sfx, env["kms"][i][km0:km0 + n_ // 128]), n_ // 128, 128, 1, 1, 1,
                     WIDTH, 2)]
            kd = v("Kd" + sfx, Kr[slot * S + t0:slot * S + t1])
            vd = v("Vd" + sfx, Vr[slot * S + t0:slot * S + t1])
            if self.kv16:
                pre += [("set", kd, ("enc_bf16_pack", k1, 1)), ("set", vd, ("enc_bf16_pack", v1, 1))]
            else:
                pre += [("set", kd, ("copy", k1)), ("set", vd, ("copy", v1))]
            oa = v("oa" + sfx, env["oa"][:n_])
            qc = v("qc" + sfx, env["qc"][:n_])
            o2 = v("o2" + sfx, env["o"][:n_])
            post += [("set", oa, ("enc_attn_blocks", q, kn, vn, bl, HEADS, HD, t0 // 128, int(self.attn_tc))),
                     ("set", tmp, ("enc_linear", B["o"], oa)),
                     ("set", tmp, ("enc_mul_vec", tmp, a + "g")),
                     ("enc_add", x, tmp),
                     ("set", h, ("enc_lnorm", x, B["n3w"], B["n3b"], WIDTH)),
                     ("set", tmp, ("enc_linear", B["q2"], h)),
                     ("set", qc, ("enc_rms", tmp, B["nq2"], WIDTH)),
                     ("set", "o", ("enc_attn_range_x2", "qc", "kc%d" % i if gpu else "kk", "vc%d" % i if gpu else "vv",
                                   "clo", "chi", HEADS, HD)),
                     ("set", tmp, ("enc_linear", B["o2"], o2)),
                     ("enc_add", x, tmp),
                     ("set", h, ("enc_lnorm", x, m + "w", m + "b", WIDTH))]
            for r0 in range(0, n_, MLP_ROWS):
                r1 = min(n_, r0 + MLP_ROWS)
                sub = "%s_%d" % (sfx, r0)
                g_, h_ = v("g" + sub, env["g"][:r1 - r0]), v("h" + sub, env["h"][r0:r1])
                post += [("set", g_, ("enc_linear", B["up"], h_)),
                         ("set", g_, ("enc_gelu", g_)),
                         ("set", v("tmp" + sub, env["tmp"][r0:r1]), ("enc_linear", B["down"], g_))]
            post += [("set", tmp, ("enc_mul_vec", tmp, m + "g")), ("enc_add", x, tmp)]
            if i == LAYERS - 1:
                post += [("set", h, ("enc_lnorm", x, "ow", "ob", WIDTH)),
                         ("set", v("vel" + sfx, env["vel"][t0:t1]), ("enc_linear", self.head, h))]
        mask = [("enc_block_mask", "qm", kms, "ids", "local", "ids", "p", "thr", bl, nb, HEADS, HD, topk)]
        return pre + mask, post

    def _get(self, key):
        return self.runner.get(key, self.gpu)

    # ---- running -------------------------------------------------------------------------------------

    def start(self, hh, ww, out_pixels=None):
        """a new video of hh x ww tokens a frame (hh, ww multiples of 8): the text's caches once, the chunks' count.
        (out_pixels: the output's H W, 256 hh ww: the top-k's ratio)"""
        if not self.text_done:
            prog, g = self._get(("text",))
            (g or prog).run()           # (the caches' device copies are the chunks')
            self.text_done = True
        assert hh % 8 == 0 and ww % 8 == 0, "FlashVSR: the tokens' grid a multiple of 8 (the frames of 128 pixels)"
        assert out_pixels in (None, 256 * hh * ww)
        self.hh, self.ww, self.n = hh, ww, 0
        self.nb = hh * ww // 64
        if self.gpu and (hh, ww) not in self.rings:
            threading.Thread(target=self._ring, args=(hh, ww), daemon=True).start()
        self.slices = []                                # the ring's slots of the slices before (oldest first)

    def inputs(self, lat, n):
        """chunk n's inputs of its latents lat (a host step, on any thread: start first): the rows' window order,
        the patches' rows and the RoPE tables in that order"""
        f, hh, ww = lat.shape[1], self.hh, self.ww
        order = window_order(f, hh, ww)
        f0 = 0 if n == 0 else 4 + 2 * n
        pos = np.stack(np.meshgrid(np.arange(f0, f0 + f), np.arange(hh), np.arange(ww), indexing="ij"), -1)
        pos = pos.reshape(-1, 3)
        cos, sin = rope_table(pos[order], [HD - 2 * (2 * (HD // 6)), 2 * (HD // 6), 2 * (HD // 6)], 10000.0)
        return dict(order=order, img=patchify(lat)[order], cos=cos, sin=sin)

    def chunk(self, lat, lq, parts=None, pre=None):
        """The DiT's output (16, f, 2 hh, 2 ww) of a chunk's latents lat (16, f, 2 hh, 2 ww: f 6 the first, then 2)
        and the LQ projector's rows lq (f hh ww, 1536, the order (f, y, x); or a list of each frame's): one program
        of the 30 blocks, their masks on the device (ENC_BLOCK_MASK). parts: a dict that gets each block's masks
        (heads, query blocks, key blocks) bool. pre: inputs(lat, n) of this chunk, made before."""
        _, f, h2, w2 = lat.shape
        hh, ww, nb = self.hh, self.ww, self.nb
        assert (h2, w2) == (2 * hh, 2 * ww) and f == (6 if self.n == 0 else 2)
        N = f * hh * ww
        S = 2 * hh * ww
        pre = self.inputs(lat, self.n) if pre is None else pre
        order = pre["order"]
        slot = 0 if self.n == 0 else (self.n + 2) % (KV_SLICES + 1)
        new = list(range(f // 2)) if self.n == 0 else [slot]
        slots = self.slices + new
        ids = np.concatenate([s_ * nb + np.arange(nb) for s_ in slots]).astype(np.int32)
        keep = parts is not None
        prog, g = self._get(("chunk", f, hh, ww, slot, keep))
        lq = lq if isinstance(lq, list) else np.asarray(lq, np.float32)
        inputs = dict(img=pre["img"], x=windowed(lq, f, hh, ww), cos=pre["cos"],
                      sin=pre["sin"], ids=ids)
        for nm, a in inputs.items():
            if g is not None:           # (from the array itself: no copy to the host's)
                assert a.shape == prog.names[nm].shape and a.dtype == prog.names[nm].dtype
                g.mirror.buffer_of(prog.names[nm]).upload(np.ascontiguousarray(a))
            else:
                prog.names[nm][...] = a
        if g is not None:
            ring, dev, fetch = self._ring(hh, ww)
            if self.n > 0:              # (each layer's cached slots: host, device, bytes)
                for i in range(LAYERS):
                    rows = []
                    for s_ in self.slices:
                        for kv in (0, 1):
                            src, dst = ring[i][kv], dev[i % 2][kv]
                            off = s_ * S * src.strides[0]
                            rows.append((src.ctypes.data + off, g.mirror.translate(dst.ctypes.data) + off,
                                         S * src.strides[0]))
                    fetch[i][...] = rows
            g.run()
            g.download("vel")
            if keep:
                for i in range(LAYERS):
                    g.download("bl%d" % i)
        else:
            prog.run()
        vel = np.empty((N, 64), np.float32)
        vel[order] = prog.names["vel"]
        if keep:
            masks = []
            for i in range(LAYERS):
                bl = prog.names["bl%d" % i]
                pos_ = {int(x): j for j, x in enumerate(ids)}
                m_ = np.zeros(bl.shape[:2] + (len(ids),), bool)
                hq, qq, jj = np.nonzero(bl >= 0)
                m_[hq, qq, [pos_[int(x)] for x in bl[hq, qq, jj]]] = True
                masks.append(m_)
            parts["masks"] = np.stack(masks)
        self.slices = slots[-KV_SLICES:]
        self.n += 1
        return unpatchify(vel, f, hh, ww)

    def release(self):
        if self.gpu:
            self.runner.release(keep_host=True)
