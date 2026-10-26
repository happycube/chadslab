"""Nemotron 3.5 Bolt on the CPU with the C kernels (P3 of docs/NEMOTRON_BOLT_PLAN.md): the step as
one program of records (reclank/ir/program.py; the CPU interpreter runs it), from the shipped
weights:

- Q8_0 (attention, the latent projections, lm_head) as stored: the matrices of a file are views of
  its memory map;
- IQ2_XXS (the experts) in the rows of KQ_IQ2R (reclank/formats/iq2.py to_rows: the same bytes, the
  parts of a row first, its scales after them); the routed experts of a layer in one stack (expert
  after expert), the record reads expert e at e x inner rows;
- NVFP4 (the Mamba projections) in the rows of KQ_NV4 (the codes of each 32 reordered, the scales
  after them: kq_nv4_pack), the same nibbles and scales;
- bf16 vectors widened to float32; the bf16 matrices (Super's latent projections, routers, the MTP
  block's eh_proj) in BF12 rows (KQ_BF12: float x; near-zero values zeroed, cops.kq_bf16_to_bf12),
  or float32 (RECLANK_NM_BF12=0, a matrix that BF12 would harm: keep.bf12_bad);
- Super (docs/NEMOTRON_SUPER_PLAN.md): FP8 with a tensor scale (Mamba, attention, lm_head) in the
  rows of KQ_MXF8 (the E4M3 codes as stored, every E8M0 127: 1.0), the tensor scale on the
  output (Mat.scale, a GP_MUL); the NVFP4 experts in KQ_NV4 rows, as Bolt's NVFP4.

compile_nemotron_step(m, t): the input names["H"] (t x hidden, the embeddings), the output
names["xn"] (the input of the head, after norm_f). The cache: conv.<i> and S.<i> of each Mamba
layer, K.<i> and V.<i> (float32 rows, position-major) of each attention layer.

    m = NemotronKQ(path)
    cache = m.cache(max_len)
    logits = m.forward(ids, cache)                     # [t, vocab]
"""
from __future__ import annotations

import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats import iq2
from reclank.formats.fp4local_st import KQ_MXF8, mxf8_pack
from reclank.models.nemotron.model import BB, LM, Checkpoint, NemotronConfig

KQ_F32, KQ_Q8_0, KQ_IQ2_XXS, KQ_NV4, KQ_IQ2R, KQ_BF12 = 0, 8, 16, 51, 67, 57
# the bf16 matrices in BF12 rows (RECLANK_NM_BF12=0: float32)
BF12 = os.environ.get("RECLANK_NM_BF12", "1") != "0"
# x of the integer products as int16 in two int8 planes (kq_quant_part16: v = 128 hi + lo), as the
# GPU (gpu.X16); RECLANK_NM_X16=0: int8 x (kq_quant_part)
X16 = os.environ.get("RECLANK_NM_X16", "1") != "0"
# the residual sums and the norm of the next block in one record (RECLANK_NM_ADDNORM=0: ADD, ADD, RMS_NORM)
ADDNORM = os.environ.get("RECLANK_NM_ADDNORM", "1") != "0"


def bf12_mat(bits):
    """A bf16 matrix (uint16 bits [rows, cols]) as KQ_BF12 rows, or None (RECLANK_NM_BF12=0, cols not a
    multiple of 32, or a zeroed value that matters: keep.bf12_bad)."""
    from reclank.models.qwen.keep import bf12_bad
    if not BF12 or bits.ndim != 2 or bits.shape[1] % 32:
        return None
    out, rep = cops.kq_bf16_to_bf12(bits, bits.shape[1])
    if bf12_bad(rep):
        return None
    return Mat(out.reshape(-1), KQ_BF12, bits.shape[0], bits.shape[1])


# Q8_0 in rows for the GPU without dp4a (NemotronGPU: Q8R; backends/cuda/csrc/kernels/quant/kquants.cu)
KQ_Q8_R = 100


def q8r_rows(data, rows, cols, out=None):
    """Q8_0 blocks (rows x cols / 32 of 34 bytes: the float16 scale, then 32 int8 values) as KQ_Q8_R rows
    (into out, rows of q8r_row_bytes(cols), when given): the int8 values of the row, then the float16 scale
    of each 32, then zeros to a multiple of 16 bytes (QwenGPU.dense's form)."""
    nb = cols // 32
    b = np.asarray(data).reshape(rows, nb, 34)
    out = np.empty((rows, q8r_row_bytes(cols)), np.uint8) if out is None else out.reshape(rows, q8r_row_bytes(cols))
    q, d = out[:, :cols], out[:, cols:cols + 2 * nb]
    q.shape, d.shape = (rows, nb, 32), (rows, nb, 2)      # (views: an error, not a copy, when not)
    q[...] = b[:, :, 2:]
    d[...] = b[:, :, :2]
    out[:, cols + 2 * nb:] = 0
    return out.reshape(-1)


def q8r_row_bytes(cols):
    return (cols // 32 * 34 + 15) // 16 * 16


class Mat:
    """A matrix as the records take it: the bytes, the type, rows, cols; scale: a factor of every
    output (FP8's tensor scale), or None."""

    def __init__(self, data, type_, rows, cols, scale=None):
        self.data, self.type, self.rows, self.cols = data, type_, rows, cols
        self.scale = scale
        # the factor as a row for GP_MUL (stride 0: the same row for each token)
        self.svec = np.full(rows, scale, np.float32) if scale is not None else None


class NemotronKQCache:
    """The state of one sequence for the program (max_len positions of the attention). kv: the
    rows of the keys and values, float32 (the CPU) or "bf16" (the GPU: bfloat16 bits, the
    reference's cache)."""

    def __init__(self, cfg, max_len, kv="f32"):
        self.max_len, self.n = max_len, 0
        self.conv, self.S, self.K, self.V = {}, {}, {}, {}
        for i, kind in enumerate(cfg.kinds):
            if kind == "mamba":
                self.conv[i] = np.zeros((cfg.conv - 1, cfg.conv_dim), np.float32)
                self.S[i] = np.zeros((cfg.m_heads, cfg.m_dim, cfg.m_state), np.float32)
            elif kind == "attention":
                dt = np.uint16 if kv == "bf16" else np.float32
                self.K[i] = np.zeros((max_len, cfg.kv_heads, cfg.head_dim), dt)
                self.V[i] = np.zeros((max_len, cfg.kv_heads, cfg.head_dim), dt)
        if "attention" in getattr(cfg, "mtp", ()):
            # (the MTP block's attention, Super: K.mtp, V.mtp)
            dt = np.uint16 if kv == "bf16" else np.float32
            self.K["mtp"] = np.zeros((max_len, cfg.kv_heads, cfg.head_dim), dt)
            self.V["mtp"] = np.zeros((max_len, cfg.kv_heads, cfg.head_dim), dt)
        # the names of the device cache of the GPU (gpu35._DevCache)
        self.kv = {i: (self.K[i], self.V[i]) for i in self.K}
        self.state = self.S

    def params(self):
        kw = {}
        for i in self.conv:
            kw["conv.%s" % i], kw["S.%s" % i] = self.conv[i], self.S[i]
        for i in self.K:
            kw["K.%s" % i], kw["V.%s" % i] = self.K[i], self.V[i]
        return kw


class NemotronKQ:
    CHUNK = 512

    def __init__(self, path, weightkeeper=True):
        """weightkeeper: take the tensors from the weightkeeper when it holds this checkpoint
        (reclank/models/nemotron/keep.py), else from the shards."""
        self.path = path
        self.cfg = NemotronConfig(path)
        self.ck = Checkpoint(path)
        self._m, self._f = {}, {}
        self.programs = {}
        self.w = None
        if weightkeeper:
            from reclank.models.nemotron import keep
            self.w = keep.attach(path)
        self._emb = self.ck.file(BB + "embeddings.weight") if self.w is None else None

    def _kept(self, name):
        """The weightkeeper's tensor name ("raw.<name>", "nv4.<name>", "iq2r.<name>" or "q8r.<name>"):
        (its entry, its bytes), or None."""
        if self.w is None:
            return None
        d = self.w.index["tensors"].get(name)
        return None if d is None else (d, self.w.tensor(name))

    def _raw(self, name):
        """A tensor of the checkpoint as the shard has it (dtype, shape), from the weightkeeper or
        the shard."""
        k = self._kept("raw." + name)
        if k is None:
            return self.ck.raw(name)
        d, b = k
        from reclank.formats.safetensors import _NP
        dt = np.uint16 if d["dtype"] == "BF16" else _NP[d["dtype"]]
        return b.view(dt).reshape(d["shape"])

    def _f32(self, name):
        """A tensor as float32 (bfloat16 widened)."""
        if self._kept("raw." + name) is None:
            return self.ck.f32(name)
        a = self._raw(name)
        if self.w.index["tensors"]["raw." + name]["dtype"] == "BF16":
            return (a.astype(np.uint32) << 16).view(np.float32)
        return a.astype(np.float32)

    # ---- the weights

    def M(self, name):
        """The matrix name (without ".weight") in the form of the records."""
        m = self._m.get(name)
        if m is not None:
            return m
        ck = self.ck
        kept = self._kept("nv4." + name) or self._kept("iq2r." + name)
        if kept is not None:
            d, b = kept
            m = self._m[name] = Mat(b, KQ_NV4 if d.get("type") != "iq2r" else KQ_IQ2R, d["rows"], d["cols"])
            return m
        if ck.file(name + ".weight").dtype(name + ".weight") == "F8_E4M3":
            # FP8 with a tensor scale (Super): the codes as stored in KQ_MXF8 rows, the scale after
            # (the bytes from the weightkeeper when it holds them)
            w = np.ascontiguousarray(self._raw(name + ".weight"))
            rows, cols = w.shape
            sc = float(np.asarray(self._f32(name + ".weight_scale")).reshape(-1)[0])
            m = Mat(mxf8_pack(w, np.full((rows, cols // 32), 127, np.uint8)).reshape(-1), KQ_MXF8, rows, cols, sc)
        elif name + ".weight_scale" in ck.where:
            w, s = ck.raw(name + ".weight"), ck.raw(name + ".weight_scale")
            g = float(np.asarray(ck.f32(name + ".weight_scale_2")).reshape(-1)[0])
            rows, cols = w.shape[0], w.shape[1] * 2
            out = np.empty(rows * cops.kq_nv4_row_bytes(cols), np.uint8)
            cops.kq_nv4_pack([np.ascontiguousarray(w)], [np.ascontiguousarray(s)], [g], rows, cols, out)
            m = Mat(out, KQ_NV4, rows, cols)
        else:
            w = self._raw(name + ".weight")
            if w.dtype == np.uint8 and w.ndim == 3 and w.shape[-1] == 34:
                m = Mat(w.reshape(-1), KQ_Q8_0, w.shape[0], w.shape[1] * 32)
            elif w.dtype == np.uint8 and w.ndim == 3 and w.shape[-1] == 66:
                m = Mat(iq2.to_rows(w).reshape(-1), KQ_IQ2R, w.shape[0], w.shape[1] * 256)
            else:
                m = None
                if ck.file(name + ".weight").dtype(name + ".weight") == "BF16":
                    m = bf12_mat(np.ascontiguousarray(w).view(np.uint16).reshape(w.shape[0], -1))
                if m is None:
                    f = np.ascontiguousarray(self._f32(name + ".weight"))
                    m = Mat(f.reshape(-1), KQ_F32, f.shape[0], f.shape[1])
        self._m[name] = m
        return m

    def experts_on(self, i, which, node):
        """The weightkeeper's copy of the stack of experts(i, which) on NUMA node node (the same bytes),
        or None (no weightkeeper, or no copy there)."""
        k = self._kept("exps.%s.%s" % (i, which))
        if k is None:
            return None
        nodes = self.w.index.get("nodes", [])
        base = self.experts(i, which)
        for p, a, _ids in k[1]:
            if p < len(nodes) and nodes[p] == node:
                st = a.reshape(-1)
                if st.ctypes.data == base.data.ctypes.data:
                    return None                 # (the stack experts() gives)
                return Mat(st, base.type, base.rows, base.cols)
        return None

    def node1(self, i):
        """NM_MOE's operands 14 and 15 for the experts of layer i: the stacks' copies on node 1 (its
        threads read them), or ()."""
        u, d = self.experts_on(i, "up", 1), self.experts_on(i, "down", 1)
        return () if u is None or d is None else (u.data, d.data)

    def experts(self, i, which):
        """The routed experts of layer i ("up" or "down"; i "mtp": the MTP block's) in one stack: (Mat
        of one expert, the stack)."""
        key = ("exps", i, which)
        m = self._m.get(key)
        if m is not None:
            return m
        E = self.cfg.moe[i if i in self.cfg.moe else min(self.cfg.moe)][0]
        k = self._kept("exps.%s.%s" % (i, which))
        if k is not None:
            d, parts = k
            stack = parts[0][1].reshape(-1)         # (the GPU's node, or node 0: whole there)
            rows = self.cfg.moe[i][1] if which == "up" else self.cfg.latent
            cols = self.cfg.latent if which == "up" else self.cfg.moe[i][1]
            m = self._m[key] = Mat(stack, KQ_NV4 if d.get("type") == "nv4" else KQ_IQ2R, rows, cols)
            return m
        if i == "mtp":
            p = LM + "mtp.layers.1.mixer.experts.%%d.%s_proj.weight" % which
        else:
            p = BB + "layers.%d.mixer.experts.%%d.%s_proj.weight" % (i, which)
        if (p % 0)[:-len(".weight")] + ".weight_scale_2" in self.ck.where:
            # NVFP4 experts (Super): each in KQ_NV4 rows, expert after expert
            b = (p % 0)[:-len(".weight")]
            w0 = self.ck.raw(b + ".weight")
            rows, cols = w0.shape[0], w0.shape[1] * 2
            per = rows * cops.kq_nv4_row_bytes(cols)
            stack = np.empty(E * per, np.uint8)
            for e in range(E):
                be = (p % e)[:-len(".weight")]
                g = float(np.asarray(self.ck.f32(be + ".weight_scale_2")).reshape(-1)[0])
                cops.kq_nv4_pack([np.ascontiguousarray(self.ck.raw(be + ".weight"))],
                                 [np.ascontiguousarray(self.ck.raw(be + ".weight_scale"))], [g], rows, cols,
                                 stack[e * per:(e + 1) * per])
            m = self._m[key] = Mat(stack, KQ_NV4, rows, cols)
            return m
        first = self.ck.raw(p % 0)
        rows, nb = first.shape[0], first.shape[1]
        per = rows * iq2.row_bytes(nb * 256)
        stack = np.empty(E * per, np.uint8)
        for e in range(E):
            iq2.to_rows(self.ck.raw(p % e), stack[e * per:(e + 1) * per])
        m = self._m[key] = Mat(stack, KQ_IQ2R, rows, nb * 256)
        return m

    def F(self, name, shape=None):
        a = self._f.get((name, shape))
        if a is None:
            a = np.ascontiguousarray(self._f32(name), np.float32)
            a = self._f[(name, shape)] = a.reshape(shape) if shape is not None else a
        return a

    def embed(self, ids):
        n = BB + "embeddings.weight"
        if self._emb is None:
            rows = self._raw(n)[np.asarray(ids, np.int64)]          # bfloat16 bits
            return (rows.astype(np.uint32) << 16).view(np.float32)
        return np.stack([self._emb.get_row(n, int(t)) for t in ids]).astype(np.float32)

    # ---- the program

    def program(self, t):
        prog = self.programs.get(t)
        if prog is None:
            prog = self.programs[t] = compile_nemotron_step(self, t)
        return prog

    def cache(self, max_len=8192):
        return NemotronKQCache(self.cfg, max_len)

    def _run(self, prog, ids, cache, pos):
        kw = {"pos": pos, "nreal": len(ids), "hs": self.cfg.head_dim,
              "scores": np.empty(max(self.cfg.heads, os.cpu_count() or 1) * (pos + len(ids)) + 64, np.float32)}
        kw.update(cache.params())
        prog.bind(**{k: v for k, v in kw.items() if k in prog.by_name})
        prog.names["H"][:] = self.embed(ids)
        prog.run()

    def forward(self, ids, cache, logits="all"):
        """The logits of the tokens ids after cache.n tokens (logits="last": the last only)."""
        ids = list(ids)
        outs = []
        c0 = 0
        while c0 < len(ids):
            chunk = ids[c0:c0 + self.CHUNK]
            prog = self.program(len(chunk))
            self._run(prog, chunk, cache, cache.n)
            cache.n += len(chunk)
            outs.append(prog.names["xn"].copy())
            c0 += len(chunk)
        xn = np.concatenate(outs)
        return self.logits(xn[-1:] if logits == "last" else xn)

    def mtp_cache(self, max_len=8192):
        """The keys and values of the MTP block's attention for a sequence (K.mtp, V.mtp)."""
        c = self.cfg
        mc = NemotronKQCache.__new__(NemotronKQCache)
        mc.max_len, mc.n = max_len, 0
        mc.conv, mc.S = {}, {}
        mc.K = {"mtp": np.zeros((max_len, c.kv_heads, c.head_dim), np.float32)}
        mc.V = {"mtp": np.zeros((max_len, c.kv_heads, c.head_dim), np.float32)}
        mc.kv, mc.state = {"mtp": (mc.K["mtp"], mc.V["mtp"])}, mc.S
        return mc

    def mtp(self, tokens, hidden, mcache):
        """The MTP block on t rows (Super; NemotronCPU.mtp's order): tokens, the token after each
        position, and hidden [t, hidden], the target's rows after norm_f there (a prefill of the
        block over a prompt), or the out of the draft before (a draft). Return (logits [t, vocab],
        out [t, hidden])."""
        t = len(tokens)
        prog = self.programs.get(("mtp", t))
        if prog is None:
            prog = self.programs[("mtp", t)] = compile_nemotron_mtp(self, t)
        kw = {"pos": mcache.n, "nreal": t, "hs": self.cfg.head_dim,
              "scores": np.empty(max(self.cfg.heads, os.cpu_count() or 1) * (mcache.n + t) + 64, np.float32)}
        kw.update(mcache.params())
        prog.bind(**{k: v for k, v in kw.items() if k in prog.by_name})
        prog.names["H"][:] = self.embed(tokens)
        prog.names["hin"][:] = np.asarray(hidden, np.float32).reshape(t, -1)
        prog.run()
        mcache.n += t
        out = prog.names["out"].copy()
        return self.logits(out), out

    def logits(self, xn):
        m = self.M(LM + "lm_head")
        t = xn.shape[0]
        xn = np.ascontiguousarray(xn, np.float32)
        xq = np.empty((t, m.cols), np.int8)
        xs = np.empty((t, m.cols // 32), np.float32)
        xm = np.empty((t, m.cols // 16), np.float32)
        if X16:
            out = cops.nm_linear2(m.data, m.type, m.rows, m.cols, xn)
        else:
            cops.kq_quant_x(xn, xq, xs, xm)
            out = np.empty((t, m.rows), np.float32)
            cops.kq_linear(m.data, m.type, m.rows, m.cols, xq, xs, xm, xn, t, out)
        return out * np.float32(m.scale) if m.scale is not None else out


def compile_nemotron_step(model, t, em=None, upto=None):
    """A step of t tokens of a NemotronKQ as a program of records. em (the GPU: gpu.NmEmit) gives
    the records of the attention and of the routed experts in its own way (em.attention, em.moe);
    else those of the CPU. upto (a check): only the blocks before it."""
    from reclank.ir import program as P
    cfg = model.cfg
    prog = P.Program()
    hid, eps = cfg.hidden, float(cfg.eps)
    inner, conv, nh = cfg.m_inner, cfg.conv_dim, cfg.m_heads
    nq, nk, hd, L = cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.latent
    Emax = max(e for e, _, _ in cfg.moe.values())
    Imax = max(n for _, n, _ in cfg.moe.values())
    k = max(kk for _, _, kk in cfg.moe.values())
    f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
    H, hn, xn = f32(t, hid), f32(t, hid), f32(t, hid)
    wide = max(inner, cfg.shared_inner, hid)
    xq, xs, xm = np.zeros((t, wide), np.int8), f32(t, wide // 32), f32(t, wide // 16)
    x16 = X16 and em is None            # (the GPU: its own records, gpu.NmEmit)
    xl = np.zeros((t, wide), np.int8) if x16 else None
    # x as int16 for a prompt group (NM_LIN2's tiled form)
    x16b = np.zeros((t, wide), np.int16) if x16 and t >= 4 else None
    zxd, y, o = f32(t, inner + conv + nh), f32(t, inner), f32(t, hid)
    cscr = f32(t, conv)
    q, kb, vb, qout, att = f32(t, nq * hd), f32(t, nk * hd), f32(t, nk * hd), f32(t, nq * hd), f32(t, nq * hd)
    sel, dense = np.zeros((t, 1), np.int32), np.full(t, -1, np.int32)
    logits, val, idx = f32(t, Emax), f32(t, k), np.zeros((t, k), np.int32)
    lat, lato, sh, mo = f32(t, L), f32(t, L), f32(t, cfg.shared_inner), f32(t, hid)
    mscr = cops.nm_moe_scratch(t, k, L, Imax) if em is None else None
    prog.names.update(H=H, xn=xn)
    pos, scores, nreal = prog.slot("pos"), prog.slot("scores"), prog.slot("nreal")

    def quant(src, cols):
        if em is not None:
            em.quant(prog, src, cols)
        if x16:
            prog.emit(P.NM_QX2, src, t, cols, xq, xl, xs, xm)
        else:
            prog.emit(P.KQ_QUANT, src, t, cols, xq, xs, xm)

    def lin(m, src, out):
        if em is not None and em.linear(prog, m, src, out, t):
            return
        if x16 and m.type in (KQ_IQ2R, KQ_NV4, KQ_Q8_0, KQ_MXF8):
            if x16b is not None and m.type != KQ_MXF8:
                prog.emit(P.NM_LIN2, xq, xl, xs, m.data, m.type, m.rows, m.cols, t, out, x16b)
            else:
                prog.emit(P.NM_LIN2, xq, xl, xs, m.data, m.type, m.rows, m.cols, t, out)
        elif em is not None and m.type == KQ_MXF8 and m.svec is not None:
            # (the GPU: the FP8 tensor scale in the product's record, operand 10: GP_MUL's bits, one launch less)
            prog.emit(P.KQ_LINEAR, xq, xs, xm, src, m.data, m.type, m.rows, m.cols, t, out, float(m.svec[0]))
            return
        else:
            prog.emit(P.KQ_LINEAR, xq, xs, xm, src, m.data, m.type, m.rows, m.cols, t, out)
        if m.svec is not None:
            # (FP8: the tensor scale of each output)
            prog.emit(P.MUL, out, m.svec, out, t, m.rows, 0)
            prog.keep.append(m.svec)

    prog.names.update(hn=hn, zxd=zxd, y=y, logits=logits, lat=lat, sh=sh)
    pend = None                         # the block's outputs not yet in H (NM_ADDNORM adds them)

    def norm(w, out):
        if pend is None:
            prog.emit(P.RMS_NORM, H, w, out, t, hid, eps)
        elif ADDNORM:
            prog.emit(P.NM_ADDNORM, H, pend[0], pend[1], w, out, t, hid, eps)
        else:
            if pend[1] is not None:
                prog.emit(P.ADD, pend[0], pend[1], pend[0], t * hid)
            prog.emit(P.ADD, H, pend[0], H, t * hid)
            prog.emit(P.RMS_NORM, H, w, out, t, hid, eps)

    # the residual stream after these blocks, for the drafter (lilicorr.py: target_layer_ids): at the
    # start of the next block, when its norm has added them (names feat.<block>)
    capture = set(getattr(model, "capture", ()) or ())
    for i, kind in enumerate(cfg.kinds):
        if upto is not None and i >= upto:
            break
        b = BB + "layers.%d." % i
        p = b + "mixer."
        norm(model.F(b + "norm.weight"), hn)
        if i - 1 in capture:
            fb = prog.names["feat.%d" % (i - 1)] = f32(t, hid)
            prog.emit(P.COPY, H, fb, t * hid * 4)
        quant(hn, hid)
        if kind == "mamba":
            if (capture or getattr(model, "spec_rows", False)) and t <= 16:
                # (a verify group: each layer's in_proj rows kept, to replay the accepted ones: spec.py)
                zxd = prog.names["zxd.%d" % i] = f32(t, inner + conv + nh)
            lin(model.M(p + "in_proj"), hn, zxd)
            prog.emit(P.NM_MAMBA, zxd, prog.slot("conv.%d" % i), model.F(p + "conv1d.weight", (conv, cfg.conv)),
                      model.F(p + "conv1d.bias"), model.F(p + "A_log"), model.F(p + "D"),
                      model.F(p + "dt_bias"), model.F(p + "norm.weight"), prog.slot("S.%d" % i), y, cscr,
                      t, nh, cfg.m_dim, cfg.m_groups, cfg.m_state, cfg.conv, eps, float(cfg.dt_min), nreal)
            quant(y, inner)
            lin(model.M(p + "out_proj"), y, o)
        elif kind == "attention":
            lin(model.M(p + "q_proj"), hn, q)
            lin(model.M(p + "k_proj"), hn, kb)
            lin(model.M(p + "v_proj"), hn, vb)
            if em is not None:
                em.attention(prog, i, q, kb, vb, att, t)
            else:
                prog.emit(P.ATTN_PREP, q, kb, vb, None, None, None, None, prog.slot("K.%d" % i),
                          prog.slot("V.%d" % i), prog.slot("hs"), pos, t, nq, nk, hd, 0, eps,
                          float(hd ** -0.5), qout, None)
                prog.emit(P.ATTN_QSA, qout, prog.slot("K.%d" % i), 0, prog.slot("V.%d" % i), 0, scores, att,
                          nq, nk, hd, t, pos, sel, dense, 1, 3, prog.slot("hs"))
            quant(att, nq * hd)
            lin(model.M(p + "o_proj"), att, o)
        else:
            E, I, kk = cfg.moe[i]
            if getattr(model, "count_experts", False):
                # (the routing profile, P6: each layer's choices in a buffer of its own)
                idx = prog.names["idx.%d" % i] = np.zeros((t, k), np.int32)
            gate = model.M(p + "gate")
            lin(gate, hn, logits)
            # (the latent rows before the router: on the GPU the router, its split and the signal of the
            # CPU's task are one kernel, the rows' copy to the host first)
            lin(model.M(p + "fc1_latent_proj"), hn, lat)
            prog.emit(P.ROUTER_SIG, logits, t, E, kk, val, idx, model.F(p + "gate.e_score_correction_bias"),
                      float(cfg.route_scale), 1e-20)
            if em is not None and hasattr(em, "count"):
                em.count(prog, i, idx, kk)      # (the GPU's pool of hot experts follows the choices: gpu.NmEmit)

            def shared(p=p):
                # the shared expert in the full width, on the same normed input (xq still holds it)
                lin(model.M(p + "shared_experts.up_proj"), hn, sh)
                prog.emit(P.RELU2, sh, sh, t * cfg.shared_inner)
                quant(sh, cfg.shared_inner)
                lin(model.M(p + "shared_experts.down_proj"), sh, o)

            if em is not None:
                # (the GPU: the shared expert while the CPU runs the cold pairs, before the wait for them)
                em.moe(prog, i, lat, idx, val, lato, t, side=shared)
            else:
                up, down = model.experts(i, "up"), model.experts(i, "down")
                prog.emit(P.NM_MOE, lat, idx, val, t, kk, up.data, up.type, down.data, down.type, L, I, mscr, lato,
                          int(x16), *model.node1(i))
                prog.keep += [up.data, down.data]
                shared()
            quant(lato, L)
            lin(model.M(p + "fc2_latent_proj"), lato, mo)
        pend = (o, mo if kind not in ("mamba", "attention") else None)
    norm(model.F(BB + "norm_f.weight"), xn)
    prog.keep += [a for a in (dense, sel, mscr) if a is not None]
    prog.tokens = t
    return prog.finish()


def compile_nemotron_mtp(model, t, em=None):
    """The MTP block of Super on t rows as a program (the records of compile_nemotron_step's blocks):
    names["H"] the embeddings of the tokens, names["hin"] the hidden rows; u = eh_proj(cat(enorm(H),
    hnorm(hin))) as the sum of eh_proj's two column halves; the attention block on K.mtp, V.mtp; s = u
    + a; the latent MoE block (the MTP's experts, its shared expert); names["out"] =
    final_layernorm(m + s). em (the GPU: gpu.NmEmit) gives its quantized rows, products, attention
    and experts as in compile_nemotron_step."""
    from reclank.ir import program as P
    cfg = model.cfg
    prog = P.Program()
    hid, eps = cfg.hidden, float(cfg.eps)
    nq, nk, hd, L = cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.latent
    E, I, k = cfg.moe[min(cfg.moe)]
    f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
    p0, p1 = LM + "mtp.layers.0.", LM + "mtp.layers.1."
    H, hin, out = f32(t, hid), f32(t, hid), f32(t, hid)
    ea, ha, u, u2 = f32(t, hid), f32(t, hid), f32(t, hid), f32(t, hid)
    wide = max(cfg.shared_inner, hid, nq * hd)
    xq, xs, xm = np.zeros((t, wide), np.int8), f32(t, wide // 32), f32(t, wide // 16)
    x16 = X16 and em is None
    xl = np.zeros((t, wide), np.int8) if x16 else None
    hn, q, kb, vb, qout, att, o = f32(t, hid), f32(t, nq * hd), f32(t, nk * hd), f32(t, nk * hd), f32(t, nq * hd), \
        f32(t, nq * hd), f32(t, hid)
    sel, dense = np.zeros((t, 1), np.int32), np.full(t, -1, np.int32)
    logits, val, idx = f32(t, E), f32(t, k), np.zeros((t, k), np.int32)
    lat, lato, sh, mo, o2 = f32(t, L), f32(t, L), f32(t, cfg.shared_inner), f32(t, hid), f32(t, hid)
    mscr = cops.nm_moe_scratch(t, k, L, I) if em is None else None
    prog.names.update(H=H, hin=hin, out=out)
    pos, scores = prog.slot("pos"), prog.slot("scores")

    def quant(src, cols):
        if em is not None:
            em.quant(prog, src, cols)
        if x16:
            prog.emit(P.NM_QX2, src, t, cols, xq, xl, xs, xm)
        elif em is None:
            prog.emit(P.KQ_QUANT, src, t, cols, xq, xs, xm)

    def lin(m, src, out_):
        if em is not None and em.linear(prog, m, src, out_, t):
            pass
        elif x16 and m.type in (KQ_IQ2R, KQ_NV4, KQ_Q8_0, KQ_MXF8):
            prog.emit(P.NM_LIN2, xq, xl, xs, m.data, m.type, m.rows, m.cols, t, out_)
        elif em is not None and m.type == KQ_MXF8 and m.svec is not None:
            prog.emit(P.KQ_LINEAR, xq, xs, xm, src, m.data, m.type, m.rows, m.cols, t, out_, float(m.svec[0]))
            return
        else:
            prog.emit(P.KQ_LINEAR, xq, xs, xm, src, m.data, m.type, m.rows, m.cols, t, out_)
        if m.svec is not None:
            prog.emit(P.MUL, out_, m.svec, out_, t, m.rows, 0)
            prog.keep.append(m.svec)

    # eh_proj [hidden, 2 hidden] as its two column halves (the embedding's first), made once for all the
    # programs (on the GPU each new host array is a new device copy: 134 MB a program)
    if getattr(model, "_eh_halves", None) is None:
        eh = model.F(p0 + "eh_proj.weight")
        # (BF12 rows of the bf16 bits: F widened them)
        halves = [np.ascontiguousarray(eh[:, :hid]), np.ascontiguousarray(eh[:, hid:])]
        bf = model.ck.file(p0 + "eh_proj.weight").dtype(p0 + "eh_proj.weight") == "BF16"
        model._eh_halves = tuple((bf and bf12_mat((h.view(np.uint32) >> 16).astype(np.uint16))) or
                                 Mat(h.reshape(-1), KQ_F32, hid, hid) for h in halves)
    eh_a, eh_b = model._eh_halves
    prog.keep += [eh_a.data, eh_b.data]
    prog.emit(P.RMS_NORM, H, model.F(p0 + "enorm.weight"), ea, t, hid, eps)
    prog.emit(P.RMS_NORM, hin, model.F(p0 + "hnorm.weight"), ha, t, hid, eps)
    lin(eh_a, ea, u)
    lin(eh_b, ha, u2)
    prog.emit(P.ADD, u, u2, u, t * hid)
    # the attention block
    a = p0 + "mixer."
    prog.emit(P.RMS_NORM, u, model.F(p0 + "norm.weight"), hn, t, hid, eps)
    quant(hn, hid)
    lin(model.M(a + "q_proj"), hn, q)
    lin(model.M(a + "k_proj"), hn, kb)
    lin(model.M(a + "v_proj"), hn, vb)
    if em is not None:
        em.attention(prog, "mtp", q, kb, vb, att, t)
    else:
        prog.emit(P.ATTN_PREP, q, kb, vb, None, None, None, None, prog.slot("K.mtp"), prog.slot("V.mtp"),
                  prog.slot("hs"), pos, t, nq, nk, hd, 0, eps, float(hd ** -0.5), qout, None)
        prog.emit(P.ATTN_QSA, qout, prog.slot("K.mtp"), 0, prog.slot("V.mtp"), 0, scores, att,
                  nq, nk, hd, t, pos, sel, dense, 1, 3, prog.slot("hs"))
    quant(att, nq * hd)
    lin(model.M(a + "o_proj"), att, o)
    prog.emit(P.ADD, u, o, u, t * hid)          # (s = u + a, in u)
    # the latent MoE block
    b = p1 + "mixer."
    prog.emit(P.RMS_NORM, u, model.F(p1 + "norm.weight"), hn, t, hid, eps)
    quant(hn, hid)
    lin(model.M(b + "gate"), hn, logits)
    prog.emit(P.ROUTER_SIG, logits, t, E, k, val, idx, model.F(b + "gate.e_score_correction_bias"),
              float(cfg.route_scale), 1e-20)
    lin(model.M(b + "fc1_latent_proj"), hn, lat)

    def shared():
        lin(model.M(b + "shared_experts.up_proj"), hn, sh)
        prog.emit(P.RELU2, sh, sh, t * cfg.shared_inner)
        quant(sh, cfg.shared_inner)
        lin(model.M(b + "shared_experts.down_proj"), sh, o2)

    if em is not None:
        em.moe(prog, "mtp", lat, idx, val, lato, t, side=shared)
    else:
        up, down = model.experts("mtp", "up"), model.experts("mtp", "down")
        prog.emit(P.NM_MOE, lat, idx, val, t, k, up.data, up.type, down.data, down.type, L, I, mscr, lato, int(X16),
                  *model.node1("mtp"))
        prog.keep += [up.data, down.data]
        shared()
    quant(lato, L)
    lin(model.M(b + "fc2_latent_proj"), lato, mo)
    prog.emit(P.ADD, mo, o2, mo, t * hid)       # (m)
    prog.emit(P.ADD, u, mo, u, t * hid)         # (m + s)
    prog.emit(P.RMS_NORM, u, model.F(p1 + "final_layernorm.weight"), out, t, hid, eps)
    prog.keep += [a_ for a_ in (sel, dense, mscr) if a_ is not None]
    prog.tokens = t
    return prog.finish()
