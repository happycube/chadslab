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
- bf16 vectors and the routers widened to float32 (the router a KQ_F32 matrix: float x).

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
from reclank.models.nemotron.model import BB, LM, Checkpoint, NemotronConfig

KQ_F32, KQ_Q8_0, KQ_IQ2_XXS, KQ_NV4, KQ_IQ2R = 0, 8, 16, 51, 67
# x of the integer products as int16 in two int8 planes (kq_quant_part16: v = 128 hi + lo), as the
# GPU (gpu.X16); RECLANK_NM_X16=0: int8 x (kq_quant_part)
X16 = os.environ.get("RECLANK_NM_X16", "1") != "0"
# the residual sums and the norm of the next block in one record (RECLANK_NM_ADDNORM=0: ADD, ADD, RMS_NORM)
ADDNORM = os.environ.get("RECLANK_NM_ADDNORM", "1") != "0"


class Mat:
    """A matrix as the records take it: the bytes, the type, rows, cols."""

    def __init__(self, data, type_, rows, cols):
        self.data, self.type, self.rows, self.cols = data, type_, rows, cols


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
        # the names of the device cache of the GPU (gpu35._DevCache)
        self.kv = {i: (self.K[i], self.V[i]) for i in self.K}
        self.state = self.S

    def params(self):
        kw = {}
        for i in self.conv:
            kw["conv.%d" % i], kw["S.%d" % i] = self.conv[i], self.S[i]
        for i in self.K:
            kw["K.%d" % i], kw["V.%d" % i] = self.K[i], self.V[i]
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
        """The weightkeeper's tensor name ("raw.<name>", "nv4.<name>" or "iq2r.<name>"): (its entry,
        its bytes), or None."""
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
        if name + ".weight_scale" in ck.where:
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
                f = np.ascontiguousarray(self._f32(name + ".weight"))
                m = Mat(f.reshape(-1), KQ_F32, f.shape[0], f.shape[1])
        self._m[name] = m
        return m

    def experts(self, i, which):
        """The routed experts of layer i ("up" or "down") in one stack: (Mat of one expert, the
        stack)."""
        key = ("exps", i, which)
        m = self._m.get(key)
        if m is not None:
            return m
        E = self.cfg.moe[i][0]
        k = self._kept("exps.%d.%s" % (i, which))
        if k is not None:
            d, parts = k
            stack = parts[0][1].reshape(-1)         # (the GPU's node, or node 0: whole there)
            rows = self.cfg.moe[i][1] if which == "up" else self.cfg.latent
            cols = self.cfg.latent if which == "up" else self.cfg.moe[i][1]
            m = self._m[key] = Mat(stack, KQ_IQ2R, rows, cols)
            return m
        p = BB + "layers.%d.mixer.experts.%%d.%s_proj.weight" % (i, which)
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

    def logits(self, xn):
        m = self.M(LM + "lm_head")
        t = xn.shape[0]
        xn = np.ascontiguousarray(xn, np.float32)
        xq = np.empty((t, m.cols), np.int8)
        xs = np.empty((t, m.cols // 32), np.float32)
        xm = np.empty((t, m.cols // 16), np.float32)
        if X16:
            return cops.nm_linear2(m.data, m.type, m.rows, m.cols, xn)
        cops.kq_quant_x(xn, xq, xs, xm)
        out = np.empty((t, m.rows), np.float32)
        cops.kq_linear(m.data, m.type, m.rows, m.cols, xq, xs, xm, xn, t, out)
        return out


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
        if x16 and m.type in (KQ_IQ2R, KQ_NV4, KQ_Q8_0):
            if x16b is not None:
                prog.emit(P.NM_LIN2, xq, xl, xs, m.data, m.type, m.rows, m.cols, t, out, x16b)
            else:
                prog.emit(P.NM_LIN2, xq, xl, xs, m.data, m.type, m.rows, m.cols, t, out)
            return
        prog.emit(P.KQ_LINEAR, xq, xs, xm, src, m.data, m.type, m.rows, m.cols, t, out)

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
            if capture and t <= 16:
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
            prog.emit(P.ROUTER_SIG, logits, t, E, kk, val, idx, model.F(p + "gate.e_score_correction_bias"),
                      float(cfg.route_scale), 1e-20)
            lin(model.M(p + "fc1_latent_proj"), hn, lat)
            if em is not None:
                em.moe(prog, i, lat, idx, val, lato, t)
            else:
                up, down = model.experts(i, "up"), model.experts(i, "down")
                prog.emit(P.NM_MOE, lat, idx, val, t, kk, up.data, up.type, down.data, down.type, L, I, mscr, lato,
                          int(x16))
                prog.keep += [up.data, down.data]
            # the shared expert in the full width, on the same normed input (xq still holds it)
            lin(model.M(p + "shared_experts.up_proj"), hn, sh)
            prog.emit(P.RELU2, sh, sh, t * cfg.shared_inner)
            quant(lato, L)
            lin(model.M(p + "fc2_latent_proj"), lato, mo)
            quant(sh, cfg.shared_inner)
            lin(model.M(p + "shared_experts.down_proj"), sh, o)
        pend = (o, mo if kind not in ("mamba", "attention") else None)
    norm(model.F(BB + "norm_f.weight"), xn)
    prog.keep += [a for a in (dense, sel, mscr) if a is not None]
    prog.tokens = t
    return prog.finish()
