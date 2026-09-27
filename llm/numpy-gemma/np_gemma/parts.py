"""Run a decode step as several programs, one for each part of the machine.

SPLIT_PLAN.md, phase 1. The program of np_gemma/program.py runs a step in one
team of threads. Here the compiler makes one program for each part. On a
machine with NUMA, a part is one node. A call to gemma_run_parts runs all the
parts at the same time, each in its own team of threads.

Each operation of a layer has a place (PLACES):

    all    each part runs the operation on its own buffers. This is for a
           small operation, such as a norm, an add, or the router.
    rows   each part computes a range of the rows of the output. The output
           is one buffer that all parts can read.
    heads  each part runs the attention heads of a range of the key and
           value heads, and the query heads that go with them. This is for
           the projections of the query, the key, and the value, their norm
           and rope, the write to the cache, and the attention.
    one    only part 0 runs the operation. With NP_GEMMA_PART_ATTN=one, the
           operations of the attention use this place in place of heads.

An operation of the place rows, heads, or one writes shared buffers. The
other parts must wait before they read such a buffer. The compiler keeps the
set of the shared buffers that a part wrote since the last barrier. It puts a
barrier across the parts (XBAR) before an operation that reads one of them.

An operation of the place one needs no barrier for a buffer that an
operation of the place one wrote, because part 0 runs both. The same is
true for the place heads: a part reads only the heads that it wrote. An
operation that reads only its own buffers, such as the router, needs no
barrier. Thus the attention of a layer needs no barrier before it, and one
barrier after it, before the output projection.

A range of rows starts at a multiple of ALIGN (16). Then each part computes
each value with the same instructions as the program of one part. Thus the
bits are the same. scripts/check_parts.py checks this.

The experts split by rows too (see gp_moe_part in the C file). Each part
holds a copy of its rows of every expert. The compiler makes the copies one
time for each model and keeps them. Phase 1 splits only a step of one token.

The entry points:

    compile_parts(model, attn, n)             the programs of a step in n parts
    decode_step(model, cache, tokens, ...)    bind, run, return the hidden state

Model.forward uses decode_step when NP_GEMMA_PARTS is 2 or more. The size of
the team of each part is NP_GEMMA_PART_TEAM, or else OMP_NUM_THREADS divided
by the count of parts.
"""
from __future__ import annotations

import os

import numpy as np

from . import cops
from . import program as P

ALIGN = 16

PLACES = {
    "rms_norm_multi4": "rows",
    "int4": "rows",
    "gelu_mul_int4": "rows",
    "moe": "rows",
    # The operations of the attention. In the form of a layer, int4_multi4
    # is the projection of the query, the key, and the value. The operation
    # copy makes the value from the key.
    "int4_multi4": "heads",
    "copy": "heads",
    "qkv_norm_rope": "heads",
    "kv_write": "heads",
    "attn_qc": "heads",
    "attn_f32": "heads",
}

# The attention in part 0 only, the form of the first version. A check can
# compare the two.
PLACES_ONE = dict(PLACES, int4_multi4="rows", copy="all", qkv_norm_rope="one",
                  kv_write="one", attn_qc="one", attn_f32="one")


def ranges(rows, parts, align=ALIGN):
    """Return the range of rows (start, end) of each part. Each start is a
    multiple of align. The last part ends at rows."""
    cut = [0]
    for k in range(1, parts):
        cut.append(min(rows, int(round(rows * k / parts / align)) * align))
    cut.append(rows)
    return [(cut[k], cut[k + 1]) for k in range(parts)]


class PartCompiler(P.Compiler):
    """The compiler of one part. See the module text.

    shared is the list of the buffers that all parts read. Each part asks for
    the shared buffers in the same order, because each part compiles the same
    form. Thus request i of each part gets buffer i. bar is the barrier of the
    parts.
    """

    def __init__(self, model, part, parts, shared, bar):
        super().__init__(model)
        self.part = part
        self.parts = parts
        self._shared = shared
        self._n_shared = 0
        self.bar = bar
        self.places = PLACES_ONE if os.environ.get("NP_GEMMA_PART_ATTN") == "one" else PLACES
        self.layer = None
        # The shared buffers written since the last barrier, by id, with the
        # place of the operation that wrote each one.
        self.dirty = {}
        self._made = []          # the shared buffers of the current operation
        self.place = "all"       # the place of the operation that compiles now

    def shared(self, shape, dtype=np.float32):
        """Return the next shared buffer. The first part makes it."""
        i = self._n_shared
        self._n_shared += 1
        if i == len(self._shared):
            self._shared.append(np.zeros(shape, dtype=dtype))
        b = self._shared[i]
        assert b.shape == tuple(shape), "the parts ask for different shared buffers"
        self._made.append(b)
        return b

    def buffer(self, shape, dtype=np.float32):
        """A new buffer for a result. An operation of the place one writes a
        shared buffer, because the other parts read its result. The other
        operations write the buffers of their part."""
        if self.place == "one":
            return self.shared(shape, dtype)
        return np.zeros(shape, dtype=dtype)

    def compile(self, form):
        """As Compiler.compile. Keep the index of the layer, for the ranges
        of the heads."""
        if form[0] == "layer":
            self.layer = form[1]
        return super().compile(form)

    def heads(self):
        """Return the range of the key and value heads of this part in the
        current layer, (g0, g1), and the query heads for each of them."""
        plan = self.cfg.plan[self.layer]
        g0, g1 = ranges(plan.num_kv_heads, self.parts, 1)[self.part]
        return g0, g1, plan.num_q_heads // plan.num_kv_heads

    def kernel(self, head, vals, out=None):
        place = self.places.get(head, "all")
        reads = [a for a in _arrays(vals) if id(a) in self.dirty]
        if any(place not in ("one", "heads") or self.dirty[id(a)] != place for a in reads):
            self.p.emit(P.XBAR, self.bar, self.parts)
            self.dirty.clear()
        self.place = place
        self._made = []
        fn = PART_KERNELS.get(head) if place != "one" else None
        if place == "heads":
            fn = HEAD_KERNELS[head]
        emit = self.p.emit
        if place == "one" and self.part != 0:
            # The same steps as part 0, so that the shared buffers keep their
            # order, but no records.
            self.p.emit = lambda *a: None
        try:
            if fn is not None:
                assert out is None, "a split operation cannot write an existing buffer"
                r = fn(self, *vals)
            else:
                r = super().kernel(head, vals, out)
        finally:
            self.p.emit = emit
            self.place = "all"
        if place != "all":
            written = list(self._made)
            if place in ("one", "heads"):
                # These operations can change their shared operands in place,
                # such as the rope of the query.
                written += [a for a in _arrays(vals) if any(a is b for b in self._shared)]
            for b in written:
                self.dirty[id(b)] = place
        return r

    def rows(self, a, r0, r1):
        """Return rows r0 to r1 of a weight array: a view, not a copy."""
        return a[r0:r1]


def _arrays(vals):
    """Return the arrays among the operands of an operation. A matrix is a
    tuple of arrays."""
    out = []
    for v in vals:
        if isinstance(v, np.ndarray):
            out.append(v)
        elif isinstance(v, tuple):
            out += _arrays(v)
    return out


# ---- the operations of the place rows --------------------------------------------

def _part_mats(c, mats):
    """Return the operands of up to four int4 matrices for the rows of this
    part, and the shared output of each matrix."""
    args, outs = [], []
    for m in range(4):
        if m < len(mats) and mats[m] is not None:
            w, s = mats[m]
            o = c.shared((1, w.shape[0]))
            r0, r1 = ranges(w.shape[0], c.parts)[c.part]
            outs.append(o)
            if r1 > r0:
                args += [c.rows(w, r0, r1), c.rows(s, r0, r1), o[0, r0:r1], r1 - r0]
                continue
        args += [None, None, None, 0]
    return args, outs


def pk_int4_multi4(c, x, *mats):
    assert x.shape[0] == 1, "the parts split only a step of one token"
    args, outs = _part_mats(c, mats)
    c.p.emit(P.INT4_MULTI4, x, x.shape[1], *args)
    return tuple(outs)


def pk_rms_norm_multi4(c, x, wn, *mats):
    assert x.shape[0] == 1, "the parts split only a step of one token"
    args, outs = _part_mats(c, mats)
    scratch = np.zeros(x.shape[1], dtype=np.float32)
    c.p.emit(P.RMS_NORM_MULTI4, x, np.ascontiguousarray(wn, dtype=np.float32), scratch,
             x.shape[1], float(c.eps), *args)
    return tuple(outs)


def pk_int4(c, mat, x):
    assert x.shape[0] == 1, "the parts split only a step of one token"
    w, s = mat
    out = c.shared((1, w.shape[0]))
    r0, r1 = ranges(w.shape[0], c.parts)[c.part]
    if r1 > r0:
        c.p.emit(P.INT4_LINEAR, x, c.rows(w, r0, r1), c.rows(s, r0, r1), out[0, r0:r1],
                 r1 - r0, x.shape[1])
    return out


def pk_gelu_mul_int4(c, g, u, mat):
    assert g.shape[0] == 1, "the parts split only a step of one token"
    w, s = mat
    out = c.shared((1, w.shape[0]))
    r0, r1 = ranges(w.shape[0], c.parts)[c.part]
    if r1 > r0:
        c.p.emit(P.GELU_MUL_INT4, g, u, g.size, np.zeros(g.size, dtype=np.float32),
                 c.rows(w, r0, r1), c.rows(s, r0, r1), out[0, r0:r1], r1 - r0, g.size)
    return out


def expert_rows(model, layer, part, parts):
    """Return the copy of the rows of one part of the experts of a layer.

    The gate and up matrix of an expert holds inner gate rows, then inner up
    rows. The part takes gate rows a0 to a1 and the up rows with the same
    index, in that order, so the GELU finds the gate and the up of a value at
    the same distance as in the whole matrix. The part takes down rows c0 to
    c1. The model keeps the copies for the next compile.
    """
    key = (layer, part, parts)
    store = model.__dict__.setdefault("_part_experts", {})
    if key in store:
        return store[key]
    w = model._layers[layer]
    gu_q, gu_s = w["experts.gate_up_proj"]
    dn_q, dn_s = w["experts.down_proj"]
    inner = model.cfg.moe_intermediate_size
    a0, a1 = ranges(inner, parts)[part]
    c0, c1 = ranges(dn_q.shape[1], parts)[part]
    gu = [np.ascontiguousarray(np.concatenate([a[:, a0:a1], a[:, inner + a0:inner + a1]],
                                              axis=1)) for a in (gu_q, gu_s)]
    dn = [np.ascontiguousarray(a[:, c0:c1]) for a in (dn_q, dn_s)]
    store[key] = (gu, dn, a0, a1, c0, c1)
    return store[key]


def pk_moe(c, h, val, idx, layer):
    assert h.shape[0] == 1, "the parts split only a step of one token"
    (gu_q, gu_s), (dn_q, dn_s), a0, a1, c0, c1 = expert_rows(c.model, layer, c.part,
                                                            c.parts)
    inner = c.cfg.moe_intermediate_size
    hidden = h.shape[1]
    top_k = idx.size
    ni, nd = a1 - a0, c1 - c0
    assert ni > 0 and nd > 0, "too many parts for the rows of the experts"
    act2 = c.shared((top_k, inner))
    out = c.shared((1, hidden))
    c.p.emit(P.MOE_PART, h, val, idx, top_k, gu_q, gu_s, dn_q, dn_s, 2 * ni, hidden, nd,
             ni, np.zeros(top_k, dtype=np.int32), np.zeros((top_k, 2 * ni), np.float32),
             np.zeros((top_k, ni), np.float32), act2, a0, inner,
             np.zeros((top_k, nd), np.float32), out, c0, c.bar, c.parts)
    return out


PART_KERNELS = {
    "int4_multi4": pk_int4_multi4,
    "rms_norm_multi4": pk_rms_norm_multi4,
    "int4": pk_int4,
    "gelu_mul_int4": pk_gelu_mul_int4,
    "moe": pk_moe,
}


# ---- the operations of the place heads --------------------------------------------

# Part p runs the key and value heads g0 to g1 - 1 and the query heads that
# go with them. The rows of a head are adjacent in the projections, in the
# cache, and in the output of the attention. Thus each operation gets the
# rows of its heads with an offset and a count.

def _kv_rows(c):
    g0, g1, rep = c.heads()
    hd = c.cfg.plan[c.layer].head_dim
    return g0 * hd, g1 * hd, g0 * rep * hd, g1 * rep * hd


def hk_int4_multi4(c, x, *mats):
    """The query, key, and value projections, for the rows of the heads of
    this part. mats is (q, k) or (q, k, v)."""
    assert x.shape[0] == 1, "the parts split only a step of one token"
    k0, k1, q0, q1 = _kv_rows(c)
    args, outs = [], []
    for m in range(4):
        if m < len(mats):
            w, s = mats[m]
            r0, r1 = (q0, q1) if m == 0 else (k0, k1)
            o = c.shared((1, w.shape[0]))
            outs.append(o)
            if r1 > r0:
                args += [c.rows(w, r0, r1), c.rows(s, r0, r1), o[0, r0:r1], r1 - r0]
                continue
        args += [None, None, None, 0]
    if k1 > k0:
        c.p.emit(P.INT4_MULTI4, x, x.shape[1], *args)
    return tuple(outs)


def hk_copy(c, x):
    """The value of a global layer: a copy of the key, for the heads of
    this part."""
    k0, k1, _q0, _q1 = _kv_rows(c)
    out = c.shared(x.shape, x.dtype)
    if k1 > k0:
        c.p.emit(P.COPY, x[0, k0:k1], out[0, k0:k1], (k1 - k0) * x.itemsize)
    return out


def hk_qkv_norm_rope(c, q, k, v, qn, kn, cos, sin, layer):
    k0, k1, q0, q1 = _kv_rows(c)
    hd = c.cfg.plan[layer].head_dim
    nq, nk = (q1 - q0) // hd, (k1 - k0) // hd
    if nk:
        c.p.emit(P.QKV_NORM_ROPE, q[0, q0:q1], np.ascontiguousarray(qn, dtype=np.float32),
                 nq, k[0, k0:k1], np.ascontiguousarray(kn, dtype=np.float32), nk,
                 v[0, k0:k1], nk, cos, sin, nq, nk, hd, float(c.eps))


def hk_kv_write(c, layer, k, v, row, qc=1):
    """As k_kv_write, for the heads of this part. The addresses of the cache
    row get the offset of the first head."""
    k0, k1, _q0, _q1 = _kv_rows(c)
    if k1 == k0:
        return
    plan = c.cfg.plan[layer]
    per = plan.num_kv_heads * plan.head_dim
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    at = lambda name, size, off: c.scalar(  # noqa: E731
        "+", [P._addr(c, s(name), row, size), off])
    if qc:
        q = [at("kq", 2 * per, 2 * k0), at("ks", 4 * (per // 32), 4 * (k0 // 32)),
             at("vq", 2 * per, 2 * k0), at("vs", 4 * (per // 32), 4 * (k0 // 32))]
    else:
        q = [0, 0, 0, 0]
    c.p.emit(P.KV_WRITE, k[0, k0:k1], v[0, k0:k1], at("k", 4 * per, 4 * k0),
             at("v", 4 * per, 4 * k0), *q, k1 - k0)


def hk_attn_qc(c, layer, q, lo, n):
    """As k_attn_qc, for the heads of this part."""
    plan = c.cfg.plan[layer]
    hd, qh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
    per = kvh * hd
    k0, k1, q0, q1 = _kv_rows(c)
    out = c.shared((1, qh * hd))
    if k1 == k0:
        return out
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    at = lambda name, size, off: c.scalar("+", [P._addr(c, s(name), lo, size), off])  # noqa: E731
    c.p.emit(P.ATTN_QC_H, q[0, q0:q1], at("kq", 2 * per, 2 * k0),
             at("ks", 4 * (per // 32), 4 * (k0 // 32)), at("vq", 2 * per, 2 * k0),
             at("vs", 4 * (per // 32), 4 * (k0 // 32)), c.p.slot("scores"), out[0, q0:q1],
             (q1 - q0) // hd, (k1 - k0) // hd, hd, n, per, per // 32)
    return out


def hk_attn_f32(c, layer, q, lo, n):
    """As k_attn_f32, for the heads of this part."""
    plan = c.cfg.plan[layer]
    hd, qh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
    per = kvh * hd
    k0, k1, q0, q1 = _kv_rows(c)
    out = c.shared((1, qh * hd))
    if k1 == k0:
        return out
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    at = lambda name, off: c.scalar("+", [P._addr(c, s(name), lo, 4 * per), off])  # noqa: E731
    base = c.scalar("+", [s("base"), lo])
    c.p.emit(P.ATTN_F32_H, q[0, q0:q1], at("k", 4 * k0), at("v", 4 * k0), c.p.slot("scores"),
             out[0, q0:q1], (q1 - q0) // hd, (k1 - k0) // hd, hd, n, c.p.slot("pos"), base,
             plan.sliding_window or 0, per)
    return out


HEAD_KERNELS = {
    "int4_multi4": hk_int4_multi4,
    "copy": hk_copy,
    "qkv_norm_rope": hk_qkv_norm_rope,
    "kv_write": hk_kv_write,
    "attn_qc": hk_attn_qc,
    "attn_f32": hk_attn_f32,
}


# ---- the programs of a step ------------------------------------------------------

class Parts:
    """The programs of the parts of a step, their shared buffers, and their
    barrier."""

    def __init__(self, progs, shared, bar):
        self.progs = progs
        self.shared = shared
        self.bar = bar
        self.addrs = np.array([p.buf.ctypes.data for p in progs], dtype=np.int64)
        self.team = int(os.environ.get("NP_GEMMA_PART_TEAM", "0"))
        self.layers = progs[0].layers
        self.attn = progs[0].attn
        self.tokens = 1

    def bind(self, model, cache, pos):
        """Prepare the cache and bind the parameters of each part. The cache
        work runs one time. A part gets only the parameters that it uses.
        Each part gets its own scores buffer, because the parts run their
        attention at the same time."""
        kw = P.step_params(self.progs[0], model, cache, pos)
        for k, p in enumerate(self.progs):
            if k > 0 and "scores" in kw:
                sc = getattr(p, "scores", None)
                if sc is None or sc.size < kw["scores"].size:
                    p.scores = np.zeros(kw["scores"].size, dtype=np.float32)
                kw = dict(kw, scores=p.scores)
            p.bind(**{n: v for n, v in kw.items() if n in p.by_name})

    def run(self):
        rc = cops.gp_run_parts(self.addrs, self.team, self.bar)
        if rc != 0:
            raise RuntimeError("gemma_run_parts returned %d" % rc)


def compile_parts(model, attn="qc", n=2):
    """Compile a step of one token into n programs. "x" is the input of each
    part, and "xn" of part 0 is the result."""
    shared = []
    # The count, then the generation on a different cache line.
    bar = np.zeros(16, dtype=np.int64)
    progs = []
    for k in range(n):
        c = PartCompiler(model, k, n, shared, bar)
        c.env["x"] = np.zeros((1, model.cfg.hidden_size), dtype=np.float32)
        c.p.slot("pos")
        c.compile(P.step_form(model, attn, 1))
        c.p.layers = list(range(model.cfg.num_hidden_layers))
        c.p.attn = attn
        c.p.tokens = 1
        c.p.keep.append(bar)
        progs.append(c.p.finish())
    # Each part must wait at the same barriers. An expert operation waits one
    # time inside.
    waits = [sum(op in (P.XBAR, P.MOE_PART) for op, _ in p.recs) for p in progs]
    assert len(set(waits)) == 1, "the parts wait at different barriers: %s" % waits
    return Parts(progs, shared, bar)


def decode_step(model, cache, tokens, pos, attn, n):
    """Run a step of one token in n parts. Return the hidden state after the
    final norm, shape (1, hidden). As program.decode_step."""
    progs = model.__dict__.setdefault("_programs", {})
    key = ("parts", attn, n)
    parts = progs.get(key)
    if parts is None:
        parts = progs[key] = compile_parts(model, attn, n)
    parts.bind(model, cache, pos)
    x = model.embed(tokens)
    for p in parts.progs:
        p.names["x"][:] = x
    parts.run()
    return parts.progs[0].names["xn"].copy()
