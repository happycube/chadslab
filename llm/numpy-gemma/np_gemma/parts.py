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
    one    only part 0 runs the operation. This is for an operation that
           changes shared data in place: the norm and the rope of the query
           and the key, the write to the cache, and the attention.

An operation of the place rows or one writes shared buffers. The other
parts must wait before they read such a buffer. The compiler keeps the set
of the shared buffers that a part wrote since the last barrier. It puts a
barrier across the parts (XBAR) before an operation that reads one of them.
An operation of the place one needs no barrier for a buffer that an
operation of the place one wrote, because part 0 runs both. An operation
that reads only its own buffers, such as the router, needs no barrier.

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
    "int4_multi4": "rows",
    "rms_norm_multi4": "rows",
    "int4": "rows",
    "gelu_mul_int4": "rows",
    "moe": "rows",
    "qkv_norm_rope": "one",
    "kv_write": "one",
    "attn_qc": "one",
    "attn_f32": "one",
}


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

    def kernel(self, head, vals, out=None):
        place = PLACES.get(head, "all")
        reads = [a for a in _arrays(vals) if id(a) in self.dirty]
        if any(place != "one" or self.dirty[id(a)] != "one" for a in reads):
            self.p.emit(P.XBAR, self.bar, self.parts)
            self.dirty.clear()
        self.place = place
        self._made = []
        fn = PART_KERNELS.get(head)
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
            if place == "one":
                # An operation of the place one can change its shared operands
                # in place, such as the rope of the query.
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
    assert x.shape[0] == 1, "phase 1 splits only a step of one token"
    args, outs = _part_mats(c, mats)
    c.p.emit(P.INT4_MULTI4, x, x.shape[1], *args)
    return tuple(outs)


def pk_rms_norm_multi4(c, x, wn, *mats):
    assert x.shape[0] == 1, "phase 1 splits only a step of one token"
    args, outs = _part_mats(c, mats)
    scratch = np.zeros(x.shape[1], dtype=np.float32)
    c.p.emit(P.RMS_NORM_MULTI4, x, np.ascontiguousarray(wn, dtype=np.float32), scratch,
             x.shape[1], float(c.eps), *args)
    return tuple(outs)


def pk_int4(c, mat, x):
    assert x.shape[0] == 1, "phase 1 splits only a step of one token"
    w, s = mat
    out = c.shared((1, w.shape[0]))
    r0, r1 = ranges(w.shape[0], c.parts)[c.part]
    if r1 > r0:
        c.p.emit(P.INT4_LINEAR, x, c.rows(w, r0, r1), c.rows(s, r0, r1), out[0, r0:r1],
                 r1 - r0, x.shape[1])
    return out


def pk_gelu_mul_int4(c, g, u, mat):
    assert g.shape[0] == 1, "phase 1 splits only a step of one token"
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
    assert h.shape[0] == 1, "phase 1 splits only a step of one token"
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
        work runs one time. A part gets only the parameters that it uses."""
        kw = P.step_params(self.progs[0], model, cache, pos)
        for p in self.progs:
            p.bind(**{k: v for k, v in kw.items() if k in p.by_name})

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
