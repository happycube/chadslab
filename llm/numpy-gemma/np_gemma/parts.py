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
    cols   the paired split (NP_GEMMA_PART_PAIRED=1): each part multiplies
           the columns of the input that it computed itself (its attention
           heads for the output projection, its rows of the gate and the up
           map for the down map) into a sum of the whole output, and the
           parts then add their sums (one barrier).

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

With the KQ_Q4X copies of the model (ops.q4x_pack_model), a part reads the
rows of those copies (groups of 16 rows, float32 activations), and its
copies of the experts are KQ_Q4X too.

On a machine with NUMA, each part reads weights in the memory of its own node
(np_gemma/numa.py). compile_parts finds the node of the team of each part.
The rows of the int4 matrices and of the experts that a part reads are then
copies on that node. The model keeps the copies for the next compile.
NP_GEMMA_NUMA=0 turns this off, and the parts then read views of the model
as before. The cache and the small weights of the place all stay where they
are.

The output head runs in the parts too. Each part computes its range of the
rows of the tied head from a copy on its node, and writes them into one
logits buffer whose pages are on the head node, the node of part 0. The main
thread, which reads the logits, is pinned to that node. Model.logits then
gives that buffer for the hidden state of the step. The rows have the kernel
of Model.logits, so the logits have its bits:

    a Q6_K head (the QAT GGUFs of Google)   Q6K_LINEAR, float32 x
    a Q4_0 head with a KQ_Q4X copy          KQ_QUANT of the part's own
                                            hidden state, then KQ_LINEAR

Another form of head stays in Model.logits. NP_GEMMA_PART_HEAD=0 turns this
off.

Each part also reads its own copy of the router of each layer (the float32
projection and its two scales, 44 MB for the 26B), because each part runs
the whole router.

The cache of the parts (PartKVCache) holds the rows of the KV heads of each
part in a buffer of that part, on its node. A part of a decode step writes
and reads only its own buffer. A prompt block runs its attention in the parts
too (PART_PREFILL): each part writes its rows, then runs the flash kernel for
its heads. The other work of a prompt stays in one team. With NP_GEMMA_PARTS
of 2 or more, KVCache(...) gives a PartKVCache (model.KVCache.__new__).

The rows of a part need not be the same share for each part (the balance).
A part on a slower node, or a GPU paired with a CPU, should get fewer rows.
The first NP_GEMMA_PART_BALANCE steps (default 8, after 2 to warm up) of a
new program run with the time of each record (gemma_run_parts_prof). Each
record has a kind (Program.rec_kind): rows (the operations split by rows:
the matrices, the experts, and the output head), xbar (the barriers), or
fixed (the attention of the heads, the norms, the other operations, whose
share does not move). From the time of part p, F_p fixed and R_p for the
rows at the share s_p, the time of a share is u_p = R_p / s_p, and the
shares s'_p = (T - F_p) / u_p with the sum 1 make the parts end at the same
time T (balance_shares). The program is then compiled again with those
shares (ranges with weights, cuts at multiples of 32), and the copies of
the old ranges go. A split by rows gives the same bits at each cut, so the
balance does not change the bits (with the paired split it changes the
order of the sums of the down map, as any split does). NP_GEMMA_PART_WEIGHTS
(as "1,1.1") gives the shares and skips the measure; NP_GEMMA_PART_BALANCE=0
keeps the even split.

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
from . import numa
from . import ops
from . import program as P
from .model import KVCache

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

# The paired split: the output projection and the down map split by columns.
# A layer of the 12B then has two barriers (the sums of the two), not four.
# The sums of the parts change the order of the additions, so the bits are
# not those of one part.
PLACES_PAIRED = dict(PLACES, int4="cols", gelu_mul_int4="cols")

# The attention in part 0 only, the form of the first version. A check can
# compare the two.
PLACES_ONE = dict(PLACES, int4_multi4="rows", copy="all", qkv_norm_rope="one",
                  kv_write="one", attn_qc="one", attn_f32="one")


def ranges(rows, parts, align=ALIGN, weights=None):
    """Return the range of rows (start, end) of each part. Each start is a
    multiple of align. The last part ends at rows. weights gives the share
    of each part (the balance); None is the same share for each."""
    cut = [0]
    if weights is None:
        for k in range(1, parts):
            cut.append(min(rows, int(round(rows * k / parts / align)) * align))
    else:
        total, acc = float(sum(weights)), 0.0
        for k in range(1, parts):
            acc += weights[k - 1]
            cut.append(max(cut[-1], min(rows, int(round(rows * acc / total / align)) * align)))
    cut.append(rows)
    return [(cut[k], cut[k + 1]) for k in range(parts)]


# The cuts of a split with weights: whole blocks of 32 values, so that a cut
# of the gate rows is also a cut of the columns of the down map (the paired
# split), and whole groups of 16 rows (KQ_Q4X).
WALIGN = 32


class PartCompiler(P.Compiler):
    """The compiler of one part. See the module text.

    shared is the list of the buffers that all parts read. Each part asks for
    the shared buffers in the same order, because each part compiles the same
    form. Thus request i of each part gets buffer i. bar is the barrier of the
    parts.
    """

    def __init__(self, model, part, parts, shared, bar, node=None, split_kv=False,
                 paired=False, weights=None, nodes=None):
        super().__init__(model)
        self.nodes = nodes or [None] * parts   # the node of each part
        self.weights = weights   # the share of each part of the rows (the balance), or None
        self.kinds = {}          # record index -> rows, fixed, or xbar (Program.rec_kind)
        self.used = set()        # the keys of the copies that this program reads
        self.part = part
        self.parts = parts
        self.node = node         # the NUMA node of the weights, or None
        self.split_kv = split_kv  # the cache of the part holds only its heads (PartKVCache)
        self._shared = shared
        self._n_shared = 0
        self.bar = bar
        self.places = PLACES_ONE if os.environ.get("NP_GEMMA_PART_ATTN") == "one" else PLACES
        if paired:
            self.places = dict(self.places, int4="cols", gelu_mul_int4="cols")
        self.layer = None
        # The shared buffers written since the last barrier, by id, with the
        # place of the operation that wrote each one.
        self.dirty = {}
        self._made = []          # the shared buffers of the current operation
        self.place = "all"       # the place of the operation that compiles now

    def shared(self, shape, dtype=np.float32, node=None):
        """Return the next shared buffer. The first part makes it, in the
        memory of node when node is given."""
        i = self._n_shared
        self._n_shared += 1
        if i == len(self._shared):
            if node is None:
                self._shared.append(np.zeros(shape, dtype=dtype))
            else:
                b = numa.empty_on(shape, dtype, node)
                b[...] = 0
                self._shared.append(b)
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

    def rranges(self, rows):
        """The ranges of the rows of a split by rows: the same share for each
        part, or the shares of the balance (self.weights)."""
        if self.weights is None:
            return ranges(rows, self.parts)
        return ranges(rows, self.parts, WALIGN, self.weights)

    def heads(self):
        """Return the key and value heads (g0, g1) and the query heads (h0,
        h1) of this part in the current layer (head_split)."""
        return head_split(self.cfg.plan[self.layer], self.parts, self.part, self.split_kv)

    def qsplit(self):
        """True when the parts of the current layer split the query heads and
        share key and value heads (head_split)."""
        return self.split_kv and self.cfg.plan[self.layer].num_kv_heads < self.parts

    def head_out(self, shape, dtype=np.float32):
        """The output of a projection of the place heads: a shared buffer,
        or with qsplit a buffer of the part, because the parts compute the
        same rows of a shared key head and only the part reads them."""
        return self.buffer(shape, dtype) if self.qsplit() else self.shared(shape, dtype)

    def kernel(self, head, vals, out=None):
        place = self.places.get(head, "all")
        reads = [a for a in _arrays(vals) if id(a) in self.dirty]
        if place == "cols":
            # A part reads only the columns that it wrote (its heads, or its
            # rows of the gate and the up map).
            need = any(self.dirty[id(a)] not in ("rows", "heads") for a in reads)
        else:
            need = any(place not in ("one", "heads") or self.dirty[id(a)] != place
                       for a in reads)
        n0 = len(self.p.recs)
        if need:
            self.p.emit(P.XBAR, self.bar, self.parts)
            self.dirty.clear()
        self.place = place
        self._made = []
        fn = PART_KERNELS.get(head) if place != "one" else None
        if place == "heads":
            fn = HEAD_KERNELS[head]
        elif place == "cols":
            fn = COL_KERNELS[head]
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
        # The kind of each record, for the balance: the rows move with the
        # shares (the paired down map follows the gate rows); the rest does
        # not.
        moves = place == "rows" or (place == "cols" and head == "gelu_mul_int4")
        for pc in range(n0, len(self.p.recs)):
            op = self.p.recs[pc][0]
            self.kinds[pc] = "xbar" if op == P.XBAR else ("rows" if moves else "fixed")
        if place not in ("all", "cols"):
            # A paired operation (cols) ends with its barrier and gives a
            # buffer of the part.
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

    def mat_rows(self, w, s, r0, r1):
        """The operands (w, s) of rows r0 to r1 of an int4 matrix: the rows of
        its KQ_Q4X copy and no scales (ops.q4x_pack_model; r0 and r1 are
        multiples of ALIGN, so the rows are whole groups of 16), else views
        of w and s. A row has the operations of the program of one part:
        the same bits."""
        q = P.ops._Q4X.get(w.ctypes.data) if self.q4x else None
        if q is not None:
            rb = w.shape[1] * 18
            return self.local(q, r0 * rb, r1 * rb), None
        return self.local(w, r0, r1), self.local(s, r0, r1)

    def mat_cols(self, w, s, c0, c1):
        """The operands (w, s) of columns c0 to c1 of an int4 matrix (c0 and
        c1 multiples of 32): a matrix of all the rows and c1 - c0 columns,
        as KQ_Q4X with no scales when the model has the KQ_Q4X copies, on
        the node of the part. The model keeps the copies."""
        assert c0 % 32 == 0 and c1 % 32 == 0, "a column range must be whole blocks"
        store = self.model.__dict__.setdefault("_part_rows", {})
        key = (w.ctypes.data, w.shape, "cols", c0, c1, self.node)
        self.used.add(key)
        if key not in store:
            ws = np.ascontiguousarray(w[:, c0 // 32:c1 // 32])
            ss = np.ascontiguousarray(s[:, c0 // 32:c1 // 32])
            if self.q4x and P.ops._Q4X.get(w.ctypes.data) is not None:
                pair = (cops.kq_q4x_pack(ws, ss), None)
            else:
                pair = (ws, ss)
            if self.node is not None:
                pair = tuple(None if a is None else numa.copy_on(a, self.node) for a in pair)
            store[key] = pair
        return store[key]

    def local(self, a, r0, r1):
        """Rows r0 to r1 of a weight array a: a view, or with a node a copy in
        the memory of the node (numa.copy_on). The model keeps the copies,
        one for each array, range, and node."""
        if self.node is None:
            return self.rows(a, r0, r1)
        store = self.model.__dict__.setdefault("_part_rows", {})
        key = (a.ctypes.data, a.shape, r0, r1, self.node)
        self.used.add(key)
        if key not in store:
            store[key] = numa.copy_on(a[r0:r1], self.node)
        return store[key]


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
            r0, r1 = c.rranges(w.shape[0])[c.part]
            outs.append(o)
            if r1 > r0:
                args += [*c.mat_rows(w, s, r0, r1), o[0, r0:r1], r1 - r0]
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
    r0, r1 = c.rranges(w.shape[0])[c.part]
    if r1 > r0:
        c.p.emit(P.INT4_LINEAR, x, *c.mat_rows(w, s, r0, r1), out[0, r0:r1],
                 r1 - r0, x.shape[1])
    return out


def pk_gelu_mul_int4(c, g, u, mat):
    assert g.shape[0] == 1, "the parts split only a step of one token"
    w, s = mat
    out = c.shared((1, w.shape[0]))
    r0, r1 = c.rranges(w.shape[0])[c.part]
    if r1 > r0:
        c.p.emit(P.GELU_MUL_INT4, g, u, g.size, np.zeros(g.size, dtype=np.float32),
                 *c.mat_rows(w, s, r0, r1), out[0, r0:r1], r1 - r0, g.size)
    return out


def expert_rows(model, layer, part, parts, node=None, weights=None):
    """Return the copy of the rows of one part of the experts of a layer.

    The gate and up matrix of an expert holds inner gate rows, then inner up
    rows. The part takes gate rows a0 to a1 and the up rows with the same
    index, in that order, so the GELU finds the gate and the up of a value at
    the same distance as in the whole matrix. The part takes down rows c0 to
    c1. With a node, the copies are in the memory of that node. weights gives
    the shares of the balance (ranges). The model keeps the copies for the
    next compile.
    """
    key = (layer, part, parts, node, None if weights is None else tuple(weights))
    store = model.__dict__.setdefault("_part_experts", {})
    if key in store:
        return store[key]
    w = model._layers[layer]
    gu_q, gu_s = w["experts.gate_up_proj"]
    dn_q, dn_s = w["experts.down_proj"]
    inner = model.cfg.moe_intermediate_size
    if weights is None:
        a0, a1 = ranges(inner, parts)[part]
        c0, c1 = ranges(dn_q.shape[1], parts)[part]
    else:
        a0, a1 = ranges(inner, parts, WALIGN, weights)[part]
        c0, c1 = ranges(dn_q.shape[1], parts, WALIGN, weights)[part]
    gu = [np.ascontiguousarray(np.concatenate([a[:, a0:a1], a[:, inner + a0:inner + a1]],
                                              axis=1)) for a in (gu_q, gu_s)]
    dn = [np.ascontiguousarray(a[:, c0:c1]) for a in (dn_q, dn_s)]
    if P.ops._Q4X_MOE.get(gu_q.ctypes.data) is not None:
        # The KQ_Q4X copies of these rows (gp_moe_part reads them with no
        # scales): 2 ni rows and nd rows for each expert.
        gu = [cops.kq_q4x_pack(*gu), None]
        dn = [cops.kq_q4x_pack(*dn), None]
    if node is not None:
        gu = [None if a is None else numa.copy_on(a, node) for a in gu]
        dn = [None if a is None else numa.copy_on(a, node) for a in dn]
    store[key] = (gu, dn, a0, a1, c0, c1)
    return store[key]


def pk_moe(c, h, val, idx, layer):
    assert h.shape[0] == 1, "the parts split only a step of one token"
    (gu_q, gu_s), (dn_q, dn_s), a0, a1, c0, c1 = expert_rows(c.model, layer, c.part,
                                                            c.parts, c.node, c.weights)
    c.used.add(("experts", layer, c.part, c.parts, c.node,
                None if c.weights is None else tuple(c.weights)))
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


def pk_router(c, x, layer):
    """As k_router for one token, with the copies of the router weights on
    the node of the part."""
    assert x.shape[0] == 1, "the parts split only a step of one token"
    w = c.model._layers[layer]
    cfg = c.cfg

    def own(a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        return c.local(a, 0, a.shape[0])

    proj = own(w["router.proj"])
    top_k = cfg.top_k_experts
    val = c.buffer(top_k)
    idx = np.zeros(top_k, dtype=np.int32)
    c.p.emit(P.ROUTER, x, own(w["router.scale"]), proj, own(w["router.per_expert_scale"]),
             x.shape[1], proj.shape[0], top_k, float(c.eps), float(cfg.hidden_size ** -0.5),
             val, idx, c.buffer(x.shape[1]), c.buffer(proj.shape[0]))
    return val, idx


PART_KERNELS = {
    "router": pk_router,
    "int4_multi4": pk_int4_multi4,
    "rms_norm_multi4": pk_rms_norm_multi4,
    "int4": pk_int4,
    "gelu_mul_int4": pk_gelu_mul_int4,
    "moe": pk_moe,
}


# ---- the operations of the place heads --------------------------------------------

# Part p runs the key and value heads g0 to g1 - 1 and the query heads h0 to
# h1 - 1 (head_split). The rows of a head are adjacent in the projections,
# in the cache, and in the output of the attention. Thus each operation gets
# the rows of its heads with an offset and a count.

def head_split(plan, parts, part, split_kv=False):
    """Return (g0, g1, h0, h1): the key and value heads and the query heads
    of a part in a layer.

    The parts split the key and value heads, with the query heads of each.
    A layer with fewer key and value heads than parts (a global layer of
    the 12B has one) would leave a part with no work. With a PartKVCache
    (split_kv), the parts then split the query heads instead, and each part
    computes and keeps the key and value heads of its query heads: a shared
    head is in the cache of each part that needs it."""
    kvh, qh = plan.num_kv_heads, plan.num_q_heads
    rep = qh // kvh
    if split_kv and kvh < parts:
        h0, h1 = ranges(qh, parts, 1)[part]
        g0, g1 = h0 // rep, -(-h1 // rep)
        assert g1 - g0 <= 1 or (h0 % rep == 0 and h1 % rep == 0), \
            "the query heads of a part must be whole groups or one group"
        return g0, g1, h0, h1
    g0, g1 = ranges(kvh, parts, 1)[part]
    return g0, g1, g0 * rep, g1 * rep


def _kv_rows(c):
    g0, g1, h0, h1 = c.heads()
    hd = c.cfg.plan[c.layer].head_dim
    return g0 * hd, g1 * hd, h0 * hd, h1 * hd


def _cache_row(c, layer, k0, k1):
    """The values in a row of the cache that this part reads (per), and the
    offset of its first head in the row. The cache of all heads (KVCache)
    has the rows of every head; the cache of the part (PartKVCache) has only
    the rows of its own heads, from offset 0."""
    if c.split_kv:
        return k1 - k0, 0
    plan = c.cfg.plan[layer]
    return plan.num_kv_heads * plan.head_dim, k0


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
            o = c.head_out((1, w.shape[0]))
            outs.append(o)
            if r1 > r0:
                args += [*c.mat_rows(w, s, r0, r1), o[0, r0:r1], r1 - r0]
                continue
        args += [None, None, None, 0]
    if k1 > k0:
        c.p.emit(P.INT4_MULTI4, x, x.shape[1], *args)
    return tuple(outs)


def hk_copy(c, x):
    """The value of a global layer: a copy of the key, for the heads of
    this part."""
    k0, k1, _q0, _q1 = _kv_rows(c)
    out = c.head_out(x.shape, x.dtype)
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
    per, c0 = _cache_row(c, layer, k0, k1)
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    at = lambda name, size, off: c.scalar(  # noqa: E731
        "+", [P._addr(c, s(name), row, size), off])
    if qc:
        q = [at("kq", 2 * per, 2 * c0), at("ks", 4 * (per // 32), 4 * (c0 // 32)),
             at("vq", 2 * per, 2 * c0), at("vs", 4 * (per // 32), 4 * (c0 // 32))]
    else:
        q = [0, 0, 0, 0]
    # KVCache has no float rows: null addresses for them
    c.p.emit(P.KV_WRITE, k[0, k0:k1], v[0, k0:k1], 0, 0, *q, k1 - k0)


def hk_attn_qc(c, layer, q, lo, n):
    """As k_attn_qc, for the heads of this part."""
    plan = c.cfg.plan[layer]
    hd, qh = plan.head_dim, plan.num_q_heads
    k0, k1, q0, q1 = _kv_rows(c)
    out = c.shared((1, qh * hd))
    if k1 == k0:
        return out
    per, c0 = _cache_row(c, layer, k0, k1)
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    at = lambda name, size, off: c.scalar("+", [P._addr(c, s(name), lo, size), off])  # noqa: E731
    c.p.emit(P.ATTN_QC_H, q[0, q0:q1], at("kq", 2 * per, 2 * c0),
             at("ks", 4 * (per // 32), 4 * (c0 // 32)), at("vq", 2 * per, 2 * c0),
             at("vs", 4 * (per // 32), 4 * (c0 // 32)), c.p.slot("scores"), out[0, q0:q1],
             (q1 - q0) // hd, (k1 - k0) // hd, hd, n, per, per // 32,
             plan.num_q_heads // plan.num_kv_heads)
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


def pk_head(c, xn, head, out):
    """The rows of the output head of this part into out, the logits buffer
    on the head node. head is (form, bytes) of part_head."""
    form, w = head
    cols, rows = xn.shape[1], out.shape[1]
    r0, r1 = c.rranges(rows)[c.part]
    if form == "q6k":
        if r1 > r0:
            c.p.emit(P.Q6K_LINEAR, c.local(w, r0, r1), xn, out[0, r0:r1], r1 - r0, cols, 1)
        return
    rb = cols // 32 * 18
    xq = (c.buffer((1, cols), np.int8), c.buffer((1, cols // 32)), c.buffer((1, cols // 16)))
    c.p.emit(P.KQ_QUANT, xn, 1, cols, *xq)
    if r1 > r0:
        c.p.emit(P.KQ_LINEAR, *xq, xn, c.local(w, r0 * rb, r1 * rb), cops.KQ_Q4X, r1 - r0,
                 cols, 1, out[0, r0:r1])


def part_head(model):
    """The tied output head for the parts: ("q6k", the Q6_K rows (rows, row
    bytes)) or ("q4x", the KQ_Q4X bytes), or None (another form of head, or
    NP_GEMMA_PART_HEAD=0)."""
    if os.environ.get("NP_GEMMA_PART_HEAD", "1") == "0":
        return None
    if model._embed_q6k_bytes is not None:
        return "q6k", model._embed_q6k_bytes
    if model._embed_q is not None:
        qx = P.ops._Q4X_HEAD.get(model._embed_q.ctypes.data)
        if qx is not None:
            return "q4x", qx
    return None


# ---- the operations of the place cols (the paired split) ----------------------

def _sum_parts(c, rows, emit_part):
    """The sum of the parts: each part writes its sum of the whole output
    (emit_part), and after a barrier each part adds the sums in the order of
    the parts, so all the parts get the same bits.

    The sums have a copy on the node of each part. A part writes its sum
    into the copy on its node, then copies it into the copy of each other
    part before the barrier (stores across the link, which do not stall the
    core). After the barrier each part reads only the copy on its node."""
    sums = [c.shared((c.parts, rows), node=c.nodes[k]) for k in range(c.parts)]
    mine = sums[c.part]
    emit_part(mine[c.part])
    for k in range(c.parts):
        if k != c.part:
            c.p.emit(P.COPY, mine[c.part], sums[k][c.part], rows * 4)
    c.p.emit(P.XBAR, c.bar, c.parts)
    c.dirty.clear()
    out = c.buffer((1, rows))
    c.p.emit(P.ADD, mine[0], mine[1], out[0], rows)
    for k in range(2, c.parts):
        c.p.emit(P.ADD, out[0], mine[k], out[0], rows)
    return out


def ck_int4(c, mat, x):
    """The output projection: the columns of the query heads of this part
    (head_split), the attention outputs that this part computed."""
    assert x.shape[0] == 1, "the parts split only a step of one token"
    plan = c.cfg.plan[c.layer]
    assert x.shape[1] == plan.num_q_heads * plan.head_dim, \
        "the paired int4 is the output projection"
    w, s = mat
    _k0, _k1, q0, q1 = _kv_rows(c)
    rows = w.shape[0]

    def emit(o):
        # A part with no query heads in the layer (one KV head and one
        # KVCache) leaves its row of the sums at 0 (c.shared is zeros).
        if q1 > q0:
            c.p.emit(P.INT4_LINEAR, x[0, q0:q1], *c.mat_cols(w, s, q0, q1), o, rows, q1 - q0)
    return _sum_parts(c, rows, emit)


def ck_gelu_mul_int4(c, g, u, mat):
    """The down map: the columns of the rows of the gate and the up map of
    this part (the ranges of rms_norm_multi4)."""
    assert g.shape[0] == 1, "the parts split only a step of one token"
    w, s = mat
    r0, r1 = c.rranges(g.shape[1])[c.part]
    assert r1 > r0, "a part with no rows of the gate"
    rows, n = w.shape[0], r1 - r0
    return _sum_parts(c, rows, lambda o: c.p.emit(
        P.GELU_MUL_INT4, g[0, r0:r1], u[0, r0:r1], n, np.zeros(n, dtype=np.float32),
        *c.mat_cols(w, s, r0, r1), o, rows, n))


COL_KERNELS = {
    "int4": ck_int4,
    "gelu_mul_int4": ck_gelu_mul_int4,
}


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
        self.nodes = [None] * len(progs)
        self.logits = None       # the logits of the step, when the parts run the head
        self.split_kv = False    # the parts read a PartKVCache
        self.weights = None      # the shares of the rows (None: the same for each part)
        self.used = set()        # the keys of the copies of the programs
        self.calib = None        # the measure of the balance: [steps, ms], or None

    def bind(self, model, cache, pos):
        """Prepare the cache and bind the parameters of each part. The cache
        work runs one time. A part gets only the parameters that it uses.
        Each part gets its own scores buffer, because the parts run their
        attention at the same time."""
        kw = P.step_params(self.progs[0], model, cache, pos)
        base = kw
        for k, p in enumerate(self.progs):
            if self.split_kv:
                # The cache of the part: the rows of its own heads.
                kw = dict(base)
                for i in self.layers:
                    for name, a in zip(("kq", "ks", "vq", "vs"), cache.part(i, k)):
                        kw["%s.%d" % (name, i)] = a
            if k > 0 and "scores" in kw:
                sc = getattr(p, "scores", None)
                if sc is None or sc.size < kw["scores"].size:
                    p.scores = np.zeros(kw["scores"].size, dtype=np.float32)
                kw = dict(kw, scores=p.scores)
            p.bind(**{n: v for n, v in kw.items() if n in p.by_name})

    def run_prof(self, ms):
        """run, and add the ms of each record of part p to ms[p, pc]
        (gemma_run_parts_prof). ms is (parts, records) float64."""
        rc = cops.gp_run_parts_prof(self.addrs, self.team, self.bar, ms)
        if rc != 0:
            raise RuntimeError("gemma_run_parts_prof returned %d" % rc)

    def run(self):
        rc = cops.gp_run_parts(self.addrs, self.team, self.bar)
        if rc != 0:
            raise RuntimeError("gemma_run_parts returned %d" % rc)


def _xbar_prefetch(p, copies):
    """Give each XBAR record of the program p the weights of the next
    operation that reads a copy of the part (copies: address -> bytes): the
    first such operand in the records after the barrier, before the next
    barrier or MOE_PART (the experts of a token are not known before). The
    threads prefetch them during the wait (gp_xbar)."""
    recs = p.recs
    for i, (op, args) in enumerate(recs):
        if op != P.XBAR:
            continue
        for op2, args2 in recs[i + 1:]:
            if op2 in (P.XBAR, P.MOE_PART):
                break
            hit = next((v for tag, v in args2 if tag == P.T_INT and v in copies), None)
            if hit is not None:
                args[2:] = [(P.T_INT, hit), (P.T_INT, copies[hit])]
                break


def compile_parts(model, attn="qc", n=2, split_kv=False, paired=False, weights=None):
    """Compile a step of one token into n programs. "x" is the input of each
    part, and "xn" of part 0 is the result. split_kv: the cache is a
    PartKVCache of n parts. paired: the paired split (PLACES_PAIRED).
    weights: the shares of the rows of each part (the balance), or None."""
    shared = []
    # The flags of the barrier (gp_xbar): one cache line of 8 int64 for each
    # part, after a line that is not used.
    bar = np.zeros(8 * (n + 1), dtype=np.int64)
    nodes = [None] * n
    if numa.enabled():
        nodes = numa.part_nodes(n, int(os.environ.get("NP_GEMMA_PART_TEAM", "0")))
    head = part_head(model)
    logits = None
    if head is not None:
        # The pages of the logits come on the head node when the parts first
        # write them.
        node = nodes[0] if nodes[0] is not None else 0
        cols = model.cfg.hidden_size
        rows = head[1].shape[0] if head[0] == "q6k" else head[1].nbytes // (cols // 32 * 18)
        logits = numa.empty_on((1, rows), np.float32, node)
        if nodes[0] is not None:
            numa.pin_thread(node)
    progs = []
    used = set()
    for k in range(n):
        c = PartCompiler(model, k, n, shared, bar, nodes[k], split_kv, paired, weights, nodes)
        c.env["x"] = np.zeros((1, model.cfg.hidden_size), dtype=np.float32)
        c.p.slot("pos")
        c.compile(P.step_form(model, attn, 1))
        if logits is not None:
            n0 = len(c.p.recs)
            pk_head(c, c.env["xn"], head, logits)
            for pc in range(n0, len(c.p.recs)):
                c.kinds[pc] = "rows"
        c.p.rec_kind = [c.kinds.get(pc, "fixed") for pc in range(len(c.p.recs))]
        if os.environ.get("NP_GEMMA_PART_PREFETCH", "1") != "0":
            copies = {}
            for v in model.__dict__.get("_part_rows", {}).values():
                for a in (v if isinstance(v, tuple) else (v,)):
                    if a is not None:
                        copies[a.ctypes.data] = a.nbytes
            _xbar_prefetch(c.p, copies)
        used |= c.used
        c.p.layers = list(range(model.cfg.num_hidden_layers))
        c.p.attn = attn
        c.p.tokens = 1
        c.p.keep.append(bar)
        progs.append(c.p.finish())
    # Each part must wait at the same barriers. An expert operation waits one
    # time inside.
    waits = [sum(op in (P.XBAR, P.MOE_PART) for op, _ in p.recs) for p in progs]
    assert len(set(waits)) == 1, "the parts wait at different barriers: %s" % waits
    parts = Parts(progs, shared, bar)
    parts.nodes = nodes
    parts.logits = logits
    parts.split_kv = split_kv
    parts.weights = weights
    parts.used = used
    return parts


def balance_shares(ms, kinds, shares, lo=0.05, waits=None):
    """The shares of the rows that make the parts end at the same time.

    ms (parts, records): the time of each record of each part over some
    steps; kinds: the kind of each record of each part (Program.rec_kind);
    shares: the shares of the rows of that run. Part p took F_p for the
    fixed records and R_p for the rows. A share s takes s R_p / s_p, so the
    parts end at the same time T when s'_p = (T - F_p) u_p^-1, u_p = R_p /
    s_p, and the s'_p add to 1: T = (1 + sum F_p / u_p) / sum 1 / u_p. A
    share stays at lo or more. waits (ms for each part) is the wait inside
    the records of the rows (the barrier of MOE_PART), taken from R. Return
    the new shares, and the measured F and R."""
    shares = np.asarray(shares, dtype=np.float64) / float(np.sum(shares))
    n = len(shares)
    F = np.array([sum(ms[p, pc] for pc, k in enumerate(kinds[p]) if k == "fixed")
                  for p in range(n)])
    R = np.array([sum(ms[p, pc] for pc, k in enumerate(kinds[p]) if k == "rows")
                  for p in range(n)])
    if waits is not None:
        R = np.maximum(R - np.asarray(waits, dtype=np.float64), 0.05 * R)
    u = R / shares
    T = (1.0 + np.sum(F / u)) / np.sum(1.0 / u)
    new = np.clip((T - F) / u, lo, None)
    return new / new.sum(), F, R


def _balance_env():
    """(steps of the measure, the shares given), from NP_GEMMA_PART_BALANCE
    and NP_GEMMA_PART_WEIGHTS."""
    given = os.environ.get("NP_GEMMA_PART_WEIGHTS")
    w = [float(v) for v in given.split(",")] if given else None
    return int(os.environ.get("NP_GEMMA_PART_BALANCE", "8")), w


def _rebalance(model, key, parts, attn, n, split, paired, weights):
    """Compile the parts again with the shares weights, and let the copies
    that only the old programs read go (each program keeps its arrays, so a
    copy goes when no program reads it)."""
    progs = model.__dict__["_programs"]
    new = compile_parts(model, attn, n, split, paired, [float(v) for v in weights])
    progs[key] = new
    keep = set()
    for other in progs.values():
        keep |= getattr(other, "used", set())
    rows = model.__dict__.get("_part_rows", {})
    for k in [k for k in rows if k not in keep]:
        del rows[k]
    experts = model.__dict__.get("_part_experts", {})
    for k in [k for k in experts if ("experts",) + tuple(k) not in keep]:
        del experts[k]
    return new


def decode_step(model, cache, tokens, pos, attn, n):
    """Run a step of one token in n parts. Return the hidden state after the
    final norm, shape (1, hidden). As program.decode_step.

    A new program measures its first steps and is then compiled again with
    the shares of the balance (see the module text)."""
    progs = model.__dict__.setdefault("_programs", {})
    split = cache.split
    if split:
        assert cache.parts == n, "the cache has %d parts, the step %d" % (cache.parts, n)
    paired = os.environ.get("NP_GEMMA_PART_PAIRED", "0") == "1"
    key = ("parts", attn, n, split, paired)
    parts = progs.get(key)
    if parts is None:
        steps, given = _balance_env()
        if given is not None:
            assert len(given) == n, "NP_GEMMA_PART_WEIGHTS gives %d shares for %d parts" % (
                len(given), n)
            parts = progs[key] = compile_parts(model, attn, n, split, paired, given)
        else:
            parts = progs[key] = compile_parts(model, attn, n, split, paired)
            if steps > 0:
                # 2 steps to warm up, then the measure.
                parts.calib = [-2, steps,
                               np.zeros((n, max(len(p.recs) for p in parts.progs)))]
        if split:
            assert parts.nodes == cache.nodes, "the parts and the cache are on other nodes"
    parts.bind(model, cache, pos)
    x = model.embed(tokens)
    for p in parts.progs:
        p.names["x"][:] = x
    if parts.calib is not None and parts.calib[0] >= 0:
        if parts.calib[0] == 0:
            cops.gp_xbar_stats()         # reset: the waits of the measure only
        parts.run_prof(parts.calib[2])
    else:
        parts.run()
    if parts.calib is not None:
        parts.calib[0] += 1
        if parts.calib[0] == parts.calib[1]:
            steps, ms = parts.calib[1], parts.calib[2]
            old = parts.weights or [1.0] * n
            kinds = [p.rec_kind for p in parts.progs]
            # The barriers inside MOE_PART: all the waits of the barriers, less
            # the time of the XBAR records.
            xs = cops.gp_xbar_stats()[:n]
            xrec = np.array([sum(ms[p, pc] for pc, k in enumerate(kinds[p]) if k == "xbar")
                             for p in range(n)])
            waits = np.maximum(1e3 * (xs[:, 0] + xs[:, 1]) - xrec, 0.0)
            shares, F, R = balance_shares(ms, kinds, old, waits=waits)
            parts.calib = None
            # The measure, for a report (ms a step of each part).
            model._parts_balance = {"shares": shares.tolist(), "fixed_ms": (F / steps).tolist(),
                                    "rows_ms": (R / steps).tolist()}
            if np.max(np.abs(shares - np.asarray(old) / np.sum(old))) > 0.005:
                # The output of this step comes from the old program; the next
                # step runs the new one.
                xn = parts.progs[0].names["xn"].copy()
                logits = parts.logits
                _rebalance(model, key, parts, attn, n, split, paired, shares)
                model._parts_xn = xn if logits is not None else None
                model._parts_logits = logits
                return xn
    xn = parts.progs[0].names["xn"].copy()
    # Model.logits gives the logits of the parts for this xn.
    model._parts_xn = xn if parts.logits is not None else None
    model._parts_logits = parts.logits
    return xn


# ---- the cache of the parts ------------------------------------------------------

class PartKVCache(KVCache):
    """A KVCache with a buffer for each part of a step (np_gemma/parts.py).

    The buffers of part p hold the rows of its KV heads (ranges of the heads,
    as the place heads), in the memory of the node of its team. A decode
    step in parts reads and writes only the buffers of each part. A prompt
    block runs its attention in the parts (prefill_attention, PART_PREFILL).

    The other readers and writers of KVCache work too: they gather the heads
    of the parts into new arrays (read, read_qc), or scatter them (write,
    write_q). That is a copy, so they are slow. Only the int16 form.
    """

    split = True

    def __init__(self, cfg, max_len=4096, kv=None, parts=None):
        super().__init__(cfg, max_len, kv)
        if self.kv != "int16":
            raise ValueError("PartKVCache takes only the int16 form")
        n = parts or max(2, int(os.environ.get("NP_GEMMA_PARTS", "2")))
        self.parts = n
        self.team = int(os.environ.get("NP_GEMMA_PART_TEAM", "0"))
        self.nodes = [None] * n
        if numa.enabled():
            self.nodes = numa.part_nodes(n, self.team)
        layers = cfg.num_hidden_layers
        # The heads of each part in each layer (head_split): (g0, g1) of the
        # key and value heads, and (h0, h1) of the query heads. In a layer
        # with fewer key and value heads than parts (a global layer of the
        # 12B has one), the parts split the query heads, and a shared key
        # and value head is in the buffers of each part that reads it.
        split = [[head_split(cfg.plan[i], n, p, True) for p in range(n)]
                 for i in range(layers)]
        self.heads = [[(g0, g1) for g0, g1, _h0, _h1 in s] for s in split]
        self.qheads = [[(h0, h1) for _g0, _g1, h0, h1 in s] for s in split]
        self.pbuf = [None] * layers   # for each layer: (kq, ks, vq, vs) of each part
        self._scratch = [{} for _ in range(n)]
        self._bar = np.zeros(8 * (n + 1), dtype=np.int64)
        self._pending = {}

    def _alloc(self, shape, dtype, p):
        node = self.nodes[p]
        if node is None:
            return np.empty(shape, dtype=dtype)
        return numa.empty_on(shape, dtype, node)

    def _cap(self, layer):
        b = self.pbuf[layer]
        return 0 if b is None else b[0][0].shape[0]

    def _grow(self, layer, cap):
        old = self._cap(layer)
        hd = self.cfg.plan[layer].head_dim
        new = []
        for p, (g0, g1) in enumerate(self.heads[layer]):
            nk = g1 - g0
            bufs = (self._alloc((cap, nk, hd), np.int16, p),
                    self._alloc((cap, nk, hd // 32), np.float32, p),
                    self._alloc((cap, nk, hd), np.int16, p),
                    self._alloc((cap, nk, hd // 32), np.float32, p))
            if old:
                for dst, src in zip(bufs, self.pbuf[layer][p]):
                    dst[:old] = src[:old]
            new.append(bufs)
        self.pbuf[layer] = new

    def _buffers(self, layer):
        return [a for b in self.pbuf[layer] for a in b]

    def part(self, layer, p):
        """The buffers (kq, ks, vq, vs) of part p in a layer."""
        return self.pbuf[layer][p]

    def _scatter(self, layer, start, arrays):
        """Store rows (kq, ks, vq, vs) of all heads in the buffers of the parts."""
        t = arrays[0].shape[0]
        for p, (g0, g1) in enumerate(self.heads[layer]):
            for dst, src in zip(self.pbuf[layer][p], arrays):
                dst[start:start + t] = src[:, g0:g1]

    def _joined(self, layer, j, lo, hi):
        """Rows lo to hi of buffer j (kq, ks, vq, vs) with all heads: a copy.
        A head that is in the buffers of two parts comes from the first."""
        cols, done = [], 0
        for p, (g0, g1) in enumerate(self.heads[layer]):
            if g1 > done:
                cols.append(self.pbuf[layer][p][j][lo:hi, max(g0, done) - g0:g1 - g0])
                done = g1
        return np.concatenate(cols, axis=1)

    def _store_qc(self, layer, start, k, v):
        t = k.shape[0]
        plan = self.cfg.plan[layer]
        hd, nkv = plan.head_dim, plan.num_kv_heads
        kq, ks = ops.quantize_i16(k.reshape(t, nkv, hd // 32, 32))
        vq, vs = ops.quantize_i16(v.reshape(t, nkv, hd // 32, 32))
        self._scatter(layer, start, (kq.reshape(t, nkv, hd), ks, vq.reshape(t, nkv, hd), vs))

    def read(self, layer, end, lo=0):
        base = self.base[layer]
        n = end - base - lo
        out = []
        for jq, js in ((0, 1), (2, 3)):
            q, sc = self._joined(layer, jq, lo, lo + n), self._joined(layer, js, lo, lo + n)
            x = np.empty(q.shape, np.float32)
            ops._cops.dequantize_i16_groups(q, sc, x)
            out.append(x)
        return out[0], out[1], base + lo

    def read_qc(self, layer, end):
        n = end - self.base[layer]
        return tuple(self._joined(layer, j, 0, n) for j in range(4)) + (self.base[layer],)

    def rows_q(self, layer, start_pos, t):
        start = self.prepare(layer, start_pos, t)
        plan = self.cfg.plan[layer]
        hd, nkv = plan.head_dim, plan.num_kv_heads
        rows = (np.empty((t, nkv, hd), np.int16), np.empty((t, nkv, hd // 32), np.float32),
                np.empty((t, nkv, hd), np.int16), np.empty((t, nkv, hd // 32), np.float32))
        self._pending[layer] = (start, rows)
        return rows

    def rows_q_done(self, layer, start_pos, t):
        start, rows = self._pending.pop(layer)
        self._scatter(layer, start, rows)
        self.end[layer] = start_pos + t

    def _buf(self, p, name, size):
        """Float32 scratch of part p, on its node."""
        b = self._scratch[p].get(name)
        if b is None or b.size < size:
            b = self._scratch[p][name] = self._alloc(max(size, 2 * (b.size if b is not None
                                                                    else 0)), np.float32, p)
        return b[:size]

    def prefill_attention(self, layer, start_pos, q, k, v, positions, window):
        """Write a prompt block to the cache and run its attention, each part
        for its heads in its team (PART_PREFILL). q is (t, q heads, head_dim),
        k and v (t, kv heads, head_dim), after the norms and the rope. Return
        the (t, q heads, head_dim) result of ops.flash_prefill, with its bits."""
        t = q.shape[0]
        plan = self.cfg.plan[layer]
        qh, kvh, hd = plan.num_q_heads, plan.num_kv_heads, plan.head_dim
        start = self.prepare(layer, start_pos, t)
        self.end[layer] = start_pos + t
        rows = self.end[layer] - self.base[layer]
        q = np.ascontiguousarray(q, dtype=np.float32)
        k = np.ascontiguousarray(k, dtype=np.float32)
        v = np.ascontiguousarray(v, dtype=np.float32)
        pos = np.ascontiguousarray(positions, dtype=np.int32)
        out = np.empty((t, qh, hd), dtype=np.float32)
        progs = []
        for p, (g0, g1) in enumerate(self.heads[layer]):
            nk = g1 - g0
            h0, h1 = self.qheads[layer][p]
            nq = h1 - h0
            ip = np.array([t, start, rows, self.base[layer], window, qh, kvh, hd, g0, g1, h0, h1],
                          dtype=np.int32)
            pr = P.Program()
            if nk == 0 or nq == 0:
                progs.append(pr.finish())
                continue
            pr.emit(P.PART_PREFILL, q, k, v, out, *self.pbuf[layer][p],
                    self._buf(p, "kf", rows * nk * hd), self._buf(p, "vf", rows * nk * hd),
                    self._buf(p, "qp", t * nq * hd), self._buf(p, "op", t * nq * hd), pos, ip)
            pr.keep.append(ip)
            progs.append(pr.finish())
        addrs = np.array([pr.buf.ctypes.data for pr in progs], dtype=np.int64)
        rc = cops.gp_run_parts(addrs, self.team, self._bar)
        if rc != 0:
            raise RuntimeError("gemma_run_parts returned %d" % rc)
        return out
