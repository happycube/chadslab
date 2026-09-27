"""Build a program for a decode step and run it in C.

PERF_PLAN.md, phase 2. A Python loop of a decode step makes about 420 calls
to C, and each call opens an OpenMP region. Here Python describes the step as
data, and one call to C runs it in one region.

The description is a nested expression in the style of Lisp:

    ("let", "h", ("rms_norm", "x", ("w", 0, "input_layernorm")))

The compiler turns it into a flat list of records. A record is one operation
with up to NARG operands. An operand is an integer (an address is an
integer), a float32, or a slot of the environment. The environment holds the
parameters of the program, for example the position, and its scalar
variables. The program and its environment are one int64 array:

    int64  magic, env count, record count, 0
    int64  env[env count]
    record code[record count]

A call binds the parameters, which writes their slots, and then runs C. Thus
the state of a step is part of the program, and dump() shows its values. A
cache buffer that moves when it grows needs only a new bind, not a new
program.

The scalar operations (+, -, *, max, min) compute the values that change for
each step, such as the first key row of the sliding window. Each thread
computes them into its own copy of the environment, so they need no barrier.

run_py() runs the same records in Python. It calls the C entry points of
today, one at a time, so a check can compare the two after each record.

The entry points:

    compile_step(model, attn)   the program of a whole decode step
    ready(model, cache)         the attention mode for a cache, or None
    decode_step(model, ...)     bind the step, run it, return the hidden state
    compile_layers(model, ...)  the program of some layers, for a check

The attention mode attn is "qc" or "f32". The mode "qc" reads the int16 copy
of the cache (NP_GEMMA_ATTN=1, the default). The mode "f32" reads the float
cache (NP_GEMMA_ATTN=0, the default of the server).
Model.forward uses decode_step for one token when ready() allows it. The
program gives the same bits as the Python loop of Model; see
scripts/check_program.py.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

from . import cops, ops

# The size of a record and the first word of a program. They must agree with
# GP_NARG and GP_MAGIC in np_gemma/csrc/bf16_linear.c.
NARG = 24
MAGIC = 0x4750524F47303031

# The tag of an operand: an integer (also an address), the bits of a float32,
# or the index of a slot of the environment.
T_NONE, T_INT, T_F32, T_SLOT = 0, 1, 2, 3

# The operation codes. They must agree with the enum of gemma_run in the C
# file. 1 to 15 are scalar operations. The comment of each case in gp_step
# gives the order of the operands.
S_MOV, S_ADD, S_SUB, S_MUL, S_MAX, S_MIN = 1, 2, 3, 4, 5, 6
RMS_NORM, ADD, MUL_S, COPY, GELU, MUL = 16, 17, 18, 19, 20, 21
INT4_LINEAR, INT4_MULTI4, RMS_NORM_MULTI4, GELU_MUL_INT4 = 32, 33, 34, 35
INT4_LINEAR_MT, INT4_MULTI4_MT, GELU_MUL_ROWS, BF16_LINEAR = 36, 37, 38, 39
QKV_NORM_ROPE, KV_WRITE, ATTN_QC, ATTN_F32 = 48, 49, 50, 51
ATTN_QC_MT, ATTN_F32_MT, QKV_NORM, ROPE, KV_WRITE_HEADS, ATTN_F32H = 52, 53, 54, 55, 56, 57
ROUTER, MOE, ROUTER_MT, MOE_MT, MOE_N = 64, 65, 66, 67, 68
# The operations of a program in parts (np_gemma/parts.py, SPLIT_PLAN.md).
XBAR, MOE_PART, ATTN_QC_H, ATTN_F32_H = 80, 81, 82, 83
# The records that move work between the GPU and the CPU (np_gemma/gpu.py).
TO_HOST, CPU_JOIN, TO_DEV, HOT_SPLIT, HOT_MOE = 84, 85, 86, 87, 88
MOE_GPU, FETCH, FETCH_WAIT, FETCH_DONE, HOT_SPLIT_MT = 89, 90, 91, 92, 93
F32_LINEAR, DRAFT_HEAD, ARGMAX, ADD_NORM, COUNT = 94, 95, 96, 97, 98
# The MLX affine format, the Gated DeltaNet, and Qwen3.5 (QWEN_PLAN.md).
MA_QUANT, MA_LINEAR, MA_MOE, ROUTER_TOPK, GDN, ATTN_PREP, SIGMUL = 100, 101, 102, 103, 104, 105, 106
KQ_QUANT, KQ_LINEAR, KQ_MOE = 107, 108, 109
# The GPU only (np_gemma/qwen_gpu.py).
KQ_HOT_MOE, KQ_MULTI, ADD_RMS, KQ_GROUP_MOE = 110, 111, 112, 113

OP_NAMES = {v: k for k, v in dict(
    S_MOV=S_MOV, S_ADD=S_ADD, S_SUB=S_SUB, S_MUL=S_MUL, S_MAX=S_MAX, S_MIN=S_MIN,
    RMS_NORM=RMS_NORM, ADD=ADD, MUL_S=MUL_S, COPY=COPY, INT4_LINEAR=INT4_LINEAR,
    INT4_MULTI4=INT4_MULTI4, RMS_NORM_MULTI4=RMS_NORM_MULTI4,
    GELU_MUL_INT4=GELU_MUL_INT4, QKV_NORM_ROPE=QKV_NORM_ROPE, KV_WRITE=KV_WRITE,
    ATTN_QC=ATTN_QC, ATTN_F32=ATTN_F32, ROUTER=ROUTER, MOE=MOE,
    INT4_LINEAR_MT=INT4_LINEAR_MT, INT4_MULTI4_MT=INT4_MULTI4_MT,
    GELU_MUL_ROWS=GELU_MUL_ROWS, ATTN_QC_MT=ATTN_QC_MT, ATTN_F32_MT=ATTN_F32_MT,
    ROUTER_MT=ROUTER_MT, MOE_MT=MOE_MT, GELU=GELU, MUL=MUL, BF16_LINEAR=BF16_LINEAR,
    QKV_NORM=QKV_NORM, ROPE=ROPE, KV_WRITE_HEADS=KV_WRITE_HEADS,
    ATTN_F32H=ATTN_F32H, XBAR=XBAR, MOE_PART=MOE_PART,
    ATTN_QC_H=ATTN_QC_H, ATTN_F32_H=ATTN_F32_H, TO_HOST=TO_HOST, CPU_JOIN=CPU_JOIN,
    TO_DEV=TO_DEV, MOE_N=MOE_N, HOT_SPLIT=HOT_SPLIT, HOT_MOE=HOT_MOE, MOE_GPU=MOE_GPU,
    FETCH=FETCH, FETCH_WAIT=FETCH_WAIT, FETCH_DONE=FETCH_DONE,
    HOT_SPLIT_MT=HOT_SPLIT_MT, F32_LINEAR=F32_LINEAR, DRAFT_HEAD=DRAFT_HEAD,
    ARGMAX=ARGMAX, ADD_NORM=ADD_NORM, COUNT=COUNT, MA_QUANT=MA_QUANT, MA_LINEAR=MA_LINEAR,
    MA_MOE=MA_MOE, ROUTER_TOPK=ROUTER_TOPK, GDN=GDN, ATTN_PREP=ATTN_PREP, SIGMUL=SIGMUL,
    KQ_QUANT=KQ_QUANT, KQ_LINEAR=KQ_LINEAR, KQ_MOE=KQ_MOE, KQ_HOT_MOE=KQ_HOT_MOE,
    KQ_MULTI=KQ_MULTI, ADD_RMS=ADD_RMS, KQ_GROUP_MOE=KQ_GROUP_MOE).items()}

# One record: the operation, the flags (not used yet), the tag of each
# operand, and the value of each operand. The C struct gp_rec has the same
# layout; Program.finish checks the size.
REC = np.dtype([("op", "<i4"), ("flags", "<i4"), ("tag", "u1", (NARG,)),
                ("v", "<i8", (NARG,))])


class Slot:
    """One slot of the environment. It holds an int64."""

    def __init__(self, index, name):
        self.index = index
        self.name = name

    def __repr__(self):
        return "$" + self.name


def _f32_bits(x):
    """Return the bits of a float32 as an int, for a T_F32 operand."""
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def _bits_f32(b):
    """Return the float32 that the low 32 bits of b hold."""
    return float(np.array([b & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0])


class Program:
    """A list of records and an environment. See the module text."""

    def __init__(self):
        self.slots = []            # Slot objects, in order
        self.by_name = {}          # the slot of each name
        self.init = []             # the first value of each slot
        self.recs = []             # (op, [(tag, value), ...])
        # A literal operand holds the address of an array. The program keeps
        # the array, so the address stays good for the life of the program.
        self.keep = []
        self.bound = {}            # the arrays that a bind points to
        self.buf = None            # the int64 array that C runs; see finish
        self.names = {}            # the buffers of the compiler, by name

    # ---- building ----------------------------------------------------------
    def slot(self, name, value=0):
        """Return the slot of a name. Make it when it does not exist."""
        s = self.by_name.get(name)
        if s is None:
            s = Slot(len(self.slots), name)
            self.slots.append(s)
            self.by_name[name] = s
            self.init.append(int(value))
        return s

    def temp(self):
        """Return a new slot for the result of a scalar operation."""
        return self.slot("t%d" % len(self.slots))

    def _enc(self, v):
        """Return the (tag, value) of an operand.

        A Slot gives its index. An array gives its address; the program keeps
        the array. None gives a null address. An int is a literal, and a float
        becomes the bits of a float32.
        """
        if isinstance(v, Slot):
            return (T_SLOT, v.index)
        if v is None:
            return (T_INT, 0)
        if isinstance(v, np.ndarray):
            assert v.flags.c_contiguous, "an operand array must be contiguous"
            self.keep.append(v)
            return (T_INT, v.ctypes.data)
        if isinstance(v, (bool, int, np.integer)):
            return (T_INT, int(v))
        if isinstance(v, (float, np.floating)):
            return (T_F32, _f32_bits(v))
        raise TypeError("operand %r" % (v,))

    def emit(self, op, *args):
        """Add one record. The operands follow the order of gp_step in C."""
        assert len(args) <= NARG
        self.recs.append((op, [self._enc(a) for a in args]))

    def finish(self):
        """Make the int64 array of the program.

        The array holds the header, the environment, and the records. The
        environment holds the first value of each slot. env and code are views
        into the array, so bind writes the array that C reads.
        """
        n_env = len(self.slots)
        code = np.zeros(len(self.recs), dtype=REC)
        for i, (op, args) in enumerate(self.recs):
            code[i]["op"] = op
            for k, (tag, val) in enumerate(args):
                code[i]["tag"][k] = tag
                code[i]["v"][k] = val
        assert REC.itemsize == cops.gp_record_size()
        buf = np.zeros(4 + n_env + code.nbytes // 8, dtype=np.int64)
        buf[0] = np.int64(MAGIC)
        buf[1] = n_env
        buf[2] = len(self.recs)
        buf[4:4 + n_env] = self.init
        buf[4 + n_env:] = code.view(np.int64)
        self.buf = buf
        self.env = buf[4:4 + n_env]
        self.code = buf[4 + n_env:].view(REC)
        return self

    # ---- running -----------------------------------------------------------
    def bind(self, **kw):
        """Write parameters. A value is an int, a float, or an array.

        An array gives its address. The program holds the array until the next
        bind of the same name, so the address stays good for the run.
        """
        for name, v in kw.items():
            s = self.by_name[name]
            if isinstance(v, np.ndarray):
                self.bound[name] = v
                self.env[s.index] = v.ctypes.data
            elif isinstance(v, (float, np.floating)):
                self.env[s.index] = _f32_bits(v)
            else:
                self.env[s.index] = int(v)

    def run(self, limit=-1):
        """Run the records in C. limit runs only the first records."""
        rc = cops.gp_run(self.buf, limit)
        if rc != 0:
            raise RuntimeError("gemma_run returned %d" % rc)

    def profile(self):
        """Run the records in C with a barrier after each one. Return the
        time of each record in ms (CPU_PLAN.md, phase 0)."""
        return cops.gp_profile(self.buf, len(self.recs))

    def profile_ops(self, reps=3):
        """Run profile() reps times. Return {op name: (ms, count)} for one
        run, the largest first."""
        acc = {}
        for _ in range(reps):
            for (op, _a), ms in zip(self.recs, self.profile()):
                name = OP_NAMES.get(op, str(op))
                t, n = acc.get(name, (0.0, 0))
                acc[name] = (t + ms / reps, n + 1.0 / reps)
        return dict(sorted(acc.items(), key=lambda kv: -kv[1][0]))

    def dump(self, limit=None):
        """Return the program as text: the slots, then the records."""
        lines = ["env:"]
        for s in self.slots:
            lines.append("  $%-16s %d" % (s.name, int(self.env[s.index])))
        lines.append("code:")
        names = {s.index: s.name for s in self.slots}
        for i, (op, args) in enumerate(self.recs[:limit]):
            ops_ = []
            for tag, val in args:
                if tag == T_SLOT:
                    ops_.append("$" + names[val])
                elif tag == T_F32:
                    ops_.append("%gf" % _bits_f32(val))
                else:
                    ops_.append("0x%x" % val if val > 1 << 20 else str(val))
            lines.append("  %3d %-16s %s" % (i, OP_NAMES[op], " ".join(ops_)))
        return "\n".join(lines)

    def run_py(self, limit=-1):
        """Run the records in Python with the C entry points of today."""
        e = [int(x) for x in self.env]
        n = len(self.recs) if limit < 0 else min(limit, len(self.recs))
        for op, args in self.recs[:n]:
            _py_step(op, args, e)


# ---- the Python interpreter --------------------------------------------------

def _val(args, e, k):
    """Return operand k: its slot value in e, or its literal."""
    tag, v = args[k]
    return e[v] if tag == T_SLOT else v


def _f(args, e, k):
    """Return operand k as a float32."""
    return _bits_f32(_val(args, e, k))


def _arr(addr, n, ctype=ctypes.c_float):
    """Return a NumPy view of n values at an address. It copies nothing."""
    return np.ctypeslib.as_array((ctype * n).from_address(addr))


def _py_step(op, a, e):
    """Run one record in Python.

    A kernel record calls the C entry point of the Python path. That entry
    point opens its own region. A record that the Python path of Model does in
    NumPy uses NumPy here. The result is the reference for a check of each
    record.
    """
    L = cops._lib
    V = lambda k: _val(a, e, k)  # noqa: E731
    F = lambda k: ctypes.c_float(_f(a, e, k))  # noqa: E731
    if op == S_MOV:
        e[a[0][1]] = V(1)
    elif op == S_ADD:
        e[a[0][1]] = V(1) + V(2)
    elif op == S_SUB:
        e[a[0][1]] = V(1) - V(2)
    elif op == S_MUL:
        e[a[0][1]] = V(1) * V(2)
    elif op == S_MAX:
        e[a[0][1]] = max(V(1), V(2))
    elif op == S_MIN:
        e[a[0][1]] = min(V(1), V(2))
    elif op == RMS_NORM:
        L.gemma_rms_norm(V(0), V(1) or None, V(2), V(3), V(4), F(5))
    elif op == COUNT:
        n = V(2) * V(3)
        np.add.at(_arr(V(1), V(4), ctypes.c_int32), _arr(V(0), n, ctypes.c_int32), 1)
    elif op == ADD_NORM:
        rows, cols = V(4), V(5)
        o = _arr(V(0), rows * cols).reshape(rows, cols)
        x = _arr(V(2), rows * cols).reshape(rows, cols)
        s = 1.0 / np.sqrt(np.mean(o * o, axis=1, keepdims=True) + np.float32(F(6)))
        y = (x + o * s.astype(np.float32) * _arr(V(1), cols)) * np.float32(F(7))
        _arr(V(3), rows * cols)[:] = y.reshape(-1)
        if V(9):
            s2 = 1.0 / np.sqrt(np.mean(y * y, axis=1, keepdims=True) + np.float32(F(6)))
            _arr(V(9), rows * cols)[:] = (y * s2.astype(np.float32) * _arr(V(8), cols)).reshape(-1)
    elif op == ADD:
        n = V(3)
        np.add(_arr(V(0), n), _arr(V(1), n), out=_arr(V(2), n))
    elif op == MUL_S:
        n = V(3)
        np.multiply(_arr(V(0), n), np.float32(_f(a, e, 1)), out=_arr(V(2), n))
    elif op == COPY:
        ctypes.memmove(V(1), V(0), V(2))
    elif op == INT4_LINEAR:
        L.gemma_int4_linear(V(1), V(2), V(0), V(3), V(4), V(5), 1, 32)
    elif op == INT4_MULTI4:
        args = []
        for m in range(4):
            b = 2 + 4 * m
            args += [V(b) or None, V(b + 1) or None, V(b + 2) or None, V(b + 3)]
        L.gemma_int4_multi4(*args, V(0), V(1))
    elif op == RMS_NORM_MULTI4:
        args = []
        for m in range(4):
            b = 5 + 4 * m
            args += [V(b) or None, V(b + 1) or None, V(b + 2) or None, V(b + 3)]
        L.gemma_rms_norm_multi4(V(0), V(1), V(2), V(3), F(4), *args)
    elif op == GELU_MUL_INT4:
        L.gemma_gelu_mul_int4(V(0), V(1), V(2), V(3), V(4), V(5), V(6), V(7), V(8))
    elif op == QKV_NORM_ROPE:
        L.gemma_qkv_norm_rope(V(0), V(1), V(2), V(3), V(4), V(5), V(6), V(7),
                              V(8), V(9), V(10), V(11), V(12), F(13))
    elif op == KV_WRITE:
        n = V(8)
        k = _arr(V(0), n)
        v = _arr(V(1), n)
        _arr(V(2), n)[:] = k
        _arr(V(3), n)[:] = v
        if not V(4):
            return
        kq, ks = ops.quantize_i16(k.reshape(-1, 32))
        vq, vs = ops.quantize_i16(v.reshape(-1, 32))
        _arr(V(4), n, ctypes.c_int16)[:] = kq.reshape(-1)
        _arr(V(5), n // 32)[:] = ks.reshape(-1)
        _arr(V(6), n, ctypes.c_int16)[:] = vq.reshape(-1)
        _arr(V(7), n // 32)[:] = vs.reshape(-1)
    elif op == ATTN_QC:
        qh, kvh, hd, n = V(7), V(8), V(9), V(10)
        q = _arr(V(0), qh * hd)
        o = ops.attn_decode(q.reshape(qh, hd), _arr(V(1), n * kvh * hd, ctypes.c_int16),
                            _arr(V(2), n * kvh * hd // 32), _arr(V(3), n * kvh * hd, ctypes.c_int16),
                            _arr(V(4), n * kvh * hd // 32), qh, kvh, hd, n)
        _arr(V(6), qh * hd)[:] = o.reshape(-1)
    elif op == ATTN_F32:
        qh, kvh, hd, n = V(5), V(6), V(7), V(8)
        o = cops.attn_decode_f32s(_arr(V(0), qh * hd).reshape(qh, hd),
                                  _arr(V(1), n * kvh * hd).reshape(n, kvh, hd),
                                  _arr(V(2), n * kvh * hd).reshape(n, kvh, hd),
                                  V(9), V(10), V(11))
        _arr(V(4), qh * hd)[:] = o.reshape(-1)
    elif op == ROUTER:
        L.gemma_router(V(0), V(1), V(2), V(3), V(4), V(5), V(6), F(7), F(8), V(9), V(10))
    elif op == MOE:
        top_k, gu_rows, cols, dn_rows, inner = V(3), V(8), V(9), V(10), V(11)
        h = _arr(V(0), cols)
        val = _arr(V(1), top_k)
        idx = _arr(V(2), top_k, ctypes.c_int32)
        ids = np.unique(idx).astype(np.int32)
        act = np.empty((ids.size, 2 * inner), dtype=np.float32)
        act2 = np.empty((ids.size, inner), dtype=np.float32)
        de = np.empty((ids.size, dn_rows), dtype=np.float32)
        L.gemma_moe_gemv_gelu(V(4), V(5), h.ctypes.data, ids.ctypes.data, ids.size,
                              act.ctypes.data, act2.ctypes.data, gu_rows, cols, 0, inner)
        L.gemma_int4_moe_gemv(V(6), V(7), act2.ctypes.data, ids.ctypes.data, ids.size,
                              de.ctypes.data, dn_rows, inner, inner)
        out = _arr(V(16), dn_rows)
        out[:] = 0.0
        for j in range(ids.size):
            slot = np.nonzero(idx == ids[j])[0]
            out += de[j] * val[slot[0]]
    elif op == INT4_LINEAR_MT:
        L.gemma_int4_linear_mt(V(1), V(2), V(0), V(3), V(4), V(5), V(6))
    elif op == INT4_MULTI4_MT:
        args = []
        for m in range(4):
            b = 3 + 4 * m
            args += [V(b) or None, V(b + 1) or None, V(b + 2) or None, V(b + 3)]
        L.gemma_int4_multi4_mt(*args, V(0), V(1), V(2))
    elif op == GELU_MUL_ROWS:
        rows, inner = V(3), V(4)
        g = _arr(V(0), rows * inner).reshape(rows, inner)
        u = _arr(V(1), rows * inner).reshape(rows, inner)
        _arr(V(2), rows * inner)[:] = ops.gelu_mul_rows(g, u).reshape(-1)
    elif op in (ATTN_QC_MT, ATTN_F32_MT):
        qc = op == ATTN_QC_MT
        o = 7 if qc else 5
        qh, kvh, hd, t = V(o), V(o + 1), V(o + 2), V(o + 3)
        pos, base, window = V(o + 4), V(o + 5), V(o + 6)
        q = _arr(V(0), t * qh * hd).reshape(t, qh, hd)
        p_ = np.arange(t) + pos
        lo = np.maximum(0, p_ - window + 1 - base) if window else np.zeros(t, np.int64)
        n = p_ + 1 - base - lo
        rows = int((n + lo).max())
        per = kvh * hd
        out = _arr(V(6 if qc else 4), t * qh * hd).reshape(t, qh, hd)
        if qc:
            kq = _arr(V(1), rows * per, ctypes.c_int16).reshape(rows, kvh, hd)
            ks = _arr(V(2), rows * per // 32).reshape(rows, kvh, hd // 32)
            vq = _arr(V(3), rows * per, ctypes.c_int16).reshape(rows, kvh, hd)
            vs = _arr(V(4), rows * per // 32).reshape(rows, kvh, hd // 32)
            out[:] = ops.attn_decode_mt(q, kq, ks, vq, vs, qh, kvh, hd, lo, n)
        else:
            K = _arr(V(1), rows * per).reshape(rows, kvh, hd)
            Vv = _arr(V(2), rows * per).reshape(rows, kvh, hd)
            for j in range(t):
                out[j] = cops.attn_decode_f32s(q[j], K[lo[j]:lo[j] + n[j]],
                                               Vv[lo[j]:lo[j] + n[j]], pos + j,
                                               base + lo[j], window)
    elif op == ROUTER_MT:
        L.gemma_router_mt(V(0), V(1), V(2), V(3), V(4), V(5), V(6), F(7), F(8), V(9), V(10),
                          V(11))
    elif op == MOE_MT:
        t, top_k, gu_rows, cols, dn_rows, inner = V(3), V(4), V(9), V(10), V(11), V(12)
        h = _arr(V(0), t * cols).reshape(t, cols)
        val = _arr(V(1), t * top_k).reshape(t, top_k)
        idx = _arr(V(2), t * top_k, ctypes.c_int32).reshape(t, top_k)
        # The steps of Model._moe_mt.
        flat = idx.reshape(-1)
        order = np.argsort(flat, kind="stable")
        tok = order // top_k
        slot = order % top_k
        ids, counts = np.unique(flat, return_counts=True)
        poff = np.concatenate([[0], np.cumsum(counts)])
        act = ops.moe_gemv_mt(_ArrW(V(5)), _ArrW(V(6)), h, ids, poff, tok, gu_rows, cols,
                              cols, inner)
        de = ops.moe_gemv_mt(_ArrW(V(7)), _ArrW(V(8)), act, ids, poff, np.arange(tok.size),
                             dn_rows, inner, inner)
        contrib = de * val[tok, slot][:, None]
        m = np.argsort(tok, kind="stable").reshape(t, top_k)
        out = _arr(V(22), t * dn_rows).reshape(t, dn_rows)
        out[:] = 0.0
        for k in range(top_k):
            out += contrib[m[:, k]]
    elif op == GELU:
        n = V(2)
        _arr(V(1), n)[:] = cops.gelu(_arr(V(0), n))
    elif op == MUL:
        rows, cols, bs = V(3), V(4), V(5)
        a_ = _arr(V(0), rows * cols).reshape(rows, cols)
        b_ = np.lib.stride_tricks.as_strided(
            _arr(V(1), (rows - 1) * bs + cols), (rows, cols), (4 * bs, 4))
        _arr(V(2), rows * cols).reshape(rows, cols)[:] = a_ * b_
    elif op == BF16_LINEAR:
        L.gemma_bf16_linear(V(1), V(0), V(2), V(3), V(4), V(5))
    elif op == QKV_NORM:
        L.gemma_qkv_norm(V(0), V(1) or None, V(2), V(3) or None, V(4) or None, V(5),
                         V(6) or None, V(7), V(8), F(9))
    elif op == ROPE:
        L.gemma_rope(V(0), V(1), V(2), V(3) or None, V(4), V(5), V(6), V(7), V(8))
    elif op == KV_WRITE_HEADS:
        hs, pos, t, kvh, hd = V(4), V(5), V(6), V(7), V(8)
        k = _arr(V(0), t * kvh * hd).reshape(t, kvh, hd)
        v = _arr(V(1), t * kvh * hd).reshape(t, kvh, hd)
        for h in range(kvh):
            base_ = h * hs + pos * hd
            _arr(V(2) + 4 * base_, t * hd).reshape(t, hd)[:] = k[:, h]
            _arr(V(3) + 4 * base_, t * hd).reshape(t, hd)[:] = v[:, h]
    elif op == ATTN_F32H:
        qh, kvh, hd, t, pos, hs, window, slide = (V(5), V(6), V(7), V(8), V(9), V(10),
                                                  V(11), V(12))
        q = _arr(V(0), t * qh * hd).reshape(t, qh, hd)
        out = _arr(V(4), t * qh * hd).reshape(t, qh, hd)
        span = (kvh - 1) * hs + (pos + t) * hd
        K = np.lib.stride_tricks.as_strided(_arr(V(1), span), (kvh, pos + t, hd), (4 * hs, 4 * hd, 4))
        Vv = np.lib.stride_tricks.as_strided(_arr(V(2), span), (kvh, pos + t, hd), (4 * hs, 4 * hd, 4))
        for j in range(t):
            p_ = pos + j
            lo = max(0, p_ - window + 1) if (slide and window) else 0
            out[j] = ops.attn_decode_f32(q[j:j + 1], K[:, lo:p_ + 1, :], Vv[:, lo:p_ + 1, :],
                                         p_, lo, window)[0]
    else:
        raise ValueError("op %d" % op)


class _ArrW:
    """An address in the place of a weight array, for a cops function that
    reads only .ctypes.data of the weight."""

    def __init__(self, addr):
        self.ctypes = type("C", (), {"data": addr})()


# ---- the compiler --------------------------------------------------------------

class Compiler:
    """Turn the expressions of a model into the records of a Program.

    The special forms:

        (seq f ...)            compile each form in order
        (layer i f ...)        the same; i names the layer for a reader
        (let name e)           compile e and give its value a name
        (let (a b ...) e)      the same for an operation with several results
        (set name e)           compile the kernel expression e into the
                               buffer of name, in place
        (slot name)            the slot of a parameter
        (w layer key)          a weight of a layer; (w None "norm") is the
                               final norm
        (m module)             an E4B matrix: int4 blocks or bfloat16
        (t key)                an E4B tensor, such as a norm weight
        (+ a b ...) (- ...) (* ...) (max ...) (min ...)
                               scalar operations on ints and slots

    Every other head is a kernel operation of KERNELS. A name is a value of a
    let, or else a parameter slot of that name. A kernel operation makes a new
    buffer for its result, except in a set.
    """

    def __init__(self, model, prog=None):
        self.model = model
        self.cfg = model.cfg
        self.eps = model.cfg.rms_norm_eps
        self.p = prog or Program()
        self.env = self.p.names

    def value(self, x):
        """Return the value of an operand of a form.

        A name gives the value of a let, or else a parameter slot of
        PARAMS or PARAM_PREFIXES. Any other name is an error. A tuple is
        a form to compile. Any other value (an int, a float, an array) stays
        as it is.
        """
        if isinstance(x, str):
            if x in self.env:
                return self.env[x]
            if x in PARAMS or x.startswith(PARAM_PREFIXES):
                return self.p.slot(x)
            raise NameError("the form uses %r, which no let gives and which is not "
                            "a parameter" % x)
        if isinstance(x, tuple):
            return self.expr(x)
        return x

    def buffer(self, shape, dtype=np.float32):
        """Return a new buffer for a result. The program keeps it."""
        return np.zeros(shape, dtype=dtype)

    def compile(self, form):
        """Compile a top-level form: a seq, a layer, or one expression."""
        head = form[0]
        if head in ("seq", "layer"):
            body = form[2:] if head == "layer" else form[1:]
            for f in body:
                self.compile(f)
            return None
        return self.expr(form)

    def expr(self, form):
        """Compile one expression and return its value."""
        head, args = form[0], form[1:]
        if head == "let":
            names, e = args
            v = self.value(e)
            if isinstance(names, tuple):
                for n_, v_ in zip(names, v):
                    self.env[n_] = v_
            else:
                self.env[names] = v
            return v
        if head == "set":
            name, e = args
            return self.expr_into(e, self.env[name])
        if head == "slot":
            return self.p.slot(args[0])
        if head == "w":
            layer, key = args
            if layer is None:
                return {"norm": self.model._norm_w}[key]
            return self.model._layers[layer][key]
        if head == "m":
            # An E4B matrix: the int4 blocks, or else the bfloat16 copy.
            entry = self.model.q4(args[0])
            return entry if entry is not None else self.model.W16(args[0])
        if head == "t":
            # An E4B tensor that the quantization did not touch.
            return np.ascontiguousarray(self.model.T(args[0]), dtype=np.float32)
        if head in SCALAR:
            return self.scalar(head, [self.value(a) for a in args])
        return self.kernel(head, [self.value(a) for a in args])

    def expr_into(self, form, out):
        """Compile a kernel expression that writes an existing buffer."""
        return self.kernel(form[0], [self.value(a) for a in form[1:]], out)

    def kernel(self, head, vals, out=None):
        """Compile one kernel operation of KERNELS on the values of its
        operands. The compiler of a part (np_gemma/parts.py) changes this."""
        fn = KERNELS[head]
        return fn(self, *vals) if out is None else fn(self, *vals, out=out)

    def scalar(self, head, vals):
        """Compile a scalar operation. Return an int or a slot.

        Constants fold in Python. An operand that is a slot makes one record
        for each pair of operands, from left to right.
        """
        if all(isinstance(v, (int, np.integer)) for v in vals):
            r = vals[0]
            for v in vals[1:]:
                r = SCALAR[head][1](r, v)
            return int(r)
        r = vals[0]
        for v in vals[1:]:
            t = self.p.temp()
            self.p.emit(SCALAR[head][0], t, r, v)
            r = t
        return r


# The names of the parameters that bind_step writes. Any other name in a form
# must come from a let. Thus a name with an error stops the compiler.
PARAMS = ("pos", "scores")
PARAM_PREFIXES = ("base.", "cos.", "sin.")


SCALAR = {
    "+": (S_ADD, lambda a, b: a + b),
    "-": (S_SUB, lambda a, b: a - b),
    "*": (S_MUL, lambda a, b: a * b),
    "max": (S_MAX, max),
    "min": (S_MIN, min),
}


# ---- the kernel operations ------------------------------------------------------
# Each function takes the compiler and the values of its operands. It emits the
# records and returns the result. With out, it writes that buffer (a set).
# Each operation calls the same kernel as the Python path of Model, so the
# program keeps the bits of that path.

def k_rms_norm(c, x, w, out=None):
    """(rms_norm x w): the norm of each row of x. As ops.rms_norm."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(RMS_NORM, x, w, out, x.shape[0], x.shape[1], float(c.eps))
    return out


def k_add(c, a, b, out=None):
    """(add a b): a + b. As the NumPy add of the Python path."""
    out = c.buffer(a.shape) if out is None else out
    c.p.emit(ADD, a, b, out, a.size)
    return out


def k_add_norm(c, x, o, w, s=1.0, out=None):
    """(add_norm x o w [s]): (x + rms_norm(o) w) s in one pass. The GPU group
    of the E4B uses it in place of rms_norm, add, and mul; the values are
    the same."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(ADD_NORM, o, w, x, out, x.shape[0], x.shape[1], float(c.eps),
             float(np.asarray(s).reshape(-1)[0]), 0, 0)
    return out


def k_add_norm2(c, x, o, w, w2, s=1.0):
    """(add_norm2 x o w w2 [s]): x = (x + rms_norm(o) w) s in place, as
    add_norm. Return rms_norm(x) w2, the input of the next matrix, from the
    same kernel. The values are those of add_norm and rms_norm."""
    out2 = c.buffer(x.shape)
    c.p.emit(ADD_NORM, o, w, x, x, x.shape[0], x.shape[1], float(c.eps),
             float(np.asarray(s).reshape(-1)[0]), w2, out2)
    return out2


def k_gelu_mul(c, g, u, out=None):
    """(gelu_mul g u): gelu(g) * u in one pass, as gelu then mul_v."""
    out = c.buffer(g.shape) if out is None else out
    t = g.shape[0] if g.ndim > 1 else 1
    c.p.emit(GELU_MUL_ROWS, g, u, out, t, g.size // t)
    return out


def k_mul(c, x, s, out=None):
    """(mul x s): x times the float32 s. As x * layer_scalar in NumPy. s can
    be an array of one value."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(MUL_S, x, float(np.asarray(s).reshape(-1)[0]), out, x.size)
    return out


def k_copy(c, x, out=None):
    """(copy x): a copy of x. The global layers use the key as the value."""
    out = c.buffer(x.shape, x.dtype) if out is None else out
    c.p.emit(COPY, x, out, x.nbytes)
    return out


def _mats(c, mats, t=1):
    """Return the operands of up to four int4 matrices and their outputs. The
    outputs are new buffers of the compiler c."""
    args, outs = [], []
    for m in range(4):
        if m < len(mats) and mats[m] is not None:
            w, s = mats[m]
            o = c.buffer((t, w.shape[0]))
            args += [w, s, o, w.shape[0]]
            outs.append(o)
        else:
            args += [None, None, None, 0]
    return args, outs


def k_int4_multi4(c, x, *mats):
    """(int4_multi4 x m ...): up to four int4 matrices on the row x in one
    kernel. One row uses ops.int4_multi4, and a group of rows uses
    ops.int4_multi4_mt. Return one (rows of x, rows of m) buffer for each
    matrix."""
    t = x.shape[0]
    args, outs = _mats(c, mats, t)
    if t == 1:
        c.p.emit(INT4_MULTI4, x, x.shape[1], *args)
    else:
        c.p.emit(INT4_MULTI4_MT, x, x.shape[1], t, *args)
    return tuple(outs)


def k_rms_norm_multi4(c, x, wn, *mats):
    """(rms_norm_multi4 x wn m ...): the norm of x, then up to four int4
    matrices on the result. As ops.rms_norm_multi4. A group makes the norm
    and the matrices as two operations, as the group path of
    Model._decoder_layer does."""
    if x.shape[0] > 1:
        return k_int4_multi4(c, k_rms_norm(c, x, wn), *mats)
    args, outs = _mats(c, mats)
    scratch = c.buffer(x.shape[1])
    c.p.emit(RMS_NORM_MULTI4, x, np.ascontiguousarray(wn, dtype=np.float32), scratch,
             x.shape[1], float(c.eps), *args)
    return tuple(outs)


def k_gelu_mul_int4(c, g, u, mat, out=None):
    """(gelu_mul_int4 g u m): gelu(g) * u, then the int4 matrix m. As
    ops.gelu_mul_int4. A group uses ops.gelu_mul_rows and the group matrix
    kernel."""
    if g.shape[0] > 1:
        t, inner = g.shape
        h = c.buffer((t, inner))
        c.p.emit(GELU_MUL_ROWS, g, u, h, t, inner)
        return k_int4(c, mat, h, out)
    w, s = mat
    rows, cols = w.shape[0], g.size
    out = c.buffer((1, rows)) if out is None else out
    c.p.emit(GELU_MUL_INT4, g, u, g.size, c.buffer(g.size), w, s, out, rows, cols)
    return out


def k_int4(c, mat, x, out=None):
    """(int4 m x): the int4 matrix m on the rows of x. As Model.linear for
    one token, and as ops.linear_int4_mt for a group."""
    w, s = mat
    t = x.shape[0]
    out = c.buffer((t, w.shape[0])) if out is None else out
    if t == 1:
        c.p.emit(INT4_LINEAR, x, w, s, out, w.shape[0], x.shape[1])
    else:
        c.p.emit(INT4_LINEAR_MT, x, w, s, out, w.shape[0], x.shape[1], t)
    return out


def k_qkv_norm_rope(c, q, k, v, qn, kn, cos, sin, layer):
    """(qkv_norm_rope q k v qn kn cos sin layer): the norms of q, k, and v,
    then the rope of q and k. The
    operation changes q, k, and v in place. cos and sin are slots. bind_step
    fills them with the rope table of the step."""
    plan = c.cfg.plan[layer]
    hd = plan.head_dim
    c.p.emit(QKV_NORM_ROPE, q, np.ascontiguousarray(qn, dtype=np.float32), q.size // hd,
             k, np.ascontiguousarray(kn, dtype=np.float32), k.size // hd, v, v.size // hd,
             cos, sin, plan.num_q_heads, plan.num_kv_heads, hd, float(c.eps))


def _addr(c, base, row, stride):
    """Return base + row * stride, the address of a row, as a scalar value."""
    return c.scalar("+", [base, c.scalar("*", [row, stride])])


def k_kv_write(c, layer, k, v, row, qc=1):
    """(kv_write layer k v row qc): store the key and the value at the
    buffer row of the cache of a layer. With qc, also store the int16 copy, as
    KVCache._store_qc does. The addresses come from the slots of the layer
    and the row, with scalar operations."""
    plan = c.cfg.plan[layer]
    per = plan.num_kv_heads * plan.head_dim
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    if qc:
        # An int16 row has 2 bytes for each value and 4 bytes for each scale.
        q = [_addr(c, s("kq"), row, 2 * per), _addr(c, s("ks"), row, 4 * (per // 32)),
             _addr(c, s("vq"), row, 2 * per), _addr(c, s("vs"), row, 4 * (per // 32))]
    else:
        q = [0, 0, 0, 0]
    # The rows of a group are adjacent in the cache, so one copy stores them.
    c.p.emit(KV_WRITE, k, v,
             _addr(c, s("k"), row, 4 * per), _addr(c, s("v"), row, 4 * per), *q, k.size)


def k_attn_qc(c, layer, q, lo, n):
    """(attn_qc layer q lo n): the fused attention of one float32 query over
    n rows of the int16 cache, from buffer row lo. As Model._attend_one."""
    plan = c.cfg.plan[layer]
    hd, qh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
    per = kvh * hd
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    out = c.buffer((1, qh * hd))
    c.p.emit(ATTN_QC, q,
             _addr(c, s("kq"), lo, 2 * per), _addr(c, s("ks"), lo, 4 * (per // 32)),
             _addr(c, s("vq"), lo, 2 * per), _addr(c, s("vs"), lo, 4 * (per // 32)),
             c.p.slot("scores"), out, qh, kvh, hd, n)
    return out


def k_attn_f32(c, layer, q, lo, n):
    """(attn_f32 layer q lo n): the attention of one query over n rows of
    the float cache, from buffer row lo. As Model._attend_one with the C
    kernel (NP_GEMMA_F32_ATTN=c)."""
    plan = c.cfg.plan[layer]
    hd, qh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
    per = kvh * hd
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    out = c.buffer((1, qh * hd))
    base = c.scalar("+", [s("base"), lo])
    c.p.emit(ATTN_F32, q, _addr(c, s("k"), lo, 4 * per), _addr(c, s("v"), lo, 4 * per),
             c.p.slot("scores"), out, qh, kvh, hd, n, c.p.slot("pos"), base,
             plan.sliding_window or 0)
    return out


def k_attn_rows(c, layer, q, attn):
    """(attn_rows_qc layer q) or (attn_rows_f32 layer q): the attention of a
    group of queries. The first form reads the int16 cache, and the second
    reads the float cache. Each row of q is one query. Query j has the
    position pos + j. The operation finds the key rows of each query from
    pos, the base of the layer, and the window. As the group path of
    Model._attention."""
    plan = c.cfg.plan[layer]
    hd, qh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
    t = q.size // (qh * hd)
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    out = c.buffer((t, qh * hd))
    pos, base, window = c.p.slot("pos"), s("base"), plan.sliding_window or 0
    if attn == "qc":
        c.p.emit(ATTN_QC_MT, q, s("kq"), s("ks"), s("vq"), s("vs"), c.p.slot("scores"), out,
                 qh, kvh, hd, t, pos, base, window, np.zeros(t, np.int32),
                 np.zeros(t, np.int32))
    else:
        c.p.emit(ATTN_F32_MT, q, s("k"), s("v"), c.p.slot("scores"), out, qh, kvh, hd, t,
                 pos, base, window)
    return out


def k_router(c, x, layer):
    """(router x layer): the router of the mixture of experts. Return the
    weights and the indices of the top experts. As ops.router."""
    w = c.model._layers[layer]
    cfg = c.cfg
    top_k = cfg.top_k_experts
    t = x.shape[0]
    proj = np.ascontiguousarray(w["router.proj"], dtype=np.float32)
    if t > 1:
        val = c.buffer((t, top_k))
        idx = c.buffer((t, top_k), np.int32)
        c.p.emit(ROUTER_MT, x, np.ascontiguousarray(w["router.scale"], dtype=np.float32),
                 proj, np.ascontiguousarray(w["router.per_expert_scale"], dtype=np.float32),
                 x.shape[1], proj.shape[0], top_k, float(c.eps),
                 float(cfg.hidden_size ** -0.5), val, idx, t,
                 c.buffer((t, x.shape[1])), c.buffer((t, proj.shape[0])),
                 c.router_slots(layer) if hasattr(c, "router_slots") else 0)
        return val, idx
    val = c.buffer(top_k)
    idx = np.zeros(top_k, dtype=np.int32)
    c.p.emit(ROUTER, x, np.ascontiguousarray(w["router.scale"], dtype=np.float32), proj,
             np.ascontiguousarray(w["router.per_expert_scale"], dtype=np.float32),
             x.shape[1], proj.shape[0], top_k, float(c.eps),
             float(cfg.hidden_size ** -0.5), val, idx,
             c.buffer(x.shape[1]), c.buffer(proj.shape[0]))
    return val, idx


def k_moe(c, h, val, idx, layer):
    """(moe h val idx layer): the selected experts on the row h, and the sum
    of their outputs with the router weights. As Model._moe_one_token."""
    w = c.model._layers[layer]
    gu_q, gu_s = w["experts.gate_up_proj"]
    dn_q, dn_s = w["experts.down_proj"]
    inner = c.cfg.moe_intermediate_size
    out = c.buffer(h.shape)
    if h.shape[0] > 1:
        # The group path. The scratch holds the pairs and the jobs of the
        # sort in C, then the outputs of the two expert kernels.
        t, top_k = idx.shape
        pairs = t * top_k
        i32 = lambda n: np.zeros(n, dtype=np.int32)  # noqa: E731
        c.p.emit(MOE_MT, h, val, idx, t, top_k, gu_q, gu_s, dn_q, dn_s, gu_q.shape[1],
                 h.shape[1], dn_q.shape[1], inner, i32(pairs), i32(pairs), i32(pairs + 1),
                 i32(pairs), i32(pairs), i32(1), c.buffer((pairs, gu_q.shape[1])),
                 c.buffer((pairs, inner)), c.buffer((pairs, dn_q.shape[1])), out)
        return out
    top_k = idx.size
    c.p.emit(MOE, h, val, idx, top_k, gu_q, gu_s, dn_q, dn_s, gu_q.shape[1], h.shape[1],
             dn_q.shape[1], inner, np.zeros(top_k, dtype=np.int32),
             c.buffer((top_k, 2 * inner)), c.buffer((top_k, inner)),
             c.buffer((top_k, dn_q.shape[1])), out)
    return out


def k_gelu(c, x, out=None):
    """(gelu x): the tanh form of GELU of each value. As ops.gelu_tanh."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(GELU, x, out, x.size)
    return out


def k_mul_v(c, a, b, out=None):
    """(mul_v a b): a * b for each value. As the NumPy product."""
    out = c.buffer(a.shape) if out is None else out
    c.p.emit(MUL, a, b, out, a.shape[0], a.size // a.shape[0], a.size // a.shape[0])
    return out


def k_mul_pli(c, a, pl, layer, out=None):
    """(mul_pli a pl layer): a times the per-layer input of a layer. The
    array pl is (tokens, layers * n). Thus the slice of one layer has a row
    stride."""
    n = a.shape[1]
    out = c.buffer(a.shape) if out is None else out
    c.p.emit(MUL, a, pl.ctypes.data + 4 * layer * n, out, a.shape[0], n, pl.shape[1])
    return out


def k_linear(c, mat, x, out=None):
    """(linear m x): a matrix of the E4B model on the rows of x. An int4
    matrix uses the int4 kernels. A bfloat16 matrix uses the bfloat16 GEMV,
    as E4B.linear does for one token and for a group."""
    if isinstance(mat, tuple):
        return k_int4(c, mat, x, out)
    out = c.buffer((x.shape[0], mat.shape[0])) if out is None else out
    c.p.emit(BF16_LINEAR, x, mat, out, mat.shape[0], x.shape[1], x.shape[0])
    return out


def k_rms_norm_rows(c, x, cols, w, out=None):
    """(rms_norm_rows x cols w): the norm of each part of cols values of x."""
    view = x.reshape(-1, cols)
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(RMS_NORM, view, w, out, view.shape[0], cols, float(c.eps))
    return out


def k_qkv_norm(c, q, k, v, qn, kn, layer):
    """(qkv_norm q k v qn kn layer): the norms of q, k, and v, in place. A
    shared layer gives None for k and v. As ops.qkv_norm."""
    hd = c.cfg.plan[layer].head_dim
    c.p.emit(QKV_NORM, q, qn, q.size // hd, k, kn, 0 if k is None else k.size // hd,
             v, 0 if v is None else v.size // hd, hd, float(c.eps))


def k_rope(c, q, k, cos, sin, layer):
    """(rope q k cos sin layer): the rope of the query and the key, in place.
    As ops.rope_apply."""
    plan = c.cfg.plan[layer]
    hd = plan.head_dim
    c.p.emit(ROPE, q, q.size // hd, plan.num_q_heads, k, 0 if k is None else k.size // hd,
             plan.num_kv_heads, cos, sin, hd)


def k_kv_write_heads(c, layer, k, v):
    """(kv_write_heads layer k v): store the keys and the values of the tokens
    in the E4B cache, which keeps (heads, positions, head_dim)."""
    plan = c.cfg.plan[layer]
    per = plan.num_kv_heads * plan.head_dim
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    c.p.emit(KV_WRITE_HEADS, k, v, s("k"), s("v"), s("hs"), c.p.slot("pos"), k.size // per,
             plan.num_kv_heads, plan.head_dim)


def k_attn_e4b(c, layer, q):
    """(attn_e4b layer q): the attention of the queries over the E4B cache. A
    shared layer reads the buffers of its source layer. As E4B.attention."""
    plan = c.cfg.plan[layer]
    hd, qh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
    t = q.size // (qh * hd)
    s = lambda name: c.p.slot("%s.%d" % (name, plan.source))  # noqa: E731
    out = c.buffer((t, qh * hd))
    slide = 1 if os.environ.get("NP_GEMMA_SLIDE", "1") == "1" else 0
    c.p.emit(ATTN_F32H, q, s("k"), s("v"), c.p.slot("scores"), out, qh, kvh, hd, t,
             c.p.slot("pos"), s("hs"), plan.window, slide)
    return out


KERNELS = {
    "rms_norm": k_rms_norm,
    "add": k_add,
    "mul": k_mul,
    "copy": k_copy,
    "int4_multi4": k_int4_multi4,
    "rms_norm_multi4": k_rms_norm_multi4,
    "gelu_mul_int4": k_gelu_mul_int4,
    "int4": k_int4,
    "qkv_norm_rope": k_qkv_norm_rope,
    "kv_write": k_kv_write,
    "attn_qc": k_attn_qc,
    "attn_f32": k_attn_f32,
    # The mode is part of the name. A bare string operand is a name for the
    # compiler, so the form cannot give the mode as an operand.
    "attn_rows_qc": lambda c, layer, q: k_attn_rows(c, layer, q, "qc"),
    "attn_rows_f32": lambda c, layer, q: k_attn_rows(c, layer, q, "f32"),
    "router": k_router,
    "gelu": k_gelu,
    "gelu_mul": k_gelu_mul,
    "add_norm": k_add_norm,
    "add_norm2": k_add_norm2,
    "mul_v": k_mul_v,
    "mul_pli": k_mul_pli,
    "linear": k_linear,
    "rms_norm_rows": k_rms_norm_rows,
    "qkv_norm": k_qkv_norm,
    "rope": k_rope,
    "kv_write_heads": k_kv_write_heads,
    "attn_e4b": k_attn_e4b,
    "moe": k_moe,
}


# ---- the forms of the 26B model -------------------------------------------------

def layer_form(model, i, attn="qc", t=1):
    """Return one decoder layer as a nested expression.

    The expression follows Model._decoder_layer and Model._attention for t
    tokens. The kernel operations select the kernels of one token or of a
    group from the row count of their input. The attention of a group finds
    the key rows of each query itself. The mode attn is "qc" for the int16
    cache and "f32" for the float cache. x is the hidden state. The layer
    changes it in place. A model with the
    mixture-of-experts block (the 26B) adds the router and the experts; the
    dense model (the 12B) does not.
    """
    cfg = model.cfg
    plan = cfg.plan[i]
    kind = "s" if plan.is_sliding else "f"

    def w(name):
        return ("w", i, name)

    if plan.k_eq_v:
        qkv = (("let", ("q", "k"), ("int4_multi4", "h", w("self_attn.q_proj"),
                                    w("self_attn.k_proj"))),
               ("let", "v", ("copy", "k")))
    else:
        qkv = (("let", ("q", "k", "v"), ("int4_multi4", "h", w("self_attn.q_proj"),
                                         w("self_attn.k_proj"), w("self_attn.v_proj"))),)
    base = "base.%d" % i
    if plan.is_sliding:
        lo = ("max", 0, ("-", "pos", plan.sliding_window - 1, base))
    else:
        lo = 0
    qc = 1 if attn == "qc" else 0
    if t == 1:
        attn_forms = (("let", "lo", lo),
                      ("let", "n", ("-", ("+", "pos", 1), base, "lo")),
                      ("let", "a", ("attn_" + attn, i, "q", "lo", "n")))
    else:
        attn_forms = (("let", "a", ("attn_rows_" + attn, i, "q")),)
    # The router and the experts come before the dense feed-forward part. The
    # two parts read the same x and do not depend on each other, so the order
    # does not change the result. On a GPU (np_gemma/gpu.py), the CPU then
    # computes the experts while the GPU computes the dense part.
    if cfg.enable_moe_block:
        experts = (("let", ("val", "idx"), ("router", "x", i)),
                   ("let", "e", ("moe", ("rms_norm", "x", w("pre_feedforward_layernorm_2")),
                                 "val", "idx", i)))
        ffn = (("let", "f", ("add", ("rms_norm", "m", w("post_feedforward_layernorm_1")),
                             ("rms_norm", "e", w("post_feedforward_layernorm_2")))),)
    else:
        experts = ()
        ffn = (("let", "f", "m"),)
    return ("layer", i,
            ("let", "h", ("rms_norm", "x", w("input_layernorm"))),
            *qkv,
            ("qkv_norm_rope", "q", "k", "v", w("self_attn.q_norm"), w("self_attn.k_norm"),
             "cos." + kind, "sin." + kind, i),
            ("let", "row", ("-", "pos", base)),
            ("kv_write", i, "k", "v", "row", qc),
            *attn_forms,
            ("let", "o", ("int4", w("self_attn.o_proj"), "a")),
            ("set", "x", ("add", "x", ("rms_norm", "o", w("post_attention_layernorm")))),
            *experts,
            ("let", ("g", "u"), ("rms_norm_multi4", "x", w("pre_feedforward_layernorm"),
                                 w("mlp.gate_proj"), w("mlp.up_proj"))),
            ("let", "m", ("gelu_mul_int4", "g", "u", w("mlp.down_proj"))),
            *ffn,
            ("set", "x", ("add", "x", ("rms_norm", "f", w("post_feedforward_layernorm")))),
            ("set", "x", ("mul", "x", w("layer_scalar"))))


def step_form(model, attn="qc", t=1):
    """Return a whole decode step of t tokens: every layer, then the final
    norm into xn."""
    layers = [layer_form(model, i, attn, t) for i in range(model.cfg.num_hidden_layers)]
    return ("seq", *layers, ("let", "xn", ("rms_norm", "x", ("w", None, "norm"))))


def format_form(form, indent=0):
    """Return a nested expression as Lisp text."""
    pad = " " * indent

    def atom(x):
        if isinstance(x, tuple):
            return None
        return str(x)

    if not isinstance(form, tuple):
        return pad + str(form)
    flat = [atom(x) for x in form]
    if all(a is not None for a in flat) and len(" ".join(flat)) < 70:
        return pad + "(" + " ".join(flat) + ")"
    if form[0] == "w":
        return pad + "(w %s %s)" % form[1:]
    head = str(form[0]) if not isinstance(form[0], tuple) else format_form(form[0])
    parts = [pad + "(" + head]
    for x in form[1:]:
        if isinstance(x, tuple) and x and x[0] == "w":
            parts[-1] += " (w %s)" % x[2]
        elif isinstance(x, tuple) and all(not isinstance(y, tuple) for y in x):
            parts[-1] += " (" + " ".join(str(y) for y in x) + ")"
        elif isinstance(x, tuple):
            parts.append(format_form(x, indent + 2))
        else:
            parts[-1] += " " + str(x)
    return "\n".join(parts) + ")"


def compile_layers(model, layers, attn="qc", t=1):
    """Compile decoder layers for t tokens into one Program. The buffer "x"
    is the input and the output."""
    c = Compiler(model)
    c.env["x"] = np.zeros((t, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    for i in layers:
        c.compile(layer_form(model, i, attn, t))
    c.p.layers = list(layers)
    c.p.attn = attn
    c.p.tokens = t
    return c.p.finish()


def compile_step(model, attn="qc", t=1):
    """Compile a whole step of t tokens. "x" is the input embedding and "xn"
    the hidden state after the final norm, the input of the output head. A
    step of 2 to 16 tokens is the verify step of MTP."""
    c = Compiler(model)
    c.env["x"] = np.zeros((t, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    c.compile(step_form(model, attn, t))
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = attn
    c.p.tokens = t
    return c.p.finish()


def ready(model, cache):
    """Return the attention mode of a step program for this cache, or None.

    The mode "qc" needs the int16 copy of every layer. It is on after 128
    tokens. Before that, the Python path runs the step, because the step
    that turns the int16 copy on also quantizes the old rows.
    """
    if ops.attn_ready():
        if all(cache.qc_ready(i) for i in range(model.cfg.num_hidden_layers)):
            return "qc"
        return None
    return "f32"


def decode_step(model, cache, tokens, pos):
    """Run a step of one or more tokens with a program. Return the hidden
    state after the final norm, shape (tokens, hidden).

    The model keeps one program for each attention mode and token count. It
    makes a program on the first call with that mode and count.
    """
    tokens = [int(x) for x in tokens]
    attn = ready(model, cache)
    n_parts = int(os.environ.get("NP_GEMMA_PARTS", "1"))
    if n_parts > 1 and len(tokens) == 1:
        from . import parts
        return parts.decode_step(model, cache, tokens, pos, attn, n_parts)
    progs = model.__dict__.setdefault("_programs", {})
    key = (attn, len(tokens))
    prog = progs.get(key)
    if prog is None:
        prog = progs[key] = compile_step(model, attn, len(tokens))
    bind_step(prog, model, cache, pos)
    prog.names["x"][:] = model.embed(tokens)
    prog.run()
    return prog.names["xn"].copy()


def bind_step(prog, model, cache, pos):
    """Prepare the cache for the tokens of a step and bind the parameters.

    The cache work stays in Python: KVCache.prepare drops old rows and grows
    a buffer. The program then writes the new rows in C. In the mode "qc",
    the int16 copy of the cache must be on for every layer of the program.
    """
    kw = step_params(prog, model, cache, pos)
    prog.bind(**kw)
    return kw


def step_params(prog, model, cache, pos):
    """Prepare the cache for the tokens of a step. Return the parameters of
    the program as a dict. See bind_step."""
    kw = {"pos": pos}
    keep = []
    qc = getattr(prog, "attn", "qc") == "qc"
    t = getattr(prog, "tokens", 1)
    for i in prog.layers:
        assert not qc or cache.qc_ready(i), "the program needs the int16 cache"
        cache.prepare(i, pos, t)
        cache.end[i] = pos + t
        kw.update({"base.%d" % i: cache.base[i], "k.%d" % i: cache.k[i],
                   "v.%d" % i: cache.v[i]})
        if qc:
            kw.update({"kq.%d" % i: cache.kq[i], "ks.%d" % i: cache.ks[i],
                       "vq.%d" % i: cache.vq[i], "vs.%d" % i: cache.vs[i]})
    positions = np.arange(pos, pos + t)
    for kind, sliding in (("s", True), ("f", False)):
        plan = next((model.cfg.plan[i] for i in prog.layers
                     if model.cfg.plan[i].is_sliding == sliding), None)
        if plan is None:
            continue
        cos, sin, _ca, _sa = model._rope(plan, positions)
        keep += [cos, sin]
        kw["cos." + kind] = cos
        kw["sin." + kind] = sin
    # The scores of the widest attention: all queries of all heads over all
    # the keys.
    need = max(t * model.cfg.plan[i].num_q_heads * (pos + t) for i in prog.layers)
    sc = getattr(prog, "scores", None)
    if sc is None or sc.size < need:
        sc = np.zeros(max(need, 2 * (sc.size if sc is not None else 0)), dtype=np.float32)
        prog.scores = sc
    kw["scores"] = sc
    prog.keep_bound = keep
    return kw


# ---- the E4B model ------------------------------------------------------------

E4B_PREFIX = "model.language_model."


def e4b_layer_form(model, i, fused=False):
    """Return one decoder layer of the E4B model as a nested expression.

    The expression follows E4B.layer, E4B.attention, and E4B.mlp. A shared
    layer computes only the query. It reads the key and the value of its
    source layer.

    fused uses add_norm and gelu_mul, which do two or three operations in
    one pass and give the same values. The GPU group uses it. fused 2 also
    computes each norm of x in the kernel that changes x (add_norm2): the
    norm before the feed-forward part, and the input norm of the next layer
    (or the final norm into xn). Layer 0 then starts with its own norm, and
    the other layers take h from the layer before. The GPU decode step uses
    it; the buffer h goes from one layer to the next, so it needs a compiler
    that does not reuse the buffers of a layer.
    """
    plan = model.cfg.plan[i]
    kind = "s" if plan.is_sliding else "f"
    p = E4B_PREFIX + "layers.%d." % i

    def T(k):
        return ("t", p + k)

    def M(k):
        return ("m", p + k)

    def add_norm(o, w, scale=None):
        """x += rms_norm(o) w, then x *= scale."""
        if fused:
            return (("set", "x", ("add_norm", "x", o, T(w)) + ((T(scale),) if scale else ())),)
        forms = (("set", "x", ("add", "x", ("rms_norm", o, T(w)))),)
        return forms + ((("set", "x", ("mul", "x", T(scale))),) if scale else ())

    if plan.shared:
        attn = (("let", "q", ("linear", M("self_attn.q_proj"), "h")),
                ("qkv_norm", "q", None, None, T("self_attn.q_norm.weight"), None, i),
                ("rope", "q", None, "cos." + kind, "sin." + kind, i))
    else:
        attn = (("let", ("q", "k", "v"), ("int4_multi4", "h", M("self_attn.q_proj"),
                                         M("self_attn.k_proj"), M("self_attn.v_proj"))),
                ("qkv_norm", "q", "k", "v", T("self_attn.q_norm.weight"),
                 T("self_attn.k_norm.weight"), i),
                ("rope", "q", "k", "cos." + kind, "sin." + kind, i),
                ("kv_write_heads", i, "k", "v"))
    if fused == 2:
        last = i == model.cfg.num_hidden_layers - 1
        nxt = ("t", E4B_PREFIX + "norm.weight") if last else \
            ("t", E4B_PREFIX + "layers.%d.input_layernorm.weight" % (i + 1))
        head = (("let", "h", ("rms_norm", "x", T("input_layernorm.weight"))),) if i == 0 else ()
        mid = (("let", "hf", ("add_norm2", "x", "o", T("post_attention_layernorm.weight"),
                              T("pre_feedforward_layernorm.weight"))),)
        ffn_in = "hf"
        tail = (("let", "xn" if last else "h",
                 ("add_norm2", "x", "pp", T("post_per_layer_input_norm.weight"), nxt,
                  T("layer_scalar"))),)
    else:
        head = (("let", "h", ("rms_norm", "x", T("input_layernorm.weight"))),)
        mid = add_norm("o", "post_attention_layernorm.weight")
        ffn_in = ("rms_norm", "x", T("pre_feedforward_layernorm.weight"))
        tail = add_norm("pp", "post_per_layer_input_norm.weight", "layer_scalar")
    return ("layer", i,
            *head,
            *attn,
            ("let", "a", ("attn_e4b", i, "q")),
            ("let", "o", ("linear", M("self_attn.o_proj"), "a")),
            *mid,
            ("let", ("g", "u"), ("int4_multi4", ffn_in, M("mlp.gate_proj"), M("mlp.up_proj"))),
            ("let", "d", ("linear", M("mlp.down_proj"),
                          ("gelu_mul", "g", "u") if fused else ("mul_v", ("gelu", "g"), "u"))),
            *add_norm("d", "post_feedforward_layernorm.weight"),
            ("let", "pg", ("gelu", ("linear", M("per_layer_input_gate"), "x"))),
            ("let", "pp", ("linear", M("per_layer_projection"), ("mul_pli", "pg", "pl", i))),
            *tail)


def e4b_step_form(model, fused=False):
    """Return a whole step of the E4B model.

    The step starts with the per-layer inputs, as E4B.per_layer_inputs does
    it. That is the projection of x, its scale and norm, and the sum with the
    token part "tok". Then come the layers and the final norm into xn.
    """
    cfg = model.cfg
    n = cfg.hidden_size_per_layer_input
    return ("seq",
            ("let", "proj", ("linear", ("m", E4B_PREFIX + "per_layer_model_projection"), "x")),
            ("set", "proj", ("mul", "proj", cfg.per_layer_model_projection_scale)),
            ("let", "pn", ("rms_norm_rows", "proj", n,
                           ("t", E4B_PREFIX + "per_layer_projection_norm.weight"))),
            ("let", "pl", ("mul", ("add", "pn", "tok"), cfg.per_layer_input_scale)),
            *[e4b_layer_form(model, i, fused) for i in range(cfg.num_hidden_layers)],
            *(() if fused == 2 else
              (("let", "xn", ("rms_norm", "x", ("t", E4B_PREFIX + "norm.weight"))),)))


def compile_e4b_step(model, t=1, fused=False):
    """Compile a whole E4B step of t tokens. "x" and "tok" are the inputs;
    "xn" is the hidden state after the final norm. fused selects the fused
    operations of e4b_layer_form (the GPU)."""
    cfg = model.cfg
    c = Compiler(model)
    c.env["x"] = np.zeros((t, cfg.hidden_size), dtype=np.float32)
    c.env["tok"] = np.zeros((t, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input),
                            dtype=np.float32)
    c.p.slot("pos")
    c.compile(e4b_step_form(model, fused))
    c.p.tokens = t
    return c.p.finish()


def e4b_ready(model, cache):
    """Return True when an E4B step can run as a program.

    The program uses the fused float attention and the int4 blocks of a GGUF
    file. The cache must hold the buffers of every layer that stores a key.
    """
    if not (ops.attn_ready() and model.mode == "int4" and model._q4 and cache.n > 0):
        return False
    return all(i in cache.kv for i, p in enumerate(model.cfg.plan) if not p.shared)


def e4b_bind_step(prog, model, cache, pos):
    """Prepare the E4B cache for the tokens of a step and bind the parameters."""
    prog.bind(**e4b_step_params(prog, model, cache, pos))


def e4b_step_params(prog, model, cache, pos):
    """Prepare the E4B cache for the tokens of a step. Return the parameters
    of the program as a dict."""
    t = prog.tokens
    cache._reserve(pos + t)
    kw = {"pos": pos}
    for i, plan in enumerate(model.cfg.plan):
        if plan.shared:
            continue
        k, v = cache.kv[i]
        kw.update({"k.%d" % i: k, "v.%d" % i: v, "hs.%d" % i: k.shape[1] * k.shape[2]})
    cache.n = max(cache.n, pos + t)
    keep = []
    for kind, sliding in (("s", True), ("f", False)):
        plan = next(p for p in model.cfg.plan if p.is_sliding == sliding)
        cos, sin = model.rope_tables(plan, pos, t)
        cos = np.ascontiguousarray(cos, dtype=np.float32)
        sin = np.ascontiguousarray(sin, dtype=np.float32)
        keep += [cos, sin]
        kw["cos." + kind] = cos
        kw["sin." + kind] = sin
    need = max(p.num_q_heads * (pos + t) for p in model.cfg.plan)
    sc = getattr(prog, "scores", None)
    if sc is None or sc.size < need:
        sc = np.zeros(max(need, 2 * (sc.size if sc is not None else 0)), dtype=np.float32)
        prog.scores = sc
    kw["scores"] = sc
    prog.keep_bound = keep
    return kw


def decode_step_e4b(model, cache, tokens, pos):
    """Run an E4B step of one or more tokens with a program. Return the hidden
    state after the final norm, shape (tokens, hidden)."""
    cfg = model.cfg
    ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
    progs = model.__dict__.setdefault("_programs", {})
    prog = progs.get(ids.size)
    if prog is None:
        prog = progs[ids.size] = compile_e4b_step(model, ids.size)
    e4b_bind_step(prog, model, cache, pos)
    # The inputs, as E4B.forward and E4B.per_layer_inputs make them.
    prog.names["x"][:] = model.embed_rows(E4B_PREFIX + "embed_tokens", ids) * cfg.embed_scale
    tok = model.embed_rows(E4B_PREFIX + "embed_tokens_per_layer", ids)
    prog.names["tok"][:] = (tok * cfg.per_layer_embed_scale).reshape(ids.size, -1)
    prog.run()
    return prog.names["xn"].copy()
