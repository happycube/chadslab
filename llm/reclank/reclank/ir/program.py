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

The attention mode attn follows the form of the cache (KVCache keeps only
quantized rows): "qc" int16, "q8" int8, "qv" int16 keys and int8 values. The
mode "f32" reads a float cache (the E4B and parts of the GPU; KVCache has
none).
Model.forward uses decode_step for one token when ready() allows it. The
program gives the same bits as the Python loop of Model; see
scripts/check_program.py.
"""
from __future__ import annotations

import ctypes
import mmap
import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.core import ops


# The size of a record, the tags of the operands, and the operation codes:
# the one table of reclank/ir/opcodes.py (the C header backends/include/opcodes.h
# comes from it).
from reclank.ir.opcodes import *  # noqa: E402,F401,F403

# the ctypes of the keys and of the values of each record (Program.run_py)
_QTYPES = {op: (ctypes.c_int8 if op in (KV_WRITE8, ATTN_Q8, ATTN_Q8_MT) else ctypes.c_int16,
                ctypes.c_int16 if op in (KV_WRITE, ATTN_QC, ATTN_QC_MT) else ctypes.c_int8)
           for op in (KV_WRITE, ATTN_QC, ATTN_QC_MT, KV_WRITE8, ATTN_Q8, ATTN_Q8_MT,
                      KV_WRITEV8, ATTN_V8, ATTN_V8_MT)}

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
        # the buffers the compiler made for results (Compiler.buffer): the
        # temporaries that ir/alloc.pack may give the bytes of one arena
        self.temps = []
        # the team of gemma_run_task for this program (word 3 of the header),
        # or 0 for its default
        self.threads = 0
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
        buf[3] = self.threads
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
            raise RuntimeError("rc_run returned %d" % rc)

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
        L.rc_rms_norm(V(0), V(1) or None, V(2), V(3), V(4), F(5))
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
        L.rc_int4_linear(V(1), V(2), V(0), V(3), V(4), V(5), 1, 32)
    elif op == INT4_MULTI4:
        args = []
        for m in range(4):
            b = 2 + 4 * m
            args += [V(b) or None, V(b + 1) or None, V(b + 2) or None, V(b + 3)]
        L.rc_int4_multi4(*args, V(0), V(1))
    elif op == RMS_NORM_MULTI4:
        args = []
        for m in range(4):
            b = 5 + 4 * m
            args += [V(b) or None, V(b + 1) or None, V(b + 2) or None, V(b + 3)]
        L.rc_rms_norm_multi4(V(0), V(1), V(2), V(3), F(4), *args)
    elif op == GELU_MUL_INT4:
        L.rc_gelu_mul_int4(V(0), V(1), V(2), V(3), V(4), V(5), V(6), V(7), V(8))
    elif op == QKV_NORM_ROPE:
        L.rc_qkv_norm_rope(V(0), V(1), V(2), V(3), V(4), V(5), V(6), V(7),
                              V(8), V(9), V(10), V(11), V(12), F(13))
    elif op in (KV_WRITE, KV_WRITE8, KV_WRITEV8):
        n = V(8)
        k = _arr(V(0), n)
        v = _arr(V(1), n)
        if V(2):
            _arr(V(2), n)[:] = k
            _arr(V(3), n)[:] = v
        if not V(4):
            return
        kt, vt = _QTYPES[op]
        for src, b in ((k, 4), (v, 6)):
            quant = ops.quantize_i8 if (kt if b == 4 else vt) is ctypes.c_int8 else ops.quantize_i16
            xq, xs = quant(src.reshape(-1, 32))
            _arr(V(b), n, kt if b == 4 else vt)[:] = xq.reshape(-1)
            _arr(V(b + 1), n // 32)[:] = xs.reshape(-1)
    elif op in (ATTN_QC, ATTN_Q8, ATTN_V8):
        qh, kvh, hd, n0 = V(7), V(8), V(9), V(10)
        kt, vt = _QTYPES[op]
        t = V(11) if len(a) > 11 else 1       # a group: query j over n0 + j rows
        win = V(12) if len(a) > 12 else 0     # a window: from row max(0, n0 + j - win)
        q = _arr(V(0), t * qh * hd).reshape(t, qh, hd)
        out = _arr(V(6), t * qh * hd).reshape(t, -1)
        per = kvh * hd
        for j in range(t):
            lo = max(0, n0 + j - win) if win else 0
            n = n0 + j - lo
            kb, vb = ctypes.sizeof(kt) * per, ctypes.sizeof(vt) * per
            out[j] = ops.attn_decode(q[j], _arr(V(1) + lo * kb, n * per, kt),
                                     _arr(V(2) + lo * per // 8, n * per // 32),
                                     _arr(V(3) + lo * vb, n * per, vt),
                                     _arr(V(4) + lo * per // 8, n * per // 32), qh, kvh, hd,
                                     n).reshape(-1)
    elif op == ATTN_F32:
        qh, kvh, hd, n = V(5), V(6), V(7), V(8)
        o = cops.attn_decode_f32s(_arr(V(0), qh * hd).reshape(qh, hd),
                                  _arr(V(1), n * kvh * hd).reshape(n, kvh, hd),
                                  _arr(V(2), n * kvh * hd).reshape(n, kvh, hd),
                                  V(9), V(10), V(11))
        _arr(V(4), qh * hd)[:] = o.reshape(-1)
    elif op == ROUTER:
        L.rc_router(V(0), V(1), V(2), V(3), V(4), V(5), V(6), F(7), F(8), V(9), V(10))
    elif op == MOE:
        top_k, gu_rows, cols, dn_rows, inner = V(3), V(8), V(9), V(10), V(11)
        h = _arr(V(0), cols)
        val = _arr(V(1), top_k)
        idx = _arr(V(2), top_k, ctypes.c_int32)
        ids = np.unique(idx).astype(np.int32)
        act = np.empty((ids.size, 2 * inner), dtype=np.float32)
        act2 = np.empty((ids.size, inner), dtype=np.float32)
        de = np.empty((ids.size, dn_rows), dtype=np.float32)
        L.rc_moe_gemv_gelu(V(4), V(5), h.ctypes.data, ids.ctypes.data, ids.size,
                              act.ctypes.data, act2.ctypes.data, gu_rows, cols, 0, inner)
        L.rc_int4_moe_gemv(V(6), V(7), act2.ctypes.data, ids.ctypes.data, ids.size,
                              de.ctypes.data, dn_rows, inner, inner)
        out = _arr(V(16), dn_rows)
        out[:] = 0.0
        for j in range(ids.size):
            slot = np.nonzero(idx == ids[j])[0]
            out += de[j] * val[slot[0]]
    elif op == INT4_LINEAR_MT:
        L.rc_int4_linear_mt(V(1), V(2), V(0), V(3), V(4), V(5), V(6))
    elif op == INT4_MULTI4_MT:
        args = []
        for m in range(4):
            b = 3 + 4 * m
            args += [V(b) or None, V(b + 1) or None, V(b + 2) or None, V(b + 3)]
        L.rc_int4_multi4_mt(*args, V(0), V(1), V(2))
    elif op == GELU_MUL_ROWS:
        rows, inner = V(3), V(4)
        g = _arr(V(0), rows * inner).reshape(rows, inner)
        u = _arr(V(1), rows * inner).reshape(rows, inner)
        _arr(V(2), rows * inner)[:] = ops.gelu_mul_rows(g, u).reshape(-1)
    elif op in (ATTN_QC_MT, ATTN_F32_MT, ATTN_Q8_MT, ATTN_V8_MT):
        qc = op != ATTN_F32_MT
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
            kt, vt = _QTYPES[op]
            kq = _arr(V(1), rows * per, kt).reshape(rows, kvh, hd)
            ks = _arr(V(2), rows * per // 32).reshape(rows, kvh, hd // 32)
            vq = _arr(V(3), rows * per, vt).reshape(rows, kvh, hd)
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
        L.rc_router_mt(V(0), V(1), V(2), V(3), V(4), V(5), V(6), F(7), F(8), V(9), V(10),
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
        L.rc_bf16_linear(V(1), V(0), V(2), V(3), V(4), V(5))
    elif op == QKV_NORM:
        L.rc_qkv_norm(V(0), V(1) or None, V(2), V(3) or None, V(4) or None, V(5),
                         V(6) or None, V(7), V(8), F(9))
    elif op == ROPE:
        L.rc_rope(V(0), V(1), V(2), V(3) or None, V(4), V(5), V(6), V(7), V(8))
    elif op == KV_WRITE_HEADS:
        hs, pos, t, kvh, hd = V(4), V(5), V(6), V(7), V(8)
        k = _arr(V(0), t * kvh * hd).reshape(t, kvh, hd)
        v = _arr(V(1), t * kvh * hd).reshape(t, kvh, hd)
        rs = kvh * hd if hs == hd else hd      # position-major or head-major (gp_kv_rs)
        for h in range(kvh):
            for j in range(t):
                base_ = h * hs + (pos + j) * rs
                _arr(V(2) + 4 * base_, hd)[:] = k[j, h]
                _arr(V(3) + 4 * base_, hd)[:] = v[j, h]
    elif op == ATTN_F32H:
        qh, kvh, hd, t, pos, hs, window, slide = (V(5), V(6), V(7), V(8), V(9), V(10),
                                                  V(11), V(12))
        q = _arr(V(0), t * qh * hd).reshape(t, qh, hd)
        out = _arr(V(4), t * qh * hd).reshape(t, qh, hd)
        rs = kvh * hd if hs == hd else hd      # position-major or head-major (gp_kv_rs)
        span = (kvh - 1) * hs + (pos + t - 1) * rs + hd
        K = np.lib.stride_tricks.as_strided(_arr(V(1), span), (kvh, pos + t, hd), (4 * hs, 4 * rs, 4))
        Vv = np.lib.stride_tricks.as_strided(_arr(V(2), span), (kvh, pos + t, hd), (4 * hs, 4 * rs, 4))
        for j in range(t):
            p_ = pos + j
            lo = max(0, p_ - window + 1) if (slide and window) else 0
            out[j] = ops.attn_decode_f32(q[j:j + 1], K[:, lo:p_ + 1, :], Vv[:, lo:p_ + 1, :],
                                         p_, lo, window)[0]
    elif op == ENC_LINEAR:
        n, m, k = V(5), V(6), V(7)
        x = np.clip(_arr(V(0), n * k).reshape(n, k), _f(a, e, 8), _f(a, e, 9))
        if V(2):
            w = ops.bf16_to_f32(_arr(V(1), m * k, ctypes.c_uint16).reshape(m, k).copy())
        else:
            w = _arr(V(1), m * k).reshape(m, k)
        y = x @ w.T
        if V(3):
            y = y + _arr(V(3), m)
        _arr(V(4), n * m).reshape(n, m)[:] = np.clip(y, _f(a, e, 10), _f(a, e, 11))
    elif op == ENC_RMS:
        rows, cols = V(3), V(4)
        x = _arr(V(0), rows * cols).reshape(rows, cols)
        y = x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + _f(a, e, 5))
        if V(1):
            y = y * _arr(V(1), cols)
        _arr(V(2), rows * cols).reshape(rows, cols)[:] = y
    elif op == ENC_LNORM:
        rows, cols = V(4), V(5)
        x = _arr(V(0), rows * cols).reshape(rows, cols)
        d = x - x.mean(axis=-1, keepdims=True)
        y = d / np.sqrt((d * d).mean(axis=-1, keepdims=True) + _f(a, e, 6))
        if V(1):
            y = y * _arr(V(1), cols)
        if V(2):
            y = y + _arr(V(2), cols)
        _arr(V(3), rows * cols).reshape(rows, cols)[:] = y
    elif op == ENC_GELU:
        n = V(2)
        x = _arr(V(0), n).copy()
        if V(3):
            from math import erf
            _arr(V(1), n)[:] = 0.5 * x * (1.0 + np.vectorize(erf)(x / np.sqrt(2.0)))
        else:
            _arr(V(1), n)[:] = ops.gelu_tanh(x)
    elif op == ENC_GELU_MUL:
        n = V(3)
        _arr(V(2), n)[:] = ops.gelu_tanh(_arr(V(0), n).copy()) * _arr(V(1), n)
    elif op == ENC_ADD:
        n = V(2)
        _arr(V(0), n)[:] += np.float32(_f(a, e, 3)) * _arr(V(1), n)
    elif op == ENC_CLAMP:
        n = V(2)
        _arr(V(1), n)[:] = np.clip(_arr(V(0), n), _f(a, e, 3), _f(a, e, 4))
    elif op == ENC_BIAS_CLAMP:
        rows, cols = V(2), V(3)
        y = _arr(V(0), rows * cols).reshape(rows, cols)
        if V(1):
            y += _arr(V(1), cols)
        np.clip(y, _f(a, e, 4), _f(a, e, 5), out=y)
    elif op == ENC_SILU:
        n = V(2)
        x = _arr(V(0), n)
        _arr(V(1), n)[:] = x / (1.0 + np.exp(-x))
    elif op == ENC_SILU_MUL:
        n = V(3)
        g = _arr(V(0), n)
        _arr(V(2), n)[:] = g / (1.0 + np.exp(-g)) * _arr(V(1), n)
    elif op == ENC_MUL_VEC:
        rows, cols = V(3), V(4)
        _arr(V(2), rows * cols).reshape(rows, cols)[:] = (
            _arr(V(0), rows * cols).reshape(rows, cols) * _arr(V(1), cols))
    elif op == ENC_GLU:
        rows, cols = V(2), V(3)
        x = _arr(V(0), rows * 2 * cols).reshape(rows, 2 * cols)
        _arr(V(1), rows * cols).reshape(rows, cols)[:] = x[:, :cols] / (1.0 + np.exp(-x[:, cols:]))
    elif op == ENC_DWCONV:
        t, c, kw = V(3), V(4), V(5)
        x = _arr(V(0), t * c).reshape(t, c)
        w = _arr(V(1), c * kw).reshape(c, kw)
        xp = np.concatenate([np.zeros((kw - 1, c), np.float32), x])
        y = np.zeros((t, c), np.float32)
        for j in range(kw):
            y += xp[j:j + t] * w[:, j]
        _arr(V(2), t * c).reshape(t, c)[:] = y
    elif op == ENC_LOCAL_ATTN:
        t, heads, hd, span, cap = V(6), V(7), V(8), V(9), _f(a, e, 10)
        q, k, v = (_arr(V(i), t * heads * hd).reshape(t, heads, hd) for i in range(3))
        R = _arr(V(3), (span + 1) * heads * hd).reshape(span + 1, heads, hd)
        valid = _arr(V(4), t, ctypes.c_int32).astype(bool)
        idx = np.arange(t)[:, None] - (span - 1) + np.arange(span)[None, :]
        ok = (idx >= 0) & valid[np.clip(idx, 0, t - 1)]
        ci = np.clip(idx, 0, t - 1)
        s = np.einsum("thd,tjhd->thj", q, k[ci] + R[1:span + 1][None])
        s = np.where(ok[:, None, :], np.tanh(s / cap) * cap, np.float32(-1e9))
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        s = np.where((idx >= 0)[:, None, :], s, 0.0)
        _arr(V(5), t * heads * hd).reshape(t, heads, hd)[:] = np.einsum("thj,tjhd->thd", s, v[ci])
    elif op == ENC_ROPE2D:
        n, heads, hd = V(3), V(4), V(5)
        x = _arr(V(0), n * heads * hd).reshape(n, heads, hd)
        pos = _arr(V(1), 2 * n, ctypes.c_int32).reshape(n, 2)
        inv = _arr(V(2), hd // 4)
        q4, h2 = hd // 4, hd // 2
        for part in range(2):
            ang = pos[:, part:part + 1].astype(np.float32) * inv[None, :]
            c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
            p = x[..., part * h2:(part + 1) * h2]
            a0, b0 = p[..., :q4].copy(), p[..., q4:].copy()
            p[..., :q4] = a0 * c - b0 * s
            p[..., q4:] = b0 * c + a0 * s
    elif op == ENC_ATTN:
        n, heads, hd = V(4), V(5), V(6)
        q, k, v = (_arr(V(i), n * heads * hd).reshape(n, heads, hd) for i in range(3))
        s = np.einsum("qhd,khd->hqk", q, k)
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        _arr(V(3), n * heads * hd).reshape(n, heads, hd)[:] = np.einsum("hqk,khd->qhd", s, v)
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

    q4x: the records of one row on the CPU take the KQ_Q4X copies of the
    int4 matrices (ops.q4x_pack_model). The GPU compilers (reclank/backends/cuda/runtime.py)
    set it False.

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

    The forms of the compiler alone (they make no records; a program that
    uses them has the records of the forms they stand for):

        (scope f ...)          compile each form in a scope of its own: its
                               lets end with it (a layer is a scope too)
        (out name e)           compile e and give its value a name in the
                               block (scope, layer) around the innermost
                               block, past functions and loops (what a layer
                               gives the next: the norm hn, the result xn)
        (quote x)              x as it is (a string, not a name)
        (setq name e)          give e to the nearest name that is bound
        (def f (p ...) f ...)  a function: (f a ...) compiles its forms in a
                               scope where p ... are the values of a ...;
                               its value is that of its last form. A last
                               parameter after &rest gets a tuple of the
                               other values
        (lambda (p ...) f ...) a function with no name, a value (let, an
                               argument of a def): it keeps the lets it sees
                               where it is made (their values then), and a
                               call sees its parameters, those lets, then
                               the scopes of the call. ((lambda ...) a ...)
                               and (e a ...) of any e whose value is a
                               function call it
        (defmacro m (p ...) f ...)
                               a macro: (m a ...) compiles its forms with p
                               ... the forms a ... (not their values); the
                               value of the last is a form, which compiles
                               in place of (m a ...). &rest as in a def
        (qq x)                 x as it is, but (uq e) in it is the value of
                               e, and (uqs e) the values of e spliced in
                               (the forms of a macro)
        (list e ...)           a tuple of the values (a form, in a macro)
        (car e) (cdr e)        the first value of a tuple, the tuple of the
        (cons a e) (len e)     others; a before the values of the tuple e;
                               the count of its values
        (str e ...)            the text of the values, joined (a name of a
                               tensor: (m (str (quote "layers.") i ...)))
        (gensym "p")           a new name (a let of a macro that is not a
                               name of the forms around it)
        (for i lo hi f ...)    the forms for i = lo, ..., hi - 1, each in a
                               scope of its own (unrolled: no loop record)
        (if c a b)             a or b (or nothing) by the value of c, which
                               must be known (an int, a bool)
        (plan i key) (cfg key) a value of the plan of layer i, of the config
        (ref "base." i)        the value of the name "base." + str(i)
        (= a b) (< a b) ... (and ...) (or ...) (not a)
                               comparisons of known values

    Every other head is a function of a def, else a kernel operation of
    KERNELS. A name is a value of a let (the innermost scope first), or else
    a parameter slot of that name. A kernel operation makes a new buffer for
    its result, except in a set.
    """

    q4x = True
    layer_scope = True      # a layer is a scope: its lets end with it

    def __init__(self, model, prog=None):
        self.model = model
        self.cfg = model.cfg
        self.eps = model.cfg.rms_norm_eps
        self.p = prog or Program()
        self.env = self.p.names     # the names of the program (the outermost scope)
        self.frames = []            # the scopes inside it, the innermost last
        self.frame_kinds = []             # "block" (scope, layer), "fn" (a def), "loop" (a for)
        self.n_gensym = 0

    def lookup(self, name):
        """(True, value) of a name bound in a scope, the innermost first, or
        (False, None)."""
        for f in reversed(self.frames):
            if name in f:
                return True, f[name]
        if name in self.env:
            return True, self.env[name]
        return False, None

    def bind(self, name, v, out=False):
        """Give v the name in the current scope. out: in the block (scope,
        layer) around the innermost block, past the scopes of functions and
        loops (what a layer of a for gives the next one), or the names of the
        program."""
        if not out:
            (self.frames[-1] if self.frames else self.env)[name] = v
            return
        k = len(self.frames) - 1
        while k >= 0 and self.frame_kinds[k] != "block":
            k -= 1                  # the innermost block
        k -= 1
        while k >= 0 and self.frame_kinds[k] != "block":
            k -= 1                  # the block around it
        (self.frames[k] if k >= 0 else self.env)[name] = v

    def push(self, frame, kind):
        self.frames.append(frame)
        self.frame_kinds.append(kind)

    def pop(self):
        self.frames.pop()
        self.frame_kinds.pop()

    def rebind(self, name, v):
        """Give v to the nearest scope that has the name."""
        for f in reversed(self.frames):
            if name in f:
                f[name] = v
                return
        if name not in self.env:
            raise NameError("setq of %r, which no scope has" % name)
        self.env[name] = v

    def layer_begin(self, i):
        """A layer starts (a subclass: its pools of buffers, its index)."""

    def layer_end(self, i):
        """A layer ends."""

    def value(self, x):
        """Return the value of an operand of a form.

        A name gives the value of a let, or else a parameter slot of
        PARAMS or PARAM_PREFIXES. Any other name is an error. A tuple is
        a form to compile. Any other value (an int, a float, an array) stays
        as it is.
        """
        if isinstance(x, str):
            found, v = self.lookup(x)
            if found:
                return v
            if x in PARAMS or x.startswith(PARAM_PREFIXES):
                return self.p.slot(x)
            raise NameError("the form uses %r, which no let gives and which is not "
                            "a parameter" % x)
        if isinstance(x, tuple):
            return self.expr(x)
        return x

    def buffer(self, shape, dtype=np.float32):
        """Return a new buffer for a result. The program keeps it (a
        temporary: prog.temps)."""
        b = lazy_zeros(shape, dtype)
        self.p.temps.append(b)
        return b

    def compile(self, form):
        """Compile a top-level form: a seq, a layer, a scope, a for, or one
        expression. Return the value of the last form."""
        if not isinstance(form, tuple):
            return self.value(form)        # a name, a number (a value as it is)
        head = form[0]
        if head == "seq":
            v = None
            for f in form[1:]:
                v = self.compile(f)
            return v
        if head == "layer":
            i = self.value(form[1])
            self.layer_begin(i)
            try:
                return self._block(form[2:], self.layer_scope)
            finally:
                self.layer_end(i)
        if head == "scope":
            return self._block(form[1:], True)
        if head == "for":
            var, lo, hi, body = form[1], self.value(form[2]), self.value(form[3]), form[4:]
            v = None
            for i in range(int(lo), int(hi)):
                self.push({var: i}, "loop")
                try:
                    v = self._block(body, False)
                finally:
                    self.pop()
            return v
        return self.expr(form)

    def _block(self, body, scope):
        """Compile the forms of body (in a new scope); the value of the last."""
        if scope:
            self.push({}, "block")
        try:
            v = None
            for f in body:
                v = self.compile(f)
            return v
        finally:
            if scope:
                self.pop()

    def apply(self, fn, args):
        """The value of (f a ...) of a def or a lambda: its forms in a scope
        of its parameters (and, a lambda, of the lets it keeps)."""
        vals = [self.value(a) for a in args]
        frame = _params(fn, vals)
        n = 0
        if fn.env is not None:
            self.push(fn.env, "fn")
            n += 1
        self.push(frame, "fn")
        n += 1
        try:
            return self._block(fn.body, False)
        finally:
            for _ in range(n):
                self.pop()

    def expand(self, macro, args):
        """The form of (m a ...) of a defmacro: its forms with its parameters
        the forms a ..."""
        self.push(_params(macro, list(args)), "fn")
        try:
            return self._block(macro.body, False)
        finally:
            self.pop()

    def closure(self):
        """The lets of every scope around (the innermost last wins): what a
        lambda keeps."""
        env = {}
        for f in self.frames:
            env.update(f)
        return env

    def quasi(self, x):
        """The value of (qq x)."""
        if not isinstance(x, tuple) or not x:
            return x
        if x[0] == "uq":
            return self.value(x[1])
        out = []
        for y in x:
            if isinstance(y, tuple) and y and y[0] == "uqs":
                out.extend(self.value(y[1]))
            else:
                out.append(self.quasi(y))
        return tuple(out)

    def expr(self, form):
        """Compile one expression and return its value."""
        head, args = form[0], form[1:]
        if head in ("let", "out"):
            names, e = args
            v = self.value(e)
            out = head == "out"
            if isinstance(names, tuple):
                for n_, v_ in zip(names, v):
                    self.bind(n_, v_, out)
            else:
                self.bind(names, v, out)
            return v
        if head == "quote":
            return args[0]
        if head == "setq":
            name, e = args
            v = self.value(e)
            self.rebind(name, v)
            return v
        if head == "set":
            name, e = args
            found, buf = self.lookup(name)
            if not found:
                raise NameError("set of %r, which no let gives" % name)
            return self.expr_into(e, buf)
        if head == "def":
            name, params, body = args[0], tuple(args[1]), args[2:]
            fn = Func(name, params, body)
            self.bind(name, fn)
            return fn
        if head == "lambda":
            return Func("lambda", tuple(args[0]), args[1:], self.closure())
        if head == "defmacro":
            name, params, body = args[0], tuple(args[1]), args[2:]
            m = Macro(name, params, body)
            self.bind(name, m)
            return m
        if head == "qq":
            return self.quasi(args[0])
        if head == "list":
            return tuple(self.value(a) for a in args)
        if head in ("car", "cdr"):
            v = self.value(args[0])
            if not isinstance(v, tuple) or not v:
                raise TypeError("(%s ...) of %r, which is not a tuple of values" % (head, v))
            return v[0] if head == "car" else v[1:]
        if head == "len":
            return len(self.value(args[0]))
        if head == "cons":
            return (self.value(args[0]),) + tuple(self.value(args[1]))
        if head == "str":
            return "".join(str(self.value(a)) for a in args)
        if head == "gensym":
            self.n_gensym += 1
            return "%s#%d" % (args[0] if args else "g", self.n_gensym)
        if head == "if":
            c = self.value(args[0])
            if not isinstance(c, (bool, int, np.integer)):
                raise TypeError("the condition of an if must be known when it compiles, not %r" % (c,))
            if c:
                return self.value(args[1])
            return self.value(args[2]) if len(args) > 2 else None
        if head == "plan":
            return getattr(self.cfg.plan[int(self.value(args[0]))], args[1])
        if head == "cfg":
            return getattr(self.cfg, args[0])
        if head == "ref":
            return self.value("%s%s" % (args[0], self.value(args[1])))
        if head in COMPARE:
            vals = [self.value(a) for a in args]
            if not all(isinstance(v, (bool, int, float, np.integer, type(None))) for v in vals):
                raise TypeError("(%s ...) compares known values, not %r" % (head, vals))
            return COMPARE[head](*vals)
        if head in ("seq", "layer", "scope", "for"):
            return self.compile(form)
        if isinstance(head, str):
            found, fn = self.lookup(head)
            if found and isinstance(fn, Macro):
                return self.compile(self.expand(fn, args))
            if found and isinstance(fn, Func):
                return self.apply(fn, args)
        elif isinstance(head, tuple):
            fn = self.value(head)
            if not isinstance(fn, Func):
                raise TypeError("the head %r of a form is not a function" % (head,))
            return self.apply(fn, args)
        if head == "slot":
            return self.p.slot(args[0])
        if head == "w":
            layer, key = args
            if isinstance(layer, (str, tuple)):
                layer = self.value(layer)       # (a name of a for or a def, a sum)
            if layer is None:
                return {"norm": self.model._norm_w}[key]
            v = self.model._layers[layer][key]
            if (getattr(self, "q4_kq", False) and key in _Q4KQ_KEYS and isinstance(v, tuple)
                    and v[0].dtype == np.uint8 and v[0].ndim == 3 and v[0].shape[-1] == 18):
                # A Q4_0 matrix for the GPU products with int8 x (the GPU
                # step and verify groups of the 26B, RECLANK_GPU_Q4_I8_DENSE).
                return _Q4KQ(v[0])
            return v
        if head == "m":
            # An E4B matrix: the int4 blocks, or else the bfloat16 copy (a
            # small matrix, which W16 keeps in float32, too). A compiler with
            # q4_kq (the GPU step and verify groups) takes a Q4_0 matrix as a
            # GGUF product (E4B.kq_q4: int8 x on the GPU).
            entry = None
            if isinstance(args[0], tuple):
                args = (self.value(args[0]),)      # (str ...)
            if getattr(self, "q4_kq", False) and hasattr(self.model, "kq_q4"):
                entry = self.model.kq_q4(args[0])
            if entry is None:
                entry = self.model.q4(args[0])
            if entry is None and hasattr(self.model, "kq"):
                entry = self.model.kq(args[0])     # a K quant of a GGUF file
            if entry is None:
                entry = self.model.W16(args[0])
            if entry is None:
                entry = ops.to_bf16(self.model.W(args[0]))
            return entry
        if head == "t":
            # An E4B tensor that the quantization did not touch.
            name = self.value(args[0]) if isinstance(args[0], tuple) else args[0]
            return np.ascontiguousarray(self.model.T(name), dtype=np.float32)
        if head in SCALAR:
            return self.scalar(head, [self.value(a) for a in args])
        return self.kernel(head, [self.value(a) for a in args])

    def expr_into(self, form, out):
        """Compile a kernel expression that writes an existing buffer (a
        macro there: its form)."""
        if isinstance(form[0], str):
            found, m = self.lookup(form[0])
            if found and isinstance(m, Macro):
                return self.expr_into(self.expand(m, form[1:]), out)
        return self.kernel(form[0], [self.value(a) for a in form[1:]], out)

    def kernel(self, head, vals, out=None):
        """Compile one kernel operation of KERNELS on the values of its
        operands. The compiler of a part (reclank/models/gemma4/parts.py) changes this."""
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


# The bytes from which lazy_zeros maps an array of its own.
LAZY_MIN = 1 << 20


def lazy_zeros(shape, dtype=np.float32):
    """A new array of zeros for a temporary of a program. From LAZY_MIN bytes
    on, an anonymous map of its own: its pages take no memory until written,
    and munmap gives them back when it goes. A large group of a model with a
    buffer of each layer for each result (packed by ir/alloc.py, which then
    drops them: the device holds only the arena) asks for GBs that no one
    writes; np.zeros on memory that malloc had freed before (its heap)
    wrote them all: 11 GB, 8 s, for a mixed group of 1024 rows of Qwen."""
    dtype = np.dtype(dtype)
    shape = tuple(shape) if isinstance(shape, (tuple, list)) else (int(shape),)
    n = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if n < LAZY_MIN:
        return np.zeros(shape, dtype)
    m = mmap.mmap(-1, n, flags=mmap.MAP_PRIVATE)
    if n >= 4 << 20 and hasattr(mmap, "MADV_HUGEPAGE"):
        m.madvise(mmap.MADV_HUGEPAGE)       # (as numpy does for its large arrays)
    return np.frombuffer(m, dtype).reshape(shape)


# The names of the parameters that bind_step writes. Any other name in a form
# must come from a let. Thus a name with an error stops the compiler.
PARAMS = ("pos", "scores")
PARAM_PREFIXES = ("base.", "cos.", "sin.")
# The comparisons of known values (the forms of the compiler: if).
COMPARE = {
    "=": lambda a, b: a == b, "!=": lambda a, b: a != b, "<": lambda a, b: a < b,
    ">": lambda a, b: a > b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b,
    "and": lambda *v: all(v), "or": lambda *v: any(v), "not": lambda a: not a,
}


class Func:
    """A function of a def, (def name (params ...) forms ...), or of a
    lambda (env: the lets it keeps)."""

    def __init__(self, name, params, body, env=None):
        self.name, self.params, self.body, self.env = name, params, body, env

    def __repr__(self):
        return "<%s %s %s>" % ("lambda" if self.env is not None else "def", self.name, self.params)


class Macro(Func):
    """A macro: (defmacro name (params ...) forms ...)."""

    def __repr__(self):
        return "<defmacro %s %s>" % (self.name, self.params)


def _params(fn, vals):
    """The scope of the parameters of fn (a Func or a Macro) for the values
    (or forms) vals: a last parameter after &rest gets the tuple of the rest."""
    ps = fn.params
    if "&rest" in ps:
        k = ps.index("&rest")
        if k != len(ps) - 2:
            raise TypeError("%s: &rest comes before the last parameter" % fn.name)
        if len(vals) < k:
            raise TypeError("%s takes at least %d values, not %d" % (fn.name, k, len(vals)))
        frame = dict(zip(ps[:k], vals[:k]))
        frame[ps[-1]] = tuple(vals[k:])
        return frame
    if len(vals) != len(ps):
        raise TypeError("%s takes %d values, not %d" % (fn.name, len(ps), len(vals)))
    return dict(zip(ps, vals))


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


def k_ffn_out(c, m, w1, e, w2, w, x, s, wn):
    """(ffn_out m w1 e w2 w x s wn): the end of a layer of the 26B.
    f = rms_norm(m, w1) + rms_norm(e, w2); x = (x + rms_norm(f, w)) s in
    place. Return rms_norm(x, wn), the input norm of the next layer, or x
    when wn is None (the last layer). The records of the separate forms (a
    GPU compiler emits GP_FFN_OUT)."""
    f = k_add(c, k_rms_norm(c, m, w1), k_rms_norm(c, e, w2))
    k_add(c, x, k_rms_norm(c, f, w), out=x)
    k_mul(c, x, s, out=x)
    return x if wn is None else k_rms_norm(c, x, wn)


def k_copy(c, x, out=None):
    """(copy x): a copy of x. The global layers use the key as the value."""
    out = c.buffer(x.shape, x.dtype) if out is None else out
    c.p.emit(COPY, x, out, x.nbytes)
    return out


def _q4x(c, w, s, t=1):
    """The operands of an int4 matrix for a record of t rows (a step, or a
    verify group of MTP): its KQ_Q4X copy and no scales (ops.q4x_pack_model)
    on the CPU (the compiler has q4x; the GPU compilers and that of the parts
    do not), else w and s. Each token of a group has the operations of a
    step: the same bits."""
    if getattr(c, "q4x", False):
        q = ops._Q4X.get(w.ctypes.data)
        if q is not None:
            return q, None
    return w, s


def _mats(c, mats, t=1):
    """Return the operands of up to four int4 matrices and their outputs. The
    outputs are new buffers of the compiler c."""
    args, outs = [], []
    for m in range(4):
        if m < len(mats) and mats[m] is not None:
            w, s = mats[m]
            o = c.buffer((t, w.shape[0]))
            args += list(_q4x(c, w, s, t)) + [o, w.shape[0]]
            outs.append(o)
        else:
            args += [None, None, None, 0]
    return args, outs


def k_int4_multi4(c, x, *mats):
    """(int4_multi4 x m ...): up to four int4 matrices on the row x in one
    kernel. One row uses ops.int4_multi4, and a group of rows uses
    ops.int4_multi4_mt. Return one (rows of x, rows of m) buffer for each
    matrix. A matrix that is not an int4 pair (a K quant of a GGUF file of
    the E2B or the E4B) makes each matrix one (linear m x)."""
    t = x.shape[0]
    if any(m is not None and not isinstance(m, tuple) for m in mats):
        # One quantization of x for the GGUF matrices of the group.
        xq = k_kq_quant(c, x) if any(_is_kq(m) for m in mats if m is not None) else None
        return tuple(None if m is None else k_kq(c, m, x, xq=xq) if _is_kq(m) else
                     k_linear(c, m, x) for m in mats)
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
    if x.shape[0] > 1 or any(_is_kq(m) for m in mats if m is not None):
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
    if g.shape[0] > 1 or _is_kq(mat):
        # A GGUF product (_Q4KQ) also takes this form: the GPU makes the
        # gate, the up matrix, and GELU_MUL_ROWS one kernel (k_kq_glu_i8).
        t, inner = g.shape
        h = c.buffer((t, inner))
        c.p.emit(GELU_MUL_ROWS, g, u, h, t, inner)
        return k_int4(c, mat, h, out)
    w, s = mat
    rows, cols = w.shape[0], g.size
    out = c.buffer((1, rows)) if out is None else out
    c.p.emit(GELU_MUL_INT4, g, u, g.size, c.buffer(g.size), *_q4x(c, w, s), out, rows, cols)
    return out


def k_int4(c, mat, x, out=None):
    """(int4 m x): the int4 matrix m on the rows of x. As Model.linear for
    one token, and as ops.linear_int4_mt for a group. A matrix of the GGUF
    products (_Q4KQ) makes GP_KQ_LINEAR."""
    if _is_kq(mat):
        return k_kq(c, mat, x, out)
    w, s = mat
    t = x.shape[0]
    out = c.buffer((t, w.shape[0])) if out is None else out
    if t == 1:
        c.p.emit(INT4_LINEAR, x, *_q4x(c, w, s), out, w.shape[0], x.shape[1])
    else:
        c.p.emit(INT4_LINEAR_MT, x, *_q4x(c, w, s, t), out, w.shape[0], x.shape[1], t)
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
    # qc 1: the int16 cache; qc 2: the int8 cache (1 byte for each value); qc
    # 3: int16 keys and int8 values. KVCache has no float rows, so the record
    # stores none (null addresses).
    kesz = 1 if qc == 2 else 2
    vesz = 1 if qc in (2, 3) else 2
    if qc:
        # An int16 row has 2 bytes for each value and 4 bytes for each scale.
        q = [_addr(c, s("kq"), row, kesz * per), _addr(c, s("ks"), row, 4 * (per // 32)),
             _addr(c, s("vq"), row, vesz * per), _addr(c, s("vs"), row, 4 * (per // 32))]
    else:
        q = [0, 0, 0, 0]
    # The rows of a group are adjacent in the cache, so one copy stores them.
    if qc:
        c.p.emit({1: KV_WRITE, 2: KV_WRITE8, 3: KV_WRITEV8}[qc], k, v, 0, 0, *q, k.size)
        return
    c.p.emit(KV_WRITE, k, v,
             _addr(c, s("k"), row, 4 * per), _addr(c, s("v"), row, 4 * per), *q, k.size)


def k_tq_rot(c, x, inverse):
    """The rotation of each group of 32 values of x in place, or its inverse
    (the rotated forms of the cache, model.kv_rot: rq8, k16vr8)."""
    c.p.emit(TQ_ROT, x, x.size // 32, int(inverse))


def kv_rot_forms():
    """The forms of the rotations of a layer: after qkv_norm_rope (the query
    and the keys for rotated keys, the values for rotated values), and after
    the attention (its output back, for rotated values)."""
    from reclank.models.gemma4.model import kv_rot
    rk, rv = kv_rot()
    before = ((("tq_rot", "q", 0), ("tq_rot", "k", 0)) if rk else ()) + \
        ((("tq_rot", "v", 0),) if rv else ())
    after = (("tq_rot", "a", 1),) if rv else ()
    return before, after


def k_attn_qc(c, layer, q, lo, n, q8=False, t=1, window=0):
    """(attn_qc layer q lo n): the fused attention of one float32 query over
    n rows of the int16 cache, from buffer row lo. As Model._attend_one.
    (attn_q8 layer q lo n): the same over the int8 cache (q8 True), and
    (attn_qv layer q lo n) over int16 keys and int8 values (q8 "v"). t > 1:
    t queries (rows of q), query j over n + j rows (operand 11; the GPU
    kernel of a global layer of the 26B only, gpu.py attn_rows_small).
    window > 0 (operand 12; the GPU kernel of a sliding layer of the 26B):
    query j over the rows max(0, n + j - window) to n + j - 1 from lo, and
    operand 13 the base of the layer (the position of row 0)."""
    plan = c.cfg.plan[layer]
    hd, qh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
    per = kvh * hd
    kesz = 1 if q8 is True else 2
    vesz = 1 if q8 else 2
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    out = c.buffer((t, qh * hd))
    op = ATTN_V8 if q8 == "v" else (ATTN_Q8 if q8 else ATTN_QC)
    c.p.emit(op, q,
             _addr(c, s("kq"), lo, kesz * per), _addr(c, s("ks"), lo, 4 * (per // 32)),
             _addr(c, s("vq"), lo, vesz * per), _addr(c, s("vs"), lo, 4 * (per // 32)),
             c.p.slot("scores"), out, qh, kvh, hd, n,
             *((t, window, s("base")) if window else (t,) if t > 1 else ()))
    return out


def k_attn_f32(c, layer, q, lo, n):
    """(attn_f32 layer q lo n): the attention of one query over n rows of
    the float cache, from buffer row lo. As Model._attend_one with the C
    kernel (RECLANK_F32_ATTN=c)."""
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
    if attn in ("qc", "q8", "qv"):
        # "lim" (a GPU group, compile_split_group): the last key of each query
        # less pos, so that the tokens of an image see each other.
        lim = c.env.get("lim")
        op = {"q8": ATTN_Q8_MT, "qv": ATTN_V8_MT}.get(attn, ATTN_QC_MT)
        c.p.emit(op, q, s("kq"), s("ks"), s("vq"),
                 s("vs"), c.p.slot("scores"), out,
                 qh, kvh, hd, t, pos, base, window, np.zeros(t, np.int32),
                 np.zeros(t, np.int32), lim)
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
    q4x = ops._Q4X_MOE.get(gu_q.ctypes.data) if getattr(c, "q4x", False) else None
    if q4x is not None:
        # The experts in groups of 16 rows (KQ_Q4X, ops.q4x_pack_model), with
        # float32 activations: the products of MOE (to 4e-6) in 81 per cent
        # of the time over the 30 layers. Each token of a verify group has
        # the operations of a step: the same bits.
        E = gu_q.shape[0]
        t, top_k = idx.reshape(h.shape[0], -1).shape
        mats = ops.q4x_moe_mats(q4x)
        c.p.keep.append(mats)
        hidden = h.shape[1]
        if ops.DECODE_X16:
            # int16 rows of h and of the GELU (act bit 2), from the float rows h
            c.p.emit(KQ_MOE, None, None, None, idx.reshape(t, top_k), val.reshape(t, top_k), t,
                     top_k, E, mats, None, hidden, inner,
                     cops.kq_moe_scratch(t, top_k, E, hidden, inner), out, None, 1 | 4, h)
            return out
        if ops.Q4X_INT8_DECODE:
            # AVX2: int8 x (KQ_QUANT, then kq_moe with the GELU only)
            hq, hs, hm = k_kq_quant(c, h.reshape(t, hidden))
            c.p.emit(KQ_MOE, hq, hs, hm, idx.reshape(t, top_k), val.reshape(t, top_k), t,
                     top_k, E, mats, None, hidden, inner,
                     cops.kq_moe_scratch(t, top_k, E, hidden, inner), out, None, 1, None)
            return out
        c.p.emit(KQ_MOE, c.buffer((t, hidden), np.int8), c.buffer((t, hidden // 32)),
                 c.buffer((t, hidden // 16)), idx.reshape(t, top_k), val.reshape(t, top_k), t,
                 top_k, E, mats, None, hidden, inner,
                 cops.kq_moe_scratch(t, top_k, E, hidden, inner), out, None, 3, h)
        return out
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
    if _is_kq(mat):
        return k_kq(c, mat, x, out)
    out = c.buffer((x.shape[0], mat.shape[0])) if out is None else out
    c.p.emit(BF16_LINEAR, x, mat, out, mat.shape[0], x.shape[1], x.shape[0])
    return out


class _Q4KQ:
    """A Q4_0 int4 matrix of a Model (the blocks of _load_layer) as a matrix
    of the GGUF products (ggml type 2), for the GPU products with int8 x
    (GP_KQ_LINEAR). The data is a view of the int4 blocks."""

    def __init__(self, w):
        self.data, self.type = w.reshape(-1), 2
        self.rows, self.cols = int(w.shape[0]), int(w.shape[1]) * 32

    def c(self):
        return (self.data, self.type)


# The dense matrices of a layer that a compiler with q4_kq gives as _Q4KQ.
_Q4KQ_KEYS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
              "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")


def kq_e3(type_):
    """True for the type of an EXL3 MUL1 matrix (KQ_E3_BASE + K,
    reclank/formats/exl3.py): its GPU product is a record of its own, not a part of
    a GP_KQ_MULTI."""
    t = type_[1] if isinstance(type_, tuple) else type_
    return isinstance(t, (int, np.integer)) and 70 < int(t) <= 78


def _is_kq(mat):
    """A matrix in the blocks of the GGUF products (E4B.kq)."""
    return hasattr(mat, "c") and hasattr(mat, "rows") and hasattr(mat, "type")


def k_kq_quant(c, x):
    """The int8 rows of x for GP_KQ_LINEAR (a scale for each 32, a sum for
    each 16). The GPU products read x itself, so a GPU drops the record."""
    t, cols = x.shape
    xq = (c.buffer((t, cols), np.int8), c.buffer((t, cols // 32)), c.buffer((t, cols // 16)))
    c.p.emit(KQ_QUANT, x, t, cols, *xq)
    return xq


def k_kq(c, mat, x, out=None, xq=None):
    """A GGUF matrix (E4B.kq) on the rows of x: GP_KQ_LINEAR."""
    t = x.shape[0]
    if xq is None:
        xq = k_kq_quant(c, x)
    out = c.buffer((t, mat.rows)) if out is None else out
    c.p.emit(KQ_LINEAR, *xq, x, mat.data, mat.type, mat.rows, mat.cols, t, out)
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
    slide = 1 if os.environ.get("RECLANK_SLIDE", "1") == "1" else 0
    c.p.emit(ATTN_F32H, q, s("k"), s("v"), c.p.slot("scores"), out, qh, kvh, hd, t,
             c.p.slot("pos"), s("hs"), plan.window, slide)
    return out


# ---- the media encoders (reclank/encoders/runtime.py) ----

def k_enc_linear(c, lin, x, out=None):
    """(enc_linear lin x): a linear of an encoder (gemma4_encoders._Linear):
    the clamps of its input, W (bfloat16 or float32), the bias, the clamps
    of its output. The buffer "_scratch" of the compiler holds the clamped
    input."""
    n = x.shape[0]
    k = x.size // n
    # lin.kq: W in a type of the GGUF records (encoders.runtime.to_bf12: BF12),
    # in a GPU program; lin.q8 with the compiler's q8
    kq = getattr(lin, "kq", None) if not getattr(c, "pack", False) else None
    if kq is None and getattr(c, "q8", False):
        kq = getattr(lin, "q8", None)
    m = lin.w.shape[0] if kq is None else kq.rows
    out = c.buffer((n, m)) if out is None else out
    # FP4-Local's vision (models/qwen/vision.py): the input of the linear in bfloat16 values or as
    # dynamic MXFP8 (in place: the inputs of the encoder's linears are buffers of their own), the
    # output rounded to bfloat16
    act = getattr(lin, "act", None)
    if act == "bf16":
        c.p.emit(BF16_ROUND, x, n * k)
    elif act == "mx8":
        c.p.emit(MX8_ACT, x, n * k)
    if kq is not None:
        out = _k_enc_linear_kq(c, lin, kq, x, n, m, k, out)
        if act is not None:
            c.p.emit(BF16_ROUND, out, n * m)
        return out
    wbf = 1 if lin.w.dtype == np.uint16 else 0
    inf = float("inf")
    scratch = None
    if lin.imin is not None:
        scratch = c.env["_scratch"]
        assert scratch.size >= n * k, "the scratch of enc_linear is too small"
    x16 = None
    if getattr(c, "pack", False) and os.environ.get("RECLANK_ENC_X16", "1") != "0":
        # A CPU program: W in groups of 16 rows (kq_x16f_body), made once.
        x16 = getattr(lin, "x16", None)
        if x16 is None:
            x16 = lin.x16 = cops.kq_pack_x16f(lin.w, wbf, m, k)
    c.p.emit(ENC_LINEAR, x, lin.w, wbf, lin.b, out, n, m, k,
             -inf if lin.imin is None else float(lin.imin),
             inf if lin.imax is None else float(lin.imax),
             -inf if lin.omin is None else float(lin.omin),
             inf if lin.omax is None else float(lin.omax), scratch, x16)
    if act is not None:
        c.p.emit(BF16_ROUND, out, n * m)
    return out


def _k_enc_linear_kq(c, lin, q8, x, n, m, k, out):
    """enc_linear with the weights of a GGUF type: Q8_0 (lin.q8, and
    lin.q8x16 for a CPU with VNNI) or BF12 (lin.kq, on the GPU): the input
    clamps (ENC_CLAMP), then the product of the GGUF records (KQ_QUANT and
    KQ_LINEAR; the GPU quantizes x itself), then the bias and the output
    clamps (ENC_BIAS_CLAMP). The int8 rows of x share the buffers "_xq",
    "_xs", and "_xm" of the compiler."""
    inf = float("inf")
    xin = x
    if lin.imin is not None:
        # The whole scratch array, not a view: the mirror of a GPU program
        # makes a device copy for each array of the program, so a view at
        # the address of the scratch had a device copy of its own.
        xin = c.env["_scratch"]
        c.p.emit(ENC_CLAMP, x, xin, n * k, float(lin.imin), float(lin.imax))
    pack = getattr(c, "pack", False)
    mat = getattr(lin, "q8x16", None) if pack else None
    mat = mat or q8
    if pack:
        xq = (c.env["_xq"], c.env["_xs"], c.env["_xm"])     # whole arrays, as the scratch
        c.p.emit(KQ_QUANT, xin, n, k, *xq)
    else:
        xq = (None, None, None)
    c.p.emit(KQ_LINEAR, *xq, xin, mat.data, mat.type, mat.rows, mat.cols, n, out)
    if lin.b is not None or lin.omin is not None:
        c.p.emit(ENC_BIAS_CLAMP, out, lin.b, n, m,
                 -inf if lin.omin is None else float(lin.omin),
                 inf if lin.omax is None else float(lin.omax))
    return out


def k_enc_rms(c, x, w, cols, out=None):
    """(enc_rms x w cols): the RMS norm of each row of cols values (w None:
    no weight)."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(ENC_RMS, x, w, out, x.size // cols, cols, float(c.eps))
    return out


def k_enc_lnorm(c, x, w, b, cols, out=None):
    """(enc_lnorm x w b cols): the LayerNorm of each row of cols values (w
    and b may be None)."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(ENC_LNORM, x, w, b, out, x.size // cols, cols, float(c.eps))
    return out


def k_enc_gelu(c, x, erf=False, out=None):
    """(enc_gelu x [erf]): gelu(x), the tanh form or the erf form."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(ENC_GELU, x, out, x.size, 1 if erf else 0)
    return out


def k_enc_gelu_mul(c, g, u, out=None):
    """(enc_gelu_mul g u): gelu_tanh(g) u."""
    out = c.buffer(g.shape) if out is None else out
    c.p.emit(ENC_GELU_MUL, g, u, out, g.size)
    return out


def k_enc_add(c, x, y, s=1.0):
    """(enc_add x y [s]): x += s y in place. Return x."""
    c.p.emit(ENC_ADD, x, y, x.size, float(s))
    return x


def k_enc_silu(c, x, out=None):
    """(enc_silu x): x sigmoid(x)."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(ENC_SILU, x, out, x.size)
    return out


def k_enc_silu_mul(c, g, u, out=None):
    """(enc_silu_mul g u): silu(g) u."""
    out = c.buffer(g.shape) if out is None else out
    c.p.emit(ENC_SILU_MUL, g, u, out, g.size)
    return out


def k_enc_mul_vec(c, x, vec, out=None):
    """(enc_mul_vec x vec): each row of x times vec."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(ENC_MUL_VEC, x, vec, out, x.size // vec.size, vec.size)
    return out


def k_enc_glu(c, x, out=None):
    """(enc_glu x): a sigmoid(b) of the halves a, b of each row."""
    rows, cols = x.shape[0], x.size // x.shape[0] // 2
    out = c.buffer((rows, cols)) if out is None else out
    c.p.emit(ENC_GLU, x, out, rows, cols)
    return out


def k_enc_dwconv(c, x, w, out=None):
    """(enc_dwconv x w): the causal depthwise conv of the rows of x with w
    (channels, kernel)."""
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(ENC_DWCONV, x, w, out, x.shape[0], w.shape[0], w.shape[1])
    return out


def k_enc_local_attn(c, q, k, v, r, valid, heads, hd, span, cap, out=None):
    """(enc_local_attn q k v r valid heads hd span cap): the local attention
    of gemma4a (Gemma4Audio._attention)."""
    out = c.buffer(q.shape) if out is None else out
    c.p.emit(ENC_LOCAL_ATTN, q, k, v, r, valid, out, q.shape[0], heads, hd, span, float(cap))
    return out


def k_enc_rope2d(c, x, pos, inv, heads, hd):
    """(enc_rope2d x pos inv heads hd): the axial 2D RoPE of gemma4v in
    place. Return x."""
    c.p.emit(ENC_ROPE2D, x, pos, inv, x.shape[0], heads, hd)
    return x


def k_enc_attn(c, q, k, v, heads, hd, out=None):
    """(enc_attn q k v heads hd): the attention of every query over every
    key, scale 1."""
    n = q.shape[0]
    out = c.buffer(q.shape) if out is None else out
    c.p.emit(ENC_ATTN, q, k, v, out, n, heads, hd, np.full(n, n - 1, np.int32))
    return out


KERNELS = {
    "enc_linear": k_enc_linear,
    "enc_rms": k_enc_rms,
    "enc_gelu_mul": k_enc_gelu_mul,
    "enc_lnorm": k_enc_lnorm,
    "enc_gelu": k_enc_gelu,
    "enc_add": k_enc_add,
    "enc_rope2d": k_enc_rope2d,
    "enc_attn": k_enc_attn,
    "enc_silu": k_enc_silu,
    "enc_silu_mul": k_enc_silu_mul,
    "enc_mul_vec": k_enc_mul_vec,
    "enc_glu": k_enc_glu,
    "enc_dwconv": k_enc_dwconv,
    "enc_local_attn": k_enc_local_attn,
    "rms_norm": k_rms_norm,
    "add": k_add,
    "mul": k_mul,
    "ffn_out": k_ffn_out,
    "copy": k_copy,
    "int4_multi4": k_int4_multi4,
    "rms_norm_multi4": k_rms_norm_multi4,
    "gelu_mul_int4": k_gelu_mul_int4,
    "int4": k_int4,
    "qkv_norm_rope": k_qkv_norm_rope,
    "kv_write": k_kv_write,
    "tq_rot": lambda c, x, inverse: k_tq_rot(c, x, inverse),
    "attn_qc": k_attn_qc,
    "attn_f32": k_attn_f32,
    # The mode is part of the name. A bare string operand is a name for the
    # compiler, so the form cannot give the mode as an operand.
    "attn_rows_qc": lambda c, layer, q: k_attn_rows(c, layer, q, "qc"),
    "attn_q8": lambda c, layer, q, lo, n: k_attn_qc(c, layer, q, lo, n, q8=True),
    "attn_rows_q8": lambda c, layer, q: k_attn_rows(c, layer, q, "q8"),
    "attn_qv": lambda c, layer, q, lo, n: k_attn_qc(c, layer, q, lo, n, q8="v"),
    "attn_rows_qv": lambda c, layer, q: k_attn_rows(c, layer, q, "qv"),
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
