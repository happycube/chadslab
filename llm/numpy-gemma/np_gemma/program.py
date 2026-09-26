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
the state of a step is part of the program, and dump() shows its values.

run_py() runs the same records in Python. It calls the C entry points of
today, one at a time, so a check can compare the two after each record.
"""
from __future__ import annotations

import ctypes

import numpy as np

from . import cops, ops

NARG = 24
MAGIC = 0x4750524F47303031

T_NONE, T_INT, T_F32, T_SLOT = 0, 1, 2, 3

S_MOV, S_ADD, S_SUB, S_MUL, S_MAX, S_MIN = 1, 2, 3, 4, 5, 6
RMS_NORM, ADD, MUL_S, COPY = 16, 17, 18, 19
INT4_LINEAR, INT4_MULTI4, RMS_NORM_MULTI4, GELU_MUL_INT4 = 32, 33, 34, 35
QKV_NORM_ROPE, KV_WRITE, ATTN_Q8 = 48, 49, 50
ROUTER, MOE = 64, 65

OP_NAMES = {v: k for k, v in dict(
    S_MOV=S_MOV, S_ADD=S_ADD, S_SUB=S_SUB, S_MUL=S_MUL, S_MAX=S_MAX, S_MIN=S_MIN,
    RMS_NORM=RMS_NORM, ADD=ADD, MUL_S=MUL_S, COPY=COPY, INT4_LINEAR=INT4_LINEAR,
    INT4_MULTI4=INT4_MULTI4, RMS_NORM_MULTI4=RMS_NORM_MULTI4,
    GELU_MUL_INT4=GELU_MUL_INT4, QKV_NORM_ROPE=QKV_NORM_ROPE, KV_WRITE=KV_WRITE,
    ATTN_Q8=ATTN_Q8, ROUTER=ROUTER, MOE=MOE).items()}

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
    return int(np.array([x], dtype=np.float32).view(np.uint32)[0])


def _bits_f32(b):
    return float(np.array([b & 0xFFFFFFFF], dtype=np.uint32).view(np.float32)[0])


class Program:
    """A list of records and an environment. See the module text."""

    def __init__(self):
        self.slots = []            # Slot objects, in order
        self.by_name = {}
        self.init = []             # the first value of each slot
        self.recs = []             # (op, [(tag, value), ...])
        self.keep = []             # the arrays that the literals point to
        self.bound = {}            # the arrays that a bind points to
        self.buf = None
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
        return self.slot("t%d" % len(self.slots))

    def _enc(self, v):
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
        assert len(args) <= NARG
        self.recs.append((op, [self._enc(a) for a in args]))

    def finish(self):
        """Make the int64 array of the program."""
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
        """Write parameters. A value is an int, a float, or an array."""
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
    tag, v = args[k]
    return e[v] if tag == T_SLOT else v


def _f(args, e, k):
    return _bits_f32(_val(args, e, k))


def _arr(addr, n, ctype=ctypes.c_float):
    return np.ctypeslib.as_array((ctype * n).from_address(addr))


def _py_step(op, a, e):
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
        kq, ks = ops.quantize_q8(k.reshape(-1, 32))
        vq, vs = ops.quantize_q8(v.reshape(-1, 32))
        _arr(V(4), n, ctypes.c_int8)[:] = kq.reshape(-1)
        _arr(V(5), n // 32)[:] = ks.reshape(-1)
        _arr(V(6), n, ctypes.c_int8)[:] = vq.reshape(-1)
        _arr(V(7), n // 32)[:] = vs.reshape(-1)
    elif op == ATTN_Q8:
        qh, kvh, hd, n = V(9), V(10), V(11), V(12)
        q = _arr(V(0), qh * hd)
        o = ops.attn_decode(q.reshape(qh, hd), _arr(V(3), n * kvh * hd, ctypes.c_int8),
                            _arr(V(4), n * kvh * hd // 32), _arr(V(5), n * kvh * hd, ctypes.c_int8),
                            _arr(V(6), n * kvh * hd // 32), qh, kvh, hd, n)
        _arr(V(8), qh * hd)[:] = o.reshape(-1)
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
    else:
        raise ValueError("op %d" % op)


# ---- the compiler --------------------------------------------------------------

class Compiler:
    """Turn the expressions of a model into the records of a Program."""

    def __init__(self, model, prog=None):
        self.model = model
        self.cfg = model.cfg
        self.eps = model.cfg.rms_norm_eps
        self.p = prog or Program()
        self.env = self.p.names

    # A symbol is a name of the environment of the compiler, or a parameter.
    def value(self, x):
        if isinstance(x, str):
            if x in self.env:
                return self.env[x]
            return self.p.slot(x)
        if isinstance(x, tuple):
            return self.expr(x)
        return x

    def buffer(self, shape, dtype=np.float32):
        return np.zeros(shape, dtype=dtype)

    def compile(self, form):
        head = form[0]
        if head in ("seq", "layer"):
            body = form[2:] if head == "layer" else form[1:]
            for f in body:
                self.compile(f)
            return None
        return self.expr(form)

    def expr(self, form):
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
            return self.model._layers[layer][key]
        if head in SCALAR:
            return self.scalar(head, [self.value(a) for a in args])
        fn = KERNELS[head]
        return fn(self, *[self.value(a) for a in args])

    def expr_into(self, form, out):
        """Compile a kernel expression that writes an existing buffer."""
        fn = KERNELS[form[0]]
        return fn(self, *[self.value(a) for a in form[1:]], out=out)

    def scalar(self, head, vals):
        # Fold the constants in Python. A slot operand makes a record.
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


SCALAR = {
    "+": (S_ADD, lambda a, b: a + b),
    "-": (S_SUB, lambda a, b: a - b),
    "*": (S_MUL, lambda a, b: a * b),
    "max": (S_MAX, max),
    "min": (S_MIN, min),
}


# ---- the kernel operations ------------------------------------------------------
# Each function takes the compiler and the values of its operands. It emits the
# records and returns the result.

def k_rms_norm(c, x, w, out=None):
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(RMS_NORM, x, w, out, x.shape[0], x.shape[1], float(c.eps))
    return out


def k_add(c, a, b, out=None):
    out = c.buffer(a.shape) if out is None else out
    c.p.emit(ADD, a, b, out, a.size)
    return out


def k_mul(c, x, s, out=None):
    out = c.buffer(x.shape) if out is None else out
    c.p.emit(MUL_S, x, float(s), out, x.size)
    return out


def k_copy(c, x, out=None):
    out = c.buffer(x.shape, x.dtype) if out is None else out
    c.p.emit(COPY, x, out, x.nbytes)
    return out


def _mats(mats):
    args, outs = [], []
    for m in range(4):
        if m < len(mats) and mats[m] is not None:
            w, s = mats[m]
            o = np.zeros((1, w.shape[0]), dtype=np.float32)
            args += [w, s, o, w.shape[0]]
            outs.append(o)
        else:
            args += [None, None, None, 0]
    return args, outs


def k_int4_multi4(c, x, *mats):
    args, outs = _mats(mats)
    c.p.emit(INT4_MULTI4, x, x.shape[1], *args)
    return tuple(outs)


def k_rms_norm_multi4(c, x, wn, *mats):
    args, outs = _mats(mats)
    scratch = c.buffer(x.shape[1])
    c.p.emit(RMS_NORM_MULTI4, x, np.ascontiguousarray(wn, dtype=np.float32), scratch,
             x.shape[1], float(c.eps), *args)
    return tuple(outs)


def k_gelu_mul_int4(c, g, u, mat, out=None):
    w, s = mat
    rows, cols = w.shape[0], g.size
    out = c.buffer((1, rows)) if out is None else out
    c.p.emit(GELU_MUL_INT4, g, u, g.size, c.buffer(g.size), w, s, out, rows, cols)
    return out


def k_int4(c, mat, x, out=None):
    w, s = mat
    out = c.buffer((1, w.shape[0])) if out is None else out
    c.p.emit(INT4_LINEAR, x, w, s, out, w.shape[0], x.shape[1])
    return out


def k_qkv_norm_rope(c, q, k, v, qn, kn, cos, sin, layer):
    plan = c.cfg.plan[layer]
    hd = plan.head_dim
    c.p.emit(QKV_NORM_ROPE, q, np.ascontiguousarray(qn, dtype=np.float32), q.size // hd,
             k, np.ascontiguousarray(kn, dtype=np.float32), k.size // hd, v, v.size // hd,
             cos, sin, plan.num_q_heads, plan.num_kv_heads, hd, float(c.eps))


def _addr(c, base, row, stride):
    """Return base + row * stride, folded when both are constants."""
    return c.scalar("+", [base, c.scalar("*", [row, stride])])


def k_kv_write(c, layer, k, v, row):
    plan = c.cfg.plan[layer]
    n = plan.num_kv_heads * plan.head_dim
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    c.p.emit(KV_WRITE, k, v,
             _addr(c, s("k"), row, 4 * n), _addr(c, s("v"), row, 4 * n),
             _addr(c, s("kq"), row, n), _addr(c, s("ks"), row, 4 * (n // 32)),
             _addr(c, s("vq"), row, n), _addr(c, s("vs"), row, 4 * (n // 32)), n)


def k_attn_q8(c, layer, q, lo, n):
    plan = c.cfg.plan[layer]
    hd, qh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
    per = kvh * hd
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    out = c.buffer((1, qh * hd))
    c.p.emit(ATTN_Q8, q, c.buffer(qh * hd, np.int8), c.buffer(qh * hd // 32),
             _addr(c, s("kq"), lo, per), _addr(c, s("ks"), lo, 4 * (per // 32)),
             _addr(c, s("vq"), lo, per), _addr(c, s("vs"), lo, 4 * (per // 32)),
             c.p.slot("scores"), out, qh, kvh, hd, n)
    return out


def k_router(c, x, layer):
    w = c.model._layers[layer]
    cfg = c.cfg
    top_k = cfg.top_k_experts
    val = c.buffer(top_k)
    idx = np.zeros(top_k, dtype=np.int32)
    proj = np.ascontiguousarray(w["router.proj"], dtype=np.float32)
    c.p.emit(ROUTER, x, np.ascontiguousarray(w["router.scale"], dtype=np.float32), proj,
             np.ascontiguousarray(w["router.per_expert_scale"], dtype=np.float32),
             x.shape[1], proj.shape[0], top_k, float(c.eps),
             float(cfg.hidden_size ** -0.5), val, idx,
             c.buffer(x.shape[1]), c.buffer(proj.shape[0]))
    return val, idx


def k_moe(c, h, val, idx, layer):
    w = c.model._layers[layer]
    gu_q, gu_s = w["experts.gate_up_proj"]
    dn_q, dn_s = w["experts.down_proj"]
    inner = c.cfg.moe_intermediate_size
    top_k = idx.size
    out = c.buffer(h.shape)
    c.p.emit(MOE, h, val, idx, top_k, gu_q, gu_s, dn_q, dn_s, gu_q.shape[1], h.shape[1],
             dn_q.shape[1], inner, np.zeros(top_k, dtype=np.int32),
             c.buffer((top_k, 2 * inner)), c.buffer((top_k, inner)),
             c.buffer((top_k, dn_q.shape[1])), out)
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
    "attn_q8": k_attn_q8,
    "router": k_router,
    "moe": k_moe,
}


# ---- the forms of the 26B model -------------------------------------------------

def layer_form(model, i):
    """Return one decoder layer of the 26B model as a nested expression.

    The expression follows Model._decoder_layer and Model._attention for one
    token with the int8 cache. x is the hidden state; the layer changes it in
    place.
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
    return ("layer", i,
            ("let", "h", ("rms_norm", "x", w("input_layernorm"))),
            *qkv,
            ("qkv_norm_rope", "q", "k", "v", w("self_attn.q_norm"), w("self_attn.k_norm"),
             "cos." + kind, "sin." + kind, i),
            ("let", "row", ("-", "pos", base)),
            ("kv_write", i, "k", "v", "row"),
            ("let", "lo", lo),
            ("let", "n", ("-", ("+", "pos", 1), base, "lo")),
            ("let", "a", ("attn_q8", i, "q", "lo", "n")),
            ("let", "o", ("int4", w("self_attn.o_proj"), "a")),
            ("set", "x", ("add", "x", ("rms_norm", "o", w("post_attention_layernorm")))),
            ("let", ("g", "u"), ("rms_norm_multi4", "x", w("pre_feedforward_layernorm"),
                                 w("mlp.gate_proj"), w("mlp.up_proj"))),
            ("let", "m", ("gelu_mul_int4", "g", "u", w("mlp.down_proj"))),
            ("let", ("val", "idx"), ("router", "x", i)),
            ("let", "e", ("moe", ("rms_norm", "x", w("pre_feedforward_layernorm_2")),
                          "val", "idx", i)),
            ("let", "f", ("add", ("rms_norm", "m", w("post_feedforward_layernorm_1")),
                          ("rms_norm", "e", w("post_feedforward_layernorm_2")))),
            ("set", "x", ("add", "x", ("rms_norm", "f", w("post_feedforward_layernorm")))),
            ("set", "x", ("mul", "x", w("layer_scalar"))))


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


def compile_layers(model, layers):
    """Compile decoder layers into one Program. The buffer "x" is the input
    and the output."""
    c = Compiler(model)
    c.env["x"] = np.zeros((1, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    for i in layers:
        c.compile(layer_form(model, i))
    c.p.layers = list(layers)
    return c.p.finish()


def bind_step(prog, model, cache, pos):
    """Prepare the cache for one token at pos and bind the parameters.

    The cache work stays in Python: KVCache.prepare drops old rows and grows
    a buffer. The program then writes the new rows in C. The int8 copy of the
    cache must be on for every layer of the program.
    """
    kw = {"pos": pos}
    keep = []
    for i in prog.layers:
        assert cache.q8_ready(i), "the program needs the int8 cache"
        cache.prepare(i, pos, 1)
        cache.end[i] = pos + 1
        kw.update({"base.%d" % i: cache.base[i], "k.%d" % i: cache.k[i],
                   "v.%d" % i: cache.v[i], "kq.%d" % i: cache.kq[i],
                   "ks.%d" % i: cache.ks[i], "vq.%d" % i: cache.vq[i],
                   "vs.%d" % i: cache.vs[i]})
    positions = np.array([pos])
    for kind, sliding in (("s", True), ("f", False)):
        plan = next((model.cfg.plan[i] for i in prog.layers
                     if model.cfg.plan[i].is_sliding == sliding), None)
        if plan is None:
            continue
        cos, sin, _ca, _sa = model._rope(plan, positions)
        keep += [cos, sin]
        kw["cos." + kind] = cos
        kw["sin." + kind] = sin
    # The scores of the widest attention: all query heads over all keys.
    need = max(model.cfg.plan[i].num_q_heads * (pos + 1) for i in prog.layers)
    sc = getattr(prog, "scores", None)
    if sc is None or sc.size < need:
        sc = np.zeros(max(need, 2 * (sc.size if sc is not None else 0)), dtype=np.float32)
        prog.scores = sc
    kw["scores"] = sc
    prog.bind(**kw)
    prog.keep_bound = keep
