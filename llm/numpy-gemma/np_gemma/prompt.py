"""A block of a prompt as one program (np_gemma/program.py).

The Python path of a prompt block (Model.forward, Model._decoder_layer) calls
about 20 kernels for each layer from Python, and NumPy adds the residual
rows on one thread. Here the compiler makes one program of the whole block
with the step form of t tokens (program.step_form), and the records of the
team do the work. Each operation has the kernel of the Python path, so the
result has its bits:

    the products          KQ_QUANT and KQ_LINEAR on the KQ_Q4X copies
                          (ops._q4x_linear: kq_quant_x, kq_linear)
    the norms             RMS_NORM (gemma_rms_norm_body over the team)
    gelu(g) * u           GELU, then MUL (ops.gelu_tanh, then the NumPy product)
    the residual, scale   ADD, MUL_S (one add, one product for each value)
    the attention         ATTN_PREFILL_QC (ops.flash_prefill_qc on the int16
                          cache)
    the cache, the norms
    of q and k, the rope  the records of the steps (KV_WRITE, QKV_NORM_ROPE)

The buffers of a layer are the buffers of the layer before (the compiler
gives them in the same order); only x goes from a layer to the next, and the
forms write it in place. A program of 256 tokens of the 12B then holds about
120 MB of buffers, not one set for each of the 48 layers.

A model with experts (the 26B) adds two records:

    the router            ROUTER_MT: the fused router of a step, for each
                          token of the block
    the experts           KQ_QUANT, then KQ_MOE with the GELU (ops._q4x_moe:
                          kq_quant_x, kq_moe_act), int8 x

The router of the Python path for a block of the prompt is a float32 matrix
product (BLAS) and the NumPy softmax, whose sums a record cannot give
again. The program takes the router of the steps in its place: each token
of a prompt selects the experts that a step of that token selects. So the
prompt of the 26B has the bits of the Python path with ops.router_mt as its
router, not of the Python path (a router of another order of sums can
select another expert where two are near).

With Model.prompt_act "16" (the default of the 26B, or NP_GEMMA_INT4_Q8=16
for every model; ops.prompt_act) the products take int16 x
in place of int8 (a scale for each 32, gemma_quant_group32_i16): KQ_QUANT16
and KQ_LINEAR16 (kq_q4x_gemm16, vpdpwssd) for the matrices, and KQ_MOE with
the int16 rows of h and of the GELU (act bit 2, kq_q4x_rows16) for the
experts. A product is then 2e-5 of its size off float32, not 5e-3, and the
26B gives the NLL of float32 (SPLIT_PLAN.md). The products take about 2.2
times the time of int8: the prompt of the 26B about 1.1 times (its experts
are a small part of the time), that of the 12B about 2 times. The Python path of prompt_act "16" takes float32
products for the matrices and the int16 tile for the experts, so the two
do not have the same bits.

prompt_ready says when the program serves a block: int4 with the KQ_Q4X
copies (and those of the experts), int8 or int16 x, a KVCache of int16,
the C flash kernel, and no hook. A block with media (the soft rows of an
image or a clip) takes them as the Python path does (prompt_step).
NP_GEMMA_PROMPT_PROGRAM=0 keeps the Python path.

With a PartKVCache (NP_GEMMA_PARTS), the program writes and reads the cache
of each part: for each part in turn, one PART_PREFILL record (the team
writes the rows of the heads of the part in its buffers, on its node, and
runs the attention of its query heads), as PartKVCache.prefill_attention,
so the same bits. The products stay in one team: the prompt kernel
(kq_q4x_gemm) gains 1.7 to 1.9 times from the second node with the x of a
block in L2, so a split of the rows over the parts would add barriers for
little.
"""
from __future__ import annotations

import os
from collections import defaultdict

import numpy as np

from . import cops
from . import program as P


class PromptCompiler(P.Compiler):
    """The compiler of a prompt block: the kernels of the Python path of a
    prompt (PROMPT_KERNELS), and the buffers of each layer used again."""

    def __init__(self, model, cache=None):
        super().__init__(model)
        self._pool = defaultdict(list)       # (shape, dtype) -> buffers
        self._used = defaultdict(int)        # (shape, dtype) -> buffers taken in this layer
        # A PartKVCache: its heads and its count of parts; the parameters of
        # each PART_PREFILL record (layer, part, int32 array) for the bind.
        self.split = cache if getattr(cache, "split", False) else None
        self.part_params = []
        self.moe_scratch = {}                # the scratch of kq_moe for each shape
        self.q16 = model.prompt_act == "16"  # int16 x for the products (NP_GEMMA_INT4_Q8=16)
        # NP_GEMMA_Q16_PARTS (a test): "dense" or "moe" keeps int16 x for
        # those products only, int8 for the others
        parts = os.environ.get("NP_GEMMA_Q16_PARTS", "all")
        self.q16_dense = self.q16 and parts in ("all", "dense")
        self.q16_moe = self.q16 and parts in ("all", "moe")

    def buffer(self, shape, dtype=np.float32):
        key = (tuple(int(v) for v in np.atleast_1d(shape)), np.dtype(dtype).str)
        i = self._used[key]
        self._used[key] += 1
        pool = self._pool[key]
        if i == len(pool):
            pool.append(np.zeros(key[0], dtype=dtype))
        return pool[i]

    def compile(self, form):
        if form[0] == "layer":
            # A layer takes the buffers of the layer before, in the same order.
            self._used = defaultdict(int)
        return super().compile(form)

    def kernel(self, head, vals, out=None):
        fn = PROMPT_KERNELS.get(head)
        if fn is None:
            return super().kernel(head, vals, out)
        assert out is None, "a kernel of the prompt does not write an existing buffer"
        return fn(self, *vals)


def _q4x(mat):
    """The KQ_Q4X copy of an int4 matrix (w: rows x blocks x 18), its rows
    and its columns."""
    w, _s = mat
    qx = P.ops._Q4X.get(w.ctypes.data)
    if qx is None:
        raise ValueError("the prompt program needs the KQ_Q4X copies (NP_GEMMA_Q4X)")
    return qx, w.shape[0], w.shape[1] * 32


def _quant(c, x):
    """The quantized rows of x for the products: int8 (KQ_QUANT: xq, xs, xm)
    or, with c.q16, int16 (KQ_QUANT16: xq, xs)."""
    if not c.q16_dense:
        return P.k_kq_quant(c, x)
    t, cols = x.shape
    xq = (c.buffer((t, cols), np.int16), c.buffer((t, cols // 32)))
    c.p.emit(P.KQ_QUANT16, x, t, cols, *xq)
    return xq


def _lin(c, mat, x, xq):
    """KQ_LINEAR (int8) or KQ_LINEAR16 (int16) of the KQ_Q4X matrix on the
    quantized rows xq of x."""
    qx, rows, cols = _q4x(mat)
    t = x.shape[0]
    out = c.buffer((t, rows))
    if c.q16_dense:
        c.p.emit(P.KQ_LINEAR16, qx, rows, cols, *xq, t, out)
    else:
        c.p.emit(P.KQ_LINEAR, *xq, x, qx, cops.KQ_Q4X, rows, cols, t, out)
    return out


def pk_int4(c, mat, x):
    """The output projection: Model.linear (ops._q4x_linear)."""
    return _lin(c, mat, x, _quant(c, x))


def pk_int4_multi4(c, x, *mats):
    """The query, the key, and the value: one quantized x for the products."""
    xq = _quant(c, x)
    return tuple(_lin(c, m, x, xq) for m in mats if m is not None)


def pk_rms_norm_multi4(c, x, wn, *mats):
    """The norm before the feed-forward part, then the gate and the up map."""
    h = P.k_rms_norm(c, x, np.ascontiguousarray(wn, dtype=np.float32))
    xq = _quant(c, h)
    return tuple(_lin(c, m, h, xq) for m in mats if m is not None)


def pk_gelu_mul_int4(c, g, u, mat):
    """gelu(g) * u (ops.gelu_tanh, then the product of NumPy), then the down
    map."""
    t, inner = g.shape
    gg = c.buffer(g.shape)
    c.p.emit(P.GELU, g, gg, g.size)
    m = c.buffer(g.shape)
    c.p.emit(P.MUL, gg, u, m, t, inner, inner)
    return _lin(c, mat, m, _quant(c, m))


def pk_kv_write(c, layer, k, v, row, qc=1):
    """The write of the rows of the block: KV_WRITE for one KVCache; for a
    PartKVCache the PART_PREFILL records write the rows of each part."""
    if c.split is None:
        return P.k_kv_write(c, layer, k, v, row, qc)
    return None


def _attn_parts(c, layer, q):
    """The attention of the block with a PartKVCache: one PART_PREFILL record
    for each part, on the buffers of the part (slots pkq.L.p and so on) and
    the int32 parameters that prompt_step fills (the buffer rows)."""
    plan = c.cfg.plan[layer]
    t = q.shape[0]
    qh, kvh, hd = plan.num_q_heads, plan.num_kv_heads, plan.head_dim
    k, v = c.env["k"], c.env["v"]
    out = c.buffer((t, qh * hd))
    for p, ((g0, g1), (h0, h1)) in enumerate(zip(c.split.heads[layer], c.split.qheads[layer])):
        if g1 == g0 or h1 == h0:
            continue
        ip = np.zeros(12, dtype=np.int32)
        ip[[0, 4, 5, 6, 7, 8, 9, 10, 11]] = [t, plan.sliding_window or 0, qh, kvh, hd,
                                            g0, g1, h0, h1]
        c.p.keep.append(ip)
        c.part_params.append((layer, p, ip))
        slot = lambda name: c.p.slot("p%s.%d.%d" % (name, layer, p))  # noqa: E731
        c.p.emit(P.PART_PREFILL, q, k, v, out, slot("kq"), slot("ks"), slot("vq"), slot("vs"),
                 None, None, c.buffer((t, (h1 - h0) * hd)), c.buffer((t, (h1 - h0) * hd)),
                 c.p.slot("positions"), ip, _limit(c, plan))
    return out


def _limit(c, plan):
    """The slot of the last key of each query (media: the tokens of an image
    see each other; 0, no limit, without media), for the layers where
    Model._media_limit applies: the sliding ones, and every one with
    NP_GEMMA_BIDIR_ALL."""
    from .model import _BIDIR_ALL
    return c.p.slot("limit") if (plan.sliding_window or _BIDIR_ALL) else None


def pk_attn_rows_qc(c, layer, q):
    """The attention of the block over the int16 cache of the layer, after
    the write of its rows (ops.flash_prefill_qc)."""
    if c.split is not None:
        return _attn_parts(c, layer, q)
    plan = c.cfg.plan[layer]
    t = q.shape[0]
    s = lambda name: c.p.slot("%s.%d" % (name, layer))  # noqa: E731
    base = s("base")
    n = c.scalar("-", [c.scalar("+", [c.p.slot("pos"), t]), base])
    out = c.buffer((t, plan.num_q_heads * plan.head_dim))
    c.p.emit(P.ATTN_PREFILL_QC, q, s("kq"), s("ks"), s("vq"), s("vs"), c.p.slot("positions"),
             base, plan.sliding_window or 0, out, t, n, plan.num_q_heads, plan.num_kv_heads,
             plan.head_dim, _limit(c, plan))
    return out


def pk_moe(c, h, val, idx, layer):
    """The experts of the block (ops._q4x_moe): the int8 x of h, then kq_moe
    with the GELU of the gate, on the KQ_Q4X copies of the experts."""
    w = c.model._layers[layer]
    gu_q = w["experts.gate_up_proj"][0]
    e = P.ops._Q4X_MOE.get(gu_q.ctypes.data)
    if e is None:
        raise ValueError("the prompt program needs the KQ_Q4X copies of the experts")
    t, hidden = h.shape
    top_k = idx.reshape(t, -1).shape[1]
    experts, inner = gu_q.shape[0], c.cfg.moe_intermediate_size
    mats = P.ops.q4x_moe_mats(e)
    c.p.keep.append(mats)
    key = (t, top_k, experts, hidden, inner)
    sc = c.moe_scratch.get(key)
    if sc is None:
        sc = c.moe_scratch[key] = cops.kq_moe_scratch(t, top_k, experts, hidden, inner)
    out = c.buffer((t, hidden))
    if c.q16_moe:
        # int16 rows of h and of the GELU (act bit 2), from the float rows h
        c.p.emit(P.KQ_MOE, None, None, None, idx.reshape(t, top_k), val.reshape(t, top_k), t,
                 top_k, experts, mats, None, hidden, inner, sc, out, None, 1 | 4, h)
        return out
    hq, hs, hm = P.k_kq_quant(c, h)
    c.p.emit(P.KQ_MOE, hq, hs, hm, idx.reshape(t, top_k), val.reshape(t, top_k), t, top_k,
             experts, mats, None, hidden, inner, sc, out, None, 1, None)
    return out


PROMPT_KERNELS = {
    "int4": pk_int4,
    "int4_multi4": pk_int4_multi4,
    "rms_norm_multi4": pk_rms_norm_multi4,
    "gelu_mul_int4": pk_gelu_mul_int4,
    "attn_rows_qc": pk_attn_rows_qc,
    "kv_write": pk_kv_write,
    "moe": pk_moe,
}


def compile_prompt(model, attn, t, cache=None):
    """Compile a prompt block of t tokens. "x" is the input embedding and
    "xn" the hidden state after the final norm. cache: a PartKVCache gives
    the attention of the parts (its heads)."""
    c = PromptCompiler(model, cache)
    c.env["x"] = np.zeros((t, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    c.p.slot("positions")
    c.p.slot("scores")          # bind_step binds it; the attention keeps its own scratch
    c.p.slot("limit")           # media: the last key of each query (prompt_step)
    c.compile(P.step_form(model, attn, t))
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = attn
    c.p.tokens = t
    c.p.part_params = c.part_params
    return c.p.finish()


def prompt_ready(model, cache, t):
    """True when a program serves a prompt block of t tokens of model with
    cache (see the module text)."""
    from .model import KVCache
    if os.environ.get("NP_GEMMA_PROMPT_PROGRAM", "1") == "0" or t < 2:
        return False
    if model._dtype != "int4" or not model.keep_weights:
        return False
    if model.cfg.enable_moe_block and not (P.ops._Q4X_MOE and P.ops.router_ready()):
        return False
    if not P.ops._Q4X or model.prompt_act not in ("1", "16"):
        return False
    if model.prompt_act == "16" and not cops.kq_q16_ok():
        return False
    if os.environ.get("NP_GEMMA_FLASH", "1") != "1" or not P.ops.flash_ready():
        return False
    if not isinstance(cache, KVCache):
        return False
    if cache.kdtype != np.int16 or cache.vdtype != np.int16:
        return False
    plan = model.cfg.plan
    return all(cops._lib.gemma_attn_prefill_qc_ok(p.num_q_heads, p.num_kv_heads, p.head_dim)
               for p in plan)


def prompt_step(model, cache, tokens, pos, media=()):
    """Run a prompt block of the tokens at pos with a program. Return the
    hidden state after the final norm, (tokens, hidden).

    media: the spans of soft tokens of the block (Model._media_in), as the
    Python path: a soft token takes the row of its image or clip, and the
    tokens of a bidirectional span see each other (Model._media_limit, the
    limit of the attention)."""
    tokens = [int(v) for v in tokens]
    t = len(tokens)
    attn = P.ready(model, cache)
    split = bool(getattr(cache, "split", False))
    progs = model.__dict__.setdefault("_programs", {})
    key = ("prompt", attn, t, split, getattr(cache, "parts", 1), model.prompt_act)
    prog = progs.get(key)
    if prog is None:
        prog = progs[key] = compile_prompt(model, attn, t, cache if split else None)
    # As bind_step: the cache makes room for the rows (prepare) and the
    # parameters of the step; a PartKVCache has no kq.L arrays (None).
    kw = P.step_params(prog, model, cache, pos)
    prog.bind(**{n: v for n, v in kw.items() if n in prog.by_name and v is not None})
    if split:
        for layer, p, ip in prog.part_params:
            base = cache.base[layer]
            ip[1], ip[2], ip[3] = pos - base, pos + t - base, base
            prog.bind(**{"p%s.%d.%d" % (name, layer, p): a
                         for name, a in zip(("kq", "ks", "vq", "vs"), cache.part(layer, p))})
    positions = np.arange(pos, pos + t, dtype=np.int32)
    prog.bind(positions=positions)
    limit = model._media_limit(positions) if media else None
    prog.bind(limit=0 if limit is None else limit.astype(np.int32))
    x = prog.names["x"]
    x[:] = model.embed(tokens)
    for sp in media:
        lo, hi = max(sp.start, pos), min(sp.end, pos + t)
        x[lo - pos:hi - pos] = sp.rows[lo - sp.start:hi - sp.start]
    prog.run()
    return prog.names["xn"].copy()
