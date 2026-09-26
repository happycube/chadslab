"""Give the numeric functions for the model.

All functions use NumPy only.

Functions:
    rms_norm      Normalize the last axis. Multiply by a weight.
    linear        Multiply x by W.
    bf16_to_f32   Convert bfloat16 data to float32 data.
    linear_bf16   Multiply x by W. W is in bfloat16 format.
    linear_q6k    Multiply x by W. W is in the Q6_K block format.
    dequantize_q6k  Change Q6_K block bytes into float32 values.
    gelu_tanh     Apply the GELU activation function.
    softmax       Change scores into probabilities.
    softcap       Limit the size of the logits.
"""
from __future__ import annotations

import os

import numpy as np

# The C kernel and the Numba kernels are optional. Use them when they are available.
try:
    from . import cops as _cops
except Exception:
    _cops = None
try:
    from . import numba_ops as _numba_ops
except Exception:
    _numba_ops = None

# The constant sqrt(2/pi). The GELU function uses it.
GELU_C = 0.7978845608028654

# The state of the C library and the choice of kernel path. A decode step calls
# the functions below about 600 times for each token, and a lookup of the
# environment costs about 3 microseconds of the step. Every value here is
# therefore read one time, at import. Set the environment variables before the
# import of this module.
_COPS_READY = _cops is not None and _cops.available()
_KERNEL_MODE = os.environ.get("NP_GEMMA_KERNEL", "auto").lower()
_FUSED_QKV = os.environ.get("NP_GEMMA_FUSED_QKV", "1") == "1"
_INT4_MULTI4 = os.environ.get("NP_GEMMA_INT4_MULTI4", "1") == "1"
_INT4_Q8 = os.environ.get("NP_GEMMA_INT4_Q8", "1") == "1"
_INT4_Q8_GEMV = os.environ.get("NP_GEMMA_INT4_Q8_GEMV", "0") == "1"
# The int8 tile needs AVX-512. The int8 dot product for one token needs VNNI.
_INT4_Q8_OK = _INT4_Q8 and _COPS_READY and bool(getattr(_cops, "AVX512", False))
_INT4_Q8_GEMV_OK = (_INT4_Q8_GEMV and _COPS_READY
                    and bool(getattr(_cops, "VNNI", False)))
_QKV_OK = _FUSED_QKV and _COPS_READY
_MULTI4_OK = _INT4_MULTI4 and _COPS_READY

# A one-value array for a kernel argument that the kernel does not read. A
# shared layer of the E4B model has no key of its own.
_EMPTY_F32 = np.zeros(1, dtype=np.float32)


def _w32(w):
    """Return a weight as a contiguous float32 array. None gives an empty one."""
    if w is None:
        return _EMPTY_F32
    return np.ascontiguousarray(w, dtype=np.float32)


def rms_norm(x, weight=None, eps=1e-6):
    """Normalize the last axis of x. Multiply by the weight.

    The formula is x * pow(mean(x * x) + eps, -0.5) * weight.

    Note: the weight is the full scale. Do not add 1 to the weight.
    """
    if _COPS_READY:
        x2 = np.ascontiguousarray(x, dtype=np.float32)
        shape = x2.shape
        x2 = x2.reshape(-1, shape[-1])
        w2 = None if weight is None else _w32(weight)
        return _cops.rms_norm(x2, w2, eps).reshape(shape)
    x32 = np.asarray(x, dtype=np.float32)
    # Calculate the mean of the squares. Add eps for stability.
    mean_sq = np.mean(x32 * x32, axis=-1, keepdims=True) + eps
    y = x32 * np.power(mean_sq, -0.5)
    if weight is not None:
        y = y * np.asarray(weight, dtype=np.float32)
    return y


def linear(x, w):
    """Multiply x by W. Use the transpose of W.

    W has the shape (out_features, in_features).
    """
    return np.asarray(x, dtype=np.float32) @ np.asarray(w, dtype=np.float32).T


def bf16_to_f32(u16):
    """Convert bfloat16 data to float32 data.

    Move the 16 data bits to the top of a 32-bit word. The shift occurs in
    place. Thus only one float32 temporary is necessary.
    """
    raw = np.asarray(u16, dtype=np.uint16)
    out = raw.astype(np.uint32)   # One temporary. Then shift in place.
    out <<= 16
    return out.view(np.float32)


def to_bf16(x):
    """Round float32 data to bfloat16. Return the raw uint16 values.

    Round to the nearest even value before the shift. Work in a chunk, so the
    float32 data of the chunk stays in the cache.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    flat = x.reshape(-1).view(np.uint32)
    out = np.empty(flat.shape, dtype=np.uint16)
    step = 1 << 20
    for i in range(0, flat.size, step):
        u = flat[i:i + step]
        out[i:i + step] = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    return out.reshape(x.shape)


# The number of output rows in one dequant block. A small block keeps the
# float32 data in cache. A large block lowers the Python work. Change the value
# with the environment variable NP_GEMMA_BF16_CHUNK.
LINEAR_BF16_CHUNK = int(os.environ.get("NP_GEMMA_BF16_CHUNK", "8192"))

def quantize_int8(w, group=None):
    """Quantize W to int8. Return the int8 data and the float32 scales.

    Each row has one symmetric scale for each group. The formula for one group
    is scale = max(abs(group)) / 127.

    Set group to the number of columns for one scale for each row. The default
    is one scale for each row. A smaller group gives a smaller error. A smaller
    group also makes the multiply slower.
    """
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    group = cols if group is None else group
    groups = cols // group
    wg = w.reshape(rows, groups, group)
    scale = np.max(np.abs(wg), axis=2) / 127.0
    scale = np.where(scale == 0.0, 1e-12, scale).astype(np.float32)
    q = np.rint(wg / scale[:, :, None]).clip(-127.0, 127.0).astype(np.int8)
    return q.reshape(rows, cols), scale


def int8_group(q, scales):
    """Return the number of columns in one int8 scale group."""
    return q.shape[1] // scales.shape[1]


def linear_int8_numpy(x, q, scales):
    """Multiply x by W with NumPy. W is int8 data. Dequantize W first."""
    x = np.asarray(x, dtype=np.float32)
    rows, cols = q.shape
    group = int8_group(q, scales)
    groups = cols // group
    w = (q.reshape(rows, groups, group).astype(np.float32) * scales[:, :, None]).reshape(rows, cols)
    return x @ w.T


def linear_bf16_numpy(x, w_u16, chunk=LINEAR_BF16_CHUNK):
    """Multiply x by W with NumPy. W is raw bfloat16 data.

    Convert one block of output rows at a time. Only one float32 block exists at
    a time. Thus the memory stays at the bfloat16 size.
    """
    x = np.asarray(x, dtype=np.float32)
    out_dim = w_u16.shape[0]
    out = np.empty((x.shape[0], out_dim), dtype=np.float32)
    for start in range(0, out_dim, chunk):
        stop = min(start + chunk, out_dim)
        out[:, start:stop] = x @ bf16_to_f32(w_u16[start:stop]).T
    return out


def linear_bf16(x, w_u16, chunk=LINEAR_BF16_CHUNK):
    """Multiply x by W. W is raw bfloat16 data.

    Use the fastest available kernel. The order is C, then Numba, then NumPy.
    Set the environment variable NP_GEMMA_KERNEL to "c", "numba", or "numpy" to
    select one kernel.
    """
    mode = _KERNEL_MODE
    if mode != "numpy":
        if mode in ("auto", "c") and _COPS_READY:
            return _cops.linear_bf16(x, w_u16)
        if mode in ("auto", "numba") and _numba_ops is not None and _numba_ops.enabled():
            return _numba_ops.linear_bf16(x, w_u16)
    return linear_bf16_numpy(x, w_u16, chunk=chunk)


def dequantize_q6k(w_bytes, cols):
    """Return float32 values from raw Q6_K block bytes.

    w_bytes holds blocks of 210 bytes. One block gives 256 values. cols is the
    value count in one row. Use this function when the C kernel is not ready.
    """
    raw = np.ascontiguousarray(w_bytes, dtype=np.uint8).reshape(-1, 210)
    nb = raw.shape[0]
    ql = raw[:, 0:128].reshape(nb, 2, 64)
    qh = raw[:, 128:192].reshape(nb, 2, 32)
    sc = np.ascontiguousarray(raw[:, 192:208]).view(np.int8).astype(np.int16).reshape(nb, 2, 8)
    d = np.ascontiguousarray(raw[:, 208:210]).view("<f2").astype(np.float32).reshape(nb)
    q1 = ((ql[:, :, 0:32] & 0x0F) | (((qh >> 0) & 3) << 4)).astype(np.int16) - 32
    q2 = ((ql[:, :, 32:64] & 0x0F) | (((qh >> 2) & 3) << 4)).astype(np.int16) - 32
    q3 = ((ql[:, :, 0:32] >> 4) | (((qh >> 4) & 3) << 4)).astype(np.int16) - 32
    q4 = ((ql[:, :, 32:64] >> 4) | (((qh >> 6) & 3) << 4)).astype(np.int16) - 32
    out = np.concatenate([
        q1 * np.repeat(sc[:, :, 0:2], 16, axis=2),
        q2 * np.repeat(sc[:, :, 2:4], 16, axis=2),
        q3 * np.repeat(sc[:, :, 4:6], 16, axis=2),
        q4 * np.repeat(sc[:, :, 6:8], 16, axis=2),
    ], axis=2)
    out = out.reshape(nb, 256).astype(np.float32) * d[:, None]
    return out.reshape(-1)


def linear_q6k_numpy(x, w_bytes, cols):
    """Multiply x by W with NumPy. W is raw Q6_K block bytes."""
    x = np.asarray(x, dtype=np.float32)
    rows = w_bytes.shape[0]
    w = dequantize_q6k(w_bytes, cols).reshape(rows, cols)
    return x @ w.T


def linear_q6k(x, w_bytes, cols):
    """Multiply x by W. W is the raw Q6_K block data of a 2-D tensor.

    w_bytes has shape (rows, blocks in one row * 210). cols is the value count
    in one row. The C kernel keeps the weights in the Q6_K format. Thus the
    output head reads 6.05 bits for each weight in place of 16 bits.
    """
    if _KERNEL_MODE != "numpy" and _COPS_READY:
        return _cops.linear_q6k(x, w_bytes, cols)
    return linear_q6k_numpy(x, w_bytes, cols)


# The key and value cache keeps an int8 copy only when it holds at least this
# many values. Below it the float32 path is faster.
ATTN_MIN = int(os.environ.get("NP_GEMMA_ATTN_MIN", "128"))


def qkv_ready():
    """Return True when the fused norm and RoPE kernels are ready.

    Set NP_GEMMA_FUSED_QKV=0 to give each tensor its own norm and its own
    rotation. That is three norm calls and two rope calls for each layer in
    place of two calls.
    """
    return _QKV_OK


def qkv_norm(q, q_w, k, k_w, v, eps):
    """Apply the RMSNorm of the query, the key, and the value in place.

    q, k, and v are (rows, head_dim). A shared layer of the E4B model has no
    key and no value of its own, so k and v may be None. The value norm of
    that model has no weight, so k_w and the weight of the value may be None.
    """
    if qkv_ready():
        _cops.qkv_norm(q, _w32(q_w), q.shape[0],
                       k, _w32(k_w), 0 if k is None else k.shape[0],
                       v, 0 if v is None else v.shape[0], q.shape[1], eps)
        return
    for x, w in ((q, q_w), (k, k_w), (v, None)):
        if x is not None:
            x[:] = rms_norm(x, w, eps)

def rope_apply(q, k, cos, sin, q_heads, k_heads, head_dim):
    """Apply RoPE to the query and the key in place. k may be None."""
    if qkv_ready():
        _cops.rope_apply(q, q.shape[0], q_heads,
                         k, 0 if k is None else k.shape[0], k_heads,
                         np.ascontiguousarray(cos, dtype=np.float32),
                         np.ascontiguousarray(sin, dtype=np.float32), head_dim)
        return
    from . import rope as _rope
    for x, heads in ((q, q_heads), (k, k_heads)):
        if x is not None:
            y = x.reshape(-1, heads, head_dim)
            y[:] = _rope.apply(y, cos, sin)


def router_ready():
    """Return True when the fused router kernel is ready."""
    return _cops is not None and _cops.available()


def router(x, scale, proj, per_expert, top_k, eps, hscale):
    """Run the router for one token. Return the weights and the expert indices."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    scale = np.ascontiguousarray(scale, dtype=np.float32)
    proj = np.ascontiguousarray(proj, dtype=np.float32)
    per_expert = np.ascontiguousarray(per_expert, dtype=np.float32)
    hidden = x.shape[1]
    experts = proj.shape[0]
    val = np.empty((1, top_k), dtype=np.float32)
    idx = np.empty((1, top_k), dtype=np.int32)
    _cops.router(x[0], scale, proj, per_expert, hidden, experts, top_k, eps, hscale,
                 val.reshape(-1), idx.reshape(-1))
    return val, idx.astype(np.int64)


# Use the fused entry points of the C library for a decode step. Each one runs
# two kernels in one call. Set NP_GEMMA_FUSED_STEP=0 to compare with the
# separate calls.
_FUSED_STEP = os.environ.get("NP_GEMMA_FUSED_STEP", "1") == "1"
_FUSED_OK = _FUSED_STEP and _COPS_READY

# A scratch buffer for the fused entry points. It belongs to this module. A
# caller must not keep a result across another call that uses the same size.
#
# The address is kept with the buffer. One decode step hands this buffer to
# the same kernel 30 times, one for each layer, and the read of the address
# costs about 1.5 microseconds.
_SCRATCH = np.zeros(1, dtype=np.float32)
_SCRATCH_A = _SCRATCH.ctypes.data


def _scratch(n):
    """Return (buffer, address) for a scratch of at least n float32 values.

    The buffer is made one time and reused. The address goes with it, so the
    caller does not read it again.
    """
    global _SCRATCH, _SCRATCH_A
    if _SCRATCH.size < n:
        _SCRATCH = np.empty(n, dtype=np.float32)
        _SCRATCH_A = _SCRATCH.ctypes.data
    return _SCRATCH, _SCRATCH_A


def qkv_norm_rope(q, q_w, k, k_w, v, cos, sin, q_heads, k_heads, head_dim, eps,
                  cos_a=None, sin_a=None):
    """The three norms of the attention block and the two rotations.

    One call in place of two. q, k, and v are (rows, head_dim). k and v may be
    None. cos_a and sin_a give the addresses of the tables, which the caller
    read one time. They are for the fused call only.
    """
    if _FUSED_OK:
        _cops.qkv_norm_rope(
            q, None if q_w is None else _w32(q_w), k,
            None if k_w is None else _w32(k_w), v,
            cos if cos_a is None else cos_a,
            sin if sin_a is None else sin_a,
            q_heads, k_heads, head_dim, eps)
        return
    qkv_norm(q, q_w, k, k_w, v, eps)
    rope_apply(q, k, cos, sin, q_heads, k_heads, head_dim)


def rms_norm_multi4(x, wn, eps, mats, cols):
    """Normalize one row, then run up to four int4 matrices on the result.

    One call in place of two. mats is a list of up to four (packed, scales)
    pairs. A None entry skips a matrix. Return a list of results, or None for
    a skipped matrix.
    """
    if _FUSED_OK:
        _buf, addr = _scratch(cols)
        return _cops.rms_norm_multi4(
            np.ascontiguousarray(x, dtype=np.float32),
            None if wn is None else _w32(wn), addr, eps, mats, cols)
    h = rms_norm(x, wn, eps)
    h = np.ascontiguousarray(h, dtype=np.float32).reshape(1, -1)
    if _COPS_READY:
        # The kernel that the model used before the fusion. It gives the same
        # bits as the fused call.
        return int4_multi4(mats, h, cols)
    return [None if m is None else linear_int4(h, m[0], m[1]).reshape(-1)
            for m in mats]


def gelu_mul_int4(g, u, packed, scales, rows, cols):
    """gelu(g) * u, then one int4 matrix on the result."""
    if _FUSED_OK:
        _buf, addr = _scratch(g.size)
        return _cops.gelu_mul_int4(g, u, addr, packed, scales, rows, cols)
    h = (gelu_tanh(g) * u).reshape(1, -1)
    return linear_int4(h, packed, scales).reshape(-1)


def moe_gemv_gelu(w, scales, x, ids, rows, cols, xstride, inner):
    """The gate and up projection of the selected experts, then the GELU.

    One call in place of two. Return one row of inner values for each job.
    """
    if _FUSED_OK:
        return _cops.moe_gemv_gelu(w, scales, x, ids, ids.size, rows, cols,
                                   xstride, inner)
    act = int4_moe_gemv(w, scales, x, ids, rows, cols, xstride)
    return gelu_tanh(act[:, :inner]) * act[:, inner:]


def int4_multi4_ready():
    """Return True when the fused multi-matrix kernel is ready.

    Set NP_GEMMA_INT4_MULTI4=0 to compare the fused kernel with one call for
    each matrix.
    """
    return _MULTI4_OK


# The small-group path. An MTP verify step runs the target on two to about
# eight tokens. These kernels read each weight block one time for the whole
# group, and they give each token the same bits as a decode step. The path
# copies the kernels of the float decode step, so it needs the fused step and
# the float activation. Set NP_GEMMA_MT=0 to use the prompt kernels instead.
_MT = os.environ.get("NP_GEMMA_MT", "1") == "1"


def mt_ready(tokens):
    """Return True when the small-group kernels serve this token count."""
    return (_MT and 2 <= tokens <= getattr(_cops, "MT_MAX", 0)
            and _FUSED_OK and _MULTI4_OK and not _INT4_Q8_GEMV)


def router_mt(x, scale, proj, per_expert, top_k, eps, hscale):
    """Run the router for a small group of tokens, as router does for one."""
    return _cops.router_mt(x, _w32(scale), _w32(proj), _w32(per_expert),
                           top_k, eps, hscale)


def linear_bf16_mt(x, w_u16):
    """Multiply a small group of rows of x by W. W is raw bfloat16 data."""
    return _cops.linear_bf16_gemv(x, w_u16)


def linear_int4_mt(x, packed, scales):
    """Multiply a small group of rows of x by W. W is packed 4-bit data."""
    return _cops.linear_int4_mt(x, packed, scales)


def int4_multi4_mt(mats, x, cols):
    """Run up to four int4 matrices on the same small group of rows."""
    return _cops.int4_multi4_mt(mats, x, cols)


def gelu_mul_rows(g, u):
    """Return gelu(g) * u one row at a time, as the decode step does.

    The kernel treats a tail shorter than one vector in a different way, so a
    call for each row keeps the tail of each row the same.
    """
    return np.stack([_cops.gelu_mul_pair(g[j], u[j]) for j in range(g.shape[0])])


def moe_gemv_mt(w, scales, x, ids, poff, xi, rows, cols, xstride, inner=0):
    """Run the selected experts of a small group of tokens. See cops.moe_gemv_mt."""
    return _cops.moe_gemv_mt(w, scales, x, ids, poff, xi, rows, cols, xstride, inner)


# The activation format for the matrices of one token. The int8 form reads 8
# weight rows in one multiply, but it must apply the group scale in float32
# afterwards. The C test .cache/gemv_real.c gives 1.0 to 1.2 times the speed of
# the float form, and the end-to-end test cannot separate the two from the
# machine noise. The float form is therefore the default. Set
# NP_GEMMA_INT4_Q8_GEMV=1 to select the int8 form on a machine with VNNI.
def gemv_mode():
    """Return "float" or "int8", the activation format for one token."""
    return "int8" if _INT4_Q8_GEMV else "float"


def int4_multi4(mats, x, cols):
    """Run up to four int4 matrices on the same one-row x."""
    if _INT4_Q8_GEMV_OK:
        return _cops.int4_q8_multi4(mats, x, cols)
    return _cops.int4_multi4(mats, x, cols)


def attn_ready():
    """Return True when the fused attention kernel is ready.

    Set NP_GEMMA_ATTN=0 to use the float32 key and value cache and the NumPy
    attention path. That path matches the reference more closely.
    """
    if os.environ.get("NP_GEMMA_ATTN", "1") == "0":
        return False
    return _cops is not None and _cops.available()


def quantize_q8(x):
    """Quantize the last axis of x to int8.

    The length of the last axis must be a multiple of 32. Return the int8 data
    and one float32 scale for each group of 32 values.
    """
    x = np.asarray(x, dtype=np.float32)
    shape = x.shape
    if _cops is not None and _cops.available() and x.size and shape[-1] % 32 == 0:
        cols = shape[-1]
        qx, sx, _sumx = _cops.quantize_q8_groups(x.reshape(-1, cols))
        return qx.reshape(shape), sx.reshape(shape[:-1])
    flat = x.reshape(-1, 32)
    amax = np.max(np.abs(flat), axis=1)
    scale = np.where(amax > 0.0, amax / 127.0, 1e-12).astype(np.float32)
    q = np.rint(flat / scale[:, None]).clip(-127.0, 127.0).astype(np.int8)
    return q.reshape(shape), scale.reshape(shape[:-1])


def attn_decode(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, n):
    """Run the fused attention for one query token.

    q is the query after RoPE, with the shape (q_heads, head_dim). The key and
    value caches hold int8 values and one float32 scale for each group of 32.
    Return the attention output, with the shape (q_heads, head_dim).
    """
    g = head_dim // 32
    qq, qs = quantize_q8(q.reshape(q_heads, g, 32))
    qq = np.ascontiguousarray(qq).reshape(q_heads, head_dim)
    qs = np.ascontiguousarray(qs)
    scores = np.empty((q_heads, n), dtype=np.float32)
    out = np.empty((q_heads, head_dim), dtype=np.float32)
    _cops.attn_decode(qq, qs, kq, ks, vq, vs, scores, out,
                      q_heads, kv_heads, head_dim, n)
    return out


def attn_decode_mt(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, lo, n):
    """Run the fused attention for a small group of query tokens in one call.

    q is (tokens, q_heads, head_dim). Token t reads the cache rows lo[t] to
    lo[t] + n[t] - 1. Each token gets the bits of attn_decode on those rows.
    Return (tokens, q_heads, head_dim).
    """
    t = q.shape[0]
    g = head_dim // 32
    qq, qs = quantize_q8(q.reshape(t, q_heads, g, 32))
    qq = np.ascontiguousarray(qq).reshape(t, q_heads, head_dim)
    qs = np.ascontiguousarray(qs)
    return _cops.attn_decode_mt(qq, qs, kq, ks, vq, vs, q_heads, kv_heads,
                                head_dim, lo, n)


def attn_decode_f32(q, k, v, pos, base=0, window=0):
    """Run the fused float32 attention for one query token.

    q is (1, q_heads, head_dim) after RoPE, or (q_heads, head_dim). k and v
    are (kv_heads, keys, head_dim), the layout of the cache. They may be a
    part of a larger buffer: the kernel uses the stride between two heads, so
    it copies nothing. pos is the position of the query and base is the
    position of key zero. A window of zero turns the sliding window off.
    Return the output, with the shape of q.
    """
    k = np.asarray(k, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    q2 = np.ascontiguousarray(q, dtype=np.float32).reshape(-1)
    head_dim = k.shape[2]
    q_heads = q2.size // head_dim
    n = k.shape[1]
    scores = np.empty((q_heads, n), dtype=np.float32)
    out = np.empty((q_heads, head_dim), dtype=np.float32)
    _cops.attn_decode_f32(q2, k, v, scores, out, q_heads, k.shape[0],
                          head_dim, n, k.strides[0] // 4, v.strides[0] // 4,
                          int(pos), int(base), int(window))
    return out.reshape(q.shape)


def int4_moe_ready():
    """Return True when the fused expert kernel is ready."""
    return _cops is not None and _cops.available()


def int4_moe_gemv(w, scales, x, ids, rows, cols, xstride):
    """Multiply each selected expert matrix by its input row.

    Use the fused kernel when the C library is ready. Otherwise, run one call
    for each expert. When the machine has VNNI, use the int8 activation, which
    holds eight weight rows in the lanes of one multiply.
    """
    if _COPS_READY:
        if _INT4_Q8_GEMV_OK:
            return _cops.int4_q8_moe_gemv(w, scales, x, ids, rows, cols, xstride)
        return _cops.int4_moe_gemv(w, scales, x, ids, rows, cols, xstride)
    ids = np.asarray(ids, dtype=np.int64)
    out = np.empty((ids.size, rows), dtype=np.float32)
    for j in range(ids.size):
        row = x if xstride == 0 else x[j * xstride:j * xstride + cols]
        out[j] = linear_int4(row, w[ids[j]], scales[ids[j]])
    return out


# The number of columns in one int4 scale group.
INT4_GROUP = 32


def pack_int4_blocks(scale, qs):
    """Join a scale and 16 nibble bytes into the Q4_0 block layout.

    Return an array of shape (..., 18) of uint8. Bytes 0 and 1 hold the scale
    as float16. Bytes 2 to 17 hold the 16 nibble bytes. A nibble holds the
    value plus 8. This is the layout of the Q4_0 type of the GGUF format.
    """
    head = scale.astype(np.float16).view(np.uint8).reshape(scale.shape + (2,))
    return np.concatenate([head, qs], axis=-1)


def quantize_int4(w, group=INT4_GROUP):
    """Quantize W to 4-bit. Return the packed data and the float32 scales.

    One byte holds two values. A block of 32 values uses 16 bytes. For block bo,
    byte j holds value bo*32+j in the low nibble and value bo*32+16+j in the
    high nibble. A value is a signed 4-bit number.

    group gives the number of columns in one scale group. The default is 32.
    This value matches the quantization-aware training of the model. A smaller
    group gives a smaller error and a slower multiply. Use group=None for one
    scale for each row.
    """
    w = np.asarray(w, dtype=np.float32)
    rows, cols = w.shape
    group = cols if group is None else group
    groups = cols // group
    wg = w.reshape(rows, groups, group)
    # Use the offset-8 nibble and the block layout of Q4_0. Thus a Q4_0 file
    # needs no change of the nibbles.
    scale = np.max(np.abs(wg), axis=2) / 8.0
    scale = np.where(scale == 0.0, 1e-12, scale).astype(np.float32)
    q = np.rint(wg / scale[:, :, None]).clip(-8.0, 7.0).astype(np.int16)
    qb = (q + 8).astype(np.uint8).reshape(rows, groups, 2, group // 2)
    qs = (qb[:, :, 0, :] | (qb[:, :, 1, :] << 4)).astype(np.uint8)
    return pack_int4_blocks(scale, qs), scale


def convert_w4a16(packed_i32, scale_bf16):
    """Change the w4a16 packing to the runtime 4-bit packing.

    The input is the pack-quantized data of the compressed-tensors format. For
    one row, byte i holds column 2i in the low nibble and column 2i+1 in the
    high nibble. A value is the nibble minus 8.

    The output uses the layout of quantize_int4. The function also returns the
    scales as float32 values.
    """
    rows = packed_i32.shape[0]
    b = packed_i32.view(np.uint8).reshape(rows, -1)
    cols = b.shape[1] * 2
    vals = np.empty((rows, cols), dtype=np.uint8)
    vals[:, 0::2] = b & 0x0F
    vals[:, 1::2] = (b >> 4) & 0x0F
    # The w4a16 nibble is already the offset-8 nibble. Reorder the pairs of
    # columns to the block layout. The value of a nibble stays the same.
    groups = cols // 32
    qb = vals.reshape(rows, groups, 2, 16)
    qs = (qb[:, :, 0, :] | (qb[:, :, 1, :] << 4)).astype(np.uint8)
    scale = np.ascontiguousarray(scale_bf16, dtype=np.float32)
    return pack_int4_blocks(scale, qs), scale


def dequantize_int4(packed, scales):
    """Return float32 values from packed 4-bit data."""
    # packed holds blocks of 18 bytes: a float16 scale and 16 nibble bytes.
    # The value of a nibble is the nibble minus 8.
    rows = packed.shape[0]
    groups = int(np.prod(packed.shape[1:-1]))
    qs = packed.reshape(rows, groups, 18)[:, :, 2:18]
    lo = (qs & 0x0F).astype(np.int8) - 8
    hi = ((qs >> 4) & 0x0F).astype(np.int8) - 8
    qb = np.empty((rows, groups, 2, 16), dtype=np.int8)
    qb[:, :, 0, :] = lo
    qb[:, :, 1, :] = hi
    q = qb.reshape(rows, groups * 32)
    scale = scales.reshape(rows, groups)
    return q.astype(np.float32) * np.repeat(scale, 32, axis=1)


def int4_group(packed, scales):
    """Return the number of columns in one int4 scale group."""
    return INT4_GROUP


def linear_int4_numpy(x, packed, scales):
    """Multiply x by W with NumPy. W is packed 4-bit data. Dequantize W first."""
    x = np.asarray(x, dtype=np.float32)
    return x @ dequantize_int4(packed, scales).T


# Use the int4 kernel with int8 activations. The kernel quantizes the
# activations to int8 with one scale for each group of 32 columns. The dot then
# uses the integer multiply maddubs. Set NP_GEMMA_INT4_Q8=0 to compare with
# the float path. The 26B model gives the same token ids on the Paris test.
# The smallest token count for the int8 tile. A smaller count wastes the token
# lanes and pays for the quantization of the activations.
_INT4_Q8_TOKENS = int(os.environ.get("NP_GEMMA_INT4_Q8_TOKENS", "2"))
# The token block of the int8 tile. The value 8 selects the narrow 256-bit
# kernel, which wastes fewer lanes for an expert with fewer than sixteen tokens.
# The value must match I4Q_TB or I4Q2_TB in the C kernel.
_INT4_Q8_TB = int(os.environ.get("NP_GEMMA_INT4_Q8_TB", "16"))
if _INT4_Q8_TB == 8 and _cops is not None and _cops.available():
    _cops.set_int4_q8_tb8(1)


def int4_q8_ready():
    """Return True when the int4 kernel with int8 activations is ready."""
    return _INT4_Q8_OK


def int4_q8_gemv_ready():
    """Return True when the int8 dot product for one token is ready.

    A machine without VNNI keeps the float kernels. Set
    NP_GEMMA_INT4_Q8_GEMV=0 for a comparison on a machine with VNNI.
    """
    return _INT4_Q8_GEMV_OK


def linear_int4_q8_gemv(x, packed, scales):
    """Multiply a one-row x by W with int8 activations.

    The C kernel uses the integer multiply vpdpbusd, which does 32 byte
    products in one step. The weight rows fill the lanes of the register, so no
    lane is idle. The kernel quantizes the activation itself, because a
    separate quantization call costs more than the kernel at this size.
    """
    return _cops.int4_q8_gemv_x(x, packed, scales)


def int4_q8_multi4(mats, x, cols):
    """Run up to four int4 matrices on the same one-row x with int8 data."""
    return _cops.int4_q8_multi4(mats, x, cols)


def linear_int4_q8(x, packed, scales):
    """Multiply x by W with int8 activations. W is packed 4-bit data.

    Quantize x to int8 with one scale for each group of 32 columns. The C kernel
    then uses the integer multiply maddubs on the nibbles of W. The token count
    is padded to a full token block.
    """
    tokens = x.shape[0]
    stride = (tokens + _INT4_Q8_TB - 1) // _INT4_Q8_TB * _INT4_Q8_TB
    qxt, sx, sumx = _cops.quantize_q8_t(x, stride)
    return _cops.int4_q8_tile(qxt, sx, sumx, packed, scales, INT4_GROUP, tokens)


def linear_int4_q8_wide(x, packed, scales):
    """Multiply x by W with the wide int8 tile.

    The wide tile reads 32 tokens for one weight decode instead of 16. That
    halves the weight decode and the weight broadcast for each token. A test
    uses it.
    """
    tokens = x.shape[0]
    stride = (tokens + 31) // 32 * 32
    qxt, sx, sumx = _cops.quantize_q8_t(x, stride)
    return _cops.int4_q8_tile32(qxt, sx, sumx, packed, scales, INT4_GROUP, tokens)


def int4_q8_moe_ready():
    """Return True when the fused int8 mixture-of-experts kernel is ready."""
    return int4_q8_ready()


def moe_int4_q8(h, gu, dn, val, idx, inner):
    """Run the selected experts for a prompt with int8 activations.

    h is (tokens, hidden). gu is the packed gate and up projection with one
    matrix for each expert, plus its float32 scales. dn is the same for the down
    projection. idx is (tokens, top_k) and val gives the router weight of each
    expert. Return the sum of the expert outputs, weighted by the router.

    The experts run in one parallel region for the gate and up projection and
    one for the down projection. The gather, the GELU, and the scatter stay in
    C, so the loop over the experts holds no work on the NumPy side.
    """
    gu_p, gu_s = gu
    dn_p, dn_s = dn
    tokens, hidden = h.shape
    top_k = idx.shape[1]
    flat_e = np.asarray(idx, dtype=np.int64).reshape(-1)
    flat_t = np.repeat(np.arange(tokens), top_k)
    order = np.argsort(flat_e, kind="stable")
    # src maps a row of the expert scratch to a row of h.
    src = flat_t[order].astype(np.int32)
    eid, _first, counts = np.unique(flat_e[order], return_index=True,
                                    return_counts=True)
    ntok = counts.astype(np.int32)
    off = np.zeros(eid.size, dtype=np.int32)
    if eid.size > 1:
        np.cumsum(ntok[:-1], out=off[1:])
    n = tokens * top_k
    # A tile may read one token block past the end of an expert. Leave a slack.
    stride = n + 16
    qxt, sx, sumx = _cops.quantize_q8_t_moe(h, hidden, stride, off, ntok, src)
    act = _cops.int4_q8_moe(gu_p, gu_s, qxt, sx, sumx, gu_p.shape[1], hidden,
                            stride, off, ntok, eid)[:n]
    act2 = _cops.gelu_mul(act, inner)
    qxt2, sx2, sumx2 = _cops.quantize_q8_t_moe(act2, inner, stride, off, ntok)
    de = _cops.int4_q8_moe(dn_p, dn_s, qxt2, sx2, sumx2, dn_p.shape[1], inner,
                           stride, off, ntok, eid)[:n]
    w = np.ascontiguousarray(np.asarray(val, dtype=np.float32).reshape(-1)[order])
    out = np.zeros_like(h)
    _cops.moe_scatter(out, de, src, w, hidden, n)
    return out


# Use the int4 prompt GEMM for this many tokens or more. The GEMM decodes a
# row block to float32 one time and reuses it for every token block. The one
# row dot decodes the weights again for each token. Below one full token block
# the GEMM does no work, so use the one-row dot.
_INT4_GEMM_TOKENS = int(os.environ.get("NP_GEMMA_INT4_GEMM_TOKENS", "64"))


def linear_int4(x, packed, scales):
    """Multiply x by W. W is packed 4-bit data.

    Use the C kernel when the C path is available. Otherwise, dequantize W and
    use NumPy.
    """
    if _KERNEL_MODE != "numpy" and _COPS_READY:
        group = INT4_GROUP
        tokens = x.shape[0]
        if tokens == 1 and _INT4_Q8_GEMV_OK:
            return linear_int4_q8_gemv(x, packed, scales)
        if tokens >= _INT4_Q8_TOKENS and _INT4_Q8_OK:
            return linear_int4_q8(x, packed, scales)
        if tokens >= _INT4_GEMM_TOKENS:
            xt = np.ascontiguousarray(x.T)
            return _cops.linear_int4_gemm(x, xt, packed, scales, group)
        if tokens >= _cops.INT4_TILE_TOKENS:
            # A small group of tokens. Use the token-vectorized tile. It reads
            # the x block one time for several weight rows.
            xt = np.ascontiguousarray(x.T)
            return _cops.linear_int4_tile(x, xt, packed, scales, group)
        return _cops.linear_int4(x, packed, scales, group)
    return linear_int4_numpy(x, packed, scales)


# Use the int8 GEMM for a prompt with this many tokens or more. A test gave
# the same speed at 32 tokens and 2 to 3 times more speed above 64 tokens.
_INT8_GEMM_TOKENS = 32


def pack_int8_16x16(q):
    """Pack an int8 matrix in blocks of 16 rows and 16 columns.

    Block (tr, tc) holds the 16 columns of 16 rows. Column c of the block holds
    the 16 rows of one column. One block is 256 bytes and fills four cache
    lines.
    """
    rows, cols = q.shape
    p = q.reshape(rows // 16, 16, cols // 16, 16)
    p = p.transpose(0, 2, 3, 1)
    return np.ascontiguousarray(p)


def linear_int8(x, q, scales, packed=None):
    """Multiply x by W. W is int8 data.

    Set packed to a (cols, rows) int8 copy of W to use the fast prompt GEMM on
    AVX-512. The copy keeps the weight of each column together.

    The C integer kernel uses one scale for each row of W. It also quantizes x
    to int8. The kernel uses integer multiply and add.
    Use the NumPy path for a group of columns or when the C path is absent.
    """
    mode = _KERNEL_MODE
    if mode != "numpy" and _cops is not None and _cops.available():
        group = int8_group(q, scales)
        # The integer kernel quantizes the activations too. This step causes a
        # small loss of accuracy. Use NP_GEMMA_INT8_INT=1 to select it.
        if os.environ.get("NP_GEMMA_INT8_INT") == "1" and group >= q.shape[1]:
            return _cops.linear_int8_s8(x, q, scales)
        # The one-row kernel reads x again for each row. Use the GEMM for a
        # prompt, because the x data is then too large for the cache.
        tokens = x.shape[0]
        if tokens >= _INT8_GEMM_TOKENS and group >= q.shape[1]:
            xt = np.ascontiguousarray(x.T)
            if packed is not None and _cops.AVX512:
                return _cops.linear_int8_gemm_blk(x, xt, q, packed, scales)
            return _cops.linear_int8_gemm(x, xt, q, scales)
        return _cops.linear_int8_float(x, q, scales, group)
    return linear_int8_numpy(x, q, scales)



def topk_k(x, k):
    """Return the k largest values of the last axis and their indices.

    The values and the indices are in decreasing order. Use this function for
    the router of the mixture-of-experts block.
    """
    x = np.asarray(x)
    part = np.argpartition(-x, k - 1, axis=-1)[..., :k]
    val = np.take_along_axis(x, part, axis=-1)
    order = np.argsort(-val, axis=-1)
    idx = np.take_along_axis(part, order, axis=-1)
    val = np.take_along_axis(val, order, axis=-1)
    return val, idx


def gelu_tanh(x):
    """Apply the tanh approximation of GELU.

    This function agrees with torch.nn.functional.gelu(approximate="tanh").
    """
    if _COPS_READY:
        return _cops.gelu(x)
    x = np.asarray(x, dtype=np.float32)
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + 0.044715 * x * x * x)))


def softmax_mask(scores, positions, n_rep, base, window):
    """Apply the causal mask, the sliding window mask, and the softmax.

    scores is (kv_heads, tokens, heads_per_group, keys). positions gives the
    position of each query token. The kernel works in place and gives the
    probabilities of the last axis.
    """
    if _COPS_READY and scores.ndim == 4:
        scores = np.ascontiguousarray(scores, dtype=np.float32)
        positions = np.ascontiguousarray(positions, dtype=np.int32)
        _cops.softmax_mask(scores, positions, n_rep, base, window)
        return scores
    kpos = base + np.arange(scores.shape[-1])
    mask = kpos[None, :] <= positions[:, None]
    if window:
        mask &= (positions[:, None] - kpos[None, :]) < window
    scores = np.where(mask[None, :, None, :], scores, np.float32(-1e30))
    return softmax(scores, axis=-1)


def flash_ready():
    """Return True when the C flash attention kernel is ready."""
    return _cops is not None and _cops.available()


def flash_prefill(q, k, v, positions, base, window):
    """Run the C flash attention kernel for a prompt."""
    return _cops.attn_prefill(q, k, v, positions, base, window)


def _select_flash_impl():
    """Apply NP_GEMMA_ATTN_IMPL. A test and a benchmark use this."""
    if _cops is None or not _cops.available():
        return
    name = os.environ.get("NP_GEMMA_ATTN_IMPL", "").lower()
    which = {"": 0, "auto": 0, "c": 1, "scalar": 1, "avx2": 2, "avx512": 3}.get(name)
    if which is not None:
        _cops.attn_prefill_impl(which)


_select_flash_impl()


def softmax(x, axis=-1):
    """Change scores into probabilities.

    Subtract the maximum value before the exponent. This step prevents
    overflow.
    """
    x = np.asarray(x, dtype=np.float32)
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def softcap(logits, cap):
    """Limit the size of the logits. Apply tanh(logits / cap) * cap.

    The tanh is monotonic, so this step cannot change the choice of a greedy
    token. The sampling path needs the value, so the step stays.
    """
    if _COPS_READY:
        return _cops.softcap(logits, cap)
    return np.tanh(np.asarray(logits, dtype=np.float32) / cap) * cap
