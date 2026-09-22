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


def rms_norm(x, weight=None, eps=1e-6):
    """Normalize the last axis of x. Multiply by the weight.

    The formula is x * pow(mean(x * x) + eps, -0.5) * weight.

    Note: the weight is the full scale. Do not add 1 to the weight.
    """
    x32 = np.asarray(x, dtype=np.float32)
    if _cops is not None and _cops.available() and x32.ndim >= 1:
        shape = x32.shape
        x2 = np.ascontiguousarray(x32).reshape(-1, shape[-1])
        w2 = None if weight is None else np.ascontiguousarray(weight, dtype=np.float32)
        return _cops.rms_norm(x2, w2, eps).reshape(shape)
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
    mode = os.environ.get("NP_GEMMA_KERNEL", "auto").lower()
    if mode != "numpy":
        if mode in ("auto", "c") and _cops is not None and _cops.available():
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
    mode = os.environ.get("NP_GEMMA_KERNEL", "auto").lower()
    if mode != "numpy" and _cops is not None and _cops.available():
        return _cops.linear_q6k(x, w_bytes, cols)
    return linear_q6k_numpy(x, w_bytes, cols)


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


def int4_moe_ready():
    """Return True when the fused expert kernel is ready."""
    return _cops is not None and _cops.available()


def int4_moe_gemv(w, scales, x, ids, rows, cols, xstride):
    """Multiply each selected expert matrix by its input row.

    Use the fused kernel when the C library is ready. Otherwise, run one call
    for each expert.
    """
    if _cops is not None and _cops.available():
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


# Use the int4 prompt GEMM for this many tokens or more. The GEMM decodes a
# row block to float32 one time and reuses it for every token block. The one
# row dot decodes the weights again for each token. Below one full token block
# the GEMM does no work, so use the one-row dot.
_INT4_GEMM_TOKENS = 64


def linear_int4(x, packed, scales):
    """Multiply x by W. W is packed 4-bit data.

    Use the C kernel when the C path is available. Otherwise, dequantize W and
    use NumPy.
    """
    mode = os.environ.get("NP_GEMMA_KERNEL", "auto").lower()
    if mode != "numpy" and _cops is not None and _cops.available():
        group = int4_group(packed, scales)
        # The C kernel uses the block-32 layout. Use NumPy for another group.
        if group == INT4_GROUP:
            tokens = x.shape[0]
            if tokens >= _INT4_GEMM_TOKENS:
                xt = np.ascontiguousarray(x.T)
                return _cops.linear_int4_gemm(x, xt, packed, scales, group)
            if tokens >= _cops.INT4_TILE_TOKENS:
                # A small group of tokens. Use the token-vectorized tile. It
                # reads the x block one time for several weight rows.
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
    mode = os.environ.get("NP_GEMMA_KERNEL", "auto").lower()
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
    x = np.asarray(x, dtype=np.float32)
    if _cops is not None and _cops.available():
        return _cops.gelu(x)
    return 0.5 * x * (1.0 + np.tanh(GELU_C * (x + 0.044715 * x * x * x)))


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
    """Limit the size of the logits. Apply tanh(logits / cap) * cap."""
    return np.tanh(np.asarray(logits, dtype=np.float32) / cap) * cap
