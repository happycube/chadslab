"""Build and load small C kernels for the model.

This module does three tasks:
1. Compile one C file to a shared library.
2. Load the library with ctypes.
3. Multiply x by W with a C kernel.

ctypes is part of Python. Thus the model needs no new package. The model needs
only a C compiler. The module compiles the library one time. The library then
stays in the package directory.

The C kernels give an AVX-512 version and an AVX2 version. The C code selects
the version at run time. A CPU without AVX-512 uses the AVX2 version.

If the compiler is missing, the model uses the NumPy path or the Numba path.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from shutil import which

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "csrc" / "bf16_linear.c"
_LIB_DIR = _HERE / "_libs"
# The code builds two libraries. The first library uses an AVX2 baseline. The
# second library uses an AVX-512 baseline. The code loads the AVX-512 library
# when the CPU gives AVX-512. Otherwise, the code loads the AVX2 library.
_FLAGS_COMMON = ["-O3", "-funroll-loops", "-fopenmp", "-shared", "-fPIC", "-lm"]
_FLAGS = _FLAGS_COMMON + ["-mavx2", "-mfma"]
_FLAGS_AVX512 = _FLAGS_COMMON + ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mfma"]

_MAX_GEMM_TOKENS = 1024

_void_p = ctypes.c_void_p
_int = ctypes.c_int


def _compiler():
    """Return the name of a C compiler. Return None when no compiler is present."""
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if name and which(name):
            return name
    return None


def have_avx512():
    """Return True when the CPU gives AVX-512F and AVX-512BW."""
    try:
        import numpy.core._multiarray_umath as _m

        features = _m.__cpu_features__
        return bool(features.get("AVX512F")) and bool(features.get("AVX512BW"))
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            text = fh.read()
        return "avx512f" in text and "avx512bw" in text
    except OSError:
        return False


def _build(flags=None):
    """Build the shared library when necessary. Return the library path.

    The file name contains a hash. The hash covers the source text, the
    platform, the compiler version, and the flags. A new hash gives a new file.
    Thus the module does not use a library from a different machine.
    """
    flags = _FLAGS if flags is None else flags
    cc = _compiler()
    if cc is None or not _SRC.exists():
        return None
    try:
        version = subprocess.check_output([cc, "--version"], text=True, stderr=subprocess.STDOUT).splitlines()[0]
    except Exception:
        version = "unknown"
    key = hashlib.sha256(
        (_SRC.read_text() + "|" + sys.platform + "|" + platform.machine() + "|" + version + "|"
         + " ".join(flags)).encode()
    ).hexdigest()[:16]
    lib = _LIB_DIR / ("libgemma_" + key + ".so")
    if lib.exists():
        return lib
    _LIB_DIR.mkdir(parents=True, exist_ok=True)
    tmp = lib.with_suffix(".so.tmp")
    cmd = [cc] + flags + ["-o", str(tmp), str(_SRC)]
    subprocess.run(cmd, check=True, capture_output=True)
    tmp.replace(lib)
    return lib


# NP_GEMMA_ARCH=avx2 or avx512 forces one library. The default detects the CPU.
_ENV_ARCH = os.environ.get("NP_GEMMA_ARCH", "").lower()
AVX512 = True if _ENV_ARCH == "avx512" else (False if _ENV_ARCH == "avx2" else have_avx512())
# The smallest token group for the int4 tile. A smaller group uses the four-row
# dot. The AVX-512 tile masks a partial token block, so a small group is fine.
INT4_TILE_TOKENS = 8
_lib = None
try:
    _path = _build(_FLAGS_AVX512 if AVX512 else _FLAGS)
    if _path is not None:
        _lib = ctypes.CDLL(str(_path))
        _lib.gemma_bf16_linear.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_bf16_linear.restype = None
        _lib.gemma_bf16_gemm.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_bf16_gemm.restype = None
        _lib.gemma_int8_s8.argtypes = [_void_p, _void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int8_s8.restype = None
        _lib.gemma_int8_linear.argtypes = [_void_p, _void_p, _void_p, _void_p, _int, _int, _int, _int]
        _lib.gemma_int8_linear.restype = None
        _lib.gemma_int8_gemm.argtypes = [_void_p, _void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int8_gemm.restype = None
        _lib.gemma_int8_gemm_set_tokens_outer.argtypes = [_int]
        _lib.gemma_int8_gemm_set_tokens_outer.restype = None
        _lib.gemma_int8_gemm_set_kc.argtypes = [_int]
        _lib.gemma_int8_gemm_set_kc.restype = None
        _lib.gemma_int8_gemm_set_kv.argtypes = [_int]
        _lib.gemma_int8_gemm_set_kv.restype = None
        _lib.gemma_int8_gemm_set_panel.argtypes = [_int]
        _lib.gemma_int8_gemm_set_panel.restype = None
        _lib.gemma_int8_gemm_set_packed.argtypes = [_int]
        _lib.gemma_int8_gemm_set_packed.restype = None
        _lib.gemma_int8_gemm_set_ml.argtypes = [_int]
        _lib.gemma_int8_gemm_set_ml.restype = None
        _lib.gemma_int8_gemm_pw.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                            _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int8_gemm_pw.restype = None
        _lib.gemma_int8_gemm_blk.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                             _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int8_gemm_blk.restype = None
        _lib.gemma_int8_set_rows4.argtypes = [_int]
        _lib.gemma_int8_set_rows4.restype = None
        _lib.gemma_int4_set_rows4.argtypes = [_int]
        _lib.gemma_int4_set_rows4.restype = None
        _lib.gemma_int4_linear.argtypes = [_void_p, _void_p, _void_p, _void_p, _int, _int, _int, _int]
        _lib.gemma_int4_linear.restype = None
        _lib.gemma_int4_gemm.argtypes = [_void_p, _void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int4_gemm.restype = None
        _lib.gemma_int4_gemm_tile_run.argtypes = [_void_p, _void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int4_gemm_tile_run.restype = None
        _lib.gemma_q6k_linear.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_q6k_linear.restype = None
        _lib.gemma_int4_moe_gemv.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                             _int, _void_p, _int, _int, _int]
        _lib.gemma_int4_moe_gemv.restype = None
        _lib.gemma_attn_decode.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                           _void_p, _void_p, _void_p, _void_p,
                                           _int, _int, _int, _int]
        _lib.gemma_attn_decode.restype = None
        _lib.gemma_int4_multi4.argtypes = ([_void_p, _void_p, _void_p, _int] * 4) + [_void_p, _int]
        _lib.gemma_int4_multi4.restype = None
        _lib.gemma_rms_norm.argtypes = [_void_p, _void_p, _void_p, _int, _int, ctypes.c_float]
        _lib.gemma_rms_norm.restype = None
        _lib.gemma_gelu.argtypes = [_void_p, _void_p, _int]
        _lib.gemma_gelu.restype = None
        _lib.gemma_int8_pair.argtypes = [_void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int8_pair.restype = None
        _lib.gemma_int8_pf.argtypes = [_void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int8_pf.restype = None
        _lib.gemma_f32_linear.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_f32_linear.restype = None
except Exception:
    _lib = None

HAVE_C = _lib is not None


def available():
    """Return True when the C kernels are ready."""
    return HAVE_C


def set_gemm_ml(on):
    """Select the multi-level (cache-blocked) prompt GEMM."""
    if _lib is not None:
        _lib.gemma_int8_gemm_set_ml(1 if on else 0)


def set_gemm_packed(on):
    """Select the packed int8 weight layout for the prompt GEMM."""
    if _lib is not None:
        _lib.gemma_int8_gemm_set_packed(1 if on else 0)


def set_gemm_panel(on):
    """Select the float weight panel (True) or the int8 tile (False)."""
    if _lib is not None:
        _lib.gemma_int8_gemm_set_panel(1 if on else 0)


def set_gemm_kv(on):
    """Select the K-vectorized int8 GEMM tile (True) or the token-vectorized one."""
    if _lib is not None:
        _lib.gemma_int8_gemm_set_kv(1 if on else 0)


def set_gemm_kc(kc):
    """Select the K chunk size of the int8 GEMM. Use 0 to turn the block off."""
    if _lib is not None:
        _lib.gemma_int8_gemm_set_kc(int(kc))


def set_gemm_tokens_outer(on):
    """Select the token-outer (True) or row-outer (False) int8 GEMM order."""
    if _lib is not None:
        _lib.gemma_int8_gemm_set_tokens_outer(1 if on else 0)


def set_rows4(on):
    """Select the four-row loops (True) or the one-row loops (False).

    The four-row loop reads each x block one time for four weight rows. Use it
    for one token. It applies to the int8 kernel and the int4 kernel. Use False
    for a comparison.
    """
    if _lib is not None:
        _lib.gemma_int8_set_rows4(1 if on else 0)
        _lib.gemma_int4_set_rows4(1 if on else 0)


def _call(func, w, x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty((x.shape[0], w.shape[0]), dtype=np.float32)
    func(w.ctypes.data, x.ctypes.data, out.ctypes.data,
         ctypes.c_int(w.shape[0]), ctypes.c_int(w.shape[1]), ctypes.c_int(x.shape[0]))
    return out


def linear_bf16(x, w_u16):
    """Multiply x by W. W is raw bfloat16 data.

    Use the GEMM kernel for a long prompt. Use the GEMV kernel for one token or
    a short prompt.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    tokens = x.shape[0]
    if 1 < tokens <= _MAX_GEMM_TOKENS:
        xt = np.ascontiguousarray(x.T)
        out = np.empty((tokens, w_u16.shape[0]), dtype=np.float32)
        _lib.gemma_bf16_gemm(w_u16.ctypes.data, xt.ctypes.data, out.ctypes.data,
                             ctypes.c_int(w_u16.shape[0]), ctypes.c_int(w_u16.shape[1]),
                             ctypes.c_int(tokens))
        return out
    return _call(_lib.gemma_bf16_linear, w_u16, x)


def quantize_activations_s8(x):
    """Quantize x to int8. Return the int8 data and one float32 scale for each row.

    The formula for one row is scale = max(abs(row)) / 127.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    amax = np.max(np.abs(x), axis=1)
    scale = np.where(amax > 0.0, amax / 127.0, 1e-12).astype(np.float32)
    q = np.rint(x / scale[:, None]).clip(-127.0, 127.0).astype(np.int8)
    return q, scale


def linear_int8_float(x, q_i8, scales, group):
    """Multiply x by W. W is int8 data. Keep x as float32.

    Use the C kernel when the C path is available. Otherwise, dequantize W and
    use NumPy.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty((x.shape[0], q_i8.shape[0]), dtype=np.float32)
    _lib.gemma_int8_linear(q_i8.ctypes.data, scales.ctypes.data, x.ctypes.data, out.ctypes.data,
                           ctypes.c_int(q_i8.shape[0]), ctypes.c_int(q_i8.shape[1]),
                           ctypes.c_int(x.shape[0]), ctypes.c_int(group))
    return out


def linear_int8_gemm(x, xt, q_i8, scales):
    """Multiply x by W for a prompt of many tokens. W is int8 data.

    x is (tokens, cols). xt is x transposed, that is (cols, tokens). The kernel
    reads each x block one time for several weight rows.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    xt = np.ascontiguousarray(xt, dtype=np.float32)
    out = np.empty((x.shape[0], q_i8.shape[0]), dtype=np.float32)
    _lib.gemma_int8_gemm(q_i8.ctypes.data, scales.ctypes.data, x.ctypes.data,
                         xt.ctypes.data, out.ctypes.data,
                         ctypes.c_int(q_i8.shape[0]), ctypes.c_int(q_i8.shape[1]),
                         ctypes.c_int(x.shape[0]))
    return out


def linear_int8_gemm_pw(x, xt, q_i8, packed, scales):
    """Multiply x by W. Use the packed copy of W. packed is (cols, rows)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    xt = np.ascontiguousarray(xt, dtype=np.float32)
    out = np.empty((x.shape[0], q_i8.shape[0]), dtype=np.float32)
    _lib.gemma_int8_gemm_pw(q_i8.ctypes.data, packed.ctypes.data, scales.ctypes.data,
                            x.ctypes.data, xt.ctypes.data, out.ctypes.data,
                            ctypes.c_int(q_i8.shape[0]), ctypes.c_int(q_i8.shape[1]),
                            ctypes.c_int(x.shape[0]))
    return out


def linear_int8_gemm_blk(x, xt, q_i8, packed, scales):
    """Multiply x by W. Use the blocked packed copy of W."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    xt = np.ascontiguousarray(xt, dtype=np.float32)
    out = np.empty((x.shape[0], q_i8.shape[0]), dtype=np.float32)
    _lib.gemma_int8_gemm_blk(q_i8.ctypes.data, packed.ctypes.data, scales.ctypes.data,
                             x.ctypes.data, xt.ctypes.data, out.ctypes.data,
                             ctypes.c_int(q_i8.shape[0]), ctypes.c_int(q_i8.shape[1]),
                             ctypes.c_int(x.shape[0]))
    return out


def linear_int8_s8(x, q_i8, scales):
    """Multiply x by W. W is int8 data with one scale for each row.

    The code quantizes x to int8. The C kernel then uses integer multiply and
    add. Thus the kernel does not convert the values to float32.
    """
    qx, sx = quantize_activations_s8(x)
    out = np.empty((qx.shape[0], q_i8.shape[0]), dtype=np.float32)
    _lib.gemma_int8_s8(q_i8.ctypes.data, scales.ctypes.data, qx.ctypes.data, sx.ctypes.data,
                       out.ctypes.data, ctypes.c_int(q_i8.shape[0]), ctypes.c_int(q_i8.shape[1]),
                       ctypes.c_int(qx.shape[0]))
    return out


def linear_int8_pair(x, q_i8, scales):
    """Multiply x by W with the two-row int8 kernel. Use this for a test."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, cols = q_i8.shape
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    _lib.gemma_int8_pair(q_i8.ctypes.data, scales.ctypes.data, x.ctypes.data, out.ctypes.data,
                         ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(x.shape[0]))
    return out


def linear_int8_pf(x, q_i8, scales):
    """Multiply x by W with the prefetch int8 kernel. Use this for a test."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows = q_i8.shape[0]
    cols = q_i8.shape[1]
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    _lib.gemma_int8_pf(q_i8.ctypes.data, scales.ctypes.data, x.ctypes.data, out.ctypes.data,
                       ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_int(x.shape[0]))
    return out


def linear_int4(x, packed, scales, group):
    """Multiply x by W. W is packed 4-bit data. scales gives one scale for each group."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows = packed.shape[0]
    cols = packed.shape[1] * 32
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    _lib.gemma_int4_linear(packed.ctypes.data, scales.ctypes.data, x.ctypes.data, out.ctypes.data,
                           ctypes.c_int(rows), ctypes.c_int(cols),
                           ctypes.c_int(x.shape[0]), ctypes.c_int(group))
    return out


def linear_q6k(x, w_bytes, cols):
    """Multiply x by W. W is the raw Q6_K block data of a 2-D tensor.

    w_bytes has shape (rows, blocks in one row * 210). cols is the value count
    in one row. The kernel decodes the blocks.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows = w_bytes.shape[0]
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    _lib.gemma_q6k_linear(w_bytes.ctypes.data, x.ctypes.data, out.ctypes.data,
                          ctypes.c_int(rows), ctypes.c_int(cols),
                          ctypes.c_int(x.shape[0]))
    return out


def linear_int4_gemm(x, xt, packed, scales, group):
    """Multiply x by W for a prompt. W is packed 4-bit data.

    The kernel decodes a row block to float32 one time and reuses it for every
    token block. xt is x transposed, that is (cols, tokens).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    xt = np.ascontiguousarray(xt, dtype=np.float32)
    rows = packed.shape[0]
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    _lib.gemma_int4_gemm(packed.ctypes.data, scales.ctypes.data, x.ctypes.data,
                         xt.ctypes.data, out.ctypes.data,
                         ctypes.c_int(rows), ctypes.c_int(x.shape[1]),
                         ctypes.c_int(x.shape[0]))
    return out


def linear_int4_tile(x, xt, packed, scales, group):
    """Multiply x by W for a small group of tokens. W is packed 4-bit data.

    The tile reads the x block one time for several weight rows. Use it for a
    group of tokens that is smaller than the token block of the multi-level
    GEMM. xt is x transposed, that is (cols, tokens).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    xt = np.ascontiguousarray(xt, dtype=np.float32)
    rows = packed.shape[0]
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    _lib.gemma_int4_gemm_tile_run(packed.ctypes.data, scales.ctypes.data, x.ctypes.data,
                                  xt.ctypes.data, out.ctypes.data,
                                  ctypes.c_int(rows), ctypes.c_int(x.shape[1]),
                                  ctypes.c_int(x.shape[0]))
    return out


def linear_f32(x, w):
    """Multiply x by W. W is float32 data. Use the C kernel."""
    return _call(_lib.gemma_f32_linear, w, x)


def int4_multi4(mats, x, cols):
    """Run up to four int4 matrices on the same one-row x.

    mats is a list of up to four (packed, scales) pairs. A None entry skips a
    matrix. Return a list of float32 outputs, or None for a skipped matrix.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    args = []
    outs = []
    for i in range(4):
        if i < len(mats) and mats[i] is not None:
            w, s = mats[i]
            o = np.empty(w.shape[0], dtype=np.float32)
            args += [w.ctypes.data, s.ctypes.data, o.ctypes.data, ctypes.c_int(w.shape[0])]
            outs.append(o)
        else:
            args += [None, None, None, ctypes.c_int(0)]
            outs.append(None)
    args += [x.ctypes.data, ctypes.c_int(cols)]
    _lib.gemma_int4_multi4(*args)
    return outs


def attn_decode(qq, qs, kq, ks, vq, vs, scores, out,
                q_heads, kv_heads, head_dim, n):
    """Run the fused attention for one query token. All arrays must be ready."""
    _lib.gemma_attn_decode(qq.ctypes.data, qs.ctypes.data, kq.ctypes.data, ks.ctypes.data,
                           vq.ctypes.data, vs.ctypes.data, scores.ctypes.data, out.ctypes.data,
                           ctypes.c_int(q_heads), ctypes.c_int(kv_heads),
                           ctypes.c_int(head_dim), ctypes.c_int(n))


def int4_moe_gemv(w, scales, x, ids, rows, cols, xstride):
    """Multiply each selected expert matrix by its input row.

    w has the shape (experts, rows, groups, 18). scales has the shape
    (experts, rows, groups). ids gives the selected experts. x has one row for
    each job, with a stride of xstride. A stride of 0 gives the same x to each
    job. Return (jobs, rows).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    jobs = int(ids.size)
    out = np.empty((jobs, rows), dtype=np.float32)
    _lib.gemma_int4_moe_gemv(w.ctypes.data, scales.ctypes.data, x.ctypes.data,
                             ids.ctypes.data, ctypes.c_int(jobs), out.ctypes.data,
                             ctypes.c_int(rows), ctypes.c_int(cols),
                             ctypes.c_int(xstride))
    return out


def rms_norm(x, w, eps):
    """Normalize the last axis of x. Multiply by the weight w. w may be None."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows, cols = x.shape
    out = np.empty_like(x)
    wp = None if w is None else w.ctypes.data
    _lib.gemma_rms_norm(x.ctypes.data, wp, out.ctypes.data,
                        ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_float(eps))
    return out


def gelu(x):
    """Apply the tanh approximation of GELU to every value of x."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x)
    _lib.gemma_gelu(x.ctypes.data, out.ctypes.data, ctypes.c_int(x.size))
    return out
