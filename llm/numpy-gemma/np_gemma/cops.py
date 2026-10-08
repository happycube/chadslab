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
import time
import sys
from pathlib import Path
from shutil import which

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE / "csrc" / "bf16_linear.c"
# The files that bf16_linear.c includes; the hash of the library covers them.
_SOURCES = [_SRC] + [_HERE / "csrc" / n for n in ("moe.c", "mlx_affine.c", "kquants.c", "deltanet.c",
                                                 "hyperconn.c", "qsa.c", "tq6_tables.h")]
_LIB_DIR = _HERE / "_libs"
# The code builds three libraries. The first library uses an AVX2 baseline. The
# second library uses an AVX-512 baseline. The third adds the VNNI
# instruction. The code loads the fastest library that the CPU gives.
# --wrap=GOMP_parallel: every parallel region passes __wrap_GOMP_parallel
# (csrc/bf16_linear.c), which reports the teams outside the planned ones
# (team_warn)
_FLAGS_COMMON = ["-O3", "-funroll-loops", "-fopenmp", "-shared", "-fPIC", "-lm",
                 "-Wl,--wrap=GOMP_parallel"]
# F16C: every CPU with AVX2 has it, and the float16 scales use it.
_FLAGS = _FLAGS_COMMON + ["-mavx2", "-mfma", "-mf16c"]
_FLAGS_AVX512 = _FLAGS_COMMON + ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mfma", "-mf16c"]
_FLAGS_VNNI = _FLAGS_COMMON + ["-mavx512f", "-mavx512bw", "-mavx512vl",
                               "-mavx512vnni", "-mfma", "-mf16c"]

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


def have_vnni():
    """Return True when the CPU gives AVX-512 VNNI."""
    try:
        import numpy.core._multiarray_umath as _m

        return bool(_m.__cpu_features__.get("AVX512VNNI"))
    except Exception:
        pass
    try:
        with open("/proc/cpuinfo") as fh:
            return "avx512_vnni" in fh.read()
    except OSError:
        return False


def lib_ready(lib):
    """lib exists: mark it used now (its mtime orders the sweep). Return True
    when it exists."""
    try:
        os.utime(lib)
        return True
    except OSError:
        return False


def sweep_libs(lib, prefix, keep):
    """Keep the newest keep libraries of the prefix next to lib
    (NP_GEMMA_LIBS_KEEP, else keep; lib always stays), and remove the
    temporary files of more than an hour. A process that loaded a removed
    library keeps its mapping."""
    keep = int(os.environ.get("NP_GEMMA_LIBS_KEEP", keep))
    try:
        libs = sorted((p for p in lib.parent.glob(prefix + "*.so")
                       if p.name[len(prefix):-3].isalnum() and len(p.name) == len(lib.name)),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for p in libs[keep:]:
            if p != lib:
                p.unlink()
        now = time.time()
        for p in lib.parent.glob("*.tmp"):
            if now - p.stat().st_mtime > 3600:
                p.unlink()
    except OSError:
        pass


def build_lib(cmd_of, lib, prefix, keep):
    """Build lib with the command cmd_of(tmp) and return it. The temporary
    file has the pid in its name, so two processes that build the same lib
    do not write one file; the rename is atomic. Then sweep_libs."""
    lib.parent.mkdir(parents=True, exist_ok=True)
    tmp = lib.with_name("%s.%d.tmp" % (lib.name, os.getpid()))
    try:
        subprocess.run(cmd_of(tmp), check=True, capture_output=True)
        os.replace(tmp, lib)
    finally:
        if tmp.exists():
            tmp.unlink()
    sweep_libs(lib, prefix, keep)
    return lib


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
        ("".join(p.read_text() for p in _SOURCES) + "|" + sys.platform + "|" + platform.machine() + "|" + version + "|"
         + " ".join(flags)).encode()
    ).hexdigest()[:16]
    lib = _LIB_DIR / ("libgemma_" + key + ".so")
    if lib_ready(lib):
        return lib
    # three builds (AVX2, AVX-512, VNNI) for each version of the sources
    return build_lib(lambda tmp: [cc] + flags + ["-o", str(tmp), str(_SRC)], lib, "libgemma_", 24)


# NP_GEMMA_ARCH=avx2 or avx512 forces one library. The default detects the CPU.
_ENV_ARCH = os.environ.get("NP_GEMMA_ARCH", "").lower()
if _ENV_ARCH == "vnni":
    VNNI = True
elif _ENV_ARCH in ("avx512", "avx2"):
    VNNI = False
else:
    VNNI = have_vnni()
AVX512 = True if _ENV_ARCH in ("avx512", "vnni") else (
    False if _ENV_ARCH == "avx2" else have_avx512())
# The smallest token group for the int4 tile. A smaller group uses the four-row
# dot. The AVX-512 tile masks a partial token block, so a small group is fine.
# The smallest token group that uses the int4 tile. The one-row dot decodes
# the weights again for each token. The tile decodes a weight block one time
# for a group of tokens. A mixture-of-experts layer gives a small group of
# tokens to each expert, so the tile is the better kernel for it.
INT4_TILE_TOKENS = int(os.environ.get("NP_GEMMA_INT4_TILE_TOKENS", "2"))
_lib = None
try:
    _path = _build(_FLAGS_VNNI if VNNI else (_FLAGS_AVX512 if AVX512 else _FLAGS))
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
        _lib.gemma_quantize_q8_groups.argtypes = [_void_p, _void_p, _void_p, _void_p, _int, _int]
        _lib.gemma_quantize_q8_groups.restype = None
        _lib.gemma_quantize_q8_t.argtypes = [_void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_quantize_q8_t.restype = None
        _lib.gemma_int4_q8_tile_run.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                                _void_p, _void_p, _int, _int, _int, _int]
        _lib.gemma_int4_q8_tile_run.restype = None
        _lib.gemma_int4_q8_tile_run32.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                                  _void_p, _void_p, _int, _int, _int, _int]
        _lib.gemma_int4_q8_tile_run32.restype = None
        _lib.gemma_int4_q8_gemv.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                            _void_p, _void_p, _int, _int]
        _lib.gemma_int4_q8_gemv.restype = None
        _lib.gemma_int4_q8_gemv_x.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                              _int, _int]
        _lib.gemma_int4_q8_gemv_x.restype = None
        _lib.gemma_int4_q8_multi4.argtypes = (
            [_void_p, _void_p, _void_p, _int] * 4) + [_void_p, _int]
        _lib.gemma_int4_q8_multi4.restype = None
        _lib.gemma_int4_q8_moe_gemv.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                                _int, _void_p, _int, _int, _int]
        _lib.gemma_int4_q8_moe_gemv.restype = None
        _lib.gemma_quantize_q8_t_moe.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                                 _void_p, _int, _int, _void_p, _void_p,
                                                 _int]
        _lib.gemma_quantize_q8_t_moe.restype = None
        _lib.gemma_gelu_mul.argtypes = [_void_p, _void_p, _int, _int]
        _lib.gemma_gelu_mul.restype = None
        _lib.gemma_moe_scatter.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                           _int, _int]
        _lib.gemma_moe_scatter.restype = None
        _lib.gemma_softmax_mask.argtypes = [_void_p, _int, _int, _void_p, _int,
                                           _int, _int, _int]
        _lib.gemma_softmax_mask.restype = None
        _lib.gemma_int4_q8_moe_run.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                               _void_p, _void_p, _int, _int, _int,
                                               _void_p, _void_p, _void_p, _int]
        _lib.gemma_int4_q8_moe_run.restype = None
        _lib.gemma_int4_q8_set_tb8.argtypes = [_int]
        _lib.gemma_int4_q8_set_tb8.restype = None
        _lib.gemma_int4_q8_set_prefetch.argtypes = [_int]
        _lib.gemma_int4_q8_set_prefetch.restype = None
        _lib.gemma_q6k_linear.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_q6k_linear.restype = None
        _lib.gemma_q6k_rows.argtypes = [_void_p, _void_p, _int, _int, _void_p]
        _lib.gemma_q6k_rows.restype = None
        _lib.gemma_kq45_rows.argtypes = [_void_p, _void_p, _int, _int, _int, _void_p]
        _lib.gemma_kq45_rows.restype = None
        _lib.gemma_q4_0_rows.argtypes = [_void_p, _void_p, _int, _int, _void_p]
        _lib.gemma_q4_0_rows.restype = None
        _lib.gemma_argmax.argtypes = [_void_p, ctypes.c_int64]
        _lib.gemma_profile.argtypes = [_void_p, _void_p]
        _lib.ma_quant_x.argtypes = [_void_p, _int, _int, _int, _void_p, _void_p, _void_p]
        _lib.ma_quant_x.restype = None
        _lib.ma_linear.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int, _void_p, _void_p,
                                   _void_p, _int, _void_p]
        _lib.ma_linear.restype = None
        _lib.ma_moe.argtypes = [_void_p] * 6 + [_int, _int, _int, _void_p, _void_p, _int, _int,
                                                 _void_p, _void_p]
        _lib.ma_moe.restype = None
        _lib.ma_moe_scratch.argtypes = [_int] * 5
        _lib.ma_moe_scratch.restype = ctypes.c_size_t
        _lib.gdn_step.argtypes = [_void_p, _void_p, _void_p, _int] + [_void_p] * 9 + \
            [_int] * 5 + [ctypes.c_float, _void_p, _int]
        _lib.gdn_step.restype = None
        _lib.kq_quant_x.argtypes = [_void_p, _int, _int, _void_p, _void_p, _void_p]
        _lib.kq_quant_x.restype = None
        _lib.kq_linear.argtypes = [_void_p, _int, _int, _int] + [_void_p] * 4 + [_int, _void_p]
        _lib.kq_linear.restype = None
        _lib.kq_rows.argtypes = [_void_p, _int, _int, _void_p, _int, _void_p]
        _lib.kq_rows.restype = None
        _lib.kq_to_q8_0.argtypes = [_void_p, _int, ctypes.c_int64, _int, _void_p]
        _lib.kq_to_q8_0.restype = None
        _lib.kq_to_q6_k.argtypes = [_void_p, _int, ctypes.c_int64, _int, _void_p]
        _lib.kq_to_q6_k.restype = None
        _lib.kq_bf16_to_bf12.argtypes = [_void_p, ctypes.c_int64, _int, _void_p, _void_p]
        _lib.kq_bf16_to_bf12.restype = None
        _lib.kq_bf12_to_bf16.argtypes = [_void_p, ctypes.c_int64, _int, _void_p]
        _lib.kq_bf12_to_bf16.restype = None
        _lib.kq_pack_bf12x16.argtypes = [_void_p, ctypes.c_int64, _int, _void_p]
        _lib.kq_pack_bf12x16.restype = None
        _lib.kq_nv4_pack.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int, _void_p]
        _lib.kq_nv4_pack.restype = None
        _lib.kq_nvx_pack.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int, _void_p]
        _lib.kq_nvx_pack.restype = None
        _lib.kq_pack_x16f.argtypes = [_void_p, _int, ctypes.c_int64, _int, _void_p]
        _lib.kq_pack_x16f.restype = None
        _lib.kq_q4x_pack.argtypes = [_void_p, _void_p, ctypes.c_int64, _int, _void_p]
        _lib.kq_q4x_pack.restype = _int
        _lib.kq_moe_act.argtypes = [_void_p] * 5 + [_int] * 3 + [_void_p, _void_p, _int, _int,
                                                                _void_p, _void_p, _int, _void_p]
        _lib.kq_moe_act.restype = None
        _lib.kq_pack_q8x16.argtypes = [_void_p, ctypes.c_int64, _int, _void_p]
        _lib.kq_pack_q8x16.restype = None
        _lib.kq_gather.argtypes = [_void_p, ctypes.c_int64, _int, _void_p]
        _lib.kq_gather.restype = None
        _lib.kq_memcpy_par.argtypes = [_void_p, _void_p, ctypes.c_int64]
        _lib.kq_memcpy_par.restype = None
        _lib.kq_set_node0_share.argtypes = [ctypes.c_float]
        _lib.kq_set_node0_share.restype = None
        _lib.kq_set_moe_rot.argtypes = [ctypes.c_int]
        _lib.kq_set_moe_rot.restype = None
        _lib.kq_get_node0_share.argtypes = []
        _lib.kq_get_node0_share.restype = ctypes.c_float
        _lib.kq_calib_nodes.argtypes = [_void_p, _void_p] + [_int] * 7 + [_void_p]
        _lib.kq_calib_nodes.restype = None
        _lib.kq_moe.argtypes = [_void_p] * 5 + [_int, _int, _int, _void_p, _void_p, _int, _int,
                                                 _void_p, _void_p]
        _lib.kq_moe.restype = None
        _lib.kq_moe_scratch.argtypes = [_int] * 5
        _lib.kq_moe_scratch.restype = ctypes.c_size_t
        _lib.gdn_commit.argtypes = [_void_p, _void_p, _void_p] + [_int] * 6
        _lib.gdn_commit.restype = None
        _lib.gdn_log_floats.argtypes = [_int] * 5
        _lib.gdn_log_floats.restype = ctypes.c_size_t
        _lib.gemma_argmax.restype = ctypes.c_int64
        _lib.gemma_int4_moe_gemv.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                             _int, _void_p, _int, _int, _int]
        _lib.gemma_int4_moe_gemv.restype = None
        _lib.gemma_attn_decode_f32.argtypes = [_void_p, _void_p, _void_p,
                                               _void_p, _void_p, _int, _int,
                                               _int, _int, ctypes.c_long,
                                               ctypes.c_long, _int, _int, _int]
        _lib.gemma_attn_decode_f32.restype = None
        _lib.gemma_attn_decode.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                           _void_p, _void_p, _void_p, _void_p,
                                           _int, _int, _int, _int]
        _lib.gemma_attn_decode.restype = None
        _lib.gemma_int4_multi4.argtypes = ([_void_p, _void_p, _void_p, _int] * 4) + [_void_p, _int]
        _lib.gemma_int4_multi4.restype = None
        _lib.gemma_router.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                      _int, _int, _int, ctypes.c_float, ctypes.c_float,
                                      _void_p, _void_p]
        _lib.gemma_router.restype = None
        _lib.gemma_qkv_norm.argtypes = [_void_p, _void_p, _int, _void_p, _void_p, _int,
                                        _void_p, _int, _int, ctypes.c_float]
        _lib.gemma_qkv_norm.restype = None
        _lib.gemma_rope.argtypes = [_void_p, _int, _int, _void_p, _int, _int,
                                    _void_p, _void_p, _int]
        _lib.gemma_rope.restype = None
        _lib.gemma_rms_norm.argtypes = [_void_p, _void_p, _void_p, _int, _int, ctypes.c_float]
        _lib.gemma_rms_norm.restype = None
        _lib.gemma_gelu.argtypes = [_void_p, _void_p, _int]
        _lib.gemma_gelu.restype = None
        _lib.gemma_softcap.argtypes = [_void_p, _void_p, _int, ctypes.c_float]
        _lib.gemma_softcap.restype = None
        _lib.gemma_qkv_norm_rope.argtypes = [
            _void_p, _void_p, _int, _void_p, _void_p, _int, _void_p, _int,
            _void_p, _void_p, _int, _int, _int, ctypes.c_float]
        _lib.gemma_qkv_norm_rope.restype = None
        _lib.gemma_rms_norm_multi4.argtypes = (
            [_void_p, _void_p, _void_p, _int, ctypes.c_float]
            + [_void_p, _void_p, _void_p, _int] * 4)
        _lib.gemma_rms_norm_multi4.restype = None
        _lib.gemma_gelu_mul_int4.argtypes = [
            _void_p, _void_p, _int, _void_p, _void_p, _void_p, _void_p,
            _int, _int]
        _lib.gemma_gelu_mul_int4.restype = None
        _lib.gemma_moe_gemv_gelu.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                             _int, _void_p, _void_p, _int, _int,
                                             _int, _int]
        _lib.gemma_moe_gemv_gelu.restype = None
        _lib.gemma_attn_decode_mt.argtypes = [_void_p] * 8 + [_int, _int, _int,
                                                               _void_p, _void_p, _int, _int]
        _lib.gemma_attn_decode_mt.restype = None
        _lib.gemma_attn_decode_f32s.argtypes = [_void_p] * 5 + [_int] * 7
        _lib.gemma_attn_decode_f32s.restype = None
        _lib.gemma_quantize_i16_groups.argtypes = [_void_p, _void_p, _void_p, ctypes.c_long]
        _lib.gemma_quantize_i16_groups.restype = None
        _lib.gemma_quantize_i8_groups.argtypes = [_void_p, _void_p, _void_p, ctypes.c_long]
        _lib.gemma_quantize_i8_groups.restype = None
        _lib.gemma_dequantize_i8_groups.argtypes = [_void_p, _void_p, _void_p, ctypes.c_long]
        _lib.gemma_dequantize_i8_groups.restype = None
        for nm in ("gemma_tq6_quantize", "gemma_tq6_dequantize_rotated"):
            getattr(_lib, nm).argtypes = [_void_p, _void_p, _void_p, ctypes.c_long]
            getattr(_lib, nm).restype = None
        _lib.gemma_tq6_rotate.argtypes = [_void_p, ctypes.c_long, _int]
        _lib.gemma_tq6_rotate.restype = None
        _lib.gemma_attn_decode_q8.argtypes = [_void_p] * 7 + [_int] * 3 + [_void_p, _void_p,
                                                                         _int, _int, _int]
        _lib.gemma_attn_decode_q8.restype = _int
        _lib.gemma_dequantize_i16_groups.argtypes = [_void_p, _void_p, _void_p, ctypes.c_long]
        _lib.gemma_dequantize_i16_groups.restype = None
        _lib.gemma_attn_decode_i16.argtypes = [_void_p] * 7 + [_int] * 4
        _lib.gemma_attn_decode_i16.restype = None
        _lib.gemma_attn_decode_i16_mt.argtypes = [_void_p] * 7 + [_int] * 3 + [
            _void_p, _void_p, _int, _int]
        _lib.gemma_attn_decode_i16_mt.restype = None
        _lib.gemma_quantize_q16_t.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_quantize_q16_t.restype = None
        _lib.gemma_int4_q16_tile_run.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                                 _void_p, _int, _int, _int, _int]
        _lib.gemma_int4_q16_tile_run.restype = None
        _lib.gemma_gather_t_moe.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_gather_t_moe.restype = None
        _lib.gemma_int4_f32_moe_run.argtypes = [_void_p, _void_p, _void_p, _void_p, _int,
                                                _int, _int, _void_p, _void_p, _void_p, _int]
        _lib.gemma_int4_f32_moe_run.restype = None
        _lib.gemma_quantize_q16_t_moe.argtypes = [_void_p, _void_p, _void_p, _void_p, _int,
                                                  _int, _int]
        _lib.gemma_quantize_q16_t_moe.restype = None
        _lib.gemma_int4_q16_moe_run.argtypes = [_void_p, _void_p, _void_p, _void_p, _void_p,
                                                _int, _int, _int, _void_p, _void_p, _void_p,
                                                _int]
        _lib.gemma_int4_q16_moe_run.restype = None
        _lib.gemma_run.argtypes = [_void_p, _int]
        _lib.gemma_run.restype = _int
        _lib.gemma_run_parts.argtypes = [_void_p, _int, _int, _void_p]
        _lib.gemma_run_parts.restype = _int
        _lib.gemma_run_parts_prof.argtypes = [_void_p, _int, _int, _void_p, _void_p, _int]
        _lib.gemma_run_parts_prof.restype = _int
        _lib.gemma_xbar_stats.argtypes = [_void_p]
        _lib.gemma_xbar_stats.restype = _int
        _lib.kq_linear16.argtypes = [_void_p, _int, _int, _void_p, _void_p, _void_p, _int, _void_p]
        _lib.kq_linear16.restype = None
        _lib.kq_q16_ok.argtypes = []
        _lib.kq_q16_ok.restype = _int
        _lib.gemma_part_cpus.argtypes = [_int, _int, _void_p]
        _lib.gemma_part_cpus.restype = _int
        _lib.gemma_gp_record_size.argtypes = []
        _lib.gemma_gp_record_size.restype = _int
        _lib.gemma_router_mt.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                         _int, _int, _int, ctypes.c_float, ctypes.c_float,
                                         _void_p, _void_p, _int]
        _lib.gemma_router_mt.restype = None
        _lib.gemma_gelu_mul_pair.argtypes = [_void_p, _void_p, _void_p, _int]
        _lib.gemma_gelu_mul_pair.restype = None
        _lib.gemma_int4_linear_mt.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                              _int, _int, _int]
        _lib.gemma_int4_linear_mt.restype = None
        _lib.gemma_int4_multi4_mt.argtypes = (
            [_void_p, _void_p, _void_p, _int] * 4) + [_void_p, _int, _int]
        _lib.gemma_int4_multi4_mt.restype = None
        _lib.gemma_int4_moe_gemv_mt.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                                _void_p, _void_p, _int, _void_p,
                                                _int, _int, _int]
        _lib.gemma_int4_moe_gemv_mt.restype = None
        _lib.gemma_moe_gemv_gelu_mt.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                                _void_p, _void_p, _int, _void_p,
                                                _void_p, _int, _int, _int, _int]
        _lib.gemma_moe_gemv_gelu_mt.restype = None
        _lib.gemma_int8_pair.argtypes = [_void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int8_pair.restype = None
        _lib.gemma_int8_pf.argtypes = [_void_p, _void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_int8_pf.restype = None
        _lib.gemma_f32_linear.argtypes = [_void_p, _void_p, _void_p, _int, _int, _int]
        _lib.gemma_f32_linear.restype = None
        _lib.gemma_ct_linear.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                         _int, _int, _int, _int]
        _lib.gemma_ct_linear.restype = None
        _lib.gemma_ct_set_rows4.argtypes = [_int]
        _lib.gemma_ct_set_rows4.restype = None
except Exception:
    _lib = None

HAVE_C = _lib is not None


def available():
    """Return True when the C kernels are ready."""
    return HAVE_C


def set_int4_prefetch(on):
    """Select the prefetch of the next weight group in the int8 tile. A test."""
    if _lib is not None:
        _lib.gemma_int4_q8_set_prefetch(1 if on else 0)


def set_int4_q8_tb8(on):
    """Select the narrow (eight token) block of the int8 tile."""
    if _lib is not None:
        _lib.gemma_int4_q8_set_tb8(1 if on else 0)


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


def set_ct_rows4(on):
    """Select the four-row loop (1) or the one-row loop (0) of ct_linear."""
    _lib.gemma_ct_set_rows4(1 if on else 0)


def ct_linear(x, packed, scale, bits, cols=None):
    """Multiply x by W. W is a weight of a compressed-tensors file, packed.

    This is the layout the E4B mobile-ct checkpoint uses, and it is not the
    layout of linear_int4. packed holds the int32 words of the row, and scale
    holds one float32 value for each row (the "channel" strategy). The kernel
    reads the words in place, so the caller keeps the weights packed and no
    float32 copy is ever built.

    bits is 4 or 2. cols is the value count of one row; when it is None the
    value count follows from the packed shape.

    x is (tokens, cols) and the result is (tokens, rows).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    packed = np.asarray(packed)
    if packed.dtype != np.int32:
        packed = packed.view(np.int32)
    if not packed.flags.c_contiguous:
        packed = np.ascontiguousarray(packed)
    rows, words = packed.shape
    covered = words * 32 // bits
    if cols is None:
        cols = covered
    elif cols != covered:
        raise ValueError("the packed row holds %d values, not %d" % (covered, cols))
    if cols % 16:
        raise ValueError("cols must be a multiple of 16, got %d" % cols)
    scale = np.ascontiguousarray(scale, dtype=np.float32).reshape(-1)
    if scale.shape[0] != rows:
        raise ValueError("scale has %d values for %d rows" % (scale.shape[0], rows))
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    _lib.gemma_ct_linear(packed.ctypes.data, scale.ctypes.data, x.ctypes.data,
                         out.ctypes.data, ctypes.c_int(rows), ctypes.c_int(cols),
                         ctypes.c_int(x.shape[0]), ctypes.c_int(bits))
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


def gp_profile(buf, n):
    """Run a program with the time of each record (gemma_profile). Return
    the times in ms, one for each of the n records."""
    ms = np.zeros(n, dtype=np.float64)
    rc = _lib.gemma_profile(buf.ctypes.data, ms.ctypes.data)
    if rc != 0:
        raise RuntimeError("gemma_profile returned %d" % rc)
    return ms


def argmax(x):
    """Return the index of the largest value of a float32 vector, as
    np.argmax, about 10 times faster on a row of logits."""
    return int(_lib.gemma_argmax(x.ctypes.data, ctypes.c_int64(x.size)))


def ma_quant_x(x, bits, xq, xs, xsum):
    """Quantize the rows of x (float32, contiguous) for the MLX affine
    products of bits bits: xq (int8, the shape of x), xs and xsum (one value
    for each group of 64). See csrc/mlx_affine.c."""
    t, cols = x.shape
    _lib.ma_quant_x(x.ctypes.data, t, cols, bits, xq.ctypes.data, xs.ctypes.data, xsum.ctypes.data)


def ma_linear(q, scales, biases, bits, rows, cols, xq, xs, xsum, t, out):
    """out (t x rows) = x W^T for an MLX affine matrix (q, scales, biases)."""
    _lib.ma_linear(q.ctypes.data, scales.ctypes.data, biases.ctypes.data, bits, rows, cols,
                   xq.ctypes.data, xs.ctypes.data, xsum.ctypes.data, t, out.ctypes.data)


def ma_moe_mats(gate, up, down, shared):
    """The descriptor of the matrices of ma_moe: 6 x (q, scales, biases,
    bits) as int64 (0 for no shared expert). Each matrix is a QMat.c()."""
    rows = []
    for m in (gate, up, down) + (tuple(shared) if shared is not None else (None, None, None)):
        rows += [0, 0, 0, 0] if m is None else [m[0].ctypes.data, m[1].ctypes.data,
                                                 m[2].ctypes.data, m[3]]
    return np.array(rows, dtype=np.int64)


def ma_moe_scratch(t, k, experts, hidden, inner):
    return np.empty(_lib.ma_moe_scratch(t, k, experts, hidden, inner), dtype=np.uint8)


def ma_moe(hq4, hq8, hs, hsum, ids, val, experts, mats, shared_logit, hidden, inner, scratch,
           out):
    """The experts of t tokens with MLX affine weights (csrc/mlx_affine.c,
    ma_moe_body). ids and val are (t, k); mats comes from ma_moe_mats;
    shared_logit (t values) or None."""
    t, k = ids.shape
    _lib.ma_moe(hq4.ctypes.data, hq8.ctypes.data, hs.ctypes.data, hsum.ctypes.data,
                ids.ctypes.data, val.ctypes.data, t, k, experts, mats.ctypes.data,
                None if shared_logit is None else shared_logit.ctypes.data, hidden, inner,
                scratch.ctypes.data, out.ctypes.data)


def gdn_step(qkv, conv, conv_w, z, a, b, A_log, dt_bias, norm_w, S, out, scratch, k_heads,
             v_heads, k_dim, v_dim, eps, flags=0):
    """The Gated DeltaNet for the rows of qkv, in order (csrc/deltanet.c).
    conv and S are the state; the call changes them. flags: 1 for the order of
    the value heads of the GGUF files, 2 for a sigmoid gate of the norm
    (qwen.gdn_flags)."""
    t = qkv.shape[0]
    _lib.gdn_step(qkv.ctypes.data, conv.ctypes.data, conv_w.ctypes.data, conv_w.shape[1],
                  z.ctypes.data, a.ctypes.data, b.ctypes.data, A_log.ctypes.data,
                  dt_bias.ctypes.data, norm_w.ctypes.data, S.ctypes.data, out.ctypes.data,
                  scratch.ctypes.data, t, k_heads, v_heads, k_dim, v_dim, float(eps), None,
                  int(flags))


def gdn_log_floats(t, k_heads, v_heads, k_dim, v_dim):
    """The size of the log of a verify group of t tokens (csrc/deltanet.c)."""
    return int(_lib.gdn_log_floats(t, k_heads, v_heads, k_dim, v_dim))


def gdn_commit(conv, S, log, n, kernel, k_heads, v_heads, k_dim, v_dim):
    """Apply the first n tokens of the log of a verify group to conv and S."""
    _lib.gdn_commit(conv.ctypes.data, S.ctypes.data, log.ctypes.data, n, kernel, k_heads,
                    v_heads, k_dim, v_dim)


def kq_quant_x(x, xq, xs, xm):
    """Quantize the rows of x (float32, contiguous) for the GGUF products:
    xq (int8, the shape of x), xs (one value for each 32), xm (one value for
    each 16). See csrc/kquants.c."""
    t, cols = x.shape
    _lib.kq_quant_x(x.ctypes.data, t, cols, xq.ctypes.data, xs.ctypes.data, xm.ctypes.data)


def kq_linear(w, type_, rows, cols, xq, xs, xm, x, t, out):
    """out (t x rows) = x W^T for a GGUF matrix w (the raw blocks) of ggml
    type type_ (0, 8, 12, 13, 14)."""
    _lib.kq_linear(w.ctypes.data, type_, rows, cols, xq.ctypes.data, xs.ctypes.data,
                   xm.ctypes.data, x.ctypes.data, t, out.ctypes.data)


def kq_rows(w, type_, cols, ids):
    """The float32 rows ids of a GGUF matrix (the embeddings)."""
    ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
    out = np.empty((ids.size, cols), dtype=np.float32)
    _lib.kq_rows(w.ctypes.data, type_, cols, ids.ctypes.data, ids.size, out.ctypes.data)
    return out


KQ_Q8_0, KQ_BF16, KQ_NV4, KQ_NVX = 8, 30, 51, 53


def kq_to_q8_0(src, cols):
    """Q8_0 blocks (uint8, rows x cols / 32 * 34) of a matrix: bfloat16 as
    uint16, or float32."""
    rows = src.size // cols
    src = np.ascontiguousarray(src)
    out = np.empty(rows * (cols // 32) * 34, dtype=np.uint8)
    _lib.kq_to_q8_0(src.ctypes.data, 1 if src.dtype == np.uint16 else 0, rows, cols,
                    out.ctypes.data)
    return out


def kq_to_q6_k(src, cols):
    """Q6_K blocks (uint8, rows x cols / 256 * 210; cols a multiple of 256)
    of a matrix: bfloat16 as uint16, or float32 (ggml quantize_row_q6_K_ref)."""
    assert cols % 256 == 0, cols
    rows = src.size // cols
    src = np.ascontiguousarray(src)
    out = np.empty(rows * (cols // 256) * 210, dtype=np.uint8)
    _lib.kq_to_q6_k(src.ctypes.data, 1 if src.dtype == np.uint16 else 0, rows, cols,
                    out.ctypes.data)
    return out


def bf12_row_bytes(cols):
    """The bytes of a BF12 row (BF12_PLAN.md): cols + cols / 2 + cols / 32, to 16."""
    return (cols + cols // 2 + cols // 32 + 15) // 16 * 16


def kq_bf16_to_bf12(src, cols):
    """BF12 rows (uint8, rows x bf12_row_bytes(cols)) of bfloat16 rows (uint16;
    cols a multiple of 32), and the report: a dict of zeroed (the nonzero
    values that decode to 0: more than 15 binades under the largest of their
    group of 32), collisions (-2^(E-15), which decode to 0), nonfinite (Inf,
    NaN), largest (the largest zeroed value in absolute value), rms (of the
    finite values), worst (up to 10 of (value, row, column, the largest of its
    group), the largest first)."""
    assert cols % 32 == 0, cols
    src = np.ascontiguousarray(src).view(np.uint16)
    rows = src.size // cols
    out = np.empty((rows, bf12_row_bytes(cols)), dtype=np.uint8)
    rep = np.zeros(6 + 40, np.float64)
    _lib.kq_bf16_to_bf12(src.ctypes.data, rows, cols, out.ctypes.data, rep.ctypes.data)
    n = rows * cols - int(rep[2])
    k = int(rep[5])
    return out, dict(zeroed=int(rep[0]), collisions=int(rep[1]), nonfinite=int(rep[2]),
                     largest=float(rep[3]), rms=float(np.sqrt(rep[4] / max(n, 1))),
                     worst=[(float(rep[6 + 4 * i]), int(rep[7 + 4 * i]), int(rep[8 + 4 * i]),
                             float(rep[9 + 4 * i])) for i in range(k)])


def kq_pack_bf12x16(src, rows, cols):
    """BF12 rows as KQ_BF12X16 (groups of 16 rows, the last one padded; csrc/
    kquants.c kq_x16f_col): 784 bytes a block of 32 columns of a group."""
    src = np.ascontiguousarray(src).view(np.uint8)
    out = np.empty(((rows + 15) // 16) * (cols // 32) * 784, dtype=np.uint8)
    _lib.kq_pack_bf12x16(src.ctypes.data, rows, cols, out.ctypes.data)
    return out


def kq_bf12_to_bf16(src, rows, cols):
    """The bfloat16 bits (uint16, rows x cols) of BF12 rows."""
    src = np.ascontiguousarray(src).view(np.uint8)
    out = np.empty((rows, cols), dtype=np.uint16)
    _lib.kq_bf12_to_bf16(src.ctypes.data, rows, cols, out.ctypes.data)
    return out


KQ_Q8X16 = 60


def kq_pack_q8x16(q8, rows, cols):
    """Q8_0 rows (uint8; rows a multiple of 16) to KQ_Q8X16 (csrc/kquants.c)."""
    assert rows % 16 == 0 and cols % 32 == 0
    out = np.empty(rows * (cols // 32) * 36, dtype=np.uint8)
    _lib.kq_pack_q8x16(np.ascontiguousarray(q8).ctypes.data, rows, cols, out.ctypes.data)
    return out


KQ_BF16X16, KQ_F32X16 = 61, 62
KQ_Q4X = 54


def kq_q4x_pack(packed, scales):
    """The int4 matrix of the Gemma 4 26B (Q4_0 blocks of 18 bytes, rows x
    blocks x 18; float32 scales, rows x blocks) to KQ_Q4X: groups of 16 rows
    (csrc/kquants.c). A stack of matrices packs as one matrix of all the
    rows."""
    packed = np.ascontiguousarray(packed)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    nb = packed.shape[-2]
    rows = packed.size // (nb * 18)
    assert rows % 16 == 0
    out = np.empty(rows * nb * 18, dtype=np.uint8)
    if _lib.kq_q4x_pack(packed.ctypes.data, scales.ctypes.data, rows, nb * 32, out.ctypes.data):
        raise ValueError("a scale is not exact in float16")
    return out


def kq_moe_act(hq, hs, hm, ids, val, experts, mats, shared_logit, hidden, inner, scratch, out,
               gelu=True, hf=None, x16=False):
    """kq_moe with the tanh GELU of the gate (gelu) in place of SiLU. hf (the
    float32 rows of h; KQ_Q4X matrices): float32 activations, no
    quantization; with x16, the int16 rows of hf and of the GELU (act bit 2;
    hq, hs, hm not read)."""
    t, k = ids.shape
    _lib.kq_moe_act(hq.ctypes.data, hs.ctypes.data, hm.ctypes.data, ids.ctypes.data,
                    val.ctypes.data, t, k, experts, mats.ctypes.data,
                    None if shared_logit is None else shared_logit.ctypes.data, hidden, inner,
                    scratch.ctypes.data, out.ctypes.data,
                    (1 if gelu else 0) | (4 if x16 else (2 if hf is not None else 0)),
                    None if hf is None else hf.ctypes.data)


def kq_pack_x16f(a, bf, rows, cols):
    """float32 (bf False) or bfloat16 (as uint16) rows to KQ_F32X16 or
    KQ_BF16X16: groups of 16 rows, the rows past rows zeros (csrc/kquants.c)."""
    out = np.empty(((rows + 15) // 16) * 16 * cols * (2 if bf else 4), dtype=np.uint8)
    _lib.kq_pack_x16f(np.ascontiguousarray(a).ctypes.data, 1 if bf else 0, rows, cols,
                      out.ctypes.data)
    return out


def kq_memcpy_par(dst, src):
    """Copy the bytes of src into dst (both contiguous, the same size) on all
    the threads of OpenMP (kq_memcpy_par)."""
    assert dst.nbytes == src.nbytes and dst.flags.c_contiguous and src.flags.c_contiguous
    _lib.kq_memcpy_par(dst.ctypes.data, src.ctypes.data, ctypes.c_int64(dst.nbytes))


def kq_calib_nodes(mats, mats1, experts, hidden, inner, ncold, reps, nth):
    """The mean compute time of a thread of each half of a team of nth threads
    (bound spread: node 0, node 1) on the cold experts of a decode step
    (kq_calib_nodes): mats and mats1 are (layers, 12) int64 tables
    (kq_moe_mats; mats1 the copies of node 1, or None)."""
    mats = np.ascontiguousarray(mats, dtype=np.int64)
    m1 = None if mats1 is None else np.ascontiguousarray(mats1, dtype=np.int64)
    out = np.zeros(2)
    _lib.kq_calib_nodes(mats.ctypes.data, None if m1 is None else m1.ctypes.data, mats.shape[0],
                        experts, hidden, inner, ncold, reps, nth, out.ctypes.data)
    return out


def kq_gather(addrs, nbytes):
    """The rows of nbytes bytes at the addresses addrs (int64), read by many
    threads at a time (rows of a memory map)."""
    addrs = np.ascontiguousarray(addrs, dtype=np.int64)
    out = np.empty((addrs.size, nbytes), dtype=np.uint8)
    _lib.kq_gather(addrs.ctypes.data, addrs.size, nbytes, out.ctypes.data)
    return out


def kq_nv4_row_bytes(cols):
    """A KQ_NV4 row: the codes, the scales, the scale of the matrix, zeros to
    a multiple of 16 bytes (csrc/kquants.c)."""
    return (cols // 2 + cols // 16 + 4 + 15) // 16 * 16


def kq_nvx_row_bytes(cols):
    """A "row" of KQ_NVX: a group of 16 rows is 16 times this (csrc/kquants.c)."""
    return cols // 32 * 18 + 1


def kq_nvx_pack(ws, ss, gs, rows, cols, out):
    """As kq_nv4_pack, to KQ_NVX groups of 16 rows (out: n x rows x
    kq_nvx_row_bytes(cols) bytes)."""
    assert rows % 16 == 0 and cols % 32 == 0
    n = len(ws)
    wp = np.array([w.ctypes.data for w in ws], dtype=np.int64)
    sp = np.array([x.ctypes.data for x in ss], dtype=np.int64)
    g = np.ascontiguousarray(gs, dtype=np.float32)
    assert out.nbytes >= n * rows * kq_nvx_row_bytes(cols)
    _lib.kq_nvx_pack(wp.ctypes.data, sp.ctypes.data, g.ctypes.data, n, rows, cols, out.ctypes.data)


def kq_nv4_pack(ws, ss, gs, rows, cols, out):
    """The NVFP4 matrices of ModelOpt (ws: uint8 rows x cols / 2; ss: E4M3
    rows x cols / 16; gs: the float32 scales) to KQ_NV4 rows in out (uint8,
    n x rows x kq_nv4_row_bytes(cols))."""
    n = len(ws)
    wp = np.array([w.ctypes.data for w in ws], dtype=np.int64)
    sp = np.array([x.ctypes.data for x in ss], dtype=np.int64)
    g = np.ascontiguousarray(gs, dtype=np.float32)
    assert out.nbytes >= n * rows * kq_nv4_row_bytes(cols)
    _lib.kq_nv4_pack(wp.ctypes.data, sp.ctypes.data, g.ctypes.data, n, rows, cols, out.ctypes.data)


def kq_moe_mats(gate, up, down, shared):
    """The descriptor of the matrices of kq_moe: 6 x (data, type) as int64
    (0 for no shared expert). Each matrix is a (data, type) pair."""
    rows = []
    for m in (gate, up, down) + (tuple(shared) if shared is not None else (None, None, None)):
        rows += [0, 0] if m is None else [m[0].ctypes.data, m[1]]
    return np.array(rows, dtype=np.int64)


def kq_moe_scratch(t, k, experts, hidden, inner):
    return np.empty(_lib.kq_moe_scratch(t, k, experts, hidden, inner), dtype=np.uint8)


def kq_moe(hq, hs, hm, ids, val, experts, mats, shared_logit, hidden, inner, scratch, out):
    """The experts of t tokens with GGUF weights (csrc/kquants.c,
    kq_moe_body). ids and val are (t, k); mats comes from kq_moe_mats."""
    t, k = ids.shape
    _lib.kq_moe(hq.ctypes.data, hs.ctypes.data, hm.ctypes.data, ids.ctypes.data,
                val.ctypes.data, t, k, experts, mats.ctypes.data,
                None if shared_logit is None else shared_logit.ctypes.data, hidden, inner,
                scratch.ctypes.data, out.ctypes.data)


def q6k_rows(table, ids, cols):
    """Return the rows ids of a Q6_K table as float32, shape (len(ids), cols).

    table is the raw block data of the whole table (a uint8 array).
    """
    ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
    out = np.empty((ids.size, cols), dtype=np.float32)
    _lib.gemma_q6k_rows(table.ctypes.data, ids.ctypes.data, ctypes.c_int(ids.size),
                        ctypes.c_int(cols), out.ctypes.data)
    return out


def q4_0_rows(table, ids, cols):
    """Return the rows ids of a Q4_0 table as float32, shape (len(ids), cols),
    with the bits of gguf._dequant."""
    ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
    out = np.empty((ids.size, cols), dtype=np.float32)
    _lib.gemma_q4_0_rows(table.ctypes.data, ids.ctypes.data, ctypes.c_int(ids.size),
                         ctypes.c_int(cols), out.ctypes.data)
    return out


def kq45_rows(table, ids, cols, five):
    """Return the rows ids of a Q4_K (five False) or Q5_K table as float32,
    shape (len(ids), cols), with the bits of gguf._dequant."""
    ids = np.ascontiguousarray(ids, dtype=np.int64).reshape(-1)
    out = np.empty((ids.size, cols), dtype=np.float32)
    _lib.gemma_kq45_rows(table.ctypes.data, ids.ctypes.data, ctypes.c_int(ids.size),
                         ctypes.c_int(cols), ctypes.c_int(1 if five else 0), out.ctypes.data)
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


def _pa(a):
    """Return the address of a buffer argument.

    An argument is an array, or an address that the caller read one time. The
    second form is for a buffer that one function hands to this function more
    than one time. The read of `ndarray.ctypes.data` costs about 1.5
    microseconds, because it builds two objects, so it must not happen again
    for each call.
    """
    if a is None or isinstance(a, int):
        return a
    return a.ctypes.data


def quantize_q8_groups(x):
    """Quantize the last axis of x to int8 with one scale for each group of 32.

    Return qx (tokens, cols) int8, sx (tokens, groups) float32, and sumx
    (tokens, groups) int32, the integer sum of each group.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    tokens, cols = x.shape
    groups = cols // 32
    qx = np.empty((tokens, cols), dtype=np.int8)
    sx = np.empty((tokens, groups), dtype=np.float32)
    sumx = np.empty((tokens, groups), dtype=np.int32)
    _lib.gemma_quantize_q8_groups(x.ctypes.data, qx.ctypes.data, sx.ctypes.data,
                                  sumx.ctypes.data, ctypes.c_int(tokens), ctypes.c_int(cols))
    return qx, sx, sumx


def quantize_q8_t(x, stride=None):
    """Quantize the last axis of x to int8 in the transposed tile layout.

    The layout for one group is (k / 4, token, 4). stride gives the token stride
    of qxt, sx, and sumx. It defaults to the token count. The values from tokens
    to stride are zero.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    tokens, cols = x.shape
    groups = cols // 32
    if stride is None:
        stride = tokens
    qxt = np.zeros(groups * 8 * stride * 4, dtype=np.int8)
    sx = np.zeros(groups * stride, dtype=np.float32)
    sumx = np.zeros(groups * stride, dtype=np.int32)
    _lib.gemma_quantize_q8_t(x.ctypes.data, qxt.ctypes.data, sx.ctypes.data,
                             sumx.ctypes.data, ctypes.c_int(tokens),
                             ctypes.c_int(cols), ctypes.c_int(stride))
    return qxt.reshape(groups * 8, stride, 4), sx.reshape(groups, stride), \
        sumx.reshape(groups, stride)


def int4_q8_tile(qxt, sx, sumx, packed, scales, group, tokens):
    """Multiply x by W for one group of tokens. W is packed 4-bit data.

    qxt, sx, and sumx come from quantize_q8_t. The kernel quantizes the
    activations to int8 and uses integer multiply and add.
    """
    qxt = np.ascontiguousarray(qxt, dtype=np.int8)
    sx = np.ascontiguousarray(sx, dtype=np.float32)
    sumx = np.ascontiguousarray(sumx, dtype=np.int32)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    rows = packed.shape[0]
    cols = packed.shape[1] * 32
    stride = qxt.shape[1]
    out = np.empty((tokens, rows), dtype=np.float32)
    _lib.gemma_int4_q8_tile_run(packed.ctypes.data, scales.ctypes.data,
                                qxt.ctypes.data, sx.ctypes.data, sumx.ctypes.data,
                                out.ctypes.data, ctypes.c_int(rows), ctypes.c_int(cols),
                                ctypes.c_int(tokens), ctypes.c_int(stride))
    return out


def int4_q8_tile32(qxt, sx, sumx, packed, scales, group, tokens):
    """The wide tile: 32 tokens for one weight decode. A test uses it."""
    qxt = np.ascontiguousarray(qxt, dtype=np.int8)
    sx = np.ascontiguousarray(sx, dtype=np.float32)
    sumx = np.ascontiguousarray(sumx, dtype=np.int32)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    rows = packed.shape[0]
    cols = packed.shape[1] * 32
    stride = qxt.shape[1]
    out = np.empty((tokens, rows), dtype=np.float32)
    _lib.gemma_int4_q8_tile_run32(packed.ctypes.data, scales.ctypes.data,
                                  qxt.ctypes.data, sx.ctypes.data, sumx.ctypes.data,
                                  out.ctypes.data, ctypes.c_int(rows), ctypes.c_int(cols),
                                  ctypes.c_int(tokens), ctypes.c_int(stride))
    return out


def int4_q8_gemv(qx, sx, sumx, packed, scales):
    """Multiply one token by W with int8 activations. W is packed 4-bit data.

    qx is (cols,) int8, sx is (groups,) float32, and sumx is (groups,) int32,
    all from quantize_q8_groups. The kernel keeps the weight rows in the lanes
    of one register and the token in the group, so no lane is idle.
    """
    qx = np.ascontiguousarray(qx, dtype=np.int8)
    sx = np.ascontiguousarray(sx, dtype=np.float32)
    sumx = np.ascontiguousarray(sumx, dtype=np.int32)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    rows = packed.shape[0]
    cols = packed.shape[1] * 32
    out = np.empty(rows, dtype=np.float32)
    _lib.gemma_int4_q8_gemv(packed.ctypes.data, scales.ctypes.data,
                            qx.ctypes.data, sx.ctypes.data, sumx.ctypes.data,
                            out.ctypes.data, ctypes.c_int(rows), ctypes.c_int(cols))
    return out


def int4_q8_gemv_x(x, packed, scales):
    """Multiply a one-row x by W with int8 activations, from a float32 x.

    The quantization of x and the dot product stay in one call, so the caller
    starts one parallel region and pays for one call.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    packed = np.ascontiguousarray(packed, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    rows = packed.shape[0]
    cols = packed.shape[1] * 32
    out = np.empty(rows, dtype=np.float32)
    _lib.gemma_int4_q8_gemv_x(packed.ctypes.data, scales.ctypes.data,
                              x.ctypes.data, out.ctypes.data,
                              ctypes.c_int(rows), ctypes.c_int(cols))
    return out


def int4_q8_multi4(mats, x, cols):
    """Run up to four int4 matrices on the same one-row x with int8 data.

    mats is a list of up to four (packed, scales) pairs. A None entry skips a
    matrix. Return a list of float32 outputs, or None for a skipped matrix.
    The activation is quantized one time for all four matrices.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    args = []
    outs = []
    for i in range(4):
        if i < len(mats) and mats[i] is not None:
            w, s = mats[i]
            w = np.ascontiguousarray(w, dtype=np.uint8)
            s = np.ascontiguousarray(s, dtype=np.float32)
            o = np.empty(w.shape[0], dtype=np.float32)
            args += [w.ctypes.data, s.ctypes.data, o.ctypes.data,
                     ctypes.c_int(w.shape[0])]
            outs.append(o)
        else:
            args += [None, None, None, ctypes.c_int(0)]
            outs.append(None)
    args += [x.ctypes.data, ctypes.c_int(cols)]
    _lib.gemma_int4_q8_multi4(*args)
    return outs


def int4_q8_moe_gemv(w, scales, x, ids, rows, cols, xstride):
    """Multiply each selected expert matrix by its input row with int8 data.

    w has the shape (experts, rows, groups, 18). scales has the shape
    (experts, rows, groups). ids gives the selected experts. x has one row for
    each job, with a stride of xstride. A stride of 0 gives the same x to each
    job. Return (jobs, rows).
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    jobs = int(ids.size)
    out = np.empty((jobs, rows), dtype=np.float32)
    _lib.gemma_int4_q8_moe_gemv(w.ctypes.data, scales.ctypes.data, x.ctypes.data,
                                ids.ctypes.data, ctypes.c_int(jobs),
                                out.ctypes.data, ctypes.c_int(rows),
                                ctypes.c_int(cols), ctypes.c_int(xstride))
    return out


def quantize_q8_t_moe(x, cols, stride, off, ntok, src=None):
    """Quantize every expert's rows to the transposed int8 layout.

    off gives the token offset of each expert and ntok its token count. stride
    is the token stride of qxt, sx, and sumx. Leave a slack of one token block.
    src maps a destination row to a row of x, so the caller needs no gather. A
    null src gives the identity.

    The padding rows hold no value. A tile reads them but stores only the real
    rows, so the buffers start as empty.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    groups = cols // 32
    qxt = np.empty(groups * 8 * stride * 4, dtype=np.int8)
    sx = np.empty(groups * stride, dtype=np.float32)
    sumx = np.empty(groups * stride, dtype=np.int32)
    off = np.ascontiguousarray(off, dtype=np.int32)
    ntok = np.ascontiguousarray(ntok, dtype=np.int32)
    sp = None if src is None else np.ascontiguousarray(src, dtype=np.int32)
    _lib.gemma_quantize_q8_t_moe(x.ctypes.data, None if sp is None else sp.ctypes.data,
                                 qxt.ctypes.data, sx.ctypes.data, sumx.ctypes.data,
                                 ctypes.c_int(cols), ctypes.c_int(stride),
                                 off.ctypes.data, ntok.ctypes.data,
                                 ctypes.c_int(off.size))
    return (qxt.reshape(groups * 8, stride, 4), sx.reshape(groups, stride),
            sumx.reshape(groups, stride))


def gelu_mul(x, inner):
    """Apply the GELU to the gate half of x and multiply by the up half."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty((x.shape[0], inner), dtype=np.float32)
    _lib.gemma_gelu_mul(x.ctypes.data, out.ctypes.data, ctypes.c_int(x.shape[0]),
                        ctypes.c_int(inner))
    return out


def softmax_mask(x, positions, n_rep, base, window):
    """Apply the causal mask, the window mask, and the softmax in place.

    x is (kv_heads, tokens, heads_per_group, keys) and must be contiguous.
    """
    _lib.gemma_softmax_mask(x.ctypes.data,
                            ctypes.c_int(x.shape[0] * x.shape[1] * x.shape[2]),
                            ctypes.c_int(x.shape[3]), positions.ctypes.data,
                            ctypes.c_int(x.shape[1]), ctypes.c_int(n_rep),
                            ctypes.c_int(base), ctypes.c_int(window))


if _lib is not None:
    _lib.gemma_attn_prefill_qc.argtypes = [_void_p] * 6 + [_int, _int, _void_p] + [_int] * 5 + [_void_p]
    _lib.gemma_attn_prefill_qc.restype = None
    _lib.gemma_attn_prefill_qc_ok.argtypes = [_int, _int, _int]
    _lib.gemma_attn_prefill_qc_ok.restype = _int
    _lib.gemma_attn_prefill.argtypes = [_void_p, _void_p, _void_p, _void_p,
                                        _int, _int, _void_p, _int, _int,
                                        _int, _int, _int]
    _lib.gemma_attn_prefill.restype = None
    _lib.gemma_attn_prefill_set_impl.argtypes = [_int]
    _lib.gemma_attn_prefill_set_impl.restype = None


def attn_prefill_impl(which):
    """Select the flash attention version.

    0 takes the best that the build gives, 1 the straight C version, 2 the AVX2
    version, and 3 the AVX-512 version. A test uses this.
    """
    if _lib is not None:
        _lib.gemma_attn_prefill_set_impl(ctypes.c_int(int(which)))


def attn_prefill_qc(q, kq, ks, vq, vs, positions, base, window, limit=None):
    """attn_prefill over the int16 cache: kq, vq (keys, kv_heads, head_dim)
    int16 and ks, vs (keys, kv_heads, head_dim / 32) float32, as
    KVCache.read_qc gives them. The same values as attn_prefill on the rows
    of KVCache.read. limit (None, or t) gives the last key position of each
    query in place of its position (the tokens of an image, the mask of
    ops.softmax_mask_limit). Return None when the kernel does not take the
    shape."""
    t, q_heads, hd = q.shape
    n, kv_heads, _ = kq.shape
    if not _lib.gemma_attn_prefill_qc_ok(q_heads, kv_heads, hd):
        return None
    q = np.ascontiguousarray(q, dtype=np.float32)
    args = [np.ascontiguousarray(a) for a in (kq, ks, vq, vs)]
    assert args[0].dtype == np.int16 and args[2].dtype == np.int16
    positions = np.ascontiguousarray(positions, dtype=np.int32)
    if limit is not None:
        limit = np.ascontiguousarray(limit, dtype=np.int32)
    out = np.empty((t, q_heads, hd), dtype=np.float32)
    _lib.gemma_attn_prefill_qc(q.ctypes.data, *(a.ctypes.data for a in args),
                               positions.ctypes.data, ctypes.c_int(int(base)),
                               ctypes.c_int(int(window)), out.ctypes.data, ctypes.c_int(t),
                               ctypes.c_int(n), ctypes.c_int(q_heads), ctypes.c_int(kv_heads),
                               ctypes.c_int(hd), None if limit is None else limit.ctypes.data)
    return out


def attn_prefill(q, k, v, positions, base, window):
    """Run flash attention for a prompt.

    q is (tokens, q_heads, head_dim). k and v are (keys, kv_heads, head_dim).
    positions gives the position of each query. base is the position of key
    zero. A window of zero turns the sliding window off. Return the
    (tokens, q_heads, head_dim) result.
    """
    q = np.ascontiguousarray(q, dtype=np.float32)
    k = np.ascontiguousarray(k, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    positions = np.ascontiguousarray(positions, dtype=np.int32)
    t, q_heads, hd = q.shape
    n, kv_heads, _ = k.shape
    out = np.empty((t, q_heads, hd), dtype=np.float32)
    _lib.gemma_attn_prefill(q.ctypes.data, k.ctypes.data, v.ctypes.data,
                            positions.ctypes.data, ctypes.c_int(int(base)),
                            ctypes.c_int(int(window)), out.ctypes.data,
                            ctypes.c_int(t), ctypes.c_int(n),
                            ctypes.c_int(q_heads), ctypes.c_int(kv_heads),
                            ctypes.c_int(hd))
    return out


def moe_scatter(out, de, rows, w, hidden, n):
    """Add de[j] * w[j] to the row rows[j] of out. out is (tokens, hidden)."""
    _lib.gemma_moe_scatter(out.ctypes.data, de.ctypes.data, rows.ctypes.data,
                           w.ctypes.data, ctypes.c_int(n), ctypes.c_int(hidden))


def int4_q8_moe(w, scales, qxt, sx, sumx, rows, cols, stride, off, ntok, eid):
    """Multiply the selected experts by their rows with int8 activations.

    w holds one matrix for each expert. eid gives the matrix index of each job.
    Return one row for each token of every expert, in the same order.
    """
    w = np.ascontiguousarray(w, dtype=np.uint8)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    qxt = np.ascontiguousarray(qxt, dtype=np.int8)
    sx = np.ascontiguousarray(sx, dtype=np.float32)
    sumx = np.ascontiguousarray(sumx, dtype=np.int32)
    off = np.ascontiguousarray(off, dtype=np.int32)
    ntok = np.ascontiguousarray(ntok, dtype=np.int32)
    eid = np.ascontiguousarray(eid, dtype=np.int32)
    out = np.empty((stride, rows), dtype=np.float32)
    _lib.gemma_int4_q8_moe_run(w.ctypes.data, scales.ctypes.data, qxt.ctypes.data,
                               sx.ctypes.data, sumx.ctypes.data, out.ctypes.data,
                               ctypes.c_int(rows), ctypes.c_int(cols),
                               ctypes.c_int(stride), off.ctypes.data,
                               ntok.ctypes.data, eid.ctypes.data,
                               ctypes.c_int(eid.size))
    return out


def linear_f32(x, w):
    """Multiply x by W. W is float32 data. Use the C kernel."""
    return _call(_lib.gemma_f32_linear, w, x)


def qkv_norm(q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps):
    """Apply the RMSNorm of the query, the key, and the value in place.

    A data pointer may be None. Then the matching row count must be zero. A
    shared layer of the E4B model has no key and no value of its own.
    """
    _lib.gemma_qkv_norm(q.ctypes.data, q_w.ctypes.data, ctypes.c_int(q_rows),
                        None if k is None else k.ctypes.data, k_w.ctypes.data,
                        ctypes.c_int(k_rows),
                        None if v is None else v.ctypes.data, ctypes.c_int(v_rows),
                        ctypes.c_int(head_dim), ctypes.c_float(eps))


def rope_apply(q, q_rows, q_heads, k, k_rows, k_heads, cos, sin, head_dim):
    """Apply RoPE to the query and the key in place. k may be None."""
    _lib.gemma_rope(q.ctypes.data, ctypes.c_int(q_rows), ctypes.c_int(q_heads),
                    None if k is None else k.ctypes.data, ctypes.c_int(k_rows),
                    ctypes.c_int(k_heads), cos.ctypes.data, sin.ctypes.data,
                    ctypes.c_int(head_dim))


def router(x, scale, proj, per_expert, hidden, experts, top_k, eps, hscale, val, idx):
    """Run the mixture-of-experts router for one token."""
    _lib.gemma_router(x.ctypes.data, scale.ctypes.data, proj.ctypes.data,
                      per_expert.ctypes.data, ctypes.c_int(hidden),
                      ctypes.c_int(experts), ctypes.c_int(top_k),
                      ctypes.c_float(eps), ctypes.c_float(hscale),
                      val.ctypes.data, idx.ctypes.data)


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


def attn_decode_f32(q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
                    k_head_stride, v_head_stride, pos, base, window):
    """Run the fused float32 attention for one query token.

    q is (q_heads, head_dim). k and v are (kv_heads, n, head_dim) parts of a
    larger buffer. The strides give the distance between two heads, in values.
    scores is a scratch array of (q_heads, n). All arrays must be float32.
    """
    _lib.gemma_attn_decode_f32(q.ctypes.data, k.ctypes.data, v.ctypes.data,
                               scores.ctypes.data, out.ctypes.data,
                               ctypes.c_int(q_heads), ctypes.c_int(kv_heads),
                               ctypes.c_int(head_dim), ctypes.c_int(n),
                               ctypes.c_long(k_head_stride),
                               ctypes.c_long(v_head_stride),
                               ctypes.c_int(pos), ctypes.c_int(base),
                               ctypes.c_int(window))


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
    wp = _pa(w)
    _lib.gemma_rms_norm(x.ctypes.data, wp, out.ctypes.data,
                        ctypes.c_int(rows), ctypes.c_int(cols), ctypes.c_float(eps))
    return out


def softcap(logits, cap):
    """Return tanh(logits / cap) * cap. The kernel works out of place."""
    x = np.ascontiguousarray(logits, dtype=np.float32)
    out = np.empty_like(x)
    _lib.gemma_softcap(x.ctypes.data, out.ctypes.data, ctypes.c_int(x.size),
                       ctypes.c_float(cap))
    return out


def gelu(x):
    """Apply the tanh approximation of GELU to every value of x."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    out = np.empty_like(x)
    _lib.gemma_gelu(x.ctypes.data, out.ctypes.data, ctypes.c_int(x.size))
    return out


def qkv_norm_rope(q, q_w, k, k_w, v, cos, sin, q_heads, k_heads, head_dim, eps):
    """The three norms and the two rotations of one attention block.

    One call in place of two. q, k, and v are (rows, head_dim). k and v may be
    None. cos and sin must be contiguous float32.
    """
    z = 0
    _lib.gemma_qkv_norm_rope(
        q.ctypes.data, _pa(q_w) or z, q.shape[0],
        z if k is None else k.ctypes.data, _pa(k_w) or z,
        0 if k is None else k.shape[0],
        z if v is None else v.ctypes.data, 0 if v is None else v.shape[0],
        _pa(cos), _pa(sin), ctypes.c_int(q_heads),
        ctypes.c_int(k_heads), ctypes.c_int(head_dim), ctypes.c_float(eps))


def rms_norm_multi4(x, wn, scratch, eps, mats, cols):
    """Normalize one row, then run up to four int4 matrices on the result.

    mats is a list of up to four (packed, scales) pairs. A None entry skips a
    matrix. Return a list of float32 outputs, or None for a skipped matrix.
    scratch holds cols float32 values and belongs to the caller.
    """
    args = []
    outs = []
    for i in range(4):
        if i < len(mats) and mats[i] is not None:
            w, sc = mats[i]
            o = np.empty(w.shape[0], dtype=np.float32)
            args += [w.ctypes.data, sc.ctypes.data, o.ctypes.data,
                     ctypes.c_int(w.shape[0])]
            outs.append(o)
        else:
            args += [None, None, None, ctypes.c_int(0)]
            outs.append(None)
    _lib.gemma_rms_norm_multi4(x.ctypes.data, _pa(wn), _pa(scratch),
                               ctypes.c_int(cols), ctypes.c_float(eps), *args)
    return outs


def gelu_mul_int4(g, u, scratch, packed, scales, rows, cols):
    """gelu(g) * u, then one int4 matrix. scratch holds the inner values."""
    o = np.empty(rows, dtype=np.float32)
    _lib.gemma_gelu_mul_int4(g.ctypes.data, u.ctypes.data, ctypes.c_int(g.size),
                             _pa(scratch), _pa(packed),
                             _pa(scales), o.ctypes.data,
                             ctypes.c_int(rows), ctypes.c_int(cols))
    return o


def moe_gemv_gelu(w, scales, x, ids, jobs, rows, cols, xstride, inner):
    """The gate and up projection of the experts, then the GELU and multiply.

    Return one row of inner values for each job. The gate and the up part go
    into a scratch buffer of the call.
    """
    act = np.empty((jobs, 2 * inner), dtype=np.float32)
    out = np.empty((jobs, inner), dtype=np.float32)
    _lib.gemma_moe_gemv_gelu(w.ctypes.data, scales.ctypes.data, x.ctypes.data,
                             ids.ctypes.data, ctypes.c_int(jobs),
                             act.ctypes.data, out.ctypes.data,
                             ctypes.c_int(rows), ctypes.c_int(cols),
                             ctypes.c_int(xstride), ctypes.c_int(inner))
    return out


# ---- a small group of tokens (the MTP verify step) --------------------------
# Each kernel gives every token the same result, bit for bit, as the one-token
# kernel. A group holds at most MT_MAX tokens.
MT_MAX = 16


def linear_int4_mt(x, packed, scales):
    """Multiply the rows of x by W. W is packed 4-bit data. Return (tokens, rows)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    rows = packed.shape[0]
    cols = packed.shape[1] * 32
    out = np.empty((x.shape[0], rows), dtype=np.float32)
    _lib.gemma_int4_linear_mt(packed.ctypes.data, scales.ctypes.data, x.ctypes.data,
                              out.ctypes.data, ctypes.c_int(rows), ctypes.c_int(cols),
                              ctypes.c_int(x.shape[0]))
    return out


def int4_multi4_mt(mats, x, cols):
    """Run up to four int4 matrices on the same rows of x.

    Return a list of (tokens, rows) outputs, or None for a skipped matrix.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    t = x.shape[0]
    args = []
    outs = []
    for i in range(4):
        if i < len(mats) and mats[i] is not None:
            w, s = mats[i]
            o = np.empty((t, w.shape[0]), dtype=np.float32)
            args += [w.ctypes.data, s.ctypes.data, o.ctypes.data, ctypes.c_int(w.shape[0])]
            outs.append(o)
        else:
            args += [None, None, None, ctypes.c_int(0)]
            outs.append(None)
    args += [x.ctypes.data, ctypes.c_int(cols), ctypes.c_int(t)]
    _lib.gemma_int4_multi4_mt(*args)
    return outs


def moe_gemv_mt(w, scales, x, ids, poff, xi, rows, cols, xstride, inner=0):
    """Run the selected experts of a small group of tokens.

    ids gives the expert of each job. The pairs of job j are poff[j] to
    poff[j + 1] - 1, and pair p reads the x row xi[p]. Return one row for each
    pair. With inner > 0, apply the GELU and the multiply and return inner
    values for each pair.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    ids = np.ascontiguousarray(ids, dtype=np.int32)
    poff = np.ascontiguousarray(poff, dtype=np.int32)
    xi = np.ascontiguousarray(xi, dtype=np.int32)
    jobs = int(ids.size)
    pairs = int(poff[-1])
    act = np.empty((pairs, rows), dtype=np.float32)
    if inner:
        out = np.empty((pairs, inner), dtype=np.float32)
        _lib.gemma_moe_gemv_gelu_mt(w.ctypes.data, scales.ctypes.data, x.ctypes.data,
                                    ids.ctypes.data, poff.ctypes.data, xi.ctypes.data,
                                    ctypes.c_int(jobs), act.ctypes.data, out.ctypes.data,
                                    ctypes.c_int(rows), ctypes.c_int(cols),
                                    ctypes.c_int(xstride), ctypes.c_int(inner))
        return out
    _lib.gemma_int4_moe_gemv_mt(w.ctypes.data, scales.ctypes.data, x.ctypes.data,
                                ids.ctypes.data, poff.ctypes.data, xi.ctypes.data,
                                ctypes.c_int(jobs), act.ctypes.data,
                                ctypes.c_int(rows), ctypes.c_int(cols),
                                ctypes.c_int(xstride))
    return act


def gelu_mul_pair(g, u):
    """Return gelu(g) * u for two float32 arrays of the same size."""
    g = np.ascontiguousarray(g, dtype=np.float32)
    u = np.ascontiguousarray(u, dtype=np.float32)
    out = np.empty_like(g)
    _lib.gemma_gelu_mul_pair(g.ctypes.data, u.ctypes.data, out.ctypes.data,
                             ctypes.c_int(g.size))
    return out


def router_mt(x, scale, proj, per_expert, top_k, eps, hscale):
    """Run the router for a small group of tokens. Return (val, idx)."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    t, hidden = x.shape
    experts = proj.shape[0]
    val = np.empty((t, top_k), dtype=np.float32)
    idx = np.empty((t, top_k), dtype=np.int32)
    _lib.gemma_router_mt(x.ctypes.data, scale.ctypes.data, proj.ctypes.data,
                         per_expert.ctypes.data, ctypes.c_int(hidden),
                         ctypes.c_int(experts), ctypes.c_int(top_k),
                         ctypes.c_float(eps), ctypes.c_float(hscale),
                         val.ctypes.data, idx.ctypes.data, ctypes.c_int(t))
    return val, idx.astype(np.int64)


def attn_decode_mt(qq, qs, kq, ks, vq, vs, q_heads, kv_heads, head_dim, lo, n):
    """Run the fused decode attention for a small group of query tokens.

    qq and qs are the int8 queries and their scales, (tokens, q_heads, ...).
    Token t reads the cache rows lo[t] to lo[t] + n[t] - 1. Return
    (tokens, q_heads, head_dim).
    """
    lo = np.ascontiguousarray(lo, dtype=np.int32)
    n = np.ascontiguousarray(n, dtype=np.int32)
    t = int(lo.size)
    nmax = int(n.max())
    scores = np.empty((t, q_heads, nmax), dtype=np.float32)
    out = np.empty((t, q_heads, head_dim), dtype=np.float32)
    _lib.gemma_attn_decode_mt(qq.ctypes.data, qs.ctypes.data, kq.ctypes.data, ks.ctypes.data,
                              vq.ctypes.data, vs.ctypes.data, scores.ctypes.data,
                              out.ctypes.data, ctypes.c_int(q_heads), ctypes.c_int(kv_heads),
                              ctypes.c_int(head_dim), lo.ctypes.data, n.ctypes.data,
                              ctypes.c_int(nmax), ctypes.c_int(t))
    return out


def linear_bf16_gemv(x, w_u16):
    """Multiply the rows of x by W with the GEMV kernel. W is raw bfloat16 data.

    The kernel runs each token with the steps of a one-token call, so each
    token gets the same bits. A small token group uses it in place of the GEMM.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    return _call(_lib.gemma_bf16_linear, w_u16, x)


def gp_run(buf, limit=-1):
    """Run a program of np_gemma.program. buf is its int64 array."""
    return _lib.gemma_run(buf.ctypes.data, ctypes.c_int(limit))


def gp_run_parts(addrs, team, bar):
    """Run the programs of the parts of a step at the same time. addrs is an
    int64 array of the address of each program. bar is the int64 barrier
    array of the programs."""
    return _lib.gemma_run_parts(addrs.ctypes.data, ctypes.c_int(addrs.size),
                                ctypes.c_int(team), bar.ctypes.data)


def gp_run_parts_prof(addrs, team, bar, ms):
    """gp_run_parts with the time of each record: ms is a float64 array
    (parts, records), and part p adds the ms of record pc to ms[p, pc]."""
    assert ms.dtype == np.float64 and ms.flags.c_contiguous and ms.shape[0] == addrs.size
    return _lib.gemma_run_parts_prof(addrs.ctypes.data, ctypes.c_int(addrs.size),
                                     ctypes.c_int(team), bar.ctypes.data, ms.ctypes.data,
                                     ctypes.c_int(ms.shape[1]))


def gp_xbar_stats():
    """Return and reset the measures of the barriers of the parts: an array
    (8, 3) with, for each part, the s that its team waited for the other
    parts, the s of its team barrier before that, and the count."""
    out = np.zeros(8 * 3, dtype=np.float64)
    _lib.gemma_xbar_stats(out.ctypes.data)
    return out.reshape(8, 3)


def kq_linear16(qx, rows, cols, x):
    """The KQ_Q4X matrix qx (rows x cols) on the rows of x with int16 x (a
    scale for each 32, gemma_quant_group32_i16): kq_linear16."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    t = x.shape[0]
    xq = np.empty((t, cols), np.int16)
    xs = np.empty((t, cols // 32), np.float32)
    out = np.empty((t, rows), np.float32)
    _lib.kq_linear16(qx.ctypes.data, rows, cols, x.ctypes.data, xq.ctypes.data, xs.ctypes.data,
                     t, out.ctypes.data)
    return out


def kq_q16_ok():
    """True when the library has the int16 kernels of the prompt (VNNI)."""
    return bool(_lib.kq_q16_ok())


def gp_part_cpus(nparts, team):
    """Return the CPU of each thread of the parts of gemma_run_parts, an int32
    array (nparts, team); -1 for a thread that did not run."""
    cpus = np.full(nparts * max(team, 4096), -1, dtype=np.int32)
    team = _lib.gemma_part_cpus(ctypes.c_int(nparts), ctypes.c_int(team), cpus.ctypes.data)
    return cpus[:nparts * team].reshape(nparts, team)


def gp_record_size():
    """Return the size of one program record in C."""
    return _lib.gemma_gp_record_size()


def attn_decode_f32s(q, k, v, pos, base, window):
    """The attention of one query over rows of the float cache of Model.

    q is (q_heads, head_dim). k and v are (n, kv_heads, head_dim). They are a
    contiguous part of the cache that starts at position base. Return
    (q_heads, head_dim).
    """
    q = np.ascontiguousarray(q, dtype=np.float32)
    n, kvh, hd = k.shape
    qh = q.size // hd
    scores = np.empty((qh, n), dtype=np.float32)
    out = np.empty((qh, hd), dtype=np.float32)
    _lib.gemma_attn_decode_f32s(q.ctypes.data, k.ctypes.data, v.ctypes.data,
                                scores.ctypes.data, out.ctypes.data, qh, kvh, hd, n,
                                int(pos), int(base), int(window))
    return out


def quantize_i16_groups(x):
    """Quantize groups of 32 float32 values to int16. Return (q, scales)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    groups = x.size // 32
    q = np.empty(x.size, dtype=np.int16)
    s = np.empty(groups, dtype=np.float32)
    _lib.gemma_quantize_i16_groups(x.ctypes.data, q.ctypes.data, s.ctypes.data,
                                   ctypes.c_long(groups))
    return q, s


def dequantize_i16_groups(q, s, out):
    """Write the float32 values of groups of 32 int16 values q with the
    scales s into out (contiguous, q.size values)."""
    q = np.ascontiguousarray(q, dtype=np.int16)
    s = np.ascontiguousarray(s, dtype=np.float32)
    assert out.flags.c_contiguous and out.dtype == np.float32 and out.size == q.size
    assert s.size * 32 == q.size
    _lib.gemma_dequantize_i16_groups(q.ctypes.data, s.ctypes.data, out.ctypes.data,
                                     ctypes.c_long(s.size))


def quantize_i8_groups(x):
    """Quantize groups of 32 float32 values to int8 (the int8 cache: the
    scale max |x| / 127). Return (q, scales)."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    groups = x.size // 32
    q = np.empty(x.size, dtype=np.int8)
    s = np.empty(groups, dtype=np.float32)
    _lib.gemma_quantize_i8_groups(x.ctypes.data, q.ctypes.data, s.ctypes.data,
                                  ctypes.c_long(groups))
    return q, s


def tq6_quantize(x):
    """The TQ6 form (np_gemma/tq6.py) of groups of 32 float32 values: the
    bytes (24 for a group) and the norms."""
    x = np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    groups = x.size // 32
    b = np.empty(groups * 24, dtype=np.uint8)
    s = np.empty(groups, dtype=np.float32)
    _lib.gemma_tq6_quantize(x.ctypes.data, b.ctypes.data, s.ctypes.data, ctypes.c_long(groups))
    return b, s


def tq6_dequantize_rotated(b, s):
    """The rotated values of the TQ6 groups (bytes b, norms s)."""
    b = np.ascontiguousarray(b, dtype=np.uint8).reshape(-1)
    s = np.ascontiguousarray(s, dtype=np.float32).reshape(-1)
    out = np.empty(s.size * 32, dtype=np.float32)
    _lib.gemma_tq6_dequantize_rotated(b.ctypes.data, s.ctypes.data, out.ctypes.data,
                                      ctypes.c_long(s.size))
    return out


def kq_set_moe_rot(on):
    """The RQ8_0 experts (kquants.c kq_moe_rot): the MoE kernels of the CPU
    rotate the act of each pair. A state of the process."""
    _lib.kq_set_moe_rot(int(bool(on)))


def tq6_rotate(x, inverse=False):
    """The TQ6 rotation of each group of 32 values of x (a copy), or its inverse."""
    y = np.array(x, dtype=np.float32, copy=True, order="C")
    _lib.gemma_tq6_rotate(y.ctypes.data, ctypes.c_long(y.size // 32), int(inverse))
    return y


def dequantize_i8_groups(q, s, out):
    """dequantize_i16_groups for int8 values (the int8 cache)."""
    q = np.ascontiguousarray(q, dtype=np.int8)
    s = np.ascontiguousarray(s, dtype=np.float32)
    assert out.flags.c_contiguous and out.dtype == np.float32 and out.size == q.size
    assert s.size * 32 == q.size
    _lib.gemma_dequantize_i8_groups(q.ctypes.data, s.ctypes.data, out.ctypes.data,
                                    ctypes.c_long(s.size))


def attn_decode_q8(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, lo, n):
    """The attention of tokens queries q (tokens, q_heads, head_dim) over the
    int8 cache (int8 values; int8 keys, or int16 keys of the form k16v8). Token t reads the rows lo[t] to lo[t] + n[t] - 1 (lo None:
    rows 0 to n[t] - 1). Return (tokens, q_heads, head_dim)."""
    q = np.ascontiguousarray(q, dtype=np.float32)
    t = q.shape[0]
    n = np.ascontiguousarray(n, dtype=np.int32).reshape(t)
    nmax = int(n.max())
    scores = np.empty((t, q_heads, nmax), dtype=np.float32)
    out = np.empty((t, q_heads, head_dim), dtype=np.float32)
    lo_p = None
    if lo is not None:
        lo = np.ascontiguousarray(lo, dtype=np.int32).reshape(t)
        lo_p = lo.ctypes.data
    rc = _lib.gemma_attn_decode_q8(q.ctypes.data, kq.ctypes.data, ks.ctypes.data,
                                   vq.ctypes.data, vs.ctypes.data, scores.ctypes.data,
                                   out.ctypes.data, q_heads, kv_heads, head_dim, lo_p,
                                   n.ctypes.data, nmax, t, int(kq.dtype == np.int16))
    if not rc:
        raise RuntimeError("the int8 cache has no attention kernel for %d heads of %d values"
                           % (q_heads // kv_heads, head_dim))
    return out


def attn_decode_i16(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, n):
    """The attention of one float32 query over n rows of the int16 cache."""
    scores = np.empty((q_heads, n), dtype=np.float32)
    out = np.empty((q_heads, head_dim), dtype=np.float32)
    _lib.gemma_attn_decode_i16(q.ctypes.data, kq.ctypes.data, ks.ctypes.data,
                               vq.ctypes.data, vs.ctypes.data, scores.ctypes.data,
                               out.ctypes.data, q_heads, kv_heads, head_dim, int(n))
    return out


def attn_decode_i16_mt(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, lo, n):
    """The attention of a group of float32 queries over the int16 cache."""
    lo = np.ascontiguousarray(lo, dtype=np.int32)
    n = np.ascontiguousarray(n, dtype=np.int32)
    t = int(lo.size)
    nmax = int(n.max())
    scores = np.empty((t, q_heads, nmax), dtype=np.float32)
    out = np.empty((t, q_heads, head_dim), dtype=np.float32)
    _lib.gemma_attn_decode_i16_mt(q.ctypes.data, kq.ctypes.data, ks.ctypes.data,
                                  vq.ctypes.data, vs.ctypes.data, scores.ctypes.data,
                                  out.ctypes.data, q_heads, kv_heads, head_dim,
                                  lo.ctypes.data, n.ctypes.data, nmax, t)
    return out


def linear_int4_q16(x, packed, scales, tb=16):
    """Multiply x by W with int16 activations. W is packed 4-bit data.

    Quantize x to int16 with one scale for each group of 32 values, then run
    the int16 tile. The token count is padded to a full token block of tb.
    """
    x = np.ascontiguousarray(x, dtype=np.float32)
    tokens, cols = x.shape
    rows = packed.shape[0]
    stride = (tokens + tb - 1) // tb * tb
    qxt = np.zeros(cols * stride, dtype=np.int16)
    sx = np.zeros((cols // 32) * stride, dtype=np.float32)
    _lib.gemma_quantize_q16_t(x.ctypes.data, qxt.ctypes.data, sx.ctypes.data,
                              tokens, cols, stride)
    out = np.empty((tokens, rows), dtype=np.float32)
    _lib.gemma_int4_q16_tile_run(packed.ctypes.data, scales.ctypes.data, qxt.ctypes.data,
                                 sx.ctypes.data, out.ctypes.data, rows, cols, tokens, stride)
    return out


def gather_t_moe(x, src, stride, n):
    """Gather n rows of x (by src, or in order) into a (cols, stride) buffer."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    cols = x.shape[1]
    xt = np.zeros(cols * stride, dtype=np.float32)
    _lib.gemma_gather_t_moe(x.ctypes.data, None if src is None else src.ctypes.data,
                            xt.ctypes.data, cols, stride, n)
    return xt


def int4_f32_moe(w, scales, xt, rows, cols, stride, off, ntok, eid):
    """Run the float tile of every selected expert in one region."""
    eid = np.ascontiguousarray(eid, dtype=np.int32)
    out = np.empty((stride, rows), dtype=np.float32)
    _lib.gemma_int4_f32_moe_run(w.ctypes.data, scales.ctypes.data, xt.ctypes.data,
                                out.ctypes.data, rows, cols, stride, off.ctypes.data,
                                ntok.ctypes.data, eid.ctypes.data, int(eid.size))
    return out


def quantize_q16_t_moe(x, src, cols, stride, n):
    """Quantize n expert rows of x (by src, or in order) to the int16 tile layout."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    qxt = np.zeros(cols * stride, dtype=np.int16)
    sx = np.zeros((cols // 32) * stride, dtype=np.float32)
    _lib.gemma_quantize_q16_t_moe(x.ctypes.data, None if src is None else src.ctypes.data,
                                  qxt.ctypes.data, sx.ctypes.data, cols, stride, n)
    return qxt, sx


def int4_q16_moe(w, scales, qxt, sx, rows, cols, stride, off, ntok, eid):
    """Run the int16 tile of every selected expert in one region."""
    eid = np.ascontiguousarray(eid, dtype=np.int32)
    out = np.empty((stride, rows), dtype=np.float32)
    _lib.gemma_int4_q16_moe_run(w.ctypes.data, scales.ctypes.data, qxt.ctypes.data,
                                sx.ctypes.data, out.ctypes.data, rows, cols, stride,
                                off.ctypes.data, ntok.ctypes.data, eid.ctypes.data,
                                int(eid.size))
    return out


def team_warn(min_threads=None):
    """Report (stderr, once a place of the call) every OpenMP team of at least
    min_threads threads that is not a planned one (gemma_run_task and the
    runners). None: NP_GEMMA_TEAM_WARN, default 5; 0 turns it off. The
    servers turn it on after the load (a load may use all the threads)."""
    if _lib is None:
        return
    if min_threads is None:
        min_threads = int(os.environ.get("NP_GEMMA_TEAM_WARN", "5"))
    _lib.gemma_team_warn.argtypes = [_int]
    _lib.gemma_team_warn(int(min_threads))


def team_warn_stats(max_sites=64):
    """[(place, count)] of the teams that team_warn reported: the place is the
    return address in this library, as its nearest symbol and offset."""
    if _lib is None:
        return []
    sites = (ctypes.c_void_p * max_sites)()
    counts = (ctypes.c_long * max_sites)()
    _lib.gemma_team_warn_stats.restype = _int
    n = _lib.gemma_team_warn_stats(sites, counts, max_sites)
    out = []
    for i in range(n):
        info = _DlInfo()
        name = "?"
        if _libdl.dladdr(ctypes.c_void_p(sites[i]), ctypes.byref(info)) and info.dli_sname:
            name = "%s+%#x" % (info.dli_sname.decode(), sites[i] - info.dli_saddr)
        out.append((name, int(counts[i])))
    return out


class _DlInfo(ctypes.Structure):
    _fields_ = [("dli_fname", ctypes.c_char_p), ("dli_fbase", ctypes.c_void_p),
                ("dli_sname", ctypes.c_char_p), ("dli_saddr", ctypes.c_void_p)]


_libdl = ctypes.CDLL(None)
_libdl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(_DlInfo)]
