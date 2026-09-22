"""Speed up the bfloat16 multiply with a Numba just-in-time kernel.

The NumPy path dequantizes one block of weights to float32. Then it multiplies.
That path writes a float32 block and reads the block again.

The Numba kernel reads the bfloat16 data. It converts each value during the
multiply. Thus the kernel does not write a float32 block. The memory traffic is
about one third of the NumPy path.

A lookup table converts one bfloat16 bit pattern to one float32 value. The
table has 65536 entries. The table size is 256 KB. The table stays in the CPU
cache.

If Numba is not available, the model uses the NumPy path.
"""
from __future__ import annotations

import os

import numpy as np

try:
    from numba import njit, prange

    HAVE_NUMBA = True
except Exception:  # numba is optional
    HAVE_NUMBA = False


def enabled():
    """Return True when the model can use the Numba kernels."""
    return HAVE_NUMBA and os.environ.get("NP_GEMMA_NO_NUMBA") != "1"


if HAVE_NUMBA:
    # Entry i gives the float32 value of the bfloat16 bit pattern i.
    BF16_LUT = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)

    @njit(cache=True, fastmath=True, parallel=True)
    def _linear_bf16_kernel(x, w_u16, lut, out):
        rows = w_u16.shape[0]
        cols = w_u16.shape[1]
        t_count = x.shape[0]
        for i in prange(rows):
            for t in range(t_count):
                acc = 0.0
                for k in range(cols):
                    acc += x[t, k] * lut[w_u16[i, k]]
                out[t, i] = acc

    @njit(cache=True, fastmath=True, parallel=True)
    def _linear_f32_kernel(x, w, out):
        rows = w.shape[0]
        cols = w.shape[1]
        t_count = x.shape[0]
        for i in prange(rows):
            for t in range(t_count):
                acc = 0.0
                for k in range(cols):
                    acc += x[t, k] * w[i, k]
                out[t, i] = acc

    def linear_bf16(x, w_u16):
        """Multiply x by W. W is raw bfloat16 data. Convert during the multiply."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty((x.shape[0], w_u16.shape[0]), dtype=np.float32)
        _linear_bf16_kernel(x, w_u16, BF16_LUT, out)
        return out

    def linear_f32(x, w):
        """Multiply x by W. W is float32 data. Use the Numba kernel."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        out = np.empty((x.shape[0], w.shape[0]), dtype=np.float32)
        _linear_f32_kernel(x, w, out)
        return out

else:
    BF16_LUT = None

    def linear_bf16(x, w_u16):
        raise RuntimeError("Numba is not available")

    def linear_f32(x, w):
        raise RuntimeError("Numba is not available")
