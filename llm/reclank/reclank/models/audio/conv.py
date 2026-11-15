"""Convs over rows (time) as the forms of an encoder program: the encoders of
long audio (the Higgs codec's DAC encoder at 24 kHz, HuBERT's feature
extractor at 16 kHz).

A conv of k taps (stride s, dilation d, padding p) is the gather of its
taps (ENC_GATHER: output row t reads input rows s t - p + d j, -1 past the
ends: zeros) and ENC_LINEAR of W (out, k in) (dac.conv_lin), in tiles of
output rows whose gathered rows fit the compiler's buffer "_col" (tile
values): an encoder of a minute of 24 kHz audio needs no gigabytes of
im2col.
"""
from __future__ import annotations

import numpy as np

TILE = 1 << 22          # the values of "_col": 16 MB


def conv_rows(rows, k, stride=1, pad=0, dil=1):
    """The output rows of a conv (torch's Conv1d)."""
    return (rows + 2 * pad - dil * (k - 1) - 1) // stride + 1


class Convs:
    """The forms of convs into a compiler c (an _EncCompiler): forms holds
    them in order; view(base, rows, cols) names a view of an env array."""

    def __init__(self, c, tile=TILE):
        self.c, self.env, self.tile = c, c.env, tile
        self.forms = []
        self.count = 0
        self.env["_col"] = np.zeros(tile, np.float32)

    def view(self, base, rows, cols, start=0):
        """A name of rows x cols values of env[base] from value start on."""
        self.count += 1
        name = "_cv%d" % self.count
        a = self.env[base].reshape(-1)
        self.env[name] = a[start:start + rows * cols].reshape(rows, cols)
        return name

    def rows_of(self, name, r0, r1):
        """A name of the rows [r0, r1) of the 2-D env[name]."""
        self.count += 1
        new = "_cv%d" % self.count
        self.env[new] = self.env[name][r0:r1]
        return new

    def gather_linear(self, x, idx, cols, lin, out):
        """out = lin of the gather of x's rows (cols values) by idx (rows,
        blocks), in tiles of rows."""
        n, blocks = idx.shape
        step = self.tile // (blocks * cols)
        assert step >= 1, "a row of the gather is past _col"
        for r0 in range(0, n, step):
            r1 = min(n, r0 + step)
            cv = self.view("_col", r1 - r0, blocks * cols)
            ov = out if (r0 == 0 and r1 == n) else self.rows_of(out, r0, r1)
            self.forms += [("set", cv, ("enc_gather", x, np.ascontiguousarray(idx[r0:r1]), cols)),
                           ("set", ov, ("enc_linear", lin, cv))]

    def conv(self, x, rows, cin, lin, out, stride=1, pad=None, dil=1):
        """out = the conv lin (a conv_lin: W (out, k cin), lin.k taps) of x
        (rows, cin); pad None: (k - 1) d / 2 (the same rows at stride 1).
        Return the output rows."""
        k = lin.k
        pad = (k - 1) * dil // 2 if pad is None else pad
        n = conv_rows(rows, k, stride, pad, dil)
        if k == 1 and stride == 1 and pad == 0:
            self.forms.append(("set", out, ("enc_linear", lin, x)))
            return n
        t = np.arange(n)[:, None] * stride - pad + dil * np.arange(k)[None, :]
        idx = np.where((t >= 0) & (t < rows), t, -1).astype(np.int32)
        self.gather_linear(x, idx, cin, lin, out)
        return n
