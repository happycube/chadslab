"""The scheduler of Wan 2.2's pipeline: diffusers' UniPCMultistepScheduler with flow sigmas
(use_flow_sigmas, flow_shift, prediction_type flow_prediction, predict_x0, solver bh2 of order 2,
lower_order_final, final_sigmas_type zero), on the host in float32 as diffusers' (torch's float32
tensors, each operation rounded once, its scalars as 0-d float32 tensors):

Ported from diffusers (src/diffusers/schedulers/scheduling_unipc_multistep.py; Apache License
2.0; Copyright 2025 TSAIL Team and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    the sigmas: NumPy float64 linspace(1, 1 / 1000, steps + 1) but the last, shifted
        (shift s / (1 + (shift - 1) s)), the first less 1e-6 when it is 1, then float32 with a
        last sigma of 0; the timesteps the float64 sigmas times 1000, truncated to integers
    a step: the model's flow output to x0 (sample - sigma output), the corrector (UniC) of the
        last step's sample, the predictor (UniP) to the next sigma; order 1 at the first and the
        last step, else 2; the corrector's two weights the solution of its 2 x 2 system
        (torch.linalg.solve: MKL's sgesv, written out in float32)

The scalars' log and expm1 as torch's on the CPU: log the float64 value rounded to float32
(torch's is MKL's vmsLn of high accuracy, not correctly rounded and not open: 11 of the 10100
values of the schedules of 1 to 100 steps an ulp apart, so the step coefficients of 49, 95 and 98
steps differ, a corrector weight by at most 2.2e-4 of itself where the 2 x 2 system is nearly
singular; the other step counts, 25 and 50 among them, the same bits); expm1 Sleef's expm1f_u10
(AVX-512 with FMA), written out in float32 (the bits of torch's on 10000 values and on every
step of 2 to 100 steps). check_wan.py sched compares the steps' bits.
"""
from __future__ import annotations

from fractions import Fraction

import numpy as np

F = np.float32


def _log(x):
    with np.errstate(divide="ignore"):
        return F(np.log(np.float64(F(x))))


def _fma(a, b, c):
    """a b + c (float32) rounded once to float32 (the nearer neighbour of the exact value, ties to
    even)."""
    s = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    f = F(float(s))
    if not np.isfinite(f) or Fraction(float(f)) == s:
        return f
    lo, hi = (np.nextafter(f, F(-np.inf)), f) if Fraction(float(f)) > s else (f, np.nextafter(f, F(np.inf)))
    dl, dh = s - Fraction(float(lo)), Fraction(float(hi)) - s
    if dl != dh:
        return lo if dl < dh else hi
    return lo if (lo.view(np.int32) & 1) == 0 else hi


# Sleef's double-float operations (sleefsimdsp.c, FMA): a pair (x, y) is the value x + y
def _add2_df(x, y):
    s = F(x[0] + y)
    v = F(s - x[0])
    return s, F(F(F(x[0] - F(s - v)) + F(y - v)) + x[1])


def _add2_dd(x, y):
    s = F(x[0] + y[0])
    v = F(s - x[0])
    return s, F(F(F(x[0] - F(s - v)) + F(y[0] - v)) + F(x[1] + y[1]))


def _add_fd(x, y):
    s = F(x + y[0])
    return s, F(F(F(x - s) + y[0]) + y[1])


def _mul_df(x, y):
    s = F(x[0] * y)
    return s, _fma(x[1], y, _fma(x[0], y, -s))


def _mul_dd(x, y):
    s = F(x[0] * y[0])
    return s, _fma(x[0], y[1], _fma(x[1], y[0], _fma(x[0], y[0], -s)))


def _squ(x):
    s = F(x[0] * x[0])
    return s, _fma(F(x[0] + x[0]), x[1], _fma(x[0], x[0], -s))


def _ldexp2(x, q):
    return F(F(x * F(2.0 ** (q >> 1))) * F(2.0 ** (q - (q >> 1))))


def _expm1(a):
    """Sleef's xexpm1f (expk2f of (a, 0), less 1) in float32."""
    a = F(a)
    if a > F(88.72283172607421875):
        return F(np.inf)
    if a < F(-16.635532333438687426013570):
        return F(-1)
    if a == 0:
        return a
    u = F(a * F(1.442695040888963407359924681001892137426645954152985934135449406931))
    q = int(np.rint(u))
    s = _add2_df((a, F(0)), F(F(q) * -F(0.693145751953125)))
    s = _add2_df(s, F(F(q) * -F(1.428606765330187045e-06)))
    u = F(0.1980960224e-3)
    for c in (0.1394256484e-2, 0.8333456703e-2, 0.4166637361e-1):
        u = _fma(u, s[0], F(c))
    t = _add2_df(_mul_df(s, u), F(0.166666659414234244790680580464e+0))
    t = _add2_df(_mul_dd(s, t), F(0.5))
    t = _add2_dd(s, _mul_dd(_squ(s), t))
    t = _add_fd(F(1), t)
    d = _add2_df((_ldexp2(t[0], q), _ldexp2(t[1], q)), F(-1.0))
    return F(d[0] + d[1])


def solve2(R, b):
    """x of R x = b (2 x 2, float32) as torch.linalg.solve on the CPU (MKL's sgesv): the
    partial pivot, the reciprocal of the first pivot, fused multiply-adds in the elimination and
    the substitutions, the divisions as products of reciprocals (the bits of torch's on every
    system of the schedules of 8 to 100 steps)."""
    a = np.array(R, F)
    y = np.array(b, F)
    if abs(a[1, 0]) > abs(a[0, 0]):
        a = a[::-1].copy()
        y = y[::-1].copy()
    r = F(F(1) / a[0, 0])
    l_ = F(a[1, 0] * r)
    u22 = _fma(-l_, a[0, 1], a[1, 1])
    y1 = _fma(-y[0], l_, y[1])
    x1 = F(y1 * F(F(1) / u22))
    y0 = _fma(-x1, a[0, 1], y[0])
    x0 = F(y0 * r)
    return np.array([x0, x1], F)


class UniPC:
    """The scheduler of a scheduler_config.json (Wan 2.2's)."""

    def __init__(self, cfg):
        assert cfg.get("use_flow_sigmas") and cfg.get("prediction_type") == "flow_prediction"
        assert cfg.get("predict_x0", True) and cfg.get("solver_type", "bh2") == "bh2"
        assert cfg.get("solver_order", 2) == 2 and cfg.get("lower_order_final", True)
        assert cfg.get("final_sigmas_type", "zero") == "zero" and not cfg.get("use_dynamic_shifting")
        assert not cfg.get("thresholding") and not cfg.get("disable_corrector") and not cfg.get("shift_terminal")
        assert not cfg.get("use_karras_sigmas") and not cfg.get("use_exponential_sigmas")
        assert not cfg.get("use_beta_sigmas")
        self.train = cfg.get("num_train_timesteps", 1000)
        self.shift = float(cfg.get("flow_shift", 1.0))

    def set_timesteps(self, steps):
        """The sigmas (steps + 1, float32) and the timesteps (steps, int64) of a run; a new run."""
        s = np.linspace(1, 1 / self.train, steps + 1)[:-1]
        s = self.shift * s / (1 + (self.shift - 1) * s)
        if np.fabs(s[0] - 1) < 1e-6:
            s[0] -= 1e-6
        self.timesteps = (s * self.train).astype(np.int64)
        self.sigmas = np.concatenate([s, [0]]).astype(F)
        self.steps = steps
        self.outputs = [None, None]
        self.lower = 0
        self.last = None
        self.index = 0
        self.order = None
        return self.sigmas, self.timesteps

    def _lam(self, i):
        """alpha, sigma and lambda (log alpha - log sigma) of sigma i, float32."""
        s = self.sigmas[i]
        a = F(F(1) - s)
        return a, s, F(_log(a) - _log(s))

    def _coefs(self, h):
        """hh = -h, h phi_1 = expm1(hh), B_h = expm1(hh) (bh2), and the weights b of orders 1 and
        2 (the loop of diffusers' updates)."""
        hh = F(-h)
        hp1 = _expm1(hh)
        bh = _expm1(hh)
        hpk = F(F(hp1 / hh) - F(1))
        b, fact = [], 1
        for i in (1, 2):
            b.append(F(F(hpk * F(fact)) / bh))
            fact *= i + 1
            hpk = F(F(hpk / hh) - F(1 / fact))
        return hh, hp1, bh, b

    def _predict(self, x, order):
        """multistep_uni_p_bh_update of the sample x (predict_x0, bh2)."""
        i = self.index
        m0 = self.outputs[-1]
        at, st, lt = self._lam(i + 1)
        a0, s0, l0 = self._lam(i)
        with np.errstate(invalid="ignore"):
            h = F(lt - l0)
        D1 = None
        if order == 2:
            ai, si, li = self._lam(i - 1)
            rk = F(F(li - l0) / h)
            D1 = ((self.outputs[-2] - m0) / rk).astype(F)
        with np.errstate(divide="ignore", invalid="ignore"):
            hh, hp1, bh, _ = self._coefs(h)
        xt = (F(st / s0) * x - F(at * hp1) * m0).astype(F)
        if D1 is not None:
            # rhos_p 0.5 (order 2)
            xt = (xt - F(at * bh) * (F(0.5) * D1)).astype(F)
        return xt

    def _correct(self, m, last, x, order):
        """multistep_uni_c_bh_update: the corrected sample x of the converted output m, the last
        step's sample last."""
        i = self.index
        m0 = self.outputs[-1]
        at, st, lt = self._lam(i)
        a0, s0, l0 = self._lam(i - 1)
        h = F(lt - l0)
        hh, hp1, bh, b = self._coefs(h)
        D1 = None
        if order == 2:
            ai, si, li = self._lam(i - 2)
            rk = F(F(li - l0) / h)
            D1 = ((self.outputs[-2] - m0) / rk).astype(F)
            rhos = solve2([[F(1), F(1)], [rk, F(1)]], b)
        else:
            rhos = np.array([F(0.5)], F)
        xt = (F(st / s0) * last - F(at * hp1) * m0).astype(F)
        d1t = (m - m0).astype(F)
        corr = rhos[-1] * d1t
        if D1 is not None:
            corr = (rhos[0] * D1).astype(F) + corr
        return (xt - F(at * bh) * corr.astype(F)).astype(F)

    def step(self, out, x):
        """The latents after the current step of the model's flow output out (after the guidance)
        of latents x, float32."""
        i = self.index
        out, x = np.asarray(out, F), np.asarray(x, F)
        m = (x - self.sigmas[i] * out).astype(F)
        if i > 0 and self.last is not None:
            x = self._correct(m, self.last, x, self.order)
        self.outputs = [self.outputs[1], m]
        order = min(2, self.steps - i)
        self.order = min(order, self.lower + 1)
        self.last = x
        prev = self._predict(x, self.order)
        self.lower = min(self.lower + 1, 2)
        self.index += 1
        return prev
