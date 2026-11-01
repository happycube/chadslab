"""The schedulers of Stable Diffusion's pipelines, on the host in float32 as
diffusers' (torch's float32 tensors, each operation rounded):

Ported from diffusers (src/diffusers/schedulers/scheduling_pndm.py, Copyright
2025 Zhejiang University Team and The HuggingFace Team;
schedulers/scheduling_ddim.py, Copyright 2025 Stanford University Team and The
HuggingFace Team; Apache License 2.0). See THIRD_PARTY_NOTICES.md.

    PNDMScheduler with skip_prk_steps (SD 1.5, 2.0-base: PLMS, the linear
        multistep of up to four outputs; its timesteps the "leading" ones and
        the next to last twice, steps + 1 UNet calls)
    DDIMScheduler (SD 2.0-768), eta 0

with epsilon or v_prediction, the betas "scaled_linear" (torch.linspace of
float32 of the square roots, squared), alphas_cumprod a cumulative product
in double (torch's accumulation on the CPU) rounded to float32.
"""
from __future__ import annotations

import numpy as np

F = np.float32


def torch_linspace(start, end, steps):
    """torch.linspace(start, end, steps, dtype=float32): the step in float32,
    the first half start + step i, the second end - step (steps - 1 - i),
    each rounded once (a fused multiply-add: the same bits as torch 2.15's
    on the CPU)."""
    start, end = F(start), F(end)
    step = np.float64(F((end - start) / F(steps - 1)))
    i = np.arange(steps)
    lo = np.float64(start) + step * i
    hi = np.float64(end) - step * (steps - 1 - i)
    return np.where(i < steps // 2, lo, hi).astype(F)


class Scheduler:
    """A scheduler of a scheduler_config.json; kind: "PNDMScheduler" or
    "DDIMScheduler" (the pipeline's model_index.json)."""

    def __init__(self, cfg, kind):
        assert kind in ("PNDMScheduler", "DDIMScheduler"), kind
        assert cfg.get("beta_schedule") == "scaled_linear" and not cfg.get("trained_betas")
        assert cfg.get("timestep_spacing", "leading") == "leading" and not cfg.get("clip_sample")
        assert not cfg.get("thresholding") and not cfg.get("rescale_betas_zero_snr")
        assert kind == "DDIMScheduler" or cfg.get("skip_prk_steps")
        self.kind, self.cfg = kind, cfg
        self.train = cfg.get("num_train_timesteps", 1000)
        self.offset = cfg.get("steps_offset", 0)
        self.pred = cfg.get("prediction_type", "epsilon")
        assert self.pred in ("epsilon", "v_prediction")
        betas = (torch_linspace(cfg["beta_start"] ** 0.5, cfg["beta_end"] ** 0.5, self.train) ** 2).astype(F)
        alphas = (F(1) - betas).astype(F)
        self.alphas_cumprod = np.cumprod(alphas.astype(np.float64)).astype(F)
        self.final = F(1) if cfg.get("set_alpha_to_one", True) else self.alphas_cumprod[0]

    def timesteps(self, steps):
        """The timesteps of the UNet's calls (int64)."""
        ratio = self.train // steps
        t = (np.arange(steps) * ratio).round().astype(np.int64) + self.offset
        if self.kind == "DDIMScheduler":
            return t[::-1].copy()
        # PLMS: the next to last twice, then reversed
        return np.concatenate([t[:-1], t[-2:-1], t[-1:]])[::-1].copy()

    def start(self, steps):
        """A new run of steps."""
        self.steps = steps
        self.ets, self.counter, self.cur = [], 0, None

    def _alpha(self, t):
        return self.alphas_cumprod[t] if t >= 0 else self.final

    def step(self, out, t, x):
        """The latents after the timestep t of the model's output out (after
        the guidance) of latents x, float32."""
        t = int(t)
        prev = t - self.train // self.steps
        if self.kind == "DDIMScheduler":
            a, ap = self._alpha(t), self._alpha(prev)
            b = F(1) - a
            if self.pred == "epsilon":
                x0 = (x - b ** F(0.5) * out) / a ** F(0.5)
                eps = out
            else:
                x0 = a ** F(0.5) * x - b ** F(0.5) * out
                eps = a ** F(0.5) * out + b ** F(0.5) * x
            # eta 0: std_dev_t 0, (1 - alpha_prev - 0) ** 0.5
            return (ap ** F(0.5) * x0 + (F(1) - ap - F(0)) ** F(0.5) * eps).astype(F)
        # PNDMScheduler.step_plms
        if self.counter != 1:
            self.ets = self.ets[-3:]
            self.ets.append(out)
        else:
            prev = t
            t = t + self.train // self.steps
        e = self.ets
        if len(e) == 1 and self.counter == 0:
            self.cur = x
        elif len(e) == 1 and self.counter == 1:
            out = (out + e[-1]) / F(2)
            x, self.cur = self.cur, None
        elif len(e) == 2:
            out = (F(3) * e[-1] - e[-2]) / F(2)
        elif len(e) == 3:
            out = (F(23) * e[-1] - F(16) * e[-2] + F(5) * e[-3]) / F(12)
        else:
            out = F(1 / 24) * (F(55) * e[-1] - F(59) * e[-2] + F(37) * e[-3] - F(9) * e[-4])
        self.counter += 1
        a, ap = self._alpha(t), self._alpha(prev)
        b, bp = F(1) - a, F(1) - ap
        if self.pred == "v_prediction":
            out = a ** F(0.5) * out + b ** F(0.5) * x
        coef = (ap / a) ** F(0.5)
        den = a * bp ** F(0.5) + (a * b * ap) ** F(0.5)
        return (coef * x - (ap - a) * out / den).astype(F)
