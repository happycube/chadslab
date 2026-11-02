"""DiffusionGemma's sampling: the canvases one after the other, each denoised
in up to max_steps decoder passes (transformers' generation_diffusion_gemma.py,
Copyright 2026 the HuggingFace Team; Apache License 2.0; the settings of the
checkpoint's generation_config.json).

A canvas starts as uniform random tokens. Each step, counted down from N to 1:

- the decoder's logits over the canvas, given the last step's processed logits
  as self-conditioning (none at the first step), divided by the temperature
  t_min + (t_max - t_min) * step / N (0.8 down to 0.4);
- a sample of each position from the softmax, and its argmax;
- the entropy-bounded acceptance (arXiv 2505.24857): sort the positions by the
  entropy of their distribution, keep the k lowest such that the sum of their
  entropies less the largest is at most entropy_bound (0.1); those take the
  sample, the others become random tokens again;
- the adaptive stop: the argmax canvas the same as the last step's and the mean
  entropy under confidence (0.005).

The canvas's result is the argmax canvas of the last step (not the sampled
one). It goes to the encoder (appended to the cache) unless it holds an end
token; the tokens after the first end token are dropped.

The engine gives two functions: encode(ids) (append to the cache) and
decode(canvas, sc_logits) (the raw logits with the soft cap; the cache
unchanged). Both run in float32 here; the sampler's arithmetic is NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Settings:
    max_steps: int = 48
    t_max: float = 0.8
    t_min: float = 0.4
    entropy_bound: float = 0.1
    confidence: float = 0.005
    stability: int = 1
    canvas: int = 256
    vocab: int = 262144
    eos: tuple = (1, 106, 50)

    @classmethod
    def from_config(cls, gen, cfg):
        """From generation_config.json and config.json."""
        return cls(max_steps=gen.get("max_denoising_steps", 48), t_max=gen.get("t_max", 0.8),
                   t_min=gen.get("t_min", 0.4),
                   entropy_bound=gen.get("sampler_config", {}).get("entropy_bound", 0.1),
                   confidence=gen.get("confidence_threshold", 0.005),
                   stability=gen.get("stability_threshold", 1), canvas=cfg["canvas_length"],
                   vocab=cfg["text_config"]["vocab_size"],
                   eos=tuple(gen.get("eos_token_id", (1, 106, 50))))


@dataclass
class Stats:
    steps: list = field(default_factory=list)      # decoder passes of each canvas
    tokens: int = 0

    @property
    def per_pass(self):
        return self.tokens / max(1, sum(self.steps))


def log_softmax(z):
    z = z - z.max(-1, keepdims=True)
    return z - np.log(np.exp(z).sum(-1, keepdims=True))


def entropy(logp):
    """The entropy of each row of log-probabilities (a masked token, -inf, adds 0)."""
    p = np.exp(logp)
    return -(p * np.where(p > 0, logp, 0)).sum(-1)


def sample_rows(logp, u):
    """One sample of each row by the inverse of the cumulative sum, at the uniforms u."""
    p = np.exp(logp.astype(np.float64))
    c = np.cumsum(p, axis=-1)
    return np.minimum((c < (u * c[:, -1])[:, None]).sum(-1), p.shape[1] - 1)


def accept_mask(ent, bound):
    """The positions the entropy-bounded sampler keeps."""
    order = np.argsort(ent, kind="stable")
    se = ent[order]
    keep = (np.cumsum(se) - se) <= bound
    mask = np.zeros(ent.shape, dtype=bool)
    mask[order] = keep
    return mask


def cpu_steps(decode, s):
    """The step of a canvas from the raw logits of decode(canvas, sc_logits): a function
    step(canvas, temp, ban, u) -> (entropy, argmax, sample) of each position, that keeps the
    processed logits for the self-conditioning of the next step (none at the first)."""
    eos = np.asarray(s.eos)
    state = {"sc": None}

    def step(cur, temp, ban, u):
        z = decode(cur, state["sc"]) / np.float32(temp)
        if ban.any():
            z[np.ix_(np.nonzero(ban)[0], eos)] = -np.inf
        logp = log_softmax(z)
        state["sc"] = z
        return entropy(logp), z.argmax(-1), sample_rows(logp, u)
    return step


def denoise(decode, s, rng, on_step=None, fixed=None, no_end=0, step=None):
    """Denoise one canvas. Returns (the argmax canvas, the steps run).

    decode(canvas, sc_logits) gives the raw logits (cpu_steps); or step, a function of a
    canvas (cpu_steps' form: the GPU's DG_SAMPLE gives the three rows itself).
    fixed: None, or the canvas's pinned tokens (-1 where free): they start in
    place, are never renoised, and are the result there; the entropy bound and
    the stopping test count only the free positions. no_end: the end tokens
    cannot be chosen at positions before it (the last pinned token is after)."""
    if step is None:
        step = cpu_steps(decode, s)
    cur = rng.integers(0, s.vocab, s.canvas)
    pin = None if fixed is None else np.asarray(fixed) >= 0
    if pin is not None and not pin.any():
        pin = None
    if pin is not None:
        cur[pin] = fixed[pin]
        if pin.all():
            return np.asarray(fixed).copy(), 0
    free = None if pin is None else ~pin
    ban = np.arange(s.canvas) < no_end
    prev = None
    steps = 0
    am = cur
    for k in range(s.max_steps, 0, -1):
        temp = s.t_min + (s.t_max - s.t_min) * (k / s.max_steps)
        ent, am, draw = step(cur, temp, ban, rng.random(s.canvas))
        am, draw = np.array(am), np.array(draw)
        steps += 1
        if free is None:
            keep = accept_mask(ent, s.entropy_bound)
            conf = ent.mean()
        else:
            keep = np.ones(s.canvas, dtype=bool)
            keep[free] = accept_mask(ent[free], s.entropy_bound)
            am[pin] = fixed[pin]
            draw[pin] = fixed[pin]
            conf = ent[free].mean()
        cur = np.where(keep, draw, rng.integers(0, s.vocab, s.canvas))
        stable = s.stability == 0 or (prev is not None and np.array_equal(prev, am))
        prev = am
        if on_step is not None:
            on_step(k, am, keep, ent)
        if stable and conf < s.confidence:
            break
    return am, steps


def parse_template(text, encode):
    """A response template: text with holes [[n]] of n free tokens. Returns the
    tokens with -1 for each free one. encode: text -> ids (no special tokens)."""
    import re
    out = []
    for part in re.split(r"(\[\[\d+\]\])", text):
        if not part:
            continue
        m = re.fullmatch(r"\[\[(\d+)\]\]", part)
        out.extend([-1] * int(m.group(1)) if m else list(encode(part)))
    return np.asarray(out, dtype=np.int64)


def generate(encode, decode, prompt, max_new, s, rng, on_canvas=None, on_step=None, fixed=None,
             steps=None):
    """Encode the prompt, then canvases until an end token or max_new tokens.
    Returns (the new tokens, Stats). The end token is kept, as transformers'
    generate keeps it (the pad tokens after it are dropped).

    fixed: the response's pinned tokens from its start (-1 where free; after
    its end every token is free; parse_template). A pinned start is a prefix:
    it goes to the encoder with the prompt and is the start of the result. The
    others are pinned in their canvases (denoise), and no end token comes
    before the last of them. steps: None (decode's logits, cpu_steps), or a function that gives
    a new step function (denoise's step) for each canvas."""
    out = []
    fixed = None if fixed is None else np.asarray(fixed, dtype=np.int64)
    prompt = np.asarray(prompt, dtype=np.int64)
    if fixed is not None:
        k = 0
        while k < fixed.size and fixed[k] >= 0:
            k += 1
        out = fixed[:k].tolist()
        prompt = np.concatenate([prompt, fixed[:k]])
        fixed = fixed[k:]
        if not (fixed >= 0).any():
            fixed = None
    last = -1 if fixed is None else int(np.nonzero(fixed >= 0)[0][-1])
    encode(prompt)
    st = Stats()
    canvases = -(-max(0, max_new - len(out)) // s.canvas)
    canvases = max(canvases, -(-(last + 1) // s.canvas))
    for c in range(canvases):
        f = None
        if fixed is not None and c * s.canvas <= last:
            f = np.full(s.canvas, -1, dtype=np.int64)
            seg = fixed[c * s.canvas:(c + 1) * s.canvas]
            f[:seg.size] = seg
        no_end = min(s.canvas, max(0, last - c * s.canvas))
        am, n = denoise(decode, s, rng, on_step, fixed=f, no_end=no_end,
                        step=None if steps is None else steps())
        st.steps.append(n)
        ends = np.nonzero(np.isin(am[no_end:], s.eos))[0] + no_end
        if ends.size:
            got = am[:ends[0] + 1].tolist()
        else:
            got = am.tolist()
        out.extend(got)
        if on_canvas is not None:
            on_canvas(c, got)
        if ends.size or (len(out) >= max_new and c * s.canvas >= last):
            break
        encode(am.astype(np.int64))
    if last < 0:
        out = out[:max_new]
    st.tokens = len(out)
    return out, st
