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
    step(canvas, temp, ban, u) -> (entropy, argmax, sample, the argmax's probability, its
    probability under the raw logits: no temperature, no ban) of each position, that keeps the processed logits for the self-conditioning of the next step (none at
    the first)."""
    eos = np.asarray(s.eos)
    state = {"sc": None}

    def step(cur, temp, ban, u):
        raw = decode(cur, state["sc"])
        z = raw / np.float32(temp)
        if ban.any():
            z[np.ix_(np.nonzero(ban)[0], eos)] = -np.inf
        logp = log_softmax(z)
        state["sc"] = z
        am = z.argmax(-1)
        praw = np.exp(log_softmax(raw)[np.arange(len(am)), am])
        return entropy(logp), am, sample_rows(logp, u), np.exp(logp.max(-1)), praw
    return step


def denoise_iter(decode, s, rng, on_step=None, fixed=None, no_end=0, step=None, events=False):
    """denoise as a generator: with events, a dict after each step (step: the steps left; draft: the
    argmax canvas; prob: each position's probability of its draft token, of the processed logits
    (None when the step function gives none); raw: its probability under the raw logits (the
    model's: no temperature, no ban; or None); pinned positions have 1 for both; entropy; kept: the positions the sampler kept);
    its return value (StopIteration) is (the argmax canvas, the steps run)."""
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
        res = step(cur, temp, ban, rng.random(s.canvas))
        ent, am, draw = res[:3]
        pmax = res[3] if len(res) > 3 else None
        praw = res[4] if len(res) > 4 else None
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
        if events:
            pm, pr = (None if v is None else np.array(v, dtype=np.float32) for v in (pmax, praw))
            for v in (pm, pr):
                if v is not None and pin is not None:
                    v[pin] = 1.0
            yield {"step": k, "draft": am.copy(), "prob": pm, "raw": pr, "entropy": np.array(ent, dtype=np.float32),
                   "kept": keep.copy()}
        if stable and conf < s.confidence:
            break
    return am, steps


def denoise(decode, s, rng, on_step=None, fixed=None, no_end=0, step=None):
    """Denoise one canvas. Returns (the argmax canvas, the steps run).

    decode(canvas, sc_logits) gives the raw logits (cpu_steps); or step, a function of a
    canvas (cpu_steps' form: the GPU's DG_SAMPLE gives the rows itself).
    fixed: None, or the canvas's pinned tokens (-1 where free): they start in
    place, are never renoised, and are the result there; the entropy bound and
    the stopping test count only the free positions. no_end: the end tokens
    cannot be chosen at positions before it (the last pinned token is after)."""
    g = denoise_iter(decode, s, rng, on_step, fixed, no_end, step)
    while True:
        try:
            next(g)
        except StopIteration as e:
            return e.value


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


def generate_iter(encode, decode, prompt, max_new, s, rng, on_step=None, fixed=None, steps=None, stats=None,
                  events=False):
    """generate's canvases one at a time: yield the tokens of the answer as each canvas is done (a
    pinned start first: the prefix), at most max_new tokens unless pinned tokens come later (they
    are all kept). stats (or None): a Stats that gets each canvas's passes and the tokens. events:
    also a dict after each denoising step (denoise_iter's, with "canvas": its index), before the
    canvas's tokens."""
    st = stats if stats is not None else Stats()
    fixed = None if fixed is None else np.asarray(fixed, dtype=np.int64)
    prompt = np.asarray(prompt, dtype=np.int64)
    head = []
    if fixed is not None:
        k = 0
        while k < fixed.size and fixed[k] >= 0:
            k += 1
        head = fixed[:k].tolist()
        prompt = np.concatenate([prompt, fixed[:k]])
        fixed = fixed[k:]
        if not (fixed >= 0).any():
            fixed = None
    last = -1 if fixed is None else int(np.nonzero(fixed >= 0)[0][-1])
    encode(prompt)
    n_out = len(head)
    if head:
        st.tokens += len(head)
        yield head
    canvases = -(-max(0, max_new - n_out) // s.canvas)
    canvases = max(canvases, -(-(last + 1) // s.canvas))
    for c in range(canvases):
        f = None
        if fixed is not None and c * s.canvas <= last:
            f = np.full(s.canvas, -1, dtype=np.int64)
            seg = fixed[c * s.canvas:(c + 1) * s.canvas]
            f[:seg.size] = seg
        no_end = min(s.canvas, max(0, last - c * s.canvas))
        it = denoise_iter(decode, s, rng, on_step, fixed=f, no_end=no_end,
                          step=None if steps is None else steps(), events=events)
        while True:
            try:
                ev = next(it)
            except StopIteration as e:
                am, n = e.value
                break
            ev["canvas"] = c
            yield ev
        st.steps.append(n)
        ends = np.nonzero(np.isin(am[no_end:], s.eos))[0] + no_end
        got = am[:ends[0] + 1].tolist() if ends.size else am.tolist()
        if last < 0 and n_out + len(got) > max_new:
            got = got[:max(0, max_new - n_out)]
        n_out += len(got)
        st.tokens += len(got)
        yield got
        if ends.size or (n_out >= max_new and c * s.canvas >= last):
            return
        encode(am.astype(np.int64))


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
    st = Stats()
    out = []
    pinned_start = fixed is not None and len(fixed) and fixed[0] >= 0
    for c, got in enumerate(generate_iter(encode, decode, prompt, max_new, s, rng, on_step, fixed, steps, st)):
        out.extend(got)
        if on_canvas is not None and not (pinned_start and c == 0):
            on_canvas(len(st.steps) - 1, got)
    return out, st
