"""Select the next token from the logits.

The settings match the OpenAI API: temperature, top_k, top_p, min_p, and the
three penalties. A temperature of zero selects the most probable token. The
class keeps the count of each token in the history for the penalties.
"""
from __future__ import annotations

import numpy as np

from . import ops


def _softmax(x):
    """Return the softmax of one row of float64 values."""
    m = float(np.max(x))
    e = np.exp(x - m)
    s = float(np.sum(e))
    if s <= 0.0:
        return np.full(x.shape, 1.0 / x.size, dtype=np.float64)
    return e / s


class Sampler:
    """Hold the sampling settings and the random state of one generation.

    temperature        Divide the logits. Zero selects the best token.
    top_k              Keep the k most probable tokens.
    top_p              Keep the smallest set with this share of the mass.
    min_p              Drop a token below this share of the best probability.
    repetition_penalty Divide a positive logit and multiply a negative one for
                       a token that the history holds.
    presence_penalty   Subtract this value for a token in the history.
    frequency_penalty  Subtract this value for each use in the history.
    seed               Seed the random generator.
    """

    def __init__(self, temperature=1.0, top_k=None, top_p=None, min_p=None,
                 repetition_penalty=None, presence_penalty=None,
                 frequency_penalty=None, seed=None):
        self.temperature = float(temperature)
        self.top_k = None if top_k in (None, 0) else max(int(top_k), 1)
        self.top_p = None if top_p in (None, 0.0, 1.0) else float(top_p)
        self.min_p = None if min_p in (None, 0.0) else float(min_p)
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.rng = np.random.default_rng(seed)
        self.counts = {}

    @property
    def greedy(self):
        """Return True when the sampler always selects the best token."""
        return (self.temperature <= 0.0 and self.repetition_penalty in (None, 1.0)
                and not self.presence_penalty and not self.frequency_penalty)

    def reset(self, ids=()):
        """Start a new history. Use the prompt tokens as the history."""
        self.counts = {}
        for i in ids:
            i = int(i)
            self.counts[i] = self.counts.get(i, 0) + 1

    def _penalize(self, scores):
        """Change the scores of the tokens that the history holds."""
        if not self.counts:
            return scores
        ids = np.fromiter(self.counts.keys(), dtype=np.int64, count=len(self.counts))
        n = np.fromiter(self.counts.values(), dtype=np.float64, count=len(self.counts))
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            v = scores[ids]
            scores[ids] = np.where(v > 0.0, v / self.repetition_penalty,
                                   v * self.repetition_penalty)
        if self.presence_penalty:
            scores[ids] -= self.presence_penalty
        if self.frequency_penalty:
            scores[ids] -= self.frequency_penalty * n
        return scores

    def __call__(self, logits):
        """Return the id of the next token and add it to the history."""
        if self.greedy:
            # No copy to float64: the argmax of the float32 values is the same.
            token = ops.argmax(np.ascontiguousarray(logits, dtype=np.float32).reshape(-1))
            self.counts[token] = self.counts.get(token, 0) + 1
            return token
        scores = np.asarray(logits, dtype=np.float64).copy()
        scores = self._penalize(scores)
        if self.temperature <= 0.0:
            token = ops.argmax(scores)
        else:
            scores /= self.temperature
            if self.top_k is not None:
                k = min(self.top_k, scores.size)
                cand = np.argpartition(scores, -k)[-k:]
            else:
                cand = np.arange(scores.size)
            order = np.argsort(scores[cand])[::-1]
            cand = cand[order]
            probs = _softmax(scores[cand])
            if self.top_p is not None:
                c = np.cumsum(probs)
                cut = int(np.searchsorted(c, self.top_p)) + 1
                probs[cut:] = 0.0
            if self.min_p is not None:
                probs[probs < self.min_p * probs[0]] = 0.0
            total = float(probs.sum())
            if total <= 0.0:
                token = int(cand[0])
            else:
                token = int(cand[self.rng.choice(probs.size, p=probs / total)])
        self.counts[token] = self.counts.get(token, 0) + 1
        return token
