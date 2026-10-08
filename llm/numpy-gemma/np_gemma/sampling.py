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


def _top_idx(x, k):
    """Return the ids of the k largest values of the float32 row x, in no
    order. A threshold from every 64th value first keeps a few hundred ids,
    and the partition runs on them: np.argpartition on a row of 262144 values
    took 0.65 ms. Too few ids above the threshold use the whole row."""
    n = x.size
    if k * 64 >= n:
        return np.argpartition(x, -k)[-k:]
    smp = x[::64]
    r = min(smp.size, k // 64 + 4)
    thr = np.partition(smp, -r)[-r]
    idx = np.flatnonzero(x >= thr)
    if idx.size < k:
        return np.argpartition(x, -k)[-k:]
    return idx[np.argpartition(x[idx], -k)[-k:]]


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
    mtp_accept         "exact" (the default): an MTP draft stays only when the
                       sample picks it, so the text follows the distribution
                       of the settings. "in_set": a draft also stays when the
                       settings allow it (draft_ok). That keeps more drafts,
                       but the text moves toward the choices of the drafter.
    mtp_floor          With "in_set": a draft stays only when its probability
                       is at least this share of the best probability.
    """

    def __init__(self, temperature=1.0, top_k=None, top_p=None, min_p=None,
                 repetition_penalty=None, presence_penalty=None,
                 frequency_penalty=None, seed=None, mtp_accept="exact", mtp_floor=None):
        self.temperature = float(temperature)
        self.top_k = None if top_k in (None, 0) else max(int(top_k), 1)
        self.top_p = None if top_p in (None, 0.0, 1.0) else float(top_p)
        self.min_p = None if min_p in (None, 0.0) else float(min_p)
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.rng = np.random.default_rng(seed)
        self.counts = {}
        if mtp_accept not in ("exact", "in_set"):
            raise ValueError("mtp_accept must be exact or in_set")
        self.mtp_accept = mtp_accept
        # guard(token) -> token (or None): a rule that may put another token in
        # the place of the picked one (serve_qwen4.ThinkGuard); fixed applies it
        self.guard = None
        self.mtp_floor = None if mtp_floor in (None, 0.0) else float(mtp_floor)

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

    def _penalty_on(self):
        """Return True when a penalty changes some scores."""
        return bool(self.counts) and (self.repetition_penalty not in (None, 1.0)
                                      or bool(self.presence_penalty)
                                      or bool(self.frequency_penalty))

    def _penalize_vals(self, v, n):
        """Return the float64 scores v of tokens with the history counts n,
        after the penalties."""
        if self.repetition_penalty and self.repetition_penalty != 1.0:
            v = np.where(v > 0.0, v / self.repetition_penalty, v * self.repetition_penalty)
        if self.presence_penalty:
            v = v - self.presence_penalty
        if self.frequency_penalty:
            v = v - self.frequency_penalty * n
        return v

    def _scores(self, logits):
        """Return the float32 scores of a row after the penalties. Only the
        tokens of the history change, so a row without penalties is not
        copied."""
        x = np.asarray(logits, dtype=np.float32).reshape(-1)
        if not self._penalty_on():
            return x
        x = x.copy()
        ids = np.fromiter(self.counts.keys(), dtype=np.int64, count=len(self.counts))
        n = np.fromiter(self.counts.values(), dtype=np.float64, count=len(self.counts))
        x[ids] = self._penalize_vals(x[ids].astype(np.float64), n).astype(np.float32)
        return x

    def _cut(self, cand, probs):
        """Apply top_p and min_p to the probabilities of cand (in the order of
        their scores)."""
        if self.top_p is not None:
            c = np.cumsum(probs)
            cut = int(np.searchsorted(c, self.top_p)) + 1
            probs[cut:] = 0.0
        if self.min_p is not None:
            probs[probs < self.min_p * probs[0]] = 0.0
        return cand, probs

    def _dist(self, logits):
        """Return the candidate ids and their probabilities after the
        penalties, the temperature, top_k, top_p, and min_p. The ids are in
        the order of their scores, except with the temperature alone (all the
        tokens, in the order of the ids). The probabilities do not sum to one;
        zero marks a candidate that the settings drop. The temperature is above
        zero.

        Only the candidates are sorted. top_k selects them with _top_idx.
        top_p and min_p take the tokens within a width of the best score (8
        times the temperature, then 16, 32, 64, then all) until the mass
        reaches top_p, or until min_p drops every token outside. The full
        sort took 16 ms for a row of 262144 values."""
        x = self._scores(logits)
        t = self.temperature
        if self.top_k is not None:
            cand = _top_idx(x, min(self.top_k, x.size))
            s = x[cand].astype(np.float64) / t
            order = np.argsort(s)[::-1]
            cand = cand[order]
            return self._cut(cand, _softmax(s[order]))
        if self.top_p is None and self.min_p is None:
            return np.arange(x.size), _softmax(x.astype(np.float64) / t)
        m = float(x.max())
        z = float(np.exp((x - np.float32(m)) * np.float32(1.0 / t)).sum(dtype=np.float64))
        width = 8.0
        while True:
            full = width > 64.0
            cand = np.arange(x.size) if full else np.flatnonzero(x >= m - t * width)
            s = x[cand].astype(np.float64)
            order = np.argsort(s)[::-1]
            cand = cand[order]
            probs = np.exp((s[order] - m) / t) / z
            if (full or (self.top_p is not None and probs.sum() >= self.top_p)
                    or (self.min_p is not None and np.exp(-width) < self.min_p)):
                return self._cut(cand, probs)
            width *= 2.0

    def sparse_k(self):
        """Return the count of candidates that sample_sparse needs for a row
        (the GPU selects them, E4B.logits_topk), or 0 when candidates cannot
        settle a row: the temperature alone, or penalties without top_k."""
        if self.greedy or self.temperature <= 0.0:
            return 0
        if self.top_k is not None:
            # A penalty can push tokens of the history out of the best k.
            k = self.top_k + (64 if self._penalty_on() else 0)
            return k if k <= 1024 else 0
        if (self.top_p is None and self.min_p is None) or self._penalty_on():
            return 0
        return 256

    def _dist_sparse(self, ids, vals, stat):
        """Return (cand, probs) as _dist from the candidates of a row: ids and
        vals, the k largest logits (in any order), and stat, the row max and
        the sum of exp((l - max) / temperature). Every other logit is at most
        the smallest of vals. Return None when the candidates do not settle
        the result; the caller then gives the whole row to _dist."""
        t = self.temperature
        ids = np.asarray(ids).reshape(-1).astype(np.int64)
        vals = np.asarray(vals, dtype=np.float32).reshape(-1)
        b = float(vals.min())
        if self._penalty_on():
            if self.top_k is None:
                return None
            if ((self.repetition_penalty or 1.0) < 1.0 or (self.presence_penalty or 0.0) < 0.0
                    or (self.frequency_penalty or 0.0) < 0.0):
                return None            # a penalty that raises a score
            n = np.array([self.counts.get(int(i), 0) for i in ids], dtype=np.float64)
            h = n > 0
            if h.any():
                vals = vals.copy()
                vals[h] = self._penalize_vals(vals[h].astype(np.float64), n[h]).astype(np.float32)
        if self.top_k is not None:
            k = self.top_k
            if k > vals.size:
                return None
            order = np.argsort(vals.astype(np.float64))[::-1][:k]
            if float(vals[order[-1]]) < b:
                return None            # a token outside can be in the best k
            s = vals[order].astype(np.float64) / t
            return self._cut(ids[order], _softmax(s))
        m, z = float(stat[0]), float(stat[1])
        order = np.argsort(vals.astype(np.float64))[::-1]
        probs = np.exp((vals[order].astype(np.float64) - m) / t) / z
        if not ((self.top_p is not None and probs.sum() >= self.top_p)
                or (self.min_p is not None and np.exp((b - m) / t) < self.min_p)):
            return None
        return self._cut(ids[order], probs)

    def _sample(self, cand, probs):
        """Return a token of cand with the probabilities probs (not
        normalized), and add it to the history."""
        total = float(probs.sum())
        if total <= 0.0:
            token = int(cand[0])
        else:
            token = int(cand[self.rng.choice(probs.size, p=probs / total)])
        self.counts[token] = self.counts.get(token, 0) + 1
        return token

    def __call__(self, logits):
        """Return the id of the next token and add it to the history."""
        if self.greedy:
            # No copy to float64: the argmax of the float32 values is the same.
            token = ops.argmax(np.ascontiguousarray(logits, dtype=np.float32).reshape(-1))
            self.counts[token] = self.counts.get(token, 0) + 1
            return token
        if self.temperature <= 0.0:
            token = ops.argmax(self._scores(logits).astype(np.float64))
            self.counts[token] = self.counts.get(token, 0) + 1
            return token
        return self._sample(*self._dist(logits))

    def sample_sparse(self, ids, vals, stat):
        """Return the token of a row from its candidates (_dist_sparse), as
        __call__ on the whole row, or None when they do not settle it (the
        history does not change then)."""
        if self.greedy or self.temperature <= 0.0:
            return None
        d = self._dist_sparse(ids, vals, stat)
        return None if d is None else self._sample(*d)

    def in_set_active(self):
        """Return True when draft_ok can keep a draft: "in_set", a
        temperature above zero, and a setting that limits the tokens (top_k,
        top_p, min_p, or mtp_floor). With the temperature alone every token
        is allowed, so every draft stays; the rule then is "exact"."""
        return (self.mtp_accept == "in_set" and not self.greedy and self.temperature > 0.0
                and (self.top_k is not None or self.top_p is not None
                     or self.min_p is not None or self.mtp_floor is not None))

    def _allowed(self, cand, probs, token):
        hit = np.nonzero(cand == int(token))[0]
        if hit.size == 0 or probs[hit[0]] <= 0.0:
            return False
        return self.mtp_floor is None or probs[hit[0]] >= self.mtp_floor * float(probs.max())

    def draft_ok(self, logits, token):
        """Return True when the settings allow the MTP draft token at this row
        (in_set_active): the token has a probability above zero after top_k,
        top_p, and min_p, and at least mtp_floor times the best probability.
        The call uses no random numbers and does not change the history
        (accept does)."""
        if not self.in_set_active():
            return False
        return self._allowed(*self._dist(logits), token)

    def draft_ok_sparse(self, ids, vals, stat, token):
        """draft_ok from the candidates of a row (_dist_sparse), or None when
        they do not settle it."""
        if not self.in_set_active():
            return False
        d = self._dist_sparse(ids, vals, stat)
        return None if d is None else self._allowed(*d, token)

    def accept(self, token):
        """Add a token that draft_ok kept to the history (as __call__ does)."""
        token = int(token)
        self.counts[token] = self.counts.get(token, 0) + 1

    def fixed(self, token):
        """Return the token that takes the place of a picked token (already in
        the history) by guard, the history changed to it; the token itself
        when there is no guard or the guard keeps it. Call it once for each
        token of the answer, in order (the guard may keep a state)."""
        token = int(token)
        if self.guard is None:
            return token
        t = int(self.guard(token))
        if t != token:
            n = self.counts.get(token, 0)
            if n <= 1:
                self.counts.pop(token, None)
            else:
                self.counts[token] = n - 1
            self.counts[t] = self.counts.get(t, 0) + 1
        return t
