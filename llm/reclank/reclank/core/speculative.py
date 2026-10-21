"""Speculative decoding (MTP) for all the models: one loop, two interfaces.

A drafter proposes tokens from the last token and the hidden state of the
target at its row; the target checks [token] + drafts as one group; a draft
stays while it is the token that the sampler of the plain decode picks at
its row, or (Sampler mtp_accept "in_set", opt-in) a token that the settings
of the sampler allow (Sampler.draft_ok). The target keeps the rows of the
tokens it emits. With "exact" every emitted token is the token of the plain
decode, picked by the same sampler in the same order: greedy, or a sampler
with a seed, gives the text of the plain decode.

    Target   verify(tokens, pos, pick) -> RowPicker of the rows of the group
             commit(tokens, n, pos): keep the rows of the first n tokens
             hidden(n): the hidden rows of the first n tokens (the drafter's)
    Drafter  draft(token, h, pos, n, eos_ids) -> up to n tokens after token
             observe(tokens, hidden, pos): the rows the target kept (tokens
             at pos.., with the hidden rows of the positions before them)
             flush(): at the end of a stream

The adapters: GemmaTarget and GemmaDrafter (Model or E4B, with
assistant.Assistant or gpu.GPUDrafter: a small model that reads the cache of
the target), QwenTarget and QwenMTPDrafter (Qwen4GPU and its MTP layer,
which keeps a cache of its own: observe gives it the rows to run).

    for t in stream(target, drafter, nxt, h, pos, n_draft, pick, eos_ids, n):
        ...

A target with verify_co (QwenTarget with coop: the lanes of the server) hands
its runs to a scheduler: the stream yields a DeviceOp there, and the
scheduler runs it (alone, or as a pair with another lane's) and sends None
back (server/worker.py). Its consumer passes the DeviceOps up (yield from).
"""
from __future__ import annotations

import os
import time

import numpy as np

from reclank.core import ops


class DeviceOp:
    """A run that a generation hands to the scheduler of the lanes (server/worker.py): "run" (a
    step of one token or a verify group of more, tokens at pos, on the lane's cache), "pause"
    (a prompt between its parts: the other lane may run), or "yield" (a decode between runs of its
    own: the other lane may run; it goes on at the next tick)."""

    __slots__ = ("kind", "tokens", "pos")

    def __init__(self, kind, tokens=(), pos=0):
        self.kind, self.tokens, self.pos = kind, list(tokens), pos

    def __repr__(self):
        return "DeviceOp(%s, %d tokens at %d)" % (self.kind, len(self.tokens), self.pos)


def relay(gen, on_item=None):
    """yield from gen, and call on_item(item) for each item that is not a DeviceOp (a token).
    Return the value of gen."""
    r = None
    while True:
        try:
            x = gen.send(r)
        except StopIteration as stop:
            return stop.value
        if isinstance(x, DeviceOp):
            r = yield x
            continue
        if on_item is not None:
            on_item(x)
        r = yield x


def greedy_pick(logits):
    """Return the most probable token of one row of logits."""
    return ops.argmax(logits)


class RowPicker:
    """The tokens of the rows of x (the last rows of a target step) for pick
    (greedy_pick or a Sampler), the cheapest way:

    - a greedy pick takes the best tokens of the GPU (target.argmax_rows),
      and copies no logits;
    - a Sampler takes the candidates of the GPU (target.logits_topk,
      Sampler.sample_sparse): the k best logits of a row, not the whole row
      (a row of 262144 values took 1.6 ms to sample on the host);
    - else, or when the candidates do not settle a row, the whole logits.

    Every path gives the token of pick on the whole row, so the text does not
    change. RECLANK_SPARSE=0 turns the candidates off, for a test."""

    def __init__(self, target, x, pick):
        self.target, self.x, self.pick = target, x, pick
        self.logits = self.picks = self.sparse = None
        if pick is greedy_pick or getattr(pick, "greedy", False):
            if hasattr(target, "argmax_rows"):
                self.picks = target.argmax_rows(x)
        elif (hasattr(pick, "sparse_k") and hasattr(target, "logits_topk")
              and os.environ.get("RECLANK_SPARSE", "1") != "0"):
            k = pick.sparse_k()
            if k:
                self.sparse = target.logits_topk(x, k, pick.temperature)

    def full(self):
        """Return the logits of all the rows (one copy)."""
        if self.logits is None:
            self.logits = self.target.logits(self.x)
        return self.logits

    def token(self, j):
        """Return the token of pick at row j (the history of a Sampler takes
        it), after its guard (Sampler.fixed)."""
        return self._fixed(self._token(j))

    def _fixed(self, t):
        return self.pick.fixed(t) if hasattr(self.pick, "fixed") else t

    def _token(self, j):
        if self.picks is not None:
            t = int(self.picks[j])
            if hasattr(self.pick, "accept"):
                self.pick.accept(t)
            return t
        if self.sparse is not None:
            ids, vals, stat = self.sparse
            t = self.pick.sample_sparse(ids[j], vals[j], stat[j])
            if t is not None:
                return t
        return self.pick(self.full()[j])

    def in_set(self):
        """Return True when the pick keeps drafts that its settings allow
        (Sampler mtp_accept "in_set")."""
        return (self.picks is None and getattr(self.pick, "in_set_active", None) is not None
                and self.pick.in_set_active())

    def draft_ok(self, j, token):
        """Sampler.draft_ok of the draft token at row j."""
        if self.sparse is not None:
            ids, vals, stat = self.sparse
            ok = self.pick.draft_ok_sparse(ids[j], vals[j], stat[j], token)
            if ok is not None:
                return ok
        return self.pick.draft_ok(self.full()[j], token)


def stream(target, drafter, nxt, h, pos, n_draft, pick, eos_ids=(), max_new_tokens=1 << 30,
           stats=None, room=None):
    """Yield the new tokens of a speculative decode, one at a time.

    The token nxt is the first new token; the target holds the rows before
    pos, not nxt. h is the hidden row of the target that predicted nxt (for
    a drafter that takes it). pick(logits) selects a token from one row of
    target logits: the sampler of the plain decode. room(pos), if given,
    limits the drafts of a round (the cache). The optional dict stats gets
    the counts of steps, drafts, accepted drafts (in_set: those that only
    the settings allowed), and the seconds of the drafts and the verifies.

    A round ends at an EOS token or at max_new_tokens, also inside the kept
    drafts: the target then keeps the rows of the tokens up to there."""
    st = stats if stats is not None else {}
    for key in ("steps", "drafts", "accepted", "in_set"):
        st.setdefault(key, 0)
    for key in ("draft_s", "verify_s"):
        st.setdefault(key, 0.0)
    emitted = 0
    try:
        while True:
            yield nxt
            emitted += 1
            if nxt in eos_ids or emitted >= max_new_tokens:
                return
            k = min(n_draft, max_new_tokens - emitted - 1)
            if room is not None:
                k = min(k, room(pos))
            t0 = time.perf_counter()
            d = drafter.draft(nxt, h, pos, k, eos_ids) if k > 0 else []
            t1 = time.perf_counter()
            batch = [nxt] + d
            if hasattr(target, "verify_co"):
                rp = yield from target.verify_co(batch, pos, pick)
            else:
                rp = target.verify(batch, pos, pick)
            in_set = rp.in_set()
            # Row j gives the token after batch[j]: a kept draft, or the pick
            # there (the last token of the round).
            out, stop, kept = [], False, 0
            for j in range(len(d) + 1):
                if in_set and j < len(d) and rp.draft_ok(j, d[j]):
                    pick.accept(d[j])
                    t = rp._fixed(d[j])     # (a guard may put another token there)
                    kept += t == d[j]
                else:
                    t = rp.token(j)
                out.append(t)
                if t in eos_ids or emitted + len(out) >= max_new_tokens:
                    stop = True
                    break
                if j == len(d) or t != d[j]:
                    break
            st["steps"] += 1
            st["drafts"] += len(d)
            st["accepted"] += sum(1 for a, b in zip(out, d) if a == b)
            st["in_set"] += kept
            # the target keeps the rows of nxt and of the kept drafts
            n = len(out)
            target.commit(batch, n, pos)
            hid = target.hidden(n)
            st["draft_s"] += t1 - t0
            st["verify_s"] += time.perf_counter() - t1
            drafter.observe(out, hid, pos + 1)
            pos += n
            for t in out[:-1]:
                yield t
                emitted += 1
            if stop:
                yield out[-1]
                return
            h = hid[-1:]
            nxt = out[-1]
    finally:
        drafter.flush()


# ---- Gemma: Model or E4B, with assistant.Assistant or gpu.GPUDrafter ----

class GemmaTarget:
    """A Gemma model (model.Model, e4b.E4B) and its cache as a Target: ids
    gets the tokens of the kept rows."""

    def __init__(self, model, cache, ids):
        self.model, self.cache, self.ids = model, cache, ids
        self.x = None

    def verify(self, tokens, pos, pick):
        self.x = self.model.forward(tokens, cache=self.cache, start_pos=pos)
        return RowPicker(self.model, self.x, pick)

    def commit(self, tokens, n, pos):
        self.ids.extend(tokens[:n])
        self.cache.truncate(pos + n)

    def hidden(self, n):
        return self.x[:n]


class GemmaDrafter:
    """assistant.Assistant or gpu.GPUDrafter as a Drafter: it reads the cache
    of the target and takes the hidden row of the last token."""

    def __init__(self, drafter, model, cache):
        self.drafter, self.model, self.cache = drafter, model, cache

    def draft(self, token, h, pos, n, eos_ids):
        return self.drafter.draft(self.model, token, h, pos, self.cache, n, eos_ids)

    def observe(self, tokens, hidden, pos):
        pass

    def flush(self):
        pass


# ---- Qwen3.8: Qwen4GPU and its MTP layer ----

class QwenTarget:
    """A Qwen4GPU with its cache attached as a Target: a group of one token
    is a step, of more a verify group (commit keeps n of its rows). The
    hidden rows are the streams after the last layer (the input of the MTP
    layer). on_commit(tokens, streams), if given, is told the kept tokens
    and their streams (serve_qwen4: its ids and the stream of the end)."""

    def __init__(self, dev, on_commit=None, coop=False):
        self.dev, self.on_commit = dev, on_commit
        self.coop = coop        # the runs go to the scheduler of the lanes (DeviceOp)
        self.n = 0
        self.H = None

    def verify(self, tokens, pos, pick):
        if len(tokens) > 1:
            self.dev.verify(tokens, pos)
        else:
            self.dev.step(tokens[0], pos)
        self.n = len(tokens)
        self.H = self.dev.streams(self.n)
        return RowPicker(self, self.n, pick)

    def verify_co(self, tokens, pos, pick):
        """verify as a generator: with coop the scheduler runs the step or the verify group (a
        DeviceOp), else this does."""
        if not self.coop:
            return self.verify(tokens, pos, pick)
        yield DeviceOp("run", tokens, pos)
        self.n = len(tokens)
        self.H = self.dev.streams(self.n)
        return RowPicker(self, self.n, pick)

    # the row interface of RowPicker: x is the count of the last rows
    def argmax_rows(self, x):
        return self.dev.argmax(rows=x)

    def logits(self, x):
        return np.asarray(self.dev.logits(rows=x)).reshape(x, -1)

    def commit(self, tokens, n, pos):
        if self.n > 1:
            self.dev.commit(n)
        if self.on_commit is not None:
            self.on_commit(list(tokens[:n]), self.H[:n])

    def hidden(self, n):
        return self.H[:n]


# RECLANK_MTP_STOP (0: off): the drafts of a round of the Qwen MTP layer go
# on while the product of their probabilities (the softmax of the head at
# the best token) is at least this; the first draft under it is the last
# (made already, the verify takes it too). The count of drafts is then at
# most --draft. A model can have its own default (mtp_stop: the Gemma 4 26B
# server 0.9); the variable, when set, wins (0 turns it off).
_MTP_STOP_ENV = os.environ.get("RECLANK_MTP_STOP", "")
MTP_STOP = float(_MTP_STOP_ENV or 0)


def mtp_stop(default=0.0):
    """RECLANK_MTP_STOP when it is set, else default."""
    return float(_MTP_STOP_ENV) if _MTP_STOP_ENV.strip() else float(default)


class QwenMTPDrafter:
    """The MTP layer of a Qwen4GPU as a Drafter: its rows are the tokens at
    each position with the streams of the model at the position before
    (observe queues them; draft runs them, in groups of at most 256, then
    chains the drafts on the streams of the layer). The drafts are the best
    tokens of the head on its rows; with stop (RECLANK_MTP_STOP) they end
    at the first where the product of their probabilities falls under it.
    mtp_cpu runs the layer on the CPU (Qwen4CPU.mtp_step, a cache of its
    own). flush runs the queued rows but the last (serve_qwen4: the cache of
    the MTP layer then holds the rows of all the tokens the model holds),
    when flush_rows is set."""

    def __init__(self, dev, mtp_cpu=False, flush_rows=False, stop=None):
        self.dev, self.mtp_cpu, self.flush_rows = dev, mtp_cpu, flush_rows
        self.stop = MTP_STOP if stop is None else stop
        self.pend = None            # (tokens, streams, position of the first)
        self.mcache = None
        if mtp_cpu:
            from reclank.models.qwen.qwen38 import Qwen4MTPCache
            self.mcache = Qwen4MTPCache(dev.cfg, dev.cache.max_len)

    def _rows(self, H, ids, pos):
        """Run the MTP layer on rows; return the draft of the last, its
        probability (1.0 with no stop), and the streams of the layer (of the
        last group)."""
        d = hm = None
        p = 1.0
        if not self.mtp_cpu and hasattr(self.dev, "mtp_rows"):
            # (the rows of a prompt in mixed groups: Qwen4GPU.mtp_rows)
            hm = self.dev.mtp_rows(H, ids, pos)
            if self.stop > 0:
                tk, pr = self.dev.argmax(prob=True)
                d, p = int(tk[0]), float(pr[0])
            else:
                d = int(self.dev.argmax()[0])
            return d, p, hm
        for c0 in range(0, len(ids), 256):
            c1 = min(len(ids), c0 + 256)
            if self.mtp_cpu:
                xm, hm = self.dev.model.mtp_step(H[c0:c1], ids[c0:c1], self.mcache, pos + c0)
                lg = np.asarray(self.dev.logits(x=xm[-1:]), np.float64).reshape(-1)
                d = int(np.argmax(lg))
                if self.stop > 0:
                    p = float(1.0 / np.exp(lg - lg[d]).sum())
            else:
                hm = self.dev.mtp(H[c0:c1], ids[c0:c1], pos + c0)
                if self.stop > 0:
                    tk, pr = self.dev.argmax(prob=True)
                    d, p = int(tk[0]), float(pr[0])
                else:
                    d = int(self.dev.argmax()[0])
        return d, p, hm

    def observe(self, tokens, hidden, pos):
        tokens, hidden = list(tokens), np.asarray(hidden).reshape(len(tokens), -1)
        if self.pend is None:
            self.pend = (tokens, hidden, pos)
        else:
            t0, h0, p0 = self.pend
            assert p0 + len(t0) == pos, "the rows of the MTP layer must follow"
            self.pend = (t0 + tokens, np.concatenate([h0, hidden]), p0)

    def draft(self, token, h, pos, n, eos_ids):
        if n <= 0:
            return []
        ids, H, p0 = self.pend
        assert p0 + len(ids) - 1 == pos and ids[-1] == token, "the queued rows end at the token"
        d, p, hm = self._rows(H, ids, p0)
        self.pend = None
        drafts = [d]
        while p >= self.stop and len(drafts) < n:
            d, q, hm = self._rows(hm[-1:], drafts[-1:], pos + len(drafts))
            drafts.append(d)
            p *= q
        return drafts

    def flush(self):
        if self.flush_rows and self.pend is not None and len(self.pend[0]) > 1:
            ids, H, p0 = self.pend
            self._rows(H[:-1], ids[:-1], p0)
            self.pend = (ids[-1:], H[-1:], p0 + len(ids) - 1)
