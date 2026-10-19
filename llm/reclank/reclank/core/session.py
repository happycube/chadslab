"""The contexts of the server between requests, for every family
(REFACTOR_PLAN.md, section 5.1): a new turn sends the whole conversation, and
the model reads only the tokens after the longest start that a context holds.
The cache keeps a few contexts (--contexts): one on the device, the others in
host memory; a request that goes on from one of them brings it back.

ContextCache is the part they share: the contexts with the most recent
first, at most max_contexts of them and at most max_bytes of host memory
(the least recently used goes first), and the count of the cached tokens.
Two kinds of context:

- SessionPool (the Gemma models): a context is a session (model.Session)
  with a cache of its own in host memory, which can go back to any start;
  the model puts the one in use on the GPU (Model._gpu_attach) and writes
  the new rows of the one before back to its host cache.
- RowCache (Qwen3.6, Qwen3.8, DeepSeek V4.1): one cache on the device, with
  a state that cannot go back (the DeltaNet, the window rings). A context is
  a list of points (the engine's copy of that state at a position) and the
  rows of its positions (the arrays with a row for each position, or each
  block of positions: keys and values, compressed rows, index keys). The
  context on the device holds its rows there; when another one comes back,
  the rows of the one on the device go to host memory (those it does not
  hold there yet), and every row of the one that comes back goes up. A
  start that both have is copied too: the same keys do not give the same
  bits when another prompt computed them (the kernels of a group of 16
  tokens or fewer and those of a larger group round differently), and a
  context that comes back must give the answer it gave on the device.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np


def common_len(a, b):
    """The count of the first items that a and b share (lists of keys)."""
    n = min(len(a), len(b))
    i, step = 0, 4096
    # whole chunks first (list equality runs in C), then the items
    while i + step <= n and a[i:i + step] == b[i:i + step]:
        i += step
    while i < n and a[i] == b[i]:
        i += 1
    return i


def _size(n):
    return "%.2f GB" % (n / 1e9) if n >= 1e8 else "%.1f MB" % (n / 1e6)


def state_nbytes(obj):
    """The bytes of the arrays in a state (arrays, and lists, tuples and
    dicts of them)."""
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        return sum(state_nbytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(state_nbytes(v) for v in obj)
    return 0


class ContextCache:
    """The contexts a model keeps (items, the most recent first): at most
    max_contexts, and at most max_bytes of host memory (0: no limit); the
    least recently used goes first, never the one in use (keep)."""

    def __init__(self, max_contexts=4, max_bytes=0):
        self.items = []
        self.max_contexts = max(1, int(max_contexts))
        self.max_bytes = max_bytes
        self.hits = [0, 0, 0]       # requests, prompt tokens, cached tokens

    def nbytes(self, item):
        return 0

    def drop(self, item):
        """An item leaves the cache."""

    def total(self):
        return sum(self.nbytes(it) for it in self.items)

    def touch(self, item):
        if item in self.items:
            self.items.remove(item)
        self.items.insert(0, item)
        self.trim(keep=(item,))

    def trim(self, keep=()):
        """Drop the least recently used items past the limits."""
        while len(self.items) > 1:
            over = len(self.items) > self.max_contexts or (
                self.max_bytes and self.total() > self.max_bytes)
            if not over:
                return
            old = next((it for it in reversed(self.items) if it not in keep), None)
            if old is None:
                return
            self.items.remove(old)
            self.drop(old)

    def note(self, n, k):
        """Count a request of n prompt tokens of which k were cached; return
        the line of all requests."""
        h = self.hits
        h[0], h[1], h[2] = h[0] + 1, h[1] + n, h[2] + k
        return "all %d requests: %d of %d cached (%.0f%%)" % (h[0], h[2], h[1], 100.0 * h[2] / max(h[1], 1))


class ContextDir:
    """The contexts of a SessionPool on disk (--context-dir), so that a long
    prompt (a document of 64K tokens: 6 minutes on a card of 4 GB) is read
    once and kept past the server. A file is a session (Session.save: its
    ids and the rows of its cache, npz) with a json beside it (its tag and
    count); the tag (the model file and the cache form) must match. The
    pool saves a session after a prompt of at least min_tokens new tokens,
    and when it drops one; a request whose start a file holds better than
    the sessions in memory (by gain tokens) loads it. At most max_bytes
    (the oldest files go first). A file has the name of the conversation id
    of the client when it gave one (server/backend.py session_of: hashed,
    s-<hash>), so a conversation keeps one file as it grows, else of its
    tokens (t-<hash>)."""

    def __init__(self, path, tag, max_bytes=64e9, min_tokens=4096, gain=1024):
        self.path = path
        self.tag = tag
        self.max_bytes = max_bytes
        self.min_tokens = min_tokens
        self.gain = gain
        self._ids = {}          # file -> (mtime, ids)
        os.makedirs(path, exist_ok=True)

    def _files(self):
        out = []
        for f in os.listdir(self.path):
            if f.endswith(".npz") and not f.endswith(".tmp.npz"):
                p = os.path.join(self.path, f)
                try:
                    meta = json.load(open(p[:-4] + ".json"))
                except (OSError, ValueError):
                    continue
                if meta.get("tag") == self.tag:
                    out.append((p, meta))
        return out

    def _file_ids(self, p):
        t = os.path.getmtime(p)
        got = self._ids.get(p)
        if got is None or got[0] != t:
            with np.load(p) as st:
                got = self._ids[p] = (t, [int(x) for x in st["ids"]])
        return got[1]

    def name_of(self, sid):
        """The file of the conversation id sid."""
        return os.path.join(self.path, "s-%s.npz" % hashlib.sha1(sid.encode()).hexdigest()[:20])

    def best(self, prompt_ids, sid=None):
        """The file with the longest start of prompt_ids, and the length
        (None, 0 for none); the file of the conversation sid first (it wins
        a tie)."""
        best, best_n = None, 0
        files = self._files()
        if sid is not None:
            own = self.name_of(sid)
            files.sort(key=lambda pm: pm[0] != own)
        for p, meta in files:
            try:
                n = common_len(self._file_ids(p), list(prompt_ids))
            except (OSError, ValueError, KeyError):
                continue
            if n > best_n:
                best, best_n = p, n
        return best, best_n

    def save(self, session):
        """Write a session (in place of its file before, which an older state
        of it holds); keep the directory under max_bytes. Its sid (the
        conversation id of the client, or None) names the file."""
        if getattr(session, "saved_n", None) == len(session.ids) or getattr(session, "media_keys", None):
            return
        sid = getattr(session, "sid", None)
        if sid is not None:
            p = self.name_of(sid)
        else:
            p = os.path.join(self.path, "t-%s.npz" % hashlib.sha1(
                np.asarray(session.ids, np.int32).tobytes()).hexdigest()[:20])
        t0 = time.time()
        nb = session.save(p, self.tag)
        with open(p[:-4] + ".json.tmp", "w") as f:
            json.dump({"tag": self.tag, "tokens": len(session.ids), "bytes": nb,
                       "session": sid is not None}, f)
        os.replace(p[:-4] + ".json.tmp", p[:-4] + ".json")
        old = getattr(session, "disk_path", None)
        if old and old != p and os.path.basename(old).startswith("t-"):
            self.remove(old)        # (a file of a conversation stays its own)
        session.disk_path, session.saved_n = p, len(session.ids)
        print("[reclank] context of %d tokens saved: %s (%s, %.1f s)" % (
            len(session.ids), os.path.basename(p), _size(nb), time.time() - t0), file=sys.stderr, flush=True)
        self.trim(keep=p)

    def remove(self, p):
        for f in (p, p[:-4] + ".json"):
            try:
                os.remove(f)
            except OSError:
                pass
        self._ids.pop(p, None)

    def trim(self, keep=None):
        files = sorted(self._files(), key=lambda pm: os.path.getmtime(pm[0]))
        total = sum(os.path.getsize(p) for p, _m in files)
        for p, _m in files:
            if total <= self.max_bytes:
                break
            if p != keep:
                total -= os.path.getsize(p)
                self.remove(p)


class SessionPool(ContextCache):
    """The sessions of a model whose cache can go back. make(max_len) gives a
    new session (model.Session). store (a ContextDir, or None) keeps them on
    disk too."""

    def __init__(self, make, max_sessions=4, min_shared=16, store=None):
        super().__init__(max_sessions)
        self.make = make
        # A shared start below this count is not worth a cache hit, and it
        # would evict the conversation that the cache should keep.
        self.min_shared = min_shared
        self.store = store

    def drop(self, item):
        """A session leaves the pool: to disk first when it is long enough."""
        if self.store is not None and len(getattr(item, "ids", ())) >= self.store.min_tokens:
            try:
                self.store.save(item)
            except (OSError, ValueError) as e:
                print("[reclank] context not saved: %s" % e, file=sys.stderr, flush=True)

    def done(self, session, sid=None):
        """After a request: save a session that read a long prompt (sid: the
        conversation id of the client, the name of its file)."""
        if sid is not None:
            session.sid = sid
        if self.store is not None and getattr(session, "prefilled", 0) >= self.store.min_tokens:
            try:
                self.store.save(session)
            except (OSError, ValueError) as e:
                print("[reclank] context not saved: %s" % e, file=sys.stderr, flush=True)

    @property
    def sessions(self):
        return self.items

    def pick(self, prompt_ids, max_tokens, sid=None):
        """Return (the session with the longest shared start, its length), or
        a new session and 0.

        A hit moves the session to the front, so a smaller cache keeps the
        conversation that is in use and drops the one that is not. The soft
        tokens of media match only the same media (Session.common). With a
        store, a file whose start is longer by store.gain comes back (sid:
        the conversation id of the client, whose own file wins a tie).
        """
        best = None
        best_n = 0
        spans = getattr(prompt_ids, "spans", None)
        for session in self.items:
            n = (session.common(prompt_ids, spans) if hasattr(session, "common")
                 else common_len(session.ids, prompt_ids))
            if n > best_n:
                best = session
                best_n = n
        if self.store is not None and spans is None:
            p, n = self.store.best(prompt_ids, sid)
            if p is not None and n >= max(self.min_shared, best_n + self.store.gain):
                s = self.make(len(prompt_ids) + max_tokens + 8)
                t0 = time.time()
                try:
                    s.load(p, self.store.tag)
                except (OSError, ValueError, KeyError) as e:
                    print("[reclank] context file not loaded: %s" % e, file=sys.stderr, flush=True)
                else:
                    s.disk_path, s.saved_n = p, len(s.ids)
                    print("[reclank] context of %d tokens loaded from %s (%.1f s)" % (
                        len(s.ids), os.path.basename(p), time.time() - t0), file=sys.stderr, flush=True)
                    self.touch(s)
                    return s, s.common(prompt_ids) if hasattr(s, "common") else n
        if best is not None and best_n >= self.min_shared:
            self.touch(best)
            return best, best_n
        best = self.make(len(prompt_ids) + max_tokens + 8)
        self.touch(best)
        return best, 0


class BufferRows:
    """The rows of a position-major device array (a Buffer of the GPU
    backend): per positions a row, rb bytes a row."""

    def __init__(self, buf, per, rb):
        self.buf, self.per, self.rb = buf, per, rb
        self.rows = buf.nbytes // rb

    def read(self, r0, r1):
        h = np.empty((r1 - r0) * self.rb, np.uint8)
        if r1 > r0:
            self.buf.download_at(h, r0 * self.rb)
        return h

    def write(self, r0, h):
        if h.nbytes:
            self.buf.upload_at(h, r0 * self.rb)


class HostRows:
    """The rows of a position-major host array (a model on the CPU)."""

    def __init__(self, a, per):
        self.a = a.reshape(a.shape[0], -1).view(np.uint8)
        self.per, self.rb, self.rows = per, self.a.shape[1], a.shape[0]

    def read(self, r0, r1):
        return self.a[r0:r1].reshape(-1).copy()

    def write(self, r0, h):
        n = h.nbytes // self.rb
        self.a[r0:r0 + n] = h.reshape(n, self.rb)


class Context:
    """A conversation that a RowCache keeps: keys (the keys of the positions
    of its longest point), points {position: the engine's state there}, and
    the rows held in host memory (host: [first row, end row, bytes] for each
    array)."""

    def __init__(self, keys=()):
        self.keys = list(keys)
        self.points = {}
        self.host = None

    def host_bytes(self):
        return sum(h[2].nbytes for h in self.host) if self.host else 0


class RowCache(ContextCache):
    """The contexts of a model with one cache on the device (see the module
    text). io gives the arrays of positions of that cache: io.rows() a list of
    BufferRows or HostRows (the same arrays in the same order every time),
    io.reserve(n) before the rows below n are written from the host (the
    device lends the rows past those in use: DeviceCache.share_tail).

    begin() finds where a prompt goes on from and brings that context to the
    device; the engine then restores the point it returns (or starts a fresh
    cache at 0), runs the rest of the prompt, and gives its new points
    (point())."""

    def __init__(self, io, max_contexts=4, max_bytes=16e9, name="cache", max_points=4):
        super().__init__(max_contexts, max_bytes)
        self.max_points = max_points    # the points of a context (the last ones stay)
        self.io = io
        self.name = name
        self.live = None        # the context on the device
        self.moved = []         # what the last begin() copied (a line of the log)
        # the RowCaches of the other lanes (share_with): their contexts in host memory are these
        # items too, and the one on each device is that lane's alone
        self.peers = []

    def share_with(self, other):
        """Share the contexts in host memory with other (the RowCache of another lane of the
        same model: the same arrays of positions, the same sizes): a context goes on from either
        device."""
        self.items = other.items
        self.hits = other.hits
        self.peers.append(other)
        other.peers.append(self)

    def _others(self):
        return [p.live for p in self.peers if p.live is not None]

    def trim(self, keep=()):
        super().trim(keep=tuple(keep) + tuple(self._others()))

    def nbytes(self, c):
        return c.host_bytes() + sum(state_nbytes(st) for st in c.points.values())

    def clear(self):
        others = self._others()
        self.items[:] = [it for it in self.items if any(it is o for o in others)]
        self.live = None

    def _best(self, keys, has_logits, as_is=None):
        """The longest start of keys that a context holds (begin): (k, its context, the state of
        its point or None). The context on another lane's device is not one."""
        n = len(keys)
        best_k, best_c, best_st = 0, None, None
        lv = self.live
        if lv is not None and as_is is not None and 0 < len(as_is) < n and common_len(as_is, keys) == len(as_is):
            best_k, best_c = len(as_is), lv
        others = self._others()
        for c in self.items:
            if any(c is o for o in others):
                continue
            m = common_len(c.keys, keys)
            for s, st in c.points.items():
                if best_k < s <= m and (s < n or has_logits(st)):
                    best_k, best_c, best_st = s, c, st
        return best_k, best_c, best_st

    def peek(self, keys, has_logits, as_is=None):
        """What begin(keys) would find, with no change: (k, True when it is the context on this
        device)."""
        k, c, _st = self._best(keys, has_logits, as_is)
        return k, c is not None and c is self.live

    def begin(self, keys, has_logits, as_is=None):
        """The longest start of keys that a context holds: a point of one of
        them (a point at the whole prompt only when has_logits(state): there
        must be a token to run otherwise), or as_is (the keys that the device
        cache has read, the answer of the last request too). Bring its
        context to the device; return (k, the state of the point, or None to
        go on from as_is, or k == 0: a fresh cache)."""
        best_k, best_c, best_st = self._best(keys, has_logits, as_is)
        lv = self.live
        t0, self.moved = time.time(), []
        if best_c is None:
            # a new context: every row of the device is written again
            if lv is not None:
                self._store(lv)
            self.live = Context()
            self.items.insert(0, self.live)
            self.trim(keep=(self.live,))
            self._log(t0)
            return 0, None
        if best_c is lv:
            # the context on the device goes on from best_k (an as_is start
            # writes past the points)
            if best_st is not None:
                self._cut(lv, best_k)
            self.touch(lv)
            self._log(t0)
            return best_k, best_st
        # a context in host memory: the rows of the device below k are its
        # own rows after the copy
        c, k = best_c, best_k
        if lv is not None:
            self._store(lv)
        self._cut(c, k)
        self._load(c, k)
        self.live = c
        self.touch(c)
        self._log(t0)
        return k, best_st

    def point(self, keys, state):
        """A point of the context on the device after keys (the engine's
        state there). One at the same place goes."""
        lv = self.live
        if lv is None:
            lv = self.live = Context()
            self.items.insert(0, lv)
        if len(keys) >= len(lv.keys):
            lv.keys = list(keys)
        lv.points[len(keys)] = state
        for s in sorted(lv.points)[:-self.max_points]:
            del lv.points[s]
        self.trim(keep=(lv,))

    # ---- the rows

    def _rows(self, a, p0, p1, end=False):
        """The rows of array a for the positions p0 .. p1 - 1 (end: the row
        of an open block at p1 too)."""
        r1 = -(-p1 // a.per) if end else p1 // a.per
        return min(p0 // a.per, a.rows), min(r1, a.rows)

    def _hold(self, c, p0, p1, arrs):
        """Copy to c.host the rows of the positions p0 .. p1 - 1 that it does
        not hold (they are on the device and they are c's)."""
        if c.host is None:
            c.host = [[0, 0, np.empty(0, np.uint8)] for _ in arrs]
        moved, tokens = 0, 0
        for a, h in zip(arrs, c.host):
            r0, r1 = self._rows(a, p0, p1, end=True)
            if r1 <= r0:
                continue
            h0, h1, data = h
            if h1 <= h0:
                h[0], h[1], h[2] = r0, r1, a.read(r0, r1)
                moved += h[2].nbytes
                tokens = max(tokens, (r1 - r0) * a.per)
                continue
            parts = []
            if r0 < h0:
                parts.append(a.read(r0, h0))
            parts.append(data)
            if r1 > h1:
                parts.append(a.read(h1, r1))
            if len(parts) > 1:
                n = sum(p.nbytes for p in parts) - data.nbytes
                moved += n
                tokens = max(tokens, n // a.rb * a.per)
                h[0], h[1], h[2] = min(r0, h0), max(r1, h1), np.concatenate(parts)
        if moved:
            self.moved.append("%d tokens to host (%s)" % (min(tokens, p1 - p0), _size(moved)))

    def _store(self, c):
        """The context on the device goes to host memory: the rows of its
        positions that it does not hold there yet are copied."""
        if not c.points:
            self.items[:] = [it for it in self.items if it is not c]
            return
        self._hold(c, 0, len(c.keys), self.io.rows())

    def _cut(self, c, k):
        """c goes on from k: its points past k go, and its rows in host
        memory past k are no longer its."""
        c.points = {s: st for s, st in c.points.items() if s <= k}
        c.keys = c.keys[:k]
        if c.host:
            for a, h in zip(self.io.rows(), c.host):
                r1 = k // a.per
                if h[1] > r1:
                    keep = max(0, r1 - h[0])
                    h[1], h[2] = h[0] + keep, h[2][:keep * a.rb].copy()

    def _load(self, c, k):
        """Copy the rows of c for the positions below k to the device."""
        self.io.reserve(k)
        arrs, moved = self.io.rows(), 0
        for a, h in zip(arrs, c.host or ()):
            # (the row of an open block of positions is not held: _cut; the
            # run that completes the block writes it)
            r1 = min(self._rows(a, 0, k, end=True)[1], h[1])
            assert h[0] == 0, h[:2]
            if r1 > 0:
                a.write(0, h[2][:r1 * a.rb])
                moved += r1 * a.rb
        if moved:
            self.moved.append("%d tokens to the device (%s)" % (k, _size(moved)))

    def _log(self, t0):
        if self.moved:
            print("[%s] contexts: %s in %.2f s; %d held, %s in host memory" % (
                self.name, ", ".join(self.moved), time.time() - t0, len(self.items), _size(self.total())),
                file=sys.stderr, flush=True)
