"""The reuse of a cache between the turns of a chat (REFACTOR_PLAN.md,
section 5.1): a new turn sends the whole conversation, and the model reads
only the tokens after the longest start that a cache holds.

Two kinds of cache:

- A cache that can go back (the Gemma models: KVCache, the key and value
  rows): SessionPool keeps a few sessions (model.Session) and picks the one
  with the longest shared start; the session cuts its cache there.
- A cache with a recurrent state that cannot go back (Qwen3.6 and Qwen3.8:
  the DeltaNet): SnapshotList keeps copies of the state at the starts of the
  answers, and best() finds the longest start that the cache as it is, or a
  snapshot, holds.
"""
from __future__ import annotations

from reclank.core.messages import common_len


class SessionPool:
    """The sessions of a model whose cache can go back. make(max_len) gives a
    new session (model.Session)."""

    def __init__(self, make, max_sessions=4, min_shared=16):
        self.make = make
        self.sessions = []
        self.max_sessions = max_sessions
        # A shared start below this count is not worth a cache hit, and it
        # would evict the conversation that the cache should keep.
        self.min_shared = min_shared

    def pick(self, prompt_ids, max_tokens):
        """Return (the session with the longest shared start, its length), or
        a new session and 0.

        A hit moves the session to the front, so a smaller cache keeps the
        conversation that is in use and drops the one that is not. The soft
        tokens of media match only the same media (Session.common).
        """
        best = None
        best_n = 0
        spans = getattr(prompt_ids, "spans", None)
        for session in self.sessions:
            n = (session.common(prompt_ids, spans) if hasattr(session, "common")
                 else common_len(session.ids, prompt_ids))
            if n > best_n:
                best = session
                best_n = n
        if best is not None and best_n >= self.min_shared:
            self.sessions.remove(best)
            self.sessions.insert(0, best)
            return best, best_n
        best = self.make(len(prompt_ids) + max_tokens + 8)
        self.sessions.insert(0, best)
        del self.sessions[self.max_sessions:]
        return best, 0


class SnapshotList:
    """Snapshots of a recurrent state: (keys, state, extra...) tuples, the
    oldest first. keys are the token keys (media.keys) that the state has
    read; state is the copy of the arrays."""

    def __init__(self, max_snaps=4):
        self.snaps = []
        self.max_snaps = max_snaps

    def clear(self):
        self.snaps = []

    def add(self, keys, *rest):
        """A snapshot after keys; one at the same place goes."""
        keys = list(keys)
        self.snaps = [sn for sn in self.snaps if sn[0] != keys]
        self.snaps.append((keys,) + rest)
        del self.snaps[:-self.max_snaps]

    def best(self, keys, cache_ids, has_logits):
        """The longest start of keys (n of them) that the cache holds as it
        is (cache_ids: the cache read exactly these keys) or that a snapshot
        holds. A snapshot with no logits (has_logits(sn) false) must leave a
        token to run. Return (k, snapshot or None); k 0: nothing."""
        n = len(keys)
        k, snap = 0, None
        c = len(cache_ids)
        if 0 < c < n and keys[:c] == cache_ids:
            k = c
        for sn in self.snaps:
            s = len(sn[0])
            if s > k and (s < n or (s == n and has_logits(sn))) and keys[:s] == sn[0]:
                k, snap = s, sn
        return k, snap
