"""The lanes of the server (--lanes 2) with no model: the session id of a request, the choice of a
lane, and the scheduler of server/worker.py on fake engines."""
import contextlib
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from reclank.core.speculative import DeviceOp  # noqa: E402
from server.backend import choose_lane, lane_of, session_of  # noqa: E402
from server.worker import LaneScheduler  # noqa: E402


def test_session_of():
    assert session_of({}, {"x-session-affinity": "s1"}) == "h:s1"           # pi-ai (dsh)
    assert session_of({}, {"session_id": "s2"}) == "h:s2"
    assert session_of({}, {"x-session-id": "s3"}) == "h:s3"                 # OpenRouter form
    assert session_of({"prompt_cache_key": "k"}, {}) == "b:k"               # OpenAI API
    assert session_of({"metadata": {"user_id": "user_ab_account_cd_session_ef"}}, {}) == "b:session_ef"
    assert session_of({"metadata": {"user_id": "user_ab"}}, {}) is None     # a user, not a session
    assert session_of({"user": "u"}, {"x-client-request-id": "r"}) is None
    assert lane_of({"id_slot": 1}, {}, 2) == 1                              # llama.cpp
    assert lane_of({}, {"x-reclank-lane": "0"}, 2) == 0
    assert lane_of({"lane": 5}, {}, 2) is None


def test_choose_lane():
    # the conversation's lane when free
    assert choose_lane([900, 0], [True, False], [False, False], [5, 1], None) == (0, "its context")
    # its lane busy (a title beside it): the free one
    assert choose_lane([900, 0], [True, False], [True, False], [5, 1], None)[0] == 1
    # a context in host memory counts on both lanes: the session's lane
    assert choose_lane([300, 300], [False, False], [False, False], [1, 5], 1) == (1, "its session")
    # a new conversation: the free lane whose last request ended first
    assert choose_lane([0, 0], [False, False], [False, False], [7, 3], None) == (1, "free")
    assert choose_lane([0, 0], [False, False], [True, False], [1, 3], None) == (1, "free")
    # all busy: the conversation's lane, else the session's, else the oldest end
    assert choose_lane([900, 0], [True, False], [True, True], [5, 1], 1)[0] == 0
    assert choose_lane([0, 0], [False, False], [True, True], [5, 1], 0)[0] == 0
    assert choose_lane([0, 0], [False, False], [True, True], [5, 1], None)[0] == 1
    assert choose_lane([900, 0], [True, False], [False, False], [5, 1], None, named=1) == (1, "named")
    # all busy, with the waits: the lane that frees first (a decode, not a long prompt)
    assert choose_lane([0, 0], [False, False], [True, True], [1, 5], None, wait=[8000, 2048]) == (1, "waits")
    assert choose_lane([0, 0], [False, False], [True, True], [1, 5], 0, wait=[8000, 2048])[0] == 1
    # near waits: the session's lane (its context's copy)
    assert choose_lane([0, 0], [False, False], [True, True], [5, 1], 0, wait=[2400, 2048]) == (0, "its session (waits)")
    assert choose_lane([900, 0], [True, False], [True, True], [5, 1], None, wait=[2048, 2048]) == (0, "its context (waits)")


class _Dev:
    def __init__(self, log, name):
        self.log, self.name = log, name

    def step(self, token, pos):
        self.log.append((self.name, "step", pos))

    def verify(self, tokens, pos):
        self.log.append((self.name, "verify", pos))


class _Engine:
    def __init__(self, log, name):
        self.log, self.dev = log, _Dev(log, name)

    @contextlib.contextmanager
    def on(self):
        yield self.dev

    def pair(self, a, b):
        self.log.append(("pair", len(a.tokens), len(b.tokens)))
        return len(a.tokens) > 1 and len(b.tokens) > 1


def _gen(prompt_parts, rounds, first, part_s=0.0):
    for _ in range(prompt_parts - 1):
        time.sleep(part_s)
        yield DeviceOp("pause")
    time.sleep(part_s)
    for r in range(rounds):
        yield first + r
        yield DeviceOp("run", [first + r] * 4, r)


def _run(sch, jobs):
    qs = []
    for lane, gen in jobs:
        q = queue.Queue()
        sch.admit(lane, lambda g=gen: g, q, threading.Event())
        qs.append(q)
    errs = []
    while sch.busy():
        sch.tick(errs.append)
    assert not errs, errs
    outs = []
    for q in qs:
        out = []
        while not q.empty():
            kind, v = q.get()
            if kind == "item":
                out.append(v)
        outs.append(out)
    return outs


def test_scheduler_pairs():
    log = []
    sch = LaneScheduler([_Engine(log, "A"), _Engine(log, "B")])
    outs = _run(sch, [(0, _gen(1, 3, 100)), (1, _gen(1, 3, 200))])
    assert outs == [[100, 101, 102], [200, 201, 202]]
    assert sch.pairs == 3 and sch.solos == 0


def test_scheduler_prompt_beside_decode():
    # lane B reads a prompt of 3 parts while lane A decodes: A runs alone between B's parts, then
    # the two pair
    log = []
    sch = LaneScheduler([_Engine(log, "A"), _Engine(log, "B")])
    outs = _run(sch, [(0, _gen(1, 6, 100)), (1, _gen(3, 2, 200))])
    assert outs == [[100, 101, 102, 103, 104, 105], [200, 201]]
    assert sch.solos >= 1 and sch.pairs >= 1
    assert sch.pairs * 2 + sch.solos == 8


def test_scheduler_prompt_first_lane():
    # the prompt on lane A (the first slot) and the decode on lane B: the decode still runs between
    # the parts for the share of a part's time (the pause held while B decodes), not one round a part
    log = []
    sch = LaneScheduler([_Engine(log, "A"), _Engine(log, "B")])
    outs = _run(sch, [(1, _gen(1, 400, 200)), (0, _gen(3, 1, 100, part_s=0.05))])
    assert outs == [list(range(200, 600)), [100]]
    # 2 pauses of ~25 ms (PROMPT_SHARE of 50 ms parts): B runs many rounds alone before A's first
    # run (the end of its prompt)
    first_a = next(k for k, e in enumerate(log) if e[0] in ("A", "pair"))
    assert first_a > 10, first_a


def test_scheduler_wait():
    sch = LaneScheduler([_Engine([], "A"), _Engine([], "B")])
    assert sch.wait(0) == 0
    sch.reserved[0] += 1
    sch.reserved_tokens[0] += 8000
    sch.admit(0, lambda: _gen(2, 1, 1), queue.Queue(), threading.Event(), 8000)
    assert sch.wait(0) == 8000 + 2048 and sch.reserved_tokens[0] == 0
