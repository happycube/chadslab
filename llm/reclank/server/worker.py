"""The one model thread of the server: the load, the warm-up, and then
each job (a generation, an encoder) in turn.

The CPU part of a GPU step runs in the thread that runs the step, and each
thread has a team of OpenMP threads of its own. With a team for each
request as well as that of the load, libgomp has more threads than CPUs,
and its barriers sleep and wake (a step of 58 ms, not 46, on Qwen3.8). So
every family runs its model on this one thread; the request threads get
the tokens from a queue, and a caller that stops early stops the job.
"""
from __future__ import annotations

import collections
import os
import queue
import sys
import threading
import time
import traceback

from reclank.core.speculative import DeviceOp

# The exit status after a fatal error (the CUDA context is lost), so a
# supervisor (scripts/serve_forever.sh) can start the server again.
EXIT_FATAL = 70


# The lanes (--lanes 2): while one lane reads a prompt, the other decodes between its parts for this
# share of the time of a part (the prompt then takes 1 + share times as long)
PROMPT_SHARE = float(os.environ.get("RECLANK_LANE_PROMPT_SHARE", "0.5"))
# the tokens of a part of a prompt (engine.LANE_PART), and what a request's answer counts in prompt
# tokens when the route weighs the waits of the lanes (an agent's turn: a few hundred tokens)
LANE_PART = int(os.environ.get("RECLANK_LANE_PART", "4096"))
ANSWER_COST = int(os.environ.get("RECLANK_LANE_ANSWER_COST", "2048"))


class _Slot:
    """The generation of a lane: its generator, the queue of its consumer, its stop, the run it
    handed to the scheduler (DeviceOp) or None, and when a paused prompt goes on."""

    __slots__ = ("gen", "q", "stop", "op", "until", "left")

    def __init__(self, gen, q, stop, left=0):
        self.gen, self.q, self.stop = gen, q, stop
        self.op, self.until = None, 0.0
        self.left = left        # the prompt tokens it has still to read (an estimate)


class LaneScheduler:
    """Two generations at once on the model thread (docs/LANES_PLAN.md): the engines of the
    lanes (engines[0] the model's own state, engines[1] lane B: engine.on() puts its state in
    place). A generation is a generator of tokens that hands its runs over as DeviceOps
    (reclank/core/speculative.py): a step or a verify group ("run"), a pause between the parts
    of a prompt, or a "yield" (a decode that ran its own work, as Nemotron's MTP rounds: it goes on
    at the next tick, and counts as decoding for the other lane's pause). Each tick resumes the lanes that have no run pending, then runs the runs: the two
    at once when both lanes decode (engines[0].pair), else each alone. A lane's requests wait in
    its queue (backend.route chose the lane)."""

    def __init__(self, engines):
        self.engines = engines
        n = len(engines)
        self.slots = [None] * n
        self.waiting = [collections.deque() for _ in range(n)]
        self.reserved = [0] * n         # routed, not yet here (backend.route)
        self.reserved_tokens = [0] * n  # their prompt tokens to read
        self.last_used = [0.0] * n
        self.pairs = self.solos = 0

    def busy(self):
        return any(s is not None for s in self.slots) or any(self.waiting)

    def lane_busy(self, i):
        return self.slots[i] is not None or bool(self.waiting[i]) or self.reserved[i] > 0

    def wait(self, i):
        """What a new request on lane i waits for, in prompt tokens: the prompts its requests
        have still to read, and ANSWER_COST for each one's answer."""
        n = self.reserved[i] + len(self.waiting[i])
        tokens = self.reserved_tokens[i] + sum(w[3] for w in self.waiting[i])
        if self.slots[i] is not None:
            n += 1
            tokens += self.slots[i].left
        return tokens + n * ANSWER_COST

    def admit(self, lane, make, q, stop, tokens=0):
        self.reserved[lane] = max(0, self.reserved[lane] - 1)
        self.reserved_tokens[lane] = max(0, self.reserved_tokens[lane] - tokens)
        self.waiting[lane].append((make, q, stop, tokens))

    def _end(self, i, kind, value=None):
        s = self.slots[i]
        self.slots[i] = None
        self.last_used[i] = time.monotonic()
        s.q.put((kind, value))

    def _fail(self, i, exc, note):
        s = self.slots[i]
        try:
            with self.engines[i].on():
                s.gen.close()
        except BaseException:
            pass
        note(exc)
        self._end(i, "error", exc)

    def _resume(self, i, note):
        """Run lane i's generation until it hands over a run or ends."""
        s = self.slots[i]
        t0 = time.monotonic()
        try:
            with self.engines[i].on():
                while True:
                    if s.stop.is_set():
                        s.gen.close()
                        self._end(i, "end")
                        return
                    try:
                        x = s.gen.send(None)
                    except StopIteration:
                        self._end(i, "end")
                        return
                    if isinstance(x, DeviceOp):
                        s.op = x
                        break
                    s.q.put(("item", x))
        except BaseException as exc:
            self._fail(i, exc, note)
            return
        if s.op.kind == "pause":
            s.left = max(0, s.left - LANE_PART)
            other = any(self.slots[j] is not None for j in range(len(self.slots)) if j != i)
            s.until = time.monotonic() + (PROMPT_SHARE * (time.monotonic() - t0) if other else 0.0)
        else:
            s.left = 0

    def _run_alone(self, i, note):
        s = self.slots[i]
        op, s.op = s.op, None
        try:
            with self.engines[i].on() as dev:
                if len(op.tokens) == 1:
                    dev.step(op.tokens[0], op.pos)
                else:
                    dev.verify(op.tokens, op.pos)
            self.solos += 1
        except BaseException as exc:
            self._fail(i, exc, note)

    def tick(self, note):
        """One round of the lanes."""
        for i, w in enumerate(self.waiting):
            while self.slots[i] is None and w:
                make, q, stop, tokens = w.popleft()
                if stop.is_set():
                    q.put(("end", None))
                    continue
                try:
                    with self.engines[i].on():
                        gen = make()
                except BaseException as exc:
                    note(exc)
                    q.put(("error", exc))
                    continue
                self.slots[i] = _Slot(gen, q, stop, tokens)
        now = time.monotonic()
        for i, s in enumerate(self.slots):
            if s is None:
                continue
            if s.op is not None and s.op.kind == "pause":
                # the pause holds while another lane decodes: it has a run pending, or it has
                # none yet (it ran this tick or the last; resumed below, it hands one over). Not
                # while the other reads a prompt too (paused): then the parts take turns.
                decodes = any(t is not None and (t.op is None or t.op.kind in ("run", "yield"))
                              for j, t in enumerate(self.slots) if j != i)
                if now < s.until and decodes and not s.stop.is_set():
                    continue
                s.op = None
            if s.op is not None and s.op.kind == "yield":
                s.op = None
            if s.op is None:
                self._resume(i, note)
        runs = [i for i, s in enumerate(self.slots) if s is not None and s.op is not None and s.op.kind == "run"]
        if len(runs) == 2:
            a, b = self.slots[runs[0]], self.slots[runs[1]]
            try:
                paired = self.engines[0].pair(a.op, b.op)
            except BaseException as exc:
                for i in runs:
                    self._fail(i, exc, note)
                return
            if paired:
                a.op = b.op = None
                self.pairs += 1
                return
        for i in runs:
            self._run_alone(i, note)


class ModelWorker:
    """Run functions on one thread. load(fn) runs the first one and returns
    its value; call(fn) runs one and returns its value; stream(gen) runs a
    generator and yields its items; stream(gen, lane=i) a generation of lane i
    (lanes: a LaneScheduler)."""

    def __init__(self):
        self.jobs = queue.Queue()
        self.engine = None          # for the fatal errors (engine.fatal)
        self.lanes = None           # a LaneScheduler (--lanes 2)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="model")
        self._thread.start()

    def _loop(self):
        while True:
            lanes = self.lanes
            if lanes is not None and lanes.busy():
                # the other jobs (a route, an encoder) between the rounds of the lanes
                while True:
                    try:
                        job = self.jobs.get_nowait()
                    except queue.Empty:
                        break
                    self._job(job)
                try:
                    lanes.tick(self._note)
                except BaseException as exc:
                    print("[reclank] the lanes failed:", file=sys.stderr, flush=True)
                    traceback.print_exc(file=sys.stderr)
                    self._note(exc)
                self._fatal()
                continue
            self._job(self.jobs.get())

    def _job(self, job):
        try:
            job()
        except BaseException as exc:
            print("[reclank] a job failed:", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            self._note(exc)
        self._fatal()

    def _fatal(self):
        """After a job: a fatal error (engine.fatal) ends the process."""
        eng = self.engine
        if eng is not None and getattr(eng, "fatal", None) is not None:
            try:
                d = eng.crash_report(eng.fatal)
                if d:
                    print("[reclank] the crash report: %s (scripts/replay_crash.py %s)" % (d, d),
                          file=sys.stderr, flush=True)
            except Exception as exc:
                print("[reclank] the crash report failed: %r" % exc, file=sys.stderr, flush=True)
            print("[reclank] fatal error, the process ends (exit %d): %s" % (EXIT_FATAL, eng.fatal),
                  file=sys.stderr, flush=True)
            time.sleep(1.0)     # the caller sends the error to its client
            os._exit(EXIT_FATAL)

    def _note(self, exc):
        if self.engine is not None:
            self.engine.note_error(exc)

    def call(self, fn):
        """Run fn on the model thread; return its value or raise its error."""
        if threading.current_thread() is self._thread:
            return fn()
        q = queue.Queue()

        def job():
            try:
                q.put((True, fn()))
            except BaseException as exc:
                self._note(exc)
                q.put((False, exc))
        self.jobs.put(job)
        ok, value = q.get()
        if not ok:
            raise value
        return value

    load = call

    def stream(self, make, lane=None, tokens=0, stop=None):
        """Yield the items of the generator make() that runs on the model
        thread. A caller that stops early stops it. lane: a generation of that
        lane (the LaneScheduler runs it beside the other lane's); tokens: the
        prompt tokens it reads (backend.route's estimate). stop: an Event that
        stops it too, set by another thread: a caller that waits here for the
        first token of a long prompt cannot stop early (the scheduler checks it
        between the parts of the prompt; a request still waiting never starts)."""
        q, stop = queue.Queue(), (stop if stop is not None else threading.Event())
        if lane is not None and self.lanes is not None:
            self.jobs.put(lambda: self.lanes.admit(lane, make, q, stop, tokens))
            try:
                while True:
                    kind, value = q.get()
                    if kind == "end":
                        return
                    if kind == "error":
                        raise value
                    yield value
            finally:
                stop.set()

        def job():
            try:
                for item in make():
                    q.put(("item", item))
                    if stop.is_set():
                        break
                q.put(("end", None))
            except BaseException as exc:
                self._note(exc)
                q.put(("error", exc))
        self.jobs.put(job)
        try:
            while True:
                kind, value = q.get()
                if kind == "end":
                    return
                if kind == "error":
                    raise value
                yield value
        finally:
            stop.set()
