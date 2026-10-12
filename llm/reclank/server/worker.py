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

import os
import queue
import sys
import threading
import time
import traceback

# The exit status after a fatal error (the CUDA context is lost), so a
# supervisor (scripts/serve_forever.sh) can start the server again.
EXIT_FATAL = 70


class ModelWorker:
    """Run functions on one thread. load(fn) runs the first one and returns
    its value; call(fn) runs one and returns its value; stream(gen) runs a
    generator and yields its items."""

    def __init__(self):
        self.jobs = queue.Queue()
        self.engine = None          # for the fatal errors (engine.fatal)
        self._thread = threading.Thread(target=self._loop, daemon=True, name="model")
        self._thread.start()

    def _loop(self):
        while True:
            job = self.jobs.get()
            try:
                job()
            except BaseException as exc:
                print("[reclank] a job failed:", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                self._note(exc)
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

    def stream(self, make):
        """Yield the items of the generator make() that runs on the model
        thread. A caller that stops early stops it."""
        q, stop = queue.Queue(), threading.Event()

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
