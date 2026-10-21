"""Lanes of Nemotron Super on the GPU (the design of docs/LANES_PLAN.md, as DeepSeek V4.1's lanes.py): a
second sequence whose steps go at the same time as the model's own (lane A), so that the GPU runs one
lane's dense layers while the CPU runs the other's cold experts.

A step of NemotronGPU is GPU-only between its MoE layers (~11 ms) and CPU-bound inside them (~13 ms):
alone, each side waits for the other about half the time. Lane B has the state of a sequence of its own
(NemotronGPU.LANE_ATTRS: its cache and Mamba states, its programs, their pinned rows and scratches; its
steps' scratches of the C side its own), swapped in by NmLane.view; the weights, the pool of hot experts
and the CPU team are shared. run_both runs lane B's step on a thread and a stream of its own
(LaneRunner, run_pair), its CPU tasks on the model thread in turn with lane A's; the pool's tables change
only between pairs (NemotronGPU.after_run of each after both are done). A lane's prompt runs alone.

    lane = NmLane(g, ctx=32768)
    la, lb = run_both(g, lane, ya, pa, yb, pb)          # a step of each at once: their logits
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

from reclank.backends.cuda.gpumm import mem
from reclank.backends.cuda.runtime import LaneRunner, run_blocks_of, run_pair

# the positions of lane B's cache (RECLANK_NM_LANE_CTX; its keys and values: ~32 KB a position on Super)
LANE_CTX = int(os.environ.get("RECLANK_NM_LANE_CTX", "32768"))


def lane_cpu():
    """The CPU of lane B's thread (it spins while its run goes): RECLANK_NM_LANE_CPU, else the last CPU of
    the process; load() keeps it out of the CPU team (RECLANK_RESERVED_CPUS) before the team is planned.
    None: no pin (and a warning)."""
    v = os.environ.get("RECLANK_NM_LANE_CPU")
    if v is not None:
        return int(v) if int(v) >= 0 else None
    try:
        from reclank.runtime.moe_parts import allowed
        cpus = sorted(allowed())
    except Exception:
        cpus = sorted(os.sched_getaffinity(0))
    return cpus[-1] if cpus else None


def reserve_lane_cpu():
    """Lane B's CPU out of the CPU team (before its first run plans it): the CPU, or None."""
    c = lane_cpu()
    if c is None:
        print("[lanes] no CPU for lane B's thread: it runs unpinned", file=sys.stderr, flush=True)
        return None
    have = [x for x in os.environ.get("RECLANK_RESERVED_CPUS", "").split(",") if x.strip()]
    if str(c) not in have:
        os.environ["RECLANK_RESERVED_CPUS"] = ",".join(have + [str(c)])
    return c


class NmLane:
    """Lane B of g (a NemotronGPU): a sequence of ctx positions of its own. mtp: its NemotronMTPSpec is
    made in its view by the caller (spec rounds of lane B run in its view)."""

    def __init__(self, g, ctx=None, lane_id=1):
        from reclank.models.nemotron.gpu import _NmDevCache
        from reclank.models.nemotron.kq import NemotronKQCache
        self.g = g
        self.id = int(lane_id)
        ctx = int(ctx or LANE_CTX)
        saved = {a: getattr(g, a) for a in g.LANE_ATTRS}
        try:
            g.cache = NemotronKQCache(g.cfg, ctx, kv="bf16")
            g.cache_dev = _NmDevCache()
            g.progs, g._bufs, g._ascr, g.media, g.lane_id = {}, {}, None, [], self.id
            g._two_sets()
            g.reset()
            self.state = {a: getattr(g, a) for a in g.LANE_ATTRS}
        finally:
            for a, v in saved.items():
                setattr(g, a, v)
        c = lane_cpu()
        self.runner = LaneRunner({c} if c is not None else None)
        # the sizes of its runs that ran once alone on its stream (their graphs recorded with nothing
        # beside them)
        self.warm = set()

    @contextlib.contextmanager
    def view(self):
        """g with lane B's state in place of the model's (on the model thread, no run going): g's runs,
        prefill, reset, mtp and a NemotronMTPSpec made in it then serve lane B."""
        g = self.g
        saved = {a: getattr(g, a) for a in g.LANE_ATTRS}
        for a in g.LANE_ATTRS:
            setattr(g, a, self.state[a])
        try:
            yield g
        finally:
            for a in g.LANE_ATTRS:
                self.state[a] = getattr(g, a)
                setattr(g, a, saved[a])

    def close(self):
        self.runner.close()


def run_both(g, lane, ya, pa, yb, pb):
    """A step of the token ya at pa on the model and of yb at pb on lane B at once (the same programs and
    bits as each alone). Return the logits of each."""
    ra = g._prepare([ya], pa)
    blocks_a = run_blocks_of(ra[1])
    mem().lock(blocks_a)
    with lane.view():
        rb = g._prepare([yb], pb)
    blocks_b = run_blocks_of(rb[1])
    mem().lock(blocks_b)
    if 1 not in lane.warm:
        # lane B's first step alone on its stream (its graphs), then the model's alone (each run locks
        # its own blocks)
        mem().unlock(blocks_a + blocks_b)
        lane.runner.solo(rb[1])
        lane.warm.add(1)
        ra[1].run()
    else:
        run_pair(ra[1], rb[1], lane.runner, blocks_a + blocks_b)
    la = g._finish(ra, pa)
    with lane.view():
        lb = g._finish(rb, pb)
    return la, lb


def prefill(g, lane, ids):
    """A prompt from an empty state on lane B (alone). The logits of its last token."""
    with lane.view():
        g.reset()
        return g.prefill(ids, 0)


def greedy_pair(g, lane, ids_a, ids_b, n):
    """Greedy plain steps of two prompts at once, n tokens each (lane A the model's state from empty,
    lane B its own): the tokens of each, and the logits of their last steps."""
    g.reset()
    lga = g.prefill(ids_a, 0)
    lgb = prefill(g, lane, ids_b)
    ya, yb = int(np.argmax(lga)), int(np.argmax(lgb))
    pa, pb = len(ids_a), len(ids_b)
    out_a, out_b = [ya], [yb]
    while len(out_a) < n:
        lga, lgb = run_both(g, lane, ya, pa, yb, pb)
        ya, yb = int(np.argmax(lga)), int(np.argmax(lgb))
        pa, pb = pa + 1, pb + 1
        out_a.append(ya)
        out_b.append(yb)
    return (out_a, lga), (out_b, lgb)


def greedy_alone(g, ids, n, lane=None):
    """The same greedy plain steps of one prompt alone (on lane B's state when lane is given): its
    tokens and the logits of its last step."""
    ctx = lane.view() if lane is not None else contextlib.nullcontext()
    with ctx:
        g.reset()
        lg = g.prefill(ids, 0)
        y, out = int(np.argmax(lg)), []
        out.append(y)
        while len(out) < n:
            lg = g.step(y)
            y = int(np.argmax(lg))
            out.append(y)
    return out, lg
