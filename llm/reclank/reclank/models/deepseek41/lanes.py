"""Lanes of DeepSeek V4.1 Flash on the GPU (docs/OVERLAP_PLAN.md 2.1; the design of
docs/LANES_PLAN.md for Qwen3.8): a second sequence whose runs go at the same time as the model's
own (lane A), so that the GPU runs one lane's layers while the CPU runs the other's cold experts.

The programs of DSV41GPU hold the addresses of their state (the caches, the compressor states)
from the time they are made, so a lane does not bind a cache as Qwen's do: it has the state of a
sequence of its own (DSV41GPU.lane_state: LANE_ATTRS, its engram hashes, DSpark's rings), and its
programs are the model's made under its view (DSLane.view swaps the state in; run and DSpark.run
key them by the lane, so freeing the programs frees both lanes'). Its steps and verify groups get
scratches of the C side of their own (they run beside the model's); its prompt runs alone, on
programs of its own that share the model's prompt scratch. The weights, the hot experts (DSHot)
and the CPU teams are shared: run_pair runs lane B on a thread and a stream of its own, its CPU
tasks on the model thread; the slots of the hot experts change only between pairs, from the
choices of both.

    lane = DSLane(g, ds)
    (la, ha), (lb, hb) = run_both(g, lane, [ya], pa, [yb], pb)      # a step of each at once
"""
from __future__ import annotations

import contextlib
import copy
import os
import sys

import numpy as np

from reclank.backends.cuda.gpumm import mem
from reclank.backends.cuda.runtime import LaneRunner, run_blocks_of, run_pair
from reclank.models.deepseek41.gpu import VERIFY_MAX


def lane_cpus(g):
    """A CPU for lane B's thread (it spins while its run goes): of the process's, not one of the
    parts' teams nor the model thread's; None (and a warning) when there is none."""
    from reclank.runtime import moe_parts
    used = set(int(c) for c in g.parts.cpus_arr) if getattr(g, "parts", None) is not None else set()
    free = sorted(moe_parts.allowed() - used - set(os.sched_getaffinity(0)))
    if not free:
        print("[lanes] no CPU outside the parts and the model thread for lane B's thread",
              file=sys.stderr, flush=True)
        return None
    return {free[-1]}


class DSLane:
    """Lane B of g (a DSV41GPU), with ds (its DSpark, or None)."""

    def __init__(self, g, ds=None, max_len=None, lane_id=1):
        self.g, self.ds = g, ds
        self.id = int(lane_id)
        self.state = g.lane_state(self.id, max_len)
        # the engram hashes of its sequence (the ids so far: Hashes.cache)
        self.hashes = copy.copy(g.m.hashes)
        self.hashes.cache = np.zeros_like(g.m.hashes.cache)
        self.rings = None
        if ds is not None:
            self.rings = [g.cache_rows(g.m.win, g.m.head_dim, "w") for _ in range(ds.blocks)]
            for r in self.rings:
                g.mirror.add(r)
        # its state known to the mirror now (reset and the contexts read their device copies before
        # any of its programs is made; the model's come with its first program)
        with self.view():
            for a in g.state_arrays() + [g.akeys]:
                g.mirror.add(a)
        self.runner = LaneRunner(lane_cpus(g))
        # the sizes of its runs that ran once alone on its stream (before any pair: their first
        # launch there with nothing beside it)
        self.warm = set()

    @contextlib.contextmanager
    def view(self):
        """g (and DSpark) with lane B's state in place of the model's: on the model thread, with no
        run going. g's run, prefill, reset, verify, restore and DSpark's prefill, draft, seed then
        serve lane B, on its own programs."""
        g, ds = self.g, self.ds
        saved = {a: getattr(g, a, None) for a in g.LANE_ATTRS}
        for a in g.LANE_ATTRS:
            setattr(g, a, self.state[a])
        hashes, g.m.hashes = g.m.hashes, self.hashes
        if ds is not None:
            rings, ds.rings = ds.rings, self.rings
        try:
            yield g
        finally:
            for a in g.LANE_ATTRS:
                self.state[a] = getattr(g, a, None)
                setattr(g, a, saved[a])
            self.hashes, g.m.hashes = g.m.hashes, hashes
            if ds is not None:
                self.rings, ds.rings = ds.rings, rings

    def close(self):
        self.runner.close()


def _runs(g, lane, ta, tb):
    """The programs of ta rows (the model) and tb rows (lane B), both alive: making one can free
    the other (make_room)."""
    for _ in range(4):
        ra = g.run(ta)
        with lane.view():
            rb = g.run(tb)
        if ra.g.handle and rb.g.handle:
            return ra, rb
    raise RuntimeError("lanes: the GPU has no room for the programs of both lanes")


def run_both(g, lane, ta, pa, tb, pb):
    """The tokens ta at pa on the model and tb at pb on lane B at once (each at most VERIFY_MAX: a
    step or a verify group; the same program and bits as g.verify of each alone). The hot cache
    lands its copies before, learns the choices of both after, and starts its next copies.
    Return ((logits [len(ta), vocab], main_h or None) of the model, the same of lane B)."""
    assert 1 <= len(ta) <= VERIFY_MAX and 1 <= len(tb) <= VERIFY_MAX, (len(ta), len(tb))
    g.fit(pa + len(ta))
    with lane.view():
        g.fit(pb + len(tb))
    hot = g.hot
    if hot is not None:
        # the steps left for later (forward) of either, then the copies landed: the slots stay
        # as they are while the two run
        g._hot_flush()
        with lane.view():
            g._hot_flush()
        hot.land()
    ra, rb = _runs(g, lane, len(ta), len(tb))
    if len(tb) not in lane.warm:
        # lane B's first run of this size alone on its stream, then the model's alone
        with lane.view():
            g._inputs(rb, tb, pb)
        lane.runner.solo(rb.g)
        lane.warm.add(len(tb))
        g._inputs(ra, ta, pa)
        ra.g.run()
    else:
        mem().poll()
        # each program's blocks locked right after its bind (as Qwen's lanes)
        g._inputs(ra, ta, pa)
        la = run_blocks_of(ra.g)
        mem().lock(la)
        with lane.view():
            g._inputs(rb, tb, pb)
        lb = run_blocks_of(rb.g)
        mem().lock(lb)
        run_pair(ra.g, rb.g, lane.runner, la + lb)
    out = []
    for r, view in ((ra, contextlib.nullcontext()), (rb, lane.view())):
        with view:
            r.g.download("logits")
            if r.main_h is not None:
                g.download(r.main_h)
            g.last = r
            out.append((r.logits.copy(), r.main_h.copy() if r.main_h is not None else None))
    if hot is not None:
        hot.observe(g.download(ra.ids), len(ta), ra.lay)
        hot.observe(g.download(rb.ids), len(tb), rb.lay)
        hot.plan(len(ta) + len(tb))
    return out[0], out[1]


def prefill(g, lane, ids, ds=None):
    """A prompt from an empty state on lane B (alone; ds: with DSpark's rings). Return the logits
    of its last token and (ds) its input of DSpark."""
    with lane.view():
        g.reset()
        if ds is not None:
            ds.reset()
            return ds.prefill(ids)
        return g.prefill(ids), None


def greedy_pair(g, lane, ids_a, ids_b, n, ds=None, stats=None):
    """Greedy decode of two prompts at once, n tokens each: plain steps (ds None) or DSpark rounds
    (the drafts of each lane in turn on the model thread, the two verify groups as a pair, the
    rows of the refused drafts undone on each). The model holds lane A from an empty state, lane B
    its own. The tokens of each; stats: the pairs and the drafts taken of each."""
    st = stats if stats is not None else {}
    st.update(pairs=0, drafted_a=0, drafted_b=0, taken_a=0, taken_b=0)
    g.reset()
    if ds is not None:
        ds.reset()
        lga, ma = ds.prefill(ids_a)
    else:
        lga, ma = g.prefill(ids_a), None
    lgb, mb = prefill(g, lane, ids_b, ds)
    lane_st = {
        "a": dict(y=int(np.argmax(lga)), pos=len(ids_a), main=ma, mpos=len(ids_a) - 1, out=[]),
        "b": dict(y=int(np.argmax(lgb)), pos=len(ids_b), main=mb, mpos=len(ids_b) - 1, out=[]),
    }
    views = {"a": contextlib.nullcontext, "b": lane.view}
    while min(len(s["out"]) for s in lane_st.values()) < n:
        rows = {}
        for nm, s in lane_st.items():
            drafts = []
            if ds is not None:
                with views[nm]():
                    d, conf = ds.draft(s["y"], s["main"], s["mpos"])
                drafts = [int(v) for v in d[:ds.choose(conf)]]
            rows[nm] = drafts
        (lg_a, mh_a), (lg_b, mh_b) = run_both(g, lane, [lane_st["a"]["y"]] + rows["a"], lane_st["a"]["pos"],
                                              [lane_st["b"]["y"]] + rows["b"], lane_st["b"]["pos"])
        st["pairs"] += 1
        for nm, lg, mh in (("a", lg_a, mh_a), ("b", lg_b, mh_b)):
            s, d = lane_st[nm], rows[nm]
            a = 0
            while a < len(d) and int(np.argmax(lg[a])) == d[a]:
                a += 1
            if a < len(d):
                with views[nm]():
                    g.restore(len(d) + 1, s["pos"], a + 1)
            st["drafted_" + nm] += len(d)
            st["taken_" + nm] += a
            s["out"] += [s["y"]] + d[:a]
            if mh is not None:
                s["main"], s["mpos"] = mh[:a + 1], s["pos"]
            s["y"] = int(np.argmax(lg[a]))
            s["pos"] += a + 1
    return lane_st["a"]["out"][:n], lane_st["b"]["out"][:n]


def greedy_alone(g, ids, n, ds=None):
    """The same greedy decode of one prompt on the model alone (the reference of greedy_pair):
    plain steps, or DSpark rounds with the same choice of lengths, each verify group alone."""
    g.reset()
    if ds is not None:
        ds.reset()
        lg, main = ds.prefill(ids)
    else:
        lg, main = g.prefill(ids), None
    y, pos, mpos, out = int(np.argmax(lg)), len(ids), len(ids) - 1, []
    while len(out) < n:
        d = []
        if ds is not None:
            dd, conf = ds.draft(y, main, mpos)
            d = [int(v) for v in dd[:ds.choose(conf)]]
        lg, mh = g.verify([y] + d, pos)
        a = 0
        while a < len(d) and int(np.argmax(lg[a])) == d[a]:
            a += 1
        if a < len(d):
            g.restore(len(d) + 1, pos, a + 1)
        out += [y] + d[:a]
        if mh is not None:
            main, mpos = mh[:a + 1], pos
        y = int(np.argmax(lg[a]))
        pos += a + 1
    return out[:n]
