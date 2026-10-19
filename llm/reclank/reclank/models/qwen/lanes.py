"""Lane B of a Qwen4GPU (docs/LANES_PLAN.md): a second decode whose steps run
at the same time as the model's own (lane A), so that the GPU runs one
lane's layers while the CPU runs the other's cold experts.

Lane B has a step program of its own (its buffers, its packed temporaries in
arenas of its own, its pinned cold lists), its own KV cache on the GPU (owner
cache1), and its own QSA scratch; it shares the weights, the hot experts, the
pool, the prompt programs, and the head. Its prompt runs alone, with the
model's prompt programs on its cache (Lane.view). A pair (step_pair) runs
both steps at once (runtime.run_pair: lane B on a thread and a stream of its
own, its CPU tasks on the model thread); the slots of the hot experts change
only between pairs, from the selections of both lanes in turn.

    lane = Lane(dev, ctx=32768)
    lane.attach(Qwen4Cache(dev.cfg, n))
    ya, yb = dev_first_token, lane.prefill(ids_b)
    step_pair(dev, lane, ya, pos_a, yb, pos_b); ya, yb = dev.argmax()[0], lane.argmax()[0]
"""
from __future__ import annotations

import contextlib
import os
import sys

import numpy as np

from reclank.ir import alloc
from reclank.ir import program as P
from reclank.backends.cuda.gpumm import mem
from reclank.backends.cuda.runtime import LaneRunner, _check, lib, run_blocks_of, run_pair, run_pair_job

# RECLANK_LANES_OVERLAP (1): in MTP lanes, lane B's drafts run in the pair, on its thread, before its
# verify group (beside lane A's verify group); 0: before the pair, on the model thread
OVERLAP = os.environ.get("RECLANK_LANES_OVERLAP", "1") != "0"
# the drafts of a round of lane B in the pair: groups of the MTP layer of at most this many rows (its
# own programs); more rows (a prompt's) run before the pair
LANE_MTP_ROWS = 4


def lane_cpus(dev):
    """A CPU for lane B's thread (it spins while its step runs): of the
    process's, not one of the parts' teams nor the model thread's; None (and a
    warning) when there is none."""
    from reclank.runtime import moe_parts
    used = set(int(c) for c in dev.parts.cpus_arr) if getattr(dev, "parts", None) is not None else set()
    free = sorted(moe_parts.allowed() - used - set(os.sched_getaffinity(0)))
    if not free:
        print("[lanes] no CPU outside the parts and the model thread for lane B's thread",
              file=sys.stderr, flush=True)
        return None
    return {free[-1]}


class Lane:
    """Lane B of dev (a Qwen4GPU)."""

    def __init__(self, dev):
        from reclank.models.qwen.gpu38 import _DevCache4
        self.dev = dev
        self.arenas = alloc.Arenas(dev.arenas.small)
        # the routers' choices of its verify groups (sel_of), for HotCache
        self.sel_bufs = {}
        self.ips = {}
        self.prog, self.g = self._program(1)
        # its verify groups (MTP lanes): t -> (prog, g), and those run once
        self.vprogs = {}
        self.vwarm = set()
        self.pending = None
        self.last_prog = None
        self.rows = 1
        self.mcache = None
        # its MTP layer (drafts in the pair): t -> (prog, g), its head programs, the cold list of its
        # MTP step, the choices of its runs for HotCache (scored after the pair), the input of the head
        self.mprogs = {}
        self.hprogs = {}
        self.mip = None
        self.msel = []
        self.mlast = None
        self.mtp_last = None
        self.cache_dev = _DevCache4()
        self.cache_dev.owner = "cache1"
        self.cache_dev.before = dev._share_before
        self.cache = None
        self.qsa_buf = None
        self.qpos_version = -1
        self.last = None
        self.runner = LaneRunner(lane_cpus(dev))
        self.warm = False

    def _program(self, t, verify=False):
        """A program of lane B (its step, or its verify group of t rows): a
        compile of its own (its arenas, its sel buffers) and C scratches of
        its own."""
        dev = self.dev
        ips = {i: st.get("ip") for i, st in dev.stores.items()}
        arenas, sel_bufs = dev.arenas, dev.sel_bufs
        dev.arenas, dev.sel_bufs = self.arenas, self.sel_bufs
        try:
            prog = dev._compile(t, verify=verify)
        finally:
            dev.arenas, dev.sel_bufs = arenas, sel_bufs
            # its cold lists (the selection of its step, for HotCache); the
            # model's stay those of lane A
            if t == 1:
                self.ips = {i: st.get("ip") for i, st in dev.stores.items()}
            for i, st in dev.stores.items():
                st["ip"] = ips[i]
        ops = {op for op, _args in prog.recs}
        assert P.CPU_JOIN not in ops and P.CPU_START not in ops, \
            "lane B needs the flag form of the CPU tasks (RECLANK_GPU_FLAGS=1)"
        # its scratches of the C side its own (the programs of a model share
        # them, one at a time: the hot MoE of the two steps read each other's
        # tables, an out-of-range address)
        lib().rc_gpu_scratch_own(1)
        try:
            g = dev._build_program(prog, ("lane", t, verify))
        finally:
            lib().rc_gpu_scratch_own(0)
        g.run_blocks = self._run_blocks
        # live for the model's LRU: a program it frees keeps the device copies of the arrays
        # lane B's programs share
        dev.__dict__.setdefault("extra_progs", []).append(prog)
        return prog, g

    def _own(self, build):
        """build() with lane B's arenas, sel buffers, and C scratches; the model's cold lists stay.
        Return (its result, the stores' cold lists as build left them)."""
        dev = self.dev
        ips = {i: st.get("ip") for i, st in dev.stores.items()}
        arenas, sel_bufs = dev.arenas, dev.sel_bufs
        dev.arenas, dev.sel_bufs = self.arenas, self.sel_bufs
        lib().rc_gpu_scratch_own(1)
        try:
            out = build()
        finally:
            lib().rc_gpu_scratch_own(0)
            dev.arenas, dev.sel_bufs = arenas, sel_bufs
            left = {i: st.get("ip") for i, st in dev.stores.items()}
            for i, st in dev.stores.items():
                st["ip"] = ips[i]
        return out, left

    def mtp_prog(self, t):
        """Lane B's program of the MTP layer for t rows (as Qwen4GPU._mtp_group)."""
        e = self.mprogs.get(t)
        if e is None:
            from reclank.models.qwen.gpu38 import _Emit4, _fuse, compile_qwen4_step
            dev = self.dev

            def build():
                dev._pool = {}
                prog = dev._pack(_fuse(compile_qwen4_step(_Emit4(dev, t, False, False, bias=True), t, mtp=True), t))
                g = dev._build_program(prog, ("lane", "mtp", t))
                g.run_blocks = self._run_blocks
                return prog, g
            e, left = self._own(build)
            if t == 1:
                self.mip = left[dev.cfg.num_hidden_layers]
            self.mprogs[t] = e
            dev.__dict__.setdefault("extra_progs", []).append(e[0])
        return e

    def _draft_head(self):
        return "mtp_head.weight" in self.dev.model.g.tensors and os.environ.get("RECLANK_FP4L_MTP_HEAD", "1") != "0"

    def head_prog(self, prob=False):
        """Lane B's head program of one row (Qwen4GPU._head: its C scratches its own)."""
        key = (1, "p" if prob else "", "d" if self._draft_head() else "")
        e = self.hprogs.get(key)
        if e is None:
            dev = self.dev
            saved, dev.head_progs = dev.head_progs, {}
            try:
                e, _left = self._own(lambda: dev._head(1, prob, self._draft_head()))
            finally:
                dev.head_progs = saved
            self.hprogs[key] = e
            dev.__dict__.setdefault("extra_progs", []).append(e[0])
        return e

    def mtp(self, Hs, ids, pos):
        """The MTP layer on lane B's rows (as Qwen4GPU.mtp, at most LANE_MTP_ROWS rows), on its own
        programs; no HotCache work (the choices wait in msel: flush_scores). Safe beside lane A's
        run (in the pair): mtp_ready first. Return the streams of the layer."""
        from reclank.models.qwen.qwen35 import rope_positions
        dev = self.dev
        cfg = dev.cfg
        t = len(ids)
        size = next(s for s in (1, 2, 4) if s >= t)
        prog, g = self.mtp_prog(size)
        e, h = prog.names["e"], prog.names["h"]
        e[:t] = dev.model.embed(ids)
        e[t:] = 0.0
        h[:t] = Hs.reshape(t, -1)
        h[t:] = 0.0
        g.upload("e")
        g.upload("h")
        L = cfg.num_hidden_layers
        cos, sin = dev.model.rope(rope_positions(self.cache, pos, size))
        assert pos + size <= self.cache_dev.share_to or not self.cache_dev.ON, "mtp_ready first"
        kw = {"pos": pos, "nreal": t, "cos": np.ascontiguousarray(cos, np.float32),
              "sin": np.ascontiguousarray(sin, np.float32),
              "scores": np.empty(cfg.num_heads * (pos + size) + 64, np.float32)}
        for nm, a in zip(("kq", "ks", "vq", "vs"), self.mcache.kv[L]):
            kw["%s.%d" % (nm, L)] = a
        g.bind(kw, self.cache_dev, scratch=("scores",))
        g.run()
        g.download("H")
        k = cfg.top_k
        if size == 1:
            self.msel.append((1, self.mip[k + 1:2 * k + 1].copy()))
        else:
            sel = self.sel_bufs[size]
            g.mirror.buffer_of(sel).download(sel)
            self.msel.append((t, sel[L, :t].copy()))
        self.mlast = g.mirror.buffer_of(prog.names["xn"]).ptr + (t - 1) * cfg.hidden_size * 4
        return prog.names["H"][:t].copy()

    def mtp_argmax(self, prob=False):
        """The draft of lane B's last MTP run (its head program)."""
        hp, hg = self.head_prog(prob)
        _check(lib().rc_gpu_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr, self.mlast,
                                self.dev.cfg.hidden_size * 4))
        hg.run()
        hg.download("tok")
        if prob:
            hg.download("prob")
            return hp.names["tok"].copy(), hp.names["prob"].copy()
        return hp.names["tok"].copy()

    def mtp_ready(self, pos, rows):
        """On the model thread before a pair whose drafts reach pos + rows: the programs, the
        scratch of their scores, and the cache rows kept (share_tail), so that nothing in the
        pair allocates or changes what HotCache sees."""
        from reclank.models.qwen.qwen35 import rope_positions
        need = pos + rows + 4
        self.cache_dev.share_tail(need, self.cache_dev.before)
        nb = (self.dev.cfg.num_heads * need + 64) * 4
        for t in (1, 2, 4):
            _prog, g = self.mtp_prog(t)
            kw = {}
            b = g.named.get("scores")
            if b is None or b.nbytes < nb:
                kw["scores"] = np.empty(nb // 4, np.float32)
            if "cos" not in g.named:
                cos, sin = self.dev.model.rope(rope_positions(self.cache, 0, t))
                kw["cos"], kw["sin"] = np.ascontiguousarray(cos, np.float32), np.ascontiguousarray(sin, np.float32)
            if kw:
                g.bind(kw, None, scratch=("scores",))
        self.head_prog(False)
        # its verify group of rows tokens: the QSA scratch and its scores for the last position
        e = self.vprogs.get(rows)
        if e is not None:
            with self.view() as dev:
                kw = dev._params(pos, rows, rows)
            e[1].bind({k: v for k, v in kw.items() if k in ("scores",)}, None, scratch=("scores",))

    def flush_scores(self):
        """HotCache: the choices of lane B's MTP runs (as Qwen4GPU.mtp scores them), on the model
        thread after the pair."""
        hc = self.dev.hot_cache
        msel, self.msel = self.msel, []
        if hc is None:
            return
        L = self.dev.cfg.num_hidden_layers
        idx = [r for r, e in enumerate(hc.layers) if e["layer"] == L]
        for t, sel in msel:
            if t == 1:
                hc.score_rows(idx, sel[None])
            else:
                hc.score_group(idx, sel[:, None, :])

    def _run_blocks(self):
        dev = self.dev
        return self.cache_dev.blocks() + (dev.pool.run_locks() if dev.pool is not None else [])

    # the state of a lane that the model's attributes hold while that lane runs: (the model's
    # attribute, lane B's field)
    SWAP = (("cache", "cache"), ("cache_dev", "cache_dev"), ("_qsa_buf", "qsa_buf"),
            ("_qpos_version", "qpos_version"), ("mcache", "mcache"), ("last", "last"),
            ("last_prog", "last_prog"), ("rows", "rows"), ("_pending", "pending"), ("_mtp_last", "mtp_last"))

    @contextlib.contextmanager
    def view(self):
        """The model with lane B's state in place of lane A's (its cache, its last run, its verify
        group to commit, its MTP cache): on the model thread, between pairs. The model's own
        programs then serve lane B (a prompt, a step or a verify group alone)."""
        dev = self.dev
        saved = [getattr(dev, d, None) for d, _f in self.SWAP]
        for d, f in self.SWAP:
            setattr(dev, d, getattr(self, f))
        try:
            yield dev
        finally:
            for d, f in self.SWAP:
                setattr(self, f, getattr(dev, d, None))
            for (d, _f), v in zip(self.SWAP, saved):
                setattr(dev, d, v)

    def attach(self, cache):
        """lane B's cache (its device copy, cache1), with a new cache of the
        MTP layer (as Qwen4GPU.attach)."""
        self.cache_dev.attach(cache)
        self.cache = cache
        dev = self.dev
        if dev.has_mtp:
            from reclank.models.qwen.qwen38 import Qwen4MTPCache
            self.mcache = Qwen4MTPCache(dev.cfg, cache.max_len)
            self.cache_dev.add(self.mcache.kv[dev.cfg.num_hidden_layers], fresh=True)
        self.cache_dev.finish()

    def prefill(self, ids, streams=False):
        """Run a prompt on lane B's cache (alone). Return the first token
        (greedy); with streams, also the streams after the last layer of
        each token (the input of its MTP layer)."""
        with self.view() as dev:
            H = dev.prefill(ids, streams=streams)
            self.last = dev.last
            tok = int(dev.argmax(1)[0])
        self.cache.n = len(ids)
        self.rows = 1
        return (tok, H) if streams else tok

    # ---- its verify groups (MTP lanes) ----

    def verify_prog(self, t):
        e = self.vprogs.get(t)
        if e is None:
            e = self.vprogs[t] = self._program(t, verify=True)
        return e

    def prepare_verify(self, tokens, pos):
        """The inputs and parameters of lane B's verify group of tokens at
        pos (as Qwen4GPU.group with verify). Return its program."""
        t = len(tokens)
        prog, g = self.verify_prog(t)
        with self.view() as dev:
            snap = (dev.cache.ple_ids.copy(),
                    {i: dev._download(dev.cache.ple_conv[i]) for i in dev.cfg.ple_layers})
            dev._inputs(prog, g, list(tokens), pos)
            g.bind(dev._params(pos, t, t), self.cache_dev, scratch=("scores", "qsa_scratch"))
        self.pending = (prog, pos, list(tokens), snap)
        return g

    def after_verify(self):
        prog, _pos, tokens, _snap = self.pending
        g = self.vprogs[len(tokens)][1]
        t = len(tokens)
        self.last = g.mirror.buffer_of(prog.names["xn"]).ptr + (t - 1) * self.dev.cfg.hidden_size * 4
        self.last_prog = (prog, g)
        self.rows = t

    def streams(self, rows):
        """The streams after the last layer of the last rows rows of lane B's
        last verify group (Qwen4GPU.streams)."""
        prog, g = self.last_prog
        g.download("H")
        return prog.names["H"][self.rows - rows:self.rows].copy()

    def commit(self, n):
        """Keep the first n tokens of lane B's last verify group
        (Qwen4GPU.commit: the DeltaNet states, HotCache scores its
        choices)."""
        with self.view() as dev:
            dev._pending = self.pending
            sel_bufs, dev.sel_bufs = dev.sel_bufs, self.sel_bufs
            try:
                dev.commit(n)
            finally:
                dev.sel_bufs = sel_bufs
        self.pending = None

    def draft(self, drafter, token, pos, n):
        """drafter.draft (a QwenMTPDrafter of the model) on lane B's MTP
        cache."""
        with self.view():
            return drafter.draft(token, None, pos, n, ())

    def prepare_step(self, token, pos):
        """The inputs and parameters of lane B's step of token at pos."""
        with self.view() as dev:
            dev._inputs(self.prog, self.g, [token])
            self.g.bind(dev._params(pos, 1, 1), self.cache_dev, scratch=("scores", "qsa_scratch"))

    def after_step(self, pos):
        self.cache.n = pos + 1
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.last_prog = (self.prog, self.g)
        self.rows = 1

    def argmax(self, prob=False, rows=1):
        """The best token of each of the last rows rows of lane B's last run
        (the model's head)."""
        dev = self.dev
        saved = (dev.last, dev.rows, dev.hot_cache.due if dev.hot_cache is not None else False)
        dev.last, dev.rows = self.last, self.rows
        if dev.hot_cache is not None:
            dev.hot_cache.due = False
        try:
            return dev.argmax(rows, prob=prob)
        finally:
            dev.last, dev.rows = saved[0], saved[1]
            if dev.hot_cache is not None:
                dev.hot_cache.due = saved[2]

    def observe(self):
        """HotCache: the selection of lane B's last step (as HotCache.observe
        of the model's)."""
        hc = self.dev.hot_cache
        if hc is None:
            return
        k = hc.top_k
        rows = [r for r, e in enumerate(hc.layers) if e.get("step", True)]
        ips = [self.ips[hc.layers[r]["layer"]] for r in rows]
        hc.steps += 1
        hc.cold += int(sum(int(ip[k]) for ip in ips))
        hc.score_rows(rows, np.stack([ip[k + 1:2 * k + 1] for ip in ips]))

    def close(self):
        self.runner.close()
        self.g.close()
        for _p, g in list(self.vprogs.values()) + list(self.mprogs.values()) + list(self.hprogs.values()):
            g.close()
        mine = {id(self.prog)} | {id(p) for p, _g in (list(self.vprogs.values()) + list(self.mprogs.values())
                                                     + list(self.hprogs.values()))}
        self.mprogs, self.hprogs = {}, {}
        self.dev.extra_progs = [p for p in getattr(self.dev, "extra_progs", []) if id(p) not in mine]
        self.vprogs = {}
        self.cache_dev.unshare()
        for _a, b in list(self.cache_dev.bufs.values()):
            self.cache_dev._free(b)
        self.cache_dev.bufs = {}


def step_pair(dev, lane, ta, pa, tb, pb):
    """One step of each lane at once: token ta at pa on the model (lane A),
    tb at pb on lane B. dev.argmax() and lane.argmax() then give their next
    tokens; HotCache scores lane A's selection at dev.argmax (as a step), and
    lane B's here, after both runs."""
    hc = dev.hot_cache
    dev._warm_on()
    if hc is not None:
        hc.prepare()
    if not lane.warm:
        # lane B's first run alone (its graphs on its stream)
        lane.prepare_step(tb, pb)
        lane.runner.solo(lane.g, [lane.qsa_buf] if lane.qsa_buf is not None else [])
        lane.after_step(pb)
        lane.warm = True
        lane.observe()
        dev.step(ta, pa)
        return
    # each program's blocks locked right after its bind: the other's bind can
    # reclaim memory, and GpuMem moves blocks that are not locked
    mem().poll()
    dev._inputs(dev.prog, dev.g, [ta])
    dev.g.bind(dev._params(pa, 1, 1), dev.cache_dev, scratch=("scores", "qsa_scratch"))
    # (and the QSA scratch: its address is a parameter, not an array of the bind)
    locked = run_blocks_of(dev.g) + ([dev._qsa_buf] if dev._qsa_buf is not None else [])
    mem().lock(locked)
    lane.prepare_step(tb, pb)
    lb = run_blocks_of(lane.g) + ([lane.qsa_buf] if lane.qsa_buf is not None else [])
    mem().lock(lb)
    run_pair(dev.g, lane.g, lane.runner, locked + lb)
    dev.cache.n = pa + 1
    dev.last = dev.g.mirror.buffer_of(dev.prog.names["xn"]).ptr
    dev.last_prog = (dev.prog, dev.g)
    dev.rows = 1
    lane.after_step(pb)
    lane.observe()
    if hc is not None:
        hc.due = True


def greedy_pair(dev, lane, ids_a, ids_b, n):
    """Greedy decode of two prompts at once (n tokens each; dev and lane
    attached to caches): the tokens of each."""
    dev.prefill(ids_a)
    ya = int(dev.argmax(1)[0])
    yb = lane.prefill(ids_b)
    out_a, out_b = [ya], [yb]
    pa, pb = len(ids_a), len(ids_b)
    while len(out_a) < n:
        step_pair(dev, lane, ya, pa, yb, pb)
        ya = int(dev.argmax(1)[0])
        yb = int(lane.argmax()[0])
        out_a.append(ya)
        out_b.append(yb)
        pa += 1
        pb += 1
    return out_a, out_b


# ---- MTP lanes: a verify group of each lane at once ----

class _LaneMTP:
    """The model of a QwenMTPDrafter of lane B: groups of up to LANE_MTP_ROWS rows on lane B's own
    MTP programs (Lane.mtp: in the pair or not), more (a prompt's) on the model's through
    Lane.view (on the model thread, not in a pair)."""

    def __init__(self, lane):
        self.lane = lane
        self.own = True
        self.tok = None

    def mtp_rows(self, H, ids, pos):
        if len(ids) <= LANE_MTP_ROWS:
            self.own = True
            return self.lane.mtp(H, ids, pos)
        self.own = False
        with self.lane.view() as dev:
            hm = dev.mtp_rows(H, ids, pos)
            self.tok = dev.argmax(prob=True)
        return hm

    def argmax(self, prob=False):
        if self.own:
            return self.lane.mtp_argmax(prob)
        return self.tok if prob else self.tok[0]


def _bind_a(dev, ta, pa):
    """Lane A's verify group of ta at pa: its inputs, parameters, and locked blocks (as
    Qwen4GPU.group with verify). Return (prog, g, snap, locked)."""
    t = len(ta)
    prog, g = dev._group(t, True)
    snap = (dev.cache.ple_ids.copy(), {i: dev._download(dev.cache.ple_conv[i]) for i in dev.cfg.ple_layers})
    dev._inputs(prog, g, list(ta), pa)
    g.bind(dev._params(pa, t, t), dev.cache_dev, scratch=("scores", "qsa_scratch"))
    locked = run_blocks_of(g) + ([dev._qsa_buf] if dev._qsa_buf is not None else [])
    mem().lock(locked)
    return prog, g, snap, locked


def _after_a(dev, prog, g, snap, ta, pa):
    t = len(ta)
    dev.last = g.mirror.buffer_of(prog.names["xn"]).ptr + (t - 1) * dev.cfg.hidden_size * 4
    dev.last_prog = (prog, g)
    dev.rows = t
    dev._pending = (prog, pa, list(ta), snap)


def verify_pair_drafting(dev, lane, ta, pa, db, tb0, pb, draft):
    """verify_pair with lane B's drafts in the pair: lane B's thread drafts (db, a QwenMTPDrafter
    of _LaneMTP: Lane.mtp, its own programs) from its token tb0 at pb, then runs its verify group,
    while the model thread runs lane A's verify group of ta at pa. Lane.mtp_ready(pb, draft + 1)
    first. Return lane B's drafts."""
    hc = dev.hot_cache
    dev._warm_on()
    if hc is not None:
        hc.prepare()
    mem().poll()
    prog, g, snap, locked = _bind_a(dev, ta, pa)
    box = {}

    def job():
        drb = box["drafts"] = db.draft(tb0, None, pb, draft, ())
        bb = [tb0] + drb
        if len(bb) not in lane.vwarm:
            return (), []           # (its first run of this size alone, after the pair)
        gb = lane.prepare_verify(bb, pb)
        lb = run_blocks_of(gb) + ([lane.qsa_buf] if lane.qsa_buf is not None else [])
        mem().lock(lb)
        sq = gb.prog.by_name.get("seq")
        if sq is not None:
            gb.seq = getattr(gb, "seq", 0) + 1
            gb.env[sq.index] = gb.seq
        _check(lib().rc_gpu_run(gb.handle, gb.env.ctypes.data))
        lane.after_verify()
        box["ran"] = True
        return (gb,), lb
    run_pair_job(g, job, lane.runner, locked)
    _after_a(dev, prog, g, snap, ta, pa)
    drb = box["drafts"]
    if not box.get("ran"):
        bb = [tb0] + drb
        gb = lane.prepare_verify(bb, pb)
        lane.runner.solo(gb, [lane.qsa_buf] if lane.qsa_buf is not None else [])
        lane.after_verify()
        lane.vwarm.add(len(bb))
    return drb

def verify_pair(dev, lane, ta, pa, tb, pb):
    """A verify group of each lane at once (as Qwen4GPU.verify): tokens ta at
    pa on the model, tb at pb on lane B. dev.argmax(rows) and lane.argmax(
    rows=) then give the picks of their rows, dev.commit and lane.commit keep
    the rows of the kept tokens (HotCache scores their choices there)."""
    hc = dev.hot_cache
    dev._warm_on()
    if hc is not None:
        hc.prepare()
    if len(tb) not in lane.vwarm:
        # the first run of lane B's verify group of this size alone
        gb = lane.prepare_verify(tb, pb)
        lane.runner.solo(gb, [lane.qsa_buf] if lane.qsa_buf is not None else [])
        lane.after_verify()
        lane.vwarm.add(len(tb))
        dev.verify(ta, pa)
        return
    mem().poll()
    prog, g, snap, locked = _bind_a(dev, ta, pa)
    gb = lane.prepare_verify(tb, pb)
    lb = run_blocks_of(gb) + ([lane.qsa_buf] if lane.qsa_buf is not None else [])
    mem().lock(lb)
    run_pair(g, gb, lane.runner, locked + lb)
    _after_a(dev, prog, g, snap, ta, pa)
    lane.after_verify()


def greedy_mtp_pair(dev, lane, ids_a, ids_b, n, draft=3, stats=None):
    """Greedy decode of two prompts at once with MTP drafts (n tokens each;
    dev and lane attached to caches with room for n + draft + 1 rows more):
    each lane drafts with the MTP layer (one after the other, on the model
    thread), then the two verify groups run as a pair (speculative.stream's
    round, greedy); when one lane has its n tokens the other goes on alone.
    The tokens of each; stats gets the pair rounds, the drafts kept of each
    lane in them, and their seconds."""
    import time
    from reclank.core.speculative import QwenMTPDrafter
    Ha = dev.prefill(ids_a, streams=True)
    ya = int(dev.argmax(1)[0])
    yb, Hb = lane.prefill(ids_b, streams=True)
    da, db = QwenMTPDrafter(dev), QwenMTPDrafter(_LaneMTP(lane))
    z = np.zeros((1, Ha.shape[1]), np.float32)
    da.observe(list(ids_a) + [ya], np.concatenate([z, Ha]), 0)
    db.observe(list(ids_b) + [yb], np.concatenate([z, Hb]), 0)
    out = {"a": [ya], "b": [yb]}
    pos = {"a": len(ids_a), "b": len(ids_b)}
    st = stats if stats is not None else {}
    for key in ("rounds", "kept_a", "kept_b", "draft_s", "verify_s"):
        st.setdefault(key, 0)

    def keep(nm, d, r, H, drafter):
        kept = []
        for j in range(len(d) + 1):
            kept.append(int(r[j]))
            if j == len(d) or int(r[j]) != d[j]:
                break
        k = len(kept)
        if nm == "a":
            dev.commit(k)
        else:
            lane.commit(k)
        drafter.observe(kept, H[:k], pos[nm] + 1)
        pos[nm] += k
        out[nm].extend(kept)
        return k

    while len(out["a"]) < n and len(out["b"]) < n:
        t0 = time.perf_counter()
        dra = da.draft(out["a"][-1], None, pos["a"], draft, ())
        ba = [out["a"][-1]] + dra
        lane.mtp_ready(pos["b"], draft + 1)
        if OVERLAP and len(db.pend[0]) <= LANE_MTP_ROWS and draft + 1 in lane.vwarm:
            t1 = time.perf_counter()
            drb = verify_pair_drafting(dev, lane, ba, pos["a"], db, out["b"][-1], pos["b"], draft)
            bb = [out["b"][-1]] + drb
        else:
            drb = db.draft(out["b"][-1], None, pos["b"], draft, ())
            t1 = time.perf_counter()
            bb = [out["b"][-1]] + drb
            verify_pair(dev, lane, ba, pos["a"], bb, pos["b"])
        lane.flush_scores()
        Ha, Hb = dev.streams(len(ba)), lane.streams(len(bb))
        ra = dev.argmax(rows=len(ba))
        rb = lane.argmax(rows=len(bb))
        st["kept_a"] += keep("a", dra, ra, Ha, da) - 1
        st["kept_b"] += keep("b", drb, rb, Hb, db) - 1
        st["rounds"] += 1
        st["draft_s"] += t1 - t0
        st["verify_s"] += time.perf_counter() - t1
    # the lane left alone
    while len(out["a"]) < n:
        d = da.draft(out["a"][-1], None, pos["a"], draft, ())
        b = [out["a"][-1]] + d
        dev.verify(b, pos["a"])
        keep("a", d, dev.argmax(rows=len(b)), dev.streams(len(b)), da)
    while len(out["b"]) < n:
        lane.mtp_ready(pos["b"], draft + 1)
        d = db.draft(out["b"][-1], None, pos["b"], draft, ())
        lane.flush_scores()
        b = [out["b"][-1]] + d
        lane.vwarm.add(len(b))
        dev._warm_on()
        if dev.hot_cache is not None:
            dev.hot_cache.prepare()
        gb = lane.prepare_verify(b, pos["b"])
        lane.runner.solo(gb, [lane.qsa_buf] if lane.qsa_buf is not None else [])
        lane.after_verify()
        keep("b", d, lane.argmax(rows=len(b)), lane.streams(len(b)), db)
    return out["a"][:n], out["b"][:n]
