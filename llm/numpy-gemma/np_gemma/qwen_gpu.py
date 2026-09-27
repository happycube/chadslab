"""Qwen3.5 / Qwen3.6 MoE on the GPU, with the experts split.

QWEN_PLAN.md, phase 4, for the GGUF file (QwenGGUFProgram). The design is
that of the 26B (np_gemma/gpu.py, ModelGPU):

- The dense part runs on the GPU: the products (Q8_0 and F32), the Gated
  DeltaNet, and the gated attention. The router, the shared expert, and the
  head (Q6_K) also run there. The GPU products read x in float32.
- Each layer holds some experts on the GPU (the hot experts, in slots). The
  GPU computes the selected hot experts and the shared expert
  (GP_KQ_HOT_MOE). The CPU computes the other selected experts (the cold
  experts) at the same time. GP_TO_HOST copies the input, and GP_CPU_JOIN
  runs a CPU program (KQ_QUANT and KQ_MOE of cops). GP_TO_DEV copies the
  output back.
- HotCache (np_gemma/gpu.py) changes the hot experts as the text goes on.

Three kinds of program:

- the step (one token);
- a group with the experts split as in the step (an MTP verify group, or
  a short part of a prompt). A group of at most MT tokens runs the
  attention of each query with the kernel of the step. Then it gives the
  bits of steps;
- a large group (a part of a prompt). The experts of each layer come to
  the GPU (GP_FETCH, as for the 26B), and GP_KQ_GROUP_MOE computes them.

A group can have more rows than tokens. The slot nreal gives the count of
the real tokens; the other rows do not change the state of the linear
layers.

    g = QwenGPU(model)                   # model: a QwenGGUFProgram
    cache = QwenCache(model.cfg, 8192)
    g.attach(cache)
    h = g.prefill(ids)                   # the prompt; the last hidden state
    g.step(token, pos); logits = g.logits()
    h = g.verify(tokens, pos); g.commit(n)
    g.detach(cache)                      # the host cache has the new values
"""
from __future__ import annotations

import os

import numpy as np

from . import cops
from . import program as P
from .gpu import Buffer, GPUProgram, HotCache, _check, lib, mem_info, pinned
from .qwen import cache_params, compile_qwen_step

Q8_0, Q8_R = 8, 100     # the ggml type; its rows for the GPU (csrc/gpu.cu)
MT = 16                 # the largest small group (MT_MAX of csrc/gpu.cu)
# The shortest part of a prompt that runs as a large group. A group of 1024
# rows takes about 2.6 s (mostly the copies of the experts, 7.5 GB/s); split
# groups of 256 rows take about as long for 700 tokens (QWEN_PLAN.md).
FETCH_MIN = int(os.environ.get("NP_GEMMA_GPU_FETCH_MIN", "700"))
SPLIT_SIZES = (16, 64, 128, 256)  # the split groups of a prompt
FETCH_SIZES = (1024,)             # the large groups


class _Emit:
    """The model as compile_qwen_step sees it for a GPU program: the products
    read copies of the weights that the GPU holds, and the experts go to
    QwenGPU.emit_moe."""

    def __init__(self, dev, t, verify, fetch):
        self.dev = dev
        self.m = dev.model
        self.t = t
        self.verify = verify
        self.fetch = fetch

    def __getattr__(self, name):
        return getattr(self.m, name)

    def emit_lin(self, prog, xb, name, out):
        m = self.m.M(name)
        w, type_ = self.dev.dense(name)
        prog.emit(P.KQ_LINEAR, xb["xq"], xb["xs"], xb["xm"], xb["src"], w, type_, m.rows, m.cols,
                  xb["t"], out)

    def moe_scratch(self, t):
        return None

    def emit_moe(self, prog, xb, p, idx, val, slog, scratch, out):
        i = int(p.split(".")[1])
        if self.fetch:
            self.dev._moe_fetch(prog, i, xb["src"], idx, val, slog, out)
        else:
            self.dev._moe_split(prog, i, xb["src"], idx, val, slog, out)

    def emit_attn(self, prog, q, K, V, scores, out, nq, nk, hd, t, pos, hs):
        if t == 1 or t > MT:
            prog.emit(P.ATTN_F32H, q, K, V, scores, out, nq, nk, hd, t, pos, hs, 0, 0)
            return
        # A small group: each query with the kernel of the step (the same
        # bits as a step). Query j is at pos + j.
        for j in range(t):
            pj = prog.temp()
            prog.emit(P.S_ADD, pj, pos, j)
            prog.emit(P.ATTN_F32H, q[j:j + 1], K, V, scores, out[j:j + 1], nq, nk, hd, 1, pj, hs,
                      0, 0)


def _fuse(prog, t):
    """Fewer launches: drop GP_KQ_QUANT (the GPU products read x); GP_ADD and
    the GP_RMS_NORM of its output become GP_ADD_RMS; up to 5 products of a
    small group on the same x become one GP_KQ_MULTI."""
    recs = [r for r in prog.recs if r[0] != P.KQ_QUANT]
    out = []
    i = 0
    while i < len(recs):
        op, a = recs[i]
        if op == P.ADD and i + 1 < len(recs) and recs[i + 1][0] == P.RMS_NORM:
            b = recs[i + 1][1]
            if a[2] == a[0] and b[0] == a[0]:
                out.append((P.ADD_RMS, [a[0], a[1], b[1], b[2], b[3], b[4], b[5]]))
                i += 2
                continue
        if op == P.KQ_LINEAR and t <= MT:
            group = [a]
            j = i + 1
            while (j < len(recs) and recs[j][0] == P.KQ_LINEAR and recs[j][1][3] == a[3]
                   and len(group) < 5):
                group.append(recs[j][1])
                j += 1
            if len(group) > 1:
                args = [a[3], a[7], a[8], (P.T_INT, len(group))]
                for g in group:
                    args += [g[4], g[5], g[6], g[9]]
                out.append((P.KQ_MULTI, args))
                i = j
                continue
        out.append((op, a))
        i += 1
    prog.recs = out
    return prog.finish()


class QwenGPU:
    """Qwen3.5 MoE (a QwenGGUFProgram) on the GPU; see the module text.

    hot_gb is the memory for the hot experts. None takes NP_GEMMA_GPU_HOT_GB,
    or else the free memory of the GPU less the dense part, the head, the two
    buffers of the copies of a large group, and 2 GB for the rest (the
    caches, the programs, and their graphs). counts
    (layers x experts, or None) selects the first hot experts; else each
    layer starts with its first experts, and HotCache changes them."""

    def __init__(self, model, hot_gb=None, counts=None, graph=True):
        self.model = model
        cfg = self.cfg = model.cfg
        self.graph = graph
        self._dense = {}
        self.cpu_progs = []
        self.n_events = 0
        E, L = cfg.num_experts, model.n_layers
        self.E, self.L = E, L
        # The bytes of one expert of each matrix (gate, up, down), the most
        # over the layers.
        self.per = [max(model.M("layers.%d.mlp.switch_mlp.%s" % (i, n)).data.nbytes // E
                        for i in range(L)) for n in ("gate_proj", "up_proj", "down_proj")]
        per = sum(self.per)
        if hot_gb is None and os.environ.get("NP_GEMMA_GPU_HOT_GB"):
            hot_gb = float(os.environ["NP_GEMMA_GPU_HOT_GB"])
        if hot_gb is None:
            # n slots take L n per; the two buffers take 2 (E - n) per.
            head = model.M("lm_head").data.nbytes
            room = mem_info()[0] - self._dense_bytes() - head - 2.0e9 - 2 * E * per
            n = int(max(1, room // ((L - 2) * per)))
        else:
            n = int(max(1, hot_gb * 1e9 // (L * per)))
        self.n_slots = n = min(E, n)
        self.stores = {}
        for i in range(L):
            if counts is not None:
                init = np.sort(np.argsort(-np.asarray(counts[i]), kind="stable")[:n])
            else:
                init = np.arange(n)
            self.stores[i] = self._store(i, init)
        self.prog = self._compile(1)
        self.g = GPUProgram(self.prog, graph=graph)
        self.groups = {}        # (t, verify) -> (Program, GPUProgram)
        self.stage = None       # the two buffers of the copies of a large group
        self.tables = {}        # layer -> (gate, up, down tables, ranges)
        self.tables_version = None
        self.cache_dev = _DevCache()
        self.cache = None
        self.head = None
        self.rows = 1
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.hot_cache = None
        if os.environ.get("NP_GEMMA_GPU_HOT_DYN", "1") != "0":
            mir = self.g.mirror
            layers = []
            for i, st in sorted(self.stores.items()):
                layers.append(dict(
                    layer=i, slots=st["slots"], dslots=mir.buffer_of(st["slots"]).ptr,
                    parts=[(src.ctypes.data, nb, mir.buffer_of(dst).ptr)
                           for src, nb, dst in st["parts"]],
                    ip=st["ip"]))
            self.hot_cache = HotCache(self, layers=layers, top_k=cfg.top_k)

    # ---- the weights ----

    def _dense_bytes(self):
        """The bytes of the dense tensors of the file: all but the experts,
        the embeddings, and the head."""
        g = self.model.g
        n = 0
        for name in g.tensors:
            if "_exps" not in name and name not in ("token_embd.weight", "output.weight"):
                n += g.raw(name)[0].nbytes
        return n

    def dense(self, name):
        """A copy of the blocks of a dense matrix, and its type. The GPU
        program holds its own copy of each array (np_gemma/gpu.py, Mirror),
        not a view into the memory map of the file. A Q8_0 matrix goes to
        rows of int8 values and then their scales (type 100, Q8_R in
        csrc/gpu.cu), for 16-byte loads."""
        e = self._dense.get(name)
        if e is None:
            m = self.model.M(name)
            if m.type == Q8_0:
                b = m.data.reshape(m.rows, m.cols // 32, 34)
                a = np.concatenate([b[:, :, 2:].reshape(m.rows, m.cols),
                                    b[:, :, :2].reshape(m.rows, m.cols // 16)], axis=1)
                e = (np.ascontiguousarray(a).reshape(-1), Q8_R)
            else:
                e = (np.array(m.data), m.type)
            self._dense[name] = e
        return e

    def _store(self, i, init):
        """The slots of the experts of layer i: the gate, up, and down
        blocks of the experts init, the slot table, and the parts for
        HotCache."""
        E = self.cfg.num_experts
        p = "layers.%d.mlp.switch_mlp." % i
        mats = [self.model.M(p + n) for n in ("gate_proj", "up_proj", "down_proj")]
        st = {"mats": mats, "parts": []}
        for key, m in zip(("gate", "up", "down"), mats):
            nb = m.data.nbytes // E
            src = m.data.reshape(E, nb)
            st[key] = np.ascontiguousarray(src[init]).reshape(-1)
            st["parts"].append((m.data, nb, st[key]))
        slots = np.full(E, -1, dtype=np.int32)
        slots[init] = np.arange(len(init), dtype=np.int32)
        st["slots"] = slots
        return st

    # ---- the programs ----

    def _buf(self, key, make):
        """A scratch array of the program that compiles now: one for all the
        layers (the layers run in order)."""
        a = self._pool.get(key)
        if a is None:
            a = self._pool[key] = make()
        return a

    def _compile(self, t, verify=False, fetch=False):
        self._pool = {}
        prog = compile_qwen_step(_Emit(self, t, verify, fetch), t, verify=verify)
        if fetch:
            # The copies of the experts of layers 0 and 1 start first.
            first = []
            for i in (0, 1):
                ranges = self.tables[i][3]
                first.append((P.FETCH, [prog._enc(ranges), prog._enc(len(ranges)),
                                        prog._enc(i), prog._enc(i % 2)]))
            prog.recs[:0] = first
        return _fuse(prog, t)

    def _moe_split(self, prog, i, h, idx, val, slog, out):
        """The hot experts and the shared expert on the GPU, the cold experts
        on the CPU at the same time (a step, or a group that does not copy
        the experts)."""
        cfg = self.cfg
        st = self.stores[i]
        k, E, hidden, inner = cfg.top_k, cfg.num_experts, cfg.hidden_size, cfg.moe_inter
        t = h.shape[0]
        z = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        hp = self._buf("hp", lambda: pinned((t, hidden)))
        if t == 1:
            cold = np.zeros(2 * k + 1, dtype=np.int32)
            cold_val = np.zeros(k, dtype=np.float32)
            prog.emit(P.HOT_SPLIT, idx, val, st["slots"], cold, cold_val, k, 1)
            vp, ip = pinned((k,)), pinned((2 * k + 1,), np.int32)
            st["ip"] = ip       # the selection of the step, for HotCache
        else:
            cold = self._buf("cold", lambda: np.zeros((t, k), dtype=np.int32))
            cold_val = self._buf("cold_val", lambda: z(t, k))
            prog.emit(P.HOT_SPLIT_MT, idx, val, st["slots"], cold, cold_val, t * k,
                      prog.slot("nreal"), k)
            vp = self._buf("vp", lambda: pinned((t, k)))
            ip = self._buf("ip", lambda: pinned((t, k), np.int32))
        ev = self.n_events
        self.n_events += 1
        prog.emit(P.TO_HOST, h, hp, hp.nbytes, cold_val, vp, vp.nbytes, cold, ip, ip.nbytes, ev)
        s = "layers.%d.mlp.shared_expert." % i
        sg = self.model.M(s + "gate_proj")
        assert sg.rows == inner
        gm, _um, dm = st["mats"]
        gpu_part = self._buf("gpu_part", lambda: z(t, hidden))
        pairs = t * k + t
        prog.emit(P.KQ_HOT_MOE, h, val, idx, st["slots"], st["gate"], st["up"], st["down"],
                  self._buf("act", lambda: z(pairs, 2 * inner)),
                  self._buf("act2", lambda: z(pairs, inner)),
                  self._buf("de", lambda: z(pairs, hidden)), gpu_part, k, inner, hidden, gm.type,
                  dm.type, t, self.dense(s + "gate_proj")[0], self.dense(s + "up_proj")[0],
                  self.dense(s + "down_proj")[0], self.dense(s + "gate_proj")[1], slog)
        # The CPU part: the cold experts, on the pinned copies.
        cc = P.Program()
        xq = self._buf("xq", lambda: np.zeros((t, hidden), np.int8))
        xs, xm = self._buf("xs", lambda: z(t, hidden // 32)), self._buf("xm", lambda: z(t, hidden // 16))
        cc.emit(P.KQ_QUANT, hp, t, hidden, xq, xs, xm)
        mats = cops.kq_moe_mats(*(m.c() for m in st["mats"]), None)
        host_out = self._buf("host_out", lambda: z(t, hidden))
        cc.emit(P.KQ_MOE, xq, xs, xm, ip, vp, t, k, E, mats, None, hidden, inner,
                self._buf("cpu_scratch", lambda: cops.kq_moe_scratch(t, k, E, hidden, inner)),
                host_out, ip[k:] if t == 1 else None)
        cc.keep.append(mats)
        cpu = cc.finish()
        self.cpu_progs.append(cpu)
        part = self._buf("part", lambda: z(t, hidden))
        prog.emit(P.CPU_JOIN, cpu.buf, ev)
        prog.emit(P.TO_DEV, host_out, part, part.nbytes)
        prog.emit(P.ADD, gpu_part, part, out, t * hidden)

    def _moe_fetch(self, prog, i, h, idx, val, slog, out):
        """All the experts of a large group on the GPU. The cold experts of
        layer i come to buffer i % 2 (GP_FETCH); the copies of layer i + 2
        start when layer i is done with the buffer."""
        cfg = self.cfg
        st = self.stores[i]
        k, E, hidden, inner = cfg.top_k, cfg.num_experts, cfg.hidden_size, cfg.moe_inter
        t = h.shape[0]
        P_ = t * k + t
        tg, tu, td, _ranges = self.tables[i]
        s = "layers.%d.mlp.shared_expert." % i
        gm, _um, dm = st["mats"]
        tiles = -(-P_ // 64) + E + 1
        work = self._buf("work", lambda: np.zeros(8 + 2 * (E + 2) + 2 * P_ + 3 * tiles, np.int32))
        z = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        prog.emit(P.FETCH_WAIT, i)
        prog.emit(P.KQ_GROUP_MOE, h, val, idx, t, k, E, hidden, inner, tg, tu, td, gm.type,
                  dm.type, self.dense(s + "gate_proj")[0], self.dense(s + "up_proj")[0],
                  self.dense(s + "down_proj")[0], self.dense(s + "gate_proj")[1], slog, work,
                  self._buf("act", lambda: z(P_, 2 * inner)), self._buf("act2", lambda: z(P_, inner)),
                  self._buf("de", lambda: z(P_, hidden)), out, prog.slot("nreal"))
        prog.emit(P.FETCH_DONE, i % 2)
        if i + 2 < self.L:
            ranges = self.tables[i + 2][3]
            prog.emit(P.FETCH, ranges, len(ranges), i + 2, i % 2)

    def _make_tables(self):
        """The two buffers of the copies of a large group, and the tables of
        each layer: the device address of each expert (a hot slot, or its
        place in the buffer), and the copies of the cold experts."""
        if self.stage is not None:
            return
        cold = self.E - self.n_slots
        self.stage = [[Buffer(max(1, cold * nb)) for nb in self.per] for _ in range(2)]
        for i in range(self.L):
            self.tables[i] = (np.zeros(self.E, np.int64), np.zeros(self.E, np.int64),
                              np.zeros(self.E, np.int64), np.zeros((3 * self.E, 3), np.int64))

    def _fill_tables(self):
        """Fill the tables again when the slots changed (HotCache)."""
        version = self.hot_cache.version if self.hot_cache is not None else 0
        if self.tables_version == version:
            return
        mir = self.g.mirror
        for i in range(self.L):
            st = self.stores[i]
            tabs, ranges = self.tables[i][:3], self.tables[i][3]
            stage = self.stage[i % 2]
            slots = st["slots"]
            rows = []
            for part, (tab, (src, nb, dst), buf) in enumerate(zip(tabs, st["parts"], stage)):
                dbase = mir.buffer_of(dst).ptr
                rank, run = 0, None
                for x in range(self.E):
                    if slots[x] >= 0:
                        tab[x] = dbase + int(slots[x]) * nb
                        continue
                    tab[x] = buf.ptr + rank * nb
                    if run is not None and run[3] == x - 1:
                        run[2] += nb
                        run[3] = x
                    else:
                        run = [src.ctypes.data + x * nb, int(tab[x]), nb, x]
                        rows.append(run)
                    rank += 1
            ranges[:] = 0
            for r, run in enumerate(rows):
                ranges[r] = run[:3]
            for tab in tabs:
                if tab.ctypes.data in mir.bufs:     # else the first program copies it
                    mir.buffer_of(tab).upload(tab)
        self.tables_version = version

    def _group(self, t, verify=False, fetch=False):
        key = (t, verify, fetch)
        e = self.groups.get(key)
        if e is None:
            if fetch:
                self._make_tables()
                self._fill_tables()
            prog = self._compile(t, verify, fetch)
            g = GPUProgram(prog, graph=self.graph, mirror=self.g.mirror)
            e = self.groups[key] = (prog, g)
        return e

    # ---- the cache ----

    def attach(self, cache):
        """Copy the cache to the GPU. From now on the GPU copy is the true
        one."""
        self.cache_dev.attach(cache)
        self.cache = cache

    def detach(self, cache):
        self.cache_dev.to_host()
        self.cache_dev.release()

    def _params(self, pos, t, nreal):
        cache = self.cache
        assert pos + t <= cache.max_len, "the cache is too short"
        cos, sin = self.model.rope(np.arange(pos, pos + t))
        kw = {"pos": pos, "nreal": nreal, "cos": np.ascontiguousarray(cos, np.float32),
              "sin": np.ascontiguousarray(sin, np.float32),
              "scores": np.empty(self.cfg.num_heads * (pos + t) + 64, np.float32)}
        kw.update(cache_params(self.model, cache))
        return kw

    # ---- the runs ----

    def step(self, token, pos):
        """Run one token at position pos. logits() gives its logits."""
        if self.hot_cache is not None:
            self.hot_cache.prepare()
        self.prog.names["x"][:] = self.model.embed([token])
        self.g.upload("x")
        self.g.bind(self._params(pos, 1, 1), self.cache_dev, scratch=("scores",))
        self.g.run()
        self.cache.n = pos + 1
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.rows = 1
        if self.hot_cache is not None:
            self.hot_cache.due = True

    def group(self, tokens, pos, size=None, verify=False, fetch=False):
        """Run tokens from position pos as one group (of size rows, with
        padding). Return the hidden states after the final norm. With
        verify, the linear layers keep their state until commit(). With
        fetch, all the experts run on the GPU (a large group)."""
        t = len(tokens)
        size = size or t
        prog, g = self._group(size, verify, fetch)
        if self.hot_cache is not None:
            # A large group needs the slots of its tables: wait for the
            # copies of HotCache.
            self.hot_cache.prepare(wait=fetch)
        if fetch:
            self._fill_tables()
        x = prog.names["x"]
        x[:t] = self.model.embed(tokens)
        x[t:] = 0.0
        g.upload("x")
        g.bind(self._params(pos, size, t), self.cache_dev, scratch=("scores",))
        g.run()
        g.download("xn")
        hidden = self.cfg.hidden_size
        self.last = g.mirror.buffer_of(prog.names["xn"]).ptr + (t - 1) * hidden * 4
        self.rows = t
        if verify:
            self._pending = (prog, pos)
        else:
            self.cache.n = pos + t
        return prog.names["xn"][:t].copy()

    def verify(self, tokens, pos):
        """An MTP verify group: see group()."""
        return self.group(tokens, pos, verify=True)

    def commit(self, n):
        """Keep the first n tokens of the last verify group."""
        prog, pos = self._pending
        cfg = self.cfg
        mir = self.g.mirror
        for i in range(self.L):
            if cfg.layer_types[i] == "full_attention":
                continue
            _check(lib().gg_gdn_commit(
                self.cache_dev.device(self.cache.conv[i]), self.cache_dev.device(self.cache.state[i]),
                mir.buffer_of(prog.names["log.%d" % i]).ptr, n, cfg.conv_kernel, cfg.lin_k_heads,
                cfg.lin_v_heads, cfg.lin_k_dim, cfg.lin_v_dim))
        self.cache.n = pos + n
        self._pending = None

    def _sizes(self, rem, room):
        """The next group of a prompt with rem tokens left and room rows
        left in the cache: (size, tokens, fetch)."""
        if rem >= FETCH_MIN:
            for s in sorted(FETCH_SIZES, reverse=True):
                if s <= room and (rem >= s or s == min(FETCH_SIZES)):
                    return s, min(rem, s), True
        for s in SPLIT_SIZES:
            if s >= rem and s <= room:
                return s, rem, False
        s = max(z for z in SPLIT_SIZES if z <= max(room, MT))
        return s, min(rem, s), False

    def prefill(self, ids, pos=0):
        """Run a prompt from position pos. Return the hidden state of its
        last token."""
        ids = list(ids)
        c0 = 0
        h = None
        room = self.cache.max_len
        while c0 < len(ids):
            size, n, fetch = self._sizes(len(ids) - c0, room - pos - c0)
            h = self.group(ids[c0:c0 + n], pos + c0, size, fetch=fetch)
            c0 += n
        return h[-1:]

    def logits(self, rows=1):
        """The logits of the last rows of the last step or group (the head
        runs on the GPU; HotCache runs on the host meanwhile)."""
        cfg = self.cfg
        m = self.model.M("lm_head")
        if self.head is None:
            w = np.ascontiguousarray(m.data)
            self.head = Buffer(w.nbytes)
            self.head.upload(w)
            self.out = Buffer(4 * m.rows * MT)
            self.host_logits = pinned((MT, m.rows))
        assert 1 <= rows <= min(self.rows, MT)
        step = cfg.hidden_size * 4
        _check(lib().gg_q6k_head(self.head.ptr, self.last - (rows - 1) * step, self.out.ptr,
                                 m.rows, cfg.hidden_size, 0.0, rows))
        if self.hot_cache is not None and self.hot_cache.due:
            self.hot_cache.observe()
        _check(lib().gg_d2h(self.host_logits.ctypes.data, self.out.ptr, rows * m.rows * 4))
        return self.host_logits[:rows].copy() if rows > 1 else self.host_logits[0].copy()

    def close(self):
        """Free the memory of the GPU: the programs, the weights, the
        buffers, and the cache."""
        if self.hot_cache is not None:
            self.hot_cache.prepare(wait=True)
        for _p, g in self.groups.values():
            g.close()
        self.g.close()
        for b in self.g.mirror.bufs.values():
            b.free()
        self.g.mirror.bufs = {}
        for bufs in self.stage or ():
            for b in bufs:
                b.free()
        for b in (self.head, getattr(self, "out", None)):
            if b is not None:
                b.free()
        self.cache_dev.release()
        self.groups = {}
        self.stage = None
        self.hot_cache = None


class _DevCache:
    """The device copy of the arrays of a QwenCache: the keys and values of
    the full layers, the convolution and the state of the linear layers."""

    def __init__(self):
        self.bufs = {}

    def attach(self, cache):
        self.release()
        arrays = [a for kv in cache.kv.values() for a in kv]
        arrays += list(cache.conv.values()) + list(cache.state.values())
        for a in arrays:
            b = Buffer(a.nbytes)
            b.upload(np.ascontiguousarray(a))
            self.bufs[id(a)] = (a, b)

    def device(self, a):
        e = self.bufs.get(id(a))
        return None if e is None else e[1].ptr

    def to_host(self):
        for a, b in self.bufs.values():
            b.download(a)

    def release(self):
        for _a, b in self.bufs.values():
            b.free()
        self.bufs = {}
