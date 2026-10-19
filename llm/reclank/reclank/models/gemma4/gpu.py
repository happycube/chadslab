"""The Gemma 4 models on the GPU: the 26B and 12B (ModelGPU, its split
experts and the GPU cache GPUKV), the E4B (E4BGPU), and the MTP drafter
(GPUDrafter). The library, GPUProgram, Mirror, and HotCache are in
reclank/backends/cuda/runtime.py."""
from __future__ import annotations

from pathlib import Path
import os
import time

import numpy as np

from reclank.ir import alloc
from reclank.ir import program as P
from reclank.models.gemma4 import forms
from reclank.backends.cuda.gpumm import Buffer, DeviceCache, ProgramLRU, mem, mem_info, pinned
from reclank.models.gemma4.model import KV_KEEP
from reclank.backends.cuda.runtime import (  # noqa: F401
    GPUCache, GPUProgram, HotCache, MT_CPU, PoolCompiler, WarmPool, _argmax_rows,
    _check, _topk_rows, expert_host, fill_chunks, fill_tables, has_tc, int8_tiles, lib, stage_room)


_HERE = Path(__file__).resolve().parent


class E4BGroupCompiler(PoolCompiler):
    """The compiler of a group of the E4B model. A small group (at most MT_CPU
    tokens, such as an MTP verify group) runs the attention of each query
    with the record of a decode step. That kernel computes in float32, as
    the decode step does, so the verify group agrees with the decode."""

    def kernel(self, head, vals, out=None):
        if head == "attn_e4b" and vals[1].shape[0] <= MT_CPU:
            return self.attn_small(*vals)
        return super().kernel(head, vals, out)

    def attn_small(self, layer, q):
        plan = self.cfg.plan[layer]
        hd, qh = plan.head_dim, plan.num_q_heads
        t = q.shape[0]
        out = self.buffer((t, qh * hd))
        s = lambda name: self.p.slot("%s.%d" % (name, plan.source))  # noqa: E731
        slide = 1 if os.environ.get("RECLANK_SLIDE", "1") == "1" else 0
        for j in range(t):
            pos_j = self.scalar("+", [self.p.slot("pos"), j])
            self.p.emit(P.ATTN_F32H, q[j:j + 1], s("k"), s("v"), self.p.slot("scores"),
                        out[j:j + 1], qh, plan.num_kv_heads, hd, 1, pos_j, s("hs"),
                        plan.window, slide)
        return out


def _fuse_kq(prog):
    """Fewer launches of the GGUF products of an E4B program on the GPU:
    drop GP_KQ_QUANT (the GPU products read x), and make up to 5 GP_KQ_LINEAR
    of a few tokens on the same x one GP_KQ_MULTI (the query, the key, and
    the value; the gate and the up matrix). The values are the same."""
    recs = [r for r in prog.recs if r[0] != P.KQ_QUANT]
    out, i = [], 0
    while i < len(recs):
        op, a = recs[i]
        if op == P.KQ_LINEAR and a[8][0] == P.T_INT and a[8][1] <= MT_CPU and not P.kq_e3(a[5]):
            group, j = [a], i + 1
            while (j < len(recs) and recs[j][0] == P.KQ_LINEAR and recs[j][1][3] == a[3]
                   and len(group) < 5 and not P.kq_e3(recs[j][1][5])):
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


# 1: the Q4_0 matrices of an E4B step and of a verify group run with int8 x
# and dp4a (E4B.kq_q4, GP_KQ_LINEAR), as the K quants. With float32 x, the
# loads of x for each token made a group of 3 tokens cost 1.75 steps
# (MTP_PLAN.md). 0 keeps float32 x (GP_INT4_LINEAR). The prompt keeps the
# int4 records either way.
Q4_I8 = os.environ.get("RECLANK_GPU_Q4_I8", "1") != "0"

# The dense Q4_0 matrices of a step and of a verify group of the 12B and the
# 26B with int8 x and dp4a (program._Q4KQ), as Q4_I8 for the E4B. Unset: on
# for a dense model (the 12B: a verify group of 3 tokens 35.4 -> 24.9 ms, KL
# 0.00012 -> 0.0004), off for the 26B (its activations have larger outliers:
# KL 0.0001 -> 0.0009; MTP_PLAN.md). 1 or 0 sets it for both.
Q4_I8_DENSE = os.environ.get("RECLANK_GPU_Q4_I8_DENSE")


def q4_i8_dense(model):
    """Return True when the Q4_0 matrices of the GPU step and small groups of
    model take int8 x (Q4_I8_DENSE)."""
    if Q4_I8_DENSE is not None:
        return Q4_I8_DENSE == "1"
    return not model.cfg.enable_moe_block


# The groups of up to this many tokens start the head right after the step
# (E4BGPU._head_ahead); 0 turns it off.
HEAD_AHEAD = int(os.environ.get("RECLANK_GPU_HEAD_AHEAD", "8"))


def compile_e4b_group(model, t, arena=None):
    """Compile a step of t tokens of the E4B model for the GPU. A small group
    (at most MT_CPU tokens, an MTP verify group) has the form of the decode
    step (fused 2, no reuse of the buffers), so each token gets the bits of a
    decode step and MTP gives the tokens of the plain decode. A larger group
    (a part of a prompt) reuses the buffers of the layers (PoolCompiler)
    unless the temporaries are packed (alloc.pack, into arena: see
    compile_split_group)."""
    cfg = model.cfg
    small = t <= MT_CPU
    c = E4BGroupCompiler(model, pool=not small and not alloc.PACK)
    c.q4_kq = small and Q4_I8
    c.env["x"] = np.zeros((t, cfg.hidden_size), dtype=np.float32)
    c.env["tok"] = np.zeros((t, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input),
                            dtype=np.float32)
    c.p.slot("pos")
    c.compile(forms.e4b_step_form(model, fused=2 if small else True))
    c.p.tokens = t
    prog = _fuse_kq(c.p.finish())
    alloc.pack(prog, arena=arena)   # the temporaries by liveness (ir/alloc.py)
    return prog


class E4BGPU(ProgramLRU):
    """The decode step, the groups of tokens, and the prompt pass of the
    E4B model on the GPU. See the module text."""

    def __init__(self, model, graph=True):
        self.model = model
        self.chunk = chunk_for(mem_info()[0])
        progs = model.__dict__.setdefault("_programs", {})
        # The GPU step has the fused operations; the CPU step (key 1) has not.
        prog = progs.get("gpu1")
        if prog is None:
            prog = progs["gpu1"] = _fuse_kq(forms.compile_e4b_step(model, 1, fused=2, q4_kq=Q4_I8))
            alloc.pack(prog)
        self.prog = prog
        # the arenas of the temporaries of the programs (alloc.Arenas: the
        # step and the small groups, the large groups)
        self.arenas = alloc.Arenas(MT_CPU)
        self.arenas.keep(1, prog)
        self.g = GPUProgram(prog, graph=graph)
        self.graph = graph
        self.cache = GPUCache()
        self.head = None
        self.groups = {}      # t -> (Program, GPUProgram)
        self._lru_init()
        self.cache.oom = self._evict
        self.last = self.g.mirror.buffer_of(prog.names["xn"]).ptr
        self.rows = 1
        self._mm_freeze()       # GpuMem: the weights and the step stay
        self._run_blocks = self.cache.blocks

    def attach(self, cache):
        self.cache.attach(cache)

    def detach(self, cache):
        self.cache.to_host()
        self.cache.release()

    def step(self, tokens, pos, cache):
        """Run a step of one token. Return the hidden state after the final
        norm, shape (1, hidden)."""
        model, cfg = self.model, self.model.cfg
        ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
        assert ids.size == 1, "the GPU runs a step of one token"
        self.cache.reserve(cache, pos + 1)
        self.cache.share_tail(pos + 1)          # (GpuMem: the rows past the run are lent)
        kw = forms.e4b_step_params(self.prog, model, cache, pos, self._rope)
        names = self.prog.names
        names["x"][:] = model.embed_rows(forms.E4B_PREFIX + "embed_tokens", ids) * cfg.embed_scale
        tok = model.embed_rows(forms.E4B_PREFIX + "embed_tokens_per_layer", ids)
        names["tok"][:] = (tok * cfg.per_layer_embed_scale).reshape(1, -1)
        self.g.upload("x")
        self.g.upload("tok")
        self.g.bind(kw, self.cache)
        self.g.run()
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.rows = 1
        self._head_ahead(1)
        self.g.download("xn")
        return names["xn"].copy()

    def _group(self, t):
        e = self.groups.get(t)
        if e is None:
            prog = self.arenas.keep(t, compile_e4b_group(self.model, t,
                                                         arena=self.arenas.arena_for(t)))
            # without the tensor cores (Pascal) the int4 products of a large
            # group take int8 x on dp4a (k_gemm_dp4a), as those of the 12B
            e = self.groups[t] = (prog, self._build_program(prog, t, i8=0 if has_tc() else 1))
        self._used(t)
        return e

    def group(self, tokens, pos, cache, size=None, media=None):
        """Run a group of tokens from position pos. Return the hidden states
        after the final norm, shape (len(tokens), hidden). size pads the group
        to a program of that many tokens. The padding rows write cache rows
        after the group, and a later step writes them again.

        media is a list of media.Span (absolute positions): their rows take
        the place of the token rows, and the per-layer token rows of those
        positions are those of the pad token (id 0), as in transformers."""
        model, cfg = self.model, self.model.cfg
        ids = np.asarray(tokens, dtype=np.int64).reshape(-1)
        t = ids.size
        size = size or t
        prog, g = self._group(size)
        if self._ensure(cache):
            self.cache.attach(cache)
        self.cache.reserve(cache, pos + size)
        self.cache.share_tail(pos + size)       # (GpuMem: the rows past the run are lent)
        kw = forms.e4b_step_params(prog, model, cache, pos, self._rope)
        cache.n = pos + t
        names = prog.names
        names["x"][:t] = model.embed_rows(forms.E4B_PREFIX + "embed_tokens", ids) * cfg.embed_scale
        names["x"][t:] = 0.0
        tok_ids = ids
        for sp in media or ():
            lo, hi = max(sp.start, pos), min(sp.end, pos + t)
            if lo < hi:
                names["x"][lo - pos:hi - pos] = sp.rows[lo - sp.start:hi - sp.start]
                if tok_ids is ids:
                    tok_ids = ids.copy()
                tok_ids[lo - pos:hi - pos] = 0
        tok = model.embed_rows(forms.E4B_PREFIX + "embed_tokens_per_layer", tok_ids)
        names["tok"][:t] = (tok * cfg.per_layer_embed_scale).reshape(t, -1)
        names["tok"][t:] = 0.0
        g.upload("x")
        g.upload("tok")
        g.bind(kw, self.cache)
        g.run()
        self.last = g.mirror.buffer_of(names["xn"]).ptr + (t - 1) * cfg.hidden_size * 4
        self.rows = t
        if size == t and t <= HEAD_AHEAD:
            self._head_ahead(t)       # an MTP verify group reads every row
        g.download("xn")
        return names["xn"][:t].copy()

    def _ensure(self, cache):
        """Make the buffers of every layer that stores a key, as
        E4BCache.append does. Return True when a buffer is new."""
        new = False
        for plan in self.model.cfg.plan:
            if plan.shared or plan.idx in cache.kv:
                continue
            shape = (cache.cap, plan.num_kv_heads, plan.head_dim)      # position-major
            store = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
            cache.kv[plan.idx] = store
            if plan.stores:
                cache.shared[plan.kind] = store
            new = True
        return new

    def prefill(self, ids, pos, cache, media=None):
        """Run a prompt from position pos in chunks of self.chunk tokens. A group
        of up to MT_CPU tokens runs as it is. A longer chunk goes to a
        program of the next power of two. Return the hidden states of every
        token. media: see group (the E4B is causal, so a chunk can split an
        image)."""
        ids = list(ids)
        out = []
        for c0 in range(0, len(ids), self.chunk):
            chunk = ids[c0:c0 + self.chunk]
            size = None
            if len(chunk) > MT_CPU:
                size = min(self.chunk, 1 << (len(chunk) - 1).bit_length())
            spans = [sp for sp in media or () if sp.start < pos + c0 + len(chunk)
                     and sp.end > pos + c0]
            if spans:
                out.append(self.group(chunk, pos + c0, cache, size, media=spans))
            else:
                out.append(self.group(chunk, pos + c0, cache, size))
        return np.concatenate(out)

    def logits(self, rows=1):
        """Return the logits of the last rows of the last step or group, with
        the soft cap, shape (rows, vocabulary). The first call copies the
        head to the GPU."""
        model = self.model
        if self.head is None and model._gpu_head() is None:
            return self._kq_logits(rows)
        vocab = self._run_head(rows)
        _check(lib().rc_gpu_d2h(self.host_logits.ctypes.data, self.out.ptr, rows * vocab * 4))
        return self.host_logits[:rows].copy()

    def topk(self, rows, k, temperature):
        """Return the candidates of sampling of the last rows (_topk_rows), or
        None for a head that is not Q6_K or Q4_0 (GPU._kq_head)."""
        if self.head is None and self.model._gpu_head() is None:
            return None
        self._run_head(rows)
        return _topk_rows(self, rows, k, temperature)

    def _run_head(self, rows):
        """Run the head on the last rows into self.out. The first call copies
        the head to the GPU. Return the vocabulary size."""
        model, cfg = self.model, self.model.cfg
        if self.head is None:
            kind, w = model._gpu_head()
            w = np.ascontiguousarray(w)
            self.head_fn = lib().rc_gpu_q6k_head if kind == "q6k" else lib().rc_gpu_q4_head
            vocab = w.shape[0]
            self.head = Buffer(w.nbytes, "weights")
            self.head.upload(w)
            self.out = Buffer(4 * vocab * MT_CPU, "weights")
            self.host_logits = pinned((MT_CPU, vocab))     # a faster copy from the GPU
        assert 1 <= rows <= min(self.rows, MT_CPU)
        vocab = self.host_logits.shape[1]
        cap = float(cfg.final_logit_softcapping or 0.0)
        first = self.last - (rows - 1) * cfg.hidden_size * 4
        _check(self.head_fn(self.head.ptr, first, self.out.ptr, vocab, cfg.hidden_size,
                            cap, rows))
        return vocab

    def _rope(self, kind, pos, t):
        """The GPU addresses of the cosine and the sine rows of positions pos
        to pos + t - 1 for the layers of a kind ("s", "f"): tables of all the
        positions up to a limit, made once (rope.cos_sin, the same bits as a
        table of those positions). A step then copies no table."""
        tabs = self.__dict__.setdefault("rope_tabs", {})
        e = tabs.get(kind)
        if e is None or pos + t > e[0]:
            if e is not None:
                e[1].free()
                e[2].free()
            n = max(4096, 2 * (e[0] if e else 0), pos + t)
            plan = next(p for p in self.model.cfg.plan if p.is_sliding == (kind == "s"))
            from reclank.models.gemma4 import rope as rope_mod
            cos, sin = rope_mod.cos_sin(self.model.cfg.rope_inv_freq(plan), np.arange(n))
            bc, bs = Buffer(cos.nbytes), Buffer(sin.nbytes)
            bc.upload(np.ascontiguousarray(cos))
            bs.upload(np.ascontiguousarray(sin))
            e = tabs[kind] = (n, bc, bs, cos.shape[1] * 4)
        return e[1].ptr + pos * e[3], e[2].ptr + pos * e[3]

    def _head_ahead(self, rows):
        """Start the head of the last rows rows now, before the copy of the
        hidden state to the host, so the GPU runs it right after the step.
        logits() and argmax() then only copy the result. A GGUF head that is
        not Q6_K or Q4_0 only (_kq_head)."""
        self.head_done = None
        model = self.model
        if HEAD_AHEAD and self.head is None and model._gpu_head() is None:
            self._kq_head(rows)

    def _kq_head(self, rows):
        """Run the head of a GGUF file whose token table is not Q6_K (the UD
        files of the E2B and the E4B keep it in Q5_K) on the last rows rows:
        a program of one GP_KQ_LINEAR, the soft cap (GP_SOFTCAP), and the
        best token of each row (GP_ARGMAX, the first index of the largest
        value, as np.argmax). Return the Program and the GPUProgram; the
        logits are "out" and the tokens "tok"."""
        model, cfg = self.model, self.model.cfg
        progs = self.__dict__.setdefault("kq_heads", {})
        e = progs.get(rows)
        if e is None:
            k = model.kq(model.head)
            if k is None:
                raise RuntimeError("the GPU head needs a Q6_K or a K quant head")
            hp = P.Program()
            hx = np.zeros((rows, cfg.hidden_size), np.float32)
            out = np.zeros((rows, k.rows), np.float32)
            tok = np.zeros(rows, np.int32)
            hp.names.update(x=hx, out=out, tok=tok)
            hp.emit(P.KQ_LINEAR, None, None, None, hx, k.data, k.type, k.rows, k.cols, rows, out)
            cap = cfg.final_logit_softcapping
            if cap:
                hp.emit(P.SOFTCAP, out, out.size, float(cap))
            for j in range(rows):
                hp.emit(P.ARGMAX, out[j], k.rows, tok[j:j + 1])
            hp = hp.finish()
            e = progs[rows] = (hp, GPUProgram(hp, graph=self.graph, mirror=self.g.mirror))
        hp, hg = e
        assert 1 <= rows <= self.rows
        if getattr(self, "head_done", None) == (self.last, rows):
            return hp, hg             # _head_ahead ran it
        step = cfg.hidden_size * 4
        _check(lib().rc_gpu_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr,
                            self.last - (rows - 1) * step, rows * step))
        hg.run()
        self.head_done = (self.last, rows)
        return hp, hg

    def _kq_logits(self, rows):
        """The logits of _kq_head, with the soft cap of the GPU, through a
        pinned buffer (a faster copy)."""
        hp, hg = self._kq_head(rows)
        out = hp.names["out"]
        pin = self.__dict__.setdefault("kq_pinned", {})
        if rows not in pin:
            pin[rows] = pinned(out.shape)
        hg.mirror.buffer_of(out).download(pin[rows])
        return pin[rows].copy()

    def argmax(self, rows=1):
        """Return the best token of each of the last rows of the last step or
        group: the tokens of np.argmax on logits(rows), with a copy of rows
        integers from the GPU and not of the logits."""
        model = self.model
        if self.head is None and model._gpu_head() is None:
            hp, hg = self._kq_head(rows)
            hg.download("tok")
            return [int(v) for v in hp.names["tok"]]
        self._run_head(rows)
        return _argmax_rows(self, rows)


def _arrays(vals):
    out = []
    for v in vals:
        if isinstance(v, np.ndarray):
            out.append(v)
        elif isinstance(v, tuple):
            out += _arrays(v)
    return out


class SplitCompiler(PoolCompiler):
    """The compiler of a step of the 26B model for the GPU, with the experts
    on the CPU.

    The operation moe becomes a GP_TO_HOST record: the copy of its input, its
    weights, and its indices to pinned host memory. Its CPU program is the
    MOE record of the program of one part, on those pinned buffers. The first
    operation that reads the output of the experts gets two records before
    it: GP_CPU_JOIN, which runs the CPU program, and GP_TO_DEV, which copies
    the output to the GPU. The operations between the two parts run on the
    GPU while the CPU computes the experts.
    """

    def __init__(self, model, hot=None, kv="int16", pool=False, stage=None):
        super().__init__(model, pool)
        self.kv = kv
        # The two device buffers for the weights of the experts of a layer
        # (see moe_group_gpu), or None.
        self.stage = stage
        # id of an output of the experts -> (CPU program, host output, event,
        # device buffer of the CPU part, device buffer of the GPU part).
        self.pending = {}
        self.n_events = 0
        self.cpu_progs = []
        self.hot = hot or {}   # layer -> the experts that the GPU holds
        # For a step program: layer -> (hot experts, gate and up blocks, down
        # blocks, slots). slots gives the slot of each expert in the blocks,
        # or -1; HotCache changes it. For a group program: (slots, device
        # address of the gate and up blocks, of the down blocks), see
        # moe_group_gpu.
        self.hot_stores = {}
        # layer -> the entry of hot_stores of the step program, for a small
        # group (see moe_hot_group).
        self.hot_host = {}
        # layer -> the pinned array that GP_HOT_SPLIT fills: the cold
        # experts, their count, and the selection of the step (HotCache).
        self.hot_ip = {}
        # layer -> (tgu, tdn, ranges) of a large group (see tables).
        self.tables_of = {}
        # (layers, experts) int32, or None: a group counts the selections of
        # its routers there (GP_COUNT, HotCache.seed).
        self.counts = None
        # The mixed form of a large group (see moe_group_mix): layer -> the
        # array that marks the experts on the GPU (hot or copied), and the
        # pinned buffers and the scratch that the CPU parts of all the
        # layers share (one CPU part runs at a time).
        self.mix = False
        self.gslots_of = {}
        self.mix_bufs = None
        # Chunked staging (moe_group_chunks): the pieces of each layer, and
        # layer -> (tgu, tdn, ranges, gslots, fetch) of each piece.
        self.chunks = 1
        self.chunks_of = {}
        # The flag form of the handoff to the CPU (a step): no boundary
        # record, so the step is one graph (to_cpu, join).
        self.flags = False
        self.fused = False      # the fused forms of the step (add_norm2, ffn_out)
        self.flag_count = {}    # id of the flags -> the device int of the count
        # every expert on the GPU (ModelGPU resident): the experts have no CPU part
        self.resident = False
        # the hot experts in the pool of HotCache (ModelGPU POOL_HOT): every
        # layer with experts has a slot table, with no store of its own;
        # warm_pool: the ExpertPool (a warm expert of a large group has the
        # address of its block)
        self.pooled = False
        self.warm_pool = None

    def kernel(self, head, vals, out=None):
        for a in _arrays(vals):
            if id(a) in self.pending:
                self.join(a)
        if head == "router" and self.counts is not None and vals[0].shape[0] > 1:
            val, idx = super().kernel(head, vals, out)
            layer = vals[1]
            self.p.emit(P.COUNT, idx, self.counts[layer], idx.shape[1], self.p.slot("nreal"),
                        self.counts.shape[1])
            return val, idx
        if head == "moe":
            assert out is None
            return self.moe(*vals)
        if head == "ffn_out" and self.fused:
            m, w1, e, w2, w, x, s, wn = vals
            out2 = self.buffer(x.shape) if wn is not None else None
            self.p.emit(P.FFN_OUT, m, w1, e, w2, w, x, x, x.shape[0], x.shape[1],
                        float(self.eps), float(np.asarray(s).reshape(-1)[0]), wn, out2)
            return out2 if out2 is not None else x
        if head == "kv_write" and self.kv in ("int16", "int8", "k16v8"):
            return self.kv_write16(*vals)
        if head in ("attn_rows_qc", "attn_rows_q8", "attn_rows_qv") and vals[1].shape[0] <= MT_CPU:
            return self.attn_rows_small(*vals)
        if head in ("attn_qc", "attn_q8", "attn_qv") and self.fdts(vals[0]):
            # a step over a window: its rows from row 0 (k_attn_fdts, as a group)
            layer, q, lo, n = vals
            return P.k_attn_qc(self, layer, q, 0, self.value(("+", lo, n)),
                               q8={"attn_q8": True, "attn_qv": "v"}.get(head, False),
                               window=self.cfg.plan[layer].sliding_window)
        return super().kernel(head, vals, out)

    def router_slots(self, layer):
        """The slot of each expert of a layer on the GPU, -1 for a cold one.
        The router of a group reads it for the test of gg_set_reuse."""
        if layer in self.hot_host:
            return self.hot_host[layer][3]
        n = self.cfg.num_experts
        slots = np.full(n, -1, dtype=np.int32)
        hot = self.hot.get(layer, [])
        slots[hot] = np.arange(len(hot), dtype=np.int32)
        self.__dict__.setdefault("_router_slots", []).append(slots)
        return slots

    def fdts(self, layer):
        """A sliding layer of the 26B (head_dim 256, 2 query heads a key head)
        takes the record of a window (k_attn_fdts) in a step and in a group."""
        plan = self.cfg.plan[layer]
        return (FDT_TS and has_tc() and plan.is_sliding and bool(plan.sliding_window) and plan.head_dim == 256 and
                plan.num_q_heads == 2 * plan.num_kv_heads)

    def attn_rows_small(self, layer, q):
        """The attention of a small group, one query at a time with the
        record of a decode step (GP_ATTN_QC). Its kernel splits the keys of
        each head into chunks, and it is faster for a few queries than the
        group kernel. Query j has the position pos + j. A global layer of
        the 26B (head_dim 512, 8 query heads a key head) takes the group of
        up to FDT_TMAX queries in one record (operand 11: t; k_attn_fdtc
        reads the rows once, with the numbers of a step for each query), and
        a sliding layer (fdts) too, with its window (k_attn_fdts)."""
        plan = self.cfg.plan[layer]
        hd, qh = plan.head_dim, plan.num_q_heads
        t = q.shape[0]
        base = "base.%d" % layer
        q8 = {"int8": True, "k16v8": "v"}.get(self.kv, False)
        if self.fdts(layer) or (
                FDT_TC and has_tc() and not plan.is_sliding and hd == 512 and qh == 8 * plan.num_kv_heads):
            # records of up to FDT_TMAX queries: query j0 + i of a record
            # sees n + i rows from its first, as a step at its position
            window = plan.sliding_window if self.fdts(layer) else 0
            if t <= FDT_TMAX:
                n_v = self.value(("-", ("+", "pos", 1), base))
                return P.k_attn_qc(self, layer, q, 0, n_v, q8=q8, t=t, window=window)
            out = self.buffer((t, qh * hd))
            for j0 in range(0, t, FDT_TMAX):
                tc = min(FDT_TMAX, t - j0)
                n_v = self.value(("-", ("+", ("+", "pos", j0), 1), base))
                a = P.k_attn_qc(self, layer, q[j0:j0 + tc], 0, n_v, q8=q8, t=tc, window=window)
                self.p.emit(P.COPY, a, out[j0:j0 + tc], a.nbytes)
            return out
        out = self.buffer((t, qh * hd))
        for j in range(t):
            p = ("+", "pos", j)
            if plan.is_sliding:
                lo = ("max", 0, ("-", p, plan.sliding_window - 1, base))
            else:
                lo = 0
            lo_v = self.value(lo)
            n_v = self.value(("-", ("+", p, 1), base, lo_v))
            a = P.k_attn_qc(self, layer, q[j:j + 1], lo_v, n_v, q8=q8)
            self.p.emit(P.COPY, a, out[j:j + 1], a.nbytes)
        return out

    def kv_write16(self, layer, k, v, row, qc=1):
        """As k_kv_write, for a cache with the int16 copy only (or the int8
        cache: KV_WRITE8). The float addresses are null, so the kernel does
        not store float rows."""
        plan = self.cfg.plan[layer]
        per = plan.num_kv_heads * plan.head_dim
        kesz = 1 if self.kv == "int8" else 2
        vesz = 1 if self.kv in ("int8", "k16v8") else 2
        sl = lambda name: self.p.slot("%s.%d" % (name, layer))  # noqa: E731
        q = [P._addr(self, sl("kq"), row, kesz * per), P._addr(self, sl("ks"), row, per // 8),
             P._addr(self, sl("vq"), row, vesz * per), P._addr(self, sl("vs"), row, per // 8)]
        op = {"int8": P.KV_WRITE8, "k16v8": P.KV_WRITEV8}.get(self.kv, P.KV_WRITE)
        c = konly_layers(self.model, self.kv).get(layer)
        if c is not None:
            # keys only (ko_tab): the address of the keys for the values,
            # and 1 / c for their scales
            q[2] = q[0]
            self.p.emit(op, k, v, 0, 0, *q, k.size, float(np.float32(1.0 / c)))
            return
        self.p.emit(op, k, v, 0, 0, *q, k.size)

    def to_cpu(self, *copies, count=None):
        """Hand work to the CPU: copy each (device array, pinned array) to the
        host. Return the event (boundary form) or the flags (flag form: two
        int64 in pinned memory, in and out) for join. count (or None) is the
        device int of the length of the work of the CPU: with 0, the GPU does
        not wait for the CPU (GP_AWAIT)."""
        if self.flags:
            # GP_SIGNAL writes the data to the pinned arrays itself.
            ops = []
            for src, dst in copies:
                ops += [src, dst, dst.nbytes]
            ops += [0, 0, 0] * (3 - len(copies))
            fl = pinned((2,), np.int64)
            self.p.keep.append(fl)
            self.p.emit(P.SIGNAL, fl[0:1], self.p.slot("seq"), *ops)
            self.flag_count[id(fl)] = count
            return fl
        ev = self.n_events
        self.n_events += 1
        ops = []
        for src, dst in copies:
            ops += [src, dst, dst.nbytes]
        ops += [0, 0, 0] * (3 - len(copies))
        self.p.emit(P.TO_HOST, *ops, ev)
        return ev

    def join(self, a):
        cpu, host, ev, part, gpu_part, *started = self.pending.pop(id(a))
        if started and started[0]:
            self.p.emit(P.CPU_WAIT)         # a CPU_START runs it (moe_group_mix)
        elif isinstance(ev, np.ndarray):
            # The flag form: the runner runs the CPU program when the GPU
            # signals, and the GPU waits for its flag.
            seq = self.p.slot("seq")
            self.p.emit(P.CPU_TASK, cpu.buf, ev[0:1], ev[1:2], seq)
            count = self.flag_count.pop(id(ev), None)
            self.p.emit(P.AWAIT, ev[1:2], seq, host, part, part.nbytes,
                        count if count is not None else 0)
            if gpu_part is not None:
                self.p.emit(P.ADD, gpu_part, part, a, a.size)
            return
        else:
            self.p.emit(P.CPU_JOIN, cpu.buf, ev)
        self.p.emit(P.TO_DEV, host, part, part.nbytes)
        if gpu_part is not None:
            self.p.emit(P.ADD, gpu_part, part, a, a.size)

    def moe(self, h, val, idx, layer):
        if h.shape[0] > 1 and self.stage is not None and self.mix:
            return self.moe_group_mix(h, val, idx, layer)
        if h.shape[0] > 1 and self.stage is not None and self.chunks > 1:
            return self.moe_group_chunks(h, val, idx, layer)
        if h.shape[0] > 1 and self.stage is not None:
            return self.moe_group_gpu(h, val, idx, layer)
        if h.shape[0] > 1 and layer in self.hot_host:
            return self.moe_hot_group(h, val, idx, layer)
        if h.shape[0] > 1:
            return self.moe_group_cpu(h, val, idx, layer)
        if self.hot.get(layer) or (self.pooled and self.cfg.enable_moe_block and not self.resident):
            return self.moe_hot(h, val, idx, layer, self.hot.get(layer, []))
        hp, vp, ip = pinned(h.shape), pinned(val.shape), pinned(idx.shape, np.int32)
        ev = self.to_cpu((h, hp), (val, vp), (idx, ip))
        cc = P.Compiler(self.model)
        host_out = P.k_moe(cc, hp, vp, ip, layer)
        if self.flags:
            # a copy node of a graph reads pinned memory only
            po = pinned(host_out.shape)
            cc.p.emit(P.COPY, host_out, po, po.nbytes)
            host_out = po
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        dev_out = self.buffer(h.shape)
        self.pending[id(dev_out)] = (cpu, host_out, ev, dev_out, None)
        return dev_out

    def tables(self, layer):
        """Return the tables of the device addresses of the experts of a
        layer (gate and up, down), and the list of the copies of the cold
        experts: (host address, device address, bytes) for each run of
        adjacent cold experts. The arrays are made one time for each layer;
        fill_tables fills them again when the hot experts change."""
        if layer not in self.tables_of:
            n = self.model._layers[layer]["experts.gate_up_proj"][0].shape[0]
            tgu = np.zeros(n, dtype=np.int64)
            tdn = np.zeros(n, dtype=np.int64)
            ranges = np.zeros((2 * n, 3), dtype=np.int64)
            self.tables_of[layer] = (tgu, tdn, ranges)
            self.gslots_of[layer] = np.full(n, -1, dtype=np.int32)
            # (a mixed group: no cold expert yet; plan_mix fills them before
            # each run, at most stage_room of them)
            fill_tables(self.model, layer, self.hot_stores.get(layer), self.stage, tgu, tdn,
                        ranges, self.gslots_of[layer], pre=set() if self.mix else None,
                        pool=self.warm_pool)
        return self.tables_of[layer]

    def fetch(self, layer):
        """Emit the copy of the weights of the cold experts of a layer to the
        device buffer layer % 2."""
        _tgu, _tdn, ranges = self.tables(layer)
        self.p.emit(P.FETCH, ranges, len(ranges), layer, layer % 2)

    def moe_group_gpu(self, h, val, idx, layer):
        """The experts of a large group on the GPU (GP_MOE_GPU). All the
        experts of the layer come to a device buffer with GP_FETCH. The copy
        of layer l + 2 starts when the experts of layer l are done with the
        buffer."""
        w = self.model._layers[layer]
        gu_q, dn_q = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
        inner = self.cfg.moe_intermediate_size
        t, top_k = idx.shape
        pairs = t * top_k
        hidden = h.shape[1]
        tgu, tdn, _ranges = self.tables(layer)
        i32 = lambda n: self.buffer((n,), np.int32)  # noqa: E731
        out = self.buffer(h.shape)
        self.p.emit(P.FETCH_WAIT, layer)
        q4x = 1 if expert_host(self.model, layer)[2] else 0
        self.p.emit(P.MOE_GPU, h, val, idx, t, top_k, q4x, 0, gu_q.shape[1], hidden,
                    dn_q.shape[1], inner, i32(256), i32(257), i32(256), i32(pairs), i32(pairs),
                    i32(1 + 2 * (pairs // 64 + 129)), self.buffer((pairs, gu_q.shape[1])),
                    self.buffer((pairs, inner)), self.buffer((pairs, dn_q.shape[1])), out,
                    tgu, tdn)
        self.p.emit(P.FETCH_DONE, layer % 2)
        if layer + 2 < self.cfg.num_hidden_layers:
            self.fetch(layer + 2)
        return out

    def chunk_tables(self, layer):
        """The tables of the pieces of a layer (chunked staging): a list of
        (tgu, tdn, ranges, gslots, k) for each piece, k its fetch (layer
        times the pieces, plus the piece). fill_chunks fills them, and
        again when the hot experts change."""
        if layer not in self.chunks_of:
            n = self.model._layers[layer]["experts.gate_up_proj"][0].shape[0]
            self.chunks_of[layer] = [
                (np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64),
                 np.zeros((2 * n, 3), dtype=np.int64), np.full(n, -1, dtype=np.int32),
                 layer * self.chunks + c) for c in range(self.chunks)]
            fill_chunks(self.model, layer, self.hot_stores.get(layer), self.stage,
                        self.chunks_of[layer], pool=self.warm_pool)
        return self.chunks_of[layer]

    def fetch_chunk(self, k):
        """Emit the copy of piece k % chunks of layer k // chunks to buffer
        k % 2 (fetch k)."""
        _tgu, _tdn, ranges, _gslots, _k = self.chunk_tables(k // self.chunks)[k % self.chunks]
        self.p.emit(P.FETCH, ranges, len(ranges), k, k % 2)

    def moe_group_chunks(self, h, val, idx, layer):
        """The experts of a large group on the GPU when the buffers hold
        fewer experts than a layer has (chunked staging): the cold experts
        of the layer come in pieces (chunk_tables), one buffer each in turn.
        For each piece GP_HOT_SPLIT_MT gives GP_MOE_GPU the pairs of its
        experts (-1 for the others; the hot experts go with piece 0), and
        the outputs of the pieces add up. The copy of piece k + 2 starts when
        piece k is done with its buffer, so each cold expert crosses the bus
        once for the group, as with buffers of all of them, in a fraction of
        their memory (the 26B on a card of 4096 MiB)."""
        w = self.model._layers[layer]
        gu_q, dn_q = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
        inner = self.cfg.moe_intermediate_size
        t, top_k = idx.shape
        pairs = t * top_k
        hidden = h.shape[1]
        i32 = lambda n: self.buffer((n,), np.int32)  # noqa: E731
        q4x = 1 if expert_host(self.model, layer)[2] else 0
        total = self.cfg.num_hidden_layers * self.chunks
        out, part = self.buffer(h.shape), self.buffer(h.shape)
        cold, cold_val, gidx = i32(pairs), self.buffer((pairs,)), i32(pairs)
        for c, (tgu, tdn, _ranges, gslots, k) in enumerate(self.chunk_tables(layer)):
            self.p.emit(P.HOT_SPLIT_MT, idx, val, gslots, cold, cold_val, pairs,
                        self.p.slot("nreal"), top_k, gidx)
            self.p.emit(P.FETCH_WAIT, k)
            dst = out if c == 0 else part
            self.p.emit(P.MOE_GPU, h, val, gidx, t, top_k, q4x, 0, gu_q.shape[1], hidden,
                        dn_q.shape[1], inner, i32(256), i32(257), i32(256), i32(pairs), i32(pairs),
                        i32(1 + 2 * (pairs // 64 + 129)), self.buffer((pairs, gu_q.shape[1])),
                        self.buffer((pairs, inner)), self.buffer((pairs, dn_q.shape[1])), dst,
                        tgu, tdn)
            self.p.emit(P.FETCH_DONE, k % 2)
            if k + 2 < total:
                self.fetch_chunk(k + 2)
            if c > 0:
                self.p.emit(P.ADD, out, part, out, out.size)
        return out

    def moe_group_mix(self, h, val, idx, layer):
        """The experts of a large group, split between the GPU and the CPU
        (a mixed group). GP_FETCH copies only some cold experts: the ones
        that ModelGPU.plan_mix predicts, from the routers of the group
        before. gslots marks the experts on the GPU (hot or copied).
        GP_HOT_SPLIT_MT gives the pairs of the other experts to the CPU, and
        the rest to GP_MOE_GPU (-1 for a pair of the CPU). The CPU computes
        its pairs (KQ_MOE of the KQ_Q4X experts, int8 activations, as the
        prompt of the CPU; without those copies, an AVX-512 CPU with no VNNI
        or the plain C build, MOE_MT of the int4 experts) while the GPU
        computes its pairs. The output is the sum of the two parts, at the
        first reader (join)."""
        w = self.model._layers[layer]
        gu_q, dn_q = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
        inner = self.cfg.moe_intermediate_size
        t, top_k = idx.shape
        pairs = t * top_k
        hidden = h.shape[1]
        E = gu_q.shape[0]
        tgu, tdn, _ranges = self.tables(layer)
        gslots = self.gslots_of[layer]
        i32 = lambda n: self.buffer((n,), np.int32)  # noqa: E731
        q4x = P.ops._Q4X_MOE.get(gu_q.ctypes.data)
        if self.mix_bufs is None:
            self.mix_bufs = dict(
                hp=pinned((t, hidden), free=True), vp=pinned((t, top_k), free=True),
                ip=pinned((t, top_k), np.int32, free=True), out=pinned((t, hidden), free=True))
            if q4x is not None:
                from reclank.backends.cpu import cops
                self.mix_bufs.update(
                    xq=np.zeros((t, hidden), np.int8), xs=np.zeros((t, hidden // 32), np.float32),
                    xm=np.zeros((t, hidden // 16), np.float32),
                    scratch=cops.kq_moe_scratch(t, top_k, E, hidden, inner))
        b = self.mix_bufs
        cold, cold_val, gidx = i32(pairs), self.buffer((pairs,)), i32(pairs)
        # The split and the copy to the host do not need the copies of the
        # experts, so the CPU part can start before they end.
        self.p.emit(P.HOT_SPLIT_MT, idx, val, gslots, cold, cold_val, pairs,
                    self.p.slot("nreal"), top_k, gidx)
        ev = self.n_events
        self.n_events += 1
        self.p.emit(P.TO_HOST, h, b["hp"], b["hp"].nbytes, cold_val, b["vp"], b["vp"].nbytes,
                    cold, b["ip"], b["ip"].nbytes, ev)
        # The CPU part, on the helper thread (GP_CPU_START).
        cc = P.Compiler(self.model)
        if q4x is not None:
            mats = P.ops.q4x_moe_mats(q4x)
            cc.p.keep.append(mats)
            cc.p.emit(P.KQ_QUANT, b["hp"], t, hidden, b["xq"], b["xs"], b["xm"], b["ip"], top_k)
            cc.p.emit(P.KQ_MOE, b["xq"], b["xs"], b["xm"], b["ip"], b["vp"], t, top_k, E, mats, None,
                      hidden, inner, b["scratch"], b["out"], None, 1, None)
        else:
            # the int4 experts (MOE_MT skips the pairs of the GPU: -1 in ip)
            host_out = P.k_moe(cc, b["hp"], b["vp"], b["ip"], layer)
            cc.p.emit(P.COPY, host_out, b["out"], b["out"].nbytes)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        self.p.emit(P.CPU_START, cpu.buf, ev)
        self.p.emit(P.FETCH_WAIT, layer)
        gpu_part = self.buffer(h.shape)
        self.p.emit(P.MOE_GPU, h, val, gidx, t, top_k, 1 if q4x is not None else 0, 0, gu_q.shape[1], hidden,
                    dn_q.shape[1], inner, i32(256), i32(257), i32(256), i32(pairs), i32(pairs),
                    i32(1 + 2 * (pairs // 64 + 129)), self.buffer((pairs, gu_q.shape[1])),
                    self.buffer((pairs, inner)), self.buffer((pairs, dn_q.shape[1])), gpu_part,
                    tgu, tdn)
        self.p.emit(P.FETCH_DONE, layer % 2)
        if layer + 2 < self.cfg.num_hidden_layers:
            self.fetch(layer + 2)
        out = self.buffer(h.shape)
        self.pending[id(out)] = (cpu, b["out"], ev, self.buffer(h.shape), gpu_part, True)
        return out

    def moe_hot_group(self, h, val, idx, layer):
        """The experts of a small group (at most MT_CPU tokens) when the GPU
        holds some of them. GP_HOT_SPLIT_MT marks each pair (token, slot):
        the GPU computes the pairs of the hot experts with GP_HOT_MOE, and
        the CPU computes the other pairs with GP_MOE_MT. The output is the
        sum of the two parts. The hot experts are the arrays of the step
        program, so the GPU holds them one time."""
        _hot, gu_store, dn_store, slots = self.hot_host[layer]
        w = self.model._layers[layer]
        dn_rows = w["experts.down_proj"][0].shape[1]
        inner = self.cfg.moe_intermediate_size
        t, top_k = idx.shape
        hidden = h.shape[1]
        pairs = t * top_k
        q4x = 1 if expert_host(self.model, layer)[2] else 0
        if self.resident:
            # every expert on the GPU (the resident mode): no CPU part
            out = self.buffer(h.shape)
            self.p.emit(P.HOT_MOE, h, val, idx, slots, gu_store, dn_store,
                        self.buffer((pairs, 2 * inner)), self.buffer((pairs, inner)),
                        self.buffer((pairs, hidden)), out, top_k, 2 * inner, hidden,
                        dn_rows, inner, t, q4x)
            return out
        cold = self.buffer((t, top_k), np.int32)
        cold_val = self.buffer((t, top_k))
        self.p.emit(P.HOT_SPLIT_MT, idx, val, slots, cold, cold_val, t * top_k)
        # (freed with the program: the pool's warm frees the groups after a prompt)
        hp, vp, ip = pinned(h.shape, free=True), pinned((t, top_k), free=True), pinned((t, top_k), np.int32, free=True)
        ev = self.n_events
        self.n_events += 1
        self.p.emit(P.TO_HOST, h, hp, h.nbytes, cold_val, vp, vp.nbytes, cold, ip, ip.nbytes, ev)
        gpu_part = self.buffer(h.shape)
        self.p.emit(P.HOT_MOE, h, val, idx, slots, gu_store, dn_store,
                    self.buffer((pairs, 2 * inner)), self.buffer((pairs, inner)),
                    self.buffer((pairs, hidden)), gpu_part, top_k, 2 * inner, hidden,
                    dn_rows, inner, t, q4x)
        cc = P.Compiler(self.model)
        # An expert device: the cold pairs of its experts to it (GP_NM_REMOTE_POST: a group), the others
        # to the CPU (its list in place of ip), and REMOTE_JOIN adds its rows to the CPU's.
        from reclank.backends.cuda.devices import expert_device
        xdev = expert_device(self.model)
        if xdev is not None:
            lcold, posted = np.zeros((t, top_k), dtype=np.int32), np.zeros(1, dtype=np.int64)
            cc.p.keep += [lcold, posted]
            cc.p.emit(P.NM_REMOTE_POST, hp, ip, vp, pairs, xdev.ctx_bytes, layer, lcold, posted, top_k)
            ip = lcold
        host_out = P.k_moe(cc, hp, vp, ip, layer)
        if xdev is not None:
            cc.p.emit(P.REMOTE_JOIN, xdev.ctx_bytes, posted, host_out)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        out = self.buffer(h.shape)
        self.pending[id(out)] = (cpu, host_out, ev, self.buffer(h.shape), gpu_part)
        return out

    def moe_group_cpu(self, h, val, idx, layer):
        """The experts of a group of tokens on the CPU: the MOE_MT record of
        the CPU interpreter on pinned copies of the input."""
        hp, vp, ip = pinned(h.shape, free=True), pinned(val.shape, free=True), pinned(idx.shape, np.int32, free=True)
        ev = self.n_events
        self.n_events += 1
        self.p.emit(P.TO_HOST, h, hp, h.nbytes, val, vp, val.nbytes, idx, ip, idx.nbytes, ev)
        cc = P.Compiler(self.model)
        host_out = P.k_moe(cc, hp, vp, ip, layer)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        dev_out = self.buffer(h.shape)
        self.pending[id(dev_out)] = (cpu, host_out, ev, dev_out, None)
        return dev_out

    def moe_hot(self, h, val, idx, layer, hot):
        """The experts of a layer when the GPU holds some of them.

        GP_HOT_SPLIT writes the selected experts that the GPU does not hold
        (the cold experts), with their weights and their count. GP_TO_HOST
        copies them and the input to the host. GP_HOT_MOE computes the hot
        experts on the GPU, while the CPU computes the cold experts with
        GP_MOE_N. The output is the sum of the two parts.
        """
        w = self.model._layers[layer]
        gu_q, gu_s = w["experts.gate_up_proj"]
        dn_q, dn_s = w["experts.down_proj"]
        inner = self.cfg.moe_intermediate_size
        top_k = idx.size
        hidden = h.shape[1]
        slots = np.full(gu_q.shape[0], -1, dtype=np.int32)
        hgu, hdn, q4x = expert_host(self.model, layer)
        if self.pooled:
            # every hot expert in a block of the pool (slot HotCache.WARM +
            # block): the stores are stubs
            hot = []
            gu_store, dn_store = np.zeros(16, np.uint8), np.zeros(16, np.uint8)
        else:
            slots[hot] = np.arange(len(hot), dtype=np.int32)
            gu_store = np.ascontiguousarray(hgu[hot])
            dn_store = np.ascontiguousarray(hdn[hot])
        self.hot_stores[layer] = (list(hot), gu_store, dn_store, slots)
        if not q4x:
            for store, scales in ((gu_q[hot], gu_s), (dn_q[hot], dn_s)):
                half = store[..., :2].copy().view(np.float16)[..., 0].astype(np.float32)
                if not np.array_equal(half, scales[hot]):
                    raise ValueError("an expert has float32 scales that are not its float16 "
                                     "scales")
        if self.resident:
            # every expert on the GPU (the resident mode): no CPU part
            out = self.buffer(h.shape)
            self.p.emit(P.HOT_MOE, h, val, idx, slots, gu_store, dn_store,
                        self.buffer((top_k, 2 * inner)), self.buffer((top_k, inner)),
                        self.buffer((top_k, hidden)), out, top_k, 2 * inner, hidden,
                        dn_q.shape[1], inner, 1, 1 if q4x else 0)
            return out
        # The cold experts, their count, and the selection of the step.
        cold = np.zeros(2 * top_k + 1, dtype=np.int32)
        cold_val = self.buffer(top_k)
        self.p.emit(P.HOT_SPLIT, idx, val, slots, cold, cold_val, top_k, 1)
        hp, vp, ip = pinned(h.shape), pinned((top_k,)), pinned((2 * top_k + 1,), np.int32)
        self.hot_ip[layer] = ip
        ev = self.to_cpu((h, hp), (cold_val, vp), (cold, ip), count=cold[top_k:top_k + 1])
        gpu_part = self.buffer(h.shape)
        self.p.emit(P.HOT_MOE, h, val, idx, slots, gu_store, dn_store,
                    self.buffer((top_k, 2 * inner)), self.buffer((top_k, inner)),
                    self.buffer((top_k, hidden)), gpu_part, top_k, 2 * inner, hidden,
                    dn_q.shape[1], inner, 1, 1 if q4x else 0)
        cc = P.Compiler(self.model)
        host_out = pinned(h.shape) if self.flags else np.zeros(h.shape, dtype=np.float32)
        # An expert device (backends/cuda/devices.py: a second GPU of this
        # process with experts this one lacks): the cold experts it holds go
        # to it, the others to the CPU below (its lists in place of ip and
        # vp), and REMOTE_JOIN adds its sum to the CPU's.
        from reclank.backends.cuda.devices import expert_device
        xdev = expert_device(self.model)
        if xdev is not None:
            lidx = np.zeros(2 * top_k + 1, dtype=np.int32)
            lval = np.zeros(top_k, dtype=np.float32)
            posted = np.zeros(1, dtype=np.int64)
            cc.p.emit(P.REMOTE_POST, hp, vp, ip, ip[top_k:top_k + 1], xdev.ctx_bytes, layer, hidden,
                      lidx, lval, lidx[top_k:top_k + 1], posted)
            ip, vp = lidx, lval
        q4x = P.ops._Q4X_MOE.get(gu_q.ctypes.data)
        if q4x is not None:
            # The cold experts in groups of 16 rows (KQ_Q4X), float32
            # activations: GP_KQ_MOE with the count in memory (cold_idx[top_k])
            # gives the products of GP_MOE_N to 4e-6, in less time.
            from reclank.backends.cpu import cops
            E = gu_q.shape[0]
            mats = P.ops.q4x_moe_mats(q4x)
            cc.p.keep.append(mats)
            cc.p.emit(P.KQ_MOE, np.zeros((1, hidden), np.int8), np.zeros((1, hidden // 32), np.float32),
                      np.zeros((1, hidden // 16), np.float32), ip, vp, 1, top_k, E, mats, None,
                      hidden, inner, cops.kq_moe_scratch(1, top_k, E, hidden, inner), host_out,
                      ip[top_k:], 3, hp)
        else:
            cc.p.emit(P.MOE_N, hp, vp, ip, ip[top_k:], gu_q, gu_s, dn_q, dn_s, gu_q.shape[1],
                      hidden, dn_q.shape[1], inner, np.zeros(top_k, dtype=np.int32),
                      np.zeros((top_k, 2 * inner), np.float32), np.zeros((top_k, inner), np.float32),
                      np.zeros((top_k, dn_q.shape[1]), np.float32), host_out)
        if xdev is not None:
            cc.p.emit(P.REMOTE_JOIN, xdev.ctx_bytes, posted, host_out)
        cpu = cc.p.finish()
        self.cpu_progs.append(cpu)
        out = self.buffer(h.shape)
        self.pending[id(out)] = (cpu, host_out, ev, self.buffer(h.shape), gpu_part)
        return out


def pick_hot(model, counts, budget):
    """Return the experts that the GPU holds, as a dict layer -> the sorted
    expert indices. counts has shape (layers, experts): the selections of each
    expert on a text (scripts/expert_use.py). The set takes the most used
    experts over all layers, up to budget bytes."""
    w = model._layers[0]
    per = (w["experts.gate_up_proj"][0][0].nbytes + w["experts.down_proj"][0][0].nbytes)
    n = int(budget // per)
    flat = np.asarray(counts).reshape(-1)
    order = np.argsort(flat, kind="stable")[::-1][:n]
    order = order[flat[order] > 0]
    e = counts.shape[1]
    hot = {}
    for k in order:
        hot.setdefault(int(k // e), []).append(int(k % e))
    return {layer: sorted(v) for layer, v in hot.items()}


def resident_hot(model):
    """Every expert of every layer (the resident mode), after a test that
    they fit in the free memory of the GPU with 3 GB left for the rest."""
    w = model._layers[0]
    n = w["experts.gate_up_proj"][0].shape[0]
    per = (w["experts.gate_up_proj"][0][0].nbytes + w["experts.down_proj"][0][0].nbytes)
    layers = model.cfg.num_hidden_layers
    need, free = per * n * layers, mem_info()[0]
    if need + 3.0e9 > free:
        raise MemoryError("--resident: the experts take %.1f GB and the GPU has %.1f GB free "
                          "(3 GB more are needed for the rest)" % (need / 1e9, free / 1e9))
    return {i: list(range(n)) for i in range(layers)}


# The counts of the experts of the 26B QAT model on 800 tokens of each of four
# texts. The texts are the README, Python code, C code, and notes in English
# (SPLIT_PLAN.md).
HOT_COUNTS = _HERE / "data" / "gemma-4-26B-expert-counts.npz"


def hot_counts(model):
    """Return the counts of the experts for the hot experts, or None."""
    path = os.environ.get("RECLANK_GPU_HOT")
    if path == "0":
        return None
    f = np.load(path or HOT_COUNTS)
    counts = sum(f[k] for k in f.files)
    shape = (model.cfg.num_hidden_layers, model.cfg.num_experts)
    if counts.shape != shape:
        if path:
            raise ValueError("the counts of %s have the shape %s, not %s"
                             % (path, counts.shape, shape))
        return None
    return counts


# The tokens of a chunk of a prompt pass on the GPU. Each chunk copies the
# weights of the cold experts to the GPU. For the 26B that is up to 11 GB,
# about 1.6 s over PCIe 3 x8 (6.7 GB/s). A long chunk pays for the copy with
# more tokens. With 2048 the copy is hidden behind the other work of the
# layers even with 2 GB of hot experts (the wait is 0.3% of a group); with
# 1024 the wait is 32% of a group at 2 GB. On a GPU with less than SMALL_GPU
# bytes free when the model loads the default is SMALL_CHUNK: its programs
# take half the memory, for the cache (the 26B with 4 GB: prompts of 96K
# tokens in place of 64K; 120.2 tok/s in place of 123.2 at 64K). An 8 GB
# card (8.3 GB free at the load) takes CHUNK: the 26B on the GTX 1070, a
# prompt of 8192 tokens, 1024 375 tok/s, 2048 412, 4096 431; the cache then
# still fits 160K tokens (engine.py: the default context of such a card).
# RECLANK_GPU_CHUNK sets it for every GPU.
_CHUNK_ENV = os.environ.get("RECLANK_GPU_CHUNK")
CHUNK = int(_CHUNK_ENV or "2048")
SMALL_GPU = 7e9
SMALL_CHUNK = 1024


def chunk_for(free):
    """The tokens of a chunk of a prompt for a GPU with free bytes free."""
    return CHUNK if _CHUNK_ENV or free >= SMALL_GPU else min(CHUNK, SMALL_CHUNK)

# The tokens of an image see each other in every layer on the GPU, as on the
# CPU (reclank.models.gemma4.model._BIDIR_ALL); RECLANK_BIDIR_ALL=0 is not on the GPU yet.
_BIDIR_ALL_OK = os.environ.get("RECLANK_BIDIR_ALL", "1") == "1"

# The record of a group of up to FDT_TMAX queries for a global layer of the
# 26B (attn_rows_small; FT_TMAX of backends/cuda/csrc; a larger group takes
# records of FDT_TMAX); RECLANK_GPU_FDT_TC=0 (the old kernel, one query a
# record) one record for each query.
FDT_TMAX = 4
FDT_TC = os.environ.get("RECLANK_GPU_FDT_TC", "1") != "0"

# the sliding layers (SplitCompiler.fdts, k_attn_fdts); RECLANK_GPU_FDT_TS=0
# the record of a step for each query, k_attn_fdt (a test)
FDT_TS = FDT_TC and os.environ.get("RECLANK_GPU_FDT_TS", "1") != "0"

# The shortest part of a prompt that runs with the experts on the GPU. The
# copy of their weights costs about 1.9 s, and the CPU takes about as long for
# 128 tokens.
PREFILL_MIN = int(os.environ.get("RECLANK_GPU_PREFILL_MIN", "128"))

# The mixed groups of a prompt of the 26B (SplitCompiler.moe_group_mix):
# each layer copies some of its cold experts (the most used in the group
# before), and the CPU computes the pairs of the other cold experts.
# RECLANK_GPU_MIX=0 turns the mixed groups off. RECLANK_GPU_MIX_PRE gives
# the count of copied experts of each layer (-1: all); the default (auto)
# takes the count of the least time in the model of ModelGPU.plan_mix.
MIX = os.environ.get("RECLANK_GPU_MIX", "1") != "0"   # (the Qwen models read a size)
# RECLANK_GPU_STAGE_MIN (8): the fewest cold experts a layer that the buffers
# of a mixed group hold (ModelGPU._stage) before the large groups go without
# them (all their cold experts on the CPU).
STAGE_MIN = int(os.environ.get("RECLANK_GPU_STAGE_MIN", "8"))
# RECLANK_GPU_CHUNK_MAX (8): the most pieces of a layer in chunked staging
# (SplitCompiler.moe_group_chunks: the cold experts of a layer in buffers
# that hold fewer, one piece after another), before the mixed groups take the
# rest to the CPU. 30 layers of 8 pieces stay under GG_FETCH_MAX (256).
CHUNK_MAX = int(os.environ.get("RECLANK_GPU_CHUNK_MAX", "8"))

# The handoff to the CPU of a decode step inside its graph (SplitCompiler.
# to_cpu and join, GP_SIGNAL and GP_AWAIT): the step is one graph, and the
# runner runs the CPU parts as the GPU signals them. 0 keeps the boundary
# records (GP_TO_HOST, GP_CPU_JOIN, GP_TO_DEV).
FLAGS = os.environ.get("RECLANK_GPU_FLAGS", "1") == "1"

# RECLANK_GPU_POOL_HOT (auto): the hot experts of the 26B in the pool of
# HotCache (WarmPool, the cache system of the Qwen models): every layer has
# a slot table and no store of its own; the first hot experts are blocks of
# the pool's "hot" segments, which a program, the cache or a prompt takes
# back when it needs the memory. 0 keeps a store of fixed slots in each
# layer (the hot budget for good); 1 always the pool.
POOL_HOT = os.environ.get("RECLANK_GPU_POOL_HOT", "auto")
# auto: the pool when the budget of the hot experts (RECLANK_GPU_HOT_GB, or
# the free memory less 6 GB) holds less than POOL_FIT of the experts; with
# more (the 26B on the 3090: 3766 of its 3840 experts) the fixed slots, which
# keep 2% of the decode the pool costs (tg128 110.5 against 107.9).
POOL_FIT = 0.9


def experts_bytes(model):
    """The bytes of all the experts of a model (0 for a dense model)."""
    if not model.cfg.enable_moe_block:
        return 0
    return sum(a.nbytes for i in range(model.cfg.num_hidden_layers)
               for a in expert_host(model, i)[:2])
# RECLANK_GPU_WARM (free): the warm experts of the 26B in more blocks of the
# pool (WarmPool.warm_mode): "free" the free memory at a decode step and a
# small group (that a prompt took and let go), and the rows the cache lends;
# "lend" those rows only; 0 none.
WARM = os.environ.get("RECLANK_GPU_WARM", "free")
# RECLANK_GPU_WARM_KEEP (0.3e9): the free bytes the warm segments leave at a
# decode step (into the reserve of GpuMem: they go first when a Buffer needs
# the memory). RECLANK_GPU_WARM_MIN (1e9): with less for the pool, the
# programs of the large groups (a prompt) and the buffers of the cold experts
# go first (the next prompt makes them again): the 26B on a card of 4 GB
# after a prompt of 64K tokens.
WARM_KEEP = float(os.environ.get("RECLANK_GPU_WARM_KEEP", "0.3e9"))
WARM_MIN = float(os.environ.get("RECLANK_GPU_WARM_MIN", "1e9"))
# RECLANK_GPU_WARM_SHORT (1): the programs of the large groups and the
# buffers of the cold experts go also at the first decode step after a
# prompt when the pool has less room than the experts the GPU does not hold
# (WarmPool.warm_short). The 26B on the GTX 1070 after a prompt of 8192
# tokens: 1424 experts in the pool against 1016, decode 36.4 tok/s against
# 34.7; the next prompt makes the programs again (a prompt of 8192 tokens
# took as long, one of 300 tokens 0.07 s more).
WARM_SHORT = os.environ.get("RECLANK_GPU_WARM_SHORT", "1") == "1"

# The fused forms of the step of the 26B (program.layer_form, fused): the
# norms and the adds of the end of the attention and of the end of each
# layer in two kernels (GP_ADD_NORM, GP_FFN_OUT) in place of about ten.
FUSED = os.environ.get("RECLANK_GPU_FUSED", "1") == "1"

# 1: a small group (an MTP verify group of the 26B) takes the fused forms
# too (compile_split_group). 0 keeps the separate norms, for a test.
GROUP_FUSED = os.environ.get("RECLANK_GPU_GROUP_FUSED", "1") == "1"

# The fused dense layer (forms.layer_def, fused) for the large groups of
# a dense model too (a prompt pass of the 12B): fewer kernels for the norms,
# the adds, and the int8 x of the products.
PREFILL_FUSED = os.environ.get("RECLANK_GPU_PREFILL_FUSED", "1") == "1"

# The rows of the tokens of a large group (a prompt pass) from the Q4_0 token
# table on the GPU (ModelGPU._embed_gpu), not from Model.embed and a copy of
# the rows: about 30 ms for each chunk of 2048 tokens of the 12B. With little
# free memory (a card of 10 GB: less than two tables above the reserve of
# GpuMem) a prompt takes its rows from the CPU and does not copy the table
# for them; the head of the decode copies it later, when the programs of the
# prompt can give their memory (the 12B on 10 GB: a prompt of 24K tokens
# failed on the copy of the table, 566 MB, in the middle of the prompt).
EMBED_GPU = os.environ.get("RECLANK_GPU_EMBED", "1") == "1"
MIX_PRE = os.environ.get("RECLANK_GPU_MIX_PRE", "auto")

# The model of the time of a layer of a mixed group, in ms: the other GPU
# work of the layer for each token (gpu_tok; with the copies of the CPU
# part to the host and back), the GPU experts for each pair (gpu_pair), the
# copy rate in bytes for each ms (rate), and the CPU part: cpu_a for each
# layer, cpu_e for each expert (the read of its weights), and cpu_b for each
# pair (KQ_Q4X, int8). The first values are those
# of an RTX 5060 Ti on PCIe 3 x8 with 18 threads; each mixed group measures
# them again (ModelGPU.calibrate_mix), so the plan follows the machine and
# its other load. RECLANK_GPU_MIX_CAL=0 keeps the first values.
MIX_START = dict(gpu_tok=0.007, gpu_pair=0.65e-3, rate=3.3e6, cpu_a=1.0, cpu_e=0.06,
                 cpu_b=4e-3)
MIX_CAL = os.environ.get("RECLANK_GPU_MIX_CAL", "1") == "1"
MIX_LOG = os.environ.get("RECLANK_GPU_MIX_LOG", "0") == "1"


def compile_split_group(model, t, hot=None, kv="int16", stage=None, hot_stores=None,
                        hot_host=None, counts=None, mix=False, resident=False, pool=None,
                        arena=None, chunks=1):
    """Compile a step of t tokens for the GPU: a chunk of a prompt, or the
    group of an MTP verify step. The form is the group form of the layers.

    A group of more than MT_CPU tokens computes the experts on the GPU, with
    the weights in the two device buffers of stage. A smaller group sends
    them to the CPU, as a step does."""
    assert kv in ("int16", "int8", "k16v8"), "a group on the GPU reads a quantized cache"
    gpu_experts = model.cfg.enable_moe_block and t > MT_CPU and stage is not None
    # A small group (an MTP verify group) takes the fused forms of the step
    # (add_norm2, ffn_out; GP_ADD_NORM and GP_FFN_OUT take rows), so it runs
    # the norms of the step and fewer kernels. It keeps its buffers, as the
    # step does: the fused layer reads the norm "hn" of the layer before.
    small = t <= MT_CPU and not gpu_experts and FUSED and GROUP_FUSED
    big_dense = (t > MT_CPU and not model.cfg.enable_moe_block and FUSED and GROUP_FUSED
                 and PREFILL_FUSED)
    # (a buffer of each layer of its own when the temporaries are packed:
    # the buffers that every layer shares live through the whole program)
    c = SplitCompiler(model, hot, kv, pool=not small and not alloc.PACK,
                      stage=stage if gpu_experts else None)
    c.fused = small or big_dense
    # int8 x for the dense Q4_0 matrices of a small group, as the step (its
    # own switch, not that of the fused forms).
    c.q4_kq = t <= MT_CPU and not gpu_experts and q4_i8_dense(model)
    c.hot_stores = hot_stores or {}
    c.hot_host = hot_host or {}
    c.counts = counts
    c.resident = resident
    c.warm_pool = pool
    c.mix = mix and gpu_experts
    c.chunks = chunks if gpu_experts and not c.mix else 1
    c.env["x"] = np.zeros((t, model.cfg.hidden_size), dtype=np.float32)
    # The last key of each query less pos (the attention records): row j sees
    # up to its position unless it is in an image (ModelGPU.group). A group of
    # MT_CPU tokens or fewer (an MTP verify group) runs the attention of a
    # step for each query, with no such record, and a prompt puts each image
    # in a larger group (ModelGPU._prefill).
    if t > MT_CPU:
        c.env["lim"] = np.arange(t, dtype=np.int32)
    c.p.slot("pos")
    if gpu_experts and c.chunks > 1:
        c.fetch_chunk(0)
        c.fetch_chunk(1)
    elif gpu_experts:
        c.fetch(0)
        c.fetch(1)
    mode = {"int8": "q8", "k16v8": "qv"}.get(kv, "qc")
    c.compile(forms.step_form(model, mode, t, fused=c.fused))
    assert not c.pending, "an output of the experts has no reader"
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = mode
    c.p.tokens = t
    c.p.cpu_progs = c.cpu_progs
    c.p.tables_of = c.tables_of
    c.p.gslots_of = c.gslots_of
    c.p.chunks_of = c.chunks_of
    c.p.mix = c.mix
    prog = _fuse_kq(c.p.finish()) if c.q4_kq else c.p.finish()
    alloc.pack(prog, arena=arena)   # the temporaries by liveness (ir/alloc.py)
    return prog


def compile_split_step(model, hot=None, kv="int16", resident=False, pooled=False, arena=None):
    """Compile a step of one token of the 26B model (or of a dense model) for
    the GPU. "x" is the input and "xn" the result. hot gives the experts that
    the GPU holds (see pick_hot). kv is the form of the cache on the GPU:
    "int16" (the form of the CPU program, a scale for each group of 32
    values) or "float". pooled: the hot experts in the pool of HotCache
    (POOL_HOT): a slot table in each layer with experts, no stores."""
    c = SplitCompiler(model, hot, kv)
    c.resident = resident
    c.pooled = pooled
    c.flags = FLAGS
    c.fused = FUSED
    c.q4_kq = q4_i8_dense(model)
    c.env["x"] = np.zeros((1, model.cfg.hidden_size), dtype=np.float32)
    c.p.slot("pos")
    mode = {"int16": "qc", "int8": "q8", "k16v8": "qv"}.get(kv, "f32")
    c.compile(forms.step_form(model, mode, 1, fused=FUSED))
    assert not c.pending, "an output of the experts has no reader"
    c.p.layers = list(range(model.cfg.num_hidden_layers))
    c.p.attn = mode
    c.p.tokens = 1
    c.p.cpu_progs = c.cpu_progs
    c.p.hot_stores = c.hot_stores
    c.p.hot_ip = c.hot_ip
    prog = _fuse_kq(c.p.finish()) if c.q4_kq else c.p.finish()
    alloc.pack(prog, arena=arena)   # the temporaries by liveness (ir/alloc.py)
    return prog


# RECLANK_GPU_KONLY: the global layers (k_eq_v) keep only their int16 keys
# on the GPU, and for the values only the scales vs = ks / c (ko_tab of
# kernels/operands.cu): K = c rope(V), with c the k_norm weight (one value
# for every element in the 12B and the 26B), so V = unrope(K) / c and the
# kernels turn back the pairs that turn (64 of the 256 of a head of 512).
# A row of a global layer takes 1152 bytes for each key head in place of
# 2176. The attention output against float K and V (8192 tokens of the
# 12B, the last 128 queries) has the error of the int16 cache: 1.07e-5 to
# 2.93e-5 relative in each global layer, the same to 3 digits.
# 0: the keys and the values (a test).
KONLY = os.environ.get("RECLANK_GPU_KONLY", "1") == "1"


def konly_layers(model, kv):
    """Return layer -> c of the layers whose GPU cache keeps only keys (see
    KONLY): the global layers whose k_norm weight is one value, with the
    int16 cache. RECLANK_GPU_FLASH=1 (k_flash_tc, a test) does not take
    them."""
    if not KONLY or kv != "int16" or os.environ.get("RECLANK_GPU_FLASH") == "1":
        return {}
    got = model.__dict__.setdefault("_konly_c", {})
    if kv not in got:
        out = {}
        for p in model.cfg.plan:
            if not p.k_eq_v:
                continue
            w = np.asarray(model.load_layer(p.idx)["self_attn.k_norm"], dtype=np.float32).reshape(-1)
            if w.size == p.head_dim and w[0] != 0 and np.all(w == w[0]):
                out[p.idx] = float(w[0])
        got[kv] = out
    return got[kv]


_KO = {}


def ko_table(cfg):
    """Copy the cos and the sin of the pairs that turn in the global layers
    to the GPU (ko_tab: two tables, of position >> 8 and of position & 255,
    float64 angles; positions up to 2^20), once."""
    plan = next(p for p in cfg.plan if p.k_eq_v)
    inv = np.asarray(cfg.rope_inv_freq(plan), dtype=np.float64)
    rd = int(np.count_nonzero(inv))
    assert rd % 8 == 0 and not np.any(inv[rd:]), "the pairs that turn come first"
    half = plan.head_dim // 2
    key = (rd, half, inv[:rd].tobytes())
    if _KO.get("key") == key:
        return
    nhi = 4096
    tabs = []
    for n, step in ((nhi, 256.0), (256, 1.0)):
        a = np.arange(n, dtype=np.float64)[:, None] * step * inv[None, :rd]
        t = np.ascontiguousarray(np.stack([np.cos(a), np.sin(a)], -1).astype(np.float32))
        b = Buffer(t.nbytes, "weights")
        b.upload(t)
        tabs.append(b)
    _check(lib().rc_gpu_ko_table(tabs[0].ptr, tabs[1].ptr, rd, half, nhi))
    _KO.update(key=key, bufs=tabs)


class GPUKV(DeviceCache):
    """The cache of the 26B model on the GPU (gpumm.DeviceCache: locked; the
    rows past the positions in use are lent to GpuMem).

    Each layer has buffers of (rows, kv_heads, head_dim) values. Row 0 has the
    position base, as in KVCache. The form "int16" keeps the int16 copy of
    KVCache only: kq and vq, and ks and vs, one float32 scale for each group
    of 32 values. That is about half of the float form. The form "float"
    keeps k and v.

    A layer with a window drops its oldest rows when the buffer holds more
    than two windows, as KVCache.prepare does, with a copy on the GPU.
    attach() copies the rows of a KVCache to the GPU. detach() writes the rows
    that only the GPU has into the KVCache with KVCache.write. With the form
    int16, those rows come back as the int16 values times their scales.
    """

    def __init__(self, cfg, kv="int16", max_chunk=512, konly=None):
        super().__init__("cache0")
        self.cfg = cfg
        self.form = kv
        self.max_chunk = max_chunk
        self.window = cfg.sliding_window or 0
        n = cfg.num_hidden_layers
        self.names = ("kq", "ks", "vq", "vs") if kv != "float" else ("k", "v")
        # layer -> c of the layers with keys only (konly_layers): no vq
        self.konly = dict(konly or {})
        # the bytes of a value of kq and of vq
        self.kesz = 1 if kv == "int8" else 2
        self.vesz = 1 if kv in ("int8", "k16v8") else 2
        self.bufs = [dict() for _ in range(n)]
        self.cap = [0] * n
        self.base = [0] * n
        self.end = [0] * n
        self.host_end = [0] * n

    def _names(self, i):
        """The buffers of layer i: no vq for a layer with keys only."""
        return ("kq", "ks", "vs") if i in self.konly else self.names

    def _vs(self, i, ks):
        """The scales of the values of a layer with keys only: ks / c (as
        k_kv_write: ks times the float32 of 1 / c)."""
        return (np.asarray(ks, dtype=np.float32) * np.float32(1.0 / self.konly[i])).astype(np.float32)

    def _row(self, i, name):
        """Return the bytes of one row of the buffer name of layer i."""
        plan = self.cfg.plan[i]
        per = plan.num_kv_heads * plan.head_dim
        return {"k": 4 * per, "v": 4 * per, "kq": self.kesz * per, "vq": self.vesz * per,
                "ks": per // 8, "vs": per // 8}[name]

    def nbytes(self):
        return sum(b.nbytes for d in self.bufs for b in d.values())

    def blocks(self):
        return [b for d in self.bufs for b in d.values()]

    def _alloc(self, i, cap, keep_rows=0):
        """Give layer i buffers of cap rows. Keep the first keep_rows rows.
        When the memory runs out, self.oom frees programs and it tries again;
        else the new buffers go and MemoryError comes up."""
        new = {}
        if self.bufs[i]:
            # the old buffers go: the memory they lent first
            self.unshare(getattr(self, "share_before", None))
        try:
            for name in self._names(i):
                rb = self._row(i, name)
                new[name] = self._new(cap * rb, rows=cap, base=lambda i=i: self.base[i])
                if keep_rows:
                    _check(lib().rc_gpu_d2d(new[name].ptr, self.bufs[i][name].ptr, keep_rows * rb))
        except MemoryError:
            for b in new.values():
                self._free(b)
            raise
        if self.bufs[i]:
            lib().rc_gpu_sync()
            for b in self.bufs[i].values():
                self._free(b)
        self.bufs[i], self.cap[i] = new, cap

    def _host_rows(self, cache, i, rows):
        """Return the host rows of layer i in the form of the GPU."""
        if self.form == "float":
            # the host keeps quantized rows only: their float values
            k, v, _ = cache.read(i, cache.base[i] + rows)
            return {"k": k, "v": v}
        if getattr(cache, "kv", "int16") != self.form:
            raise ValueError("the GPU cache is %s and the host cache %s"
                             % (self.form, getattr(cache, "kv", "int16")))
        if i in self.konly:
            return {"kq": cache.kq[i][:rows], "ks": cache.ks[i][:rows],
                    "vs": self._vs(i, cache.ks[i][:rows])}
        return {"kq": cache.kq[i][:rows], "ks": cache.ks[i][:rows],
                "vq": cache.vq[i][:rows], "vs": cache.vs[i][:rows]}

    def attach(self, cache, max_len):
        # the rows of this cache go where the last lent memory; the copies of
        # HotCache into blocks there land first (share_before: else a block
        # they lock fails the take back, a session from disk after another)
        self.unshare(getattr(self, "share_before", None))
        for i in range(self.cfg.num_hidden_layers):
            rows = cache.end[i] - cache.base[i]
            plan = self.cfg.plan[i]
            cap = ((2 * self.window + KV_KEEP + self.max_chunk + 64) if plan.is_sliding
                   else max(max_len, rows + 64))
            cap = max(cap, rows + 64)
            if not self.bufs[i] or self.cap[i] < cap:
                self._alloc(i, cap)
            self.base[i], self.end[i] = cache.base[i], cache.end[i]
            self.host_end[i] = cache.end[i]
            if rows > 0:
                for name, a in self._host_rows(cache, i, rows).items():
                    self.bufs[i][name].upload(np.ascontiguousarray(a))

    def prepare(self, i, pos, t=1):
        """Make room for the rows of positions pos to pos + t - 1 in layer i."""
        if self.cfg.plan[i].is_sliding:
            w = self.window
            if pos - self.base[i] > 2 * w + KV_KEEP:
                keep = pos - w + 1 - KV_KEEP
                off = keep - self.base[i]
                rows = self.end[i] - keep
                if rows > 0:
                    for name in self._names(i):
                        rb = self._row(i, name)
                        b = self.bufs[i][name]
                        _check(lib().rc_gpu_d2d(b.ptr, b.ptr + off * rb, rows * rb))
                self.base[i] = keep
                # The host may lack some rows that the GPU dropped; only the
                # rows of the window matter to a later step, and detach starts
                # at base. host_end stays the end of the host rows: sync()
                # reads a host end below it as a truncate, and it then cut end
                # (a prompt pass on the GPU past two windows, or a long decode)
                # while the next drop moved base with no rows: the rows of the
                # window no longer matched their positions.
        need = pos + t - self.base[i]
        if need > self.cap[i]:
            self._alloc(i, max(need + 64, 2 * self.cap[i]), self.end[i] - self.base[i])
        self.end[i] = max(self.end[i], pos + t)

    def detach(self, cache, layers=None):
        """Write the rows that the GPU made into the host cache: of every
        layer, or of the layers in the list layers."""
        for i in (range(self.cfg.num_hidden_layers) if layers is None else layers):
            start = max(self.host_end[i], cache.end[i], self.base[i])
            n = self.end[i] - start
            if n <= 0:
                continue
            plan = self.cfg.plan[i]
            shape = (n, plan.num_kv_heads, plan.head_dim)
            if i in self.konly:
                self._detach_konly(cache, i, start, n, shape)
                continue
            # The int16 rows go straight into the host cache when it keeps
            # an int16 copy of them (KVCache.rows_q).
            views = (cache.rows_q(i, start, n) if self.form != "float"
                     and hasattr(cache, "rows_q") else None)
            got = {}
            for j, name in enumerate(self.names):
                rb = self._row(i, name)
                if views is not None:
                    a = views[j]
                    assert a.flags.c_contiguous and a.nbytes == n * rb
                else:
                    dt = {"kq": np.int8 if self.kesz == 1 else np.int16,
                          "vq": np.int8 if self.vesz == 1 else np.int16}.get(name, np.float32)
                    a = np.empty(n * rb // np.dtype(dt).itemsize, dtype=dt)
                _check(lib().rc_gpu_d2h(a.ctypes.data,
                                    self.bufs[i][name].ptr + (start - self.base[i]) * rb, a.nbytes))
                got[name] = a
            if views is not None:
                cache.rows_q_done(i, start, n)
            elif self.form == "float":
                cache.write(i, start, got["k"].reshape(shape), got["v"].reshape(shape))
            elif hasattr(cache, "write_q"):
                sshape = shape[:2] + (shape[2] // 32,)
                cache.write_q(i, start, got["kq"].reshape(shape), got["ks"].reshape(sshape),
                              got["vq"].reshape(shape), got["vs"].reshape(sshape))
            else:
                k = (got["kq"].reshape(-1, 32) * got["ks"][:, None]).reshape(shape)
                v = (got["vq"].reshape(-1, 32) * got["vs"][:, None]).reshape(shape)
                cache.write(i, start, k, v)
            self.host_end[i] = self.end[i]

    def _detach_konly(self, cache, i, start, n, shape):
        """detach for a layer with keys only: the host gets the keys and the
        values made from them, as the kernels make them: the pairs that do
        not turn are the int16 keys with vs = ks / c, the turned ones are
        turned back in float and quantized again."""
        from reclank.core import ops
        per = shape[1] * shape[2]
        kq = np.empty(shape, np.int16)
        ks = np.empty(shape[:2] + (shape[2] // 32,), np.float32)
        off = start - self.base[i]
        _check(lib().rc_gpu_d2h(kq.ctypes.data, self.bufs[i]["kq"].ptr + off * 2 * per, kq.nbytes))
        _check(lib().rc_gpu_d2h(ks.ctypes.data, self.bufs[i]["ks"].ptr + off * per // 8, ks.nbytes))
        vq, vs = kq.copy(), self._vs(i, ks)
        inv = np.asarray(self.cfg.rope_inv_freq(self.cfg.plan[i]), dtype=np.float64)
        rd, half = int(np.count_nonzero(inv)), shape[2] // 2
        if rd:
            ang = np.outer(start + np.arange(n, dtype=np.float64), inv[:rd])
            cos = np.cos(ang).astype(np.float32)[:, None, :]
            sin = np.sin(ang).astype(np.float32)[:, None, :]
            deq = lambda a, b: (kq[..., a:b].reshape(n, shape[1], -1, 32).astype(np.float32)  # noqa: E731
                                * ks[..., a // 32:b // 32, None]).reshape(n, shape[1], b - a)
            x, y = deq(0, rd), deq(half, half + rd)
            cinv = np.float32(1.0 / self.konly[i])
            for a, val in ((0, (x * cos + y * sin) * cinv), (half, (y * cos - x * sin) * cinv)):
                q, sc = ops.quantize_i16(np.ascontiguousarray(val, dtype=np.float32).reshape(-1, 32))
                vq[..., a:a + rd] = q.reshape(n, shape[1], rd)
                vs[..., a // 32:(a + rd) // 32] = sc.reshape(n, shape[1], rd // 32)
        if hasattr(cache, "write_q"):
            cache.write_q(i, start, kq, ks, vq, vs)
        else:
            k = (kq.reshape(-1, 32) * ks.reshape(-1)[:, None]).reshape(shape)
            cache.write(i, start, k, (vq.reshape(-1, 32) * vs.reshape(-1)[:, None]).reshape(shape))
        self.host_end[i] = self.end[i]

    def window_get(self, i, start, n):
        """The rows of positions start .. start + n - 1 of layer i (host
        arrays by name), or None when the GPU does not hold them all
        (Model.window_snapshot)."""
        if start < self.base[i] or start + n > self.end[i]:
            return None
        plan = self.cfg.plan[i]
        out = {}
        for name in self._names(i):
            rb = self._row(i, name)
            dt = np.float32 if name in ("ks", "vs", "k", "v") else (np.int8 if rb == plan.num_kv_heads * plan.head_dim else np.int16)
            a = np.empty(n * rb // np.dtype(dt).itemsize, dt)
            if n:
                _check(lib().rc_gpu_d2h(a.ctypes.data, self.bufs[i][name].ptr + (start - self.base[i]) * rb,
                                    n * rb))
            out[name] = a
        return out

    def window_put(self, i, rows, start, end):
        """Layer i holds the rows of positions start .. end - 1 (rows of
        window_get) from buffer row 0: base start, end end
        (Model.window_restore)."""
        n = end - start
        if self.cap[i] < n:
            self._alloc(i, max(n + 64, self.cap[i]))
        for name in self._names(i):
            a = np.ascontiguousarray(rows[name])
            _check(lib().rc_gpu_h2d(self.bufs[i][name].ptr, a.ctypes.data, a.nbytes))
        self.base[i], self.end[i] = start, end
        self.host_end[i] = end

    def truncate(self, n):
        """Cut the rows back to n positions, as KVCache.truncate does. Return
        False, with no change, when a layer with a window dropped a row that
        a token at n sees (the rows before base are gone on the GPU)."""
        w = self.window
        for i in range(self.cfg.num_hidden_layers):
            first = max(0, n - w + 1) if (self.cfg.plan[i].is_sliding and w) else n
            if self.base[i] > first:
                return False
        for i in range(self.cfg.num_hidden_layers):
            self.end[i] = min(self.end[i], n)
            self.host_end[i] = min(self.host_end[i], n)
        return True

    def sync(self, cache):
        """Follow a truncate of the host cache (KVCache.truncate). A row
        after the end of the host is not a row of the history any more."""
        for i in range(self.cfg.num_hidden_layers):
            if cache.end[i] < self.host_end[i]:
                self.host_end[i] = cache.end[i]
                self.end[i] = min(self.end[i], cache.end[i])

    def params(self):
        kw = {}
        for i in range(self.cfg.num_hidden_layers):
            kw["base.%d" % i] = self.base[i]
            for name in self._names(i):
                kw["%s.%d" % (name, i)] = self.bufs[i][name].ptr
            if i in self.konly:
                # the values are the keys (ko_tab); a turned row needs its position
                assert self.base[i] == 0, "a layer with keys only starts at position 0"
                kw["vq.%d" % i] = self.bufs[i]["kq"].ptr
        return kw


class ModelGPU(WarmPool, ProgramLRU):
    """A decode step of the 26B model on the GPU, with the experts on the CPU.
    A dense model (the 12B) runs wholly on the GPU if it fits. See the module
    text and GPUKV.

        g = gpu.ModelGPU(model)
        g.attach(cache)                  # after the prompt pass on the CPU
        xn = g.step([token], pos)        # the hidden state after the final norm
        logits = g.logits()              # the head of the last step
        g.detach(cache)                  # the host cache has the new rows

    The hot experts of the 26B are blocks of the pool of HotCache (POOL_HOT,
    WarmPool: the cache system of the Qwen models), with the warm experts in
    the memory the prompts let go (WARM).
    """

    warm_mode = WARM        # WarmPool (RECLANK_GPU_WARM, _WARM_KEEP, _WARM_MIN)
    warm_keep = WARM_KEEP
    warm_min = WARM_MIN
    warm_short = WARM_SHORT
    warm_regrow = True
    hot_run = True          # HotCache: the uses of the conversation decide the copies

    def __init__(self, model, graph=True, hot=None, kv=None, resident=False):
        """hot is a dict layer -> experts (see pick_hot), or None. resident
        puts every expert on the GPU (the resident mode, REFACTOR_PLAN.md
        section 4.6): no CPU part, no HotCache, no copies of experts;
        MemoryError when they do not fit.

        Without hot, a model with experts gets the most used experts of a
        file of counts (scripts/expert_use.py). The file is RECLANK_GPU_HOT,
        or else HOT_COUNTS when its shape agrees with the model.
        RECLANK_GPU_HOT=0 turns the hot experts off. RECLANK_GPU_HOT_GB
        gives the budget in GB. The default budget is the free memory of the
        GPU less 6 GB: the step, its head, and a cache of 4096 tokens take
        about 2.7 GB, a prompt pass takes about 1.3 GB more, and the display
        needs some memory too.
        """
        self.model = model
        self.resident = bool(resident and model.cfg.enable_moe_block)
        free0 = mem_info()[0]       # (the memory before the weights)
        self.chunk = chunk_for(free0)
        gb = os.environ.get("RECLANK_GPU_HOT_GB")
        budget = float(gb) * 1e9 if gb else max(0.0, free0 - 6.0e9)
        if self.resident:
            hot = resident_hot(model)
        elif hot is None and model.cfg.enable_moe_block:
            counts = hot_counts(model)
            if counts is not None:
                hot = pick_hot(model, counts, budget)
        self.hot = hot or {}
        from reclank.models.gemma4 import model as _model_mod
        kv = kv or (_model_mod.KV_FORM if _model_mod.KV_FORM != "int16"
                    else os.environ.get("RECLANK_GPU_KV", "int16"))
        kv = _model_mod.kv_base(kv)     # (rq8, k16vr8: the step form rotates; model.kv_rot)
        self.kv_form = kv
        self.graph = graph
        # the hot experts in the pool of HotCache (POOL_HOT): the first ones
        # go to its "hot" segments below, and the step has no fixed slots
        self.pool = None
        self.pooled = (model.cfg.enable_moe_block and not self.resident and
                       os.environ.get("RECLANK_GPU_HOT_DYN", "1") != "0" and
                       (POOL_HOT == "1" or (POOL_HOT == "auto" and
                                            budget < POOL_FIT * experts_bytes(model))))
        hot_init = {}
        if self.pooled:
            hot_init, self.hot = self.hot, {}
        self.prog = compile_split_step(model, self.hot, kv, resident=self.resident,
                                       pooled=self.pooled)
        # the arenas of the temporaries of the programs (alloc.Arenas: the
        # step and the small groups, the large groups)
        self.arenas = alloc.Arenas(MT_CPU)
        self.arenas.keep(1, self.prog)
        self.g = GPUProgram(self.prog, graph=graph)
        konly = konly_layers(model, kv)
        if konly:
            ko_table(model.cfg)
        self.kv = GPUKV(model.cfg, kv, max_chunk=self.chunk, konly=konly)
        self.kv.share_before = self._share_before
        self.groups = {}      # t -> (Program, GPUProgram) of a group of t tokens
        self._lru_init()
        self.kv.oom = self._evict
        self.head = None
        self.rows = 1
        self.max_len = 4096
        # The device address of the hidden state of the last row of the last
        # step or group, for the output head.
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        # The hot experts follow the text (HotCache). RECLANK_GPU_HOT_DYN=0
        # keeps the first set.
        self.hot_cache = None
        if (self.prog.hot_stores and not self.resident and
                os.environ.get("RECLANK_GPU_HOT_DYN", "1") != "0"):
            self.hot_cache = HotCache(self)
            from reclank.backends.cuda.devices import expert_device
            xdev = expert_device(self.model)
            if xdev is not None:
                # exclusive caching: the slots do not take the expert device's experts
                self.hot_cache.exclude = np.stack([xdev.held[e["layer"]].astype(bool)
                                                   for e in self.hot_cache.layers])
        if self.pooled and self.hot_cache is not None:
            self._pool_first(hot_init)
        self._mm_freeze()       # GpuMem: the weights, the stores, the step stay
        if self.pooled:
            self._warm_reclaimers()

    def _run_blocks(self):
        """GpuMem: each run uses the cache and the pool (the kernels read the
        segment table at run time)."""
        return self.kv.blocks() + (self.pool.run_locks() if self.pool is not None else [])

    def _share_before(self):
        """Before the memory the cache lends changes: the copies of HotCache
        into blocks of the pool land first (they lock their segments)."""
        if self.hot_cache is not None:
            self.hot_cache.prepare(wait=True)

    def _pool_first(self, hot_init):
        """The pool of HotCache (blocks of the size of the experts of the
        26B: the gate and up rows, the down rows) and its first hot experts
        (hot_init: layer -> experts, the budget of pick_hot) in its "hot"
        segments."""
        hc = self.hot_cache
        per = [0, 0, 0]
        for e in hc.layers:
            for p, (_src, nb, _dst) in enumerate(e["parts"]):
                per[p] = max(per[p], nb)
        self.per = per
        self._ensure_pool()
        entries = [(r, hot_init.get(e["layer"], []), e["parts"], e["slots"])
                   for r, e in enumerate(hc.layers)]
        E = hc.score.shape[1]
        for r, x, b in self._pool_place(entries):
            self.pool.owner[b] = r * E + x
            hc.held[r, x] = True
        hc._upload(list(range(len(hc.layers))))

    def attach(self, cache):
        self.kv.attach(cache, max(self.max_len, cache.max_len))
        if self.hot_cache is not None:
            self.hot_cache.new_conversation()       # (a new cache: a new conversation)

    def detach(self, cache):
        self.kv.detach(cache)

    def _params(self, pos, t):
        """Prepare the cache for t rows from pos. Return the parameters."""
        model, cfg = self.model, self.model.cfg
        for i in range(cfg.num_hidden_layers):
            self.kv.prepare(i, pos, t)
        # (GpuMem: the rows past the run are lent; the copies of HotCache
        # into lent blocks land first)
        self.kv.share_tail(pos + t, self._share_before)
        kw = {"pos": pos}
        kw.update(self.kv.params())
        positions = np.arange(pos, pos + t)
        for kind, sliding in (("s", True), ("f", False)):
            plan = next((p for p in cfg.plan if p.is_sliding == sliding), None)
            if plan is None:
                continue
            cos, sin, _ca, _sa = model._rope(plan, positions)
            kw["cos." + kind] = np.ascontiguousarray(cos, dtype=np.float32)
            kw["sin." + kind] = np.ascontiguousarray(sin, dtype=np.float32)
        # The attention of a large group needs no scores buffer on the GPU.
        # A small group runs the attention of a decode step for each query.
        need = max(p.num_q_heads * (pos + t) for p in cfg.plan) if t <= MT_CPU else 16
        kw["scores"] = np.empty(need, dtype=np.float32)
        return kw

    def refresh_experts(self):
        """Between prompts (the engine's idle): the expert device's experts by the conversation's
        routing (devices.refresh_hot). Return (experts copied, bytes, seconds), or None."""
        from reclank.backends.cuda.devices import expert_device, refresh_hot
        st = getattr(self, "_xstate", None)
        if st is None:
            st = self._xstate = {}
        return refresh_hot(expert_device(self.model), self.hot_cache, st)

    def step(self, tokens, pos):
        """Run a step of one token. Return the hidden state after the final
        norm, shape (1, hidden)."""
        assert len(tokens) == 1, "step runs one token; group runs more"
        if self.hot_cache is not None:
            self.hot_cache.prepare()
        if self.pooled:
            self._warm_on()
        kw = self._params(pos, 1)
        self.prog.names["x"][:] = self.model.embed(tokens)
        self.g.upload("x")
        self.g.bind(kw)
        self.g.run()
        self.g.download("xn")
        if self.hot_cache is not None:
            # logits() runs it while the GPU runs the head, or else the next
            # prepare().
            self.hot_cache.due = True
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.rows = 1
        return self.prog.names["xn"].copy()

    def _hot_devices(self):
        """Return layer -> (slots, device address of the gate and up blocks
        of the hot experts, of their down blocks), from the step program."""
        out = {}
        for layer, (_hot, gu, dn, slots) in self.prog.hot_stores.items():
            out[layer] = (slots, self.g.mirror.buffer_of(gu).ptr, self.g.mirror.buffer_of(dn).ptr)
        return out

    def _stage(self, cap=None, floor=None):
        """Return the two device buffers for the weights of the cold experts
        of a layer, for a large group. The GPU holds the hot experts already,
        so a buffer holds only the cold experts of the layer with the most,
        or cap of them. floor (or None: all or nothing): with too little
        memory, buffers for fewer (halves, down to floor) - the pieces of
        chunked staging or the places of a mixed group (whose CPU takes the
        cold experts the buffers do not hold).
        Where all or nothing gave the 26B on a card of 4096 MiB the CPU
        experts alone for a prompt of 48K tokens: 51 tok/s."""
        if getattr(self, "stage", None) is None:
            w = self.model._layers[0]
            gu, dn = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
            n = gu.shape[0]
            cold = max(n - len(self.hot.get(i, ())) for i in range(self.model.cfg.num_hidden_layers))
            cold = max(cold, 1)     # (resident: no cold expert, the buffers stay unused)
            if cap is not None:
                cold = min(cold, cap)
            while True:
                made = []
                try:
                    for _ in range(2):
                        made.append(Buffer(cold * gu.nbytes // n))
                        made.append(Buffer(cold * dn.nbytes // n))
                    break
                except MemoryError:
                    # GpuMem keeps a block until it is freed: the buffers made
                    # so far go too
                    for b in made:
                        b.free()
                    if floor is None or cold <= floor:
                        raise
                    cold = max(floor, cold // 2)
            self.stage = [(made[0], made[1]), (made[2], made[3])]
        return self.stage

    def _group(self, t):
        e = self.groups.get(t)
        if e is None:
            # GP_MOE_GPU of the KQ_Q4X experts needs the int8 tiles (the tensor
            # cores, or dp4a on Pascal); without them a large group takes the
            # form of a small one (the hot experts on the GPU, the cold ones on
            # the CPU).
            stage = None
            # The mixed groups: with the KQ_Q4X copies (VNNI or AVX2) always;
            # without them (an AVX-512 CPU with no VNNI, the plain C build)
            # only when the buffers of all the cold experts or the program
            # beside them do not fit (mix_short: then the buffers shrink and
            # the CPU takes the cold experts they do not hold). The plain
            # large group with all of them is 4% faster there (370 against
            # 385 tok/s for 8K tokens on the GTX 1070).
            can_mix = MIX and not self.resident
            mix = can_mix and (expert_host(self.model, 0)[2] or getattr(self, "mix_short", False))
            n = self.model._layers[0]["experts.gate_up_proj"][0].shape[0]
            least = -(-n // CHUNK_MAX)      # the fewest places of chunked staging
            cap = getattr(self, "stage_cap", None)
            chunks = 1
            # GP_MOE_GPU takes the int4 experts in float32 too (k_moe_gemm):
            # without the int8 products (Maxwell) the buffers come all the same
            if (t > MT_CPU and self.model.cfg.enable_moe_block
                    and (int8_tiles() or not expert_host(self.model, 0)[2])
                    and not getattr(self, "stage_off", False)):
                try:
                    if mix:
                        stage = self._stage(cap=cap, floor=STAGE_MIN)
                    else:
                        # all the cold experts at once, or else (stage_cap set)
                        # chunked staging, down to CHUNK_MAX pieces
                        stage = self._stage(cap=cap, floor=least if cap is not None else None)
                except MemoryError:
                    if not mix and cap is None:
                        self.stage_cap = n // 2
                        return self._group(t)
                    if can_mix and not mix:
                        self.mix_short = True
                        return self._group(t)
                    # Little memory (the 26B on a card of 4 GB with a long
                    # cache): no buffers for the cold experts; the large
                    # groups take the form of a small one, their cold
                    # experts on the CPU (and no new try, which would
                    # first free the programs and the hot experts).
                    self.stage_off = True
                if stage is not None and not mix:
                    room = stage_room(self.model, stage)
                    if room < n - min(len(self.hot.get(i, ())) for i in range(self.model.cfg.num_hidden_layers)):
                        # fewer places than cold experts: pieces (every expert
                        # of a layer may be cold when HotCache moves them)
                        chunks = -(-n // room)
            prog = compile_split_group(self.model, t, kv=self.kv_form, stage=stage,
                                       hot_stores=self._hot_devices() if stage else None,
                                       hot_host=self.prog.hot_stores,
                                       counts=self.hot_cache.counts if self.hot_cache else None,
                                       mix=mix, resident=self.resident,
                                       pool=self.pool, arena=self.arenas.arena_for(t), chunks=chunks)
            self.arenas.keep(t, prog)
            # The tensor cores round the input of a product to float16. The
            # router of the experts then selects another expert more often:
            # 93% of the top tokens agree with the CPU, against 98% with the
            # float32 products. The prompt pass of the 26B waits for the copy
            # of the experts anyway, so it keeps float32 products.
            tc = os.environ.get("RECLANK_GPU_TC_MOE", "0") == "1" or \
                not self.model.cfg.enable_moe_block
            # The int4 dense products take int8 x all the same (k_gemm_q8): an
            # int8 block has its own scale, so it has no overflow. On a chat
            # text, 98.9% of the top tokens agree with float32 (KL 0.002), and
            # the prompt pass is 1.6 times as fast (README). With the int16
            # prompt of the CPU (Model.prompt_act "16", the default of the
            # 26B) they and the KQ_Q4X experts take the int16 form: x as two
            # int8 planes, q = 128 hi + lo, and two int8 products of the
            # tensor cores for each block (k_quant_x2).
            i8 = os.environ.get("RECLANK_GPU_I8_MOE", "1") == "1"
            if i8 and self.model.prompt_act == "16":
                i8 = 2
            # The attention runs on the tensor cores all the same
            # (k_flash_qc_tc): the queries and the keys have a norm, so
            # float16 has no overflow there. It is ten times as fast as the
            # float32 kernel, and on a chat text the KL to float32 goes from
            # 0.0020 to 0.0024.
            atc = os.environ.get("RECLANK_GPU_ATTN_TC", "1") == "1"
            try:
                built = self._build_program(prog, t, tc=tc, i8=i8, atc=atc)
            except MemoryError:
                if stage is None:
                    raise
                room = stage_room(self.model, stage)
                if not mix and room > least:
                    # the buffers fit, but then the program did not: smaller
                    # buffers in more pieces (stage_cap), and a new try
                    self._warm_programs()
                    self.stage_cap = max(least, room // 2)
                    return self._group(t)
                if can_mix and (room > STAGE_MIN or not mix):
                    # no more pieces: a mixed group (its CPU takes what the
                    # buffers do not hold)
                    self._warm_programs()
                    self.mix_short = True
                    self.stage_cap = max(STAGE_MIN, room // 2)
                    return self._group(t)
                # the buffers of the cold experts fit, but then the program
                # did not (the 26B on 4 GB in Pascal mode, the cache of keys
                # only: 32K tokens): the large groups without them
                self._stage_drop()
                return self._group(t)
            e = self.groups[t] = (prog, built)
        self._used(t)
        return e

    def _warm_programs(self):
        """WarmPool: the programs of the large groups and the buffers of the
        cold experts give their memory to the warm experts (a later large
        group makes them again)."""
        for t in [t for t in self.groups if t > MT_CPU]:
            self._lru.pop(("groups", t), None)
            self._free_program(*self.groups.pop(t))
        for pair in getattr(self, "stage", None) or ():
            for b in pair:
                b.free()
        self.stage = None
        if getattr(self, "head_cpu", False) and self.head is None:
            # the head on the CPU (_run_head): the memory of the prompt goes
            # to it first, then to the pool
            try:
                self._load_head()
                self.head_cpu = False
            except MemoryError:
                pass

    def _warm_prompt_held(self):
        return any(t > MT_CPU for t in self.groups) or getattr(self, "stage", None) is not None

    def _stage_drop(self):
        """Free the buffers of the cold experts (_stage) and the programs of
        the large groups that use them; the large groups then keep their
        cold experts on the CPU."""
        self._warm_programs()
        self.stage_off = True

    def group(self, tokens, pos, size=None, media=None):
        """Run a group of tokens from position pos. Return the hidden states
        after the final norm, shape (len(tokens), hidden).

        size pads the group to a program of that many tokens. The padding
        rows write cache rows after the group. The cache then forgets them:
        a later step writes those rows again.

        media is a list of media.Span (absolute positions): their rows take
        the place of the token rows, and the tokens of a span with bidir see
        each other (the whole span must be in the group).
        """
        t = len(tokens)
        size = size or t
        prog, g = self._group(size)
        if self.hot_cache is not None:
            # A large group copies the cold experts to buffers of a fixed
            # size, so it needs every slot full: wait for the copies.
            self.hot_cache.prepare(wait=size > MT_CPU)
        if self.pooled and size <= MT_CPU and not getattr(self, "in_prompt", False):
            self._warm_on()         # (an MTP verify group, as a step; not the end of a prompt)
        mixed = size > MT_CPU and getattr(prog, "mix", False)
        kw = self._params(pos, size)
        kw["nreal"] = t
        x = prog.names["x"]
        on_gpu = size > MT_CPU and not media and self._embed_gpu(g, prog, tokens, size)
        if not on_gpu:
            x[:t] = self.model.embed(tokens)
            x[t:] = 0.0
        lim = prog.names.get("lim")
        if lim is not None:
            lim[:] = np.arange(lim.shape[0], dtype=np.int32)
        for sp in media or ():
            lo, hi = max(sp.start, pos), min(sp.end, pos + t)
            if lo >= hi:
                continue
            x[lo - pos:hi - pos] = sp.rows[lo - sp.start:hi - sp.start]
            if sp.bidir and lim is not None and _BIDIR_ALL_OK:
                if sp.start < pos or sp.end > pos + t:
                    raise ValueError("a group splits the image at %d to %d" % (sp.start, sp.end))
                lim[lo - pos:hi - pos] = sp.end - 1 - pos
        if lim is not None:
            g.upload("lim")
        if not on_gpu:
            g.upload("x")
        # The tables of a large group last: the cache, the head, the program
        # above and its binding may take blocks of the pool (their warm experts
        # go cold, and a block may move). bind came after the tables: its
        # arrays (the scores grow with pos) took memory from the pool, so a
        # table kept the address of a freed segment (the 26B with a cache of
        # 131072 tokens on the GTX 1070: an illegal address in the experts).
        g.bind(kw)
        if self.hot_cache is not None and size > MT_CPU and not mixed:
            self.hot_cache.refresh(prog, g)
        if mixed:
            self.plan_mix(prog, g)
            if MIX_CAL:
                _check(lib().rc_gpu_mix_stats_on(1))
        t0 = time.perf_counter()
        g.run()
        g.download("xn")
        wall = time.perf_counter() - t0
        if self.hot_cache is not None:
            self.hot_cache.seed(g, t, prompt=getattr(self, "in_prompt", False))
        if mixed and MIX_CAL:
            self.calibrate_mix(prog, size, wall)
            _check(lib().rc_gpu_mix_stats_on(0))
        for i in range(self.model.cfg.num_hidden_layers):
            self.kv.end[i] = min(self.kv.end[i], pos + t)
        hidden = self.model.cfg.hidden_size
        self.first = g.mirror.buffer_of(prog.names["xn"]).ptr
        self.last = self.first + (t - 1) * hidden * 4
        self.rows = t
        return prog.names["xn"][:t].copy()

    def plan_mix(self, prog, g):
        """Choose the cold experts that each layer of a mixed group copies to
        the GPU: the MIX_PRE cold experts with the most selections in the
        group before (HotCache.last), or else in the file of counts. The
        CPU takes the other cold experts that the routers select. Fill the
        tables and the marks of the experts on the GPU, and upload them."""
        last = self.hot_cache.last if self.hot_cache is not None else None
        if last is None or not last.any():
            if getattr(self, "_static_counts", None) is None:
                c = hot_counts(self.model)
                self._static_counts = c if c is not None else np.zeros(
                    (self.model.cfg.num_hidden_layers, self.model.cfg.num_experts))
            last = self._static_counts
        hot = self._hot_devices()
        t = prog.tokens
        cfg = self.model.cfg
        cal = self.mix_cal
        per = sum(a.nbytes for a in expert_host(self.model, 0)[:2]) / cfg.num_experts
        a_ms = cal["gpu_tok"] * t
        self.mix_plan = []
        for layer, (tgu, tdn, ranges) in prog.tables_of.items():
            h = hot.get(layer)
            slots = h[0] if h is not None else np.full(tgu.shape[0], -1)
            pred = np.asarray(last[layer], dtype=np.float64)
            pred = pred * (t * cfg.top_k_experts / max(pred.sum(), 1.0))
            order = np.argsort(-pred, kind="stable")
            cold = [int(x) for x in order if slots[x] < 0]
            if MIX_PRE != "auto":
                m = len(cold) if int(MIX_PRE) < 0 else int(MIX_PRE)
            elif not expert_host(self.model, layer)[2]:
                # the int4 experts on the CPU (no KQ_Q4X copies: MOE_MT) are
                # slow next to the copies: the CPU takes only the cold experts
                # that the buffers do not hold (the GTX 1070 with all of them on
                # the GPU: 8K tokens at 385 tok/s, with the model's split 299)
                m = len(cold)
            else:
                # The time of the layer for m copied experts: the copies go on
                # during the work of the layers before, and the CPU part runs
                # during the copy wait and the GPU experts.
                cp = pred[cold]
                rest = cp.sum() - np.concatenate(([0.0], np.cumsum(cp)))
                used = cp > 0
                rest_e = used.sum() - np.concatenate(([0], np.cumsum(used)))
                ms = np.arange(len(cold) + 1)
                gpu_pairs = pred.sum() - rest
                cpu = cal["cpu_a"] + rest_e * cal["cpu_e"] + rest * cal["cpu_b"]
                tm = np.maximum(ms * per / cal["rate"],
                                a_ms + np.maximum(gpu_pairs * cal["gpu_pair"],
                                                  np.where(rest > 0, cpu, 0.0)))
                # The model is flat where the copies stay behind the other
                # work and the CPU part ends before the GPU experts. There the
                # most copies leave the CPU the least work: a margin for the
                # errors of the prediction and for other load on the CPU.
                m = int(ms[tm <= tm.min() + 0.25].max())
            # at most the places of the buffers (fill_tables): the most used of the cold experts
            m = min(m, stage_room(self.model, self.stage))
            self.mix_plan.append(m)
            pre = set(cold[:m])
            gslots = prog.gslots_of[layer]
            fill_tables(self.model, layer, h, self.stage, tgu, tdn, ranges, gslots, pre, pool=self.pool)
            for a in (tgu, tdn, gslots):
                _check(lib().rc_gpu_h2d(g.mirror.buffer_of(a).ptr, a.ctypes.data, a.nbytes))
        prog.hot_version = None

    @property
    def mix_cal(self):
        """The model of the time of a mixed group (MIX_START, then the
        measures of calibrate_mix)."""
        if "_mix_cal" not in self.__dict__:
            self._mix_cal = dict(MIX_START)
        return self._mix_cal

    def calibrate_mix(self, prog, t, wall):
        """Fit the model of plan_mix to the measures of the mixed group that
        just ran (gg_mix_stats; wall is its time in s): the copy rate, the
        time of the CPU part against its pairs in each layer, and the other
        GPU work for each token. Each new value is the mean of the old one
        and the measure."""
        buf = np.zeros(4096)
        n = lib().rc_gpu_mix_stats(buf.ctypes.data, buf.size)
        if n < 0:
            raise RuntimeError(lib().rc_gpu_last_error().decode())
        nbytes, secs, nf, nj = buf[0], buf[1], int(buf[2]), int(buf[3])
        waits = buf[4:4 + 3 * (nf + nj)].reshape(-1, 3)     # kind, wait, work before
        cpu_s = buf[4 + 3 * (nf + nj):4 + 3 * (nf + nj) + nj]
        cal = self.mix_cal
        new = {}
        if secs > 0 and nbytes > 0:
            new["rate"] = nbytes / (1e3 * secs)
        last = self.hot_cache.last if self.hot_cache is not None else None
        layers = sorted(prog.gslots_of)
        # Each layer gives a wait of the copies, then a wait of the CPU part.
        # The work before a copy wait is the other work of the layer (from the
        # wait of the CPU part of the layer before); the work before a CPU
        # wait is the GPU experts.
        kinds = waits[:, 0] if len(waits) else np.zeros(0)
        other = waits[1:][kinds[1:] == 0, 2]
        if other.size:
            new["gpu_tok"] = 1e3 * float(np.median(other)) / t
        if last is not None and nj == len(layers):
            cpu_pairs = np.array([float(last[l][prog.gslots_of[l] < 0].sum()) for l in layers])
            cpu_exp = np.array([float((last[l][prog.gslots_of[l] < 0] > 0).sum())
                                for l in layers])
            gpu_pairs = np.array([float(last[l][prog.gslots_of[l] >= 0].sum()) for l in layers])
            ms = 1e3 * cpu_s
            # ms = cpu_a + cpu_e experts + cpu_b pairs, with no term below 0:
            # the terms that come out below 0 go, and the fit runs again.
            cols = {"cpu_a": np.ones_like(ms), "cpu_e": cpu_exp, "cpu_b": cpu_pairs}
            keys = [k for k in cols if np.ptp(cols[k]) > 0 or k == "cpu_a"]
            while keys:
                coef = np.linalg.lstsq(np.stack([cols[k] for k in keys], 1), ms, rcond=None)[0]
                if (coef >= 0).all():
                    new.update({k: float(c) for k, c in zip(keys, coef)})
                    new.update({k: 0.0 for k in cols if k not in keys})
                    break
                keys = [k for k, c in zip(keys, coef) if c >= 0]
            gw = 1e3 * waits[kinds == 1, 2]
            if gw.size == len(layers) and gpu_pairs.sum() > 0:
                new["gpu_pair"] = float(gw.sum() / gpu_pairs.sum())
        for k, v in new.items():
            cal[k] = 0.5 * (cal[k] + v)
        if MIX_LOG:
            fw = 1e3 * float(waits[kinds == 0, 1].sum()) if len(waits) else 0.0
            cw = 1e3 * float(waits[kinds == 1, 1].sum()) if len(waits) else 0.0
            print("mix: %d tokens %.0f ms: copy waits %.0f ms, CPU waits %.0f ms, plan %.1f; "
                  "measured %s -> %s" % (t, 1e3 * wall, fw, cw, np.mean(self.mix_plan),
                                         {k: round(v, 5) for k, v in new.items()},
                                         {k: round(v, 5) for k, v in cal.items()}))

    def prefill(self, ids, pos=0, media=None):
        """Run a prompt from position pos. Return the hidden states after the
        final norm of every token. media: see group.

        A part of PREFILL_MIN tokens or more goes in chunks of self.chunk tokens,
        with the experts on the GPU. A short chunk goes to a program of the
        next power of two. Each such chunk copies the weights of the experts
        to the GPU. A shorter part goes in groups of 16 tokens, with the
        experts on the CPU, which costs less than that copy.
        """
        self.in_prompt = True       # for HotCache.seed
        try:
            return self._prefill(ids, pos, media or ())
        finally:
            self.in_prompt = False

    def _prefill(self, ids, pos, media=()):
        out = []
        c0 = 0
        while c0 < len(ids):
            end = min(c0 + self.chunk, len(ids))
            # A chunk ends before an image that it would split (an image has
            # at most 1120 tokens, less than a chunk).
            for sp in media:
                if sp.bidir and pos + c0 < sp.end and sp.start < pos + end < sp.end:
                    if sp.start > pos + c0:
                        end = sp.start - pos
                    else:
                        end = min(len(ids), sp.end - pos)
            chunk = ids[c0:end]
            spans = [sp for sp in media if sp.start < pos + end and sp.end > pos + c0]
            if (len(chunk) >= PREFILL_MIN or not self.model.cfg.enable_moe_block
                    or any(sp.bidir for sp in spans)):
                size = max(MT_CPU + 1, 1 << (len(chunk) - 1).bit_length())
                out.append(self.group(chunk, pos + c0, min(size, self.chunk), media=spans))
            else:
                chunk = ids[c0:c0 + MT_CPU]
                spans = [sp for sp in media if sp.start < pos + c0 + len(chunk)
                         and sp.end > pos + c0]
                out.append(self.group(chunk, pos + c0, MT_CPU, media=spans))
            c0 += len(chunk)
        return np.concatenate(out)

    def logits(self, rows=1):
        """Return the logits of the last rows of the last step or group, with
        the soft cap, shape (rows, vocabulary). An MTP verify group needs the
        logits of each of its rows. At most 16 rows."""
        vocab = self._run_head(rows)
        _check(lib().rc_gpu_d2h(self.host_logits.ctypes.data, self.out.ptr, rows * vocab * 4))
        return self.host_logits[:rows].copy()

    def topk(self, rows, k, temperature):
        """Return the candidates of sampling of the last rows (_topk_rows)."""
        self._run_head(rows)
        return _topk_rows(self, rows, k, temperature)

    def argmax(self, rows=1):
        """Return the best token of each of the last rows (_argmax_rows): the
        tokens of np.argmax on logits(rows), with a copy of rows integers."""
        self._run_head(rows)
        return _argmax_rows(self, rows)

    def _load_head(self):
        """Copy the head (the tied token table) to the GPU, on the first
        call."""
        if self.head is not None:
            return
        model = self.model
        e3 = getattr(model, "_head_e3", None)
        if e3 is not None:
            # the head of an EXL3 directory (lm_head, a KQ_E3 matrix): the
            # kernels of GP_KQ_LINEAR with the softcap (rc_gpu_e3_linear)
            w = e3.block()
            K = e3.K

            def e3_head(wp, x, out, vocab, hidden, cap, rows):
                return lib().rc_gpu_e3_linear(wp, K, vocab, hidden, x, rows, out, cap)
            self.head_fn = e3_head
            vocab = e3.rows
        elif model._embed_q6k is not None:
            w = np.ascontiguousarray(model._embed_q6k_bytes)
            self.head_fn = lib().rc_gpu_q6k_head
        elif model._embed_q is not None and model._dtype == "int4":
            # A file that keeps the head in Q4_0: the blocks of the file
            # (gguf.int4_packed), 18 bytes for 32 weights.
            w = np.ascontiguousarray(model._embed_q).reshape(model._embed_q.shape[0], -1)
            self.head_fn = lib().rc_gpu_q4_head
        else:
            raise RuntimeError("the GPU head needs a Q6_K or a Q4_0 head")
        if e3 is None:
            vocab = w.shape[0]
        if getattr(self, "host_logits", None) is None or self.host_logits.shape[1] != vocab:
            # the small buffers first: a call again after a MemoryError of
            # the head (_load_head_low) makes them only once
            self.out = Buffer(4 * vocab * MT_CPU, "weights")
            self.host_logits = pinned((MT_CPU, vocab))     # a faster copy from the GPU
        self.head_bytes = w.nbytes
        head = Buffer(w.nbytes, "weights")
        head.upload(w)
        self.head = head

    def _embed_gpu(self, g, prog, tokens, size):
        """Write the rows of the tokens (and zeros for the padding to size)
        into x of the program on the GPU, with the Q4_0 token table of the
        head (gg_embed_q4_rows). Return False when the head is not Q4_0."""
        model = self.model
        if not EMBED_GPU or model._embed_q6k is not None or model._embed_q is None \
                or model._dtype != "int4":
            return False
        if self.head is None and mem().room() < 2 * model._embed_q.nbytes:
            return False        # little memory: the rows from the CPU (EMBED_GPU)
        self._load_head()
        if getattr(self, "embed_ids", None) is None or self.embed_ids.nbytes < 4 * size:
            self.embed_ids = Buffer(4 * max(size, self.chunk))
            self.embed_ids_host = pinned((max(size, self.chunk),), np.int32)
        ids = self.embed_ids_host[:size]
        ids[:len(tokens)] = tokens
        ids[len(tokens):] = -1
        _check(lib().rc_gpu_h2d(self.embed_ids.ptr, ids.ctypes.data, 4 * size))
        x = g.mirror.buffer_of(prog.names["x"]).ptr
        _check(lib().rc_gpu_embed_q4_rows(self.head.ptr, self.embed_ids.ptr, x, size,
                                      model.cfg.hidden_size,
                                      float(np.float32(model.cfg.embed_scale))))
        return True

    def _load_head_low(self, rows):
        """_load_head with little memory (the 12B on a card of 10 GB after a
        long prompt): copy the rows that the head reads into a buffer of
        their own, then let programs go, the last one too (its output is
        that copy now), until the head fits."""
        step = self.model.cfg.hidden_size * 4
        if getattr(self, "xkeep", None) is None:
            self.xkeep = Buffer(MT_CPU * step, "weights")
        _check(lib().rc_gpu_d2d(self.xkeep.ptr, self.last - (rows - 1) * step, rows * step))
        self.last = self.xkeep.ptr + (rows - 1) * step
        while True:
            try:
                self._load_head()
                return
            except MemoryError:
                if not self._evict(last=True):
                    raise

    def _run_head(self, rows):
        """Run the head on the last rows into self.out. The first call copies
        the head to the GPU; when it does not fit even after the programs
        went (_load_head_low: the 26B on a card of 4 GB after a prompt of
        48K tokens, where the copy needs 0.61 GB in one block), the CPU runs
        it (_cpu_head) until the GPU has room. Return the vocabulary size."""
        model, cfg = self.model, self.model.cfg
        assert 1 <= rows <= min(self.rows, MT_CPU)
        if self.head is None and not getattr(self, "head_cpu", False):
            try:
                try:
                    self._load_head()
                except MemoryError:
                    self._load_head_low(rows)
            except MemoryError:
                if getattr(self, "out", None) is None:
                    raise
                self.head_cpu = True
        elif self.head is None and mem().room() >= self.head_bytes:
            # the CPU head, and the GPU has room again (a smaller cache): a
            # try with no program let go
            try:
                self._load_head()
                self.head_cpu = False
            except MemoryError:
                pass
        if self.head is None:
            vocab = self._cpu_head(rows)
        else:
            vocab = self.host_logits.shape[1]
            cap = float(cfg.final_logit_softcapping or 0.0)
            step = cfg.hidden_size * 4
            _check(self.head_fn(self.head.ptr, self.last - (rows - 1) * step, self.out.ptr,
                                vocab, cfg.hidden_size, cap, rows))
        if self.hot_cache is not None and self.hot_cache.due:
            self.hot_cache.observe()
        return vocab

    def _cpu_head(self, rows):
        """The head on the CPU (Model.logits of the rows of the hidden state,
        with the soft cap) into self.out, so that argmax and topk read it
        there as from the GPU head. Return the vocabulary size."""
        cfg = self.model.cfg
        step = cfg.hidden_size * 4
        x = np.empty((rows, cfg.hidden_size), np.float32)
        _check(lib().rc_gpu_d2h(x.ctypes.data, self.last - (rows - 1) * step, rows * step))
        lg = np.ascontiguousarray(self.model.logits(x), np.float32)
        vocab = lg.shape[1]
        self.host_logits[:rows] = lg
        _check(lib().rc_gpu_h2d(self.out.ptr, self.host_logits.ctypes.data, rows * vocab * 4))
        return vocab


def offload(model, experts_gb=0.0, resident=False):
    """Put the weights of a Model outside the experts, and its output head, on
    the GPU. The decode steps of one token then run there (RECLANK_GPU=1).

    experts_gb 0 keeps every expert on the CPU. A value above 0 also puts the
    most used experts on the GPU, up to that many GB. None takes the default
    budget of ModelGPU. The function copies the weights now, so the first
    request does not wait for them. Return the ModelGPU.

    For an E4B model, put the whole model on the GPU (E4BGPU). experts_gb
    has no effect, because the E4B has no experts. Return the E4BGPU.

    resident puts every expert on the GPU (ModelGPU); experts_gb then has no
    effect. A dense model is resident anyway.
    """
    from reclank.models.gemma4 import model as model_mod
    os.environ["RECLANK_GPU"] = "1"
    if hasattr(model, "mode"):
        from reclank.models.gemma4 import e4b as e4b_mod
        e4b_mod._GPU = True
        g = E4BGPU(model)
        g.logits()      # copies the head to the GPU
        model._gpu, model._gpu_cache, model._gpu_xn = g, None, None
        return g
    model_mod._GPU = True
    hot = {}
    if experts_gb is None:
        hot = None
    elif experts_gb > 0 and model.cfg.enable_moe_block and not resident:
        counts = hot_counts(model)
        hot = pick_hot(model, counts, experts_gb * 1e9) if counts is not None else {}
    g = ModelGPU(model, hot=hot, resident=resident)
    g.logits()          # copies the head to the GPU
    model._gpu = g
    model._gpu_cache = None
    model._gpu_xn = None
    return g


def describe(g):
    """Return one line about the GPU part of a ModelGPU or an E4BGPU."""
    free, total = mem_info()
    if not hasattr(g, "hot"):
        return "GPU: %.2f GB of weights and buffers; %.1f of %.1f GB free" % (
            g.g.mirror.nbytes() / 1e9, free / 1e9, total / 1e9)
    pool = getattr(g, "pool", None)
    hot = int((pool.owner >= 0).sum() + (pool.owner == -4).sum()) if pool is not None else \
        sum(len(v) for v in g.hot.values())
    return ("GPU: %.2f GB of weights and buffers, %d hot experts%s, head %.2f GB; "
            "%.1f of %.1f GB free" % (g.g.mirror.nbytes() / 1e9, hot,
                                       " in the pool (%.2f GB)" % (pool.nbytes() / 1e9) if pool is not None else "",
                                       (g.head.nbytes if g.head else 0) / 1e9,
                                       free / 1e9, total / 1e9))


def room(g):
    """The bytes that the cache and the programs of a ModelGPU or an E4BGPU
    can still take: the free memory above the reserve of GpuMem and the pool
    of hot experts (it gives way), less the head when it is not on the GPU
    yet (the first prompt copies it)."""
    used, _ = mem().usage()
    free = mem_info()[0] - mem().c_reserve + used.get("pool", 0)
    model = getattr(g, "model", None)
    if getattr(g, "head", True) is None and model is not None:
        w = model._embed_q6k_bytes if model._embed_q6k is not None else model._embed_q
        free -= getattr(w, "nbytes", 0) or 0
    return max(0, free)


# ---- the MTP drafter on the GPU (SPLIT_PLAN.md, phase 5) --------------------

def quantize_q4_0(w):
    """Quantize a float32 matrix to int4 blocks with float16 scales. Return
    (blocks, scales), as ops.quantize_int4 does: the scale of the QAT grid
    when the block has one. The float32 scales equal the float16 scales of the
    blocks, which the GPU kernels read."""
    from reclank.core import ops
    blocks, scale = ops.quantize_int4(np.asarray(w, dtype=np.float32))
    scale = scale.astype(np.float16).astype(np.float32)
    return np.ascontiguousarray(blocks), scale


def compile_drafter(a, target_cfg, cache_form="e4b", rot=(False, False), prob=False):
    """Compile one draft step of the assistant a (an Assistant with float32
    weights) for the GPU. The steps are those of Assistant.step and of its
    head: the centroid head (E4B), or the full head and an argmax (26B).

    cache_form is the cache of the target: "e4b", the float cache of the E4B
    (heads, positions, head_dim), or "qc", the int16 cache of the 26B on the
    GPU (GPUKV). For "qc" the step binds the addresses of the first key row
    and the count of the rows (the slots kq.s, ks.s, vq.s, vs.s, n.s and the
    same with .f), as Assistant._attention finds them.

    The input "xin" holds the embedding of the token and the hidden state h,
    with the backbone size each. The step writes the next h into the second
    half of xin, so the next draft step finds it there. "token" gets the
    draft token. The attention of a layer reads the cache of the target:
    the query at pos sees the rows before pos. The record GP_ATTN_F32H sees
    the rows to its pos operand, so the step gives it pos - 1 (the slot
    "pd"), and a window one smaller. prob: "prob" gets the softmax
    probability of the draft token (among the tokens of the chosen centroids
    with the centroid head), for RECLANK_MTP_STOP.
    """
    cfg = a.cfg
    bk = a.backbone
    c = P.Compiler(a)
    xin = np.zeros((1, 2 * bk), dtype=np.float32)
    c.env["xin"] = xin
    q4 = lambda w: quantize_q4_0(w)  # noqa: E731
    u = P.k_int4(c, q4(a.pre), xin)
    tplan = {True: next(p for p in target_cfg.plan if p.is_sliding),
             False: next(p for p in target_cfg.plan if not p.is_sliding)}
    for i, w in enumerate(a.layers):
        plan = cfg.plan[i]
        kind = "s" if plan.is_sliding else "f"
        hd, nq = plan.head_dim, plan.num_q_heads
        h = P.k_rms_norm(c, u, w["input_layernorm"])
        q = P.k_int4(c, q4(w["self_attn.q_proj"]), h)
        q = P.k_rms_norm_rows(c, q, hd, w["self_attn.q_norm"])
        P.k_rope(c, q, None, c.p.slot("cos." + kind), c.p.slot("sin." + kind), i)
        att = c.buffer((1, nq * hd))
        window = plan.sliding_window or 0
        kvh = tplan[plan.is_sliding].num_kv_heads
        if cache_form in ("qc", "q8", "qv"):
            sl = lambda n: c.p.slot("%s.%s" % (n, kind))  # noqa: E731
            op = {"q8": P.ATTN_Q8, "qv": P.ATTN_V8}.get(cache_form, P.ATTN_QC)
            if rot[0]:      # the rotated keys of the target (rq8): the query too
                c.p.emit(P.TQ_ROT, q, q.size // 32, 0)
            c.p.emit(op, q, sl("kq"), sl("ks"),
                     sl("vq"), sl("vs"), c.p.slot("scores"), att, nq, kvh, hd, sl("n"))
            if rot[1]:      # rotated values (rq8, k16vr8): the output back
                c.p.emit(P.TQ_ROT, att, att.size // 32, 1)
        else:
            c.p.emit(P.ATTN_F32H, q, c.p.slot("k." + kind), c.p.slot("v." + kind),
                     c.p.slot("scores"), att, nq, kvh, hd, 1,
                     c.p.slot("pd"), c.p.slot("hs." + kind), max(0, window - 1), 1)
        o = P.k_int4(c, q4(w["self_attn.o_proj"]), att)
        u = P.k_add(c, u, P.k_rms_norm(c, o, w["post_attention_layernorm"]))
        m = P.k_rms_norm(c, u, w["pre_feedforward_layernorm"])
        g = P.k_int4(c, q4(w["mlp.gate_proj"]), m)
        up = P.k_int4(c, q4(w["mlp.up_proj"]), m)
        mm = P.k_int4(c, q4(w["mlp.down_proj"]), P.k_mul_v(c, P.k_gelu(c, g), up))
        u = P.k_add(c, u, P.k_rms_norm(c, mm, w["post_feedforward_layernorm"]))
        u = P.k_mul(c, u, w["layer_scalar"])
    un = P.k_rms_norm(c, u, a.norm)
    hnext = P.k_int4(c, q4(a.post), un)
    token = np.zeros(1, dtype=np.int32)
    pr = (np.zeros(1, dtype=np.float32),) if prob else ()
    if a.centroids is not None:
        n_cent, per = a.order.shape
        cent = np.ascontiguousarray(a.centroids, dtype=np.float32)
        clog = c.buffer(n_cent)
        c.p.emit(P.F32_LINEAR, un, cent, clog, n_cent, cfg.hidden_size)
        c.p.emit(P.DRAFT_HEAD, un, clog, np.ascontiguousarray(a.order, dtype=np.int32),
                 np.ascontiguousarray(a.head_f32, dtype=np.float32), c.buffer(a.top_k * per),
                 np.zeros(a.top_k, dtype=np.int32), token, n_cent, per, a.top_k,
                 cfg.hidden_size, *pr)
    else:
        # The full head: the int4 product with the table of the drafter, and
        # the best token. The assistant has no soft cap.
        logits = P.k_int4(c, q4(a.head), un)
        c.p.emit(P.ARGMAX, logits, logits.size, token, *pr)
    c.p.emit(P.COPY, hnext, xin[:, bk:], bk * 4)
    c.env["token"] = token
    if prob:
        c.env["prob"] = pr[0]
    return c.p.finish()


# 1: the draft steps of a round run one after another on the GPU, with no
# copy to the host between them (GPUDrafter._draft_chain), when the target
# keeps its token table on the GPU as a Q4_0 head. 0 keeps a copy of the
# token and of the embedding for each step.
DRAFT_CHAIN = os.environ.get("RECLANK_GPU_DRAFT_CHAIN", "1") == "1"


class GPUDrafter:
    """The MTP drafter on the GPU, for the E4B model or the 26B model. draft()
    has the arguments of Assistant.draft, so mtp_stream takes either.

    The drafter reads the cache of the target on the GPU. For the E4B, those
    are the buffers that its shared layers reuse. For the 26B, they are the
    int16 buffers of two layers of GPUKV (assistant.shared_layers). Thus it
    needs no copy of the cache to the host. Its weights are its own int4
    blocks with float16 scales (quantize_q4_0), so its drafts can differ a
    little from the CPU drafter. The target checks each draft, so the text
    does not change.
    """

    on_gpu = True

    def __init__(self, path, target, weights=None, stop=None):
        from reclank.core.speculative import MTP_STOP
        from reclank.models.gemma4.assistant import Assistant
        self.a = Assistant(path, dtype="f32", weights=weights)
        self.target = target
        self.p_min = 0.0
        # RECLANK_MTP_STOP: the drafts end at the first where the product of
        # their probabilities falls under it (the verify takes that one too)
        self.stop = MTP_STOP if stop is None else stop
        self.e4b = hasattr(target, "mode")      # E4B has a mode; Model does not
        from reclank.models.gemma4 import model as _model_mod
        form = "e4b" if self.e4b else {"int8": "q8", "k16v8": "qv"}.get(
            _model_mod.kv_base(_model_mod.KV_FORM), "qc")
        self.kesz = 1 if form == "q8" else 2
        self.vesz = 1 if form in ("q8", "qv") else 2
        self.prog = compile_drafter(self.a, target.cfg, form,
                                    rot=(False, False) if self.e4b else _model_mod.kv_rot(),
                                    prob=self.stop > 0)
        self.g = None

    def draft(self, target, token, h, pos, cache, n, eos_ids=()):
        target._gpu_attach(cache)
        tg = target._gpu
        if self.g is None:
            # Its arrays share the mirror of the target: keep them on the GPU
            # when the target frees a program (ProgramLRU), and free the
            # programs of the target used least recently when the memory
            # runs out.
            tg.__dict__.setdefault("extra_progs", []).append(self.prog)
            while True:
                try:
                    self.g = GPUProgram(self.prog, mirror=tg.g.mirror)
                    break
                except MemoryError:
                    # The drafts read h from the host, so the output of the
                    # last group can go too.
                    if not (hasattr(tg, "_evict") and tg._evict(last=True)):
                        raise
        a, bk = self.a, self.a.backbone
        kw = {"pd": pos - 1}
        if not self.e4b:
            from reclank.models.gemma4.assistant import shared_layers
            layers = dict(zip(("s", "f"), shared_layers(target.cfg)))
        for kind, sliding in (("s", True), ("f", False)):
            plan = next(p for p in a.cfg.plan if p.is_sliding == sliding)
            if self.e4b:
                store = cache.shared["sliding_attention" if sliding else "full_attention"]
                kw["k." + kind], kw["v." + kind] = store[0], store[1]
                kw["hs." + kind] = store[0].shape[2]       # position-major
            else:
                # The rows of positions lo to pos - 1 of the target layer, as
                # Assistant._attention reads them.
                i = layers[kind]
                kv = tg.kv
                base = kv.base[i]
                lo = max(0, pos - a.cfg.sliding_window + 1 - base) if sliding else 0
                tp = target.cfg.plan[i]
                per = tp.num_kv_heads * tp.head_dim
                b = kv.bufs[i]
                assert (kv.kesz, kv.vesz) == (self.kesz, self.vesz), \
                    "the drafter and the GPU cache differ in form"
                kw.update({"kq." + kind: b["kq"].ptr + lo * kv.kesz * per,
                           "ks." + kind: b["ks"].ptr + lo * per // 8,
                           "vq." + kind: b.get("vq", b["kq"]).ptr + lo * kv.vesz * per,
                           "vs." + kind: b["vs"].ptr + lo * per // 8,
                           "n." + kind: pos - base - lo})
            cos, sin = a._cos_sin(plan, pos)
            kw["cos." + kind] = np.ascontiguousarray(cos, dtype=np.float32)
            kw["sin." + kind] = np.ascontiguousarray(sin, dtype=np.float32)
        kw["scores"] = np.empty(max(p.num_q_heads for p in a.cfg.plan) * (pos + 1), np.float32)
        self.g.bind(kw, tg.cache if self.e4b else None)
        xin = self.g.mirror.buffer_of(self.prog.names["xin"]).ptr
        tokbuf = self.g.mirror.buffer_of(self.prog.names["token"])
        hh = np.ascontiguousarray(h, dtype=np.float32).reshape(-1)
        _check(lib().rc_gpu_h2d(xin + bk * 4, hh.ctypes.data, bk * 4))
        if (DRAFT_CHAIN and getattr(tg, "head", None) is not None
                and getattr(tg, "head_fn", None) is lib().rc_gpu_q4_head):
            return self._draft_chain(target, tg, token, n, xin, tokbuf, eos_ids)
        out = []
        got = np.zeros(1, dtype=np.int32)
        pg = np.zeros(1, dtype=np.float32)
        prod = 1.0
        for _ in range(n):
            emb = np.ascontiguousarray(target.embed([token]), dtype=np.float32).reshape(-1)
            _check(lib().rc_gpu_h2d(xin, emb.ctypes.data, bk * 4))
            self.g.run()
            tokbuf.download(got)
            token = int(got[0])
            out.append(token)
            if token in eos_ids:
                break
            if self.stop > 0:
                self.g.mirror.buffer_of(self.prog.names["prob"]).download(pg)
                prod *= float(pg[0])
                if prod < self.stop:
                    break
        return out

    def _draft_chain(self, target, tg, token, n, xin, tokbuf, eos_ids):
        """The n draft steps with no copy to the host between them: the
        embedding of each draft token comes from the Q4_0 head of the target
        on the GPU (gg_embed_q4, the values of target.embed), and each token
        goes to a device array. One copy at the end brings the n tokens. The
        drafts after an end token are dropped. With stop, each step copies
        the probability of its draft to the host (4 bytes), and the steps end
        at the first draft where the product of the probabilities falls
        under it."""
        bk = self.a.backbone
        if getattr(self, "chain_out", None) is None:
            self.chain_out = Buffer(4 * 64)
            self.chain_host = pinned((64,), np.int32)
            self.chain_prob = pinned((1,), np.float32)
        n = min(n, 64)
        scale = float(np.float32(target.cfg.embed_scale))
        first = np.array([token], dtype=np.int32)
        _check(lib().rc_gpu_h2d(tokbuf.ptr, first.ctypes.data, 4))
        pbuf = self.g.mirror.buffer_of(self.prog.names["prob"]).ptr if self.stop > 0 else None
        prod = 1.0
        steps = n
        for j in range(n):
            _check(lib().rc_gpu_embed_q4(tg.head.ptr, tokbuf.ptr, xin, bk, scale))
            self.g.run()
            _check(lib().rc_gpu_d2d(self.chain_out.ptr + 4 * j, tokbuf.ptr, 4))
            if pbuf is not None:
                _check(lib().rc_gpu_d2h(self.chain_prob.ctypes.data, pbuf, 4))
                prod *= float(self.chain_prob[0])
                if prod < self.stop:
                    steps = j + 1
                    break
        _check(lib().rc_gpu_d2h(self.chain_host.ctypes.data, self.chain_out.ptr, 4 * steps))
        out = []
        for t in self.chain_host[:steps]:
            out.append(int(t))
            if int(t) in eos_ids:
                break
        return out
