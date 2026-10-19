"""Qwen3.8-Flash-Next on the GPU, with the experts split (QWEN38_PLAN.md,
phase 5).

Qwen4GPU is QwenGPU (reclank/models/qwen/gpu35.py) with the program of
compile_qwen4_step. The design stays:

- The dense part runs on the GPU: the products (Q8_0 as rows, F32), the
  gated residual and the n-gram layer (the records of backends/cpu/csrc/common/hyperconn.c),
  the DeltaNet, and QSA (QSA_SELECT, and one ATTN_QSA for each query).
- The GPU holds some experts of each layer (HotCache changes them). The
  CPU computes the other selected experts at the same time.
- The head (Q8_0) runs on the GPU in its own program (one for each count
  of rows).
- The MTP layer (when the model has it) is a program of its own
  (compile_qwen4_step with mtp), with its cache on the GPU. Its experts
  are split too, with its own slots (RECLANK_GPU_MTP_SLOTS, or else
  twice the slots of a layer, at most 0.5 GB; the budget of the hot
  experts does not count them).
- HotCache scores the steps, and also the tokens of an MTP verify group
  that stay (commit) and each row of the MTP layer. A group copies the
  selection of each layer to an array (sel_of) for that.

- A prompt of at least MIX_MIN tokens runs in mixed groups of mix_size
  rows (RECLANK_GPU_MIX, RECLANK_GPU_MIX_MIN). In each layer, after the
  router, GP_MOE_PLAN (backends/cpu/csrc/common/moe.c) splits the experts: the GPU takes the
  experts with the most tokens, and a worker thread copies them to one
  buffer (as much as the free memory holds); the CPU takes the rest at the
  same time (GP_CPU_START); the hot experts stay on the GPU. The split keeps
  the time of the copies and the time of the CPU about equal.

The host makes the inputs of a run: the embeddings in each stream and the
rows of the n-gram table.

    m = Qwen4CPU(path, mtp=mtp_path)
    g = Qwen4GPU(m, hot_gb=1.0)
    cache = Qwen4Cache(m.cfg, 4096)
    g.attach(cache)
    g.prefill(ids); logits = g.logits()
    g.step(token, pos); logits = g.logits()
"""
from __future__ import annotations

import os

import numpy as np

from reclank.ir import alloc
from reclank.ir import program as P
from reclank.backends.cuda.runtime import GPUProgram, _check, lib
from reclank.backends.cuda.gpumm import Buffer, pinned
from reclank.models.qwen.gpu35 import MIX_MIN, MT, QwenGPU, _DevCache, _fuse
from reclank.models.qwen.qwen35 import media_inputs, rope_positions
from reclank.models.qwen.qwen38 import compile_qwen4_step

# RECLANK_GPU_RESIDENT_BIAS (0: off): the router of a step, a verify group,
# and the MTP layer (not a prompt group) adds this to the logit of each
# expert on the GPU (hot or warm, HotCache) when it chooses the top k; the
# weights of the chosen experts stay those of the plain softmax. Fewer cold
# experts for the CPU, at a cost in the fidelity to the model (the choice
# then depends on what the GPU holds).
RESIDENT_BIAS = float(os.environ.get("RECLANK_GPU_RESIDENT_BIAS", "0") or 0)
# RECLANK_GPU_HOT_SWAP (0: off): in the same programs, a near tie at the edge
# of the top k goes to the GPU: from the lightest, a chosen expert not on
# the GPU gives its place to the best expert on the GPU that was not chosen
# when that one has at least this times its weight (0.9: within 10%); the
# first cold expert with no such expert ends it. HotCache learns from the
# choice before the swaps, so what it holds follows the model's routing.
HOT_SWAP = float(os.environ.get("RECLANK_GPU_HOT_SWAP", "0") or 0)

# The rows of a prompt in the MTP layer (mtp_rows): with RECLANK_GPU_MTP_MIX
# (e.g. 1024; 0, the default: off) mixed groups of up to that many rows (as
# the prompt's groups: the experts by expert on the tensor cores, the cold
# ones on the CPU), else groups of 256 as a step (the MoE a GEMV for each row
# and expert). EXL3 5.05, a prompt of 16K tokens: the layer's 2.2 s to 1.65 s
# (1024) or 1.48 s (4096), the drafts accepted as before; at 64K no gain (the
# dense attention of the layer is most of it), and the verify groups after
# 16K 35.5 -> 37.2 ms (the memory of the second mixed program, from the pool?).
MTP_MIX = int(os.environ.get("RECLANK_GPU_MTP_MIX", "0"))


class _Emit4:
    """The model as compile_qwen4_step sees it for a GPU program."""

    # compile_qwen4_step: buffers of each layer, for alloc.pack
    pack_temps = alloc.PACK

    def __init__(self, dev, t, verify, fetch, mix=False, bias=False, prefetch=True):
        self.dev, self.m, self.t, self.fetch, self.mix = dev, dev.model, t, fetch, mix
        self.prefetch = prefetch    # a mixed group: the copies of the layer before its plan
        dev._lpool = None
        self.bias = bias        # a step, a verify group, or the MTP layer: RESIDENT_BIAS
        self.colds = {}         # layer -> the cold list of a step whose selection the router writes

    def __getattr__(self, name):
        return getattr(self.m, name)

    def layer_begin(self, prog, i):
        self.dev._layer_begin(prog)

    def dense_rot(self, gname):
        return self.m.dense_rot(gname)

    def emit_lin4(self, prog, xb, gname, out):
        m = self.m.K(gname)
        w, type_ = self.dev.dense(gname)
        prog.emit(P.KQ_LINEAR, xb["xq"], xb["xs"], xb["xm"], xb["src"], w, type_, m.rows, m.cols,
                  xb["t"], out)

    def moe_scratch4(self, t):
        return None

    def router_bias4(self, i):
        """The extra operands of ROUTER_TOPK of layer i: the slot table of
        its experts, RESIDENT_BIAS, HOT_SWAP, and the array of the choice
        before the swaps (for HotCache: the selection part of the cold list
        of a step, or the row of sel_of of a group), or none."""
        if (RESIDENT_BIAS == 0 and HOT_SWAP == 0) or not self.bias or i not in self.dev.stores:
            return ()
        sel = None
        if HOT_SWAP > 0:
            k = self.cfg.top_k
            if self.t == 1:
                cold = self.colds[i] = np.zeros(2 * k + 1, dtype=np.int32)
                sel = cold[k + 1:]
            else:
                sel = self.dev.sel_of(self.t)[i]
        return (self.dev.stores[i]["slots"], RESIDENT_BIAS, HOT_SWAP, sel)

    def emit_moe4(self, prog, i, xb, idx, val, slog, scratch, out):
        if self.fetch:
            self.dev._moe_fetch(prog, i, xb["src"], idx, val, slog, out)
            return
        if self.mix:
            self.dev._moe_mix(prog, i, xb["src"], idx, val, slog, out, prefetch=self.prefetch)
            return
        # with HOT_SWAP the router wrote the choice for HotCache already
        swapped = HOT_SWAP > 0 and self.bias and i in self.dev.stores
        if self.t > 1 and not swapped:
            sel = self.dev.sel_of(self.t)
            prog.emit(P.COPY, idx, sel[i], idx.nbytes)
        self.dev._moe_split(prog, i, xb["src"], idx, val, slog, out, cold=self.colds.pop(i, None))

    def emit_attn_qsa(self, prog, q, base, scores, out, t, pos, sel, cnt, maxsel, form=0):
        cfg = self.m.cfg
        if t > MT and os.environ.get("RECLANK_GPU_ATTN_MT", "1") != "0":
            # A large group: one record (k_attn_qsa_mt; RECLANK_GPU_ATTN_MT=0 for a test).
            # cnt -1 for each query (the dense layer: the MTP layer): maxsel
            # 0 says so to the GPU (k_attn_dense_tc; no row reads sel then)
            if (cnt == -1).all():
                maxsel = 0
            prog.emit(P.ATTN_QSA, q, *base, scores, out, cfg.num_heads, cfg.num_kv_heads,
                      cfg.head_dim, t, pos, sel, cnt, maxsel, form, prog.slot("hs"))
            return
        # One record for each query: the kernel of a step (the same bits).
        for j in range(t):
            pj = prog.temp()
            prog.emit(P.S_ADD, pj, pos, j)
            prog.emit(P.ATTN_QSA, q[j:j + 1], *base, scores, out[j:j + 1], cfg.num_heads,
                      cfg.num_kv_heads, cfg.head_dim, 1, pj, sel[j:j + 1], cnt[j:j + 1], maxsel,
                      form, prog.slot("hs"))


class _DevCache4(_DevCache):
    """_DevCache with the state of the n-gram layer and of the indexer, and
    the cache of the MTP layer."""

    def attach(self, cache):
        super().attach(cache)
        self.add(list(cache.ple_conv.values()) + list(cache.idx_k.values())
                 + list(cache.idx_blk.values()))

    def buffer(self, a):
        return self.bufs[id(a)][1]


class Qwen4GPU(QwenGPU):
    """Qwen3.8-Flash-Next (a Qwen4CPU) on the GPU; see the module text."""

    def __init__(self, model, hot_gb=None, counts=None, graph=True, ctx=None):
        self.head_progs = {}
        self.mtp_progs = {}
        self.sel_bufs = {}
        self._qpos_version = -1
        self._qsa_buf = None        # the scratch of QSA_SELECT (_params)
        self.mcache = None
        L = model.cfg.num_hidden_layers
        self.has_mtp = "blk.%d.nextn.eh_proj.weight" % L in model.g.tensors
        self.model = model
        # RQ8_0 experts: the MoE kernels of the GPU rotate the act (Qwen4CPU
        # sets those of the CPU)
        lib().rc_gpu_set_moe_rot(int(getattr(model, "rot_experts", False)))
        super().__init__(model, hot_gb=hot_gb, counts=counts, graph=graph, ctx=ctx)
        self.cache_dev = _DevCache4()
        self.cache_dev.before = self._share_before

    def _cache_types(self):
        from reclank.models.qwen.qwen38 import Qwen4Cache, Qwen4MTPCache
        return (Qwen4Cache, Qwen4MTPCache) if self.has_mtp else (Qwen4Cache,)

    def _more_stores(self):
        if self.has_mtp:
            L = self.cfg.num_hidden_layers
            per = sum(self._expert_nb(self.model.M("layers.%d.mlp.switch_mlp.%s" % (L, x)))
                      for x in ("gate_proj", "up_proj", "down_proj"))
            n = int(os.environ.get("RECLANK_GPU_MTP_SLOTS",
                                   min(2 * self.n_slots, int(0.5e9 // per))))
            st = self.stores[L] = self._store(L, np.arange(min(n, self.E)))
            st["ip"] = np.zeros(2 * self.cfg.top_k + 1, np.int32)   # a step does not run it

    def _before_hot(self):
        if self.has_mtp:
            self._mtp_group(1)

    # ---- the hooks of the mixed groups (QwenGPU.mix) ----

    def _compile_mix(self, t):
        self._pool = {}
        return self._pack(_fuse(compile_qwen4_step(_Emit4(self, t, False, False, mix=True), t), t))

    def _ensure_head(self):
        self._head(1)

    def _pinned_arrays(self):
        """The stores of the experts, and the weights of the head: a program
        of the head that the LRU frees gives back its buffers, not the weights
        (973 MB in BF12), which a new head program would need again when no
        program can give memory (a mixed group of 4096 rows with the NVFP4
        experts: cudaMalloc of the head failed)."""
        out = super()._pinned_arrays()
        if getattr(self, "_head_w", None) is not None:
            out.append(self._head_w)
        return out

    def _scratch_names(self):
        return ("scores", "qsa_scratch")

    def _dense_bytes(self, bf16_as_q8=None):
        """The bytes of the dense tensors on the GPU (bfloat16 counts as Q8_0
        when the model requantizes them: Qwen4CPU dense; BF12 as the mode of
        RECLANK_DENSE)."""
        from reclank.formats.gguf import tensor_bytes
        g = self.model.g
        q8 = getattr(self.model, "dense", "bf16") in ("q8", "rq8") if bf16_as_q8 is None else bf16_as_q8
        n = 0
        skip = ("token_embd.weight", "output.weight", "per_layer_token_embd.weight",
                "per_layer_token_embd.scale")
        L = self.model.cfg.num_hidden_layers
        mode = os.environ.get("RECLANK_DENSE", "bf12")
        for name, (dims, t, _o) in g.tensors.items():
            if "_exps" in name or name in skip or name.startswith("blk.%d." % L):
                continue
            if t == 57 and mode in ("q8", "rq8", "bf16"):
                t = 30 if mode == "bf16" else 8     # Qwen4CPU.K: BF12 as another mode
            n += tensor_bytes(dims, 8 if (t == 30 and q8) else t)
        return n

    def _compile(self, t, verify=False, fetch=False):
        self._pool = {}
        prog = compile_qwen4_step(_Emit4(self, t, verify, fetch, bias=(t == 1 or verify) and not fetch),
                                  t, verify=verify)
        if fetch:
            first = []
            for i in (0, 1):
                ranges = self.tables[i][3]
                first.append((P.FETCH, [prog._enc(ranges), prog._enc(len(ranges)),
                                        prog._enc(i), prog._enc(i % 2)]))
            prog.recs[:0] = first
        prog = self._pack(_fuse(prog, t))
        # the routers' choices of a group (sel_of at its compile: lanes.Lane compiles with its own)
        prog.sel_buf = self.sel_bufs.get(t)
        return prog

    def _params(self, pos, t, nreal):
        cfg = self.cfg
        kw = super()._params(pos, t, nreal)
        cache = self.cache
        for i in cfg.ple_layers:
            kw["pleconv.%d" % i] = cache.ple_conv[i]
        for i, a in cache.idx_k.items():
            kw["idxk.%d" % i], kw["blk.%d" % i] = a, cache.idx_blk[i]
        # The scratch of QSA_SELECT: nbmax keys of 8 bytes for each query of
        # a launch (gg_qsa_rows of them; backends/cuda/csrc QSA_ROWS). A query at
        # row p reads (p + 1) / ratio blocks, so nbmax follows the rows of
        # this run, not the context. One buffer serves all the programs, and
        # it grows (twice the size, at most that of the context).
        ratio = min(r for r in cfg.compress_ratios if r)
        nbmax = (pos + t) // ratio + 1
        rows = min(t, lib().rc_gpu_qsa_rows())
        extra = lib().rc_gpu_qsa_extra()            # the queries of k_qsa_score_tc, first
        need = extra + rows * nbmax * 8 + 64
        if self._qsa_buf is None or self._qsa_buf.nbytes < need:
            most = extra + rows * (cache.max_len // ratio + 1) * 8 + 64
            grow = max(need, min(2 * self._qsa_buf.nbytes, most) if self._qsa_buf is not None else 0)
            if self._qsa_buf is not None:
                self._qsa_buf.free()
                self._qsa_buf = None
            self._qsa_buf = self._alloc(max(grow, 1 << 20))
        kw["qsa_scratch"] = self._qsa_buf.ptr
        kw["nbmax"] = nbmax
        # The M-RoPE positions of the rows (after an image): a device copy,
        # copied again when set_rope changes them.
        kw["qpos"] = 0
        if cache.qpos is not None:
            if id(cache.qpos) not in self.cache_dev.bufs:
                self.cache_dev.add([cache.qpos])
            elif self._qpos_version != cache.qpos_version:
                self.cache_dev.buffer(cache.qpos).upload(cache.qpos)
            self._qpos_version = cache.qpos_version
            kw["qpos"] = cache.qpos
        return kw

    # ---- the runs ----

    def _inputs(self, prog, g, ids, pos=0, media=None):
        """The inputs of a run of the tokens ids: the embeddings in each
        stream, and the rows of the n-gram table (the host keeps its
        last tokens). media: the spans of the images (cache positions)."""
        cfg = self.cfg
        t = len(ids)
        x = media_inputs(self.model.embed(ids), pos, media)
        H = prog.names["H"]
        H[:t] = np.repeat(x[:, None, :], cfg.hc_count, axis=1).reshape(t, -1)
        H[t:] = 0.0
        g.upload("H")
        if cfg.ple_layers:
            ple = prog.names["ple"]
            ple[:t] = self.model.ple_rows(ids, self.cache)
            ple[t:] = 0.0
            g.upload("ple")

    def step(self, token, pos):
        self._warm_on()
        if self.hot_cache is not None:
            self.hot_cache.prepare()
        self._inputs(self.prog, self.g, [token])
        self.g.bind(self._params(pos, 1, 1), self.cache_dev, scratch=("scores", "qsa_scratch"))
        self.g.run()
        self.cache.n = pos + 1
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.last_prog = (self.prog, self.g)
        self.rows = 1
        if self.hot_cache is not None:
            self.hot_cache.due = True

    def group(self, tokens, pos, size=None, verify=False, fetch=False, media=None):
        """Run tokens from position pos as one group (see QwenGPU.group).
        Return the input of the head of each token."""
        t = len(tokens)
        size = size or t
        prog, g = self._group(size, verify, fetch)
        self._warm_on()
        if self.hot_cache is not None:
            self.hot_cache.prepare(wait=fetch)
        if fetch:
            self._fill_tables()
        if verify:
            snap = (self.cache.ple_ids.copy(),
                    {i: self._download(self.cache.ple_conv[i]) for i in self.cfg.ple_layers})
        self._inputs(prog, g, list(tokens), pos, media)
        g.bind(self._params(pos, size, t), self.cache_dev, scratch=("scores", "qsa_scratch"))
        g.run()
        g.download("xn")
        hidden = self.cfg.hidden_size
        self.last = g.mirror.buffer_of(prog.names["xn"]).ptr + (t - 1) * hidden * 4
        self.last_prog = (prog, g)
        self.rows = t
        if verify:
            self._pending = (prog, pos, list(tokens), snap)
        else:
            self.cache.n = pos + t
        return prog.names["xn"][:t].copy()

    def prefill(self, ids, pos=0, streams=False, media=None):
        """Run a prompt from position pos. Return the input of the head of
        its last token; with streams, the streams after the last layer of
        each token (the input of the MTP layer). media: the spans of the
        images (cache positions)."""
        ids = list(ids)
        c0, h, hs = 0, None, []
        room = self.cache.max_len
        while c0 < len(ids):
            rem = len(ids) - c0
            # The mixed group: mix_size rows, or the smallest of 256, 512, ...
            # that holds the rest and fits in the cache.
            ms = self.mix_size
            size = next((sz for sz in (256, 512, 1024, 2048, 4096)
                         if sz <= ms and sz >= min(rem, ms) and sz <= room - pos - c0), 0)
            # FP4-Local (b12x): a prompt part of more than 64 tokens in a mixed group (A4), as vLLM's
            # prefill rows outside its CUDA graphs of 64 tokens (the plan's 2.2); the rest A16
            mn = 65 if getattr(self, "b12x", False) and "RECLANK_GPU_MIX_MIN" not in os.environ else MIX_MIN
            if ms > 0 and rem >= mn and size:
                n = min(rem, size)
                h = self.mix(ids[c0:c0 + n], pos + c0, size, media)
                if streams:
                    hs.append(self.streams(n))
                c0 += n
                continue
            size, n, fetch = self._sizes(rem, room - pos - c0)
            h = self.group(ids[c0:c0 + n], pos + c0, size, fetch=fetch, media=media)
            if streams:
                hs.append(self.streams(n))
            c0 += n
        return np.concatenate(hs) if streams else h[-1:]

    # ---- the MTP layer ----

    def attach(self, cache):
        """Copy the cache to the GPU, with a new cache of the MTP layer."""
        # the buffers of the last cache stay in the pool until the MTP cache
        # has taken its own (QwenGPU.attach then frees the rest)
        self.cache_dev.attach(cache)
        self.cache = cache
        if self.has_mtp:
            from reclank.models.qwen.qwen38 import Qwen4MTPCache
            self.mcache = Qwen4MTPCache(self.cfg, cache.max_len)
            self.cache_dev.add(self.mcache.kv[self.cfg.num_hidden_layers], fresh=True)
        self.cache_dev.finish()

    def _mtp_group(self, t):
        e = self.mtp_progs.get(t)
        if e is None:
            self._pool = {}
            prog = self._pack(_fuse(compile_qwen4_step(_Emit4(self, t, False, False, bias=True), t,
                                                       mtp=True), t))
            e = self.mtp_progs[t] = (prog, self._gpu_program(prog, ("mtp_progs", t)))
        self._used(t, "mtp_progs")
        return e

    def mtp(self, Hs, ids, pos):
        """The MTP layer on rows j = the streams at pos + j - 1 (Hs, host
        rows) and the token at pos + j. logits() then gives the drafts.
        Return the streams of the layer (host rows)."""
        cfg = self.cfg
        t = len(ids)
        size = next((s for s in (1, 2, 4, 8, 16, 64, 128, 256) if s >= t), None)
        assert size is not None, "a group of the MTP layer has at most 256 rows"
        prog, g = self._mtp_group(size)
        self._warm_on()
        if self.hot_cache is not None:
            self.hot_cache.prepare()
        e, h = prog.names["e"], prog.names["h"]
        e[:t] = self.model.embed(ids)
        e[t:] = 0.0
        h[:t] = Hs.reshape(t, -1)
        h[t:] = 0.0
        g.upload("e")
        g.upload("h")
        L = cfg.num_hidden_layers
        # the rows of the MTP layer have the positions of the rows of the model
        cos, sin = self.model.rope(rope_positions(self.cache, pos, size))
        self.cache_dev.share_tail(pos + size, self.cache_dev.before)
        kw = {"pos": pos, "nreal": t, "cos": np.ascontiguousarray(cos, np.float32),
              "sin": np.ascontiguousarray(sin, np.float32),
              "scores": np.empty(cfg.num_heads * (pos + size) + 64, np.float32)}
        for nm, a in zip(("kq", "ks", "vq", "vs"), self.mcache.kv[L]):
            kw["%s.%d" % (nm, L)] = a
        g.bind(kw, self.cache_dev, scratch=("scores",))
        g.run()
        g.download("H")
        if self.hot_cache is not None:
            if size == 1:
                k = cfg.top_k
                r = [i for i, e2 in enumerate(self.hot_cache.layers) if e2["layer"] == L]
                self.hot_cache.score_rows(r, self.stores[L]["ip"][None, k + 1:2 * k + 1])
            else:
                self._score(size, t, (L,))
        self.last = g.mirror.buffer_of(prog.names["xn"]).ptr + (t - 1) * cfg.hidden_size * 4
        self._mtp_last = self.last      # (the head of the drafts: _draft_head)
        self.rows = t
        return prog.names["H"][:t].copy()

    def mtp_rows(self, Hs, ids, pos):
        """The MTP layer on the rows of a prompt (as mtp, any count): mixed
        groups of up to MTP_MIX rows (the experts on the tensor cores by
        expert, the cold ones on the CPU, as the prompt's groups) where they
        fit, the rest in groups of 256. Return the streams of the layer of
        the last group; logits() then gives the draft of the last row."""
        hm, c0, n = None, 0, len(ids)
        mixed = MTP_MIX >= MIX_MIN and self.mix_size > 0
        while c0 < n:
            rem = n - c0
            if mixed and rem >= MIX_MIN:
                size = next(s for s in (256, 512, 1024, 2048, 4096) if s >= min(rem, MTP_MIX))
                c1 = c0 + min(rem, size)
                hm = self.mtp_mix(Hs[c0:c1], ids[c0:c1], pos + c0, size)
            else:
                c1 = c0 + min(rem, 256)
                hm = self.mtp(Hs[c0:c1], ids[c0:c1], pos + c0)
            c0 = c1
        return hm

    def _mtp_mix_group(self, t):
        """The MTP layer as a mixed group of t rows (mix_progs ("mtp", t)),
        and the room of the pool for its copies (as _mix_group)."""
        key = ("mtp", t)
        e = self.mix_progs.get(key)
        if e is None:
            self.free_ring()
            self._ensure_head()
            self._pool = {}
            prog = self._pack(_fuse(compile_qwen4_step(_Emit4(self, t, False, False, mix=True, prefetch=False),
                                                       t, mtp=True), t))
            e = self.mix_progs[key] = (prog, self._gpu_program(prog, ("mix_progs", key)))
        self._running = ("mix_progs", key)
        self._used(key, "mix_progs")
        self.mix_cap = self._pool_room(("mix_progs", key))
        return e

    def mtp_mix(self, Hs, ids, pos, size):
        """mtp for the rows of a prompt as one mixed group of size rows (no
        prefetch: its plan copies the experts of the layer it gives the
        GPU). The selections of these rows do not score HotCache (as the
        prompt's groups)."""
        cfg = self.cfg
        t = len(ids)
        L = cfg.num_hidden_layers
        try:
            prog, g = self._mtp_mix_group(size)
            if self.hot_cache is not None:
                self.hot_cache.prepare(wait=True)      # the plan reads the slots
            self._nreal_h[0] = t
            e, h = prog.names["e"], prog.names["h"]
            e[:t] = self.model.embed(ids)
            e[t:] = 0.0
            h[:t] = Hs.reshape(t, -1)
            h[t:] = 0.0
            g.upload("e")
            g.upload("h")
            cos, sin = self.model.rope(rope_positions(self.cache, pos, size))
            self.cache_dev.share_tail(pos + size, self.cache_dev.before)
            kw = {"pos": pos, "nreal": t, "cos": np.ascontiguousarray(cos, np.float32),
                  "sin": np.ascontiguousarray(sin, np.float32),
                  "scores": np.empty(cfg.num_heads * (pos + size) + 64, np.float32)}
            for nm, a in zip(("kq", "ks", "vq", "vs"), self.mcache.kv[L]):
                kw["%s.%d" % (nm, L)] = a
            g.bind(kw, self.cache_dev, scratch=("scores",))
            # the blocks of the pool for its copies, after every allocation of the run (_mix_run)
            hc = self._warm_hc()
            if hc is not None:
                self.mix_places = hc.pool_reserve(self.mix_cap)
            else:
                free = np.flatnonzero(self.pool.owner == -1)[:self.mix_cap]
                self.pool.owner[free] = -2
                self.mix_places = free.astype(np.int32)
            self._warm_desc(self.mix_places)
            self.last_run = {"kind": "mtp_mix", "pos": int(pos), "rows": int(t), "size": int(size)}
            g.run()
            g.download("H")
        finally:
            self._running = None
        self.last = g.mirror.buffer_of(prog.names["xn"]).ptr + (t - 1) * cfg.hidden_size * 4
        self._mtp_last = self.last
        self.rows = t
        return prog.names["H"][:t].copy()

    def streams(self, rows):
        """The streams after the last layer of the last rows rows of the
        last run (the input of the MTP layer)."""
        prog, g = self.last_prog
        g.download("H")
        n = self.rows
        return prog.names["H"][n - rows:n].copy()

    def _download(self, a):
        out = np.empty_like(a)
        self.cache_dev.buffer(a).download(out)
        return out

    def commit(self, n):
        """Keep the first n tokens of the last verify group: the DeltaNet and
        HotCache's scores (QwenGPU.commit), then the state of the n-gram layer
        from its copy and the first n inputs of its convolution."""
        prog, pos, ids, (ple_ids, ple_conv) = self._pending
        self._pending = (prog, pos)
        super().commit(n)
        cfg = self.cfg
        ctx = cfg.ple_ngram - 1
        self.cache.ple_ids = np.concatenate([ple_ids, np.asarray(ids[:n], np.int64)])[-ctx:].copy()
        mir = self.g.mirror
        for i, old in ple_conv.items():
            gn = prog.names["gn.%d" % i]
            mir.buffer_of(gn).download(gn)
            new = np.ascontiguousarray(np.concatenate([old, gn[:n]])[-old.shape[0]:])
            self.cache_dev.buffer(self.cache.ple_conv[i]).upload(new)

    def _draft_head(self, x=None):
        """True when the logits are those of the MTP layer's last run (or of host rows x, its drafts)
        and the model has a head of its own for them (FP4-Local: mtp_head, NVFP4)."""
        if "mtp_head.weight" not in self.model.g.tensors or os.environ.get("RECLANK_FP4L_MTP_HEAD", "1") == "0":
            return False
        return x is not None or (getattr(self, "_mtp_last", None) is not None and self._mtp_last == self.last)

    def _head(self, rows, prob=False, draft=False):
        """The program of the head for rows rows (made once); prob: with the
        probability of the best token of each row too; draft: the MTP drafter's head (mtp_head)."""
        key = (rows, "p" if prob else "", "d" if draft else "")
        e = self.head_progs.get(key)
        if e is None:
            cfg = self.cfg
            name = "mtp_head.weight" if draft else "output.weight"
            m = self.model.K(name)
            w, type_ = self.dense(name)
            self._head_w = w        # its device copy stays (_pinned_arrays)
            hp = P.Program()
            # the logits in pinned memory: a row is 1 MB (248320 values), and
            # its copy from the GPU took 0.5 ms more than the head itself
            # into pageable memory (2.0 ms in place of 1.5; 4 rows 4.4 ms)
            hx, out = np.zeros((rows, cfg.hidden_size), np.float32), pinned((rows, m.rows))
            out[:] = 0.0
            # the best token of each row too (GP_ARGMAX, the first index of
            # the largest value, as np.argmax): argmax() downloads only it
            tok = pinned((rows,), np.int32)
            hp.names.update(x=hx, out=out, tok=tok)
            p = None
            if prob:
                p = pinned((rows,), np.float32)
                hp.names["prob"] = p
            hp.emit(P.KQ_LINEAR, None, None, None, hx, w, type_, m.rows, m.cols, rows, out)
            if getattr(cfg, "b12x", False):
                hp.emit(P.BF16_ROUND, out, rows * m.rows)     # (vLLM's heads give bfloat16 logits)
            for j in range(rows):
                hp.emit(P.ARGMAX, out[j], m.rows, tok[j:j + 1], *((p[j:j + 1],) if prob else ()))
            hp = hp.finish()
            e = self.head_progs[key] = (hp, self._gpu_program(hp, ("head_progs", key)))
        self._used(key, "head_progs")
        return e

    def argmax(self, rows=1, prob=False):
        """The best token of each of the last rows rows of the last run (the
        head and GP_ARGMAX on the GPU): an int32 array, the tokens of
        np.argmax(logits(rows), axis=1), with no copy of the logits (1 MB a
        row) to the host. For greedy MTP. prob: also the softmax probability
        of each (a float32 array): (tokens, probabilities)."""
        cfg = self.cfg
        hp, hg = self._head(rows, prob, self._draft_head())
        assert 1 <= rows <= self.rows
        step = cfg.hidden_size * 4
        _check(lib().rc_gpu_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr,
                            self.last - (rows - 1) * step, rows * step))
        hg.run()
        if self.hot_cache is not None and self.hot_cache.due:
            self.hot_cache.observe()
        hg.download("tok")
        if prob:
            hg.download("prob")
            return hp.names["tok"].copy(), hp.names["prob"].copy()
        return hp.names["tok"].copy()

    def logits(self, rows=1, x=None):
        """The logits of the last rows rows of the last run, or of the rows
        of x (host arrays: the drafts of the MTP layer)."""
        cfg = self.cfg
        if x is not None:
            rows = x.shape[0]
        hp, hg = self._head(rows, draft=self._draft_head(x))
        if x is not None:
            hp.names["x"][:] = x
            hg.upload("x")
        else:
            assert 1 <= rows <= self.rows
            step = cfg.hidden_size * 4
            _check(lib().rc_gpu_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr,
                                self.last - (rows - 1) * step, rows * step))
        hg.run()
        if self.hot_cache is not None and self.hot_cache.due:
            self.hot_cache.observe()
        hg.download("out")
        out = hp.names["out"]
        return out.copy() if rows > 1 else out[0].copy()

    def close(self):
        if self._qsa_buf is not None:
            self._qsa_buf.free()
            self._qsa_buf = None
        for _p, g in list(self.head_progs.values()) + list(self.mtp_progs.values()):
            g.close()
        self.head_progs, self.mtp_progs = {}, {}
        super().close()


def generate_mtp_gpu(dev, prompt, n_new, draft=3, stop=None, stats=None, mtp_cpu=False,
                     media=None, pick=None):
    """Generation with MTP drafts: the model and the MTP layer on the GPU
    (dev, a Qwen4GPU with its cache attached), as Qwen4CPU.generate_mtp; the
    loop of speculative.stream. pick (default greedy) is the sampler of the
    plain decode (a Sampler with mtp_accept "in_set" keeps drafts that its
    settings allow). mtp_cpu runs the MTP layer on the CPU (Qwen4CPU.
    mtp_step). media: the spans of the images of the prompt (Qwen4GPU.
    prefill)."""
    import time
    from reclank.core.speculative import QwenMTPDrafter, QwenTarget, RowPicker, greedy_pick, stream
    pick = pick or greedy_pick
    prompt = list(prompt)
    n = len(prompt)
    target = QwenTarget(dev)
    H = dev.prefill(prompt, streams=True, media=media)
    tok = RowPicker(target, 1, pick).token(0)
    drafter = QwenMTPDrafter(dev, mtp_cpu=mtp_cpu)
    # the rows of the MTP layer: each token with the stream of the model at
    # the position before (zeros before the first)
    drafter.observe(prompt + [tok], np.concatenate([np.zeros((1, H.shape[1]), np.float32), H]), 0)
    st = {}
    t_start = time.time()
    out = list(stream(target, drafter, tok, None, n, draft, pick, tuple(stop or ()), n_new, st,
                      room=lambda p: dev.cache.max_len - p - 2))
    if stats is not None:
        stats.update(rounds=st["steps"], accepted=st["accepted"], drafted=st["drafts"],
                     in_set=st["in_set"], decode_s=time.time() - t_start, draft=st["draft_s"],
                     verify=st["verify_s"])
    return out[:n_new]

