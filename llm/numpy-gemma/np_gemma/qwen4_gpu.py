"""Qwen3.8-Flash-Next on the GPU, with the experts split (QWEN38_PLAN.md,
phase 5).

Qwen4GPU is QwenGPU (np_gemma/qwen_gpu.py) with the program of
compile_qwen4_step. The design stays:

- The dense part runs on the GPU: the products (Q8_0 as rows, F32), the
  gated residual and the n-gram layer (the records of csrc/hyperconn.c),
  the DeltaNet, and QSA (QSA_SELECT, and one ATTN_QSA for each query).
- The GPU holds some experts of each layer (HotCache changes them). The
  CPU computes the other selected experts at the same time.
- The head (Q8_0) runs on the GPU in its own program (one for each count
  of rows).
- The MTP layer (when the model has it) is a program of its own
  (compile_qwen4_step with mtp), with its cache on the GPU. Its experts
  are split too, with its own slots (NP_GEMMA_GPU_MTP_SLOTS, or else
  twice the slots of a layer, at most 0.5 GB; the budget of the hot
  experts does not count them).
- HotCache scores the steps, and also the tokens of an MTP verify group
  that stay (commit) and each row of the MTP layer. A group copies the
  selection of each layer to an array (sel_of) for that.

- A prompt of at least MIX_MIN tokens runs in mixed groups of MIX_SIZE
  rows (NP_GEMMA_GPU_MIX, NP_GEMMA_GPU_MIX_MIN). In each layer, after the
  router, GP_MOE_PLAN (csrc/moe.c) splits the experts: the GPU takes the
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

from . import program as P
from .gpu import GPUProgram, _check, lib
from .gpumm import Buffer, pinned
from .qwen_gpu import MIX_MIN, MIX_SIZE, MT, QwenGPU, _DevCache, _fuse
from .qwen import media_inputs, rope_positions
from .qwen4 import compile_qwen4_step



class _Emit4:
    """The model as compile_qwen4_step sees it for a GPU program."""

    def __init__(self, dev, t, verify, fetch, mix=False):
        self.dev, self.m, self.t, self.fetch, self.mix = dev, dev.model, t, fetch, mix

    def __getattr__(self, name):
        return getattr(self.m, name)

    def dense_rot(self, gname):
        return self.m.dense_rot(gname)

    def emit_lin4(self, prog, xb, gname, out):
        m = self.m.K(gname)
        w, type_ = self.dev.dense(gname)
        prog.emit(P.KQ_LINEAR, xb["xq"], xb["xs"], xb["xm"], xb["src"], w, type_, m.rows, m.cols,
                  xb["t"], out)

    def moe_scratch4(self, t):
        return None

    def emit_moe4(self, prog, i, xb, idx, val, slog, scratch, out):
        if self.fetch:
            self.dev._moe_fetch(prog, i, xb["src"], idx, val, slog, out)
            return
        if self.mix:
            self.dev._moe_mix(prog, i, xb["src"], idx, val, slog, out)
            return
        if self.t > 1:
            sel = self.dev.sel_of(self.t)
            prog.emit(P.COPY, idx, sel[i], idx.nbytes)
        self.dev._moe_split(prog, i, xb["src"], idx, val, slog, out)

    def emit_attn_qsa(self, prog, q, base, scores, out, t, pos, sel, cnt, maxsel, form=0):
        cfg = self.m.cfg
        if t > MT and os.environ.get("NP_GEMMA_GPU_ATTN_MT", "1") != "0":
            # A large group: one record (k_attn_qsa_mt; NP_GEMMA_GPU_ATTN_MT=0 for a test).
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

    def add(self, arrays, fresh=None):
        """More arrays (the buffers of attach's pool first; fresh: zero them,
        else upload; None: as the last attach)."""
        if fresh is not None:
            self._fresh = fresh
        self._put(arrays)

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
        lib().gg_set_moe_rot(int(getattr(model, "rot_experts", False)))
        super().__init__(model, hot_gb=hot_gb, counts=counts, graph=graph, ctx=ctx)
        self.cache_dev = _DevCache4()
        self.cache_dev.before = self._share_before

    def _cache_types(self):
        from .qwen4 import Qwen4Cache, Qwen4MTPCache
        return (Qwen4Cache, Qwen4MTPCache) if self.has_mtp else (Qwen4Cache,)

    def _more_stores(self):
        if self.has_mtp:
            L = self.cfg.num_hidden_layers
            per = sum(self._expert_nb(self.model.M("layers.%d.mlp.switch_mlp.%s" % (L, x)))
                      for x in ("gate_proj", "up_proj", "down_proj"))
            n = int(os.environ.get("NP_GEMMA_GPU_MTP_SLOTS",
                                   min(2 * self.n_slots, int(0.5e9 // per))))
            st = self.stores[L] = self._store(L, np.arange(min(n, self.E)))
            st["ip"] = np.zeros(2 * self.cfg.top_k + 1, np.int32)   # a step does not run it

    def _before_hot(self):
        if self.has_mtp:
            self._mtp_group(1)

    # ---- the hooks of the mixed groups (QwenGPU.mix) ----

    def _compile_mix(self, t):
        self._pool = {}
        return _fuse(compile_qwen4_step(_Emit4(self, t, False, False, mix=True), t), t)

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

    def sel_of(self, t):
        """The selection of each layer of a group of t rows (layers + 1, t,
        top_k), for HotCache."""
        a = self.sel_bufs.get(t)
        if a is None:
            a = self.sel_bufs[t] = np.zeros((self.cfg.num_hidden_layers + 1, t, self.cfg.top_k),
                                            np.int32)
        return a

    def _score(self, t, rows, layers):
        """HotCache: score rows rows of the last group of t rows in layers."""
        hc = self.hot_cache
        if hc is None or rows == 0:
            return
        sel = self.sel_of(t)
        self.g.mirror.buffer_of(sel).download(sel)
        idx = [r for r, e in enumerate(hc.layers) if e["layer"] in layers]
        lay = [hc.layers[r]["layer"] for r in idx]
        hc.score_group(idx, sel[lay, :rows].transpose(1, 0, 2))

    def _dense_bytes(self, bf16_as_q8=None):
        """The bytes of the dense tensors on the GPU (bfloat16 counts as Q8_0
        when the model requantizes them: Qwen4CPU dense; BF12 as the mode of
        NP_GEMMA_DENSE)."""
        from .gguf import tensor_bytes
        g = self.model.g
        q8 = getattr(self.model, "dense", "bf16") in ("q8", "rq8") if bf16_as_q8 is None else bf16_as_q8
        n = 0
        skip = ("token_embd.weight", "output.weight", "per_layer_token_embd.weight",
                "per_layer_token_embd.scale")
        L = self.model.cfg.num_hidden_layers
        mode = os.environ.get("NP_GEMMA_DENSE", "bf12")
        for name, (dims, t, _o) in g.tensors.items():
            if "_exps" in name or name in skip or name.startswith("blk.%d." % L):
                continue
            if t == 57 and mode in ("q8", "rq8", "bf16"):
                t = 30 if mode == "bf16" else 8     # Qwen4CPU.K: BF12 as another mode
            n += tensor_bytes(dims, 8 if (t == 30 and q8) else t)
        return n

    def _compile(self, t, verify=False, fetch=False):
        self._pool = {}
        prog = compile_qwen4_step(_Emit4(self, t, verify, fetch), t, verify=verify)
        if fetch:
            first = []
            for i in (0, 1):
                ranges = self.tables[i][3]
                first.append((P.FETCH, [prog._enc(ranges), prog._enc(len(ranges)),
                                        prog._enc(i), prog._enc(i % 2)]))
            prog.recs[:0] = first
        return _fuse(prog, t)

    def _params(self, pos, t, nreal):
        cfg = self.cfg
        kw = super()._params(pos, t, nreal)
        cache = self.cache
        for i in cfg.ple_layers:
            kw["pleconv.%d" % i] = cache.ple_conv[i]
        for i, a in cache.idx_k.items():
            kw["idxk.%d" % i], kw["blk.%d" % i] = a, cache.idx_blk[i]
        # The scratch of QSA_SELECT: nbmax keys of 8 bytes for each query of
        # a launch (gg_qsa_rows of them; csrc/gpu.cu QSA_ROWS). A query at
        # row p reads (p + 1) / ratio blocks, so nbmax follows the rows of
        # this run, not the context. One buffer serves all the programs, and
        # it grows (twice the size, at most that of the context).
        ratio = min(r for r in cfg.compress_ratios if r)
        nbmax = (pos + t) // ratio + 1
        rows = min(t, lib().gg_qsa_rows())
        extra = lib().gg_qsa_extra()            # the queries of k_qsa_score_tc, first
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
            # The mixed group: MIX_SIZE rows, or the smallest of 256, 512, ...
            # that holds the rest and fits in the cache.
            size = next((sz for sz in (256, 512, 1024, 2048, 4096)
                         if sz <= MIX_SIZE and sz >= min(rem, MIX_SIZE) and sz <= room - pos - c0), 0)
            if MIX_SIZE > 0 and rem >= MIX_MIN and size:
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
            from .qwen4 import Qwen4MTPCache
            self.mcache = Qwen4MTPCache(self.cfg, cache.max_len)
            self.cache_dev.add(self.mcache.kv[self.cfg.num_hidden_layers], fresh=True)
        self.cache_dev.finish()

    def _mtp_group(self, t):
        e = self.mtp_progs.get(t)
        if e is None:
            self._pool = {}
            prog = _fuse(compile_qwen4_step(_Emit4(self, t, False, False), t, mtp=True), t)
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
        """Keep the first n tokens of the last verify group: the DeltaNet
        (QwenGPU.commit), then the state of the n-gram layer from its copy
        and the first n inputs of its convolution."""
        prog, pos, ids, (ple_ids, ple_conv) = self._pending
        self._pending = (prog, pos)
        super().commit(n)
        self._score(prog.tokens, n, range(self.cfg.num_hidden_layers))
        cfg = self.cfg
        ctx = cfg.ple_ngram - 1
        self.cache.ple_ids = np.concatenate([ple_ids, np.asarray(ids[:n], np.int64)])[-ctx:].copy()
        mir = self.g.mirror
        for i, old in ple_conv.items():
            gn = prog.names["gn.%d" % i]
            mir.buffer_of(gn).download(gn)
            new = np.ascontiguousarray(np.concatenate([old, gn[:n]])[-old.shape[0]:])
            self.cache_dev.buffer(self.cache.ple_conv[i]).upload(new)

    def _head(self, rows):
        """The program of the head for rows rows (made once)."""
        e = self.head_progs.get(rows)
        if e is None:
            cfg = self.cfg
            m = self.model.K("output.weight")
            w, type_ = self.dense("output.weight")
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
            hp.emit(P.KQ_LINEAR, None, None, None, hx, w, type_, m.rows, m.cols, rows, out)
            for j in range(rows):
                hp.emit(P.ARGMAX, out[j], m.rows, tok[j:j + 1])
            hp = hp.finish()
            e = self.head_progs[rows] = (hp, self._gpu_program(hp, ("head_progs", rows)))
        self._used(rows, "head_progs")
        return e

    def argmax(self, rows=1):
        """The best token of each of the last rows rows of the last run (the
        head and GP_ARGMAX on the GPU): an int32 array, the tokens of
        np.argmax(logits(rows), axis=1), with no copy of the logits (1 MB a
        row) to the host. For greedy MTP."""
        cfg = self.cfg
        hp, hg = self._head(rows)
        assert 1 <= rows <= self.rows
        step = cfg.hidden_size * 4
        _check(lib().gg_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr,
                            self.last - (rows - 1) * step, rows * step))
        hg.run()
        if self.hot_cache is not None and self.hot_cache.due:
            self.hot_cache.observe()
        hg.download("tok")
        return hp.names["tok"].copy()

    def logits(self, rows=1, x=None):
        """The logits of the last rows rows of the last run, or of the rows
        of x (host arrays: the drafts of the MTP layer)."""
        cfg = self.cfg
        if x is not None:
            rows = x.shape[0]
        hp, hg = self._head(rows)
        if x is not None:
            hp.names["x"][:] = x
            hg.upload("x")
        else:
            assert 1 <= rows <= self.rows
            step = cfg.hidden_size * 4
            _check(lib().gg_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr,
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
    from .speculative import QwenMTPDrafter, QwenTarget, RowPicker, greedy_pick, stream
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

