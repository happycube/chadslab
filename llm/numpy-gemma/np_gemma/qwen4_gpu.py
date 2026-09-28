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
from .qwen_gpu import MT, QwenGPU, _DevCache, _fuse
from .qwen4 import compile_qwen4_step


class _Emit4:
    """The model as compile_qwen4_step sees it for a GPU program."""

    def __init__(self, dev, t, verify, fetch):
        self.dev, self.m, self.t, self.fetch = dev, dev.model, t, fetch

    def __getattr__(self, name):
        return getattr(self.m, name)

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
        if self.t > 1:
            sel = self.dev.sel_of(self.t)
            prog.emit(P.COPY, idx, sel[i], idx.nbytes)
        self.dev._moe_split(prog, i, xb["src"], idx, val, slog, out)

    def emit_attn_qsa(self, prog, q, base, scores, out, t, pos, sel, cnt, maxsel):
        # One record for each query (the GPU kernels take one query).
        cfg = self.m.cfg
        for j in range(t):
            pj = prog.temp()
            prog.emit(P.S_ADD, pj, pos, j)
            prog.emit(P.ATTN_QSA, q[j:j + 1], *base, scores, out[j:j + 1], cfg.num_heads,
                      cfg.num_kv_heads, cfg.head_dim, 1, pj, sel[j:j + 1], cnt[j:j + 1], maxsel)


class _DevCache4(_DevCache):
    """_DevCache with the state of the n-gram layer and of the indexer, and
    the cache of the MTP layer."""

    def attach(self, cache):
        super().attach(cache)
        self.add(list(cache.ple_conv.values()) + list(cache.idx_k.values())
                 + list(cache.idx_blk.values()))

    def add(self, arrays):
        from .gpu import Buffer
        for a in arrays:
            b = Buffer(a.nbytes)
            b.upload(np.ascontiguousarray(a))
            self.bufs[id(a)] = (a, b)

    def buffer(self, a):
        return self.bufs[id(a)][1]


class Qwen4GPU(QwenGPU):
    """Qwen3.8-Flash-Next (a Qwen4CPU) on the GPU; see the module text."""

    def __init__(self, model, hot_gb=None, counts=None, graph=True):
        self.head_progs = {}
        self.mtp_progs = {}
        self.sel_bufs = {}
        self.mcache = None
        L = model.cfg.num_hidden_layers
        self.has_mtp = "blk.%d.nextn.eh_proj.weight" % L in model.g.tensors
        super().__init__(model, hot_gb=hot_gb, counts=counts, graph=graph)
        self.cache_dev = _DevCache4()

    def _more_stores(self):
        if self.has_mtp:
            L = self.cfg.num_hidden_layers
            per = sum(self.model.M("layers.%d.mlp.switch_mlp.%s" % (L, x)).data.nbytes
                      for x in ("gate_proj", "up_proj", "down_proj")) // self.E
            n = int(os.environ.get("NP_GEMMA_GPU_MTP_SLOTS",
                                   min(2 * self.n_slots, int(0.5e9 // per))))
            st = self.stores[L] = self._store(L, np.arange(min(n, self.E)))
            st["ip"] = np.zeros(2 * self.cfg.top_k + 1, np.int32)   # a step does not run it

    def _before_hot(self):
        if self.has_mtp:
            self._mtp_group(1)

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
        for j in range(rows):
            hc.score_rows(idx, sel[lay, j])

    def _dense_bytes(self):
        g = self.model.g
        n = 0
        skip = ("token_embd.weight", "output.weight", "per_layer_token_embd.weight")
        L = self.cfg.num_hidden_layers
        for name in g.tensors:
            if "_exps" in name or name in skip or name.startswith("blk.%d." % L):
                continue
            n += g.raw(name)[0].nbytes
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
        nbmax = cache.max_len // min(r for r in cfg.compress_ratios if r) + 1
        kw["qsa_scratch"] = np.empty(t * nbmax * 8 // 4 + 16, np.float32)
        kw["nbmax"] = nbmax
        return kw

    # ---- the runs ----

    def _inputs(self, prog, g, ids):
        """The inputs of a run of the tokens ids: the embeddings in each
        stream, and the rows of the n-gram table (the host keeps its
        last tokens)."""
        cfg = self.cfg
        t = len(ids)
        x = self.model.embed(ids)
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

    def group(self, tokens, pos, size=None, verify=False, fetch=False):
        """Run tokens from position pos as one group (see QwenGPU.group).
        Return the input of the head of each token."""
        t = len(tokens)
        size = size or t
        prog, g = self._group(size, verify, fetch)
        if self.hot_cache is not None:
            self.hot_cache.prepare(wait=fetch)
        if fetch:
            self._fill_tables()
        if verify:
            snap = (self.cache.ple_ids.copy(),
                    {i: self._download(self.cache.ple_conv[i]) for i in self.cfg.ple_layers})
        self._inputs(prog, g, list(tokens))
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

    def prefill(self, ids, pos=0, streams=False):
        """Run a prompt from position pos. Return the input of the head of
        its last token; with streams, the streams after the last layer of
        each token (the input of the MTP layer)."""
        ids = list(ids)
        c0, h, hs = 0, None, []
        room = self.cache.max_len
        while c0 < len(ids):
            size, n, fetch = self._sizes(len(ids) - c0, room - pos - c0)
            h = self.group(ids[c0:c0 + n], pos + c0, size, fetch=fetch)
            if streams:
                hs.append(self.streams(n))
            c0 += n
        return np.concatenate(hs) if streams else h[-1:]

    # ---- the MTP layer ----

    def attach(self, cache):
        """Copy the cache to the GPU, with a new cache of the MTP layer."""
        super().attach(cache)
        if self.has_mtp:
            from .qwen4 import Qwen4MTPCache
            self.mcache = Qwen4MTPCache(self.cfg, cache.max_len)
            self.cache_dev.add(self.mcache.kv[self.cfg.num_hidden_layers])

    def _mtp_group(self, t):
        e = self.mtp_progs.get(t)
        if e is None:
            self._pool = {}
            prog = _fuse(compile_qwen4_step(_Emit4(self, t, False, False), t, mtp=True), t)
            e = self.mtp_progs[t] = (prog, GPUProgram(prog, graph=self.graph, mirror=self.g.mirror))
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
        cos, sin = self.model.rope(np.arange(pos, pos + size))
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

    def logits(self, rows=1, x=None):
        """The logits of the last rows rows of the last run, or of the rows
        of x (host arrays: the drafts of the MTP layer)."""
        cfg = self.cfg
        if x is not None:
            rows = x.shape[0]
        e = self.head_progs.get(rows)
        if e is None:
            m = self.model.K("output.weight")
            w, type_ = self.dense("output.weight")
            hp = P.Program()
            hx, out = np.zeros((rows, cfg.hidden_size), np.float32), np.zeros((rows, m.rows), np.float32)
            hp.names.update(x=hx, out=out)
            hp.emit(P.KQ_LINEAR, None, None, None, hx, w, type_, m.rows, m.cols, rows, out)
            hp = hp.finish()
            e = self.head_progs[rows] = (hp, GPUProgram(hp, graph=self.graph, mirror=self.g.mirror))
        hp, hg = e
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
        for _p, g in list(self.head_progs.values()) + list(self.mtp_progs.values()):
            g.close()
        self.head_progs, self.mtp_progs = {}, {}
        super().close()


def generate_mtp_gpu(dev, prompt, n_new, draft=3, stop=None, stats=None, mtp_cpu=False):
    """Greedy generation with MTP drafts: the model and the MTP layer on the
    GPU (dev, a Qwen4GPU with its cache attached), as Qwen4CPU.generate_mtp.
    mtp_cpu runs the MTP layer on the CPU (Qwen4CPU.mtp_step)."""
    import time
    from .qwen4 import Qwen4MTPCache
    m, cfg = dev.model, dev.cfg
    mcache = Qwen4MTPCache(cfg, dev.cache.max_len)

    def mtp_rows(Hs, ids, p):
        """The MTP layer; return the logits of its last row and its streams."""
        if mtp_cpu:
            xm, hm = m.mtp_step(Hs, ids, mcache, p)
            return dev.logits(x=xm[-1:]), hm
        hm = dev.mtp(Hs, ids, p)
        return dev.logits(), hm
    prompt = list(prompt)
    n = len(prompt)
    H = dev.prefill(prompt, streams=True)
    tok = int(np.argmax(dev.logits()))
    out, pos = [tok], n
    m_ids = prompt + [tok]
    m_H = np.concatenate([np.zeros((1, H.shape[1]), np.float32), H])
    m_pos = 0
    rounds = acc_total = 0
    tm = {"draft": 0.0, "verify": 0.0}
    t_start = time.time()
    while len(out) < n_new and (stop is None or tok not in stop):
        t0 = time.time()
        for c0 in range(0, len(m_ids), 256):
            c1 = min(len(m_ids), c0 + 256)
            lg, hm = mtp_rows(m_H[c0:c1], m_ids[c0:c1], m_pos + c0)
        drafts = []
        while True:
            drafts.append(int(np.argmax(lg)))
            if len(drafts) == draft:
                break
            lg, hm = mtp_rows(hm[-1:], drafts[-1:], pos + len(drafts))
        t1 = time.time()
        dev.verify([tok] + drafts, pos)
        best = np.argmax(dev.logits(rows=draft + 1), axis=1)
        Hv = dev.streams(draft + 1)
        a = 0
        while a < draft and best[a] == drafts[a]:
            a += 1
        dev.commit(a + 1)
        tok = int(best[a])
        out.extend(drafts[:a] + [tok])
        m_ids, m_H, m_pos = drafts[:a] + [tok], Hv[:a + 1], pos + 1
        tm["draft"] += t1 - t0
        tm["verify"] += time.time() - t1
        rounds += 1
        acc_total += a
        pos += a + 1
    if stats is not None:
        stats.update(rounds=rounds, accepted=acc_total, drafted=rounds * draft,
                     decode_s=time.time() - t_start, **tm)
    return out[:n_new]

