"""GLM-5.3-Flash on the GPU (P5 of docs/GLM53_PLAN.md): GLM53GPU is the machinery of Qwen3.8 on the
GPU (reclank/models/qwen/gpu38.py, Qwen4GPU; gpu35.QwenGPU: the hot experts and HotCache, the cold
experts on the CPU at the same time, the mixed groups of a prompt, the head) with the program of
compile_glm53_step (reclank/models/glm53/program.py) and GLM's cache (GLM53Cache: the KDA state, the
latents and the pooled index keys of the DSA layers).

The experts: EXL3 stacks of 288 (GLM53Prog.K), the CPU part with act bit 512 (the clamped SwiGLU:
GLM53Prog.moe_act), the GPU act kernels with rc_gpu_set_moe_clamp; the shared expert ungated (its
logit +inf in the program).

    m = GLM53Prog("/home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw")
    g = GLM53GPU(m)
    cache = m.cache(8192)
    g.attach(cache)
    g.prefill(ids); logits = g.logits()
    g.step(token, pos); logits = g.logits()
"""
from __future__ import annotations

import os

import numpy as np

from reclank.backends.cuda.runtime import _check, lib
from reclank.ir import alloc
from reclank.ir import program as P
from reclank.models.glm53.program import GLM53Cache, compile_glm53_step
from reclank.models.qwen.gpu35 import QwenGPU
from reclank.models.qwen.gpu38 import Qwen4GPU


class GLM53Emit:
    """The model as compile_glm53_step sees it for a GPU program (gpu38._Emit4 for GLM)."""

    pack_temps = alloc.PACK

    def __init__(self, dev, t, verify, fetch, mix=False):
        from reclank.models.qwen.gpu35 import has_tc
        self.dev, self.m, self.t, self.fetch, self.mix = dev, dev.model, t, fetch, mix
        self.tc = has_tc() and os.environ.get("RECLANK_GPU_GLM_ATTN_TC", "1") != "0"
        dev._lpool = None

    def __getattr__(self, name):
        return getattr(self.m, name)

    @property
    def maxn(self):
        return self.dev.maxn

    @property
    def dspark(self):
        return self.dev.dspark

    def attn_keys(self, maxk, dl):
        """The keys buffer of DS_ATTN for groups of up to 8 rows (one for the model)."""
        if getattr(self.dev, "_akeys", None) is None:
            self.dev._akeys = np.zeros((8, maxk, dl), np.float32)
        return self.dev._akeys

    def layer_begin(self, prog, i):
        self.dev._layer_begin(prog)

    def dense(self, gname):
        return self.dev.dense(gname)

    def moe(self, prog, i, x, idx, val, slog, out):
        if self.fetch:
            self.dev._moe_fetch(prog, i, x, idx, val, slog, out)
            return
        if self.mix:
            self.dev._moe_mix(prog, i, x, idx, val, slog, out)
            return
        if self.t > 1:
            sel = self.dev.sel_of(self.t)
            prog.emit(P.COPY, idx, sel[i], idx.nbytes)
        self.dev._moe_split(prog, i, x, idx, val, slog, out)


class GLM53GPU(Qwen4GPU):
    """GLM-5.3-Flash (a GLM53Prog) on the GPU; see the module text."""

    def __init__(self, model, hot_gb=None, counts=None, graph=True, ctx=None, dspark=None):
        """dspark: the drafter (GLM53DSpark): the programs make its context rows, and its weights
        count with the dense ones (fewer hot experts)."""
        self.dspark = dspark
        self.head_progs, self.mtp_progs, self.sel_bufs = {}, {}, {}
        self._qsa_buf = None
        self.mcache = None
        self.has_mtp = False
        self.model = model
        # the CPU parts split the units of a group of 3 tokens or more dynamically (moe_part.c: a
        # verify group of 3, 5, 9 tokens 4%, 11%, 16% faster; a step stays static)
        os.environ.setdefault("RECLANK_PART_DYN_MIN", "3")
        # the pools of the scores of the indexer (a literal of the programs): those of the largest
        # context (ctx, RECLANK_GPU_CTX, else 8192)
        self.maxn = int(ctx or os.environ.get("RECLANK_GPU_CTX", "0") or 8192) // model.cfg.index_pool + 1
        lib().rc_gpu_set_moe_clamp(1)
        if counts is None:
            # the first hot experts: the routing profile of real text (scripts/glm53_expert_profile.py),
            # RECLANK_EXPERT_COUNTS or <model>/expert_profile.npy; counts=False: none (the first experts)
            path = os.environ.get("RECLANK_EXPERT_COUNTS") or os.path.join(getattr(model, "path", ""),
                                                                           "expert_profile.npy")
            if os.path.exists(path):
                counts = np.load(path)
        QwenGPU.__init__(self, model, hot_gb=hot_gb, counts=counts if counts is not False else None, graph=graph,
                         ctx=ctx)

    def _cache_types(self):
        return (GLM53Cache,)

    def _dense_bytes(self, bf16_as_q8=None):
        """The bytes of the dense tensors on the GPU: all but the experts, the embeddings and the
        head."""
        from reclank.backends.cpu import cops
        from reclank.formats.exl3 import KQ_E3_BASE
        n = 0
        for name, (dims, t, _o) in self.model.g.tensors.items():
            if "_exps" in name or name in ("token_embd.weight", "output.weight"):
                continue
            if t >= KQ_E3_BASE:
                n += cops.e3_bytes(t - KQ_E3_BASE, dims[0], dims[1])
            else:
                n += int(np.prod(dims)) * (4 if t == 0 else 2)
        if self.dspark is not None:
            n += self.dspark.nbytes()
        return n

    def _scratch_names(self):
        return ("scores",)

    def _compile(self, t, verify=False, fetch=False):
        self._pool = {}
        prog = compile_glm53_step(GLM53Emit(self, t, verify, fetch), t, verify=verify)
        if fetch:
            first = []
            for i in self.moe_layers[:2]:
                ranges = self.tables[i][3]
                first.append((P.FETCH, [prog._enc(ranges), prog._enc(len(ranges)),
                                        prog._enc(i), prog._enc(i % 2)]))
            prog.recs[:0] = first
        return self._pack(prog)

    def _compile_mix(self, t):
        self._pool = {}
        return self._pack(compile_glm53_step(GLM53Emit(self, t, False, False, mix=True), t))

    def _params(self, pos, t, nreal):
        cfg, cache = self.cfg, self.cache
        assert pos + t <= cache.max_len, "the cache is too short"
        self.cache_dev.share_tail(pos + t, self.cache_dev.before)
        kw = {"pos": pos, "nreal": nreal, "maxlen": cache.max_len,
              "scores": np.empty(t * self.maxn, np.float32)}
        for i, a in cache.conv.items():
            kw["conv.%d" % i], kw["S.%d" % i] = a, cache.state[i]
        for i, (lat, pool, ks, ss) in cache.kv.items():
            kw.update({"lat.%d" % i: lat, "pool.%d" % i: pool, "kst.%d" % i: ks, "sst.%d" % i: ss})
        return kw

    def attach(self, cache):
        assert cache.max_len // self.cfg.index_pool + 1 <= self.maxn, \
            "a cache of %d positions: the model was made for %d (ctx)" % (cache.max_len, 4 * (self.maxn - 1))
        super().attach(cache)

    def commit(self, n):
        """Keep the first n tokens of the last verify group: the KDA layers from their logs; the
        open pool of each DSA layer as after the n tokens (DS_COMPRESS_FIX: from the state before
        the group and the keys and gates of its tokens). The latents and the pooled keys of the
        other tokens stay until a run writes those positions again."""
        prog, pos = self._pending[:2]
        cfg = self.cfg
        mir = self.g.mirror
        for i in self.cache.conv:
            _check(lib().rc_gpu_gdn_commit(
                self.cache_dev.device(self.cache.conv[i]), self.cache_dev.device(self.cache.state[i]),
                mir.buffer_of(prog.names["log.%d" % i]).ptr, n, cfg.conv_kernel, cfg.kda_heads,
                cfg.kda_heads, cfg.kda_dim, cfg.kda_dim, 4))
        if self.cache.kv:
            # (on the verify program: closed with it when the LRU frees it, gpumm _free_program)
            g = getattr(prog, "commit_g", None)
            if g is None:
                cp = P.Program()
                for i in self.cache.kv:
                    cp.emit(P.DS_COMPRESS_FIX, cp.slot("kst.%d" % i), cp.slot("sst.%d" % i), prog.names["pbk.%d" % i],
                            prog.names["pbs.%d" % i], prog.names["pk.%d" % i], prog.names["pg.%d" % i],
                            cp.slot("pos"), cp.slot("keep"), cfg.index_pool, cfg.index_dim)
                from reclank.backends.cuda.runtime import GPUProgram
                g = prog.commit_g = GPUProgram(cp.finish(), graph=False, mirror=mir)
                prog.__dict__.setdefault("side", []).append(g)
            kw = {"pos": pos, "keep": n}
            for i, (_lat, _pool, ks, ss) in self.cache.kv.items():
                kw["kst.%d" % i], kw["sst.%d" % i] = ks, ss
            g.bind(kw, self.cache_dev)
            g.run()
        self.cache.n = pos + n
        self._pending = None

    def close(self):
        for p in self._live_programs():
            for g in getattr(p, "side", ()):
                g.close()
            p.side = []
        super().close()
