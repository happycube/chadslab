"""Speculative decoding of Nemotron 3.5 Bolt with its LiLiCorr drafter (lilicorr.py): the CPU engine
(kq.NemotronKQ) verifies the drafter's block (docs/NEMOTRON_BOLT_PLAN.md, P8).

A round: the drafter's 7 tokens after the last one; the target runs [last, drafts] as one group of 8
(its Mamba states saved first); the drafts it agrees with are taken, and its own next token after them.
The drafter's context gets the target's rows of the taken positions. When a draft is refused, the
Mamba states go back to the saved ones and the group's in_proj rows of the taken tokens run through
NM_MAMBA again (nreal: the rows after them do not change the state); the attention cache only keeps
the taken positions (cache.n). The choice: greedy (the target's argmax), or a sampler of the target's
rows (sample(logits) -> token: a draft is taken when the target's sample is it, which keeps the
target's distribution).
"""
from __future__ import annotations

import time

import numpy as np

from reclank.models.nemotron.kq import BB


def _to_other(prog):
    """Operand sd of each NM_MAMBA record: the slot "sdelta" (the GPU: the new states to the other set)."""
    from reclank.ir import program as P
    for op, args in prog.recs:
        if op == P.NM_MAMBA:
            args += [prog._enc(None)] * (23 - len(args)) + [prog._enc(prog.slot("sdelta"))]


class NemotronSpec:
    def __init__(self, m, drafter):
        self.m, self.d = m, drafter
        if list(getattr(m, "capture", ()) or ()) != list(drafter.target_ids):
            m.capture = list(drafter.target_ids)
            m.programs = {}
        self.block = drafter.block
        self._replay = None
        self.stats = []

    # ---- the target

    def _run(self, ids, cache):
        """The target on ids at cache.n: (the rows after the last block, the drafter's rows)."""
        m = self.m
        prog = m.program(len(ids))
        m._run(prog, ids, cache, cache.n)
        feats = np.concatenate([prog.names["feat.%d" % b] for b in self.d.target_ids], 1)
        return prog, prog.names["xn"], feats

    def _replay_prog(self, sd=False):
        """NM_MAMBA of each Mamba layer on the in_proj rows of the verify group (nreal: the taken rows).
        sd: the new states to the other set (operand sd: the slot "sdelta"; NemotronGPU.swap)."""
        from reclank.ir import program as P
        m, cfg = self.m, self.m.cfg
        vp = m.program(self.block)
        t = self.block
        prog = P.Program()
        y, cscr = np.zeros((t, cfg.m_inner), np.float32), np.zeros((t, cfg.conv_dim), np.float32)
        nreal = prog.slot("nreal")
        conv, nh = cfg.conv_dim, cfg.m_heads
        for i, kind in enumerate(cfg.kinds):
            if kind != "mamba":
                continue
            pp = BB + "layers.%d.mixer." % i
            prog.emit(P.NM_MAMBA, vp.names["zxd.%d" % i], prog.slot("conv.%d" % i), m.F(pp + "conv1d.weight", (conv, cfg.conv)),
                      m.F(pp + "conv1d.bias"), m.F(pp + "A_log"), m.F(pp + "D"), m.F(pp + "dt_bias"), m.F(pp + "norm.weight"),
                      prog.slot("S.%d" % i), y, cscr, t, nh, cfg.m_dim, cfg.m_groups, cfg.m_state, cfg.conv,
                      float(cfg.eps), float(cfg.dt_min), nreal)
        prog.keep += [y, cscr]
        if sd:
            _to_other(prog)
        return prog.finish()

    # ---- a sequence

    def prefill(self, ids, cache):
        """The prompt (chunks of the engine), its rows into the drafter's context: the last logits."""
        m = self.m
        xn = None
        for c0 in range(0, len(ids), m.CHUNK):
            chunk = ids[c0:c0 + m.CHUNK]
            pos = cache.n
            _, xn, feats = self._run(chunk, cache)
            cache.n += len(chunk)
            self.d.inject(None, feats, np.arange(pos, pos + len(chunk)))
        return m.logits(xn[-1:])[0]

    def round(self, last, cache, sample=None):
        """One round after the token last (at position cache.n): the tokens taken (1 .. block)."""
        m, d = self.m, self.d
        n = cache.n
        t0 = time.time()
        drafts = d.draft(None, last, n)
        t1 = time.time()
        saved = {i: (cache.conv[i].copy(), cache.S[i].copy()) for i in cache.conv}
        ids = [int(last)] + drafts
        prog, xn, feats = self._run(ids, cache)
        lg = m.logits(xn)
        t2 = time.time()
        tt = [int(np.argmax(r)) if sample is None else int(sample(r)) for r in lg]
        a = 0
        while a < len(drafts) and drafts[a] == tt[a]:
            a += 1
        if a < len(drafts):
            for i, (cv, S) in saved.items():
                cache.conv[i][:] = cv
                cache.S[i][:] = S
            if self._replay is None:
                self._replay = self._replay_prog()
            kw = {"nreal": a + 1}
            kw.update({k: v for k, v in cache.params().items() if k in self._replay.by_name})
            self._replay.bind(**kw)
            self._replay.run()
        cache.n = n + a + 1
        d.inject(None, feats[:a + 1], np.arange(n, n + a + 1))
        self.stats.append((a, t1 - t0, t2 - t1, time.time() - t2))
        return drafts[:a] + [tt[a]]

    def generate(self, ids, n_new, cache, sample=None, stop=()):
        out = []
        lg = self.prefill(list(ids), cache)
        last = int(np.argmax(lg)) if sample is None else int(sample(lg))
        out.append(last)
        while len(out) < n_new and last not in stop:
            got = self.round(last, cache, sample)
            out += got
            last = got[-1]
            if any(g in stop for g in got):
                break
        return out[:n_new]


class NemotronGPUSpec(NemotronSpec):
    """The rounds on the GPU engine (gpu.NemotronGPU): the drafter on the GPU (lilicorr.LiLiCorrGPU),
    a verify group of exactly 8 rows with the head on each; the group and the replay of the taken
    rows read the Mamba states in use and write the other set (NemotronGPU.swap: no copies)."""

    def __init__(self, gdev, drafter):
        self.g = gdev
        super().__init__(gdev.m, drafter)
        self._v = None

    def _vprog(self):
        """The verify group: the step of 8 rows (gpu.NmEmit), the head on every row."""
        from reclank.backends.cuda.runtime import GPUProgram
        from reclank.ir import program as P
        from reclank.models.nemotron.gpu import NmEmit
        from reclank.models.nemotron.kq import compile_nemotron_step
        if self._v is None:
            t, g = self.block, self.g
            prog = compile_nemotron_step(self.m, t, em=NmEmit(g, t))
            head = self.m.M("language_model.lm_head")
            lg = np.zeros((t, head.rows), np.float32)
            prog.recs.append((P.KQ_LINEAR, [prog._enc(a) for a in (
                None, None, None, prog.names["xn"], head.data, head.type, head.rows, head.cols, t, lg)]))
            prog.names["logits"] = lg
            _to_other(prog)
            prog.finish()
            self._v = (prog, GPUProgram(prog, graph=g.graph, mirror=g.mirror))
        return self._v

    def _bind_run(self, e, extra=None):
        prog, gp = e
        kw = {"pos": self.g.cache.n, "nreal": self.block, "hs": self.g.cfg.head_dim, "sdelta": self.g.sdelta()}
        kw.update(self.g.cache.params())
        kw.update(extra or {})
        gp.bind({k: v for k, v in kw.items() if k in prog.by_name}, self.g.cache_dev, scratch=("scores",))
        gp.run()

    def _replay_gpu(self):
        from reclank.backends.cuda.runtime import GPUProgram
        if self._replay is None:
            self._vprog()
            # (the in_proj rows of the verify group: the same host arrays, the same device buffers)
            vp = self._v[0]
            self.m.programs[self.block] = vp
            prog = self._replay_prog(sd=True)
            self._replay = (prog, GPUProgram(prog, graph=self.g.graph, mirror=self.g.mirror))
        return self._replay

    def prefill(self, ids, cache=None):
        """The prompt on the GPU (its groups); the drafter's rows from each group."""
        g = self.g
        g.reset()
        pos, c0, lg = 0, 0, None
        from reclank.models.nemotron.gpu import GROUPS
        while c0 < len(ids):
            left = len(ids) - c0
            size = next((s for s in GROUPS[::-1] if s <= left), 1)
            lg = g._run(ids[c0:c0 + size], pos + c0)
            prog, gp = g._prog(size)
            for b in self.d.target_ids:
                gp.download("feat.%d" % b)
            feats = np.concatenate([prog.names["feat.%d" % b] for b in self.d.target_ids], 1)
            self.d.inject(None, feats[:size], np.arange(pos + c0, pos + c0 + size))
            c0 += size
        return lg

    def round(self, last, cache=None, sample=None):
        g, d = self.g, self.d
        n = g.cache.n
        if n + self.block > g.cache.max_len:
            raise ValueError("the cache holds %d positions: a verify group of %d at %d" % (
                g.cache.max_len, self.block, n))
        t0 = time.time()
        drafts = d.draft(None, last, n)
        t1 = time.time()
        prog, gp = self._vprog()
        x = prog.names["H"]
        x[:] = self.m.embed([int(last)] + drafts)
        gp.upload("H")
        self._bind_run(self._v)
        gp.download("logits")
        for b in d.target_ids:
            gp.download("feat.%d" % b)
        lg = prog.names["logits"]
        t2 = time.time()
        tt = [int(np.argmax(r)) if sample is None else int(sample(r)) for r in lg]
        a = 0
        while a < len(drafts) and drafts[a] == tt[a]:
            a += 1
        if a < len(drafts):
            # (the replay reads the states the group read, still in use, and writes the other set)
            self._bind_run(self._replay_gpu(), {"nreal": a + 1})
        g.swap()
        g.cache.n = n + a + 1
        feats = np.concatenate([prog.names["feat.%d" % b] for b in d.target_ids], 1)
        d.inject(None, feats[:a + 1], np.arange(n, n + a + 1))
        self.stats.append((a, t1 - t0, t2 - t1, time.time() - t2))
        return drafts[:a] + [tt[a]]

    def generate(self, ids, n_new, cache=None, sample=None, stop=()):
        out = []
        lg = self.prefill(list(ids))
        last = int(np.argmax(lg)) if sample is None else int(sample(lg))
        out.append(last)
        while len(out) < n_new and last not in stop:
            got = self.round(last, None, sample)
            out += got
            last = got[-1]
            if any(gg in stop for gg in got):
                break
        return out[:n_new]
