"""The decode step of Qwen3.5 / Qwen3.6 MoE on the GPU, with the experts split.

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

The prompt pass runs on the CPU (QwenGGUFProgram); attach() then copies the
cache to the GPU.

    g = QwenGPU(model)                   # model: a QwenGGUFProgram
    h = model.forward(ids, cache)        # the prompt on the CPU
    g.attach(cache)
    g.step(token, pos); logits = g.logits()
    g.detach(cache)                      # the host cache has the new values
"""
from __future__ import annotations

import os

import numpy as np

from . import cops
from . import program as P
from .gpu import Buffer, GPUProgram, HotCache, _check, lib, mem_info, pinned
from .qwen import compile_qwen_step

Q8_0, Q8_R = 8, 100     # the ggml type, and its rows for the GPU (csrc/gpu.cu)


class _Emit:
    """The model as compile_qwen_step sees it for the GPU step: the products
    read copies of the weights that the GPU holds, and the experts split
    between the GPU and the CPU (QwenGPU.emit_moe)."""

    def __init__(self, dev):
        self.dev = dev
        self.m = dev.model

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
        self.dev.emit_moe(prog, xb, p, idx, val, slog, out)


class QwenGPU:
    """The decode step of a QwenGGUFProgram on the GPU (see the module
    text). hot_gb is the memory for the hot experts; None takes
    NP_GEMMA_GPU_HOT_GB, or else the free memory of the GPU less the dense
    part, the head, and 2 GB for the rest. counts (layers x experts, or None)
    selects the first hot experts; else each layer starts with its first
    experts, and HotCache changes them."""

    def __init__(self, model, hot_gb=None, counts=None, graph=True):
        self.model = model
        cfg = self.cfg = model.cfg
        self._dense = {}
        self.cpu_progs = []
        self.n_events = 0
        E, L = cfg.num_experts, model.n_layers
        # The bytes of one expert (gate, up, down) of each layer.
        per = []
        for i in range(L):
            p = "layers.%d.mlp.switch_mlp." % i
            per.append(sum(model.M(p + n).data.nbytes // E
                           for n in ("gate_proj", "up_proj", "down_proj")))
        if hot_gb is None and os.environ.get("NP_GEMMA_GPU_HOT_GB"):
            hot_gb = float(os.environ["NP_GEMMA_GPU_HOT_GB"])
        if hot_gb is None:
            head = model.M("lm_head").data.nbytes
            hot_gb = max(0.0, (mem_info()[0] - self._dense_bytes() - head - 2.0e9) / 1e9)
        n = int(min(E, max(1, hot_gb * 1e9 // (L * max(per)))))
        self.n_slots = n
        self.stores = {}
        for i in range(L):
            if counts is not None:
                init = np.sort(np.argsort(-np.asarray(counts[i]), kind="stable")[:n])
            else:
                init = np.arange(n)
            self.stores[i] = self._store(i, init)
        self.prog = compile_qwen_step(_Emit(self), 1)
        self.g = GPUProgram(self.prog, graph=graph)
        self.cache_dev = _DevCache()
        self.head = None
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
        n = 0
        for name, m in self.model._m.items():
            if "switch_mlp" not in name and name not in ("lm_head", "embed_tokens"):
                n += m.data.nbytes
        # The matrices that the model has not read yet: about 1.7 GB for
        # the UD-Q4_K_M file.
        return max(n, 1.7e9)

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

    # ---- the experts of a layer ----

    def emit_moe(self, prog, xb, p, idx, val, slog, out):
        cfg = self.cfg
        i = int(p.split(".")[1])
        st = self.stores[i]
        k, E, hidden, inner = cfg.top_k, cfg.num_experts, cfg.hidden_size, cfg.moe_inter
        h = xb["src"]
        t = h.shape[0]
        assert t == 1, "the GPU step runs one token"
        cold = np.zeros(2 * k + 1, dtype=np.int32)
        cold_val = np.zeros(k, dtype=np.float32)
        prog.emit(P.HOT_SPLIT, idx, val, st["slots"], cold, cold_val, k, 1)
        hp, vp, ip = pinned((1, hidden)), pinned((k,)), pinned((2 * k + 1,), np.int32)
        st["ip"] = ip
        ev = self.n_events
        self.n_events += 1
        prog.emit(P.TO_HOST, h, hp, hp.nbytes, cold_val, vp, vp.nbytes, cold, ip, ip.nbytes, ev)
        # The GPU part: the hot experts and the shared expert.
        s = "layers.%d.mlp.shared_expert." % i
        sg, su, sd = (self.model.M(s + n) for n in ("gate_proj", "up_proj", "down_proj"))
        assert sg.type == su.type == sd.type and sg.rows == inner
        gm, _um, dm = st["mats"]
        gpu_part = np.zeros((1, hidden), np.float32)
        pairs = k + 1
        prog.emit(P.KQ_HOT_MOE, h, val, idx, st["slots"], st["gate"], st["up"], st["down"],
                  np.zeros((pairs, 2 * inner), np.float32), np.zeros((pairs, inner), np.float32),
                  np.zeros((pairs, hidden), np.float32), gpu_part, k, inner, hidden, gm.type,
                  dm.type, 1, self.dense(s + "gate_proj")[0], self.dense(s + "up_proj")[0],
                  self.dense(s + "down_proj")[0], self.dense(s + "gate_proj")[1], slog)
        # The CPU part: the cold experts, on the pinned copies.
        cc = P.Program()
        xq = np.zeros((1, hidden), np.int8)
        xs, xm = np.zeros((1, hidden // 32), np.float32), np.zeros((1, hidden // 16), np.float32)
        cc.emit(P.KQ_QUANT, hp, 1, hidden, xq, xs, xm)
        mats = cops.kq_moe_mats(*(m.c() for m in st["mats"]), None)
        host_out = np.zeros((1, hidden), np.float32)
        cc.emit(P.KQ_MOE, xq, xs, xm, ip, vp, 1, k, E, mats, None, hidden, inner,
                cops.kq_moe_scratch(1, k, E, hidden, inner), host_out, ip[k:])
        cc.keep.append(mats)
        cpu = cc.finish()
        self.cpu_progs.append(cpu)
        part = np.zeros((1, hidden), np.float32)
        prog.emit(P.CPU_JOIN, cpu.buf, ev)
        prog.emit(P.TO_DEV, host_out, part, part.nbytes)
        prog.emit(P.ADD, gpu_part, part, out, hidden)

    # ---- the steps ----

    def attach(self, cache):
        """Copy the cache to the GPU. From now on the GPU copy is the true
        one."""
        self.cache_dev.attach(cache)
        self.cache = cache

    def detach(self, cache):
        self.cache_dev.to_host()
        self.cache_dev.release()

    def _params(self, pos):
        cfg = self.cfg
        cache = self.cache
        cos, sin = self.model.rope(np.arange(pos, pos + 1))
        kw = {"pos": pos, "cos": np.ascontiguousarray(cos, np.float32),
              "sin": np.ascontiguousarray(sin, np.float32),
              "scores": np.empty(cfg.num_heads * (pos + 1) + 64, np.float32)}
        for i in range(self.model.n_layers):
            if cfg.layer_types[i] == "full_attention":
                K, V = cache.kv[i]
                kw["K.%d" % i], kw["V.%d" % i] = K, V
                kw["hs"] = K.shape[1] * K.shape[2]
            else:
                kw["conv.%d" % i], kw["S.%d" % i] = cache.conv[i], cache.state[i]
        return kw

    def step(self, token, pos):
        """Run one token at position pos. logits() gives its logits."""
        if self.hot_cache is not None:
            self.hot_cache.prepare()
        self.prog.names["x"][:] = self.model.embed([token])
        self.g.upload("x")
        self.g.bind(self._params(pos), self.cache_dev, scratch=("scores",))
        self.g.run()
        self.cache.n = pos + 1
        if self.hot_cache is not None:
            self.hot_cache.due = True

    def logits(self):
        """The logits of the last step (the head runs on the GPU; HotCache
        runs on the host meanwhile)."""
        cfg = self.cfg
        m = self.model.M("lm_head")
        if self.head is None:
            w = np.ascontiguousarray(m.data)
            self.head = Buffer(w.nbytes)
            self.head.upload(w)
            self.out = Buffer(4 * m.rows)
            self.host_logits = pinned((1, m.rows))
        xn = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        _check(lib().gg_q6k_head(self.head.ptr, xn, self.out.ptr, m.rows, cfg.hidden_size, 0.0, 1))
        if self.hot_cache is not None and self.hot_cache.due:
            self.hot_cache.observe()
        _check(lib().gg_d2h(self.host_logits.ctypes.data, self.out.ptr, m.rows * 4))
        return self.host_logits[0].copy()


class _DevCache:
    """The device copy of the arrays of a QwenCache: the keys and values of
    the full layers, the convolution and the state of the linear layers."""

    def __init__(self):
        self.bufs = {}

    def attach(self, cache):
        self.release()
        arrays = [a for kv in cache.kv.values() for a in kv] if isinstance(cache.kv, dict) else \
            [a for kv in cache.kv if kv is not None for a in kv]
        arrays += list(cache.conv.values()) if isinstance(cache.conv, dict) else \
            [a for a in cache.conv if a is not None]
        arrays += list(cache.state.values()) if isinstance(cache.state, dict) else \
            [a for a in cache.state if a is not None]
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
