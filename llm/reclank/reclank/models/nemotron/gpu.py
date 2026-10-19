"""Nemotron 3.5 Bolt on the GPU (P4 of docs/NEMOTRON_BOLT_PLAN.md): the program of
compile_nemotron_step (reclank/models/nemotron/kq.py) on the GPU runner, with

- the dense part on the GPU as stored (Q8_0, KQ_NV4 rows of the NVFP4, IQ2_XXS shared experts,
  float32 routers and vectors), the head (Q8_0) too; the embedding rows from the host;
- the Mamba state and the float rows of the attention's keys and values on the GPU (the cache);
- the routed experts split: a fixed hot set of each layer on the GPU (the most used experts of a
  routing profile, else the first ones; as much as the memory holds), the rest on the CPU at the
  same time: NM_SPLIT marks the cold pairs, SIGNAL copies the latent rows and the pairs to pinned
  memory, NM_HOT_MOE runs the hot pairs on the GPU while a CPU_TASK runs NM_MOE on the cold ones
  (the full stacks in host memory), AWAIT brings its rows back (no wait when no pair is cold).

Everything runs on Pascal (sm_61): float32 math, no tensor cores.

    m = NemotronKQ(path)
    g = NemotronGPU(m, ctx=32768)          # hot_gb: the memory of the hot experts
    logits = g.prefill(ids)                # the last token's
    logits = g.step(token)
"""
from __future__ import annotations

import os
import time

import numpy as np

from reclank.backends.cpu import cops
from reclank.backends.cuda.gpumm import mem_info, pinned
from reclank.backends.cuda.runtime import GPUProgram, Mirror
from reclank.formats import iq2
from reclank.ir import program as P
from reclank.models.nemotron.kq import KQ_IQ2R, KQ_NV4, KQ_Q8_0, LM, NemotronKQCache, compile_nemotron_step
from reclank.models.qwen.gpu35 import _DevCache

# the programs of a prompt: a prompt runs in parts of exactly these sizes, the largest first (no
# padding: the head reads the last row of the part)
GROUPS = (4, 16, 64, 256, 1024)
ATT_PARTS = 256                 # NM_ATTN2: the most (token, chunk) pairs of a run (backends/cuda nm.cu)


def load_counts(path):
    """The routing profile of the model (scripts/nemotron_profile.py: expert_profile.npz beside
    the model, or RECLANK_EXPERT_COUNTS): {layer: the count of each expert}, or None."""
    f = os.environ.get("RECLANK_EXPERT_COUNTS") or os.path.join(path, "expert_profile.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    return {int(k[1:]): z[k] for k in z.files if k.startswith("l")}


def task_threads():
    """The team of the CPU's part of a run (RECLANK_GPU_CPU_THREADS, else): half the CPUs of the
    process on a machine of two or more NUMA nodes (the 2-socket Xeon: 22 of 47 gave the CPU part
    of a layer 0.38 ms, 44 0.56), else all but one."""
    n = len(os.sched_getaffinity(0))
    try:
        from reclank.runtime import numa
        many = numa.enabled()
    except Exception:
        many = False
    return max(1, n // 2 if many else n - 1)
# the GPU memory the hot experts leave besides the dense part, the cache and the prompt programs
# (NemotronGPU.prompt_bytes): the step's program, the graphs, the allocator's rounding of the cache
# (74 MB), the runtime's own. A fixed 1 GB for all of it ran the RTX 3060 out of memory in its first
# prompt (the programs of the groups take 0.88 GB with their graphs); with prompt_bytes and 0.5 a
# prompt of 4K tokens left 0.10 GB before the groups of 4, 16 and 256 (0.24 GB) were made.
MARGIN_GB = float(os.environ.get("RECLANK_NM_MARGIN_GB", "0.9"))


# the runs of at least NemotronGPU.zc_min rows run every expert on the GPU, the cold ones read from
# the host memory the GPU maps (NM_GROUP_MOE; an expert device runs its experts beside, through the
# CPU's task); RECLANK_NM_ZC=0: the split of the steps (the CPU's cold experts) for all.
# RECLANK_NM_ZC_MIN (default 1: the steps too). The 3060 (28-36% hot): alone, a step 7.3 -> 20.8 tok/s
# (4K of the docs); with the 1070 a step on a code prompt 15.2 split -> 25.0 here (on the docs of the
# routing profile 26.1 -> 24.0: the split's CPU part is small only where the profile holds), the
# drafter's verify group 305 -> 148 ms (the 1070 idle) -> 82 ms (its experts beside).
ZC = os.environ.get("RECLANK_NM_ZC", "1") != "0"
ZC_MIN = int(os.environ["RECLANK_NM_ZC_MIN"]) if os.environ.get("RECLANK_NM_ZC_MIN") else None
# the slowest link to the host for it (GB/s, measured at start): a group of 1024 tokens pulls the cold
# experts once (~12 GB with a quarter hot), so even 0.8 GB/s (a P102-100 at x4 Gen1, ~64 tok/s) beats
# 4-6 cores (~30 tok/s); below this the CPU takes the cold experts of a prompt
ZC_GBPS = float(os.environ.get("RECLANK_NM_ZC_GBPS", "0.5"))


def h2d_gbps(nbytes=64 << 20, reps=4):
    """The bandwidth of copies from pinned host memory to the GPU, GB/s."""
    import time
    from reclank.backends.cuda.gpumm import Buffer
    from reclank.backends.cuda.runtime import _check, lib
    h = pinned((nbytes,), np.uint8)
    h[:] = 1
    b = Buffer(nbytes)
    _check(lib().rc_gpu_h2d(b.ptr, h.ctypes.data, nbytes))
    _check(lib().rc_gpu_d2h(h.ctypes.data, b.ptr, 16))         # (a copy to the host waits for the stream)
    t0 = time.time()
    for _ in range(reps):
        _check(lib().rc_gpu_h2d(b.ptr, h.ctypes.data, nbytes))
    _check(lib().rc_gpu_d2h(h.ctypes.data, b.ptr, 16))
    dt = time.time() - t0
    b.free()
    return reps * nbytes / dt / 1e9

# the IQ2_XXS products on int8 x (dp4a: NM_QX, NM_IQ2_LIN, the int8 form of NM_HOT_MOE);
# RECLANK_NM_I8=0 keeps the float products
I8 = os.environ.get("RECLANK_NM_I8", "1") != "0"
# x of the int8 products as int16 in two int8 planes (q = 256 hi + lo: two dp4a for each 4 values;
# int8 x left ~0.5% noise in each product); RECLANK_NM_X16=0: int8 x
X16 = os.environ.get("RECLANK_NM_X16", "1") != "0"
# the tiled int8 product of the prompt groups (NM_GEMM8); RECLANK_NM_GEMM8=0: the row kernels
GEMM8 = os.environ.get("RECLANK_NM_GEMM8", "1") != "0"
# a group of 2 to 8 rows (a verify group of the drafter's block) on the tensor cores (sm_80 up;
# RECLANK_NM_TC8=0: the rows kernels)
TC8 = os.environ.get("RECLANK_NM_TC8", "1") != "0" and os.environ.get("RECLANK_NM_TC", "1") != "0"
# the int8 x of the products made by the record that writes the float rows (NM_ADDNORM, NM_MAMBA's
# norm, RELU2, NM_HOT_MOE: their operands q, s, ql), once for all the products on them; RECLANK_NM_QFUSE=0:
# an NM_QX record before each product
QFUSE = os.environ.get("RECLANK_NM_QFUSE", "1") != "0"
# (the operand of the float rows a record writes, its operand q)
_QAT = {P.NM_ADDNORM: (4, 8), P.NM_MAMBA: (9, 20), P.RELU2: (1, 3), P.NM_HOT_MOE: (8, 19)}


class _View:
    """nbytes of a device Buffer from byte off (a set of the Mamba states: _NmDevCache)."""

    def __init__(self, buf, off, nbytes):
        self.buf, self.off, self.nbytes = buf, off, nbytes

    @property
    def ptr(self):
        return self.buf.ptr + self.off

    def upload(self, a):
        self.buf.upload_at(a, self.off)

    def download(self, a):
        self.buf.download_at(a, self.off)


class _NmDevCache(_DevCache):
    """The device cache of NemotronGPU: the Mamba states of a layer, both sets, in one buffer (c.st);
    the views of the sets (c.conv, c.S, c.other) at their offsets in it."""

    def attach(self, cache):
        import types
        super().attach(types.SimpleNamespace(kv=cache.kv, conv={}, state=cache.st, max_len=cache.max_len,
                                             n=getattr(cache, "n", 1)))
        self.views = {}
        for i, st in cache.st.items():
            b = self.bufs[id(st)][1]
            for v in (cache.conv[i], cache.S[i]) + cache.other[i]:
                self.views[id(v)] = (v, _View(b, v.ctypes.data - st.ctypes.data, v.nbytes))

    def buf(self, a):
        """The device Buffer (or _View) of an array of the cache."""
        e = self.bufs.get(id(a)) or self.views.get(id(a))
        return e[1]

    def device(self, a):
        e = self.bufs.get(id(a)) or getattr(self, "views", {}).get(id(a))
        return None if e is None else e[1].ptr


class NmEmit:
    """What compile_nemotron_step asks of the GPU: the attention over the float rows of the cache,
    the split experts of a layer, and the IQ2_XXS matrices (the shared experts) on int8 x."""

    def __init__(self, dev, t):
        self.dev, self.t = dev, t
        self._qx = {}
        self._scr = {}
        self._new = {}                  # cols: (src, the record that wrote it) since quant
        self._done = {}                 # cols: the src the int8 rows of cols hold
        # the flags of the last CPU task of this program: its GP_AWAIT does not wait with no cold pair
        # (cnt 0), so the next signal waits for it before it writes the pinned rows that task reads
        # (bufs: one set for all the layers; GP_SIGNAL operand 11, signal_wait)
        self._last_fl = None

    def quant(self, prog, src, cols):
        """compile_nemotron_step: the float rows src (t x cols) are new. Their int8 rows are made once,
        before the first product on them: by the record that wrote them (its operands q, s, ql) when
        it is the last record on src, else an NM_QX record."""
        enc, rec = prog._enc(src), None
        for r in reversed(prog.recs):
            if enc in r[1]:
                at = _QAT.get(r[0])
                rec = r if at is not None and len(r[1]) > at[0] and r[1][at[0]] == enc else None
                break
        self._new[cols] = (src, rec)
        self._done.pop(cols, None)

    def linear(self, prog, m, src, out, t):
        """A matrix of the step; False: the record of the CPU (KQ_LINEAR). On int8 x (NM_QX): the
        IQ2_XXS ones always (NM_IQ2_LIN), and in a prompt group the NVFP4 (more than 8 rows: the rows
        kernel k_nv4_gemv2 takes up to 8, a verify group of the drafter) and Q8_0 ones (more than 16)
        too (NM_GEMM8: the tiled product)."""
        if not I8:
            return False
        # (2 to 8 rows: k_nm_t8v on the tensor cores, k_nm_t8d (dp4a) without; more than 16: the tiles)
        small = 2 <= t <= 8 and TC8
        gemm = GEMM8 and m.type in (KQ_IQ2R, KQ_NV4, KQ_Q8_0) and (small or t > 16 or (t > 8 and m.type == KQ_NV4))
        if not gemm and m.type != KQ_IQ2R:
            return False
        q = self._qx.get(m.cols)
        if q is None:
            q = self._qx[m.cols] = (np.zeros((t, m.cols), np.int8), np.zeros((t, m.cols // 32), np.float32),
                                    np.zeros((t, m.cols), np.int8) if X16 else None)
        new = self._new.get(m.cols)
        if new is None or new[0] is not src or self._done.get(m.cols) is not src:
            rec = new[1] if new is not None and new[0] is src and QFUSE and m.cols % 32 == 0 else None
            if rec is not None:
                args, at = rec[1], _QAT[rec[0]][1]
                args += [prog._enc(None)] * (at - len(args)) + [prog._enc(a) for a in q]
            else:
                prog.emit(P.NM_QX, src, t, m.cols, q[0], q[1], q[2])
            if new is not None and new[0] is src:
                self._done[m.cols] = src
        if gemm:
            prog.emit(P.NM_GEMM8, q[0], q[1], m.data, m.type, m.rows, m.cols, t, out, q[2])
        else:
            prog.emit(P.NM_IQ2_LIN, q[0], q[1], m.data, m.rows, m.cols, t, out, q[2])
        return True

    def attention(self, prog, i, q, k, v, out, t):
        """The keys and values into the bfloat16 rows of the cache (NM_KV), then the attention in
        chunks of positions (NM_ATTN2: each block one KV head, its 16 query heads, a chunk; then
        the chunks merged)."""
        cfg = self.dev.cfg
        K, V, pos = prog.slot("K.%d" % i), prog.slot("V.%d" % i), prog.slot("pos")
        prog.emit(P.NM_KV, k, v, K, V, cfg.kv_heads * cfg.head_dim, t, pos)
        prog.emit(P.NM_ATTN2, q, K, V, out, cfg.heads, cfg.kv_heads, cfg.head_dim, t, pos,
                  self.dev.attn_scratch(), float(cfg.head_dim ** -0.5))

    def scratch(self, name, n, dt=np.float32):
        """A buffer of the program that every layer shares (the records run in order): n values of
        the largest layer."""
        a = self._scr.get(name)
        if a is None or a.size < n:
            a = self._scr[name] = np.zeros(n, dt)
        return a

    def _hot(self, prog, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t, Im):
        S = self.scratch
        if I8:
            prog.emit(P.NM_HOT_MOE, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t,
                      S("qx", t * L, np.int8), S("sx", t * L // 32), S("qa", t * k * Im, np.int8),
                      S("sa", t * k * Im // 32), S("qxl", t * L, np.int8) if X16 else None,
                      S("qal", t * k * Im, np.int8) if X16 else None)
        else:
            prog.emit(P.NM_HOT_MOE, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t)

    def moe(self, prog, i, lat, idx, val, out, t):
        d, cfg = self.dev, self.dev.cfg
        E, inner, k = cfg.moe[i]
        L = cfg.latent
        Im = max(n for _, n, _ in cfg.moe.values())
        Em = max(e for e, _, _ in cfg.moe.values())
        S = self.scratch
        if t >= d.zc_min and d.zc_tables is not None:
            pup, pdn = d.zc_tables[i]
            gidx, gout, xd = idx, out, d.xdev if t <= 8 else None
            if xd is not None:
                # an expert device: the pairs of its experts to it (through the CPU's task, as a step's
                # cold pairs), the others here; the sums added after
                held = xd.held[i, :E].astype(bool)
                mine, rest = np.where(held, -1, 0).astype(np.int32), np.where(held, 0, -1).astype(np.int32)
                d._xkeep += [mine, rest]
                bufs = d.bufs(t)
                cnt = S("cnt", 1, np.int32)
                gidx, gout, part = S("zidx", t * k, np.int32), S("gout", t * L), S("part", t * L)
                prog.emit(P.NM_SPLIT, idx, mine, S("cold", t * k, np.int32), cnt, t * k, k, prog.slot("nreal"))
                prog.emit(P.NM_SPLIT, idx, rest, gidx, S("zcnt", 1, np.int32), t * k, k, prog.slot("nreal"))
                fl = pinned((2,), np.int64)
                fl[:] = 0
                prog.keep.append(fl)
                seq = prog.slot("seq")
                prev, self._last_fl = self._last_fl, fl
                prog.emit(P.SIGNAL, fl[0:1], seq, lat, bufs["lat"], bufs["lat"].nbytes, val, bufs["val"],
                          bufs["val"].nbytes, S("cold", t * k, np.int32), bufs["cold"], bufs["cold"].nbytes,
                          prev[1:2] if prev is not None else None)
            prog.emit(P.NM_GROUP_MOE, lat, gidx, val, pup, pdn, S("qx", t * L, np.int8), S("sx", t * L // 32),
                      S("act", t * k * Im), S("qa", t * k * Im, np.int8), S("sa", t * k * Im // 32),
                      S("de", t * k * L), gout, S("work", 2 * Em + 1 + t * k, np.int32), k, L, inner, t, E,
                      prog.slot("nreal"), S("qxl", t * L, np.int8) if X16 else None,
                      S("qal", t * k * Im, np.int8) if X16 else None)
            if xd is not None:
                # the CPU's task: the post, NM_MOE on what the device refused (else nothing: zeros), the join
                cc = P.Program()
                upm, dnm = d.m.experts(i, "up"), d.m.experts(i, "down")
                lcold, posted = np.zeros(t * k, np.int32), np.zeros(1, np.int64)
                d._xkeep += [lcold, posted]
                cc.emit(P.NM_REMOTE_POST, bufs["lat"], bufs["cold"], bufs["val"], t * k, xd.ctx_bytes, i, lcold,
                        posted, k)
                cc.emit(P.NM_MOE, bufs["lat"], lcold, bufs["val"], t, k, upm.data, upm.type, dnm.data, dnm.type,
                        L, inner, bufs["scratch"], bufs["out"], int(X16))
                cc.emit(P.REMOTE_JOIN, xd.ctx_bytes, posted, bufs["out"])
                cpu = cc.finish()
                d.cpu_progs.append(cpu)
                prog.emit(P.CPU_TASK, cpu.buf, fl[0:1], fl[1:2], seq)
                prog.emit(P.AWAIT, fl[1:2], seq, bufs["out"], part, t * L * 4, cnt)
                prog.emit(P.ADD, gout, part, out, t * L)
            return
        slots, up, down = d.hot[i]
        act, de = S("act", t * k * Im), S("de", t * k * L)
        if d.hot_n[i] == E:
            # every expert of the layer on the GPU: no part of the CPU (split, signal, task, await, add)
            self._hot(prog, lat, idx, val, slots, up, down, act, de, out, k, L, inner, t, Im)
            return
        bufs = d.bufs(t)
        cold, cnt = S("cold", t * k, np.int32), S("cnt", 1, np.int32)
        gout, part = S("gout", t * L), S("part", t * L)
        prog.emit(P.NM_SPLIT, idx, slots, cold, cnt, t * k, k, prog.slot("nreal"))
        fl = pinned((2,), np.int64)
        fl[:] = 0
        prog.keep.append(fl)
        seq = prog.slot("seq")
        prev, self._last_fl = self._last_fl, fl
        prog.emit(P.SIGNAL, fl[0:1], seq, lat, bufs["lat"], bufs["lat"].nbytes, val, bufs["val"],
                  bufs["val"].nbytes, cold, bufs["cold"], bufs["cold"].nbytes, prev[1:2] if prev is not None else None)
        self._hot(prog, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t, Im)
        # the CPU's part: the cold pairs on the full stacks in host memory (a run of up to 8 rows:
        # those of an expert device's experts on it, its sum added after)
        cc = P.Program()
        upm, dnm = d.m.experts(i, "up"), d.m.experts(i, "down")
        cold_cpu, xd = bufs["cold"], d.xdev if t <= (8 if os.environ.get("RECLANK_NM_XGROUP", "1") != "0" else 1) else None
        if xd is not None:
            cold_cpu, posted = np.zeros(t * k, np.int32), np.zeros(1, np.int64)
            d._xkeep += [cold_cpu, posted]
            cc.emit(P.NM_REMOTE_POST, bufs["lat"], bufs["cold"], bufs["val"], t * k, xd.ctx_bytes, i, cold_cpu,
                    posted, k)
        cc.emit(P.NM_MOE, bufs["lat"], cold_cpu, bufs["val"], t, k, upm.data, upm.type, dnm.data, dnm.type,
                L, inner, bufs["scratch"], bufs["out"], int(X16))
        if xd is not None:
            cc.emit(P.REMOTE_JOIN, xd.ctx_bytes, posted, bufs["out"])
        cpu = cc.finish()
        d.cpu_progs.append(cpu)
        prog.emit(P.CPU_TASK, cpu.buf, fl[0:1], fl[1:2], seq)
        prog.emit(P.AWAIT, fl[1:2], seq, bufs["out"], part, t * L * 4, cnt)
        prog.emit(P.ADD, gout, part, out, t * L)


class NemotronGPU:
    def __init__(self, model, ctx=8192, hot_gb=None, counts=None, graph=True):
        self.m, self.cfg = model, model.cfg
        cfg = self.cfg
        self.ctx = ctx
        self.graph = graph
        self.cpu_progs = []
        self._bufs = {}
        self.mirror = Mirror()
        self.cache = NemotronKQCache(cfg, ctx, kv="bf16")
        self._two_sets()
        self._ascr = None
        self.cache_dev = _NmDevCache()
        # the hot experts: the share of every layer's experts that the memory holds
        per = {i: inner * iq2.row_bytes(cfg.latent) + cfg.latent * iq2.row_bytes(inner) for i, (_E, inner, _k) in cfg.moe.items()}
        total = sum(per[i] * cfg.moe[i][0] for i in cfg.moe)
        if hot_gb is None and os.environ.get("RECLANK_GPU_HOT_GB"):
            hot_gb = float(os.environ["RECLANK_GPU_HOT_GB"])
        if hot_gb is None:
            free = mem_info()[0]
            hot_gb = max(0.0, (free - self.dense_bytes() - self.kv_bytes(ctx) - self.prompt_bytes()) / 1e9 -
                         MARGIN_GB)
        share = min(1.0, hot_gb * 1e9 / total)
        self.hot, self.hot_n = {}, {}
        for i, (E, inner, k) in cfg.moe.items():
            n = int(share * E)
            if counts is not None and i in counts:
                pick = np.sort(np.argsort(-np.asarray(counts[i]), kind="stable")[:n])
            else:
                pick = np.arange(n)
            self.hot_n[i] = n
            self.hot[i] = self._hot_stacks(i, pick)
        self.hot_share = share
        self.hot_bytes = sum(up.nbytes + down.nbytes for _s, up, down in self.hot.values())
        self.counts = counts
        # a second GPU of the process with cold experts (RECLANK_EXPERT_GPUS: backends/cuda/devices.py)
        from reclank.backends.cuda.devices import nm_expert_device
        self.xdev = nm_expert_device(self)
        self._xkeep = []
        self.zc_min = ZC_MIN if ZC_MIN is not None else 1
        self.progs = {}
        self.zc_tables = None
        self._zc_reg = {}
        self._prog(1)
        self.link_gbps = h2d_gbps()
        if ZC and self.link_gbps >= ZC_GBPS:
            self._zero_copy()
            if self.zc_min <= 1:
                # (the step's program again, now with the tables: its experts by NM_GROUP_MOE too)
                self.progs.pop(1, None)
                self._prog(1)

    # ---- the sizes

    def dense_bytes(self):
        """The bytes of the weights on the GPU but the routed experts."""
        n = 0
        cfg = self.cfg
        for i, kind in enumerate(cfg.kinds):
            p = "language_model.backbone.layers.%d.mixer." % i
            if kind == "mamba":
                names = ("in_proj", "out_proj")
            elif kind == "attention":
                names = ("q_proj", "k_proj", "v_proj", "o_proj")
            else:
                names = ("gate", "fc1_latent_proj", "fc2_latent_proj", "shared_experts.up_proj",
                         "shared_experts.down_proj")
            for nm in names:
                n += self.m.M(p + nm).data.nbytes
        return n + self.m.M(LM + "lm_head").data.nbytes

    def prompt_bytes(self):
        """The device memory of the programs of the prompt groups (GROUPS, made at the first
        prompt): each a part that grows with its rows (the step's activations, which the layers
        share; the routed experts' scratch; the int8 x of the products) and one that does not
        (rc_gpu_load's own buffers). The 3060 measured 538 KB a row and 19.5 MB a program for the
        74B (824 MB for the five; this gives 552 KB, 853 MB)."""
        c = self.cfg
        hid, L, inner, conv = c.hidden, c.latent, c.m_inner, c.conv_dim
        hq, hk = c.heads * c.head_dim, c.kv_heads * c.head_dim
        Emax = max(e for e, _, _ in c.moe.values())
        Im = max(n for _, n, _ in c.moe.values())
        k = max(kk for _, _, kk in c.moe.values())
        wide = max(inner, c.shared_inner, hid)
        act = 4 * (5 * hid + 2 * inner + 2 * conv + c.m_heads + 3 * hq + 2 * hk + Emax + 2 * k + 2 * L +
                   c.shared_inner) + wide * (1 + 4 / 32 + 4 / 16)
        moe = 6 * k * Im + 4 * k * Im / 32 + 4 * k * L + 10 * L + 4 * L / 32 + 4 * k
        qx = sum(2 * n + 4 * n / 32 for n in {hid, inner, c.shared_inner, L, hq})
        return int(sum((act + moe + qx) * t + 20e6 for t in GROUPS))

    def kv_bytes(self, ctx):
        c = self.cfg
        attn = sum(1 for k in c.kinds if k == "attention")
        mamba = sum(1 for k in c.kinds if k == "mamba")
        return attn * 2 * ctx * c.kv_heads * c.head_dim * 2 + mamba * 2 * (
            c.m_heads * c.m_dim * c.m_state + (c.conv - 1) * c.conv_dim) * 4

    def _hot_stacks(self, i, pick):
        """(slots, up, down): the GPU slot of each expert (or -1), the hot experts' blocks."""
        E = self.cfg.moe[i][0]
        slots = np.full(E, -1, np.int32)
        ups, dns = [], []
        upm, dnm = self.m.experts(i, "up"), self.m.experts(i, "down")
        pu, pd = upm.data.nbytes // E, dnm.data.nbytes // E
        for s, e in enumerate(pick):
            slots[e] = s
            ups.append(upm.data[e * pu:(e + 1) * pu])
            dns.append(dnm.data[e * pd:(e + 1) * pd])
        up = np.concatenate(ups) if ups else np.zeros(256, np.uint8)
        down = np.concatenate(dns) if dns else np.zeros(256, np.uint8)
        return slots, up, down

    def _zero_copy(self):
        """The tables of NM_GROUP_MOE: the device address of each expert's up and down rows, its
        slot on the GPU if it has one (the step's program put the hot stacks there), else the
        stacks in host memory mapped for the GPU (registered once; 17 GB pinned)."""
        from reclank.backends.cuda.runtime import lib
        L = lib()
        tables = {}
        for i, (E, inner, k) in self.cfg.moe.items():
            slots, up, down = self.hot[i]
            upm, dnm = self.m.experts(i, "up"), self.m.experts(i, "down")
            devs = []
            for a in (upm.data, dnm.data):
                d = self._zc_reg.get(id(a))
                if d is None:
                    d = L.rc_gpu_host_register_rw(a.ctypes.data, a.nbytes)
                    if not d:
                        print("NemotronGPU: the experts cannot be mapped for the GPU (%s): prompt groups split as "
                              "the steps" % L.rc_gpu_last_error().decode(), flush=True)
                        self.zc_tables = None
                        return
                    self._zc_reg[id(a)] = d
                devs.append(d)
            pu, pd = upm.data.nbytes // E, dnm.data.nbytes // E
            hu = self.mirror.translate(up.ctypes.data) if self.hot_n[i] else None
            hd = self.mirror.translate(down.ctypes.data) if self.hot_n[i] else None
            pup, pdn = np.zeros(E, np.int64), np.zeros(E, np.int64)
            for e in range(E):
                s = int(slots[e])
                pup[e] = hu + s * pu if s >= 0 else devs[0] + e * pu
                pdn[e] = hd + s * pd if s >= 0 else devs[1] + e * pd
            tables[i] = (pup, pdn)
        self.zc_tables = tables

    def attn_scratch(self):
        """The partial sums of the chunks of NM_ATTN2: at most ATT_PARTS (token, chunk) pairs of
        each query head, hd + 2 floats each."""
        if self._ascr is None:
            c = self.cfg
            self._ascr = np.zeros(max(ATT_PARTS, GROUPS[-1]) * c.heads * (c.head_dim + 2), np.float32)
        return self._ascr

    def bufs(self, t):
        """The pinned rows of the CPU's part of a run of t rows (all layers share them: the tasks
        run in order, and each layer's AWAIT copies its rows before the next layer signals; with no
        cold pair the AWAIT does not wait, so the next signal waits for the task: NmEmit._last_fl)."""
        b = self._bufs.get(t)
        if b is None:
            cfg = self.cfg
            k = max(kk for _, _, kk in cfg.moe.values())
            inner = max(n for _, n, _ in cfg.moe.values())
            L = cfg.latent
            b = self._bufs[t] = {"lat": pinned((t, L)), "val": pinned((t, k)), "cold": pinned((t, k), np.int32),
                                 "out": pinned((t, L)), "scratch": cops.nm_moe_scratch(t, k, L, inner)}
        return b

    # ---- the programs

    def _prog(self, t):
        e = self.progs.get(t)
        if e is None:
            prog = compile_nemotron_step(self.m, t, em=NmEmit(self, t))
            # the head on the last row of the group
            head = self.m.M(LM + "lm_head")
            logits = np.zeros((1, head.rows), np.float32)
            xn = prog.names["xn"]
            prog.recs.append((P.KQ_LINEAR, [prog._enc(a) for a in (
                None, None, None, xn[t - 1:t], head.data, head.type, head.rows, head.cols, 1, logits)]))
            prog.names["logits"] = logits
            prog.finish()
            g = GPUProgram(prog, graph=self.graph, mirror=self.mirror)
            e = self.progs[t] = (prog, g)
        return e

    def _run(self, ids, pos):
        t = len(ids)
        size = next((s for s in (1,) + GROUPS if s >= t), None)
        assert size is not None, "a run of at most %d tokens" % GROUPS[-1]
        # (NM_KV writes the rows of all size tokens at pos: past max_len they were the next buffers on
        # the GPU, and a later run read a broken one)
        if pos + size > self.cache.max_len:
            raise ValueError("the cache holds %d positions: a run of %d at %d" % (self.cache.max_len, size, pos))
        prog, g = self._prog(size)
        x = prog.names["H"]
        x[:t] = self.m.embed(ids)
        x[t:] = 0.0
        g.upload("H")
        kw = {"pos": pos, "nreal": t, "hs": self.cfg.head_dim}
        kw.update(self.cache.params())
        g.bind(kw, self.cache_dev, scratch=("scores",))
        g.run()
        g.download("logits")
        self.cache.n = pos + t
        return prog.names["logits"][0].copy()

    # ---- the API

    def _two_sets(self):
        """The Mamba states of each layer (the conv rows, the state) in two sets, in one array of the
        layer (c.st): c.conv / c.S are views of the set in use, c.other the other's. A verify group of
        spec decoding and its replay read the set in use and write the other (NM_MAMBA operand sd:
        self.sdelta()), then swap() makes it the one in use: no copies of the states."""
        c = self.cache
        c.st, c.other = {}, {}
        for i in list(c.conv):
            nc, ns = c.conv[i].size, c.S[i].size
            st = c.st[i] = np.zeros(2 * (nc + ns), np.float32)
            sh = (c.conv[i].shape, c.S[i].shape)
            c.conv[i], c.S[i] = st[:nc].reshape(sh[0]), st[nc:nc + ns].reshape(sh[1])
            c.other[i] = (st[nc + ns:2 * nc + ns].reshape(sh[0]), st[2 * nc + ns:].reshape(sh[1]))
            self._set_bytes = (nc + ns) * 4      # (the same in every Mamba layer)
        c.cur = 0

    def sdelta(self):
        """The bytes from the set in use to the other (NM_MAMBA's sd)."""
        return self._set_bytes if self.cache.cur == 0 else -self._set_bytes

    def swap(self):
        """The other set of the Mamba states in use."""
        c = self.cache
        for i in c.st:
            (c.conv[i], c.S[i]), c.other[i] = c.other[i], (c.conv[i], c.S[i])
        c.cur ^= 1

    def reset(self):
        """A fresh cache (zeros) on the GPU."""
        c = self.cache
        if c.cur:
            self.swap()
        for a in c.st.values():
            a[:] = 0
        c.n = 0
        self.cache_dev.attach(c)
        self.cache_dev.finish()

    def prefill(self, ids, pos=None):
        """Run the tokens ids from pos (default: the cache's n); the logits of the last."""
        pos = self.cache.n if pos is None else pos
        if pos == 0:
            self.reset()
        out = None
        c0 = 0
        while c0 < len(ids):
            left = len(ids) - c0
            size = next((s for s in GROUPS[::-1] if s <= left), 1)
            out = self._run(ids[c0:c0 + size], pos + c0)
            c0 += size
        return out

    def step(self, token):
        return self._run([token], self.cache.n)
