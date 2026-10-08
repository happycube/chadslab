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
  the GPU (GP_FETCH, as for the 26B), and GP_KQ_GROUP_MOE computes them;
- a mixed group (a part of a prompt of at least MIX_MIN tokens, the
  default): after the router of each layer, GP_MOE_PLAN gives the GPU the
  experts with the most tokens (copied to a buffer) and the CPU the rest
  (mix; see np_gemma/qwen4_gpu.py). Qwen3.6, pp2048: 547 tok/s with the
  large groups, 960 with mixed groups of 2048 rows.

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
import sys
import time

import numpy as np

from . import cops
from . import program as P
from .gpu import GPUProgram, HotCache, _check, lib
from .gpumm import Buffer, DeviceCache, ExpertPool, ProgramLRU, mem, mem_info, pinned, weak_method
from .qwen import cache_params, compile_qwen_step, media_inputs, rope_positions

Q8_0, Q8_R = 8, 100     # the ggml type; its rows for the GPU (csrc/gpu.cu)
MT = 16                 # the largest small group (MT_MAX of csrc/gpu.cu)
# The shortest part of a prompt that runs as a large group. A group of 1024
# rows takes about 2.6 s (mostly the copies of the experts, 7.5 GB/s); split
# groups of 256 rows take about as long for 700 tokens (QWEN_PLAN.md).
FETCH_MIN = int(os.environ.get("NP_GEMMA_GPU_FETCH_MIN", "700"))
SPLIT_SIZES = (16, 64, 128, 256)  # the split groups of a prompt
FETCH_SIZES = (1024,)             # the large groups
# The handoff to the CPU inside the graph of a step or a small group (flags
# in pinned memory; see gpu.FLAGS). NP_GEMMA_GPU_FLAGS=0 keeps the boundary
# records.
FLAGS = os.environ.get("NP_GEMMA_GPU_FLAGS", "1") == "1"
# The mixed groups of a prompt (QwenGPU.mix; see the module text of
# np_gemma/qwen4_gpu.py).
# The rows of a mixed group: 2048 gave 491 tok/s at pp8192 for Qwen3.8 (402
# with 1024) and 932 for Qwen3.6 (754). With the calibration that balances
# the waits (calibrate_mix), on the 2-socket Xeon with the clocks down and
# the cache of 262144 (29 hot experts): 4096 rows 564 tok/s at pp8192, 2048
# rows 410 (the copies, 85% of a group of 2048, serve twice the tokens; the
# GPU is busy 95% of a group of 4096).
MIX_SIZE = int(os.environ.get("NP_GEMMA_GPU_MIX", "4096"))
MIX_MIN = int(os.environ.get("NP_GEMMA_GPU_MIX_MIN", "256"))  # the shortest prompt part for one
# The model of the costs of GP_MOE_PLAN, in ns: for each expert on the CPU,
# and for each of its tokens; for each expert that the GPU copies. The
# experts in groups of 16 rows (KQ_NVX, 53) are faster on the CPU.
MIX_CPU = {53: (50000, 5000)}
# For the other formats (the K quants of Qwen3.6): a sweep of the cost of a
# copy gave 938 tok/s at pp8192 with (75000, 15000) and 1031 with half of
# these costs (the same as twice the cost of a copy).
MIX_CPU_DEFAULT = (37500, 7500)


def mix_cpu_cost(xtype):
    """(for each expert, for each token) of the CPU for experts of xtype."""
    a, b = MIX_CPU.get(xtype, MIX_CPU_DEFAULT)
    return (int(os.environ.get("NP_GEMMA_GPU_MIX_CPU_A", a)),
            int(os.environ.get("NP_GEMMA_GPU_MIX_CPU_B", b)))


# the first cost of a copy (calibrate_mix moves it): an expert of the
# MIX-BF12 file (4.4 MB) took 0.37 ms at 11.8 GB/s, and the balance settled
# at 160-330 us (700 us before the balance: 292 tok/s at pp8192 against 410)
MIX_GPU = int(os.environ.get("NP_GEMMA_GPU_MIX_GPU", "300000"))
# NP_GEMMA_MOE_X16 (default 1): the experts read x with the precision of int16 (int8
# activations: KL 1.3e-2 to float32 activations on real text, the floor of
# two near-equal runs about 0.6e-2; int16 5.9e-3, the same top token 99.2%
# against 96.9%): the CPU parts as int16 split in two int8 planes (act bit 5
# of KQ_MOE, kq_quant_part16, kq_t2_tile), the GPU experts of a group in
# float32 (NP_GEMMA_GPU_MOE_F32; those of a step read float32 x already).
# A hot day, team 20: plain decode the same (16.8 tok/s), MTP 3 drafts 6%
# slower, the 8K prompt 2-11% slower once the mix calibration settles.
MOE_X16 = os.environ.get("NP_GEMMA_MOE_X16", "1") == "1"     # 0: int8
if MOE_X16:
    os.environ.setdefault("NP_GEMMA_GPU_MOE_F32", "1")
MIX_NUMA = os.environ.get("NP_GEMMA_GPU_MIX_NUMA", "1") != "0"


def _host_bytes(addr, n):
    """A copy of n bytes of host memory at addr (uint8)."""
    import ctypes as _ct
    return np.frombuffer((_ct.c_uint8 * n).from_address(addr), np.uint8).copy()


def mix_threads():
    """The team of the CPU part of a mixed group (NP_GEMMA_GPU_MIX_THREADS):
    a group has much more work than a step, so it takes more cores than the
    team of a step; the default leaves 4 of OMP_NUM_THREADS: the model
    thread and the copy workers (gpu._place_threads: their CPUs are out of
    the teams), and room for the same count of threads on each node (36 of
    the 38 other CPUs of taskset -c 0-19,24-43: 18 and 18); the other
    processes go to other CPUs (scripts/cpu_fence.sh). 0: the team of the step. The
    2-socket Xeon, a layer of a real-text group of 2048 rows: 45.3 ms with 24
    threads, 39.9 with 32, 38.7 with 40, 37.2 with 44."""
    v = os.environ.get("NP_GEMMA_GPU_MIX_THREADS")
    if v is not None:
        return int(v)
    from . import numa
    n = int(os.environ.get("OMP_NUM_THREADS", "0") or 0)
    return max(n - 4, n // 2) if numa.enabled() and n > 0 else 0
# NP_GEMMA_GPU_MIX_CAL=1 (the default) tunes the cost of a copy as the mixed
# groups run (QwenGPU.calibrate_mix); "log" only measures; 0 keeps MIX_GPU.
# On the 5060 Ti (PCIe 5) it once made Qwen3.8 slower (460 tok/s at pp8192,
# 480 without it). On the 2-socket Xeon and the 3090 (PCIe 3.0, the experts
# page-locked) the best fixed cost was about 1.2 ms (667 tok/s for a bf16
# prompt of 8192; 618 with 0.7 ms), and the calibration 680.
# NP_GEMMA_GPU_MIX_LOG=1 prints the measures.
MIX_CAL = os.environ.get("NP_GEMMA_GPU_MIX_CAL", "1") in ("1", "log")
MIX_CAL_APPLY = os.environ.get("NP_GEMMA_GPU_MIX_CAL", "1") == "1"
MIX_LOG = os.environ.get("NP_GEMMA_GPU_MIX_LOG", "0") == "1"
# The free memory that the pool leaves is the reserve of GpuMem for the C
# side (NP_GEMMA_GPU_C_RESERVE, 0.8 GB); a new program reclaims the pool's
# free-memory segments as it needs them.
# NP_GEMMA_GPU_WARM: the warm experts of HotCache in the blocks of its pool
# (gpu.ExpertPool): "lend" (the default) its segments of the room lent by
# the image encoder (QwenGPU.lend_warm; serve_qwen4 --mmproj-gpu lend) and of
# the free memory that the mixed groups of a prompt took (until a program
# needs it: free_ring); "free" also the free memory at the first decode
# step; 0 none (the pool holds only the copies of the prompts)
WARM = os.environ.get("NP_GEMMA_GPU_WARM", "lend")
# NP_GEMMA_GPU_POOL_HOT (1): the hot experts in the pool too (its "hot"
# segments), not in a store of each layer; 0 keeps the stores.
POOL_HOT = os.environ.get("NP_GEMMA_GPU_POOL_HOT", "1") != "0"
# NP_GEMMA_GPU_PREFETCH (1): a mixed group copies the experts each layer will
# likely copy (those its last group took to the GPU, the most tokens first)
# during the layer before, into blocks of the pool; its plan takes them as on
# the GPU and copies only the rest (QwenGPU._fill_prefetch). PREFETCH_FRAC:
# the share of the experts of the last group to prefetch.
PREFETCH = os.environ.get("NP_GEMMA_GPU_PREFETCH", "1") != "0"
PREFETCH_FRAC = float(os.environ.get("NP_GEMMA_GPU_PREFETCH_FRAC", "1.0"))
# NP_GEMMA_GPU_PREFETCH_COST: the cost of a prefetched copy for the size of the
# prediction, as a share of the cost of a copy of the plan (gpu_c): it runs
# while the GPU works, not on the path of the layer.
PREFETCH_COST = float(os.environ.get("NP_GEMMA_GPU_PREFETCH_COST", "1.0"))
PRE_F = 128             # the fetch events of the prefetch (gpu.cu GG_FETCH_MAX 256)
# A long context (its cache on the GPU) can leave no room for the buffer:
# then the free memory kept goes down to MIX_KEEP_MIN, for a pool of
# MIX_RING_MIN experts. A group copies about 28 experts in each layer; at
# 32K, a buffer of 0 to 3 experts gave 267 tok/s, one of 189 367 tok/s.
MIX_KEEP_MIN = float(os.environ.get("NP_GEMMA_GPU_MIX_KEEP_MIN", "0.3e9"))
MIX_RING_MIN = int(os.environ.get("NP_GEMMA_GPU_MIX_RING_MIN", "64"))


class _Emit:
    """The model as compile_qwen_step sees it for a GPU program: the products
    read copies of the weights that the GPU holds, and the experts go to
    QwenGPU.emit_moe."""

    def __init__(self, dev, t, verify, fetch, mix=False):
        self.dev = dev
        self.m = dev.model
        self.t = t
        self.verify = verify
        self.fetch = fetch
        self.mix = mix

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
        if self.mix:
            self.dev._moe_mix(prog, i, xb["src"], idx, val, slog, out)
        elif self.fetch:
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


def _on_dax(path):
    """True when the file path is on a mount with the dax option (/proc/mounts:
    ext4 or xfs on persistent memory, dax=always)."""
    try:
        real = os.path.realpath(path)
        best, opts = "", ""
        with open("/proc/mounts") as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 4 and (real == parts[1] or real.startswith(parts[1].rstrip("/") + "/")) \
                        and len(parts[1]) > len(best):
                    best, opts = parts[1], parts[3]
        return any(o == "dax" or o.startswith("dax=") and o != "dax=never" for o in opts.split(","))
    except OSError:
        return False


class QwenGPU(ProgramLRU):
    """Qwen3.5 MoE (a QwenGGUFProgram) on the GPU; see the module text.

    hot_gb is the memory for the hot experts. None takes NP_GEMMA_GPU_HOT_GB,
    or else the free memory of the GPU less the dense part, the head, the two
    buffers of the copies of a large group, and 2 GB for the rest (the
    caches, the programs, and their graphs). counts
    (layers x experts, or None) selects the first hot experts; else each
    layer starts with its first experts, and HotCache changes them."""

    def __init__(self, model, hot_gb=None, counts=None, graph=True, ctx=None):
        self.model = model
        cfg = self.cfg = model.cfg
        self.graph = graph
        self._dense = {}
        self.cpu_progs = []
        self.n_events = 0
        # the mixed groups of a prompt (mix)
        self.mix_progs = {}
        # The programs of the sizes in use, the least recently used first:
        # (the name of their dict, key). When the GPU memory runs out, a new
        # program makes room by freeing the oldest (_gpu_program).
        self._mm_init("qwen%x" % id(self))     # gpumm.ProgramLRU, GpuMem
        self.pool = None             # the pool of HotCache (ExpertPool)
        self.mix_cap = 0
        self.mix_places = None
        self.mix_desc = {}          # layer -> the desc of GP_MOE_PLAN
        self.mix_stats = {}         # layer -> the stats of its last plan
        self._nreal_h = np.zeros(1, np.int64)
        E, L = cfg.num_experts, model.n_layers
        self.E, self.L = E, L
        # The bytes of one expert of each matrix (gate, up, down), the most
        # over the layers.
        self.per = [max(model.M("layers.%d.mlp.switch_mlp.%s" % (i, n)).data.nbytes // E
                        for i in range(L)) for n in ("gate_proj", "up_proj", "down_proj")]
        per = sum(self.per)
        if hot_gb is None and os.environ.get("NP_GEMMA_GPU_HOT_GB"):
            hot_gb = float(os.environ["NP_GEMMA_GPU_HOT_GB"])
        # ctx (or NP_GEMMA_GPU_CTX): the tokens of the largest cache that will
        # be attached; the hot experts leave room for it (_kv_bytes)
        self.ctx = int(ctx or os.environ.get("NP_GEMMA_GPU_CTX", "0") or 0)
        if hot_gb is None:
            # n slots take L n per; the two buffers take 2 (E - n) per.
            head = model.M("lm_head").data.nbytes
            # the programs (NP_GEMMA_GPU_PROGRAM_GB): 2 GB, 4.5 with mixed
            # groups of 4096 rows. With the NVFP4 experts (smaller: 2 E per
            # leaves less) the graph of the mixed group of 4096 rows did not
            # fit with 2 GB, and with 3 GB only when 1.12 GB more was free (the
            # room of the image encoder, bench_prompt_real --lend-gb): a
            # server with no encoder failed at its first long prompt.
            prog_gb = float(os.environ.get("NP_GEMMA_GPU_PROGRAM_GB", 4.5 if MIX_SIZE >= 4096 else 2.0))
            room = mem_info()[0] - self._dense_bytes() - head - prog_gb * 1e9 - 2 * E * per - \
                self._kv_bytes(self.ctx)
            n = int(max(1, room // ((L - 2) * per)))
        else:
            n = int(max(1, hot_gb * 1e9 // (L * per)))
        self.n_slots = n = min(E, n)
        self._pin_experts()
        self._numa_copy_experts()
        self._dram_experts()
        self._register_experts()
        self._calib_nodes()
        # The hot experts are blocks of the pool of HotCache (gpu.ExpertPool,
        # its "hot" segments: n of each layer to start; then they follow the
        # text, with the warm ones, across the layers). A layer whose experts
        # do not fit the blocks (the MTP layer of a file of mixed types) keeps
        # a store of its own.
        self.pool = ExpertPool(self.per)
        self.stores = {}
        for i in range(L):
            if counts is not None:
                init = np.sort(np.argsort(-np.asarray(counts[i]), kind="stable")[:n])
            else:
                init = np.arange(n)
            self.stores[i] = self._store(i, init)
        self._more_stores()
        self._pool_hot()
        self.prog = self._compile(1)
        self.g = GPUProgram(self.prog, graph=graph)
        self.groups = {}        # (t, verify) -> (Program, GPUProgram)
        self.stage = None       # the two buffers of the copies of a large group
        self.tables = {}        # layer -> (gate, up, down tables, ranges)
        self.tables_version = None
        self.cache_dev = _DevCache()
        self.cache_dev.before = self._share_before
        self.cache = None
        self.head = None
        self.rows = 1
        self.last = self.g.mirror.buffer_of(self.prog.names["xn"]).ptr
        self.hot_cache = None
        self._before_hot()
        if os.environ.get("NP_GEMMA_GPU_HOT_DYN", "1") != "0":
            mir = self.g.mirror
            layers = []
            for i, st in sorted(self.stores.items()):
                # step: the selection of a step (ip) counts for the layer.
                layers.append(dict(
                    layer=i, slots=st["slots"], dslots=mir.buffer_of(st["slots"]).ptr,
                    parts=[(srcs, nb, mir.buffer_of(dst).ptr)
                           for srcs, nb, dst in st["parts"]],
                    ip=st.get("ip"), step=i < L))
            self.hot_cache = HotCache(self, layers=layers, top_k=cfg.top_k)
            self._ensure_pool()
            row_of = {e["layer"]: r for r, e in enumerate(layers)}
            for i, x, b in self._pool_init:
                self.pool.owner[b] = row_of[i] * self.E + x
        # GpuMem: all that the model has now stays (the weights, the stores of
        # the experts, the step and the programs made here); the later
        # programs, the pool, and the cache come and go
        self._mm_freeze()
        mem().add_reclaimer(10, self._mm_name + "-pool", weak_method(self._reclaim_pool))
        mem().add_reclaimer(15, self._mm_name + "-hot", weak_method(self._reclaim_hot))

    def _run_blocks(self):
        """GpuMem: each run uses the cache and the pool (the kernels read the
        segment table at run time)."""
        return self.cache_dev.blocks() + (self.pool.buffers() if self.pool is not None else [])

    def _share_before(self):
        """Before the memory a cache lends changes: the copies of HotCache
        into blocks of the pool land first (they lock their segments)."""
        if self.hot_cache is not None:
            self.hot_cache.prepare(wait=True)

    def _reclaim_pool(self, nbytes, keep):
        """GpuMem: the free-memory segments of the pool, those with the
        fewest warm experts first."""
        hc = self._warm_hc()
        if hc is not None:
            return hc.pool_shrink("free", nbytes)
        pool = self.pool
        if pool is None:
            return 0
        segs = [sg for sg in pool.segments("free")
                if not (pool.owner[sg * pool.K:(sg + 1) * pool.K] == -2).any()
                and not pool.bufs[sg].locks][:-(-int(nbytes) // pool.seg_bytes)]
        pool.drop(segs)
        return len(segs) * pool.seg_bytes

    def _reclaim_hot(self, nbytes, keep):
        """GpuMem: segments of the hot experts, those with the fewest warm
        experts first (after the free segments and the stale programs,
        before the programs in use: see ProgramLRU.STALE)."""
        hc = self._warm_hc()
        return hc.pool_shrink("hot", nbytes) if hc is not None else 0

    def _program_dicts(self):
        return ("groups", "mix_progs", "head_progs", "mtp_progs")

    def _pinned_arrays(self):
        """The stores of the experts stay: HotCache keeps the device addresses
        of their slot tables and slots. The step program holds the stores of
        the model, but only the programs of the MTP layer hold its store, so
        a freed MTP program freed them (a copy to the old address then
        crashed)."""
        out = []
        for st in self.stores.values():
            out += [st["slots"]] + [dst for _src, _nb, dst in st["parts"]]
        return out

    def _more_stores(self):
        """Stores of other layers (the MTP layer of Qwen4GPU)."""

    def _before_hot(self):
        """Before HotCache: the programs that hold the other stores."""

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
        rows of int8 values, then their scales, then zeros to a multiple of
        16 bytes (type 100, Q8_R in csrc/gpu.cu), for 16-byte loads."""
        e = self._dense.get(name)
        if e is None:
            m = self.model.M(name)
            data, type_ = m.data, m.type
            if type_ == Q8_0:
                b = data.reshape(m.rows, m.cols // 32, 34)
                pad = -(m.cols // 32 * 34) % 16      # rows of a multiple of 16 bytes
                a = np.concatenate([b[:, :, 2:].reshape(m.rows, m.cols),
                                    b[:, :, :2].reshape(m.rows, m.cols // 16),
                                    np.zeros((m.rows, pad), np.uint8)], axis=1)
                e = (np.ascontiguousarray(a).reshape(-1), Q8_R)
            else:
                e = (np.array(m.data), m.type)
            self._dense[name] = e
        return e

    def _cache_types(self):
        """The cache classes of a context on the GPU (their arrays go there)."""
        from .qwen import QwenCache
        return (QwenCache,)

    def _kv_bytes(self, ctx):
        """The bytes of the caches of ctx tokens: the bytes of a token (the
        difference of two small caches) times ctx, and the fixed state (the
        DeltaNet state, the convolutions)."""
        if ctx <= 0:
            return 0

        def nbytes(cls, n):
            seen, tot = set(), 0
            stack = [vars(cls(self.cfg, n))]
            while stack:
                x = stack.pop()
                if isinstance(x, np.ndarray):
                    if id(x) not in seen:
                        seen.add(id(x))
                        tot += x.nbytes
                elif isinstance(x, dict):
                    stack.extend(x.values())
                elif isinstance(x, (list, tuple)):
                    stack.extend(x)
            return tot
        tot = 0
        for cls in self._cache_types():
            a, b = nbytes(cls, 1024), nbytes(cls, 2048)
            tot += a + (b - a) * (ctx - 1024) / 1024
        return int(tot)

    def _pin_experts(self):
        """NP_GEMMA_GPU_ZC=n (0, the default: off): the GPU computes the
        first n cold experts of each layer of a step too, reading them over
        PCIe from a copy of the experts in host memory that the GPU can map,
        while the CPU computes the other cold experts (GP_HOT_SPLIT,
        k_kqh_nvx). The read-only map of the file cannot be registered with
        CUDA here, so the experts of each layer go to anonymous memory (the
        pages interleaved over the NUMA nodes, so both CPUs read them at the
        same rate), registered and mapped: the copy takes the memory of all
        the experts (about 68 GB for Qwen3.8). The CPU and the copies of
        HotCache then read the copy too (KMat.data). PCIe 3.0 x16 gives about
        11 GB/s: an expert of 2.8 MB in 0.25 ms, while the CPU computes about
        3. NP_GEMMA_GPU_ZC_MT: the experts of a group (an MTP verify
        group), else the same n."""
        self.zc = int(os.environ.get("NP_GEMMA_GPU_ZC", "0"))
        self.zc_mt = int(os.environ.get("NP_GEMMA_GPU_ZC_MT", str(self.zc)))
        self.zc_dev = {}
        pin = os.environ.get("NP_GEMMA_GPU_PIN")
        if (pin == "0") or (pin is None and self.zc <= 0 and self.zc_mt <= 0):
            return
        from concurrent.futures import ThreadPoolExecutor
        from . import numa
        layers = [i for i in range(self.L + 1)
                  if "blk.%d.ffn_gate_exps.weight" % i in self.model.g.tensors]
        mats = [(i, name, self.model.M("layers.%d.mlp.switch_mlp.%s" % (i, name)))
                for i in layers for name in ("gate_proj", "up_proj", "down_proj")]
        # only the KQ_NVX experts (k_kqh_nvx): the layers of other experts (the
        # Q8_0 experts of the MTP layer) keep the CPU for their cold experts
        bad = {i for i, _n, m in mats if m.type != 53}
        mats = [x for x in mats if x[0] not in bad]
        if not mats:
            self.zc = self.zc_mt = 0
            return
        t0 = time.time()

        def pin(job):
            i, name, m = job
            a = numa.empty_interleaved(m.data.nbytes)
            a[:] = m.data
            d = lib().gg_host_register_rw(a.ctypes.data, a.nbytes)
            if not d:
                raise RuntimeError(lib().gg_last_error().decode())
            return i, name, m, a, d
        with ThreadPoolExecutor(16) as ex:
            for i, name, m, a, d in ex.map(pin, mats):
                m.data = a
                self.zc_dev.setdefault(i, {})[name] = d
        for i, z in self.zc_dev.items():
            # the operand of GP_KQ_HOT_MOE: the mapped gate, up, and down
            z["table"] = np.array([z["gate_proj"], z["up_proj"], z["down_proj"]], np.int64)
        if os.environ.get("NP_GEMMA_QUIET") != "1":
            print("the experts in pinned host memory for the GPU (NP_GEMMA_GPU_ZC=%d, groups %d): "
                  "%.1f GB in %.1f s" % (self.zc, self.zc_mt, sum(m.data.nbytes for _i, _n, m in mats)
                                         / 1e9, time.time() - t0), file=sys.stderr)

    def _numa_copy_experts(self):
        """A copy of the experts on each NUMA node. NP_GEMMA_GPU_NUMA_COPY:
        "auto" (the default) when the GGUF reader staged the file from a DAX
        mount into memory of one node (gguf._stage_dax, Optane): that copy
        is the copy of its node, and the copy of the other node comes from
        it, memory to memory; "1" also from the map of a file; "0" never.
        The CPU part of a step (KQ_MOE) reads on each thread the copy of its
        own node (the operand mats1, kq_my_node), so every read is local; the
        copies of HotCache read the copy of the node of the GPU (the copy
        workers run there). It takes twice the memory of the experts (141 GB
        for Qwen3.8). From the map of a file on NVMe it gave nothing (plain
        46.3 against 47.0 tok/s)."""
        from . import numa
        self.numa_mats1 = {}
        opt = os.environ.get("NP_GEMMA_GPU_NUMA_COPY", "auto")
        g = self.model.g
        staged = None
        if getattr(g, "staged", False):
            staged = getattr(g, "expert_node", None)
            if staged is None:
                staged = getattr(g, "stage_node", None)
        if opt == "auto":
            opt = "1" if staged is not None or getattr(g, "experts_mapped", False) else "0"
        if opt != "1" or getattr(self, "zc_dev", None) or not numa.enabled():
            return
        nodes = sorted(set(numa.node_of_cpu().values()))
        if len(nodes) != 2:
            return
        layers = [i for i in range(self.L + 1)
                  if "blk.%d.ffn_gate_exps.weight" % i in g.tensors]
        mats = [(i, name, self.model.M("layers.%d.mlp.switch_mlp.%s" % (i, name)))
                for i in layers for name in ("gate_proj", "up_proj", "down_proj")]
        gnode = self._gpu_node()
        other = nodes[1] if gnode == nodes[0] else nodes[0]
        xbytes = sum(m.data.nbytes for _i, _n, m in mats)
        # the experts do not fit the node of the GPU (or NP_GEMMA_GPU_NUMA_LAYOUT
        # split): each expert on one node (or both, as room allows), each
        # node's threads on its experts only (slot0, slot1; csrc/kquants.c)
        layout = os.environ.get("NP_GEMMA_GPU_NUMA_LAYOUT", "auto")
        if layout == "split" or (layout == "auto" and staged != gnode and
                                 xbytes > self._node_budget(gnode, xbytes)):
            if self._numa_split(layers, mats, gnode, other, xbytes):
                return
        if getattr(g, "experts_mapped", False) and \
                os.environ.get("NP_GEMMA_GPU_NUMA_COPY", "auto") == "auto":
            # the experts of an overlay file (gguf.GGUFOverlay) come from its
            # map: a copy on the node of the GPU first, as a staged one
            t0 = time.time()
            for i, name, m in mats:
                src = m.data.reshape(-1).view(np.uint8)
                a0 = numa.empty_on(src.nbytes, np.uint8, gnode)
                cops.kq_memcpy_par(a0, src)
                m.data = a0
            staged = gnode
            # the pages of the file in the page cache go: else they hold
            # memory of both nodes, and the copy on the other node (bound
            # MPOL_PREFERRED) can land on the node of the GPU
            for o in g.over:
                o.drop_cache()
            if os.environ.get("NP_GEMMA_QUIET") != "1":
                print("the experts of %s on node %d: %.1f GB in %.1f s" % (
                    ", ".join(os.path.basename(o.path) for o in g.over), gnode, xbytes / 1e9,
                    time.time() - t0), file=sys.stderr)
        if staged == gnode and xbytes > 0.8 * numa.node_bytes(other):
            # the experts do not fit the other node (the Q8_0 experts of
            # Qwen3.8, 131 GB, and node 1 of the Xeon, 129 GB)
            self._numa_partial(layers, mats, gnode, other, xbytes)
            return
        t0 = time.time()
        total = 0
        for i, name, m in mats:
            src = m.data.reshape(-1).view(np.uint8)
            if staged in (gnode, other):
                # the staged copy is the copy of its node: one copy more
                a_new = numa.empty_on(src.nbytes, np.uint8, other if staged == gnode else gnode)
                cops.kq_memcpy_par(a_new, src)
                a0, a1 = (src, a_new) if staged == gnode else (a_new, src)
                total += src.nbytes
            else:
                a0 = numa.empty_on(src.nbytes, np.uint8, gnode)
                a1 = numa.empty_on(src.nbytes, np.uint8, other)
                cops.kq_memcpy_par(a0, src)
                cops.kq_memcpy_par(a1, a0)
                total += 2 * src.nbytes
            m.data = a0
            self.numa_mats1.setdefault(i, {})[name] = a1
        if os.environ.get("NP_GEMMA_QUIET") != "1":
            dt = time.time() - t0
            print("the experts on node %d and on node %d (NP_GEMMA_GPU_NUMA_COPY%s): %.1f GB copied "
                  "in %.1f s (%.1f GB/s)" % (gnode, other, ", from the staged copy" if staged is not None
                                             else "", total / 1e9, dt, total / 1e9 / max(dt, 1e-9)),
                  file=sys.stderr)

    def _node_budget(self, node, xbytes):
        """The bytes of experts for node: NP_GEMMA_GPU_NODE<n>_GB, else 70% of
        its memory less the tensors staged there (not the experts) and less
        NP_GEMMA_GPU_NODE1_KEEP_GB (4)."""
        from . import numa
        opt = os.environ.get("NP_GEMMA_GPU_NODE%d_GB" % node)
        if opt is not None and opt != "auto":
            return float(opt) * 1e9
        g = self.model.g
        on = 0
        if getattr(g, "stage_node", None) == node:
            staged = sum(a.nbytes for _lo, _hi, a in getattr(g, "_stage", ()))
            on = staged if getattr(g, "experts_mapped", False) else max(0, staged - xbytes)
        keep = float(os.environ.get("NP_GEMMA_GPU_NODE1_KEEP_GB", "4")) * 1e9
        return 0.7 * numa.node_bytes(node) - on - keep

    def _numa_split(self, layers, mats, gnode, other, xbytes):
        """The experts split over the two nodes: n0 of each layer on the node
        of the GPU, the rest on the other one, and as many as its room allows
        of the first on both (the most used first, NP_GEMMA_EXPERT_COUNTS,
        else by index); compact copies with slot0 and slot1 (the slot of
        each expert, or -1). The CPU part takes slot0 too (GP_KQ_MOE), and
        the copies to the GPU read each expert where it is (_expert_srcs).
        False when they do not fit the two nodes. Read them from the map of
        the file, not a staged copy: a staged copy of all of them stays."""
        from . import numa
        E, L = self.E, len(layers)
        per = xbytes / (L * E)
        b0, b1 = self._node_budget(gnode, xbytes), self._node_budget(other, xbytes)
        n0 = int(min(E, max(0, b0) // (L * per)))
        n1 = int(min(E, max(0, b1) // (L * per)))
        quiet = os.environ.get("NP_GEMMA_QUIET") == "1"
        if n0 + n1 < E or n0 < 1:
            if not quiet:
                print("the experts do not fit the two nodes (%d + %d of %d a layer): no split"
                      % (n0, n1, E), file=sys.stderr)
            return False
        counts = None
        path = os.environ.get("NP_GEMMA_EXPERT_COUNTS")
        if path:
            counts = np.load(path)
        t0 = time.time()
        total = 0
        self.numa_slot0, self.numa_slot1 = {}, {}
        self.numa_nb = {}
        for i in layers:
            order = (np.argsort(-counts[i], kind="stable") if counts is not None and i < counts.shape[0]
                     else np.arange(E))
            on0 = np.sort(order[:n0])
            dup = order[:max(0, n1 - (E - n0))]                     # on both
            on1 = np.sort(np.concatenate([order[n0:], dup]))
            for nd, ids, key in ((gnode, on0, 0), (other, on1, 1)):
                sl = np.full(E, -1, np.int32)
                sl[ids] = np.arange(len(ids), dtype=np.int32)
                (self.numa_slot0 if key == 0 else self.numa_slot1)[i] = sl
            for li, name, m in mats:
                if li != i:
                    continue
                nb = m.data.nbytes // E
                self.numa_nb[id(m)] = nb        # (m.data: the experts of node 0 then)
                src = m.data.reshape(E, nb)
                a0 = numa.empty_on(len(on0) * nb, np.uint8, gnode).reshape(len(on0), nb)
                a1 = numa.empty_on(len(on1) * nb, np.uint8, other).reshape(len(on1), nb)
                for k, x in enumerate(on0):
                    a0[k] = src[x]
                for k, x in enumerate(on1):
                    a1[k] = src[x]
                m.data = a0.reshape(-1)
                self.numa_mats1.setdefault(i, {})[name] = a1.reshape(-1)
                total += a0.nbytes + a1.nbytes
        self.numa_split = True
        if not quiet:
            print("the experts split over node %d (%d of each layer) and node %d (%d, %d of them on "
                  "both): %.1f GB in %.1f s" % (gnode, n0, other, len(on1), len(dup), total / 1e9,
                                               time.time() - t0), file=sys.stderr)
        return True

    def _expert_nb(self, m):
        """The bytes of an expert of the expert matrix m: m.data has the E
        experts, or after a split only those of node 0 (_numa_split)."""
        nb = getattr(self, "numa_nb", {}).get(id(m))
        return nb if nb is not None else m.data.nbytes // self.cfg.num_experts

    def _expert_srcs(self, i, name, m, nb):
        """The host address of the bytes of each expert of a matrix of layer i
        (int64 E): in m.data at its index, or (a split) at its slot of the
        copy of node 0, else of node 1."""
        E = self.E
        s0 = getattr(self, "numa_slot0", {}).get(i)
        if s0 is None:
            return m.data.ctypes.data + np.arange(E, dtype=np.int64) * nb
        s1 = self.numa_slot1[i]
        a1 = self.numa_mats1[i][name]
        return np.where(s0 >= 0, m.data.ctypes.data + s0.astype(np.int64) * nb,
                        a1.ctypes.data + s1.astype(np.int64) * nb).astype(np.int64)

    def _numa_partial(self, layers, mats, gnode, other, xbytes):
        """One full copy of the experts on the node of the GPU, and a copy of
        some experts of each layer on the other node (RQ8_EXPERTS_PLAN.md,
        layout 1): NP_GEMMA_GPU_NODE1_GB of them ("auto": 70% of the memory
        of that node less the tensors staged there and less
        NP_GEMMA_GPU_NODE1_KEEP_GB, default 4: with 397 experts a layer the
        server left 0.6-0.8 GB of node 1 free; 0: none, layout 3). The
        CPU part of a step gives the threads of that node the experts it has
        (kq_moe_small_body: slot1, the slot of each expert in the compact
        copy, or -1). The experts: the most used of
        NP_GEMMA_EXPERT_COUNTS (an .npy of (layers + 1, experts) counts), or
        else the experts after the first hot ones."""
        from . import numa
        self.numa_slot1 = {}
        opt = os.environ.get("NP_GEMMA_GPU_NODE1_GB", "auto")
        g = self.model.g
        if opt == "auto":
            # the tensors staged on that node: all but the experts
            on_other = 0
            if getattr(g, "stage_node", None) == other:
                staged = sum(a.nbytes for _lo, _hi, a in getattr(g, "_stage", ()))
                # the experts of an overlay file are not in the staged copy
                on_other = staged if getattr(g, "experts_mapped", False) else max(0, staged - xbytes)
            keep = float(os.environ.get("NP_GEMMA_GPU_NODE1_KEEP_GB", "4")) * 1e9
            budget = 0.7 * numa.node_bytes(other) - on_other - keep
        else:
            budget = float(opt) * 1e9
        per = xbytes / (len(layers) * self.E)              # an expert: gate, up, down
        n1 = int(min(self.E, max(0, budget) // (len(layers) * per)))
        quiet = os.environ.get("NP_GEMMA_QUIET") == "1"
        if n1 < 8:
            if not quiet:
                print("the experts on node %d only: %.1f GB, no room for a copy on node %d"
                      % (gnode, xbytes / 1e9, other), file=sys.stderr)
            return
        counts = None
        path = os.environ.get("NP_GEMMA_EXPERT_COUNTS")
        if path:
            counts = np.load(path)
        t0 = time.time()
        total = 0
        for i in layers:
            if counts is not None and i < counts.shape[0]:
                chosen = np.sort(np.argsort(-counts[i], kind="stable")[:n1])
            else:
                chosen = (np.arange(n1) + self.n_slots) % self.E
            slot1 = np.full(self.E, -1, np.int32)
            slot1[chosen] = np.arange(len(chosen), dtype=np.int32)
            self.numa_slot1[i] = slot1
            # the runs of adjacent chosen experts: one copy each
            cut = np.flatnonzero(np.diff(chosen) != 1) + 1
            runs = [(int(r[0]), len(r)) for r in np.split(chosen, cut)]
            for name in ("gate_proj", "up_proj", "down_proj"):
                m = self.model.M("layers.%d.mlp.switch_mlp.%s" % (i, name))
                src = m.data.reshape(-1).view(np.uint8)
                nb = src.nbytes // self.E
                a1 = numa.empty_on(len(chosen) * nb, np.uint8, other)
                o = 0
                for e0, n in runs:
                    cops.kq_memcpy_par(a1[o:o + n * nb], src[e0 * nb:(e0 + n) * nb])
                    o += n * nb
                total += len(chosen) * nb
                self.numa_mats1.setdefault(i, {})[name] = a1
        if not quiet:
            dt = time.time() - t0
            print("the experts on node %d, and %d of each layer on node %d (%s): %.1f GB in %.1f s"
                  % (gnode, n1, other, "the most used of %s" % path if counts is not None
                     else "after the first hot ones", total / 1e9, dt), file=sys.stderr)

    def _calib_nodes(self):
        """The share of the cold experts of a decode step for each NUMA node
        (the threads of the CPU part are bound spread: half on each node).
        The nodes can stream at different rates (6 memory channels on node 0
        of the 2-socket Xeon, 4 on node 1), and a step waits for the slower
        half. NP_GEMMA_CPU_NODE_SHARE: 0.5 (the default; "auto" gave 0.62 to
        0.64 and 50.3 tok/s against 51.8 with 0.5 on Qwen3.8 q8); "auto" measures the compute time of each half on random experts of a few
        layers with an even split and gives the slower half less (share0 =
        t1 / (t0 + t1), even within 5 percent); a number sets it. On the Xeon
        with 6 channels on node 0 and 4 on node 1 the halves were within the
        noise: the CPU part of a step is not bound by the channels."""
        from . import numa
        opt = os.environ.get("NP_GEMMA_CPU_NODE_SHARE", "0.5")
        if opt == "auto" and getattr(self, "numa_split", False):
            opt = "0.5"     # (kq_calib_nodes takes no slot0)
        if opt != "auto":
            cops._lib.kq_set_node0_share(float(opt))
            return
        if not numa.enabled() or len(set(numa.node_of_cpu().values())) != 2:
            return
        nth = int(os.environ.get("NP_GEMMA_GPU_CPU_THREADS", "0") or 0) or \
            int(os.environ.get("OMP_NUM_THREADS", "0") or 0) // 2
        if nth < 2 or nth % 2:
            return
        layers = [i for i in range(self.L) if "blk.%d.ffn_gate_exps.weight" % i in self.model.g.tensors]
        layers = layers[::max(1, len(layers) // 8)][:8]
        mats = np.stack([cops.kq_moe_mats(*(self.model.M("layers.%d.mlp.switch_mlp.%s" % (i, n)).c()
                                            for n in ("gate_proj", "up_proj", "down_proj")), None)
                         for i in layers])
        m1 = getattr(self, "numa_mats1", {})
        mats1 = None
        if m1 and all(i in m1 for i in layers) and not getattr(self, "numa_slot1", None):
            mats1 = np.stack([cops.kq_moe_mats(*((m1[i][n], self.model.M(
                "layers.%d.mlp.switch_mlp.%s" % (i, n)).type) for n in ("gate_proj", "up_proj", "down_proj")),
                None) for i in layers])
        cfg = self.cfg
        k, E, hidden, inner = cfg.top_k, cfg.num_experts, cfg.hidden_size, cfg.moe_inter
        ncold = max(1, k // 2)
        t0 = time.time()
        cops._lib.kq_set_node0_share(0.5)
        cops.kq_calib_nodes(mats, mats1, E, hidden, inner, ncold, 20, nth)       # warm
        # one measure of an even split (400 calls): a half that takes longer
        # gets less; within 5 percent, an even split (two rounds of 200 calls
        # moved the share between 0.45 and 0.55 with no gain in the decode)
        b = cops.kq_calib_nodes(mats, mats1, E, hidden, inner, ncold, 400, nth)
        if b[0] <= 0 or b[1] <= 0:
            return
        share = b[1] / (b[0] + b[1])
        if abs(share - 0.5) < 0.025:
            share = 0.5
        cops._lib.kq_set_node0_share(share)
        self.node0_share = float(cops._lib.kq_get_node0_share())
        if os.environ.get("NP_GEMMA_QUIET") != "1":
            print("the share of node 0 in the cold experts of a step: %.3f (%d threads; the last "
                  "measure %.0f / %.0f us a thread; %.1f s)" % (self.node0_share, nth, 1e6 * b[0],
                                                               1e6 * b[1], time.time() - t0),
                  file=sys.stderr)

    def _register_experts(self):
        """NP_GEMMA_GPU_REGISTER (1 by default): when the experts are in memory
        of the process (staged from a DAX mount, or a copy on each node),
        page-lock the copy that the copy workers read (cudaHostRegister):
        the copies of HotCache and of the mixed groups of a prompt are then
        DMA at the rate of the bus, not a memcpy into the pinned buffers of
        a worker and then DMA. The map of a file cannot be registered."""
        if os.environ.get("NP_GEMMA_GPU_REGISTER", "1") == "0":
            return
        g = self.model.g
        if not (getattr(g, "staged", False) or getattr(self, "numa_mats1", None)):
            return
        layers = [i for i in range(self.L + 1)
                  if "blk.%d.ffn_gate_exps.weight" % i in g.tensors]
        t0 = time.time()
        total = 0
        self._registered = []
        for i in layers:
            for name in ("gate_proj", "up_proj", "down_proj"):
                a = self.model.M("layers.%d.mlp.switch_mlp.%s" % (i, name)).data
                if not lib().gg_host_register_rw(a.ctypes.data, a.nbytes):
                    print("cudaHostRegister: %s" % lib().gg_last_error().decode(), file=sys.stderr)
                    return
                self._registered.append(a)
                total += a.nbytes
                a1 = self.numa_mats1.get(i, {}).get(name) if getattr(self, "numa_split", False) else None
                if a1 is not None:
                    # a split: the copies read the experts of node 1 there
                    if not lib().gg_host_register_rw(a1.ctypes.data, a1.nbytes):
                        print("cudaHostRegister: %s" % lib().gg_last_error().decode(), file=sys.stderr)
                        return
                    self._registered.append(a1)
                    total += a1.nbytes
        if os.environ.get("NP_GEMMA_QUIET") != "1":
            print("the experts page-locked for the copies to the GPU: %.1f GB in %.1f s" % (
                total / 1e9, time.time() - t0), file=sys.stderr)

    def _dram_experts(self):
        """NP_GEMMA_GPU_EXPERT_COPY: "auto" (the default) copies the experts
        into memory of the process when the model file is on a DAX mount
        (Optane in App Direct mode: the map of the file reads the module
        itself, at about 13 GB/s, not DRAM, and the CPU part of a step reads
        its cold experts at about 100 GB/s); "1" always, "0" never. The rest
        of the file (the PLE table, the embeddings) stays on the mount: a step
        reads a few rows of it. NP_GEMMA_GPU_EXPERT_NODE: "interleave" (the
        default: the pages over the NUMA nodes, as the CPU team is spread) or
        the number of a node."""
        from . import numa
        opt = os.environ.get("NP_GEMMA_GPU_EXPERT_COPY", "auto")
        if opt == "0" or getattr(self, "zc_dev", None) or getattr(self, "numa_mats1", None):
            return
        path = getattr(self.model.g, "path", None)
        if opt == "auto" and (getattr(self.model.g, "staged", False) or not (path and _on_dax(path))):
            return      # the GGUF reader has the experts in memory already (gguf._stage_dax)
        layers = [i for i in range(self.L + 1)
                  if "blk.%d.ffn_gate_exps.weight" % i in self.model.g.tensors]
        mats = [self.model.M("layers.%d.mlp.switch_mlp.%s" % (i, name))
                for i in layers for name in ("gate_proj", "up_proj", "down_proj")]
        where = os.environ.get("NP_GEMMA_GPU_EXPERT_NODE", "interleave")
        t0 = time.time()
        total = 0
        for m in mats:
            n = m.data.nbytes
            a = numa.empty_interleaved(n) if where == "interleave" else \
                numa.empty_on(n, np.uint8, int(where))
            cops.kq_memcpy_par(a, m.data.reshape(-1).view(np.uint8))
            m.data = a
            total += n
        if os.environ.get("NP_GEMMA_QUIET") != "1":
            dt = time.time() - t0
            print("the experts in memory (%s, from %s): %.1f GB in %.1f s, %.1f GB/s" % (
                where, path, total / 1e9, dt, total / 1e9 / max(dt, 1e-9)), file=sys.stderr)

    def _gpu_node(self):
        """The NUMA node of the GPU (local_cpulist of its PCI device), or 0."""
        from . import numa
        try:
            import ctypes
            bus = ctypes.create_string_buffer(32)
            lib().gg_pci_bus_id(bus, 32)
            with open("/sys/bus/pci/devices/%s/numa_node" % bus.value.decode().lower()) as f:
                n = int(f.read())
            return n if n >= 0 else 0
        except Exception:
            return 0

    def _store(self, i, init):
        """The slots of the experts of layer i: the slot table, the parts for
        HotCache, and the gate, up, and down blocks of a store of its own
        with the experts init when they do not fit the blocks of the pool;
        else a stub, and the experts init go to the pool (_pool_hot)."""
        E = self.cfg.num_experts
        p = "layers.%d.mlp.switch_mlp." % i
        mats = [self.model.M(p + n) for n in ("gate_proj", "up_proj", "down_proj")]
        nbs = [self._expert_nb(m) for m in mats]
        pooled = POOL_HOT and all(nb <= sp for nb, sp in zip(nbs, self.pool.stride))
        st = {"mats": mats, "parts": [], "pooled": pooled, "init": np.asarray(init)}
        # parts: (the host address of each expert (_expert_srcs), its bytes,
        # the device copy); srctab: the three of them for GP_MOE_PLAN (desc[24])
        for key, name, m, nb in zip(("gate", "up", "down"), ("gate_proj", "up_proj", "down_proj"),
                                    mats, nbs):
            srcs = self._expert_srcs(i, name, m, nb)
            if pooled:
                st[key] = np.zeros(16, np.uint8)
            else:
                st[key] = np.concatenate([_host_bytes(int(srcs[x]), nb) for x in init]) \
                    if len(init) else np.zeros(0, np.uint8)
            st["parts"].append((srcs, nb, st[key]))
        st["srctab"] = np.ascontiguousarray(np.stack([pt[0] for pt in st["parts"]]))
        slots = np.full(E, -1, dtype=np.int32)
        if not pooled:
            slots[init] = np.arange(len(init), dtype=np.int32)
        st["slots"] = slots
        return st

    def _pool_hot(self):
        """The first hot experts of the pooled layers: segments of the pool
        ("hot", the budget of the hot experts), a block for each expert of
        init, its parts copied there, and its slot WARM + block."""
        rows = [(i, st) for i, st in sorted(self.stores.items()) if st["pooled"]]
        need = sum(len(st["init"]) for _i, st in rows)
        self._pool_init = []            # (store layer, expert, block) for HotCache
        if not need:
            return
        pool = self.pool
        pool.grow(-(-need // pool.K) * pool.seg_bytes, "hot")
        free = list(np.flatnonzero(pool.owner == -1))
        for i, st in rows:
            for x in st["init"]:
                if not free:
                    break
                b = int(free.pop(0))
                for part, (srcs, nb, _dst) in enumerate(st["parts"]):
                    _check(lib().gg_h2d(pool.addr(part, b), int(srcs[int(x)]), nb))
                st["slots"][x] = HotCache.WARM + b
                pool.owner[b] = -4          # held; HotCache sets the owner
                self._pool_init.append((i, int(x), b))

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
        zdev = getattr(self, "zc_dev", {}).get(i)
        nzc = (self.zc if t == 1 else self.zc_mt) if zdev else 0
        zc = self._buf("zc", lambda: np.zeros(t * k + t, dtype=np.int32)) if nzc > 0 else None
        if t == 1:
            cold = np.zeros(2 * k + 1, dtype=np.int32)
            cold_val = np.zeros(k, dtype=np.float32)
            if zc is not None:
                prog.emit(P.HOT_SPLIT, idx, val, st["slots"], cold, cold_val, k, 1, zc, nzc)
            else:
                prog.emit(P.HOT_SPLIT, idx, val, st["slots"], cold, cold_val, k, 1)
            vp, ip = pinned((k,)), pinned((2 * k + 1,), np.int32)
            st["ip"] = ip       # the selection of the step, for HotCache
        else:
            cold = self._buf("cold", lambda: np.zeros((t, k), dtype=np.int32))
            cold_val = self._buf("cold_val", lambda: z(t, k))
            if zc is not None:
                prog.emit(P.HOT_SPLIT_MT, idx, val, st["slots"], cold, cold_val, t * k,
                          prog.slot("nreal"), k, None, zc, nzc)
            else:
                prog.emit(P.HOT_SPLIT_MT, idx, val, st["slots"], cold, cold_val, t * k,
                          prog.slot("nreal"), k)
            vp = self._buf("vp", lambda: pinned((t, k)))
            ip = self._buf("ip", lambda: pinned((t, k), np.int32))
        if FLAGS:
            # The flag form (csrc/gpu.cu, GP_SIGNAL): no boundary record, so a
            # step is one graph. The flags of each layer are its own.
            fl = pinned((2,), np.int64)
            prog.keep.append(fl)
            prog.emit(P.SIGNAL, fl[0:1], prog.slot("seq"), h, hp, hp.nbytes, cold_val, vp,
                      vp.nbytes, cold, ip, ip.nbytes)
        else:
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
                  self.dense(s + "down_proj")[0], self.dense(s + "gate_proj")[1], slog,
                  *((zc, zdev["table"]) if zc is not None else ()))
        # The CPU part: the cold experts, on the pinned copies.
        cc = P.Program()
        # A step or a verify group (t <= 4): KQ_MOE takes the float rows (act
        # bit 3, kq_moe_small_body: the int8 rows, the sort, and the act with
        # few barriers); NP_GEMMA_CPU_MOE_SMALL=0 keeps KQ_QUANT and KQ_MOE.
        small = t <= 4 and os.environ.get("NP_GEMMA_CPU_MOE_SMALL", "1") != "0"
        xq = self._buf("xq", lambda: np.zeros((t, hidden), np.int8))
        xs, xm = self._buf("xs", lambda: z(t, hidden // 32)), self._buf("xm", lambda: z(t, hidden // 16))
        if not small:
            cc.emit(P.KQ_QUANT, hp, t, hidden, xq, xs, xm)
        mats = cops.kq_moe_mats(*(m.c() for m in st["mats"]), None)
        # pinned in the flag form: GP_AWAIT reads it through the map
        host_out = self._buf("host_out", lambda: pinned((t, hidden)) if FLAGS else z(t, hidden))
        # the copy of the experts on the other node (_numa_copy_experts): the
        # threads there read it
        m1 = getattr(self, "numa_mats1", {}).get(i)
        slot1 = getattr(self, "numa_slot1", {}).get(i)
        mats1 = None
        if m1:
            mats1 = cops.kq_moe_mats(*((m1[n], m.type) for n, m in
                                       zip(("gate_proj", "up_proj", "down_proj"), st["mats"])), None)
            cc.keep.append(mats1)
        cc.emit(P.KQ_MOE, xq, xs, xm, ip, vp, t, k, E, mats, None, hidden, inner,
                self._buf("cpu_scratch", lambda: cops.kq_moe_scratch(t, k, E, hidden, inner)),
                host_out, ip[k:] if t == 1 else None, (8 if small else 0) | (32 if MOE_X16 else 0),
                hp, mats1, slot1,
                getattr(self, "numa_slot0", {}).get(i))
        cc.keep.append(mats)
        cpu = cc.finish()
        self.cpu_progs.append(cpu)
        part = self._buf("part", lambda: z(t, hidden))
        if FLAGS:
            # The runner runs the CPU program when the GPU signals; the GPU
            # waits for it, unless no expert of the step is cold (cold[k]).
            seq = prog.slot("seq")
            prog.emit(P.CPU_TASK, cpu.buf, fl[0:1], fl[1:2], seq)
            prog.emit(P.AWAIT, fl[1:2], seq, host_out, part, part.nbytes,
                      cold[k:k + 1] if t == 1 else 0)
        else:
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
        # the cold experts of the layer with the fewest hot ones (the hot
        # experts of a layer change: _fill_tables makes them again if more)
        cold = self.E - min(int((st["slots"] >= 0).sum()) for st in self.stores.values())
        self.stage_cold = cold
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
        cold = self.E - min(int((self.stores[i]["slots"] >= 0).sum()) for i in range(self.L))
        if cold > self.stage_cold:
            for bufs in self.stage:
                for b in bufs:
                    b.free()
            self.stage_cold = cold
            self.stage = [[Buffer(max(1, cold * nb)) for nb in self.per] for _ in range(2)]
        for i in range(self.L):
            st = self.stores[i]
            tabs, ranges = self.tables[i][:3], self.tables[i][3]
            stage = self.stage[i % 2]
            slots = st["slots"]
            rows = []
            for part, (tab, (srcs, nb, dst), buf) in enumerate(zip(tabs, st["parts"], stage)):
                dbase = mir.buffer_of(dst).ptr
                rank, run = 0, None
                for x in range(self.E):
                    if slots[x] >= 0:
                        tab[x] = self.hot_cache._slot_addr(part, dbase, nb, int(slots[x])) \
                            if self.hot_cache is not None else dbase + int(slots[x]) * nb
                        continue
                    tab[x] = buf.ptr + rank * nb
                    a = int(srcs[x])
                    if run is not None and run[3] == x - 1 and run[0] + run[2] == a:
                        run[2] += nb
                        run[3] = x
                    else:
                        run = [a, int(tab[x]), nb, x]
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
            e = self.groups[key] = (prog, self._gpu_program(prog, ("groups", key)))
        self._used(key)
        return e

    # ---- the memory of the programs ----

    def _gpu_program(self, prog, entry):
        """The GPUProgram of prog (entry: (the name of its dict, key));
        gpumm.ProgramLRU._build_program."""
        return self._build_program(prog, entry)

    def _alloc(self, nbytes, keep=None):
        """A device Buffer of nbytes (GpuMem frees memory for it: the
        free-memory segments of the pool, then the programs used least
        recently). Not between _mix_group and the run of its group (the
        run needs the blocks of the pool it reserved)."""
        return Buffer(nbytes, keep=keep)

    # ---- the mixed groups of a prompt ----

    def _moe_mix(self, prog, i, h, idx, val, slog, out):
        """The experts of layer i of a mixed group (see the module text)."""
        cfg = self.cfg
        st = self.stores[i]
        k, E, hidden, inner = cfg.top_k, cfg.num_experts, cfg.hidden_size, cfg.moe_inter
        t = h.shape[0]
        z = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        hp = self._buf("hp", lambda: pinned((t, hidden)))
        vp = self._buf("vp", lambda: pinned((t, k)))
        ip = self._buf("ip", lambda: pinned((t, k), np.int32))
        ev = self.n_events
        self.n_events += 1
        prog.emit(P.TO_HOST, h, hp, hp.nbytes, val, vp, vp.nbytes, idx, ip, ip.nbytes, ev)
        # The plan, on the host.
        tab_h = self._buf("tab_h", lambda: pinned((3 * E,), np.int64))
        gidx_h = self._buf("gidx_h", lambda: pinned((t, k), np.int32))
        cidx = self._buf("cidx", lambda: np.zeros((t, k), np.int32))
        ranges = self._buf("ranges", lambda: np.zeros((3 * E, 3), np.int64))
        desc = self.mix_desc.setdefault(i, np.zeros(25, np.int64))
        stats = self.mix_stats.setdefault(i, np.zeros(8, np.int64))
        pc = P.Program()
        ca, cb = mix_cpu_cost(st["mats"][0].type)
        # NP_GEMMA_GPU_MIX_SPLIT (1): the hot experts and the shared expert run
        # during the copies, the copied experts after them (two records)
        split = os.environ.get("NP_GEMMA_GPU_MIX_SPLIT", "1") != "0"
        gidx2_h = self._buf("gidx2_h", lambda: pinned((t, k), np.int32)) if split else None
        # prefetch (see mix): the experts of the layer copied before its plan
        pre_on = PREFETCH and split
        gidx3_h = None
        if pre_on:
            pr = self._pre(i)
            gidx3_h = self._buf("gidx3_h", lambda: pinned((t, k), np.int32))
            # the gidx3 of this program: desc (one for all the mixed programs
            # of the layer) gets it at each run (_pre_desc)
            self.__dict__.setdefault("mix_gidx3", {})[t] = gidx3_h
            desc[19], desc[21], desc[22] = pr["slot"].ctypes.data, gidx3_h.ctypes.data, \
                pr["cnt"].ctypes.data
        pc.emit(P.MOE_PLAN, ip, self._nreal_h, t, k, E, st["slots"], desc, ca, cb, MIX_GPU,
                tab_h, gidx_h, cidx, ranges, stats, gidx2_h)
        plan = pc.finish()
        self.cpu_progs.append(plan)
        prog.emit(P.CPU_START, plan.buf, ev)
        prog.emit(P.CPU_WAIT)
        tab_d = self._buf("tab_d", lambda: np.zeros(3 * E, np.int64))
        gidx_d = self._buf("gidx_d", lambda: np.zeros((t, k), np.int32))
        prog.emit(P.TO_DEV, tab_h, tab_d, tab_d.nbytes)
        prog.emit(P.TO_DEV, gidx_h, gidx_d, gidx_d.nbytes)
        if split:
            gidx2_d = self._buf("gidx2_d", lambda: np.zeros((t, k), np.int32))
            prog.emit(P.TO_DEV, gidx2_h, gidx2_d, gidx2_d.nbytes)
        if pre_on:
            gidx3_d = self._buf("gidx3_d", lambda: np.zeros((t, k), np.int32))
            prog.emit(P.TO_DEV, gidx3_h, gidx3_d, gidx3_d.nbytes)
        # the copies of the plan; with prefetch the layers alternate between
        # two sets of blocks (buffer i % 2), and the experts of the next layer
        # follow on the copy stream (the copies of this layer first: they are
        # on its path)
        b = i % 2 if pre_on else 0
        if pre_on and i == 0:
            prog.emit(P.FETCH, self._pre(0)["ranges"], 3 * E, PRE_F, 0)
        prog.emit(P.FETCH, ranges, 3 * E, i, b)
        if pre_on and i + 1 < self.L:
            prog.emit(P.FETCH, self._pre(i + 1)["ranges"], 3 * E, PRE_F + i + 1, (i + 1) % 2)
        # The experts of the CPU, on a helper thread, while the copies go on.
        cc = P.Program()
        xq = self._buf("xq", lambda: np.zeros((t, hidden), np.int8))
        xs, xm = self._buf("xs", lambda: z(t, hidden // 32)), self._buf("xm", lambda: z(t, hidden // 16))
        # only the rows with an expert of the CPU (cidx >= 0)
        cc.emit(P.KQ_QUANT, hp, t, hidden, xq, xs, xm, cidx, k)
        mats = cops.kq_moe_mats(*(m.c() for m in st["mats"]), None)
        # pinned: its copy to the GPU (TO_DEV, after the CPU experts) is on the
        # path of each layer; from pageable memory it ran at 1.8 GB/s, 12 ms of
        # each layer of a group of 2048 tokens
        host_out = self._buf("host_out_mix", lambda: pinned((t, hidden)))
        # the copy of the experts on the other node, as in the step: on the
        # 2-socket Xeon a layer of a real-text group took 38.9 ms with 40
        # threads that all read node 0, 31.8 with it (NP_GEMMA_GPU_MIX_NUMA=0:
        # none)
        m1 = getattr(self, "numa_mats1", {}).get(i) if MIX_NUMA else None
        mats1 = None
        if m1:
            mats1 = cops.kq_moe_mats(*((m1[n], m.type) for n, m in
                                       zip(("gate_proj", "up_proj", "down_proj"), st["mats"])), None)
            cc.keep.append(mats1)
        cc.emit(P.KQ_MOE, xq, xs, xm, cidx, vp, t, k, E, mats, None, hidden, inner,
                self._buf("cpu_scratch", lambda: cops.kq_moe_scratch(t, k, E, hidden, inner)),
                host_out, None, 32 if MOE_X16 else 0, hp, mats1,
                getattr(self, "numa_slot1", {}).get(i) if mats1 is not None else None,
                getattr(self, "numa_slot0", {}).get(i))
        cc.keep.append(mats)
        cc.threads = mix_threads()
        cpu = cc.finish()
        self.cpu_progs.append(cpu)
        prog.emit(P.CPU_START, cpu.buf, ev)
        # The experts of the GPU (hot and copied) and the shared expert.
        P_ = t * k + t
        s = "layers.%d.mlp.shared_expert." % i
        gm, _um, dm = st["mats"]
        tiles = -(-P_ // 64) + E + 1
        work = self._buf("work", lambda: np.zeros(8 + 2 * (E + 2) + 2 * P_ + 3 * tiles, np.int32))
        gpu_part = self._buf("gpu_part", lambda: z(t, hidden))
        act = self._buf("act", lambda: z(P_, 2 * inner))
        act2 = self._buf("act2", lambda: z(P_, inner))
        de = self._buf("de", lambda: z(P_, hidden))
        shared = [self.dense(s + "gate_proj")[0], self.dense(s + "up_proj")[0],
                  self.dense(s + "down_proj")[0]]
        if split:
            # the hot experts and the shared expert while the copies go on
            prog.emit(P.KQ_GROUP_MOE, h, val, gidx_d, t, k, E, hidden, inner, tab_d[:E],
                      tab_d[E:2 * E], tab_d[2 * E:], gm.type, dm.type, *shared,
                      self.dense(s + "gate_proj")[1], slog, work, act, act2, de, gpu_part,
                      prog.slot("nreal"))
            gpu_part2 = self._buf("gpu_part2", lambda: z(t, hidden))
            if pre_on:
                # the prefetched experts (their copies were before those of
                # the plan) while the copies of the plan go on
                prog.emit(P.FETCH_WAIT, PRE_F + i)
                prog.emit(P.KQ_GROUP_MOE, h, val, gidx3_d, t, k, E, hidden, inner, tab_d[:E],
                          tab_d[E:2 * E], tab_d[2 * E:], gm.type, dm.type, None, None, None,
                          self.dense(s + "gate_proj")[1], slog, work, act, act2, de, gpu_part2,
                          prog.slot("nreal"))
                prog.emit(P.ADD, gpu_part, gpu_part2, gpu_part, t * hidden)
            prog.emit(P.FETCH_WAIT, i)
            # the copied experts (sgate null: no shared expert)
            prog.emit(P.KQ_GROUP_MOE, h, val, gidx2_d, t, k, E, hidden, inner, tab_d[:E],
                      tab_d[E:2 * E], tab_d[2 * E:], gm.type, dm.type, None, None, None,
                      self.dense(s + "gate_proj")[1], slog, work, act, act2, de, gpu_part2,
                      prog.slot("nreal"))
            prog.emit(P.ADD, gpu_part, gpu_part2, gpu_part, t * hidden)
        else:
            prog.emit(P.FETCH_WAIT, i)
            prog.emit(P.KQ_GROUP_MOE, h, val, gidx_d, t, k, E, hidden, inner, tab_d[:E],
                      tab_d[E:2 * E], tab_d[2 * E:], gm.type, dm.type, *shared,
                      self.dense(s + "gate_proj")[1], slog, work, act, act2, de, gpu_part,
                      prog.slot("nreal"))
        prog.emit(P.FETCH_DONE, b)
        prog.emit(P.CPU_WAIT)
        part = self._buf("part", lambda: z(t, hidden))
        prog.emit(P.TO_DEV, host_out, part, part.nbytes)
        prog.emit(P.ADD, gpu_part, part, out, t * hidden)

    def free_ring(self):
        """Give back all the segments of the pool from the free memory
        (ExpertPool "free"): their warm experts go cold. For a user outside
        GpuMem (an image encoder that allocates with torch or cudaMalloc);
        the Buffers of GpuMem reclaim them as they need. The next mixed group
        (or decode step) takes the free memory again."""
        hc = self._warm_hc()
        if hc is not None:
            hc.pool_drop("free")
        elif self.pool is not None:
            self.pool.drop(self.pool.segments("free"))

    def _warm_hc(self):
        """The HotCache that uses the pool (the slot tables point into its
        blocks even when a test takes hot_cache away), or None."""
        hc = getattr(self, "warm_owner", None) or self.hot_cache
        return hc if hc is not None and hc.pool is not None else None

    def _ensure_pool(self):
        """The pool of HotCache (gpu.ExpertPool, blocks of the size of the
        experts of the layers), made once."""
        if self.pool is None:
            self.pool = ExpertPool(self.per)
        hc = self.hot_cache
        if hc is not None and hc.pool is None:
            hc.set_pool(self.pool)
            if WARM in ("0", "off"):
                hc.pool_rows[:] = False         # the copies of the prompts only
            self.warm_owner = hc
        # the lent room first: release_warm gives exactly it back
        lend = getattr(self, "warm_lend", 0)
        if lend > 0 and not self.pool.segments("lend"):
            self.pool.grow(lend, "lend")
        return self.pool

    def lend_warm(self, nbytes):
        """Room that the image encoder lends (serve_qwen4 --mmproj-gpu lend):
        nbytes of the GPU for the pool until release_warm. Its segments come
        at the next decode step (_warm_on)."""
        self.warm_lend = int(nbytes)

    def release_warm(self):
        """Give the lent room back (an image to encode): the warm experts of
        its segments go cold, and they are freed. The next decode step takes
        it again."""
        hc = self._warm_hc()
        if hc is not None:
            hc.pool_drop("lend")
        elif self.pool is not None:
            self.pool.drop(self.pool.segments("lend"))

    def _warm_desc(self, places=None, sets=None):
        """The table of the pool, and the blocks for the copies of a mixed
        group (places, int32; or sets: two, for the even and the odd layers,
        with prefetch), in the desc of each mixed group (GP_MOE_PLAN: a warm
        expert is hot there; the copies go to places)."""
        pt = self.pool.table.ctypes.data if self.pool is not None else 0
        for i, desc in self.mix_desc.items():
            st = self.stores[i]
            for p in range(3):
                srcs, nb, dst = st["parts"][p]
                desc[p] = self.g.mirror.buffer_of(dst).ptr
                desc[3 + p] = nb
                desc[6 + p] = int(srcs[0])
                desc[9 + p] = 0
            desc[16] = pt
            # the host address of each expert (a split), else 0: desc[6 + p] + e nb
            desc[24] = st["srctab"].ctypes.data if getattr(self, "numa_split", False) else 0
            if sets is not None:
                places = sets[i % 2]
            if places is not None:
                desc[12] = len(places)
                desc[17] = places.ctypes.data
            if sets is None:
                desc[19:23] = 0

    def _pre_desc(self, size):
        """The prefetch fields of the desc of each layer for a run of the
        mixed program of size rows: desc[19] and desc[22] (of the layer),
        desc[21] the gidx3 of THIS program (0s: a program with no prefetch).
        The desc of a layer is one for all its mixed programs, and the
        compile of a program set desc[21] to its own gidx3: a program of
        another size then ran with the gidx3 of the last compiled one. Its
        plan wrote there (past its end for a larger group), and its own
        gidx3, copied to the GPU, kept the experts of an older run: an expert
        on the CPU now has 0 in the table, and the prefetched products read
        its weights at 0 (cudaErrorIllegalAddress at 89033 tokens of a
        coding agent's session, after 47 requests)."""
        g3 = getattr(self, "mix_gidx3", {}).get(size)
        for i, desc in self.mix_desc.items():
            if g3 is None:
                desc[19] = desc[21] = desc[22] = 0
                continue
            pr = self._pre(i)
            desc[19], desc[21], desc[22] = pr["slot"].ctypes.data, g3.ctypes.data, pr["cnt"].ctypes.data

    def _pre(self, i):
        """The prefetch of layer i (persistent: all the mixed programs read
        it): the copies (3 E rows of host address, device address, bytes),
        the place of each expert in the set of the layer (or -1), and the
        tokens of each expert in its last group (written by the plan)."""
        pre = self.__dict__.setdefault("mix_pre", {})
        e = pre.get(i)
        if e is None:
            E = self.E
            e = pre[i] = dict(ranges=np.zeros((3 * E, 3), np.int64),
                              slot=np.full(E, -1, np.int32), cnt=np.zeros(E, np.int32))
        return e

    def _fill_prefetch(self, sets):
        """Before a mixed group: the experts each layer copies before its
        plan, in the first blocks of its set: of the experts its last group
        would copy with no prefetch (stats), PREFETCH_FRAC of them,
        the most tokens first, but not those hot now. In the order of the
        experts, so that runs of experts are one copy."""
        pool = self.pool
        for i in range(self.L):
            pr, st = self._pre(i), self.stores[i]
            stats = self.mix_stats.get(i)
            desc = self.mix_desc.get(i)
            if desc is None:
                continue
            pr["slot"][:] = -1
            pr["ranges"][:] = 0
            places = sets[i % 2]
            # the experts the last plan would copy with no prefetch (stats[7];
            # its copies alone before the first prefetch)
            n_last = int(stats[7] if stats is not None and stats[6] else
                         (stats[0] if stats is not None else 0))
            cnt = pr["cnt"]
            cand = np.flatnonzero((cnt > 0) & (st["slots"] < 0))
            cand = cand[np.argsort(-cnt[cand], kind="stable")]
            n = max(0, min(len(places) - 16, int(round(PREFETCH_FRAC * n_last)), len(cand)))
            chosen = np.sort(cand[:n])
            pr["slot"][chosen] = np.arange(n, dtype=np.int32)
            rows = []
            for p in range(3):
                srcs, nb, _dst = st["parts"][p]
                for place, x in enumerate(chosen):
                    a = int(srcs[int(x)])
                    d = pool.addr(p, int(places[place]))
                    if rows and rows[-1][0] + rows[-1][2] == a and rows[-1][1] + rows[-1][2] == d:
                        rows[-1][2] += nb
                    else:
                        rows.append([a, d, nb])
            if rows:
                pr["ranges"][:len(rows)] = rows
            desc[20] = n
            # the cost of a prefetched copy for the next prediction (moe.c
            # stats[7])
            desc[23] = max(1, int((desc[15] if desc[15] > 0 else MIX_GPU) * PREFETCH_COST))

    def _warm_on(self):
        """Before a decode step, a small group, or the MTP layer: the blocks
        that the last mixed group reserved go back to the pool, and the pool
        takes the lent room (lend_warm). Its segments of free memory (made
        for the prompts) stay until something needs the memory (free_ring)."""
        hc = self.hot_cache
        if hc is None:
            return
        pool = self._ensure_pool()
        hc.pool_unreserve()
        self._grow_shared()
        if WARM == "free" and not pool.segments("free"):
            # the free memory too, before a prompt makes segments of it
            # (without freeing programs for it)
            pool.grow(mem().room(), "free")

    def _grow_shared(self):
        """Segments of the pool in the memory that the cache lends (GpuMem
        shared: the rows past those in use and a margin)."""
        pool = self.pool
        if pool is None:
            return
        k = mem().shared_room(pool.seg_bytes)
        if k:
            pool.grow(k * pool.seg_bytes, "shared")

    def _pool_room(self, keep, evict=True):
        """Before a mixed group: the pool takes the free memory less the C
        reserve of GpuMem (GpuMem.room; the programs used least recently go first, but not keep, until it
        has MIX_RING_MIN blocks). Return the count of blocks for the copies:
        at most the experts of a layer."""
        self._ensure_head()
        pool = self._ensure_pool()
        per = pool.seg_bytes / pool.K

        def blocks():
            return pool.nbytes() // pool.seg_bytes * pool.K

        self._grow_shared()
        pool.grow(mem().room(), "free")
        while evict and blocks() < MIX_RING_MIN and self._evict(keep):
            pool.grow(mem().room(), "free")
        if blocks() < MIX_RING_MIN and evict:
            pool.grow(max(0, mem_info()[0] - MIX_KEEP_MIN), "free")
        cap = int(min(self.E, blocks()))
        if cap < 32:
            import warnings
            warnings.warn("the pool holds %d experts (%.2f GB each): the CPU takes almost all the "
                          "experts of a prompt (less hot_gb gives it room)" % (cap, per / 1e9))
        return cap

    def _mix_group(self, t):
        """The program of a mixed group of t rows, and the room of the pool
        for its copies (made after the programs and the head)."""
        e = self.mix_progs.get(t)
        if e is None:
            # A group of another size: the head, then its program, then the
            # pool takes the memory that is left (the plans read the table of
            # the pool and the blocks from desc, not from the programs). The
            # head after the program could take its memory by closing it:
            # with the NVFP4 experts (more of them in the pool) the head of
            # _pool_room closed the program of 4096 rows, and the run read
            # its freed arrays.
            self.free_ring()
            self._ensure_head()
            self._pool = {}
            prog = self._compile_mix(t)
            e = self.mix_progs[t] = (prog, self._gpu_program(prog, ("mix_progs", t)))
        # from here to the end of its run (mix) the program stays
        # (ProgramLRU._evict skips it)
        self._running = ("mix_progs", t)
        self._used(t, "mix_progs")
        self.mix_cap = self._pool_room(("mix_progs", t))
        return e

    def mix(self, tokens, pos, size, media=None):
        """Run tokens from position pos as one mixed group of size rows.
        Return the input of the head of each token."""
        t = len(tokens)
        # the parameters first: they can take memory (_alloc frees the buffer
        # of the copies, which _mix_group then makes for this run)
        kw = self._params(pos, size, t)
        # _mix_group marks the program running: until its run, the
        # allocations (the head, the parameters, the blocks of the copies)
        # do not close it (ProgramLRU._evict)
        try:
            prog, g = self._mix_group(size)
            return self._mix_run(prog, g, tokens, pos, size, t, media, kw)
        finally:
            self._running = None

    def _mix_run(self, prog, g, tokens, pos, size, t, media, kw):
        if self.hot_cache is not None:
            self.hot_cache.prepare(wait=True)      # the plan reads the slots
        self._nreal_h[0] = t
        self._inputs(prog, g, list(tokens), pos, media)
        g.bind(kw, self.cache_dev, scratch=self._scratch_names())
        # the blocks of the pool for the copies, after every allocation of the
        # run (an allocation can reclaim free-memory segments of the pool;
        # reserved blocks keep theirs); the warm experts of the lowest scores
        # give theirs if too few are free
        hc = self._warm_hc()
        if hc is not None:
            self.mix_places = hc.pool_reserve(self.mix_cap)
        else:
            free = np.flatnonzero(self.pool.owner == -1)[:self.mix_cap]
            self.pool.owner[free] = -2
            self.mix_places = free.astype(np.int32)
        if PREFETCH and os.environ.get("NP_GEMMA_GPU_MIX_SPLIT", "1") != "0":
            # two sets of blocks: a layer's prefetch fills its set while the
            # layer before uses the other
            half = len(self.mix_places) // 2
            self.mix_sets = (np.ascontiguousarray(self.mix_places[:half]),
                             np.ascontiguousarray(self.mix_places[half:2 * half]))
            self._fill_prefetch(self.mix_sets)
            self._warm_desc(sets=self.mix_sets)
            self._pre_desc(size)
        else:
            self._warm_desc(self.mix_places)       # (the desc of a new program too)
        if MIX_CAL:
            _check(lib().gg_mix_stats_on(1))
        t0 = time.perf_counter()
        first = not getattr(g, "ran", False)
        self.last_run = {"kind": "mix", "pos": int(pos), "rows": int(t), "size": int(size),
                         "first": first, "mix_cap": int(self.mix_cap),
                         "share_to": int(getattr(self.cache_dev, "share_to", 0))}
        try:
            g.run()
        except RuntimeError:
            try:
                free = "%.2f GB" % (mem_info()[0] / 1e9)
            except RuntimeError:
                free = "unknown (the context is lost)"
            print("[qwen_gpu] mixed group of %d rows at position %d failed (first run %s): free %s, "
                  "ring %d experts, programs %s" % (size, pos, first, free, self.mix_cap,
                                                    list(self._lru)), file=sys.stderr, flush=True)
            raise
        g.ran = True
        g.download("xn")
        if MIX_CAL:
            self.calibrate_mix(time.perf_counter() - t0, t)
            _check(lib().gg_mix_stats_on(0))
        self.last = g.mirror.buffer_of(prog.names["xn"]).ptr + (t - 1) * self.cfg.hidden_size * 4
        self.last_prog = (prog, g)
        self.rows = t
        self.cache.n = pos + t
        return prog.names["xn"][:t].copy()

    def calibrate_mix(self, wall, rows):
        """Tune the cost of a copy of GP_MOE_PLAN (gpu_c) to the mixed groups
        as they run. The costs that the copy thread and the CPU part measure
        do not give the best plan: the copies come from pageable memory, so
        the driver copies them with the CPU too, and the CPU part is slower
        when the plan copies more. The GPU waits for the copies, then for the
        CPU part, so the plan to find is the one with the least sum of the
        two waits (gg_mix_stats). On Qwen3.6 that sum was 727 ms for a group
        of 2048 with gpu_c 0.7 ms, 590 with 1.5 ms, and 765 with 2.2 ms.

        After each group gpu_c moves toward the balance of the two waits
        (see below). The value stays for the next prompts. The plans read it
        from desc (csrc/moe.c)."""
        buf = np.zeros(4096)
        n = lib().gg_mix_stats(buf.ctypes.data, buf.size)
        if n < 0:
            raise RuntimeError(lib().gg_last_error().decode())
        nf, nj = int(buf[2]), int(buf[3])
        waits = buf[4:4 + 3 * (nf + nj)].reshape(-1, 3)
        kinds = waits[:, 0] if len(waits) else np.zeros(0)
        copy_w, cpu_w = float(waits[kinds == 0, 1].sum()), float(waits[kinds == 1, 1].sum())
        w = (copy_w + cpu_w) / max(rows, 1)
        layers = sorted(self.mix_stats)
        tune = self.__dict__.setdefault("mix_tune", dict(f=1.0, last=None))
        # The balance of the two waits of the GPU: more copies (a lower
        # gpu_c) move experts from the CPU to the GPU, so the CPU waits go
        # down and the copy waits up; the least sum is near where they meet.
        # gpu_c moves by their ratio to the power 0.35 (at most 1.27x a group), with
        # a floor of 2% of the group so that small waits do not swing it.
        # The search of the sum by steps (1.25, then smaller) stopped near
        # its start: on the 2-socket Xeon with the clocks down, 950 us
        # (factor 1.36 of 700 us), 292 tok/s at pp8192 (the GPU waiting for
        # the CPU 56% of the group), where a fixed 250 us gave 416.
        c = 0.02 * max(wall, 1e-3)
        r = float(np.clip((copy_w + c) / (cpu_w + c), 0.5, 2.0))
        tune["f"] = float(np.clip(tune["f"] * r ** 0.35, 0.05, 8.0))
        tune["last"] = w
        for i, desc in self.mix_desc.items() if MIX_CAL_APPLY else ():
            # the costs of the CPU of the type of each layer (a file of mixed
            # types: RQ6_MIX_PLAN.md)
            a, b = mix_cpu_cost(self.stores[i]["mats"][0].type)
            desc[13:16] = [max(1, int(a)), max(1, int(b)), max(1, int(MIX_GPU * tune["f"]))]
        if MIX_LOG:
            st = np.array([self.mix_stats[i] for i in layers])
            print("mix: %d rows %.0f ms; copy waits %.0f ms, CPU waits %.0f ms; copied %.1f, "
                  "prefetched %.1f (%.1f used), CPU %.1f experts a layer; copies %.2f GB in %.0f ms "
                  "(%.1f GB/s); next gpu_c %.0f us (factor %.2f)" % (
                      rows, 1e3 * wall, 1e3 * copy_w, 1e3 * cpu_w, st[:, 0].mean(), st[:, 6].mean(),
                      st[:, 5].mean(), st[:, 2].mean(), buf[0] / 1e9, 1e3 * buf[1],
                      buf[0] / max(buf[1], 1e-9) / 1e9, MIX_GPU * tune["f"] / 1e3, tune["f"]))

    # ---- the hooks of the mixed groups (Qwen4GPU changes them) ----

    def _compile_mix(self, t):
        """The program of a mixed group of t rows."""
        self._pool = {}
        return _fuse(compile_qwen_step(_Emit(self, t, False, False, mix=True), t), t)

    def _ensure_head(self):
        """Make the head on the GPU (before the buffer of the copies takes
        the free memory)."""
        if self.head is None:
            m = self.model.M("lm_head")
            w = np.ascontiguousarray(m.data)
            self.head = Buffer(w.nbytes, "weights")
            self.head.upload(w)
            self.out = Buffer(4 * m.rows * MT, "weights")
            self.host_logits = pinned((MT, m.rows))

    def _scratch_names(self):
        return ("scores",)

    def _inputs(self, prog, g, ids, pos=0, media=None):
        """The input of a run of the tokens ids from pos: their embeddings,
        and the rows of the images of media."""
        t = len(ids)
        x = prog.names["x"]
        x[:t] = media_inputs(self.model.embed(ids), pos, media)
        x[t:] = 0.0
        g.upload("x")

    # ---- the cache ----

    def attach(self, cache):
        """Copy the cache to the GPU. From now on the GPU copy is the true
        one."""
        self.cache_dev.attach(cache)
        self.cache_dev.finish()
        self.cache = cache

    def detach(self, cache):
        self.cache_dev.to_host()
        self.cache_dev.release()

    def _params(self, pos, t, nreal):
        cache = self.cache
        assert pos + t <= cache.max_len, "the cache is too short"
        # the rows of the cache past those of this run (and a margin) are lent
        # to GpuMem (the pool of HotCache can take them)
        self.cache_dev.share_tail(pos + t, self.cache_dev.before)
        cos, sin = self.model.rope(rope_positions(cache, pos, t))
        kw = {"pos": pos, "nreal": nreal, "cos": np.ascontiguousarray(cos, np.float32),
              "sin": np.ascontiguousarray(sin, np.float32),
              "scores": np.empty(self.cfg.num_heads * (pos + t) + 64, np.float32)}
        kw.update(cache_params(self.model, cache))
        return kw

    # ---- the runs ----

    def step(self, token, pos):
        """Run one token at position pos. logits() gives its logits."""
        self._warm_on()
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

    def group(self, tokens, pos, size=None, verify=False, fetch=False, media=None):
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
        x[:t] = media_inputs(self.model.embed(tokens), pos, media)
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
        """An MTP verify group: see group(). It writes the rows of all the
        tokens (the keys and values, the keys of the indexer, and in
        serve_qwen4.py the MTP rows); commit() keeps n of them and moves
        cache.n, and the other rows stay until a later run writes them. See
        the rule of np_gemma/qwen.py _QwenRuns.verify: no reader may take a
        row at or after cache.n."""
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

    def _fetch_fits(self):
        """The two buffers of the copies of a large group exist, or fit in
        the free memory of the GPU with 0.5 GB left."""
        if self.stage is not None:
            return True
        need = 2 * (self.E - self.n_slots) * sum(self.per)
        return mem_info()[0] >= need + 0.5e9

    def _sizes(self, rem, room):
        """The next group of a prompt with rem tokens left and room rows
        left in the cache: (size, tokens, fetch). A large group runs only if
        the buffers of its copies fit."""
        if rem >= FETCH_MIN and self._fetch_fits():
            for s in sorted(FETCH_SIZES, reverse=True):
                if s <= room and (rem >= s or s == min(FETCH_SIZES)):
                    return s, min(rem, s), True
        for s in SPLIT_SIZES:
            if s >= rem and s <= room:
                return s, rem, False
        s = max(z for z in SPLIT_SIZES if z <= max(room, MT))
        return s, min(rem, s), False

    def prefill(self, ids, pos=0, media=None):
        """Run a prompt from position pos. Return the hidden state of its
        last token. media: the spans of its images (cache positions)."""
        ids = list(ids)
        c0 = 0
        h = None
        room = self.cache.max_len
        while c0 < len(ids):
            rem = len(ids) - c0
            # A mixed group (mix): MIX_SIZE rows, or the smallest of 256,
            # 512, ... that holds the rest and fits in the cache.
            size = next((sz for sz in (256, 512, 1024, 2048, 4096)
                         if sz <= MIX_SIZE and sz >= min(rem, MIX_SIZE)
                         and sz <= room - pos - c0), 0)
            if MIX_SIZE > 0 and rem >= MIX_MIN and size:
                n = min(rem, size)
                h = self.mix(ids[c0:c0 + n], pos + c0, size, media)
                c0 += n
                continue
            size, n, fetch = self._sizes(rem, room - pos - c0)
            h = self.group(ids[c0:c0 + n], pos + c0, size, fetch=fetch, media=media)
            c0 += n
        return h[-1:]

    def logits(self, rows=1):
        """The logits of the last rows of the last step or group (the head
        runs on the GPU; HotCache runs on the host meanwhile)."""
        cfg = self.cfg
        m = self.model.M("lm_head")
        self._ensure_head()
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
        for _p, g in list(self.groups.values()) + list(self.mix_progs.values()):
            g.close()
        self.mix_progs = {}
        hc = self._warm_hc()
        if hc is not None:
            hc.pool_drop()
        if self.pool is not None:
            self.pool.close()
        mem().remove_reclaimer(self._mm_name + "-pool")
        mem().remove_reclaimer(self._mm_name + "-hot")
        self._mm_close()
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


class _DevCache(DeviceCache):
    """The device copy of the arrays of a QwenCache: the keys and values of
    the full layers, the convolution and the state of the linear layers
    (gpumm.DeviceCache: locked; the rows past the positions in use are lent
    to GpuMem)."""

    def __init__(self):
        super().__init__("cache0")
        self.bufs = {}
        self.max_len = 0
        self.before = None      # QwenGPU: HotCache.prepare(wait=True), before lent memory changes

    def attach(self, cache):
        """The device buffers of the arrays of cache. The buffers of the last
        cache with the same sizes are used again (finish() frees the others),
        and a fresh cache (n == 0: its arrays are zeros) is zeroed on the
        GPU: a cache of 256K positions took about 2 s to upload, for each new
        conversation of the server, though a step reads only its first n
        positions. The memory the last cache lent stays lent when the
        buffers are used again (only the rows it keeps are written: no reader
        takes a row at or after cache.n); else it comes back first."""
        self._n = getattr(cache, "n", 1)
        if self._n > self.share_to or cache.max_len != self.max_len:
            self.unshare(self.before)
        self.max_len = cache.max_len
        self._pool = {}
        for _a, b in self.bufs.values():
            self._pool.setdefault(b.nbytes, []).append(b)
        self.bufs = {}
        self._fresh = self._n == 0
        arrays = [a for kv in cache.kv.values() for a in kv]
        arrays += list(cache.conv.values()) + list(cache.state.values())
        self._put(arrays)

    def _put(self, arrays):
        for a in arrays:
            pool = getattr(self, "_pool", {}).get(a.nbytes)
            b = pool.pop() if pool else self._new(a.nbytes)
            # position-major: a row for each position, or for each block of 4
            # (the keys of the blocks of the indexer)
            n0 = a.shape[0] if a.ndim else 0
            per = 1 if n0 == self.max_len else (4 if n0 == self.max_len // 4 + 1 else 0)
            old = self.rows.get(b.id)
            if old is not None and (old[1] != per or old[2] != (a.nbytes // n0 if per else 0)):
                self.unshare(self.before)       # another layout in a buffer that lent memory
            if per:
                self.rows[b.id] = (b, per, a.nbytes // n0, n0, None)
            else:
                self.rows.pop(b.id, None)
            # the rows the cache keeps (the rest may be lent: share_tail)
            nb = a.nbytes
            if per and self.share_to:
                nb = min(nb, -(-self.share_to // per) * (a.nbytes // n0))
            if getattr(self, "_fresh", False):
                _check(lib().gg_zero(b.ptr, nb))
            else:
                flat = np.ascontiguousarray(a).reshape(-1).view(np.uint8)
                _check(lib().gg_h2d(b.ptr, flat.ctypes.data, nb))
            self.bufs[id(a)] = (a, b)

    def finish(self):
        """Free the buffers of the last cache that attach did not use (the
        memory lent comes back first)."""
        left = [b for bs in getattr(self, "_pool", {}).values() for b in bs]
        if left:
            self.unshare(self.before)
        for b in left:
            self._free(b)
        self._pool = {}

    def device(self, a):
        e = self.bufs.get(id(a))
        return None if e is None else e[1].ptr

    def blocks(self):
        return [b for _a, b in self.bufs.values()]

    def to_host(self):
        for a, b in self.bufs.values():
            b.download(a)

    def release(self):
        self.unshare(self.before)
        self.finish()
        for _a, b in self.bufs.values():
            self._free(b)
        self.bufs = {}
