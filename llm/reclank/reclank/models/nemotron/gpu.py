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
from reclank.backends.cuda.runtime import GPUProgram, Mirror, int8_tiles
from reclank.formats import iq2
from reclank.ir import program as P
from reclank.models.nemotron.kq import (KQ_IQ2R, KQ_NV4, KQ_Q8_0, KQ_Q8_R, LM, Mat, NemotronKQCache,
                                        compile_nemotron_step, q8r_rows)
from reclank.models.qwen.gpu35 import _DevCache

# the programs of a prompt: a prompt runs in parts of exactly these sizes, the largest first (no
# padding: the head reads the last row of the part)
GROUPS = tuple(int(v) for v in os.environ.get("RECLANK_NM_GROUPS", "4,16,64,256,1024").split(","))
# Super with the pool: the larger prompt groups, their programs made for a long prompt with memory the
# pool gives back, freed after it (NemotronGPU.relax); RECLANK_NM_BIG_GROUPS ("" none)
BIG_GROUPS = tuple(int(v) for v in os.environ.get("RECLANK_NM_BIG_GROUPS", "2048,4096").split(",") if v.strip())
ATT_PARTS = 256                 # NM_ATTN2: the most (token, chunk) pairs of a run (backends/cuda nm.cu)
# Super: the hot experts of runs of at most this many rows on the GPU (NM_HOT_NV4: a warp a row of a pair);
# a larger group's pairs all on the CPU (RECLANK_NM_NV4_HOT_MAX)
NV4_HOT_MAX = int(os.environ.get("RECLANK_NM_NV4_HOT_MAX", "16"))


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
    of a layer 0.38 ms, 44 0.56), else all but one. The CPUs of the process (its taskset, as OpenMP's
    places have them), not of the calling thread: the runtime binds the main thread to one CPU."""
    try:
        from reclank.runtime.moe_parts import allowed
        n = len(allowed())
    except Exception:
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
# the runs of at most ZC_ROWS rows of those: the step's kernels (NM_HOT_MOE) on NM_GROUP_MOE's tables,
# a warp on rows of a pair (NM_GROUP_MOE's grid is every expert by every tile of 32 rows: at one token
# 7680-35712 blocks for ~6 experts, 0.1-0.4 ms a matrix on the RTX 3060, 15.9 ms of a 39 ms step).
# RECLANK_NM_ZC_ROWS=0: NM_GROUP_MOE for all.
ZC_ROWS = int(os.environ.get("RECLANK_NM_ZC_ROWS", "8"))
# a GPU of the float forms (no dp4a, no tensor cores: NemotronGPU.i8 False) splits its steps (the CPU's cold
# experts beside the GPU's hot ones) and plans its prompt groups (PLAN_F): its float products of every cold
# expert over the link lose to the CPU's (the Quadro M4000 at 8 GB/s, 12% hot, by zero copy against split:
# decode 7.8 against 11.3 tok/s, a prompt of 256 tokens 22 against 31, of 1024 32 against 33).
# RECLANK_NM_ZC_FLOAT=1: zero copy as the int8 GPUs
ZC_FLOAT = os.environ.get("RECLANK_NM_ZC_FLOAT", "0") == "1"
# Super: the time of a run of each size, s (the 3090 and the Xeon, real text, NM_PLAN's groups, 17% hot;
# 2048 and 4096, staged, from a prompt of 8K tokens against 1024's there, their programs' making included):
# NemotronGPU.parts picks the runs of a prompt by it (RECLANK_NM_GROUP_COST "size:s,...")
GROUP_COST = {1: 0.036, 4: 0.071, 16: 0.374, 64: 0.761, 256: 1.304, 1024: 3.104, 2048: 3.75, 4096: 4.67}
if os.environ.get("RECLANK_NM_GROUP_COST"):
    GROUP_COST = {int(a): float(b) for a, b in (kv.split(":") for kv in os.environ["RECLANK_NM_GROUP_COST"].split(","))}
# Super's prompt groups (NM_PLAN): the cost of a pair of a cold expert on the CPU and of a cold expert
# on the GPU (its copy over the link: 3.17 MB at ~10.6 GB/s), ns; RECLANK_NM_PLAN_CPU=0: every pair on
# the GPU
PLAN_CPU = float(os.environ.get("RECLANK_NM_PLAN_CPU", "12600"))
PLAN_GPU = float(os.environ.get("RECLANK_NM_PLAN_GPU", "300000"))
# a GPU of the float forms (no dp4a: Maxwell), Bolt's prompt groups of more than ZC_ROWS rows: NM_PLAN as Super's,
# the cold experts of few pairs the CPU's, the others with the hot ones NM_GROUP_MOE's float tiles on the mapped
# experts (the step's kernels took the hot pairs one by one); the CPU's cost of a pair, the GPU's of a cold expert
# (its copy hides behind the tiles' products) and of each pair it runs, ns. The Quadro M4000 with the Xeon W-2133
# (11 threads), against the split: a prompt of 256 tokens 44 -> 46 tok/s, 1024 54 -> 62, 4096 53 -> 60 (the
# GPU's pair at 20500 ns took most pairs, its tiles 8.6 s of a 1024 group: 54; at 45000 60, 90000 61; 50000
# and 200000 a cold expert the same). RECLANK_NM_PLAN_F=0: split as the steps
PLAN_F = os.environ.get("RECLANK_NM_PLAN_F", "1") != "0"
PLAN_F_CPU = float(os.environ.get("RECLANK_NM_PLAN_F_CPU", "43000"))
PLAN_F_GPU = float(os.environ.get("RECLANK_NM_PLAN_F_GPU", "0"))
PLAN_F_PAIR = float(os.environ.get("RECLANK_NM_PLAN_F_PAIR", "65000"))
# refresh_experts (between prompts: the engine's idle; backends/cuda/devices.py REFRESH, REFRESH_HALF):
# RECLANK_EXPERT_REFRESH=0 keeps the expert device's first set (the routing profile's); the routing
# before a call weighs half for each RECLANK_EXPERT_REFRESH_HALF tokens (128) of the new (three
# two-turn chats, the second turns: a half-life of a turn or more 37.1-37.5 tok/s, none 38.1; a few
# tokens of a short answer should not undo the turns before them)
# the slowest link to the host for it (GB/s, measured at start): a group of 1024 tokens pulls the cold
# experts once (~12 GB with a quarter hot), so even 0.8 GB/s (a P102-100 at x4 Gen1, ~64 tok/s) beats
# 4-6 cores (~30 tok/s); below this the CPU takes the cold experts of a prompt
ZC_GBPS = float(os.environ.get("RECLANK_NM_ZC_GBPS", "0.5"))


class _Q8RModel:
    """The model as NemotronGPU's programs read it on a GPU without dp4a (Q8R): each Q8_0 matrix of M as
    KQ_Q8_R rows, the weightkeeper's ("q8r.<name>": models/nemotron/keep.py) or made once (q8r_rows); the
    rest, and the attributes set on it, the model's."""

    def __init__(self, m):
        object.__setattr__(self, "_base", m)
        object.__setattr__(self, "_mats", {})

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        setattr(self._base, name, value)

    def M(self, name):
        m = self._mats.get(name)
        if m is None:
            m = self._base.M(name)
            if m is not None and m.type == KQ_Q8_0:
                kept = self._base._kept("q8r." + name)
                m = Mat(kept[1] if kept is not None else q8r_rows(m.data, m.rows, m.cols), KQ_Q8_R, m.rows,
                        m.cols, m.scale)
            self._mats[name] = m
        return m


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
# RECLANK_NM_I8=0 keeps the float products, as a GPU without dp4a or the tensor cores (Maxwell:
# NemotronGPU.i8)
I8 = os.environ.get("RECLANK_NM_I8", "1") != "0"
# a GPU without dp4a (Maxwell: KQ_LINEAR's float k_kq_linear, k_kq_gemm) takes the Q8_0 matrices as KQ_Q8_R
# rows, made at the load (_Q8RModel): a lane read 4 values of Q8_0's 34-byte blocks by two 2-byte loads (the
# Quadro M4000's head of 131072 x 4096 16.6 ms, 34 GB/s); RECLANK_NM_Q8R=0 keeps Q8_0
Q8R = os.environ.get("RECLANK_NM_Q8R", "1") != "0"
# x of the int8 products as int16 in two int8 planes (q = 256 hi + lo: two dp4a for each 4 values;
# int8 x left ~0.5% noise in each product); RECLANK_NM_X16=0: int8 x
X16 = os.environ.get("RECLANK_NM_X16", "1") != "0"
# the tiled int8 product of the prompt groups (NM_GEMM8); RECLANK_NM_GEMM8=0: the row kernels
GEMM8 = os.environ.get("RECLANK_NM_GEMM8", "1") != "0"
# a group of 2 to 8 rows (a verify group of the drafter's block) on the tensor cores (sm_80 up;
# RECLANK_NM_TC8=0: the rows kernels)
TC8 = os.environ.get("RECLANK_NM_TC8", "1") != "0" and os.environ.get("RECLANK_NM_TC", "1") != "0"
# the small records of a layer folded into others (the router's count and split: k_router_sig_r; the sum
# of the CPU's part in its AWAIT); RECLANK_NM_FUSE=0: records of their own
FUSE = os.environ.get("RECLANK_NM_FUSE", "1") != "0"
# the int8 x of the products made by the record that writes the float rows (NM_ADDNORM, NM_MAMBA's
# norm, RELU2, NM_HOT_MOE: their operands q, s, ql), once for all the products on them; RECLANK_NM_QFUSE=0:
# an NM_QX record before each product
QFUSE = os.environ.get("RECLANK_NM_QFUSE", "1") != "0"
# (the operand of the float rows a record writes, its operand q)
_QAT = {P.NM_ADDNORM: (4, 8), P.NM_MAMBA: (9, 20), P.RELU2: (1, 3), P.NM_HOT_MOE: (8, 19)}


# Super: the hot experts in a pool of segments on the GPU (NmPool: any expert of any layer in any slot; the
# programs read each expert's address from its layer's table, so the set changes with no new program); the
# set follows the text (the scores of the routers' choices) and gives segments to a large prompt group.
# RECLANK_NM_POOL=0: the fixed stacks of each layer. RECLANK_NM_SEG: the experts of a segment.
POOL = os.environ.get("RECLANK_NM_POOL", "1") != "0"
# the GPU memory left free beside a big group's program (its graphs, the allocator's rounding), GB
BIG_MARGIN_GB = float(os.environ.get("RECLANK_NM_BIG_MARGIN_GB", "0.4"))
# Super's prompt groups of at least STAGE_MIN rows with the pool: each MoE layer's cold experts copied to one
# of two buffers on the GPU while the layers before run (GP_FETCH: the copy worker's pull kernel from the
# registered stacks, ~12 GB/s, 111 ms a layer), NM_GROUP_MOE then reading them there (k_nm_gtile_nv4c), every
# pair on the GPU (the CPU's share and its tasks stalled the runner between the copies' boundaries: 358 tok/s
# against 516 on a group of 4096). A group of 1024 moves the same copies for a quarter of the tokens: there the
# mapped host memory and the CPU's share stay faster (258 against 227). RECLANK_NM_STAGE=0: the mapped memory
STAGE = os.environ.get("RECLANK_NM_STAGE", "1") != "0"
STAGE_MIN = int(os.environ.get("RECLANK_NM_STAGE_MIN", "2048"))
# the stage buffers in the tiles of the GPU (NV4T: k_nm_tile_pull permutes the KQ_NV4 rows as it copies them,
# every expert of the layer, the hot ones from their slots; k_nm_gtile_nv4t reads each lane's fragments in
# one load); RECLANK_NM_TILES=0: the rows as they are (k_nm_gtile_nv4c)
TILES = os.environ.get("RECLANK_NM_TILES", "1") != "0"
# what the pool gives back when a program finds no room (NemotronGPU._load), GB
ROOM_GB = float(os.environ.get("RECLANK_NM_ROOM_GB", "0.5"))
SEG_EXPERTS = int(os.environ.get("RECLANK_NM_SEG", "32"))
# the copies a token after a run (RECLANK_NM_HOT_BUDGET), at most HOT_CAP a run (RECLANK_NM_HOT_CAP); an
# expert comes in where its score beats the worst held one's by HOT_MARGIN; the score of an expert decays
# by HOT_DECAY a token (RECLANK_NM_HOT_DECAY); RECLANK_NM_HOT_FOLLOW=0: the first set stays
HOT_BUDGET = int(os.environ.get("RECLANK_NM_HOT_BUDGET", "4"))
HOT_CAP = int(os.environ.get("RECLANK_NM_HOT_CAP", "64"))
HOT_MARGIN = float(os.environ.get("RECLANK_NM_HOT_MARGIN", "0.5"))
HOT_DECAY = float(os.environ.get("RECLANK_NM_HOT_DECAY", "0.98"))
HOT_FOLLOW = os.environ.get("RECLANK_NM_HOT_FOLLOW", "1") != "0"
PLAN_EVERY = int(os.environ.get("RECLANK_NM_HOT_EVERY", "16"))
# the most copies after a prompt group (its choices: the conversation's experts; RECLANK_NM_HOT_CAP_GROUP)
HOT_CAP_GROUP = int(os.environ.get("RECLANK_NM_HOT_CAP_GROUP", "256"))


def _patch(prog, rec, k, v):
    """Operand k of the record rec of prog set to v (the operands before it that it lacks: none)."""
    args = rec[1]
    while len(args) <= k:
        args.append((P.T_NONE, 0))
    args[k] = prog._enc(v)


class NmPool:
    """The hot experts of NemotronGPU (Super) in segments of SEG_EXPERTS slots on the GPU. A slot holds the
    up and the down rows of one expert of any layer (every layer's experts have the same size). Each layer
    has its tables: slots (the slot of each expert, -1 cold: NM_HOT_NV4 and NM_PLAN read them) and pup /
    pdn, each expert's device address: its slot's when held, else its rows in host memory, which the GPU
    maps (the weightkeeper's stacks, registered: NM_GROUP_MOE reads them there; NM_HOT_NV4 reads the
    held ones). An expert comes in by a copy on the copy worker (rc_gpu_cache_copy, DMA from the
    registered stacks) while the runs go on, and is held from the first run after it landed; one leaves
    at once (the GPU is idle between runs). The scores: decay a token, plus the uses (observe); the first
    ones from the routing profile."""

    def __init__(self, g, counts):
        from reclank.backends.cuda.runtime import lib
        self.g = g
        m, cfg = g.m, g.cfg
        self.layers = sorted(cfg.moe)
        self.row = {i: r for r, i in enumerate(self.layers)}
        E = {cfg.moe[i][0] for i in self.layers}
        assert len(E) == 1, "NmPool: every layer the same count of experts"
        self.E = E.pop()
        up0, dn0 = m.experts(self.layers[0], "up"), m.experts(self.layers[0], "down")
        self.pu, self.pd = up0.data.nbytes // self.E, dn0.data.nbytes // self.E
        self.per = self.pu + self.pd
        self.seg_bytes = SEG_EXPERTS * self.per
        L = lib()
        tabs = list(self.layers) + (["mtp"] if cfg.mtp else [])
        self.host, self.slots, self.pup, self.pdn = {}, {}, {}, {}
        # (each layer's tables the rows of one array of each: one upload of each for all the changes)
        self._slots = np.full((len(tabs), self.E), -1, np.int32)
        self._pup = np.zeros((len(tabs), self.E), np.int64)
        self._pdn = np.zeros((len(tabs), self.E), np.int64)
        for q, i in enumerate(tabs):
            upm, dnm = m.experts(i, "up"), m.experts(i, "down")
            assert upm.data.nbytes == self.pu * self.E and dnm.data.nbytes == self.pd * self.E
            devs = []
            for a in (upm.data, dnm.data):
                d = g._zc_reg.get(id(a))
                if d is None:
                    d = L.rc_gpu_host_register_rw(a.ctypes.data, a.nbytes)
                    if not d:
                        raise RuntimeError("NmPool: the experts cannot be registered (%s)" % L.rc_gpu_last_error().decode())
                    g._zc_reg[id(a)] = d
                devs.append(int(d))
            self.host[i] = (upm.data.ctypes.data, dnm.data.ctypes.data, devs[0], devs[1])
            ar = np.arange(self.E, dtype=np.int64)
            self.pup[i], self.pdn[i], self.slots[i] = self._pup[q], self._pdn[q], self._slots[q]
            self.pup[i][:] = devs[0] + ar * self.pu
            self.pdn[i][:] = devs[1] + ar * self.pd
        self.segs = []                  # the Buffers of the segments
        # each slot: the row of its expert's layer (self.layers) and the expert, -1 free; inc: its copy on
        # its way
        self.own_r = np.zeros(0, np.int64)
        self.own_e = np.zeros(0, np.int64)
        self.inc = np.zeros(0, bool)
        # (no profile: the first experts of every layer, as many of each: a tiny score falling with the index)
        self.score = np.tile(1e-6 * (self.E - np.arange(self.E)) / self.E, (len(self.layers), 1))
        if counts is not None:
            # (the rate of each expert in the profile, as the score it would reach: rate / (1 - decay))
            for i in self.layers:
                c = np.asarray(counts.get(i, np.zeros(self.E)), np.float64)
                k = cfg.moe[i][2]
                tok = c.sum() / k if c.sum() > 0 else 1.0
                self.score[self.row[i]] = c / tok / (1.0 - HOT_DECAY)
        self.incoming = np.zeros(self.score.shape, bool)
        self.pending = []               # (job, ranges, [(layer, expert, slot)])
        self.copies = 0
        self.hits = self.pairs = 0
        self.owed = 0                   # the tokens since the last plan
        # the uses of each expert, counted on the GPU (GP_COUNT: every program adds to them), and their
        # values at the last observe
        self.cnt = np.zeros((len(self.layers), self.E), np.int32)
        self.cnt_last = np.zeros_like(self.cnt)
        self.changed = set()            # the layers whose tables are not on the GPU yet

    # ---- the slots

    def slot_addr(self, s):
        """The device addresses of slot s's up and down rows."""
        b = self.segs[s // SEG_EXPERTS].ptr
        j = s % SEG_EXPERTS
        return b + j * self.pu, b + SEG_EXPERTS * self.pu + j * self.pd

    def held(self):
        return int((self.own_r >= 0).sum())

    def nbytes(self):
        return len(self.segs) * self.seg_bytes

    def add_segments(self, n):
        """Up to n new segments (fewer when the memory runs out): their slots free. The count added."""
        from reclank.backends.cuda.gpumm import Buffer
        k = 0
        for _ in range(n):
            try:
                b = Buffer(self.seg_bytes, "program", pinned=True)
            except MemoryError:
                break
            self.segs.append(b)
            self.own_r = np.concatenate([self.own_r, np.full(SEG_EXPERTS, -1, np.int64)])
            self.own_e = np.concatenate([self.own_e, np.full(SEG_EXPERTS, -1, np.int64)])
            self.inc = np.concatenate([self.inc, np.zeros(SEG_EXPERTS, bool)])
            k += 1
        return k

    def _cold(self, i, e):
        _hu, _hd, du, dd = self.host[i]
        self.slots[i][e] = -1
        self.pup[i][e] = du + e * self.pu
        self.pdn[i][e] = dd + e * self.pd
        self.changed.add(i)

    def upload(self):
        """The tables to the GPU when a layer's changed (the programs' copies: the mirror's)."""
        if not self.changed:
            return
        mir = self.g.mirror
        for a, tab in ((self._slots, self.slots), (self._pup, self.pup), (self._pdn, self.pdn)):
            if mir.owner(a.ctypes.data) is not None:
                b = mir.buffer_of(a)
                for i in self.changed:
                    row = tab[i]
                    b.upload_at(row, row.ctypes.data - a.ctypes.data)
        self.changed = set()

    # ---- the copies

    def _copy(self, items):
        from reclank.backends.cuda.runtime import lib
        ranges = []
        for i, e, s in items:
            hu, hd, _du, _dd = self.host[i]
            au, ad = self.slot_addr(s)
            ranges += [hu + e * self.pu, au, self.pu, hd + e * self.pd, ad, self.pd]
            self.incoming[self.row[i], e] = True
            self.own_r[s], self.own_e[s], self.inc[s] = self.row[i], e, True
        ra = np.array(ranges, np.int64)
        job = lib().rc_gpu_cache_copy(ra.ctypes.data, len(ranges) // 3)
        if job < 0:
            raise RuntimeError(lib().rc_gpu_last_error().decode())
        self.pending.append((job, ra, items))
        self.copies += len(items)

    def land(self, wait=False):
        """The copies that are done: their experts held (their tables changed); wait: all of them."""
        from reclank.backends.cuda.runtime import lib
        still = []
        for job, ra, items in self.pending:
            r = lib().rc_gpu_cache_query(job, 1 if wait else 0)
            if r < 0:
                raise RuntimeError(lib().rc_gpu_last_error().decode())
            if r == 0:
                still.append((job, ra, items))
                continue
            for i, e, s in items:
                au, ad = self.slot_addr(s)
                self.slots[i][e] = s
                self.pup[i][e], self.pdn[i][e] = au, ad
                self.inc[s] = False
                self.incoming[self.row[i], e] = False
                self.changed.add(i)
        self.pending = still

    def observe(self):
        """The routers' choices since the last call (the counts on the GPU, GP_COUNT: their growth), into
        the scores (decayed by the tokens since then)."""
        mir = self.g.mirror
        if mir.owner(self.cnt.ctypes.data) is None:
            return
        mir.buffer_of(self.cnt).download(self.cnt)
        delta = (self.cnt - self.cnt_last).astype(np.float64)
        self.cnt_last[:] = self.cnt
        self.score *= HOT_DECAY ** self.owed
        self.score += delta
        # (the pairs the GPU held: hits / pairs, a measure)
        self.hits += int(delta[self._slots[:len(self.layers)] >= 0].sum())
        self.pairs += int(delta.sum())

    def plan(self, n=None, margin=HOT_MARGIN):
        """Up to n copies: the best experts not held into the free slots, then each where it beats the
        worst held one by margin (that one leaves: its tables cold at once). n None: the free slots
        only, all of them."""
        sc = self.score
        R, E = sc.shape
        held = np.zeros(R * E, bool)
        used = self.own_r >= 0
        held[self.own_r[used] * E + self.own_e[used]] = True
        cand = np.where(held | self.incoming.ravel() | (sc.ravel() <= 0), -np.inf, sc.ravel())
        free = np.nonzero(~used)[0]
        limit = len(cand) if n is None else n
        want = min(limit, int(np.isfinite(cand).sum()))
        if want <= 0:
            return 0
        top = np.argpartition(-cand, want - 1)[:want] if want < len(cand) else np.arange(len(cand))
        top = top[np.argsort(-cand[top], kind="stable")]
        items = []
        nf = min(len(free), want)
        for c in range(nf):
            r, e = divmod(int(top[c]), E)
            items.append((self.layers[r], e, int(free[c])))
        rest = top[nf:]
        if len(rest) and n is not None:
            # (swaps: the worst held ones that are not on their way, cheapest first)
            hs = np.nonzero(used & ~self.inc)[0]
            if len(hs):
                hsc = sc[self.own_r[hs], self.own_e[hs]]
                m = min(len(rest), len(hs))
                worst = np.argpartition(hsc, m - 1)[:m] if m < len(hs) else np.arange(len(hs))
                worst = worst[np.argsort(hsc[worst], kind="stable")]
                for c, v in zip(rest, worst):
                    if cand[c] <= hsc[v] + margin:
                        break
                    s = int(hs[v])
                    self._cold(self.layers[self.own_r[s]], int(self.own_e[s]))
                    self.own_r[s] = self.own_e[s] = -1
                    r, e = divmod(int(c), E)
                    items.append((self.layers[r], e, s))
        if items:
            self._copy(items)
        return len(items)

    def shrink(self, nbytes):
        """Segments back to the GPU for at least nbytes (the ones whose experts score least): their
        experts cold. The bytes freed."""
        self.land(wait=True)
        n = min(len(self.segs), -(-int(nbytes) // self.seg_bytes))
        if n <= 0:
            return 0
        used = self.own_r >= 0
        w = np.where(used, self.score[np.maximum(self.own_r, 0), np.maximum(self.own_e, 0)], 0.0)
        worth = w.reshape(-1, SEG_EXPERTS).sum(1)
        gone = set(np.argsort(worth, kind="stable")[:n].tolist())
        keep = [k for k in range(len(self.segs)) if k not in gone]
        for k in sorted(gone):
            for s in range(k * SEG_EXPERTS, (k + 1) * SEG_EXPERTS):
                if self.own_r[s] >= 0:
                    self._cold(self.layers[self.own_r[s]], int(self.own_e[s]))
            self.segs[k].free()
        # (the kept segments' experts stay where they are; their slots renumbered: segment keep[q] is q)
        idx = np.concatenate([np.arange(k * SEG_EXPERTS, (k + 1) * SEG_EXPERTS) for k in keep]) if keep else \
            np.zeros(0, np.int64)
        self.segs = [self.segs[k] for k in keep]
        self.own_r, self.own_e, self.inc = self.own_r[idx], self.own_e[idx], self.inc[idx]
        for s in np.nonzero(self.own_r >= 0)[0]:
            i = self.layers[self.own_r[s]]
            self.slots[i][self.own_e[s]] = s
            self.changed.add(i)
        self.upload()
        return n * self.seg_bytes


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
        if not self.dev.i8:
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
                args += [prog._enc(None)] * (at + 3 - len(args))
                args[at:at + 3] = [prog._enc(a) for a in q]
            else:
                prog.emit(P.NM_QX, src, t, m.cols, q[0], q[1], q[2])
            if new is not None and new[0] is src:
                self._done[m.cols] = src
        if gemm:
            prog.emit(P.NM_GEMM8, q[0], q[1], m.data, m.type, m.rows, m.cols, t, out, q[2])
        else:
            prog.emit(P.NM_IQ2_LIN, q[0], q[1], m.data, m.rows, m.cols, t, out, q[2])
        return True

    def count(self, prog, i, idx, k):
        """compile_nemotron_step, after layer i's ROUTER_SIG: its record kept (the split of _moe goes
        into it), and its choices added to the pool's counts on the GPU (NemotronGPU.after_run reads
        them): operands 9 and 13 of the record (k_router_sig_r: GP_COUNT's sums, no launch of its own)."""
        enc = prog._enc(idx)
        rec = prog.recs[-1] if prog.recs and prog.recs[-1][0] == P.ROUTER_SIG and prog.recs[-1][1][5] == enc else None
        self._router = rec if FUSE else None
        pool = self.dev.pool
        if pool is not None and HOT_FOLLOW and i in pool.row:
            if self._router is not None:
                _patch(prog, rec, 9, pool.cnt[pool.row[i]])
                _patch(prog, rec, 13, prog.slot("nreal"))
            else:
                prog.emit(P.COUNT, idx, pool.cnt[pool.row[i]], k, prog.slot("nreal"))

    def attention(self, prog, i, q, k, v, out, t):
        """The keys and values into the bfloat16 rows of the cache (NM_KV), then the attention in
        chunks of positions (NM_ATTN2: each block one KV head, its 16 query heads, a chunk; then
        the chunks merged)."""
        cfg = self.dev.cfg
        K, V, pos = prog.slot("K.%s" % i), prog.slot("V.%s" % i), prog.slot("pos")
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

    def _hot(self, prog, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t, Im, tab=False):
        """NM_HOT_MOE on the pairs of idx: the experts of slots (>= 0) in the stacks up, down; tab: up
        and down are NM_GROUP_MOE's tables (each expert's address: a slot or the mapped host memory),
        every pair of idx >= 0 runs."""
        S = self.scratch
        if self.dev.nv4:
            # (Super's KQ_NV4 experts: float x, docs/NEMOTRON_SUPER_PLAN.md S5; the pool's tables of
            # each expert's address)
            pool = self.dev.pool
            tabs = (pool.pup[self._layer], pool.pdn[self._layer]) if pool is not None else ()
            prog.emit(P.NM_HOT_NV4, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t, *tabs)
            return
        more = [None, None, None, 1] if tab else []
        if self.dev.i8:
            prog.emit(P.NM_HOT_MOE, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t,
                      S("qx", t * L, np.int8), S("sx", t * L // 32), S("qa", t * k * Im, np.int8),
                      S("sa", t * k * Im // 32), S("qxl", t * L, np.int8) if X16 else None,
                      S("qal", t * k * Im, np.int8) if X16 else None, *more)
        else:
            prog.emit(P.NM_HOT_MOE, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t,
                      *([None] * 6 + more if tab else []))

    def moe(self, prog, i, lat, idx, val, out, t, side=None):
        """The routed experts of layer i (out: their sums). side (or None): emits the records that do
        not need them (the shared expert); here after the CPU's task starts and before the wait for
        it, so the GPU runs them while the CPU runs the cold pairs."""
        self._side, self._layer = side, i
        self._moe(prog, i, lat, idx, val, out, t)
        self.side_now()
        routes = self.dev.routes
        if routes is not None and i in self.dev.cfg.moe:
            # (the choices into refresh_experts' counts: here, not after the router, whose record and the
            # GP_SIGNAL after it are one kernel, k_nm_route_signal)
            prog.emit(P.COUNT, idx, routes[i], self.dev.cfg.moe[i][2], prog.slot("nreal"), routes.shape[1])

    def _await_add(self, prog, fl, seq, rows, part, gout, out, n, cnt):
        """The wait for the CPU's task and its rows added to the GPU's (gout): out = gout + rows, in the
        AWAIT (operand 6: GP_ADD's sum, no launch of its own) or by an ADD after it."""
        if FUSE:
            prog.emit(P.AWAIT, fl[1:2], seq, rows, out, n * 4, cnt, gout)
        else:
            prog.emit(P.AWAIT, fl[1:2], seq, rows, part, n * 4, cnt)
            prog.emit(P.ADD, gout, part, out, n)

    def side_now(self):
        """The records of moe's side, once (before a GP_AWAIT; at the end with no CPU task)."""
        side, self._side = getattr(self, "_side", None), None
        if side is not None:
            side()

    def _moe(self, prog, i, lat, idx, val, out, t):
        d, cfg = self.dev, self.dev.cfg
        # (i "mtp": the MTP block's experts, Super; every pair on the CPU for now)
        E, inner, k = cfg.moe[i] if i in cfg.moe else cfg.moe[min(cfg.moe)]
        L = cfg.latent
        Im = max(n for _, n, _ in cfg.moe.values())
        Em = max(e for e, _, _ in cfg.moe.values())
        S = self.scratch
        if d.nv4 and (t > NV4_HOT_MAX or i == "mtp"):
            # Super, a prompt group: every pair on the CPU (NM_MOE's tiles), until the GPU has a group
            # form of the KQ_NV4 experts (S6)
            slots_all = d.no_hot[i]
        else:
            slots_all = None
        fplan = not d.i8 and not ZC_FLOAT and PLAN_F and t > ZC_ROWS
        if t >= d.zc_min and (d.i8 or ZC_FLOAT or fplan) and d.zc_tables is not None and i in d.zc_tables:
            pup, pdn = d.zc_tables[i]
            gidx, gout, xd = idx, out, d.xdev if t <= 8 else None
            mix = (d.nv4 and PLAN_CPU > 0 or fplan) and not d.staged(t)
            if mix:
                # Super's prompt group (S6): NM_PLAN gives the cold experts of few pairs to the CPU (NM_MOE
                # in its task, as a step's cold pairs), the others and the hot ones to NM_GROUP_MOE; the
                # sums added after
                bufs = d.bufs(t)
                cold, cnt = S("cold", t * k, np.int32), S("cnt", 1, np.int32)
                gidx, gout, part = S("zidx", t * k, np.int32), S("gout", t * L), S("part", t * L)
                prog.emit(P.NM_PLAN, idx, d.hot[i][0], cold, cnt, gidx, t * k, k, prog.slot("nreal"), E,
                          *((PLAN_F_CPU, PLAN_F_GPU, PLAN_F_PAIR) if fplan else (float(PLAN_CPU), float(PLAN_GPU))))
                fl = pinned((2,), np.int64, free=True)
                fl[:] = 0
                prog.keep.append(fl)
                seq = prog.slot("seq")
                prev, self._last_fl = self._last_fl, fl
                prog.emit(P.SIGNAL, fl[0:1], seq, lat, bufs["lat"], bufs["lat"].nbytes, val, bufs["val"],
                          bufs["val"].nbytes, cold, bufs["cold"], bufs["cold"].nbytes,
                          prev[1:2] if prev is not None else None)
            elif xd is not None:
                # an expert device: the pairs of its experts to it (through the CPU's task, as a step's
                # cold pairs), the others here; the sums added after
                mine, rest = d.xsplit(i)
                bufs = d.bufs(t)
                cnt = S("cnt", 1, np.int32)
                gidx, gout, part = S("zidx", t * k, np.int32), S("gout", t * L), S("part", t * L)
                prog.emit(P.NM_SPLIT, idx, mine, S("cold", t * k, np.int32), cnt, t * k, k, prog.slot("nreal"))
                prog.emit(P.NM_SPLIT, idx, rest, gidx, S("zcnt", 1, np.int32), t * k, k, prog.slot("nreal"))
                fl = pinned((2,), np.int64, free=True)
                fl[:] = 0
                prog.keep.append(fl)
                seq = prog.slot("seq")
                prev, self._last_fl = self._last_fl, fl
                prog.emit(P.SIGNAL, fl[0:1], seq, lat, bufs["lat"], bufs["lat"].nbytes, val, bufs["val"],
                          bufs["val"].nbytes, S("cold", t * k, np.int32), bufs["cold"], bufs["cold"].nbytes,
                          prev[1:2] if prev is not None else None)
            st = d.stage_tabs() if d.staged(t) and d.pool is not None and i in d.pool.row else None
            if st is not None:
                # (the layer's cold experts in buffer j % 2: wait for their copy; the stage tables' rows)
                j = d.pool.row[i]
                pup, pdn = st["pup"][j], st["pdn"][j]
                prog.emit(P.FETCH_WAIT, j)
                prog.stage = True
            if t <= ZC_ROWS and not d.nv4:
                # (a step or a verify group: the step's kernels on the tables, a warp on rows of a pair)
                self._hot(prog, lat, gidx, val, d.hot[i][0], pup, pdn, S("act", t * k * Im), S("de", t * k * L),
                          gout, k, L, inner, t, Im, tab=True)
            else:
                # (KQ_NV4: the up tiles quantize the act themselves, no float act)
                prog.emit(P.NM_GROUP_MOE, lat, gidx, val, pup, pdn, S("qx", t * L, np.int8), S("sx", t * L // 32),
                          S("act", 32 if d.nv4 else t * k * Im), S("qa", t * k * Im, np.int8), S("sa", t * k * Im // 32),
                          S("de", t * k * L), gout, S("work", 2 * Em + 1 + t * k, np.int32), k, L, inner, t, E,
                          prog.slot("nreal"), S("qxl", t * L, np.int8) if X16 else None,
                          S("qal", t * k * Im, np.int8) if X16 else None, KQ_NV4 if d.nv4 else None,
                          (2 if TILES else 1) if st is not None else None)   # (22: the experts in the GPU's memory; 2: NV4T)
            if st is not None:
                # (the buffer free; the copies of the layer two on into it)
                prog.emit(P.FETCH_DONE, j % 2)
                if j + 2 < len(d.pool.layers):
                    prog.emit(P.FETCH, st["ranges"][j + 2], len(st["ranges"][j + 2]), j + 2, j % 2, *d.stage_mode())
            if mix:
                cc = P.Program()
                upm, dnm = d.m.experts(i, "up"), d.m.experts(i, "down")
                cc.emit(P.NM_MOE, bufs["lat"], bufs["cold"], bufs["val"], t, k, upm.data, upm.type, dnm.data, dnm.type,
                        L, inner, bufs["scratch"], bufs["out"], int(X16), *d.m.node1(i))
                cpu = cc.finish()
                prog.cpu_progs.append(cpu)
                prog.emit(P.CPU_TASK, cpu.buf, fl[0:1], fl[1:2], seq)
                self.side_now()
                self._await_add(prog, fl, seq, bufs["out"], part, gout, out, t * L, cnt)
            elif xd is not None:
                # the CPU's task: the post, NM_MOE on what the device refused (else nothing: zeros), the join
                cc = P.Program()
                upm, dnm = d.m.experts(i, "up"), d.m.experts(i, "down")
                lcold, posted = np.zeros(t * k, np.int32), np.zeros(1, np.int64)
                d._xkeep += [lcold, posted]
                cc.emit(P.NM_REMOTE_POST, bufs["lat"], bufs["cold"], bufs["val"], t * k, xd.ctx_bytes, i, lcold,
                        posted, k)
                cc.emit(P.NM_MOE, bufs["lat"], lcold, bufs["val"], t, k, upm.data, upm.type, dnm.data, dnm.type,
                        L, inner, bufs["scratch"], bufs["out"], int(X16), *d.m.node1(i))
                cc.emit(P.REMOTE_JOIN, xd.ctx_bytes, posted, bufs["out"])
                cpu = cc.finish()
                prog.cpu_progs.append(cpu)
                prog.emit(P.CPU_TASK, cpu.buf, fl[0:1], fl[1:2], seq)
                self.side_now()
                self._await_add(prog, fl, seq, bufs["out"], part, gout, out, t * L, cnt)
            return
        slots, up, down = d.hot[i]
        if slots_all is not None:
            slots = slots_all
        act, de = S("act", t * k * Im), S("de", t * k * L)
        if d.hot_n[i] == E and slots_all is None and d.pool is None:
            # every expert of the layer on the GPU: no part of the CPU (split, signal, task, await, add)
            self._hot(prog, lat, idx, val, slots, up, down, act, de, out, k, L, inner, t, Im)
            return
        bufs = d.bufs(t)
        cold, cnt = S("cold", t * k, np.int32), S("cnt", 1, np.int32)
        gout, part = S("gout", t * L), S("part", t * L)
        rec = getattr(self, "_router", None)
        if rec is not None and t == 1 and rec[1][5] == prog._enc(idx):
            # (a step's split in the router's record: k_router_sig_r; a group's rows would share its one
            # block: a group of 4 1-2 ms slower)
            for a, v in ((10, slots), (11, cold), (12, cnt), (13, prog.slot("nreal"))):
                _patch(prog, rec, a, v)
        else:
            prog.emit(P.NM_SPLIT, idx, slots, cold, cnt, t * k, k, prog.slot("nreal"))
        fl = pinned((2,), np.int64, free=True)
        fl[:] = 0
        prog.keep.append(fl)
        seq = prog.slot("seq")
        prev, self._last_fl = self._last_fl, fl
        prog.emit(P.SIGNAL, fl[0:1], seq, lat, bufs["lat"], bufs["lat"].nbytes, val, bufs["val"],
                  bufs["val"].nbytes, cold, bufs["cold"], bufs["cold"].nbytes, prev[1:2] if prev is not None else None)
        if slots_all is None:
            self._hot(prog, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t, Im)
        # the CPU's part: the cold pairs on the full stacks in host memory, the threads of node 1 on its
        # copy when the weightkeeper has one (a run of up to 8 rows: those of an expert device's experts
        # on it, its sum added after)
        cc = P.Program()
        upm, dnm = d.m.experts(i, "up"), d.m.experts(i, "down")
        cold_cpu, xd = bufs["cold"], d.xdev if t <= (8 if os.environ.get("RECLANK_NM_XGROUP", "1") != "0" else 1) else None
        if xd is not None:
            cold_cpu, posted = np.zeros(t * k, np.int32), np.zeros(1, np.int64)
            d._xkeep += [cold_cpu, posted]
            cc.emit(P.NM_REMOTE_POST, bufs["lat"], bufs["cold"], bufs["val"], t * k, xd.ctx_bytes, i, cold_cpu,
                    posted, k)
        cc.emit(P.NM_MOE, bufs["lat"], cold_cpu, bufs["val"], t, k, upm.data, upm.type, dnm.data, dnm.type,
                L, inner, bufs["scratch"], bufs["out"], int(X16), *d.m.node1(i))
        if xd is not None:
            cc.emit(P.REMOTE_JOIN, xd.ctx_bytes, posted, bufs["out"])
        cpu = cc.finish()
        prog.cpu_progs.append(cpu)
        prog.emit(P.CPU_TASK, cpu.buf, fl[0:1], fl[1:2], seq)
        self.side_now()
        if slots_all is not None:
            # (every pair the CPU's: its sums are the output)
            prog.emit(P.AWAIT, fl[1:2], seq, bufs["out"], out, t * L * 4, cnt)
            return
        self._await_add(prog, fl, seq, bufs["out"], part, gout, out, t * L, cnt)


class NemotronGPU:
    # the state of one sequence: lanes.NmLane puts a second one in place (its view) for lane B's runs
    # (the programs of more than NV4_HOT_MAX rows, a prompt's, are both lanes': gprogs; a prompt runs alone and
    # its run binds the lane's cache)
    LANE_ATTRS = ("cache", "cache_dev", "progs", "_bufs", "_ascr", "media", "lane_id")

    def __init__(self, model, ctx=8192, hot_gb=None, counts=None, graph=True, reserve_gb=0.0):
        from reclank.backends.cuda.runtime import caps
        self.m, self.cfg = model, model.cfg
        if Q8R and not caps().get("dp4a"):
            self.m = _Q8RModel(model)
        cfg = self.cfg
        self.reserve_gb = float(reserve_gb)     # (the room kept out of the hot experts: relax keeps it free too)
        # Super's NVFP4 experts (KQ_NV4 rows): the hot pairs of a step or a verify group by NM_HOT_NV4, a
        # prompt group's on the CPU (NM_GROUP_MOE reads IQ2_XXS rows only; S6)
        self.nv4 = model.experts(min(cfg.moe), "up").type == KQ_NV4
        self.ctx = ctx
        self.graph = graph
        self.groups = GROUPS
        self.big = {}                   # the big groups' programs: size -> the mirror's arrays of their own
        self.lane_id = 0                # (lane B: 1; its programs get scratches of the C side of their own)
        # the image spans of the sequence: (first position, rows [n, hidden]) whose rows replace the
        # embeddings of their positions (Super's vision: radio.py)
        self.media = []
        self._bufs = {}
        self.mirror = Mirror()
        self.cache = NemotronKQCache(cfg, ctx, kv="bf16")
        self._two_sets()
        self._ascr = None
        self.cache_dev = _NmDevCache()
        # the hot experts: the share of every layer's experts that the memory holds
        per = {i: (model.experts(i, "up").data.nbytes + model.experts(i, "down").data.nbytes) // E
               for i, (E, _inner, _k) in cfg.moe.items()}
        total = sum(per[i] * cfg.moe[i][0] for i in cfg.moe)
        if hot_gb is None and os.environ.get("RECLANK_GPU_HOT_GB"):
            hot_gb = float(os.environ["RECLANK_GPU_HOT_GB"])
        if hot_gb is None:
            free = mem_info()[0]
            hot_gb = max(0.0, (free - self.dense_bytes() - self.kv_bytes(ctx) - self.prompt_bytes()) / 1e9 -
                         MARGIN_GB - reserve_gb)
        share = min(1.0, hot_gb * 1e9 / total)
        self.hot, self.hot_n = {}, {}
        self.pool = None
        self._zc_reg = {}
        if self.nv4 and POOL:
            # (Super: the pool of segments, filled with the experts of the best scores of every layer)
            self.pool = pool = NmPool(self, counts)
            pool.add_segments(int(min(hot_gb * 1e9, total) // pool.seg_bytes))
            pool.plan(None)
            pool.land(wait=True)
            pool.changed = set()
            dummy = np.zeros(256, np.uint8)
            for i, (E, inner, k) in cfg.moe.items():
                self.hot[i] = (pool.slots[i], dummy, dummy)
                self.hot_n[i] = int((pool.slots[i] >= 0).sum())
            share = pool.held() / sum(E for E, _n, _k in cfg.moe.values())
        else:
            for i, (E, inner, k) in cfg.moe.items():
                n = int(share * E)
                if counts is not None and i in counts:
                    pick = np.sort(np.argsort(-np.asarray(counts[i]), kind="stable")[:n])
                else:
                    pick = np.arange(n)
                self.hot_n[i] = n
                self.hot[i] = self._hot_stacks(i, pick)
        self.hot_share = share
        if self.pool is not None:
            self.groups = tuple(sorted(set(GROUPS) | set(BIG_GROUPS)))
            self.pool_target = len(self.pool.segs)
        # (every expert of a layer cold: a Super prompt group's)
        self.no_hot = {i: np.full(E, -1, np.int32) for i, (E, _n, _k) in cfg.moe.items()}
        if cfg.mtp:
            E0 = cfg.moe[min(cfg.moe)][0]
            self.no_hot["mtp"] = np.full(E0, -1, np.int32)
            self.hot["mtp"] = (self.no_hot["mtp"], np.zeros(256, np.uint8), np.zeros(256, np.uint8))
            self.hot_n["mtp"] = 0
        self.hot_bytes = (self.pool.nbytes() if self.pool is not None else
                          sum(up.nbytes + down.nbytes for _s, up, down in self.hot.values()))
        self.counts = counts
        # a second GPU of the process with cold experts (RECLANK_EXPERT_GPUS: backends/cuda/devices.py)
        from reclank.backends.cuda.devices import nm_expert_device
        self.xdev = nm_expert_device(self)
        self._xkeep = []
        # (Super: the steps and the verify groups split, the CPU's cold experts beat their copies over the
        # link; a prompt group of more than NV4_HOT_MAX rows every pair on the GPU, NM_GROUP_MOE's
        # k_nm_gtile_nv4 on the tensor cores)
        self.zc_min = ZC_MIN if ZC_MIN is not None else NV4_HOT_MAX + 1 if self.nv4 else 1
        # the int8 products: a GPU without dp4a or the tensor cores (Maxwell) takes the float forms
        # (kernels/nemotron/iq2_f32.cu: NM_HOT_MOE's, NM_GROUP_MOE's tiles, KQ_IQ2R's KQ_LINEAR)
        self.i8 = I8 and int8_tiles()
        # every program counts the router's choices of each layer here (GP_COUNT; route_counts() reads
        # them back): with an expert device (refresh_experts), or RECLANK_NM_ROUTES=1
        Em = max(E for E, _n, _k in cfg.moe.values())
        from reclank.backends.cuda.devices import REFRESH
        self.routes = (np.zeros((cfg.layers, Em), np.int32) if self.xdev is not None and REFRESH
                       or os.environ.get("RECLANK_NM_ROUTES") == "1" else None)
        self._xsplit = {}
        self._seen = None if self.routes is None else self.routes.copy()
        self._score = None
        self.progs = {}                 # the programs of a step and of the small groups (this lane's)
        self.gprogs = {}                # those of the prompt groups (both lanes')
        self.zc_tables = None
        self._prog(1)
        self.link_gbps = h2d_gbps()
        from reclank.backends.cuda.runtime import caps
        if ZC and self.link_gbps >= ZC_GBPS and (not self.nv4 or caps().get("tc")) and (self.i8 or ZC_FLOAT or PLAN_F):
            t0 = time.time()
            self._zero_copy()
            if self.nv4 and self.zc_tables is not None:
                print("NemotronGPU: the experts mapped for the prompt groups in %.1f s" % (time.time() - t0), flush=True)
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
        prompt; with the pool, BIG_GROUPS's take theirs from it): group_bytes of each."""
        return int(sum(self.group_bytes(t) for t in GROUPS))

    def group_bytes(self, t):
        """The device memory of the program of a prompt group of t rows: a part that grows with its
        rows (the step's activations, which the layers share; the routed experts' scratch; the int8 x
        of the products) and one that does not (rc_gpu_load's own buffers). The 3060 measured 538 KB a
        row and 19.5 MB a program for the 74B (824 MB for the five; this gives 552 KB, 853 MB)."""
        c = self.cfg
        hid, L, inner, conv = c.hidden, c.latent, c.m_inner, c.conv_dim
        hq, hk = c.heads * c.head_dim, c.kv_heads * c.head_dim
        Emax = max(e for e, _, _ in c.moe.values())
        Im = max(n for _, n, _ in c.moe.values())
        k = max(kk for _, _, kk in c.moe.values())
        wide = max(inner, c.shared_inner, hid)
        act = 4 * (5 * hid + 2 * inner + 2 * conv + c.m_heads + 3 * hq + 2 * hk + Emax + 2 * k + 2 * L +
                   c.shared_inner) + wide * (1 + 4 / 32 + 4 / 16)
        moe = (2 if self.nv4 else 6) * k * Im + 4 * k * Im / 32 + 4 * k * L + 10 * L + 4 * L / 32 + 4 * k
        qx = sum(2 * n + 4 * n / 32 for n in {hid, inner, c.shared_inner, L, hq})
        return (act + moe + qx) * t + 20e6

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
        stacks in host memory mapped for the GPU (registered once; 17 GB pinned). With the pool: its
        tables."""
        from reclank.backends.cuda.runtime import lib
        if self.pool is not None:
            self.zc_tables = {i: (self.pool.pup[i], self.pool.pdn[i]) for i in self.pool.pup}
            return
        L = lib()
        tables = {}
        layers = list(self.cfg.moe.items())
        if self.cfg.mtp and self.nv4:
            layers.append(("mtp", self.cfg.moe[min(self.cfg.moe)]))     # (the MTP block's prompt parts)
        for i, (E, inner, k) in layers:
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
            self._ascr = np.zeros(max(ATT_PARTS, self.groups[-1]) * c.heads * (c.head_dim + 2), np.float32)
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
        store = self.progs if t <= NV4_HOT_MAX else self.gprogs
        e = store.get(t)
        if e is None:
            big = self.pool is not None and t in BIG_GROUPS
            if big:
                # (the memory of a big group's program from the pool: relax gives it back after the prompt)
                need = self.group_bytes(t) + BIG_MARGIN_GB * 1e9 - mem_info()[0]
                if need > 0:
                    n = self.pool.shrink(need)
                    print("NemotronGPU: the hot experts %.1f -> %.1f GB for the program of a group of %d" % (
                        (self.pool.nbytes() + n) / 1e9, self.pool.nbytes() / 1e9, t), flush=True)
                before = set(self.mirror.arrays)
                t0 = time.time()
            prog = compile_nemotron_step(self.m, t, em=NmEmit(self, t))
            if getattr(prog, "stage", False):
                # (the copies of the first two MoE layers' cold experts start first)
                st = self.stage_tabs()
                prog.recs[:0] = [(P.FETCH, [prog._enc(st["ranges"][j]), prog._enc(len(st["ranges"][j])), prog._enc(j),
                                            prog._enc(j % 2)] + [prog._enc(v) for v in self.stage_mode()])
                                 for j in range(min(2, len(self.pool.layers)))]
            # the head on the last real row of the group (slot "last": a group of more rows than tokens)
            head = self.m.M(LM + "lm_head")
            logits = np.zeros((1, head.rows), np.float32)
            xn = prog.names["xn"]
            src = xn[t - 1:t]
            if t > 1:
                src = np.zeros((1, xn.shape[1]), np.float32)
                iota = np.arange(t, dtype=np.int32)
                prog.keep.append(iota)
                prog.emit(P.DS_GATHER, xn, iota, prog.slot("last"), src, xn.shape[1])
            prog.recs.append((P.KQ_LINEAR, [prog._enc(a) for a in (
                None, None, None, src, head.data, head.type, head.rows, head.cols, 1, logits)]))
            prog.names["logits"] = logits
            prog.finish()
            g = self._load(prog, t)
            e = store[t] = (prog, g)
            if big:
                g.prepare()
                self.big[t] = set(self.mirror.arrays) - before
                print("NemotronGPU: the program of a group of %d made in %.1f s" % (t, time.time() - t0), flush=True)
        return e

    def staged(self, t):
        """A group of t rows takes its cold experts by the copies to the stage buffers."""
        return STAGE and self.pool is not None and self.zc_tables is not None and t >= STAGE_MIN

    def stage_tabs(self):
        """The tables of the staged groups (shared by their programs): for each MoE layer j (the pool's
        rows) the device address of each expert (its slot when held, else its place in stage buffer j % 2;
        stage_fill writes them before each run) and the ranges of its cold experts' copies (2 E rows, a
        range of 0 bytes copies nothing)."""
        st = getattr(self, "_stage", None)
        if st is None:
            R, E = len(self.pool.layers), self.pool.E
            st = self._stage = {"pup": np.zeros((R, E), np.int64), "pdn": np.zeros((R, E), np.int64),
                                "ranges": [np.zeros((2 * E, 3), np.int64) for _ in range(R)], "bufs": None}
        return st

    def stage_mode(self):
        """GP_FETCH's operands 4-6 of the stage copies: NV4T (mode 1, the columns of the up and down rows), or ()."""
        if not TILES:
            return ()
        E, inner, k = self.cfg.moe[self.pool.layers[0]]
        return (1, self.cfg.latent, inner)

    def stage_fill(self):
        """Before a staged run: the stage buffers (the pool gives back their room), and the tables and
        ranges of every layer from the pool's set now."""
        from reclank.backends.cuda.gpumm import Buffer
        st, pool = self.stage_tabs(), self.pool
        E, pu, pd = pool.E, pool.pu, pool.pd
        if st["bufs"] is None:
            need = 2 * E * pool.per
            short = need + BIG_MARGIN_GB * 1e9 - mem_info()[0]
            if short > 0:
                n = pool.shrink(short)
                print("NemotronGPU: the hot experts -%.1f GB for the stage buffers (%.1f GB)" % (n / 1e9, pool.nbytes() / 1e9),
                      flush=True)
            st["bufs"] = [Buffer(E * pool.per, "program", pinned=True) for _ in range(2)]
        ar = np.arange(E, dtype=np.int64)
        if TILES:
            # (every expert in the buffer at its own place, in NV4T: the cold ones from the host's rows, the hot ones
            # from their slots, both through k_nm_tile_pull)
            for j, i in enumerate(pool.layers):
                b = st["bufs"][j % 2].ptr
                up, dn = b + ar * pu, b + E * pu + ar * pd
                st["pup"][j], st["pdn"][j] = up, dn
                held = pool.slots[i] >= 0
                hu, hd, _du, _dd = pool.host[i]
                su = np.where(held, pool.pup[i], hu + ar * pu)
                sd = np.where(held, pool.pdn[i], hd + ar * pd)
                rg = st["ranges"][j]
                rg[0::2] = np.stack([su, up, np.full(E, pu)], 1)
                rg[1::2] = np.stack([sd, dn, np.full(E, pd)], 1)
            for a in (st["pup"], st["pdn"]):
                if self.mirror.owner(a.ctypes.data) is not None:
                    self.mirror.buffer_of(a).upload(a)
            return
        for j, i in enumerate(pool.layers):
            b = st["bufs"][j % 2].ptr
            held = pool.slots[i] >= 0
            cold = np.nonzero(~held)[0]
            q = np.arange(len(cold), dtype=np.int64)
            up, dn = b + q * pu, b + E * pu + q * pd
            st["pup"][j] = np.where(held, pool.pup[i], 0)
            st["pdn"][j] = np.where(held, pool.pdn[i], 0)
            st["pup"][j][cold], st["pdn"][j][cold] = up, dn
            hu, hd, _du, _dd = pool.host[i]
            rg = st["ranges"][j]
            rg[:] = 0
            rg[0:2 * len(cold):2] = np.stack([hu + cold * pu, up, np.full(len(cold), pu)], 1)
            rg[1:2 * len(cold):2] = np.stack([hd + cold * pd, dn, np.full(len(cold), pd)], 1)
        del ar
        for a in (st["pup"], st["pdn"]):
            if self.mirror.owner(a.ctypes.data) is not None:
                self.mirror.buffer_of(a).upload(a)

    def stage_free(self):
        st = getattr(self, "_stage", None)
        if st is not None and st["bufs"] is not None:
            for b in st["bufs"]:
                b.free()
            st["bufs"] = None

    def relax(self):
        """After a prompt: the big groups' programs freed (with the device copies of their own arrays),
        the pool back to its segments, their free slots taken by the best experts (the copies land
        over the next runs)."""
        self.stage_free()
        if self.pool is None or (not self.big and len(self.pool.segs) >= self.pool_target):
            return
        keep = {a.ctypes.data for a in (self.pool._slots, self.pool._pup, self.pool._pdn, self.pool.cnt)}
        if getattr(self, "_stage", None) is not None:
            keep |= {self._stage["pup"].ctypes.data, self._stage["pdn"].ctypes.data}
        for t, own in self.big.items():
            _prog, g = self.gprogs.pop(t)
            for b in g.named.values():
                b.free()
            g.named = {}
            g.close()
            for st in own - keep:
                self.mirror.forget(st)
        self.big = {}
        # (back to its segments while the GPU keeps MARGIN_GB and the reserve free: the programs made since may hold
        # some; the image encoder's room had gone to the hot experts after a long prompt)
        room = int((mem_info()[0] - (MARGIN_GB + self.reserve_gb) * 1e9) // self.pool.seg_bytes)
        k = self.pool.add_segments(max(0, min(self.pool_target - len(self.pool.segs), room)))
        if k:
            self.pool.plan(None)
            print("NemotronGPU: the hot experts back to %.1f GB (%d copies on their way)" % (
                self.pool.nbytes() / 1e9, sum(len(it) for _j, _r, it in self.pool.pending)), flush=True)

    def _mtp_prog(self, t, head=True):
        """The MTP block on t rows (kq.compile_nemotron_mtp with NmEmit: its attention on K.mtp / V.mtp
        of the cache, its experts on the CPU), with head the head on every row (a prompt's rows need
        none: their keys and values only)."""
        store = self.progs if t <= NV4_HOT_MAX else self.gprogs
        e = store.get(("mtp", t, head))
        if e is None:
            # (the MTP block's prompt parts of more than 256 rows: freed with the big groups after the prompt)
            big = self.pool is not None and t > 256
            before = set(self.mirror.arrays) if big else None
            from reclank.models.nemotron.kq import compile_nemotron_mtp
            prog = compile_nemotron_mtp(self.m, t, em=NmEmit(self, t))
            if head:
                hm = self.m.M(LM + "lm_head")
                logits = np.zeros((t, hm.rows), np.float32)
                prog.recs.append((P.KQ_LINEAR, [prog._enc(a) for a in (
                    None, None, None, prog.names["out"], hm.data, hm.type, hm.rows, hm.cols, t, logits)]))
                prog.names["logits"] = logits
            prog.finish()
            g = self._load(prog, t)
            e = store[("mtp", t, head)] = (prog, g)
            if big:
                self.big[("mtp", t, head)] = set(self.mirror.arrays) - before
        return e

    def _load(self, prog, t):
        """The GPUProgram of prog (t rows); when the GPU has no room for it, the pool gives back segments
        (at least ROOM_GB) and it tries again (relax takes them back while the GPU keeps MARGIN_GB free)."""
        for k in range(8):
            try:
                with self.lane_scratch(t):
                    return GPUProgram(prog, graph=self.graph, mirror=self.mirror)
            except MemoryError as exc:
                if self.pool is None or not self.pool.segs or k == 7:
                    raise
                n = self.pool.shrink(ROOM_GB * 1e9)
                print("NemotronGPU: no room for a program of %d rows (%s): the hot experts -%.1f GB (%.1f GB)" % (
                    t, exc, n / 1e9, self.pool.nbytes() / 1e9), flush=True)

    def lane_scratch(self, t):
        """(with) the making of a program of t rows: lane B's steps and verify groups get scratches of the C
        side of their own (they run beside lane A's; a prompt's run alone and share them)."""
        import contextlib
        if not self.lane_id or t > NV4_HOT_MAX:
            return contextlib.nullcontext()
        from reclank.backends.cuda.runtime import lib

        @contextlib.contextmanager
        def own():
            lib().rc_gpu_scratch_own(1)
            try:
                yield
            finally:
                lib().rc_gpu_scratch_own(0)
        return own()

    def mtp(self, tokens, hidden, pos, head=True):
        """The MTP block (Super) on t rows at the positions pos.. (its keys and values there): tokens,
        the token after each position; hidden [t, hidden], the target's rows after norm_f there (or
        the out of the draft before). Return (logits [t, vocab], out [t, hidden])."""
        t = len(tokens)
        prog, g = self._mtp_prog(t, head)
        prog.names["H"][:] = self.m.embed(tokens)
        prog.names["hin"][:] = np.asarray(hidden, np.float32).reshape(t, -1)
        g.upload("H")
        g.upload("hin")
        kw = {"pos": pos, "nreal": t, "hs": self.cfg.head_dim}
        kw.update(self.cache.params())
        g.bind(kw, self.cache_dev, scratch=("scores",))
        g.run()
        if not head:
            return None, None
        g.download("logits")
        g.download("out")
        hm = self.m.M(LM + "lm_head")
        lg = prog.names["logits"].copy()
        return (lg * np.float32(hm.scale) if hm.scale is not None else lg), prog.names["out"].copy()

    def last_hidden(self, t):
        """The rows after norm_f of the last run of t rows (a group of the prompt or a step)."""
        prog, g = self._prog(next(s for s in (1,) + self.groups if s >= t))
        g.download("xn")
        return prog.names["xn"][:t].copy()

    def _run(self, ids, pos):
        r = self._prepare(ids, pos)
        r[1].run()
        return self._finish(r, pos)

    def _prepare(self, ids, pos):
        """The program of a run of ids at pos, its input and its parameters: (prog, gp, t)."""
        t = len(ids)
        size = next((s for s in (1,) + self.groups if s >= t), None)
        assert size is not None, "a run of at most %d tokens" % self.groups[-1]
        # (NM_KV writes the rows of all size tokens at pos: past max_len they were the next buffers on
        # the GPU, and a later run read a broken one)
        if pos + size > self.cache.max_len:
            raise ValueError("the cache holds %d positions: a run of %d at %d" % (self.cache.max_len, size, pos))
        prog, g = self._prog(size)
        x = prog.names["H"]
        x[:t] = self.m.embed(ids)
        x[t:] = 0.0
        for s0, rows in self.media:
            a, b = max(s0, pos), min(s0 + rows.shape[0], pos + t)
            if a < b:
                x[a - pos:b - pos] = rows[a - s0:b - s0]
        g.upload("H")
        kw = {"pos": pos, "nreal": t, "hs": self.cfg.head_dim, "last": t - 1}
        kw.update(self.cache.params())
        g.bind(kw, self.cache_dev, scratch=("scores",))
        if getattr(prog, "stage", False):
            self.stage_fill()
        return prog, g, t

    def _finish(self, r, pos):
        """After the run r (_prepare's) at pos: the pool, the cache's length, the logits of its last row."""
        prog, g, t = r
        self.after_run(prog, g, t)
        g.download("logits")
        self.cache.n = pos + t
        head = self.m.M(LM + "lm_head")
        lg = prog.names["logits"][0].copy()
        return lg * np.float32(head.scale) if head.scale is not None else lg

    def after_run(self, prog, gp, t):
        """The pool after a run of t real rows (the GPU idle between runs): the copies that are done
        held, the routers' choices observed, the next copies planned, the tables to the GPU."""
        pool = self.pool
        if pool is None:
            return
        pool.land()
        if HOT_FOLLOW:
            # (the counts read and the plan made every PLAN_EVERY tokens of the runs of a few rows, with
            # their budget)
            pool.owed += t
            if pool.owed >= PLAN_EVERY or t > PLAN_EVERY:
                pool.observe()
                pool.plan(min(HOT_BUDGET * pool.owed, HOT_CAP if t <= PLAN_EVERY else HOT_CAP_GROUP))
                pool.owed = 0
        pool.upload()
        for i in pool.layers:
            self.hot_n[i] = int((pool.slots[i] >= 0).sum())

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

    def parts(self, n, room):
        """The tokens of each run of a prompt of n tokens (room: the positions the cache has left). Bolt:
        the groups of GROUPS whole, the largest first. Super (NM_PLAN's groups): the runs of the least
        time by GROUP_COST, a group taking fewer tokens than its rows where that is faster (943 tokens
        one group of 1024, not 3 x 256 + 2 x 64 + ...; a group of 16 is slower than four of 4)."""
        if not self.nv4 or self.zc_tables is None:
            out, c0 = [], 0
            while c0 < n:
                out.append(next((s for s in self.groups[::-1] if s <= n - c0), 1))
                c0 += out[-1]
            return out
        sizes = [s for s in (1,) + self.groups if s in GROUP_COST]
        head = []
        top = self.groups[-1]
        while n > 2 * top and room - n >= 0:
            head.append(top)                # (the largest whole, then the plan of the rest)
            n -= top
            room -= top
        best, pick = [0.0] * (n + 1), [0] * (n + 1)
        for m in range(1, n + 1):
            best[m] = float("inf")
            for s in sizes:
                # (a group of s rows at the end of the cache: NM_KV writes all its rows)
                if s > m and n - m + s > room:
                    continue
                c = GROUP_COST[s] + best[max(0, m - s)]
                if c < best[m]:
                    best[m], pick[m] = c, s
        out, m = head, n
        while m > 0:
            out.append(min(pick[m], m))
            m -= out[-1]
        return out

    def prefill(self, ids, pos=None, rows=None, relax=True):
        """Run the tokens ids from pos (default: the cache's n); the logits of the last. rows(at,
        hidden): after each run, its rows after norm_f (the MTP block's input). relax False: the big
        groups' programs kept (a prompt in parts: relax after the last)."""
        pos = self.cache.n if pos is None else pos
        if pos == 0:
            self.reset()
        out = None
        c0 = 0
        for n in self.parts(len(ids), self.cache.max_len - pos):
            out = self._run(ids[c0:c0 + n], pos + c0)
            if rows is not None:
                rows(pos + c0, self.last_hidden(n))
            c0 += n
        if relax:
            self.relax()
        return out

    def step(self, token):
        return self._run([token], self.cache.n)

    def route_counts(self):
        """(layers, experts): the router's choices of all runs so far (RECLANK_NM_ROUTES=1)."""
        mir = self.mirror
        if mir.owner(self.routes.ctypes.data) is not None:
            mir.buffer_of(self.routes).download(self.routes)
        return self.routes.copy()

    def xsplit(self, i):
        """The NM_SPLIT tables of layer i with an expert device (all programs share them): mine (-1:
        the device's experts, its pairs to the CPU's task) and rest (-1: the others, the pairs here)."""
        e = self._xsplit.get(i)
        if e is None:
            E = self.cfg.moe[i][0]
            e = self._xsplit[i] = (np.zeros(E, np.int32), np.zeros(E, np.int32))
            self._split_fill(i)
        return e

    def _split_fill(self, i):
        mine, rest = self._xsplit[i]
        held = self.xdev.held[i, :mine.size].astype(bool)
        mine[:] = np.where(held, -1, 0)
        rest[:] = np.where(held, 0, -1)

    def refresh_experts(self):
        """Between prompts (no run in flight): the expert device's experts anew by the routing since the
        last call and before (half for each REFRESH_HALF tokens since, devices.py; the profile breaks the
        ties of the experts not chosen yet): the experts of the text at hand. A conversation's own turns
        predict its next, another's do not (three two-turn chats: the pairs of the second turns that no
        GPU holds 11-19% -> 4-6%, its decode 30.5-34.6 -> 37.7-38.6 tok/s; 400-1300 experts, 0.5-1.6
        GB in 0.07-0.22 s). Return (experts copied, bytes, seconds), or None."""
        from reclank.backends.cuda.devices import REFRESH, REFRESH_HALF
        if self.xdev is None or self.routes is None or not REFRESH:
            return None
        now = self.route_counts()
        new = (now - self._seen).astype(np.float64)
        self._seen = now
        if not new.any():
            return None
        if self._score is None:
            self._score = np.zeros_like(new)
            for i in self.cfg.moe:
                c = np.asarray(self.counts[i], np.float64) if self.counts and i in self.counts else None
                if c is not None and c.sum() > 0:
                    self._score[i, :c.size] = 1e-6 * c / c.sum()
        tokens = new.sum() / sum(k for _E, _n, k in self.cfg.moe.values())
        self._score = 0.5 ** (tokens / REFRESH_HALF) * self._score + new
        t0 = time.time()
        n, nbytes = self.xdev.refresh({i: self._score[i] for i in self.cfg.moe})
        for i in self._xsplit:
            self._split_fill(i)
            for a in self._xsplit[i]:
                self.mirror.buffer_of(a).upload(a)
        return n, nbytes, time.time() - t0
