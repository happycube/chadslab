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
# with the pool, GROUPS of at least BIG_FROM rows are big too: made from the pool at a prompt, freed after it (their memory
# the hot experts' while the GPU decodes: the 1024's 0.66 GB on the 3060, about 4% more of a step's pairs on the GPU); the
# smaller ones stay (a chat's every turn runs them)
BIG_FROM = int(os.environ.get("RECLANK_NM_BIG_FROM", "1024"))
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


def cpu_threads():
    """The threads of the CPU's part of a run (RECLANK_GPU_CPU_THREADS, else task_threads)."""
    return max(1, int(os.environ.get("RECLANK_GPU_CPU_THREADS") or task_threads()))


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
# (unset: 2048, or 1024 with a small CPU team, SMALL_CPU: the W-2133 + 3060, a prompt of 1024 staged 165 tok/s against
# 133 mapped with the CPU's share)
STAGE_MIN = int(os.environ["RECLANK_NM_STAGE_MIN"]) if os.environ.get("RECLANK_NM_STAGE_MIN") else None
# a CPU team of fewer threads: the experts' rows rather across the links (TAKE, STAGE_MIN)
SMALL_CPU = int(os.environ.get("RECLANK_NM_SMALL_CPU", "16"))
# Super's steps and verify groups with the pool on a small CPU (SMALL_CPU): NM_TAKE gives the GPU some of each token's
# cold experts (those the expert device lacks), NM_PULL copies their rows across the link into a stage while the CPU's
# task runs the others (the W-2133's 11 threads ~37 GB/s against the 3060's link ~12; its DRAM's 53 GB/s shared:
# the CPU's pairs slower by ~11% beside the copies, so 2 experts gave +4% on the 3060 alone, +2% with the 1070).
# RECLANK_NM_TAKE: the experts a token (a group: that times its tokens); unset: the share of the link in the two rates
# (the CPU's ~TAKE_THREAD_GBPS a thread of its team) of a token's expected cold experts, on a small CPU; 0: none
TAKE = int(os.environ["RECLANK_NM_TAKE"]) if os.environ.get("RECLANK_NM_TAKE") else None
TAKE_THREAD_GBPS = float(os.environ.get("RECLANK_NM_TAKE_THREAD_GBPS", "3.2"))
# the stage buffers in the tiles of the GPU (NV4T: k_nm_tile_pull permutes the KQ_NV4 rows as it copies them,
# every expert of the layer, the hot ones from their slots; k_nm_gtile_nv4t reads each lane's fragments in
# one load); RECLANK_NM_TILES=0: the rows as they are (k_nm_gtile_nv4c)
TILES = os.environ.get("RECLANK_NM_TILES", "1") != "0"
# the fetches of a run (experts.cu GG_FETCH_MAX: their events, reused round)
FETCH_MAX = 256
# the stage in chunks: the act and the outputs of a window of a chunk's pairs at a time (WIN_SLACK times a chunk's share
# of a group's pairs; a chunk of more runs in more windows), not of all the group's: the 4096's scratch 0.88 -> 0.22 GB
# in 8 chunks; RECLANK_NM_WIN=0: the group's
WIN = os.environ.get("RECLANK_NM_WIN", "1") != "0"
# a big prompt group's per-row work in sub-batches of SUB rows (kq._compile_nemotron_sub: the residual stream and the
# routed experts' rows of all of them, the rest of SUB): unset, 1024 on a GPU of less than 16 GB; 0: none
SUB = int(os.environ["RECLANK_NM_SUB"]) if os.environ.get("RECLANK_NM_SUB") else None
WIN_SLACK = float(os.environ.get("RECLANK_NM_WIN_SLACK", "2"))
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
        # (an expert device's experts as good as held: its pairs run beside the CPU's, off the critical path;
        # those the pool holds too leave first)
        xd = getattr(self.g, "xdev", None)
        xh = xd.held[np.asarray(self.layers), :E].astype(bool).ravel() if xd is not None else None
        cand = np.where(held | self.incoming.ravel() | (sc.ravel() <= 0) | (xh if xh is not None else False),
                        -np.inf, sc.ravel())
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
                if xh is not None:
                    hsc = np.where(xh[self.own_r[hs] * E + self.own_e[hs]], -np.inf, hsc)
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

    def attention(self, prog, i, q, k, v, out, t, pos=None):
        """The keys and values into the bfloat16 rows of the cache (NM_KV), then the attention in
        chunks of positions (NM_ATTN2: each block one KV head, its 16 query heads, a chunk; then
        the chunks merged). pos: the slot of the rows' first position (a sub-batch's), else pos."""
        cfg = self.dev.cfg
        K, V, pos = prog.slot("K.%s" % i), prog.slot("V.%s" % i), pos if pos is not None else prog.slot("pos")
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

    def _hot(self, prog, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t, Im, tab=False, tabs=None):
        """NM_HOT_MOE on the pairs of idx: the experts of slots (>= 0) in the stacks up, down; tab: up
        and down are NM_GROUP_MOE's tables (each expert's address: a slot or the mapped host memory),
        every pair of idx >= 0 runs."""
        S = self.scratch
        if self.dev.nv4:
            # (Super's KQ_NV4 experts: float x, docs/NEMOTRON_SUPER_PLAN.md S5; the pool's tables of
            # each expert's address)
            pool = self.dev.pool
            if tabs is None:
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
            gidx, gout, xd = idx, out, d.xdev if t <= 8 and i in cfg.moe else None
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
                # (the stage tables' rows of the layer)
                j = d.pool.row[i]
                pup, pdn = st["pup"][j], st["pdn"][j]
                prog.stage = True
            if t <= ZC_ROWS and not d.nv4:
                # (a step or a verify group: the step's kernels on the tables, a warp on rows of a pair)
                self._hot(prog, lat, gidx, val, d.hot[i][0], pup, pdn, S("act", t * k * Im), S("de", t * k * L),
                          gout, k, L, inner, t, Im, tab=True)
            else:
                # (KQ_NV4: the up tiles quantize the act themselves, no float act)
                # (the pairs' scratch: a window's with the stage in chunks, else the group's; the positions of the pairs
                # after the sort's order, inv, with windows)
                win = d.win_cap(t, d.stage_c) if st is not None else None
                npr = win or t * k
                gargs = [lat, gidx, val, pup, pdn, S("qx", t * L, np.int8), S("sx", t * L // 32),
                         S("act", 32 if d.nv4 else npr * Im), S("qa", npr * Im, np.int8), S("sa", npr * Im // 32),
                         S("de", npr * L), gout, S("work", 2 * Em + 1 + (2 if win else 1) * t * k, np.int32), k, L, inner,
                         t, E, prog.slot("nreal"), S("qxl", t * L, np.int8) if X16 else None,
                         S("qal", npr * Im, np.int8) if X16 else None, KQ_NV4 if d.nv4 else None,
                         (2 if TILES else 1) if st is not None else None]   # (22: the experts in the GPU's memory; 2: NV4T)
                if st is None:
                    prog.emit(P.NM_GROUP_MOE, *gargs)
                else:
                    # the layer's experts in d.stage_c chunks (q = C j + c in buffer q % 2): wait for a chunk's
                    # copies, its tiles in windows of its pairs (operand 23: x0 | nx << 16 | pass << 32 | cap << 40, nm.cu
                    # nm_window; the x and the sort with the first, the sums of each window, or of the last chunk), the
                    # buffer free, the copies of the chunk two on into it
                    C = d.stage_c
                    Ec, nq = E // C, len(st["ranges"])
                    passes = -(-t * k // win) if win else 1
                    for c in range(C):
                        q = j * C + c
                        prog.emit(P.FETCH_WAIT, q % FETCH_MAX)
                        for w in range(passes):
                            prog.emit(P.NM_GROUP_MOE, *(gargs + ([c * Ec | Ec << 16 | w << 32 | (win or 0) << 40]
                                                                  if C > 1 else [])))
                        prog.emit(P.FETCH_DONE, q % 2)
                        if q + 2 < nq:
                            prog.emit(P.FETCH, st["ranges"][q + 2], len(st["ranges"][q + 2]), (q + 2) % FETCH_MAX, q % 2,
                                      *d.stage_mode())
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
        g = d.take(t) if slots_all is None and d.pool is not None and i in d.pool.row else 0
        tabs = None
        if g:
            # (some cold experts the GPU's: out of the CPU's pairs, their rows copied across the link into a stage
            # beside the CPU's task, marked in gsl and their stage's addresses in gup, gdn for NM_HOT_NV4; the
            # router's record then runs alone, not with the SIGNAL)
            pool = d.pool
            gsl, gup, gdn = S("gsl", E + 1, np.int32), S("gup", E, np.int64), S("gdn", E, np.int64)
            take = (cold, cnt, slots, d.xmask(i), gsl, t * k, E, g, pool.pup[i], pool.pdn[i], gup, gdn,
                    S("tstage", g * pool.per, np.uint8), pool.pu, pool.pd, S("tlist", g + 1, np.int32))
            prog.emit(P.NM_TAKE, *take)
            slots, tabs = gsl, (gup, gdn)
        fl = pinned((2,), np.int64, free=True)
        fl[:] = 0
        prog.keep.append(fl)
        seq = prog.slot("seq")
        prev, self._last_fl = self._last_fl, fl
        prog.emit(P.SIGNAL, fl[0:1], seq, lat, bufs["lat"], bufs["lat"].nbytes, val, bufs["val"],
                  bufs["val"].nbytes, cold, bufs["cold"], bufs["cold"].nbytes, prev[1:2] if prev is not None else None)
        if g:
            prog.emit(P.NM_PULL, *take)
        if slots_all is None:
            self._hot(prog, lat, idx, val, slots, up, down, act, de, gout, k, L, inner, t, Im, tabs=tabs)
        # the CPU's part: the cold pairs on the full stacks in host memory, the threads of node 1 on its
        # copy when the weightkeeper has one (a run of up to 8 rows: those of an expert device's experts
        # on it, its sum added after)
        cc = P.Program()
        upm, dnm = d.m.experts(i, "up"), d.m.experts(i, "down")
        # (not the MTP block's layer: the device holds none of its experts)
        cold_cpu, xd = bufs["cold"], d.xdev if i in cfg.moe and t <= (
            8 if os.environ.get("RECLANK_NM_XGROUP", "1") != "0" else 1) else None
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
        self._mem_total = mem_info()[1]         # (the GPU's memory: sub_rows)
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
        self._xmask = {}
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
        if self.pool is not None:
            self._fit_big()

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
        """The device memory of the programs of the prompt groups that stay (GROUPS, made at the first
        prompt; with the pool, the big ones take theirs from it: is_big): group_bytes of each."""
        return int(sum(self.group_bytes(t) for t in GROUPS if not self.is_big(t)))

    def is_big(self, t):
        """A prompt group's program made from the pool at a prompt and freed after it: BIG_GROUPS's, and GROUPS's of
        BIG_FROM rows up (Super's pool)."""
        return bool(self.nv4 and POOL) and (t in BIG_GROUPS or t >= BIG_FROM)

    def group_bytes(self, t, win=None, sub=None):
        """The device memory of the program of a prompt group of t rows: a part that grows with its
        rows (the step's activations, which the layers share; the routed experts' scratch; the int8 x
        of the products) and one that does not (rc_gpu_load's own buffers). The 3060 measured 538 KB a
        row and 19.5 MB a program for the 74B (824 MB for the five; this gives 552 KB, 853 MB). win: the pairs'
        scratch of a window of that many pairs (win_cap), not of the group's."""
        c = self.cfg
        hid, L, inner, conv = c.hidden, c.latent, c.m_inner, c.conv_dim
        hq, hk = c.heads * c.head_dim, c.kv_heads * c.head_dim
        Emax = max(e for e, _, _ in c.moe.values())
        Im = max(n for _, n, _ in c.moe.values())
        k = max(kk for _, _, kk in c.moe.values())
        wide = max(inner, c.shared_inner, hid)
        act = 4 * (5 * hid + 2 * inner + 2 * conv + c.m_heads + 3 * hq + 2 * hk + Emax + 2 * k + 2 * L +
                   c.shared_inner) + wide * (1 + 4 / 32 + 4 / 16)
        pair = (2 if self.nv4 else 6) * Im + 4 * Im / 32 + 4 * L       # (a pair's act planes, their scales, its outputs)
        moe = pair * k + 10 * L + 4 * L / 32 + 4 * k
        qx = sum(2 * n + 4 * n / 32 for n in {hid, inner, c.shared_inner, L, hq})
        if win:
            # (the pairs' scratch of a window of win pairs, and their positions)
            moe, pairs = moe - pair * k + 4 * k, pair * win
        else:
            pairs = 0.0
        if sub and sub < t:
            # (sub-batches: the residual stream, the rows after norm_f, the routed experts' rows and their int8 x of
            # every row; the rest of sub)
            keep = 4 * (2 * hid + 2 * k + 2 * L)
            return keep * t + moe * t + (act - keep + qx) * sub + pairs + 20e6
        return (act + moe + qx) * t + pairs + 20e6

    def sub_rows(self, t):
        """The rows of a sub-batch of a prompt group of t rows (SUB), or None."""
        s = SUB if SUB is not None else (1024 if self.pool is not None and getattr(self, "_mem_total", 0) < 16e9 else 0)
        return s if s and t > s and t % s == 0 else None

    def win_cap(self, t, c):
        """The pairs of a window of a group of t rows staged in c chunks (WIN), or None (the group's scratch)."""
        if not (WIN and TILES and c > 1):
            return None
        k = max(kk for _E, _n, kk in self.cfg.moe.values())
        return int(-(-WIN_SLACK * t * k / c // 64)) * 64

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
            big = self.pool is not None and self.is_big(t)
            if big:
                # (one big group's program at a time, as _fit_big counts the room: those of other sizes freed first; a
                # prompt in parts of several sizes (2048 then 4096) ran a 12 GB card out of memory beside the stage,
                # and kept while the pool could give the next, the two left the MTP block's part none. The memory from
                # the pool: relax gives it back after the prompt)
                self._free_big([k for k in self.big if isinstance(k, int)])
                gb = self.group_bytes(t, self.win_cap(t, self.stage_c) if self.staged(t) else None, self.sub_rows(t))
                need = gb + BIG_MARGIN_GB * 1e9 - mem_info()[0]
                if need > 0:
                    n = self.pool.shrink(need)
                    print("NemotronGPU: the hot experts %.1f -> %.1f GB for the program of a group of %d" % (
                        (self.pool.nbytes() + n) / 1e9, self.pool.nbytes() / 1e9, t), flush=True)
                before = set(self.mirror.arrays)
                t0 = time.time()
            prog = compile_nemotron_step(self.m, t, em=NmEmit(self, t), sub=self.sub_rows(t) if big else None)
            if getattr(prog, "stage", False):
                # (the copies of the first two MoE layers' cold experts start first)
                st = self.stage_tabs()
                prog.recs[:0] = [(P.FETCH, [prog._enc(st["ranges"][q]), prog._enc(len(st["ranges"][q])), prog._enc(q),
                                            prog._enc(q % 2)] + [prog._enc(v) for v in self.stage_mode()])
                                 for q in range(min(2, len(st["ranges"])))]
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
        return (STAGE and self.pool is not None and self.zc_tables is not None and t >= self.stage_min() and
                t in getattr(self, "stage_ok", ()))

    def _fit_big(self):
        """The big groups the GPU holds, and those it stages. A prompt's room: the free memory and every
        segment of the pool, less what comes later (the programs of GROUPS, made at the first prompt; the
        cache, attached by the first run; the reserve) and BIG_MARGIN_GB. A big group's program takes its
        share of it, a staged group's two stage buffers (2 E experts) the rest; a group without the room
        for those reads the mapped memory. The RTX 3060 (12 GB): 2048 and 4096 mapped (their programs 1.2
        and 2.4 GB, the buffers 3.2)."""
        pool = self.pool
        kv = 0 if self.cache_dev.bufs else self.kv_bytes(self.ctx)
        room = (mem_info()[0] + pool.nbytes() - self.prompt_bytes() - kv -
                (self.reserve_gb + BIG_MARGIN_GB) * 1e9)
        self.mtp_part = None
        if self.cfg.mtp and self.reserve_gb > 0:
            # (the MTP block's prompt parts of more than 256 rows: big programs of their own beside the group's, the
            # reserve holding the smaller ones; a group's per-row memory as theirs. Its largest part that leaves a big
            # group room, mtp_part (spec.NemotronMTPSpec.flush): the 3060 with MTP 1024, the groups of 2048 then fit)
            from reclank.models.nemotron.spec import FLUSH_PARTS
            parts = sorted((p for p in FLUSH_PARTS if p > 256), reverse=True)
            for p in parts:
                r = room - self.group_bytes(p)
                if any(self.group_bytes(t) <= r for t in BIG_GROUPS if t in self.groups) or p == parts[-1]:
                    self.mtp_part, room = p, r
                    break
        # (the stage of a layer in chunks of its experts when two whole layers do not fit beside a staged program, a
        # window of a chunk's pairs at a time in its scratch: the 3060's groups of 4096 rows in 4 of 128 experts; the
        # fewest chunks that stage the most sizes; NV4T only. A big group staged, or unstaged when its whole program fits)
        def own(t, c):
            return self.group_bytes(t, self.win_cap(t, c), self.sub_rows(t)) if self.is_big(t) else 0.0
        cand = [t for t in self.groups if t >= self.stage_min()]
        self.stage_c, self.stage_ok = 1, set()
        for c in ((1, 2, 4, 8, 16) if TILES else (1,)):
            ok = {t for t in cand if pool.E % c == 0 and own(t, c) + 2 * (pool.E // c) * pool.per <= room}
            if len(ok) > len(self.stage_ok):
                self.stage_c, self.stage_ok = c, ok
            if len(ok) == len(cand):
                break
        big = [t for t in self.groups if self.is_big(t) and (t in self.stage_ok or own(t, 1) <= room)]
        if STAGE and self.zc_tables is not None and (len(big) < len([t for t in self.groups if self.is_big(t)]) or
                                                      self.stage_c > 1 or not self.stage_ok.issuperset(cand)):
            print("NemotronGPU: a prompt's room %.1f GB: the groups of %s rows, %s staged%s%s" % (
                room / 1e9, ", ".join(map(str, big)) or "no big", ", ".join(map(str, sorted(self.stage_ok))) or "none",
                " (in %d chunks of a layer)" % self.stage_c if self.stage_c > 1 and self.stage_ok else "",
                "; the MTP block's parts of at most %d" % self.mtp_part if self.mtp_part else ""), flush=True)
        self.groups = tuple(sorted({t for t in GROUPS if not self.is_big(t)} | set(big)))

    def stage_tabs(self):
        """The tables of the staged groups (shared by their programs): for each MoE layer j (the pool's
        rows) the device address of each expert (its slot when held, else its place in stage buffer j % 2;
        stage_fill writes them before each run) and the ranges of its cold experts' copies (2 E rows, a
        range of 0 bytes copies nothing)."""
        st = getattr(self, "_stage", None)
        if st is None:
            R, E, C = len(self.pool.layers), self.pool.E, getattr(self, "stage_c", 1)
            st = self._stage = {"pup": np.zeros((R, E), np.int64), "pdn": np.zeros((R, E), np.int64),
                                "ranges": [np.zeros((2 * E // C, 3), np.int64) for _ in range(R * C)], "bufs": None}
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
        C = getattr(self, "stage_c", 1)
        Ec = E // C
        if st["bufs"] is None:
            need = 2 * Ec * pool.per
            short = need + BIG_MARGIN_GB * 1e9 - mem_info()[0]
            if short > 0:
                n = pool.shrink(short)
                print("NemotronGPU: the hot experts -%.1f GB for the stage buffers (%.1f GB)" % (n / 1e9, pool.nbytes() / 1e9),
                      flush=True)
            st["bufs"] = [Buffer(Ec * pool.per, "program", pinned=True) for _ in range(2)]
        ar = np.arange(E, dtype=np.int64)
        if TILES:
            # (every expert in the buffer of its chunk at its own place, in NV4T: the cold ones from the host's rows,
            # the hot ones from their slots, both through k_nm_tile_pull; chunk q = C j + c in buffer q % 2)
            loc = np.arange(Ec, dtype=np.int64)
            for j, i in enumerate(pool.layers):
                held = pool.slots[i] >= 0
                hu, hd, _du, _dd = pool.host[i]
                su = np.where(held, pool.pup[i], hu + ar * pu)
                sd = np.where(held, pool.pdn[i], hd + ar * pd)
                for c in range(C):
                    q, sl = j * C + c, slice(c * Ec, (c + 1) * Ec)
                    b = st["bufs"][q % 2].ptr
                    up, dn = b + loc * pu, b + Ec * pu + loc * pd
                    st["pup"][j][sl], st["pdn"][j][sl] = up, dn
                    rg = st["ranges"][q]
                    rg[0::2] = np.stack([su[sl], up, np.full(Ec, pu)], 1)
                    rg[1::2] = np.stack([sd[sl], dn, np.full(Ec, pd)], 1)
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
        self._free_big(list(self.big))
        # (back to its segments while the GPU keeps BIG_MARGIN_GB, the reserve (the image encoder's room had gone to
        # the hot experts after a long prompt) and the room of the prompt groups' programs not made yet free: MARGIN_GB
        # went to what was made after the hot set was sized (the step's programs, the cache at the first run), and
        # keeping it all free again left the 3060's pool 0.5 GB short after every prompt; a program that finds no room
        # later makes the pool give some back, _load)
        unmade = sum(self.group_bytes(t) for t in GROUPS if not self.is_big(t) and t not in self.gprogs and t not in self.progs)
        room = int((mem_info()[0] - (BIG_MARGIN_GB + self.reserve_gb) * 1e9 - unmade) // self.pool.seg_bytes)
        k = self.pool.add_segments(max(0, min(self.pool_target - len(self.pool.segs), room)))
        if k:
            self.pool.plan(None)
            print("NemotronGPU: the hot experts back to %.1f GB (%d copies on their way)" % (
                self.pool.nbytes() / 1e9, sum(len(it) for _j, _r, it in self.pool.pending)), flush=True)

    def _free_big(self, keys):
        """The big programs of keys (self.big's) freed, with the device copies of their own arrays."""
        keep = {a.ctypes.data for a in (self.pool._slots, self.pool._pup, self.pool._pdn, self.pool.cnt)}
        if getattr(self, "_stage", None) is not None:
            keep |= {self._stage["pup"].ctypes.data, self._stage["pdn"].ctypes.data}
        for t in keys:
            own = self.big.pop(t)
            _prog, g = self.gprogs.pop(t)
            for b in g.named.values():
                b.free()
            g.named = {}
            g.close()
            for st in own - keep:
                self.mirror.forget(st)

    def _mtp_prog(self, t, head=True):
        """The MTP block on t rows (kq.compile_nemotron_mtp with NmEmit: its attention on K.mtp / V.mtp
        of the cache, its experts on the CPU), with head the head on every row (a prompt's rows need
        none: their keys and values only)."""
        store = self.progs if t <= NV4_HOT_MAX else self.gprogs
        e = store.get(("mtp", t, head))
        if e is None:
            # (the MTP block's prompt parts of more than 256 rows: freed with the big groups after the prompt, and one
            # of them at a time, as _fit_big counts its room (mtp_part))
            big = self.pool is not None and t > 256
            if big:
                self._free_big([k for k in self.big if isinstance(k, tuple) and k[0] == "mtp"])
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
        nsb, sb = getattr(prog, "sub", (0, 0))
        for s in range(nsb):
            # (a sub-batch's first position and its real rows: kq._compile_nemotron_sub)
            kw["pos.%d" % s] = pos + s * sb
            kw["nreal.%d" % s] = max(0, min(sb, t - s * sb))
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

    def xmask(self, i):
        """NM_TAKE's held of layer i: 1 for the expert device's experts (its pairs the CPU's task posts), or None."""
        if self.xdev is None:
            return None
        a = self._xmask.get(i)
        if a is None:
            a = self._xmask[i] = np.zeros(self.cfg.moe[i][0], np.int32)
            a[:] = self.xdev.held[i, :a.size]
        return a

    def take(self, t):
        """NM_TAKE's g for a run of t rows (TAKE): the cold experts of each token the GPU reads across the link."""
        if TAKE is not None:
            return TAKE * t
        g = getattr(self, "_take1", None)
        if g is None:
            k = max(kk for _E, _n, kk in self.cfg.moe.values())
            th = cpu_threads()
            link = getattr(self, "link_gbps", None) or h2d_gbps()
            # (a token's cold experts: those neither the pool nor the expert device holds, about 60% of k here)
            g = self._take1 = int(0.6 * k * link / (link + TAKE_THREAD_GBPS * th)) if th < SMALL_CPU else 0
        return g * t

    def stage_min(self):
        """The least rows of a staged prompt group (STAGE_MIN)."""
        return STAGE_MIN if STAGE_MIN is not None else 1024 if cpu_threads() < SMALL_CPU else 2048

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
        for i, a in self._xmask.items():
            a[:] = self.xdev.held[i, :a.size]
            if self.mirror.owner(a.ctypes.data) is not None:
                self.mirror.buffer_of(a).upload(a)
        return n, nbytes, time.time() - t0
