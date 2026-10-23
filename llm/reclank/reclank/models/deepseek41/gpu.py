"""DeepSeek V4.1 Flash on the GPU, the routed experts in the parts
(DEEPSEEK_V41_PLAN.md Phase 3): the program of a decode step.

- The dense weights go to the GPU once, from the maps of the shards: the
  FP8 matrices as they are (GP_DS_FP8_LINEAR; wo_a with its 8 groups), the
  bfloat16 matrices as BF12 (GP_KQ_LINEAR), the head too. No host copy
  stays: each operand is a placeholder (an untouched anonymous map of its
  size) whose device copy the Mirror holds.
- The step: the records of kernels/dsv41/step.cu for the hyper-connections,
  the norms, the rope, the window, the compressor, the indexer and its top
  k, the attention with the sinks, the router, the shared expert, engram.
  The forms are those of the CPU model in near mode (model.py).
- The routed experts: all cold for now (the hot experts on the GPU come
  next): GP_SIGNAL gives the normed input, the choice and the weights to
  pinned memory, GP_CPU_TASK runs GP_KQ_MOE_PART on the rows of each node
  in the weightkeeper's segments (flag 128: the clamped SwiGLU and the weight of
  the pair before the down rows), GP_AWAIT brings their sum back.
- The state (the window rings, the compressed caches, the index keys, the
  compressor's open group) is in arrays of this object with device copies;
  load_state copies a DSV41CPU's state in (a check from the same state).
- The host gives the inputs of a step: the embedding in the 4 streams and
  the engram rows of the token (the hashes need only the ids).

    m = DSV41CPU(path)                 # the weights, the hashes, the tables
    g = DSV41GPU(m)
    m.forward(ids, 0); g.load_state(m)
    logits = g.step(token, len(ids))
"""
from __future__ import annotations

import mmap
import os
import sys
import time

import numpy as np

from reclank.backends.cpu import cops
from reclank.backends.cuda.gpumm import Buffer, pinned
from reclank.backends.cuda.runtime import GPUProgram, Mirror
from reclank.formats import dsv41_st as D
from reclank.formats.gguf import BF12

Q8_R = 100      # the rows of Q8_0 for the GPU (backends/cuda/csrc; gpu35.Q8_R)
from reclank.ir import program as P
from reclank.runtime import moe_parts


# the most tokens of a group of a prompt (prefill): on the real prompts
# (bounded replay, two halves, warm, groups of multiples of GROUP_STEP) of
# 1414 tokens 1024: 4.4 s, 2048 (one group of 1408): 4.0-4.1 s with 12 or 13
# GB of hot experts; of 4414 tokens 1024: 12.9 s (13 GB), 2048: 12.3 s (11
# GB), 4096: 12.1-12.5 s (9 GB: 1.2 GB more scratch for each doubling); the
# logits the same bits for all
GROUP = int(os.environ.get("RECLANK_DS_GROUP", "2048"))
# the largest verify group of DSpark (the last token and up to 5 drafts): its
# programs give the logits of every row and keep what restore undoes; their
# rows have the bits of a step (the attention splits its keys up to here)
VERIFY_MAX = 8
# RECLANK_DS_SIGNAL_CHAIN=0: a step's signals do not wait for the task before (signal_wait; a test:
# with RECLANK_CPU_TASK_DELAY_US the inputs of a task with no cold expert then change under it)
SIGNAL_CHAIN = os.environ.get("RECLANK_DS_SIGNAL_CHAIN", "1") != "0"
# spec (DSV41GPU's, default on; the engine turns it off for --lanes 2, whose two lanes' steps run at
# once: the guess is one per process): a step's CPU, idle while the GPU runs a layer's attention, guesses
# that layer's cold experts (its router on the layer before's FFN input) and reads their rows into its
# caches until the layer's signal comes (moe_part.c flag 4096), a task right after the layer before's; the
# real task then finds them there (cops.ds_spec_stats: the hits, 63% of the cold experts). The bits do not
# change (the task takes the real choice). A step 34.63 -> 33.62 ms (3 pairs of runs of 400 steps).
# RECLANK_DS_SPEC=0/1 over all; RECLANK_DS_SPEC_N: the most experts a layer (6); RECLANK_DS_SPEC_DN=0:
# the gate and up rows alone (33.79 ms)
SPEC = None if os.environ.get("RECLANK_DS_SPEC") is None else os.environ["RECLANK_DS_SPEC"] != "0"
# the decoder's bounded replay of a long prompt (DSV41GPU replay; RECLANK_DS_REPLAY=0: every layer on
# every token): KL against the full forward 3.6e-3 (1000 tokens), 6.2e-3 (2300), 3.0e-3 (5018), 1.1e-2
# (prompt3), the same argmax; a prompt in parts has the bits of the prompt whole
REPLAY = os.environ.get("RECLANK_DS_REPLAY", "1") != "0"
# make_room: a program that finds the GPU full first frees the others (made again when run), but not
# twice in ROOM_S seconds: then the working set of programs does not fit, and the hot experts give a
# slot of each layer (RECLANK_DS_ROOM_S)
ROOM_S = float(os.environ.get("RECLANK_DS_ROOM_S", "60"))
# relax: the hot experts grow back (up to their slots at the load) while the GPU keeps this much free
# (the programs of a short sequence, the image encoder; RECLANK_DS_GROW_MARGIN_GB)
GROW_MARGIN_GB = float(os.environ.get("RECLANK_DS_GROW_MARGIN_GB", "4"))
# the GPU's share of the cold experts of a prompt group (docs/DS_PREFILL_PLAN.md Phase 1): after a
# layer's router each half of the group takes the STAGE cold experts of the most pairs (at least
# STAGE_MINP each; RECLANK_DS_STAGE, 0: off; RECLANK_DS_STAGE_MINP): the GPU copies them from the
# weightkeeper's registered segments into its stage (GP_DS_STAGE_PULL, STAGE_BLOCKS blocks reading the
# host memory, beside the stream's kernels) while the CPU runs the other cold experts, and runs their
# pairs before the CPU's task ends (GP_DS_STAGE_JOIN). Half B takes A's experts of the layer too (still
# in the stage) and picks its own after them. The real prompt of 4414 tokens: 11.37 -> 9.76 s (8 of 32
# pairs, 4 blocks: a slow copy leaves the kernels beside it their speed; 6 to 12 experts within 1%, 16
# and more slower: the GPU the bound)
STAGE = int(os.environ.get("RECLANK_DS_STAGE", "8"))
STAGE_MINP = int(os.environ.get("RECLANK_DS_STAGE_MINP", "32"))
STAGE_BLOCKS = int(os.environ.get("RECLANK_DS_STAGE_BLOCKS", "4"))
# a group's halves: the pull of a half in pieces of STAGE_CHUNK experts, each after one of the other
# half's records (the first after its read of the CPU's sums, then its attention's), the rest before the
# join (RECLANK_DS_STAGE_CHUNK, 0: the whole pull after the signal)
STAGE_CHUNK = int(os.environ.get("RECLANK_DS_STAGE_CHUNK", "0"))


def gpu_oom(e):
    """Whether an exception is the GPU out of memory (a cudaMalloc of the mirror, a graph's instance)."""
    return isinstance(e, MemoryError) or (isinstance(e, RuntimeError) and "out of memory" in str(e))


# the groups of a prompt: the largest multiple of GROUP_STEP tokens up to the
# group, below it the powers of two (each group reads every cold expert
# again: the prompt of 1414 tokens in 1408 + 4 + 2 in place of 1024 + 256 +
# 128 + 4 + 2); RECLANK_DS_GROUP_STEP 0: the powers of two alone
GROUP_STEP = int(os.environ.get("RECLANK_DS_GROUP_STEP", "128"))
# the GPU memory of the hot experts by default (RECLANK_DS_HOT_GB): 12 GB
# leaves room for 1M of cache, groups of 2048 and DSpark (13: no room for the
# graphs of DSpark); prompt3 4.06 s with 12, 4.03 s with 13
HOT_GB = float(os.environ.get("RECLANK_DS_HOT_GB", "12"))
# with DSpark: 2 GB less (its weights, the seed and draft programs, the verify
# groups and their graphs; with 12 GB a prompt of 5K tokens left no room for
# a graph)
HOT_GB_DSPARK = HOT_GB - 2
# the positions the programs serve (cap) at first: the index scores of a
# program take t x cap floats and the rope tables on the GPU cap rows; cap
# doubles (the programs made again) when a run reaches past it, up to the
# model's max_len (the caches have max_len rows from the start: 0.9 KB a
# position, 0.93 GB for 1M)
CAP = int(os.environ.get("RECLANK_DS_CAP", "8192"))
# the most memory of the index scores of a program (t x cap floats, the
# candidates' blocks): past it (a long sequence, a large group) they go in
# chunks of entries (score_chunk; a group of 1024 at cap 128K: chunks of 64K)
SCORE_MB = int(os.environ.get("RECLANK_DS_SCORE_MB", "512"))


def check_omp_threads():
    """DeepSeek V4.1 on the GPU runs with OMP_NUM_THREADS=1, and nothing else:
    the CPU parts of the experts have their own teams, pinned (GP_RUN_PARTS);
    an OpenMP team of the main thread (36 threads under OMP_NUM_THREADS=36: a
    small conversion each step) spins on the cores of the parts after each
    region. Set it before reclank loads (the scripts of DeepSeek do); OpenMP
    reads it when it starts."""
    import ctypes
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise RuntimeError("DeepSeek V4.1 on the GPU needs OMP_NUM_THREADS=1 (it is %r)" %
                           os.environ.get("OMP_NUM_THREADS"))
    from reclank.backends.cpu import cops  # noqa: F401  (the library, and OpenMP with it)
    n = ctypes.CDLL("libgomp.so.1").omp_get_max_threads()
    if n != 1:
        raise RuntimeError("DeepSeek V4.1 on the GPU needs OMP_NUM_THREADS=1: OpenMP started with %d threads "
                           "(set it before reclank loads)" % n)


def _placeholder(nbytes):
    """An array of nbytes in an untouched anonymous map: the host side of a
    device copy (its address is the key of the Mirror; no RAM is used)."""
    mm = mmap.mmap(-1, max(int(nbytes), 1))
    return np.frombuffer(mm, np.uint8, count=int(nbytes))


class DSV41GPU:
    """See the module text."""

    def __init__(self, m, graph=True, keep_layers=True, hot_gb=HOT_GB, counts=None, hot_dyn=True, cache="min",
                 group=None, rq8=False, x8_prefill=False, replay=None, ab=True, spec=True):
        """hot_gb: the GPU memory of the hot experts (whole experts, 18.8 MB
        each, the same count for each layer; default HOT_GB, 12); counts (layers x experts, or
        None): the first hot experts of each layer are the most counted, else
        the first ones; hot_dyn: the slots follow the choices (DSHot), else
        they stay. cache: the form of the window rings, the compressed rows and
        the index keys: "min" (the default: the smallest that holds their
        values exactly: the window rows in FP8, E4M3 and an E8M0 scale for
        each 32; the compressed rows in FP4, E2M1 and an E4M3 scale for each
        16, the form of the model; the index keys in FP4 with an E8M0 scale
        for each 32: about 0.9 KB a position), or "bf16" (3.2 KB a position);
        both give the logits of float32 to the bit. group: the
        tokens of a group of a prompt (default RECLANK_DS_GROUP, 2048). rq8:
        the bfloat16 matrices (the head, the routers, the compressors, the
        indexer's) in RQ8_0 (Q8_0 of the rows turned in each 32, x turned the
        same) in place of BF12: 8.5 bits a value in place of 12.25, not exact
        (an option; the FP8 matrices stay as they are). replay (default
        REPLAY, on): a prompt of
        more than the window runs the decoder (layers 20..39) on its last
        window of tokens only (DeepSeek's Decoder SWA Bounded Replay, V4.1
        3.2.2: the way DeepSeek serves the model, simulated in its
        post-training; not the bits of the full forward of model.py); False:
        every layer on every token. ab: a group of a prompt in two halves
        whose records interleave (the CPU's experts of one while the GPU
        runs the other; the same bits). spec: a step's CPU reads the cold
        experts it guesses for each layer while the GPU runs its attention
        (SPEC; one lane alone)."""
        check_omp_threads()
        self._pin_to_gpu_node()
        self.m = m
        self.cform = {"bf16": 1, "min": 2}[cache]
        self.group = int(group or GROUP)
        self.rq8 = bool(rq8)
        # x8_prefill: the inputs of the routed experts in the groups of a prompt
        # rounded to 8 bits (a measure of 8-bit activations; the steps keep 16)
        self.x8_prefill = bool(x8_prefill)
        self.spec = bool(spec) if SPEC is None else SPEC
        # the decoder's first layer: the source of its global KV (CED, V4.1 2.2)
        self.dec0 = m.L // 2
        self.ab = bool(ab)
        # RECLANK_DS_HC_STEP=0: the posts, mixes and pres as three records each, the CPU's rows of its pairs
        self.hc_step = int(os.environ.get("RECLANK_DS_HC_STEP", "1"))
        # RECLANK_DS_VERIFY_MERGE=0: each half of a verify group runs its own hot and shared experts
        self.verify_merge = os.environ.get("RECLANK_DS_VERIFY_MERGE", "1") != "0"
        self.replay = bool(REPLAY if replay is None else replay) and max(m.kv_sources) == self.dec0 == m.cand_src
        self.w = m.w
        if self.w.z is None or self.w.z.index["layout"] == "dsv41-expert-planes-v1":
            raise ValueError("the GPU step needs a weightkeeper with the KQ_MXX layout (v2)")
        self.mirror = Mirror()
        self.keep_layers = keep_layers
        self.cap = min(m.max_len, CAP)
        # the lane whose state is in place (lanes.py: 0, the model's own; a lane's view swaps
        # LANE_ATTRS) and the most positions of that lane's caches
        self.lane_id = 0
        self.max_pos = m.max_len
        self.ends = {}              # lane -> the end of its last fit (relax keeps the cap of each)
        self.on_grow = []           # (DSpark) told when the programs go: cap grew
        self._state()
        self.parts = self._parts()
        self._hot_init(hot_gb, counts)
        self.graph = graph
        self.part_bufs = {}
        self._w = {}
        self._tables()
        self.runs = {}
        self.scratch = {}
        # bounded replay: the encoder's outputs of the last window of tokens of
        # a prompt (the streams, the raw mixes of its last FFN), the input of
        # the decoder
        self.rep_h = np.zeros((m.win, m.hc, m.dim), np.float32)
        # the keys of each query of a step or a verify group decoded once a
        # layer for the attention's blocks (k_ds_attn_keys), float32: the
        # programs of t <= VERIFY_MAX share it (one runs at a time; the rows
        # of each query its own)
        self.akeys = np.zeros((VERIFY_MAX, m.win + m.ix_topk, m.head_dim), np.float32)
        self.rep_raw = np.zeros((m.win, 24), np.float32)
        self._embed_table()
        self.hot = DSHot(self) if self.n_slots > 0 and hot_dyn else None
        if self.hot is not None:
            self.hot.register()
        self._stage_init()
        self.run(1)

    def _pin_to_gpu_node(self):
        """The main thread on the CPUs of the GPU's node (of those allowed),
        before anything is made: the pinned buffers (the rows of the parts,
        the embedding table, the flags) and the memory it touches first are
        then there, not across the UPI link. (The first run of the parts pins
        it to the first CPU of the team of node 0.)"""
        from reclank.runtime import numa
        try:
            import ctypes
            from reclank.backends.cuda.runtime import lib
            bus = ctypes.create_string_buffer(32)
            lib().rc_gpu_pci_bus_id(bus, 32)
            with open("/sys/bus/pci/devices/%s/numa_node" % bus.value.decode().lower()) as f:
                node = max(int(f.read()), 0)
            cpus = set(numa.node_cpus(node)) & os.sched_getaffinity(0)
            if cpus:
                os.sched_setaffinity(0, cpus)
        except (OSError, ValueError, AttributeError):
            pass

    # ---- the weights
    def _put(self, host):
        """A placeholder with the device copy of host (any contiguous array)."""
        host = np.ascontiguousarray(host)
        ph = _placeholder(host.nbytes)
        self.mirror.add(ph)
        b = Buffer(host.nbytes, "program", pinned=True)
        b.upload(host.view(np.uint8).reshape(-1))
        self.mirror.bufs[ph.ctypes.data] = b
        return ph

    def fp8(self, name):
        if ("fp8", name) not in self._w:
            codes, _dt, shape = self.w.src.raw(name, copy=False)
            sc, _sdt, sshape = self.w.src.raw(name[:-len("weight")] + "scale")
            self._w["fp8", name] = (self._put(codes), np.ascontiguousarray(sc), shape[0], shape[1])
        return self._w["fp8", name]

    def bf12(self, name):
        if ("bf12", name) not in self._w:
            b = self.w.src.bf16(name)
            rows, _rep = cops.kq_bf16_to_bf12(b, b.shape[1])
            self._w["bf12", name] = (self._put(rows), b.shape[0], b.shape[1])
        return self._w["bf12", name]

    def rq8w(self, name):
        """A bfloat16 matrix in RQ8_0 for the GPU: the Q8_0 blocks of its rows
        turned in each 32 (cops.tq6_rotate, the rotation of RQ8_0), as the rows
        of the GPU's Q8_R (the int8 values, then the float16 scales, then
        zeros to a multiple of 16 bytes)."""
        if ("rq8", name) not in self._w:
            b = self.w.src.bf16(name)
            rows, cols = b.shape
            f = (b.astype(np.uint32) << 16).view(np.float32)
            q = cops.kq_to_q8_0(cops.tq6_rotate(f), cols).reshape(rows, cols // 32, 34)
            pad = -(cols // 32 * 34) % 16
            a = np.concatenate([q[:, :, 2:].reshape(rows, cols), q[:, :, :2].reshape(rows, cols // 16),
                                np.zeros((rows, pad), np.uint8)], axis=1)
            self._w["rq8", name] = (self._put(a), rows, cols)
        return self._w["rq8", name]

    def emit_dense16(self, prog, xb, name, out):
        """x times a matrix of bfloat16 origin: BF12 (exact for its values), or
        with rq8 RQ8_0 on x turned in each 32 (GP_TQ_ROT from x into a copy)."""
        t = xb.shape[0]
        if not self.rq8:
            w, rows, cols = self.bf12(name)
            prog.emit(P.KQ_LINEAR, None, None, None, xb, w, BF12, rows, cols, t, out)
            return
        w, rows, cols = self.rq8w(name)
        xr = np.zeros((t, cols), np.float32)
        prog.emit(P.TQ_ROT, xr, t * cols // 32, 0, xb)
        prog.emit(P.KQ_LINEAR, None, None, None, xr, w, Q8_R, rows, cols, t, out)

    def vec(self, name, times=None):
        """A float32 vector (times another: the product, as the CPU model)."""
        if ("vec", name, times) not in self._w:
            v = self.w.vec(name)
            if times is not None:
                v = v * self.w.vec(times)
            self._w["vec", name, times] = np.ascontiguousarray(v, np.float32)
        return self._w["vec", name, times]

    def hcp(self, name):
        if ("hc", name) not in self._w:
            self._w["hc", name] = (self._put(self.vec(name + "_fn")), self.vec(name + "_scale"),
                                   self.vec(name + "_base"))
        return self._w["hc", name]

    # ---- the positions the programs serve
    def _tables(self):
        """The rope tables (the window's, the compressed rows': cos, sin) of
        the positions below cap: copies, the device copies of that size."""
        self.tables = [[np.array(a[:self.cap], np.float32) for a in fr] for fr in (self.m.freq_win, self.m.freq_cmp)]

    def fit(self, end):
        """The programs serve the positions below end (and the drafts of
        DSpark past it): else cap doubles until they do, and the programs, their
        scratch and the tables go (made again when run)."""
        m = self.m
        if end > self.max_pos:
            raise ValueError("position %d is past the %d positions of %s" % (
                end, self.max_pos, "lane %d" % self.lane_id if self.lane_id else "max_len"))
        self.ends[self.lane_id] = end
        need = min(end + 2 * VERIFY_MAX, m.max_len)
        if need <= self.cap:
            return
        cap = self.cap
        while cap < need:
            cap *= 2
        self.cap = min(cap, m.max_len)
        self.free_programs(tables=True)
        self._tables()

    def free_programs(self, tables=False):
        """Free the GPU memory of the programs (DSpark's too, on_grow) and of their shared scratch:
        made again when run (fit, as cap grows; the image encoder, when it needs the room: the
        programs of a long session take more as the context grows)."""
        self.close()
        for f in self.on_grow:
            f()
        for a in list(self.scratch.values()) + ([a for fr in self.tables for a in fr] if tables else []):
            self.mirror.forget(a.ctypes.data)
        self.scratch = {}

    def score_chunk(self, t, sw):
        """The entries of a chunk of the index scores of a program of t rows
        whose sources have up to sw entries: all of them when their scores
        and the blocks' max (t (sw + sw / bs) float32) fit in SCORE_MB, else
        the most (a power of two) that do; RECLANK_DS_SCORE_CHUNK: at most
        that (a check of the chunks)."""
        bs = self.m.cand_bs
        per = 4 * t * (1 + 1 / bs)
        force = int(os.environ.get("RECLANK_DS_SCORE_CHUNK", "0"))
        if per * sw <= SCORE_MB << 20 and not 0 < force < sw:
            return sw
        c = 1 << max(int((SCORE_MB << 20) / per).bit_length() - 1, 0)
        if 0 < force < sw:
            c = min(c, 1 << (force.bit_length() - 1))
        return min(max(c, 64 * bs), sw)

    # ---- the state
    def cache_rows(self, n, hd, kind):
        """An empty array of n rows of hd values of a cache: kind "w" (the
        window rows), "c" (the compressed rows), "k" (the index keys)."""
        if self.cform == 1:
            return np.zeros((n, hd), np.uint16)
        width = {"w": hd + hd // 32, "c": hd // 2 + hd // 16, "k": hd // 2 + hd // 32}[kind]
        return np.zeros((n, width), np.uint8)

    def _state(self):
        m = self.m
        hd, L = m.head_dim, m.L
        self.window = [self.cache_rows(m.win, hd, "w") for _ in range(L)]
        self.ckv = {i: self.cache_rows(a.shape[0], hd, "c") for i, a in m.ckv.items()}
        self.kcache = {i: self.cache_rows(a.shape[0], a.shape[1], "k") for i, a in m.kcache.items()}
        self.kstate = {i: np.zeros_like(a) for i, a in m.kstate.items()}
        self.sstate = {i: np.full_like(a, -np.inf) for i, a in m.sstate.items()}

    def state_arrays(self):
        return (self.window + list(self.ckv.values()) + list(self.kcache.values()) +
                list(self.kstate.values()) + list(self.sstate.values()))

    # the attributes of a sequence (lanes.py, DSLane.view): its caches and compressor states, the
    # keys of its steps' attention (shared by the programs of t <= VERIFY_MAX, one at a time), the
    # scratch of its CPU tasks, its last run, the step left for HotCache, the lane, its positions
    LANE_ATTRS = ("window", "ckv", "kcache", "kstate", "sstate", "akeys", "part_bufs", "last", "_hot_seen",
                  "lane_id", "max_pos")

    def lane_state(self, lane_id, max_len=None):
        """A new sequence's LANE_ATTRS (empty caches; the compressed rows and index keys for max_len
        positions, default the model's): the state of a lane."""
        m = self.m
        hd, L = m.head_dim, m.L
        n = min(int(max_len or m.max_len), m.max_len)

        def rows(a):
            return -(-a.shape[0] * n // m.max_len)
        return dict(
            window=[self.cache_rows(m.win, hd, "w") for _ in range(L)],
            ckv={i: self.cache_rows(rows(a), hd, "c") for i, a in m.ckv.items()},
            kcache={i: self.cache_rows(rows(a), a.shape[1], "k") for i, a in m.kcache.items()},
            kstate={i: np.zeros_like(a) for i, a in m.kstate.items()},
            sstate={i: np.full_like(a, -np.inf) for i, a in m.sstate.items()},
            akeys=np.zeros_like(self.akeys), part_bufs={}, last=None, _hot_seen=None,
            lane_id=int(lane_id), max_pos=n)

    def to_cache(self, a, kind):
        """float32 values (rows) in the form of a cache of kind "w", "c", "k",
        exact for the values the model puts there."""
        a = np.ascontiguousarray(a, np.float32)
        if self.cform == 1:
            return (a.view(np.uint32) >> 16).astype(np.uint16)
        return cache_encode(a, kind)

    def cache_f32(self, a, kind):
        """The float32 values of rows of a cache of kind "w", "c", "k"."""
        if a.dtype == np.uint16:
            return (a.astype(np.uint32) << 16).view(np.float32)
        return cache_decode(a, kind)

    def load_state(self, m):
        """The state of a DSV41CPU (after its prompt) into the device copies."""
        for dst, src in zip(self.window, m.window):
            dst[:] = self.to_cache(src, "w")
        for d, s, kind in ((self.ckv, m.ckv, "c"), (self.kcache, m.kcache, "k")):
            for i in d:
                d[i][:] = self.to_cache(s[i], kind)
        for d, s in ((self.kstate, m.kstate), (self.sstate, m.sstate)):
            for i in d:
                d[i][:] = s[i]
        for a in self.state_arrays():
            self.mirror.buffer_of(a).upload(a)

    def reset(self):
        """An empty sequence: the state as at the start, on the GPU (the host
        copies of the compressed rows and the index keys are those of the last
        load_state or save_state: 0.9 GB at max_len 1M, not touched)."""
        small = self.window + list(self.kstate.values()) + list(self.sstate.values())
        for a in self.state_arrays():
            self.mirror.buffer_of(a).zero()
        for a in small:
            a[:] = 0
        for a in self.sstate.values():
            a[:] = -np.inf
            self.mirror.buffer_of(a).upload(a)

    def save_state(self):
        for a in self.state_arrays():
            self.mirror.buffer_of(a).download(a)

    # ---- the parts on the weightkeeper's segments
    def spec_router(self, i):
        """Layer i's router for spec (the prefetch of the cold experts): its gate in bfloat16
        (rounded) and the bias of the selection, on the host (kept)."""
        sw = self.__dict__.setdefault("_spec_w", {})
        if i not in sw:
            pre = "layers.%d.ffn." % i
            w = np.ascontiguousarray(self.m.w.dense_f32(pre + "gate.weight"), np.float32).view(np.uint32)
            bf = ((w + 0x7fff + ((w >> 16) & 1)) >> 16).astype(np.uint16)
            sw[i] = (bf, np.ascontiguousarray(self.m.w.vec(pre + "gate.bias"), np.float32))
        return sw[i]

    def _parts(self):
        z = self.w.z
        ix = z.index
        cpus = moe_parts.part_cpus()
        if ix["nodes"] != moe_parts.nodes()[:len(cpus)] or len(ix["segments"]) != len(cpus):
            raise ValueError("the weightkeeper's nodes %s are not those of the parts %s" % (ix["nodes"], moe_parts.nodes()))
        desc = {}
        for i in ix["layers"]:
            k = ix["layers"].index(i)
            ds = []
            for p in range(len(cpus)):
                base = z.base[p] + k * ix["layer_stride"][p]
                d = []
                for _w, off, _r0, _nr, _cols, _n in ix["mats"][p]:
                    d += [base + off, cops.KQ_MXX, ix["stride"][p], 0]
                ds.append(d)
            desc[i] = ds
        return moe_parts.ExpertParts.from_desc(desc, ix["gu"], ix["dn"], ix["E"], ix["hidden"], ix["inner"], cpus)

    # ---- the inputs on the GPU
    def _embed_table(self, chunk=8192):
        """The embedding as BF12 rows in pinned host memory (about 1 GB of RAM,
        no GPU memory): GP_DS_EMBED reads the rows of the tokens through the
        map and decodes them on the GPU (the values of the CPU model's
        embed_rows)."""
        b = self.w.src.bf16("embed.weight")
        V, dim = b.shape
        rb = cops.bf12_row_bytes(dim)
        self.embed_tab = pinned((V, rb), np.uint8)
        for r0 in range(0, V, chunk):
            rows, _rep = cops.kq_bf16_to_bf12(np.ascontiguousarray(b[r0:r0 + chunk]), dim)
            self.embed_tab[r0:r0 + rows.shape[0]] = rows

    def engram_codes(self, layer, hashes):
        """The bytes of the rows of the hashes (codes [n, 256], scales [n, 8])
        from the shards: a gather, no arithmetic (GP_DS_ENGRAM_ROWS decodes
        them on the GPU)."""
        pre = "layers.%d.engram." % layer
        b, _dt, shape = self.w.src.raw(pre + "embed.weight", copy=False)
        s, _sdt, sshape = self.w.src.raw(pre + "embed.scale", copy=False)
        flat = np.asarray(hashes).reshape(-1)
        return np.asarray(b).reshape(shape)[flat], np.asarray(s).reshape(sshape)[flat]

    # ---- the hot experts
    def _hot_init(self, hot_gb, counts):
        ix = self.w.z.index
        H, I, E, L = ix["hidden"], ix["inner"], ix["E"], self.m.L
        self.off3 = I * (H // 32) * 17
        self.off2 = 2 * self.off3
        self.per = self.off2 + H * (I // 32) * 17
        n = int(min(E, max(0, hot_gb) * 1e9 // (L * self.per)))
        self.n_slots = self.n_max = n
        self.maps = {i: np.full(E, -1, np.int32) for i in range(L)}
        self.slot_expert = {i: np.full(max(n, 1), -1, np.int64) for i in range(L)}
        self.stores = {}
        for i in range(L):
            ph = _placeholder(max(n, 1) * self.per)
            self.mirror.add(ph)
            self.mirror.bufs[ph.ctypes.data] = Buffer(ph.nbytes, "program", pinned=True)
            self.stores[i] = ph
            first = (np.argsort(-np.asarray(counts[i]), kind="stable")[:n] if counts is not None
                     else np.arange(n))
            for slot, e in enumerate(first):
                self.load_expert(i, slot, int(e), upload_map=False)

    def expert_bytes(self, i, e):
        """Expert e of layer i whole (w1, w3, w2 in KQ_MXX): the rows of the
        parts one after the other (the groups of 16 rows in order)."""
        z = self.w.z
        out = np.empty(self.per, np.uint8)
        at = 0
        parts = range(len(z.index["segments"]))
        for name in ("w1", "w3", "w2"):
            for p in parts:
                b = z.mats(i, p)[name][e]
                out[at:at + b.size] = b
                at += b.size
        assert at == self.per
        return out

    def load_expert(self, i, slot, e, upload_map=True):
        """Expert e of layer i into slot (the expert there leaves)."""
        from reclank.backends.cuda.runtime import _check, lib
        old = int(self.slot_expert[i][slot])
        if old >= 0:
            self.maps[i][old] = -1
        host = self.expert_bytes(i, e)
        dst = self.mirror.buffer_of(self.stores[i]).ptr + slot * self.per
        _check(lib().rc_gpu_h2d(dst, host.ctypes.data, self.per))
        self.slot_expert[i][slot] = e
        self.maps[i][e] = slot
        if upload_map:
            self.mirror.buffer_of(self.maps[i]).upload(self.maps[i])

    def expert_ranges(self, i, e, slot):
        """The (host, device, bytes) ranges that copy expert e of layer i from
        the weightkeeper's segments into slot (the order of expert_bytes)."""
        z = self.w.z
        ix = z.index
        k = ix["layers"].index(i)
        dst = self.mirror.buffer_of(self.stores[i]).ptr + slot * self.per
        out = []
        for name in ("w1", "w3", "w2"):
            for p in range(len(ix["segments"])):
                for w, off, _r0, _nr, _cols, n in ix["mats"][p]:
                    if w == name:
                        out += [z.base[p] + k * ix["layer_stride"][p] + e * ix["stride"][p] + off, dst, n]
                        dst += n
        return out

    def _stage_init(self):
        """The table of GP_DS_STAGE_PULL: for each layer the ranges of an expert in the order of a slot
        (expert_ranges), each (the device address of expert 0's bytes, the bytes from an expert to the
        next, its bytes, its offset in a slot), stage_tab [L, ranges, 4]; stage_n the experts a half
        stages (0: no stage: STAGE 0, or the segments not registered or not aligned)."""
        from reclank.backends.cuda.runtime import lib
        self.stage_n, self.stage_tab = 0, None
        if STAGE <= 0 or self.hot is None or not self.hot.registered:
            return
        z = self.w.z
        ix = z.index
        dev = [lib().rc_gpu_dev_addr(b) for b in z.base]
        if not all(dev):
            print("[dsv41] no stage of cold experts: the weightkeeper's segments have no device address",
                  file=sys.stderr)
            return
        tab = []
        for i in range(self.m.L):
            k, at, rows = ix["layers"].index(i), 0, []
            for name in ("w1", "w3", "w2"):
                for p in range(len(ix["segments"])):
                    for w, off, _r0, _nr, _cols, n in ix["mats"][p]:
                        if w == name:
                            rows.append([dev[p] + k * ix["layer_stride"][p] + off, ix["stride"][p], n, at])
                            at += n
            assert at == self.per
            tab.append(rows)
        tab = np.array(tab, np.int64)
        if (tab[:, :, :3] % 16).any() or (tab[:, :, 3] % 16).any() or self.per % 16:
            print("[dsv41] no stage of cold experts: the expert ranges are not 16-byte aligned", file=sys.stderr)
            return
        self.stage_n, self.stage_tab = min(STAGE, 128), tab

    def stage_buf(self, sets):
        """The stage of the groups of a prompt (their programs run one at a time: one for them all): sets
        of stage_n slots (_compile: stage_s0); a placeholder with its device copy, freed with the programs
        (free_programs)."""
        key = ("stage", sets * self.stage_n)
        b = self.scratch.get(key)
        if b is None:
            b = _placeholder(sets * self.stage_n * self.per)
            self.mirror.add(b)
            self.mirror.bufs[b.ctypes.data] = Buffer(b.nbytes, "program", pinned=True)
            self.scratch[key] = b
        return b

    def hits(self):
        """The share of the experts of the last run that the GPU held."""
        ids = self.download(self.last.ids)
        return float(np.mean([self.maps[i][ids[i]] >= 0 for i in range(self.m.L)]))

    # ---- the program
    def _compile(self, t, stage="full", vis=False):
        """The program of t tokens at the slot pos (a step: t = 1; a group of a
        prompt: t > 1, the logits of its last token). Its inputs and outputs:
        a _Run. stage "full": all the layers; for the bounded replay of a
        prompt, "enc": the encoder (the layers before dec0, then the
        compressed rows and index keys of dec0: the decoder's global KV) and
        its outputs of the positions from the slot rbase into rep_h, rep_raw;
        "dec" (t = the window, at pos = rbase): the decoder from rep_h, its
        window from pos on, the logits. vis: the tokens may be of an image span
        (the inputs vmask, and vrows: the image's rows in place of their
        embedding; the router's bias_vl for them)."""
        m, prog = self.m, P.Program()
        L, dim, hc, hd, rd = m.L, m.dim, m.hc, m.head_dim, m.rd
        H, win, k = m.heads, m.win, m.topk
        eps, hc_eps, iters = m.eps, m.hc_eps, m.sink_iters
        I = self.w.src.inner
        # the arrays of a group of a prompt are the scratch that its programs
        # share (_scratch); the outputs that the host reads stay its own (zo)
        share, counts = t > VERIFY_MAX, {}

        own = []                    # the arrays of this program alone (close forgets them)

        def z(*sh, dtype=np.float32, pin=False):
            if share:
                return self._scratch(t, counts, sh, dtype, pin, most=win if stage == "dec" else None)
            own.append(pinned(sh, dtype, free=True) if pin else np.zeros(sh, dtype))
            return own[-1]

        def zo(*sh):
            own.append(np.zeros(sh, np.float32))
            return own[-1]
        pos, seq = prog.slot("pos"), prog.slot("seq")
        cw, cc = self.tables
        io = _Run()
        io.prog = prog
        io.t = t
        io.own = own
        E = self.dec0
        io.lay = lay = {"full": range(L), "enc": range(E), "dec": range(E, L)}[stage]
        if stage == "dec":
            assert t == win
            io.h = h = self.rep_h
        else:
            io.h = h = z(t, hc, dim)
        io.tok = z(t, dtype=np.int32)
        io.vis = vis
        io.vrows = z(t, dim) if vis and stage != "dec" else None
        io.vmask = z(t, dtype=np.int32) if vis else None
        io.eg = {i: z(t, 24 * 256) for i in m.engram_layers}
        io.egc = {i: z(t * 24, 256, dtype=np.uint8) for i in m.engram_layers}
        io.egs = {i: z(t * 24, 8, dtype=np.uint8) for i in m.engram_layers}
        io.layers = zo(L, t, hc, dim) if self.keep_layers and t <= 64 else None
        io.ids = np.zeros((L, t * k), np.int32)
        own.append(io.ids)
        io.vals = zo(L, t * k)
        # a verify group of DSpark (t <= VERIFY_MAX): the logits of every row, and
        # the backups that undo the rows of the refused drafts (restore)
        verify = t <= VERIFY_MAX
        # (pinned: the copy that ends each step does not stage through pageable memory: 0.22 -> 0.11 ms;
        # freed with the program: make_room makes them again at every request of a long session)
        io.logits = pinned((t if verify else 1, m.w.src.cfg["vocab_size"]), free=True)
        own.append(io.logits)
        tl = list(m.c.get("dspark_target_layer_ids", []))
        # the input of DSpark from its target layers (read just after the run: a
        # group's is shared scratch)
        io.main_h = (z if share else zo)(t, len(tl) * dim) if tl else None
        ring_bk = {i: self.cache_rows(t, hd, "w") for i in range(L)} if verify else {}
        own.extend(ring_bk.values())
        c16 = self.cform
        cmp_bk = {i: (z(m.ratios[i], hd), z(m.ratios[i], hd)) for i in m.kv_sources
                  if verify and m.ratios[i] > 1}
        prog.names.update(h=h, logits=io.logits)
        hp, vp = z(t, dim, pin=True), z(t * k, pin=True)
        ip = z(*((2 * k + 1,) if t == 1 else (t * k,)), dtype=np.int32, pin=True)
        host_rows = z(t * k, dim, pin=True)
        part = z(t, dim)
        # hcs (GP_DS_HC_STEP): a sublayer's post with the next one's mixes and pre in one record; the
        # CPU's sums of its experts (a row a token, pinned: no flag 64), the GPU's hot pairs summed
        # before the CPU's task ends; the blocks' sums and the counters of the step (own zeros: the
        # last block of a token puts its counter back to 0)
        hcs = self.hc_step != 0
        hfuse = self.hc_step == 1         # (2: the CPU's sums, the three records of before; a check)
        lay_set = set(lay)
        host_sum = z(t, dim, dtype=np.float64, pin=True) if hcs else None   # (float64: flag 2048)
        part_d = z(t, dim, dtype=np.float64) if hcs else None                # (the GPU's pairs, float64)
        hcs_part = z(t, 128, 26) if hcs else None
        hcs_cnt = None
        if hcs:
            own.append(np.zeros(t, np.int32))
            hcs_cnt = own[-1]
        pend_f, pend_a = {}, {}       # lane -> the layer whose FFN / attention post waits for the next record
        # lane -> the flags of its last CPU task: a step's GP_AWAIT does not wait for a task with no
        # cold expert, so the next signal waits for it before it writes the pinned inputs that task
        # reads (hp, vp, ip are one set for all the layers: GP_SIGNAL operand 11, signal_wait)
        last_task = {}
        hq = dict(qh=z(t, dim, dtype=np.int8), ql=z(t, dim, dtype=np.int8), xs=z(t, dim // 32),
                  act=z(t * k, 2 * I), act2=z(t * k, I), aqh=z(t * k, I, dtype=np.int8),
                  aql=z(t * k, I, dtype=np.int8), axs=z(t * k, I // 32), de=z(t * k, dim))
        qrank = int(m.c["q_lora_rank"])
        x, xa, qa, qr, q = z(t, dim), z(t, dim), z(t, qrank), z(t, qrank), z(t, H * hd)
        kvr, kvn, kvq = z(t, hd), z(t, hd), z(t, hd)
        o, oa, y = z(t, H * hd), z(t, m.groups * m.o_rank), z(t, dim)
        nmax = {i: (t if m.ratios[i] == 1 else t // m.ratios[i] + 1) for i in m.kv_sources}
        lat_kv0, lat_sc0 = z(t, hd), z(t, hd)
        # a verify group keeps the rows of each compressor (restore)
        lat_kvs = {i: (z(t, hd), z(t, hd)) if i in cmp_bk else (lat_kv0, lat_sc0) for i in m.kv_sources}
        latent = {i: z(nmax[i], hd) for i in m.kv_sources}
        kraw = {i: z(nmax[i], m.ix_dim) for i in m.kv_sources}
        iq, iw = z(t, m.ix_heads * m.ix_dim), z(t, m.ix_heads)
        logits_r, sg, su, sa, ys = z(t, self.w.src.E), z(t, I), z(t, I), z(t, I), z(t, dim)
        # the entries of each source below cap; the index scores of every
        # layer in one array (each layer's are read in that layer): of all
        # the entries at once, or (chw < sw) of chunks of chw entries, the
        # top k then over the chunks in turn (DS_TOPK_MERGE: the same choice)
        maxn = {i: min(self.kcache[i].shape[0], self.cap // m.ratios[i]) for i in self.kcache}
        sw = max(maxn.values())
        # (a group's scores are a view of the shared scratch, whose base has the rows of the largest
        # group (_scratch's most): the chunks fit SCORE_MB for those rows. For its own t rows a small
        # group took every entry, and its base most / t times SCORE_MB: 4 GB at cap 512K, where the
        # programs then did not fit beside the hot experts at each request of a long session)
        chw = self.score_chunk(max(t, win if stage == "dec" else max(self.group, win)) if share else t, sw)
        scores = dict.fromkeys(self.kcache, z(t, chw))
        sel = {i: z(t, m.ix_topk, dtype=np.int32) for i in m.index_sources}
        cnt = {i: z(t, dtype=np.int32) for i in m.index_sources}
        csrc = m.cand_src
        nb = maxn[csrc] // m.cand_bs + 1 if csrc in maxn else 0
        bmax = z(t, nb if chw == sw else chw // m.cand_bs)
        csel, ccnt = z(t, m.cand_topk, dtype=np.int32), z(t, dtype=np.int32)
        cmask = z(t, nb, dtype=np.uint8)
        # the k best so far of the chunks (their scores)
        svals, cvals = (z(t, m.ix_topk), z(t, m.cand_topk)) if chw < sw else (None, None)
        # the keys and values of engram, read in their layer: one array for the
        # layers (each lane its own rows)
        ekv = dict.fromkeys(m.engram_layers, z(t, (hc + 1) * dim))
        # a step splits the keys of its attention (DS_ASPLIT ranges of kernels/dsv41/step.cu)
        apart = z(t, H, 8, hd + 2) if t <= VERIFY_MAX else None
        wscale = float(np.float32(m.ix_dim ** -0.5 * m.ix_heads ** -0.5))
        ascale = float(np.float32(hd ** -0.5))

        def lin8(xb, name, out, groups=0):
            c, s, rows, cols = self.fp8(name)
            prog.emit(P.DS_FP8_LINEAR, xb, c, s, out, rows, cols, xb.shape[0], groups)

        def lin12(xb, name, out):
            self.emit_dense16(prog, xb, name, out)

        # the lanes: the rows of a run in one lane, or (ab: a group of a prompt,
        # a verify group of DSpark) in two halves A, B whose records
        # interleave, so that the CPU's experts of one half run while the GPU
        # runs the other half (the same records and bits for each row; half B
        # at the slot pos1 = pos + t / 2). Halves of a group of 64 or more take
        # the kernels of a group, of a verify group those of a step; a group of
        # 9 to 63 stays whole (its halves would take other kernels: other bits)
        lanes = [(0, t)]
        if self.ab and (t >= 64 or 2 <= t <= VERIFY_MAX):
            lanes = [(0, t // 2), (t // 2, t)]
        io.lanes = lanes
        lpos = [pos] + [prog.slot("pos%d" % j) for j in range(1, len(lanes))]
        raw_as = {i: z(t, 24) for i in lay}
        raw_fs = {i: z(t, 24) for i in lay}
        # the post's q and comb of each sublayer, made by its GP_DS_HC_PRE (from the sublayer's mixes)
        coef_as = {i: z(t, 20) for i in lay}
        coef_fs = {i: z(t, 20) for i in lay}
        # the kv source and the index source of each layer
        kvs, ixs, ks, xs_ = {}, {}, None, None
        for i in range(L):
            if m.ratios[i]:
                ks = i if i in m.kv_sources else ks
                xs_ = i if i in m.index_sources else xs_
            kvs[i], ixs[i] = ks, xs_
        if t == 1:
            colds = {i: (z(2 * k + 1, dtype=np.int32), z(k)) for i in lay}
        else:
            colds = {i: (z(t * k, dtype=np.int32), z(t * k)) for i in lay}
        x8 = self.x8_prefill and t > VERIFY_MAX
        flags = {}
        # the stage of the cold experts of the most pairs (STAGE): each half's map with its stage slots
        # (and A's: B's map starts from A's), its stage slots alone, the experts it copies in
        nstage = self.stage_n if share else 0
        if nstage:
            sbuf = self.stage_buf(3 if len(lanes) == 2 else 1)
            smaps = [z(self.w.src.E, dtype=np.int32) for _ in lanes]
            omaps = [z(self.w.src.E, dtype=np.int32) for _ in lanes]
            slists = [z(nstage, dtype=np.int32) for _ in lanes]
        hmap = {}                   # (lane, layer) -> the map of its pairs (the hot slots and the stage's)
        pulls = {}                  # lane -> [layer, the first expert of its pull not copied yet]

        def stage_s0(j, i):
            """The first stage slot of half j's experts of layer i: A's in two sets by the layer (B runs
            them in its ffn_post of layer i, after A's pull of layer i + 1 started), B's in the third."""
            if len(lanes) == 1:
                return 0
            return 2 * nstage if j else (i % 2) * nstage

        def pull(j, i, a, b):
            """GP_DS_STAGE_PULL of experts a..b of half j's list of layer i."""
            prog.emit(P.DS_STAGE_PULL, slists[j][a:b], self.stage_tab[i], self.stage_tab.shape[1], b - a, sbuf,
                      self.per, stage_s0(j, i) + a, STAGE_BLOCKS, j)

        def pump(j, rest=False):
            """The next piece of half j's pull (rest: all of it left)."""
            q = pulls.get(j)
            if q is None:
                return
            b = nstage if rest else min(nstage, q[1] + STAGE_CHUNK)
            pull(j, q[0], q[1], b)
            if b == nstage:
                del pulls[j]
            else:
                q[1] = b

        def rows(j):
            r0, r1 = lanes[j]
            return r1 - r0, lpos[j], lambda a, per=1: a[r0 * per:r1 * per]

        def kv_write(i, x, ps, n, S, first=True):
            """The compressed rows and index keys of kv source i from its x
            (first: the first lane, which keeps the compressor's state of the
            start of a verify group for restore)."""
            a, ratio = "layers.%d.attn." % i, m.ratios[i]
            c = a + "compressor."
            lat_kv, lat_sc = S(lat_kvs[i][0]), S(lat_kvs[i][1])
            lin12(x, c + "wkv.weight", lat_kv)
            if ratio > 1:
                lin12(x, c + "wgate.weight", lat_sc)
                bk = cmp_bk.get(i, (None, None)) if first else (None, None)
                prog.emit(P.DS_COMPRESS, lat_kv, lat_sc, self.kstate[i], self.sstate[i],
                          self.vec(c + "norm.weight"), latent[i], ps, ratio, hd, eps, n, nmax[i], bk[0], bk[1])
            else:
                prog.emit(P.DS_COMPRESS, lat_kv, None, None, None, self.vec(c + "norm.weight"), latent[i],
                          ps, 1, hd, eps, n, nmax[i])
            lin12(latent[i], a + "indexer.wk.weight", kraw[i])
            prog.emit(P.DS_CMP_WRITE, latent[i], kraw[i], self.vec(a + "indexer.k_norm.weight"), cc[0],
                      cc[1], self.ckv[i], self.kcache[i], ps, ratio, hd, m.ix_dim, rd, eps, n, nmax[i], c16)

        def prev_of(i):
            """The raw mixes, scale, base of the FFN before layer i (or Nones)."""
            if i - 1 in raw_fs:
                return (raw_fs[i - 1],) + self.hcp("layers.%d.hc_ffn" % (i - 1))[1:]
            if stage == "dec":
                return (self.rep_raw,) + self.hcp("layers.%d.hc_ffn" % (E - 1))[1:]
            return (None, None, None)

        def attn(j, i):
            """The records of layer i up to its attention's output in the streams."""
            n, ps, S = rows(j)
            hl, p = S(h), "layers.%d." % i
            if i in m.engram_layers:
                e = p + "engram."
                lin8(S(io.eg[i]), e + "wkv.weight", S(ekv[i]))
                prog.emit(P.DS_ENGRAM, S(ekv[i]), hl, self.vec(e + "q_weight", e + "k_weight"), n, dim, hc, eps)
            fa, sca, ba = self.hcp(p + "hc_attn")
            raw_a = S(raw_as[i])
            pi = pend_f.pop(j, None)
            if pi is not None:
                # the FFN of layer pi's post, this layer's mixes and pre
                _ff, scf_p, bf_p = self.hcp("layers.%d.hc_ffn" % pi)
                cold = colds[pi][0]
                prog.emit(P.DS_HC_STEP, S(part_d), S(ys), S(host_sum), cold[k:k + 1] if t == 1 else None, hl,
                          S(coef_fs[pi]), fa, raw_a, S(raw_fs[pi]), scf_p, bf_p, self.vec(p + "attn_norm.weight"),
                          S(x), sca, ba, S(coef_as[i]), S(hcs_part), S(hcs_cnt), n, dim, eps, hc_eps, iters, 1)
                if io.layers is not None:
                    lr = S(io.layers[pi])
                    prog.emit(P.COPY, hl, lr, lr.nbytes)      # (layer pi's streams after its post)
                if i in tl:
                    prog.emit(P.DS_STREAM_MEAN, hl, S(io.main_h), n, dim, io.main_h.shape[1], tl.index(i) * dim, hc)
            else:
                if i in tl:
                    prog.emit(P.DS_STREAM_MEAN, hl, S(io.main_h), n, dim, io.main_h.shape[1], tl.index(i) * dim, hc)
                prog.emit(P.DS_HC_MIX, hl, fa, raw_a, n, hc * dim, eps)
                pm = prev_of(i)
                prog.emit(P.DS_HC_PRE, hl, None if pm[0] is None else S(pm[0]), pm[1], pm[2],
                          self.vec(p + "attn_norm.weight"), S(x), n, dim, eps, hc_eps, raw_a, sca, ba, iters,
                          S(coef_as[i]))
            a = p + "attn."
            ratio = m.ratios[i]
            cs = cc if ratio else cw
            pump(1 - j)
            lin8(S(x), a + "wq_a.weight", S(qa))
            prog.emit(P.RMS_NORM, S(qa), self.vec(a + "q_norm.weight"), S(qr), n, qrank, eps)
            lin8(S(qr), a + "wq_b.weight", S(q))
            pump(1 - j)
            prog.emit(P.DS_ROPE, S(q), cs[0], cs[1], ps, n, H, hd, rd, 0, 0)
            lin8(S(x), a + "wkv.weight", S(kvr))
            prog.emit(P.RMS_NORM, S(kvr), self.vec(a + "kv_norm.weight"), S(kvn), n, hd, eps)
            prog.emit(P.DS_WIN_WRITE, S(kvn), cs[0], cs[1], S(kvq), ps, n, win, hd, rd)
            ckv = sl = cn = None
            if ratio:
                kv_src, idx_src = kvs[i], ixs[i]
                if i == kv_src and not (stage == "dec" and i == E):      # (the encoder groups wrote them)
                    kv_write(i, S(x), ps, n, S, j == 0)
                if i in m.index_sources:
                    ix = a + "indexer."
                    lin8(S(qr), ix + "wq_b.weight", S(iq))
                    prog.emit(P.DS_ROPE, S(iq), cc[0], cc[1], ps, n, m.ix_heads, m.ix_dim, rd, 0, 0)
                    prog.emit(P.DS_FP4, S(iq), S(iq).size, 32)
                    lin12(S(x), ix + "weights_proj.weight", S(iw))
                    mask = S(cmask) if 0 <= csrc < i else None
                    if chw == sw:
                        prog.emit(P.DS_IDX_SCORE, S(iq), S(iw), wscale, self.kcache[kv_src], ps, ratio, mask,
                                  m.cand_bs, S(scores[kv_src]), m.ix_heads, m.ix_dim, sw, n, nb, c16)
                        if i == csrc:
                            prog.emit(P.DS_CAND, S(scores[kv_src]), ps, ratio, m.cand_bs, S(bmax), nb, n, sw)
                            prog.emit(P.DS_TOPK, S(bmax), ps, ratio, m.cand_bs, m.cand_topk, S(csel), S(ccnt),
                                      S(cmask), nb, n)
                        prog.emit(P.DS_TOPK, S(scores[kv_src]), ps, ratio, 1, m.ix_topk, S(sel[i]), S(cnt[i]), None,
                                  sw, n)
                    else:
                        nch, cb = -(-maxn[kv_src] // chw), chw // m.cand_bs
                        for c in range(nch):
                            c0, first, last = c * chw, int(c == 0), int(c == nch - 1)
                            prog.emit(P.DS_IDX_SCORE, S(iq), S(iw), wscale, self.kcache[kv_src], ps, ratio, mask,
                                      m.cand_bs, S(scores[kv_src]), m.ix_heads, m.ix_dim, chw, n, nb, c16, c0)
                            if i == csrc:
                                prog.emit(P.DS_CAND, S(scores[kv_src]), ps, ratio, m.cand_bs, S(bmax), cb, n, chw, c0)
                                prog.emit(P.DS_TOPK_MERGE, S(bmax), ps, ratio, m.cand_bs, m.cand_topk, S(csel),
                                          S(ccnt), S(cvals), cb, n, c0 // m.cand_bs, first, S(cmask) if last else None,
                                          nb, last)
                            prog.emit(P.DS_TOPK_MERGE, S(scores[kv_src]), ps, ratio, 1, m.ix_topk, S(sel[i]),
                                      S(cnt[i]), S(svals), chw, n, c0, first, None, 0, 0)
                ckv, sl, cn = self.ckv[kv_src], S(sel[idx_src]), S(cnt[idx_src])
            pump(1 - j)
            prog.emit(P.DS_ATTN, S(q), self.window[i], ps, win, ckv, sl, cn, self.vec(a + "attn_sink"), S(o), H, hd,
                      ascale, m.ix_topk, S(kvq), n, apart, 2 if stage == "dec" else 0, c16,
                      prog.slot("rbase") if stage == "dec" else 0, self.akeys if apart is not None else None)
            prog.emit(P.DS_RING_PUT, S(kvq), self.window[i], ps, n, win, hd,
                      S(ring_bk[i]) if i in ring_bk else None, c16)
            pump(1 - j)
            prog.emit(P.DS_ROPE, S(o), cs[0], cs[1], ps, n, H, hd, rd, 1, 0)
            lin8(S(o), a + "wo_a.weight", S(oa), groups=m.groups)
            pump(1 - j)
            lin8(S(oa), a + "wo_b.weight", S(y))
            pump(1 - j)
            if hfuse:
                pend_a[j] = i           # (the post with the FFN's mixes and pre: ffn_pre)
            else:
                prog.emit(P.DS_HC_POST, S(y), None, hl, raw_a, sca, ba, hl, n, dim, hc_eps, iters, S(coef_as[i]))

        def flag_of(j, i):
            """The flags of layer i's CPU task of lane j (in, out: pinned; made once)."""
            if (j, i) not in flags:
                fl = flags[j, i] = pinned((2,), np.int64, free=True)
                fl[:] = 0
                prog.keep.append(fl)
                own.append(fl)
            return flags[j, i]

        def ffn_pre(j, i, local=True):
            """The FFN of layer i up to its CPU task: the router, the input of
            the cold experts to the host (the signal of the task), the hot
            experts, the shared one (local; else ffn_local later); then the
            task."""
            n, ps, S = rows(j)
            hl, p = S(h), "layers.%d." % i
            _fa, sca, ba = self.hcp(p + "hc_attn")
            ff, scf, bf = self.hcp(p + "hc_ffn")
            raw_f = S(raw_fs[i])
            if pend_a.pop(j, None) is not None:
                # the attention's post, the FFN's mixes and pre
                prog.emit(P.DS_HC_STEP, S(y), None, None, None, hl, S(coef_as[i]), ff, raw_f, S(raw_as[i]), sca, ba,
                          self.vec(p + "ffn_norm.weight"), S(xa), scf, bf, S(coef_fs[i]), S(hcs_part), S(hcs_cnt),
                          n, dim, eps, hc_eps, iters, 0)
            else:
                prog.emit(P.DS_HC_MIX, hl, ff, raw_f, n, hc * dim, eps)
                prog.emit(P.DS_HC_PRE, hl, S(raw_as[i]), sca, ba, self.vec(p + "ffn_norm.weight"), S(xa), n, dim,
                          eps, hc_eps, raw_f, scf, bf, iters, S(coef_fs[i]))
            f = p + "ffn."
            ids, vals = S(io.ids[i], k), S(io.vals[i], k)
            lin12(S(xa), f + "gate.weight", S(logits_r))
            prog.emit(P.DS_ROUTER, S(logits_r), self.vec(f + "gate.bias"), ids, vals, n, logits_r.shape[1], k,
                      float(m.route_scale), self.vec(f + "gate.bias_vl") if vis else None,
                      S(io.vmask) if vis else None)
            # the routed experts: the hot ones on the GPU, the cold ones in the
            # parts at the same time (their de rows, flag 64), the shared one
            cold, cold_val = colds[i]
            mp = hmap[j, i] = self.maps[i]
            if nstage:
                # the cold experts of the most pairs to the stage (after A's of the layer, for B), their
                # pairs the GPU's: the split leaves them out of the CPU's task
                base = hmap[0, i] if j else self.maps[i]
                mp = hmap[j, i] = smaps[j]
                prog.emit(P.DS_STAGE_PICK, ids, base, mp, slists[j], n * k, self.w.src.E, nstage, STAGE_MINP,
                          stage_s0(j, i), omaps[j])
            if t == 1:
                ipl = ip
                prog.emit(P.HOT_SPLIT, ids, vals, mp, cold, cold_val, k, 1)
            else:
                cold, cold_val, ipl = S(cold, k), S(cold_val, k), S(ip, k)
                prog.emit(P.HOT_SPLIT_MT, ids, vals, mp, cold, cold_val, n * k)
            fl = flag_of(j, i)
            hpl, vpl = S(hp), S(vp, k)
            prev = last_task.get(j) if t == 1 and SIGNAL_CHAIN else None     # (t > 1: GP_AWAIT always waits)
            last_task[j] = fl
            prog.emit(P.SIGNAL, fl[0:1], seq, S(xa), hpl, hpl.nbytes, cold_val, vpl, vpl.nbytes, cold, ipl, ipl.nbytes,
                      prev[1:2] if prev is not None else None)
            if nstage:
                # (after the signal: the CPU's task runs while the GPU copies, beside the kernels after;
                # joined in ffn_post)
                pump(1 - j)
                if STAGE_CHUNK and len(lanes) == 2:
                    pulls[j] = [i, 0]
                else:
                    pull(j, i, 0, nstage)
            if local:
                ffn_local(j, i)
            if hcs:
                # (the CPU's sum of its experts for each token: no flag 64)
                cpu = self.parts.program(i, n, k, hpl, ipl, vpl, S(host_sum), ip[k:k + 1] if t == 1 else None,
                                         32 | 128 | 2048 | (256 if x8 else 0), self.part_bufs)
            else:
                cpu = self.parts.program(i, n, k, hpl, ipl, vpl, S(host_rows, k), ip[k:k + 1] if t == 1 else None,
                                         32 | 64 | 128 | (256 if x8 else 0), self.part_bufs)
            io.cpu.append(cpu)
            own.append(cpu.buf)         # (in the mirror with the program's keep: forgotten with it)
            prog.emit(P.CPU_TASK, cpu.buf, fl[0:1], fl[1:2], seq)
            if self.spec and t == 1 and i + 1 < L and i + 1 in self.parts.pm:
                # the next layer's guess, once this task is done (its flag out)
                gw, gb = self.spec_router(i + 1)
                sp = self.parts.program_spec(i + 1, k, hpl, gw, gb, self.maps[i + 1], fl[1:2],
                                             flag_of(j, i + 1)[0:1])
                io.cpu.append(sp)
                sfl = pinned((1,), np.int64, free=True)
                sfl[:] = 0
                prog.keep.append(sfl)
                own.append(sfl)
                own.append(sp.buf)
                prog.emit(P.CPU_TASK, sp.buf, fl[1:2], sfl, seq)

        def ffn_local(j, i):
            """The hot experts and the shared one of layer i on the rows of lane
            j (None: all the rows)."""
            n, _ps, S = rows(j) if j is not None else (t, None, lambda a, per=1: a)
            f = "layers.%d.ffn." % i
            ids = S(io.ids[i], k)
            # (with the stage: the hot slots now, the stage's pairs and the sum in ffn_post, after the pull)
            mp = self.maps[i] if nstage else hmap.get((j, i), self.maps[i])
            hot_moe(j, i, mp)
            if hcs and not nstage:
                # the GPU's pairs summed now, while the CPU's task runs (the CPU sends its own sum)
                prog.emit(P.DS_COMBINE, ids, mp, S(hq["de"], k), None, S(part_d), k, dim, n, int(t > 1), 1)
            sx = f + "shared_experts."
            lin8(S(xa), sx + "w1.weight", S(sg))
            lin8(S(xa), sx + "w3.weight", S(su))
            prog.emit(P.DS_SWIGLU, S(sg), S(su), S(sa), n * I, float(m.limit))
            lin8(S(sa), sx + "w2.weight", S(ys))

        def hot_moe(j, i, mp):
            """DS_HOT_MOE of the pairs of lane j (None: all the rows) whose experts mp gives a slot."""
            n, _ps, S = rows(j) if j is not None else (t, None, lambda a, per=1: a)
            prog.emit(P.DS_HOT_MOE, S(xa), S(io.ids[i], k), S(io.vals[i], k), mp, self.stores[i], self.per,
                      self.off3, self.off2, n, k, dim, I, S(hq["qh"]), S(hq["ql"]), S(hq["xs"]), S(hq["act"], k),
                      S(hq["act2"], k), S(hq["aqh"], k), S(hq["aql"], k), S(hq["axs"], k), S(hq["de"], k),
                      self.w.src.E, int(x8), sbuf if nstage else None)

        def ffn_post(j, i):
            """The rows of the CPU's task back, the sum of the experts into the streams."""
            n, _ps, S = rows(j)
            hl, p = S(h), "layers.%d." % i
            _ff, scf, bf = self.hcp(p + "hc_ffn")
            fl, cold = flags[j, i], colds[i][0]
            if nstage:
                # the stage's pairs once the pull is in (the CPU's task still runs), then all the GPU's
                # pairs summed as before (one DS_COMBINE: the same order)
                pump(j, rest=True)
                prog.emit(P.DS_STAGE_JOIN, j)
                hot_moe(j, i, omaps[j])
                if hcs:
                    prog.emit(P.DS_COMBINE, S(io.ids[i], k), smaps[j], S(hq["de"], k), None, S(part_d), k, dim, n,
                              int(t > 1), 1)
            prog.emit(P.AWAIT, fl[1:2], seq, None, None, 0, cold[k:k + 1] if t == 1 else None)
            if hcs:
                if hfuse and i + 1 in lay_set and i + 1 not in m.engram_layers:
                    pend_f[j] = i       # (the post with the next layer's mixes and pre: attn)
                else:
                    prog.emit(P.DS_HC_POST, S(part_d), S(ys), hl, S(raw_fs[i]), scf, bf, hl, n, dim, hc_eps, iters,
                              S(coef_fs[i]), S(host_sum), cold[k:k + 1] if t == 1 else None, 1)
                    if io.layers is not None:
                        lr = S(io.layers[i])
                        prog.emit(P.COPY, hl, lr, lr.nbytes)
                return
            prog.emit(P.DS_COMBINE, S(io.ids[i], k), hmap.get((j, i), self.maps[i]), S(hq["de"], k), S(host_rows, k),
                      S(part), k, dim, n, int(t > 1))           # (the rows of GP_HOT_SPLIT_MT, also in a lane of one token)
            prog.emit(P.DS_HC_POST, S(part), S(ys), hl, S(raw_fs[i]), scf, bf, hl, n, dim, hc_eps, iters,
                      S(coef_fs[i]))
            if io.layers is not None:
                lr = S(io.layers[i])
                prog.emit(P.COPY, hl, lr, lr.nbytes)      # (the lane's rows: the other may be a layer behind)

        if stage != "dec":
            prog.emit(P.DS_EMBED, self.embed_tab, io.tok, h, t, dim, hc)
            if vis:
                prog.emit(P.DS_EMBED_ROWS, io.vrows, io.vmask, h, t, dim, hc)
            for i in m.engram_layers:
                prog.emit(P.DS_ENGRAM_ROWS, io.egc[i], io.egs[i], io.eg[i], t * 24, 256)
        if len(lanes) == 1:
            for i in lay:
                attn(0, i)
                ffn_pre(0, i)
                ffn_post(0, i)
        else:
            # A's task runs on the CPU while the GPU runs B (and the reverse).
            # A verify group of 2: the hot and shared experts of A and B after
            # B's signal, on both rows at once (A's are read only by its
            # ffn_post, after them: the weights once, not once a half; the
            # kernels of t <= 8 give each row the bits of a step): 51.0 -> 49.4
            # ms, the GPU its bound. Not for more rows: the CPU is the bound and
            # A's experts filled the GPU's wait for B's task (6 rows 103 -> 106)
            merge = t == 2 and self.verify_merge
            lay_ = list(lay)
            for n_, i in enumerate(lay_):
                attn(0, i)
                ffn_pre(0, i, local=not merge)
                if n_ > 0:
                    ffn_post(1, lay_[n_ - 1])
                attn(1, i)
                ffn_pre(1, i, local=not merge)
                if merge:
                    ffn_local(None, i)
                ffn_post(0, i)
            ffn_post(1, lay_[-1])
        prev = prev_of(lay[-1] + 1)
        if stage == "enc":
            # the decoder's global KV from the encoder's output (CED), and the
            # input of the bounded replay
            fa, _sca, _ba = self.hcp("layers.%d.hc_attn" % E)
            raw_a = z(t, 24)
            prog.emit(P.DS_HC_MIX, h, fa, raw_a, t, hc * dim, eps)
            prog.emit(P.DS_HC_PRE, h, prev[0], prev[1], prev[2], self.vec("layers.%d.attn_norm.weight" % E), x, t,
                      dim, eps, hc_eps)
            kv_write(E, x, pos, t, lambda a, per=1: a)
            rbase = prog.slot("rbase")
            prog.emit(P.DS_ROWS_AT, h, self.rep_h, pos, rbase, t, win, hc * dim)
            prog.emit(P.DS_ROWS_AT, prev[0], self.rep_raw, pos, rbase, t, win, 24)
        else:
            xf = z(t, dim)
            prog.emit(P.DS_HC_PRE, h, prev[0], prev[1], prev[2], self.vec("norm.weight"), xf, t, dim, eps, hc_eps)
            lin12(xf if verify else xf[t - 1:t], "head.weight", io.logits)
        prog.keep += [hp, vp, ip, host_rows, self.rep_h, self.rep_raw] + self.state_arrays()
        io.debug = dict(scores=scores, sel=sel, cnt=cnt, cmask=cmask, csel=csel, ccnt=ccnt)
        if io.layers is not None:
            prog.names["layers"] = io.layers
        prog.finish()
        try:
            io.g = GPUProgram(prog, graph=self.graph, mirror=self.mirror)
            if verify:
                # the restore of a verify group: slot keep, the tokens that stay
                rp = P.Program()
                rpos, keep = rp.slot("pos"), rp.slot("keep")
                for i in range(L):
                    rp.emit(P.DS_RING_RESTORE, self.window[i], ring_bk[i], rpos, t, keep, win, hd, c16)
                for i, (bk, bs) in cmp_bk.items():
                    rp.emit(P.DS_COMPRESS_FIX, self.kstate[i], self.sstate[i], bk, bs, lat_kvs[i][0], lat_kvs[i][1],
                            rpos, keep, m.ratios[i], hd)
                rp.keep += self.state_arrays()
                rp.finish()
                io.restore = GPUProgram(rp, graph=False, mirror=self.mirror)
            # the graphs recorded now: the memory of a program shows here (run: make_room), never in
            # the middle of a run (a verify group's segments half done)
            io.g.prepare()
        except BaseException:
            self.drop_run(io)
            raise
        return io

    def _scratch(self, t, counts, shape, dtype, pin, store=None, most=None):
        """An array of the program of a group of t tokens (t > VERIFY_MAX).
        Those programs run one at a time, and no run reads what a run before
        left in these arrays, so they share them: the k-th array with rows of
        a shape (and a dtype) of each program is a view of one array, made
        for the groups of up to self.group tokens (or the window), a larger
        one when a program needs it. counts: the arrays of the program so far;
        store: the bases (default self.scratch; DSpark's seeds have their own);
        most: the rows a new base is made for (default the group or the
        window: the decoder of a bounded replay and DSpark's seeds see only
        the window)."""
        store = self.scratch if store is None else store
        most = max(self.group, self.m.win) if most is None else most
        dt = np.dtype(dtype)
        key = (pin, dt.str, tuple(shape[1:]))
        counts[key] = counts.get(key, -1) + 1
        key += (counts[key],)
        n = int(np.prod(shape))
        b = store.get(key)
        if b is None or b.size < n:
            size = max(n, -(-n * most // t))
            b = store[key] = pinned((size,), dt, free=True) if pin else np.zeros(size, dt)
        return b[:n].reshape(shape)

    def run(self, t, part="full", vis=False):
        """The program of t tokens (made the first time); vis: the one that
        takes the rows of an image span (_compile)."""
        key = (t, part, "vis") if vis else (t, part)
        if self.lane_id:
            key = ("lane", self.lane_id) + key
        r = self.runs.get(key)
        while r is None:
            try:
                r = self.runs[key] = self._compile_lane(t, part, vis)
            except (MemoryError, RuntimeError) as e:
                if not gpu_oom(e) or not self.make_room(e):
                    raise
        return r

    def _compile_lane(self, t, part, vis):
        """_compile; a lane's step or verify group with scratches of the C side of its own:
        the programs of a lane and the model's run at the same time (lanes.py), and the programs of
        one share gg_scratch_get's (the hot MoE of two runs would read each other's tables)."""
        if not self.lane_id or t > VERIFY_MAX:
            return self._compile(t, part, vis)      # (a prompt's programs run alone)
        from reclank.backends.cuda.runtime import lib
        lib().rc_gpu_scratch_own(1)
        try:
            return self._compile(t, part, vis)
        finally:
            lib().rc_gpu_scratch_own(0)

    def make_room(self, err=None):
        """GPU memory for a program that did not fit (its graphs or its arrays: err), at a point where
        no run is half done: the other programs go (DSpark's too; made again when run), unless they
        went for room in the last ROOM_S seconds (the programs a session needs do not fit together:
        each would close the others in turn), then the hot experts give a slot of each layer (for
        good: the context of a long session keeps growing). False when nothing is left to free."""
        now = time.monotonic()
        have = bool(self.runs) or bool(self.scratch) or any(getattr(getattr(f, "__self__", None), "progs", None)
                                                            for f in self.on_grow)
        if have and now - getattr(self, "_room_t", -1e9) > ROOM_S:
            self._room_t = now
            self.free_programs()
            print("[dsv41] the GPU full (%s): the programs freed" % err, file=sys.stderr, flush=True)
            return True
        n = self.n_slots
        if self.shrink_hot(1):
            print("[dsv41] the GPU full (%s): the hot experts %d -> %d a layer (%.1f GB)" % (
                err, n, self.n_slots, self.n_slots * self.m.L * self.per / 1e9), file=sys.stderr, flush=True)
            return True
        if have:
            self.free_programs()
            return True
        return False

    def relax(self, end):
        """Before a sequence of end positions (a request, at a point where no run is half done):
        after a longer one, cap comes back down to what this one needs (the programs and the tables
        go; smaller ones are made when run), then the hot experts grow back (grow_hot)."""
        m = self.m
        # (the other lanes keep theirs: the cap and the programs are shared)
        others = [e for lid, e in self.ends.items() if lid != self.lane_id]
        need = min(max([end] + others) + 2 * VERIFY_MAX, m.max_len)
        cap = min(m.max_len, CAP)
        while cap < need:
            cap *= 2
        cap = min(cap, m.max_len)
        if cap < self.cap:
            self.cap = cap
            self.free_programs(tables=True)
            self._tables()
        return self.grow_hot()

    def grow_hot(self):
        """The slots that make_room took back, while the GPU keeps GROW_MARGIN_GB free: each layer's
        store made again larger, its experts copied over on the GPU (the programs hold its address:
        they go); the new slots start free, and DSHot fills them with its best experts as it copies.
        The slots added a layer."""
        from reclank.backends.cuda.runtime import _check, lib
        from reclank.backends.cuda.gpumm import mem_info
        n, L = self.n_slots, self.m.L
        if self.hot is None or n >= self.n_max:
            return 0
        slot = L * self.per
        k = min(self.n_max - n, int((mem_info()[0] - GROW_MARGIN_GB * 1e9) // slot))
        if k <= 0:
            return 0
        self.hot.land(wait=True)
        self.free_programs()
        n2 = n + k
        for i in range(L):
            old = self.stores[i]
            ph = _placeholder(n2 * self.per)
            self.mirror.add(ph)
            b = self.mirror.bufs[ph.ctypes.data] = Buffer(ph.nbytes, "program", pinned=True)
            if n > 0:
                _check(lib().rc_gpu_d2d(b.ptr, self.mirror.buffer_of(old).ptr, n * self.per))
                _check(lib().rc_gpu_sync())
            self.mirror.forget(old.ctypes.data)
            self.stores[i] = ph
            se = np.full(n2, -1, np.int64)
            se[:n] = self.slot_expert[i][:n]
            self.slot_expert[i] = se
        self.n_slots = n2
        print("[dsv41] the hot experts %d -> %d a layer (%.1f GB; the GPU %.1f GB free)" % (
            n, n2, n2 * slot / 1e9, mem_info()[0] / 1e9), file=sys.stderr, flush=True)
        return k

    def shrink_hot(self, k=1):
        """k slots of each layer's hot experts back to the GPU: the experts with the lowest scores
        leave (DSHot; else the last slots), each layer's store made again smaller (the programs hold
        its address: they go) and the experts that stay copied in from the weightkeeper. The bytes
        freed (0: none held)."""
        from reclank.backends.cuda.runtime import _check, lib
        n = self.n_slots
        if n <= 0:
            return 0
        if self.hot is not None:
            self.hot.land(wait=True)
        n2 = max(0, n - k)
        self.free_programs()
        L = self.m.L
        items, ranges = [], []
        for i in range(L):
            held = [(s, int(e)) for s, e in enumerate(self.slot_expert[i][:n]) if e >= 0]
            if self.hot is not None:
                held.sort(key=lambda se: -self.hot.score[i, se[1]])
            self.mirror.forget(self.stores[i].ctypes.data)
            ph = _placeholder(max(n2, 1) * self.per)
            self.mirror.add(ph)
            self.mirror.bufs[ph.ctypes.data] = Buffer(ph.nbytes, "program", pinned=True)
            self.stores[i] = ph
            self.maps[i][:] = -1
            self.slot_expert[i] = np.full(max(n2, 1), -1, np.int64)
            for slot, (_s, e) in enumerate(held[:n2]):
                items.append((i, slot, e))
                if self.hot is not None:
                    ranges += self.expert_ranges(i, e, slot)
        self.n_slots = n2
        if self.hot is not None:
            self.hot.incoming[:] = False
            if ranges:
                ra = np.array(ranges, np.int64)
                job = lib().rc_gpu_cache_copy(ra.ctypes.data, len(ranges) // 3)
                if job < 0 or lib().rc_gpu_cache_query(job, 1) < 0:
                    raise RuntimeError(lib().rc_gpu_last_error().decode())
        for i, slot, e in items:
            if self.hot is None and n2 > 0:
                host = self.expert_bytes(i, e)
                _check(lib().rc_gpu_h2d(self.mirror.buffer_of(self.stores[i]).ptr + slot * self.per,
                                        host.ctypes.data, self.per))
            self.slot_expert[i][slot] = e
            self.maps[i][e] = slot
        for i in range(L):
            self.mirror.buffer_of(self.maps[i]).upload(self.maps[i])
        return (n - n2) * L * self.per

    # ---- the inputs, a step, a prompt
    def engram_rows(self, layer, hashes):
        """The 24 rows of each token (E4M3 values times their E8M0 scales), as
        the CPU model reads them: [tokens, 24 * 256]."""
        pre = "layers.%d.engram." % layer
        b, _dt, shape = self.w.src.raw(pre + "embed.weight", copy=False)
        s, _sdt, sshape = self.w.src.raw(pre + "embed.scale", copy=False)
        table = np.asarray(b).reshape(shape)
        scales = np.asarray(s).reshape(sshape)
        hashes = np.asarray(hashes)
        flat = hashes.reshape(-1)
        vals = D.E4M3[table[flat]].reshape(-1, shape[1] // 32, 32) * D.E8M0[scales[flat]][..., None]
        return vals.reshape(hashes.shape[0], -1).astype(np.float32)

    def _inputs(self, r, tokens, pos, rbase=0, media=None):
        """The inputs of a run: the token ids (the GPU reads and decodes their
        embedding rows) and the bytes of the engram rows of their hashes.
        media (a program of vis): (rows [t, dim] or None, mask [t]) of the
        tokens of image spans. Their n-grams stop at them (Hashes: dead) and
        engram adds nothing to them: their rows are zero, so the value of wkv
        is zero (model.py Engram: the gate shut)."""
        m = self.m
        dead = None
        if r.vis:
            rows, mask = media
            r.vmask[:] = mask
            self.mirror.buffer_of(r.vmask).upload(r.vmask)
            if r.vrows is not None:
                r.vrows[:] = rows
                self.mirror.buffer_of(r.vrows).upload(r.vrows)
            dead = np.asarray(mask) != 0
        if r.lay.start > 0:
            # the decoder of a bounded replay: its input is the encoder's (rep_h)
            r.g.bind(self._slots(r, pos, rbase))
            return
        tokens = np.asarray(tokens, np.int64)
        hashes = m.hashes.ids(tokens, pos, dead)
        r.tok[:] = tokens
        self.mirror.buffer_of(r.tok).upload(r.tok)
        for j, i in enumerate(m.engram_layers):
            r.egc[i][:], r.egs[i][:] = self.engram_codes(i, hashes[:, j])
            if dead is not None:
                r.egc[i].reshape(len(tokens), -1)[dead] = 0
            self.mirror.buffer_of(r.egc[i]).upload(r.egc[i])
            self.mirror.buffer_of(r.egs[i]).upload(r.egs[i])
        r.g.bind(self._slots(r, pos, rbase))

    def _slots(self, r, pos, rbase):
        """The slots of a run: pos, rbase, the first position of each lane."""
        sl = {"pos": int(pos), "rbase": int(rbase)}
        for j, (r0, _r1) in enumerate(r.lanes[1:], 1):
            sl["pos%d" % j] = int(pos) + r0
        return sl

    def forward(self, tokens, pos, part="full", rbase=0, media=None):
        """len(tokens) tokens at pos (a program of that size); the logits of
        the last one [vocab] (part "enc": None). The hot cache (when the GPU
        holds experts) lands its finished copies before, learns the choice of
        the run after, and starts its next copies. media: (rows, mask) of
        the tokens of image spans (media_rows), or None."""
        self.fit(pos + len(tokens))
        r = self.run(len(tokens), part, media is not None)
        if self.hot is not None:
            # a group of a prompt waits for the copies the group before chose:
            # its run is seconds, a copy of 256 experts about half of one
            self.hot.land(wait=len(tokens) > VERIFY_MAX)
        self._inputs(r, tokens, pos, rbase, media)
        r.g.run()
        step = len(tokens) == 1 and part == "full"
        if self.hot is not None and step:
            # the choices of the step before: the GPU ends this one meanwhile (rc_gpu_run returns
            # with the last CPU task, before the head: 1.3 ms), so its plan costs the step nothing;
            # its copies start a token later
            self._hot_flush()
        if part != "enc":
            r.g.download("logits")
        if r.main_h is not None and part != "enc" and len(tokens) <= VERIFY_MAX:
            # (not an encoder's: DSpark's target layers are the decoder's, so it never writes main_h)
            # (the input of DSpark too, now: a lane's generator reads its run's outputs later, after
            # the other lane may have freed the programs for room: lanes.py, DSpark._run)
            self.download(r.main_h)
        self.last = r
        if self.hot is not None:
            ids = self.download(r.ids)
            if step:
                self._hot_seen = (ids.copy(), r.lay)
            else:
                self._hot_flush()
                self.hot.observe(ids, len(tokens), r.lay)
                self.hot.plan(len(tokens))
        return r.logits[-1].copy() if part != "enc" else None

    def _hot_flush(self):
        """The observation and the plan of a step left for later (forward), if any."""
        seen = getattr(self, "_hot_seen", None)
        if seen is not None:
            self._hot_seen = None
            self.hot.observe(seen[0], 1, seen[1])
            self.hot.plan(1)

    def verify(self, tokens, pos):
        """A verify group of DSpark (len(tokens) <= VERIFY_MAX): the logits of
        every row [t, vocab] and the input of DSpark of every row."""
        self.fit(pos + len(tokens))
        r = self.run(len(tokens))
        if self.hot is not None:
            self.hot.land()
        self._inputs(r, tokens, pos)
        r.g.run()
        r.g.download("logits")
        self.download(r.main_h)
        self.last = r
        if self.hot is not None:
            self._hot_flush()
            self.hot.observe(self.download(r.ids), len(tokens), r.lay)
            self.hot.plan(len(tokens))
        return r.logits.copy(), r.main_h.copy()

    def restore(self, t, pos, keep):
        """Undo the rows of the tokens from keep on of the last verify group of t
        tokens at pos (the refused drafts): the ring rows they replaced, the
        compressor's open group. Their compressed entries and index keys stay:
        no query reads an entry past its own position, and the true tokens
        write them again."""
        r = self.run(t)
        r.restore.bind({"pos": int(pos), "keep": int(keep)})
        r.restore.run()

    def step(self, token, pos):
        """One token at pos; the logits [vocab]. The engram hashes need the
        tokens before (the CPU model's Hashes keeps them by position)."""
        return self.forward([token], pos)

    @staticmethod
    def media_rows(spans, p0, n, dim, rows=True):
        """The input of a run of n tokens at p0 from the image spans (each
        (start position, rows [len, dim])): (rows [n, dim] or None, mask [n]
        int32), or None when no span meets the run."""
        hit = [(s, a) for s, a in spans or () if s < p0 + n and s + a.shape[0] > p0]
        if not hit:
            return None
        mask = np.zeros(n, np.int32)
        out = np.zeros((n, dim), np.float32) if rows else None
        for s, a in hit:
            lo, hi = max(s, p0), min(s + a.shape[0], p0 + n)
            mask[lo - p0:hi - p0] = 1
            if rows:
                out[lo - p0:hi - p0] = a[lo - s:hi - s]
        return out, mask

    def prefill(self, tokens, pos=0, group=None, each=None, media=None):
        """A prompt in groups of up to group tokens (default self.group): the
        largest multiple of GROUP_STEP, below it the powers of two; the logits
        of its last token.
        With replay and more tokens than the window: the groups through the
        encoder, then the last window of tokens through the decoder (bounded
        replay). each(run, pos, n): after each run that gave main_h (DSpark).
        The programs serve the whole prompt from the first group (fit).
        media: the image spans of the prompt, (start position, rows [len,
        dim]) each (DSVision.span_rows)."""
        group = group or self.group
        tokens = list(tokens)
        n_all, win = len(tokens), self.m.win
        self.fit(pos + n_all)
        rep = self.replay and n_all > win
        base = pos + n_all - win
        at, out = 0, None
        while at < n_all:
            n = min(group, n_all - at)
            if GROUP_STEP and n >= GROUP_STEP:
                n -= n % GROUP_STEP
            else:
                n = 1 << (n.bit_length() - 1)
            out = self.forward(tokens[at:at + n], pos + at, "enc" if rep else "full", base,
                               self.media_rows(media, pos + at, n, self.m.dim))
            if not rep and each is not None:
                each(self.last, pos + at, n)
            at += n
        if rep:
            out = self.forward(tokens[n_all - win:], base, "dec", base,
                               self.media_rows(media, base, win, self.m.dim, rows=False))
            if each is not None:
                each(self.last, base, win)
        return out

    @property
    def ids(self):
        return self.run(1).ids

    @property
    def layers_out(self):
        return self.run(1).layers[:, 0]

    def profile(self, token, pos):
        """The time of each kind of record of a step (ms, summed over the
        layers), the longest first; a step at pos with token (the state
        changes as in step)."""
        r = self.run(1)
        self._inputs(r, [token], pos)
        ms = r.g.profile()
        out = {}
        for (op, _args), t in zip(r.prog.recs, ms):
            name = P.OP_NAMES.get(op, str(op))
            n, s = out.get(name, (0, 0.0))
            out[name] = (n + 1, s + float(t))
        return sorted(out.items(), key=lambda kv: -kv[1][1])

    def download(self, a):
        self.mirror.buffer_of(a).download(a)
        return a

    def close(self):
        """The programs freed, and the device copies of their own arrays (the
        shared scratch stays)."""
        for r in self.runs.values():
            self.drop_run(r)
        self.runs = {}

    def drop_run(self, r):
        """A _Run's GPUPrograms closed and the device copies of its own arrays forgotten (also a
        _Run whose making failed half way)."""
        for gp in (getattr(r, "g", None), getattr(r, "restore", None)):
            if gp is not None:
                gp.close()
        for a in getattr(r, "own", ()):
            self.mirror.forget(a.ctypes.data)


class _Run:
    """A program of t tokens: its Program, GPUProgram, inputs and outputs."""

    def __init__(self):
        self.cpu = []


class DSHot:
    """The hot experts follow the choices of the router (DEEPSEEK_V41_PLAN.md
    Phase 3): a score for each expert of each layer, times decay at each token
    plus its uses; after a run, at most budget experts a token (at most cap a
    run) come in where the best expert not held beats the worst held by
    margin. A slot leaves first (its map entry -1 before the next run), the
    copy runs on the copy worker (rc_gpu_cache_copy, from the weightkeeper's
    segments) while the GPU goes on, and the expert is held (its map entry)
    from the first run after its copy is done. On the real routes of the
    prompt of 1414 tokens, 16 slots a layer and 4 copies a token held 44% of
    the pairs (all the routes in hindsight, the best 16: 40%)."""

    def __init__(self, g, decay=0.98, budget=4, cap=None, margin=0.5):
        self.g = g
        L, E = g.m.L, g.w.src.E
        # cap: the most copies of a step or a verify group (RECLANK_DS_HOT_CAP,
        # 64); cap_group: of a group of a prompt (RECLANK_DS_HOT_CAP_GROUP, 256:
        # the next group waits for them, about half a second)
        cap = int(cap or os.environ.get("RECLANK_DS_HOT_CAP", "64"))
        # budget: the copies a token (RECLANK_DS_HOT_BUDGET, a measure)
        budget = int(os.environ.get("RECLANK_DS_HOT_BUDGET", budget))
        self.cap_group = int(os.environ.get("RECLANK_DS_HOT_CAP_GROUP", "256"))
        self.decay, self.budget, self.cap, self.margin = decay, budget, cap, margin
        self.score = np.zeros((L, E))
        self.incoming = np.zeros((L, E), bool)
        self.pending = []           # (job, ranges, [(layer, slot, expert)])
        self.copies = 0

    def register(self):
        """Page-lock the weightkeeper's segments for the copies (DMA, no staging
        through the CPU, which the parts need); left as they are when CUDA
        refuses."""
        import sys
        import time
        from reclank.backends.cuda.runtime import lib
        z, t0 = self.g.w.z, time.time()
        # (writable maps, the weightkeeper's seal "size": page-locked as they are)
        self.registered = all((lib().rc_gpu_host_register_rw(b, n) != 0) if getattr(z, "writable", False)
                              else lib().rc_gpu_host_register_ro(b, n) == 0 for b, n in zip(z.base, z.size))
        print("the weightkeeper's segments page-locked for the copies: %s (%.1f s)" % (
            "yes" if self.registered else lib().rc_gpu_last_error().decode(), time.time() - t0), file=sys.stderr)

    def observe(self, ids, t, layers=None):
        """The choices of a run of t tokens (ids [L, t * k]) of its layers."""
        L, E = self.score.shape
        lay = np.arange(L) if layers is None else np.asarray(list(layers))
        self.score[lay] *= self.decay ** t
        flat = (np.asarray(ids, np.int64)[lay] + (lay.astype(np.int64) * E)[:, None]).ravel()
        self.score += np.bincount(flat, minlength=L * E).reshape(L, E)

    def land(self, wait=False):
        from reclank.backends.cuda.runtime import lib
        g, still, rows = self.g, [], set()
        for job, ranges, items in self.pending:
            r = lib().rc_gpu_cache_query(job, 1 if wait else 0)
            if r < 0:
                raise RuntimeError(lib().rc_gpu_last_error().decode())
            if r == 0:
                still.append((job, ranges, items))
                continue
            for i, slot, e in items:
                g.slot_expert[i][slot] = e
                g.maps[i][e] = slot
                self.incoming[i, e] = False
                rows.add(i)
        self.pending = still
        for i in rows:
            g.mirror.buffer_of(g.maps[i]).upload(g.maps[i])

    def plan(self, t):
        """The copies of this run: for each layer with free slots, its best
        experts not held; for each full layer, its best expert not held where
        it beats its worst held one by margin (all the layers at once)."""
        from reclank.backends.cuda.runtime import lib
        g = self.g
        L, E = self.score.shape
        n = g.n_slots
        if n == 0:
            return      # (make_room took every slot; relax gives them back)
        sc = self.score
        held = np.stack([g.maps[i] >= 0 for i in range(L)])
        cand_sc = np.where(held | self.incoming | (sc <= 0), -np.inf, sc)
        se = np.stack([g.slot_expert[i][:n] for i in range(L)])          # -1: free, or a copy on its way
        busy = {(i, s) for _j, _r, items in self.pending for i, s, _e in items}
        cand = []
        has_free = (se < 0).any(1)
        for i in np.nonzero(has_free)[0]:
            free = [s for s in range(n) if se[i, s] < 0 and (i, s) not in busy]
            if not free:
                continue
            order = np.argsort(-cand_sc[i], kind="stable")[:len(free)]
            for e, slot in zip(order, free):
                if np.isfinite(cand_sc[i, e]):
                    cand.append((sc[i, e] + 1e9, int(i), int(e), slot))
        rows_ = np.arange(L)
        best = cand_sc.argmax(1)
        bs = cand_sc[rows_, best]
        held_sc = np.where(se >= 0, sc[rows_[:, None], np.maximum(se, 0)], np.inf)
        worst = held_sc.argmin(1)
        ws = held_sc[rows_, worst]
        gain = bs - ws
        for i in np.nonzero(~has_free & np.isfinite(bs) & np.isfinite(ws) & (gain > self.margin))[0]:
            cand.append((gain[i], int(i), int(best[i]), int(worst[i])))
        cand.sort(reverse=True)
        items, ranges, rows = [], [], set()
        cap = self.cap_group if t > VERIFY_MAX else self.cap
        for _gain, i, e, slot in cand[:min(self.budget * t, cap)]:
            old = int(g.slot_expert[i][slot])
            if old >= 0:
                g.maps[i][old] = -1
                g.slot_expert[i][slot] = -1
                rows.add(i)
            self.incoming[i, e] = True
            items.append((i, slot, e))
            ranges += g.expert_ranges(i, e, slot)
        for i in rows:
            g.mirror.buffer_of(g.maps[i]).upload(g.maps[i])
        if items:
            ra = np.array(ranges, np.int64)
            job = lib().rc_gpu_cache_copy(ra.ctypes.data, len(ranges) // 3)
            if job < 0:
                raise RuntimeError(lib().rc_gpu_last_error().decode())
            self.pending.append((job, ra, items))
            self.copies += len(items)

    def _busy(self, i, slot):
        return any(ii == i and s == slot for _j, _r, items in self.pending for ii, s, _e in items)


_E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float32)


def _e4m3_codes(v):
    """The E4M3 codes of values E4M3 holds exactly (the sign of a zero kept)."""
    from reclank.formats import dsv41_st as D
    pos = D.E4M3[:127]                          # codes 0 .. 126: the values from 0 up
    a = np.abs(v)
    c = np.minimum(np.searchsorted(pos, a), 126)
    assert np.array_equal(pos[c], a), "a value that E4M3 does not hold"
    return (c | np.where(np.signbit(v), 128, 0)).astype(np.uint8)


def _e2m1_codes(v):
    a = np.abs(v)
    c = np.minimum(np.searchsorted(_E2M1, a), 7)
    assert np.array_equal(_E2M1[c], a), "a value that E2M1 does not hold"
    return (c | np.where(np.signbit(v), 8, 0)).astype(np.uint8)


def _block_scale(b, grid, pow2):
    """For each block of values (rows of b), a scale that makes every value
    one of grid (an E4M3 value; pow2: a power of two), from the candidates
    amax / y for the values y of grid (the largest first)."""
    from reclank.formats import dsv41_st as D
    amax = np.abs(b).max(-1)
    sc = np.zeros(amax.shape, np.float32)
    for ym in grid[:0:-1]:
        cand = (amax / ym).astype(np.float32)
        good = np.isin(np.abs(b / np.where(cand > 0, cand, 1)[..., None]), grid).all(-1) & (amax > 0)
        good &= (np.frexp(cand)[0] == 0.5) if pow2 else np.isin(cand, D.E4M3[:127])
        sc = np.where((sc == 0) & good, cand, sc)
    assert ((sc > 0) | (amax == 0)).all(), "a block with no scale"
    return np.where(sc > 0, sc, np.float32(1.0 if pow2 else 2.0 ** -9)).astype(np.float32)


def cache_encode(a, kind):
    """Rows of float32 values in the smallest form of a cache (DSV41GPU,
    kernels/dsv41/step.cu ds_cf): "w" FP8 (E4M3 codes, an E8M0 scale for
    each 32), "c" FP4 (E2M1 codes, an E4M3 scale for each 16), "k" FP4 (an
    E8M0 scale for each 32)."""
    from reclank.formats import dsv41_st as D
    n, hd = a.shape
    if kind == "w":
        b = a.reshape(n, hd // 32, 32)
        amax = np.abs(b).max(-1)
        ex = np.where(amax > 0, np.ceil(np.log2(np.maximum(amax, 1e-38) / 448.0)), 0).astype(np.int64)
        sc = np.ldexp(np.float32(1.0), ex).astype(np.float32)
        codes = _e4m3_codes((b / sc[..., None]).astype(np.float32)).reshape(n, hd)
        return np.concatenate([codes, (ex + 127).astype(np.uint8)], 1)
    bs = 16 if kind == "c" else 32
    b = a.reshape(n, hd // bs, bs)
    sc = _block_scale(b, _E2M1, pow2=kind == "k")
    codes = _e2m1_codes((b / sc[..., None]).astype(np.float32)).reshape(n, hd)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    sb = _e4m3_codes(sc) if kind == "c" else ((sc.view(np.uint32) >> 23) & 255).astype(np.uint8)
    return np.concatenate([packed, sb.reshape(n, -1)], 1)


def cache_decode(a, kind):
    """The float32 values of rows in the smallest form of a cache."""
    from reclank.formats import dsv41_st as D
    n, w = a.shape
    if kind == "w":
        hd = w * 32 // 33
        sc = np.ldexp(np.float32(1.0), a[:, hd:].astype(np.int64) - 127).astype(np.float32)
        return (D.E4M3[a[:, :hd]].reshape(n, hd // 32, 32) * sc[..., None]).reshape(n, hd).astype(np.float32)
    bs = 16 if kind == "c" else 32
    hd = w * 2 * bs // (bs + 2)
    p = a[:, :hd // 2]
    codes = np.empty((n, hd), np.uint8)
    codes[:, 0::2], codes[:, 1::2] = p & 15, p >> 4
    q = _E2M1[codes & 7] * np.where(codes & 8, -1, 1)
    q = np.where((codes & 8) & (codes & 7 == 0), np.float32(-0.0), q).astype(np.float32)
    sb = a[:, hd // 2:]
    sc = D.E4M3[sb] if kind == "c" else np.ldexp(np.float32(1.0), sb.astype(np.int64) - 127).astype(np.float32)
    return (q.reshape(n, hd // bs, bs) * sc[..., None]).reshape(n, hd).astype(np.float32)
