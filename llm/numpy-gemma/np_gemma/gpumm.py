"""The memory of the GPU: one manager for the Gemma and Qwen code.

GpuMem (mem()) knows every block of device memory (Buffer): its kind
(weights, program, cache, pool, lend, other) and its owner. The weights stay;
the other kinds come and go, and when memory runs out the reclaimers that
the models register free it in order of rank (the free-memory segments of
an ExpertPool, then the programs used least recently: ProgramLRU). The C side
(the code and environment of a program, its CUDA graphs, temporary copies)
allocates with cudaMalloc on its own; GpuMem keeps a reserve free for it, so
it never calls back.

    mem().report()           # the bytes of each kind, and the free memory
    Buffer(n, "cache", key)  # a block (it reclaims memory when needed)
    pool = ExpertPool(strides); pool.grow(nbytes, "free")

A Buffer is a handle: GpuMem may move a block (a new address, its bytes
copied, Buffer.ptr changed, on_move told) or its owner may give it back, at
any time, unless it is locked or pinned:

    pinned   its address is in the code of a program (the device copies of
             the arrays of a program, the weights): it stays until freed
    locked   the C side uses it now: a run (GPUProgram.run locks its blocks
             and the blocks of its model, the cache and the pool, and a fence
             unlocks them when the GPU is done: gg_run returns before), or a
             copy of HotCache into a block of the pool (until it lands)

The caches are counted by owner, so that several KV caches of their own
sizes can share the memory later; for now a model holds one at a time.
"""
from __future__ import annotations

import ctypes
import os
import time
import weakref
from collections import OrderedDict, deque

import numpy as np


def _lib():
    from .gpu import lib
    return lib()


def _check(rc):
    if rc != 0:
        from .gpu import where_text
        raise RuntimeError(_lib().gg_last_error().decode() + where_text())


def mem_info():
    """Return the free and the total memory of the GPU, in bytes."""
    f, t = ctypes.c_size_t(), ctypes.c_size_t()
    _check(_lib().gg_mem_info(ctypes.byref(f), ctypes.byref(t)))
    return f.value, t.value


class GpuMem:
    """The memory manager of the GPU (one for the process, mem()).

    Every Buffer is a block of a kind, with an owner:

      weights  the weights of the model, the head, and the buffers of the
               programs made with the model (never freed)
      program  the device copies of the arrays of a later program (freed
               when its owner drops the program)
      cache    a KV cache (owner: the cache; one at a time for now, the
               accounting is by owner so several can come later)
      pool     the segments of an ExpertPool (owner "free" or "lend")
      lend     room held for another user (the image encoder)
      encoder  the device copies of the programs of the media encoders
      other    the rest

    The CUDA calls of the C side (the environment and code of a program, its
    graphs, temporary copies) take memory that this manager does not see.
    They get a reserve: c_reserve bytes (NP_GEMMA_GPU_C_RESERVE, 0.8 GB) that
    the blocks here leave free. An allocation that would go into the reserve
    first asks the reclaimers to free memory, and so does ensure_reserve()
    before the C side allocates (a program loads); the C side never calls
    back. Growth that is not needed (the pool) takes only room().

    A reclaimer is (rank, name, fn): fn(nbytes, keep) frees about nbytes of
    its blocks (not keep: the program being made) and returns the bytes
    freed. The lowest rank goes first: the free-memory segments of the pool,
    then the programs used least recently. Weights and caches have none."""

    KINDS = ("weights", "program", "cache", "pool", "lend", "encoder", "other")

    def __init__(self):
        self.blocks = {}            # id -> Buffer
        self.reclaimers = []        # (rank, name, fn)
        self.fences = deque()       # (fence number, the blocks it unlocks), in order
        self.shared = {}            # owner -> [(address, bytes)]: lent by a cache (share)
        self.placed = {}            # id -> Buffer placed in lent memory
        self.c_reserve = int(float(os.environ.get("NP_GEMMA_GPU_C_RESERVE", "0.8e9")))

    def add_reclaimer(self, rank, name, fn):
        self.reclaimers = sorted([r for r in self.reclaimers if r[1] != name] + [(rank, name, fn)],
                                 key=lambda r: r[0])

    def remove_reclaimer(self, name):
        self.reclaimers = [r for r in self.reclaimers if r[1] != name]

    def reclaim(self, nbytes, keep=None):
        """Free about nbytes with the reclaimers, in order. Return the bytes
        freed."""
        freed = 0
        for _rank, _name, fn in list(self.reclaimers):
            if freed >= nbytes:
                break
            freed += fn(nbytes - freed, keep)
        return freed

    def room(self):
        """The bytes that a growth that is not needed can take."""
        return max(0, mem_info()[0] - self.c_reserve)

    def ensure_reserve(self, keep=None, extra=0):
        """Free memory until c_reserve + extra bytes are free (before the C
        side allocates). Return True when they are."""
        need = self.c_reserve + extra - mem_info()[0]
        if need > 0:
            self.reclaim(need, keep)
        return mem_info()[0] >= self.c_reserve + extra

    def alloc(self, nbytes, reclaim=True, keep=None):
        """A device address of nbytes. With reclaim, first free memory so that
        the block leaves the reserve; then it may still go into the reserve,
        and when cudaMalloc fails it frees more and tries again while the
        reclaimers free some. Raise MemoryError."""
        n = max(int(nbytes), 1)
        if reclaim:
            self.ensure_reserve(keep, n)
        while True:
            ptr = _lib().gg_malloc(n)
            if ptr:
                return ptr
            msg = _lib().gg_last_error().decode()
            _lib().gg_clear_error()          # else the next run reports this error
            if not reclaim or self.reclaim(n, keep) <= 0:
                raise MemoryError(msg)

    def set_kind(self, bufs, kind, owner=None):
        for b in bufs:
            b.kind, b.owner = kind, owner

    # ---- locks and fences

    def lock(self, bufs):
        for b in bufs:
            b.locks += 1

    def unlock(self, bufs):
        for b in bufs:
            b.locks -= 1

    def fence(self, bufs):
        """After work queued on the stream of the programs (a run) that uses
        bufs (locked): they unlock when the GPU is done with it. Return the
        number of the fence."""
        n = _lib().gg_fence()
        if n < 0:
            self.unlock(bufs)
            raise RuntimeError(_lib().gg_last_error().decode())
        self.fences.append((n, list(bufs)))
        return n

    def poll(self, wait=False):
        """Unlock the blocks of the fences that are done (wait: of all)."""
        while self.fences:
            r = _lib().gg_fence_done(self.fences[0][0], 1 if wait else 0)
            if r < 0:
                raise RuntimeError(_lib().gg_last_error().decode())
            if r == 0:
                return
            self.unlock(self.fences.popleft()[1])

    def wait_unlocked(self, bufs):
        """Wait for the fences that hold bufs. Return True when all are
        unlocked (a copy of HotCache holds its own lock)."""
        bufs = list(bufs)
        self.poll()
        while self.fences and any(b.locks for b in bufs):
            r = _lib().gg_fence_done(self.fences[0][0], 1)
            if r < 0:
                raise RuntimeError(_lib().gg_last_error().decode())
            self.unlock(self.fences.popleft()[1])
        return not any(b.locks for b in bufs)

    # ---- memory that a KV cache lends (DeviceCache.share_tail)

    SHARED_ALIGN = 256

    def share(self, owner, regions):
        """owner (a cache) lends regions [(address, bytes)] from now on, in
        place of the ones it lent before. The blocks placed in memory it no
        longer lends go (evict) first: the runs before must be done, and it
        is an error (AssertionError) when one of them is still locked then.
        Return the blocks evicted."""
        new = sorted((int(p), int(n)) for p, n in regions if int(n) >= self.SHARED_ALIGN)
        self.shared_ver = getattr(self, "shared_ver", 0) + 1

        def inside(b):
            return any(p <= b._ptr and b._ptr + b.nbytes <= p + n for p, n in new)
        gone = [b for b in self.placed.values() if b.region == owner and not inside(b)]
        if gone:
            self.poll(wait=True)
            locked = [b for b in gone if b.locks]
            assert not locked, ("the cache %s needs memory it lent, but %d blocks there are still "
                                "locked after the runs (%s)" % (owner, len(locked),
                                                               ", ".join(sorted({b.kind for b in locked}))))
            for b in gone:
                b.evict()
        if new:
            self.shared[owner] = new
        else:
            self.shared.pop(owner, None)
        return gone

    def _gaps(self):
        """The free spans of the lent memory: (address, bytes, owner)."""
        used = sorted((b._ptr, b.nbytes) for b in self.placed.values())
        out = []
        for owner, regions in self.shared.items():
            for p, n in regions:
                cur, end = p, p + n
                for up, un in used:
                    if up + un <= cur or up >= end:
                        continue
                    if up > cur:
                        out.append((cur, up - cur, owner))
                    cur = max(cur, up + un)
                if end > cur:
                    out.append((cur, end - cur, owner))
        return out

    def place_shared(self, nbytes):
        """A place for nbytes in the lent memory: (address, owner), or None."""
        a = self.SHARED_ALIGN
        for p, n, owner in self._gaps():
            q = -(-p // a) * a
            if q + nbytes <= p + n:
                return q, owner
        return None

    def shared_room(self, nbytes):
        """How many blocks of nbytes fit in the lent memory now. The answer
        stays until the lent memory or the blocks placed in it change
        (shared_ver): _gaps took 1.7 ms at each run of a decode step."""
        key = (getattr(self, "shared_ver", 0), int(nbytes))
        if getattr(self, "_room_key", None) == key:
            return self._room_val
        self._room_key, self._room_val = key, self._shared_room(nbytes)
        return self._room_val

    def _shared_room(self, nbytes):
        a = self.SHARED_ALIGN
        k = 0
        for p, n, _o in self._gaps():
            q = -(-p // a) * a
            if q < p + n:
                k += (p + n - q) // (-(-nbytes // a) * a)
        return k

    def move(self, b):
        """Move an unlocked, unpinned block that has on_move to a new address
        (its bytes copied). Return True when it moved."""
        self.poll()
        if b.locks or b.pinned or b.on_move is None or not b._ptr or b.region is not None:
            return False
        try:
            new = self.alloc(b.nbytes, reclaim=False)
        except MemoryError:
            return False
        old = b._ptr
        _check(_lib().gg_d2d(new, old, b.nbytes))
        _lib().gg_free(old)                 # (after the copy: cudaFree waits for the GPU)
        b._ptr = new
        b.on_move(b, old, new)
        return True

    def usage(self):
        """The bytes of each kind, and of each owner of the caches and the
        pool."""
        out = {k: 0 for k in self.KINDS}
        by = {}
        for b in self.blocks.values():
            out[b.kind] = out.get(b.kind, 0) + b.nbytes
            if b.kind in ("cache", "pool"):
                by[(b.kind, b.owner)] = by.get((b.kind, b.owner), 0) + b.nbytes
        return out, by

    def report(self):
        out, by = self.usage()
        free, total = mem_info()
        parts = ["%s %.2f" % (k, v / 1e9) for k, v in out.items() if v]
        own = ["%s:%s %.2f" % (k, o, v / 1e9) for (k, o), v in sorted(by.items(), key=str)]
        return "GPU GB: %s; free %.2f of %.2f (C reserve %.2f)%s" % (
            ", ".join(parts), free / 1e9, total / 1e9, self.c_reserve / 1e9,
            " [" + ", ".join(own) + "]" if own else "")


def weak_method(m):
    """A reclaimer for a bound method that does not keep its object alive
    (0 bytes freed once the object is gone)."""
    ref, fn = weakref.ref(m.__self__), m.__func__

    def call(nbytes, keep):
        obj = ref()
        return 0 if obj is None else fn(obj, nbytes, keep)
    return call


_MEM = None


def mem():
    """The memory manager of the GPU (GpuMem)."""
    global _MEM
    if _MEM is None:
        _MEM = GpuMem()
    return _MEM


class Buffer:
    """A handle of a block of device memory, of a kind and an owner (GpuMem).
    ptr is its address now: it holds while the block is locked or pinned.
    on_move(buf, old, new) is told when GpuMem moves it (None: it does not
    move)."""

    _next = 0

    def __init__(self, nbytes, kind="other", owner=None, reclaim=True, keep=None, pinned=False,
                 on_move=None, shared=False, on_evict=None):
        """shared: "only" places the block in memory that a cache lends
        (GpuMem.share; MemoryError when none fits), True there or else by
        cudaMalloc. on_evict(buf): told before GpuMem takes it away (the cache
        needs its memory back)."""
        self.nbytes = int(nbytes)
        self.kind, self.owner = kind, owner
        self.pinned, self.on_move, self.locks = pinned, on_move, 0
        self.on_evict, self.region = on_evict, None
        self._ptr = None
        if shared:
            at = mem().place_shared(self.nbytes)
            if at is None and shared == "only":
                raise MemoryError("no room in the memory that the caches lend")
            if at is not None:
                self._ptr, self.region = at
        if self._ptr is None:
            self._ptr = mem().alloc(self.nbytes, reclaim, keep)
        Buffer._next += 1
        self.id = Buffer._next
        mem().blocks[self.id] = self
        if self.region is not None:
            mem().placed[self.id] = self
            mem().shared_ver = getattr(mem(), "shared_ver", 0) + 1

    @property
    def ptr(self):
        return self._ptr

    def upload(self, a):
        assert a.flags.c_contiguous and a.nbytes <= self.nbytes
        _check(_lib().gg_h2d(self.ptr, a.ctypes.data, a.nbytes))

    def download(self, a):
        assert a.flags.c_contiguous and a.nbytes <= self.nbytes
        _check(_lib().gg_d2h(a.ctypes.data, self.ptr, a.nbytes))

    def zero(self):
        """All the bytes to 0, on the GPU."""
        _check(_lib().gg_zero(self.ptr, self.nbytes))

    def free(self):
        if self._ptr:
            if self.locks:
                mem().wait_unlocked([self])
            mem().blocks.pop(self.id, None)
            if mem().placed.pop(self.id, None) is not None:
                mem().shared_ver = getattr(mem(), "shared_ver", 0) + 1
            if self.region is None:
                _lib().gg_free(self._ptr)
            self._ptr = None

    def evict(self):
        """GpuMem takes the block away: its owner lets go (on_evict), then it
        is freed."""
        if self.on_evict is not None:
            self.on_evict(self)
        self.free()

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


class DeviceCache:
    """The device copy of a KV cache, for every model type: Buffers of kind
    "cache" (owner: the cache), locked for good (they hold the state; the
    runs read them).

    The arrays with a row for each position (or for each block of positions:
    per_row) are position-major, so the positions in use are a prefix: before
    each run, share_tail(need) lends GpuMem the rest of each of them past
    need + margin (NP_GEMMA_GPU_SHARE_MARGIN positions, 4096), where the pool
    of HotCache can place blocks. The margin moves only when a run needs more
    (or a new cache comes: unshare). GpuMem evicts its blocks from memory the
    cache takes back; one still locked after the runs is an AssertionError.
    NP_GEMMA_GPU_SHARE=0 lends nothing."""

    MARGIN = int(os.environ.get("NP_GEMMA_GPU_SHARE_MARGIN", "4096"))
    ON = os.environ.get("NP_GEMMA_GPU_SHARE", "1") != "0"

    def __init__(self, owner="cache0"):
        self.owner = owner
        # Buffer id -> (Buffer, positions per row, bytes per row, rows, base):
        # base() the position of row 0 (None: 0; the 26B drops old rows)
        self.rows = {}
        self.share_to = 0       # the positions kept; past them the arrays are lent

    def _new(self, nbytes, rows=None, per_row=1, base=None):
        """A locked cache Buffer; rows: its rows when position-major."""
        b = Buffer(nbytes, "cache", self.owner)
        b.locks += 1
        if rows:
            self.rows[b.id] = (b, per_row, nbytes // rows, rows, base)
        return b

    def _free(self, b):
        self.rows.pop(b.id, None)
        b.locks -= 1
        b.free()

    def share_tail(self, need, before=None):
        """Before a run that needs positions below need: lend the rest past
        need + margin (only when need passes the last margin). before(), if
        given, runs first when the lent memory changes (HotCache.prepare:
        its copies into lent blocks must be done)."""
        if not self.ON or need <= self.share_to:
            return
        if before is not None:
            before()
        self.share_to = need + self.MARGIN
        a = GpuMem.SHARED_ALIGN
        regions = []
        for b, per, rb, rows, base in self.rows.values():
            r0 = -(-(self.share_to - (base() if base is not None else 0)) // per)
            if r0 < rows:
                off = -(-(r0 * rb) // a) * a
                if off < b.nbytes:
                    regions.append((b.ptr + off, b.nbytes - off))
        mem().share(self.owner, regions)

    def unshare(self, before=None):
        """Take all the lent memory back (a new cache, a free)."""
        if self.owner in mem().shared and before is not None:
            before()
        mem().share(self.owner, [])
        self.share_to = 0


def _alloc_retry(nbytes, oom=None, kind="other", owner=None):
    """A device Buffer of nbytes (GpuMem reclaims memory for it). When the
    memory still runs out, oom() (if given) frees memory, and it tries again
    while oom() frees some."""
    while True:
        try:
            return Buffer(nbytes, kind, owner)
        except MemoryError:
            if oom is None or not oom():
                raise


def pinned(shape, dtype=np.float32):
    """Return a NumPy array in pinned host memory. A copy between the GPU and
    pinned memory does not wait for the host. The memory is not freed."""
    n = int(np.prod(shape)) * np.dtype(dtype).itemsize
    ptr = _lib().gg_host_alloc(max(n, 1))
    if not ptr:
        raise MemoryError(_lib().gg_last_error().decode())
    raw = (ctypes.c_uint8 * max(n, 1)).from_address(ptr)
    return np.frombuffer(raw, dtype=np.uint8, count=n).view(dtype).reshape(shape)


class ExpertPool:
    """Expert blocks on the GPU that come and go, for HotCache: the warm
    experts of the decode (more than the fixed slots of each layer, given to
    the layers that use them most), the copies of the mixed groups of a
    prompt (reserve), and the memory that a program or an image needs back
    (drop).

    The memory comes in segments of K blocks (a Buffer each), from a source:
    "lend" (the room that the image encoder lends) or "free" (the free memory
    of the GPU). Block i is place i % K of segment i // K; its part p
    (gate, up, down) is at the segment's part p + (i % K) * stride[p]; stride
    is the bytes of an expert of the layers that use the pool. gpu.cu
    (kqh_wslot) and csrc/moe.c (the plan) read the table of the segments.
    owner[i]: -3 no memory, -2 reserved, -1 free, else row * E + expert (the
    HotCache row and expert of the block)."""

    def __init__(self, stride):
        k, seg = ctypes.c_int(), ctypes.c_int()
        _check(_lib().gg_pool_dims(ctypes.byref(k), ctypes.byref(seg)))
        self.K, self.SEG = k.value, seg.value
        self.stride = [int(x) for x in stride]
        self.seg_bytes = self.K * sum(self.stride)
        # the segment addresses of each part, then the strides (gg_set_pool)
        self.table = np.zeros(3 * self.SEG + 3, np.int64)
        self.table[3 * self.SEG:] = self.stride
        self.bufs = [None] * self.SEG
        self.src = [None] * self.SEG
        self.owner = np.full(self.K * self.SEG, -3, np.int64)
        self.evict_hook = None      # HotCache.pool_evict (set_pool)

    def _upload(self):
        _check(_lib().gg_set_pool(self.table.ctypes.data))

    def addr(self, part, i):
        """The device address of part of block i."""
        return int(self.table[part * self.SEG + i // self.K]) + (i % self.K) * self.stride[part]

    def nbytes(self, src=None):
        return self.seg_bytes * sum(1 for x in self.src if x is not None and src in (None, x))

    def grow(self, nbytes, src):
        """Add segments of up to nbytes (as many as fit, until the memory
        runs out). Return the bytes added."""
        added = 0
        for sg in range(self.SEG):
            if added + self.seg_bytes > nbytes:
                break
            if self.bufs[sg] is not None:
                continue
            try:
                if src == "shared":
                    # in the memory a cache lends; it goes when the cache
                    # takes it back (_evicted)
                    b = Buffer(self.seg_bytes, "pool", src, reclaim=False, shared="only",
                               on_evict=self._evicted)
                else:
                    b = Buffer(self.seg_bytes, "pool", src, reclaim=False, on_move=self._moved)
            except MemoryError:
                break
            b.seg = sg
            self.bufs[sg], self.src[sg] = b, src
            off = 0
            for p in range(3):
                self.table[p * self.SEG + sg] = b.ptr + off
                off += self.K * self.stride[p]
            self.owner[sg * self.K:(sg + 1) * self.K] = -1
            added += self.seg_bytes
        if added:
            self._upload()
        return added

    def _evicted(self, b):
        """GpuMem takes segment b.seg back (its cache needs the memory): its
        warm experts go cold (evict_hook: HotCache.pool_evict), the table
        forgets it."""
        sg = b.seg
        blocks = np.arange(sg * self.K, (sg + 1) * self.K)
        if self.evict_hook is not None:
            self.evict_hook(blocks)
        self.bufs[sg], self.src[sg] = None, None
        self.table[[p * self.SEG + sg for p in range(3)]] = 0
        self.owner[blocks] = -3
        self._upload()

    def _moved(self, b, old, new):
        """GpuMem moved segment b.seg: the table follows (no run uses the pool
        then: a run locks all its segments)."""
        off = 0
        for p in range(3):
            self.table[p * self.SEG + b.seg] = new + off
            off += self.K * self.stride[p]
        self._upload()

    def buffers(self):
        """The Buffers of the segments (a run locks them all)."""
        return [b for b in self.bufs if b is not None]

    def segments(self, src=None):
        return [sg for sg in range(self.SEG) if self.src[sg] is not None and src in (None, self.src[sg])]

    def drop(self, segs):
        """Free segments whose blocks no slot uses (HotCache.pool_drop)."""
        if not segs:
            return
        mem().wait_unlocked([self.bufs[sg] for sg in segs])
        for sg in segs:
            self.bufs[sg].free()
            self.bufs[sg], self.src[sg] = None, None
            self.table[[p * self.SEG + sg for p in range(3)]] = 0
            self.owner[sg * self.K:(sg + 1) * self.K] = -3
        self._upload()

    def close(self):
        self.drop(self.segments())


class ProgramLRU:
    """The programs of a GPU model that come and go, in the order of their
    use: the dicts named by _program_dicts (key -> (Program, GPUProgram);
    "groups" by default). Each size keeps its own buffers, so a server that
    sees prompts of many sizes fills the GPU with them (a 12B server ran out
    of memory at a prompt of 21674 tokens). GpuMem then frees the programs
    used least recently (_reclaim_programs, rank 20). The step program
    (self.prog) and the program used last (logits() reads its output) stay.

    An entry of the order is (dict name, key); _used(key) and _evict(key)
    take a key of "groups" too. _mm_init(name) starts the order and the
    reclaimer; _mm_freeze() makes all the device copies of the model so far
    weights (they stay)."""

    def _program_dicts(self):
        return ("groups",)

    # A program not used for STALE seconds goes before the hot experts of a
    # pool (rank 15, qwen_gpu); the others after them. A long prompt with
    # MTP runs the mixed program and the MTP program in turn: when only the
    # programs gave memory, each closed the other at each group of 4096
    # (1.5 s to make the mixed one again) once the cache took back the
    # memory it lent to the pool (past 70K positions).
    STALE = float(os.environ.get("NP_GEMMA_GPU_PROGRAM_STALE", "30"))

    def _mm_init(self, name):
        self._lru = OrderedDict()
        self._last_use = {}
        self._building = None
        self._mm_name = name
        mem().add_reclaimer(12, name + "-stale", weak_method(self._reclaim_stale))
        mem().add_reclaimer(20, name + "-programs", weak_method(self._reclaim_programs))

    def _lru_init(self):
        self._mm_init("%s-%x" % (type(self).__name__, id(self)))

    def _mm_freeze(self):
        mir = self.g.mirror
        mem().set_kind(list(mir.bufs.values()), "weights")
        mir.kind = "program"
        self.g.run_blocks = lambda: self._run_blocks()     # (an instance may set its own)

    def _run_blocks(self):
        """The blocks of the model that each run uses (its cache, its pool):
        GPUProgram.run locks them."""
        return []

    def _mm_close(self):
        mem().remove_reclaimer(self._mm_name + "-stale")
        mem().remove_reclaimer(self._mm_name + "-programs")

    def _entry(self, key, name="groups"):
        if isinstance(key, tuple) and len(key) == 2 and key[0] in self._program_dicts():
            return key
        return (name, key)

    def _used(self, key, name="groups"):
        e = self._entry(key, name)
        self._lru[e] = None
        self._lru.move_to_end(e)
        self._last_use[e] = time.monotonic()

    def _build_program(self, prog, key, name="groups", **kw):
        """The GPUProgram of prog on the weights of the model, with its graphs
        recorded (their first launch takes memory too). The C side takes the
        reserve of GpuMem, made first; the device copies of the arrays of the
        program reclaim memory as they come (not those of this program)."""
        from .gpu import GPUProgram
        entry = self._entry(key, name)
        mir = self.g.mirror
        self._building, mir.keep = prog, entry
        try:
            while True:
                mem().ensure_reserve(entry)
                g = None
                try:
                    g = GPUProgram(prog, graph=self.graph, mirror=mir, **kw)
                    g.label = g.lru_key = entry
                    g.run_blocks = lambda: self._run_blocks()
                    g.prepare()
                    return g
                except MemoryError:
                    if g is not None:
                        g.close()
                        for b in g.named.values():
                            b.free()
                    if mem().reclaim(mem().c_reserve, entry) <= 0 and \
                            not self._evict(entry, last=True):
                        raise
        finally:
            self._building, mir.keep = None, None

    def _evict(self, keep=None, last=False):
        """Free the program used least recently (not keep). The last one used
        stays too (logits() reads its output), unless last is True: a new
        program runs before anything reads an output again. Return False
        when there is none."""
        keep = None if keep is None else self._entry(keep)
        running = getattr(self, "_running", None)       # bound for a run (QwenGPU.mix)
        for e in list(self._lru)[:None if last else -1]:
            if e == keep or e == running:
                continue
            del self._lru[e]
            name, key = e
            d = getattr(self, name, {})
            if key not in d:
                continue
            self._free_program(*d.pop(key))
            return True
        return False

    def _reclaim_stale(self, nbytes, keep):
        """GpuMem: free the programs not used for STALE seconds (not keep),
        least recent first, until about nbytes are free. Return the bytes
        freed."""
        f0 = mem_info()[0]
        now = time.monotonic()
        keep = None if keep is None else self._entry(keep)
        running = getattr(self, "_running", None)
        for e in list(self._lru)[:-1]:
            if mem_info()[0] - f0 >= nbytes:
                break
            if e == keep or e == running or now - self._last_use.get(e, 0.0) < self.STALE:
                continue
            del self._lru[e]
            name, key = e
            d = getattr(self, name, {})
            if key in d:
                self._free_program(*d.pop(key))
        return max(0, mem_info()[0] - f0)

    def _reclaim_programs(self, nbytes, keep):
        """GpuMem: free the programs used least recently (not keep) until
        about nbytes are free. Return the bytes freed."""
        f0 = mem_info()[0]
        while mem_info()[0] - f0 < nbytes and self._evict(keep):
            pass
        return max(0, mem_info()[0] - f0)

    def _live_programs(self):
        """The programs whose arrays must stay on the GPU."""
        out = [self.prog]
        for name in self._program_dicts():
            out += [p for p, _g in getattr(self, name, {}).values()]
        out += [p for p, _g in getattr(self, "kq_heads", {}).values()]
        out += list(getattr(self, "extra_progs", []))
        if getattr(self, "_building", None) is not None:
            out.append(self._building)          # a program being made
        return out

    def _pinned_arrays(self):
        """Arrays whose device copies stay though no live program holds them
        (the stores of the experts of qwen_gpu)."""
        return []

    def _free_program(self, prog, g):
        """Close a program and free its buffers, and the device copies of the
        arrays that no live program has (the weights stay), after its runs."""
        if getattr(g, "fence", 0):
            _lib().gg_fence_done(g.fence, 1)
        g.close()
        for b in g.named.values():
            b.free()

        def base(a):
            while isinstance(a.base, np.ndarray):
                a = a.base
            return a.ctypes.data

        def starts(p):
            return {base(a) for a in p.keep if isinstance(a, np.ndarray)}
        live = {base(a) for a in self._pinned_arrays()}
        for p in self._live_programs():
            live |= starts(p)
        mir = self.g.mirror
        for st in starts(prog) - live:
            mir.forget(st)
