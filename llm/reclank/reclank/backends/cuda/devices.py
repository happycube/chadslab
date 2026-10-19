"""The GPUs of the machine and their roles.

The process runs its steps on one GPU (the primary: runtime.lib(), the
device RECLANK_GPU_DEVICE) and may give the experts it lacks to other GPUs
of the same process (expert devices). The two cards of a box: the RTX 3060
(12 GB: the dense weights, a cache of 256K tokens, its HotCache) and the GTX
1070 next to it (8 GB of experts).

An expert device has a copy of the CUDA library of its own (its arch's build,
and its own state: device, stream, buffers; the CUDA runtime is linked into
each copy), loaded with RTLD_LOCAL beside the primary's. Its experts go up
once (rc_gpu_ex_store); the CPU part of each step calls rc_gpu_ex_post and
rc_gpu_ex_wait of that copy through function pointers (GP_REMOTE_POST and
GP_REMOTE_JOIN, an er_ctx): the cold experts it holds run there while the CPU
runs the others, with no process between them. The GPUs do not share memory
(GeForce cards have no peer access): a job crosses host memory, the row of a
layer each way.

NUMA: each GPU's node comes from sysfs. An expert device's pinned buffers are
made by a thread on its node (first touch), so on a machine of two sockets
its jobs cross from that node.

    RECLANK_EXPERT_GPUS   CUDA indices of the expert devices (comma list; as
                          CUDA counts with CUDA_DEVICE_ORDER=PCI_BUS_ID)
    RECLANK_EXPERT_SKIP   the most used experts the primary keeps (2000):
                          an expert device takes those after
    RECLANK_EXPERT_GB     the memory of an expert device for experts
                          (default: its free memory less 0.8 GB)

Nemotron 3.5 Bolt (NmExpertDevice, nm_expert_device): the same, the experts g's hot set lacks.
"""
from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import time

import numpy as np


def gpus():
    """The GPUs that CUDA sees (CUDA_VISIBLE_DEVICES, PCI order), each a dict:
    cuda (the CUDA index), id (nvidia-smi's index), bus, name, mem (bytes),
    arch (61 for 6.1), numa (the node, or -1)."""
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,pci.bus_id,name,memory.total,compute_cap",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = []
    for line in out.strip().splitlines():
        idx, bus, name, mem, cc = [v.strip() for v in line.split(",")]
        major, minor = cc.split(".")
        sysbus = bus.lower()[-12:]          # 00000000:17:00.0 -> 0000:17:00.0
        try:
            with open("/sys/bus/pci/devices/%s/numa_node" % sysbus) as f:
                node = int(f.read().strip())
        except (OSError, ValueError):
            node = -1
        rows.append(dict(id=int(idx), bus=bus, name=name, mem=int(mem) << 20, arch=int(major) * 10 + int(minor),
                         numa=node))
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if vis:
        by_id = {str(r["id"]): r for r in rows}
        rows = [by_id[v] for v in vis if v in by_id]
    for k, r in enumerate(rows):
        r["cuda"] = k
    return rows


def prebuild(primary=False):
    """Start the builds of the libraries of the expert devices (RECLANK_EXPERT_GPUS, the archs
    other than the primary's), and with primary the primary's, in threads (build.prebuild): from
    backends/cpu/cops.py before the CPU library's build, and from runtime.lib before the primary's.
    On the 12 threads of the W-2133 the front ends and the host compilers of the builds overlap the
    others' parallel parts (each nvcc -split-compile)."""
    from reclank.backends.cuda import build
    try:
        own = build.device_arch()
    except Exception:
        own = None
    archs = [own] if primary and own else []
    want = os.environ.get("RECLANK_EXPERT_GPUS", "").strip()
    if want:
        try:
            rows = {r["cuda"]: r for r in gpus()}
        except Exception:       # (no nvidia-smi: the expert device reports it)
            rows = {}
        for v in want.split(","):
            g = rows.get(int(v)) if v.strip().isdigit() else None
            if g is not None and (g["arch"], "") != own:
                archs.append((g["arch"], ""))
    build.prebuild(archs)


def _lib_copy(arch, primary_path):
    """The path of a build of arch for a second copy in this process: the
    build, or a copy of it under another name when the primary loaded the same
    file (dlopen gives the loaded copy for the same path)."""
    from reclank.backends.cuda import build
    path = str(build.build((arch, "")))
    if os.path.realpath(path) == os.path.realpath(str(primary_path)):
        dst = "/dev/shm/reclank-%d-%s" % (os.getpid(), os.path.basename(path))
        shutil.copyfile(path, dst)
        path = dst
    return path


class ErCtx(ctypes.Structure):
    """er_ctx of backends/cpu/csrc/common/ops_qwen.c."""
    _fields_ = [("post", ctypes.c_void_p), ("wait", ctypes.c_void_p), ("held", ctypes.c_void_p),
                ("experts", ctypes.c_int64), ("alive", ctypes.c_int64), ("post_t", ctypes.c_void_p)]


def _put_api(lib):
    vp, i = ctypes.c_void_p, ctypes.c_int
    lib.rc_gpu_ex_put.argtypes = [i, i, vp, vp, vp, vp, i]


def _refresh(dev, score, busy):
    """An expert device's experts anew (between prompts: no job may run): in each layer of dev.slots
    (expert -> slot or -1), its count of slots of the best experts by score (layer: a score for each
    expert, higher first) that busy (layer: a mask, the primary's) leaves; those that come take the
    slots of those that leave (rc_gpu_ex_put, the parts from dev.srcs(layer, expert) of dev.pers(layer)
    bytes). dev.held changes in place (the CPU's GP_*REMOTE_POST reads it). Return (experts copied,
    bytes)."""
    n = nbytes = 0
    for li, slots in dev.slots.items():
        E = slots.size
        have = slots >= 0
        sc = np.asarray(score[li][:E], np.float64) if li in score else np.zeros(E)
        if li in busy:
            sc = np.where(np.asarray(busy[li][:E], bool), -np.inf, sc)
        want = np.zeros(E, bool)
        # (the experts here first among equals: no copy for a tie)
        want[np.lexsort((~have, -sc))[:int(have.sum())]] = True
        come, go = np.flatnonzero(want & ~have), np.flatnonzero(have & ~want)
        if not len(come):
            continue
        free = slots[go].copy()
        slots[go] = -1
        slots[come] = free
        src = np.array([dev.srcs(li, int(e)) for e in come], np.int64)
        per = np.array(dev.pers(li), np.uint64)
        dev.check(dev.lib.rc_gpu_ex_put(li, len(come), free.ctypes.data, src.ctypes.data, per.ctypes.data,
                                        slots.ctypes.data, E))
        dev.held[li, :E] = want
        n += len(come)
        nbytes += len(come) * int(per.sum())
    return n, nbytes


REFRESH = os.environ.get("RECLANK_EXPERT_REFRESH", "1") != "0"
# the routing before a refresh weighs half for each REFRESH_HALF tokens of the routing since
REFRESH_HALF = float(os.environ.get("RECLANK_EXPERT_REFRESH_HALF", "128"))


def refresh_hot(xdev, hc, state):
    """Between prompts: the experts of expert device xdev anew by the routing that HotCache hc saw
    since the last call (hc.routes: the steps' selections, the expert device's too, and the prompts'
    GP_COUNT) and before it (half for each REFRESH_HALF tokens since; state keeps the score), among the
    experts that hc's slots do not hold or take; then hc.exclude follows xdev.held. Return (experts
    copied, bytes, seconds), or None."""
    if xdev is None or hc is None or not REFRESH or not hasattr(xdev, "slots"):
        return None
    new = hc.routes.astype(np.float64)
    hc.routes[:] = 0
    if not new.any():
        return None
    tokens = new.sum() / (hc.top_k * new.shape[0])
    old = state.get("score")
    state["score"] = new if old is None else 0.5 ** (tokens / REFRESH_HALF) * old + new
    rows = [e["layer"] for e in hc.layers]
    score = {li: state["score"][r] for r, li in enumerate(rows)}
    busy = {li: hc.held[r] | hc.incoming[r] for r, li in enumerate(rows)}
    t0 = time.time()
    n, nbytes = _refresh(xdev, score, busy)
    if hc.exclude is not None:
        hc.exclude[:] = xdev.held[rows].astype(bool)
    return n, nbytes, time.time() - t0


def idle_refresh(gpu, tag):
    """An engine's idle (server/worker.py: no job waits): gpu.refresh_experts(), and a line of it and
    of the steps' pairs since the last call that no GPU held (HotCache's counts)."""
    refresh = getattr(gpu, "refresh_experts", None)
    if refresh is None:
        return
    hc = getattr(gpu, "hot_cache", None)
    cpu = "" if hc is None or not hc.pairs else "; the pairs on no GPU %.1f%%" % (100.0 * hc.cpu_pairs / hc.pairs)
    if hc is not None:
        hc.pairs = hc.cpu_pairs = 0
    r = refresh()
    if r is not None and r[0]:
        print("[%s] the expert GPU's experts follow the text: %d new (%.2f GB) in %.2f s%s" % (
            tag, r[0], r[1] / 1e9, r[2], cpu), file=sys.stderr, flush=True)
    elif cpu:
        print("[%s] %s" % (tag, cpu[2:]), file=sys.stderr, flush=True)


class ExpertDevice:
    """A GPU of experts for the steps of model (see the module text)."""

    def __init__(self, model, gpu, skip=2000, gb=None):
        from reclank.backends.cuda import runtime
        from reclank.models.gemma4.gpu import hot_counts
        from reclank.runtime import numa
        self.gpu = gpu
        cfg = model.cfg
        L, E = cfg.num_hidden_layers, cfg.num_experts
        inner, hidden = cfg.moe_intermediate_size, cfg.hidden_size
        primary = runtime.lib()
        path = _lib_copy(gpu["arch"], primary._name)
        lib = self.lib = ctypes.CDLL(path, mode=os.RTLD_NOW | getattr(os, "RTLD_LOCAL", 0))
        vp, i, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        lib.rc_gpu_last_error.restype = ctypes.c_char_p
        lib.rc_gpu_ex_open.argtypes = [i, ctypes.POINTER(sz)]
        lib.rc_gpu_ex_init.argtypes = [i, i, i, i, sz, sz]
        lib.rc_gpu_ex_store.argtypes = [i, i, vp, vp, vp, i]
        lib.rc_gpu_ex_post.argtypes = [i, i, vp, vp, vp]
        lib.rc_gpu_ex_wait.argtypes = [vp]
        _put_api(lib)

        def check(rc):
            if rc != 0:
                raise RuntimeError("expert device %s: %s" % (gpu["name"], lib.rc_gpu_last_error().decode()))

        self.model, self.check = model, check
        self.slots = {}             # layer: expert -> its slot here, or -1 (refresh)
        # the pinned buffers on the GPU's node (first touch by a thread there)
        old = os.sched_getaffinity(0)
        if gpu["numa"] >= 0 and numa.enabled():
            os.sched_setaffinity(0, numa.node_cpus(gpu["numa"]))
        try:
            free = sz()
            check(lib.rc_gpu_ex_open(gpu["cuda"], ctypes.byref(free)))
            w = model._layers[0]
            gu0, dn0 = w["experts.gate_up_proj"][0], w["experts.down_proj"][0]
            gu_per, dn_per = gu0[0].nbytes, dn0[0].nbytes
            self._per = (gu_per, 0, dn_per)
            dn_rows = dn0.shape[1]
            check(lib.rc_gpu_ex_init(L, hidden, inner, dn_rows, gu_per, dn_per))
            budget = gb * 1e9 if gb else free.value - 0.8e9
            counts = hot_counts(model)
            flat = np.asarray(counts, dtype=np.float64).reshape(-1)
            order = np.argsort(-flat, kind="stable")
            n = int(budget // (gu_per + dn_per))
            held = np.zeros(flat.size, dtype=np.uint8)
            held[order[skip:skip + n]] = 1
            self.held = held.reshape(L, E)
            for layer in range(L):
                mine = np.flatnonzero(self.held[layer])
                if not len(mine):
                    continue
                lw = model._layers[layer]
                gu = np.ascontiguousarray(lw["experts.gate_up_proj"][0][mine])
                dn = np.ascontiguousarray(lw["experts.down_proj"][0][mine])
                slots = np.full(E, -1, dtype=np.int32)
                slots[mine] = np.arange(len(mine), dtype=np.int32)
                self.slots[layer] = slots
                check(lib.rc_gpu_ex_store(layer, len(mine), gu.ctypes.data, dn.ctypes.data, slots.ctypes.data, E))
        finally:
            os.sched_setaffinity(0, old)
        self.ctx = ErCtx(ctypes.cast(lib.rc_gpu_ex_post, ctypes.c_void_p).value,
                         ctypes.cast(lib.rc_gpu_ex_wait, ctypes.c_void_p).value,
                         self.held.ctypes.data, E, 1,
                         ctypes.cast(lib.rc_gpu_ex_post_t, ctypes.c_void_p).value)
        # the operand of GP_REMOTE_POST and GP_REMOTE_JOIN: the bytes of ctx
        self.ctx_bytes = np.frombuffer(ctypes.string_at(ctypes.addressof(self.ctx), ctypes.sizeof(self.ctx)),
                                       dtype=np.uint8).copy()
        print("expert device %d (%s, sm_%d, NUMA node %d): %d experts (%.2f GB) after the first %d" % (
            gpu["cuda"], gpu["name"], gpu["arch"], gpu["numa"], int(self.held.sum()),
            self.held.sum() * (gu_per + dn_per) / 1e9, skip), file=sys.stderr, flush=True)

    def srcs(self, layer, e):
        lw = self.model._layers[layer]
        return (lw["experts.gate_up_proj"][0][e].ctypes.data, 0, lw["experts.down_proj"][0][e].ctypes.data)

    def pers(self, layer):
        return self._per

    def refresh(self, score, busy):
        return _refresh(self, score, busy)


class NmExpertDevice:
    """A GPU of experts for the steps of NemotronGPU g (models/nemotron/gpu.py): of the experts
    g lacks, the most used for their bytes (the routing profile, all layers together) that its
    memory holds, as g's hot stacks (IQ2_XXS by slot). A step's CPU task posts the cold pairs it
    holds (GP_NM_REMOTE_POST: a step, or a group of up to 8 tokens, a verify group, at once) and
    adds its sum after GP_NM_MOE (GP_REMOTE_JOIN)."""

    def __init__(self, g, gpu, gb=None):
        from reclank.backends.cuda import runtime
        from reclank.runtime import numa
        self.gpu = gpu
        cfg = g.cfg
        primary = runtime.lib()
        path = _lib_copy(gpu["arch"], primary._name)
        lib = self.lib = ctypes.CDLL(path, mode=os.RTLD_NOW | getattr(os, "RTLD_LOCAL", 0))
        vp, i, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        lib.rc_gpu_last_error.restype = ctypes.c_char_p
        lib.rc_gpu_ex_open.argtypes = [i, ctypes.POINTER(sz)]
        lib.rc_gpu_ex_nm_init.argtypes = [i, i, i, i]
        lib.rc_gpu_ex_nm_store.argtypes = [i, i, vp, vp, sz, sz, i, vp, i]
        _put_api(lib)

        def check(rc):
            if rc != 0:
                raise RuntimeError("expert device %s: %s" % (gpu["name"], lib.rc_gpu_last_error().decode()))

        Em = max(E for E, _n, _k in cfg.moe.values())
        self.g, self.check = g, check
        self.slots = {}             # layer: expert -> its slot here, or -1
        old = os.sched_getaffinity(0)
        if gpu["numa"] >= 0 and numa.enabled():
            os.sched_setaffinity(0, numa.node_cpus(gpu["numa"]))
        try:
            free = sz()
            check(lib.rc_gpu_ex_open(gpu["cuda"], ctypes.byref(free)))
            kmax = max(k for _E, _n, k in cfg.moe.values())
            imax = max(n for _E, n, _k in cfg.moe.values())
            check(lib.rc_gpu_ex_nm_init(cfg.layers, cfg.latent, kmax, imax))
            budget = gb * 1e9 if gb else free.value - 0.8e9
            # the candidates: each cold expert's count for its bytes
            per, cand = {}, []
            for li, (E, _inner, _k) in cfg.moe.items():
                upm, dnm = g.m.experts(li, "up"), g.m.experts(li, "down")
                per[li] = (upm.data.nbytes // E, dnm.data.nbytes // E)
                c = np.asarray(g.counts[li], np.float64) if g.counts and li in g.counts else np.ones(E)
                hot = g.hot[li][0] >= 0
                b = per[li][0] + per[li][1]
                cand += [(c[e] / b, li, e) for e in range(E) if not hot[e]]
            cand.sort(key=lambda z: -z[0])
            held = np.zeros((cfg.layers, Em), np.uint8)
            used = 0
            for _score, li, e in cand:
                b = per[li][0] + per[li][1]
                if used + b > budget:
                    continue
                held[li, e] = 1
                used += b
            for li, (E, inner, _k) in cfg.moe.items():
                mine = np.flatnonzero(held[li, :E])
                if not len(mine):
                    continue
                upm, dnm = g.m.experts(li, "up"), g.m.experts(li, "down")
                pu, pd = per[li]
                up = np.concatenate([upm.data[e * pu:(e + 1) * pu] for e in mine])
                dn = np.concatenate([dnm.data[e * pd:(e + 1) * pd] for e in mine])
                slots = np.full(E, -1, np.int32)
                slots[mine] = np.arange(len(mine), dtype=np.int32)
                self.slots[li] = slots
                check(lib.rc_gpu_ex_nm_store(li, len(mine), up.ctypes.data, dn.ctypes.data, pu, pd, inner,
                                             slots.ctypes.data, E))
        finally:
            os.sched_setaffinity(0, old)
        self.held = held
        self.ctx = ErCtx(ctypes.cast(lib.rc_gpu_ex_post, ctypes.c_void_p).value,
                         ctypes.cast(lib.rc_gpu_ex_wait, ctypes.c_void_p).value,
                         self.held.ctypes.data, Em, 1,
                         ctypes.cast(lib.rc_gpu_ex_nm_post_t, ctypes.c_void_p).value)
        self.ctx_bytes = np.frombuffer(ctypes.string_at(ctypes.addressof(self.ctx), ctypes.sizeof(self.ctx)),
                                       dtype=np.uint8).copy()
        n_all = sum(E for E, _n, _k in cfg.moe.values())
        self.share = float(held.sum()) / n_all
        self.bytes = used
        print("expert device %d (%s, sm_%d, NUMA node %d): %d experts (%.2f GB, %d%% of all) after the hot %d%%" % (
            gpu["cuda"], gpu["name"], gpu["arch"], gpu["numa"], int(held.sum()), used / 1e9,
            round(100 * self.share), round(100 * g.hot_share)), file=sys.stderr, flush=True)

    def srcs(self, li, e):
        upm, dnm = self.g.m.experts(li, "up"), self.g.m.experts(li, "down")
        E = self.slots[li].size
        return (upm.data.ctypes.data + e * (upm.data.nbytes // E), 0, dnm.data.ctypes.data + e * (dnm.data.nbytes // E))

    def pers(self, li):
        upm, dnm = self.g.m.experts(li, "up"), self.g.m.experts(li, "down")
        E = self.slots[li].size
        return (upm.data.nbytes // E, 0, dnm.data.nbytes // E)

    def refresh(self, score, busy=None):
        """The experts here anew by score (_refresh), among those g does not hold (its hot stacks)."""
        if busy is None:
            busy = {li: self.g.hot[li][0] >= 0 for li in self.slots}
        return _refresh(self, score, busy)


class QwenExpertDevice:
    """A GPU of experts for the steps of QwenGPU g (models/qwen/gpu35.py; K-quant experts): of the
    experts that the routing profile ranks after the first skip (the primary's HotCache takes those:
    RECLANK_EXPERT_SKIP, else the blocks of its pool) and not in its first hot set, the most used that
    the GPU holds. A step's CPU task posts the cold experts it holds (GP_REMOTE_POST; a verify group
    GP_NM_REMOTE_POST) and adds its sum after GP_KQ_MOE (GP_REMOTE_JOIN); HotCache leaves them out."""

    def __init__(self, g, gpu, gb=None, skip=None):
        from reclank.backends.cuda import runtime
        from reclank.models.qwen.gpu35 import _host_bytes
        from reclank.runtime import numa
        self.gpu = gpu
        cfg = g.cfg
        E = cfg.num_experts
        layers = sorted(g.stores)
        primary = runtime.lib()
        path = _lib_copy(gpu["arch"], primary._name)
        lib = self.lib = ctypes.CDLL(path, mode=os.RTLD_NOW | getattr(os, "RTLD_LOCAL", 0))
        vp, i, sz = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        lib.rc_gpu_last_error.restype = ctypes.c_char_p
        lib.rc_gpu_ex_open.argtypes = [i, ctypes.POINTER(sz)]
        lib.rc_gpu_ex_kq_init.argtypes = [i, i, i, i]
        lib.rc_gpu_ex_kq_store.argtypes = [i, i, vp, vp, vp, sz, sz, sz, i, i, vp, i]
        _put_api(lib)

        def check(rc):
            if rc != 0:
                raise RuntimeError("expert device %s: %s" % (gpu["name"], lib.rc_gpu_last_error().decode()))

        self.g, self.check = g, check
        self.slots = {}             # layer: expert -> its slot here, or -1 (refresh)

        nl = max(layers) + 1
        old = os.sched_getaffinity(0)
        if gpu["numa"] >= 0 and numa.enabled():
            os.sched_setaffinity(0, numa.node_cpus(gpu["numa"]))
        try:
            free = sz()
            check(lib.rc_gpu_ex_open(gpu["cuda"], ctypes.byref(free)))
            check(lib.rc_gpu_ex_kq_init(nl, cfg.hidden_size, cfg.moe_inter, cfg.top_k))
            budget = gb * 1e9 if gb else free.value - 0.8e9
            nbs = self._nbs = {li: [g._expert_nb(m) for m in g.stores[li]["mats"]] for li in layers}
            per = {li: sum(nbs[li]) for li in layers}
            counts = getattr(g, "profile", None)
            flat = []
            for li in layers:
                c = np.asarray(counts[li], np.float64) if counts is not None else np.ones(E)
                init = set(int(x) for x in g.stores[li]["init"])
                flat += [(c[e], li, e) for e in range(E) if e not in init]
            flat.sort(key=lambda z: -z[0])
            if skip is None:
                # (the experts the primary's pool will hold: its free memory now, less 1.5 GB for its
                # programs and cache, at the bytes of an expert)
                skip = int(os.environ.get("RECLANK_EXPERT_SKIP", "0") or 0) or int(
                    max(0.0, runtime.mem_info()[0] - 1.5e9) / np.mean(list(per.values())))
            held = np.zeros((nl, E), np.uint8)
            used = 0
            for _c, li, e in flat[skip:]:
                if used + per[li] > budget:
                    continue
                held[li, e] = 1
                used += per[li]
            for li in layers:
                mine = np.flatnonzero(held[li])
                if not len(mine):
                    continue
                st = g.stores[li]
                blocks = []
                for (name, m), nb in zip(zip(("gate_proj", "up_proj", "down_proj"), st["mats"]), nbs[li]):
                    srcs = g._expert_srcs(li, name, m, nb)
                    blocks.append(np.concatenate([_host_bytes(int(srcs[x]), nb) for x in mine]))
                slots = np.full(E, -1, np.int32)
                slots[mine] = np.arange(len(mine), dtype=np.int32)
                self.slots[li] = slots
                gm, _um, dm = st["mats"]
                check(lib.rc_gpu_ex_kq_store(li, len(mine), blocks[0].ctypes.data, blocks[1].ctypes.data,
                                             blocks[2].ctypes.data, nbs[li][0], nbs[li][1], nbs[li][2],
                                             int(gm.type), int(dm.type), slots.ctypes.data, E))
        finally:
            os.sched_setaffinity(0, old)
        self.held = held
        self.ctx = ErCtx(ctypes.cast(lib.rc_gpu_ex_post, ctypes.c_void_p).value,
                         ctypes.cast(lib.rc_gpu_ex_wait, ctypes.c_void_p).value,
                         self.held.ctypes.data, E, 1,
                         ctypes.cast(lib.rc_gpu_ex_post_t, ctypes.c_void_p).value)
        self.ctx_bytes = np.frombuffer(ctypes.string_at(ctypes.addressof(self.ctx), ctypes.sizeof(self.ctx)),
                                       dtype=np.uint8).copy()
        print("expert device %d (%s, sm_%d, NUMA node %d): %d experts (%.2f GB) after the first %d" % (
            gpu["cuda"], gpu["name"], gpu["arch"], gpu["numa"], int(held.sum()), used / 1e9, skip),
            file=sys.stderr, flush=True)

    def srcs(self, li, e):
        st = self.g.stores[li]
        return tuple(int(self.g._expert_srcs(li, name, m, nb)[e]) for name, m, nb in
                     zip(("gate_proj", "up_proj", "down_proj"), st["mats"], self._nbs[li]))

    def pers(self, li):
        return tuple(self._nbs[li])

    def refresh(self, score, busy):
        return _refresh(self, score, busy)


def qwen_expert_device(g):
    """The expert device of QwenGPU g (RECLANK_EXPERT_GPUS: the first of them), or None."""
    want = os.environ.get("RECLANK_EXPERT_GPUS", "").strip()
    if not want:
        return None
    idx = int(want.split(",")[0])
    gp = next((r for r in gpus() if r["cuda"] == idx), None)
    if gp is None:
        raise RuntimeError("RECLANK_EXPERT_GPUS=%s: no GPU %d of CUDA_VISIBLE_DEVICES" % (want, idx))
    gb = os.environ.get("RECLANK_EXPERT_GB")
    return QwenExpertDevice(g, gp, float(gb) if gb else None)


def nm_expert_device(g):
    """The expert device of NemotronGPU g (RECLANK_EXPERT_GPUS: the first of them), or None."""
    want = os.environ.get("RECLANK_EXPERT_GPUS", "").strip()
    if not want:
        return None
    idx = int(want.split(",")[0])
    gp = next((r for r in gpus() if r["cuda"] == idx), None)
    if gp is None:
        raise RuntimeError("RECLANK_EXPERT_GPUS=%s: no GPU %d of CUDA_VISIBLE_DEVICES" % (want, idx))
    gb = os.environ.get("RECLANK_EXPERT_GB")
    return NmExpertDevice(g, gp, float(gb) if gb else None)


_DEV = {}
_PENDING = {}       # id(model) -> (thread, result list) of start_expert_device


def _make(model):
    want = os.environ.get("RECLANK_EXPERT_GPUS", "").strip()
    idx = int(want.split(",")[0])
    g = next((r for r in gpus() if r["cuda"] == idx), None)
    if g is None:
        raise RuntimeError("RECLANK_EXPERT_GPUS=%s: no GPU %d of CUDA_VISIBLE_DEVICES" % (want, idx))
    gb = os.environ.get("RECLANK_EXPERT_GB")
    return ExpertDevice(model, g, int(os.environ.get("RECLANK_EXPERT_SKIP", "2000")), float(gb) if gb else None)


def start_expert_device(model):
    """Make the expert device of model in a thread (Model.load_all, as soon as
    the experts are in memory): the load of its library and the copy of its
    experts (6 GB on the GTX 1070) go on while the primary makes its programs
    and copies its weights, and not in the first request. expert_device
    waits for it. Nothing without RECLANK_EXPERT_GPUS."""
    if not os.environ.get("RECLANK_EXPERT_GPUS", "").strip() or id(model) in _DEV or id(model) in _PENDING:
        return
    import threading
    from reclank.backends.cuda import runtime
    runtime.lib()           # the primary's library first (its build, its device), on this thread
    out = []

    def run():
        try:
            out.append(_make(model))
        except BaseException as e:      # (raised again by expert_device)
            out.append(e)
    t = threading.Thread(target=run, name="expert-device", daemon=True)
    _PENDING[id(model)] = (t, out)
    t.start()


def expert_device(model):
    """The expert device of model (RECLANK_EXPERT_GPUS: the first of them for
    now), made once (or the one start_expert_device is making), or None."""
    if not os.environ.get("RECLANK_EXPERT_GPUS", "").strip():
        return None
    key = id(model)
    if key not in _DEV:
        pend = _PENDING.pop(key, None)
        if pend is not None:
            pend[0].join()
            if isinstance(pend[1][0], BaseException):
                raise pend[1][0]
            _DEV[key] = pend[1][0]
        else:
            _DEV[key] = _make(model)
    return _DEV[key]
