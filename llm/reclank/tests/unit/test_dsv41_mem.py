"""models/deepseek41/gpu.py's program memory (docs/PROGRAM_MEMORY.md): the sub-batches of a group's lanes and the
rows of the scratch they share, the big groups, and the hot experts' slots in segments (a shrink keeps each
layer's best experts, moved on the GPU into the slots that stay; a growth adds free slots; no program goes)."""
from __future__ import annotations

import types

import numpy as np
import pytest

gpu = pytest.importorskip("reclank.models.deepseek41.gpu")


def bare(group=2048, win=128, ab=True, L=3):
    g = object.__new__(gpu.DSV41GPU)
    g.group, g.ab = group, ab
    g.m = types.SimpleNamespace(win=win, L=L)
    g.scratch, g.big_scratch = {}, {}
    return g


def group_sizes(group, step):
    """The sizes of the groups of DSV41GPU.prefill: the multiples of step up to the group, the powers of two below."""
    return sorted({v for v in range(step, group + 1, step)} | {1 << b for b in range(4, step.bit_length())})


def test_sub_ranges(monkeypatch):
    monkeypatch.setattr(gpu, "SUB", 512)
    g = bare()
    assert g.sub_ranges(2048, [(0, 1024), (1024, 2048)]) == [[(0, 512), (512, 1024)], [(1024, 1536), (1536, 2048)]]
    assert g.sub_ranges(1920, [(0, 960), (960, 1920)]) == [[(0, 512), (512, 960)], [(960, 1472), (1472, 1920)]]
    assert g.sub_ranges(384, [(0, 192), (192, 384)]) == [[(0, 192)], [(192, 384)]]
    monkeypatch.setattr(gpu, "SUB", 0)
    assert g.sub_ranges(2048, [(0, 1024), (1024, 2048)]) == [[(0, 1024)], [(1024, 2048)]]


@pytest.mark.parametrize("sub,big_from", [(512, 512), (256, 1024), (0, 512), (512, 0), (512, 4096)])
def test_scratch_rows(monkeypatch, sub, big_from):
    """every group's arrays fit the bases of its store: its rows the store's most, its sub-batches loc_rows"""
    monkeypatch.setattr(gpu, "SUB", sub)
    monkeypatch.setattr(gpu, "BIG_FROM", big_from)
    monkeypatch.setattr(gpu, "GROUP_STEP", 128)
    g = bare()
    for t in group_sizes(2048, 128) + [g.m.win]:
        store, most = g.scratch_of(t)
        assert (store is g.big_scratch) == g.is_big(t)
        assert t <= most
        lanes = [(0, t // 2), (t // 2, t)] if t >= 64 else [(0, t)]
        rows = max(r1 - r0 for sj in g.sub_ranges(t, lanes) for r0, r1 in sj)
        assert rows <= g.loc_rows(most), (t, rows, most)
        if sub:
            assert rows <= sub
    if 0 < big_from <= 2048:
        assert g.small_most() < big_from


class _Seg:
    def __init__(self, nbytes, *_a, **_kw):
        _Seg.at += 1 << 20
        self.ptr, self.nbytes, self.freed = _Seg.at, nbytes, False

    def free(self):
        self.freed = True


_Seg.at = 0


def test_shrink_and_grow_hot(monkeypatch):
    """shrink_hot: each layer keeps its best experts (by DSHot's score) in the slots that stay, those of the freed
    slots copied in; the maps follow; the last segments freed. grow_hot: new segments, their slots free."""
    L, E, n, per = 3, 8, 4, 64
    copies = []
    stub = types.SimpleNamespace(rc_gpu_d2d=lambda d, s, nb: copies.append((d, s, nb)) or 0, rc_gpu_sync=lambda: 0)
    from reclank.backends.cuda import gpumm, runtime
    monkeypatch.setattr(runtime, "lib", lambda: stub)
    monkeypatch.setattr(gpu, "Buffer", _Seg)
    monkeypatch.setattr(gpumm, "mem_info", lambda: (int(10e9), int(24e9)))
    g = bare(L=L)
    g.per, g.reserve_gb = per, 0.0
    g.mirror = types.SimpleNamespace(owner=lambda a: None,
                                     buffer_of=lambda a: types.SimpleNamespace(upload=lambda x: None))
    g.stores = {i: np.zeros(E, np.int64) for i in range(L)}
    g.segs = []
    g._segs_to(n)
    g.n_slots, g.n_max = n, E
    rng = np.random.default_rng(1)
    g.slot_expert, g.maps = {}, {}
    for i in range(L):
        held = rng.choice(E, n, replace=False)
        g.slot_expert[i] = held.astype(np.int64)
        g.maps[i] = np.full(E, -1, np.int32)
        g.maps[i][held] = np.arange(n)
    score = rng.random((L, E))
    planned = []
    g.hot = types.SimpleNamespace(score=score, land=lambda wait=False: None,
                                  plan=lambda t, most=None: planned.append(most))
    before = {i: g.slot_expert[i].copy() for i in range(L)}
    addr = {i: g.stores[i].copy() for i in range(L)}
    segs = list(g.segs)
    assert g.shrink_hot(2) == 2 * L * per
    assert g.n_slots == 2 and len(g.segs) == 2 and all(s.freed for s in segs[2:]) and not any(s.freed for s in segs[:2])
    for i in range(L):
        best = sorted(before[i], key=lambda e: -score[i, e])[:2]
        assert sorted(g.slot_expert[i][:2]) == sorted(best)
        for s, e in enumerate(g.slot_expert[i][:2]):
            assert g.maps[i][e] == s
            old = list(before[i]).index(e)
            if old != s:            # (moved: its bytes copied from its old slot)
                assert (int(addr[i][s]), int(addr[i][old]), per) in copies
        assert (g.maps[i] >= 0).sum() == 2
        assert not g.stores[i][2:].any()
    assert g.grow_hot() == E - 2
    assert g.n_slots == E and len(g.segs) == E and planned == [(E - 2) * L]
    for i in range(L):
        assert (g.slot_expert[i][2:] == -1).all() and g.stores[i][:E].all()
