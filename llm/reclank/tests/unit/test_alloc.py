"""ir/alloc.pack: the temporaries of a program in one arena by liveness."""
from __future__ import annotations

import numpy as np

from reclank.ir import alloc
from reclank.ir import program as P


def _prog(n=6):
    """A chain of ADD records over temporaries: t[i+1] = t[i] + w, then a
    COPY into the exported xn. Each t[i] lives two records."""
    prog = P.Program()
    w = np.ones(64, np.float32)
    x = np.zeros(64, np.float32)
    prog.names["x"] = x
    ts = []
    for _ in range(n):
        t = np.zeros(64, np.float32)
        prog.temps.append(t)
        ts.append(t)
    prog.emit(P.ADD, x, w, ts[0], 64)
    for i in range(1, n):
        prog.emit(P.ADD, ts[i - 1], w, ts[i], 64)
    xn = np.zeros(64, np.float32)
    prog.names["xn"] = xn
    prog.temps.append(xn)
    prog.emit(P.COPY, ts[-1], xn, xn.nbytes)
    return prog.finish(), ts, xn


def _addrs(prog):
    return [[int(v) for t, v in zip(r["tag"], r["v"]) if t == P.T_INT] for r in prog.code]


def test_pack_reuses_dead_temporaries():
    prog, ts, xn = _prog(6)
    info = alloc.pack(prog)
    assert info.packed == 6
    # each temporary lives from its record to the next one, plus a record:
    # three of 256 bytes at a time
    assert info.arena == 3 * 256
    base = prog.arena.ctypes.data
    for i, r in enumerate(prog.code):
        tags, vals = r["tag"], r["v"]
        if int(r["op"]) == P.ADD and i > 0:
            assert base <= int(vals[0]) < base + info.arena
        if int(r["op"]) == P.ADD:
            assert base <= int(vals[2]) < base + info.arena
    # the exported xn and the input x keep their arrays
    last = prog.code[-1]
    assert int(last["v"][1]) == xn.ctypes.data
    assert int(prog.code[0]["v"][0]) == prog.names["x"].ctypes.data
    # prog.recs agrees with the code (GPUProgram translates prog.recs)
    for (op, args), r in zip(prog.recs, prog.code):
        for k, (tag, v) in enumerate(args):
            assert int(r["v"][k]) == v


def test_pack_keeps_a_temporary_read_first():
    prog = P.Program()
    w = np.ones(64, np.float32)
    acc = np.zeros(64, np.float32)          # read by its first record
    out = np.zeros(64, np.float32)
    prog.temps += [acc, out]
    prog.emit(P.ADD, acc, w, out, 64)
    prog.finish()
    info = alloc.pack(prog)
    assert info.kept == 1 and info.packed == 1
    assert int(prog.code[0]["v"][0]) == acc.ctypes.data


def test_pack_keeps_the_arrays_of_an_unknown_opcode():
    prog = P.Program()
    a = np.zeros(64, np.float32)
    b = np.zeros(64, np.float32)
    prog.temps += [a, b]
    prog.emit(P.ADD, np.ones(64, np.float32), np.ones(64, np.float32), a, 64)
    prog.emit(P.ENC_ADD, a, b, 64)          # not in WRITES
    prog.finish()
    info = alloc.pack(prog)
    assert info.packed == 0


def test_pack_tq_rot_with_a_source_writes_x():
    """WRITES_IF: TQ_ROT with a source (operand 3) writes x, so x packs; in
    place its first record reads x, so x stays."""
    def prog_of(src):
        prog = P.Program()
        x = np.zeros(64, np.float32)
        out = np.zeros(64, np.float32)
        prog.temps += [x, out]
        if src:
            prog.emit(P.TQ_ROT, x, 2, 0, np.ones(64, np.float32))
        else:
            prog.emit(P.TQ_ROT, x, 2, 0)
        prog.emit(P.ADD, x, np.ones(64, np.float32), out, 64)
        prog.finish()
        return prog, x
    prog, x = prog_of(True)
    info = alloc.pack(prog)
    assert info.packed == 2 and info.kept == 0
    assert int(prog.code[0]["v"][0]) != x.ctypes.data
    prog, x = prog_of(False)
    info = alloc.pack(prog)
    assert info.kept == 1
    assert int(prog.code[0]["v"][0]) == x.ctypes.data
