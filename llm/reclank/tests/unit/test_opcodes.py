"""The opcode table (reclank/ir/opcodes.py) against its header and the runners."""
from __future__ import annotations

import os
import re

from reclank.ir import opcodes as O
from reclank.ir import program as P

PKG = os.path.dirname(os.path.dirname(os.path.abspath(O.__file__)))
CPU = os.path.join(PKG, "backends", "cpu", "csrc")
GPU = os.path.join(PKG, "backends", "cuda", "csrc")


def _cat(paths):
    out = []
    for p in paths:
        with open(p) as f:
            out.append(f.read())
    return "".join(out)


def _cpu():
    return _cat(sorted(os.path.join(d, f) for d, _, fs in os.walk(CPU) for f in fs if f.endswith(".c")))


def _gpu():
    return _cat(sorted(os.path.join(d, f) for d, _, fs in os.walk(GPU) for f in fs if f.endswith(".cu")))


def test_header_is_generated():
    assert _cat([O.HEADER]) == O.header(), "run python -m reclank.ir.opcodes"


def test_codes_are_unique():
    codes = [code for _, code, _, _ in O.TABLE]
    names = [name for name, _, _, _ in O.TABLE]
    assert len(set(codes)) == len(codes)
    assert len(set(names)) == len(names)


def test_each_runner_has_its_cases():
    cpu = set(re.findall(r"case GP_([A-Z0-9_]+):", _cpu()))
    gpu = set(re.findall(r"case GP_([A-Z0-9_]+):", _gpu()))
    for name, _, runners, _ in O.TABLE:
        assert ("cpu" in runners) == (name in cpu), "%s: the CPU runner" % name
        assert ("gpu" in runners) == (name in gpu), "%s: the GPU runner" % name
    assert cpu <= set(O.OPS) and gpu <= set(O.OPS)


def test_no_second_enum():
    for name, text in (("the CPU library", _cpu()), ("the CUDA library", _gpu())):
        assert not re.search(r"GP_[A-Z0-9_]+ = \d+", text), name


def test_program_uses_the_table():
    for name, code in O.OPS.items():
        assert getattr(P, name) == code
    assert P.OP_NAMES == O.OP_NAMES and P.NARG == O.NARG and P.MAGIC == O.MAGIC


def test_cpu_runner_fails_on_an_unknown_code():
    import pytest
    p = P.Program()
    p.emit(999, 1)
    p.finish()
    with pytest.raises(RuntimeError, match="-4"):
        p.run()
