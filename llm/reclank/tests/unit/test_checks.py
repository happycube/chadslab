"""The check scripts that need no model file, as pytest tests.

    PYTHONPATH=. python -m pytest tests/unit -q
    PYTHONPATH=. python -m pytest tests/unit -q -m "not gpu"

Each test runs one scripts/check_*.py in its own process (the libraries
build there, and a fault stays in that process) and wants exit status 0.
The GPU checks skip when there is no nvcc or when the GPU holds another
program (a server). The golden dumps (tests/golden) are the check of the
models.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CPU_CHECKS = [
    "check_chat_template",
    "check_flash",
    "check_flash_c",
    "check_gelu",
    "check_int4_q8",
    "check_kq_tiers",
    "check_rms_norm",
    "check_server",
    "check_tq6",
    "check_bf12_moe",
    "check_hotcache",
]
GPU_CHECKS = [
    "check_attn_dense",
    "check_bf12_gpu",
    "check_qsa_score --pos 3000 --rows 600,4,1",
    "check_dp4a",
    "check_mxf8",
    "check_nv4",
    "check_attn_f32t",
    "check_router_bias",
    "check_argmax",
    "check_kq_group",
    "check_attn_konly",
]


def _gpu_free():
    if not (os.environ.get("NVCC") or shutil.which("nvcc") or os.path.exists("/usr/local/cuda/bin/nvcc")):
        return "no nvcc"
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return "no nvidia-smi"
    if not out:
        return "no GPU"
    if int(out[0]) > 1000:
        return "the GPU holds %s MiB (another program)" % out[0]
    return None


def _run(name):
    name, *args = name.split()
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""))
    p = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", name + ".py")] + args, cwd=ROOT,
                       env=env, capture_output=True, text=True, timeout=1800)
    assert p.returncode == 0, "%s exit %d\n%s\n%s" % (name, p.returncode, p.stdout[-3000:],
                                                       p.stderr[-3000:])


@pytest.mark.parametrize("name", CPU_CHECKS)
def test_cpu_check(name):
    _run(name)


@pytest.mark.gpu
@pytest.mark.parametrize("name", GPU_CHECKS)
def test_gpu_check(name):
    why = _gpu_free()
    if why:
        pytest.skip(why)
    _run(name)
