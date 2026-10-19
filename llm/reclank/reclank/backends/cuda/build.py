"""The CUDA library: the toolkit, the arch of the GPU, the flags, and the build.

The code is in csrc/ (REFACTOR_PLAN.md, section 4.3): reclank_cuda.cu
includes the files of csrc/runtime and csrc/kernels in order, one
translation unit. The headers shared with the CPU are in ../include.

- The arch: RECLANK_GPU_ARCH (86 or sm_86; a suffix "a" asks for the
  arch-specific target, sm_120a), else the compute capability of the GPU
  RECLANK_GPU_DEVICE (nvidia-smi).
- The toolkit: RECLANK_CUDA (auto, 12, 13, or the path of an nvcc), else
  NVCC when it builds the arch, else the newest toolkit that builds it
  (/usr/local/cuda-13*, /usr/local/cuda-12*, the PATH). CUDA 13 stops at
  sm_75; Pascal (sm_61) needs CUDA 12.
- The build gives an explicit -arch=sm_XY (not native), and the name of the
  library has the arch and a hash of the sources, the toolkit, and the flags.
- The optimizer and ptxas in parallel (nvcc -split-compile, CUDA 11.8 on):
  RECLANK_CUDA_SPLIT threads (default 16; 1: none). The 3090 with CUDA
  13.2: 247 s in one thread (cicc 128 s, ptxas 98 s, the host compiler
  18 s), 84 s with 16.

    python -m reclank.backends.cuda.build --explain      the toolkit and the arch
    python -m reclank.backends.cuda.build --build [ARCH] build it
"""
from __future__ import annotations

import glob
import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from reclank.backends.libs import build_lib, lib_dir, lib_ready

CSRC = Path(__file__).resolve().parent / "csrc"
INCLUDE = Path(__file__).resolve().parents[1] / "include"
UNITY = CSRC / "reclank_cuda.cu"
# The lowest arch of each toolkit (major version).
MIN_ARCH = {11: 35, 12: 50, 13: 75}


def nvcc_version(nvcc):
    """(major, minor) of an nvcc, or None."""
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True)
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"release (\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def toolkits():
    """The nvcc of each toolkit on the machine: [(path, (major, minor))],
    the newest first."""
    cands = []
    for p in ([os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc")] if os.environ.get("CUDA_HOME") else []) \
            + sorted(glob.glob("/usr/local/cuda-*/bin/nvcc"), reverse=True) \
            + [shutil.which("nvcc") or "", "/usr/local/cuda/bin/nvcc"]:
        if p and os.path.exists(p):
            real = os.path.realpath(p)
            if real not in [os.path.realpath(c) for c, _ in cands]:
                v = nvcc_version(p)
                if v:
                    cands.append((p, v))
    return sorted(cands, key=lambda c: c[1], reverse=True)


def device_arch(device=None):
    """The arch to build: RECLANK_GPU_ARCH, else the compute capability of
    the GPU (86 for 8.6). Return (number, suffix) or None."""
    a = os.environ.get("RECLANK_GPU_ARCH", "").lower().replace("sm_", "")
    if a:
        m = re.fullmatch(r"(\d+)([a-z]?)", a)
        if not m:
            raise ValueError("RECLANK_GPU_ARCH=%s: a number like 86 or 120a" % a)
        return int(m.group(1)), m.group(2)
    device = int(os.environ.get("RECLANK_GPU_DEVICE", "0") if device is None else device)
    # the CUDA index through CUDA_VISIBLE_DEVICES (a number, as nvidia-smi
    # counts with CUDA_DEVICE_ORDER=PCI_BUS_ID, or a GPU-... UUID): with
    # CUDA_VISIBLE_DEVICES=1 the process runs on GPU 1, and nvidia-smi -i 0
    # gave the arch of GPU 0 (the 3060 then ran the 1070's sm_61 build, and
    # its tensor-core kernels trapped)
    which = str(device)
    vis = [v.strip() for v in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if v.strip()]
    if vis and device < len(vis):
        which = vis[device]
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader",
                                       "-i", which], text=True, stderr=subprocess.DEVNULL)
        major, minor = out.strip().split(".")
        return int(major) * 10 + int(minor), ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def builds(version, arch):
    return arch is None or arch >= MIN_ARCH.get(version[0], 0)


def pick_nvcc(arch):
    """The nvcc for the arch (see the module text)."""
    want = os.environ.get("RECLANK_CUDA", "auto").lower()
    tks = toolkits()
    if want not in ("", "auto"):
        if os.path.exists(want):
            return want, nvcc_version(want)
        for p, v in tks:
            if str(v[0]) == want:
                return p, v
        raise RuntimeError("RECLANK_CUDA=%s: no such toolkit (%s)" % (want, ", ".join(
            "%s %d.%d" % (p, *v) for p, v in tks)))
    env = os.environ.get("NVCC")
    if env and os.path.exists(env):
        v = nvcc_version(env)
        if v and builds(v, arch):
            return env, v
    for p, v in tks:
        if builds(v, arch):
            return p, v
    raise RuntimeError("no CUDA toolkit builds sm_%s (%s)" % (arch, ", ".join(
        "%s %d.%d" % (p, *v) for p, v in tks) or "none"))


def sources():
    return sorted(CSRC.rglob("*.cu")) + sorted(CSRC.rglob("*.cuh")) + sorted(INCLUDE.glob("*.h"))


def flags(arch, suffix=""):
    a = ("-arch=sm_%d%s" % (arch, suffix)) if arch else "-arch=native"
    out = ["-O3", a, "-shared", "-Xcompiler", "-fPIC", "-Wno-deprecated-gpu-targets",
           "-I", str(INCLUDE), "-I", str(CSRC)]
    # an older GPU on this one (caps.cu): the kernels of that GPU, so a
    # tensor-core kernel that runs anyway traps (port.cuh, RC_NO_SM80)
    cc = os.environ.get("RECLANK_GPU_CC", "")
    if cc.isdigit() and 0 < int(cc) < 80:
        out.append("-DRC_NO_SM80")
    # below 61 (Maxwell) the dp4a of port.cuh too: its scalar form, which
    # traps unless RECLANK_GPU_DP4A_EMU=1 (a wrong dispatch shows)
    if cc.isdigit() and 0 < int(cc) < 61:
        out.append("-DRC_NO_DP4A")
    # the source lines in the kernels (Nsight Compute's source view; a library
    # of its own: the flags are in its key)
    if os.environ.get("RECLANK_GPU_LINEINFO") == "1":
        out.append("-lineinfo")
    # Pascal (sm_6x) keeps the global loads in L2 only: -dlcm=ca caches them
    # in L1 too, as Volta and later do by default. The kernels that read x for
    # each row of weights then read it from L1 (the hot int4 experts of the 26B
    # on the GTX 1070, a warp a row: 187 -> 117 us). The flag handshake with
    # the CPU reads volatile (k_await): not cached. RECLANK_GPU_DLCM=0: off.
    if arch and arch < 70 and os.environ.get("RECLANK_GPU_DLCM", "1") != "0":
        out += ["-Xptxas", "-dlcm=ca"]
    return out


def split_flags(version):
    """-split-compile of the build (see the module text), or none."""
    env = os.environ.get("RECLANK_CUDA_SPLIT", "")
    # (a fixed default: the count is in the flags, so in the key of the library)
    n = int(env) if env.isdigit() else 16
    return ["-split-compile=%d" % n] if n > 1 and tuple(version) >= (11, 8) else []


_PRE = {}       # (arch, suffix) -> the thread of prebuild that builds that library


def prebuild(archs):
    """Start the builds of the libraries of archs ((arch, suffix) each) in threads, now: beside the
    build of the CPU library (backends/cpu/cops.py) and each other. build() of one of them waits for
    its thread (no second nvcc of the same library)."""
    import threading
    for a in archs:
        if a is None or a in _PRE:
            continue

        def run(a=a):
            try:
                build(a)
            except Exception:       # (the build of the caller gives the error)
                pass
        t = _PRE[a] = threading.Thread(target=run, name="build-sm%s%s" % a, daemon=True)
        t.start()


def build(arch=None):
    """Build the library when needed; return its path."""
    if arch is None:
        da = device_arch()
        arch, suffix = da if da else (None, "")
    else:
        arch, suffix = arch
    t = _PRE.get((arch, suffix))
    if t is not None:
        import threading
        if t is not threading.current_thread():
            t.join()
    nvcc, version = pick_nvcc(arch)
    fl = flags(arch, suffix) + split_flags(version)
    text = "".join(p.read_text() for p in sources())
    key = hashlib.sha256((text + "|%d.%d|" % version + nvcc + "|" + " ".join(fl))
                         .encode()).hexdigest()[:16]
    tag = ("sm%d%s" % (arch, suffix)) if arch else "native"
    if "-DRC_NO_SM80" in fl:
        tag += "_cc" + os.environ["RECLANK_GPU_CC"]
    prefix = "libreclank_cuda_%s_" % tag
    lib = lib_dir() / (prefix + key + ".so")
    if lib_ready(lib):
        return lib
    return build_lib(lambda tmp: [nvcc] + fl + ["-o", str(tmp), str(UNITY)], lib, prefix, 8)


def main(argv):
    if "--explain" in argv:
        da = device_arch()
        print("arch: %s" % ("sm_%d%s" % da if da else "native (no nvidia-smi)"))
        for p, v in toolkits():
            print("toolkit %d.%d %s%s" % (v[0], v[1], p, "" if builds(v, da and da[0]) else
                                          " (does not build this arch)"))
        print("picked:", pick_nvcc(da and da[0])[0])
        return 0
    if "--build" in argv:
        rest = [a for a in argv[1:] if not a.startswith("-")]
        arch = None
        if rest:
            m = re.fullmatch(r"(?:sm_)?(\d+)([a-z]?)", rest[0])
            arch = (int(m.group(1)), m.group(2))
        print(build(arch))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
