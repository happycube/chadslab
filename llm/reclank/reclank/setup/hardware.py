"""The machine for the setup report: CPU, memory, NVIDIA GPUs, toolkits, disk.

detect(models_dir) gives a Machine. assume(m, ...) replaces parts of it, to
show what a smaller machine gets (--assume-ram 16 --assume-vram 0). Each
probe degrades: no /proc, no nvidia-smi or no nvcc gives an empty answer
and a note, not an error. Nothing here loads the CUDA library or builds C.
"""
from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

GB = 1e9              # disk, as the Hub counts a download
GIB = 2.0 ** 30       # memory: RAM and VRAM in GiB, as a card's "8 GB"


@dataclass
class GPU:
    index: int
    name: str
    vram_gb: float
    free_gb: float
    cc: float | None          # the compute capability, 8.6 for sm_86; None: unknown


@dataclass
class Machine:
    os: str
    arch: str
    cpu: str = "unknown"
    threads: int = 0          # the logical CPUs of the machine
    cores: int = 0            # the physical cores (0: unknown)
    usable: int = 0           # the CPUs of this process's affinity
    nodes: list = field(default_factory=list)    # [(node, cpus, GB)]
    tier: str = "scalar"      # the C tier (backends/cpu/build.py)
    isa: list = field(default_factory=list)      # avx512f, avx512_vnni, avx2, ...
    ram_gb: float = 0.0
    ram_free_gb: float = 0.0
    gpus: list = field(default_factory=list)
    driver: str | None = None
    cuda_driver: str | None = None               # the CUDA version of the driver (nvidia-smi)
    toolkits: list = field(default_factory=list)  # [(nvcc, "12.9")]
    cc_compiler: str | None = None
    python: str = ""
    numpy: str | None = None
    hub: bool = False         # huggingface_hub importable
    hf_cli: str | None = None
    models_dir: str = ""
    disk_free_gb: float | None = None
    notes: list = field(default_factory=list)
    assumed: dict = field(default_factory=dict)

    def gpu(self):
        """The largest GPU, or None."""
        return max(self.gpus, key=lambda g: g.vram_gb) if self.gpus else None


def _read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""


def _run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu(m):
    info = _read("/proc/cpuinfo")
    names = re.findall(r"^model name\s*:\s*(.+)$", info, re.M)
    if names:
        m.cpu = re.sub(r"\s+", " ", names[0]).strip()
    elif sys.platform == "darwin":
        m.cpu = (_run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown").strip()
    else:
        m.cpu = platform.processor() or "unknown"
    m.threads = os.cpu_count() or 0
    try:
        m.usable = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        m.usable = m.threads
    # the cores: the pairs (physical id, core id) of /proc/cpuinfo
    pairs = set()
    for block in info.split("\n\n"):
        p = re.search(r"^physical id\s*:\s*(\d+)", block, re.M)
        c = re.search(r"^core id\s*:\s*(\d+)", block, re.M)
        if p and c:
            pairs.add((p.group(1), c.group(1)))
    m.cores = len(pairs)
    try:
        from reclank.backends.cpu import build
        m.tier = build.best_tier()
        flags = build._cpu_flags()
    except Exception as e:      # (no reclank on the path: the report still runs)
        m.notes.append("the C tier is unknown (%s)" % e)
        flags = set()
    m.isa = [f for f in ("avx512f", "avx512bw", "avx512vl", "avx512_vnni", "avx_vnni", "avx2", "fma", "f16c")
             if f in flags]


def _nodes(m):
    for d in sorted(glob.glob("/sys/devices/system/node/node[0-9]*"),
                    key=lambda p: int(re.search(r"(\d+)$", p).group(1))):
        node = int(re.search(r"(\d+)$", d).group(1))
        cpus = _read(os.path.join(d, "cpulist")).strip()
        mem = re.search(r"MemTotal:\s*(\d+) kB", _read(os.path.join(d, "meminfo")))
        m.nodes.append((node, cpus, int(mem.group(1)) * 1024 / GIB if mem else 0.0))


def _ram(m):
    info = _read("/proc/meminfo")
    tot = re.search(r"^MemTotal:\s*(\d+) kB", info, re.M)
    avail = re.search(r"^MemAvailable:\s*(\d+) kB", info, re.M)
    if tot:
        m.ram_gb = int(tot.group(1)) * 1024 / GIB
        m.ram_free_gb = int(avail.group(1)) * 1024 / GIB if avail else m.ram_gb
        return
    try:
        m.ram_gb = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / GIB
    except (ValueError, OSError, AttributeError):
        out = _run(["sysctl", "-n", "hw.memsize"]) if sys.platform == "darwin" else None
        m.ram_gb = int(out) / GIB if out and out.strip().isdigit() else 0.0
    m.ram_free_gb = m.ram_gb
    if not m.ram_gb:
        m.notes.append("the RAM is unknown: give it with --assume-ram GB")


def _gpus(m):
    smi = shutil.which("nvidia-smi")
    if not smi:
        if glob.glob("/proc/driver/nvidia/gpus/*"):
            m.notes.append("an NVIDIA GPU is present but nvidia-smi is not on the PATH")
        return
    q = "index,name,memory.total,memory.free,compute_cap,driver_version"
    out = _run([smi, "--query-gpu=" + q, "--format=csv,noheader,nounits"])
    if out is None:
        # (old drivers have no compute_cap field)
        out = _run([smi, "--query-gpu=index,name,memory.total,memory.free,driver_version",
                    "--format=csv,noheader,nounits"])
        old = True
    else:
        old = False
    if out is None:
        m.notes.append("nvidia-smi failed: no GPU is used (is the driver loaded?)")
        return
    for line in out.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            if old:
                idx, name, tot, free, drv = f
                cc = None
            else:
                idx, name, tot, free, cc, drv = f
                cc = float(cc) if re.fullmatch(r"\d+\.\d+", cc) else None
            m.gpus.append(GPU(int(idx), name, int(tot) / 1024, int(free) / 1024, cc))      # (MiB)
            m.driver = drv
        except ValueError:
            continue
    head = _run([smi]) or ""
    v = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", head)
    m.cuda_driver = v.group(1) if v else None


def _toolkits(m):
    try:
        from reclank.backends.cuda import build as cbuild
        m.toolkits = [(p, "%d.%d" % v) for p, v in cbuild.toolkits()]
    except Exception as e:
        m.notes.append("the CUDA toolkits are unknown (%s)" % e)


def _tools(m, models_dir):
    m.cc_compiler = os.environ.get("CC") if os.environ.get("CC") and shutil.which(os.environ["CC"]) else \
        next((shutil.which(c) for c in ("cc", "gcc", "clang") if shutil.which(c)), None)
    m.python = platform.python_version()
    try:
        import numpy
        m.numpy = numpy.__version__
    except ImportError:
        m.numpy = None
    import importlib.util
    m.hub = importlib.util.find_spec("huggingface_hub") is not None
    # the hf command: on the PATH, or beside this Python (a venv not activated)
    venv_hf = os.path.join(os.path.dirname(sys.executable), "hf")
    m.hf_cli = "hf" if shutil.which("hf") else venv_hf if os.access(venv_hf, os.X_OK) else None
    m.models_dir = models_dir
    # the free disk of the folder, or of its nearest parent that exists
    p = os.path.abspath(models_dir)
    while p and not os.path.exists(p):
        p = os.path.dirname(p)
    try:
        m.disk_free_gb = shutil.disk_usage(p).free / GB
    except OSError:
        m.disk_free_gb = None


def detect(models_dir):
    """The Machine of this process."""
    m = Machine(os=platform.system(), arch=platform.machine())
    if m.os != "Linux":
        m.notes.append("reclank runs on Linux (it uses /proc, /sys and mbind); this is %s" % m.os)
    _cpu(m)
    _nodes(m)
    _ram(m)
    _gpus(m)
    _toolkits(m)
    _tools(m, models_dir)
    return m


def assume(m, ram=None, vram=None, cc=None, tier=None, cores=None, disk=None):
    """Replace the parts of m that are given: RAM and VRAM in GB (vram 0: no
    GPU), the compute capability (6.1), the C tier, the cores, the free disk."""
    if ram is not None:
        m.ram_gb = m.ram_free_gb = float(ram)
        m.nodes = [(0, "", float(ram))]
        m.assumed["ram"] = ram
    if vram is not None:
        if float(vram) <= 0:
            m.gpus = []
        else:
            g = m.gpu()
            m.gpus = [GPU(0, "assumed GPU" if g is None or cc is not None else g.name + " (assumed size)",
                          float(vram), float(vram), cc if cc is not None else (g.cc if g else 8.6))]
            if not m.toolkits:
                m.toolkits = [("(assumed)", "12.9")]
        m.assumed["vram"] = vram
    if cc is not None:
        for g in m.gpus:
            g.cc = float(cc)
        m.assumed["cc"] = cc
    if tier is not None:
        m.tier = tier
        m.assumed["tier"] = tier
    if cores is not None:
        m.cores = m.usable = m.threads = int(cores)
        m.assumed["cores"] = cores
    if disk is not None:
        m.disk_free_gb = float(disk)
        m.assumed["disk"] = disk
    return m
