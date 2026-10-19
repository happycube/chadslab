"""The CPU library: the tiers, their flags, the unity source, and the build.

The C code is in csrc/ (REFACTOR_PLAN.md, section 3):

    csrc/common/    no intrinsics: the interpreter, the runners, the teams,
                    the MoE helpers, the DeltaNet, the hyper-connections, QSA
    csrc/kernels/   the kernels, a file for each family (kernels/kq/: the
                    GGUF formats)
    ../include/     opcodes.h (made by ir/opcodes.py) and tq6_tables.h

ORDER gives the files of the library in their order; the build includes
them in one unity source (one translation unit, as before the split, so
the compiler inlines and optimizes as it did). An entry that ends with "/"
is a family with a file for some tiers: the build takes the first file of
the chain of the tier (CHAIN) that exists, else any.c. The other files keep
the #if blocks of the ISA inside them.

The tier is RECLANK_CPU (scalar, avx2, avx2_vnni, avx512, avx512_vnni), or
the old RECLANK_ARCH (avx2, avx512, vnni), else the best tier of the CPU.

    python -m reclank.backends.cpu.build --explain [TIER]    the files of a tier
    python -m reclank.backends.cpu.build --build [TIER]      build it
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path
from shutil import which

from reclank.backends.libs import build_lib, lib_dir, lib_ready

CSRC = Path(__file__).resolve().parent / "csrc"
INCLUDE = Path(__file__).resolve().parents[1] / "include"

# --wrap=GOMP_parallel: every parallel region passes __wrap_GOMP_parallel
# (csrc/common/team_warn.c), which reports the teams outside the planned ones
FLAGS_COMMON = ["-O3", "-funroll-loops", "-fopenmp", "-shared", "-fPIC", "-lm",
                "-Wl,--wrap=GOMP_parallel", "-Werror=implicit-function-declaration"]
# F16C: every CPU with AVX2 has it, and the float16 scales use it.
TIERS = {
    "scalar": [],
    "avx2": ["-mavx2", "-mfma", "-mf16c"],
    "avx2_vnni": ["-mavx2", "-mfma", "-mf16c", "-mavxvnni"],
    "avx512": ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mfma", "-mf16c"],
    "avx512_vnni": ["-mavx512f", "-mavx512bw", "-mavx512vl", "-mavx512vnni", "-mfma", "-mf16c"],
}
CHAIN = {
    "avx512_vnni": ["avx512_vnni", "avx512", "avx2_vnni", "avx2", "scalar"],
    "avx512": ["avx512", "avx2", "scalar"],
    "avx2_vnni": ["avx2_vnni", "avx2", "scalar"],
    "avx2": ["avx2", "scalar"],
    "scalar": ["scalar"],
}
OLD_ARCH = {"avx2": "avx2", "avx512": "avx512", "vnni": "avx512_vnni"}

ORDER = [
    "common/head.c", "common/team_warn.c", "common/hotcache.c",
    "kernels/base.c", "kernels/int8.c", "kernels/int4.c", "kernels/bf16_gemm.c",
    "kernels/int8_pipe.c", "kernels/q6k.c", "kernels/router.c", "kernels/qkv_rope.c",
    "kernels/attn_decode.c", "kernels/kv16.c", "kernels/attn_split.c", "kernels/elem.c",
    "kernels/int4_q8.c", "kernels/int4_q16.c", "kernels/int8_dot.c", "kernels/moe_dense.c",
    "kernels/fused.c", "kernels/attn_prefill.c", "kernels/attn_prefill_qc.c", "kernels/ct.c",
    "kernels/bodies.c",
    "common/moe.c",
    "kernels/kq/formats.c", "kernels/kq/helpers.c", "kernels/kq/iq2.c", "kernels/kq/exl3.c", "kernels/kq/dot2/",
    "kernels/kq/dots.c", "kernels/kq/tiles.c",
    "kernels/kq/x16.c", "kernels/kq/nvx.c", "kernels/kq/q4x.c", "kernels/kq/q4x_entry.c",
    "kernels/kq/x16f.c", "kernels/kq/convert.c", "kernels/kq/moe_scratch.c", "kernels/kq/nmx2.c", "kernels/kq/mxx.c",
    "kernels/kq/moe_numa.c", "kernels/kq/exl3_moe.c", "kernels/kq/moe_small.c", "kernels/kq/moe.c",
    "kernels/kq/pack.c",
    "common/deltanet.c", "common/hyperconn.c", "common/qsa.c",
    "common/ops_qwen.c", "common/nemotron.c", "common/parts.c", "kernels/kq/moe_part.c", "common/interp.c", "common/run.c",
]


def _cpu_flags():
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("flags"):
                    return set(line.split(":", 1)[1].split())
    except OSError:
        pass
    return set()


def best_tier():
    """The best tier of this CPU."""
    if platform.machine() not in ("x86_64", "AMD64", "i686", "i386"):
        return "scalar"
    f = _cpu_flags()
    if not f:
        try:
            import numpy.core._multiarray_umath as _m
            feat = _m.__cpu_features__
            f = {k.lower() for k, v in feat.items() if v}
            f |= {"avx512_vnni"} if feat.get("AVX512VNNI") else set()
        except Exception:
            return "avx2"
    if {"avx512f", "avx512bw", "avx512vl"} <= f:
        return "avx512_vnni" if "avx512_vnni" in f else "avx512"
    if {"avx2", "fma", "f16c"} <= f:
        return "avx2_vnni" if "avx_vnni" in f else "avx2"
    return "scalar"


def tier_flags(t):
    """The flags of tier t. RECLANK_CPU_TEST_EVEX=1 builds avx2_vnni without
    -mavxvnni and with the EVEX form of vpdpbusd (a CPU with AVX-512 VNNI
    tests the code of that tier)."""
    if t == "avx2_vnni" and os.environ.get("RECLANK_CPU_TEST_EVEX") == "1":
        return [f for f in TIERS[t] if f != "-mavxvnni"] + ["-DRC_VNNI_EVEX"]
    return TIERS[t]


def tier():
    """The tier of this process: RECLANK_CPU, the old RECLANK_ARCH, or the
    best tier of the CPU."""
    t = os.environ.get("RECLANK_CPU", "").lower()
    if not t:
        t = OLD_ARCH.get(os.environ.get("RECLANK_ARCH", "").lower(), "")
    if not t or t == "auto":
        t = best_tier()
    if t not in TIERS:
        raise ValueError("RECLANK_CPU=%s: one of %s" % (t, ", ".join(TIERS)))
    return t


def files(t):
    """The files of the unity source of tier t, in their order."""
    out = []
    for entry in ORDER:
        if entry.endswith("/"):
            for c in CHAIN[t] + ["any"]:
                p = CSRC / entry / (c + ".c")
                if p.exists():
                    out.append(entry + c + ".c")
                    break
            else:
                raise FileNotFoundError("no file of %s for tier %s" % (entry, t))
        else:
            out.append(entry)
    return out


def unity(t):
    return "/* The CPU library of tier %s (reclank/backends/cpu/build.py). */\n" % t + "".join(
        '#include "%s"\n' % (CSRC / f) for f in files(t))


def compiler():
    """The name of a C compiler, or None."""
    for name in (os.environ.get("CC"), "cc", "gcc", "clang"):
        if name and which(name):
            return name
    return None


def build(t=None):
    """Build the library of tier t when needed. Return its path, or None
    without a compiler."""
    t = t or tier()
    cc = compiler()
    if cc is None:
        return None
    try:
        version = subprocess.check_output([cc, "--version"], text=True,
                                          stderr=subprocess.STDOUT).splitlines()[0]
    except Exception:
        version = "unknown"
    flags = FLAGS_COMMON + tier_flags(t)
    src = unity(t)
    texts = [p.read_text() for p in sorted(CSRC.rglob("*.c")) + sorted(INCLUDE.glob("*.h"))]
    key = hashlib.sha256("|".join(texts + [src, sys.platform, platform.machine(), version,
                                          " ".join(flags)]).encode()).hexdigest()[:16]
    prefix = "libreclank_cpu_%s_" % t
    lib = lib_dir() / (prefix + key + ".so")
    if lib_ready(lib):
        return lib
    lib.parent.mkdir(parents=True, exist_ok=True)
    unity_c = lib.with_suffix(".c")
    unity_c.write_text(src)
    return build_lib(lambda tmp: [cc] + flags + ["-I", str(INCLUDE), "-I", str(CSRC), "-o", str(tmp),
                                                 str(unity_c)], lib, prefix, 24)


def main(argv):
    t = argv[1] if len(argv) > 1 and not argv[1].startswith("-") else tier()
    if "--explain" in argv:
        print("tier %s (chain: %s); flags: %s" % (t, " > ".join(CHAIN[t]), " ".join(TIERS[t]) or "-"))
        for f in files(t):
            print("   ", f)
        return 0
    if "--build" in argv:
        print(build(t))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    tier_arg = [a for a in args if not a.startswith("-")]
    raise SystemExit(main([sys.argv[0]] + tier_arg + [a for a in args if a.startswith("-")]))
