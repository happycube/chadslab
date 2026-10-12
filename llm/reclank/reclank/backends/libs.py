"""The cache of the built libraries (the CPU tiers and the CUDA archs).

The libraries go to ~/.cache/reclank/libs/<machine>/ (RECLANK_LIBS_DIR sets
another directory), out of the source tree: the tree can be on a network
mount, where a build is slow and two machines could share a stale library.
The name of a library has a hash of its sources, flags, and compiler, so a
change of any of them makes a new file.
"""
from __future__ import annotations

import os
import platform
import subprocess
import time
from pathlib import Path


def lib_dir():
    d = os.environ.get("RECLANK_LIBS_DIR")
    return Path(d) if d else Path.home() / ".cache" / "reclank" / "libs" / platform.machine()


def lib_ready(lib):
    """lib exists: mark it used now (its mtime orders the sweep). Return True
    when it exists."""
    try:
        os.utime(lib)
        return True
    except OSError:
        return False


def sweep_libs(lib, prefix, keep):
    """Keep the newest keep libraries of the prefix next to lib
    (RECLANK_LIBS_KEEP, else keep; lib always stays), and remove the
    temporary files of more than an hour. A process that loaded a removed
    library keeps its mapping."""
    keep = int(os.environ.get("RECLANK_LIBS_KEEP", keep))
    try:
        libs = sorted((p for p in lib.parent.glob(prefix + "*.so")
                       if p.name[len(prefix):-3].isalnum() and len(p.name) == len(lib.name)),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        for p in libs[keep:]:
            if p != lib:
                p.unlink()
                src = p.with_suffix(".c")
                if src.exists():
                    src.unlink()
        now = time.time()
        for p in lib.parent.glob("*.tmp"):
            if now - p.stat().st_mtime > 3600:
                p.unlink()
    except OSError:
        pass


def build_lib(cmd_of, lib, prefix, keep):
    """Build lib with the command cmd_of(tmp) and return it. The temporary
    file has the pid in its name, so two processes that build the same lib
    do not write one file; the rename is atomic. Then sweep_libs."""
    lib.parent.mkdir(parents=True, exist_ok=True)
    tmp = lib.with_name("%s.%d.tmp" % (lib.name, os.getpid()))
    try:
        r = subprocess.run(cmd_of(tmp), capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("the build of %s failed:\n%s" % (lib.name, r.stderr[-4000:]))
        os.replace(tmp, lib)
    finally:
        if tmp.exists():
            tmp.unlink()
    sweep_libs(lib, prefix, keep)
    return lib
