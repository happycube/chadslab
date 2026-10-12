"""The settings of reclank: the RECLANK_* environment variables.

The old names NP_GEMMA_* still work. At import, each NP_GEMMA_X that has no
RECLANK_X becomes RECLANK_X, in os.environ (so the C and CUDA code, which
reads the process environment, sees it too), and one line on stderr lists
them. A RECLANK_X that is set wins over its old name.

get("X", default) reads RECLANK_X. README.md lists the settings.
"""
from __future__ import annotations

import os
import sys

PREFIX = "RECLANK_"
OLD_PREFIX = "NP_GEMMA_"


def _take_old_names():
    moved = []
    for k, v in list(os.environ.items()):
        if k.startswith(OLD_PREFIX):
            new = PREFIX + k[len(OLD_PREFIX):]
            if new not in os.environ:
                os.environ[new] = v
                moved.append(k)
    if moved and os.environ.get(PREFIX + "OLD_NAMES_QUIET", "0") == "0":
        print("[reclank] old setting names, read as RECLANK_*: %s" % ", ".join(sorted(moved)),
              file=sys.stderr, flush=True)
    return moved


OLD_NAMES = _take_old_names()


def get(name, default=None):
    """The value of RECLANK_<name>, or default."""
    return os.environ.get(PREFIX + name, default)
