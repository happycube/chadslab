#!/usr/bin/env python3
"""The Qwen server (Qwen3.6 and Qwen3.8-Flash-Next) with the options of
before: a wrapper of the one server (python -m server, server/options.py).

    --backend gpu|cpu        --backend cuda|cpu
    -m PATH                  --model PATH (default MODEL below)
    --port                   8081 by default, as before

The other options have the same names. The engine is
reclank/models/qwen/engine.py; the template and the parser of the answer
are in reclank/models/qwen/chat.py.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reclank.models.qwen.chat import MODEL  # noqa: E402,F401  (the default model file)


def translate(argv):
    out = ["--family", "qwen"]
    i = 0
    have = set()
    while i < len(argv):
        a = argv[i]
        name, eq, val = a.partition("=")
        have.add(name)
        if name == "--backend":
            v = val if eq else argv[i + 1]
            if not eq:
                i += 1
            out += ["--backend", {"gpu": "cuda"}.get(v, v)]
        else:
            out.append(a)
        i += 1
    if not have & {"-m", "--model"}:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out += ["--model", os.path.join(root, MODEL)]
    if "--port" not in have:
        out += ["--port", "8081"]
    return out


if __name__ == "__main__":
    import numpy as np
    np.seterr(over="ignore")
    from server.__main__ import main
    raise SystemExit(main(translate(sys.argv[1:])))
