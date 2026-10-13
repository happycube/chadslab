#!/usr/bin/env python3
"""The DeepSeek V4.1 Flash server: a wrapper of the one server (python -m
server, server/options.py) with the family deepseek41 and port 8081.

    -m DIR       the checkpoint directory (config.json, tokenizer.json,
                 encoding/, the safetensors; the weightkeeper serves the experts)

The engine is reclank/models/deepseek41/engine.py. DeepSeek V4.1 on the GPU runs
with OMP_NUM_THREADS=1 only (set here before reclank loads).
"""
from __future__ import annotations

import os
import sys

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def translate(argv):
    out = ["--family", "deepseek41"]
    have = {a.partition("=")[0] for a in argv}
    out += list(argv)
    if "--port" not in have:
        out += ["--port", "8081"]
    return out


if __name__ == "__main__":
    import numpy as np
    np.seterr(over="ignore")
    from server.__main__ import main
    raise SystemExit(main(translate(sys.argv[1:])))
