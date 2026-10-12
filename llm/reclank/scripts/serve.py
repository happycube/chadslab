#!/usr/bin/env python3
"""The Gemma 4 server with the options of before: a wrapper of the one server
(python -m server, server/options.py). It maps the old options:

    --gguf PATH              --model PATH (both work)
    --mtp DIR, --mtp-n N     --drafter DIR, --mtp N
    --gpu off|dense|hot      --backend cpu | --backend cuda --gpu-dense | --backend cuda

The other options have the same names. The family is gemma4.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def translate(argv):
    out = ["--family", "gemma4"]
    i = 0
    while i < len(argv):
        a = argv[i]
        name, eq, val = a.partition("=")
        def value():
            nonlocal i
            if eq:
                return val
            i += 1
            return argv[i]
        if name == "--mtp":
            out += ["--drafter", value()]
        elif name == "--mtp-n":
            out += ["--mtp", value()]
        elif name == "--gpu":
            g = value()
            out += {"off": ["--backend", "cpu"], "dense": ["--backend", "cuda", "--gpu-dense"],
                    "hot": ["--backend", "cuda"]}[g]
        else:
            out.append(a)
        i += 1
    if "--backend" not in out:
        out += ["--backend", "cpu"]
    return out


if __name__ == "__main__":
    from server.__main__ import main
    raise SystemExit(main(translate(sys.argv[1:])))
