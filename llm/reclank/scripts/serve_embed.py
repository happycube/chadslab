#!/usr/bin/env python3
"""The embedding server (EmbeddingGemma 2 on the CPU): a wrapper of python
-m server.embed (port 8082). A script of its own, so that the stop of the
chat server (its check for a python of "-m server") does not wait for it.

    -m DIR       the embeddinggemma-2 directory
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    from server.embed import main
    raise SystemExit(main(sys.argv[1:]))
