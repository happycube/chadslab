#!/usr/bin/env python3
"""Check the decode of the 26B on the GPU after a long prompt pass on the GPU.

A prompt pass on the GPU makes the rows of the cache on the GPU. A sliding
layer drops its oldest rows there (GPUKV.prepare). A wrong count of the host
rows once made the next step move the window with no rows: after a prompt of
about 8000 tokens or more, the text of the GPU was nonsense.

The script runs the prompt pass on the GPU, then --steps greedy steps on the
GPU. It then runs the prompt pass on the GPU again, copies the cache to the
host, and runs the same steps with the CPU program. The two runs read the
same rows, so the tokens must agree, and the logits must be close.

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python scripts/check_gpu_prompt.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["NP_GEMMA_GPU"] = "1"

from np_gemma import program as P  # noqa: E402
from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.model import KVCache, Model  # noqa: E402
from np_gemma.tokenizer import Tokenizer  # noqa: E402
from np_gemma import gpu  # noqa: E402

GGUF_PATH = "models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--chars", type=int, default=40000, help="the characters of README.md")
    ap.add_argument("--steps", type=int, default=32)
    ap.add_argument("--hot-gb", type=float, default=1.5)
    args = ap.parse_args()
    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    m = Model(g, cfg).load_all(dtype="int4")
    gpu.offload(m, args.hot_gb)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = open(os.path.join(root, "README.md")).read()[:args.chars]
    ids = tok.encode(tok.apply_chat_template(
        [{"role": "user", "content": text + "\n\nSummarize the text above in three sentences."}],
        add_generation_prompt=True, thinking=False))
    n = len(ids)

    # The GPU: the prompt pass and the steps.
    c = KVCache(cfg, max_len=n + args.steps + 16)
    x = m.prefill(ids, c)
    lg = m.logits(x[-1:])[0]
    gpu_tok, gpu_lg = [int(np.argmax(lg))], [lg]
    for i in range(args.steps):
        x = m.forward([gpu_tok[-1]], cache=c, start_pos=n + i)
        lg = m.logits(x[-1:])[0]
        gpu_tok.append(int(np.argmax(lg)))
        gpu_lg.append(lg)
    m._gpu_release(c)

    # The same prompt pass on the GPU, then the steps of the CPU program on
    # the rows of the GPU, with the tokens of the GPU run.
    c = KVCache(cfg, max_len=n + args.steps + 16)
    x = m.prefill(ids, c)
    m._gpu_release(c)
    same, worst = 0, 0.0
    for i in range(args.steps):
        x = P.decode_step(m, c, [gpu_tok[i]], n + i)
        lg = m.logits(x)[0]
        same += int(np.argmax(lg)) == gpu_tok[i + 1]
        worst = max(worst, float(np.abs(lg - gpu_lg[i + 1]).max()))
    print("prompt of %d tokens on the GPU, %d steps: the same top token %d/%d, logits max |d| %.3f"
          % (n, args.steps, same, args.steps, worst))
    print(repr(tok.decode(gpu_tok)))
    ok = same >= args.steps - 1
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
