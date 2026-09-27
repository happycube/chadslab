#!/usr/bin/env python3
"""Compare the n-gram ids of np_gemma.qwen4 with transformers.

QWEN38_PLAN.md, phase 1. The n-gram table of Qwen3.8-Flash-Next reads 16
rows for each token; the rows come from hashes of the last 2 and 3 tokens.
The script builds Qwen4ExpTextNGramEmbedding of transformers with a stub
table (the real table has 51B values), and gets the ids that it reads. It
compares them with Qwen4.ple_ids: in one pass, and for a prompt, then
single steps with the cache. The tokens have the eos token of PLE in the middle.

It needs torch and transformers (the venv of gemma4-12b-qat-pytorch):

    $VENV/bin/python scripts/check_qwen4_ngram.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

from np_gemma.gguf import open_gguf  # noqa: E402
from np_gemma.qwen4 import Qwen4, Qwen4Cache, config_from_gguf  # noqa: E402

DIR = "models/Qwen3.8-Flash-Next-GGUF"
PATH = DIR + "/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"


class _Stub(torch.nn.Module):
    """A table that keeps the ids it gets and gives zeros."""

    def __init__(self, n, d):
        super().__init__()
        self.d = d
        self.seen = []
        self.weight = torch.zeros(1)

    def forward(self, ids):
        self.seen.append(ids.clone())
        return torch.zeros(*ids.shape, self.d)


def main():
    c = json.load(open(os.path.join(DIR, "base/config.json")))["text_config"]
    c.pop("model_type", None)
    cfg = Qwen4ExpTextConfig(**c)
    real = torch.nn.Embedding
    torch.nn.Embedding = _Stub
    try:
        ng = M.Qwen4ExpTextNGramEmbedding(cfg, cfg.ple_embed_dim, layer_idx=1, ple_layer_index=0)
    finally:
        torch.nn.Embedding = real
    qc = config_from_gguf(open_gguf(PATH))
    ids = np.random.default_rng(0).integers(0, qc.vocab_size, 40)
    ids[[7, 20, 21]] = qc.ple_eos
    ng(torch.tensor(ids[None]), None)
    theirs = ng.ngram_embedding.seen[-1][0].numpy()

    class _Model:
        pass

    m = _Model()
    m.cfg = qc

    def cache():
        c = Qwen4Cache.__new__(Qwen4Cache)
        c.ple_ids = np.full(qc.ple_ngram - 1, qc.ple_eos, np.int64)
        return c

    one = Qwen4.ple_ids(m, ids, cache())
    c2 = cache()
    steps = np.concatenate([Qwen4.ple_ids(m, ids[:25], c2)] +
                           [Qwen4.ple_ids(m, ids[j:j + 1], c2) for j in range(25, len(ids))])
    ok1, ok2 = np.array_equal(one, theirs), np.array_equal(steps, theirs)
    print("one pass: %s; a prompt of 25, then steps: %s" % (ok1, ok2))
    print("PASS" if ok1 and ok2 else "FAIL")
    return 0 if ok1 and ok2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
