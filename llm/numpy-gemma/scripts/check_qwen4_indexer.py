#!/usr/bin/env python3
"""Compare the QSA indexer of np_gemma.qwen4 with transformers.

QWEN38_PLAN.md, phase 1. The indexer of a QSA layer keeps, for each query,
the best 512 blocks of 4 keys and the tail. It changes the attention only
after 2048 + 3 positions, so a check through the whole model is slow in
NumPy.

This script runs the indexer of one layer alone, in this runtime and in
Qwen4ExpTextQSAIndexer of transformers. The module gets the weights
of the file (the norms without the 1 of the converter). Both get the same
hidden states (--n rows). The script compares the keys that each query
keeps.

The function relu gives many scores of 0. At the cut, torch.topk keeps an
arbitrary subset of equal scores. A difference passes when the blocks that
differ have the score of the last kept block. The keys of this runtime are
float16, so a block with a score near the cut (within NEAR of it) can also
change places with the last kept block; that passes too, and the script
gives the count.

It needs torch and transformers (the venv of gemma4-12b-qat-pytorch):

    $VENV/bin/python scripts/check_qwen4_indexer.py --n 2600
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.qwen4_exp import modeling_qwen4_exp as M  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402

from np_gemma.qwen4 import Qwen4, Qwen4Cache  # noqa: E402

DIR = "models/Qwen3.8-Flash-Next-GGUF"
PATH = DIR + "/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf"
# the largest distance of a changed block from the score of the cut (relative)
NEAR = 1e-2


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=2600)
    ap.add_argument("--layer", type=int, default=3)
    args = ap.parse_args()
    i, n = args.layer, args.n
    m = Qwen4(PATH, layers=i + 1)
    c = json.load(open(os.path.join(DIR, "base/config.json")))["text_config"]
    c.pop("model_type", None)
    idx = M.Qwen4ExpTextQSAIndexer(Qwen4ExpTextConfig(**c), i)
    p = "blk.%d.indexer." % i
    with torch.no_grad():
        idx.index_qk_proj.weight.copy_(torch.from_numpy(
            np.concatenate([m.G(p + "q_proj.weight"), m.G(p + "k_proj.weight")])))
        idx.q_layernorm.weight.copy_(torch.from_numpy(m.G(p + "q_norm.weight") - 1.0))
        idx.k_layernorm.weight.copy_(torch.from_numpy(m.G(p + "k_norm.weight") - 1.0))
    h = (np.random.default_rng(0).standard_normal((n, m.cfg.hidden_size)) * 0.05).astype(np.float32)
    cos, sin = m.rope(np.arange(n))
    causal = torch.tril(torch.ones(n, n, dtype=torch.bool))[None, None]
    with torch.no_grad():
        theirs = idx(torch.from_numpy(h)[None], (torch.from_numpy(cos)[None], torch.from_numpy(sin)[None]),
                     causal, None)[0, 0].numpy()
    ours = m.qsa_mask(i, h, Qwen4Cache(m.cfg, n + 8), 0)
    ratio = m.cfg.compress_ratios[i]
    budget = m.cfg.indexer_top_k // ratio
    sparse = int(((ours.sum(1) < np.arange(1, n + 1))).sum())
    ties = near = bad = 0
    for j in np.nonzero((ours != theirs).any(1))[0]:
        score = m.qsa_scores[j]
        nb = len(score)
        a = ours[j][:nb * ratio].reshape(nb, ratio).all(1)
        b = theirs[j][:nb * ratio].reshape(nb, ratio).all(1)
        cut = np.sort(score)[::-1][budget - 1]
        tail = np.array_equal(ours[j][nb * ratio:], theirs[j][nb * ratio:])
        if np.all(score[a != b] == cut) and tail:
            ties += 1
        elif tail and np.all(np.abs(score[a != b] - cut) <= NEAR * abs(cut)):
            near += 1
        else:
            bad += 1
    print("%d queries, %d with dropped keys; %d differ at equal scores, %d near the cut, "
          "%d differ otherwise" % (n, sparse, ties, near, bad))
    ok = bad == 0 and sparse > 0
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
