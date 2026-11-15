"""Check that an E4B verify group on the GPU gives each token the bits of a decode step.

Speculative decoding (MTP, a DFlash drafter) keeps a draft when the target's pick at its row of the
verify group is the draft, and gives the tokens of plain decoding only when each row of the group
has the bits of a decode step on that token. The group's kernels keep the step's order of the terms
(k_mt_bf16_rows, k_mt_int4_rows; attn_steps_launch: each query as a step). For groups of 2 to 8
tokens on real prompts (the prompt's last 8 tokens after its other rows in the cache), the final
hidden rows and the logits of the group against 8 steps, bit for bit.

Run (a GPU; --drafter: the taps of a DFlash drafter, as the server compiles them):

    PYTHONPATH=. $PY scripts/check_e4b_group.py MODEL [--drafter DIR] [--text FILE]

MODEL: a checkpoint folder (Nemotron 3.5 Bolt E4B) or an E4B GGUF file. --text: a longer prompt (a
document to summarize) in place of the second built-in one."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

PROMPTS = {
    "code": "Write a Python function that parses a CSV file of (name, score) rows and returns the top 3 names by "
            "score, with a short docstring and type hints.",
    "chat": "Why is the sky blue? Answer in two short paragraphs.",
}


def load(path, drafter):
    from reclank.models.gemma4 import gpu as G
    from reclank.models.gemma4.e4b import E4B, E4BConfig
    if os.path.isdir(path):
        from reclank.models.gemma4.ct import Checkpoint
        src = Checkpoint(path)
    else:
        from reclank.formats.gguf import GGUF
        src = GGUF(path)
    model = E4B(src, E4BConfig({"text_config": src.text_config()}), mode="int4")
    if drafter:
        from reclank.models.gemma4.dflash import target_layers
        model.taps = tuple(target_layers(drafter))
        model.tap_sink = lambda rows, pos: None
    G.offload(model, 0.0)
    return model, src


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model")
    ap.add_argument("--drafter")
    ap.add_argument("--text", help="a document: the second prompt summarizes it")
    a = ap.parse_args()
    from reclank.models.gemma4.tokenizer import Tokenizer
    model, src = load(a.model, a.drafter)
    tok = (Tokenizer(os.path.join(a.model, "tokenizer.json")) if os.path.isdir(a.model)
           else Tokenizer.from_gguf(src))
    prompts = dict(PROMPTS)
    if a.text:
        prompts["chat"] = "Summarize this text in five bullet points.\n\n" + open(a.text).read()
    bad = 0
    for name, text in prompts.items():
        ids = tok.encode(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                                 thinking=False))
        ctx, grp = ids[:-8], ids[-8:]
        cache = model.new_cache(len(ids) + 16)
        model.forward(ctx, cache=cache, start_pos=0)
        pos = len(ctx)
        steps = np.stack([np.array(model.forward(grp[j:j + 1], cache=cache, start_pos=pos + j),
                                   np.float32).reshape(-1) for j in range(8)])
        cache.truncate(pos)
        lsteps = np.array(model.logits(steps), np.float32)
        for t in range(2, 9):
            x = np.array(model.forward(grp[:t], cache=cache, start_pos=pos), np.float32).reshape(t, -1)
            cache.truncate(pos)
            lx = np.array(model.logits(x), np.float32)
            same = np.array_equal(x, steps[:t]) and np.array_equal(lx, lsteps[:t])
            bad += not same
            print("%s (%d tokens) t=%d: %s (hidden max diff %.3g, logits %.3g)" % (
                name, len(ids), t, "same bits" if same else "DIFFERENT", np.abs(x - steps[:t]).max(),
                np.abs(lx - lsteps[:t]).max()), flush=True)
    print("PASS" if not bad else "FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
