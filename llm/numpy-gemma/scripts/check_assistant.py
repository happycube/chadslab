#!/usr/bin/env python3
"""Compare the NumPy assistant (the MTP drafter) with the transformers model.

The script runs the 26B target on a prompt with numpy-gemma. It then feeds
the same token, the same target hidden state, and the same shared keys and
values to the NumPy drafter and to Gemma4AssistantForCausalLM. It runs three
chained draft steps and prints the difference of the logits and of the next
h, and the top token of each side. The context stays shorter than the sliding
window, so both sides see every key.

Run with the Python of ../gemma4-12b-qat-pytorch, which has torch and
transformers:

    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. ../gemma4-12b-qat-pytorch/.venv/bin/python \
        scripts/check_assistant.py
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np

from np_gemma import KVCache, Model
from np_gemma.assistant import Assistant, shared_layers
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer

GGUF_PATH = "models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf"
HUB = "../gemma4-12b-qat-pytorch/.cache/huggingface/hub"
REPO = "models--google--gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant"


def snapshot(repo):
    return sorted(glob.glob(os.path.join(HUB, repo, "snapshots", "*")))[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gguf", default=GGUF_PATH)
    ap.add_argument("--assistant", default=None, help="The snapshot directory.")
    ap.add_argument("--prompt", default="Write a Python function that returns the "
                    "n-th Fibonacci number, then explain it in two sentences.")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--e4b", action="store_true", help="The GGUF file is an E4B model.")
    args = ap.parse_args()
    path = args.assistant or snapshot(REPO)

    import torch
    from transformers import Gemma4AssistantForCausalLM

    g = GGUF(args.gguf)
    tok = Tokenizer.from_gguf(g)
    if args.e4b:
        from np_gemma.e4b import E4B, E4BCache, E4BConfig
        cfg = E4BConfig({"text_config": g.text_config()})
        target = E4B(g, cfg, mode="int4")
    else:
        cfg = Config({"text_config": g.text_config()})
        target = Model(g, cfg).load_all(dtype="int4")
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                   add_generation_prompt=True, thinking=False)
    ids = tok.encode(text)
    cache = E4BCache(cfg, max_len=len(ids) + 8) if args.e4b else KVCache(cfg, max_len=len(ids) + 8)
    x = target.prefill(ids, cache)
    h = x[-1:]
    token = int(np.argmax(target.logits(h)[0]))
    pos = len(ids)
    assert pos < cfg.sliding_window, "the check needs a context below the window"

    shared = {}
    if args.e4b:
        layers = None
        for name in ("sliding_attention", "full_attention"):
            K, V = cache.shared[name]
            # (heads, keys, dim) -> (1, heads, keys, dim)
            shared[name] = (torch.from_numpy(np.ascontiguousarray(K[:, :pos]))[None],
                            torch.from_numpy(np.ascontiguousarray(V[:, :pos]))[None])
    else:
        layers = shared_layers(cfg)
        for name, layer in (("sliding_attention", layers[0]), ("full_attention", layers[1])):
            K, V, base = cache.read(layer, pos)
            assert base == 0
            # (keys, heads, dim) -> (1, heads, keys, dim)
            shared[name] = (torch.from_numpy(np.ascontiguousarray(K.transpose(1, 0, 2)))[None],
                            torch.from_numpy(np.ascontiguousarray(V.transpose(1, 0, 2)))[None])

    hf = Gemma4AssistantForCausalLM.from_pretrained(path, dtype=torch.float32,
                                                    attn_implementation="eager").eval()
    ours = {d: Assistant(path, dtype=d, q8_attn=False) for d in ("f32", "int4")}

    print("target layers %s, prompt %d tokens, first token %d %r" % (
        layers, pos, token, tok.decode([token])))
    states = {d: (token, h) for d in ours}
    ht, hh = token, torch.from_numpy(h)[None]
    worst = 0.0
    for s in range(args.steps):
        emb = torch.from_numpy(target.embed([ht]))[None]
        with torch.no_grad():
            out = hf(inputs_embeds=torch.cat([emb, hh], dim=-1),
                     position_ids=torch.tensor([[pos]]), shared_kv_states=shared)
        lr = out.logits[0, 0].numpy()
        hr = out.last_hidden_state[0].numpy()
        line = "step %d  hf top %6d" % (s, int(lr.argmax()))
        for d, a in ours.items():
            t0, h0 = states[d]
            lo, hn = a.step(target.embed([t0]), h0, pos, cache, layers)
            dl = float(np.abs(lo[0] - lr).max() / np.abs(lr).max())
            dh = float(np.abs(hn - hr).max() / np.abs(hr).max())
            if d == "f32":
                worst = max(worst, dl, dh)
            line += "  | %s top %6d  logits %.1e  h %.1e" % (d, int(lo[0].argmax()), dl, dh)
            # Each side follows its own top token, as in a real draft.
            states[d] = (int(lo[0].argmax()), hn)
        print(line)
        ht, hh = int(lr.argmax()), out.last_hidden_state
    print("f32 worst relative difference %.1e -> %s" % (worst, "PASS" if worst < 1e-4 else "FAIL"))
    g.close()


if __name__ == "__main__":
    main()
