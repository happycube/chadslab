#!/usr/bin/env python3
"""Run the Hugging Face Gemma 4 model with the weights of a GGUF file.

Use this script to compare the NumPy runtime with the Hugging Face reference.
The script loads the GGUF weights into the Hugging Face model class. Thus it
does not need the full unquantized checkpoint.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

from np_gemma.gguf import GGUF
from np_gemma.tokenizer import Tokenizer


def make_config(tc):
    """Build a Hugging Face Gemma 4 text config from the GGUF metadata."""
    return Gemma4TextConfig(
        vocab_size=tc["vocab_size"], hidden_size=tc["hidden_size"],
        intermediate_size=tc["intermediate_size"], num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"], num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"], global_head_dim=tc["global_head_dim"],
        num_global_key_value_heads=tc["num_global_key_value_heads"],
        rms_norm_eps=tc["rms_norm_eps"], sliding_window=tc["sliding_window"],
        layer_types=tc["layer_types"], attention_k_eq_v=True, enable_moe_block=True,
        num_experts=tc["num_experts"], top_k_experts=tc["top_k_experts"],
        moe_intermediate_size=tc["moe_intermediate_size"],
        max_position_embeddings=tc["max_position_embeddings"],
        final_logit_softcapping=tc["final_logit_softcapping"],
        hidden_size_per_layer_input=0, num_kv_shared_layers=0,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--ids", default="818,5279,529,7001,563")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    g = GGUF(args.gguf)
    known = set(g.names())
    cfg = make_config(g.text_config())
    dtype = getattr(torch, args.dtype)
    torch.set_default_dtype(dtype)
    model = Gemma4ForCausalLM(cfg)
    model.eval()
    t0 = time.perf_counter()
    count = 0
    with torch.no_grad():
        for n, p in list(model.named_parameters()) + list(model.named_buffers()):
            if not n.startswith("model."):
                continue
            rt = "model.language_model." + n[len("model."):]
            if rt not in known:
                continue
            p.copy_(torch.from_numpy(g.get(rt)))
            count += 1
    print("weights loaded %d tensors in %.1f s" % (count, time.perf_counter() - t0), flush=True)

    ids = [int(x) for x in args.ids.split(",")]
    input_ids = torch.tensor([ids])
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits[0, -1].float().numpy()
    print("forward %.1f s" % (time.perf_counter() - t0), flush=True)
    top = np.argsort(-logits)[:8]
    tok = Tokenizer(args.tokenizer)
    print("HF top8:", [(int(i), tok.decode([int(i)]), round(float(logits[i]), 3)) for i in top])
    g.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
