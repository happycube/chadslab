#!/usr/bin/env python3
"""Make reference outputs of the Qwen3.5 MoE model with transformers.

QWEN_PLAN.md, phase 1. The script builds Qwen3_5MoeForCausalLM of
transformers with the first --layers layers, and fills it with the MLX
weights after np_gemma.qwen dequantizes them (float32). Then it runs the
prompt and saves the output of each layer, and the logits of each row, to
an .npz file. scripts/check_qwen.py compares np_gemma.qwen with it.

It needs torch and transformers (the venv of gemma4-12b-qat-pytorch):

    $VENV/bin/python scripts/qwen_reference.py --layers 4 --out ref4.npz
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig  # noqa: E402
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM  # noqa: E402

from np_gemma.qwen import PREFIX, Qwen, QwenConfig  # noqa: E402
from np_gemma.qwen_tok import QwenTokenizer  # noqa: E402

PATH = "models/Qwen3.6-35B-A3B-OptiQ-4bit"
TEXT = ("The quick brown fox jumps over the lazy dog. In 2024, 17 * 23 = 391, "
        "and the cache of a CPU keeps recent data close to the core.")


def state_dict(model, n):
    """The weights of the first n layers, with the names of transformers."""
    sd = {}

    def deq(name):
        return torch.from_numpy(np.ascontiguousarray(model.W(name)))

    def t(name):
        return torch.from_numpy(np.ascontiguousarray(model.t(name)))

    def norm(name):
        # The MLX files keep 1 + w; transformers keeps w and adds 1.
        return t(name) - 1.0

    sd["model.embed_tokens.weight"] = deq("embed_tokens")
    sd["model.norm.weight"] = norm("norm.weight")
    sd["lm_head.weight"] = torch.from_numpy(
        model.mat(None, full="language_model.lm_head").dequant())
    for i in range(n):
        p = "layers.%d." % i
        o = "model." + p
        sd[o + "input_layernorm.weight"] = norm(p + "input_layernorm.weight")
        sd[o + "post_attention_layernorm.weight"] = norm(p + "post_attention_layernorm.weight")
        if model.cfg.layer_types[i] == "full_attention":
            for k in ("q_proj", "k_proj", "v_proj", "o_proj"):
                sd[o + "self_attn.%s.weight" % k] = deq(p + "self_attn." + k)
            for k in ("q_norm", "k_norm"):
                sd[o + "self_attn.%s.weight" % k] = norm(p + "self_attn.%s.weight" % k)
        else:
            la = p + "linear_attn."
            for k in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj"):
                sd[o + "linear_attn.%s.weight" % k] = deq(la + k)
            sd[o + "linear_attn.A_log"] = t(la + "A_log")
            sd[o + "linear_attn.dt_bias"] = t(la + "dt_bias")
            sd[o + "linear_attn.norm.weight"] = t(la + "norm.weight")
            # MLX keeps the convolution as (out, kernel, in); torch as (out, in, kernel).
            sd[o + "linear_attn.conv1d.weight"] = t(la + "conv1d.weight").permute(0, 2, 1).contiguous()
        m = p + "mlp."
        sd[o + "mlp.gate.weight"] = deq(m + "gate")
        sw = {k: model.mat(m + "switch_mlp." + k) for k in ("gate_proj", "up_proj", "down_proj")}
        E = sw["gate_proj"].q.shape[0]
        gu = np.stack([np.concatenate([sw["gate_proj"].dequant(expert=e),
                                       sw["up_proj"].dequant(expert=e)]) for e in range(E)])
        dn = np.stack([sw["down_proj"].dequant(expert=e) for e in range(E)])
        sd[o + "mlp.experts.gate_up_proj"] = torch.from_numpy(gu)
        sd[o + "mlp.experts.down_proj"] = torch.from_numpy(dn)
        for k in ("gate_proj", "up_proj", "down_proj"):
            sd[o + "mlp.shared_expert.%s.weight" % k] = deq(m + "shared_expert." + k)
        sd[o + "mlp.shared_expert_gate.weight"] = deq(m + "shared_expert_gate")
    return sd


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--path", default=PATH)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--text", default=TEXT)
    args = ap.parse_args()
    torch.set_num_threads(max(1, os.cpu_count() // 2))
    cfg = QwenConfig(args.path)
    model = Qwen(args.path, cfg)
    tok = QwenTokenizer(os.path.join(args.path, "tokenizer.json"))
    ids = tok.encode(args.text)
    tc = dict(cfg.raw)
    tc["num_hidden_layers"] = args.layers
    tc["layer_types"] = cfg.layer_types[:args.layers]
    tc.pop("model_type", None)
    hf_cfg = Qwen3_5MoeTextConfig(**tc)
    hf_cfg._attn_implementation = "eager"
    t0 = time.time()
    with torch.device("meta"):
        hf = Qwen3_5MoeForCausalLM(hf_cfg)
    sd = state_dict(model, args.layers)
    missing, unexpected = hf.load_state_dict(sd, strict=False, assign=True)
    print("load %.0f s; missing %s; unexpected %s" % (time.time() - t0, missing, unexpected))
    # The model was made on the meta device; inv_freq of RoPE is not in the
    # state dict (a buffer that is not saved), so compute it again.
    rot = hf.model.rotary_emb
    inv, _ = rot.compute_default_rope_parameters(hf_cfg)
    rot.inv_freq = inv
    rot.original_inv_freq = inv.clone()
    hf = hf.float().eval()
    with torch.no_grad():
        out = hf(torch.tensor([ids]), output_hidden_states=True, use_cache=False)
    hs = [h[0].numpy() for h in out.hidden_states]
    res = {"ids": np.array(ids), "logits": out.logits[0].numpy()}
    # hidden_states: the embeddings, then the output of each layer; the last
    # one has the final norm.
    for i in range(1, args.layers):
        res["layer.%d" % (i - 1)] = hs[i]
    res["final"] = hs[-1]
    np.savez(args.out, **res)
    print("saved %s: %d tokens, %d layers" % (args.out, len(ids), args.layers))


if __name__ == "__main__":
    main()
