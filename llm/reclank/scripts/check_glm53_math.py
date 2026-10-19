#!/usr/bin/env python3
"""The math of the NumPy model of GLM-5.3-Flash (reclank/models/glm53/model.py) against
transformers' Glm5NextTextDecoderLayer, one layer at a time, both on the same weights (the EXL3
matrices decoded to float32: GLM53CPU(float_weights=True)) and the same input streams (the
reference's dump of the streams before the layer). The difference is float rounding (about -100
dB) unless the math differs. Each layer runs as a prompt of T - 1 tokens, then a step of the last
token on the caches (the conv and delta-rule state of KDA, the latents and index keys of DSA).
With --e3 the layer also runs with the EXL3 products of the runtime (the error of the quant and
the int8 x, for scale).

Needs torch and transformers (glm5_next): the venv of scripts/glm53_reference.py.

    python scripts/check_glm53_math.py /home/cpage/models/GLM-5.3-Flash-exl3-4.05bpw \\
        ~/reclank-golden/glm53/think_layers.npz --layers 0,3,44
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from reclank.models.glm53.model import GLM53CPU  # noqa: E402


def db(got, want):
    e = np.sum((np.asarray(got, np.float64) - want) ** 2)
    return 10 * np.log10(max(e, 1e-300) / np.sum(np.asarray(want, np.float64) ** 2))


def hf_layer(m, i, cfg_hf):
    """transformers' decoder layer i with the float32 weights of the GLM53Source (the experts
    decoded as the router picks them)."""
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
    with torch.device("meta"):
        L = Glm5NextTextDecoderLayer(cfg_hf, i)
    b = "blk.%d." % i
    w = {"input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "ffn_norm.weight"}
    for g, h in (("hc_attn", "attn_hc"), ("hc_ffn", "ffn_hc")):
        for x in ("fn", "base", "scale"):
            w["%s.%s" % (h, x)] = "%s_%s" % (g, x)
    a = "self_attn."
    vals = {}
    if m.cfg.layer_types[i] == "kda":
        qkv = m.F(b + "attn_qkv.weight")
        n = qkv.shape[0] // 3
        for j, c in enumerate("qkv"):
            vals[a + c + "_proj.weight"] = qkv[j * n:(j + 1) * n]
        vals[a + "conv1d.weight"] = m.F(b + "ssm_conv1d.weight")[:, None, :]
        w.update({a + "forget_gate.f_a_proj.weight": "ssm_f_a.weight", a + "forget_gate.f_b_proj.weight": "ssm_f_b.weight",
                  a + "forget_gate.dt_bias": "ssm_dt.bias", a + "forget_gate.A_log": "ssm_a_log",
                  a + "b_proj.weight": "ssm_beta.weight", a + "g_a_proj.weight": "ssm_g_a.weight",
                  a + "g_b_proj.weight": "ssm_g_b.weight", a + "o_norm.weight": "ssm_norm.weight"})
    else:
        w.update({a + "q_a_proj.weight": "attn_q_a.weight", a + "q_a_layernorm.weight": "attn_q_a_norm.weight",
                  a + "q_b_proj.weight": "attn_q_b.weight", a + "kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
                  a + "kv_a_layernorm.weight": "attn_kv_a_norm.weight",
                  a + "indexer.wq_b.weight": "idx_q_b.weight", a + "indexer.wk.weight": "idx_k.weight",
                  a + "indexer.k_norm.weight": "idx_k_norm.weight", a + "indexer.k_norm.bias": "idx_k_norm.bias",
                  a + "indexer.weights_proj.weight": "idx_w.weight",
                  a + "indexer.index_kpool_compress_gate": "idx_gate.weight",
                  a + "indexer.index_kpool_compress_ape": "idx_ape"})
        kb, vb = m.F(b + "attn_k_b.weight"), m.F(b + "attn_v_b.weight")
        vals[a + "kv_b_proj.weight"] = np.concatenate([kb, vb], axis=1).reshape(-1, kb.shape[-1])
    w[a + "o_proj.weight"] = "attn_output.weight"
    if m.cfg.mlp_types[i] == "moe":
        w.update({"mlp.gate.weight": "ffn_gate_inp.weight", "mlp.gate.e_score_correction_bias": "exp_probs_b.bias"})
        for x in ("gate", "up", "down"):
            w["mlp.shared_experts.%s_proj.weight" % x] = "ffn_%s_shexp.weight" % x
    else:
        for x in ("gate", "up", "down"):
            w["mlp.%s_proj.weight" % x] = "ffn_%s.weight" % x
    vals.update({k: m.F(b + g) for k, g in w.items()})
    for n, v in vals.items():
        path, leaf = n.rsplit(".", 1)
        mod = L.get_submodule(path)
        t = torch.from_numpy(np.ascontiguousarray(v, np.float32))
        assert tuple(t.shape) == tuple(getattr(mod, leaf).shape), (n, t.shape, getattr(mod, leaf).shape)
        if leaf in mod._parameters:
            mod._parameters[leaf] = nn.Parameter(t, requires_grad=False)
        else:
            mod._buffers[leaf] = t
    if m.cfg.mlp_types[i] == "moe":
        e_mod = L.mlp.experts
        del e_mod._parameters["gate_up_proj"], e_mod._parameters["down_proj"]

        def forward(hs, top_k_index, top_k_weights):
            final = torch.zeros_like(hs)
            for e in sorted(top_k_index.unique().tolist()):
                tok, slot = torch.where(top_k_index == e)
                wg, wu, wd = (torch.from_numpy(m.g.dequant(b + "ffn_%s_exps.weight" % x, rows=[e])[0])
                              for x in ("gate", "up", "down"))
                cur = e_mod._apply_gate(F.linear(hs[tok], torch.cat([wg, wu])))
                final.index_add_(0, tok, F.linear(cur, wd) * top_k_weights[tok, slot, None])
            return final
        e_mod.forward = forward
    still = [n for n, p in list(L.named_parameters()) + list(L.named_buffers()) if p.is_meta]
    assert not still, still
    return L.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("dump", help="a <name>_layers.npz of scripts/glm53_reference.py")
    ap.add_argument("--layers", default="0,3")
    ap.add_argument("--e3", action="store_true")
    a = ap.parse_args()

    import torch
    from transformers import AutoConfig
    from transformers.cache_utils import DynamicCache
    torch.set_grad_enabled(False)
    m = GLM53CPU(a.model, float_weights=True)
    me = GLM53CPU(m.g) if a.e3 else None
    cfg_hf = AutoConfig.from_pretrained(a.model).text_config
    cfg_hf.dtype = torch.float32
    cfg_hf._attn_implementation = "eager"
    z = np.load(a.dump)
    T = len(z["ids"])
    worst = -1e9
    for i in (int(x) for x in a.layers.split(",")):
        H = z["h%d" % (i - 1)] if i > 0 else np.repeat(z["embed"][:, None, :], m.cfg.hc_mult, axis=1)
        H = H.astype(np.float32)
        L = hf_layer(m, i, cfg_hf)
        cache = DynamicCache(config=cfg_hf)
        outs = []
        for lo, hi in ((0, T - 1), (T - 1, T)):
            hs = torch.from_numpy(H[None, lo:hi])
            mask = torch.ones(1, hi - lo, dtype=torch.bool)
            o, _ = L(hs, attention_mask=mask, past_key_values=cache, use_cache=True)
            outs.append(o[0].numpy())
        want = np.concatenate(outs)
        res = []
        for mod in (m, me):
            if mod is None:
                continue
            c = mod.cache()
            got = []
            for lo, hi in ((0, T - 1), (T - 1, T)):
                mod.index_scores = {}
                got.append(mod.layer(i, H[lo:hi], c, lo))
                c.n = hi
            got = np.concatenate(got)
            res.append((db(got[:T - 1], want[:T - 1]), db(got[T - 1:], want[T - 1:])))
        print("layer %2d %s %s: float weights against transformers: prompt %.1f dB, step %.1f dB%s" % (
            i, m.cfg.layer_types[i], m.cfg.mlp_types[i], res[0][0], res[0][1],
            "; EXL3 products: %.1f / %.1f dB" % res[1] if len(res) > 1 else ""), flush=True)
        worst = max(worst, *res[0])
    ok = worst < -80
    print("check_glm53_math: %s (worst %.1f dB)" % ("OK" if ok else "FAILED", worst))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
