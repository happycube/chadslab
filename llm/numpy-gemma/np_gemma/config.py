"""Gemma 4 12B unified text configuration and per-layer plan."""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import rope as rope_mod


@dataclass
class LayerPlan:
    idx: int
    is_sliding: bool
    head_dim: int
    num_q_heads: int
    num_kv_heads: int

    @property
    def q_dim(self):
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self):
        return self.num_kv_heads * self.head_dim

    @property
    def k_eq_v(self):
        return not self.is_sliding

    @property
    def sliding_window(self):
        return 1024 if self.is_sliding else None


class Config:
    def __init__(self, cfg):
        tc = cfg["text_config"]
        self.hidden_size = tc["hidden_size"]
        self.intermediate_size = tc["intermediate_size"]
        self.num_hidden_layers = tc["num_hidden_layers"]
        self.num_attention_heads = tc["num_attention_heads"]
        self.num_key_value_heads = tc["num_key_value_heads"]
        self.head_dim = tc["head_dim"]
        self.global_head_dim = tc["global_head_dim"]
        self.num_global_key_value_heads = tc.get("num_global_key_value_heads", 1)
        self.rms_norm_eps = tc["rms_norm_eps"]
        self.vocab_size = tc["vocab_size"]
        self.max_position_embeddings = tc["max_position_embeddings"]
        self.sliding_window = tc["sliding_window"]
        self.final_logit_softcapping = tc.get("final_logit_softcapping")
        self.embed_scale = self.hidden_size ** 0.5
        self.rope_parameters = tc["rope_parameters"]
        self.layer_types = tc["layer_types"]
        self.plan = [self._plan(i, t) for i, t in enumerate(self.layer_types)]
        self._rope_cache = {}

    def _plan(self, idx, layer_type):
        if layer_type == "sliding_attention":
            return LayerPlan(idx, True, self.head_dim, self.num_attention_heads, self.num_key_value_heads)
        return LayerPlan(idx, False, self.global_head_dim, self.num_attention_heads, self.num_global_key_value_heads)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls(json.load(fh))

    def rope_inv_freq(self, plan):
        key = "sliding_attention" if plan.is_sliding else "full_attention"
        if key in self._rope_cache:
            return self._rope_cache[key]
        params = self.rope_parameters[key]
        if plan.is_sliding:
            inv = rope_mod.default_inv_freq(plan.head_dim, params["rope_theta"])
        else:
            inv = rope_mod.proportional_inv_freq(
                plan.head_dim, params["rope_theta"], params.get("partial_rotary_factor", 1.0)
            )
        self._rope_cache[key] = inv
        return inv
