"""Read the model configuration.

The configuration gives the layer types and the sizes. This module makes a
plan for each layer.

Each plan gives:
    head_dim        The size of one attention head.
    num_q_heads     The number of query heads.
    num_kv_heads    The number of key heads and value heads.
    k_eq_v          True for the global layers. These layers use the same
                    tensor for the key and the value.
    sliding_window  The number of visible positions. None for a global layer.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from . import rope as rope_mod


@dataclass
class LayerPlan:
    """Give the geometry of one attention layer."""

    idx: int
    is_sliding: bool
    head_dim: int
    num_q_heads: int
    num_kv_heads: int

    @property
    def q_dim(self):
        """Return the size of the query projection."""
        return self.num_q_heads * self.head_dim

    @property
    def kv_dim(self):
        """Return the size of the key projection and the value projection."""
        return self.num_kv_heads * self.head_dim

    @property
    def k_eq_v(self):
        """Return True for a global layer. A global layer uses key data for value data."""
        return not self.is_sliding

    @property
    def sliding_window(self):
        """Return the window size for a sliding layer. Return None for a global layer."""
        return 1024 if self.is_sliding else None


class Config:
    """Store the model sizes and make one plan for each layer."""

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
        # Scale the input embeddings by sqrt(hidden_size).
        self.embed_scale = self.hidden_size ** 0.5
        self.rope_parameters = tc["rope_parameters"]
        self.layer_types = tc["layer_types"]
        self.plan = [self._plan(i, t) for i, t in enumerate(self.layer_types)]
        self._rope_cache = {}

    def _plan(self, idx, layer_type):
        """Make the plan for one layer. Use the layer type."""
        if layer_type == "sliding_attention":
            return LayerPlan(idx, True, self.head_dim, self.num_attention_heads, self.num_key_value_heads)
        return LayerPlan(idx, False, self.global_head_dim, self.num_attention_heads, self.num_global_key_value_heads)

    @classmethod
    def load(cls, path):
        """Read the configuration from a JSON file."""
        with open(path) as fh:
            return cls(json.load(fh))

    def rope_inv_freq(self, plan):
        """Return the inverse frequencies for one layer. Cache the result by layer type."""
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
