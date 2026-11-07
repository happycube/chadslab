"""DiffusionGemma on the Gemma 4 26B runtime (D2 of docs/DIFFUSIONGEMMA_PLAN.md).

The checkpoint's layers are Gemma 4 26B-A4B's, so the Gemma 4 Model loads them
as they are: DGemmaST shows the decoder's tensors under Gemma 4's names, with
the encoder's layer_scalar. The weights are our own Q4_0 (the int4 mode of
gemma4.Model: quantized from the bf16 checkpoint at the first load, then kept
in the weight cache).

- encode: the Gemma 4 prompt path (Model.prefill) over the new ids at the end
  of the cache; it writes their rows and nothing else differs from Gemma 4.
- decode: the canvas through the same layers on the same kernels, in Python
  (Model.linear, _router, _moe), with the decoder's layer_scalar; the
  attention of each layer reads the cached rows (the last window - 1 of a
  layer with a window, all of a global one) and the canvas's own rows, with no
  mask; the cache is not written. The self-conditioning block and the soft
  embedding (the softmax of the last step's processed logits times the token
  table, as the int4 table's float32 values) before the first layer; the tied
  int4 head with the soft cap after the last.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.core import ops
from reclank.encoders import media as MD
from reclank.formats.safetensors import Shards
from reclank.models.gemma4 import model as g4
from reclank.models.gemma4.config import Config

G4 = g4.PREFIX                      # model.language_model.
DEC = "model.decoder."
ENC = "model.encoder.language_model."


class DGemmaST:
    """The bf16 shards under Gemma 4's names: the decoder's layers, token
    table and final norm; layer_scalar is the encoder's. Other names pass
    through (the self-conditioning block, the vision tower)."""

    def __init__(self, path, head="bf16"):
        # head "bf16": the token table (the tied head, the input embedding and
        # the soft embedding) stays bf16 with int4 layers (Model.load_all's
        # keep_embedding_bf16); "int4" quantizes it with the layers.
        self.keep_embedding_bf16 = head == "bf16"
        self.dir = path
        self.sh = Shards(path)
        # the weight cache's key: the index file (its size and time)
        self.path = os.path.join(path, "model.safetensors.index.json")

    def _src(self, name):
        if name.startswith(G4):
            rest = name[len(G4):]
            if rest.endswith(".layer_scalar"):
                return ENC + rest
            return DEC + rest
        return name

    def names(self):
        out = [G4 + n[len(DEC):] for n in self.sh.names() if n.startswith(DEC)]
        return out + [n for n in self.sh.names() if not n.startswith(DEC)]

    def get(self, name, dtype=np.float32):
        src = self._src(name)
        if self.sh.dtype(src) == "BF16":
            return ops.bf16_to_f32(self.sh.get_bf16(src))
        return self.sh.get(src, dtype)

    def get_bf16(self, name):
        return self.sh.get_bf16(self._src(name))

    def get_rows(self, name, start, stop, dtype=np.float32):
        return self.sh.get_rows(self._src(name), start, stop, dtype)

    def get_row(self, name, row, dtype=np.float32):
        return self.sh.get_row(self._src(name), row, dtype)

    def shape(self, name):
        return self.sh.shape(self._src(name))

    def dtype(self, name):
        return self.sh.dtype(self._src(name))

    def release_pages(self):
        for f in self.sh.files.values():
            f.release_pages()

    def close(self):
        self.sh.close()


class DGemma:
    """The encoder (a gemma4.Model in int4) and the decoder's canvas pass."""

    def __init__(self, path, dtype="int4", max_len=8192, head="bf16"):
        raw = json.load(open(os.path.join(path, "config.json")))
        self.raw = raw
        self.gen = json.load(open(os.path.join(path, "generation_config.json")))
        self.cfg = Config(raw)
        self.canvas = raw["canvas_length"]
        self.st = DGemmaST(path, head)
        self.enc = g4.Model(self.st, self.cfg).load_all(dtype)
        self.max_len = max_len
        L = self.cfg.num_hidden_layers
        self.dec_scalar = [float(self.st.sh.get(DEC + "layers.%d.layer_scalar" % i)[0]) for i in range(L)]
        p = DEC + "self_conditioning."
        sc = {k: self.st.sh.get(p + k + ".weight") if self.st.sh.dtype(p + k + ".weight") != "BF16"
              else ops.bf16_to_f32(self.st.sh.get_bf16(p + k + ".weight"))
              for k in ("pre_norm", "gate_proj", "up_proj", "down_proj")}
        self.sc_norm = sc["pre_norm"]
        if dtype == "int4":
            # the self-conditioning MLP in the int4 of the layers
            self.sc_mats = {k: ops.quantize_int4(sc[k]) for k in ("gate_proj", "up_proj", "down_proj")}
        else:
            self.sc_mats = {k: sc[k] for k in ("gate_proj", "up_proj", "down_proj")}
        # the token table's values as the model holds them (float32), for the soft embedding
        m = self.enc
        if m._embed_q is not None:
            self.table = ops.dequantize_int4(m._embed_q, m._embed_s)
        elif m._embed_bf16 is not None:
            self.table = ops.bf16_to_f32(m._embed_bf16)
        else:
            self.table = m._embed
        self.st.release_pages()

    def cache(self):
        return g4.KVCache(self.cfg, max_len=self.max_len)

    # ---- the encoder -------------------------------------------------------
    def encode(self, ids, cache, start, spans=None):
        """Append ids at positions start... to the cache (Gemma 4's prompt pass).

        spans: the soft rows of images (encoders/media.Span), positions
        relative to ids."""
        media = None
        if spans:
            media = [MD.Span(sp.start + start, sp.rows, sp.bidir, sp.kind, sp.key, sp.grid)
                     for sp in spans]
        self.enc.prefill(np.asarray(ids, dtype=np.int64), cache, start=start, media=media)

    # ---- the decoder -------------------------------------------------------
    def _lin(self, x, w):
        return self.enc.linear(np.ascontiguousarray(x, np.float32), w)

    def _sc_lin(self, x, name):
        """The self-conditioning MLP's products: int4 in the int4 mode, else float32."""
        w = self.sc_mats[name]
        if isinstance(w, tuple):
            return self._lin(x, w)
        return np.ascontiguousarray(x, np.float32) @ w.T

    def decoder_input(self, canvas, sc_logits=None):
        eps = self.cfg.rms_norm_eps
        x = self.enc.embed(canvas)
        if sc_logits is not None:
            z = sc_logits - sc_logits.max(-1, keepdims=True)
            p = np.exp(z)
            p /= p.sum(-1, keepdims=True)
            soft = (p @ self.table) * np.float32(self.cfg.embed_scale)
            s = ops.rms_norm(soft, self.sc_norm, eps)
            g = self._sc_lin(s, "gate_proj")
            u = self._sc_lin(s, "up_proj")
            x = x + self._sc_lin(ops.gelu_tanh(g) * u, "down_proj")
        return ops.rms_norm(x, None, eps)

    def _attention(self, h, w, plan, cos, sin, i, cache, n):
        eps = self.cfg.rms_norm_eps
        t = h.shape[0]
        hd = plan.head_dim
        q = np.ascontiguousarray(self._lin(h, w["self_attn.q_proj"])).reshape(t * plan.num_q_heads, hd)
        k = np.ascontiguousarray(self._lin(h, w["self_attn.k_proj"])).reshape(t * plan.num_kv_heads, hd)
        v = k.copy() if plan.k_eq_v else np.ascontiguousarray(
            self._lin(h, w["self_attn.v_proj"])).reshape(t * plan.num_kv_heads, hd)
        ops.qkv_norm_rope(q, w["self_attn.q_norm"], k, w["self_attn.k_norm"], v, cos, sin,
                          plan.num_q_heads, plan.num_kv_heads, hd, eps)
        q = q.reshape(t, plan.num_q_heads, hd)
        k = k.reshape(t, plan.num_kv_heads, hd)
        v = v.reshape(t, plan.num_kv_heads, hd)
        if n:
            base = cache.base[i]
            first = max(base, n - (plan.sliding_window - 1)) if plan.sliding_window else base
            K, V, _ = cache.read(i, n, lo=first - base)
            K = np.concatenate([K, k])
            V = np.concatenate([V, v])
        else:
            K, V = k, v
        nk = plan.num_kv_heads
        rep = plan.num_q_heads // nk
        qb = q.reshape(t, nk, rep, hd).transpose(1, 0, 2, 3).reshape(nk, t * rep, hd)
        s = np.matmul(qb, K.transpose(1, 2, 0))
        s -= s.max(-1, keepdims=True)
        np.exp(s, out=s)
        s /= s.sum(-1, keepdims=True)
        o = np.matmul(s, V.transpose(1, 0, 2))
        o = o.reshape(nk, t, rep, hd).transpose(1, 0, 2, 3).reshape(t, plan.q_dim)
        return self._lin(o, w["self_attn.o_proj"])

    def _layer(self, x, i, cache, n, positions):
        m = self.enc
        cfg = self.cfg
        eps = cfg.rms_norm_eps
        plan = cfg.plan[i]
        w = m.load_layer(i)
        cos, sin, _, _ = m._rope(plan, positions)
        h = ops.rms_norm(x, w["input_layernorm"], eps)
        h = self._attention(h, w, plan, cos, sin, i, cache, n)
        x = x + ops.rms_norm(h, w["post_attention_layernorm"], eps)
        residual = x
        h = ops.rms_norm(x, w["pre_feedforward_layernorm"], eps)
        g = self._lin(h, w["mlp.gate_proj"])
        u = self._lin(h, w["mlp.up_proj"])
        mo = self._lin(ops.gelu_tanh(g) * u, w["mlp.down_proj"])
        h1 = ops.rms_norm(mo, w["post_feedforward_layernorm_1"], eps)
        val, idx = m._router(residual, w)
        h2 = m._moe(ops.rms_norm(residual, w["pre_feedforward_layernorm_2"], eps), w, val, idx)
        h2 = ops.rms_norm(h2, w["post_feedforward_layernorm_2"], eps)
        x = residual + ops.rms_norm(h1 + h2, w["post_feedforward_layernorm"], eps)
        return x * np.float32(self.dec_scalar[i])

    def decode(self, canvas, cache, n, sc_logits=None):
        """The raw logits (with the soft cap) of a canvas at positions n..."""
        positions = np.arange(n, n + len(canvas))
        x = self.decoder_input(np.asarray(canvas, dtype=np.int64), sc_logits)
        for i in range(self.cfg.num_hidden_layers):
            x = self._layer(x, i, cache, n, positions)
        x = ops.rms_norm(x, self.enc._norm_w, self.cfg.rms_norm_eps)
        return self.enc.logits(x)
