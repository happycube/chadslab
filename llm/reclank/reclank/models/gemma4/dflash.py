"""The DFlash drafter of Nemotron 3.5 Bolt E4B (NVIDIA-Nemotron-3.5-Bolt-E4B-BF16-DFlash-NVFP4): block
drafting for the speculative decoding of the E4B (e4b.E4B; the session's loop, core/speculative.stream,
as the assistant's MTP).

The math follows llama.cpp's src/models/dflash.cpp and the DFlash driver of common/speculative.cpp
(MIT License; Copyright (c) 2023-2026 The ggml authors; see THIRD_PARTY_NOTICES.md), as the drafter
of Bolt 74B (models/nemotron/lilicorr.py) does, without its grouped conv, its attention sinks and its
lattice head:

- The context: for each position the target has run, the target's outputs of layers 5, 11, 17, 23
  and 35 (config target_layer_ids; transformers' hidden_states[i + 1]), concatenated (5 x 2560),
  through fc and hidden_norm: g. Each drafter layer keeps K = rope(k_norm(k_proj(g))) and V =
  v_proj(g) of the position (a ring of RING positions; the window is 1024).
- A draft: the block [last token, mask x 7] (mask: token 4) at the positions pos0 .. pos0 + 7, the
  target's token embedding rows times sqrt(hidden) (Gemma's scale, llama.cpp's embedding_scale),
  through 5 Qwen3 layers: q and k norms, NEOX rope (theta 1e6), attention over the context's window
  and the whole block (not causal), SwiGLU MLP; the final norm and the drafter's own lm_head; the
  argmax of the 7 mask slots are the drafts (in place: slot j proposes the token of pos0 + j).

The matrices are NVFP4 (W4A16: E2M1 codes in groups of 16 with an E4M3 scale, a float32 scale of the
matrix), the norms bfloat16. The programs (DFlashKQ, DFlashGPU) keep the codes as they are (KQ_NV4
rows, cops.kq_nv4_pack) and take x as the runtime's NVFP4 products do (the GPU: int8 planes,
NM_QX + NM_GEMM8, KQ_LINEAR for one row). The target gives the rows through its taps (E4B.taps,
E4B.tap_sink): each run of the target, the rows of every position it ran go into the context
(a rejected draft's position is never before the next block, and a later run writes it again).
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.formats.safetensors import SafeTensors
from reclank.models.nemotron.lilicorr import RING, LiLiCorrKQ, _rope
from reclank.models.nemotron.model import nvfp4, rms_norm, silu


def is_dflash(path):
    """True when path is a DFlash drafter's folder (config.json: DFlashDraftModel)."""
    p = os.path.join(path, "config.json") if os.path.isdir(path) else None
    if p is None or not os.path.exists(p):
        return False
    return "DFlashDraftModel" in (json.load(open(p)).get("architectures") or ())


def target_layers(path):
    return [int(v) for v in json.load(open(os.path.join(path, "config.json")))["dflash_config"]["target_layer_ids"]]


class DFlashCache:
    """The NumPy drafter's context: K and V of each layer by position (a ring)."""

    def __init__(self, d):
        self.K = [np.zeros((RING, d.kv, d.hd), np.float32) for _ in range(d.layers)]
        self.V = [np.zeros((RING, d.kv, d.hd), np.float32) for _ in range(d.layers)]
        self.pos = np.full(RING, -1, np.int64)


class DFlash:
    """The NumPy reference: inject(cache, rows, positions) after the target runs positions (rows: its
    taps); draft(cache, last, pos0, n) the next tokens. embed(ids): the target's embedding rows (with
    Gemma's scale)."""

    def __init__(self, path, embed):
        c = json.load(open(os.path.join(path, "config.json")))
        f = c["dflash_config"]
        self.embed = embed
        self.hidden, self.layers = c["hidden_size"], c["num_hidden_layers"]
        self.heads, self.kv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps, self.theta = float(c["rms_norm_eps"]), float(c["rope_theta"])
        self.block, self.mask = int(c["block_size"]), int(f["mask_token_id"])
        self.target_ids = [int(v) for v in f["target_layer_ids"]]
        self.window = int(f["swa_window_size"])
        assert not f.get("causal", False), "a causal DFlash block is not supported"
        self.st = SafeTensors(os.path.join(path, "model.safetensors"))
        self._w, self._f = {}, {}

    def f32(self, name):
        v = self._f.get(name)
        if v is None:
            v = self._f[name] = np.ascontiguousarray(self.st.get(name, dtype=np.float32))
        return v

    def W(self, name):
        """A matrix [rows, cols] as float32 (NVFP4: the values of nvfp4(); bfloat16: as stored)."""
        w = self._w.get(name)
        if w is None:
            if name + ".weight_scale" in self.st.header:
                w = nvfp4(self.st.get(name + ".weight", dtype=None), self.st.get(name + ".weight_scale", dtype=None),
                          float(np.asarray(self.f32(name + ".weight_scale_2")).reshape(-1)[0]))
            else:
                w = self.f32(name + ".weight")
            w = self._w[name] = np.ascontiguousarray(w)
        return w

    def lin(self, x, name):
        return x @ self.W(name).T

    def cache(self):
        return DFlashCache(self)

    def features(self, rows):
        """g of the target's rows [n, 5 x hidden]."""
        return rms_norm(self.lin(rows, "fc"), self.f32("hidden_norm.weight"), self.eps)

    def inject(self, cache, rows, positions):
        g = self.features(np.asarray(rows, np.float32))
        positions = np.asarray(positions)
        for i in range(self.layers):
            p = "layers.%d.self_attn." % i
            K = rms_norm(self.lin(g, p + "k_proj").reshape(-1, self.kv, self.hd), self.f32(p + "k_norm.weight"),
                         self.eps)
            cache.K[i][positions % RING] = _rope(K, positions, self.theta)
            cache.V[i][positions % RING] = self.lin(g, p + "v_proj").reshape(-1, self.kv, self.hd)
        cache.pos[positions % RING] = positions
        return g

    def _attn(self, cache, i, q, k, v, positions):
        """Each query: the context keys of its window (the positions before the block) and every key
        of the block."""
        pos0 = positions[0]
        ctx = np.nonzero((cache.pos >= 0) & (cache.pos < pos0))[0]
        out = np.empty_like(q)
        rep = self.heads // self.kv
        for j, p in enumerate(positions):
            sel = ctx[p - cache.pos[ctx] < self.window]
            K = np.concatenate([cache.K[i][sel], k], 0)
            V = np.concatenate([cache.V[i][sel], v], 0)
            for h in range(self.heads):
                s = (K[:, h // rep] @ q[j, h]) / np.sqrt(self.hd)
                e = np.exp(s - s.max())
                out[j, h] = (e @ V[:, h // rep]) / e.sum()
        return out

    def block_forward(self, cache, last, pos0):
        """The drafter on [last, mask x 7] at pos0 ..: the normed hidden states [block, hidden]."""
        ids = [int(last)] + [self.mask] * (self.block - 1)
        positions = np.arange(pos0, pos0 + self.block)
        x = np.asarray(self.embed(ids), np.float32)
        for i in range(self.layers):
            p = "layers.%d." % i
            a = rms_norm(x, self.f32(p + "input_layernorm.weight"), self.eps)
            q = rms_norm(self.lin(a, p + "self_attn.q_proj").reshape(-1, self.heads, self.hd),
                         self.f32(p + "self_attn.q_norm.weight"), self.eps)
            k = rms_norm(self.lin(a, p + "self_attn.k_proj").reshape(-1, self.kv, self.hd),
                         self.f32(p + "self_attn.k_norm.weight"), self.eps)
            v = self.lin(a, p + "self_attn.v_proj").reshape(-1, self.kv, self.hd)
            q, k = _rope(q, positions, self.theta), _rope(k, positions, self.theta)
            o = self._attn(cache, i, q, k, v, positions)
            x = x + self.lin(o.reshape(len(ids), -1), p + "self_attn.o_proj")
            m = rms_norm(x, self.f32(p + "post_attention_layernorm.weight"), self.eps)
            f = silu(self.lin(m, p + "mlp.gate_proj")) * self.lin(m, p + "mlp.up_proj")
            x = x + self.lin(f, p + "mlp.down_proj")
        return rms_norm(x, self.f32("norm.weight"), self.eps)

    def logits(self, hid):
        return self.lin(hid, "lm_head")

    def draft_cache(self, cache, last, pos0, n=None):
        """Up to n (block - 1) tokens after last (at pos0; the context holds the positions before it)."""
        hid = self.block_forward(cache, last, pos0)
        return [int(t) for t in self.logits(hid[1:]).argmax(-1)[:n or self.block - 1]]


class DFlashKQ(DFlash):
    """The drafter as programs on the CPU (NM_DPREP, NM_DATTN with no sink, the NVFP4 matrices in
    KQ_NV4 rows as LiLiCorrKQ's, its lm_head and ARGMAX of the 7 slots). target: the E4B; its taps
    feed inject (E4B.tap_sink)."""

    def __init__(self, path, target, gpu=False):
        from reclank.backends.cpu import cops
        from reclank.models.gemma4.e4b import PREFIX
        cfg = target.cfg
        super().__init__(path, lambda ids: target.embed_rows(PREFIX + "embed_tokens", np.asarray(ids))
                         * cfg.embed_scale)
        assert self.hidden == cfg.hidden_size, "the drafter's hidden size is not the target's"
        self.gpu = gpu
        self._m = {}
        self._cops = cops
        self.progs = {}
        self.ring_k = [np.zeros((RING, self.kv, self.hd), np.float32) for _ in range(self.layers)]
        self.ring_v = [np.zeros((RING, self.kv, self.hd), np.float32) for _ in range(self.layers)]
        # NM_DATTN's sink logits: none (-1e30 adds exp(-1e30 - m) = 0 to each sum)
        self.nosink = np.full(self.heads, -1e30, np.float32)
        self.stop = 0.0
        self.on_gpu = gpu

    # the NVFP4 rows and their products: LiLiCorrKQ's
    bf16, M, _lin = LiLiCorrKQ.bf16, LiLiCorrKQ.M, LiLiCorrKQ._lin

    def _scr(self, t):
        wide = max(len(self.target_ids) * self.hidden, self.heads * self.hd, 2 * self.hidden * 4)
        return dict(xq=np.zeros((t, wide), np.int8), xl=np.zeros((t, wide), np.int8),
                    xs=np.zeros((t, wide // 32), np.float32), xm=np.zeros((t, wide // 16), np.float32),
                    x16=np.zeros((t, wide), np.int16))

    def _inject_prog(self, t):
        """The context of t positions from pos: g, then k and v of each layer into the rings."""
        from reclank.ir import program as P
        prog = P.Program()
        scr = self._scr(t)
        H, nk = self.hidden, self.kv * self.hd
        feat, g = np.zeros((t, len(self.target_ids) * H), np.float32), np.zeros((t, H), np.float32)
        k, v = np.zeros((t, nk), np.float32), np.zeros((t, nk), np.float32)
        pos = prog.slot("pos")
        self._lin(prog, "fc", feat, g, t, scr)
        prog.emit(P.RMS_NORM, g, self.f32("hidden_norm.weight"), g, t, H, self.eps)
        for i in range(self.layers):
            p = "layers.%d.self_attn." % i
            self._lin(prog, p + "k_proj", g, k, t, scr)
            self._lin(prog, p + "v_proj", g, v, t, scr)
            prog.emit(P.NM_DPREP, None, k, v, None, self.f32(p + "k_norm.weight"), self.ring_k[i], self.ring_v[i], None,
                      t, self.heads, self.kv, self.hd, pos, RING, self.theta, self.eps)
        prog.names.update(feat=feat, g=g)
        prog.keep += [scr, k, v] + self.ring_k + self.ring_v
        return prog.finish()

    def _draft_prog(self):
        """The block of t = block tokens from pos (their embedding rows in x) to the tokens of the 7
        slots (tok: the argmax of the lm_head's rows)."""
        from reclank.ir import program as P
        t, H, nq, nk = self.block, self.hidden, self.heads * self.hd, self.kv * self.hd
        prog = P.Program()
        scr = self._scr(t)
        f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        x, a = f32(t, H), f32(t, H)
        q, k, v, qo, o, o2 = f32(t, nq), f32(t, nk), f32(t, nk), f32(t, nq), f32(t, nq), f32(t, H)
        I = self.M("layers.0.mlp.gate_proj")[2]
        gt, up, mm = f32(t, I), f32(t, I), f32(t, I)
        ascr = f32(t * self.heads * (self.window + t))
        hid = f32(t, H)
        V = self.M("lm_head")[2]
        lg, tok = f32(t - 1, V), np.zeros(t - 1, np.int32)
        pos = prog.slot("pos")
        for i in range(self.layers):
            p = "layers.%d." % i
            prog.emit(P.RMS_NORM, x, self.f32(p + "input_layernorm.weight"), a, t, H, self.eps)
            self._lin(prog, p + "self_attn.q_proj", a, q, t, scr)
            self._lin(prog, p + "self_attn.k_proj", a, k, t, scr)
            self._lin(prog, p + "self_attn.v_proj", a, v, t, scr)
            prog.emit(P.NM_DPREP, q, k, v, self.f32(p + "self_attn.q_norm.weight"),
                      self.f32(p + "self_attn.k_norm.weight"), self.ring_k[i], self.ring_v[i], qo, t, self.heads,
                      self.kv, self.hd, pos, RING, self.theta, self.eps)
            prog.emit(P.NM_DATTN, qo, self.ring_k[i], self.ring_v[i], self.nosink, o,
                      t, self.heads, self.kv, self.hd, pos, RING, self.window, ascr)
            self._lin(prog, p + "self_attn.o_proj", o, o2, t, scr)
            prog.emit(P.ADD, x, o2, x, t * H)
            prog.emit(P.RMS_NORM, x, self.f32(p + "post_attention_layernorm.weight"), a, t, H, self.eps)
            self._lin(prog, p + "mlp.gate_proj", a, gt, t, scr)
            self._lin(prog, p + "mlp.up_proj", a, up, t, scr)
            prog.emit(P.ENC_SILU_MUL, gt, up, mm, t * I)
            self._lin(prog, p + "mlp.down_proj", mm, o2, t, scr)
            prog.emit(P.ADD, x, o2, x, t * H)
        prog.emit(P.RMS_NORM, x, self.f32("norm.weight"), hid, t, H, self.eps)
        self._lin(prog, "lm_head", hid[1:], lg, t - 1, scr)
        for j in range(t - 1):
            prog.emit(P.ARGMAX, lg[j], V, tok[j:j + 1])
        prog.names.update(x=x, hid=hid, tok=tok)
        prog.keep += [scr, ascr, lg] + self.ring_k + self.ring_v
        return prog.finish()

    def _prog(self, key, make):
        p = self.progs.get(key)
        if p is None:
            p = self.progs[key] = make()
        return p

    def _run(self, key, make, inputs, pos, outs):
        prog = self._prog(key, make)
        for name, a in inputs.items():
            prog.names[name][:] = a
        prog.bind(pos=int(pos))
        prog.run()
        return [prog.names[n] for n in outs]

    def inject_rows(self, rows, pos0):
        """The target's taps of the positions pos0 .. (E4B.tap_sink) into the context."""
        rows = np.asarray(rows, np.float32)
        c0 = 0
        while c0 < len(rows):
            n = len(rows) - c0
            t = next(s for s in (256, 64, 16, 8, 4, 2, 1) if s <= n)
            self._run(("inject", t), lambda t=t: self._inject_prog(t), {"feat": rows[c0:c0 + t]}, pos0 + c0, ())
            c0 += t

    def draft(self, target, token, h, pos, cache, n, eos_ids=()):
        """Up to n drafts after token at pos (assistant.Assistant's signature: the session's loop)."""
        ids = [int(token)] + [self.mask] * (self.block - 1)
        tok, = self._run("draft", self._draft_prog, {"x": self.embed(ids)}, pos, ("tok",))
        out = []
        for t in tok[:min(n, self.block - 1)]:
            out.append(int(t))
            if int(t) in eos_ids:
                break
        return out

    def release(self):
        self.progs = {}


class DFlashGPU(DFlashKQ):
    """The programs on the GPU (the GPU records; the rings and the KQ_NV4 rows on the device), on the
    target's E4BGPU mirror; the draft program's 7 tokens are the only copy back."""

    def __init__(self, path, target):
        super().__init__(path, target, gpu=True)
        self.target = target

    def _mirror(self):
        return self.target._gpu.g.mirror

    def _run(self, key, make, inputs, pos, outs):
        from reclank.backends.cuda.runtime import GPUProgram
        e = self.progs.get(key)
        if e is None:
            prog = make()
            e = self.progs[key] = (prog, GPUProgram(prog, graph=True, mirror=self._mirror()))
        prog, g = e
        for name, a in inputs.items():
            prog.names[name][:] = a
            g.upload(name)
        g.bind({"pos": int(pos)})
        g.run()
        for n in outs:
            g.download(n)
        return [prog.names[n] for n in outs]

    def release(self):
        for _prog, g in self.progs.values():
            g.close()
        self.progs = {}
