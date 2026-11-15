"""The DFlash2 drafter of Nemotron 3.5 Parse (the checkpoint's dflash2/: DFlash2DraftModel, 89M parameters):
block drafting with a candidate selector for the speculative decoding of the Parse decoder
(docs/NEMOTRON_PARSE.md). The math is that of reclank's Qwen3.8 DFlash 2 drafter (models/qwen/dflash2.py, the
NumPy reference class DFlash2), whose architecture it is; the details that differ follow vLLM's native
DFlash2 for Parse (vllm/model_executor/models/qwen3_dflash.py, qwen3_dflash2.py, nemotron_parse3_5.py and
vllm/v1/worker/gpu/spec_decode/dflash, dflash2; Apache License 2.0; Copyright contributors to the vLLM
project; see THIRD_PARTY_NOTICES.md):

- The context: for each position the target has run (the visual rows, the prompt, the output so far), the
  target's residual stream after its layers 1, 4 and 8 (target_layer_ids: vLLM's aux hidden states 2, 5 and
  9, counted from the embeddings' layernorm as 0), concatenated (3 x 1024), through fc and hidden_norm; each
  drafter layer keeps K = rope(k_norm(k_proj(g))) and V = v_proj(g) of the position. No window: a block
  attends to every position before it.
- The positions are the target's: the visual rows 0 .. N - 1, then the prompt and the output (the target has
  no position embeddings; the drafter's RoPE, NEOX of theta 1e6, takes these).
- A draft: the block [the last token, mask (token 3) x K] at the positions pos0 .. pos0 + K (pos0: the first
  position the target has not run), the embeddings of the target (embed_tokens x 32: vLLM shares the target's
  scaled embedding module), 3 Qwen3 layers with the grouped dynamic conv (2 taps, groups of 16, within the
  block), attention not causal over the context and the whole block, SwiGLU; the final norm; the target's head
  (the tied embedding) on the K mask slots, the 16 best tokens of each, the selector's greedy walk from the
  last token.

The output is lossless: the drafts are only proposals, the target's steps check them (engine.ParseEngine).

    d = ParseDFlash2("MODELS/Nemotron-3.5-Parse-F20/dflash2", embed, head)
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.models.qwen.dflash2 import DFlash2

# the names of the Qwen drafter's GGUF (models/qwen/dflash2.py) -> the checkpoint's
_NAMES = {
    "fc.weight": "fc.weight", "enc.output_norm.weight": "hidden_norm.weight", "output_norm.weight": "norm.weight",
    "selector_hidden.weight": "candidate_selector.hidden_projection.weight",
    "selector_predecessor.weight": "candidate_selector.predecessor_codebook",
    "selector_successor.weight": "candidate_selector.successor_codebook",
}
_LAYER = {
    "attn_norm.weight": "input_layernorm.weight", "attn_conv_proj.weight": "attention_conv.kernel_projection.weight",
    "attn_conv_base": "attention_conv.base_kernel", "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight", "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight", "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight", "ffn_norm.weight": "post_attention_layernorm.weight",
    "ffn_conv_proj.weight": "mlp_conv.kernel_projection.weight", "ffn_conv_base": "mlp_conv.base_kernel",
    "ffn_gate.weight": "mlp.gate_proj.weight", "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
DRAFTER_FILE = "Nemotron-3.5-Parse-DFlash2-BF12.gguf"     # scripts/convert_nemoparse_f20.py


def hf_name(name):
    """The checkpoint's name of a tensor named as the Qwen drafter's GGUF."""
    if name.startswith("blk."):
        _, i, rest = name.split(".", 2)
        return "layers.%s.%s" % (i, _LAYER[rest])
    return _NAMES[name]


class ParseDFlash2(DFlash2):
    """The NumPy reference (float32) of the drafter of a dflash2 directory (the converted BF12 GGUF, else the
    download's bfloat16 safetensors): inject(cache, taps, positions) the target's taps [n, 3 x 1024] of
    positions; draft(cache, last, pos0, n) the next tokens. embed(ids): the target's scaled embedding rows;
    head(h): the target's logits."""

    def __init__(self, path, embed=None, head=None, block=None):
        cfg = json.load(open(os.path.join(path, "config.json")))
        dc = cfg["dflash_config"]
        assert not dc.get("causal") and dc.get("projector_type") == "dflash2", "a DFlash2 drafter, not causal"
        self.path = path
        self.embed, self.head = embed, head
        self.hidden, self.layers = int(cfg["hidden_size"]), int(cfg["num_hidden_layers"])
        self.heads, self.kv, self.hd = int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"]), int(cfg["head_dim"])
        self.inter = int(cfg["intermediate_size"])
        self.eps, self.theta = float(cfg["rms_norm_eps"]), float(cfg["rope_theta"])
        self.block = int(block or dc["block_size"])                 # 33: the last token and 32 drafts
        self.window = 1 << 30
        self.ck, self.cg = int(dc["conv_kernel_size"]), int(dc["conv_group_size"])
        self.rank, self.top_k = int(dc["selector_rank"]), int(dc["selector_top_k"])
        self.mask = int(dc["mask_token_id"])
        self.taps = [int(v) for v in dc["target_layer_ids"]]      # the target's layers whose outputs feed fc
        self.max_pos = int(cfg["max_position_embeddings"])
        self.stop = 0.0
        self._f = {}
        self._sel = None
        f = os.path.join(path, DRAFTER_FILE)
        if os.path.exists(f):
            from reclank.formats.gguf import GGUF
            self.g, self.st = GGUF(f, stage=False), None
        else:
            from reclank.formats.safetensors import SafeTensors
            self.g, self.st = None, SafeTensors(os.path.join(path, "model.safetensors"))

    def f32(self, name):
        v = self._f.get(name)
        if v is None:
            n = hf_name(name)
            a = self.g.dequant(n) if self.g is not None else self.st.get(n)
            v = self._f[name] = np.ascontiguousarray(a, np.float32)
        return v

    def cache(self):
        n = self.max_pos
        return dict(K=[np.zeros((n, self.kv, self.hd), np.float32) for _ in range(self.layers)],
                    V=[np.zeros((n, self.kv, self.hd), np.float32) for _ in range(self.layers)],
                    pos=np.full(n, -1, np.int64))

    def inject(self, cache, rows, positions):
        """The context of the target's taps rows at positions (cache rows by position: no ring)."""
        from reclank.models.nemotron.lilicorr import _rope
        from reclank.models.nemotron.model import rms_norm
        g = rms_norm(self.lin(np.asarray(rows, np.float32), "fc"), self.f32("enc.output_norm.weight"), self.eps)
        positions = np.asarray(positions)
        for i in range(self.layers):
            b = "blk.%d." % i
            K = rms_norm(self.lin(g, b + "attn_k").reshape(-1, self.kv, self.hd), self.f32(b + "attn_k_norm.weight"),
                         self.eps)
            cache["K"][i][positions] = _rope(K, positions, self.theta)
            cache["V"][i][positions] = self.lin(g, b + "attn_v").reshape(-1, self.kv, self.hd)
        cache["pos"][positions] = positions
        return g


def rope_tables(positions, hd, theta):
    """cos, sin [n, hd / 2] float32 of NEOX RoPE at positions (lilicorr._rope's angles)."""
    inv = theta ** (-np.arange(0, hd, 2, dtype=np.float64) / hd)
    ang = np.asarray(positions, np.float64)[:, None] * inv[None, :]
    return np.ascontiguousarray(np.cos(ang), np.float32), np.ascontiguousarray(np.sin(ang), np.float32)


class ParseDrafter(ParseDFlash2):
    """The drafter as programs of the engine's decoder runner (ParseEngine.dec_runner: one mirror with the
    target's), in C on the CPU or on the GPU; its matrices in BF12 (the bfloat16 values exactly: KQ_BF12X16
    on the CPU, BF12 rows on the GPU), the target's head and embedding rows. Two programs:

        ("inj", R)          the context of R target rows: "tin" [R, 3 x 1024] (the target's taps),
                            "ipos" their positions, "icos", "isin" (their RoPE): fc, hidden_norm, each layer's
                            k (k_norm, RoPE) and v into its cache rows ("dk<l>", "dv<l>": max_seq rows)
        ("drf", B, span)    a block of B = 1 + K rows at pos0: "dtok" [last, mask x K], "dpos" (pos0 ..),
                            "dhi" (pos0 + B for every row: the context and the whole block), "dcos", "dsin":
                            3 layers (the block's own k and v written into the caches at its rows first; a
                            later inject writes the real context there), the final norm, then of the K mask
                            rows the head's logits ("dlog" [K, vocab]) and the selector's gates ("dgate"
                            [K, 256])

    The conv's coefficients are linears too: kernel_projection's row of (side, tap, group) repeated for
    each of the group's 16 channels, the base kernel its bias; a side's conv is c0 x + c1 shift(x) (the
    shift within the block: ENC_GATHER), GP_MUL the products. The host takes the 16 best of each mask row
    and walks the selector (DFlash2.walk)."""

    def __init__(self, engine, path, block=None):
        e = engine
        super().__init__(path, embed=lambda ids: e.embed[np.asarray(ids, np.int64)] * np.float32(math.sqrt(e.width)),
                         head=None, block=block)
        assert self.g is not None, "the drafter's BF12 file (scripts/convert_nemoparse_f20.py)"
        assert self.hidden == e.width and len(self.taps) * self.hidden == self._shape("fc.weight")[1]
        self.e, self.gpu = e, e.gpu
        H = self.hidden
        ng = H // self.cg
        grp = np.arange(H) // self.cg
        L = []
        for i in range(self.layers):
            b = "blk.%d." % i
            d = dict(n1=self.f32(b + "attn_norm.weight"), n2=self.f32(b + "ffn_norm.weight"),
                     qn=self.f32(b + "attn_q_norm.weight"), kn=self.f32(b + "attn_k_norm.weight"),
                     q=self._lin(b + "attn_q"), k=self._lin(b + "attn_k"), v=self._lin(b + "attn_v"),
                     o=self._lin(b + "attn_output"), gate=self._lin(b + "ffn_gate"), up=self._lin(b + "ffn_up"),
                     down=self._lin(b + "ffn_down"))
            for kind, conv in (("attn", "a"), ("ffn", "f")):
                base = self.conv_base(i, kind)                          # [side, tap, hidden]
                for s in range(2):
                    for t in range(self.ck):
                        d["%s%d%d" % (conv, s, t)] = self._lin(b + kind + "_conv_proj",
                                                               rows=(s * self.ck + t) * ng + grp, bias=base[s, t])
            L.append(d)
        self.L = L
        self.fc = self._lin("fc")
        self.hn, self.fn = self.f32("enc.output_norm.weight"), self.f32("output_norm.weight")
        self.gate = self._lin("selector_hidden")
        self.qscale = np.full(self.heads * self.hd, self.hd ** -0.5, np.float32)
        self.sc = np.full(H, math.sqrt(e.width), np.float32)
        self.dcache = None
        self.selector()                     # (the codebooks, float32, for the host's walk)

    def _shape(self, name):
        dims = self.g.tensors[hf_name(name)][0]
        return int(dims[1]), int(dims[0])

    def _lin(self, name, rows=None, bias=None):
        """A linear of the BF12 matrix name (rows: an index array of its rows) and a float32 bias."""
        from reclank.backends.cpu import cops
        from reclank.encoders.runtime import _Plain
        from reclank.formats.gguf import BF12
        from reclank.models.gemma4.e4b import _KQ
        raw, dims, t = self.g.raw(hf_name(name + ".weight"))
        assert t == BF12, "%s is not BF12" % name
        k, m = int(dims[0]), int(dims[1])
        data = np.asarray(raw).reshape(m, -1)
        if rows is not None:
            data, m = data[rows], len(rows)
        data = np.ascontiguousarray(data).reshape(-1)
        lin = _Plain(None)
        if self.gpu:
            lin.kq = _KQ(data, BF12, m, k)
        else:
            lin.bf12 = _KQ(cops.kq_pack_bf12x16(data, m, k), cops.KQ_BF12X16, m, k)
        lin.b = None if bias is None else np.ascontiguousarray(bias, np.float32)
        return lin

    def caches(self):
        if self.dcache is None:
            n, kw = self.e.cfg.max_seq, self.kv * self.hd
            self.dcache = {}
            for i in range(self.layers):
                self.dcache["dk%d" % i] = np.zeros((n, kw), np.float32)
                self.dcache["dv%d" % i] = np.zeros((n, kw), np.float32)
        return self.dcache

    def build(self, key, gpu):
        from reclank.encoders.runtime import _EncCompiler
        from reclank.ir import program as P
        kind, R = key[0], key[1]
        H, kw, hdh = self.hidden, self.kv * self.hd, self.hd // 2
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=max(self.inter, len(self.taps) * H))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        zi = lambda *shape: np.zeros(shape, np.int32)  # noqa: E731
        c.env.update(self.caches())
        if kind == "inj":
            c.env.update(tin=z(R, len(self.taps) * H), ipos=zi(R), icos=z(R, hdh), isin=z(R, hdh), g0=z(R, H),
                         g=z(R, H), k=z(R, kw), v=z(R, kw))
            forms = [("set", "g0", ("enc_linear", self.fc, "tin")), ("set", "g", ("enc_rms", "g0", self.hn, H))]
            for i, d in enumerate(self.L):
                forms += [("set", "k", ("enc_linear", d["k"], "g")), ("set", "k", ("enc_rms", "k", d["kn"], self.hd)),
                          ("enc_rope_tab", "k", "icos", "isin", self.kv, self.hd, True),
                          ("set", "dk%d" % i, ("enc_scatter", "k", "ipos")),
                          ("set", "v", ("enc_linear", d["v"], "g")), ("set", "dv%d" % i, ("enc_scatter", "v", "ipos"))]
            c.compile(("seq", *forms))
            return self.e._team(c.p.finish(), gpu)
        span = key[2]
        K = R - 1
        qw = self.heads * self.hd
        c.lnorm_rows = True
        c.env.update(dtok=zi(R, 1), dpos=zi(R), dlo=zi(R), dhi=zi(R), dcos=z(R, hdh), dsin=z(R, hdh),
                     shift=np.array([[-1]] + [[j] for j in range(R - 1)], np.int32),
                     slots=np.arange(1, R, dtype=np.int32).reshape(K, 1), sc=self.sc,
                     e0=z(R, H), x=z(R, H), a=z(R, H), sh=z(R, H), c0=z(R, H), c1=z(R, H), b0=z(R, H), b1=z(R, H),
                     t=z(R, H), q=z(R, qw), k=z(R, kw), v=z(R, kw), o=z(R, qw), u=z(R, self.inter),
                     w=z(R, self.inter), hm=z(K, H), dlog=z(K, self.e.cfg.vocab), dgate=z(K, self.rank))

        forms = [("set", "e0", ("enc_gather", self.e.embed, "dtok", H)), ("set", "x", ("enc_mul_vec", "e0", "sc"))]
        c.compile(("seq", *forms))
        for i, d in enumerate(self.L):
            for conv_ in ("a", "f"):
                nrm = d["n1"] if conv_ == "a" else d["n2"]
                c.compile(("seq", ("set", "a", ("enc_rms", "x", nrm, H)),
                           ("set", "c0", ("enc_linear", d[conv_ + "00"], "a")),
                           ("set", "c1", ("enc_linear", d[conv_ + "01"], "a")),
                           ("set", "b0", ("enc_linear", d[conv_ + "10"], "a")),
                           ("set", "b1", ("enc_linear", d[conv_ + "11"], "a"))))
                self._conv(c, P, "a", "c0", "c1", "t")          # t: the conv of the normed input
                if conv_ == "a":
                    c.compile(("seq", ("set", "q", ("enc_linear", d["q"], "t")), ("set", "q", ("enc_rms", "q", d["qn"], self.hd)),
                               ("enc_rope_tab", "q", "dcos", "dsin", self.heads, self.hd, True),
                               ("set", "q", ("enc_mul_vec", "q", self.qscale)),
                               ("set", "k", ("enc_linear", d["k"], "t")), ("set", "k", ("enc_rms", "k", d["kn"], self.hd)),
                               ("enc_rope_tab", "k", "dcos", "dsin", self.kv, self.hd, True),
                               ("set", "dk%d" % i, ("enc_scatter", "k", "dpos")),
                               ("set", "v", ("enc_linear", d["v"], "t")), ("set", "dv%d" % i, ("enc_scatter", "v", "dpos")),
                               ("set", "o", ("enc_attn_kv", "q", "dk%d" % i, "dv%d" % i, "dlo", "dhi", self.heads,
                                             self.hd, span, self.kv, 0 if gpu else 1)),
                               ("set", "a", ("enc_linear", d["o"], "o"))))
                else:
                    c.compile(("seq", ("set", "u", ("enc_linear", d["gate"], "t")), ("set", "w", ("enc_linear", d["up"], "t")),
                               ("set", "u", ("enc_silu_mul", "u", "w")), ("set", "a", ("enc_linear", d["down"], "u"))))
                self._conv(c, P, "a", "b0", "b1", "t")
                c.compile(("enc_add", "x", "t"))
        c.compile(("seq", ("set", "a", ("enc_rms", "x", self.fn, H)), ("set", "hm", ("enc_gather", "a", "slots", H)),
                   ("set", "dlog", ("enc_linear", self.e.head, "hm")), ("set", "dgate", ("enc_linear", self.gate, "hm"))))
        return self.e._team(c.p.finish(), gpu)

    def _conv(self, c, P, src, p0, p1, out):
        """out = p0 src + p1 shift(src): the grouped conv of a block (the shift within it, row 0 zeros)."""
        env = c.env
        n = env[src].size
        c.compile(("set", "sh", ("enc_gather", src, "shift", self.hidden)))
        c.p.emit(P.MUL, env[p0], env[src], env[out], 1, n, 0)
        c.p.emit(P.MUL, env[p1], env["sh"], env["sh"], 1, n, 0)
        c.compile(("enc_add", out, "sh"))

    # ---- the host's side

    def inject_rows(self, taps, positions):
        """The context of the target's taps [n, 3 x 1024] at positions: programs of 512 rows (a prompt's) or of
        a block's rows (a round's), the rows past n at position -1 (ENC_SCATTER writes none of them), so that
        two programs serve every count."""
        positions = np.asarray(positions, np.int32)
        R = self.block if len(positions) <= self.block else 512
        for a in range(0, len(positions), R):
            p = np.full(R, -1, np.int32)
            n = min(R, len(positions) - a)
            p[:n] = positions[a:a + n]
            t = np.zeros((R, taps.shape[1]), np.float32)
            t[:n] = taps[a:a + n]
            cos, sin = rope_tables(np.maximum(p, 0), self.hd, self.theta)
            self.e._run(("inj", R), {"tin": t, "ipos": p, "icos": cos, "isin": sin}, [])

    def propose(self, last, pos0, n):
        """At most n drafts after token last at position pos0 (the target has run the positions before)."""
        B = self.block
        pos = np.arange(pos0, pos0 + B, dtype=np.int32)
        span = min(self.e.cfg.max_seq, -(-(pos0 + B) // 2048) * 2048)
        cos, sin = rope_tables(pos, self.hd, self.theta)
        lg, gate = self.e._run(("drf", B, span), {
            "dtok": np.array([[last]] + [[self.mask]] * (B - 1), np.int32), "dpos": pos,
            "dlo": np.zeros(B, np.int32), "dhi": np.full(B, pos0 + B, np.int32), "dcos": cos, "dsin": sin},
            ["dlog", "dgate"])
        cand = np.argpartition(-lg, self.top_k, axis=1)[:, :self.top_k]
        u = np.take_along_axis(lg, cand, 1)
        o = np.argsort(-u, axis=1, kind="stable")
        cand, u = np.take_along_axis(cand, o, 1), np.take_along_axis(u, o, 1)
        return self.walk(last, cand.astype(np.int32), u, gate, n=n)
