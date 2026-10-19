"""The LiLiCorr drafter of Nemotron 3.5 Bolt (DFlash block drafting with a LiLiCorr lattice head):
the NumPy reference (docs/NEMOTRON_BOLT_PLAN.md, P8). Bolt has no MTP layer; NVIDIA's drafter is the
separate checkpoint NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-LiLiCorr (1.1B, NVFP4 MLPs, bfloat16 attention).
The math follows llama.cpp's src/models/dflash.cpp and the DFlash driver of common/speculative.cpp.

- The context: for each position the target has run, the target's residual stream after blocks
  1, 18, 35, 51, 68 and 85 (config target_layer_ids: the outputs of those blocks), concatenated
  (6 x 4096), through fc (NVFP4) and hidden_norm: g. Each drafter layer keeps K = k_norm(k_proj(g))
  with RoPE at the position and V = v_proj(g) (a window of 1024 positions).
- A draft: the block [last token, mask x 7] (mask: token 990, the target's own embedding row) at the
  positions pos0 .. pos0 + 7, through 5 layers: attention over the context and the whole block (not
  causal), sliding window 1024, a sink logit a head; around the attention and the MLP a grouped
  dynamic conv over the block (2 taps, groups of 16 channels: a base kernel plus coefficients
  projected from the normed input, the same coefficients for the input side and the output side);
  SwiGLU MLP. The target's lm_head gives the logits of the 7 mask slots.
- The LiLiCorr head: the top 8 candidates of each slot (with features: log p, p, log p less the best,
  the rank, first), their token embeddings, the slot's hidden state and the anchor (g of position
  pos0 - 1) through 2 small transformer layers over the 56 candidates, then factors: the score of
  candidate b at slot s after candidate a at slot s - 1 is 8 <in(s, b), out(s - 1, a)> (slot 0: against
  the anchor). The draft walks the lattice greedily: the best successor of the chosen candidate.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.formats.safetensors import SafeTensors
from reclank.models.nemotron.model import nvfp4, rms_norm, silu

RING = 2048             # the positions of the context the layers keep (a window of 1024, and room)
LILI_HEAD = os.environ.get("RECLANK_LILI_HEAD", "program")     # "numpy": the NumPy head (LiLiCorr.lattice)


def _rope(x, pos, theta):
    """NEOX RoPE of x [n, heads, d] at the positions pos [n]."""
    d = x.shape[-1]
    inv = theta ** (-np.arange(0, d, 2, dtype=np.float64) / d)
    ang = np.asarray(pos, np.float64)[:, None] * inv[None, :]
    c, s = np.cos(ang).astype(np.float32)[:, None, :], np.sin(ang).astype(np.float32)[:, None, :]
    a, b = x[..., :d // 2], x[..., d // 2:]
    return np.concatenate([a * c - b * s, b * c + a * s], -1)


def _l2(x, eps):
    return x / np.maximum(np.sqrt((x * x).sum(-1, keepdims=True)), eps)


def _softmax(x, axis=-1):
    m = x.max(axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis, keepdims=True)


class LiLiCorrCache:
    """The drafter's context: K and V of each layer by position (a ring), and g of the last positions
    (the anchors)."""

    def __init__(self, d):
        self.K = [np.zeros((RING, d.kv, d.hd), np.float32) for _ in range(d.layers)]
        self.V = [np.zeros((RING, d.kv, d.hd), np.float32) for _ in range(d.layers)]
        self.pos = np.full(RING, -1, np.int64)
        self.g = {}


class LiLiCorr:
    """The drafter: inject(cache, feats, positions) after the target runs positions; draft(cache,
    last, pos0) the next tokens. embed(ids) and head(x) are the target's (bfloat16 embedding rows,
    lm_head)."""

    def __init__(self, path, embed, head):
        c = json.load(open(os.path.join(path, "config.json")))
        f = c["dflash_config"]
        self.embed, self.head = embed, head
        self.hidden, self.layers = c["hidden_size"], c["num_hidden_layers"]
        self.heads, self.kv, self.hd = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.eps, self.theta = float(c["rms_norm_eps"]), float(c["rope_theta"])
        self.block, self.mask = int(c["block_size"]), int(f["mask_token_id"])
        self.target_ids = list(f["target_layer_ids"])
        self.window = int(f["swa_window_size"])
        self.ck, self.cg = int(f["conv_kernel_size"]), int(f["conv_group_size"])
        self.top_k, self.lh = int(f["lilicorr_candidate_topk"]), int(f["lilicorr_hidden_size"])
        self.lheads, self.llayers = int(f["lilicorr_num_heads"]), int(f["lilicorr_num_layers"])
        self.veps, self.lscale = float(f["lilicorr_vector_eps"]), float(f["lilicorr_logit_scale"])
        self.st = SafeTensors(os.path.join(path, "model.safetensors"))
        self._w, self._f = {}, {}
        slots, k = self.block - 1, self.top_k
        rel = self.f32("lilicorr.relative_slot_bias")               # [heads, 2 block - 1]
        same = self.f32("lilicorr.same_slot_bias")                  # [heads]
        qs = np.arange(slots * k) // k
        self.lbias = rel[:, qs[:, None] - qs[None, :] + self.block - 1] + same[:, None, None] * (qs[:, None] == qs[None, :])

    # ---- the weights

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

    def lin(self, x, name, bias=False):
        y = x @ self.W(name).T
        return y + self.f32(name + ".bias") if bias else y

    def cache(self):
        return LiLiCorrCache(self)

    # ---- the context

    def features(self, rows):
        """g of target rows [n, 6 x hidden] (the hidden states after the target_ids blocks)."""
        return rms_norm(self.lin(rows, "fc"), self.f32("hidden_norm.weight"), self.eps)

    def inject(self, cache, rows, positions):
        g = self.features(np.asarray(rows, np.float32))
        positions = np.asarray(positions)
        for i in range(self.layers):
            p = "layers.%d.self_attn." % i
            K = rms_norm(self.lin(g, p + "k_proj").reshape(-1, self.kv, self.hd), self.f32(p + "k_norm.weight"), self.eps)
            K = _rope(K, positions, self.theta)
            V = self.lin(g, p + "v_proj").reshape(-1, self.kv, self.hd)
            cache.K[i][positions % RING] = K
            cache.V[i][positions % RING] = V
        cache.pos[positions % RING] = positions
        for j, q in enumerate(positions):
            cache.g[int(q)] = g[j]
        for q in [q for q in cache.g if q < positions.max() - self.block - 1]:
            del cache.g[q]
        return g

    # ---- a draft

    def _conv(self, h, dyn, base, side):
        """The grouped dynamic conv of the block h [n, hidden]: sum over the taps of (base[side, tap] +
        the token's coefficient of the group) x h shifted down by tap."""
        n = h.shape[0]
        out = np.zeros_like(h)
        for tap in range(min(self.ck, n)):
            w = base[side, tap][None, :] + np.repeat(dyn[:, side, tap, :], self.cg, axis=1)
            sh = np.zeros_like(h)
            sh[tap:] = h[:n - tap]
            out += w * sh
        return out

    def _attn(self, cache, i, q, k, v, positions, sinks):
        """Each query: the context keys of its window (positions before the block) and every key of the
        block; a sink logit a head in the softmax."""
        pos0 = positions[0]
        ctx = np.nonzero((cache.pos >= 0) & (cache.pos < pos0))[0]
        out = np.empty_like(q)
        rep = self.heads // self.kv
        for j, p in enumerate(positions):
            sel = ctx[p - cache.pos[ctx] < self.window]
            K = np.concatenate([cache.K[i][sel], k], 0)             # [m, kv, hd]
            V = np.concatenate([cache.V[i][sel], v], 0)
            for h in range(self.heads):
                s = (K[:, h // rep] @ q[j, h]) / np.sqrt(self.hd)
                m = max(s.max(), sinks[h])
                e = np.exp(s - m)
                out[j, h] = (e @ V[:, h // rep]) / (e.sum() + np.exp(sinks[h] - m))
        return out

    def block_forward(self, cache, last, pos0):
        """The drafter on [last, mask x 7] at pos0 ..: the normed hidden states [block, hidden]."""
        ids = [int(last)] + [self.mask] * (self.block - 1)
        positions = np.arange(pos0, pos0 + self.block)
        x = np.asarray(self.embed(ids), np.float32)
        ng = self.hidden // self.cg
        for i in range(self.layers):
            p = "layers.%d." % i
            a = rms_norm(x, self.f32(p + "input_layernorm.weight"), self.eps)
            dyn = self.lin(a, p + "attention_conv.kernel_projection").reshape(-1, 2, self.ck, ng)
            base = self.f32(p + "attention_conv.base_kernel")
            a = self._conv(a, dyn, base, 0)
            q = rms_norm(self.lin(a, p + "self_attn.q_proj").reshape(-1, self.heads, self.hd),
                         self.f32(p + "self_attn.q_norm.weight"), self.eps)
            k = rms_norm(self.lin(a, p + "self_attn.k_proj").reshape(-1, self.kv, self.hd),
                         self.f32(p + "self_attn.k_norm.weight"), self.eps)
            v = self.lin(a, p + "self_attn.v_proj").reshape(-1, self.kv, self.hd)
            q, k = _rope(q, positions, self.theta), _rope(k, positions, self.theta)
            o = self._attn(cache, i, q, k, v, positions, self.f32(p + "self_attn.attention_sink_bias"))
            o = self._conv(self.lin(o.reshape(len(ids), -1), p + "self_attn.o_proj"), dyn, base, 1)
            x = x + o
            m = rms_norm(x, self.f32(p + "post_attention_layernorm.weight"), self.eps)
            dyn = self.lin(m, p + "mlp_conv.kernel_projection").reshape(-1, 2, self.ck, ng)
            base = self.f32(p + "mlp_conv.base_kernel")
            m = self._conv(m, dyn, base, 0)
            m = self.lin(silu(self.lin(m, p + "mlp.gate_proj")) * self.lin(m, p + "mlp.up_proj"), p + "mlp.down_proj")
            x = x + self._conv(m, dyn, base, 1)
        return rms_norm(x, self.f32("norm.weight"), self.eps)

    def lattice(self, logits, hidden, anchor):
        """The LiLiCorr head on the slots (logits, hidden: [slots, ...]) and the anchor g: the candidates
        [slots, k] and the scores [slots, k (predecessor), k] (slot 0: predecessor 0 only)."""
        k, slots, H = self.top_k, logits.shape[0], self.lh
        part = np.argpartition(-logits, k, axis=-1)[:, :k]
        pv = np.take_along_axis(logits, part, -1)
        o = np.lexsort((part, -pv), axis=-1)                        # by logit, then by index
        cand = np.take_along_axis(part, o, -1)
        mx = logits.max(-1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(logits - mx).sum(-1))
        prob = np.exp(np.take_along_axis(logits, cand, -1) - lse[:, None]).astype(np.float32)
        lp = np.log(prob)
        rank = np.broadcast_to(np.arange(k, dtype=np.float32), lp.shape)
        feat = np.stack([lp, prob, lp - lp[:, :1], rank / max(k - 1, 1), (rank == 0).astype(np.float32)], -1)
        feat = feat.reshape(slots * k, 5)
        mu = feat.mean(-1, keepdims=True)
        fn = (feat - mu) / np.sqrt(((feat - mu) ** 2).mean(-1, keepdims=True) + 1e-5)
        fn = fn * self.f32("lilicorr.feature_norm.weight") + self.f32("lilicorr.feature_norm.bias")
        fs = self.lin(silu(self.lin(fn, "lilicorr.feature_mlp.up_proj", True)), "lilicorr.feature_mlp.down_proj", True)
        ts = self.lin(np.asarray(self.embed(cand.reshape(-1)), np.float32), "lilicorr.token_proj", True)
        ps = np.repeat(self.lin(hidden, "lilicorr.pass_hidden_proj", True), k, axis=0)
        pos = (self.f32("lilicorr.slot_embedding").reshape(-1, H)[:slots, None, :] +
               self.f32("lilicorr.rank_embedding").reshape(-1, H)[None, :k, :]).reshape(slots * k, H)
        x = ts + ps + fs + pos
        an = self.lin(anchor[None, :], "lilicorr.context_proj", True)
        hd = H // self.lheads
        for i in range(self.llayers):
            p = "lilicorr.layers.%d." % i
            a = rms_norm(x, self.f32(p + "attn_norm.weight"), self.eps)
            qkv = a @ self.f32(p + "attn.in_proj_weight").T + self.f32(p + "attn.in_proj_bias")
            q, kk, v = (qkv[:, j * H:(j + 1) * H].reshape(-1, self.lheads, hd).transpose(1, 0, 2) for j in range(3))
            att = _softmax(q @ kk.transpose(0, 2, 1) / np.sqrt(hd) + self.lbias)
            o = (att @ v).transpose(1, 0, 2).reshape(-1, H)
            x = x + self.lin(o, p + "attn.out_proj", True)
            m = rms_norm(x, self.f32(p + "mlp_norm.weight"), self.eps)
            x = x + self.lin(silu(self.lin(m, p + "mlp.up_proj", True)), p + "mlp.down_proj", True)
        x = rms_norm(x, self.f32("lilicorr.output_norm.weight"), self.eps)
        an = rms_norm(an, self.f32("lilicorr.anchor_norm.weight"), self.eps)
        anr = np.repeat(an, slots * k, axis=0)
        fac = silu(self.lin(np.concatenate([x, anr, x * anr], -1), "lilicorr.factor_input_proj", True))
        out = _l2(self.lin(fac, "lilicorr.out_head", True), self.veps).reshape(slots, k, -1)
        inn = _l2(self.lin(fac, "lilicorr.in_head", True), self.veps).reshape(slots, k, -1)
        anc = _l2(self.lin(an, "lilicorr.anchor_out_head", True), self.veps)[0]
        scores = np.zeros((slots, k, k), np.float32)
        scores[0, 0] = self.lscale * (inn[0] @ anc)
        for s in range(1, slots):
            scores[s] = self.lscale * (out[s - 1] @ inn[s].T)    # [predecessor, successor]
        return cand, scores

    def draft(self, cache, last, pos0, n=None):
        """Up to n (block - 1) tokens after last (at position pos0; the context holds the positions
        before it): the greedy walk of the lattice."""
        hid = self.block_forward(cache, last, pos0)
        logits = np.asarray(self.head(hid[1:]), np.float32)
        cand, scores = self.lattice(logits, hid[1:], cache.g[int(pos0) - 1])
        out, pred = [], 0
        for s in range(min(n or self.block - 1, self.block - 1)):
            pred = int(np.argmax(scores[s, pred]))
            out.append(int(cand[s, pred]))
        return out


class LiLiCorrKQ(LiLiCorr):
    """The drafter as programs (the records of the CPU or the GPU: NM_DPREP, NM_DATTN, NM_DCONV,
    BF16_LINEAR, the NVFP4 matrices in KQ_NV4 rows, ENC_SILU_MUL); the LiLiCorr head as programs too:
    NM_LTOPK on the logits of the slots (the candidates and their features), then, after the host
    reads the candidates' embedding rows, the lattice program (_lattice_prog: ENC_LINEAR of the
    bfloat16 matrices, the NVFP4 ones as the drafter's, NM_LATT, NM_LROWS, NM_LCAT, NM_LSCORE) to the
    scores; the greedy walk on the host (RECLANK_LILI_HEAD=numpy: the NumPy head). target: the
    NemotronKQ (its embedding rows and lm_head). gpu: emit the GPU's records (KQ_LINEAR of the KQ_NV4
    rows on float x) in place of NM_QX2 + NM_LIN2."""

    def __init__(self, path, target, gpu=False):
        super().__init__(path, embed=target.embed, head=target.logits)
        from reclank.backends.cpu import cops
        self.gpu = gpu
        self._m = {}
        self._cops = cops
        self.progs = {}
        self.ring_k = [np.zeros((RING, self.kv, self.hd), np.float32) for _ in range(self.layers)]
        self.ring_v = [np.zeros((RING, self.kv, self.hd), np.float32) for _ in range(self.layers)]
        self.g = {}

    def bf16(self, name):
        """A bfloat16 matrix as its bits (a view of the map)."""
        st = self.st
        o0, o1 = st.header[name]["data_offsets"]
        return np.frombuffer(st._mm, np.uint16, count=(o1 - o0) // 2, offset=st._base + o0).reshape(st.header[name]["shape"])

    def M(self, name):
        """(data, type, rows, cols): an NVFP4 matrix in KQ_NV4 rows, or a bfloat16 one (type None)."""
        m = self._m.get(name)
        if m is None:
            if name + ".weight_scale" in self.st.header:
                w = np.ascontiguousarray(self.st.get(name + ".weight", dtype=None))
                s = np.ascontiguousarray(self.st.get(name + ".weight_scale", dtype=None))
                g = float(np.asarray(self.f32(name + ".weight_scale_2")).reshape(-1)[0])
                rows, cols = w.shape[0], 2 * w.shape[1]
                out = np.zeros(rows * self._cops.kq_nv4_row_bytes(cols), np.uint8)
                self._cops.kq_nv4_pack([w], [s], [g], rows, cols, out)
                m = (out, 51, rows, cols)
            else:
                w = self.bf16(name + ".weight")
                m = (w, None, w.shape[0], w.shape[1])
            self._m[name] = m
        return m

    # ---- the programs

    def _lin(self, prog, name, src, out, t, scr):
        from reclank.ir import program as P
        w, typ, rows, cols = self.M(name)
        if typ is None:
            prog.emit(P.BF16_LINEAR, src, w, out, rows, cols, t)
        elif self.gpu and t > 1:
            # (the GPU's tiled int8 product on the two planes, as gpu.NmEmit)
            prog.emit(P.NM_QX, src, t, cols, scr["xq"], scr["xs"], scr["xl"])
            prog.emit(P.NM_GEMM8, scr["xq"], scr["xs"], w, typ, rows, cols, t, out, scr["xl"])
        elif self.gpu:
            prog.emit(P.KQ_LINEAR, scr["xq"], scr["xs"], scr["xm"], src, w, typ, rows, cols, t, out)
        else:
            prog.emit(P.NM_QX2, src, t, cols, scr["xq"], scr["xl"], scr["xs"], scr["xm"])
            if t >= 4:
                prog.emit(P.NM_LIN2, scr["xq"], scr["xl"], scr["xs"], w, typ, rows, cols, t, out, scr["x16"])
            else:
                prog.emit(P.NM_LIN2, scr["xq"], scr["xl"], scr["xs"], w, typ, rows, cols, t, out)
        prog.keep.append(w)

    def _scr(self, t):
        wide = max(len(self.target_ids) * self.hidden, 3 * self.hidden * 4)
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
        """The block of t = block tokens from pos (the embedding rows in x): the normed hidden states."""
        from reclank.ir import program as P
        t, H, nq, nk = self.block, self.hidden, self.heads * self.hd, self.kv * self.hd
        prog = P.Program()
        scr = self._scr(t)
        f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        x, a, a2, dyn = f32(t, H), f32(t, H), f32(t, H), f32(t, 2 * self.ck * H // self.cg)
        q, k, v, qo, o, o2 = f32(t, nq), f32(t, nk), f32(t, nk), f32(t, nq), f32(t, nq), f32(t, H)
        I = self.M("layers.0.mlp.gate_proj")[2]
        gt, up, mm = f32(t, I), f32(t, I), f32(t, I)
        ascr = f32(t * self.heads * (self.window + t))
        hid = f32(t, H)
        pos = prog.slot("pos")
        for i in range(self.layers):
            p = "layers.%d." % i
            prog.emit(P.RMS_NORM, x, self.f32(p + "input_layernorm.weight"), a, t, H, self.eps)
            self._lin(prog, p + "attention_conv.kernel_projection", a, dyn, t, scr)
            base = self.f32(p + "attention_conv.base_kernel")
            prog.emit(P.NM_DCONV, a, dyn, base, a2, t, H, self.ck, self.cg, 0)
            self._lin(prog, p + "self_attn.q_proj", a2, q, t, scr)
            self._lin(prog, p + "self_attn.k_proj", a2, k, t, scr)
            self._lin(prog, p + "self_attn.v_proj", a2, v, t, scr)
            prog.emit(P.NM_DPREP, q, k, v, self.f32(p + "self_attn.q_norm.weight"), self.f32(p + "self_attn.k_norm.weight"),
                      self.ring_k[i], self.ring_v[i], qo, t, self.heads, self.kv, self.hd, pos, RING, self.theta, self.eps)
            prog.emit(P.NM_DATTN, qo, self.ring_k[i], self.ring_v[i], self.f32(p + "self_attn.attention_sink_bias"), o,
                      t, self.heads, self.kv, self.hd, pos, RING, self.window, ascr)
            self._lin(prog, p + "self_attn.o_proj", o, o2, t, scr)
            prog.emit(P.NM_DCONV, o2, dyn, base, a2, t, H, self.ck, self.cg, 1)
            prog.emit(P.ADD, x, a2, x, t * H)
            prog.emit(P.RMS_NORM, x, self.f32(p + "post_attention_layernorm.weight"), a, t, H, self.eps)
            self._lin(prog, p + "mlp_conv.kernel_projection", a, dyn, t, scr)
            base2 = self.f32(p + "mlp_conv.base_kernel")
            prog.emit(P.NM_DCONV, a, dyn, base2, a2, t, H, self.ck, self.cg, 0)
            self._lin(prog, p + "mlp.gate_proj", a2, gt, t, scr)
            self._lin(prog, p + "mlp.up_proj", a2, up, t, scr)
            prog.emit(P.ENC_SILU_MUL, gt, up, mm, t * I)
            self._lin(prog, p + "mlp.down_proj", mm, o2, t, scr)
            prog.emit(P.NM_DCONV, o2, dyn, base2, a2, t, H, self.ck, self.cg, 1)
            prog.emit(P.ADD, x, a2, x, t * H)
        prog.emit(P.RMS_NORM, x, self.f32("norm.weight"), hid, t, H, self.eps)
        prog.names.update(x=x, hid=hid)
        prog.keep += [scr, ascr] + self.ring_k + self.ring_v
        return prog.finish()

    def _topk(self, prog, logits):
        """NM_LTOPK of the slots' logits into the head's cand and feat (one pair for every program)."""
        from reclank.ir import program as P
        slots, k = self.block - 1, self.top_k
        if not hasattr(self, "cand"):
            self.cand = np.zeros((slots, k), np.int32)
            self.feat = np.zeros((slots * k, 5), np.float32)
        prog.emit(P.NM_LTOPK, logits, slots, logits.shape[1], k, self.cand, self.feat)
        prog.names.update(cand=self.cand, feat=self.feat)
        prog.keep += [self.cand, self.feat]

    def _lattice_prog(self, hidden):
        """The head after NM_LTOPK: feat (its features), temb (the candidates' embedding rows), hidden
        (the slots' hidden states, slots x hidden) and anchor (g of the position before the block) to
        scores (slots x k x k: [s][a][b] the score of b at slot s after a at s - 1)."""
        from reclank.ir import program as P
        slots, k, H, hd = self.block - 1, self.top_k, self.lh, self.lh // self.lheads
        n = slots * k
        prog = P.Program()
        f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        inf = float("inf")
        scr = self._scr(n)
        feat = self.feat
        temb, anchor = f32(n, self.hidden), f32(1, self.hidden)
        fn, f1, fs, ts, ps, x = f32(n, 5), f32(n, H), f32(n, H), f32(n, H), f32(slots, H), f32(n, H)
        a, qkv, att, ob, u, c, an = f32(n, H), f32(n, 3 * H), f32(n, H), f32(n, H), f32(n, 2 * H), f32(1, H), f32(1, H)
        cat, fac, ov, iv, anc = f32(n, 3 * H), f32(n, H), f32(n, H), f32(n, H), f32(1, H)
        scores = f32(slots, k, k)
        L = "lilicorr."

        def enc(src, name, out, rows, m, kk, bf=True):
            w = self.bf16(L + name + ".weight") if bf else self.f32(L + name + ".weight")
            prog.emit(P.ENC_LINEAR, src, w, int(bf), self.f32(L + name + ".bias"), out, rows, m, kk, -inf, inf, -inf,
                      inf, None, None)
            prog.keep.append(w)
        if not hasattr(self, "_lconst"):
            pos = (self.f32(L + "slot_embedding").reshape(-1, H)[:slots, None, :] +
                   self.f32(L + "rank_embedding").reshape(-1, H)[None, :k, :]).reshape(n, H)
            bsum = self.f32(L + "token_proj.bias") + self.f32(L + "pass_hidden_proj.bias")
            self._lconst = (np.ascontiguousarray(pos, np.float32), np.ascontiguousarray(bsum, np.float32),
                            np.ascontiguousarray(self.lbias, np.float32))
        pos, bsum, lbias = self._lconst
        prog.emit(P.ENC_LNORM, feat, self.f32(L + "feature_norm.weight"), self.f32(L + "feature_norm.bias"), fn, n, 5, 1e-5)
        enc(fn, "feature_mlp.up_proj", f1, n, H, 5, bf=False)
        prog.emit(P.ENC_SILU, f1, f1, n * H)
        enc(f1, "feature_mlp.down_proj", fs, n, H, H)
        self._lin(prog, L + "token_proj", temb, ts, n, scr)
        self._lin(prog, L + "pass_hidden_proj", hidden, ps, slots, scr)
        prog.emit(P.NM_LROWS, ts, bsum, ps, x, n, H, k, 0)
        prog.emit(P.ENC_ADD, x, fs, n * H, 1.0)
        prog.emit(P.ENC_ADD, x, pos, n * H, 1.0)
        for i in range(self.llayers):
            p = L + "layers.%d." % i
            prog.emit(P.RMS_NORM, x, self.f32(p + "attn_norm.weight"), a, n, H, self.eps)
            w = self.bf16(p + "attn.in_proj_weight")
            prog.emit(P.ENC_LINEAR, a, w, 1, self.f32(p + "attn.in_proj_bias"), qkv, n, 3 * H, H, -inf, inf, -inf, inf,
                      None, None)
            prog.keep.append(w)
            prog.emit(P.NM_LATT, qkv, lbias, att, n, self.lheads, hd)
            self._lin(prog, p + "attn.out_proj", att, ob, n, scr)
            prog.emit(P.NM_LROWS, ob, self.f32(p + "attn.out_proj.bias"), x, x, n, H, 1, 0)
            prog.emit(P.RMS_NORM, x, self.f32(p + "mlp_norm.weight"), a, n, H, self.eps)
            self._lin(prog, p + "mlp.up_proj", a, u, n, scr)
            prog.emit(P.NM_LROWS, u, self.f32(p + "mlp.up_proj.bias"), None, u, n, 2 * H, 1, 1)
            self._lin(prog, p + "mlp.down_proj", u, ob, n, scr)
            prog.emit(P.NM_LROWS, ob, self.f32(p + "mlp.down_proj.bias"), x, x, n, H, 1, 0)
        prog.emit(P.RMS_NORM, x, self.f32(L + "output_norm.weight"), a, n, H, self.eps)
        self._lin(prog, L + "context_proj", anchor, c, 1, scr)
        prog.emit(P.NM_LROWS, c, self.f32(L + "context_proj.bias"), None, an, 1, H, 1, 0)
        prog.emit(P.RMS_NORM, an, self.f32(L + "anchor_norm.weight"), an, 1, H, self.eps)
        prog.emit(P.NM_LCAT, a, an, cat, n, H)
        enc(cat, "factor_input_proj", fac, n, H, 3 * H)
        prog.emit(P.ENC_SILU, fac, fac, n * H)
        enc(fac, "out_head", ov, n, H, H)
        enc(fac, "in_head", iv, n, H, H)
        enc(an, "anchor_out_head", anc, 1, H, H)
        prog.emit(P.NM_LSCORE, ov, iv, anc, scores, slots, k, H, self.lscale, self.veps)
        prog.names.update(temb=temb, anchor=anchor, scores=scores, feat=feat, hidden=hidden)
        prog.keep += [scr, pos, bsum, lbias, feat, temb, anchor, hidden]
        return prog.finish()

    def _walk(self, cand, scores, n):
        out, pred = [], 0
        for s in range(min(n or self.block - 1, self.block - 1)):
            pred = int(np.argmax(scores[s, pred]))
            out.append(int(cand[s, pred]))
        return out

    def _prog(self, key, make):
        p = self.progs.get(key)
        if p is None:
            p = self.progs[key] = make()
        return p

    def inject(self, cache, rows, positions):
        """The target rows of the positions (consecutive) into the context (cache: unused, the rings
        are the drafter's own)."""
        rows = np.asarray(rows, np.float32)
        positions = np.asarray(positions)
        c0 = 0
        while c0 < len(positions):
            n = len(positions) - c0
            t = next(s for s in (256, 64, 16, 8, 4, 2, 1) if s <= n)
            prog = self._prog(("inject", t), lambda t=t: self._inject_prog(t))
            prog.names["feat"][:] = rows[c0:c0 + t]
            prog.bind(pos=int(positions[c0]))
            prog.run()
            for j in range(t):
                self.g[int(positions[c0 + j])] = prog.names["g"][j].copy()
            c0 += t
        top = int(positions.max())
        for q in [q for q in self.g if q < top - self.block - 1]:
            del self.g[q]

    def block_forward(self, cache, last, pos0):
        prog = self._prog("draft", self._draft_prog)
        ids = [int(last)] + [self.mask] * (self.block - 1)
        prog.names["x"][:] = self.embed(ids)
        prog.bind(pos=int(pos0))
        prog.run()
        return prog.names["hid"].copy()

    def draft(self, cache, last, pos0, n=None):
        hid = self.block_forward(cache, last, pos0)
        logits = np.asarray(self.head(hid[1:]), np.float32)
        if LILI_HEAD == "numpy":
            cand, scores = self.lattice(logits, hid[1:], self.g[int(pos0) - 1])
            return self._walk(cand, scores, n)

        def topk():
            from reclank.ir import program as P
            prog = P.Program()
            lg = np.zeros_like(logits)
            self._topk(prog, lg)
            prog.names["lg"] = lg
            return prog.finish()
        tp = self._prog("topk", topk)
        tp.names["lg"][:] = logits
        tp.bind()
        tp.run()
        lp = self._prog("lattice", lambda: self._lattice_prog(np.zeros((self.block - 1, self.hidden), np.float32)))
        lp.names["temb"][:] = self.embed(self.cand.reshape(-1))
        lp.names["anchor"][0] = self.g[int(pos0) - 1]
        lp.names["hidden"][:] = hid[1:]
        lp.bind()
        lp.run()
        return self._walk(self.cand, lp.names["scores"], n)


class LiLiCorrGPU(LiLiCorrKQ):
    """The drafter's programs on the GPU (the GPU records; the rings on the device), beside the
    target's NemotronGPU (gpu: its Mirror, so the weights it shares stay one copy). The draft program
    ends with the target's lm_head on the 7 slots and NM_LTOPK; the host reads the candidates (56
    ints), uploads their embedding rows and the anchor, and the lattice program (on the draft's
    hidden states and features on the device) gives the scores."""

    def __init__(self, path, target, gdev):
        super().__init__(path, target, gpu=True)
        self.gdev = gdev
        self.logits = None

    def _prog(self, key, make):
        from reclank.backends.cuda.runtime import GPUProgram
        from reclank.ir import program as P
        e = self.progs.get(key)
        if e is None:
            prog = make()
            if key == "draft":
                head = self.gdev.m.M("language_model.lm_head")
                hid = prog.names["hid"]
                lg = np.zeros((self.block - 1, head.rows), np.float32)
                prog.recs.append((P.KQ_LINEAR, [prog._enc(a) for a in (
                    None, None, None, hid[1:], head.data, head.type, head.rows, head.cols, self.block - 1, lg)]))
                prog.names["logits"] = lg
                if LILI_HEAD != "numpy":
                    self._topk(prog, lg)
                prog.finish()
            e = self.progs[key] = (prog, GPUProgram(prog, graph=True, mirror=self.gdev.mirror))
        return e

    def inject(self, cache, rows, positions):
        rows = np.asarray(rows, np.float32)
        positions = np.asarray(positions)
        c0 = 0
        while c0 < len(positions):
            n = len(positions) - c0
            t = next(s for s in (256, 64, 16, 8, 4, 2, 1) if s <= n)
            prog, g = self._prog(("inject", t), lambda t=t: self._inject_prog(t))
            prog.names["feat"][:] = rows[c0:c0 + t]
            g.upload("feat")
            g.bind({"pos": int(positions[c0])})
            g.run()
            g.download("g")
            for j in range(t):
                self.g[int(positions[c0 + j])] = prog.names["g"][j].copy()
            c0 += t
        top = int(positions.max())
        for q in [q for q in self.g if q < top - self.block - 1]:
            del self.g[q]

    def block_forward(self, cache, last, pos0, head=False):
        prog, g = self._prog("draft", self._draft_prog)
        prog.names["x"][:] = self.embed([int(last)] + [self.mask] * (self.block - 1))
        g.upload("x")
        g.bind({"pos": int(pos0)})
        g.run()
        if head:
            g.download("cand")
            return None
        g.download("hid")
        g.download("logits")
        self.logits = prog.names["logits"]
        return prog.names["hid"].copy()

    def draft(self, cache, last, pos0, n=None):
        if LILI_HEAD == "numpy":
            hid = self.block_forward(cache, last, pos0)
            cand, scores = self.lattice(self.logits, hid[1:], self.g[int(pos0) - 1])
            return self._walk(cand, scores, n)
        self.block_forward(cache, last, pos0, head=True)
        hid = self.progs["draft"][0].names["hid"]
        prog, g = self._prog("lattice", lambda: self._lattice_prog(hid[1:]))
        prog.names["temb"][:] = self.embed(self.cand.reshape(-1))
        prog.names["anchor"][0] = self.g[int(pos0) - 1]
        g.upload("temb")
        g.upload("anchor")
        g.bind({})
        g.run()
        g.download("scores")
        return self._walk(self.cand, prog.names["scores"], n)
