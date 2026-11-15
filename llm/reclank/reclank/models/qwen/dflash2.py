"""The DFlash 2 drafter of Qwen3.8-27B (z-lab/Qwen3.8-27B-DFlash2-GGUF; Inco AI's DFlash2DraftModel): block
drafting with a candidate selector for the speculative decoding of the dense Qwen3.8-27B (QwenGPU, a GGUF or an
EXL3 directory; the session's loop, core/speculative.stream, as the MTP layer's).

The math follows llama.cpp's DFlash 2 (src/models/dflash.cpp and the DFlash driver of common/speculative.cpp of
the pull request "spec : add DFlash2 support"; MIT License; Copyright (c) 2023-2026 The ggml authors; see
THIRD_PARTY_NOTICES.md), as LiLiCorr's drafter of Bolt (models/nemotron/lilicorr.py) does, whose backbone it
shares (the grouped dynamic conv, the context's ring), without the attention sinks:

- The context: for each position the target has run, the target's residual stream entering layers 6, 20, 34,
  48 and 62 (the GGUF's dflash.target_layers; the outputs of layers 5, 19, 33, 47, 61), concatenated (5 x
  5120), through fc and the encoder's norm (enc.output_norm): g. Each drafter layer keeps K = rope(k_norm(
  k_proj(g))) and V = v_proj(g) of the position (a ring of RING positions; the window is 2048).
- A draft: the block [last token, mask x 7] (mask: token 248070) at the positions pos0 .. pos0 + 7, the
  target's token embedding rows, through 5 Qwen3 layers: around the attention and the MLP a grouped dynamic
  conv over the block (2 taps, groups of 16 channels: a base kernel plus the coefficients projected from the
  normed input, one projection for the input side and the output side); q and k norms, NEOX rope (theta 1e7),
  attention over the context's window and the whole block (not causal), SwiGLU MLP; the final norm (h) and the
  target's lm_head on the 7 mask slots.
- The selector: the 16 best tokens of each slot (their logits u), and for slot i after token a the score of
  candidate b: <next[b], prev[a] * gate_i> + u_i(b), gate_i = selector_hidden h_i (rank 256; prev and next are
  tables of a row a token). The draft walks it greedily from the last token: the best candidate after the
  chosen one. With stop, the drafts end before the first whose softmax over its 16 scores is under stop
  (llama.cpp's p_min).

DFlash2GPU runs it on the GPU beside the target's QwenGPU (its Mirror: the head is one copy): the context is
written inside the target's own programs (emit_tap: QwenGPU's programs call it at the start of each layer, so
every row the target runs, a prompt's, a step's, a verify group's, goes into the context with no copy to the
host; a verify group's rows after the kept ones are written again before a draft reads them, since a draft at
pos0 reads only the positions before pos0); the draft program (backbone, gate), the head and NM_LTOPK on the
GPU; the host reads 7 x 16 candidates, their log-probabilities and the 7 gates, and walks the selector.
"""
from __future__ import annotations

import os

import numpy as np

from reclank.formats.gguf import BF12, GGUF, Q8_0, _BLOCK
from reclank.models.nemotron.lilicorr import _rope
from reclank.models.nemotron.model import rms_norm, silu

RING = 4096             # the positions of the context the layers keep (the window of 2048, a block, and room)
TMAX = 256              # the most rows of a program of the target (gpu35 SPLIT_SIZES: the dense model's prompt groups)
# RECLANK_DFLASH_STOP: the selector's softmax under which the drafts of a round end (0: the 7 of the block)
STOP = float(os.environ.get("RECLANK_DFLASH_STOP", "0") or 0)
# RECLANK_DRAFT_HEAD8=0: the drafters' head on float32 x (else int8 x on a Q6R head: gpu35 head_rows x8)
HEAD8 = os.environ.get("RECLANK_DRAFT_HEAD8", "1") != "0"


def is_dflash2(path):
    """True when path is a DFlash 2 GGUF (dflash architecture with the selector)."""
    if not (isinstance(path, str) and path.endswith(".gguf") and os.path.exists(path)):
        return False
    g = GGUF(path, stage=False)
    return g.meta.get("general.architecture") == "dflash" and "selector_hidden.weight" in g.tensors


class DFlash2:
    """The NumPy reference (float32): inject(rows, positions) the target's taps of the positions [n, 5 x
    hidden]; draft(last, pos0, n) the next tokens. embed(ids): the target's embedding rows; head(h): the target's
    logits of rows."""

    def __init__(self, path, embed=None, head=None):
        self.g = GGUF(path, stage=False)
        m = self.g.meta
        k = lambda n: m["dflash." + n]  # noqa: E731
        self.embed, self.head = embed, head
        self.hidden, self.layers = int(k("embedding_length")), int(k("block_count"))
        self.heads, self.kv = int(k("attention.head_count")), int(k("attention.head_count_kv"))
        self.hd = int(k("attention.key_length"))
        self.inter = int(k("feed_forward_length"))
        self.eps, self.theta = float(k("attention.layer_norm_rms_epsilon")), float(k("rope.freq_base"))
        self.block, self.window = int(k("block_size")), int(k("attention.sliding_window"))
        self.ck, self.cg = int(k("conv_kernel_size")), int(k("conv_group_size"))
        self.rank, self.top_k = int(k("selector_rank")), int(k("selector_top_k"))
        self.mask = int(m["tokenizer.ggml.mask_token_id"])
        # the layer inputs of the target (llama.cpp's t_layer_inp: the outputs of the layers before)
        self.taps = [int(v) for v in np.asarray(k("target_layers")).reshape(-1)]
        assert int(m.get("dflash.attention.causal", 0)) == 0, "a causal DFlash block is not supported"
        sec = m.get("dflash.rope.dimension_sections")
        assert sec is None or list(np.asarray(sec).reshape(-1)[1:]) == [0] * (len(np.asarray(sec).reshape(-1)) - 1), \
            "M-RoPE sections other than the text's"
        self.stop = STOP
        self._w, self._f = {}, {}
        self._sel = None

    # ---- the weights

    def f32(self, name):
        v = self._f.get(name)
        if v is None:
            v = self._f[name] = np.ascontiguousarray(self.g.dequant(name), np.float32)
        return v

    def W(self, name):
        """A matrix [rows, cols] (float32: the reference's)."""
        return self.f32(name + ".weight")

    def lin(self, x, name):
        return x @ self.W(name).T

    def conv_base(self, i, kind):
        """The base kernel [2 (sides), kern, hidden] of layer i (kind attn or ffn)."""
        return self.f32("blk.%d.%s_conv_base" % (i, kind)).reshape(2, self.ck, self.hidden)

    def selector(self):
        """The tables of the selector (float32 [vocab, rank]): prev (the predecessor's) and next."""
        if self._sel is None:
            self._sel = (self.f32("selector_predecessor.weight"), self.f32("selector_successor.weight"))
        return self._sel

    # ---- the reference

    def cache(self):
        return dict(K=[np.zeros((RING, self.kv, self.hd), np.float32) for _ in range(self.layers)],
                    V=[np.zeros((RING, self.kv, self.hd), np.float32) for _ in range(self.layers)],
                    pos=np.full(RING, -1, np.int64))

    def inject(self, cache, rows, positions):
        g = rms_norm(self.lin(np.asarray(rows, np.float32), "fc"), self.f32("enc.output_norm.weight"), self.eps)
        positions = np.asarray(positions)
        for i in range(self.layers):
            b = "blk.%d." % i
            K = rms_norm(self.lin(g, b + "attn_k").reshape(-1, self.kv, self.hd), self.f32(b + "attn_k_norm.weight"),
                         self.eps)
            cache["K"][i][positions % RING] = _rope(K, positions, self.theta)
            cache["V"][i][positions % RING] = self.lin(g, b + "attn_v").reshape(-1, self.kv, self.hd)
        cache["pos"][positions % RING] = positions
        return g

    def _conv(self, h, dyn, base, side):
        n = h.shape[0]
        out = np.zeros_like(h)
        for tap in range(min(self.ck, n)):
            w = base[side, tap][None, :] + np.repeat(dyn[:, side, tap, :], self.cg, axis=1)
            sh = np.zeros_like(h)
            sh[tap:] = h[:n - tap]
            out += w * sh
        return out

    def _attn(self, cache, i, q, k, v, positions):
        pos0 = positions[0]
        ctx = np.nonzero((cache["pos"] >= 0) & (cache["pos"] < pos0))[0]
        out = np.empty_like(q)
        rep = self.heads // self.kv
        for j, p in enumerate(positions):
            sel = ctx[p - cache["pos"][ctx] < self.window]
            K = np.concatenate([cache["K"][i][sel], k], 0)
            V = np.concatenate([cache["V"][i][sel], v], 0)
            for h in range(self.heads):
                s = (K[:, h // rep] @ q[j, h]) / np.sqrt(self.hd)
                e = np.exp(s - s.max())
                out[j, h] = (e @ V[:, h // rep]) / e.sum()
        return out

    def block_forward(self, cache, last, pos0):
        """The normed hidden states [block, hidden] of [last, mask x 7] at pos0 .."""
        ids = [int(last)] + [self.mask] * (self.block - 1)
        positions = np.arange(pos0, pos0 + self.block)
        x = np.asarray(self.embed(ids), np.float32)
        ng = self.hidden // self.cg
        for i in range(self.layers):
            b = "blk.%d." % i
            a = rms_norm(x, self.f32(b + "attn_norm.weight"), self.eps)
            dyn = self.lin(a, b + "attn_conv_proj").reshape(-1, 2, self.ck, ng)
            base = self.conv_base(i, "attn")
            a = self._conv(a, dyn, base, 0)
            q = rms_norm(self.lin(a, b + "attn_q").reshape(-1, self.heads, self.hd), self.f32(b + "attn_q_norm.weight"),
                         self.eps)
            k = rms_norm(self.lin(a, b + "attn_k").reshape(-1, self.kv, self.hd), self.f32(b + "attn_k_norm.weight"),
                         self.eps)
            v = self.lin(a, b + "attn_v").reshape(-1, self.kv, self.hd)
            q, k = _rope(q, positions, self.theta), _rope(k, positions, self.theta)
            o = self._attn(cache, i, q, k, v, positions)
            x = x + self._conv(self.lin(o.reshape(len(ids), -1), b + "attn_output"), dyn, base, 1)
            m = rms_norm(x, self.f32(b + "ffn_norm.weight"), self.eps)
            dyn = self.lin(m, b + "ffn_conv_proj").reshape(-1, 2, self.ck, ng)
            base = self.conv_base(i, "ffn")
            m = self._conv(m, dyn, base, 0)
            m = self.lin(silu(self.lin(m, b + "ffn_gate")) * self.lin(m, b + "ffn_up"), b + "ffn_down")
            x = x + self._conv(m, dyn, base, 1)
        return rms_norm(x, self.f32("output_norm.weight"), self.eps)

    def candidates(self, hid):
        """The 16 best tokens of each slot (the rows 1 .. of hid), their log-probabilities, and the gates."""
        lg = np.asarray(self.head(hid[1:]), np.float64)
        cand = np.argsort(-lg, axis=1, kind="stable")[:, :self.top_k]
        m = lg.max(1, keepdims=True)
        lse = m + np.log(np.exp(lg - m).sum(1, keepdims=True))
        lp = np.take_along_axis(lg, cand, 1) - lse
        gate = self.lin(hid[1:], "selector_hidden")
        return cand.astype(np.int32), lp.astype(np.float32), gate.astype(np.float32)

    def walk(self, last, cand, lp, gate, n=None, eos_ids=()):
        """The drafts: the selector's greedy path from last (llama.cpp's walk; stop: its p_min)."""
        prev_t, next_t = self.selector()
        out, a = [], int(last)
        for i in range(min(n or self.block - 1, self.block - 1)):
            s = next_t[cand[i]] @ (prev_t[a] * gate[i]) + lp[i]
            b = int(np.argmax(s))
            if self.stop > 0 and 1.0 / np.exp(s - s[b]).sum() < self.stop:
                break
            a = int(cand[i, b])
            out.append(a)
            if a in eos_ids:
                break
        return out

    def draft(self, cache, last, pos0, n=None, eos_ids=()):
        hid = self.block_forward(cache, last, pos0)
        return self.walk(last, *self.candidates(hid), n=n, eos_ids=eos_ids)


class DFlash2GPU(DFlash2):
    """The drafter on the GPU beside the target's QwenGPU (QwenGPU(..., dflash=this) calls emit_tap from its
    programs; bind(dev) after). The matrices go to the GPU as the GGUF's rows (KQ_LINEAR; Q8_0 as Q8_R), fc as
    5 matrices of its column blocks (one a tap: the target's programs add each tap's product as they reach its
    layer)."""

    def __init__(self, path, model):
        super().__init__(path, embed=model.embed)
        assert self.hidden == model.cfg.hidden_size, "the drafter's hidden size is not the target's"
        self.model = model
        self.dev = None
        H, nk = self.hidden, self.kv * self.hd
        f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        self.ring_k = [f32(RING, self.kv, self.hd) for _ in range(self.layers)]
        self.ring_v = [f32(RING, self.kv, self.hd) for _ in range(self.layers)]
        # the rows of the target's programs (one set: they run one at a time)
        self.acc, self.tmp = f32(TMAX, H), f32(TMAX, H)
        self.kb, self.vb = f32(TMAX, nk), f32(TMAX, nk)
        self.nosink = np.full(self.heads, -1e30, np.float32)
        self._m = {}
        self._fc = None
        self.progs = {}
        self.capture = None         # [TMAX, hidden] a tap: the target's taps of each run (a check's; before QwenGPU)
        self.cand = np.zeros((self.block - 1, self.top_k), np.int32)
        self.feat = np.zeros(((self.block - 1) * self.top_k, 5), np.float32)

    # ---- the matrices for KQ_LINEAR

    def M(self, name):
        """(blocks, type, rows, cols) of a matrix (a GGUF name without .weight) for KQ_LINEAR."""
        e = self._m.get(name)
        if e is None:
            raw, dims, t = self.g.raw(name + ".weight")
            cols, rows = int(dims[0]), int(dims[1])
            data = np.ascontiguousarray(raw).view(np.uint8).reshape(-1)
            if t == Q8_0:
                b = data.reshape(rows, cols // 32, 34)
                pad = -(cols // 32 * 34) % 16
                data = np.ascontiguousarray(np.concatenate(
                    [b[:, :, 2:].reshape(rows, cols), b[:, :, :2].reshape(rows, cols // 16),
                     np.zeros((rows, pad), np.uint8)], axis=1)).reshape(-1)
                t = 100     # Q8_R (gpu35 dense)
            e = self._m[name] = (data, t, rows, cols)
        return e

    def fc_parts(self):
        """fc [hidden, 5 hidden] as 5 matrices [hidden, hidden] of its column blocks (KQ_LINEAR rows)."""
        if self._fc is None:
            raw, dims, t = self.g.raw("fc.weight")
            cols, rows = int(dims[0]), int(dims[1])
            n, H = len(self.taps), self.hidden
            assert cols == n * H, "fc reads %d columns, not %d taps of %d" % (cols, n, H)
            parts = []
            if t == BF12:
                from reclank.backends.cpu import cops
                # (BF12 holds bfloat16 values: a block of columns encodes again to the same values)
                full = self.g.dequant("fc.weight")
                bits = (full.view(np.uint32) >> 16).astype(np.uint16)
                for k in range(n):
                    parts.append(np.ascontiguousarray(cops.kq_bf16_to_bf12(
                        np.ascontiguousarray(bits[:, k * H:(k + 1) * H]), H)[0]).reshape(-1))
            else:
                assert t in _BLOCK and t not in (Q8_0,), "fc of type %d" % t
                bv, bb = _BLOCK[t]
                b = np.ascontiguousarray(raw).view(np.uint8).reshape(rows, cols // bv, bb)
                for k in range(n):
                    parts.append(np.ascontiguousarray(b[:, k * H // bv:(k + 1) * H // bv]).reshape(-1))
            self._fc = [(p, t, rows, H) for p in parts]
        return self._fc

    def _lin(self, prog, m, x, out, t):
        from reclank.ir import program as P
        data, typ, rows, cols = m
        prog.emit(P.KQ_LINEAR, None, None, None, x, data, typ, rows, cols, t, out)

    # ---- the context, inside the target's programs

    def emit_tap(self, prog, i, t):
        """At the start of the target's layer i of a program of t rows (QwenGPU's _Emit.layer_begin): the
        product of the residual stream x with fc's block of this tap, added to the others; after the last tap
        the encoder's norm and each layer's k and v into its ring at pos (its first nreal rows)."""
        if i not in self.taps:
            return
        from reclank.ir import program as P
        assert t <= TMAX, "a program of the target of %d rows (the drafter's rows hold %d)" % (t, TMAX)
        k = self.taps.index(i)
        H = self.hidden
        x = prog.names["x"]
        if self.capture is not None:
            # (a check's: the tap's rows of the last run, scripts/check_dflash2.py)
            prog.emit(P.COPY, x, self.capture[k][:t], t * H * 4)
            prog.keep.append(self.capture[k])
        acc, tmp = self.acc[:t], self.tmp[:t]
        self._lin(prog, self.fc_parts()[k], x, acc if k == 0 else tmp, t)
        if k > 0:
            prog.emit(P.ADD, acc, tmp, acc, t * H)
        prog.keep += [self.acc, self.tmp]
        if k < len(self.taps) - 1:
            return
        prog.emit(P.RMS_NORM, acc, self.f32("enc.output_norm.weight"), acc, t, H, self.eps)
        pos, nreal = prog.slot("pos"), prog.slot("nreal")
        kb, vb = self.kb[:t], self.vb[:t]
        for li in range(self.layers):
            b = "blk.%d." % li
            self._lin(prog, self.M(b + "attn_k"), acc, kb, t)
            self._lin(prog, self.M(b + "attn_v"), acc, vb, t)
            prog.emit(P.NM_DPREP, None, kb, vb, None, self.f32(b + "attn_k_norm.weight"), self.ring_k[li],
                      self.ring_v[li], None, t, self.heads, self.kv, self.hd, pos, RING, self.theta, self.eps, nreal)
        prog.keep += [self.kb, self.vb] + self.ring_k + self.ring_v

    # ---- a draft

    def bind(self, dev):
        """The target's QwenGPU (its Mirror, its head)."""
        self.dev = dev

    def _draft_prog(self):
        """The block of t = block tokens from pos (their embedding rows in x): the normed hidden states (hid)
        and the selector's gates of the 7 slots."""
        from reclank.ir import program as P
        t, H, nq, nk = self.block, self.hidden, self.heads * self.hd, self.kv * self.hd
        prog = P.Program()
        f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        x, a, a2, dyn = f32(t, H), f32(t, H), f32(t, H), f32(t, 2 * self.ck * H // self.cg)
        q, k, v, qo, o, o2 = f32(t, nq), f32(t, nk), f32(t, nk), f32(t, nq), f32(t, nq), f32(t, H)
        I = self.inter
        gt, up, mm = f32(t, I), f32(t, I), f32(t, I)
        ascr = f32(t * self.heads * (self.window + t))
        hid, gate = f32(t, H), f32(t - 1, self.rank)
        pos = prog.slot("pos")
        for i in range(self.layers):
            b = "blk.%d." % i
            prog.emit(P.RMS_NORM, x, self.f32(b + "attn_norm.weight"), a, t, H, self.eps)
            self._lin(prog, self.M(b + "attn_conv_proj"), a, dyn, t)
            base = np.ascontiguousarray(self.conv_base(i, "attn"))
            prog.emit(P.NM_DCONV, a, dyn, base, a2, t, H, self.ck, self.cg, 0)
            self._lin(prog, self.M(b + "attn_q"), a2, q, t)
            self._lin(prog, self.M(b + "attn_k"), a2, k, t)
            self._lin(prog, self.M(b + "attn_v"), a2, v, t)
            prog.emit(P.NM_DPREP, q, k, v, self.f32(b + "attn_q_norm.weight"), self.f32(b + "attn_k_norm.weight"),
                      self.ring_k[i], self.ring_v[i], qo, t, self.heads, self.kv, self.hd, pos, RING, self.theta,
                      self.eps)
            prog.emit(P.NM_DATTN, qo, self.ring_k[i], self.ring_v[i], self.nosink, o,
                      t, self.heads, self.kv, self.hd, pos, RING, self.window, ascr, 1)
            self._lin(prog, self.M(b + "attn_output"), o, o2, t)
            prog.emit(P.NM_DCONV, o2, dyn, base, a2, t, H, self.ck, self.cg, 1)
            prog.emit(P.ADD, x, a2, x, t * H)
            prog.emit(P.RMS_NORM, x, self.f32(b + "ffn_norm.weight"), a, t, H, self.eps)
            self._lin(prog, self.M(b + "ffn_conv_proj"), a, dyn, t)
            base2 = np.ascontiguousarray(self.conv_base(i, "ffn"))
            prog.emit(P.NM_DCONV, a, dyn, base2, a2, t, H, self.ck, self.cg, 0)
            self._lin(prog, self.M(b + "ffn_gate"), a2, gt, t)
            self._lin(prog, self.M(b + "ffn_up"), a2, up, t)
            prog.emit(P.ENC_SILU_MUL, gt, up, mm, t * I)
            self._lin(prog, self.M(b + "ffn_down"), mm, o2, t)
            prog.emit(P.NM_DCONV, o2, dyn, base2, a2, t, H, self.ck, self.cg, 1)
            prog.emit(P.ADD, x, a2, x, t * H)
            prog.keep += [base, base2]
        prog.emit(P.RMS_NORM, x, self.f32("output_norm.weight"), hid, t, H, self.eps)
        self._lin(prog, self.M("selector_hidden"), hid[1:], gate, t - 1)
        prog.names.update(x=x, hid=hid, gate=gate)
        prog.keep += [ascr, self.nosink] + self.ring_k + self.ring_v
        return prog.finish()

    def _topk_prog(self, vocab):
        from reclank.ir import program as P
        prog = P.Program()
        lg = np.zeros((self.block - 1, vocab), np.float32)
        prog.emit(P.NM_LTOPK, lg, self.block - 1, vocab, self.top_k, self.cand, self.feat)
        prog.names.update(lg=lg, cand=self.cand, feat=self.feat)
        prog.keep += [lg, self.cand, self.feat]
        return prog.finish()

    def _prog(self, key, make):
        from reclank.backends.cuda.runtime import GPUProgram
        e = self.progs.get(key)
        if e is None:
            prog = make()
            e = self.progs[key] = (prog, GPUProgram(prog, graph=True, mirror=self.dev.g.mirror))
        return e

    def _head_topk(self, src):
        """The target's head on the 7 rows at the device address src, then NM_LTOPK (cand, feat on the device)."""
        from reclank.backends.cuda.runtime import _check, lib
        dev = self.dev
        n, H = self.block - 1, self.hidden
        V = dev.model.M("lm_head").rows
        if dev._q6k_head():
            lg = dev.head_rows(src, n, x8=HEAD8)
        else:
            hp, hg = dev._kq_head(n)
            _check(lib().rc_gpu_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr, src, n * H * 4))
            hg.run()
            lg = hg.mirror.buffer_of(hp.names["out"]).ptr
        tp, tg = self._prog("topk", lambda: self._topk_prog(V))
        _check(lib().rc_gpu_d2d(tg.mirror.buffer_of(tp.names["lg"]).ptr, lg, n * V * 4))
        tg.bind({})
        tg.run()
        tg.download("cand")
        tg.download("feat")

    def candidates_gpu(self, last, pos0):
        """The draft program at pos0 (the context holds the positions before it): cand, lp, gate (host)."""
        prog, g = self._prog("draft", self._draft_prog)
        prog.names["x"][:] = self.embed([int(last)] + [self.mask] * (self.block - 1))
        g.upload("x")
        g.bind({"pos": int(pos0)})
        g.run()
        g.download("gate")
        self._head_topk(g.mirror.buffer_of(prog.names["hid"]).ptr + self.hidden * 4)
        lp = self.feat[:, 0].reshape(self.block - 1, self.top_k)
        return self.cand.copy(), lp.copy(), prog.names["gate"].copy()

    def hidden_gpu(self):
        """The draft program's hid of the last draft (host; a check's)."""
        prog, g = self.progs["draft"]
        g.download("hid")
        return prog.names["hid"].copy()

    def draft(self, cache, last, pos0, n=None, eos_ids=()):
        return self.walk(last, *self.candidates_gpu(last, pos0), n=n, eos_ids=eos_ids)

    def release(self):
        for _p, g in self.progs.values():
            g.close()
        self.progs = {}


class DFlash2Drafter:
    """DFlash2GPU as a Drafter of speculative.stream: the context is written by the target's runs (observe and
    flush have nothing to do); a round's drafts are the selector's path after the token at pos."""

    def __init__(self, d):
        self.d = d

    def draft(self, token, h, pos, n, eos_ids):
        if n <= 0:
            return []
        return self.d.draft(None, token, pos, n, eos_ids)

    def observe(self, tokens, hidden, pos):
        pass

    def flush(self):
        pass
