"""The DSpark drafter of Qwen3.8-27B (RedHatAI/Qwen3.8-27B-speculator.dspark: speculators' DSparkDraftModel, as a
GGUF of scripts/convert_dspark_gguf.py, BF12 or Q4_K): block drafting with a Markov head and a confidence head for
the speculative decoding of the dense Qwen3.8-27B (QwenGPU, a GGUF or an EXL3 directory; the session's loop,
core/speculative.stream, as DFlash 2's and the MTP layer's).

Written from speculators (vllm-project/speculators, src/speculators/models/dspark; Apache License 2.0; by Red
Hat), as GLM-5.3-Flash's drafter (models/glm53/dspark.py, scripts/glm53_dspark_reference.py); see
THIRD_PARTY_NOTICES.md:

- The context: for each position the target has run, the target's residual stream entering layers 4, 12, 20,
  28, 36, 44, 52 and 60 (the config's aux_hidden_state_layer_ids: vLLM's auxiliary hidden states, the input of
  each layer), concatenated (8 x 5120), through fc and hidden_norm: g. Each drafter layer keeps K = rope(
  k_norm(k_proj(g))) and V = v_proj(g) of the position (a ring of RING positions; the window is 2048).
- A draft: the block [last token, mask x 7] (mask: token 248077) at the positions pos0 .. pos0 + 7, the target's
  token embedding rows, through 5 Qwen3 layers (20 query heads over 4 key heads of 256, q and k norms, NEOX rope
  of theta 1e7, attention over the context's window and the whole block, not causal; SwiGLU MLP); the final norm
  (h) and the target's lm_head on all 8 rows (sample_from_anchor: row i drafts the token after slot i).
- The Markov head, in order: logits[i] += markov_w2 @ markov_w1[out[i]], out[i + 1] = argmax (out[0] the last
  token): 8 drafts. The confidence of draft i: sigmoid(proj(cat(h_i, markov_w1[out[i]])) + bias), the chance it
  is kept when the ones before it are. A round keeps the drafts while the product of their confidences is at
  least stop.

DSparkGPU runs it on the GPU beside the target's QwenGPU as DFlash2GPU does (its context written inside the
target's programs, emit_tap; the draft program, the target's head on the 8 rows, int8 x as a drafter's, and the
Markov program on the GPU); the host reads the 8 drafts and their confidences.
"""
from __future__ import annotations

import os

import numpy as np

from reclank.formats.gguf import BF12, BF16, GGUF, Q8_0, _BLOCK
from reclank.models.nemotron.lilicorr import _rope
from reclank.models.nemotron.model import rms_norm, silu
from reclank.models.qwen.dflash2 import HEAD8, RING, TMAX

# RECLANK_DSPARK_STOP: the product of the confidences under which the drafts of a round end
STOP = float(os.environ.get("RECLANK_DSPARK_STOP", "0.3") or 0)
KQ_BF16 = 30


def is_dspark(path):
    """True when path is a DSpark GGUF (scripts/convert_dspark_gguf.py)."""
    if not (isinstance(path, str) and path.endswith(".gguf") and os.path.exists(path)):
        return False
    return GGUF(path, stage=False).meta.get("general.architecture") == "dspark"


class DSpark:
    """The NumPy reference (float32): inject(cache, rows, positions) the target's taps of the positions [n, 8 x
    hidden]; draft(cache, last, pos0, n) the next tokens. embed(ids): the target's embedding rows; head(h): the
    target's logits of rows."""

    def __init__(self, path, embed=None, head=None):
        self.g = GGUF(path, stage=False)
        m = self.g.meta
        k = lambda n: m["dspark." + n]  # noqa: E731
        self.embed, self.head = embed, head
        self.hidden, self.layers = int(k("embedding_length")), int(k("block_count"))
        self.heads, self.kv = int(k("attention.head_count")), int(k("attention.head_count_kv"))
        self.hd, self.inter = int(k("attention.key_length")), int(k("feed_forward_length"))
        self.eps, self.theta = float(k("attention.layer_norm_rms_epsilon")), float(k("rope.freq_base"))
        self.block, self.window = int(k("block_size")), int(k("attention.sliding_window"))
        self.mask, self.rank, self.vocab = int(k("mask_token_id")), int(k("markov_rank")), int(k("vocab_size"))
        self.taps = [int(v) for v in np.asarray(k("target_layers")).reshape(-1)]
        self.drafts = self.block           # (the anchor's row drafts too)
        self.stop = STOP
        self._f = {}

    # ---- the weights

    def f32(self, name):
        v = self._f.get(name)
        if v is None:
            v = self._f[name] = np.ascontiguousarray(self.g.dequant(name), np.float32)
        return v

    def lin(self, x, name):
        return x @ self.f32(name).T

    # ---- the reference

    def cache(self):
        return dict(K=[np.zeros((RING, self.kv, self.hd), np.float32) for _ in range(self.layers)],
                    V=[np.zeros((RING, self.kv, self.hd), np.float32) for _ in range(self.layers)],
                    pos=np.full(RING, -1, np.int64))

    def inject(self, cache, rows, positions):
        g = rms_norm(self.lin(np.asarray(rows, np.float32), "fc.weight"), self.f32("hidden_norm.weight"), self.eps)
        positions = np.asarray(positions)
        for i in range(self.layers):
            a = "layers.%d.self_attn." % i
            K = rms_norm(self.lin(g, a + "k_proj.weight").reshape(-1, self.kv, self.hd), self.f32(a + "k_norm.weight"),
                         self.eps)
            cache["K"][i][positions % RING] = _rope(K, positions, self.theta)
            cache["V"][i][positions % RING] = self.lin(g, a + "v_proj.weight").reshape(-1, self.kv, self.hd)
        cache["pos"][positions % RING] = positions
        return g

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
        for i in range(self.layers):
            b = "layers.%d." % i
            a = rms_norm(x, self.f32(b + "input_layernorm.weight"), self.eps)
            q = rms_norm(self.lin(a, b + "self_attn.q_proj.weight").reshape(-1, self.heads, self.hd),
                         self.f32(b + "self_attn.q_norm.weight"), self.eps)
            k = rms_norm(self.lin(a, b + "self_attn.k_proj.weight").reshape(-1, self.kv, self.hd),
                         self.f32(b + "self_attn.k_norm.weight"), self.eps)
            v = self.lin(a, b + "self_attn.v_proj.weight").reshape(-1, self.kv, self.hd)
            q, k = _rope(q, positions, self.theta), _rope(k, positions, self.theta)
            o = self._attn(cache, i, q, k, v, positions)
            x = x + self.lin(o.reshape(len(ids), -1), b + "self_attn.o_proj.weight")
            m = rms_norm(x, self.f32(b + "post_attention_layernorm.weight"), self.eps)
            m = self.lin(silu(self.lin(m, b + "mlp.gate_proj.weight")) * self.lin(m, b + "mlp.up_proj.weight"),
                         b + "mlp.down_proj.weight")
            x = x + m
        return rms_norm(x, self.f32("norm.weight"), self.eps)

    def markov(self, hid, last):
        """The 8 drafts after last and the logit of each one's confidence, from the hidden states hid [8, hidden]."""
        lg = np.asarray(self.head(hid), np.float64)
        w1, w2 = self.f32("markov_head.markov_w1.weight"), self.f32("markov_head.markov_w2.weight")
        out, me = [int(last)], []
        for i in range(self.block):
            e = w1[out[i]]
            me.append(e)
            out.append(int(np.argmax(lg[i] + w2 @ e)))
        cw = self.f32("confidence_head.proj.weight").reshape(-1)
        conf = np.concatenate([hid, np.stack(me)], 1) @ cw + float(self.f32("confidence_head.proj.bias").reshape(-1)[0])
        return out[1:], conf.astype(np.float32)

    def cut(self, drafts, conf, n=None, eos_ids=()):
        """The drafts a round keeps: while the product of their confidences is at least stop, at most n, through
        the first end token."""
        surv = np.cumprod(1.0 / (1.0 + np.exp(-np.asarray(conf, np.float64))))
        out = []
        for i, d in enumerate(drafts[:min(n or self.drafts, self.drafts)]):
            if self.stop > 0 and surv[i] < self.stop:
                break
            out.append(int(d))
            if d in eos_ids:
                break
        return out

    def draft(self, cache, last, pos0, n=None, eos_ids=()):
        return self.cut(*self.markov(self.block_forward(cache, last, pos0), last), n=n, eos_ids=eos_ids)


class DSparkGPU(DSpark):
    """The drafter on the GPU beside the target's QwenGPU (QwenGPU(..., dflash=this) calls emit_tap from its
    programs; bind(dev) after). The matrices go to the GPU as the GGUF's rows (KQ_LINEAR; Q8_0 as Q8_R), fc as 8
    matrices of its column blocks (one a tap: the target's programs add each tap's product as they reach its
    layer)."""

    def __init__(self, path, model):
        super().__init__(path, embed=model.embed)
        assert self.hidden == model.cfg.hidden_size, "the drafter's hidden size is not the target's"
        assert self.vocab == model.M("lm_head").rows, "the drafter's vocabulary is not the target's head"
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
        # the Markov head: w1 the gather table (float32), w2 bfloat16 rows; the confidence's weights
        self.w1 = np.ascontiguousarray(self.f32("markov_head.markov_w1.weight"))
        raw, _dims, t = self.g.raw("markov_head.markov_w2.weight")
        assert t == BF16, "markov_w2 of type %d" % t
        self.w2 = np.ascontiguousarray(raw).view(np.uint16).reshape(-1)
        self.cw = np.ascontiguousarray(self.f32("confidence_head.proj.weight").reshape(-1))
        self.cb = float(self.f32("confidence_head.proj.bias").reshape(-1)[0])
        self.hid = f32(self.block, H)           # the draft program's normed hidden states (the Markov program reads)
        self._m = {}
        self._fc = None
        self.progs = {}
        self.capture = None         # [TMAX, hidden] a tap: the target's taps of each run (a check's; before QwenGPU)

    # ---- the matrices for KQ_LINEAR

    def M(self, name):
        """(blocks, type, rows, cols) of a matrix (a GGUF name) for KQ_LINEAR."""
        e = self._m.get(name)
        if e is None:
            raw, dims, t = self.g.raw(name)
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
        """fc [hidden, 8 hidden] as 8 matrices [hidden, hidden] of its column blocks (KQ_LINEAR rows)."""
        if self._fc is None:
            raw, dims, t = self.g.raw("fc.weight")
            cols, rows = int(dims[0]), int(dims[1])
            n, H = len(self.taps), self.hidden
            assert cols == n * H, "fc reads %d columns, not %d taps of %d" % (cols, n, H)
            parts = []
            if t in (BF12, BF16):
                from reclank.backends.cpu import cops
                # (BF12 holds bfloat16 values: a block of columns encodes again to the same values)
                full = self.g.dequant("fc.weight")
                bits = (full.view(np.uint32) >> 16).astype(np.uint16)
                for k in range(n):
                    blk = np.ascontiguousarray(bits[:, k * H:(k + 1) * H])
                    parts.append(blk.reshape(-1).view(np.uint8) if t == BF16 else
                                 np.ascontiguousarray(cops.kq_bf16_to_bf12(blk, H)[0]).reshape(-1))
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
        hidden_norm and each layer's k and v into its ring at pos (its first nreal rows)."""
        if i not in self.taps:
            return
        from reclank.ir import program as P
        assert t <= TMAX, "a program of the target of %d rows (the drafter's rows hold %d)" % (t, TMAX)
        k = self.taps.index(i)
        H = self.hidden
        x = prog.names["x"]
        if self.capture is not None:
            # (a check's: the tap's rows of the last run, scripts/check_dspark.py)
            prog.emit(P.COPY, x, self.capture[k][:t], t * H * 4)
            prog.keep.append(self.capture[k])
        acc, tmp = self.acc[:t], self.tmp[:t]
        self._lin(prog, self.fc_parts()[k], x, acc if k == 0 else tmp, t)
        if k > 0:
            prog.emit(P.ADD, acc, tmp, acc, t * H)
        prog.keep += [self.acc, self.tmp]
        if k < len(self.taps) - 1:
            return
        prog.emit(P.RMS_NORM, acc, self.f32("hidden_norm.weight"), acc, t, H, self.eps)
        pos, nreal = prog.slot("pos"), prog.slot("nreal")
        kb, vb = self.kb[:t], self.vb[:t]
        for li in range(self.layers):
            a = "layers.%d.self_attn." % li
            self._lin(prog, self.M(a + "k_proj.weight"), acc, kb, t)
            self._lin(prog, self.M(a + "v_proj.weight"), acc, vb, t)
            prog.emit(P.NM_DPREP, None, kb, vb, None, self.f32(a + "k_norm.weight"), self.ring_k[li],
                      self.ring_v[li], None, t, self.heads, self.kv, self.hd, pos, RING, self.theta, self.eps, nreal)
        prog.keep += [self.kb, self.vb] + self.ring_k + self.ring_v

    # ---- a draft

    def bind(self, dev):
        """The target's QwenGPU (its Mirror, its head)."""
        self.dev = dev

    def _draft_prog(self):
        """The block of t = block tokens from pos (their embedding rows in x): the normed hidden states (hid)."""
        from reclank.ir import program as P
        t, H, nq, nk = self.block, self.hidden, self.heads * self.hd, self.kv * self.hd
        prog = P.Program()
        f32 = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        x, a = f32(t, H), f32(t, H)
        q, k, v, qo, o, o2 = f32(t, nq), f32(t, nk), f32(t, nk), f32(t, nq), f32(t, nq), f32(t, H)
        gt, up, mm = f32(t, self.inter), f32(t, self.inter), f32(t, self.inter)
        ascr = f32(t * self.heads * (self.window + t))
        pos = prog.slot("pos")
        for i in range(self.layers):
            b = "layers.%d." % i
            prog.emit(P.RMS_NORM, x, self.f32(b + "input_layernorm.weight"), a, t, H, self.eps)
            self._lin(prog, self.M(b + "self_attn.q_proj.weight"), a, q, t)
            self._lin(prog, self.M(b + "self_attn.k_proj.weight"), a, k, t)
            self._lin(prog, self.M(b + "self_attn.v_proj.weight"), a, v, t)
            prog.emit(P.NM_DPREP, q, k, v, self.f32(b + "self_attn.q_norm.weight"),
                      self.f32(b + "self_attn.k_norm.weight"), self.ring_k[i], self.ring_v[i], qo, t, self.heads,
                      self.kv, self.hd, pos, RING, self.theta, self.eps)
            prog.emit(P.NM_DATTN, qo, self.ring_k[i], self.ring_v[i], self.nosink, o,
                      t, self.heads, self.kv, self.hd, pos, RING, self.window, ascr, 1)
            self._lin(prog, self.M(b + "self_attn.o_proj.weight"), o, o2, t)
            prog.emit(P.ADD, x, o2, x, t * H)
            prog.emit(P.RMS_NORM, x, self.f32(b + "post_attention_layernorm.weight"), a, t, H, self.eps)
            self._lin(prog, self.M(b + "mlp.gate_proj.weight"), a, gt, t)
            self._lin(prog, self.M(b + "mlp.up_proj.weight"), a, up, t)
            prog.emit(P.ENC_SILU_MUL, gt, up, mm, t * self.inter)
            self._lin(prog, self.M(b + "mlp.down_proj.weight"), mm, o2, t)
            prog.emit(P.ADD, x, o2, x, t * H)
        prog.emit(P.RMS_NORM, x, self.f32("norm.weight"), self.hid, t, H, self.eps)
        prog.names.update(x=x, hid=self.hid)
        prog.keep += [ascr, self.nosink, self.hid] + self.ring_k + self.ring_v
        return prog.finish()

    def _markov_prog(self):
        """The Markov head over the head's logits (lg, filled by a copy) from ids[0] (the last token): ids[1 ..]
        the drafts, conf the logit of each one's confidence (without the bias)."""
        from reclank.ir import program as P
        prog = P.Program()
        bs, V, rk = self.block, self.vocab, self.rank
        lg = np.zeros((bs, V), np.float32)
        me, bias, conf = np.zeros((bs, rk), np.float32), np.zeros((1, V), np.float32), np.zeros(bs, np.float32)
        ids = np.zeros(bs + 1, np.int32)
        for i in range(bs):
            prog.emit(P.DS_GATHER, self.w1, ids, i, me[i], rk)
            prog.emit(P.KQ_LINEAR, None, None, None, me[i:i + 1], self.w2, KQ_BF16, V, rk, 1, bias)
            prog.emit(P.DS_MARKOV, lg, bias, ids, i, V)
        prog.emit(P.DS_CONF, self.hid, me, self.cw, conf, bs, self.hidden, rk)
        prog.names.update(lg=lg, ids=ids, conf=conf)
        prog.keep += [lg, me, bias, conf, ids, self.w1, self.w2, self.cw, self.hid]
        return prog.finish()

    def _prog(self, key, make):
        from reclank.backends.cuda.runtime import GPUProgram
        e = self.progs.get(key)
        if e is None:
            prog = make()
            e = self.progs[key] = (prog, GPUProgram(prog, graph=True, mirror=self.dev.g.mirror))
        return e

    def drafts_gpu(self, last, pos0):
        """The draft program at pos0 (the context holds the positions before it), the target's head on its 8 rows,
        the Markov program: the 8 drafts and their confidence logits (host)."""
        from reclank.backends.cuda.runtime import _check, lib
        dev = self.dev
        prog, g = self._prog("draft", self._draft_prog)
        prog.names["x"][:] = self.embed([int(last)] + [self.mask] * (self.block - 1))
        g.upload("x")
        g.bind({"pos": int(pos0)})
        g.run()
        src = g.mirror.buffer_of(self.hid).ptr
        if dev._q6k_head():
            lg = dev.head_rows(src, self.block, x8=HEAD8)
        else:
            hp, hg = dev._kq_head(self.block)
            _check(lib().rc_gpu_d2d(hg.mirror.buffer_of(hp.names["x"]).ptr, src, self.block * self.hidden * 4))
            hg.run()
            lg = hg.mirror.buffer_of(hp.names["out"]).ptr
        mp, mg = self._prog("markov", self._markov_prog)
        _check(lib().rc_gpu_d2d(mg.mirror.buffer_of(mp.names["lg"]).ptr, lg, self.block * self.vocab * 4))
        ids = mp.names["ids"]
        ids[:] = 0
        ids[0] = int(last)
        mg.upload("ids")
        mg.bind({})
        mg.run()
        mg.download("ids")
        mg.download("conf")
        return [int(v) for v in ids[1:]], mp.names["conf"] + self.cb

    def hidden_gpu(self):
        """The draft program's hid of the last draft (host; a check's)."""
        _prog, g = self.progs["draft"]
        g.download("hid")
        return self.hid.copy()

    def draft(self, cache, last, pos0, n=None, eos_ids=()):
        return self.cut(*self.drafts_gpu(last, pos0), n=n, eos_ids=eos_ids)

    def release(self):
        for _p, g in self.progs.values():
            g.close()
        self.progs = {}
