"""The DSpark drafter of GLM-5.3-Flash on the GPU (P7 of docs/GLM53_PLAN.md):
RedHatAI/GLM-5.3-Flash-speculator.dspark-preview, its bfloat16 matrices as BF12.

The drafter (speculators' DSpark, replayed in scripts/glm53_dspark_reference.py: 4.02 tokens a verify
step on the trace with these taps):

- the context: fc of the streams entering the target layers 20, 28, 32, 36, 40, 44 (the mean of the
  4 streams after layers 19, 27, ...), hidden_norm, then for each of its 5 layers k_norm(k_proj) with
  the neox RoPE and v_proj, into rings of the last 2048 positions (the sliding window). The main
  model's programs make these rows as they run (compile_glm53_step with em.dspark: each tap through
  its slice of fc, summed; at the end the K/V into the rings), for every row they run (the rows of
  refused drafts are written again by the next run of those positions);
- a draft: the block [y, 7 x mask] at the positions after the context, 5 Qwen3 layers (64 heads of
  64, q/k norm, RoPE theta 1e4) whose attention reads the ring and the block's own rows (no causal
  mask), norm, lm_head; the Markov head in order (logits[i] += w2 w1[out[i]], out[i + 1] = argmax);
  the confidence logit of each position, proj(cat(hidden, w1[out[i]])) + bias.

The loop (DeepSeek V4.1's, reclank/models/deepseek41/dspark.py): a draft from the last token, a
verify length L from the confidences and the measured times of a draft and of a verify group of
L + 1 (0: a plain step), the group on the main model, the drafts that match its greedy choice kept
(GLM53GPU.commit), its choice after them the next token. Greedy: the tokens of plain decoding.

    m = GLM53Prog(path); ds = GLM53DSpark(); g = GLM53GPU(m, dspark=ds)
    out, stats = ds.generate(g, prompt_ids, 256)
"""
from __future__ import annotations

import json
import os
import struct
import time

import numpy as np

from reclank.backends.cpu import cops
from reclank.ir import program as P

DRAFT = "models/GLM-5.3-Flash-speculator.dspark-preview"      # (or RECLANK_GLM_DSPARK, or the path given)
KQ_BF12, KQ_BF16 = 57, 30


class GLM53DSpark:
    """The drafter's weights, rings, programs and loop (see the module text)."""

    VERIFY_MAX = 9

    def __init__(self, path=None):
        self.path = path or os.environ.get("RECLANK_GLM_DSPARK", DRAFT)
        c = json.load(open(os.path.join(self.path, "config.json")))
        tc = c["transformer_layer_config"]
        self.taps = [int(L) - 1 for L in c["aux_hidden_state_layer_ids"]]   # the streams entering each
        self.bs, self.mask = int(c["block_size"]), int(c["mask_token_id"])
        self.L, self.H, self.hd = int(tc["num_hidden_layers"]), int(tc["num_attention_heads"]), int(tc["head_dim"])
        assert int(tc["num_key_value_heads"]) == self.H and self.hd == 64
        self.dim, self.inter = int(tc["hidden_size"]), int(tc["intermediate_size"])
        self.eps, self.W = float(tc["rms_norm_eps"]), int(tc["sliding_window"])
        self.theta = float(tc["rope_parameters"]["rope_theta"])
        self.V = int(c["draft_vocab_size"])
        self.rank = int(c["markov_rank"])
        p = os.path.join(self.path, "model.safetensors")
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self._hdr = json.loads(f.read(n))
        self._hdr.pop("__metadata__", None)
        self._base = 8 + n
        self._map = np.memmap(p, np.uint8, "r")
        self.m = {}                 # name -> (BF12 data, rows, cols)
        for i in range(self.L):
            a = "layers.%d." % i
            for x in ("q", "k", "v", "o"):
                self._bf12(a + "self_attn.%s_proj.weight" % x)
            for x in ("gate", "up", "down"):
                self._bf12(a + "mlp.%s_proj.weight" % x)
        fc = self._bits("fc.weight")                                         # [4096, 6 x 4096]
        for k in range(len(self.taps)):
            part = np.ascontiguousarray(fc[:, k * self.dim:(k + 1) * self.dim])
            self.m["fc.%d" % k] = (cops.kq_bf16_to_bf12(part, self.dim)[0].reshape(-1), self.dim, self.dim)
        self._bf12("lm_head.weight")
        self.v = {}                 # the vectors, float32
        for k in self._hdr:
            if k.endswith("norm.weight"):
                self.v[k] = self._f32(k)
        self.w1 = self._f32("markov_head.markov_w1.weight")                 # [V, rank] (the gather table)
        self.w2 = np.array(self._bits("markov_head.markov_w2.weight"))       # [V, rank] bf16 (a copy: not the map)
        self.cw = np.ascontiguousarray(self._f32("confidence_head.proj.weight").reshape(-1))
        self.cb = float(self._f32("confidence_head.proj.bias").reshape(-1)[0])
        self.rings = [(np.zeros((self.W, self.dim), np.float32), np.zeros((self.W, self.dim), np.float32))
                      for _ in range(self.L)]
        self.progs = {}
        self.t_draft = 10.0
        self.t_verify = {t: 68.0 + 50.0 * (t - 1) for t in range(1, self.VERIFY_MAX + 1)}

    # ---- the weights
    def _bits(self, name):
        h = self._hdr[name]
        s, e = h["data_offsets"]
        assert h["dtype"] == "BF16", name
        return self._map[self._base + s:self._base + e].view(np.uint16).reshape(h["shape"])

    def _f32(self, name):
        return (np.asarray(self._bits(name)).astype(np.uint32) << 16).view(np.float32)

    def _bf12(self, name):
        b = np.ascontiguousarray(self._bits(name))
        rows, cols = b.shape
        self.m[name] = (cops.kq_bf16_to_bf12(b, cols)[0].reshape(-1), rows, cols)

    def nbytes(self):
        """The bytes of the drafter on the GPU (the budget of the hot experts leaves them)."""
        n = sum(d.nbytes for d, _r, _c in self.m.values()) + sum(a.nbytes for a in self.v.values())
        n += self.w1.nbytes + self.w2.nbytes + sum(k.nbytes + v.nbytes for k, v in self.rings)
        return n + (256 << 20)      # (the programs' own buffers)

    def embed_rows(self, ids):
        out = np.empty((len(ids), self.dim), np.float32)
        h = self._hdr["embed_tokens.weight"]
        s0 = self._base + h["data_offsets"][0]
        for j, t in enumerate(ids):
            row = self._map[s0 + 2 * self.dim * int(t):s0 + 2 * self.dim * (int(t) + 1)].view(np.uint16)
            out[j] = (row.astype(np.uint32) << 16).view(np.float32)
        return out

    def lin(self, prog, x, name, out, t):
        data, rows, cols = self.m[name]
        prog.emit(P.KQ_LINEAR, None, None, None, x, data, KQ_BF12, rows, cols, t, out)

    # ---- the context, in the main model's programs
    def emit_tap(self, prog, k, H, t, ctx, tmp):
        """Tap k (the streams H entering its target layer) through its slice of fc into ctx."""
        tap = tmp(t, self.dim)
        prog.emit(P.DS_STREAM_MEAN, H, tap, t, self.dim, self.dim, 0, H.shape[1] // self.dim)
        if k == 0:
            self.lin(prog, tap, "fc.0", ctx, t)
            return
        f = tmp(t, self.dim)
        self.lin(prog, tap, "fc.%d" % k, f, t)
        prog.emit(P.ADD, ctx, f, ctx, t * self.dim)

    def emit_context(self, prog, ctx, t, pos, nreal, tmp):
        """hidden_norm of ctx, then the K and V of each layer into its ring at pos (the real rows)."""
        cn = tmp(t, self.dim)
        prog.emit(P.RMS_NORM, ctx, self.v["hidden_norm.weight"], cn, t, self.dim, self.eps)
        for i in range(self.L):
            a = "layers.%d.self_attn." % i
            kk, k2, vv = tmp(t, self.dim), tmp(t, self.dim), tmp(t, self.dim)
            self.lin(prog, cn, a + "k_proj.weight", kk, t)
            prog.emit(P.RMS_NORM, kk, self.v[a + "k_norm.weight"], k2, t * self.H, self.hd, self.eps)
            prog.emit(P.GLM_ROPE, k2, pos, t, self.H, self.hd, self.theta)
            self.lin(prog, cn, a + "v_proj.weight", vv, t)
            rk, rv = self.rings[i]
            prog.emit(P.GLM_RING_PUT, k2, rk, pos, t, self.W, self.dim, nreal)
            prog.emit(P.GLM_RING_PUT, vv, rv, pos, t, self.W, self.dim, nreal)
        prog.keep += [a for kv in self.rings for a in kv]

    # ---- a draft
    def _compile(self, dev):
        bs, dim, H, hd = self.bs, self.dim, self.H, self.hd
        prog = P.Program()
        z = lambda *sh: np.zeros(sh, np.float32)  # noqa: E731
        pos = prog.slot("pos")
        x, a, q, k, v, q2, k2, o, y = (z(bs, dim) for _ in range(9))
        gg, uu, ss = z(bs, self.inter), z(bs, self.inter), z(bs, self.inter)
        hn, logits, me, bias, conf = z(bs, dim), z(bs, self.V), z(bs, self.rank), z(1, self.V), z(bs)
        ids = np.zeros(bs + 1, np.int32)
        prog.names.update(x=x, ids=ids, conf=conf)
        for i in range(self.L):
            p = "layers.%d." % i
            prog.emit(P.RMS_NORM, x, self.v[p + "input_layernorm.weight"], a, bs, dim, self.eps)
            for nm, out in (("q", q), ("k", k), ("v", v)):
                self.lin(prog, a, p + "self_attn.%s_proj.weight" % nm, out, bs)
            prog.emit(P.RMS_NORM, q, self.v[p + "self_attn.q_norm.weight"], q2, bs * H, hd, self.eps)
            prog.emit(P.RMS_NORM, k, self.v[p + "self_attn.k_norm.weight"], k2, bs * H, hd, self.eps)
            prog.emit(P.GLM_ROPE, q2, pos, bs, H, hd, self.theta)
            prog.emit(P.GLM_ROPE, k2, pos, bs, H, hd, self.theta)
            rk, rv = self.rings[i]
            prog.emit(P.GLM_DSP_ATTN, q2, rk, rv, k2, v, o, pos, self.W, H, hd, bs)
            self.lin(prog, o, p + "self_attn.o_proj.weight", y, bs)
            prog.emit(P.ADD, x, y, x, bs * dim)
            prog.emit(P.RMS_NORM, x, self.v[p + "post_attention_layernorm.weight"], a, bs, dim, self.eps)
            self.lin(prog, a, p + "mlp.gate_proj.weight", gg, bs)
            self.lin(prog, a, p + "mlp.up_proj.weight", uu, bs)
            prog.emit(P.DS_SWIGLU, gg, uu, ss, bs * self.inter, 3.0e38)
            self.lin(prog, ss, p + "mlp.down_proj.weight", y, bs)
            prog.emit(P.ADD, x, y, x, bs * dim)
        prog.emit(P.RMS_NORM, x, self.v["norm.weight"], hn, bs, dim, self.eps)
        self.lin(prog, hn, "lm_head.weight", logits, bs)
        for i in range(bs):
            prog.emit(P.DS_GATHER, self.w1, ids, i, me[i], self.rank)
            prog.emit(P.KQ_LINEAR, None, None, None, me[i:i + 1], self.w2, KQ_BF16, self.V, self.rank, 1, bias)
            prog.emit(P.DS_MARKOV, logits, bias, ids, i, self.V)
        prog.emit(P.DS_CONF, hn, me, self.cw, conf, bs, dim, self.rank)
        prog.keep += [a for kv in self.rings for a in kv]
        return prog.finish()

    def draft(self, dev, y, pos):
        """The 8 drafts after token y at position pos (the context: the rings), and the confidence
        logit of each."""
        # a program of the model's LRU (gpumm.ProgramLRU, as gpu38's MTP programs): its buffers
        # come and go with the memory the cache lends (a GPUProgram of its own got buffers in the
        # cache's lent rows, which the next steps wrote: the tokens changed)
        key = "dspark"
        e = dev.mtp_progs.get(key)
        if e is None:
            prog = self._compile(dev)
            e = dev.mtp_progs[key] = (prog, dev._gpu_program(prog, ("mtp_progs", key)))
        dev._used(key, "mtp_progs")
        prog, g = e
        tokens = [y] + [self.mask] * (self.bs - 1)
        prog.names["x"][:] = self.embed_rows(tokens)
        g.upload("x")
        ids = prog.names["ids"]
        ids[:] = 0
        ids[0] = y
        g.upload("ids")
        g.bind({"pos": int(pos)})
        g.run()
        g.download("ids")
        g.download("conf")
        return ids[1:].copy(), prog.names["conf"] + self.cb

    def reset(self):
        """(nothing: a run writes the rows of the positions it runs before a draft reads them)"""

    # ---- the loop
    def choose(self, conf):
        """The verify length: the most expected tokens a millisecond, from the survival of each
        prefix (the product of the sigmoids of the confidences) and the measured times."""
        p = 1.0 / (1.0 + np.exp(-conf.astype(np.float64)))
        surv = np.cumprod(p)
        best, bl = 1.0 / self.t_verify[1], 0
        for L in range(1, len(conf) + 1):
            if L + 1 > self.VERIFY_MAX:
                break
            rate = (1.0 + surv[:L].sum()) / (self.t_draft + self.t_verify[L + 1])
            if rate > best:
                best, bl = rate, L
        return bl

    def generate(self, dev, prompt, n, fixed=None, stop=()):
        """n greedy tokens after the prompt, from a fresh cache attached to dev (GLM53GPU made with
        this drafter). fixed: that many drafts every round (else choose). Returns (tokens, stats)."""
        prompt = list(prompt)
        dev.prefill(prompt)
        y, pos = int(dev.argmax(1)[0]), len(prompt)
        out, st = [], dict(rounds=0, plain=0, drafted=0, accepted=0, lengths=[])
        t_start = time.time()
        while len(out) < n:
            t0 = time.time()
            drafts, conf = self.draft(dev, y, pos)
            self.t_draft = 0.9 * self.t_draft + 0.1 * (time.time() - t0) * 1e3
            L = self.choose(conf) if fixed is None else fixed
            t0 = time.time()
            if L == 0:
                dev.step(y, pos)
                out.append(y)
                y, pos = int(dev.argmax(1)[0]), pos + 1
                self.t_verify[1] = 0.8 * self.t_verify[1] + 0.2 * (time.time() - t0) * 1e3
                st["plain"] += 1
            else:
                dev.verify([y] + [int(d) for d in drafts[:L]], pos)
                tok = dev.argmax(L + 1)
                a = 0
                while a < L and int(tok[a]) == int(drafts[a]):
                    a += 1
                dev.commit(a + 1)
                out += [y] + [int(d) for d in drafts[:a]]
                y, pos = int(tok[a]), pos + a + 1
                t = L + 1
                self.t_verify[t] = 0.8 * self.t_verify[t] + 0.2 * (time.time() - t0) * 1e3
                st["rounds"] += 1
                st["drafted"] += L
                st["accepted"] += a
                st["lengths"].append(L)
            if any(tk in stop for tk in out[-(L + 1):]):
                break
        st["seconds"] = time.time() - t_start
        st["t_verify"] = {t: round(v, 1) for t, v in self.t_verify.items()}
        st["t_draft"] = round(self.t_draft, 1)
        return out[:n], st
