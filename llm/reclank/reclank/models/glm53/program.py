"""The program of a step of GLM-5.3-Flash on the GPU (P5 of docs/GLM53_PLAN.md): the records of
the layers of the NumPy model (reclank/models/glm53/model.py) for the GPU runner, and the model as
the GPU machinery of Qwen (reclank/models/qwen/gpu35.py, QwenGPU) takes it.

compile_glm53_step(em, t) has the input names["H"] (t x 4 hidden: the embeddings in each stream)
and the output names["xn"] (the input of the head). A layer:

- mHC (DeepSeek V4.1's math): GLM_HC_MIX (the mixes of the streams, and the post and comb
  coefficients once a token), DS_HC_PRE (the block input and its RMSNorm), then GLM_HC_POST (post y
  + comb^T H, the streams in place);
- KDA: the EXL3 qkv (KQ_LINEAR), the float16 low-rank pairs and beta (F16_GLINEAR), GDN with flag
  4 (the decay of each channel) and the sigmoid gate (2), the EXL3 o_proj;
- DSA: q_a, its norm, q_b, kv_a and its norm; the latents into the cache (DS_RING_PUT over the
  whole cache: bfloat16 rows); the indexer: wk x, the pool gate, weights_proj (F16_GLINEAR),
  GLM_POOL, DS_IDX_SCORE, DS_TOPK (512 pools), GLM_EXPAND (their tokens and the tail); the
  absorbed MLA: F16_GLINEAR (W_UK^T q), DS_ATTN (no window, no sink: -inf, scale 1/16) over the
  selected latents, F16_GLINEAR (W_UV o), the EXL3 o_proj;
- the MLP: layers 0-2 the EXL3 gate, up, DS_SWIGLU (the clamp), down; the MoE layers the float16
  router, ROUTER_SIG (sigmoid, the bias for the choice, 8 weights normed with 1e-20, x 2.5), then
  the experts of the machinery (em.moe: the hot experts and the shared expert on the GPU, the cold
  ones on the CPU; the shared expert ungated: its logit +inf, sigmoid 1).

The head: DS_STREAM_MEAN of the streams, RMS_NORM.

The slots of a run: pos, nreal, maxlen (the rows of the cache), scores (the scratch of the scores
of the pools, t x em.maxn: the pools of the largest context, a literal of the records), and the
cache: conv.<i>, S.<i> (KDA), lat.<i>, pool.<i>, kst.<i>,
sst.<i> (DSA: the latents, the pooled keys, the keys and gates of the open pool).
"""
from __future__ import annotations

import os

import numpy as np

from reclank.formats.exl3 import KQ_E3_BASE
from reclank.formats.exl3_glm53 import KQ_F16

LAT_FORM = 1        # the latents: bfloat16 (1 KB a token and layer)
POOL_FORM = 0       # the pooled keys: float32


class GLM53Cache:
    """The state of a sequence on the GPU (its arrays go there: gpu35._DevCache): the conv rows and
    the delta-rule state of each KDA layer; for each DSA layer (kv) the latents [max_len, 512]
    (bfloat16 bits), the pooled keys [max_len / 4 + 1, 128] (float32), the keys and gates of the
    open pool [4, 128] each."""

    def __init__(self, cfg, max_len=4096):
        self.max_len, self.n = max_len, 0
        self.ple_ids = np.zeros(0, np.int64)        # (no n-gram layer: gpu38.Qwen4GPU.group)
        C = 3 * cfg.kda_heads * cfg.kda_dim
        L = getattr(cfg, "n_layers", cfg.num_hidden_layers)
        self.conv, self.state, self.kv = {}, {}, {}
        for i in range(L):
            if cfg.layer_types[i] == "kda":
                self.conv[i] = np.zeros((cfg.conv_kernel - 1, C), np.float32)
                self.state[i] = np.zeros((cfg.kda_heads, cfg.kda_dim, cfg.kda_dim), np.float32)
            else:
                lat = np.zeros((max_len, cfg.kv_lora_rank), np.uint16 if LAT_FORM == 1 else np.float32)
                pool = np.zeros((max_len // cfg.index_pool + 1, cfg.index_dim), np.float32)
                self.kv[i] = (lat, pool, np.zeros((cfg.index_pool, cfg.index_dim), np.float32),
                              np.zeros((cfg.index_pool, cfg.index_dim), np.float32))


class _Mat:
    """A matrix as the machinery takes it (qwen35.KMat): data, type, rows, cols."""

    def __init__(self, data, type_, rows, cols, nb=None):
        self.data, self.type, self.rows, self.cols, self.rot = data, type_, rows, cols, False
        if nb is not None:
            self.nb = nb

    def c(self):
        return (self.data, self.type)


# the names of the Qwen3.6 modules that gpu35 asks for (M), in the GGUF names of GLM53Source
_MOD = {"switch_mlp.gate_proj": "ffn_gate_exps.weight", "switch_mlp.up_proj": "ffn_up_exps.weight",
        "switch_mlp.down_proj": "ffn_down_exps.weight", "shared_expert.gate_proj": "ffn_gate_shexp.weight",
        "shared_expert.up_proj": "ffn_up_shexp.weight", "shared_expert.down_proj": "ffn_down_shexp.weight"}


class GLM53Prog:
    """GLM-5.3-Flash as QwenGPU takes a model: g (the GGUF view: GLM53Source), cfg (GLM53Config with
    the fields of the machinery), K / M (matrices), F (float32 tensors), embed."""

    moe_act = 512           # the CPU experts: the clamped SwiGLU (kq_e3_moe_body, the parts)
    # the CPU experts on int8 x (RECLANK_MOE_X16=1: int16, two planes): decode 9.5% faster at 2048
    # (16.8 -> 18.4 tok/s), the KL of the decode rows against the reference within the noise of runs
    # (1.04e-2; int16 9.7e-3 to 1.27e-2), the prompt rows 9.8e-2 against 9.4e-2
    moe_x16 = os.environ.get("RECLANK_MOE_X16", "0") == "1"
    rot_experts = False
    dense = "bf16"

    def __init__(self, path, layers=None, src=None):
        """The tensors from the weightkeeper when it holds the directory (keep_exl3.KeptEXL3: the
        stacks of the experts in its segments), else from the files (GLM53Source)."""
        from reclank.formats.exl3_glm53 import GLM53Source
        from reclank.models.qwen import keep
        self.path = path
        if src is None and layers is None and keep.holds(path):
            from reclank.models.qwen.keep_exl3 import KeptEXL3
            src = KeptEXL3(path)
        self.g = src or GLM53Source(path, mtp=False, layers=layers)
        c = self.cfg = self.g.config()
        self.n_layers = layers or c.num_hidden_layers
        # the fields of gpu35 / gpu38
        c.n_layers = self.n_layers
        c.shared_inter = c.moe_inter
        c.dense_layers = sum(1 for t in c.mlp_types if t == "dense")
        assert c.mlp_types[:c.dense_layers] == ["dense"] * c.dense_layers
        c.hc_count = c.hc_mult
        c.ple_layers = []
        c.lin_k_heads = c.lin_v_heads = c.kda_heads
        c.lin_k_dim = c.lin_v_dim = c.kda_dim
        self._k, self._f = {}, {}

    def K(self, gname):
        m = self._k.get(gname)
        if m is None and getattr(self.g, "kept", False) and self.g.index["tensors"][gname]["place"] == "experts":
            # a stack in the weightkeeper (as qwen38.Qwen4CPU.K): kept, the parts of each node
            d = self.g.index["tensors"][gname]
            m = _Mat(None, int(d["type"]), int(d["dims"][1]), int(d["dims"][0]), int(d["nb"]))
            m.kept = self.g.experts(gname)
            m.data = m.kept[0][1].reshape(-1)
            self._k[gname] = m
        if m is None:
            a, dims, t = self.g.raw(gname)
            if t >= KQ_E3_BASE and len(dims) == 3:
                # a stack of experts: rows, cols of one; nb the bytes of one
                m = _Mat(a.reshape(-1).view(np.uint8), t, int(dims[1]), int(dims[0]), a.nbytes // int(dims[2]))
            else:
                m = _Mat(a, t, int(dims[1]) if len(dims) > 1 else 1, int(dims[0]))
            self._k[gname] = m
        return m

    def M(self, name):
        if name == "lm_head":
            return self.K("output.weight")
        if name in self.g.tensors:
            return self.K(name)
        i, mod = name.split(".", 2)[1], name.split(".", 3)[3]
        return self.K("blk.%s.%s" % (i, _MOD[mod]))

    def F(self, gname, shape=None):
        a = self._f.get((gname, shape))
        if a is None:
            a = np.ascontiguousarray(self.g.dequant(gname), np.float32)
            a = self._f[(gname, shape)] = a.reshape(shape) if shape is not None else a
        return a

    def f16(self, gname):
        """The float16 bits of a KQ_F16 matrix (rows x cols, uint16)."""
        a = self._f.get((gname, "f16"))
        if a is None:
            assert self.g.tensors[gname][1] == KQ_F16, gname
            a = self._f[(gname, "f16")] = np.ascontiguousarray(self.g.raw(gname)[0]).view(np.uint16)
        return a

    def f16cat(self, names):
        """The float16 bits of several KQ_F16 matrices of one width, their rows one after the other
        (a step: one product in place of several; made once)."""
        key = ("cat",) + tuple(names)
        a = self._f.get(key)
        if a is None:
            a = self._f[key] = np.ascontiguousarray(np.concatenate([self.f16(n).reshape(
                self.g.tensors[n][0][1], self.g.tensors[n][0][0]) for n in names]))
        return a

    def kvb(self, i):
        """(W_UK^T [64 x 512, 256], W_UV [64 x 256, 512]) of DSA layer i, float16 bits: the groups
        of F16_GLINEAR."""
        a = self._f.get(("kvb", i))
        if a is None:
            b = "blk.%d." % i
            kb, vb = (np.ascontiguousarray(self.g.raw(b + n)[0]).view(np.uint16).reshape(
                tuple(reversed(self.g.tensors[b + n][0]))) for n in ("attn_k_b.weight", "attn_v_b.weight"))
            # [H, 256, 512] each
            H, dk, dl = kb.shape
            a = self._f[("kvb", i)] = (np.ascontiguousarray(kb.transpose(0, 2, 1)).reshape(H * dl, dk),
                                       np.ascontiguousarray(vb).reshape(H * vb.shape[1], dl))
        return a

    def embed(self, ids):
        return self.g.embed_rows(ids).astype(np.float32)

    def cache(self, max_len):
        return GLM53Cache(self.cfg, max_len)


def compile_glm53_step(em, t, verify=False):
    """The program of t tokens (see the module text). em: the model and the machinery as the
    program sees them (gpu.GLM53Emit): cfg, F, f16, kvb, dense (the device copy of an EXL3 matrix
    and its type), moe (the experts of a layer), layer_begin, pack_temps; verify: the KDA layers
    write a log (names["log.<i>"]) and keep their state, the DSA layers the open pool before the
group (GLM53GPU.commit)."""
    from reclank.ir import program as P
    cfg = em.cfg
    g = em.g
    prog = P.Program()
    hid, hc, eps = cfg.hidden_size, cfg.hc_mult, float(cfg.rms_norm_eps)
    HD = hc * hid
    nh, kd = cfg.kda_heads, cfg.kda_dim
    cd, vd = 3 * nh * kd, nh * kd
    H_, dk, dl, dv = cfg.num_heads, cfg.qk_head_dim, cfg.kv_lora_rank, cfg.v_head_dim
    ratio, ID, IH = cfg.index_pool, cfg.index_dim, cfg.index_heads
    kp = cfg.index_topk // ratio
    maxk = cfg.index_topk + ratio - 1
    k, E = cfg.top_k, cfg.num_experts
    H, xn = np.zeros((t, HD), np.float32), np.zeros((t, hid), np.float32)
    prog.names.update(H=H, xn=xn)
    pos, nreal = prog.slot("pos"), prog.slot("nreal")
    maxlen, scores = prog.slot("maxlen"), prog.slot("scores")
    maxn = em.maxn
    pack = getattr(em, "pack_temps", False)
    sink = np.full(H_, -np.inf, np.float32)
    slog = np.full(t, np.inf, np.float32)           # the shared expert ungated: sigmoid(+inf) = 1
    prog.keep += [sink, slog]

    def tmp(*sh, dtype=np.float32):
        a = P.lazy_zeros(sh, dtype)
        if pack:
            prog.temps.append(a)
        return a

    def lin(gname, x, out, rows=t):
        dims, type_, _ = g.tensors[gname]
        if type_ >= KQ_E3_BASE:
            w, ty = em.dense(gname)
            prog.emit(P.KQ_LINEAR, None, None, None, x, w, ty, int(dims[1]), int(dims[0]), rows, out)
        else:
            assert type_ == KQ_F16, (gname, type_)
            prog.emit(P.F16_GLINEAR, x, em.f16(gname), out, int(dims[1]), int(dims[0]), rows, 1)

    def hc_pre(prefix, norm, raw, coef, x):
        prog.emit(P.GLM_HC_MIX, H, em.F(prefix + "_fn"), em.F(prefix + "_base"), em.F(prefix + "_scale"), raw, coef,
                  tmp(t, 32, 25), t, HD, eps, float(cfg.hc_eps), cfg.hc_iters)
        prog.emit(P.DS_HC_PRE, H, raw, em.F(prefix + "_scale"), em.F(prefix + "_base"), em.F(norm), x, t, hid,
                  eps, float(cfg.hc_eps))

    def hc_post(coef, y):
        prog.emit(P.GLM_HC_POST, y, H, coef, H, t, hid)

    def kda(i, x, y):
        b = "blk.%d." % i
        qkv, att, gscr = tmp(t, cd), tmp(t, vd), tmp(t * cd)
        lin(b + "attn_qkv.weight", x, qkv)
        if t == 1:
            # a step: f_a, g_a and b_proj (one input) as one product, then f_b and g_b as one
            # grouped product (the outputs side by side: a row alone, so the slices are its parts)
            fgb, fg = tmp(1, 2 * kd + nh), tmp(1, 2 * vd)
            w1 = em.f16cat([b + "ssm_f_a.weight", b + "ssm_g_a.weight", b + "ssm_beta.weight"])
            w2 = em.f16cat([b + "ssm_f_b.weight", b + "ssm_g_b.weight"])
            prog.emit(P.F16_GLINEAR, x, w1, fgb, 2 * kd + nh, hid, 1, 1)
            prog.emit(P.F16_GLINEAR, fgb, w2, fg, 2 * vd, kd, 1, 2)
            f, gate, beta = fg[:, :vd], fg[:, vd:], fgb[:, 2 * kd:]
        else:
            fa, f, beta, ga, gate = tmp(t, kd), tmp(t, vd), tmp(t, nh), tmp(t, kd), tmp(t, vd)
            lin(b + "ssm_f_a.weight", x, fa)
            lin(b + "ssm_f_b.weight", fa, f)
            lin(b + "ssm_beta.weight", x, beta)
            lin(b + "ssm_g_a.weight", x, ga)
            lin(b + "ssm_g_b.weight", ga, gate)
        log = None
        if verify:
            log = prog.names["log.%d" % i] = np.zeros(t * (cd + nh * 3 * kd), np.float32)
        prog.emit(P.GDN, qkv, prog.slot("conv.%d" % i), em.F(b + "ssm_conv1d.weight"), cfg.conv_kernel, gate, f,
                  beta, em.F(b + "ssm_a_log"), em.F(b + "ssm_dt.bias"), em.F(b + "ssm_norm.weight"),
                  prog.slot("S.%d" % i), att, gscr, t, nh, nh, kd, kd, eps, log, 2 | 4, nreal)
        lin(b + "attn_output.weight", att, y)

    def dsa(i, x, y):
        b = "blk.%d." % i
        qa, qr, q = tmp(t, cfg.q_lora_rank), tmp(t, cfg.q_lora_rank), tmp(t, H_ * dk)
        c0, c = tmp(t, dl), tmp(t, dl)
        lin(b + "attn_q_a.weight", x, qa)
        prog.emit(P.RMS_NORM, qa, em.F(b + "attn_q_a_norm.weight"), qr, t, cfg.q_lora_rank, eps)
        lin(b + "attn_q_b.weight", qr, q)
        pbk = pbs = None
        lin(b + "attn_kv_a_mqa.weight", x, c0)
        prog.emit(P.RMS_NORM, c0, em.F(b + "attn_kv_a_norm.weight"), c, t, dl, eps)
        lat = prog.slot("lat.%d" % i)
        prog.emit(P.DS_RING_PUT, c, lat, pos, t, maxlen, dl, None, LAT_FORM)
        # the indexer
        kr, gt, wt, qi = tmp(t, ID), tmp(t, ID), tmp(t, IH), tmp(t, IH * ID)
        if verify:
            # the normed keys and the gates of the group, and the open pool before it: the commit
            # (GLM53GPU.commit, DS_COMPRESS_FIX) makes the open pool of the kept tokens
            kr, gt = (prog.names.setdefault("%s.%d" % (n, i), np.zeros((t, ID), np.float32)) for n in ("pk", "pg"))
            pbk, pbs = (prog.names.setdefault("%s.%d" % (n, i), np.zeros((ratio, ID), np.float32))
                        for n in ("pbk", "pbs"))
        psel, pcnt = tmp(t, kp, dtype=np.int32), tmp(t, dtype=np.int32)
        sel, cnt = tmp(t, maxk, dtype=np.int32), tmp(t, dtype=np.int32)
        if t == 1 and not verify:
            # a step: wk, the pool gate and weights_proj (one input) as one product
            kgw = tmp(1, 2 * ID + IH)
            prog.emit(P.F16_GLINEAR, x, em.f16cat([b + "idx_k.weight", b + "idx_gate.weight", b + "idx_w.weight"]),
                      kgw, 2 * ID + IH, hid, 1, 1)
            kr, gt, wt = kgw[:, :ID], kgw[:, ID:2 * ID], kgw[:, 2 * ID:]
        else:
            lin(b + "idx_k.weight", x, kr)
            lin(b + "idx_gate.weight", x, gt)
            lin(b + "idx_w.weight", x, wt)
        lin(b + "idx_q_b.weight", qr, qi)
        pool = prog.slot("pool.%d" % i)
        prog.emit(P.GLM_POOL, kr, gt, prog.slot("kst.%d" % i), prog.slot("sst.%d" % i), em.F(b + "idx_ape"), pool,
                  pos, ratio, ID, POOL_FORM, t, 1e-6, pbk, pbs, em.F(b + "idx_k_norm.weight"),
                  em.F(b + "idx_k_norm.bias"))
        prog.emit(P.DS_IDX_SCORE, qi, wt, float((ID * IH) ** -0.5), pool, pos, ratio, None, 1, scores, IH, ID,
                  maxn, t, 0, POOL_FORM, 0)
        prog.emit(P.DS_TOPK, scores, pos, ratio, 1, kp, psel, pcnt, None, maxn, t)
        prog.emit(P.GLM_EXPAND, psel, pcnt, sel, cnt, pos, ratio, kp, maxk, t)
        # the absorbed MLA
        wukt, wuv = em.kvb(i)
        ql, ol, att = tmp(t, H_ * dl), tmp(t, H_ * dl), tmp(t, H_ * dv)
        prog.emit(P.F16_GLINEAR, q, wukt, ql, H_ * dl, dk, t, H_)
        # a step or a verify group: the keys in 8 ranges, each query's keys decoded once into the
        # keys buffer of the model (k_ds_attn_keys, k_ds_attn_s, k_ds_attn_join); one block for 4 heads
        # over all the keys took 1.8 ms a layer at 8K
        if t > 8 and LAT_FORM == 1 and getattr(em, "tc", False):
            # a prompt group: the tensor cores (k_glm_attn_tc; k_ds_attn_g took 510 ms a layer of 4096 rows at 8K)
            prog.emit(P.GLM_ATTN, ql, lat, sel, cnt, ol, H_, dl, float(dk ** -0.5), maxk, t)
        else:
            part, akeys = (tmp(t, H_, 8, dl + 2), em.attn_keys(maxk, dl)) if t <= 8 else (None, None)
            prog.emit(P.DS_ATTN, ql, None, pos, 0, lat, sel, cnt, sink, ol, H_, dl, float(dk ** -0.5), maxk, None, t,
                      part, None, LAT_FORM, None, akeys)
        prog.emit(P.F16_GLINEAR, ol, wuv, att, H_ * dv, dl, t, H_)
        lin(b + "attn_output.weight", att, y)

    def mlp(i, x, y):
        b = "blk.%d." % i
        n = cfg.dense_inter
        gg, uu, a = tmp(t, n), tmp(t, n), tmp(t, n)
        lin(b + "ffn_gate.weight", x, gg)
        lin(b + "ffn_up.weight", x, uu)
        prog.emit(P.DS_SWIGLU, gg, uu, a, t * n, float(cfg.swiglu_limit))
        lin(b + "ffn_down.weight", a, y)

    def moe(i, x, y):
        b = "blk.%d." % i
        logits, val, idx = tmp(t, E), tmp(t, k), tmp(t, k, dtype=np.int32)
        lin(b + "ffn_gate_inp.weight", x, logits)
        prog.emit(P.ROUTER_SIG, logits, t, E, k, val, idx, em.F(b + "exp_probs_b.bias"),
                  float(cfg.routed_scale), 1e-20)
        em.moe(prog, i, x, idx, val, slog, y)

    ds = getattr(em, "dspark", None)
    ctx = tmp(t, hid) if ds is not None else None
    for i in range(em.n_layers):
        b = "blk.%d." % i
        if ds is not None and i in ds.taps:
            # the DSpark context: the streams entering the next layer, through their slice of fc
            ds.emit_tap(prog, ds.taps.index(i), H, t, ctx, tmp)
        em.layer_begin(prog, i)
        raw, coef, x, y = tmp(t, 24), tmp(t, 20), tmp(t, hid), tmp(t, hid)
        hc_pre(b + "hc_attn", b + "attn_norm.weight", raw, coef, x)
        (kda if cfg.layer_types[i] == "kda" else dsa)(i, x, y)
        hc_post(coef, y)
        raw2, coef2, x2, y2 = tmp(t, 24), tmp(t, 20), tmp(t, hid), tmp(t, hid)
        hc_pre(b + "hc_ffn", b + "ffn_norm.weight", raw2, coef2, x2)
        (moe if cfg.mlp_types[i] == "moe" else mlp)(i, x2, y2)
        hc_post(coef2, y2)
    if ds is not None:
        ds.emit_context(prog, ctx, t, pos, nreal, tmp)
    hm = tmp(t, hid)
    prog.emit(P.DS_STREAM_MEAN, H, hm, t, hid, hid, 0, hc)
    prog.emit(P.RMS_NORM, hm, em.F("output_norm.weight"), xn, t, hid, eps)
    prog.tokens = t
    return prog.finish()
