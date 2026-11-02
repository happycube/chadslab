"""DiffusionGemma on the GPU, from the BF12 GGUF (D3 of docs/DIFFUSIONGEMMA_PLAN.md).

Three programs (records of reclank/ir/program.py, one CUDA graph each) of a block of C = 256 rows:

    enc    the encoder over a block (a prompt chunk, padded; or a finished canvas): the rows'
           keys and values into the cache (DG_KV_PUT), causal attention over it (DG_ATTN)
    dec0   the decoder over the canvas, the first step (no self-conditioning)
    dec1   a later step: the soft embedding of the last step's probabilities (KQ_LINEAR on the
           transposed BF12 token table) through the self-conditioning MLP

A decoder pass writes the canvas's keys and values after the cache (they are not part of it: the
encoder writes the block's own there later) and reads the cache and the canvas with no mask: in
a layer with a window the last window - 1 cached rows, in a global one all. After the head (the
tied table, the soft cap) DG_SAMPLE gives each row's entropy, argmax and sample; the processed
logits stay on the device and their softmax is the next step's self-conditioning input.

The matrices are BF12 (KQ_LINEAR: the tensor cores with x as two bfloat16 planes for the 256
rows). The norms, the rope (the tables of transformers' float32 recipe, from the host) and the
router run on the GPU. The experts, milestone 1, run on the CPU: KQ_MOE (the tanh GELU, float32
rows of BF12 experts in register tiles: act bit 7) between TO_HOST and TO_DEV (CPU_JOIN).

The KV cache: float32 arrays of each layer, shared by the programs (the Mirror's device copies,
written on the device). A windowed layer's buffer holds S rows from position base on; before a
block would pass its end, its last window - 1 rows move to the front.

    m = DGemmaGPU("MODELS/diffusiongemma-26B-A4B-it-BF12-GGUF")
    m.reset()
    m.encode(prompt_ids)
    out, stats = sampler.generate(m.encode, None, ..., steps=m.steps)
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from reclank.backends.cpu import cops
from reclank.ir import program as P
from reclank.formats.gguf import GGUF
from reclank.models.dgemma.model import inv_freq_f32
from reclank.models.gemma4.config import Config
from reclank.models.gemma4.e4b import _KQ

BF12 = 57


class _Lin:
    """A matrix for enc_linear: BF12 rows (lin.kq), no clamps, no bias."""

    def __init__(self, raw, rows, cols, type_=BF12):
        self.w = None
        self.b = None
        self.imin = self.imax = self.omin = self.omax = None
        self.kq = _KQ(raw, type_, rows, cols)


def _gguf_file(path):
    if os.path.isdir(path):
        names = [f for f in os.listdir(path) if f.endswith(".gguf")]
        assert len(names) == 1, "one .gguf file in %s, not %s" % (path, names)
        return os.path.join(path, names[0]), path
    return path, os.path.dirname(path)


class DGemmaGPU:
    """The model on the GPU (the experts on the CPU). The cache holds max_len positions."""

    # the device bytes besides the experts' (the dense BF12 matrices, the transposed table, the
    # cache, the logits and probabilities, the programs' scratch: 8.6 GB measured) and a margin
    RESERVE = 10.0e9

    def __init__(self, path, max_len=8192, slide_rows=4096, gpu_layers=None):
        from reclank.backends.cuda.gpumm import pinned
        from reclank.backends.cuda.runtime import Mirror
        gpath, d = _gguf_file(path)
        raw_cfg = json.load(open(os.path.join(d, "config.json")))
        self.gen_cfg = json.load(open(os.path.join(d, "generation_config.json")))
        self.raw_cfg = raw_cfg
        self.cfg = Config(raw_cfg)
        cfg = self.cfg
        self.C = raw_cfg["canvas_length"]
        self.H = cfg.hidden_size
        self.L = cfg.num_hidden_layers
        self.vocab = cfg.vocab_size
        self.eps = cfg.rms_norm_eps
        self.window = cfg.sliding_window
        self.E, self.topk, self.inner = cfg.num_experts, cfg.top_k_experts, cfg.moe_intermediate_size
        self.max_len, self.S = max_len, slide_rows
        assert slide_rows >= self.window - 1 + 2 * self.C
        t0 = time.time()
        g = self.g = GGUF(gpath)

        def raw(name):
            return g.raw(name)[0]

        def lin(name):
            r, dims, t = g.raw(name)
            assert t == BF12, (name, t)
            return _Lin(r, int(np.prod(dims[1:])), int(dims[0]))

        f32 = lambda name: np.ascontiguousarray(g.dequant(name), dtype=np.float32).reshape(-1)  # noqa: E731
        self.table = raw("token_embd.weight")
        self.head = lin("token_embd.weight")
        self.norm = f32("output_norm.weight")
        self.sc = dict(pre=f32("self_cond_pre_norm.weight"), gate=lin("self_cond_gate.weight"),
                       up=lin("self_cond_up.weight"), down=lin("self_cond_down.weight"))
        # the transposed table (hidden rows of vocab values) for the soft embedding
        tab = g.dequant("token_embd.weight")
        bits = np.ascontiguousarray(tab.T).view(np.uint32) >> 16
        del tab
        tt, _rep = cops.kq_bf16_to_bf12(bits.astype(np.uint16), self.vocab)
        del bits
        self.table_t = _Lin(tt.reshape(-1), self.H, self.vocab)
        self.embed_scale = np.float32(self.H ** 0.5)
        self.blk = []
        for i in range(self.L):
            p = "blk.%d." % i
            plan = cfg.plan[i]
            b = dict(plan=plan,
                     ln_in=f32(p + "attn_norm.weight"), q=lin(p + "attn_q.weight"), k=lin(p + "attn_k.weight"),
                     v=None if plan.k_eq_v else lin(p + "attn_v.weight"), o=lin(p + "attn_output.weight"),
                     qn=f32(p + "attn_q_norm.weight"), kn=f32(p + "attn_k_norm.weight"),
                     post_attn=f32(p + "post_attention_norm.weight"), pre_ff=f32(p + "ffn_norm.weight"),
                     gate=lin(p + "ffn_gate.weight"), up=lin(p + "ffn_up.weight"), down=lin(p + "ffn_down.weight"),
                     post_ff1=f32(p + "post_ffw_norm_1.weight"), pre_ff2=f32(p + "pre_ffw_norm_2.weight"),
                     post_ff2=f32(p + "post_ffw_norm_2.weight"), post_ff=f32(p + "post_ffw_norm.weight"),
                     r_proj=np.ascontiguousarray(g.dequant(p + "ffn_gate_inp.weight"), np.float32),
                     r_scale=f32(p + "ffn_gate_inp.scale"), r_pes=f32(p + "ffn_down_exps.scale"))
            for k, n in (("dec", "layer_output_scale.weight"), ("enc", "enc_layer_output_scale.weight")):
                b[k + "_vec"] = np.full(self.H, f32(p + n)[0], np.float32)
            # the experts for KQ_MOE, as the file has them: each expert's gate rows then its up
            # rows (no up matrix: KQ_MOE's fused stack), and the down rows
            gu, down = raw(p + "ffn_gate_up_exps.weight"), raw(p + "ffn_down_exps.weight")
            b["mats"] = cops.kq_moe_mats((gu, BF12), None, (down, BF12), None)
            b["experts"] = (gu, down)
            self.blk.append(b)
        g.drop_cache() if hasattr(g, "drop_cache") else None
        print("dgemma gpu: weights %.0f s" % (time.time() - t0), flush=True)
        # the cache (float32 rows of kvh x hd each layer)
        self.K, self.V = [], []
        for i in range(self.L):
            plan = cfg.plan[i]
            rows = self.S if plan.is_sliding else self.max_len + self.C
            self.K.append(np.zeros((rows, plan.kv_dim), np.float32))
            self.V.append(np.zeros((rows, plan.kv_dim), np.float32))
        C = self.C
        self.z = np.zeros((C, self.vocab), np.float32)
        self.probs = np.zeros((C, self.vocab), np.float32)
        self.eos = np.asarray(self.gen_cfg.get("eos_token_id", [1, 106, 50]), np.int32)
        # the CPU's experts: one set of pinned rows for every layer and program (the layers run
        # one after the other)
        self.hp, self.po = pinned((C, self.H)), pinned((C, self.H))
        self.vp, self.ip = pinned((C, self.topk)), pinned((C, self.topk), np.int32)
        self.moe_scratch = cops.kq_moe_scratch(C, self.topk, self.E, self.H, self.inner)
        self.hq = np.zeros((C, self.H), np.int8)
        self.hs = np.zeros((C, self.H // 32), np.float32)
        self.hm = np.zeros((C, self.H // 16), np.float32)
        # the layers whose experts the GPU holds (DG_MOE): as many as fit after RESERVE (or
        # RECLANK_DG_GPU_LAYERS); the others' experts run on the CPU
        if gpu_layers is None:
            env = os.environ.get("RECLANK_DG_GPU_LAYERS")
            if env is not None:
                gpu_layers = int(env)
            else:
                from reclank.backends.cuda.gpumm import mem_info
                per = sum(a.nbytes for a in self.blk[0]["experts"])
                gpu_layers = max(0, min(self.L, int((mem_info()[0] - self.RESERVE) // per)))
        self.gpu_layers = set(range(int(gpu_layers)))
        print("dgemma gpu: the experts of %d layers on the GPU, %d on the CPU" % (
            len(self.gpu_layers), self.L - len(self.gpu_layers)), flush=True)
        self.mirror = Mirror()
        self.mirror.kind = "dgemma"
        self.progs = {}
        self.inv = {True: inv_freq_f32(cfg.head_dim, cfg.rope_parameters["sliding_attention"]["rope_theta"], 1.0),
                    False: inv_freq_f32(cfg.global_head_dim, cfg.rope_parameters["full_attention"]["rope_theta"],
                                        cfg.rope_parameters["full_attention"].get("partial_rotary_factor", 1.0))}
        self.reset()

    # ---- the cache -------------------------------------------------------------------------
    def reset(self):
        """An empty cache: n positions, the windowed buffers from position base."""
        self.n = 0
        self.base = 0

    def _room(self):
        """Before a block at n: the windowed layers' last window - 1 rows to the front when the
        block (and a canvas after it) would pass the end of their buffers."""
        if self.n - self.base + 2 * self.C <= self.S:
            return
        from reclank.encoders.runtime import _EncCompiler
        keep = self.window - 1
        src = self.n - self.base - keep
        c = _EncCompiler(self.eps)
        for i in range(self.L):
            if self.cfg.plan[i].is_sliding:
                for a in (self.K[i], self.V[i]):
                    c.p.emit(P.COPY, a[src:src + keep], a[:keep], a[:keep].nbytes)
        self._run_once(c.p.finish())
        self.base = self.n - keep

    def _run_once(self, prog):
        from reclank.backends.cuda.runtime import GPUProgram
        gp = GPUProgram(prog, graph=False, mirror=self.mirror)
        try:
            gp.run()
            gp.sync() if hasattr(gp, "sync") else None
        finally:
            gp.close()

    # ---- the programs ----------------------------------------------------------------------
    def _program(self, kind):
        from reclank.encoders.runtime import _EncCompiler
        C, H, L = self.C, self.H, self.L
        c = _EncCompiler(self.eps)
        c.cfg = self.cfg
        z = lambda *shape: np.zeros(shape, np.float32)   # noqa: E731
        i32 = lambda *shape: np.zeros(shape, np.int32)   # noqa: E731
        hs, hf = self.cfg.head_dim, self.cfg.global_head_dim
        c.env.update(x=z(C, H), cs=z(C, hs // 2), ss=z(C, hs // 2), cf=z(C, hf // 2), sf=z(C, hf // 2),
                     los=i32(C), his=i32(C), lof=i32(C), hif=i32(C), rows=i32(1), rowf=i32(1),
                     _scratch=z(C * max(H, 8192)))
        if kind != "enc":
            c.env.update(temp=z(1), ban=i32(C), u=z(C), ent=z(C), am=i32(C), draw=i32(C))
        e = c.env
        x = e["x"]
        if kind == "dec1":
            # the soft embedding of the last step's probabilities, then the self-conditioning MLP
            soft = P.k_enc_linear(c, self.table_t, self.probs)
            soft = P.k_enc_mul_vec(c, soft, np.full(H, self.embed_scale, np.float32))
            s = P.k_enc_rms(c, soft, self.sc["pre"], H)
            gm = P.k_enc_gelu_mul(c, P.k_enc_linear(c, self.sc["gate"], s), P.k_enc_linear(c, self.sc["up"], s))
            P.k_enc_add(c, x, P.k_enc_linear(c, self.sc["down"], gm))
        if kind != "enc":
            x = P.k_enc_rms(c, x, None, H)
        n_ev = 0
        cpu_progs = []
        for i, b in enumerate(self.blk):
            plan = b["plan"]
            hd, nh, kvh = plan.head_dim, plan.num_q_heads, plan.num_kv_heads
            cos, sin = (e["cs"], e["ss"]) if plan.is_sliding else (e["cf"], e["sf"])
            lo, hi = (e["los"], e["his"]) if plan.is_sliding else (e["lof"], e["hif"])
            row = e["rows"] if plan.is_sliding else e["rowf"]
            h = P.k_enc_rms(c, x, b["ln_in"], H)
            q = P.k_enc_rms(c, P.k_enc_linear(c, b["q"], h), b["qn"], hd)
            kr = P.k_enc_linear(c, b["k"], h)
            v = P.k_enc_rms(c, kr if b["v"] is None else P.k_enc_linear(c, b["v"], h), None, hd)
            k = P.k_enc_rms(c, kr, b["kn"], hd)
            P.k_enc_rope_tab(c, q, cos, sin, nh, hd, neox=True)
            P.k_enc_rope_tab(c, k, cos, sin, kvh, hd, neox=True)
            P.k_dg_kv_put(c, k, v, self.K[i], self.V[i], row)
            o = P.k_dg_attn(c, q, self.K[i], self.V[i], lo, hi, nh, kvh, hd)
            a = P.k_enc_rms(c, P.k_enc_linear(c, b["o"], o), b["post_attn"], H)
            P.k_enc_add(c, x, a)
            # the shared MLP and the experts, both on the residual x
            h = P.k_enc_rms(c, x, b["pre_ff"], H)
            gm = P.k_enc_gelu_mul(c, P.k_enc_linear(c, b["gate"], h), P.k_enc_linear(c, b["up"], h))
            h1 = P.k_enc_rms(c, P.k_enc_linear(c, b["down"], gm), b["post_ff1"], H)
            val, idx = c.buffer((C, self.topk)), c.buffer((C, self.topk), np.int32)
            c.p.emit(P.ROUTER_MT, x, b["r_scale"], b["r_proj"], b["r_pes"], H, self.E, self.topk,
                     float(self.eps), float(H ** -0.5), val, idx, C, c.buffer((C, H)), c.buffer((C, self.E)), 0)
            h2 = P.k_enc_rms(c, x, b["pre_ff2"], H)
            if i in self.gpu_layers:
                mo = P.k_dg_moe(c, h2, idx, val, b["experts"][0], b["experts"][1], self.E, self.inner)
            else:
                mo = self._moe_cpu(c, b, h2, val, idx, n_ev, cpu_progs)
                n_ev += 1
            h2 = P.k_enc_rms(c, mo, b["post_ff2"], H)
            P.k_enc_add(c, h1, h2)
            P.k_enc_add(c, x, P.k_enc_rms(c, h1, b["post_ff"], H))
            x = P.k_enc_mul_vec(c, x, b["enc_vec" if kind == "enc" else "dec_vec"], out=x)
        if kind != "enc":
            xn = P.k_enc_rms(c, x, self.norm, H)
            P.k_enc_linear(c, self.head, xn, out=self.z)
            c.p.emit(P.SOFTCAP, self.z, self.z.size, float(self.cfg.final_logit_softcapping))
            P.k_dg_sample(c, self.z, self.probs, e["temp"], e["ban"], self.eos, e["u"], e["ent"], e["am"], e["draw"])
        prog = c.p.finish()
        from reclank.ir import alloc
        alloc.pack(prog)     # the temporaries by liveness
        prog.cpu_progs = cpu_progs
        prog.keep.extend(cpu_progs)
        return prog

    def _moe_cpu(self, c, b, h2, val, idx, ev, cpu_progs):
        """The routed experts of a layer on the CPU: the rows to pinned memory, KQ_MOE, back."""
        C, H = self.C, self.H
        c.p.emit(P.TO_HOST, h2, self.hp, self.hp.nbytes, val, self.vp, self.vp.nbytes, idx, self.ip,
                 self.ip.nbytes, ev)
        cp = P.Program()
        cp.emit(P.KQ_MOE, self.hq, self.hs, self.hm, self.ip, self.vp, C, self.topk, self.E, b["mats"], None,
                H, self.inner, self.moe_scratch, self.po, None, 3 | 128, self.hp)
        cpu = cp.finish()
        cpu.keep.extend([b["mats"]] + list(b["experts"]))
        cpu_progs.append(cpu)
        c.p.emit(P.CPU_JOIN, cpu.buf, ev)
        out = c.buffer((C, H))
        c.p.emit(P.TO_DEV, self.po, out, self.po.nbytes)
        return out

    def _get(self, kind):
        g = self.progs.get(kind)
        if g is None:
            from reclank.backends.cuda.runtime import GPUProgram
            t0 = time.time()
            prog = self._program(kind)
            g = GPUProgram(prog, graph=True, mirror=self.mirror)
            g.prog_ = prog
            self.progs[kind] = g
            print("dgemma gpu: program %s %.1f s" % (kind, time.time() - t0), flush=True)
        return g

    # ---- the inputs of a block ----------------------------------------------------------------
    def _tables(self, pos):
        out = {}
        for tag, sl in (("s", True), ("f", False)):
            f = np.asarray(pos, np.float32)[:, None] * self.inv[sl][None, :]
            out["c" + tag], out["s" + tag] = np.cos(f), np.sin(f)
        return out

    def _set(self, g, d):
        prog = g.prog_
        for k, a in d.items():
            prog.names[k][...] = a
            g.upload(k)

    def embed(self, ids):
        return cops.kq_rows(self.table, BF12, self.H, ids) * self.embed_scale

    def encode(self, ids):
        """The encoder over ids at the end of the cache (blocks of C rows)."""
        ids = np.asarray(ids, np.int64)
        for a in range(0, len(ids), self.C):
            part = ids[a:a + self.C]
            m = len(part)
            assert self.n + self.C <= self.max_len, "the cache holds %d positions" % self.max_len
            self._room()
            blk = np.zeros(self.C, np.int64)
            blk[:m] = part
            n, base = self.n, self.base
            i = np.arange(self.C)
            pos = n + i
            d = dict(x=self.embed(blk), **self._tables(pos))
            d.update(los=np.maximum(0, pos - base - (self.window - 1)).astype(np.int32),
                     his=(pos - base + 1).astype(np.int32), lof=np.zeros(self.C, np.int32),
                     hif=(pos + 1).astype(np.int32), rows=np.array([n - base], np.int32),
                     rowf=np.array([n], np.int32))
            g = self._get("enc")
            self._set(g, d)
            g.run()
            self.n += m

    def steps(self):
        """A step function for a new canvas (sampler.denoise's step): the decoder at positions
        n..n + C - 1, then DG_SAMPLE; the first step dec0, the others dec1."""
        state = {"first": True}
        n, base, C = self.n, self.base, self.C
        assert n + C <= self.max_len + self.C
        self._room()
        n, base = self.n, self.base
        pos = n + np.arange(C)
        fixed = dict(**self._tables(pos))
        fixed.update(los=np.full(C, max(0, n - base - (self.window - 1)), np.int32),
                     his=np.full(C, n - base + C, np.int32), lof=np.zeros(C, np.int32),
                     hif=np.full(C, n + C, np.int32), rows=np.array([n - base], np.int32),
                     rowf=np.array([n], np.int32))

        def step(cur, temp, ban, u):
            kind = "dec0" if state["first"] else "dec1"
            g = self._get(kind)
            d = dict(fixed) if state["first"] or state.get("set") != kind else {}
            d.update(x=self.embed(np.asarray(cur, np.int64)), temp=np.array([temp], np.float32),
                     ban=np.asarray(ban, np.int32), u=np.asarray(u, np.float32))
            self._set(g, d)
            state["set"] = kind
            g.run()
            for k in ("ent", "am", "draw"):
                g.download(k)
            state["first"] = False
            pr = g.prog_.names
            return pr["ent"].copy(), pr["am"].copy(), pr["draw"].copy()
        return step
