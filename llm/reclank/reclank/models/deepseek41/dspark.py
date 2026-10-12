"""DSpark of DeepSeek V4.1 Flash on the GPU (DEEPSEEK_V41_PLAN.md Phase 5): the
draft of 5 tokens and the loop of the speculative decoding.

The drafter (the reference's DSparkBlock, model.py; mtp.0 .. mtp.2 of the
checkpoint): three blocks with the window attention only (128 positions, the
rope of the window layers) and 128 routed experts each (top 3, all on the
GPU: 7.2 GB); its input is main_x = main_norm(main_proj(the mean of the
streams at the inputs of layers 37, 38, 39)) of each main token: each block
puts kv_norm(wkv(main_x)) of the main tokens in its ring, then the block of 5
(the last token and 4 noise tokens, at the positions after it) attends to the
ring and to all 5 rows. The head: the main head on the norm of the last
hc_pre; the Markov head (rank 256) adds the bias of each draft token to the
logits of the next position, one after the other (greedy); the confidence
head gives the logit of the acceptance of each position.

The loop (ours; the reference implements the forward only): a draft from the
last token, a verify length L from the confidences (the expected accepted
tokens over the measured time of the draft and of a verify group of L + 1),
the verify group [y, d1 .. dL] on the main model (DSV41GPU.verify: the logits
of each row), the drafts that match the main model's choice accepted, its
choice after them the next token; the rows of the refused drafts undone
(DSV41GPU.restore). Greedy: the tokens are those of plain decoding (a row of a
verify group has the bits of a step).

    m = DSV41CPU(path); g = DSV41GPU(m, hot_gb=HOT_GB_DSPARK)
    ds = DSpark(g)
    out, stats = ds.generate(prompt_ids, 256)
"""
from __future__ import annotations

import time

import numpy as np

from reclank.backends.cpu import cops
from reclank.backends.cuda.gpumm import Buffer, pinned
from reclank.backends.cuda.runtime import GPUProgram, _check, lib
from reclank.ir import program as P
from reclank.models.deepseek41.gpu import VERIFY_MAX, _placeholder


class DSpark:
    """See the module text."""

    def __init__(self, g, where="cpu1"):
        """where: the routed experts of the blocks on the GPU ("gpu": 7.2 GB of
        its memory) or in the RAM of node 1, run there by a part of its CPUs
        ("cpu1": the GPU's memory for more hot experts of the main model)."""
        self.g = g
        self.where = where
        m = g.m
        c = m.c
        self.blocks = int(c["num_nextn_predict_layers"])
        self.bs = int(c["dspark_block_size"])
        self.noise = int(c["dspark_noise_token_id"])
        self.targets = list(c["dspark_target_layer_ids"])
        self.rank = int(c["dspark_markov_rank"])
        self.E = int(c["dspark_n_routed_experts"])
        self.k = int(c["dspark_num_experts_per_tok"])
        self.rings = [g.cache_rows(m.win, m.head_dim, "w") for _ in range(self.blocks)]
        for r in self.rings:
            g.mirror.add(r)
        self._experts()
        self.markov_embed = g._put(self._bf12_values("mtp.%d.markov_head.embed.weight" % (self.blocks - 1)))
        self.progs = {}
        self.scratch = {}           # the arrays of the seed programs (one at a time) share these
        g.on_grow.append(self.drop)
        # the times of a draft and of a verify group of t tokens (ms), measured
        # as the loop runs (moving averages); the first guesses from a step
        self.t_draft = 12.0
        self.t_verify = {t: 63.0 + 9.0 * (t - 1) for t in range(1, VERIFY_MAX + 1)}

    # ---- the weights
    def _bf12_values(self, name):
        """A bfloat16 matrix as the float32 values of its BF12 form (near)."""
        b = self.g.w.src.bf16(name)
        rows, _rep = cops.kq_bf16_to_bf12(b, b.shape[1])
        bits = cops.kq_bf12_to_bf16(rows, b.shape[0], b.shape[1])
        return (bits.astype(np.uint32) << 16).view(np.float32)

    def _experts(self):
        """The routed experts of the blocks, whole, in KQ_MXX: on the GPU (a
        store of E slots for each block, its map the identity), or in the RAM
        of node 1 with a part of its CPUs (ExpertParts of one part)."""
        g, src = self.g, self.g.w.src
        H, I = src.hidden, src.inner
        self.stores, self.maps = [], []
        if self.where == "cpu1":
            from reclank.runtime import moe_parts, numa
            node = moe_parts.nodes()[-1]
            cpus = [moe_parts.part_cpus()[-1]]
            desc = {}
            self.ram = []
            for b in range(self.blocks):
                arr = numa.empty_on((self.E, g.per), np.uint8, node, strict=True)
                for e in range(self.E):
                    ex = src.expert(b, e, prefix="mtp")
                    arr[e, :g.off3] = cops.kq_mxx_pack(*ex["w1"], I, H)[0].reshape(-1)
                    arr[e, g.off3:g.off2] = cops.kq_mxx_pack(*ex["w3"], I, H)[0].reshape(-1)
                    arr[e, g.off2:] = cops.kq_mxx_pack(*ex["w2"], H, I)[0].reshape(-1)
                self.ram.append(arr)
                base = arr.ctypes.data
                desc[b] = [[base, cops.KQ_MXX, g.per, 0, base + g.off3, cops.KQ_MXX, g.per, 0,
                            base + g.off2, cops.KQ_MXX, g.per, 0]]
            self.parts = moe_parts.ExpertParts.from_desc(desc, [(0, I)], [(0, H)], self.E, H, I, cpus,
                                                         part_nodes=[node])
            self.part_bufs = {}
            return
        for b in range(self.blocks):
            ph = _placeholder(self.E * g.per)
            g.mirror.add(ph)
            buf = Buffer(ph.nbytes, "program", pinned=True)
            g.mirror.bufs[ph.ctypes.data] = buf
            for e in range(self.E):
                ex = src.expert(b, e, prefix="mtp")
                w1, _ = cops.kq_mxx_pack(*ex["w1"], I, H)
                w3, _ = cops.kq_mxx_pack(*ex["w3"], I, H)
                w2, _ = cops.kq_mxx_pack(*ex["w2"], H, I)
                host = np.concatenate([w1.reshape(-1), w3.reshape(-1), w2.reshape(-1)])
                assert host.nbytes == g.per
                _check(lib().rc_gpu_h2d(buf.ptr + e * g.per, host.ctypes.data, g.per))
            self.stores.append(ph)
            self.maps.append(np.arange(self.E, dtype=np.int32))

    # ---- the programs
    def _compile(self, tm, draft):
        """tm main rows into the rings (positions mpos ..), then (draft) the
        block of bs at pos = mpos + tm."""
        g, m = self.g, self.g.m
        prog = P.Program()
        dim, hc, hd, rd, H, win = m.dim, m.hc, m.head_dim, m.rd, m.heads, m.win
        eps, hc_eps, iters = m.eps, m.hc_eps, m.sink_iters
        I, bs, k = g.w.src.inner, self.bs, self.k
        own = []                        # the arrays of this program alone (drop forgets them)

        def z(*sh):
            own.append(np.zeros(sh, np.float32))
            return own[-1]

        # the arrays of tm rows of a seed (a group of a prompt): views of the
        # shared scratch (the programs run one at a time), made for the window
        # with the bounded replay (its decoder's group alone seeds), else the group
        counts = {}
        most = m.win if g.replay else max(g.group, m.win)

        def zt(*sh):
            if tm > VERIFY_MAX:
                return g._scratch(tm, counts, sh, np.float32, False, self.scratch, most)
            return z(*sh)

        pos, mpos, seq = prog.slot("pos"), prog.slot("mpos"), prog.slot("seq")
        cw = g.tables[0]                # no compression: the rope of the window layers
        io = _DRun()
        io.own = own
        io.main_h = zt(tm, len(self.targets) * dim)
        prog.names["main_h"] = io.main_h
        mx = zt(tm, dim)

        def lin8(xb, name, out, groups=0):
            c, s, rows, cols = g.fp8(name)
            prog.emit(P.DS_FP8_LINEAR, xb, c, s, out, rows, cols, xb.shape[0], groups)

        def lin12(xb, name, out):
            g.emit_dense16(prog, xb, name, out)

        mp = zt(tm, dim)
        lin8(io.main_h, "mtp.0.main_proj.weight", mp)
        prog.emit(P.RMS_NORM, mp, g.vec("mtp.0.main_norm.weight"), mx, tm, dim, eps)
        kvr, kvn, kvq = zt(tm, hd), zt(tm, hd), zt(tm, hd)
        for b in range(self.blocks):
            a = "mtp.%d.attn." % b
            lin8(mx, a + "wkv.weight", kvr)
            prog.emit(P.RMS_NORM, kvr, g.vec(a + "kv_norm.weight"), kvn, tm, hd, eps)
            prog.emit(P.DS_WIN_WRITE, kvn, cw[0], cw[1], kvq, mpos, tm, win, hd, rd)
            prog.emit(P.DS_RING_PUT, kvq, self.rings[b], mpos, tm, win, hd, None, g.cform)
        if draft:
            io.h = h = z(bs, hc, dim)
            prog.names["h"] = h
            io.ids = np.zeros(bs + 1, np.int32)
            prog.names["ids"] = io.ids
            io.conf = z(bs)
            prog.names["conf"] = io.conf
            qrank = int(m.c["q_lora_rank"])
            x, xa, qa, qr, q = z(bs, dim), z(bs, dim), z(bs, qrank), z(bs, qrank), z(bs, H * hd)
            bkr, bkn, bkq = z(bs, hd), z(bs, hd), z(bs, hd)
            o, oa, y = z(bs, H * hd), z(bs, m.groups * m.o_rank), z(bs, dim)
            lr, sg, su, sa, ys, part = z(bs, self.E), z(bs, I), z(bs, I), z(bs, I), z(bs, dim), z(bs, dim)
            ids_r, vals_r = np.zeros((bs, k), np.int32), z(bs, k)
            hq = dict(qh=np.zeros((bs, dim), np.int8), ql=np.zeros((bs, dim), np.int8), xs=z(bs, dim // 32),
                      act=z(bs * k, 2 * I), act2=z(bs * k, I), aqh=np.zeros((bs * k, I), np.int8),
                      aql=np.zeros((bs * k, I), np.int8), axs=z(bs * k, I // 32), de=z(bs * k, dim))
            rows_dummy = pinned((1, dim))
            prog.keep.append(rows_dummy)
            hp, vp, ip, host_out = pinned((bs, dim)), pinned((bs * k,)), pinned((bs * k,), np.int32), pinned((bs, dim))
            prog.keep += [hp, vp, ip, host_out]
            prev = None
            for b in range(self.blocks):
                p = "mtp.%d." % b
                a = p + "attn."
                fa, sca, ba = g.hcp(p + "hc_attn")
                raw_a = z(bs, 24)
                prog.emit(P.DS_HC_MIX, h, fa, raw_a, bs, hc * dim, eps)
                pm = prev if prev is not None else (None, None, None)
                prog.emit(P.DS_HC_PRE, h, pm[0], pm[1], pm[2], g.vec(p + "attn_norm.weight"), x, bs, dim, eps, hc_eps)
                lin8(x, a + "wq_a.weight", qa)
                prog.emit(P.RMS_NORM, qa, g.vec(a + "q_norm.weight"), qr, bs, qrank, eps)
                lin8(qr, a + "wq_b.weight", q)
                prog.emit(P.DS_ROPE, q, cw[0], cw[1], pos, bs, H, hd, rd, 0, 0)
                lin8(x, a + "wkv.weight", bkr)
                prog.emit(P.RMS_NORM, bkr, g.vec(a + "kv_norm.weight"), bkn, bs, hd, eps)
                prog.emit(P.DS_WIN_WRITE, bkn, cw[0], cw[1], bkq, pos, bs, win, hd, rd)
                prog.emit(P.DS_ATTN, q, self.rings[b], pos, win, None, None, None, g.vec(a + "attn_sink"), o, H, hd,
                          float(np.float32(hd ** -0.5)), m.ix_topk, bkq, bs, None, 1, g.cform)
                prog.emit(P.DS_ROPE, o, cw[0], cw[1], pos, bs, H, hd, rd, 1, 0)
                lin8(o, a + "wo_a.weight", oa, groups=m.groups)
                lin8(oa, a + "wo_b.weight", y)
                prog.emit(P.DS_HC_POST, y, None, h, raw_a, sca, ba, h, bs, dim, hc_eps, iters)
                ff, scf, bf = g.hcp(p + "hc_ffn")
                raw_f = z(bs, 24)
                prog.emit(P.DS_HC_MIX, h, ff, raw_f, bs, hc * dim, eps)
                prog.emit(P.DS_HC_PRE, h, raw_a, sca, ba, g.vec(p + "ffn_norm.weight"), xa, bs, dim, eps, hc_eps)
                f = p + "ffn."
                lin12(xa, f + "gate.weight", lr)
                prog.emit(P.DS_ROUTER, lr, g.vec(f + "gate.bias"), ids_r, vals_r, bs, self.E, k, float(m.route_scale))
                sx = f + "shared_experts."
                if self.where == "cpu1":
                    # the routed experts in the part of node 1 while the GPU does the shared one
                    fl = pinned((2,), np.int64)
                    fl[:] = 0
                    prog.keep.append(fl)
                    prog.emit(P.SIGNAL, fl[0:1], seq, xa, hp, hp.nbytes, vals_r, vp, vp.nbytes, ids_r, ip, ip.nbytes)
                    lin8(xa, sx + "w1.weight", sg)
                    lin8(xa, sx + "w3.weight", su)
                    prog.emit(P.DS_SWIGLU, sg, su, sa, bs * I, float(m.limit))
                    lin8(sa, sx + "w2.weight", ys)
                    cpu = self.parts.program(b, bs, k, hp, ip, vp, host_out, None, 32 | 128, self.part_bufs)
                    io.cpu.append(cpu)
                    prog.emit(P.CPU_TASK, cpu.buf, fl[0:1], fl[1:2], seq)
                    prog.emit(P.AWAIT, fl[1:2], seq, host_out, part, part.nbytes, None)
                else:
                    prog.emit(P.DS_HOT_MOE, xa, ids_r, vals_r, self.maps[b], self.stores[b], g.per, g.off3, g.off2,
                              bs, k, dim, I, hq["qh"], hq["ql"], hq["xs"], hq["act"], hq["act2"], hq["aqh"],
                              hq["aql"], hq["axs"], hq["de"])
                    prog.emit(P.DS_COMBINE, ids_r, self.maps[b], hq["de"], rows_dummy, part, k, dim, bs)
                    lin8(xa, sx + "w1.weight", sg)
                    lin8(xa, sx + "w3.weight", su)
                    prog.emit(P.DS_SWIGLU, sg, su, sa, bs * I, float(m.limit))
                    lin8(sa, sx + "w2.weight", ys)
                prog.emit(P.DS_HC_POST, part, ys, h, raw_f, scf, bf, h, bs, dim, hc_eps, iters)
                prev = (raw_f, scf, bf)
            last = "mtp.%d." % (self.blocks - 1)
            xh, xn = z(bs, dim), z(bs, dim)
            prog.emit(P.DS_HC_PRE, h, prev[0], prev[1], prev[2], None, xh, bs, dim, eps, hc_eps)
            prog.emit(P.RMS_NORM, xh, g.vec(last + "norm.weight"), xn, bs, dim, eps)
            V = m.w.src.cfg["vocab_size"]
            io.logits = z(bs, V)
            lin12(xn, "head.weight", io.logits)
            me, bias = z(bs, self.rank), z(1, V)
            for i in range(bs):
                prog.emit(P.DS_GATHER, self.markov_embed, io.ids, i, me[i], self.rank)
                lin12(me[i:i + 1], last + "markov_head.head.weight", bias)
                prog.emit(P.DS_MARKOV, io.logits, bias, io.ids, i, V)
            cwt = np.ascontiguousarray(g.w.src.f32(last + "confidence_head.proj.weight").reshape(-1), np.float32)
            prog.emit(P.DS_CONF, xh, me, cwt, io.conf, bs, dim, self.rank)
        prog.keep += self.rings
        prog.finish()
        io.prog = prog
        io.g = GPUProgram(prog, graph=g.graph, mirror=g.mirror)
        return io

    def drop(self):
        """The programs freed with the device copies of their own arrays (the
        main model's cap grew: they read its rope tables)."""
        for r in self.progs.values():
            r.g.close()
            for a in r.own:
                self.g.mirror.forget(a.ctypes.data)
        for a in self.scratch.values():
            self.g.mirror.forget(a.ctypes.data)
        self.progs, self.scratch = {}, {}

    def warm(self):
        """The programs of the loop made and their graphs recorded before it
        runs (the verify groups and the drafts of 1 to VERIFY_MAX rows): a
        first use costs about 300 ms, which the measured rates of "auto" took
        for the cost of DSpark (it then chose plain steps)."""
        for t in range(1, VERIFY_MAX + 1):
            self.g.run(t).g.prepare()
            self.run(t, True).g.prepare()

    def run(self, tm, draft):
        key = (tm, draft)
        if key not in self.progs:
            self.progs[key] = self._compile(tm, draft)
        return self.progs[key]

    def reset(self):
        for r in self.rings:
            r[:] = 0
            self.g.mirror.buffer_of(r).upload(r)

    def seed(self, main_h, mpos):
        """The main rows of a prompt group into the rings (no draft)."""
        r = self.run(main_h.shape[0], False)
        r.main_h[:] = main_h
        self.g.mirror.buffer_of(r.main_h).upload(r.main_h)
        r.g.bind({"mpos": int(mpos), "pos": int(mpos + main_h.shape[0])})
        r.g.run()

    def draft(self, y, main_h, mpos):
        """The main rows into the rings, then the draft after y (at mpos +
        len(main_h)): the 5 draft tokens and the logits of their confidence."""
        g = self.g
        r = self.run(main_h.shape[0], True)
        r.main_h[:] = main_h
        g.mirror.buffer_of(r.main_h).upload(r.main_h)
        tokens = np.full(self.bs, self.noise, np.int64)
        tokens[0] = y
        r.h[:] = np.repeat(g.w.embed_rows(tokens)[:, None, :], g.m.hc, 1)
        g.mirror.buffer_of(r.h).upload(r.h)
        r.ids[:] = 0
        r.ids[0] = y
        g.mirror.buffer_of(r.ids).upload(r.ids)
        r.g.bind({"mpos": int(mpos), "pos": int(mpos + main_h.shape[0])})
        r.g.run()
        g.download(r.ids)
        g.download(r.conf)
        return r.ids[1:].copy(), r.conf.copy()

    # ---- the loop
    def choose(self, conf):
        """The verify length: the most expected tokens a millisecond, from the
        survival of each prefix (the product of the sigmoids of the
        confidences) and the measured times."""
        p = 1.0 / (1.0 + np.exp(-conf.astype(np.float64)))
        surv = np.cumprod(p)
        best, bl = 1.0 / (self.t_draft + self.t_verify[1]), 0
        for L in range(1, len(conf) + 1):
            if L + 1 > VERIFY_MAX:
                break
            rate = (1.0 + surv[:L].sum()) / (self.t_draft + self.t_verify[L + 1])
            if rate > best:
                best, bl = rate, L
        return bl

    def prefill(self, prompt, group=None):
        """The prompt on the main model in groups, each group's rows seeding the
        rings; the logits of its last token and the input of DSpark of it."""
        g, last = self.g, {}

        def each(r, at, n):
            last["h"] = g.download(r.main_h)
            self.seed(last["h"], at)

        lg = g.prefill(prompt, 0, group, each)
        return lg, last["h"][-1:].copy()

    def generate(self, prompt, n, spec="auto", fixed=None, on_token=None, probe=8):
        """n tokens after the prompt (greedy). spec False: plain steps (the
        check of the bits); True: a draft every round; "auto": a draft only
        while the rounds of DSpark give more tokens a second than plain steps
        (both rates measured as it runs; every probe-th step tries the other).
        fixed: always that many drafts. Returns the tokens and the stats."""
        g = self.g
        g.reset()
        self.reset()
        lg, main_last = self.prefill(prompt)
        self.warm()
        y, pos = int(np.argmax(lg)), len(prompt)
        out = []
        stats = dict(rounds=0, plain=0, drafted=0, accepted=0, lengths=[], conf=[])
        main_rows, mpos = main_last, pos - 1       # the main rows the rings lack
        rate_spec = rate_plain = None              # tokens a ms (moving averages)
        t_start = time.time()
        it = 0
        while len(out) < n:
            it += 1
            use = bool(spec)
            if spec == "auto" and rate_spec is not None and rate_plain is not None:
                use = rate_spec > rate_plain
                if it % probe == 0:
                    use = not use
            elif spec == "auto" and rate_plain is None and rate_spec is not None:
                use = False
            t_round = time.time()
            if not use:
                if spec is False:
                    out.append(y)
                    lg = g.step(y, pos)
                    y, pos = int(np.argmax(lg)), pos + 1
                    continue
                # a plain step that keeps the input of DSpark (the rings stay whole)
                logits, main_h = g.verify([y], pos)
                if main_rows is not None and len(main_rows):
                    self.seed(main_rows, mpos)
                main_rows, mpos = main_h[:1], pos
                out.append(y)
                if on_token is not None:
                    on_token(y)
                y, pos = int(np.argmax(logits[0])), pos + 1
                stats["plain"] += 1
                dt = (time.time() - t_round) * 1e3
                rate_plain = 1.0 / dt if rate_plain is None else 0.8 * rate_plain + 0.2 / dt
                continue
            t0 = time.time()
            drafts, conf = self.draft(y, main_rows, mpos)
            self.t_draft = 0.9 * self.t_draft + 0.1 * (time.time() - t0) * 1e3
            L = self.choose(conf) if fixed is None else fixed
            t0 = time.time()
            logits, main_h = g.verify([y] + [int(d) for d in drafts[:L]], pos)
            t = L + 1
            self.t_verify[t] = 0.8 * self.t_verify[t] + 0.2 * (time.time() - t0) * 1e3
            choice = np.argmax(logits, -1)
            a = 0
            while a < L and int(choice[a]) == int(drafts[a]):
                a += 1
            emitted = [y] + [int(d) for d in drafts[:a]]
            if a < L:
                g.restore(t, pos, a + 1)
            out += emitted
            if on_token is not None:
                for tok in emitted:
                    on_token(tok)
            stats["rounds"] += 1
            stats["drafted"] += L
            stats["accepted"] += a
            stats["lengths"].append(L)
            stats["conf"].append((conf.copy(), [int(choice[i]) == int(drafts[i]) for i in range(min(L, len(drafts)))]))
            main_rows, mpos = main_h[:a + 1], pos
            y, pos = int(choice[a]), pos + a + 1
            r = len(emitted) / ((time.time() - t_round) * 1e3)
            rate_spec = r if rate_spec is None else 0.8 * rate_spec + 0.2 * r
        stats["seconds"] = time.time() - t_start
        stats["rate_spec"], stats["rate_plain"] = rate_spec, rate_plain
        return out[:n], stats


class _DRun:
    """A DSpark program: its Program, GPUProgram, inputs and outputs."""

    def __init__(self):
        self.cpu = []
