#!/usr/bin/env python3
"""The reference of GLM-5.3-Flash (glm5_next) on the CPU, from the original FP8 checkpoint.

The transformers code (models/glm5_next, Glm5NextTextModel: the text model alone, no vision tower,
no MTP layer 45) runs as it is, in float32 on the CPU, one prompt at a time with its cache, and the
untied lm_head after it. The weights come from the shards (model.language_model.* -> *), each FP8
matrix (E4M3, 128 x 128 blocks of weight_scale_inv) dequantized to float32 as it is read:

- every weight but the routed experts once at the start (float32); q/k/v_conv1d concatenated into
  conv1d, f_a/f_b/dt_bias/A_log into forget_gate, hc_{attn,ffn}_* into {attn,ffn}_hc, as the
  conversion mapping of transformers does;
- the routed experts (experts.N.{gate,up,down}_proj, each its own FP8 tensor) as the router picks
  them: the forward of Glm5NextTextExperts is the eager loop of transformers, with each expert's
  gate_up and down dequantized from an LRU of FP8 experts in memory (--expert-cache-gb), the misses
  of a layer read in the order of the shards.

Reads of the shards drop their pages (POSIX_FADV_DONTNEED).

Each prompt runs as one prompt pass, then --steps greedy decode steps. For each prompt it writes
<out>/<name>.npz:

    ids, prompt          the prompt (the original chat template, the HF tokenizer)
    gen                  the greedy tokens (to EOS or --steps)
    top_ids, top_vals    [T + len(gen) - 1, 64]: the top 64 logits of each row
    lse, argmax          [T + len(gen) - 1]: the log-sum-exp and the argmax
    step_logits          [1 + min(16, len(gen) - 1), vocab]: the full logits of the last prompt row
                         and of the first 16 decode rows

Row r of top_ids / lse / argmax is the prediction for token r + 1 of ids + gen. The rows of gen run
as decode steps, the rows of ids as a prompt.

--dump NAME: the prompt pass of that prompt also writes <out>/<name>_layers.npz: embed [T, 4096],
h<i> [T, 4, 4096] (the streams after layer i), topk<i> / topw<i> [T, 8] (the router's picks of
MoE layer i) and norm [T, 4096] (the input of lm_head).

--trace-prefill N --trace-decode M (first, before the prompts): a prompt of exactly N tokens (a
design document cut to fit the template), then M greedy decode steps (EOS does not stop them), every
value of every layer for all N + M rows in <out>/trace_<N>_<M>.npz: ids [N], gen [M + 1], embed,
for each layer i attn_in<i> / attn_out<i> [rows, 4096] (the input of the attention block after the
hc collapse and the norm, its output), mid<i> [rows, 4, 4096] (the streams after the attention),
mlp_in<i> / mlp_out<i>, h<i> (the streams after the layer), topk<i> / topw<i> of the MoE layers,
norm, and logits [rows, vocab] (all float32).

--forced FILE.npz (ids, gen: the tokens of another run): the reference teacher-forced on ids + gen, in
chunks of 256 rows; <out>/<name of FILE>_ref.npz gets top_ids, top_vals, lse, argmax of every row.

    taskset -c 0-47 python -I scripts/glm53_reference.py /space/models/GLM-5.3-Flash \\
        --out ~/reclank-golden/glm53 --threads 40 --dump think
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import OrderedDict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
N_TOP = 64
N_FULL = 16
MAIN = "model.language_model."
B = 128                                             # the FP8 block


def log(msg):
    print("ref: " + msg, file=sys.stderr, flush=True)


def trace_prompt(tok, n):
    """The ids of a templated prompt of exactly n tokens: a design document, its tokens cut to fit."""
    doc = open(os.path.join(ROOT, "docs", "MOE_PLAN.md")).read()
    mark = "\u0000DOC\u0000"
    text = tok.apply_chat_template([{"role": "user", "content": "Here is a design document:\n\n" + mark +
                                     "\n\nExplain in detail what it plans and why."}],
                                   add_generation_prompt=True, tokenize=False)
    pre, suf = text.split(mark)
    ip, isf = (tok(x, add_special_tokens=False)["input_ids"] for x in (pre, suf))
    idoc = tok(doc, add_special_tokens=False)["input_ids"]
    k = n - len(ip) - len(isf)
    assert 0 < k <= len(idoc), (n, len(ip), len(isf))
    return ip + idoc[:k] + isf


def prompts():
    """(name, messages): real text of about 200 tokens, and a short question."""
    doc = open(os.path.join(ROOT, "docs", "MOE_PLAN.md")).read()[:500]
    doc = doc[:doc.rindex("\n")]
    u = lambda t: [{"role": "user", "content": t}]  # noqa: E731
    return [
        ("doc", u("Here is the start of a design document:\n\n" + doc + "\n\nSummarize it in two sentences.")),
        ("think", u("List the first ten prime numbers, separated by commas.")),
    ]


class Shards:
    """name -> (path, dtype, shape, start, end) of the shards; reads that drop their pages."""

    def __init__(self, path):
        where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.t = {}
        self.fds = {}
        for f in sorted(set(where.values())):
            p = os.path.join(path, f)
            with open(p, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            hdr.pop("__metadata__", None)
            for k, v in hdr.items():
                s, e = v["data_offsets"]
                self.t[k] = (p, v["dtype"], tuple(v["shape"]), 8 + n + s, 8 + n + e)
        self.read_bytes = 0

    def raw(self, name):
        p, dt, shape, off, end = self.t[name]
        n = end - off
        if p not in self.fds:
            self.fds[p] = os.open(p, os.O_RDONLY)
        fd = self.fds[p]
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            got += os.preadv(fd, [mv[got:]], off + got)
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
        self.read_bytes += n
        return buf

    def f32(self, name):
        """A tensor in float32; an FP8 matrix times its block scales (name.weight_scale_inv)."""
        import torch
        p, dt, shape, s, e = self.t[name]
        b = self.raw(name)
        if dt == "F32":
            return np.frombuffer(b, np.float32).reshape(shape).copy()
        if dt == "BF16":
            return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)
        assert dt == "F8_E4M3", (name, dt)
        return fp8(b, self.f32(name + "_scale_inv"), shape)


def fp8(raw, scale, shape):
    """E4M3 bytes [R, C] times the scale of each 128 x 128 block [R / 128, C / 128], float32."""
    import torch
    R, C = shape
    assert R % B == 0 and C % B == 0 and scale.shape == (R // B, C // B), (shape, scale.shape)
    w = torch.frombuffer(raw, dtype=torch.uint8).view(torch.float8_e4m3fn).float().view(R // B, B, C // B, B)
    return (w * torch.from_numpy(np.ascontiguousarray(scale, np.float32))[:, None, :, None]).view(R, C).numpy()


class Experts:
    """The routed experts: an LRU of FP8 experts (the bytes and scales of gate, up and down), the
    misses of a layer read in the order of the shards."""

    PARTS = ("gate_proj", "up_proj", "down_proj")

    def __init__(self, sh, cfg, cache_gb):
        self.sh = sh
        per = 3 * cfg.moe_intermediate_size * cfg.hidden_size
        self.cap = int(cache_gb * 1e9 / per)
        self.lru = OrderedDict()
        self.loads = 0
        self.hits = 0

    def name(self, layer, e, part):
        return MAIN + "layers.%d.mlp.experts.%d.%s.weight" % (layer, e, part)

    def need(self, layer, experts):
        """Every expert of the list in the LRU (most recent last)."""
        miss = []
        for e in experts:
            key = (layer, e)
            if key in self.lru:
                self.lru.move_to_end(key)
                self.hits += 1
            else:
                miss.append(e)
        where = lambda e: self.sh.t[self.name(layer, e, "gate_proj")][::3]  # noqa: E731  (path, start)
        for e in sorted(miss, key=where):
            w = []
            for part in self.PARTS:
                n = self.name(layer, e, part)
                w.append((self.sh.raw(n), self.sh.f32(n + "_scale_inv"), self.sh.t[n][2]))
            self.lru[(layer, e)] = w
            self.loads += 1
            while len(self.lru) > self.cap:
                self.lru.popitem(last=False)

    def weights(self, layer, e):
        """(gate_up [2 I, H], down [H, I]) of an expert in float32, torch."""
        import torch
        g, u, d = (fp8(*x) for x in self.lru[(layer, e)])
        return torch.from_numpy(np.concatenate([g, u])), torch.from_numpy(d)


def experts_forward(mod, ex, layer):
    """The forward of Glm5NextTextExperts (the eager loop of transformers) with the weights of each
    expert from Experts."""
    import torch
    import torch.nn.functional as F

    def forward(hidden_states, top_k_index, top_k_weights):
        final = torch.zeros_like(hidden_states)
        hit = sorted(top_k_index.unique().tolist())
        ex.need(layer, hit)
        for e in hit:
            token_idx, top_k_pos = torch.where(top_k_index == e)
            gu, dn = ex.weights(layer, e)
            cur = mod._apply_gate(F.linear(hidden_states[token_idx], gu))
            cur = F.linear(cur, dn) * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, cur.to(final.dtype))
        return final
    return forward


def source(n):
    """The name in the shards of a parameter of Glm5NextTextModel (None: conv1d, built from three)."""
    for a, b in ((".self_attn.forget_gate.", ".self_attn."), (".attn_hc.", ".hc_attn_"), (".ffn_hc.", ".hc_ffn_")):
        if a in n:
            n = n.replace(a, b)
    return MAIN + n


def build(model_dir, sh, expert_cache_gb, layers=None):
    """The text model on the meta device, then every weight but the routed experts from the shards
    (float32), the experts through Experts; the lm_head (float32, torch)."""
    import torch
    from torch import nn
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextModel

    cfg = AutoConfig.from_pretrained(model_dir).text_config
    cfg.dtype = torch.float32
    cfg._attn_implementation = "eager"
    cfg._experts_implementation = "eager"
    if layers:
        # (a test of the script) the first layers only
        cfg.num_hidden_layers = layers
        for k in ("layer_types", "mlp_layer_types", "indexer_types"):
            setattr(cfg, k, getattr(cfg, k)[:layers])
    with torch.device("meta"):
        model = Glm5NextTextModel._from_config(cfg, dtype=torch.float32)
    model.eval()
    t0 = time.time()
    used = set()
    for n, _ in list(model.named_parameters()) + list(model.named_buffers()):
        if ".mlp.experts." in n:
            continue
        path, leaf = n.rsplit(".", 1)
        if n.endswith(".self_attn.conv1d.weight"):
            pre = MAIN + path.rsplit(".", 1)[0] + "."
            srcs = [pre + c + "_conv1d.weight" for c in "qkv"]
            w = np.concatenate([sh.f32(s) for s in srcs])
        else:
            srcs = [source(n)]
            assert srcs[0] in sh.t, (n, srcs[0])
            w = sh.f32(srcs[0])
        mod = model.get_submodule(path)
        t = torch.from_numpy(np.ascontiguousarray(w))
        assert tuple(t.shape) == tuple(getattr(mod, leaf).shape), (n, t.shape, getattr(mod, leaf).shape)
        if leaf in mod._parameters:
            mod._parameters[leaf] = nn.Parameter(t, requires_grad=False)
        else:
            mod._buffers[leaf] = t
        used.update(srcs)
        used.update(s + "_scale_inv" for s in srcs if s + "_scale_inv" in sh.t)
    lm_head = torch.from_numpy(sh.f32("lm_head.weight"))
    used.add("lm_head.weight")
    left = [n for n in sh.t if n not in used and not n.startswith(("model.visual.", MAIN + "layers.45."))
            and ".mlp.experts." not in n and ".shared_head." not in n]
    assert layers or not left, left[:5]
    ex = Experts(sh, cfg, expert_cache_gb)
    for i, layer in enumerate(model.layers):
        if cfg.mlp_layer_types[i] == "sparse":
            e = layer.mlp.experts
            e.forward = experts_forward(e, ex, i)
            del e._parameters["gate_up_proj"], e._parameters["down_proj"]
    log("dense weights in %.0f s (%.1f GB read, %.1f GB float32)" % (
        time.time() - t0, sh.read_bytes / 1e9,
        4 * (sum(p.numel() for p in model.parameters()) + lm_head.numel()) / 1e9))
    still = [n for n, b in model.named_buffers() if b.is_meta] + [n for n, p in model.named_parameters() if p.is_meta]
    assert not still, still[:5]
    return model, lm_head, ex


def check_fp8(sh):
    """The dequantized FP8 of a few matrices against the HF quantizer's formula on blocks (torch)."""
    import torch
    for n in (MAIN + "layers.0.mlp.gate_proj.weight", MAIN + "layers.3.self_attn.q_b_proj.weight",
              MAIN + "layers.3.mlp.experts.7.down_proj.weight"):
        p, dt, shape, s, e = sh.t[n]
        got = sh.f32(n)
        w = torch.frombuffer(sh.raw(n), dtype=torch.uint8).view(torch.float8_e4m3fn).float().reshape(shape)
        sc = torch.from_numpy(sh.f32(n + "_scale_inv"))
        want = w * sc.repeat_interleave(B, 0).repeat_interleave(B, 1)
        assert np.array_equal(got, want.numpy()), n
    log("FP8 dequantization: 3 matrices match the per-block formula")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--only", default=None, help="names of prompts, comma separated")
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--expert-cache-gb", type=float, default=260.0)
    ap.add_argument("--layers", type=int, default=None, help="(a test of the script) only the first N layers")
    ap.add_argument("--dump", default=None, help="the prompt whose prompt pass writes the layers' states")
    ap.add_argument("--trace-prefill", type=int, default=0)
    ap.add_argument("--trace-decode", type=int, default=256)
    ap.add_argument("--forced", default=None, help="an npz of ids and gen: the logits of those tokens")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    gcfg = json.load(open(os.path.join(a.model, "generation_config.json")))
    eos = set(gcfg["eos_token_id"] if isinstance(gcfg["eos_token_id"], list) else [gcfg["eos_token_id"]])
    sh = Shards(a.model)
    check_fp8(sh)
    model, lm_head, ex = build(a.model, sh, a.expert_cache_gb, a.layers)
    cfg = model.config

    dump = {}
    hooks = []

    def record(on):
        for h in hooks:
            h.remove()
        hooks.clear()
        if not on:
            return
        hooks.append(model.embed_tokens.register_forward_hook(
            lambda m, i, o: dump.__setitem__("embed", o[0].numpy().copy())))
        hooks.append(model.norm.register_forward_hook(
            lambda m, i, o: dump.__setitem__("norm", o[0].numpy().copy())))
        for i, layer in enumerate(model.layers):
            hooks.append(layer.register_forward_hook(
                lambda m, args, o, i=i: dump.__setitem__("h%d" % i, o[0][0].numpy().copy())))
            if cfg.mlp_layer_types[i] == "sparse":
                def gate(m, args, o, i=i):
                    dump["topw%d" % i] = o[1].numpy().copy()
                    dump["topk%d" % i] = o[2].numpy().astype(np.int16)
                hooks.append(layer.mlp.gate.register_forward_hook(gate))

    def run(x, cache):
        h = model(input_ids=torch.tensor([x]), past_key_values=cache, use_cache=True).last_hidden_state
        return torch.nn.functional.linear(h[0], lm_head).float()

    def rows(lg, rec):
        tv, ti = torch.topk(lg, N_TOP, dim=-1)
        rec["top_ids"].append(ti.numpy().astype(np.int32))
        rec["top_vals"].append(tv.numpy())
        rec["lse"].append(torch.logsumexp(lg, dim=-1).numpy())
        rec["argmax"].append(lg.argmax(-1).numpy().astype(np.int32))

    if a.trace_prefill:
        rec = {}

        def put(k, v):
            rec.setdefault(k, []).append(np.array(v, np.float32))
        hk = [model.embed_tokens.register_forward_hook(lambda m, i, o: put("embed", o[0])),
              model.norm.register_forward_hook(lambda m, i, o: put("norm", o[0]))]
        for i, layer in enumerate(model.layers):
            hk.append(layer.register_forward_hook(lambda m, args, o, i=i: put("h%d" % i, o[0][0])))
            hk.append(layer.ffn_hc.register_forward_pre_hook(lambda m, args, i=i: put("mid%d" % i, args[0][0])))
            hk.append(layer.self_attn.register_forward_pre_hook(
                lambda m, args, kw, i=i: put("attn_in%d" % i, kw["hidden_states"][0]), with_kwargs=True))
            hk.append(layer.self_attn.register_forward_hook(
                lambda m, args, o, i=i: put("attn_out%d" % i, (o[0] if isinstance(o, tuple) else o)[0])))
            hk.append(layer.mlp.register_forward_pre_hook(lambda m, args, i=i: put("mlp_in%d" % i, args[0][0])))
            hk.append(layer.mlp.register_forward_hook(lambda m, args, o, i=i: put("mlp_out%d" % i, o[0])))
            if cfg.mlp_layer_types[i] == "sparse":
                def gate(m, args, o, i=i):
                    put("topw%d" % i, o[1])
                    rec.setdefault("topk%d" % i, []).append(o[2].numpy().astype(np.int16))
                hk.append(layer.mlp.gate.register_forward_hook(gate))
        ids = trace_prompt(tok, a.trace_prefill)
        cache = DynamicCache(config=cfg)
        t0 = time.time()
        lg = run(ids, cache)
        put("logits", lg)
        gen = [int(lg[-1].argmax())]
        log("trace: prompt of %d tokens in %.0f s (%.1f GB read; experts: %d read, %d from memory)" % (
            len(ids), time.time() - t0, sh.read_bytes / 1e9, ex.loads, ex.hits))
        t0 = time.time()
        for j in range(a.trace_decode):
            lg = run([gen[-1]], cache)
            put("logits", lg)
            gen.append(int(lg[-1].argmax()))
            if (j + 1) % 16 == 0:
                log("trace: %d steps, %.1f s a step (%.1f GB read; experts %d read, %d from memory)" % (
                    j + 1, (time.time() - t0) / (j + 1), sh.read_bytes / 1e9, ex.loads, ex.hits))
        for h in hk:
            h.remove()
        hit = [j for j, t in enumerate(gen[:-1]) if t in eos]
        log("trace: %r%s" % (tok.decode(gen)[-300:], "; EOS at steps %s (the steps went on)" % hit if hit else ""))
        out = os.path.join(a.out, "trace_%d_%d.npz" % (a.trace_prefill, a.trace_decode))
        np.savez(out, ids=np.array(ids, np.int32), gen=np.array(gen, np.int32), prompt=np.array(tok.decode(ids)),
                 **{k: np.concatenate(v) for k, v in rec.items()})
        log("written %s (%.1f GB)" % (out, os.path.getsize(out) / 1e9))
        del rec

    if a.forced:
        z = np.load(a.forced)
        toks = z["ids"].tolist() + z["gen"].tolist()
        cache = DynamicCache(config=cfg)
        rec = {k: [] for k in ("top_ids", "top_vals", "lse", "argmax")}
        t0 = time.time()
        for c0 in range(0, len(toks), 256):
            rows(run(toks[c0:c0 + 256], cache), rec)
            log("forced: %d of %d rows (%.0f s)" % (min(len(toks), c0 + 256), len(toks), time.time() - t0))
        out = os.path.join(a.out, os.path.basename(a.forced)[:-4] + "_ref.npz")
        np.savez(out, ids=z["ids"], gen=z["gen"], **{k: np.concatenate(v) for k, v in rec.items()})
        log("written %s" % out)
        return

    for name, msgs in prompts():
        if a.only and name not in a.only.split(","):
            continue
        text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        log("%s: %d tokens: %r" % (name, len(ids), text[-120:]))
        cache = DynamicCache(config=cfg)
        rec = {k: [] for k in ("top_ids", "top_vals", "lse", "argmax")}
        full, gen = [], []
        t0 = time.time()
        record(name == a.dump)
        lg = run(ids, cache)
        record(False)
        if name == a.dump:
            np.savez(os.path.join(a.out, name + "_layers.npz"), ids=np.array(ids, np.int32), **dump)
            dump.clear()
        rows(lg, rec)
        full.append(lg[-1].numpy().copy())
        gen.append(int(lg[-1].argmax()))
        log("%s: prompt in %.0f s (%.1f GB read; experts: %d read, %d from memory)" % (
            name, time.time() - t0, sh.read_bytes / 1e9, ex.loads, ex.hits))
        t0 = time.time()
        while gen[-1] not in eos and len(gen) < a.steps:
            lg = run([gen[-1]], cache)
            rows(lg, rec)
            if len(full) <= N_FULL:
                full.append(lg[-1].numpy().copy())
            gen.append(int(lg[-1].argmax()))
            if len(gen) % 10 == 0:
                log("%s: %d tokens, %.1f s a step (%.1f GB read; experts %d read, %d from memory)" % (
                    name, len(gen), (time.time() - t0) / (len(gen) - 1), sh.read_bytes / 1e9, ex.loads, ex.hits))
        out = tok.decode(gen)
        log("%s: %d tokens: %r" % (name, len(gen), out[-300:]))
        np.savez(os.path.join(a.out, name + ".npz"), ids=np.array(ids, np.int32), gen=np.array(gen, np.int32),
                 prompt=np.array(text), text=np.array(out),
                 top_ids=np.concatenate(rec["top_ids"]), top_vals=np.concatenate(rec["top_vals"]),
                 lse=np.concatenate(rec["lse"]), argmax=np.concatenate(rec["argmax"]),
                 step_logits=np.stack(full))
    log("written to %s" % a.out)


if __name__ == "__main__":
    main()
