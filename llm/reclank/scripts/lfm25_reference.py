#!/usr/bin/env python3
"""The reference of LFM2.5-24B-A2B on the CPU (docs/LFM25_24B_PLAN.md
Phase 0).

The transformers code (models/lfm2_moe) runs as it is, in float32 on the
CPU, layer by layer (Lfm2MoeDecoderLayer). The weights come from the shards:
the dense ones (every matrix but the routed experts: 1.0B values, 4 GB in
float32) once at the start, and the routed experts of a layer as the router
picks them. A pre-hook of Lfm2MoeExperts reads the experts of top_k_index
into one float32 stack that all layers share, with an LRU of bfloat16
experts in memory (--expert-cache-gb). The checkpoint stores each expert as
w1 / w3 / w2 (gate, up, down). The model's fused stack gate_up_proj is
[w1; w3], which chunk(2) splits back into gate and up. Reads of the shards
drop their pages (POSIX_FADV_DONTNEED).

The prompts run layer by layer (each layer once for all prompts), then
--steps greedy decode steps, all the prompts in each step. The model always
thinks (the template ends in "<think>"), so a short answer still takes a few
hundred tokens. For each prompt it writes <out>/<name>.npz:

    ids, prompt          the prompt (the model's template, the HF tokenizer)
    gen                  the greedy tokens (to EOS or --steps)
    hs                   [41, T + len(gen), 2048] float32: hs[0] the
                         embeddings, hs[i + 1] the output of layer i (before
                         the final norm)
    routes, route_w      [40, T + len(gen), 4]: the experts of the router and
                         their weights (-1 / 0 in the dense layers 0 and 1)
    route_margin         [40, T + len(gen)]: (sigmoid + bias) of the 4th
                         choice less that of the 5th: how close the
                         selection was (NaN in the dense layers)
    top_ids, top_vals    [T + len(gen), 64]: the top 64 logits of each row
    lse, argmax          [T + len(gen)]: the log-sum-exp and the argmax
    step_logits          [min(16, len(gen)), vocab]: the full logits of the
                         rows that chose the first 16 tokens of gen

Row r of top_ids / lse / argmax is the prediction for token r + 1 of
ids + gen. The rows of gen run as decode steps (the conv state and the
attention cache), the rows of ids as a prompt.

    OMP_NUM_THREADS=4 taskset -c 20-23 python -I \\
        scripts/lfm25_reference.py ~/models/LFM2.5-24B-A2B --out ~/reclank-golden/lfm25
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
EOS = (124900, 124895)          # <|im_end|>, <|endoftext|>
N_TOP = 64
N_FULL = 16

TOOLS = [{"name": "get_weather", "description": "Get the current weather for a city.",
          "parameters": {"type": "object", "properties": {
              "city": {"type": "string", "description": "The city name"},
              "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}},
         {"name": "search_files", "description": "Search the files of a repository for a regular expression.",
          "parameters": {"type": "object", "properties": {
              "pattern": {"type": "string"}, "path": {"type": "string"}}, "required": ["pattern"]}}]


def log(msg):
    print("ref: " + msg, file=sys.stderr, flush=True)


def prompts():
    """(name, messages, tools): real text and code, a tool call, and
    non-Latin text (the tokenizer expansion of LFM2.5)."""
    doc = open(os.path.join(ROOT, "docs", "MOE_PLAN.md")).read()[:6000]
    src = open(os.path.join(ROOT, "reclank", "core", "session.py")).read()
    code = src[src.index("def common_len"):src.index("class SessionPool")].rstrip()
    u = lambda t: [{"role": "user", "content": t}]
    return [
        ("doc", u("Here is a design document:\n\n" + doc + "\n\nWhat are its three main decisions? Answer briefly."),
         None),
        ("think", u("List the first ten prime numbers, separated by commas."), None),
        ("code", u("```python\n" + code + "\n```\n\nExplain what this code does, then write a pytest unit test "
                   "for common_len."), None),
        ("tool", u("What's the weather in Lyon right now, in celsius? And find where SessionPool is defined "
                   "under reclank/."), TOOLS),
        ("intl", u("Translate into Hindi, Japanese and Russian: \"The cache keeps the state of each "
                   "conversation, so a new turn only reads its new tokens.\""), None),
    ]


class Shards:
    """name -> (path, dtype, shape, start, end) of the shards; reads that
    drop their pages from the page cache."""

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

    def _fd(self, p):
        if p not in self.fds:
            self.fds[p] = os.open(p, os.O_RDONLY)
        return self.fds[p]

    def raw(self, p, off, n):
        fd = self._fd(p)
        buf = bytearray(n)
        mv = memoryview(buf)
        got = 0
        while got < n:
            got += os.preadv(fd, [mv[got:]], off + got)
        os.posix_fadvise(fd, off, n, os.POSIX_FADV_DONTNEED)
        self.read_bytes += n
        return buf

    def bf16(self, name):
        p, dt, shape, s, e = self.t[name]
        assert dt == "BF16", (name, dt)
        return np.frombuffer(self.raw(p, s, e - s), np.uint16).reshape(shape)

    def f32(self, name):
        p, dt, shape, s, e = self.t[name]
        if dt == "F32":
            return np.frombuffer(self.raw(p, s, e - s), np.float32).reshape(shape).copy()
        return f32(self.bf16(name))


def f32(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


class Experts:
    """The routed experts: one float32 stack of each matrix that every layer
    shares, filled with the experts of the layer that runs (need), from an
    LRU of bfloat16 experts or the shards."""

    def __init__(self, sh, cfg, cache_gb):
        import torch
        E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
        self.sh = sh
        self.I = I
        self.gu = torch.empty(E, 2 * I, H)
        self.dn = torch.empty(E, H, I)
        self.layer = None
        self.have = set()
        self.lru = OrderedDict()
        self.cap = int(cache_gb * 1e9 / (3 * I * H * 2))
        self.loads = 0
        self.hits = 0

    def need(self, layer, experts):
        import torch
        if layer != self.layer:
            self.layer, self.have = layer, set()
        for e in sorted(set(experts) - self.have):
            key = (layer, e)
            if key in self.lru:
                w = self.lru.pop(key)
                self.hits += 1
            else:
                pre = "model.layers.%d.feed_forward.experts.%d." % (layer, e)
                w = tuple(self.sh.bf16(pre + x + ".weight") for x in ("w1", "w3", "w2"))
                self.loads += 1
            self.gu[e, :self.I] = torch.from_numpy(f32(w[0]))
            self.gu[e, self.I:] = torch.from_numpy(f32(w[1]))
            self.dn[e] = torch.from_numpy(f32(w[2]))
            self.lru[key] = w
            while len(self.lru) > self.cap:
                self.lru.popitem(last=False)
            self.have.add(e)


def build(model_dir, sh, expert_cache_gb):
    """The model on the meta device, then every weight but the routed
    experts from the shards (float32), the experts through Experts."""
    import torch
    from torch import nn
    from transformers import AutoConfig
    from transformers.models.lfm2_moe.modeling_lfm2_moe import Lfm2MoeForCausalLM, Lfm2MoeRotaryEmbedding

    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.dtype = torch.float32
    with torch.device("meta"):
        model = Lfm2MoeForCausalLM._from_config(cfg, dtype=torch.float32)
    model.eval()

    def put(name, t, buffer=False):
        path, leaf = name.rsplit(".", 1)
        mod = model.get_submodule(path)
        if buffer:
            mod._buffers[leaf] = t
        else:
            mod._parameters[leaf] = nn.Parameter(t, requires_grad=False)

    t0 = time.time()
    params = [n for n, _ in model.named_parameters()]
    buffers = [n for n, b in model.named_buffers() if n.endswith("expert_bias")]
    dense = [n for n in params if ".experts." not in n]
    missing = [n for n in dense + buffers if n not in sh.t]
    assert not missing, missing[:5]
    for n in dense:
        put(n, torch.from_numpy(sh.f32(n)))
    for n in buffers:
        put(n, torch.from_numpy(sh.f32(n)), buffer=True)
    model.lm_head.weight = model.model.embed_tokens.weight          # tied
    left = [n for n in sh.t if ".experts." not in n and n not in set(dense + buffers)]
    assert not left, left[:5]
    log("dense weights in %.0f s (%.1f GB read)" % (time.time() - t0, sh.read_bytes / 1e9))

    ex = Experts(sh, cfg, expert_cache_gb)
    lm = model.model
    for i, layer in enumerate(lm.layers):
        if i < cfg.num_dense_layers:
            continue
        e = layer.feed_forward.experts
        e._parameters["gate_up_proj"] = nn.Parameter(ex.gu, requires_grad=False)
        e._parameters["down_proj"] = nn.Parameter(ex.dn, requires_grad=False)
        e.register_forward_pre_hook(lambda mod, args, i=i: ex.need(i, args[1].unique().tolist()))
    # the buffers made at init (rope) are on meta: make them again on the CPU
    lm.pos_emb = Lfm2MoeRotaryEmbedding(config=cfg)
    still = [n for n, b in model.named_buffers() if b.is_meta] + [n for n, p in model.named_parameters()
                                                                    if p.is_meta]
    assert not still, still[:5]
    return model, ex


class Run:
    """One prompt: its inputs, cache, and the records of its rows."""

    def __init__(self, name, ids):
        self.name = name
        self.ids = ids
        self.gen = []
        self.done = False
        self.hs, self.routes, self.route_w, self.margin = [], [], [], []
        self.top_ids, self.top_vals, self.lse, self.argmax = [], [], [], []
        self.step_logits = []
        self.extra = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--only", default=None, help="names of prompts, comma separated")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--expert-cache-gb", type=float, default=16.0)
    ap.add_argument("--layers", type=int, default=None, help="(a test of the script) only the first N layers")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    sh = Shards(a.model)
    model, ex = build(a.model, sh, a.expert_cache_gb)
    lm = model.model
    cfg = model.config
    head = model.lm_head.weight
    L = cfg.num_hidden_layers
    K = cfg.num_experts_per_tok
    if a.layers:
        lm.layers = lm.layers[:a.layers]

    runs = []
    for name, msgs, tools in prompts():
        if a.only and name not in a.only.split(","):
            continue
        text = tok.apply_chat_template(msgs, tools=tools, add_generation_prompt=True, tokenize=False)
        ids = tok(text, add_special_tokens=False)["input_ids"]
        r = Run(name, ids)
        r.extra["prompt"] = np.array(text)
        runs.append(r)
        log("%s: %d tokens" % (name, len(ids)))

    hidden = {}
    for r in runs:
        hidden[r.name] = lm.embed_tokens(torch.tensor([r.ids]))
        r.cache = DynamicCache(config=cfg)
        r.cur_pos = torch.arange(len(r.ids)).view(1, -1)

    def run_layers(active):
        """Every layer for the rows of the active runs (layer-major); the
        hidden state after the final norm of each."""
        prep = {}
        for r in active:
            x = hidden[r.name]
            p = r.cur_pos
            mk = dict(config=cfg, inputs_embeds=x, attention_mask=None, past_key_values=r.cache, position_ids=p)
            masks = {"full_attention": create_causal_mask(**mk), "conv": create_recurrent_attention_mask(**mk)}
            prep[r.name] = (masks, lm.pos_emb(x, p), p)
            n = x.shape[1]
            r._hs = [x[0].numpy().copy()]
            r._routes = np.full((L, n, K), -1, np.int16)
            r._route_w = np.zeros((L, n, K), np.float32)
            r._margin = np.full((L, n), np.nan, np.float32)
        for i, layer in enumerate(lm.layers):
            cap = {}
            hk = None
            moe = i >= cfg.num_dense_layers
            if moe:
                hk = layer.feed_forward.gate.register_forward_hook(lambda mod, args, out: cap.__setitem__("r", out))
                bias = layer.feed_forward.expert_bias
            for r in active:
                masks, pe, tp = prep[r.name]
                x = layer(hidden[r.name], position_embeddings=pe, attention_mask=masks[cfg.layer_types[i]],
                          position_ids=tp, past_key_values=r.cache)
                hidden[r.name] = x
                r._hs.append(x[0].numpy().copy())
                if moe:
                    logits, w, idx = cap["r"]
                    r._routes[i] = idx.numpy().astype(np.int16)
                    r._route_w[i] = w.float().numpy()
                    srt = torch.sort(logits.sigmoid() + bias, dim=-1, descending=True).values
                    r._margin[i] = (srt[:, K - 1] - srt[:, K]).numpy()
            if hk is not None:
                hk.remove()
        out = {}
        for r in active:
            r.hs.append(np.stack(r._hs))
            r.routes.append(r._routes)
            r.route_w.append(r._route_w)
            r.margin.append(r._margin)
            out[r.name] = lm.embedding_norm(hidden[r.name])[0]
        return out

    def logits_rows(r, h, full_rows):
        for c0 in range(0, h.shape[0], 256):
            lg = h[c0:c0 + 256] @ head.T
            tv, ti = torch.topk(lg, N_TOP, dim=-1)
            r.top_ids.append(ti.numpy().astype(np.int32))
            r.top_vals.append(tv.numpy())
            r.lse.append(torch.logsumexp(lg, dim=-1).numpy())
            r.argmax.append(lg.argmax(-1).numpy().astype(np.int32))
            for j in full_rows:
                if c0 <= j < c0 + lg.shape[0]:
                    r.step_logits.append(lg[j - c0].numpy().copy())
        return int(r.argmax[-1][-1])

    t0 = time.time()
    out = run_layers(runs)
    for r in runs:
        nxt = logits_rows(r, out[r.name], [out[r.name].shape[0] - 1])
        r.gen.append(nxt)
        r.done = nxt in EOS
    log("prompts in %.0f s (%.1f GB read; experts: %d read, %d from memory)" % (
        time.time() - t0, sh.read_bytes / 1e9, ex.loads, ex.hits))

    for s in range(1, a.steps):
        active = [r for r in runs if not r.done]
        if not active:
            break
        t0 = time.time()
        for r in active:
            n = len(r.ids) + len(r.gen) - 1
            if "full_attention" in cfg.layer_types[:len(lm.layers)]:
                assert r.cache.get_seq_length() == n, (r.cache.get_seq_length(), n)
            hidden[r.name] = lm.embed_tokens(torch.tensor([[r.gen[-1]]]))
            r.cur_pos = torch.tensor([[n]])
        out = run_layers(active)
        for r in active:
            nxt = logits_rows(r, out[r.name], [0] if len(r.gen) < N_FULL else [])
            r.gen.append(nxt)
            r.done = nxt in EOS or len(r.gen) >= a.steps
        if s % 16 == 0 or len(active) < len(runs):
            log("step %d: %d prompts, %.1f s (%.1f GB read; experts %d read, %d from memory)" % (
                s, len(active), time.time() - t0, sh.read_bytes / 1e9, ex.loads, ex.hits))

    for r in runs:
        text = tok.decode(r.gen)
        log("%s: %d tokens: %r" % (r.name, len(r.gen), text[-300:]))
        # the last token of gen has no row of its own (it was not run)
        np.savez(os.path.join(a.out, r.name + ".npz"),
                 ids=np.array(r.ids, np.int32), gen=np.array(r.gen, np.int32), text=np.array(text),
                 hs=np.concatenate(r.hs, axis=1), routes=np.concatenate(r.routes, axis=1),
                 route_w=np.concatenate(r.route_w, axis=1), route_margin=np.concatenate(r.margin, axis=1),
                 top_ids=np.concatenate(r.top_ids), top_vals=np.concatenate(r.top_vals),
                 lse=np.concatenate(r.lse), argmax=np.concatenate(r.argmax),
                 step_logits=np.stack(r.step_logits), **r.extra)
    log("written to %s" % a.out)


if __name__ == "__main__":
    main()
