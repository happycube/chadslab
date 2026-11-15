#!/usr/bin/env python3
"""The reference of MiniCPM-V 4.7 35B-A3B on the CPU (docs/MINICPM_V47_PLAN.md
Phase 0).

The transformers code (models/minicpmv4_7, its text model qwen3_5_moe) runs
as it is, in float32 on the CPU: the processor (torchvision backend, the
default of HF and vLLM), get_image_features, get_rope_index (canvas M-RoPE),
and each Qwen3_5MoeDecoderLayer. The weights come from the shards: the dense
ones (every matrix but the routed experts, the vision tower, the head) once
at the start, float32 (12 GB); the routed experts of a layer as the router
picks them (a pre-hook of Qwen3_5MoeExperts reads the experts of top_k_index
into one float32 stack that all layers share), with an LRU of bfloat16
experts in memory (--expert-cache-gb). Reads of the shards drop their pages
(POSIX_FADV_DONTNEED), so the process holds about 30 GB beside the
weightkeeper and leaves the page cache alone.

The prompts run layer by layer (each layer once for all prompts, so the
experts are read once for all), then --steps greedy decode steps, all the
prompts in each step. For each prompt it writes <out>/<name>.npz:

    ids, prompt          the prompt (template of the model, the processor)
    gen                  the greedy tokens (to EOS or --steps)
    positions            [4, T + len(gen)]: text, then M-RoPE t, h, w
    hs                   [41, T + len(gen), 2048] float32: hs[0] the input of
                         layer 0, hs[i + 1] the output of layer i (before
                         the final norm)
    routes, route_w      [40, T + len(gen), 8]: the experts of the router and
                         their weights
    top_ids, top_vals    [T + len(gen), 64]: the top 64 logits of each row
    lse, argmax          [T + len(gen)]: the log-sum-exp and the argmax
    step_logits          [min(16, len(gen)), vocab]: the full logits of the
                         rows that chose the first 16 tokens of gen (the last
                         prompt row, then the steps)
    pixel_values, target_sizes, mm_token_type_ids, vis_hidden, vis_out
                         (an image prompt) the input of the tower, its last
                         hidden state and the rows that replace the
                         <|image_pad|> embeddings

Row r of top_ids / lse / argmax is the prediction for token r + 1 of
ids + gen; the rows of gen run as decode steps (the recurrent path of the
DeltaNet and the cache of the attention), the rows of ids as a prompt.

    python scripts/minicpm_reference.py MODELS/MiniCPM-V-4.7-35B-A3B --out OUT_DIR [--steps 128]
    (DS_IMAGES: a directory of test images, e.g. DeepSeek-V4.1-Flash's inference/examples/images)
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
DS_IMAGES = os.environ.get("DS_IMAGES", "")
EOS = (248044, 248046)
N_TOP = 64
N_FULL = 16


def log(msg):
    print("ref: " + msg, file=sys.stderr, flush=True)


def prompts(model_dir):
    """(name, user content, thinking, downsample_mode): real text, code and
    images."""
    doc = open(os.path.join(ROOT, "docs", "MOE_PLAN.md")).read()[:6000]
    src = open(os.path.join(ROOT, "reclank", "core", "session.py")).read()
    a = src.index("def common_len")
    b = src.index("class SessionPool")
    code = src[a:b].rstrip()
    chart = os.path.join(model_dir, "minicpm_v4dot7_assets", "minicpmv4.7", "overall_performance_35ba3b_instruct.png")
    return [
        ("doc", "Here is a design document:\n\n" + doc + "\n\nWhat are its three main decisions? Answer briefly.",
         False, None),
        ("think", "List the first ten prime numbers, separated by commas.", True, None),
        ("code", "```python\n" + code + "\n```\n\nExplain what this code does, then write a pytest unit test "
         "for common_len.", False, None),
        ("photo", [("image", DS_IMAGES + "corn.jpeg"), ("text", "What is in this image? Answer in one sentence.")],
         False, "16x"),
        ("chart", [("image", chart), ("text", "Which model has the highest overall score in this chart? "
                                              "Give the number.")], False, "16x"),
        ("photo4x", [("image", DS_IMAGES + "carrots.jpeg"), ("text", "Describe this image in two sentences.")],
         False, "4x"),
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

    def bf16(self, name, part=None):
        """The tensor as uint16 (bfloat16 bits); part (i, count) of the first
        dim only."""
        p, dt, shape, s, e = self.t[name]
        assert dt == "BF16", (name, dt)
        if part is None:
            return np.frombuffer(self.raw(p, s, e - s), np.uint16).reshape(shape)
        i, c = part
        row = (e - s) // shape[0]
        return np.frombuffer(self.raw(p, s + i * row, c * row), np.uint16).reshape((c,) + shape[1:])

    def f32(self, name):
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
        self.gu = torch.empty(E, 2 * I, H)
        self.dn = torch.empty(E, H, I)
        self.layer = None
        self.have = set()
        self.lru = OrderedDict()
        self.per = (2 * I * H + H * I) * 2
        self.cap = int(cache_gb * 1e9 / self.per)
        self.loads = 0
        self.hits = 0

    def need(self, layer, experts):
        import torch
        if layer != self.layer:
            self.layer, self.have = layer, set()
        pre = "model.language_model.layers.%d.mlp.experts." % layer
        for e in sorted(set(experts) - self.have):
            key = (layer, e)
            if key in self.lru:
                gu, dn = self.lru.pop(key)
                self.hits += 1
            else:
                gu = self.sh.bf16(pre + "gate_up_proj", (e, 1))[0]
                dn = self.sh.bf16(pre + "down_proj", (e, 1))[0]
                self.loads += 1
            self.gu[e] = torch.from_numpy(f32(gu))
            self.dn[e] = torch.from_numpy(f32(dn))
            self.lru[key] = (gu, dn)
            while len(self.lru) > self.cap:
                self.lru.popitem(last=False)
            self.have.add(e)


def build(model_dir, sh, expert_cache_gb):
    """The model on the meta device, then every weight but the routed
    experts from the shards (float32), the experts through Experts."""
    import torch
    from torch import nn
    from transformers import AutoConfig
    from transformers.models.minicpmv4_7.modeling_minicpmv4_7 import MiniCPMV4_7ForConditionalGeneration
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

    cfg = AutoConfig.from_pretrained(model_dir)
    cfg.dtype = torch.float32
    cfg.text_config.dtype = torch.float32
    cfg.vision_config.dtype = torch.float32
    with torch.device("meta"):
        model = MiniCPMV4_7ForConditionalGeneration._from_config(cfg, dtype=torch.float32)
    model.eval()

    def put(name, t):
        mod, leaf = model, name
        if "." in name:
            path, leaf = name.rsplit(".", 1)
            mod = model.get_submodule(path)
        mod._parameters[leaf] = nn.Parameter(t, requires_grad=False)

    t0 = time.time()
    names = [n for n, _ in model.named_parameters()]
    missing = [n for n in names if n not in sh.t]
    extra = [n for n in sh.t if n not in set(names)]
    assert not missing and not extra, (missing[:5], extra[:5])
    for n in names:
        if ".mlp.experts." in n:
            continue
        put(n, torch.from_numpy(sh.f32(n)))
    log("dense weights in %.0f s (%.1f GB read)" % (time.time() - t0, sh.read_bytes / 1e9))

    tc = cfg.text_config
    ex = Experts(sh, tc, expert_cache_gb)
    lm = model.model.language_model
    for i, layer in enumerate(lm.layers):
        e = layer.mlp.experts
        e._parameters["gate_up_proj"] = nn.Parameter(ex.gu, requires_grad=False)
        e._parameters["down_proj"] = nn.Parameter(ex.dn, requires_grad=False)
        e.register_forward_pre_hook(lambda mod, args, i=i: ex.need(i, args[1].unique().tolist()))
    # the buffers made at init (rope) are on meta: make them again on the CPU
    lm.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=tc)
    left = [n for n, b in model.named_buffers() if b.is_meta]
    assert not left, left
    return model, ex


def chat(proc, content, thinking, dm):
    from PIL import Image
    if isinstance(content, str):
        parts = [{"type": "text", "text": content}]
    else:
        parts = [{"type": "image", "image": Image.open(v)} if k == "image" else {"type": "text", "text": v}
                 for k, v in content]
    kw = {"enable_thinking": thinking}
    if dm:
        kw["downsample_mode"] = dm
    out = proc.apply_chat_template([{"role": "user", "content": parts}], tokenize=True, add_generation_prompt=True,
                                   return_dict=True, return_tensors="pt", **kw)
    return out


class Run:
    """One prompt: its inputs, cache, and the records of its rows."""

    def __init__(self, name, inputs, dm):
        self.name = name
        self.inputs = inputs
        self.dm = dm
        self.ids = inputs["input_ids"][0].tolist()
        self.gen = []
        self.done = False
        self.hs = []            # per row block: [41, n, H]
        self.routes = []        # per row block: [40, n, 8]
        self.route_w = []
        self.pos = []           # per row block: [4, n]
        self.top_ids, self.top_vals, self.lse, self.argmax = [], [], [], []
        self.step_logits = []
        self.extra = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=128)
    ap.add_argument("--only", default=None, help="names of prompts, comma separated")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--expert-cache-gb", type=float, default=16.0)
    ap.add_argument("--layers", type=int, default=None, help="(a test of the script) only the first N layers")
    a = ap.parse_args()

    import torch
    from transformers import AutoProcessor
    from transformers.cache_utils import DynamicCache
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    proc = AutoProcessor.from_pretrained(a.model)
    sh = Shards(a.model)
    model, ex = build(a.model, sh, a.expert_cache_gb)
    m = model.model
    lm = m.language_model
    tc = model.config.text_config
    head = model.lm_head.weight
    if a.layers:
        lm.layers = lm.layers[:a.layers]

    runs = []
    for name, content, thinking, dm in prompts(a.model):
        if a.only and name not in a.only.split(","):
            continue
        r = Run(name, chat(proc, content, thinking, dm), dm)
        r.extra["prompt"] = np.array(proc.tokenizer.decode(r.ids))
        runs.append(r)
        log("%s: %d tokens" % (name, len(r.ids)))

    # ---- the prompts: embeddings, images, positions
    hidden = {}
    for r in runs:
        inp = r.inputs
        ids_t = inp["input_ids"]
        x = lm.embed_tokens(ids_t)
        if "pixel_values" in inp:
            t0 = time.time()
            vo = m.get_image_features(inp["pixel_values"][:1], inp["target_sizes"], downsample_mode=r.dm)
            feats = torch.cat(vo.pooler_output, dim=0)
            mask = (ids_t == model.config.image_token_id).unsqueeze(-1).expand_as(x)
            assert int(mask[..., 0].sum()) == feats.shape[0], (int(mask[..., 0].sum()), feats.shape)
            x = x.masked_scatter(mask, feats)
            r.extra.update(pixel_values=inp["pixel_values"].numpy(), target_sizes=inp["target_sizes"].numpy(),
                           mm_token_type_ids=inp["mm_token_type_ids"].numpy(),
                           vis_hidden=vo.last_hidden_state.numpy(), vis_out=feats.numpy())
            kw = {k: inp[k] for k in ("input_ids", "attention_mask", "target_sizes", "mm_token_type_ids")}
            kw["downsample_mode"] = r.dm
            pos = model._prepare_position_ids_for_generation(ids_t, kw)
            r.delta = int(m.rope_deltas.reshape(-1)[0])
            log("%s: image %d rows in %.1f s, rope delta %d" % (r.name, feats.shape[0], time.time() - t0, r.delta))
        else:
            n = len(r.ids)
            pos = torch.arange(n).view(1, 1, n).expand(4, 1, n)
            r.delta = 0
        r.cache = DynamicCache(config=tc)
        hidden[r.name] = x
        r.pos.append(pos[:, 0].numpy().copy())
        r.cur_pos = pos

    def run_layers(active):
        """Every layer for the rows of the active runs (layer-major); the
        hidden state after the final norm of each."""
        prep = {}
        for r in active:
            x = hidden[r.name]
            p = r.cur_pos
            mk = dict(config=tc, inputs_embeds=x, attention_mask=None, past_key_values=r.cache, position_ids=p[0])
            masks = {"full_attention": create_causal_mask(**mk),
                     "linear_attention": create_recurrent_attention_mask(**mk)}
            prep[r.name] = (masks, lm.rotary_emb(x, p[1:]), p[0])
            r._hs = [x[0].numpy().copy()]
            r._routes, r._route_w = [], []
        for i, layer in enumerate(lm.layers):
            cap = {}
            hk = layer.mlp.gate.register_forward_hook(lambda mod, args, out: cap.__setitem__("r", out))
            for r in active:
                masks, pe, tp = prep[r.name]
                x = layer(hidden[r.name], position_embeddings=pe, attention_mask=masks[tc.layer_types[i]],
                          position_ids=tp, past_key_values=r.cache, use_cache=True)
                hidden[r.name] = x
                r._hs.append(x[0].numpy().copy())
                _, w, idx = cap["r"]
                r._routes.append(idx.numpy().astype(np.int16))
                r._route_w.append(w.float().numpy())
            hk.remove()
        out = {}
        for r in active:
            r.hs.append(np.stack(r._hs))
            r.routes.append(np.stack(r._routes))
            r.route_w.append(np.stack(r._route_w))
            out[r.name] = lm.norm(hidden[r.name])[0]
        return out

    def logits_rows(r, h, full_rows):
        """Top 64, lse, argmax of each row of h; the full logits of the rows
        in full_rows (indices into h)."""
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

    # ---- greedy decode steps
    for s in range(1, a.steps):
        active = [r for r in runs if not r.done]
        if not active:
            break
        t0 = time.time()
        for r in active:
            tok = r.gen[-1]
            n = len(r.ids) + len(r.gen) - 1
            if "full_attention" in tc.layer_types[:len(lm.layers)]:
                assert r.cache.get_seq_length() == n, (r.cache.get_seq_length(), n)
            hidden[r.name] = lm.embed_tokens(torch.tensor([[tok]]))
            p = torch.tensor([n, n + r.delta, n + r.delta, n + r.delta]).view(4, 1, 1)
            r.cur_pos = p
            r.pos.append(p[:, 0].numpy().copy())
        out = run_layers(active)
        for r in active:
            nxt = logits_rows(r, out[r.name], [0] if len(r.gen) < N_FULL else [])
            r.gen.append(nxt)
            r.done = nxt in EOS or len(r.gen) >= a.steps
        if s % 8 == 0 or len(active) < len(runs):
            log("step %d: %d prompts, %.1f s (%.1f GB read; experts %d read, %d from memory)" % (
                s, len(active), time.time() - t0, sh.read_bytes / 1e9, ex.loads, ex.hits))

    for r in runs:
        text = proc.tokenizer.decode(r.gen)
        log("%s: %d tokens: %r" % (r.name, len(r.gen), text[:300]))
        # the last token of gen has no row of its own (it was not run)
        np.savez(os.path.join(a.out, r.name + ".npz"),
                 ids=np.array(r.ids, np.int32), gen=np.array(r.gen, np.int32), text=np.array(text),
                 positions=np.concatenate(r.pos, axis=1), hs=np.concatenate(r.hs, axis=1),
                 routes=np.concatenate(r.routes, axis=1), route_w=np.concatenate(r.route_w, axis=1),
                 top_ids=np.concatenate(r.top_ids), top_vals=np.concatenate(r.top_vals),
                 lse=np.concatenate(r.lse), argmax=np.concatenate(r.argmax),
                 step_logits=np.stack(r.step_logits), **r.extra)
    log("written to %s" % a.out)


if __name__ == "__main__":
    main()
