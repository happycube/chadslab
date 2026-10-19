#!/usr/bin/env python3
"""The reference of Qwen3.8-Flash-Next (qwen4_exp) on the CPU, from the original bfloat16 checkpoint.

The transformers code (models/qwen4_exp, Qwen4ExpForCausalLM: the text model alone, no vision tower,
no MTP layer) runs as it is, in float32 on the CPU, one prompt at a time with its cache. The weights
come from the shards (model.language_model.* -> model.*):

- every weight but the routed experts and the n-gram table once at the start (float32, about 20 GB);
- the routed experts (one fused tensor of each layer: experts.gate_up_proj (E, 2 I, H) and
  experts.down_proj (E, H, I), each expert a contiguous slice) as the router picks them: a pre-hook
  of Qwen4ExpTextExperts reads the experts of top_k_index into one float32 stack that all layers
  share, with an LRU of bfloat16 experts in memory (--expert-cache-gb), the misses read in the order
  of the experts;
- the n-gram table of the PLE layer (128 shards of rows, 320M rows of 160 values): a module in place
  of its nn.Embedding that reads the rows it is given (row r: shard r // rows_per_shard).

Reads of the shards drop their pages (POSIX_FADV_DONTNEED).

Each prompt runs as one prompt pass, then --steps greedy decode steps. For each prompt it writes
<out>/<name>.npz:

    ids, prompt          the prompt (the model's template, the HF tokenizer)
    gen                  the greedy tokens (to EOS or --steps)
    top_ids, top_vals    [T + len(gen) - 1, 64]: the top 64 logits of each row
    lse, argmax          [T + len(gen) - 1]: the log-sum-exp and the argmax
    step_logits          [1 + min(16, len(gen) - 1), vocab]: the full logits of the last prompt row
                         and of the first 16 decode rows

Row r of top_ids / lse / argmax is the prediction for token r + 1 of ids + gen. The rows of gen run
as decode steps, the rows of ids as a prompt.

--forced FILE.npz (ids, gen: the tokens of another run): the reference teacher-forced on ids + gen, in
chunks of 256 rows; <out>/<name of FILE>_ref.npz gets top_ids, top_vals, lse, argmax of every row.

    OMP_NUM_THREADS=1 taskset -c 24-47 python -I scripts/qwen38_flash_reference.py \\
        /space/models/Qwen3.8-Flash-Next --out ~/reclank-golden/qwen38_flash --threads 24
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


def log(msg):
    print("ref: " + msg, file=sys.stderr, flush=True)


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

    def raw(self, p, off, n):
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
        p, dt, shape, s, e = self.t[name]
        b = self.raw(p, s, e - s)
        if dt == "F32":
            return np.frombuffer(b, np.float32).reshape(shape).copy()
        if dt == "I64":
            return np.frombuffer(b, np.int64).reshape(shape).copy()
        assert dt == "BF16", (name, dt)
        return (np.frombuffer(b, np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(shape)

    def bf16_slice(self, name, index):
        """The bfloat16 bits of [index] of a tensor (its first dimension), uint16."""
        p, dt, shape, s, e = self.t[name]
        assert dt == "BF16", (name, dt)
        per = int(np.prod(shape[1:]))
        b = self.raw(p, s + 2 * per * index, 2 * per)
        return np.frombuffer(b, np.uint16).reshape(shape[1:])


def f32(u16):
    return (np.asarray(u16).astype(np.uint32) << 16).view(np.float32)


class Experts:
    """The routed experts: one float32 stack of each fused matrix that every layer shares, filled with
    the experts of the layer that runs (need), from an LRU of bfloat16 experts or the shards."""

    def __init__(self, sh, cfg, cache_gb):
        import torch
        E, I, H = cfg.num_experts, cfg.moe_intermediate_size, cfg.hidden_size
        self.sh = sh
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
                pre = MAIN + "layers.%d.mlp.experts." % layer
                w = (self.sh.bf16_slice(pre + "gate_up_proj", e), self.sh.bf16_slice(pre + "down_proj", e))
                self.loads += 1
            self.gu[e] = torch.from_numpy(f32(w[0]))
            self.dn[e] = torch.from_numpy(f32(w[1]))
            self.lru[key] = w
            while len(self.lru) > self.cap:
                self.lru.popitem(last=False)
            self.have.add(e)


def row_table(sh, prefix):
    """A module in place of the nn.Embedding of the n-gram table: forward(ids) reads those rows of
    the shards (float32)."""
    import torch
    from torch import nn
    names = sorted((n for n in sh.t if n.startswith(prefix + ".shard_")),
                   key=lambda n: int(n[len(prefix) + 7:].split(".")[0]))
    rows = [sh.t[n][2][0] for n in names]
    assert len(set(rows[:-1])) <= 1, "the shards hold the same count of rows (the last may be short)"
    per, dim = rows[0], sh.t[names[0]][2][1]

    class RowTable(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.empty(0)            # (the device of the module: the CPU)
            self.num_rows = sum(rows)

        def forward(self, ids):
            flat = ids.reshape(-1).tolist()
            out = np.empty((len(flat), dim), np.float32)
            for j, r in enumerate(flat):
                n = names[r // per]
                p, dt, shape, s, e = sh.t[n]
                b = sh.raw(p, s + 2 * dim * (r % per), 2 * dim)
                out[j] = f32(np.frombuffer(b, np.uint16))
            return torch.from_numpy(out).reshape(*ids.shape, dim)
    return RowTable()


def build(model_dir, sh, expert_cache_gb, layers=None):
    """The model on the meta device, then every weight but the routed experts and the n-gram table
    from the shards (float32), the experts through Experts, the table through row_table."""
    import torch
    from torch import nn
    from transformers import AutoConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM, Qwen4ExpTextRotaryEmbedding

    cfg = AutoConfig.from_pretrained(model_dir).text_config
    cfg.dtype = torch.float32
    if layers:
        # (a test of the script) the first layers only
        cfg.num_hidden_layers = layers
        cfg.layer_types = cfg.layer_types[:layers]
    with torch.device("meta"):
        model = Qwen4ExpForCausalLM._from_config(cfg, dtype=torch.float32)
    model.eval()
    t0 = time.time()
    lm = model.model
    tables = {}
    for i, layer in enumerate(lm.layers):
        if layer.ple is not None:
            src = MAIN + "layers.%d.ple.ple_embedding.ngram_embedding" % i
            tables[i] = row_table(sh, src)
            layer.ple.ple_embedding.ngram_embedding = tables[i]
    used = set()
    for n, _ in list(model.named_parameters()):
        if ".mlp.experts." in n:
            continue
        src = n.replace("model.", MAIN, 1) if n.startswith("model.") else n
        assert src in sh.t, (n, src)
        path, leaf = n.rsplit(".", 1)
        model.get_submodule(path)._parameters[leaf] = nn.Parameter(torch.from_numpy(sh.f32(src)), requires_grad=False)
        used.add(src)
    for n, b in list(model.named_buffers()):
        src = n.replace("model.", MAIN, 1)
        if src in sh.t:
            path, leaf = n.rsplit(".", 1)
            model.get_submodule(path)._buffers[leaf] = torch.from_numpy(sh.f32(src))
            used.add(src)
    left = [n for n in sh.t if n not in used and not n.startswith(("mtp.", "model.visual."))
            and ".mlp.experts." not in n and ".ngram_embedding.shard_" not in n]
    assert layers or not left, left[:5]
    log("dense weights in %.0f s (%.1f GB read)" % (time.time() - t0, sh.read_bytes / 1e9))
    ex = Experts(sh, cfg, expert_cache_gb)
    for i, layer in enumerate(lm.layers):
        e = layer.mlp.experts
        e._parameters["gate_up_proj"] = nn.Parameter(ex.gu, requires_grad=False)
        e._parameters["down_proj"] = nn.Parameter(ex.dn, requires_grad=False)
        e.register_forward_pre_hook(lambda mod, args, i=i: ex.need(i, args[1].unique().tolist()))
    # the buffers made at init on meta: make them again on the CPU
    lm.rotary_emb = Qwen4ExpTextRotaryEmbedding(config=cfg)
    still = [n for n, b in model.named_buffers() if b.is_meta] + [n for n, p in model.named_parameters() if p.is_meta]
    assert not still, still[:5]
    return model, ex, tables


def check_table(sh, table, prefix):
    """The rows of the table module against the safetensors library on a few rows."""
    import torch
    from safetensors import safe_open
    per = sh.t[prefix + ".shard_0.weight"][2][0]
    rows = [0, 1, per - 1, per, 5 * per + 12345, table.num_rows - 1]
    got = table(torch.tensor(rows)).numpy()
    for j, r in enumerate(rows):
        n = prefix + ".shard_%d.weight" % (r // per)
        with safe_open(sh.t[n][0], "pt") as f:
            want = f.get_slice(n)[r % per].float().numpy()
        assert np.array_equal(got[j], want), r
    log("n-gram table: rows %s match the safetensors library" % rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--only", default=None, help="names of prompts, comma separated")
    ap.add_argument("--threads", type=int, default=24)
    ap.add_argument("--expert-cache-gb", type=float, default=80.0)
    ap.add_argument("--layers", type=int, default=None, help="(a test of the script) only the first N layers")
    ap.add_argument("--forced", default=None, help="an npz of ids and gen: the logits of those tokens")
    a = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from transformers.cache_utils import DynamicCache

    torch.set_num_threads(a.threads)
    torch.set_grad_enabled(False)
    os.makedirs(a.out, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    eos = {tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")}
    sh = Shards(a.model)
    model, ex, tables = build(a.model, sh, a.expert_cache_gb, a.layers)
    cfg = model.config
    for i, t in tables.items():
        check_table(sh, t, MAIN + "layers.%d.ple.ple_embedding.ngram_embedding" % i)

    def run(x, pos, cache):
        out = model(input_ids=torch.tensor([x]), past_key_values=cache, use_cache=True,
                    position_ids=torch.arange(pos, pos + len(x)).view(1, -1))
        return out.logits[0].float()

    def rows(lg, rec):
        tv, ti = torch.topk(lg, N_TOP, dim=-1)
        rec["top_ids"].append(ti.numpy().astype(np.int32))
        rec["top_vals"].append(tv.numpy())
        rec["lse"].append(torch.logsumexp(lg, dim=-1).numpy())
        rec["argmax"].append(lg.argmax(-1).numpy().astype(np.int32))

    if a.forced:
        z = np.load(a.forced)
        toks = z["ids"].tolist() + z["gen"].tolist()
        cache = DynamicCache(config=cfg)
        rec = {k: [] for k in ("top_ids", "top_vals", "lse", "argmax")}
        t0 = time.time()
        for c0 in range(0, len(toks), 256):
            rows(run(toks[c0:c0 + 256], c0, cache), rec)
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
        log("%s: %d tokens" % (name, len(ids)))
        cache = DynamicCache(config=cfg)
        rec = {k: [] for k in ("top_ids", "top_vals", "lse", "argmax")}
        full, gen = [], []
        t0 = time.time()
        lg = run(ids, 0, cache)
        rows(lg, rec)
        full.append(lg[-1].numpy().copy())
        gen.append(int(lg[-1].argmax()))
        log("%s: prompt in %.0f s (%.1f GB read; experts: %d read, %d from memory)" % (
            name, time.time() - t0, sh.read_bytes / 1e9, ex.loads, ex.hits))
        t0 = time.time()
        while gen[-1] not in eos and len(gen) < a.steps:
            lg = run([gen[-1]], len(ids) + len(gen) - 1, cache)
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
