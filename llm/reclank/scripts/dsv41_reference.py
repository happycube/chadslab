#!/usr/bin/env python3
"""The reference of DeepSeek V4.1 Flash on the CPU (DEEPSEEK_V41_PLAN.md Phase 0).

The model code of the checkpoint (inference/model.py, engram.py) runs as it
is, in torch on the CPU, with the ops of scripts/dsv41_ref/kernel.py in place
of its tilelang kernels. The weights come from the shards layer by layer
(each layer's dense weights loaded before it runs and dropped after; the
routed experts a token picks loaded for its MoE, then dropped; the engram
rows read from the shards), so the process holds a few GB beside the
zygote. The weights take the forms of the reference's convert.py: wo_a
dequantized to bfloat16, the experts as float4_e2m1fn_x2, the rest as
model.py declares them (the head and the compressors of ratio 2 in float32).

For each prompt (a chat turn through encoding.py, at position 0 in one
chunk) it writes an .npz: the ids, the logits of every prompt position, the
output of every layer (float32, hc x dim) at every prompt position, then the
greedy tokens of --steps decode steps with their logits and the output of
every layer of the first --layer-steps steps. Text only for now (no vision,
no DSpark).

Run in its own environment (torch on the CPU, transformers, tokenizers,
safetensors, sympy):

    taskset -c 0-47 /home/cpage/npg-run/dsref-venv/bin/python scripts/dsv41_reference.py \\
        /home/cpage/models/DeepSeek-V4.1-Flash --out /home/cpage/npg-run/dsv41-ref [--steps 16]
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

PROMPTS = [
    "What is the capital of France? Answer in one word.",
    "Write a haiku about a pelican riding a bicycle.",
    "List the first ten prime numbers, separated by commas.",
]


def log(msg):
    print("ref: " + msg, file=sys.stderr, flush=True)


class Shards:
    """name -> (file, dtype, shape, start, end) of the shards, and reads."""

    def __init__(self, path):
        self.path = path
        where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.t = {}
        for f in sorted(set(where.values())):
            with open(os.path.join(path, f), "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                h = json.loads(fh.read(n))
            h.pop("__metadata__", None)
            for k, v in h.items():
                o0, o1 = v["data_offsets"]
                self.t[k] = (f, v["dtype"], tuple(v["shape"]), 8 + n + o0, 8 + n + o1)
        self.maps = {}

    def raw(self, name):
        f, dt, shape, o0, o1 = self.t[name]
        with open(os.path.join(self.path, f), "rb") as fh:
            fh.seek(o0)
            return np.frombuffer(fh.read(o1 - o0), np.uint8), dt, shape

    def memmap(self, name):
        f, dt, shape, o0, o1 = self.t[name]
        if name not in self.maps:
            self.maps[name] = np.memmap(os.path.join(self.path, f), np.uint8, "r", o0, (o1 - o0,)).reshape(
                shape[0], -1)
        return self.maps[name]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--layer-steps", type=int, default=4)
    ap.add_argument("--threads", type=int, default=40)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--text", action="append", help="a prompt in place of the built-in ones")
    ap.add_argument("--long-file", action="append", help="a long prompt: the text of these files, cut to "
                    "--long-tokens, then a request to summarize it (the compressed KV and the indexers "
                    "past their top-k)")
    ap.add_argument("--long-tokens", type=int, default=1400)
    ap.add_argument("--dump-layers", help="a check: for the first prompt's prefill, the inputs (h, pre_mix) "
                    "and the intermediates of these layers (comma separated) to --dump-out, then stop")
    ap.add_argument("--dump-out")
    ap.add_argument("--keep-last", type=int, default=32,
                    help="the layer outputs of only the last N prompt positions of a prompt over 64 tokens")
    a = ap.parse_args()

    inf = os.path.join(a.ckpt, "inference")
    sys.path[:0] = [os.path.join(HERE, "dsv41_ref"), inf, os.path.join(a.ckpt, "encoding")]
    import torch
    import kernel  # noqa: F401  (ours: scripts/dsv41_ref/kernel.py)
    assert kernel.__file__.startswith(os.path.join(HERE, "dsv41_ref")), kernel.__file__
    import model as M
    from transformers import AutoTokenizer
    from encoding import encode_messages

    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(0)
    sh = Shards(a.ckpt)
    cfg = json.load(open(os.path.join(inf, "config.json")))
    cfg.update(max_batch_size=1, max_seq_len=a.max_seq_len, vision_n_layers=0, dspark_block_size=0,
               temperature=0.0)
    args = M.ModelArgs(**{k: v for k, v in cfg.items() if k in M.ModelArgs.__dataclass_fields__})

    # ---- the patches: no tensors for the routed experts nor the engram tables
    layer_of_table = {n: lid for n, lid in zip(args.engram_num_embeddings, args.engram_layer_ids)}

    class ShardEngramEmbedding(torch.nn.Module):
        def __init__(self, num_embeddings, dim):
            super().__init__()
            self.layer = layer_of_table[num_embeddings]
            self.dim = dim

        def forward(self, indices):
            w = sh.memmap("layers.%d.engram.embed.weight" % self.layer)
            s = sh.memmap("layers.%d.engram.embed.scale" % self.layer)
            idx = indices.reshape(-1).numpy()
            vals = torch.from_numpy(np.ascontiguousarray(w[idx])).view(torch.float8_e4m3fn).float()
            sc = torch.from_numpy(np.ascontiguousarray(s[idx])).view(torch.float8_e8m0fnu).float()
            v = (vals.unflatten(-1, (-1, 32)) * sc.unsqueeze(-1)).flatten(-2).to(torch.bfloat16)
            return v.reshape(*indices.shape, self.dim)

    M.ParallelEngramEmbedding = ShardEngramEmbedding

    def moe_init(self, layer_id, args_):
        torch.nn.Module.__init__(self)
        n_routed, n_act = args_.get_moe_config(layer_id)
        self.layer_id, self.dim = layer_id, args_.dim
        self.n_routed_experts, self.n_activated_experts = n_routed, n_act
        self.inter, self.limit = args_.moe_inter_dim, args_.swiglu_limit
        self.gate = M.Gate(layer_id, args_)
        self.shared_experts = M.Expert(args_.dim, args_.moe_inter_dim, swiglu_limit=args_.swiglu_limit)

    def expert(self, i):
        e = M.Expert(self.dim, self.inter, dtype=torch.float4_e2m1fn_x2, swiglu_limit=self.limit)
        for w in ("w1", "w2", "w3"):
            base = "layers.%d.ffn.experts.%d.%s." % (self.layer_id, i, w)
            raw, _dt, shape = sh.raw(base + "weight")
            lin = getattr(e, w)
            lin.weight.data = torch.from_numpy(raw.copy()).view(torch.float4_e2m1fn_x2).reshape(shape)
            sraw, _sdt, sshape = sh.raw(base + "scale")
            lin.scale.data = torch.from_numpy(sraw.copy()).view(torch.float8_e8m0fnu).reshape(sshape)
            lin.weight.scale = lin.scale
        return e

    def moe_forward(self, x, image_mask=None):
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x, None)
        y = torch.zeros_like(x, dtype=torch.float32)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        self.last_indices = indices.clone()
        for i in range(self.n_routed_experts):
            if counts[i] == 0:
                continue
            e = expert(self, i)
            idx, top = torch.where(indices == i)
            y[idx] += e(x[idx], weights[idx, top, None])
        y += self.shared_experts(x)
        return y.type_as(x).view(shape)

    M.MoE.__init__ = moe_init
    M.MoE.forward = moe_forward

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.ckpt)
    model = M.Transformer(args, tok)
    log("the model built, the engram token map made (%.0f s)" % (time.time() - t0))

    # ---- the weights
    def to_param(name, p):
        raw, dt, shape = sh.raw(name)
        if name.endswith("wo_a.weight"):
            w = torch.from_numpy(raw.copy()).view(torch.float8_e4m3fn).reshape(shape).float()
            sraw, _sdt, sshape = sh.raw(name[:-len("weight")] + "scale")
            s = torch.from_numpy(sraw.copy()).view(torch.float8_e8m0fnu).reshape(sshape).float()
            w = w.unflatten(0, (-1, 32)).unflatten(-1, (-1, 32)) * s[:, None, :, None]
            return w.flatten(2, 3).flatten(0, 1).to(p.dtype)
        tdt = {"BF16": torch.bfloat16, "F32": torch.float32, "F8_E4M3": torch.float8_e4m3fn,
               "F8_E8M0": torch.float8_e8m0fnu, "I8": torch.int8}[dt]
        t = torch.from_numpy(raw.copy()).view(tdt).reshape(shape)
        return t if t.dtype == p.dtype else t.to(p.dtype)

    def params_of(prefix, mod):
        return [(prefix + n, p) for n, p in mod.named_parameters()]

    def load(prefix, mod):
        for name, p in params_of(prefix, mod):
            p.data = to_param(name, p)

    def unload(prefix, mod):
        for _name, p in params_of(prefix, mod):
            p.data = torch.empty(0, dtype=p.dtype)

    load("embed.", model.embed)
    load("norm.", model.norm)
    load("head.", model.head)
    for i, layer in enumerate(model.layers):
        unload("layers.%d." % i, layer)
    log("embed, norm, head loaded")

    @torch.inference_mode()
    def forward(ids, start_pos, keep_layers):
        x = torch.tensor([ids], dtype=torch.long)
        hashes = model.engram_hash(x, start_pos, None)
        h = model.embed(x).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
        pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
        outs, routes = [], []
        for i, layer in enumerate(model.layers):
            load("layers.%d." % i, layer)
            if layer.engram is not None:
                h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
            h, pre_mix = layer(h, start_pos, pre_mix, None)
            unload("layers.%d." % i, layer)
            routes.append(layer.ffn.last_indices.numpy().astype(np.int16))
            if keep_layers:
                outs.append(h[0].float().numpy())
        h = layer.hc_pre(h, pre_mix)
        logits = model.head(model.norm(h), full_logits=True)[0].float().numpy()
        return logits, (np.stack(outs, 1) if keep_layers else None), np.stack(routes, 1)

    os.makedirs(a.out, exist_ok=True)
    if a.dump_layers:
        want = {int(v) for v in a.dump_layers.split(",")}
        rec, cur = {}, {"layer": -1}

        def keep(name, t):
            if cur["layer"] in want and t is not None:
                rec["L%d.%s" % (cur["layer"], name)] = t.detach().float().numpy().copy()

        def wrap(cls, meth, name, outs):
            orig = getattr(cls, meth)

            def f(self, *args, **kw):
                out = orig(self, *args, **kw)
                if isinstance(out, tuple):
                    for o, n in zip(out, outs):
                        if torch.is_tensor(o):
                            keep(name + "." + n, o)
                elif torch.is_tensor(out):
                    keep(name, out)
                if args and torch.is_tensor(args[0]):
                    keep(name + ".in", args[0])
                return out
            setattr(cls, meth, f)

        wrap(M.Attention, "forward", "attn", [])
        wrap(M.Compressor, "forward", "compressor", [])
        wrap(M.Indexer, "forward", "indexer", [])
        wrap(M.Attention, "_window_kv", "window", ["kv", "idx"])
        wrap(M.MoE, "forward", "moe", [])
        wrap(M.Engram, "forward", "engram", [])
        x = torch.tensor([tok.encode(encode_messages([{"role": "user", "content": (a.text or PROMPTS)[0]}],
                                                     thinking_mode="chat"), add_special_tokens=False)])
        with torch.inference_mode():
            hashes = model.engram_hash(x, 0, None)
            h = model.embed(x).unsqueeze(2).repeat(1, 1, args.hc_mult, 1)
            pre_mix = M.make_identity_pre_mix(h, args.hc_mult)
            for i, layer in enumerate(model.layers):
                if i > max(want):
                    break
                cur["layer"] = i
                if i in want:
                    rec["L%d.h_in" % i] = h.float().numpy().copy()
                    rec["L%d.pre_mix" % i] = pre_mix.float().numpy().copy()
                load("layers.%d." % i, layer)
                if layer.engram is not None:
                    h = layer.engram(h, hashes[:, :, layer.engram.layer_hash_index, :], None)
                if i in want:
                    rec["L%d.h_engram" % i] = h.float().numpy().copy()
                    for nm in ("compress_kv_cache", "window_kv_cache"):
                        pass
                h, pre_mix = layer(h, 0, pre_mix, None)
                if i in want:
                    rec["L%d.h_out" % i] = h.float().numpy().copy()
                    rec["L%d.routes" % i] = layer.ffn.last_indices.numpy().copy()
                    att = layer.attn
                    if att.is_kv_source:
                        rec["L%d.ckv" % i] = att.compress_kv_cache[0, :x.shape[1] // att.compress_ratio].float().numpy().copy()
                    if att.indexer is not None and att.indexer.owns_k:
                        rec["L%d.kcache" % i] = att.indexer.k_cache[0, :x.shape[1] // att.compress_ratio].float().numpy().copy()
                    if M.shared_attn.topk_idxs is not None:
                        rec["L%d.topk" % i] = M.shared_attn.topk_idxs.numpy().copy()
                unload("layers.%d." % i, layer)
        np.savez(a.dump_out, ids=x.numpy()[0], **rec)
        log("dumped %d arrays to %s" % (len(rec), a.dump_out))
        return
    texts = list(a.text or PROMPTS[:a.prompts])
    if a.long_file:
        body = "\n\n".join(open(f).read() for f in a.long_file)
        body = tok.decode(tok.encode(body, add_special_tokens=False)[:a.long_tokens])
        texts.append(body + "\n\nSummarize the text above in two sentences.")
    for pi, text in enumerate(texts):
        prompt = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
        ids = tok.encode(prompt, add_special_tokens=False)
        log("prompt %d: %d tokens: %r" % (pi, len(ids), prompt[:120]))
        # a fresh state for each prompt: the caches are written from position 0 on
        t1 = time.time()
        logits, layers_out, routes = forward(ids, 0, True)
        first = 0 if len(ids) <= 64 else len(ids) - a.keep_last
        layers_out = layers_out[first:]
        log("prefill %.1f s; the top token %d %r" % (time.time() - t1, int(logits[-1].argmax()),
                                                    tok.decode([int(logits[-1].argmax())])))
        gen, step_logits, step_layers, step_routes = [], [], [], []
        nxt = int(logits[-1].argmax())
        pos = len(ids)
        for s in range(a.steps):
            gen.append(nxt)
            t1 = time.time()
            lg, lo, rt = forward([nxt], pos, s < a.layer_steps)
            step_logits.append(lg[-1])
            step_routes.append(rt[0])
            if lo is not None:
                step_layers.append(lo[0])
            pos += 1
            nxt = int(lg[-1].argmax())
            log("step %d (%.1f s): %d %r" % (s, time.time() - t1, nxt, tok.decode([nxt])))
            if nxt == tok.eos_token_id:
                break
        out = os.path.join(a.out, "prompt%d.npz" % pi)
        np.savez(out, ids=np.array(ids), prompt=prompt, logits=logits.astype(np.float32),
                 layers=layers_out.astype(np.float32), layers_first=first, routes=routes, gen=np.array(gen + [nxt]),
                 step_logits=np.array(step_logits, np.float32),
                 step_layers=np.array(step_layers, np.float32), step_routes=np.array(step_routes),
                 text=tok.decode(gen + [nxt]))
        log("prompt %d: %r -> %s" % (pi, tok.decode(gen + [nxt]), out))


if __name__ == "__main__":
    main()
