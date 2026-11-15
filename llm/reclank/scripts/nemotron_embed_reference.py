#!/usr/bin/env python3
"""The reference of Nemotron 3.5 Embed (1B, 8B) and Nemotron 3.5 Rerank
(8B): transformers in float32 on the CPU, as the model cards run them
(AutoProcessor.apply_chat_template, then the model).

Run it in a venv with torch, torchvision, pillow and transformers >= 5.20
(not the venv of reclank). The reranker's class is the checkpoint's model.py
(trust_remote_code): run it in the sandbox (work/gen/sandbox/sbx.sh) with the
model directory bound read-only.

The inputs are a JSONL file, one object a line:

    embed:  {"name", "kind": "query" | "document", "text": ..., "image": a path,
             "longest_edge": the resize of the image (default the processor's 1120)}
    rerank: {"name", "query": ..., "text": ..., "image": ..., "longest_edge": ...}

(a text, an image, or both; a query of embed is text alone). --image-dir
takes the place of the directory of each image path (the sandbox's binds).
For each input it writes <out>/<name>.npz:

    ids     the token ids of the template (no BOS, no EOS)
    tok     [n, width] the last hidden state
    emb     (embed) [width] the mean of tok, L2-normalized
    pooled  (rerank) [width] the mean of tok; score: the logit (model.py)
    soft    (an image) [k, width] the rows of the [IMG] tokens (the projector's)
    pixels  (an image) [3, h, w] the processor's pixel values
    hs      [layers + 1, n, width] (at most --hs-max tokens) the input of each
            layer and the last hidden state

--attn eager (the default) runs the attention with a mask (none: every key):
transformers' sdpa with no mask takes the attention module's is_causal.

    python scripts/nemotron_embed_reference.py --model DIR --inputs in.jsonl --out refs/ [--rerank]
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rerank", action="store_true", help="the reranker (model.py: trust_remote_code)")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--hs-max", type=int, default=600)
    ap.add_argument("--attn", default="eager", help="eager, sdpa")
    ap.add_argument("--only", default=None, help="names, comma-separated")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoModelForSequenceClassification, AutoProcessor
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    proc = AutoProcessor.from_pretrained(a.model)
    t0 = time.time()
    if a.rerank:
        model = AutoModelForSequenceClassification.from_pretrained(
            a.model, trust_remote_code=True, dtype=torch.float32, attn_implementation=a.attn).eval()
        base = model.model
    else:
        model = base = AutoModel.from_pretrained(a.model, dtype=torch.float32, attn_implementation=a.attn).eval()
    print("loaded in %.1f s (%s)" % (time.time() - t0, type(model).__name__), flush=True)
    seen = {}
    # the rows of the [IMG] tokens: the projector's output
    base.multi_modal_projector.register_forward_hook(lambda m, i, o: seen.__setitem__("soft", o))
    if a.rerank:
        base.register_forward_hook(lambda m, i, o: seen.__setitem__("out", o))
    only = set(a.only.split(",")) if a.only else None
    os.makedirs(a.out, exist_ok=True)
    for line in open(a.inputs):
        item = json.loads(line)
        if only and item["name"] not in only:
            continue
        t0 = time.time()
        seen.clear()
        img = None
        if item.get("image"):
            path = item["image"]
            if a.image_dir:
                path = os.path.join(a.image_dir, os.path.basename(path))
            img = Image.open(path).convert("RGB")
        kw = {}
        if img is not None and item.get("longest_edge"):
            kw["processor_kwargs"] = {"images_kwargs": {"size": {"longest_edge": int(item["longest_edge"])}}}
        if a.rerank:
            content = []
            if item.get("text") is not None:
                content.append({"type": "text", "text": item["text"]})
            if img is not None:
                content.append({"type": "image", "image": img})
            msgs = [{"role": "query", "content": [{"type": "text", "text": item["query"]}]},
                    {"role": "document", "content": content}]
        else:
            content = [] if img is None else [{"type": "image", "image": img}]
            if item.get("text") is not None:
                content.append({"type": "text", "text": item["text"]})
            msgs = [{"role": "system", "content": "query:" if item["kind"] == "query" else "passage:"},
                    {"role": "user", "content": content}]
        f = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False, return_dict=True,
                                     return_tensors="pt", **kw)
        n = f["input_ids"].shape[1]
        args = dict(input_ids=f["input_ids"], attention_mask=f["attention_mask"])
        if "pixel_values" in f:
            args.update(pixel_values=f["pixel_values"].float(), image_sizes=f["image_sizes"])
        d = dict(ids=f["input_ids"][0].numpy().astype(np.int32))
        if a.rerank:
            out = model(**args, output_hidden_states=n <= a.hs_max)
            d["score"] = np.float32(out.logits.reshape(-1)[0].item())
            hidden = seen["out"]
        else:
            hidden = model(**args, output_hidden_states=n <= a.hs_max)
        tok = hidden.last_hidden_state[0].float()
        d["tok"] = tok.numpy()
        mean = tok.mean(0).numpy()
        if a.rerank:
            d["pooled"] = mean
        else:
            d["emb"] = (mean / np.linalg.norm(mean)).astype(np.float32)
        if n <= a.hs_max and hidden.hidden_states is not None:
            d["hs"] = np.stack([h[0].float().numpy() for h in hidden.hidden_states])
        if "soft" in seen:
            d["soft"] = seen["soft"].reshape(-1, seen["soft"].shape[-1]).float().numpy()
            d["pixels"] = f["pixel_values"][0].float().numpy()
            d["image_size"] = np.asarray(f["image_sizes"][0], np.int32)
        np.savez(os.path.join(a.out, item["name"] + ".npz"), **d)
        print("%s: %d tokens%s, %.1f s%s" % (
            item["name"], n, ", image %s" % "x".join(str(int(s)) for s in d["image_size"]) if "soft" in seen else "",
            time.time() - t0, ", score %.6f" % d["score"] if a.rerank else ""), flush=True)


if __name__ == "__main__":
    main()
