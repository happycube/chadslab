"""umT5-XXL references for scripts/check_umt5.py: Wan 2.2's text encoder (transformers'
UMT5EncoderModel) in float32 on the CPU, through diffusers' WanPipeline._get_t5_prompt_embeds
(prompt_clean, the tokenizer padded and truncated to max_sequence_length, the last hidden state
cut to each prompt's tokens and padded back with zeros), on real prompts.

    umt5_reference.py WAN_DIR out.npz --prompt TEXT [--prompt TEXT ...] [--max-len 512]
        [--threads 12]

WAN_DIR: a Wan 2.2 diffusers directory (text_encoder/, tokenizer/). out.npz, for each prompt i:
"prompt<i>" the text, "clean<i>" the text after prompt_clean, "ids<i>" and "mask<i>" the
tokenizer's (max_len), "h<i>" the hidden states of the prompt's tokens (layers + 1, tokens,
d_model: the embeddings, then the output of each block, the last before the final norm), "emb<i>"
the pipeline's prompt_embeds (max_len, d_model, float32). "buckets" the relative position
buckets of the encoder for the distances -(max_len - 1) .. max_len - 1 (compute_bias's), "attn"
the attention implementation. --tok-texts: "tok<j>" the ids (max_len, padded) of each line of the
file after prompt_clean, "toktext<j>" the line (the tokenizer alone). Run in the generation venv."""
import argparse
import os
import time

import numpy as np
import torch
from transformers import AutoTokenizer, UMT5EncoderModel
from diffusers import WanPipeline
from diffusers.pipelines.wan.pipeline_wan import prompt_clean

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--prompt", action="append", required=True)
ap.add_argument("--max-len", type=int, default=512)
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--tok-texts", help="a file of texts (one a line) for the tokenizer alone")
a = ap.parse_args()
torch.set_num_threads(a.threads)

t0 = time.time()
te = UMT5EncoderModel.from_pretrained(os.path.join(a.model, "text_encoder"), torch_dtype=torch.float32).eval()
tok = AutoTokenizer.from_pretrained(os.path.join(a.model, "tokenizer"))
pipe = WanPipeline(tokenizer=tok, text_encoder=te, transformer=None, vae=None, scheduler=None)
print("loaded in %.1f s (%s, attention %s)" % (time.time() - t0, type(tok).__name__,
                                               te.config._attn_implementation), flush=True)
res = {"attn": np.array(te.config._attn_implementation)}

if a.tok_texts:
    for j, line in enumerate(open(a.tok_texts, encoding="utf-8").read().split("\n")):
        if not line.strip():
            continue
        line = line.replace("\\n", "\n").replace("\\t", "\t")
        res["toktext%d" % j] = np.array(line)
        res["tok%d" % j] = tok([prompt_clean(line)], padding="max_length", max_length=a.max_len, truncation=True,
                               add_special_tokens=True, return_tensors="np").input_ids[0].astype(np.int32)

# the buckets of every distance (memory - query) of a prompt of max_len tokens
att = te.encoder.block[0].layer[0].SelfAttention
rel = torch.arange(-(a.max_len - 1), a.max_len, dtype=torch.long)
res["buckets"] = att._relative_position_bucket(rel).numpy().astype(np.int32)

taps = []
hooks = [te.encoder.embed_tokens.register_forward_hook(lambda m, i, o: taps.append(o.detach()))]
for b in te.encoder.block:
    hooks.append(b.register_forward_hook(lambda m, i, o: taps.append((o[0] if isinstance(o, tuple) else o).detach())))

with torch.no_grad():
    for i, p in enumerate(a.prompt):
        t1 = time.time()
        res["prompt%d" % i] = np.array(p)
        res["clean%d" % i] = np.array(prompt_clean(p))
        t = tok([prompt_clean(p)], padding="max_length", max_length=a.max_len, truncation=True,
                add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
        n = int(t.attention_mask.sum())
        res["ids%d" % i] = t.input_ids[0].numpy().astype(np.int32)
        res["mask%d" % i] = t.attention_mask[0].numpy().astype(np.int32)
        taps.clear()
        emb = pipe._get_t5_prompt_embeds(p, max_sequence_length=a.max_len, device=torch.device("cpu"),
                                         dtype=torch.float32)
        assert len(taps) == len(te.encoder.block) + 1, len(taps)
        res["h%d" % i] = torch.stack([x[0, :n] for x in taps]).numpy().astype(np.float32)
        res["emb%d" % i] = emb[0].numpy().astype(np.float32)
        print("prompt %d: %d tokens, %.1f s" % (i, n, time.time() - t1), flush=True)
for h in hooks:
    h.remove()
np.savez(a.out, **res)
print("wrote", a.out)
