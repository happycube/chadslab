#!/usr/bin/env python3
"""The reference of EmbeddingGemma 2 (google/embeddinggemma-2): the text
and audio embeddings of sentence-transformers, in float32 on the CPU.

Run it in a venv with torch, transformers >= 5.19 and sentence-transformers
>= 6.1 (not the venv of reclank). The inputs are a JSONL file, one object a
line: {"name": ..., "text": ..., "prompt": a prompt name of the model or
null}, or {"name": ..., "audio": a path (any format of ffmpeg), "start": s,
"end": s}: the clip from start to end as 16 kHz mono float32 (ffmpeg, as
reclank's decode_audio), or {"name": ..., "image": a path, "budget": the
soft tokens (70, 140, 280, 560, 1120)}. For each input it writes <out>/<name>.npz:

    ids     the token ids of the prompt and the text (<bos> ... <eos>)
    tok     [n, 768] the last hidden state (the projection to 768 of the
            final norm), before the pooling
    emb     [768] the embedding of SentenceTransformer.encode (the mean of
            tok, normalized)
    hs      [24, n, 512] (inputs of at most --hs-max tokens) hs[0] the
            embeddings, hs[i + 1] the output of layer i (to layer 22: the
            last hidden state of transformers takes the place of layer 23's)
    soft    (audio, image) [k, 512] the rows of the placeholders (embed_audio,
            embed_vision)

    python scripts/embedgemma2_reference.py --model models/embeddinggemma-2 \\
        --inputs texts.jsonl --out refs/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np


def decode(path, start, end):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", path, "-f", "f32le", "-ac", "1", "-ar", "16000", "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)
    return np.ascontiguousarray(x[int(round(start * 16000)):int(round(end * 16000))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hs-max", type=int, default=3000)
    a = ap.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_grad_enabled(False)
    st = SentenceTransformer(a.model, device="cpu", model_kwargs={"torch_dtype": torch.float32})
    hf = st[0].model
    os.makedirs(a.out, exist_ok=True)
    for line in open(a.inputs):
        item = json.loads(line)
        t0 = time.time()
        if "image" in item:
            from PIL import Image
            img = Image.open(item["image"]).convert("RGB")
            budget = int(item.get("budget", 280))
            proc = st[0].processor
            f = proc(text=["<|image|>"], images=[[img]], return_tensors="pt",
                     images_kwargs={"max_soft_tokens": budget})
            out = hf(input_ids=f["input_ids"], pixel_values=f["pixel_values"],
                     image_position_ids=f["image_position_ids"], attention_mask=f["attention_mask"])
            soft = torch.cat(hf.get_image_features(f["pixel_values"], f["image_position_ids"],
                                                   return_dict=True).pooler_output).float().numpy()
            tok = out.last_hidden_state[0].float().numpy()
            emb = tok.mean(0)
            emb /= np.linalg.norm(emb)
            check = ""
            if budget == 280:       # the default of sentence-transformers: its vector too
                e2 = st.encode([{"image": img}], convert_to_numpy=True)[0]
                check = ", cos with SentenceTransformer.encode %.7f" % float(e2 @ emb)
            np.savez(os.path.join(a.out, item["name"] + ".npz"), ids=f["input_ids"][0].numpy().astype(np.int32),
                     tok=tok, emb=emb.astype(np.float32), soft=soft)
            print("%s: budget %d, %d tokens, %.1f s%s" % (item["name"], budget, tok.shape[0], time.time() - t0,
                                                          check), flush=True)
            continue
        if "audio" in item:
            x = decode(item["audio"], item["start"], item["end"])
            inp = {"audio": {"array": x, "sampling_rate": 16000}}
            f = st.preprocess([inp])
            emb = st.encode([inp], convert_to_numpy=True)[0]
            out = hf(input_ids=f["input_ids"], input_features=f["input_features"],
                     input_features_mask=f["input_features_mask"], attention_mask=f["attention_mask"])
            au = hf.get_audio_features(f["input_features"], f["input_features_mask"], return_dict=True)
            soft = au.pooler_output[au.attention_mask].float().numpy()
            tok = out.last_hidden_state[0].float().numpy()
            mean = tok.mean(0)
            mean /= np.linalg.norm(mean)
            np.savez(os.path.join(a.out, item["name"] + ".npz"), ids=f["input_ids"][0].numpy().astype(np.int32),
                     tok=tok, emb=emb.astype(np.float32), soft=soft, samples=x)
            print("%s: %.2f s of audio, %d tokens, %.1f s, cos(mean of tok, emb) %.7f" % (
                item["name"], len(x) / 16000, tok.shape[0], time.time() - t0, float(mean @ emb)), flush=True)
            continue
        prompt = item.get("prompt")
        text = (st.prompts[prompt] if prompt else "") + item["text"]
        ids = st.tokenizer(text, return_tensors="pt")["input_ids"]
        emb = st.encode([item["text"]], prompt_name=prompt, convert_to_numpy=True)[0]
        n = ids.shape[1]
        out = hf(input_ids=ids, output_hidden_states=n <= a.hs_max)
        tok = out.last_hidden_state[0].float().numpy()
        mean = tok.mean(0)
        mean /= np.linalg.norm(mean)
        d = dict(ids=ids[0].numpy().astype(np.int32), tok=tok, emb=emb.astype(np.float32))
        if n <= a.hs_max:
            d["hs"] = np.stack([h[0].float().numpy() for h in out.hidden_states[:-1]])
        np.savez(os.path.join(a.out, item["name"] + ".npz"), **d)
        print("%s: %d tokens, %.1f s, cos(mean of tok, emb) %.7f" % (
            item["name"], n, time.time() - t0, float(mean @ emb)), flush=True)


if __name__ == "__main__":
    main()
