#!/usr/bin/env python3
"""The references of the music embeddings (docs/MUSIC_EMBED_PLAN.md):
MuQ-MuLan and LAION CLAP (music_audioset), in float32 on the CPU.

Run it in a venv with torch, transformers 4.48 and muq 0.1.0 (not the venv
of reclank; transformers 5 breaks MuQ's conformer). The inputs are a JSONL
file, one object a line: {"name": ..., "text": ...}, or {"name": ...,
"audio": a path (any format of ffmpeg), "start": s}: a clip of 10 s from
start, mono, at 24 kHz for MuQ-MuLan and 48 kHz for CLAP (ffmpeg). For each
input it writes <out>/<name>.npz:

    text:   clap_ids, clap_tok [n, 768] (RoBERTa's last hidden state),
            clap_emb [512]; muq_ids, muq_xlmr [n, 768] (XLM-R's last hidden
            state), muq_tok [n, 768] (after MuLan's 8 layers), muq_emb [512]
    audio:  s24, s48 (the samples); clap_mel [1001, 64], clap_emb [512];
            muq_mel [128, 1000] (normalized), muq_conv [250, 1024] (the
            subsampling), muq_hid [250, 1024] (the conformer's last hidden
            state), muq_emb [512]

Each text is embedded alone: MuQ-MuLan's mean over the rows takes the pads
of a batch too (no mask in its 8 layers), so a batched text has another
vector.

    python scripts/music_reference.py --muq models/MuQ-MuLan-large \\
        --clap models/larger_clap_music_fixed --inputs music.jsonl --out refs/
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time

import numpy as np


def decode(path, start, secs, rate):
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", "%.3f" % start, "-t", "%.3f" % secs, "-i", path,
           "-f", "f32le", "-ac", "1", "-ar", str(rate), "-"]
    x = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.float32)
    n = int(round(secs * rate))
    return np.ascontiguousarray(np.pad(x, (0, max(0, n - len(x))))[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--muq", required=True, help="the MuQ-MuLan directory (config.json, pytorch_model.bin)")
    ap.add_argument("--clap", required=True, help="the CLAP directory in the ClapModel layout")
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import torch
    from muq import MuQMuLan
    from transformers import ClapModel, ClapProcessor
    torch.set_grad_enabled(False)
    mq = MuQMuLan(config=json.load(open(os.path.join(a.muq, "config.json"))))
    r = mq.load_state_dict(torch.load(os.path.join(a.muq, "pytorch_model.bin"), map_location="cpu",
                                      weights_only=True), strict=False)
    assert not r.missing_keys and not r.unexpected_keys, r
    mq = mq.eval()
    mu = mq.mulan_module
    clap = ClapModel.from_pretrained(a.clap).eval()
    proc = ClapProcessor.from_pretrained(a.clap)
    os.makedirs(a.out, exist_ok=True)
    norm = lambda t: (t / t.norm(dim=-1, keepdim=True))[0].numpy()  # noqa: E731
    for line in open(a.inputs):
        item = json.loads(line)
        t0 = time.time()
        out = {}
        if "text" in item:
            t = item["text"]
            tok = proc.tokenizer(t, return_tensors="pt")
            o = clap.text_model(**tok)
            out.update(clap_ids=tok["input_ids"][0].numpy(), clap_tok=o.last_hidden_state[0].numpy(),
                       clap_emb=norm(clap.text_projection(o.pooler_output)))
            mt = mu.text
            ti = mt.tokenizer([t], return_tensors="pt")
            xl = mt.model(**ti).last_hidden_state
            h = mt.transformer(mt.proj(xl))
            out.update(muq_ids=ti["input_ids"][0].numpy(), muq_xlmr=xl[0].numpy(), muq_tok=h[0].numpy(),
                       muq_emb=norm(mu.text_to_latents(h.mean(-2))))
            # the whole path of muq's own code, for one text: the same vector
            ref = mu.get_text_latents(raw_texts=[t])[0].numpy()
            assert float(ref @ out["muq_emb"]) > 0.999999
        else:
            s24 = decode(item["audio"], item["start"], 10, 24000)
            s48 = decode(item["audio"], item["start"], 10, 48000)
            f = proc.feature_extractor(s48, sampling_rate=48000, return_tensors="pt")
            out.update(s24=s24, s48=s48, clap_mel=f["input_features"][0, 0].numpy(),
                       clap_emb=norm(clap.get_audio_features(**f)))
            am = mu.audio.model.model          # MuQModel
            x = torch.from_numpy(s24)[None]
            mel = am.normalize(am.preprocessing(x, features=["melspec_2048"]))["melspec_2048"]
            conv = am.conv(mel)
            hid = am.conformer(conv, output_hidden_states=True)["hidden_states"][-1]
            out.update(muq_mel=mel[0].numpy(), muq_conv=conv[0].numpy(), muq_hid=hid[0].numpy(),
                       muq_emb=norm(mu.audio_to_latents(mu.audio.transformer(mu.audio.proj(hid)).mean(-2))))
            ref = mu.get_audio_latents(x)[0].numpy()
            assert float(ref @ out["muq_emb"]) > 0.999999
        np.savez(os.path.join(a.out, item["name"] + ".npz"), **out)
        print("%-14s %.2f s" % (item["name"], time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
