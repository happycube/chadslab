"""Higgs TTS 3 references for scripts/check_higgs.py: the backbone (transformers' Qwen3 with
the checkpoint's weights, its RoPE's cos and sin rounded to bfloat16 as the model was
trained), greedy codes through the delay pattern's state machine (sglang-omni's
higgs_tts/sampler.py: step), and the codec's decode (transformers'
HiggsAudioV2TokenizerModel, the weights bundled in the checkpoint), in float32 on the CPU.

The prompt and the sampler's state machine written from SGLang-Omni
(sglang_omni/models/higgs_tts/sampler.py; Apache License 2.0; Copyright 2025-2026 SGLang-Omni Team).
See THIRD_PARTY_NOTICES.md.

    higgs_reference.py MODEL_DIR CODEC_CONFIG out.npz --text TEXT [--max-frames 600] [--wav out.wav]

out.npz: "ids" the prompt (<|tts|> <|text|> text <|audio|>), "codes" the delayed codes
(L, 8) of the steps, "logits" (L, 8, 1026) the head's logits of each step, "raw" the codes
(L - 7, 8) after the delay is undone (BOC/EOC -> 0), "audio" the 24 kHz waveform.
CODEC_CONFIG: sglang-omni's higgs_tts/configs/higgs_audio_v2_tokenizer.json (the same as
bosonai/higgs-audio-v2-tokenizer's config.json). Run in the generation venv."""
import argparse
import json
import os
import wave

import numpy as np
import torch
from safetensors import safe_open
from transformers import (AutoTokenizer, HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel, Qwen3Config,
                          Qwen3ForCausalLM)

BOC, EOC = 1024, 1025
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("codec_config")
ap.add_argument("out")
ap.add_argument("--text", required=True)
ap.add_argument("--max-frames", type=int, default=600)
ap.add_argument("--wav")
a = ap.parse_args()
torch.set_num_threads(46)
cfg = json.load(open(os.path.join(a.model, "config.json")))
tc = dict(cfg["text_config"])
tc["rope_theta"] = (tc.get("rope_parameters") or {}).get("rope_theta") or tc.get("rope_theta") or 1e6
model = Qwen3ForCausalLM(Qwen3Config(**tc)).float().eval()
where = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
state, codec_state, mm = {}, {}, None
CODEC = "tied.embedding.modality_embeddings.0.model."
for shard in sorted(set(where.values())):
    with safe_open(os.path.join(a.model, shard), framework="pt") as f:
        for n in f.keys():
            t = f.get_tensor(n).float()
            if n.startswith(CODEC):
                codec_state[n[len(CODEC):]] = t
            elif n == "tied.embedding.modality_embeddings.0.embedding.weight":
                mm = t
            elif n == "tied.embedding.text_embedding.weight":
                state["model.embed_tokens.weight"] = t
            elif n.startswith("body."):
                state["model." + n[len("body."):]] = t
missing, unexpected = model.load_state_dict(state, strict=False)
assert not [m for m in missing if m != "lm_head.weight"] and not unexpected, (missing, unexpected)
rot = model.model.rotary_emb
fwd = rot.forward


def rope_bf16(*args, **kw):
    cos, sin = fwd(*args, **kw)
    return cos.to(torch.bfloat16).float(), sin.to(torch.bfloat16).float()


rot.forward = rope_bf16
tok = AutoTokenizer.from_pretrained(a.model)
sp = {t: tok.convert_tokens_to_ids(t) for t in ("<|tts|>", "<|text|>", "<|audio|>")}
ids = [sp["<|tts|>"], sp["<|text|>"]] + tok.encode(a.text, add_special_tokens=False) + [sp["<|audio|>"]]
N, V = cfg["audio_encoder_config"]["num_codebooks"], cfg["audio_encoder_config"]["vocab_size"]
offsets = torch.arange(N) * V
res = {"ids": np.array(ids, np.int32), "text": np.array(a.text)}
codes, logits = [], []
with torch.no_grad():
    emb = model.model.embed_tokens(torch.tensor([ids]))
    out = model.model(inputs_embeds=emb, use_cache=True)
    past, h = out.past_key_values, out.last_hidden_state[0, -1]
    delay, countdown, done = 0, None, False
    for step in range(a.max_frames):
        lg = (h @ mm.T).reshape(N, V)
        c = lg.argmax(-1)
        logits.append(lg.numpy())
        if delay < N:
            if delay + 1 < N:
                c[delay + 1:] = BOC
            delay += 1
        elif countdown is not None:
            countdown -= 1
            done = countdown <= 0
        elif int(c[0]) == EOC:
            countdown = N - 2
        codes.append(c.numpy().astype(np.int32))
        if done:
            break
        e = mm[c + offsets].sum(0)[None, None]
        out = model.model(inputs_embeds=e, past_key_values=past, use_cache=True)
        past, h = out.past_key_values, out.last_hidden_state[0, -1]
    print("%d steps (%s)" % (len(codes), "done" if done else "max frames"), flush=True)
    delayed = np.stack(codes)
    rows = len(delayed) - (N - 1)
    raw = np.stack([delayed[c:c + rows, c] for c in range(N)], 1)
    raw = np.where(raw >= V - 2, 0, raw)
    ccfg = HiggsAudioV2TokenizerConfig.from_json_file(a.codec_config)
    codec = HiggsAudioV2TokenizerModel(ccfg).float().eval()
    miss, _ = codec.load_state_dict(codec_state, strict=False)
    print("codec: %d tensors, %d missing" % (len(codec_state), len(miss)), flush=True)
    audio = codec.decode(torch.from_numpy(raw.T.astype(np.int64))[None]).audio_values[0, 0].numpy()
res.update(codes=delayed, logits=np.stack(logits), raw=raw.astype(np.int32), audio=audio.astype(np.float32))
np.savez(a.out, **res)
if a.wav:
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
    with wave.open(a.wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(pcm.tobytes())
print("saved %s: %d frames, %.2f s of audio" % (a.out, len(raw), len(audio) / 24000))
