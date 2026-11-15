"""Higgs TTS 3 voice cloning references for scripts/check_higgs_clone.py: the codec's encoder
(transformers' HiggsAudioV2TokenizerModel.encode, the weights bundled in the checkpoint) of a
reference clip with the intermediates of its stages, and (with --text) the greedy codes of the
clone prompt (sglang-omni's higgs_tts/text_tokenizer.py build_prompt: <|tts|> <|ref_text|> the
transcript <|ref_audio|> the delayed reference codes <|text|> the text <|audio|>; the reference
codes' rows the sum of their codebooks' embedding rows, as HiggsFusedMultiTextEmbedding), as
higgs_reference.py: transformers' Qwen3 with its RoPE's cos and sin rounded to bfloat16, float32
on the CPU.

The clone prompt written from SGLang-Omni (sglang_omni/models/higgs_tts/text_tokenizer.py;
Apache License 2.0; Copyright 2025-2026 SGLang-Omni Team). See THIRD_PARTY_NOTICES.md.

    higgs_clone_reference.py MODEL_DIR CODEC_CONFIG out.npz --ref-wav clip.wav [--ref-text T]
        [--text TEXT] [--max-frames 600] [--wav out.wav]

clip.wav: 24 kHz mono 16-bit (else resampled to 24 kHz by torchaudio, as sglang-omni's
encode_reference; a clip under 1 s padded with zeros to 1 s).
out.npz: "wav" the 24 kHz input (float32), "wav16" its 16 kHz resample, "extract" the
HuBERT feature extractor's rows (frames, 512), "hub0" and "hublast" the first and the last of
its hidden states, "nhid" their count, "semantic" the mean of the hidden states after the
downsampling (the encoder's semantic input), "e_semantic" (T, 768) and "e_acoustic" (T, 256)
the encoders' outputs, "emb" (T, 1024) fc's, "ref_codes" (T, 8) the RVQ codes. With --text:
"ids" the prompt (-100 at the reference codes' rows), "ref_delayed" (T + 7, 8), "codes" the
delayed codes (L, 8) of the greedy steps, "logits" (L, 8, 1026), "raw" (L - 7, 8) the codes
with the delay undone, "audio" the 24 kHz waveform. Run in the generation venv."""
import argparse
import json
import os
import wave

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from safetensors import safe_open
from transformers import (AutoTokenizer, HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel, Qwen3Config,
                          Qwen3ForCausalLM)

BOC, EOC = 1024, 1025
PLACEHOLDER = -100
ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("codec_config")
ap.add_argument("out")
ap.add_argument("--ref-wav", required=True)
ap.add_argument("--ref-text")
ap.add_argument("--text")
ap.add_argument("--max-frames", type=int, default=600)
ap.add_argument("--wav")
a = ap.parse_args()
torch.set_num_threads(len(os.sched_getaffinity(0)))
cfg = json.load(open(os.path.join(a.model, "config.json")))
where = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
CODEC = "tied.embedding.modality_embeddings.0.model."
state, codec_state, mm = {}, {}, None
for shard in sorted(set(where.values())):
    with safe_open(os.path.join(a.model, shard), framework="pt") as f:
        for n in f.keys():
            if n.startswith(CODEC):
                codec_state[n[len(CODEC):]] = f.get_tensor(n).float()
            elif a.text is None:
                continue
            elif n == "tied.embedding.modality_embeddings.0.embedding.weight":
                mm = f.get_tensor(n).float()
            elif n == "tied.embedding.text_embedding.weight":
                state["model.embed_tokens.weight"] = f.get_tensor(n).float()
            elif n.startswith("body."):
                state["model." + n[len("body."):]] = f.get_tensor(n).float()

with wave.open(a.ref_wav) as w:
    assert w.getnchannels() == 1 and w.getsampwidth() == 2
    sr = w.getframerate()
    x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768
wav = torch.from_numpy(x)[None, None]
if sr != 24000:
    wav = torchaudio.functional.resample(wav, sr, 24000)
if wav.shape[-1] < 24000:
    wav = F.pad(wav, (0, 24000 - wav.shape[-1]))
res = {"wav": wav[0, 0].numpy().astype(np.float32)}

ccfg = HiggsAudioV2TokenizerConfig.from_json_file(a.codec_config)
codec = HiggsAudioV2TokenizerModel(ccfg).float().eval()
miss, _ = codec.load_state_dict(codec_state, strict=False)
print("codec: %d tensors, %d missing %s" % (len(codec_state), len(miss), miss[:4]), flush=True)
got = {}


def keep(name, fn=lambda o: o):
    def hook(mod, inp, out):
        got[name] = fn(out)
    return hook


codec.semantic_model.feature_extractor.register_forward_hook(keep("extract", lambda o: o[0].T))
codec.semantic_model.register_forward_hook(keep("hidden", lambda o: o.hidden_states))
codec.encoder_semantic.register_forward_hook(keep("e_semantic", lambda o: o[0].T))
codec.acoustic_encoder.register_forward_hook(keep("e_acoustic", lambda o: o[0].T))
codec.fc.register_forward_hook(keep("emb", lambda o: o[0]))
with torch.no_grad():
    codes = codec.encode(wav).audio_codes[0].T                      # (T, 8)
    res["wav16"] = torchaudio.functional.resample(wav, 24000, 16000)[0, 0].numpy()
    hid = got["hidden"]
    sem = torch.stack(list(hid), 1).mean(1)[0, ::codec.config.semantic_downsample_factor]
res.update(extract=got["extract"].numpy(), hub0=hid[0][0].numpy(), hublast=hid[-1][0].numpy(),
           nhid=np.array(len(hid)), semantic=sem.numpy(), e_semantic=got["e_semantic"].numpy(),
           e_acoustic=got["e_acoustic"].numpy(), emb=got["emb"].numpy(), ref_codes=codes.numpy().astype(np.int32))
print("encode: %d samples, %d hidden states, %d frames" % (wav.shape[-1], len(hid), codes.shape[0]), flush=True)

if a.text is not None:
    tc = dict(cfg["text_config"])
    tc["rope_theta"] = (tc.get("rope_parameters") or {}).get("rope_theta") or tc.get("rope_theta") or 1e6
    model = Qwen3ForCausalLM(Qwen3Config(**tc)).float().eval()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not [m for m in missing if m != "lm_head.weight"] and not unexpected, (missing, unexpected)
    rot = model.model.rotary_emb
    fwd = rot.forward

    def rope_bf16(*args, **kw):
        cos, sin = fwd(*args, **kw)
        return cos.to(torch.bfloat16).float(), sin.to(torch.bfloat16).float()

    rot.forward = rope_bf16
    tok = AutoTokenizer.from_pretrained(a.model)
    sp = {t: tok.convert_tokens_to_ids(t) for t in ("<|tts|>", "<|ref_text|>", "<|ref_audio|>", "<|text|>",
                                                     "<|audio|>")}
    N, V = cfg["audio_encoder_config"]["num_codebooks"], cfg["audio_encoder_config"]["vocab_size"]
    T = codes.shape[0]
    # sglang-omni's apply_delay_pattern: codebook c's row t + c is frame t's, BOC before, EOC after
    delayed = torch.full((T + N - 1, N), EOC, dtype=torch.long)
    t_idx = torch.arange(T + N - 1)
    for c in range(N):
        delayed[t_idx < c, c] = BOC
        delayed[c:c + T, c] = codes[:, c]
    ids = [sp["<|tts|>"]]
    if a.ref_text:
        ids += [sp["<|ref_text|>"]] + tok.encode(a.ref_text, add_special_tokens=False)
    ids += [sp["<|ref_audio|>"]] + [PLACEHOLDER] * len(delayed)
    ids += [sp["<|text|>"]] + tok.encode(a.text, add_special_tokens=False) + [sp["<|audio|>"]]
    offsets = torch.arange(N) * V
    res.update(ids=np.array(ids, np.int32), text=np.array(a.text), ref_text=np.array(a.ref_text or ""),
               ref_delayed=delayed.numpy().astype(np.int32))
    out_codes, logits = [], []
    with torch.no_grad():
        idt = torch.tensor(ids)
        hole = idt == PLACEHOLDER
        emb = model.model.embed_tokens(torch.where(hole, 0, idt))
        emb[hole] = mm[delayed + offsets].sum(1)
        out = model.model(inputs_embeds=emb[None], use_cache=True)
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
            out_codes.append(c.numpy().astype(np.int32))
            if done:
                break
            e = mm[c + offsets].sum(0)[None, None]
            out = model.model(inputs_embeds=e, past_key_values=past, use_cache=True)
            past, h = out.past_key_values, out.last_hidden_state[0, -1]
        print("%d steps (%s)" % (len(out_codes), "done" if done else "max frames"), flush=True)
        dl = np.stack(out_codes)
        rows = len(dl) - (N - 1)
        raw = np.stack([dl[c:c + rows, c] for c in range(N)], 1)
        raw = np.where(raw >= V - 2, 0, raw)
        audio = codec.decode(torch.from_numpy(raw.T.astype(np.int64))[None]).audio_values[0, 0].numpy()
    res.update(codes=dl, logits=np.stack(logits), raw=raw.astype(np.int32), audio=audio.astype(np.float32))
    if a.wav:
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2")
        with wave.open(a.wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(24000)
            w.writeframes(pcm.tobytes())
np.savez(a.out, **res)
print("saved %s" % a.out)
