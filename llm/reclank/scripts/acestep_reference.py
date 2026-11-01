"""ACE-Step 1.5 (turbo) references for scripts/check_acestep.py: the authors' pipeline
(github.com/ace-step/ACE-Step-1.5, acestep.inference.generate_music) without the 5 Hz LM
(thinking off), in float32 on the CPU, its parts saved by hooks.

    acestep_reference.py SRC_DIR out.npz --caption TEXT [--lyrics TEXT] [--duration 30]
        [--bpm 120] [--key "C major"] [--seed 42] [--language en] [--wav out.wav]

SRC_DIR: the clone of the repository (its acestep package); ACESTEP_CHECKPOINTS_DIR: the
weights (ACE-Step/Ace-Step1.5). out.npz: "text_ids", "lyric_ids"; "text_hidden" (Qwen3-
Embedding's last hidden state), "lyric_hidden" (its embedding rows); "enc", "enc_mask" (the
packed conditions), "context" (the context latents), "noise"; for each step i "x<i>",
"t<i>", "v<i>" (the DiT's input, timestep, output); "latent" (the steps' result); "audio"
(the VAE's waveform, channels x samples, 48 kHz). --thinking: the 5 Hz LM first (backend "pt",
float32): "lm_metadata", "lm_codes", and each LM forward's input ids and last logits
("lm_ids<i>", "lm_logits<i>"). Run in a venv with transformers < 4.58."""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--caption", required=True)
ap.add_argument("--lyrics", default="[Instrumental]")
ap.add_argument("--duration", type=float, default=30.0)
ap.add_argument("--bpm", type=int, default=None)
ap.add_argument("--key", default="")
ap.add_argument("--language", default="en")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--wav")
ap.add_argument("--thinking", action="store_true", help="the 5 Hz LM first (the authors' default)")
a = ap.parse_args()
sys.path.insert(0, a.src)
torch.set_num_threads(46)
from acestep.handler import AceStepHandler  # noqa: E402
from acestep.inference import GenerationConfig, GenerationParams, generate_music  # noqa: E402

h = AceStepHandler()
msg = h.initialize_service(project_root=a.src, config_path="acestep-v15-turbo", device="cpu", use_mlx_dit=False)
print(msg, flush=True)
h.dtype = torch.float32
for name in ("model", "vae", "text_encoder"):
    m = getattr(h, name, None)
    if m is not None:
        m.float()
if getattr(h, "silence_latent", None) is not None:
    h.silence_latent = h.silence_latent.float()
model = h.model
lm = None
lm_calls = []
if a.thinking:
    from acestep.llm_inference import LLMHandler  # noqa: E402
    lm = LLMHandler()
    print(lm.initialize(checkpoint_dir=os.environ["ACESTEP_CHECKPOINTS_DIR"], lm_model_path="acestep-5Hz-lm-1.7B",
                        backend="pt", device="cpu", dtype=torch.float32), flush=True)
    lfw = lm.llm.forward

    def lm_forward(*args, **kw):
        out = lfw(*args, **kw)
        ids = kw.get("input_ids", args[0] if args else None)
        lm_calls.append((ids.detach().numpy().astype(np.int32) if ids is not None else None,
                         out.logits[:, -1].detach().float().numpy()))
        return out

    lm.llm.forward = lm_forward
    gsc = lm.generate_with_stop_condition

    def generate_with_stop_condition(**kw):
        out = gsc(**kw)
        print("LM:", {k: (v if not isinstance(v, (list, str)) or len(v) < 400 else "%s..." % str(v)[:400])
                      for k, v in out.items() if k != "extra_outputs"}, flush=True)
        res["lm_metadata"] = np.array(repr(out.get("metadata")))
        res["lm_codes"] = np.array(str(out.get("audio_codes")))
        return out

    lm.generate_with_stop_condition = generate_with_stop_condition
res = {"caption": np.array(a.caption), "lyrics": np.array(a.lyrics)}
steps = []

ga = model.generate_audio


def generate_audio(**kw):
    for k in ("text_hidden_states", "text_attention_mask", "lyric_hidden_states", "lyric_attention_mask",
              "refer_audio_acoustic_hidden_states_packed", "refer_audio_order_mask", "src_latents", "chunk_masks",
              "is_covers", "silence_latent"):
        v = kw.get(k)
        if isinstance(v, torch.Tensor):
            res["in_" + k] = v.detach().float().numpy() if v.is_floating_point() else v.detach().numpy()
    for k in ("shift", "infer_steps", "timesteps", "seed", "infer_method", "dcw_enabled", "dcw_mode", "dcw_scaler",
              "dcw_high_scaler", "dcw_wavelet", "sampler_mode"):
        if k in kw and kw[k] is not None and not isinstance(kw[k], torch.Tensor):
            res["arg_" + k] = np.array(kw[k])
    print("generate_audio args:", {k: (v if not isinstance(v, torch.Tensor) else tuple(v.shape))
                                   for k, v in kw.items() if not k.startswith("non_cover")}, flush=True)
    out = ga(**kw)
    res["latent"] = out["target_latents"][0].detach().float().numpy()
    return out


model.generate_audio = generate_audio
pc = model.prepare_condition


def prepare_condition(**kw):
    enc, mask, ctx = pc(**kw)
    if "enc" not in res:
        res["enc"], res["enc_mask"] = enc[0].float().numpy(), mask[0].numpy()
        res["context"] = ctx[0].float().numpy()
    return enc, mask, ctx


model.prepare_condition = prepare_condition
pn = model.prepare_noise


def prepare_noise(*args, **kw):
    n = pn(*args, **kw)
    res.setdefault("noise", n[0].float().numpy())
    return n


model.prepare_noise = prepare_noise
dfw = model.decoder.forward


def decoder_forward(*args, **kw):
    out = dfw(*args, **kw)
    i = len(steps)
    res["x%d" % i] = kw["hidden_states"][0].float().numpy()
    res["t%d" % i] = np.array(float(kw["timestep"][0]))
    res["v%d" % i] = out[0][0].float().numpy()
    steps.append(i)
    return out


model.decoder.forward = decoder_forward
te = h.text_encoder
tef = te.forward


def text_forward(*args, input_ids=None, **kw):
    out = tef(*args, input_ids=input_ids, **kw)
    res.setdefault("text_ids", input_ids[0].numpy().astype(np.int32))
    res.setdefault("text_hidden", out.last_hidden_state[0].float().numpy())
    return out


te.forward = text_forward
emb = te.embed_tokens
ef = emb.forward


def embed_forward(ids):
    out = ef(ids)
    if ids.dim() == 2 and "lyric_ids" not in res and "text_ids" in res:
        res["lyric_ids"] = ids[0].numpy().astype(np.int32)
        res["lyric_hidden"] = out[0].float().numpy()
    return out


emb.forward = embed_forward
vdec = h.vae.decode


def vae_decode(z, *args, **kw):
    out = vdec(z, *args, **kw)
    sample = out.sample if hasattr(out, "sample") else out[0]
    res.setdefault("vae_in", z[0].float().numpy())
    res.setdefault("audio", sample[0].float().numpy())
    return out


h.vae.decode = vae_decode
cot = dict(use_cot_metas=False, use_cot_caption=False, use_cot_language=False) if not a.thinking else {}
params = GenerationParams(task_type="text2music", caption=a.caption, lyrics=a.lyrics, duration=a.duration,
                          bpm=a.bpm, keyscale=a.key, vocal_language=a.language, thinking=a.thinking, seed=a.seed,
                          **cot)
config = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[a.seed], audio_format="wav")
with tempfile.TemporaryDirectory() as d:
    out = generate_music(h, lm, params, config, save_dir=d)
    print("result:", getattr(out, "status_message", None) or getattr(out, "error", None), flush=True)
    files = [f for f in os.listdir(d)]
    print("files:", files)
    if a.wav and files:
        import shutil
        shutil.copy(os.path.join(d, [f for f in files if f.endswith(".wav")][0]), a.wav)
if lm_calls:
    # each LM forward: its input ids (the prompt, then one token a step) and its last row's logits
    res["lm_n"] = np.array(len(lm_calls))
    for i, (ids, lg) in enumerate(lm_calls):
        res["lm_ids%d" % i], res["lm_logits%d" % i] = ids, lg.astype(np.float32)
np.savez(a.out, **res)
print("saved", a.out, sorted(res))
