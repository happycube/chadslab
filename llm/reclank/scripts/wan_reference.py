"""Wan 2.2 TI2V-5B and T2V-A14B references for scripts/check_wan.py: diffusers' WanPipeline (text
to video; the 5B with expand_timesteps: a timestep a token, all the same without an image; the
A14B, a model_index.json with boundary_ratio, with its two experts: transformer for the steps of
t >= boundary_ratio * 1000, transformer_2 for the others, at --guidance and --guidance2) in
float32 on the CPU, its parts saved on real prompts.

The loop written from diffusers (src/diffusers/pipelines/wan/pipeline_wan.py; Apache License 2.0;
Copyright 2025 The Wan Team and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    wan_reference.py MODEL_DIR EMBEDS OUT_DIR [--keys fox,waves] [--size 480x832] [--frames 17]
        [--steps 25] [--seed 42] [--guidance 5.0] [--guidance2 G] [--threads 11] [--no-vae] [--taps]

EMBEDS: a torch file of the pipeline's prompt embeddings (encode_prompt: umT5-XXL, 512 rows,
zeros past the prompt's tokens) {key: (1, 512, 4096), "neg": the negative prompt's}. For each key
OUT_DIR/KEY.npz: "emb", "nemb" (512, 4096); "noise" (48, F, H/16, W/16) of the seed (the
pipeline's prepare_latents); "sigmas", "timesteps"; for each step i "x<i>" the DiT's input
latents, "vc<i>" and "vu<i>" its outputs of the prompt and of the negative prompt, "v<i>" the
guided output; "experts" each step's expert (0 transformer, 1 transformer_2; the A14B); "latent" the last latents; "z" the VAE's input (latents / (1 / std) + mean, as the
pipeline), "video" the VAE's output (3, frames, H, W) in [-1, 1] (clamped, as decode). The loop is
the pipeline's (__call__ of pipeline_wan.py) written out, so the outputs of each call are kept.
--taps: OUT_DIR/taps.npz, the DiT's intermediate tensors of the first key's first call (the
prompt at step 0): the RoPE tables, the patch embedding, temb, timestep_proj, the text rows, the
blocks' outputs, the last norm's rows; and of the VAE decoder's first two chunks (the latent
frames 0 and 1): the outputs of conv_in, the mid block and each up block. Run in the generation
venv (diffusers from git)."""
import argparse
import json
import os
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("embeds")
ap.add_argument("out")
ap.add_argument("--keys", default="fox,waves")
ap.add_argument("--size", default="480x832")
ap.add_argument("--frames", type=int, default=17)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--guidance", type=float, default=5.0)
ap.add_argument("--guidance2", type=float, default=None, help="the A14B's low-noise expert's (default: --guidance)")
ap.add_argument("--threads", type=int, default=11)
ap.add_argument("--no-vae", action="store_true")
ap.add_argument("--taps", action="store_true")
a = ap.parse_args()
torch.set_num_threads(a.threads)
from diffusers import AutoencoderKLWan, UniPCMultistepScheduler, WanPipeline, WanTransformer3DModel  # noqa: E402

H, W = (int(v) for v in a.size.split("x"))
os.makedirs(a.out, exist_ok=True)
t0 = time.time()
ratio = json.load(open(os.path.join(a.model, "model_index.json"))).get("boundary_ratio")
tr = WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer", torch_dtype=torch.float32).eval()
tr2 = None if ratio is None else \
    WanTransformer3DModel.from_pretrained(a.model, subfolder="transformer_2", torch_dtype=torch.float32).eval()
vae = None if a.no_vae else AutoencoderKLWan.from_pretrained(a.model, subfolder="vae", torch_dtype=torch.float32).eval()
sch = UniPCMultistepScheduler.from_pretrained(a.model, subfolder="scheduler")
pipe = WanPipeline(tokenizer=None, text_encoder=None, transformer=tr, transformer_2=tr2, vae=vae, scheduler=sch,
                   boundary_ratio=ratio, expand_timesteps=ratio is None)
# (the 5B's VAE 16 x 16 x 4, the A14B's Wan 2.1 VAE 8 x 8 x 4: the pipeline reads them from the VAE)
pipe.vae_scale_factor_spatial, pipe.vae_scale_factor_temporal = (16 if ratio is None else 8), 4
boundary = None if ratio is None else ratio * sch.config.num_train_timesteps
g2 = a.guidance if a.guidance2 is None else a.guidance2
print("loaded %.1f s" % (time.time() - t0), flush=True)
emb = torch.load(a.embeds)
taps = {}


def tap(name, keep):
    def hook(mod, args, out):
        if keep():
            o = out[0] if isinstance(out, tuple) else out
            taps.setdefault(name, []).append(o.detach().clone() if torch.is_tensor(o) else o)
    return hook


with torch.no_grad():
    for n_key, key in enumerate(a.keys.split(",")):
        pe, ne = emb[key], emb["neg"]
        res = {"emb": pe[0].numpy(), "nemb": ne[0].numpy()}
        g = torch.Generator("cpu").manual_seed(a.seed)
        lat = pipe.prepare_latents(1, tr.config.in_channels, H, W, a.frames, torch.float32, torch.device("cpu"), g)
        res["noise"] = lat[0].numpy()
        sch.set_timesteps(a.steps, device="cpu")
        ts = sch.timesteps
        res["sigmas"], res["timesteps"] = sch.sigmas.numpy(), ts.numpy()
        mask = torch.ones(lat.shape, dtype=torch.float32)
        sch.set_begin_index(0)
        handles = []
        if a.taps and n_key == 0:
            first = {"on": True}
            keep = lambda: first["on"]  # noqa: E731
            handles.append(tr.patch_embedding.register_forward_hook(tap("patch", keep)))
            handles.append(tr.condition_embedder.register_forward_hook(
                lambda m, i, o: taps.update(temb=o[0].clone(), tproj=o[1].clone(), text=o[2].clone()) if keep() else None))
            handles.append(tr.rope.register_forward_hook(
                lambda m, i, o: taps.update(rope_cos=o[0].clone(), rope_sin=o[1].clone()) if keep() else None))
            for j, b in enumerate(tr.blocks):
                handles.append(b.register_forward_hook(tap("block%d" % j, keep)))
            handles.append(tr.norm_out.register_forward_hook(tap("norm_out", keep)))
            handles.append(tr.proj_out.register_forward_hook(tap("proj_out", keep)))
        experts = []
        for i, t in enumerate(ts):
            t1 = time.time()
            res["x%d" % i] = lat[0].numpy()
            # (the pipeline's choice of the expert and its guidance)
            low = boundary is not None and not t >= boundary
            model, gs = (tr2, g2) if low else (tr, a.guidance)
            experts.append(int(low))
            if ratio is None:
                temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(lat.shape[0], -1)
            else:
                timestep = t.expand(lat.shape[0])
            vc = model(hidden_states=lat, timestep=timestep, encoder_hidden_states=pe, return_dict=False)[0]
            if handles:
                first["on"] = False
                for h_ in handles:
                    h_.remove()
                handles = []
            vu = model(hidden_states=lat, timestep=timestep, encoder_hidden_states=ne, return_dict=False)[0]
            v = vu + gs * (vc - vu)
            res["vc%d" % i], res["vu%d" % i], res["v%d" % i] = vc[0].numpy(), vu[0].numpy(), v[0].numpy()
            lat = sch.step(v, t, lat, return_dict=False)[0]
            print("%s step %d t %d expert %d %.1f s" % (key, i, int(t), int(low), time.time() - t1), flush=True)
        res["experts"] = np.array(experts, np.int32)
        res["latent"] = lat[0].numpy()
        if vae is not None:
            mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(lat.dtype)
            istd = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(lat.dtype)
            z = lat / istd + mean
            res["z"] = z[0].numpy()
            vh = []
            if a.taps and n_key == 0:
                calls = {"n": 0}
                dec = vae.decoder
                # the decoder runs once a latent frame (a chunk): keep the first two chunks
                vh.append(dec.register_forward_pre_hook(lambda m, i: calls.update(n=calls["n"] + 1)))
                keep2 = lambda: calls["n"] <= 2  # noqa: E731
                vh.append(dec.conv_in.register_forward_hook(tap("vae_conv_in", keep2)))
                vh.append(dec.mid_block.register_forward_hook(tap("vae_mid", keep2)))
                for j, b in enumerate(dec.up_blocks):
                    vh.append(b.register_forward_hook(tap("vae_up%d" % j, keep2)))
                vh.append(dec.register_forward_hook(tap("vae_dec", keep2)))
            t1 = time.time()
            res["video"] = vae.decode(z, return_dict=False)[0][0].numpy()
            for h_ in vh:
                h_.remove()
            print("%s vae %.1f s" % (key, time.time() - t1), flush=True)
        np.savez(os.path.join(a.out, key + ".npz"), **res)
        print("saved", key, flush=True)
        if taps and n_key == 0:
            out = {}
            for k, v in taps.items():
                if isinstance(v, list):
                    for j, x in enumerate(v):
                        out[k if len(v) == 1 else "%s_%d" % (k, j)] = x.numpy()
                else:
                    out[k] = v.numpy()
            np.savez(os.path.join(a.out, "taps.npz"), **out)
            print("saved taps: %s" % " ".join(sorted(out)), flush=True)
            taps.clear()
print("total %.1f s" % (time.time() - t0))
