"""Stable Diffusion 1.5 / 2.0 references for scripts/check_sd.py: diffusers'
StableDiffusionPipeline in float32 on the CPU (safety_checker None), its parts saved on
real prompts.

    sd_reference.py MODEL_DIR out.npz --prompt TEXT [--negative TEXT] [--size 512]
                    [--steps 25] [--guidance 7.5] [--seed 0] [--png out.png]

The pipeline's own scheduler (model_index.json: SD 1.5 and 2.0-base PNDM, 2.0-768 DDIM
with v-prediction), eta 0. out.npz: "prompt", "negative"; "ids", "uids" the prompt's
and the negative prompt's tokens (77); "emb", "uemb" the text encoder's output (77,
width); "scheduler" its class name, "alphas_cumprod", "timesteps"; "noise" the first
latents of the seed (4, H/8, W/8); for each step i "x<i>" the latents before it, "e<i>"
the UNet's output (2, 4, H/8, W/8: the negative's then the prompt's), "t<i>" the
timestep; "latent" the last latents; "image" the VAE's output (3, H, W) in [-1, 1].
Run in the generation venv (diffusers from git)."""
import argparse

import numpy as np
import torch
from diffusers import StableDiffusionPipeline

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--prompt", required=True)
ap.add_argument("--negative", default="")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--steps", type=int, default=25)
ap.add_argument("--guidance", type=float, default=7.5)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--png")
a = ap.parse_args()
torch.set_num_threads(a.threads)
pipe = StableDiffusionPipeline.from_pretrained(a.model, torch_dtype=torch.float32, safety_checker=None,
                                               requires_safety_checker=False)
sch = pipe.scheduler
res = {"prompt": np.array(a.prompt), "negative": np.array(a.negative), "scheduler": np.array(type(sch).__name__),
       "prediction": np.array(sch.config.prediction_type), "guidance": np.float32(a.guidance)}
print("scheduler", type(sch).__name__, sch.config.prediction_type, flush=True)
calls = []


def hook(mod, args, kwargs, out):
    sample = args[0] if args else kwargs["sample"]
    t = args[1] if len(args) > 1 else kwargs["timestep"]
    calls.append((sample[0].numpy().copy(), float(t), out[0].numpy().copy()))  # out: (noise_pred,)


pipe.unet.register_forward_hook(hook, with_kwargs=True)
with torch.no_grad():
    tok = pipe.tokenizer
    enc = lambda s: tok(s, padding="max_length", max_length=tok.model_max_length, truncation=True,  # noqa: E731
                        return_tensors="pt").input_ids[0].numpy().astype(np.int32)
    res["ids"], res["uids"] = enc(a.prompt), enc(a.negative)
    emb, uemb = pipe.encode_prompt(a.prompt, "cpu", 1, True, negative_prompt=a.negative)
    res["emb"], res["uemb"] = emb[0].numpy(), uemb[0].numpy()
    res["alphas_cumprod"] = sch.alphas_cumprod.numpy()
    g = torch.Generator("cpu").manual_seed(a.seed)
    lat = pipe.prepare_latents(1, pipe.unet.config.in_channels, a.size, a.size, torch.float32, "cpu", g)
    res["noise"] = lat[0].numpy()
    lat_out = pipe(prompt_embeds=emb, negative_prompt_embeds=uemb, height=a.size, width=a.size,
                   num_inference_steps=a.steps, guidance_scale=a.guidance, latents=lat.clone(),
                   output_type="latent").images
    res["timesteps"] = sch.timesteps.numpy()
    for i, (x, t, e) in enumerate(calls):
        res["x%d" % i] = x
        res["t%d" % i] = np.float32(t)
        res["e%d" % i] = e
    res["latent"] = lat_out[0].numpy()
    img = pipe.vae.decode(lat_out / pipe.vae.config.scaling_factor, return_dict=False)[0]
    res["image"] = img[0].numpy()
    if a.png:
        pipe.image_processor.postprocess(img, output_type="pil")[0].save(a.png)
np.savez(a.out, **res)
print("saved", a.out, "%d UNet calls" % len(calls))
