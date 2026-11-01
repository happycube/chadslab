"""FLUX.2 [klein] references for scripts/check_flux2.py: diffusers' Flux2KleinPipeline in
float32 on the CPU, its parts saved on real prompts.

The loop written from diffusers (src/diffusers/pipelines/flux2/pipeline_flux2_klein.py; Apache
License 2.0; Copyright 2025 Black Forest Labs and The HuggingFace Team). See THIRD_PARTY_NOTICES.md.

    flux2_reference.py MODEL_DIR out.npz --prompt TEXT [--size 512 | --height H --width W]
        [--image PATH ...] [--steps 4] [--seed 0] [--threads 46] [--png out.png]
    flux2_reference.py MODEL_DIR out.npz --pre-only --image PATH ...

out.npz: "ids", "mask" the prompt's tokens (512, the chat template, padded); "emb" the
text encoder's hidden states of layers 9, 18 and 27 side by side (512, 7680); "noise" the
packed latents of the seed (tokens, 128); "sigmas", "timesteps"; for each step i "x<i>"
the DiT's input latents and "v<i>" its output; "latent" the last latents (tokens, 128);
"z" the VAE's input (32, H/8, W/8, after the batch-norm and the unpatchify); "image" the
VAE's output (3, H, W) in [-1, 1]. --image (once a reference image: editing) as the
pipeline's image list, loaded by diffusers' load_image ("images" their paths): "ref_pixels<i>" each image's
preprocess output (3, H, W) in [-1, 1], "ref_latent<i>" its VAE encoder's mean (32, H/8,
W/8), "ref_rows" the packed and normalized rows of all (n, 128), "ref_ids" their position
ids; the DiT's calls take them after the noisy rows ("v<i>" the noisy rows' output). The
size: --height and --width, else the first image's (as the pipeline), else --size.
--pre-only: the images' pixels alone ("images", "ref_pixels<i>"; no model loaded). Run in
the generation venv (diffusers from git)."""
import argparse

import numpy as np
import torch
from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.image_processor import Flux2ImageProcessor
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
from diffusers.utils import load_image

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--prompt", default="")
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--height", type=int)
ap.add_argument("--width", type=int)
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--threads", type=int, default=46)
ap.add_argument("--pre-only", action="store_true")
ap.add_argument("--steps", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--png")
a = ap.parse_args()
torch.set_num_threads(a.threads)


def preprocess(processor, vae_scale_factor, res):
    """The reference images as the pipeline's __call__ (step 4): their tensors, and the first
    one's height and width."""
    height = width = None
    cond = []
    for i, path in enumerate(a.image):
        img = load_image(path)
        processor.check_image_input(img)
        iw, ih = img.size
        if iw * ih > 1024 * 1024:
            img = processor._resize_to_target_area(img, 1024 * 1024)
            iw, ih = img.size
        m = vae_scale_factor * 2
        iw, ih = iw // m * m, ih // m * m
        img = processor.preprocess(img, height=ih, width=iw, resize_mode="crop")
        res["ref_pixels%d" % i] = img[0].numpy()
        cond.append(img)
        height, width = height or ih, width or iw
        print("image %s: %s -> %dx%d" % (path, load_image(path).size, iw, ih), flush=True)
    res["images"] = np.array(a.image)
    return cond, height, width


if a.pre_only:
    # the pipeline's processor (vae_scale_factor 8 of the VAE's 4 levels, times 2)
    res = {}
    preprocess(Flux2ImageProcessor(vae_scale_factor=16), 8, res)
    np.savez(a.out, **res)
    print("saved", a.out)
    raise SystemExit(0)
pipe = Flux2KleinPipeline.from_pretrained(a.model, torch_dtype=torch.float32)
assert pipe.vae_scale_factor == 8
res = {"prompt": np.array(a.prompt)}
with torch.no_grad():
    tok = pipe.tokenizer
    text = tok.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    t = tok(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    res["ids"], res["mask"] = t["input_ids"][0].numpy().astype(np.int32), t["attention_mask"][0].numpy().astype(np.int32)
    emb, text_ids = pipe.encode_prompt(prompt=a.prompt, device="cpu")
    res["emb"] = emb[0].numpy()
    print("text: %d tokens, emb %s" % (int(res["mask"].sum()), emb.shape), flush=True)
    cond, ih, iw = preprocess(pipe.image_processor, pipe.vae_scale_factor, res)
    height, width = a.height or ih or a.size, a.width or iw or a.size
    res["size"] = np.array([height, width])
    g = torch.Generator("cpu").manual_seed(a.seed)
    c = pipe.transformer.config.in_channels // 4
    lat, lat_ids = pipe.prepare_latents(1, c, height, width, torch.float32, "cpu", g)
    res["noise"] = lat[0].numpy()
    ref_rows = ref_ids = None
    if cond:
        # the encoder's latents of each image, kept on the way through the pipeline's own path
        encode, seen = pipe.vae.encode, []

        def keep_encode(x, *args, **kw):
            out = encode(x, *args, **kw)
            seen.append(out.latent_dist.mode()[0].numpy())
            return out
        pipe.vae.encode = keep_encode
        ref_rows, ref_ids = pipe.prepare_image_latents(cond, 1, g, "cpu", torch.float32)
        pipe.vae.encode = encode
        for i, z in enumerate(seen):
            res["ref_latent%d" % i] = z
        res["ref_rows"], res["ref_ids"] = ref_rows[0].numpy(), ref_ids[0].numpy()
        print("references: %d rows" % ref_rows.shape[1], flush=True)
    n = a.steps
    sigmas = np.linspace(1.0, 1 / n, n)
    mu = compute_empirical_mu(image_seq_len=lat.shape[1], num_steps=n)
    ts, _ = retrieve_timesteps(pipe.scheduler, n, "cpu", sigmas=sigmas, mu=mu)
    res["sigmas"], res["timesteps"] = pipe.scheduler.sigmas.numpy(), ts.numpy()
    pipe.scheduler.set_begin_index(0)
    for i, tt in enumerate(ts):
        res["x%d" % i] = lat[0].numpy()
        x, ids = (lat, lat_ids) if ref_rows is None else \
            (torch.cat([lat, ref_rows], dim=1), torch.cat([lat_ids, ref_ids], dim=1))
        v = pipe.transformer(hidden_states=x, timestep=tt.expand(1) / 1000, guidance=None,
                             encoder_hidden_states=emb, txt_ids=text_ids, img_ids=ids,
                             return_dict=False)[0]
        v = v[:, :lat.size(1)]
        res["v%d" % i] = v[0].numpy()
        lat = pipe.scheduler.step(v, tt, lat, return_dict=False)[0]
        print("step %d t %.2f" % (i, float(tt)), flush=True)
    res["latent"] = lat[0].numpy()
    h = 2 * (height // (pipe.vae_scale_factor * 2))
    w = 2 * (width // (pipe.vae_scale_factor * 2))
    z = pipe._unpack_latents_with_ids(lat, lat_ids, h // 2, w // 2)
    mean = pipe.vae.bn.running_mean.view(1, -1, 1, 1)
    std = torch.sqrt(pipe.vae.bn.running_var.view(1, -1, 1, 1) + pipe.vae.config.batch_norm_eps)
    z = pipe._unpatchify_latents(z * std + mean)
    res["z"] = z[0].numpy()
    img = pipe.vae.decode(z, return_dict=False)[0]
    res["image"] = img[0].numpy()
    if a.png:
        pipe.image_processor.postprocess(img, output_type="pil")[0].save(a.png)
np.savez(a.out, **res)
print("saved", a.out)
