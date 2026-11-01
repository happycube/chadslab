"""Z-Image Turbo references for scripts/check_zimage.py: diffusers' ZImagePipeline in float32 on
the CPU, its parts saved on real prompts (text to image; Turbo: guidance 0, no CFG).

The loop written from diffusers (src/diffusers/pipelines/z_image/pipeline_z_image.py; Apache
License 2.0; Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team). See
THIRD_PARTY_NOTICES.md.

    zimage_reference.py MODEL_DIR out.npz --prompt TEXT [--size 1024 | --height H --width W]
        [--steps 9] [--seed 0] [--threads 23] [--png out.png] [--taps taps.npz]

out.npz: "prompt"; "ids" the prompt's tokens (the chat template: a user message, the assistant's
turn, enable_thinking as the pipeline; at most 512); "emb" the text encoder's hidden_states[-2]
of them (T, 2560: the pipeline's prompt_embeds, the padding dropped); "noise" the latents of the
seed (16, H/8, W/8; the pipeline's prepare_latents); "sigmas", "timesteps"; for each step i
"x<i>" the DiT's input latents, "t<i>" its timestep ((1000 - t) / 1000, as the pipeline gives
it) and "v<i>" its output (the model's: the pipeline negates it into the scheduler); "latent"
the last latents; "z" the VAE's input (latents / scaling_factor + shift_factor); "image" the
VAE's output (3, H, W) in [-1, 1] (not clamped); "sched_sigmas" (100, 101) the sigmas of 1 to
100 steps (NaN past each's end). The loop is the pipeline's (__call__ of pipeline_z_image.py)
written out, so the outputs of each call are kept. --taps: the DiT's intermediate rows of the
first call: the RoPE tables (freqs_cis of the image's and the caption's rows), the image rows
after the embedder (with the pad token) and each noise refiner, the caption rows after the
embedder and each context refiner, each main layer's output, adaln_input, the final layer's
output. Run in the generation venv (diffusers from git)."""
import argparse
import time

import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--prompt", required=True)
ap.add_argument("--size", type=int, default=1024)
ap.add_argument("--height", type=int)
ap.add_argument("--width", type=int)
ap.add_argument("--steps", type=int, default=9)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--threads", type=int, default=23)
ap.add_argument("--png")
ap.add_argument("--taps")
a = ap.parse_args()
torch.set_num_threads(a.threads)
from diffusers import ZImagePipeline  # noqa: E402
from diffusers.pipelines.z_image.pipeline_z_image import (calculate_shift, get_default_z_image_sigmas,  # noqa: E402
                                                          retrieve_timesteps)

H, W = a.height or a.size, a.width or a.size
t0 = time.time()
pipe = ZImagePipeline.from_pretrained(a.model, torch_dtype=torch.float32)
tr = pipe.transformer.eval()
print("loaded %.1f s" % (time.time() - t0), flush=True)
res = {"prompt": np.array(a.prompt)}
taps = {}
with torch.no_grad():
    # the ids as _encode_prompt makes them
    text = pipe.tokenizer.apply_chat_template([{"role": "user", "content": a.prompt}], tokenize=False,
                                              add_generation_prompt=True, enable_thinking=True)
    t = pipe.tokenizer([text], padding="max_length", max_length=512, truncation=True, return_tensors="pt")
    m = t.attention_mask[0].bool()
    res["ids"] = t.input_ids[0][m].numpy().astype(np.int32)
    t1 = time.time()
    pe, _ = pipe.encode_prompt(prompt=a.prompt, device="cpu", do_classifier_free_guidance=False)
    res["emb"] = pe[0].numpy()
    print("text: %d tokens, emb %s, %.1f s" % (len(res["ids"]), tuple(pe[0].shape), time.time() - t1), flush=True)
    g = torch.Generator("cpu").manual_seed(a.seed)
    lat = pipe.prepare_latents(1, tr.in_channels, H, W, torch.float32, "cpu", g)
    res["noise"] = lat[0].numpy()
    seq = (lat.shape[2] // 2) * (lat.shape[3] // 2)
    sc = pipe.scheduler.config
    mu = calculate_shift(seq, sc.get("base_image_seq_len", 256), sc.get("max_image_seq_len", 4096),
                         sc.get("base_shift", 0.5), sc.get("max_shift", 1.15))
    tab = np.full((100, 101), np.nan, np.float32)
    for n in range(1, 101):
        retrieve_timesteps(pipe.scheduler, n, "cpu", sigmas=get_default_z_image_sigmas(n), mu=mu)
        tab[n - 1, :n + 1] = pipe.scheduler.sigmas.numpy()
    res["sched_sigmas"] = tab
    ts, n = retrieve_timesteps(pipe.scheduler, a.steps, "cpu", sigmas=get_default_z_image_sigmas(a.steps), mu=mu)
    res["sigmas"], res["timesteps"] = pipe.scheduler.sigmas.numpy(), ts.numpy()
    pipe.scheduler.set_begin_index(0)
    handles = []
    if a.taps:
        def keep(name):
            def hook(mod, args, out):
                o = out[0] if isinstance(out, tuple) else out
                taps[name] = o.detach().clone().numpy()
            return hook
        handles.append(tr.all_x_embedder["2-1"].register_forward_hook(keep("x_embed")))
        handles.append(tr.cap_embedder.register_forward_hook(keep("cap_embed")))
        handles.append(tr.t_embedder.register_forward_hook(keep("adaln")))
        for j, b in enumerate(tr.noise_refiner):
            handles.append(b.register_forward_hook(keep("noise%d" % j)))
        for j, b in enumerate(tr.context_refiner):
            handles.append(b.register_forward_hook(keep("ctx%d" % j)))
        for j, b in enumerate(tr.layers):
            handles.append(b.register_forward_hook(keep("layer%d" % j)))
        handles.append(tr.all_final_layer["2-1"].register_forward_hook(keep("final")))
        # the RoPE tables of the image's rows, then the caption's (the order of _prepare_sequence's calls)
        rope = tr.rope_embedder.__class__.__call__

        def rope_tap(self, ids):
            out = rope(self, ids)
            taps["rope%d" % sum(1 for k in taps if k.startswith("rope"))] = torch.view_as_real(out).numpy().copy()
            return out
        tr.rope_embedder.__class__.__call__ = rope_tap
    for i, tt in enumerate(ts):
        t1 = time.time()
        res["x%d" % i] = lat[0].numpy()
        timestep = (1000 - tt.expand(1)) / 1000
        res["t%d" % i] = timestep.numpy()
        out = tr([lat.unsqueeze(2)[0]], timestep, pe, return_dict=False)[0][0]
        if handles:
            for h_ in handles:
                h_.remove()
            handles = []
            tr.rope_embedder.__class__.__call__ = rope
            np.savez(a.taps, **taps)
            print("saved taps: %s" % " ".join(sorted(taps)), flush=True)
        v = out.squeeze(1)
        res["v%d" % i] = v.numpy()
        lat = pipe.scheduler.step(-v[None].to(torch.float32), tt, lat, return_dict=False)[0]
        print("step %d t %.3f %.1f s" % (i, float(tt), time.time() - t1), flush=True)
    res["latent"] = lat[0].numpy()
    z = lat / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    res["z"] = z[0].numpy()
    t1 = time.time()
    img = pipe.vae.decode(z, return_dict=False)[0]
    res["image"] = img[0].numpy()
    print("vae %.1f s" % (time.time() - t1), flush=True)
    if a.png:
        pipe.image_processor.postprocess(img, output_type="pil")[0].save(a.png)
np.savez(a.out, **res)
print("saved %s, total %.1f s" % (a.out, time.time() - t0))
