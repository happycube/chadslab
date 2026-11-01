"""Qwen-Image-2.1 references for scripts/check_qwen_image.py: diffusers' QwenImage21Pipeline in
float32 on the CPU, its parts saved on real prompts (text to image) and real photos (editing).

The loop written from diffusers (src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py;
Apache License 2.0; Copyright 2026 Qwen-Image Team, The HuggingFace Team). See
THIRD_PARTY_NOTICES.md.

    qwen_image_reference.py MODEL_DIR out.npz --prompt TEXT [--size 512] [--steps 20] [--seed 0]
        [--negative TEXT --cfg 4.0] [--image PATH ...] [--text-only] [--threads 12] [--png out.png]

out.npz: "ids" the prompt's tokens (the pipeline's template: the system message, the user's
prompt, the assistant's turn), "drop" the system message's tokens the pipeline drops; "emb" the
text encoder's last hidden state (before its final norm) of the tokens past drop (T, 4096);
"noise" the packed latents of the seed (tokens, 64); "sigmas", "timesteps"; for each step i
"x<i>" the DiT's input latents and "v<i>" its output (with CFG: "vc<i>" and "vu<i>" of the
prompt and of the negative prompt, and "nemb", "nids" of the negative prompt); "latent" the
last latents (tokens, 64); "z" the VAE's input (64, H/16, W/16, after the latents' mean and
std); "image" the VAE's output (4, H, W: RGBA) in [-1, 1]. The DiT runs as the pipeline's
loop: the text's keys and values cached at the first step (use_kv_cache). Run in the
generation venv (diffusers from git).

With --image (editing; --size is the pipeline's output_resolution, the output's size that of
the last image's aspect ratio, as the pipeline): "hw" the output's (height, width); for each
image k "src<k>" the file's pixels (H, W, 4: RGBA, as the pipeline converts it), "size<k>" its
resized (width, height), "vae_in<k>" the VAE's input (4, h, w) in [-1, 1], "ilat<k>" its
latents (h/16 w/16, 64) after the latents' mean and std; the processor's "pixels" (patches,
1536), "grid" (images, 3), "mm" (the token types: 1 an image's); the vision tower's "vis" (the
merger's rows) and "ds0".."ds2" (the deepstack mergers'); "pos" (3, tokens) the text model's
MRoPE positions; "imgpad" (T,) the image slots of the rows of emb. --text-only stops after the
text encoder (no DiT, no VAE)."""
import argparse
import os

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from diffusers import QwenImage21Pipeline
from diffusers.models.transformers.transformer_qwenimage21 import QwenImage21KVCache
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import (calculate_dimensions, calculate_shift,
                                                                  retrieve_timesteps)

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--prompt", required=True)
ap.add_argument("--negative")
ap.add_argument("--cfg", type=float, default=1.0)
ap.add_argument("--size", type=int, default=512)
ap.add_argument("--steps", type=int, default=20)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--image", action="append", default=[])
ap.add_argument("--text-only", action="store_true")
ap.add_argument("--threads", type=int, default=12)
ap.add_argument("--png")
a = ap.parse_args()
torch.set_num_threads(a.threads)


class Processor(Qwen3VLProcessor):
    """Qwen3VLProcessor with no video processor (which needs torchvision, which the venv has
    not): the image processor is then the PIL backend (Qwen2VLImageProcessorPil); a text alone
    gives the tokenizer's ids. apply_chat_template of one conversation as a batch of one."""

    def __init__(self, image_processor=None, tokenizer=None, chat_template=None, **kw):
        super().__init__(image_processor, tokenizer, None, chat_template=chat_template)

    def apply_chat_template(self, conversation, **kw):
        ids = super().apply_chat_template(conversation, **kw)
        return [ids] if ids and isinstance(ids[0], int) else ids


pdir = os.path.join(a.model, "processor")
proc = Processor(AutoImageProcessor.from_pretrained(pdir), AutoTokenizer.from_pretrained(pdir),
                 open(os.path.join(pdir, "chat_template.jinja")).read())
pipe = QwenImage21Pipeline.from_pretrained(a.model, processor=proc, torch_dtype=torch.float32)
res = {"prompt": np.array(a.prompt)}
cfg = a.negative is not None and a.cfg > 1
edit = bool(a.image)
with torch.no_grad():
    # the condition images as the pipeline's __call__: RGBA, one resize (Lanczos) to the
    # output_resolution's area at the image's aspect ratio for the text encoder and the VAE
    sizes, inputs, vae_images = [], None, None
    height = width = a.size
    if edit:
        inputs, vae_images = [], []
        imgs = [Image.open(p) for p in a.image]
        for k, img in enumerate(imgs):
            img.load()
            if img.mode != "RGBA":
                img = img.convert("RGBA")
            res["src%d" % k] = np.asarray(img)
            iw, ih, _ = calculate_dimensions(a.size * a.size, img.size[0] / img.size[1])
            sizes.append((iw, ih))
            res["size%d" % k] = np.array([iw, ih])
            inputs.append(pipe.image_processor.resize(img, width=iw, height=ih))
            vae_images.append(pipe.image_processor.preprocess(img, width=iw, height=ih).unsqueeze(2))
            res["vae_in%d" % k] = vae_images[-1][0, :, 0].numpy()
        width, height, _ = calculate_dimensions(a.size * a.size, imgs[-1].size[0] / imgs[-1].size[1])
        width, height = width // 32 * 32, height // 32 * 32
        res["hw"] = np.array([height, width])
        print("images %s, output %d x %d" % (sizes, width, height), flush=True)

    def encode(prompt, key):
        if edit:
            # the processor's inputs as _get_qwen_prompt_embeds builds them
            rep = "".join(("" if k == 0 else " ") + "<image%d><|vision_start|><|image_pad|><|vision_end|>" % (k + 1)
                          for k in range(len(inputs)))
            text = pipe.prompt_template_ti2i.replace("<image1><|vision_start|><|image_pad|><|vision_end|>", rep)
            text = text.format(prompt or " ")
            cond = []
            for img in inputs:
                white = Image.new("RGB", img.size, (255, 255, 255))
                white.paste(img, mask=img.getchannel("A"))
                cond.append(white)
            t = pipe.processor(text=[text], images=cond, padding=True, padding_side="left", return_tensors="pt")
            res[key + "pixels"] = t["pixel_values"].numpy()
            res[key + "grid"] = t["image_grid_thw"].numpy()
            res[key + "mm"] = t["mm_token_type_ids"][0].numpy().astype(np.int32)
            vm = pipe.text_encoder.model.visual(t["pixel_values"], grid_thw=t["image_grid_thw"])
            res[key + "vis"] = vm.pooler_output.numpy()
            for j, d in enumerate(vm.deepstack_features):
                res[key + "ds%d" % j] = d.numpy()
            pos, _ = pipe.text_encoder.model.get_rope_index(t["input_ids"], t["mm_token_type_ids"],
                                                            t["image_grid_thw"], attention_mask=t["attention_mask"])
            res[key + "pos"] = pos[:, 0].numpy().astype(np.int32)
        else:
            text = pipe.prompt_template_t2i.format(prompt)
            t = pipe.processor(text=[text], padding=True, padding_side="left", return_tensors="pt")
        res[key + "ids"] = t["input_ids"][0].numpy().astype(np.int32)
        emb, mask, pad = pipe.encode_prompt(prompt=prompt, image=inputs, device="cpu")
        assert mask is None
        assert edit or not bool(pad.any())
        res[key + "emb"] = emb[0].numpy()
        res[key + "imgpad"] = pad[0].numpy()
        print("%s: %d tokens, emb %s, image slots %d" % (key or "prompt", len(res[key + "ids"]), tuple(emb.shape),
                                                        int(pad.sum())), flush=True)
        return emb, pad
    res["drop"] = np.array(pipe._drop_idx)
    emb, pad = encode(a.prompt, "")
    if cfg:
        res["negative"], res["cfg"] = np.array(a.negative), np.array(a.cfg)
        nemb, npad = encode(a.negative, "n")
    if a.text_only:
        np.savez(a.out, **res)
        print("saved", a.out)
        raise SystemExit
    g = torch.Generator("cpu").manual_seed(a.seed)
    c = pipe.transformer.config.in_channels
    lat, ilat = pipe.prepare_latents(vae_images, 1, c, height, width, torch.float32, "cpu", g)
    res["noise"] = lat[0].numpy()
    if edit:
        r0 = 0
        for k, (iw, ih) in enumerate(sizes):
            r1 = r0 + (ih // 16) * (iw // 16)
            res["ilat%d" % k] = ilat[0, r0:r1].numpy()
            r0 = r1
    n = a.steps
    sigmas = np.linspace(1.0, 1 / n, n)
    sc = pipe.scheduler.config
    mu = calculate_shift(lat.shape[1], sc.get("base_image_seq_len", 256), sc.get("max_image_seq_len", 4096),
                         sc.get("base_shift", 0.5), sc.get("max_shift", 1.15))
    ts, _ = retrieve_timesteps(pipe.scheduler, n, "cpu", sigmas=sigmas, mu=mu)
    res["sigmas"], res["timesteps"] = pipe.scheduler.sigmas.numpy(), ts.numpy()
    shapes = [[*[(1, ih // 16, iw // 16) for iw, ih in sizes], (1, height // 16, width // 16)]]
    slots = lambda m: torch.cat([m, m.new_ones(m.shape[0], lat.shape[1] // 4)], dim=1)  # noqa: E731
    nb = len(pipe.transformer.transformer_blocks)
    cache, ncache = QwenImage21KVCache(nb), QwenImage21KVCache(nb)
    pipe.scheduler.set_begin_index(0)
    for i, tt in enumerate(ts):
        res["x%d" % i] = lat[0].numpy()
        mode = "extract" if i == 0 else "cached"
        ts_ = tt.expand(1).to(torch.float32) / 1000
        x = lat if ilat is None else torch.cat([ilat, lat], dim=1)
        v = pipe.transformer(hidden_states=x, timestep=ts_, encoder_hidden_states=emb, encoder_hidden_states_mask=None,
                             img_shapes=shapes, img_mask=slots(pad), kv_cache=cache, kv_cache_mode=mode,
                             return_dict=False)[0][:, -lat.shape[1]:]
        if cfg:
            vu = pipe.transformer(hidden_states=x, timestep=ts_, encoder_hidden_states=nemb,
                                  encoder_hidden_states_mask=None, img_shapes=shapes, img_mask=slots(npad),
                                  kv_cache=ncache, kv_cache_mode=mode, return_dict=False)[0][:, -lat.shape[1]:]
            res["vc%d" % i], res["vu%d" % i] = v[0].numpy(), vu[0].numpy()
            v = vu + a.cfg * (v - vu)
        res["v%d" % i] = v[0].numpy()
        lat = pipe.scheduler.step(v, tt, lat, return_dict=False)[0]
        print("step %d t %.2f" % (i, float(tt)), flush=True)
    res["latent"] = lat[0].numpy()
    z = pipe._unpack_latents(lat, height, width, pipe.vae_scale_factor)
    vc = pipe.vae.config
    mean = torch.tensor(vc.latents_mean).view(1, vc.z_dim, 1, 1, 1)
    std = torch.tensor(vc.latents_std).view(1, vc.z_dim, 1, 1, 1)
    z = z * std + mean
    res["z"] = z[0, :, 0].numpy()
    img = pipe.vae.decode(z, return_dict=False)[0][:, :, 0]
    res["image"] = img[0].numpy()
    if a.png:
        pipe.image_processor.postprocess(img, output_type="pil")[0].save(a.png)
np.savez(a.out, **res)
print("saved", a.out)
