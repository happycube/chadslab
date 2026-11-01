"""Qwen-Image-2.1 (Qwen/Qwen-Image-2.1, diffusers' layout): text to image as
diffusers' QwenImage21Pipeline (true_cfg_scale 1 by default: no CFG).

Ported from diffusers
(src/diffusers/pipelines/qwenimage21/pipeline_qwenimage21.py, Copyright 2026
Qwen-Image Team, The HuggingFace Team;
schedulers/scheduling_flow_match_euler_discrete.py, Copyright 2025 Stability
AI, Katherine Crowson and The HuggingFace Team; Apache License 2.0). See
THIRD_PARTY_NOTICES.md.

    the prompt in the pipeline's template (a system message, the user's
        prompt, the assistant's turn: a raw string, not the chat template)
    the text encoder (Qwen3-VL-8B's language model, qwen3.dense: a text
        alone needs no vision tower; its MRoPE of three equal positions is
        RoPE): the last layer's hidden state before the final norm, the
        system message's tokens dropped (T, 4096)
    the DiT (dit.py): the text's keys and values once (causal_condition),
        then the Euler steps of the flow over the image's tokens (the
        latents unpatched: 64 values a token, H/16 x W/16 tokens;
        FlowMatchEulerDiscreteScheduler: linspace(1, 1 / steps), the
        exponential time shift of mu from the tokens, stretched to end at
        shift_terminal); with CFG (a negative prompt and cfg > 1) the
        negative prompt's velocity too, neg + cfg (cond - neg)
    the VAE's decoder (vae.py) of the latents (times their std plus their
        mean): RGBA

The text encoder runs on the CPU (its matrices in BF12 by default, text_bf12:
KQ_BF12X16, 10.6 GB, the bfloat16 values but those more than 15 binades under
their group's largest; text_bf12=False or RECLANK_QWEN_TEXT_BF12=0: the
bfloat16 rows, KQ_BF16X16, 13.9 GB; with the embeddings 15 GB, which does not
fit on a 24 GB GPU beside the DiT's 14 GB); text_gpu puts it on the GPU (BF12
rows through KQ_LINEAR, the bfloat16 rows' kernel; alone there: it is freed
after the prompt). On the GPU the vision tower's matrices are BF12 with it
(its MLP's down matrices of 4304 columns stay bfloat16; on the CPU bfloat16,
faster there). On the GPU the DiT's memory (14 GB of weights; 14.6 GB in all
at 1024 x 1024, 16.4 GB at 2048) is freed before the VAE runs (4.2 GB at
1024 x 1024, 13 GB at 2048) and the VAE's after the image, unless keep: the
next image takes the DiT's weights back (a few seconds).

Editing (images: up to the pipeline's 10 reference images; the output's
size by default that of the last image's aspect ratio at the area of
resolution^2, as the pipeline's output_resolution):

    each image as the pipeline's: RGBA, resized (Lanczos) to the area
        resolution^2 at its aspect ratio (sides of 32: calculate_dimensions);
        for the vision tower composited over white, the processor's rescale
        and normalization (Qwen2VLImageProcessor: the size is already its
        smart_resize's), patches of 16 x 16 in the order of the merge blocks;
        for the VAE 2x - 1 of all four channels
    the vision tower (qwen.vision.QwenVision of the text encoder's
        checkpoint: Qwen3-VL's ViT with its deepstack mergers) of each image:
        its rows and the deepstack rows of its H/32 x W/32 tokens
    the prompt in the edit template (<imageK><|vision_start|>, an
        <|image_pad|> a token of the image, <|vision_end|>, the images joined
        by a space, before the prompt), the text encoder with the vision rows
        in place of the image tokens' embeddings, the deepstack rows added
        after its first three layers, the MRoPE of the image tokens'
        positions (qwen3.dense's media)
    the VAE's encoder (vae.py) of each image: the latents' mean, less their
        mean, over their std
    the DiT's prefix: the text rows (past the system message) with each
        image's slots in place of its latents' tokens (a slot is 2 x 2
        tokens), the images as blocks (dit.py); then the steps as text to
        image, the target's frame after the prefix

On the GPU the vision tower (1.2 GB of weights) and the VAE's encoder are
runners of the request's scope (runners()), done after their turn: RESIDENCY
may then evict them for the DiT (a script that edits on the GPU turns it on:
RECLANK_RESIDENCY=1, or check_qwen_image.py --gpu). An edit's DiT cache (the
prefix's keys and values) is F20 by default, 0.63 MB a row of the prefix (a
1024 x 1024 reference 4096 rows: 2.7 GB; float32's 4.3 GB), and on the GPU it
is in host memory, a block at a time fetched to the GPU (dit.py: about 0.19
MB of the GPU a reference row, for the prefix's and the steps' buffers;
RECLANK_QWEN_CACHE_HOST=0: on the GPU); cache_f20=False or
RECLANK_QWEN_CACHE_F20=0: float32 on the GPU.

Not here: the README's masks and painted annotations are images in the
pipeline's terms (another reference image, the prompt saying what it
marks): nothing more to port; an RGBA reference's alpha reaches the VAE
(the vision tower sees it over white).
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners, to_bf12_dev
from reclank.formats.gguf import BF16
from reclank.models.qwen.tokenizer import QwenTokenizer
from reclank.models.qwen.vision import MERGE, PATCH, QwenVision, patches_of, smart_resize, vision_weights
from reclank.models.qwen3.dense import Qwen3Dense, _KQ
from reclank.models.qwen_image.dit import PAD, QwenImageDiT
from reclank.models.qwen_image.vae import QwenImageVAE
from reclank.runtime.threads import team_threads

SYSTEM = "Comprehend and analyze the provided prompt."
TEMPLATE = ("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
            "<|im_start|>assistant\n")
MAX_IMAGES = 10     # the model card's reference images


def calculate_dimensions(area, ratio):
    """The pipeline's (width, height) of an image of area pixels at the
    aspect ratio (width / height): sides rounded to multiples of 32."""
    width = math.sqrt(area * ratio)
    height = width / ratio
    return round(width / 32) * 32, round(height / 32) * 32


def _bf12(lins, gpu, label, threads=None):
    """The bfloat16 matrices lins in BF12 (encoders.runtime.to_bf12_dev: BF12 rows on the GPU,
    KQ_BF12X16 on the CPU; a matrix its rule refuses stays bfloat16). The conversion is an OpenMP
    team's: beside a GPU (OMP_NUM_THREADS=1) a team of threads (team_threads()) for it alone, the
    main thread's ICV 1 again after (the text encoder's 13.9 GB in 83 s on one thread)."""
    import ctypes
    gomp = ctypes.CDLL("libgomp.so.1")
    one = gomp.omp_get_max_threads() == 1
    if one:
        gomp.omp_set_num_threads(threads or team_threads())
    try:
        kept = sum(not to_bf12_dev(lin, gpu) for lin in lins)
    finally:
        if one:
            gomp.omp_set_num_threads(1)
    if kept:
        print("qwen-image %s: %d of %d matrices stay bfloat16 (not BF12)" % (label, kept, len(lins)), flush=True)


def condition_image(img, resolution=1024):
    """A reference image (PIL) as the pipeline's: RGBA, resized (Lanczos,
    VaeImageProcessor.resize) to the area resolution^2 at its aspect ratio."""
    from PIL import Image
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    w, h = calculate_dimensions(resolution * resolution, img.size[0] / img.size[1])
    return img.resize((w, h), resample=Image.LANCZOS)


def vae_pixels(img):
    """The VAE's input (4, H, W) in [-1, 1] of a resized reference (RGBA):
    x / 255 (float32), 2x - 1."""
    x = np.asarray(img).astype(np.float32) / np.float32(255.0)
    return np.ascontiguousarray((np.float32(2.0) * x - np.float32(1.0)).transpose(2, 0, 1))


class QwenImage21:
    """A Qwen-Image-2.1 directory. parts: the parts to load ("text", "dit",
    "vae", or "all"). cache_f20: an edit's DiT cache in F20 (None:
    RECLANK_QWEN_CACHE_F20, on unless 0; False: float32). text_bf12: the text
    encoder's and the vision tower's matrices in BF12 (None:
    RECLANK_QWEN_TEXT_BF12, on unless 0; False: their bfloat16 as stored)."""

    def __init__(self, path, gpu=False, cpu_threads=None, parts=("all",), text_gpu=False, cache_f20=None,
                 text_bf12=None):
        self.path, self.gpu = path, bool(gpu)
        parts = set(parts)
        every = "all" in parts
        pdir = os.path.join(path, "processor")
        self.tok = QwenTokenizer(os.path.join(pdir, "tokenizer.json"))
        self.pad = self.tok.special["<|endoftext|>"]
        # the system message's tokens, which the pipeline drops (_drop_idx: the chat template
        # of the system message alone)
        self.drop = len(self.tok.encode("<|im_start|>system\n" + SYSTEM + "<|im_end|>\n"))
        tdir = os.path.join(path, "text_encoder")
        tcfg = json.load(open(os.path.join(tdir, "config.json")))["text_config"]
        self.text = Qwen3Dense(tdir, (tcfg["num_hidden_layers"],), gpu=gpu and text_gpu, cpu_threads=cpu_threads,
                               prefix="model.language_model.", cfg=tcfg, cpu_bf16=True) \
            if every or "text" in parts else None
        self.text_gpu = gpu and text_gpu
        self.cpu_threads = cpu_threads
        self.text_bf12 = os.environ.get("RECLANK_QWEN_TEXT_BF12", "1") != "0" if text_bf12 is None else bool(text_bf12)
        if self.text is not None and self.text_bf12:
            _bf12([L[k] for L in self.text.layers for k in ("q", "k", "v", "o", "gate", "up", "down")], self.text_gpu,
                  "text encoder", cpu_threads)
        self.dit = QwenImageDiT(os.path.join(path, "transformer"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "dit" in parts else None
        # an edit's DiT cache in F20 (None: RECLANK_QWEN_CACHE_F20, on unless 0)
        self.cache_f20 = os.environ.get("RECLANK_QWEN_CACHE_F20", "1") != "0" if cache_f20 is None else bool(cache_f20)
        if self.dit is not None:
            self.dit.f20_media = self.cache_f20
        self.vae = QwenImageVAE(os.path.join(path, "vae"), gpu=gpu, cpu_threads=cpu_threads) \
            if every or "vae" in parts else None
        self.sched = json.load(open(os.path.join(path, "scheduler", "scheduler_config.json")))
        # editing: the processor's settings, the vision tower (loaded at the first edit)
        self.pp = json.load(open(os.path.join(pdir, "preprocessor_config.json")))
        assert (self.pp["patch_size"], self.pp["merge_size"], self.pp["temporal_patch_size"]) == (PATCH, MERGE, 2)
        self.tdir = tdir
        self._vision = None
        self.image_pad = self.tok.special["<|image_pad|>"]

    @property
    def vision(self):
        """The vision tower of the text encoder's checkpoint (its bfloat16
        matrices as stored, or on the GPU BF12 with text_bf12), on the GPU when gpu."""
        if self._vision is None:
            v = QwenVision(vision_weights(self.tdir))
            v.gpu = self.gpu
            lins = [v.mm0, v.mm2] + [b[k] for b in v.blk for k in ("q", "k", "v", "o", "up", "down")] + \
                [d[k] for d in v.ds for k in ("fc1", "fc2")]
            if self.text_bf12 and self.gpu:
                # (on the CPU KQ_BF12X16's decode made the tower slower: 58 s against 34 s of a 1024
                # reference on 12 threads, for 0.2 GB)
                _bf12(lins, self.gpu, "vision tower", self.cpu_threads)
            if self.gpu:
                # its bfloat16 matrices as the text encoder's and the DiT's on the GPU: KQ_BF16 (x as
                # two bfloat16 planes, float32 sums; or BF12 rows, the same kernel's), and float32
                # attention (ENC_ATTN_TILE): QwenVision's float16 x and q, k, v put its rows 1.5e-3
                # (1 - cos, worst, at 1024) off diffusers', these 4.4e-6 (8e-7 at 512)
                v.attn_gpu = "f32"
                for lin in lins:
                    if lin.w is not None and lin.w.dtype == np.uint16:
                        lin.kq = _KQ(lin.w.view(np.uint8), BF16, *lin.w.shape)
            self._vision = v
        return self._vision

    def tokenize(self, prompt):
        """The ids of a prompt in the pipeline's template (an empty prompt: a
        space, as the pipeline)."""
        return np.array(self.tok.encode(TEMPLATE.format(prompt or " ")), np.int32)

    def runners(self):
        """The GPU runners of the parts (name: _Runner; the text encoder's with text_gpu; the vision
        tower's once loaded, and the VAE encoder's: an edit's)."""
        r = gpu_runners(self.gpu, text=self.text if self.text_gpu else None, dit=self.dit, vae=self.vae)
        if self.gpu and self.vae is not None:
            r["vae_enc"] = self.vae.enc_runner
        if self.gpu and self._vision is not None:
            r["vision"] = self._vision.runner
        return r

    def close(self):
        """Free the parts' GPU memory and the DiT's host caches (unregistered from CUDA first: a
        pipeline dropped without close() does it in the DiT's finalizer)."""
        for k, r in self.runners().items():
            if k != "dit":
                r.release(keep_host=False)
        if self.dit is not None:
            self.dit.close()

    def text_embed(self, ids):
        """The DiT's text rows (T, 4096) of a prompt's ids: the text
        encoder's last hidden state past the system message (the encoder's
        rows padded to a multiple of 64 at the end, which the causal rows do
        not see)."""
        n = len(ids)
        rows = -(-n // PAD) * PAD
        x = np.full(rows, self.pad, np.int32)
        x[:n] = ids
        h = self.text.hidden(x[None], [n])[0][0]
        if self.text_gpu:
            self.text.runner.release(keep_host=True)
        return np.ascontiguousarray(h[self.drop:n])

    def vision_patches(self, img):
        """The vision tower's patches (n, 768) and grid (gh, gw) of a resized
        reference (RGBA): composited over white, the processor's rescale
        (float64 x 1/255, to float32) and normalization ((x - mean) / std,
        float32), in the order of the merge blocks. Its size (multiples of 32)
        is its own smart_resize's: the processor's resize keeps it."""
        from PIL import Image
        pp = self.pp
        lo, hi = pp["size"]["shortest_edge"], pp["size"]["longest_edge"]
        if smart_resize(img.size[1], img.size[0], min_pixels=lo, max_pixels=hi) != (img.size[1], img.size[0]):
            raise ValueError("a reference of %d x %d is outside the processor's sizes (%d to %d pixels)" % (
                img.size[0], img.size[1], lo, hi))
        white = Image.new("RGB", img.size, (255, 255, 255))
        white.paste(img, mask=img.getchannel("A"))
        x = (np.asarray(white, np.float64) * pp["rescale_factor"]).astype(np.float32)
        x = ((x - np.array(pp["image_mean"], np.float32)) / np.array(pp["image_std"], np.float32)).astype(np.float32)
        return patches_of(x)

    def vision_rows(self, refs):
        """The vision tower's rows of resized references: a list of (rows
        (tokens, 4096), the deepstack rows (a list), (gh, gw) patches)."""
        out = []
        for img in refs:
            p, grid = self.vision_patches(img)
            y, ds = self.vision.encode_deepstack(p, grid)
            out.append((y, ds, grid))
        return out

    def tokenize_edit(self, prompt, grids):
        """The ids of a prompt with images of grids (gh, gw) of patches in the
        pipeline's edit template (the processor's <|image_pad|> a token)."""
        imgs = "".join(("" if k == 0 else " ") + "<image%d><|vision_start|>" % (k + 1) +
                       "<|image_pad|>" * (gh * gw // (MERGE * MERGE)) + "<|vision_end|>"
                       for k, (gh, gw) in enumerate(grids))
        return np.array(self.tok.encode(TEMPLATE.format(imgs + (prompt or " "))), np.int32)

    @staticmethod
    def rope_index(ids, image_pad, grids):
        """Qwen3-VL's MRoPE positions (3, tokens) (get_rope_index): a text
        token at (p, p, p), p + 1; an image of gh x gw patches at (p, p + row,
        p + column) of its tokens, p + max of its sides in tokens."""
        n = len(ids)
        pos = np.zeros((3, n), np.int64)
        p, i, k = 0, 0, 0
        while i < n:
            if ids[i] != image_pad:
                pos[:, i] = p
                p += 1
                i += 1
                continue
            th, tw = grids[k][0] // MERGE, grids[k][1] // MERGE
            pos[0, i:i + th * tw] = p
            pos[1, i:i + th * tw] = p + np.repeat(np.arange(th), tw)
            pos[2, i:i + th * tw] = p + np.tile(np.arange(tw), th)
            p += max(th, tw)
            i += th * tw
            k += 1
        return pos

    def edit_embed(self, prompt, vis):
        """The text encoder's rows past the system message (T, 4096) of a
        prompt with the references' vision rows vis (vision_rows), and which
        of them are image slots (T,)."""
        grids = [g for _, _, g in vis]
        ids = self.tokenize_edit(prompt, grids)
        n = len(ids)
        rows = -(-n // PAD) * PAD
        x = np.full(rows, self.pad, np.int32)
        x[:n] = ids
        pos = np.zeros((3, rows), np.int64)
        pos[:, :n] = self.rope_index(ids, self.image_pad, grids)
        pos[:, n:] = pos[:, n - 1:n] + 1 + np.arange(rows - n)
        slot = np.flatnonzero(ids == self.image_pad)
        starts = [int(i) for i in slot if i == slot[0] or ids[i - 1] != self.image_pad]
        blocks = [(s0, s0 + y.shape[0]) for s0, (y, _, _) in zip(starts, vis)]
        media = dict(blocks=blocks, rows=np.concatenate([y for y, _, _ in vis]), pos=pos,
                     deep=[np.concatenate([ds[j] for _, ds, _ in vis]) for j in range(len(vis[0][1]))])
        h = self.text.hidden(x[None], [n], media=media)[0][0]
        return np.ascontiguousarray(h[self.drop:n]), ids[self.drop:n] == self.image_pad

    def reference_latents(self, refs):
        """The VAE's packed latents (h w, 64) of resized references, and
        their (h, w)."""
        lats, shapes = [], []
        for img in refs:
            z = self.vae.encode(vae_pixels(img))
            lats.append(self.vae.pack(z))
            shapes.append(z.shape[1:])
        return lats, shapes

    @staticmethod
    def joint(emb, slots, lats, shapes):
        """The DiT's condition of an edit (the transformer's joint sequence
        before the target): emb's text rows with each image's run of slots in
        place of its latents' tokens (h w rows), their positions
        (QwenImage21Rope) and keys (a text row the rows up to it, an image's
        rows up to its end); the target's frame after them."""
        T = emb.shape[0]
        P = T - int(slots.sum()) + sum(h * w for h, w in shapes)
        txt = np.zeros((P, emb.shape[1]), np.float32)
        pos = np.zeros((P, 3), np.int64)
        ahi = np.zeros(P, np.int32)
        blocks = []
        r = f = i = k = 0
        while i < T:
            if not slots[i]:
                txt[r], pos[r], ahi[r] = emb[i], f, r + 1
                r, f, i = r + 1, f + 1, i + 1
                continue
            h, w = shapes[k]
            assert slots[i:i + h * w // 4].all()
            blocks.append((r, r + h * w))
            pos[r:r + h * w, 0] = f
            pos[r:r + h * w, 1] = np.repeat(np.arange(h) - (h - h // 2), w)
            pos[r:r + h * w, 2] = np.tile(np.arange(w) - (w - w // 2), h)
            ahi[r:r + h * w] = r + h * w
            r, f, i, k = r + h * w, f + max(h, w), i + h * w // 4, k + 1
        assert r == P and k == len(shapes)
        return dict(txt=txt, media=dict(blocks=blocks, lat=np.concatenate(lats), pos=pos, ahi=ahi), frame=f)

    def sigmas(self, steps, tokens):
        """The sigmas of the steps (and the 0 after the last) of an image of
        tokens, and the timesteps: linspace(1, 1 / steps) shifted
        (exponential) by mu = calculate_shift(tokens), stretched to end at
        shift_terminal, float32 as the scheduler's numpy."""
        s = self.sched
        assert s.get("use_dynamic_shifting") and s.get("time_shift_type") == "exponential"
        b0, b1 = s.get("base_image_seq_len", 256), s.get("max_image_seq_len", 4096)
        m0, m1 = s.get("base_shift", 0.5), s.get("max_shift", 1.15)
        m = (m1 - m0) / (b1 - b0)
        mu = tokens * m + (m0 - m * b0)
        sig = np.linspace(1.0, 1.0 / steps, steps).astype(np.float32)
        sig = math.exp(mu) / (math.exp(mu) + (1 / sig - 1) ** 1.0)
        if s.get("shift_terminal"):
            om = 1 - sig
            sig = 1 - om / (om[-1] / (1 - s["shift_terminal"]))
        sig = sig.astype(np.float32)
        return np.append(sig, np.float32(0)), (sig * np.float32(1000)).astype(np.float32)

    def denoise(self, noise, txt, hh, ww, steps, neg=None, cfg=1.0, keep=None):
        """The latents (hh ww, 64) of the Euler steps of the flow from noise
        (hh ww, 64) with the text rows txt (T, 4096) or an edit's condition
        (joint); neg: the negative prompt's (CFG of scale cfg). keep: a list
        that gets each step's (input, velocity). The DiT stays pinned from the
        prefix (its keys and values on the GPU) to the last step."""
        sig, ts = self.sigmas(steps, hh * ww)
        c = txt if isinstance(txt, dict) else dict(txt=txt, media=None, frame=txt.shape[0])
        cn = neg if neg is None or isinstance(neg, dict) else dict(txt=neg, media=None, frame=neg.shape[0])
        with RESIDENCY.use(*gpu_runners(self.gpu, dit=self.dit).values()):
            self.dit.prefix(c["txt"], 0, c["media"])
            if cn is not None:
                self.dit.prefix(cn["txt"], 1, cn["media"])
            x = noise.astype(np.float32)
            for i in range(steps):
                # the pipeline's timestep: t / 1000 (float32) into the DiT, which takes * 1000
                t = float(np.float32(np.float32(ts[i]) / np.float32(1000)) * np.float32(1000))
                v = self.dit.velocity(x, t, c["txt"].shape[0], hh, ww, 0, c["frame"])
                if cn is not None:
                    vu = self.dit.velocity(x, t, cn["txt"].shape[0], hh, ww, 1, cn["frame"])
                    v = (vu + np.float32(cfg) * (v - vu)).astype(np.float32)
                if keep is not None:
                    keep.append((x.copy(), v))
                x = (x + (sig[i + 1] - sig[i]) * v).astype(np.float32)
        return x

    def image(self, latents, hh, ww):
        """The image (4, 16 hh, 16 ww) (RGBA) in [-1, 1] of the latents."""
        return self.vae.decode(self.vae.unpack(latents, hh, ww))

    def generate(self, prompt, size=None, steps=40, seed=0, noise=None, negative=None, cfg=1.0, keep=False,
                 images=None, resolution=1024):
        """An image (H, W, 4) uint8 RGBA of a prompt: size (or (height,
        width)) a multiple of 32 (rounded down to one, as the pipeline;
        default resolution, or with images the last one's aspect ratio at the
        area resolution^2); images: reference images (PIL) to edit, each
        resized to the area resolution^2; noise (tokens, 64) in place of the
        seed's (NumPy's normal: not torch's, so not diffusers' image of the
        same seed); CFG with a negative prompt and cfg > 1. On the GPU the
        DiT's memory is freed before the VAE and the VAE's after it, unless
        keep (RESIDENCY decides, each part unpinned after its turn)."""
        images = list(images or [])
        if len(images) > MAX_IMAGES:
            raise ValueError("at most %d reference images" % MAX_IMAGES)
        if size is None:
            size = resolution
            if images:
                w, h = calculate_dimensions(resolution * resolution, images[-1].size[0] / images[-1].size[1])
                size = (h, w)
        hgt, wid = (size, size) if isinstance(size, int) else size
        hh, ww = hgt // 32 * 2, wid // 32 * 2
        use_neg = negative is not None and cfg > 1
        if images:
            self.vision     # (loaded now: its runner in the request's scope)
        r = self.runners()
        with RESIDENCY.use(*r.values()) as use:
            if images:
                refs = [condition_image(im, resolution) for im in images]
                vis = self.vision_rows(refs)
                use.done(r.get("vision"))
                emb, slots = self.edit_embed(prompt, vis)
                nemb, nslots = self.edit_embed(negative, vis) if use_neg else (None, None)
                use.done(self.text.runner)
                lats, shapes = self.reference_latents(refs)
                use.done(r.get("vae_enc"))
                txt = self.joint(emb, slots, lats, shapes)
                neg = self.joint(nemb, nslots, lats, shapes) if use_neg else None
            else:
                txt = self.text_embed(self.tokenize(prompt))
                neg = self.text_embed(self.tokenize(negative)) if use_neg else None
                use.done(self.text.runner)
            if noise is None:
                noise = np.random.default_rng(seed).standard_normal((hh * ww, self.vae.latent)).astype(np.float32)
            lat = self.denoise(noise, txt, hh, ww, steps, neg, cfg)
            use.done(self.dit.runner)
            if self.gpu and not keep:
                self.dit.runner.release(keep_host=True)
            img = self.image(lat, hh, ww)
            if self.gpu and not keep:
                self.vae.runner.release(keep_host=True)
        img = np.round((img.transpose(1, 2, 0) / 2 + 0.5).clip(0, 1) * 255)
        return img.astype(np.uint8)
