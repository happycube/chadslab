"""The image input of Nemotron 3.5 Super VL (docs/NEMOTRON_SUPER_PLAN.md, item 7 and S8): the
checkpoint's image_processing_nemotron_h_omni.py, modeling_radio.py and the vision projector of
modeling_nemotron_h_omni.py, op for op in NumPy float32. The slow reference of the runtime.

    the image (RGB; an RGBA image loses its alpha, as PIL's convert) -> one tile of
    (hp x wp) patches of 16 pixels: the aspect kept, the patches between min_num_patches
    (1024) and max_num_patches (13312), each side even -> torch's antialiased bicubic
    (a = -0.5, the PIL filter) of the uint8 values as float32 -> (x / 255 - mean) / std
    -> RADIO ViT-H/16:
       the patches (channel, row, column) -> patch linear (no bias) + the CPE table
       (128 x 128 x 1280) resized bilinear (align_corners False) to max(hp, wp) square,
       then its first hp rows and wp columns -> 3 CLS + 7 register tokens in front ->
       32 blocks: x + attn(LN(x)) (16 heads of 80, qkv with bias, proj with bias), then
       x + fc2(gelu_erf(fc1(LN(x)))) (layer scale 1: none in the checkpoint);
       no final norm in RADIO
    -> the features: the patch tokens (the 10 prefix tokens dropped)
    -> vision_final_layernorm (LN 1280, eps 1e-6; there because the LLM has an MTP layer)
    -> pixel shuffle (ps v2, ratio 0.5): each 2 x 2 block of patches one row of 5120,
       channel dh * 2560 + dw * 1280 + c
    -> mlp1: RMSNorm (eps 1e-5) -> linear1 5120 -> 20480 -> relu^2 -> linear2 -> 4096
       (no biases)
    -> one row of the LLM's width for each 2 x 2 block, in raster order.

There are no 512-pixel tiles and no thumbnail: force_image_size (512) only sets the model's
unused num_image_token; the processor makes one dynamic-resolution tile per image.

The prompt sees an image as <img> + <image> x (hp * wp / 4) + </img> (ids 19, 18, 20); the
image rows replace the <image> positions in order. The chat template writes each image part
of a user message as "<image>" ("<image k><image>" when the message has several), with a
newline in front when it follows text.

    v = RadioVision("/home/cpage/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4")
    rows, n = v.image("chart.png")          # rows [n, 4096] float32
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

from reclank.models.nemotron.model import Checkpoint

IMG_CONTEXT, IMG_START, IMG_END = 18, 19, 20
VM = "vision_model.radio_model."


# ---- the tile (image_processing_nemotron_h_omni.py)

def image_budgets(n_images, min_patches=1024, max_patches=13312, max_model_len=16384, factor=2):
    """The patch budget of each image of a request (the same for all of them)."""
    budget = max((max_model_len - 4) * factor * factor, min_patches * n_images)
    cap = max_patches if max_patches and max_patches > 0 else float("inf")
    return [max(min(budget, cap), min_patches) for _ in range(n_images)]


def target_patches(width, height, budget, patch=16, min_patches=1024, factor=2):
    """The (wp, hp) patch grid of an image of width x height pixels for a budget of patches."""
    ch = round(height / patch + 0.5)
    cw = round(width / patch + 0.5)
    f = min(math.sqrt(budget / (ch * cw)), 1.0)
    th, tw = math.floor(f * ch), math.floor(f * cw)
    if budget > min_patches and th * tw < min_patches:
        up = math.sqrt(min_patches / (th * tw))
        th, tw = math.ceil(up * th), math.ceil(up * tw)
    r = th % factor
    if r:
        th = th + factor - r if (th + factor - r) * tw <= budget else max(factor, th - r)
    r = tw % factor
    if r:
        tw = tw + factor - r if th * (tw + factor - r) <= budget else max(factor, tw - r)
    return tw, th


def _cubic_aa(x):
    """The cubic filter of torch's antialiased bicubic (a = -0.5), in float32."""
    a = np.float32(-0.5)
    x = np.abs(x.astype(np.float32))
    w1 = ((a + 2) * x - (a + 3)) * x * x + 1
    w2 = ((a * x - 5 * a) * x + 8 * a) * x - 4 * a
    return np.where(x < 1, w1, np.where(x < 2, w2, 0)).astype(np.float32)


def resize_matrix(n_in, n_out):
    """[n_out, n_in] float32: one axis of torch's interpolate(bicubic, antialias=True,
    align_corners=False), the weights as aten's _compute_indices_min_size_weights_aa."""
    scale = np.float32(n_in / n_out)
    support = np.float32(2.0 * scale) if scale >= 1 else np.float32(2.0)
    inv = np.float32(1.0 / scale) if scale >= 1 else np.float32(1.0)
    most = int(math.ceil(support)) * 2 + 1
    m = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        center = np.float32(float(scale) * (i + 0.5))
        lo = max(int(float(center - support) + 0.5), 0)
        n = min(max(min(int(float(center + support) + 0.5), n_in) - lo, 0), most)
        d = (np.arange(n) + lo).astype(np.float32) - center
        w = _cubic_aa(((d.astype(np.float64) + 0.5) * float(inv)).astype(np.float32))
        tot = np.float32(0)
        for v in w:
            tot = np.float32(tot + v)
        if tot != 0:
            w = w / tot
        m[i, lo:lo + n] = w
    return m


def resize_bicubic_aa(x, height, width):
    """x [c, h, w] float32 resized to [c, height, width]: the width first, then the height
    (aten's separable order)."""
    if x.shape[2] != width:
        x = np.ascontiguousarray(np.matmul(x, resize_matrix(x.shape[2], width).T))
    if x.shape[1] != height:
        x = np.ascontiguousarray(np.matmul(resize_matrix(x.shape[1], height)[None], x))
    return x


def load_rgb(image):
    """A PIL image (or a path, or bytes) as uint8 [h, w, 3]."""
    from PIL import Image
    if isinstance(image, (str, os.PathLike)):
        image = Image.open(image)
    elif isinstance(image, (bytes, bytearray)):
        import io
        image = Image.open(io.BytesIO(image))
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def preprocess(rgb, budget, mean, std, patch=16, min_patches=1024, factor=2):
    """uint8 [h, w, 3] -> (pixels [3, hp * 16, wp * 16] float32, hp, wp)."""
    h, w = rgb.shape[:2]
    wp, hp = target_patches(w, h, budget, patch, min_patches, factor)
    x = np.ascontiguousarray(rgb.transpose(2, 0, 1)).astype(np.float32)
    x = resize_bicubic_aa(x, hp * patch, wp * patch)
    mean = np.asarray(mean, np.float32).reshape(3, 1, 1)
    std = np.asarray(std, np.float32).reshape(3, 1, 1)
    return ((x / np.float32(255.0) - mean) / std).astype(np.float32), hp, wp


# ---- the prompt (processing_nemotron_h_omni.py, chat_template.jinja)

def image_tokens(hp, wp, factor=2):
    """The number of <image> positions of an image of hp x wp patches."""
    return hp * wp // (factor * factor)


def image_span(n):
    """The token ids of an image of n rows."""
    return [IMG_START] + [IMG_CONTEXT] * n + [IMG_END]


def render_content(parts):
    """The text of a structured user message as the chat template renders it: parts are
    {"type": "image"} or {"type": "text", "text": ...} (image_url / input_image the same)."""
    images = sum(p["type"] in ("image", "image_url", "input_image") for p in parts)
    text = "".join(p.get("text") or "" for p in parts if p["type"] in ("text", "input_text"))
    if "<image>" in text:
        images = 0
    out, k, after_text = "", 0, False
    for p in parts:
        sep = "\n" if after_text and not out.endswith("\n") else ""
        if p["type"] in ("image", "image_url", "input_image"):
            k += 1
            if images > 0:
                out += sep + ("<image %d><image>" % k if images > 1 else "<image>")
                after_text = False
        elif p["type"] in ("text", "input_text"):
            out += p.get("text") or ""
            if p.get("text"):
                after_text = True
    return out


def expand_images(text, counts):
    """The processor's expansion of each <image> of the prompt text into its span."""
    for n in counts:
        assert "<image>" in text
        text = text.replace("<image>", "<img>" + "<|placeholder|>" * n + "</img>", 1)
    return text.replace("<|placeholder|>", "<image>")


# ---- the encoder (modeling_radio.py) and the projector (modeling_nemotron_h_omni.py)

def bilinear_matrix(n_in, n_out):
    """[n_out, n_in] float32: one axis of torch's interpolate(bilinear, align_corners=False)."""
    scale = np.float32(n_in / n_out)
    m = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        src = np.float32(float(scale) * (i + 0.5) - 0.5)
        src = max(src, np.float32(0))
        i0 = min(int(math.floor(src)), n_in - 1)
        l1 = np.float32(min(max(src - np.float32(i0), 0), 1))
        i1 = i0 + (1 if i0 < n_in - 1 else 0)
        m[i, i0] += np.float32(1) - l1
        m[i, i1] += l1
    return m


def layer_norm(x, w, b, eps):
    m = x.mean(axis=-1, keepdims=True)
    d = x - m
    return (d / np.sqrt((d * d).mean(axis=-1, keepdims=True) + np.float32(eps)) * w + b).astype(np.float32)


def rms_norm(x, w, eps):
    return (x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + np.float32(eps)) * w).astype(np.float32)


# erf (Cephes ndtr.c): x T(x^2) / U(x^2) for |x| <= 1; 1 - exp(-x^2) P(x) / Q(x) above (to 1e-16)
_T = [9.60497373987051638749e0, 9.00260197203842689217e1, 2.23200534594684319226e3,
      7.00332514112805075473e3, 5.55923013010394962768e4]
_U = [1.0, 3.35617141647503099647e1, 5.21357949780152679795e2, 4.59432382970980127987e3,
      2.26290000613890934246e4, 4.92673942608635921086e4]
_P = [2.46196981473530512524e-10, 5.64189564831068821977e-1, 7.46321056442269912687e0,
      4.86371970985681366614e1, 1.96520832956077098242e2, 5.26445194995477358631e2,
      9.34528527171957607540e2, 1.02755188689515710272e3, 5.57535335369399327526e2]
_Q = [1.0, 1.32281951154744992508e1, 8.67072140885989742329e1, 3.54937778887819891062e2,
      9.75708501743205489753e2, 1.82390916687909736289e3, 2.24633760818710981792e3,
      1.65666309194161350182e3, 5.57535340817727675546e2]


def _poly(c, x):
    y = np.full_like(x, c[0])
    for v in c[1:]:
        y = y * x + v
    return y


def erf(x):
    """erf of a float64 array."""
    a = np.minimum(np.abs(x), 6.0)
    z = a * a
    small = a * _poly(_T, z) / _poly(_U, z)
    big = 1.0 - np.exp(-z) * _poly(_P, a) / _poly(_Q, a)
    return np.copysign(np.where(a <= 1.0, small, big), x)


def gelu_erf(x):
    """The exact gelu (ACT2FN["gelu"]), through float64."""
    x64 = x.astype(np.float64)
    return (0.5 * x64 * (1.0 + erf(x64 * (1.0 / math.sqrt(2.0))))).astype(np.float32)


def pixel_shuffle(x, h, w):
    """[h * w, c] (raster) -> [(h / 2) * (w / 2), 4 c]: the projector's pixel_shuffle (ps v2,
    scale 0.5) as its views and permutes."""
    c = x.shape[-1]
    x = x.reshape(h, w // 2, 2 * c).transpose(1, 0, 2)
    x = x.reshape(w // 2, h // 2, 4 * c).transpose(1, 0, 2)
    return np.ascontiguousarray(x.reshape(-1, 4 * c))


class RadioVision:
    """RADIO ViT-H/16, the final layer norm, pixel shuffle and mlp1 of the checkpoint, in NumPy
    float32 (the weights widened from bfloat16 once: 3.3 GB)."""

    def __init__(self, path, rows=2048):
        self.path = path
        cfg = json.load(open(os.path.join(path, "config.json")))
        pre = json.load(open(os.path.join(path, "preprocessor_config.json")))
        vc = cfg["vision_config"]
        assert cfg["ps_version"] == "v2" and cfg["downsample_ratio"] == 0.5 and not vc["use_swiglu_ffn"]
        assert vc["hidden_act"] == "gelu" and vc["layerscale_value"] == 1.0
        self.patch = int(vc["patch_size"])
        self.width = int(vc["hidden_size"])
        self.heads = int(vc["num_attention_heads"])
        self.hd = self.width // self.heads
        self.layers = int(vc["num_hidden_layers"])
        self.eps = float(vc["layer_norm_eps"])
        self.prefix = int(vc["num_cls_tokens"]) + int(vc["num_registers"])
        self.max_side = int(vc["max_img_size"]) // self.patch
        self.factor = int(round(1.0 / cfg["downsample_ratio"]))
        self.mean, self.std = pre["norm_mean"], pre["norm_std"]
        self.min_patches = int(pre["min_num_patches"])
        self.max_patches = int(pre["max_num_patches"])
        self.max_model_len = int(pre["max_model_len"])
        self.rows = rows
        ck = Checkpoint(path)
        f = ck.f32
        g = VM + "model.patch_generator."
        self.embed = f(g + "embedder.weight")                     # [1280, 768]
        self.pos = f(g + "pos_embed").reshape(self.max_side, self.max_side, self.width)
        self.cls = f(g + "cls_token.token")                       # [10, 1280]
        self.blk = []
        for i in range(self.layers):
            b = "%smodel.blocks.%d." % (VM, i)
            self.blk.append({k: f(b + k) for k in (
                "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias",
                "mlp.fc2.weight", "mlp.fc2.bias")})
        self.fln = (f("vision_projector.vision_final_layernorm.weight"),
                    f("vision_projector.vision_final_layernorm.bias"))
        self.m_norm = f("mlp1.0.weight")
        self.m_up = f("mlp1.1.weight")                            # [20480, 5120]
        self.m_down = f("mlp1.3.weight")                          # [4096, 20480]
        for st in ck._files.values():
            st.release_pages()

    # ---- the pixels

    def pixels(self, image, n_images=1, max_tokens=None):
        """(pixels [3, H, W], hp, wp) of an image of a request of n_images images (max_tokens: at most
        that many rows, a smaller budget of patches than the checkpoint's)."""
        mp = self.max_patches
        if max_tokens:
            mp = min(mp, max(self.min_patches, self.factor * self.factor * int(max_tokens)))
        budget = image_budgets(n_images, self.min_patches, mp, self.max_model_len, self.factor)[0]
        return preprocess(load_rgb(image), budget, self.mean, self.std, self.patch,
                          self.min_patches, self.factor)

    # ---- RADIO

    def pos_embed(self, hp, wp):
        """[hp * wp, 1280]: the CPE table resized to max(hp, wp) square, its top left hp x wp."""
        d = max(hp, wp)
        mh = bilinear_matrix(self.max_side, d)[:hp]
        mw = bilinear_matrix(self.max_side, d)[:wp]
        t = np.einsum("ij,jkc->ikc", mh, self.pos)
        return np.einsum("lk,ikc->ilc", mw, t).reshape(hp * wp, self.width).astype(np.float32)

    def embeddings(self, pixels):
        """The token rows [10 + hp * wp, 1280] that enter the blocks."""
        c, hh, ww = pixels.shape
        p = self.patch
        hp, wp = hh // p, ww // p
        x = pixels.reshape(c, hp, p, wp, p).transpose(1, 3, 0, 2, 4).reshape(hp * wp, c * p * p)
        x = np.ascontiguousarray(x) @ self.embed.T + self.pos_embed(hp, wp)
        return np.concatenate([self.cls, x], axis=0).astype(np.float32)

    def attention(self, x, b):
        n, w, hd = x.shape[0], self.width, self.hd
        qkv = x @ b["attn.qkv.weight"].T + b["attn.qkv.bias"]
        q = qkv[:, :w].reshape(n, self.heads, hd).transpose(1, 0, 2)
        k = qkv[:, w:2 * w].reshape(n, self.heads, hd).transpose(1, 0, 2)
        v = qkv[:, 2 * w:].reshape(n, self.heads, hd).transpose(1, 0, 2)
        scale = np.float32(hd ** -0.5)
        out = np.empty((n, self.heads, hd), np.float32)
        for h in range(self.heads):
            kt = np.ascontiguousarray(k[h].T)
            vh = np.ascontiguousarray(v[h])
            for r in range(0, n, self.rows):
                s = (q[h, r:r + self.rows] @ kt) * scale
                s -= s.max(axis=-1, keepdims=True)
                np.exp(s, out=s)
                s /= s.sum(axis=-1, keepdims=True)
                out[r:r + self.rows, h] = s @ vh
        return out.reshape(n, w) @ b["attn.proj.weight"].T + b["attn.proj.bias"]

    def mlp(self, x, b):
        y = np.empty_like(x)
        for r in range(0, x.shape[0], self.rows):
            u = gelu_erf(x[r:r + self.rows] @ b["mlp.fc1.weight"].T + b["mlp.fc1.bias"])
            y[r:r + self.rows] = u @ b["mlp.fc2.weight"].T + b["mlp.fc2.bias"]
        return y

    def vit(self, pixels):
        """RADIO's last hidden state [10 + hp * wp, 1280]."""
        x = self.embeddings(pixels)
        for b in self.blk:
            x = x + self.attention(layer_norm(x, b["norm1.weight"], b["norm1.bias"], self.eps), b)
            x = x + self.mlp(layer_norm(x, b["norm2.weight"], b["norm2.bias"], self.eps), b)
        return x

    def features(self, pixels):
        """RADIO's features: the patch tokens [hp * wp, 1280] (CLS and registers dropped)."""
        return self.vit(pixels)[self.prefix:]

    # ---- the projector

    def project(self, feats, hp, wp):
        """The image rows [hp * wp / 4, 4096] of RADIO's features."""
        x = layer_norm(feats, self.fln[0], self.fln[1], self.eps)
        x = rms_norm(pixel_shuffle(x, hp, wp), self.m_norm, 1e-5)
        y = np.empty((x.shape[0], self.m_down.shape[0]), np.float32)
        for r in range(0, x.shape[0], self.rows):
            u = np.maximum(x[r:r + self.rows] @ self.m_up.T, 0)
            y[r:r + self.rows] = (u * u) @ self.m_down.T
        return y

    def image(self, image, n_images=1):
        """(rows [n, 4096] float32, n) of an image: the rows of its <image> positions."""
        px, hp, wp = self.pixels(image, n_images)
        rows = self.project(self.features(px), hp, wp)
        return rows, rows.shape[0]


def _bf16_bits(a):
    """float32 values widened from bfloat16 back to their bits (exact)."""
    a = np.ascontiguousarray(a, np.float32)
    return (a.view(np.uint32) >> 16).astype(np.uint16)


class RadioGPU(RadioVision):
    """RadioVision on the GPU: RADIO and the final layer norm as one program of the encoder records
    (reclank/encoders/runtime.py: ENC_LINEAR with the bfloat16 weights on the tensor cores, ENC_LNORM,
    ENC_ATTN, ENC_GELU), the projector a second one (RMSNorm, the two linears, relu^2 as a
    GP_RELU2); the CPE table, the patches and the pixel shuffle on the host. The host gives the
    blocks x0: the prefix tokens in its first rows, the CPE rows after them; the patch linear adds
    to it (the prefix rows' patches are zeros). max_patches (optional): a smaller budget of
    patches than the checkpoint's 13312 (its attention and memory)."""

    def __init__(self, path, max_patches=None):
        super().__init__(path)
        from reclank.encoders.runtime import _Runner
        from reclank.models.qwen.vision import _Lin
        if max_patches:
            self.max_patches = min(self.max_patches, int(max_patches))
        w = self.width
        self.lin_patch = _Lin(_bf16_bits(self.embed), None)
        self.gblk = []
        for b in self.blk:
            qkv, qb = b["attn.qkv.weight"], b["attn.qkv.bias"]
            self.gblk.append(dict(
                n1=(b["norm1.weight"], b["norm1.bias"]), n2=(b["norm2.weight"], b["norm2.bias"]),
                q=_Lin(_bf16_bits(qkv[:w]), qb[:w]), k=_Lin(_bf16_bits(qkv[w:2 * w]), qb[w:2 * w]),
                v=_Lin(_bf16_bits(qkv[2 * w:]), qb[2 * w:]),
                o=_Lin(_bf16_bits(b["attn.proj.weight"]), b["attn.proj.bias"]),
                fc1=_Lin(_bf16_bits(b["mlp.fc1.weight"]), b["mlp.fc1.bias"]),
                fc2=_Lin(_bf16_bits(b["mlp.fc2.weight"]), b["mlp.fc2.bias"])))
        self.lin_up = _Lin(_bf16_bits(self.m_up), None)
        self.lin_down = _Lin(_bf16_bits(self.m_down), None)
        self.scale = np.full(w, self.hd ** -0.5, np.float32)
        self.gvit = _Runner(self._vit_program, keep_gpu=1)
        self.gproj = _Runner(self._proj_program, keep_gpu=1)
        # (the float32 copies of the big matrices are not needed on the host any more)
        for b in self.blk:
            for k_ in ("attn.qkv.weight", "attn.proj.weight", "mlp.fc1.weight", "mlp.fc2.weight"):
                b[k_] = None
        self.m_up = self.m_down = None

    def _vit_program(self, n, gpu=True):
        """n tokens (the prefix and the patches): inputs "p" [n, 768] (zeros in the prefix rows), "x0"
        [n, 1280]; output "f" [n, 1280] (the final layer norm of the last hidden state)."""
        from reclank.encoders.runtime import _EncCompiler
        w, H, hd = self.width, self.heads, self.hd
        kp = self.lin_patch.w.shape[1]
        inter = self.gblk[0]["fc1"].w.shape[0]
        c = _EncCompiler(self.eps, pack=not gpu, rows=n, kmax=max(kp, w, inter))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(p=z(n, kp), x0=z(n, w), x=z(n, w), h=z(n, w), q=z(n, w), k=z(n, w), v=z(n, w), o=z(n, w),
                     t=z(n, w), g=z(n, inter), f=z(n, w), _scratch=np.zeros(n * max(kp, w, inter), np.float32))
        forms = [("set", "x", ("enc_linear", self.lin_patch, "p")), ("enc_add", "x", "x0")]
        for b in self.gblk:
            forms += [
                ("set", "h", ("enc_lnorm", "x", b["n1"][0], b["n1"][1], w)),
                ("set", "q", ("enc_linear", b["q"], "h")),
                ("set", "k", ("enc_linear", b["k"], "h")),
                ("set", "v", ("enc_linear", b["v"], "h")),
                ("set", "q", ("enc_mul_vec", "q", self.scale)),
                ("set", "o", ("enc_attn", "q", "k", "v", H, hd)),
                ("set", "t", ("enc_linear", b["o"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", b["n2"][0], b["n2"][1], w)),
                ("set", "g", ("enc_linear", b["fc1"], "h")),
                ("set", "g", ("enc_gelu", "g", True)),
                ("set", "t", ("enc_linear", b["fc2"], "g")),
                ("enc_add", "x", "t"),
            ]
        forms.append(("set", "f", ("enc_lnorm", "x", self.fln[0], self.fln[1], w)))
        c.compile(("seq", *forms))
        return c.p.finish()

    def _proj_program(self, m, gpu=True):
        """m rows of the pixel shuffle: input "b" [m, 5120]; output "y" [m, 4096]."""
        from reclank.encoders.runtime import _EncCompiler
        from reclank.ir import program as P
        kb, up, dim = self.m_norm.shape[0], self.lin_up.w.shape[0], self.lin_down.w.shape[0]
        c = _EncCompiler(1e-5, pack=not gpu, rows=m, kmax=max(kb, up))
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        c.env.update(b=z(m, kb), r=z(m, kb), u=z(m, up), y=z(m, dim), _scratch=np.zeros(m * max(kb, up), np.float32))
        c.compile(("seq", ("set", "r", ("enc_rms", "b", self.m_norm, kb)), ("set", "u", ("enc_linear", self.lin_up, "r"))))
        c.p.emit(P.RELU2, c.env["u"], c.env["u"], m * up)
        c.compile(("set", "y", ("enc_linear", self.lin_down, "u")))
        return c.p.finish()

    def image(self, image, n_images=1, max_tokens=None):
        """(rows [n, 4096] float32, n) of an image, on the GPU."""
        px, hp, wp = self.pixels(image, n_images, max_tokens)
        p = self.patch
        c = px.shape[0]
        npatch = hp * wp
        patches = px.reshape(c, hp, p, wp, p).transpose(1, 3, 0, 2, 4).reshape(npatch, c * p * p)
        n = self.prefix + npatch
        pin = np.zeros((n, c * p * p), np.float32)
        pin[self.prefix:] = patches
        x0 = np.empty((n, self.width), np.float32)
        x0[:self.prefix] = self.cls
        x0[self.prefix:] = self.pos_embed(hp, wp)
        f = self.gvit.run(n, True, {"p": pin, "x0": x0}, "f")[self.prefix:]
        b = pixel_shuffle(f, hp, wp)
        rows = self.gproj.run(b.shape[0], True, {"b": b}, "y")
        return rows, rows.shape[0]

    def release(self):
        """Free the GPU memory of the programs and the weights (the next image uploads them again)."""
        self.gvit.release()
        self.gproj.release()
