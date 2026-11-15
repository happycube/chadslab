"""Nemotron 3.5 Parse (nvidia/Nemotron-3.5-Parse, RadioMBartPrefixForCausalLM): a page image to
text with layout (boxes, classes), tables and formulas. The checkpoint's own code
(hf_nemotron_parse_modeling.py, hf_nemotron_parse_processor.py, hf_prefix_geometry.py), op for op
in NumPy float32: the slow reference of the programs of engine.py (docs/NEMOTRON_PARSE.md).

Ported from the Nemotron 3.5 Parse checkpoint's code (hf_nemotron_parse_modeling.py,
hf_nemotron_parse_processor.py, hf_prefix_geometry.py, nemotron_parse/engine/markdown.py and
text_cleanup.py, nemotron_parse/postprocessing.py; OpenMDW License 1.1; Copyright (c) 2026, NVIDIA
CORPORATION). RADIO's blocks are those of models/nemotron/radio.py. See THIRD_PARTY_NOTICES.md.

    the page (RGB) -> if taller than 2048 or wider than 1664: PIL's bilinear resize to fit, the
    aspect kept (LongestMaxSizeHW: int() of the other side) -> the content (ch, cw) -> white
    padding at the bottom to a multiple of 16 rows and at the right to a multiple of 64 columns
    (H, W) -> x / 255, (x - CLIP mean) / CLIP std -> patches of 16 x 16 (channel, row, column)
    -> C-RADIOv4-SO400M (ViT-SO400M/16, width 1152, 16 heads of 72, MLP 4304, 27 blocks, LN eps
       1e-6, no final norm): the patch linear (768 -> 1152, no bias) + the position rows: the
       learned 128 x 128 table resized (torch's antialiased bilinear) to the content's patches
       (ceil(ch / 16) x ceil(cw / 16); the average pool where a side is one patch), then sampled
       (grid_sample, bilinear, border) at each patch's centre within the content, normalized by
       the content's size (the padding patches clamp to the edge); 10 prefix tokens (CLS and
       registers) in front; x + attn(LN(x)) (qkv and proj with bias), x + fc2(gelu_erf(fc1(LN(x))))
    -> the neck: conv1 (1 x 1, 1152 -> 1024, bias) and LN1 on the patch rows; conv2 (1 x 4, stride
       4 along a row of patches, no bias) and LN2: (H / 16) x (W / 64) rows; the summary (tokens
       0 and 1, 2304) -> sum_proj -> LN3: one row after them
    -> the decoder (mBART, no positions): [the visual rows; embed(prompt) x 32] -> layernorm_embedding
       -> 10 pre-LN layers (causal attention over the whole sequence, the visual rows too: 16
       query heads and 8 KV heads of 64, scale 1/8, biases; FFN 4096 with gelu_erf; LN eps 1e-5)
       -> layer_norm -> logits = h embed^T (tied)

Generation (generation_config.json and the card's contract): greedy, repetition penalty 1.1
(transformers': each token of the prompt and of the output so far, its logit / 1.1 when positive,
x 1.1 when not), </s> forced at the last of max_new_tokens (8744; forced_eos_token_id), up to </s>.

    p = ParseRef("MODELS/Nemotron-3.5-Parse")
    ids, text = p.parse("page.png")
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

from reclank.formats.safetensors import SafeTensors
from reclank.models.nemotron.radio import RadioVision, gelu_erf, layer_norm, load_rgb

PROMPT = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>"
MAX_NEW = 8744                      # the card's max_tokens (generation_config.json has 9000)
REP_PENALTY = 1.1
PATCH = 16
PREFIX = 10                         # CLS and register tokens of RADIO
POS_SIDE = 128                      # the learned position table: 128 x 128 rows
VM = "encoder.model_encoder.radio_model.model."


class ParseConfig:
    """The fields of config.json and preprocessor_config.json that the port reads."""

    def __init__(self, path):
        cfg = json.load(open(os.path.join(path, "config.json")))
        pre = json.load(open(os.path.join(path, "preprocessor_config.json")))
        dec = cfg["decoder"]
        assert cfg.get("legacy_causal_vision_prefix") and cfg["navit_coordinate_frame"] == "content_normalized"
        assert pre["navit_packing"] and pre["navit_padding_mode"] == "top_left_white" and pre["do_normalize"]
        assert dec["activation_function"] == "gelu" and dec["scale_embedding"] and dec["add_final_layer_norm"]
        self.width = int(dec["d_model"])                            # 1024
        self.heads = int(dec["decoder_attention_heads"])            # 16
        self.kv_heads = int(dec["self_attention_kv_heads"])         # 8
        self.layers = int(dec["decoder_layers"])                    # 10
        self.ffn = int(dec["decoder_ffn_dim"])                      # 4096
        self.vocab = int(dec["vocab_size"])                         # 72512 (the tokenizer: 72505)
        self.max_seq = int(cfg["max_sequence_length"])              # 16384: the visual rows and the text
        self.eos = int(cfg["eos_token_id"])
        self.max_h, self.max_w = (int(v) for v in pre["final_size"])    # 2048, 1664
        self.mean = [float(v) for v in pre["image_mean"]]
        self.std = [float(v) for v in pre["image_std"]]
        self.vit_width = int(cfg["vision_spatial_dim"])             # 1152
        self.summary = int(cfg["vision_summary_dim"])               # 2304: tokens 0 and 1
        self.num_extra_heads = int(cfg.get("num_extra_heads", 0))


# ---- the weights

F20_FILE = "Nemotron-3.5-Parse-F20.gguf"    # scripts/convert_nemoparse_f20.py
MLP_PAD = 32                                # RADIO's MLP of 4304 padded to 4320: F20's columns of 32


def program_layout(name, a):
    """A tensor of the checkpoint (float32) in the layout of the programs, the values the same: conv1 as
    [1024, 1152]; conv2 as the linear [1024, 4096] of 4 patches' rows side by side (tap, then channel);
    pos_embed as [16384, 1152]; RADIO's mlp.fc1 (and bias) with zero rows and mlp.fc2 with zero columns
    to a multiple of 32."""
    if name.endswith("encoder.conv1.weight"):
        a = a[:, :, 0]
    elif name.endswith("encoder.conv2.weight"):
        a = a[:, :, 0, :].transpose(0, 2, 1).reshape(a.shape[0], -1)
    elif name.endswith("patch_generator.pos_embed"):
        a = a.reshape(-1, a.shape[-1])
    elif ".mlp.fc1." in name and a.shape[0] % MLP_PAD:
        a = np.pad(a, ((0, -a.shape[0] % MLP_PAD),) + ((0, 0),) * (a.ndim - 1))
    elif name.endswith(".mlp.fc2.weight") and a.shape[1] % MLP_PAD:
        a = np.pad(a, ((0, 0), (0, -a.shape[1] % MLP_PAD)))
    return np.ascontiguousarray(a, np.float32)


class Weights:
    """The tensors of a checkpoint directory in the programs' layout (program_layout): the download's
    float32 model.safetensors, or the converted F20 file (F20_FILE: each matrix F20, the vectors float32).
    f32(name) the float32 values (copies: the map's pages go after release()); f20(name) the F20 rows
    (uint8 [rows, f20_row_bytes]) of a matrix of the F20 file, else None."""

    def __init__(self, path):
        self.path = path
        f = os.path.join(path, F20_FILE)
        if os.path.exists(f):
            from reclank.formats.gguf import GGUF
            self.g, self.st = GGUF(f, stage=False), None
            self.kind = "F20"
        else:
            self.g, self.st = None, SafeTensors(os.path.join(path, "model.safetensors"))
            self.kind = "float32"

    def names(self):
        return list(self.g.tensors) if self.g is not None else self.st.names()

    def f32(self, name):
        if self.g is not None:
            return np.array(self.g.dequant(name), np.float32)
        return program_layout(name, np.array(self.st.get(name), np.float32))

    def f20(self, name):
        """(F20 rows, rows, cols) of a matrix of the F20 file, else None."""
        if self.g is None:
            return None
        from reclank.formats.gguf import F20
        raw, dims, t = self.g.raw(name)
        if t != F20:
            return None
        cols, rows = int(dims[0]), int(dims[1])
        return np.array(raw).reshape(rows, -1), rows, cols

    def release(self):
        """Drop the pages of the map (the arrays given out are copies)."""
        if self.st is not None:
            self.st.release_pages()


# ---- the tokens (tokenizer.json: byte-level BPE; the prompts are added tokens only)

def _byte_decoder():
    """GPT-2's map of the characters of byte-level BPE tokens to bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) + \
        list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class Tokens:
    """The tokens of tokenizer.json: text() of ids as the tokenizers library's decode with
    skip_special_tokens False (a token whose characters are all in the byte-level map gives those
    bytes, another, as the added tokens " | " or "\\n\\n", its own UTF-8; then the bytes as UTF-8,
    errors replaced), and encode() of a prompt made of added tokens (the task's control tokens)."""

    def __init__(self, path):
        t = json.load(open(os.path.join(path, "tokenizer.json"), encoding="utf-8"))
        self.strs = {i: s for s, i in t["model"]["vocab"].items()}
        self.added = {a["content"]: a["id"] for a in t["added_tokens"]}
        self.special = {a["id"] for a in t["added_tokens"] if a["special"]}
        for s, i in self.added.items():
            self.strs[i] = s
        self.bytes = _byte_decoder()
        self._pieces = {}
        # the added tokens longest first (the prompt's split)
        self._re = re.compile("|".join(re.escape(s) for s in sorted(self.added, key=len, reverse=True)))
        self.x_ids = np.array(sorted(i for s, i in self.added.items() if s.startswith("<x_")), np.int64)
        self.class_ids = {i for s, i in self.added.items() if s.startswith("<class_")}

    def id(self, s):
        return self.added[s]

    def piece(self, i):
        """The bytes of token i."""
        p = self._pieces.get(i)
        if p is None:
            s = self.strs.get(int(i), "")
            p = bytes(self.bytes[c] for c in s) if all(c in self.bytes for c in s) else s.encode("utf-8")
            self._pieces[i] = p
        return p

    def text(self, ids):
        return b"".join(self.piece(int(i)) for i in ids).decode("utf-8", "replace")

    def encode(self, prompt):
        """The ids of a prompt of added tokens ("</s><s><predict_bbox>..."; ValueError on other text)."""
        ids, pos = [], 0
        for m in self._re.finditer(prompt):
            if m.start() != pos:
                break
            ids.append(self.added[m.group(0)])
            pos = m.end()
        if pos != len(prompt) or not ids:
            raise ValueError("a prompt of Nemotron Parse is its control tokens only (as %r)" % PROMPT)
        return ids


# ---- the page (hf_nemotron_parse_processor.py)

def page_pixels(image, cfg):
    """(patches [P, 768] float32, (H, W), (ch, cw)) of a page (a file, bytes, a PIL image or uint8
    [h, w, 3]): the processor's resize, white padding, normalization and packed patches."""
    from PIL import Image
    rgb = image if isinstance(image, np.ndarray) else load_rgb(image)
    h, w = rgb.shape[:2]
    aspect = w / h
    nh, nw = h, w
    if h > cfg.max_h:
        nh = cfg.max_h
        nw = int(nh * aspect)
    if nw > cfg.max_w:
        nw = cfg.max_w
        nh = int(nw / aspect)
    if (nh, nw) != (h, w):
        rgb = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.Resampling.BILINEAR))
    ch, cw = rgb.shape[:2]
    H, W = -(-ch // 16) * 16, -(-cw // 64) * 64
    page = np.full((H, W, 3), 255, np.uint8)
    page[:ch, :cw] = rgb
    x = page.transpose(2, 0, 1).astype(np.float32) / np.float32(255.0)
    x = (x - np.asarray(cfg.mean, np.float32).reshape(3, 1, 1)) / np.asarray(cfg.std, np.float32).reshape(3, 1, 1)
    R, C = H // PATCH, W // PATCH
    patches = x.reshape(3, R, PATCH, C, PATCH).transpose(1, 3, 0, 2, 4).reshape(R * C, 3 * PATCH * PATCH)
    return np.ascontiguousarray(patches, np.float32), (H, W), (ch, cw)


# ---- the position rows (hf_prefix_geometry.py, content_normalized)

def aa_linear_matrix(n_in, n_out):
    """[n_out, n_in] float32: one axis of torch's interpolate(bilinear, antialias=True,
    align_corners=False), the weights of aten's _compute_weights_aa in float32."""
    f32 = np.float32
    scale = f32(n_in / n_out)
    support = f32(scale) if scale >= 1 else f32(1.0)
    inv = f32(1.0) / scale if scale >= 1 else f32(1.0)
    m = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        center = scale * f32(i + 0.5)
        lo = max(int(center - support + f32(0.5)), 0)
        n = max(min(int(center + support + f32(0.5)), n_in) - lo, 0)
        w = []
        for j in range(n):
            d = abs((f32(j + lo) - center + f32(0.5)) * inv)
            w.append(f32(1) - d if d < 1 else f32(0))
        tot = f32(0)
        for v in w:
            tot = f32(tot + v)
        for j, v in enumerate(w):
            m[i, lo + j] = v / tot if tot != 0 else v
    return m


def avg_pool_matrix(n_in, n_out):
    """[n_out, n_in] float32: one axis of adaptive_avg_pool2d."""
    m = np.zeros((n_out, n_in), np.float32)
    for i in range(n_out):
        a, b = (i * n_in) // n_out, -(-((i + 1) * n_in) // n_out)
        m[i, a:b] = np.float32(1.0) / np.float32(b - a)
    return m


def sample_matrix(n, content, size):
    """[n, size] float32: grid_sample (bilinear, border, align_corners False) at the centres of n
    patches of 16 pixels along an axis of content pixels, over a table of size rows: the patch's
    centre within the content (its part past the content's edge cut off), normalized by content."""
    f32 = np.float32
    m = np.zeros((n, size), np.float32)
    for i in range(n):
        start = f32(i * PATCH)
        ext = min(max(f32(content) - start, f32(0)), f32(PATCH))
        g = min(start + f32(0.5) * ext, f32(content)) / f32(content)
        g = f32(2.0) * g - f32(1.0)
        x = ((g + f32(1)) * f32(size) - f32(1)) / f32(2)
        x = min(max(x, f32(0)), f32(size - 1))
        x0 = int(math.floor(x))
        w1 = x - f32(x0)
        m[i, x0] += f32(1) - w1
        if w1 != 0:
            m[i, x0 + 1] += w1
    return m


def position_rows(table, hw, content):
    """[R * C, 1152]: the position rows of a page of hw = (H, W) pixels with content = (ch, cw);
    table [128, 128, 1152] float32."""
    (H, W), (ch, cw) = hw, content
    R, C = H // PATCH, W // PATCH
    cr, cc = -(-ch // PATCH), -(-cw // PATCH)
    resize = avg_pool_matrix if min(cr, cc) == 1 else aa_linear_matrix
    my = sample_matrix(R, ch, cr) @ resize(POS_SIDE, cr)        # [R, 128]
    mx = sample_matrix(C, cw, cc) @ resize(POS_SIDE, cc)        # [C, 128]
    t = (my @ table.reshape(POS_SIDE, -1)).reshape(R, POS_SIDE, -1)     # [R, 128, 1152]
    return np.matmul(mx[None], t).reshape(R * C, -1).astype(np.float32)


# ---- the reference

class ParseRef(RadioVision):
    """The whole model in NumPy float32 (RADIO's blocks, attention() and mlp() of RadioVision: the
    same names of timm's Block), and greedy generation with a cache of keys and values."""

    def __init__(self, path, rows=2048):
        self.path = path
        self.cfg = cfg = ParseConfig(path)
        W = Weights(path)
        f = W.f32
        self.patch = PATCH
        self.width = cfg.vit_width
        self.heads = 16
        self.hd = self.width // self.heads
        self.layers = sum(1 for n in W.names() if n.startswith(VM + "blocks.") and n.endswith(".norm1.weight"))
        self.eps = 1e-6
        self.prefix = PREFIX
        self.rows = rows
        self.tok = Tokens(path)
        self.embed = f(VM + "patch_generator.embedder.weight")                  # [1152, 768]
        self.pos = f(VM + "patch_generator.pos_embed").reshape(POS_SIDE, POS_SIDE, self.width)
        self.cls = f(VM + "patch_generator.cls_token.token")                    # [10, 1152]
        self.blk = []
        for i in range(self.layers):
            b = "%sblocks.%d." % (VM, i)
            self.blk.append({k: f(b + k) for k in (
                "norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                "attn.proj.bias", "norm2.weight", "norm2.bias", "mlp.fc1.weight", "mlp.fc1.bias",
                "mlp.fc2.weight", "mlp.fc2.bias")})
        e = "encoder."
        self.conv1 = (f(e + "conv1.weight"), f(e + "conv1.bias"))                # [1024, 1152]
        self.ln1 = (f(e + "layer_norm1.weight"), f(e + "layer_norm1.bias"))
        # conv2 (1 x 4, stride 4): a linear of 4 patches' rows side by side ((tap, channel) columns)
        self.conv2 = f(e + "conv2.weight")                                      # [1024, 4096]
        self.ln2 = (f(e + "layer_norm2.weight"), f(e + "layer_norm2.bias"))
        self.sum_proj = (f(e + "sum_proj.weight"), f(e + "sum_proj.bias"))
        self.ln3 = (f(e + "layer_norm3.weight"), f(e + "layer_norm3.bias"))
        d = "decoder."
        self.tok_embed = f(d + "embed_tokens.weight")                           # [72512, 1024]
        self.ln_emb = (f(d + "layernorm_embedding.weight"), f(d + "layernorm_embedding.bias"))
        self.ln_end = (f(d + "layer_norm.weight"), f(d + "layer_norm.bias"))
        self.dec = []
        for i in range(cfg.layers):
            g = lambda n: f("%slayers.%d.%s" % (d, i, n))  # noqa: E731
            self.dec.append({k: g(k) for k in (
                "self_attn_layer_norm.weight", "self_attn_layer_norm.bias", "self_attn.q_proj.weight",
                "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias",
                "self_attn.v_proj.weight", "self_attn.v_proj.bias", "self_attn.out_proj.weight",
                "self_attn.out_proj.bias", "final_layer_norm.weight", "final_layer_norm.bias",
                "fc1.weight", "fc1.bias", "fc2.weight", "fc2.bias")})
        W.release()

    # ---- the encoder

    def pixels(self, image):
        return page_pixels(image, self.cfg)

    def embeddings(self, patches, hw, content):
        """The token rows [10 + R C, 1152] that enter the blocks."""
        x = patches @ self.embed.T + position_rows(self.pos, hw, content)
        return np.concatenate([self.cls, x.astype(np.float32)], axis=0)

    def radio(self, patches, hw, content):
        """RADIO's last hidden state [10 + R C, 1152]."""
        x = self.embeddings(patches, hw, content)
        for b in self.blk:
            x = x + self.attention(layer_norm(x, b["norm1.weight"], b["norm1.bias"], self.eps), b)
            x = x + self.mlp(layer_norm(x, b["norm2.weight"], b["norm2.bias"], self.eps), b)
        return x

    def neck(self, hidden, hw):
        """The visual rows [(H / 16) (W / 64) + 1, 1024] of RADIO's last hidden state."""
        R, C = hw[0] // PATCH, hw[1] // PATCH
        feat = hidden[PREFIX:] @ self.conv1[0].T + self.conv1[1]
        feat = layer_norm(feat, *self.ln1, 1e-6)
        feat = feat.reshape(R, C // 4, 4 * feat.shape[1]) @ self.conv2.T
        feat = layer_norm(feat.reshape(R * C // 4, -1), *self.ln2, 1e-6)
        summ = hidden[:2].reshape(1, -1) @ self.sum_proj[0].T + self.sum_proj[1]
        summ = layer_norm(summ, *self.ln3, 1e-6)
        return np.concatenate([feat, summ], axis=0).astype(np.float32)

    def encode(self, image):
        """(the visual rows, hw, content) of a page."""
        patches, hw, content = self.pixels(image)
        return self.neck(self.radio(patches, hw, content), hw), hw, content

    # ---- the decoder

    def new_cache(self):
        return [[np.zeros((0, self.cfg.kv_heads * 64), np.float32)] * 2 for _ in range(self.cfg.layers)]

    def decode(self, x, cache, taps=()):
        """The final hidden rows of x [t, 1024] (embeddings) after the rows in cache (each layer's
        keys and values, extended in place). taps: layers whose outputs (the residual stream) go to
        self.tapped [t, 1024 len(taps)] (the DFlash2 drafter's context)."""
        c = self.cfg
        H, KV, hd = c.heads, c.kv_heads, c.width // c.heads
        t = x.shape[0]
        h = layer_norm(x, *self.ln_emb, 1e-5)
        tapped = []
        for li, (L, kv) in enumerate(zip(self.dec, cache)):
            a = layer_norm(h, L["self_attn_layer_norm.weight"], L["self_attn_layer_norm.bias"], 1e-5)
            q = a @ L["self_attn.q_proj.weight"].T + L["self_attn.q_proj.bias"]
            k = a @ L["self_attn.k_proj.weight"].T + L["self_attn.k_proj.bias"]
            v = a @ L["self_attn.v_proj.weight"].T + L["self_attn.v_proj.bias"]
            kv[0] = np.concatenate([kv[0], k.astype(np.float32)])
            kv[1] = np.concatenate([kv[1], v.astype(np.float32)])
            n = kv[0].shape[0]
            past = n - t
            o = np.empty((t, H, hd), np.float32)
            kk = kv[0].reshape(n, KV, hd)
            vv = kv[1].reshape(n, KV, hd)
            qq = q.reshape(t, H, hd)
            for hh in range(H):
                kt = np.ascontiguousarray(kk[:, hh // (H // KV)].T)
                vh = np.ascontiguousarray(vv[:, hh // (H // KV)])
                for r in range(0, t, self.rows):
                    r1 = min(t, r + self.rows)
                    s = (qq[r:r1, hh] @ kt) * np.float32(hd ** -0.5)
                    s[np.arange(n)[None, :] > np.arange(r, r1)[:, None] + past] = -np.inf
                    s -= s.max(axis=-1, keepdims=True)
                    np.exp(s, out=s)
                    s /= s.sum(axis=-1, keepdims=True)
                    o[r:r1, hh] = s @ vh
            h = h + (o.reshape(t, -1) @ L["self_attn.out_proj.weight"].T + L["self_attn.out_proj.bias"])
            a = layer_norm(h, L["final_layer_norm.weight"], L["final_layer_norm.bias"], 1e-5)
            u = gelu_erf(a @ L["fc1.weight"].T + L["fc1.bias"])
            h = (h + (u @ L["fc2.weight"].T + L["fc2.bias"])).astype(np.float32)
            if li in taps:
                tapped.append(h)
        self.tapped = np.concatenate(tapped, axis=1) if tapped else None
        return layer_norm(h, *self.ln_end, 1e-5)

    def text_rows(self, ids):
        return self.tok_embed[np.asarray(ids, np.int64)] * np.float32(math.sqrt(self.cfg.width))

    def logits(self, h):
        return (h @ self.tok_embed.T).astype(np.float32)

    def generate(self, vis, prompt_ids, max_new=MAX_NEW, forced=None):
        """(new ids, the logits of each step) after the visual rows vis and the prompt: greedy with the
        rules of next_token. forced: the ids to feed instead of the chosen ones (teacher forcing)."""
        cache = self.new_cache()
        h = self.decode(np.concatenate([vis, self.text_rows(prompt_ids)]), cache)[-1:]
        seq, out, logs = list(prompt_ids), [], []
        while True:
            lg = self.logits(h)[0]
            logs.append(lg)
            t = next_token(lg, seq, len(out), max_new, self.cfg.eos)
            if forced is not None:
                if len(out) >= len(forced):
                    break
                t = int(forced[len(out)])
            out.append(t)
            seq.append(t)
            if forced is None and (t == self.cfg.eos or len(out) >= max_new):
                break
            h = self.decode(self.text_rows([t]), cache)
        return out, logs

    def parse(self, image, prompt=PROMPT, max_new=MAX_NEW):
        vis = self.encode(image)[0]
        out = self.generate(vis, self.tok.encode(prompt), max_new)[0]
        return out, self.tok.text(out)


def apply_rules(lg, seq, n_new, max_new, eos, penalty=REP_PENALTY):
    """The logits processors of generate() in place on one row: the repetition penalty over the
    tokens of seq (the prompt and the output so far), and </s> alone at the last of max_new."""
    if penalty != 1.0:
        ids = np.unique(np.asarray(seq, np.int64))
        s = lg[ids]
        lg[ids] = np.where(s < 0, s * np.float32(penalty), s / np.float32(penalty))
    if n_new == max_new - 1:
        lg[:] = -np.inf
        lg[eos] = 0
    return lg


class Rules:
    """apply_rules and the greedy choice with the set of tokens seen so far kept as they come (a step's
    cost does not grow with the output)."""

    def __init__(self, prompt, max_new, eos, penalty=REP_PENALTY):
        self.max_new, self.eos, self.penalty = max_new, eos, np.float32(penalty)
        self.seen = set()
        self.ids = np.zeros(0, np.int64)
        for t in prompt:
            self.add(t)

    def add(self, t):
        if t not in self.seen:
            self.seen.add(t)
            self.ids = np.append(self.ids, np.int64(t))

    def choose(self, lg, n_new, sampler=None):
        """The token after logits lg (not changed) when n_new tokens came before it: the most likely,
        or sampler(the processed logits)."""
        if n_new == self.max_new - 1:
            return self.eos
        if self.penalty != 1:
            s = lg[self.ids]
            lg = lg.copy()
            lg[self.ids] = np.where(s < 0, s * self.penalty, s / self.penalty)
        return int(np.argmax(lg)) if sampler is None else sampler(lg)


class Sampler:
    """A sampled token of processed logits: temperature, then top_k (0: all) and top_p, then a draw of
    a generator of seed (the order of vLLM's sampler; the card's contract is greedy, the checkpoint's
    DocumentParser samples its crops and detections at temperature 0.1 or 0.2, top-k 50)."""

    def __init__(self, temperature, top_k=0, top_p=1.0, seed=0):
        self.t, self.k, self.p = float(temperature), int(top_k), float(top_p)
        self.rng = np.random.default_rng(seed)

    def __call__(self, lg):
        x = lg.astype(np.float64) / self.t
        if 0 < self.k < x.size:
            x[np.argpartition(x, -self.k)[:-self.k]] = -np.inf
        x = np.exp(x - x.max())
        x /= x.sum()
        if self.p < 1:
            o = np.argsort(-x, kind="stable")
            c = np.cumsum(x[o])
            cut = o[int(np.searchsorted(c, self.p)) + 1:]
            x[cut] = 0
            x /= x.sum()
        return int(self.rng.choice(x.size, p=x))


def next_token(lg, seq, n_new, max_new, eos, penalty=REP_PENALTY):
    """The greedy token after the rules (lg is not changed)."""
    return int(np.argmax(apply_rules(lg.copy(), seq, n_new, max_new, eos, penalty)))


# ---- the output (nemotron_parse/engine/markdown.py, text_cleanup.py, postprocessing.py)

_NUM = r"\d+(?:\.\d+)?"
_BLOCK = re.compile(r"<x_(%s)><y_(%s)>(.*?)<x_(%s)><y_(%s)><class_([^>]+)>" % ((_NUM,) * 4), re.S)
_DEDUPE = re.compile(r"<x_([\d.]+)><y_([\d.]+)>(.*?)<x_([\d.]+)><y_([\d.]+)><class_([^>]+)>", re.S)
_EMPTY = re.compile(r"(?m)^[ \t]*<x_%s><y_%s><class_[A-Za-z][A-Za-z_-]*>\s*(?=<x_%s><y_%s>)" % ((_NUM,) * 4))
_ROLES = {"Text", "List-item", "Caption", "Section-header", "Page-header", "Page-footer", "Footnote", "Title"}
_SOFT = re.compile(r"(?<=[^\W\d_])­(?=[^\W\d_])")
_BULLET = re.compile(r"^[ \t]*[•◦▪●‣⁃–—][ \t]+")


def clean_raw(raw):
    """The raw text without the control tokens of the prompt and </s>, an empty block's prefix
    removed (repair_empty_prefix), each repeated block (class, box, text) after its first dropped
    (dedupe_exact_elements)."""
    raw = re.sub(r"^(?:</s>|<s>|<predict_[a-z_]+>|<no_[a-z]+>|<output_[a-z_]+>)+", "", raw or "")
    raw = raw.replace("</s>", "")
    if "```" not in raw and not re.search(r"<class_(?:Code|Formula)>", raw):
        raw = _EMPTY.sub("", raw)
    seen = set()

    def one(m):
        key = (m[6], tuple(map(float, (m[1], m[2], m[4], m[5]))), m[3])
        if key in seen:
            return ""
        seen.add(key)
        return m[0]
    return _DEDUPE.sub(one, raw)


def _normalize(cls, text):
    if cls in _ROLES and not any(s in text for s in ("`", "$", r"\(", r"\[")):
        text = _SOFT.sub("", text)
    if cls == "List-item":
        text = _BULLET.sub("- ", text, count=1)
    return text


def blocks(raw, image_size=None):
    """{"blocks": [{"class", "bbox" (x1, y1, x2, y2 within the content, 0..1), "bbox_pixels" (with
    image_size = (width, height) of the page), "text"}], "markdown", "raw_out"} of a raw output.
    The markdown joins the blocks' texts in order (# for a Title, ## for a Section-header); tables
    stay LaTeX (the checkpoint's nemotron_parse package renders them as HTML)."""
    cleaned = clean_raw(raw)
    out, md = [], []
    for m in _BLOCK.finditer(cleaned):
        x1, y1, body, x2, y2, cls = m.groups()
        box = [float(x1), float(y1), float(x2), float(y2)]
        b = {"class": cls, "bbox": box, "text": body}
        if image_size is not None:
            w, h = image_size
            xa, xb = sorted((box[0], box[2]))
            ya, yb = sorted((box[1], box[3]))
            b["bbox_pixels"] = [min(s, max(0.0, v * s)) for v, s in ((xa, w), (ya, h), (xb, w), (yb, h))]
        out.append(b)
        t = _normalize(cls, body).replace("<tbc>", "").strip()
        if t:
            if cls == "Title" and not t.startswith("#"):
                t = "# " + t
            elif cls == "Section-header" and not t.startswith("#"):
                t = "## " + t
            md.append(t)
    if not out:
        md = [re.sub(r"<x_[^>]+>|<y_[^>]+>|<class_[^>]+>", "", cleaned).replace("<tbc>", "").strip()]
    return {"blocks": out, "markdown": "\n\n".join(md).strip(), "raw_out": cleaned,
            "coordinate_frame": "content_normalized"}
