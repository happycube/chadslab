"""EmbeddingGemma 2 (google/embeddinggemma-2): text in, one vector of 768
floats out (sentence-transformers: the model, the mean of its token rows,
then the L2 norm).

Ported from transformers
(src/transformers/models/embedding_gemma2/modeling_embedding_gemma2.py; Apache
License 2.0; Copyright 2026 the HuggingFace Team). See THIRD_PARTY_NOTICES.md.

The text model follows transformers (models/embedding_gemma2): an encoder
of 24 layers of width 512, every layer bidirectional, no cache:

    the token embeddings times sqrt(512)
    the per-layer inputs (PLE, from the embeddings alone): one linear to
        24 x 512, times 512^-0.5, an RMS norm of each 512
    each layer: RMS norm, q k v (4 query heads; the sliding layers 2 KV
        heads of 256, the full layers 1 of 512), an RMS norm of each head of
        q and k and one with no weight of v, RoPE (NEOX halves; theta 10^4
        sliding, 10^6 full), attention of scale 1 (sliding: the keys within
        512 on each side), o, a post norm, + x; RMS norm, the gated FFN with
        gelu tanh, a post norm, + x; the PLE block (gelu(gate x) times the
        layer's per-layer input, a projection, a post norm, + x); times the
        layer scalar
    the final norm, the projection to 768

The projection is linear, so the mean of the token rows goes before it (one
row per input, not one per token). Matryoshka: the first 512, 256 or 128
values, normalized again.

Audio (audio_rows): the Conformer of Gemma 4 (models/gemma4/encoders.py
Gemma4Audio, written for the mmproj GGUF of llama.cpp; _AudioWeights gives it
the tensors of the checkpoint under the GGUF names) on at most 30 s of 16 kHz
mono (the feature extractor of transformers cuts at 480000 samples): 25 rows
a second, each through the RMS norm with no weight and the projection to
512 (embed_audio). An audio input is <bos> <|audio> the rows <audio|> <eos>;
the rows take the place of the embeddings of the placeholders (with no
sqrt(512)), and no prompt goes before them.

Images (image_rows): the vision tower of Gemma 4 (Gemma4Vision, the GGUF
names again: _VisionWeights): the image resized to a budget of soft tokens
(70, 140, 280 (the default), 560, 1120: 9 patches of 16 x 16 each), the
patches through 16 layers, each 3 x 3 block averaged into one row, the
projection to 512 (embed_vision). An image input is <bos> <|image> the rows
<image|> <eos>.

Each tower (text, audio, vision) runs on the GPU or the CPU (devices): a
program on the CPU takes a team of cpu_threads (program.threads), so that a
process of the GPU (OMP_NUM_THREADS=1) still has threads for it.

The text model runs as a program (reclank/ir/program.py, the records
ENC_*, as the media encoders): in C on the CPU (the bfloat16 matrices in
groups of 16 rows), or as one CUDA graph on the GPU (gpu=True: the matrices
in BF12, encoders.runtime.to_bf12). Its records of its own: ENC_ROPE1D (the
1D RoPE) and ENC_ATTN_RANGE (each query over a range of keys: the sliding
window, every key, and the inputs packed in one pass each over its own
keys; KV heads shared). A program is made for a count of rows (_bucket: a
few sizes an octave; the rows past the inputs are zeros, each over itself
alone). The table of the token embeddings stays on the host in bfloat16 (the
rows of the ids, times sqrt(512)); the mean of the rows of each input and the
projection to 768 are on the host. RECLANK_ENC_PY=1 runs the NumPy layers of
forward_numpy instead (the reference of the program). All the arithmetic is
float32 (on the GPU the BF12 matrices), as the float32 reference
(scripts/embedgemma2_reference.py).
"""
from __future__ import annotations

import json
import math
import os
import subprocess

import numpy as np

from reclank.core import ops
from reclank.encoders.runtime import _ENC_PY, _EncCompiler, _Plain, _Runner, _shape, to_bf12
from reclank.formats.safetensors import SafeTensors
from reclank.runtime.threads import team_threads
from reclank.models.gemma4.tokenizer import Tokenizer

DIMS = (768, 512, 256, 128)
AUDIO_RATE = 16000
AUDIO_MAX_S = 30        # the cut of the feature extractor (480000 samples)


def decode_audio(src, start=None, end=None):
    """16 kHz mono float32 samples of an audio file (a path, or the bytes of
    a file) in any format of ffmpeg (MP3, WAV, FLAC, OGG, M4A, ...): ffmpeg
    mixes the channels and resamples. start, end: seconds (a cut)."""
    cmd = ["ffmpeg", "-v", "error", "-nostdin"]
    if start:
        cmd += ["-ss", "%.3f" % start]
    if end is not None:
        cmd += ["-to", "%.3f" % end]
    data = None
    if isinstance(src, (bytes, bytearray)):
        cmd += ["-i", "pipe:0"]
        data = bytes(src)
    else:
        cmd += ["-i", str(src)]
    cmd += ["-f", "f32le", "-ac", "1", "-ar", str(AUDIO_RATE), "pipe:1"]
    # (not reclank's OMP_PROC_BIND: ffmpeg links libgomp, which binds it to one CPU)
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_PROC_BIND", "OMP_PLACES")}
    r = subprocess.run(cmd, input=data, capture_output=True, env=env)
    if r.returncode != 0:
        raise ValueError("ffmpeg cannot decode the audio: %s" % r.stderr.decode("utf-8", "replace").strip()[-300:])
    return np.frombuffer(r.stdout, dtype=np.float32).copy()


class _VisionWeights:
    """The vision tower of the checkpoint under the names of the mmproj GGUF
    that Gemma4Vision reads. The GGUF patch linear is a conv (width,
    channel, row, column); the checkpoint's is a linear over (row, column,
    channel)."""

    BLOCK = {"ln1": "input_layernorm", "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj",
             "attn_v": "self_attn.v_proj", "attn_out": "self_attn.o_proj",
             "attn_q_norm": "self_attn.q_norm", "attn_k_norm": "self_attn.k_norm",
             "attn_post_norm": "post_attention_layernorm", "ln2": "pre_feedforward_layernorm",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
             "ffn_post_norm": "post_feedforward_layernorm"}

    def __init__(self, st, cfg):
        self.st = st
        self.meta = {"clip.vision.projector_type": "gemma4v",
                     "clip.vision.embedding_length": cfg["hidden_size"],
                     "clip.vision.block_count": cfg["num_hidden_layers"],
                     "clip.vision.attention.head_count": cfg["num_attention_heads"],
                     "clip.vision.attention.layer_norm_epsilon": cfg.get("rms_norm_eps", 1e-6)}
        self.width = cfg["hidden_size"]

    def _name(self, name):
        base, _, leaf = name.rpartition(".")
        if base == "v.patch_embd":
            return "vision_tower.patch_embedder.input_proj.weight"
        if base == "v.position_embd":
            return "vision_tower.patch_embedder.position_embedding_table"
        if base == "mm.input_projection":
            return "embed_vision.embedding_projection." + leaf
        if base.startswith("v.blk."):
            _, _, i, key = base.split(".", 3)
            if key not in self.BLOCK:
                return None
            hf = "vision_tower.encoder.layers.%s.%s" % (i, self.BLOCK[key])
            if leaf == "weight" and hf + ".linear.weight" in self.st.header:
                return hf + ".linear.weight"
            return hf + "." + leaf
        return None

    def has(self, name):
        n = self._name(name)
        return n is not None and n in self.st.header

    def f32(self, name):
        x = np.ascontiguousarray(self.st.get(self._name(name)), dtype=np.float32)
        if name == "v.patch_embd.weight":
            # (width, row, column, channel) -> the GGUF conv (width, channel, row, column)
            p = int(round(math.sqrt(x.shape[1] // 3)))
            x = np.ascontiguousarray(x.reshape(self.width, p, p, 3).transpose(0, 3, 1, 2))
        return x

    def scalar(self, name):
        return float(self.f32(name).reshape(-1)[0]) if self.has(name) else None

    def matrix(self, name):
        return np.ascontiguousarray(self.st.get_bf16(self._name(name)))


class _AudioWeights:
    """The audio tower of the checkpoint under the names of the mmproj GGUF
    that Gemma4Audio reads, with the changes of the llama.cpp converter:
    per_dim_scale holds softplus(per_dim_scale), the depthwise conv has no
    middle axis, and the lconv1d norms trade names (the GGUF conv_norm is
    pre_layer_norm, norm_conv is conv_norm). A clippable linear's scalars
    (input_min ...) sit beside its matrix."""

    BLOCK = {"ffn_norm": "feed_forward1.pre_layer_norm", "ffn_up": "feed_forward1.ffw_layer_1",
             "ffn_down": "feed_forward1.ffw_layer_2", "ffn_post_norm": "feed_forward1.post_layer_norm",
             "ffn_norm_1": "feed_forward2.pre_layer_norm", "ffn_up_1": "feed_forward2.ffw_layer_1",
             "ffn_down_1": "feed_forward2.ffw_layer_2", "ffn_post_norm_1": "feed_forward2.post_layer_norm",
             "attn_pre_norm": "norm_pre_attn", "attn_post_norm": "norm_post_attn",
             "attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj",
             "attn_out": "self_attn.post", "attn_k_rel": "self_attn.relative_k_proj",
             "per_dim_scale": "self_attn.per_dim_scale", "conv_norm": "lconv1d.pre_layer_norm",
             "norm_conv": "lconv1d.conv_norm", "conv_pw1": "lconv1d.linear_start",
             "conv_pw2": "lconv1d.linear_end", "conv_dw": "lconv1d.depthwise_conv1d", "ln2": "norm_out"}
    TOP = {"a.input_projection": "audio_tower.subsample_conv_projection.input_proj_linear",
           "a.pre_encode.out": "audio_tower.output_proj",
           "mm.a.input_projection": "embed_audio.embedding_projection"}

    def __init__(self, st, cfg):
        self.st = st
        self.meta = {"clip.audio.projector_type": "gemma4a",
                     "clip.audio.embedding_length": cfg["hidden_size"],
                     "clip.audio.block_count": cfg["num_hidden_layers"],
                     "clip.audio.attention.head_count": cfg["num_attention_heads"]}

    def _name(self, name):
        """The checkpoint name of a GGUF name (None: no such tensor)."""
        base, _, leaf = name.rpartition(".")
        if base.startswith("a.conv1d."):
            parts = base.split(".")         # a.conv1d.i[.norm]
            hf = "audio_tower.subsample_conv_projection.layer%s.%s" % (
                parts[2], "norm" if parts[-1] == "norm" else "conv")
        elif base.startswith("a.blk."):
            _, _, i, key = base.split(".", 3)
            if key not in self.BLOCK:
                return None
            hf = "audio_tower.layers.%s.%s" % (i, self.BLOCK[key])
            if key == "per_dim_scale":
                return hf
        elif base in self.TOP:
            hf = self.TOP[base]
        else:
            return None
        if leaf == "weight" and hf + ".linear.weight" in self.st.header:
            return hf + ".linear.weight"       # a clippable linear
        return hf + "." + leaf

    def has(self, name):
        n = self._name(name)
        return n is not None and n in self.st.header

    def f32(self, name):
        x = np.ascontiguousarray(self.st.get(self._name(name)), dtype=np.float32)
        if ".per_dim_scale." in name:
            x = np.logaddexp(0.0, x).astype(np.float32)     # softplus
        if ".conv_dw." in name:
            x = x.reshape(x.shape[0], -1)
        return x

    def scalar(self, name):
        return float(self.f32(name).reshape(-1)[0]) if self.has(name) else None

    def matrix(self, name):
        """A (out, in) matrix as raw bfloat16 (uint16)."""
        return np.ascontiguousarray(self.st.get_bf16(self._name(name)))


def _bucket(n):
    """The rows of the program of n rows: 64 at least, else the next of 8
    sizes an octave (a multiple of 16; at most 1/8 more rows)."""
    if n <= 64:
        return 64
    p = 1 << (int(n).bit_length() - 1)
    for k in range(8, 17):
        size = -(-p * k // 8 // 16) * 16
        if size >= n:
            return size
    return 2 * p


class EmbeddingGemma2:
    """The text model of an embeddinggemma-2 directory. embed(texts) returns
    the vectors (len(texts), dim). gpu: the program on the GPU (BF12)."""

    def __init__(self, path, gpu=False, audio=None, vision=None, cpu_threads=None):
        """gpu: the text model on the GPU; audio, vision: "gpu", "cpu" (default
        that of the text model), or "off" (not loaded: such an input is an
        error); cpu_threads: the team of a program on the CPU (default the
        CPUs of the process)."""
        self.path = path
        self.gpu = bool(gpu)
        dev = "gpu" if self.gpu else "cpu"
        self.devices = {"text": dev, "audio": audio or dev, "vision": vision or dev}
        for k, v in self.devices.items():
            if v not in ("gpu", "cpu", "off") or (k == "text" and v == "off"):
                raise ValueError("the device of %s must be gpu, cpu or off, not %r" % (k, v))
        self.cpu_threads = int(cpu_threads or team_threads())
        cfg = json.load(open(os.path.join(path, "config.json")))
        tc = cfg["text_config"]
        self.width = tc["hidden_size"]
        self.layers = tc["num_hidden_layers"]
        self.heads = tc["num_attention_heads"]
        self.ple_width = tc["hidden_size_per_layer_input"]
        self.eps = float(tc["rms_norm_eps"])
        self.window = int(tc["sliding_window"])
        self.dim = int(tc["embedding_dim"])
        self.bos, self.eos = int(tc["bos_token_id"]), int(tc["eos_token_id"])
        self.audio_id = int(cfg["audio_token_id"])
        self.boa, self.eoa = int(cfg["boa_token_id"]), int(cfg["eoa_token_index"])
        self._audio_cfg = cfg.get("audio_config")
        self._audio = None
        self.image_id = int(cfg["image_token_id"])
        self.boi, self.eoi = int(cfg["boi_token_id"]), int(cfg["eoi_token_id"])
        self._vision_cfg = cfg.get("vision_config")
        self._vision = None
        # the most tokens of an input (the model card: a context of 8192)
        self.max_tokens = 8192
        st = json.load(open(os.path.join(path, "config_sentence_transformers.json")))
        self.prompts = dict(st.get("prompts") or {})
        self.tok = Tokenizer(os.path.join(path, "tokenizer.json"))

        W = SafeTensors(os.path.join(path, "model.safetensors"))
        self._st = W        # the table is a view of its map
        p = "language_model."
        f32 = lambda n: np.ascontiguousarray(W.get(p + n), dtype=np.float32)  # noqa: E731
        # a matrix (out, in): its raw bfloat16 rows
        bf = lambda n: _Plain(np.ascontiguousarray(W.get_bf16(p + n)))  # noqa: E731
        self.table = W.get_bf16(p + "embed_tokens.weight")
        self.embed_scale = np.float32(math.sqrt(self.width))
        # the PLE projection a layer at a time: its rows of layer i are that
        # layer's 512 (the reshape of transformers)
        ple = W.get_bf16(p + "ple.per_layer_model_projection.weight")
        pw = self.ple_width
        self.ple_lin = [_Plain(np.ascontiguousarray(ple[i * pw:(i + 1) * pw])) for i in range(self.layers)]
        self.ple_norm = f32("ple.per_layer_projection_norm.weight")
        self.norm = f32("norm.weight")
        # the projection to 768 on the host: (in, out), x @ w
        self.out_proj = np.ascontiguousarray(W.get(p + "embedding_projection.weight").T)
        theta = {k: float(v["rope_theta"]) for k, v in tc["rope_parameters"].items()}
        wide = {int(k): v for k, v in (tc.get("per_layer_config") or {}).items()}
        self.blk = []
        self._inv = {}
        for i in range(self.layers):
            q = "layers.%d." % i
            kind = tc["layer_types"][i]
            hd = int(wide.get(i, {}).get("head_dim", tc["head_dim"]))
            kv = int(wide.get(i, {}).get("num_key_value_heads", tc["num_key_value_heads"]))
            key = (hd, theta[kind])
            if key not in self._inv:
                self._inv[key] = (1.0 / (theta[kind] ** (np.arange(0, hd, 2, dtype=np.float32) / hd))
                                  ).astype(np.float32)
            self.blk.append(dict(
                sliding=kind == "sliding_attention", hd=hd, kv=kv, inv=self._inv[key],
                ln_in=f32(q + "input_layernorm.weight"),
                q=bf(q + "self_attn.q_proj.weight"), k=bf(q + "self_attn.k_proj.weight"),
                v=bf(q + "self_attn.v_proj.weight"), o=bf(q + "self_attn.o_proj.weight"),
                qn=f32(q + "self_attn.q_norm.weight"), kn=f32(q + "self_attn.k_norm.weight"),
                post_attn=f32(q + "post_attention_layernorm.weight"),
                pre_ff=f32(q + "pre_feedforward_layernorm.weight"),
                gate=bf(q + "mlp.gate_proj.weight"), up=bf(q + "mlp.up_proj.weight"),
                down=bf(q + "mlp.down_proj.weight"),
                post_ff=f32(q + "post_feedforward_layernorm.weight"),
                ple_gate=bf(q + "ple_block.per_layer_input_gate.weight"),
                ple_proj=bf(q + "ple_block.per_layer_projection.weight"),
                ple_post=f32(q + "ple_block.post_per_layer_input_norm.weight"),
                scalar=np.float32(f32(q + "layer_scalar").reshape(-1)[0])))
            self.blk[-1]["svec"] = np.full(self.width, self.blk[-1]["scalar"], np.float32)
        if self.gpu:
            # the matrices in BF12 (a matrix that the rule of to_bf12 refuses stays bfloat16)
            lins = self.ple_lin + [b[k] for b in self.blk
                                   for k in ("q", "k", "v", "o", "gate", "up", "down", "ple_gate", "ple_proj")]
            kept = sum(not to_bf12(lin) for lin in lins)
            if kept:
                print("embeddinggemma-2: %d of %d matrices stay bfloat16 (not BF12)" % (kept, len(lins)),
                      flush=True)
        self.runner = _Runner(self._team(self.program), keep=6, keep_gpu=4)

    def _team(self, build):
        """build(n, gpu) that gives a CPU program the team of cpu_threads."""
        def make(n, gpu):
            p = build(n, gpu)
            if not gpu:
                p.threads = self.cpu_threads
                p.buf[3] = self.cpu_threads     # word 3 of the header: the team
            return p
        return make

    # ---- the tokens -------------------------------------------------------------

    def tokens(self, text, prompt=None):
        """The ids of one input: <bos>, the prompt (a name of the model's
        prompts, or the text of one), the text, <eos>."""
        if prompt:
            text = self.prompts.get(prompt, prompt) + text
        return [self.bos] + self.tok.encode(text) + [self.eos]

    # ---- the model ---------------------------------------------------------------

    def _w(self, lin):
        """The float32 transpose (in, out) of a matrix, for forward_numpy."""
        t = getattr(lin, "f32t", None)
        if t is None:
            if lin.w is None:
                raise ValueError("the matrices are BF12 on the GPU; forward_numpy needs gpu=False")
            t = lin.f32t = np.ascontiguousarray(ops.bf16_to_f32(lin.w).T)
        return t

    def _rope(self, x, pos, inv):
        """x (n, heads, hd) in place: NEOX halves, the angles pos * inv."""
        ang = pos[:, None].astype(np.float32) * inv[None, :]
        c, s = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
        h = x.shape[-1] // 2
        a, b = x[..., :h].copy(), x[..., h:].copy()
        x[..., :h] = a * c - b * s
        x[..., h:] = b * c + a * s
        return x

    BLOCK = 1024        # the queries of one product of the attention (forward_numpy)

    def _attention(self, q, k, v, lo, hi):
        """The attention of query i over the keys [lo[i], hi[i]) (scale 1)
        with NumPy, in blocks of queries."""
        n, H, hd = q.shape
        kv = k.shape[1]
        rep = H // kv
        out = np.empty_like(q)
        qs = q.reshape(n, kv, rep, hd).transpose(1, 2, 0, 3)
        ks, vs = k.transpose(1, 2, 0), v.transpose(1, 0, 2)
        for i0 in range(0, n, self.BLOCK):
            i1 = min(n, i0 + self.BLOCK)
            j0, j1 = int(lo[i0:i1].min()), int(hi[i0:i1].max())
            s = np.matmul(np.ascontiguousarray(qs[:, :, i0:i1]).reshape(kv, rep * (i1 - i0), hd),
                          np.ascontiguousarray(ks[:, :, j0:j1]))
            s = s.reshape(kv, rep, i1 - i0, j1 - j0)
            j = np.arange(j0, j1)[None, :]
            s[:, :, (j < lo[i0:i1, None]) | (j >= hi[i0:i1, None])] = -np.inf
            s -= s.max(-1, keepdims=True)
            np.exp(s, out=s)
            s /= s.sum(-1, keepdims=True)
            o = np.matmul(s.reshape(kv, rep * (i1 - i0), j1 - j0), np.ascontiguousarray(vs[:, j0:j1]))
            out[i0:i1] = o.reshape(kv, rep, i1 - i0, hd).transpose(2, 0, 1, 3).reshape(i1 - i0, H, hd)
        return out

    def forward_numpy(self, e, pos, ranges):
        """The NumPy layers (the reference of the program): the rows of the
        final norm for the embeddings e (n, 512)."""
        n, H, eps = e.shape[0], self.heads, self.eps
        lin = lambda x, w: np.matmul(x, self._w(w))  # noqa: E731
        rms = lambda x, w: ops.rms_norm(x, w, eps)  # noqa: E731
        x = e.copy()
        for i, b in enumerate(self.blk):
            hd, kv = b["hd"], b["kv"]
            lo, hi = ranges["sliding" if b["sliding"] else "full"]
            pl = rms(lin(e, self.ple_lin[i]) * np.float32(self.width ** -0.5), self.ple_norm)
            h = rms(x, b["ln_in"])
            q = rms(lin(h, b["q"]).reshape(n, H, hd), b["qn"])
            k = rms(lin(h, b["k"]).reshape(n, kv, hd), b["kn"])
            v = rms(lin(h, b["v"]).reshape(n, kv, hd), None)
            self._rope(q, pos, b["inv"])
            self._rope(k, pos, b["inv"])
            a = self._attention(q, k, v, lo, hi).reshape(n, H * hd)
            x = x + rms(lin(a, b["o"]), b["post_attn"])
            h = rms(x, b["pre_ff"])
            f = lin(ops.gelu_tanh(lin(h, b["gate"])) * lin(h, b["up"]), b["down"])
            x = x + rms(f, b["post_ff"])
            g = ops.gelu_tanh(lin(x, b["ple_gate"])) * pl
            x = x + rms(lin(g, b["ple_proj"]), b["ple_post"])
            x *= b["scalar"]
        return rms(x, self.norm)

    def program(self, n, gpu=False):
        """The program of the layers for n rows (the records ENC_*): inputs
        "e" (the embeddings), "pos" (the position of each row in its input),
        "slo", "shi", "flo", "fhi" (the keys of each row in the sliding and
        the full layers); output "xn" (n, 512), the rows of the final norm."""
        w, H, pw = self.width, self.heads, self.ple_width
        inter = _shape(self.blk[0]["gate"])[0]
        hds = {b["sliding"]: (b["hd"], b["kv"]) for b in self.blk}
        kmax = max(w, inter, H * max(h for h, _ in hds.values()))
        c = _EncCompiler(self.eps, pack=not gpu, rows=n, kmax=kmax)
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        i32 = lambda: np.zeros(n, np.int32)  # noqa: E731
        c.env.update(e=z(n, w), x=z(n, w), h=z(n, w), t=z(n, w), pl=z(n, pw), g2=z(n, pw),
                     g=z(n, inter), u=z(n, inter), xn=z(n, w),
                     pos=i32(), slo=i32(), shi=i32(), flo=i32(), fhi=i32(),
                     _scratch=np.zeros(n * kmax, np.float32))
        for sl, (hd, kv) in hds.items():
            tag = "s" if sl else "f"
            c.env.update(**{"q" + tag: z(n, H * hd), "k" + tag: z(n, kv * hd),
                            "v" + tag: z(n, kv * hd), "o" + tag: z(n, H * hd)})
        scale = np.full(pw, self.width ** -0.5, np.float32)
        forms = [("set", "x", ("copy", "e"))]
        for i, b in enumerate(self.blk):
            hd, kv = b["hd"], b["kv"]
            tag = "s" if b["sliding"] else "f"
            q, k, v, o = "q" + tag, "k" + tag, "v" + tag, "o" + tag
            lo, hi = ("slo", "shi") if b["sliding"] else ("flo", "fhi")
            forms += [
                ("set", "pl", ("enc_linear", self.ple_lin[i], "e")),
                ("set", "pl", ("enc_mul_vec", "pl", scale)),
                ("set", "pl", ("enc_rms", "pl", self.ple_norm, pw)),
                ("set", "h", ("enc_rms", "x", b["ln_in"], w)),
                ("set", q, ("enc_linear", b["q"], "h")),
                ("set", q, ("enc_rms", q, b["qn"], hd)),
                ("set", k, ("enc_linear", b["k"], "h")),
                ("set", k, ("enc_rms", k, b["kn"], hd)),
                ("set", v, ("enc_linear", b["v"], "h")),
                ("set", v, ("enc_rms", v, None, hd)),
                ("enc_rope1d", q, "pos", b["inv"], H, hd),
                ("enc_rope1d", k, "pos", b["inv"], kv, hd),
                ("set", o, ("enc_attn_range", q, k, v, lo, hi, H, kv, hd)),
                ("set", "t", ("enc_linear", b["o"], o)),
                ("set", "t", ("enc_rms", "t", b["post_attn"], w)),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_rms", "x", b["pre_ff"], w)),
                ("set", "g", ("enc_linear", b["gate"], "h")),
                ("set", "u", ("enc_linear", b["up"], "h")),
                ("set", "g", ("enc_gelu_mul", "g", "u")),
                ("set", "t", ("enc_linear", b["down"], "g")),
                ("set", "t", ("enc_rms", "t", b["post_ff"], w)),
                ("enc_add", "x", "t"),
                ("set", "g2", ("enc_linear", b["ple_gate"], "x")),
                ("set", "g2", ("enc_gelu_mul", "g2", "pl")),
                ("set", "t", ("enc_linear", b["ple_proj"], "g2")),
                ("set", "t", ("enc_rms", "t", b["ple_post"], w)),
                ("enc_add", "x", "t"),
                ("set", "x", ("enc_mul_vec", "x", b["svec"])),
            ]
        forms.append(("set", "xn", ("enc_rms", "x", self.norm, w)))
        c.compile(("seq", *forms))
        return c.p.finish()

    def _inputs(self, ids, spans, soft, n):
        """The inputs of the program of n rows (n at least the ids)."""
        m = len(ids)
        ids = np.asarray(ids, dtype=np.int64)
        e = np.zeros((n, self.width), np.float32)
        e[:m] = ops.bf16_to_f32(self.table[ids]) * self.embed_scale
        where = np.flatnonzero((ids == self.audio_id) | (ids == self.image_id))
        if len(where) or soft is not None:
            soft = np.zeros((0, self.width), np.float32) if soft is None else soft
            if len(where) != len(soft):
                raise ValueError("%d media placeholders, %d media rows" % (len(where), len(soft)))
            e[where] = soft
        # the rows past the inputs: each over itself alone
        r = np.arange(n, dtype=np.int32)
        pos, slo, shi, flo, fhi = np.zeros(n, np.int32), r.copy(), r + 1, r.copy(), r + 1
        W = self.window
        for a, b in spans:
            i = r[a:b]
            pos[a:b] = i - a
            slo[a:b] = np.maximum(a, i - W)
            shi[a:b] = np.minimum(b, i + W + 1)
            flo[a:b], fhi[a:b] = a, b
        return dict(e=e, pos=pos, slo=slo, shi=shi, flo=flo, fhi=fhi)

    def forward(self, ids, spans, soft=None):
        """The rows of the final norm (sum of tokens, 512) for the packed ids
        of the inputs (spans: their (start, end)). soft: the rows (k, 512) of
        the k media placeholders of the ids, in their order."""
        m = len(ids)
        if _ENC_PY:
            d = self._inputs(ids, spans, soft, m)
            return self.forward_numpy(d["e"], d["pos"], {"sliding": (d["slo"], d["shi"]),
                                                         "full": (d["flo"], d["fhi"])})
        n = _bucket(m)
        return self.runner.run(n, self.gpu, self._inputs(ids, spans, soft, n), "xn")[:m]

    def embed_ids(self, seqs, dim=None, tokens=False, soft=None):
        """The vectors (len(seqs), dim) of lists of ids, L2-normalized.
        soft: for each sequence the rows of its media placeholders (audio,
        image), or None.
        tokens: also the token rows (sum of tokens, 768), before the mean (a
        check against the reference)."""
        dim = dim or self.dim
        if dim not in DIMS:
            raise ValueError("dimensions must be one of %s" % (DIMS,))
        spans, a = [], 0
        for s in seqs:
            spans.append((a, a + len(s)))
            a += len(s)
        rows = None
        if soft is not None and any(r is not None for r in soft):
            rows = np.concatenate([r for r in soft if r is not None])
        xn = self.forward([t for s in seqs for t in s], spans, rows)
        means = np.stack([xn[a:b].mean(0) for a, b in spans])
        y = (means @ self.out_proj)[:, :dim]
        y /= np.linalg.norm(y, axis=1, keepdims=True)
        if tokens:
            return y, xn @ self.out_proj
        return y

    def embed(self, texts, prompt=None, dim=None):
        """The vectors (len(texts), dim) of texts (each with the prompt, a
        name of the model's prompts or its text)."""
        seqs = [self.tokens(t, prompt) for t in texts]
        for s in seqs:
            if len(s) > self.max_tokens:
                raise ValueError("an input has %d tokens; the model takes at most %d"
                                 % (len(s), self.max_tokens))
        return self.embed_ids(seqs, dim)

    # ---- audio ---------------------------------------------------------------------

    @property
    def audio(self):
        """The audio encoder (Gemma4Audio), loaded at the first use."""
        if self._audio is None:
            if not self._audio_cfg:
                raise ValueError("the checkpoint has no audio tower")
            if self.devices["audio"] == "off":
                raise ValueError("the audio tower is off (devices)")
            from reclank.models.gemma4.encoders import Gemma4Audio
            A = Gemma4Audio(_AudioWeights(self._st, self._audio_cfg))
            A.runner = _Runner(self._team(A.program))
            if self.devices["audio"] == "gpu":
                # its program on the GPU, the matrices in BF12 (as the text model's)
                A.gpu = True
                lins = [A.in_proj, A.out_proj, A.proj]
                for b in A.blk:
                    lins += [b[k] for k in ("q", "k", "v", "o", "rel", "pw1", "pw2")]
                    lins += [b["ff1"][1], b["ff1"][2], b["ff2"][1], b["ff2"][2]]
                kept = sum(not to_bf12(lin) for lin in lins)
                if kept:
                    print("embeddinggemma-2 audio: %d of %d matrices stay bfloat16 (not BF12)"
                          % (kept, len(lins)), flush=True)
            self._audio = A
        return self._audio

    def audio_rows(self, samples):
        """The rows (n, 512) of 16 kHz mono samples (at most 30 s): 25 a
        second."""
        from reclank.models.gemma4.encoders import audio_features
        x = np.asarray(samples, dtype=np.float32).reshape(-1)
        if len(x) > AUDIO_MAX_S * AUDIO_RATE:
            raise ValueError("the clip has %.1f s; an audio input takes at most %d s"
                             % (len(x) / AUDIO_RATE, AUDIO_MAX_S))
        if len(x) < AUDIO_RATE // 25:
            raise ValueError("the clip is shorter than one row (40 ms)")
        mel, valid = audio_features(x)
        return self.audio.encode(mel, valid)

    def audio_input(self, samples):
        """The ids and the rows of an audio input: <bos> <|audio> the
        placeholders <audio|> <eos>."""
        rows = self.audio_rows(samples)
        return [self.bos, self.boa] + [self.audio_id] * len(rows) + [self.eoa, self.eos], rows

    # ---- images -------------------------------------------------------------------

    BUDGETS = (70, 140, 280, 560, 1120)

    @property
    def vision(self):
        """The vision encoder (Gemma4Vision), loaded at the first use."""
        if self._vision is None:
            if not self._vision_cfg:
                raise ValueError("the checkpoint has no vision tower")
            if self.devices["vision"] == "off":
                raise ValueError("the vision tower is off (devices)")
            from reclank.models.gemma4.encoders import Gemma4Vision
            V = Gemma4Vision(_VisionWeights(self._st, self._vision_cfg))
            V.runner = _Runner(self._team(V.program))
            if self.devices["vision"] == "gpu":
                V.gpu = True
                # (V.proj stays bfloat16: it runs on the host, after the pool)
                lins = [b[k] for b in V.blk for k in ("q", "k", "v", "o", "gate", "up", "down")]
                kept = sum(not to_bf12(lin) for lin in lins)
                if kept:
                    print("embeddinggemma-2 vision: %d of %d matrices stay bfloat16 (not BF12)"
                          % (kept, len(lins)), flush=True)
            self._vision = V
        return self._vision

    def image_rows(self, src, budget=280):
        """The rows (soft tokens, 512) of an image (a path, the bytes of a
        file, or a PIL image) at a budget of soft tokens."""
        from reclank.models.gemma4.encoders import image_patches16
        from reclank.models.gemma4.unified import image_pixels
        if budget not in self.BUDGETS:
            raise ValueError("the image budget must be one of %s" % (self.BUDGETS,))
        V = self.vision
        patches, pos = image_patches16(image_pixels(src, budget))
        return V.encode(patches, pos)

    def image_input(self, src, budget=280):
        """The ids and the rows of an image input: <bos> <|image> the
        placeholders <image|> <eos>."""
        rows = self.image_rows(src, budget)
        return [self.bos, self.boi] + [self.image_id] * len(rows) + [self.eoi, self.eos], rows
