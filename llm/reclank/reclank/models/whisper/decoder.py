"""Whisper's text decoder (openai/whisper-large-v3 as transformers'
WhisperForConditionalGeneration), with the encoder of encoder.py: the
transcription (or the translation to English) of windows of 30 s.

Ported from transformers (src/transformers/models/whisper/modeling_whisper.py,
Copyright 2022 The OpenAI Authors and The HuggingFace Inc. team;
models/whisper/generation_whisper.py, Copyright 2024 The HuggingFace Inc. team;
models/whisper/tokenization_whisper.py, Copyright 2022 The HuggingFace Inc. team;
generation/logits_process.py, Copyright 2024 The HuggingFace Inc. team and Google
DeepMind; Apache License 2.0). See THIRD_PARTY_NOTICES.md.

    the token's row of embed_tokens + the position's row of embed_positions
        (448 positions)
    32 pre-LN layers of width 1280 (20 heads of 64, FFN 5120 with GELU erf):
        LN, the causal attention over the window's tokens (q with bias and
        scale 1/8, k without a bias, v and the output with bias), + x;
        LN, the attention over the encoder's 1500 rows of the window (the
        same projections; their keys and values made once a window), + x;
        LN, fc1, GELU erf, fc2, + x. Then the decoder's LN, and the logits:
        the product with embed_tokens (tied)

The programs share their caches through one _Runner (one mirror of the
arrays on the GPU): the encoder's program of n windows ends with the keys and
values of each decoder layer over its hidden states ("ck<l>", "cv<l>": n x
1536 rows, a window's in its first 1500); a step program takes r tokens of each
of n windows, puts their keys and values in the windows' rows of the cache of
its own tokens ("sk<l>", "sv<l>": n x 448 rows, ENC_SCATTER), and attends over
those (ENC_ATTN_KV: its token's window up to its position) and over the
window's encoder rows. A GPU program takes the checkpoint's float16 matrices
as they are (as encoder.py; the products of a step's few rows:
k_enc_gemv_f16, float32 sums; its layer norms a block a row,
c.lnorm_rows), a CPU program float32 (exact). The keys and values of the
caches are float32.

Generation as transformers' generate with return_timestamps False and one
beam: the prompt <|startoftranscript|> <|lang|> <|task|> <|notimestamps|>,
then the most likely token with the generation config's suppress_tokens at
minus infinity (and its begin_suppress_tokens at the first), up to
<|endoftext|> or 448 tokens.

segments() is transformers' WhisperForConditionalGeneration.generate of one
audio of any length (4.48: greedy, no temperature fallback, no thresholds,
condition_on_prev_tokens False), with return_timestamps (always past 30 s):
    the features of the whole audio (log_mel_all; at most 30 s: log_mel, padded)
    the language, when none is given: detect_language's most likely language
        token after <|startoftranscript|> over the first 3000 frames
    a window of the features at seek (3000 frames, zeros past the end), the
        prompt <|startofprev|> + the prompt's tokens (each window, as
        transformers with condition_on_prev_tokens False) + <|startoftranscript|>
        <|lang|> <|task|>, the tokens with WhisperTimeStampLogitsProcessor's
        rules (_timestamp_rules)
    its segments at each pair of timestamp tokens; seek moves to the end
        timestamp of its last whole segment (a window that ends in a single
        timestamp: its 3000 frames), as _retrieve_segment
"""
from __future__ import annotations

import json
import os
import zlib

import numpy as np

from reclank.encoders.runtime import RESIDENCY, _EncCompiler, _Runner
from reclank.formats.safetensors import SafeTensors
from reclank.models.music.text import _lin
from reclank.models.whisper.encoder import FRAMES, RATE, ROWS, SAMPLES, WIN_ROWS, WhisperEncoder, log_mel, log_mel_all
from reclank.runtime.threads import team_threads

TIME_PRECISION = 0.02           # a timestamp token's step (s): 2 frames of the features
LANGUAGES = dict(e.split(" ", 1) for e in (
    "en english,zh chinese,de german,es spanish,ru russian,ko korean,fr french,ja japanese,pt portuguese,"
    "tr turkish,pl polish,ca catalan,nl dutch,ar arabic,sv swedish,it italian,id indonesian,hi hindi,"
    "fi finnish,vi vietnamese,he hebrew,uk ukrainian,el greek,ms malay,cs czech,ro romanian,da danish,"
    "hu hungarian,ta tamil,no norwegian,th thai,ur urdu,hr croatian,bg bulgarian,lt lithuanian,la latin,"
    "mi maori,ml malayalam,cy welsh,sk slovak,te telugu,fa persian,lv latvian,bn bengali,sr serbian,"
    "az azerbaijani,sl slovenian,kn kannada,et estonian,mk macedonian,br breton,eu basque,is icelandic,"
    "hy armenian,ne nepali,mn mongolian,bs bosnian,kk kazakh,sq albanian,sw swahili,gl galician,mr marathi,"
    "pa punjabi,si sinhala,km khmer,sn shona,yo yoruba,so somali,af afrikaans,oc occitan,ka georgian,"
    "be belarusian,tg tajik,sd sindhi,gu gujarati,am amharic,yi yiddish,lo lao,uz uzbek,fo faroese,"
    "ht haitian creole,ps pashto,tk turkmen,nn nynorsk,mt maltese,sa sanskrit,lb luxembourgish,my myanmar,"
    "bo tibetan,tl tagalog,mg malagasy,as assamese,tt tatar,haw hawaiian,ln lingala,ha hausa,ba bashkir,"
    "jw javanese,su sundanese,yue cantonese").split(","))         # the names of the language tokens


def split_gpt2(text):
    """The pieces of text by GPT-2's pre-tokenizer (tokenizer.json's ByteLevel
    with use_regex), with unicodedata as qwen.tokenizer.split_words:

        's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
    """
    from reclank.models.qwen.tokenizer import _cat
    out, pos, n = [], 0, len(text)
    while pos < n:
        end = None
        if text[pos] == "'":
            for c in ("s", "t", "re", "ve", "m", "ll", "d"):
                if text.startswith(c, pos + 1):
                    end = pos + 1 + len(c)
                    break
        if end is None:
            j = pos + 1 if text[pos] == " " else pos
            k = _cat(text[j]) if j < n else None
            if k in ("L", "N"):
                end = j
                while end < n and _cat(text[end]) == k:
                    end += 1
            elif k is not None and k != "S":
                end = j
                while end < n and _cat(text[end]) not in ("S", "L", "N"):
                    end += 1
        if end is None:                 # white space: leave the last before a non-space to the next piece
            end = pos
            while end < n and _cat(text[end]) == "S":
                end += 1
            if end < n and end - 1 > pos:
                end -= 1
        out.append(text[pos:end])
        pos = end
    return out


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


def _lse(x):
    """log(sum(exp(x))) of a row of logits, in float64 (-inf: all -inf)."""
    m = float(np.max(x))
    if m == -np.inf:
        return m
    return m + float(np.log(np.exp(x.astype(np.float64) - m).sum()))


def _compression(text):
    """The zlib ratio of a text's UTF-8 (openai-whisper's compression_ratio)."""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


class Tokens:
    """The tokens of a checkpoint (vocab.json, added_tokens.json): the ids of
    the special tokens, and text() of ids."""

    def __init__(self, path):
        self.path = path
        self._bpe = None
        vocab = json.load(open(os.path.join(path, "vocab.json"), encoding="utf-8"))
        added = json.load(open(os.path.join(path, "added_tokens.json"), encoding="utf-8"))
        self.ids = dict(vocab)
        self.ids.update(added)
        self.special = set(added.values())
        self.strs = {i: s for s, i in self.ids.items()}
        self.bytes = _byte_decoder()

    def id(self, s):
        return self.ids[s]

    def text(self, ids, special=False):
        """The text of ids (the special tokens dropped, unless special)."""
        out, b = [], bytearray()
        for i in ids:
            i = int(i)
            if i in self.special:
                if special:
                    out.append(b.decode("utf-8", "replace"))
                    b = bytearray()
                    out.append(self.strs.get(i, ""))
                continue
            b += bytes(self.bytes[ch] for ch in self.strs[i])
        out.append(b.decode("utf-8", "replace"))
        return "".join(out)

    def encode(self, text):
        """The ids of text: tokenizer.json's byte-level BPE (no normalizer)
        with GPT-2's pre-tokenizer (split_gpt2); a special token's text is
        its id."""
        if self._bpe is None:
            from reclank.models.qwen.tokenizer import QwenTokenizer
            self._bpe = QwenTokenizer(os.path.join(self.path, "tokenizer.json"))
            self._bpe._split = split_gpt2
        return self._bpe.encode(text)


class Whisper:
    """A Whisper checkpoint directory (transformers' layout): the encoder
    (encoder.WhisperEncoder) and the decoder, on the CPU or the GPU, n
    windows at once (batch). transcribe() gives the tokens and the text of
    windows of samples or features."""

    def __init__(self, path, gpu=False, cpu_threads=None, batch=4):
        self.path = path
        self.W = SafeTensors(os.path.join(path, "model.safetensors"))
        self.gpu, self.cpu_threads, self.batch = bool(gpu), cpu_threads, batch
        cfg = json.load(open(os.path.join(path, "config.json")))
        gen = json.load(open(os.path.join(path, "generation_config.json")))
        assert not cfg.get("scale_embedding") and cfg["activation_function"] == "gelu"
        self.width, self.heads = cfg["d_model"], cfg["decoder_attention_heads"]
        self.max_tokens = cfg["max_target_positions"]                    # 448
        self.vocab = cfg["vocab_size"]
        hd = self.width // self.heads
        self.enc = WhisperEncoder(path, gpu=gpu, cpu_threads=cpu_threads, W=self.W, batch=batch)
        self.tok = Tokens(path)
        self.gen = gen
        self.eot = gen["eos_token_id"]
        self.suppress = np.array(gen.get("suppress_tokens", []), np.int64)
        self.begin_suppress = np.array(gen.get("begin_suppress_tokens", []), np.int64)
        self.sot, self.prev_sot = gen["decoder_start_token_id"], gen["prev_sot_token_id"]
        self.no_ts = gen["no_timestamps_token_id"]
        self.ts_begin = self.no_ts + 1                                  # <|0.00|>
        self.no_speech = self.no_ts - 1                                 # <|nospeech|>
        self.max_init_ts = gen.get("max_initial_timestamp_index")
        self.lang_ids = {k[2:-2]: v for k, v in gen["lang_to_id"].items()}   # "sk": <|sk|>
        p = "model.decoder."
        f = lambda n: np.asarray(self.W.get(p + n), dtype=np.float32)  # noqa: E731
        self.eps = 1e-5
        self.embed = f("embed_tokens.weight")                           # (vocab, 1280): the rows of the tokens
        self.pos = f("embed_positions.weight")                          # (448, 1280)
        self.proj = _lin(self.W, p + "embed_tokens", bias=False)        # the logits (tied)
        self.layers = []
        for i in range(cfg["decoder_layers"]):
            r = p + "layers.%d." % i
            g = lambda n: f("layers.%d.%s" % (i, n))  # noqa: E731
            self.layers.append(dict(
                ln_a=(g("self_attn_layer_norm.weight"), g("self_attn_layer_norm.bias")),
                aq=_lin(self.W, r + "self_attn.q_proj", scale=hd ** -0.5),
                ak=_lin(self.W, r + "self_attn.k_proj", bias=False),
                av=_lin(self.W, r + "self_attn.v_proj"), ao=_lin(self.W, r + "self_attn.out_proj"),
                ln_c=(g("encoder_attn_layer_norm.weight"), g("encoder_attn_layer_norm.bias")),
                cq=_lin(self.W, r + "encoder_attn.q_proj", scale=hd ** -0.5),
                ck=_lin(self.W, r + "encoder_attn.k_proj", bias=False),
                cv=_lin(self.W, r + "encoder_attn.v_proj"), co=_lin(self.W, r + "encoder_attn.out_proj"),
                ln_f=(g("final_layer_norm.weight"), g("final_layer_norm.bias")),
                fc1=_lin(self.W, r + "fc1"), fc2=_lin(self.W, r + "fc2")))
        self.ln_end = (f("layer_norm.weight"), f("layer_norm.bias"))
        if self.gpu and os.environ.get("RECLANK_WHISPER_F32", "0") != "1":
            for lin in [self.proj] + [L[k] for L in self.layers
                                      for k in ("aq", "ak", "av", "ao", "cq", "ck", "cv", "co", "fc1", "fc2")]:
                lin.w = lin.w.astype(np.float16)
        self.caches = {}
        # the programs of the encoder and of the steps, with one mirror (the caches they share)
        self.runner = _Runner(self._build, keep=8, keep_gpu=4)

    def runners(self):
        """The GPU runners (name: _Runner): the encoder's and the steps' programs (one mirror: the
        caches), and the encoder's own (WhisperEncoder.encode)."""
        return {"model": self.runner, "encoder": self.enc.runner} if self.gpu else {}

    def _span(self):
        """The scope of a run whose caches live on the GPU from the encoder's program to the last
        step: the runner stays pinned (RESIDENCY)."""
        return RESIDENCY.use(self.runner if self.gpu else None)

    # ---- the programs ----------------------------------------------------

    def cache(self, n):
        """The caches of n windows: the decoder's keys and values over the
        encoder's rows ("ck<l>", "cv<l>") and over its own tokens ("sk<l>",
        "sv<l>")."""
        c = self.caches.get(n)
        if c is None:
            w = self.width
            c = {}
            for i in range(len(self.layers)):
                for name, rows in (("ck", WIN_ROWS), ("cv", WIN_ROWS), ("sk", self.max_tokens), ("sv", self.max_tokens)):
                    c["%s%d" % (name, i)] = np.zeros((n * rows, w), np.float32)
            self.caches[n] = c
        return c

    def _team(self, p, gpu):
        if not gpu:
            p.threads = p.buf[3] = self.cpu_threads or team_threads()
        return p

    def _build(self, key, gpu):
        kind, n = key[0], key[1]
        if kind == "enc":
            return self._team(self.enc.program(n, gpu, tail=self._cross), gpu)
        return self._team(self.step_program(n, key[2], gpu), gpu)

    def _cross(self, c, n):
        """The forms after the encoder's program of n windows: each decoder
        layer's keys and values over "hid", into the caches."""
        cache = self.cache(n)
        c.env.update(cache)
        forms = []
        for i, L in enumerate(self.layers):
            forms += [("set", "ck%d" % i, ("enc_linear", L["ck"], "hid")),
                      ("set", "cv%d" % i, ("enc_linear", L["cv"], "hid"))]
        return forms

    def step_program(self, n, r=1, gpu=False):
        """The program of a step of r tokens of each of n windows (rows: the
        window's tokens one after another, then the next window's). Inputs:
        "tok" and "pos" (n r, 1) int32: a row's token and position; "kvi" (n
        r) int32: its row in the caches of the tokens (window 448 + pos);
        "slo", "shi": its keys there (window 448 to its own row, included);
        "clo", "chi": its window's encoder rows. Output "logits" (n r, vocab)."""
        w, H = self.width, self.heads
        hd = w // H
        R = n * r
        inter = self.layers[0]["fc1"].w.shape[0]
        c = _EncCompiler(self.eps, pack=not gpu, rows=R, kmax=inter)
        c.lnorm_rows = True             # few rows: a block a row on the GPU
        z = lambda *shape: np.zeros(shape, np.float32)  # noqa: E731
        zi = lambda *shape: np.zeros(shape, np.int32)  # noqa: E731
        win = np.repeat(np.arange(n), r)
        c.env.update(self.cache(n))
        c.env.update(tok=zi(R, 1), pos=zi(R, 1), kvi=zi(R), slo=zi(R), shi=zi(R),
                     clo=(win * WIN_ROWS).astype(np.int32), chi=(win * WIN_ROWS + ROWS).astype(np.int32),
                     x=z(R, w), h=z(R, w), t=z(R, w), u=z(R, inter), q=z(R, w), k=z(R, w), v=z(R, w),
                     o=z(R, w), logits=z(R, self.vocab))
        forms = [
            ("set", "x", ("enc_gather", self.embed, "tok", w)),
            ("set", "t", ("enc_gather", self.pos, "pos", w)),
            ("enc_add", "x", "t"),
        ]
        for i, L in enumerate(self.layers):
            sk, sv, ck, cv = "sk%d" % i, "sv%d" % i, "ck%d" % i, "cv%d" % i
            forms += [
                ("set", "h", ("enc_lnorm", "x", L["ln_a"][0], L["ln_a"][1], w)),
                ("set", "q", ("enc_linear", L["aq"], "h")),
                ("set", "k", ("enc_linear", L["ak"], "h")),
                ("set", "v", ("enc_linear", L["av"], "h")),
                ("set", sk, ("enc_scatter", "k", "kvi")),
                ("set", sv, ("enc_scatter", "v", "kvi")),
                ("set", "o", ("enc_attn_kv", "q", sk, sv, "slo", "shi", H, hd, self.max_tokens)),
                ("set", "t", ("enc_linear", L["ao"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", L["ln_c"][0], L["ln_c"][1], w)),
                ("set", "q", ("enc_linear", L["cq"], "h")),
                ("set", "o", ("enc_attn_kv", "q", ck, cv, "clo", "chi", H, hd, ROWS)),
                ("set", "t", ("enc_linear", L["co"], "o")),
                ("enc_add", "x", "t"),
                ("set", "h", ("enc_lnorm", "x", L["ln_f"][0], L["ln_f"][1], w)),
                ("set", "u", ("enc_linear", L["fc1"], "h")),
                ("set", "u", ("enc_gelu", "u", True)),
                ("set", "t", ("enc_linear", L["fc2"], "u")),
                ("enc_add", "x", "t"),
            ]
        forms += [
            ("set", "h", ("enc_lnorm", "x", self.ln_end[0], self.ln_end[1], w)),
            ("set", "logits", ("enc_linear", self.proj, "h")),
        ]
        c.compile(("seq", *forms))
        return c.p.finish()

    def _run(self, key, inputs, outputs):
        """Write the inputs (name: array), run the program of key, and return
        copies of the outputs."""
        prog, g = self.runner.get(key, self.gpu)
        for name, a in inputs.items():
            prog.names[name][...] = a
            if g is not None:
                g.upload(name)
        if g is not None:
            g.run()
            for name in outputs:
                g.download(name)
        else:
            prog.run()
        return [prog.names[name].copy() for name in outputs]

    # ---- the steps -------------------------------------------------------

    def encode(self, features):
        """Run the encoder's program of len(features) windows (features (128,
        3000) each, log_mel): their keys and values of the decoder."""
        m = np.concatenate([np.ascontiguousarray(f.T) for f in features])
        self._run(("enc", len(features)), {"m": m}, [])

    def step(self, n, tokens, positions):
        """The logits (n r, vocab) of r tokens of each of n windows (tokens,
        positions (n, r)), after the encoder's program of the n windows and
        the steps of each window's tokens before them."""
        tokens, positions = np.asarray(tokens, np.int32), np.asarray(positions, np.int32)
        r = tokens.shape[1]
        win = np.repeat(np.arange(n), r).astype(np.int32)
        kvi = win * self.max_tokens + positions.reshape(-1)
        assert positions.max() < self.max_tokens
        inputs = dict(tok=tokens.reshape(-1, 1), pos=positions.reshape(-1, 1), kvi=kvi,
                      slo=win * self.max_tokens, shi=kvi + 1)
        return self._run(("step", n, r), inputs, ["logits"])[0]

    def prompt(self, lang="sk", task="transcribe", timestamps=False):
        """<|startoftranscript|> <|lang|> <|task|>, and <|notimestamps|> unless
        timestamps."""
        t = self.tok
        p = [t.id("<|startoftranscript|>"), t.id("<|%s|>" % lang), t.id("<|%s|>" % task)]
        return p if timestamps else p + [t.id("<|notimestamps|>")]

    def generate(self, features, lang="sk", task="transcribe", max_tokens=None):
        """The greedy tokens of each window (prompt, the tokens, <|endoftext|>
        if it came), in programs of self.batch windows."""
        out = []
        for i in range(0, len(features), self.batch):
            with self._span():
                out += self._generate(features[i:i + self.batch], self.prompt(lang, task), max_tokens)[0]
        return out

    def _generate(self, features, prompt, max_tokens, timestamps=False, encoded=False, stats=False):
        """The greedy tokens of n windows after prompt (the same for each),
        with timestamps the rules of _timestamp_rules; encoded: the encoder's
        program of these windows ran already. stats: also each window's sum
        of the log-probabilities of its tokens (of the processed logits, to
        <|endoftext|> included) and the probability of <|nospeech|> at
        <|startoftranscript|>."""
        n = len(features)
        limit = min(max_tokens or self.max_tokens, self.max_tokens)
        if not encoded:
            self.encode(features)
        seqs = [list(prompt) for _ in range(n)]
        done = np.zeros(n, bool)
        info = [dict(sum_logprob=0.0, tokens=0, no_speech_prob=None) for _ in range(n)]
        for p, t in enumerate(prompt[:-1]):
            logits = self.step(n, np.full((n, 1), t), np.full((n, 1), p))
            if stats and t == self.sot:
                for j in range(n):
                    info[j]["no_speech_prob"] = float(np.exp(logits[j, self.no_speech] - _lse(logits[j])))
        cur = np.full(n, prompt[-1])
        p = len(prompt) - 1
        while p < limit - 1 and not done.all():
            logits = self.step(n, cur[:, None], np.full((n, 1), p))
            logits[:, self.suppress] = -np.inf
            if p == len(prompt) - 1:
                logits[:, self.begin_suppress] = -np.inf
            if timestamps:
                self._timestamp_rules(logits, seqs, len(prompt))
            nxt = logits.argmax(-1)
            for j in range(n):
                if not done[j]:
                    seqs[j].append(int(nxt[j]))
                    done[j] = nxt[j] == self.eot
                    if stats:
                        info[j]["sum_logprob"] += float(logits[j, nxt[j]] - _lse(logits[j]))
                        info[j]["tokens"] += 1
            cur = np.where(done, self.eot, nxt)
            p += 1
        return seqs, info

    def _timestamp_rules(self, logits, seqs, begin):
        """transformers' WhisperTimeStampLogitsProcessor on the logits (n,
        vocab) of the next token of each of seqs (its tokens from begin: the
        generated ones), in place: no <|notimestamps|>; timestamps in pairs
        (after a pair a text token, after a single one a timestamp or
        <|endoftext|>); no timestamp before the last (nor the last again,
        but as the end of its pair); the first token a timestamp of at most
        max_initial_timestamp_index; and a timestamp when the probability of
        all of them is above each text token's."""
        tb = self.ts_begin
        logits[:, self.no_ts] = -np.inf
        for k, s in enumerate(seqs):
            got = s[begin:]
            last = len(got) >= 1 and got[-1] >= tb
            penult = len(got) < 2 or got[-2] >= tb
            if last:
                if penult:
                    logits[k, tb:] = -np.inf
                else:
                    logits[k, :self.eot] = -np.inf
            ts = [t for t in got if t >= tb]
            if ts:
                logits[k, tb:ts[-1] if last and not penult else ts[-1] + 1] = -np.inf
            if not got:
                logits[k, :tb] = -np.inf
                if self.max_init_ts is not None:
                    logits[k, tb + self.max_init_ts + 1:] = -np.inf
            if _lse(logits[k, tb:]) > logits[k, :tb].max():
                logits[k, :tb] = -np.inf

    def detect_language(self, features):
        """transformers' detect_language of a window's features (128, 3000):
        the code ("sk") of the most likely language token after
        <|startoftranscript|>. Runs the encoder's program of the window."""
        with self._span():
            self.encode([features])
            logits = self.step(1, [[self.sot]], [[0]])[0]
        codes = sorted(self.lang_ids, key=self.lang_ids.get)
        ids = np.array([self.lang_ids[c] for c in codes])
        return codes[int(logits[ids].argmax())]

    def segments(self, samples, lang=None, task="transcribe", prompt=None, timestamps=True):
        """transformers' generate of one audio (16 kHz samples, any length; the
        steps of the module's header). lang: a code ("sk"; None: detected);
        prompt: the ids of a previous text (the last 223 kept). timestamps
        False (at most 30 s only): one window without timestamps, as
        generate(). Gives {"language", "duration", "text", "tokens",
        "segments": [{"seek" (frames), "start", "end" (s), "tokens", "text",
        "avg_logprob", "no_speech_prob", "compression_ratio" (the window's:
        its sum of log-probabilities over its tokens, the probability of
        <|nospeech|> at <|startoftranscript|>, the zlib ratio of its text's
        UTF-8)}]}."""
        x = np.asarray(samples, np.float32).reshape(-1)
        F = log_mel(x) if len(x) <= SAMPLES else log_mel_all(x)
        T = F.shape[1]
        timestamps = timestamps or T > FRAMES
        with self._span():      # (the language's window: its encoder program serves the first window)
            encoded = False
            if lang is None:
                lang = self.detect_language(F[:, :FRAMES])
                encoded = True
            elif lang not in self.lang_ids:
                raise ValueError("unknown language %r" % lang)
            if task not in ("transcribe", "translate"):
                raise ValueError("task must be transcribe or translate")
            dec = ([self.prev_sot] + list(prompt)[-(self.max_tokens // 2 - 1):] if prompt else []) + \
                self.prompt(lang, task, timestamps)
            segs, seek = [], 0
            while seek < T:
                nf = min(T - seek, FRAMES)
                f = F[:, seek:seek + nf]
                if nf < FRAMES:
                    f = np.pad(f, ((0, 0), (0, FRAMES - nf)))
                seqs, info = self._generate([f], dec, None, timestamps=timestamps, encoded=encoded and seek == 0,
                                            stats=True)
                new, info = seqs[0][len(dec):], info[0]
                if new and new[-1] == self.eot:
                    new = new[:-1]
                text = self.tok.text(new)
                w = dict(seek=seek, avg_logprob=info["sum_logprob"] / max(info["tokens"], 1),
                         no_speech_prob=info["no_speech_prob"], compression_ratio=_compression(text))
                parts, offset = self._split(new, seek * TIME_PRECISION / 2, nf)
                for start, end, toks in parts:
                    segs.append(dict(w, start=start, end=end, tokens=toks, text=self.tok.text(toks)))
                # (transformers would loop on an offset of 0, which its rules do not let happen)
                seek += offset if offset > 0 else nf
        tokens = [t for s in segs for t in s["tokens"]]
        return dict(language=lang, duration=len(x) / RATE, text=self.tok.text(tokens), tokens=tokens, segments=segs)

    def _split(self, new, offset, nf):
        """transformers' _retrieve_segment of a window's tokens (no prompt, no
        <|endoftext|>) at time offset (s), nf frames of audio: [(start, end,
        tokens)] and the frames to seek."""
        tb = self.ts_begin
        is_ts = [t >= tb for t in new]
        single = is_ts[-2:] == [False, True]
        cuts = [i + 1 for i in range(len(new) - 1) if is_ts[i] and is_ts[i + 1]]
        if not cuts:
            # one segment: to the last timestamp, else to the window's end
            ts = [t for t in new if t >= tb]
            last = int(np.float32(nf) * np.float32(0.01) / np.float32(TIME_PRECISION))   # (torch's float32)
            if ts and ts[-1] != tb:
                last = ts[-1] - tb
            return [(offset, offset + last * TIME_PRECISION, new)], nf
        if single:
            cuts.append(len(new))
        else:
            cuts[-1] += 1           # the last pair in the last segment
        out, a = [], 0
        for i, b in enumerate(cuts):
            toks = new[a:b]
            end = toks[-2] if i == len(cuts) - 1 and not single else toks[-1]
            out.append((offset + (toks[0] - tb) * TIME_PRECISION, offset + (end - tb) * TIME_PRECISION, toks))
            a = b
        return out, nf if single else (new[a - 2] - tb) * 2

    def transcribe(self, windows, lang="sk", task="transcribe", max_tokens=None):
        """[(tokens, text)] of windows of 16 kHz samples (at most 30 s each)."""
        seqs = self.generate([log_mel(s) for s in windows], lang, task, max_tokens)
        return [(s, self.tok.text(s)) for s in seqs]
