"""ACE-Step 1.5's 5 Hz planner LM (acestep-5Hz-lm-1.7B: Qwen3 1.7B, its
embedding tied, 65535 audio code tokens after the 151669 of Qwen3), the
authors' "thinking" (llm_inference.py generate_with_stop_condition, the "pt"
backend; constrained_logits_processor.py).

Ported from ACE-Step 1.5 (acestep/llm_inference.py,
acestep/constrained_logits_processor.py; MIT License, Copyright (c) 2026
ACEStep; its model files Apache License 2.0, Copyright 2025 The ACESTEO Team).
See THIRD_PARTY_NOTICES.md.

    phase 1, the metadata: the prompt (a system turn of the instruction, a
        user turn of the caption and lyrics), no CFG; the constrained decoding
        writes "<think>\\n", then bpm, caption, duration, keyscale, language
        and timesignature as YAML lines (the field names forced; the values
        from prefix trees of their valid values; the caption free text, which
        ends when the model writes the next field's name), then <|im_end|>
    phase 2, the codes: the prompt again with the metadata as YAML in
        <think>...</think> (yaml.dump, sorted keys), and an unconditional
        prompt ("NO USER INPUT" and an empty think) as the second row: CFG
        2.0 over the code tokens (0 to 63999) and <|im_end|>, exactly 5 a
        second of the duration, then <|im_end|>
    sampling: the constraint's whitelist, top-p 0.9 of the logits, then
        temperature 0.85 (the "pt" backend's order); NumPy's generator

The LM's steps are qwen3.decode step programs (the two rows of CFG two
sequences). The codes go to the DiT (dit.py: detokenize) in place of the
silence latent, with the metadata's caption, bpm, keyscale, timesignature and
duration.
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.encoders.runtime import RESIDENCY, gpu_runners
from reclank.formats.safetensors import SafeTensors
from reclank.models.qwen.tokenizer import QwenTokenizer
from reclank.models.qwen3.decode import Qwen3Decoder
from reclank.models.qwen3.dense import lin16

INSTRUCTION = "Generate audio semantic tokens based on the given conditions:"
NEGATIVE = "NO USER INPUT"
FIELDS = ("bpm", "caption", "duration", "keyscale", "language", "timesignature")
LANGUAGES = ["ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hr",
             "ht", "hu", "id", "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no", "pa", "pl", "pt", "ro",
             "ru", "sa", "sk", "sr", "sv", "sw", "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
             "unknown"]
KEYS = ["%s%s %s" % (n, a, m) for n in "ABCDEFG" for a in ("", "#", "b", "♯", "♭")
        for m in ("major", "minor")]
CODE0, CODES, MAX_CODE = 151669, 65535, 63999
NL = 198
MAX_DURATION = 600          # the >= 24 GB tier of the authors' gpu_config (480 below it)


def chat(system, user, assistant=""):
    return ("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n%s"
            % (system, user, assistant))


class Constraint:
    """The authors' MetadataConstrainedLogitsProcessor for one generation:
    allowed(scores) gives the whitelist of the next token (None: any),
    update(token) advances."""

    def __init__(self, planner, phase, user=None, target_codes=None):
        self.p, self.phase = planner, phase
        self.user = dict(user or {})
        self.target = target_codes
        self.codes = 0
        self.done = False
        if phase == "codes":
            self.state = ("codes",)
            return
        self.queue = list(planner.fixed["<think>"]) + [NL]
        self.state = ("fixed", "bpm")

    # the order of the fields: (name state, value state) each
    def _next_field(self, field):
        i = FIELDS.index(field)
        return FIELDS[i + 1] if i + 1 < len(FIELDS) else None

    def _enter_name(self, field):
        if field is None:                           # phase 1 ends with <|im_end|> in place of </think>
            self.queue, self.state = [self.p.eos], ("end",)
            return
        self.queue, self.state = list(self.p.fixed[field + ":"]), ("name", field)

    def _enter_value(self, field):
        self.prefix = ()
        if field in self.user:
            self.queue = self.p.tok.encode(" %s\n" % self.user[field])
            self.state = ("forced", field)
        elif field == "caption":
            self.state, self.cap_n, self.after_nl, self.ending, self.pending = ("caption",), 0, False, False, ""
        else:
            self.state = ("value", field)

    def allowed(self, scores):
        """The whitelist of the next token (an array of ids), or None."""
        p, kind = self.p, self.state[0]
        if kind == "codes":
            if self.target is not None and self.codes >= self.target:
                return np.array([p.eos])
            ids = p.code_ids
            return ids if self.target is not None else np.append(ids, p.eos)
        if kind in ("fixed", "name", "forced", "end"):
            return np.array(self.queue[:1])
        if kind == "caption":
            if self.ending:
                return None
            if self.after_nl:
                top = int(np.argmax(scores))
                if not p.text(top).startswith((" ", "\t")):
                    self.ending = True
                    return None
            if self.cap_n >= 512:
                return np.array([NL])
            return p.caption_ok
        field = self.state[1]
        tree = p.trees[field]
        nxt = tree.get(self.prefix, set())
        if field in ("keyscale", "language", "timesignature"):
            if NL in nxt:
                return np.array([NL])
            if field == "language" and not self.prefix:
                cand = np.array(sorted(nxt))
                return cand[np.argmax(scores[cand])][None]
        return np.array(sorted(nxt)) if nxt else np.array([NL])

    def update(self, t):
        p, kind = self.p, self.state[0]
        if kind == "codes":
            if CODE0 <= t < CODE0 + CODES:
                self.codes += 1
            return
        if kind in ("fixed", "name", "forced", "end"):
            self.queue.pop(0)
            if self.queue:
                return
            if kind == "end":
                self.done = True
            elif kind == "fixed":
                self._enter_name("bpm")
            elif kind == "name":
                self._enter_value(self.state[1])
            else:
                self._enter_name(self._next_field(self.state[1]))
            return
        if kind == "caption":
            if self.ending:
                self.pending += p.text(t)
                if ":" in self.pending:
                    name = self.pending.split(":")[0].strip().lower()
                    if name in FIELDS[2:]:
                        self._enter_value(name)
                    else:
                        self._enter_name("duration")
                return
            self.cap_n += 1
            self.after_nl = "\n" in p.text(t)
            return
        field = self.state[1]
        if t == NL:
            self._enter_name(self._next_field(field))
        else:
            self.prefix = self.prefix + (t,)


class Planner:
    """The planner LM of an Ace-Step1.5 directory (acestep-5Hz-lm-1.7B)."""

    def __init__(self, path, gpu=False, cpu_threads=None, max_tokens=4096):
        d = os.path.join(path, "acestep-5Hz-lm-1.7B")
        cfg = json.load(open(os.path.join(d, "config.json")))
        self.W = W = SafeTensors(os.path.join(d, "model.safetensors"))
        self.lm = Qwen3Decoder(W, cfg, "layers.", "norm.weight", {"logits": lin16(W, "embed_tokens", gpu, bias=False)},
                               gpu=gpu, cpu_threads=cpu_threads, max_tokens=max_tokens)
        self.tok = QwenTokenizer(os.path.join(d, "tokenizer.json"))
        self.vocab = cfg["vocab_size"]
        self.eos = self.tok.special["<|im_end|>"]
        self.pad = self.tok.special["<|endoftext|>"]
        self.code_ids = np.arange(CODE0, CODE0 + MAX_CODE + 1)
        self._text = {}
        # the forced strings: the longest prefix that is one token, again and again
        self.fixed = {s: self._fixed(s) for s in ["<think>"] + [f + ":" for f in FIELDS]}
        ok = np.ones(self.vocab, bool)
        ok[self.tok.encode("`")] = False
        ok[CODE0:CODE0 + MAX_CODE + 1] = False
        self.caption_ok = np.nonzero(ok)[0]
        values = dict(bpm=[str(v) for v in range(30, 301)], duration=[str(v) for v in range(10, MAX_DURATION + 1)],
                      keyscale=KEYS, language=LANGUAGES, timesignature=["2", "3", "4", "6"])
        self.trees = {}
        for field, vals in values.items():
            head = self.tok.encode(field + ":")
            tree = {}
            for v in vals:
                ids = self.tok.encode("%s: %s" % (field, v))
                assert ids[:len(head)] == head, (field, v)
                seq = tuple(ids[len(head):]) + (NL,)
                for i in range(len(seq)):
                    tree.setdefault(seq[:i], set()).add(seq[i])
            self.trees[field] = tree

    def _fixed(self, s):
        out = []
        while s:
            for n in range(len(s), 0, -1):
                ids = self.tok.encode(s[:n])
                if len(ids) == 1:
                    out.append(ids[0])
                    s = s[n:]
                    break
        return out

    def text(self, t):
        x = self._text.get(t)
        if x is None:
            x = self._text[t] = self.tok.decode([int(t)])
        return x

    def embed(self, ids):
        rows = {t: self.W.get_row("embed_tokens.weight", int(t)) for t in set(ids)}
        return np.stack([rows[t] for t in ids]).astype(np.float32)

    # ---- the prompts ---------------------------------------------------------

    def prompt_cot(self, caption, lyrics):
        return chat("# Instruction\n%s\n\n" % INSTRUCTION, "# Caption\n%s\n\n# Lyric\n%s\n" % (caption, lyrics))

    @staticmethod
    def cot_yaml(meta):
        """<think>, the metadata as the authors' yaml.dump (sorted keys), </think>."""
        import yaml
        items = {}
        for k in FIELDS:
            v = meta.get(k)
            if v is None:
                continue
            if k == "timesignature" and isinstance(v, str) and v.endswith("/4"):
                v = v[:-2]
            if isinstance(v, str) and v.isdigit():
                v = int(v)
            items[k] = v
        return "<think>\n%s\n</think>" % yaml.dump(items, allow_unicode=True, sort_keys=True).strip()

    def prompt_codes(self, caption, lyrics, meta):
        return chat("# Instruction\n%s\n\n" % INSTRUCTION, "# Caption\n%s\n\n# Lyric\n%s\n" % (caption, lyrics),
                    self.cot_yaml(meta) + "\n\n")

    def prompt_uncond(self):
        return chat("# Instruction\n%s\n\n" % INSTRUCTION, NEGATIVE, "<think>\n\n</think>\n\n")

    # ---- generation ------------------------------------------------------------

    def _prefill(self, prompts):
        """Run the prompts (each its own sequence); the logits of each one's
        last token (n, vocab)."""
        ids = [self.tok.encode(p) for p in prompts]
        r = max(len(x) for x in ids)
        r = -(-r // 32) * 32
        x = np.zeros((len(ids) * r, self.lm.width), np.float32)
        pos = np.full((len(ids), r), -1)
        for j, s in enumerate(ids):
            x[j * r:j * r + len(s)] = self.embed(s)
            pos[j, :len(s)] = np.arange(len(s))
        lg, = self.lm.step(x, pos, n=len(ids), outputs=("logits",))
        return np.stack([lg[j * r + len(s) - 1] for j, s in enumerate(ids)]), [len(s) for s in ids], ids

    def sample(self, lg, keep, temperature, top_p, rng):
        """The "pt" backend's sampling of logits lg (vocab,) float32 within
        the whitelist keep (None: all)."""
        x = np.full(lg.shape, -np.inf, np.float32)
        if keep is None:
            x[:] = lg
        else:
            x[keep] = lg[keep]
        if top_p is not None and top_p < 1.0:
            fin = np.nonzero(np.isfinite(x))[0]
            order = fin[np.argsort(-x[fin], kind="stable")]
            p = np.exp((x[order] - x[order[0]]).astype(np.float64))
            cum = np.cumsum(p / p.sum())
            drop = np.zeros(len(order), bool)
            drop[1:] = cum[:-1] > top_p
            x[order[drop]] = -np.inf
        if temperature <= 0:
            return int(np.argmax(x))
        fin = np.nonzero(np.isfinite(x))[0]
        z = (x[fin] / np.float32(temperature)).astype(np.float64)
        p = np.exp(z - z.max())
        return int(fin[rng.choice(len(fin), p=p / p.sum())])

    def generate(self, prompts, constraint, cfg=1.0, temperature=0.85, top_p=0.9, rng=None, max_new=3500,
                 forced=None, keep=None):
        """The generated ids of prompts (one; two with CFG: the conditional
        and the unconditional). forced: the ids to feed (a check's teacher
        forcing; keep gets each step's raw logits (n, vocab)). The LM stays
        pinned from the prompts to the last step (its caches on the GPU)."""
        rng = rng or np.random.default_rng()
        with RESIDENCY.use(*gpu_runners(self.lm.gpu, lm=self.lm).values()):
            lg, pos, _ = self._prefill(prompts)
            out = []
            for step in range(max_new):
                if keep is not None:
                    keep.append(lg)
                if len(prompts) == 2:
                    c, u = lg[0], lg[1]
                    comb = u + np.float32(cfg) * (c - u)
                else:
                    comb = lg[0]
                allowed = constraint.allowed(comb)
                t = int(forced[step]) if forced is not None else self.sample(comb, allowed, temperature, top_p, rng)
                out.append(t)
                constraint.update(t)
                if t in (self.eos, self.pad) or constraint.done or (forced is not None and step + 1 == len(forced)):
                    break
                x = self.embed([t] * len(prompts))
                lg = self.lm.step(x, np.array(pos)[:, None], n=len(prompts), outputs=("logits",))[0]
                pos = [q + 1 for q in pos]
        return out

    def parse(self, ids):
        """The metadata of phase 1's ids (the authors' parse_lm_output)."""
        text = self.tok.decode([t for t in ids if not CODE0 <= t < CODE0 + CODES])
        if "<think>" in text and "</think>" in text:
            text = text.split("<think>", 1)[1].split("</think>", 1)[0]
        meta, key = {}, None
        for line in text.split("\n"):
            if line.startswith("<") or not line.strip():
                continue
            if not line[0] in " \t" and ":" in line:
                key, v = line.split(":", 1)
                key = key.strip().lower()
                meta[key] = v
            elif key is not None:
                meta[key] += "\n" + line
        out = {}
        for k, v in meta.items():
            if k in ("bpm", "duration"):
                v = v.strip()
                out[k] = int(v) if v.isdigit() else v
            elif k == "caption":
                out[k] = " ".join(x.strip() for x in v.split("\n") if x.strip())
            elif k in ("genres", "keyscale", "language", "timesignature"):
                out[k] = v.strip()
        return out

    def plan(self, caption, lyrics, user=None, duration=None, seed=None, temperature=0.85, top_p=0.9, cfg=2.0):
        """The metadata (phase 1) and the codes (phase 2) of a request; user:
        the metadata the user gave (bpm, keyscale, timesignature, duration)."""
        rng = np.random.default_rng(seed)
        user = {k: v for k, v in (user or {}).items() if v not in (None, "")}
        if all(k in user for k in ("bpm", "keyscale", "timesignature", "duration")):
            meta = dict(user)
        else:
            ids = self.generate([self.prompt_cot(caption, lyrics)], Constraint(self, "cot", user),
                                temperature=temperature, top_p=top_p, rng=rng)
            meta = self.parse(ids)
        dur = duration or meta.get("duration")
        dur = min(max(float(dur if isinstance(dur, (int, float)) else 30), 10), MAX_DURATION)
        n = int(dur * 5)
        ids = self.generate([self.prompt_codes(caption, lyrics, meta), self.prompt_uncond()],
                            Constraint(self, "codes", target_codes=n), cfg=cfg, temperature=temperature, top_p=top_p,
                            rng=rng, max_new=n + 10)
        codes = np.array([t - CODE0 for t in ids if CODE0 <= t < CODE0 + CODES], np.int64)
        return meta, np.clip(codes, 0, MAX_CODE)
