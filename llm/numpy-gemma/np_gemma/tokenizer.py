"""Pure-Python (NumPy-only project) Gemma 4 tokenizer: HF BPE + byte fallback + chat template.

Reproduces the pipeline in tokenizer.json:
  normalizer  : Replace(" ", U+2581)
  pre_tokenizer: Split(" ")  (no-op after normalization)
  model       : BPE, byte_fallback=true
  decoder     : Replace(U+2581, " ") -> ByteFallback -> Fuse
"""
from __future__ import annotations

import json
from pathlib import Path

SPACE = "\u2581"


class Tokenizer:
    def __init__(self, tokenizer_json):
        data = json.loads(Path(tokenizer_json).read_text())
        m = data["model"]
        if m.get("type") != "BPE":
            raise ValueError("expected a BPE model, got " + str(m.get("type")))
        self.vocab = m["vocab"]
        self.ids_to_tokens = {i: t for t, i in self.vocab.items()}
        self.merges = {}
        for rank, pair in enumerate(m["merges"]):
            if isinstance(pair, str):
                a, b = pair.split(" ", 1)
            else:
                a, b = pair
            self.merges[(a, b)] = rank
        self.byte_fallback = bool(m.get("byte_fallback", False))
        self.unk_token = m.get("unk_token")
        self.unk_id = self.vocab.get(self.unk_token)
        self.byte_tokens = {b: "<0x" + format(b, "02X") + ">" for b in range(256)}
        self.added = {}
        self.special_ids = set()
        for a in data.get("added_tokens", []):
            self.added[a["content"]] = a["id"]
            if a.get("special"):
                self.special_ids.add(a["id"])
        self._added_sorted = sorted(self.added, key=len, reverse=True)
        self.bos_id = self.added.get("<bos>", self.vocab.get("<bos>"))
        self.eos_id = self.added.get("<eos>", self.vocab.get("<eos>"))
        self.pad_id = self.added.get("<pad>", self.vocab.get("<pad>"))

    # ---- encoding ----------------------------------------------------------
    def _normalize(self, text):
        return text.replace(" ", SPACE)

    def _initial(self, piece):
        out = []
        for ch in piece:
            if ch in self.vocab:
                out.append(ch)
            elif self.byte_fallback:
                for b in ch.encode("utf-8"):
                    name = self.byte_tokens[b]
                    out.append(name if name in self.vocab else self.unk_token)
            else:
                out.append(self.unk_token)
        return out

    def _bpe(self, piece):
        syms = self._initial(piece)
        while len(syms) > 1:
            best_rank = None
            best_i = -1
            for i in range(len(syms) - 1):
                r = self.merges.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
                    best_i = i
            if best_i < 0:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        return syms

    def _matches_added(self, text, i):
        for token in self._added_sorted:
            if text.startswith(token, i):
                return token
        return None

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        n = len(text)
        while i < n:
            token = self._matches_added(text, i)
            if token is not None:
                ids.append(self.added[token])
                i += len(token)
                continue
            j = i
            while j < n and self._matches_added(text, j) is None:
                j += 1
            piece = self._normalize(text[i:j])
            for sym in self._bpe(piece):
                ids.append(self.vocab.get(sym, self.unk_id))
            i = j
        return ids

    # ---- decoding ----------------------------------------------------------
    def _token_string(self, i):
        if i in self.ids_to_tokens:
            return self.ids_to_tokens[i]
        for content, tid in self.added.items():
            if tid == i:
                return content
        return ""

    def decode(self, ids, skip_special_tokens=False):
        tokens = []
        for i in ids:
            if skip_special_tokens and i in self.special_ids:
                continue
            tokens.append(self._token_string(int(i)).replace(SPACE, " "))
        out = bytearray()
        pending = bytearray()

        def flush():
            if pending:
                out.extend(pending.decode("utf-8", errors="replace").encode("utf-8"))
                pending.clear()

        for t in tokens:
            if len(t) == 6 and t.startswith("<0x") and t.endswith(">"):
                try:
                    pending.append(int(t[3:5], 16))
                    continue
                except ValueError:
                    pass
            flush()
            out.extend(t.encode("utf-8"))
        flush()
        return out.decode("utf-8", errors="replace")

    # ---- chat --------------------------------------------------------------
    def apply_chat_template(self, messages, add_generation_prompt=True, thinking=False):
        parts = ["<bos>"]
        system = [m for m in messages if m.get("role") == "system"]
        if thinking:
            parts.append("<|turn>system\n<|think|>\n")
            for m in system:
                parts.append(str(m.get("content", "")))
            parts.append("<turn|>\n")
        else:
            for m in system:
                parts.append("<|turn>system\n" + str(m.get("content", "")) + "<turn|>\n")
        for m in messages:
            role = m.get("role")
            if role == "system":
                continue
            role = "model" if role == "assistant" else role
            parts.append("<|turn>" + role + "\n" + str(m.get("content", "")) + "<turn|>\n")
        if add_generation_prompt:
            parts.append("<|turn>model\n")
            if not thinking:
                parts.append("<|channel>thought\n<channel|>")
        return "".join(parts)
