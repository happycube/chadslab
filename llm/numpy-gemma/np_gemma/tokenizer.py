"""Change text into token ids. Change token ids back into text.

The tokenizer uses the BPE data in tokenizer.json. The pipeline has four steps:
1. Replace each space with the character U+2581.
2. Split the text. After step 1, this step does nothing.
3. Apply the BPE merges. Use byte fallback for a character that is not in the
   vocabulary.
4. Decode the ids. The decoder replaces U+2581 with a space, joins the byte
   tokens, and joins the parts.

This module also supplies the chat template.
"""
from __future__ import annotations

import json
from pathlib import Path

# The sentence-piece space character.
SPACE = "▁"


class Tokenizer:
    """Convert text to token ids and back.

    The constructor reads these parts from tokenizer.json:
        vocab          The token strings and their ids.
        merges         The BPE merge rules and their ranks.
        added_tokens   The special tokens.
    """

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
        # Make the byte token name for each byte value. For example: <0x1F>.
        self.byte_tokens = {b: "<0x" + format(b, "02X") + ">" for b in range(256)}
        self.added = {}
        self.special_ids = set()
        for a in data.get("added_tokens", []):
            self.added[a["content"]] = a["id"]
            if a.get("special"):
                self.special_ids.add(a["id"])
        # Sort the special tokens by length. Test the longest token first.
        self._added_sorted = sorted(self.added, key=len, reverse=True)
        self.bos_id = self.added.get("<bos>", self.vocab.get("<bos>"))
        self.eos_id = self.added.get("<eos>", self.vocab.get("<eos>"))
        self.pad_id = self.added.get("<pad>", self.vocab.get("<pad>"))

    # ---- encoding ----------------------------------------------------------
    def _normalize(self, text):
        """Replace each space with U+2581."""
        return text.replace(" ", SPACE)

    def _initial(self, piece):
        """Make the first symbol list for one piece of text.

        Use one symbol for each character. If the character is not in the
        vocabulary, use the byte tokens of its UTF-8 bytes.
        """
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
        """Merge adjacent symbols.

        Always merge the pair with the lowest rank. Stop when no adjacent pair
        has a rank.
        """
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
        """Return the special token that starts at position i.

        Return None when no special token starts at that position.
        """
        for token in self._added_sorted:
            if text.startswith(token, i):
                return token
        return None

    def encode(self, text, add_special_tokens=False):
        """Change text into token ids.

        Keep each special token as one id. Apply the normalizer and the BPE to
        the other parts.
        """
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
        """Return the token string for one id. Return an empty string for an unknown id."""
        if i in self.ids_to_tokens:
            return self.ids_to_tokens[i]
        for content, tid in self.added.items():
            if tid == i:
                return content
        return ""

    def decode(self, ids, skip_special_tokens=False):
        """Change token ids into text.

        Join adjacent byte tokens first. Then decode the bytes as UTF-8.
        Replace an invalid byte sequence with the replacement character.
        """
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
            # A byte token has the form <0xXX>. Collect its byte value.
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
        """Build the chat prompt from the message list.

        Add one turn for each message. Change the role "assistant" to "model".
        Add a system turn. If thinking is true, open the system turn with the
        think token. If thinking is false, close an empty thought channel.
        """
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
