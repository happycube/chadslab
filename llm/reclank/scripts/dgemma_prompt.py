#!/usr/bin/env python3
"""Ask a DiffusionGemma server (python -m server -m ...-BF12.gguf) with a prompt and, optionally, the
start or more of the answer: a response template whose text stands where it is and whose holes
[[n]] the model writes (n tokens each); after the template the model goes on freely.

    python scripts/dgemma_prompt.py --prompt "Write a short speech by a Party official." \\
        --response "[[40]]'What lessons do we learn from this fact, comrades? The lesson—which is also one of \\
the fundamental principles of Ingsoc—[[200]]"
    python scripts/dgemma_prompt.py --prompt-file q.txt --response-file r.txt --drafts
    python scripts/dgemma_prompt.py --prompt "What is this?" --image picture.jpg   # server: --mmproj
    python scripts/dgemma_prompt.py --chat          # turns from stdin: "prompt" or "prompt ||| template"

--drafts shows each denoising step as it comes (the canvas and step, the tokens the sampler kept,
the mean entropy, the mean probability of the draft tokens and their mean raw probability, the
draft with its tokens under --low probability in brackets; --by raw judges by the raw one);
--drafts-json FILE writes every step's record (the tokens, the probability and the entropy of
each position, the kept positions) as JSON lines. A position's probability ("prob") is that of
the sampler's logits (the step's temperature, the end tokens banned before the pinned text);
its raw probability ("raw_prob") is the model's own (its logits at temperature 1). --chat keeps the conversation, so the server's cache serves
the turns before (only the new part of a prompt is read). The key: --key or RECLANK_API_KEY.
"""
from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
import urllib.request


def post_stream(url, key, body):
    """The events of a streamed chat completion (dicts)."""
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": "Bearer " + key} if key else {})})
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            yield json.loads(data)


def show_draft(d, low, by="prob"):
    """One line a step: the canvas and step, the kept positions, the mean entropy, the mean
    probability and raw probability, and the draft with the tokens under low probability (by:
    "prob" or "raw") in brackets."""
    ent = d.get("entropy", [])
    mean = lambda v: sum(v) / max(1, len(v))  # noqa: E731
    head = "[canvas %d, step %2d: %3d kept, entropy %.3f" % (
        d["canvas"], d["step"], sum(d.get("kept", [])), mean(ent))
    for k, name in (("prob", "p"), ("raw_prob", "raw")):
        if d.get(k) is not None:
            head += ", %s %.3f" % (name, mean(d[k]))
    head += "] "
    prob, pieces = d.get("raw_prob" if by == "raw" else "prob"), d.get("pieces")
    if prob is not None and pieces is not None:
        text = "".join("[%s]" % s if p < low else s for s, p in zip(pieces, prob))
    else:
        text = d.get("text", "")
    print((head + text.replace("\n", " "))[:600], file=sys.stderr, flush=True)


def ask(a, messages, template, drafts_out):
    body = {"messages": messages + ([{"role": "assistant", "content": template}] if template else []),
            "stream": True, "max_tokens": a.max_tokens, "diffusion_drafts": bool(a.drafts or drafts_out)}
    if a.seed is not None:
        body["seed"] = a.seed
    if a.think:
        body["chat_template_kwargs"] = {"enable_thinking": True}
    t0 = time.time()
    reasoning, content, n_drafts = [], [], 0
    for ev in post_stream(a.url.rstrip("/"), a.key, body):
        if ev.get("object") == "diffusion.draft":
            n_drafts += 1
            if drafts_out is not None:
                drafts_out.write(json.dumps(ev["draft"]) + "\n")
            if a.drafts:
                show_draft(ev["draft"], a.low, a.by)
            continue
        for ch in ev.get("choices", []):
            delta = ch.get("delta", {})
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
                if a.show_reasoning:
                    sys.stdout.write(delta["reasoning_content"])
                    sys.stdout.flush()
            if delta.get("content"):
                content.append(delta["content"])
                sys.stdout.write(delta["content"])
                sys.stdout.flush()
    sys.stdout.write("\n")
    print("(%.1f s%s)" % (time.time() - t0, ", %d draft steps" % n_drafts if n_drafts else ""), file=sys.stderr)
    return "".join(content)


def read(path_or_text, path):
    if path:
        with open(path, encoding="utf-8") as f:
            return f.read()
    return path_or_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get("RECLANK_URL", "http://127.0.0.1:8081"))
    ap.add_argument("--key", default=os.environ.get("RECLANK_API_KEY"))
    ap.add_argument("--prompt")
    ap.add_argument("--prompt-file")
    ap.add_argument("--image", help="an image file (the server needs --mmproj)")
    ap.add_argument("--response", help="the answer's template: pinned text and [[n]] holes")
    ap.add_argument("--response-file")
    ap.add_argument("--system")
    ap.add_argument("--think", action="store_true", help="the thought channel on")
    ap.add_argument("--show-reasoning", action="store_true", help="print the thought too")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--drafts", action="store_true", help="show each denoising step (stderr)")
    ap.add_argument("--drafts-json", help="write every step's record to this file (JSON lines)")
    ap.add_argument("--low", type=float, default=0.5, help="--drafts: the probability counted as low")
    ap.add_argument("--by", choices=("prob", "raw"), default="prob",
                    help="--drafts: --low of the sampler's probability or of the raw one")
    ap.add_argument("--chat", action="store_true", help="turns from stdin: 'prompt' or 'prompt ||| template'")
    a = ap.parse_args()
    out = open(a.drafts_json, "w", encoding="utf-8") if a.drafts_json else None
    messages = [{"role": "system", "content": a.system}] if a.system else []
    if not a.chat:
        prompt = read(a.prompt, a.prompt_file)
        if not prompt:
            ap.error("--prompt or --prompt-file (or --chat)")
        content = prompt
        if a.image:
            with open(a.image, "rb") as f:
                data = base64.b64encode(f.read()).decode("ascii")
            mime = mimetypes.guess_type(a.image)[0] or "image/png"
            content = [{"type": "image_url", "image_url": {"url": "data:%s;base64,%s" % (mime, data)}},
                       {"type": "text", "text": prompt}]
        ask(a, messages + [{"role": "user", "content": content}], read(a.response, a.response_file), out)
        return
    if a.image:
        ap.error("--image with --chat is not supported")
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        prompt, _, template = line.partition("|||")
        messages.append({"role": "user", "content": prompt.strip()})
        answer = ask(a, messages, template.strip() or None, out)
        # the history holds the answer, not its thought (the model card's advice)
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
