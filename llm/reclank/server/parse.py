"""The document parsing server: python -m server.parse -m DIR [--cpu] [--no-drafter] [--port 8101].

Nemotron 3.5 Parse (reclank/models/nemoparse, docs/NEMOTRON_PARSE.md): a page image to its text with
layout, a block a region of the page: its class (Text, Title, Section-header, List-item, Table,
Picture, Formula, Caption, Page-header, Page-footer, Footnote, Code, Chart, ...), its box and its
text (Markdown; tables as LaTeX tabular; formulas as LaTeX). DIR is the converted directory
(scripts/convert_nemoparse_f20.py: the matrices in F20, the DFlash2 drafter in BF12) or the download
(float32). The model runs on the GPU or with --cpu in C, float32 sums and activations either way; the
DFlash2 drafter (dflash2/ of DIR) drafts 15 tokens a round on the CPU, the MTP head one a step on the
GPU (RECLANK_NPARSE_DFLASH=1: DFlash2 there too), the output the plain greedy one either way
(--no-drafter: no DFlash2). One page at a time; a PDF's
pages are images rendered by the client (150 dpi is the card's example). The API:

    POST /v1/chat/completions   OpenAI's chat: the last user message holds one image part
                                ({"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}})
                                and, optionally, a text part: the task prompt of control tokens
                                (default "</s><s><predict_bbox><predict_classes><output_markdown>
                                <predict_text_in_pic>"; see the card's "Task prompts").
                                "max_tokens" (default 8744), "repetition_penalty" (default 1.1),
                                "stream"; "temperature" (default 0: greedy, the card's contract),
                                "top_k", "top_p", "seed" (default 0) sample instead (the
                                checkpoint's DocumentParser samples its crops at 0.1 to 0.2).
                                message.content is the raw output (the class and coordinate
                                tokens kept, </s> dropped), as vLLM's with skip_special_tokens
                                false; the response's "parse" holds the structured form (below).
    POST /v1/parse              {"image": base64 or a data URI, "prompt", "max_tokens",
                                "repetition_penalty", "temperature", "top_k", "top_p", "seed"}
                                -> {"raw": the raw output, "blocks",
                                "markdown", "image_size": [width, height], "finish_reason",
                                "usage", "timings"}
    GET  /v1/models, /health

The structured form (model.blocks, as the checkpoint's postprocess_document): "blocks": [{"class",
"bbox": [x1, y1, x2, y2] (0..1 of the page's content), "bbox_pixels" (in the sent image's pixels),
"text"}] in reading order, an exact repeat of a block dropped; "markdown": the blocks' texts joined
(a Title "# ", a Section-header "## "; tables stay LaTeX: the checkpoint's nemotron_parse package
turns the raw output into HTML tables and charts when a client needs them).
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import threading
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import http as H  # noqa: E402
from server.media import MediaSource  # noqa: E402


class Parser:
    """The model, its lock, and a request's page to its output."""

    def __init__(self, path, model_id, gpu=True, cpu_threads=None, drafter=True):
        from reclank.models.nemoparse.engine import ParseEngine
        from reclank.models.nemoparse.model import MAX_NEW, PROMPT, REP_PENALTY
        t0 = time.time()
        self.e = ParseEngine(path, gpu=gpu, cpu_threads=cpu_threads, drafter=drafter)
        self.model_id = model_id
        self.lock = threading.Lock()
        self.prompt, self.max_new, self.penalty = PROMPT, MAX_NEW, REP_PENALTY
        print("nemotron parse: %s (%s) on the %s, drafts: %s, in %.1f s" % (
            path, self.e.W.kind, "GPU" if gpu else "CPU",
            "DFlash2" if self.e.df is not None else "MTP" if self.e.mtp else "none", time.time() - t0), flush=True)

    def runners(self):
        return self.e.runners()

    def options(self, body):
        """(prompt, max_tokens, repetition_penalty, sampler) of a request (ValueError when not valid)."""
        from reclank.models.nemoparse.model import Sampler
        prompt = body.get("prompt") or self.prompt
        self.e.tok.encode(prompt)
        n = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_new)
        if n < 1:
            raise ValueError("max_tokens must be at least 1")
        p = float(body.get("repetition_penalty") or self.penalty)
        if not p > 0:
            raise ValueError("repetition_penalty must be positive")
        t = float(body.get("temperature") or 0)
        k = int(body.get("top_k") if body.get("top_k") is not None else 0)
        tp = float(body.get("top_p") if body.get("top_p") is not None else 1.0)
        if t < 0 or not 0 < tp <= 1:
            raise ValueError("temperature must be at least 0, top_p in (0, 1]")
        sampler = Sampler(t, max(k, 0), tp, int(body.get("seed") or 0)) if t > 0 and k != 1 else None
        return prompt, min(n, self.max_new), p, sampler

    def run(self, image, prompt, max_new, penalty, sampler=None, on_text=None):
        """The parse of a page (bytes of an image file): engine.parse()'s result with "raw" (the new
        tokens' text, </s> dropped) and "parsed" (model.blocks)."""
        from reclank.models.nemoparse.model import blocks
        tok = self.e.tok
        on_token = None
        if on_text is not None:
            got = bytearray()
            sent = [0]

            def piece(t):
                if t == self.e.cfg.eos:
                    return
                got.extend(tok.piece(t))
                text = got.decode("utf-8", "ignore")
                if len(text) > sent[0]:
                    on_text(text[sent[0]:])
                    sent[0] = len(text)
            on_token = piece
        with self.lock:
            r = self.e.parse(image, prompt, max_new, penalty, on_token=on_token, sampler=sampler)
        ids = r["ids"][:-1] if r["ids"] and r["ids"][-1] == self.e.cfg.eos else r["ids"]
        r["raw"] = tok.text(ids)
        r["parsed"] = blocks(r["raw"], r["image_size"])
        return r


def _image_of(body, media):
    """The bytes of the request's image: "image" (/v1/parse), or the last user message's image part."""
    if body.get("image") is not None:
        v = body["image"]
        return media.bytes_of(v) if str(v).startswith("data:") else base64.b64decode(v), None
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise ValueError("messages must be a list with a user message")
    user = [m for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
    if not user:
        raise ValueError("no user message")
    content = user[-1].get("content")
    parts = content if isinstance(content, list) else [{"type": "text", "text": content or ""}]
    images = [p for p in parts if isinstance(p, dict) and p.get("type") in ("image_url", "image", "input_image")]
    if len(images) != 1:
        raise ValueError("a request holds one page image (%d given)" % len(images))
    text = "".join(p.get("text") or "" for p in parts if isinstance(p, dict) and p.get("type") in ("text", "input_text"))
    return media.image(images[0], None)[0], text.strip() or None


class _ParseHandler(H._Handler):
    server_version = "reclank-parse/1.0"

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        p = self.server.parser
        if path in ("/health", ""):
            return self._json(200, {"status": "ok", "model": p.model_id})
        if not self._authorized():
            return
        if self._admin(path, False):
            return
        if path in ("/v1/models", "/models"):
            m = {"id": p.model_id, "object": "model", "owned_by": "reclank", "max_tokens": p.max_new,
                 "default_prompt": p.prompt, "devices": "gpu" if p.e.gpu else "cpu", "weights": p.e.W.kind,
                 "drafts": "dflash2" if p.e.df is not None else "mtp" if p.e.mtp else "none"}
            share = self.server.gpu_share
            if share is not None:
                m["gpu_locked"] = bool(share.models().get(p.model_id))
            return self._json(200, {"object": "list", "data": [m]})
        self._error(404, "no route %s" % self.path, code="not_found")

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if not self._authorized():
            return
        if self._admin(path, True):
            return
        chat = path in ("/v1/chat/completions", "/chat/completions")
        if not chat and path not in ("/v1/parse", "/parse"):
            return self._error(404, "no route %s" % self.path, code="not_found")
        p = self.server.parser
        try:
            body = self._body()
            if not isinstance(body, dict):
                raise ValueError("the body must be a JSON object")
            image, text = _image_of(body, self.server.media)
            if text and not body.get("prompt"):
                body["prompt"] = text
            prompt, max_new, penalty, sampler = p.options(body)
        except H._BodyError as exc:
            return self._error(exc.status, str(exc))
        except (ValueError, TypeError, KeyError) as exc:
            return self._error(400, str(exc))
        stream = chat and bool(body.get("stream"))
        cid = "chatcmpl-%d" % int(time.time() * 1e6)
        created = int(time.time())
        if stream:
            self._sse_start()

            def on_text(piece):
                self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": p.model_id,
                           "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": p.model_id,
                       "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]})
        try:
            r = p.run(image, prompt, max_new, penalty, sampler, on_text=on_text if stream else None)
        except (ValueError, OSError) as exc:      # (an image PIL cannot read)
            if stream:
                self._sse({"error": {"message": str(exc), "type": "invalid_request_error"}})
                return self._sse_end()
            return self._error(400, str(exc))
        prompt_tokens = r["prefix_rows"] + len(p.e.tok.encode(prompt))
        usage = {"prompt_tokens": prompt_tokens, "completion_tokens": r["tokens"],
                 "total_tokens": prompt_tokens + r["tokens"]}
        timings = {"encode_s": round(r["encode_s"], 3), "decode_s": round(r["decode_s"], 3),
                   "tokens_per_s": round(r["tokens"] / max(r["decode_s"], 1e-9), 1), "steps": r["steps"],
                   "drafted": r["drafted"]}
        if not getattr(self.server, "quiet", False):
            print("parse: %dx%d, %d rows, %d tokens (%s), encode %.2f s, decode %.2f s (%.0f tokens/s)" % (
                r["image_size"][0], r["image_size"][1], r["prefix_rows"], r["tokens"], r["finish_reason"],
                r["encode_s"], r["decode_s"], timings["tokens_per_s"]), flush=True)
        parsed = r["parsed"]
        if stream:
            self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": p.model_id,
                       "choices": [{"index": 0, "delta": {}, "finish_reason": r["finish_reason"]}],
                       "usage": usage, "parse": {"blocks": parsed["blocks"], "markdown": parsed["markdown"],
                                                 "image_size": list(r["image_size"])}})
            self._sse_data("[DONE]")
            return self._sse_end()
        if chat:
            return self._json(200, {
                "id": cid, "object": "chat.completion", "created": created, "model": p.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": r["raw"]},
                             "finish_reason": r["finish_reason"]}],
                "usage": usage, "timings": timings,
                "parse": {"blocks": parsed["blocks"], "markdown": parsed["markdown"],
                          "image_size": list(r["image_size"])}})
        self._json(200, {"model": p.model_id, "raw": r["raw"], "blocks": parsed["blocks"],
                         "markdown": parsed["markdown"], "image_size": list(r["image_size"]),
                         "coordinate_frame": parsed["coordinate_frame"], "finish_reason": r["finish_reason"],
                         "usage": usage, "timings": timings})


def main(argv=None):
    ap = argparse.ArgumentParser(description="the document parsing server (Nemotron 3.5 Parse)")
    ap.add_argument("-m", "--model", required=True, help="the Nemotron-3.5-Parse checkpoint directory")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8101)
    ap.add_argument("--model-id", default="nemotron-3.5-parse")
    ap.add_argument("--cpu", action="store_true", help="run on the CPU instead of the GPU")
    ap.add_argument("--no-drafter", action="store_true", help="no DFlash2 drafter (dflash2/ of the model)")
    ap.add_argument("--cpu-threads", type=int, default=None,
                    help="the team of a program on the CPU (default: the CPUs of the process)")
    ap.add_argument("--media-dir", default=None, help="a directory of page images that a request may name")
    ap.add_argument("--quiet", action="store_true")
    H.add_http_args(ap)
    from server.gpushare import add_gpu_args
    add_gpu_args(ap)
    a = ap.parse_args(argv)
    parser = Parser(a.model, a.model_id, gpu=not a.cpu, cpu_threads=a.cpu_threads, drafter=not a.no_drafter)
    from server import idle
    httpd = idle.server(H._Server, (a.host, a.port), _ParseHandler)
    httpd.parser = parser
    httpd.media = MediaSource(a.media_dir)
    httpd.quiet = a.quiet
    httpd.gpu_share = None
    models = {a.model_id: parser} if parser.runners() else {}
    locks = {m for m in a.pin.split(",") if m}
    if locks - set(models) - {"*"}:
        ap.error("--gpu-lock: no GPU model %s" % ", ".join(sorted(locks - set(models) - {"*"})))
    if models:
        # the encoder's and the decoder's programs on the GPU in RESIDENCY's hands (a peer's room)
        from reclank.backends.cuda.gpumm import mem_info
        from reclank.encoders.runtime import RESIDENCY
        RESIDENCY.enabled = True
        RESIDENCY.budget = max(0, mem_info()[1] - RESIDENCY.reserve)
        for part, r in parser.runners().items():
            r.label = "%s/%s" % (a.model_id, part)
            if "*" in locks or a.model_id in locks:
                RESIDENCY.pin(r)
        if a.gpu_peers == "on":
            from server.gpushare import RunnerShare, attach
            httpd.gpu_share = attach(RunnerShare("parse", lambda: models, lambda: [parser.lock], locks=locks,
                                                 port=httpd.server_address[1], host=a.host, api_key=a.api_key,
                                                 wait=a.gpu_wait, hold=a.gpu_hold, quiet=a.quiet))
    httpd.api_key = a.api_key or None
    httpd.cors_origin = a.cors_origin or None
    httpd.max_body = int(a.max_body_mb * (1 << 20))
    print("reclank parse server http://%s:%d/v1  model=%s  key %s" % (
        a.host, httpd.server_address[1], a.model_id, "required" if httpd.api_key else "OFF"), flush=True)
    try:
        return idle.serve(httpd, a.idle_exit)
    except KeyboardInterrupt:
        return 0
    finally:
        httpd.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
