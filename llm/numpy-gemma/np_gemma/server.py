"""An OpenAI compatible HTTP server for the NumPy Gemma runtime.

The server uses only the Python standard library. It gives these paths:

    GET  /v1/models
    GET  /v1/models/{id}
    POST /v1/chat/completions
    POST /v1/completions
    GET  /health

The chat path takes the fields that the OpenAI clients send: model, messages,
max_tokens, temperature, top_k, top_p, min_p, stop, stream, seed, and the three
penalties. A stream uses the server-sent-event format of the OpenAI API. The
server also takes the bare paths with no /v1.

The model is not thread safe and shares one key and value cache. The server
runs one request at a time. Other requests wait.
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

from .chat import parse_output
from .model import Session
from .sampling import Sampler


# A stream sends this often while it waits for a token. A prompt read can take
# minutes, and a client closes a stream that stays silent for too long.
KEEPALIVE_SECONDS = 15.0


def content_text(content):
    """Return the text of a message content. Take a string or a part list."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, dict):
                if part.get("text") is not None:
                    out.append(str(part["text"]))
            else:
                out.append(str(part))
        return "".join(out)
    return str(content)


def stop_strings(value):
    """Return the list of stop strings. Take a string or a list."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        return [str(v) for v in value if v]
    return []


def truncate(text, stops):
    """Cut the text at the first stop string. Return the text and the cut."""
    cut = None
    for s in stops:
        i = text.find(s)
        if i >= 0 and (cut is None or i < cut):
            cut = i
    if cut is None:
        return text, False
    return text[:cut], True


REPEAT_MIN = 3
REPEAT_MAX = 256


def repeat_len(text):
    """Return the length of a repeated tail, or 0 when the text is fine.

    A greedy decode can fall into a loop and repeat one phrase to the token
    limit. A tail that holds the same block three times or more is a loop.
    """
    n = len(text)
    for size in range(16, REPEAT_MAX + 1, 4):
        need = size * REPEAT_MIN
        if n < need:
            continue
        block = text[n - size:]
        if block and text[n - need:] == block * REPEAT_MIN:
            return need
    return 0


def cut_repeat(text):
    """Remove a repeated tail, keeping the first copy of the block."""
    need = repeat_len(text)
    if not need:
        return text
    return text[:len(text) - need + need // REPEAT_MIN]


def tool_done(text):
    """Return True when the model asks the caller for a tool result.

    The model writes <|tool_response> to ask for the result. Without this test
    the model repeats the marker until it reaches the token limit. The marker
    also stops a turn that waits for a result with no new call."""
    return "<|tool_response>" in text


def common_len(a, b):
    """Return the count of the first items that a and b share."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class Backend:
    """Hold the model, the tokenizer, and the default settings."""

    def __init__(self, model, tokenizer, cfg=None, model_id="np-gemma",
                 thinking=False, max_tokens=1024, temperature=1.0, top_k=None,
                 top_p=None, drafter=None, n_draft=2):
        self.model = model
        # The MTP drafter and the count of drafts for each step. None turns
        # MTP off.
        self.drafter = drafter
        self.n_draft = n_draft
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.model_id = model_id
        self.thinking = thinking
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.lock = threading.Lock()
        # Keep the key and value cache between the turns of a chat. The model
        # then reads only the new tokens of a turn, not the whole history.
        self.sessions = []
        self.max_sessions = 4
        # A shared prefix below this count is not worth a cache hit, and it
        # would evict the conversation that the cache should keep.
        self.min_shared = 16

    def model_info(self, name=None):
        """Return one model description."""
        return {"id": name or self.model_id, "object": "model", "created": 0,
                "owned_by": "np-gemma"}

    def prompt_ids(self, messages, thinking=None, tools=None):
        """Return the token ids of a chat prompt."""
        think = self.thinking if thinking is None else bool(thinking)
        # Keep every field. tool_calls on an assistant message and tool_call_id
        # on a tool message carry the tool result to the model.
        msgs = []
        for m in messages:
            msg = dict(m)
            msg["role"] = m.get("role", "user")
            msg["content"] = content_text(m.get("content"))
            msgs.append(msg)
        text = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                                  thinking=think, tools=tools)
        return self.tokenizer.encode(text)

    def completion_ids(self, prompt):
        """Return the token ids of a raw completion prompt."""
        return self.tokenizer.encode(prompt)

    def sampler(self, req):
        """Build the sampler from the request fields and the defaults."""
        def pick(name, default):
            v = req.get(name)
            return default if v is None else v
        return Sampler(temperature=pick("temperature", self.temperature),
                       top_k=pick("top_k", self.top_k), top_p=pick("top_p", self.top_p),
                       min_p=req.get("min_p"),
                       repetition_penalty=req.get("repetition_penalty"),
                       presence_penalty=req.get("presence_penalty"),
                       frequency_penalty=req.get("frequency_penalty"),
                       seed=req.get("seed"))

    def stop_ids(self):
        """Return the token ids that end a generation."""
        return set(getattr(self.tokenizer, "stop_ids", ()) or ())

    def pick_session(self, prompt_ids, max_tokens):
        """Return the session with the longest shared prefix, or a new one.

        A hit moves the session to the front, so a smaller cache keeps the
        conversation that is in use and drops the one that is not.
        """
        best = None
        best_n = 0
        for session in self.sessions:
            n = common_len(session.ids, prompt_ids)
            if n > best_n:
                best = session
                best_n = n
        if best is not None and best_n >= self.min_shared:
            self.sessions.remove(best)
            self.sessions.insert(0, best)
            return best, best_n
        best = Session(self.model, max_len=len(prompt_ids) + max_tokens + 8,
                       drafter=self.drafter, n_draft=self.n_draft)
        self.sessions.insert(0, best)
        del self.sessions[self.max_sessions:]
        return best, 0

    def generate(self, prompt_ids, max_tokens, sampler, eos_ids):
        """Yield one token id at a time for a prompt."""
        session, shared = self.pick_session(prompt_ids, max_tokens)
        print('[np-gemma] reuse=%d new=%d' % (shared, len(prompt_ids) - shared),
              file=sys.stderr, flush=True)
        yield from session.generate_stream(prompt_ids, max_new_tokens=max_tokens,
                                           eos_ids=eos_ids, sampler=sampler)
        st = session.mtp_stats
        if self.drafter is not None and st.get("drafts"):
            print('[np-gemma] mtp steps=%d drafts=%d accepted=%d (%d%%)' % (
                st["steps"], st["drafts"], st["accepted"],
                100 * st["accepted"] // st["drafts"]), file=sys.stderr, flush=True)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "np-gemma/1.0"

    # ---- small helpers -----------------------------------------------------

    def log_message(self, fmt, *args):
        if getattr(self.server, "quiet", False):
            return
        super().log_message(fmt, *args)

    def _json(self, status, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status, message, kind="invalid_request_error"):
        self._json(status, {"error": {"message": message, "type": kind, "code": None}})

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b""
        return json.loads(raw) if raw.strip() else {}

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _sse(self, obj):
        data = ("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode("utf-8")
        self.wfile.write(("%x\r\n" % len(data)).encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _sse_data(self, text):
        data = ("data: " + text + "\n\n").encode("utf-8")
        self.wfile.write(("%x\r\n" % len(data)).encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _sse_end(self):
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()

    # ---- routes ------------------------------------------------------------

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self):
        path = urlsplit(self.path).path.rstrip("/")
        backend = self.server.backend
        if path in ("/v1/models", "/models"):
            self._json(200, {"object": "list", "data": [backend.model_info()]})
        elif path.startswith("/v1/models/") or path.startswith("/models/"):
            self._json(200, backend.model_info(path.rsplit("/", 1)[-1]))
        elif path in ("", "/health", "/healthz", "/v1/health"):
            self._json(200, {"status": "ok", "model": backend.model_id})
        else:
            self._error(404, "unknown path: " + path)

    def do_POST(self):
        path = urlsplit(self.path).path.rstrip("/")
        if path.endswith("/chat/completions"):
            self._chat()
        elif path.endswith("/completions"):
            self._completions()
        else:
            self._error(404, "unknown path: " + path)

    def _chat(self):
        try:
            req = self._body()
        except Exception as exc:
            self._error(400, "invalid JSON: %s" % exc)
            return
        messages = req.get("messages")
        if not isinstance(messages, list) or not messages:
            self._error(400, "messages must be a non-empty list")
            return
        backend = self.server.backend
        thinking = req.get("thinking")
        if thinking is None and req.get("reasoning_effort"):
            thinking = req.get("reasoning_effort") not in ("none", "off", 0)
        print('[np-gemma] messages=%d tools=%d' % (len(messages),
              len(req.get("tools") or [])), file=sys.stderr, flush=True)
        try:
            prompt_ids = backend.prompt_ids(messages, thinking, req.get("tools"))
        except Exception as exc:
            self._error(400, "prompt failed: %s" % exc)
            return
        self._run(req, prompt_ids, chat=True)

    def _completions(self):
        try:
            req = self._body()
        except Exception as exc:
            self._error(400, "invalid JSON: %s" % exc)
            return
        prompt = req.get("prompt", "")
        if isinstance(prompt, list):
            prompt = prompt[0] if prompt else ""
        if not isinstance(prompt, str):
            prompt = str(prompt)
        prompt_ids = self.server.backend.completion_ids(prompt)
        self._run(req, prompt_ids, chat=False)

    def _run(self, req, prompt_ids, chat):
        backend = self.server.backend
        max_tokens = req.get("max_tokens")
        if max_tokens is None:
            max_tokens = req.get("max_completion_tokens")
        if max_tokens is None:
            max_tokens = backend.max_tokens
        max_tokens = max(int(max_tokens), 0)
        print('[np-gemma] max_tokens=%d stream=%s' % (max_tokens, bool(req.get('stream'))),
              file=sys.stderr, flush=True)
        stops = stop_strings(req.get("stop"))
        sampler = backend.sampler(req)
        eos = backend.stop_ids()
        model_id = req.get("model") or backend.model_id
        if req.get("stream"):
            self._stream(req, prompt_ids, max_tokens, sampler, stops, eos, model_id, chat)
        else:
            self._block(prompt_ids, max_tokens, sampler, stops, eos, model_id, chat)

    def _ids(self, prompt_ids, max_tokens, sampler, eos):
        """Run the generation. Return the token ids and the finish reason."""
        tok = self.server.backend.tokenizer
        out = []
        finish = "length"
        for token in self.server.backend.generate(prompt_ids, max_tokens, sampler, eos):
            out.append(token)
            if token in eos:
                finish = "stop"
                break
            raw = tok.decode(out, skip_special_tokens=False)
            if tool_done(raw):
                finish = ("tool_calls" if parse_output(raw)["tool_calls"]
                          else "stop")
                break
            if repeat_len(raw):
                finish = "stop"
                break
        return out, finish

    def _block(self, prompt_ids, max_tokens, sampler, stops, eos, model_id, chat):
        backend = self.server.backend
        with backend.lock:
            out, finish = self._ids(prompt_ids, max_tokens, sampler, eos)
        print('[np-gemma] prompt=%d out=%d finish=%s' % (len(prompt_ids), len(out), finish),
              file=sys.stderr, flush=True)
        raw = backend.tokenizer.decode(out, skip_special_tokens=False)
        parsed = parse_output(raw)
        text, cut = truncate(cut_repeat(parsed["content"]), stops)
        calls = parsed["tool_calls"]
        if calls:
            finish = "tool_calls"
        elif cut:
            finish = "stop"
        created = int(time.time())
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        usage = {"prompt_tokens": len(prompt_ids), "completion_tokens": len(out),
                 "total_tokens": len(prompt_ids) + len(out)}
        if chat:
            obj = "chat.completion"
            message = {"role": "assistant",
                       "content": (text or None) if calls else text}
            if parsed["reasoning"]:
                message["reasoning_content"] = parsed["reasoning"]
            if calls:
                message["tool_calls"] = calls
            choice = {"index": 0, "message": message, "finish_reason": finish}
        else:
            obj = "text_completion"
            choice = {"index": 0, "text": text, "logprobs": None,
                      "finish_reason": "stop" if cut else finish}
        self._json(200, {"id": cid, "object": obj, "created": created,
                         "model": model_id, "choices": [choice], "usage": usage})

    def _stream(self, req, prompt_ids, max_tokens, sampler, stops, eos, model_id, chat):
        backend = self.server.backend
        tok = backend.tokenizer
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))
        cid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())
        obj = "chat.completion.chunk" if chat else "text_completion"

        def event(delta, finish):
            return {"id": cid, "object": obj, "created": created,
                    "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        def piece(kind, text):
            if kind == "reasoning_content":
                return {"reasoning_content": text}
            return {"content": text} if chat else {"text": text}

        self._sse_start()
        out = []
        sent_reason = ""
        sent_content = ""
        calls = []
        finish = "length"

        def advance(sent, value, kind):
            """Send the part of value that sent does not hold. Return the new sent.

            A reader joins the deltas, so a shorter value cannot correct one that
            already went out. Forget sent when the value no longer extends it.
            """
            if not value.startswith(sent):
                sent = ""
            if len(value) > len(sent):
                self._sse(event(piece(kind, value[len(sent):]), None))
                sent = value
            return sent

        # Read the prompt and generate on a worker, so this thread can keep the
        # connection alive. A long prompt read sends no token for minutes, and a
        # client closes a stream that stays silent for too long.
        q = queue.Queue()
        done = threading.Event()

        def worker():
            try:
                with backend.lock:
                    for token in backend.generate(prompt_ids, max_tokens, sampler, eos):
                        q.put(("token", token))
                        if done.is_set():
                            break
                q.put(("end", None))
            except BaseException as exc:
                q.put(("error", exc))

        threading.Thread(target=worker, daemon=True).start()
        try:
            if chat:
                self._sse(event({"role": "assistant", "content": ""}, None))
            while True:
                try:
                    kind, value = q.get(timeout=KEEPALIVE_SECONDS)
                except queue.Empty:
                    self._sse(event({}, None))
                    continue
                if kind == "end":
                    break
                if kind == "error":
                    raise value
                token = value
                out.append(token)
                if token in eos:
                    finish = "stop"
                    done.set()
                    break
                raw = tok.decode(out, skip_special_tokens=False)
                parsed = parse_output(raw)
                calls = parsed["tool_calls"]
                if tool_done(raw):
                    finish = "tool_calls" if calls else "stop"
                    done.set()
                    break
                if repeat_len(raw):
                    finish = "stop"
                    sent_content = advance(sent_content, cut_repeat(parsed["content"]),
                                           "content")
                    done.set()
                    break
                if chat:
                    sent_reason = advance(sent_reason, parsed["reasoning"],
                                          "reasoning_content")
                text, cut = truncate(cut_repeat(parsed["content"]), stops)
                sent_content = advance(sent_content, text, "content")
                if cut:
                    finish = "stop"
                    done.set()
                    break
            if calls:
                finish = "tool_calls"
                for i, call in enumerate(calls):
                    self._sse(event({"tool_calls": [{
                        "index": i, "id": call["id"], "type": "function",
                        "function": {"name": call["function"]["name"],
                                     "arguments": call["function"]["arguments"]}}]}, None))
            self._sse(event({}, finish))
            if include_usage:
                self._sse({"id": cid, "object": obj, "created": created,
                           "model": model_id, "choices": [],
                           "usage": {"prompt_tokens": len(prompt_ids),
                                     "completion_tokens": len(out),
                                     "total_tokens": len(prompt_ids) + len(out)}})
            self._sse_data("[DONE]")
            self._sse_end()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:
            try:
                self._sse({"error": {"message": str(exc), "type": "server_error"}})
                self._sse_data("[DONE]")
                self._sse_end()
            except Exception:
                pass


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(backend, host="127.0.0.1", port=8080, quiet=False):
    """Make the HTTP server. Use port 0 for a free port."""
    httpd = _Server((host, port), _Handler)
    httpd.backend = backend
    httpd.quiet = quiet
    return httpd


def serve(backend, host="127.0.0.1", port=8080, quiet=False):
    """Run the HTTP server until the process stops."""
    httpd = make_server(backend, host, port, quiet)
    where = "http://%s:%d/v1" % (host, httpd.server_address[1])
    print("np-gemma server %s  model=%s" % (where, backend.model_id), flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
