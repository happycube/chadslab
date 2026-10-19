"""The chat of the Qwen models (Qwen3.6, Qwen3.8): the tokenizer with the
chat template of the model (QwenTok), the think levels of a request
(thinking_of), and the parser of the answer (qwen_parse_output: the
reasoning, and the tool calls in the XML and the JSON forms of the
template). The server (server/) and the Qwen engine (engine.py) use them.
"""
from __future__ import annotations

import json
import os
import re
import uuid

from reclank.core.messages import MEDIA_TYPES
from reclank.models.qwen.tokenizer import ESCAPE, QwenTokenizer

MODEL = "models2/Qwen3.8-Flash-Next-NVFP4-GGUF/Qwen3.8-Flash-Next-NVFP4-bf16.gguf"


# The words that close a think part at --think-budget (then </think>).
THINK_CLOSE = "\n\nI have thought about this long enough; now I write the answer.\n</think>\n\n"
# The words when fewer than --wrap-left tokens of the context are left: in a
# think part (then </think>), and in the answer.
WRAP_THINK = "\n\nThe context is almost full, so I stop thinking and answer briefly now.\n</think>\n\n"
WRAP_ANSWER = "\n\n(The context is almost full, so I wrap up briefly now.)\n\n"




def _generation_ext():
    """The {% generation %} ... {% endgeneration %} tag of the templates of
    transformers (the mask of the assistant's tokens; LFM2): its body as it
    is."""
    from jinja2 import nodes
    from jinja2.ext import Extension

    class Generation(Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.Scope(body, lineno=lineno)
    return Generation


class QwenTok:
    """The tokenizer interface of reclank/server.py on QwenTokenizer, with
    the chat template of the model."""

    def __init__(self, path, template, text=None):
        import jinja2
        self.t = QwenTokenizer(path)
        self.stop_ids = list(self.t.stop_ids)
        # bos_token of tokenizer_config.json beside the tokenizer (the LFM2
        # template starts with it; the Qwen templates do not use it)
        self.bos_token = path.get("bos_token", "") if isinstance(path, dict) else ""
        cfg = None if isinstance(path, dict) else os.path.join(os.path.dirname(os.path.abspath(path)),
                                                               "tokenizer_config.json")
        if cfg is not None and os.path.exists(cfg):
            b = json.load(open(cfg)).get("bos_token")
            self.bos_token = (b.get("content", "") if isinstance(b, dict) else b) or ""
        env = jinja2.Environment(trim_blocks=True, lstrip_blocks=True,
                                 extensions=["jinja2.ext.loopcontrols", _generation_ext()])
        env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
        env.filters["items"] = lambda d: list(d.items()) if isinstance(d, dict) else []

        def raise_exception(msg):
            raise ValueError(msg)
        env.globals["raise_exception"] = raise_exception
        if text is None:
            text = open(template, encoding="utf-8").read()
        self.template = env.from_string(text)

    def encode(self, text):
        return self.t.encode(text)

    def decode(self, ids, skip_special_tokens=False):
        """The text of the answer for the parsers: the special tokens <think>
        and </think> as the channel of the Gemma output (reclank/models/gemma4/chat.py,
        parse_output), the others as their text. A marker that the model
        wrote as plain text (a quote of code: "<think>", "</tool_call>",
        "<|im_end|>") has ESCAPE in place of its "<", so no parser takes it;
        qwen_parse_output gives the "<" back."""
        t = self.t
        parts = []
        run = []

        def flush():
            if run:
                parts.append(PLAIN_MARK.sub(ESCAPE, t.decode(run)))
                run.clear()
        for i in ids:
            tok = t.id_to_token.get(int(i), "")
            if tok in t.special:
                flush()
                parts.append({"<think>": "<|channel>thought\n", "</think>": "<channel|>"}.get(tok, tok))
            else:
                run.append(i)
        flush()
        return "".join(parts)

    def apply_chat_template(self, messages, add_generation_prompt=True, thinking=False,
                            tools=None, empty_thought_block=True, effort=None, preserve=None):
        kw = {"reasoning_effort": effort} if thinking and effort else {}
        if preserve is not None:
            kw["preserve_thinking"] = bool(preserve)
        text = self.template.render(messages=messages, tools=tools or None,
                                    add_generation_prompt=add_generation_prompt,
                                    enable_thinking=bool(thinking), bos_token=self.bos_token, **kw)
        return text


# The levels of the request to those of the template (None: no think part).
EFFORT = {"none": None, "off": None, "disabled": None, "false": None,
          "minimal": "low", "low": "low", "medium": "medium",
          "high": "xhigh", "xhigh": "xhigh", "max": "xhigh", "on": "xhigh", "true": "xhigh",
          "enabled": "xhigh"}


def thinking_of(req, default):
    """The think level of a request (None: off), else default."""
    kw = req.get("chat_template_kwargs") or {}
    level = default
    for v in (kw.get("reasoning_effort"), (req.get("reasoning") or {}).get("effort")
              if isinstance(req.get("reasoning"), dict) else None, req.get("reasoning_effort")):
        if v is not None:
            level = EFFORT.get(str(v).lower(), level)
    for v in (kw.get("enable_thinking"), req.get("thinking")):
        if isinstance(v, dict):
            v = v.get("type")
        if v is None:
            continue
        on = EFFORT.get(str(v).lower(), "xhigh") is not None
        if not on:
            level = None
        elif level is None:
            level = "xhigh"
    return level


# The tool calls of the answer (the form of the template).


TOOLS = {}      # the tools of the request: name -> the properties of its parameters
# The "<" of a marker in plain text of the answer (QwenTok.decode): the tags
# of Qwen and the markers of the Gemma parser (SPECIAL of reclank/models/gemma4/chat.py).
MARK_TAIL = r"\|[^>\n]*?\|>|\|[a-z_]+>|[a-z_]+\|>|/?think>|/?tool_call>|/?tool_response>"
PLAIN_MARK = re.compile("<(?=" + MARK_TAIL + ")")
# ESCAPE stands for "<" only in front of a marker (a U+E000 of the answer
# itself stays): _plain gives the "<" back.
UNMARK = re.compile(ESCAPE + "(?=" + MARK_TAIL + ")")
CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.S)
FUNC = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
OPEN = "<tool_call>"


def _value(text, prop):
    """A parameter in the type of the schema: a string as it is, else JSON."""
    if (prop or {}).get("type") == "string":
        return text
    try:
        return json.loads(text)
    except ValueError:
        return text


def _plain(x):
    """x with the "<" of the markers that QwenTok.decode escaped."""
    if isinstance(x, str):
        return UNMARK.sub("<", x) if ESCAPE in x else x
    if isinstance(x, list):
        return [_plain(v) for v in x]
    if isinstance(x, dict):
        return {_plain(k): _plain(v) for k, v in x.items()}
    return x


def _call(name, args):
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": _plain(name), "arguments": json.dumps(_plain(args),
                                                                       ensure_ascii=False)}}


def qwen_parse_output(text):
    """parse_output of reclank/models/gemma4/chat.py (the reasoning), and the tool calls of
    Qwen: the XML form of the template (<function=...><parameter=...>, also
    with no </function>), and the JSON form ({"name": ..., "arguments": ...}).
    A tool call that is not complete is pending: a stream holds it."""
    from reclank.models.gemma4.chat import parse_output as gemma_parse_output
    p = gemma_parse_output(text)
    calls = []

    def call(m):
        inner = m.group(1)
        f = FUNC.search(inner)
        if f:
            name, props = f.group(1), TOOLS.get(f.group(1), {})
            args = {k: _value(v, props.get(k)) for k, v in PARAM.findall(f.group(2))}
            calls.append(_call(name, args))
            return ""
        try:
            obj = json.loads(inner.strip())
            args = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(args, str):
                args = json.loads(args)
            calls.append(_call(obj["name"], args))
            return ""
        except (ValueError, KeyError, TypeError, AttributeError):
            return m.group(0)                       # not a call: the text stays
    body = CALL.sub(call, p["content"])
    pending = ""
    i = body.find(OPEN)
    if i >= 0:
        body, pending = body[:i], body[i:]
    else:
        for k in range(min(len(body), len(OPEN) - 1), 0, -1):
            if OPEN.startswith(body[-k:]):
                body, pending = body[:-k], body[-k:]
                break
    return {"reasoning": _plain(p["reasoning"]), "content": _plain(body.strip()),
            "tool_calls": _plain(p["tool_calls"]) + calls, "pending": _plain(pending)}




# The tool calls of LFM2: a Python list of calls between two special tokens,
# <|tool_call_start|>[get_weather(city='Lyon'), search(q="x")]<|tool_call_end|>
# (or a JSON list when the system prompt asks for JSON).
LFM_OPEN, LFM_CLOSE = "<|tool_call_start|>", "<|tool_call_end|>"
LFM_CALL = re.compile(re.escape(LFM_OPEN) + r"(.*?)" + re.escape(LFM_CLOSE), re.S)
_PY_CONST = {"true": True, "false": False, "null": None}


def _py_value(node):
    """The value of an argument of a Pythonic call: a literal, or a bare
    true / false / null (JSON words)."""
    import ast
    if isinstance(node, ast.Name) and node.id in _PY_CONST:
        return _PY_CONST[node.id]
    return ast.literal_eval(node)


def _py_name(node):
    import ast
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _py_name(node.value) + "." + node.attr
    raise ValueError("not a function name")


def lfm_calls(inner):
    """The calls of the text between <|tool_call_start|> and
    <|tool_call_end|>: [(name, arguments)], or None when it is not a list of
    calls. A positional argument takes the name of the parameter of that
    place (TOOLS)."""
    import ast
    s = inner.strip()
    try:
        tree = ast.parse(s, mode="eval").body
        items = tree.elts if isinstance(tree, (ast.List, ast.Tuple)) else [tree]
        out = []
        for c in items:
            if not isinstance(c, ast.Call):
                raise ValueError("not a call")
            name = _py_name(c.func)
            names = list(TOOLS.get(name, {}))
            args = {}
            for j, a in enumerate(c.args):
                args[names[j] if j < len(names) else "arg%d" % j] = _py_value(a)
            for kw in c.keywords:
                if kw.arg is None:
                    raise ValueError("**kwargs")
                args[kw.arg] = _py_value(kw.value)
            out.append((name, args))
        return out
    except (SyntaxError, ValueError, TypeError):
        pass
    try:
        obj = json.loads(s)
        out = []
        for o in obj if isinstance(obj, list) else [obj]:
            a = o.get("arguments", o.get("parameters", {}))
            out.append((o["name"], json.loads(a) if isinstance(a, str) else a))
        return out
    except (ValueError, KeyError, TypeError, AttributeError):
        return None


def lfm_parse_output(text):
    """The tool calls of LFM2 (lfm_calls), then parse_output of
    reclank/models/gemma4/chat.py on the rest (the reasoning; it drops the
    special tokens of the text, so the calls come out first). A call that is
    not complete is pending: a stream holds it."""
    from reclank.models.gemma4.chat import parse_output as gemma_parse_output
    calls = []

    def call(m):
        got = lfm_calls(m.group(1))
        if got is None:
            return m.group(1)                       # not calls: the text stays
        calls.extend(_call(n, a) for n, a in got)
        return ""
    body = LFM_CALL.sub(call, str(text))
    pending = ""
    i = body.find(LFM_OPEN)
    if i >= 0:
        body, pending = body[:i], body[i:]
    else:
        for k in range(min(len(body), len(LFM_OPEN) - 1), 0, -1):
            if LFM_OPEN.startswith(body[-k:]):
                body, pending = body[:-k], body[-k:]
                break
    p = gemma_parse_output(body)
    return {"reasoning": _plain(p["reasoning"]), "content": _plain(p["content"]),
            "tool_calls": _plain(p["tool_calls"]) + calls, "pending": _plain(p["pending"] or pending)}


def _escaped(x, esc):
    """A copy of the messages (or the tools) x with esc on each string; the
    media parts keep their data."""
    if isinstance(x, str):
        return esc(x)
    if isinstance(x, list):
        return [_escaped(v, esc) for v in x]
    if isinstance(x, dict):
        if x.get("type") in MEDIA_TYPES:
            return x
        return {k: _escaped(v, esc) for k, v in x.items()}
    return x


def _message(m):
    """A message of the request for the template: the arguments of a tool
    call as a dict."""
    msg = dict(m)
    msg["role"] = m.get("role", "user")
    calls = []
    for c in m.get("tool_calls") or []:
        c = json.loads(json.dumps(c))
        f = c.get("function", c)
        if isinstance(f.get("arguments"), str):
            try:
                f["arguments"] = json.loads(f["arguments"]) if f["arguments"].strip() else {}
            except ValueError:
                f["arguments"] = {}
        calls.append(c)
    if calls:
        msg["tool_calls"] = calls
    return msg
