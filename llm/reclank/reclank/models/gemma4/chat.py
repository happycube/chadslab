"""Render the canonical Gemma 4 chat template and read the model output.

The official template is the Jinja file chat_template.jinja. This module gives
the same text as that template without a Jinja engine. It also reads the
thought channel, the answer, and the tool calls from the model output.

The tokens of the format are:

    <|turn>role\n ... <turn|>\n      one turn
    <|tool> declaration <tool|>      one tool declaration
    <|tool_call>call:name{...}<tool_call|>
    <|tool_response>response:name{...}<tool_response|>
    <|channel>thought\n ... <channel|>   the reasoning
    <|"|>                            a string quote
"""
from __future__ import annotations

import json
import re
import uuid

SPECIAL = re.compile(r"<\|[^>]*?\|>|<\|[a-z_]+>|<[a-z_]+\|>")


# ---------- the template -------------------------------------------------

def format_argument(argument, escape_keys=True):
    """Write one argument in the template format."""
    if argument is None:
        return "null"
    if isinstance(argument, bool):
        return "true" if argument else "false"
    if isinstance(argument, str):
        return '<|"|>' + argument + '<|"|>'
    if isinstance(argument, dict):
        out = ["{"]
        for i, key in enumerate(sorted(argument)):
            if i:
                out.append(",")
            out.append('<|"|>' + str(key) + '<|"|>' if escape_keys else str(key))
            out.append(":")
            out.append(format_argument(argument[key], escape_keys))
        out.append("}")
        return "".join(out)
    if isinstance(argument, (list, tuple)):
        return "[" + ",".join(format_argument(v, escape_keys) for v in argument) + "]"
    return str(argument)


def _standard(value):
    return str(value or "").upper()


def format_parameters(properties, required=None, filter_keys=False):
    """Write the parameter part of a tool declaration."""
    skip = {"description", "type", "properties", "required", "nullable"}
    out = []
    first = False
    for key in sorted(properties):
        value = properties[key]
        if not isinstance(value, dict):
            value = {}
        if filter_keys and key in skip:
            continue
        if first:
            out.append(",")
        first = True
        out.append(str(key) + ":{")
        comma = False
        if value.get("description"):
            out.append('description:<|"|>' + str(value["description"]) + '<|"|>')
            comma = True
        vtype = _standard(value.get("type"))
        if vtype == "STRING":
            if value.get("enum"):
                out.append("," if comma else "")
                comma = True
                out.append("enum:" + format_argument(value["enum"]))
        elif vtype == "ARRAY":
            items = value.get("items")
            if isinstance(items, dict) and items:
                out.append("," if comma else "")
                comma = True
                out.append("items:{")
                it_first = False
                for ikey in sorted(items):
                    ival = items[ikey]
                    if ival is None:
                        continue
                    if it_first:
                        out.append(",")
                    it_first = True
                    if ikey == "properties":
                        out.append("properties:{")
                        if isinstance(ival, dict):
                            out.append(format_parameters(ival, items.get("required") or []))
                        out.append("}")
                    elif ikey == "required":
                        out.append("required:[" + ",".join(
                            '<|"|>' + str(x) + '<|"|>' for x in ival) + "]")
                    elif ikey == "type":
                        if isinstance(ival, str):
                            out.append("type:" + format_argument(ival.upper()))
                        else:
                            out.append("type:" + format_argument(
                                [str(x).upper() for x in ival]))
                    else:
                        out.append(str(ikey) + ":" + format_argument(ival))
                out.append("}")
        if value.get("nullable"):
            out.append("," if comma else "")
            comma = True
            out.append("nullable:true")
        if vtype == "OBJECT":
            props = value.get("properties")
            if isinstance(props, dict):
                out.append("," if comma else "")
                comma = True
                out.append("properties:{")
                out.append(format_parameters(props, value.get("required") or []))
                out.append("}")
            elif isinstance(value, dict):
                out.append("," if comma else "")
                comma = True
                out.append("properties:{")
                out.append(format_parameters(value, value.get("required") or [],
                                             filter_keys=True))
                out.append("}")
            if value.get("required"):
                out.append("," if comma else "")
                comma = True
                out.append("required:[" + ",".join(
                    '<|"|>' + str(x) + '<|"|>' for x in value["required"]) + "]")
        out.append("," if comma else "")
        out.append('type:<|"|>' + vtype + '<|"|>}')
    return "".join(out)


def format_function_declaration(tool):
    """Write one tool declaration for the system turn."""
    fn = tool.get("function", tool)
    out = ["declaration:" + str(fn.get("name", ""))
           + '{description:<|"|>' + str(fn.get("description") or "") + '<|"|>']
    params = fn.get("parameters")
    if params:
        out.append(",parameters:{")
        if params.get("properties"):
            out.append("properties:{")
            out.append(format_parameters(params["properties"], params.get("required") or []))
            out.append("},")
        if params.get("required"):
            out.append("required:[" + ",".join(
                '<|"|>' + str(x) + '<|"|>' for x in params["required"]) + "],")
        if params.get("type"):
            out.append('type:<|"|>' + _standard(params["type"]) + '<|"|>}')
    if "response" in fn:
        rd = fn.get("response") or {}
        out.append(",response:{")
        if rd.get("description"):
            out.append('description:<|"|>' + str(rd["description"]) + '<|"|>,')
        if _standard(rd.get("type")) == "OBJECT":
            out.append('type:<|"|>' + _standard(rd["type"]) + '<|"|>}')
    out.append("}")
    return "".join(out)


def format_tool_response_block(name, response):
    """Write one tool response."""
    out = ["<|tool_response>"]
    if isinstance(response, dict):
        out.append("response:" + str(name) + "{")
        for i, key in enumerate(sorted(response)):
            if i:
                out.append(",")
            out.append(str(key) + ":" + format_argument(response[key], escape_keys=False))
        out.append("}")
    else:
        out.append("response:" + str(name) + "{value:"
                   + format_argument(response, escape_keys=False) + "}")
    out.append("<tool_response|>")
    return "".join(out)


def strip_thinking(text):
    """Remove the thought channels from the text."""
    out = []
    for part in str(text).split("<channel|>"):
        if "<|channel>" in part:
            out.append(part.split("<|channel>")[0])
        else:
            out.append(part)
    return "".join(out).strip()


def _text_of(content):
    """Return the trimmed text of a content value."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, (list, tuple)):
        out = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                out.append(str(item.get("text") or "").strip())
        return "".join(out)
    return ""


def _note(media, part):
    """Add a media part to the list media (if a list): the parts in the order
    of their placeholders in the prompt."""
    if media is not None:
        media.append(part)


def render_chat(messages, tools=None, add_generation_prompt=True,
                enable_thinking=False, preserve_thinking=False,
                bos_token="<bos>", empty_thought_block=True, media=None):
    """Return the chat prompt text.

    messages follows the OpenAI shape. A tool call takes an arguments mapping
    or a JSON string. A tool result uses the role "tool" and the field
    tool_call_id. The result matches the canonical Jinja template. Give a
    list as media to get each image, audio, or video part, in the order of
    its placeholder in the text.

    Set empty_thought_block to False for the E2B and E4B models. When thinking
    is off, the 12B, the 26B, and the 31B start the answer with an empty
    `<|channel>thought\\n<channel|>` block. The E2B and E4B models do not: they
    go straight to the answer. The model card states the difference under
    "Disabled Thinking Behavior".
    """
    messages = list(messages or [])
    loop = messages
    out = [bos_token]
    last_type = None
    prev_role = None
    if enable_thinking or tools or (messages and messages[0].get("role") in ("system", "developer")):
        out.append("<|turn>system\n")
        if enable_thinking:
            out.append("<|think|>\n")
            last_type = "think"
        if messages and messages[0].get("role") in ("system", "developer"):
            content = messages[0].get("content")
            if isinstance(content, str):
                out.append(content.strip())
            elif isinstance(content, (list, tuple)):
                for item in content:
                    if isinstance(item, dict) and item.get("text") is not None:
                        out.append(str(item["text"]).strip() + " ")
            loop = messages[1:]
        for tool in (tools or []):
            out.append("<|tool>" + format_function_declaration(tool).strip() + "<tool|>")
        if tools:
            last_type = "tool"
        out.append("<turn|>\n")

    last_user = -1
    for i, m in enumerate(loop):
        if m.get("role") == "user":
            last_user = i

    for idx, message in enumerate(loop):
        role_in = message.get("role")
        if role_in == "tool":
            continue
        last_type = None
        role = "model" if role_in == "assistant" else role_in
        if not (role == "model" and prev_role == "assistant"):
            out.append("<|turn>" + str(role) + "\n")
        thinking = message.get("reasoning") or message.get("reasoning_content")
        if thinking and (idx > last_user or (preserve_thinking and message.get("tool_calls"))):
            out.append("<|channel>thought\n" + str(thinking) + "\n<channel|>")
        calls = message.get("tool_calls") or []
        for tc in calls:
            fn = tc.get("function", {})
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            out.append("<|tool_call>call:" + str(fn.get("name", "")) + "{")
            if isinstance(args, dict):
                for i, key in enumerate(sorted(args)):
                    if i:
                        out.append(",")
                    out.append(str(key) + ":" + format_argument(args[key], escape_keys=False))
            elif args is not None:
                raise ValueError("tool_calls[].function.arguments must be a JSON object")
            out.append("}<tool_call|>")
        if calls:
            last_type = "tool_call"

        tr_flag = False
        if message.get("tool_responses"):
            for tr in message["tool_responses"]:
                out.append(format_tool_response_block(tr.get("name", "unknown"),
                                                      tr.get("response", "")))
                tr_flag = True
                last_type = "tool_response"
        elif calls:
            stopped = False
            for k in range(idx + 1, len(loop)):
                if stopped:
                    break
                follow = loop[k]
                if follow.get("role") != "tool":
                    stopped = True
                    continue
                name = follow.get("name") or "unknown"
                for tc in calls:
                    if tc.get("id") == follow.get("tool_call_id"):
                        name = tc.get("function", {}).get("name", name)
                body = follow.get("content")
                if isinstance(body, str):
                    out.append(format_tool_response_block(name, body))
                elif isinstance(body, (list, tuple)):
                    out.append(format_tool_response_block(name, _text_of(body)))
                    for part in body:
                        if isinstance(part, dict):
                            if part.get("type") in ("image", "image_url"):
                                out.append("<|image|>")
                                _note(media, part)
                            elif part.get("type") in ("audio", "input_audio"):
                                out.append("<|audio|>")
                                _note(media, part)
                            elif part.get("type") in ("video", "video_url"):
                                out.append("<|video|>")
                                _note(media, part)
                else:
                    out.append(format_tool_response_block(name, body))
                tr_flag = True
                last_type = "tool_response"

        content = message.get("content")
        if isinstance(content, str):
            captured = strip_thinking(content) if role == "model" else content.strip()
        else:
            parts = []
            for item in (content or []):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    t = str(item.get("text") or "")
                    parts.append(strip_thinking(t) if role == "model" else t.strip())
                elif item.get("type") in ("image", "image_url"):
                    parts.append("<|image|>")
                    _note(media, item)
                elif item.get("type") in ("audio", "input_audio"):
                    parts.append("<|audio|>")
                    _note(media, item)
                elif item.get("type") in ("video", "video_url"):
                    parts.append("<|video|>")
                    _note(media, item)
            captured = "".join(parts)
        out.append(captured)
        has_content = len(captured.strip()) > 0

        next_role = None
        for j in range(idx + 1, len(loop)):
            if loop[j].get("role") != "tool":
                next_role = loop[j].get("role")
                break
        continues = (role == "model" and next_role == "assistant"
                     and (not calls or tr_flag))
        if last_type == "tool_call" and not tr_flag:
            out.append("<|tool_response>")
        elif continues:
            pass
        elif not (tr_flag and not has_content and next_role is None):
            out.append("<turn|>\n")
        prev_role = role_in

    if add_generation_prompt:
        if last_type not in ("tool_response", "tool_call"):
            out.append("<|turn>model\n")
            if not enable_thinking and empty_thought_block:
                out.append("<|channel>thought\n<channel|>")
        elif last_type == "tool_response" and enable_thinking:
            out.append("<|channel>thought\n")
    return "".join(out)


# ---------- the model output ---------------------------------------------

def _scalar(text):
    if text == "true":
        return True
    if text == "false":
        return False
    if text == "null":
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    return text


class _Args:
    """Read the argument object of a tool call."""

    def __init__(self, text):
        self.s = text
        self.i = 0

    def _ws(self):
        while self.i < len(self.s) and self.s[self.i] in " \t\r\n":
            self.i += 1

    def parse(self):
        self._ws()
        return self.value()

    def _quoted(self):
        self.i += 5
        j = self.s.find('<|"|>', self.i)
        if j < 0:
            j = len(self.s)
        v = self.s[self.i:j]
        self.i = min(j + 5, len(self.s))
        return v

    def value(self):
        self._ws()
        if self.s.startswith('<|"|>', self.i):
            return self._quoted()
        c = self.s[self.i:self.i + 1]
        if c == "{":
            return self.obj()
        if c == "[":
            return self.arr()
        j = self.i
        while j < len(self.s) and self.s[j] not in ",}]:":
            j += 1
        text = self.s[self.i:j].strip()
        self.i = j
        return _scalar(text)

    def key(self):
        self._ws()
        if self.s.startswith('<|"|>', self.i):
            return self._quoted()
        j = self.i
        while j < len(self.s) and self.s[j] != ":":
            j += 1
        key = self.s[self.i:j].strip()
        self.i = j
        return key

    def obj(self):
        self.i += 1
        out = {}
        self._ws()
        if self.s[self.i:self.i + 1] == "}":
            self.i += 1
            return out
        while self.i < len(self.s):
            key = self.key()
            self._ws()
            if self.s[self.i:self.i + 1] == ":":
                self.i += 1
            out[key] = self.value()
            self._ws()
            if self.s[self.i:self.i + 1] == ",":
                self.i += 1
                continue
            if self.s[self.i:self.i + 1] == "}":
                self.i += 1
            break
        return out

    def arr(self):
        self.i += 1
        out = []
        self._ws()
        if self.s[self.i:self.i + 1] == "]":
            self.i += 1
            return out
        while self.i < len(self.s):
            out.append(self.value())
            self._ws()
            if self.s[self.i:self.i + 1] == ",":
                self.i += 1
                continue
            if self.s[self.i:self.i + 1] == "]":
                self.i += 1
            break
        return out


def parse_tool_call(body):
    """Read one tool call. Return the OpenAI shape, or None."""
    body = body.strip()
    if body.startswith("call:"):
        body = body[5:]
    name, sep, rest = body.partition("{")
    name = name.strip()
    if not name:
        return None
    text = "{" + rest if sep else "{}"
    try:
        args = _Args(text).parse()
    except Exception:
        args = {}
    if not isinstance(args, dict):
        args = {}
    return {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
            "function": {"name": name,
                         "arguments": json.dumps(args, ensure_ascii=False)}}


def parse_output(text):
    """Split the model output into the reasoning, the answer, and the calls.

    Return a dict with the keys reasoning, content, tool_calls, and pending.
    pending holds a tool call that the text has not closed yet. A stream must
    hold it back.
    """
    body = str(text)
    # A model that waits for a tool result repeats the empty thought channel.
    # Remove every channel, not only the first. The label of a channel (the
    # text after <|channel> to the newline: "thought") is not read: every
    # channel is reasoning. That fails closed (an unknown channel is never
    # shown as the answer); a model with a channel for the answer needs a
    # rule here.
    parts = []
    while True:
        start = body.find("<|channel>")
        if start < 0:
            break
        lbl = start + len("<|channel>")
        nl = body.find("\n", lbl)
        end = body.find("<channel|>", lbl)
        if nl < 0:
            # The channel name is still arriving. Hold the text back. A
            # partial name must never reach the caller as the reasoning,
            # because the caller emits a delta and cannot take it back.
            parts.append("")
            body = body[:start]
            break
        if end >= 0 and end < nl:
            # An empty channel, closed before any newline.
            parts.append("")
            body = body[:start] + body[end + len("<channel|>"):]
            continue
        if end < 0:
            parts.append(body[nl + 1:])
            body = body[:start]
            break
        parts.append(body[nl + 1:end])
        body = body[:start] + body[end + len("<channel|>"):]
    reasoning = "\n".join(p.strip() for p in parts if p.strip())
    calls = []
    pending = ""
    while True:
        a = body.find("<|tool_call>")
        if a < 0:
            break
        b = body.find("<tool_call|>", a)
        if b < 0:
            pending = body[a:]
            body = body[:a]
            break
        chunk = body[a + len("<|tool_call>"):b]
        body = body[:a] + body[b + len("<tool_call|>"):]
        call = parse_tool_call(chunk)
        if call:
            calls.append(call)
    content = SPECIAL.sub("", body).strip()
    return {"reasoning": reasoning.strip(), "content": content,
            "tool_calls": calls, "pending": pending}
