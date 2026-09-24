"""Compare render_chat with the canonical Jinja template.

    python scripts/check_chat_template.py

The script loads np_gemma/chat_template.jinja with Jinja2 and renders a set of
message lists. It then renders the same lists with np_gemma.chat.render_chat.
The two texts must agree.
"""
from __future__ import annotations

import os
import sys

from jinja2.sandbox import ImmutableSandboxedEnvironment

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma.chat import render_chat  # noqa: E402

TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "np_gemma", "chat_template.jinja")

TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city"},
                "unit": {"type": "string", "enum": ["c", "f"]},
                "days": {"type": "integer", "description": "How many"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["location"],
        },
    },
}

CASES = [
    ("user only", {
        "messages": [{"role": "user", "content": "Hello"}]}),
    ("system and user", {
        "messages": [{"role": "system", "content": "Be brief"},
                     {"role": "user", "content": "Hello"}]}),
    ("thinking", {
        "messages": [{"role": "user", "content": "Hello"}],
        "enable_thinking": True}),
    ("tools only", {
        "messages": [{"role": "user", "content": "Weather in Paris"}],
        "tools": [TOOL]}),
    ("tools and system", {
        "messages": [{"role": "system", "content": "Use a tool"},
                     {"role": "user", "content": "Weather in Paris"}],
        "tools": [TOOL]}),
    ("assistant call and tool result", {
        "messages": [
            {"role": "user", "content": "Weather in Paris"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather",
                              "arguments": {"location": "Paris", "unit": "c"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
             "content": "sunny"},
        ],
        "tools": [TOOL]}),
    ("tool result then thinking", {
        "messages": [
            {"role": "user", "content": "Weather in Paris"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function",
                 "function": {"name": "get_weather", "arguments": {"location": "Paris"}}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
        ],
        "tools": [TOOL], "enable_thinking": True}),
    ("reasoning content", {
        "messages": [
            {"role": "user", "content": "Think"},
            {"role": "assistant", "reasoning_content": "I reason",
             "content": "The answer"},
            {"role": "user", "content": "Again"}]}),
    ("reasoning preserve with calls", {
        "messages": [
            {"role": "user", "content": "Think"},
            {"role": "assistant", "reasoning_content": "I reason",
             "tool_calls": [{"id": "call_9", "type": "function",
                             "function": {"name": "get_weather",
                                          "arguments": {"location": "Rome"}}}]},
            {"role": "tool", "tool_call_id": "call_9", "content": "rain"},
        ],
        "tools": [TOOL], "preserve_thinking": True}),
    ("two assistant turns", {
        "messages": [
            {"role": "user", "content": "One"},
            {"role": "assistant", "content": "First"},
            {"role": "assistant", "content": "Second"},
            {"role": "user", "content": "Two"}]}),
    ("content parts", {
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "Look"}, {"type": "image"}]}]}),
    ("no generation prompt", {
        "messages": [{"role": "user", "content": "Hello"}],
        "add_generation_prompt": False}),
]


def main():
    with open(TEMPLATE_PATH) as fh:
        text = fh.read()
    env = ImmutableSandboxedEnvironment()
    env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
    env.globals["strftime_now"] = lambda fmt: ""
    tpl = env.from_string(text)
    ok = True
    for name, case in CASES:
        want = tpl.render(messages=case["messages"], tools=case.get("tools"),
                          add_generation_prompt=case.get("add_generation_prompt", True),
                          enable_thinking=case.get("enable_thinking", False),
                          preserve_thinking=case.get("preserve_thinking", False),
                          bos_token="<bos>")
        got = render_chat(messages=case["messages"], tools=case.get("tools"),
                          add_generation_prompt=case.get("add_generation_prompt", True),
                          enable_thinking=case.get("enable_thinking", False),
                          preserve_thinking=case.get("preserve_thinking", False),
                          bos_token="<bos>")
        same = want == got
        ok &= same
        print("%-30s %s" % (name, "ok" if same else "DIFF"))
        if not same:
            for i in range(min(len(want), len(got))):
                if want[i] != got[i]:
                    print("  first diff at %d" % i)
                    print("  want %r" % want[max(0, i - 40):i + 60])
                    print("  got  %r" % got[max(0, i - 40):i + 60])
                    break
            else:
                print("  length want=%d got=%d" % (len(want), len(got)))
                print("  want tail %r" % want[len(got):][:120])
                print("  got tail  %r" % got[len(want):][:120])
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
