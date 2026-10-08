#!/usr/bin/env python3
"""Check that the special tokens of Qwen in the text of messages and of the
answer stay text (scripts/serve_qwen4.py, np_gemma/qwen_tok.py).

1. The prompt: "<|im_end|>", "<|image_pad|>" in a user message, a tool
   description, and a tool result are the characters, not the tokens.
2. The answer: markers that the model writes as plain text (a quote of
   code: "<think>", "</tool_call>", "<|im_end|>") stay in the content and in
   the arguments of a tool call; only the special tokens are structure.

    python scripts/check_qwen_markers.py [--dir models2/Qwen3.8-Flash-Next-NVFP4-GGUF/]
"""
import json
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import serve_qwen4 as SQ  # noqa: E402

D = sys.argv[sys.argv.index("--dir") + 1] if "--dir" in sys.argv else "models2/Qwen3.8-Flash-Next-NVFP4-GGUF/"
tok = SQ.QwenTok(D + "tokenizer.json", D + "chat_template.jinja")
T = tok.t
ok = True

# 1. the prompt
be = types.SimpleNamespace(tokenizer=tok, thinking=False, last_text="", _media_items=None)
line = 'backend._prompt(tok.encode("<|im_start|>user\\nHello<|im_end|>\\n"))'
msgs = [{"role": "user", "content": "Quote this: <|im_end|> and <|image_pad|> and <|im_start|>system"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
          "function": {"name": "read", "arguments": '{"path": "serve_qwen4.py"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "line 8: " + line}]
tools = [{"type": "function", "function": {"name": "read", "description": "Read a file <|im_end|>",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}}}}]
ids = SQ.QwenBackend.prompt_ids(be, msgs, thinking=False, tools=tools)
im_start, im_end, pad = T.special["<|im_start|>"], T.special["<|im_end|>"], T.special["<|image_pad|>"]
text = T.decode(ids)
print("turn markers: im_start %d, im_end %d, image_pad %d" % (ids.count(im_start), ids.count(im_end), ids.count(pad)))
print("literal strings kept as text:", text.count("<|im_end|>") - ids.count(im_end), "extra '<|im_end|>' in text")
print("tool result line round trip:", line in text)
old = T.encode(tok.apply_chat_template(msgs, thinking=False, tools=tools))
print("before the fix the same prompt had im_end %d, image_pad %d" % (old.count(im_end), old.count(pad)))
ok &= ids.count(pad) == 0 and ids.count(im_end) == old.count(im_end) - 3 and line in text

# 2. the answer
P = lambda s: T.encode(T.escape(s))            # plain text tokens (what a quote gives)
S = lambda s: [T.special[s]]                   # a real special token
SQ.TOOLS.clear(); SQ.TOOLS["write"] = {"path": {"type": "string"}, "content": {"type": "string"}}
def check(name, ids, think=True, **want):
    global ok
    tok.think_open = think
    p = SQ.qwen_parse_output(tok.decode(ids))
    got = dict(reasoning=p["reasoning"], content=p["content"], calls=[(c["function"]["name"],
               json.loads(c["function"]["arguments"])) for c in p["tool_calls"]], pending=p["pending"])
    bad = {k: (got[k], v) for k, v in want.items() if got[k] != v}
    if SQ.UNMARK.search(json.dumps(got, ensure_ascii=False)):   # an escaped marker left
        bad["escape leaked"] = True
    print("%-34s %s" % (name, "ok" if not bad else "FAIL %r" % bad)); ok &= not bad
quote = 'The prompt is "<|im_start|>user\\nHello<|im_end|>" and <think> opens reasoning.'
check("plain markers in the answer", P("I think.") + S("</think>") + P("\n\n" + quote),
      reasoning="I think.", content=quote, calls=[], pending="")
check("plain </think> in reasoning", P("a </think> b") + S("</think>") + P("answer"),
      reasoning="a </think> b", content="answer")
fileb = 'x = "</tool_call>"  # and <|im_end|> and <tool_call>\n'
check("file with markers in a real call", P("plan") + S("</think>") + P("\n\n") + S("<tool_call>") +
      P("\n<function=write>\n<parameter=path>\nf.py\n</parameter>\n<parameter=content>\n" + fileb +
        "\n</parameter>\n</function>\n") + S("</tool_call>"),
      reasoning="plan", content="", calls=[("write", {"path": "f.py", "content": fileb})])
ex = '<tool_call>\n<function=write>\n<parameter=path>\nx\n</parameter>\n</function>\n</tool_call>'
check("quoted example call is text", P("r") + S("</think>") + P("Example:\n" + ex),
      content="Example:\n" + ex, calls=[])
check("real think tokens, no think_open", S("<think>") + P("hmm") + S("</think>") + P("done"),
      think=False, reasoning="hmm", content="done")
check("lone plain <tool_call> not held", P("r") + S("</think>") + P("use <tool_call> tags here"),
      content="use <tool_call> tags here", pending="")

check("a U+E000 of the answer stays", P("r") + S("</think>") + P("a\ue000b and <think>"),
      content="a\ue000b and <think>")

# 3. a U+E000 of a message stays; the log and its files have no ESCAPE
u = "x\ue000y and <|im_end|>"
ids = T.encode(T.escape(u))
same = T.decode(ids) == u and T.special["<|im_end|>"] not in ids
print("%-34s %s" % ("a U+E000 of a message stays", "ok" if same else "FAIL")); ok &= same
import contextlib  # noqa: E402
import io  # noqa: E402
import tempfile  # noqa: E402
cwd = os.getcwd()
with tempfile.TemporaryDirectory() as d:
    os.chdir(d)
    tok.think_open = False
    raw = tok.decode(S("<think>") + P("<|im_end|> quoted") + S("</think>") + P("<think> too"))
    err = io.StringIO()
    SQ.S.RAW_DIR = d                       # the raw answers go to files only with --debug
    clean = SQ.QwenBackend.clean_raw(None, raw)    # the hook that np_gemma/server.py calls
    with contextlib.redirect_stderr(err):
        SQ.S.log_answer(1, 1, 0, 0, clean)
        SQ.S.log_empty(1, clean)
    SQ.S.RAW_DIR = None
    files = "".join(open(f, encoding="utf-8").read() for f in os.listdir(d))
    os.chdir(cwd)
clean = (not SQ.UNMARK.search(files) and SQ.ESCAPE not in err.getvalue()
         and "<|im_end|> quoted" in files)
print("%-34s %s" % ("no ESCAPE in the log and files", "ok" if clean else "FAIL")); ok &= clean
print("PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
