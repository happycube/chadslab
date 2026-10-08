import sys, json
sys.path.insert(0, '.')
import numpy as np
from np_gemma import Model, Tokenizer, Session
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.chat import render_chat, parse_output
from np_gemma.sampling import Sampler
from np_gemma.server import repeat_len

g = GGUF('models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf')
tok = Tokenizer.from_gguf(g)
cfg = Config({'text_config': g.text_config()})
m = Model(g, cfg).load_all(dtype='int4')

SETTINGS = [
    ('greedy', {'temperature': 0.0}),
    ('temp0.3', {'temperature': 0.3, 'top_p': 0.95, 'top_k': 40}),
    ('model1.', {'temperature': 1.0, 'top_p': 0.95, 'top_k': 64}),
]
TOOLS = [{'type': 'function', 'function': {
    'name': 'read_file', 'description': 'Read a file',
    'parameters': {'type': 'object', 'properties': {'path': {'type': 'string'}},
                   'required': ['path']}}}]

def gen(msgs, st, seed, mx, tools=None):
    text = render_chat(msgs, tools=tools, add_generation_prompt=True, enable_thinking=False)
    ids = tok.encode(text)
    s = Session(m, max_len=len(ids) + mx + 8)
    kw = dict(st)
    kw['seed'] = seed
    out = list(s.generate_stream(ids, max_new_tokens=mx, eos_ids=set(), sampler=Sampler(**kw)))
    return tok.decode(out, skip_special_tokens=False)

OPEN = ['List 40 one-line suggestions to improve a Python project README. One per line.',
        'Write a detailed 25-row table of the files in a project and what each does.']
print('== open-ended, 250 tokens: loop = a repeated tail ==', flush=True)
loops = {}
for name, st in SETTINGS:
    n_loop = 0
    for pi, p in enumerate(OPEN):
        for seed in range(3):
            t = gen([{'role': 'user', 'content': p}], st, seed, 250)
            rl = repeat_len(t)
            n_loop += 1 if rl else 0
            print('%-8s p%d seed%d len=%4d loop=%s' % (name, pi, seed, len(t), 'YES' if rl else 'no'), flush=True)
    loops[name] = n_loop
print('loop counts:', json.dumps(loops), flush=True)

print('== tool call, 96 tokens, 5 seeds ==', flush=True)
msgs = [{'role': 'system', 'content': 'You are a coding agent. Use the tool.'},
        {'role': 'user', 'content': 'Read the file /etc/hosts. Use read_file.'}]
for name, st in SETTINGS:
    good = 0
    for seed in range(5):
        t = gen(msgs, st, seed, 96, TOOLS)
        p = parse_output(t)
        c = p['tool_calls']
        ok = (len(c) == 1 and c[0]['function']['name'] == 'read_file'
              and 'etc/hosts' in c[0]['function']['arguments'])
        good += 1 if ok else 0
        if not ok:
            print('   miss:', repr(t[:110]), flush=True)
    print('%-8s correct tool %d/5' % (name, good), flush=True)
