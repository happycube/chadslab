import sys
sys.path.insert(0, '.')
from np_gemma import Model, Tokenizer, Session
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.chat import render_chat, parse_output
from np_gemma.sampling import Sampler

g = GGUF('models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf')
tok = Tokenizer.from_gguf(g)
cfg = Config({'text_config': g.text_config()})
m = Model(g, cfg).load_all(dtype='int4')
msgs = [{'role': 'user', 'content': 'What is 17 times 24?'}]
text = render_chat(msgs, add_generation_prompt=True, enable_thinking=True)
print('PROMPT TAIL', repr(text[-70:]))
ids = tok.encode(text)
s = Session(m, max_len=len(ids) + 200)
out = list(s.generate_stream(ids, max_new_tokens=60, eos_ids=set(),
                             sampler=Sampler(temperature=0.0)))
print('first 14 token ids:', out[:14])
for i in out[:14]:
    print('  %6d %r' % (i, tok.decode([i], skip_special_tokens=False)))
raw = tok.decode(out, skip_special_tokens=False)
print('RAW HEAD', repr(raw[:260]))
p = parse_output(raw)
print('reasoning', repr(p['reasoning'][:120]))
print('content  ', repr(p['content'][:120]))
