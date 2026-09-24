import sys, time
sys.path.insert(0, '.')
import numpy as np
from np_gemma import Model, Tokenizer, Session
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.chat import render_chat
import np_gemma.ops as ops

g = GGUF('models/gemma-4-26B-qat-q4_0/gemma-4-26B_q4_0-it.gguf')
tok = Tokenizer.from_gguf(g)
cfg = Config({'text_config': g.text_config()})
m = Model(g, cfg).load_all(dtype='int4')
ids = tok.encode(render_chat(
    [{'role': 'user', 'content': open('README.md').read()[:2000] + '\nSummarise the above.'}],
    add_generation_prompt=True, enable_thinking=False))
s = Session(m, max_len=len(ids) + 50)
s.prefill(ids)
last = ids[-1]
xq = m.forward([last], cache=s.cache, start_pos=len(ids) - 1)
orig = ops.attn_ready
ops.attn_ready = lambda: False
xf = m.forward([last], cache=s.cache, start_pos=len(ids) - 1)
ops.attn_ready = orig
print('hidden |x| max %.3f mean %.3f' % (np.abs(xf).max(), np.abs(xf).mean()))
print('abs err max %.5f  rel-to-max %.4f%%  rel-to-mean %.3f%%'
      % (np.abs(xq - xf).max(), 100 * np.abs(xq - xf).max() / np.abs(xf).max(),
         100 * np.abs(xq - xf).mean() / np.abs(xf).mean()))

def time_decode(attn_on, steps=12):
    if not attn_on:
        ops.attn_ready = lambda: False
    st = Session(m, max_len=len(ids) + 200)
    st.prefill(ids)
    pos = len(ids)
    cur = last
    t0 = time.perf_counter()
    for _ in range(steps):
        x = m.forward([cur], cache=st.cache, start_pos=pos)
        cur = int(np.argmax(m.logits(x[-1:])[0]))
        pos += 1
    dt = time.perf_counter() - t0
    ops.attn_ready = orig
    return dt / steps * 1000

print('decode ms/token  q8-kv %.1f   float-kv %.1f' % (time_decode(True), time_decode(False)))
