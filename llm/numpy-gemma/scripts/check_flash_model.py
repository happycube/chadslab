import os, sys, time
sys.path.insert(0, '.')
import numpy as np
from np_gemma import Model, Tokenizer, Session
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.chat import render_chat

g = GGUF('models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf')
tok = Tokenizer.from_gguf(g)
cfg = Config({'text_config': g.text_config()})
m = Model(g, cfg).load_all(dtype='int4')
ids = tok.encode(render_chat(
    [{'role': 'user', 'content': open('README.md').read()[:3000] + '\nSummarise.'}],
    add_generation_prompt=True, enable_thinking=False))
print('prompt tokens', len(ids), flush=True)

def run(flash):
    os.environ['NP_GEMMA_FLASH'] = '1' if flash else '0'
    s = Session(m, max_len=len(ids) + 20)
    t0 = time.perf_counter()
    s.prefill(ids)
    dt = time.perf_counter() - t0
    return m.logits(s._x[-1:])[0], dt

l0, t0 = run(False)
l1, t1 = run(True)
print('plain  %.2f s   flash  %.2f s   speedup %.2fx' % (t0, t1, t0 / t1), flush=True)
print('logit maxdiff %.5f  top1 plain %d flash %d %s'
      % (np.abs(l0 - l1).max(), int(np.argmax(l0)), int(np.argmax(l1)),
         'SAME' if np.argmax(l0) == np.argmax(l1) else 'DIFFERENT'), flush=True)

# prefill scaling: the sliding layers should gain the most
for n in (1024, 2048):
    sub = ids[:n]
    for flash in (False, True):
        os.environ['NP_GEMMA_FLASH'] = '1' if flash else '0'
        s = Session(m, max_len=n + 8)
        t0 = time.perf_counter()
        s.prefill(sub)
        print('n=%d flash=%d %.2f s' % (n, flash, time.perf_counter() - t0), flush=True)
