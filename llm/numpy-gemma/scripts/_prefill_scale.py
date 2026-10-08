import sys, time
sys.path.insert(0, '.')
from np_gemma import Model, Tokenizer, KVCache
from np_gemma.config import Config
from np_gemma.gguf import GGUF
g = GGUF('models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf')
tok = Tokenizer.from_gguf(g)
cfg = Config({'text_config': g.text_config()})
m = Model(g, cfg).load_all(dtype='int4')
print('prefill_chunk', m.prefill_chunk, flush=True)
for n in (256, 512, 1024, 2048, 4096):
    ids = [1000] * n
    cache = KVCache(cfg, max_len=n + 8)
    t0 = time.perf_counter()
    m.prefill(ids, cache)
    dt = time.perf_counter() - t0
    print('%6d tokens  %8.2f s  %7.1f tok/s' % (n, dt, n / dt), flush=True)
