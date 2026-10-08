import sys
sys.path.insert(0, '.')
import numpy as np
from np_gemma import Model, Tokenizer, Session, KVCache
from np_gemma.config import Config
from np_gemma.gguf import GGUF
from np_gemma.chat import render_chat
import np_gemma.ops as ops

g = GGUF('models2/gemma-4-26B-unsloth-UD-Q4_K_XL/gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf')
tok = Tokenizer.from_gguf(g)
cfg = Config({'text_config': g.text_config()})
m = Model(g, cfg).load_all(dtype='int4')
ids = tok.encode(render_chat(
    [{'role': 'user', 'content': open('README.md').read()[:2000] + '\nSummarise.'}],
    add_generation_prompt=True, enable_thinking=False))
s = Session(m, max_len=len(ids) + 20)
s.prefill(ids)

cap = {'cur': -1}
orig_attn, orig_read = ops.attn_decode, KVCache.read_qc
def pr(self, layer, end):
    cap['cur'] = layer
    cap.setdefault('n', {})[layer] = end - self.base[layer]
    return orig_read(self, layer, end)
def pa(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, n, *a, **k):
    L = cap['cur']
    cap.setdefault('q', {})[L] = np.array(q)
    cap.setdefault('meta', {})[L] = (q_heads, kv_heads, head_dim, n, 1)
    return orig_attn(q, kq, ks, vq, vs, q_heads, kv_heads, head_dim, n, *a, **k)
KVCache.read_qc = pr
ops.attn_decode = pa
m.forward(ids[-1:], cache=s.cache, start_pos=len(ids) - 1)
KVCache.read_qc = orig_read
ops.attn_decode = orig_attn

def softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()

def q8_block(x):
    sh = x.shape
    x = x.reshape(*sh[:-1], -1, 32)
    sc = np.max(np.abs(x), axis=-1, keepdims=True)
    sc = np.where(sc > 0, sc / 127.0, 1e-12)
    return (np.rint(x / sc).clip(-127, 127) * sc).reshape(*sh[:-1], sh[-1])

def q8_channel(x):
    sc = np.max(np.abs(x), axis=0, keepdims=True)
    sc = np.where(sc > 0, sc / 127.0, 1e-12)
    return np.rint(x / sc).clip(-127, 127) * sc

allk = []
for layer in sorted(cap['q']):
    q_heads, kv_heads, hd, n, amin = cap['meta'][layer]
    n_rep = q_heads // kv_heads
    K, V, _ = s.cache.read(layer, s.cache.base[layer] + n)   # dequantized rows
    K, V = K.astype(np.float64), V.astype(np.float64)
    q = cap['q'][layer][0].astype(np.float64)
    k = K[:, 0, :]
    v = V[:, 0, :]
    ref = softmax(k @ q) @ v
    rows = []
    def run(kk, vv, qq, label):
        out = softmax(kk @ qq) @ vv
        d = np.abs(out - ref)
        rows.append((label, d.max(), 100 * d.max() / np.abs(ref).max(), d.mean()))
    kk, vv, qq = q8_block(k), q8_block(v), q8_block(q.reshape(1, -1))[0]
    run(kk, vv, qq, 'q8 q + q8 k + q8 v (ours)')
    run(kk, vv, q,      'q8 k + q8 v, query float')
    run(kk, v,  q,      'q8 k only')
    run(k,  vv, q,      'q8 v only')
    run(k,  v,  qq,     'q8 query only')
    run(q8_channel(k), vv, q, 'k per-channel + v per-token')
    run(q8_channel(k), v, q,  'k per-channel only')
    d = {label: pct for label, mx, pct, mn in rows}
    o = d['q8 q + q8 k + q8 v (ours)']
    kk_ = d['q8 k only']
    vv_ = d['q8 v only']
    kc = d['k per-channel only']
    qq_ = d['q8 query only']
    allk.append((o, kk_, vv_, kc, qq_, d['q8 k + q8 v, query float']))
    print('L%-2d hd=%-3d kvh=%-2d ours=%.3f%%  k=%.3f%%  v=%.3f%%  k-perchan=%.3f%%  q=%.3f%%'
          % (layer, hd, kv_heads, o, kk_, vv_, kc, qq_))

a = np.array(allk)
print('')
print('mean over layers: ours=%.3f%%  k-only=%.3f%%  v-only=%.3f%%  k-perchan-only=%.3f%%  q-only=%.3f%%  k+v noq=%.3f%%'
      % tuple(a.mean(axis=0)))
print('worst layer:      ours=%.3f%%' % a[:,0].max())
