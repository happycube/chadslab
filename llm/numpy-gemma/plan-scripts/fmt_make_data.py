"""Test data for the BF12 / RQ10 / RQ12 test kernels (plan-scripts/BF12_PLAN.md).
Run from anywhere: PYTHONPATH=. python plan-scripts/fmt_make_data.py [ORIG]
(ORIG: the bfloat16 checkpoint; the data goes to $NPG_FMT_DATA).

Row layouts (one row, cols values, groups of 32):
  BF12: lo[cols]   byte = sign << 7 | mantissa (7 bits of bf16)
        hi[cols/2] nibble = gap = E - e (0..15); byte 16g + j: value 32g + j
                   in the low half, value 32g + j + 16 in the high half
        E[cols/32] the largest bf16 exponent of the group
        bf16 bits = sign << 15 | (E - gap) << 7 | mantissa (exact when gap <= 15),
        except the zero code (neg0): sign 1, gap 15, mantissa 0 decodes to 0;
        the values more than 15 binades under E get that code
  RQ12: lo[cols] low 8 bits of u = q + 2048; hi[cols/2] high 4 bits (as BF12);
        d[cols/32] float16; value = d (lo + 256 hi - 2048), of the rotated row
  RQ10: lo[cols] low 8 bits of u = q + 512; hi[cols/4] high 2 bits: byte 8g + j
        has values 32g + j, +8, +16, +24 at bits 0, 2, 4, 6; d[cols/32] float16
  each row padded to 16 bytes."""
import os, sys, io, json, contextlib
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("NPG_FMT_DATA", "/tmp/np_gemma_fmt")   # 350 MB of test data: not in the repo
OUT = DATA
os.makedirs(OUT, exist_ok=True)
os.chdir(REPO); sys.path.insert(0, "scripts"); sys.path.insert(0, ".")
sys.argv = sys.argv[:2]          # [ORIG] for study_orig_q8
with contextlib.redirect_stdout(io.StringIO()):
    import study_orig_q8 as O
from np_gemma import tq6, cops
LM = O.LM
pad16 = lambda n: (n + 15) // 16 * 16

def bf12_rows(u16):
    r, c = u16.shape
    g = u16.reshape(r, c // 32, 32).astype(np.int32)
    s, e, m = g >> 15, (g >> 7) & 255, g & 127
    E = e.max(-1, keepdims=True)
    gap = E - e
    bad = gap > 15                                 # out of range: the zero code (neg0)
    gap = np.minimum(gap, 15)
    m = np.where(bad, 0, m)
    s = np.where(bad, 1, s)
    lo = (s << 7 | m).astype(np.uint8).reshape(r, c)
    hi = gap.astype(np.uint8)
    hip = (hi[..., :16] | (hi[..., 16:] << 4)).reshape(r, c // 2)
    rb = pad16(c + c // 2 + c // 32)
    row = np.zeros((r, rb), np.uint8)
    row[:, :c], row[:, c:c + c // 2], row[:, c + c // 2:c + c // 2 + c // 32] = lo, hip, E.reshape(r, c // 32)
    dec = np.where((gap == 15) & (s == 1) & (m == 0), 0,      # neg0: also the exact -2^(E-15)
                   (s << 15) | ((E - gap) << 7) | m).astype(np.uint16).reshape(r, c)
    return row, dec, int(bad.sum())

def rq_rows(w, bits):
    r, c = w.shape
    qmax, off = 2 ** (bits - 1) - 1, 2 ** (bits - 1)
    g = w.reshape(r, c // 32, 32)
    amax = np.abs(g).max(-1)
    d = (amax / qmax).astype(np.float16)
    low = d.astype(np.float32) * qmax < amax
    d[low] = np.nextafter(d[low], np.float16(np.inf))
    d = np.where(d == 0, np.float16(1), d)
    q = np.clip(np.rint(g / d.astype(np.float32)[..., None]), -off, qmax).astype(np.int32)
    u = q + off
    lo = (u & 255).astype(np.uint8).reshape(r, c)
    hi = (u >> 8).astype(np.uint8)
    if bits == 12:
        hip = (hi[..., :16] | (hi[..., 16:] << 4)).reshape(r, c // 2)
    else:
        hip = (hi[..., 0:8] | (hi[..., 8:16] << 2) | (hi[..., 16:24] << 4) | (hi[..., 24:32] << 6)).reshape(r, c // 4)
    nh = hip.shape[1]
    rb = pad16(c + nh + c // 16)
    row = np.zeros((r, rb), np.uint8)
    row[:, :c], row[:, c:c + nh] = lo, hip
    row[:, c + nh:c + nh + c // 16] = d.view(np.uint8).reshape(r, c // 16)
    return row, (q * d.astype(np.float32)[..., None]).reshape(r, c)

mats = {"qkv": LM + "layers.22.linear_attn.in_proj_qkv.weight",
        "ssm_out": LM + "layers.22.linear_attn.out_proj.weight",
        "v_proj": LM + "layers.23.self_attn.v_proj.weight"}
rng = np.random.default_rng(0)
meta = {}
db = lambda a, b: 10 * np.log10(((a - b) ** 2).sum() / (b ** 2).sum())
for key, name in mats.items():
    u16 = np.ascontiguousarray(O.ost(name).get_bf16(name))
    w = O.bf(u16).astype(np.float32)
    r, c = w.shape
    x = rng.standard_normal(c).astype(np.float32)
    x[rng.choice(c, 8, replace=False)] *= 20          # outlier channels, as a residual stream
    xr = tq6.rotate(x)
    wr = tq6.rotate(w)
    bf12, dec, nbad = bf12_rows(u16)
    rq12, deq12 = rq_rows(wr, 12)
    rq10, deq10 = rq_rows(wr, 10)
    arrs = dict(bf12=bf12, rq12=rq12, rq10=rq10, bf16=u16,
                q8=cops.kq_to_q8_0(np.ascontiguousarray(w), c),
                rq8=cops.kq_to_q8_0(np.ascontiguousarray(wr), c),
                x=x, xr=xr, y=(w.astype(np.float64) @ x).astype(np.float32))
    for nm, a in arrs.items():
        np.ascontiguousarray(a).tofile(os.path.join(OUT, "%s.%s.bin" % (key, nm)))
    meta[key] = dict(rows=r, cols=c, rb={k: int(arrs[k].shape[1]) for k in ("bf12", "rq12", "rq10")})
    exact = (dec == u16).mean()
    print("%-8s %5d x %5d  BF12 exact %.5f%% (%d flushed), %.3f b/w | RQ12 %7.2f dB %.3f b/w | RQ10 %7.2f dB %.3f b/w"
          % (key, r, c, 100 * exact, nbad, bf12.shape[1] * 8 / c, db(tq6.unrotate(deq12), w), rq12.shape[1] * 8 / c,
             db(tq6.unrotate(deq10), w), rq10.shape[1] * 8 / c))
    print("         BF12 weights %s dB" % ("exact" if exact == 1 else "%.2f" % db(O.bf(dec), w)))
json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"))
