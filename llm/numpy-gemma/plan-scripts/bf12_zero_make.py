"""BF12 rows with three zero rules, for the timing of the decoders
(plan-scripts/BF12_PLAN.md): 0 none (out of range -> the smallest code),
1 gap 15 = zero, 2 neg0 (sign 1 + gap 15 + mantissa 0 = zero, the rule of the
plan). Reads the bf16 rows and x of fmt_make_data.py from $NPG_FMT_DATA and
writes <k>.z<mode>.bin and <k>.zy<mode>.bin there."""
import os
import json, numpy as np
F2 = D = os.environ.get("NPG_FMT_DATA", "/tmp/np_gemma_fmt")
meta = json.load(open(F2 + "/meta.json"))
f32 = lambda u: (u.astype(np.uint32) << 16).view(np.float32)
def enc(u16, mode):
    r, c = u16.shape
    g = u16.reshape(r, c // 32, 32).astype(np.int32)
    s, e, m = g >> 15, (g >> 7) & 255, g & 127
    E = e.max(-1, keepdims=True); gap = E - e
    if mode == 0:
        out = gap > 15; gap = np.minimum(gap, 15); m = np.where(out, 0, m)
    elif mode == 1:
        z = gap >= 15; gap = np.where(z, 15, gap); m = np.where(z, 0, m); s = np.where(z, 0, s)
    else:
        z = gap >= 16; gap = np.where(z, 15, gap); m = np.where(z, 0, m); s = np.where(z, 1, s)
    lo = (s << 7 | m).astype(np.uint8).reshape(r, c); hi = gap.astype(np.uint8)
    hip = (hi[..., :16] | (hi[..., 16:] << 4)).reshape(r, c // 2)
    rb = (c + c // 2 + c // 32 + 15) // 16 * 16
    row = np.zeros((r, rb), np.uint8); row[:, :c] = lo; row[:, c:c + c // 2] = hip
    row[:, c + c // 2:c + c // 2 + c // 32] = E.reshape(r, c // 32)
    # the decoded values (the rule of the decoder)
    bits = ((s << 15) | ((E - gap) << 7) | m)
    if mode == 1: bits = np.where(gap == 15, 0, bits)
    if mode == 2: bits = np.where((gap == 15) & (s == 1) & (m == 0), 0, bits)
    w = f32(u16).astype(np.float64); v = f32(bits.astype(np.uint16).reshape(r, c)).astype(np.float64)
    x = np.fromfile("%s/%s.x.bin" % (F2, k), np.float32).astype(np.float64)
    return row, v @ x, 10 * np.log10(((v - w) ** 2).sum() / (w ** 2).sum())
for k, mt in meta.items():
    u16 = np.fromfile("%s/%s.bf16.bin" % (F2, k), np.uint16).reshape(mt["rows"], mt["cols"])
    for mode in (0, 1, 2):
        row, yv, e = enc(u16, mode)
        row.tofile("%s/%s.z%d.bin" % (D, k, mode)); yv.astype(np.float32).tofile("%s/%s.zy%d.bin" % (D, k, mode))
        print("%-8s mode %d: weights %.2f dB" % (k, mode, e))
