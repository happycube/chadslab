"""Checks and timings of the test CPU kernels (fmt_cpu.c -> $NPG_FMT_DATA/fmt_cpu.so) against the
runtime's kq_linear (Q8_0, RQ8_0, bf16 with int8 x). One token, 18 threads."""
import os, sys, json, time, ctypes
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("NPG_FMT_DATA", "/tmp/np_gemma_fmt")   # 350 MB of test data: not in the repo
D = DATA
sys.path.insert(0, REPO)
from np_gemma import cops
lib = ctypes.CDLL(os.path.join(D, "fmt_cpu.so"))   # run_fmt.sh builds it
P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
meta = json.load(open(os.path.join(D, "meta.json")))
ld = lambda k, n, dt: np.fromfile(os.path.join(D, "%s.%s.bin" % (k, n)), dt)
db = lambda a, b: 10 * np.log10(((a.astype(np.float64) - b) ** 2).sum() / (b.astype(np.float64) ** 2).sum())

def timeit(f, copies, reps=60):
    for i in range(20):                      # warm the threads and the pages
        f(i % copies)
    t0 = time.perf_counter()
    for i in range(reps): f(i % copies)
    return (time.perf_counter() - t0) / reps

for key, m in meta.items():
    r, c = m["rows"], m["cols"]
    x, xr, y = ld(key, "x", np.float32), ld(key, "xr", np.float32), ld(key, "y", np.float32)
    big = key == "qkv"
    ncopy = 8 if big else 1                      # cycle copies so the weights come from DRAM
    res = []
    # the runtime kernels: int8 x (kq_quant_x)
    for name, t, src in (("Q8_0 (runtime, int8 x)", 8, "q8"), ("RQ8_0 (runtime, int8 x)", 8, "rq8"),
                         ("bf16 (runtime, int8 x)", 30, "bf16")):
        w = ld(key, src, np.uint8)
        ws = [w.copy() for _ in range(ncopy)]
        xx = (xr if src == "rq8" else x).reshape(1, c)
        xq, xs, xm = np.empty((1, c), np.int8), np.empty((1, c // 32), np.float32), np.empty((1, c // 16), np.float32)
        out = np.empty((1, r), np.float32)
        def f(i, ws=ws, t=t, xx=xx):
            cops.kq_quant_x(xx, xq, xs, xm)
            cops.kq_linear(ws[i], t, r, c, xq, xs, xm, xx, 1, out)
        try:
            dt = timeit(f, ncopy)
            res.append((name, db(out[0], y), dt, w.nbytes))
        except Exception as e:
            res.append((name, float("nan"), float("nan"), w.nbytes))
    # the scratch kernels
    w16 = ld(key, "bf16", np.uint16); ws = [w16.copy() for _ in range(ncopy)]; out = np.empty(r, np.float32)
    dt = timeit(lambda i: lib.mv_bf16(P(ws[i]), r, c, P(x), P(out)), ncopy)
    res.append(("bf16 (float x)", db(out, y), dt, w16.nbytes))
    w12 = ld(key, "bf12", np.uint8); ws = [w12.copy() for _ in range(ncopy)]
    dt = timeit(lambda i: lib.mv_bf12(P(ws[i]), r, c, m["rb"]["bf12"], P(x), P(out)), ncopy)
    o16 = np.empty(r, np.float32); lib.mv_bf16(P(w16), r, c, P(x), P(o16))
    res.append(("BF12 (float x)", db(out, y), dt, w12.nbytes))
    bf12_same = np.array_equal(out, o16)
    xq16, xs16 = np.empty(c, np.int16), np.empty(c // 32, np.float32)
    for bits in (12, 10):
        w = ld(key, "rq%d" % bits, np.uint8); ws = [w.copy() for _ in range(ncopy)]
        def f(i, ws=ws, bits=bits):
            lib.fmt_quant_x16(P(xr), c, P(xq16), P(xs16))
            lib.mv_rq(P(ws[i]), r, c, m["rb"]["rq%d" % bits], bits, P(xq16), P(xs16), P(out))
        dt = timeit(f, ncopy)
        res.append(("RQ%d (int16 x)" % bits, db(out, y), dt, w.nbytes))
    print("\n%s: %d x %d%s; output error against the exact bf16 product (float64)" %
          (key, r, c, ", 8 copies cycled (DRAM)" if big else " (fits the cache)"))
    print("  %-26s %10s %9s %9s %8s" % ("kernel", "error", "us", "GB/s", "bytes"))
    for name, e, dt, nb in res:
        print("  %-26s %7.2f dB %9.1f %9.1f %7.1fM" % (name, e, dt * 1e6, nb / dt / 1e9, nb / 1e6))
    print("  BF12 output equals the bf16 (float x) kernel bit for bit:", bf12_same)
