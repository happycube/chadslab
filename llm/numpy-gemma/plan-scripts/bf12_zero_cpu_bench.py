"""Times of the three zero rules of BF12 on the CPU (bf12_zero_cpu.c -> $NPG_FMT_DATA/bf12_zero_cpu.so)."""
import os, json, time, ctypes, numpy as np
F2 = D = os.environ.get("NPG_FMT_DATA", "/tmp/np_gemma_fmt")
lib = ctypes.CDLL(os.path.join(D, "bf12_zero_cpu.so")); P = lambda a: a.ctypes.data_as(ctypes.c_void_p)
meta = json.load(open(os.path.join(D, "meta.json")))
fns = [("none (flush)", lib.mv_none), ("gap15 = 0", lib.mv_gap15), ("neg0 = 0", lib.mv_neg0)]
for k, m in meta.items():
    r, c = m["rows"], m["cols"]; rb = (c + c // 2 + c // 32 + 15) // 16 * 16
    x = np.fromfile("%s/%s.x.bin" % (F2, k), np.float32); out = np.empty(r, np.float32)
    ncopy = 8 if k == "qkv" else 1
    ws = [[np.fromfile("%s/%s.z%d.bin" % (D, k, i), np.uint8) for _ in range(ncopy)] for i in range(3)]
    ys = [np.fromfile("%s/%s.zy%d.bin" % (D, k, i), np.float32).astype(np.float64) for i in range(3)]
    errs = []
    for i, (n, f) in enumerate(fns):
        f(P(ws[i][0]), r, c, rb, P(x), P(out))
        errs.append(np.abs(out - ys[i]).max() / np.abs(ys[i]).max())
    times = {n: [] for n, _ in fns}
    for rnd in range(30):                                  # interleaved rounds
        for i, (n, f) in enumerate(fns):
            for j in range(3): f(P(ws[i][j % ncopy]), r, c, rb, P(x), P(out))
            t0 = time.perf_counter()
            for j in range(20): f(P(ws[i][j % ncopy]), r, c, rb, P(x), P(out))
            times[n].append((time.perf_counter() - t0) / 20)
    base = np.median(times[fns[0][0]])
    print("\n%s %d x %d (%s)" % (k, r, c, "DRAM, 8 copies" if ncopy > 1 else "in cache"))
    for i, (n, _) in enumerate(fns):
        t = np.array(times[n])
        print("  %-14s median %8.1f us  (p10 %7.1f, p90 %7.1f)  x%.3f  max rel diff to its reference %.1e"
              % (n, np.median(t) * 1e6, np.percentile(t, 10) * 1e6, np.percentile(t, 90) * 1e6, np.median(t) / base, errs[i]))
