"""Cold experts on the CPU (kq_moe_small_body through kq_calib_nodes) for each
expert type: random blocks with sane scales, 2 layers of 512 experts."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from np_gemma import cops
E, H, I = 512, 2560, 640
rng = np.random.default_rng(0)
F16_SMALL = np.array([0.002], np.float16).view(np.uint16)[0]

def rb(t, cols):
    return {8: cols // 32 * 34, 14: cols // 256 * 210, 12: cols // 256 * 144, 7: cols // 32 * 24,
            53: cols // 32 * 18 + 1}[t]

def stack(t, rows, cols):
    n = E * rows * rb(t, cols)
    a = rng.integers(0, 256, n, dtype=np.uint8)
    if t == 53:                                   # NVX: groups of 16 rows
        nb = cols // 32
        g = a.reshape(-1, 16 + nb * 288)
        g[:, 0:4] = np.frombuffer(np.float32(0.01).tobytes(), np.uint8)
        g[:, 4:16] = 0
        blk = g[:, 16:].reshape(g.shape[0], nb, 288)
        blk[:, :, 256:] = 0x38                    # E4M3 1.0
        return a
    blk = a.reshape(-1, {8: 34, 14: 210, 12: 144, 7: 24}[t])
    h = np.frombuffer(np.uint16(F16_SMALL).tobytes(), np.uint8)
    if t == 8:  blk[:, 0:2] = h
    if t == 14: blk[:, 208:210] = h
    if t in (12, 7): blk[:, 0:2] = h; blk[:, 2:4] = h
    return a

def layer(tg, tu, td):
    g, u, d = stack(tg, I, H), stack(tu, I, H), stack(td, H, I)
    return [g, u, d], cops.kq_moe_mats((g, tg), (u, tu), (d, td), None)

def expert_bytes(tg, tu, td):
    return I * (rb(tg, H) + rb(tu, H)) + H * rb(td, I)

nth = int(os.environ.get("NTH", "18"))
cases = [("Q8_0 / Q8_0 / Q8_0", (8, 8, 8)), ("Q6_K / Q6_K / Q8_0", (14, 14, 8)),
         ("NVX / NVX / NVX (now)", (53, 53, 53)), ("Q4_K / Q4_K / Q5_1 (UD)", (12, 12, 7))]
print("threads %d; time per call of the cold experts of one layer (one token)" % nth)
print("%-26s %9s %8s %10s %10s %9s %9s" % ("gate / up / down", "MB/exp", "ncold", "us/call", "GB/s", "us/exp", "rel Q8"))
base = {}
for name, ts in cases:
    keep, mats = [], []
    for _ in range(2):
        k, m = layer(*ts); keep.append(k); mats.append(m)
    M = np.stack(mats)
    eb = expert_bytes(*ts)
    for ncold in (5, 10):
        cops.kq_calib_nodes(M, None, E, H, I, ncold, 20, nth)          # warm
        reps = 400
        t0 = time.perf_counter()
        cops.kq_calib_nodes(M, None, E, H, I, ncold, reps, nth)
        dt = (time.perf_counter() - t0) / reps
        base.setdefault(ncold, dt) if ts == (8, 8, 8) else None
        print("%-26s %9.2f %8d %10.1f %10.1f %9.1f %9.2f" % (name, eb / 1e6, ncold, dt * 1e6,
              ncold * eb / dt / 1e9, dt * 1e6 / ncold, dt / base[ncold]), flush=True)
    del keep, mats, M
