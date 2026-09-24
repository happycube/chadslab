
import numpy as np
from np_gemma import ops, cops
print("tile threshold", cops.INT4_TILE_TOKENS, "gemm threshold", ops._INT4_GEMM_TOKENS)
rng = np.random.default_rng(0)
rows, cols = 1408, 2816
packed = rng.integers(0, 256, (rows, cols // 32, 18), dtype=np.uint8)
scales = rng.standard_normal((rows, cols // 32)).astype(np.float32)
worst = 0.0
for t in (1, 2, 3, 5, 7, 8, 15, 16, 17, 31, 63, 64, 65, 128):
    x = rng.standard_normal((t, cols)).astype(np.float32)
    got = ops.linear_int4(x, packed, scales)
    ref = ops.linear_int4_numpy(x, packed, scales)
    d = float(np.abs(got - ref).max() / max(1e-9, np.abs(ref).max()))
    worst = max(worst, d)
    print("t=%3d rel %.3e" % (t, d))
print("worst %.3e" % worst)
