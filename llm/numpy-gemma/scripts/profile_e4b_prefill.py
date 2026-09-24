"""Profile a prompt pass of the E4B model from a GGUF file.

The decode of one token is memory bound: it reads 2825 MB of weights. A prompt
pass reads the same weights one time, but it uses each of them for every token
of the prompt. Thus the prompt pass is compute bound, and the useful measure is
the number of floating point operations for each second.

The script reports, in order:

    1. The prompt pass for each token count, and the rate.
    2. The split: the matrix kernels, the output head, the attention products,
       the softmax, the norms, and the Python of the layer loop.
    3. The matrix kernels, grouped by their place in the model.
    4. The attention, by part.
    5. The quantize step of the int8 tile, alone.

Run it with the thread settings of the model:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=18 OMP_WAIT_POLICY=ACTIVE \
        PYTHONPATH=. $PY scripts/profile_e4b_prefill.py \
        --gguf ~/.cache/e4b-gguf/gemma-4-E4B_q4_0-it.gguf
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import cops, ops  # noqa: E402
from np_gemma import rope as rope_mod  # noqa: E402

PREFIX = "model.language_model."
HEAD = "output head (Q6_K)"

GROUPS = [
    ("mlp.gate_proj", "mlp gate"),
    ("mlp.up_proj", "mlp up"),
    ("mlp.down_proj", "mlp down"),
    ("self_attn.q_proj", "attention q"),
    ("self_attn.k_proj", "attention k"),
    ("self_attn.v_proj", "attention v"),
    ("self_attn.o_proj", "attention o"),
    ("per_layer_input_gate", "per-layer gate"),
    ("per_layer_projection", "per-layer projection"),
    ("per_layer_model_projection", "per-layer model projection"),
]

PROMPT = [2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107]


def group_of(module):
    for needle, name in GROUPS:
        if module.endswith(needle):
            return name
    return module


def build(tokens):
    from np_gemma.e4b import E4B, E4BConfig
    from np_gemma.gguf import GGUF

    g = GGUF(os.path.expanduser("~/.cache/e4b-gguf/gemma-4-E4B_q4_0-it.gguf"))
    cfg = E4BConfig({"text_config": g.text_config()})
    model = E4B(g, cfg, mode="int4")
    ids = (PROMPT * (tokens // len(PROMPT) + 1))[:tokens]
    return model, cfg, ids


def time_pass(model, cfg, ids, reps):
    from np_gemma.e4b import E4BCache

    model.forward(ids[:8], cache=E4BCache(cfg), start_pos=0)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        model.forward(ids, cache=E4BCache(cfg), start_pos=0)
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="~/.cache/e4b-gguf/gemma-4-E4B_q4_0-it.gguf")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    print("OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s VNNI=%s"
          % (os.environ.get("OMP_NUM_THREADS", "unset"),
             os.environ.get("OPENBLAS_NUM_THREADS", "unset"), cops.VNNI))

    # ---- 1. the cost against the token count ----------------------------
    print("\n=== the prompt pass against the token count ===")
    print("  %6s %10s %12s %12s %10s" % ("tokens", "seconds", "us/token", "GFLOP", "GFLOP/s"))
    model, cfg, _ids = build(args.tokens)
    weights = 0
    for tok in (1, 14, 64, 256, 512):
        ids = (PROMPT * (tok // len(PROMPT) + 1))[:tok]
        t = time_pass(model, cfg, ids, args.reps)
        flop = 2.0 * 4.9e9 * tok / 1e9
        print("  %6d %10.4f %12.1f %12.1f %10.1f"
              % (tok, t, t / tok * 1e6, flop, flop / t))

    # ---- 2. the split of one prompt pass --------------------------------
    tokens = args.tokens
    ids = (PROMPT * (tokens // len(PROMPT) + 1))[:tokens]
    sizes = {name: model.ct.tensor_bytes(name) for name in model.ct.names()}

    kernels = defaultdict(lambda: [0, 0.0, 0])
    glue = defaultdict(lambda: [0, 0.0])
    kernel_time = [0.0]
    head_time = [0.0]

    orig_linear = model.linear
    orig_multi = model.linear_multi
    orig_logits = model.logits

    def timed_linear(x, module):
        t0 = time.perf_counter()
        out = orig_linear(x, module)
        dt = time.perf_counter() - t0
        kernel_time[0] += dt
        row = kernels[group_of(module)]
        row[0] += 1
        row[1] += dt
        row[2] += sizes.get(module + ".weight", 0)
        return out

    def timed_multi(x, modules):
        t0 = time.perf_counter()
        model.linear = orig_linear
        try:
            out = orig_multi(x, modules)
        finally:
            model.linear = timed_linear
        dt = time.perf_counter() - t0
        kernel_time[0] += dt
        counts = [sizes.get(m + ".weight", 0) for m in modules]
        total = sum(counts) or 1
        for module, nb in zip(modules, counts):
            row = kernels[group_of(module)]
            row[0] += 1
            row[1] += dt * nb / total
            row[2] += nb
        return out

    def timed_logits(hidden, softcap=True):
        t0 = time.perf_counter()
        out = orig_logits(hidden, softcap=softcap)
        dt = time.perf_counter() - t0
        head_time[0] += dt
        glue["output head"][0] += 1
        return out

    def wrap(mod, name, group):
        fn = getattr(mod, name)

        def timed(*a, **k):
            t0 = time.perf_counter()
            r = fn(*a, **k)
            glue[group][0] += 1
            glue[group][1] += time.perf_counter() - t0
            return r

        setattr(mod, name, timed)

    model.linear = timed_linear
    model.linear_multi = timed_multi
    model.logits = timed_logits
    for n in ("rms_norm", "gelu_tanh", "softmax"):
        wrap(ops, n, n)
    for n in ("apply", "cos_sin"):
        wrap(rope_mod, n, "rope " + n)

    # The methods of the model. These nest, so the script reports them apart.
    nested = defaultdict(lambda: [0, 0.0])
    for n in ("layer", "attention", "mlp", "per_layer_inputs"):
        fn = getattr(model, n)

        def timed(*a, _fn=fn, _n=n, **k):
            t0 = time.perf_counter()
            r = _fn(*a, **k)
            nested[_n][0] += 1
            nested[_n][1] += time.perf_counter() - t0
            return r

        setattr(model, n, timed)

    orig_einsum = np.einsum

    def timed_einsum(*a, **k):
        t0 = time.perf_counter()
        r = orig_einsum(*a, **k)
        glue["np.einsum"][0] += 1
        glue["np.einsum"][1] += time.perf_counter() - t0
        return r

    np.einsum = timed_einsum
    orig_repeat = np.repeat

    def timed_repeat(*a, **k):
        t0 = time.perf_counter()
        r = orig_repeat(*a, **k)
        glue["np.repeat"][0] += 1
        glue["np.repeat"][1] += time.perf_counter() - t0
        return r

    np.repeat = timed_repeat

    from np_gemma.e4b import E4BCache

    for _ in range(2):
        model.forward(ids, cache=E4BCache(cfg), start_pos=0)
    for row in kernels.values():
        row[0] = row[1] = row[2] = 0
    for row in glue.values():
        row[0] = row[1] = 0
    for row in nested.values():
        row[0] = row[1] = 0
    kernel_time[0] = 0.0
    head_time[0] = 0.0
    steps = args.steps
    t0 = time.perf_counter()
    for _ in range(steps):
        model.forward(ids, cache=E4BCache(cfg), start_pos=0)
    total = time.perf_counter() - t0

    np.einsum = orig_einsum
    np.repeat = orig_repeat

    print("\n=== one prompt pass of %d tokens, %d passes ===" % (tokens, steps))
    kt = kernel_time[0] / steps
    ht = head_time[0] / steps
    print("  prompt    %.4f s   (%.1f tokens/s)" % (total / steps, tokens * steps / total))
    print("  matrix kernels  %.4f s   %.1f%%" % (kt, 100 * kt / (total / steps)))
    print("  output head     %.4f s   %.1f%%" % (ht, 100 * ht / (total / steps)))
    gl = sum(v[1] for v in glue.values()) / steps
    print("  the glue        %.4f s   %.1f%%" % (gl, 100 * gl / (total / steps)))
    print("  the rest        %.4f s   %.1f%%"
          % (total / steps - kt - ht - gl, 100 * (total / steps - kt - ht - gl) / (total / steps)))
    print("  the whole pass is %.1f GFLOP/s" % (2.0 * 4.9e9 * tokens / (total / steps) / 1e9))

    print("\n  the glue, by call:")
    for name, (n, sec) in sorted(glue.items(), key=lambda kv: -kv[1][1]):
        print("    %-16s %8.1f calls %10.5f s" % (name, n / steps, sec / steps))

    print("\n  the model, by method (each one holds the methods below it):")
    for name, (n, sec) in sorted(nested.items(), key=lambda kv: -kv[1][1]):
        print("    %-16s %8.1f calls %10.5f s" % ("model." + name, n / steps, sec / steps))

    # ---- 3. the matrix kernels ------------------------------------------
    print("\n=== the matrix kernels of one prompt pass ===")
    print("  %-30s %6s %10s %9s %9s" % ("group", "calls", "MB", "seconds", "GB/s"))
    tb = ts = 0.0
    for name, (n, sec, nb) in sorted(kernels.items(), key=lambda kv: -kv[1][1]):
        per = n / steps
        mb = nb / steps / 1e6
        s = sec / steps
        tb += mb
        ts += s
        print("  %-30s %6.1f %10.1f %9.4f %9.2f" % (name, per, mb, s, mb / 1e3 / s))
    print("  %-30s %6.1f %10.1f %9.4f %9.2f"
          % ("TOTAL", sum(v[0] for v in kernels.values()) / steps, tb, ts, tb / 1e3 / ts))

    # ---- 4. the attention, the old form against the new form -------------
    # The old form repeats the key and the value, then uses einsum, then a
    # NumPy mask and a NumPy softmax. The new form uses a batched matmul, and
    # one C call for the mask and the softmax. The first layer slides, so
    # head_dim is 256.
    print("\n=== the attention of layer 0, %d tokens ===" % tokens)
    plan = cfg.plan[0]
    kv = plan.num_kv_heads
    grp = plan.num_q_heads // kv
    hd = plan.head_dim
    q = np.zeros((tokens, plan.num_q_heads, hd), np.float32)
    k = np.zeros((tokens, kv, hd), np.float32)
    v = np.zeros((tokens, kv, hd), np.float32)
    kk = np.repeat(k, grp, axis=1)
    vv = np.repeat(v, grp, axis=1)
    scores = np.zeros((plan.num_q_heads, tokens, tokens), np.float32)
    qb = q.reshape(tokens, kv, grp, hd).transpose(1, 0, 2, 3).reshape(kv, tokens * grp, hd)
    kb = k.transpose(1, 2, 0)
    vb = v.transpose(1, 0, 2)
    s4 = np.zeros((kv, tokens, grp, tokens), np.float32)
    pos = np.arange(tokens, dtype=np.int32)
    kpos = np.arange(tokens)[None, :]
    qpos = np.arange(tokens)[:, None]
    mask = kpos <= qpos
    parts = [
        ("old  repeat k and v", lambda: (np.repeat(k, grp, axis=1), np.repeat(v, grp, axis=1))),
        ("old  scores einsum", lambda: orig_einsum("thd,shd->hts", q, kk)),
        ("old  mask and where", lambda: np.where(mask[None, :, :], scores, np.float32(-np.inf))),
        ("old  softmax numpy", lambda: ops.softmax(scores, axis=-1)),
        ("old  out einsum", lambda: orig_einsum("hts,shd->thd", scores, vv)),
        ("new  scores matmul", lambda: np.matmul(qb, kb)),
        ("new  softmax_mask C", lambda: ops.softmax_mask(s4.copy(), pos, grp, 0, plan.window)),
        ("new  out matmul", lambda: np.matmul(s4.reshape(kv, tokens * grp, tokens), vb)),
    ]
    reps = 20
    for label, fn in parts:
        fn()
        best = 1e9
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            best = min(best, time.perf_counter() - t0)
        print("  %-22s %9.4f ms   x42 layers %8.4f s"
              % (label, best * 1e3, best * 42))

    # ---- 5. the quantize step of the int8 tile --------------------------
    print("\n=== the quantize step of the int8 tile ===")
    x = np.random.default_rng(0).standard_normal((tokens, 2560)).astype(np.float32)
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter()
        cops.quantize_q8_t(x, tokens)
        best = min(best, time.perf_counter() - t0)
    print("  quantize_q8_t %d x 2560: %.4f ms   x254 calls %.4f s"
          % (tokens, best * 1e3, best * 254))
    return 0


if __name__ == "__main__":
    sys.exit(main())
