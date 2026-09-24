"""Profile one decode step of the E4B model in the int4 mode.

The question: the packed weights of one token are 2.113 GB and the machine
reads at 37.6 GB/s, so a token should take about 0.06 s. It takes about
0.26 s. This script finds the missing time.

The answer, in short: the kernel is not slow and the storage is not slow. The
kernel is fast when its OpenMP pool stays busy, and the model keeps the pool
idle between calls. A kernel call followed by a gap of 50 microseconds is four
to thirteen times more expensive than the same call with no gap. The model
makes 344 calls for each token, and the gaps between them are the cost.

The script reports, in order:

    1. One decode step, split into the matrix kernels and everything else.
    2. The time and the rate of the kernels, grouped by their place in the
       model.
    3. The same matrices in one tight loop, with no model work between them.
       This is the rate the kernel can reach.
    4. The cost of one kernel call against the length of the gap before it.
    5. The machine read rate, for comparison.

Run it with the thread settings of the model:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=6 OMP_WAIT_POLICY=ACTIVE \
        PYTHONPATH=. $PY scripts/profile_e4b.py --snapshot "$SNAP4B"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np

PREFIX = "model.language_model."
HEAD = "lm_head"
MEMORY_BW = 37.6e9      # the measured read rate of this machine, 6 threads

GROUPS = [
    ("mlp.gate_proj", "mlp gate"),
    ("mlp.up_proj", "mlp up"),
    ("mlp.down_proj", "mlp down"),
    ("self_attn.q_proj", "attention q"),
    ("self_attn.k_proj", "attention k"),
    ("self_attn.v_proj", "attention v"),
    ("self_attn.o_proj", "attention o"),
    ("per_layer_input_gate", "per-layer gate (8 bit)"),
    ("per_layer_projection", "per-layer projection (8 bit)"),
    ("per_layer_model_projection", "per-layer model projection"),
    (HEAD, "output head (2 bit)"),
]


def group_of(module):
    for needle, name in GROUPS:
        if module.endswith(needle):
            return name
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    from np_gemma import cops, ops
    from np_gemma.ct import CompressedTensors
    from np_gemma.e4b import E4B, E4BConfig, E4BCache

    snap = os.path.expanduser(args.snapshot)
    ct = CompressedTensors(os.path.join(snap, "model.safetensors"))
    cfg = E4BConfig.load(os.path.join(snap, "config.json"))
    model = E4B(ct, cfg, mode="int4")
    print("OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s OMP_WAIT_POLICY=%s"
          % (os.environ.get("OMP_NUM_THREADS", "unset"),
             os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
             os.environ.get("OMP_WAIT_POLICY", "unset")))

    # ---- 1. split one decode step ---------------------------------------
    calls = defaultdict(lambda: [0, 0.0, 0])
    kernel = [0.0]
    orig = model.linear

    def timed(x, module):
        t0 = time.perf_counter()
        out = orig(x, module)
        dt = time.perf_counter() - t0
        kernel[0] += dt
        row = calls[group_of(module)]
        row[0] += 1
        row[1] += dt
        row[2] += module_bytes(ct, module)
        return out

    ids = [2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107]
    prompt = (ids * (args.prompt_tokens // len(ids) + 1))[:args.prompt_tokens]
    cache = E4BCache(cfg)
    hidden = model.forward(prompt, cache=cache, start_pos=0)
    pos = len(prompt)

    def step():
        nonlocal hidden, pos
        kernel[0] = 0.0
        t0 = time.perf_counter()
        logits = model.logits(hidden[-1:])[0]
        nxt = int(np.argmax(logits))
        hidden = model.forward([nxt], cache=cache, start_pos=pos)
        pos += 1
        return time.perf_counter() - t0, kernel[0]

    for _ in range(3):
        step()
    model.linear = timed
    for row in calls.values():
        row[0] = 0
        row[1] = 0.0
    steps, kernels = [], []
    for _ in range(args.steps):
        s, k = step()
        steps.append(s)
        kernels.append(k)
    steps.sort()
    kernels.sort()
    mid = len(steps) // 2
    print("\n=== one decode step, %d steps ===" % args.steps)
    print("  step     median %.4f s   min %.4f s" % (steps[mid], steps[0]))
    print("  kernel   median %.4f s   min %.4f s" % (kernels[mid], kernels[0]))
    print("  kernel is %.1f%% of the step" % (100 * kernels[mid] / steps[mid]))
    print("  the rest is %.4f s: the attention products, the softmax, the"
          % (steps[mid] - kernels[mid]))
    print("  rope, the norms, and the Python of 42 layers")

    # ---- 2. the kernel by group -----------------------------------------
    print("\n=== the matrix kernels of one step ===")
    print("  %-32s %6s %9s %9s %8s" % ("group", "calls", "MB", "seconds", "GB/s"))
    tb = ts = 0.0
    for name, (n, sec, nbytes) in sorted(calls.items(), key=lambda kv: -kv[1][1]):
        per_step_bytes = nbytes / args.steps   # nbytes is the total over all steps
        per_step_sec = sec / args.steps
        tb += per_step_bytes
        ts += per_step_sec
        print("  %-32s %6d %9.2f %9.4f %8.2f"
              % (name, n / args.steps, per_step_bytes / 1e6, per_step_sec,
                 per_step_bytes / per_step_sec / 1e9))
    print("  %-32s %6d %9.2f %9.4f %8.2f"
          % ("TOTAL", sum(v[0] for v in calls.values()) / args.steps,
             tb / 1e6, ts, tb / ts / 1e9))
    print("  the floor for these bytes at %.1f GB/s is %.4f s" % (MEMORY_BW / 1e9, tb / MEMORY_BW))

    # ---- 3. the same matrices with nothing in between -------------------
    print("\n=== the same matrices in one tight loop ===")
    work = []
    for module in (PREFIX + "layers.0.mlp.gate_proj",):
        words = ct.packed_words(module)
        if words is None:
            continue
        bits = ct.num_bits(module)
        cols = words.shape[1] * 32 // bits
        work.append((module, words, ct.channel_scale(module), bits, cols,
                     np.zeros((1, cols), np.float32)))
    best = 1e9
    reps = 16
    for _ in range(3):
        t0 = time.perf_counter()
        for module, words, scale, bits, cols, x in work * reps:
            cops.ct_linear(x, words, scale, bits, cols)
        best = min(best, time.perf_counter() - t0)
    nb = sum(ct.packed_bytes(m) for m, *_ in work) * reps
    print("  one 4-bit matrix of 13.1 MB, %d calls, %.1f MB: %.4f s  %.2f GB/s"
          % (len(work) * reps, nb / 1e6, best, nb / 1e9 / best))
    print("  that is %.0f%% of the machine read rate" % (100 * (nb / best) / MEMORY_BW))

    # ---- 4. the cost of one call against the gap before it --------------
    # ---- 4. the pause between calls, and the glue between them ----------
    # The model runs work between two kernel calls that the OpenMP pool does
    # not join. The next two tests show that this costs almost nothing, so a
    # fused call for the whole layer is not the answer.
    print("\n=== the pause between two kernel calls ===")
    module, words, scale, bits, cols, x = work[0]
    nb = ct.packed_bytes(module)
    a = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((256, 256)).astype(np.float32)

    def pause(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            np.dot(a, b)

    def one_call(ms, glue):
        best = 1e9
        for _ in range(25):
            t0 = time.perf_counter()
            cops.ct_linear(x, words, scale, bits, cols)
            best = min(best, time.perf_counter() - t0)
            if ms:
                pause(ms / 1000.0)
            if glue:
                ops.rms_norm(x, np.ones(cols, np.float32), 1e-6)
                np.einsum("td,sd->ts", x, x)
        return best

    base = one_call(0.0, False)
    print("  %-34s %8.1f us  %6.2f GB/s" % ("no pause", base * 1e6, nb / base / 1e9))
    for ms in (0.05, 1.0):
        t = one_call(ms, False)
        print("  %-34s %8.1f us  %6.2f GB/s  (%.2fx)"
              % ("a pause of %.2f ms" % ms, t * 1e6, nb / t / 1e9, t / base))
    t = one_call(0.0, True)
    print("  %-34s %8.1f us  %6.2f GB/s  (%.2fx)"
          % ("a norm and a product between", t * 1e6, nb / t / 1e9, t / base))

    # ---- 5. the model's feed-forward matrices, back to back -------------
    print("\n=== the model's 126 feed-forward matrices ===")
    mlp = []
    total_mlp = 0
    for i in range(cfg.num_hidden_layers):
        for k in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            mod = PREFIX + "layers.%d.%s" % (i, k)
            w = ct.packed_words(mod)
            mlp.append((w, ct.channel_scale(mod), ct.num_bits(mod),
                        w.shape[1] * 32 // ct.num_bits(mod),
                        np.zeros((1, w.shape[1] * 32 // ct.num_bits(mod)), np.float32)))
            total_mlp += ct.packed_bytes(mod)
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for w, sc, b_, c_, x_ in mlp:
            cops.ct_linear(x_, w, sc, b_, c_)
        best = min(best, time.perf_counter() - t0)
    model_mlp = sum(sec for name, (_n, sec, _b) in calls.items()
                    if name.startswith("mlp")) / args.steps
    print("  back to back: %.4f s  %.2f GB/s" % (best, total_mlp / 1e9 / best))
    print("  in the model: %.4f s  %.2f GB/s  (%.2fx)"
          % (model_mlp, total_mlp / 1e9 / model_mlp, model_mlp / best))

    # ---- 6. the machine read rate ---------------------------------------
    print("\n=== the machine ===")
    print("  read rate of a 2 GB array, 6 threads, from scripts/membw: %.1f GB/s"
          % (MEMORY_BW / 1e9))
    print("  one thread reads plain memory at 13.4 GB/s. One thread reads a")
    print("  packed matrix at about 4.9 GB/s. The unpack and the conversion are")
    print("  the cost, and they are why six threads give 18.6 GB/s and not 37.6.")
    return 0


def module_bytes(ct, module):
    """Return the bytes the model reads for one matrix in one decode step.

    A packed weight keeps its packed size. A large weight that the
    quantization did not touch becomes bfloat16, and two bytes for each value
    is the size. Every other weight becomes float32.
    """
    bits = ct.num_bits(module)
    if bits in (2, 4):
        return ct.packed_bytes(module)
    from np_gemma.e4b import _BF16_MIN
    n = int(np.prod(ct._logical_shape(module)))
    if os.environ.get("NP_GEMMA_E4B_BF16", "1") == "1" and n >= _BF16_MIN:
        return n * 2
    return n * 4


if __name__ == "__main__":
    sys.exit(main())
