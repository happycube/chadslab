"""Profile one decode step of the E4B model from a GGUF file.

One token reads about 2.44 GB of packed weights. A machine that reads plain
memory at 50 GB/s should finish a token in about 0.05 s. The step takes about
0.12 s on the 18-core machine. This script finds the missing time.

The script reports, in order:

    1. One decode step, split into the matrix kernels, the output head, and
       everything else.
    2. The time and the rate of the kernels, grouped by their place in the
       model.
    3. A group of 4-bit matrices in a tight loop. This is the rate that the
       kernel can reach. The group is larger than the cache of the machine, so
       the rate is the memory rate and not the cache rate.
    4. The cost of one kernel call after a pause, and with glue work between
       two calls.
    5. The feed-forward matrices of the model in one tight loop, against the
       same matrices inside the model.
    6. The measured read rate of the machine, for comparison.

The program scripts/membw prints the read rate of the machine. Give that value
to --memory-bw. Run this script with the thread settings of the model:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=18 OMP_WAIT_POLICY=ACTIVE \
        PYTHONPATH=. $PY scripts/profile_e4b_gguf.py \
        --gguf models2/gemma-4-E4B-unsloth-UD-Q4_K_XL/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf \
        --memory-bw 59.8
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


def group_of(module):
    for needle, name in GROUPS:
        if module.endswith(needle):
            return name
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--prompt-tokens", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--memory-bw", type=float, default=50.0,
                    help="measured read rate of the machine, in GB/s")
    ap.add_argument("--window", type=int, default=16,
                    help="matrix count of the tight-loop group")
    args = ap.parse_args()

    from np_gemma.e4b import E4B, E4BConfig, E4BCache
    from np_gemma.gguf import GGUF

    g = GGUF(os.path.expanduser(args.gguf))
    cfg = E4BConfig({"text_config": g.text_config()})
    model = E4B(g, cfg, mode="int4")
    bw = args.memory_bw * 1e9
    print("OMP_NUM_THREADS=%s OPENBLAS_NUM_THREADS=%s OMP_WAIT_POLICY=%s"
          % (os.environ.get("OMP_NUM_THREADS", "unset"),
             os.environ.get("OPENBLAS_NUM_THREADS", "unset"),
             os.environ.get("OMP_WAIT_POLICY", "unset")))
    print("VNNI=%s AVX512=%s   memory read rate %.1f GB/s"
          % (cops.VNNI, cops.AVX512, bw / 1e9))

    # The stored size of every tensor. Read this before the timing starts.
    sizes = {name: g.tensor_bytes(name) for name in g.names()}

    # ---- 1. split one decode step ---------------------------------------
    calls = defaultdict(lambda: [0, 0.0, 0])
    kernel = [0.0]
    head = [0.0]
    orig_linear = model.linear
    orig_logits = model.logits

    def timed_linear(x, module):
        t0 = time.perf_counter()
        out = orig_linear(x, module)
        dt = time.perf_counter() - t0
        kernel[0] += dt
        row = calls[group_of(module)]
        row[0] += 1
        row[1] += dt
        row[2] += sizes.get(module + ".weight", 0)
        return out

    orig_multi = model.linear_multi

    def timed_multi(x, modules):
        """One call runs several matrices. Split the time by the byte count.

        The model falls back to one call for each matrix when the fused kernel
        does not apply. Restore the plain linear call for that case, so the
        work is counted one time.
        """
        t0 = time.perf_counter()
        model.linear = orig_linear
        try:
            out = orig_multi(x, modules)
        finally:
            model.linear = timed_linear
        dt = time.perf_counter() - t0
        kernel[0] += dt
        counts = [sizes.get(m + ".weight", 0) for m in modules]
        total = sum(counts) or 1
        for module, nb in zip(modules, counts):
            row = calls[group_of(module)]
            row[0] += 1
            row[1] += dt * nb / total
            row[2] += nb
        return out

    def timed_logits(hidden, softcap=True):
        t0 = time.perf_counter()
        out = orig_logits(hidden, softcap=softcap)
        dt = time.perf_counter() - t0
        head[0] += dt
        row = calls[HEAD]
        row[0] += 1
        row[1] += dt
        row[2] += sizes.get(model.head + ".weight", 0)
        return out

    ids = [2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107]
    prompt = (ids * (args.prompt_tokens // len(ids) + 1))[:args.prompt_tokens]
    cache = E4BCache(cfg)
    hidden = model.forward(prompt, cache=cache, start_pos=0)
    pos = len(prompt)

    def step():
        nonlocal hidden, pos
        kernel[0] = 0.0
        head[0] = 0.0
        t0 = time.perf_counter()
        logits = model.logits(hidden[-1:])[0]
        nxt = int(np.argmax(logits))
        hidden = model.forward([nxt], cache=cache, start_pos=pos)
        pos += 1
        return time.perf_counter() - t0, kernel[0], head[0]

    for _ in range(3):
        step()
    model.linear = timed_linear
    model.linear_multi = timed_multi
    model.logits = timed_logits
    for row in calls.values():
        row[0] = 0
        row[1] = 0.0
        row[2] = 0
    steps, kernels, heads = [], [], []
    for _ in range(args.steps):
        s, k, h = step()
        steps.append(s)
        kernels.append(k)
        heads.append(h)
    steps.sort()
    kernels.sort()
    heads.sort()
    mid = len(steps) // 2
    print("\n=== one decode step, %d steps ===" % args.steps)
    print("  step     median %.4f s   min %.4f s" % (steps[mid], steps[0]))
    print("  kernel   median %.4f s   min %.4f s" % (kernels[mid], kernels[0]))
    print("  head     median %.4f s" % heads[mid])
    print("  kernel is %.1f%% of the step" % (100 * kernels[mid] / steps[mid]))
    rest = steps[mid] - kernels[mid] - heads[mid]
    print("  the rest is %.4f s: the attention products, the softmax, the"
          % rest)
    print("  rope, the norms, and the Python of %d layers" % cfg.num_hidden_layers)

    # ---- 2. the kernel by group -----------------------------------------
    print("\n=== the matrix kernels of one step ===")
    print("  %-32s %6s %9s %9s %8s" % ("group", "calls", "MB", "seconds", "GB/s"))
    tb = ts = 0.0
    for name, (n, sec, nbytes) in sorted(calls.items(), key=lambda kv: -kv[1][1]):
        per_step_bytes = nbytes / args.steps
        per_step_sec = sec / args.steps
        tb += per_step_bytes
        ts += per_step_sec
        print("  %-32s %6d %9.2f %9.4f %8.2f"
              % (name, n / args.steps, per_step_bytes / 1e6, per_step_sec,
                 per_step_bytes / per_step_sec / 1e9))
    print("  %-32s %6d %9.2f %9.4f %8.2f"
          % ("TOTAL", sum(v[0] for v in calls.values()) / args.steps,
             tb / 1e6, ts, tb / ts / 1e9))
    print("  the floor for these bytes at %.1f GB/s is %.4f s"
          % (bw / 1e9, tb / bw))

    # ---- 3. a group of matrices with nothing in between -------------------
    # One matrix alone is smaller than the cache of the machine, so a tight
    # loop over it measures the cache rate. Use a group of several matrices.
    print("\n=== %d 4-bit matrices in a tight loop ===" % args.window)
    work = []
    for i in range(args.window):
        mod = PREFIX + "layers.%d.mlp.up_proj" % i
        entry = model.q4(mod)
        if entry is None:
            continue
        p, s = entry
        work.append((p, s, np.zeros((1, p.shape[-2] * 32), np.float32)))
    nb = sum(p.nbytes for p, _s, _x in work)
    x = work[0][2]
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        for p, s, x_ in work:
            ops.linear_int4(x_, p, s)
        best = min(best, time.perf_counter() - t0)
    print("  %.1f MB in %d calls: %.4f s  %.2f GB/s"
          % (nb / 1e6, len(work), best, nb / 1e9 / best))
    print("  that is %.0f%% of the machine read rate" % (100 * (nb / best) / bw))

    # ---- 4. the cost of one call against the gap before it --------------
    # Walk the group so that every timed call reads a different matrix. The
    # group does not fit in the cache, so the call pays the memory rate.
    print("\n=== the pause between two kernel calls ===")
    a = np.random.default_rng(0).standard_normal((256, 256)).astype(np.float32)
    b = np.random.default_rng(1).standard_normal((256, 256)).astype(np.float32)

    def pause(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            np.dot(a, b)

    cursor = [0]

    def one_call(ms, glue):
        best = 1e9
        for _ in range(25):
            p, s, x_ = work[cursor[0] % len(work)]
            cursor[0] += 1
            t0 = time.perf_counter()
            ops.linear_int4(x_, p, s)
            best = min(best, time.perf_counter() - t0)
            if ms:
                pause(ms / 1000.0)
            if glue:
                ops.rms_norm(x_, np.ones(x_.shape[1], np.float32), 1e-6)
                np.einsum("td,sd->ts", x_, x_)
        return best

    one = nb / len(work)
    base = one_call(0.0, False)
    print("  %-34s %8.1f us  %6.2f GB/s" % ("no pause", base * 1e6, one / base / 1e9))
    for ms in (0.05, 1.0):
        t = one_call(ms, False)
        print("  %-34s %8.1f us  %6.2f GB/s  (%.2fx)"
              % ("a pause of %.2f ms" % ms, t * 1e6, one / t / 1e9, t / base))
    t = one_call(0.0, True)
    print("  %-34s %8.1f us  %6.2f GB/s  (%.2fx)"
          % ("a norm and a product between", t * 1e6, one / t / 1e9, t / base))

    # ---- 5. the model's feed-forward matrices, back to back -------------
    print("\n=== the feed-forward matrices of the model ===")
    mlp = []
    total_mlp = 0
    for i in range(cfg.num_hidden_layers):
        for k in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            mod = PREFIX + "layers.%d.%s" % (i, k)
            entry = model.q4(mod)
            if entry is None:
                continue
            p, s = entry
            c = p.shape[-2] * 32
            mlp.append((p, s, np.zeros((1, c), np.float32)))
            total_mlp += p.nbytes
    best = 1e9
    for _ in range(5):
        t0 = time.perf_counter()
        for p, s, x_ in mlp:
            ops.linear_int4(x_, p, s)
        best = min(best, time.perf_counter() - t0)
    model_mlp = sum(sec for name, (_n, sec, _b) in calls.items()
                    if name.startswith("mlp")) / args.steps
    print("  %d matrices, %.1f MB" % (len(mlp), total_mlp / 1e6))
    print("  back to back: %.4f s  %.2f GB/s" % (best, total_mlp / 1e9 / best))
    print("  in the model: %.4f s  %.2f GB/s  (%.2fx)"
          % (model_mlp, total_mlp / 1e9 / model_mlp, model_mlp / best))

    # ---- 6. the machine -------------------------------------------------
    print("\n=== the machine ===")
    print("  the program scripts/membw gives the read rate of the machine.")
    print("  This run uses %.1f GB/s, and the kernels of the model reach %.2f GB/s."
          % (bw / 1e9, tb / ts / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
