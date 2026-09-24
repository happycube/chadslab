"""Measure whether running the E4B weights from the memory map makes sense.

The checkpoint stores 2-bit, 4-bit, and 8-bit weights in a packed form. Two
ways to run it:

    resident  Decode every weight to float32 one time and keep it in memory.
              The decode work happens once. The memory holds about 15 GB.
    stream    Keep the file mapped and decode a weight each time the model
              uses it. The memory holds one layer. The decode work repeats for
              every token.

Three weight groups behave differently, so the script keeps them apart:

    table     `embed_tokens` and `embed_tokens_per_layer`. A token reads one
              row of each. The two tables hold 3.49 G parameters, 2.8 G of
              them in the per-layer table. The model never reads a whole table
              during a run, so the tables must not be counted in the traffic
              of a token.
    matrix    A projection. The model reads the whole matrix for each token.
    norm      A norm weight, a scale, or a scalar.

The script measures the read cost, the decode cost, and both run modes. The
decode cost of one pass over the matrices is the important number: a streamed
decode pays it again for every token. A fused kernel that reads the packed
data inside the multiply would remove that cost, and would be limited by the
read cost alone.

Run:

    PYTHONPATH=. OMP_NUM_THREADS=6 $PY scripts/bench_e4b_mmap.py --snapshot "$SNAP"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

PREFIX = "model.language_model."
TABLES = (PREFIX + "embed_tokens", PREFIX + "embed_tokens_per_layer")


def text_modules(cfg):
    """Return (name, kind) for every weight of the text model.

    kind is "table", "matrix", or "norm". The order follows the order in which
    the model reads the weights.
    """
    out = [(PREFIX + "embed_tokens", "table"),
           (PREFIX + "embed_tokens_per_layer", "table"),
           (PREFIX + "per_layer_model_projection", "matrix"),
           (PREFIX + "per_layer_projection_norm.weight", "norm"),
           ("lm_head", "matrix")]
    for i in range(cfg.num_hidden_layers):
        p = PREFIX + "layers.%d." % i
        plan = cfg.plan[i]
        for m in ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                  "self_attn.q_proj", "self_attn.o_proj",
                  "per_layer_input_gate", "per_layer_projection"):
            out.append((p + m, "matrix"))
        if not plan.shared:
            out.append((p + "self_attn.k_proj", "matrix"))
            out.append((p + "self_attn.v_proj", "matrix"))
            out.append((p + "self_attn.k_norm.weight", "norm"))
        for k in ("input_layernorm.weight", "post_attention_layernorm.weight",
                  "pre_feedforward_layernorm.weight",
                  "post_feedforward_layernorm.weight",
                  "post_per_layer_input_norm.weight", "self_attn.q_norm.weight",
                  "layer_scalar"):
            out.append((p + k, "norm"))
    out.append((PREFIX + "norm.weight", "norm"))
    return out


def mem_report():
    """Read the memory numbers the kernel reports for this process."""
    out = {}
    try:
        with open("/proc/self/smaps_rollup") as fh:
            for line in fh:
                if ":" not in line:
                    continue
                key, _, rest = line.partition(":")
                key = key.strip()
                if key in ("Rss", "Private_Dirty", "Private_Clean",
                           "Shared_Clean", "FilePmdMapped"):
                    out[key] = rest.split()[0]
    except OSError:
        pass
    return out


def fmt_kb(v):
    try:
        return "%.2f GB" % (int(v) / 1024 / 1024)
    except (TypeError, ValueError):
        return "?"


def read_spans(fd, spans, buf):
    """Read the byte ranges of every stored weight. Do no arithmetic.

    Return the elapsed seconds. This is the floor for a kernel that keeps the
    weights packed: the kernel cannot finish before the bytes arrive. Keep the
    file descriptor open across the calls. On some file systems the page cache
    does not survive a close and a reopen.
    """
    view = memoryview(buf)
    t0 = time.time()
    for off, length in spans:
        done = 0
        while done < length:
            n = min(len(buf), length - done)
            done += os.preadv(fd, [view[:n]], off + done)
    return time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=3)
    ap.add_argument("--prefill", type=int, default=32)
    ap.add_argument("--modes", default="resident,stream")
    args = ap.parse_args()

    from np_gemma.ct import CompressedTensors
    from np_gemma.e4b import E4B, E4BConfig, E4BCache

    snap = args.snapshot
    path = os.path.join(snap, "model.safetensors")
    cfg = E4BConfig.load(os.path.join(snap, "config.json"))
    ct = CompressedTensors(path)
    mods = text_modules(cfg)

    # ---- 1. the sizes ------------------------------------------------------
    print("=== sizes ===")
    file_bytes = os.path.getsize(path)
    stored = {"table": 0, "matrix": 0, "norm": 0}
    logical = 0
    per_bits = {}
    per_dtype = {}
    for name, kind in mods:
        nbytes = ct.packed_bytes(name)
        stored[kind] += nbytes
        n = int(np.prod(ct._logical_shape(name)))
        logical += n * 4
        bits = ct.num_bits(name)
        if bits is None:
            dt = ct.dtype(name)
            per_dtype[dt] = per_dtype.get(dt, 0) + n
        else:
            per_bits[bits] = per_bits.get(bits, 0) + n
    per_token = stored["matrix"] + stored["norm"]
    print("  file                          %12d bytes  (%.3f GB)"
          % (file_bytes, file_bytes / 1e9))
    print("  text model, stored            %12d bytes  (%.3f GB)"
          % (sum(stored.values()), sum(stored.values()) / 1e9))
    print("    two embedding tables        %12d bytes  (%.3f GB)  read by row"
          % (stored["table"], stored["table"] / 1e9))
    print("    matrices and norms          %12d bytes  (%.3f GB)  read in full for a token"
          % (per_token, per_token / 1e9))
    print("  text model, as float32        %12d bytes  (%.3f GB)" % (logical, logical / 1e9))
    for bits in sorted(per_bits):
        print("    %2d-bit weights             %12d parameters" % (bits, per_bits[bits]))
    for dt in sorted(per_dtype):
        print("    %-4s weights                %12d parameters" % (dt, per_dtype[dt]))
    print("  the file also holds the vision tower, the audio tower, and the two")
    print("  input projections. The text model does not read them.")

    # ---- 1b. where the file lives -----------------------------------------
    # A memory map is only as good as the storage under it. A map over a
    # network file system turns every page fault into a network round trip.
    print("\n=== the storage under the file ===")
    stat = os.statvfs(path)
    real = os.path.realpath(path)
    mount = "unknown"
    try:
        best = ""
        with open("/proc/mounts") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                point = parts[1].replace("\\040", " ")
                if real.startswith(point) and len(point) > len(best):
                    best = point
                    mount = "%s on %s (%s)" % (parts[0], point, parts[2])
    except OSError:
        pass
    print("  %s" % mount)
    print("  free space on that mount: %.1f GB of %.1f GB"
          % (stat.f_bavail * stat.f_frsize / 1e9, stat.f_blocks * stat.f_frsize / 1e9))

    # ---- 2. the read floor ------------------------------------------------
    print("\n=== the read floor: read the stored bytes through the file ===")
    spans = []
    for name, kind in mods:
        t = ct.st.header[ct._stored_key(name)]
        o0, o1 = t["data_offsets"]
        spans.append((ct.st._base + o0, o1 - o0))
    matrix_spans = []
    for (name, kind), span in zip(mods, spans):
        if kind != "table":
            matrix_spans.append(span)
    buf = bytearray(8 << 20)
    total = sum(length for _off, length in spans)
    fd = os.open(path, os.O_RDONLY)
    cold = read_spans(fd, spans, buf)
    warm = read_spans(fd, spans, buf)
    warm_matrix = read_spans(fd, matrix_spans, buf)
    print("  whole text model, first read  %8.3f s for %.3f GB  (%.2f GB/s)"
          % (cold, total / 1e9, total / 1e9 / cold))
    print("  same file, second read        %8.3f s for %.3f GB  (%.2f GB/s)"
          % (warm, total / 1e9, total / 1e9 / warm))
    print("  per-token weights, second read%8.3f s for %.3f GB  (%.2f GB/s)"
          % (warm_matrix, per_token / 1e9, per_token / 1e9 / warm_matrix))
    print("  The first number is what the first token pays. A fused kernel that")
    print("  reads the packed data once for each token cannot beat the third")
    print("  number when the page cache holds the file.")

    # The memory map is the form the model uses in stream mode. A page fault
    # is the unit of work of a map, and over a network file system a fault can
    # cost a round trip. Take a fresh map and touch one byte in every page.
    import mmap as mmap_mod
    mm = mmap_mod.mmap(fd, 0, access=mmap_mod.ACCESS_READ)
    step = 4096
    t0 = time.time()
    acc = 0
    pages = 0
    views = []
    for off, length in spans:
        v = np.frombuffer(mm, dtype=np.uint8, count=length, offset=off)
        views.append(v)
        acc += int(v[::step].sum())
        pages += (length + step - 1) // step
    fault_t = time.time() - t0
    print("  memory map, one touch per page %8.3f s for %d pages (%.0f pages/s)"
          % (fault_t, pages, pages / fault_t))
    print("  that is the map equivalent of the first read above, at %d KiB a page."
          % (step // 1024))
    views.clear()
    try:
        mm.close()
    except BufferError:
        # A view still points into the map. The process ends soon; the kernel
        # releases the map then.
        pass
    os.close(fd)

    # ---- 3. the decode work on its own ------------------------------------
    print("\n=== the decode work for one pass over the matrices and norms ===")
    t0 = time.time()
    written = 0
    for name, kind in mods:
        if kind == "table":
            continue
        a = ct.dequant(name) if kind == "matrix" else ct.plain(name)
        written += a.nbytes
    dt = time.time() - t0
    print("  %.2f s to decode the per-token weights once (%.2f GB of float32)"
          % (dt, written / 1e9))
    print("  that is %.1f GB/s of float32 written" % (written / 1e9 / dt))
    print("  a streamed decode pays this cost again for every token")

    # ---- 4. the modes -----------------------------------------------------
    ids = [2, 105, 2364, 107, 818, 5279, 529, 7001, 563, 106, 107, 105, 4368, 107]
    prompt = (ids * (args.prefill // len(ids) + 1))[:args.prefill]
    map_mode = {"resident": "f32", "stream": "stream", "int4": "int4"}
    results = {}
    for mode in args.modes.split(","):
        if mode not in map_mode:
            print("unknown mode %r" % mode)
            continue
        print("\n=== mode %s ===" % mode)
        ct2 = CompressedTensors(path)
        model = E4B(ct2, cfg, mode=map_mode[mode])
        before = mem_report()
        t0 = time.time()
        cache = E4BCache(cfg)
        hidden = model.forward(prompt, cache=cache, start_pos=0)
        prefill_t = time.time() - t0
        after = mem_report()
        print("  prefill %d tokens: %.2f s (%.3f s/token)"
              % (len(prompt), prefill_t, prefill_t / len(prompt)))
        print("  Rss %s -> %s   Private_Dirty %s -> %s   FilePmdMapped %s -> %s"
              % (fmt_kb(before.get("Rss")), fmt_kb(after.get("Rss")),
                 fmt_kb(before.get("Private_Dirty")), fmt_kb(after.get("Private_Dirty")),
                 fmt_kb(before.get("FilePmdMapped")), fmt_kb(after.get("FilePmdMapped"))))
        pos = len(prompt)
        step_times = []
        for _ in range(args.max_new_tokens):
            # Time the whole step: the output head and the 42 layers.
            t0 = time.time()
            logits = model.logits(hidden[-1:])[0]
            nxt = int(np.argmax(logits))
            hidden = model.forward([nxt], cache=cache, start_pos=pos)
            step_times.append(time.time() - t0)
            pos += 1
        dec = sum(step_times) / max(1, len(step_times))
        print("  decode: %.3f s/token over %d tokens" % (dec, len(step_times)))
        print("  each token: %s" % "  ".join("%.2f" % t for t in step_times))
        read_bytes = written if map_mode[mode] == "f32" else per_token
        print("  reads %.2f GB of %s for each token: %.2f GB/s"
              % (read_bytes / 1e9,
                 "float32" if map_mode[mode] == "f32" else "packed data",
                 read_bytes / 1e9 / dec))
        results[mode] = (prefill_t / len(prompt), dec)
        del model
        ct2.close()
        import gc
        gc.collect()

    # ---- 5. the verdict ---------------------------------------------------
    print("\n=== summary ===")
    print("  %-10s %14s %14s" % ("mode", "prefill s/tok", "decode s/tok"))
    for mode, (pt, dtok) in results.items():
        print("  %-10s %14.3f %14.3f" % (mode, pt, dtok))
    if "resident" in results and "stream" in results:
        pr, dr = results["resident"]
        ps, ds = results["stream"]
        print("\n  stream prefill overhead: %+.3f s/token (%+.0f%%)"
              % (ps - pr, 100 * (ps - pr) / pr if pr else 0))
        print("  stream decode overhead:  %+.3f s/token (%+.0f%%)"
              % (ds - dr, 100 * (ds - dr) / dr if dr else 0))
        print("  stream holds one layer of memory; resident holds %.1f GB"
              % (logical / 1e9))
    return 0


if __name__ == "__main__":
    sys.exit(main())
