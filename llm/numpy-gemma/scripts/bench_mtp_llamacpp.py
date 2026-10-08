"""Measure the MTP drafter of Gemma 4 in llama.cpp for each drafter quant.

The script starts llama-server one time for each configuration. It sends the
same prompts with greedy selection. It prints the decode rate, the drafts, the
accepted drafts, and a check that the text is the same as the text without a
drafter. Speculative decoding with greedy selection must not change the text.

Run from the numpy-gemma directory:

    python3 scripts/bench_mtp_llamacpp.py
    python3 scripts/bench_mtp_llamacpp.py --quants q4_0 mxfp4 --n-max 2 3 4

Another target and its drafter, with the CUDA build and all layers on the
GPU:

    python3 scripts/bench_mtp_llamacpp.py --target E4B.gguf \
        --drafter models/assistants/gemma-4-E4B-it-assistant-{}.gguf --quants bf16 \
        --server ../llama.cpp/build-cuda/bin/llama-server --extra="-ngl 99"

The script uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER = os.path.join(ROOT, "..", "llama.cpp", "build", "bin", "llama-server")
TARGET = os.path.join(ROOT, "models", "gemma-4-26B-qat-q4_0", "gemma-4-26B_q4_0-it.gguf")
DRAFTER = os.path.join(ROOT, "models", "assistants",
                       "gemma-4-26B-A4B-it-qat-q4_0-unquantized-assistant-{}.gguf")
PORT = 18089

PROMPTS = [
    ("code", "Write a Python function that returns the n-th Fibonacci number, "
             "then explain it in two sentences."),
    ("prose", "Explain why the sky is blue in one paragraph."),
    ("list", "List the planets of the solar system with one fact about each."),
    ("math", "Solve 17 * 23 step by step, then check the result by division."),
]


def post(path, body):
    req = urllib.request.Request("http://127.0.0.1:%d%s" % (PORT, path),
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def wait_ready(proc):
    for _ in range(300):
        if proc.poll() is not None:
            raise RuntimeError("llama-server stopped; see the log")
        try:
            with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("llama-server did not start")


# The fields of each request beyond the messages (main sets them).
BODY = {}


def run(label, extra, max_tokens, log, server=SERVER, target=TARGET):
    cmd = [server, "-m", target, "-c", "4096", "--port", str(PORT)] + extra
    with open(log, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
    try:
        wait_ready(proc)
        rows = []
        for name, text in PROMPTS:
            d = post("/v1/chat/completions", {
                "messages": [{"role": "user", "content": text}],
                "max_tokens": max_tokens, "temperature": 0, **BODY})
            t = d["timings"]
            rows.append((name, d["choices"][0]["message"]["content"],
                         t["predicted_n"], t["predicted_ms"],
                         t.get("draft_n", 0), t.get("draft_n_accepted", 0)))
        return rows
    finally:
        proc.terminate()
        proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quants", nargs="+",
                    default=["bf16", "q8_0", "q4_0", "mxfp4", "nvfp4"])
    ap.add_argument("--n-max", nargs="+", type=int, default=[3])
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--log", default="/tmp/bench_mtp_llamacpp.log")
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--drafter", default=DRAFTER, help="The path, with {} for the quant.")
    ap.add_argument("--server", default=SERVER)
    ap.add_argument("--extra", default="", help="More arguments of llama-server.")
    ap.add_argument("--no-think", action="store_true",
                    help="enable_thinking false. The template of Gemma 4 in llama-server turns"
                         " thinking on by default, and scripts/check_mtp.py runs with it off.")
    args = ap.parse_args()
    if args.no_think:
        BODY["chat_template_kwargs"] = {"enable_thinking": False}
    more = args.extra.split()

    configs = [("none", "-", list(more))]
    for q in args.quants:
        path = args.drafter.format(q)
        if not os.path.exists(path):
            print("skip %s: %s is missing" % (q, path))
            continue
        for n in args.n_max:
            configs.append((q, str(n), ["-md", path, "--spec-type", "draft-mtp",
                                        "--spec-draft-n-max", str(n)] + more))

    base = None
    print("%-6s %3s  %-6s %8s %7s %9s  %s" % ("draft", "n", "prompt", "tok/s",
                                              "drafts", "accepted", "same text"))
    for q, n, extra in configs:
        rows = run(q, extra, args.max_tokens, args.log, args.server, args.target)
        if base is None:
            base = {r[0]: r[1] for r in rows}
        tok = ms = dn = da = 0
        for name, text, pn, pms, drn, dra in rows:
            same = "yes" if text == base[name] else "NO"
            acc = "%d (%d%%)" % (dra, 100 * dra // drn) if drn else "-"
            print("%-6s %3s  %-6s %8.2f %7s %9s  %s" % (q, n, name, 1000 * pn / pms,
                                                        drn or "-", acc, same))
            tok, ms, dn, da = tok + pn, ms + pms, dn + drn, da + dra
        acc = "%d (%d%%)" % (da, 100 * da // dn) if dn else "-"
        print("%-6s %3s  %-6s %8.2f %7s %9s" % (q, n, "ALL", 1000 * tok / ms,
                                                dn or "-", acc))
        print(flush=True)


if __name__ == "__main__":
    main()
