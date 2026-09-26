#!/usr/bin/env python3
"""Run an OpenAI compatible server for a GGUF model.

    python scripts/serve.py --gguf PATH [--host 127.0.0.1] [--port 8080]
                            [--dtype int4] [--temperature 0.0] [--top-k 40]

Point an OpenAI client at http://127.0.0.1:8080/v1 . The model id is the file
name. Use --thinking to open the thought channel of the Gemma 4 chat template.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import Model, Tokenizer  # noqa: E402
from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma.server import Backend, serve  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="A tokenizer.json file. The GGUF data is the default.")
    ap.add_argument("--dtype", default="int4")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--model-id", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override the temperature. The default is the value that the "
                         "model file recommends, or 1.0. Zero selects the best token.")
    ap.add_argument("--top-k", type=int, default=None,
                    help="Override top_k. The default is the model recommendation.")
    ap.add_argument("--top-p", type=float, default=None,
                    help="Override top_p. The default is the model recommendation.")
    ap.add_argument("--thinking", action="store_true",
                    help="Open the thought channel of the chat template.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--kv-attn", choices=["float", "q8"], default="float",
                    help="Attention over the key and value cache. float is exact."
                         " q8 is about 1.6 times faster and less accurate, which can"
                         " make a long greedy generation repeat itself.")
    ap.add_argument("--mtp", default=None, metavar="DIR",
                    help="The snapshot directory of the Gemma 4 assistant model (the"
                         " MTP drafter). See MTP_DRAFTER_QUANT.md, step 1.")
    ap.add_argument("--mtp-n", type=int, default=2,
                    help="The count of drafts for each MTP step.")
    ap.add_argument("--mtp-dtype", choices=("int4", "int8", "f32"), default="int4")
    args = ap.parse_args()

    # The int8 cache quantizes the keys and the values. It is faster and it
    # changes the hidden state by about 0.3 per cent, which is enough to send a
    # long greedy generation into a loop. Use it only when it is asked for.
    os.environ["NP_GEMMA_ATTN"] = "1" if args.kv_attn == "q8" else "0"

    g = GGUF(args.gguf)
    tok = Tokenizer(args.tokenizer) if args.tokenizer else Tokenizer.from_gguf(g)
    cfg = Config({"text_config": g.text_config()})
    print("loading %s ..." % args.gguf, flush=True)
    model = Model(g, cfg).load_all(dtype=args.dtype)
    model_id = args.model_id or os.path.basename(args.gguf).rsplit(".", 1)[0]
    # The model file gives the sampling that the model wants. Use it unless the
    # caller asks for another value. A greedy default is not the wish of the
    # model and it hides the setting from the client.
    rec = {k: g.meta.get("general.sampling." + k) for k in ("temp", "top_p", "top_k")}
    temperature = args.temperature if args.temperature is not None else \
        float(rec["temp"] if rec["temp"] is not None else 1.0)
    top_p = args.top_p if args.top_p is not None else \
        (float(rec["top_p"]) if rec["top_p"] is not None else None)
    top_k = args.top_k if args.top_k is not None else \
        (int(rec["top_k"]) if rec["top_k"] is not None else None)
    print("sampling temperature=%s top_p=%s top_k=%s" % (temperature, top_p, top_k),
          flush=True)
    drafter = None
    if args.mtp:
        from np_gemma.assistant import Assistant
        print("loading the MTP drafter %s ..." % args.mtp, flush=True)
        drafter = Assistant(args.mtp, dtype=args.mtp_dtype)
    backend = Backend(model, tok, cfg, model_id=model_id, thinking=args.thinking,
                      max_tokens=args.max_tokens, temperature=temperature,
                      top_k=top_k, top_p=top_p, drafter=drafter, n_draft=args.mtp_n)
    serve(backend, host=args.host, port=args.port, quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
