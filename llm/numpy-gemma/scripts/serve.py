#!/usr/bin/env python3
"""Run an OpenAI compatible server for a GGUF model.

    python scripts/serve.py --gguf PATH [--host 127.0.0.1] [--port 8080]
                            [--dtype int4] [--temperature 0.0] [--top-k 40]

Point an OpenAI client at http://127.0.0.1:8080/v1 . The model id is the file
name. The thought channel of the Gemma 4 chat template is open by default
(--thinking auto), but not when --max-context is less than 32768.
A request sets it with "thinking" (true, false, or {"type": "enabled"}),
"reasoning_effort", "reasoning": {"effort": ...}, or "chat_template_kwargs":
{"enable_thinking": ...}.

The server takes the 26B and 12B models (Model) and the E2B and E4B models
(E4B). It finds the kind from the GGUF data: the E2B and E4B models have
inputs for each layer.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from np_gemma import Model, Tokenizer  # noqa: E402
from np_gemma.config import Config  # noqa: E402
from np_gemma.gguf import GGUF  # noqa: E402
from np_gemma import server as SV  # noqa: E402
from np_gemma.server import Backend, serve  # noqa: E402


# The words that close a thought channel at --think-budget (then <channel|>),
# and when fewer than --wrap-left tokens of the context are left: in the
# thought channel (then <channel|>), and in the answer.
THINK_CLOSE = "\n\nI have thought about this long enough; now I write the answer.<channel|>"
WRAP_THINK = "\n\nThe context is almost full, so I stop thinking and answer briefly now.<channel|>"
WRAP_ANSWER = "\n\n(The context is almost full, so I wrap up briefly now.)\n\n"

# The least --max-context for the thought channel by default (--thinking auto).
THINK_MIN_CONTEXT = 32768


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
    ap.add_argument("--thinking", nargs="?", const="on", choices=("auto", "on", "off"),
                    default="auto",
                    help="Open the thought channel of the chat template for a request that"
                         " does not set it. The Gemma 4 template has one level (on). auto:"
                         " on, but off when --max-context is less than %d (the thought"
                         " part takes room)." % THINK_MIN_CONTEXT)
    ap.add_argument("--think-budget", type=int, default=6000,
                    help="The most tokens of a thought channel: then the server closes it"
                         " (THINK_CLOSE) and the model writes the answer (0: no limit); at most"
                         " half of the max_tokens of the request too.")
    ap.add_argument("--wrap-left", type=int, default=500,
                    help="When fewer than this many tokens of the context are left, tell the model"
                         " to wrap up (WRAP_THINK, WRAP_ANSWER; 0: none).")
    ap.add_argument("--no-think-below", type=int, default=500,
                    help="A request with max_tokens below this answers with no think part (a"
                         " client's title or a short check; the thought took all of its tokens)."
                         " 0: no rule.")
    ap.add_argument("--mmproj", default=None, metavar="GGUF",
                    help="The mmproj GGUF of the model: image, video, and audio input"
                         " (the Gemma 4 12B: gemma4uv and gemma4ua; the E2B, E4B, and 26B:"
                         " gemma4v, and gemma4a on the E2B and E4B).")
    ap.add_argument("--mmproj-q8", choices=("auto", "on", "off"), default="auto",
                    help="Q8_0 weights in the image and audio encoders (half the bytes of"
                         " bfloat16; the same answers on the checks). auto: on for the E2B"
                         " and E4B, off for the others.")
    ap.add_argument("--image-budget", type=int, default=280,
                    help="The most soft tokens of an image: 70, 140, 280, 560, or 1120. A"
                         " request sets it with image_url.detail (low 70, high 1120).")
    ap.add_argument("--video-frames", type=int, default=32,
                    help="The frames of a video (evenly spaced, as transformers).")
    ap.add_argument("--video-budget", type=int, default=70,
                    help="The soft tokens of a video frame: 70, 140, 280, 560, or 1120.")
    ap.add_argument("--media-dir", default=None, metavar="DIR",
                    help="A directory of local media files that a request can name (a path"
                         " or file:// URL). Without it, media come only as data URIs.")
    ap.add_argument("--debug", default=None, metavar="DIR",
                    help="Write each turn to DIR/NNNNN-<time>.json: the request, the prompt"
                         " text, the think setting, the raw answer, its parts, and the tool"
                         " calls. DIR/turns.log gets a line for each turn.")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--kv-attn", choices=["int16", "int8", "k16v8", "rq8", "k16vr8", "float"], default="int16",
                    help="Attention over the key and value cache. int16 reads an int16"
                         " copy of the cache with a float query. Its error is about 4e-5"
                         " of the attention output, and at a long context it reads half"
                         " the bytes of float. int8 (NP_GEMMA_KV_INT8=1) keeps only int8"
                         " keys and values: half the bytes of int16. k16v8"
                         " (NP_GEMMA_KV_INT8=v) keeps int16 keys and int8 values. The"
                         " cache keeps no float rows; float runs the float attention of"
                         " Python over dequantized rows.")
    ap.add_argument("--mtp", default=None, metavar="DIR",
                    help="The snapshot directory of the Gemma 4 assistant model (the"
                         " MTP drafter). See MTP_DRAFTER_QUANT.md, step 1.")
    ap.add_argument("--mtp-n", type=int, default=2,
                    help="The count of drafts for each MTP step.")
    ap.add_argument("--mtp-dtype", choices=("int4", "int8", "f32"), default="int4")
    ap.add_argument("--mtp-accept", choices=("exact", "in_set"), default="exact",
                    help="The rule of the MTP drafts with sampling. exact keeps a draft only"
                         " when the sample picks it (the text of the plain decode). in_set also"
                         " keeps a draft that top_k, top_p, min_p, and --mtp-floor allow: more"
                         " drafts stay, but the text moves toward the drafter. A request can"
                         " give mtp_accept.")
    ap.add_argument("--mtp-floor", type=float, default=None,
                    help="With in_set: keep a draft only when its probability is at least this"
                         " share of the best probability. A request can give mtp_floor.")
    ap.add_argument("--gpu", choices=("off", "dense", "hot"), default="off",
                    help="dense puts the weights outside the experts and the output head on"
                         " the GPU; the experts stay on the CPU. hot also puts the most used"
                         " experts on the GPU. For the E4B, dense and hot put the whole model"
                         " on the GPU. Needs nvcc. --mtp turns MTP on (NP_GEMMA_MTP=0 turns"
                         " it off).")
    ap.add_argument("--max-context", type=int, default=None,
                    help="The most tokens of a request, the prompt and the answer. A longer"
                         " prompt gets an error (400), and max_tokens is cut to fit. The"
                         " default has no limit (the model has 262144).")
    ap.add_argument("--gpu-experts-gb", type=float, default=None,
                    help="With --gpu hot, the GPU memory for the experts. The default is the"
                         " free memory less 6 GB.")
    SV.add_http_args(ap)
    args = ap.parse_args()

    # The cache copy was int8 with an int8 query. Its error sent a long greedy
    # generation into a loop, so the server used the float cache. The copy is
    # now int16 with a float query, which is as accurate as the float cache
    # and faster at a long context.
    os.environ["NP_GEMMA_ATTN"] = "0" if args.kv_attn == "float" else "1"
    if args.kv_attn in ("int8", "k16v8", "rq8", "k16vr8"):
        from np_gemma import model as _model
        _model.KV_FORM = args.kv_attn

    g = GGUF(args.gguf)
    tok = Tokenizer(args.tokenizer) if args.tokenizer else Tokenizer.from_gguf(g)
    tc = g.text_config()
    e4b = bool(tc.get("hidden_size_per_layer_input"))
    print("loading %s ..." % args.gguf, flush=True)
    if e4b:
        from np_gemma.e4b import E4B, E4BConfig
        cfg = E4BConfig({"text_config": tc})
        model = E4B(g, cfg, mode="int4" if args.dtype == "int4" else "f32")
        if args.mtp and args.gpu != "off":
            # The E4B with its drafter on the GPU is faster with MTP than
            # without it (SPLIT_PLAN.md, phase 5).
            os.environ.setdefault("NP_GEMMA_MTP", "1")
    else:
        cfg = Config({"text_config": tc})
        model = Model(g, cfg).load_all(dtype=args.dtype)
    model_id = args.model_id or os.path.basename(args.gguf).rsplit(".", 1)[0]
    if args.gpu != "off":
        from np_gemma import gpu
        print("copying the weights to the GPU ...", flush=True)
        dev = gpu.offload(model, 0.0 if args.gpu == "dense" else args.gpu_experts_gb)
        print(gpu.describe(dev), flush=True)
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
    if args.mtp and "NP_GEMMA_MTP" not in os.environ:
        # With a GPU the default of assistant.mtp_enabled is off; the drafter
        # was asked for, so turn MTP on.
        os.environ["NP_GEMMA_MTP"] = "1"
    if args.mtp:
        print("loading the MTP drafter %s ..." % args.mtp, flush=True)
        if args.gpu != "off":
            # The drafter on the GPU reads the cache on the GPU.
            from np_gemma.gpu import GPUDrafter
            drafter = GPUDrafter(args.mtp, model)
        else:
            from np_gemma.assistant import Assistant
            drafter = Assistant(args.mtp, dtype=args.mtp_dtype)
    thinking = args.thinking == "on" or (
        args.thinking == "auto" and (args.max_context is None or args.max_context >= THINK_MIN_CONTEXT))
    print("thinking %s by default (--thinking %s)" % ("on" if thinking else "off", args.thinking),
          flush=True)
    backend = Backend(model, tok, cfg, model_id=model_id, thinking=thinking,
                      max_tokens=args.max_tokens, temperature=temperature,
                      top_k=top_k, top_p=top_p, drafter=drafter, n_draft=args.mtp_n,
                      empty_thought_block=not e4b, max_context=args.max_context,
                      mtp_accept=args.mtp_accept, mtp_floor=args.mtp_floor)
    backend.no_think_below = args.no_think_below
    # the thought channel of Gemma 4 (<|channel>thought ... <channel|>):
    # the budget of a think part and the words to wrap up (np_gemma/think_guard.py)
    one = lambda t: tok.encode(t)[0] if len(tok.encode(t)) == 1 else None  # noqa: E731
    backend.think_budget, backend.wrap_left = args.think_budget, max(0, args.wrap_left)
    if one("<|channel>") is not None and one("<channel|>") is not None:
        backend.think_tokens = {
            "open": one("<|channel>"), "close": one("<channel|>"),
            "force": tok.encode(THINK_CLOSE), "wrap_think": tok.encode(WRAP_THINK),
            "wrap_answer": tok.encode(WRAP_ANSWER),
            "tool_open": one("<|tool_call>"), "tool_close": one("<tool_call|>")}
    if args.mmproj:
        from np_gemma.gemma4_encoders import load_embedder
        # The encoders on the GPU when the model is (NP_GEMMA_MEDIA_GPU=0: CPU).
        media_gpu = args.gpu != "off" and os.environ.get("NP_GEMMA_MEDIA_GPU", "1") != "0"
        q8 = args.mmproj_q8 == "on" or (args.mmproj_q8 == "auto" and e4b)
        backend.embedder = load_embedder(args.mmproj, gpu=media_gpu, q8=q8)
        backend.image_budget = args.image_budget
        backend.media_dir = args.media_dir
        backend.video_frames = args.video_frames
        backend.video_budget = args.video_budget
        print("media input: %s (images%s), image budget %d%s" % (
            args.mmproj, " and audio" if backend.embedder.has_audio else "",
            args.image_budget, ", Q8_0 encoder weights" if getattr(backend.embedder, "q8", False)
            else ""), flush=True)
    if args.debug:
        os.makedirs(args.debug, exist_ok=True)
        backend.debug = args.debug
        SV.RAW_DIR = args.debug          # the raw answers (last-answer.txt, empty-*.txt)
    serve(backend, host=args.host, port=args.port, quiet=args.quiet, api_key=args.api_key,
          cors_origin=args.cors_origin, max_body=int(args.max_body_mb * (1 << 20)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
