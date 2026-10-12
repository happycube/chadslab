"""The options of the server: one set for every family (REFACTOR_PLAN.md,
section 5.3). An option that a family does not use has no effect there; a
value of None takes the default of the family (the engine gives it)."""
from __future__ import annotations

import argparse
import os

from server.http import DEFAULT_API_KEY, MAX_BODY

THINKING = ("auto", "on", "off", "low", "medium", "high", "xhigh")


def parser():
    ap = argparse.ArgumentParser(prog="python -m server", description=(
        "An OpenAI compatible server for the models of reclank (Gemma 4, Qwen3.6, Qwen3.8). "
        "The family comes from the model file; --family forces one."))
    g = ap.add_argument_group("the model")
    g.add_argument("--model", "-m", "--gguf", dest="model", required=True,
                   help="the GGUF file of the model")
    g.add_argument("--family", choices=("auto", "gemma4", "qwen"), default="auto")
    g.add_argument("--tokenizer", "--tok", dest="tokenizer", default=None,
                   help="tokenizer.json (Gemma: the GGUF data by default; Qwen: next to the model)")
    g.add_argument("--template", default=None,
                   help="chat_template.jinja (Qwen: next to the model, else in the GGUF)")
    g.add_argument("--dtype", default=None, help="Gemma: the weight mode (int4, f32)")
    g.add_argument("--kv-attn", default=None,
                   choices=["int16", "int8", "k16v8", "rq8", "k16vr8", "float"],
                   help="Gemma: the form of the key and value cache (default int16)")
    g.add_argument("--experts", default=None,
                   help="Qwen3.8: a file of the routed experts in another form "
                        "(scripts/convert_q8_gguf.py --experts-only); default: the one "
                        "*nvfp4-experts*.gguf next to the model; none: those of the model")
    g = ap.add_argument_group("where it runs")
    g.add_argument("--backend", choices=("auto", "cpu", "cuda"), default="auto",
                   help="auto: Gemma on the CPU, Qwen on the GPU (the defaults of before)")
    g.add_argument("--hot-gb", "--gpu-experts-gb", dest="hot_gb", type=float, default=None,
                   help="the GPU memory of the hot experts (default: the free memory after the "
                        "rest)")
    g.add_argument("--moe-rows", choices=("auto", "all", "off"), default=None,
                   help="Qwen on the GPU: the cold experts give their rows and the GPU adds every "
                        "expert in the order of the router (the same bits whatever experts the GPU "
                        "holds): auto (the default) for steps, verify groups and prompt groups of up "
                        "to 64 tokens; all for every group; off (RECLANK_GPU_MOE_ROWS)")
    g.add_argument("--resident", action="store_true",
                   help="every weight on the GPU: no experts on the CPU, no copies of experts "
                        "(an error when they do not fit)")
    g.add_argument("--gpu-dense", action="store_true",
                   help="Gemma 26B: only the weights outside the experts on the GPU (no hot experts)")
    g.add_argument("--ctx", "--max-context", dest="ctx", type=int, default=None,
                   help="the tokens of the context (Qwen: the cache, default 98304; Gemma: the "
                        "most tokens of a request, default no limit)")
    g = ap.add_argument_group("the answer")
    g.add_argument("--max-tokens", type=int, default=None, help="the default of a request")
    g.add_argument("--max-tokens-floor", type=int, default=0,
                   help="the least max_tokens of a request of at least --max-tokens-floor-min")
    g.add_argument("--max-tokens-floor-min", type=int, default=8192)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--top-k", type=int, default=None)
    g.add_argument("--top-p", type=float, default=None)
    g.add_argument("--thinking", nargs="?", const="on", choices=THINKING, default=None,
                   help="the think part by default (Gemma: auto, on, off; Qwen: low, medium, "
                        "xhigh, off); a request can set it")
    g.add_argument("--think-budget", type=int, default=6000,
                   help="the most tokens of a think part (0: no limit); at most half of the "
                        "max_tokens of the request")
    g.add_argument("--wrap-left", type=int, default=500,
                   help="when fewer tokens of the context are left, words that tell the model to "
                        "wrap up (0: none)")
    g.add_argument("--no-think-below", type=int, default=500,
                   help="a request with a smaller max_tokens answers with no think part (0: no rule)")
    g = ap.add_argument_group("MTP")
    g.add_argument("--drafter", default=None, metavar="DIR",
                   help="Gemma: the assistant model (the MTP drafter)")
    g.add_argument("--mtp", type=int, default=None,
                   help="the drafts of each MTP step (Gemma: 2 with --drafter, 3 with the drafter on "
                        "the GPU and RECLANK_MTP_STOP 0.9, 2 when less than 1.5 GB of the GPU is left after "
                        "the weights, 0 on a card of less than 8 GB; Qwen3.8: 3; 0: off, no drafter)")
    g.add_argument("--mtp-dtype", choices=("int4", "int8", "f32"), default=None)
    g.add_argument("--mtp-accept", choices=("exact", "in_set"), default="exact")
    g.add_argument("--mtp-floor", type=float, default=None)
    g = ap.add_argument_group("media")
    g.add_argument("--mmproj", default=None,
                   help="the media encoder (Gemma: the mmproj GGUF; Qwen3.6: the mmproj GGUF; "
                        "Qwen3.8: the checkpoint directory)")
    g.add_argument("--mmproj-q8", choices=("auto", "on", "off"), default=None)
    g.add_argument("--mmproj-gpu", choices=("auto", "keep", "reserve", "lend", "release", "off"),
                   default=None, help="Qwen: the encoder on the GPU")
    g.add_argument("--image-budget", type=int, default=None)
    g.add_argument("--video-frames", type=int, default=None)
    g.add_argument("--video-budget", type=int, default=None)
    g.add_argument("--media-dir", default=None)
    g = ap.add_argument_group("the server")
    g.add_argument("--host", default="127.0.0.1")
    g.add_argument("--port", type=int, default=8080)
    g.add_argument("--model-id", default=None)
    g.add_argument("--api-key", default=os.environ.get("RECLANK_API_KEY", DEFAULT_API_KEY),
                   help="the key of each request (Authorization: Bearer KEY); default "
                        "RECLANK_API_KEY, else %r; \"\" turns the check off" % DEFAULT_API_KEY)
    g.add_argument("--cors-origin", default=None)
    g.add_argument("--max-body-mb", type=float, default=MAX_BODY / (1 << 20))
    g.add_argument("--debug", default=None, metavar="DIR",
                   help="write each turn (the request, the prompt, the raw answer, its parts) to DIR")
    g.add_argument("--crash-dir", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "reclank-crash"),
                   help="a fatal error (a lost CUDA context) writes a report here ('': none)")
    g.add_argument("--quiet", action="store_true")
    return ap
