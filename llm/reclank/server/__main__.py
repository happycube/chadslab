"""The server of reclank: python -m server --model PATH [options] (see
server/options.py). The family of the model comes from the file (the GGUF
general.architecture): Gemma 4, Qwen3.6, or Qwen3.8; or of a checkpoint directory
(its config.json): DeepSeek V4.1 Flash."""
from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import http as H  # noqa: E402
from server.backend import Backend  # noqa: E402
from server.options import parser  # noqa: E402
from server.worker import ModelWorker  # noqa: E402

FAMILIES = ("gemma4", "qwen", "deepseek41", "nemotron")


def family_module(name):
    if name == "gemma4":
        from reclank.models.gemma4 import engine
    elif name == "deepseek41":
        from reclank.models.deepseek41 import engine
    elif name == "nemotron":
        from reclank.models.nemotron import engine
    else:
        from reclank.models.qwen import engine
    return engine


def detect(opts):
    """The family of the model file (or --family) and its metadata: the GGUF
    metadata, or the config.json of a checkpoint directory."""
    if os.path.isdir(opts.model):
        from reclank.models.deepseek41.engine import load_meta
        meta = load_meta(opts.model)
    else:
        from reclank.formats.gguf import GGUF
        meta = GGUF(opts.model, stage=False).meta      # the metadata only: no copy of a file on Optane
    if opts.family != "auto":
        return opts.family, meta
    for name in FAMILIES:
        if family_module(name).detect(meta):
            return name, meta
    raise SystemExit("%s: no family takes the architecture %r" % (
        opts.model, meta.get("general.architecture", meta.get("model_type"))))


def build(opts):
    """Load the model on the model thread; return the Backend."""
    family, meta = detect(opts)
    mod = family_module(family)
    worker = ModelWorker()
    if family == "gemma4":
        engine, d = worker.load(lambda: mod.load(opts, meta))
    else:
        engine, d = worker.load(lambda: mod.load(opts, meta, worker))
    worker.engine = engine
    if len(getattr(engine, "lanes", None) or [engine]) > 1:
        from server.worker import LaneScheduler
        worker.lanes = LaneScheduler(engine.lanes)
    pick = lambda v, k: d.get(k) if v is None else v  # noqa: E731
    backend = Backend(engine, worker, model_id=opts.model_id or d.get("model_id", "reclank"),
                      thinking=d.get("thinking"), max_tokens=pick(opts.max_tokens, "max_tokens"),
                      temperature=pick(opts.temperature, "temperature"), top_k=pick(opts.top_k, "top_k"),
                      top_p=pick(opts.top_p, "top_p"), mtp_accept=opts.mtp_accept,
                      mtp_floor=opts.mtp_floor, media_dir=opts.media_dir,
                      repetition_penalty=d.get("repetition_penalty"))
    backend.no_think_below = max(0, opts.no_think_below)
    backend.max_floor = opts.max_tokens_floor
    backend.floor_min = opts.max_tokens_floor_min
    print("%s: sampling temperature=%s top_p=%s top_k=%s%s" % (
        family, backend.temperature, backend.top_p, backend.top_k,
        " repetition_penalty=%s" % backend.repetition_penalty if backend.repetition_penalty else ""),
        flush=True)
    if opts.debug:
        os.makedirs(opts.debug, exist_ok=True)
        backend.debug = opts.debug
        H.RAW_DIR = opts.debug       # the raw answers (last-answer.txt, empty-*.txt)
    return backend


def main(argv=None):
    opts = parser().parse_args(argv)
    backend = build(opts)
    H.serve(backend, host=opts.host, port=opts.port, quiet=opts.quiet, api_key=opts.api_key,
            cors_origin=opts.cors_origin, max_body=int(opts.max_body_mb * (1 << 20)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
