"""reclank: an LLM runtime in NumPy, C, and CUDA (Gemma 4, Qwen3.6, Qwen3.8).

The package (REFACTOR_PLAN.md):

    runtime/   threads, NUMA          formats/   GGUF, safetensors, NVFP4, TQ6
    ir/        programs, compiler     core/      ops, sampling, MTP, think guard
    encoders/  media encoders         backends/  cpu (cops), cuda (runtime, gpumm)
    models/    gemma4, qwen           server.py  the HTTP server

The names below (Model, Tokenizer, ...) load on first use. The old flat
module names (reclank.model, reclank.gpu, ...) still import (_compat.py).
"""
# First: the old setting names (NP_GEMMA_*) as RECLANK_* (reclank/settings.py).
from . import settings  # noqa: F401
# Then the thread counts and the NUMA policy, before NumPy and OpenMP make
# their threads (runtime/threads.py).
from .runtime import threads as _threads
from . import _compat

_threads.setup()
_compat.install()

_EXPORTS = {
    "Config": "models.gemma4.config", "SafeTensors": "formats.safetensors",
    "CompressedTensors": "models.gemma4.ct", "E4B": "models.gemma4.e4b",
    "E4BConfig": "models.gemma4.e4b", "E4BCache": "models.gemma4.e4b",
    "Model": "models.gemma4.model", "KVCache": "models.gemma4.model",
    "Session": "models.gemma4.model", "Sampler": "core.sampling",
    "Tokenizer": "models.gemma4.tokenizer",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    import importlib
    mod = _EXPORTS.get(name)
    if mod is not None:
        value = getattr(importlib.import_module("reclank." + mod), name)
        globals()[name] = value
        return value
    if name in _compat.OLD:
        return importlib.import_module("reclank." + name)
    raise AttributeError("module 'reclank' has no attribute %r" % name)
