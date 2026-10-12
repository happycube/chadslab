"""The old flat module names of the package (before the layout of
REFACTOR_PLAN.md, phase P2): `import reclank.model` gives the module
reclank.models.gemma4.model (the same object). A module that was split in
two (program, gpu, gemma4_encoders) gives a module that reads its names from
both parts; a name set on it does not reach them. This shim goes in phase P9.
"""
import importlib
import importlib.abc
import importlib.util
import sys
import types

OLD = {
    "numa": "runtime.numa", "gguf": "formats.gguf", "st": "formats.safetensors",
    "st_qwen4": "formats.nvfp4_st", "tq6": "formats.tq6", "ops": "core.ops",
    "numba_ops": "core.numba_ops", "sampling": "core.sampling", "speculative": "core.speculative",
    "think_guard": "core.think_guard", "weight_cache": "core.weight_cache",
    "media": "encoders.media", "cops": "backends.cpu.cops", "gpumm": "backends.cuda.gpumm",
    "model": "models.gemma4.model", "e4b": "models.gemma4.e4b", "unified": "models.gemma4.unified",
    "parts": "models.gemma4.parts", "prompt": "models.gemma4.prompt",
    "assistant": "models.gemma4.assistant", "config": "models.gemma4.config",
    "rope": "models.gemma4.rope", "tokenizer": "models.gemma4.tokenizer",
    "chat": "models.gemma4.chat", "ct": "models.gemma4.ct", "flash": "models.gemma4.flash",
    "qwen": "models.qwen.qwen35", "qwen4": "models.qwen.qwen38", "qwen_gpu": "models.qwen.gpu35",
    "qwen4_gpu": "models.qwen.gpu38", "qwen_tok": "models.qwen.tokenizer",
    "vision_qwen": "models.qwen.vision",
    "program": ("ir.program", "models.gemma4.forms"),
    "gpu": ("backends.cuda.runtime", "models.gemma4.gpu"),
    "gemma4_encoders": ("encoders.runtime", "models.gemma4.encoders"),
}


class _Both(types.ModuleType):
    """The names of two modules (the generic part first)."""

    def __init__(self, name, parts):
        super().__init__(name)
        self.__parts = parts

    def __getattr__(self, name):
        for m in self.__parts:
            if hasattr(m, name):
                return getattr(m, name)
        raise AttributeError("module %r has no attribute %r" % (self.__name__, name))


class _Finder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, name, path=None, target=None):
        if name.startswith("reclank.") and name[len("reclank."):] in OLD:
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        new = OLD[spec.name[len("reclank."):]]
        if isinstance(new, str):
            return importlib.import_module("reclank." + new)
        return _Both(spec.name, [importlib.import_module("reclank." + n) for n in new])

    def exec_module(self, module):
        pass


def install():
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
