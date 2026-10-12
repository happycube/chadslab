"""The old name of the package reclank.

`import np_gemma` gives the package reclank, and `import np_gemma.X` gives
the module reclank.X: the same objects, not copies. This shim goes in phase
P9 of REFACTOR_PLAN.md.
"""
import importlib
import importlib.abc
import importlib.util
import sys

import reclank


class _Alias(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """np_gemma.X -> the module reclank.X."""

    def find_spec(self, name, path=None, target=None):
        if name.startswith("np_gemma."):
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return importlib.import_module("reclank" + spec.name[len("np_gemma"):])

    def exec_module(self, module):
        pass


if not any(isinstance(f, _Alias) for f in sys.meta_path):
    sys.meta_path.insert(0, _Alias())
sys.modules[__name__] = reclank
