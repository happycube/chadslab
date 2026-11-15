"""Run a repo script with every torch.load forced to weights_only=True (classes a checkpoint needs are allowlisted
by SAFE_GLOBALS, "module:Name,..."), the script's own directory first on the path.
    python safe_run.py SCRIPT ARGS..."""
import importlib, os, runpy, sys
import torch

_load = torch.load


def load(*a, **k):
    k["weights_only"] = True
    return _load(*a, **k)


torch.load = load
allow = []
for spec in filter(None, os.environ.get("SAFE_GLOBALS", "").split(",")):
    mod, name = spec.split(":")
    allow.append(getattr(importlib.import_module(mod), name))
if allow:
    torch.serialization.add_safe_globals(allow)
script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
