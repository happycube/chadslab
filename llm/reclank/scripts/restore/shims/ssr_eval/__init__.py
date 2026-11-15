"""Stand-in for ssr_eval (its PyPI package won't install): A2SB uses it only for validation metrics against a
reference, never in inference."""
from . import metrics  # noqa: F401
