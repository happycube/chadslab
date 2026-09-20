# scripts/_bootstrap.py
"""
Shared bootstrap.  Import this *before* transformers / huggingface_hub.

It pins every Hugging Face cache inside <project>/.cache/huggingface so a run
never writes to $HOME, which is read-only under the DSH file sandbox.
"""
from __future__ import annotations

import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
_HF_HOME = ROOT / ".cache" / "huggingface"

os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HOME / "hub"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# --- Checkpoints -----------------------------------------------------------
# Primary target: Google's QAT checkpoint stored at half precision. The weights
# already live on the Q4_0 lattice, so quantizing them to int4 in PyTorch is
# near-lossless. ~23.95 GB on disk (11.96 B params, bf16).
MODEL_ID = "google/gemma-4-12B-it-qat-q4_0-unquantized"

# Same QAT weights, pre-serialized as compressed-tensors w4a16 (~10.3 GB).
# Convenient alternative: no in-process quantization step, vLLM-friendly.
W4A16_MODEL_ID = "google/gemma-4-12B-it-qat-w4a16-ct"

# Non-QAT bf16 baseline for accuracy comparisons.
BASELINE_MODEL_ID = "google/gemma-4-12B-it"

# Multi-token-prediction drafter that pairs with the QAT target model.
ASSISTANT_MODEL_ID = "google/gemma-4-12B-it-qat-q4_0-unquantized-assistant"

__all__ = [
    "ROOT",
    "MODEL_ID",
    "W4A16_MODEL_ID",
    "BASELINE_MODEL_ID",
    "ASSISTANT_MODEL_ID",
]
