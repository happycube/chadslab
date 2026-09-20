#!/usr/bin/env python3
"""Sanity-check the Gemma 4 PyTorch environment without loading any weights."""
from __future__ import annotations

import platform
import shutil
import sys

import _bootstrap

import torch


def main() -> int:
    print("python      ", sys.version.split()[0], platform.machine())
    print("torch       ", torch.__version__)
    print("cuda        ", torch.cuda.is_available(),
          f"(devices={torch.cuda.device_count()})")
    try:
        import transformers
        print("transformers", transformers.__version__)
    except Exception as exc:  # pragma: no cover
        print("transformers import failed:", exc)
        return 1
    for mod in ("accelerate", "compressed_tensors", "torchao", "safetensors",
                "sentencepiece", "tokenizers", "huggingface_hub"):
        try:
            m = __import__(mod)
            print(f"{mod:<12}", getattr(m, "__version__", "(no __version__)"))
        except Exception as exc:
            print(f"{mod:<12} MISSING ({exc})")

    print("HF_HOME     ", _bootstrap.__file__ and __import__("os").environ.get("HF_HOME"))
    total, used, free = shutil.disk_usage(_bootstrap.ROOT)
    print(f"disk        {free / 1e9:.1f} GB free of {total / 1e9:.1f} GB")

    from transformers import AutoConfig, AutoProcessor
    cfg = AutoConfig.from_pretrained(_bootstrap.MODEL_ID)
    print("config      ", type(cfg).__name__, "| model_type:", cfg.model_type,
          "| layers:", cfg.text_config.num_hidden_layers)
    proc = AutoProcessor.from_pretrained(_bootstrap.MODEL_ID)
    print("processor   ", type(proc).__name__)
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
