#!/usr/bin/env python3
"""Download a Gemma 4 QAT checkpoint into the project-local Hugging Face cache."""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401  (sets HF_HOME before huggingface_hub import)

from huggingface_hub import snapshot_download


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=_bootstrap.MODEL_ID,
                        help="HF repo id (default: the w4a16 4-bit QAT checkpoint)")
    parser.add_argument("--local-dir", default=None,
                        help="Also materialize a plain directory here (default: HF cache only)")
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()

    path = snapshot_download(
        repo_id=args.model,
        revision=args.revision,
        local_dir=args.local_dir,
        # Tokenizer/config/processor are tiny; the single safetensors blob is ~10 GB.
        max_workers=8,
    )
    print(f"downloaded {args.model} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
