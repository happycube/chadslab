#!/usr/bin/env python3
"""
Run Google Gemma 4 12B QAT in PyTorch.

Default checkpoint: google/gemma-4-12B-it-qat-q4_0-unquantized
  -> 11.96 B params stored in bf16 on the Q4_0 QAT lattice (~24 GB on disk).
     We quantize it to int4 at load time with torchao, which matches the lattice
     the model was trained against.

Examples:
  python scripts/chat.py --prompt "Explain grouped-query attention briefly."
  python scripts/chat.py --quant int4 --prompt "..." --max-new-tokens 64
  python scripts/chat.py --quant int8 --prompt "..."          # CPU-friendly fallback
  python scripts/chat.py --quant bf16 --prompt "..."          # unquantized reference
  python scripts/chat.py --quant nf4  --prompt "..."          # bitsandbytes NF4
  python scripts/chat.py --quant none --model google/gemma-4-12B-it-qat-w4a16-ct
"""
from __future__ import annotations

import argparse
import sys
import time

import _bootstrap  # noqa: F401  (sets HF_HOME before transformers import)

import torch
from transformers import AutoModelForMultimodalLM, AutoProcessor

# Multimodal glue is kept in bf16: quantizing the vision/audio projectors hurts
# quality and buys almost nothing (they are a tiny fraction of the parameters).
_MODULES_TO_NOT_CONVERT = [
    "lm_head",
    "embed_vision",
    "embed_audio",
    "vision_embedder",
]

QUANT_CHOICES = ("auto", "int4", "int8", "nf4", "bf16", "none")


def rss_gb() -> float:
    try:
        import os
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / 1e9
    except Exception:
        return float("nan")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=_bootstrap.MODEL_ID)
    p.add_argument("--prompt", default="Write a short joke about saving RAM.")
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--quant", choices=QUANT_CHOICES, default="auto",
                   help="Weight quantization applied at load time (default: auto).")
    p.add_argument("--group-size", type=int, default=32,
                   help="torchao int4 group size; QAT uses 32 (default: 32).")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--thinking", action="store_true",
                   help="Enable Gemma's thinking/reasoning mode.")
    p.add_argument("--dtype", default="auto",
                   help="Compute dtype: 'auto' (recommended), 'bfloat16', 'float32'.")
    p.add_argument("--device", default="auto",
                   help="accelerate device_map, e.g. 'auto', 'cpu', 'cuda:0'.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=64)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--inspect", action="store_true",
                   help="Print the class of a sample quantized weight after loading.")
    return p.parse_args()


def resolve_quant(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "int4"
    # torchao int4 weight-only kernels are CUDA-first; int8 is the reliable CPU path.
    return "int8"


def build_quant_config(name: str, group_size: int):
    if name in ("bf16", "none"):
        return None
    if name == "int4":
        from torchao.quantization import Int4WeightOnlyConfig
        from transformers import TorchAoConfig
        ao = Int4WeightOnlyConfig(group_size=group_size)
        return TorchAoConfig(ao, modules_to_not_convert=_MODULES_TO_NOT_CONVERT)
    if name == "int8":
        from torchao.quantization import Int8WeightOnlyConfig
        from transformers import TorchAoConfig
        ao = Int8WeightOnlyConfig()
        return TorchAoConfig(ao, modules_to_not_convert=_MODULES_TO_NOT_CONVERT)
    if name == "nf4":
        from transformers import BitsAndBytesConfig
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise ValueError(f"unknown quant: {name}")


def main() -> int:
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    quant = resolve_quant(args.quant)
    print(f"[env ] torch {torch.__version__} | cuda={torch.cuda.is_available()} "
          f"| quant={quant}", flush=True)
    if quant == "int4" and not torch.cuda.is_available():
        print("[warn] torchao int4 kernels need mslk (CUDA wheels only). "
              "On CPU use --quant nf4 for 4-bit or --quant int8.", flush=True)

    print(f"[load] processor: {args.model}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)

    quant_config = build_quant_config(quant, args.group_size)
    print(f"[load] model    : {args.model} (dtype={args.dtype}, device_map={args.device})",
          flush=True)
    t0 = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(
        args.model,
        dtype=args.dtype,
        device_map=args.device,
        quantization_config=quant_config,
    )
    model.eval()
    print(f"[load] done in {time.perf_counter() - t0:.1f}s | "
          f"footprint {model.get_memory_footprint() / 1e9:.2f} GB | "
          f"RSS {rss_gb():.2f} GB", flush=True)
    if args.inspect:
        shown = 0
        for name, mod in model.named_modules():
            w = getattr(mod, "weight", None)
            if w is not None and any(k in name for k in ("q_proj", "gate_proj", "mlp")):
                print(f"[insp] {name}: {type(w).__name__} dtype={w.dtype}", flush=True)
                shown += 1
                if shown >= 3:
                    break

    messages = [
        {"role": "system", "content": args.system},
        {"role": "user", "content": args.prompt},
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=args.thinking,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]

    t1 = time.perf_counter()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=max(args.temperature, 1e-5),
            top_p=args.top_p,
            top_k=args.top_k,
        )
    dt = time.perf_counter() - t1
    n_new = outputs.shape[-1] - input_len
    print(f"[gen ] {n_new} tokens in {dt:.1f}s ({n_new / max(dt, 1e-9):.2f} tok/s)",
          flush=True)

    response = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    try:
        parsed = processor.parse_response(response, prefix=inputs["input_ids"])
    except Exception:
        parsed = response
    print("\n===== response =====")
    print(parsed if isinstance(parsed, str) else response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
