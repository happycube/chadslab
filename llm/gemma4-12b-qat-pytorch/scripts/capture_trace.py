#!/usr/bin/env python3
"""
Capture intermediate activations for one token position, for comparison
against a from-scratch implementation.

Runs a single forward pass over a Gemma 4 checkpoint and writes one
safetensors file plus a JSON manifest. Tensors are stored per token position
(batch dim dropped), so shapes are unambiguous.

Tensor inventory (all at --position unless noted):
  embed_tokens                     embedding output for the position
  inputs_embeds                    embedding output, full sequence
  layers.NN.input_layernorm        pre-attention RMSNorm
  layers.NN.self_attn.q_proj       query projection
  layers.NN.self_attn.k_proj       key projection
  layers.NN.self_attn.v_proj       value projection
  layers.NN.self_attn.o_proj       attention output projection
  layers.NN.self_attn.q_norm       per-head Q RMSNorm (heads, head_dim)
  layers.NN.self_attn.k_norm       per-head K RMSNorm (heads, head_dim)
  layers.NN.post_attention_layernorm
  layers.NN.pre_feedforward_layernorm
  layers.NN.mlp.gate_proj          GeGLU gate
  layers.NN.mlp.up_proj            GeGLU up
  layers.NN.mlp.down_proj          GeGLU down
  layers.NN.post_feedforward_layernorm
  layers.NN.out                    residual stream out of decoder block NN
  norm                             final RMSNorm output (position)
  last_hidden_state                final RMSNorm output, full sequence
  lm_head                          raw logits BEFORE final_logit_softcapping
  logits                           softcapped logits at --position
  logits_last                      softcapped logits at the last position
  attn.NN                          attention probabilities (heads, kv) with --attentions

NN is zero-padded to two digits: layers.00 ... layers.47.

Examples:
  python scripts/capture_trace.py --token-id 2 --quant bf16
  python scripts/capture_trace.py --prompt "The capital of France is" --position -1
  python scripts/capture_trace.py --prompt "Hello" --quant nf4 --all-positions --attentions
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import _bootstrap
from chat import build_quant_config

import torch
from safetensors.torch import save_file
from transformers import AutoModelForMultimodalLM, AutoProcessor

LEAF_MODULES = {
    "input_layernorm",
    "post_attention_layernorm",
    "pre_feedforward_layernorm",
    "post_feedforward_layernorm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}
EXTRA_MODULES = {
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
}
LAYER_RE = re.compile(r"^model\.language_model\.layers\.\d+\.")
BLOCK_RE = re.compile(r"^model\.language_model\.layers\.\d+$")
PREFIX = "model.language_model."

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def is_captured(name: str) -> bool:
    if name in EXTRA_MODULES:
        return True
    if not LAYER_RE.match(name):
        return False
    tail = name.split(".layers.", 1)[1].split(".", 1)[1]
    return tail in LEAF_MODULES


def canon(name: str) -> str:
    return name[len(PREFIX):] if name.startswith(PREFIX) else name


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=_bootstrap.MODEL_ID)
    p.add_argument("--quant", default="bf16",
                   choices=("auto", "int4", "int8", "nf4", "bf16", "none"),
                   help="bf16 is the golden reference (default).")
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--device", default="auto")
    p.add_argument("--prompt", default=None)
    p.add_argument("--system", default="You are a helpful assistant.")
    p.add_argument("--token-id", type=int, default=None,
                   help="Feed exactly this one token id instead of a prompt.")
    p.add_argument("--thinking", action="store_true")
    p.add_argument("--position", type=int, default=0,
                   help="Token position to capture; negative counts from the end.")
    p.add_argument("--all-positions", action="store_true",
                   help="Store decoder-block outputs and embeddings for the whole sequence.")
    p.add_argument("--attentions", action="store_true",
                   help="Also capture attention probabilities (forces eager attention).")
    p.add_argument("--save-dtype", choices=tuple(DTYPES), default="float32")
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def extract_tensor(module_output):
    if isinstance(module_output, (tuple, list)):
        for x in module_output:
            if torch.is_tensor(x):
                return x
        return None
    return module_output if torch.is_tensor(module_output) else None


def slice_tensor(t: torch.Tensor, pos: int, name: str = "") -> torch.Tensor:
    """Pick one token position and drop the batch dim, so shapes are unambiguous."""
    if name.endswith(("q_norm", "k_norm")) and t.dim() == 4:
        return t[0, pos]        # (b, seq, heads, head_dim) -> (heads, head_dim)
    if t.dim() == 4:            # attention probs (b, heads, q, kv)
        return t[0, :, pos, :]  # -> (heads, kv)
    if t.dim() == 3:            # (b, seq, feat)
        return t[0, pos]        # -> (feat,)
    if t.dim() == 2:            # (seq, feat) or (b, feat)
        return t[pos] if t.shape[0] > 1 else t[0]
    return t


def topk_tokens(row: torch.Tensor, tokenizer, k: int = 5):
    probs = torch.softmax(row.float(), dim=-1)
    vals, idx = torch.topk(probs, k)
    return [
        {"id": int(i), "token": tokenizer.decode([int(i)]), "prob": round(float(v), 8)}
        for v, i in zip(vals, idx)
    ]


def curated_config(model) -> dict:
    tc = model.config.get_text_config()
    raw = tc.to_dict()   # avoids per-layer attributes that raise on direct getattr
    keys = [
        "model_type", "num_hidden_layers", "hidden_size", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "head_dim", "global_head_dim",
        "sliding_window", "vocab_size", "max_position_embeddings", "rms_norm_eps",
        "final_logit_softcapping", "attention_k_eq_v", "num_kv_shared_layers",
        "layer_types", "rope_parameters", "tie_word_embeddings",
    ]
    cfg = {k: raw.get(k) for k in keys}
    cfg["quantization_config"] = getattr(model.config, "quantization_config", None)
    return cfg


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)

    quant = args.quant
    print(f"[env ] torch {torch.__version__} | cuda={torch.cuda.is_available()} | quant={quant}",
          flush=True)

    processor = AutoProcessor.from_pretrained(args.model)
    quant_config = None if quant in ("bf16", "none") else build_quant_config(quant, args.group_size)

    load_kwargs = dict(dtype=args.dtype, device_map=args.device, quantization_config=quant_config)
    if args.attentions:
        # SDPA cannot return attention weights; force the eager path.
        load_kwargs["attn_implementation"] = "eager"
    print(f"[load] {args.model} (dtype={args.dtype}, device_map={args.device}, "
          f"attn={load_kwargs.get('attn_implementation', 'default')})", flush=True)
    t0 = time.perf_counter()
    model = AutoModelForMultimodalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()
    print(f"[load] done in {time.perf_counter() - t0:.1f}s", flush=True)

    if args.token_id is not None:
        input_ids = torch.tensor([[args.token_id]], dtype=torch.long)
        prompt_text = None
    else:
        prompt_text = args.prompt or "The capital of France is"
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": prompt_text})
        enc = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=args.thinking,
        )
        input_ids = enc["input_ids"]
    input_ids = input_ids.to(model.device)
    seq_len = input_ids.shape[-1]
    pos = args.position if args.position >= 0 else seq_len + args.position
    if not 0 <= pos < seq_len:
        print(f"[err ] position {args.position} out of range for seq_len={seq_len}", file=sys.stderr)
        return 2

    captured: dict[str, torch.Tensor] = {}

    def make_hook(key: str, full: bool = False):
        def hook(module, inputs, output):
            t = extract_tensor(output)
            if t is None:
                return
            t = t.detach()
            if full:
                captured[key] = t.to("cpu", torch.float32)[0].contiguous()
            else:
                captured[key] = slice_tensor(t, pos, key).to("cpu", torch.float32).contiguous()
        return hook

    full_keys = {
        "model.language_model.embed_tokens": "inputs_embeds",
        "model.language_model.norm": "last_hidden_state",
    }
    handles = []
    for name, mod in model.named_modules():
        if is_captured(name):
            handles.append(mod.register_forward_hook(make_hook(canon(name))))
        if name in full_keys:
            handles.append(mod.register_forward_hook(make_hook(full_keys[name], full=True)))
        elif BLOCK_RE.match(name):
            handles.append(mod.register_forward_hook(make_hook(canon(name) + ".out", full=args.all_positions)))

    print(f"[run ] seq_len={seq_len} capture position={pos} "
          f"(token id {int(input_ids[0, pos])})", flush=True)
    t1 = time.perf_counter()
    outputs = model(input_ids=input_ids, use_cache=False, output_attentions=args.attentions)
    print(f"[run ] forward in {time.perf_counter() - t1:.1f}s", flush=True)
    for h in handles:
        h.remove()

    tensors: dict[str, torch.Tensor] = dict(captured)

    logits = outputs.logits.detach().to("cpu", torch.float32)[0]   # (seq, vocab), softcapped
    pred_pos = topk_tokens(logits[pos], processor.tokenizer)
    pred_last = topk_tokens(logits[-1], processor.tokenizer)
    tensors["logits"] = logits[pos].contiguous()
    tensors["logits_last"] = logits[-1].contiguous()

    if args.attentions and outputs.attentions is not None:
        for i, a in enumerate(outputs.attentions):
            t = a.detach().to("cpu", torch.float32)
            tensors[f"attn.{i:02d}"] = (t[0, :, pos, :] if t.dim() == 4 else t[0]).contiguous()

    # .clone() forces distinct storage: safetensors refuses tensors that share memory
    # (e.g. logits and logits_last are the same row when seq_len == 1).
    tensors = {k: v.to(DTYPES[args.save_dtype]).clone().contiguous() for k, v in tensors.items()}

    short = args.model.split("/")[-1]
    out = Path(args.out) if args.out else (_bootstrap.ROOT / "traces" / f"{short}_{quant}_pos{args.position}")
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, out / "tensors.safetensors",
              metadata={"format": "pt", "model": args.model, "quant": quant})

    import transformers
    manifest = {
        "model": args.model,
        "quant": quant,
        "load_dtype": args.dtype,
        "save_dtype": args.save_dtype,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda": torch.cuda.is_available(),
        "prompt": prompt_text,
        "system": args.system if args.token_id is None else None,
        "token_id": args.token_id,
        "thinking": args.thinking,
        "seed": args.seed,
        "seq_len": seq_len,
        "position": pos,
        "position_token_id": int(input_ids[0, pos]),
        "position_token": processor.tokenizer.decode([int(input_ids[0, pos])]),
        "input_ids": input_ids[0].tolist(),
        "tokens": [processor.tokenizer.decode([int(i)]) for i in input_ids[0].tolist()],
        "position_prediction_top5": pred_pos,
        "first_generated_token": {"id": pred_last[0]["id"], "token": pred_last[0]["token"],
                                  "top5": pred_last},
        "config": curated_config(model),
        "tensors": {k: {"shape": list(v.shape), "dtype": str(v.dtype).replace("torch.", "")}
                    for k, v in tensors.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    print(f"[save] {len(tensors)} tensors -> {out}")
    print(f"[next] position {pos} -> {pred_pos[0]['token']!r} (id {pred_pos[0]['id']})")
    print(f"[next] first generated token -> {pred_last[0]['token']!r} (id {pred_last[0]['id']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
