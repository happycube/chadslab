"""Compare the NumPy E4B model with the HuggingFace reference, layer by layer.

The script runs in two phases so that both models never use memory at the
same time.

Phase one writes a trace with HuggingFace:

    PYTHONPATH=. $PY scripts/e4b_trace.py hf --snapshot "$SNAP" \
        --prompt "The capital of France is" --out .cache/hf_e4b.npz

Phase two compares our model with that trace:

    PYTHONPATH=. $PY scripts/e4b_trace.py np --snapshot "$SNAP" \
        --trace .cache/hf_e4b.npz

The comparison prints the maximum difference at each layer. A correct
implementation shows a small difference that grows slowly, because the
reference runs in bfloat16 and our model runs in float32. A jump at one layer
points at the bug in that layer.

Two notes about the reference:

1.  `Gemma4ForConditionalGeneration.from_pretrained` does not work for this
    checkpoint in transformers 5.17. The compressed-tensors loader keeps
    `weight_packed` and `weight_scale` as parameters, removes `weight`, and
    then the shared `_init_weights` asks for `module.weight`. The load stops.

    If the call is patched to continue, the model runs and returns zeros: the
    dequantize step does not happen for these modules. So this script does not
    use `from_pretrained`. It builds a plain `Gemma4TextModel`, which has no
    quantized modules, and fills it from `np_gemma.ct`. `check_ct.py` shows
    that the reader matches the `compressed_tensors` library bit for bit, so
    the reference weights are the values the library would produce.

2.  The reference still runs HuggingFace's own forward pass. That is the point
    of the comparison: it checks the architecture, not the decode.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def chat_ids(snapshot, prompt, thinking=False):
    """Turn a prompt into token ids with the Gemma 4 chat template."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot)
    messages = [{"role": "user", "content": prompt}]
    out = tok.apply_chat_template(messages, tokenize=True,
                                  add_generation_prompt=True,
                                  enable_thinking=thinking)
    # A recent transformers returns a BatchEncoding even when tokenize is True.
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return [int(t) for t in out]


def build_reference(snapshot, dtype):
    """Build a plain Gemma4TextModel and fill it from the checkpoint.

    Return the model, the reader, and the output head.
    """
    import torch
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    from np_gemma.ct import CompressedTensors

    cfg = AutoConfig.from_pretrained(snapshot).text_config
    plans = cfg.per_layer_config
    print("reference config: sliding head_dim=%d full head_dim=%d kv_heads=%d/%d"
          % (plans[0].head_dim, plans[5].head_dim, plans[0].num_key_value_heads,
             plans[5].num_key_value_heads))

    model = Gemma4TextModel(cfg)
    model.eval()

    ct = CompressedTensors(os.path.join(snapshot, "model.safetensors"))
    sd = {}
    for key in list(model.state_dict().keys()):
        base = "model.language_model." + key
        # A quantized weight lives at `<module>.weight_packed` and a plain
        # weight at `<module>.weight`. Strip the suffix and let the reader
        # decide which form is present.
        module = base[:-len(".weight")] if base.endswith(".weight") else base
        arr = np.ascontiguousarray(ct.dequant(module))
        sd[key] = torch.from_numpy(arr).to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("state dict: missing=%s unexpected=%s" % (missing[:5], unexpected[:5]))
    model.to(dtype)
    head = torch.from_numpy(np.ascontiguousarray(ct.dequant("lm_head"))).to(dtype)
    return model, ct, head


# ---------------------------------------------------------------------------
def run_hf(args):
    import torch

    torch.set_grad_enabled(False)
    ids = chat_ids(args.snapshot, args.prompt, args.thinking)
    print("prompt tokens: %d" % len(ids))
    print("ids:", ids[:32], "..." if len(ids) > 32 else "")

    dtype = torch.bfloat16
    t0 = time.time()
    model, ct, head = build_reference(args.snapshot, dtype)
    print("built the reference in %.1f s" % (time.time() - t0))

    store = {}
    handles = []
    lm = model

    def grab(name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            store[name] = t.detach().float().numpy()[0]
            return None
        return hook

    handles.append(lm.embed_tokens.register_forward_hook(grab("emb")))
    for i, layer in enumerate(lm.layers):
        handles.append(layer.register_forward_hook(grab("layer.%d" % i)))

    original_ple = lm.project_per_layer_inputs

    def ple_hook(inputs_embeds, per_layer_inputs=None):
        out = original_ple(inputs_embeds, per_layer_inputs)
        store["ple"] = out.detach().float().numpy()[0]
        return out

    lm.project_per_layer_inputs = ple_hook

    input_ids = torch.tensor([ids], dtype=torch.long)
    t0 = time.time()
    out = lm(input_ids=input_ids, use_cache=False)
    print("forward in %.1f s" % (time.time() - t0))

    hidden = out.last_hidden_state
    logits = hidden @ head.T
    cap = model.config.final_logit_softcapping
    if cap:
        logits = torch.tanh(logits / cap) * cap
    store["final"] = logits.detach().float().numpy()[0]

    for h in handles:
        h.remove()
    lm.project_per_layer_inputs = original_ple

    print("reference hidden absmax=%.4g  logits absmax=%.4g"
          % (np.abs(store["layer.41"]).max(), np.abs(store["final"]).max()))
    np.savez_compressed(args.out, **{k: v.astype(np.float32) for k, v in store.items()})
    np.savez(args.out.replace(".npz", ".ids.npz"), ids=np.array(ids))
    print("saved %d arrays to %s" % (len(store), args.out))


# ---------------------------------------------------------------------------
def run_np(args):
    from np_gemma.ct import CompressedTensors
    from np_gemma.e4b import E4B, E4BConfig

    trace = np.load(args.trace)
    snap_ids = args.trace.replace(".npz", ".ids.npz")
    ids = list(np.load(snap_ids)["ids"]) if os.path.exists(snap_ids) else None
    if ids is None:
        ids = chat_ids(args.snapshot, args.prompt, args.thinking)
    print("ids: %d tokens" % len(ids))

    ct = CompressedTensors(os.path.join(args.snapshot, "model.safetensors"))
    cfg = E4BConfig.load(os.path.join(args.snapshot, "config.json"))
    print("config:", cfg.describe())

    model = E4B(ct, cfg, resident=not args.stream)
    store = {}

    def hook(i, x):
        store["layer.%d" % i] = np.array(x, dtype=np.float32)

    t0 = time.time()
    hidden = model.forward(ids, hook=hook)
    print("forward in %.1f s" % (time.time() - t0))
    store["final"] = model.logits(hidden)
    emb = model.embed_rows("model.language_model.embed_tokens", ids) * cfg.embed_scale
    store["emb"] = emb
    store["ple"] = model.per_layer_inputs(ids, emb)

    keys = (["emb", "ple"] + ["layer.%d" % i for i in range(cfg.num_hidden_layers)]
            + ["final"])
    print("\n%-10s %-16s %-12s %-12s %s"
          % ("array", "shape", "max|ref|", "max|diff|", "rel"))
    layer_diffs = []
    for key in keys:
        if key not in trace:
            print("%-10s MISSING in the trace" % key)
            continue
        ref = trace[key]
        got = store[key]
        if ref.shape != got.shape:
            print("%-10s SHAPE %s != %s" % (key, got.shape, ref.shape))
            layer_diffs.append((key, float("inf")))
            continue
        d = np.abs(ref - got)
        md = float(d.max())
        scale = float(np.abs(ref).max())
        rel = md / scale if scale else md
        if key.startswith("layer."):
            layer_diffs.append((key, md))
        print("%-10s %-16s %-12.5g %-12.5g %.4f"
              % (key, str(tuple(got.shape)), scale, md, rel))

    print("\n--- the largest jump between neighbouring layers ---")
    prev = None
    worst_jump = (None, -1.0)
    for k, v in layer_diffs:
        if prev is not None:
            jump = v - prev[1]
            if jump > worst_jump[1]:
                worst_jump = ((prev[0], k), jump)
        prev = (k, v)
    if worst_jump[0]:
        print("  %s -> %s : +%.5g" % (worst_jump[0][0], worst_jump[0][1], worst_jump[1]))
    print("\nlast layer diff: %.5g" % (layer_diffs[-1][1] if layer_diffs else float("nan")))
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["hf", "np"])
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--out", default=".cache/hf_e4b.npz")
    ap.add_argument("--trace", default=".cache/hf_e4b.npz")
    ap.add_argument("--stream", action="store_true",
                    help="do not keep our weights in memory; decode from the map")
    args = ap.parse_args()
    if args.phase == "hf":
        return run_hf(args)
    return run_np(args)


if __name__ == "__main__":
    sys.exit(main())
