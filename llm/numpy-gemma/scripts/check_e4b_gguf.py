"""Check the E4B GGUF path against a HuggingFace reference from the same file.

`e4b_trace.py` checks the compressed-tensors checkpoint. This script does the
same for the GGUF file. The reference uses the weights that the GGUF file
holds, so the check measures the model code and not the quantization.

The output head of the GGUF file is the token embedding. The two matrices hold
the same values in the checkpoint, so the reference uses the embedding table
for the head as well.

Run:

    PYTHONPATH=. $PY scripts/check_e4b_gguf.py --gguf ~/.cache/e4b-gguf/gemma-4-E4B_q4_0-it.gguf \
        --config-snapshot "$SNAP4B"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np


def build_reference(gguf_path, config_snapshot, dtype):
    """Build a Gemma4TextModel and fill it from the GGUF file."""
    import torch
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    from np_gemma.gguf import GGUF

    cfg = AutoConfig.from_pretrained(config_snapshot).text_config
    model = Gemma4TextModel(cfg)
    model.eval()
    g = GGUF(gguf_path)
    sd = {}
    for key in list(model.state_dict().keys()):
        arr = np.ascontiguousarray(g.get("model.language_model." + key))
        sd[key] = torch.from_numpy(arr).to(dtype)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print("state dict: missing=%s unexpected=%s" % (missing[:4], unexpected[:4]))
    model.to(dtype)
    head = sd["embed_tokens.weight"]
    return model, g, head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--config-snapshot", required=True,
                    help="a snapshot with the same architecture, for the config only")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    args = ap.parse_args()

    import torch
    torch.set_grad_enabled(False)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from e4b_trace import chat_ids
    from np_gemma.e4b import E4B, E4BConfig, E4BCache

    ids = chat_ids(args.config_snapshot, args.prompt)
    print("prompt tokens: %d" % len(ids))

    t0 = time.time()
    ref, g, head = build_reference(args.gguf, args.config_snapshot, torch.bfloat16)
    print("reference built in %.1f s" % (time.time() - t0))

    seen = {}

    def grab(name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            seen[name] = t.detach().float().numpy()[0]
            return None
        return hook

    handles = [ref.layers[i].register_forward_hook(grab("layer.%d" % i))
               for i in (0, 20, 41)]
    out = ref(input_ids=torch.tensor([ids]), use_cache=False)
    for h in handles:
        h.remove()
    last = out.last_hidden_state[0, -1].float()
    cap = ref.config.final_logit_softcapping
    ref_logits = last @ head.float().T
    if cap:
        ref_logits = torch.tanh(ref_logits / cap) * cap
    ref_logits = ref_logits.numpy()

    cfg = E4BConfig({"text_config": g.text_config()})
    model = E4B(g, cfg, mode="int4")
    store = {}

    def hook(i, x):
        store["layer.%d" % i] = np.array(x, dtype=np.float32)

    hidden = model.forward(ids, hook=hook)
    my_logits = model.logits(hidden)[-1]

    print("\n%-10s %-12s %-12s %s" % ("array", "max|ref|", "max|diff|", "cosine"))
    for key in ("layer.0", "layer.20", "layer.41"):
        a, b = seen[key], store[key]
        cos = float(a.ravel() @ b.ravel() / (np.linalg.norm(a) * np.linalg.norm(b)))
        print("%-10s %-12.4g %-12.4g %.6f" % (key, np.abs(a).max(), np.abs(a - b).max(), cos))
    cos = float(ref_logits @ my_logits / (np.linalg.norm(ref_logits) * np.linalg.norm(my_logits)))
    print("%-10s %-12.4g %-12.4g %.6f"
          % ("logits", np.abs(ref_logits).max(), np.abs(ref_logits - my_logits).max(), cos))
    print("\nfirst token: reference %d, ours %d"
          % (int(np.argmax(ref_logits)), int(np.argmax(my_logits))))

    # Greedy tokens from the reference and from our model.
    def ref_greedy(n):
        seq = list(ids)
        for _ in range(n):
            o = ref_forward(seq)
            seq.append(int(np.argmax(o)))
        return seq[len(ids):]

    def ref_forward(seq):
        with torch.no_grad():
            r = ref(input_ids=torch.tensor([seq]), use_cache=False)
        h = r.last_hidden_state[0, -1].float()
        o = h @ head.float().T
        if cap:
            o = torch.tanh(o / cap) * cap
        return o.numpy()

    ref_ids = ref_greedy(args.max_new_tokens)

    my_ids = model.generate(ids, max_new_tokens=args.max_new_tokens)
    print("\nreference: %s" % ref_ids)
    print("ours:      %s" % my_ids)
    n = min(len(ref_ids), len(my_ids))
    same = ref_ids[:n] == my_ids[:n]
    print("OK: all %d tokens agree" % n if same
          else "FAIL: the tokens differ")
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
