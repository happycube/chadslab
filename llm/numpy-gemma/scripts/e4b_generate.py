"""Check the E4B token output against the HuggingFace reference.

This script builds the HuggingFace reference model, generates a few tokens
greedily, then runs our model and compares.

The reference runs the whole sequence again for each new token. It does not
use a cache. The shared key and value layers make an incremental reference
awkward, and the full sequence is the clear form.

Our model runs the prompt one time and then one token at a time. The script
also generates with our model without a cache. The two forms must agree; that
checks the cache and the key sharing.

Run:

    PYTHONPATH=. $PY scripts/e4b_generate.py --snapshot "$SNAP" \
        --prompt "The capital of France is" --max-new-tokens 8
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def reference_greedy(model, head, ids, n, cap):
    """Generate n tokens with the reference. Run the full sequence each step."""
    import torch
    ids = list(ids)
    out_ids = []
    for _ in range(n):
        with torch.no_grad():
            out = model(input_ids=torch.tensor([ids]), use_cache=False)
            logits = out.last_hidden_state[0, -1] @ head.T
            if cap:
                logits = torch.tanh(logits / cap) * cap
        nxt = int(torch.argmax(logits.float()))
        out_ids.append(nxt)
        ids.append(nxt)
    return out_ids


def stateless_greedy(model, ids, n):
    """Generate n tokens with our model. Do not keep a cache."""
    ids = list(ids)
    out_ids = []
    for _ in range(n):
        hidden = model.forward(ids)
        logits = model.logits(hidden[-1:])[0]
        nxt = int(np.argmax(logits))
        out_ids.append(nxt)
        ids.append(nxt)
    return out_ids


def project_ids(snapshot, prompt, thinking=False):
    """Turn a prompt into token ids with our own tokenizer and template.

    The E2B and E4B models do not write an empty thought block when thinking is
    off, so the call passes empty_thought_block=False.
    """
    from np_gemma.tokenizer import Tokenizer
    tok = Tokenizer(os.path.join(snapshot, "tokenizer.json"))
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   thinking=thinking,
                                   empty_thought_block=False)
    return tok.encode(text), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--skip-reference", action="store_true")
    ap.add_argument("--mode", default="f32", choices=["f32", "stream", "int4"],
                    help="how the model keeps the weights")
    args = ap.parse_args()

    from e4b_trace import build_reference, chat_ids
    from np_gemma.ct import CompressedTensors
    from np_gemma.e4b import E4B, E4BConfig, E4BCache

    ids, tok = project_ids(args.snapshot, args.prompt)
    print("prompt: %r" % args.prompt)
    print("our tokenizer:   %s" % ids)
    if not args.skip_reference:
        hf = chat_ids(args.snapshot, args.prompt)
        print("huggingface ids: %s" % hf)
        if hf != ids:
            print("FAIL: the tokenizer or the chat template disagrees with HuggingFace")
            return 1
        print("OK: the tokenizer and the chat template agree")

    ref_ids = None
    if not args.skip_reference:
        import torch
        torch.set_grad_enabled(False)
        t0 = time.time()
        ref_model, _ct, head = build_reference(args.snapshot, torch.bfloat16)
        print("reference built in %.1f s" % (time.time() - t0))
        cap = ref_model.config.final_logit_softcapping
        t0 = time.time()
        ref_ids = reference_greedy(ref_model, head, ids, args.max_new_tokens, cap)
        print("reference generated in %.1f s: %s" % (time.time() - t0, ref_ids))
        del ref_model, head
        import gc
        gc.collect()

    ct = CompressedTensors(os.path.join(args.snapshot, "model.safetensors"))
    cfg = E4BConfig.load(os.path.join(args.snapshot, "config.json"))
    model = E4B(ct, cfg, mode=args.mode)
    print("mode: %s" % args.mode)

    cache = E4BCache(cfg)
    t0 = time.time()
    hidden = model.forward(ids, cache=cache, start_pos=0)
    print("our prefill with a cache in %.1f s" % (time.time() - t0))

    out_ids = []
    pos = len(ids)
    t0 = time.time()
    for _ in range(args.max_new_tokens):
        logits = model.logits(hidden[-1:])[0]
        nxt = int(np.argmax(logits))
        out_ids.append(nxt)
        hidden = model.forward([nxt], cache=cache, start_pos=pos)
        pos += 1
    step_time = (time.time() - t0) / max(1, args.max_new_tokens)
    print("our incremental decode: %s (%.2f s/token)" % (out_ids, step_time))

    t0 = time.time()
    stateless = stateless_greedy(model, ids, args.max_new_tokens)
    print("our stateless decode: %s (%.1f s total)" % (stateless, time.time() - t0))

    print("\n--- results ---")
    ok = True
    if stateless != out_ids:
        print("FAIL: the cached and the stateless runs differ")
        print("  cached:    %s" % out_ids)
        print("  stateless: %s" % stateless)
        ok = False
    else:
        print("OK: the cached and the stateless runs agree")
    if ref_ids is not None:
        n = min(len(ref_ids), len(out_ids))
        same = ref_ids[:n] == out_ids[:n]
        print("reference:   %s" % ref_ids)
        print("our model:   %s" % out_ids)
        if same:
            print("OK: all %d tokens agree with the reference" % n)
        else:
            first = next(i for i in range(n) if ref_ids[i] != out_ids[i])
            print("first difference at position %d: reference %d, ours %d"
                  % (first, ref_ids[first], out_ids[first]))
            ok = False

    print("\nour text: %r" % tok.decode(ids + out_ids))
    if ref_ids is not None:
        from transformers import AutoTokenizer
        hf_tok = AutoTokenizer.from_pretrained(args.snapshot)
        print("hf text:  %r" % hf_tok.decode(ids + ref_ids))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
