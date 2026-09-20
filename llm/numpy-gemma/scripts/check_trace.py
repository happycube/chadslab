"""Compare the NumPy forward pass against a captured Hugging Face trace."""
from __future__ import annotations

import argparse
import json

import numpy as np

from np_gemma import Config, Model, SafeTensors

FULL_KEYS = ("inputs_embeds", "last_hidden_state", "lm_head", "logits", "logits_last")


def is_full(key):
    return key in FULL_KEYS or (key.startswith("layers.") and key.endswith(".out"))


def stats(ref, got):
    ref = np.asarray(ref, dtype=np.float32)
    got = np.asarray(got, dtype=np.float32)
    if ref.shape != got.shape:
        return None, None, None, (ref.shape, got.shape)
    d = np.abs(ref - got)
    rn = float(np.linalg.norm(ref.ravel()))
    gn = float(np.linalg.norm(got.ravel()))
    cos = float(np.dot(ref.ravel(), got.ravel()) / (rn * gn + 1e-30))
    return float(d.max()), float(d.mean()), cos, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--trace", required=True)
    ap.add_argument("--input-ids", default=None)
    ap.add_argument("--position", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    ap.add_argument("--top", type=int, default=50)
    args = ap.parse_args()

    manifest = json.load(open(args.trace + "/manifest.json"))
    ids = [int(x) for x in args.input_ids.split(",")] if args.input_ids else manifest["input_ids"]
    pos = manifest["position"] if args.position is None else args.position
    if pos < 0:
        pos += len(ids)
    ref = SafeTensors(args.trace + "/tensors.safetensors")
    cfg = Config.load(args.config)

    captured = {}
    with SafeTensors(args.weights) as st:
        model = Model(st, cfg)
        x = model.forward(ids, hook=lambda k, v: captured.__setitem__(k, v), max_layers=args.layers)
        full = args.layers is None or args.layers >= cfg.num_hidden_layers
        raw = model.logits(x, apply_softcap=False) if full else None
        soft = (np.tanh(raw / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
                if full else None)
    if full:
        # Trace stores pre-softcap lm_head at --position, and softcapped logits at
        # both --position and the last position.
        captured["lm_head"] = raw[pos]
        captured["logits"] = soft[pos]
        captured["logits_last"] = soft[-1]

    print("trace   :", args.trace)
    print("model   :", args.weights)
    print("input   :", ids, "| position:", pos, "| tokens:", manifest["tokens"])
    print("layers  :", args.layers if args.layers is not None else cfg.num_hidden_layers)
    print()
    rows = []
    for key in sorted(ref.names()):
        if key not in captured:
            continue
        got = captured[key]
        if not is_full(key):
            got = got[pos] if got.ndim > 0 and got.shape[0] == len(ids) else got
        mx, mn, cos, shape = stats(ref.get(key), got)
        rows.append((key, mx, mn, cos, shape))
    rows.sort(key=lambda r: (-1 if r[1] is None else r[1]), reverse=True)
    print(f"{'key':<46} {'max_abs':>11} {'mean_abs':>11} {'cosine':>10}")
    print("-" * 82)
    for key, mx, mn, cos, shape in rows[: args.top]:
        if shape is not None:
            print(f"{key:<46} shape mismatch ref={shape[0]} got={shape[1]}")
        else:
            print(f"{key:<46} {mx:>11.3e} {mn:>11.3e} {cos:>10.6f}")
    ok = sum(1 for r in rows if r[1] is not None)
    bad = sum(1 for r in rows if r[1] is not None and r[3] is not None and r[3] < 0.999)
    missing = [k for k in captured if k not in set(ref.names())]
    if full:
        ref_last = ref.get("logits_last")
        got_last = captured["logits_last"]
        ref_top = int(np.argmax(ref_last))
        got_top = int(np.argmax(got_last))
        order = np.argsort(-got_last)[:5]
        tok = manifest["tokens"]
        print()
        print("first generated token (last prompt position):")
        print(f"  HF argmax    : {ref_top}")
        print(f"  NumPy argmax : {got_top}  {'MATCH' if ref_top == got_top else 'MISMATCH'}")
        print(f"  NumPy top-5  : {[int(i) for i in order]}")
    print()
    print(f"compared {ok} tensors | cosine < 0.999: {bad} | numpy-only: {len(missing)} extra keys")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
