#!/usr/bin/env python3
"""
Compare two captured traces (or any two safetensors with matching keys).

    python scripts/compare_trace.py --ref traces/ref --cand traces/mine

Each side may be a trace directory (containing tensors.safetensors) or a bare
.safetensors file. Use --key-map to translate candidate key names, e.g.
{"my_layers.0.attn_out": "layers.0.self_attn.o_proj"}.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file


def load(path: str):
    p = Path(path)
    if p.is_dir():
        manifest = {}
        mf = p / "manifest.json"
        if mf.exists():
            manifest = json.loads(mf.read_text())
        return load_file(p / "tensors.safetensors"), manifest
    return load_file(p), {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference trace dir or safetensors")
    ap.add_argument("--cand", required=True, help="candidate trace dir or safetensors")
    ap.add_argument("--key-map", default=None, help="JSON file mapping candidate key -> ref key")
    ap.add_argument("--top", type=int, default=80)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref, ref_manifest = load(args.ref)
    cand, _ = load(args.cand)
    kmap = json.loads(Path(args.key_map).read_text()) if args.key_map else {}
    cand = {kmap.get(k, k): v for k, v in cand.items()}

    rows = []
    for key, a in ref.items():
        b = cand.get(key)
        if b is None:
            continue
        a = a.float()
        b = b.float()
        if a.shape != b.shape:
            rows.append({"key": key, "ref_shape": list(a.shape), "cand_shape": list(b.shape),
                         "note": "shape mismatch"})
            continue
        diff = (a - b).abs()
        denom = float(a.norm())
        rows.append({
            "key": key,
            "shape": list(a.shape),
            "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()),
            "rel_l2": (float(diff.norm()) / denom) if denom else float("nan"),
            "cosine": float(torch.nn.functional.cosine_similarity(
                a.flatten(), b.flatten(), dim=0)),
        })

    missing = sorted(k for k in ref if k not in cand)
    extra = sorted(k for k in cand if k not in ref)
    rows.sort(key=lambda r: r.get("max_abs", float("inf")), reverse=True)

    print(f"ref  : {args.ref} ({len(ref)} tensors)")
    print(f"cand : {args.cand} ({len(cand)} tensors)")
    if ref_manifest:
        print(f"model: {ref_manifest.get('model')}  quant={ref_manifest.get('quant')}  "
              f"position={ref_manifest.get('position')}  seq_len={ref_manifest.get('seq_len')}")
    print()
    print(f"{'tensor':<44} {'shape':<18} {'max_abs':>12} {'mean_abs':>12} {'rel_l2':>10} {'cosine':>10}")
    print("-" * 112)
    for r in rows[: args.top]:
        if "note" in r:
            print(f"{r['key']:<44} {str(r['ref_shape']):<18} {r['note']}")
            continue
        print(f"{r['key']:<44} {str(r['shape']):<18} {r['max_abs']:>12.3e} "
              f"{r['mean_abs']:>12.3e} {r['rel_l2']:>10.3e} {r['cosine']:>10.6f}")
    print()
    if missing:
        print(f"missing in candidate ({len(missing)}): {', '.join(missing[:12])}"
              + (" ..." if len(missing) > 12 else ""))
    if extra:
        print(f"extra in candidate ({len(extra)}): {', '.join(extra[:12])}"
              + (" ..." if len(extra) > 12 else ""))

    summary = {
        "ref": args.ref,
        "cand": args.cand,
        "n_common": len(rows),
        "missing_in_candidate": missing,
        "extra_in_candidate": extra,
        "tensors": rows,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
