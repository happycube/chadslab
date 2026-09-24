"""Summarise a safetensors index: what tensors a checkpoint holds."""
from __future__ import annotations
import json
import sys
from collections import Counter

path = sys.argv[1]
d = json.load(open(path))
wm = d["weight_map"]
print("tensors:", len(wm), " total_size:", d.get("metadata", {}).get("total_size"))
pref = Counter()
for name in wm:
    parts = name.split(".")
    pref[".".join(parts[:3])] += 1
print("\n--- prefixes ---")
for k, v in sorted(pref.items(), key=lambda x: -x[1]):
    print("  %-58s %d" % (k, v))

def show(title, pred, limit=60):
    hits = sorted(n for n in wm if pred(n))
    print("\n--- %s (%d) ---" % (title, len(hits)))
    for n in hits[:limit]:
        print("   ", n)
    if len(hits) > limit:
        print("    ... and %d more" % (len(hits) - limit))

show("layer 0, language model", lambda n: n.startswith("model.language_model.layers.0."))
show("any per_layer", lambda n: "per_layer" in n)
show("embed / lm_head", lambda n: "embed" in n or "lm_head" in n)
show("scales and zero points", lambda n: n.endswith(("weight_scale", "weight_zero_point", "input_scale")), 20)
show("vision tower top", lambda n: n.startswith("model.vision_tower.") and n.count(".") <= 3, 20)
