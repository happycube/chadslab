"""Read the safetensors header of a remote checkpoint with a range request."""
from __future__ import annotations
import json
import struct
import sys

buf = open(sys.argv[1], "rb").read()
n = struct.unpack("<Q", buf[:8])[0]
print("header bytes:", n, " fetched:", len(buf), " complete:", len(buf) >= 8 + n)
hdr = json.loads(buf[8:8 + n])
names = [k for k in hdr if k != "__metadata__"]
print("tensors:", len(names))

def sh(name):
    t = hdr.get(name)
    return "%s %s" % (t["dtype"], t["shape"]) if t else "(none)"

print("\n--- top level text ---")
for name in sorted(names):
    if name.startswith(("lm_head.", "model.language_model.")) and \
       name.count(".") <= 3 and "layers." not in name:
        print("  %-62s %s" % (name, sh(name)))

print("\n--- layer 0 ---")
for name in sorted(names):
    if name.startswith("model.language_model.layers.0."):
        print("  %-62s %s" % (name.replace("model.language_model.layers.0.", ""), sh(name)))

print("\n--- which layers hold their own k and v ---")
row = []
for i in range(42):
    p = "model.language_model.layers.%d.self_attn." % i
    row.append("%d:%s" % (i, "KV" if (p + "k_proj.weight_packed") in hdr else "--"))
print("  " + " ".join(row[:21]))
print("  " + " ".join(row[21:]))

print("\n--- layer 5 (full attention) ---")
for name in sorted(names):
    if name.startswith("model.language_model.layers.5."):
        print("  %-62s %s" % (name.replace("model.language_model.layers.5.", ""), sh(name)))
