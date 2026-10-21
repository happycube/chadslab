#!/usr/bin/env python3
"""The inventory of Nemotron Super (S0 of docs/NEMOTRON_SUPER_PLAN.md): every tensor of the checkpoint
by its role, its shape against the config, its stored type against hf_quant_config.json (NVFP4: U8
codes, F8_E4M3 scales of 16, F32 weight_scale_2 and input_scale; FP8: F8_E4M3 with an F32 scale and
an input_scale; the rest BF16 or F32), and the bytes of each role. Reads the headers only.

    python scripts/check_nemotron_super_inventory.py /home/cpage/models/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4
"""
from __future__ import annotations

import collections
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.models.nemotron.model import NemotronConfig  # noqa: E402

BYTES = {"F8_E4M3": 1, "U8": 1, "BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4}


def headers(path):
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for f in sorted(set(idx.values())):
        with open(os.path.join(path, f), "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        h.pop("__metadata__", None)
        out.update(h)
    return out


def main():
    path = sys.argv[1]
    c = NemotronConfig(path)
    hq = json.load(open(os.path.join(path, "hf_quant_config.json")))["quantization"]["quantized_layers"]
    T = headers(path)
    H, L, E = c.hidden, c.latent, c.moe[min(c.moe)][0]
    inner = c.moe[min(c.moe)][1]
    proj = c.m_inner + c.conv_dim + c.m_heads
    want = {"mixer.in_proj": (proj, H), "mixer.out_proj": (H, c.m_inner),
            "mixer.q_proj": (c.heads * c.head_dim, H), "mixer.k_proj": (c.kv_heads * c.head_dim, H),
            "mixer.v_proj": (c.kv_heads * c.head_dim, H), "mixer.o_proj": (H, c.heads * c.head_dim),
            "up_proj": (inner, L), "down_proj": (L, inner),
            "shared_experts.up_proj": (c.shared_inner, H), "shared_experts.down_proj": (H, c.shared_inner),
            "fc1_latent_proj": (L, H), "fc2_latent_proj": (H, L), "gate": (E, H)}
    by_role, bad, n, unlisted = collections.Counter(), [], 0, collections.Counter()
    for name, t in T.items():
        n += 1
        size = BYTES[t["dtype"]]
        for s in t["shape"]:
            size *= s
        role = ("vision" if name.startswith(("vision_model.", "mlp1.", "vision_projector.")) else
                "mtp" if ".mtp." in name else
                "routed experts" if ".experts." in name else
                "shared experts" if "shared_experts" in name else
                "lm_head" if "lm_head" in name else "embedding" if "embeddings" in name else "backbone")
        by_role[role] += size
        m = re.match(r"language_model\.(?:backbone\.layers|mtp\.layers)\.\d+\.(.*)\.(weight|weight_scale|weight_scale_2|input_scale)$", name)
        if not m:
            continue
        mod, part = m.group(1), m.group(2)
        key = next((k for k in sorted(want, key=len, reverse=True) if mod.endswith(k)), None)
        if key is None:
            continue
        r, cc = want[key]
        base = name.rsplit(".", 1)[0]
        q = hq.get(base, {}).get("quant_algo")
        if q is None:
            # (the MTP block is not in hf_quant_config.json: its type from its tensors)
            q = ("NVFP4" if base + ".weight_scale_2" in T else
                 "FP8" if T.get(base + ".weight", {}).get("dtype") == "F8_E4M3" else None)
            if q is not None and part == "weight":
                unlisted[q] += 1
        if part == "weight":
            if q == "NVFP4":
                exp = ("U8", [r, cc // 2])
            elif q == "FP8":
                exp = ("F8_E4M3", [r, cc])
            else:
                exp = ("BF16", [r, cc])
            if (t["dtype"], t["shape"]) != exp:
                bad.append((name, t["dtype"], t["shape"], exp))
        elif part == "weight_scale" and q == "NVFP4" and (t["dtype"], t["shape"]) != ("F8_E4M3", [r, cc // 16]):
            bad.append((name, t["dtype"], t["shape"], ("F8_E4M3", [r, cc // 16])))
    for role, b in by_role.most_common():
        print("%-16s %7.2f GB" % (role, b / 1e9))
    print("%d tensors, %.1f GB; %d not as expected; quantized but not in hf_quant_config.json (typed by their"
          " tensors): %s" % (n, sum(by_role.values()) / 1e9, len(bad), dict(unlisted)))
    for x in bad[:20]:
        print("  ", x)
    experts = sum(1 for k in T if re.search(r"backbone\.layers\.\d+\.mixer\.experts\.\d+\.up_proj\.weight$", k))
    print("routed experts in the backbone: %d (config: %d x %d = %d)" % (experts, len(c.moe), E, len(c.moe) * E))
    ok = not bad and experts == len(c.moe) * E
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
