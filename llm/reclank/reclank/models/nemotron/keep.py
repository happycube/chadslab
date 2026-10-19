"""Nemotron 3.5 Bolt in the weightkeeper (reclank/runtime/weightkeeper.py, the tensors layout;
docs/NEMOTRON_BOLT_PLAN.md): what a round of NemotronKQ would read from the shards or make at its
start, made once and handed to every round.

- the routed experts of each layer in the two stacks NemotronKQ.experts makes (up, down: expert
  after expert, the IQ2_XXS blocks in KQ_IQ2R rows: reclank/formats/iq2.py to_rows), whole on every
  node when they fit the memory of each (--copies auto: 17 GB a node), else whole on the GPU's node
  (NM_MOE reads a layer's stack at one address);
- the Mamba projections in the KQ_NV4 rows of kq_nv4_pack (the NVFP4 nibbles and scales as stored);
- the shared experts in KQ_IQ2R rows ("iq2r.<name>");
- every other tensor its bytes as the shard has them (Q8_0 blocks, bfloat16, float32).

The segments are sealed for size only (the default): a round maps them writable, so the GPU maps the
stacks for the experts of a prompt (cudaHostRegister; NemotronGPU._zero_copy).

    python scripts/weightkeeper.py /home/cpage/models/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
    m = NemotronKQ(path)            # takes the weightkeeper when it holds this checkpoint
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from reclank.models.qwen.keep import _up
from reclank.models.qwen.keep_exl3 import copies_mode
from reclank.models.qwen.keep_fp4l import _pool, ident  # noqa: F401  (ident: the plugin's)

LAYOUT = "tensors-v1"
IQ2 = "iq2r"                    # the form of the IQ2_XXS tensors (an index without it: the blocks as stored)
MODEL = "nemotron-bolt"


def detect(path):
    try:
        c = json.load(open(os.path.join(path, "config.json")))
    except OSError:
        return False
    from reclank.models.nemotron.engine import detect as d
    return d(c)


def _model(path):
    from reclank.models.nemotron.kq import NemotronKQ
    return NemotronKQ(path, weightkeeper=False)


def layout(path, nodes, share, o):
    """The index: "exps.<layer>.<up|down>" (place experts: whole on each node it lists),
    "nv4.<name>" (the KQ_NV4 rows), "iq2r.<name>" (the KQ_IQ2R rows) and "raw.<name>" (the bytes of
    the shard) on the GPU's node."""
    from reclank.backends.cpu import cops
    from reclank.formats import iq2
    m = _model(path)
    cfg, ck = m.cfg, m.ck
    parts = len(nodes)
    dpart = nodes.index(o.dense_node) if o.dense_node in nodes else 0
    ebytes = 0
    stacks = []
    for i, (E, inner, _k) in cfg.moe.items():
        for which, rows, cols in (("up", inner, cfg.latent), ("down", cfg.latent, inner)):
            nb = rows * iq2.row_bytes(cols)
            stacks.append(("exps.%d.%s" % (i, which), i, which, E, nb))
            ebytes += E * nb
    copies = copies_mode(o, ebytes, nodes) if parts > 1 else "split"
    tens, used, order = {}, [0] * parts, []

    def at(p, n):
        off = used[p]
        used[p] = _up(off + n)
        return off
    for name, i, which, E, nb in stacks:
        ps = [[p, at(p, E * nb), list(range(E))] for p in (range(parts) if copies == "both" else [dpart])]
        tens[name] = dict(place="experts", E=E, nb=nb, parts=ps, layer=i, which=which, kind="experts")
        order.append(name)
    for name in ck.where:
        if ".experts." in name and ".shared_experts." not in name or not name.startswith("language_model."):
            continue                    # (the routed experts: the stacks; the vision tower: P7)
        if name.endswith((".weight_scale", ".weight_scale_2")):
            continue
        base = name[:-len(".weight")] if name.endswith(".weight") else None
        if base is not None and base + ".weight_scale" in ck.where:
            rows, half = ck.shape(name)
            n = rows * cops.kq_nv4_row_bytes(2 * half)
            tens["nv4." + base] = dict(place="node", seg=dpart, off=at(dpart, n), bytes=n, rows=rows,
                                       cols=2 * half, kind="dense")
            order.append("nv4." + base)
            continue
        st = ck.file(name)
        t = st.header[name]
        if t["dtype"] == "U8" and len(t["shape"]) == 3 and t["shape"][-1] == 66:
            rows, cols = t["shape"][0], t["shape"][1] * 256
            n = rows * iq2.row_bytes(cols)
            tens["iq2r." + name[:-len(".weight")]] = dict(place="node", seg=dpart, off=at(dpart, n), bytes=n, rows=rows,
                                                          cols=cols, type="iq2r", kind="dense")
            order.append("iq2r." + name[:-len(".weight")])
            continue
        n = t["data_offsets"][1] - t["data_offsets"][0]
        tens["raw." + name] = dict(place="node", seg=dpart, off=at(dpart, n), bytes=n, dtype=t["dtype"],
                                   shape=list(t["shape"]), kind="dense")
        order.append("raw." + name)
    ck.close()
    return dict(layout=LAYOUT, model=MODEL, iq2=IQ2, source="nemotron", file=os.path.abspath(path), nodes=list(nodes),
                share=list(share), dense_part=dpart, copies=copies,
                types=dict(dense="as stored (NVFP4 in KQ_NV4 rows, IQ2_XXS in KQ_IQ2R rows)",
                           experts="iq2_xxs stacks in KQ_IQ2R rows", ngram="none"),
                tensors=tens, order=order, meta=[], part_bytes=list(used))


def fill(sh, index, segs, threads, log):
    """The tensors into the segments: the bytes of the shards, the KQ_NV4 and KQ_IQ2R rows packed,
    then the stacks of experts (each expert's KQ_IQ2R rows after the other)."""
    from reclank.backends.cpu import cops
    from reclank.formats import iq2
    m = _model(index["file"])
    ck = m.ck
    t0, n = time.time(), 0
    for name in index["order"]:
        d = index["tensors"][name]
        if d["place"] != "node":
            continue
        dst = segs[d["seg"]].array(d["off"], d["bytes"])
        if name.startswith("nv4."):
            base = name[4:]
            w, s = ck.raw(base + ".weight"), ck.raw(base + ".weight_scale")
            g = float(np.asarray(ck.f32(base + ".weight_scale_2")).reshape(-1)[0])
            cops.kq_nv4_pack([np.ascontiguousarray(w)], [np.ascontiguousarray(s)], [g], d["rows"], d["cols"], dst)
        elif name.startswith("iq2r."):
            iq2.to_rows(ck.raw(name[5:] + ".weight"), dst)
        else:
            dst[:] = ck.bytes(name[4:])
        n += d["bytes"]
    log("the dense tensors: %.1f GB in %.0f s" % (n / 1e9, time.time() - t0))
    t0 = time.time()

    def one(name):
        d = index["tensors"][name]
        p_ = "language_model.backbone.layers.%d.mixer.experts.%%d.%s_proj.weight" % (d["layer"], d["which"])
        for p, off, ids in d["parts"]:
            dst = segs[p].array(off, d["E"] * d["nb"]).reshape(d["E"], d["nb"])
            for e in range(d["E"]):
                iq2.to_rows(ck.raw(p_ % e), dst[e])
    stacks = [nm for nm in index["order"] if index["tensors"][nm]["place"] == "experts"]
    _pool(stacks, max(1, min(8, threads // 4)), one, log, "the stacks of experts")
    log("the experts: %d stacks in %.0f s" % (len(stacks), time.time() - t0))
    ck.close()


class _Plugin:
    layout = staticmethod(layout)
    fill = staticmethod(fill)
    ident = staticmethod(ident)


PLUGIN = _Plugin()


def attach(path):
    """The weightkeeper's segments when it holds this checkpoint (Nemotron), else None."""
    try:
        from reclank.runtime import weightkeeper as W
        w = W.attach(ckpt=path)
    except (OSError, ValueError):
        return None
    if w.index.get("model") != MODEL or w.index.get("iq2") != IQ2:
        return None
    return w
