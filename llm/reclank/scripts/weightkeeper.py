"""The weightkeeper (reclank/runtime/weightkeeper.py): the weights of a large model
loaded once, handed to the rounds. DeepSeek V4.1 Flash (DEEPSEEK_V41_PLAN.md
4.2: the expert rows of every node) or Qwen3.8-Flash-Next (the original
bfloat16 checkpoint as the GGUF tensors of the runtime, reclank/models/qwen/keep.py);
the model comes from the checkpoint's config.json. An EXL3 directory of
Qwen3.8-Flash-Next (reclank/models/qwen/keep_exl3.py): its MUL1 blocks, the
experts whole on every node when they fit (--copies), the n-gram table read
from its file.

    python scripts/weightkeeper.py CKPT                 # as the user: pools filled beforehand, else 4 KB
    sudo .venv/bin/python scripts/weightkeeper.py CKPT --lock
                                                        # 1 GB pages, the memory locked in full,
                                                        # then it serves as the invoking user
    python scripts/weightkeeper.py --status | --stop
    python scripts/weightkeeper.py --check CKPT         # a round: map, compare rows with the shards
    python scripts/weightkeeper.py --gguf OUT CKPT      # Qwen: the tensors held as one GGUF file

Options: --share (the rows of part 0, default 0.5: balanced channels),
--layers 0-3 (a test), --dense (the dense tensors in a segment too, on
--dense-node), --pages 1g|2m|4k (the largest page), --threads, --socket,
--user/--group (the uid to drop to; default SUDO_UID/SUDO_GID), --seal size|write
(size, the default: the rounds map the segments writable so CUDA page-locks
them; write: read-only maps, the copies to the GPU stage). Qwen: the least
type of each kind of tensor, --dense-type, --experts-type, --ngram-type
(bf12, the default: the bfloat16 values in 12.25 bits; bf16; q8; rq8 for the
experts).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from reclank.runtime import weightkeeper as Z  # noqa: E402


def _ints(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def check_tensors(ckpt, path, n):
    """The tensors layout (Qwen3.8): random experts, n-gram rows, and dense
    and small tensors against a new conversion of the checkpoint (the same
    bytes)."""
    from reclank.backends.cpu import cops
    from reclank.formats.qwen38_orig import OrigSource, tiler
    from reclank.models.qwen import keep
    t0 = time.time()
    g = keep.KeptGGUF(ckpt, Z.attach(path, ckpt=ckpt))
    ix = g.index
    print("mapped %s GB in %.2f s (locked: %s; types %s)" % (
        [round(s["bytes"] / 1e9, 1) for s in ix["segments"]], time.time() - t0, ix["locked"], ix["types"]))
    if ix.get("source") == "gguf":
        return check_gguf(g, ckpt, n)
    if ix.get("source") == "fp4l":
        return check_fp4l(g, ckpt, n)
    src = OrigSource(ckpt)
    tiled = tiler(src.config())
    rng = random.Random(1)
    bad, done = 0, 0
    names = list(ix["tensors"])
    xs = [nm for nm in names if ix["tensors"][nm]["place"] == "experts"]
    for _ in range(n):
        gname = rng.choice(xs)
        d = ix["tensors"][gname]
        e = rng.randrange(d["E"])
        hf, lo, hi, _t = src._map[gname][1]
        ref, _r = keep.convert(np.asarray(src._bf16(hf)[e, lo:hi]), d["type"])
        got = None
        for _node, arr, ids in g.experts(gname):
            k = np.searchsorted(ids, e)
            if k < len(ids) and ids[k] == e:
                got = arr[k]
        done += 1
        if got is None or got.tobytes() != ref.tobytes():
            bad += 1
            print("MISMATCH %s expert %d" % (gname, e))
    if "per_layer_token_embd.weight" in ix["tensors"]:
        d = ix["tensors"]["per_layer_token_embd.weight"]
        g0, starts = 0, []
        for sh in src._ple_shards:
            starts.append(g0)
            g0 += src._shape(sh)[0]
        for _ in range(n):
            k = rng.randrange(len(src._ple_shards))
            r = rng.randrange(src._shape(src._ple_shards[k])[0])
            ref, _r = keep.convert(np.asarray(src._bf16(src._ple_shards[k])[r:r + 1]), d["type"])
            got = cops.kq_gather(g.row_addrs("per_layer_token_embd.weight", [starts[k] + r]), d["rb"])
            done += 1
            if got.tobytes() != ref.tobytes():
                bad += 1
                print("MISMATCH n-gram row %d" % (starts[k] + r))
    for gname in [nm for nm in names if ix["tensors"][nm]["place"] == "node"][::max(1, len(names) // (4 * n))]:
        d = ix["tensors"][gname]
        kind, s_ = src._map[gname]
        if kind == "dense" and gname.endswith(("attn_qkv.weight", "attn_gate.weight", "ssm_out.weight")):
            a = tiled(gname, np.asarray(src._bf16(s_)))
        elif kind == "f32":
            a = tiled(gname, src._make(gname))
        else:
            a = np.asarray(src._make(gname))
        if d["kind"] == "dense" and d["type"] != 30:
            ref, _r = keep.convert(np.ascontiguousarray(a).view(np.uint16), d["type"])
        else:
            ref = np.ascontiguousarray(a, np.float32 if d["type"] == 0 else None)
        done += 1
        if g.w.tensor(gname).tobytes() != ref.tobytes():
            bad += 1
            print("MISMATCH %s" % gname)
    print("%d tensors and rows compared: %s" % (done, "all equal" if not bad else "%d differ" % bad))
    return 1 if bad else 0


def check_fp4l(g, path, n):
    """FP4-Local (keep_fp4l): random experts of a few stacks, n-gram rows of a few shards and
    node tensors against FP4LSource packing them again (the same bytes)."""
    from reclank.backends.cpu import cops
    from reclank.formats.fp4local_st import FP4LSource
    src = FP4LSource(path)
    d = g.index["tensors"]
    rng = random.Random(1)
    bad = done = 0
    xs = [x for x in d if d[x]["place"] == "experts"]
    for name in rng.sample(xs, min(4, len(xs))):
        ref = src.pack_experts(name, np.empty(d[name]["E"] * d[name]["nb"], np.uint8)).reshape(d[name]["E"], -1)
        for _ in range(n):
            e = rng.randrange(d[name]["E"])
            for _nd, arr, ids in g.experts(name):
                k = list(ids).index(e) if e in ids else None
                if k is not None:
                    done += 1
                    bad += arr[k].tobytes() != ref[e].tobytes()
    if "per_layer_token_embd.weight" in d:
        dp = d["per_layer_token_embd.weight"]
        for k in sorted(rng.sample(range(len(src._ple_shards)), 3)) + [len(src._ple_shards) - 1]:
            rows, r0 = src.pack_ple(k)
            rows = rows.reshape(-1, dp["rb"])
            pick = np.array(sorted(rng.randrange(rows.shape[0]) for _ in range(100 * n)) + [0, rows.shape[0] - 1])
            done += 1
            bad += cops.kq_gather(g.row_addrs("per_layer_token_embd.weight", r0 + pick), dp["rb"]).tobytes() != \
                rows[pick].tobytes()
    for name in rng.sample([x for x in d if d[x]["place"] == "node"], min(n, 64)):
        done += 1
        bad += np.ascontiguousarray(src.raw(name)[0]).reshape(-1).view(np.uint8).tobytes() != g.w.tensor(name).tobytes()
    print("%d compared with the checkpoint: %s" % (done, "all equal" if not bad else "%d differ" % bad))
    return 1 if bad else 0


def check_gguf(g, path, n):
    """A weightkeeper of a GGUF file: random tensors, experts, and n-gram rows
    against the bytes of the file."""
    from reclank.backends.cpu import cops
    from reclank.formats.gguf import GGUF
    f = GGUF(path, stage=False)
    d = g.index["tensors"]
    rng = random.Random(1)
    bad = done = 0
    for name in rng.sample([x for x in d if d[x]["place"] == "node"], min(n, 64)):
        done += 1
        bad += f.raw(name)[0].view(np.uint8).reshape(-1).tobytes() != g.w.tensor(name).tobytes()
    xs = [x for x in d if d[x]["place"] == "experts"]
    for _ in range(n):
        name = rng.choice(xs)
        e = rng.randrange(d[name]["E"])
        ref = f.raw(name)[0].view(np.uint8).reshape(d[name]["E"], -1)[e]
        got = [arr[list(ids).index(e)] for _nd, arr, ids in g.experts(name) if e in ids][0]
        done += 1
        bad += ref.tobytes() != got.tobytes()
    if "per_layer_token_embd.weight" in d:
        dp = d["per_layer_token_embd.weight"]
        raw = f.raw("per_layer_token_embd.weight")[0].view(np.uint8).reshape(dp["rows"], dp["rb"])
        rows = np.array(sorted(rng.randrange(dp["rows"]) for _ in range(100 * n)))
        done += 1
        bad += cops.kq_gather(g.row_addrs("per_layer_token_embd.weight", rows), dp["rb"]).tobytes() != \
            raw[rows].tobytes()
    print("%d compared with the file: %s" % (done, "all equal" if not bad else "%d differ" % bad))
    return 1 if bad else 0


def check(ckpt, path, n):
    t0 = time.time()
    c = Z.request("status", path)
    with c:
        if json.loads(Z._recv_msg(c)).get("layout") == Z.LAYOUT_TENSORS:
            return check_tensors(ckpt, path, n)
    w = Z.attach(path)
    t1 = time.time()
    ix = w.index
    w.check(ckpt)
    sh = Z.Shards(ckpt)
    print("mapped %s GB in %.2f s (locked: %s); the checkpoint matches" % (
        [round(s["bytes"] / 1e9, 1) for s in ix["segments"]], t1 - t0, ix["locked"]))
    rng = random.Random(1)
    bad = 0
    v2 = ix["layout"] == Z.LAYOUT
    if v2:
        from reclank.backends.cpu import cops
    for _ in range(n):
        layer = rng.choice(ix["layers"])
        e = rng.randrange(ix["E"])
        for part in range(len(ix["segments"])):
            if v2:
                ms = w.mats(layer, part)
                for name, _off, r0, nr, cols, nbytes in ix["mats"][part]:
                    f, _dt, shape, o0, _o1 = sh.t[Z.EXPERT % (layer, e, name + ".weight")]
                    sf, _sdt, sshape, s0, _s1 = sh.t[Z.EXPERT % (layer, e, name + ".scale")]
                    with open(sh.file(f), "rb") as fh:
                        fh.seek(o0 + r0 * (cols // 2))
                        codes = fh.read(nr * cols // 2)
                    with open(sh.file(sf), "rb") as fh:
                        fh.seek(s0 + r0 * (cols // 32))
                        scales = fh.read(nr * cols // 32)
                    ref, _odd = cops.kq_mxx_pack(np.frombuffer(codes, np.uint8), np.frombuffer(scales, np.uint8),
                                                 nr, cols)
                    if ms[name][e].tobytes() != ref.tobytes():
                        bad += 1
                        print("MISMATCH layer %d expert %d part %d %s" % (layer, e, part, name))
                continue
            pl = w.planes(layer, part)
            for name, _off, r0, nr, rb in ix["planes"][part]:
                f, _dt, _shape, o0, o1 = sh.t[Z.EXPERT % (layer, e, name)]
                with open(sh.file(f), "rb") as fh:
                    fh.seek(o0 + r0 * rb)
                    ref = fh.read(nr * rb)
                if pl[name][e].tobytes() != ref:
                    bad += 1
                    print("MISMATCH layer %d expert %d part %d %s" % (layer, e, part, name))
    if ix.get("dense"):
        for name in list(ix["dense"]["tensors"])[:: max(1, len(ix["dense"]["tensors"]) // n)]:
            f, _dt, _shape, o0, o1 = sh.t[name]
            with open(sh.file(f), "rb") as fh:
                fh.seek(o0)
                if w.dense(name).tobytes() != fh.read(o1 - o0):
                    bad += 1
                    print("MISMATCH dense %s" % name)
    print("%d experts x %d parts compared: %s" % (n, len(ix["segments"]), "all equal" if not bad else
                                                  "%d planes differ" % bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--share", type=float, default=0.5)
    ap.add_argument("--nodes", type=_ints)
    ap.add_argument("--layers", type=_ints)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--lock", action="store_true")
    ap.add_argument("--dense", action="store_true")
    ap.add_argument("--dense-node", type=int, default=0)
    ap.add_argument("--pages", choices=("1g", "2m", "4k"), default="1g")
    ap.add_argument("--layout", choices=("mxx", "planes"), default="mxx",
                    help="mxx (v2, the default): the matrices in KQ_MXX; planes (v1): the shard's rows")
    ap.add_argument("--dense-type", choices=("bf12", "bf16", "q8"), default="bf12")
    ap.add_argument("--experts-type", choices=("bf12", "bf16", "q8", "rq8"), default="bf12")
    ap.add_argument("--ngram-type", choices=("bf12", "bf16", "q8"), default="bf12")
    ap.add_argument("--counts", help="Qwen: the counts of the router (layers x experts, .npy): the "
                    "experts of a layer go to the nodes in snake order of their counts (else in turn)")
    ap.add_argument("--copies", choices=("auto", "split", "both"), default="auto",
                    help="an EXL3 directory: the experts split between the nodes, or whole on every node "
                         "(auto: when they fit)")
    ap.add_argument("--seal", choices=("size", "write"), default="size",
                    help="the seals of the segments: size (the rounds map them writable, so CUDA page-locks "
                         "them for DMA) or write (read-only maps; the copies to the GPU stage)")
    ap.add_argument("--socket")
    ap.add_argument("--user", type=int)
    ap.add_argument("--group", type=int)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gguf", metavar="OUT", help="Qwen: write the tensors the weightkeeper holds for "
                    "CKPT as one GGUF file (their types: BF12 by default), then exit")
    ap.add_argument("--check-n", type=int, default=20)
    a = ap.parse_args()
    if a.status or a.stop:
        c = Z.request("status" if a.status else "stop", a.socket)
        with c:
            out = Z._recv_msg(c).decode()
        print(json.dumps(json.loads(out), indent=1) if a.status else out)
        return 0
    if not a.ckpt:
        ap.error("the checkpoint directory")
    if a.check:
        return check(a.ckpt, a.socket, a.check_n)
    if a.gguf:
        from reclank.models.qwen import keep
        keep.export_gguf(keep.KeptGGUF(a.ckpt, Z.attach(a.socket, ckpt=a.ckpt)), a.gguf,
                         log=lambda m: print(m, flush=True))
        return 0
    o = Z.Options(ckpt=a.ckpt, share=a.share, nodes=a.nodes, layers=a.layers, threads=a.threads,
                  lock=a.lock, dense=a.dense, dense_node=a.dense_node,
                  maxpage={"1g": Z.G1, "2m": Z.M2, "4k": Z.K4}[a.pages], layout=a.layout, sock=a.socket,
                  dense_type=a.dense_type, experts_type=a.experts_type, ngram_type=a.ngram_type, counts=a.counts,
                  copies=a.copies, seal=a.seal,
                  user=a.user, group=a.group)
    return Z.run(o) or 0


if __name__ == "__main__":
    sys.exit(main())
