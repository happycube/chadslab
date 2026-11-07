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
    python scripts/weightkeeper.py --status | --stop    # the default keeper (--name NAME: that one)
    python scripts/weightkeeper.py --list               # the keepers running and their entries
    python scripts/weightkeeper.py --serve              # a keeper with an empty catalog
    python scripts/weightkeeper.py --add CKPT [options] # an entry loaded by the running keeper
    python scripts/weightkeeper.py --status ID | --drop ID   # an entry's status; an entry dropped
    python scripts/weightkeeper.py --serve --host-budget 200 --keep ID,...
                                                        # the entries at most 200 GB: a load drops the
                                                        # unleased ones (least recently leased first)
                                                        # but not those kept (--add CKPT --keep: that
                                                        # entry; --keep ID / --unkeep ID: the running
                                                        # keeper's)
    python scripts/weightkeeper.py --serve --save DIR   # each entry's converted segments written to DIR
                                                        # once, read back from there before the checkpoint
    python scripts/weightkeeper.py CKPT --name NAME     # a second keeper beside the default one
    python scripts/weightkeeper.py --check CKPT         # a round: map, compare rows with the shards
    python scripts/weightkeeper.py --gguf OUT CKPT      # Qwen: the tensors held as one GGUF file

Several keepers: --name NAME serves on /tmp/reclank-weightkeeper-<uid>-NAME.sock (without it the
default socket, as before); a round finds the keeper that holds its checkpoint among all of them,
or the one RECLANK_WEIGHTKEEPER=NAME names. Before it takes a page a keeper checks what it will hold
against the memory available (each node and the host) and refuses with what is missing. As root
a keeper gives back only the pool pages it added; the last root keeper restores the settings.
Wan 2.2 T2V-A14B (a diffusers directory with two experts: reclank/models/wan/keep.py): both
experts' F20 rows, 71.4 GB on --dense-node.

Options: --share (the rows of part 0, default 0.5: balanced channels),
--layers 0-3 (a test), --dense (the dense tensors in a segment too, on
--dense-node; a GGUF file: its dense and small tensors too, else the rounds
read them in the file), --pages 1g|2m|4k (the largest page), --threads, --socket,
--user/--group (the uid to drop to; default SUDO_UID/SUDO_GID), --seal size|write
(size, the default: the rounds map the segments writable so CUDA page-locks
them; write: read-only maps, the copies to the GPU stage). Qwen: the least
type of each kind of tensor, --dense-type, --experts-type, --ngram-type
(bf12, the default: the bfloat16 values in 12.25 bits; bf16; q8; rq8 for the
experts). A Gemma 4 GGUF file: --q4x auto|on|off (auto, the default: the Q4_0
matrices as KQ_Q4X, the only copy of them; off: as the file has them).
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
    against the bytes of the file (a Q4_0 matrix the keeper holds as KQ_Q4X:
    the file's bytes packed again)."""
    from reclank.backends.cpu import cops
    from reclank.formats.gguf import GGUF, Q4_0, Q4X
    from reclank.models.qwen import keep
    f = GGUF(path, stage=False)
    d = g.index["tensors"]
    rng = random.Random(1)
    bad = done = 0

    def ref_of(name, b, rows):
        if d[name]["type"] == Q4X and d[name].get("src") == Q4_0:
            b = b.copy()
            keep._q4x_pack_bytes(b, rows, int(d[name]["dims"][0]))
        return b
    nodes = [x for x in d if d[x]["place"] == "node"]     # (none: the dense tensors in the file)
    for name in rng.sample(nodes, min(n, 64, len(nodes))):
        done += 1
        ref = f.raw(name)[0].view(np.uint8).reshape(-1)
        bad += ref_of(name, ref, int(d[name]["dims"][1]) if len(d[name]["dims"]) == 2 else 1).tobytes() != \
            g.w.tensor(name).tobytes()
    xs = [x for x in d if d[x]["place"] == "experts"]
    for _ in range(n if xs else 0):     # (a dense model: no experts)
        name = rng.choice(xs)
        e = rng.randrange(d[name]["E"])
        ref = ref_of(name, f.raw(name)[0].view(np.uint8).reshape(d[name]["E"], -1)[e], d[name]["rows"])
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


def check_wan(ckpt, path, n):
    """Wan A14B (models/wan/keep.py): n random tensors of each expert (and every block's first and
    last matrix) against a new conversion of the shards (to_f20's bytes, or the float32 as stored)."""
    from reclank.backends.cpu import cops
    from reclank.formats.safetensors import Shards
    w = Z.attach(path, ckpt=ckpt)
    d = w.index["tensors"]
    rng = random.Random(1)
    bad = done = 0
    for sub in ("transformer", "transformer_2"):
        sh = Shards(os.path.join(ckpt, sub), "diffusion_pytorch_model.safetensors.index.json")
        keys = [k for k in d if k.startswith(sub + "/")]
        pick = rng.sample(keys, min(n, len(keys))) + [k for k in keys if k.endswith((".attn1.to_q.weight",
                                                                                     ".ffn.net.2.weight"))][:4]
        for k in pick:
            t, name = d[k], k[len(sub) + 1:]
            a = np.ascontiguousarray(sh.get(name), np.float32)
            ref = cops.kq_f32_to_f20(a.reshape(t["rows"], t["cols"]), t["cols"])[0] if t["kind"] == "f20" else a
            done += 1
            if w.tensor(k).tobytes() != ref.tobytes():
                bad += 1
                print("MISMATCH %s" % k)
        sh.close()
    print("%d tensors compared with a new conversion: %s" % (done, "all equal" if not bad else "%d differ" % bad))
    return 1 if bad else 0


def check(ckpt, path, n):
    t0 = time.time()
    c = Z.request("status", path)
    with c:
        st = json.loads(Z._recv_msg(c))
    if st.get("model") == "wan-a14b":
        return check_wan(ckpt, path, n)
    if st.get("layout") == Z.LAYOUT_TENSORS:
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
    ap.add_argument("--q4x", choices=("auto", "on", "off"), default="auto",
                    help="a Gemma 4 GGUF file: its Q4_0 matrices as KQ_Q4X (auto, on), the only copy the "
                         "runtime keeps, or as the file has them (off)")
    ap.add_argument("--copies", choices=("auto", "split", "both"), default="auto",
                    help="an EXL3 directory: the experts split between the nodes, or whole on every node "
                         "(auto: when they fit)")
    ap.add_argument("--seal", choices=("size", "write"), default="size",
                    help="the seals of the segments: size (the rounds map them writable, so CUDA page-locks "
                         "them for DMA) or write (read-only maps; the copies to the GPU stage)")
    ap.add_argument("--socket")
    ap.add_argument("--name", help="the keeper's name (its own socket); --stop / --status: that keeper")
    ap.add_argument("--list", action="store_true", help="the keepers running (name, model, memory held)")
    ap.add_argument("--user", type=int)
    ap.add_argument("--group", type=int)
    ap.add_argument("--status", nargs="?", const=True, metavar="ID",
                    help="the keeper's status (an entry's: its ID or checkpoint)")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--serve", action="store_true", help="a keeper with an empty catalog (--add fills it)")
    ap.add_argument("--add", action="store_true", help="CKPT (with the options) loaded by the running keeper "
                    "as an entry of its catalog, in the background (--status ID: how far)")
    ap.add_argument("--drop", metavar="ID", help="an entry dropped from the keeper's catalog (its ID or checkpoint)")
    ap.add_argument("--host-budget", type=float, metavar="GB",
                    help="the entries together at most GB: a load over it (or one the memory cannot take) drops "
                         "the entries no server leases, the least recently leased first (none: nothing is dropped)")
    ap.add_argument("--keep", nargs="?", const=True, metavar="ID,...",
                    help="entries never dropped for the budget: alone, the entry this command loads; IDs or "
                         "checkpoints, a keeper's (at its start, or of the running keeper)")
    ap.add_argument("--unkeep", metavar="ID", help="an entry the running keeper may drop again")
    ap.add_argument("--save", metavar="DIR", help="the converted-weights cache (none by default): each entry's "
                    "segments written to DIR once, and read back from there (O_DIRECT) before the checkpoint")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--gguf", metavar="OUT", help="Qwen: write the tensors the weightkeeper holds for "
                    "CKPT as one GGUF file (their types: BF12 by default), then exit")
    ap.add_argument("--check-n", type=int, default=20)
    a = ap.parse_args()
    if a.list:
        ks = Z.keepers()
        for name, path, st in ks:
            for e in st.get("entries") or ([st] if st.get("checkpoint") else []):
                use = ("leased by %s" % ",".join(map(str, e["leases"])) if e.get("leases") else
                       "idle %.0f s" % e["idle_s"] if e.get("idle_s") is not None else "")
                print("%-16s pid %-8s %-28s %-16s %7.1f GB%s%s  %s  %s  (%s)" % (
                    name or "(default)", st.get("pid"), e.get("id"), e.get("model") or e.get("status"),
                    sum(x["gb"] for x in e.get("segments", [])), " locked" if e.get("locked") else "",
                    " kept" if e.get("kept") else "", use, e.get("checkpoint") or e.get("error"), path))
            if st.get("host_budget_gb") is not None:
                print("%-16s the host budget %.1f GB, %.1f GB held" % ("", st["host_budget_gb"], st["held_gb"]))
            if not st.get("entries") and not st.get("checkpoint"):
                print("%-16s pid %-8s (no entries)  (%s)" % (name or "(default)", st.get("pid"), path))
        if not ks:
            print("no weightkeeper running")
        return 0
    sock = a.socket or (Z.socket_path(a.name) if a.name else None)
    keys = a.keep.split(",") if isinstance(a.keep, str) else None
    if keys and not a.ckpt and not a.serve or a.unkeep:
        out = 0
        for k in keys or a.unkeep.split(","):
            c = Z.request(("keep " if keys else "unkeep ") + (os.path.abspath(k) if os.sep in k else k), sock)
            with c:
                r = json.loads(Z._recv_msg(c))
            print(json.dumps(r))
            out |= "error" in r
        return out
    if a.status or a.stop or a.drop:
        op = ("status" + ("" if a.status is True else " " + a.status) if a.status
              else "drop " + a.drop if a.drop else "stop")
        c = Z.request(op, sock)
        with c:
            out = Z._recv_msg(c).decode()
        print(out if a.stop else json.dumps(json.loads(out), indent=1))
        return 0
    if a.serve:
        return Z.run(Z.Options(ckpt=None, threads=a.threads, sock=a.socket, name=a.name, user=a.user,
                               group=a.group, lock=a.lock, host_budget=a.host_budget, keep_keys=keys,
                               save=a.save)) or 0
    if not a.ckpt:
        ap.error("the checkpoint directory")
    if a.check:
        # (no socket or name: the keeper that holds CKPT, else the default one)
        return check(a.ckpt, sock or Z.find(a.ckpt)[0], a.check_n)
    if a.gguf:
        from reclank.models.qwen import keep
        keep.export_gguf(keep.KeptGGUF(a.ckpt, Z.attach(sock, ckpt=a.ckpt)), a.gguf,
                         log=lambda m: print(m, flush=True))
        return 0
    o = Z.Options(ckpt=a.ckpt, share=a.share, nodes=a.nodes, layers=a.layers, threads=a.threads,
                  lock=a.lock, dense=a.dense, dense_node=a.dense_node,
                  maxpage={"1g": Z.G1, "2m": Z.M2, "4k": Z.K4}[a.pages], layout=a.layout, sock=a.socket,
                  dense_type=a.dense_type, experts_type=a.experts_type, ngram_type=a.ngram_type, counts=a.counts,
                  copies=a.copies, seal=a.seal, q4x=a.q4x, name=a.name,
                  user=a.user, group=a.group, host_budget=a.host_budget, keep=a.keep is True, keep_keys=keys,
                  save=a.save)
    if a.add:
        o.ckpt = os.path.abspath(o.ckpt)
        d = {k: v for k, v in vars(o).items() if k not in ("sock", "name", "user", "group", "host_budget",
                                                           "keep_keys", "save")}
        c = Z.request("load " + json.dumps(d), sock)
        with c:
            r = json.loads(Z._recv_msg(c))
        print(json.dumps(r))
        return 1 if "error" in r else 0
    return Z.run(o) or 0


if __name__ == "__main__":
    sys.exit(main())
