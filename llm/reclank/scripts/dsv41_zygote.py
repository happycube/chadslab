"""The zygote of DeepSeek V4.1 Flash (reclank/runtime/zygote.py, DEEPSEEK_V41_PLAN.md
4.2): the expert rows of every node loaded once, handed to the rounds.

    python scripts/dsv41_zygote.py CKPT                 # as the user: pools filled beforehand, else 4 KB
    sudo .venv/bin/python scripts/dsv41_zygote.py CKPT --lock
                                                        # 1 GB pages, the memory locked in full,
                                                        # then it serves as the invoking user
    python scripts/dsv41_zygote.py --status | --stop
    python scripts/dsv41_zygote.py --check CKPT         # a round: map, compare rows with the shards

Options: --share (the rows of part 0, default 0.5: balanced channels),
--layers 0-3 (a test), --dense (the dense tensors in a segment too, on
--dense-node), --pages 1g|2m|4k (the largest page), --threads, --socket,
--user/--group (the uid to drop to; default SUDO_UID/SUDO_GID).
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

from reclank.runtime import zygote as Z  # noqa: E402


def _ints(text):
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += range(int(a), int(b or a) + 1)
    return out


def check(ckpt, path, n):
    t0 = time.time()
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
    ap.add_argument("--socket")
    ap.add_argument("--user", type=int)
    ap.add_argument("--group", type=int)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--stop", action="store_true")
    ap.add_argument("--check", action="store_true")
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
    o = Z.Options(ckpt=a.ckpt, share=a.share, nodes=a.nodes, layers=a.layers, threads=a.threads,
                  lock=a.lock, dense=a.dense, dense_node=a.dense_node,
                  maxpage={"1g": Z.G1, "2m": Z.M2, "4k": Z.K4}[a.pages], layout=a.layout, sock=a.socket,
                  user=a.user, group=a.group)
    return Z.run(o) or 0


if __name__ == "__main__":
    sys.exit(main())
