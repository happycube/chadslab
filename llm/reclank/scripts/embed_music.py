#!/usr/bin/env python3
"""A music library in the music embeddings (docs/MUSIC_EMBED_PLAN.md:
MuQ-MuLan and CLAP side by side, reclank/models/music/model.py): index the
files of a directory, then search them by text, find more like a track, or
group them.

    index DIR --lib LIB      each audio file under DIR: two windows of 30 s
                             (centered at 20 % and at 60 % of its length; a
                             file of at most 60 s whole), each window's vector,
                             and the track's (the mean of its windows, each
                             model's half normalized); the fingerprints of
                             the duplicates. LIB/library.npz and
                             LIB/library.json; a second run embeds only the
                             new or changed files.
    query LIB "text" [--top 10] [--part muq|clap]
                             (--part: one model's half alone; MuQ-MuLan
                             knows names and instruments better, CLAP genres)
    like LIB NAME [--top 10] the tracks nearest to a track (a part of its
                             name or path); its duplicates left out
    clusters LIB [--k 20]    k-means (cosine) of the tracks
    dups LIB                 the groups of the same recording

Duplicates: Chromaprint (ffmpeg's chromaprint muxer) of the first 120 s of
each file at its own speed and at 0.97 ... 1.03 (asetrate: pitch and tempo
together, as a tape run fast or slow); two files are the same recording when
their words, aligned at the most common offset of the words they share,
differ in less than 20 % of their bits over at least 160 words (20 s). The
fingerprints stay in LIB/fp.npz.

The towers run in this process (--cpu: on the CPU; else the audio towers on
the GPU in BF12). Run a GPU index with OMP_NUM_THREADS=1.

    python scripts/embed_music.py index ~/Music --lib lib/ --models models/music
    python scripts/embed_music.py query lib/ "solo piano" --models models/music
    python scripts/embed_music.py like lib/ "Court Minstrel"
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXT = {".mp3", ".m4a", ".webm", ".ogg", ".opus", ".flac", ".wav", ".aac", ".wma", ".mka", ".mp4"}
WIN_S, AT = 30.0, (0.2, 0.6)
FP_SECS, FP_SPEEDS = 120, (1.0, 0.97, 0.98, 0.99, 1.01, 1.02, 1.03)
FP_SHARED, FP_OVERLAP, FP_BER = 20, 160, 0.20
POP16 = np.array([bin(i).count("1") for i in range(1 << 16)], np.uint8)


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def duration(path):
    from reclank.models.music.frontend import tool_env
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, env=tool_env())
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def windows(dur):
    """The (start, end) seconds of a track's windows: centered at AT of its
    length (inside the track)."""
    if dur <= 2 * WIN_S:
        return [(0.0, dur)]
    out = []
    for a in AT:
        s = round(min(max(dur * a - WIN_S / 2, 0.0), dur - WIN_S), 3)
        out.append((s, s + WIN_S))
    return out


def load_lib(lib):
    p = os.path.join(lib, "library.json")
    if not os.path.exists(p):
        return [], None
    meta = json.load(open(p))
    d = np.load(os.path.join(lib, "library.npz"))
    return meta["tracks"], {k: d[k] for k in d.files}


# ---- index ------------------------------------------------------------------

_BN = None


def _worker(cpus, bn):
    """The initializer of a worker process: the CPUs of the taskset (the
    main thread, bound by OpenMP, has one), CLAP's BatchNorm of the bands."""
    global _BN
    _BN = bn
    try:
        os.sched_setaffinity(0, cpus)
    except OSError:
        pass


def prepare(path):
    """In a worker process (reclank.models.music.frontend: NumPy alone): the
    duration, and each window's inputs of the programs, the mel of each
    MuQ clip and the patches of each CLAP clip."""
    from reclank.models.music import frontend as F
    dur = duration(path)
    if not dur or dur < 1.0:
        return path, dur, None
    out = []
    for a, b in windows(dur):
        try:
            s24, s48 = F.decode(path, F.MUQ_RATE, a, min(b, dur)), F.decode(path, F.CLAP_RATE, a, min(b, dur))
            preps = {"muq": [F.muq_mel(c).reshape(-1, 1) for c in F.clips(s24, F.MUQ_RATE)],
                     "clap": [F.clap_patches(c, _BN) for c in F.clips(s48, F.CLAP_RATE)]}
        except ValueError:
            return path, dur, None
        out.append((a, b, preps))
    return path, dur, out


def prepared(files, workers, bn):
    """prepare() of each file in worker processes, in order, at most 3
    workers' worth ahead."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    from reclank.backends.cpu import cops
    cpus = cops.place_cpus() or os.sched_getaffinity(0)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_worker,
                             initargs=(cpus, bn)) as ex:
        ahead = collections.deque()
        it = iter(files)
        for f in it:
            ahead.append(ex.submit(prepare, f))
            if len(ahead) >= 3 * workers:
                break
        while ahead:
            yield ahead.popleft().result()
            f = next(it, None)
            if f is not None:
                ahead.append(ex.submit(prepare, f))


def embedded(m, items, timer):
    """(path, dur, wins, vw) of each (path, dur, wins) of prepared(), in order;
    vw: each window's {model: its vector}. MuQ-MuLan's clips go in programs
    of muq.BATCH clips across the tracks (a program of 5 clips takes 28 ms a
    clip, of 1 clip 51 ms), CLAP's a clip at a time. timer[0] adds the time
    in the programs."""
    from reclank.models.music.model import PARTS
    from reclank.models.music.muq import BATCH
    muq, clap = m.audio_towers["muq"], m.audio_towers["clap"]
    pending = collections.deque()       # [path, dur, wins, the MuQ hidden states of each window, clap vectors]
    queue = []                          # (the MuQ hidden states of a window, clip, prep) not run

    def run(n):
        tb = time.time()
        part, queue[:] = queue[:n], queue[n:]
        for (hs, i, _), h in zip(part, muq.run_many([x for _, _, x in part], batch=BATCH)):
            hs[i] = h
        timer[0] += time.time() - tb

    def done():
        while pending and all(h is not None for hs in pending[0][3] for h in hs):
            path, dur, wins, hss, cv = pending.popleft()
            vw = []
            for hs, c in zip(hss, cv):
                v = muq.vectors(hs).mean(0)
                vw.append({"muq": v / np.linalg.norm(v), "clap": c})
            assert set(vw[0]) == set(PARTS)
            yield path, dur, wins, vw

    for path, dur, wins in items:
        if wins is None:
            pending.append([path, dur, None, [], []])
        else:
            tb = time.time()
            hss, cv = [], []
            for _, _, preps in wins:
                v = clap.vectors([clap.run(x) for x in preps["clap"]]).mean(0)
                cv.append(v / np.linalg.norm(v))
                hs = [None] * len(preps["muq"])
                queue.extend((hs, i, x) for i, x in enumerate(preps["muq"]))
                hss.append(hs)
            timer[0] += time.time() - tb
            pending.append([path, dur, wins, hss, cv])
            while len(queue) >= BATCH:
                run(BATCH)
        yield from done()
    run(len(queue))
    yield from done()


def index(a):
    from reclank.models.music.model import PARTS, MusicEmbedder, join
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(os.path.expanduser(a.dir)) for f in fs
                   if os.path.splitext(f)[1].lower() in EXT)
    old, arr = load_lib(a.lib)
    known = {t["path"]: (t, arr["track"][i], arr["win"][i]) for i, t in enumerate(old)
             if os.path.exists(t["path"]) and os.path.getsize(t["path"]) == t["size"]
             and os.path.getmtime(t["path"]) == t["mtime"]}
    todo = [f for f in files if f not in known]
    print("%d files: %d indexed, %d to embed" % (len(files), len(files) - len(todo), len(todo)), flush=True)
    os.makedirs(a.lib, exist_ok=True)
    new = {}
    if todo:
        m = MusicEmbedder(a.models, gpu=not a.cpu, text="off", cpu_threads=a.threads)
        t0 = time.time()
        timer = [0.0]       # the main thread's time in the programs (the rest: waiting for the workers)
        for k, (path, dur, wins, vw) in enumerate(embedded(m, prepared(todo, a.workers, m.audio_towers["clap"].bn),
                                                           timer)):
            if wins is None:
                print("skipped (no audio): %s" % path, flush=True)
                continue
            track = join({p: norm(np.mean([w[p] for w in vw], 0))[None] for p in PARTS})[0]
            win = np.stack([join({p: w[p][None] for p in PARTS})[0] for w in vw])
            if len(win) == 1:
                win = np.concatenate([win, win])
            st = os.stat(path)
            new[path] = (dict(path=path, name=os.path.basename(path), dur=dur, size=st.st_size,
                              mtime=st.st_mtime, windows=[[w[0], w[1]] for w in wins]), track, win)
            if (k + 1) % 50 == 0 or k + 1 == len(todo):
                dt = time.time() - t0
                print("%d / %d  %.2f tracks/s  (the programs %.2f s a track)" % (
                    k + 1, len(todo), (k + 1) / dt, timer[0] / (k + 1)), flush=True)
    rows = [known.get(f) or new.get(f) for f in files]
    rows = [r for r in rows if r is not None]
    tracks = [r[0] for r in rows]
    np.savez(os.path.join(a.lib, "library.npz"), track=np.stack([r[1] for r in rows]).astype(np.float32),
             win=np.stack([r[2] for r in rows]).astype(np.float32))
    from reclank.models.music.model import wide_cpus
    t0 = time.time()
    groups = fingerprint_groups([t["path"] for t in tracks], os.path.join(a.lib, "fp.npz"), a.workers, wide_cpus)
    print("fingerprints: %.1f s" % (time.time() - t0), flush=True)
    json.dump({"models": "MuQ-MuLan + CLAP (reclank/models/music)", "dim": 1024, "tracks": tracks,
               "dups": [[tracks[i]["path"] for i in g] for g in groups]},
              open(os.path.join(a.lib, "library.json"), "w"), indent=1, ensure_ascii=False)
    print("library: %d tracks, %d duplicate groups (%d tracks) -> %s" % (
        len(tracks), len(groups), sum(map(len, groups)), a.lib), flush=True)


# ---- fingerprints --------------------------------------------------------------

_ENV = None


def _fp(path, speed):
    global _ENV
    if _ENV is None:
        from reclank.models.music.frontend import tool_env
        _ENV = tool_env()
    af = ["-af", "aresample=44100,asetrate=%.1f,aresample=44100" % (44100 * speed)] if speed != 1.0 else []
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-t", str(FP_SECS / speed), "-i", path, "-ac", "1"] +
                       af + ["-f", "chromaprint", "-fp_format", "raw", "-"], capture_output=True, env=_ENV)
    return np.frombuffer(r.stdout, np.uint32).copy() if r.returncode == 0 else np.zeros(0, np.uint32)


def _ber(a, b, off):
    lo, hi = max(0, off), min(len(a), len(b) + off)
    if hi <= lo:
        return 1.0, 0
    x = a[lo:hi] ^ b[lo - off:hi - off]
    bits = POP16[x & 0xFFFF].astype(np.int64).sum() + POP16[x >> 16].astype(np.int64).sum()
    return bits / (32 * (hi - lo)), hi - lo


def fingerprint_groups(paths, cache, workers, init=None):
    """The groups (lists of indices of paths) of the same recording."""
    old = dict(np.load(cache, allow_pickle=False)) if os.path.exists(cache) else {}
    have = {p: i for i, p in enumerate(old.get("paths", np.array([], str)).tolist())}
    var = {}
    for sp in FP_SPEEDS:
        key = "%.3f" % sp
        prev = [old["w" + key][a:b] for a, b in zip(old["s" + key][:-1], old["s" + key][1:])] if "w" + key in old else []
        need = [p for p in paths if p not in have or have[p] >= len(prev)]
        with ThreadPoolExecutor(workers, initializer=init) as ex:
            got = dict(zip(need, ex.map(lambda p: _fp(p, sp), need)))
        var[sp] = [got[p] if p in got else prev[have[p]] for p in paths]
    save = {"paths": np.array(paths)}
    for sp, f in var.items():
        key = "%.3f" % sp
        save["w" + key] = np.concatenate(f) if f else np.zeros(0, np.uint32)
        save["s" + key] = np.cumsum([0] + [len(x) for x in f])
    np.savez(cache, **save)
    base = var[1.0]
    index = collections.defaultdict(list)
    for i, f in enumerate(base):
        for pos, w in enumerate(f):
            index[int(w)].append((i, pos))
    index = {w: h for w, h in index.items() if len(h) <= 50}     # a word in many tracks (silence) says nothing
    best = {}
    for sp, vf in var.items():
        offs = collections.defaultdict(collections.Counter)
        for i, f in enumerate(vf):
            for p, w in enumerate(f):
                for j, q in index.get(int(w), ()):
                    if j != i:
                        offs[i, j][p - q] += 1
        for (i, j), c in offs.items():
            if sum(c.values()) < FP_SHARED:
                continue
            e, ov = _ber(vf[i], base[j], c.most_common(1)[0][0])
            k = (min(i, j), max(i, j))
            if e < FP_BER and ov >= FP_OVERLAP and e < best.get(k, 1.0):
                best[k] = e
    parent = list(range(len(paths)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j in best:
        parent[root(i)] = root(j)
    groups = collections.defaultdict(list)
    for i in {i for k in best for i in k}:
        groups[root(i)].append(i)
    return [sorted(g) for g in groups.values()]


# ---- queries -------------------------------------------------------------------

def show(tracks, sims, order, top):
    for i in order[:top]:
        print("  %.3f  %s" % (sims[i], tracks[i]["name"]))


def query(a):
    from reclank.models.music.model import MusicEmbedder
    tracks, arr = load_lib(a.lib)
    m = MusicEmbedder(a.models, gpu=False, audio="off")
    if a.part:
        half = {"muq": slice(0, 512), "clap": slice(512, 1024)}[a.part]
        s = norm(arr["track"][:, half]) @ m.texts([a.text], parts=True)[a.part][0]
    else:
        s = arr["track"] @ m.texts([a.text])[0]
    show(tracks, s, np.argsort(-s), a.top)


def find(tracks, name):
    hits = [i for i, t in enumerate(tracks) if name.lower() in t["path"].lower()]
    if not hits:
        raise SystemExit("no track matches %r" % name)
    if len(hits) > 1:
        print("%d tracks match; the first: %s" % (len(hits), tracks[hits[0]]["name"]))
    return hits[0]


def like(a):
    tracks, arr = load_lib(a.lib)
    meta = json.load(open(os.path.join(a.lib, "library.json")))
    i = find(tracks, a.name)
    same = {i}
    for g in meta.get("dups", []):
        if tracks[i]["path"] in g:
            same |= {k for k, t in enumerate(tracks) if t["path"] in g}
    s = arr["track"] @ arr["track"][i]
    order = [k for k in np.argsort(-s) if k not in same]
    print("like %s%s" % (tracks[i]["name"], " (and %d copies left out)" % (len(same) - 1) if len(same) > 1 else ""))
    show(tracks, s, order, a.top)


def clusters(a):
    tracks, arr = load_lib(a.lib)
    X = arr["track"]
    rng = np.random.default_rng(0)
    C = X[rng.choice(len(X), a.k, replace=False)]
    for _ in range(50):
        lab = np.argmax(X @ C.T, 1)
        C = norm(np.stack([X[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(a.k)]))
    lab = np.argmax(X @ C.T, 1)
    for k in np.argsort(-np.bincount(lab, minlength=a.k)):
        idx = np.flatnonzero(lab == k)
        if not len(idx):
            continue
        core = idx[np.argsort(-(X[idx] @ C[k]))][:5]
        print("cluster %d: %d tracks" % (k, len(idx)))
        for i in core:
            print("    %s" % tracks[i]["name"])


def dups(a):
    meta = json.load(open(os.path.join(a.lib, "library.json")))
    for g in meta.get("dups", []):
        print(" | ".join(os.path.basename(p) for p in g))
    print("%d groups" % len(meta.get("dups", [])))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("index")
    s.add_argument("dir")
    s.add_argument("--lib", required=True)
    s.add_argument("--models", default="models/music", help="the directory with clap/ and muq-mulan/")
    s.add_argument("--cpu", action="store_true", help="the audio towers on the CPU (float32)")
    s.add_argument("--threads", type=int, default=None, help="the team of a CPU program")
    s.add_argument("--workers", type=int, default=12, help="threads of decoding and fingerprints")
    s.set_defaults(fn=index)
    s = sub.add_parser("query")
    s.add_argument("lib")
    s.add_argument("text")
    s.add_argument("--models", default="models/music")
    s.add_argument("--top", type=int, default=10)
    s.add_argument("--part", choices=("muq", "clap"), default=None, help="one model's half alone")
    s.set_defaults(fn=query)
    s = sub.add_parser("like")
    s.add_argument("lib")
    s.add_argument("name")
    s.add_argument("--top", type=int, default=10)
    s.set_defaults(fn=like)
    s = sub.add_parser("clusters")
    s.add_argument("lib")
    s.add_argument("--k", type=int, default=20)
    s.set_defaults(fn=clusters)
    s = sub.add_parser("dups")
    s.add_argument("lib")
    s.set_defaults(fn=dups)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
