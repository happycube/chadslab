"""The bytes of engram's rows of DeepSeek V4.1 Flash read from the shards (DSV41GPU._inputs: each token's 24 rows
of 256 codes and 8 scales of each engram layer, tables of 98 GB on the NVMe at random).

Fancy indexing of the shard's map read them by page faults, one at a time, each reading around it (128 KB of the
device's read-ahead: 175 KB from the NVMe a row of 264 bytes; a 1000-token prompt's 48000 rows 18.5 s cold, 8.6 GB
read, the page cache filled with rows no one asks for). EngramReader reads them through the page cache (the rows'
pages alone, ~2.8 KB a read; its pages stay for the next prompt that has the n-grams) from pools of workers made
once (cops.RowsPool) on the process's spare CPUs: by io_uring (URING; 4 rings of 64 for a prompt's rows, RINGS_NOW
for a step's), else, when io_uring cannot be set up, by pread (32 workers, 8 for a step). The real 16K prompt's
1.54M reads on the server's 2 spare CPUs (21, 45), cold: 4 rings 5.05 s and 9.7 CPU-s, 32 pread workers 6.52 s and
12.4 CPU-s (warm 1.19 and 1.35 s); on 47 CPUs 2.84 and 4.03 s (10.9 and 30 CPU-s; the NVMe's ~540K reads a second).
EngramAhead reads a prompt's rows on a thread of its own, a chunk of CHUNK tokens at a time and at most MOST tokens
before the groups that take them (Hashes.ids needs only the ids and the positions), while the GPU and the parts run
the groups before.

    rd = EngramReader(m.w.src, m.engram_layers)
    rd.read(hashes, [(codes, scales) of each layer])                 # hashes [T, layers, heads]
    ah = EngramAhead(rd, hashes, tokens, pos); ah.take(pos, n, outs)  # the rows of positions pos .. pos + n
"""
from __future__ import annotations

import os
import sys
import threading

import numpy as np

# io_uring through the page cache (RECLANK_DS_ENGRAM_URING=0: pread; pread too when io_uring cannot be set up)
URING = os.environ.get("RECLANK_DS_ENGRAM_URING", "1") != "0"
# a prompt's pool: io_uring RINGS rings of QD (the NVMe's ~540K reads a second with 4, the same with 8; qd 32 to 512
# the same), or THREADS pread workers (RECLANK_DS_ENGRAM_THREADS)
RINGS, QD = 4, 64
THREADS = int(os.environ.get("RECLANK_DS_ENGRAM_THREADS", "32"))
# a step's pool (a step's 96 reads, a verify group's up to 768): io_uring RINGS_NOW rings (RECLANK_DS_ENGRAM_RINGS_NOW),
# or 8 pread workers, a worker for 12 reads. A step's 96 reads on the server's 2 spare CPUs, cold / warm (medians of
# 64 steps of real text): 4 rings 1.19 / 0.31 ms, 1 ring 1.38 / 0.26, 8 pread workers of the pool 1.79 / 0.32 (made
# for each read 1.7 / 0.49; one read after another 10.1 / 0.23)
RINGS_NOW = int(os.environ.get("RECLANK_DS_ENGRAM_RINGS_NOW", "4"))
# a prompt's rows read ahead (RECLANK_DS_ENGRAM_AHEAD: the most tokens before the group that takes them; 0: none)
MOST = int(os.environ.get("RECLANK_DS_ENGRAM_AHEAD", "16384"))
CHUNK = 1024


class EngramReader:
    """The rows of engram's tables through the page cache (see the module text); avoid: the CPUs its workers keep
    off (the parts' teams); uring: io_uring (default URING)."""

    def __init__(self, src, layers, avoid=(), uring=None, quiet=False):
        from reclank.models import media
        self.tabs = []
        self._fds = {}
        for i in layers:
            pre = "layers.%d.engram.embed." % i
            f, _dt, shape, o0, _o1 = src.t[pre + "weight"]
            fs, _sdt, sshape, s0, _s1 = src.t[pre + "scale"]
            self.tabs.append((self._fd(src.path, f), o0, int(shape[1]), self._fd(src.path, fs), s0, int(sshape[1])))
        # the workers on the CPUs of the process's start (RECLANK_START_CPUS: the calling thread is bound to one,
        # and they would inherit it) less avoid (the parts' teams), else on all of them
        start = media.start_cpus() or set(os.sched_getaffinity(0))
        self.cpus = sorted(start - set(avoid)) or sorted(start)
        self.want_uring = URING if uring is None else bool(uring)
        self.quiet = quiet
        self._pools = {}

    def _fd(self, path, f):
        if f not in self._fds:
            self._fds[f] = os.open(os.path.join(path, f), os.O_RDONLY)
        return self._fds[f]

    def pool(self, ahead):
        """The pool of a prompt's reads (ahead) or of a step's, made at its first use (io_uring, or pread when
        it cannot be set up; said once)."""
        from reclank.backends.cpu import cops
        p = self._pools.get(ahead)
        if p is None:
            n = (RINGS if ahead else RINGS_NOW) if self.want_uring else (THREADS if ahead else 8)
            p = cops.RowsPool(n, self.want_uring, QD, self.cpus)
            if self.want_uring and not p.uring:
                p = cops.RowsPool(THREADS if ahead else 8, False, QD, self.cpus)
            self._pools[ahead] = p
            if not self.quiet:
                print("[dsv41] engram rows of %s: %s on CPUs %s" % (
                    "a prompt" if ahead else "a step",
                    "io_uring, %d rings of %d" % (p.workers, QD) if p.uring else "pread, %d workers%s" % (
                        p.workers, " (no io_uring here)" if self.want_uring else ""),
                    ",".join(map(str, p.cpus))), file=sys.stderr, flush=True)
        return p

    def outs(self, n):
        """New arrays for the rows of n tokens of each layer: (codes [n heads, 256], scales [n heads, 8])."""
        return [(np.empty((n, w), np.uint8), np.empty((n, ws), np.uint8)) for _fd, _o, w, _fs, _s, ws in self.tabs]

    def read(self, hashes, outs, ahead=False):
        """The rows of hashes [T, layers, heads] into outs (each layer's codes [T heads, 256] and scales
        [T heads, 8], contiguous), by the prompt's pool (ahead) or the step's."""
        fds, offs, lens, dsts = [], [], [], []
        for j, (fd, o0, w, fs, s0, ws) in enumerate(self.tabs):
            rows = np.ascontiguousarray(hashes[:, j], np.int64).reshape(-1)
            codes, scales = outs[j]
            assert codes.flags.c_contiguous and scales.flags.c_contiguous and codes.shape == (len(rows), w)
            n = len(rows)
            at = np.arange(n, dtype=np.int64)
            fds += [np.full(n, fd, np.int32), np.full(n, fs, np.int32)]
            offs += [o0 + rows * w, s0 + rows * ws]
            lens += [np.full(n, w, np.int64), np.full(n, ws, np.int64)]
            dsts += [codes.ctypes.data + at * w, scales.ctypes.data + at * ws]
        n = sum(len(a) for a in fds)
        p = self.pool(ahead)
        p.read(np.concatenate(fds), np.concatenate(offs), np.concatenate(lens), np.concatenate(dsts),
               p.workers if (ahead or p.uring) else max(1, min(p.workers, n // 12)))


class EngramAhead:
    """A prompt's rows read ahead on a thread (see the module text): hashes [T, layers, heads] of its tokens at the
    positions pos .. (dead: the tokens of image spans, as Hashes.ids had them, or None)."""

    def __init__(self, reader, hashes, tokens, pos, dead=None, most=None, chunk=CHUNK):
        self.rd, self.hashes, self.pos0, self.n = reader, hashes, int(pos), int(hashes.shape[0])
        self.tokens = np.asarray(tokens, np.int64)
        self.dead = None if dead is None else np.asarray(dead, bool)
        self.most, self.chunk = MOST if most is None else int(most), int(chunk)
        self.chunks = {}            # the first row of a chunk -> its outs
        self.ready = 0              # the rows read
        self.taken = 0              # the rows the groups took (the chunks below go)
        self.want = 0               # the end of the rows a group waits for
        self.stop, self.err = False, None
        self.cv = threading.Condition()
        self.th = threading.Thread(target=self._run, name="engram-ahead", daemon=True)
        self.th.start()

    def _run(self):
        try:
            for c0 in range(0, self.n, self.chunk):
                with self.cv:
                    while not self.stop and c0 - max(self.taken, self.want) > self.most:
                        self.cv.wait()
                    if self.stop:
                        return
                c1 = min(c0 + self.chunk, self.n)
                outs = self.rd.outs((c1 - c0) * self.hashes.shape[2])
                self.rd.read(self.hashes[c0:c1], outs, ahead=True)
                with self.cv:
                    self.chunks[c0] = outs
                    self.ready = c1
                    self.cv.notify_all()
        except BaseException as e:      # (take reads them itself)
            with self.cv:
                self.err = e
                self.cv.notify_all()

    def covers(self, pos, tokens, dead=None):
        """The rows of these tokens at pos .. are read here (the same ids and image tokens, so the same hashes:
        they need only the ids and the positions)."""
        a, n = int(pos) - self.pos0, len(tokens)
        if self.err is not None or self.stop or not (0 <= a and a + n <= self.n):
            return False
        c = a - a % self.chunk
        with self.cv:
            kept = c >= self.ready or c in self.chunks
        if not kept or not np.array_equal(self.tokens[a:a + n], np.asarray(tokens, np.int64)):
            return False
        mine = None if self.dead is None else self.dead[a:a + n]
        if dead is None or not np.any(dead):
            return mine is None or not mine.any()
        return mine is not None and np.array_equal(mine, np.asarray(dead, bool))

    def take(self, pos, n, outs):
        """The rows of positions pos .. pos + n into outs (as EngramReader.read's), once read; False when the
        thread failed (the caller reads them)."""
        a, b, h = int(pos) - self.pos0, int(pos) - self.pos0 + int(n), self.hashes.shape[2]
        with self.cv:
            self.want = max(self.want, b)
            self.cv.notify_all()
            while self.ready < b and self.err is None and not self.stop:
                self.cv.wait()
            if self.ready < b:
                return False
            for c0 in sorted(self.chunks):
                c1 = min(c0 + self.chunk, self.n)
                lo, hi = max(a, c0), min(b, c1)
                if lo >= hi:
                    continue
                for (dc, ds), (sc, ss) in zip(outs, self.chunks[c0]):
                    dc[(lo - a) * h:(hi - a) * h] = sc[(lo - c0) * h:(hi - c0) * h]
                    ds[(lo - a) * h:(hi - a) * h] = ss[(lo - c0) * h:(hi - c0) * h]
            self.taken = max(self.taken, b)
            for c0 in [c for c in self.chunks if c + self.chunk <= self.taken]:
                del self.chunks[c0]
            self.cv.notify_all()
        return True

    def close(self):
        with self.cv:
            self.stop = True
            self.chunks = {}
            self.cv.notify_all()
