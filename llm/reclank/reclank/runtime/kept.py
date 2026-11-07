"""A model's host arrays in the weightkeeper (docs/GPU_SWAP_PLAN.md 3.7, step 4): what a model makes at
its load (its converted linears: F20, BF12 rows; its float32 tensors) made once, held by the keeper,
and mapped by every later load (a load a map: no read, no conversion).

The model's own loader makes the bytes, so a kept array is the bytes that loader makes, by
construction: the keeper runs no model code. A load:

    k = Kept("zimage-dit", path, dict(gpu=gpu), code=[__name__])
    lin = k.linear(name, lambda: <the linear as the model makes it>)
    a = k.array(name, lambda: <the array>)
    k.done()

- **read**: the keeper holds the entry (or its saved copy, --save: restored): each array and linear
  a read-only view of the keeper's segment; make is not called.
- **write**: a keeper answers but has no entry: each make()'s arrays copied into one memfd of this
  process (on the GPU's node, 2 MB pages where allowed), the model holding views of it (one copy, not
  two); done() seals its size and puts it in the keeper (the connection this process's lease). A
  later load maps it.
- **off** (the default): RECLANK_KEPT is not 1, or no keeper answers: make() as before, nothing
  kept. Opt-in, so that a keeper (the default socket: RECLANK_WEIGHTKEEPER=NAME for another) holds a
  server's models only when asked: a put entry stays in the keeper after the server exits, until a
  drop or the keeper's --host-budget takes it.

The entry's ID: the label and a hash of the key: the checkpoint's files (name, size, mtime), the
options the model passes (those that shape its arrays: gpu, f20, ...), and the code (the sources of
the packages named in code, the encoders' runtime, ops, this module; the CPU library's name, a hash
of every C source; the keeper's VERSION). In read mode a name the entry lacks, or an array of
another shape, is made (and said): the code key makes that a bug.
"""
from __future__ import annotations

import ctypes
import hashlib
import importlib
import json
import os
import socket
import sys
import time

import numpy as np

from reclank.runtime import numa
from reclank.runtime import weightkeeper as Z

RESERVE = 1 << 40       # the address range a write mode's memfd grows in
GROW = 256 << 20


def _files(path):
    """(name, size, mtime) of the checkpoint's files: path's own, or the files of its directory (one
    level)."""
    if os.path.isfile(path):
        st = os.stat(path)
        return [[os.path.basename(path), st.st_size, st.st_mtime_ns]]
    out = []
    for f in sorted(os.listdir(path)):
        p = os.path.join(path, f)
        if os.path.isfile(p):
            st = os.stat(p)
            out.append([f, st.st_size, st.st_mtime_ns])
    return out


def _sources(code):
    """The text of the .py files of each package named (a module's package directory), and of the
    modules every load's conversions run."""
    dirs = set()
    for m in list(code) + ["reclank.encoders.runtime", "reclank.core.ops", __name__]:
        f = getattr(sys.modules.get(m) or importlib.import_module(m), "__file__", None)
        if f:
            dirs.add(os.path.dirname(f) if m in code else f)
    out = []
    for d in sorted(dirs):
        fs = [os.path.join(d, x) for x in sorted(os.listdir(d)) if x.endswith(".py")] if os.path.isdir(d) else [d]
        out += [[os.path.relpath(f, os.path.dirname(os.path.dirname(Z.__file__))), open(f).read()] for f in fs]
    return out


def entry_key(label, path, options, code):
    from reclank.backends.cpu import cops
    return dict(label=label, path=os.path.realpath(path), files=_files(path), options=options or {},
                code=hashlib.sha1(json.dumps(_sources(code)).encode()).hexdigest(),
                lib=os.path.basename(str(cops._path)), version=Z.VERSION)


class _Writer:
    """One memfd growing at a reserved address: the arrays copied in, each at 64 bytes."""

    def __init__(self, label, node):
        self.fd = Z.memfd_create(("kept-" + label)[:200], Z.MFD_CLOEXEC | Z.MFD_ALLOW_SEALING)
        self.resv = Z._mmap(None, RESERVE, Z.PROT_NONE, Z.MAP_PRIVATE | Z.MAP_ANONYMOUS | Z.MAP_NORESERVE)
        self.base = Z._up(self.resv, Z.M2)
        self.node, self.mapped, self.used = node, 0, 0

    def put(self, a):
        """[offset, dtype, shape] of a copied in, and the view of it."""
        a = np.ascontiguousarray(a)
        off = Z._up(self.used, Z.ALIGN)
        end = off + a.nbytes
        if end > self.mapped:
            new = Z._up(max(end, self.mapped + GROW), Z.M2)
            os.ftruncate(self.fd, new)
            Z._mmap(self.base + self.mapped, new - self.mapped, Z.PROT_READ | Z.PROT_WRITE,
                    Z.MAP_SHARED | Z.MAP_FIXED, self.fd, self.mapped)
            if numa.enabled():
                numa._mbind(self.base + self.mapped, new - self.mapped, self.node, policy=numa.MPOL_BIND)
            Z._libc.madvise(self.base + self.mapped, new - self.mapped, Z.MADV_HUGEPAGE)
            self.mapped = new
        v = _view(self.base, off, a.dtype, a.shape)
        v.flags.writeable = True
        v[...] = a
        v.flags.writeable = False
        self.used = end
        return [off, a.dtype.str, list(a.shape)], v

    def seal(self):
        """The memfd cut to its size (2 MB) and sealed for size; the rest of the range given back.
        Return its bytes."""
        size = Z._up(max(self.used, 1), Z.M2)
        if self.mapped > size:
            Z._libc.munmap(self.base + size, self.mapped - size)
            os.ftruncate(self.fd, size)
        tail = self.resv + RESERVE - (self.base + size)
        if self.base > self.resv:
            Z._libc.munmap(self.resv, self.base - self.resv)
        Z._libc.munmap(self.base + size, tail)
        import fcntl
        fcntl.fcntl(self.fd, Z.F_ADD_SEALS, Z.SEALS_SIZE)
        return size


def _view(base, off, dtype, shape):
    dt = np.dtype(dtype)
    n = int(np.prod(shape)) * dt.itemsize
    a = np.frombuffer((ctypes.c_uint8 * n).from_address(base + off), np.uint8).view(dt).reshape(shape)
    a.flags.writeable = False
    return a


class Kept:
    """A model's host arrays from the weightkeeper, or made and put there (the module's text)."""

    def __init__(self, label, path, options=None, code=(), node=0, log=None):
        self.label, self.path = label, path
        self.log = log or (lambda m: print(m, flush=True))
        self.mode, self.w, self.wr, self.items = None, None, None, {}
        self.made = self.read = 0
        self.t0 = time.time()
        if os.environ.get("RECLANK_KEPT", "0") != "1":
            return
        self.sock = Z.socket_path(Z.env_name())
        if not Z._answers(self.sock):
            return
        self.key = entry_key(label, path, options, code)
        self.id = "%s-%s" % (label, hashlib.sha1(json.dumps(self.key, sort_keys=True).encode()).hexdigest()[:8])
        try:
            self.w = self._get()
        except (OSError, ValueError) as e:
            self.log("kept %s: the weightkeeper: %s" % (self.id, e))
            return
        if self.w is not None:
            self.mode, self.items = "read", self.w.index["items"]
        else:
            self.mode, self.wr = "write", _Writer(label, node)

    @classmethod
    def off(cls):
        """A Kept that keeps nothing (make() as before)."""
        k = cls.__new__(cls)
        k.mode, k.w, k.wr, k.items, k.made, k.read, k.t0 = None, None, None, {}, 0, 0, time.time()
        k.log = lambda m: None
        return k

    def _get(self):
        """The entry mapped (restored from the keeper's saved copy first when it has one), or None."""
        def ask(op):
            c = Z.request(op, self.sock)
            with c:
                return json.loads(Z._recv_msg(c))
        r = ask("restore " + self.id)
        while r.get("status") == "loading":
            time.sleep(0.2)
            r = ask("restore " + self.id)
        if r.get("status") != "ready":
            return None
        w = Z.Weights(self.sock, key=self.id)
        if w.index.get("layout") != Z.LAYOUT_STORE or w.index.get("key") != json.loads(json.dumps(self.key)):
            w.end_lease()
            raise ValueError("the entry %s holds another key" % self.id)
        return w

    # ---- the arrays

    def _spec_view(self, spec):
        off, dt, shape = spec
        return _view(self.w.base[0], off, dt, shape)

    def array(self, name, make):
        """The array of name: the keeper's (read), or make()'s (copied into the memfd: write)."""
        if self.mode == "read":
            spec = self.items.get(name)
            if spec is not None and "a" in spec:
                self.read += 1
                return self._spec_view(spec["a"])
            self.log("kept %s: %s not in the entry: made" % (self.id, name))
            return make()
        a = make()
        if self.mode == "write" and isinstance(a, np.ndarray):
            spec, a = self.wr.put(a)
            self.items[name] = dict(a=spec)
            self.made += 1
        elif self.mode == "write":
            self._refuse(name, type(a))
        return a

    def linear(self, name, make):
        """A linear (an object of arrays, _KQ matrices and plain values) of name: rebuilt from the
        keeper's arrays (read), or make()'s, its arrays moved into the memfd (write)."""
        from reclank.models.gemma4.e4b import _KQ
        if self.mode == "read":
            spec = self.items.get(name)
            if spec is None or "lin" not in spec:
                self.log("kept %s: %s not in the entry: made" % (self.id, name))
                return make()
            mod, cls = spec["cls"]
            lin = object.__new__(getattr(importlib.import_module(mod), cls))
            for k, s in spec["lin"].items():
                if "a" in s:
                    v = self._spec_view(s["a"])
                elif "kq" in s:
                    v = _KQ(self._spec_view(s["kq"]), *s["t"])
                elif "np" in s:
                    v = np.dtype(s["np"]).type(s["v"])
                else:
                    v = tuple(s["v"]) if s.get("tuple") else s["v"]
                setattr(lin, k, v)
            self.read += 1
            return lin
        lin = make()
        if self.mode != "write":
            return lin
        attrs = {}
        for k, v in vars(lin).items():
            if isinstance(v, np.ndarray):
                spec, nv = self.wr.put(v)
                attrs[k] = dict(a=spec)
                setattr(lin, k, nv)
            elif isinstance(v, _KQ):
                spec, nv = self.wr.put(np.asarray(v.data))
                attrs[k] = dict(kq=spec, t=[int(v.type), int(v.rows), int(v.cols)])
                v.data = nv
            elif isinstance(v, np.generic):
                attrs[k] = dict(np=v.dtype.str, v=v.item())
            elif v is None or isinstance(v, (bool, int, float, str)):
                attrs[k] = dict(v=v)
            elif isinstance(v, tuple) and all(isinstance(x, (int, np.integer)) for x in v):
                attrs[k] = dict(v=[int(x) for x in v], tuple=True)
            else:
                self._refuse("%s.%s" % (name, k), type(v))
                return lin
        self.items[name] = dict(lin=attrs, cls=[type(lin).__module__, type(lin).__qualname__])
        self.made += 1
        return lin

    def _refuse(self, name, t):
        """A value the entry cannot hold: nothing put (the arrays made so far stay this process's)."""
        self.log("kept %s: %s is a %s: not kept" % (self.id, name, t.__name__))
        self.mode = "refused"

    def done(self):
        """read: say so. write: the memfd sealed and put in the keeper (this process's lease)."""
        dt = time.time() - self.t0
        if self.mode == "read":
            self.log("kept %s: %d arrays and linears from the weightkeeper (%.1f GB) in %.1f s" % (
                self.id, self.read, self.w.size[0] / 1e9, dt))
            return
        if self.mode != "write" or not self.items:
            return
        size = self.wr.seal()
        index = dict(version=Z.VERSION, layout=Z.LAYOUT_STORE, model=self.label, id=self.id, key=self.key,
                     checkpoint=os.path.realpath(self.path), items=self.items, part_bytes=[size],
                     nodes=[self.wr.node], types=dict(dense="as made", experts="none", ngram="none"))
        try:
            c = Z.request("put " + json.dumps(index), self.sock)
            socket.send_fds(c, [b"F"], [self.wr.fd])
            r = json.loads(Z._recv_msg(c))
        except (OSError, ValueError) as e:
            r = dict(error=str(e))
            c = None
        os.close(self.wr.fd)
        if "error" in r:
            if c is not None:
                c.close()
            self.log("kept %s: not put in the weightkeeper: %s (%.1f GB made here)" % (self.id, r["error"],
                                                                                      size / 1e9))
            return
        Z._LEASES.append(c)
        self.log("kept %s: %d arrays and linears made (%.1f s) and put in the weightkeeper (%.1f GB)" % (
            self.id, self.made, dt, size / 1e9))
