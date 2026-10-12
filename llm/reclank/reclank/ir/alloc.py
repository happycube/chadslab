"""Pack the temporaries of a program into one array by their liveness.

A compiler makes a buffer for each result (Compiler.buffer: prog.temps;
the Qwen steps with model.pack_temps, and QwenGPU._tmp).
Most of them live a few records: the norm of a layer, the products, the
scratch of a kernel. pack() gives the temporaries that do not live at the
same time the same bytes of one array, the arena, and rewrites the
operands that point into them. The device copy (gpu.Mirror) of the arena
then replaces the copies of all those buffers: the step of the 26B kept
10 MB of temporaries for 0.3 MB live at once, a group of 16 tokens 177 MB
for 5.4 MB.

A temporary lives from its first record to its last, plus one record (a
kernel may start while the one before it ends: programmatic dependent
launch). It goes into the arena only when

- the compiler made it (prog.temps), and the caller does not read it (the
  names of exports: x, xn, ...);
- every record that uses it has an opcode of WRITES (whose kernels read
  and write their arrays in the order of the records), and its first
  record writes it (an operand of WRITES): then no record reads what an
  earlier temporary left in its bytes, or what an earlier run left (a
  buffer the compiler filled is a constant that its first record reads:
  it stays).

The kernels read the same values, so the bits are those of the program
before. RECLANK_PACK=0 turns it off.
"""
from __future__ import annotations

import os

import numpy as np

from reclank.ir import program as P

PACK = os.environ.get("RECLANK_PACK", "1") != "0"
# RECLANK_PACK_POISON=1 (a test): the arena starts as NaN in place of zeros,
# so a record that reads a temporary before any record wrote it gives NaN
# (the goldens with it keep their bits)
POISON = os.environ.get("RECLANK_PACK_POISON", "0") == "1"

# The names that the callers of a program read or write (upload, download,
# the device address of the last rows): never packed.
EXPORTS = ("x", "xn", "lim", "out", "prob", "tok", "token", "xin", "h", "e")

# opcode -> the operands its kernels write wholly (before any read of them).
# The other array operands of these opcodes are read, or written in place
# after a record before wrote them.
WRITES = {
    P.RMS_NORM: (2,),
    P.ADD: (2,),
    P.MUL_S: (2,),
    P.COPY: (1,),
    P.GELU: (1,),
    P.MUL: (2,),
    P.INT4_LINEAR: (3,),
    P.INT4_LINEAR_MT: (3,),
    P.INT4_MULTI4: (4, 8, 12, 16),
    P.INT4_MULTI4_MT: (5, 9, 13, 17),
    P.RMS_NORM_MULTI4: (2, 7, 11, 15, 19),
    P.GELU_MUL_INT4: (3, 6),
    P.GELU_MUL_ROWS: (2,),
    P.QKV_NORM_ROPE: (),                # in place on q, k, v
    P.QKV_NORM: (),                     # in place on q, k, v
    P.ROPE: (),                         # in place on q, k
    P.KV_WRITE_HEADS: (),               # reads k, v; writes the cache
    P.ATTN_F32H: (4,),
    P.KV_WRITE: (),                     # reads k, v; writes the cache
    P.ATTN_QC: (6,),
    P.ATTN_QC_MT: (6,),
    # the other caches of Qwen: the records of the int16 cache
    P.KV_WRITE8: (),
    P.KV_WRITEV8: (),
    P.KV_WRITETQ: (),
    P.TQ_ROT: (),                       # in place (with a source: WRITES_IF)
    P.ATTN_Q8: (6,),
    P.ATTN_V8: (6,),
    P.ATTN_TQ: (6,),
    P.ATTN_Q8_MT: (6,),
    P.ATTN_V8_MT: (6,),
    P.ATTN_TQ_MT: (6,),
    # Qwen: the query, its gate, and the key (the value stays in its
    # product); the gate of the attention; the output (11; zeros in the
    # padding rows) and the convolution rows (12) of the linear layers;
    # x += o then its norm
    P.ATTN_PREP: (18, 19, 20),
    P.SIGMUL: (2,),
    P.GDN: (11, 12),
    P.ADD_RMS: (3,),
    P.ROUTER_TOPK: (4, 5),
    # Qwen 3.8: the streams (hyperconn.cu: norm, act, mix, add in place on
    # H, the concatenation of the MTP layer), the n-gram layer (its gate; the
    # convolution adds to H), the indexer (the selection and its counts) and
    # its attention
    P.HC_NORM: (2,),
    P.HC_ACT: (1,),
    P.HC_MIX: (2,),
    P.HC_ADD: (),
    P.HC_CAT: (2,),
    P.PLE_GATE: (3,),
    P.PLE_CONV: (),
    P.QSA_SELECT: (17, 18),
    P.ATTN_QSA: (6,),
    P.ROUTER: (9, 10, 11, 12),
    P.ROUTER_MT: (9, 10, 12, 13),
    P.ADD_NORM: (3, 9),
    P.FFN_OUT: (12,),                   # (6: x in place)
    P.KQ_LINEAR: (9,),
    P.KQ_MULTI: (7, 11, 15, 19, 23),
    P.HOT_SPLIT: (3, 4),
    P.HOT_SPLIT_MT: (3, 4, 8),
    P.HOT_MOE: (6, 7, 8, 9),
    # the experts of Qwen on the GPU: the gate and up rows, the act, the
    # down rows (read only where written), the sum
    P.KQ_HOT_MOE: (7, 8, 9, 10),
    P.KQ_GROUP_MOE: (19, 20, 21, 22),
    P.TO_HOST: (),                      # reads its device sources
    P.TO_DEV: (1,),
    P.SIGNAL: (),
    P.AWAIT: (3,),
    P.CPU_JOIN: (),
    P.CPU_TASK: (),
    P.CPU_START: (),
    P.CPU_WAIT: (),
    P.FETCH: (),
    P.FETCH_WAIT: (),
    P.FETCH_DONE: (),
    P.COUNT: (),                        # reads idx; the counts are the host's
    # the experts of a large group: the counts, offsets and fills of
    # k_moe_sort (it clears them), its tiles, then the products and the sum
    P.MOE_GPU: (11, 12, 13, 14, 15, 16, 17, 18, 19, 20),
}
for _op in (P.S_MOV, P.S_ADD, P.S_SUB, P.S_MUL, P.S_MAX, P.S_MIN):
    WRITES[_op] = ()
# GP_HC_PRE (gpu35._fuse_hc): out, inj, and the parts of its two kernels
WRITES[P.HC_PRE] = (4, 5, 6)
# GP_MOE_COMBINE: out
WRITES[P.MOE_COMBINE] = (6,)
# GP_DS_MXX_LINEAR: out, the int16 planes and the scales of x (scratch)
WRITES[P.DS_MXX_LINEAR] = (2, 6, 7, 8)
# GP_DS_FP8_LINEAR: out
WRITES[P.DS_FP8_LINEAR] = (3,)
# the records of a DeepSeek step (kernels/dsv41/step.cu): the ones that write
# an operand wholly; the others write in place, or only some entries (the
# compressor on the step that ends a group, the scores of the step's
# entries, the first cnt of sel)
WRITES[P.DS_HC_MIX] = (2,)
WRITES[P.DS_HC_PRE] = (5,)
WRITES[P.DS_WIN_WRITE] = (3,)
WRITES[P.DS_ATTN] = (8,)
WRITES[P.DS_ROUTER] = (2, 3)
WRITES[P.DS_SWIGLU] = (2,)
WRITES[P.DS_HOT_MOE] = (12, 13, 14, 16, 17, 18, 19)     # (act and de: only the hot pairs)
WRITES[P.DS_COMBINE] = (4,)
WRITES[P.DS_STREAM_MEAN] = ()                 # (columns off .. off + dim of out)
WRITES[P.DS_CONF] = (3,)
WRITES[P.DS_EMBED] = (2,)
WRITES[P.DS_ENGRAM_ROWS] = (2,)
WRITES[P.DS_ROWS_AT] = (1,)

# opcode -> (operand, flag operand): the operand is written wholly when the
# flag operand is set (not 0). TQ_ROT with a source (operand 3; gpu35.
# _fuse_small of a COPY and the TQ_ROT of the copy) writes x: in place it
# read it, and its buffer stayed out of the arena (Qwen3.8, a mixed group
# of 4096 rows: 34 buffers of 42 MB).
WRITES_IF = {P.TQ_ROT: (0, 3)}


class Arenas:
    """The arenas of the programs of a model (the arena of pack): the step
    and the small groups (at most small tokens: the MTP verify groups) share
    one, the large groups (the chunks of a prompt) another, each the
    largest so far. The programs of a model run one at a time on one stream
    and keep nothing in their temporaries from one run to the next (what a
    caller reads is in EXPORTS), so they can share the bytes. Two arenas,
    not one: a program that GpuMem frees (gpumm.ProgramLRU) gives back the
    device copy of its arena only when no live program keeps it, so with
    one arena the step and the verify groups kept that of the largest
    prompt group (157 MB of a mixed group of Qwen3.6, 416 MB of Qwen3.8),
    and when the memory ran short a freed program gave back nothing: the
    LRU then freed every program, and the next steps made them again."""

    def __init__(self, small):
        self.small = small
        self.by = {}

    def arena_for(self, tokens):
        """The arena for a program of this many tokens (or None)."""
        return self.by.get(tokens <= self.small)

    def keep(self, tokens, prog):
        """After pack: keep the arena of prog if it is the largest of its
        kind. Return prog."""
        k = tokens <= self.small
        a = getattr(prog, "arena", None)
        if a is not None and (self.by.get(k) is None or a.nbytes > self.by[k].nbytes):
            self.by[k] = a
        return prog

    def pack(self, prog, tokens):
        """pack prog into the arena of its kind."""
        pack(prog, arena=self.arena_for(tokens))
        return self.keep(tokens, prog)


class PackInfo:
    """What pack() did: the temporaries packed and kept, their bytes, the
    arena."""

    def __init__(self):
        self.packed = 0
        self.kept = 0
        self.before = 0
        self.arena = 0
        self.shared = False

    def __repr__(self):
        return "PackInfo(%d packed in %.1f MB of %.1f MB, %d kept%s)" % (
            self.packed, self.arena / 1e6, self.before / 1e6, self.kept,
            ", in the shared arena" if self.shared else "")


def pack(prog, exports=EXPORTS, align=256, arena=None):
    """Pack the temporaries of a finished program (prog.code) into one
    arena (prog.arena, kept by prog.keep). Return a PackInfo.

    arena (or None): the arena of the other programs of the model, which
    this one takes when it is large enough. The programs of a model run
    one at a time, on one stream, and keep nothing in their temporaries
    from one run to the next (what a caller reads is in EXPORTS), so they
    can share the bytes: the device holds the largest program's arena once
    in place of an arena for each size of group. Else the program gets a
    new arena of its size (the caller keeps the larger one for the next
    programs; the programs on the old one keep it until they go)."""
    info = PackInfo()
    temps = {a.ctypes.data: a for a in getattr(prog, "temps", ()) if a.nbytes}
    if not PACK or not temps:
        return info
    starts = np.array(sorted(temps), dtype=np.int64)
    ends = starts + np.array([temps[a].nbytes for a in starts], dtype=np.int64)
    exported = set()
    for name in exports:
        a = prog.names.get(name)
        if isinstance(a, np.ndarray):
            while isinstance(a.base, np.ndarray):
                a = a.base
            exported.add(a.ctypes.data)
    first, last, pinned, uses = {}, {}, set(exported), []
    code = prog.code
    # the operands that point into a temporary, in the order of the records
    ii, kk = np.nonzero(code["tag"] == P.T_INT)
    vv = code["v"][ii, kk]
    jj = np.searchsorted(starts, vv, side="right") - 1
    hit = (jj >= 0) & (vv < ends[np.maximum(jj, 0)])
    ops = code["op"]
    for i, k, a in zip(ii[hit].tolist(), kk[hit].tolist(), starts[jj[hit]].tolist()):
        w = WRITES.get(int(ops[i]))
        uses.append((i, k, a))
        if w is None:
            pinned.add(a)               # an opcode with no row: its arrays stay
        first.setdefault(a, i)
        if first[a] == i and (w is None or k not in w):
            c = WRITES_IF.get(int(ops[i]))
            if not (c is not None and k == c[0] and code["tag"][i, c[1]] == P.T_INT and
                    code["v"][i, c[1]] != 0):
                pinned.add(a)           # its first record reads it
        last[a] = i
    cand = [a for a in first if a not in pinned]
    info.kept = len(first) - len(cand)
    info.before = sum(temps[a].nbytes for a in cand)
    if not cand:
        return info
    # the lowest offset that no temporary live at the same time holds, the
    # largest first
    cand.sort(key=lambda a: (-temps[a].nbytes, first[a]))
    # the placed ones: offset, end, first record, last record + 1
    po, pe, pf, pl = (np.zeros(len(cand), np.int64) for _ in range(4))
    off_of = {}
    for k, a in enumerate(cand):
        n = -(-temps[a].nbytes // align) * align
        f, l_ = first[a], last[a] + 1
        live = (pf[:k] <= l_) & (pl[:k] >= f)
        bo, be = po[:k][live], pe[:k][live]
        order = np.lexsort((be, bo))
        o = 0
        for b0, b1 in zip(bo[order].tolist(), be[order].tolist()):
            if o + n <= b0:
                break
            o = max(o, b1)
        po[k], pe[k], pf[k], pl[k] = o, o + n, f, l_
        off_of[a] = o
    size = int(pe.max())
    if arena is None or arena.nbytes < size:
        arena = np.full(size, 0xFF, np.uint8) if POISON else np.zeros(size, np.uint8)
    base = arena.ctypes.data
    # both forms of the records: the code (prog.buf) and prog.recs, from
    # which GPUProgram translates the addresses to the device
    recs = prog.recs
    assert len(recs) == len(code), "prog.recs and prog.code differ"
    for i, k, a in uses:
        if a in off_of:
            v = base + off_of[a] + (int(code[i]["v"][k]) - a)
            code[i]["v"][k] = v
            tag, old = recs[i][1][k]
            assert tag == P.T_INT and old == int(v - base - off_of[a] + a)
            recs[i][1][k] = (tag, v)
    # the packed temporaries go (their host arrays: a large group of
    # fresh buffers for each layer holds GBs of address space)
    gone = set()
    for a in off_of:
        t = temps[a]
        gone.add(id(t))
        if isinstance(t.base, np.ndarray):
            gone.add(id(t.base))        # (a view of the array of a map: lazy_zeros)
    prog.keep = [a for a in prog.keep
                 if id(a) not in gone and not (isinstance(a, np.ndarray) and id(a.base) in gone)]
    prog.temps = [a for a in prog.temps if a.ctypes.data not in off_of]
    prog.keep.append(arena)
    prog.arena = arena
    info.packed = len(cand)
    info.arena = size
    info.shared = arena.nbytes > size
    prog.pack_info = info
    return info

