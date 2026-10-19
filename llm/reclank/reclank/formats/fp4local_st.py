"""Qwen3.8-Flash-Next from nvidia/Qwen3.8-Flash-Next-FP4-Local (docs/FP4LOCAL_PLAN.md):
NVIDIA's QAD checkpoint (ModelOpt 0.47.0, quant_algo MIXED_PRECISION).

FP4LSource is NVFP4Source (reclank/formats/nvfp4_st.py: the GGUF view that
reclank/models/qwen/qwen38.py reads) on the tensors of this checkpoint:

- the routed experts of the 48 layers: NVFP4 (E2M1 codes, an E4M3 scale for
  each 16, the float32 weight_scale_2 of each matrix) whose block scales are
  stored in the CSF encoding "csf" (X.weight_scale.nvfp4_csf_fixed and
  .nvfp4_csf_exceptions; cops.csf_decode, csrc/kernels/kq/convert.c): decoded
  to the E4M3 bytes they define, then packed as KQ_NVX (groups of 16 rows:
  the codes, the E4M3 scales, the float32 scale of the matrix; lossless). The
  input scales of each matrix (the A4 contract of the prompt rows):
  input_scales();
- the routed experts of the MTP layer: NVFP4 with plain E4M3 scales, KQ_NVX;
- the MXFP8 matrices (an E4M3 weight and an E8M0 scale for each 1 x 32:
  w = e4m3 x 2^(s - 127); the attention, the DeltaNet projections, the
  indexer, the shared experts): the large ones as they are, KQ_MXF8 rows
  (the codes, then the scales; the GPU's kernels take x in bfloat16 and round
  their output to bfloat16, b12x's a16 mode); RECLANK_FP4L_MXF8=0: their
  exact bfloat16 values (KQ_BF16; 4 significant bits, exponents -72 to 1);
  the small ones (the DeltaNet's a and b, the indexer) float32, mxfp8();
- the norms: bfloat16 (the 97 hyper-connection norms and the 3 n-gram norms
  are float32 in the file; vLLM loads them into bfloat16 parameters, so they
  are rounded once, as there), plus 1 where nvfp4_st adds it;
- the n-gram table: NVFP4 rows of 160 (128 shards of codes [rows, 80] and
  E4M3 scales [rows, 10], one weight_scale_2): ple_rows decodes the rows it
  reads (float32, exact; the model rounds them to bfloat16, cfg.ple_bf16);
- bfloat16 tensors as they are.

attach_mtp puts the MTP layer of another file over this one's (RECLANK_FP4L_MTP: the
BF12 GGUF's blk.48 as it is, a drafter whose drafts the model's verify checks; the
plan's D8).

    src = FP4LSource("models/Qwen3.8-Flash-Next-FP4-Local")
"""
from __future__ import annotations

import json
import os

import numpy as np

from reclank.backends.cpu import cops
from reclank.formats.nvfp4_st import E4M3, KQ_BF16, KQ_F32, KQ_NVX, MAIN, NVFP4Source

KQ_MXF8 = 66


def mxf8_row_bytes(cols):
    """A KQ_MXF8 row: the E4M3 codes, the E8M0 scale of each 32, zeros to 16 bytes."""
    return (cols + cols // 32 + 15) // 16 * 16


def mxf8_pack(codes, scales):
    """KQ_MXF8 rows (uint8 [rows, mxf8_row_bytes(cols)]) of an MXFP8 matrix (codes [rows, cols],
    scales [rows, cols / 32])."""
    codes, scales = np.asarray(codes, np.uint8), np.asarray(scales, np.uint8)
    rows, cols = codes.shape
    out = np.zeros((rows, mxf8_row_bytes(cols)), np.uint8)
    out[:, :cols] = codes
    out[:, cols:cols + cols // 32] = scales
    return out

FIX, EXC = ".weight_scale.nvfp4_csf_fixed", ".weight_scale.nvfp4_csf_exceptions"
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0., -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)


def is_fp4local(path):
    """True for a checkpoint directory with the CSF experts of FP4-Local."""
    try:
        q = json.load(open(os.path.join(path, "config.json"))).get("quantization_config", {})
    except (OSError, ValueError):
        return False
    layers = q.get("quantized_layers", {})
    return q.get("quant_algo") == "MIXED_PRECISION" and any(
        v.get("weight_scale_encoding") == "csf" for v in layers.values())


def bf16r(a):
    """float32 values rounded to bfloat16 (to nearest even), as float32."""
    u = np.ascontiguousarray(a, np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).astype(np.uint32).view(np.float32)


def mxfp8_dequant(codes, scales):
    """The float32 values of an MXFP8 matrix (codes: E4M3 bytes [n, k], scales: E8M0
    bytes [n, k / 32])."""
    codes = np.asarray(codes)
    n, k = codes.shape
    s = np.ldexp(np.float32(1), np.asarray(scales, np.int32) - 127).astype(np.float32)
    return (E4M3[codes].reshape(n, k // 32, 32) * s[:, :, None]).reshape(n, k)


def e2m1_codes(v):
    """The E2M1 codes (0-15: sign bit 8) of v, to nearest even, saturating at 6 (cvt.rn.satfinite.e2m1x2)."""
    a = np.minimum(np.abs(v), 6.0)
    grid = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    mid = (grid[1:] + grid[:-1]) / 2
    idx = np.searchsorted(mid, a, side="left")              # ties: the lower code
    tie = (idx < 7) & (a == mid[np.minimum(idx, 6)])
    idx = np.where(tie & (idx % 2 == 1), idx + 1, idx)       # a tie to the even code
    return (idx | np.where((v < 0) & (idx > 0), 8, 0)).astype(np.uint8)


def e4m3_codes(v):
    """The E4M3 codes of v >= 0 (to nearest even, subnormals, saturating at 448)."""
    pos = E4M3[:127]
    a = np.minimum(np.asarray(v, np.float32), np.float32(448.0))
    i = np.clip(np.searchsorted(pos, a), 1, 126)
    lo, hi = pos[i - 1], pos[i]
    up = (a - lo > hi - a) | ((a - lo == hi - a) & (i % 2 == 0))
    return np.where(up, i, i - 1).astype(np.uint8)


def mtp_head_nvfp4(w):
    """The MTP drafter's head of vLLM (VLLM_MTP_NVFP4_LM_HEAD, online/nvfp4.py, nvfp4_utils.cuh): the
    BF16 lm_head (float32 values, rows x cols) quantized once: gs = 2688 / amax; for each 16, SF =
    gs (vmax / 6), the E4M3 scale e4m3(SF), the codes e2m1(w / (sf / gs)); alpha = 1 / gs. The
    approximate reciprocals of the kernel are IEEE here (they only move the drafts). Returns
    (codes [rows, cols / 2] uint8, low nibble the even value; scales [rows, cols / 16] E4M3; alpha)."""
    w = np.asarray(w, np.float32)
    rows, cols = w.shape
    gs = np.float32(2688.0) / np.float32(max(float(np.abs(w).max()), 1e-8))
    codes = np.empty((rows, cols // 2), np.uint8)
    scales = np.empty((rows, cols // 16), np.uint8)
    for r0 in range(0, rows, 8192):
        b = w[r0:r0 + 8192].reshape(-1, cols // 16, 16)
        vmax = np.abs(b).max(axis=2)
        sf8 = e4m3_codes(gs * (vmax * (np.float32(1) / np.float32(6))))
        sf = E4M3[sf8]
        with np.errstate(divide="ignore"):
            os_ = np.where(sf == 0, np.float32(0), np.float32(1) / (sf * (np.float32(1) / gs))).astype(np.float32)
        q = e2m1_codes(b * os_[:, :, None]).reshape(b.shape[0], cols)
        codes[r0:r0 + 8192] = q[:, 0::2] | (q[:, 1::2] << 4)
        scales[r0:r0 + 8192] = sf8
    return codes, scales, float(np.float32(1) / gs)


def nvfp4_rows(codes, scales, g):
    """The float32 values of NVFP4 rows (codes [n, k / 2] uint8, low nibble the even
    value; scales [n, k / 16] E4M3 bytes; g the float32 global scale)."""
    codes = np.asarray(codes)
    n, h = codes.shape
    q = np.empty((n, 2 * h), np.uint8)
    q[:, 0::2], q[:, 1::2] = codes & 15, codes >> 4
    v = E2M1[q].reshape(n, -1, 16) * E4M3[np.asarray(scales)][:, :, None]
    return (v * np.float32(g)).reshape(n, 2 * h).astype(np.float32)


def attach_mtp(g, path):
    """The tensors of the MTP layer (blk.L.*) of the GGUF file path in the place of those of g
    (FP4LSource or keep_fp4l.KeptFP4L), each with its own type (the BF12 file's: BF12 matrices and
    experts, float32 norms): g.tensors gives them, g.raw and g.dequant read them from path, and a
    weightkeeper's index no longer lists them (Qwen4CPU.K reads them as a matrix of a file, not
    as the weightkeeper's experts). Every tensor of path must be one of g's layer L, of the same
    shape."""
    from reclank.formats.gguf import GGUF
    p = GGUF(path)
    pre = "blk.%d." % int(g.config().num_hidden_layers)
    for n in p._order:
        if not n.startswith(pre) or n not in g.tensors:
            raise ValueError("%s: %s is not a tensor of FP4-Local's MTP layer" % (path, n))
        if tuple(int(x) for x in p.tensors[n][0]) != tuple(int(x) for x in g.tensors[n][0]):
            raise ValueError("%s: %s has the shape %s, not %s" % (path, n, p.tensors[n][0], g.tensors[n][0]))
    names = set(p._order)
    for n in p._order:
        g.tensors[n] = p.tensors[n]
    if hasattr(g, "index"):
        g.index = dict(g.index, tensors={n: d for n, d in g.index["tensors"].items() if n not in names})
    g._mtp, g._mtp_names = p, names
    return sorted(names)


class FP4LSource(NVFP4Source):
    """The GGUF view of FP4-Local (see the module text)."""

    as_stored = True        # the model keeps every type (qwen38.dense_mode, Qwen4CPU.K): no requantization

    _mtp_names = frozenset()        # (attach_mtp)

    def __init__(self, path, headers=None):
        super().__init__(path, headers, experts="nvx")
        self.csf_stats = None

    # ---- the files ----

    def _is_mxfp8(self, name):
        if not name.endswith(".weight"):
            return False
        s = name[:-7] + ".weight_scale"
        return s in self.where and self._file(s).dtype(s) == "U8"

    def mxfp8(self, name):
        """The float32 values of the MXFP8 matrix name (<key>.weight)."""
        f = self._file(name)
        return mxfp8_dequant(f.get(name, dtype=None), self._file(name[:-7] + ".weight_scale").get(
            name[:-7] + ".weight_scale", dtype=None))

    def _get(self, name, dtype=np.float32):
        if dtype is not None and self._is_mxfp8(name):
            return self.mxfp8(name)
        a = super()._get(name, dtype=dtype)
        if dtype is not None and (name.endswith("norm.weight") or ".norm_" in name or "_norm_" in name):
            a = bf16r(a)            # (a no-op on the bfloat16 norms)
        return a

    def _bf16(self, name):
        if self._is_mxfp8(name):
            w = self.mxfp8(name)
            b = (w.view(np.uint32) >> 16).astype(np.uint16)
            assert np.array_equal((b.astype(np.uint32) << 16).view(np.float32), w), name
            return b
        return super()._bf16(name)

    # ---- the map of the names ----

    def _dense_mat(self, gname, hf):
        if self._is_mxfp8(hf) and os.environ.get("RECLANK_FP4L_MXF8", "1") != "0":
            rows, cols = self._shape(hf)
            self._add(gname, "mxf8", hf, (cols, rows), KQ_MXF8)
            return
        super()._dense_mat(gname, hf)

    def _experts(self, b, m, mtp):
        if mtp:
            return super()._experts(b, m, False)        # plain E4M3 scales: the "nv4exps" of nvfp4_st
        for x in ("gate", "up", "down"):
            rows, half = self._shape(m + "experts.0.%s_proj.weight" % x)
            self._add(b + "ffn_%s_exps.weight" % x, "csfexps", (m, x), (half * 2, rows, self.E), KQ_NVX)

    def _build(self):
        super()._build()
        shards = sorted((n for n in self.where if ".ngram_embedding.shard_" in n and n.endswith(".weight")),
                        key=lambda n: int(n.rpartition(".shard_")[2].partition(".")[0]))
        if "lm_head.weight" in self.where and "mtp.fc_embedding.weight" in self.where:
            # the MTP drafter's own head (the plan's 2.4): lm_head in NVFP4 (KQ_NV4 rows)
            rows, cols = self._shape("lm_head.weight")
            self._add("mtp_head.weight", "mtphead", None, (cols, rows), 51)
        if shards:
            self._ple_shards = shards
            sizes = [self._shape(n)[0] for n in shards]
            self._ple_start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
            half = self._shape(shards[0])[1]
            self._add("per_layer_token_embd.weight", "ple", None, (2 * half, int(self._ple_start[-1])), KQ_F32)

    def config(self):
        cfg = super().config()
        cfg.ple_bf16 = True             # the n-gram rows rounded to bfloat16 (ple_embedding/reference.py)
        # the contract of the experts and their inputs (docs/FP4LOCAL_PLAN.md 2): the MoE and block
        # inputs rounded to bfloat16 (BF16_ROUND), b12x's W4A16 on the steps and verify groups
        # (RECLANK_FP4L_CONTRACT=0: the earlier kernels, a comparison)
        cfg.b12x = os.environ.get("RECLANK_FP4L_CONTRACT", "1") != "0"
        # the QSA cache in E4M3 (the plan's 2.9: vLLM's --kv-cache-dtype fp8, scales 1)
        cfg.kv_form = os.environ.get("RECLANK_QWEN_KV", "e4m3")
        return cfg

    # ---- the tensors ----

    def csf_scales(self, m, x, e):
        """The E4M3 block scales (uint8 [rows, cols / 16]) of expert e of matrix x of the
        layer prefix m, and the checks of the decode (cops.csf_decode)."""
        p = m + "experts.%d.%s_proj" % (e, x)
        rows, half = self._shape(p + ".weight")
        return cops.csf_decode(self._get(p + FIX, dtype=None), self._get(p + EXC, dtype=None), rows, half * 2 // 16)

    def input_scales(self, layer):
        """{gate, up, down: float32 [E]}: the input_scale of each expert's matrices (the
        global scales of the A4 activations of prompt rows)."""
        m = MAIN + "layers.%d.mlp." % layer
        return {x: np.array([float(np.asarray(self._get(m + "experts.%d.%s_proj.input_scale" % (e, x))).reshape(-1)[0])
                             for e in range(self.E)], np.float32) for x in ("gate", "up", "down")}

    def pack_experts(self, gname, out=None):
        """The KQ_NVX stack of a matrix of routed experts (E x rows groups of 16 rows) in out
        (uint8, made on huge pages when None): the CSF experts of the 48 layers or the plain
        ones of the MTP layer."""
        kind, (m, x) = self._map[gname]
        rows, half = self._shape(m + "experts.0.%s_proj.weight" % x)
        cols = 2 * half
        ws, ss, gs = [], [], []
        agg = dict(bad_base=0, bad_exc=0, inside=0, max_base=0, zeros=0, subnormal=0)
        for e in range(self.E):
            p = m + "experts.%d.%s_proj." % (e, x)
            ws.append(self._get(p + "weight", dtype=None))
            if kind == "csfexps":
                s, st = self.csf_scales(m, x, e)
                for k, v in st.items():
                    agg[k] = max(agg[k], v) if k == "max_base" else agg[k] + v
            else:
                s = self._get(p + "weight_scale", dtype=None)
            ss.append(s)
            gs.append(float(np.asarray(self._get(p + "weight_scale_2")).reshape(-1)[0]))
        if agg["bad_base"] or agg["bad_exc"]:
            raise ValueError("%s: CSF scales out of the format: %s" % (gname, agg))
        if out is None:
            from reclank.runtime import numa
            out = numa.empty_huge(self.E * rows * cops.kq_nvx_row_bytes(cols))
        cops.kq_nvx_pack(ws, ss, gs, rows, cols, out)
        if kind == "csfexps":
            # the A4 global scale of the inputs (the plan's 2.2) in bytes 4-7 of each group's head
            # (zeros before): gate and up the layer's gs1 = min over experts of 1 / max(input_scale
            # gate, up), down each expert's gs2 = 1 / input_scale_down
            isc = self.input_scales(int(m.split(".layers.")[1].split(".")[0]))
            if x == "down":
                a4 = (np.float32(1) / isc["down"]).astype(np.float32)
            else:
                a4 = np.full(self.E, np.min(np.float32(1) / np.maximum(isc["gate"], isc["up"])), np.float32)
            grp = out[:self.E * rows * cops.kq_nvx_row_bytes(cols)].reshape(self.E, rows // 16, 16 * cops.kq_nvx_row_bytes(cols))
            grp[:, :, 4:8] = np.repeat(a4.view(np.uint8).reshape(self.E, 1, 4), rows // 16, axis=1)
        self.csf_stats = agg
        return out

    def pack_ple(self, k, out=None):
        """The rows of n-gram shard k as KQ_NV4 rows of 160 (the codes, the 10 E4M3 scales,
        the float32 global scale; cops.kq_nv4_row_bytes): (out, the first row of the shard)."""
        n = self._ple_shards[k]
        codes, scales = self._get(n, dtype=None), self._get(n + "_scale", dtype=None)
        rows, half = codes.shape
        if out is None:
            out = np.empty(rows * cops.kq_nv4_row_bytes(2 * half), np.uint8)
        cops.kq_nv4_pack([np.asarray(codes)], [np.asarray(scales)], [self._ple_scale()], rows, 2 * half, out)
        return out, int(self._ple_start[k])

    def _make(self, gname):
        kind, hf = self._map[gname]
        if kind == "csfexps":
            return self.pack_experts(gname)
        if kind == "mxf8":
            return mxf8_pack(self._get(hf, dtype=None), self._get(hf[:-7] + ".weight_scale", dtype=None)).reshape(-1)
        if kind == "mtphead":
            codes, scales, alpha = mtp_head_nvfp4(self._get("lm_head.weight"))
            rows, half = codes.shape
            out = np.empty(rows * cops.kq_nv4_row_bytes(2 * half), np.uint8)
            cops.kq_nv4_pack([codes], [scales], [alpha], rows, 2 * half, out)
            return out
        return super()._make(gname)

    def raw(self, gname):
        if gname in self._mtp_names:
            return self._mtp.raw(gname)
        return super().raw(gname)

    def dequant(self, gname, rows=None):
        if gname in self._mtp_names:
            return self._mtp.dequant(gname, rows)
        kind, hf = self._map[gname]
        if kind == "mxf8":
            w = self.mxfp8(hf)
            return w if rows is None else w[np.asarray(rows)]
        return super().dequant(gname, rows)

    def ple_rows(self, rows):
        """The float32 rows of the n-gram table (NVFP4, decoded exactly)."""
        rows = np.asarray(rows, dtype=np.int64).reshape(-1)
        base = self._cache.get("_ple_base")
        if base is None:
            self._ple_hold = [(self._get(n, dtype=None), self._get(n + "_scale", dtype=None)) for n in self._ple_shards]
            base = self._cache["_ple_base"] = np.array([[c.ctypes.data, s.ctypes.data] for c, s in self._ple_hold],
                                                       np.int64)
            self._ple_random()
        half = self._ple_hold[0][0].shape[1]
        k = np.searchsorted(self._ple_start, rows, side="right") - 1
        r = rows - self._ple_start[k]
        codes = cops.kq_gather(base[k, 0] + r * half, half).reshape(len(rows), half)
        scales = cops.kq_gather(base[k, 1] + r * (half // 8), half // 8).reshape(len(rows), half // 8)
        return nvfp4_rows(codes, scales, self._ple_scale())

    def _ple_scale(self):
        v = self._cache.get("_ple_scale")
        if v is None:
            name = self._ple_shards[0].split(".shard_")[0] + ".weight_scale_2"
            v = self._cache["_ple_scale"] = float(np.asarray(self._get(name)).reshape(-1)[0])
        return v

    def _ple_random(self):
        import mmap
        names = [n for n in self._ple_shards] + [n + "_scale" for n in self._ple_shards]
        for f in {self.where[x] for x in names}:
            st = self._files[f]
            spans = [st.header[x]["data_offsets"] for x in names if self.where[x] == f]
            lo = st._base + min(a for a, _b in spans)
            hi = st._base + max(b for _a, b in spans)
            lo -= lo % mmap.PAGESIZE
            for adv in ("MADV_NOHUGEPAGE", "MADV_RANDOM"):
                try:
                    st._mm.madvise(getattr(mmap, adv), lo, hi - lo)
                except (AttributeError, OSError, ValueError):
                    pass


__all__ = ["FP4LSource", "is_fp4local", "bf16r", "mxfp8_dequant", "nvfp4_rows", "E2M1"]
