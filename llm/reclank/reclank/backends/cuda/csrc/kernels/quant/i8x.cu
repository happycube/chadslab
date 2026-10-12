/* ---------- the products of a step and of a small group with int8 x ----------
 * x as int8 with a scale for each 32 values (k_kq_quant_x, one time for a
 * record), and __dp4a on the int8 values of Q8_R: a lane reads 16 bytes of w
 * and 16 of x (not 64 of float32 x) and makes 4 dp4a and one float fma for
 * 16 values. The other types read x as before. A token alone and in a group
 * run the same code: the same bits. RECLANK_GPU_I8X=0: float32 x for all. */
/* The types of kq_row_i8 (the others read x as float32). */
__host__ __device__ __forceinline__ int kq_i8_type(int type)
{
    return type == KQ_Q8_R || type == KQ_Q8_0 || type == KQ_Q4_K || type == KQ_Q5_K ||
           type == KQ_Q6_K || type == KQ_Q4_0;
}

/* kq_row_i8 for NT tokens (t <= NT of them live): the x of token j is at
 * x + j cols, xq + j cols, and xs + j cols / 32. The kernel unpacks each
 * weight word once for all the tokens. Each token adds its terms in the
 * order of one token, so res[j] has the bits of kq_row_i8 of that token (a
 * verify group of MTP gets the values of the decode steps). */
template <int NT, bool HALF = false>
__device__ void kq_rows_i8(int type, const uint8_t *w, const float *x, const int8_t *xq,
                           const float *xs, int cols, int t, float *res)
{
    if (!kq_i8_type(type) || xq == NULL) {
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                res[j] = kq_row(type, w, x + (size_t)j * cols, cols);
            }
        }
        return;
    }
    /* HALF: a row for each 16 lanes (the lanes 0 to 15 of the row, and the
     * strides of half a warp); k_kq_glu_i8. */
    int lane = threadIdx.x % (HALF ? 16 : 32), np = cols / 32;
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    if (type == KQ_Q8_0) {
        /* lane l: 16 values (half l % 2) of block l / 2 of the 16 blocks of a
         * step. The blocks are 34 bytes, so the loads take 2 bytes. */
        int hf = lane & 1;
        for (int b = lane >> 1; b < cols / 32; b += HALF ? 8 : 16) {
            const uint8_t *blk = w + (size_t)b * 34;
            const uint16_t *q = (const uint16_t *)(blk + 2 + 16 * hf);
            uint32_t qw[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                qw[k] = (uint32_t)q[2 * k] | ((uint32_t)q[2 * k + 1] << 16);
            }
            float d = kq_half(blk);
            #pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
                if (jt < t) {
                    uint4 xv = *(const uint4 *)(xq + (size_t)jt * cols + b * 32 + 16 * hf);
                    int si = __dp4a((int)qw[0], (int)xv.x, 0);
                    si = __dp4a((int)qw[1], (int)xv.y, si);
                    si = __dp4a((int)qw[2], (int)xv.z, si);
                    si = __dp4a((int)qw[3], (int)xv.w, si);
                    sum[jt] += (float)si * (d * xs[jt * np + b]);
                }
            }
        }
    } else if (type == KQ_Q4_0) {
        /* As Q8_0: lane l takes the 16 values hf = l % 2 of block l / 2 (the
         * low 4 bits of the 16 bytes for hf 0, the high 4 bits for hf 1).
         * dp4a of the codes q (0 to 15) and x, minus 8 times the sum of x:
         * the sum of (q - 8) x. */
        int hf = lane & 1;
        for (int b = lane >> 1; b < cols / 32; b += HALF ? 8 : 16) {
            const uint8_t *blk = w + (size_t)b * 18;
            const uint16_t *q = (const uint16_t *)(blk + 2);
            uint32_t qw[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                qw[k] = (((uint32_t)q[2 * k] | ((uint32_t)q[2 * k + 1] << 16)) >> (4 * hf)) &
                        0x0F0F0F0Fu;
            }
            float d = kq_half(blk);
            #pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
                if (jt < t) {
                    uint4 xv = *(const uint4 *)(xq + (size_t)jt * cols + b * 32 + 16 * hf);
                    int si = __dp4a((int)qw[0], (int)xv.x, 0);
                    si = __dp4a((int)qw[1], (int)xv.y, si);
                    si = __dp4a((int)qw[2], (int)xv.z, si);
                    si = __dp4a((int)qw[3], (int)xv.w, si);
                    int sx = __dp4a(0x01010101, (int)xv.x, 0);
                    sx = __dp4a(0x01010101, (int)xv.y, sx);
                    sx = __dp4a(0x01010101, (int)xv.z, sx);
                    sx = __dp4a(0x01010101, (int)xv.w, sx);
                    sum[jt] += (float)(si - 8 * sx) * (d * xs[jt * np + b]);
                }
            }
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        /* lane l = 8 sb + 2 c + hf: superblock 4 i + sb of step i, the 16
         * bytes hf of the 32 bytes c of the quants (16 values of part 2 c in
         * the low 4 bits, 16 of part 2 c + 1 in the high 4 bits). A warp
         * reads 4 superblocks for each step, 16 bytes for each lane. The
         * mins take the sum of the int8 x (dp4a with ones). */
        int five = type == KQ_Q5_K, hf = lane & 1, c = (lane >> 1) & 3, sbk = lane >> 3;
        int nbk = cols / 256;
        size_t bs = five ? 176 : 144;
        for (int b = sbk; b < nbk; b += HALF ? 2 : 4) {
            const uint8_t *blk = w + (size_t)b * bs;
            uint4 q = *(const uint4 *)(blk + (five ? 48 : 16) + 32 * c + 16 * hf);
            uint32_t qv[4] = {q.x, q.y, q.z, q.w}, lo[4], hi[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[k] = qv[k] & 0x0f0f0f0fu;
                hi[k] = (qv[k] >> 4) & 0x0f0f0f0fu;
            }
            if (five) {
                uint4 h = *(const uint4 *)(blk + 16 + 16 * hf);
                uint32_t hv[4] = {h.x, h.y, h.z, h.w};
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    lo[k] |= ((hv[k] >> (2 * c)) & 0x01010101u) << 4;
                    hi[k] |= ((hv[k] >> (2 * c + 1)) & 0x01010101u) << 4;
                }
            }
            int s0, m0, s1, m1;
            kq_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            float d = kq_half(blk), dm = kq_half(blk + 2);
            int base = b * 256 + 64 * c + 16 * hf;
            #pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
                if (jt < t) {
                    const int8_t *xj = xq + (size_t)jt * cols;
                    uint4 xl = *(const uint4 *)(xj + base), xh = *(const uint4 *)(xj + base + 32);
                    float sl = xs[jt * np + base / 32], sh = xs[jt * np + base / 32 + 1];
                    int dl = __dp4a((int)lo[0], (int)xl.x, 0);
                    dl = __dp4a((int)lo[1], (int)xl.y, dl);
                    dl = __dp4a((int)lo[2], (int)xl.z, dl);
                    dl = __dp4a((int)lo[3], (int)xl.w, dl);
                    int dh = __dp4a((int)hi[0], (int)xh.x, 0);
                    dh = __dp4a((int)hi[1], (int)xh.y, dh);
                    dh = __dp4a((int)hi[2], (int)xh.z, dh);
                    dh = __dp4a((int)hi[3], (int)xh.w, dh);
                    int nl = __dp4a(0x01010101, (int)xl.x, 0);
                    nl = __dp4a(0x01010101, (int)xl.y, nl);
                    nl = __dp4a(0x01010101, (int)xl.z, nl);
                    nl = __dp4a(0x01010101, (int)xl.w, nl);
                    int nh = __dp4a(0x01010101, (int)xh.x, 0);
                    nh = __dp4a(0x01010101, (int)xh.y, nh);
                    nh = __dp4a(0x01010101, (int)xh.z, nh);
                    nh = __dp4a(0x01010101, (int)xh.w, nh);
                    sum[jt] += d * ((float)(s0 * dl) * sl + (float)(s1 * dh) * sh) -
                               dm * ((float)(m0 * nl) * sl + (float)(m1 * nh) * sh);
                }
            }
        }
    } else if (type == KQ_Q6_K) {
        /* lane l = 8 sb + 4 n + 2 u + hf: superblock 4 i + sb of step i,
         * half n, the 16 values hf of the quarters u (the low 4 bits of ql)
         * and u + 2 (the high 4 bits), with 2 bits of qh each. The code is
         * that value minus 32. The blocks are 210 bytes: 2-byte loads. */
        int hf = lane & 1, u = (lane >> 1) & 1, n = (lane >> 2) & 1, sbk = lane >> 3;
        for (int b = sbk; b < cols / 256; b += HALF ? 2 : 4) {
            const uint8_t *ql = w + (size_t)b * 210;
            const uint16_t *pl = (const uint16_t *)(ql + 64 * n + 32 * u + 16 * hf);
            const uint16_t *ph = (const uint16_t *)(ql + 128 + 32 * n + 16 * hf);
            const int8_t *sc = (const int8_t *)(ql + 192);
            float d = kq_half(ql + 208);
            uint32_t cl[4], ch[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t lq = (uint32_t)pl[2 * k] | ((uint32_t)pl[2 * k + 1] << 16);
                uint32_t hq = (uint32_t)ph[2 * k] | ((uint32_t)ph[2 * k + 1] << 16);
                cl[k] = __vsub4((lq & 0x0f0f0f0fu) | (((hq >> (2 * u)) & 0x03030303u) << 4),
                                0x20202020u);
                ch[k] = __vsub4(((lq >> 4) & 0x0f0f0f0fu) |
                                (((hq >> (2 * u + 4)) & 0x03030303u) << 4), 0x20202020u);
            }
            float fl = d * (float)sc[8 * n + hf + 2 * u], fh = d * (float)sc[8 * n + hf + 2 * u + 4];
            int base = b * 256 + 128 * n + 32 * u + 16 * hf;
            #pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
                if (jt < t) {
                    const int8_t *xj = xq + (size_t)jt * cols;
                    uint4 xl = *(const uint4 *)(xj + base), xh = *(const uint4 *)(xj + base + 64);
                    int dl = __dp4a((int)cl[0], (int)xl.x, 0);
                    dl = __dp4a((int)cl[1], (int)xl.y, dl);
                    dl = __dp4a((int)cl[2], (int)xl.z, dl);
                    dl = __dp4a((int)cl[3], (int)xl.w, dl);
                    int dh = __dp4a((int)ch[0], (int)xh.x, 0);
                    dh = __dp4a((int)ch[1], (int)xh.y, dh);
                    dh = __dp4a((int)ch[2], (int)xh.z, dh);
                    dh = __dp4a((int)ch[3], (int)xh.w, dh);
                    sum[jt] += fl * xs[jt * np + base / 32] * (float)dl +
                               fh * xs[jt * np + base / 32 + 2] * (float)dh;
                }
            }
        }
    } else {
        const __half *d = (const __half *)(w + cols);
        for (int c = 16 * lane; c < cols; c += HALF ? 256 : 512) {
            uint4 q = *(const uint4 *)(w + c);
            float dc = __half2float(d[c / 32]);
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (j < t) {
                    uint4 xv = *(const uint4 *)(xq + (size_t)j * cols + c);
                    int si = __dp4a((int)q.x, (int)xv.x, 0);
                    si = __dp4a((int)q.y, (int)xv.y, si);
                    si = __dp4a((int)q.z, (int)xv.z, si);
                    si = __dp4a((int)q.w, (int)xv.w, si);
                    sum[j] += (float)si * (dc * xs[j * np + c / 32]);
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        for (int o = HALF ? 8 : 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(RC_FULL_MASK, v, o);
        }
        res[j] = v;
    }
}

__device__ float kq_row_i8(int type, const uint8_t *w, const float *x, const int8_t *xq,
                           const float *xs, int cols)
{
    float v;
    kq_rows_i8<1>(type, w, x, xq, xs, cols, 1, &v);
    return v;
}

/* The products of row w on t tokens, NT tokens at a time (kq_rows_i8). A
 * kernel has one NT, so a step (NT 1) keeps the registers of one token. */
#define KQ_NT 4

template <int NT>
__device__ void kq_row_i8_t(int type, const uint8_t *w, const float *x, const int8_t *xq,
                            const float *xs, int cols, int t, float *out, size_t ostr)
{
    for (int j0 = 0; j0 < t; j0 += NT) {
        float res[NT];
        int n = t - j0 < NT ? t - j0 : NT;
        kq_rows_i8<NT>(type, w, x + (size_t)j0 * cols, xq + (size_t)j0 * cols,
                       xs + (size_t)j0 * (cols / 32), cols, n, res);
        if (threadIdx.x % 32 == 0) {
            for (int j = 0; j < n; ++j) {
                out[(size_t)(j0 + j) * ostr] = res[j];
            }
        }
    }
}

/* kq_row_i8_t for a group: each part of up to KQ_NT tokens takes the
 * kq_rows_i8 of its count (NT 0 in the kernels). */
__device__ void kq_row_i8_g(int type, const uint8_t *w, const float *x, const int8_t *xq,
                            const float *xs, int cols, int t, float *out, size_t ostr)
{
    for (int j0 = 0; j0 < t; j0 += KQ_NT) {
        float res[KQ_NT];
        int n = t - j0 < KQ_NT ? t - j0 : KQ_NT;
        const float *xj = x + (size_t)j0 * cols;
        const int8_t *qj = xq + (size_t)j0 * cols;
        const float *sj = xs + (size_t)j0 * (cols / 32);
        if (n == 1) {
            kq_rows_i8<1>(type, w, xj, qj, sj, cols, 1, res);
        } else if (n == 2) {
            kq_rows_i8<2>(type, w, xj, qj, sj, cols, 2, res);
        } else if (n == 3) {
            kq_rows_i8<3>(type, w, xj, qj, sj, cols, 3, res);
        } else {
            /* n == KQ_NT. A constant count lets the compiler drop the tests
             * of the live tokens (a runtime count made a group of 3 or 4
             * 1.5 to 2.5 times slower). */
            kq_rows_i8<KQ_NT>(type, w, xj, qj, sj, cols, KQ_NT, res);
        }
        if (threadIdx.x % 32 == 0) {
            for (int j = 0; j < n; ++j) {
                out[(size_t)(j0 + j) * ostr] = res[j];
            }
        }
    }
}



/* NT KQ_G8 in the kernels: a group of 5 to 8 tokens in one part, with the
 * kq_rows_i8 of its count (kq_rows_i8_n): the weights are read once for the
 * group. Parts of KQ_NT read them twice for 5 tokens, so a verify group of
 * 5 rows of the E4B or the 12B cost almost as much as one of 8 (MTP with 4
 * drafts slower than with 3 by a quarter). Each token adds its terms in the
 * order of one token in any count: the same bits. */
#define KQ_G8 8

/* kq_rows_i8 of the count n, LO <= n <= HI, as a constant (a runtime count
 * made a group 1.5 to 2.5 times slower: kq_row_i8_g); res has HI values. */
template <int LO, int HI, bool HALF = false>
__device__ __forceinline__ void kq_rows_i8_n(int n, int type, const uint8_t *w, const float *x,
                                             const int8_t *xq, const float *xs, int cols, float *res)
{
    if constexpr (LO == HI) {
        kq_rows_i8<LO, HALF>(type, w, x, xq, xs, cols, LO, res);
    } else {
        if (n == LO) {
            kq_rows_i8<LO, HALF>(type, w, x, xq, xs, cols, LO, res);
        } else {
            kq_rows_i8_n<LO + 1, HI, HALF>(n, type, w, x, xq, xs, cols, res);
        }
    }
}

/* kq_row_i8_g for a group of 5 to KQ_G8 tokens: one part. */
__device__ void kq_row_i8_g8(int type, const uint8_t *w, const float *x, const int8_t *xq,
                             const float *xs, int cols, int t, float *out, size_t ostr)
{
    float res[KQ_G8];
    kq_rows_i8_n<5, KQ_G8>(t, type, w, x, xq, xs, cols, res);
    if (threadIdx.x % 32 == 0) {
        for (int j = 0; j < t; ++j) {
            out[(size_t)j * ostr] = res[j];
        }
    }
}

/* GP_KQ_LINEAR (t <= MT_MAX) with the int8 x of the scratch. FT >= 0 fixes
 * the type when the kernel compiles (KQ_Q4_0): the compiler then drops the
 * code of the other types, and the kernel takes fewer registers (the
 * general form took 92 to 110). The values are the same. */
template <int NT, int FT = -1>
__global__ void k_kq_linear_i8(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), type = FT >= 0 ? FT : DI(5), t = DI(8);
    if (row >= rows) {
        return;
    }
    const uint8_t *w = DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(type, cols);
    if (NT == KQ_G8) {
        kq_row_i8_g8(type, w, DP(const float, 3), xq, xs, cols, t, DP(float, 9) + row,
                     (size_t)rows);
    } else if (NT == 0) {
        kq_row_i8_g(type, w, DP(const float, 3), xq, xs, cols, t, DP(float, 9) + row,
                    (size_t)rows);
    } else {
        kq_row_i8_t<NT == 0 ? 1 : NT>(type, w, DP(const float, 3), xq, xs, cols, t,
                                       DP(float, 9) + row, (size_t)rows);
    }
}

/* GP_KQ_MULTI of the gate and the up matrix (the same rows), and the
 * GP_GELU_MUL_ROWS after it, in one kernel: out[j][r] = gelu(g) u, as
 * k_gelu_mul_rows. The lanes 0 to 15 of warp r take row r of the gate, the
 * lanes 16 to 31 row r of the up matrix (kq_rows_i8 HALF); the two halves
 * read the same x. NT 0: a group (the counts 1 to KQ_NT, as kq_row_i8_g);
 * KQ_G8: a group of 5 to 8 in one part. */
template <int NT, int FT = -1>
__global__ void k_kq_glu_i8(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs,
                            float *out)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32, hs = (threadIdx.x / 16) % 2;
    int cols = DI(1), t = DI(2), rows = DI(6);
    if (row >= rows) {
        return;
    }
    int type = FT >= 0 ? FT : DI(5 + 4 * hs);
    const uint8_t *w = DP(const uint8_t, 4 + 4 * hs) + (size_t)row * kq_row_bytes(type, cols);
    const float *x = DP(const float, 0);
    for (int j0 = 0; j0 < t; j0 += (NT == 0 ? KQ_NT : NT)) {
        float res[KQ_G8];
        int n = t - j0 < (NT == 0 ? KQ_NT : NT) ? t - j0 : (NT == 0 ? KQ_NT : NT);
        const float *xj = x + (size_t)j0 * cols;
        const int8_t *qj = xq + (size_t)j0 * cols;
        const float *sj = xs + (size_t)j0 * (cols / 32);
        if (NT == KQ_G8) {
            kq_rows_i8_n<5, KQ_G8, true>(n, type, w, xj, qj, sj, cols, res);
        } else if (NT == 1 || n == 1) {
            kq_rows_i8<1, true>(type, w, xj, qj, sj, cols, 1, res);
        } else if (n == 2) {
            kq_rows_i8<2, true>(type, w, xj, qj, sj, cols, 2, res);
        } else if (n == 3) {
            kq_rows_i8<3, true>(type, w, xj, qj, sj, cols, 3, res);
        } else {
            kq_rows_i8<KQ_NT, true>(type, w, xj, qj, sj, cols, KQ_NT, res);
        }
        for (int j = 0; j < n; ++j) {
            float gv = __shfl_sync(RC_FULL_MASK, res[j], 0), uv = __shfl_sync(RC_FULL_MASK, res[j], 16);
            if (threadIdx.x % 32 == 0) {
                out[(size_t)(j0 + j) * rows + row] =
                    0.5f * gv * (1.0f + tanhf(0.7978845608028654f * (gv + 0.044715f * gv * gv * gv)))
                    * uv;
            }
        }
    }
}

/* GP_KQ_MULTI with the int8 x of the scratch (FT as k_kq_linear_i8). */
template <int NT, int FT = -1>
__global__ void k_kq_multi_i8(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int n = DI(3), cols = DI(1), t = DI(2), m = 0;
    while (m < n && row >= DI(6 + 4 * m)) {
        row -= DI(6 + 4 * m);
        ++m;
    }
    if (m >= n) {
        return;
    }
    int type = FT >= 0 ? FT : DI(5 + 4 * m), rows = DI(6 + 4 * m);
    const uint8_t *w = DP(const uint8_t, 4 + 4 * m) + (size_t)row * kq_row_bytes(type, cols);
    if (NT == KQ_G8) {
        kq_row_i8_g8(type, w, DP(const float, 0), xq, xs, cols, t, DP(float, 7 + 4 * m) + row,
                     (size_t)rows);
    } else if (NT == 0) {
        kq_row_i8_g(type, w, DP(const float, 0), xq, xs, cols, t, DP(float, 7 + 4 * m) + row,
                    (size_t)rows);
    } else {
        kq_row_i8_t<NT == 0 ? 1 : NT>(type, w, DP(const float, 0), xq, xs, cols, t,
                                       DP(float, 7 + 4 * m) + row, (size_t)rows);
    }
}

/* ---------- two rows a warp (Q4_0, a small group) ----------
 * The Q4_0 branch of kq_rows_i8 on the rows w0 and w1 at once: each row
 * keeps the lanes, the terms and their order of a row alone (the same bits:
 * a token of a group still gets the values of a decode step), and the int8
 * x of a block and the sum of it are read once for both. With one row a
 * warp a lane read 16 bytes of x and a scale for each token against 18 bytes
 * of weights, so the reads of x limited a group (4 tokens on the 12B cost
 * 2.2 times one token); the int4 kernels of the 26B gained 1.4 to 1.9 times
 * from the same change (k_mt_int4_rows). RECLANK_GPU_KQ_R2=0: one row. */
template <int NT, bool HALF = false>
__device__ void kq_rows2_q40(const uint8_t *w0, const uint8_t *w1, const int8_t *xq, const float *xs,
                             int cols, float *res0, float *res1)
{
    int lane = threadIdx.x % (HALF ? 16 : 32), np = cols / 32, hf = lane & 1;
    float s0[NT], s1[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        s0[j] = 0.f;
        s1[j] = 0.f;
    }
    for (int b = lane >> 1; b < cols / 32; b += HALF ? 8 : 16) {
        const uint8_t *b0 = w0 + (size_t)b * 18, *b1 = w1 + (size_t)b * 18;
        const uint16_t *q0 = (const uint16_t *)(b0 + 2), *q1 = (const uint16_t *)(b1 + 2);
        uint32_t qa[4], qb[4];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            qa[k] = (((uint32_t)q0[2 * k] | ((uint32_t)q0[2 * k + 1] << 16)) >> (4 * hf)) & 0x0F0F0F0Fu;
            qb[k] = (((uint32_t)q1[2 * k] | ((uint32_t)q1[2 * k + 1] << 16)) >> (4 * hf)) & 0x0F0F0F0Fu;
        }
        float da = kq_half(b0), db = kq_half(b1);
        #pragma unroll
        for (int jt = 0; jt < NT; ++jt) {
            uint4 xv = *(const uint4 *)(xq + (size_t)jt * cols + b * 32 + 16 * hf);
            int sa = __dp4a((int)qa[0], (int)xv.x, 0);
            sa = __dp4a((int)qa[1], (int)xv.y, sa);
            sa = __dp4a((int)qa[2], (int)xv.z, sa);
            sa = __dp4a((int)qa[3], (int)xv.w, sa);
            int sb = __dp4a((int)qb[0], (int)xv.x, 0);
            sb = __dp4a((int)qb[1], (int)xv.y, sb);
            sb = __dp4a((int)qb[2], (int)xv.z, sb);
            sb = __dp4a((int)qb[3], (int)xv.w, sb);
            int sx = __dp4a(0x01010101, (int)xv.x, 0);
            sx = __dp4a(0x01010101, (int)xv.y, sx);
            sx = __dp4a(0x01010101, (int)xv.z, sx);
            sx = __dp4a(0x01010101, (int)xv.w, sx);
            float xsv = xs[jt * np + b];
            s0[jt] += (float)(sa - 8 * sx) * (da * xsv);
            s1[jt] += (float)(sb - 8 * sx) * (db * xsv);
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v0 = s0[j], v1 = s1[j];
        for (int o = HALF ? 8 : 16; o > 0; o >>= 1) {
            v0 += __shfl_xor_sync(RC_FULL_MASK, v0, o);
            v1 += __shfl_xor_sync(RC_FULL_MASK, v1, o);
        }
        res0[j] = v0;
        res1[j] = v1;
    }
}

/* kq_rows2_q40 of the count n, LO <= n <= HI, as a constant (kq_rows_i8_n). */
template <int LO, int HI, bool HALF = false>
__device__ __forceinline__ void kq_rows2_q40_n(int n, const uint8_t *w0, const uint8_t *w1, const int8_t *xq,
                                               const float *xs, int cols, float *res0, float *res1)
{
    if constexpr (LO == HI) {
        kq_rows2_q40<LO, HALF>(w0, w1, xq, xs, cols, res0, res1);
    } else {
        if (n == LO) {
            kq_rows2_q40<LO, HALF>(w0, w1, xq, xs, cols, res0, res1);
        } else {
            kq_rows2_q40_n<LO + 1, HI, HALF>(n, w0, w1, xq, xs, cols, res0, res1);
        }
    }
}

/* The rows w0 and w1 on a group of t tokens, in the parts of kq_row_i8_g
 * (G8 false: parts of KQ_NT) or in one part (G8: 4 to KQ_G8 tokens; a
 * group of 4 in the parts of KQ_NT took 18.1 ms on the 12B, in one part
 * 15.0, one of 5 16.8):
 * row 0 to out0 (stride ostr0), row 1 to out1 (stride ostr1; none: null). */
template <bool G8>
__device__ void kq_row2_q40_g(const uint8_t *w0, const uint8_t *w1, const int8_t *xq, const float *xs,
                              int cols, int t, float *out0, size_t ostr0, float *out1, size_t ostr1)
{
    for (int j0 = 0; j0 < t; j0 += G8 ? KQ_G8 : KQ_NT) {
        float a[KQ_G8], c[KQ_G8];
        int n = G8 ? t : (t - j0 < KQ_NT ? t - j0 : KQ_NT);
        const int8_t *qj = xq + (size_t)j0 * cols;
        const float *sj = xs + (size_t)j0 * (cols / 32);
        if (G8) {
            kq_rows2_q40_n<KQ_NT, KQ_G8>(n, w0, w1, qj, sj, cols, a, c);
        } else {
            kq_rows2_q40_n<1, KQ_NT>(n, w0, w1, qj, sj, cols, a, c);
        }
        if (threadIdx.x % 32 == 0) {
            for (int j = 0; j < n; ++j) {
                out0[(size_t)(j0 + j) * ostr0] = a[j];
                if (out1 != NULL) {
                    out1[(size_t)(j0 + j) * ostr1] = c[j];
                }
            }
        }
    }
}

/* k_kq_linear_i8 of a Q4_0 matrix on a group (NT 0: 2 or 3 tokens, or
 * more than 8 in parts; KQ_G8: 4 to 8), two rows a warp (kq_rows2_q40). */
template <int NT>
__global__ void k_kq_linear_i8x2(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs)
{
    PDL_START();
    int row = (blockIdx.x * KQ_RPB + threadIdx.x / 32) * 2;
    int rows = DI(6), cols = DI(7), t = DI(8);
    if (row >= rows) {
        return;
    }
    size_t rb = kq_row_bytes(KQ_Q4_0, cols);
    const uint8_t *w0 = DP(const uint8_t, 4) + (size_t)row * rb;
    float *o = DP(float, 9) + row;
    bool two = row + 1 < rows;
    kq_row2_q40_g<NT == KQ_G8>(w0, two ? w0 + rb : w0, xq, xs, cols, t, o, (size_t)rows,
                               two ? o + 1 : NULL, (size_t)rows);
}

/* k_kq_multi_i8 of Q4_0 matrices on a group, two rows a warp: the rows 2 w
 * and 2 w + 1 of the matrices one after the other (the second may be row 0
 * of the next matrix). */
template <int NT>
__global__ void k_kq_multi_i8x2(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs)
{
    PDL_START();
    int row = (blockIdx.x * KQ_RPB + threadIdx.x / 32) * 2;
    int n = DI(3), cols = DI(1), t = DI(2), m = 0;
    while (m < n && row >= DI(6 + 4 * m)) {
        row -= DI(6 + 4 * m);
        ++m;
    }
    if (m >= n) {
        return;
    }
    size_t rb = kq_row_bytes(KQ_Q4_0, cols);
    int rows = DI(6 + 4 * m);
    const uint8_t *w0 = DP(const uint8_t, 4 + 4 * m) + (size_t)row * rb, *w1 = w0;
    float *o0 = DP(float, 7 + 4 * m) + row, *o1 = NULL;
    size_t s1 = (size_t)rows;
    if (row + 1 < rows) {
        w1 = w0 + rb;
        o1 = o0 + 1;
    } else if (m + 1 < n) {
        w1 = DP(const uint8_t, 4 + 4 * (m + 1));
        o1 = DP(float, 7 + 4 * (m + 1));
        s1 = (size_t)DI(6 + 4 * (m + 1));
    }
    kq_row2_q40_g<NT == KQ_G8>(w0, w1, xq, xs, cols, t, o0, (size_t)rows, o1, s1);
}

/* k_kq_glu_i8 of Q4_0 gate and up matrices on a group, two rows a warp: the
 * lanes 0 to 15 the rows of the gate, 16 to 31 those of the up matrix. */
template <int NT>
__global__ void k_kq_glu_i8x2(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs,
                              float *out)
{
    PDL_START();
    int row = (blockIdx.x * KQ_RPB + threadIdx.x / 32) * 2, hs = (threadIdx.x / 16) % 2;
    int cols = DI(1), t = DI(2), rows = DI(6);
    if (row >= rows) {
        return;
    }
    size_t rb = kq_row_bytes(KQ_Q4_0, cols);
    const uint8_t *w0 = DP(const uint8_t, 4 + 4 * hs) + (size_t)row * rb;
    bool two = row + 1 < rows;
    const uint8_t *w1 = two ? w0 + rb : w0;
    for (int j0 = 0; j0 < t; j0 += (NT == KQ_G8 ? KQ_G8 : KQ_NT)) {
        float a[KQ_G8], c[KQ_G8];
        int n = NT == KQ_G8 ? t : (t - j0 < KQ_NT ? t - j0 : KQ_NT);
        const int8_t *qj = xq + (size_t)j0 * cols;
        const float *sj = xs + (size_t)j0 * (cols / 32);
        if (NT == KQ_G8) {
            kq_rows2_q40_n<KQ_NT, KQ_G8, true>(n, w0, w1, qj, sj, cols, a, c);
        } else {
            kq_rows2_q40_n<1, KQ_NT, true>(n, w0, w1, qj, sj, cols, a, c);
        }
        for (int j = 0; j < n; ++j) {
            float gv = __shfl_sync(RC_FULL_MASK, a[j], 0), uv = __shfl_sync(RC_FULL_MASK, a[j], 16);
            float gw = __shfl_sync(RC_FULL_MASK, c[j], 0), uw = __shfl_sync(RC_FULL_MASK, c[j], 16);
            if (threadIdx.x % 32 == 0) {
                out[(size_t)(j0 + j) * rows + row] =
                    0.5f * gv * (1.0f + tanhf(0.7978845608028654f * (gv + 0.044715f * gv * gv * gv)))
                    * uv;
                if (two) {
                    out[(size_t)(j0 + j) * rows + row + 1] =
                        0.5f * gw * (1.0f + tanhf(0.7978845608028654f * (gw + 0.044715f * gw * gw * gw)))
                        * uw;
                }
            }
        }
    }
}

/* The two rows a warp of a Q4_0 group of t tokens on rows rows
 * (RECLANK_GPU_KQ_R2=0: never): a group (t 2 or more) on a matrix of 2048
 * rows or more, as k_mt_int4_rows (mt_rows_r). */
static int kq_r2(int64_t t, int64_t rows)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_KQ_R2");
        on = !(v && v[0] == '0');
    }
    return on && t >= 2 && rows >= 2048;
}

/* 8 values of a row from column c0 (a multiple of 8), as float32. */
__device__ void kq_dequant8(int type, const uint8_t *w, int cols, int c0, float *v)
{
    if (type == KQ_F32) {
        float4 a = *(const float4 *)((const float *)w + c0);
        float4 b = *(const float4 *)((const float *)w + c0 + 4);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else if (type == KQ_Q8_R) {
        float d = __half2float(((const __half *)(w + cols))[c0 / 32]);
        uint2 q = *(const uint2 *)(w + c0);
        for (int u = 0; u < 4; ++u) {
            v[u] = d * (float)(int8_t)((q.x >> (8 * u)) & 255);
            v[4 + u] = d * (float)(int8_t)((q.y >> (8 * u)) & 255);
        }
    } else if (type == KQ_Q8_0) {
        const uint8_t *blk = w + (size_t)(c0 / 32) * 34;
        float d = kq_half(blk);
        for (int u = 0; u < 8; ++u) {
            v[u] = d * (float)(int8_t)blk[2 + c0 % 32 + u];
        }
    } else if (type == KQ_Q4_0) {
        const uint8_t *blk = w + (size_t)(c0 / 32) * 18;
        float d = kq_half(blk);
        int j = c0 % 32, sh = j < 16 ? 0 : 4;
        for (int u = 0; u < 8; ++u) {
            v[u] = d * (float)((int)((blk[2 + j % 16 + u] >> sh) & 15) - 8);
        }
    } else if (type == KQ_BF12) {
        uint32_t bits[8];
        kq_bf12_bits<8>(w, cols, c0, bits);
        for (int u = 0; u < 8; ++u) {
            v[u] = __uint_as_float(bits[u] << 16);
        }
    } else if (type == KQ_BF16) {
        uint4 q = *(const uint4 *)((const uint16_t *)w + c0);
        uint32_t qq[4] = {q.x, q.y, q.z, q.w};
        for (int u = 0; u < 4; ++u) {
            v[2 * u] = kq_bf16((uint16_t)(qq[u] & 0xffff));
            v[2 * u + 1] = kq_bf16((uint16_t)(qq[u] >> 16));
        }
    } else if (type == KQ_NV4) {
        float g = *(const float *)(w + cols / 2 + cols / 16);
        int b = c0 / 32, j = c0 % 32, hi = j >= 16;
        float sc = 0.5f * g * kq_e4m3(w[cols / 2 + 2 * b + hi]);
        const uint8_t *q = w + 16 * b + (j % 16);
        for (int u = 0; u < 8; ++u) {
            v[u] = sc * kq_e2m1x2((q[u] >> (hi ? 4 : 0)) & 15);
        }
    } else if (type == KQ_Q5_1) {
        const uint8_t *blk = w + (size_t)(c0 / 32) * 24;
        float d = kq_half(blk), m = kq_half(blk + 2);
        uint32_t qh = *(const uint32_t *)(blk + 4);
        for (int u = 0; u < 8; ++u) {
            int j = c0 % 32 + u;
            int q = j < 16 ? blk[8 + j] & 15 : blk[8 + j - 16] >> 4;
            v[u] = d * (float)(q | (((qh >> j) & 1) << 4)) + m;
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        int five = type == KQ_Q5_K;
        const uint8_t *blk = w + (size_t)(c0 / 256) * (five ? 176 : 144);
        int q = c0 % 256, c = q / 64, off = q % 64, hi = off >= 32, l = off % 32;
        int j = 2 * c + hi, sc, m;
        kq_scale_min(blk + 4, j, &sc, &m);
        float d = kq_half(blk) * (float)sc, dm = kq_half(blk + 2) * (float)m;
        const uint8_t *qs = blk + (five ? 48 : 16) + 32 * c + l;
        for (int u = 0; u < 8; ++u) {
            int nib = hi ? qs[u] >> 4 : qs[u] & 15;
            if (five) {
                nib |= ((blk[16 + l + u] >> j) & 1) << 4;
            }
            v[u] = d * (float)nib - dm;
        }
    } else {
        const uint8_t *blk = w + (size_t)(c0 / 256) * 210;
        int q = c0 % 256, h = q / 128, rr = q % 128, u4 = rr / 32, l = rr % 32;
        const uint8_t *ql = blk + 64 * h + ((u4 & 1) ? 32 : 0) + l;
        const uint8_t *qh = blk + 128 + 32 * h + l;
        float d = kq_half(blk + 208) * (float)((const int8_t *)(blk + 192))[8 * h + l / 16 + 2 * u4];
        for (int u = 0; u < 8; ++u) {
            int lo = u4 >= 2 ? ql[u] >> 4 : ql[u] & 15;
            v[u] = d * (float)((lo | (((qh[u] >> (2 * u4)) & 3) << 4)) - 32);
        }
    }
}

/* A tile of the product of a group: rows r0 .. r0 + 63 of W (nrows rows,
 * rb bytes each) on n rows of x (row j at X + xmap[j] * xstride, or
 * X + j * xstride). O[j * ostride + r] gets row r, token j. 256 threads;
 * each computes 4 rows by 4 tokens. The weights come to shared memory as
 * float32, 32 columns at a time. */
#define KG_B 64
#define KG_K 32
__device__ void kq_tile(int type, const uint8_t *W, size_t rb, int nrows, int r0, int cols,
                        const float *X, size_t xstride, const int *xmap, int n, float *O,
                        size_t ostride)
{
    __shared__ __align__(16) float Ws[KG_K][KG_B + 4];
    __shared__ __align__(16) float Xs[KG_K][KG_B + 4];
    int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    int lr = tid / 4, lc = (tid % 4) * 8;
    const uint8_t *wrow = r0 + lr < nrows ? W + (size_t)(r0 + lr) * rb : NULL;
    const float *xrow = lr < n ? X + (size_t)(xmap ? xmap[lr] : lr) * xstride : NULL;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int k0 = 0; k0 < cols; k0 += KG_K) {
        float v[8];
        if (wrow != NULL && type == KQ_NVX) {
            int row = r0 + lr;
            kq_dequant8_nvx(W + (size_t)(row & ~15) * rb, row & 15, k0 + lc, v);
        } else if (wrow != NULL) {
            kq_dequant8(type, wrow, cols, k0 + lc, v);
        } else {
            for (int u = 0; u < 8; ++u) {
                v[u] = 0.f;
            }
        }
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            Ws[lc + u][lr] = v[u];
        }
        if (xrow != NULL) {
            float4 a = *(const float4 *)(xrow + k0 + lc), b = *(const float4 *)(xrow + k0 + lc + 4);
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
            v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
            for (int u = 0; u < 8; ++u) {
                v[u] = 0.f;
            }
        }
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            Xs[lc + u][lr] = v[u];
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KG_K; ++kk) {
            float4 a = *(const float4 *)&Ws[kk][ty * 4];
            float4 b = *(const float4 *)&Xs[kk][tx * 4];
            float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int rr = r0 + ty * 4 + i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int jj = tx * 4 + j;
            if (rr < nrows && jj < n) {
                O[(size_t)jj * ostride + rr] = acc[i][j];
            }
        }
    }
}

/* GP_KQ_HOT_MOE with int8 x (the types of kq_i8_type: the RQ6_K / RQ8_0
 * experts of Qwen3.8 are Q6_K and Q8_0 rows): x of the t tokens and then the
 * act of the pairs quantized once (kt_quant, a scale for each 32), a warp
 * for each row of an expert (kq_rows_i8, as the products of a step), in
 * place of kq_row on float32 x (10 KB of x read for each row of 2 KB):
 * Qwen3.8 126 -> see README us a layer. The pairs and the slots as k_kqh_gu
 * and k_kqh_dn. */
/* The x of the hot experts (RECLANK_GPU_KQH_X): "x2" (the default) int16 x
 * as two int8 planes (k_quant_x2: q = 128 hi + lo, a scale for each 32;
 * kq_row_x2), "i8" int8 x (kq_rows_i8), "f32" float32 x (k_kqh_gu,
 * k_kqh_dn). 0 for f32, 1 for i8, 2 for x2. */
static int kqh_x_mode(void)
{
    static int mode = -1;
    if (mode < 0) {
        const char *v = getenv("RECLANK_GPU_KQH_X");
        mode = v == NULL ? 2 : !strcmp(v, "f32") ? 0 : !strcmp(v, "i8") ? 1 : 2;
    }
    return mode;
}

/* The types of kq_row_x2 (the others read x as float32). */
__host__ __device__ __forceinline__ int kq_x2_type(int type)
{
    return type == KQ_Q8_0 || type == KQ_Q6_K || type == KQ_Q8_R;
}

/* A row of Q8_0, Q6_K, or Q8_R (the lanes of kq_rows_i8) on x in the int16 form:
 * hi and lo planes, the scale of each 32 values (k_quant_x2). The sum of a
 * part is 128 dp4a(w, hi) + dp4a(w, lo), exact in int32. Other types:
 * kq_row on the float32 x. */
__device__ float kq_row_x2(int type, const uint8_t *w, const float *x, const int8_t *xh,
                           const int8_t *xl, const float *xs, int cols)
{
    if (!kq_x2_type(type)) {
        return kq_row(type, w, x, cols);
    }
    int lane = threadIdx.x % 32;
    float sum = 0.f;
    if (type == KQ_Q8_0) {
        int hf = lane & 1;
        for (int b = lane >> 1; b < cols / 32; b += 16) {
            const uint8_t *blk = w + (size_t)b * 34;
            const uint16_t *q = (const uint16_t *)(blk + 2 + 16 * hf);
            uint32_t qw[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                qw[k] = (uint32_t)q[2 * k] | ((uint32_t)q[2 * k + 1] << 16);
            }
            uint4 hv = *(const uint4 *)(xh + b * 32 + 16 * hf), lv = *(const uint4 *)(xl + b * 32 + 16 * hf);
            int sh = __dp4a((int)qw[0], (int)hv.x, 0);
            sh = __dp4a((int)qw[1], (int)hv.y, sh);
            sh = __dp4a((int)qw[2], (int)hv.z, sh);
            sh = __dp4a((int)qw[3], (int)hv.w, sh);
            int sl = __dp4a((int)qw[0], (int)lv.x, 0);
            sl = __dp4a((int)qw[1], (int)lv.y, sl);
            sl = __dp4a((int)qw[2], (int)lv.z, sl);
            sl = __dp4a((int)qw[3], (int)lv.w, sl);
            sum += (float)(128 * sh + sl) * (kq_half(blk) * xs[b]);
        }
    } else if (type == KQ_Q8_R) {
        /* the int8 values of the row, then a float16 scale for each 32
         * (QwenGPU.dense: the shared experts) */
        const __half *dq = (const __half *)(w + cols);
        for (int c = 16 * lane; c < cols; c += 512) {
            uint4 q = *(const uint4 *)(w + c);
            uint4 hv = *(const uint4 *)(xh + c), lv = *(const uint4 *)(xl + c);
            int sh = __dp4a((int)q.x, (int)hv.x, 0);
            sh = __dp4a((int)q.y, (int)hv.y, sh);
            sh = __dp4a((int)q.z, (int)hv.z, sh);
            sh = __dp4a((int)q.w, (int)hv.w, sh);
            int sl = __dp4a((int)q.x, (int)lv.x, 0);
            sl = __dp4a((int)q.y, (int)lv.y, sl);
            sl = __dp4a((int)q.z, (int)lv.z, sl);
            sl = __dp4a((int)q.w, (int)lv.w, sl);
            sum += (float)(128 * sh + sl) * (__half2float(dq[c / 32]) * xs[c / 32]);
        }
    } else {
        int hf = lane & 1, u = (lane >> 1) & 1, n = (lane >> 2) & 1, sbk = lane >> 3;
        for (int b = sbk; b < cols / 256; b += 4) {
            const uint8_t *ql = w + (size_t)b * 210;
            const uint16_t *pl = (const uint16_t *)(ql + 64 * n + 32 * u + 16 * hf);
            const uint16_t *ph = (const uint16_t *)(ql + 128 + 32 * n + 16 * hf);
            const int8_t *sc = (const int8_t *)(ql + 192);
            float d = kq_half(ql + 208);
            uint32_t cl[4], ch[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t lq = (uint32_t)pl[2 * k] | ((uint32_t)pl[2 * k + 1] << 16);
                uint32_t hq = (uint32_t)ph[2 * k] | ((uint32_t)ph[2 * k + 1] << 16);
                cl[k] = __vsub4((lq & 0x0f0f0f0fu) | (((hq >> (2 * u)) & 0x03030303u) << 4), 0x20202020u);
                ch[k] = __vsub4(((lq >> 4) & 0x0f0f0f0fu) | (((hq >> (2 * u + 4)) & 0x03030303u) << 4),
                                0x20202020u);
            }
            float fl = d * (float)sc[8 * n + hf + 2 * u], fh = d * (float)sc[8 * n + hf + 2 * u + 4];
            int base = b * 256 + 128 * n + 32 * u + 16 * hf;
            uint4 h0 = *(const uint4 *)(xh + base), h1 = *(const uint4 *)(xh + base + 64);
            uint4 l0 = *(const uint4 *)(xl + base), l1 = *(const uint4 *)(xl + base + 64);
            int a = __dp4a((int)cl[0], (int)h0.x, 0);
            a = __dp4a((int)cl[1], (int)h0.y, a);
            a = __dp4a((int)cl[2], (int)h0.z, a);
            a = __dp4a((int)cl[3], (int)h0.w, a);
            int a2 = __dp4a((int)cl[0], (int)l0.x, 0);
            a2 = __dp4a((int)cl[1], (int)l0.y, a2);
            a2 = __dp4a((int)cl[2], (int)l0.z, a2);
            a2 = __dp4a((int)cl[3], (int)l0.w, a2);
            int c = __dp4a((int)ch[0], (int)h1.x, 0);
            c = __dp4a((int)ch[1], (int)h1.y, c);
            c = __dp4a((int)ch[2], (int)h1.z, c);
            c = __dp4a((int)ch[3], (int)h1.w, c);
            int c2 = __dp4a((int)ch[0], (int)l1.x, 0);
            c2 = __dp4a((int)ch[1], (int)l1.y, c2);
            c2 = __dp4a((int)ch[2], (int)l1.z, c2);
            c2 = __dp4a((int)ch[3], (int)l1.w, c2);
            sum += fl * xs[base / 32] * (float)(128 * a + a2) + fh * xs[base / 32 + 2] * (float)(128 * c + c2);
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    return sum;
}

/* GP_KQ_HOT_MOE with int8 x (mode 1: kq_rows_i8) or int16 x (mode 2: two
 * planes, kq_row_x2; xl the lo plane): x of the t tokens and then the act
 * of the pairs quantized once (a scale for each 32), a warp for each row of
 * an expert, in place of kq_row on float32 x (10 KB of x read for each row
 * of 2 KB). The pairs and the slots as k_kqh_gu and k_kqh_dn. */
__global__ void k_kqh_gu_i8(const gp_rec *r, const int64_t *e, const int8_t *xq, const int8_t *xl,
                            const float *xs, int jofs = 0)
{
    PDL_START();
    int j = blockIdx.y + jofs, slot = kqh_slot(r, e, j), up = blockIdx.z;
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int inner = DI(12), hidden = DI(13), k = DI(11), tk = DI(16) * k;
    if (slot < 0 || row >= inner) {
        return;
    }
    int shared = j >= tk;
    int type = shared ? DI(20) : DI(14);
    size_t rb = kq_row_bytes(type, hidden);
    const uint8_t *w = shared ? DP(const uint8_t, up ? 18 : 17)
                              : kqh_wslot(r, e, slot, up, (size_t)inner * rb);
    int tok = shared ? j - tk : j / k;
    const float *x = DP(const float, 0) + (size_t)tok * hidden;
    float v;
    if (xl != NULL) {
        v = kq_row_x2(type, w + (size_t)row * rb, x, xq + (size_t)tok * hidden, xl + (size_t)tok * hidden,
                      xs + (size_t)tok * (hidden / 32), hidden);
    } else {
        kq_rows_i8<1>(type, w + (size_t)row * rb, x, xq + (size_t)tok * hidden,
                      xs + (size_t)tok * (hidden / 32), hidden, 1, &v);
    }
    if (threadIdx.x % 32 == 0) {
        DP(float, 7)[(size_t)j * 2 * inner + up * inner + row] = v;
    }
}

__global__ void k_kqh_dn_i8(const gp_rec *r, const int64_t *e, const int8_t *aq, const int8_t *al,
                            const float *as, int jofs = 0)
{
    PDL_START();
    int j = blockIdx.y + jofs, slot = kqh_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int inner = DI(12), hidden = DI(13), tk = DI(16) * DI(11);
    if (slot < 0 || row >= hidden) {
        return;
    }
    int shared = j >= tk;
    int type = shared ? DI(20) : DI(15);
    size_t rb = kq_row_bytes(type, inner);
    const uint8_t *w = shared ? DP(const uint8_t, 19) : kqh_wslot(r, e, slot, 2, (size_t)hidden * rb);
    const float *x = DP(const float, 8) + (size_t)j * inner;
    float v;
    if (al != NULL) {
        v = kq_row_x2(type, w + (size_t)row * rb, x, aq + (size_t)j * inner, al + (size_t)j * inner,
                      as + (size_t)j * (inner / 32), inner);
    } else {
        kq_rows_i8<1>(type, w + (size_t)row * rb, x, aq + (size_t)j * inner, as + (size_t)j * (inner / 32),
                      inner, 1, &v);
    }
    if (threadIdx.x % 32 == 0) {
        DP(float, 9)[(size_t)j * hidden + row] = v;
    }
}

/* A row of Q8_0 or Q6_K on int16 x (the two planes and xs of kt_quant_x2)
 * with the operations of the CPU (kq_rows4_q8_x16_t1, kq_rows4_q6_x16_t1,
 * kq_t2_tile): 16 lanes (half a warp) are the 16 lanes of a vector of
 * AVX-512. Lane L of a step of 64 values (Q8_0: blocks 2q and 2q + 1; Q6_K:
 * a half of a block, its first 64 values (ia) and then the next 64 (ib))
 * takes values 4L to 4L + 3: the exact int32 sum 128 dp4a(w, hi) +
 * dp4a(w, lo), then acc = fma(sum, d xs, acc) (d xs one product; Q6_K: d
 * times the scale of the 16 values, then times xs). Q6_K: macc = fma(d sc,
 * xm, macc) for scale L of each block first, xm = xs times the sum of the
 * 16 values (kq_quant_part16). The lanes add as _mm512_reduce_add_ps of GCC
 * (lane i + i + 8, then + 4, + 2, + 1: the xor butterfly); Q6_K subtracts 32
 * times the sum of macc. The result: the bits of the cold experts of the
 * CPU (scripts/check_nvx_hot.py --q6). */
__device__ __forceinline__ float kq_row_x2c(int type, const uint8_t *w, const int8_t *xh, const int8_t *xl,
                                            const float *xs, int cols)
{
    const int L = threadIdx.x % 16;
    const unsigned mask = (threadIdx.x % 32) < 16 ? 0x0000ffffu : 0xffff0000u;
    float acc = 0.f, macc = 0.f;
    if (type == KQ_Q8_0) {
        for (int q = 0; q < cols / 64; ++q) {
            int b = 2 * q + (L >> 3);
            const uint8_t *blk = w + (size_t)b * 34;
            const uint16_t *pw = (const uint16_t *)(blk + 2 + 4 * (L & 7));
            int w4 = (int)((uint32_t)pw[0] | ((uint32_t)pw[1] << 16));
            int h4 = *(const int *)(xh + 64 * q + 4 * L), l4 = *(const int *)(xl + 64 * q + 4 * L);
            int is = 128 * __dp4a(w4, h4, 0) + __dp4a(w4, l4, 0);
            acc = fmaf((float)is, __fmul_rn(kq_half(blk), xs[b]), acc);
        }
    } else {
        for (int b = 0; b < cols / 256; ++b) {
            const uint8_t *blk = w + (size_t)b * 210;
            const int8_t *sc = (const int8_t *)(blk + 192);
            float d = kq_half(blk + 208);
            /* xm of the 16 values L of the block: xs times the sum of q */
            const int8_t *hq = xh + 256 * b + 16 * L, *lq = xl + 256 * b + 16 * L;
            int sq = 0;
            #pragma unroll
            for (int u = 0; u < 16; ++u) {
                sq += 128 * hq[u] + lq[u];
            }
            float xm = __fmul_rn(xs[8 * b + L / 2], (float)sq);
            macc = fmaf(__fmul_rn(d, (float)sc[L]), xm, macc);
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const uint8_t *ql = blk + 64 * h, *qh = blk + 128 + 32 * h;
                /* A: values 128 h + 4L .. (low 4 bits of ql, bits of qh from
                 * 0 or 2); B: values 128 h + 64 + 4L .. (high 4 bits, from 4
                 * or 6) */
                int i0 = 4 * L, sh = i0 < 32 ? 0 : 2;
                /* (a block is 210 bytes: 2-byte loads) */
                const uint16_t *pl = (const uint16_t *)(ql + i0), *ph = (const uint16_t *)(qh + (i0 & 31));
                uint32_t lw = (uint32_t)pl[0] | ((uint32_t)pl[1] << 16);
                uint32_t hw = (uint32_t)ph[0] | ((uint32_t)ph[1] << 16);
                int a4 = (int)((lw & 0x0f0f0f0fu) | (((hw >> sh) & 0x03030303u) << 4));
                int b4 = (int)(((lw >> 4) & 0x0f0f0f0fu) | (((hw >> (sh + 4)) & 0x03030303u) << 4));
                int xo = 256 * b + 128 * h + 4 * L;
                int ia = 128 * __dp4a(a4, *(const int *)(xh + xo), 0) + __dp4a(a4, *(const int *)(xl + xo), 0);
                int ib = 128 * __dp4a(b4, *(const int *)(xh + xo + 64), 0) +
                         __dp4a(b4, *(const int *)(xl + xo + 64), 0);
                float xa = xs[8 * b + 4 * h + (L >> 3)], xc = xs[8 * b + 4 * h + 2 + (L >> 3)];
                acc = fmaf((float)ia, __fmul_rn(__fmul_rn(d, (float)sc[8 * h + L / 4]), xa), acc);
                acc = fmaf((float)ib, __fmul_rn(__fmul_rn(d, (float)sc[8 * h + 4 + L / 4]), xc), acc);
            }
        }
    }
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        acc = __fadd_rn(acc, __shfl_xor_sync(mask, acc, o));
        macc = __fadd_rn(macc, __shfl_xor_sync(mask, macc, o));
    }
    if (type == KQ_Q8_0) {
        return acc;
    }
#ifdef KQ_X2C_FMA
    return fmaf(-32.f, macc, acc);
#else
    return __fsub_rn(acc, __fmul_rn(32.f, macc));
#endif
}

/* The types of kq_row_x2c. */
__host__ __device__ __forceinline__ int kq_x2c_type(int type, int cols)
{
    return type == KQ_Q8_0 ? cols % 64 == 0 : type == KQ_Q6_K && cols % 256 == 0;
}

/* The routed pairs (j < t top_k) of GP_KQ_HOT_MOE with kq_row_x2c: two rows a
 * warp (a half warp each); the gate and up (dn 0, blockIdx.z) or the down
 * rows (dn 1). The shared expert takes k_kqh_gu_i8 and k_kqh_dn_i8. */
template <int DN>
__global__ void k_kqh_x2c(const gp_rec *r, const int64_t *e, const int8_t *xq, const int8_t *xl,
                          const float *xs)
{
    PDL_START();
    int j = blockIdx.y, slot = kqh_slot(r, e, j), up = DN ? 2 : blockIdx.z;
    int row = blockIdx.x * 2 * ROWS_PER_BLOCK + 2 * (threadIdx.x / 32) + (threadIdx.x % 32) / 16;
    int inner = DI(12), hidden = DI(13), k = DI(11);
    int rows = DN ? hidden : inner, cols = DN ? inner : hidden;
    if (slot < 0 || row >= rows) {
        return;
    }
    int type = DN ? DI(15) : DI(14);
    size_t rb = kq_row_bytes(type, cols);
    const uint8_t *w = kqh_wslot(r, e, slot, up, (size_t)rows * rb);
    size_t x0 = DN ? (size_t)j * inner : (size_t)(j / k) * hidden;
    float v = kq_row_x2c(type, w + (size_t)row * rb, xq + x0, xl + x0, xs + x0 / 32, cols);
    if (threadIdx.x % 16 == 0) {
        if (DN) {
            DP(float, 9)[(size_t)j * hidden + row] = v;
        } else {
            DP(float, 7)[(size_t)j * 2 * inner + up * inner + row] = v;
        }
    }
}
