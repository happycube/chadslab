/* ---------- the products of large groups on dp4a (Pascal) ----------
 * kt_tile_dp4a: kt_tile (qmoe_tc.cu) for a GPU without the tensor cores. The
 * same inputs (x as int8 with xs and xsum for each block of 32, or the int16
 * form of KT_Q4X), the same loads of a step of 128 columns, and the same
 * values of each format: w = d q + mn for each block of 32 (two scales, of
 * the two halves of 16, for NV4, NVX, and Q6_K).
 *
 * The tile is 64 rows of x by 128 rows of w, 256 threads; a thread takes 4
 * rows of x (tg + 16 u) and 8 rows of w (rg + 16 i), as k_gemm_dp4a. For each
 * block of the step, the threads first decode the 128 rows of w once, to 8
 * words of 4 int8 values and their scales in shared memory (the fragments of
 * kt_tile, for all 4 places c of a row); then the inner loop is dp4a alone.
 * The int32 sum of a block (or of a half) is exact; the float32 sums go as
 * those of kt_tile, in another order. Its float: i2f_exact (|sum| < 2^22),
 * or with the int16 form of x a conversion (the sums of the unsigned codes
 * of Q4_K, Q5_K, Q5_1 and of the Q8 formats reach 2^23 to 2^26 there).
 *
 * One buffer: 48 KB of shared memory at most on Pascal (37 KB here, 45 KB
 * with the int16 form), given by the kernel (sm, KD_SMEM). */
#define KD_BM 64
#define KD_Q5K 8                   /* Q5_K: the tiles of dp4a only (kd_format) */
#define KD_WS 12                   /* the words of a decoded row (8 and 4 of pad) */
#define KD_AW(X2) (KT_K * (1 + (X2)) + 16)
#define KD_SMEM(X2) (KD_BM * KD_AW(X2) + KT_BN * KT_RB + KD_BM * 8 * 4 + KT_BN * KD_WS * 4 + \
                     3 * KT_BN * 4)

/* The format of the tiles of dp4a for a type: kt_format, and Q5_K. */
__host__ __device__ __forceinline__ int kd_format(int type)
{
    return type == KQ_Q5_K ? KD_Q5K : kt_format(type);
}

/* kt_load_w, and KD_Q5K: a step of 128 columns is half a block of 176 bytes;
 * a row of a step holds the head (d, dmin, the 12 bytes of scales), the 32
 * bytes of high bits (all; bit j is sub-block j), and the 64 bytes of low 4
 * bits of the half (112 bytes). */
template <int F>
__device__ __forceinline__ void kd_load_w(uint8_t *dst, const uint8_t *wrow, int cols, int k0)
{
    if (F == KD_Q5K) {
        int lane = threadIdx.x % 32;
        const uint8_t *blk = wrow + (size_t)(k0 / 256) * 176;
        if (lane < 3) {
            cp_async16(dst + 16 * lane, blk + 16 * lane);
        } else if (lane < 7) {
            cp_async16(dst + 16 * lane, blk + 48 + 64 * ((k0 / 128) % 2) + 16 * (lane - 3));
        }
    } else {
        kt_load_w<F>(dst, wrow, cols, k0);
    }
}

/* The words c and 4 + c (values 4c..4c+3 and 16+4c..16+4c+3) of block bk of
 * row n of the step at k0 in bsb (the shared memory of the raw step). */
template <int F>
__device__ __forceinline__ void kd_frag(const uint8_t *bsb, int n, int bk, int k0, int c, uint32_t *b)
{
    if (F == KD_Q5K) {
        const uint8_t *rp = bsb + n * KT_RB, *qs = rp + 48 + 32 * (bk / 2);
        int sh = 4 * (bk % 2), sb = 4 * ((k0 / 128) % 2) + bk;
        uint32_t l0 = (*(const uint32_t *)(qs + 4 * c) >> sh) & 0x0f0f0f0fu;
        uint32_t l1 = (*(const uint32_t *)(qs + 16 + 4 * c) >> sh) & 0x0f0f0f0fu;
        uint32_t h0 = (*(const uint32_t *)(rp + 16 + 4 * c) >> sb) & 0x01010101u;
        uint32_t h1 = (*(const uint32_t *)(rp + 32 + 4 * c) >> sb) & 0x01010101u;
        b[0] = l0 | (h0 << 4);
        b[1] = l1 | (h1 << 4);
    } else if (F == KT_Q4X || F == KT_NVX) {
        /* the groups of 16 rows: row r of a group is byte lane r % 8 of a
         * word, its low (r < 8) or high 4 bits */
        int r = n % 16, rr = r % 8, sh = 4 * (r / 8);
        const uint8_t *blk = F == KT_Q4X ? bsb + (n / 16) * KT_GB4 + 288 * bk
                                         : bsb + (n / 16) * KT_GB + 16 + KQ_NVX_BB * bk;
        uint32_t qa = *(const uint32_t *)(blk + 32 * c + 4 * rr) >> sh;
        uint32_t qb = *(const uint32_t *)(blk + 32 * (4 + c) + 4 * rr) >> sh;
        if (F == KT_Q4X) {
            b[0] = __vsub4(qa & 0x0f0f0f0fu, 0x08080808u);
            b[1] = __vsub4(qb & 0x0f0f0f0fu, 0x08080808u);
        } else {
            b[0] = kt_e2m1x4(qa);
            b[1] = kt_e2m1x4(qb);
        }
    } else if (F == KT_NV4) {
        uint32_t q = *(const uint32_t *)(bsb + n * KT_RB + 16 * bk + 4 * c);
        b[0] = kt_e2m1x4(q);
        b[1] = kt_e2m1x4(q >> 4);
    } else if (F == KT_Q6K) {
        const uint8_t *rp = bsb + n * KT_RB;
        const uint8_t *ql = rp + 32 * (bk & 1);
        int sh = 4 * (bk >> 1);
        uint32_t l0 = (*(const uint32_t *)(ql + 4 * c) >> sh) & 0x0f0f0f0fu;
        uint32_t l1 = (*(const uint32_t *)(ql + 16 + 4 * c) >> sh) & 0x0f0f0f0fu;
        uint32_t h0 = (*(const uint32_t *)(rp + 64 + 4 * c) >> (2 * bk)) & 0x03030303u;
        uint32_t h1 = (*(const uint32_t *)(rp + 80 + 4 * c) >> (2 * bk)) & 0x03030303u;
        b[0] = __vsub4(l0 | (h0 << 4), 0x20202020u);
        b[1] = __vsub4(l1 | (h1 << 4), 0x20202020u);
    } else {
        kt_frag<F>(bsb + n * KT_RB, bk, c, b);
    }
}

/* The scales of the two halves (dl, dh) and the term mn of block bk of row n. */
template <int F>
__device__ __forceinline__ void kd_scale(const uint8_t *bsb, int n, int bk, int k0, float *dl,
                                         float *dh, float *mn)
{
    *mn = 0.f;
    if (F == KT_Q4X) {
        const uint8_t *blk = bsb + (n / 16) * KT_GB4 + 288 * bk;
        *dl = *dh = __half2float(*(const __half *)(blk + 256 + 2 * (n % 16)));
    } else if (F == KT_NVX) {
        const uint8_t *gb = bsb + (n / 16) * KT_GB;
        const uint8_t *blk = gb + 16 + KQ_NVX_BB * bk;
        float gg = 128.f * *(const float *)gb;
        *dl = gg * kq_e4m3s(blk[256 + n % 16]);
        *dh = gg * kq_e4m3s(blk[272 + n % 16]);
    } else if (F == KT_NV4) {
        const uint8_t *rs = bsb + n * KT_RB;
        float gg = 0.5f * *(const float *)(rs + 72);
        *dl = gg * kq_e4m3(rs[64 + 2 * bk]);
        *dh = gg * kq_e4m3(rs[64 + 2 * bk + 1]);
    } else if (F == KD_Q5K) {
        kt_scale<KT_Q4K>(bsb + n * KT_RB, bk, k0, dl, mn);     /* the head of Q4_K */
        *dh = *dl;
    } else if (F == KT_Q6K) {
        const uint8_t *rs = bsb + n * KT_RB;
        float d = kq_half(rs + 104);
        *dl = d * (float)(int8_t)rs[96 + 2 * bk];
        *dh = d * (float)(int8_t)rs[96 + 2 * bk + 1];
    } else {
        kt_scale<F>(bsb + n * KT_RB, bk, k0, dl, mn);
        *dh = *dl;
    }
}

/* A tile: out[m * ostride + n] for m < mcount (rows of x: xmap[m], or m)
 * and n0 <= n < n0 + 128 (n < rows) of w (rb bytes each), as kt_tile. */
template <int F, int X2 = 0>
__device__ void kt_tile_dp4a(const uint8_t *w, size_t rb, int rows, int n0, int cols,
                             const int8_t *xq, const float *xs, const float *xsum, const int *xmap,
                             int mcount, float *out, size_t ostride, uint8_t *sm,
                             const int8_t *xl = NULL)
{
    constexpr bool SPLIT = F == KT_NV4 || F == KT_NVX || F == KT_Q6K;
    constexpr bool MN = F == KT_Q51 || F == KT_Q4K || F == KD_Q5K;
    constexpr int AW = KD_AW(X2);
    int8_t (*as_)[AW] = (int8_t (*)[AW])sm;
    uint8_t *bs = sm + KD_BM * AW;
    float (*ss)[8] = (float (*)[8])(bs + KT_BN * KT_RB);
    int (*dw)[KD_WS] = (int (*)[KD_WS])(&ss[KD_BM][0]);
    float *sdl = (float *)&dw[KT_BN][0], *sdh = sdl + KT_BN, *smn = sdh + KT_BN;
    const int tid = threadIdx.x, warp = tid / 32, rg = tid % 16, tg = tid / 16;
    const int nb = cols / 32, steps = F == KT_Q4X ? (cols + KT_K - 1) / KT_K : cols / KT_K;
    float acc[4][8];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[u][i] = 0.f;
        }
    }
    auto xrow = [&](int m) {
        int mm = m < mcount ? m : mcount - 1;
        return xmap ? xmap[mm] : mm;
    };
    for (int st = 0; st < steps; ++st) {
        const int k0 = st * KT_K;
        const int nbk = F == KT_Q4X ? min(4, nb - k0 / 32) : 4;
        /* the step: x, its scales, the raw rows of w (as kt_tile) */
        for (int e = tid; e < KD_BM * KT_K / 16; e += blockDim.x) {
            int m = e / (KT_K / 16), kk = (e % (KT_K / 16)) * 16;
            if (F != KT_Q4X || kk < 32 * nbk) {
                cp_async16(&as_[m][kk], xq + (size_t)xrow(m) * cols + k0 + kk);
                if (X2) {
                    cp_async16(&as_[m][KT_K + kk], xl + (size_t)xrow(m) * cols + k0 + kk);
                }
            }
        }
        for (int e = tid; e < KD_BM * 4; e += blockDim.x) {
            int m = e / 4, bk = e % 4;
            if (bk >= nbk) {
                continue;
            }
            size_t o = (size_t)xrow(m) * nb + k0 / 32 + bk;
            ss[m][bk] = xs[o];
            ss[m][4 + bk] = xsum[o];
        }
        if (F == KT_Q4X) {
            int nch = 18 * nbk;
            for (int e = tid; e < 8 * nch; e += blockDim.x) {
                int gi = e / nch, ch = e % nch;
                int grp = min(n0 / 16 + gi, rows / 16 - 1);
                cp_async16(bs + gi * KT_GB4 + 16 * ch,
                           w + (size_t)grp * 16 * rb + (size_t)(k0 / 32) * 288 + 16 * ch);
            }
        } else if (F == KT_NVX) {
            for (int e = tid; e < 8 * (KT_GB / 16); e += blockDim.x) {
                int gi = e / (KT_GB / 16), ch = e % (KT_GB / 16);
                int grp = min(n0 / 16 + gi, rows / 16 - 1);
                const uint8_t *src = w + (size_t)grp * 16 * rb +
                                     (ch == 0 ? 0 : 16 + (size_t)(k0 / 32) * KQ_NVX_BB + 16 * (ch - 1));
                cp_async16(bs + gi * KT_GB + 16 * ch, src);
            }
        } else {
            for (int n = warp; n < KT_BN; n += 8) {
                int row = min(n0 + n, rows - 1);
                kd_load_w<F>(bs + n * KT_RB, w + (size_t)row * rb, cols, k0);
            }
        }
        cp_async_commit();
        cp_async_wait0();
        __syncthreads();
        #pragma unroll 1
        for (int bk = 0; bk < nbk; ++bk) {
            /* decode: 2 threads a row, the places c = 2h, 2h + 1 */
            {
                int n = tid % KT_BN, h = tid / KT_BN;
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    uint32_t b[2];
                    kd_frag<F>(bs, n, bk, k0, 2 * h + cc, b);
                    dw[n][2 * h + cc] = (int)b[0];
                    dw[n][4 + 2 * h + cc] = (int)b[1];
                }
                if (h == 0) {
                    kd_scale<F>(bs, n, bk, k0, &sdl[n], &sdh[n], &smn[n]);
                }
            }
            __syncthreads();
            const int kk = bk * 32;
            float s[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                s[u] = ss[tg + 16 * u][bk];
            }
            /* the parts with a scale of their own: the block, or its two
             * halves (SPLIT); for each, the sums of the planes of x (the int16
             * form: 128 times those of hi, then lo) */
            constexpr int NH = SPLIT ? 2 : 1;
            #pragma unroll
            for (int hp = 0; hp < NH; ++hp) {
                int isum[4][8];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        isum[u][i] = 0;
                    }
                }
                #pragma unroll 1
                for (int p = 0; p <= X2; ++p) {
                    if (X2 && p == 1) {
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                isum[u][i] *= 128;
                            }
                        }
                    }
                    #pragma unroll
                    for (int hw = SPLIT ? hp : 0; hw < (SPLIT ? hp + 1 : 2); ++hw) {
                        int4 a[4];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            a[u] = *(const int4 *)&as_[tg + 16 * u][p * KT_K + kk + 16 * hw];
                        }
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            int4 b = *(const int4 *)&dw[rg + 16 * i][4 * hw];
                            #pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                isum[u][i] = rc_dp4a(b.x, a[u].x, isum[u][i]);
                                isum[u][i] = rc_dp4a(b.y, a[u].y, isum[u][i]);
                                isum[u][i] = rc_dp4a(b.z, a[u].z, isum[u][i]);
                                isum[u][i] = rc_dp4a(b.w, a[u].w, isum[u][i]);
                            }
                        }
                    }
                }
                const float *dd = hp ? sdh : sdl;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float d = dd[rg + 16 * i];
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        float f = X2 ? (float)isum[u][i] : i2f_exact(isum[u][i]);
                        acc[u][i] = fmaf(f, s[u] * d, acc[u][i]);
                    }
                }
            }
            if (MN) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    float su = ss[tg + 16 * u][4 + bk];
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        acc[u][i] = fmaf(su, smn[rg + 16 * i], acc[u][i]);
                    }
                }
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        int m = tg + 16 * u;
        if (m >= mcount) {
            continue;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int n = n0 + rg + 16 * i;
            if (n < rows) {
                out[(size_t)m * ostride + n] = acc[u][i];
            }
        }
    }
}

/* GP_KQ_LINEAR of a large group on dp4a: blockIdx.x the tile of 64 tokens,
 * blockIdx.y the tile of 128 rows (k_kq_tc). X2: x in the int16 form (xl, the
 * lo plane; RECLANK_GPU_KQ_X2). */
template <int F, int X2>
__global__ void __launch_bounds__(256, 2) k_kq_dp4a(const gp_rec *r, const int64_t *e,
                                                    const int8_t *xq, const float *xs,
                                                    const float *xsum, int unused,
                                                    const int8_t *xl)
{
    PDL_START();
    __shared__ __align__(16) uint8_t kdsm[KD_SMEM(X2)];
    int rows = DI(6), cols = DI(7), t = DI(8), type = DI(5);
    int m0 = blockIdx.x * KD_BM;
    kt_tile_dp4a<F, X2>(DP(const uint8_t, 4), kq_row_bytes(type, cols), rows, blockIdx.y * KT_BN, cols,
                        xq + (size_t)m0 * cols, xs + (size_t)m0 * (cols / 32),
                        xsum + (size_t)m0 * (cols / 32), NULL, min(KD_BM, t - m0),
                        DP(float, 9) + (size_t)m0 * rows, (size_t)rows, kdsm,
                        X2 ? xl + (size_t)m0 * cols : NULL);
}

/* GP_KQ_GROUP_MOE on dp4a (k_qmoe_gu_tc, k_qmoe_dn_tc). shared 0: the tiles
 * of the routed experts, of format KFG; 1: the tiles of the shared expert
 * (x == E), of format KFG (its own launch, so it can have any format). X2:
 * the int16 form of x (k_quant_x2; the lo plane at xl), for the precision of
 * RECLANK_MOE_X16 (the float32 x of k_qmoe_gu there). */
template <int KFG, int X2>
__global__ void __launch_bounds__(256, 2) k_qmoe_gu_dp4a(const gp_rec *r, const int64_t *e,
                                                         const int8_t *xq, const float *xs,
                                                         const float *xsum, int shared,
                                                         const int8_t *xl)
{
    PDL_START();
    __shared__ __align__(16) uint8_t kdsm[KD_SMEM(X2)];
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    if ((x == E) != (shared != 0)) {
        return;
    }
    int row0 = blockIdx.x * KT_BN, up = row0 >= inner;
    float *out = DP(float, 19) + (size_t)p0 * 2 * inner + (up ? inner : 0);
    const uint8_t *W = x < E ? (const uint8_t *)(intptr_t)DP(const int64_t, up ? 9 : 8)[x]
                             : DP(const uint8_t, up ? 14 : 13);
    kt_tile_dp4a<KFG, X2>(W, kq_row_bytes(x < E ? DI(11) : DI(16), hidden), inner,
                          row0 - (up ? inner : 0), hidden, xq, xs, xsum, w.ptok + p0, n, out,
                          (size_t)2 * inner, kdsm, xl);
}

template <int KFD, int X2>
__global__ void __launch_bounds__(256, 2) k_qmoe_dn_dp4a(const gp_rec *r, const int64_t *e,
                                                         const int8_t *xq, const float *xs,
                                                         const float *xsum, int shared,
                                                         const int8_t *xl)
{
    PDL_START();
    __shared__ __align__(16) uint8_t kdsm[KD_SMEM(X2)];
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    if ((x == E) != (shared != 0)) {
        return;
    }
    float *out = DP(float, 21) + (size_t)p0 * hidden;
    const int8_t *xr = xq + (size_t)p0 * inner;
    const float *sr = xs + (size_t)p0 * (inner / 32), *ur = xsum + (size_t)p0 * (inner / 32);
    const uint8_t *W = x < E ? (const uint8_t *)(intptr_t)DP(const int64_t, 10)[x]
                             : DP(const uint8_t, 15);
    kt_tile_dp4a<KFD, X2>(W, kq_row_bytes(x < E ? DI(12) : DI(16), inner), hidden,
                          blockIdx.x * KT_BN, inner, xr, sr, ur, NULL, n, out, (size_t)hidden, kdsm,
                          X2 ? xl + (size_t)p0 * inner : NULL);
}

/* GP_MOE_GPU with KQ_Q4X experts on dp4a (k_moe_gemm_q4x). */
template <int X2>
__global__ void __launch_bounds__(256, 2) k_moe_gemm_q4x_dp4a(const gp_rec *r, const int64_t *e,
                                                              const int8_t *xq, const float *xs,
                                                              const float *xsum, int gather,
                                                              const int64_t *wtab, int rows,
                                                              int cols, float *out,
                                                              const int8_t *xl)
{
    PDL_START();
    __shared__ __align__(16) uint8_t kdsm[KD_SMEM(X2)];
    const int *tiles = DP(const int, 16);
    const int *off = DP(const int, 12);
    int tile = blockIdx.y;
    if (tile >= tiles[0]) {
        return;
    }
    int ex = tiles[1 + 2 * tile], q0 = tiles[2 + 2 * tile];
    int q1 = min(off[ex + 1], q0 + GM);
    size_t rb = (size_t)(cols / 32) * 18, nbx = (size_t)(cols / 32);
    const uint8_t *W = (const uint8_t *)(intptr_t)wtab[ex];
    if (gather) {
        kt_tile_dp4a<KT_Q4X, X2>(W, rb, rows, blockIdx.x * KT_BN, cols, xq, xs, xsum,
                                 DP(const int, 14) + q0, q1 - q0, out + (size_t)q0 * rows,
                                 (size_t)rows, kdsm, xl);
    } else {
        kt_tile_dp4a<KT_Q4X, X2>(W, rb, rows, blockIdx.x * KT_BN, cols, xq + (size_t)q0 * cols,
                                 xs + (size_t)q0 * nbx, xsum + (size_t)q0 * nbx, NULL, q1 - q0,
                                 out + (size_t)q0 * rows, (size_t)rows, kdsm,
                                 X2 ? xl + (size_t)q0 * cols : NULL);
    }
}

/* The products of large groups on dp4a: a GPU without the tensor cores and
 * with dp4a (sm_61); RECLANK_GPU_DP4A=0 keeps the float32 tiles (a test). */
static int kt_dp4a_on(void)
{
    return !gg_caps.tc && dp4a_on();
}

/* RECLANK_GPU_KQ_X2=1: GP_KQ_LINEAR on dp4a with x in the int16 form (more
 * precision, twice the dp4a; scripts/check_dp4a.py tests the int16 form of
 * every format with it). Read at each launch. */
static int kq_x2_on(void)
{
    const char *v = getenv("RECLANK_GPU_KQ_X2");
    return v != NULL && v[0] == '1';
}
