/* ---------- the parts and rows of k_pr_mv (pr.cu) ----------
 * pr_mv_part decodes part p of a row of any of the types of Qwen3.8-27B's records (pr_fmt.cuh's, Q4_K, Q5_K,
 * Q6_K, Q8_R, float32) to 8 words of int8 values and its scales; pr_rows sums a row with NT tokens. A header
 * of its own so the standalone bench (scratch mvb.cu) runs the same code. */
/* the types of k_pr_mv */
__host__ __device__ __forceinline__ int kq_pr_mv_type(int type)
{
    return kq_pr_type(type) || type == KQ_Q4_K || type == KQ_Q5_K || type == KQ_Q6_K || type == KQ_Q8_R ||
           type == KQ_F32;
}

/* Part p of a row of type F: the 8 words (values 4k .. 4k+3 of the 32), the scales of values 0-15 (dl) and
 * 16-31 (dh), and mn, the factor of the part's xsum to take off (Q4_K, Q5_K: dmin m). */
template <int F>
__device__ __forceinline__ void pr_mv_part(const uint8_t *w, int cols, int p, const uint32_t *t3x, const uint32_t *t3s,
                                        uint32_t *wd, float *dl, float *dh, float *mn)
{
    if (F == KQ_Q4_K || F == KQ_Q5_K) {
        /* part j of the block: the 32 bytes of codes of parts 2 (j / 2), + 1 (the low 4 bits, the high) */
        const int j = p & 7;
        const uint8_t *blk = w + (size_t)(p >> 3) * (F == KQ_Q5_K ? 176 : 144);
        const uint8_t *qs = blk + (F == KQ_Q5_K ? 48 : 16) + 32 * (j >> 1);
        const uint4 a = *(const uint4 *)qs, b = *(const uint4 *)(qs + 16);
        const uint32_t q[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
        const int sh = 4 * (j & 1);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            wd[k] = (q[k] >> sh) & 0x0f0f0f0fu;
        }
        if (F == KQ_Q5_K) {
            const uint4 ha = *(const uint4 *)(blk + 16), hb = *(const uint4 *)(blk + 32);
            const uint32_t h[8] = {ha.x, ha.y, ha.z, ha.w, hb.x, hb.y, hb.z, hb.w};
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                wd[k] |= ((h[k] >> j) & 0x01010101u) << 4;
            }
        }
        int sc, m;
        kq_scale_min(blk + 4, j, &sc, &m);
        const uint32_t dd = *(const uint32_t *)blk;
        *dl = *dh = __half2float(__ushort_as_half((uint16_t)(dd & 0xffffu))) * (float)sc;
        *mn = __half2float(__ushort_as_half((uint16_t)(dd >> 16))) * (float)m;
    } else if (F == KQ_Q6_K) {
        /* part J = 4 n + q of the block (210 bytes: 2-byte loads): ql bytes 64 n + 32 (q & 1) .., the 4 bits
         * q >> 1; qh bytes 128 + 32 n .., bits 2 q; the scales 8 n + 2 q, + 1; the codes less 32 */
        const int J = p & 7, n = J >> 2, q = J & 3;
        const uint8_t *blk = w + (size_t)(p >> 3) * 210;
        const uint16_t *pl = (const uint16_t *)(blk + 64 * n + 32 * (q & 1));
        const uint16_t *ph = (const uint16_t *)(blk + 128 + 32 * n);
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t lq = (uint32_t)pl[2 * k] | ((uint32_t)pl[2 * k + 1] << 16);
            const uint32_t hq = (uint32_t)ph[2 * k] | ((uint32_t)ph[2 * k + 1] << 16);
            const uint32_t v = ((lq >> (4 * (q >> 1))) & 0x0f0f0f0fu) | (((hq >> (2 * q)) & 0x03030303u) << 4);
            /* v - 32 for each byte (v < 64): bit 5 flipped, and 0xc0 where it was 0 */
            wd[k] = (v ^ 0x20202020u) | ((~v & 0x20202020u) * 6u);
        }
        const float d = __half2float(*(const __half *)(blk + 208));
        const int8_t *sc = (const int8_t *)(blk + 192) + 8 * n + 2 * q;
        *dl = d * (float)sc[0];
        *dh = d * (float)sc[1];
        *mn = 0.f;
    } else if (F == KQ_Q8_R) {
        const uint4 a = *(const uint4 *)(w + 32 * (size_t)p), b = *(const uint4 *)(w + 32 * (size_t)p + 16);
        wd[0] = a.x; wd[1] = a.y; wd[2] = a.z; wd[3] = a.w;
        wd[4] = b.x; wd[5] = b.y; wd[6] = b.z; wd[7] = b.w;
        *dl = *dh = __half2float(((const __half *)(w + cols))[p]);
        *mn = 0.f;
    } else {
        pr_part(F, w, cols, p, t3x, t3s, wd, dl, dh);
        *mn = 0.f;
    }
}

/* The sums of row w with NT tokens (the x of token j at xq + j cols, xs and xsum + j cols / 32); res[j],
 * each lane. */
template <int F, int NT>
__device__ __forceinline__ void pr_rows(const uint8_t *w, int cols, const int8_t *xq, const float *xs,
                                        const float *xsum, const uint32_t *t3x, const uint32_t *t3s, float *res)
{
    constexpr bool SPLIT = F == KQ_PR_Q3K || F == KQ_PR_IQ2S || F == KQ_Q6_K || F == KQ_PR_Q6K;
    constexpr bool MIN = F == KQ_Q4_K || F == KQ_Q5_K;
    const int lane = threadIdx.x % 32, np = cols / 32;
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    for (int p = lane; p < np; p += 32) {
        /* (token 0's x loads before the decode, so their latency is under it) */
        const uint4 a0 = *(const uint4 *)(xq + 32 * p), b0 = *(const uint4 *)(xq + 32 * p + 16);
        uint32_t wd[8];
        float dl, dh, mn;
        pr_mv_part<F>(w, cols, p, t3x, t3s, wd, &dl, &dh, &mn);
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int8_t *xp = xq + (size_t)j * cols + 32 * p;
            const uint4 a = j == 0 ? a0 : *(const uint4 *)xp, b = j == 0 ? b0 : *(const uint4 *)(xp + 16);
            int sl = rc_dp4a((int)wd[0], (int)a.x, 0);
            sl = rc_dp4a((int)wd[1], (int)a.y, sl);
            sl = rc_dp4a((int)wd[2], (int)a.z, sl);
            sl = rc_dp4a((int)wd[3], (int)a.w, sl);
            int sh = rc_dp4a((int)wd[4], (int)b.x, SPLIT ? 0 : sl);
            sh = rc_dp4a((int)wd[5], (int)b.y, sh);
            sh = rc_dp4a((int)wd[6], (int)b.z, sh);
            sh = rc_dp4a((int)wd[7], (int)b.w, sh);
            const float xsj = xs[(size_t)j * np + p];
            const float v = SPLIT ? fmaf(dl, i2f_exact(sl), dh * i2f_exact(sh)) : dl * i2f_exact(sh);
            if (MIN) {
                sum[j] += fmaf(xsj, v, -mn * xsum[(size_t)j * np + p]);
            } else {
                sum[j] = fmaf(xsj, v, sum[j]);
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(RC_FULL_MASK, v, o);
        }
        res[j] = v;
    }
}

/* pr_rows for R rows at once (row k at w + k rb): each token's x loads once for the R rows (a verify group's
 * products read x from L1 for each row: 8 tokens x 32 bytes a part, 18 times the weights' bytes of a
 * part of IQ3_S). res[k * NT + j]. The terms of a token and a row in the order of pr_rows (the same bits). */
template <int F, int NT, int R>
__device__ __forceinline__ void pr_rows_r(const uint8_t *w, size_t rb, int cols, const int8_t *xq, const float *xs,
                                          const float *xsum, const uint32_t *t3x, const uint32_t *t3s, float *res)
{
    constexpr bool SPLIT = F == KQ_PR_Q3K || F == KQ_PR_IQ2S || F == KQ_Q6_K || F == KQ_PR_Q6K;
    constexpr bool MIN = F == KQ_Q4_K || F == KQ_Q5_K;
    const int lane = threadIdx.x % 32, np = cols / 32;
    float sum[R][NT];
    #pragma unroll
    for (int k = 0; k < R; ++k) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            sum[k][j] = 0.f;
        }
    }
    for (int p = lane; p < np; p += 32) {
        uint32_t wd[R][8];
        float dl[R], dh[R], mn[R];
        #pragma unroll
        for (int k = 0; k < R; ++k) {
            pr_mv_part<F>(w + (size_t)k * rb, cols, p, t3x, t3s, wd[k], &dl[k], &dh[k], &mn[k]);
        }
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int8_t *xp = xq + (size_t)j * cols + 32 * p;
            const uint4 a = *(const uint4 *)xp, b = *(const uint4 *)(xp + 16);
            const float xsj = xs[(size_t)j * np + p], xmj = MIN ? xsum[(size_t)j * np + p] : 0.f;
            #pragma unroll
            for (int k = 0; k < R; ++k) {
                int sl = rc_dp4a((int)wd[k][0], (int)a.x, 0);
                sl = rc_dp4a((int)wd[k][1], (int)a.y, sl);
                sl = rc_dp4a((int)wd[k][2], (int)a.z, sl);
                sl = rc_dp4a((int)wd[k][3], (int)a.w, sl);
                int sh = rc_dp4a((int)wd[k][4], (int)b.x, SPLIT ? 0 : sl);
                sh = rc_dp4a((int)wd[k][5], (int)b.y, sh);
                sh = rc_dp4a((int)wd[k][6], (int)b.z, sh);
                sh = rc_dp4a((int)wd[k][7], (int)b.w, sh);
                const float v = SPLIT ? fmaf(dl[k], i2f_exact(sl), dh[k] * i2f_exact(sh)) : dl[k] * i2f_exact(sh);
                if (MIN) {
                    sum[k][j] += fmaf(xsj, v, -mn[k] * xmj);
                } else {
                    sum[k][j] = fmaf(xsj, v, sum[k][j]);
                }
            }
        }
    }
    #pragma unroll
    for (int k = 0; k < R; ++k) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            float v = sum[k][j];
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            res[k * NT + j] = v;
        }
    }
}

/* float32 rows (Q2_K's small matrices): the float x */
template <int NT>
__device__ __forceinline__ void pr_rows_f32(const float *w, int cols, const float *x, float *res)
{
    const int lane = threadIdx.x % 32;
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    for (int c = 4 * lane; c < cols; c += 128) {
        const float4 a = *(const float4 *)(w + c);
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float4 b = *(const float4 *)(x + (size_t)j * cols + c);
            sum[j] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(RC_FULL_MASK, v, o);
        }
        res[j] = v;
    }
}

template <int NT>
__device__ __forceinline__ void pr_row_any(int type, const uint8_t *w, int cols, const float *x, const int8_t *xq,
                                           const float *xs, const float *xsum, const uint32_t *t3x,
                                           const uint32_t *t3s, float *res)
{
    switch (type) {
    case KQ_PR_IQ4X: pr_rows<KQ_PR_IQ4X, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_PR_IQ3S: pr_rows<KQ_PR_IQ3S, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_PR_IQ3X: pr_rows<KQ_PR_IQ3X, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_PR_Q3K: pr_rows<KQ_PR_Q3K, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_PR_IQ2S: pr_rows<KQ_PR_IQ2S, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_PR_IQ2X: pr_rows<KQ_PR_IQ2X, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_PR_Q6K: pr_rows<KQ_PR_Q6K, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_Q4_K: pr_rows<KQ_Q4_K, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_Q5_K: pr_rows<KQ_Q5_K, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_Q6_K: pr_rows<KQ_Q6_K, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    case KQ_Q8_R: pr_rows<KQ_Q8_R, NT>(w, cols, xq, xs, xsum, t3x, t3s, res); break;
    default: pr_rows_f32<NT>((const float *)w, cols, x, res); break;
    }
}


/* ---- the products of a prompt's group (t > MT_MAX) ----
 * k_pr_gemm: out[m * rows + n] for the tile of PR_GBM tokens by PR_GBN rows (blockIdx.x, blockIdx.y) of a
 * matrix of type F (pr_mv_part's types) on int8 x. A step of 128 columns: the x of the tile to shared
 * memory, each thread decodes 2 of the tile's 128 x 4 parts (pr_mv_part, from global memory) to words and
 * scales in shared memory; one sync; then each thread's 4 tokens by 8 rows over the 4 parts, dp4a alone in
 * the inner loop; one sync. (The dp4a tiles of qmoe_dp4a.cu decode and sync for each 32 columns: two
 * syncs, the decode serial with the products; ~0.5-0.8 T dp4a/s on a P6000.) */
#define PR_GBM 64                    /* the tokens of a tile */
#define PR_GBN 128
#define PR_GDW 36                    /* the words of a decoded row in shared memory (32 and 4 of pad) */

template <int F>
__global__ void __launch_bounds__(256, 2) k_pr_gemm(const uint8_t *W, size_t rb, int rows, int cols, int t,
                                                     const int8_t *xq, const float *xs, const float *xsum, float *out)
{
    /* (BM the tokens of the tile, NT its threads, TG its groups of 4 tokens: tg + TG u. 128 tokens with 512
     * threads (1 block an SM) ran 6 times slower in the bench) */
    constexpr int BM = PR_GBM, NT = 4 * BM, TG = NT / 16;
    constexpr bool SPLIT = F == KQ_PR_Q3K || F == KQ_PR_IQ2S || F == KQ_Q6_K || F == KQ_PR_Q6K;
    constexpr bool MIN = F == KQ_Q4_K || F == KQ_Q5_K;
    constexpr int TAB = F == KQ_PR_IQ3X ? PR_IQ3X_TAB : F == KQ_PR_IQ3S ? PR_IQ3S_TAB : 1;
    __shared__ __align__(16) int8_t sx[BM][128 + 16];
    __shared__ float sxs[BM][4], sxm[BM][4];
    __shared__ __align__(16) uint32_t dw[PR_GBN][PR_GDW];
    __shared__ float sdl[PR_GBN][4], sdh[PR_GBN][4], smn[PR_GBN][4];
    __shared__ uint32_t tab[TAB];
    PDL_START();
    if (F == KQ_PR_IQ3X || F == KQ_PR_IQ3S) {
        pr_tables_fill(F == KQ_PR_IQ3X ? tab : NULL, F == KQ_PR_IQ3S ? tab : NULL);
        __syncthreads();
    }
    const int tid = threadIdx.x, rg = tid % 16, tg = tid / 16;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * PR_GBN, np = cols / 32;
    float acc[4][8];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[u][i] = 0.f;
        }
    }
    for (int k0 = 0; k0 < cols; k0 += 128) {
        /* the x of the step: 64 rows of 128 bytes (the rows past t: the last one), their scales */
        #pragma unroll
        for (int e = tid; e < BM * 8; e += NT) {
            const int m = e / 8, c = e % 8, mm = min(m0 + m, t - 1);
            *(uint4 *)&sx[m][16 * c] = *(const uint4 *)(xq + (size_t)mm * cols + k0 + 16 * c);
        }
        {
            const int m = tid / 4, bk = tid % 4, mm = min(m0 + m, t - 1);       /* (4 BM threads: one a scale) */
            sxs[m][bk] = xs[(size_t)mm * np + k0 / 32 + bk];
            if (MIN) {
                sxm[m][bk] = xsum[(size_t)mm * np + k0 / 32 + bk];
            }
        }
        /* the parts of the step: 128 rows by 4 (two a thread, or one with 512) */
        #pragma unroll
        for (int j = 0; j < 512 / NT; ++j) {
            const int it = tid + NT * j, n = it / 4, bk = it % 4, row = min(n0 + n, rows - 1);
            uint32_t wd[8];
            float dl, dh, mn;
            pr_mv_part<F>(W + (size_t)row * rb, cols, k0 / 32 + bk, F == KQ_PR_IQ3X ? tab : NULL, F == KQ_PR_IQ3S ? tab : NULL,
                       wd, &dl, &dh, &mn);
            *(uint4 *)&dw[n][8 * bk] = make_uint4(wd[0], wd[1], wd[2], wd[3]);
            *(uint4 *)&dw[n][8 * bk + 4] = make_uint4(wd[4], wd[5], wd[6], wd[7]);
            sdl[n][bk] = dl;
            sdh[n][bk] = dh;
            smn[n][bk] = mn;
        }
        __syncthreads();
        /* (the 4 parts one at a time: unrolled, the 128 registers spilled, 1.6 -> 1.1 T dp4a/s) */
        #pragma unroll 1
        for (int bk = 0; bk < 4; ++bk) {
            int il[4][8], ih[4][8];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    il[u][i] = 0;
                    ih[u][i] = 0;
                }
            }
            #pragma unroll
            for (int hw = 0; hw < 2; ++hw) {
                int4 a[4];
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    a[u] = *(const int4 *)&sx[tg + TG * u][32 * bk + 16 * hw];
                }
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int4 b = *(const int4 *)&dw[rg + 16 * i][8 * bk + 4 * hw];
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        /* (SPLIT: the second half of the part to its own sum, its own scale) */
                        int &acc_i = SPLIT && hw == 1 ? ih[u][i] : il[u][i];
                        int s = rc_dp4a(b.x, a[u].x, acc_i);
                        s = rc_dp4a(b.y, a[u].y, s);
                        s = rc_dp4a(b.z, a[u].z, s);
                        acc_i = rc_dp4a(b.w, a[u].w, s);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int n = rg + 16 * i;
                const float dl = sdl[n][bk], dh = SPLIT ? sdh[n][bk] : 0.f, mn = MIN ? smn[n][bk] : 0.f;
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    const int m = tg + TG * u;
                    float v = dl * i2f_exact(il[u][i]);
                    if (SPLIT) {
                        v = fmaf(dh, i2f_exact(ih[u][i]), v);
                    }
                    acc[u][i] = fmaf(sxs[m][bk], v, acc[u][i]);
                    if (MIN) {
                        acc[u][i] = fmaf(-mn, sxm[m][bk], acc[u][i]);
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const int m = m0 + tg + TG * u;
        if (m >= t) {
            continue;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int n = n0 + rg + 16 * i;
            if (n < rows) {
                out[(size_t)m * rows + n] = acc[u][i];
            }
        }
    }
}
