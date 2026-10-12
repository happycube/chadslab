/* ---------- the products of large groups on the tensor cores (Qwen3.8) ----------
 * QWEN38_PLAN.md, phase 5. As k_gemm_q8: x as int8 with a scale for each 32
 * values (kq_quant_x: xs, and xsum = xs * the sum of the 32 int8 values),
 * and mma.sync m16n8k32 on the int8 values of 32 columns. Each format of w
 * gives, for each block of 32 values of a row, int8 values q, a scale d and
 * a term mn with w = d q + (the same mn for all 32 values). So
 *
 *     sum over the block of w x = xs d (the int32 sum of q x) + mn xsum.
 *
 *     Q8_R   the rows of the GPU (int8 values, then float16 scales): mn = 0
 *     Q8_0   blocks of 34 bytes: mn = 0
 *     Q5_1   w = d q + m (q of 5 bits): mn = m
 *     Q4_K   w = d sc q - dmin mb (sub-blocks of 32 of a block of 256)
 *
 * A tile is BM rows of x (tokens or pairs) by 128 rows of w, with 8 warps;
 * the steps of 128 columns come to shared memory by cp.async in NS
 * buffers. A step of a row of w holds its bytes for the 128 columns (the
 * head of the block of 256 and 64 bytes of values for Q4_K). */
#define KT_BN 128
#define KT_K 128
#define KT_RB 144                           /* the bytes of a row of w in a step (136 used) */
enum { KT_Q8R = 0, KT_Q80 = 1, KT_Q51 = 2, KT_Q4K = 3, KT_NV4 = 4, KT_NVX = 5, KT_Q4X = 6, KT_Q6K = 7 };
/* KT_Q6K (the RQ6_K experts, RQ6_MIX_PLAN.md): a step of 128 columns is half
 * a Q6_K block; a row of a step holds its 64 bytes of low 4 bits, 32 of high
 * 2 bits, 8 int8 scales, and d (106 bytes). Block bk of the step: the
 * values - 32 as int8, a scale d sc for each 16 (the sums of the two halves,
 * as NVFP4). The blocks of 210 bytes are only 2-aligned: plain loads. */
/* KT_Q4X (the KQ_Q4X experts of the 26B): 8 groups of the 4 blocks of the
 * step, no head. cols can end in a part of a step (704: 22 blocks). */
#define KT_GB4 (4 * 288)
/* KT_NVX: a step of a tile holds the 8 groups of its 128 rows, each the 16
 * bytes of its head and the 4 blocks of the step (KT_GB bytes). */
#define KT_GB (16 + 4 * KQ_NVX_BB)

/* mma.sync m16n8k16 on int8 values: the two halves of a block of 32 of NVFP4
 * (a scale for each 16 values) have sums of their own. */
__device__ __forceinline__ void mma16816s8(int *c, uint32_t a0, uint32_t a1, uint32_t b0)
{
#if RC_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
#else
    __trap();
#endif
}

__host__ __device__ __forceinline__ int kt_format(int type)
{
    return type == KQ_Q8_R ? KT_Q8R : type == KQ_Q8_0 ? KT_Q80 : type == KQ_Q5_1 ? KT_Q51 :
           type == KQ_Q4_K ? KT_Q4K : type == KQ_NV4 ? KT_NV4 : type == KQ_NVX ? KT_NVX :
           type == KQ_Q6_K ? KT_Q6K : -1;
}

/* x (n rows of cols values) to int8, with xs and xsum for each block of 32. */
__global__ void k_kq_quant_x(const float *x, int8_t *q, float *xs, float *xsum, size_t blocks)
{
    PDL_START();
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t bi = tid / 8;
    bool live = bi < blocks;
    float4 v = live ? *(const float4 *)(x + tid * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    quant8_block(v, live, bi, threadIdx.x % 8, q, xs, xsum);
}

/* Copy the bytes of row `row` of w for the step at column k0 to dst. */
template <int F>
__device__ __forceinline__ void kt_load_w(uint8_t *dst, const uint8_t *wrow, int cols, int k0)
{
    int lane = threadIdx.x % 32;       /* called by one warp for each row, lanes split the copies */
    if (F == KT_Q8R) {
        if (lane < 8) {
            cp_async16(dst + 16 * lane, wrow + k0 + 16 * lane);
        } else if (lane == 8) {
            cp_async8(dst + 128, wrow + cols + k0 / 16);
        }
    } else if (F == KT_Q80) {
        if (lane < 17) {
            cp_async8(dst + 8 * lane, wrow + (size_t)(k0 / 32) * 34 + 8 * lane);
        }
    } else if (F == KT_Q51) {
        if (lane < 6) {
            cp_async16(dst + 16 * lane, wrow + (size_t)(k0 / 32) * 24 + 16 * lane);
        }
    } else if (F == KT_Q6K) {
        /* 53 halfwords: ql (32), qh (16), the 8 scales (4), d (1) */
        const uint8_t *blk = wrow + (size_t)(k0 / 256) * 210;
        int h = (k0 / 128) % 2;
        for (int i = lane; i < 53; i += 32) {
            int o = i < 32 ? 64 * h + 2 * i : i < 48 ? 128 + 32 * h + 2 * (i - 32)
                  : i < 52 ? 192 + 8 * h + 2 * (i - 48) : 208;
            *(uint16_t *)(dst + 2 * i) = *(const uint16_t *)(blk + o);
        }
    } else if (F == KT_NV4) {
        /* the codes of the 4 blocks (64 bytes, 16-aligned), their 8 scales, the
         * scale of the matrix: 64 + 8 + 4 bytes of shared memory */
        if (lane < 4) {
            cp_async16(dst + 16 * lane, wrow + k0 / 2 + 16 * lane);
        } else if (lane == 4) {
            cp_async8(dst + 64, wrow + cols / 2 + k0 / 16);
        } else if (lane == 5) {
            cp_async4(dst + 72, wrow + cols / 2 + cols / 16);
        }
    } else {
        const uint8_t *blk = wrow + (size_t)(k0 / 256) * 144;
        if (lane == 0) {
            cp_async16(dst, blk);
        } else if (lane < 5) {
            cp_async16(dst + 16 * lane, blk + 16 + 64 * ((k0 / 128) % 2) + 16 * (lane - 1));
        }
    }
}

__device__ __forceinline__ uint32_t kt_hi4(uint32_t h)
{
    return ((h & 1) << 4) | ((h & 2) << 11) | ((h & 4) << 18) | ((h & 8) << 25);
}

/* The two registers of the fragment of B (values 4c..4c+3 and 16+4c.. of
 * block bk of the step) of a row in shared memory. */
template <int F>
__device__ __forceinline__ void kt_frag(const uint8_t *rp, int bk, int c, uint32_t *b)
{
    if (F == KT_Q8R) {
        b[0] = *(const uint32_t *)(rp + bk * 32 + 4 * c);
        b[1] = *(const uint32_t *)(rp + bk * 32 + 16 + 4 * c);
    } else if (F == KT_Q80) {
        const uint8_t *q = rp + bk * 34 + 2;
        b[0] = (uint32_t)*(const uint16_t *)(q + 4 * c) | ((uint32_t)*(const uint16_t *)(q + 4 * c + 2) << 16);
        b[1] = (uint32_t)*(const uint16_t *)(q + 16 + 4 * c) |
               ((uint32_t)*(const uint16_t *)(q + 16 + 4 * c + 2) << 16);
    } else if (F == KT_Q51) {
        const uint8_t *blk = rp + bk * 24;
        uint32_t qh = *(const uint32_t *)(blk + 4);
        uint32_t u = *(const uint32_t *)(blk + 8 + 4 * c);
        b[0] = (u & 0x0f0f0f0fu) | kt_hi4(qh >> (4 * c));
        b[1] = ((u >> 4) & 0x0f0f0f0fu) | kt_hi4(qh >> (16 + 4 * c));
    } else {
        const uint8_t *qs = rp + 16 + 32 * (bk / 2);
        int sh = 4 * (bk % 2);
        b[0] = (*(const uint32_t *)(qs + 4 * c) >> sh) & 0x0f0f0f0fu;
        b[1] = (*(const uint32_t *)(qs + 16 + 4 * c) >> sh) & 0x0f0f0f0fu;
    }
}

/* The scale d and the term mn of block bk of the step at k0 of a row. */
template <int F>
__device__ __forceinline__ void kt_scale(const uint8_t *rp, int bk, int k0, float *d, float *mn)
{
    if (F == KT_Q8R) {
        *d = __half2float(*(const __half *)(rp + 128 + 2 * bk));
        *mn = 0.f;
    } else if (F == KT_Q80) {
        *d = kq_half(rp + bk * 34);
        *mn = 0.f;
    } else if (F == KT_Q51) {
        *d = kq_half(rp + bk * 24);
        *mn = kq_half(rp + bk * 24 + 2);
    } else {
        int sc, m;
        kq_scale_min(rp + 4, 4 * ((k0 / 128) % 2) + bk, &sc, &m);
        *d = kq_half(rp) * (float)sc;
        *mn = -kq_half(rp + 2) * (float)m;
    }
}

/* A tile: out[m * ostride + n] for m < mcount (rows of x: xmap[m], or m)
 * and n0 <= n < n0 + 128 (n < rows) of w (rb bytes each). */
/* X2 (KT_Q4X only): the int16 form of x (k_quant_x2), xl its lo plane with
 * the rows of xq; a row of a step holds the hi values, then the lo values. */
template <int F, int BM, int NS, int X2 = 0>
__device__ void kt_tile(const uint8_t *w, size_t rb, int rows, int n0, int cols,
                        const int8_t *xq, const float *xs, const float *xsum, const int *xmap,
                        int mcount, float *out, size_t ostride, uint8_t *sm,
                        const int8_t *xl = NULL)
{
    const int MI = BM / 32;             /* the 16-row parts of a warp */
    constexpr int AW = KT_K * (1 + X2) + 16;
    int8_t (*as_)[BM][AW] = (int8_t (*)[BM][AW])sm;
    uint8_t (*bs)[KT_BN][KT_RB] = (uint8_t (*)[KT_BN][KT_RB])(sm + NS * BM * AW);
    float (*ss)[BM][8] = (float (*)[BM][8])(sm + NS * (BM * AW + KT_BN * KT_RB));
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * (BM / 2), wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    int nb = cols / 32, steps = F == KT_Q4X ? (cols + KT_K - 1) / KT_K : cols / KT_K;
    float acc[MI][4][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    auto xrow = [&](int m) {
        int mm = m < mcount ? m : mcount - 1;
        return xmap ? xmap[mm] : mm;
    };
    auto load = [&](int st, int b) {
        int k0 = st * KT_K;
        /* the blocks of the step (4, or fewer at the end of a KT_Q4X row) */
        int nbk = F == KT_Q4X ? min(4, nb - k0 / 32) : 4;
        for (int e = threadIdx.x; e < BM * KT_K / 16; e += blockDim.x) {
            int m = e / (KT_K / 16), kk = (e % (KT_K / 16)) * 16;
            if (F != KT_Q4X || kk < 32 * nbk) {
                cp_async16(&as_[b][m][kk], xq + (size_t)xrow(m) * cols + k0 + kk);
                if (X2) {
                    cp_async16(&as_[b][m][KT_K + kk], xl + (size_t)xrow(m) * cols + k0 + kk);
                }
            }
        }
        for (int e = threadIdx.x; e < BM * 4; e += blockDim.x) {
            int m = e / 4, bk = e % 4;
            if (F == KT_Q4X && bk >= nbk) {
                continue;
            }
            size_t o = (size_t)xrow(m) * nb + k0 / 32 + bk;
            cp_async4(&ss[b][m][bk], xs + o);
            cp_async4(&ss[b][m][4 + bk], xsum + o);
        }
        if (F == KT_Q4X) {
            uint8_t *gs = &bs[b][0][0];
            int nch = 18 * nbk;         /* 288 bytes a block */
            for (int e = threadIdx.x; e < 8 * nch; e += blockDim.x) {
                int gi = e / nch, ch = e % nch;
                int grp = min(n0 / 16 + gi, rows / 16 - 1);
                cp_async16(gs + gi * KT_GB4 + 16 * ch,
                           w + (size_t)grp * 16 * rb + (size_t)(k0 / 32) * 288 + 16 * ch);
            }
        } else if (F == KT_NVX) {
            /* 8 groups: the head, then the 4 blocks of the step */
            uint8_t *gs = &bs[b][0][0];
            for (int e = threadIdx.x; e < 8 * (KT_GB / 16); e += blockDim.x) {
                int gi = e / (KT_GB / 16), ch = e % (KT_GB / 16);
                int grp = min(n0 / 16 + gi, rows / 16 - 1);
                const uint8_t *src = w + (size_t)grp * 16 * rb +
                                     (ch == 0 ? 0 : 16 + (size_t)(k0 / 32) * KQ_NVX_BB + 16 * (ch - 1));
                cp_async16(gs + gi * KT_GB + 16 * ch, src);
            }
        } else {
            for (int n = warp; n < KT_BN; n += 8) {
                int row = min(n0 + n, rows - 1);
                kt_load_w<F>(bs[b][n], w + (size_t)row * rb, cols, k0);
            }
        }
        cp_async_commit();
    };
    for (int st = 0; st < NS - 1; ++st) {
        if (st < steps) {
            load(st, st);
        } else {
            cp_async_commit();
        }
    }
    for (int st = 0; st < steps; ++st) {
        int b = st % NS;
#if RC_SM80
        if (NS == 2) {
            asm volatile("cp.async.wait_group 0;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 1;\n" ::);
        }
#endif
        __syncthreads();
        if (st + NS - 1 < steps) {
            load(st + NS - 1, (st + NS - 1) % NS);
        } else {
            cp_async_commit();
        }
        int k0 = st * KT_K;
        if (F == KT_Q4X) {
            int nbk = min(4, nb - k0 / 32);
            #pragma unroll
            for (int bk = 0; bk < 4; ++bk) {
                if (bk >= nbk) {
                    break;
                }
                uint32_t bf[4][2];
                float dw[4][2];
                #pragma unroll
                for (int jg = 0; jg < 2; ++jg) {
                    const uint8_t *blk = &bs[b][0][0] + (wn / 16 + jg) * KT_GB4 + 288 * bk;
                    uint32_t qa = *(const uint32_t *)(blk + 32 * c + 4 * g);
                    uint32_t qb = *(const uint32_t *)(blk + 32 * (4 + c) + 4 * g);
                    bf[2 * jg][0] = __vsub4(qa & 0x0f0f0f0fu, 0x08080808u);
                    bf[2 * jg][1] = __vsub4(qb & 0x0f0f0f0fu, 0x08080808u);
                    bf[2 * jg + 1][0] = __vsub4((qa >> 4) & 0x0f0f0f0fu, 0x08080808u);
                    bf[2 * jg + 1][1] = __vsub4((qb >> 4) & 0x0f0f0f0fu, 0x08080808u);
                    const __half *d = (const __half *)(blk + 256);
                    #pragma unroll
                    for (int sb = 0; sb < 2; ++sb) {
                        dw[2 * jg + sb][0] = __half2float(d[8 * sb + 2 * c]);
                        dw[2 * jg + sb][1] = __half2float(d[8 * sb + 2 * c + 1]);
                    }
                }
                #pragma unroll
                for (int i = 0; i < MI; ++i) {
                    int r0 = wm + i * 16, kk = bk * 32;
                    uint32_t a[4], al[4];
                    a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                    a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                    a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                    a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                    if (X2) {
                        al[0] = *(const uint32_t *)&as_[b][r0 + g][KT_K + kk + 4 * c];
                        al[1] = *(const uint32_t *)&as_[b][r0 + g + 8][KT_K + kk + 4 * c];
                        al[2] = *(const uint32_t *)&as_[b][r0 + g][KT_K + kk + 4 * c + 16];
                        al[3] = *(const uint32_t *)&as_[b][r0 + g + 8][KT_K + kk + 4 * c + 16];
                    }
                    float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int ci[4] = {0, 0, 0, 0};
                        if (X2) {
                            mma16832_x2(ci, a, al, bf[j]);
                        } else {
                            mma16832(ci, a, bf[j]);
                        }
                        acc[i][j][0] += i2f_exact(ci[0]) * s0 * dw[j][0];
                        acc[i][j][1] += i2f_exact(ci[1]) * s0 * dw[j][1];
                        acc[i][j][2] += i2f_exact(ci[2]) * s1 * dw[j][0];
                        acc[i][j][3] += i2f_exact(ci[3]) * s1 * dw[j][1];
                    }
                }
            }
            continue;
        }
        if (F == KT_NV4 || F == KT_NVX || F == KT_Q6K) {
            #pragma unroll
            for (int bk = 0; bk < 4; ++bk) {
                uint32_t bf[4][2];
                float dl[4][2], dh[4][2];
                if (F == KT_Q6K) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint8_t *rp = bs[b][wn + j * 8 + g];
                        const uint8_t *ql = rp + 32 * (bk & 1);
                        int sh = 4 * (bk >> 1);
                        uint32_t l0 = (*(const uint32_t *)(ql + 4 * c) >> sh) & 0x0f0f0f0fu;
                        uint32_t l1 = (*(const uint32_t *)(ql + 16 + 4 * c) >> sh) & 0x0f0f0f0fu;
                        uint32_t h0 = (*(const uint32_t *)(rp + 64 + 4 * c) >> (2 * bk)) & 0x03030303u;
                        uint32_t h1 = (*(const uint32_t *)(rp + 80 + 4 * c) >> (2 * bk)) & 0x03030303u;
                        bf[j][0] = __vsub4(l0 | (h0 << 4), 0x20202020u);
                        bf[j][1] = __vsub4(l1 | (h1 << 4), 0x20202020u);
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const uint8_t *rs = bs[b][wn + j * 8 + 2 * c + h];
                            float d = kq_half(rs + 104);
                            dl[j][h] = d * (float)(int8_t)rs[96 + 2 * bk];
                            dh[j][h] = d * (float)(int8_t)rs[96 + 2 * bk + 1];
                        }
                    }
                } else if (F == KT_NVX) {
                    /* rows wn .. wn + 31: 2 groups; a group gives the
                     * fragments of rows g and g + 8 from 2 loads */
                    #pragma unroll
                    for (int jg = 0; jg < 2; ++jg) {
                        const uint8_t *gb = &bs[b][0][0] + (wn / 16 + jg) * KT_GB;
                        const uint8_t *blk = gb + 16 + KQ_NVX_BB * bk;
                        float gg = 128.f * *(const float *)gb;
                        uint32_t qa = *(const uint32_t *)(blk + 32 * c + 4 * g);
                        uint32_t qb = *(const uint32_t *)(blk + 32 * (4 + c) + 4 * g);
                        bf[2 * jg][0] = kt_e2m1x4(qa);
                        bf[2 * jg][1] = kt_e2m1x4(qb);
                        bf[2 * jg + 1][0] = kt_e2m1x4(qa >> 4);
                        bf[2 * jg + 1][1] = kt_e2m1x4(qb >> 4);
                        #pragma unroll
                        for (int sb = 0; sb < 2; ++sb) {
                            #pragma unroll
                            for (int h = 0; h < 2; ++h) {
                                int rr = 8 * sb + 2 * c + h;
                                dl[2 * jg + sb][h] = gg * kq_e4m3s(blk[256 + rr]);
                                dh[2 * jg + sb][h] = gg * kq_e4m3s(blk[272 + rr]);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        uint32_t q = *(const uint32_t *)(bs[b][wn + j * 8 + g] + 16 * bk + 4 * c);
                        bf[j][0] = kt_e2m1x4(q);
                        bf[j][1] = kt_e2m1x4(q >> 4);
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const uint8_t *rs = bs[b][wn + j * 8 + 2 * c + h];
                            float gg = 0.5f * *(const float *)(rs + 72);
                            dl[j][h] = gg * kq_e4m3(rs[64 + 2 * bk]);
                            dh[j][h] = gg * kq_e4m3(rs[64 + 2 * bk + 1]);
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < MI; ++i) {
                    int r0 = wm + i * 16, kk = bk * 32;
                    uint32_t a[4];
                    a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                    a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                    a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                    a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                    float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int cl[4] = {0, 0, 0, 0}, ch[4] = {0, 0, 0, 0};
                        mma16816s8(cl, a[0], a[1], bf[j][0]);
                        mma16816s8(ch, a[2], a[3], bf[j][1]);
                        acc[i][j][0] += s0 * (i2f_exact(cl[0]) * dl[j][0] + i2f_exact(ch[0]) * dh[j][0]);
                        acc[i][j][1] += s0 * (i2f_exact(cl[1]) * dl[j][1] + i2f_exact(ch[1]) * dh[j][1]);
                        acc[i][j][2] += s1 * (i2f_exact(cl[2]) * dl[j][0] + i2f_exact(ch[2]) * dh[j][0]);
                        acc[i][j][3] += s1 * (i2f_exact(cl[3]) * dl[j][1] + i2f_exact(ch[3]) * dh[j][1]);
                    }
                }
            }
            continue;
        }
        #pragma unroll
        for (int bk = 0; bk < 4; ++bk) {
            uint32_t bf[4][2];
            float dw[4][2], mw[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                kt_frag<F>(bs[b][wn + j * 8 + g], bk, c, bf[j]);
                kt_scale<F>(bs[b][wn + j * 8 + 2 * c], bk, k0, &dw[j][0], &mw[j][0]);
                kt_scale<F>(bs[b][wn + j * 8 + 2 * c + 1], bk, k0, &dw[j][1], &mw[j][1]);
            }
            #pragma unroll
            for (int i = 0; i < MI; ++i) {
                int r0 = wm + i * 16, kk = bk * 32;
                uint32_t a[4];
                a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                float u0 = ss[b][r0 + g][4 + bk], u1 = ss[b][r0 + g + 8][4 + bk];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int ci[4] = {0, 0, 0, 0};
                    mma16832(ci, a, bf[j]);
                    acc[i][j][0] += i2f_exact(ci[0]) * s0 * dw[j][0] + u0 * mw[j][0];
                    acc[i][j][1] += i2f_exact(ci[1]) * s0 * dw[j][1] + u0 * mw[j][1];
                    acc[i][j][2] += i2f_exact(ci[2]) * s1 * dw[j][0] + u1 * mw[j][0];
                    acc[i][j][3] += i2f_exact(ci[3]) * s1 * dw[j][1] + u1 * mw[j][1];
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < MI; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = n0 + wn + j * 8 + 2 * c;
            int m = wm + i * 16 + g;
            if (m < mcount) {
                if (n < rows) out[(size_t)m * ostride + n] = acc[i][j][0];
                if (n + 1 < rows) out[(size_t)m * ostride + n + 1] = acc[i][j][1];
            }
            if (m + 8 < mcount) {
                if (n < rows) out[(size_t)(m + 8) * ostride + n] = acc[i][j][2];
                if (n + 1 < rows) out[(size_t)(m + 8) * ostride + n + 1] = acc[i][j][3];
            }
        }
    }
}

#define KT_SMEM(BM, NS) ((size_t)(NS) * ((BM) * (KT_K + 16) + KT_BN * KT_RB + (BM) * 8 * 4))
#define KT_SMEM_X2(BM, NS) ((size_t)(NS) * ((BM) * (2 * KT_K + 16) + KT_BN * KT_RB + (BM) * 8 * 4))

/* GP_KQ_LINEAR of a large group on the tensor cores: blockIdx.x is the tile
 * of 128 tokens, blockIdx.y the tile of 128 rows. xq, xs, xsum: the scratch. */
template <int F>
__global__ void __launch_bounds__(256) k_kq_tc(const gp_rec *r, const int64_t *e, const int8_t *xq,
                                               const float *xs, const float *xsum)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t ktsm[];
    int rows = DI(6), cols = DI(7), t = DI(8), type = DI(5);
    int m0 = blockIdx.x * 128;
    kt_tile<F, 128, 2>(DP(const uint8_t, 4), kq_row_bytes(type, cols), rows, blockIdx.y * KT_BN, cols,
                       xq + (size_t)m0 * cols, xs + (size_t)m0 * (cols / 32),
                       xsum + (size_t)m0 * (cols / 32), NULL, min(128, t - m0),
                       DP(float, 9) + (size_t)m0 * rows, (size_t)rows, ktsm);
}

/* GP_KQ_LINEAR of a large group (t > MT_MAX). */
__global__ void __launch_bounds__(256) k_kq_gemm(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int type = DI(5), rows = DI(6), cols = DI(7), t = DI(8);
    int j0 = blockIdx.y * KG_B;
    kq_tile(type, DP(const uint8_t, 4), kq_row_bytes(type, cols), rows, blockIdx.x * KG_B, cols,
            DP(const float, 3) + (size_t)j0 * cols, (size_t)cols, NULL, min(KG_B, t - j0),
            DP(float, 9) + (size_t)j0 * rows, (size_t)rows);
}

/* GP_KQ_GROUP_MOE: the experts of a large group on the GPU.
 *
 *     h, val, idx, t, k, E, hidden, inner, tgate, tup, tdown, gtype, dtype,
 *     sgate, sup, sdown, stype, slog, work, act, act2, de, out, nreal
 *
 * tgate, tup, tdown give the device address of each expert (a hot slot, or
 * the buffer of the copies of GP_FETCH). The shared expert is expert E.
 * k_qmoe_sort sorts the pairs (token, expert) of the first nreal tokens by
 * expert, and makes the tiles of at most 64 pairs of one expert. Then the
 * gate and up rows, silu(gate) up, the down rows, and the sum. */
struct qmoe_w {
    int *cnt, *start, *ptok, *pof, *tiles;
};

__device__ __forceinline__ qmoe_w qmoe_work(const gp_rec *r, const int64_t *e)
{
    int E = DI(5), P = DI(3) * DI(4) + DI(3);
    qmoe_w w;
    w.cnt = DP(int, 18) + 8;
    w.start = w.cnt + E + 2;
    w.ptok = w.start + E + 2;
    w.pof = w.ptok + P;
    w.tiles = w.pof + P;
    return w;
}

/* k_qmoe_sort with a block of threads: the counts and the places of the
 * pairs with atomics in shared memory, the starts and the tiles on thread 0.
 * The pairs of an expert can come in another order than in k_qmoe_sort; a
 * pair has the same values in any place of its tiles. One thread took 5 ms
 * for a group of 2048 tokens of Qwen3.8 (20480 pairs, 512 experts). */
__global__ void __launch_bounds__(1024) k_qmoe_sort_par(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ int qs_sm[];
    int t = DI(3), k = DI(4), E = DI(5);
    int nr = DI(23);
    nr = nr > 0 && nr < t ? nr : t;
    const int *idx = DP(const int, 2);
    int *work = DP(int, 18);
    qmoe_w w = qmoe_work(r, e);
    int *scnt = qs_sm, *sfill = qs_sm + E + 2;
    for (int x = threadIdx.x; x <= E; x += blockDim.x) {
        scnt[x] = 0;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < nr * k; q += blockDim.x) {
        if (idx[q] >= 0) {
            atomicAdd(&scnt[idx[q]], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        scnt[E] = DP(const void, 13) != NULL ? nr : 0;     /* no shared expert: sgate null */
        int a = 0, nt = 0;
        for (int x = 0; x <= E; ++x) {
            w.start[x] = a;
            sfill[x] = a;
            for (int s2 = 0; s2 < scnt[x]; s2 += KG_B) {
                w.tiles[3 * nt] = x;
                w.tiles[3 * nt + 1] = a + s2;
                w.tiles[3 * nt + 2] = min(KG_B, scnt[x] - s2);
                ++nt;
            }
            a += scnt[x];
        }
        w.start[E + 1] = a;
        work[0] = nt;
        work[1] = a;
    }
    __syncthreads();
    for (int x = threadIdx.x; x <= E; x += blockDim.x) {
        w.cnt[x] = scnt[x];
    }
    for (int q = threadIdx.x; q < t * k; q += blockDim.x) {
        if (q / k >= nr || idx[q] < 0) {
            w.pof[q] = -1;
            continue;
        }
        int pos = atomicAdd(&sfill[idx[q]], 1);
        w.ptok[pos] = q / k;
        w.pof[q] = pos;
    }
    const int shared = DP(const void, 13) != NULL;
    for (int j = threadIdx.x; j < t; j += blockDim.x) {
        int pos = -1;
        if (j < nr && shared) {
            pos = w.start[E] + j;
            w.ptok[pos] = j;
        }
        w.pof[t * k + j] = pos;
    }
}

static int qmoe_sort_par_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_QMOE_SORT_PAR");
        on = !(v && v[0] == '0');
    }
    return on;
}

__global__ void k_qmoe_sort(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    if (threadIdx.x != 0) {
        return;
    }
    int t = DI(3), k = DI(4), E = DI(5);
    int nr = DI(23);
    nr = nr > 0 && nr < t ? nr : t;
    const int *idx = DP(const int, 2);
    int *work = DP(int, 18);
    qmoe_w w = qmoe_work(r, e);
    for (int x = 0; x <= E; ++x) {
        w.cnt[x] = 0;
    }
    for (int q = 0; q < nr * k; ++q) {
        if (idx[q] >= 0) {          /* -1: an expert of the CPU (GP_MOE_PLAN) */
            w.cnt[idx[q]]++;
        }
    }
    w.cnt[E] = DP(const void, 13) != NULL ? nr : 0;
    int a = 0, nt = 0;
    for (int x = 0; x <= E; ++x) {
        w.start[x] = a;
        for (int s = 0; s < w.cnt[x]; s += KG_B) {
            w.tiles[3 * nt] = x;
            w.tiles[3 * nt + 1] = a + s;
            w.tiles[3 * nt + 2] = min(KG_B, w.cnt[x] - s);
            ++nt;
        }
        a += w.cnt[x];
        w.cnt[x] = 0;
    }
    w.start[E + 1] = a;
    for (int q = 0; q < t * k; ++q) {
        if (q / k >= nr || idx[q] < 0) {
            w.pof[q] = -1;
            continue;
        }
        int x = idx[q], pos = w.start[x] + w.cnt[x]++;
        w.ptok[pos] = q / k;
        w.pof[q] = pos;
    }
    for (int j = 0; j < t; ++j) {
        int pos = -1;
        if (j < nr && DP(const void, 13) != NULL) {
            pos = w.start[E] + w.cnt[E]++;
            w.ptok[pos] = j;
        }
        w.pof[t * k + j] = pos;
    }
    work[0] = nt;
    work[1] = a;
}

__global__ void __launch_bounds__(256) k_qmoe_gu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    int row0 = blockIdx.x * KG_B, up = row0 >= inner;
    int type = x < E ? DI(11) : DI(16);
    const uint8_t *W = x < E ? (const uint8_t *)(intptr_t)DP(const int64_t, up ? 9 : 8)[x]
                             : DP(const uint8_t, up ? 14 : 13);
    kq_tile(type, W, kq_row_bytes(type, hidden), inner, row0 - (up ? inner : 0), hidden,
            DP(const float, 0), (size_t)hidden, w.ptok + p0, n,
            DP(float, 19) + (size_t)p0 * 2 * inner + (up ? inner : 0), (size_t)2 * inner);
}

/* rot: as k_kqh_act (the warps cover whole groups: the count is a
 * multiple of 32) */
__global__ void k_qmoe_act(const gp_rec *r, const int64_t *e, int rot)
{
    PDL_START();
    int inner = DI(7);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)DP(const int, 18)[1] * inner) {
        return;
    }
    int64_t p = i / inner, c = i % inner;
    const float *a = DP(const float, 19) + p * 2 * inner;
    float v = qw_silu(a[c]) * a[inner + c];
    if (rot) {
        v = tq6_rot_warp(v, (int)(c % 32), false);
    }
    DP(float, 20)[i] = v;
}

__global__ void __launch_bounds__(256) k_qmoe_dn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    int type = x < E ? DI(12) : DI(16);
    const uint8_t *W = x < E ? (const uint8_t *)(intptr_t)DP(const int64_t, 10)[x]
                             : DP(const uint8_t, 15);
    kq_tile(type, W, kq_row_bytes(type, inner), hidden, blockIdx.x * KG_B, inner,
            DP(const float, 20) + (size_t)p0 * inner, (size_t)inner, NULL, n,
            DP(float, 21) + (size_t)p0 * hidden, (size_t)hidden);
}

__global__ void k_qmoe_sum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int c = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    int t = DI(3), k = DI(4), hidden = DI(6);
    if (c >= hidden) {
        return;
    }
    const float *val = DP(const float, 1), *de = DP(const float, 21);
    float acc = 0.f;
    int ps = w.pof[t * k + tok];
    /* no shared expert (sgate null: the copied experts of a mixed group,
     * after its hot experts): only the routed pairs */
    const int noshared = DP(const void, 13) == NULL;
    if (ps >= 0 || noshared) {
        if (ps >= 0) {
            acc = de[(size_t)ps * hidden + c] / (1.f + expf(-DP(const float, 17)[tok]));
        }
        for (int s2 = 0; s2 < k; ++s2) {
            int p = w.pof[tok * k + s2];
            if (p >= 0) {           /* -1: an expert of the CPU (GP_MOE_PLAN) */
                acc += val[tok * k + s2] * de[(size_t)p * hidden + c];
            }
        }
    }
    DP(float, 22)[(size_t)tok * hidden + c] = acc;
}


/* GP_KQ_GROUP_MOE on the tensor cores: the tiles of k_qmoe_sort (at most
 * KG_B pairs of one expert), 128 rows of w for each block. xq, xs, xsum:
 * the int8 rows of h (gate and up) or of act2 (down). The shared expert
 * (x == E) has the Q8_R rows of the GPU. */
template <int KFG>
__global__ void __launch_bounds__(256) k_qmoe_gu_tc(const gp_rec *r, const int64_t *e,
                                                    const int8_t *xq, const float *xs,
                                                    const float *xsum, int skip_shared = 0)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t ktsm[];
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    if (x == E && skip_shared) {
        return;         /* the shared expert in bfloat16 (gg_gemm_bf16) */
    }
    int row0 = blockIdx.x * KT_BN, up = row0 >= inner;
    float *out = DP(float, 19) + (size_t)p0 * 2 * inner + (up ? inner : 0);
    if (x < E) {
        const uint8_t *W = (const uint8_t *)(intptr_t)DP(const int64_t, up ? 9 : 8)[x];
        kt_tile<KFG, 64, 3>(W, kq_row_bytes(DI(11), hidden), inner, row0 - (up ? inner : 0), hidden,
                           xq, xs, xsum, w.ptok + p0, n, out, (size_t)2 * inner, ktsm);
    } else {
        kt_tile<KT_Q8R, 64, 3>(DP(const uint8_t, up ? 14 : 13), kq_row_bytes(KQ_Q8_R, hidden), inner,
                               row0 - (up ? inner : 0), hidden, xq, xs, xsum, w.ptok + p0, n, out,
                               (size_t)2 * inner, ktsm);
    }
}

template <int KFD>
__global__ void __launch_bounds__(256) k_qmoe_dn_tc(const gp_rec *r, const int64_t *e,
                                                    const int8_t *xq, const float *xs,
                                                    const float *xsum, int skip_shared = 0)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t ktsm[];
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    if (x == E && skip_shared) {
        return;
    }
    float *out = DP(float, 21) + (size_t)p0 * hidden;
    const int8_t *xr = xq + (size_t)p0 * inner;
    const float *sr = xs + (size_t)p0 * (inner / 32), *ur = xsum + (size_t)p0 * (inner / 32);
    if (x < E) {
        const uint8_t *W = (const uint8_t *)(intptr_t)DP(const int64_t, 10)[x];
        kt_tile<KFD, 64, 3>(W, kq_row_bytes(DI(12), inner), hidden, blockIdx.x * KT_BN, inner, xr, sr,
                           ur, NULL, n, out, (size_t)hidden, ktsm);
    } else {
        kt_tile<KT_Q8R, 64, 3>(DP(const uint8_t, 15), kq_row_bytes(KQ_Q8_R, inner), hidden,
                               blockIdx.x * KT_BN, inner, xr, sr, ur, NULL, n, out, (size_t)hidden,
                               ktsm);
    }
}

/* GP_MOE_GPU with KQ_Q4X experts (operand 5 = 1): a tile of k_moe_tiles (at
 * most GM = 64 pairs of one expert) and 128 rows of its matrix, on the
 * tensor cores with int8 x (xq, xs, xsum of kt_quant). gather: row q of x
 * is row pair_tok[q] of h (gate and up); else x is (pairs, cols) (down). */
/* X2: the int16 form of x (k_quant_x2), xl its lo plane; 2 buffers. */
template <int X2>
__global__ void __launch_bounds__(256) k_moe_gemm_q4x(const gp_rec *r, const int64_t *e,
                                                      const int8_t *xq, const float *xs,
                                                      const float *xsum, int gather,
                                                      const int64_t *wtab, int rows, int cols,
                                                      float *out, const int8_t *xl)
{
    constexpr int NS = X2 ? 2 : 3;
    PDL_START();
    extern __shared__ __align__(16) uint8_t ktsm[];
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
        kt_tile<KT_Q4X, 64, NS, X2>(W, rb, rows, blockIdx.x * KT_BN, cols, xq, xs, xsum,
                                    DP(const int, 14) + q0, q1 - q0, out + (size_t)q0 * rows,
                                    (size_t)rows, ktsm, xl);
    } else {
        kt_tile<KT_Q4X, 64, NS, X2>(W, rb, rows, blockIdx.x * KT_BN, cols,
                                    xq + (size_t)q0 * cols, xs + (size_t)q0 * nbx,
                                    xsum + (size_t)q0 * nbx, NULL, q1 - q0,
                                    out + (size_t)q0 * rows, (size_t)rows, ktsm,
                                    X2 ? xl + (size_t)q0 * cols : NULL);
    }
}
