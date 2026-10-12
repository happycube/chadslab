/* KQ_Q4X: -8 times the sum of each 32 int8 values of n rows of x (the term of
 * the codes q = w + 8 of kq_q4x_rows). A row of xn has the stride cols / 16,
 * as that of kq_nvx_xsum (the scratch of kq_moe_body serves both). */
static void kq_q4x_xsum(const int8_t *xq, int n, int cols, int32_t *xn)
{
    for (int j = 0; j < n; ++j) {
        for (int b = 0; b < cols / 32; ++b) {
            const int8_t *p = xq + (size_t)j * cols + 32 * b;
            int32_t sum = 0;
            for (int u = 0; u < 32; ++u) {
                sum += p[u];
            }
            xn[(size_t)j * (cols / 16) + b] = -8 * sum;
        }
    }
}

/* The sums of kq_rows_n for a type of groups of 16 rows (xn). */
static inline void kq_x16_xsum(int type, const int8_t *xq, int cols, int32_t *xn)
{
    if (type == KQ_Q4X) {
        kq_q4x_xsum(xq, 1, cols, xn);
    } else {
        kq_nvx_xsum(xq, 1, cols, xn);
    }
}

#if defined(__AVX512VNNI__)
/* The 16 rows of the KQ_Q4X group at wg on n tokens: out[j * ostride + r]
 * gets row r, token j. xn: kq_q4x_xsum of the tokens. As kq_nvx_rows, with
 * the codes as they are (q = w + 8, 0 to 15: the unsigned operand of
 * vpdpbusd), the sum of a block from -8 sum(x), and one scale of each row
 * for each block. A token alone and in a group has the same operations. */
static void kq_q4x_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32;
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int j = 0; j < n; j += KQ_Q4X_TB) {
        int nt = n - j < KQ_Q4X_TB ? n - j : KQ_Q4X_TB;
        __m512 f[KQ_Q4X_TB];
        for (int tt = 0; tt < KQ_Q4X_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
                __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                W[k] = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            }
            __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 256)));
            /* Two tokens at a time, each with two sums (the even and the odd
             * steps): four chains of 4 vpdpbusd in place of one chain of 8
             * for each token. A chain of 8 (5 cycles each) left the ports
             * half idle with the scheduler full (resource_stalls 35 per
             * cent of the cycles). The sums are int32, so the order of the
             * adds does not change them: the same bits. */
            int tt = 0;
            for (; tt + 2 <= nt; tt += 2) {
                const int8_t *xb0 = xq + (size_t)(j + tt) * cols + 32 * b;
                const int8_t *xb1 = xb0 + cols;
                __m512i a0 = _mm512_set1_epi32(xn[(size_t)(j + tt) * (cols / 16) + b]);
                __m512i a1 = _mm512_setzero_si512();
                __m512i c0 = _mm512_set1_epi32(xn[(size_t)(j + tt + 1) * (cols / 16) + b]);
                __m512i c1 = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    int32_t x0, x1, y0, y1;
                    memcpy(&x0, xb0 + 4 * k, 4);
                    memcpy(&x1, xb0 + 4 * k + 4, 4);
                    memcpy(&y0, xb1 + 4 * k, 4);
                    memcpy(&y1, xb1 + 4 * k + 4, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    c0 = _mm512_dpbusd_epi32(c0, W[k], _mm512_set1_epi32(y0));
                    a1 = _mm512_dpbusd_epi32(a1, W[k + 1], _mm512_set1_epi32(x1));
                    c1 = _mm512_dpbusd_epi32(c1, W[k + 1], _mm512_set1_epi32(y1));
                }
                f[tt] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(a0, a1)),
                                        _mm512_mul_ps(d, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b])),
                                        f[tt]);
                f[tt + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(c0, c1)),
                                            _mm512_mul_ps(d, _mm512_set1_ps(xs[(size_t)(j + tt + 1) * nb + b])),
                                            f[tt + 1]);
            }
            for (; tt < nt; ++tt) {
                const int8_t *xb = xq + (size_t)(j + tt) * cols + 32 * b;
                __m512i a0 = _mm512_set1_epi32(xn[(size_t)(j + tt) * (cols / 16) + b]);
                __m512i a1 = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    int32_t x0, x1;
                    memcpy(&x0, xb + 4 * k, 4);
                    memcpy(&x1, xb + 4 * k + 4, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    a1 = _mm512_dpbusd_epi32(a1, W[k + 1], _mm512_set1_epi32(x1));
                }
                f[tt] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_add_epi32(a0, a1)),
                                        _mm512_mul_ps(d, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b])),
                                        f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, f[tt]);
        }
    }
}

/* The prompt form (kq_q4x_gemm): the codes of a group of 16 rows unpacked
 * once (u: nb x 8 steps x 64 bytes, the unsigned operand of vpdpbusd as
 * kq_q4x_rows makes it; dsc: nb x 16 float scales), then tiles of G groups
 * and T tokens with all the sums in registers. kq_q4x_rows kept the float
 * sums of its 16 tokens in an array that the compiler put on the stack (a
 * load and a store for each token and block), and unpacked the codes once
 * for each 16 tokens. Each value has the operations of kq_q4x_rows: the
 * int32 sum of a block (exact in any order), then the same cvt, mul, and
 * fma, block after block: the same bits. */
static void kq_q4x_unpack(const uint8_t *wg, int nb, uint8_t *u, float *dsc)
{
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            _mm512_store_si512((__m512i *)(u + ((size_t)b * 8 + k) * 64),
                               _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
        }
        _mm512_store_ps(dsc + (size_t)b * 16,
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 256))));
    }
}

/* The tokens of a tile: 2 groups and 6 tokens are 12 chains of vpdpbusd (5
 * cycles each, 2 ports), 24 sums in registers. */
#ifndef KQ_Q4X_TT
#define KQ_Q4X_TT 6
#endif

/* G groups (1 or 2) of 16 rows on T tokens (1 to KQ_Q4X_TT): out[t * ostride + r]. */
static inline __attribute__((always_inline)) void kq_q4x_tile(
    const uint8_t *u0, const uint8_t *u1, const float *d0, const float *d1, int nb, int cols,
    const int8_t *xq, const float *xs, const int32_t *xn, float *out, size_t ostride,
    const int G, const int T)
{
    __m512 f[2][KQ_Q4X_TT];
    for (int g = 0; g < G; ++g) {
        for (int t = 0; t < T; ++t) {
            f[g][t] = _mm512_setzero_ps();
        }
    }
    for (int b = 0; b < nb; ++b) {
        __m512i a[2][KQ_Q4X_TT];
        for (int t = 0; t < T; ++t) {
            __m512i c = _mm512_set1_epi32(xn[(size_t)t * (cols / 16) + b]);
            for (int g = 0; g < G; ++g) {
                a[g][t] = c;
            }
        }
        for (int k = 0; k < 8; ++k) {
            __m512i w0 = _mm512_load_si512((const __m512i *)(u0 + ((size_t)b * 8 + k) * 64));
            __m512i w1 = G > 1 ? _mm512_load_si512((const __m512i *)(u1 + ((size_t)b * 8 + k) * 64))
                               : w0;
            for (int t = 0; t < T; ++t) {
                int32_t x4;
                memcpy(&x4, xq + (size_t)t * cols + 32 * b + 4 * k, 4);
                __m512i xv = _mm512_set1_epi32(x4);
                a[0][t] = _mm512_dpbusd_epi32(a[0][t], w0, xv);
                if (G > 1) {
                    a[1][t] = _mm512_dpbusd_epi32(a[1][t], w1, xv);
                }
            }
        }
        __m512 e0 = _mm512_load_ps(d0 + (size_t)b * 16);
        __m512 e1 = G > 1 ? _mm512_load_ps(d1 + (size_t)b * 16) : e0;
        for (int t = 0; t < T; ++t) {
            __m512 sx = _mm512_set1_ps(xs[(size_t)t * nb + b]);
            f[0][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[0][t]), _mm512_mul_ps(e0, sx), f[0][t]);
            if (G > 1) {
                f[1][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[1][t]), _mm512_mul_ps(e1, sx),
                                          f[1][t]);
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        _mm512_storeu_ps(out + (size_t)t * ostride, f[0][t]);
        if (G > 1) {
            _mm512_storeu_ps(out + (size_t)t * ostride + 16, f[1][t]);
        }
    }
}

/* The scratch of the unpacked codes of a thread (2 groups). */
static __thread uint8_t *kq_q4x_u;
static __thread float *kq_q4x_d;
static __thread size_t kq_q4x_un;

/* KQ_Q4X on t tokens (4 or more), inside a parallel region: blocks of
 * tokens (their int8 x in L2 with the codes of 2 groups), the pairs of
 * groups over the threads, tiles of 2 groups and 4 tokens. */
static void kq_q4x_gemm(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int t, float *out)
{
    const int nb = cols / 32, groups = rows / 16;
    const size_t gb = 16 * kq_row_bytes(KQ_Q4X, cols);
    const size_t need = (size_t)nb * 8 * 64;
    if (kq_q4x_un < need) {
        free(kq_q4x_u);
        free(kq_q4x_d);
        kq_q4x_u = (uint8_t *)aligned_alloc(64, 2 * need);
        kq_q4x_d = (float *)aligned_alloc(64, 2 * (size_t)nb * 16 * sizeof(float));
        kq_q4x_un = need;
    }
    uint8_t *u0 = kq_q4x_u, *u1 = kq_q4x_u + need;
    float *d0 = kq_q4x_d, *d1 = kq_q4x_d + (size_t)nb * 16;
    /* the int8 x of a block of tokens about 512 KB */
    int tb = (512 * 1024 / cols) / KQ_Q4X_TT * KQ_Q4X_TT;
    tb = tb < 2 * KQ_Q4X_TT ? 2 * KQ_Q4X_TT : (tb > 126 ? 126 : tb);
    /* The items: a block of tokens and a pair of groups, the pairs of a
     * block next to each other, so a thread takes adjacent pairs of one
     * block (its x in L2). A matrix of few rows (the down map: 240 groups,
     * 120 pairs) then still gives each thread the same work. */
    /* blocks of the same size (a last block of a few tokens left the
     * threads of its items idle) */
    const int blocks = (t + tb - 1) / tb;
    tb = ((t + blocks - 1) / blocks + KQ_Q4X_TT - 1) / KQ_Q4X_TT * KQ_Q4X_TT;
    const int pairs = (groups + 1) / 2;
    #pragma omp for schedule(static)
    for (int it = 0; it < blocks * pairs; ++it) {
        const int j0 = (it / pairs) * tb, gp = it % pairs;
        if (j0 >= t) {
            continue;
        }
        const int nt = t - j0 < tb ? t - j0 : tb;
        {
            const int g = 2 * gp, two = g + 1 < groups;
            kq_q4x_unpack(w + (size_t)g * gb, nb, u0, d0);
            if (two) {
                kq_q4x_unpack(w + (size_t)(g + 1) * gb, nb, u1, d1);
            }
            for (int jj = 0; jj < nt; jj += KQ_Q4X_TT) {
                const int j = j0 + jj, T = nt - jj < KQ_Q4X_TT ? nt - jj : KQ_Q4X_TT;
                const int8_t *xj = xq + (size_t)j * cols;
                const float *sj = xs + (size_t)j * nb;
                const int32_t *nj = xn + (size_t)j * (cols / 16);
                float *oj = out + (size_t)j * rows + 16 * g;
#define KQ_Q4X_TILE(GG, TT) kq_q4x_tile(u0, u1, d0, d1, nb, cols, xj, sj, nj, oj, (size_t)rows, GG, TT)
                if (two) {
                    switch (T) {
                    case 6: KQ_Q4X_TILE(2, 6); break;
                    case 5: KQ_Q4X_TILE(2, 5); break;
                    case 4: KQ_Q4X_TILE(2, 4); break;
                    case 3: KQ_Q4X_TILE(2, 3); break;
                    case 2: KQ_Q4X_TILE(2, 2); break;
                    default: KQ_Q4X_TILE(2, 1); break;
                    }
                } else {
                    switch (T) {
                    case 6: KQ_Q4X_TILE(1, 6); break;
                    case 5: KQ_Q4X_TILE(1, 5); break;
                    case 4: KQ_Q4X_TILE(1, 4); break;
                    case 3: KQ_Q4X_TILE(1, 3); break;
                    case 2: KQ_Q4X_TILE(1, 2); break;
                    default: KQ_Q4X_TILE(1, 1); break;
                    }
                }
#undef KQ_Q4X_TILE
            }
        }
    }
}
#define KQ_Q4X_GEMM 1

/* ---- int16 x on KQ_Q4X (the prompt with RECLANK_INT4_Q8=16) ----
 *
 * The int8 x of a prompt (rc_kq_quant_x) rounds each value to 1/127 of the
 * largest of its 32; on the 26B a change of 1e-6 in a router weight then
 * moved 20 per cent of the top tokens of a prompt of 256, and the NLL of a
 * text grew by up to 0.37 against float32. int16 x (gemma_quant_group32_i16:
 * 1/32767 of the largest of 32, to the nearest even) gives the NLL of
 * float32. vpdpwssd multiplies 2 int16 of a lane by 2 int16 weights: 32
 * products for each instruction, half of vpdpbusd. The weights are the
 * codes less 8 as int16, so a block needs no correction term: at most
 * 32 * 32767 * 8 < 2^31.
 *
 * kq_q4x_unpack16 gives each block of 32 columns of a group of 16 rows 16
 * vectors: vector j holds, in lane r, the weights of row r at columns 2j and
 * 2j + 1 (the lanes of kq_q4x_rows: rows 0 to 7 from the low 4 bits, 8 to
 * 15 from the high 4 bits). */
static void kq_q4x_unpack16(const uint8_t *wg, int nb, int16_t *u, float *dsc)
{
    const __m256i m4 = _mm256_set1_epi8(15);
    const __m512i eight = _mm512_set1_epi16(8);
    const __m512i ia = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    const __m512i ib = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m512i lo = _mm512_cvtepu8_epi16(_mm256_and_si256(q, m4));
            __m512i hi = _mm512_cvtepu8_epi16(_mm256_and_si256(_mm256_srli_epi16(q, 4), m4));
            /* dword 2L of lo holds columns 4k, 4k + 1 of row L, dword 2L + 1
             * columns 4k + 2, 4k + 3; hi the same for row L + 8 */
            _mm512_store_si512((__m512i *)(u + ((size_t)b * 16 + 2 * k) * 32),
                               _mm512_sub_epi16(_mm512_permutex2var_epi32(lo, ia, hi), eight));
            _mm512_store_si512((__m512i *)(u + ((size_t)b * 16 + 2 * k + 1) * 32),
                               _mm512_sub_epi16(_mm512_permutex2var_epi32(lo, ib, hi), eight));
        }
        _mm512_store_ps(dsc + (size_t)b * 16,
                        _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 256))));
    }
}

/* G groups (1 or 2) of 16 rows on T tokens (G 2: T to 4; G 1: T to 8), nb
 * blocks of 32 columns (a chunk of the row): the int16 x of token t at
 * xq + t * xstride, its scales at xs + t * sstride: out[t * ostride + r].
 * As kq_q4x_tile: the int32 sum of a block, then the scale of the block of
 * the row times the scale of x, block after block. ACC 1 goes on from the
 * sums in out (the chunk before): the same additions in the same order. */
#define KQ16_TMAX 8
static inline __attribute__((always_inline)) void kq_q4x_tile16(
    const int16_t *u0, const int16_t *u1, const float *d0, const float *d1, int nb,
    size_t xstride, size_t sstride, const int16_t *xq, const float *xs, float *out,
    size_t ostride, const int G, const int T, const int ACC)
{
    __m512 f[2][KQ16_TMAX];
    for (int g = 0; g < G; ++g) {
        for (int t = 0; t < T; ++t) {
            f[g][t] = ACC ? _mm512_loadu_ps(out + (size_t)t * ostride + 16 * g) : _mm512_setzero_ps();
        }
    }
    for (int b = 0; b < nb; ++b) {
        __m512i a[2][KQ16_TMAX];
        for (int g = 0; g < G; ++g) {
            for (int t = 0; t < T; ++t) {
                a[g][t] = _mm512_setzero_si512();
            }
        }
        for (int j = 0; j < 16; ++j) {
            __m512i w0 = _mm512_load_si512((const __m512i *)(u0 + ((size_t)b * 16 + j) * 32));
            __m512i w1 = G > 1 ? _mm512_load_si512((const __m512i *)(u1 + ((size_t)b * 16 + j) * 32))
                               : w0;
            for (int t = 0; t < T; ++t) {
                int32_t x2;
                memcpy(&x2, xq + (size_t)t * xstride + 32 * b + 2 * j, 4);
                __m512i xv = _mm512_set1_epi32(x2);
                a[0][t] = _mm512_dpwssd_epi32(a[0][t], w0, xv);
                if (G > 1) {
                    a[1][t] = _mm512_dpwssd_epi32(a[1][t], w1, xv);
                }
            }
        }
        __m512 e0 = _mm512_load_ps(d0 + (size_t)b * 16);
        __m512 e1 = G > 1 ? _mm512_load_ps(d1 + (size_t)b * 16) : e0;
        for (int t = 0; t < T; ++t) {
            __m512 sx = _mm512_set1_ps(xs[(size_t)t * sstride + b]);
            f[0][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[0][t]), _mm512_mul_ps(e0, sx), f[0][t]);
            if (G > 1) {
                f[1][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a[1][t]), _mm512_mul_ps(e1, sx),
                                          f[1][t]);
            }
        }
    }
    for (int t = 0; t < T; ++t) {
        _mm512_storeu_ps(out + (size_t)t * ostride, f[0][t]);
        if (G > 1) {
            _mm512_storeu_ps(out + (size_t)t * ostride + 16, f[1][t]);
        }
    }
}

/* The scratch of the unpacked codes of a thread (2 groups, int16). */
static __thread int16_t *kq16_u;
static __thread float *kq16_d;
static __thread size_t kq16_un;

static void kq16_scratch(int nb)
{
    const size_t need = (size_t)nb * 16 * 32;
    if (kq16_un < need) {
        free(kq16_u);
        free(kq16_d);
        kq16_u = (int16_t *)aligned_alloc(64, 2 * need * sizeof(int16_t));
        kq16_d = (float *)aligned_alloc(64, 2 * (size_t)nb * 16 * sizeof(float));
        kq16_un = need;
    }
}

/* One group of 16 rows on n tokens (an expert of kq_moe_body): the codes
 * unpacked once, then tiles of 8 tokens. */
/* The blocks of a chunk of columns: the unpacked codes of a group (1 KB a
 * block) of a chunk stay in L2 with x (a row of 15360 columns gave 480 KB a
 * group, and the tiles read them from L3). */
#define KQ16_CHUNK 128

static void kq_q4x_rows16(const uint8_t *wg, int cols, const int16_t *xq, const float *xs, int n,
                          float *out, size_t ostride)
{
    const int nb = cols / 32;
    kq16_scratch(nb < KQ16_CHUNK ? nb : KQ16_CHUNK);
    int16_t *u0 = kq16_u;
    float *d0 = kq16_d;
    for (int c0 = 0; c0 < nb; c0 += KQ16_CHUNK) {
        const int nc = nb - c0 < KQ16_CHUNK ? nb - c0 : KQ16_CHUNK;
        kq_q4x_unpack16(wg + (size_t)c0 * KQ_Q4X_BB, nc, u0, d0);
        for (int j = 0; j < n; j += KQ16_TMAX) {
            const int T = n - j < KQ16_TMAX ? n - j : KQ16_TMAX;
            const int16_t *xj = xq + (size_t)j * cols + 32 * (size_t)c0;
            const float *sj = xs + (size_t)j * nb + c0;
            float *oj = out + (size_t)j * ostride;
#define KQ16_TILE1(TT) do { if (c0) kq_q4x_tile16(u0, u0, d0, d0, nc, cols, nb, xj, sj, oj, ostride, 1, TT, 1); \
                            else kq_q4x_tile16(u0, u0, d0, d0, nc, cols, nb, xj, sj, oj, ostride, 1, TT, 0); } while (0)
            switch (T) {
            case 8: KQ16_TILE1(8); break;
            case 7: KQ16_TILE1(7); break;
            case 6: KQ16_TILE1(6); break;
            case 5: KQ16_TILE1(5); break;
            case 4: KQ16_TILE1(4); break;
            case 3: KQ16_TILE1(3); break;
            case 2: KQ16_TILE1(2); break;
            default: KQ16_TILE1(1); break;
            }
#undef KQ16_TILE1
        }
    }
}

/* KQ_Q4X on t tokens of int16 x, inside a parallel region: as kq_q4x_gemm,
 * with tiles of 2 groups and 4 tokens. */
static void kq_q4x_gemm16(const uint8_t *w, int rows, int cols, const int16_t *xq, const float *xs,
                          int t, float *out)
{
    const int nb = cols / 32, groups = rows / 16;
    const size_t gb = 16 * kq_row_bytes(KQ_Q4X, cols);
    const int cb = nb < KQ16_CHUNK ? nb : KQ16_CHUNK;
    kq16_scratch(cb);
    int16_t *u0 = kq16_u, *u1 = kq16_u + kq16_un;
    float *d0 = kq16_d, *d1 = kq16_d + (size_t)cb * 16;
    /* the int16 x of a block of tokens about 512 KB, blocks of the same size */
    int tb = (256 * 1024 / cols) & ~3;
    tb = tb < 8 ? 8 : (tb > 128 ? 128 : tb);
    const int blocks = (t + tb - 1) / tb;
    tb = ((t + blocks - 1) / blocks + 3) & ~3;
    const int pairs = (groups + 1) / 2;
    #pragma omp for schedule(static)
    for (int it = 0; it < blocks * pairs; ++it) {
        const int j0 = (it / pairs) * tb, gp = it % pairs;
        if (j0 >= t) {
            continue;
        }
        const int nt = t - j0 < tb ? t - j0 : tb;
        const int g = 2 * gp, two = g + 1 < groups;
        for (int c0 = 0; c0 < nb; c0 += cb) {
        const int nc = nb - c0 < cb ? nb - c0 : cb;
        kq_q4x_unpack16(w + (size_t)g * gb + (size_t)c0 * KQ_Q4X_BB, nc, u0, d0);
        if (two) {
            kq_q4x_unpack16(w + (size_t)(g + 1) * gb + (size_t)c0 * KQ_Q4X_BB, nc, u1, d1);
        }
        for (int jj = 0; jj < nt; jj += 4) {
            const int j = j0 + jj, T = nt - jj < 4 ? nt - jj : 4;
            const int16_t *xj = xq + (size_t)j * cols + 32 * (size_t)c0;
            const float *sj = xs + (size_t)j * nb + c0;
            float *oj = out + (size_t)j * rows + 16 * g;
#define KQ16_TILE(GG, TT) do { if (c0) kq_q4x_tile16(u0, u1, d0, d1, nc, cols, nb, xj, sj, oj, (size_t)rows, GG, TT, 1); \
                               else kq_q4x_tile16(u0, u1, d0, d1, nc, cols, nb, xj, sj, oj, (size_t)rows, GG, TT, 0); } while (0)
            if (two) {
                switch (T) {
                case 4: KQ16_TILE(2, 4); break;
                case 3: KQ16_TILE(2, 3); break;
                case 2: KQ16_TILE(2, 2); break;
                default: KQ16_TILE(2, 1); break;
                }
            } else {
                switch (T) {
                case 4: KQ16_TILE(1, 4); break;
                case 3: KQ16_TILE(1, 3); break;
                case 2: KQ16_TILE(1, 2); break;
                default: KQ16_TILE(1, 1); break;
                }
            }
#undef KQ16_TILE
        }
        }
    }
}
#define KQ_Q4X_Q16 1
#elif defined(__AVX2__)
/* AVX2 (no VNNI): a 32-bit lane is a row, the low 4 bits of a step are rows
 * 0 to 7 and the high 4 bits rows 8 to 15. vpmaddubsw of the codes (0 to
 * 15) and 4 int8 x (the same in each lane) gives 2 sums of 2 in each lane;
 * the 8 steps of a block add as int16 (at most 8 * 2 * 15 * 127 = 30480),
 * then one vpmaddwd adds the pairs. A token alone and in a group has the
 * same operations. */
/* kq_q4x_rows for NT tokens (a constant 1 to 4): the codes of a step go to
 * the low and the high rows once for the NT tokens. */
static inline __attribute__((always_inline)) void kq_q4x_rows_n(
    const uint8_t *wg, int cols, const int8_t *xq, const float *xs, const int32_t *xn,
    const int NT, float *out, size_t ostride)
{
    int nb = cols / 32;
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 flo[4], fhi[4];
    for (int t = 0; t < NT; ++t) {
        flo[t] = _mm256_setzero_ps();
        fhi[t] = _mm256_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        __m256i slo[4], shi[4];
        for (int t = 0; t < NT; ++t) {
            slo[t] = _mm256_setzero_si256();
            shi[t] = _mm256_setzero_si256();
        }
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            for (int t = 0; t < NT; ++t) {
                int32_t x4;
                memcpy(&x4, xq + (size_t)t * cols + 32 * b + 4 * k, 4);
                __m256i xv = _mm256_set1_epi32(x4);
#if defined(RC_DPBUSD256)
                slo[t] = RC_DPBUSD256(slo[t], lo, xv);      /* int32 lanes */
                shi[t] = RC_DPBUSD256(shi[t], hi, xv);
#else
                slo[t] = _mm256_add_epi16(slo[t], _mm256_maddubs_epi16(lo, xv));
                shi[t] = _mm256_add_epi16(shi[t], _mm256_maddubs_epi16(hi, xv));
#endif
            }
        }
        __m256 dlo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 256)));
        __m256 dhi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 272)));
        for (int t = 0; t < NT; ++t) {
            __m256i nv = _mm256_set1_epi32(xn[(size_t)t * (cols / 16) + b]);
#if defined(RC_DPBUSD256)
            __m256i alo = _mm256_add_epi32(slo[t], nv);
            __m256i ahi = _mm256_add_epi32(shi[t], nv);
#else
            __m256i alo = _mm256_add_epi32(_mm256_madd_epi16(slo[t], ones), nv);
            __m256i ahi = _mm256_add_epi32(_mm256_madd_epi16(shi[t], ones), nv);
#endif
            __m256 sx = _mm256_set1_ps(xs[(size_t)t * nb + b]);
            flo[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(alo), _mm256_mul_ps(dlo, sx), flo[t]);
            fhi[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ahi), _mm256_mul_ps(dhi, sx), fhi[t]);
        }
    }
    for (int t = 0; t < NT; ++t) {
        _mm256_storeu_ps(out + (size_t)t * ostride, flo[t]);
        _mm256_storeu_ps(out + (size_t)t * ostride + 8, fhi[t]);
    }
}

static void kq_q4x_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32;
    /* Up to 4 tokens for each unpack of the codes (the sums of each token
     * do not change). The 26B prompt of 512 tokens, 6 threads: 61.3 tok/s
     * with 2, 64.9 with 4. */
    for (int j = 0; j < n;) {
        const int8_t *qj = xq + (size_t)j * cols;
        const float *sj = xs + (size_t)j * nb;
        const int32_t *nj = xn + (size_t)j * (cols / 16);
        int left = n - j;
        if (left >= 4) {
            kq_q4x_rows_n(wg, cols, qj, sj, nj, 4, out + (size_t)j * ostride, ostride);
            j += 4;
        } else if (left >= 3) {
            kq_q4x_rows_n(wg, cols, qj, sj, nj, 3, out + (size_t)j * ostride, ostride);
            j += 3;
        } else if (left >= 2) {
            kq_q4x_rows_n(wg, cols, qj, sj, nj, 2, out + (size_t)j * ostride, ostride);
            j += 2;
        } else {
            kq_q4x_rows_n(wg, cols, qj, sj, nj, 1, out + (size_t)j * ostride, ostride);
            j += 1;
        }
    }
}

/* ---- int16 x on KQ_Q4X, AVX2 (the prompt with RECLANK_INT4_Q8=16) ----
 *
 * The int8 x of the prompt cost the AVX2 build most of its distance from
 * float32 (the 26B, 64 tokens and 40 steps against the NumPy decode: KL
 * 0.22 with int8 x, 0.064 with float32 prompt products). As the VNNI form
 * above: the codes less 8 as int16, unpacked once for a chunk of a group,
 * then vpmaddwd (8 lanes of 2 products) of a pair of columns of the 16 rows
 * by the 2 int16 x of the pair, added as int32 (a block: at most 32 * 32767
 * * 8 < 2^31). Twice the instructions of the int8 form (vpmaddubsw, 4
 * columns a lane). The arithmetic of a token does not depend on the tile:
 * a token alone and in a group give the same bits.
 *
 * kq_q4x_unpack16: each block of 32 columns of a group of 16 rows gives 32
 * vectors: vector 2 j + h holds, in lane r, the weights of row 8 h + r at
 * columns 2 j and 2 j + 1. */
static void kq_q4x_unpack16(const uint8_t *wg, int nb, int16_t *u, float *dsc)
{
    const __m256i m4 = _mm256_set1_epi8(15), eight = _mm256_set1_epi16(8);
    /* bytes 0, 1 (columns 4 k, 4 k + 1) and 2, 3 of each 32-bit lane to int16 */
    const __m256i sa = _mm256_setr_epi8(0, -1, 1, -1, 4, -1, 5, -1, 8, -1, 9, -1, 12, -1, 13, -1,
                                        0, -1, 1, -1, 4, -1, 5, -1, 8, -1, 9, -1, 12, -1, 13, -1);
    const __m256i sb = _mm256_setr_epi8(2, -1, 3, -1, 6, -1, 7, -1, 10, -1, 11, -1, 14, -1, 15, -1,
                                        2, -1, 3, -1, 6, -1, 7, -1, 10, -1, 11, -1, 14, -1, 15, -1);
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        int16_t *ub = u + (size_t)b * 32 * 16;
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            /* the lanes of a 32-byte step: rows 0 to 7 (low 4 bits), 8 to
             * 15 (high), columns 4 k to 4 k + 3 */
            _mm256_store_si256((__m256i *)(ub + (4 * k + 0) * 16),
                               _mm256_sub_epi16(_mm256_shuffle_epi8(lo, sa), eight));
            _mm256_store_si256((__m256i *)(ub + (4 * k + 1) * 16),
                               _mm256_sub_epi16(_mm256_shuffle_epi8(hi, sa), eight));
            _mm256_store_si256((__m256i *)(ub + (4 * k + 2) * 16),
                               _mm256_sub_epi16(_mm256_shuffle_epi8(lo, sb), eight));
            _mm256_store_si256((__m256i *)(ub + (4 * k + 3) * 16),
                               _mm256_sub_epi16(_mm256_shuffle_epi8(hi, sb), eight));
        }
        _mm256_store_ps(dsc + (size_t)b * 16,
                        _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 256))));
        _mm256_store_ps(dsc + (size_t)b * 16 + 8,
                        _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 272))));
    }
}

/* A group of 16 rows on T tokens (1 to KQ16_TMAX), nb blocks of 32 columns
 * (a chunk of the row): the int16 x of token t at xq + t * xstride, its
 * scales at xs + t * sstride: out[t * ostride + r]. The int32 sum of a
 * block, then the scale of the block of the row times the scale of x, block
 * after block. ACC 1 goes on from the sums in out (the chunk before). */
#define KQ16_TMAX 4
static inline __attribute__((always_inline)) void kq_q4x_tile16(
    const int16_t *u, const float *d, int nb, size_t xstride, size_t sstride, const int16_t *xq,
    const float *xs, float *out, size_t ostride, const int T, const int ACC)
{
    __m256 flo[KQ16_TMAX], fhi[KQ16_TMAX];
    for (int t = 0; t < T; ++t) {
        flo[t] = ACC ? _mm256_loadu_ps(out + (size_t)t * ostride) : _mm256_setzero_ps();
        fhi[t] = ACC ? _mm256_loadu_ps(out + (size_t)t * ostride + 8) : _mm256_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        const int16_t *ub = u + (size_t)b * 32 * 16;
        __m256i alo[KQ16_TMAX], ahi[KQ16_TMAX];
        for (int t = 0; t < T; ++t) {
            alo[t] = _mm256_setzero_si256();
            ahi[t] = _mm256_setzero_si256();
        }
        for (int j = 0; j < 16; ++j) {
            __m256i w0 = _mm256_load_si256((const __m256i *)(ub + (2 * j) * 16));
            __m256i w1 = _mm256_load_si256((const __m256i *)(ub + (2 * j + 1) * 16));
            for (int t = 0; t < T; ++t) {
                int32_t x2;
                memcpy(&x2, xq + (size_t)t * xstride + 32 * b + 2 * j, 4);
                __m256i xv = _mm256_set1_epi32(x2);
#if defined(RC_DPWSSD256)
                alo[t] = RC_DPWSSD256(alo[t], w0, xv);
                ahi[t] = RC_DPWSSD256(ahi[t], w1, xv);
#else
                alo[t] = _mm256_add_epi32(alo[t], _mm256_madd_epi16(w0, xv));
                ahi[t] = _mm256_add_epi32(ahi[t], _mm256_madd_epi16(w1, xv));
#endif
            }
        }
        __m256 dlo = _mm256_load_ps(d + (size_t)b * 16), dhi = _mm256_load_ps(d + (size_t)b * 16 + 8);
        for (int t = 0; t < T; ++t) {
            __m256 sx = _mm256_set1_ps(xs[(size_t)t * sstride + b]);
            flo[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(alo[t]), _mm256_mul_ps(dlo, sx), flo[t]);
            fhi[t] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ahi[t]), _mm256_mul_ps(dhi, sx), fhi[t]);
        }
    }
    for (int t = 0; t < T; ++t) {
        _mm256_storeu_ps(out + (size_t)t * ostride, flo[t]);
        _mm256_storeu_ps(out + (size_t)t * ostride + 8, fhi[t]);
    }
}

/* The scratch of the unpacked codes of a thread (one group, int16). */
static __thread int16_t *kq16_u;
static __thread float *kq16_d;
static __thread size_t kq16_un;

static void kq16_scratch(int nb)
{
    const size_t need = (size_t)nb * 16 * 32;
    if (kq16_un < need) {
        free(kq16_u);
        free(kq16_d);
        kq16_u = (int16_t *)aligned_alloc(64, need * sizeof(int16_t));
        kq16_d = (float *)aligned_alloc(64, (size_t)nb * 16 * sizeof(float));
        kq16_un = need;
    }
}

/* The blocks of a chunk of columns: the unpacked codes of a chunk (1 KB a
 * block) and the x of a block of tokens stay in the L2 (256 KB on the
 * Core i5-8500 of the target). */
#define KQ16_CHUNK 64

static inline void kq_q4x_tile16_t(const int16_t *u, const float *d, int nc, int cols, int nb,
                                   const int16_t *xj, const float *sj, float *oj, size_t ostride,
                                   int T, int acc)
{
#define KQ16_T(TT) do { if (acc) kq_q4x_tile16(u, d, nc, cols, nb, xj, sj, oj, ostride, TT, 1); \
                        else kq_q4x_tile16(u, d, nc, cols, nb, xj, sj, oj, ostride, TT, 0); } while (0)
    switch (T) {
    case 4: KQ16_T(4); break;
    case 3: KQ16_T(3); break;
    case 2: KQ16_T(2); break;
    default: KQ16_T(1); break;
    }
#undef KQ16_T
}

/* One token of int16 x without the unpack (a step): x = 256 xh + xl with
 * xh = x >> 8 (signed) and xl = x & 255 (unsigned), two int8 planes for
 * vpmaddubsw as the int8 form (the codes the unsigned operand with xh, the
 * signed one with xl). The sum of a block, 256 sum(c xh) + sum(c xl) - 8
 * sum(x), is the int32 sum of kq_q4x_tile16, and the float steps are its
 * steps: the same bits as a group. The int16 sums of xh hold 8 steps (8 * 2
 * * 15 * 128 = 30720), those of xl 4 (4 * 2 * 255 * 15 = 30600). */
/* The planes of the int16 row xq (cols values): h, l and -8 times the sum
 * of each block. */
static void kq_planes16_to(const int16_t *xq, int cols, int8_t *kp_h, uint8_t *kp_l, int32_t *kp_n)
{
    const int nb = cols / 32;
    const __m256i m8 = _mm256_set1_epi16(255), ones = _mm256_set1_epi16(1);
    for (int b = 0; b < nb; ++b) {
        __m256i v0 = _mm256_loadu_si256((const __m256i *)(xq + 32 * b));
        __m256i v1 = _mm256_loadu_si256((const __m256i *)(xq + 32 * b + 16));
        __m256i h = _mm256_packs_epi16(_mm256_srai_epi16(v0, 8), _mm256_srai_epi16(v1, 8));
        __m256i l = _mm256_packus_epi16(_mm256_and_si256(v0, m8), _mm256_and_si256(v1, m8));
        _mm256_storeu_si256((__m256i *)(kp_h + 32 * b), _mm256_permute4x64_epi64(h, 0xD8));
        _mm256_storeu_si256((__m256i *)(kp_l + 32 * b), _mm256_permute4x64_epi64(l, 0xD8));
        __m256i s = _mm256_add_epi32(_mm256_madd_epi16(v0, ones), _mm256_madd_epi16(v1, ones));
        __m128i s4 = _mm_add_epi32(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0x4E));
        s4 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0xB1));
        kp_n[b] = -8 * _mm_cvtsi128_si32(s4);
    }
}

/* The planes of a row in the scratch of a thread. */
static __thread int8_t *kp_h;
static __thread uint8_t *kp_l;
static __thread int32_t *kp_n;
static __thread size_t kp_cap;

static void kq_planes16(const int16_t *xq, int cols)
{
    if (kp_cap < (size_t)cols) {
        free(kp_h);
        free(kp_l);
        free(kp_n);
        kp_h = (int8_t *)aligned_alloc(64, (size_t)cols + 64);
        kp_l = (uint8_t *)aligned_alloc(64, (size_t)cols + 64);
        kp_n = (int32_t *)aligned_alloc(64, ((size_t)cols / 32 + 16) * sizeof(int32_t));
        kp_cap = cols;
    }
    kq_planes16_to(xq, cols, kp_h, kp_l, kp_n);
}

static void kq_q4x_rows_p16(const uint8_t *wg, int cols, const int8_t *kp_h, const uint8_t *kp_l,
                            const int32_t *kp_n, const float *xs, float *out)
{
    const int nb = cols / 32;
    const __m256i m4 = _mm256_set1_epi8(15), ones = _mm256_set1_epi16(1);
    __m256 flo = _mm256_setzero_ps(), fhi = _mm256_setzero_ps();
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        __m256i hlo = _mm256_setzero_si256(), hhi = _mm256_setzero_si256();
        __m256i llo = _mm256_setzero_si256(), lhi = _mm256_setzero_si256();
        __m256i wlo = _mm256_setzero_si256(), whi = _mm256_setzero_si256();
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i cl = _mm256_and_si256(q, m4), ch = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            int32_t h4, l4;
            memcpy(&h4, kp_h + 32 * b + 4 * k, 4);
            memcpy(&l4, kp_l + 32 * b + 4 * k, 4);
            __m256i vh = _mm256_set1_epi32(h4), vl = _mm256_set1_epi32(l4);
#if defined(RC_DPBUSD256)
            /* int32 lanes: no int16 sums, so no cut at k 3 */
            hlo = RC_DPBUSD256(hlo, cl, vh);
            hhi = RC_DPBUSD256(hhi, ch, vh);
            llo = RC_DPBUSD256(llo, vl, cl);
            lhi = RC_DPBUSD256(lhi, vl, ch);
#else
            hlo = _mm256_add_epi16(hlo, _mm256_maddubs_epi16(cl, vh));
            hhi = _mm256_add_epi16(hhi, _mm256_maddubs_epi16(ch, vh));
            llo = _mm256_add_epi16(llo, _mm256_maddubs_epi16(vl, cl));
            lhi = _mm256_add_epi16(lhi, _mm256_maddubs_epi16(vl, ch));
            if (k == 3) {
                wlo = _mm256_madd_epi16(llo, ones);
                whi = _mm256_madd_epi16(lhi, ones);
                llo = lhi = _mm256_setzero_si256();
            }
#endif
        }
        __m256i nv = _mm256_set1_epi32(kp_n[b]);
#if defined(RC_DPBUSD256)
        (void)ones;
        __m256i alo = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(hlo, 8), _mm256_add_epi32(wlo, llo)), nv);
        __m256i ahi = _mm256_add_epi32(_mm256_add_epi32(_mm256_slli_epi32(hhi, 8), _mm256_add_epi32(whi, lhi)), nv);
#else
        __m256i alo = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_slli_epi32(_mm256_madd_epi16(hlo, ones), 8),
                             _mm256_add_epi32(wlo, _mm256_madd_epi16(llo, ones))), nv);
        __m256i ahi = _mm256_add_epi32(
            _mm256_add_epi32(_mm256_slli_epi32(_mm256_madd_epi16(hhi, ones), 8),
                             _mm256_add_epi32(whi, _mm256_madd_epi16(lhi, ones))), nv);
#endif
        __m256 dlo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 256)));
        __m256 dhi = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(blk + 272)));
        __m256 sx = _mm256_set1_ps(xs[b]);
        flo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(alo), _mm256_mul_ps(dlo, sx), flo);
        fhi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(ahi), _mm256_mul_ps(dhi, sx), fhi);
    }
    _mm256_storeu_ps(out, flo);
    _mm256_storeu_ps(out + 8, fhi);
}

/* One group of 16 rows on n tokens (an expert of kq_moe_body; a step).
 * A token alone takes kq_q4x_rows_p16 (the same bits). */
static void kq_q4x_rows16(const uint8_t *wg, int cols, const int16_t *xq, const float *xs, int n,
                          float *out, size_t ostride)
{
    const int nb = cols / 32;
    if (n == 1) {
        kq_planes16(xq, cols);
        kq_q4x_rows_p16(wg, cols, kp_h, kp_l, kp_n, xs, out);
        return;
    }
    kq16_scratch(nb < KQ16_CHUNK ? nb : KQ16_CHUNK);
    for (int c0 = 0; c0 < nb; c0 += KQ16_CHUNK) {
        const int nc = nb - c0 < KQ16_CHUNK ? nb - c0 : KQ16_CHUNK;
        kq_q4x_unpack16(wg + (size_t)c0 * KQ_Q4X_BB, nc, kq16_u, kq16_d);
        for (int j = 0; j < n; j += KQ16_TMAX) {
            const int T = n - j < KQ16_TMAX ? n - j : KQ16_TMAX;
            kq_q4x_tile16_t(kq16_u, kq16_d, nc, cols, nb, xq + (size_t)j * cols + 32 * (size_t)c0,
                            xs + (size_t)j * nb + c0, out + (size_t)j * ostride, ostride, T, c0 > 0);
        }
    }
}

/* KQ_Q4X on t tokens of int16 x, inside a parallel region: a task is a
 * block of tokens of a group. */
static void kq_q4x_gemm16(const uint8_t *w, int rows, int cols, const int16_t *xq, const float *xs,
                          int t, float *out)
{
    const int nb = cols / 32, groups = rows / 16;
    const size_t gb = 16 * kq_row_bytes(KQ_Q4X, cols);
    if (t == 1) {
        /* one token (the head of a step): the planes once for each thread */
        kq_planes16(xq, cols);
        #pragma omp for schedule(static)
        for (int g = 0; g < groups; ++g) {
            kq_q4x_rows_p16(w + (size_t)g * gb, cols, kp_h, kp_l, kp_n, xs, out + 16 * g);
        }
        return;
    }
    const int cb = nb < KQ16_CHUNK ? nb : KQ16_CHUNK;
    kq16_scratch(cb);
    /* the int16 x of a block of tokens about 64 KB */
    int tb = (32 * 1024 / cols) & ~3;
    tb = tb < 8 ? 8 : (tb > 128 ? 128 : tb);
    const int blocks = (t + tb - 1) / tb;
    tb = ((t + blocks - 1) / blocks + 3) & ~3;
    #pragma omp for schedule(static)
    for (int it = 0; it < blocks * groups; ++it) {
        const int j0 = (it / groups) * tb, g = it % groups;
        if (j0 >= t) {
            continue;
        }
        const int nt = t - j0 < tb ? t - j0 : tb;
        for (int c0 = 0; c0 < nb; c0 += cb) {
            const int nc = nb - c0 < cb ? nb - c0 : cb;
            kq_q4x_unpack16(w + (size_t)g * gb + (size_t)c0 * KQ_Q4X_BB, nc, kq16_u, kq16_d);
            for (int jj = 0; jj < nt; jj += KQ16_TMAX) {
                const int j = j0 + jj, T = nt - jj < KQ16_TMAX ? nt - jj : KQ16_TMAX;
                kq_q4x_tile16_t(kq16_u, kq16_d, nc, cols, nb, xq + (size_t)j * cols + 32 * (size_t)c0,
                                xs + (size_t)j * nb + c0, out + (size_t)j * rows + 16 * g,
                                (size_t)rows, T, c0 > 0);
            }
        }
    }
}
#define KQ_Q4X_Q16 1
#else
/* Without VNNI: the same sums in C. */
static void kq_q4x_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32;
    (void)xn;
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
                int32_t a = 0;
                for (int v = 0; v < 32; ++v) {
                    int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                    a += (c - 8) * xq[(size_t)j * cols + 32 * b + v];
                }
                f += (float)a * kq_h(blk + 256 + 2 * r) * xs[(size_t)j * nb + b];
            }
            out[(size_t)j * ostride + r] = f;
        }
    }
}
#endif

/* The 16 rows of the KQ_Q4X group at wg on n rows of float32 x (row stride
 * xstride): out[j * ostride + r]. No quantization of x: the decode of the
 * Gemma 4 26B keeps float32 activations (the MOE record gives the float
 * products to 4e-6). For each step, the codes of the 16 rows (a lane is a
 * row) to float32, 4 values at a time, and one fma for each value. */
#if defined(__AVX512F__)
/* kq_q4x_rows_f for a constant count nt (1 to 4) of tokens: their sums stay
 * in registers. Each token adds its terms in the order of one token (k, then
 * u), so a group gives the bits of the steps. */
static inline __attribute__((always_inline)) void kq_q4x_rows_fn(const uint8_t *wg, int nb,
                                                                  const float *x, size_t xstride,
                                                                  const int nt, float *out,
                                                                  size_t ostride)
{
    const __m256i m4 = _mm256_set1_epi8(15);
    const __m512i mff = _mm512_set1_epi32(255), eight = _mm512_set1_epi32(8);
    __m512 f[4];
    for (int j = 0; j < nt; ++j) {
        f[j] = _mm512_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
        __m512 acc[4];
        for (int j = 0; j < nt; ++j) {
            acc[j] = _mm512_setzero_ps();
        }
        for (int k = 0; k < 8; ++k) {
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
            __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            __m512i W = _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1);
            for (int u = 0; u < 4; ++u) {
                __m512 cf = _mm512_cvtepi32_ps(_mm512_sub_epi32(
                    _mm512_and_si512(_mm512_srli_epi32(W, 8 * u), mff), eight));
                for (int j = 0; j < nt; ++j) {
                    acc[j] = _mm512_fmadd_ps(cf, _mm512_set1_ps(x[(size_t)j * xstride + 32 * b + 4 * k + u]),
                                             acc[j]);
                }
            }
        }
        __m512 sc = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(blk + 256)));
        for (int j = 0; j < nt; ++j) {
            f[j] = _mm512_fmadd_ps(acc[j], sc, f[j]);
        }
    }
    for (int j = 0; j < nt; ++j) {
        _mm512_storeu_ps(out + (size_t)j * ostride, f[j]);
    }
}
#endif

static void kq_q4x_rows_f(const uint8_t *wg, int cols, const float *x, size_t xstride, int n,
                          float *out, size_t ostride)
{
    int nb = cols / 32;
#if defined(__AVX512F__)
    /* Up to 4 tokens at a time: the codes become float32 once for all of
     * them (kq_q4x_rows_fn). */
    for (int j0 = 0; j0 < n; j0 += 4) {
        const float *xj = x + (size_t)j0 * xstride;
        float *oj = out + (size_t)j0 * ostride;
        switch (n - j0 < 4 ? n - j0 : 4) {
        case 1: kq_q4x_rows_fn(wg, nb, xj, xstride, 1, oj, ostride); break;
        case 2: kq_q4x_rows_fn(wg, nb, xj, xstride, 2, oj, ostride); break;
        case 3: kq_q4x_rows_fn(wg, nb, xj, xstride, 3, oj, ostride); break;
        default: kq_q4x_rows_fn(wg, nb, xj, xstride, 4, oj, ostride); break;
        }
    }
#else
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
                float acc = 0.f;
                for (int v = 0; v < 32; ++v) {
                    int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                    acc += (float)(c - 8) * x[(size_t)j * xstride + 32 * b + v];
                }
                f += acc * kq_h(blk + 256 + 2 * r);
            }
            out[(size_t)j * ostride + r] = f;
        }
    }
#endif
}
