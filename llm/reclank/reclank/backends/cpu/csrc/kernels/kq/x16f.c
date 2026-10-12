/* KQ_NVX: out (t x rows), inside a parallel region; blocks of MA_TB tokens
 * with the groups of 16 rows over the threads. */
static void kq_nvx_body(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        int t, float *out)
{
    int32_t *xn = NULL;
    size_t gb = 16 * kq_row_bytes(KQ_NVX, cols);
    #pragma omp single copyprivate(xn)
    xn = (int32_t *)malloc((size_t)t * (cols / 16) * 4);
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        kq_nvx_xsum(xq + (size_t)j * cols, 1, cols, xn + (size_t)j * (cols / 16));
    }
    for (int j0 = 0; j0 < t; j0 += MA_TB) {
        int nt = t - j0 < MA_TB ? t - j0 : MA_TB;
        #pragma omp for schedule(static) nowait
        for (int g = 0; g < rows / 16; ++g) {
            kq_nvx_rows(w + (size_t)g * gb, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * (cols / 32),
                        xn + (size_t)j0 * (cols / 16), nt, out + (size_t)j0 * rows + 16 * g,
                        (size_t)rows);
        }
    }
    #pragma omp barrier
    #pragma omp single
    free(xn);
}

#if defined(__AVX512F__)
#ifndef KQ_X16F_GB
#define KQ_X16F_GB 4
#endif

/* ng (at most 4) groups of 16 rows (gb bytes apart; the last has nr rows)
 * on at most 4 tokens j0 .. j1 - 1: the groups give independent sums, so
 * the fma of a token do not wait on each other (one group on one token
 * waits 4 cycles for each column). A missing group repeats group 0 and is
 * not stored. For each row and token: one fma for each column, in order, as
 * kq_x16f_group16: the same bits. */
/* Column c of a group of 16 rows of a KQ_BF16X16 (bf 1), KQ_BF12X16 (bf 2),
 * or KQ_F32X16 (bf 0) matrix as 16 floats. KQ_BF12X16: in each block of 32
 * columns (784 bytes) the 16 rows' sign and mantissa bytes column by column
 * (512), their gap nibbles (256: byte 16 j + r has the gap of column j of
 * row r in its low half, of column j + 16 in its high half), and their
 * largest exponents (16); the bits of BF12 (kq_bf12_bits). */
static inline __attribute__((always_inline)) __m512 kq_x16f_col(const uint8_t *wg, int bf, int c)
{
    if (bf == 1) {
        return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
            _mm256_loadu_si256((const __m256i *)(wg + (size_t)32 * c))), 16));
    }
    if (bf == 2) {
        const uint8_t *blk = wg + (size_t)(c >> 5) * 784;
        int cc = c & 31;
        __m512i l = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(blk + 16 * cc)));
        __m512i h = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(blk + 512 + 16 * (cc & 15))));
        __m512i gap = _mm512_and_si512(_mm512_srlv_epi32(h, _mm512_set1_epi32(4 * (cc >> 4))),
                                       _mm512_set1_epi32(15));
        __m512i e = _mm512_and_si512(_mm512_sub_epi32(_mm512_cvtepu8_epi32(
                        _mm_loadu_si128((const __m128i *)(blk + 768))), gap), _mm512_set1_epi32(255));
        __m512i b = _mm512_or_si512(_mm512_or_si512(
                        _mm512_slli_epi32(_mm512_and_si512(l, _mm512_set1_epi32(0x80)), 24),
                        _mm512_slli_epi32(e, 23)),
                        _mm512_slli_epi32(_mm512_and_si512(l, _mm512_set1_epi32(0x7f)), 16));
        __mmask16 z = _mm512_cmpeq_epi32_mask(gap, _mm512_set1_epi32(15)) &
                      _mm512_cmpeq_epi32_mask(l, _mm512_set1_epi32(0x80));
        return _mm512_castsi512_ps(_mm512_maskz_mov_epi32((__mmask16)~z, b));
    }
    return _mm512_loadu_ps((const float *)(wg + (size_t)64 * c));
}

static void kq_x16f_groups4(const uint8_t *wg, size_t gb, int ng, int nr, int bf, int cols,
                            size_t xstride, const float *x, int j0, int j1, float *out,
                            size_t ostride)
{
    int nt = j1 - j0;
    const uint8_t *gp[4];
    for (int g = 0; g < 4; ++g) {
        gp[g] = wg + (g < ng ? g : 0) * gb;
    }
    __m512 f[4][4];
    for (int g = 0; g < 4; ++g) {
        for (int tt = 0; tt < 4; ++tt) {
            f[g][tt] = _mm512_setzero_ps();
        }
    }
    const float *xr = x + (size_t)j0 * xstride;
    for (int c = 0; c < cols; ++c) {
        __m512 wv[4];
        #pragma GCC unroll 4
        for (int g = 0; g < 4; ++g) {
            wv[g] = kq_x16f_col(gp[g], bf, c);
        }
        #pragma GCC unroll 4
        for (int tt = 0; tt < 4; ++tt) {
            if (tt < nt) {
                __m512 xv = _mm512_set1_ps(xr[(size_t)tt * xstride + c]);
                #pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
                    f[g][tt] = _mm512_fmadd_ps(wv[g], xv, f[g][tt]);
                }
            }
        }
    }
    for (int tt = 0; tt < nt; ++tt) {
        for (int g = 0; g < ng; ++g) {
            int n = g == ng - 1 ? nr : 16;
            __mmask16 mk = (__mmask16)(n >= 16 ? 0xffff : (1u << n) - 1);
            _mm512_mask_storeu_ps(out + (size_t)(j0 + tt) * ostride + 16 * g, mk, f[g][tt]);
        }
    }
}

/* One group of 16 rows on tokens j0 .. j1 - 1, 16 tokens at a time, with
 * loops of fixed counts (the 16 sums stay in registers; each broadcast is
 * an operand of its fma). A batch of fewer than 16 tokens repeats its last
 * token and does not store it. */
static void kq_x16f_group16(const uint8_t *wg, int bf, int cols, size_t xstride, const float *x,
                            int j0, int j1, int nr, float *out, size_t ostride)
{
    __mmask16 mk = (__mmask16)(nr >= 16 ? 0xffff : (1u << nr) - 1);
    for (int j = j0; j < j1; j += 16) {
        int nt = j1 - j < 16 ? j1 - j : 16;
        const float *xp[16];
        for (int tt = 0; tt < 16; ++tt) {
            xp[tt] = x + (size_t)(j + (tt < nt ? tt : nt - 1)) * xstride;
        }
        __m512 f[16];
        for (int tt = 0; tt < 16; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int c = 0; c < cols; ++c) {
            __m512 wv = kq_x16f_col(wg, bf, c);
            #pragma GCC unroll 16
            for (int tt = 0; tt < 16; ++tt) {
                f[tt] = _mm512_fmadd_ps(wv, _mm512_set1_ps(xp[tt][c]), f[tt]);
            }
        }
        #pragma GCC unroll 16
        for (int tt = 0; tt < 16; ++tt) {
            if (tt < nt) {
                _mm512_mask_storeu_ps(out + (size_t)(j + tt) * ostride, mk, f[tt]);
            }
        }
    }
}
#elif defined(__AVX2__)
/* AVX2: one group of 16 rows (two vectors of 8) on tokens j0 .. j1 - 1, 6
 * tokens at a time (12 sums, 2 vectors of W, and the broadcast take 15 of
 * the 16 registers). One fma for each row, token, and column, in order. */
static void kq_x16f_group(const uint8_t *wg, int bf, int cols, size_t xstride, const float *x,
                          int j0, int j1, int nr, float *out, size_t ostride)
{
    for (int j = j0; j < j1; j += 6) {
        int nt = j1 - j < 6 ? j1 - j : 6;
        const float *xp[6];
        for (int tt = 0; tt < 6; ++tt) {
            xp[tt] = x + (size_t)(j + (tt < nt ? tt : nt - 1)) * xstride;
        }
        __m256 f[6][2];
        for (int tt = 0; tt < 6; ++tt) {
            f[tt][0] = _mm256_setzero_ps();
            f[tt][1] = _mm256_setzero_ps();
        }
        for (int c = 0; c < cols; ++c) {
            __m256 w0, w1;
            if (bf == 2) {
                float tmp[16];
                kq_bf12x16_col(wg, c, tmp);
                w0 = _mm256_loadu_ps(tmp);
                w1 = _mm256_loadu_ps(tmp + 8);
            } else if (bf) {
                __m256i u = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(wg + (size_t)32 * c)));
                __m256i v = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)(wg + (size_t)32 * c + 16)));
                w0 = _mm256_castsi256_ps(_mm256_slli_epi32(u, 16));
                w1 = _mm256_castsi256_ps(_mm256_slli_epi32(v, 16));
            } else {
                w0 = _mm256_loadu_ps((const float *)(wg + (size_t)64 * c));
                w1 = _mm256_loadu_ps((const float *)(wg + (size_t)64 * c + 32));
            }
            #pragma GCC unroll 6
            for (int tt = 0; tt < 6; ++tt) {
                __m256 xv = _mm256_set1_ps(xp[tt][c]);
                f[tt][0] = _mm256_fmadd_ps(w0, xv, f[tt][0]);
                f[tt][1] = _mm256_fmadd_ps(w1, xv, f[tt][1]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            float buf[16];
            _mm256_storeu_ps(buf, f[tt][0]);
            _mm256_storeu_ps(buf + 8, f[tt][1]);
            float *o = out + (size_t)(j + tt) * ostride;
            for (int r = 0; r < nr; ++r) {
                o[r] = buf[r];
            }
        }
    }
}
#else
/* Without AVX2: the same sums in C (not the bits of the AVX-512 path). */
static void kq_x16f_group(const uint8_t *wg, int bf, int cols, size_t xstride, const float *x,
                          int j0, int j1, int nr, float *out, size_t ostride)
{
    for (int j = j0; j < j1; ++j) {
        for (int r = 0; r < nr; ++r) {
            float s = 0.f;
            for (int c = 0; c < cols; ++c) {
                float v;
                if (bf == 2) {
                    float tmp[16];
                    kq_bf12x16_col(wg, c, tmp);
                    v = tmp[r];
                } else if (bf) {
                    uint32_t u = (uint32_t)((const uint16_t *)wg)[16 * c + r] << 16;
                    memcpy(&v, &u, 4);
                } else {
                    v = ((const float *)wg)[16 * c + r];
                }
                s += v * x[(size_t)j * xstride + c];
            }
            out[(size_t)j * ostride + r] = s;
        }
    }
}
#endif

/* out (t x rows) of a KQ_BF16X16 or KQ_F32X16 matrix, inside a parallel
 * region. A task: at most 4 tokens, 4 groups or 1 (kq_x16f_groups4); more
 * tokens, KQ_X16F_GB groups on 16 tokens, so x comes from the cache of the
 * core for all of them (kq_x16f_group16). Tiles of 4 groups by 64 tokens in
 * chunks of 256 columns, with x as [column][token], were slower (44 to 48
 * ms, not 28, for 10240 x 2560 on 512 tokens). */
static void kq_x16f_body(const uint8_t *w, int type, int rows, int cols, const float *x, int t,
                         float *out)
{
    int ng = (rows + 15) / 16, bf = type == KQ_BF16X16 ? 1 : (type == KQ_BF12X16 ? 2 : 0);
    size_t gb = (size_t)16 * kq_row_bytes(type, cols);
    int u = 1, tk = 16;
#if defined(__AVX512F__)
    if (t <= 4) {
        /* 4 groups in a task when there are enough for all the threads */
        u = ng >= 4 * omp_get_num_threads() ? 4 : 1, tk = 4;
    } else {
        u = KQ_X16F_GB;
    }
#endif
    int nu = (ng + u - 1) / u, nj = (t + tk - 1) / tk;
    #pragma omp for schedule(static)
    for (int e = 0; e < nu * nj; ++e) {
        int g = u * (e / nj), j0 = tk * (e % nj), j1 = j0 + tk < t ? j0 + tk : t;
        int n = ng - g < u ? ng - g : u;                   /* the groups of the task */
        int last = rows - 16 * (g + n - 1);                /* the rows of its last group */
        last = last < 16 ? last : 16;
        const uint8_t *wg = w + (size_t)g * gb;
        float *o = out + (size_t)16 * g;
#if defined(__AVX512F__)
        if (t <= 4) {
            kq_x16f_groups4(wg, gb, n, last, bf, cols, (size_t)cols, x, j0, j1, o, (size_t)rows);
            continue;
        }
        for (int gi = 0; gi < n; ++gi) {
            kq_x16f_group16(wg + gi * gb, bf, cols, (size_t)cols, x, j0, j1, gi == n - 1 ? last : 16,
                            o + 16 * gi, (size_t)rows);
        }
#else
        kq_x16f_group(wg, bf, cols, (size_t)cols, x, j0, j1, last, o, (size_t)rows);
#endif
    }
}

static void kq_linear_body(const uint8_t *w, int type, int rows, int cols, const int8_t *xq,
                           const float *xs, const float *xm, const float *x, int t, float *out)
{
    if (kq_e3_bits(type)) {
        /* an EXL3 matrix (kernels/kq/exl3.c): reads the float rows x */
        kq_e3_linear_body(w, kq_e3_bits(type), rows, cols, x, t, out);
        return;
    }
    if (type == KQ_BF16X16 || type == KQ_F32X16 || type == KQ_BF12X16) {
        kq_x16f_body(w, type, rows, cols, x, t, out);
        return;
    }
    if (type == KQ_NVX) {
        kq_nvx_body(w, rows, cols, xq, xs, t, out);
        return;
    }
    if (type == KQ_Q4X) {
        kq_q4x_body(w, rows, cols, xq, xs, t, out);
        return;
    }
    size_t rb = kq_row_bytes(type, cols);
#if defined(__AVX512VNNI__) || defined(__AVX2__)
    if (type == KQ_Q8X16) {
        kq_x16_body(w, rows, cols, xq, xs, t, out);
        return;
    }
#endif
#if defined(__AVX512VNNI__)
    if (t >= 4 && rows % 4 == 0 && kq_tiles(type, cols)) {
        /* Blocks of MA_TB tokens (x in L2). */
        for (int j0 = 0; j0 < t; j0 += MA_TB) {
            int nb = t - j0 < MA_TB ? t - j0 : MA_TB;
            #pragma omp for schedule(static) nowait
            for (int r4 = 0; r4 < rows / 4; ++r4) {
                kq_rows4(w + (size_t)4 * r4 * rb, rb, type, cols, xq + (size_t)j0 * cols,
                         xs + (size_t)j0 * (cols / 32), xm + (size_t)j0 * (cols / 16), NULL, nb,
                         out + (size_t)j0 * rows + 4 * r4, (size_t)rows);
            }
        }
        #pragma omp barrier
        return;
    }
#endif
    if (rows < 64 && t > 1) {
        /* few rows (the small float32 matrices): tasks of a row and 16
         * tokens, so the tokens take the threads; the sums of each token
         * are those of kq_row */
        int nj = (t + 15) / 16;
        #pragma omp for schedule(static)
        for (int e = 0; e < rows * nj; ++e) {
            int r = e / nj, j0 = 16 * (e % nj), n = t - j0 < 16 ? t - j0 : 16;
            kq_row(w + (size_t)r * rb, type, cols, xq + (size_t)j0 * cols,
                   xs + (size_t)j0 * (cols / 32), xm + (size_t)j0 * (cols / 16),
                   x ? x + (size_t)j0 * cols : NULL, n, out + (size_t)j0 * rows + r, (size_t)rows);
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; ++r) {
        kq_row(w + (size_t)r * rb, type, cols, xq, xs, xm, x, t, out + r, (size_t)rows);
    }
}

void rc_kq_linear(const uint8_t *w, int type, int rows, int cols, const int8_t *xq, const float *xs,
               const float *xm, const float *x, int t, float *out)
{
    #pragma omp parallel
    kq_linear_body(w, type, rows, cols, xq, xs, xm, x, t, out);
}
