/* The rows of a down task of the cold experts of a step (kq_moe_small_body)
 * for the types of 4-row groups: RECLANK_MOE_NDN (4, 8, or 16; 16 by
 * default). A down row has 640 values: 4 rows (2.7 KB) a task left the
 * work of a task (the check of the act, the setup, the sums) a large part. */
static int kq_ndn_rows(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_MOE_NDN");
        n = v ? atoi(v) : 16;
        if (n != 4 && n != 8 && n != 16) {
            n = 4;
        }
    }
    return n;
}

/* RECLANK_Q8_4ROWS=0: kq_dot1 a row at a time for Q8_0 on 1 to 3 tokens. */
static int kq_q8_4rows(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_Q8_4ROWS");
        on = !(v && v[0] == '0');
    }
    return on;
}

#if defined(__AVX512VNNI__)
/* RECLANK_KQ_TILE2 (default 1): the tiles of Q8_0 and Q6_K as
 * kq_rows4_t2; 0 keeps kq_tile4. RECLANK_KQ_TILE2_MIN (2): the fewest
 * tokens for it (fewer: kq_dot4_q8_0, kq_dot1). */
static int kq_tile2_min(void)
{
    static int n = -2;
    if (n == -2) {
        const char *v = getenv("RECLANK_KQ_TILE2");
        const char *m = getenv("RECLANK_KQ_TILE2_MIN");
        n = v && v[0] == '0' ? -1 : (m ? atoi(m) : 2);
    }
    return n;
}

#define KQ_T2_NV 64          /* the most steps of 64 values of a row (4096 values) */

/* The scales of 4 rows for kq_t2_tile, one vector for each step of 64
 * values in the lanes of its products (dE: d for Q8_0, d * sc for Q6_K),
 * and for Q8_0 the sums of the weights times -128 (nE). ds gets the 16
 * scales of each Q6_K block (the term corr). The values of kq_row_scales. */
static void kq_t2_rows(const uint8_t *const wr[4], int type, int cols, __m512 dE[4][KQ_T2_NV],
                       __m512i nE[4][KQ_T2_NV], float ds[4][4 * KQ_T2_NV])
{
    const __m512i z = _mm512_setzero_si512(), k80 = _mm512_set1_epi8((char)0x80);
    for (int i = 0; i < 4; ++i) {
        if (type == KQ_Q8_0) {
            for (int q = 0; q < cols / 64; ++q) {
                const uint8_t *b0 = wr[i] + (size_t)q * 68;
                dE[i][q] = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(kq_h(b0)),
                                                _mm512_set1_ps(kq_h(b0 + 34)));
                __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36));
                nE[i][q] = _mm512_sub_epi32(z, _mm512_dpbusd_epi32(z, k80, wv));
            }
        } else {
            const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
            for (int b = 0; b < cols / 256; ++b) {
                const uint8_t *blk = wr[i] + (size_t)b * 210;
                __m512 dv = _mm512_mul_ps(_mm512_set1_ps(kq_h(blk + 208)), _mm512_cvtepi32_ps(
                    _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(blk + 192)))));
                _mm512_storeu_ps(ds[i] + 16 * b, dv);
                for (int s = 0; s < 4; ++s) {
                    dE[i][4 * b + s] = _mm512_permutexvar_ps(
                        _mm512_add_epi32(iA, _mm512_set1_epi32(4 * s)), dv);
                }
            }
        }
    }
}

/* 4 rows by NT tokens (a constant), as kq_tile4: each (row, token) pair adds
 * in the order of kq_dot1 with the same scales (S = ds * xs) and integer
 * sums, so the same bits; the scales of a step are dE times the scales of
 * the token, made in the loop (no kq_prep). Q8_0: x + 128 (unsigned) times w
 * (signed), from -128 sum(w) (nE): the sums of kq_dot_q8_0 with no sign
 * steps. pf (or null): prefetch the 4 rows after these (the next task) as
 * the loop goes. */
static inline __attribute__((always_inline)) void kq_t2_tile(
    const uint8_t *const wr[4], size_t rb, int type, int cols, __m512 dE[4][KQ_T2_NV],
    __m512i nE[4][KQ_T2_NV], float ds[4][4 * KQ_T2_NV], const int8_t *xq, const float *xs,
    const float *xm, const int NT, const uint8_t *pf, float *out, size_t ostr_r, size_t ostr_t,
    const int8_t *xl)
{
    /* xl (or null): the low plane of x as int16 (kq_quant_part16): xq the high
     * one; the integer sum of a step is 128 sum(hi w) + sum(lo w), exact */
    __m512 acc[4][4];
    float corr[4][4] = {{0}};
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
#pragma GCC unroll 4
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = _mm512_setzero_ps();
        }
    }
    const int np = cols / 32;
    const size_t span = 4 * rb;
    if (type == KQ_Q8_0) {
        const __m512i k80 = _mm512_set1_epi8((char)0x80);
        const int nq = cols / 64;
        const size_t per = (span + nq - 1) / nq;
        for (int q = 0; q < nq; ++q) {
            if (pf) {
                for (size_t o = 0; o < per + 64; o += 64) {
                    _mm_prefetch((const char *)pf + (size_t)q * per + o, _MM_HINT_T1);
                }
            }
            __m512i w[4];
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i) {
                const uint8_t *b0 = wr[i] + (size_t)q * 68;
                w[i] = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36));
            }
#pragma GCC unroll 4
            for (int j = 0; j < NT; ++j) {
                __m512i xu = _mm512_xor_si512(
                    _mm512_loadu_si512((const void *)(xq + (size_t)j * cols + (size_t)q * 64)), k80);
                __m512i xul = xl == NULL ? xu : _mm512_xor_si512(
                    _mm512_loadu_si512((const void *)(xl + (size_t)j * cols + (size_t)q * 64)), k80);
                const float *s = xs + (size_t)j * np + 2 * q;
                __m512 xv = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(s[0]), _mm512_set1_ps(s[1]));
#pragma GCC unroll 4
                for (int i = 0; i < 4; ++i) {
                    __m512i is = _mm512_dpbusd_epi32(nE[i][q], xu, w[i]);
                    if (xl != NULL) {
                        is = _mm512_add_epi32(_mm512_slli_epi32(is, 7),
                                              _mm512_dpbusd_epi32(nE[i][q], xul, w[i]));
                    }
                    acc[i][j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_mul_ps(dE[i][q], xv),
                                                acc[i][j]);
                }
            }
        }
    } else {
        const __m512i z = _mm512_setzero_si512();
        const int nb = cols / 256;
        const size_t per = (span + nb - 1) / nb;
        for (int b = 0; b < nb; ++b) {
            if (pf) {
                for (size_t o = 0; o < per + 64; o += 64) {
                    _mm_prefetch((const char *)pf + (size_t)b * per + o, _MM_HINT_T1);
                }
            }
            for (int h = 0; h < 2; ++h) {
                __m512i A[4], B[4];
#pragma GCC unroll 4
                for (int i = 0; i < 4; ++i) {
                    kq_q6_values(wr[i] + (size_t)b * 210, h, &A[i], &B[i]);
                }
#pragma GCC unroll 4
                for (int j = 0; j < NT; ++j) {
                    const int8_t *xb = xq + (size_t)j * cols + (size_t)b * 256 + 128 * h;
                    __m512i x0 = _mm512_loadu_si512((const void *)xb);
                    __m512i x1 = _mm512_loadu_si512((const void *)(xb + 64));
                    const int8_t *xlb = xl == NULL ? xb : xl + (size_t)j * cols + (size_t)b * 256 + 128 * h;
                    __m512i x0l = _mm512_loadu_si512((const void *)xlb);
                    __m512i x1l = _mm512_loadu_si512((const void *)(xlb + 64));
                    const float *s = xs + (size_t)j * np + 8 * b + 4 * h;
                    __m512 xa = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(s[0]), _mm512_set1_ps(s[1]));
                    __m512 xc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(s[2]), _mm512_set1_ps(s[3]));
#pragma GCC unroll 4
                    for (int i = 0; i < 4; ++i) {
                        __m512i ia = _mm512_dpbusd_epi32(z, A[i], x0), ib = _mm512_dpbusd_epi32(z, B[i], x1);
                        if (xl != NULL) {
                            ia = _mm512_add_epi32(_mm512_slli_epi32(ia, 7), _mm512_dpbusd_epi32(z, A[i], x0l));
                            ib = _mm512_add_epi32(_mm512_slli_epi32(ib, 7), _mm512_dpbusd_epi32(z, B[i], x1l));
                        }
                        acc[i][j] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(ia),
                            _mm512_mul_ps(dE[i][4 * b + 2 * h], xa), acc[i][j]);
                        acc[i][j] = _mm512_fmadd_ps(
                            _mm512_cvtepi32_ps(ib),
                            _mm512_mul_ps(dE[i][4 * b + 2 * h + 1], xc), acc[i][j]);
                    }
                }
            }
        }
        /* the term of the sums of x (kq_prep) */
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < NT; ++j) {
                __m512 macc = _mm512_setzero_ps();
                for (int b = 0; b < nb; ++b) {
                    macc = _mm512_fmadd_ps(_mm512_loadu_ps(ds[i] + 16 * b),
                                           _mm512_loadu_ps(xm + (size_t)j * (cols / 16) + 16 * b), macc);
                }
                corr[i][j] = 32.f * _mm512_reduce_add_ps(macc);
            }
        }
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
#pragma GCC unroll 4
        for (int j = 0; j < NT; ++j) {
            out[(size_t)i * ostr_r + (size_t)j * ostr_t] = _mm512_reduce_add_ps(acc[i][j]) - corr[i][j];
        }
    }
}

/* RECLANK_KQ_T1 (default 1): one token of int16 x (a decode step) on 4
 * rows of Q8_0 in one pass (kq_rows4_q8_x16_t1), not kq_rows4_t2 (its scales
 * and sums of the rows first, for tiles of tokens). */
static int kq_t1_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_KQ_T1");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* Rows r .. r + 3 of Q8_0 (at w, rb bytes each; cols a multiple of 64) on
 * one token of int16 x (xq the high plane, xl the low one, xs the scales):
 * the bits of kq_t2_tile (NT 1), in one pass. The integer sum of each lane
 * is the same: there x + 128 (unsigned) times w, from -128 sum(w) of each
 * row (nE, kq_t2_rows); here w + 128 (w ^ 0x80) times x, less 128 sum(x),
 * one vector of each step of 64 values for the 4 rows. The scales of a
 * step: the two fp16 d of its blocks in the lanes (the values of kq_h).
 * On the 2-socket Xeon at 1.2 GHz kq_rows4_t2 did 4 rows of 640 in 0.66 us
 * from cache (one thread): the scales and the sums of the rows were as much
 * work as the products for one token. */
static void kq_rows4_q8_x16_t1(const uint8_t *w, size_t rb, int cols, const int8_t *xq,
                               const float *xs, const int8_t *xl, float *out, const uint8_t *pf)
{
    const __m512i k80 = _mm512_set1_epi8((char)0x80), z = _mm512_setzero_si512();
    const int nq = cols / 64;
    const size_t per = (4 * rb + nq - 1) / nq;
    __m512 acc[4];
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        acc[i] = _mm512_setzero_ps();
    }
    for (int q = 0; q < nq; ++q) {
        if (pf) {
            for (size_t o = 0; o < per + 64; o += 64) {
                _mm_prefetch((const char *)pf + (size_t)q * per + o, _MM_HINT_T1);
            }
        }
        __m512i xh = _mm512_loadu_si512((const void *)(xq + (size_t)q * 64));
        __m512i xo = _mm512_loadu_si512((const void *)(xl + (size_t)q * 64));
        /* 128 sum(x) of each lane, as (hi << 7) + lo */
        __m512i C = _mm512_add_epi32(_mm512_slli_epi32(_mm512_dpbusd_epi32(z, k80, xh), 7),
                                     _mm512_dpbusd_epi32(z, k80, xo));
        __m512 xv = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(xs[2 * q]), _mm512_set1_ps(xs[2 * q + 1]));
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            const uint8_t *b0 = w + (size_t)i * rb + (size_t)q * 68;
            __m512i wu = _mm512_xor_si512(kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36)), k80);
            __m512i is = _mm512_sub_epi32(
                _mm512_dpbusd_epi32(_mm512_slli_epi32(_mm512_dpbusd_epi32(z, wu, xh), 7), wu, xo), C);
            uint16_t d0, d1;
            memcpy(&d0, b0, 2);
            memcpy(&d1, b0 + 34, 2);
            __m512 dE = _mm512_cvtph_ps(_mm256_mask_blend_epi16(0xff00, _mm256_set1_epi16((short)d0),
                                                                _mm256_set1_epi16((short)d1)));
            acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_mul_ps(dE, xv), acc[i]);
        }
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        out[i] = _mm512_reduce_add_ps(acc[i]);
    }
}

/* Rows r .. r + 3 of Q6_K (cols a multiple of 256) on one token of int16 x:
 * the bits of kq_t2_tile (NT 1) in one pass: the scales of a block (d times
 * its 16 scales, as kq_t2_rows) where the block is read, and the term of
 * the sums of x (xm) in the same order. */
static void kq_rows4_q6_x16_t1(const uint8_t *w, size_t rb, int cols, const int8_t *xq,
                               const float *xs, const float *xm, const int8_t *xl, float *out,
                               const uint8_t *pf)
{
    const __m512i z = _mm512_setzero_si512();
    const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const int nb = cols / 256, np = cols / 32;
    const size_t per = (4 * rb + nb - 1) / nb;
    (void)np;
    __m512 acc[4], macc[4];
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        acc[i] = _mm512_setzero_ps();
        macc[i] = _mm512_setzero_ps();
    }
    for (int b = 0; b < nb; ++b) {
        if (pf) {
            for (size_t o = 0; o < per + 64; o += 64) {
                _mm_prefetch((const char *)pf + (size_t)b * per + o, _MM_HINT_T1);
            }
        }
        __m512 dv[4];
        const __m512 xmv = _mm512_loadu_ps(xm + 16 * b);
#pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            const uint8_t *blk = w + (size_t)i * rb + (size_t)b * 210;
            dv[i] = _mm512_mul_ps(_mm512_set1_ps(kq_h(blk + 208)), _mm512_cvtepi32_ps(
                _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(blk + 192)))));
            macc[i] = _mm512_fmadd_ps(dv[i], xmv, macc[i]);
        }
        for (int h = 0; h < 2; ++h) {
            const int8_t *xb = xq + (size_t)b * 256 + 128 * h, *xlb = xl + (size_t)b * 256 + 128 * h;
            __m512i x0 = _mm512_loadu_si512((const void *)xb), x1 = _mm512_loadu_si512((const void *)(xb + 64));
            __m512i x0l = _mm512_loadu_si512((const void *)xlb), x1l = _mm512_loadu_si512((const void *)(xlb + 64));
            const float *sx = xs + 8 * b + 4 * h;
            __m512 xa = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sx[0]), _mm512_set1_ps(sx[1]));
            __m512 xc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sx[2]), _mm512_set1_ps(sx[3]));
            const __m512i pa = _mm512_add_epi32(iA, _mm512_set1_epi32(8 * h));
            const __m512i pb = _mm512_add_epi32(iA, _mm512_set1_epi32(8 * h + 4));
#pragma GCC unroll 4
            for (int i = 0; i < 4; ++i) {
                __m512i A, B;
                kq_q6_values(w + (size_t)i * rb + (size_t)b * 210, h, &A, &B);
                __m512i ia = _mm512_dpbusd_epi32(_mm512_slli_epi32(_mm512_dpbusd_epi32(z, A, x0), 7), A, x0l);
                __m512i ib = _mm512_dpbusd_epi32(_mm512_slli_epi32(_mm512_dpbusd_epi32(z, B, x1), 7), B, x1l);
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia),
                                         _mm512_mul_ps(_mm512_permutexvar_ps(pa, dv[i]), xa), acc[i]);
                acc[i] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ib),
                                         _mm512_mul_ps(_mm512_permutexvar_ps(pb, dv[i]), xc), acc[i]);
            }
        }
    }
#pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        out[i] = _mm512_reduce_add_ps(acc[i]) - 32.f * _mm512_reduce_add_ps(macc[i]);
    }
}

/* Rows r .. r + 3 of Q8_0 or Q6_K (at w, rb bytes each; cols a multiple of
 * 64 or 256, at most 64 KQ_T2_NV) on n tokens: the scales of the rows one
 * time, then tiles of 4 tokens and one of the rest. */
static void kq_rows4_t2(const uint8_t *w, size_t rb, int type, int cols, const int8_t *xq,
                        const float *xs, const float *xm, int n, float *out, size_t ostride,
                        const int8_t *xl)
{
    const uint8_t *wr[4] = {w, w + rb, w + 2 * rb, w + 3 * rb};
    __m512 dE[4][KQ_T2_NV];
    __m512i nE[4][KQ_T2_NV];
    float ds[4][4 * KQ_T2_NV];
    kq_t2_rows(wr, type, cols, dE, nE, ds);
    const uint8_t *pf = kq_pf_bytes() ? w + 4 * rb : NULL;
    for (int j0 = 0; j0 < n; j0 += 4) {
        int nt = n - j0 < 4 ? n - j0 : 4;
        const int8_t *x = xq + (size_t)j0 * cols;
        const float *s = xs + (size_t)j0 * (cols / 32), *m = xm + (size_t)j0 * (cols / 16);
        float *o = out + (size_t)j0 * ostride;
        const uint8_t *p = j0 == 0 ? pf : NULL;
        if (xl != NULL) {
            const int8_t *l = xl + (size_t)j0 * cols;
            switch (nt) {
            case 4: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 4, p, o, 1, ostride, l); break;
            case 3: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 3, p, o, 1, ostride, l); break;
            case 2: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 2, p, o, 1, ostride, l); break;
            default: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 1, p, o, 1, ostride, l); break;
            }
            continue;
        }
        switch (nt) {
        case 4: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 4, p, o, 1, ostride, NULL); break;
        case 3: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 3, p, o, 1, ostride, NULL); break;
        case 2: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 2, p, o, 1, ostride, NULL); break;
        default: kq_t2_tile(wr, rb, type, cols, dE, nE, ds, x, s, m, 1, p, o, 1, ostride, NULL); break;
        }
    }
}
#endif

static void kq_rows4(const uint8_t *w, size_t rb, int type, int cols, const int8_t *xq,
                     const float *xs, const float *xm, const float *x, int n, float *out,
                     size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (type == KQ_NV4 && n > 1 && cols % 64 == 0 && cols <= KQ_NV4_MAX) {
        kq_rows4_nv4(w, rb, cols, xq, xs, n, out, ostride);
        return;
    }
    if ((type == KQ_Q8_0 || (type == KQ_Q6_K && cols % 256 == 0)) && cols % 64 == 0 &&
        cols <= 64 * KQ_T2_NV && kq_tile2_min() >= 1 && n >= kq_tile2_min()) {
        kq_rows4_t2(w, rb, type, cols, xq, xs, xm, n, out, ostride, NULL);
        return;
    }
    if (type == KQ_Q8_0 && n < 4 && kq_q8_4rows()) {
        for (int j = 0; j < n; ++j) {
            kq_dot4_q8_0(w, rb, cols, xq + (size_t)j * cols, xs + (size_t)j * (cols / 32),
                         out + (size_t)j * ostride, 1);
        }
        return;
    }
    if (n >= 4 && kq_tiles(type, cols)) {
        const uint8_t *wr[4] = {w, w + rb, w + 2 * rb, w + 3 * rb};
        float ds[4][KQ_S], dm[4][KQ_S];
        for (int i = 0; i < 4; ++i) {
            kq_row_scales(wr[i], type, cols, ds[i], dm[i]);
        }
        for (int j0 = 0; j0 < n; j0 += 4) {
            int nt = n - j0 < 4 ? n - j0 : 4;
            kq_tile4(wr, ds, dm, type, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * (cols / 32),
                     xm + (size_t)j0 * (cols / 16), nt, out + (size_t)j0 * ostride, 1, ostride);
        }
        return;
    }
#endif
    for (int i = 0; i < 4; ++i) {
        kq_row(w + (size_t)i * rb, type, cols, xq, xs, xm, x, n, out + i, ostride);
    }
}
