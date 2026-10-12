/* The 6-bit scale and min of part j of a Q4_K or Q5_K block (ggml,
 * get_scale_min_k4). */
static inline void kq_scale_min(const uint8_t *q, int j, int *sc, int *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* The float values of block b of a row (the reference, and the path
 * without VNNI). out gets 32 (Q8_0) or 256 values. */
static const int8_t kq_iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                            1, 13, 25, 38, 53, 69, 89, 113};

/* The values of a block of 32 of a type of blocks of 32. */
static inline int kq_block32(int type)
{
    return type == KQ_Q8_0 || type == KQ_Q5_1 || type == KQ_IQ4_NL || type == KQ_BF16 ||
           type == KQ_NV4;
}

static void kq_block_values(int type, const uint8_t *row, int cols, int b, float *out)
{
    if (type == KQ_BF16) {
        const uint16_t *h = (const uint16_t *)row + (size_t)b * 32;
        for (int i = 0; i < 32; ++i) {
            out[i] = kq_bf16(h[i]);
        }
    } else if (type == KQ_NV4) {
        float g = kq_nv4_g(row, cols);
        const uint8_t *q = kq_nv4_codes(row, b), *sc = kq_nv4_scales(row, cols, b);
        float s0 = 0.5f * g * kq_e4m3(sc[0]), s1 = 0.5f * g * kq_e4m3(sc[1]);
        for (int j = 0; j < 16; ++j) {
            out[j] = s0 * (float)kq_e2m1x2[q[j] & 15];
            out[j + 16] = s1 * (float)kq_e2m1x2[q[j] >> 4];
        }
    } else if (type == KQ_Q5_1) {
        const uint8_t *blk = row + (size_t)b * 24;
        float d = kq_h(blk), m = kq_h(blk + 2);
        uint32_t qh;
        memcpy(&qh, blk + 4, 4);
        for (int j = 0; j < 16; ++j) {
            int lo = (blk[8 + j] & 15) | (((qh >> j) & 1) << 4);
            int hi = (blk[8 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
            out[j] = d * (float)lo + m;
            out[j + 16] = d * (float)hi + m;
        }
    } else if (type == KQ_IQ4_NL) {
        const uint8_t *blk = row + (size_t)b * 18;
        float d = kq_h(blk);
        for (int j = 0; j < 16; ++j) {
            out[j] = d * (float)kq_iq4nl_values[blk[2 + j] & 15];
            out[j + 16] = d * (float)kq_iq4nl_values[blk[2 + j] >> 4];
        }
    } else if (type == KQ_Q8_0) {
        const uint8_t *blk = row + (size_t)b * 34;
        float d = kq_h(blk);
        for (int i = 0; i < 32; ++i) {
            out[i] = d * (float)(int8_t)blk[2 + i];
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        int five = type == KQ_Q5_K;
        const uint8_t *blk = row + (size_t)b * (five ? 176 : 144);
        float d = kq_h(blk), dm = kq_h(blk + 2);
        const uint8_t *qh = blk + 16, *qs = blk + (five ? 48 : 16);
        for (int c = 0; c < 4; ++c) {
            int s0, m0, s1, m1;
            kq_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            for (int l = 0; l < 32; ++l) {
                int lo = qs[32 * c + l] & 15, hi = qs[32 * c + l] >> 4;
                if (five) {
                    lo |= ((qh[l] >> (2 * c)) & 1) << 4;
                    hi |= ((qh[l] >> (2 * c + 1)) & 1) << 4;
                }
                out[64 * c + l] = d * s0 * lo - dm * m0;
                out[64 * c + 32 + l] = d * s1 * hi - dm * m1;
            }
        }
    } else if (type == KQ_Q6_K) {
        const uint8_t *blk = row + (size_t)b * 210;
        float d = kq_h(blk + 208);
        const int8_t *sc = (const int8_t *)(blk + 192);
        for (int h = 0; h < 2; ++h) {
            const uint8_t *ql = blk + 64 * h, *qh = blk + 128 + 32 * h;
            for (int l = 0; l < 32; ++l) {
                int q1 = (ql[l] & 15) | (((qh[l] >> 0) & 3) << 4);
                int q2 = (ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4);
                int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                float *o = out + 128 * h;
                o[l] = d * sc[8 * h + l / 16] * (q1 - 32);
                o[l + 32] = d * sc[8 * h + 2 + l / 16] * (q2 - 32);
                o[l + 64] = d * sc[8 * h + 4 + l / 16] * (q3 - 32);
                o[l + 96] = d * sc[8 * h + 6 + l / 16] * (q4 - 32);
            }
        }
    }
}

#if defined(__AVX512VNNI__)
static inline __attribute__((always_inline)) __m512i kq_two256(const int8_t *a, const int8_t *b)
{
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a)),
                              _mm256_loadu_si256((const __m256i *)b), 1);
}

static float kq_dot_f32(const float *w, int cols, const float *x)
{
    __m512 acc = _mm512_setzero_ps();
    for (int c = 0; c < cols; c += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + c), _mm512_loadu_ps(x + c), acc);
    }
    return _mm512_reduce_add_ps(acc);
}
#endif

#if defined(__AVX512VNNI__)
/* The scales of one row that do not depend on x: ds gets d (Q8_0), d * sc
 * (Q4_K, Q5_K: one for each 32 values; Q6_K: one for each 16), and dm gets
 * dmin * m (Q4_K, Q5_K). */
static void kq_row_scales(const uint8_t *w, int type, int cols, float *ds, float *dm)
{
    if (type == KQ_Q8_0) {
        for (int i = 0; i < cols / 32; ++i) {
            ds[i] = kq_h(w + (size_t)i * 34);
        }
    } else if (type == KQ_Q5_1) {
        /* w = d q + m: the term m sum(x) is the mins term of Q4_K with -m. */
        for (int i = 0; i < cols / 32; ++i) {
            ds[i] = kq_h(w + (size_t)i * 24);
            dm[i] = -kq_h(w + (size_t)i * 24 + 2);
        }
    } else if (type == KQ_Q6_K) {
        for (int b = 0; b < cols / 256; ++b) {
            const uint8_t *blk = w + (size_t)b * 210;
            __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)(blk + 192))));
            _mm512_storeu_ps(ds + 16 * b, _mm512_mul_ps(_mm512_set1_ps(kq_h(blk + 208)), sc));
        }
    } else {
        size_t bs = type == KQ_Q5_K ? 176 : 144;
        for (int b = 0; b < cols / 256; ++b) {
            const uint8_t *blk = w + (size_t)b * bs;
            /* The 6-bit scales and mins with 32-bit masks (as ggml): the
             * bytes of u are the 8 scales, then the 8 mins. */
            uint32_t u[4];
            memcpy(u, blk + 4, 12);
            const uint32_t k1 = 0x3f3f3f3f, k2 = 0x0f0f0f0f, k3 = 0x03030303;
            u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
            uint32_t mins = u[1] & k1;
            u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
            u[2] = mins;
            u[0] &= k1;
            __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)u)));
            __m512 dd = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(kq_h(blk)),
                                             _mm512_set1_ps(kq_h(blk + 2)));
            v = _mm512_mul_ps(dd, v);
            /* lanes 0 .. 7 to ds, 8 .. 15 to dm (masked stores) */
            _mm512_mask_storeu_ps(ds + 8 * b, 0x00ff, v);
            _mm512_mask_storeu_ps(dm + 8 * b - 8, 0xff00, v);
        }
    }
}

/* The scales of one row on one token: S gets the scale of each part (ds *
 * xs, in the order of the parts), and the return value is the term that
 * the product subtracts at the end: the mins of Q4_K and Q5_K (dm times
 * xm), 32 times the sums of Q6_K (ds times xm). kq_dot1 and the tiles both
 * use it, so they give the same bits. */
static float kq_prep(int type, int cols, const float *ds, const float *dm, const float *xs,
                     const float *xm, float *S)
{
    /* two lanes for each value of 8 (a scale of each 32 for 2 sums of 16) */
    const __m512i twice = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    if (type == KQ_Q6_K) {
        __m512 macc = _mm512_setzero_ps();
        for (int b = 0; b < cols / 256; ++b) {
            __m512 dv = _mm512_loadu_ps(ds + 16 * b);
            __m512 xv = _mm512_permutexvar_ps(twice, _mm512_maskz_loadu_ps(0xff, xs + 8 * b));
            _mm512_storeu_ps(S + 16 * b, _mm512_mul_ps(dv, xv));
            macc = _mm512_fmadd_ps(dv, _mm512_loadu_ps(xm + 16 * b), macc);
        }
        return 32.f * _mm512_reduce_add_ps(macc);
    }
    int np = cols / 32;
    for (int j = 0; j < np; j += 16) {
        __mmask16 m = np - j >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << (np - j)) - 1);
        _mm512_mask_storeu_ps(S + j, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, ds + j),
                                                      _mm512_maskz_loadu_ps(m, xs + j)));
    }
    if (type == KQ_Q8_0) {
        return 0.f;
    }
    __m512 macc = _mm512_setzero_ps();
    for (int j = 0; j < np; j += 8) {
        __mmask16 m = np - j >= 8 ? (__mmask16)0xffff : (__mmask16)((1u << (2 * (np - j))) - 1);
        __m512 dv = _mm512_permutexvar_ps(twice, _mm512_maskz_loadu_ps(np - j >= 8 ? (__mmask16)0xff : (__mmask16)((1u << (np - j)) - 1), dm + j));
        macc = _mm512_fmadd_ps(dv, _mm512_maskz_loadu_ps(m, xm + 2 * j), macc);
    }
    return _mm512_reduce_add_ps(macc);
}

/* The 4-bit (and fifth bit) values of part pair p of a Q4_K or Q5_K block,
 * in the lanes of kq_dot_q45k. */
static inline __attribute__((always_inline)) void kq_q45_values(const uint8_t *blk, int five, int p, __m512i *lo, __m512i *hi)
{
    const __m512i m4 = _mm512_set1_epi8(0x0f), one = _mm512_set1_epi8(1);
    const uint8_t *qs = blk + (five ? 48 : 16);
    __m512i wv = _mm512_loadu_si512((const void *)(qs + 64 * p));
    __m512i l = _mm512_and_si512(wv, m4), h = _mm512_and_si512(_mm512_srli_epi16(wv, 4), m4);
    if (five) {
        __m512i qhv = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 16)));
        __m512i c = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(4 * p),
                                            _mm512_set1_epi16(4 * p + 2));
        __m512i c1 = _mm512_add_epi16(c, _mm512_set1_epi16(1));
        l = _mm512_or_si512(l, _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(qhv, c), one), 4));
        h = _mm512_or_si512(h, _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(qhv, c1), one), 4));
    }
    *lo = l;
    *hi = h;
}

/* The 6-bit values of half h of a Q6_K block, in the lanes of kq_dot_q6k. */
static inline __attribute__((always_inline)) void kq_q6_values(const uint8_t *blk, int h, __m512i *A, __m512i *B)
{
    const __m512i m4 = _mm512_set1_epi8(0x0f), three = _mm512_set1_epi8(3);
    const __m512i cA = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(0),
                                               _mm512_set1_epi16(2));
    const __m512i cB = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(4),
                                               _mm512_set1_epi16(6));
    __m512i wv = _mm512_loadu_si512((const void *)(blk + 64 * h));
    __m512i qhv = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 128 + 32 * h)));
    *A = _mm512_or_si512(_mm512_and_si512(wv, m4), _mm512_slli_epi16(
        _mm512_and_si512(_mm512_srlv_epi16(qhv, cA), three), 4));
    *B = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(wv, 4), m4), _mm512_slli_epi16(
        _mm512_and_si512(_mm512_srlv_epi16(qhv, cB), three), 4));
}


/* The 5-bit values of two Q5_1 blocks (64 values in their order). The fifth
 * bits of the two blocks are the 64 bits of a mask. */
static inline __attribute__((always_inline)) __m512i kq_q51_values(const uint8_t *b0,
                                                                   const uint8_t *b1)
{
    const __m128i m4 = _mm_set1_epi8(0x0f);
    __m128i qa = _mm_loadu_si128((const __m128i *)(b0 + 8));
    __m128i qb = _mm_loadu_si128((const __m128i *)(b1 + 8));
    __m512i v = _mm512_castsi128_si512(_mm_and_si128(qa, m4));
    v = _mm512_inserti32x4(v, _mm_and_si128(_mm_srli_epi16(qa, 4), m4), 1);
    v = _mm512_inserti32x4(v, _mm_and_si128(qb, m4), 2);
    v = _mm512_inserti32x4(v, _mm_and_si128(_mm_srli_epi16(qb, 4), m4), 3);
    uint32_t ha, hb;
    memcpy(&ha, b0 + 4, 4);
    memcpy(&hb, b1 + 4, 4);
    __mmask64 hi = (__mmask64)ha | ((__mmask64)hb << 32);
    return _mm512_mask_or_epi32(v, 0xffff, v, _mm512_maskz_mov_epi8(hi, _mm512_set1_epi8(16)));
}

/* The products of one row on one token. S and corr come from kq_prep, as
 * in the tiles. */
static float kq_dot_q5_1(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < cols / 32; i += 2) {
        const uint8_t *b0 = w + (size_t)i * 24;
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), kq_q51_values(b0, b0 + 24),
                                         _mm512_loadu_si512((const void *)(xq + (size_t)i * 32)));
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(S[i]), _mm512_set1_ps(S[i + 1]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static float kq_dot_q8_0(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < cols / 32; i += 2) {
        const uint8_t *b0 = w + (size_t)i * 34, *b1 = b0 + 34;
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b1 + 2));
        __m512i xv = _mm512_loadu_si512((const void *)(xq + (size_t)i * 32));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(wv), sx);
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(S[i]), _mm512_set1_ps(S[i + 1]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static float kq_dot_q45k(const uint8_t *w, int cols, int five, const int8_t *xq, const float *S)
{
    size_t bs = five ? 176 : 144;
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
        const float *f = S + 8 * b;
        for (int p = 0; p < 2; ++p) {
            /* 64 bytes: parts 4p (low 4 bits of the first 32 bytes), 4p + 1
             * (their high 4 bits), 4p + 2 and 4p + 3 (the next 32 bytes). */
            __m512i lo, hi;
            kq_q45_values(w + (size_t)b * bs, five, p, &lo, &hi);
            const int8_t *xb = xq + (size_t)b * 256 + 128 * p;
            __m512i ia = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, kq_two256(xb, xb + 64));
            __m512i ib = _mm512_dpbusd_epi32(_mm512_setzero_si512(), hi,
                                             kq_two256(xb + 32, xb + 96));
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia),
                                  _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(f[4 * p]),
                                                       _mm512_set1_ps(f[4 * p + 2])), acc);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ib),
                                  _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(f[4 * p + 1]),
                                                       _mm512_set1_ps(f[4 * p + 3])), acc);
        }
    }
    return _mm512_reduce_add_ps(acc);
}

/* RECLANK_KQ_PF: software prefetch this many bytes ahead in the row loops
 * of the single-token products of Q6_K and Q8_0 (the cold experts of a
 * decode step; 0: none). A thread of the CPU part read about 5 to 6 GB/s
 * (the latency of memory, not the bandwidth of the node); 1024 bytes ahead:
 * the cold experts of a step on 24 threads 484 -> 409 us (Q8_0), 408 -> 360
 * (RQ6_MIX_PLAN.md M1); on 32, 453 -> 430 and 390 -> 357. Only loads: the
 * same bits. */
static int kq_pf_bytes(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_KQ_PF");
        n = v ? atoi(v) : 1024;
    }
    return n;
}

static float kq_dot_q6k(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    const int pf = kq_pf_bytes();
    /* lane i of a product of 64 values belongs to the part of 16 i / 4 */
    const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i iB = _mm512_add_epi32(iA, _mm512_set1_epi32(4));
    const __m512i i8 = _mm512_set1_epi32(8);
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
        if (pf) {
            const char *p = (const char *)w + (size_t)b * 210 + pf;
            _mm_prefetch(p, _MM_HINT_T0);
            _mm_prefetch(p + 64, _MM_HINT_T0);
            _mm_prefetch(p + 128, _MM_HINT_T0);
            _mm_prefetch(p + 192, _MM_HINT_T0);
        }
        __m512 fv = _mm512_loadu_ps(S + 16 * b);
        for (int h = 0; h < 2; ++h) {
            __m512i A, B;
            kq_q6_values(w + (size_t)b * 210, h, &A, &B);
            const int8_t *xb = xq + (size_t)b * 256 + 128 * h;
            __m512i pa = _mm512_dpbusd_epi32(_mm512_setzero_si512(), A,
                                             _mm512_loadu_si512((const void *)xb));
            __m512i pb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), B,
                                             _mm512_loadu_si512((const void *)(xb + 64)));
            __m512i ha = h ? _mm512_add_epi32(iA, i8) : iA, hb = h ? _mm512_add_epi32(iB, i8) : iB;
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pa), _mm512_permutexvar_ps(ha, fv), acc);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pb), _mm512_permutexvar_ps(hb, fv), acc);
        }
    }
    return _mm512_reduce_add_ps(acc);
}

/* The most parts of a row (a part is 32 values, 16 for Q6_K), and a pad of
 * 16. 1024 parts take the down rows of the E4B and the E2B in Q6_K (10240
 * and 12288 values); with 512 they took a slow path (4.4 GB/s, not 40). */
#define KQ_S 1040

/* A tile of 4 rows (wr, with their kq_row_scales ds and dm) by 4 tokens (the prompt pass, a group of an
 * expert), for the formats with int8 x. Each (row, token) pair adds in the
 * order of kq_dot1, so the tile gives the same bits. A short tile repeats
 * its last token. out[i * ostr_r + j * ostr_t] gets row i, token j. */
static void kq_tile4(const uint8_t *const wr[4], float ds[4][KQ_S], float dm[4][KQ_S],
                     int type, int cols, const int8_t *xq, const float *xs, const float *xm,
                     int nt, float *out, size_t ostr_r, size_t ostr_t)
{
    const int8_t *xr[4];
    float S[4][4][KQ_S];
    float corr[4][4];
    for (int j = 0; j < 4; ++j) {
        int jj = j < nt ? j : nt - 1;
        xr[j] = xq + (size_t)jj * cols;
        for (int i = 0; i < 4; ++i) {
            corr[i][j] = kq_prep(type, cols, ds[i], dm[i], xs + (size_t)jj * (cols / 32),
                                 xm + (size_t)jj * (cols / 16), S[i][j]);
        }
    }
    __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
    __m512 a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512 a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    __m512 a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    const __m512i z = _mm512_setzero_si512();
/* acc += (w . x) * the scales of S[I][J] at base, permuted by idx */
#define KQ_ACC(A, I, J, W, X, IDX, BASE) \
    A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(z, W, X)), \
                        _mm512_permutexvar_ps(IDX, _mm512_loadu_ps(S[I][J] + (BASE))), A)
    if (type == KQ_Q8_0) {
        for (int q = 0; q < cols / 64; ++q) {
            int base = (2 * q) & ~15, o = (2 * q) & 15;
            __m512i idx = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o),
                                                  _mm512_set1_epi32(o + 1));
            __m512i w0, w1, w2, w3;
            __mmask64 n0, n1, n2, n3;
#define KQ8_W(I) { const uint8_t *b0 = wr[I] + (size_t)q * 68; \
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36)); \
        n##I = _mm512_movepi8_mask(wv); w##I = _mm512_abs_epi8(wv); }
            KQ8_W(0) KQ8_W(1) KQ8_W(2) KQ8_W(3)
#undef KQ8_W
#define KQ8_J(J, A0, A1, A2, A3) { \
        __m512i xv = _mm512_loadu_si512((const void *)(xr[J] + (size_t)q * 64)); \
        KQ_ACC(A0, 0, J, w0, _mm512_mask_sub_epi8(xv, n0, z, xv), idx, base); \
        KQ_ACC(A1, 1, J, w1, _mm512_mask_sub_epi8(xv, n1, z, xv), idx, base); \
        KQ_ACC(A2, 2, J, w2, _mm512_mask_sub_epi8(xv, n2, z, xv), idx, base); \
        KQ_ACC(A3, 3, J, w3, _mm512_mask_sub_epi8(xv, n3, z, xv), idx, base); }
            KQ8_J(0, a00, a10, a20, a30) KQ8_J(1, a01, a11, a21, a31)
            KQ8_J(2, a02, a12, a22, a32) KQ8_J(3, a03, a13, a23, a33)
#undef KQ8_J
        }
    } else if (type == KQ_Q5_1) {
        for (int q = 0; q < cols / 64; ++q) {
            int base = (2 * q) & ~15, o = (2 * q) & 15;
            __m512i idx = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o),
                                                  _mm512_set1_epi32(o + 1));
            __m512i w0 = kq_q51_values(wr[0] + (size_t)q * 48, wr[0] + (size_t)q * 48 + 24);
            __m512i w1 = kq_q51_values(wr[1] + (size_t)q * 48, wr[1] + (size_t)q * 48 + 24);
            __m512i w2 = kq_q51_values(wr[2] + (size_t)q * 48, wr[2] + (size_t)q * 48 + 24);
            __m512i w3 = kq_q51_values(wr[3] + (size_t)q * 48, wr[3] + (size_t)q * 48 + 24);
#define KQ51_J(J, A0, A1, A2, A3) { \
        __m512i xv = _mm512_loadu_si512((const void *)(xr[J] + (size_t)q * 64)); \
        KQ_ACC(A0, 0, J, w0, xv, idx, base); KQ_ACC(A1, 1, J, w1, xv, idx, base); \
        KQ_ACC(A2, 2, J, w2, xv, idx, base); KQ_ACC(A3, 3, J, w3, xv, idx, base); }
            KQ51_J(0, a00, a10, a20, a30) KQ51_J(1, a01, a11, a21, a31)
            KQ51_J(2, a02, a12, a22, a32) KQ51_J(3, a03, a13, a23, a33)
#undef KQ51_J
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        int five = type == KQ_Q5_K;
        size_t bs = five ? 176 : 144;
        for (int b = 0; b < cols / 256; ++b) {
            int base = (8 * b) & ~15, o = (8 * b) & 15;
            for (int p = 0; p < 2; ++p) {
                __m512i ia = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o + 4 * p),
                                                     _mm512_set1_epi32(o + 4 * p + 2));
                __m512i ib = _mm512_add_epi32(ia, _mm512_set1_epi32(1));
                __m512i l0, h0, l1, h1, l2, h2, l3, h3;
                kq_q45_values(wr[0] + (size_t)b * bs, five, p, &l0, &h0);
                kq_q45_values(wr[1] + (size_t)b * bs, five, p, &l1, &h1);
                kq_q45_values(wr[2] + (size_t)b * bs, five, p, &l2, &h2);
                kq_q45_values(wr[3] + (size_t)b * bs, five, p, &l3, &h3);
#define KQ4_J(J, A0, A1, A2, A3) { \
        const int8_t *xb = xr[J] + (size_t)b * 256 + 128 * p; \
        __m512i xa = kq_two256(xb, xb + 64), xc = kq_two256(xb + 32, xb + 96); \
        KQ_ACC(A0, 0, J, l0, xa, ia, base); KQ_ACC(A0, 0, J, h0, xc, ib, base); \
        KQ_ACC(A1, 1, J, l1, xa, ia, base); KQ_ACC(A1, 1, J, h1, xc, ib, base); \
        KQ_ACC(A2, 2, J, l2, xa, ia, base); KQ_ACC(A2, 2, J, h2, xc, ib, base); \
        KQ_ACC(A3, 3, J, l3, xa, ia, base); KQ_ACC(A3, 3, J, h3, xc, ib, base); }
                KQ4_J(0, a00, a10, a20, a30) KQ4_J(1, a01, a11, a21, a31)
                KQ4_J(2, a02, a12, a22, a32) KQ4_J(3, a03, a13, a23, a33)
#undef KQ4_J
            }
        }
    } else {
        const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
        for (int b = 0; b < cols / 256; ++b) {
            for (int h = 0; h < 2; ++h) {
                __m512i ha = _mm512_add_epi32(iA, _mm512_set1_epi32(8 * h));
                __m512i hb = _mm512_add_epi32(ha, _mm512_set1_epi32(4));
                __m512i A0, B0, A1, B1, A2, B2, A3, B3;
                kq_q6_values(wr[0] + (size_t)b * 210, h, &A0, &B0);
                kq_q6_values(wr[1] + (size_t)b * 210, h, &A1, &B1);
                kq_q6_values(wr[2] + (size_t)b * 210, h, &A2, &B2);
                kq_q6_values(wr[3] + (size_t)b * 210, h, &A3, &B3);
#define KQ6_J(J, C0, C1, C2, C3) { \
        const int8_t *xb = xr[J] + (size_t)b * 256 + 128 * h; \
        __m512i x0 = _mm512_loadu_si512((const void *)xb); \
        __m512i x1 = _mm512_loadu_si512((const void *)(xb + 64)); \
        KQ_ACC(C0, 0, J, A0, x0, ha, 16 * b); KQ_ACC(C0, 0, J, B0, x1, hb, 16 * b); \
        KQ_ACC(C1, 1, J, A1, x0, ha, 16 * b); KQ_ACC(C1, 1, J, B1, x1, hb, 16 * b); \
        KQ_ACC(C2, 2, J, A2, x0, ha, 16 * b); KQ_ACC(C2, 2, J, B2, x1, hb, 16 * b); \
        KQ_ACC(C3, 3, J, A3, x0, ha, 16 * b); KQ_ACC(C3, 3, J, B3, x1, hb, 16 * b); }
                KQ6_J(0, a00, a10, a20, a30) KQ6_J(1, a01, a11, a21, a31)
                KQ6_J(2, a02, a12, a22, a32) KQ6_J(3, a03, a13, a23, a33)
#undef KQ6_J
            }
        }
    }
#undef KQ_ACC
    __m512 acc[4][4] = {{a00, a01, a02, a03}, {a10, a11, a12, a13},
                        {a20, a21, a22, a23}, {a30, a31, a32, a33}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < nt; ++j) {
            out[(size_t)i * ostr_r + (size_t)j * ostr_t] =
                _mm512_reduce_add_ps(acc[i][j]) - corr[i][j];
        }
    }
}

/* The tiles apply to a format with int8 x and a row of at most KQ_S - 16
 * parts. */
static inline int kq_tiles(int type, int cols)
{
    int parts = type == KQ_Q6_K ? cols / 16 : cols / 32;
    return type != KQ_F32 && type != KQ_IQ4_NL && type != KQ_BF16 && type != KQ_NV4 &&
           type != KQ_BF12 && parts <= KQ_S - 16 &&
           (!kq_block32(type) || cols % 64 == 0);
}
#endif

#if defined(__AVX512VNNI__)
/* One row on one token, with the scales of the row (kq_row_scales). */
static float kq_dot_scaled(const uint8_t *w, int type, int cols, const float *ds,
                           const float *dm, const int8_t *xq, const float *xs, const float *xm)
{
    float S[KQ_S];
    float corr = kq_prep(type, cols, ds, dm, xs, xm, S);
    switch (type) {
    case KQ_Q8_0: return kq_dot_q8_0(w, cols, xq, S) - corr;
    case KQ_Q5_1: return kq_dot_q5_1(w, cols, xq, S) - corr;
    case KQ_Q4_K: return kq_dot_q45k(w, cols, 0, xq, S) - corr;
    case KQ_Q5_K: return kq_dot_q45k(w, cols, 1, xq, S) - corr;
    }
    return kq_dot_q6k(w, cols, xq, S) - corr;
}
#endif

/* One row of type type on one token: xq, xs, xm (quantized), or x for F32. */
#if defined(__AVX512VNNI__)
/* BF16 row, int8 x: sum over the blocks of xs * sum(w * xq). */
static float kq_dot_bf16(const uint8_t *w, int cols, const int8_t *xq, const float *xs)
{
    const uint16_t *h = (const uint16_t *)w;
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 32; ++b) {
        __m512i hw = _mm512_loadu_si512((const void *)(h + 32 * b));
        __m512 w0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(hw)), 16));
        __m512 w1 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(hw, 1)), 16));
        __m128i xa = _mm_loadu_si128((const __m128i *)(xq + 32 * b));
        __m128i xb = _mm_loadu_si128((const __m128i *)(xq + 32 * b + 16));
        __m512 x0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(xa));
        __m512 x1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(xb));
        __m512 p = _mm512_fmadd_ps(w1, x1, _mm512_mul_ps(w0, x0));
        acc = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[b]), acc);
    }
    return _mm512_reduce_add_ps(acc);
}

/* NVFP4 row (KQ_NV4), int8 x. For each block: the codes to int8 (twice the
 * values), vpdpbusd with |w| and x with the sign of w (8 sums of 4 values:
 * the first 4 of values 0-15, the last 4 of 16-31), each half times its
 * scale and xs. */
static float kq_dot_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs)
{
    float g = kq_nv4_g(w, cols);
    const __m128i lut = _mm_loadu_si128((const __m128i *)kq_e2m1x2);
    const __m128i m4 = _mm_set1_epi8(15);
    const __m512i lut5 = _mm512_broadcast_i32x4(lut), m45 = _mm512_set1_epi8(15);
    /* the scale of each of the 16 sums: 4 for each half block */
    const __m512i sidx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    __m512 acc5 = _mm512_setzero_ps();
    int nb = cols / 32, b = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *s0p = kq_nv4_scales(w, cols, b);
        __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(w, b));
        /* lanes of 128 bits: the low codes of block 0, its high codes, the
         * low codes of block 1, its high codes (the order of x) */
        __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
        q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
        __m512i codes = _mm512_and_si512(
            _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
        __m512i wv = _mm512_shuffle_epi8(lut5, codes);
        __m512i xv = _mm512_loadu_si512((const void *)(xq + 32 * b));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(wv), sx);
        __m128 s4 = _mm_setr_ps(kq_e4m3_tab[s0p[0]] * xs[b], kq_e4m3_tab[s0p[1]] * xs[b],
                                kq_e4m3_tab[s0p[2]] * xs[b + 1], kq_e4m3_tab[s0p[3]] * xs[b + 1]);
        __m512 sc = _mm512_permutexvar_ps(sidx, _mm512_castps128_ps512(s4));
        acc5 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc5);
    }
    __m256 acc = _mm256_setzero_ps();
    for (; b < nb; ++b) {
        const uint8_t *sp = kq_nv4_scales(w, cols, b);
        __m128i q = _mm_loadu_si128((const __m128i *)kq_nv4_codes(w, b));
        __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
        __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
        __m256i wv = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + 32 * b));
        __m256i sx = _mm256_sign_epi8(xv, wv);
        __m256i is = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(wv), sx);
        float s0 = kq_e4m3_tab[sp[0]] * xs[b], s1 = kq_e4m3_tab[sp[1]] * xs[b];
        __m256 sc = _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is), sc, acc);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    return 0.5f * g * (_mm_cvtss_f32(r) + _mm512_reduce_add_ps(acc5));
}
#endif

#if defined(__AVX512VNNI__)
#define KQ_NV4_MAX 4096
static void kq_row_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs, int n,
                       float *out, size_t ostride);
#endif

#if defined(__AVX512F__)
/* A BF12 row on float x (plan-scripts/fmt_cpu.c bf12_dot): the bfloat16 bits
 * of each group of 32 in registers, then the fma of float32. */
static float kq_dot_bf12(const uint8_t *w, int cols, const float *x)
{
    const uint8_t *lo = w, *hi = w + cols, *E = w + cols + cols / 2;
    const __m512i m7f = _mm512_set1_epi16(0x7f), m80 = _mm512_set1_epi16(0x80);
    const __m512i g15 = _mm512_set1_epi16(15), m15 = _mm512_set1_epi16(15);
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int g = 0; g < cols / 32; ++g) {
        __m512i l = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(lo + 32 * g)));
        __m256i h16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(hi + 16 * g)));
        __m512i gp = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(h16, _mm512_castsi512_si256(m15))),
                                        _mm256_srli_epi16(h16, 4), 1);
        __m512i e = _mm512_sub_epi16(_mm512_set1_epi16(E[g]), gp);
        __m512i b = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(l, m80), 8),
                                                    _mm512_slli_epi16(e, 7)), _mm512_and_si512(l, m7f));
        b = _mm512_maskz_mov_epi16(~(_mm512_cmpeq_epi16_mask(gp, g15) & _mm512_cmpeq_epi16_mask(l, m80)), b);
        a0 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                 _mm512_castsi512_si256(b)), 16)), _mm512_loadu_ps(x + 32 * g), a0);
        a1 = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                 _mm512_extracti64x4_epi64(b, 1)), 16)), _mm512_loadu_ps(x + 32 * g + 16), a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}
#endif

static float kq_dot1(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                     const float *xm, const float *x)
{
    if (type == KQ_BF12) {
#if defined(__AVX512F__)
        return kq_dot_bf12(w, cols, x);
#else
        float s = 0.f;
        for (int c = 0; c < cols; ++c) {
            uint32_t u = (uint32_t)kq_bf12_bits(w, cols, c) << 16;
            float v;
            memcpy(&v, &u, 4);
            s += v * x[c];
        }
        return s;
#endif
    }
#if defined(__AVX512VNNI__)
    if (type == KQ_F32) {
        return kq_dot_f32((const float *)w, cols, x);
    }
    if (type == KQ_BF16) {
        return kq_dot_bf16(w, cols, xq, xs);
    }
    if (type == KQ_NV4) {
        /* With cols a multiple of 64, the sums and their order are those
         * of kq_row_nv4 (a group): the same bits. */
        return kq_dot_nv4(w, cols, xq, xs);
    }
    if (kq_tiles(type, cols)) {
        float ds[KQ_S], dm[KQ_S];
        kq_row_scales(w, type, cols, ds, dm);
        return kq_dot_scaled(w, type, cols, ds, dm, xq, xs, xm);
    }
#endif
#if defined(KQ_DOT2)
    {
        /* the integer products of AVX2 (dot2/avx2.c) for their formats */
        float r;
        if (kq_dot2(w, type, cols, xq, xs, xm, x, &r)) {
            return r;
        }
    }
#endif
    (void)xm;
    if (type == KQ_F32) {
        float s = 0.f;
        for (int c = 0; c < cols; ++c) {
            s += ((const float *)w)[c] * x[c];
        }
        return s;
    }
    int bv = kq_block32(type) ? 32 : 256;
    float v[256], s = 0.f;
    for (int b = 0; b < cols / bv; ++b) {
        kq_block_values(type, w, cols, b, v);
        for (int i = 0; i < bv; ++i) {
            int c = b * bv + i;
            s += v[i] * xs[c / 32] * (float)xq[c];
        }
    }
    return s;
}

/* One row on n tokens. out[j * ostride] gets token j. */
#if defined(__AVX512VNNI__)
/* NVFP4 on several tokens: the row to |w| (int8), its signs, and a scale for
 * each 16 values, one time; then for each token 64 values at a time. The
 * sums are those of kq_dot_nv4 (the same int32 sums of 16 values, the same
 * scales), so a group gives the bits of steps. */
static void kq_row_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs, int n,
                       float *out, size_t ostride)
{
    __attribute__((aligned(64))) int8_t aw[KQ_NV4_MAX];
    __attribute__((aligned(64))) float sc[KQ_NV4_MAX / 16];
    __mmask64 negs[KQ_NV4_MAX / 64];
    float g = kq_nv4_g(w, cols);
    int nb = cols / 32;
    const __m512i lut5 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
    const __m512i m45 = _mm512_set1_epi8(15);
    for (int b = 0; b < nb; b += 2) {
        /* as kq_dot_nv4: two blocks to 64 int8 in the order of x */
        const uint8_t *sp = kq_nv4_scales(w, cols, b);
        __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(w, b));
        __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
        q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
        __m512i codes = _mm512_and_si512(
            _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
        __m512i wv = _mm512_shuffle_epi8(lut5, codes);
        negs[b / 2] = _mm512_movepi8_mask(wv);
        _mm512_store_si512((void *)(aw + 32 * b), _mm512_abs_epi8(wv));
        sc[2 * b] = kq_e4m3_tab[sp[0]];
        sc[2 * b + 1] = kq_e4m3_tab[sp[1]];
        sc[2 * b + 2] = kq_e4m3_tab[sp[2]];
        sc[2 * b + 3] = kq_e4m3_tab[sp[3]];
    }
    const __m512i sidx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    for (int j = 0; j < n; ++j) {
        const int8_t *xr = xq + (size_t)j * cols;
        const float *sr = xs + (size_t)j * nb;
        __m512 acc = _mm512_setzero_ps();
        for (int c = 0; c < cols; c += 64) {
            __m512i xv = _mm512_loadu_si512((const void *)(xr + c));
            __m512i sx = _mm512_mask_sub_epi8(xv, negs[c / 64], _mm512_setzero_si512(), xv);
            __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                             _mm512_load_si512((const void *)(aw + c)), sx);
            int b = c / 32;
            __m128 s4 = _mm_setr_ps(sc[2 * b] * sr[b], sc[2 * b + 1] * sr[b],
                                    sc[2 * b + 2] * sr[b + 1], sc[2 * b + 3] * sr[b + 1]);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is),
                                  _mm512_permutexvar_ps(sidx, _mm512_castps128_ps512(s4)), acc);
        }
        out[(size_t)j * ostride] = 0.5f * g * _mm512_reduce_add_ps(acc);
    }
}
#endif

static void kq_row(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                   const float *xm, const float *x, int n, float *out, size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (type == KQ_NV4 && n > 1 && cols % 64 == 0 && cols <= KQ_NV4_MAX) {
        kq_row_nv4(w, cols, xq, xs, n, out, ostride);
        return;
    }
    if (n > 1 && type != KQ_F32 && kq_tiles(type, cols)) {
        /* The scales of the row one time for all the tokens. */
        float ds[KQ_S], dm[KQ_S];
        kq_row_scales(w, type, cols, ds, dm);
        for (int j = 0; j < n; ++j) {
            out[(size_t)j * ostride] = kq_dot_scaled(
                w, type, cols, ds, dm, xq + (size_t)j * cols, xs + (size_t)j * (cols / 32),
                xm + (size_t)j * (cols / 16));
        }
        return;
    }
#endif
    for (int j = 0; j < n; ++j) {
        out[(size_t)j * ostride] = kq_dot1(w, type, cols, xq + (size_t)j * cols,
                                           xs + (size_t)j * (cols / 32),
                                           xm + (size_t)j * (cols / 16),
                                           x ? x + (size_t)j * cols : NULL);
    }
}

#if defined(__AVX512VNNI__)
/* Four Q8_0 rows (at w, rb bytes each) on one token: the operations of
 * kq_dot1 for each row (kq_row_scales, kq_prep, kq_dot_q8_0: the same
 * bits), the rows in one loop with their own sums. One row at a time, a
 * row of 640 values (the down rows of the experts) ran at about 3.7 GB/s of
 * a thread: a chain of 10 steps of convert and FMA, and the passes of the
 * scales; four chains in step share the loads of x and fill the core. */
static void kq_dot4_q8_0(const uint8_t *w, size_t rb, int cols, const int8_t *xq, const float *xs,
                         float *out, size_t ostride)
{
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    const __m512i z = _mm512_setzero_si512();
    const uint8_t *w0 = w, *w1 = w + rb, *w2 = w + 2 * rb, *w3 = w + 3 * rb;
    const int pf = kq_pf_bytes();
    for (int i = 0; i < cols / 32; i += 2) {
        if (pf) {
            /* the 4 rows are one span of 4 rb bytes: 68 bytes of each row a step */
            _mm_prefetch((const char *)w0 + (size_t)i * 34 + pf, _MM_HINT_T0);
            _mm_prefetch((const char *)w1 + (size_t)i * 34 + pf, _MM_HINT_T0);
            _mm_prefetch((const char *)w2 + (size_t)i * 34 + pf, _MM_HINT_T0);
            _mm_prefetch((const char *)w3 + (size_t)i * 34 + pf, _MM_HINT_T0);
        }
        __m512i xv = _mm512_loadu_si512((const void *)(xq + (size_t)i * 32));
        float x0 = xs[i], x1 = xs[i + 1];
#define KQ4_ROW(A, W) { \
        const uint8_t *b0 = W + (size_t)i * 34, *b1 = b0 + 34; \
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b1 + 2)); \
        __m512i sx = _mm512_mask_sub_epi8(xv, _mm512_movepi8_mask(wv), z, xv); \
        __m512i is = _mm512_dpbusd_epi32(z, _mm512_abs_epi8(wv), sx); \
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(kq_h(b0) * x0), \
                                         _mm512_set1_ps(kq_h(b1) * x1)); \
        A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, A); }
        KQ4_ROW(a0, w0) KQ4_ROW(a1, w1) KQ4_ROW(a2, w2) KQ4_ROW(a3, w3)
#undef KQ4_ROW
    }
    out[0] = _mm512_reduce_add_ps(a0);
    out[ostride] = _mm512_reduce_add_ps(a1);
    out[2 * ostride] = _mm512_reduce_add_ps(a2);
    out[3 * ostride] = _mm512_reduce_add_ps(a3);
}
#endif

/* Rows r .. r + 3 (at w, rb bytes each) on n tokens: tiles of 4 tokens for
 * n >= 4, else one row at a time. out[i + j * ostride] gets row r + i,
 * token j. */
#if defined(__AVX512VNNI__)
/* NVFP4, rows r .. r + 3 on n tokens (the experts of a group): the 4 rows
 * to |w| (int8), the signs, and the E4M3 scales in the order of the 16 sums
 * of a chunk of 64 values, one time; then tiles of 4 rows by 4 tokens, so a
 * chunk of a row and of a token is loaded one time for 4 products. The sums,
 * the scales, and their order are those of kq_dot_nv4: the bits of steps. */
static void kq_rows4_nv4(const uint8_t *w, size_t rb, int cols, const int8_t *xq,
                         const float *xs, int n, float *out, size_t ostride)
{
    __attribute__((aligned(64))) int8_t aw[4][KQ_NV4_MAX];
    __attribute__((aligned(64))) float wsc[4][KQ_NV4_MAX / 64][16];
    __mmask64 negs[4][KQ_NV4_MAX / 64];
    float g[4];
    int nb = cols / 32, nc = cols / 64;
    const __m512i lut5 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
    const __m512i m45 = _mm512_set1_epi8(15);
    for (int r = 0; r < 4; ++r) {
        const uint8_t *wr = w + (size_t)r * rb;
        g[r] = kq_nv4_g(wr, cols);
        for (int b = 0; b < nb; b += 2) {
            const uint8_t *sp = kq_nv4_scales(wr, cols, b);
            __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(wr, b));
            __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
            q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
            __m512i codes = _mm512_and_si512(
                _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
            __m512i wv = _mm512_shuffle_epi8(lut5, codes);
            negs[r][b / 2] = _mm512_movepi8_mask(wv);
            _mm512_store_si512((void *)(aw[r] + 32 * b), _mm512_abs_epi8(wv));
            float s4[4] = {kq_e4m3_tab[sp[0]], kq_e4m3_tab[sp[1]], kq_e4m3_tab[sp[2]],
                           kq_e4m3_tab[sp[3]]};
            for (int l = 0; l < 16; ++l) {
                wsc[r][b / 2][l] = s4[l / 4];
            }
        }
    }
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        __m512 acc[4][4];
        for (int r = 0; r < 4; ++r) {
            for (int t = 0; t < 4; ++t) {
                acc[r][t] = _mm512_setzero_ps();
            }
        }
        for (int c = 0; c < nc; ++c) {
            __m512i xv[4];
            __m512 xsv[4];
            for (int t = 0; t < 4; ++t) {
                const float *sr = xs + (size_t)(j + t) * nb;
                xv[t] = _mm512_loadu_si512((const void *)(xq + (size_t)(j + t) * cols + 64 * c));
                xsv[t] = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sr[2 * c]),
                                              _mm512_set1_ps(sr[2 * c + 1]));
            }
            for (int r = 0; r < 4; ++r) {
                __m512i wa = _mm512_load_si512((const void *)(aw[r] + 64 * c));
                __m512 ws = _mm512_load_ps(wsc[r][c]);
                for (int t = 0; t < 4; ++t) {
                    __m512i sx = _mm512_mask_sub_epi8(xv[t], negs[r][c], _mm512_setzero_si512(), xv[t]);
                    __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wa, sx);
                    acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_mul_ps(ws, xsv[t]),
                                                acc[r][t]);
                }
            }
        }
        for (int r = 0; r < 4; ++r) {
            for (int t = 0; t < 4; ++t) {
                out[(size_t)(j + t) * ostride + r] = 0.5f * g[r] * _mm512_reduce_add_ps(acc[r][t]);
            }
        }
    }
    for (; j < n; ++j) {                    /* the last tokens: one at a time */
        for (int r = 0; r < 4; ++r) {
            out[(size_t)j * ostride + r] = kq_dot_nv4(w + (size_t)r * rb, cols, xq + (size_t)j * cols,
                                                      xs + (size_t)j * nb);
        }
    }
}
#endif
