/* The GGUF block formats of llama.cpp: products for one token, a small
 * group, and the experts of a MoE layer.
 *
 * This file is part of the cops library: bf16_linear.c includes it, and it
 * uses the helpers there (fp16_to_f32) and in moe.c.
 *
 * The formats (the numbers are the ggml types):
 *
 *     KQ_F32   0   float32 rows.
 *     KQ_Q8_0  8   blocks of 32: d (fp16), 32 int8. w = d * q.
 *     KQ_Q4_K 12   blocks of 256 (144 bytes): d, dmin (fp16), 12 bytes of
 *                  6-bit scales and mins for 8 parts of 32, 128 bytes of
 *                  4-bit values. w = d * sc * q - dmin * m.
 *     KQ_Q5_K 13   Q4_K with a fifth bit (32 more bytes, qh).
 *     KQ_Q6_K 14   blocks of 256 (210 bytes): 128 bytes of the low 4 bits,
 *                  64 bytes of the high 2 bits, 16 int8 scales (one for
 *                  each 16 values), d. w = d * sc * (q - 32).
 *
 * The products quantize x to int8 in its natural order, with one scale xs
 * for each 32 values and the sum xm of the int8 values of each 16 values
 * (kq_quant_x). The same x serves all the formats. For each part of a
 * block:
 *
 *     Q8_0:  y += d * xs * sum(q * xq)
 *     Q4_K:  y += d * sc * xs * sum(q * xq) - dmin * m * xs * sum(xq)
 *     Q6_K:  y += d * sc * xs * (sum(q * xq) - 32 * sum(xq))
 *
 * The values of the K formats are not negative, which is the operation of
 * the VNNI instruction vpdpbusd (u8 x s8 -> s32). Q8_0 is signed: the
 * product uses |q| and x with the sign of q.
 *
 * All the products of one row use one function for one token, so a token
 * gives the same bits alone and in a group (an MTP verify group).
 */
#define KQ_F32 0
#define KQ_Q8_0 8
#define KQ_Q4_K 12
#define KQ_Q5_K 13
#define KQ_Q6_K 14

/* The bytes of one row of cols values. */
static inline size_t kq_row_bytes(int type, int cols)
{
    switch (type) {
    case KQ_F32: return (size_t)cols * 4;
    case KQ_Q8_0: return (size_t)cols / 32 * 34;
    case KQ_Q4_K: return (size_t)cols / 256 * 144;
    case KQ_Q5_K: return (size_t)cols / 256 * 176;
    case KQ_Q6_K: return (size_t)cols / 256 * 210;
    }
    return 0;
}

static inline float kq_h(const uint8_t *p)
{
    uint16_t v = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
#if defined(__F16C__)
    return _cvtsh_ss(v);
#else
    return fp16_to_f32(v);
#endif
}

/* Quantize the part g (32 values) of one row of x. */
static void kq_quant_part(const float *xr, int g, int8_t *qr, float *xs, float *xm)
{
    const float *v = xr + g * 32;
    float m = 0.f;
    for (int i = 0; i < 32; ++i) {
        float a = fabsf(v[i]);
        m = a > m ? a : m;
    }
    float inv = m > 0.f ? 127.f / m : 0.f;
    xs[g] = m / 127.f;
    int32_t s0 = 0, s1 = 0;
    for (int i = 0; i < 32; ++i) {
        int q = (int)lrintf(v[i] * inv);
        int8_t qv = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
        qr[g * 32 + i] = qv;
        if (i < 16) {
            s0 += qv;
        } else {
            s1 += qv;
        }
    }
    xm[2 * g] = (float)s0;
    xm[2 * g + 1] = (float)s1;
}

/* Quantize t rows of x (cols % 32 == 0), inside a parallel region: xq (t x
 * cols), xs (t x cols / 32), xm (t x cols / 16). */
static void kq_quant_body(const float *x, int t, int cols, int8_t *xq, float *xs, float *xm)
{
    int np = cols / 32;
    #pragma omp for schedule(static)
    for (int x2 = 0; x2 < t * np; ++x2) {
        int r = x2 / np, g = x2 % np;
        kq_quant_part(x + (size_t)r * cols, g, xq + (size_t)r * cols, xs + (size_t)r * np,
                      xm + (size_t)r * 2 * np);
    }
}

void kq_quant_x(const float *x, int t, int cols, int8_t *xq, float *xs, float *xm)
{
    #pragma omp parallel
    kq_quant_body(x, t, cols, xq, xs, xm);
}

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
static void kq_block_values(int type, const uint8_t *row, int b, float *out)
{
    if (type == KQ_Q8_0) {
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
            int32_t sm[16];
            for (int j = 0; j < 8; ++j) {
                kq_scale_min(blk + 4, j, &sm[j], &sm[8 + j]);
            }
            __m512 v = _mm512_cvtepi32_ps(_mm512_loadu_si512((const void *)sm));
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
 * the product subtracts at the end (the mins of Q4_K and Q5_K, 32 times the
 * sums of Q6_K). kq_dot1 and the tiles both use it, so they give the same
 * bits. */
static float kq_prep(int type, int cols, const float *ds, const float *dm, const float *xs,
                     const float *xm, float *S)
{
    if (type == KQ_Q6_K) {
        /* one scale of x for each 2 parts of 16 */
        const __m512i half = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
        __m512 macc = _mm512_setzero_ps();
        for (int b = 0; b < cols / 256; ++b) {
            __m512 xv = _mm512_permutexvar_ps(half, _mm512_maskz_loadu_ps(0xff, xs + 8 * b));
            __m512 sv = _mm512_mul_ps(_mm512_loadu_ps(ds + 16 * b), xv);
            _mm512_storeu_ps(S + 16 * b, sv);
            macc = _mm512_fmadd_ps(sv, _mm512_loadu_ps(xm + 16 * b), macc);
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
    /* the mins: dm * xs * (the sum of the 32 values of x of the part) */
    const __m512i ev = _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    const __m512i od = _mm512_add_epi32(ev, _mm512_set1_epi32(1));
    __m512 macc = _mm512_setzero_ps();
    for (int j = 0; j < np; j += 16) {
        __mmask16 m = np - j >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << (np - j)) - 1);
        __mmask16 m0 = np - j >= 8 ? (__mmask16)0xffff : (__mmask16)((1u << (2 * (np - j))) - 1);
        __mmask16 m1 = np - j >= 16 ? (__mmask16)0xffff
                     : (np - j > 8 ? (__mmask16)((1u << (2 * (np - j) - 16)) - 1) : 0);
        __m512 a = _mm512_maskz_loadu_ps(m0, xm + 2 * j), c = _mm512_maskz_loadu_ps(m1, xm + 2 * j + 16);
        __m512 sum = _mm512_add_ps(_mm512_permutex2var_ps(a, ev, c), _mm512_permutex2var_ps(a, od, c));
        __m512 f = _mm512_mul_ps(_mm512_maskz_loadu_ps(m, dm + j), _mm512_maskz_loadu_ps(m, xs + j));
        macc = _mm512_fmadd_ps(f, sum, macc);
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


/* The products of one row on one token. S and corr come from kq_prep, as
 * in the tiles. */
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

static float kq_dot_q6k(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    /* lane i of a product of 64 values belongs to the part of 16 i / 4 */
    const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i iB = _mm512_add_epi32(iA, _mm512_set1_epi32(4));
    const __m512i i8 = _mm512_set1_epi32(8);
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
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

#define KQ_S 528           /* the most parts of a row, and a pad of 16 */

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
    int parts = type == KQ_Q8_0 ? cols / 32 : (type == KQ_Q6_K ? cols / 16 : cols / 32);
    return type != KQ_F32 && parts <= KQ_S - 16 && (type != KQ_Q8_0 || cols % 64 == 0);
}
#endif

/* One row of type type on one token: xq, xs, xm (quantized), or x for F32. */
static float kq_dot1(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                     const float *xm, const float *x)
{
#if defined(__AVX512VNNI__)
    if (type == KQ_F32) {
        return kq_dot_f32((const float *)w, cols, x);
    }
    if (kq_tiles(type, cols)) {
        float ds[KQ_S], dm[KQ_S], S[KQ_S];
        kq_row_scales(w, type, cols, ds, dm);
        float corr = kq_prep(type, cols, ds, dm, xs, xm, S);
        switch (type) {
        case KQ_Q8_0: return kq_dot_q8_0(w, cols, xq, S) - corr;
        case KQ_Q4_K: return kq_dot_q45k(w, cols, 0, xq, S) - corr;
        case KQ_Q5_K: return kq_dot_q45k(w, cols, 1, xq, S) - corr;
        case KQ_Q6_K: return kq_dot_q6k(w, cols, xq, S) - corr;
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
    int bv = type == KQ_Q8_0 ? 32 : 256;
    float v[256], s = 0.f;
    for (int b = 0; b < cols / bv; ++b) {
        kq_block_values(type, w, b, v);
        for (int i = 0; i < bv; ++i) {
            int c = b * bv + i;
            s += v[i] * xs[c / 32] * (float)xq[c];
        }
    }
    return s;
}

/* One row on n tokens. out[j * ostride] gets token j. */
static void kq_row(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                   const float *xm, const float *x, int n, float *out, size_t ostride)
{
    for (int j = 0; j < n; ++j) {
        out[(size_t)j * ostride] = kq_dot1(w, type, cols, xq + (size_t)j * cols,
                                           xs + (size_t)j * (cols / 32),
                                           xm + (size_t)j * (cols / 16),
                                           x ? x + (size_t)j * cols : NULL);
    }
}

/* Rows r .. r + 3 (at w, rb bytes each) on n tokens: tiles of 4 tokens for
 * n >= 4, else one row at a time. out[i + j * ostride] gets row r + i,
 * token j. */
static void kq_rows4(const uint8_t *w, size_t rb, int type, int cols, const int8_t *xq,
                     const float *xs, const float *xm, const float *x, int n, float *out,
                     size_t ostride)
{
#if defined(__AVX512VNNI__)
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

/* out (t x rows) = x W^T, inside a parallel region. x is the float input
 * (for F32), and xq, xs, xm its quantization (kq_quant_body). */
static void kq_linear_body(const uint8_t *w, int type, int rows, int cols, const int8_t *xq,
                           const float *xs, const float *xm, const float *x, int t, float *out)
{
    size_t rb = kq_row_bytes(type, cols);
#if defined(__AVX512VNNI__)
    if (t >= 4 && rows % 4 == 0 && kq_tiles(type, cols)) {
        /* Blocks of MA_TB tokens, as ma_linear_body. */
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
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; ++r) {
        kq_row(w + (size_t)r * rb, type, cols, xq, xs, xm, x, t, out + r, (size_t)rows);
    }
}

void kq_linear(const uint8_t *w, int type, int rows, int cols, const int8_t *xq, const float *xs,
               const float *xm, const float *x, int t, float *out)
{
    #pragma omp parallel
    kq_linear_body(w, type, rows, cols, xq, xs, xm, x, t, out);
}

/* The float values of rows ids of a matrix (the embeddings). */
void kq_rows(const uint8_t *w, int type, int cols, const int64_t *ids, int n, float *out)
{
    size_t rb = kq_row_bytes(type, cols);
    int bv = type == KQ_Q8_0 ? 32 : 256;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const uint8_t *row = w + (size_t)ids[i] * rb;
        if (type == KQ_F32) {
            memcpy(out + (size_t)i * cols, row, (size_t)cols * 4);
            continue;
        }
        for (int b = 0; b < cols / bv; ++b) {
            kq_block_values(type, row, b, out + (size_t)i * cols + (size_t)b * bv);
        }
    }
}

/* The size of the scratch of kq_moe_body, in bytes. */
size_t kq_moe_scratch(int t, int k, int experts, int hidden, int inner)
{
    size_t P = (size_t)t * (k + 1);
    size_t n = 0;
    n += (size_t)(experts + 2) * 4 * 3 + 64;           /* counts, starts, used */
    n += P * 4 * 2;                                    /* pair token, pair of */
    n += P * hidden + P * (hidden / 32) * 4 + P * (hidden / 16) * 4;
    n += P * 2 * inner * 4;                            /* gate and up */
    n += P * inner + P * (inner / 32) * 4 + P * (inner / 16) * 4;
    n += P * hidden * 4;                               /* down outputs */
    return n + 64 * 16;
}

/* One matrix of the MoE: data and type. */
typedef struct {
    const uint8_t *w;
    int type;
} kq_mat;

static inline kq_mat kq_mat_of(const int64_t *d)
{
    kq_mat m;
    m.w = (const uint8_t *)(intptr_t)d[0];
    m.type = (int)d[1];
    return m;
}

/* The experts of t tokens in the GGUF formats, as ma_moe_body. h is
 * quantized (hq, hs, hm; kq_quant_body). mats has 6 descriptors (w, type):
 * gate, up, down of the stacked experts, then of the shared expert (w = 0
 * for none). The weight of the shared expert is sigmoid(shared_logit). */
static void kq_moe_body(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                        const float *val, int t, int k, int experts, const int64_t *mats,
                        const float *shared_logit, int hidden, int inner, uint8_t *scratch,
                        float *out)
{
    kq_mat G = kq_mat_of(mats), U = kq_mat_of(mats + 2), D = kq_mat_of(mats + 4);
    kq_mat SG = kq_mat_of(mats + 6), SU = kq_mat_of(mats + 8), SD = kq_mat_of(mats + 10);
    int shared = SG.w != NULL;
    int ne = experts + shared;
    int P = t * k + (shared ? t : 0);
    int nph = hidden / 32, npi = inner / 32;
    uint8_t *p = scratch;
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P * 4);
    int *nused_p = (int *)ma_take(&p, 64);
    int8_t *xq = (int8_t *)ma_take(&p, (size_t)P * hidden);
    float *xs = (float *)ma_take(&p, (size_t)P * nph * 4);
    float *xm = (float *)ma_take(&p, (size_t)P * nph * 2 * 4);
    float *act = (float *)ma_take(&p, (size_t)P * 2 * inner * 4);
    int8_t *aq = (int8_t *)ma_take(&p, (size_t)P * inner);
    float *as = (float *)ma_take(&p, (size_t)P * npi * 4);
    float *am = (float *)ma_take(&p, (size_t)P * npi * 2 * 4);
    float *de = (float *)ma_take(&p, (size_t)P * hidden * 4);
    moe_sort_pairs(ids, t, k, experts, shared, cnt, start, used, pair_tok, pair_of, nused_p);
    int nu = *nused_p;
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        int j = pair_tok[q];
        memcpy(xq + (size_t)q * hidden, hq + (size_t)j * hidden, (size_t)hidden);
        memcpy(xs + (size_t)q * nph, hs + (size_t)j * nph, (size_t)nph * 4);
        memcpy(xm + (size_t)q * 2 * nph, hm + (size_t)j * 2 * nph, (size_t)nph * 8);
    }
    /* Gate and up: 2 * inner rows of each used expert, in tasks of 4 rows. */
    #pragma omp for schedule(static)
    for (int x = 0; x < nu * 2 * inner / 4; ++x) {
        int e = used[x / (2 * inner / 4)], rr = (x % (2 * inner / 4)) * 4, isup = rr >= inner;
        int r = rr % inner;
        kq_mat m = e < experts ? (isup ? U : G) : (isup ? SU : SG);
        size_t rb = kq_row_bytes(m.type, hidden);
        const uint8_t *w = m.w + (e < experts ? (size_t)e * inner * rb : 0);
        int s0 = start[e];
        int n = (e + 1 < ne ? start[e + 1] : P) - s0;
        kq_rows4(w + (size_t)r * rb, rb, m.type, hidden, xq + (size_t)s0 * hidden,
                 xs + (size_t)s0 * nph, xm + (size_t)s0 * 2 * nph, NULL, n,
                 act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
    }
    /* silu(gate) * up, and its quantization for down. */
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        float *a = act + (size_t)q * 2 * inner;
        for (int i = 0; i < inner; ++i) {
            a[i] = ma_silu(a[i]) * a[inner + i];
        }
        for (int g = 0; g < npi; ++g) {
            kq_quant_part(a, g, aq + (size_t)q * inner, as + (size_t)q * npi,
                          am + (size_t)q * 2 * npi);
        }
    }
    /* Down: hidden rows of each used expert, in tasks of 4 rows. */
    #pragma omp for schedule(static)
    for (int x = 0; x < nu * hidden / 4; ++x) {
        int e = used[x / (hidden / 4)], r = (x % (hidden / 4)) * 4;
        kq_mat m = e < experts ? D : SD;
        size_t rb = kq_row_bytes(m.type, inner);
        const uint8_t *w = m.w + (e < experts ? (size_t)e * hidden * rb : 0);
        int s0 = start[e];
        int n = (e + 1 < ne ? start[e + 1] : P) - s0;
        kq_rows4(w + (size_t)r * rb, rb, m.type, inner, aq + (size_t)s0 * inner,
                 as + (size_t)s0 * npi, am + (size_t)s0 * 2 * npi, NULL, n,
                 de + (size_t)s0 * hidden + r, (size_t)hidden);
    }
    moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
}

void kq_moe(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
            const float *val, int t, int k, int experts, const int64_t *mats,
            const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out);
}
