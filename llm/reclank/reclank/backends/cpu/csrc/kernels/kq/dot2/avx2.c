/* ---------- one row of a GGUF format on one token, AVX2 ----------
 * The integer products of the K quants (Q4_K, Q5_K, Q6_K) and of Q8_0 on
 * the int8 x of kq_quant_part, 32 values at a time, for a CPU without AVX-512
 * VNNI (the tiers avx2, avx2_vnni, and avx512). Before, these formats ran the
 * float loop of kq_dot1 (kq_block_values) there.
 *
 * KQ2_DOT(acc, u, s) adds to the 8 int32 lanes of acc the sums of 4 products
 * of the unsigned bytes u and the signed bytes s: vpmaddubsw and vpmaddwd
 * (AVX2), or vpdpbusd (AVX-VNNI, dot2/avx2_vnni.c). The weights are
 * unsigned (Q4_K, Q5_K, Q6_K: q up to 63) or |q| with the sign moved to x
 * (Q8_0: up to 128), so no pair of vpmaddubsw saturates (2 * 128 * 127 =
 * 32512): both forms give the same lanes, and the same result.
 *
 * Each block adds its int32 lanes, as float32 times its scale (d, the 6-bit
 * scale, and the scale xs of x), to a vector of 8 sums; the mins of Q4_K and
 * Q5_K and the offset 32 of Q6_K use the sums of x (xm). The order of the
 * sums is not that of the float loop or of the VNNI kernels: the results are
 * close to them, not the same.
 */
#define KQ_DOT2 1

#ifndef KQ2_DOT
static inline __m256i kq2_dot(__m256i acc, __m256i u, __m256i s)
{
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s),
                                                   _mm256_set1_epi16(1)));
}
#define KQ2_DOT(acc, u, s) kq2_dot(acc, u, s)
#endif

static inline float kq2_hsum(__m256 v)
{
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

static inline void kq2_scale_min(const uint8_t *q, int j, int *sc, int *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

static float kq2_q8_0(const uint8_t *w, int cols, const int8_t *xq, const float *xs)
{
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < cols / 32; ++b) {
        const uint8_t *blk = w + (size_t)b * 34;
        __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 2));
        __m256i x = _mm256_loadu_si256((const __m256i *)(xq + 32 * b));
        __m256i s = KQ2_DOT(_mm256_setzero_si256(), _mm256_abs_epi8(q), _mm256_sign_epi8(x, q));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), _mm256_set1_ps(kq_h(blk) * xs[b]), acc);
    }
    return kq2_hsum(acc);
}

static float kq2_q45k(const uint8_t *w, int cols, int five, const int8_t *xq, const float *xs,
                      const float *xm)
{
    const __m256i m4 = _mm256_set1_epi8(15), one = _mm256_set1_epi8(1);
    __m256 acc = _mm256_setzero_ps();
    float mins = 0.f;
    for (int b = 0; b < cols / 256; ++b) {
        const uint8_t *blk = w + (size_t)b * (five ? 176 : 144);
        float d = kq_h(blk), dmin = kq_h(blk + 2);
        const uint8_t *qs = blk + (five ? 48 : 16);
        __m256i qh = five ? _mm256_loadu_si256((const __m256i *)(blk + 16)) : _mm256_setzero_si256();
        for (int c = 0; c < 4; ++c) {
            int s0, m0, s1, m1, j = 8 * b + 2 * c;
            kq2_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq2_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            __m256i q = _mm256_loadu_si256((const __m256i *)(qs + 32 * c));
            __m256i lo = _mm256_and_si256(q, m4);
            __m256i hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
            if (five) {
                /* bit 2c (lo) and 2c + 1 (hi) of each byte of qh: the fifth bit */
                __m256i b0 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * c), one);
                __m256i b1 = _mm256_and_si256(_mm256_srli_epi16(qh, 2 * c + 1), one);
                lo = _mm256_or_si256(lo, _mm256_slli_epi16(b0, 4));
                hi = _mm256_or_si256(hi, _mm256_slli_epi16(b1, 4));
            }
            const int8_t *x = xq + 256 * b + 64 * c;
            __m256i a = KQ2_DOT(_mm256_setzero_si256(), lo, _mm256_loadu_si256((const __m256i *)x));
            __m256i e = KQ2_DOT(_mm256_setzero_si256(), hi, _mm256_loadu_si256((const __m256i *)(x + 32)));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(a), _mm256_set1_ps(d * (float)s0 * xs[j]), acc);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(e), _mm256_set1_ps(d * (float)s1 * xs[j + 1]), acc);
            mins += dmin * ((float)m0 * (xm[2 * j] + xm[2 * j + 1]) +
                            (float)m1 * (xm[2 * j + 2] + xm[2 * j + 3]));
        }
    }
    return kq2_hsum(acc) - mins;
}

static float kq2_q6k(const uint8_t *w, int cols, const int8_t *xq, const float *xs, const float *xm)
{
    const __m256i m4 = _mm256_set1_epi8(15), m2 = _mm256_set1_epi8(3);
    __m256 acc = _mm256_setzero_ps();
    float off = 0.f;
    for (int b = 0; b < cols / 256; ++b) {
        const uint8_t *blk = w + (size_t)b * 210;
        float d = kq_h(blk + 208);
        const int8_t *sc = (const int8_t *)(blk + 192);
        for (int h = 0; h < 2; ++h) {
            const uint8_t *ql = blk + 64 * h, *qhp = blk + 128 + 32 * h;
            __m256i l0 = _mm256_loadu_si256((const __m256i *)ql);
            __m256i l1 = _mm256_loadu_si256((const __m256i *)(ql + 32));
            __m256i qh = _mm256_loadu_si256((const __m256i *)qhp);
            __m256i q[4];
            q[0] = _mm256_or_si256(_mm256_and_si256(l0, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(qh, m2), 4));
            q[1] = _mm256_or_si256(_mm256_and_si256(l1, m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 2), m2), 4));
            q[2] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l0, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 4), m2), 4));
            q[3] = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(l1, 4), m4),
                                   _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(qh, 6), m2), 4));
            for (int k = 0; k < 4; ++k) {
                /* 32 values: two groups of 16 with their own scale */
                int c0 = 256 * b + 128 * h + 32 * k;
                const int8_t *x = xq + c0;
                __m256i s = KQ2_DOT(_mm256_setzero_si256(), q[k], _mm256_loadu_si256((const __m256i *)x));
                float ga = d * (float)sc[8 * h + 2 * k], gb = d * (float)sc[8 * h + 2 * k + 1];
                __m256 scale = _mm256_mul_ps(_mm256_setr_ps(ga, ga, ga, ga, gb, gb, gb, gb),
                                             _mm256_set1_ps(xs[c0 / 32]));
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(s), scale, acc);
                off += ga * xm[c0 / 16] + gb * xm[c0 / 16 + 1];
            }
        }
    }
    return kq2_hsum(acc) - 32.f * off;
}

static float kq2_f32(const float *w, int cols, const float *x)
{
    __m256 acc = _mm256_setzero_ps();
    int c = 0;
    for (; c + 8 <= cols; c += 8) {
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(w + c), _mm256_loadu_ps(x + c), acc);
    }
    float s = kq2_hsum(acc);
    for (; c < cols; ++c) {
        s += w[c] * x[c];
    }
    return s;
}

/* One row on one token, when this file has a form for the type: *out gets
 * it and the return is 1; else 0 (kq_dot1 runs its float loop). */
static int kq_dot2(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                   const float *xm, const float *x, float *out)
{
    switch (type) {
    case KQ_Q8_0:
        *out = kq2_q8_0(w, cols, xq, xs);
        return 1;
    case KQ_Q4_K:
    case KQ_Q5_K:
        if (cols % 256) {
            return 0;
        }
        *out = kq2_q45k(w, cols, type == KQ_Q5_K, xq, xs, xm);
        return 1;
    case KQ_Q6_K:
        if (cols % 256) {
            return 0;
        }
        *out = kq2_q6k(w, cols, xq, xs, xm);
        return 1;
    case KQ_F32:
        if (x == NULL) {
            return 0;
        }
        *out = kq2_f32((const float *)w, cols, x);
        return 1;
    default:
        return 0;
    }
}
