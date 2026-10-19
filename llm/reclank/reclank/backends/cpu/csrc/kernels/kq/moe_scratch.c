/* The scratch of kq_e3_moe_body (kernels/kq/exl3_moe.c) for t tokens of k
 * experts and a shared expert, with the low planes of x16: the same takes,
 * each rounded up by ma_take. */
static size_t kq_e3_moe_bytes(int t, int k, int experts, int hidden, int inner)
{
    size_t P0 = (size_t)t * (k + 1), ne = (size_t)experts + 1;
    size_t nh = hidden / 128, ni = inner / 128;
    size_t n = (ne + 1) * 4 * 3 + P0 * 4 * 3 + 64 + 16 * 64;
    n += 2 * (P0 * (2 * hidden + inner) + P0 * (2 * nh + ni) * 8);
    n += P0 * (4 * inner + 2 * hidden) * 4 + P0 * (2 * inner + hidden) * 4;
    return n + 64 * 48;
}

/* The size of the scratch of kq_moe_body, in bytes. */
size_t rc_kq_moe_scratch(int t, int k, int experts, int hidden, int inner)
{
    size_t P = (size_t)t * (k + 1);
    size_t n = 0;
    n += (size_t)(experts + 2) * 4 * 3 + 64;           /* counts, starts, used */
    n += P * 4 * 2;                                    /* pair token, pair of */
    n += P * hidden + P * (hidden / 32) * 4 + P * (hidden / 16) * 4;
    n += P * 2 * inner * 4;                            /* gate and up */
    n += P * inner + P * (inner / 32) * 4 + P * (inner / 16) * 4;
    n += P * hidden * 4;                               /* down outputs */
    n += P * (hidden / 16) * 4 + P * (inner / 16) * 4;  /* the sums of KQ_NVX */
    n += P * (hidden / 16) * 4 + P * (inner / 16) * 4;  /* and of the low planes (x16) */
    n += P * hidden * 2 + P * (hidden / 32) * 4;       /* int16 h (act bit 2) */
    n += P * inner * 2 + P * (inner / 32) * 4;         /* int16 GELU */
    /* kq_moe_small_body: the counters of the experts, the int8 rows of the
     * tokens */
    n += (size_t)(experts + 2) * 4 * 2 + (size_t)t * hidden + (size_t)t * (hidden / 32) * 12 + 64 * 3;
    /* act bit 5 (x16): the low int8 planes of h, of the GELU, and of the
     * rows of the tokens (kq_quant_part16) */
    n += P * hidden + P * inner + (size_t)t * hidden + 64 * 3;
    n += (size_t)(experts + 2) * 4 + 64;               /* kq_moe_body: kq_used_split */
#if !defined(__AVX512VNNI__) && defined(__AVX2__)
    /* the planes of the int16 rows (AVX2, kq_q4x_rows_p16) */
    n += P * 2 * (hidden + inner) + P * ((hidden + inner) / 32) * 4 + 64 * 6;
#endif
    n += 64 * 20;
    /* the MUL1 experts (kq_e3_moe_body) take the same scratch */
    size_t e3 = kq_e3_moe_bytes(t, k, experts, hidden, inner);
    return n > e3 ? n : e3;
}

/* x16 (act bit 5 of kq_moe_body): x as int16 in each 32 values, v =
 * rint(x 16192 / max) (about -84 dB, against -45 for int8), split v = 128 hi
 * + lo with hi in [-126, 127] and lo in [-64, 63]: two int8 planes for the
 * VNNI kernels (kq_t2_tile; the integer sum of a step 128 sum(hi w) +
 * sum(lo w) is exact, so the bits do not depend on the kernel). xs gets
 * max / 16192, xm the two sums of 16 of v times xs (the int8 forms). */
static void kq_quant_part16(const float *xr, int g, int8_t *hi, int8_t *lo, float *xs, float *xm)
{
    const float *v = xr + 32 * g;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    /* The same values as the loops below (cvtps rounds to the nearest even,
     * as lrintf in the default mode; srai is the >> of the loop). The
     * scalar loops took 39 us of the 58 of the single region of a decode
     * step's CPU part (one token, 2560 values) with the clocks down. */
    {
        __m512 a = _mm512_loadu_ps(v), b = _mm512_loadu_ps(v + 16);
        float mv = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a), _mm512_abs_ps(b)));
        float iv = mv > 0.f ? 16192.f / mv : 0.f;
        xs[g] = mv / 16192.f;
        const __m512i qlo = _mm512_set1_epi32(-16192), qhi = _mm512_set1_epi32(16192);
        __m512i qa = _mm512_min_epi32(qhi, _mm512_max_epi32(qlo, _mm512_cvtps_epi32(
            _mm512_mul_ps(a, _mm512_set1_ps(iv)))));
        __m512i qb = _mm512_min_epi32(qhi, _mm512_max_epi32(qlo, _mm512_cvtps_epi32(
            _mm512_mul_ps(b, _mm512_set1_ps(iv)))));
        const __m512i c64 = _mm512_set1_epi32(64);
        __m512i ha = _mm512_srai_epi32(_mm512_add_epi32(qa, c64), 7);
        __m512i hb = _mm512_srai_epi32(_mm512_add_epi32(qb, c64), 7);
        __m512i la = _mm512_sub_epi32(qa, _mm512_slli_epi32(ha, 7));
        __m512i lb = _mm512_sub_epi32(qb, _mm512_slli_epi32(hb, 7));
        _mm_storeu_si128((__m128i *)(hi + 32 * g), _mm512_cvtepi32_epi8(ha));
        _mm_storeu_si128((__m128i *)(hi + 32 * g + 16), _mm512_cvtepi32_epi8(hb));
        _mm_storeu_si128((__m128i *)(lo + 32 * g), _mm512_cvtepi32_epi8(la));
        _mm_storeu_si128((__m128i *)(lo + 32 * g + 16), _mm512_cvtepi32_epi8(lb));
        xm[2 * g] = xs[g] * (float)_mm512_reduce_add_epi32(qa);
        xm[2 * g + 1] = xs[g] * (float)_mm512_reduce_add_epi32(qb);
        return;
    }
#endif
    float m = 0.f;
    for (int i = 0; i < 32; ++i) {
        m = fmaxf(m, fabsf(v[i]));
    }
    float iv = m > 0.f ? 16192.f / m : 0.f;
    xs[g] = m / 16192.f;
    int s0 = 0, s1 = 0;
    for (int i = 0; i < 32; ++i) {
        int q = (int)lrintf(v[i] * iv);
        q = q > 16192 ? 16192 : q < -16192 ? -16192 : q;
        int h = (q + 64) >> 7;
        hi[32 * g + i] = (int8_t)h;
        lo[32 * g + i] = (int8_t)(q - 128 * h);
        if (i < 16) {
            s0 += q;
        } else {
            s1 += q;
        }
    }
    xm[2 * g] = xs[g] * (float)s0;
    xm[2 * g + 1] = xs[g] * (float)s1;
}

/* The int16 rows made 8-bit (a measure of 8-bit activations, flag 256 of
 * GP_KQ_MOE_PART): q = 128 hi + lo becomes 128 rint(q / 128), so hi holds the
 * 8-bit value (at most 127: |q| <= 16192) and lo is 0; the scale stays. The
 * GPU rounds the same way (k_quant_x2r), so the bits do not depend on where
 * an expert runs. The KQ_MXX kernels then take the high plane alone (xl
 * NULL: kq_mxx_rows1). */
static void kq_round8(int8_t *hi, int8_t *lo, int n)
{
    /* rint(q / 128), ties to even, in integers (the loop vectorizes) */
    for (int i = 0; i < n; ++i) {
        int q = 128 * hi[i] + lo[i], r = q >> 7, m = q & 127;
        hi[i] = (int8_t)(r + (m > 64 || (m == 64 && (r & 1))));
        lo[i] = 0;
    }
}

static void kq_quant_row16(const float *xr, int cols, int8_t *hi, int8_t *lo, float *xs, float *xm)
{
    for (int g = 0; g < cols / 32; ++g) {
        kq_quant_part16(xr, g, hi, lo, xs, xm);
    }
}

/* The types and shapes that x16 takes (kq_t2_tile). */
static int kq_x16_ok(int type, int cols)
{
#if defined(__AVX512VNNI__)
    return (type == KQ_Q8_0 || (type == KQ_Q6_K && cols % 256 == 0)) && cols % 64 == 0 &&
           cols <= 64 * KQ_T2_NV;
#else
    (void)type;
    (void)cols;
    return 0;
#endif
}

/* The types that x16 takes in the experts of the KQ formats (kq_rows_n2):
 * those of kq_x16_ok (the tiles of kq_t2_tile), and KQ_NVX (kq_nvx_rows2,
 * a row group of 16 at a time). */
static void kq_mxx_rows2(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                         const int32_t *xn, const int8_t *xl, int n, float *out, size_t ostride);
static void kq_mxx_rows32(const uint8_t *wg, int cols, const int8_t *xq, const float *xs, const int32_t *xn,
                          const int8_t *xl, int n, float *out, size_t ostride);

static int kq_x16s_ok(int type, int cols)
{
    if (type == KQ_MXX) {
        return cols % 32 == 0;          /* KQ_MXX: int16 x only (kq_mxx_rows2), on every tier */
    }
#if defined(__AVX512VNNI__)
    return kq_x16_ok(type, cols) || (type == KQ_NVX && cols % 32 == 0);
#else
    return kq_x16_ok(type, cols);
#endif
}

/* rows rows (a multiple of 16) from row r of a matrix on n tokens, as
 * kq_rows4: KQ_NVX in groups of 16 (xn: kq_nvx_xsum of the tokens), the
 * other types 4 rows at a time. */
static void kq_rows_n(const uint8_t *w, size_t rb, int type, int nr, int cols, const int8_t *xq,
                      const float *xs, const float *xm, const int32_t *xn, int n, float *out,
                      size_t ostride)
{
    for (int i = 0; i < nr; i += type == KQ_NVX || type == KQ_Q4X ? 16 : 4) {
        if (type == KQ_NVX) {
            kq_nvx_rows(w + (size_t)i * rb, cols, xq, xs, xn, n, out + i, ostride);
        } else if (type == KQ_Q4X) {
            kq_q4x_rows(w + (size_t)i * rb, cols, xq, xs, xn, n, out + i, ostride);
        } else {
            kq_rows4(w + (size_t)i * rb, rb, type, cols, xq, xs, xm, NULL, n, out + i, ostride);
        }
    }
}

/* kq_rows_n, or (xl not null) the rows on x as int16 (xq the high plane, xl
 * the low one, xs and xm of kq_quant_part16; kq_x16s_ok types): kq_rows4_t2
 * (KQ_NVX: kq_nvx_rows2, xn the sums of both planes)
 * for any count of tokens, so the bits of a token do not depend on the
 * count. */
static void kq_rows_n2(const uint8_t *w, size_t rb, int type, int nr, int cols, const int8_t *xq,
                       const float *xs, const float *xm, const int32_t *xn, const int8_t *xl, int n,
                       float *out, size_t ostride)
{
    if (type == KQ_MXX) {
        /* xn: the sums of both planes of each token (kq_mxx_rows2); two groups at
         * a time where the rows allow (kq_mxx_rows32) */
        int i = 0;
        for (; i + 32 <= nr; i += 32) {
            kq_mxx_rows32(w + (size_t)i * rb, cols, xq, xs, xn, xl, n, out + i, ostride);
        }
        for (; i < nr; i += 16) {
            kq_mxx_rows2(w + (size_t)i * rb, cols, xq, xs, xn, xl, n, out + i, ostride);
        }
        return;
    }
#if defined(__AVX512VNNI__)
    if (xl != NULL) {
        if (type == KQ_NVX) {
            /* xn: the sums of both planes of each token (kq_nvx_rows2) */
            for (int i = 0; i < nr; i += 16) {
                kq_nvx_rows2(w + (size_t)i * rb, cols, xq, xs, xn, xl, n, out + i, ostride);
            }
            return;
        }
        if (n == 1 && type == KQ_Q8_0 && cols % 64 == 0 && kq_t1_on()) {
            for (int i = 0; i < nr; i += 4) {
                kq_rows4_q8_x16_t1(w + (size_t)i * rb, rb, cols, xq, xs, xl, out + i,
                                   kq_pf_bytes() ? w + (size_t)(i + 4) * rb : NULL);
            }
            return;
        }
        if (n == 1 && type == KQ_Q6_K && cols % 256 == 0 && kq_t1_on()) {
            for (int i = 0; i < nr; i += 4) {
                kq_rows4_q6_x16_t1(w + (size_t)i * rb, rb, cols, xq, xs, xm, xl, out + i,
                                   kq_pf_bytes() ? w + (size_t)(i + 4) * rb : NULL);
            }
            return;
        }
        for (int i = 0; i < nr; i += 4) {
            kq_rows4_t2(w + (size_t)i * rb, rb, type, cols, xq, xs, xm, n, out + i, ostride, xl);
        }
        return;
    }
#endif
    kq_rows_n(w, rb, type, nr, cols, xq, xs, xm, xn, n, out, ostride);
}
