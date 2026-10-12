/* KQ_NVX: -12 times the sum of each 16 int8 values of n rows of x (the
 * term of the codes + 12 of kq_nvx_rows). */
static void kq_nvx_xsum(const int8_t *xq, int n, int cols, int32_t *xn)
{
    for (int j = 0; j < n; ++j) {
        for (int h = 0; h < cols / 16; ++h) {
            const int8_t *p = xq + (size_t)j * cols + 16 * h;
            int32_t sum = 0;
            for (int u = 0; u < 16; ++u) {
                sum += p[u];
            }
            xn[(size_t)j * (cols / 16) + h] = -12 * sum;
        }
    }
}

#if defined(__AVX512VNNI__)
/* E2M1 codes to twice their values plus 12 (0 to 24: the unsigned operand
 * of vpdpbusd). */
/* 16 tokens for each decode of a block (8: 1.5% slower at 512 tokens, 7% at
 * 2048; the experts of a prompt are near the rate of the memory). */
#define KQ_NVX_TB 16
static const uint8_t kq_e2m1u[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};

/* 16 E4M3 scales as float32 times 2^-8: the bits go to float16 as they are
 * (the exponent bias of float16 is 8 more), exact for all the codes. */
static inline __m512 kq_e4m3x16(const uint8_t *p)
{
    __m256i b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p));
    __m256i h = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x7f)), 7),
                                _mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x80)), 8));
    return _mm512_cvtph_ps(h);
}

/* The 16 rows of the KQ_NVX group at wg on n tokens: out[j * ostride + r]
 * gets row r, token j. xn: kq_nvx_xsum of the tokens. For each block, 8
 * tokens at a time: the 8 steps to u8 (codes + 12) in registers; for each
 * token, 4 vpdpbusd for each half (a lane is a row; x broadcast, signed),
 * from -12 sum(x), which removes the 12. A token alone and in a group has
 * the same operations: the same bits. */
static void kq_nvx_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32, nh = cols / 16;
    float g;
    memcpy(&g, wg, 4);
    const __m512 gs = _mm512_set1_ps(128.f * g);        /* 0.5 g, and 2^8 of kq_e4m3x16 */
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1u));
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int j = 0; j < n; j += KQ_NVX_TB) {
        int nt = n - j < KQ_NVX_TB ? n - j : KQ_NVX_TB;
        __m512 f[KQ_NVX_TB];
        for (int tt = 0; tt < KQ_NVX_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
                __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                W[k] = _mm512_shuffle_epi8(lut, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
            }
            __m512 s0 = kq_e4m3x16(blk + 256), s1 = kq_e4m3x16(blk + 272);
            for (int tt = 0; tt < nt; ++tt) {
                const int8_t *xb = xq + (size_t)(j + tt) * cols + 32 * b;
                const int32_t *nx = xn + (size_t)(j + tt) * nh + 2 * b;
                __m512i a0 = _mm512_set1_epi32(nx[0]), a1 = _mm512_set1_epi32(nx[1]);
                for (int k = 0; k < 4; ++k) {
                    int32_t x0, x1;
                    memcpy(&x0, xb + 4 * k, 4);
                    memcpy(&x1, xb + 16 + 4 * k, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    a1 = _mm512_dpbusd_epi32(a1, W[4 + k], _mm512_set1_epi32(x1));
                }
                __m512 p = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), s1,
                                           _mm512_mul_ps(_mm512_cvtepi32_ps(a0), s0));
                f[tt] = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b]), f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], gs));
        }
    }
}
/* kq_nvx_rows on x as int16, two int8 planes (xq the high one, xl the low
 * one, xs of kq_quant_part16): xn holds for each token the kq_nvx_xsum of
 * the high plane, then that of the low plane (2 cols / 16 values). The sums
 * of each half are a = 128 a_hi + a_lo (exact), then the operations of
 * kq_nvx_rows: the bits of k_kqh_nvx_q<X2> of the GPU. */
static void kq_nvx_rows2(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                         const int32_t *xn, const int8_t *xl, int n, float *out, size_t ostride)
{
    int nb = cols / 32, nh = cols / 16;
    float g;
    memcpy(&g, wg, 4);
    const __m512 gs = _mm512_set1_ps(128.f * g);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1u));
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int j = 0; j < n; j += KQ_NVX_TB) {
        int nt = n - j < KQ_NVX_TB ? n - j : KQ_NVX_TB;
        __m512 f[KQ_NVX_TB];
        for (int tt = 0; tt < KQ_NVX_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
                __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                W[k] = _mm512_shuffle_epi8(lut, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
            }
            __m512 s0 = kq_e4m3x16(blk + 256), s1 = kq_e4m3x16(blk + 272);
            for (int tt = 0; tt < nt; ++tt) {
                size_t xo = (size_t)(j + tt) * cols + 32 * b;
                const int32_t *nx = xn + (size_t)(j + tt) * 2 * nh + 2 * b;
                __m512i a0 = _mm512_set1_epi32(nx[0]), a1 = _mm512_set1_epi32(nx[1]);
                __m512i l0 = _mm512_set1_epi32(nx[nh]), l1 = _mm512_set1_epi32(nx[nh + 1]);
                for (int k = 0; k < 4; ++k) {
                    int32_t x0, x1, y0, y1;
                    memcpy(&x0, xq + xo + 4 * k, 4);
                    memcpy(&x1, xq + xo + 16 + 4 * k, 4);
                    memcpy(&y0, xl + xo + 4 * k, 4);
                    memcpy(&y1, xl + xo + 16 + 4 * k, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    a1 = _mm512_dpbusd_epi32(a1, W[4 + k], _mm512_set1_epi32(x1));
                    l0 = _mm512_dpbusd_epi32(l0, W[k], _mm512_set1_epi32(y0));
                    l1 = _mm512_dpbusd_epi32(l1, W[4 + k], _mm512_set1_epi32(y1));
                }
                a0 = _mm512_add_epi32(_mm512_slli_epi32(a0, 7), l0);
                a1 = _mm512_add_epi32(_mm512_slli_epi32(a1, 7), l1);
                __m512 p = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), s1,
                                           _mm512_mul_ps(_mm512_cvtepi32_ps(a0), s0));
                f[tt] = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b]), f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], gs));
        }
    }
}
#else
/* Without VNNI: the same sums in C (not the bits of the VNNI path). */
static void kq_nvx_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    static const int8_t e2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int nb = cols / 32;
    float g;
    memcpy(&g, wg, 4);
    (void)xn;
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
                float p = 0.f;
                for (int h = 0; h < 2; ++h) {
                    int32_t a = 0;
                    for (int v = 16 * h; v < 16 * h + 16; ++v) {
                        int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                        a += e2[c] * xq[(size_t)j * cols + 32 * b + v];
                    }
                    p += (float)a * kq_e4m3_tab[blk[256 + 16 * h + r]];
                }
                f += p * xs[(size_t)j * nb + b];
            }
            out[(size_t)j * ostride + r] = 0.5f * g * f;
        }
    }
}
#endif
