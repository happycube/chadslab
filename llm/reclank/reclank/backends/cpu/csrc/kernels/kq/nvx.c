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
/* The 16 rows of the KQ_NVX group at wg on n float32 rows of x (x + j xstride): the products of
 * the A16 contract of FP4-Local (docs/FP4LOCAL_PLAN.md: b12x's W4A16, x in bfloat16 values). The
 * products E2M1 x are exact (the codes through a table of 16 floats, vpermps); for each half block
 * (16 columns, one E4M3 scale a row) four float32 sums a_u (u: the column in its step of 4, fma in
 * the order of the steps), t = (a0 + a1) + (a2 + a3), then f = fma(t, s, f) over the halves in
 * order; out[j ostride + r] = f g (raw 0) or f (raw 1: the FC2 of the contract, whose g goes with
 * the router weight). Four sums a token keep the FMA chains apart (one chain ran at 5.8 G
 * weights/s a core). A token alone and in a group has the same operations. */
/* The 16 rows of the KQ_NVX group at wg on n rows of A4 x (FP4-Local's prompt rows: b12x's NVFP4
 * activations): xq the codes as twice their E2M1 values (int8, -12 to 12), sx the E4M3 scale of each
 * 16 (float), xn kq_nvx_xsum of xq. For each half block the exact int32 sum a of the codes'
 * products (vpdpbusd, from -12 sum(x)), f = fma(a, s sx, f) over the halves in order;
 * out[j ostride + r] = f 64 g / gs (= the sum of (q sf)(E2M1 s) times alpha = ws2 / gs; gs: the
 * A4 global scale in bytes 4-7 of the group's head). A token alone and in a group has the same
 * operations. */
static void kq_nvx_rows_a4(const uint8_t *wg, int cols, const int8_t *xq, const float *sx, const int32_t *xn, int n,
                           float *out, size_t ostride)
{
    int nb = cols / 32, nh = cols / 16;
    float g, gsc;
    memcpy(&g, wg, 4);
    memcpy(&gsc, wg + 4, 4);
    const __m512 post = _mm512_set1_ps(64.f * (g / gsc));
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
                const float *sxb = sx + (size_t)(j + tt) * nh + 2 * b;
                __m512i a0 = _mm512_set1_epi32(nx[0]), a1 = _mm512_set1_epi32(nx[1]);
                for (int k = 0; k < 4; ++k) {
                    int32_t x0, x1;
                    memcpy(&x0, xb + 4 * k, 4);
                    memcpy(&x1, xb + 16 + 4 * k, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    a1 = _mm512_dpbusd_epi32(a1, W[4 + k], _mm512_set1_epi32(x1));
                }
                f[tt] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a0), _mm512_mul_ps(s0, _mm512_set1_ps(sxb[0])), f[tt]);
                f[tt] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), _mm512_mul_ps(s1, _mm512_set1_ps(sxb[1])), f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], post));
        }
    }
}

#define KQ_NVF_TB 4
static void kq_nvx_rows_f(const uint8_t *wg, int cols, const float *x, size_t xstride, int n, float *out,
                          size_t ostride, int raw)
{
    static const float e2[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    int nb = cols / 32;
    float g;
    memcpy(&g, wg, 4);
    const __m512 lut = _mm512_loadu_ps(e2), post = _mm512_set1_ps(raw ? 256.f : 256.f * g);
    const __m512i m15 = _mm512_set1_epi32(15);
    __m512i sh[4];
    for (int u = 0; u < 4; ++u) {
        /* lanes 0-7: rows 0-7 (the low 4 bits of byte u of their word), 8-15: rows 8-15 (high) */
        sh[u] = _mm512_set_epi32(8 * u + 4, 8 * u + 4, 8 * u + 4, 8 * u + 4, 8 * u + 4, 8 * u + 4, 8 * u + 4, 8 * u + 4,
                                 8 * u, 8 * u, 8 * u, 8 * u, 8 * u, 8 * u, 8 * u, 8 * u);
    }
    for (int j = 0; j < n; j += KQ_NVF_TB) {
        int nt = n - j < KQ_NVF_TB ? n - j : KQ_NVF_TB;
        __m512 f[KQ_NVF_TB];
        for (int tt = 0; tt < KQ_NVF_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
            for (int h = 0; h < 2; ++h) {
                __m512 a[KQ_NVF_TB][4];
                for (int tt = 0; tt < KQ_NVF_TB; ++tt) {
                    a[tt][0] = a[tt][1] = a[tt][2] = a[tt][3] = _mm512_setzero_ps();
                }
                for (int k = 4 * h; k < 4 * h + 4; ++k) {
                    __m256i q8 = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
                    __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q8), q8, 1);
                    __m512 w[4];
                    for (int u = 0; u < 4; ++u) {
                        w[u] = _mm512_permutexvar_ps(_mm512_and_si512(_mm512_srlv_epi32(q, sh[u]), m15), lut);
                    }
                    const float *xb = x + (size_t)j * xstride + 32 * b + 4 * k;
                    for (int tt = 0; tt < nt; ++tt) {
                        const float *xt = xb + (size_t)tt * xstride;
                        for (int u = 0; u < 4; ++u) {
                            a[tt][u] = _mm512_fmadd_ps(w[u], _mm512_set1_ps(xt[u]), a[tt][u]);
                        }
                    }
                }
                __m512 s = kq_e4m3x16(blk + 256 + 16 * h);      /* x 2^-8 */
                for (int tt = 0; tt < nt; ++tt) {
                    __m512 t = _mm512_add_ps(_mm512_add_ps(a[tt][0], a[tt][1]), _mm512_add_ps(a[tt][2], a[tt][3]));
                    f[tt] = _mm512_fmadd_ps(t, s, f[tt]);
                }
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], post));
        }
    }
}
#else
/* kq_nvx_rows_a4 in C (the same sums). */
static void kq_nvx_rows_a4(const uint8_t *wg, int cols, const int8_t *xq, const float *sx, const int32_t *xn, int n,
                           float *out, size_t ostride)
{
    static const int8_t e2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int nb = cols / 32, nh = cols / 16;
    float g, gsc;
    memcpy(&g, wg, 4);
    memcpy(&gsc, wg + 4, 4);
    (void)xn;
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
                for (int h = 0; h < 2; ++h) {
                    int32_t a = 0;
                    for (int v = 16 * h; v < 16 * h + 16; ++v) {
                        int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                        a += e2[c] * xq[(size_t)j * cols + 32 * b + v];
                    }
                    f = fmaf((float)a, kq_e4m3_tab[blk[256 + 16 * h + r]] * (1.f / 256.f) * sx[(size_t)j * nh + 2 * b + h], f);
                }
            }
            out[(size_t)j * ostride + r] = f * (64.f * (g / gsc));
        }
    }
}
/* kq_nvx_rows_f in C (the same operations in the same order). */
static void kq_nvx_rows_f(const uint8_t *wg, int cols, const float *x, size_t xstride, int n, float *out,
                          size_t ostride, int raw)
{
    static const float e2[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
    int nb = cols / 32;
    float g;
    memcpy(&g, wg, 4);
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
                for (int h = 0; h < 2; ++h) {
                    float a[4] = {0.f, 0.f, 0.f, 0.f};
                    for (int v = 16 * h; v < 16 * h + 16; ++v) {
                        int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                        a[v % 4] = fmaf(e2[c], x[(size_t)j * xstride + 32 * b + v], a[v % 4]);
                    }
                    f = fmaf((a[0] + a[1]) + (a[2] + a[3]), kq_e4m3_tab[blk[256 + 16 * h + r]] * (1.f / 256.f), f);
                }
            }
            out[(size_t)j * ostride + r] = f * (raw ? 256.f : 256.f * g);
        }
    }
}
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

/* out (n x rows) = the A16 products of a KQ_NVX matrix (rows a multiple of 16) on n float rows of x
 * (kq_nvx_rows_f; raw: without the matrix's g). A check of the kernel (cops.kq_nvx_linear_f). */
void rc_kq_nvx_linear_f(const uint8_t *w, int rows, int cols, const float *x, int n, float *out, int raw)
{
    size_t gb = 16 * kq_row_bytes(KQ_NVX, cols);
    #pragma omp parallel for schedule(static)
    for (int gi = 0; gi < rows / 16; ++gi) {
        kq_nvx_rows_f(w + (size_t)gi * gb, cols, x, (size_t)cols, n, out + 16 * gi, (size_t)rows, raw);
    }
}

/* A4 x of FP4-Local (b12x's quantize_block_fp4, fast_math off): each 16 values of x (cols, a
 * multiple of 16) with the global scale gs: amax, sf = e4m3(min(amax gs / 6, 448)) (to nearest even,
 * saturating), the code of |v| from the thresholds (sf / gs) {0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5}
 * (to nearest even on |v|, clamped at 6); q gets twice the E2M1 value (int8), sx the value of sf. */
static float kq_e4m3_rne(float v)
{
    /* the nearest E4M3 value of v >= 0 (subnormals; ties to the even code), at most 448 */
    if (!(v > 0.f)) {
        return 0.f;
    }
    if (v >= 448.f) {
        return 448.f;
    }
    int lo = 0, hi = 126;           /* kq_e4m3_tab[0..126] ascending */
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if (kq_e4m3_tab[m] <= v) {
            lo = m;
        } else {
            hi = m;
        }
    }
    float a = kq_e4m3_tab[lo], b = kq_e4m3_tab[hi];
    if (v - a < b - v) {
        return a;
    }
    if (v - a > b - v) {
        return b;
    }
    return (lo % 2 == 0) ? a : b;
}

static void kq_a4_quant(const float *x, int cols, float gs, int8_t *q, float *sx)
{
    for (int h = 0; h < cols / 16; ++h) {
        const float *v = x + 16 * h;
        float amax = 0.f;
        for (int i = 0; i < 16; ++i) {
            amax = fmaxf(amax, fabsf(v[i]));
        }
        float sf = kq_e4m3_rne(fminf((amax * gs) / 6.f, 448.f));
        sx[h] = sf;
        int8_t *qq = q + 16 * h;
        if (sf == 0.f || gs == 0.f) {
            memset(qq, 0, 16);
            continue;
        }
        float vs = sf / gs;
        float t1 = vs * 0.25f, t2 = vs * 0.75f, t3 = vs * 1.25f, t4 = vs * 1.75f, t5 = vs * 2.5f,
              t6 = vs * 3.5f, t7 = vs * 5.f;
        for (int i = 0; i < 16; ++i) {
            float a = fabsf(v[i]);
            int c = a > t7 ? 12 : a >= t6 ? 8 : a > t5 ? 6 : a >= t4 ? 4 : a > t3 ? 3 : a >= t2 ? 2 : a > t1 ? 1 : 0;
            qq[i] = (int8_t)(v[i] < 0.f ? -c : c);
        }
    }
}

/* cops.kq_a4_quant: rows of x to the A4 form (a check against the reference's quantizer) */
void rc_kq_a4_quant(const float *x, int n, int cols, float gs, int8_t *q, float *sx)
{
    for (int j = 0; j < n; ++j) {
        kq_a4_quant(x + (size_t)j * cols, cols, gs, q + (size_t)j * cols, sx + (size_t)j * (cols / 16));
    }
}

/* cops.kq_nvx_linear_a4: n rows of A4 x times a KQ_NVX matrix (kq_nvx_rows_a4) */
void rc_kq_nvx_linear_a4(const uint8_t *w, int rows, int cols, const int8_t *q, const float *sx, int n, float *out)
{
    int32_t *xn = (int32_t *)malloc((size_t)n * (cols / 16) * 4);
    kq_nvx_xsum(q, n, cols, xn);
    size_t gb = 16 * kq_row_bytes(KQ_NVX, cols);
    #pragma omp parallel for schedule(static)
    for (int gi = 0; gi < rows / 16; ++gi) {
        kq_nvx_rows_a4(w + (size_t)gi * gb, cols, q, sx, xn, n, out + 16 * gi, (size_t)rows);
    }
    free(xn);
}

/* the E4M3 code of x (to nearest even, subnormals, saturating at 448), as the GPU's rc_e4m3_code */
static uint8_t kq_e4m3_code(float x)
{
    uint8_t sgn = x < 0.f || (x == 0.f && signbit(x)) ? 0x80 : 0;
    float v = kq_e4m3_rne(fabsf(x));
    int lo = 0, hi = 127;
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if (kq_e4m3_tab[m] <= v) {
            lo = m;
        } else {
            hi = m;
        }
    }
    return sgn | (uint8_t)lo;
}

/* cops.kv_e4m3_codes: the E4M3 codes of n values (the FP8 cache of FP4-Local) */
void rc_kv_e4m3_codes(const float *x, int64_t n, uint8_t *out)
{
    for (int64_t i = 0; i < n; ++i) {
        out[i] = kq_e4m3_code(x[i]);
    }
}
