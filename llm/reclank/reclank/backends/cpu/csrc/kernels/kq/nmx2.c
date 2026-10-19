/* ---------- one row on one token of x as int16 in two int8 planes (Nemotron) ----------
 * x of kq_quant_part16 (moe_scratch.c): v = 128 hi + lo in each 32 values (hi in [-126, 127], lo in
 * [-64, 63]), xs the scale of each 32. A row of IQ2_XXS, KQ_NV4 or Q8_0 against it: the integer sum
 * of each part 128 sum(w hi) + sum(w lo), exact, times the scales (the precision of int16 x: the
 * noise of int8 x was ~0.5% of a product). |w| unsigned with the sign on x (vpmaddubsw: 128 * 127 *
 * 2 < 32768, no saturation; vpdpbusd where the CPU has it: the same lanes). The scalar tier: the
 * float loop of the values. */

#if defined(__AVX2__)
static inline __m256i kq_x2_dot(__m256i acc, __m256i u, __m256i s)
{
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
    return _mm256_dpbusd_epi32(acc, u, s);
#elif defined(__AVXVNNI__)
    return _mm256_dpbusd_avx_epi32(acc, u, s);
#else
    return _mm256_add_epi32(acc, _mm256_madd_epi16(_mm256_maddubs_epi16(u, s), _mm256_set1_epi16(1)));
#endif
}

static inline float kq_x2_hsum(__m256 acc)
{
    __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    h = _mm_add_ps(h, _mm_movehl_ps(h, h));
    h = _mm_add_ss(h, _mm_movehdup_ps(h));
    return _mm_cvtss_f32(h);
}

/* 128 dot(u, hi') + dot(u, lo') of 32 values (u unsigned, the signs already on hi', lo') */
static inline __m256i kq_x2_part(__m256i u, __m256i h, __m256i l)
{
    __m256i a = kq_x2_dot(_mm256_setzero_si256(), u, h);
    __m256i b = kq_x2_dot(_mm256_setzero_si256(), u, l);
    return _mm256_add_epi32(_mm256_slli_epi32(a, 7), b);
}
#endif

static float kq_dot_x2(const uint8_t *w, int type, int cols, const int8_t *xh, const int8_t *xl,
                       const float *xs)
{
#if defined(__AVX2__)
    __m256 acc = _mm256_setzero_ps();
    if (type == KQ_IQ2_XXS || type == KQ_IQ2R) {
        for (int b = 0; b < cols / 256; ++b) {
            float d = kq_iq2_d(w, type, cols, b) * 0.125f;
            for (int j = 0; j < 8; ++j) {
                uint32_t aux[2];
                memcpy(aux, kq_iq2_part(w, type, b, j), 8);
                const uint8_t *ix = (const uint8_t *)aux;
                __m256i u = _mm256_set_epi64x((long long)kq_iq2xxs_grid[ix[3]], (long long)kq_iq2xxs_grid[ix[2]],
                                              (long long)kq_iq2xxs_grid[ix[1]], (long long)kq_iq2xxs_grid[ix[0]]);
                __m256i m = _mm256_set_epi64x((long long)kq_iq2xxs_signs64[(aux[1] >> 21) & 127],
                                              (long long)kq_iq2xxs_signs64[(aux[1] >> 14) & 127],
                                              (long long)kq_iq2xxs_signs64[(aux[1] >> 7) & 127],
                                              (long long)kq_iq2xxs_signs64[aux[1] & 127]);
                int p = 8 * b + j;
                __m256i h = _mm256_loadu_si256((const __m256i *)(xh + 32 * p));
                __m256i l = _mm256_loadu_si256((const __m256i *)(xl + 32 * p));
                h = _mm256_sub_epi8(_mm256_xor_si256(h, m), m);
                l = _mm256_sub_epi8(_mm256_xor_si256(l, m), m);
                float sc = d * (float)(2 * (aux[1] >> 28) + 1) * xs[p];
                acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(kq_x2_part(u, h, l)), _mm256_set1_ps(sc), acc);
            }
        }
        return kq_x2_hsum(acc);
    }
    if (type == KQ_NV4) {
        const __m256i lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
        const __m128i m4 = _mm_set1_epi8(15);
        for (int b = 0; b < cols / 32; ++b) {
            __m128i q = _mm_loadu_si128((const __m128i *)kq_nv4_codes(w, b));
            __m256i c = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4));
            __m256i v = _mm256_shuffle_epi8(lut, c);
            __m256i u = _mm256_abs_epi8(v);
            __m256i h = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(xh + 32 * b)), v);
            __m256i l = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(xl + 32 * b)), v);
            const uint8_t *sc = kq_nv4_scales(w, cols, b);
            float f = 0.5f * xs[b];
            __m256 scale = _mm256_set_m128(_mm_set1_ps(f * kq_e4m3(sc[1])), _mm_set1_ps(f * kq_e4m3(sc[0])));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(kq_x2_part(u, h, l)), scale, acc);
        }
        return kq_x2_hsum(acc) * kq_nv4_g(w, cols);
    }
    if (type == KQ_Q8_0) {
        for (int b = 0; b < cols / 32; ++b) {
            const uint8_t *blk = w + (size_t)b * 34;
            __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 2));
            __m256i u = _mm256_abs_epi8(q);
            __m256i h = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(xh + 32 * b)), q);
            __m256i l = _mm256_sign_epi8(_mm256_loadu_si256((const __m256i *)(xl + 32 * b)), q);
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(kq_x2_part(u, h, l)), _mm256_set1_ps(kq_h(blk) * xs[b]), acc);
        }
        return kq_x2_hsum(acc);
    }
#endif
    /* the float loop: the values of the blocks against 128 hi + lo */
    int bv = kq_block32(type) ? 32 : 256;
    float v[256], s = 0.f;
    for (int b = 0; b < cols / bv; ++b) {
        kq_block_values(type, w, cols, b, v);
        for (int i = 0; i < bv; ++i) {
            int c = b * bv + i;
            s += v[i] * xs[c / 32] * (float)(128 * (int)xh[c] + (int)xl[c]);
        }
    }
    return s;
}

#if defined(__AVX2__)
/* ---------- IQ2_XXS / KQ_IQ2R rows on int16 x: R rows a pass, on T tokens (kq_rows_x2_iq2) ----------
 * kq_dot_x2 decodes each part of a row for each token: the four grid entries and the four sign
 * masks (8 lookups) and the scale of the part (in scalars), then two sign flips, the products and a
 * float product. Here the grid entries and the signs of a part are read once for the T tokens (a
 * group of a few rows of NM_LIN2: one row on 2 or 3 tokens), the x of a part once for the R rows,
 * the table entries by broadcasts and blends (loads, not the inserts of _mm256_set_epi64x), the signs
 * as +1 / -1 bytes for vpsignb (one flip, not a xor and a subtraction), and the 8 factors d (2 s + 1)
 * of a block in one vector pass. Each row and token gets the int32 lanes of kq_dot_x2 and its float
 * operations in its order (d (2 s + 1), times xs, fmadd into the 8 lanes, the sum): the same bits.
 * The AVX2 tier: one row on one token 9.3 -> 7.4 ms for a 5376 x 4096 matrix (a thread). */
static uint64_t kq_iq2_pm1[128];       /* IQ2XXS_SIGNS64 as bytes of +1 and -1 */
__attribute__((constructor)) static void kq_iq2_pm1_init(void)
{
    for (int i = 0; i < 128; ++i) {
        uint64_t s = kq_iq2xxs_signs64[i], v = 0;
        for (int k = 0; k < 8; ++k) {
            v |= (uint64_t)(((s >> (8 * k)) & 0xff) ? 0xff : 0x01) << (8 * k);
        }
        kq_iq2_pm1[i] = v;
    }
}

/* the entries i0 .. i3 of a table of 8-byte rows as one vector (four broadcasts, three blends) */
static inline __m256i kq_iq2_q4(const uint64_t *t, unsigned i0, unsigned i1, unsigned i2, unsigned i3)
{
    __m256i a = _mm256_blend_epi32(_mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)(t + i0))),
                                   _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)(t + i1))), 0x0c);
    __m256i b = _mm256_blend_epi32(_mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)(t + i2))),
                                   _mm256_broadcastq_epi64(_mm_loadl_epi64((const __m128i *)(t + i3))), 0xc0);
    return _mm256_blend_epi32(a, b, 0xf0);
}

/* the lanes of kq_x2_part: 128 dot(u, hi') + dot(u, lo'), the same integers (the high plane
 * weighted by 128 in vpmaddwd, not shifted after it, without VNNI) */
static inline __m256i kq_x2_part2(__m256i u, __m256i h, __m256i l)
{
#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__)
    return kq_x2_part(u, h, l);
#else
    __m256i a = _mm256_madd_epi16(_mm256_maddubs_epi16(u, h), _mm256_set1_epi16(128));
    __m256i b = _mm256_madd_epi16(_mm256_maddubs_epi16(u, l), _mm256_set1_epi16(1));
    return _mm256_add_epi32(a, b);
#endif
}

/* the 8 factors d (2 s_j + 1) of block b of a row, each as kq_dot_x2 makes it (d * 0.125, then
 * times the float of 2 s + 1) */
static inline __m256 kq_iq2_f8(const uint8_t *w, int type, int cols, int b)
{
    const uint8_t *p0 = kq_iq2_part(w, type, b, 0);
    __m256i a = _mm256_srli_epi64(_mm256_loadu_si256((const __m256i *)p0), 60);
    __m256i c = _mm256_srli_epi64(_mm256_loadu_si256((const __m256i *)(p0 + 32)), 60);
    const __m256i pk = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    __m256i v = _mm256_permute2x128_si256(_mm256_permutevar8x32_epi32(a, pk), _mm256_permutevar8x32_epi32(c, pk), 0x20);
    __m256 f = _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_add_epi32(v, v), _mm256_set1_epi32(1)));
    return _mm256_mul_ps(_mm256_set1_ps(kq_iq2_d(w, type, cols, b) * 0.125f), f);
}

/* R rows (from w, rb apart) on T tokens (xh, xl cols apart, xs cols / 32 apart): out[r ors + j ots] */
#define KQ_IQ2_RT(R, T)                                                                                            \
    static void kq_iq2_rt_##R##_##T(const uint8_t *w, size_t rb, int type, int cols, const int8_t *xh,           \
                                    const int8_t *xl, const float *xs, float *out, size_t ors, size_t ots)       \
    {                                                                                                              \
        __m256 acc[R][T];                                                                                          \
        for (int r = 0; r < R; ++r)                                                                                \
            for (int j = 0; j < T; ++j)                                                                            \
                acc[r][j] = _mm256_setzero_ps();                                                                   \
        const int nxs = cols / 32;                                                                                 \
        for (int b = 0; b < cols / 256; ++b) {                                                                     \
            __m256 f8[R];                                                                                          \
            for (int r = 0; r < R; ++r)                                                                            \
                f8[r] = kq_iq2_f8(w + r * rb, type, cols, b);                                                      \
            for (int q = 0; q < 8; ++q) {                                                                          \
                const int p = 8 * b + q;                                                                           \
                const __m256i qq = _mm256_set1_epi32(q);                                                           \
                __m256i h0[T], l0[T];                                                                              \
                __m256 xsp[T];                                                                                     \
                for (int j = 0; j < T; ++j) {                                                                      \
                    h0[j] = _mm256_loadu_si256((const __m256i *)(xh + (size_t)j * cols + 32 * p));               \
                    l0[j] = _mm256_loadu_si256((const __m256i *)(xl + (size_t)j * cols + 32 * p));               \
                    xsp[j] = _mm256_set1_ps(xs[(size_t)j * nxs + p]);                                              \
                }                                                                                                  \
                for (int r = 0; r < R; ++r) {                                                                      \
                    uint32_t aux[2];                                                                               \
                    memcpy(aux, kq_iq2_part(w + r * rb, type, b, q), 8);                                           \
                    const uint8_t *ix = (const uint8_t *)aux;                                                      \
                    const __m256i u = kq_iq2_q4(kq_iq2xxs_grid, ix[0], ix[1], ix[2], ix[3]);                      \
                    const __m256i m = kq_iq2_q4(kq_iq2_pm1, aux[1] & 127, (aux[1] >> 7) & 127,                    \
                                                (aux[1] >> 14) & 127, (aux[1] >> 21) & 127);                      \
                    const __m256 fq = _mm256_permutevar8x32_ps(f8[r], qq);                                         \
                    for (int j = 0; j < T; ++j) {                                                                  \
                        __m256i h = _mm256_sign_epi8(h0[j], m), l = _mm256_sign_epi8(l0[j], m);                   \
                        acc[r][j] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(kq_x2_part2(u, h, l)),                     \
                                                    _mm256_mul_ps(fq, xsp[j]), acc[r][j]);                        \
                    }                                                                                              \
                }                                                                                                  \
            }                                                                                                      \
        }                                                                                                          \
        for (int r = 0; r < R; ++r)                                                                                \
            for (int j = 0; j < T; ++j)                                                                            \
                out[r * ors + j * ots] = kq_x2_hsum(acc[r][j]);                                                    \
    }
KQ_IQ2_RT(4, 1)
KQ_IQ2_RT(1, 1)
KQ_IQ2_RT(2, 2)
KQ_IQ2_RT(1, 2)
KQ_IQ2_RT(2, 3)
KQ_IQ2_RT(1, 3)
#undef KQ_IQ2_RT

/* nr rows of IQ2_XXS / KQ_IQ2R on t tokens (1 to 3) of x as int16 planes: out[r ors + j ots], each the
 * bits of kq_dot_x2 */
static void kq_rows_x2_iq2(const uint8_t *w, size_t rb, int type, int cols, int nr, const int8_t *xh,
                           const int8_t *xl, const float *xs, int t, float *out, size_t ors, size_t ots)
{
    int r = 0;
    if (t == 1) {
        for (; r + 4 <= nr; r += 4) {
            kq_iq2_rt_4_1(w + r * rb, rb, type, cols, xh, xl, xs, out + r * ors, ors, ots);
        }
        for (; r < nr; ++r) {
            kq_iq2_rt_1_1(w + r * rb, rb, type, cols, xh, xl, xs, out + r * ors, ors, ots);
        }
    } else if (t == 2) {
        for (; r + 2 <= nr; r += 2) {
            kq_iq2_rt_2_2(w + r * rb, rb, type, cols, xh, xl, xs, out + r * ors, ors, ots);
        }
        for (; r < nr; ++r) {
            kq_iq2_rt_1_2(w + r * rb, rb, type, cols, xh, xl, xs, out + r * ors, ors, ots);
        }
    } else {
        for (; r + 2 <= nr; r += 2) {
            kq_iq2_rt_2_3(w + r * rb, rb, type, cols, xh, xl, xs, out + r * ors, ors, ots);
        }
        for (; r < nr; ++r) {
            kq_iq2_rt_1_3(w + r * rb, rb, type, cols, xh, xl, xs, out + r * ors, ors, ots);
        }
    }
}
#endif

#if (defined(__AVX512VNNI__) && defined(__AVX512VL__)) || defined(__AVXVNNI__)
#define NM_X2_VNNI 1
/* 4 rows of KQ_NV4 or Q8_0 against x as two planes on VNNI: the planes made unsigned once (hu = hi +
 * 128, lu = lo + 128, so they do not depend on the signs of a row and serve the 4 rows), vpdpbusd of
 * them and of ones against the int8 values w: sum(w hi) = sum(w hu) - 128 sum(w), so the part is
 * 128 sum(w hu) + sum(w lu) - 16512 sum(w) (exact: the int32 lanes do not saturate). */
static void kq_rows4_x2_vnni(const uint8_t *w, size_t rb, int type, int cols, const uint8_t *hu,
                             const uint8_t *lu, const float *xs, int nr, float *out)
{
    __m256 acc[4] = {_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps()};
    const __m256i ones = _mm256_set1_epi8(1), lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
    const __m128i m4 = _mm_set1_epi8(15);
    for (int b = 0; b < cols / 32; ++b) {
        const __m256i h = _mm256_loadu_si256((const __m256i *)(hu + 32 * b));
        const __m256i l = _mm256_loadu_si256((const __m256i *)(lu + 32 * b));
        for (int r = 0; r < nr; ++r) {
            const uint8_t *wr = w + (size_t)r * rb;
            __m256i v;
            __m256 scale;
            if (type == KQ_NV4) {
                __m128i q = _mm_loadu_si128((const __m128i *)kq_nv4_codes(wr, b));
                v = _mm256_shuffle_epi8(lut, _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4)));
                const uint8_t *sc = kq_nv4_scales(wr, cols, b);
                const float f = 0.5f * xs[b] * kq_nv4_g(wr, cols);
                scale = _mm256_set_m128(_mm_set1_ps(f * kq_e4m3(sc[1])), _mm_set1_ps(f * kq_e4m3(sc[0])));
            } else {
                const uint8_t *blk = wr + (size_t)b * 34;
                v = _mm256_loadu_si256((const __m256i *)(blk + 2));
                scale = _mm256_set1_ps(kq_h(blk) * xs[b]);
            }
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
            __m256i sh = _mm256_dpbusd_epi32(_mm256_setzero_si256(), h, v);
            __m256i sl = _mm256_dpbusd_epi32(_mm256_setzero_si256(), l, v);
            __m256i sw = _mm256_dpbusd_epi32(_mm256_setzero_si256(), ones, v);
#else
            __m256i sh = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), h, v);
            __m256i sl = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), l, v);
            __m256i sw = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), ones, v);
#endif
            __m256i p = _mm256_sub_epi32(_mm256_add_epi32(_mm256_slli_epi32(sh, 7), sl),
                                         _mm256_mullo_epi32(sw, _mm256_set1_epi32(16512)));
            acc[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(p), scale, acc[r]);
        }
    }
    for (int r = 0; r < nr; ++r) {
        out[r] = kq_x2_hsum(acc[r]);
    }
}
#endif

#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
/* kq_rows4_x2_vnni with 512 bits: two blocks of 32 a step (cols a multiple of 64). NVFP4: the 32 code
 * bytes of the two blocks as 64 nibbles (block A low, A high, B low, B high: the 4 lanes of 128 bits
 * in the order of the values) through one vpshufb; the 4 half-block scales spread over their lanes
 * by one permute. The correction 16512 sum(w) by shifts. */
static void kq_rows4_x2_512(const uint8_t *w, size_t rb, int type, int cols, const uint8_t *hu,
                            const uint8_t *lu, const float *xs, int nr, float *out)
{
    __m512 acc[4] = {_mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps(), _mm512_setzero_ps()};
    const __m512i ones = _mm512_set1_epi8(1);
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
    const __m256i m4 = _mm256_set1_epi8(15);
    /* the lanes of the 16 scales of 8 blocks for the step of blocks 2q, 2q + 1 (A0 A1 B0 B1, 4 lanes each) */
    const __m512i sp[4] = {
        _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0),
        _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4),
        _mm512_set_epi32(11, 11, 11, 11, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8),
        _mm512_set_epi32(15, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12)};
    /* the x scale of each of the 16 halves (block b of 8: lanes 2b, 2b + 1) */
    const __m512i two = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    float g[4];
    for (int r = 0; r < nr; ++r) {
        g[r] = type == KQ_NV4 ? 0.5f * kq_nv4_g(w + (size_t)r * rb, cols) : 1.f;
    }
    for (int b8 = 0; b8 < cols / 32; b8 += 8) {
        const int nb = cols / 32 - b8 < 8 ? cols / 32 - b8 : 8;      /* (even: cols % 64 == 0) */
        const __m512 xsv = _mm512_permutexvar_ps(two, _mm512_castps256_ps512(_mm256_loadu_ps(xs + b8)));
        __m512 scl[4];
        for (int r = 0; r < nr; ++r) {
            const uint8_t *wr = w + (size_t)r * rb;
            if (type == KQ_NV4) {
                __m512i ix = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)kq_nv4_scales(wr, cols, b8)));
                scl[r] = _mm512_mul_ps(_mm512_mul_ps(_mm512_i32gather_ps(ix, kq_e4m3_tab, 4), xsv), _mm512_set1_ps(g[r]));
            } else {
                float d[16];
                for (int q = 0; q < nb; ++q) {
                    d[2 * q] = d[2 * q + 1] = kq_h(wr + (size_t)(b8 + q) * 34);
                }
                for (int q = nb; q < 8; ++q) {
                    d[2 * q] = d[2 * q + 1] = 0.f;
                }
                scl[r] = _mm512_mul_ps(_mm512_loadu_ps(d), xsv);
            }
        }
        for (int q = 0; q < nb; q += 2) {
            const int b = b8 + q;
            const __m512i h = _mm512_loadu_si512((const void *)(hu + 32 * b));
            const __m512i l = _mm512_loadu_si512((const void *)(lu + 32 * b));
            for (int r = 0; r < nr; ++r) {
                const uint8_t *wr = w + (size_t)r * rb;
                __m512i v;
                if (type == KQ_NV4) {
                    const __m256i c = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(wr, b));   /* A, B */
                    const __m512i lh = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(c, m4)),
                                                          _mm256_and_si256(_mm256_srli_epi16(c, 4), m4), 1);
                    /* [A lo, B lo, A hi, B hi] -> [A lo, A hi, B lo, B hi] */
                    v = _mm512_shuffle_epi8(lut, _mm512_shuffle_i64x2(lh, lh, _MM_SHUFFLE(3, 1, 2, 0)));
                } else {
                    const uint8_t *blk = wr + (size_t)b * 34;
                    v = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(blk + 2))),
                                           _mm256_loadu_si256((const __m256i *)(blk + 36)), 1);
                }
                __m512i sh = _mm512_dpbusd_epi32(_mm512_setzero_si512(), h, v);
                __m512i sl = _mm512_dpbusd_epi32(_mm512_setzero_si512(), l, v);
                __m512i sw = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, v);
                __m512i p = _mm512_add_epi32(_mm512_slli_epi32(sh, 7), sl);
                p = _mm512_sub_epi32(p, _mm512_add_epi32(_mm512_slli_epi32(sw, 14), _mm512_slli_epi32(sw, 7)));
                acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(p), _mm512_permutexvar_ps(sp[q / 2], scl[r]), acc[r]);
            }
        }
    }
    for (int r = 0; r < nr; ++r) {
        out[r] = _mm512_reduce_add_ps(acc[r]);
    }
}
#endif

/* ---------- the tiled product of a prompt group (t >= NM_TILE_T) ----------
 * x as int16 (128 hi + lo: exact) against the values of the rows decoded to int16, vpmaddwd
 * (vpdpwssd on VNNI): signed by signed, no offsets, every type on every tier. The lanes of a vector
 * are rows: a tile of NM_RT rows (two vectors) on NM_NQ tokens, each pair of values of x broadcast
 * against the pairs of the rows; the int32 lanes sum a unit (the values of a scale: 32 for IQ2_XXS
 * and Q8_0, 16 for NVFP4), then one conversion and one fma with the scale of each row times that of
 * the token. No horizontal sums. A task: R rows (a share of the threads), the columns in chunks of
 * NM_KC: each tile of rows decoded (to the rows of a buffer, then 8 x 8 transposes to the pairs of
 * the 32 rows) and run on every token while it is in the L1. */
#define NM_TILE_T 4
#define NM_KC 256
#if defined(__AVX2__)
#if defined(__AVX512F__) && defined(__AVX512BW__)
#define NM_VL 16
#define NM_NQ 6
typedef __m512i nm_vi;
typedef __m512 nm_vf;
#define NM_LDI(p) _mm512_loadu_si512((const void *)(p))
#define NM_LDF(p) _mm512_loadu_ps(p)
#define NM_STF(p, v) _mm512_storeu_ps(p, v)
#define NM_SETI(x) _mm512_set1_epi32(x)
#define NM_SETF(x) _mm512_set1_ps(x)
#define NM_ZI() _mm512_setzero_si512()
#define NM_ZF() _mm512_setzero_ps()
#define NM_CVT(x) _mm512_cvtepi32_ps(x)
#define NM_MUL(a, b) _mm512_mul_ps(a, b)
#define NM_FMA(a, b, c) _mm512_fmadd_ps(a, b, c)
#define NM_HSUM(a) _mm512_reduce_add_ps(a)
#define NM_ADD(a, b) _mm512_add_ps(a, b)
#define NM_SUB(a, b) _mm512_sub_ps(a, b)
#define NM_DIV(a, b) _mm512_div_ps(a, b)
#define NM_MAX(a, b) _mm512_max_ps(a, b)
#define NM_MIN(a, b) _mm512_min_ps(a, b)
#define NM_ROUND(a) _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define NM_POW2(n) _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23))
#if defined(__AVX512VNNI__)
#define NM_DOT(a, w, x) ((a) = _mm512_dpwssd_epi32(a, w, x))
#else
#define NM_DOT(a, w, x) ((a) = _mm512_add_epi32(a, _mm512_madd_epi16(w, x)))
#endif
#else
#define NM_VL 8
#define NM_NQ 3
typedef __m256i nm_vi;
typedef __m256 nm_vf;
#define NM_LDI(p) _mm256_loadu_si256((const __m256i *)(p))
#define NM_LDF(p) _mm256_loadu_ps(p)
#define NM_STF(p, v) _mm256_storeu_ps(p, v)
#define NM_SETI(x) _mm256_set1_epi32(x)
#define NM_SETF(x) _mm256_set1_ps(x)
#define NM_ZI() _mm256_setzero_si256()
#define NM_ZF() _mm256_setzero_ps()
#define NM_CVT(x) _mm256_cvtepi32_ps(x)
#define NM_MUL(a, b) _mm256_mul_ps(a, b)
#define NM_FMA(a, b, c) _mm256_fmadd_ps(a, b, c)
#define NM_HSUM(a) kq_x2_hsum(a)
#define NM_ADD(a, b) _mm256_add_ps(a, b)
#define NM_SUB(a, b) _mm256_sub_ps(a, b)
#define NM_DIV(a, b) _mm256_div_ps(a, b)
#define NM_MAX(a, b) _mm256_max_ps(a, b)
#define NM_MIN(a, b) _mm256_min_ps(a, b)
#define NM_ROUND(a) _mm256_round_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)
#define NM_POW2(n) _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23))
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
#define NM_DOT(a, w, x) ((a) = _mm256_dpwssd_epi32(a, w, x))
#elif defined(__AVXVNNI__)
#define NM_DOT(a, w, x) ((a) = _mm256_dpwssd_avx_epi32(a, w, x))
#else
/* (in asm: gcc reassociates the sums of the intrinsics over the unrolled pairs and spills) */
#define NM_DOT(a, w, x) do { nm_vi t_; __asm__("vpmaddwd %3, %2, %1\n\tvpaddd %1, %0, %0" : "+x"(a), "=&x"(t_) : "x"(w), "x"(x)); } while (0)
#endif
#endif
#define NM_RT (2 * NM_VL)

/* exp of the lanes (Cephes: 2^n e^r, |r| <= ln2 / 2, a polynomial of degree 6; ~1 ulp) and silu */
static inline nm_vf nm_expv(nm_vf x)
{
    x = NM_MIN(NM_MAX(x, NM_SETF(-87.3f)), NM_SETF(88.3f));
    const nm_vf n = NM_ROUND(NM_MUL(x, NM_SETF(1.44269504089f)));
    nm_vf r = NM_FMA(n, NM_SETF(-0.693359375f), x);
    r = NM_FMA(n, NM_SETF(2.12194440e-4f), r);
    nm_vf p = NM_SETF(1.9875691500e-4f);
    p = NM_FMA(p, r, NM_SETF(1.3981999507e-3f));
    p = NM_FMA(p, r, NM_SETF(8.3334519073e-3f));
    p = NM_FMA(p, r, NM_SETF(4.1665795894e-2f));
    p = NM_FMA(p, r, NM_SETF(1.6666665459e-1f));
    p = NM_FMA(p, r, NM_SETF(5.0000001201e-1f));
    p = NM_FMA(NM_MUL(p, r), r, NM_ADD(r, NM_SETF(1.f)));
    return NM_MUL(p, NM_POW2(n));
}

static inline nm_vf nm_siluv(nm_vf x)
{
    return NM_DIV(x, NM_ADD(NM_SETF(1.f), nm_expv(NM_SUB(NM_ZF(), x))));
}

/* the thread's buffers of the tiled form */
static __thread uint8_t *nm_tile_mem;
static __thread size_t nm_tile_cap;

static void *nm_tile_scratch(size_t n)
{
    if (n > nm_tile_cap) {
        free(nm_tile_mem);
        nm_tile_mem = (uint8_t *)aligned_alloc(64, (n + 63) & ~(size_t)63);
        nm_tile_cap = n;
    }
    return nm_tile_mem;
}

/* IQ2_XXS for the tiles: the grid as int16 (8 a code) and the signs as +-1 int16 (8 an index) */
static int16_t nm_grid16[256][8] __attribute__((aligned(16)));
static int16_t nm_sign16[128][8] __attribute__((aligned(16)));

__attribute__((constructor)) static void nm_tables16(void)
{
    for (int i = 0; i < 256; ++i) {
        for (int k = 0; k < 8; ++k) {
            nm_grid16[i][k] = (int16_t)((kq_iq2xxs_grid[i] >> (8 * k)) & 255);
        }
    }
    for (int i = 0; i < 128; ++i) {
        for (int k = 0; k < 8; ++k) {
            nm_sign16[i][k] = ((kq_iq2xxs_signs64[i] >> (8 * k)) & 255) ? -1 : 1;
        }
    }
}

/* values c .. c + 15 (c a multiple of 16) of a row as int16, and on the first of a unit its scale */
static inline __attribute__((always_inline)) __m256i nm_dec16(const uint8_t *w, const int type, int cols, int c, float g,
                                                              float *sc)
{
    if (type == KQ_IQ2_XXS || type == KQ_IQ2R) {
        uint32_t aux[2];
        memcpy(aux, kq_iq2_part(w, type, c / 256, (c / 32) % 8), 8);
        const int l = (c / 8) % 4;
        const uint8_t *ix = (const uint8_t *)aux;
        __m256i v = _mm256_set_m128i(_mm_load_si128((const __m128i *)nm_grid16[ix[l + 1]]),
                                     _mm_load_si128((const __m128i *)nm_grid16[ix[l]]));
        __m256i m = _mm256_set_m128i(_mm_load_si128((const __m128i *)nm_sign16[(aux[1] >> (7 * (l + 1))) & 127]),
                                     _mm_load_si128((const __m128i *)nm_sign16[(aux[1] >> (7 * l)) & 127]));
        if (l == 0) {
            *sc = kq_iq2_d(w, type, cols, c / 256) * (float)(2 * (aux[1] >> 28) + 1) * 0.125f;
        }
        return _mm256_sign_epi16(v, m);
    }
    if (type == KQ_NV4) {
        const int b = c / 32, h = (c / 16) % 2;
        __m128i q = _mm_loadu_si128((const __m128i *)kq_nv4_codes(w, b));
        q = _mm_and_si128(h ? _mm_srli_epi16(q, 4) : q, _mm_set1_epi8(15));
        *sc = g * kq_e4m3(kq_nv4_scales(w, cols, b)[h]);
        return _mm256_cvtepi8_epi16(_mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)kq_e2m1x2), q));
    }
    const uint8_t *blk = w + (size_t)(c / 32) * 34;
    if (c % 32 == 0) {
        *sc = kq_h(blk);
    }
    return _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(blk + 2 + (c % 32))));
}

/* 8 rows of 8 int32 (r) to b (rows of stride sb), transposed */
static inline void nm_t8x8(__m256i *r, int32_t *b, size_t sb)
{
    __m256i t[8];
    for (int i = 0; i < 8; i += 2) {
        t[i] = _mm256_unpacklo_epi32(r[i], r[i + 1]);
        t[i + 1] = _mm256_unpackhi_epi32(r[i], r[i + 1]);
    }
    for (int i = 0; i < 8; i += 4) {
        r[i] = _mm256_unpacklo_epi64(t[i], t[i + 2]);
        r[i + 1] = _mm256_unpackhi_epi64(t[i], t[i + 2]);
        r[i + 2] = _mm256_unpacklo_epi64(t[i + 1], t[i + 3]);
        r[i + 3] = _mm256_unpackhi_epi64(t[i + 1], t[i + 3]);
    }
    for (int i = 0; i < 4; ++i) {
        _mm256_storeu_si256((__m256i *)(b + i * sb), _mm256_permute2x128_si256(r[i], r[i + 4], 0x20));
        _mm256_storeu_si256((__m256i *)(b + (i + 4) * sb), _mm256_permute2x128_si256(r[i], r[i + 4], 0x31));
    }
}

/* the NM_RT rows from w (rows of rb bytes), values c0 .. c0 + kc - 1, to the pairs of the tile (pw:
 * [pair][NM_RT]) and the scales (ps: [unit][NM_RT]): 16 values of 8 rows, then a transpose */
static inline __attribute__((always_inline)) void nm_decode_tile_t(const uint8_t *w, size_t rb, const int type, int cols,
                                                                   int c0, int kc, int32_t *pw, float *ps)
{
    const int U = type == KQ_NV4 ? 16 : 32;
    for (int r8 = 0; r8 < NM_RT; r8 += 8) {
        float g[8];
        for (int i = 0; i < 8; ++i) {
            g[i] = type == KQ_NV4 ? 0.5f * kq_nv4_g(w + (size_t)(r8 + i) * rb, cols) : 0.f;
        }
        for (int c = 0; c < kc; c += 16) {
            __m256i r[8];
            float sc[8];
            for (int i = 0; i < 8; ++i) {
                r[i] = nm_dec16(w + (size_t)(r8 + i) * rb, type, cols, c0 + c, g[i], &sc[i]);
            }
            if (c % U == 0) {
                memcpy(ps + (size_t)(c / U) * NM_RT + r8, sc, sizeof sc);
            }
            nm_t8x8(r, pw + (size_t)(c / 2) * NM_RT + r8, NM_RT);
        }
    }
}

static void nm_decode_tile(const uint8_t *w, size_t rb, int type, int cols, int c0, int kc, int32_t *pw, float *ps)
{
    if (type == KQ_IQ2R) {
        nm_decode_tile_t(w, rb, KQ_IQ2R, cols, c0, kc, pw, ps);
    } else if (type == KQ_IQ2_XXS) {
        nm_decode_tile_t(w, rb, KQ_IQ2_XXS, cols, c0, kc, pw, ps);
    } else if (type == KQ_NV4) {
        nm_decode_tile_t(w, rb, KQ_NV4, cols, c0, kc, pw, ps);
    } else {
        nm_decode_tile_t(w, rb, KQ_Q8_0, cols, c0, kc, pw, ps);
    }
}

/* the tile of NM_RT rows: pw the pairs ([pair][NM_RT] int32), ps the scales ([unit][NM_RT]), nu units
 * of U values (kd = U / 2 pairs: constants, the pairs unrolled), on the tokens x[q] (int16, from the
 * chunk's first value), xs[q] (per 32), out o[q] (+ the first row; first: = instead of +=) */
static inline __attribute__((always_inline)) void nm_tile_k(const int32_t *pw, const float *ps, int nu, const int kd,
                                                            const int U, const int16_t *const *x, const float *const *xs,
                                                            float *const *o, int first, int nq)
{
    nm_vf f[2][NM_NQ];
    for (int q = 0; q < NM_NQ; ++q) {
        f[0][q] = first ? NM_ZF() : NM_LDF(o[q]);
        f[1][q] = first ? NM_ZF() : NM_LDF(o[q] + NM_VL);
    }
    for (int u = 0; u < nu; ++u) {
        nm_vi s[2][NM_NQ];
        for (int q = 0; q < NM_NQ; ++q) {
            s[0][q] = s[1][q] = NM_ZI();
        }
        const int32_t *wu = pw + (size_t)u * kd * NM_RT;
        #pragma GCC unroll 16
        for (int k = 0; k < kd; ++k) {
            const nm_vi w0 = NM_LDI(wu + k * NM_RT), w1 = NM_LDI(wu + k * NM_RT + NM_VL);
            #pragma GCC unroll 8
            for (int q = 0; q < NM_NQ; ++q) {
                int32_t xp;
                memcpy(&xp, x[q] + 2 * (u * kd + k), 4);
                const nm_vi xb = NM_SETI(xp);
                NM_DOT(s[0][q], w0, xb);
                NM_DOT(s[1][q], w1, xb);
            }
        }
        const nm_vf s0 = NM_LDF(ps + (size_t)u * NM_RT), s1 = NM_LDF(ps + (size_t)u * NM_RT + NM_VL);
        for (int q = 0; q < NM_NQ; ++q) {
            const nm_vf a = NM_SETF(xs[q][u * U / 32]);
            f[0][q] = NM_FMA(NM_CVT(s[0][q]), NM_MUL(s0, a), f[0][q]);
            f[1][q] = NM_FMA(NM_CVT(s[1][q]), NM_MUL(s1, a), f[1][q]);
        }
    }
    for (int q = 0; q < nq; ++q) {
        NM_STF(o[q], f[0][q]);
        NM_STF(o[q] + NM_VL, f[1][q]);
    }
}

static __attribute__((noinline)) void nm_tile(const int32_t *pw, const float *ps, int nu, int U, const int16_t *const *x,
                                              const float *const *xs, float *const *o, int first, int nq)
{
    if (U == 16) {
        nm_tile_k(pw, ps, nu, 8, 16, x, xs, o, first, nq);
    } else {
        nm_tile_k(pw, ps, nu, 16, 32, x, xs, o, first, nq);
    }
}

/* the bytes of values c0 .. c0 + kc - 1 of NM_RT rows from w to the L2 (the decode reads 32 rows a
 * few lines each: more streams than the prefetchers follow) */
static inline void nm_prefetch_tile(const uint8_t *w, size_t rb, int type, int cols, int c0, int kc)
{
    size_t a, b, a2 = 0, b2 = 0;
    if (type == KQ_IQ2R) {
        a = (size_t)c0 / 4, b = (size_t)(c0 + kc) / 4;
        a2 = (size_t)cols / 4 + (size_t)c0 / 128, b2 = (size_t)cols / 4 + (size_t)(c0 + kc) / 128;
    } else if (type == KQ_IQ2_XXS) {
        a = (size_t)c0 / 256 * 66, b = (size_t)(c0 + kc) / 256 * 66;
    } else if (type == KQ_NV4) {
        a = (size_t)c0 / 2, b = (size_t)(c0 + kc) / 2;
        a2 = (size_t)cols / 2 + (size_t)c0 / 16, b2 = (size_t)cols / 2 + (size_t)(c0 + kc) / 16;
    } else {
        a = (size_t)c0 / 32 * 34, b = (size_t)(c0 + kc) / 32 * 34;
    }
    for (int r = 0; r < NM_RT; ++r) {
        const uint8_t *row = w + (size_t)r * rb;
        for (size_t o = a & ~(size_t)63; o < b; o += 64) {
            _mm_prefetch((const char *)(row + o), _MM_HINT_T1);
        }
        for (size_t o = a2 & ~(size_t)63; o < b2; o += 64) {
            _mm_prefetch((const char *)(row + o), _MM_HINT_T1);
        }
    }
}

/* rows r0 .. r1 - 1 (a multiple of NM_RT) of w on n tokens: xr[i] the token's x (int16), sr[i] its
 * scales (per 32), orow[i] its out (row 0): in chunks of NM_KC columns, each tile of rows decoded
 * once (to the rows of a buffer, 8 x 8 transposes to the pairs) and run on the n tokens */
static void nm_task_tiled(const uint8_t *w, size_t rb, int type, int cols, int r0, int r1, int n,
                          const int16_t *const *xr, const float *const *sr, float *const *orow)
{
    const int U = type == KQ_NV4 ? 16 : 32;
    int32_t *pw = (int32_t *)nm_tile_scratch((size_t)NM_RT * NM_KC * 2 + (size_t)NM_RT * (NM_KC / 16) * 4);
    float *ps = (float *)(pw + NM_RT * NM_KC / 2);
    for (int c0 = 0; c0 < cols; c0 += NM_KC) {
        const int kc = cols - c0 < NM_KC ? cols - c0 : NM_KC, nu = kc / U;
        for (int ro = r0; ro < r1; ro += NM_RT) {
            if (ro == r0) {
                nm_prefetch_tile(w + (size_t)ro * rb, rb, type, cols, c0, kc);
            }
            /* (the next tile's bytes while this one runs) */
            if (ro + NM_RT < r1) {
                nm_prefetch_tile(w + (size_t)(ro + NM_RT) * rb, rb, type, cols, c0, kc);
            } else if (c0 + NM_KC < cols) {
                nm_prefetch_tile(w + (size_t)r0 * rb, rb, type, cols, c0 + NM_KC, cols - c0 - NM_KC < NM_KC ? cols - c0 - NM_KC : NM_KC);
            }
            nm_decode_tile(w + (size_t)ro * rb, rb, type, cols, c0, kc, pw, ps);
            for (int j0 = 0; j0 < n; j0 += NM_NQ) {
                const int16_t *xq[NM_NQ];
                const float *sq[NM_NQ];
                float *oq[NM_NQ];
                for (int q = 0; q < NM_NQ; ++q) {
                    const int j = j0 + q < n ? j0 + q : n - 1;
                    xq[q] = xr[j] + c0;
                    sq[q] = sr[j] + c0 / 32;
                    oq[q] = orow[j] + ro;
                }
                nm_tile(pw, ps, nu, U, xq, sq, oq, c0 == 0, n - j0 < NM_NQ ? n - j0 : NM_NQ);
            }
        }
    }
}

/* the tiles take a type at these columns */
static inline int nm_tiled_ok(int type, int rows, int cols)
{
    const int iq2 = type == KQ_IQ2_XXS || type == KQ_IQ2R;
    return rows % NM_RT == 0 && cols % 32 == 0 && (!iq2 || cols % NM_KC == 0) &&
           (iq2 || type == KQ_NV4 || type == KQ_Q8_0);
}

/* the int16 x of n rows of the two planes (cols each) */
static inline void nm_x16_row(const int8_t *xh, const int8_t *xl, int cols, int16_t *x16)
{
    for (int c = 0; c < cols; c += 16) {
        __m256i a = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xh + c)));
        __m256i b = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(xl + c)));
        _mm256_storeu_si256((__m256i *)(x16 + c), _mm256_add_epi16(_mm256_slli_epi16(a, 7), b));
    }
}

/* the tiled form of GP_NM_LIN2: x16 (t x cols int16, made here); the rows of a task: the threads'
 * shares in tiles, at most 512 */
static void nm_lin2_tiled(const int8_t *xh, const int8_t *xl, const float *xs, const uint8_t *w, int type,
                          int rows, int cols, int t, float *out, int16_t *x16)
{
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        nm_x16_row(xh + (size_t)j * cols, xl + (size_t)j * cols, cols, x16 + (size_t)j * cols);
    }
    const size_t rb = kq_row_bytes(type, cols);
    const int nth = omp_get_num_threads();
    int R = (rows + nth - 1) / nth;
    R = (R + NM_RT - 1) / NM_RT * NM_RT;
    R = R > 512 ? 512 : R;
    const int16_t **xr = (const int16_t **)malloc((size_t)t * 3 * sizeof(void *));
    const float **sr = (const float **)(xr + t);
    float **orow = (float **)(sr + t);
    for (int j = 0; j < t; ++j) {
        xr[j] = x16 + (size_t)j * cols;
        sr[j] = xs + (size_t)j * (cols / 32);
        orow[j] = out + (size_t)j * rows;
    }
    #pragma omp for schedule(dynamic, 1)
    for (int r0 = 0; r0 < rows; r0 += R) {
        nm_task_tiled(w, rb, type, cols, r0, r0 + R < rows ? r0 + R : rows, t, xr, sr, orow);
    }
    free(xr);
}
#endif

/* t rows of x (float32) to the two planes and the scales (kq_quant_part16; xm: scratch) */
static void kq_quant_x2(const float *x, int t, int cols, int8_t *xh, int8_t *xl, float *xs, float *xm)
{
    #pragma omp for schedule(static)
    for (int x0 = 0; x0 < t * (cols / 32); ++x0) {
        int j = x0 / (cols / 32), g = x0 % (cols / 32);
        kq_quant_part16(x + (size_t)j * cols, g, xh + (size_t)j * cols, xl + (size_t)j * cols,
                        xs + (size_t)j * (cols / 32), xm + (size_t)j * (cols / 16));
    }
}

/* GP_NM_LIN2: xh, xl, xs (kq_quant_x2), w, type, rows, cols, t, out (t x rows), and for a prompt
 * group (t >= NM_TILE_T) x16 (t x cols int16: the tiled form) or none; else the rows in tasks of
 * 16, each row on every token (the row stays in the cache). */
static void nm_lin2_body(const int8_t *xh, const int8_t *xl, const float *xs, const uint8_t *w, int type,
                         int rows, int cols, int t, float *out, int16_t *x16)
{
    const size_t rb = kq_row_bytes(type, cols);
#if defined(__AVX2__)
    if (x16 != NULL && t >= NM_TILE_T && nm_tiled_ok(type, rows, cols)) {
        nm_lin2_tiled(xh, xl, xs, w, type, rows, cols, t, out, x16);
        return;
    }
#endif
#if defined(NM_X2_VNNI)
    if ((type == KQ_NV4 || type == KQ_Q8_0) && cols <= 16384) {
        /* each thread makes the unsigned planes of a token once, then its share of the groups of 4
         * rows (static: contiguous rows a thread, as kq_linear_body) */
        uint8_t hu[16384], lu[16384];
        float o[4];
        for (int j = 0; j < t; ++j) {
            const int8_t *h = xh + (size_t)j * cols, *l = xl + (size_t)j * cols;
            const __m256i bias = _mm256_set1_epi8((char)0x80);
            for (int c = 0; c < cols; c += 32) {
                _mm256_storeu_si256((__m256i *)(hu + c), _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(h + c)), bias));
                _mm256_storeu_si256((__m256i *)(lu + c), _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)(l + c)), bias));
            }
            #pragma omp for schedule(static)
            for (int r = 0; r < rows; r += 4) {
                int nr = rows - r < 4 ? rows - r : 4;
#if defined(__AVX512VNNI__) && defined(__AVX512BW__)
                if (cols % 64 == 0) {
                    kq_rows4_x2_512(w + (size_t)r * rb, rb, type, cols, hu, lu, xs + (size_t)j * (cols / 32), nr, o);
                } else
#endif
                kq_rows4_x2_vnni(w + (size_t)r * rb, rb, type, cols, hu, lu, xs + (size_t)j * (cols / 32), nr, o);
                for (int q = 0; q < nr; ++q) {
                    out[(size_t)j * rows + r + q] = o[q];
                }
            }
        }
        return;
    }
#endif
#if defined(__AVX2__)
    if ((type == KQ_IQ2R || type == KQ_IQ2_XXS) && t <= 3) {
        /* the rows in tasks of 16, each decoded once for the tokens (kq_rows_x2_iq2) */
        #pragma omp for schedule(dynamic, 4)
        for (int r0 = 0; r0 < rows; r0 += 16) {
            int nr = r0 + 16 < rows ? 16 : rows - r0;
            kq_rows_x2_iq2(w + (size_t)r0 * rb, rb, type, cols, nr, xh, xl, xs, t, out + r0, 1, (size_t)rows);
        }
        return;
    }
#endif
    #pragma omp for schedule(dynamic, 4)
    for (int r0 = 0; r0 < rows; r0 += 16) {
        int r1 = r0 + 16 < rows ? r0 + 16 : rows;
        for (int r = r0; r < r1; ++r) {
            for (int j = 0; j < t; ++j) {
                out[(size_t)j * rows + r] = kq_dot_x2(w + (size_t)r * rb, type, cols, xh + (size_t)j * cols,
                                                      xl + (size_t)j * cols, xs + (size_t)j * (cols / 32));
            }
        }
    }
}

/* The logits of one row of x (the head on two planes): rc_ for cops.nm_linear2. */
void rc_nm_linear2(const uint8_t *w, int type, int rows, int cols, const float *x, int t, float *out)
{
    int8_t *xh = (int8_t *)malloc((size_t)t * cols), *xl = (int8_t *)malloc((size_t)t * cols);
    float *xs = (float *)malloc((size_t)t * cols / 32 * 4), *xm = (float *)malloc((size_t)t * cols / 16 * 4);
    int16_t *x16 = t >= NM_TILE_T ? (int16_t *)malloc((size_t)t * cols * 2) : NULL;
    #pragma omp parallel
    {
        kq_quant_x2(x, t, cols, xh, xl, xs, xm);
        nm_lin2_body(xh, xl, xs, w, type, rows, cols, t, out, x16);
    }
    free(x16);
    free(xh);
    free(xl);
    free(xs);
    free(xm);
}
