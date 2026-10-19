/* ---------- EXL3 (ExLlamaV3, MIT license; EXL3_PLAN.md) ----------
 *
 * The trellis of a k x n matrix: tiles of 16 inputs by 16 outputs, (k / 16)
 * x (n / 16) of them in that order, each 16 K uint16 words (K bits for each
 * weight). The words of a tile, read as uint32 in memory order, are a
 * circular stream of 256 K bits; weight t of the tile takes the 16 bits that
 * end at bit (t + 1) K (exllamav3_ext/quant/exl3_dq.cuh, dq), its position in
 * the tile that of the fragments of the tensor cores (tensor_core_perm of
 * modules/quant/exl3_lib/quantize.py). Its value from the 16-bit code
 * (quant/codebook.cuh), with the float16 steps of the kernels:
 *   cb 0 3INST: x = code * 89226354 + 64248484, then the two float16 halves of
 *               (x & 0x8fff8fff) ^ 0x3b603b60, added (float16);
 *   cb 1 MCG:   x = code * 0xCBAC1FED, then as 3INST;
 *   cb 2 MUL1:  x = code * 0x83DCD12D, s = the sum of its 4 bytes + 0x6400 as
 *               float16 bits, the value fma(s, 0x1eee, 0xc931) (float16);
 *   cb 3 MUL1 without the rounding to float16: (1024 + s) k_inv + k_bias, the
 *               values of the products of this runtime (KQ_E3).
 * The float16 operations are exact in float64 here, then rounded once to the
 * nearest even float16: the bits of the kernels of ExLlamaV3. */

static float kq_e3_h2f(uint16_t h)
{
    int s = h >> 15, e = (h >> 10) & 31, m = h & 1023;
    double v;
    if (e == 0) {
        v = ldexp((double)m, -24);
    } else if (e == 31) {
        v = m ? NAN : INFINITY;
    } else {
        v = ldexp((double)(m | 1024), e - 25);
    }
    return (float)(s ? -v : v);
}

/* The float16 nearest to d (ties to even), for d an exact sum or product of
 * float16 values (no double rounding). */
static uint16_t kq_e3_d2h(double d)
{
    uint16_t s = d < 0 ? 0x8000 : 0;
    double a = fabs(d);
    if (a == 0.0) {
        return s;
    }
    if (a < ldexp(1.0, -14)) {                  /* the subnormals (or the carry to 2^-14) */
        return (uint16_t)(s | (uint16_t)nearbyint(ldexp(a, 24)));
    }
    int ex;
    frexp(a, &ex);                              /* a = m 2^ex, m in [0.5, 1) */
    int e = ex - 1;                             /* a = 1.f 2^e */
    double m = nearbyint((ldexp(a, -e) - 1.0) * 1024.0);
    if (m >= 1024.0) {
        m = 0.0;
        ++e;
    }
    if (e > 15) {
        return (uint16_t)(s | 0x7c00);
    }
    return (uint16_t)(s | ((e + 15) << 10) | (int)m);
}

#define KQ_E3_MUL 0x83DCD12Du

static inline uint32_t kq_e3_bsum(uint32_t x)
{
    return (x & 255) + ((x >> 8) & 255) + ((x >> 16) & 255) + (x >> 24);
}

static float kq_e3_value(uint32_t code, int cb)
{
    if (cb >= 2) {
        uint32_t sum = kq_e3_bsum(code * KQ_E3_MUL);
        double h = kq_e3_h2f((uint16_t)(0x6400u + sum));
        double v = h * kq_e3_h2f(0x1eee) + kq_e3_h2f(0xc931);
        return cb == 3 ? (float)v : kq_e3_h2f(kq_e3_d2h(v));
    }
    uint32_t x = cb == 1 ? code * 0xCBAC1FEDu : code * 89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    double lo = kq_e3_h2f((uint16_t)(x & 0xffff)), hi = kq_e3_h2f((uint16_t)(x >> 16));
    return kq_e3_h2f(kq_e3_d2h(lo + hi));
}

/* The position in the tile (row r of the 16 inputs, column c of the 16
 * outputs: r 16 + c) of each of the 256 weights of the stream. */
static void kq_e3_perm(int *perm)
{
    for (int t = 0; t < 32; ++t) {
        int r0 = (t % 4) * 2, r1 = r0 + 1, r2 = r0 + 8, r3 = r0 + 9, c0 = t / 4, c1 = c0 + 8;
        int *p = perm + t * 8;
        p[0] = r0 * 16 + c0;
        p[1] = r1 * 16 + c0;
        p[2] = r2 * 16 + c0;
        p[3] = r3 * 16 + c0;
        p[4] = r0 * 16 + c1;
        p[5] = r1 * 16 + c1;
        p[6] = r2 * 16 + c1;
        p[7] = r3 * 16 + c1;
    }
}

/* The 16-bit code of weight t of a tile (w32 = 8 K words). */
static inline uint32_t kq_e3_code(const uint32_t *p, int t, int K, int w32)
{
    int b1 = (t + 1) * K + 256 * K, b0 = b1 - 16;   /* the bits [b0, b1), circular */
    int i0 = b0 / 32, i1 = (b1 - 1) / 32;
    int s0 = (i1 + 1) * 32 - b1;
    uint64_t a = p[i0 % w32], b = p[i1 % w32];
    return (uint32_t)(((a << 32) | b) >> s0) & 0xffff;
}

/* The weights W' (k rows of inputs by n columns of outputs, float32) of the
 * trellis of a matrix in its rotated basis; K bits (1 to 8), codebook cb. */
void rc_exl3_decode(const uint16_t *trellis, int k, int n, int K, int cb, float *out)
{
    int perm[256];
    kq_e3_perm(perm);
    const int tn = n / 16, words = 16 * K, w32 = words / 2;
    #pragma omp parallel for schedule(static)
    for (int64_t tile = 0; tile < (int64_t)(k / 16) * tn; ++tile) {
        const int ti = (int)(tile / tn), tj = (int)(tile % tn);
        const uint32_t *p = (const uint32_t *)(trellis + (size_t)tile * words);
        for (int t = 0; t < 256; ++t) {
            int e = perm[t];
            out[(size_t)(ti * 16 + e / 16) * n + tj * 16 + e % 16] =
                kq_e3_value(kq_e3_code(p, t, K, w32), cb);
        }
    }
}

/* ---------- the MUL1 matrices of this runtime (KQ_E3) ----------
 *
 * Type KQ_E3_BASE + K (K bits, 1 to 8): a matrix of k inputs (cols) and n
 * outputs (rows) is one block: the trellis of the file ((k / 16) (n / 16)
 * tiles of 32 K bytes), then suh (k float16), then svh (n float16); the
 * experts of a layer are such blocks one after the other. The product
 * (reclank/formats/exl3.py):
 *
 *     xh = H (x * suh)          H: the Hadamard transform of each 128 values
 *     y' = xh W'                W': the MUL1 values of the codes (cb 3)
 *     y  = (H y') * svh
 *
 * xh is quantized to int8 with a scale for each 128 values (xs; xn the sum
 * of the int8 values). A MUL1 value is (1024 + s) k_inv + k_bias with s the
 * byte sum of code * KQ_E3_MUL, so
 *
 *     y'_j = k_inv sum_b xs_b sum_{i in b} q_i (s_ij - 510) + (1534 k_inv + k_bias) sum_b xs_b xn_b
 *
 * (the products of ExLlamaV3's int8 GEMV, exl3_gemv_int8_kernel.cuh, with a
 * scale for each 128 values, not for each row). s is 510 on average (the
 * sum of 4 bytes): the integers of a block less 510 xn_b (exact) leave a
 * constant of -0.0018 for each unit of xs xn, in place of -3.45 for the sums
 * of q s, so the float32 sums do not cancel (the outputs are about 1e-4 of
 * those sums: that cancellation was most of the error of the kernel against
 * the decoded weights). With AVX512-VNNI, vpdpbusd of the products code *
 * KQ_E3_MUL (bytes) and of x splat in the 4 bytes gives q s of 16 weights in
 * one instruction. */

#define KQ_E3_MID 510

#define KQ_E3_BASE 70

static inline int kq_e3_bits(int type)
{
    return type > KQ_E3_BASE && type <= KQ_E3_BASE + 8 ? type - KQ_E3_BASE : 0;
}

static inline size_t kq_e3_bytes(int K, int k, int n)
{
    return (size_t)k * n * K / 8 + 2 * (size_t)k + 2 * (size_t)n;
}

static inline float kq_e3_kinv(void)
{
    return 0.0067672729492187500f;     /* float16 0x1eee */
}

static inline float kq_e3_kbias(void)
{
    return -10.3828125f;               /* float16 0xc931 */
}

/* The Hadamard transform of 128 values (Sylvester order, / sqrt(128)). */
static void kq_e3_fwht128(float *v)
{
    for (int h = 1; h < 128; h *= 2) {
        for (int i = 0; i < 128; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                float a = v[j], b = v[j + h];
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
    }
    const float s = 0.08838834764831845f;
    for (int i = 0; i < 128; ++i) {
        v[i] *= s;
    }
}

#if defined(__AVX512F__)
/* The transforms of 128 values in 8 registers: the butterflies of
 * kq_e3_fwht128, each value the same additions in the same order (the same
 * bits): the strides 1 to 8 within a register (the partner from a lane swap;
 * a + b in the lower lane of a pair, a - b in the upper), 16 to 64 across. */
static inline __m512 kq_e3_bfly(__m512 v, __m512 sw, __mmask16 hi)
{
    return _mm512_mask_sub_ps(_mm512_add_ps(v, sw), hi, sw, v);
}

static inline void kq_e3_fwht128_v(__m512 *v)
{
    for (int i = 0; i < 8; ++i) {
        __m512 x = v[i];
        x = kq_e3_bfly(x, _mm512_permute_ps(x, 0xB1), 0xAAAA);
        x = kq_e3_bfly(x, _mm512_permute_ps(x, 0x4E), 0xCCCC);
        x = kq_e3_bfly(x, _mm512_shuffle_f32x4(x, x, 0xB1), 0xF0F0);
        x = kq_e3_bfly(x, _mm512_shuffle_f32x4(x, x, 0x4E), 0xFF00);
        v[i] = x;
    }
    for (int d = 1; d < 8; d *= 2) {
        for (int i = 0; i < 8; ++i) {
            if (!(i & d)) {
                __m512 a = v[i], b = v[i + d];
                v[i] = _mm512_add_ps(a, b);
                v[i + d] = _mm512_sub_ps(a, b);
            }
        }
    }
    const __m512 s = _mm512_set1_ps(0.08838834764831845f);
    for (int i = 0; i < 8; ++i) {
        v[i] = _mm512_mul_ps(v[i], s);
    }
}

/* v = H (x * suh) of block b, and max |v| */
static inline float kq_e3_had_v(const float *x, const uint8_t *suh, int b, __m512 *v)
{
    __m512 m = _mm512_setzero_ps();
    for (int i = 0; i < 8; ++i) {
        __m512 h = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(suh + 2 * (128 * b + 16 * i))));
        v[i] = _mm512_mul_ps(_mm512_loadu_ps(x + 128 * b + 16 * i), h);
    }
    kq_e3_fwht128_v(v);
    for (int i = 0; i < 8; ++i) {
        m = _mm512_max_ps(m, _mm512_abs_ps(v[i]));
    }
    return _mm512_reduce_max_ps(m);
}
#endif

/* Block b (128 values) of xh = H (x * suh) as int8: xq, the scale xs, the sum xn. */
static void kq_e3_had_in(const float *x, const uint8_t *suh, int b, int8_t *xq, float *xs, int32_t *xn)
{
#if defined(__AVX512F__)
    {
        __m512 v[8];
        float m = kq_e3_had_v(x, suh, b, v);
        float inv = m > 0.f ? 127.f / m : 0.f;
        const __m512 vi = _mm512_set1_ps(inv);
        const __m512i lo = _mm512_set1_epi32(-127), hi = _mm512_set1_epi32(127);
        __m512i sum = _mm512_setzero_si512();
        for (int i = 0; i < 8; ++i) {
            __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(v[i], vi));
            q = _mm512_min_epi32(_mm512_max_epi32(q, lo), hi);
            _mm_storeu_si128((__m128i *)(xq + 128 * b + 16 * i), _mm512_cvtepi32_epi8(q));
            sum = _mm512_add_epi32(sum, q);
        }
        xs[b] = m / 127.f;
        xn[b] = _mm512_reduce_add_epi32(sum);
        return;
    }
#endif
    float v[128];
    for (int i = 0; i < 128; ++i) {
        v[i] = x[128 * b + i] * kq_h(suh + 2 * (128 * b + i));
    }
    kq_e3_fwht128(v);
    float m = 0.f;
    for (int i = 0; i < 128; ++i) {
        m = fmaxf(m, fabsf(v[i]));
    }
    float s = m / 127.f, inv = m > 0.f ? 127.f / m : 0.f;
    int32_t sum = 0;
    for (int i = 0; i < 128; ++i) {
        int q = (int)lrintf(v[i] * inv);
        q = q > 127 ? 127 : q < -127 ? -127 : q;
        xq[128 * b + i] = (int8_t)q;
        sum += q;
    }
    xs[b] = s;
    xn[b] = sum;
}

/* Block b of xh as int16 in two int8 planes (RECLANK_E3_X16; the MoE of
 * exl3_moe.c): q16 = round(xh / s16) with s16 = max |xh| / 32512, hi =
 * floor(q16 / 256 + 1/2) (-127 to 127), lo = q16 - 256 hi (-128 to 127).
 * The products are linear in the integers, so those of hi with the scale
 * 256 s16 and of lo with s16 add up to those of q16 (and the constants of
 * kq_e3_const too). */
static void kq_e3_had_in16(const float *x, const uint8_t *suh, int b, int8_t *xh, float *sh, int32_t *nh,
                           int8_t *xl, float *sl, int32_t *nl)
{
#if defined(__AVX512F__)
    {
        __m512 v[8];
        float m = kq_e3_had_v(x, suh, b, v);
        float s = m / 32512.f, inv = m > 0.f ? 32512.f / m : 0.f;
        const __m512 vi = _mm512_set1_ps(inv), q256 = _mm512_set1_ps(1.f / 256.f), half = _mm512_set1_ps(0.5f);
        const __m512i lim = _mm512_set1_epi32(32512);
        __m512i shi = _mm512_setzero_si512(), slo = _mm512_setzero_si512();
        for (int i = 0; i < 8; ++i) {
            __m512i q = _mm512_cvtps_epi32(_mm512_mul_ps(v[i], vi));
            q = _mm512_min_epi32(_mm512_max_epi32(q, _mm512_sub_epi32(_mm512_setzero_si512(), lim)), lim);
            /* hi = floor(q / 256 + 1/2) (exact in float), lo = q - 256 hi */
            __m512 f = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(q), q256), half);
            __m512i h = _mm512_cvtps_epi32(_mm512_roundscale_ps(f, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
            __m512i l = _mm512_sub_epi32(q, _mm512_slli_epi32(h, 8));
            _mm_storeu_si128((__m128i *)(xh + 128 * b + 16 * i), _mm512_cvtepi32_epi8(h));
            _mm_storeu_si128((__m128i *)(xl + 128 * b + 16 * i), _mm512_cvtepi32_epi8(l));
            shi = _mm512_add_epi32(shi, h);
            slo = _mm512_add_epi32(slo, l);
        }
        sh[b] = 256.f * s;
        sl[b] = s;
        nh[b] = _mm512_reduce_add_epi32(shi);
        nl[b] = _mm512_reduce_add_epi32(slo);
        return;
    }
#endif
    float v[128];
    for (int i = 0; i < 128; ++i) {
        v[i] = x[128 * b + i] * kq_h(suh + 2 * (128 * b + i));
    }
    kq_e3_fwht128(v);
    float m = 0.f;
    for (int i = 0; i < 128; ++i) {
        m = fmaxf(m, fabsf(v[i]));
    }
    float s = m / 32512.f, inv = m > 0.f ? 32512.f / m : 0.f;
    int32_t shi = 0, slo = 0;
    for (int i = 0; i < 128; ++i) {
        int q = (int)lrintf(v[i] * inv);
        q = q > 32512 ? 32512 : q < -32512 ? -32512 : q;
        int hi = (int)floorf((float)q / 256.f + 0.5f);
        int lo = q - 256 * hi;
        xh[128 * b + i] = (int8_t)hi;
        xl[128 * b + i] = (int8_t)lo;
        shi += hi;
        slo += lo;
    }
    sh[b] = 256.f * s;
    sl[b] = s;
    nh[b] = shi;
    nl[b] = slo;
}

/* The constant of y' for one row of x: ((1024 + 510) k_inv + k_bias) sum_b
 * xs_b xn_b (the columns of kq_e3_cols are the sums of q (s - 510)). */
static float kq_e3_const(const float *xs, const int32_t *xn, int k)
{
    double c = 0;
    for (int b = 0; b < k / 128; ++b) {
        c += (double)xs[b] * xn[b];
    }
    return (float)(((1024.0 + KQ_E3_MID) * kq_e3_kinv() + kq_e3_kbias()) * c);
}

/* Block ob (128 outputs) of y = (H (y' + c)) * svh for one row. */
static void kq_e3_had_out(const float *yp, const uint8_t *svh, int ob, float c, float *out)
{
#if defined(__AVX512F__)
    {
        __m512 v[8];
        const __m512 vc = _mm512_set1_ps(c);
        for (int i = 0; i < 8; ++i) {
            v[i] = _mm512_add_ps(_mm512_loadu_ps(yp + 128 * ob + 16 * i), vc);
        }
        kq_e3_fwht128_v(v);
        for (int i = 0; i < 8; ++i) {
            __m512 h = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(svh + 2 * (128 * ob + 16 * i))));
            _mm512_storeu_ps(out + 128 * ob + 16 * i, _mm512_mul_ps(v[i], h));
        }
        return;
    }
#endif
    float v[128];
    for (int i = 0; i < 128; ++i) {
        v[i] = yp[128 * ob + i] + c;
    }
    kq_e3_fwht128(v);
    for (int i = 0; i < 128; ++i) {
        out[128 * ob + i] = v[i] * kq_h(svh + 2 * (128 * ob + i));
    }
}

/* The 16 outputs of tile column tj of y' (without the constant: the sums of
 * q (s - 510), kq_e3_const) for t rows of x (xq: t x k, row stride k; xs: t
 * x k / 128), into yp (t x n, row stride ys): the scalar form. */
static void kq_e3_cols_ref(const uint8_t *w, int K, int k, int n, int tj, const int8_t *xq,
                           const float *xs, int t, float *yp, size_t ys)
{
    int perm[256];
    kq_e3_perm(perm);
    const int tn = n / 16, w32 = 8 * K;
    for (int j = 0; j < t; ++j) {
        double facc[16] = {0};
        int32_t iacc[16] = {0};
        const int8_t *xr = xq + (size_t)j * k;
        for (int ti = 0; ti < k / 16; ++ti) {
            const uint32_t *p = (const uint32_t *)(w + ((size_t)ti * tn + tj) * 32 * K);
            for (int s = 0; s < 256; ++s) {
                int e = perm[s];
                iacc[e % 16] += xr[ti * 16 + e / 16] * (int)kq_e3_bsum(kq_e3_code(p, s, K, w32) * KQ_E3_MUL);
            }
            if (ti % 8 == 7) {
                int32_t xn = 0;
                for (int i = 0; i < 128; ++i) {
                    xn += xr[(ti / 8) * 128 + i];
                }
                for (int c = 0; c < 16; ++c) {
                    facc[c] += (double)xs[(size_t)j * (k / 128) + ti / 8] * (iacc[c] - KQ_E3_MID * xn);
                    iacc[c] = 0;
                }
            }
        }
        for (int c = 0; c < 16; ++c) {
            yp[(size_t)j * ys + tj * 16 + c] = (float)(kq_e3_kinv() * facc[c]);
        }
    }
}

#if defined(__AVX512VNNI__)
/* The constants of a K for the 16 groups of 16 weights of a tile (weights
 * 16 G to 16 G + 15): the first word of the window of 16 words that holds
 * the codes of the group (base, a word of the tile; group 0 starts at the
 * last word, the stream is circular), and for each weight the two 16-bit
 * units of the stream that hold its code, for one vpermw (hw: the low and
 * the high halfword of its dword, relative to the window; unit u of the
 * stream is halfword u ^ 1 of the words in memory), and the right shift of
 * the pair (hs). With K >= 2 (16 words or more in a tile) the window of the
 * last groups starts at word 8 K - 16, so that no window reads past the
 * tile: the line after it is the next tile column's (another task's), and
 * its miss cost the time of a tile from memory in every tile. */
typedef struct {
    int base[16];
    uint16_t hw[16][32];
    int32_t hs[16][16];
} kq_e3_ktab;

static kq_e3_ktab kq_e3_tabs[9];
static int kq_e3_tab_done[9];

static const kq_e3_ktab *kq_e3_tab(int K)
{
    kq_e3_ktab *T = &kq_e3_tabs[K];
    if (__atomic_load_n(&kq_e3_tab_done[K], __ATOMIC_ACQUIRE)) {
        return T;
    }
    #pragma omp critical(kq_e3_tab)
    if (!kq_e3_tab_done[K]) {
        for (int g = 0; g < 16; ++g) {
            /* the first bit of the group: (16 g + 1) K - 16, before 0 in group 0 */
            int first = (16 * g + 1) * K - 16;
            int base = first < 0 ? -1 : first / 32;
            if (K >= 2 && base > 8 * K - 16) {
                base = 8 * K - 16;
            }
            T->base[g] = base;
            for (int d = 0; d < 16; ++d) {
                int e = (16 * g + d + 1) * K;           /* the end of the code */
                /* the units ua, ub = ua + 1 of the bits [e - 16, e) (circular:
                 * 16 K units in a tile) */
                int nu = 16 * K, ub = (e / 16) % nu, ua = (e / 16 - 1 + nu) % nu;
                T->hw[g][2 * d] = (uint16_t)(((ub ^ 1) - 2 * base + nu) % nu);
                T->hw[g][2 * d + 1] = (uint16_t)(((ua ^ 1) - 2 * base + nu) % nu);
                T->hs[g][d] = 16 - e % 16;
            }
        }
        __atomic_store_n(&kq_e3_tab_done[K], 1, __ATOMIC_RELEASE);
    }
    return T;
}

/* The bytes of the x splats of a row of 16 inputs: for the weights of
 * groups of parity P, dword d of the group reads row
 * 2 (2 P + d / 8) + {0, 1, 8, 9}[d % 4] (kq_e3_perm). */
static __m512i kq_e3_splat_ctl(int P)
{
    static const int add[4] = {0, 1, 8, 9};
    uint8_t c[64];
    for (int d = 0; d < 16; ++d) {
        int row = 2 * (2 * P + d / 8) + add[d % 4];
        for (int b = 0; b < 4; ++b) {
            c[4 * d + b] = (uint8_t)row;
        }
    }
    return _mm512_loadu_si512(c);
}

/* The 16 column sums of the 8 accumulators of a block of 128 inputs (ia[c]:
 * dwords 0-3 and 8-11 column c, 4-7 and 12-15 column c + 8), in column
 * order; exact. */
static inline __m512i kq_e3_fold(const __m512i *ia)
{
    __m256i u[8];
    for (int c = 0; c < 8; ++c) {
        u[c] = _mm256_add_epi32(_mm512_castsi512_si256(ia[c]), _mm512_extracti64x4_epi64(ia[c], 1));
    }
    /* lane 0: columns 0-3 (4-7), lane 1: columns 8-11 (12-15) */
    __m256i a = _mm256_hadd_epi32(_mm256_hadd_epi32(u[0], u[1]), _mm256_hadd_epi32(u[2], u[3]));
    __m256i b = _mm256_hadd_epi32(_mm256_hadd_epi32(u[4], u[5]), _mm256_hadd_epi32(u[6], u[7]));
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_permute2x128_si256(a, b, 0x20)),
                              _mm256_permute2x128_si256(a, b, 0x31), 1);
}

/* The sum of 128 int8 values (xn of a block). */
static inline int32_t kq_e3_sum128(const int8_t *x)
{
    const __m512i one = _mm512_set1_epi8(1);
    __m512i a = _mm512_dpbusd_epi32(_mm512_setzero_si512(), one, _mm512_loadu_si512(x));
    return _mm512_reduce_add_epi32(_mm512_dpbusd_epi32(a, one, _mm512_loadu_si512(x + 64)));
}

/* The tile rows ahead of a column that a task prefetches: its tiles are a
 * row of tiles apart (n / 16 tiles, 40 KB for 5120 outputs at 4 bits), a
 * page or more, which the hardware prefetchers do not cross. */
#define KQ_E3_PF 20

/* The products code * KQ_E3_MUL of the 16 weights of group g of the tile at
 * p (its 8 K words). */
static inline __attribute__((always_inline)) __m512i kq_e3_group(const uint32_t *p, int w32,
                                                                 const kq_e3_ktab *T, int g)
{
    __m512i win;
    if (T->base[g] < 0) {
        /* the last word, then words 0 to 14 */
        win = _mm512_alignr_epi32(_mm512_loadu_si512(p), _mm512_set1_epi32((int)p[w32 - 1]), 15);
    } else {
        /* (K 1: the last groups read past the tile, the next tile or suh
         * after the last tile; those words are not used) */
        win = _mm512_loadu_si512(p + T->base[g]);
    }
    /* the two units of each code in a dword (one vpermw), shifted */
    __m512i pair = _mm512_permutexvar_epi16(_mm512_loadu_si512(T->hw[g]), win);
    __m512i code = _mm512_and_si512(_mm512_srlv_epi32(pair, _mm512_loadu_si512(T->hs[g])),
                                    _mm512_set1_epi32(0xffff));
    return _mm512_mullo_epi32(code, _mm512_set1_epi32((int)KQ_E3_MUL));
}

/* Prefetch the tile at pf (tb bytes; K 1 reads 32 bytes past it). */
static inline void kq_e3_prefetch(const char *pf, size_t tb, int K)
{
    for (size_t o = 0; o < tb; o += 64) {
        _mm_prefetch(pf + o, _MM_HINT_T0);
    }
    _mm_prefetch(pf + (K >= 2 ? tb - 1 : tb + 31), _MM_HINT_T0);
}

/* The VNNI form of kq_e3_cols_ref for R (1 or 2) rows of x: the codes of a
 * group of 16 weights once, a vpdpbusd for each row. The integers of each
 * block of 128 inputs go to the 16 sums of the columns (kq_e3_fold), then
 * to float32 with the scale of the block. */
static inline __attribute__((always_inline)) void kq_e3_cols_rows(
    const uint8_t *w, int K, int k, int n, int tj, const int8_t *x0, const int8_t *x1, const float *xs0,
    const float *xs1, float *o0, float *o1, const int R, const kq_e3_ktab *T, __m512i ctl0, __m512i ctl1)
{
    const int tn = n / 16, w32 = 8 * K;
    const size_t tb = 32 * (size_t)K, rs = (size_t)tn * tb;     /* the bytes of a tile, of a row of tiles */
    __m512i ia[2][8];
    __m512 fa[2] = {_mm512_setzero_ps(), _mm512_setzero_ps()};
    for (int c = 0; c < 8; ++c) {
        ia[0][c] = ia[1][c] = _mm512_setzero_si512();
    }
    const uint8_t *pw = w + (size_t)tj * tb;
    const int nti = k / 16;
    for (int ti = 0; ti < nti; ++ti, pw += rs) {
        if (ti + KQ_E3_PF < nti) {
            kq_e3_prefetch((const char *)(pw + KQ_E3_PF * rs), tb, K);
        } else if (tj + 1 < tn) {
            /* the first tiles of the next column, the next task of a thread
             * as the tasks of a matrix split (a column of down, 40 tiles, ran
             * half its tiles with none ahead) */
            kq_e3_prefetch((const char *)(w + (size_t)(tj + 1) * tb + (size_t)(ti + KQ_E3_PF - nti) * rs), tb, K);
        }
        const uint32_t *p = (const uint32_t *)pw;
        __m512i b0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(x0 + 16 * ti)));
        __m512i s00 = _mm512_shuffle_epi8(b0, ctl0), s01 = _mm512_shuffle_epi8(b0, ctl1);
        __m512i s10 = s00, s11 = s01;
        if (R == 2) {
            __m512i b1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(x1 + 16 * ti)));
            s10 = _mm512_shuffle_epi8(b1, ctl0);
            s11 = _mm512_shuffle_epi8(b1, ctl1);
        }
        #pragma GCC unroll 16
        for (int g = 0; g < 16; ++g) {
            __m512i pr = kq_e3_group(p, w32, T, g);
            ia[0][g / 2] = _mm512_dpbusd_epi32(ia[0][g / 2], pr, (g & 1) ? s01 : s00);
            if (R == 2) {
                ia[1][g / 2] = _mm512_dpbusd_epi32(ia[1][g / 2], pr, (g & 1) ? s11 : s10);
            }
        }
        if (ti % 8 == 7) {
            /* the 16 columns less 510 xn of the block (kq_e3_const) */
            const int b = ti / 8;
            __m512i c0 = _mm512_sub_epi32(kq_e3_fold(ia[0]),
                                          _mm512_set1_epi32(KQ_E3_MID * kq_e3_sum128(x0 + 128 * b)));
            fa[0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(xs0[b]), fa[0]);
            if (R == 2) {
                __m512i c1 = _mm512_sub_epi32(kq_e3_fold(ia[1]),
                                              _mm512_set1_epi32(KQ_E3_MID * kq_e3_sum128(x1 + 128 * b)));
                fa[1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c1), _mm512_set1_ps(xs1[b]), fa[1]);
            }
            for (int c = 0; c < 8; ++c) {
                ia[0][c] = ia[1][c] = _mm512_setzero_si512();
            }
        }
    }
    const __m512 kinv = _mm512_set1_ps(kq_e3_kinv());
    _mm512_storeu_ps(o0 + tj * 16, _mm512_mul_ps(fa[0], kinv));
    if (R == 2) {
        _mm512_storeu_ps(o1 + tj * 16, _mm512_mul_ps(fa[1], kinv));
    }
}

/* kq_e3_cols_rows for t rows (3 to 8) of x: the codes of each block of 8
 * tiles once, to the products of a buffer (8 KB, in L1), then the
 * vpdpbusd of each pair of rows over it. */
static void kq_e3_cols_many(const uint8_t *w, int K, int k, int n, int tj, const int8_t *xq,
                            const float *xs, int t, float *yp, size_t ys, const kq_e3_ktab *T, __m512i ctl0,
                            __m512i ctl1)
{
    const int tn = n / 16, w32 = 8 * K, nb = k / 128;
    const size_t tb = 32 * (size_t)K, rs = (size_t)tn * tb;
    __m512i pr[8][16];
    __m512 fa[8];
    for (int j = 0; j < t; ++j) {
        fa[j] = _mm512_setzero_ps();
    }
    const uint8_t *pw = w + (size_t)tj * tb;
    for (int b = 0; b < nb; ++b, pw += 8 * rs) {
        if (b + 1 < nb) {
            for (int r = 0; r < 8; ++r) {
                kq_e3_prefetch((const char *)(pw + (8 + r) * rs), tb, K);
            }
        } else if (tj + 1 < tn) {
            for (int r = 0; r < 8; ++r) {       /* (the next column, as kq_e3_cols_rows) */
                kq_e3_prefetch((const char *)(w + (size_t)(tj + 1) * tb + (size_t)r * rs), tb, K);
            }
        }
        for (int r = 0; r < 8; ++r) {
            const uint32_t *p = (const uint32_t *)(pw + r * rs);
            #pragma GCC unroll 16
            for (int g = 0; g < 16; ++g) {
                pr[r][g] = kq_e3_group(p, w32, T, g);
            }
        }
        for (int j0 = 0; j0 < t; j0 += 2) {
            const int two = j0 + 1 < t;
            const int8_t *x0 = xq + (size_t)j0 * k + 128 * b, *x1 = two ? x0 + k : x0;
            __m512i ia[2][8];
            for (int c = 0; c < 8; ++c) {
                ia[0][c] = ia[1][c] = _mm512_setzero_si512();
            }
            for (int r = 0; r < 8; ++r) {
                __m512i b0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(x0 + 16 * r)));
                __m512i b1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(x1 + 16 * r)));
                __m512i s00 = _mm512_shuffle_epi8(b0, ctl0), s01 = _mm512_shuffle_epi8(b0, ctl1);
                __m512i s10 = _mm512_shuffle_epi8(b1, ctl0), s11 = _mm512_shuffle_epi8(b1, ctl1);
                #pragma GCC unroll 16
                for (int g = 0; g < 16; ++g) {
                    ia[0][g / 2] = _mm512_dpbusd_epi32(ia[0][g / 2], pr[r][g], (g & 1) ? s01 : s00);
                    ia[1][g / 2] = _mm512_dpbusd_epi32(ia[1][g / 2], pr[r][g], (g & 1) ? s11 : s10);
                }
            }
            /* the 16 columns less 510 xn of the block (kq_e3_const) */
            __m512i c0 = _mm512_sub_epi32(kq_e3_fold(ia[0]),
                                          _mm512_set1_epi32(KQ_E3_MID * kq_e3_sum128(x0)));
            fa[j0] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c0), _mm512_set1_ps(xs[(size_t)j0 * nb + b]), fa[j0]);
            if (two) {
                __m512i c1 = _mm512_sub_epi32(kq_e3_fold(ia[1]),
                                              _mm512_set1_epi32(KQ_E3_MID * kq_e3_sum128(x1)));
                fa[j0 + 1] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(c1),
                                             _mm512_set1_ps(xs[(size_t)(j0 + 1) * nb + b]), fa[j0 + 1]);
            }
        }
    }
    const __m512 kinv = _mm512_set1_ps(kq_e3_kinv());
    for (int j = 0; j < t; ++j) {
        _mm512_storeu_ps(yp + (size_t)j * ys + tj * 16, _mm512_mul_ps(fa[j], kinv));
    }
}

/* The VNNI form of kq_e3_cols_ref: one or two rows of x as they come (the
 * codes of a group for one or two vpdpbusd); 3 or more rows in chunks of 8
 * with the codes of each block once (kq_e3_cols_many).
 *
 * Not faster: the weights decoded ahead (s - 510 as int16 in the order of
 * the stream, two rows of a column in a dword, vpdpwssd; the bits of this
 * kernel), e.g. the cold experts of the next layer while the CPU waits for
 * the GPU (its router on the layer's input before attention picks 81% of
 * them). On the Xeon 8268 (Qwen3.8-Flash-Next EXL3 5.05, a node's half of 7
 * cold experts, 34 MB) the int16 products ran at 0.6-0.7x of this kernel:
 * from L3 one core reads about 12 GB/s, 6 G weights/s at 2 bytes, this
 * kernel's rate; from L2 (1 MB a core, 2-3 half experts a node) 1.4-1.7x;
 * and the decode ahead took longer than these products. */
static void kq_e3_cols(const uint8_t *w, int K, int k, int n, int tj, const int8_t *xq,
                       const float *xs, int t, float *yp, size_t ys)
{
    const kq_e3_ktab *T = kq_e3_tab(K);
    const __m512i ctl0 = kq_e3_splat_ctl(0), ctl1 = kq_e3_splat_ctl(1);
    const int nb = k / 128;
    if (t >= 3) {
        for (int j0 = 0; j0 < t; j0 += 8) {
            kq_e3_cols_many(w, K, k, n, tj, xq + (size_t)j0 * k, xs + (size_t)j0 * nb,
                            t - j0 < 8 ? t - j0 : 8, yp + (size_t)j0 * ys, ys, T, ctl0, ctl1);
        }
        return;
    }
    int j0 = 0;
    for (; j0 + 1 < t; j0 += 2) {
        kq_e3_cols_rows(w, K, k, n, tj, xq + (size_t)j0 * k, xq + (size_t)(j0 + 1) * k,
                        xs + (size_t)j0 * nb, xs + (size_t)(j0 + 1) * nb, yp + (size_t)j0 * ys,
                        yp + (size_t)(j0 + 1) * ys, 2, T, ctl0, ctl1);
    }
    if (j0 < t) {
        kq_e3_cols_rows(w, K, k, n, tj, xq + (size_t)j0 * k, NULL, xs + (size_t)j0 * nb, NULL,
                        yp + (size_t)j0 * ys, NULL, 1, T, ctl0, ctl1);
    }
}
#else
#define kq_e3_cols kq_e3_cols_ref
#endif

/* The float32 rows of the n-gram table of an EXL3 directory (formats/
 * exl3_qwen4.py, ngram_decode, whose bits these are): n packed rows of words
 * int16 (a float16 scale, then a ring of dim K-bit symbols, dim = 16 (words
 * - 1) / K); value i of a row is tab[c] scale + bias[head[r]] (dim each),
 * c the low 16 bits of the symbols i, i - 1, .. (circular) at K bits each.
 * On the calling thread (rc_kq_gather); NumPy took 2.7 s of a prompt of
 * 8192 tokens. No fma: the product and the sum rounded as NumPy does. */
__attribute__((optimize("fp-contract=off")))
void rc_e3_ngram_decode(const int16_t *packed, int64_t n, int words, int K, const float *tab, const float *bias,
                        const int64_t *head, float *out)
{
    const int nw = words - 1, dim = nw * 16 / K, J = (15 + K) / K;
    const uint64_t mask = ((uint64_t)1 << K) - 1;
    uint64_t sym[1024];
    if (dim > 1024) {
        return;
    }
    for (int64_t r = 0; r < n; ++r) {
        const uint16_t *row = (const uint16_t *)(packed + (size_t)r * words);
        const float sc = kq_h((const uint8_t *)row);
        const uint16_t *st = row + 1;
        for (int i = 0; i < dim; ++i) {
            int bit = i * K, w = bit >> 4;
            uint64_t win = (uint64_t)st[w] | ((uint64_t)st[(w + 1) % nw] << 16);
            sym[i] = (win >> (bit & 15)) & mask;
        }
        const float *b = bias + (size_t)head[r] * dim;
        float *o = out + (size_t)r * dim;
        /* the code of value i slides: (code of i - 1) << K | symbol i, low
         * 16 bits (J K >= 16: the symbol that leaves is past them) */
        uint64_t c = 0;
        for (int j = J - 1; j >= 1; --j) {
            c = (c << K) | sym[dim - j];
        }
        for (int i = 0; i < dim; ++i) {
            c = ((c << K) | sym[i]) & 0xFFFF;
            float v = tab[c] * sc;
            o[i] = v + b[i];
        }
    }
}

/* out (t x n) = x W for a KQ_E3 matrix (in a parallel region; every thread
 * calls it). */
static int kq_e3_x16(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_E3_X16");
        on = v != NULL && v[0] == '1';
    }
    return on;
}

static void kq_e3_linear_body(const uint8_t *w, int K, int n, int k, const float *x, int t, float *out)
{
    const uint8_t *suh = w + (size_t)k * n * K / 8, *svh = suh + 2 * (size_t)k;
    /* RECLANK_E3_X16=1: x as int16 in two int8 planes (kq_e3_had_in16), a
     * second pass of the products for the low plane */
    const int x16 = kq_e3_x16(), pl = x16 ? 2 : 1;
    const int nb = k / 128;
    const size_t one = (size_t)t * k + (size_t)t * nb * 8 + (size_t)t * n * 4 + 64;
    int8_t *mem = NULL;
    #pragma omp single copyprivate(mem)
    mem = (int8_t *)malloc(pl * one);
    int8_t *xq[2];
    float *xs[2], *yp[2];
    int32_t *xn[2];
    for (int u = 0; u < pl; ++u) {
        xq[u] = mem + u * one;
        xs[u] = (float *)(((uintptr_t)(xq[u] + (size_t)t * k) + 15) & ~(uintptr_t)15);
        xn[u] = (int32_t *)(xs[u] + (size_t)t * nb);
        yp[u] = (float *)(xn[u] + (size_t)t * nb);
    }
    #pragma omp for schedule(static)
    for (int q = 0; q < t * nb; ++q) {
        int j = q / nb;
        if (x16) {
            kq_e3_had_in16(x + (size_t)j * k, suh, q % nb, xq[0] + (size_t)j * k, xs[0] + (size_t)j * nb,
                           xn[0] + (size_t)j * nb, xq[1] + (size_t)j * k, xs[1] + (size_t)j * nb,
                           xn[1] + (size_t)j * nb);
        } else {
            kq_e3_had_in(x + (size_t)j * k, suh, q % nb, xq[0] + (size_t)j * k, xs[0] + (size_t)j * nb,
                         xn[0] + (size_t)j * nb);
        }
    }
    /* tasks of a plane, a tile column and up to 8 rows of x */
    const int tn = n / 16, tg = (t + 7) / 8;
    #pragma omp for schedule(dynamic, 1)
    for (int q = 0; q < pl * tn * tg; ++q) {
        int u = q / (tn * tg), r = q % (tn * tg);
        int tj = r % tn, j0 = 8 * (r / tn), m = t - j0 < 8 ? t - j0 : 8;
        kq_e3_cols(w, K, k, n, tj, xq[u] + (size_t)j0 * k, xs[u] + (size_t)j0 * nb, m, yp[u] + (size_t)j0 * n,
                   (size_t)n);
    }
    #pragma omp for schedule(static)
    for (int q = 0; q < t * (n / 128); ++q) {
        int j = q / (n / 128), ob = q % (n / 128);
        float c = kq_e3_const(xs[0] + (size_t)j * nb, xn[0] + (size_t)j * nb, k);
        if (x16) {
            float *y0 = yp[0] + (size_t)j * n + 128 * ob;
            const float *y1 = yp[1] + (size_t)j * n + 128 * ob;
            for (int i = 0; i < 128; ++i) {
                y0[i] += y1[i];
            }
            c += kq_e3_const(xs[1] + (size_t)j * nb, xn[1] + (size_t)j * nb, k);
        }
        kq_e3_had_out(yp[0] + (size_t)j * n, svh, ob, c, out + (size_t)j * n);
    }
    #pragma omp single
    free(mem);
}

void rc_e3_linear(const uint8_t *w, int K, int n, int k, const float *x, int t, float *out)
{
    #pragma omp parallel
    kq_e3_linear_body(w, K, n, k, x, t, out);
}

/* The scalar products (a check of the VNNI form): yp (t x n) = y' without the
 * constant, from xq and xs of rc_e3_had_in. */
void rc_e3_cols(const uint8_t *w, int K, int k, int n, const int8_t *xq, const float *xs, int t,
                float *yp, int ref)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int tj = 0; tj < n / 16; ++tj) {
        if (ref) {
            kq_e3_cols_ref(w, K, k, n, tj, xq, xs, t, yp, (size_t)n);
        } else {
            kq_e3_cols(w, K, k, n, tj, xq, xs, t, yp, (size_t)n);
        }
    }
}

void rc_e3_had_in(const float *x, const uint8_t *suh, int k, int t, int8_t *xq, float *xs, int32_t *xn)
{
    for (int j = 0; j < t; ++j) {
        for (int b = 0; b < k / 128; ++b) {
            kq_e3_had_in(x + (size_t)j * k, suh, b, xq + (size_t)j * k, xs + (size_t)j * (k / 128),
                         xn + (size_t)j * (k / 128));
        }
    }
}
