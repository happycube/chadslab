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
 *     y'_j = k_inv sum_b xs_b sum_{i in b} q_i s_ij + (1024 k_inv + k_bias) sum_b xs_b xn_b
 *
 * (the products of ExLlamaV3's int8 GEMV, exl3_gemv_int8_kernel.cuh, with a
 * scale for each 128 values, not for each row). With AVX512-VNNI, vpdpbusd
 * of the products code * KQ_E3_MUL (bytes) and of x splat in the 4 bytes
 * gives q s of 16 weights in one instruction. */

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

/* Block b (128 values) of xh = H (x * suh) as int8: xq, the scale xs, the sum xn. */
static void kq_e3_had_in(const float *x, const uint8_t *suh, int b, int8_t *xq, float *xs, int32_t *xn)
{
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

/* The constant of y' for one row of x: (1024 k_inv + k_bias) sum_b xs_b xn_b. */
static float kq_e3_const(const float *xs, const int32_t *xn, int k)
{
    double c = 0;
    for (int b = 0; b < k / 128; ++b) {
        c += (double)xs[b] * xn[b];
    }
    return (float)((1024.0 * kq_e3_kinv() + kq_e3_kbias()) * c);
}

/* Block ob (128 outputs) of y = (H (y' + c)) * svh for one row. */
static void kq_e3_had_out(const float *yp, const uint8_t *svh, int ob, float c, float *out)
{
    float v[128];
    for (int i = 0; i < 128; ++i) {
        v[i] = yp[128 * ob + i] + c;
    }
    kq_e3_fwht128(v);
    for (int i = 0; i < 128; ++i) {
        out[128 * ob + i] = v[i] * kq_h(svh + 2 * (128 * ob + i));
    }
}

/* The 16 outputs of tile column tj of y' (without the constant) for t rows
 * of x (xq: t x k, row stride k; xs: t x k / 128), into yp (t x n, row
 * stride ys): the scalar form. */
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
                for (int c = 0; c < 16; ++c) {
                    facc[c] += (double)xs[(size_t)j * (k / 128) + ti / 8] * iacc[c];
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
 * 16 G to 16 G + 15): the first word of the 16 words that hold the codes of
 * the group (base, a word of the tile; group 0 starts at the last word, the
 * stream is circular), and for each weight the words that hold its code
 * (lo, and hi = lo - 1) relative to base, and the right shift of lo. */
typedef struct {
    int base[16];
    int32_t lo[16][16], hi[16][16], sr[16][16], sl[16][16];
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
            T->base[g] = base;
            for (int d = 0; d < 16; ++d) {
                int e = (16 * g + d + 1) * K;           /* the end of the code */
                int ib = (e - 1) / 32, sh = 32 * (ib + 1) - e;
                T->lo[g][d] = ib - base;
                T->hi[g][d] = ib - 1 - base;
                T->sr[g][d] = sh;
                T->sl[g][d] = 32 - sh;
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

/* The VNNI form of kq_e3_cols_ref, two rows of x at a time. */
static void kq_e3_cols(const uint8_t *w, int K, int k, int n, int tj, const int8_t *xq,
                       const float *xs, int t, float *yp, size_t ys)
{
    const kq_e3_ktab *T = kq_e3_tab(K);
    const int tn = n / 16, w32 = 8 * K;
    const __m512i ctl0 = kq_e3_splat_ctl(0), ctl1 = kq_e3_splat_ctl(1);
    const __m512i mul = _mm512_set1_epi32((int)KQ_E3_MUL), m16 = _mm512_set1_epi32(0xffff);
    for (int j0 = 0; j0 < t; j0 += 2) {
        const int two = j0 + 1 < t;
        __m512i ia[2][8];
        __m512 fa[2][8];
        for (int c = 0; c < 8; ++c) {
            ia[0][c] = ia[1][c] = _mm512_setzero_si512();
            fa[0][c] = fa[1][c] = _mm512_setzero_ps();
        }
        const int8_t *x0 = xq + (size_t)j0 * k, *x1 = two ? x0 + k : x0;
        for (int ti = 0; ti < k / 16; ++ti) {
            const uint32_t *p = (const uint32_t *)(w + ((size_t)ti * tn + tj) * 32 * K);
            __m512i b0 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(x0 + 16 * ti)));
            __m512i b1 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)(x1 + 16 * ti)));
            __m512i s00 = _mm512_shuffle_epi8(b0, ctl0), s01 = _mm512_shuffle_epi8(b0, ctl1);
            __m512i s10 = _mm512_shuffle_epi8(b1, ctl0), s11 = _mm512_shuffle_epi8(b1, ctl1);
            #pragma GCC unroll 16
            for (int g = 0; g < 16; ++g) {
                __m512i win;
                if (T->base[g] < 0) {
                    /* the last word, then words 0 to 14 */
                    win = _mm512_alignr_epi32(_mm512_loadu_si512(p), _mm512_set1_epi32((int)p[w32 - 1]),
                                              15);
                } else {
                    /* (the last groups read past the tile: the next tile, or
                     * suh after the last tile; those words are not used) */
                    win = _mm512_loadu_si512(p + T->base[g]);
                }
                __m512i lo = _mm512_permutexvar_epi32(_mm512_loadu_si512(T->lo[g]), win);
                __m512i hi = _mm512_permutexvar_epi32(_mm512_loadu_si512(T->hi[g]), win);
                __m512i code = _mm512_and_si512(
                    _mm512_or_si512(_mm512_srlv_epi32(lo, _mm512_loadu_si512(T->sr[g])),
                                    _mm512_sllv_epi32(hi, _mm512_loadu_si512(T->sl[g]))), m16);
                __m512i pr = _mm512_mullo_epi32(code, mul);
                ia[0][g / 2] = _mm512_dpbusd_epi32(ia[0][g / 2], pr, (g & 1) ? s01 : s00);
                ia[1][g / 2] = _mm512_dpbusd_epi32(ia[1][g / 2], pr, (g & 1) ? s11 : s10);
            }
            if (ti % 8 == 7) {
                __m512 xs0 = _mm512_set1_ps(xs[(size_t)j0 * (k / 128) + ti / 8]);
                __m512 xs1 = _mm512_set1_ps(two ? xs[(size_t)(j0 + 1) * (k / 128) + ti / 8] : 0.f);
                for (int c = 0; c < 8; ++c) {
                    fa[0][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia[0][c]), xs0, fa[0][c]);
                    fa[1][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia[1][c]), xs1, fa[1][c]);
                    ia[0][c] = ia[1][c] = _mm512_setzero_si512();
                }
            }
        }
        for (int jj = 0; jj < 1 + two; ++jj) {
            float *o = yp + (size_t)(j0 + jj) * ys + tj * 16;
            for (int c = 0; c < 8; ++c) {
                /* dwords 0-3, 8-11: column c; 4-7, 12-15: column c + 8 */
                o[c] = kq_e3_kinv() * _mm512_mask_reduce_add_ps(0x0f0f, fa[jj][c]);
                o[c + 8] = kq_e3_kinv() * _mm512_mask_reduce_add_ps(0xf0f0, fa[jj][c]);
            }
        }
    }
}
#else
#define kq_e3_cols kq_e3_cols_ref
#endif

/* out (t x n) = x W for a KQ_E3 matrix (in a parallel region; every thread
 * calls it). */
static void kq_e3_linear_body(const uint8_t *w, int K, int n, int k, const float *x, int t, float *out)
{
    const uint8_t *suh = w + (size_t)k * n * K / 8, *svh = suh + 2 * (size_t)k;
    int8_t *xq = NULL;
    #pragma omp single copyprivate(xq)
    xq = (int8_t *)malloc((size_t)t * k + (size_t)t * (k / 128) * 8 + (size_t)t * n * 4 + 64);
    float *xs = (float *)(((uintptr_t)(xq + (size_t)t * k) + 15) & ~(uintptr_t)15);
    int32_t *xn = (int32_t *)(xs + (size_t)t * (k / 128));
    float *yp = (float *)(xn + (size_t)t * (k / 128));
    const int nb = k / 128;
    #pragma omp for schedule(static)
    for (int q = 0; q < t * nb; ++q) {
        int j = q / nb;
        kq_e3_had_in(x + (size_t)j * k, suh, q % nb, xq + (size_t)j * k, xs + (size_t)j * nb,
                     xn + (size_t)j * nb);
    }
    /* tasks of a tile column and up to 8 rows of x */
    const int tn = n / 16, tg = (t + 7) / 8;
    #pragma omp for schedule(dynamic, 1)
    for (int q = 0; q < tn * tg; ++q) {
        int tj = q % tn, j0 = 8 * (q / tn), m = t - j0 < 8 ? t - j0 : 8;
        kq_e3_cols(w, K, k, n, tj, xq + (size_t)j0 * k, xs + (size_t)j0 * nb, m, yp + (size_t)j0 * n,
                   (size_t)n);
    }
    #pragma omp for schedule(static)
    for (int q = 0; q < t * (n / 128); ++q) {
        int j = q / (n / 128);
        kq_e3_had_out(yp + (size_t)j * n, svh, q % (n / 128),
                      kq_e3_const(xs + (size_t)j * nb, xn + (size_t)j * nb, k), out + (size_t)j * n);
    }
    #pragma omp single
    free(xq);
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
