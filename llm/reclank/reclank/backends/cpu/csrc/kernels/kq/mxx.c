/* DeepSeek V4.1 Flash (DEEPSEEK_V41_PLAN.md 3, Phase 1): its two weight types.
 *
 *     KQ_MXX  64   MXFP4 in groups of 16 rows (the routed experts): the E2M1
 *                  codes of the checkpoint (two a byte, the low half the
 *                  first value) with an E8M0 scale for each 32 values of a
 *                  row (2^(e - 127)). A group: for each block of 32 columns
 *                  272 bytes: 8 steps of 32 bytes (the codes of KQ_NVX: step
 *                  s holds values 4s to 4s + 3 of the 16 rows; byte 4r + u,
 *                  r < 8, has value 4s + u of row r in its low half and of
 *                  row r + 8 in its high half), then the 16 E8M0 scales of
 *                  the rows. A "row" is cols / 32 * 17 bytes. No other
 *                  scale: w = 2^(e - 127) e2m1(code).
 *     KQ_FP8B32 65 FP8 E4M3 rows as stored, with an E8M0 scale for each block
 *                  of 32 x 32 (the dense matrices): the codes (rows x cols),
 *                  then the scales (rows / 32 x cols / 32, rounded up).
 *
 * The products (near, the default; exact, RECLANK_DSV41_EXACT):
 *
 *   KQ_MXX near: x as int16 in two int8 planes (kq_quant_part16: v = 128 hi
 *     + lo, xs for each 32). For each row and block the exact int32 sum a =
 *     sum(2 e2m1 v) (the VNNI step of KQ_NVX: the codes + 12 as u8, from -12
 *     sum(x)); then p = a * 2^(e - 127) (exact: |a| < 2^24, a power of two)
 *     and f = fma(p, xs, f) block after block; the row is 0.5 f. The GPU
 *     takes the same operations (the bits of a token do not depend on where
 *     its expert runs).
 *   KQ_MXX exact: x rounded as the reference rounds it (act_quant of
 *     kernel.py: E4M3 with the power of two above amax / 448 for each 32,
 *     amax at least 1e-4); for each block the float32 sum of the 32 products
 *     of the E4M3 and E2M1 values (in order), times both scales (powers of
 *     two: exact), added to f.
 *   KQ_FP8B32: near takes x as float32; exact rounds it as above. For each
 *     block of 32 columns the float32 sum of the products, times the block's
 *     scales, added.
 */

/* The E2M1 values (convert.py FP4_TABLE); twice them: kq_e2m1x2 of helpers.c. */
static const float kq_e2m1f[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                                   -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

static inline float kq_e8m0f(uint8_t e)
{
    uint32_t u = e ? (uint32_t)e << 23 : 0x00400000u;     /* e = 0: 2^-127 */
    float f;
    memcpy(&f, &u, 4);
    return f;
}

static inline float kq_e4m3f(uint8_t b)
{
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? (float)m * (1.f / 512.f) : ldexpf(1.f + (float)m / 8.f, e - 7);
    return b & 128 ? -v : v;
}

/* float (|v| <= 448) to the nearest E4M3 value, ties to even (the cast of
 * the reference: __nv_cvt_float_to_fp8 with satfinite, after its clamp). */
static inline float kq_round_e4m3(float v)
{
    float a = fabsf(v);
    if (a == 0.f) {
        return v;
    }
    int e;
    frexpf(a, &e);                          /* a = f 2^e, f in [0.5, 1) */
    int ex = e - 1;                         /* a in [2^ex, 2^(ex+1)) */
    if (ex < -6) {
        ex = -6;                            /* the subnormals: steps of 2^-9 */
    }
    float step = ldexpf(1.f, ex - 3);
    float q = rintf(a / step) * step;       /* rintf: ties to even in the default mode */
    if (q > 448.f) {
        q = 448.f;
    }
    return v < 0.f ? -q : q;
}

/* fast_round_scale of kernel.py: 2^ceil(log2(v)) from the bits of v. */
static inline float kq_pow2_ceil(float v)
{
    uint32_t u;
    memcpy(&u, &v, 4);
    int e = (int)((u >> 23) & 0xff) - 127 + ((u & 0x7fffff) != 0);
    uint32_t r = (uint32_t)(e + 127) << 23;
    float f;
    memcpy(&f, &r, 4);
    return f;
}

/* act_quant of the reference on 32 values: the E4M3 values (as float) and
 * the scale. */
static void kq_act_e4m3_32(const float *x, float *q, float *s)
{
    float amax = 0.f;
    for (int i = 0; i < 32; ++i) {
        amax = fmaxf(amax, fabsf(x[i]));
    }
    amax = fmaxf(amax, 1e-4f);
    float sc = kq_pow2_ceil(amax * (1.f / 448.f));
    for (int i = 0; i < 32; ++i) {
        float v = x[i] / sc;
        v = v > 448.f ? 448.f : v < -448.f ? -448.f : v;
        q[i] = kq_round_e4m3(v);
    }
    *s = sc;
}

/* MXFP4 rows of the checkpoint (codes: rows x cols / 2 bytes, scales: rows x
 * cols / 32) to KQ_MXX groups (rows a multiple of 16). Returns the count of
 * scales 0 or 255 (2^-127 and NaN: none expected). */
int64_t rc_kq_mxx_pack(const uint8_t *codes, const uint8_t *scales, int rows, int cols, uint8_t *out)
{
    int nb = cols / 32;
    int64_t odd = 0;
    #pragma omp parallel for schedule(static) reduction(+:odd)
    for (int g = 0; g < rows / 16; ++g) {
        for (int b = 0; b < nb; ++b) {
            uint8_t *blk = out + ((size_t)g * nb + b) * KQ_MXX_BB;
            for (int s = 0; s < 8; ++s) {
                for (int r = 0; r < 8; ++r) {
                    for (int u = 0; u < 4; ++u) {
                        int v = 32 * b + 4 * s + u;
                        const uint8_t *c0 = codes + (size_t)(16 * g + r) * (cols / 2);
                        const uint8_t *c1 = codes + (size_t)(16 * g + r + 8) * (cols / 2);
                        int lo = (c0[v / 2] >> (4 * (v & 1))) & 15;
                        int hi = (c1[v / 2] >> (4 * (v & 1))) & 15;
                        blk[32 * s + 4 * r + u] = (uint8_t)(lo | hi << 4);
                    }
                }
            }
            for (int r = 0; r < 16; ++r) {
                uint8_t e = scales[(size_t)(16 * g + r) * nb + b];
                blk[256 + r] = e;
                odd += e == 0 || e == 255;
            }
        }
    }
    return odd;
}

/* rc_kq_mxx_pack on the calling thread alone (the zygote packs from many
 * threads at once). */
void rc_kq_mxx_pack_serial(const uint8_t *codes, const uint8_t *scales, int rows, int cols, uint8_t *out)
{
    int nb = cols / 32;
    for (int g = 0; g < rows / 16; ++g) {
        for (int b = 0; b < nb; ++b) {
            uint8_t *blk = out + ((size_t)g * nb + b) * KQ_MXX_BB;
            for (int s = 0; s < 8; ++s) {
                for (int r = 0; r < 8; ++r) {
                    const uint8_t *c0 = codes + (size_t)(16 * g + r) * (cols / 2) + 16 * b + 2 * s;
                    const uint8_t *c1 = codes + (size_t)(16 * g + r + 8) * (cols / 2) + 16 * b + 2 * s;
                    /* values 4s .. 4s + 3 of the block: bytes 2s and 2s + 1 of its 16 */
                    uint8_t *o = blk + 32 * s + 4 * r;
                    o[0] = (uint8_t)((c0[0] & 15) | (c1[0] & 15) << 4);
                    o[1] = (uint8_t)((c0[0] >> 4) | (c1[0] & 0xf0));
                    o[2] = (uint8_t)((c0[1] & 15) | (c1[1] & 15) << 4);
                    o[3] = (uint8_t)((c0[1] >> 4) | (c1[1] & 0xf0));
                }
            }
            for (int r = 0; r < 16; ++r) {
                blk[256 + r] = scales[(size_t)(16 * g + r) * nb + b];
            }
        }
    }
}

/* The code of row r (0 to 15) and value v (0 to 31) of a KQ_MXX block. */
static inline int kq_mxx_code(const uint8_t *blk, int r, int v)
{
    return (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
}

/* The float values of the rows of a KQ_MXX matrix (a check). */
void rc_kq_mxx_dequant(const uint8_t *w, int rows, int cols, float *out)
{
    int nb = cols / 32;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wg = w + (size_t)(i / 16) * nb * KQ_MXX_BB;
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + (size_t)b * KQ_MXX_BB;
            float s = kq_e8m0f(blk[256 + i % 16]);
            for (int v = 0; v < 32; ++v) {
                out[(size_t)i * cols + 32 * b + v] = kq_e2m1f[kq_mxx_code(blk, i % 16, v)] * s;
            }
        }
    }
}

#if defined(__AVX512VNNI__)
/* The bytes ahead of its block that kq_mxx_rows2_wide and kq_mxx_rows1 ask
 * the caches for (RECLANK_MXX_PF; 0: none). */
static int kq_mxx_pf(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_MXX_PF");
        n = v ? atoi(v) : 0;
    }
    return n;
}

/* RECLANK_MXX_TAIL (default 1; 0: a measure): the tokens of
 * kq_mxx_rows2_wide past its fours in twos and ones with split chains. */
static int kq_mxx_tail(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_MXX_TAIL");
        n = v ? atoi(v) : 1;
    }
    return n;
}

/* The bytes of step 32 bytes at p of a KQ_MXX block as the u8 weights of its
 * 16 rows (the low halves: rows 0-7 in the low 256 bits; the high halves:
 * rows 8-15 in the high 256), e2m1 x 2 + 12 (kq_e2m1u): the 32 bytes in both
 * halves (a broadcast load), the high half shifted down 4, the nibbles, the
 * table: 3 operations of ports 0 and 5 (an and, a shift and an and on 256
 * bits, an insert, the table: 5 before). */
static inline __m512i kq_mxx_w(const uint8_t *p, __m512i lut)
{
    __m512i q = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)p));
    q = _mm512_mask_srli_epi16(q, (__mmask32)0xFFFF0000u, q, 4);
    return _mm512_shuffle_epi8(lut, _mm512_and_si512(q, _mm512_set1_epi8(15)));
}

static inline void kq_mxx_prefetch(const uint8_t *blk, int pf)
{
    if (pf > 0) {
        for (int c = 0; c < KQ_MXX_BB; c += 64) {
            _mm_prefetch((const char *)(blk + pf + c), _MM_HINT_T0);
        }
    }
}

/* The 16 rows of the KQ_MXX group at wg on n tokens of int16 x (the planes
 * xq, xl; xs; xn: for each token the kq_nvx_xsum of the high plane, then of
 * the low one, 2 cols / 16 values): out[j * ostride + r] (see the top). */
/* 32 rows (two KQ_MXX groups, one after the other) for n tokens, 4 at a time:
 * for each step of each block the weights of both groups are decoded once and
 * each broadcast of x feeds both (kq_mxx_rows2 gives each broadcast, a load,
 * one dot product: the load ports bounded it at 1.3 G of them a second on a
 * core, against 3.4 for vpdpbusd alone). 16 sums in flight (4 tokens, 2
 * groups, 2 planes). Each row has the operations of kq_mxx_rows2 (the exact
 * integer sum of each block, then p = a 2^(e - 127), f = fma(p, xs, f) over
 * the blocks in order, 0.5 f): the same bits. */
__attribute__((noinline)) static void kq_mxx_rows2x2(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                                                      const int32_t *xn, const int8_t *xl, int n, float *out,
                                                      size_t ostride)
{
    int nb = cols / 32, nh = cols / 16;
    const uint8_t *wg1 = wg + (size_t)nb * KQ_MXX_BB;
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1u));
    for (int j = 0; j < n; j += 4) {
        int nt = n - j < 4 ? n - j : 4;
        __m512 f0[4], f1[4];
        for (int u = 0; u < 4; ++u) {
            f0[u] = _mm512_setzero_ps();
            f1[u] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *b0 = wg + (size_t)b * KQ_MXX_BB, *b1 = wg1 + (size_t)b * KQ_MXX_BB;
            __m512i A0[4], L0[4], A1[4], L1[4];
            const int8_t *qx[4], *lx[4];
            for (int u = 0; u < 4; ++u) {
                int tu = u < nt ? u : 0;                /* a short tile repeats its first token */
                const int32_t *nx = xn + (size_t)(j + tu) * 2 * nh + 2 * b;
                A0[u] = A1[u] = _mm512_set1_epi32(nx[0] + nx[1]);
                L0[u] = L1[u] = _mm512_set1_epi32(nx[nh] + nx[nh + 1]);
                qx[u] = xq + (size_t)(j + tu) * cols + 32 * b;
                lx[u] = xl + (size_t)(j + tu) * cols + 32 * b;
            }
            for (int k = 0; k < 8; ++k) {
                __m512i W0 = kq_mxx_w(b0 + 32 * k, lut), W1 = kq_mxx_w(b1 + 32 * k, lut);
                for (int u = 0; u < 4; ++u) {
                    int32_t x0, y0;
                    memcpy(&x0, qx[u] + 4 * k, 4);
                    memcpy(&y0, lx[u] + 4 * k, 4);
                    __m512i bx = _mm512_set1_epi32(x0), by = _mm512_set1_epi32(y0);
                    A0[u] = _mm512_dpbusd_epi32(A0[u], W0, bx);
                    L0[u] = _mm512_dpbusd_epi32(L0[u], W0, by);
                    A1[u] = _mm512_dpbusd_epi32(A1[u], W1, bx);
                    L1[u] = _mm512_dpbusd_epi32(L1[u], W1, by);
                }
            }
            __m512 sc0 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(b0 + 256))), 23));
            __m512 sc1 = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(b1 + 256))), 23));
            for (int u = 0; u < nt; ++u) {
                __m512 xsb = _mm512_set1_ps(xs[(size_t)(j + u) * nb + b]);
                __m512i a0 = _mm512_add_epi32(_mm512_slli_epi32(A0[u], 7), L0[u]);
                __m512i a1 = _mm512_add_epi32(_mm512_slli_epi32(A1[u], 7), L1[u]);
                f0[u] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a0), sc0), xsb, f0[u]);
                f1[u] = _mm512_fmadd_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(a1), sc1), xsb, f1[u]);
            }
        }
        for (int u = 0; u < nt; ++u) {
            _mm512_storeu_ps(out + (size_t)(j + u) * ostride, _mm512_mul_ps(f0[u], _mm512_set1_ps(0.5f)));
            _mm512_storeu_ps(out + (size_t)(j + u) * ostride + 16, _mm512_mul_ps(f1[u], _mm512_set1_ps(0.5f)));
        }
    }
}

__attribute__((noinline)) static void kq_mxx_rows2_wide(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                         const int32_t *xn, const int8_t *xl, int n, float *out, size_t ostride)
{
    int nb = cols / 32, nh = cols / 16;
    const int pf = kq_mxx_pf(), tail = kq_mxx_tail();
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1u));
    for (int j = 0; j < n; j += KQ_NVX_TB) {
        int nt = n - j < KQ_NVX_TB ? n - j : KQ_NVX_TB;
        __m512 f[KQ_NVX_TB];
        for (int tt = 0; tt < KQ_NVX_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + (size_t)b * KQ_MXX_BB;
            kq_mxx_prefetch(blk, pf);
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                W[k] = kq_mxx_w(blk + 32 * k, lut);
            }
            __m512 sc = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(blk + 256))), 23));
            /* G tokens at a time, the sum of each plane of each token in 2
             * chains (the even and the odd steps): vpdpbusd waits 5 cycles
             * for its sum, and one chain of 8 a plane ran at 0.6 of them a
             * cycle (1.24 G/s on a core) for 2. The integer sums are exact in
             * any order: the same bits. */
#define KQ_MXX_TOK(G)                                                                              \
            for (; tt + (G) <= nt; tt += (G)) {                                                    \
                __m512i a0[G], a1[G], l0[G], l1[G];                                                \
                const int8_t *qx[G], *lx[G];                                                       \
                for (int u = 0; u < (G); ++u) {                                                    \
                    size_t xo = (size_t)(j + tt + u) * cols + 32 * b;                              \
                    const int32_t *nx = xn + (size_t)(j + tt + u) * 2 * nh + 2 * b;                \
                    a0[u] = _mm512_set1_epi32(nx[0] + nx[1]);                                      \
                    l0[u] = _mm512_set1_epi32(nx[nh] + nx[nh + 1]);                                \
                    a1[u] = _mm512_setzero_si512();                                                \
                    l1[u] = _mm512_setzero_si512();                                                \
                    qx[u] = xq + xo;                                                               \
                    lx[u] = xl + xo;                                                               \
                }                                                                                  \
                for (int k = 0; k < 8; k += 2) {                                                   \
                    for (int u = 0; u < (G); ++u) {                                                \
                        int32_t x0, x1, y0, y1;                                                    \
                        memcpy(&x0, qx[u] + 4 * k, 4);                                             \
                        memcpy(&x1, qx[u] + 4 * k + 4, 4);                                         \
                        memcpy(&y0, lx[u] + 4 * k, 4);                                             \
                        memcpy(&y1, lx[u] + 4 * k + 4, 4);                                         \
                        a0[u] = _mm512_dpbusd_epi32(a0[u], W[k], _mm512_set1_epi32(x0));           \
                        a1[u] = _mm512_dpbusd_epi32(a1[u], W[k + 1], _mm512_set1_epi32(x1));       \
                        l0[u] = _mm512_dpbusd_epi32(l0[u], W[k], _mm512_set1_epi32(y0));           \
                        l1[u] = _mm512_dpbusd_epi32(l1[u], W[k + 1], _mm512_set1_epi32(y1));       \
                    }                                                                              \
                }                                                                                  \
                for (int u = 0; u < (G); ++u) {                                                    \
                    __m512i a = _mm512_add_epi32(a0[u], a1[u]), l = _mm512_add_epi32(l0[u], l1[u]); \
                    a = _mm512_add_epi32(_mm512_slli_epi32(a, 7), l);                              \
                    __m512 p = _mm512_mul_ps(_mm512_cvtepi32_ps(a), sc);                           \
                    f[tt + u] = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[(size_t)(j + tt + u) * nb + b]), f[tt + u]); \
                }                                                                                  \
            }
            int tt = 0;
            KQ_MXX_TOK(4)
            if (tail) {
                /* the tokens past the fours in split chains too (one chain of
                 * 8 a plane waits 40 cycles) */
                KQ_MXX_TOK(2)
                KQ_MXX_TOK(1)
            }
#undef KQ_MXX_TOK
            for (; tt < nt; ++tt) {
                size_t xo = (size_t)(j + tt) * cols + 32 * b;
                const int32_t *nx = xn + (size_t)(j + tt) * 2 * nh + 2 * b;
                __m512i a = _mm512_set1_epi32(nx[0] + nx[1]);
                __m512i l = _mm512_set1_epi32(nx[nh] + nx[nh + 1]);
                for (int k = 0; k < 8; ++k) {
                    int32_t x0, y0;
                    memcpy(&x0, xq + xo + 4 * k, 4);
                    memcpy(&y0, xl + xo + 4 * k, 4);
                    a = _mm512_dpbusd_epi32(a, W[k], _mm512_set1_epi32(x0));
                    l = _mm512_dpbusd_epi32(l, W[k], _mm512_set1_epi32(y0));
                }
                a = _mm512_add_epi32(_mm512_slli_epi32(a, 7), l);
                __m512 p = _mm512_mul_ps(_mm512_cvtepi32_ps(a), sc);
                f[tt] = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b]), f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], _mm512_set1_ps(0.5f)));
        }
    }
}
/* kq_mxx_rows2 on x of one plane (xl NULL: the 8-bit rows of flag 256 of
 * GP_KQ_MOE_PART, whose low plane is 0): the sums of the high plane alone, a
 * = 128 sum(2 e2m1 hi), the integer of kq_mxx_rows2 on those rows (its low
 * sums are 0), so the same bits with half the dot products. 8 tokens at a
 * time, each in 2 chains (the even and the odd steps). */
__attribute__((noinline)) static void kq_mxx_rows1(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                                                   const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32, nh = cols / 16;
    const int pf = kq_mxx_pf();
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1u));
    for (int j = 0; j < n; j += KQ_NVX_TB) {
        int nt = n - j < KQ_NVX_TB ? n - j : KQ_NVX_TB;
        __m512 f[KQ_NVX_TB];
        for (int tt = 0; tt < KQ_NVX_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + (size_t)b * KQ_MXX_BB;
            kq_mxx_prefetch(blk, pf);
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                W[k] = kq_mxx_w(blk + 32 * k, lut);
            }
            __m512 sc = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(blk + 256))), 23));
#define KQ_MXX_TOK1(G)                                                                             \
            for (; tt + (G) <= nt; tt += (G)) {                                                    \
                __m512i a0[G], a1[G];                                                              \
                const int8_t *qx[G];                                                               \
                for (int u = 0; u < (G); ++u) {                                                    \
                    const int32_t *nx = xn + (size_t)(j + tt + u) * 2 * nh + 2 * b;                \
                    a0[u] = _mm512_set1_epi32(nx[0] + nx[1]);                                      \
                    a1[u] = _mm512_setzero_si512();                                                \
                    qx[u] = xq + (size_t)(j + tt + u) * cols + 32 * b;                             \
                }                                                                                  \
                for (int k = 0; k < 8; k += 2) {                                                   \
                    for (int u = 0; u < (G); ++u) {                                                \
                        int32_t x0, x1;                                                            \
                        memcpy(&x0, qx[u] + 4 * k, 4);                                             \
                        memcpy(&x1, qx[u] + 4 * k + 4, 4);                                         \
                        a0[u] = _mm512_dpbusd_epi32(a0[u], W[k], _mm512_set1_epi32(x0));           \
                        a1[u] = _mm512_dpbusd_epi32(a1[u], W[k + 1], _mm512_set1_epi32(x1));       \
                    }                                                                              \
                }                                                                                  \
                for (int u = 0; u < (G); ++u) {                                                    \
                    __m512i a = _mm512_slli_epi32(_mm512_add_epi32(a0[u], a1[u]), 7);              \
                    __m512 p = _mm512_mul_ps(_mm512_cvtepi32_ps(a), sc);                           \
                    f[tt + u] = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[(size_t)(j + tt + u) * nb + b]), f[tt + u]); \
                }                                                                                  \
            }
            int tt = 0;
            KQ_MXX_TOK1(8)
            KQ_MXX_TOK1(4)
            KQ_MXX_TOK1(2)
            KQ_MXX_TOK1(1)
#undef KQ_MXX_TOK1
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], _mm512_set1_ps(0.5f)));
        }
    }
}

static void kq_mxx_rows2(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                         const int32_t *xn, const int8_t *xl, int n, float *out, size_t ostride)
{
    if (xl == NULL) {
        kq_mxx_rows1(wg, cols, xq, xs, xn, n, out, ostride);
        return;
    }
    if (n >= 8) {
        /* many tokens (a group of a prompt): the split chains, 10% faster at 16
         * tokens; kept apart, as they made this function 35% slower for one
         * token (a step) */
        kq_mxx_rows2_wide(wg, cols, xq, xs, xn, xl, n, out, ostride);
        return;
    }
    int nb = cols / 32, nh = cols / 16;
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1u));
    for (int j = 0; j < n; j += KQ_NVX_TB) {
        int nt = n - j < KQ_NVX_TB ? n - j : KQ_NVX_TB;
        __m512 f[KQ_NVX_TB];
        for (int tt = 0; tt < KQ_NVX_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + (size_t)b * KQ_MXX_BB;
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                W[k] = kq_mxx_w(blk + 32 * k, lut);
            }
            __m512 sc = _mm512_castsi512_ps(_mm512_slli_epi32(
                _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(blk + 256))), 23));
            for (int tt = 0; tt < nt; ++tt) {
                size_t xo = (size_t)(j + tt) * cols + 32 * b;
                const int32_t *nx = xn + (size_t)(j + tt) * 2 * nh + 2 * b;
                __m512i a = _mm512_set1_epi32(nx[0] + nx[1]);
                __m512i l = _mm512_set1_epi32(nx[nh] + nx[nh + 1]);
                for (int k = 0; k < 8; ++k) {
                    int32_t x0, y0;
                    memcpy(&x0, xq + xo + 4 * k, 4);
                    memcpy(&y0, xl + xo + 4 * k, 4);
                    a = _mm512_dpbusd_epi32(a, W[k], _mm512_set1_epi32(x0));
                    l = _mm512_dpbusd_epi32(l, W[k], _mm512_set1_epi32(y0));
                }
                a = _mm512_add_epi32(_mm512_slli_epi32(a, 7), l);
                __m512 p = _mm512_mul_ps(_mm512_cvtepi32_ps(a), sc);
                f[tt] = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b]), f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], _mm512_set1_ps(0.5f)));
        }
    }
}
#else
/* Without VNNI: the same operations in C (the same bits: exact integer sums,
 * an exact product with the scale, one fma a block). */
static void kq_mxx_rows2(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                         const int32_t *xn, const int8_t *xl, int n, float *out, size_t ostride)
{
    int nb = cols / 32;
    (void)xn;
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + (size_t)b * KQ_MXX_BB;
                int32_t a = 0;
                for (int v = 0; v < 32; ++v) {
                    size_t xo = (size_t)j * cols + 32 * b + v;
                    a += kq_e2m1x2[kq_mxx_code(blk, r, v)] * (128 * xq[xo] + (xl ? xl[xo] : 0));
                }
                float p = (float)a * kq_e8m0f(blk[256 + r]);
                f = fmaf(p, xs[(size_t)j * nb + b], f);
            }
            out[(size_t)j * ostride + r] = 0.5f * f;
        }
    }
}
#endif

/* 32 rows (two groups): kq_mxx_rows2x2 for 4 tokens or more (a group of a
 * prompt; with VNNI), else kq_mxx_rows2 for each group (a step, or x of one
 * plane). */
static void kq_mxx_rows32(const uint8_t *wg, int cols, const int8_t *xq, const float *xs, const int32_t *xn,
                          const int8_t *xl, int n, float *out, size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (n >= 4 && xl != NULL) {
        kq_mxx_rows2x2(wg, cols, xq, xs, xn, xl, n, out, ostride);
        return;
    }
#endif
    kq_mxx_rows2(wg, cols, xq, xs, xn, xl, n, out, ostride);
    kq_mxx_rows2(wg + (size_t)(cols / 32) * KQ_MXX_BB, cols, xq, xs, xn, xl, n, out + 16, ostride);
}

/* out (t x rows) = x W^T for a KQ_MXX matrix, near: x as int16 (a check and
 * the CPU path of a dense use; the experts take kq_mxx_rows2 in the MoE). */
void rc_kq_mxx_linear16(const uint8_t *w, int rows, int cols, const float *x, int t, float *out)
{
    int nb = cols / 32, nh = cols / 16;
    int8_t *hi = (int8_t *)malloc((size_t)t * cols), *lo = (int8_t *)malloc((size_t)t * cols);
    float *xs = (float *)malloc((size_t)t * nb * 4), *xm = (float *)malloc((size_t)t * nh * 4);
    int32_t *xn = (int32_t *)malloc((size_t)t * 2 * nh * 4);
    for (int j = 0; j < t; ++j) {
        for (int g = 0; g < nb; ++g) {
            kq_quant_part16(x + (size_t)j * cols, g, hi + (size_t)j * cols, lo + (size_t)j * cols,
                            xs + (size_t)j * nb, xm + (size_t)j * nh);
        }
        kq_nvx_xsum(hi + (size_t)j * cols, 1, cols, xn + (size_t)j * 2 * nh);
        kq_nvx_xsum(lo + (size_t)j * cols, 1, cols, xn + (size_t)j * 2 * nh + nh);
    }
    float *tmp = (float *)malloc((size_t)t * 16 * 4 * (rows / 16));
    /* RECLANK_MXX32=1 (a measure): the rows two groups at a time (kq_mxx_rows32,
     * as the parts) */
    const char *v32 = getenv("RECLANK_MXX32");
    if (v32 && v32[0] == '1' && rows % 32 == 0) {
        #pragma omp parallel for schedule(static)
        for (int g = 0; g < rows / 32; ++g) {
            float *o = tmp + (size_t)g * t * 32;
            kq_mxx_rows32(w + (size_t)2 * g * nb * KQ_MXX_BB, cols, hi, xs, xn, lo, t, o, 32);
            for (int j = 0; j < t; ++j) {
                memcpy(out + (size_t)j * rows + 32 * g, o + (size_t)j * 32, 32 * 4);
            }
        }
    } else {
    #pragma omp parallel for schedule(static)
    for (int g = 0; g < rows / 16; ++g) {
        float *o = tmp + (size_t)g * t * 16;
        kq_mxx_rows2(w + (size_t)g * nb * KQ_MXX_BB, cols, hi, xs, xn, lo, t, o, 16);
        for (int j = 0; j < t; ++j) {
            memcpy(out + (size_t)j * rows + 16 * g, o + (size_t)j * 16, 16 * 4);
        }
    }
    }
    free(tmp);
    free(hi);
    free(lo);
    free(xs);
    free(xm);
    free(xn);
}

/* out (t x rows) = x W^T for a KQ_MXX matrix, exact (see the top). */
void rc_kq_mxx_linear_exact(const uint8_t *w, int rows, int cols, const float *x, int t, float *out)
{
    int nb = cols / 32;
    float *q = (float *)malloc((size_t)t * cols * 4), *s = (float *)malloc((size_t)t * nb * 4);
    for (int j = 0; j < t; ++j) {
        for (int b = 0; b < nb; ++b) {
            kq_act_e4m3_32(x + (size_t)j * cols + 32 * b, q + (size_t)j * cols + 32 * b, s + (size_t)j * nb + b);
        }
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wg = w + (size_t)(i / 16) * nb * KQ_MXX_BB;
        for (int j = 0; j < t; ++j) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + (size_t)b * KQ_MXX_BB;
                const float *qb = q + (size_t)j * cols + 32 * b;
                float sum = 0.f;
                for (int v = 0; v < 32; ++v) {
                    sum = fmaf(kq_e2m1f[kq_mxx_code(blk, i % 16, v)], qb[v], sum);
                }
                f += sum * s[(size_t)j * nb + b] * kq_e8m0f(blk[256 + i % 16]);
            }
            out[(size_t)j * rows + i] = f;
        }
    }
    free(q);
    free(s);
}

/* out (t x rows) = x W^T for a KQ_FP8B32 matrix (w: the codes, sc: the
 * scales), near (exact 0) or exact (1). */
void rc_kq_fp8b32_linear(const uint8_t *w, const uint8_t *sc, int rows, int cols, const float *x, int t,
                         int exact, float *out)
{
    int nb = cols / 32;
    float *q = (float *)malloc((size_t)t * cols * 4), *s = (float *)malloc((size_t)t * nb * 4);
    for (int j = 0; j < t; ++j) {
        for (int b = 0; b < nb; ++b) {
            if (exact) {
                kq_act_e4m3_32(x + (size_t)j * cols + 32 * b, q + (size_t)j * cols + 32 * b,
                               s + (size_t)j * nb + b);
            } else {
                memcpy(q + (size_t)j * cols + 32 * b, x + (size_t)j * cols + 32 * b, 32 * 4);
                s[(size_t)j * nb + b] = 1.f;
            }
        }
    }
    float tab[256];
    for (int c = 0; c < 256; ++c) {
        tab[c] = kq_e4m3f((uint8_t)c);
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wr = w + (size_t)i * cols;
        for (int j = 0; j < t; ++j) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const float *qb = q + (size_t)j * cols + 32 * b;
                float sum = 0.f;
                for (int v = 0; v < 32; ++v) {
                    sum = fmaf(tab[wr[32 * b + v]], qb[v], sum);
                }
                f += sum * s[(size_t)j * nb + b] * kq_e8m0f(sc[(size_t)(i / 32) * nb + b]);
            }
            out[(size_t)j * rows + i] = f;
        }
    }
    free(q);
    free(s);
}

/* The reference's act_quant of rows of x (in place: the E4M3 values times
 * their scale), for the exact mode and the checks. */
void rc_dsv41_act_quant(float *x, int t, int cols)
{
    for (int j = 0; j < t; ++j) {
        for (int b = 0; b < cols / 32; ++b) {
            float q[32], s;
            kq_act_e4m3_32(x + (size_t)j * cols + 32 * b, q, &s);
            for (int v = 0; v < 32; ++v) {
                x[(size_t)j * cols + 32 * b + v] = q[v] * s;
            }
        }
    }
}
