/* ---- the decode attention over the int16 cache, split by keys ----
 *
 * As the attention of a decode step on the GPU (k_attn_fd): an item is a
 * query token, a key head, and a chunk of AS_LEN(n) keys. It does all the
 * query heads of that key head, so it reads each key and value row once.
 * It converts a row to float32 times the scale of each group of 32 values
 * once (as_row), and each query head then takes one dot product with one
 * sum across the lanes (AVX2 summed each group of 32 before). The item
 * writes, for each of its query heads, the sum of exp(score - m) times the
 * value rows, m, and l (the sum of exp(score - m)) to the scratch. A second
 * pass joins the chunks of each query head, in the order of the chunks.
 *
 * The chunk length depends only on n, so a query of a group gives the bits
 * of the same query in a step. Many items keep all the threads busy: the
 * 16 query heads of the 26B over 6 threads left two threads idle. */
#define AS_MAXC 32
#define AS_MINLEN 64
#define AS_LEN(n) ((n) <= AS_MINLEN * AS_MAXC ? AS_MINLEN : ((n) + AS_MAXC - 1) / AS_MAXC)
/* the query values of the heads of an item: 8192 takes the 16 query heads of
 * 512 values of a global layer of the 12B (one key head) */
#define AS_MAXQ 8192

/* The scratch of the split attention, one set for each team: the team of
 * one program (slot 0), or the team of each part of rc_run_parts (slot
 * part + 1), because the parts run it at the same time. */
#define AS_SLOTS 9
typedef struct {
    float *parts;
    size_t parts_n;
    float *isc;                       /* the scores of the items, by thread */
    size_t isc_n;
    int16_t *qq;                      /* the int16 queries (AVX2) */
    float *sq;
    size_t q_n, sq_n;
} as_scratch;
static as_scratch as_scr[AS_SLOTS];

static as_scratch *as_team(void)
{
    int slot = omp_get_level() >= 2 ? omp_get_ancestor_thread_num(1) + 1 : 0;
    return &as_scr[slot >= 0 && slot < AS_SLOTS ? slot : 0];
}

static int as_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_ATTN_SPLIT");
        on = !(v && v[0] == '0');
    }
    return on;
}

#if GEMMA_X86
/* 8 values as floats; 16 values as int16 lanes. */
static inline __attribute__((always_inline)) __m256 as_load8f(const void *p, const int I8)
{
    return _mm256_cvtepi32_ps(I8 ? _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i *)p))
                                 : _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i *)p)));
}

static inline __attribute__((always_inline)) __m256i as_load16(const void *p, const int I8)
{
    return I8 ? _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)p))
              : _mm256_loadu_si256((const __m256i *)p);
}
#endif

/* The cache values are int16 (I8 0), int8 (I8 1, RECLANK_KV_INT8), or TQ6
 * (I8 2, 3 bytes for 4 values, the rotated form; tq6_deq32): the helpers
 * below take the element size as a constant, and the bodies
 * (gemma_attn_split_i16_body, gemma_attn_split_i8_body, ...) inline them. */
#define AS_ATE(p, idx, i8) ((const void *)((const char *)(p) + \
    ((i8) == 2 ? (size_t)(idx) * 3 / 4 : (size_t)(idx) * ((i8) ? 1 : 2))))
#define AS_AT(p, idx) AS_ATE(p, idx, I8)
#define AS_EL(p, i) (I8 ? (float)((const int8_t *)(p))[i] : (float)((const int16_t *)(p))[i])

/* out[i] = (float)row[i] * scale of the group of i, for hd values. */
/* FP4-Local's FP8 cache (docs/FP4LOCAL_PLAN.md 2.9): the bytes of the int8 cache are E4M3 codes, the
 * scales 1 (rc_set_kv_e4m3; GP_KV_WRITE8 writes them so). */
static int rc_kv_e4m3 = 0;
static float rc_kv_e4m3_tab[256];

void rc_set_kv_e4m3(int on)
{
    for (int b = 0; b < 256; ++b) {
        int e = (b >> 3) & 15, m = b & 7;
        float v = e == 0 ? (float)m / 512.f : ldexpf(1.f + (float)m / 8.f, e - 7);
        rc_kv_e4m3_tab[b] = (b & 0x80) ? -v : v;
    }
    rc_kv_e4m3 = on;
}

static inline __attribute__((always_inline)) void as_row(const void *row, const float *sc, int hd,
                                                         float *out, const int I8)
{
    if (I8 == 1 && rc_kv_e4m3) {
        for (int i = 0; i < hd; ++i) {
            out[i] = rc_kv_e4m3_tab[((const uint8_t *)row)[i]] * sc[i / 32];
        }
        return;
    }
    if (I8 == 2) {
        for (int gg = 0; gg < hd / 32; ++gg) {
            tq6_deq32((const uint8_t *)row + (size_t)gg * 24, sc[gg], out + (size_t)gg * 32);
        }
        return;
    }
    for (int gg = 0; gg < hd / 32; ++gg) {
        const void *rp = AS_AT(row, (size_t)gg * 32);
        float *op = out + (size_t)gg * 32;
#if GEMMA_X86 && defined(__AVX512F__)
        __m512 s5 = _mm512_set1_ps(sc[gg]);
        for (int i = 0; i < 32; i += 16) {
            __m512 v = _mm512_cvtepi32_ps(I8 ?
                _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)AS_AT(rp, i))) :
                _mm512_cvtepi16_epi32(_mm256_loadu_si256((const __m256i *)AS_AT(rp, i))));
            _mm512_storeu_ps(op + i, _mm512_mul_ps(v, s5));
        }
#elif GEMMA_X86
        __m256 s8 = _mm256_set1_ps(sc[gg]);
        for (int i = 0; i < 32; i += 8) {
            __m256 v = as_load8f(AS_AT(rp, i), I8);
            _mm256_storeu_ps(op + i, _mm256_mul_ps(v, s8));
        }
#else
        for (int i = 0; i < 32; ++i) {
            op[i] = AS_EL(rp, i) * sc[gg];
        }
#endif
    }
}

static inline float as_dot(const float *a, const float *b, int hd)
{
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    for (int i = 0; i < hd; i += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i), _mm512_loadu_ps(b + i), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + i + 16), _mm512_loadu_ps(b + i + 16), acc1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
#elif GEMMA_X86
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    for (int i = 0; i < hd; i += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), acc1);
    }
    return hsum256_ps(_mm256_add_ps(acc0, acc1));
#else
    float acc = 0.0f;
    for (int i = 0; i < hd; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
#endif
}

/* o += p b, hd values. */
static inline void as_axpy(float *o, float p, const float *b, int hd)
{
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 p5 = _mm512_set1_ps(p);
    for (int i = 0; i < hd; i += 16) {
        _mm512_storeu_ps(o + i, _mm512_fmadd_ps(p5, _mm512_loadu_ps(b + i), _mm512_loadu_ps(o + i)));
    }
#elif GEMMA_X86
    __m256 p8 = _mm256_set1_ps(p);
    for (int i = 0; i < hd; i += 8) {
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(p8, _mm256_loadu_ps(b + i), _mm256_loadu_ps(o + i)));
    }
#else
    for (int i = 0; i < hd; ++i) {
        o[i] += p * b[i];
    }
#endif
}

#if GEMMA_X86 && !defined(__AVX512F__)
/* The AVX2 forms of the two passes of an item (nh query heads, cn keys).
 *
 * The scores: the queries are int16 with a scale for each group of 32
 * values (qq, sq: as_quant_q), and vpmaddwd multiplies them with the int16
 * keys. Each vpmaddwd result (8 sums of 2 products) goes to float on its
 * own, so no int32 sum can overflow. A group adds into a float
 * accumulator with the scale of the key group times that of the query
 * group, and one sum across the lanes ends a key. The keys of the item are
 * loaded once for all its heads.
 *
 * The values: the item keeps the output of a block of 8 U values of each
 * head in registers over all its keys, so it loads and stores the output
 * once, not once for each key. The weight of a key times the scale of the
 * value group is one broadcast. */
/* The rows of the cache are kv_heads * head_dim values apart (4 KB for
 * the layers of the 26B with a window), so each row of an item starts a
 * new page, where the hardware prefetch stops. The passes fetch the row
 * AS_PF keys ahead. */
#define AS_PF 10
static inline __attribute__((always_inline)) void as_prefetch_row(const void *row, int hd,
                                                                  const int I8)
{
    for (int i = 0; i < hd * (I8 ? 1 : 2); i += 64) {
        _mm_prefetch((const char *)row + i, _MM_HINT_T0);
    }
}

/* exp(x) of 8 floats: the polynomial of Cephes expf (about 1 ulp), with the
 * power of two in the exponent bits. libm expf took most of the time of an
 * item outside its two passes. */
static inline __m256 as_exp256(__m256 x)
{
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647949f));
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647949f));
    __m256 fx = _mm256_floor_ps(_mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f),
                                                _mm256_set1_ps(0.5f)));
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x);
    __m256 y = _mm256_set1_ps(1.9875691500e-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073e-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894e-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459e-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201e-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));
    __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvttps_epi32(fx), _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

/* sc[j] = exp(sc[j] - m), or 0 below exp(-64) (as the scalar loop); return
 * the sum. */
static inline float as_softmax_avx2(float *sc, int cn, float m)
{
    __m256 m8 = _mm256_set1_ps(m), lim = _mm256_set1_ps(-64.0f), acc = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= cn; j += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(sc + j), m8);
        __m256 ev = _mm256_and_ps(as_exp256(d), _mm256_cmp_ps(d, lim, _CMP_GE_OQ));
        _mm256_storeu_ps(sc + j, ev);
        acc = _mm256_add_ps(acc, ev);
    }
    float l = hsum256_ps(acc);
    for (; j < cn; ++j) {
        float d = sc[j] - m;
        sc[j] = d < -64.0f ? 0.0f : expf(d);
        l += sc[j];
    }
    return l;
}

/* The query of one head as int16 with one scale (14 bits: |q| <= 16383),
 * so the two vpmaddwd results of a group of 32 add as int32 with no
 * overflow (at most 2 * 2 * 16383 * 32767 < 2^31). */
static inline void as_quant_q(const float *q, int hd, int16_t *qq, float *sq)
{
    float m = 0.0f;
    for (int i = 0; i < hd; ++i) {
        float a = fabsf(q[i]);
        m = a > m ? a : m;
    }
    float inv = m > 0.0f ? 16383.0f / m : 0.0f;
    *sq = m > 0.0f ? m / 16383.0f : 0.0f;
    for (int i = 0; i < hd; i += 8) {
        __m256i v = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_loadu_ps(q + i), _mm256_set1_ps(inv)));
        __m128i p16 = _mm_packs_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
        _mm_storeu_si128((__m128i *)(qq + i), p16);
    }
}

static inline __attribute__((always_inline)) void as_scores_avx2(const int16_t *qq, const float *sq, int nh, const void *kp,
                                  const float *ksp, int hd, float *out, size_t ostride,
                                  const int I8)
{
    __m256 acc[8];
    for (int h = 0; h < nh; ++h) {
        acc[h] = _mm256_setzero_ps();
    }
    for (int gg = 0; gg < hd / 32; ++gg) {
        __m256i k0 = as_load16(AS_AT(kp, (size_t)gg * 32), I8);
        __m256i k1 = as_load16(AS_AT(kp, (size_t)gg * 32 + 16), I8);
        __m256 kscale = _mm256_set1_ps(ksp[gg]);
        for (int h = 0; h < nh; ++h) {
            const int16_t *qh = qq + (size_t)h * hd + (size_t)gg * 32;
            __m256i d = _mm256_add_epi32(_mm256_madd_epi16(k0, _mm256_loadu_si256((const __m256i *)qh)),
                                         _mm256_madd_epi16(k1, _mm256_loadu_si256((const __m256i *)(qh + 16))));
            acc[h] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(d), kscale, acc[h]);
        }
    }
    for (int h = 0; h < nh; ++h) {
        out[(size_t)h * ostride] = hsum256_ps(acc[h]) * sq[h];
    }
}

/* as_scores_avx2 for a TQ6 key row: the row goes to float32 (tq6_deq32),
 * then nh float32 queries (q, rotated) read each 8 values once. */
static inline __attribute__((always_inline)) void as_scores_tq_avx2(const float *q, int nh,
                                                                   const uint8_t *kp,
                                                                   const float *ksp, int hd,
                                                                   float *out, size_t ostride)
{
    float row[1024] __attribute__((aligned(32)));
    for (int gg = 0; gg < hd / 32; ++gg) {
        tq6_deq32(kp + (size_t)gg * 24, ksp[gg], row + (size_t)gg * 32);
    }
    __m256 acc[8];
    for (int h = 0; h < nh; ++h) {
        acc[h] = _mm256_setzero_ps();
    }
    for (int i = 0; i < hd; i += 8) {
        __m256 k = _mm256_load_ps(row + i);
        for (int h = 0; h < nh; ++h) {
            acc[h] = _mm256_fmadd_ps(_mm256_loadu_ps(q + (size_t)h * hd + i), k, acc[h]);
        }
    }
    for (int h = 0; h < nh; ++h) {
        out[(size_t)h * ostride] = hsum256_ps(acc[h]);
    }
}

/* o (nh rows of hd, row stride ostride) = the sum over the cn keys of
 * p[h][j] (row stride pstride) times the value rows. */
#endif

#if GEMMA_X86 && !defined(__AVX512F__)
#define AS_VBLOCK(U)                                                                         \
    for (int d0 = 0; d0 < hd; d0 += 8 * (U)) {                                               \
        __m256 acc[8][U];                                                                    \
        for (int h = 0; h < nh; ++h)                                                         \
            for (int u = 0; u < (U); ++u)                                                    \
                acc[h][u] = _mm256_setzero_ps();                                             \
        int gg = d0 / 32;                                                                    \
        for (int j = 0; j < cn; ++j) {                                                       \
            if (d0 == 0 && j + AS_PF < cn)                                                   \
                as_prefetch_row(AS_AT(vq, (base + j + AS_PF) * kv_stride + (size_t)kv * hd), \
                                hd, I8);                                                     \
            if (!live[j]) continue;                                                          \
            size_t r = base + (size_t)j;                                                     \
            const void *vp = AS_AT(vq, r * kv_stride + (size_t)kv * hd + d0);                \
            float vsc = vs[r * ks_stride + (size_t)kv * (hd / 32) + gg];                     \
            __m256 v[U];                                                                     \
            for (int u = 0; u < (U); ++u)                                                    \
                v[u] = as_load8f(AS_AT(vp, 8 * u), I8);                                      \
            for (int h = 0; h < nh; ++h) {                                                   \
                float pw = p[(size_t)h * pstride + j];                                       \
                if (pw == 0.0f) continue;                                                    \
                __m256 b = _mm256_set1_ps(pw * vsc);                                         \
                for (int u = 0; u < (U); ++u)                                                \
                    acc[h][u] = _mm256_fmadd_ps(b, v[u], acc[h][u]);                         \
            }                                                                                \
        }                                                                                    \
        for (int h = 0; h < nh; ++h)                                                         \
            for (int u = 0; u < (U); ++u)                                                    \
                _mm256_storeu_ps(o + (size_t)h * ostride + d0 + 8 * u, acc[h][u]);           \
    }

static inline __attribute__((always_inline)) void
as_values_avx2_n(const float *p, size_t pstride, const unsigned char *live, int nh,
                 const void *vq, const float *vs, size_t kv_stride, size_t ks_stride, int kv,
                 int hd, size_t base, int cn, float *o, size_t ostride, const int I8)
{
    if (nh <= 2) {
        AS_VBLOCK(4)
    } else if (nh <= 4) {
        AS_VBLOCK(2)
    } else {
        AS_VBLOCK(1)
    }
}

/* The value pass by groups of 32 values: the rows of up to AS_VSUB keys go
 * to float32 times the scale of the group once (vf, in L1), and then each
 * pair of heads reads them: one broadcast of the weight and 4 fma for each
 * key. as_values_avx2 converted a row once for each pair of
 * heads (4 times for the 8 heads of a global layer of the 26B). */
#define AS_VSUB 128
static inline __attribute__((always_inline)) void
as_values_avx2_g(const float *p, size_t pstride, const unsigned char *live, int nh,
                 const void *vq, const float *vs, size_t kv_stride, size_t ks_stride, int kv,
                 int hd, size_t base, int cn, float *o, size_t ostride, const int I8)
{
    float vf[AS_VSUB * 32] __attribute__((aligned(32)));
    int idx[AS_VSUB];
    int nb = hd / 32;
    for (int gg = 0; gg < nb; ++gg) {
        for (int j0 = 0; j0 < cn; j0 += AS_VSUB) {
            int jn = cn - j0 < AS_VSUB ? cn - j0 : AS_VSUB;
            int m = 0;                                     /* the live keys */
            for (int j = 0; j < jn; ++j) {
                if (!live[j0 + j]) {
                    continue;
                }
                size_t r = base + (size_t)(j0 + j);
                const void *vp = AS_AT(vq, r * kv_stride + (size_t)kv * hd + (size_t)gg * 32);
                if (I8 == 2) {
                    tq6_deq32((const uint8_t *)vp, vs[r * ks_stride + (size_t)kv * nb + gg],
                              vf + (size_t)m * 32);
                } else {
                    __m256 vsc = _mm256_set1_ps(vs[r * ks_stride + (size_t)kv * nb + gg]);
                    for (int u = 0; u < 4; ++u) {
                        _mm256_store_ps(vf + (size_t)m * 32 + 8 * u,
                                        _mm256_mul_ps(as_load8f(AS_AT(vp, 8 * u), I8), vsc));
                    }
                }
                idx[m] = j0 + j;
                ++m;
            }
            for (int h0 = 0; h0 < nh; h0 += 2) {
                int two = nh - h0 >= 2;
                float *o0 = o + (size_t)h0 * ostride + (size_t)gg * 32;
                float *o1 = two ? o0 + ostride : o0;
                __m256 a0[4], a1[4];
                for (int u = 0; u < 4; ++u) {
                    a0[u] = j0 ? _mm256_loadu_ps(o0 + 8 * u) : _mm256_setzero_ps();
                    a1[u] = (j0 && two) ? _mm256_loadu_ps(o1 + 8 * u) : _mm256_setzero_ps();
                }
                const float *p0 = p + (size_t)h0 * pstride;
                const float *p1 = p + (size_t)(two ? h0 + 1 : h0) * pstride;
                for (int k = 0; k < m; ++k) {
                    const float *v = vf + (size_t)k * 32;
                    __m256 b0 = _mm256_broadcast_ss(p0 + idx[k]), b1 = _mm256_broadcast_ss(p1 + idx[k]);
                    for (int u = 0; u < 4; ++u) {
                        __m256 vv = _mm256_load_ps(v + 8 * u);
                        a0[u] = _mm256_fmadd_ps(b0, vv, a0[u]);
                        a1[u] = _mm256_fmadd_ps(b1, vv, a1[u]);
                    }
                }
                for (int u = 0; u < 4; ++u) {
                    _mm256_storeu_ps(o0 + 8 * u, a0[u]);
                    if (two) {
                        _mm256_storeu_ps(o1 + 8 * u, a1[u]);
                    }
                }
            }
        }
    }
}

static inline __attribute__((always_inline)) void
as_values_avx2(const float *p, size_t pstride, const unsigned char *live, int nh,
               const void *vq, const float *vs, size_t kv_stride, size_t ks_stride, int kv,
               int hd, size_t base, int cn, float *o, size_t ostride, const int I8)
{
    if (nh > 2 && nh <= 8) {
        as_values_avx2_g(p, pstride, live, nh, vq, vs, kv_stride, ks_stride, kv, hd, base, cn,
                         o, ostride, I8);
        return;
    }
    /* Two heads at a time: 8 accumulators of 4 vectors each fit the 16
     * registers, and the 8 converted values serve two heads. Eight heads in
     * one pass (1 vector each) took 268 us for a global layer of the 26B. */
    for (int h0 = 0; h0 < nh; h0 += 2) {
        if (nh - h0 >= 2) {
            as_values_avx2_n(p + (size_t)h0 * pstride, pstride, live, 2, vq, vs, kv_stride,
                             ks_stride, kv, hd, base, cn, o + (size_t)h0 * ostride, ostride, I8);
        } else {
            as_values_avx2_n(p + (size_t)h0 * pstride, pstride, live, 1, vq, vs, kv_stride,
                             ks_stride, kv, hd, base, cn, o + (size_t)h0 * ostride, ostride, I8);
        }
    }
}
#undef AS_VBLOCK
#endif

/* The attention of tokens queries over the int16 cache (call it inside a
 * parallel region). Query t reads the rows lo[t] to lo[t] + n[t] - 1 (lo
 * null: rows 0 to n[t] - 1). q and out are (tokens, q_heads, head_dim).
 * The kernel keeps the scores in its own scratch (a block for each thread),
 * so it does not read the scores operand of the record: the callers size it
 * in other ways (the prompt of Qwen3.6 gives a block of nmax for each
 * thread). Return 0 when the shape does not fit (the caller then uses
 * gemma_attn_decode_i16_mt_body). */
static inline __attribute__((always_inline)) int
as_split_body(const float *q, const void *kq, const float *ks, const void *vq, const float *vs,
              float *scores, float *out, int q_heads, int kv_heads, int head_dim, const int *lo,
              const int *n, int nmax, int tokens, const int KI8,
              const int VI8, int row_heads)
{
    as_scratch *S = as_team();
    int n_rep = q_heads / kv_heads;
    /* TQ6 (KI8 = VI8 = 2): the generic passes over float rows (as_row) */
    const int TQ = KI8 == 2;
    /* the int8 cache has no other kernel, so RECLANK_ATTN_SPLIT=0 keeps
     * this one for it */
    if ((!KI8 && !VI8 && !as_on()) || head_dim % 32 != 0 || head_dim > 1024 ||
        n_rep * head_dim > AS_MAXQ) {
        return 0;
    }
    /* a row of the cache holds row_heads heads; kq and ks point at the first
     * head of this call */
    size_t kv_stride = (size_t)row_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)row_heads * (size_t)(head_dim / 32);
    int g = head_dim / 32;
    size_t pw = (size_t)head_dim + 2;
    size_t need = (size_t)tokens * (size_t)q_heads * AS_MAXC * pw;
    /* the scores of an item: n_rep rows of the longest chunk, a block for
     * each thread (the scores operand of the callers has other sizes) */
    int ilen = 0;
    for (int j = 0; j < tokens; ++j) {
        int l2 = AS_LEN(n[j]);
        ilen = l2 > ilen ? l2 : ilen;
    }
    size_t ineed = (size_t)omp_get_num_threads() * (size_t)n_rep * (size_t)ilen;
    #pragma omp single
    {
        if (S->isc_n < ineed) {
            free(S->isc);
            S->isc = (float *)malloc(ineed * sizeof(float));
            S->isc_n = S->isc ? ineed : 0;
        }
        if (S->parts_n < need) {
            free(S->parts);
            S->parts = (float *)malloc(need * sizeof(float));
            S->parts_n = S->parts ? need : 0;
        }
#if GEMMA_X86 && !defined(__AVX512F__)
        /* the values and the scales grow on their own: a group of 2 tokens
         * with heads of 512 values needs the values of 4 tokens with heads
         * of 256, but half the scales */
        size_t nq = (size_t)tokens * q_heads;
        if (S->q_n < nq * head_dim) {
            free(S->qq);
            S->qq = (int16_t *)malloc(nq * head_dim * sizeof(int16_t));
            S->q_n = S->qq ? nq * head_dim : 0;
        }
        if (S->sq_n < nq) {
            free(S->sq);
            S->sq = (float *)malloc(nq * sizeof(float));
            S->sq_n = S->sq ? nq : 0;
            if (S->sq == NULL) {
                S->q_n = 0;
            }
        }
#endif
    }
    if (S->parts == NULL || S->isc == NULL) {
        return 0;
    }
    float *isc = S->isc + (size_t)omp_get_thread_num() * (size_t)n_rep * (size_t)ilen;
    (void)scores;
    (void)nmax;
#if GEMMA_X86 && !defined(__AVX512F__)
    if (S->q_n == 0) {
        return 0;
    }
    /* the int16 queries, once for each token and head (the items of the
     * chunks of a head share them) */
    if (!TQ) {
        #pragma omp for schedule(static)
        for (int u = 0; u < tokens * q_heads; ++u) {
            as_quant_q(q + (size_t)u * head_dim, head_dim, S->qq + (size_t)u * head_dim,
                       S->sq + u);
        }
    }
#endif
    /* the items: (token, key head, chunk); token t has cdiv(n[t], AS_LEN) */
    int per = kv_heads * AS_MAXC;
    int total = tokens * per;
    #pragma omp for schedule(dynamic, 1)
    for (int u = 0; u < total; ++u) {
        int t = u / per, kv = (u % per) / AS_MAXC, c = u % AS_MAXC;
        int nt = n[t], len = AS_LEN(nt);
        int r0 = c * len;
        if (r0 >= nt) {
            continue;
        }
        int cn = nt - r0 < len ? nt - r0 : len;
        size_t base = (size_t)(lo ? lo[t] : 0) + (size_t)r0;
        float row[1024];
        int h0 = kv * n_rep;
        const float *qt = q + ((size_t)t * q_heads + h0) * head_dim;
#if GEMMA_X86 && !defined(__AVX512F__)
        const int16_t *qq = S->qq + ((size_t)t * q_heads + h0) * head_dim;
        const float *sq = S->sq + (size_t)t * q_heads + h0;
/* as_scores_avx2 keeps one accumulator for each head, up to 8: more heads
 * (the 16 of a global layer of the 12B) go in blocks of 8, with the same
 * values for each head */
#define AS_SC(NH)                                                                           \
        for (int j = 0; j < cn; ++j) {                                                      \
            size_t r = base + (size_t)j;                                                    \
            if (j + AS_PF < cn) {                                                           \
                as_prefetch_row(AS_ATE(kq, (r + AS_PF) * kv_stride + (size_t)kv * head_dim, \
                                       KI8), head_dim, KI8);                                \
                as_prefetch_row(AS_ATE(vq, (r + AS_PF) * kv_stride + (size_t)kv * head_dim, \
                                       VI8), head_dim, VI8);                                \
            }                                                                               \
            for (int hb = 0; hb < (NH); hb += 8) {                                          \
                as_scores_avx2(qq + (size_t)hb * head_dim, sq + hb,                         \
                               (NH) - hb < 8 ? (NH) - hb : 8,                               \
                               AS_ATE(kq, r * kv_stride + (size_t)kv * head_dim, KI8),      \
                               ks + r * ks_stride + (size_t)kv * g, head_dim,               \
                               isc + (size_t)hb * ilen + j, ilen, KI8);                     \
            }                                                                               \
        }
        if (!TQ) {
            switch (n_rep) {
            case 1: AS_SC(1) break;
            case 2: AS_SC(2) break;
            case 4: AS_SC(4) break;
            case 8: AS_SC(8) break;
            default: AS_SC(n_rep)
            }
        }
#undef AS_SC
        for (int j = 0; TQ && n_rep <= 8 && j < cn; ++j) {
            size_t r = base + (size_t)j;
            as_scores_tq_avx2(qt, n_rep, (const uint8_t *)AS_ATE(kq, r * kv_stride +
                                                                  (size_t)kv * head_dim, 2),
                              ks + r * ks_stride + (size_t)kv * g, head_dim, isc + j, ilen);
        }
        for (int j = 0; TQ && n_rep > 8 && j < cn; ++j) {
#else
        for (int j = 0; j < cn; ++j) {
#endif
            size_t r = base + (size_t)j;
            as_row(AS_ATE(kq, r * kv_stride + (size_t)kv * head_dim, KI8),
                   ks + r * ks_stride + (size_t)kv * g, head_dim, row, KI8);
            for (int hh = 0; hh < n_rep; ++hh) {
                isc[(size_t)hh * ilen + j] =
                    as_dot(qt + (size_t)hh * head_dim, row, head_dim);
            }
        }
        for (int hh = 0; hh < n_rep; ++hh) {
            float *sc = isc + (size_t)hh * ilen;
            float m = sc[0];
            for (int j = 1; j < cn; ++j) {
                m = sc[j] > m ? sc[j] : m;
            }
            /* A weight below exp(-64) (2e-28 of the largest) is 0. Its
             * product with a value scale can be a denormal, and a product
             * with a denormal takes a slow microcode path on x86. The value
             * pass then skips the key. */
#if GEMMA_X86 && !defined(__AVX512F__)
            float l = as_softmax_avx2(sc, cn, m);
#else
            float l = 0.0f;
            for (int j = 0; j < cn; ++j) {
                float d = sc[j] - m;
                sc[j] = d < -64.0f ? 0.0f : expf(d);
                l += sc[j];
            }
#endif
            float *pp = S->parts + (((size_t)t * q_heads + h0 + hh) * AS_MAXC + c) * pw;
            for (int d = 0; d < head_dim; ++d) {
                pp[d] = 0.0f;
            }
            pp[head_dim] = m;
            pp[head_dim + 1] = l;
        }
#if GEMMA_X86 && !defined(__AVX512F__)
        unsigned char live[1 << 12];
        if ((!TQ || (n_rep > 2 && n_rep <= 8)) && cn <= (1 << 12)) {
            for (int j = 0; j < cn; ++j) {
                int any = 0;
                for (int hh = 0; hh < n_rep; ++hh) {
                    any |= isc[(size_t)hh * ilen + j] != 0.0f;
                }
                live[j] = (unsigned char)any;
            }
            as_values_avx2(isc, ilen, live, n_rep,
                           vq, vs, kv_stride, ks_stride, kv, head_dim, base, cn,
                           S->parts + (((size_t)t * q_heads + h0) * AS_MAXC + c) * pw,
                           AS_MAXC * pw, VI8);
            continue;
        }
#endif
        for (int j = 0; j < cn; ++j) {
            size_t r = base + (size_t)j;
            int any = 0;
            for (int hh = 0; hh < n_rep; ++hh) {
                any |= isc[(size_t)hh * ilen + j] != 0.0f;
            }
            if (!any) {
                continue;
            }
            as_row(AS_ATE(vq, r * kv_stride + (size_t)kv * head_dim, VI8),
                   vs + r * ks_stride + (size_t)kv * g, head_dim, row, VI8);
            for (int hh = 0; hh < n_rep; ++hh) {
                float p = isc[(size_t)hh * ilen + j];
                if (p != 0.0f) {
                    float *pp = S->parts + (((size_t)t * q_heads + h0 + hh) * AS_MAXC + c) * pw;
                    as_axpy(pp, p, row, head_dim);
                }
            }
        }
    }
    /* the join of the chunks of each query head */
    #pragma omp for schedule(static)
    for (int u = 0; u < tokens * q_heads; ++u) {
        int t = u / q_heads;
        int nc = (n[t] + AS_LEN(n[t]) - 1) / AS_LEN(n[t]);
        const float *pp = S->parts + (size_t)u * AS_MAXC * pw;
        float M = pp[head_dim];
        for (int c = 1; c < nc; ++c) {
            float mc = pp[(size_t)c * pw + head_dim];
            M = mc > M ? mc : M;
        }
        float den = 0.0f;
        float w[AS_MAXC];
        for (int c = 0; c < nc; ++c) {
            float d = pp[(size_t)c * pw + head_dim] - M;
            w[c] = d < -64.0f ? 0.0f : expf(d);
            den += w[c] * pp[(size_t)c * pw + head_dim + 1];
        }
        float inv = 1.0f / den;
        float *o = out + (size_t)u * head_dim;
#if GEMMA_X86
        /* the chunks in their order for each value, 8 values at a time */
        for (int d = 0; d < head_dim; d += 8) {
            __m256 acc = _mm256_setzero_ps();
            for (int c = 0; c < nc; ++c) {
                if (w[c] != 0.0f) {
                    acc = _mm256_fmadd_ps(_mm256_set1_ps(w[c]),
                                          _mm256_loadu_ps(pp + (size_t)c * pw + d), acc);
                }
            }
            _mm256_storeu_ps(o + d, _mm256_mul_ps(acc, _mm256_set1_ps(inv)));
        }
#else
        for (int d = 0; d < head_dim; ++d) {
            float acc = 0.0f;
            for (int c = 0; c < nc; ++c) {
                if (w[c] != 0.0f) {
                    acc += w[c] * pp[(size_t)c * pw + d];
                }
            }
            o[d] = acc * inv;
        }
#endif
    }
    return 1;
}

static int gemma_attn_split_i16_body(const float *q, const int16_t *kq, const float *ks,
                                     const int16_t *vq, const float *vs, float *scores,
                                     float *out, int q_heads, int kv_heads, int head_dim,
                                     const int *lo, const int *n, int nmax, int tokens)
{
    return as_split_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n,
                         nmax, tokens, 0, 0, kv_heads);
}

/* gemma_attn_split_i16_body for the heads of a part: a row of the cache holds
 * row_heads heads, and kq, ks, vq, vs point at the first head of the part
 * (GP_ATTN_QC_H). Each head has the bits of gemma_attn_split_i16_body. */
static int gemma_attn_split_i16_rows_body(const float *q, const int16_t *kq, const float *ks,
                                          const int16_t *vq, const float *vs, float *out,
                                          int q_heads, int kv_heads, int head_dim, int n,
                                          int row_heads)
{
    return as_split_body(q, kq, ks, vq, vs, NULL, out, q_heads, kv_heads, head_dim, NULL, &n,
                         n, 1, 0, 0, row_heads);
}

/* The same over the int8 cache (RECLANK_KV_INT8): values int8 with a scale
 * of max |x| / 127 for each group of 32 (gemma_quant_group32_i8). */
static int gemma_attn_split_i8_body(const float *q, const int8_t *kq, const float *ks,
                                    const int8_t *vq, const float *vs, float *scores,
                                    float *out, int q_heads, int kv_heads, int head_dim,
                                    const int *lo, const int *n, int nmax, int tokens)
{
    return as_split_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n,
                         nmax, tokens, 1, 1, kv_heads);
}

/* The same over the cache with int16 keys and int8 values (RECLANK_KV_INT8
 * v: KVCache kv="k16v8"). */
static int gemma_attn_split_v8_body(const float *q, const int16_t *kq, const float *ks,
                                    const int8_t *vq, const float *vs, float *scores,
                                    float *out, int q_heads, int kv_heads, int head_dim,
                                    const int *lo, const int *n, int nmax, int tokens)
{
    return as_split_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n,
                         nmax, tokens, 0, 1, kv_heads);
}

/* The same over the TQ6 cache (reclank/formats/tq6.py): q has the rotation, and so
 * has out (GP_TQ_ROT gives them and takes it away). */
static int gemma_attn_split_tq6_body(const float *q, const uint8_t *kq, const float *ks,
                                     const uint8_t *vq, const float *vs, float *scores,
                                     float *out, int q_heads, int kv_heads, int head_dim,
                                     const int *lo, const int *n, int nmax, int tokens)
{
    return as_split_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n,
                         nmax, tokens, 2, 2, kv_heads);
}
#undef AS_AT
#undef AS_ATE
#undef AS_EL

/* gemma_attn_split_i16_body in its own parallel region (a test). Return 1
 * when it ran. */
int rc_attn_split_i16(const float *q, const int16_t *kq, const float *ks, const int16_t *vq,
                         const float *vs, float *scores, float *out, int q_heads, int kv_heads,
                         int head_dim, int n)
{
    int ok = 0;
    #pragma omp parallel
    {
        int r = gemma_attn_split_i16_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                          head_dim, NULL, &n, n, 1);
        #pragma omp master
        ok = r;
    }
    return ok;
}

/* The attention of one query (lo null, n[0] keys) or of a group of queries
 * over the int8 cache (k16: the int16 keys of the form k16v8), in its own
 * parallel region (the Python paths and the drafter). Return 0 when the
 * shape does not fit. */
int rc_attn_decode_q8(const float *q, const void *kq, const float *ks, const int8_t *vq,
                         const float *vs, float *scores, float *out, int q_heads, int kv_heads,
                         int head_dim, const int *lo, const int *n, int nmax, int tokens,
                         int k16)
{
    int ok = 0;
    #pragma omp parallel
    {
        int r = k16 ? gemma_attn_split_v8_body(q, (const int16_t *)kq, ks, vq, vs, scores, out,
                                               q_heads, kv_heads, head_dim, lo, n, nmax, tokens)
                    : gemma_attn_split_i8_body(q, (const int8_t *)kq, ks, vq, vs, scores, out,
                                               q_heads, kv_heads, head_dim, lo, n, nmax, tokens);
        #pragma omp master
        ok = r;
    }
    return ok;
}

/* rc_attn_decode_q8 over the TQ6 cache (a test): q and out in the
 * rotated form. */
int rc_attn_decode_tq6(const float *q, const uint8_t *kq, const float *ks, const uint8_t *vq,
                          const float *vs, float *out, int q_heads, int kv_heads, int head_dim,
                          const int *lo, const int *n, int nmax, int tokens)
{
    int ok = 0;
    #pragma omp parallel
    {
        int r = gemma_attn_split_tq6_body(q, kq, ks, vq, vs, NULL, out, q_heads, kv_heads,
                                          head_dim, lo, n, nmax, tokens);
        #pragma omp master
        ok = r;
    }
    return ok;
}

/* The split form when it fits, as GP_ATTN_QC_MT, so the Python path of a
 * group (ops.attn_decode_mt) gives the bits of the program. */
void rc_attn_decode_i16_mt(const float *q, const int16_t *kq, const float *ks,
                              const int16_t *vq, const float *vs, float *scores,
                              float *out, int q_heads, int kv_heads, int head_dim,
                              const int *lo, const int *n, int nmax, int tokens)
{
    #pragma omp parallel
    {
        if (!gemma_attn_split_i16_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                       head_dim, lo, n, nmax, tokens)) {
            gemma_attn_decode_i16_mt_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                          head_dim, lo, n, nmax, tokens);
        }
    }
}

/* The fused decode attention for a small group of query tokens in one
 * parallel region. Token t reads the key rows lo[t] to lo[t] + n[t] - 1 of the
 * int8 cache. A decode step of that token reads the same rows.
 *
 * The shapes: qq is (tokens, q_heads, head_dim). qs is (tokens, q_heads,
 * groups). The array scores holds tokens * q_heads * nmax values. The array
 * out is (tokens, q_heads, head_dim). */
static void gemma_attn_decode_mt_body(const int8_t *qq, const float *qs,
                          const int8_t *kq, const float *ks,
                          const int8_t *vq, const float *vs,
                          float *scores, float *out,
                          int q_heads, int kv_heads, int head_dim,
                          const int *lo, const int *n, int nmax, int tokens)
{
    int g = head_dim / 32;
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)g;
    int total = tokens * q_heads;
    #pragma omp for schedule(static)
    for (int u = 0; u < total; ++u) {
        int t = u / q_heads;
        int h = u % q_heads;
        size_t r = (size_t)lo[t];
        attn_decode_head(qq + (size_t)u * (size_t)head_dim, qs + (size_t)u * (size_t)g,
                         kq + r * kv_stride, ks + r * ks_stride,
                         vq + r * kv_stride, vs + r * ks_stride,
                         scores + (size_t)u * (size_t)nmax,
                         out + (size_t)u * (size_t)head_dim, h / n_rep, head_dim,
                         kv_stride, ks_stride, n[t]);
    }
}

void rc_attn_decode_mt(const int8_t *qq, const float *qs,
                          const int8_t *kq, const float *ks,
                          const int8_t *vq, const float *vs,
                          float *scores, float *out,
                          int q_heads, int kv_heads, int head_dim,
                          const int *lo, const int *n, int nmax, int tokens)
{
    #pragma omp parallel
    gemma_attn_decode_mt_body(qq, qs, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, lo, n, nmax, tokens);
}
