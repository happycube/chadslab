/* ---------- the int16 copy of the key and value cache ----------
 *
 * The cache keeps a copy of each key and value as int16, with one float32
 * scale for each group of 32 values. The query stays float32. The error of
 * the attention output is then about 4e-5 of its size. An int8 copy with an
 * int8 query gave about 1e-2, which moved a logit by up to 2.6. The int16
 * copy reads 2.125 bytes for each value, against 4 for the float cache.
 *
 * The quantizer is plain C with lrintf, which rounds to the nearest even
 * value. Every caller uses it, so the Python path and the program give the
 * same bits. */
static inline float gemma_quant_group32_i16(const float *x, int16_t *q)
{
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) {
        float a = fabsf(x[k]);
        if (a > amax) {
            amax = a;
        }
    }
    float sc = amax > 0.0f ? amax / 32767.0f : 1e-12f;
    for (int k = 0; k < 32; ++k) {
        long v = lrintf(x[k] / sc);
        if (v > 32767) {
            v = 32767;
        }
        if (v < -32767) {
            v = -32767;
        }
        q[k] = (int16_t)v;
    }
    return sc;
}

/* Quantize groups of 32 values to int16. x holds groups * 32 values. */
static void gemma_quantize_i16_groups_body(const float *x, int16_t *q, float *s,
                                           long groups)
{
    #pragma omp for schedule(static)
    for (long g = 0; g < groups; ++g) {
        s[g] = gemma_quant_group32_i16(x + (size_t)g * 32, q + (size_t)g * 32);
    }
}

void rc_quantize_i16_groups(const float *x, int16_t *q, float *s, long groups)
{
    #pragma omp parallel if(groups >= 1024)
    gemma_quantize_i16_groups_body(x, q, s, groups);
}

/* The int8 cache (RECLANK_KV_INT8): a group of 32 values gets the scale
 * max |x| / 127, and each value is x / scale rounded to the nearest (ties
 * to even, lrintf), as the int16 form. */
static inline float gemma_quant_group32_i8(const float *x, int8_t *q)
{
    float amax = 0.0f;
    for (int k = 0; k < 32; ++k) {
        float a = fabsf(x[k]);
        if (a > amax) {
            amax = a;
        }
    }
    float sc = amax > 0.0f ? amax / 127.0f : 1e-12f;
    for (int k = 0; k < 32; ++k) {
        long v = lrintf(x[k] / sc);
        v = v > 127 ? 127 : (v < -127 ? -127 : v);
        q[k] = (int8_t)v;
    }
    return sc;
}

/* The float32 values of groups of 32 int8 values times the scale of the
 * group (the int8 cache: KVCache.read). */
void rc_dequantize_i8_groups(const int8_t *q, const float *s, float *out, long groups)
{
    #pragma omp parallel for schedule(static) if(groups >= 1024)
    for (long g = 0; g < groups; ++g) {
        for (int k = 0; k < 32; ++k) {
            out[(size_t)g * 32 + k] = (float)q[(size_t)g * 32 + k] * s[g];
        }
    }
}

void rc_quantize_i8_groups(const float *x, int8_t *q, float *s, long groups)
{
    #pragma omp parallel for schedule(static) if(groups >= 1024)
    for (long g = 0; g < groups; ++g) {
        s[g] = gemma_quant_group32_i8(x + (size_t)g * 32, q + (size_t)g * 32);
    }
}

/* The TQ6 cache of the Qwen models (reclank/formats/tq6.py has the description,
 * backends/include/tq6_tables.h the tables). A group of 32 values: the signs of
 * TQ6_SIGNS, the Walsh-Hadamard transform divided by sqrt(32), then the L2
 * norm (float32) and an index of 6 bits into tq6_cb for each value. The 24
 * bytes of a group: byte j < 16 has the low 4 bits of index j and of index
 * j + 16; byte 16 + j has the high 2 bits of indices j, j + 8, j + 16,
 * j + 24. */
#include "tq6_tables.h"
static const float tq6_cb[64] = TQ6_CODEBOOK;
static const float tq6_edges[63] = TQ6_EDGES;

/* The rotation of 32 values in place (inverse 0), or its inverse. */
static inline void tq6_rot32(float *x, int inverse)
{
    const float r = 0.17677669529663687f;    /* 1 / sqrt(32) */
    if (!inverse) {
        for (int i = 0; i < 32; ++i) {
            x[i] = (TQ6_SIGNS >> i) & 1u ? -x[i] : x[i];
        }
    }
    for (int h = 1; h < 32; h *= 2) {
        for (int i = 0; i < 32; i += 2 * h) {
            for (int j = i; j < i + h; ++j) {
                float a = x[j], b = x[j + h];
                x[j] = a + b;
                x[j + h] = a - b;
            }
        }
    }
    for (int i = 0; i < 32; ++i) {
        x[i] *= r;
        if (inverse && ((TQ6_SIGNS >> i) & 1u)) {
            x[i] = -x[i];
        }
    }
}

/* Quantize a group of 32 values to 24 bytes; return the norm. No fused
 * multiply-add: the GPU (k_kv_write_tq) gives the same bits. */
__attribute__((optimize("fp-contract=off")))
static float tq6_quant_group(const float *x, uint8_t *b)
{
    float y[32];
    memcpy(y, x, sizeof(y));
    tq6_rot32(y, 0);
    /* the sum of the squares in the order of the warp of the GPU (a sum
     * with xor shuffles), so both give the same bits */
    float s[32];
    for (int i = 0; i < 32; ++i) {
        s[i] = y[i] * y[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        for (int i = 0; i < off; ++i) {
            s[i] = s[i] + s[i + off];
        }
    }
    float nrm = sqrtf(s[0]);
    float inv = 1.0f / (nrm > 1e-30f ? nrm : 1e-30f);
    uint8_t idx[32];
    for (int i = 0; i < 32; ++i) {
        float u = y[i] * inv;
        /* the count of the edges below u (numpy searchsorted, side left) */
        int lo = 0, hi = 63;
        while (lo < hi) {
            int m = (lo + hi) / 2;
            if (tq6_edges[m] < u) {
                lo = m + 1;
            } else {
                hi = m;
            }
        }
        idx[i] = (uint8_t)lo;
    }
    for (int j = 0; j < 16; ++j) {
        b[j] = (uint8_t)((idx[j] & 15) | ((idx[j + 16] & 15) << 4));
    }
    for (int j = 0; j < 8; ++j) {
        b[16 + j] = (uint8_t)((idx[j] >> 4) | ((idx[j + 8] >> 4) << 2) |
                              ((idx[j + 16] >> 4) << 4) | ((idx[j + 24] >> 4) << 6));
    }
    return nrm;
}

/* The 32 rotated values of a group: norm * tq6_cb[index]. */
static inline __attribute__((always_inline)) void tq6_deq32(const uint8_t *b, float nrm,
                                                            float *out)
{
#if GEMMA_X86
    __m128i x = _mm_loadu_si128((const __m128i *)b);
    __m128i m15 = _mm_set1_epi8(15);
    __m256i nib = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(x, 4), m15),
                                   _mm_and_si128(x, m15));
    int64_t h8;
    memcpy(&h8, b + 16, 8);
    __m256i hb = _mm256_and_si256(_mm256_srlv_epi64(_mm256_set1_epi64x(h8),
                                                    _mm256_set_epi64x(6, 4, 2, 0)),
                                  _mm256_set1_epi8(3));
    __m256i idx = _mm256_or_si256(nib, _mm256_slli_epi16(hb, 4));
    __m128i i0 = _mm256_castsi256_si128(idx), i1 = _mm256_extracti128_si256(idx, 1);
#if defined(__AVX512F__)
    __m512 t0 = _mm512_loadu_ps(tq6_cb), t1 = _mm512_loadu_ps(tq6_cb + 16);
    __m512 t2 = _mm512_loadu_ps(tq6_cb + 32), t3 = _mm512_loadu_ps(tq6_cb + 48);
    __m512 s5 = _mm512_set1_ps(nrm);
    __m512i k32 = _mm512_set1_epi32(32);
    for (int h = 0; h < 2; ++h) {
        __m512i iv = _mm512_cvtepu8_epi32(h ? i1 : i0);
        __mmask16 up = _mm512_test_epi32_mask(iv, k32);
        __m512 v = _mm512_mask_blend_ps(up, _mm512_permutex2var_ps(t0, iv, t1),
                                        _mm512_permutex2var_ps(t2, iv, t3));
        _mm512_storeu_ps(out + 16 * h, _mm512_mul_ps(v, s5));
    }
#else
    /* the codebook is symmetric: index i >= 32 has the value tq6_cb[i] and
     * index i < 32 the value -tq6_cb[63 - i]. Four permutes of the 32
     * positive values and two blends find the value, then the sign. */
    __m256 s8 = _mm256_set1_ps(nrm);
    __m256 t0 = _mm256_loadu_ps(tq6_cb + 32), t1 = _mm256_loadu_ps(tq6_cb + 40);
    __m256 t2 = _mm256_loadu_ps(tq6_cb + 48), t3 = _mm256_loadu_ps(tq6_cb + 56);
    for (int h = 0; h < 4; ++h) {
        __m128i src = h < 2 ? i0 : i1;
        __m256i iv = _mm256_cvtepu8_epi32(h % 2 ? _mm_srli_si128(src, 8) : src);
        /* t: 0 for i >= 32, all ones for i < 32; m: the index of the positive value */
        __m256i t = _mm256_sub_epi32(_mm256_srli_epi32(iv, 5), _mm256_set1_epi32(1));
        __m256i m = _mm256_xor_si256(iv, t);
        __m256 b3 = _mm256_castsi256_ps(_mm256_slli_epi32(m, 28));
        __m256 lo = _mm256_blendv_ps(_mm256_permutevar8x32_ps(t0, m),
                                     _mm256_permutevar8x32_ps(t1, m), b3);
        __m256 hi = _mm256_blendv_ps(_mm256_permutevar8x32_ps(t2, m),
                                     _mm256_permutevar8x32_ps(t3, m), b3);
        __m256 v = _mm256_blendv_ps(lo, hi, _mm256_castsi256_ps(_mm256_slli_epi32(m, 27)));
        v = _mm256_xor_ps(v, _mm256_castsi256_ps(_mm256_slli_epi32(t, 31)));
        _mm256_storeu_ps(out + 8 * h, _mm256_mul_ps(v, s8));
    }
#endif
#else
    for (int j = 0; j < 32; ++j) {
        int lo4 = j < 16 ? b[j] & 15 : b[j - 16] >> 4;
        int hi2 = (b[16 + j % 8] >> (2 * (j / 8))) & 3;
        out[j] = nrm * tq6_cb[lo4 | (hi2 << 4)];
    }
#endif
}

/* The NumPy forms of the tests: groups of 32 values. */
void rc_tq6_quantize(const float *x, uint8_t *b, float *nrm, long groups)
{
    #pragma omp parallel for schedule(static) if(groups >= 1024)
    for (long g = 0; g < groups; ++g) {
        nrm[g] = tq6_quant_group(x + (size_t)g * 32, b + (size_t)g * 24);
    }
}

void rc_tq6_dequantize_rotated(const uint8_t *b, const float *nrm, float *out, long groups)
{
    for (long g = 0; g < groups; ++g) {
        tq6_deq32(b + (size_t)g * 24, nrm[g], out + (size_t)g * 32);
    }
}

void rc_tq6_rotate(float *x, long groups, int inverse)
{
    #pragma omp parallel for schedule(static) if(groups >= 1024)
    for (long g = 0; g < groups; ++g) {
        tq6_rot32(x + (size_t)g * 32, inverse);
    }
}

/* The float32 values of groups of 32 int16 values: q times the scale of
 * the group, as numpy gives them (KVCache.write_q). */
static void gemma_dequantize_i16_groups_body(const int16_t *q, const float *s, float *out,
                                             long groups)
{
    #pragma omp for schedule(static)
    for (long g = 0; g < groups; ++g) {
        const int16_t *qg = q + (size_t)g * 32;
        float *og = out + (size_t)g * 32;
        float sc = s[g];
        for (int k = 0; k < 32; ++k) {
            og[k] = (float)qg[k] * sc;
        }
    }
}

void rc_dequantize_i16_groups(const int16_t *q, const float *s, float *out, long groups)
{
    #pragma omp parallel if(groups >= 1024)
    gemma_dequantize_i16_groups_body(q, s, out, groups);
}

/* The dot product of 32 float32 values and 32 int16 values. */
static inline float dot32_f32_i16(const float *a, const int16_t *b)
{
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 b0 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
        _mm256_loadu_si256((const __m256i *)b)));
    __m512 b1 = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
        _mm256_loadu_si256((const __m256i *)(b + 16))));
    __m512 acc = _mm512_mul_ps(_mm512_loadu_ps(a), b0);
    acc = _mm512_fmadd_ps(_mm512_loadu_ps(a + 16), b1, acc);
    return _mm512_reduce_add_ps(acc);
#elif GEMMA_X86
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < 32; i += 8) {
        __m256 bv = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
            _mm_loadu_si128((const __m128i *)(b + i))));
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), bv, acc);
    }
    return hsum256_ps(acc);
#else
    float acc = 0.0f;
    for (int i = 0; i < 32; ++i) {
        acc += a[i] * (float)b[i];
    }
    return acc;
#endif
}

/* One query head over n rows of the int16 cache. qh is float32. rows (or
 * null for rows 0 to n - 1) gives the rows to read (QSA of qwen4exp). */
static inline void attn_i16_head_rows(const float *qh, const int16_t *kq, const float *ks,
                                      const int16_t *vq, const float *vs, float *sc,
                                      float *oh, int kv, int head_dim, size_t kv_stride,
                                      size_t ks_stride, int n, const int32_t *rows)
{
    int g = head_dim / 32;
    for (int j = 0; j < n; ++j) {
        size_t r = rows ? (size_t)rows[j] : (size_t)j;
        const int16_t *kp = kq + r * kv_stride + (size_t)kv * (size_t)head_dim;
        const float *ksp = ks + r * ks_stride + (size_t)kv * (size_t)g;
        float s = 0.0f;
        for (int gg = 0; gg < g; ++gg) {
            s += dot32_f32_i16(qh + (size_t)gg * 32, kp + (size_t)gg * 32) * ksp[gg];
        }
        sc[j] = s;
    }
    float m = sc[0];
    for (int j = 1; j < n; ++j) {
        if (sc[j] > m) {
            m = sc[j];
        }
    }
    float l = 0.0f;
    for (int j = 0; j < n; ++j) {
        float e = expf(sc[j] - m);
        sc[j] = e;
        l += e;
    }
    float inv = 1.0f / l;
    for (int d = 0; d < head_dim; ++d) {
        oh[d] = 0.0f;
    }
    for (int j = 0; j < n; ++j) {
        float p = sc[j] * inv;
        size_t r = rows ? (size_t)rows[j] : (size_t)j;
        const int16_t *vp = vq + r * kv_stride + (size_t)kv * (size_t)head_dim;
        const float *vsp = vs + r * ks_stride + (size_t)kv * (size_t)g;
        for (int gg = 0; gg < g; ++gg) {
            const int16_t *vpp = vp + (size_t)gg * 32;
            float *op = oh + (size_t)gg * 32;
#if GEMMA_X86 && defined(__AVX512F__)
            __m512 pv = _mm512_set1_ps(p * vsp[gg]);
            for (int i = 0; i < 32; i += 16) {
                __m512 vf = _mm512_cvtepi32_ps(_mm512_cvtepi16_epi32(
                    _mm256_loadu_si256((const __m256i *)(vpp + i))));
                _mm512_storeu_ps(op + i, _mm512_fmadd_ps(pv, vf, _mm512_loadu_ps(op + i)));
            }
#elif GEMMA_X86
            __m256 pv = _mm256_set1_ps(p * vsp[gg]);
            for (int i = 0; i < 32; i += 8) {
                __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(
                    _mm_loadu_si128((const __m128i *)(vpp + i))));
                _mm256_storeu_ps(op + i, _mm256_fmadd_ps(pv, vf, _mm256_loadu_ps(op + i)));
            }
#else
            float pv = p * vsp[gg];
            for (int i = 0; i < 32; ++i) {
                op[i] += pv * (float)vpp[i];
            }
#endif
        }
    }
}

static inline void attn_i16_head(const float *qh, const int16_t *kq, const float *ks,
                                 const int16_t *vq, const float *vs, float *sc,
                                 float *oh, int kv, int head_dim, size_t kv_stride,
                                 size_t ks_stride, int n)
{
    attn_i16_head_rows(qh, kq, ks, vq, vs, sc, oh, kv, head_dim, kv_stride, ks_stride, n, NULL);
}

/* The attention of one float32 query over n rows of the int16 cache. q is
 * (q_heads, head_dim). scores holds q_heads * n values. */
/* The same with the row strides of the cache as arguments. A part of a
 * program in parts (SPLIT_PLAN.md) runs only some heads. Its cache pointers
 * then point at its first head, and a row of the cache holds more heads than
 * the part runs. */
static void gemma_attn_decode_i16_s_body(const float *q, const int16_t *kq, const float *ks,
                                         const int16_t *vq, const float *vs, float *scores,
                                         float *out, int q_heads, int kv_heads,
                                         int head_dim, int n, size_t kv_stride,
                                         size_t ks_stride)
{
    int n_rep = q_heads / kv_heads;
    #pragma omp for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        attn_i16_head(q + (size_t)h * (size_t)head_dim, kq, ks, vq, vs,
                      scores + (size_t)h * (size_t)n, out + (size_t)h * (size_t)head_dim,
                      h / n_rep, head_dim, kv_stride, ks_stride, n);
    }
}

static void gemma_attn_decode_i16_body(const float *q, const int16_t *kq, const float *ks,
                                       const int16_t *vq, const float *vs, float *scores,
                                       float *out, int q_heads, int kv_heads,
                                       int head_dim, int n)
{
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)(head_dim / 32);
    #pragma omp for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        attn_i16_head(q + (size_t)h * (size_t)head_dim, kq, ks, vq, vs,
                      scores + (size_t)h * (size_t)n, out + (size_t)h * (size_t)head_dim,
                      h / n_rep, head_dim, kv_stride, ks_stride, n);
    }
}

static int gemma_attn_split_i16_body(const float *q, const int16_t *kq, const float *ks,
                                     const int16_t *vq, const float *vs, float *scores,
                                     float *out, int q_heads, int kv_heads, int head_dim,
                                     const int *lo, const int *n, int nmax, int tokens);

/* The split form when it fits (the attention of the MTP drafter over the
 * target cache), else one head for each thread. */
void rc_attn_decode_i16(const float *q, const int16_t *kq, const float *ks,
                           const int16_t *vq, const float *vs, float *scores,
                           float *out, int q_heads, int kv_heads, int head_dim, int n)
{
    #pragma omp parallel
    {
        if (!gemma_attn_split_i16_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                       head_dim, NULL, &n, n, 1)) {
            gemma_attn_decode_i16_body(q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
                                       head_dim, n);
        }
    }
}

/* The same for a group of queries. Query t reads the rows lo[t] to
 * lo[t] + n[t] - 1. A decode step of that query reads the same rows. q is
 * (tokens, q_heads, head_dim). scores holds nmax values for each thread
 * (the threads of the region; a larger buffer also serves). */
static void gemma_attn_decode_i16_mt_body(const float *q, const int16_t *kq,
                                          const float *ks, const int16_t *vq,
                                          const float *vs, float *scores, float *out,
                                          int q_heads, int kv_heads, int head_dim,
                                          const int *lo, const int *n, int nmax,
                                          int tokens)
{
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)(head_dim / 32);
    int total = tokens * q_heads;
    #pragma omp for schedule(static)
    for (int u = 0; u < total; ++u) {
        int t = u / q_heads;
        int h = u % q_heads;
        size_t r = (size_t)lo[t];
        attn_i16_head(q + (size_t)u * (size_t)head_dim, kq + r * kv_stride,
                      ks + r * ks_stride, vq + r * kv_stride, vs + r * ks_stride,
                      scores + (size_t)omp_get_thread_num() * (size_t)nmax,
                      out + (size_t)u * (size_t)head_dim, h / n_rep, head_dim, kv_stride,
                      ks_stride, n[t]);
    }
}
