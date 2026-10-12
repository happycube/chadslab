/* ---------- fused attention for one query token ----------
 * A decode step makes one query token for each layer. The NumPy path then
 * builds a score matrix and calls a batched matrix product for a very small
 * matrix. This kernel does the full step: the scores, the softmax, and the
 * weighted sum of the values. It reads an int8 key cache and an int8 value
 * cache. Each cache holds one float32 scale for each group of 32 values.
 *
 * The caller gives the query in int8 form with its own group scales. The
 * query head h uses the key and value head h / (q_heads / kv_heads).
 */

#if GEMMA_X86
__attribute__((target("avx2,fma")))
static inline int32_t dot_i8_i8(const int8_t *a, const int8_t *b, int n)
{
    __m256i acc = _mm256_setzero_si256();
    for (int i = 0; i + 15 < n; i += 16) {
        __m256i va = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a + i)));
        __m256i vb = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(b + i)));
        acc = _mm256_add_epi32(acc, _mm256_madd_epi16(va, vb));
    }
    return hsum_epi32_avx2(acc);
}
#else
static inline int32_t dot_i8_i8(const int8_t *a, const int8_t *b, int n)
{
    int32_t d = 0;
    for (int i = 0; i < n; ++i) {
        d += (int32_t)a[i] * (int32_t)b[i];
    }
    return d;
}
#endif

/* Return the dot product of two float32 rows. */
#if GEMMA_X86 && defined(__AVX512F__)
static inline float dot_f32_f32(const float *a, const float *b, int n)
{
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    int k = 0;
    for (; k + 32 <= n; k += 32) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k + 16),
                             _mm512_loadu_ps(b + k + 16), a1);
    }
    float s = _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
    for (; k + 16 <= n; k += 16) {
        s += _mm512_reduce_add_ps(_mm512_mul_ps(_mm512_loadu_ps(a + k),
                                                _mm512_loadu_ps(b + k)));
    }
    for (; k < n; ++k) {
        s += a[k] * b[k];
    }
    return s;
}
#elif GEMMA_X86
static inline float dot_f32_f32(const float *a, const float *b, int n)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), a0);
        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8),
                             _mm256_loadu_ps(b + k + 8), a1);
    }
    __m256 s = _mm256_add_ps(a0, a1);
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    float out = _mm_cvtss_f32(r);
    for (; k < n; ++k) {
        out += a[k] * b[k];
    }
    return out;
}
#else
static inline float dot_f32_f32(const float *a, const float *b, int n)
{
    float s = 0.0f;
    for (int k = 0; k < n; ++k) {
        s += a[k] * b[k];
    }
    return s;
}
#endif

/* The row stride of a float32 KV cache from its head stride hs: position-
 * major (positions, kv heads, hd) has hs == hd and rows of kvh hd values;
 * head-major (kv heads, positions, hd) rows of hd (gpu.cu kv_rs). */
static inline size_t gp_kv_rs(size_t hs, int kvh, int hd)
{
    return hs == (size_t)hd ? (size_t)kvh * (size_t)hd : (size_t)hd;
}

/* One query token against a float32 key and value cache.
 *
 * q is (q_heads, head_dim). k and v are (kv_heads, n, head_dim), which is the
 * layout of the cache of the E4B model. The key and the value of one head lie
 * together, so the kernel reads them in order and copies nothing. A layout of
 * (n, kv_heads, head_dim) costs 2.4 times in the dot product, because the
 * hardware prefetch then jumps over the other head for each key. The batched
 * matrix multiply that the model used before also needs a transpose of the
 * key for each layer, and that copy costs 1.8 ms for a global layer at a
 * context of 512 tokens.
 *
 * scores is a scratch array of (q_heads, n) float32 values. pos is the
 * position of the query and base is the position of key zero. A window of
 * zero turns the sliding window off. out is (q_heads, head_dim).
 *
 * k_head_stride and v_head_stride give the distance between two heads, in
 * values. The cache holds a buffer that is larger than the part in use, so
 * the distance is not n * head_dim. The kernel reads the part in place. A
 * copy of the key and the value for each layer costs 98 MB for a token at a
 * context of 512.
 *
 * The parallel loop covers the query heads. A decode step of this model has 8
 * of them, so the number of busy threads is 8 and not 18.
 */
static void gemma_attn_decode_f32_body(const float *q, const float *k, const float *v,
                                       float *scores, float *out,
                                       int q_heads, int kv_heads, int head_dim, int n,
                                       long k_head_stride, long v_head_stride,
                                       long k_row, long v_row,
                                       int pos, int base, int window)
{
    if (n <= 0 || head_dim <= 0 || q_heads < kv_heads) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    #pragma omp for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        const int kv = h / n_rep;
        const float *qh = q + (size_t)h * (size_t)head_dim;
        const float *kh = k + (size_t)kv * (size_t)k_head_stride;
        const float *vh = v + (size_t)kv * (size_t)v_head_stride;
        float *sc = scores + (size_t)h * (size_t)n;
        float *oh = out + (size_t)h * (size_t)head_dim;
        for (int j = 0; j < n; ++j) {
            const int kp = base + j;
            if (kp > pos || (window > 0 && pos - kp >= window)) {
                sc[j] = -INFINITY;
                continue;
            }
            sc[j] = dot_f32_f32(kh + (size_t)j * (size_t)k_row, qh, head_dim);
        }
        float m = -INFINITY;
        for (int j = 0; j < n; ++j) {
            if (sc[j] > m) {
                m = sc[j];
            }
        }
        if (m == -INFINITY) {
            m = 0.0f;
        }
        float l = 0.0f;
        for (int j = 0; j < n; ++j) {
            float e = expf(sc[j] - m);
            sc[j] = e;
            l += e;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        int d = 0;
#if GEMMA_X86 && defined(__AVX512F__)
        for (; d + 16 <= head_dim; d += 16) {
            _mm512_storeu_ps(oh + d, _mm512_setzero_ps());
        }
#elif GEMMA_X86
        for (; d + 8 <= head_dim; d += 8) {
            _mm256_storeu_ps(oh + d, _mm256_setzero_ps());
        }
#endif
        for (; d < head_dim; ++d) {
            oh[d] = 0.0f;
        }
        for (int j = 0; j < n; ++j) {
            const float p = sc[j] * inv;
            if (p == 0.0f) {
                continue;
            }
            const float *vp = vh + (size_t)j * (size_t)v_row;
            d = 0;
#if GEMMA_X86 && defined(__AVX512F__)
            const __m512 pv = _mm512_set1_ps(p);
            for (; d + 16 <= head_dim; d += 16) {
                _mm512_storeu_ps(oh + d, _mm512_fmadd_ps(
                    pv, _mm512_loadu_ps(vp + d), _mm512_loadu_ps(oh + d)));
            }
#elif GEMMA_X86
            const __m256 pv = _mm256_set1_ps(p);
            for (; d + 8 <= head_dim; d += 8) {
                _mm256_storeu_ps(oh + d, _mm256_fmadd_ps(
                    pv, _mm256_loadu_ps(vp + d), _mm256_loadu_ps(oh + d)));
            }
#endif
            for (; d < head_dim; ++d) {
                oh[d] += p * vp[d];
            }
        }
    }
}

/* A float32 cache by its head strides: the row stride from them (gp_kv_rs:
 * position-major, the form of all the caches now, or head-major). */
void rc_attn_decode_f32(const float *q, const float *k, const float *v,
                           float *scores, float *out,
                           int q_heads, int kv_heads, int head_dim, int n,
                           long k_head_stride, long v_head_stride,
                           int pos, int base, int window)
{
    long kr = (long)gp_kv_rs((size_t)k_head_stride, kv_heads, head_dim);
    long vr = (long)gp_kv_rs((size_t)v_head_stride, kv_heads, head_dim);
    #pragma omp parallel
    gemma_attn_decode_f32_body(q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
                               k_head_stride, v_head_stride, kr, vr, pos, base, window);
}

/* The cache of Model keeps one row for each position: (rows, kv_heads,
 * head_dim). The head stride is head_dim and the row stride is
 * kv_heads * head_dim. A decode step with the float cache uses this kernel. */
void rc_attn_decode_f32s(const float *q, const float *k, const float *v,
                            float *scores, float *out,
                            int q_heads, int kv_heads, int head_dim, int n,
                            int pos, int base, int window)
{
    #pragma omp parallel
    gemma_attn_decode_f32_body(q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
                               head_dim, head_dim, (long)kv_heads * head_dim,
                               (long)kv_heads * head_dim, pos, base, window);
}

/* The int8 attention for one query token. The attribute keeps the AVX2
 * version, which the baseline build needs. */
#if GEMMA_X86
__attribute__((target("avx2,fma")))
#endif
/* One query head of the fused decode attention over n key rows. */
static inline void attn_decode_head(const int8_t *qh, const float *qsh,
                                    const int8_t *kq, const float *ks,
                                    const int8_t *vq, const float *vs,
                                    float *sc, float *oh, int kv, int head_dim,
                                    size_t kv_stride, size_t ks_stride, int n)
{
    int g = head_dim / 32;
    for (int j = 0; j < n; ++j) {
        const int8_t *kp = kq + (size_t)j * kv_stride + (size_t)kv * (size_t)head_dim;
        const float *ksp = ks + (size_t)j * ks_stride + (size_t)kv * (size_t)g;
        float s = 0.0f;
        for (int gg = 0; gg < g; ++gg) {
            s += (float)dot_i8_i8(kp + (size_t)gg * 32, qh + (size_t)gg * 32, 32)
                 * ksp[gg] * qsh[gg];
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
        const int8_t *vp = vq + (size_t)j * kv_stride + (size_t)kv * (size_t)head_dim;
        const float *vsp = vs + (size_t)j * ks_stride + (size_t)kv * (size_t)g;
        for (int gg = 0; gg < g; ++gg) {
#if GEMMA_X86
            __m256 pv = _mm256_set1_ps(p * vsp[gg]);
            const int8_t *vpp = vp + (size_t)gg * 32;
            float *op = oh + (size_t)gg * 32;
            for (int i = 0; i < 32; i += 8) {
                __m256 vf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(
                    _mm_loadl_epi64((const __m128i *)(vpp + i))));
                _mm256_storeu_ps(op + i, _mm256_fmadd_ps(pv, vf, _mm256_loadu_ps(op + i)));
            }
#else
            const float pv = p * vsp[gg];
            const int8_t *vpp = vp + (size_t)gg * 32;
            float *op = oh + (size_t)gg * 32;
            for (int i = 0; i < 32; ++i) {
                op[i] = fmaf(pv, (float)vpp[i], op[i]);
            }
#endif
        }
    }
}

static void gemma_attn_decode_body(const int8_t *qq, const float *qs,
                       const int8_t *kq, const float *ks,
                       const int8_t *vq, const float *vs,
                       float *scores, float *out,
                       int q_heads, int kv_heads, int head_dim, int n)
{
    int g = head_dim / 32;
    int n_rep = q_heads / kv_heads;
    size_t kv_stride = (size_t)kv_heads * (size_t)head_dim;
    size_t ks_stride = (size_t)kv_heads * (size_t)g;
    #pragma omp for schedule(static)
    for (int h = 0; h < q_heads; ++h) {
        attn_decode_head(qq + (size_t)h * (size_t)head_dim, qs + (size_t)h * (size_t)g,
                         kq, ks, vq, vs, scores + (size_t)h * (size_t)n,
                         out + (size_t)h * (size_t)head_dim, h / n_rep, head_dim,
                         kv_stride, ks_stride, n);
    }
}

void rc_attn_decode(const int8_t *qq, const float *qs,
                       const int8_t *kq, const float *ks,
                       const int8_t *vq, const float *vs,
                       float *scores, float *out,
                       int q_heads, int kv_heads, int head_dim, int n)
{
    #pragma omp parallel
    gemma_attn_decode_body(qq, qs, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, n);
}
