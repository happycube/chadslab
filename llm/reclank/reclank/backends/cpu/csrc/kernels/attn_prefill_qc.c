/* ---------- the prompt attention over the int16 cache ----------
 *
 * rc_attn_prefill on the quantized cache (KVCache: int16 values and a
 * float32 scale for each group of 32), with no float copy of the cache:
 * the team dequantizes the keys while it transposes them (kt), and the
 * values of the visible rows only (vf), then runs the tasks of
 * rc_attn_prefill. A value is (float)q * scale, as KVCache.read gives
 * it, and the tasks are the same: the bits of KVCache.read and
 * rc_attn_prefill. The scratch is one set for each team (the parts run
 * this at the same time). */
typedef struct {
    float *kt, *vf;
    size_t kt_n, vf_n;
} aq_scratch;
static aq_scratch aq_scr[AS_SLOTS];

static void gemma_attn_prefill_qc_body(const float *q, const int16_t *kq, const float *ks,
                                       const int16_t *vq, const float *vs,
                                       const int32_t *positions, int base, int window,
                                       float *out, int t, int n, int q_heads, int kv_heads,
                                       int hd, const int32_t *limit)
{
#if GEMMA_X86
#if defined(__AVX512F__)
    const int wide = (gemma_attn_impl == 2 || hd % 16 != 0) ? 8 : 16;
#else
    const int wide = 8;
#endif
    if (t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads || hd % 32 != 0 || hd % wide != 0 ||
        gemma_attn_impl == 1) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    int lo_key = 0;
    if (window > 0) {
        lo_key = positions[0] - window + 1 - base;
        if (lo_key < 0) {
            lo_key = 0;
        }
    }
    const int nvis = n - lo_key;
    if (nvis <= 0) {
        #pragma omp for schedule(static)
        for (long i = 0; i < (long)t * q_heads * hd; ++i) {
            out[i] = 0.0f;
        }
        return;
    }
    const int ld = ((nvis + wide - 1) / wide) * wide;
    const int g = hd / 32;
    const size_t row = (size_t)kv_heads * (size_t)hd;
    aq_scratch *S = &aq_scr[omp_get_level() >= 2 ? omp_get_ancestor_thread_num(1) + 1 : 0];
    #pragma omp single
    {
        size_t need = (size_t)kv_heads * hd * ld;
        if (S->kt_n < need) {
            free(S->kt);
            S->kt = (float *)aligned_alloc(64, ((need * 4 + 63) / 64) * 64);
            S->kt_n = need;
        }
        need = (size_t)n * row;     /* rows lo_key to n - 1 are written */
        if (S->vf_n < need) {
            free(S->vf);
            S->vf = (float *)aligned_alloc(64, ((need * 4 + 63) / 64) * 64);
            S->vf_n = need;
        }
    }
    float *kt = S->kt, *vf = S->vf;
    /* the keys: the transposed float rows of each head, blocks of 16 keys */
    const int jbs = (nvis + 15) / 16;
    #pragma omp for schedule(static)
    for (int it = 0; it < kv_heads * jbs; ++it) {
        const int h = it / jbs, jb = (it % jbs) * 16;
        const int je = jb + 16 < nvis ? jb + 16 : nvis;
        float *dsth = kt + (size_t)h * (size_t)hd * (size_t)ld;
        for (int j = jb; j < je; ++j) {
            const size_t r = (size_t)(lo_key + j);
            const int16_t *src = kq + r * row + (size_t)h * hd;
            const float *sc = ks + (r * kv_heads + h) * (size_t)g;
            for (int d = 0; d < hd; ++d) {
                dsth[(size_t)d * ld + j] = (float)src[d] * sc[d / 32];
            }
        }
    }
    /* the values of the visible rows, in groups of 32 */
    #pragma omp for schedule(static)
    for (long gi = (long)lo_key * kv_heads * g; gi < (long)n * kv_heads * g; ++gi) {
        const int16_t *src = vq + (size_t)gi * 32;
        float *dst = vf + (size_t)gi * 32;
        const float sc = vs[gi];
        for (int e = 0; e < 32; ++e) {
            dst[e] = (float)src[e] * sc;
        }
    }
    const long rows = (long)t * n_rep, last = rows - 1;
#if defined(__AVX512F__)
    if (wide == 16) {
        const int nrb = (int)((rows + ATTN5_MC - 1) / ATTN5_MC);
        gemma_attn_prefill_avx512_tasks(q, vf, kt, positions, base, window, out, n, q_heads,
                                        kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb,
                                        kv_heads * nrb, limit);
        return;
    }
#endif
    const int nrb = (int)((rows + ATTN2_MC - 1) / ATTN2_MC);
    gemma_attn_prefill_avx2_tasks(q, vf, kt, positions, base, window, out, n, q_heads,
                                  kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb,
                                  kv_heads * nrb, limit);
#endif
}

/* 1 when rc_attn_prefill_qc takes this shape (else the caller reads the
 * float rows and runs rc_attn_prefill). */
int rc_attn_prefill_qc_ok(int q_heads, int kv_heads, int hd)
{
#if GEMMA_X86
    return gemma_attn_impl != 1 && hd > 0 && hd % 32 == 0 && q_heads >= kv_heads &&
           q_heads % kv_heads == 0;
#else
    (void)q_heads; (void)kv_heads; (void)hd;
    return 0;
#endif
}

void rc_attn_prefill_qc(const float *q, const int16_t *kq, const float *ks, const int16_t *vq,
                           const float *vs, const int32_t *positions, int base, int window,
                           float *out, int t, int n, int q_heads, int kv_heads, int hd,
                           const int32_t *limit)
{
    #pragma omp parallel
    gemma_attn_prefill_qc_body(q, kq, ks, vq, vs, positions, base, window, out, t, n, q_heads,
                               kv_heads, hd, limit);
}
