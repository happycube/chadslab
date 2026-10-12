/* ---------- mixture-of-experts router ----------
 * The router for one token does a small matrix product, a softmax, and a
 * top-k. The matrices are small, and the NumPy path then costs more in the
 * call of each small function than in the work. This kernel does the full
 * step. The parallel loop covers the experts.
 */
/* The logit of one expert: r times a row of the router projection. The
 * vector forms keep 4 accumulators and add them in one order; the router of
 * a step and of a group (gemma_router_mt_body) both use it, so a group
 * selects the experts of the steps. The scalar loop with 4 accumulators
 * did not vectorize (gcc keeps the order of float sums), and the router
 * took 95 us a layer on AVX2 with 6 threads. */
static inline float router_dot(const float *r, const float *pe, int hidden)
{
    int k = 0;
    float a = 0.0f;
#if GEMMA_X86 && defined(__AVX512F__)
    __m512 c0 = _mm512_setzero_ps(), c1 = _mm512_setzero_ps();
    __m512 c2 = _mm512_setzero_ps(), c3 = _mm512_setzero_ps();
    for (; k + 64 <= hidden; k += 64) {
        c0 = _mm512_fmadd_ps(_mm512_loadu_ps(r + k), _mm512_loadu_ps(pe + k), c0);
        c1 = _mm512_fmadd_ps(_mm512_loadu_ps(r + k + 16), _mm512_loadu_ps(pe + k + 16), c1);
        c2 = _mm512_fmadd_ps(_mm512_loadu_ps(r + k + 32), _mm512_loadu_ps(pe + k + 32), c2);
        c3 = _mm512_fmadd_ps(_mm512_loadu_ps(r + k + 48), _mm512_loadu_ps(pe + k + 48), c3);
    }
    a = _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(c0, c1), _mm512_add_ps(c2, c3)));
#elif GEMMA_X86
    __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
    __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
    for (; k + 32 <= hidden; k += 32) {
        c0 = _mm256_fmadd_ps(_mm256_loadu_ps(r + k), _mm256_loadu_ps(pe + k), c0);
        c1 = _mm256_fmadd_ps(_mm256_loadu_ps(r + k + 8), _mm256_loadu_ps(pe + k + 8), c1);
        c2 = _mm256_fmadd_ps(_mm256_loadu_ps(r + k + 16), _mm256_loadu_ps(pe + k + 16), c2);
        c3 = _mm256_fmadd_ps(_mm256_loadu_ps(r + k + 24), _mm256_loadu_ps(pe + k + 24), c3);
    }
    a = hsum256_ps(_mm256_add_ps(_mm256_add_ps(c0, c1), _mm256_add_ps(c2, c3)));
#endif
    for (; k < hidden; ++k) {
        a += r[k] * pe[k];
    }
    return a;
}

static void gemma_router_body(const float *x, const float *scale, const float *proj,
                              const float *per_expert, int hidden, int experts,
                              int top_k, float eps, float hscale, float *val,
                              int *idx, float *r, float *logits)
{
    #pragma omp single
    {
        float ss = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            ss += x[k] * x[k];
        }
        float inv = 1.0f / sqrtf(ss / (float)hidden + eps);
        for (int k = 0; k < hidden; ++k) {
            r[k] = x[k] * inv * scale[k] * hscale;
        }
    }
    #pragma omp for schedule(static)
    for (int e = 0; e < experts; ++e) {
        logits[e] = router_dot(r, proj + (size_t)e * (size_t)hidden, hidden);
    }
    #pragma omp single
    {
        float m = logits[0];
        for (int e = 1; e < experts; ++e) {
            if (logits[e] > m) {
                m = logits[e];
            }
        }
        float sum = 0.0f;
        for (int e = 0; e < experts; ++e) {
            logits[e] = expf(logits[e] - m);
            sum += logits[e];
        }
        float invs = 1.0f / sum;
        for (int e = 0; e < experts; ++e) {
            logits[e] *= invs;
        }
        for (int j = 0; j < top_k; ++j) {
            int best = 0;
            float bv = logits[0];
            for (int e = 1; e < experts; ++e) {
                if (logits[e] > bv) {
                    bv = logits[e];
                    best = e;
                }
            }
            val[j] = bv;
            idx[j] = best;
            logits[best] = -1.0f;
        }
        float vs = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            vs += val[j];
        }
        float invv = 1.0f / vs;
        for (int j = 0; j < top_k; ++j) {
            val[j] = val[j] * invv * per_expert[idx[j]];
        }
    }
}

void rc_router(const float *x, const float *scale, const float *proj,
                  const float *per_expert, int hidden, int experts, int top_k,
                  float eps, float hscale, float *val, int *idx)
{
    float r[hidden];
    float logits[experts];
    #pragma omp parallel
    gemma_router_body(x, scale, proj, per_expert, hidden, experts, top_k, eps,
                      hscale, val, idx, r, logits);
}

/* The router for a small group of tokens. The steps for each token are the
 * steps of rc_router, so each token selects the same experts with the same
 * weights. One parallel region covers the logits of every token. x is
 * (tokens, hidden), val and idx are (tokens, top_k). tokens is at most 16. */
static void gemma_router_mt_body(const float *x, const float *scale, const float *proj,
                                 const float *per_expert, int hidden, int experts,
                                 int top_k, float eps, float hscale, float *val,
                                 int *idx, int tokens, float *r, float *logits)
{
    #pragma omp single
    for (int t = 0; t < tokens; ++t) {
        const float *xt = x + (size_t)t * (size_t)hidden;
        float *rt = r + (size_t)t * (size_t)hidden;
        float ss = 0.0f;
        for (int k = 0; k < hidden; ++k) {
            ss += xt[k] * xt[k];
        }
        float inv = 1.0f / sqrtf(ss / (float)hidden + eps);
        for (int k = 0; k < hidden; ++k) {
            rt[k] = xt[k] * inv * scale[k] * hscale;
        }
    }
    #pragma omp for schedule(static)
    for (int e = 0; e < experts; ++e) {
        const float *pe = proj + (size_t)e * (size_t)hidden;
        for (int t = 0; t < tokens; ++t) {
            logits[(size_t)t * (size_t)experts + e] =
                router_dot(r + (size_t)t * (size_t)hidden, pe, hidden);
        }
    }
    #pragma omp single
    for (int t = 0; t < tokens; ++t) {
        float *lg = logits + (size_t)t * (size_t)experts;
        float *vt = val + (size_t)t * (size_t)top_k;
        int *it = idx + (size_t)t * (size_t)top_k;
        float m = lg[0];
        for (int e = 1; e < experts; ++e) {
            if (lg[e] > m) {
                m = lg[e];
            }
        }
        float sum = 0.0f;
        for (int e = 0; e < experts; ++e) {
            lg[e] = expf(lg[e] - m);
            sum += lg[e];
        }
        float invs = 1.0f / sum;
        for (int e = 0; e < experts; ++e) {
            lg[e] *= invs;
        }
        for (int j = 0; j < top_k; ++j) {
            int best = 0;
            float bv = lg[0];
            for (int e = 1; e < experts; ++e) {
                if (lg[e] > bv) {
                    bv = lg[e];
                    best = e;
                }
            }
            vt[j] = bv;
            it[j] = best;
            lg[best] = -1.0f;
        }
        float vs = 0.0f;
        for (int j = 0; j < top_k; ++j) {
            vs += vt[j];
        }
        float invv = 1.0f / vs;
        for (int j = 0; j < top_k; ++j) {
            vt[j] = vt[j] * invv * per_expert[it[j]];
        }
    }
}

void rc_router_mt(const float *x, const float *scale, const float *proj,
                     const float *per_expert, int hidden, int experts, int top_k,
                     float eps, float hscale, float *val, int *idx, int tokens)
{
    float *r = (float *)malloc((size_t)tokens * (size_t)hidden * sizeof(float));
    float *logits = (float *)malloc((size_t)tokens * (size_t)experts * sizeof(float));
    #pragma omp parallel
    gemma_router_mt_body(x, scale, proj, per_expert, hidden, experts, top_k, eps,
                         hscale, val, idx, tokens, r, logits);
    free(r);
    free(logits);
}
