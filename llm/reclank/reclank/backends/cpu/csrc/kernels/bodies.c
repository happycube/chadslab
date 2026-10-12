/* ---------- bodies of the fused entry points ----------
 * Each kernel above has a body and a wrapper. The body holds the loops with an
 * orphaned "omp for". The wrapper opens the region and calls the body. A caller
 * that is already in a region, such as the program interpreter of
 * PERF_PLAN.md (phase 2), calls the bodies. The threads then stay in one
 * region for many kernels.
 *
 * The fused entry points call two kernels. Their bodies call the two bodies in
 * the same order. An "omp for" and an "omp single" end with a barrier, so the
 * second step sees the whole result of the first. The entry points keep their
 * own calls, so the result of the Python path does not change.
 */

static void gemma_q6k_linear_body(const uint8_t *w, const float *x, float *out,
                                  int rows, int cols, int tokens)
{
#if GEMMA_X86
    if (gemma_have_avx512()) {
        gemma_q6k_avx512_body(w, x, out, rows, cols, tokens);
    } else {
        gemma_q6k_avx2_body(w, x, out, rows, cols, tokens);
    }
#else
    gemma_q6k_scalar_body(w, x, out, rows, cols, tokens);
#endif
}

static void gemma_rms_norm_multi4_body(const float *x, const float *wn, float *scratch,
                                       int cols, float eps,
                                       const uint8_t *w0, const float *s0, float *o0, int r0,
                                       const uint8_t *w1, const float *s1, float *o1, int r1,
                                       const uint8_t *w2, const float *s2, float *o2, int r2,
                                       const uint8_t *w3, const float *s3, float *o3, int r3)
{
    #pragma omp single
    gemma_rms_norm_row(x, wn, scratch, cols, eps);
    gemma_int4_multi4_body(w0, s0, o0, r0, w1, s1, o1, r1,
                           w2, s2, o2, r2, w3, s3, o3, r3, scratch, cols);
}

static void gemma_gelu_mul_int4_body(const float *g, const float *u, int inner,
                                     float *scratch, const uint8_t *w, const float *s,
                                     float *out, int rows, int cols)
{
    #pragma omp single
    rc_gelu_mul_pair(g, u, scratch, inner);
    gemma_int4_linear_body(w, s, scratch, out, rows, cols, 1, 32);
}

static void gemma_moe_gemv_gelu_body(const uint8_t *w, const float *scales, const float *x,
                                     const int *ids, int jobs, float *act, float *out,
                                     int rows, int cols, int xstride, int inner)
{
    gemma_int4_moe_gemv_body(w, scales, x, ids, jobs, act, rows, cols, xstride);
    gemma_gelu_mul_body(act, out, jobs, inner);
}

static void gemma_moe_gemv_gelu_mt_body(const uint8_t *w, const float *scales,
                                        const float *x, const int *ids, const int *poff,
                                        const int *xi, int jobs, float *act, float *out,
                                        int rows, int cols, int xstride, int inner)
{
    gemma_int4_moe_gemv_mt_body(w, scales, x, ids, poff, xi, jobs, act, rows, cols, xstride);
    gemma_gelu_mul_body(act, out, poff[jobs], inner);
}

static void gemma_qkv_norm_rope_body(float *q, const float *q_w, int q_rows,
                                     float *k, const float *k_w, int k_rows,
                                     float *v, int v_rows, const float *cos,
                                     const float *sin, int q_heads, int k_heads,
                                     int head_dim, float eps)
{
    gemma_qkv_norm_body(q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps);
    gemma_rope_body(q, q_rows, q_heads, k, k_rows, k_heads, cos, sin, head_dim);
}
