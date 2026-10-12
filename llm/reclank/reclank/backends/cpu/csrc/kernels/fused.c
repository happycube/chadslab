/* ---------- fused entry points for a decode step ----------
 * A decode step of the 26B model makes about 600 calls to this library. Each
 * call costs several microseconds before the kernel starts, because Python
 * must read the address of every array. The boundary between Python and C is
 * therefore a real part of the step.
 *
 * Every function below runs two kernels in one call. Each one is a
 * composition of the kernels above: the arithmetic is the arithmetic of those
 * kernels, in the same order. Thus the result does not change and the checks
 * of the model still hold.
 */

/* out = gelu(g) * u for n values, with the gate and the up part in separate
 * arrays. That is the shape of the shared MLP of the 26B model. */
void rc_gelu_mul_pair(const float *g, const float *u, float *out, int n)
{
    const float c = 0.7978845608028654f;
#if GEMMA_X86 && defined(__AVX512F__)
    const __m512 cv = _mm512_set1_ps(c);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 k3 = _mm512_set1_ps(0.044715f);
    const int nv = n & ~15;
    for (int i = 0; i < nv; i += 16) {
        const __m512 v = _mm512_loadu_ps(g + i);
        const __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
        const __m512 t = _mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v));
        const __m512 r = _mm512_mul_ps(_mm512_mul_ps(half, v),
                                       _mm512_add_ps(one, gemma_tanh_ps(t)));
        _mm512_storeu_ps(out + i, _mm512_mul_ps(r, _mm512_loadu_ps(u + i)));
    }
    for (int i = nv; i < n; ++i) {
        const float v = g[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))) * u[i];
    }
#else
    for (int i = GELU_MUL_VEC(g, u, out, n); i < n; ++i) {
        const float v = g[i];
        out[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))) * u[i];
    }
#endif
}

/* The three norms of the query, the key, and the value, and then the rotation
 * of the query and the key. v and k may be null. */
void rc_qkv_norm_rope(float *q, const float *q_w, int q_rows,
                         float *k, const float *k_w, int k_rows,
                         float *v, int v_rows, const float *cos,
                         const float *sin, int q_heads, int k_heads,
                         int head_dim, float eps)
{
    rc_qkv_norm(q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps);
    rc_rope(q, q_rows, q_heads, k, k_rows, k_heads, cos, sin, head_dim);
}

/* The norm of one row, then up to four int4 matrices on the result. scratch
 * holds cols values and belongs to the caller, which reuses it. */
void rc_rms_norm_multi4(const float *x, const float *wn, float *scratch,
                           int cols, float eps,
                           const uint8_t *w0, const float *s0, float *o0, int r0,
                           const uint8_t *w1, const float *s1, float *o1, int r1,
                           const uint8_t *w2, const float *s2, float *o2, int r2,
                           const uint8_t *w3, const float *s3, float *o3, int r3)
{
    rc_rms_norm(x, wn, scratch, 1, cols, eps);
    rc_int4_multi4(w0, s0, o0, r0, w1, s1, o1, r1,
                      w2, s2, o2, r2, w3, s3, o3, r3, scratch, cols);
}

/* gelu(g) * u, then one int4 matrix on the result. The gate and the up part
 * have inner values. The matrix has cols columns. scratch holds inner values
 * and belongs to the caller. */
void rc_gelu_mul_int4(const float *g, const float *u, int inner,
                         float *scratch, const uint8_t *w, const float *s,
                         float *out, int rows, int cols)
{
    rc_gelu_mul_pair(g, u, scratch, inner);
    rc_int4_linear(w, s, scratch, out, rows, cols, 1, 32);
}

/* The gate and up projection of the selected experts, then the GELU and the
 * multiply. act holds one row of 2 * inner values for each job and out holds
 * one row of inner values for each job. The two buffers must not overlap: the
 * output of one job would otherwise fall on the up part of another job. */
void rc_moe_gemv_gelu(const uint8_t *w, const float *scales, const float *x,
                         const int *ids, int jobs, float *act, float *out,
                         int rows, int cols, int xstride, int inner)
{
    rc_int4_moe_gemv(w, scales, x, ids, jobs, act, rows, cols, xstride);
    rc_gelu_mul(act, out, jobs, inner);
}

/* Apply the causal mask, the sliding window mask, and the softmax to the last
 * axis of x, in place. x is (rows, cols) and each row holds the scores of one
 * query head. The query token of a row is (row / n_rep) % n_tokens and its
 * position is positions[token]. The key of column c has the position base + c.
 * A key is masked when its position is after the query or when the distance is
 * the window or more. A window of zero turns the window off. */
static void gemma_softmax_mask_body(float *x, int rows, int cols, const int32_t *positions,
                        int n_tokens, int n_rep, int base, int window)
{
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const int tok = (r / n_rep) % n_tokens;
        const int q = positions[tok];
        float *row = x + (size_t)r * (size_t)cols;
        float m = -INFINITY;
        for (int c = 0; c < cols; ++c) {
            const int kp = base + c;
            if (kp > q || (window > 0 && q - kp >= window)) {
                row[c] = -INFINITY;
            } else if (row[c] > m) {
                m = row[c];
            }
        }
        if (m == -INFINITY) {
            m = 0.0f;
        }
        float l = 0.0f;
        for (int c = 0; c < cols; ++c) {
            float e = expf(row[c] - m);
            row[c] = e;
            l += e;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        for (int c = 0; c < cols; ++c) {
            row[c] *= inv;
        }
    }
}

void rc_softmax_mask(float *x, int rows, int cols, const int32_t *positions,
                        int n_tokens, int n_rep, int base, int window)
{
    #pragma omp parallel if(rows >= 8)
    gemma_softmax_mask_body(x, rows, cols, positions, n_tokens, n_rep, base, window);
}
