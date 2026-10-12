/* ---------- the query, key, and value norm of one layer ----------
 * One attention layer applies RMSNorm to the query, the key, and the value.
 * Three NumPy calls then pay the call cost three times. This kernel applies
 * all three in one parallel loop. The value has no weight. The rows are
 * (tokens * heads, head_dim) and the data changes in place.
 */
static void gemma_qkv_norm_body(float *q, const float *q_w, int q_rows,
                                float *k, const float *k_w, int k_rows,
                                float *v, int v_rows, int head_dim, float eps)
{
    int total = q_rows + k_rows + v_rows;
    #pragma omp for schedule(static)
    for (int r = 0; r < total; ++r) {
        float *x;
        const float *w;
        if (r < q_rows) {
            x = q + (size_t)r * (size_t)head_dim;
            w = q_w;
        } else if (r < q_rows + k_rows) {
            x = k + (size_t)(r - q_rows) * (size_t)head_dim;
            w = k_w;
        } else {
            x = v + (size_t)(r - q_rows - k_rows) * (size_t)head_dim;
            w = NULL;
        }
        float ss = 0.0f;
        for (int i = 0; i < head_dim; ++i) {
            ss += x[i] * x[i];
        }
        float s = 1.0f / sqrtf(ss / (float)head_dim + eps);
        if (w != NULL) {
            for (int i = 0; i < head_dim; ++i) {
                /* The same order as rc_rms_norm, so the two give the same
                 * bits. */
                x[i] = x[i] * s * w[i];
            }
        } else {
            for (int i = 0; i < head_dim; ++i) {
                x[i] *= s;
            }
        }
    }
}

void rc_qkv_norm(float *q, const float *q_w, int q_rows,
                    float *k, const float *k_w, int k_rows,
                    float *v, int v_rows, int head_dim, float eps)
{
    int total = q_rows + k_rows + v_rows;
    #pragma omp parallel if(total >= 8)
    gemma_qkv_norm_body(q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps);
}

/* ---------- rotary position embedding ----------
 * Apply RoPE to the query and the key in place. The value does not turn. The
 * cos and sin tables have one row for each token and the full head width.
 * The two halves of the head use the first half of the table, because the
 * table joins the frequency vector to itself.
 */
/* The numpy path computes x * cos + rotate_half(x) * sin with two multiplies
 * and one add. A fused multiply and add rounds one time fewer, so the two
 * forms differ in the last bit. The model is sensitive to that difference
 * over a long context, and the two norm kernels of this file agree with the
 * numpy path exactly. Turn the contraction off here as well. */
#if defined(__GNUC__) && !defined(__clang__)
__attribute__((optimize("-ffp-contract=off")))
#endif
static void gemma_rope_body(float *q, int q_rows, int q_heads,
                            float *k, int k_rows, int k_heads,
                            const float *cos, const float *sin, int head_dim)
{
    int d = head_dim / 2;
    int total = q_rows + k_rows;
    #pragma omp for schedule(static)
    for (int r = 0; r < total; ++r) {
        float *x;
        int tok;
        if (r < q_rows) {
            x = q + (size_t)r * (size_t)head_dim;
            tok = r / q_heads;
        } else {
            int j = r - q_rows;
            x = k + (size_t)j * (size_t)head_dim;
            tok = j / k_heads;
        }
        const float *c = cos + (size_t)tok * (size_t)head_dim;
        const float *s = sin + (size_t)tok * (size_t)head_dim;
        for (int i = 0; i < d; ++i) {
            float a = x[i];
            float b = x[i + d];
            x[i] = a * c[i] - b * s[i];
            x[i + d] = b * c[i] + a * s[i];
        }
    }
}

void rc_rope(float *q, int q_rows, int q_heads,
                float *k, int k_rows, int k_heads,
                const float *cos, const float *sin, int head_dim)
{
    int total = q_rows + k_rows;
    #pragma omp parallel if(total >= 8)
    gemma_rope_body(q, q_rows, q_heads, k, k_rows, k_heads, cos, sin, head_dim);
}
