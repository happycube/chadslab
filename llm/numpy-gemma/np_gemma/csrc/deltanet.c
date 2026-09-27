/* The Gated DeltaNet (the linear attention of Qwen3.5 / Qwen3-Next).
 *
 * This file is part of the cops library: bf16_linear.c includes it. The
 * recurrence of one value head h for one token (torch_recurrent_gated_delta_
 * rule of transformers):
 *
 *     q, k = l2norm(q), l2norm(k); q = q / sqrt(k_dim)
 *     decay = exp(-exp(A_log[h]) * softplus(a + dt_bias[h])); beta = sigmoid(b)
 *     S = S * decay;  d = (v - S^T k) * beta;  S = S + k d^T;  o = S^T q
 *     out = rms_norm(o) * norm_w * silu(z)
 */

static inline float gdn_silu(float v)
{
    return v / (1.f + expf(-v));
}

/* The Gated DeltaNet for t tokens, in order (the decode step, or a small
 * group). qkv (t x conv_dim) is the output of in_proj_qkv; conv (3 x
 * conv_dim) holds the last inputs and gets the new ones. z (t x v_heads *
 * v_dim), a and b (t x v_heads). S (v_heads x k_dim x v_dim) is the state.
 * out (t x v_heads * v_dim) gets rms_norm(o) * norm_w * silu(z).
 *
 * One parallel region: first the convolution of all the channels (tokens
 * in order), then each value head on its own thread. */
void gdn_step(const float *qkv, float *conv, const float *conv_w, int kernel,
              const float *z, const float *a, const float *b, const float *A_log,
              const float *dt_bias, const float *norm_w, float *S, float *out, float *scratch,
              int t, int k_heads, int v_heads, int k_dim, int v_dim, float eps)
{
    int kd = k_heads * k_dim, vd = v_heads * v_dim, cd = 2 * kd + vd;
    int rep = v_heads / k_heads;
    float *cv = scratch;                              /* t x cd, after conv and silu */
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int c = 0; c < cd; ++c) {
            const float *w = conv_w + (size_t)c * kernel;
            float hist[8];
            for (int j = 0; j < kernel - 1; ++j) {
                hist[j] = conv[(size_t)j * cd + c];
            }
            for (int i = 0; i < t; ++i) {
                float xin = qkv[(size_t)i * cd + c];
                float v = w[kernel - 1] * xin;
                for (int j = 0; j < kernel - 1; ++j) {
                    v += w[j] * hist[j];
                }
                cv[(size_t)i * cd + c] = gdn_silu(v);
                for (int j = 0; j < kernel - 2; ++j) {
                    hist[j] = hist[j + 1];
                }
                hist[kernel - 2] = xin;
            }
            for (int j = 0; j < kernel - 1; ++j) {
                conv[(size_t)j * cd + c] = hist[j];
            }
        }
        #pragma omp for schedule(static)
        for (int hv = 0; hv < v_heads; ++hv) {
            int hk = hv / rep;
            float *Sh = S + (size_t)hv * k_dim * v_dim;
            float q[256], kk[256], o[256], kv[256];
            float Aexp = expf(A_log[hv]);
            for (int i = 0; i < t; ++i) {
                const float *row = cv + (size_t)i * cd;
                const float *qs = row + hk * k_dim, *ks = row + kd + hk * k_dim;
                const float *vs = row + 2 * kd + hv * v_dim;
                float nq = 0.f, nk = 0.f;
                for (int d = 0; d < k_dim; ++d) {
                    nq += qs[d] * qs[d];
                    nk += ks[d] * ks[d];
                }
                float iq = 1.f / sqrtf(nq + 1e-6f) / sqrtf((float)k_dim), ik = 1.f / sqrtf(nk + 1e-6f);
                for (int d = 0; d < k_dim; ++d) {
                    q[d] = qs[d] * iq;
                    kk[d] = ks[d] * ik;
                }
                float av = a[(size_t)i * v_heads + hv] + dt_bias[hv];
                float sp = av > 20.f ? av : log1pf(expf(av));
                float decay = expf(-Aexp * sp);
                float beta = 1.f / (1.f + expf(-b[(size_t)i * v_heads + hv]));
                /* S *= decay; kv = S^T k */
                for (int e = 0; e < v_dim; ++e) {
                    kv[e] = 0.f;
                }
                for (int d = 0; d < k_dim; ++d) {
                    float *Sr = Sh + (size_t)d * v_dim;
                    float kd_ = kk[d];
                    for (int e = 0; e < v_dim; ++e) {
                        Sr[e] *= decay;
                        kv[e] += kd_ * Sr[e];
                    }
                }
                /* delta = (v - kv) * beta; S += k delta^T; o = S^T q */
                for (int e = 0; e < v_dim; ++e) {
                    kv[e] = (vs[e] - kv[e]) * beta;
                    o[e] = 0.f;
                }
                for (int d = 0; d < k_dim; ++d) {
                    float *Sr = Sh + (size_t)d * v_dim;
                    float kd_ = kk[d], qd = q[d];
                    for (int e = 0; e < v_dim; ++e) {
                        Sr[e] += kd_ * kv[e];
                        o[e] += qd * Sr[e];
                    }
                }
                /* the gated norm */
                float ss = 0.f;
                for (int e = 0; e < v_dim; ++e) {
                    ss += o[e] * o[e];
                }
                float inv = 1.f / sqrtf(ss / (float)v_dim + eps);
                const float *zr = z + (size_t)i * vd + (size_t)hv * v_dim;
                float *orow = out + (size_t)i * vd + (size_t)hv * v_dim;
                for (int e = 0; e < v_dim; ++e) {
                    orow[e] = o[e] * inv * norm_w[e] * gdn_silu(zr[e]);
                }
            }
        }
    }
}
