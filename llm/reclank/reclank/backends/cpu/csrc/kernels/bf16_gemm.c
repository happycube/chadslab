/* ---------- bfloat16 GEMM for a long prompt ---------- */

void rc_bf16_gemm(const uint16_t *w, const float *xt, float *out,
                     int rows, int cols, int tokens)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint16_t *wi = w + (size_t)i * (size_t)cols;
        float acc[1024];
        int limit = tokens < 1024 ? tokens : 1024;
        for (int t = 0; t < limit; ++t) {
            acc[t] = 0.0f;
        }
        for (int k = 0; k < cols; ++k) {
            float wv = bf16_to_f32(wi[k]);
            const float *xk = xt + (size_t)k * (size_t)tokens;
            #pragma omp simd
            for (int t = 0; t < limit; ++t) {
                acc[t] += xk[t] * wv;
            }
        }
        for (int t = 0; t < limit; ++t) {
            out[(size_t)t * (size_t)rows + i] = acc[t];
        }
    }
}
