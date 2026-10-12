/* KQ_Q4X: out (t x rows), inside a parallel region, as kq_nvx_body. */
/* The int16 x of t rows of cols values (inside a parallel region): a scale
 * for each 32 (gemma_quant_group32_i16, as the int16 tile of the prompt). */
static void kq_quant16_body(const float *x, int t, int cols, int16_t *xq, float *xs)
{
    const int nb = cols / 32;
    #pragma omp for schedule(static)
    for (long i = 0; i < (long)t * nb; ++i) {
        const long j = i / nb, g = i % nb;
        xs[j * nb + g] = gemma_quant_group32_i16(x + (size_t)j * cols + 32 * g,
                                                 xq + (size_t)j * cols + 32 * g);
    }
}

/* 1 when the build has the int16 kernels of the prompt (VNNI, AVX2). */
int rc_kq_q16_ok(void)
{
#ifdef KQ_Q4X_Q16
    return 1;
#else
    return 0;
#endif
}

/* KQ_Q4X on t tokens of int16 x (kq_quant16_body), inside a parallel region. */
static void kq_linear16_body(const uint8_t *w, int rows, int cols, const int16_t *xq,
                             const float *xs, int t, float *out)
{
#ifdef KQ_Q4X_Q16
    kq_q4x_gemm16(w, rows, cols, xq, xs, t, out);
#else
    (void)w; (void)rows; (void)cols; (void)xq; (void)xs; (void)t; (void)out;
#endif
}

/* The int16 x and the product, for a check or a benchmark (bench_q4x_gemm.py). */
void rc_kq_linear16(const uint8_t *w, int rows, int cols, const float *x, int16_t *xq, float *xs,
                 int t, float *out)
{
    #pragma omp parallel
    {
        kq_quant16_body(x, t, cols, xq, xs);
        kq_linear16_body(w, rows, cols, xq, xs, t, out);
    }
}

/* RECLANK_Q4X_GEMM=0 keeps kq_q4x_rows for a group of tokens (a check). */
static int getenv_gemm_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_Q4X_GEMM");
        on = !(v && v[0] == '0');
    }
    return on;
}

static void kq_q4x_body(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        int t, float *out)
{
    int32_t *xn = NULL;
    size_t gb = 16 * kq_row_bytes(KQ_Q4X, cols);
    #pragma omp single copyprivate(xn)
    xn = (int32_t *)malloc((size_t)t * (cols / 16) * 4);
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        kq_q4x_xsum(xq + (size_t)j * cols, 1, cols, xn + (size_t)j * (cols / 16));
    }
#ifdef KQ_Q4X_GEMM
    if (t >= 4 && getenv_gemm_on()) {
        kq_q4x_gemm(w, rows, cols, xq, xs, xn, t, out);
        #pragma omp barrier
        #pragma omp single
        free(xn);
        return;
    }
#endif
    for (int j0 = 0; j0 < t; j0 += MA_TB) {
        int nt = t - j0 < MA_TB ? t - j0 : MA_TB;
        #pragma omp for schedule(static) nowait
        for (int g = 0; g < rows / 16; ++g) {
            kq_q4x_rows(w + (size_t)g * gb, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * (cols / 32),
                        xn + (size_t)j0 * (cols / 16), nt, out + (size_t)j0 * rows + 16 * g,
                        (size_t)rows);
        }
    }
    #pragma omp barrier
    #pragma omp single
    free(xn);
}
