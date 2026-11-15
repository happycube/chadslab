/* ---------- GP_KQ_LINEAR and GP_KQ_MULTI of a step and of a small group on pr_fmt.cuh's formats ----------
 * The products of t <= MT_MAX tokens (a step, the verify groups of MTP and of a drafter) on the int8 x of
 * k_kq_quant_x (xs a scale for each 32, xsum that scale times the sum of the 32 int8 values). A warp takes R
 * rows; lane l the parts l, l + 32, ... (32 values each) of each: it decodes a part once (pr_mv_part: 8 words of
 * int8 values, the scales of its two halves, the factor of xsum) for all the tokens, then dp4a of each
 * token's x, whose loads serve the R rows. Each token adds its terms in the order of a token alone, so a
 * token of a verify group gets the values of a step (the same bits for any NT and R).
 *
 * A record of Qwen3.8-27B mixes the types of pr_fmt.cuh with Q4_K, Q5_K, Q6_K, Q8_R (the Q8_0 rows) and
 * float32 rows (Q2_K's); all have kernels here (kq_pr_mv_type), one launch for each matrix, each kernel of
 * one type (the switch of all of them in one kernel took 111 registers for a token, 128 for a group).
 * Under a power cap the instructions of a value set the rate (a P6000 at 150 W streams 390 GB/s with up to
 * ~2 integer instructions a byte, 311 GB/s with 4). The rates of each type: scratch bench mvb.cu, real rows
 * of the 27B (docs/QWEN27_P6000.md). */
#define PR_WARPS 8

#include "pr_mv.cuh"

/* A matrix of type F (rows of rb bytes, cols values) on t tokens, NT at a time, R rows a warp: out[j * rows +
 * row]. x: the float rows (KQ_F32's), xq / xs / xsum the int8 x. The IQ3 types' grids (pr_tables_fill) in
 * shared memory, made by each block. A launch for each matrix of a record (the kernel of one type keeps
 * the registers of that type: the switch of all of them took 111 for a token, 2 blocks an SM). */
template <int F, int NT, int R>
__global__ void __launch_bounds__(32 * PR_WARPS) k_pr_mv(const uint8_t *W, size_t rb, int rows, int cols, int t,
                                                         const float *x, const int8_t *xq, const float *xs,
                                                         const float *xsum, float *out)
{
    PDL_START();
    constexpr int TAB = F == KQ_PR_IQ3X ? PR_IQ3X_TAB : F == KQ_PR_IQ3S ? PR_IQ3S_TAB : 1;
    __shared__ uint32_t tab[TAB];
    if (F == KQ_PR_IQ3X || F == KQ_PR_IQ3S) {
        pr_tables_fill(F == KQ_PR_IQ3X ? tab : NULL, F == KQ_PR_IQ3S ? tab : NULL);
        __syncthreads();
    }
    const int r0 = (blockIdx.x * PR_WARPS + threadIdx.x / 32) * R, np = cols / 32;
    if (r0 >= rows) {
        return;
    }
    for (int j0 = 0; j0 < t; j0 += NT) {
        float res[R * NT];
        if (F == KQ_F32) {
            pr_rows_f32<NT>((const float *)(W + (size_t)r0 * rb), cols, x + (size_t)j0 * cols, res);
        } else {
            if (R == 1 || r0 + R <= rows) {
                pr_rows_r<F, NT, R>(W + (size_t)r0 * rb, rb, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * np,
                                    xsum + (size_t)j0 * np, F == KQ_PR_IQ3X ? tab : NULL, F == KQ_PR_IQ3S ? tab : NULL, res);
            } else {
                /* the last rows (rows not a multiple of R): one at a time */
                for (int k = 0; k < R; ++k) {
                    float r1[NT];
                    if (r0 + k < rows) {
                        pr_rows_r<F, NT, 1>(W + (size_t)(r0 + k) * rb, rb, cols, xq + (size_t)j0 * cols,
                                            xs + (size_t)j0 * np, xsum + (size_t)j0 * np, F == KQ_PR_IQ3X ? tab : NULL,
                                            F == KQ_PR_IQ3S ? tab : NULL, r1);
                        #pragma unroll
                        for (int j = 0; j < NT; ++j) {
                            res[k * NT + j] = r1[j];
                        }
                    }
                }
            }
        }
        if (threadIdx.x % 32 == 0) {
            #pragma unroll
            for (int k = 0; k < R; ++k) {
                #pragma unroll
                for (int j = 0; j < NT; ++j) {
                    if (j0 + j < t && r0 + k < rows) {
                        out[(size_t)(j0 + j) * rows + r0 + k] = res[k * NT + j];
                    }
                }
            }
        }
    }
}

/* RECLANK_GPU_PR_ALL=1: k_pr_mv for the records of the other types it has (Q4_K, Q5_K, Q6_K, Q8_R) too, not
 * only those with a type of pr_fmt.cuh (QwenGPU sets it for a model with such types: the 27B's Q4_K records
 * took the generic kernels, 11 ms a step on a P6000). Read at the first launch. */
static int pr_all_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RECLANK_GPU_PR_ALL");
        v = s ? atoi(s) != 0 : 0;
    }
    return v;
}

/* RECLANK_GPU_PR_MV=0: no k_pr_mv (the generic kernels on float x for pr_fmt.cuh's types: slow, the check). */
static int pr_mv_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RECLANK_GPU_PR_MV");
        v = s ? atoi(s) != 0 : 1;
    }
    return v;
}

/* RECLANK_GPU_PR_R: the rows a warp of a group of 4 or more tokens (4: the 27B's verify groups of 8 on a
 * P6000, 96 -> 163 GB/s over its types; each token's x read once for the R rows). A step: 1. */
static int pr_r_groups(void)
{
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RECLANK_GPU_PR_R");
        v = s ? atoi(s) : 4;
        if (v != 1 && v != 2 && v != 4) {
            v = 4;
        }
    }
    return v;
}

template <int F, int NT, int R>
static void pr_mv1(const uint8_t *w, int rows, int cols, int t, const float *x, const int8_t *xq, const float *xs,
                   const float *xsum, float *out, cudaStream_t s)
{
    const unsigned blocks = (unsigned)((rows + PR_WARPS * R - 1) / (PR_WARPS * R));
    k_pr_mv<F, NT, R><<<blocks, 32 * PR_WARPS, 0, s>>>(w, kq_row_bytes(F, cols), rows, cols, (int)t, x, xq, xs, xsum,
                                                         out);
}

template <int F>
static void pr_mv_nt(const uint8_t *w, int rows, int cols, int64_t t, const float *x, const int8_t *xq,
                     const float *xs, const float *xsum, float *out, cudaStream_t s)
{
    const int r = F == KQ_F32 ? 1 : pr_r_groups();
#define PR_NT(N, RR) pr_mv1<F, N, RR>(w, rows, cols, (int)t, x, xq, xs, xsum, out, s)
#define PR_NTR(N) do { if (r == 4) PR_NT(N, 4); else if (r == 2) PR_NT(N, 2); else PR_NT(N, 1); } while (0)
    switch (t <= 1 ? 1 : t >= 8 ? 8 : (int)t) {
    case 1: PR_NT(1, 1); break;
    case 2: PR_NT(2, 1); break;
    case 3: PR_NT(3, 1); break;
    case 4: PR_NTR(4); break;
    case 5: PR_NTR(5); break;
    case 6: PR_NTR(6); break;
    case 7: PR_NTR(7); break;
    default: PR_NTR(8); break;
    }
#undef PR_NTR
#undef PR_NT
}

/* k_pr_mv for one matrix (w: type rows x cols) of a record on t tokens (x float, and quantized: xq, then xs and
 * xsum after kqx_n bytes). */
static void pr_mv_launch(const uint8_t *w, int type, int rows, int cols, int64_t t, const float *x, const int8_t *xq,
                         size_t kqx_n, float *out, cudaStream_t s)
{
    const float *xs = (const float *)(xq + kqx_n), *xsum = xs + kqx_n / 32;
    switch (type) {
    case KQ_PR_IQ4X: pr_mv_nt<KQ_PR_IQ4X>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_PR_IQ3S: pr_mv_nt<KQ_PR_IQ3S>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_PR_IQ3X: pr_mv_nt<KQ_PR_IQ3X>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_PR_Q3K: pr_mv_nt<KQ_PR_Q3K>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_PR_IQ2S: pr_mv_nt<KQ_PR_IQ2S>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_PR_IQ2X: pr_mv_nt<KQ_PR_IQ2X>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_PR_Q6K: pr_mv_nt<KQ_PR_Q6K>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_Q4_K: pr_mv_nt<KQ_Q4_K>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_Q5_K: pr_mv_nt<KQ_Q5_K>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_Q6_K: pr_mv_nt<KQ_Q6_K>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    case KQ_Q8_R: pr_mv_nt<KQ_Q8_R>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    default: pr_mv_nt<KQ_F32>(w, rows, cols, t, x, xq, xs, xsum, out, s); break;
    }
}

/* RECLANK_GPU_PR_GEMM=0: the prompt's groups of pr_fmt.cuh's types on the float tiles of k_kq_gemm (kq_pr_dequant8:
 * the check), not k_pr_gemm */
static int pr_gemm_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RECLANK_GPU_PR_GEMM");
        v = s ? atoi(s) != 0 : 1;
    }
    return v;
}

/* k_pr_gemm for a matrix (w: type rows x cols) on a group of t tokens (the int8 x at xq: xs and xsum after
 * kqx_n bytes) */
static void pr_gemm_launch(const uint8_t *w, int type, int rows, int cols, int64_t t, const int8_t *xq, size_t kqx_n,
                           float *out, cudaStream_t s)
{
    const float *xs = (const float *)(xq + kqx_n), *xsum = xs + kqx_n / 32;
    const dim3 gr((unsigned)((t + PR_GBM - 1) / PR_GBM), (unsigned)((rows + PR_GBN - 1) / PR_GBN));
    const size_t rb = kq_row_bytes(type, cols);
#define PR_G(F) k_pr_gemm<F><<<gr, 256, 0, s>>>(w, rb, rows, cols, (int)t, xq, xs, xsum, out)
    switch (type) {
    case KQ_PR_IQ4X: PR_G(KQ_PR_IQ4X); break;
    case KQ_PR_IQ3S: PR_G(KQ_PR_IQ3S); break;
    case KQ_PR_IQ3X: PR_G(KQ_PR_IQ3X); break;
    case KQ_PR_Q3K: PR_G(KQ_PR_Q3K); break;
    case KQ_PR_IQ2S: PR_G(KQ_PR_IQ2S); break;
    case KQ_PR_IQ2X: PR_G(KQ_PR_IQ2X); break;
    case KQ_Q4_K: PR_G(KQ_Q4_K); break;
    case KQ_Q5_K: PR_G(KQ_Q5_K); break;
    case KQ_Q6_K: PR_G(KQ_Q6_K); break;
    case KQ_PR_Q6K: PR_G(KQ_PR_Q6K); break;
    default: PR_G(KQ_Q8_R); break;
    }
#undef PR_G
}
