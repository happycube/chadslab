/* ---------- bfloat16 rows for a small group (an MTP verify group) ----------
 * kq_row_part of KQ_BF16 for up to NT tokens: the lanes take the columns of
 * kq_row_part (8 lane + 256 part, step 256 nparts), and each chunk of 8
 * weights is loaded and converted once for all the tokens, not once for
 * each token. Each token adds its terms in the order of one token, and the
 * sums of the lanes in the same order: res[j] has the bits of kq_row_part
 * of that token (a verify group keeps the values of the steps). On the
 * matrices of Qwen3.8 (bf16, RECLANK_DENSE=bf16) 4 tokens took about 1.8
 * times the time of one; RECLANK_GPU_BF16_NT=0 keeps the loop over the
 * tokens. */
template <int NT>
__device__ __forceinline__ void kq_row_bf16_nt(int type, const uint8_t *w, const float *x, int cols,
                                               int t, int part, int nparts, float *res)
{
    int lane = threadIdx.x % 32;
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    const uint16_t *h = (const uint16_t *)w;
    /* KQ_BF12: the order of kq_row_part (kq_bf12_row) */
    if (type == KQ_BF12) {
        kq_bf12_row<NT>(w, x, cols, t, part, nparts, sum);
    }
    for (int c = 8 * lane + 256 * part; type == KQ_BF16 && c < cols; c += 256 * nparts) {
        uint4 q = *(const uint4 *)(h + c);
        uint32_t qq[4] = {q.x, q.y, q.z, q.w};
        float wv[8];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            wv[2 * u] = kq_bf16((uint16_t)(qq[u] & 0xffff));
            wv[2 * u + 1] = kq_bf16((uint16_t)(qq[u] >> 16));
        }
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                const float4 *x4 = (const float4 *)(x + (size_t)j * cols + c);
                float4 a = x4[0], b = x4[1];
                float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    sum[j] += wv[2 * u] * xv[2 * u] + wv[2 * u + 1] * xv[2 * u + 1];
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(RC_FULL_MASK, v, o);
        }
        res[j] = v;
    }
}

/* KQ_BF12 rows (n rows of cols values) to bfloat16 rows in dst: a thread
 * for each 16 values. */
__global__ void k_bf12_to_bf16(const uint8_t *w, int n, int cols, uint16_t *dst)
{
    PDL_START();
    int per = cols / 16;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * per) {
        return;
    }
    int64_t row = i / per;
    int c = (int)(i % per) * 16;
    uint32_t b[16];
    kq_bf12_bits<16>(w + (size_t)row * kq_row_bytes(KQ_BF12, cols), cols, c, b);
    uint4 *o = (uint4 *)(dst + (size_t)row * cols + c);
    o[0] = make_uint4(b[0] | b[1] << 16, b[2] | b[3] << 16, b[4] | b[5] << 16, b[6] | b[7] << 16);
    o[1] = make_uint4(b[8] | b[9] << 16, b[10] | b[11] << 16, b[12] | b[13] << 16, b[14] | b[15] << 16);
}

/* As gg_gemm_bf16 for KQ_BF12 rows: k_gemm_bf16_tc<X2, 1> (bf12_fused_on),
 * else chunks of rows (at most w16_n values, a multiple of T2N rows) to
 * bfloat16 rows in the scratch w16 (k_bf12_to_bf16), then gg_gemm_bf16. */
static void gg_gemm_bf12(const float *x, size_t ldx, const int *xrow0, const uint8_t *w, float *out,
                         size_t ldo, const int *orow0, int t, int rows, int cols, uint16_t *xb,
                         uint16_t *w16, size_t w16_n, cudaStream_t s)
{
    if (bf12_fused_on()) {
        gg_gemm_bf16(x, ldx, xrow0, (const uint16_t *)w, out, ldo, orow0, t, rows, cols, xb, s, 1);
        return;
    }
    int step = (int)(w16_n / (size_t)cols);
    step = step >= T2N ? step / T2N * T2N : step;
    size_t rb = kq_row_bytes(KQ_BF12, cols);
    for (int r0 = 0; r0 < rows; r0 += step) {
        int n = rows - r0 < step ? rows - r0 : step;
        k_bf12_to_bf16<<<(unsigned)(((int64_t)n * (cols / 16) + 255) / 256), 256, 0, s>>>(
            w + (size_t)r0 * rb, n, cols, w16);
        gg_gemm_bf16(x, ldx, xrow0, w16, out + r0, ldo, orow0, t, n, cols, xb, s);
    }
}

/* The types of the small-group kernels of bfloat16 rows (kq_row_bf16_nt). */
__host__ __device__ __forceinline__ int kq_bf_type(int type)
{
    return type == KQ_BF16 || type == KQ_BF12;
}

static int bf16_nt_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_BF16_NT");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* GP_KQ_LINEAR of a KQ_BF16 matrix for 2 to NT tokens (as k_kq_linear). */
template <int NT>
__global__ void k_kq_linear_bf16(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), t = DI(8);
    if (row >= rows) {
        return;
    }
    float res[NT];
    kq_row_bf16_nt<NT>(DI(5), DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(DI(5), cols),
                       DP(const float, 3), cols, t, 0, 1, res);
    if (threadIdx.x % 32 == 0) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                DP(float, 9)[(size_t)j * rows + row] = res[j];
            }
        }
    }
}

/* As k_kq_linear_split (KQ_SPLIT warps for each row) for 2 to NT tokens. */
template <int NT>
__global__ void __launch_bounds__(32 * KQ_SPLIT) k_kq_linear_split_bf16(const gp_rec *r,
                                                                        const int64_t *e)
{
    PDL_START();
    __shared__ float red[KQ_SPLIT][NT];
    int row = blockIdx.x, w = threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), t = DI(8);
    float res[NT];
    kq_row_bf16_nt<NT>(DI(5), DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(DI(5), cols),
                       DP(const float, 3), cols, t, w, KQ_SPLIT, res);
    if (threadIdx.x % 32 == 0) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            red[w][j] = res[j];
        }
    }
    __syncthreads();
    if (threadIdx.x < t) {
        float sm = 0.f;
        #pragma unroll
        for (int k = 0; k < KQ_SPLIT; ++k) {
            sm += red[k][threadIdx.x];
        }
        DP(float, 9)[(size_t)threadIdx.x * rows + row] = sm;
    }
}

/* GP_KQ_LINEAR of a KQ_BF12 matrix for 1 to NT tokens (as k_kq_linear,
 * kq_bf12_row: the bits of kq_row for each token). k_kq_linear holds the
 * registers of every type (144: 12 warps an SM); this kernel holds at most
 * 64 for a token (BF12_MINB). One token, the 3090 at its power cap: 10240
 * x 2560 59 us (bfloat16 68, Q8_R 46), 10240 x 320 14 us (23), the head
 * 248320 x 2560 2.5 ms (2.1); 2 to 4 tokens 1.3-1.8 times bfloat16 (the
 * decode is bound by the integer instructions at 500-900 MHz). Now for
 * RECLANK_GPU_BF12_TC=0 and groups of 9 to 16 tokens: k_kq_bf12_tc takes
 * 1 to 8 (the same time for 1 and for 4 tokens, below bfloat16). */
#define BF12_MINB 8         /* blocks of KQ_RPB warps an SM for a token: at most 64 registers */
template <int NT>
__global__ void __launch_bounds__(32 * KQ_RPB, NT == 1 ? BF12_MINB : 1) k_kq_linear_bf12(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), t = DI(8);
    if (row >= rows) {
        return;
    }
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    kq_bf12_row<NT>(DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(KQ_BF12, cols), DP(const float, 3),
                    cols, t, 0, 1, sum);
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(RC_FULL_MASK, v, o);
        }
        if (threadIdx.x % 32 == 0 && j < t) {
            DP(float, 9)[(size_t)j * rows + row] = v;
        }
    }
}
