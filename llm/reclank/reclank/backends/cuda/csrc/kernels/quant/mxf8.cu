/* KQ_MXF8 (FP4-Local, docs/FP4LOCAL_PLAN.md): the MXFP8 matrices of the checkpoint as they are, a
 * row of cols E4M3 codes, then the E8M0 scale of each 32 (value e4m3 2^(s - 127)), zeros to 16
 * bytes. The products are b12x's a16 mode: x in bfloat16 values (rounded here when it is not),
 * the weights exact, float32 sums, the output rounded to bfloat16.
 *
 * k_mxf8_gemv: a step or a few tokens, as k_fp8b32_gemv (kernels/dsv41/types.cu): a warp for R
 * rows, 4 values a lane a step, each word decoded once for the NT tokens; the scale of each row's
 * block of 32. The operations of a row and token are those of one alone (the same bits).
 *
 * k_mxf8_tc: a group (t > 8) on the tensor cores, mma m16n8k8 tf32 (the E4M3 values and the
 * bfloat16 values of x are exact in tf32: one part each), as k_fp8b32_tc with the scales of the
 * rows; 64 rows x 32 tokens a block. */

__device__ __forceinline__ float4 mxf8_bf16r4(float4 v)
{
    return make_float4(rc_bf16r_k(v.x), rc_bf16r_k(v.y), rc_bf16r_k(v.z), rc_bf16r_k(v.w));
}

template <int R, int NT>
__device__ __forceinline__ void mxf8_step(const uint32_t *q, const float *s, const float *x, size_t xstride, int tn,
                                          float (*acc)[NT])
{
    float wv[R][4];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        fp8x4_f16(q[r], wv[r]);
    }
#pragma unroll
    for (int jj = 0; jj < NT; ++jj) {
        if (NT == 1 || jj < tn) {
            float4 xv = mxf8_bf16r4(*(const float4 *)(x + jj * xstride));
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float p = wv[r][0] * xv.x;
                p = fmaf(wv[r][1], xv.y, p);
                p = fmaf(wv[r][2], xv.z, p);
                p = fmaf(wv[r][3], xv.w, p);
                acc[r][jj] = fmaf(p, s[r], acc[r][jj]);
            }
        }
    }
}

template <int R, int NT>
__global__ void __launch_bounds__(256) k_mxf8_gemv(const uint8_t *w, int rows, int cols, const float *x, int t,
                                                  float *out, float sc)
{
    int row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    if (row0 >= rows) {
        return;
    }
    const size_t rb = kq_row_bytes(KQ_MXF8, cols);
    const uint8_t *wr = w + (size_t)row0 * rb;
    for (int j0 = 0; j0 < t; j0 += NT) {
        int tn = t - j0 < NT ? t - j0 : NT;
        const float *x0 = x + (size_t)j0 * cols;
        float acc[R][NT];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int jj = 0; jj < NT; ++jj) {
                acc[r][jj] = 0.f;
            }
        }
        int c = 4 * lane;
        for (; c + 384 < cols; c += 512) {
            uint32_t q[4][R];
            float s[4][R];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    q[u][r] = *(const uint32_t *)(wr + (size_t)r * rb + c + 128 * u);
                    s[u][r] = e8m0f(wr[(size_t)r * rb + cols + (c + 128 * u) / 32]);
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                mxf8_step<R, NT>(q[u], s[u], x0 + c + 128 * u, (size_t)cols, tn, acc);
            }
        }
        for (; c < cols; c += 128) {
            uint32_t q[R];
            float s[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                q[r] = *(const uint32_t *)(wr + (size_t)r * rb + c);
                s[r] = e8m0f(wr[(size_t)r * rb + cols + c / 32]);
            }
            mxf8_step<R, NT>(q, s, x0 + c, (size_t)cols, tn, acc);
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int jj = 0; jj < NT; ++jj) {
                if (NT == 1 || jj < tn) {
                    float v = acc[r][jj];
                    for (int o = 16; o > 0; o >>= 1) {
                        v += __shfl_xor_sync(RC_FULL_MASK, v, o);
                    }
                    if (lane == 0 && row0 + r < rows) {
                        out[(size_t)(j0 + jj) * rows + row0 + r] = rc_bf16r_k(v) * sc;
                    }
                }
            }
        }
    }
}

/* k_mxf8_gemv for 2-8 tokens (a verify group): each block stages the x of its tokens for 512 columns
 * at a time in shared memory, rounded to bfloat16 once (k_mxf8_gemv reads and rounds x again in each
 * warp: at 4 tokens its time was 2.3 times a step's); each lane's products and sums are those of
 * k_mxf8_gemv in the same order (the same bits). */
template <int R, int NT>
__global__ void __launch_bounds__(256) k_mxf8_gemv_s(const uint8_t *w, int rows, int cols, const float *x, int t,
                                                    float *out, float sc)
{
    __shared__ __align__(16) float xsm[NT][512];
    int row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    const bool live = row0 < rows;
    const size_t rb = kq_row_bytes(KQ_MXF8, cols);
    const uint8_t *wr = w + (size_t)(live ? row0 : 0) * rb;
    for (int j0 = 0; j0 < t; j0 += NT) {
        int tn = t - j0 < NT ? t - j0 : NT;
        const float *x0 = x + (size_t)j0 * cols;
        float acc[R][NT];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int jj = 0; jj < NT; ++jj) {
                acc[r][jj] = 0.f;
            }
        }
        for (int cb = 0; cb < cols; cb += 512) {
            const int span = cols - cb < 512 ? cols - cb : 512;      /* (a multiple of 128) */
            __syncthreads();
            for (int i = threadIdx.x; i < NT * (span / 4); i += 256) {
                int jj = i / (span / 4), c4 = 4 * (i % (span / 4));
                float4 v = jj < tn ? mxf8_bf16r4(*(const float4 *)(x0 + (size_t)jj * cols + cb + c4))
                                   : make_float4(0.f, 0.f, 0.f, 0.f);
                *(float4 *)&xsm[jj][c4] = v;
            }
            __syncthreads();
            if (!live) {
                continue;
            }
            const int nu = span / 128;
            uint32_t q[4][R];
            float s[4][R];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u < nu) {
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        const int c = cb + 4 * lane + 128 * u;
                        q[u][r] = *(const uint32_t *)(wr + (size_t)r * rb + c);
                        s[u][r] = e8m0f(wr[(size_t)r * rb + cols + c / 32]);
                    }
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (u < nu) {
                    float wv[R][4];
#pragma unroll
                    for (int r = 0; r < R; ++r) {
                        fp8x4_f16(q[u][r], wv[r]);
                    }
#pragma unroll
                    for (int jj = 0; jj < NT; ++jj) {
                        if (jj < tn) {
                            float4 xv = *(const float4 *)&xsm[jj][4 * lane + 128 * u];
#pragma unroll
                            for (int r = 0; r < R; ++r) {
                                float p = wv[r][0] * xv.x;
                                p = fmaf(wv[r][1], xv.y, p);
                                p = fmaf(wv[r][2], xv.z, p);
                                p = fmaf(wv[r][3], xv.w, p);
                                acc[r][jj] = fmaf(p, s[u][r], acc[r][jj]);
                            }
                        }
                    }
                }
            }
        }
        if (!live) {
            continue;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int jj = 0; jj < NT; ++jj) {
                if (jj < tn) {
                    float v = acc[r][jj];
                    for (int o = 16; o > 0; o >>= 1) {
                        v += __shfl_xor_sync(RC_FULL_MASK, v, o);
                    }
                    if (lane == 0 && row0 + r < rows) {
                        out[(size_t)(j0 + jj) * rows + row0 + r] = rc_bf16r_k(v) * sc;
                    }
                }
            }
        }
    }
}

#define MXT_BM 64
#define MXT_BN 32
__global__ void __launch_bounds__(128) k_mxf8_tc(const uint8_t *w, int rows, int cols, const float *x, int t,
                                                float *out, float osc)
{
    __shared__ float tab[256];
    __shared__ float ws[MXT_BM][33];
    __shared__ float xs[MXT_BN][33];
    __shared__ float sc[MXT_BM];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        tab[i] = e4m3f((uint32_t)i);
    }
    const size_t rb = kq_row_bytes(KQ_MXF8, cols);
    int r0 = blockIdx.x * MXT_BM, n0 = blockIdx.y * MXT_BN;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    int nb = cols / 32;
    float acc[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[nt][i] = 0.f;
        }
    }
    for (int b = 0; b < nb; ++b) {
        __syncthreads();
        {
            int rr = threadIdx.x / 2, half = threadIdx.x % 2;
            uint4 q = make_uint4(0, 0, 0, 0);
            if (r0 + rr < rows) {
                q = *(const uint4 *)(w + (size_t)(r0 + rr) * rb + 32 * b + 16 * half);
                if (half == 0) {
                    sc[rr] = e8m0f(w[(size_t)(r0 + rr) * rb + cols + b]);
                }
            } else if (half == 0) {
                sc[rr] = 0.f;
            }
            uint32_t wv[4] = {q.x, q.y, q.z, q.w};
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                ws[rr][16 * half + i] = tab[(wv[i / 4] >> (8 * (i % 4))) & 255];
            }
        }
        for (int i = threadIdx.x; i < MXT_BN * 32; i += blockDim.x) {
            int tk = i / 32, kk = i % 32;
            xs[tk][kk] = n0 + tk < t ? rc_bf16r_k(x[(size_t)(n0 + tk) * cols + 32 * b + kk]) : 0.f;
        }
        __syncthreads();
        float part[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                part[nt][i] = 0.f;
            }
        }
#pragma unroll
        for (int ks = 0; ks < 32; ks += 8) {
            int rw = 16 * warp + g;
            uint32_t a[4] = {__float_as_uint(ws[rw][ks + c]), __float_as_uint(ws[rw + 8][ks + c]),
                             __float_as_uint(ws[rw][ks + c + 4]), __float_as_uint(ws[rw + 8][ks + c + 4])};
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                uint32_t bx[2] = {__float_as_uint(xs[8 * nt + g][ks + c]), __float_as_uint(xs[8 * nt + g][ks + c + 4])};
                mma1688_tf32(part[nt], a, bx);
            }
        }
        float s0 = sc[16 * warp + g], s8 = sc[16 * warp + g + 8];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[nt][i] = fmaf(part[nt][i], i >= 2 ? s8 : s0, acc[nt][i]);
            }
        }
    }
    int row = r0 + 16 * warp + g;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        int tk = n0 + 8 * nt + 2 * c;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            int rr = row + (i >= 2 ? 8 : 0), tt = tk + (i & 1);
            if (rr < rows && tt < t) {
                out[(size_t)tt * rows + rr] = rc_bf16r_k(acc[nt][i]) * osc;
            }
        }
    }
}

/* n rows of a KQ_MXF8 matrix as their bfloat16 values (exact) into dst (n x cols): 16 values a
 * thread. */
__global__ void k_mxf8_to_bf16(const uint8_t *w, int n, int cols, uint16_t *dst)
{
    PDL_START();
    int per = cols / 16;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * per) {
        return;
    }
    int64_t row = i / per;
    int c = (int)(i % per) * 16;
    const uint8_t *wr = w + (size_t)row * kq_row_bytes(KQ_MXF8, cols);
    uint4 q = *(const uint4 *)(wr + c);
    uint32_t qq[4] = {q.x, q.y, q.z, q.w};
    float s = e8m0f(wr[cols + c / 32]);
    uint32_t o[8];
#pragma unroll
    for (int u = 0; u < 4; ++u) {
        float v[4];
        fp8x4_f16(qq[u], v);
        o[2 * u] = (__float_as_uint(v[0] * s) >> 16) | (__float_as_uint(v[1] * s) & 0xFFFF0000u);
        o[2 * u + 1] = (__float_as_uint(v[2] * s) >> 16) | (__float_as_uint(v[3] * s) & 0xFFFF0000u);
    }
    uint4 *d = (uint4 *)(dst + (size_t)row * cols + c);
    d[0] = make_uint4(o[0], o[1], o[2], o[3]);
    d[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

/* out (n values) rounded to bfloat16 in place, times sc */
__global__ void k_round_bf16_n(float *out, size_t n, float sc)
{
    PDL_START();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = rc_bf16r_k(out[i]) * sc;
    }
}

/* GP_KQ_LINEAR of a KQ_MXF8 matrix: x (operand 3), w (4), rows (6), cols (7), t (8), out (9). A
 * large group (t > MT_MAX): chunks of rows as bfloat16 in w16 (exact) and the bfloat16 tensor cores
 * of gg_gemm_bf16 (x to bfloat16 in xb), the output rounded; else the kernels above. */
/* RECLANK_GPU_MXF8_STAGE=0: the groups of 2-8 tokens on k_mxf8_gemv (x read by each warp) */
/* RECLANK_GPU_MXF8_X1=0: the two planes of x in a group's product (its low plane zeros: a measure) */
static int mxf8_x1_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_MXF8_X1");
        on = !(v && v[0] == '0');
    }
    return on;
}

static int mxf8_stage_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_MXF8_STAGE");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* sc: a factor of every output after its rounding (Super's FP8 tensor scale: GP_MUL's product, the same bits;
 * 1 for FP4-Local's) */
static int mxf8_linear_launch(const float *x, const uint8_t *w, int rows, int cols, int t, float *out, float sc,
                              uint16_t *w16, size_t w16_n, uint16_t *xb, size_t xb_n, cudaStream_t s)
{
    if (cols % 128 != 0) {
        return 1;
    }
    if (t > MT_MAX && w16 != NULL && bf16_tc_on() && (size_t)cols <= w16_n &&
        (size_t)t * cols * (bf16_tc_on() != 8 ? 2 : 1) <= xb_n) {
        /* x in bfloat16 values (vLLM's activations; gg_gemm_bf16's low plane then 0) */
        size_t nx = (size_t)t * cols;
        k_round_bf16_n<<<(unsigned)((nx + 255) / 256), 256, 0, s>>>((float *)x, nx, 1.f);
        int step = (int)(w16_n / (size_t)cols);
        step = step >= T2N ? step / T2N * T2N : step;
        size_t rb = kq_row_bytes(KQ_MXF8, cols);
        for (int r0 = 0; r0 < rows; r0 += step) {
            int n = rows - r0 < step ? rows - r0 : step;
            k_mxf8_to_bf16<<<(unsigned)(((int64_t)n * (cols / 16) + 255) / 256), 256, 0, s>>>(w + (size_t)r0 * rb, n,
                                                                                            cols, w16);
            /* (x rounded to bfloat16 above: one plane; its rows converted once, at the first chunk) */
            gg_gemm_bf16(x, (size_t)cols, NULL, w16, out + r0, (size_t)rows, NULL, t, n, cols, xb, s, 0, mxf8_x1_on(),
                         r0 > 0 && mxf8_x1_on());
        }
        size_t nv = (size_t)t * rows;
        k_round_bf16_n<<<(unsigned)((nv + 255) / 256), 256, 0, s>>>(out, nv, sc);
        return 0;
    }
    if (t > 8 && gg_caps.tc) {
        k_mxf8_tc<<<dim3((unsigned)((rows + MXT_BM - 1) / MXT_BM), (unsigned)((t + MXT_BN - 1) / MXT_BN)), 128, 0, s>>>(w, rows, cols, x, t,
                                                                                              out, sc);
        return 0;
    }
    /* R rows a warp: 2 where each SM still gets 2 blocks (a 3090, a step: Super's in_proj 18560 x 4096
     * 114 -> 95 us, out_proj 4096 x 8192 67 -> 45 us at R 2; R 4 left out_proj 128 blocks on 82 SMs, R 1
     * no faster); a row's sums do not depend on R (the same bits). RECLANK_GPU_MXF8_R: R (a measure). */
    static int rset = -1;
    if (rset < 0) {
        const char *v = getenv("RECLANK_GPU_MXF8_R");
        rset = v ? atoi(v) : 0;
    }
    const int nsm = gg_caps.sm_count > 0 ? gg_caps.sm_count : 32;
    int R = rows >= 16 * 2 * nsm ? 2 : 1;
    if (rset == 1 || rset == 2 || rset == 4) {
        R = rset;
    }
    unsigned nblk = (unsigned)((rows + 8 * R - 1) / (8 * R));
#define MXF8_GEMV(RR, NN)                                                                          \
    do {                                                                                           \
        if (NN > 1 && mxf8_stage_on()) {                                                           \
            k_mxf8_gemv_s<RR, NN><<<nblk, 256, 0, s>>>(w, rows, cols, x, t, out, sc);                   \
        } else {                                                                                   \
            k_mxf8_gemv<RR, NN><<<nblk, 256, 0, s>>>(w, rows, cols, x, t, out, sc);                     \
        }                                                                                          \
    } while (0)
#define MXF8_GEMV_T(NN)                                                                            \
    do {                                                                                           \
        if (R == 4) {                                                                              \
            MXF8_GEMV(4, NN);                                                                      \
        } else if (R == 2) {                                                                       \
            MXF8_GEMV(2, NN);                                                                      \
        } else {                                                                                   \
            MXF8_GEMV(1, NN);                                                                      \
        }                                                                                          \
    } while (0)
    if (t == 1) {
        MXF8_GEMV_T(1);
    } else if (t == 2) {
        MXF8_GEMV_T(2);
    } else if (t <= 4) {
        MXF8_GEMV_T(4);
    } else {
        MXF8_GEMV_T(8);
    }
#undef MXF8_GEMV_T
#undef MXF8_GEMV
    return 0;
}
