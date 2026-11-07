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
                                                  float *out, float sc,
        const gp_rec *rp, const int64_t *re)
{
    GG_RP_KQ(rp, re, w, x, out);
    GG_RP_SC(rp, re, sc);
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
                                                    float *out, float sc,
        const gp_rec *rp, const int64_t *re)
{
    GG_RP_KQ(rp, re, w, x, out);
    GG_RP_SC(rp, re, sc);
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
                                                float *out, float osc,
        const gp_rec *rp, const int64_t *re)
{
    GG_RP_KQ(rp, re, w, x, out);
    GG_RP_SC(rp, re, osc);
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

/* k_mxf8_tc's place without the tensor cores (Pascal: Super's dense rows of a prompt group; k_mxf8_gemv took 8 tokens
 * a pass, every row again for each 8): a block for 64 rows x 64 tokens, a thread 4 x 4 of them in float32; each 32
 * columns the rows' E4M3 values (a table) and the tokens' bfloat16 x into shared memory (column-major: a thread's 4
 * rows and 4 tokens one 16-byte load each), the next 32's loaded into registers during these; the sums of each 32
 * then times the row's scale, as k_mxf8_tc's (the order of the 32 terms its own: not the same bits). */
#define MXF_BM 64
#define MXF_BN 64
__global__ void __launch_bounds__(256) k_mxf8_gemm_f(const uint8_t *w, int rows, int cols, const float *x, int t,
                                                    float *out, float osc,
        const gp_rec *rp, const int64_t *re)
{
    GG_RP_KQ(rp, re, w, x, out);
    GG_RP_SC(rp, re, osc);
    __shared__ float tab[256];
    __shared__ __align__(16) float ws[32][MXF_BM + 4];
    __shared__ __align__(16) float xs[32][MXF_BN + 4];
    __shared__ float sc[MXF_BM];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        tab[i] = e4m3f((uint32_t)i);
    }
    const size_t rb = kq_row_bytes(KQ_MXF8, cols);
    const int r0 = blockIdx.x * MXF_BM, n0 = blockIdx.y * MXF_BN, tid = threadIdx.x, nb = cols / 32;
    const int tx = tid % 16, ty = tid / 16;                 /* tokens 4 tx .. + 3, rows 4 ty .. + 3 */
    const int lr = tid / 4, lp = tid % 4;                   /* the loader: a row's (a token's) 8 values lp */
    const bool rok = r0 + lr < rows, tok = n0 + lr < t;
    uint2 rq = make_uint2(0, 0);
    float4 rx0 = make_float4(0.f, 0.f, 0.f, 0.f), rx1 = rx0;
    uint8_t rsc = 0;
    auto load = [&](int b) {
        if (rok) {
            rq = *(const uint2 *)(w + (size_t)(r0 + lr) * rb + 32 * b + 8 * lp);
            if (lp == 0) {
                rsc = w[(size_t)(r0 + lr) * rb + cols + b];
            }
        }
        if (tok) {
            const float *xp = x + (size_t)(n0 + lr) * cols + 32 * b + 8 * lp;
            rx0 = *(const float4 *)xp;
            rx1 = *(const float4 *)(xp + 4);
        }
    };
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
            acc[i][j] = 0.f;
    load(0);
    for (int b = 0; b < nb; ++b) {
        __syncthreads();
        {
            const uint32_t wv[2] = {rq.x, rq.y};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                ws[8 * lp + i][lr] = rok ? tab[(wv[i / 4] >> (8 * (i % 4))) & 255] : 0.f;
            }
            const float xv[8] = {rx0.x, rx0.y, rx0.z, rx0.w, rx1.x, rx1.y, rx1.z, rx1.w};
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                xs[8 * lp + i][lr] = tok ? rc_bf16r_k(xv[i]) : 0.f;
            }
            if (lp == 0) {
                sc[lr] = rok ? e8m0f(rsc) : 0.f;
            }
        }
        __syncthreads();
        if (b + 1 < nb) {
            load(b + 1);
        }
        float part[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
                part[i][j] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 32; ++kk) {
            const float4 a = *(const float4 *)&ws[kk][4 * ty], bx = *(const float4 *)&xs[kk][4 * tx];
            const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {bx.x, bx.y, bx.z, bx.w};
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int j = 0; j < 4; ++j)
                    part[i][j] = fmaf(av[i], bv[j], part[i][j]);
        }
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float s = sc[4 * ty + i];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = fmaf(part[i][j], s, acc[i][j]);
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int rr = r0 + 4 * ty + i;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int tt = n0 + 4 * tx + j;
            if (rr < rows && tt < t) {
                out[(size_t)tt * rows + rr] = rc_bf16r_k(acc[i][j]) * osc;
            }
        }
    }
}

/* k_mxf8_gemm_f's place on Pascal with dp4a (RECLANK_GPU_MXF8_G8, default on: Super's dense rows of a prompt group,
 * slight differences): each 32 of a row's E4M3 values (as floats) and of a token's x to int8 by their largest |value|
 * (127 steps) in the loader, the next step's words in registers while the step at hand is summed (as nm.cu
 * k_nm_gemm8d); a part's int sum (32 terms) to float by the magic number, times the row's scale (E8M0 times its int8
 * step) and x's step into the sum; the output rounded to bfloat16 times osc. Against k_mxf8_gemm_f (random rows,
 * Gaussian x): 0.8% relative RMS, as int8 x of the other int8 products. On the P6000, 4096 tokens: the Mamba in
 * (18560 x 4096) 133 -> 77 ms (G8R 4, NP 2), out (4096 x 8192) 59 -> 28.5 (G8R 8, NP 1), attention q / o 29.7 -> 14.7. */
#define MX8_SW(r, w) (NP >= 4 ? (((((w) >> 2) ^ ((r) & 7)) << 2) | ((w) & 3)) \
                              : (((((w) >> 2) ^ ((r) & (2 * NP - 1))) << 2) | ((w) & 3)))
__device__ __forceinline__ float mx8_i2f(int c) { return __int_as_float(c + 0x4B400000) - 12582912.f; }
/* 8 floats to int8 by the largest |value| of the 4 lanes' 32 (lanes 4 apart in tid: a quad), into two words; its step */
__device__ __forceinline__ float mx8_q8(const float *v, int &w0, int &w1)
{
    float m = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) m = fmaxf(m, fabsf(v[i]));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
    const float iv = m > 0.f ? 127.f / m : 0.f;
    uint32_t a = 0, b = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        a |= (uint32_t)(__float2int_rn(v[i] * iv) & 255) << (8 * i);
        b |= (uint32_t)(__float2int_rn(v[4 + i] * iv) & 255) << (8 * i);
    }
    w0 = (int)a; w1 = (int)b;
    return m * (1.f / 127.f);
}
template <int G8R, int G8T, int NP>
__global__ void __launch_bounds__(256, 2) k_mxf8_g8(const uint8_t *w, int rows, int cols, const float *x, int t, float *out,
                                                   float osc, const gp_rec *rp, const int64_t *re)
{
    GG_RP_KQ(rp, re, w, x, out);
    GG_RP_SC(rp, re, osc);
    constexpr int BM = 16 * G8R, BN = 16 * G8T, QW = BM / 64, QX = BN / 64, BK = 32 * NP;
    __shared__ float tab[256];
    __shared__ __align__(16) int ws[BM][BK / 4];
    __shared__ __align__(16) int xs[BN][BK / 4];
    __shared__ float wsc[BM][NP], xsc[BN][NP];
    const int tid = threadIdx.x, r0 = blockIdx.x * BM, t0 = blockIdx.y * BN, l = tid % 4, ty = tid / 16, tx = tid % 16;
    for (int i = tid; i < 256; i += blockDim.x) tab[i] = e4m3f((uint32_t)i);
    const size_t rb = kq_row_bytes(KQ_MXF8, cols);
    uint2 rq[QW][NP];
    uint32_t rsb[QW][NP];
    float4 rx[QX][NP][2];
    auto load = [&](int k0) {
#pragma unroll
        for (int qq = 0; qq < QW; ++qq) {
            const int row = r0 + tid / 4 + 64 * qq;
#pragma unroll
            for (int p = 0; p < NP; ++p) {
                const int kk = k0 + 32 * p;
                rq[qq][p] = make_uint2(0, 0); rsb[qq][p] = 0;
                if (row < rows && kk < cols) {
                    const uint8_t *wr = w + (size_t)row * rb;
                    rq[qq][p] = *(const uint2 *)(wr + kk + 8 * l);
                    rsb[qq][p] = wr[cols + kk / 32];
                }
            }
        }
#pragma unroll
        for (int qq = 0; qq < QX; ++qq) {
            const int tok = t0 + tid / 4 + 64 * qq;
#pragma unroll
            for (int p = 0; p < NP; ++p) {
                const int kk = k0 + 32 * p;
                rx[qq][p][0] = rx[qq][p][1] = make_float4(0.f, 0.f, 0.f, 0.f);
                if (tok < t && kk < cols) {
                    const float *xp = x + (size_t)tok * cols + kk + 8 * l;
                    rx[qq][p][0] = *(const float4 *)xp;
                    rx[qq][p][1] = *(const float4 *)(xp + 4);
                }
            }
        }
    };
    float acc[G8R][G8T];
#pragma unroll
    for (int i = 0; i < G8R; ++i)
#pragma unroll
        for (int j = 0; j < G8T; ++j) acc[i][j] = 0.f;
    load(0);
    __syncthreads();
    for (int k0 = 0; k0 < cols; k0 += BK) {
#pragma unroll
        for (int qq = 0; qq < QW; ++qq) {
            const int rr = tid / 4 + 64 * qq;
#pragma unroll
            for (int p = 0; p < NP; ++p) {
                float v[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    v[i] = tab[(rq[qq][p].x >> (8 * i)) & 255];
                    v[4 + i] = tab[(rq[qq][p].y >> (8 * i)) & 255];
                }
                int w0, w1;
                const float st = mx8_q8(v, w0, w1);
                ws[rr][MX8_SW(rr, 8 * p + 2 * l)] = w0;
                ws[rr][MX8_SW(rr, 8 * p + 2 * l + 1)] = w1;
                if (l == 0) wsc[rr][p] = st * e8m0f((uint8_t)rsb[qq][p]);
            }
        }
#pragma unroll
        for (int qq = 0; qq < QX; ++qq) {
            const int tt = tid / 4 + 64 * qq;
#pragma unroll
            for (int p = 0; p < NP; ++p) {
                const float v[8] = {rx[qq][p][0].x, rx[qq][p][0].y, rx[qq][p][0].z, rx[qq][p][0].w,
                                    rx[qq][p][1].x, rx[qq][p][1].y, rx[qq][p][1].z, rx[qq][p][1].w};
                int w0, w1;
                const float st = mx8_q8(v, w0, w1);
                xs[tt][MX8_SW(tt, 8 * p + 2 * l)] = w0;
                xs[tt][MX8_SW(tt, 8 * p + 2 * l + 1)] = w1;
                if (l == 0) xsc[tt][p] = st;
            }
        }
        __syncthreads();
        if (k0 + BK < cols) load(k0 + BK);
#pragma unroll
        for (int p = 0; p < NP; ++p) {
            if (k0 + 32 * p >= cols) break;
            int4 xa[G8T], xb[G8T];
            float xc[G8T];
#pragma unroll
            for (int j = 0; j < G8T; ++j) {
                const int tt = tx + 16 * j;
                xa[j] = *(const int4 *)&xs[tt][MX8_SW(tt, 8 * p)];
                xb[j] = *(const int4 *)&xs[tt][MX8_SW(tt, 8 * p + 4)];
                xc[j] = xsc[tt][p];
            }
#pragma unroll
            for (int i = 0; i < G8R; ++i) {
                const int rw = ty + 16 * i;
                const int4 a = *(const int4 *)&ws[rw][MX8_SW(rw, 8 * p)];
                const int4 b = *(const int4 *)&ws[rw][MX8_SW(rw, 8 * p + 4)];
                const float sw = wsc[rw][p];
#pragma unroll
                for (int j = 0; j < G8T; ++j) {
                    int c = rc_dp4a(a.x, xa[j].x, 0);
                    c = rc_dp4a(a.y, xa[j].y, c);
                    c = rc_dp4a(a.z, xa[j].z, c);
                    c = rc_dp4a(a.w, xa[j].w, c);
                    c = rc_dp4a(b.x, xb[j].x, c);
                    c = rc_dp4a(b.y, xb[j].y, c);
                    c = rc_dp4a(b.z, xb[j].z, c);
                    c = rc_dp4a(b.w, xb[j].w, c);
                    acc[i][j] = fmaf(mx8_i2f(c), sw * xc[j], acc[i][j]);
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < G8R; ++i) {
        const int row = r0 + ty + 16 * i;
#pragma unroll
        for (int j = 0; j < G8T; ++j) {
            const int tok = t0 + tx + 16 * j;
            if (row < rows && tok < t) out[(size_t)tok * rows + row] = rc_bf16r_k(acc[i][j]) * osc;
        }
    }
}

/* RECLANK_GPU_MXF8_G8=0: Pascal's groups on k_mxf8_gemm_f (float32, as before) */
static int mxf8_g8_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_MXF8_G8");
        on = !(v && v[0] == '0');
    }
    return on && gg_caps.dp4a && !gg_caps.tc;
}

/* k_mxf8_gemm_f for the groups of more than 8 tokens without the tensor cores, unless RECLANK_GPU_MXF8_F=0
 * (k_mxf8_gemv: a measure); 1: on the tensor cores' GPUs too (a check against k_mxf8_tc) */
static int mxf8_f_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_MXF8_F");
        on = v ? atoi(v) : 2;
    }
    return on == 1 || (on == 2 && !gg_caps.tc);
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
                              uint16_t *w16, size_t w16_n, uint16_t *xb, size_t xb_n, cudaStream_t s,
                              const gp_rec *rp = NULL, const int64_t *re = NULL)
{
    /* (rp: the kernels below read x, w and out in the record, GG_RP_KQ; the tensor cores' chunks take them) */
    const float *xa = rp != NULL ? NULL : x;
    const uint8_t *wa = rp != NULL ? NULL : w;
    float *oa = rp != NULL ? NULL : out;
    const float sca = rp != NULL ? 0.f : sc;     /* (the kernels read it in the record too) */
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
    if (t > 8 && mxf8_g8_on() && cols % 32 == 0) {
        if (rows <= 8192) {
            k_mxf8_g8<8, 4, 1><<<dim3((unsigned)((rows + 127) / 128), (unsigned)((t + 63) / 64)), 256, 0, s>>>(
                wa, rows, cols, xa, t, oa, sca, rp, re);
        } else {
            k_mxf8_g8<4, 4, 2><<<dim3((unsigned)((rows + 63) / 64), (unsigned)((t + 63) / 64)), 256, 0, s>>>(
                wa, rows, cols, xa, t, oa, sca, rp, re);
        }
        return 0;
    }
    if (t > 8 && mxf8_f_on() && cols % 32 == 0) {
        k_mxf8_gemm_f<<<dim3((unsigned)((rows + MXF_BM - 1) / MXF_BM), (unsigned)((t + MXF_BN - 1) / MXF_BN)), 256, 0, s>>>(
            wa, rows, cols, xa, t, oa, sca, rp, re);
        return 0;
    }
    if (t > 8 && gg_caps.tc) {
        k_mxf8_tc<<<dim3((unsigned)((rows + MXT_BM - 1) / MXT_BM), (unsigned)((t + MXT_BN - 1) / MXT_BN)), 128, 0, s>>>(wa, rows, cols, xa, t,
                                                                                              oa, sca, rp, re);
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
    /* (Pascal's steps: R 1, the P6000's 15.3-15.4 -> 15.6-15.7 tok/s, the same bits; its verify groups keep R 2:
     * k_mxf8_gemv_s's x staged for both rows, a round 156 -> 168 ms at R 1) */
    int R = rows >= 16 * 2 * nsm && (gg_caps.tc || t > 1) ? 2 : 1;
    if (rset == 1 || rset == 2 || rset == 4) {
        R = rset;
    }
    unsigned nblk = (unsigned)((rows + 8 * R - 1) / (8 * R));
#define MXF8_GEMV(RR, NN)                                                                          \
    do {                                                                                           \
        if (NN > 1 && mxf8_stage_on()) {                                                           \
            k_mxf8_gemv_s<RR, NN><<<nblk, 256, 0, s>>>(wa, rows, cols, xa, t, oa, sca, rp, re);          \
        } else {                                                                                   \
            k_mxf8_gemv<RR, NN><<<nblk, 256, 0, s>>>(wa, rows, cols, xa, t, oa, sca, rp, re);            \
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
