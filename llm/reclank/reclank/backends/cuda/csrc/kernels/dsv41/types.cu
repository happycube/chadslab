/* DeepSeek V4.1 Flash on the GPU (DEEPSEEK_V41_PLAN.md 3, Phase 3): the products
 * of its two weight types (csrc/kernels/kq/mxx.c of the CPU gives the forms).
 *
 * k_mxx_rows_x2: the rows of a KQ_MXX matrix (MXFP4 in groups of 16 rows) on
 * int16 x (the planes of k_quant_x2, the bits of kq_quant_part16), with the
 * operations of the CPU (kq_mxx_rows2): for each block of 32 columns the
 * exact int32 sum a = 128 sum(hi w) + sum(lo w) (w = 2 e2m1, by dp4a; the
 * same integer whatever the order), p = a 2^(e - 127) (exact), then for
 * each row f = fma(p, xs, f) over the blocks in order, and 0.5 f. The bits
 * of the CPU: an expert gives the same output on either.
 *
 * k_fp8b32_gemv: the rows of an FP8 matrix with E8M0 scales of each 32 x 32
 * block on float32 x (near; the exact mode rounds x first, k_dsv41_act).
 * A warp for each row, 4 values a lane a step; not the bits of the CPU (the
 * dense weights run only on the GPU). groups > 1: the rows in that many equal
 * groups, group g on the columns g cols .. (g + 1) cols - 1 of an x row of
 * groups cols values (the block-diagonal wo_a of the attention).
 */
#define MXX_BB 272
#define MXX_NB 512          /* the most blocks of 32 columns (16384 columns) */

__device__ __forceinline__ float e8m0f(uint8_t e)
{
    return __int_as_float((int)e << 23);       /* e = 0 gives 0 (as the VNNI path of the CPU) */
}

/* The 16 rows of the KQ_MXX group wg on the int16 x of one token (xh, xl,
 * xs), by the KS warps of the block, into out[0..15] (the operations above). */
template <int KS>
__device__ __forceinline__ void mxx_group_x2(const uint8_t *wg, int cols, const int8_t *xhj, const int8_t *xlj,
                                             const float *xsj, float *out)
{
    __shared__ float pb[MXX_NB][16];
    int nb = cols / 32;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, st = lane / 4, q = lane % 4;
    for (int b = warp; b < nb; b += KS) {
        const uint8_t *blk = wg + (size_t)b * MXX_BB;
        uint2 c = *(const uint2 *)(blk + 8 * lane);
        int xv = *(const int *)(xhj + 32 * b + 4 * st);
        int xlv = *(const int *)(xlj + 32 * b + 4 * st);
        int wv[4] = {(int)kt_e2m1x4(c.x), (int)kt_e2m1x4(c.y), (int)kt_e2m1x4(c.x >> 4),
                     (int)kt_e2m1x4(c.y >> 4)};
        int a[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = 128 * __dp4a(wv[i], xv, 0) + __dp4a(wv[i], xlv, 0);
            a[i] += __shfl_xor_sync(RC_FULL_MASK, a[i], 4);
            a[i] += __shfl_xor_sync(RC_FULL_MASK, a[i], 8);
            a[i] += __shfl_xor_sync(RC_FULL_MASK, a[i], 16);
        }
        if (lane < 4) {
            /* rows 2q, 2q + 1 (the low halves of the bytes), 2q + 8, 2q + 9 (the high) */
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = 2 * q + (i & 1) + (i >> 1) * 8;
                pb[b][row] = __fmul_rn((float)a[i], e8m0f(blk[256 + row]));
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float f = 0.f;
        for (int b = 0; b < nb; ++b) {
            f = fmaf(pb[b][threadIdx.x], xsj[b], f);
        }
        out[threadIdx.x] = __fmul_rn(f, 0.5f);
    }
}

template <int KS>
__global__ void __launch_bounds__(32 * KS) k_mxx_rows_x2(const uint8_t *w, int cols, const int8_t *xh,
                                                       const int8_t *xl, const float *xs, float *out, int rows)
{
    int grp = blockIdx.x, j = blockIdx.y, nb = cols / 32;
    mxx_group_x2<KS>(w + (size_t)grp * nb * MXX_BB, cols, xh + (size_t)j * cols, xl + (size_t)j * cols,
                     xs + (size_t)j * nb, out + (size_t)j * rows + 16 * grp);
}

/* The value of an E4M3 code (no NaN), from its bits: a normal code is 2^(e -
 * 7) (1 + m / 8), so its exponent field is e + 120 and its mantissa m << 20;
 * a subnormal is m 2^-9. The sign of a zero is kept. */
__device__ __forceinline__ float e4m3f(uint32_t b)
{
    uint32_t e = (b >> 3) & 15, m = b & 7;
    float v = e ? __uint_as_float(((e + 120) << 23) | (m << 20)) : (float)m * (1.f / 512.f);
    return __uint_as_float(__float_as_uint(v) | ((b & 128u) << 24));
}

/* out[j * rows + i] for t tokens: a warp for R rows (R divides 32: the rows of
 * one E8M0 scale), 8 warps a block, the tokens NT at a time. Each word of the
 * R rows is read and decoded once for the NT tokens, and each token's x once
 * for the R rows (a verify group of DSpark: one pass of the weights, not t);
 * the operations of each row and token are those of one at a time, in the
 * same order (the same bits). R = NT = 1: a step. */
/* One step of 4 columns of k_fp8b32_gemv: the words q of its R rows decoded
 * once, then for each of the tokens (x at the step's columns) and each row
 * p = w0 x0 + w1 x1 + w2 x2 + w3 x3 (in that order), acc = fma(p, s, acc). */
/* The 4 E4M3 values of a word without the table: each code's bits (sign,
 * exponent, mantissa) at bits 15, 13..7 of a half are its value times 2^-8,
 * the subnormals too (half: 2^-14 m / 8 = m 2^-9 2^-8); the half to float
 * and the 256 are exact: the values of e4m3f to the bit. (The table of
 * e4m3f in shared memory: 5% more time in the products of a step, its
 * lookups conflicting.) */
__device__ __forceinline__ void fp8x4_f16(uint32_t q, float *w)
{
    uint32_t a = __byte_perm(q, 0u, 0x5140), b = __byte_perm(q, 0u, 0x5342);
    uint32_t ha = ((a & 0x00800080u) << 8) | ((a & 0x007f007fu) << 7);
    uint32_t hb = ((b & 0x00800080u) << 8) | ((b & 0x007f007fu) << 7);
    float2 fa = __half22float2(*reinterpret_cast<const __half2 *>(&ha));
    float2 fb = __half22float2(*reinterpret_cast<const __half2 *>(&hb));
    w[0] = fa.x * 256.f;
    w[1] = fa.y * 256.f;
    w[2] = fb.x * 256.f;
    w[3] = fb.y * 256.f;
}

template <int R, int NT>
__device__ __forceinline__ void fp8_step(const uint32_t *q, float s, const float *x, size_t xstride, int tn,
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
            float4 xv = *(const float4 *)(x + jj * xstride);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                float p = wv[r][0] * xv.x;
                p = fmaf(wv[r][1], xv.y, p);
                p = fmaf(wv[r][2], xv.z, p);
                p = fmaf(wv[r][3], xv.w, p);
                acc[r][jj] = fmaf(p, s, acc[r][jj]);
            }
        }
    }
}

template <int R, int NT>
__global__ void __launch_bounds__(256) k_fp8b32_gemv(const uint8_t *w, const uint8_t *sc, int rows, int cols,
                                                    const float *x, int t, float *out, int groups)
{
    int row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    if (row0 >= rows) {
        return;
    }
    int nb = cols / 32;
    const uint8_t *wr = w + (size_t)row0 * cols;
    const uint8_t *sr = sc + (size_t)(row0 / 32) * nb;
    const float *xg = x + (size_t)(row0 / (rows / groups)) * cols;
    size_t xstride = (size_t)cols * groups;
    for (int j0 = 0; j0 < t; j0 += NT) {
        int tn = t - j0 < NT ? t - j0 : NT;
        const float *x0 = xg + (size_t)j0 * xstride;
        float acc[R][NT];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int jj = 0; jj < NT; ++jj) {
                acc[r][jj] = 0.f;
            }
        }
        /* the words of 4 steps of the columns loaded first (16 loads of a lane in
         * flight with R = 4; 8 or 16 steps: no faster), then each step in order */
        int c = 4 * lane;
        for (; c + 384 < cols; c += 512) {
            uint32_t q[4][R];
#pragma unroll
            for (int u = 0; u < 4; ++u) {
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    q[u][r] = *(const uint32_t *)(wr + (size_t)r * cols + c + 128 * u);
                }
            }
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                fp8_step<R, NT>(q[u], e8m0f(sr[(c + 128 * u) / 32]), x0 + c + 128 * u, xstride, tn, acc);
            }
        }
        for (; c < cols; c += 128) {
            uint32_t q[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                q[r] = *(const uint32_t *)(wr + (size_t)r * cols + c);
            }
            fp8_step<R, NT>(q, e8m0f(sr[c / 32]), x0 + c, xstride, tn, acc);
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
                    if (lane == 0) {
                        out[(size_t)(j0 + jj) * rows + row0 + r] = v;
                    }
                }
            }
        }
    }
}

/* GP_DS_FP8_LINEAR for a group (t > 8, the tensor cores): mma m16n8k8 tf32.
 * The E4M3 values of w are exact in tf32; x goes in as two tf32 parts (hi =
 * x with the low 13 bits cleared, lo = x - hi: about 21 bits of x, as
 * bf12_tc.cu). For each block of 32 columns the 4 k-steps sum into a
 * partial (fp32), then acc = fma(partial, the E8M0 scale of the block of 32
 * x 32, acc). A block: 4 warps, 64 rows x 32 tokens (warp w: rows 16 w ..
 * 16 w + 15, the 4 tiles of 8 tokens); w and x of each block of 32 columns
 * through shared memory. The rows of a block are in one group (rows /
 * groups a multiple of 64). */
#define F8T_BM 64
#define F8T_BN 32
__global__ void __launch_bounds__(128) k_fp8b32_tc(const uint8_t *w, const uint8_t *sc, int rows, int cols,
                                                  const float *x, int t, float *out, int groups)
{
    __shared__ float tab[256];
    __shared__ float ws[F8T_BM][33];
    __shared__ float xh[F8T_BN][33], xl[F8T_BN][33];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        tab[i] = e4m3f((uint32_t)i);
    }
    int r0 = blockIdx.x * F8T_BM, n0 = blockIdx.y * F8T_BN;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    int nb = cols / 32, rg = rows / groups;
    const float *xg = x + (size_t)(r0 / rg) * cols;
    size_t xstride = (size_t)cols * groups;
    float acc[4][4];
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            acc[nt][i] = 0.f;
        }
    }
    const uint8_t *srow = sc + (size_t)((r0 + 16 * warp) / 32) * nb;
    for (int b = 0; b < nb; ++b) {
        __syncthreads();
        {
            int rr = threadIdx.x / 2, half = threadIdx.x % 2;
            uint4 q = make_uint4(0, 0, 0, 0);
            if (r0 + rr < rows) {
                q = *(const uint4 *)(w + (size_t)(r0 + rr) * cols + 32 * b + 16 * half);
            }
            uint32_t wv[4] = {q.x, q.y, q.z, q.w};
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                ws[rr][16 * half + i] = tab[(wv[i / 4] >> (8 * (i % 4))) & 255];
            }
        }
        for (int i = threadIdx.x; i < F8T_BN * 32; i += blockDim.x) {
            int tk = i / 32, kk = i % 32;
            float v = n0 + tk < t ? xg[(size_t)(n0 + tk) * xstride + 32 * b + kk] : 0.f;
            float hi = __uint_as_float(__float_as_uint(v) & 0xffffe000u);
            xh[tk][kk] = hi;
            xl[tk][kk] = v - hi;
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
                uint32_t bh[2] = {__float_as_uint(xh[8 * nt + g][ks + c]), __float_as_uint(xh[8 * nt + g][ks + c + 4])};
                uint32_t bl[2] = {__float_as_uint(xl[8 * nt + g][ks + c]), __float_as_uint(xl[8 * nt + g][ks + c + 4])};
                mma1688_tf32(part[nt], a, bh);
                mma1688_tf32(part[nt], a, bl);
            }
        }
        float s = e8m0f(srow[b]);
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                acc[nt][i] = fmaf(part[nt][i], s, acc[nt][i]);
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
                out[(size_t)tt * rows + rr] = acc[nt][i];
            }
        }
    }
}
