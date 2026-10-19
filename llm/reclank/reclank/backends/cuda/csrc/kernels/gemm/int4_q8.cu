/* ---------- the int4 product with int8 activations ----------
 * k_gemm_q8: as k_gemm_tc2, but the rows of x are int8 with one float32
 * scale for each block of 32 values (k_quant_q8, as the Q8_0 form of
 * ggml). The instruction mma.sync m16n8k32 (int8 inputs, int32 sums) covers
 * one int4 block of 32 columns, and its int32 sum is exact. The kernel then
 * multiplies the sum by the scale of the block of x and the scale of the
 * block of w, in float32. On the RTX 5060 Ti the int8 instruction has about
 * 5 times the rate of the float16 one (204 against 37 TOPS).
 *
 * The fragments of m16n8k32 (g = lane / 4, c = lane % 4), 4 int8 values in
 * a register:
 *
 *     A (16 x 32, rows):  a0 = A[g][4c..], a1 = A[g+8][4c..],
 *                         a2 = A[g][4c+16..], a3 = A[g+8][4c+16..]
 *     B (32 x 8, cols):   b0 = B[4c..][g], b1 = B[4c+16..][g]
 *
 * The low 4 bits of byte i of a block are value i, and the high 4 bits are
 * value i + 16. Thus b0 comes from the low 4 bits of bytes 4c to 4c + 3.
 * b1 comes from their high 4 bits. __vsub4 subtracts 8 from each byte. */
__device__ __forceinline__ void mma16832(int *c, const uint32_t *a, const uint32_t *b)
{
#if RC_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

/* x (n rows of cols values) to int8, with one scale for each block of 32:
 * q = round(x / s), s = max |x| / 127. Eight threads for each block; each
 * thread reads 4 values as a float4 and writes 4 bytes. */
__global__ void k_quant_q8(const float *x, int8_t *q, float *sc, size_t blocks)
{
    PDL_START();
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t bi = tid / 8;
    int sub = threadIdx.x % 8;
    bool live = bi < blocks;
    float4 v = live ? *(const float4 *)(x + bi * 32 + sub * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, o));
    }
    if (!live) {
        return;
    }
    float s2 = m / 127.0f;
    char4 c4;
    c4.x = (signed char)(s2 > 0.f ? __float2int_rn(v.x / s2) : 0);
    c4.y = (signed char)(s2 > 0.f ? __float2int_rn(v.y / s2) : 0);
    c4.z = (signed char)(s2 > 0.f ? __float2int_rn(v.z / s2) : 0);
    c4.w = (signed char)(s2 > 0.f ? __float2int_rn(v.w / s2) : 0);
    *(char4 *)(q + bi * 32 + sub * 4) = c4;
    if (sub == 0) {
        sc[bi] = s2;
    }
}

/* x to the two int8 planes of the int16 form (RECLANK_INT4_Q8=16 on the
 * GPU), as kq_quant_part16 of the CPU, bit for bit: m = max |x| of each
 * block of 32, q = rint(x (16192 / m)) (at most 16192), hi = (q + 64) >> 7
 * (-126 to 127), lo = q - 128 hi (-64 to 63), s = m / 16192. A product takes
 * the int8 sums of hi and of lo and 128 sum(hi w) + sum(lo w) = sum(q w),
 * exact in int32 (|q w| summed over a block of 32 is at most 16192 * 8 * 32
 * < 2^22, so i2f_exact holds). So the hot experts of the GPU and the cold
 * ones of the CPU read the same int16 x (scripts/check_nvx_hot.py). xsum (or
 * NULL): s times the sum of the 32 values of q. (Before: s = m / 16256,
 * q = rint(x / s), hi = rint(q / 128); the same precision, not the bits of
 * the CPU.) */
#define X2_QMAX 16192
/* k_quant_x2 with each q made 8-bit (DeepSeek's measure of 8-bit activations:
 * 128 rint(q / 128), the rounding of the CPU's kq_round8). */
__global__ void k_quant_x2r(const float *x, int8_t *qh, int8_t *ql, float *xs, size_t blocks)
{
    PDL_START();
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t bi = tid / 8;
    int sub = threadIdx.x % 8;
    bool live = bi < blocks;
    float4 v = live ? *(const float4 *)(x + bi * 32 + sub * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, o));
    }
    float s2 = __fdiv_rn(m, (float)X2_QMAX);
    float iv = m > 0.f ? __fdiv_rn((float)X2_QMAX, m) : 0.f;
    float vv[4] = {v.x, v.y, v.z, v.w};
    signed char h[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        int q = min(X2_QMAX, max(-X2_QMAX, __float2int_rn(__fmul_rn(vv[k], iv))));
        h[k] = (signed char)__float2int_rn((float)q / 128.f);
    }
    if (!live) {
        return;
    }
    *(char4 *)(qh + bi * 32 + sub * 4) = make_char4(h[0], h[1], h[2], h[3]);
    *(char4 *)(ql + bi * 32 + sub * 4) = make_char4(0, 0, 0, 0);
    if (sub == 0) {
        xs[bi] = s2;
    }
}

__global__ void k_quant_x2(const float *x, int8_t *qh, int8_t *ql, float *xs, float *xsum,
                           size_t blocks)
{
    PDL_START();
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t bi = tid / 8;
    int sub = threadIdx.x % 8;
    bool live = bi < blocks;
    float4 v = live ? *(const float4 *)(x + bi * 32 + sub * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, o));
    }
    float s2 = __fdiv_rn(m, (float)X2_QMAX);
    float iv = m > 0.f ? __fdiv_rn((float)X2_QMAX, m) : 0.f;
    float vv[4] = {v.x, v.y, v.z, v.w};
    int q[4], h[4], sum = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        q[k] = min(X2_QMAX, max(-X2_QMAX, __float2int_rn(__fmul_rn(vv[k], iv))));
        h[k] = (q[k] + 64) >> 7;
        sum += q[k];
    }
    for (int o = 4; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    if (!live) {
        return;
    }
    *(char4 *)(qh + bi * 32 + sub * 4) = make_char4((signed char)h[0], (signed char)h[1],
                                                    (signed char)h[2], (signed char)h[3]);
    *(char4 *)(ql + bi * 32 + sub * 4) = make_char4(
        (signed char)(q[0] - 128 * h[0]), (signed char)(q[1] - 128 * h[1]),
        (signed char)(q[2] - 128 * h[2]), (signed char)(q[3] - 128 * h[3]));
    if (sub == 0) {
        xs[bi] = s2;
        if (xsum != NULL) {
            xsum[bi] = s2 * (float)sum;
        }
    }
}

/* The int32 sum of the int16 form (k_quant_x2): mma of hi, times 128, then
 * mma of lo into the same sums. */
__device__ __forceinline__ void mma16832_x2(int *ci, const uint32_t *ah, const uint32_t *al,
                                            const uint32_t *b);

/* The float32 value of an int32 i with |i| < 2^22, with no I2F instruction
 * (it has a low rate): the bits 0x4B400000 + i are the float 12582912 + i.
 * The subtraction is exact. */
__device__ __forceinline__ float i2f_exact(int i)
{
    return __int_as_float(0x4B400000 + i) - 12582912.0f;
}

__device__ __forceinline__ void cp_async16_z(void *smem, const void *gmem, int bytes);

#define Q8K 128      /* the columns of a step of k_gemm_q8: 4 blocks */
#define Q8RB 80      /* bytes of a row of w in a step: 72 bytes, and 8 of pad */
#define Q8NS 3       /* the buffers of k_gemm_q8 */
#define Q8SMEM ((size_t)Q8NS * (T2M * (Q8K + 16) + T2N * Q8RB + T2M * (Q8K / 32) * 4))
/* X2 (the int16 form): the two planes of x in each row of a step, 2 buffers */
#define Q8SMEM_X2 ((size_t)2 * (T2M * (2 * Q8K + 16) + T2N * Q8RB + T2M * (Q8K / 32) * 4))

__device__ __forceinline__ void mma16832_x2(int *ci, const uint32_t *ah, const uint32_t *al,
                                            const uint32_t *b)
{
    mma16832(ci, ah, b);
    ci[0] *= 128;
    ci[1] *= 128;
    ci[2] *= 128;
    ci[3] *= 128;
    mma16832(ci, al, b);
}

__device__ __forceinline__ void cp_async8(void *smem, const void *gmem)
{
#if RC_SM80
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(sa), "l"(gmem));
#else
    *(uint2 *)smem = *(const uint2 *)gmem;
#endif
}

/* X2: the int16 form (k_quant_x2): xq the hi plane, xl the lo plane, a row
 * of a step holds the hi values and then the lo values; 2 buffers. */
template <int AL, int X2 = 0>
__global__ void __launch_bounds__(256) k_gemm_q8(const int8_t *xq, const float *xs,
                                                 const uint8_t *w, float *out, int t, int rows,
                                                 int cols, const int8_t *xl = NULL)
{
    PDL_START();
    /* NS buffers in dynamic shared memory: the copies of NS - 1 steps
     * run during the compute of a step. */
    constexpr int NS = X2 ? 2 : Q8NS, AW = Q8K * (1 + X2) + 16;
    extern __shared__ __align__(16) uint8_t q8sm[];
    int8_t (*as_)[T2M][AW] = (int8_t (*)[T2M][AW])q8sm;
    uint8_t (*bs)[T2N][Q8RB] = (uint8_t (*)[T2N][Q8RB])(q8sm + NS * T2M * AW);
    float (*ss)[T2M][Q8K / 32] = (float (*)[T2M][Q8K / 32])(q8sm + NS * (T2M * AW +
                                                                        T2N * Q8RB));
    /* blockIdx.x is the tile of tokens, so the blocks that read the same rows of
     * w run together and the rows come from DRAM one time. */
    int m0 = blockIdx.x * T2M, n0 = blockIdx.y * T2N;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    size_t rb = (size_t)(cols / 32) * 18;
    int nb = cols / 32, steps = cols / Q8K;
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    auto load = [&](int st, int b) {
        int k0 = st * Q8K;
        for (int e = threadIdx.x; e < T2M * Q8K / 16; e += blockDim.x) {
            int m = e / (Q8K / 16), kk = (e % (Q8K / 16)) * 16;
            int q = min(m0 + m, t - 1);
            cp_async16(&as_[b][m][kk], xq + (size_t)q * cols + k0 + kk);
            if (X2) {
                cp_async16(&as_[b][m][Q8K + kk], xl + (size_t)q * cols + k0 + kk);
            }
        }
        for (int e = threadIdx.x; e < T2M * (Q8K / 32); e += blockDim.x) {
            int m = e / (Q8K / 32), bk = e % (Q8K / 32);
            int q = min(m0 + m, t - 1);
            cp_async4(&ss[b][m][bk], xs + (size_t)q * nb + k0 / 32 + bk);
        }
        /* The 4 blocks of a row in the step are 72 bytes. With AL, they
         * start at a multiple of 8 bytes (rb is a multiple of 16), so 9
         * copies of 8 bytes take them. */
        const uint8_t *wk = w + (size_t)(k0 / 32) * 18;
        if (AL) {
            for (int e = threadIdx.x; e < T2N * 9; e += blockDim.x) {
                int n = e / 9, wd = e % 9;
                int row = min(n0 + n, rows - 1);
                cp_async8(&bs[b][n][wd * 8], wk + (size_t)row * rb + wd * 8);
            }
        } else {
            for (int e = threadIdx.x; e < T2N * 18; e += blockDim.x) {
                int n = e / 18, wd = e % 18;
                int row = min(n0 + n, rows - 1);
                cp_async4(&bs[b][n][wd * 4], wk + (size_t)row * rb + wd * 4);
            }
        }
        cp_async_commit();
    };
    for (int st = 0; st < NS - 1; ++st) {
        if (st < steps) {
            load(st, st);
        } else {
            cp_async_commit();
        }
    }
    for (int st = 0; st < steps; ++st) {
        int b = st % NS;
#if RC_SM80
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 2));
#endif
        __syncthreads();
        /* Buffer (st + NS - 1) % NS held step st - 1, which every warp
         * has passed. */
        if (st + NS - 1 < steps) {
            load(st + NS - 1, (st + NS - 1) % NS);
        } else {
            cp_async_commit();
        }
        const int sh = 0;
        #pragma unroll
        for (int bk = 0; bk < Q8K / 32; ++bk) {
            uint32_t bf[4][2];
            float dw[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t *blkp = &bs[b][wn + j * 8 + g][sh + bk * 18];
                uint32_t v = (uint32_t)*(const uint16_t *)(blkp + 2 + 4 * c) |
                             ((uint32_t)*(const uint16_t *)(blkp + 4 + 4 * c) << 16);
                bf[j][0] = __vsub4(v & 0x0f0f0f0fu, 0x08080808u);
                bf[j][1] = __vsub4((v >> 4) & 0x0f0f0f0fu, 0x08080808u);
                const uint8_t *d0 = &bs[b][wn + j * 8 + 2 * c][sh + bk * 18];
                const uint8_t *d1 = &bs[b][wn + j * 8 + 2 * c + 1][sh + bk * 18];
                dw[j][0] = __half2float(__ushort_as_half((uint16_t)(d0[0] | (d0[1] << 8))));
                dw[j][1] = __half2float(__ushort_as_half((uint16_t)(d1[0] | (d1[1] << 8))));
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r0 = wm + i * 16;
                int kk = bk * 32;
                uint32_t a[4], al[4];
                a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                if (X2) {
                    al[0] = *(const uint32_t *)&as_[b][r0 + g][Q8K + kk + 4 * c];
                    al[1] = *(const uint32_t *)&as_[b][r0 + g + 8][Q8K + kk + 4 * c];
                    al[2] = *(const uint32_t *)&as_[b][r0 + g][Q8K + kk + 4 * c + 16];
                    al[3] = *(const uint32_t *)&as_[b][r0 + g + 8][Q8K + kk + 4 * c + 16];
                }
                float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int ci[4] = {0, 0, 0, 0};
                    if (X2) {
                        mma16832_x2(ci, a, al, bf[j]);
                    } else {
                        mma16832(ci, a, bf[j]);
                    }
                    acc[i][j][0] += i2f_exact(ci[0]) * s0 * dw[j][0];
                    acc[i][j][1] += i2f_exact(ci[1]) * s0 * dw[j][1];
                    acc[i][j][2] += i2f_exact(ci[2]) * s1 * dw[j][0];
                    acc[i][j][3] += i2f_exact(ci[3]) * s1 * dw[j][1];
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = n0 + wn + j * 8 + 2 * c;
            int m = m0 + wm + i * 16 + g;
            if (m < t) {
                if (n < rows) out[(size_t)m * rows + n] = acc[i][j][0];
                if (n + 1 < rows) out[(size_t)m * rows + n + 1] = acc[i][j][1];
            }
            if (m + 8 < t) {
                if (n < rows) out[(size_t)(m + 8) * rows + n] = acc[i][j][2];
                if (n + 1 < rows) out[(size_t)(m + 8) * rows + n + 1] = acc[i][j][3];
            }
        }
    }
}

/* The float16 copy of n values, for the tensor cores. */
__global__ void k_to_half(const float *x, __half *y, size_t n)
{
    PDL_START();
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *(const float4 *)(x + i);
        *(__half2 *)(y + i) = __floats2half2_rn(v.x, v.y);
        *(__half2 *)(y + i + 2) = __floats2half2_rn(v.z, v.w);
    } else {
        for (; i < n; ++i) {
            y[i] = __float2half_rn(x[i]);
        }
    }
}

#define MT_MAX 16

/* k_mt_gemv with the token count NT fixed when the kernel compiles. The
 * loops over the tokens then unroll, and a lane keeps NT sums. The general
 * form keeps 16 sums and ran at about half the rate of one token. R rows a
 * warp (k_mt_int4_rows). */
template <int NT, int R = 1>
__global__ void k_mt_gemv_n(const float *x, const uint8_t *w, float *out, int rows, int cols)
{
    PDL_START();
    int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32) * R;
    int lane = threadIdx.x % 32, sub = lane & 3;
    if (row0 >= rows) {
        return;
    }
    int blocks = cols / 32;
    const uint8_t *wr[R];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        wr[i] = w + (size_t)min(row0 + i, rows - 1) * blocks * 18;
    }
    float sum[R][NT];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            sum[i][j] = 0.f;
        }
    }
    for (int b = lane >> 2; b < blocks; b += 8) {
        float wl[R][4], wh[R][4];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const uint8_t *blk = wr[i] + (size_t)b * 18;
            float d = __half2float(__ushort_as_half(*(const uint16_t *)blk));
            const uint16_t *qp = (const uint16_t *)(blk + 2 + 4 * sub);
            uint32_t q = (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                wl[i][u] = (float)((int)((q >> (8 * u)) & 15) - 8) * d;
                wh[i][u] = (float)((int)((q >> (8 * u + 4)) & 15) - 8) * d;
            }
        }
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float *xb = x + (size_t)j * cols + b * 32;
            float4 xl = *(const float4 *)(xb + 4 * sub);
            float4 xh = *(const float4 *)(xb + 16 + 4 * sub);
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                sum[i][j] += wl[i][0] * xl.x + wl[i][1] * xl.y + wl[i][2] * xl.z + wl[i][3] * xl.w
                           + wh[i][0] * xh.x + wh[i][1] * xh.y + wh[i][2] * xh.z + wh[i][3] * xh.w;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            float v = sum[i][j];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (lane == 0 && row0 + i < rows) {
                out[(size_t)j * rows + row0 + i] = v;
            }
        }
    }
}

/* A small group of NT tokens (an MTP verify group) on an int4 matrix, with
 * the bits of int4_row (a decode step) for each token: the same lanes, and
 * the same order of the terms (int4_part, then the scale of the block).
 * k_mt_gemv_n multiplied each weight by the scale first, so a token of a
 * group did not get the bits of a step. R rows a warp, each with the lanes
 * and the order of a row alone (the same bits): the x of a block is read
 * once for the R rows. With one row a warp, the reads of x (32 bytes a
 * token for 4 bytes of weights) limited a group: on the 3090, 4 tokens on
 * the matrices of the Gemma 4 26B took 2.5 to 3.7 times one token, and R 2
 * is 1.4 to 1.9 times as fast (scripts/bench_mt_rows.cu; R 4 is not better
 * on most; R 2 on a matrix of 1024 rows was as often slower). One row a
 * warp still for the small matrices (mt_rows_r). */
template <int NT, int R = 1>
__global__ void k_mt_int4_rows(const float *x, const uint8_t *w, float *out, int rows, int cols)
{
    PDL_START();
    int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32) * R;
    int lane = threadIdx.x % 32, sub = lane & 3;
    if (row0 >= rows) {
        return;
    }
    int blocks = cols / 32;
    const uint8_t *wr[R];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        wr[i] = w + (size_t)min(row0 + i, rows - 1) * blocks * 18;
    }
    float sum[R][NT];
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            sum[i][j] = 0.f;
        }
    }
    for (int b = lane >> 2; b < blocks; b += 8) {
        float d[R];
        uint32_t q[R];
        #pragma unroll
        for (int i = 0; i < R; ++i) {
            const uint8_t *blk = wr[i] + (size_t)b * 18;
            d[i] = __half2float(__ushort_as_half(*(const uint16_t *)blk));
            const uint16_t *qp = (const uint16_t *)(blk + 2 + 4 * sub);
            q[i] = (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
        }
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float *xb = x + (size_t)j * cols + b * 32;
            float4 xl = *(const float4 *)(xb + 4 * sub);
            float4 xh = *(const float4 *)(xb + 16 + 4 * sub);
            #pragma unroll
            for (int i = 0; i < R; ++i) {
                float acc = int4_part(q[i], xl, xh);
                sum[i][j] += d[i] * acc;
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < R; ++i) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            float v = sum[i][j];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (lane == 0 && row0 + i < rows) {
                out[(size_t)j * rows + row0 + i] = v;
            }
        }
    }
}

/* The rows of a warp for a group of t tokens on a matrix of rows rows
 * (k_mt_int4_rows, k_mt_gemv_n). */
static inline int mt_rows_r(int t, int rows)
{
    return t >= 2 && rows >= 2048 ? 2 : 1;
}

__global__ void k_mt_gemv(const float *x, const uint8_t *w, float *out, int t, int rows, int cols)
{
    PDL_START();
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32, sub = lane & 3;
    if (row >= rows) {
        return;
    }
    int blocks = cols / 32;
    const uint8_t *wr = w + (size_t)row * blocks * 18;
    float sum[MT_MAX];
    for (int j = 0; j < MT_MAX; ++j) {
        sum[j] = 0.f;
    }
    for (int b = lane >> 2; b < blocks; b += 8) {
        const uint8_t *blk = wr + (size_t)b * 18;
        float d = __half2float(__ushort_as_half(*(const uint16_t *)blk));
        const uint16_t *qp = (const uint16_t *)(blk + 2 + 4 * sub);
        uint32_t q = (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
        float wl[4], wh[4];
        for (int u = 0; u < 4; ++u) {
            wl[u] = (float)((int)((q >> (8 * u)) & 15) - 8) * d;
            wh[u] = (float)((int)((q >> (8 * u + 4)) & 15) - 8) * d;
        }
        for (int j = 0; j < MT_MAX; ++j) {
            if (j < t) {
                const float *xb = x + (size_t)j * cols + b * 32;
                float4 xl = *(const float4 *)(xb + 4 * sub);
                float4 xh = *(const float4 *)(xb + 16 + 4 * sub);
                sum[j] += wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w
                        + wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
            }
        }
    }
    for (int j = 0; j < MT_MAX; ++j) {
        if (j < t) {
            float v = sum[j];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (lane == 0) {
                out[(size_t)j * rows + row] = v;
            }
        }
    }
}

/* A bfloat16 matrix on a small group: one warp for each row. Each lane reads
 * 8 values of the row at a time and uses them for every token. */
__global__ void k_mt_gemv_bf16(const float *x, const uint16_t *w, float *out, int t, int rows,
                               int cols)
{
    PDL_START();
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const uint4 *wr = (const uint4 *)(w + (size_t)row * cols);
    float sum[MT_MAX];
    for (int j = 0; j < MT_MAX; ++j) {
        sum[j] = 0.f;
    }
    for (int i = lane; i < cols / 8; i += 32) {
        uint4 q = wr[i];
        uint32_t u[4] = {q.x, q.y, q.z, q.w};
        float wv[8];
        for (int k = 0; k < 4; ++k) {
            wv[2 * k] = __uint_as_float(u[k] << 16);
            wv[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
        }
        for (int j = 0; j < MT_MAX; ++j) {
            if (j < t) {
                /* The terms of k_bf16_linear, so a token of a group gets
                 * the bits of a decode step (an MTP verify group); x in
                 * two 16-byte loads (rows of cols floats, cols % 8 == 0). */
                const float4 *x4 = (const float4 *)(x + (size_t)j * cols + i * 8);
                const float4 a = x4[0], b = x4[1];
                const float xi[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    sum[j] += wv[2 * k] * xi[2 * k] + wv[2 * k + 1] * xi[2 * k + 1];
                }
            }
        }
    }
    for (int j = 0; j < MT_MAX; ++j) {
        if (j < t) {
            float v = sum[j];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (lane == 0) {
                out[(size_t)j * rows + row] = v;
            }
        }
    }
}

/* g, u, out, rows, inner: out = gelu(g) u for each value. */
__global__ void k_gelu_mul_rows(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)DI(3) * DI(4)) {
        float v = DP(const float, 0)[i];
        DP(float, 2)[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
                                                   (v + 0.044715f * v * v * v)))
                          * DP(const float, 1)[i];
    }
}

/* The last key that query j of a GP_ATTN_QC_MT record sees: pos + lim[j]
 * when the record has the array lim (operand 16: the tokens of an image see
 * each other), else its own position pos + j. */
__device__ __forceinline__ int64_t qc_last(const int *lim, int64_t pos, int j)
{
    return pos + (lim ? lim[j] : j);
}

/* The largest last key (less pos) of the queries j0 to jl. */
__device__ __forceinline__ int qc_tile_last(const int *lim, int j0, int jl)
{
    int m = jl;
    if (lim) {
        for (int a = j0; a <= jl; ++a) {
            m = max(m, lim[a]);
        }
    }
    return m;
}

/* The attention of a group of queries over the int16 cache, as FlashAttention
 * does it. The record GP_ATTN_QC_MT:
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, t, pos,
 *     base, window, lo, n, lim
 *
 * lim (optional, int32, t values) gives the last key of each query less pos
 * (qc_last). Without it query j sees the keys up to its own position.
 *
 * The cache addresses point at buffer row 0, whose position is base. Query j
 * has the position pos + j. It sees the rows of the positions pos + j -
 * window + 1 to pos + j, or from position 0 when window is 0.
 *
 * Block (h, tile) takes query head h and QT queries. Thread x keeps query
 * x / 16 and head_dim / 16 of its values, and the same values of the output.
 * The block reads KT key rows at a time to shared memory. Each thread
 * computes its part of the dot of its query with each key. The 16 threads of
 * a query add the parts with shuffles. Then each thread updates its output
 * with the online softmax. */
#define QT 16
#define KT 8

__global__ void k_attn_qc_mt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float ks_[KT][512];
    __shared__ float vs_[KT][512];
    const float *q = DP(const float, 0);
    const int16_t *kq = DP(const int16_t, 1);
    const float *ksc = DP(const float, 2);
    const int16_t *vq = DP(const int16_t, 3);
    const float *vsc = DP(const float, 4);
    float *out = DP(float, 6);
    int qh = DI(7), kvh = DI(8), hd = DI(9), t = DI(10), window = DI(13);
    int64_t pos = di(r, e, 11), base = di(r, e, 12);
    int h = blockIdx.x, kv = h / (qh / kvh);
    int j0 = blockIdx.y * QT;
    int qi = threadIdx.x / 16, part = threadIdx.x % 16;
    int j = j0 + qi;
    int per = hd / 16;                      /* 16 or 32 values */
    int d0 = part * per;
    bool live = j < t;
    int64_t p = pos + j;
    float qv[32], acc[32];
    for (int u = 0; u < 32; ++u) {
        qv[u] = (u < per && live) ? q[((size_t)j * qh + h) * hd + d0 + u] : 0.f;
        acc[u] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    int jl = min(t, j0 + QT) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int *lim = DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t pe = live ? qc_last(lim, pos, j) : p;
    size_t row = (size_t)kvh * hd;
    for (int64_t k0 = first; k0 <= last; k0 += KT) {
        int kn = (int)min((int64_t)KT, last - k0 + 1);
        for (int x = threadIdx.x; x < KT * hd; x += blockDim.x) {
            int kk = x / hd, i = x % hd;
            if (kk < kn) {
                size_t o = (size_t)(k0 - base + kk) * row + (size_t)kv * hd + i;
                ks_[kk][i] = (float)kq[o] * ksc[o / 32];
                vs_[kk][i] = (float)vq[o] * vsc[o / 32];
                if (kq == vq && ksc != vsc && ko_turned(i & ~7)) {
                    /* keys only (ko_tab): a turned value */
                    size_t po = o - i + ko_partner(i & ~7) + (i & 7);
                    vs_[kk][i] = ko_rot1(vs_[kk][i], (float)vq[po] * vsc[po / 32], k0 + kk, i);
                }
            }
        }
        __syncthreads();
        float sc[KT];
        for (int kk = 0; kk < KT; ++kk) {
            float sdot = 0.f;
            if (kk < kn) {
                for (int u = 0; u < 32; ++u) {
                    if (u < per) {
                        sdot += qv[u] * ks_[kk][d0 + u];
                    }
                }
            }
            for (int o = 8; o > 0; o >>= 1) {
                sdot += __shfl_xor_sync(RC_FULL_MASK, sdot, o);
            }
            int64_t kp = k0 + kk;
            bool ok = kk < kn && kp <= pe && (window == 0 || p - kp < window);
            sc[kk] = ok ? sdot : -INFINITY;
        }
        float mt = m;
        for (int kk = 0; kk < KT; ++kk) {
            mt = fmaxf(mt, sc[kk]);
        }
        if (mt > -INFINITY) {
            float scale = m == -INFINITY ? 0.f : expf(m - mt);
            l *= scale;
            for (int u = 0; u < 32; ++u) {
                acc[u] *= scale;
            }
            for (int kk = 0; kk < KT; ++kk) {
                if (sc[kk] == -INFINITY) {
                    continue;
                }
                float pk = expf(sc[kk] - mt);
                l += pk;
                for (int u = 0; u < 32; ++u) {
                    if (u < per) {
                        acc[u] += pk * vs_[kk][d0 + u];
                    }
                }
            }
            m = mt;
        }
        __syncthreads();
    }
    if (live) {
        float inv = l > 0.f ? 1.0f / l : 0.f;
        for (int u = 0; u < per; ++u) {
            out[((size_t)j * qh + h) * hd + d0 + u] = acc[u] * inv;
        }
    }
}

/* A faster form of k_attn_qc_mt for a large group. It uses tiles of FQ
 * queries by FK keys, as FlashAttention-2 does, but without tensor cores.
 *
 * Block (h, tile) takes query head h and FQ queries. Thread x belongs to
 * query x / 8 and to slice x % 8. For each tile of FK keys:
 *
 * 1. The scores: the block reads the queries and the keys to shared memory,
 *    64 values of each at a time. Thread (q, s) computes the scores of query
 *    q with keys s, s + 8, s + 16, and s + 24.
 * 2. The 8 threads of a query find the maximum and the sum of the tile with
 *    shuffles. They update the running maximum m and sum l.
 * 3. The values: the block reads the value rows to shared memory, 64 values
 *    at a time. Thread (q, s) keeps the output values s, s + 8, s + 16, ...
 *    of query q.
 *
 * The block has 256 threads. head_dim is 256 or 512. */
#define FQ 32
#define FK 32
#define FD 64

/* F32H 0: GP_ATTN_QC_MT, the int16 cache of the 26B (rows, heads, dim).
 * F32H 1: GP_ATTN_F32H, the float cache of the E4B (heads, positions, dim).
 * Its operands: q, k, v, scores, out, q_heads, kv_heads, head_dim, t, pos,
 * head_stride, window, slide. Its buffer row 0 has position 0. */
template <int F32H>
__device__ __forceinline__ float flash_kv(const gp_rec *r, const int64_t *e, int key,
                                          int64_t kp, int kv, int d)
{
    if (F32H) {
        const float *b = DP(const float, key ? 1 : 2);
        return b[(size_t)kv * di(r, e, 10) + (size_t)kp * DI(7) + d];
    }
    size_t o = (size_t)(kp - di(r, e, 12)) * DI(8) * DI(9) + (size_t)kv * DI(9) + d;
    const int16_t *q = DP(const int16_t, key ? 1 : 3);
    const float *sc = DP(const float, key ? 2 : 4);
    return (float)q[o] * sc[o / 32];
}

template <int F32H>
__global__ void k_flash_qc_mt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float qs[FQ][FD + 1];
    __shared__ float kvs[FK][FD + 1];
    __shared__ float ps[FQ][FK + 1];
    const float *q = DP(const float, 0);
    float *out = DP(float, F32H ? 4 : 6);
    int o0 = F32H ? 5 : 7;
    int qh = DI(o0), kvh = DI(o0 + 1), hd = DI(o0 + 2), t = DI(o0 + 3);
    int window = F32H ? DI(11) : DI(13);
    int64_t pos = di(r, e, F32H ? 9 : 11), base = F32H ? 0 : di(r, e, 12);
    int h = blockIdx.x, kv = h / (qh / kvh);
    int j0 = blockIdx.y * FQ;
    int qi = threadIdx.x / 8, sl = threadIdx.x % 8;
    int j = j0 + qi;
    int64_t p = pos + j;
    int jl = min(t, j0 + FQ) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int *lim = F32H ? NULL : DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t pe = j < t ? qc_last(lim, pos, j) : p;
    /* The rows of head kv: the float cache (head-major) or the int16 cache
     * (row-major, from the head offset). */
    const float *kf = NULL, *vf = NULL;
    const int16_t *kq16 = NULL, *vq16 = NULL;
    const float *ks16 = NULL, *vs16 = NULL;
    size_t rowq = (size_t)kvh * hd, frs = hd;
    if (F32H) {
        kf = DP(const float, 1) + (size_t)kv * di(r, e, 10);
        vf = DP(const float, 2) + (size_t)kv * di(r, e, 10);
        frs = kv_rs((size_t)di(r, e, 10), kvh, hd);
    } else {
        kq16 = DP(const int16_t, 1) + (size_t)kv * hd;
        vq16 = DP(const int16_t, 3) + (size_t)kv * hd;
        ks16 = DP(const float, 2) + (size_t)kv * hd / 32;
        vs16 = DP(const float, 4) + (size_t)kv * hd / 32;
    }
    float o[64];
    for (int u = 0; u < 64; ++u) {
        o[u] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int64_t k0 = first; k0 <= last; k0 += FK) {
        int kn = (int)min((int64_t)FK, last - k0 + 1);
        float sc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int d0 = 0; d0 < hd; d0 += FD) {
            for (int x = threadIdx.x; x < FQ * FD; x += blockDim.x) {
                int a = x / FD, d = x % FD;
                int jj = j0 + a;
                qs[a][d] = jj < t ? q[((size_t)jj * qh + h) * hd + d0 + d] : 0.f;
                float kv_v = 0.f;
                if (a < kn) {
                    if (F32H) {
                        kv_v = kf[(size_t)(k0 + a) * frs + d0 + d];
                    } else {
                        size_t o2 = (size_t)(k0 - base + a) * rowq + d0 + d;
                        kv_v = (float)kq16[o2] * ks16[o2 / 32];
                    }
                }
                kvs[a][d] = kv_v;
            }
            __syncthreads();
            #pragma unroll 8
            for (int d = 0; d < FD; ++d) {
                float qv = qs[qi][d];
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    sc[v] += qv * kvs[sl + 8 * v][d];
                }
            }
            __syncthreads();
        }
        float mt = m;
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            int kk = sl + 8 * v;
            int64_t kp = k0 + kk;
            bool ok = kk < kn && j < t && kp <= pe && (window == 0 || p - kp < window);
            sc[v] = ok ? sc[v] : -INFINITY;
            mt = fmaxf(mt, sc[v]);
        }
        for (int off = 4; off > 0; off >>= 1) {
            mt = fmaxf(mt, __shfl_xor_sync(RC_FULL_MASK, mt, off));
        }
        float scale = (m == -INFINITY || mt == -INFINITY) ? (m == -INFINITY ? 0.f : 1.f)
                                                          : expf(m - mt);
        float ls = 0.f;
        #pragma unroll
        for (int v = 0; v < 4; ++v) {
            float pk = sc[v] == -INFINITY ? 0.f : expf(sc[v] - mt);
            ps[qi][sl + 8 * v] = pk;
            ls += pk;
        }
        for (int off = 4; off > 0; off >>= 1) {
            ls += __shfl_xor_sync(RC_FULL_MASK, ls, off);
        }
        if (mt != -INFINITY) {
            l = l * scale + ls;
            for (int u = 0; u < 64; ++u) {
                o[u] *= scale;
            }
            m = mt;
        }
        __syncthreads();
        /* The loop has a fixed count, so the index of o is a constant and o
         * stays in registers. */
        #pragma unroll
        for (int c = 0; c < 512 / FD; ++c) {
            int d0 = c * FD;
            if (d0 >= hd) {
                break;
            }
            for (int x = threadIdx.x; x < FK * FD; x += blockDim.x) {
                int a = x / FD, d = x % FD;
                float kv_v = 0.f;
                if (a < kn) {
                    if (F32H) {
                        kv_v = vf[(size_t)(k0 + a) * frs + d0 + d];
                    } else {
                        size_t o2 = (size_t)(k0 - base + a) * rowq + d0 + d;
                        kv_v = (float)vq16[o2] * vs16[o2 / 32];
                        int dd = d0 + d;
                        if (kq16 == vq16 && ks16 != vs16 && ko_turned(dd & ~7)) {
                            /* keys only (ko_tab): a turned value */
                            size_t po = o2 - dd + ko_partner(dd & ~7) + (dd & 7);
                            kv_v = ko_rot1(kv_v, (float)vq16[po] * vs16[po / 32], k0 + a, dd);
                        }
                    }
                }
                kvs[a][d] = kv_v;
            }
            __syncthreads();
            #pragma unroll
            for (int w = 0; w < FD / 8; ++w) {
                int d = sl + 8 * w;
                float acc = 0.f;
                #pragma unroll 8
                for (int kk = 0; kk < FK; ++kk) {
                    acc += ps[qi][kk] * kvs[kk][d];
                }
                o[c * (FD / 8) + w] += acc;
            }
            __syncthreads();
        }
    }
    if (j < t) {
        float inv = l > 0.f ? 1.0f / l : 0.f;
        #pragma unroll
        for (int c = 0; c < 512 / FD; ++c) {
            if (c * FD >= hd) {
                break;
            }
            #pragma unroll
            for (int w = 0; w < FD / 8; ++w) {
                out[((size_t)j * qh + h) * hd + c * FD + sl + 8 * w] = o[c * (FD / 8) + w] * inv;
            }
        }
    }
}
