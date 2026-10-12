/* ---------- the product with a bfloat16 matrix on the tensor cores ----------
 * k_gemm_bh: as k_gemm_tc2, for a matrix of bfloat16 rows (GP_BF16_LINEAR,
 * the projection of the layer input of the E4B). The block copies steps of
 * BHK columns of x (float16) and of w (bfloat16) with cp.async into two
 * buffers. A fragment of B changes from bfloat16 to float16 in registers.
 * The sums are float32. */
#define BHK 32

__device__ __forceinline__ uint32_t bf2_to_h2(uint32_t v)
{
    __half2 h = __floats2half2_rn(__uint_as_float(v << 16), __uint_as_float(v & 0xffff0000u));
    return *(uint32_t *)&h;
}

__global__ void __launch_bounds__(256) k_gemm_bh(const __half *x, const uint16_t *w, float *out,
                                                 int t, int rows, int cols)
{
    PDL_START();
    __shared__ __align__(16) __half as_[2][T2M][BHK + 8];
    __shared__ __align__(16) uint16_t bs[2][T2N][BHK + 8];
    /* blockIdx.x is the tile of tokens, so the blocks that read the same rows of
     * w run together and the rows come from DRAM one time. */
    int m0 = blockIdx.x * T2M, n0 = blockIdx.y * T2N;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    int steps = cols / BHK;
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    /* Copy step st to buffer b. A row past the end reads the last row; the
     * tile does not write its result. */
    auto load = [&](int st, int b) {
        int k0 = st * BHK;
        for (int e = threadIdx.x; e < T2M * BHK / 8; e += blockDim.x) {
            int m = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int q = min(m0 + m, t - 1);
            cp_async16(&as_[b][m][kk], x + (size_t)q * cols + k0 + kk);
        }
        for (int e = threadIdx.x; e < T2N * BHK / 8; e += blockDim.x) {
            int n = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int row = min(n0 + n, rows - 1);
            cp_async16(&bs[b][n][kk], w + (size_t)row * cols + k0 + kk);
        }
        cp_async_commit();
    };
    load(0, 0);
    for (int st = 0; st < steps; ++st) {
        int b = st & 1;
        if (st + 1 < steps) {
            load(st + 1, b ^ 1);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < BHK / 16; ++ks) {
            int kk = ks * 16;
            uint32_t bf[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint16_t *br = &bs[b][wn + j * 8 + g][kk + 2 * c];
                bf[j][0] = bf2_to_h2(*(const uint32_t *)br);
                bf[j][1] = bf2_to_h2(*(const uint32_t *)(br + 8));
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r0 = wm + i * 16;
                uint32_t a[4];
                a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 2 * c];
                a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 2 * c];
                a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 2 * c + 8];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma16816(acc[i][j], a, bf[j]);
                }
            }
        }
        __syncthreads();
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

/* x (float32) to float16 with the clamps of an encoder linear (ENC_LINEAR). */
__global__ void k_enc_to_half(const float *x, __half *y, size_t n, float imin, float imax)
{
    PDL_START();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half_rn(fminf(fmaxf(x[i], imin), imax));
    }
}

/* The product of an encoder linear on the tensor cores: as k_gemm_bh, for
 * any cols that is a multiple of 8 (the 26B has 4304). The last step reads
 * zeros past cols. out = clamp(x W^T + b, omin, omax); b may be NULL. */
__global__ void __launch_bounds__(256) k_enc_gemm_tc(const __half *x, const uint16_t *w,
                                                     const float *bias, float *out, int t,
                                                     int rows, int cols, float omin, float omax)
{
    PDL_START();
    __shared__ __align__(16) __half as_[2][T2M][BHK + 8];
    __shared__ __align__(16) uint16_t bs[2][T2N][BHK + 8];
    int m0 = blockIdx.x * T2M, n0 = blockIdx.y * T2N;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    int steps = (cols + BHK - 1) / BHK;
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    auto load = [&](int st, int b) {
        int k0 = st * BHK;
        for (int e = threadIdx.x; e < T2M * BHK / 8; e += blockDim.x) {
            int m = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int q = min(m0 + m, t - 1);
            if (k0 + kk < cols) {
                cp_async16(&as_[b][m][kk], x + (size_t)q * cols + k0 + kk);
            } else {
                *(uint4 *)&as_[b][m][kk] = make_uint4(0, 0, 0, 0);
            }
        }
        for (int e = threadIdx.x; e < T2N * BHK / 8; e += blockDim.x) {
            int n = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int row = min(n0 + n, rows - 1);
            if (k0 + kk < cols) {
                cp_async16(&bs[b][n][kk], w + (size_t)row * cols + k0 + kk);
            } else {
                *(uint4 *)&bs[b][n][kk] = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
    };
    load(0, 0);
    for (int st = 0; st < steps; ++st) {
        int b = st & 1;
        if (st + 1 < steps) {
            load(st + 1, b ^ 1);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();
        #pragma unroll
        for (int ks = 0; ks < BHK / 16; ++ks) {
            int kk = ks * 16;
            uint32_t bf[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint16_t *br = &bs[b][wn + j * 8 + g][kk + 2 * c];
                bf[j][0] = bf2_to_h2(*(const uint32_t *)br);
                bf[j][1] = bf2_to_h2(*(const uint32_t *)(br + 8));
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r0 = wm + i * 16;
                uint32_t a[4];
                a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 2 * c];
                a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 2 * c];
                a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 2 * c + 8];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma16816(acc[i][j], a, bf[j]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = n0 + wn + j * 8 + 2 * c;
            int m = m0 + wm + i * 16 + g;
            float b0 = (bias && n < rows) ? bias[n] : 0.f;
            float b1 = (bias && n + 1 < rows) ? bias[n + 1] : 0.f;
            if (m < t) {
                if (n < rows) out[(size_t)m * rows + n] = fminf(fmaxf(acc[i][j][0] + b0, omin), omax);
                if (n + 1 < rows) out[(size_t)m * rows + n + 1] = fminf(fmaxf(acc[i][j][1] + b1, omin), omax);
            }
            if (m + 8 < t) {
                if (n < rows) out[(size_t)(m + 8) * rows + n] = fminf(fmaxf(acc[i][j][2] + b0, omin), omax);
                if (n + 1 < rows) out[(size_t)(m + 8) * rows + n + 1] = fminf(fmaxf(acc[i][j][3] + b1, omin), omax);
            }
        }
    }
}
