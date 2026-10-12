/* ---------- the int4 product on the tensor cores, pipelined ----------
 * k_gemm_tc2: a tile of 128 tokens by 128 rows for each block of 8 warps.
 * Warp w takes 64 tokens by 32 rows: 4 by 4 tiles of the instruction.
 *
 * The block copies each step of 64 columns to shared memory with cp.async.
 * It copies the float16 rows of x and the raw int4 blocks of the rows of w
 * (36 bytes of each row). Two buffers take turns, so the copy of step s + 1 runs
 * while the tensor cores compute step s.
 *
 * A fragment of B comes from the raw bytes in registers. For a 4-bit number
 * n, the float16 bits 0x6400 | n give 1024 + n exactly; less 1032 gives
 * n - 8. Two such values fill one register. Thus the kernel needs no
 * float16 copy of the weights in shared memory. */
#define T2M 128
#define T2N 128
#define T2RB 40      /* bytes of one row of a step: 2 blocks of 18, and 4 to align */

__device__ __forceinline__ void cp_async4(void *smem, const void *gmem)
{
#if RC_SM80
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa), "l"(gmem));
#else
    *(uint32_t *)smem = *(const uint32_t *)gmem;     /* (cp.async: aligned) */
#endif
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem)
{
#if RC_SM80
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
#else
    *(int4 *)smem = *(const int4 *)gmem;
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if RC_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait1()
{
#if RC_SM80
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait0()
{
#if RC_SM80
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

/* Two float16 values n0 - 8 and n1 - 8 from two 4-bit numbers. */
__device__ __forceinline__ uint32_t nib2(uint32_t n0, uint32_t n1)
{
    uint32_t bits = 0x64006400u | n0 | (n1 << 16);
    __half2 h = *(__half2 *)&bits;
    h = __hsub2(h, __float2half2_rn(1032.0f));
    return *(uint32_t *)&h;
}

__global__ void __launch_bounds__(256) k_gemm_tc2(const __half *x, const uint8_t *w, float *out,
                                                  int t, int rows, int cols)
{
    PDL_START();
    __shared__ __align__(16) __half as_[2][T2M][TK + 8];
    __shared__ __align__(16) uint8_t bs[2][T2N][T2RB];
    /* blockIdx.x is the tile of tokens, so the blocks that read the same rows of
     * w run together and the rows come from DRAM one time. */
    int m0 = blockIdx.x * T2M, n0 = blockIdx.y * T2N;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    size_t rb = (size_t)(cols / 32) * 18;
    int steps = cols / TK;
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    /* Copy step st to buffer b. A row past the end reads row 0 of its
     * matrix; the tile does not write its result. */
    auto load = [&](int st, int b) {
        int k0 = st * TK;
        for (int e = threadIdx.x; e < T2M * TK / 8; e += blockDim.x) {
            int m = e / (TK / 8), kk = (e % (TK / 8)) * 8;
            int q = min(m0 + m, t - 1);
            cp_async16(&as_[b][m][kk], x + (size_t)q * cols + k0 + kk);
        }
        for (int e = threadIdx.x; e < T2N * 9; e += blockDim.x) {
            int n = e / 9, wd = e % 9;
            int row = min(n0 + n, rows - 1);
            cp_async4(&bs[b][n][wd * 4], w + (size_t)row * rb + (size_t)(k0 / 32) * 18 + wd * 4);
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
        for (int bk = 0; bk < 2; ++bk) {
            float blk[4][4][4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    blk[i][j][0] = blk[i][j][1] = blk[i][j][2] = blk[i][j][3] = 0.f;
                }
            }
            /* B: the rows wn + 8j + g. Bytes 2c, 2c+1 and 2c+8, 2c+9 of the
             * 16 bytes of values give the fragments. Their low 4 bits are
             * columns 0 to 15, and their high 4 bits columns 16 to 31. */
            uint32_t blo[4][2], bhi[4][2];
            float dsc[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t *blkp = &bs[b][wn + j * 8 + g][bk * 18];
                uint32_t v0 = *(const uint16_t *)(blkp + 2 + 2 * c);
                uint32_t v1 = *(const uint16_t *)(blkp + 2 + 2 * c + 8);
                blo[j][0] = nib2(v0 & 15, (v0 >> 8) & 15);
                blo[j][1] = nib2(v1 & 15, (v1 >> 8) & 15);
                bhi[j][0] = nib2((v0 >> 4) & 15, (v0 >> 12) & 15);
                bhi[j][1] = nib2((v1 >> 4) & 15, (v1 >> 12) & 15);
                const uint8_t *d0 = &bs[b][wn + j * 8 + 2 * c][bk * 18];
                const uint8_t *d1 = &bs[b][wn + j * 8 + 2 * c + 1][bk * 18];
                dsc[j][0] = __half2float(__ushort_as_half((uint16_t)(d0[0] | (d0[1] << 8))));
                dsc[j][1] = __half2float(__ushort_as_half((uint16_t)(d1[0] | (d1[1] << 8))));
            }
            #pragma unroll
            for (int ks = 0; ks < 2; ++ks) {
                int kk = bk * 32 + ks * 16;
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
                        mma16816(blk[i][j], a, ks ? bhi[j] : blo[j]);
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j][0] += dsc[j][0] * blk[i][j][0];
                    acc[i][j][1] += dsc[j][1] * blk[i][j][1];
                    acc[i][j][2] += dsc[j][0] * blk[i][j][2];
                    acc[i][j][3] += dsc[j][1] * blk[i][j][3];
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
