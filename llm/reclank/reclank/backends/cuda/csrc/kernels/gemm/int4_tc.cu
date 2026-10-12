/* ---------- the int4 product on the tensor cores ----------
 * k_gemm_tc: the product of an int4 matrix and a group of rows of x, with
 * the instruction mma.sync m16n8k16 (float16 inputs, float32 sums).
 *
 * The weights: a value of an int4 block is the 4-bit number minus 8, times
 * the float16 scale of the block. The number minus 8 is exact in float16.
 * The kernel multiplies by the scale in float32, after the two steps of 16
 * columns of a block. Thus the weights lose no precision.
 *
 * The rows of x become float16 values. That rounds each value to 11
 * significant bits.
 *
 * A block takes a tile of tokens by TN rows. It steps over the columns 64
 * at a time (2 int4 blocks). Warp w takes 32 tokens by 32 rows of the tile:
 * 2 by 4 tiles of the instruction. A dense product uses tiles of 128 tokens
 * (8 warps). The experts use tiles of 64 pairs (4 warps). With gather,
 * token q of the tile is row pair_tok[q] of x (see k_moe_gemm).
 *
 * The fragments of mma.sync m16n8k16 (g = lane / 4, c = lane % 4):
 *
 *     A (16 x 16, rows):  a0 = A[g][2c..], a1 = A[g+8][2c..],
 *                         a2 = A[g][2c+8..], a3 = A[g+8][2c+8..]
 *     B (16 x 8, cols):   b0 = B[2c..][g], b1 = B[2c+8..][g]
 *     C (16 x 8):         c0, c1 = C[g][2c, 2c+1], c2, c3 = C[g+8][2c, 2c+1]
 */
#define TM 64
#define TN 64
#define TK 64

__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, const uint32_t *b)
{
#if RC_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

/* One tile of WM * 32 tokens by TN rows, with WM * 2 warps. The array x holds
 * float16 values (see k_to_half), cols in each row. The pointer w points at
 * the int4 blocks of row 0 of the matrix, which has rows rows. The tile
 * takes the tokens q0 to q1 - 1: rows of x, or rows pair_tok[q] with gather.
 * The array out has ostride values in each row. The tile writes rows n0 to
 * n0 + TN - 1.
 *
 * Each thread of the block dequantizes parts of 8 bytes of the int4 blocks,
 * so every thread shares that work. */
template <int WM>
__device__ __forceinline__ void tc_tile(const __half *x, const uint8_t *w, float *out,
                                        int q0, int q1, int n0, int rows, int cols,
                                        size_t ostride, const int *gather)
{
    const int TMv = WM * 32;
    __shared__ __align__(16) __half as_[WM * 32][TK + 8];
    __shared__ __align__(16) __half bs[TN][TK + 8];
    __shared__ float ds[TN][TK / 32];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 2) * 32, wn = (warp % 2) * 32;
    int g = lane / 4, c = lane % 4;
    float acc[2][4][4];
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 4; ++j) {
            for (int k = 0; k < 4; ++k) {
                acc[i][j][k] = 0.f;
            }
        }
    }
    size_t rb = (size_t)(cols / 32) * 18;
    for (int k0 = 0; k0 < cols; k0 += TK) {
        /* x: TMv tokens by TK columns, 8 values at a time. */
        for (int e = threadIdx.x; e < TMv * TK / 8; e += blockDim.x) {
            int m = e / (TK / 8), kk = (e % (TK / 8)) * 8;
            int q = q0 + m;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (q < q1) {
                int src = gather ? gather[q] : q;
                v = *(const uint4 *)(x + (size_t)src * cols + k0 + kk);
            }
            *(uint4 *)&as_[m][kk] = v;
        }
        /* w: TN rows by TK / 32 blocks; a unit is half of the 16 bytes of
         * values of a block: 8 bytes give 8 low and 8 high values. */
        for (int e = threadIdx.x; e < TN * (TK / 32) * 2; e += blockDim.x) {
            int n = e / ((TK / 32) * 2), rem = e % ((TK / 32) * 2);
            int bk = rem / 2, hf = rem % 2;
            int row = n0 + n;
            __half2 *dst_lo = (__half2 *)&bs[n][bk * 32 + 8 * hf];
            __half2 *dst_hi = (__half2 *)&bs[n][bk * 32 + 16 + 8 * hf];
            if (row < rows) {
                const uint8_t *blk = w + (size_t)row * rb + (size_t)(k0 / 32 + bk) * 18;
                if (hf == 0) {
                    ds[n][bk] = __half2float(__ushort_as_half((uint16_t)(blk[0] | (blk[1] << 8))));
                }
                const uint8_t *qb = blk + 2 + 8 * hf;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int b0 = qb[2 * i], b1 = qb[2 * i + 1];
                    dst_lo[i] = __halves2half2(__int2half_rn((b0 & 15) - 8), __int2half_rn((b1 & 15) - 8));
                    dst_hi[i] = __halves2half2(__int2half_rn((b0 >> 4) - 8), __int2half_rn((b1 >> 4) - 8));
                }
            } else {
                if (hf == 0) {
                    ds[n][bk] = 0.f;
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    dst_lo[i] = __float2half2_rn(0.f);
                    dst_hi[i] = __float2half2_rn(0.f);
                }
            }
        }
        __syncthreads();
        for (int bk = 0; bk < TK / 32; ++bk) {
            float blk[2][4][4];
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 4; ++j) {
                    for (int k = 0; k < 4; ++k) {
                        blk[i][j][k] = 0.f;
                    }
                }
            }
            #pragma unroll
            for (int ks = 0; ks < 32; ks += 16) {
                int kk = bk * 32 + ks;
                uint32_t a[2][4], b[4][2];
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    int r0 = wm + i * 16;
                    a[i][0] = *(const uint32_t *)&as_[r0 + g][kk + 2 * c];
                    a[i][1] = *(const uint32_t *)&as_[r0 + g + 8][kk + 2 * c];
                    a[i][2] = *(const uint32_t *)&as_[r0 + g][kk + 2 * c + 8];
                    a[i][3] = *(const uint32_t *)&as_[r0 + g + 8][kk + 2 * c + 8];
                }
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int n = wn + j * 8 + g;
                    b[j][0] = *(const uint32_t *)&bs[n][kk + 2 * c];
                    b[j][1] = *(const uint32_t *)&bs[n][kk + 2 * c + 8];
                }
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        mma16816(blk[i][j], a[i], b[j]);
                    }
                }
            }
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                float d0 = ds[wn + j * 8 + 2 * c][bk], d1 = ds[wn + j * 8 + 2 * c + 1][bk];
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    acc[i][j][0] += d0 * blk[i][j][0];
                    acc[i][j][1] += d1 * blk[i][j][1];
                    acc[i][j][2] += d0 * blk[i][j][2];
                    acc[i][j][3] += d1 * blk[i][j][3];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = n0 + wn + j * 8 + 2 * c;
            int m = q0 + wm + i * 16 + g;
            if (m < q1) {
                if (n < rows) out[(size_t)m * ostride + n] = acc[i][j][0];
                if (n + 1 < rows) out[(size_t)m * ostride + n + 1] = acc[i][j][1];
            }
            if (m + 8 < q1) {
                if (n < rows) out[(size_t)(m + 8) * ostride + n] = acc[i][j][2];
                if (n + 1 < rows) out[(size_t)(m + 8) * ostride + n + 1] = acc[i][j][3];
            }
        }
    }
}

/* The rows of a dense product: tiles of 128 tokens (8 warps). */
#define TMD 128

__global__ void k_gemm_tc(const __half *x, const uint8_t *w, float *out, int t, int rows, int cols)
{
    PDL_START();
    int q0 = blockIdx.y * TMD;
    tc_tile<4>(x, w, out, q0, min(t, q0 + TMD), blockIdx.x * TN, rows, cols, (size_t)rows, NULL);
}
