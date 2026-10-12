/* ---------- the int4 product with int8 activations, on dp4a (Pascal) ----------
 * k_gemm_dp4a: the product of k_gemm_q8 for a GPU without the tensor cores
 * (caps.tc 0: the 1070, the P102-100, the 1080 Ti). The same inputs: the
 * rows of x as int8 with one float32 scale for each block of 32 (k_quant_q8),
 * or the int16 form (X2, k_quant_x2: the planes hi and lo, q = 128 hi + lo),
 * and the int4 blocks of w (18 bytes: a float16 scale, then 16 bytes; the low
 * 4 bits of byte i are value i, the high 4 bits value i + 16, each less 8).
 *
 * dp4a (sm_61) adds the 4 products of two words of 4 int8 values to an int32,
 * at the rate of a float32 FMA: 4 products for one instruction. The block of
 * the tile is 64 tokens by 128 rows, 256 threads; a thread takes 4 tokens
 * (tg + 16 u) and 8 rows (rg + 16 i). A step reads 4 blocks of 32 columns of
 * x and of w to shared memory; w as words of 4 values of 0 to 15 (the 4 bits
 * as they are), so the inner loop is dp4a alone. The 8 is taken off with the
 * sum of the block of x: sum (v - 8) x = sum v x - 8 sum x. The int32 sum of
 * a block is exact (|sum| < 2^22, i2f_exact); the kernel then multiplies it
 * by the scale of x and of w in float32, as k_gemm_q8 does. The order of the
 * float32 sums is that of the blocks, so the values are close to those of
 * k_gemm_q8, not the same.
 *
 * The rows of w in a step are read as 18-byte blocks, 2 bytes at a time
 * through the read-only cache (a row of w starts at a multiple of 2 bytes
 * only). A step of fewer than 4 blocks (cols % 128) reads zeros. */
#define DP_BM 64
#define DP_BN 128
#define DP_KB 4                    /* the blocks of 32 columns of a step */
#define DP_WS 12                   /* the words of a row of a block in shared memory (8 and 4 of pad) */

__device__ __forceinline__ int rc_dp4a(int a, int b, int c)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610
    return __dp4a(a, b, c);
#else
    const char4 x = *(const char4 *)&a, y = *(const char4 *)&b;
    return c + x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
#endif
}

template <int X2>
__global__ void __launch_bounds__(256, 2) k_gemm_dp4a(const int8_t *xq, const float *xs,
                                                      const uint8_t *w, float *out, int t,
                                                      int rows, int cols, const int8_t *xl)
{
    PDL_START();
    /* x: [block][plane][token][8 words]; w: [block][row][DP_WS words] */
    __shared__ __align__(16) int sx[DP_KB][1 + X2][DP_BM][8];
    __shared__ __align__(16) int sw[DP_KB][DP_BN][DP_WS];
    __shared__ float sxs[DP_KB][DP_BM];          /* the scale of the block of x */
    __shared__ int sxo[DP_KB][DP_BM];            /* 8 times the sum of the block of x */
    __shared__ float sws[DP_KB][DP_BN];          /* the scale of the block of w */
    const int m0 = blockIdx.x * DP_BM, n0 = blockIdx.y * DP_BN;
    const int tid = threadIdx.x, rg = tid % 16, tg = tid / 16;
    const int nb = cols / 32;
    const size_t rb = (size_t)nb * 18;
    float acc[4][8];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[u][i] = 0.f;
        }
    }
    for (int g0 = 0; g0 < nb; g0 += DP_KB) {
        /* x: 64 tokens by 4 blocks, 16 bytes a thread (2 a block of a
         * plane); the pair of threads of a block adds its sum. */
        #pragma unroll
        for (int e0 = 0; e0 < DP_BM * DP_KB * 2; e0 += 256) {
            int e = e0 + tid, m = e / (DP_KB * 2), kb = (e / 2) % DP_KB, half = e % 2;
            int q = min(m0 + m, t - 1), gb = g0 + kb;
            int4 v = make_int4(0, 0, 0, 0), v2 = make_int4(0, 0, 0, 0);
            float s = 0.f;
            if (gb < nb) {
                size_t off = (size_t)q * cols + (size_t)gb * 32 + half * 16;
                v = __ldg((const int4 *)(xq + off));
                if (X2) {
                    v2 = __ldg((const int4 *)(xl + off));
                }
                s = __ldg(xs + (size_t)q * nb + gb);
            }
            *(int4 *)&sx[kb][0][m][half * 4] = v;
            int sum = rc_dp4a(v.x, 0x01010101, rc_dp4a(v.y, 0x01010101,
                      rc_dp4a(v.z, 0x01010101, rc_dp4a(v.w, 0x01010101, 0))));
            if (X2) {
                *(int4 *)&sx[kb][1][m][half * 4] = v2;
                int lo = rc_dp4a(v2.x, 0x01010101, rc_dp4a(v2.y, 0x01010101,
                         rc_dp4a(v2.z, 0x01010101, rc_dp4a(v2.w, 0x01010101, 0))));
                sum = 128 * sum + lo;
            }
            sum += __shfl_xor_sync(RC_FULL_MASK, sum, 1);
            if (half == 0) {
                sxs[kb][m] = s;
                sxo[kb][m] = 8 * sum;
            }
        }
        /* w: 128 rows by 4 blocks, 2 blocks a thread, decoded to words */
        #pragma unroll
        for (int e0 = 0; e0 < DP_BN * DP_KB; e0 += 256) {
            int e = e0 + tid, n = e / DP_KB, kb = e % DP_KB;
            int row = min(n0 + n, rows - 1), gb = g0 + kb;
            int wd[8];
            float s = 0.f;
            if (gb < nb) {
                const unsigned short *p =
                    (const unsigned short *)(w + (size_t)row * rb + (size_t)gb * 18);
                unsigned h[9];
                #pragma unroll
                for (int k = 0; k < 9; ++k) {
                    h[k] = __ldg(p + k);
                }
                s = __half2float(__ushort_as_half((unsigned short)h[0]));
                #pragma unroll
                for (int c = 0; c < 4; ++c) {
                    unsigned b4 = h[1 + 2 * c] | (h[2 + 2 * c] << 16);
                    wd[c] = (int)(b4 & 0x0F0F0F0Fu);
                    wd[4 + c] = (int)((b4 >> 4) & 0x0F0F0F0Fu);
                }
            } else {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    wd[k] = 0;
                }
            }
            *(int4 *)&sw[kb][n][0] = make_int4(wd[0], wd[1], wd[2], wd[3]);
            *(int4 *)&sw[kb][n][4] = make_int4(wd[4], wd[5], wd[6], wd[7]);
            sws[kb][n] = s;
        }
        __syncthreads();
        #pragma unroll 1
        for (int kb = 0; kb < DP_KB; ++kb) {
            int isum[4][8];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    isum[u][i] = 0;
                }
            }
            #pragma unroll 1
            for (int p = 0; p <= X2; ++p) {
                if (X2 && p == 1) {
                    /* q = 128 hi + lo: the sums of hi times 128, then lo */
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            isum[u][i] *= 128;
                        }
                    }
                }
                #pragma unroll
                for (int hw = 0; hw < 2; ++hw) {
                    int4 a[4];
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        a[u] = *(const int4 *)&sx[kb][p][tg + 16 * u][hw * 4];
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        int4 b = *(const int4 *)&sw[kb][rg + 16 * i][hw * 4];
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            isum[u][i] = rc_dp4a(b.x, a[u].x, isum[u][i]);
                            isum[u][i] = rc_dp4a(b.y, a[u].y, isum[u][i]);
                            isum[u][i] = rc_dp4a(b.z, a[u].z, isum[u][i]);
                            isum[u][i] = rc_dp4a(b.w, a[u].w, isum[u][i]);
                        }
                    }
                }
            }
            float ws_[8];
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                ws_[i] = sws[kb][rg + 16 * i];
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float s = sxs[kb][tg + 16 * u];
                int o = sxo[kb][tg + 16 * u];
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[u][i] = fmaf(i2f_exact(isum[u][i] - o), s * ws_[i], acc[u][i]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        int j = m0 + tg + 16 * u;
        if (j >= t) {
            continue;
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int n = n0 + rg + 16 * i;
            if (n < rows) {
                out[(size_t)j * rows + n] = acc[u][i];
            }
        }
    }
}

/* RECLANK_GPU_DP4A=0 keeps the float32 tiles (k_gemm) for the int4 products
 * of a large group on a GPU without the tensor cores (a test). */
static int dp4a_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_DP4A");
        on = !(v && v[0] == '0');
    }
    return on && gg_caps.cc >= 61;
}
