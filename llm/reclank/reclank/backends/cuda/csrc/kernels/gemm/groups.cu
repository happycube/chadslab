/* ---------- token groups: the prompt pass and the MTP group ----------
 * SPLIT_PLAN.md, phase 5. The program of a group of t tokens has the same
 * operations as the program of one token, in their group form. */

/* A matrix of int4 blocks or of float32 values times the rows of x. The
 * result is out[j][n] = sum_k x[j][k] W[n][k], for j < t and n < rows. The
 * array x has t rows of cols values, and out has t rows of rows values.
 *
 * For a small group (t <= 16), k_mt_gemv: one warp for each row of W. The
 * warp reads each block of the row one time and uses it for every token.
 *
 * For a large group, k_gemm: a tile of GM tokens by GN rows for each block.
 * The block reads a step of 32 columns of x and of W to shared memory. Then
 * each thread computes 4 by 4 values of the tile. */
#define GM 64
#define GN 64

template <int F32W>
__device__ __forceinline__ void w_block32(const void *w, int row, int cols, int g, float *out32)
{
    if (F32W == 2) {
        /* bfloat16 rows */
        const uint16_t *p = (const uint16_t *)w + (size_t)row * cols + (size_t)g * 32;
        for (int k = 0; k < 32; ++k) {
            out32[k] = __uint_as_float((uint32_t)p[k] << 16);
        }
    } else if (F32W) {
        const float *p = (const float *)w + (size_t)row * cols + (size_t)g * 32;
        for (int k = 0; k < 32; ++k) {
            out32[k] = p[k];
        }
    } else {
        const uint8_t *blk = (const uint8_t *)w + ((size_t)row * (cols / 32) + g) * 18;
        float d = __half2float(__ushort_as_half((uint16_t)(blk[0] | (blk[1] << 8))));
        for (int k = 0; k < 16; ++k) {
            int b = blk[2 + k];
            out32[k] = (float)((b & 15) - 8) * d;
            out32[k + 16] = (float)((b >> 4) - 8) * d;
        }
    }
}

template <int F32W>
__global__ void k_gemm(const float *x, const void *w, float *out, int t, int rows, int cols)
{
    PDL_START();
    __shared__ float xs[32][GM + 1];
    __shared__ float ws[32][GN + 1];
    int n0 = blockIdx.x * GN, j0 = blockIdx.y * GM;
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4] = {{0.f}};
    int groups = cols / 32;
    for (int g = 0; g < groups; ++g) {
        /* x: GM tokens by 32 columns; each thread loads 8 values. */
        for (int q = threadIdx.x; q < GM * 32; q += blockDim.x) {
            int jj = q / 32, k = q % 32;
            int j = j0 + jj;
            xs[k][jj] = j < t ? x[(size_t)j * cols + (size_t)g * 32 + k] : 0.f;
        }
        /* W: GN rows by 32 columns; one thread for each row. */
        if (threadIdx.x < GN) {
            int n = n0 + threadIdx.x;
            float v[32];
            if (n < rows) {
                w_block32<F32W>(w, n, cols, g, v);
            } else {
                for (int k = 0; k < 32; ++k) {
                    v[k] = 0.f;
                }
            }
            for (int k = 0; k < 32; ++k) {
                ws[k][threadIdx.x] = v[k];
            }
        }
        __syncthreads();
        #pragma unroll 8
        for (int k = 0; k < 32; ++k) {
            float a[4], b[4];
            for (int u = 0; u < 4; ++u) {
                a[u] = xs[k][ty * 4 + u];
                b[u] = ws[k][tx * 4 + u];
            }
            for (int u = 0; u < 4; ++u) {
                for (int v = 0; v < 4; ++v) {
                    acc[u][v] += a[u] * b[v];
                }
            }
        }
        __syncthreads();
    }
    for (int u = 0; u < 4; ++u) {
        int j = j0 + ty * 4 + u;
        if (j >= t) {
            continue;
        }
        for (int v = 0; v < 4; ++v) {
            int n = n0 + tx * 4 + v;
            if (n < rows) {
                out[(size_t)j * rows + n] = acc[u][v];
            }
        }
    }
}

/* k_gemm of an int4 matrix without int8 products (Maxwell: no tensor cores,
 * no dp4a; MAXWELL_PLAN.md, section 3.4): a tile of 128 tokens by 128 rows a
 * block of 256 threads, 8 by 8 values a thread. A step of 32 columns (one
 * int4 block of each row) goes to shared memory as float: x transposed, and
 * w decoded once for the tile by all the threads (2 a row, 16 values each:
 * i4m, then the scale, the values of w_block32) where k_gemm decoded it in 64
 * threads with conversions of integers (a quarter of the rate). A thread
 * reads its 8 values of x and of w as 4 float4 (the halves 0-63 and 64-127 of
 * the tile, so the lanes of a warp read a run: no conflicts): 64 FMA for 4
 * loads of shared memory, where k_gemm (4 by 4, scalar loads) had 16 for 8.
 * The loads of the next step go to registers while the tile works. Each value
 * adds its terms in the order of k_gemm, with the same FMA: the same bits. */
#define G8_B 128
#define G8_LD 132                  /* the floats of a row of the tiles in shared memory (4 of pad) */

__global__ void __launch_bounds__(256, 2) k_gemm8(const float *x, const uint8_t *w, float *out, int t,
                                                  int rows, int cols)
{
    PDL_START();
    __shared__ __align__(16) float xs[32][G8_LD];
    __shared__ __align__(16) float ws[32][G8_LD];
    const int n0 = blockIdx.x * G8_B, j0 = blockIdx.y * G8_B;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int groups = cols / 32;
    /* the loads of a step: x, 4 float4 a thread (token q / 8, columns 4 (q % 8));
     * w, the 8 bytes of codes of half (tid & 1) of the block of row tid / 2,
     * and its scale */
    float4 lx[4];
    uint32_t lw0, lw1;
    float ld;
    const int wrow = tid / 2, half = tid & 1;
    const bool wok = n0 + wrow < rows;
    auto fetch = [&](int g) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int q = tid + 256 * i, j = j0 + q / 8;
            lx[i] = j < t ? *(const float4 *)(x + (size_t)j * cols + (size_t)g * 32 + 4 * (q % 8))
                          : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (wok) {
            const uint8_t *blk = w + ((size_t)(n0 + wrow) * groups + g) * 18;
            const uint16_t *c = (const uint16_t *)(blk + 2 + 8 * half);
            lw0 = (uint32_t)c[0] | ((uint32_t)c[1] << 16);
            lw1 = (uint32_t)c[2] | ((uint32_t)c[3] << 16);
            ld = __half2float(__ushort_as_half(*(const uint16_t *)blk));
        } else {
            lw0 = lw1 = 0x88888888u;    /* (codes of 8: the value 0) */
            ld = 0.f;
        }
    };
    float acc[8][8];
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            acc[u][v] = 0.f;
        }
    }
    fetch(0);
    for (int g = 0; g < groups; ++g) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int q = tid + 256 * i, jj = q / 8, k = 4 * (q % 8);
            xs[k][jj] = lx[i].x;
            xs[k + 1][jj] = lx[i].y;
            xs[k + 2][jj] = lx[i].z;
            xs[k + 3][jj] = lx[i].w;
        }
        {
            /* value k (byte k % 16 of the block: low nibble for k < 16) */
            const int kb = 8 * half;
            const uint32_t h0 = lw0 >> 12, h1 = lw1 >> 12;
            ws[kb + 0][wrow] = i4m<0>(lw0) * ld;
            ws[kb + 1][wrow] = i4m<8>(lw0) * ld;
            ws[kb + 2][wrow] = i4m<16>(lw0) * ld;
            ws[kb + 3][wrow] = i4m<12>(h0) * ld;
            ws[kb + 4][wrow] = i4m<0>(lw1) * ld;
            ws[kb + 5][wrow] = i4m<8>(lw1) * ld;
            ws[kb + 6][wrow] = i4m<16>(lw1) * ld;
            ws[kb + 7][wrow] = i4m<12>(h1) * ld;
            ws[16 + kb + 0][wrow] = i4m<4>(lw0) * ld;
            ws[16 + kb + 1][wrow] = i4m<12>(lw0) * ld;
            ws[16 + kb + 2][wrow] = i4m<8>(h0) * ld;
            ws[16 + kb + 3][wrow] = i4m<16>(h0) * ld;
            ws[16 + kb + 4][wrow] = i4m<4>(lw1) * ld;
            ws[16 + kb + 5][wrow] = i4m<12>(lw1) * ld;
            ws[16 + kb + 6][wrow] = i4m<8>(h1) * ld;
            ws[16 + kb + 7][wrow] = i4m<16>(h1) * ld;
        }
        __syncthreads();
        if (g + 1 < groups) {
            fetch(g + 1);
        }
        #pragma unroll 4
        for (int k = 0; k < 32; ++k) {
            const float4 a0 = *(const float4 *)&xs[k][4 * ty], a1 = *(const float4 *)&xs[k][64 + 4 * ty];
            const float4 b0 = *(const float4 *)&ws[k][4 * tx], b1 = *(const float4 *)&ws[k][64 + 4 * tx];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            const float b[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    acc[u][v] = fmaf(a[u], b[v], acc[u][v]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        int j = j0 + (u < 4 ? 4 * ty + u : 64 + 4 * ty + u - 4);
        if (j >= t) {
            continue;
        }
        #pragma unroll
        for (int hv = 0; hv < 2; ++hv) {
            int n = n0 + 64 * hv + 4 * tx;
            float *o = out + (size_t)j * rows + n;
            if (n + 3 < rows && (rows & 3) == 0) {
                *(float4 *)o = make_float4(acc[u][4 * hv], acc[u][4 * hv + 1], acc[u][4 * hv + 2], acc[u][4 * hv + 3]);
            } else {
                for (int v = 0; v < 4; ++v) {
                    if (n + v < rows) {
                        o[v] = acc[u][4 * hv + v];
                    }
                }
            }
        }
    }
}

/* RECLANK_GPU_GEMM8=0: the int4 matrices of a large group without int8
 * products in k_gemm (4 by 4 a thread), not k_gemm8. */
static int gemm8_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_GEMM8");
        on = !(v && v[0] == '0');
    }
    return on;
}
