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
