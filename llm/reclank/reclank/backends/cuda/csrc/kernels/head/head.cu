/* ---------- the output head ---------- */

/* x (cols), a Q6_K matrix, out (rows). One warp for each row. A Q6_K block
 * holds 256 weights in 210 bytes: ql[128], qh[64], scales[16] (int8), and a
 * float16 scale d. The function dequantize_row_q6_K of ggml gives weight y
 * of the block. For half n (0 or 1) and l from 0 to 31:
 *
 *     y[128n + l]      = d sc[8n + l/16 + 0] (ql[64n + l]      & 15 | (qh[32n + l] >> 0 & 3) << 4) - 32
 *     y[128n + l + 32] = d sc[8n + l/16 + 2] (ql[64n + l + 32] & 15 | (qh[32n + l] >> 2 & 3) << 4) - 32
 *     y[128n + l + 64] = d sc[8n + l/16 + 4] (ql[64n + l]      >> 4 | (qh[32n + l] >> 4 & 3) << 4) - 32
 *     y[128n + l + 96] = d sc[8n + l/16 + 6] (ql[64n + l + 32] >> 4 | (qh[32n + l] >> 6 & 3) << 4) - 32
 *
 * Lane l computes these eight values of each block. With cap > 0, the
 * result is cap tanh(out / cap), the soft cap of the logits. */
#define HEAD_MAX 16

/* A Q6_K value m - 32 (m the 6 bits, 0 to 63) as a float without a
 * conversion of an integer (a quarter of the rate on Maxwell and Pascal):
 * the bits of 2^23 + m, less 2^23 + 32. Exact, so the bits do not change
 * (MAXWELL_PLAN.md, section 3.2: the head reads 0.74 G weights a token). */
__device__ __forceinline__ float q6v(int m)
{
    return __uint_as_float((uint32_t)m | 0x4B000000u) - 8388640.f;
}

/* NX 0: nx rows, up to HEAD_MAX. NX 1: one row (the decode step); the
 * compiler then keeps one sum and unrolls the loops. */
template <int NX>
__global__ void k_q6k_head(const uint8_t *w, const float *x, float *out, int rows, int cols,
                           float cap, int nx)
{
    constexpr int NS = NX ? NX : HEAD_MAX;
    if (NX) {
        nx = NX;
    }
    PDL_START();
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int l = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    int nb = cols / 256;
    const uint8_t *wr = w + (size_t)row * nb * 210;
    float sum[NS];
    for (int k = 0; k < NS; ++k) {
        sum[k] = 0.f;
    }
    #pragma unroll 2
    for (int b = 0; b < nb; ++b) {
        const uint8_t *ql = wr + (size_t)b * 210;
        const uint8_t *qh = ql + 128;
        const int8_t *sc = (const int8_t *)(ql + 192);
        float d = __half2float(__ushort_as_half((uint16_t)(ql[208] | (ql[209] << 8))));
        float wv[8];
        int idx[8];
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            int a = ql[64 * n + l], bq = ql[64 * n + l + 32], hq = qh[32 * n + l];
            int is = 8 * n + l / 16;
            wv[4 * n + 0] = d * (float)sc[is + 0] * q6v(((a & 15) | ((hq & 3) << 4)));
            wv[4 * n + 1] = d * (float)sc[is + 2] * q6v(((bq & 15) | (((hq >> 2) & 3) << 4)));
            wv[4 * n + 2] = d * (float)sc[is + 4] * q6v(((a >> 4) | (((hq >> 4) & 3) << 4)));
            wv[4 * n + 3] = d * (float)sc[is + 6] * q6v(((bq >> 4) | (((hq >> 6) & 3) << 4)));
            idx[4 * n + 0] = b * 256 + 128 * n + l;
            idx[4 * n + 1] = idx[4 * n + 0] + 32;
            idx[4 * n + 2] = idx[4 * n + 0] + 64;
            idx[4 * n + 3] = idx[4 * n + 0] + 96;
        }
        for (int k = 0; k < NS; ++k) {
            if (k < nx) {
                const float *xk = x + (size_t)k * cols;
                float acc = 0.f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc += wv[u] * xk[idx[u]];
                }
                sum[k] += acc;
            }
        }
    }
    for (int k = 0; k < NS; ++k) {
        if (k < nx) {
            float v = sum[k];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (l == 0) {
                out[(size_t)k * rows + row] = cap > 0.f ? cap * tanhf(v / cap) : v;
            }
        }
    }
}

/* k_q6k_head for rows of NB blocks: the warp first copies its row to shared
 * memory with 16-byte loads (the 16-byte span of the head that holds it: a row
 * of 11 blocks, the 26B's 2310 bytes, starts anywhere in a span; the span past
 * the last row stays in the allocation of the head, a cudaMalloc), then each
 * lane does the work of k_q6k_head from there, in the same order (the same
 * bits). The byte loads of k_q6k_head read the 417 MB head of Qwen3.6 at 68
 * GB/s on the GTX 1070 (6.1 ms a step); staged, 174 GB/s (2.4 ms). */
template <int NX, int NB>
__global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) k_q6k_head_s(const uint8_t *w, const float *x,
                                                                   float *out, int rows, int cols,
                                                                   float cap, int nx)
{
    constexpr int NS = NX ? NX : HEAD_MAX;
    constexpr int RB = NB * 210, NV = RB / 16 + 2, PER = (NV + 31) / 32;
    __shared__ __align__(16) uint8_t sm[ROWS_PER_BLOCK][NV * 16];
    if (NX) {
        nx = NX;
    }
    PDL_START();
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int l = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    size_t start = (size_t)row * RB, a0 = start & ~(size_t)15;
    int off = (int)(start - a0), nv = (off + RB + 15) / 16;
    uint8_t *buf = sm[threadIdx.x / 32];
    const uint4 *src = (const uint4 *)(w + a0);
    #pragma unroll
    for (int p = 0; p < PER; ++p) {
        int i = l + 32 * p;
        if (i < nv) {
            ((uint4 *)buf)[i] = __ldg(src + i);
        }
    }
    const uint8_t *wr = buf + off;
    __syncwarp();
    float sum[NS];
    for (int k = 0; k < NS; ++k) {
        sum[k] = 0.f;
    }
    #pragma unroll 2
    for (int b = 0; b < NB; ++b) {
        const uint8_t *ql = wr + b * 210;
        const uint8_t *qh = ql + 128;
        const int8_t *sc = (const int8_t *)(ql + 192);
        float d = __half2float(__ushort_as_half((uint16_t)(ql[208] | (ql[209] << 8))));
        float wv[8];
        int idx[8];
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            int a = ql[64 * n + l], bq = ql[64 * n + l + 32], hq = qh[32 * n + l];
            int is = 8 * n + l / 16;
            wv[4 * n + 0] = d * (float)sc[is + 0] * q6v(((a & 15) | ((hq & 3) << 4)));
            wv[4 * n + 1] = d * (float)sc[is + 2] * q6v(((bq & 15) | (((hq >> 2) & 3) << 4)));
            wv[4 * n + 2] = d * (float)sc[is + 4] * q6v(((a >> 4) | (((hq >> 4) & 3) << 4)));
            wv[4 * n + 3] = d * (float)sc[is + 6] * q6v(((bq >> 4) | (((hq >> 6) & 3) << 4)));
            idx[4 * n + 0] = b * 256 + 128 * n + l;
            idx[4 * n + 1] = idx[4 * n + 0] + 32;
            idx[4 * n + 2] = idx[4 * n + 0] + 64;
            idx[4 * n + 3] = idx[4 * n + 0] + 96;
        }
        for (int k = 0; k < NS; ++k) {
            if (k < nx) {
                const float *xk = x + (size_t)k * cols;
                float acc = 0.f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc += wv[u] * xk[idx[u]];
                }
                sum[k] += acc;
            }
        }
    }
    for (int k = 0; k < NS; ++k) {
        if (k < nx) {
            float v = sum[k];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (l == 0) {
                out[(size_t)k * rows + row] = cap > 0.f ? cap * tanhf(v / cap) : v;
            }
        }
    }
}

/* The output head of a Q4_0 table (a GGUF file that keeps the tied head in
 * Q4_0): the work of k_q6k_head. A row is cols / 32 blocks of 18 bytes: a
 * float16 scale d and 16 bytes. Byte j holds value j (the low nibble) and
 * value j + 16 (the high nibble), each the nibble minus 8. A warp reads four
 * blocks at a time: lane l takes bytes 2 (l % 8) and 2 (l % 8) + 1 of block
 * 4 i + l / 8. A block starts at an even address, so the two bytes are one
 * 16-bit load, and the four x values are two float2 loads (x is 8-byte
 * aligned). This layout reads the weights in long runs; a lane for each
 * value of a block (coalesced x) took 1.48 ms in place of 1.00 ms for one
 * row of the 26B head on an RTX 5060 Ti. A warp does the R rows of w from
 * row0 (U: the unroll of the loop over the blocks). Each sum adds its terms
 * in the same order for all R and NX, so a group gives the bits of a step. */
template <int NX, int R, int U>
__global__ void k_q4_head(const uint8_t *w, const float *x, float *out, int rows, int cols,
                          float cap, int nx)
{
    constexpr int NS = NX ? NX : HEAD_MAX;
    if (NX) {
        nx = NX;
    }
    PDL_START();
    int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32) * R;
    int l = threadIdx.x % 32;
    if (row0 >= rows) {
        return;
    }
    int nb = cols / 32;
    int j = 2 * (l % 8);
    const uint8_t *wr[R];
    for (int r = 0; r < R; ++r) {
        /* A row past the end reads the last row; its sum is not written. */
        wr[r] = w + (size_t)min(row0 + r, rows - 1) * nb * 18;
    }
    float sum[R][NS];
    for (int r = 0; r < R; ++r) {
        for (int k = 0; k < NS; ++k) {
            sum[r][k] = 0.f;
        }
    }
    #pragma unroll (NX ? U : 1)
    for (int b = l / 8; b < nb; b += 4) {
        float w0[R], w1[R], w2[R], w3[R];
        for (int r = 0; r < R; ++r) {
            const uint8_t *blk = wr[r] + (size_t)b * 18;
            float d = __half2float(__ushort_as_half(*(const uint16_t *)blk));
            unsigned q = *(const uint16_t *)(blk + 2 + j);
            w0[r] = d * (float)((int)(q & 15) - 8);          /* value j */
            w1[r] = d * (float)((int)((q >> 8) & 15) - 8);   /* value j + 1 */
            w2[r] = d * (float)((int)((q >> 4) & 15) - 8);   /* value j + 16 */
            w3[r] = d * (float)((int)(q >> 12) - 8);         /* value j + 17 */
        }
        int c = b * 32 + j;
        for (int k = 0; k < NS; ++k) {
            if (k < nx) {
                const float *xk = x + (size_t)k * cols + c;
                float2 lo = *(const float2 *)xk, hi = *(const float2 *)(xk + 16);
                for (int r = 0; r < R; ++r) {
                    sum[r][k] += w0[r] * lo.x + w1[r] * lo.y + w2[r] * hi.x + w3[r] * hi.y;
                }
            }
        }
    }
    for (int r = 0; r < R; ++r) {
        for (int k = 0; k < NS; ++k) {
            if (k < nx) {
                float v = sum[r][k];
                for (int o = 16; o > 0; o >>= 1) {
                    v += __shfl_xor_sync(RC_FULL_MASK, v, o);
                }
                if (l == 0 && row0 + r < rows) {
                    out[(size_t)k * rows + row0 + r] = cap > 0.f ? cap * tanhf(v / cap) : v;
                }
            }
        }
    }
}

/* The candidates of sampling (reclank/core/sampling.py, Sampler.sample_sparse):
 * the K largest logits of each row (a block for each row), the row max, and
 * the sum of exp((l - max) inv_t). The host then copies K ids and values in
 * place of the whole row. A radix select of 8 bits at a time finds the K-th
 * largest key (the bits of the float in an order of the values); the values
 * above it, and the values equal to it up to K, go out in no order. */
#define TOPK_T 1024

__device__ __forceinline__ uint32_t topk_key(float f)
{
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ float topk_block_reduce(float v, float *red, bool is_max)
{
    for (int o = 16; o > 0; o >>= 1) {
        float w = __shfl_xor_sync(RC_FULL_MASK, v, o);
        v = is_max ? fmaxf(v, w) : v + w;
    }
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    __syncthreads();
    if (lane == 0) {
        red[warp] = v;
    }
    __syncthreads();
    if (warp == 0) {
        v = lane < (int)(blockDim.x / 32) ? red[lane] : (is_max ? -INFINITY : 0.f);
        for (int o = 16; o > 0; o >>= 1) {
            float w = __shfl_xor_sync(RC_FULL_MASK, v, o);
            v = is_max ? fmaxf(v, w) : v + w;
        }
        if (lane == 0) {
            red[0] = v;
        }
    }
    __syncthreads();
    return red[0];
}

__global__ void __launch_bounds__(TOPK_T) k_topk_rows(const float *x, int vocab, int K, float inv_t,
                                                      int *ids, float *vals, float *stat)
{
    __shared__ int hist[256];
    __shared__ float red[32];
    __shared__ uint32_t prefix_s;
    __shared__ int need_s, cnt_gt, cnt_eq;
    const float *row = x + (size_t)blockIdx.x * vocab;
    int *oid = ids + (size_t)blockIdx.x * K;
    float *oval = vals + (size_t)blockIdx.x * K;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        mx = fmaxf(mx, row[i]);
    }
    mx = topk_block_reduce(mx, red, true);
    float sum = 0.f;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        sum += expf((row[i] - mx) * inv_t);
    }
    sum = topk_block_reduce(sum, red, false);
    uint32_t prefix = 0, mask = 0;
    int need = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            hist[i] = 0;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            uint32_t k = topk_key(row[i]);
            if ((k & mask) == prefix) {
                atomicAdd(&hist[(k >> shift) & 255], 1);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int acc = 0;
            for (int b = 255; b >= 0; --b) {
                if (acc + hist[b] >= need) {
                    prefix_s = prefix | ((uint32_t)b << shift);
                    need_s = need - acc;
                    break;
                }
                acc += hist[b];
            }
            cnt_gt = 0;
            cnt_eq = 0;
        }
        __syncthreads();
        prefix = prefix_s;
        need = need_s;
        mask |= 0xFFu << shift;
    }
    /* prefix is the key of the K-th largest value; need of the values equal
     * to it take the last slots. */
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        uint32_t k = topk_key(row[i]);
        if (k > prefix) {
            int p = atomicAdd(&cnt_gt, 1);
            oid[p] = i;
            oval[p] = row[i];
        } else if (k == prefix) {
            int p = atomicAdd(&cnt_eq, 1);
            if (p < need) {
                oid[K - need + p] = i;
                oval[K - need + p] = row[i];
            }
        }
    }
    if (threadIdx.x == 0) {
        stat[2 * blockIdx.x] = mx;
        stat[2 * blockIdx.x + 1] = sum;
    }
}
