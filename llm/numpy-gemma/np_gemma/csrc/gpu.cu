/* The program interpreter of np_gemma on a CUDA GPU.
 *
 * SPLIT_PLAN.md, phase 3. The CPU interpreter (gemma_run in bf16_linear.c)
 * runs the records of a program in one OpenMP region. This file runs the
 * same records on the GPU. The Python side (np_gemma/gpu.py) copies the
 * weights and the buffers of the program to the GPU. It changes each host
 * address of the records to the device address of the same data.
 *
 * Each kernel gets the address of its record and of the environment in
 * device memory. The kernel reads its operands from the record, as gp_step
 * does on the CPU. Thus the arguments of each launch stay the same from one
 * step to the next. Only the environment changes: the position, and the
 * addresses of buffers that can move, such as the cache. The first run
 * records the launches in a CUDA graph. Each later run writes the
 * environment and starts the graph with one call.
 *
 * The scalar operations (codes 1 to 15) run on the host before the launches.
 * Each of them writes a new slot of the environment, so their order does not
 * depend on the kernels.
 *
 * The kernels do not give the bits of the CPU kernels, because they add the
 * values in a different order. scripts/check_gpu.py compares the two.
 *
 * The first version covers the operations of a decode step of one token of
 * the E4B model, and its output head (gg_q6k_head). The launch of any other
 * operation gives an error.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define GP_NARG 24
#define GP_MAGIC 0x4750524f47303031LL

typedef struct {
    int32_t op;
    int32_t flags;
    uint8_t tag[GP_NARG];
    int64_t v[GP_NARG];
} gp_rec;

enum { GP_T_NONE = 0, GP_T_INT = 1, GP_T_F32 = 2, GP_T_SLOT = 3 };

/* The codes of np_gemma/program.py. */
enum {
    GP_S_MOV = 1, GP_S_ADD = 2, GP_S_SUB = 3, GP_S_MUL = 4, GP_S_MAX = 5,
    GP_S_MIN = 6,
    GP_RMS_NORM = 16, GP_ADD = 17, GP_MUL_S = 18, GP_COPY = 19, GP_GELU = 20,
    GP_MUL = 21,
    GP_INT4_LINEAR = 32, GP_INT4_MULTI4 = 33, GP_BF16_LINEAR = 39,
    GP_QKV_NORM = 54, GP_ROPE = 55, GP_KV_WRITE_HEADS = 56, GP_ATTN_F32H = 57,
};

static cudaStream_t gg_stream;
static char gg_error[256];

#define CK(x)                                                                   \
    do {                                                                        \
        cudaError_t err_ = (x);                                                 \
        if (err_ != cudaSuccess) {                                              \
            snprintf(gg_error, sizeof(gg_error), "%s: %s", #x,                  \
                     cudaGetErrorString(err_));                                 \
            return -1;                                                          \
        }                                                                       \
    } while (0)

/* ---------- the operands ---------- */

__device__ __forceinline__ int64_t di(const gp_rec *r, const int64_t *e, int k)
{
    return r->tag[k] == GP_T_SLOT ? e[r->v[k]] : r->v[k];
}

__device__ __forceinline__ float df(const gp_rec *r, const int64_t *e, int k)
{
    return __uint_as_float((uint32_t)di(r, e, k));
}

#define DP(T, k) ((T *)(intptr_t)di(r, e, (k)))
#define DI(k) ((int)di(r, e, (k)))

/* The sum of v over the block. All threads get the result. The block has at
 * most 1024 threads. */
__device__ float block_sum(float v)
{
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) {
        part[w] = v;
    }
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    v = lane < nw ? part[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, o);
    }
    return v;
}

__device__ float block_max(float v)
{
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    }
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) {
        part[w] = v;
    }
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    v = lane < nw ? part[lane] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    }
    return v;
}

/* ---------- small operations ---------- */

/* x, w (0 for none), out, rows, cols, eps. One block for each row. */
__global__ void k_rms_norm(const gp_rec *r, const int64_t *e)
{
    const float *x = DP(const float, 0) + (size_t)blockIdx.x * DI(4);
    const float *w = DP(const float, 1);
    float *out = DP(float, 2) + (size_t)blockIdx.x * DI(4);
    int cols = DI(4);
    float eps = df(r, e, 5);
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out[i] = w ? x[i] * s * w[i] : x[i] * s;
    }
}

/* a, b, out, n */
__global__ void k_add(const gp_rec *r, const int64_t *e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(3)) {
        DP(float, 2)[i] = DP(const float, 0)[i] + DP(const float, 1)[i];
    }
}

/* x, s (float), out, n */
__global__ void k_mul_s(const gp_rec *r, const int64_t *e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(3)) {
        DP(float, 2)[i] = DP(const float, 0)[i] * df(r, e, 1);
    }
}

/* src, dst, bytes */
__global__ void k_copy(const gp_rec *r, const int64_t *e)
{
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)di(r, e, 2)) {
        DP(uint8_t, 1)[i] = DP(const uint8_t, 0)[i];
    }
}

/* x, out, n. The tanh form of GELU, as ops.gelu_tanh. */
__global__ void k_gelu(const gp_rec *r, const int64_t *e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(2)) {
        float v = DP(const float, 0)[i];
        DP(float, 1)[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
                                                   (v + 0.044715f * v * v * v)));
    }
}

/* a, b, out, rows, cols, b_stride */
__global__ void k_mul(const gp_rec *r, const int64_t *e)
{
    int cols = DI(4);
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(3) * cols) {
        int row = i / cols, c = i % cols;
        DP(float, 2)[i] = DP(const float, 0)[i] * DP(const float, 1)[(size_t)row * di(r, e, 5) + c];
    }
}

/* ---------- matrices ---------- */

/* One row of int4 blocks against x. The row has cols / 32 blocks of 18
 * bytes: a float16 scale, then 16 bytes. The low 4 bits of byte i are weight
 * i, and the high 4 bits are weight i + 16. A value is the 4-bit number minus
 * 8. Four lanes share a block, so the lanes of a warp read adjacent values of
 * x. Return the sum to all lanes of the warp. */
__device__ __forceinline__ float int4_row(const uint8_t *wr, const float *x, int cols)
{
    int lane = threadIdx.x % 32;
    int sub = lane & 3;
    int blocks = cols / 32;
    float sum = 0.f;
    for (int b = lane >> 2; b < blocks; b += 8) {
        const uint8_t *blk = wr + (size_t)b * 18;
        float d = __half2float(__ushort_as_half(*(const uint16_t *)blk));
        const uint16_t *qp = (const uint16_t *)(blk + 2 + 4 * sub);
        uint32_t q = (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
        float4 xl = *(const float4 *)(x + b * 32 + 4 * sub);
        float4 xh = *(const float4 *)(x + b * 32 + 16 + 4 * sub);
        float acc = (float)((int)(q & 15) - 8) * xl.x
                  + (float)((int)((q >> 8) & 15) - 8) * xl.y
                  + (float)((int)((q >> 16) & 15) - 8) * xl.z
                  + (float)((int)((q >> 24) & 15) - 8) * xl.w
                  + (float)((int)((q >> 4) & 15) - 8) * xh.x
                  + (float)((int)((q >> 12) & 15) - 8) * xh.y
                  + (float)((int)((q >> 20) & 15) - 8) * xh.z
                  + (float)((int)((q >> 28) & 15) - 8) * xh.w;
        sum += d * acc;
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    return sum;
}

#define ROWS_PER_BLOCK 8

/* x, w, s, out, rows, cols. One warp for each row. The kernel reads the
 * float16 scale of each block, not s. np_gemma/gpu.py checks that the two
 * scales are equal. */
__global__ void k_int4_linear(const gp_rec *r, const int64_t *e)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int rows = DI(4), cols = DI(5);
    if (row >= rows) {
        return;
    }
    float v = int4_row(DP(const uint8_t, 1) + (size_t)row * (cols / 32) * 18,
                       DP(const float, 0), cols);
    if (threadIdx.x % 32 == 0) {
        DP(float, 3)[row] = v;
    }
}

/* x, cols, then (w, s, out, rows) for each of up to four matrices. A null w
 * skips a matrix. The rows of the four matrices follow each other. */
__global__ void k_int4_multi4(const gp_rec *r, const int64_t *e)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int cols = DI(1);
    int m = 0;
    for (; m < 4; ++m) {
        int rows = DP(const uint8_t, 2 + 4 * m) ? DI(5 + 4 * m) : 0;
        if (row < rows) {
            break;
        }
        row -= rows;
    }
    if (m == 4) {
        return;
    }
    float v = int4_row(DP(const uint8_t, 2 + 4 * m) + (size_t)row * (cols / 32) * 18,
                       DP(const float, 0), cols);
    if (threadIdx.x % 32 == 0) {
        DP(float, 4 + 4 * m)[row] = v;
    }
}

/* x, w, out, rows, cols, tokens (1). A bfloat16 matrix. One warp for each
 * row. Each lane reads 8 values at a time. cols must be a multiple of 8. */
__global__ void k_bf16_linear(const gp_rec *r, const int64_t *e)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int rows = DI(3), cols = DI(4);
    if (row >= rows) {
        return;
    }
    const uint4 *w = (const uint4 *)(DP(const uint16_t, 1) + (size_t)row * cols);
    const float *x = DP(const float, 0);
    float sum = 0.f;
    for (int i = lane; i < cols / 8; i += 32) {
        uint4 q = w[i];
        const float *xi = x + i * 8;
        uint32_t u[4] = {q.x, q.y, q.z, q.w};
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            sum += __uint_as_float(u[k] << 16) * xi[2 * k]
                 + __uint_as_float(u[k] & 0xffff0000u) * xi[2 * k + 1];
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    if (lane == 0) {
        DP(float, 2)[row] = sum;
    }
}

/* ---------- attention ---------- */

/* q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps. One block for
 * each row of head_dim values. A row of v has no weight. */
__global__ void k_qkv_norm(const gp_rec *r, const int64_t *e)
{
    int row = blockIdx.x;
    int qr = DI(2), kr = DI(5), hd = DI(8);
    float *x;
    const float *w;
    if (row < qr) {
        x = DP(float, 0) + (size_t)row * hd;
        w = DP(const float, 1);
    } else if (row < qr + kr) {
        x = DP(float, 3) + (size_t)(row - qr) * hd;
        w = DP(const float, 4);
    } else {
        x = DP(float, 6) + (size_t)(row - qr - kr) * hd;
        w = NULL;
    }
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)hd + df(r, e, 9));
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        x[i] = w ? x[i] * s * w[i] : x[i] * s;
    }
}

/* q, q_rows, q_heads, k, k_rows, k_heads, cos, sin, head_dim. One block for
 * each row. As gemma_rope_body. */
__global__ void k_rope(const gp_rec *r, const int64_t *e)
{
    int row = blockIdx.x;
    int qr = DI(1), hd = DI(8), d = hd / 2;
    float *x;
    int tok;
    if (row < qr) {
        x = DP(float, 0) + (size_t)row * hd;
        tok = row / DI(2);
    } else {
        x = DP(float, 3) + (size_t)(row - qr) * hd;
        tok = (row - qr) / DI(5);
    }
    const float *c = DP(const float, 6) + (size_t)tok * hd;
    const float *s = DP(const float, 7) + (size_t)tok * hd;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float a = x[i], b = x[i + d];
        x[i] = a * c[i] - b * s[i];
        x[i + d] = b * c[i] + a * s[i];
    }
}

/* k, v, kbuf, vbuf, head_stride, pos, tokens, kv_heads, head_dim. One block
 * for each head of each token. */
__global__ void k_kv_write_heads(const gp_rec *r, const int64_t *e)
{
    int kvh = DI(7), hd = DI(8);
    int j = blockIdx.x / kvh, h = blockIdx.x % kvh;
    size_t src = ((size_t)j * kvh + h) * hd;
    size_t dst = (size_t)h * di(r, e, 4) + (size_t)(di(r, e, 5) + j) * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        DP(float, 2)[dst + i] = DP(const float, 0)[src + i];
        DP(float, 3)[dst + i] = DP(const float, 1)[src + i];
    }
}

/* The attention in two kernels, with the method of FlashDecoding. A decode step has
 * few query heads (8 for the E4B model), so one block for each head leaves
 * most of the GPU idle. Here the keys of each head go to ATTN_CHUNKS blocks.
 *
 * The kernel k_attn_part: block (h, c) takes chunk c of the keys of query
 * head h. It computes these values and writes them to the scratch of the
 * program, hd + 2 values for each (h, c):
 *
 * - m, the maximum of the scores of its keys;
 * - l, the sum of exp(score - m);
 * - the sum of exp(score - m) times the value rows.
 *
 * The kernel k_attn_join: block h adds the parts of head h. Part c gets the
 * weight exp(m_c - M), where M is the largest m_c. The kernel then divides by
 * the sum of the weighted l_c.
 *
 * The record: q, k, v, scores, out, q_heads, kv_heads, head_dim, tokens (1),
 * pos, head_stride, window, slide. The keys of the step are rows lo to pos
 * of the cache. The scale is 1, as in gemma_attn_decode_f32_body. */
#define ATTN_CHUNKS 32

__device__ __forceinline__ void attn_range(const gp_rec *r, const int64_t *e, int64_t *p,
                                           int64_t *lo, int *n)
{
    int window = DI(11), slide = DI(12);
    *p = di(r, e, 9);
    *lo = (slide && window > 0) ? *p - window + 1 : 0;
    if (*lo < 0) {
        *lo = 0;
    }
    *n = (int)(*p + 1 - *lo);
}

__global__ void k_attn_part(const gp_rec *r, const int64_t *e, float *part)
{
    int h = blockIdx.x, c = blockIdx.y;
    int qh = DI(5), kvh = DI(6), hd = DI(7), window = DI(11);
    int64_t p, lo;
    int n;
    attn_range(r, e, &p, &lo, &n);
    int len = (n + ATTN_CHUNKS - 1) / ATTN_CHUNKS;
    int j0 = c * len, j1 = min(n, j0 + len);
    int kv = h / (qh / kvh);
    size_t hs = (size_t)di(r, e, 10);
    const float *q = DP(const float, 0) + (size_t)h * hd;
    const float *k = DP(const float, 1) + kv * hs + (size_t)lo * hd;
    const float *v = DP(const float, 2) + kv * hs + (size_t)lo * hd;
    float *sc = DP(float, 3) + (size_t)h * n;
    float *o = part + ((size_t)h * ATTN_CHUNKS + c) * (hd + 2);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    for (int j = j0 + warp; j < j1; j += nw) {
        int64_t kp = lo + j;
        float s = 0.f;
        for (int i = lane; i < hd; i += 32) {
            s += q[i] * k[(size_t)j * hd + i];
        }
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_xor_sync(0xffffffff, s, off);
        }
        if (lane == 0) {
            sc[j] = (kp > p || (window > 0 && p - kp >= window)) ? -INFINITY : s;
        }
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        m = fmaxf(m, sc[j]);
    }
    m = block_max(m);
    float l = 0.f;
    for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
        float x = m == -INFINITY ? 0.f : expf(sc[j] - m);
        sc[j] = x;
        l += x;
    }
    l = block_sum(l);
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int j = j0; j < j1; ++j) {
            acc += sc[j] * v[(size_t)j * hd + i];
        }
        o[i] = acc;
    }
    if (threadIdx.x == 0) {
        o[hd] = m;
        o[hd + 1] = l;
    }
}

__global__ void k_attn_join(const gp_rec *r, const int64_t *e, const float *part)
{
    int h = blockIdx.x, hd = DI(7);
    const float *ph = part + (size_t)h * ATTN_CHUNKS * (hd + 2);
    float M = -INFINITY;
    for (int c = 0; c < ATTN_CHUNKS; ++c) {
        M = fmaxf(M, ph[(size_t)c * (hd + 2) + hd]);
    }
    float wsum = 0.f;
    for (int c = 0; c < ATTN_CHUNKS; ++c) {
        float mc = ph[(size_t)c * (hd + 2) + hd];
        if (mc != -INFINITY) {
            wsum += expf(mc - M) * ph[(size_t)c * (hd + 2) + hd + 1];
        }
    }
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    float *out = DP(float, 4) + (size_t)h * hd;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0.f;
        for (int c = 0; c < ATTN_CHUNKS; ++c) {
            float mc = ph[(size_t)c * (hd + 2) + hd];
            if (mc != -INFINITY) {
                acc += expf(mc - M) * ph[(size_t)c * (hd + 2) + i];
            }
        }
        out[i] = acc * inv;
    }
}

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
__global__ void k_q6k_head(const uint8_t *w, const float *x, float *out, int rows, int cols,
                           float cap)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int l = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    int nb = cols / 256;
    const uint8_t *wr = w + (size_t)row * nb * 210;
    float sum = 0.f;
    for (int b = 0; b < nb; ++b) {
        const uint8_t *ql = wr + (size_t)b * 210;
        const uint8_t *qh = ql + 128;
        const int8_t *sc = (const int8_t *)(ql + 192);
        float d = __half2float(__ushort_as_half((uint16_t)(ql[208] | (ql[209] << 8))));
        const float *xb = x + (size_t)b * 256;
        float acc = 0.f;
        #pragma unroll
        for (int n = 0; n < 2; ++n) {
            int a = ql[64 * n + l], bq = ql[64 * n + l + 32], hq = qh[32 * n + l];
            int is = 8 * n + l / 16;
            const float *xn = xb + 128 * n + l;
            acc += (float)sc[is + 0] * (float)(((a & 15) | ((hq & 3) << 4)) - 32) * xn[0];
            acc += (float)sc[is + 2] * (float)(((bq & 15) | (((hq >> 2) & 3) << 4)) - 32) * xn[32];
            acc += (float)sc[is + 4] * (float)(((a >> 4) | (((hq >> 4) & 3) << 4)) - 32) * xn[64];
            acc += (float)sc[is + 6] * (float)(((bq >> 4) | (((hq >> 6) & 3) << 4)) - 32) * xn[96];
        }
        sum += d * acc;
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    if (l == 0) {
        out[row] = cap > 0.f ? cap * tanhf(sum / cap) : sum;
    }
}

/* ---------- the host side ---------- */

typedef struct {
    int n_env, n_code;
    int64_t *henv;       /* the environment on the host */
    int64_t *denv;       /* the environment on the GPU */
    gp_rec *hcode;       /* the records, with device addresses */
    gp_rec *dcode;
    cudaGraphExec_t exec;
    int use_graph;
    float *part;         /* the scratch of the attention */
} gg_prog;

/* The size of the scratch of the attention: 32 heads of 1024 values. */
#define GG_PART_FLOATS ((size_t)32 * ATTN_CHUNKS * (1024 + 2))

static int64_t hi(const gp_rec *r, const int64_t *e, int k)
{
    return r->tag[k] == GP_T_SLOT ? e[r->v[k]] : r->v[k];
}

/* A value that sets the size of a launch. It must be a literal, because the
 * graph keeps the size of each launch. */
static int64_t hlit(const gp_rec *r, int k, int *bad)
{
    if (r->tag[k] == GP_T_SLOT) {
        *bad = 1;
    }
    return r->v[k];
}

static int64_t cdiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

/* Launch the kernel of one record. Return 0, or -1 for an operation that
 * this file does not have or a size that is not a literal. */
static int gg_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr, const int64_t *denv)
{
    int bad = 0;
    cudaStream_t s = gg_stream;
    const int T = 256;
    switch (r->op) {
    case GP_S_MOV: case GP_S_ADD: case GP_S_SUB: case GP_S_MUL: case GP_S_MAX:
    case GP_S_MIN:
        return 0;
    case GP_RMS_NORM:
        k_rms_norm<<<(unsigned)hlit(r, 3, &bad), T, 0, s>>>(dr, denv);
        break;
    case GP_ADD:
        k_add<<<(unsigned)cdiv(hlit(r, 3, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_MUL_S:
        k_mul_s<<<(unsigned)cdiv(hlit(r, 3, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_COPY:
        k_copy<<<(unsigned)cdiv(hlit(r, 2, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_GELU:
        k_gelu<<<(unsigned)cdiv(hlit(r, 2, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_MUL:
        k_mul<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_INT4_LINEAR:
        k_int4_linear<<<(unsigned)cdiv(hlit(r, 4, &bad), ROWS_PER_BLOCK),
                        32 * ROWS_PER_BLOCK, 0, s>>>(dr, denv);
        break;
    case GP_INT4_MULTI4: {
        int64_t rows = 0;
        for (int m = 0; m < 4; ++m) {
            if (r->v[2 + 4 * m] != 0) {
                rows += hlit(r, 5 + 4 * m, &bad);
            }
        }
        k_int4_multi4<<<(unsigned)cdiv(rows, ROWS_PER_BLOCK), 32 * ROWS_PER_BLOCK, 0, s>>>(
            dr, denv);
        break;
    }
    case GP_BF16_LINEAR:
        if (hlit(r, 5, &bad) != 1 || hlit(r, 4, &bad) % 8 != 0) {
            bad = 1;
        }
        k_bf16_linear<<<(unsigned)cdiv(hlit(r, 3, &bad), ROWS_PER_BLOCK),
                        32 * ROWS_PER_BLOCK, 0, s>>>(dr, denv);
        break;
    case GP_QKV_NORM:
        k_qkv_norm<<<(unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad) + hlit(r, 7, &bad)),
                     128, 0, s>>>(dr, denv);
        break;
    case GP_ROPE:
        k_rope<<<(unsigned)(hlit(r, 1, &bad) + hlit(r, 4, &bad)), 128, 0, s>>>(dr, denv);
        break;
    case GP_KV_WRITE_HEADS:
        k_kv_write_heads<<<(unsigned)(hlit(r, 6, &bad) * hlit(r, 7, &bad)), 128, 0, s>>>(
            dr, denv);
        break;
    case GP_ATTN_F32H:
        if (hlit(r, 8, &bad) != 1) {
            bad = 1;
        }
        if (hlit(r, 5, &bad) * ATTN_CHUNKS * (hlit(r, 7, &bad) + 2) > GG_PART_FLOATS) {
            bad = 1;
            break;
        }
        k_attn_part<<<dim3((unsigned)hlit(r, 5, &bad), ATTN_CHUNKS), 128, 0, s>>>(
            dr, denv, g->part);
        k_attn_join<<<(unsigned)hlit(r, 5, &bad), 128, 0, s>>>(dr, denv, g->part);
        break;
    default:
        snprintf(gg_error, sizeof(gg_error), "no GPU kernel for operation %d", r->op);
        return -1;
    }
    if (bad) {
        snprintf(gg_error, sizeof(gg_error),
                 "operation %d: a size is not a literal, or the form is not supported", r->op);
        return -1;
    }
    return 0;
}

/* Run the scalar operations on the host environment. */
static void gg_scalars(gg_prog *g)
{
    int64_t *e = g->henv;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        switch (r->op) {
        case GP_S_MOV: e[r->v[0]] = hi(r, e, 1); break;
        case GP_S_ADD: e[r->v[0]] = hi(r, e, 1) + hi(r, e, 2); break;
        case GP_S_SUB: e[r->v[0]] = hi(r, e, 1) - hi(r, e, 2); break;
        case GP_S_MUL: e[r->v[0]] = hi(r, e, 1) * hi(r, e, 2); break;
        case GP_S_MAX: {
            int64_t a = hi(r, e, 1), b = hi(r, e, 2);
            e[r->v[0]] = a > b ? a : b;
            break;
        }
        case GP_S_MIN: {
            int64_t a = hi(r, e, 1), b = hi(r, e, 2);
            e[r->v[0]] = a < b ? a : b;
            break;
        }
        default:
            break;
        }
    }
}

extern "C" {

const char *gg_last_error(void)
{
    return gg_error;
}

int gg_init(int device)
{
    CK(cudaSetDevice(device));
    if (gg_stream == NULL) {
        CK(cudaStreamCreateWithFlags(&gg_stream, cudaStreamNonBlocking));
    }
    return 0;
}

int gg_mem_info(size_t *free_b, size_t *total_b)
{
    CK(cudaMemGetInfo(free_b, total_b));
    return 0;
}

void *gg_malloc(size_t n)
{
    void *p = NULL;
    if (cudaMalloc(&p, n) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "cudaMalloc of %zu bytes failed", n);
        return NULL;
    }
    return p;
}

int gg_free(void *p)
{
    CK(cudaFree(p));
    return 0;
}

/* Copies on the stream of the programs. A copy to the host waits for the
 * stream. */
int gg_h2d(void *d, const void *h, size_t n)
{
    CK(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, gg_stream));
    return 0;
}

int gg_d2h(void *h, const void *d, size_t n)
{
    CK(cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, gg_stream));
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

int gg_sync(void)
{
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

/* Load a program whose addresses are device addresses. Return a handle, or
 * NULL. use_graph 0 launches the kernels one at a time in each run. */
void *gg_load(const int64_t *prog, int use_graph)
{
    if (prog[0] != GP_MAGIC) {
        snprintf(gg_error, sizeof(gg_error), "not a program");
        return NULL;
    }
    gg_prog *g = (gg_prog *)calloc(1, sizeof(gg_prog));
    g->n_env = (int)prog[1];
    g->n_code = (int)prog[2];
    g->use_graph = use_graph;
    size_t env_b = (size_t)(g->n_env > 0 ? g->n_env : 1) * sizeof(int64_t);
    size_t code_b = (size_t)g->n_code * sizeof(gp_rec);
    g->hcode = (gp_rec *)malloc(code_b);
    memcpy(g->hcode, prog + 4 + g->n_env, code_b);
    if (cudaMallocHost(&g->henv, env_b) != cudaSuccess ||
        cudaMalloc(&g->denv, env_b) != cudaSuccess ||
        cudaMalloc(&g->dcode, code_b) != cudaSuccess ||
        cudaMemcpy(g->dcode, g->hcode, code_b, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMalloc(&g->part, GG_PART_FLOATS * sizeof(float)) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "gg_load: no memory");
        return NULL;
    }
    memcpy(g->henv, prog + 4, (size_t)g->n_env * sizeof(int64_t));
    return g;
}

/* Run a program. env holds the values of the slots after the bind; the run
 * computes the scalar slots itself. The run does not wait for the GPU. */
int gg_run(void *handle, const int64_t *env)
{
    gg_prog *g = (gg_prog *)handle;
    memcpy(g->henv, env, (size_t)g->n_env * sizeof(int64_t));
    gg_scalars(g);
    CK(cudaMemcpyAsync(g->denv, g->henv, (size_t)g->n_env * sizeof(int64_t),
                       cudaMemcpyHostToDevice, gg_stream));
    if (g->use_graph && g->exec != NULL) {
        CK(cudaGraphLaunch(g->exec, gg_stream));
        return 0;
    }
    if (g->use_graph) {
        CK(cudaStreamBeginCapture(gg_stream, cudaStreamCaptureModeThreadLocal));
    }
    int rc = 0;
    for (int pc = 0; pc < g->n_code && rc == 0; ++pc) {
        rc = gg_launch(g, g->hcode + pc, g->dcode + pc, g->denv);
    }
    if (g->use_graph) {
        cudaGraph_t graph;
        cudaError_t err = cudaStreamEndCapture(gg_stream, &graph);
        if (rc != 0) {
            return rc;
        }
        CK(err);
        CK(cudaGraphInstantiate(&g->exec, graph, 0));
        CK(cudaGraphDestroy(graph));
        CK(cudaGraphLaunch(g->exec, gg_stream));
        return 0;
    }
    if (rc == 0) {
        CK(cudaGetLastError());
    }
    return rc;
}

/* Run a program without the graph and measure each record. The array ms
 * gets the time of each record on the GPU, in milliseconds. This is for a profile:
 * the events add time between the kernels. */
int gg_profile(void *handle, const int64_t *env, float *ms)
{
    gg_prog *g = (gg_prog *)handle;
    memcpy(g->henv, env, (size_t)g->n_env * sizeof(int64_t));
    gg_scalars(g);
    CK(cudaMemcpyAsync(g->denv, g->henv, (size_t)g->n_env * sizeof(int64_t),
                       cudaMemcpyHostToDevice, gg_stream));
    cudaEvent_t *ev = (cudaEvent_t *)malloc(sizeof(cudaEvent_t) * (size_t)(g->n_code + 1));
    for (int pc = 0; pc <= g->n_code; ++pc) {
        CK(cudaEventCreate(&ev[pc]));
    }
    int rc = 0;
    CK(cudaEventRecord(ev[0], gg_stream));
    for (int pc = 0; pc < g->n_code && rc == 0; ++pc) {
        rc = gg_launch(g, g->hcode + pc, g->dcode + pc, g->denv);
        CK(cudaEventRecord(ev[pc + 1], gg_stream));
    }
    CK(cudaStreamSynchronize(gg_stream));
    for (int pc = 0; pc < g->n_code; ++pc) {
        CK(cudaEventElapsedTime(&ms[pc], ev[pc], ev[pc + 1]));
    }
    for (int pc = 0; pc <= g->n_code; ++pc) {
        cudaEventDestroy(ev[pc]);
    }
    free(ev);
    return rc;
}

/* The output head: out = the Q6_K matrix w times x, with the soft cap cap
 * (0 for none). All pointers are device addresses. The call does not wait
 * for the GPU. */
int gg_q6k_head(const void *w, const float *x, float *out, int rows, int cols, float cap)
{
    k_q6k_head<<<(unsigned)cdiv(rows, ROWS_PER_BLOCK), 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
        (const uint8_t *)w, x, out, rows, cols, cap);
    CK(cudaGetLastError());
    return 0;
}

int gg_unload(void *handle)
{
    gg_prog *g = (gg_prog *)handle;
    if (g->exec != NULL) {
        cudaGraphExecDestroy(g->exec);
    }
    cudaFree(g->denv);
    cudaFree(g->dcode);
    cudaFree(g->part);
    cudaFreeHost(g->henv);
    free(g->hcode);
    free(g);
    return 0;
}

}  /* extern "C" */
