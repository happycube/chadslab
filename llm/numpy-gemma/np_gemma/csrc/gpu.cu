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
 * The file covers the operations of a decode step of one token of the E4B
 * model, and of the 26B model without the experts. The output head is
 * gg_q6k_head. The launch of any other operation gives an error.
 *
 * A step of the 26B keeps its experts on the CPU (SPLIT_PLAN.md, phase 4).
 * Three records of the program move the work to the CPU and back:
 *
 * - GP_TO_HOST copies the input of the experts to pinned host memory and
 *   records an event.
 * - GP_CPU_JOIN waits for that event and runs a CPU program (gemma_run of
 *   the CPU library) that computes the experts.
 * - GP_TO_DEV copies the output of the experts back to the GPU.
 *
 * The runner launches the kernels after a GP_TO_HOST before it runs the
 * GP_CPU_JOIN. Thus the GPU computes the dense feed-forward part while the
 * CPU computes the experts. The kernels between two such records are one
 * segment, and each segment has its own CUDA graph.
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
    GP_INT4_LINEAR = 32, GP_INT4_MULTI4 = 33, GP_RMS_NORM_MULTI4 = 34,
    GP_GELU_MUL_INT4 = 35, GP_BF16_LINEAR = 39,
    GP_QKV_NORM_ROPE = 48, GP_KV_WRITE = 49, GP_ATTN_F32 = 51,
    GP_QKV_NORM = 54, GP_ROPE = 55, GP_KV_WRITE_HEADS = 56, GP_ATTN_F32H = 57,
    GP_ROUTER = 64,
    GP_TO_HOST = 84, GP_CPU_JOIN = 85, GP_TO_DEV = 86, GP_HOT_SPLIT = 87,
    GP_HOT_MOE = 88,
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

/* The norm of each row of cols values: out = x s w, where s = 1 / sqrt(mean
 * of x^2 + eps). A null w gives out = x s. One block for each row. The
 * arguments give the operands of x, w, out, cols, and eps. Thus the kernel
 * serves GP_RMS_NORM and the first step of GP_RMS_NORM_MULTI4. */
__global__ void k_rms_norm(const gp_rec *r, const int64_t *e, int xk, int wk, int ok,
                           int ck, int ek)
{
    int cols = DI(ck);
    const float *x = DP(const float, xk) + (size_t)blockIdx.x * cols;
    const float *w = DP(const float, wk);
    float *out = DP(float, ok) + (size_t)blockIdx.x * cols;
    float eps = df(r, e, ek);
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

/* g, u, inner, scratch: scratch = gelu(g) u, the first step of
 * GP_GELU_MUL_INT4. */
__global__ void k_gelu_mul(const gp_rec *r, const int64_t *e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(2)) {
        float v = DP(const float, 0)[i];
        DP(float, 3)[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
                                                   (v + 0.044715f * v * v * v)))
                          * DP(const float, 1)[i];
    }
}

/* k, v, kd, vd, kqd, ksd, vqd, vsd, n: store n values of the key and of the
 * value at kd and vd. The GPU keeps a float cache only, so the addresses of
 * the int16 copy are null. */
__global__ void k_kv_write(const gp_rec *r, const int64_t *e)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(8)) {
        DP(float, 2)[i] = DP(const float, 0)[i];
        DP(float, 3)[i] = DP(const float, 1)[i];
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

/* One warp for each row of an int4 matrix. The arguments give the operands
 * of x, w, out, rows, and cols: 0, 1, 3, 4, 5 for GP_INT4_LINEAR, and 3, 4,
 * 6, 7, 8 for the matrix of GP_GELU_MUL_INT4. The kernel reads the float16
 * scale of each block, not the float32 scales. np_gemma/gpu.py checks that
 * the two scales are equal. */
__global__ void k_int4_linear(const gp_rec *r, const int64_t *e, int xk, int wk, int ok,
                              int rk, int ck)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int rows = DI(rk), cols = DI(ck);
    if (row >= rows) {
        return;
    }
    float v = int4_row(DP(const uint8_t, wk) + (size_t)row * (cols / 32) * 18,
                       DP(const float, xk), cols);
    if (threadIdx.x % 32 == 0) {
        DP(float, ok)[row] = v;
    }
}

/* Up to four int4 matrices on the same x. The operand xk is x and ck is
 * cols. Then come (w, s, out, rows) for each matrix, from operand m0. A null
 * w skips a matrix. The rows of the four matrices follow each other. The
 * values are 0, 1, 2 for GP_INT4_MULTI4 and 2, 3, 5 for GP_RMS_NORM_MULTI4.
 * The x of GP_RMS_NORM_MULTI4 is the scratch of its norm. */
__global__ void k_int4_multi4(const gp_rec *r, const int64_t *e, int xk, int ck, int m0)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int cols = DI(ck);
    int m = 0;
    for (; m < 4; ++m) {
        int rows = DP(const uint8_t, m0 + 4 * m) ? DI(m0 + 3 + 4 * m) : 0;
        if (row < rows) {
            break;
        }
        row -= rows;
    }
    if (m == 4) {
        return;
    }
    float v = int4_row(DP(const uint8_t, m0 + 4 * m) + (size_t)row * (cols / 32) * 18,
                       DP(const float, xk), cols);
    if (threadIdx.x % 32 == 0) {
        DP(float, m0 + 2 + 4 * m)[row] = v;
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

/* The operands of the norm of the query, the key, and the value. GP_QKV_NORM
 * has them in this order. GP_QKV_NORM_ROPE has head_dim and eps at 12 and
 * 13. */
struct qkv_ix {
    int q, qw, qr, k, kw, kr, v, vr, hd, eps;
};

/* The norm of each head of q, k, and v, in place. One block for each row of
 * head_dim values. A row of v has no weight. */
__global__ void k_qkv_norm(const gp_rec *r, const int64_t *e, qkv_ix ix)
{
    int row = blockIdx.x;
    int qr = DI(ix.qr), kr = DI(ix.kr), hd = DI(ix.hd);
    float *x;
    const float *w;
    if (row < qr) {
        x = DP(float, ix.q) + (size_t)row * hd;
        w = DP(const float, ix.qw);
    } else if (row < qr + kr) {
        x = DP(float, ix.k) + (size_t)(row - qr) * hd;
        w = DP(const float, ix.kw);
    } else {
        x = DP(float, ix.v) + (size_t)(row - qr - kr) * hd;
        w = NULL;
    }
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)hd + df(r, e, ix.eps));
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        x[i] = w ? x[i] * s * w[i] : x[i] * s;
    }
}

/* The operands of the rope. GP_ROPE has them in this order. */
struct rope_ix {
    int q, qr, qh, k, kr, kh, cos, sin, hd;
};

/* The rope of each head of q and k, in place. One block for each row. As
 * gemma_rope_body. */
__global__ void k_rope(const gp_rec *r, const int64_t *e, rope_ix ix)
{
    int row = blockIdx.x;
    int qr = DI(ix.qr), hd = DI(ix.hd), d = hd / 2;
    float *x;
    int tok;
    if (row < qr) {
        x = DP(float, ix.q) + (size_t)row * hd;
        tok = row / DI(ix.qh);
    } else {
        x = DP(float, ix.k) + (size_t)(row - qr) * hd;
        tok = (row - qr) / DI(ix.kh);
    }
    const float *c = DP(const float, ix.cos) + (size_t)tok * hd;
    const float *s = DP(const float, ix.sin) + (size_t)tok * hd;
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

/* The operands of one attention, for either layout of the cache. */
struct attn_d {
    const float *q, *k, *v;
    float *sc, *out;
    int qh, kvh, hd, n, window;
    int64_t kp0, p;          /* the position of key row 0, and of the query */
    size_t hstride, rstride; /* the distance of two heads and of two rows */
};

/* GP_ATTN_F32H reads the E4B cache, which has the shape (heads, positions,
 * head_dim). Its operands:
 *
 *     q, k, v, scores, out, q_heads, kv_heads, head_dim, tokens (1), pos,
 *     head_stride, window, slide
 *
 * The pointers k and v point at position 0. With slide, the keys start at
 * the first position of the window.
 *
 * GP_ATTN_F32 reads the cache of the 26B model, which has the shape (rows,
 * kv_heads, head_dim). Its operands:
 *
 *     q, k, v, scores, out, q_heads, kv_heads, head_dim, n, pos, base, window
 *
 * The pointers k and v point at the first key row. Its position is base. */
__device__ attn_d attn_get(const gp_rec *r, const int64_t *e)
{
    attn_d a;
    a.q = DP(const float, 0);
    a.sc = DP(float, 3);
    a.out = DP(float, 4);
    a.qh = DI(5);
    a.kvh = DI(6);
    a.hd = DI(7);
    a.p = di(r, e, 9);
    if (r->op == GP_ATTN_F32H) {
        a.window = DI(11);
        int64_t lo = (DI(12) && a.window > 0) ? a.p - a.window + 1 : 0;
        if (lo < 0) {
            lo = 0;
        }
        a.kp0 = lo;
        a.n = (int)(a.p + 1 - lo);
        a.hstride = (size_t)di(r, e, 10);
        a.rstride = (size_t)a.hd;
        a.k = DP(const float, 1) + (size_t)lo * a.hd;
        a.v = DP(const float, 2) + (size_t)lo * a.hd;
    } else {
        a.window = DI(11);
        a.kp0 = di(r, e, 10);
        a.n = DI(8);
        a.hstride = (size_t)a.hd;
        a.rstride = (size_t)a.kvh * a.hd;
        a.k = DP(const float, 1);
        a.v = DP(const float, 2);
    }
    return a;
}

__global__ void k_attn_part(const gp_rec *r, const int64_t *e, float *part)
{
    attn_d a = attn_get(r, e);
    int h = blockIdx.x, c = blockIdx.y;
    int n = a.n, hd = a.hd;
    int len = (n + ATTN_CHUNKS - 1) / ATTN_CHUNKS;
    int j0 = c * len, j1 = min(n, j0 + len);
    int kv = h / (a.qh / a.kvh);
    const float *q = a.q + (size_t)h * hd;
    const float *k = a.k + kv * a.hstride;
    const float *v = a.v + kv * a.hstride;
    float *sc = a.sc + (size_t)h * n;
    float *o = part + ((size_t)h * ATTN_CHUNKS + c) * (hd + 2);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    for (int j = j0 + warp; j < j1; j += nw) {
        int64_t kp = a.kp0 + j;
        float s = 0.f;
        for (int i = lane; i < hd; i += 32) {
            s += q[i] * k[(size_t)j * a.rstride + i];
        }
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_xor_sync(0xffffffff, s, off);
        }
        if (lane == 0) {
            sc[j] = (kp > a.p || (a.window > 0 && a.p - kp >= a.window)) ? -INFINITY : s;
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
            acc += sc[j] * v[(size_t)j * a.rstride + i];
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

/* ---------- the router ----------
 * The operands of GP_ROUTER:
 *
 *     x, scale, proj, per_expert, hidden, experts, top_k, eps, hscale, val,
 *     idx, r, logits
 *
 * Three kernels do the three steps of gemma_router_body:
 *
 * 1. the norm of x into r;
 * 2. the logit of each expert;
 * 3. the softmax, and the choice of the top_k experts. */
__global__ void k_router_norm(const gp_rec *r, const int64_t *e)
{
    const float *x = DP(const float, 0);
    const float *scale = DP(const float, 1);
    float *rv = DP(float, 11);
    int hidden = DI(4);
    float ss = 0.f;
    for (int k = threadIdx.x; k < hidden; k += blockDim.x) {
        ss += x[k] * x[k];
    }
    ss = block_sum(ss);
    float inv = 1.0f / sqrtf(ss / (float)hidden + df(r, e, 7));
    float hs = df(r, e, 8);
    for (int k = threadIdx.x; k < hidden; k += blockDim.x) {
        rv[k] = x[k] * inv * scale[k] * hs;
    }
}

__global__ void k_router_logits(const gp_rec *r, const int64_t *e)
{
    int ex = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int hidden = DI(4);
    if (ex >= DI(5)) {
        return;
    }
    const float *pe = DP(const float, 2) + (size_t)ex * hidden;
    const float *rv = DP(const float, 11);
    float s = 0.f;
    for (int k = lane; k < hidden; k += 32) {
        s += rv[k] * pe[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, o);
    }
    if (lane == 0) {
        DP(float, 12)[ex] = s;
    }
}

/* One warp: the softmax, the top_k experts, and their weights. Each lane
 * keeps the logits of experts lane, lane + 32, and so on. At an equal
 * probability, the expert with the lower index wins, as in
 * gemma_router_body. At most 256 experts. */
__global__ void k_router_top(const gp_rec *r, const int64_t *e)
{
    const float *logits = DP(const float, 12);
    const float *per_expert = DP(const float, 3);
    float *val = DP(float, 9);
    int *idx = DP(int, 10);
    int experts = DI(5), top_k = DI(6);
    int lane = threadIdx.x;
    float p[8];
    float m = -INFINITY;
    for (int k = 0; k < 8; ++k) {
        int x = lane + 32 * k;
        p[k] = x < experts ? logits[x] : -INFINITY;
        m = fmaxf(m, p[k]);
    }
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    float sum = 0.f;
    for (int k = 0; k < 8; ++k) {
        p[k] = lane + 32 * k < experts ? expf(p[k] - m) : 0.f;
        sum += p[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    float invs = 1.0f / sum;
    for (int k = 0; k < 8; ++k) {
        p[k] = lane + 32 * k < experts ? p[k] * invs : -2.0f;
    }
    float vs = 0.f;
    for (int j = 0; j < top_k; ++j) {
        /* The best expert of this lane, then of the warp. */
        float bv = -3.0f;
        int bx = 1 << 30;
        for (int k = 0; k < 8; ++k) {
            if (p[k] > bv) {
                bv = p[k];
                bx = lane + 32 * k;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(0xffffffff, bv, o);
            int ox = __shfl_xor_sync(0xffffffff, bx, o);
            if (ov > bv || (ov == bv && ox < bx)) {
                bv = ov;
                bx = ox;
            }
        }
        if (lane == 0) {
            val[j] = bv;
            idx[j] = bx;
        }
        vs += bv;
        if ((bx & 31) == lane) {
            p[bx / 32] = -1.0f;
        }
    }
    if (lane == 0) {
        float invv = 1.0f / vs;
        for (int j = 0; j < top_k; ++j) {
            val[j] = val[j] * invv * per_expert[idx[j]];
        }
    }
}

/* ---------- the hot experts ----------
 * The GPU holds some experts of each layer (SPLIT_PLAN.md, the hot experts).
 * The array map of a layer gives the slot of each expert on the GPU, or -1.
 *
 * GP_HOT_SPLIT: idx, val, map, cold_idx, cold_val, top_k. Write the
 * selected experts that the GPU does not hold, and their weights, to
 * cold_idx and cold_val. cold_idx[top_k] gets their count. The CPU computes
 * these experts (GP_MOE_N of the CPU interpreter). */
__global__ void k_hot_split(const gp_rec *r, const int64_t *e)
{
    const int *idx = DP(const int, 0);
    const float *val = DP(const float, 1);
    const int *map = DP(const int, 2);
    int *cold = DP(int, 3);
    float *cold_val = DP(float, 4);
    int top_k = DI(5), n = 0;
    for (int j = 0; j < top_k; ++j) {
        if (map[idx[j]] < 0) {
            cold[n] = idx[j];
            cold_val[n] = val[j];
            ++n;
        }
    }
    cold[top_k] = n;
}

/* The operands of GP_HOT_MOE:
 *
 *     h, val, idx, map, gu, dn, act, act2, de, out, top_k, gu_rows, cols,
 *     dn_rows, inner
 *
 * The record computes the selected experts that the GPU holds. Four kernels
 * do the steps:
 *
 * 1. the gate and up rows into act;
 * 2. the GELU into act2;
 * 3. the down rows into de;
 * 4. the sum with the router weights into out.
 *
 * A block of an expert that the GPU does not hold does nothing. The arrays
 * gu and dn hold the int4 blocks of the experts of the layer, one expert
 * after the other. */
__device__ __forceinline__ int hot_slot(const gp_rec *r, const int64_t *e, int j)
{
    return DP(const int, 3)[DP(const int, 2)[j]];
}

__global__ void k_hot_gu(const gp_rec *r, const int64_t *e)
{
    int j = blockIdx.y, slot = hot_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int rows = DI(11), cols = DI(12);
    if (slot < 0 || row >= rows) {
        return;
    }
    size_t rb = (size_t)(cols / 32) * 18;
    float v = int4_row(DP(const uint8_t, 4) + ((size_t)slot * rows + row) * rb,
                       DP(const float, 0), cols);
    if (threadIdx.x % 32 == 0) {
        DP(float, 6)[(size_t)j * rows + row] = v;
    }
}

__global__ void k_hot_gelu(const gp_rec *r, const int64_t *e)
{
    int j = blockIdx.x;
    if (hot_slot(r, e, j) < 0) {
        return;
    }
    int inner = DI(14);
    const float *g = DP(const float, 6) + (size_t)j * 2 * inner;
    float *o = DP(float, 7) + (size_t)j * inner;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = g[i];
        o[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)))
               * g[inner + i];
    }
}

__global__ void k_hot_dn(const gp_rec *r, const int64_t *e)
{
    int j = blockIdx.y, slot = hot_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int rows = DI(13), inner = DI(14);
    if (slot < 0 || row >= rows) {
        return;
    }
    size_t rb = (size_t)(inner / 32) * 18;
    float v = int4_row(DP(const uint8_t, 5) + ((size_t)slot * rows + row) * rb,
                       DP(const float, 7) + (size_t)j * inner, inner);
    if (threadIdx.x % 32 == 0) {
        DP(float, 8)[(size_t)j * rows + row] = v;
    }
}

__global__ void k_hot_sum(const gp_rec *r, const int64_t *e)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int rows = DI(13), top_k = DI(10);
    if (c >= rows) {
        return;
    }
    const float *val = DP(const float, 1);
    const float *de = DP(const float, 8);
    float acc = 0.f;
    for (int j = 0; j < top_k; ++j) {
        if (hot_slot(r, e, j) >= 0) {
            acc += val[j] * de[(size_t)j * rows + c];
        }
    }
    DP(float, 9)[c] = acc;
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

/* The runner of a CPU program: gemma_run of the CPU library. Python gives its
 * address with gg_set_cpu_runner. */
typedef int (*gg_cpu_runner)(const int64_t *prog, int limit);
static gg_cpu_runner gg_cpu_run;

/* A segment: the kernel records from start to end - 1, and their graph. */
typedef struct {
    int start, end;
    cudaGraphExec_t exec;
} gg_seg;

typedef struct {
    int n_env, n_code;
    int64_t *henv;       /* the environment on the host */
    int64_t *denv;       /* the environment on the GPU */
    gp_rec *hcode;       /* the records, with device addresses */
    gp_rec *dcode;
    int use_graph;
    float *part;         /* the scratch of the attention */
    int n_seg;
    gg_seg *seg;         /* the segments, in the order of the records */
    int n_ev;
    cudaEvent_t *ev;     /* the events of the GP_TO_HOST records */
} gg_prog;

/* The size of the scratch of the attention: 32 heads of 1024 values. */
#define GG_PART_FLOATS ((size_t)32 * ATTN_CHUNKS * (1024 + 2))

static int is_boundary(int op)
{
    return op == GP_TO_HOST || op == GP_CPU_JOIN || op == GP_TO_DEV;
}

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

/* The rows of up to four matrices from operand m0, as k_int4_multi4 reads
 * them. */
static int64_t multi4_rows(const gp_rec *r, int m0, int *bad)
{
    int64_t rows = 0;
    for (int m = 0; m < 4; ++m) {
        if (r->v[m0 + 4 * m] != 0) {
            rows += hlit(r, m0 + 3 + 4 * m, bad);
        }
    }
    return rows;
}

static int attn_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr,
                       const int64_t *denv, int *bad)
{
    unsigned qh = (unsigned)hlit(r, 5, bad);
    if ((size_t)qh * ATTN_CHUNKS * (size_t)(hlit(r, 7, bad) + 2) > GG_PART_FLOATS) {
        *bad = 1;
        return 0;
    }
    k_attn_part<<<dim3(qh, ATTN_CHUNKS), 128, 0, gg_stream>>>(dr, denv, g->part);
    k_attn_join<<<qh, 128, 0, gg_stream>>>(dr, denv, g->part);
    return 0;
}

/* Launch the kernels of one record. Return 0, or -1 for an operation that
 * this file does not have or a size that is not a literal. */
static int gg_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr, const int64_t *denv)
{
    int bad = 0;
    cudaStream_t s = gg_stream;
    const int T = 256;
    const int W = 32 * ROWS_PER_BLOCK;
    switch (r->op) {
    case GP_S_MOV: case GP_S_ADD: case GP_S_SUB: case GP_S_MUL: case GP_S_MAX:
    case GP_S_MIN:
        return 0;
    case GP_RMS_NORM:
        k_rms_norm<<<(unsigned)hlit(r, 3, &bad), T, 0, s>>>(dr, denv, 0, 1, 2, 4, 5);
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
        k_int4_linear<<<(unsigned)cdiv(hlit(r, 4, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv, 0, 1, 3, 4, 5);
        break;
    case GP_INT4_MULTI4:
        k_int4_multi4<<<(unsigned)cdiv(multi4_rows(r, 2, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv, 0, 1, 2);
        break;
    case GP_RMS_NORM_MULTI4:
        /* x, wn, scratch, cols, eps, then the matrices from operand 5. */
        k_rms_norm<<<1, T, 0, s>>>(dr, denv, 0, 1, 2, 3, 4);
        k_int4_multi4<<<(unsigned)cdiv(multi4_rows(r, 5, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv, 2, 3, 5);
        break;
    case GP_GELU_MUL_INT4:
        /* g, u, inner, scratch, w, s, out, rows, cols */
        k_gelu_mul<<<(unsigned)cdiv(hlit(r, 2, &bad), T), T, 0, s>>>(dr, denv);
        k_int4_linear<<<(unsigned)cdiv(hlit(r, 7, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv, 3, 4, 6, 7, 8);
        break;
    case GP_BF16_LINEAR:
        if (hlit(r, 5, &bad) != 1 || hlit(r, 4, &bad) % 8 != 0) {
            bad = 1;
        }
        k_bf16_linear<<<(unsigned)cdiv(hlit(r, 3, &bad), ROWS_PER_BLOCK), W, 0, s>>>(dr, denv);
        break;
    case GP_QKV_NORM: {
        qkv_ix ix = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        k_qkv_norm<<<(unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad) + hlit(r, 7, &bad)),
                     128, 0, s>>>(dr, denv, ix);
        break;
    }
    case GP_ROPE: {
        rope_ix ix = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        k_rope<<<(unsigned)(hlit(r, 1, &bad) + hlit(r, 4, &bad)), 128, 0, s>>>(dr, denv, ix);
        break;
    }
    case GP_QKV_NORM_ROPE: {
        /* q, q_w, q_rows, k, k_w, k_rows, v, v_rows, cos, sin, q_heads,
         * k_heads, head_dim, eps */
        qkv_ix nx = {0, 1, 2, 3, 4, 5, 6, 7, 12, 13};
        rope_ix rx = {0, 2, 10, 3, 5, 11, 8, 9, 12};
        k_qkv_norm<<<(unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad) + hlit(r, 7, &bad)),
                     128, 0, s>>>(dr, denv, nx);
        k_rope<<<(unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad)), 128, 0, s>>>(dr, denv, rx);
        break;
    }
    case GP_KV_WRITE:
        if (r->v[4] != 0) {
            bad = 1;    /* the GPU keeps no int16 copy of the cache */
        }
        k_kv_write<<<(unsigned)cdiv(hlit(r, 8, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_KV_WRITE_HEADS:
        k_kv_write_heads<<<(unsigned)(hlit(r, 6, &bad) * hlit(r, 7, &bad)), 128, 0, s>>>(
            dr, denv);
        break;
    case GP_ATTN_F32H:
        if (hlit(r, 8, &bad) != 1) {
            bad = 1;
        }
        attn_launch(g, r, dr, denv, &bad);
        break;
    case GP_ATTN_F32:
        attn_launch(g, r, dr, denv, &bad);
        break;
    case GP_HOT_SPLIT:
        k_hot_split<<<1, 1, 0, s>>>(dr, denv);
        break;
    case GP_HOT_MOE: {
        unsigned k = (unsigned)hlit(r, 10, &bad);
        k_hot_gu<<<dim3((unsigned)cdiv(hlit(r, 11, &bad), ROWS_PER_BLOCK), k), W, 0, s>>>(
            dr, denv);
        k_hot_gelu<<<k, T, 0, s>>>(dr, denv);
        k_hot_dn<<<dim3((unsigned)cdiv(hlit(r, 13, &bad), ROWS_PER_BLOCK), k), W, 0, s>>>(
            dr, denv);
        k_hot_sum<<<(unsigned)cdiv(hlit(r, 13, &bad), T), T, 0, s>>>(dr, denv);
        break;
    }
    case GP_ROUTER:
        k_router_norm<<<1, T, 0, s>>>(dr, denv);
        k_router_logits<<<(unsigned)cdiv(hlit(r, 5, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv);
        k_router_top<<<1, 32, 0, s>>>(dr, denv);
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

/* Run a record that moves work between the GPU and the CPU.
 *
 * GP_TO_HOST: (device source, host target, bytes) three times, then the
 * index of an event. Null sources are not copied.
 * GP_CPU_JOIN: the address of a CPU program, then the index of the event.
 * GP_TO_DEV: host source, device target, bytes. */
static int gg_boundary(gg_prog *g, const gp_rec *r)
{
    switch (r->op) {
    case GP_TO_HOST:
        for (int k = 0; k < 3; ++k) {
            if (r->v[3 * k] != 0) {
                CK(cudaMemcpyAsync((void *)(intptr_t)r->v[3 * k + 1],
                                   (const void *)(intptr_t)r->v[3 * k],
                                   (size_t)r->v[3 * k + 2], cudaMemcpyDeviceToHost, gg_stream));
            }
        }
        CK(cudaEventRecord(g->ev[r->v[9]], gg_stream));
        return 0;
    case GP_CPU_JOIN:
        CK(cudaEventSynchronize(g->ev[r->v[1]]));
        if (gg_cpu_run == NULL || gg_cpu_run((const int64_t *)(intptr_t)r->v[0], -1) != 0) {
            snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_JOIN failed");
            return -1;
        }
        return 0;
    case GP_TO_DEV:
        CK(cudaMemcpyAsync((void *)(intptr_t)r->v[1], (const void *)(intptr_t)r->v[0],
                           (size_t)r->v[2], cudaMemcpyHostToDevice, gg_stream));
        return 0;
    default:
        return -1;
    }
}

/* Launch the kernels of a segment one at a time. */
static int gg_launch_seg(gg_prog *g, const gg_seg *sg)
{
    for (int pc = sg->start; pc < sg->end; ++pc) {
        if (gg_launch(g, g->hcode + pc, g->dcode + pc, g->denv) != 0) {
            return -1;
        }
    }
    return 0;
}

/* Run the segments and the boundary records in order. With the graph, the
 * first run records the graph of each segment. */
static int gg_exec(gg_prog *g)
{
    int pc = 0, k = 0;
    while (pc < g->n_code) {
        const gp_rec *r = g->hcode + pc;
        if (is_boundary(r->op)) {
            if (gg_boundary(g, r) != 0) {
                return -1;
            }
            ++pc;
            continue;
        }
        gg_seg *sg = g->seg + k++;
        if (!g->use_graph) {
            if (gg_launch_seg(g, sg) != 0) {
                return -1;
            }
        } else {
            if (sg->exec == NULL) {
                CK(cudaStreamBeginCapture(gg_stream, cudaStreamCaptureModeThreadLocal));
                int rc = gg_launch_seg(g, sg);
                cudaGraph_t graph;
                cudaError_t err = cudaStreamEndCapture(gg_stream, &graph);
                if (rc != 0) {
                    return rc;
                }
                CK(err);
                CK(cudaGraphInstantiate(&sg->exec, graph, 0));
                CK(cudaGraphDestroy(graph));
            }
            CK(cudaGraphLaunch(sg->exec, gg_stream));
        }
        pc = sg->end;
    }
    CK(cudaGetLastError());
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

void gg_set_cpu_runner(void *fn)
{
    gg_cpu_run = (gg_cpu_runner)fn;
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

/* Pinned host memory, for the copies of the handoff to the CPU. */
void *gg_host_alloc(size_t n)
{
    void *p = NULL;
    if (cudaMallocHost(&p, n) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "cudaMallocHost of %zu bytes failed", n);
        return NULL;
    }
    memset(p, 0, n);
    return p;
}

int gg_host_free(void *p)
{
    CK(cudaFreeHost(p));
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

/* A copy on the GPU. The areas can overlap. */
int gg_d2d(void *d, const void *src, size_t n)
{
    if ((const char *)d < (const char *)src + n && (const char *)src < (const char *)d + n) {
        void *tmp = NULL;
        CK(cudaMallocAsync(&tmp, n, gg_stream));
        CK(cudaMemcpyAsync(tmp, src, n, cudaMemcpyDeviceToDevice, gg_stream));
        CK(cudaMemcpyAsync(d, tmp, n, cudaMemcpyDeviceToDevice, gg_stream));
        CK(cudaFreeAsync(tmp, gg_stream));
        return 0;
    }
    CK(cudaMemcpyAsync(d, src, n, cudaMemcpyDeviceToDevice, gg_stream));
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
    /* The segments, and one event for each GP_TO_HOST. */
    g->seg = (gg_seg *)calloc((size_t)g->n_code + 1, sizeof(gg_seg));
    int in_seg = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        if (is_boundary(r->op)) {
            in_seg = 0;
            if (r->op == GP_TO_HOST && r->v[9] + 1 > g->n_ev) {
                g->n_ev = (int)r->v[9] + 1;
            }
            continue;
        }
        if (!in_seg) {
            g->seg[g->n_seg].start = pc;
            ++g->n_seg;
            in_seg = 1;
        }
        g->seg[g->n_seg - 1].end = pc + 1;
    }
    g->ev = (cudaEvent_t *)calloc((size_t)g->n_ev + 1, sizeof(cudaEvent_t));
    for (int k = 0; k < g->n_ev; ++k) {
        if (cudaEventCreateWithFlags(&g->ev[k], cudaEventDisableTiming) != cudaSuccess) {
            snprintf(gg_error, sizeof(gg_error), "gg_load: no event");
            return NULL;
        }
    }
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
    return gg_exec(g);
}

/* Run a program without the graph and measure each record. The array ms
 * gets the time of each record on the GPU, in milliseconds. A boundary
 * record gets the time of its copies and of its CPU program. This is for a
 * profile: the events add time between the kernels. */
int gg_profile(void *handle, const int64_t *env, float *ms)
{
    gg_prog *g = (gg_prog *)handle;
    memcpy(g->henv, env, (size_t)g->n_env * sizeof(int64_t));
    gg_scalars(g);
    CK(cudaMemcpyAsync(g->denv, g->henv, (size_t)g->n_env * sizeof(int64_t),
                       cudaMemcpyHostToDevice, gg_stream));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    int rc = 0;
    for (int pc = 0; pc < g->n_code && rc == 0; ++pc) {
        const gp_rec *r = g->hcode + pc;
        CK(cudaEventRecord(a, gg_stream));
        if (is_boundary(r->op)) {
            rc = gg_boundary(g, r);
        } else {
            rc = gg_launch(g, r, g->dcode + pc, g->denv);
        }
        CK(cudaEventRecord(b, gg_stream));
        CK(cudaEventSynchronize(b));
        CK(cudaEventElapsedTime(&ms[pc], a, b));
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
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
    for (int k = 0; k < g->n_seg; ++k) {
        if (g->seg[k].exec != NULL) {
            cudaGraphExecDestroy(g->seg[k].exec);
        }
    }
    for (int k = 0; k < g->n_ev; ++k) {
        cudaEventDestroy(g->ev[k]);
    }
    free(g->seg);
    free(g->ev);
    cudaFree(g->denv);
    cudaFree(g->dcode);
    cudaFree(g->part);
    cudaFreeHost(g->henv);
    free(g->hcode);
    free(g);
    return 0;
}

}  /* extern "C" */
