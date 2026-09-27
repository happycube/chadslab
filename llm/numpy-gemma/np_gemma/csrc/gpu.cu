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
#include <pthread.h>
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
    GP_QKV_NORM_ROPE = 48, GP_KV_WRITE = 49, GP_ATTN_QC = 50, GP_ATTN_F32 = 51,
    GP_QKV_NORM = 54, GP_ROPE = 55, GP_KV_WRITE_HEADS = 56, GP_ATTN_F32H = 57,
    GP_INT4_LINEAR_MT = 36, GP_INT4_MULTI4_MT = 37, GP_GELU_MUL_ROWS = 38,
    GP_ATTN_QC_MT = 52,
    GP_ROUTER = 64, GP_ROUTER_MT = 66,
    GP_TO_HOST = 84, GP_CPU_JOIN = 85, GP_TO_DEV = 86, GP_HOT_SPLIT = 87,
    GP_HOT_MOE = 88, GP_MOE_GPU = 89, GP_FETCH = 90, GP_FETCH_WAIT = 91,
    GP_FETCH_DONE = 92,
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
 * value. A null kd skips the float cache, and a null kqd skips the int16
 * cache. One thread for each group of 32 values. The int16 form is the form
 * of gemma_quant_group32_i16. Each group has a scale of max |x| / 32767. Each
 * value is x / scale, rounded to the nearest integer. */
__device__ __forceinline__ float quant32_i16(const float *x, int16_t *q)
{
    float amax = 0.f;
    for (int k = 0; k < 32; ++k) {
        amax = fmaxf(amax, fabsf(x[k]));
    }
    float sc = amax > 0.f ? amax / 32767.0f : 1e-12f;
    for (int k = 0; k < 32; ++k) {
        int v = __float2int_rn(x[k] / sc);
        q[k] = (int16_t)max(-32767, min(32767, v));
    }
    return sc;
}

__global__ void k_kv_write(const gp_rec *r, const int64_t *e)
{
    int g = blockIdx.x * blockDim.x + threadIdx.x;
    int n = DI(8);
    if (g >= n / 32) {
        return;
    }
    const float *k = DP(const float, 0) + (size_t)g * 32;
    const float *v = DP(const float, 1) + (size_t)g * 32;
    if (DP(float, 2)) {
        for (int i = 0; i < 32; ++i) {
            DP(float, 2)[(size_t)g * 32 + i] = k[i];
            DP(float, 3)[(size_t)g * 32 + i] = v[i];
        }
    }
    if (DP(int16_t, 4)) {
        DP(float, 5)[g] = quant32_i16(k, DP(int16_t, 4) + (size_t)g * 32);
        DP(float, 7)[g] = quant32_i16(v, DP(int16_t, 6) + (size_t)g * 32);
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
 * cols.
 *
 * Then come (w, s, out, rows) for each matrix, from operand m0. A null
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

/* The attention in two kernels, with the method of FlashDecoding. A decode
 * step has few heads, so one block for each head leaves most of the GPU idle.
 * Here the keys of each key and value head go to chunks of len keys. The
 * value len is n / ATTN_CHUNKS, but at least ATTN_MIN_KEYS. Thus a short context gets
 * a few small chunks, and a long context gets ATTN_CHUNKS chunks.
 *
 * The launch
 * has ATTN_CHUNKS blocks for each head. A block past the last chunk does
 * nothing.
 *
 * The kernel k_attn_part: block (kv, c) takes chunk c of the keys of key and
 * value head kv. It does this for all the query heads of that head (rep
 * heads), so it reads each key and value row one time. For each query head h and chunk c,
 * it writes hd + 2 values to the scratch of the program:
 *
 * - the sum of exp(score - m) times the value rows;
 * - m, the maximum of the scores of the chunk;
 * - l, the sum of exp(score - m).
 *
 * The kernel k_attn_join: block h adds the parts of head h. Part c gets the
 * weight exp(m_c - M), where M is the largest m_c. The kernel then divides by
 * the sum of the weighted l_c. */
#define ATTN_CHUNKS 256
#define ATTN_MIN_KEYS 32
#define ATTN_TILE 128
#define ATTN_REP 8
#define ATTN_KEYS 4

/* The keys of a chunk, and the count of chunks, for n keys. */
__device__ __forceinline__ int attn_len(int n)
{
    return max(ATTN_MIN_KEYS, (n + ATTN_CHUNKS - 1) / ATTN_CHUNKS);
}

/* The operands of one attention, for the three layouts of the cache. */
struct attn_d {
    const float *q;
    const float *k, *v;          /* the float cache */
    const int16_t *kq, *vq;      /* the int16 cache */
    const float *ks, *vs;        /* its scale for each group of 32 values */
    float *sc, *out;
    int qh, kvh, hd, n, window, i16;
    int64_t kp0, p;              /* the position of key row 0, and of the query */
    size_t hstride, rstride;     /* the distance of two heads and of two rows */
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
 * GP_ATTN_F32 reads the float cache of the 26B model, which has the shape
 * (rows, kv_heads, head_dim). Its operands:
 *
 *     q, k, v, scores, out, q_heads, kv_heads, head_dim, n, pos, base, window
 *
 * The pointers k and v point at the first key row. Its position is base.
 *
 * GP_ATTN_QC reads the int16 cache of the 26B model, with the same shape.
 * Its operands:
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, n
 *
 * The pointers point at the first key row. Every one of the n rows is in
 * the window of the query. */
__device__ attn_d attn_get(const gp_rec *r, const int64_t *e)
{
    attn_d a;
    memset(&a, 0, sizeof(a));
    a.q = DP(const float, 0);
    if (r->op == GP_ATTN_QC) {
        a.i16 = 1;
        a.kq = DP(const int16_t, 1);
        a.ks = DP(const float, 2);
        a.vq = DP(const int16_t, 3);
        a.vs = DP(const float, 4);
        a.sc = DP(float, 5);
        a.out = DP(float, 6);
        a.qh = DI(7);
        a.kvh = DI(8);
        a.hd = DI(9);
        a.n = DI(10);
        a.window = 0;
        a.kp0 = 0;
        a.p = a.n;
        a.hstride = (size_t)a.hd;
        a.rstride = (size_t)a.kvh * a.hd;
        return a;
    }
    a.sc = DP(float, 3);
    a.out = DP(float, 4);
    a.qh = DI(5);
    a.kvh = DI(6);
    a.hd = DI(7);
    a.p = di(r, e, 9);
    a.window = DI(11);
    if (r->op == GP_ATTN_F32H) {
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
        a.kp0 = di(r, e, 10);
        a.n = DI(8);
        a.hstride = (size_t)a.hd;
        a.rstride = (size_t)a.kvh * a.hd;
        a.k = DP(const float, 1);
        a.v = DP(const float, 2);
    }
    return a;
}

/* Read 8 values as floats: values i to i + 7 of row j of head kv. With key
 * 1 they come from the key cache, and with key 0 from the value cache. The
 * index i is a multiple of 8. Thus one load of 16 bytes (int16) or two loads
 * of 16 bytes (float) read them. Wide loads keep more bytes in flight. A
 * load of one value for each lane reached only about 58 GB/s. */
__device__ __forceinline__ void attn_kv8(const attn_d &a, int key, int kv, int j, int i,
                                         float *out)
{
    size_t o = (size_t)j * a.rstride + (size_t)kv * a.hstride + i;
    if (a.i16) {
        const int16_t *q = key ? a.kq : a.vq;
        float sc = (key ? a.ks : a.vs)[o / 32];
        uint4 u = *(const uint4 *)(q + o);
        uint32_t w[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            out[2 * t] = (float)(int16_t)(w[t] & 0xffff) * sc;
            out[2 * t + 1] = (float)(int16_t)(w[t] >> 16) * sc;
        }
    } else {
        const float *p = (key ? a.k : a.v) + o;
        float4 x = *(const float4 *)p, y = *(const float4 *)(p + 4);
        out[0] = x.x; out[1] = x.y; out[2] = x.z; out[3] = x.w;
        out[4] = y.x; out[5] = y.y; out[6] = y.z; out[7] = y.w;
    }
}

/* The block has 256 threads. head_dim must be 256 or 512. */
__global__ void k_attn_part(const gp_rec *r, const int64_t *e, float *part)
{
    __shared__ float qs[ATTN_REP * 512];
    __shared__ float ps[ATTN_REP * ATTN_TILE];
    __shared__ float red[ATTN_REP * 512];
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, c = blockIdx.y;
    int n = a.n, hd = a.hd, rep = a.qh / a.kvh;
    int len = attn_len(n);
    int j0 = c * len, j1 = min(n, j0 + len);
    if (j0 >= n) {
        return;
    }
    for (int i = threadIdx.x; i < rep * hd; i += blockDim.x) {
        qs[i] = a.q[(size_t)kv * rep * hd + i];
        red[i] = 0.f;
    }
    __syncthreads();
    /* The scores: each warp takes ATTN_KEYS keys at a time. Lane l reads
     * values 8 l to 8 l + 7 of each part of 256 values of each key. The loads
     * of the keys come first, and the sums of the lanes of the keys follow
     * each other, so more loads are in flight. */
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    for (int jb = j0 + warp * ATTN_KEYS; jb < j1; jb += nw * ATTN_KEYS) {
        float s[ATTN_KEYS][ATTN_REP];
        #pragma unroll
        for (int kk = 0; kk < ATTN_KEYS; ++kk) {
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                s[kk][h] = 0.f;
            }
        }
        for (int base = 0; base < hd; base += 256) {
            int i = base + 8 * lane;
            float kv8[ATTN_KEYS][8];
            #pragma unroll
            for (int kk = 0; kk < ATTN_KEYS; ++kk) {
                if (jb + kk < j1) {
                    attn_kv8(a, 1, kv, jb + kk, i, kv8[kk]);
                } else {
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        kv8[kk][u] = 0.f;
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                if (h < rep) {
                    const float *qh = qs + h * hd + i;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        float qv = qh[u];
                        #pragma unroll
                        for (int kk = 0; kk < ATTN_KEYS; ++kk) {
                            s[kk][h] += qv * kv8[kk][u];
                        }
                    }
                }
            }
        }
        #pragma unroll
        for (int h = 0; h < ATTN_REP; ++h) {
            if (h < rep) {
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    #pragma unroll
                    for (int kk = 0; kk < ATTN_KEYS; ++kk) {
                        s[kk][h] += __shfl_xor_sync(0xffffffff, s[kk][h], off);
                    }
                }
            }
        }
        if (lane == 0) {
            #pragma unroll
            for (int kk = 0; kk < ATTN_KEYS; ++kk) {
                int j = jb + kk;
                if (j >= j1) {
                    break;
                }
                int64_t kp = a.kp0 + j;
                bool masked = kp > a.p || (a.window > 0 && a.p - kp >= a.window);
                #pragma unroll
                for (int h = 0; h < ATTN_REP; ++h) {
                    if (h < rep) {
                        a.sc[(size_t)(kv * rep + h) * n + j] = masked ? -INFINITY : s[kk][h];
                    }
                }
            }
        }
    }
    __syncthreads();
    float m[ATTN_REP], l[ATTN_REP];
    #pragma unroll
    for (int h = 0; h < ATTN_REP; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
        if (h >= rep) {
            continue;
        }
        float *sc = a.sc + (size_t)(kv * rep + h) * n;
        float mh = -INFINITY;
        for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
            mh = fmaxf(mh, sc[j]);
        }
        mh = block_max(mh);
        float lh = 0.f;
        for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
            float x = mh == -INFINITY ? 0.f : expf(sc[j] - mh);
            sc[j] = x;
            lh += x;
        }
        l[h] = block_sum(lh);
        m[h] = mh;
    }
    __syncthreads();
    /* The value rows. Thread t keeps values 8 d to 8 d + 7 of every query
     * head, where d = t mod (hd / 8). The threads with the same d form
     * groups, and group g takes the keys g, g + groups, and so on, of each
     * tile. At the end, the groups add their sums in shared memory. */
    int nd = hd / 8, groups = blockDim.x / nd;
    int d = threadIdx.x % nd, grp = threadIdx.x / nd;
    float acc[ATTN_REP][8];
    #pragma unroll
    for (int h = 0; h < ATTN_REP; ++h) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[h][u] = 0.f;
        }
    }
    for (int t0 = j0; t0 < j1; t0 += ATTN_TILE) {
        int tn = min(ATTN_TILE, j1 - t0);
        for (int x = threadIdx.x; x < rep * tn; x += blockDim.x) {
            int h = x / tn, jj = x % tn;
            ps[h * ATTN_TILE + jj] = a.sc[(size_t)(kv * rep + h) * n + t0 + jj];
        }
        __syncthreads();
        #pragma unroll 4
        for (int jj = grp; jj < tn; jj += groups) {
            float vv[8];
            attn_kv8(a, 0, kv, t0 + jj, 8 * d, vv);
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                if (h < rep) {
                    float p = ps[h * ATTN_TILE + jj];
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        acc[h][u] += p * vv[u];
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int h = 0; h < ATTN_REP; ++h) {
        if (h < rep) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                atomicAdd(&red[h * hd + 8 * d + u], acc[h][u]);
            }
        }
    }
    __syncthreads();
    for (int x = threadIdx.x; x < rep * hd; x += blockDim.x) {
        int h = x / hd, i = x % hd;
        part[((size_t)(kv * rep + h) * ATTN_CHUNKS + c) * (hd + 2) + i] = red[x];
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int h = 0; h < ATTN_REP; ++h) {
            if (h < rep) {
                float *o = part + ((size_t)(kv * rep + h) * ATTN_CHUNKS + c) * (hd + 2);
                o[hd] = m[h];
                o[hd + 1] = l[h];
            }
        }
    }
}

/* Block (h, x) joins the values x * 128 to x * 128 + 127 of query head h.
 * The weight exp(m_c - M) of each chunk goes to shared memory first. */
__global__ void k_attn_join(const gp_rec *r, const int64_t *e, const float *part)
{
    __shared__ float w[ATTN_CHUNKS];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x, hd = a.hd;
    int len = attn_len(a.n);
    int nc = (a.n + len - 1) / len;
    const float *ph = part + (size_t)h * ATTN_CHUNKS * (hd + 2);
    float M = -INFINITY;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        M = fmaxf(M, ph[(size_t)c * (hd + 2) + hd]);
    }
    M = block_max(M);
    float wl = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float mc = ph[(size_t)c * (hd + 2) + hd];
        float wc = mc == -INFINITY ? 0.f : expf(mc - M);
        w[c] = wc;
        wl += wc * ph[(size_t)c * (hd + 2) + hd + 1];
    }
    float wsum = block_sum(wl);
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    __syncthreads();
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < hd) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) {
            acc += w[c] * ph[(size_t)c * (hd + 2) + i];
        }
        a.out[(size_t)h * hd + i] = acc * inv;
    }
}

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
    if (F32W) {
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

#define MT_MAX 16

__global__ void k_mt_gemv(const float *x, const uint8_t *w, float *out, int t, int rows, int cols)
{
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
                v += __shfl_xor_sync(0xffffffff, v, o);
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
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)DI(3) * DI(4)) {
        float v = DP(const float, 0)[i];
        DP(float, 2)[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
                                                   (v + 0.044715f * v * v * v)))
                          * DP(const float, 1)[i];
    }
}

/* The attention of a group of queries over the int16 cache, as FlashAttention
 * does it. The record GP_ATTN_QC_MT:
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, t, pos,
 *     base, window, lo, n
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
    int64_t last = pos + jl;
    size_t row = (size_t)kvh * hd;
    for (int64_t k0 = first; k0 <= last; k0 += KT) {
        int kn = (int)min((int64_t)KT, last - k0 + 1);
        for (int x = threadIdx.x; x < KT * hd; x += blockDim.x) {
            int kk = x / hd, i = x % hd;
            if (kk < kn) {
                size_t o = (size_t)(k0 - base + kk) * row + (size_t)kv * hd + i;
                ks_[kk][i] = (float)kq[o] * ksc[o / 32];
                vs_[kk][i] = (float)vq[o] * vsc[o / 32];
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
                sdot += __shfl_xor_sync(0xffffffff, sdot, o);
            }
            int64_t kp = k0 + kk;
            bool ok = kk < kn && kp <= p && (window == 0 || p - kp < window);
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

__global__ void k_flash_qc_mt(const gp_rec *r, const int64_t *e)
{
    __shared__ float qs[FQ][FD + 1];
    __shared__ float kvs[FK][FD + 1];
    __shared__ float ps[FQ][FK + 1];
    const float *q = DP(const float, 0);
    const int16_t *kq = DP(const int16_t, 1);
    const float *ksc = DP(const float, 2);
    const int16_t *vq = DP(const int16_t, 3);
    const float *vsc = DP(const float, 4);
    float *out = DP(float, 6);
    int qh = DI(7), kvh = DI(8), hd = DI(9), t = DI(10), window = DI(13);
    int64_t pos = di(r, e, 11), base = di(r, e, 12);
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
    int64_t last = pos + jl;
    size_t row = (size_t)kvh * hd;
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
                size_t o2 = (size_t)(k0 - base + a) * row + (size_t)kv * hd + d0 + d;
                kvs[a][d] = a < kn ? (float)kq[o2] * ksc[o2 / 32] : 0.f;
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
            bool ok = kk < kn && j < t && kp <= p && (window == 0 || p - kp < window);
            sc[v] = ok ? sc[v] : -INFINITY;
            mt = fmaxf(mt, sc[v]);
        }
        for (int off = 4; off > 0; off >>= 1) {
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, off));
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
            ls += __shfl_xor_sync(0xffffffff, ls, off);
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
                size_t o2 = (size_t)(k0 - base + a) * row + (size_t)kv * hd + d0 + d;
                kvs[a][d] = a < kn ? (float)vq[o2] * vsc[o2 / 32] : 0.f;
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

/* GP_ROUTER_MT: x, scale, proj, per_expert, hidden, experts, top_k, eps,
 * hscale, val, idx, t, r, logits. The kernels do the steps of GP_ROUTER for
 * each token. First the norm of the rows into r. Then the logits, as a
 * product with proj. Last, the top experts of each token. */
__global__ void k_router_norm_mt(const gp_rec *r, const int64_t *e)
{
    int j = blockIdx.x, hidden = DI(4);
    const float *x = DP(const float, 0) + (size_t)j * hidden;
    const float *scale = DP(const float, 1);
    float *rv = DP(float, 12) + (size_t)j * hidden;
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

__device__ void router_top_warp(const float *logits, const float *per_expert, float *val,
                                int *idx, int experts, int top_k);

__global__ void k_router_top_mt(const gp_rec *r, const int64_t *e)
{
    int j = blockIdx.x, experts = DI(5), top_k = DI(6);
    router_top_warp(DP(const float, 13) + (size_t)j * experts, DP(const float, 3),
                    DP(float, 9) + (size_t)j * top_k, DP(int, 10) + (size_t)j * top_k,
                    experts, top_k);
}

/* ---------- the experts of a large group on the GPU ----------
 * The operands of GP_MOE_GPU:
 *
 *     h, val, idx, t, top_k, 0, 0, gu_rows, cols, dn_rows, inner, cnt, off,
 *     fill, pair_tok, pair_of, tiles, act, act2, de, out, gu_tab, dn_tab
 *
 * The tables gu_tab and dn_tab give the device address of the int4 blocks
 * of each expert. A hot expert is on the GPU. A cold expert is in the
 * buffer that GP_FETCH fills. The steps:
 *
 * 1. k_moe_sort: count the pairs (token, slot) of each expert. Put the pairs
 *    in the order of the experts. The array pair_tok gets the token of each
 *    pair. The array pair_of gets the pair of each (token, slot).
 * 2. k_moe_tiles: make the list of the tiles: GM pairs of one expert each.
 * 3. k_moe_gemm (gate and up), k_moe_gelu, k_moe_gemm (down): the products
 *    of the tiles. A tile reads the weights of its expert one time for its
 *    GM pairs.
 * 4. k_moe_sum: out[j] = the sum over the slots of val[j][s] times the
 *    output of the pair of (j, s).
 *
 * The arrays cnt, off, and fill have one int for each expert, and off has
 * one more. The array tiles holds the tile count, then (expert, first pair)
 * for each tile. */
#define MOE_EXPERTS_MAX 256

__global__ void k_moe_sort(const gp_rec *r, const int64_t *e)
{
    const int *idx = DP(const int, 2);
    int pairs = DI(3) * DI(4), experts = MOE_EXPERTS_MAX;
    int *cnt = DP(int, 11), *off = DP(int, 12), *fill = DP(int, 13);
    int *pair_tok = DP(int, 14), *pair_of = DP(int, 15);
    int top_k = DI(4);
    for (int x = threadIdx.x; x < experts; x += blockDim.x) {
        cnt[x] = 0;
        fill[x] = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        atomicAdd(&cnt[idx[p]], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int a = 0;
        for (int x = 0; x < experts; ++x) {
            off[x] = a;
            a += cnt[x];
        }
        off[experts] = a;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        int ex = idx[p];
        int q = off[ex] + atomicAdd(&fill[ex], 1);
        pair_tok[q] = p / top_k;
        pair_of[p] = q;
    }
}

__global__ void k_moe_tiles(const gp_rec *r, const int64_t *e)
{
    const int *cnt = DP(const int, 11), *off = DP(const int, 12);
    int *tiles = DP(int, 16);
    int n = 0;
    for (int x = 0; x < MOE_EXPERTS_MAX; ++x) {
        for (int p0 = 0; p0 < cnt[x]; p0 += GM) {
            tiles[1 + 2 * n] = x;
            tiles[2 + 2 * n] = off[x] + p0;
            ++n;
        }
    }
    tiles[0] = n;
}

/* A tile of up to GM pairs of one expert, and GN rows of its matrix. With
 * gather, row q of A is row pair_tok[q] of a; else it is row q. */
__global__ void k_moe_gemm(const gp_rec *r, const int64_t *e, const float *a, int gather,
                           const int64_t *wtab, int rows, int cols, float *out)
{
    __shared__ float xs[32][GM + 1];
    __shared__ float ws[32][GN + 1];
    const int *tiles = DP(const int, 16);
    const int *off = DP(const int, 12);
    const int *pair_tok = DP(const int, 14);
    int tile = blockIdx.y;
    if (tile >= tiles[0]) {
        return;
    }
    int ex = tiles[1 + 2 * tile], q0 = tiles[2 + 2 * tile];
    int qend = off[ex + 1];
    int n0 = blockIdx.x * GN;
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    const uint8_t *we = (const uint8_t *)(intptr_t)wtab[ex];
    float acc[4][4] = {{0.f}};
    int groups = cols / 32;
    for (int g = 0; g < groups; ++g) {
        for (int x = threadIdx.x; x < GM * 32; x += blockDim.x) {
            int qq = x / 32, k = x % 32;
            int q = q0 + qq;
            float v = 0.f;
            if (q < qend) {
                int src = gather ? pair_tok[q] : q;
                v = a[(size_t)src * cols + (size_t)g * 32 + k];
            }
            xs[k][qq] = v;
        }
        if (threadIdx.x < GN) {
            int n = n0 + threadIdx.x;
            float v[32];
            if (n < rows) {
                w_block32<0>(we, n, cols, g, v);
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
            float av[4], bv[4];
            for (int u = 0; u < 4; ++u) {
                av[u] = xs[k][ty * 4 + u];
                bv[u] = ws[k][tx * 4 + u];
            }
            for (int u = 0; u < 4; ++u) {
                for (int v = 0; v < 4; ++v) {
                    acc[u][v] += av[u] * bv[v];
                }
            }
        }
        __syncthreads();
    }
    for (int u = 0; u < 4; ++u) {
        int q = q0 + ty * 4 + u;
        if (q >= qend) {
            continue;
        }
        for (int v = 0; v < 4; ++v) {
            int n = n0 + tx * 4 + v;
            if (n < rows) {
                out[(size_t)q * rows + n] = acc[u][v];
            }
        }
    }
}

/* act holds the gate and the up values of each pair: 2 inner values. */
__global__ void k_moe_gelu(const gp_rec *r, const int64_t *e)
{
    int inner = DI(10);
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)DI(3) * DI(4) * inner) {
        return;
    }
    size_t q = i / inner, c = i % inner;
    const float *g = DP(const float, 17) + q * 2 * inner;
    float v = g[c];
    DP(float, 18)[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
                                                (v + 0.044715f * v * v * v))) * g[inner + c];
}

__global__ void k_moe_sum(const gp_rec *r, const int64_t *e)
{
    int dn_rows = DI(9), top_k = DI(4);
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)DI(3) * dn_rows) {
        return;
    }
    size_t j = i / dn_rows, c = i % dn_rows;
    const float *val = DP(const float, 1);
    const int *pair_of = DP(const int, 15);
    const float *de = DP(const float, 19);
    float acc = 0.f;
    for (int s2 = 0; s2 < top_k; ++s2) {
        int q = pair_of[j * top_k + s2];
        acc += val[j * top_k + s2] * de[(size_t)q * dn_rows + c];
    }
    DP(float, 20)[i] = acc;
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
__device__ void router_top_warp(const float *logits, const float *per_expert, float *val,
                                int *idx, int experts, int top_k)
{
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

__global__ void k_router_top(const gp_rec *r, const int64_t *e)
{
    router_top_warp(DP(const float, 12), DP(const float, 3), DP(float, 9), DP(int, 10),
                    DI(5), DI(6));
}

/* ---------- the hot experts ----------
 * The GPU holds some experts of each layer (SPLIT_PLAN.md, the hot experts).
 * The array map of a layer gives the slot of each expert on the GPU, or -1.
 *
 * GP_HOT_SPLIT: idx, val, map, cold_idx, cold_val, top_k. Some selected
 * experts are not on the GPU. Write them to cold_idx, and their weights to
 * cold_val. cold_idx[top_k] gets their count. The CPU computes
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

/* The size of the scratch of the attention: 16 query heads of 512 values,
 * or the same count of values in another form. */
#define GG_PART_FLOATS ((size_t)16 * ATTN_CHUNKS * (512 + 2))

static int is_boundary(int op)
{
    return op == GP_TO_HOST || op == GP_CPU_JOIN || op == GP_TO_DEV || op == GP_FETCH ||
           op == GP_FETCH_WAIT || op == GP_FETCH_DONE;
}

/* ---------- the copy of the weights of the experts ----------
 * The operands of GP_FETCH are ranges, count, fetch f, and buffer b. ranges is a host array of
 * count triples (host address, device address, bytes). A worker thread
 * copies the ranges to the GPU on a stream of its own, and records the event
 * ready[f]. The host memory is not pinned: it is the map of the model file.
 * Thus the copy call waits until the data is in the buffers of the driver. The worker thread does that
 * wait, and the runner goes on with the launches. Before the copy, the
 * worker waits for the event free[b]: the last use of buffer b is done.
 *
 * GP_FETCH_WAIT f: the stream of the program waits for ready[f].
 * GP_FETCH_DONE b: the stream of the program records free[b]. */
#define GG_FETCH_MAX 256

typedef struct {
    const int64_t *ranges;
    int count;
    int f, b;
} gg_job;

static pthread_mutex_t gg_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gg_cv = PTHREAD_COND_INITIALIZER;
static gg_job gg_jobs[GG_FETCH_MAX];
static int gg_job_head, gg_job_tail;
static int gg_recorded[GG_FETCH_MAX];
static int gg_fetch_error;
static cudaEvent_t gg_ready[GG_FETCH_MAX], gg_freeev[2];
static cudaStream_t gg_copy;
static int gg_worker_on;

static void *gg_worker(void *arg)
{
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&gg_mu);
        while (gg_job_head == gg_job_tail) {
            pthread_cond_wait(&gg_cv, &gg_mu);
        }
        gg_job j = gg_jobs[gg_job_head % GG_FETCH_MAX];
        pthread_mutex_unlock(&gg_mu);
        int bad = cudaEventSynchronize(gg_freeev[j.b]) != cudaSuccess;
        for (int k = 0; k < j.count && !bad; ++k) {
            const int64_t *q = j.ranges + 3 * k;
            bad = cudaMemcpyAsync((void *)(intptr_t)q[1], (const void *)(intptr_t)q[0],
                                  (size_t)q[2], cudaMemcpyHostToDevice, gg_copy) != cudaSuccess;
        }
        bad = bad || cudaEventRecord(gg_ready[j.f], gg_copy) != cudaSuccess;
        pthread_mutex_lock(&gg_mu);
        gg_fetch_error |= bad;
        gg_recorded[j.f] = 1;
        ++gg_job_head;
        pthread_cond_broadcast(&gg_cv);
        pthread_mutex_unlock(&gg_mu);
    }
    return NULL;
}

static int gg_fetch_init(void)
{
    if (gg_worker_on) {
        return 0;
    }
    CK(cudaStreamCreateWithFlags(&gg_copy, cudaStreamNonBlocking));
    for (int k = 0; k < GG_FETCH_MAX; ++k) {
        CK(cudaEventCreateWithFlags(&gg_ready[k], cudaEventDisableTiming));
    }
    for (int k = 0; k < 2; ++k) {
        CK(cudaEventCreateWithFlags(&gg_freeev[k], cudaEventDisableTiming));
        CK(cudaEventRecord(gg_freeev[k], gg_stream));
    }
    pthread_t th;
    if (pthread_create(&th, NULL, gg_worker, NULL) != 0) {
        snprintf(gg_error, sizeof(gg_error), "no worker thread for the copies");
        return -1;
    }
    pthread_detach(th);
    gg_worker_on = 1;
    return 0;
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
    int o = r->op == GP_ATTN_QC ? 7 : 5;     /* the operand of q_heads */
    int64_t qh = hlit(r, o, bad), kvh = hlit(r, o + 1, bad), hd = hlit(r, o + 2, bad);
    if ((size_t)qh * ATTN_CHUNKS * (size_t)(hd + 2) > GG_PART_FLOATS ||
        qh % kvh != 0 || qh / kvh > ATTN_REP || (hd != 256 && hd != 512)) {
        *bad = 1;
        return 0;
    }
    k_attn_part<<<dim3((unsigned)kvh, ATTN_CHUNKS), 256, 0, gg_stream>>>(dr, denv, g->part);
    k_attn_join<<<dim3((unsigned)qh, (unsigned)cdiv(hd, 128)), 128, 0, gg_stream>>>(
        dr, denv, g->part);
    return 0;
}

/* The largest group for k_mt_gemv. A larger group uses k_gemm. A test can
 * lower it with gg_set_gemv_max to check k_gemm with a small group. */
static int gg_gemv_max = MT_MAX;

/* The product of an int4 matrix and a group of rows. The operands give x,
 * w, out, rows, cols, and t. The pointers must be literals, because the
 * kernel takes them as arguments: the compiler of a group passes arrays,
 * not slots. */
static void gemm_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int xk,
                        int wk, int ok, int rk, int ck, int tk, int *bad)
{
    if (r->tag[xk] == GP_T_SLOT || r->tag[wk] == GP_T_SLOT || r->tag[ok] == GP_T_SLOT) {
        *bad = 1;
        return;
    }
    const float *x = (const float *)(intptr_t)r->v[xk];
    const void *w = (const void *)(intptr_t)r->v[wk];
    float *out = (float *)(intptr_t)r->v[ok];
    int rows = (int)hlit(r, rk, bad), cols = (int)hlit(r, ck, bad), t = (int)hlit(r, tk, bad);
    if (t <= gg_gemv_max) {
        k_mt_gemv<<<(unsigned)cdiv(rows, ROWS_PER_BLOCK), 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
            x, (const uint8_t *)w, out, t, rows, cols);
    } else {
        k_gemm<0><<<dim3((unsigned)cdiv(rows, GN), (unsigned)cdiv(t, GM)), 256, 0, gg_stream>>>(
            x, w, out, t, rows, cols);
    }
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
        k_kv_write<<<(unsigned)cdiv(hlit(r, 8, &bad) / 32, 64), 64, 0, s>>>(dr, denv);
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
    case GP_ATTN_QC:
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
    case GP_INT4_LINEAR_MT:
        /* x, w, s, out, rows, cols, t */
        gemm_launch(r, dr, denv, 0, 1, 3, 4, 5, 6, &bad);
        break;
    case GP_INT4_MULTI4_MT:
        /* x, cols, t, then (w, s, out, rows) for up to four matrices */
        for (int m = 0; m < 4; ++m) {
            if (r->v[3 + 4 * m] != 0) {
                gemm_launch(r, dr, denv, 0, 3 + 4 * m, 5 + 4 * m, 6 + 4 * m, 1, 2, &bad);
            }
        }
        break;
    case GP_GELU_MUL_ROWS:
        k_gelu_mul_rows<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad), T), T, 0, s>>>(
            dr, denv);
        break;
    case GP_ATTN_QC_MT:
        if (hlit(r, 9, &bad) > 512 || hlit(r, 9, &bad) % 16 != 0) {
            bad = 1;
        }
        if (hlit(r, 10, &bad) > MT_MAX && hlit(r, 9, &bad) % FD == 0) {
            k_flash_qc_mt<<<dim3((unsigned)hlit(r, 7, &bad),
                                 (unsigned)cdiv(hlit(r, 10, &bad), FQ)), 256, 0, s>>>(dr, denv);
        } else {
            k_attn_qc_mt<<<dim3((unsigned)hlit(r, 7, &bad),
                                (unsigned)cdiv(hlit(r, 10, &bad), QT)), QT * 16, 0, s>>>(dr, denv);
        }
        break;
    case GP_ROUTER_MT: {
        int64_t t = hlit(r, 11, &bad), ex = hlit(r, 5, &bad);
        k_router_norm_mt<<<(unsigned)t, T, 0, s>>>(dr, denv);
        k_gemm<1><<<dim3((unsigned)cdiv(ex, GN), (unsigned)cdiv(t, GM)), 256, 0, s>>>(
            (const float *)(intptr_t)r->v[12], (const void *)(intptr_t)r->v[2],
            (float *)(intptr_t)r->v[13], (int)t, (int)ex, (int)hlit(r, 4, &bad));
        k_router_top_mt<<<(unsigned)t, 32, 0, s>>>(dr, denv);
        break;
    }
    case GP_MOE_GPU: {
        /* The pointers of the weights and of the scratch are literals. */
        int64_t t = hlit(r, 3, &bad), k = hlit(r, 4, &bad), pairs = t * k;
        int gu_rows = (int)hlit(r, 7, &bad), cols = (int)hlit(r, 8, &bad);
        int dn_rows = (int)hlit(r, 9, &bad), inner = (int)hlit(r, 10, &bad);
        unsigned max_tiles = (unsigned)(cdiv(pairs, GM) + 128);
        const int64_t *gu = (const int64_t *)(intptr_t)hlit(r, 21, &bad);
        const int64_t *dn = (const int64_t *)(intptr_t)hlit(r, 22, &bad);
        const float *h = (const float *)(intptr_t)hlit(r, 0, &bad);
        float *act = (float *)(intptr_t)hlit(r, 17, &bad);
        float *act2 = (float *)(intptr_t)hlit(r, 18, &bad);
        float *de = (float *)(intptr_t)hlit(r, 19, &bad);
        k_moe_sort<<<1, 1024, 0, s>>>(dr, denv);
        k_moe_tiles<<<1, 1, 0, s>>>(dr, denv);
        k_moe_gemm<<<dim3((unsigned)cdiv(gu_rows, GN), max_tiles), 256, 0, s>>>(
            dr, denv, h, 1, gu, gu_rows, cols, act);
        k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
        k_moe_gemm<<<dim3((unsigned)cdiv(dn_rows, GN), max_tiles), 256, 0, s>>>(
            dr, denv, act2, 0, dn, dn_rows, inner, de);
        k_moe_sum<<<(unsigned)cdiv(t * dn_rows, T), T, 0, s>>>(dr, denv);
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
    case GP_FETCH: {
        if (gg_fetch_init() != 0) {
            return -1;
        }
        gg_job j;
        j.ranges = (const int64_t *)(intptr_t)r->v[0];
        j.count = (int)r->v[1];
        j.f = (int)r->v[2];
        j.b = (int)r->v[3];
        pthread_mutex_lock(&gg_mu);
        gg_recorded[j.f] = 0;
        gg_jobs[gg_job_tail % GG_FETCH_MAX] = j;
        ++gg_job_tail;
        pthread_cond_broadcast(&gg_cv);
        pthread_mutex_unlock(&gg_mu);
        return 0;
    }
    case GP_FETCH_WAIT: {
        int f = (int)r->v[0];
        pthread_mutex_lock(&gg_mu);
        while (!gg_recorded[f]) {
            pthread_cond_wait(&gg_cv, &gg_mu);
        }
        int bad = gg_fetch_error;
        pthread_mutex_unlock(&gg_mu);
        if (bad) {
            snprintf(gg_error, sizeof(gg_error), "a copy of the weights of the experts failed");
            return -1;
        }
        CK(cudaStreamWaitEvent(gg_stream, gg_ready[f], 0));
        return 0;
    }
    case GP_FETCH_DONE:
        CK(cudaEventRecord(gg_freeev[r->v[0]], gg_stream));
        return 0;
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

void gg_set_gemv_max(int t)
{
    gg_gemv_max = t < MT_MAX ? t : MT_MAX;
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
