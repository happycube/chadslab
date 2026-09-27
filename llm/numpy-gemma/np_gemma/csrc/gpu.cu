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
    GP_FETCH_DONE = 92, GP_HOT_SPLIT_MT = 93, GP_F32_LINEAR = 94, GP_DRAFT_HEAD = 95,
    GP_ARGMAX = 96, GP_ADD_NORM = 97,
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

/* GP_ADD_NORM: o, w, x, out, rows, cols, eps, scale. For each row,
 * out = (x + o s w) scale, where s = 1 / sqrt(mean of o^2 + eps). This is
 * GP_RMS_NORM of o, GP_ADD, and GP_MUL_S in one pass, with the same
 * operations in the same order (as k_rms_norm), so the values are the same.
 * out can be x. One block of 256 threads for each row. */
__global__ void k_add_norm(const gp_rec *r, const int64_t *e)
{
    int cols = DI(5);
    const float *o = DP(const float, 0) + (size_t)blockIdx.x * cols;
    const float *w = DP(const float, 1);
    const float *x = DP(const float, 2) + (size_t)blockIdx.x * cols;
    float *out = DP(float, 3) + (size_t)blockIdx.x * cols;
    float eps = df(r, e, 6), scale = df(r, e, 7);
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        ss += o[i] * o[i];
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)cols + eps);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        /* No fused multiply-add, so the rounding is that of the three
         * kernels. */
        out[i] = __fmul_rn(__fadd_rn(x[i], __fmul_rn(__fmul_rn(o[i], s), w[i])), scale);
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
    /* The groups add their sums one after the other, in the order of the
     * group. An atomic add gives the order of arrival, and the router of the
     * 26B turns such a difference in the last bit into another expert. */
    for (int gi = 0; gi < groups; ++gi) {
        if (grp == gi) {
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                if (h < rep) {
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        red[h * hd + 8 * d + u] += acc[h][u];
                    }
                }
            }
        }
        __syncthreads();
    }
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
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
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
    int q0 = blockIdx.y * TMD;
    tc_tile<4>(x, w, out, q0, min(t, q0 + TMD), blockIdx.x * TN, rows, cols, (size_t)rows, NULL);
}

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
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa), "l"(gmem));
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem)
{
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void cp_async_wait1()
{
    asm volatile("cp.async.wait_group 1;\n" ::);
}

__device__ __forceinline__ void cp_async_wait0()
{
    asm volatile("cp.async.wait_group 0;\n" ::);
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

/* ---------- the int4 product with int8 activations ----------
 * k_gemm_q8: as k_gemm_tc2, but the rows of x are int8 with one float32
 * scale for each block of 32 values (k_quant_q8, as the Q8_0 form of
 * ggml). The instruction mma.sync m16n8k32 (int8 inputs, int32 sums) covers
 * one int4 block of 32 columns, and its int32 sum is exact. The kernel then
 * multiplies the sum by the scale of the block of x and the scale of the
 * block of w, in float32. On the RTX 5060 Ti the int8 instruction has about
 * 5 times the rate of the float16 one (204 against 37 TOPS).
 *
 * The fragments of m16n8k32 (g = lane / 4, c = lane % 4), 4 int8 values in
 * a register:
 *
 *     A (16 x 32, rows):  a0 = A[g][4c..], a1 = A[g+8][4c..],
 *                         a2 = A[g][4c+16..], a3 = A[g+8][4c+16..]
 *     B (32 x 8, cols):   b0 = B[4c..][g], b1 = B[4c+16..][g]
 *
 * The low 4 bits of byte i of a block are value i, and the high 4 bits are
 * value i + 16. Thus b0 comes from the low 4 bits of bytes 4c to 4c + 3.
 * b1 comes from their high 4 bits. __vsub4 subtracts 8 from each byte. */
__device__ __forceinline__ void mma16832(int *c, const uint32_t *a, const uint32_t *b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* x (n rows of cols values) to int8, with one scale for each block of 32:
 * q = round(x / s), s = max |x| / 127. Eight threads for each block; each
 * thread reads 4 values as a float4 and writes 4 bytes. */
__global__ void k_quant_q8(const float *x, int8_t *q, float *sc, size_t blocks)
{
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t bi = tid / 8;
    int sub = threadIdx.x % 8;
    bool live = bi < blocks;
    float4 v = live ? *(const float4 *)(x + bi * 32 + sub * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    if (!live) {
        return;
    }
    float s2 = m / 127.0f;
    char4 c4;
    c4.x = (signed char)(s2 > 0.f ? __float2int_rn(v.x / s2) : 0);
    c4.y = (signed char)(s2 > 0.f ? __float2int_rn(v.y / s2) : 0);
    c4.z = (signed char)(s2 > 0.f ? __float2int_rn(v.z / s2) : 0);
    c4.w = (signed char)(s2 > 0.f ? __float2int_rn(v.w / s2) : 0);
    *(char4 *)(q + bi * 32 + sub * 4) = c4;
    if (sub == 0) {
        sc[bi] = s2;
    }
}

/* The float32 value of an int32 i with |i| < 2^22, with no I2F instruction
 * (it has a low rate): the bits 0x4B400000 + i are the float 12582912 + i.
 * The subtraction is exact. */
__device__ __forceinline__ float i2f_exact(int i)
{
    return __int_as_float(0x4B400000 + i) - 12582912.0f;
}

__device__ __forceinline__ void cp_async16_z(void *smem, const void *gmem, int bytes);

#define Q8K 128      /* the columns of a step of k_gemm_q8: 4 blocks */
#define Q8RB 80      /* bytes of a row of w in a step: 72 bytes, and 8 of pad */
#define Q8NS 3       /* the buffers of k_gemm_q8 */
#define Q8SMEM ((size_t)Q8NS * (T2M * (Q8K + 16) + T2N * Q8RB + T2M * (Q8K / 32) * 4))

__device__ __forceinline__ void cp_async8(void *smem, const void *gmem)
{
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(sa), "l"(gmem));
}

template <int AL>
__global__ void __launch_bounds__(256) k_gemm_q8(const int8_t *xq, const float *xs,
                                                 const uint8_t *w, float *out, int t, int rows,
                                                 int cols)
{
    /* Q8NS buffers in dynamic shared memory: the copies of Q8NS - 1 steps
     * run during the compute of a step. */
    extern __shared__ __align__(16) uint8_t q8sm[];
    int8_t (*as_)[T2M][Q8K + 16] = (int8_t (*)[T2M][Q8K + 16])q8sm;
    uint8_t (*bs)[T2N][Q8RB] = (uint8_t (*)[T2N][Q8RB])(q8sm + Q8NS * T2M * (Q8K + 16));
    float (*ss)[T2M][Q8K / 32] = (float (*)[T2M][Q8K / 32])(q8sm + Q8NS * (T2M * (Q8K + 16) +
                                                                          T2N * Q8RB));
    /* blockIdx.x is the tile of tokens, so the blocks that read the same rows of
     * w run together and the rows come from DRAM one time. */
    int m0 = blockIdx.x * T2M, n0 = blockIdx.y * T2N;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    size_t rb = (size_t)(cols / 32) * 18;
    int nb = cols / 32, steps = cols / Q8K;
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    auto load = [&](int st, int b) {
        int k0 = st * Q8K;
        for (int e = threadIdx.x; e < T2M * Q8K / 16; e += blockDim.x) {
            int m = e / (Q8K / 16), kk = (e % (Q8K / 16)) * 16;
            int q = min(m0 + m, t - 1);
            cp_async16(&as_[b][m][kk], xq + (size_t)q * cols + k0 + kk);
        }
        for (int e = threadIdx.x; e < T2M * (Q8K / 32); e += blockDim.x) {
            int m = e / (Q8K / 32), bk = e % (Q8K / 32);
            int q = min(m0 + m, t - 1);
            cp_async4(&ss[b][m][bk], xs + (size_t)q * nb + k0 / 32 + bk);
        }
        /* The 4 blocks of a row in the step are 72 bytes. With AL, they
         * start at a multiple of 8 bytes (rb is a multiple of 16), so 9
         * copies of 8 bytes take them. */
        const uint8_t *wk = w + (size_t)(k0 / 32) * 18;
        if (AL) {
            for (int e = threadIdx.x; e < T2N * 9; e += blockDim.x) {
                int n = e / 9, wd = e % 9;
                int row = min(n0 + n, rows - 1);
                cp_async8(&bs[b][n][wd * 8], wk + (size_t)row * rb + wd * 8);
            }
        } else {
            for (int e = threadIdx.x; e < T2N * 18; e += blockDim.x) {
                int n = e / 18, wd = e % 18;
                int row = min(n0 + n, rows - 1);
                cp_async4(&bs[b][n][wd * 4], wk + (size_t)row * rb + wd * 4);
            }
        }
        cp_async_commit();
    };
    for (int st = 0; st < Q8NS - 1; ++st) {
        if (st < steps) {
            load(st, st);
        } else {
            cp_async_commit();
        }
    }
    for (int st = 0; st < steps; ++st) {
        int b = st % Q8NS;
        asm volatile("cp.async.wait_group %0;\n" :: "n"(Q8NS - 2));
        __syncthreads();
        /* Buffer (st + Q8NS - 1) % Q8NS held step st - 1, which every warp
         * has passed. */
        if (st + Q8NS - 1 < steps) {
            load(st + Q8NS - 1, (st + Q8NS - 1) % Q8NS);
        } else {
            cp_async_commit();
        }
        const int sh = 0;
        #pragma unroll
        for (int bk = 0; bk < Q8K / 32; ++bk) {
            uint32_t bf[4][2];
            float dw[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint8_t *blkp = &bs[b][wn + j * 8 + g][sh + bk * 18];
                uint32_t v = (uint32_t)*(const uint16_t *)(blkp + 2 + 4 * c) |
                             ((uint32_t)*(const uint16_t *)(blkp + 4 + 4 * c) << 16);
                bf[j][0] = __vsub4(v & 0x0f0f0f0fu, 0x08080808u);
                bf[j][1] = __vsub4((v >> 4) & 0x0f0f0f0fu, 0x08080808u);
                const uint8_t *d0 = &bs[b][wn + j * 8 + 2 * c][sh + bk * 18];
                const uint8_t *d1 = &bs[b][wn + j * 8 + 2 * c + 1][sh + bk * 18];
                dw[j][0] = __half2float(__ushort_as_half((uint16_t)(d0[0] | (d0[1] << 8))));
                dw[j][1] = __half2float(__ushort_as_half((uint16_t)(d1[0] | (d1[1] << 8))));
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r0 = wm + i * 16;
                int kk = bk * 32;
                uint32_t a[4];
                a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int ci[4] = {0, 0, 0, 0};
                    mma16832(ci, a, bf[j]);
                    acc[i][j][0] += i2f_exact(ci[0]) * s0 * dw[j][0];
                    acc[i][j][1] += i2f_exact(ci[1]) * s0 * dw[j][1];
                    acc[i][j][2] += i2f_exact(ci[2]) * s1 * dw[j][0];
                    acc[i][j][3] += i2f_exact(ci[3]) * s1 * dw[j][1];
                }
            }
        }
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

/* The float16 copy of n values, for the tensor cores. */
__global__ void k_to_half(const float *x, __half *y, size_t n)
{
    size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i + 3 < n) {
        float4 v = *(const float4 *)(x + i);
        *(__half2 *)(y + i) = __floats2half2_rn(v.x, v.y);
        *(__half2 *)(y + i + 2) = __floats2half2_rn(v.z, v.w);
    } else {
        for (; i < n; ++i) {
            y[i] = __float2half_rn(x[i]);
        }
    }
}

#define MT_MAX 16

/* k_mt_gemv with the token count NT fixed when the kernel compiles. The
 * loops over the tokens then unroll, and a lane keeps NT sums. The general
 * form keeps 16 sums and ran at about half the rate of one token. */
template <int NT>
__global__ void k_mt_gemv_n(const float *x, const uint8_t *w, float *out, int rows, int cols)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32, sub = lane & 3;
    if (row >= rows) {
        return;
    }
    int blocks = cols / 32;
    const uint8_t *wr = w + (size_t)row * blocks * 18;
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    for (int b = lane >> 2; b < blocks; b += 8) {
        const uint8_t *blk = wr + (size_t)b * 18;
        float d = __half2float(__ushort_as_half(*(const uint16_t *)blk));
        const uint16_t *qp = (const uint16_t *)(blk + 2 + 4 * sub);
        uint32_t q = (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
        float wl[4], wh[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            wl[u] = (float)((int)((q >> (8 * u)) & 15) - 8) * d;
            wh[u] = (float)((int)((q >> (8 * u + 4)) & 15) - 8) * d;
        }
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float *xb = x + (size_t)j * cols + b * 32;
            float4 xl = *(const float4 *)(xb + 4 * sub);
            float4 xh = *(const float4 *)(xb + 16 + 4 * sub);
            sum[j] += wl[0] * xl.x + wl[1] * xl.y + wl[2] * xl.z + wl[3] * xl.w
                    + wh[0] * xh.x + wh[1] * xh.y + wh[2] * xh.z + wh[3] * xh.w;
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        if (lane == 0) {
            out[(size_t)j * rows + row] = v;
        }
    }
}

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

/* A bfloat16 matrix on a small group: one warp for each row. Each lane reads
 * 8 values of the row at a time and uses them for every token. */
__global__ void k_mt_gemv_bf16(const float *x, const uint16_t *w, float *out, int t, int rows,
                               int cols)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const uint4 *wr = (const uint4 *)(w + (size_t)row * cols);
    float sum[MT_MAX];
    for (int j = 0; j < MT_MAX; ++j) {
        sum[j] = 0.f;
    }
    for (int i = lane; i < cols / 8; i += 32) {
        uint4 q = wr[i];
        uint32_t u[4] = {q.x, q.y, q.z, q.w};
        float wv[8];
        for (int k = 0; k < 4; ++k) {
            wv[2 * k] = __uint_as_float(u[k] << 16);
            wv[2 * k + 1] = __uint_as_float(u[k] & 0xffff0000u);
        }
        for (int j = 0; j < MT_MAX; ++j) {
            if (j < t) {
                const float *xi = x + (size_t)j * cols + i * 8;
                float a = 0.f;
                for (int k = 0; k < 8; ++k) {
                    a += wv[k] * xi[k];
                }
                sum[j] += a;
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

/* F32H 0: GP_ATTN_QC_MT, the int16 cache of the 26B (rows, heads, dim).
 * F32H 1: GP_ATTN_F32H, the float cache of the E4B (heads, positions, dim).
 * Its operands: q, k, v, scores, out, q_heads, kv_heads, head_dim, t, pos,
 * head_stride, window, slide. Its buffer row 0 has position 0. */
template <int F32H>
__device__ __forceinline__ float flash_kv(const gp_rec *r, const int64_t *e, int key,
                                          int64_t kp, int kv, int d)
{
    if (F32H) {
        const float *b = DP(const float, key ? 1 : 2);
        return b[(size_t)kv * di(r, e, 10) + (size_t)kp * DI(7) + d];
    }
    size_t o = (size_t)(kp - di(r, e, 12)) * DI(8) * DI(9) + (size_t)kv * DI(9) + d;
    const int16_t *q = DP(const int16_t, key ? 1 : 3);
    const float *sc = DP(const float, key ? 2 : 4);
    return (float)q[o] * sc[o / 32];
}

template <int F32H>
__global__ void k_flash_qc_mt(const gp_rec *r, const int64_t *e)
{
    __shared__ float qs[FQ][FD + 1];
    __shared__ float kvs[FK][FD + 1];
    __shared__ float ps[FQ][FK + 1];
    const float *q = DP(const float, 0);
    float *out = DP(float, F32H ? 4 : 6);
    int o0 = F32H ? 5 : 7;
    int qh = DI(o0), kvh = DI(o0 + 1), hd = DI(o0 + 2), t = DI(o0 + 3);
    int window = F32H ? DI(11) : DI(13);
    int64_t pos = di(r, e, F32H ? 9 : 11), base = F32H ? 0 : di(r, e, 12);
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
    /* The rows of head kv: the float cache (head-major) or the int16 cache
     * (row-major, from the head offset). */
    const float *kf = NULL, *vf = NULL;
    const int16_t *kq16 = NULL, *vq16 = NULL;
    const float *ks16 = NULL, *vs16 = NULL;
    size_t rowq = (size_t)kvh * hd;
    if (F32H) {
        kf = DP(const float, 1) + (size_t)kv * di(r, e, 10);
        vf = DP(const float, 2) + (size_t)kv * di(r, e, 10);
    } else {
        kq16 = DP(const int16_t, 1) + (size_t)kv * hd;
        vq16 = DP(const int16_t, 3) + (size_t)kv * hd;
        ks16 = DP(const float, 2) + (size_t)kv * hd / 32;
        vs16 = DP(const float, 4) + (size_t)kv * hd / 32;
    }
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
                float kv_v = 0.f;
                if (a < kn) {
                    if (F32H) {
                        kv_v = kf[(size_t)(k0 + a) * hd + d0 + d];
                    } else {
                        size_t o2 = (size_t)(k0 - base + a) * rowq + d0 + d;
                        kv_v = (float)kq16[o2] * ks16[o2 / 32];
                    }
                }
                kvs[a][d] = kv_v;
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
                float kv_v = 0.f;
                if (a < kn) {
                    if (F32H) {
                        kv_v = vf[(size_t)(k0 + a) * hd + d0 + d];
                    } else {
                        size_t o2 = (size_t)(k0 - base + a) * rowq + d0 + d;
                        kv_v = (float)vq16[o2] * vs16[o2 / 32];
                    }
                }
                kvs[a][d] = kv_v;
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

/* ---------- the attention of a large group on the tensor cores ----------
 * k_flash_tc: FlashAttention-2 with mma.sync m16n8k16. The queries, the
 * keys, the values, and the weights p become float16. The sums are float32.
 *
 * Block (h, tile) takes query head h and FQ2 queries. Warp w takes the 16
 * queries (w % 2) * 16 of the tile, and the 256 output values (w / 2) * 256
 * of each query. A head of 512 values has 4 warps. Each warp computes the
 * scores of its queries itself.
 *
 * For each step of FK2 keys:
 *
 * 1. S = Q K^T: the keys come to shared memory as float16.
 * 2. The online softmax on the fragments of S. The 4 lanes that hold a row
 *    find its maximum and its sum with shuffles.
 * 3. O += P V: the fragments of S are the fragments of A. The values come
 *    to shared memory with the keys as columns (Vt), so a fragment of B is
 *    two adjacent values.
 *
 * The record is GP_ATTN_QC_MT (F32H 0) or GP_ATTN_F32H (F32H 1). The kernel
 * k_flash_qc_mt takes the same records. head_dim is 256 or 512. */
#define FQ2 32
#define FK2 16

template <int F32H>
__global__ void k_flash_tc(const gp_rec *r, const int64_t *e)
{
    extern __shared__ __half fsm[];
    int o0 = F32H ? 5 : 7;
    int qh = DI(o0), kvh = DI(o0 + 1), hd = DI(o0 + 2), t = DI(o0 + 3);
    int window = F32H ? DI(11) : DI(13);
    int64_t pos = di(r, e, F32H ? 9 : 11), base = F32H ? 0 : di(r, e, 12);
    const float *q = DP(const float, 0);
    float *out = DP(float, F32H ? 4 : 6);
    int h = blockIdx.x, kv = h / (qh / kvh);
    int j0 = blockIdx.y * FQ2;
    int ld = hd + 8;
    __half *qs = fsm;                      /* FQ2 x ld */
    __half *ks = qs + FQ2 * ld;            /* FK2 x ld */
    __half *vt = ks + FK2 * ld;            /* hd x (FK2 + 8) */
    const int vld = FK2 + 8;
    const float *kf = NULL, *vf = NULL;
    const int16_t *kq16 = NULL, *vq16 = NULL;
    const float *ks16 = NULL, *vs16 = NULL;
    size_t rowq = (size_t)kvh * hd;
    if (F32H) {
        kf = DP(const float, 1) + (size_t)kv * di(r, e, 10);
        vf = DP(const float, 2) + (size_t)kv * di(r, e, 10);
    } else {
        kq16 = DP(const int16_t, 1) + (size_t)kv * hd;
        vq16 = DP(const int16_t, 3) + (size_t)kv * hd;
        ks16 = DP(const float, 2) + (size_t)kv * hd / 32;
        vs16 = DP(const float, 4) + (size_t)kv * hd / 32;
    }
    for (int x = threadIdx.x; x < FQ2 * hd; x += blockDim.x) {
        int a = x / hd, d = x % hd;
        int jj = j0 + a;
        qs[a * ld + d] = __float2half_rn(jj < t ? q[((size_t)jj * qh + h) * hd + d] : 0.f);
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int qg = warp % 2, dsl = warp / 2;
    int g = lane / 4, c = lane % 4;
    int jl = min(t, j0 + FQ2) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    int64_t last = pos + jl;
    int64_t p0 = pos + j0 + qg * 16 + g, p1 = p0 + 8;   /* the positions of rows g, g + 8 */
    bool live0 = j0 + qg * 16 + g < t, live1 = j0 + qg * 16 + g + 8 < t;
    float o[32][4];
    for (int i = 0; i < 32; ++i) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int64_t k0 = first; k0 <= last; k0 += FK2) {
        int kn = (int)min((int64_t)FK2, last - k0 + 1);
        __syncthreads();
        for (int x = threadIdx.x; x < FK2 * hd; x += blockDim.x) {
            int a = x / hd, d = x % hd;
            float kvk = 0.f, kvv = 0.f;
            if (a < kn) {
                if (F32H) {
                    kvk = kf[(size_t)(k0 + a) * hd + d];
                    kvv = vf[(size_t)(k0 + a) * hd + d];
                } else {
                    size_t o2 = (size_t)(k0 - base + a) * rowq + d;
                    kvk = (float)kq16[o2] * ks16[o2 / 32];
                    kvv = (float)vq16[o2] * vs16[o2 / 32];
                }
            }
            ks[a * ld + d] = __float2half_rn(kvk);
            vt[d * vld + a] = __float2half_rn(kvv);
        }
        __syncthreads();
        /* S: 16 queries by FK2 keys = 2 tiles of 8 keys. */
        float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const __half *qa = qs + (qg * 16) * ld;
        for (int kk = 0; kk < hd; kk += 16) {
            uint32_t a[4];
            a[0] = *(const uint32_t *)&qa[g * ld + kk + 2 * c];
            a[1] = *(const uint32_t *)&qa[(g + 8) * ld + kk + 2 * c];
            a[2] = *(const uint32_t *)&qa[g * ld + kk + 2 * c + 8];
            a[3] = *(const uint32_t *)&qa[(g + 8) * ld + kk + 2 * c + 8];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                uint32_t b[2];
                b[0] = *(const uint32_t *)&ks[(nt * 8 + g) * ld + kk + 2 * c];
                b[1] = *(const uint32_t *)&ks[(nt * 8 + g) * ld + kk + 2 * c + 8];
                mma16816(sc[nt], a, b);
            }
        }
        /* The mask and the online softmax. sc[nt][0..1] are row g, keys
         * nt * 8 + 2c and + 1; sc[nt][2..3] are row g + 8. */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int kk = nt * 8 + 2 * c + u;
                int64_t kp = k0 + kk;
                bool ok0 = live0 && kk < kn && kp <= p0 && (window == 0 || p0 - kp < window);
                bool ok1 = live1 && kk < kn && kp <= p1 && (window == 0 || p1 - kp < window);
                sc[nt][u] = ok0 ? sc[nt][u] : -INFINITY;
                sc[nt][2 + u] = ok1 ? sc[nt][2 + u] : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][u]);
                mx1 = fmaxf(mx1, sc[nt][2 + u]);
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, off));
        }
        float sc0 = (mx0 == -INFINITY || m0 == -INFINITY) ? (m0 == -INFINITY ? 0.f : 1.f)
                                                            : expf(m0 - mx0);
        float sc1 = (mx1 == -INFINITY || m1 == -INFINITY) ? (m1 == -INFINITY ? 0.f : 1.f)
                                                            : expf(m1 - mx1);
        float ls0 = 0.f, ls1 = 0.f;
        uint32_t pa[4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            float e0 = sc[nt][0] == -INFINITY ? 0.f : expf(sc[nt][0] - mx0);
            float e1 = sc[nt][1] == -INFINITY ? 0.f : expf(sc[nt][1] - mx0);
            float e2 = sc[nt][2] == -INFINITY ? 0.f : expf(sc[nt][2] - mx1);
            float e3 = sc[nt][3] == -INFINITY ? 0.f : expf(sc[nt][3] - mx1);
            ls0 += e0 + e1;
            ls1 += e2 + e3;
            __half2 h01 = __floats2half2_rn(e0, e1), h23 = __floats2half2_rn(e2, e3);
            pa[nt * 2 + 0] = *(uint32_t *)&h01;
            pa[nt * 2 + 1] = *(uint32_t *)&h23;
        }
        for (int off = 1; off < 4; off <<= 1) {
            ls0 += __shfl_xor_sync(0xffffffff, ls0, off);
            ls1 += __shfl_xor_sync(0xffffffff, ls1, off);
        }
        if (mx0 != -INFINITY) {
            l0 = l0 * sc0 + ls0;
            m0 = mx0;
        }
        if (mx1 != -INFINITY) {
            l1 = l1 * sc1 + ls1;
            m1 = mx1;
        }
        /* The A fragment of P: a0 = row g, keys 2c.. (tile 0); a1 = row g+8
         * (tile 0); a2 = row g, keys 8 + 2c (tile 1); a3 = row g + 8 (tile 1). */
        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            o[nt][0] *= (mx0 == -INFINITY ? 1.f : sc0);
            o[nt][1] *= (mx0 == -INFINITY ? 1.f : sc0);
            o[nt][2] *= (mx1 == -INFINITY ? 1.f : sc1);
            o[nt][3] *= (mx1 == -INFINITY ? 1.f : sc1);
            int d = dsl * 256 + nt * 8 + g;
            uint32_t b[2];
            b[0] = *(const uint32_t *)&vt[d * vld + 2 * c];
            b[1] = *(const uint32_t *)&vt[d * vld + 2 * c + 8];
            mma16816(o[nt], a, b);
        }
    }
    float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    int ja = j0 + qg * 16 + g;
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        int d = dsl * 256 + nt * 8 + 2 * c;
        if (live0) {
            out[((size_t)ja * qh + h) * hd + d] = o[nt][0] * inv0;
            out[((size_t)ja * qh + h) * hd + d + 1] = o[nt][1] * inv0;
        }
        if (live1) {
            out[((size_t)(ja + 8) * qh + h) * hd + d] = o[nt][2] * inv1;
            out[((size_t)(ja + 8) * qh + h) * hd + d + 1] = o[nt][3] * inv1;
        }
    }
}

/* ---------- the attention of a large group over a float32 cache ----------
 * k_flash_f32h: FlashAttention-2 for the record GP_ATTN_F32H (the cache of
 * the E4B model). It gives the result of k_flash_tc<1>, but it is faster:
 *
 * 1. Block (kv, tile, z) takes key and value head kv, the 16 queries of the
 *    tile, and HB query heads of that key head. The query heads share the
 *    keys and the values in shared memory, so the block reads them one time
 *    for HB heads.
 * 2. The keys and the values come to shared memory as float32 with
 *    cp.async. With ST 2, the copy of the next step of FK3 keys runs during
 *    the compute of this step.
 * 3. The fragments become float16 in registers. The rows of K have HD + 8
 *    values and the rows of V have HD + 4, so the reads of a warp do not
 *    hit the same bank.
 * 4. For HD 256, the fragments of the queries stay in registers. For HD 512
 *    they are in shared memory as float16.
 *
 * Warp w takes query head w / NS and the 256 output values (w % NS) * 256.
 * NS is HD / 256. */
#define FK3 16

__device__ __forceinline__ void cp_async16_z(void *smem, const void *gmem, int bytes)
{
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(sa), "l"(gmem), "r"(bytes));
}

__device__ __forceinline__ uint32_t pack_h2(float a, float b)
{
    __half2 h = __floats2half2_rn(a, b);
    return *(uint32_t *)&h;
}

template <int HD>
struct flash3_dims {
    static constexpr int NS = HD / 256;
    static constexpr int KLD = HD + 8;
    static constexpr int VLD = HD + 4;
    static constexpr int QLD = HD + 8;
};

template <int HD, int HB, int ST>
__global__ void __launch_bounds__(32 * HB * (HD / 256))
k_flash_f32h(const gp_rec *r, const int64_t *e)
{
    typedef flash3_dims<HD> D;
    extern __shared__ float fsm3[];
    float *kbuf = fsm3;                                   /* ST x FK3 x KLD */
    float *vbuf = kbuf + ST * FK3 * D::KLD;               /* ST x FK3 x VLD */
    __half *qs = (__half *)(vbuf + ST * FK3 * D::VLD);    /* HB x 16 x QLD (HD 512) */
    int qh = DI(5), kvh = DI(6), t = DI(8);
    int window = DI(11);
    int64_t pos = di(r, e, 9), hs = di(r, e, 10);
    const float *q = DP(const float, 0);
    float *out = DP(float, 4);
    int G = qh / kvh, kv = blockIdx.x, j0 = blockIdx.y * 16;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int hl = warp / D::NS, dsl = warp % D::NS;
    int h = kv * G + blockIdx.z * HB + hl;
    int g = lane / 4, c = lane % 4;
    const float *kf = DP(const float, 1) + (size_t)kv * hs;
    const float *vf = DP(const float, 2) + (size_t)kv * hs;
    bool live0 = j0 + g < t, live1 = j0 + g + 8 < t;
    int64_t p0 = pos + j0 + g, p1 = p0 + 8;
    int jl = min(t, j0 + 16) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < 0) {
        first = 0;
    }
    int64_t last = pos + jl;
    int nsteps = (int)((last - first) / FK3) + 1;

    auto load = [&](int stage, int64_t k0) {
        int kn = (int)min((int64_t)FK3, last - k0 + 1);
        float *kd = kbuf + stage * FK3 * D::KLD, *vd = vbuf + stage * FK3 * D::VLD;
        for (int x = threadIdx.x; x < FK3 * HD / 4; x += blockDim.x) {
            int a = x / (HD / 4), d = (x % (HD / 4)) * 4;
            bool ok = a < kn;
            size_t o2 = (size_t)(ok ? k0 + a : k0) * HD + d;
            cp_async16_z(kd + a * D::KLD + d, kf + o2, ok ? 16 : 0);
            cp_async16_z(vd + a * D::VLD + d, vf + o2, ok ? 16 : 0);
        }
        cp_async_commit();
    };
    load(0, first);

    /* The queries: rows g and g + 8 are the queries j0 + g and j0 + g + 8. */
    const float *q0r = q + ((size_t)(j0 + g) * qh + h) * HD;
    const float *q1r = q + ((size_t)(j0 + g + 8) * qh + h) * HD;
    uint32_t qa[D::NS == 1 ? HD / 16 : 1][4];
    if (D::NS == 1) {
        #pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            int k2 = D::NS == 1 ? kk / 16 : 0;
            int d = kk + 2 * c;
            qa[k2][0] = live0 ? pack_h2(q0r[d], q0r[d + 1]) : 0u;
            qa[k2][1] = live1 ? pack_h2(q1r[d], q1r[d + 1]) : 0u;
            qa[k2][2] = live0 ? pack_h2(q0r[d + 8], q0r[d + 9]) : 0u;
            qa[k2][3] = live1 ? pack_h2(q1r[d + 8], q1r[d + 9]) : 0u;
        }
    } else {
        for (int x = threadIdx.x; x < HB * 16 * HD; x += blockDim.x) {
            int hh = x / (16 * HD), a = (x / HD) % 16, d = x % HD;
            int jj = j0 + a;
            qs[(hh * 16 + a) * D::QLD + d] = __float2half_rn(
                jj < t ? q[((size_t)jj * qh + kv * G + blockIdx.z * HB + hh) * HD + d] : 0.f);
        }
    }
    const __half *qw = qs + hl * 16 * D::QLD;

    float o[32][4];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int s = 0; s < nsteps; ++s) {
        int64_t k0 = first + (int64_t)s * FK3;
        int kn = (int)min((int64_t)FK3, last - k0 + 1);
        int st = ST == 2 ? (s & 1) : 0;
        if (ST == 2 && s + 1 < nsteps) {
            load(st ^ 1, k0 + FK3);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();
        const float *kd = kbuf + st * FK3 * D::KLD;
        const float *vd = vbuf + st * FK3 * D::VLD;
        /* S: 16 queries by FK3 keys = 2 tiles of 8 keys. */
        float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            uint32_t a[4];
            if (D::NS == 1) {
                int k2 = D::NS == 1 ? kk / 16 : 0;
                a[0] = qa[k2][0];
                a[1] = qa[k2][1];
                a[2] = qa[k2][2];
                a[3] = qa[k2][3];
            } else {
                a[0] = *(const uint32_t *)&qw[g * D::QLD + kk + 2 * c];
                a[1] = *(const uint32_t *)&qw[(g + 8) * D::QLD + kk + 2 * c];
                a[2] = *(const uint32_t *)&qw[g * D::QLD + kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&qw[(g + 8) * D::QLD + kk + 2 * c + 8];
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                const float *kr = kd + (nt * 8 + g) * D::KLD + kk + 2 * c;
                float2 x0 = *(const float2 *)kr, x1 = *(const float2 *)(kr + 8);
                uint32_t b[2] = {pack_h2(x0.x, x0.y), pack_h2(x1.x, x1.y)};
                mma16816(sc[nt], a, b);
            }
        }
        /* The mask and the online softmax, as in k_flash_tc. */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int kk = nt * 8 + 2 * c + u;
                int64_t kp = k0 + kk;
                bool ok0 = live0 && kk < kn && kp <= p0 && (window == 0 || p0 - kp < window);
                bool ok1 = live1 && kk < kn && kp <= p1 && (window == 0 || p1 - kp < window);
                sc[nt][u] = ok0 ? sc[nt][u] : -INFINITY;
                sc[nt][2 + u] = ok1 ? sc[nt][2 + u] : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][u]);
                mx1 = fmaxf(mx1, sc[nt][2 + u]);
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffff, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffff, mx1, off));
        }
        float sc0 = (mx0 == -INFINITY || m0 == -INFINITY) ? (m0 == -INFINITY ? 0.f : 1.f)
                                                            : expf(m0 - mx0);
        float sc1 = (mx1 == -INFINITY || m1 == -INFINITY) ? (m1 == -INFINITY ? 0.f : 1.f)
                                                            : expf(m1 - mx1);
        float ls0 = 0.f, ls1 = 0.f;
        uint32_t pa[4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            float e0 = sc[nt][0] == -INFINITY ? 0.f : expf(sc[nt][0] - mx0);
            float e1 = sc[nt][1] == -INFINITY ? 0.f : expf(sc[nt][1] - mx0);
            float e2 = sc[nt][2] == -INFINITY ? 0.f : expf(sc[nt][2] - mx1);
            float e3 = sc[nt][3] == -INFINITY ? 0.f : expf(sc[nt][3] - mx1);
            ls0 += e0 + e1;
            ls1 += e2 + e3;
            pa[nt * 2 + 0] = pack_h2(e0, e1);
            pa[nt * 2 + 1] = pack_h2(e2, e3);
        }
        for (int off = 1; off < 4; off <<= 1) {
            ls0 += __shfl_xor_sync(0xffffffff, ls0, off);
            ls1 += __shfl_xor_sync(0xffffffff, ls1, off);
        }
        if (mx0 != -INFINITY) {
            l0 = l0 * sc0 + ls0;
            m0 = mx0;
        }
        if (mx1 != -INFINITY) {
            l1 = l1 * sc1 + ls1;
            m1 = mx1;
        }
        float f0 = mx0 == -INFINITY ? 1.f : sc0, f1 = mx1 == -INFINITY ? 1.f : sc1;
        /* O += P V. A fragment of B: b0 = V[2c, 2c + 1][d], b1 = V[2c + 8, 2c + 9][d]. */
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            o[nt][0] *= f0;
            o[nt][1] *= f0;
            o[nt][2] *= f1;
            o[nt][3] *= f1;
            int d = dsl * 256 + nt * 8 + g;
            const float *vr = vd + (2 * c) * D::VLD + d;
            uint32_t b[2] = {pack_h2(vr[0], vr[D::VLD]),
                             pack_h2(vr[8 * D::VLD], vr[9 * D::VLD])};
            mma16816(o[nt], pa, b);
        }
        __syncthreads();
        if (ST == 1 && s + 1 < nsteps) {
            load(0, k0 + FK3);
        }
    }
    float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        int d = dsl * 256 + nt * 8 + 2 * c;
        if (live0) {
            *(float2 *)&out[((size_t)(j0 + g) * qh + h) * HD + d] =
                make_float2(o[nt][0] * inv0, o[nt][1] * inv0);
        }
        if (live1) {
            *(float2 *)&out[((size_t)(j0 + g + 8) * qh + h) * HD + d] =
                make_float2(o[nt][2] * inv1, o[nt][3] * inv1);
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

/* The logits of a small group: one warp for each expert and token. */
__global__ void k_router_logits_mt(const gp_rec *r, const int64_t *e)
{
    int ex = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int j = blockIdx.y, lane = threadIdx.x % 32;
    int hidden = DI(4), experts = DI(5);
    if (ex >= experts) {
        return;
    }
    const float *pe = DP(const float, 2) + (size_t)ex * hidden;
    const float *rv = DP(const float, 12) + (size_t)j * hidden;
    float s2 = 0.f;
    for (int k = lane; k < hidden; k += 32) {
        s2 += rv[k] * pe[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    if (lane == 0) {
        DP(float, 13)[(size_t)j * experts + ex] = s2;
    }
}

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

/* k_moe_gemm on the tensor cores: the same tiles, with tc_tile. */
__global__ void k_moe_gemm_tc(const gp_rec *r, const int64_t *e, const void *a, int gather,
                              const int64_t *wtab, int rows, int cols, float *out)
{
    const int *tiles = DP(const int, 16);
    const int *off = DP(const int, 12);
    int tile = blockIdx.y;
    if (tile >= tiles[0]) {
        return;
    }
    int ex = tiles[1 + 2 * tile], q0 = tiles[2 + 2 * tile];
    int q1 = min(off[ex + 1], q0 + TM);
    tc_tile<2>((const __half *)a, (const uint8_t *)(intptr_t)wtab[ex], out, q0, q1,
               blockIdx.x * TN, rows, cols, (size_t)rows, gather ? DP(const int, 14) : NULL);
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
 *     dn_rows, inner, t
 *
 * The record serves a group of t tokens too: h is (t, cols), and val and idx
 * are (t, top_k). Pair j is slot j % top_k of token j / top_k.
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
                       DP(const float, 0) + (size_t)(j / DI(10)) * cols, cols);
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
    int tok = blockIdx.y;
    int rows = DI(13), top_k = DI(10);
    if (c >= rows) {
        return;
    }
    const float *val = DP(const float, 1);
    const float *de = DP(const float, 8);
    float acc = 0.f;
    for (int s2 = 0; s2 < top_k; ++s2) {
        int j = tok * top_k + s2;
        if (hot_slot(r, e, j) >= 0) {
            acc += val[j] * de[(size_t)j * rows + c];
        }
    }
    DP(float, 9)[(size_t)tok * rows + c] = acc;
}

/* GP_HOT_SPLIT_MT: idx, val, map, cold_idx, cold_val, pairs. Some pairs of
 * a group use an expert that the GPU does not hold. For such a pair, write
 * the expert and its weight. For the other pairs, write -1 and 0. GP_MOE_MT of the CPU skips the pairs of -1
 * (the CPU part), and GP_HOT_MOE skips the other pairs (the GPU part). */
__global__ void k_hot_split_mt(const gp_rec *r, const int64_t *e)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= DI(5)) {
        return;
    }
    int x = DP(const int, 0)[p];
    bool cold = DP(const int, 2)[x] < 0;
    DP(int, 3)[p] = cold ? x : -1;
    DP(float, 4)[p] = cold ? DP(const float, 1)[p] : 0.f;
}

/* ---------- the MTP drafter (np_gemma/gpu.py, GPUDrafter) ----------
 * GP_F32_LINEAR: x, w, out, rows, cols. A float32 matrix on one row: one
 * warp for each row.
 *
 * GP_DRAFT_HEAD: u, clog, order, head, sel, top, token, n_cent, per, top_k,
 * hidden. The centroid head of the drafter, as Assistant.logits does it.
 * The array clog holds the logits of the centroids. The kernels select the
 * top_k centroids. Then they compute the logits of the tokens of those
 * centroids: order holds per tokens for each centroid, and head is the
 * float32 table. Last, they write the best token. */
__global__ void k_f32_linear(const gp_rec *r, const int64_t *e)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32, rows = DI(3), cols = DI(4);
    if (row >= rows) {
        return;
    }
    const float *w = DP(const float, 1) + (size_t)row * cols;
    const float *x = DP(const float, 0);
    float s2 = 0.f;
    for (int k = lane; k < cols; k += 32) {
        s2 += w[k] * x[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    if (lane == 0) {
        DP(float, 2)[row] = s2;
    }
}

/* The top_k centroids: a bitonic sort of (logit, index) in shared memory,
 * largest first. One block of 1024 threads; n_cent is at most 2048. */
__global__ void k_draft_top(const gp_rec *r, const int64_t *e)
{
    __shared__ float kv[2048];
    __shared__ int ki[2048];
    const float *clog = DP(const float, 1);
    int n = DI(7), top_k = DI(9);
    int size = 1;
    while (size < n) {
        size <<= 1;
    }
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        kv[i] = i < n ? clog[i] : -INFINITY;
        ki[i] = i;
    }
    __syncthreads();
    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < size; i += blockDim.x) {
                int l = i ^ j;
                if (l > i) {
                    bool desc = (i & k) == 0;
                    bool swap = desc ? (kv[i] < kv[l] || (kv[i] == kv[l] && ki[i] > ki[l]))
                                     : (kv[i] > kv[l] || (kv[i] == kv[l] && ki[i] < ki[l]));
                    if (swap) {
                        float tv = kv[i];
                        kv[i] = kv[l];
                        kv[l] = tv;
                        int ti = ki[i];
                        ki[i] = ki[l];
                        ki[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        DP(int, 5)[i] = ki[i];
    }
}

/* The logit of each candidate token: one warp for each. */
__global__ void k_draft_sel(const gp_rec *r, const int64_t *e)
{
    int i = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32, per = DI(8), top_k = DI(9), hidden = DI(10);
    if (i >= top_k * per) {
        return;
    }
    int tok = DP(const int, 2)[(size_t)DP(const int, 5)[i / per] * per + i % per];
    const float *hrow = DP(const float, 3) + (size_t)tok * hidden;
    const float *u = DP(const float, 0);
    float s2 = 0.f;
    for (int k = lane; k < hidden; k += 32) {
        s2 += hrow[k] * u[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s2 += __shfl_xor_sync(0xffffffff, s2, o);
    }
    if (lane == 0) {
        DP(float, 4)[i] = s2;
    }
}

/* The best candidate: one block. At an equal logit, the first candidate. */
__global__ void k_draft_best(const gp_rec *r, const int64_t *e)
{
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    int per = DI(8), n = DI(9) * per;
    const float *sel = DP(const float, 4);
    float v = -INFINITY;
    int b = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (sel[i] > v) {
            v = sel[i];
            b = i;
        }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = b;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) {
            float ov = bv[threadIdx.x + s2];
            int oi = bi[threadIdx.x + s2];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int i = bi[0];
        DP(int, 6)[0] = DP(const int, 2)[(size_t)DP(const int, 5)[i / per] * per + i % per];
    }
}

/* GP_ARGMAX: x, n, out. out[0] gets the index of the largest of the n values
 * of x. At an equal value, the lower index wins, as np.argmax. One block. */
__global__ void k_argmax(const gp_rec *r, const int64_t *e)
{
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float *x = DP(const float, 0);
    int n = DI(1);
    float v = -INFINITY;
    int b = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > v) {
            v = x[i];
            b = i;
        }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = b;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) {
            float ov = bv[threadIdx.x + s2];
            int oi = bi[threadIdx.x + s2];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        DP(int, 2)[0] = bi[0];
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
#define HEAD_MAX 16

__global__ void k_q6k_head(const uint8_t *w, const float *x, float *out, int rows, int cols,
                           float cap, int nx)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int l = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    int nb = cols / 256;
    const uint8_t *wr = w + (size_t)row * nb * 210;
    float sum[HEAD_MAX];
    for (int k = 0; k < HEAD_MAX; ++k) {
        sum[k] = 0.f;
    }
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
            wv[4 * n + 0] = d * (float)sc[is + 0] * (float)(((a & 15) | ((hq & 3) << 4)) - 32);
            wv[4 * n + 1] = d * (float)sc[is + 2] * (float)(((bq & 15) | (((hq >> 2) & 3) << 4)) - 32);
            wv[4 * n + 2] = d * (float)sc[is + 4] * (float)(((a >> 4) | (((hq >> 4) & 3) << 4)) - 32);
            wv[4 * n + 3] = d * (float)sc[is + 6] * (float)(((bq >> 4) | (((hq >> 6) & 3) << 4)) - 32);
            idx[4 * n + 0] = b * 256 + 128 * n + l;
            idx[4 * n + 1] = idx[4 * n + 0] + 32;
            idx[4 * n + 2] = idx[4 * n + 0] + 64;
            idx[4 * n + 3] = idx[4 * n + 0] + 96;
        }
        for (int k = 0; k < HEAD_MAX; ++k) {
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
    for (int k = 0; k < HEAD_MAX; ++k) {
        if (k < nx) {
            float v = sum[k];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(0xffffffff, v, o);
            }
            if (l == 0) {
                out[(size_t)k * rows + row] = cap > 0.f ? cap * tanhf(v / cap) : v;
            }
        }
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
    __half *xh;          /* the float16 copy of the input of a product */
    size_t xh_n;
    int tc;              /* 1: the tensor cores for a large group */
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

/* 1: the tensor cores for the int4 products of a large group (k_gemm_tc,
 * k_moe_gemm_tc). 0: the float32 kernels (k_gemm, k_moe_gemm). */
static int gg_tc = 1;

template <int HD, int HB, int ST>
static void flash3_run(const gp_rec *dr, const int64_t *denv, int kvh, int t, int G)
{
    typedef flash3_dims<HD> D;
    size_t smem = (size_t)ST * FK3 * (D::KLD + D::VLD) * sizeof(float) +
                  (D::NS == 1 ? 0 : (size_t)HB * 16 * D::QLD * sizeof(__half));
    cudaFuncSetAttribute(k_flash_f32h<HD, HB, ST>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((unsigned)kvh, (unsigned)cdiv(t, 16), (unsigned)(G / HB));
    k_flash_f32h<HD, HB, ST><<<grid, 32 * HB * D::NS, smem, gg_stream>>>(dr, denv);
}

/* Launch k_flash_f32h for a GP_ATTN_F32H record. Return -1 when it does not
 * take the shape. NP_GEMMA_GPU_FLASH=1 selects k_flash_tc<1> for a test. */
static int flash3_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int *bad)
{
    static int old = -1;
    if (old < 0) {
        const char *v = getenv("NP_GEMMA_GPU_FLASH");
        old = v && v[0] == '1';
    }
    if (old) {
        return -1;
    }
    int64_t qh = hlit(r, 5, bad), kvh = hlit(r, 6, bad), hd = hlit(r, 7, bad);
    int64_t t = hlit(r, 8, bad);
    if (kvh <= 0 || qh % kvh) {
        return -1;
    }
    int G = (int)(qh / kvh);
    if (hd == 256) {
        if (G % 4 == 0) {
            flash3_run<256, 4, 2>(dr, denv, (int)kvh, (int)t, G);
        } else if (G % 2 == 0) {
            flash3_run<256, 2, 2>(dr, denv, (int)kvh, (int)t, G);
        } else {
            flash3_run<256, 1, 2>(dr, denv, (int)kvh, (int)t, G);
        }
        return 0;
    }
    if (hd == 512) {
        if (G % 2 == 0) {
            flash3_run<512, 2, 1>(dr, denv, (int)kvh, (int)t, G);
        } else {
            flash3_run<512, 1, 1>(dr, denv, (int)kvh, (int)t, G);
        }
        return 0;
    }
    return -1;
}

static int flash_tc_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int f32h,
                           int *bad)
{
    int o0 = f32h ? 5 : 7;
    int64_t qh = hlit(r, o0, bad), hd = hlit(r, o0 + 2, bad), t = hlit(r, o0 + 3, bad);
    if (hd != 256 && hd != 512) {
        return -1;
    }
    size_t smem = ((size_t)(FQ2 + FK2) * (hd + 8) + (size_t)hd * (FK2 + 8)) * sizeof(__half);
    unsigned warps = 2 * (unsigned)(hd / 256);
    dim3 grid((unsigned)qh, (unsigned)cdiv(t, FQ2));
    if (f32h) {
        cudaFuncSetAttribute(k_flash_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        k_flash_tc<1><<<grid, 32 * warps, smem, gg_stream>>>(dr, denv);
    } else {
        cudaFuncSetAttribute(k_flash_tc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        k_flash_tc<0><<<grid, 32 * warps, smem, gg_stream>>>(dr, denv);
    }
    return 0;
}

/* The product of an int4 matrix and a group of rows. The operands give x,
 * w, out, rows, cols, and t. The pointers must be literals, because the
 * kernel takes them as arguments: the compiler of a group passes arrays,
 * not slots. */
static void gemm_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr,
                        const int64_t *denv, int xk, int wk, int ok, int rk, int ck, int tk,
                        int *bad, int quant = 1)
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
        unsigned grid = (unsigned)cdiv(rows, ROWS_PER_BLOCK), blk = 32 * ROWS_PER_BLOCK;
        const uint8_t *wb = (const uint8_t *)w;
        switch (t) {
#define GG_NT(n) case n: k_mt_gemv_n<n><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols); break;
        GG_NT(1) GG_NT(2) GG_NT(3) GG_NT(4) GG_NT(5) GG_NT(6) GG_NT(7) GG_NT(8)
        GG_NT(9) GG_NT(10) GG_NT(11) GG_NT(12) GG_NT(13) GG_NT(14) GG_NT(15) GG_NT(16)
#undef GG_NT
        default:
            k_mt_gemv<<<grid, blk, 0, gg_stream>>>(x, wb, out, t, rows, cols);
        }
    } else if (gg_tc == 8 && g->tc && cols % Q8K == 0 && (size_t)t * cols <= g->xh_n) {
        /* int8 x: the scratch of xh holds the int8 values, then the scales. */
        size_t n = (size_t)t * cols;
        int8_t *xq = (int8_t *)g->xh;
        float *xs = (float *)((char *)g->xh + ((n + 255) & ~(size_t)255));
        /* quant 0: the scratch already holds x as int8, from the matrix
         * before this one in GP_INT4_MULTI4_MT. */
        if (quant) {
            k_quant_q8<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(
                x, xq, xs, n / 32);
        }
        dim3 grid((unsigned)cdiv(t, T2M), (unsigned)cdiv(rows, T2N));
        static int attr = 0;
        if (!attr) {
            cudaFuncSetAttribute(k_gemm_q8<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)Q8SMEM);
            cudaFuncSetAttribute(k_gemm_q8<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)Q8SMEM);
            attr = 1;
        }
        if ((cols / 32) % 8 == 0 && ((uintptr_t)w & 15) == 0) {
            /* The rows of w start at a multiple of 16 bytes. */
            k_gemm_q8<1><<<grid, 256, Q8SMEM, gg_stream>>>(xq, xs, (const uint8_t *)w, out, t,
                                                          rows, cols);
        } else {
            k_gemm_q8<0><<<grid, 256, Q8SMEM, gg_stream>>>(xq, xs, (const uint8_t *)w, out, t,
                                                          rows, cols);
        }
    } else if (gg_tc && g->tc && cols % TK == 0 && (size_t)t * cols <= g->xh_n) {
        size_t n = (size_t)t * cols;
        if (quant) {
            k_to_half<<<(unsigned)cdiv((int64_t)n / 4 + 1, 256), 256, 0, gg_stream>>>(x, g->xh, n);
        }
        if (gg_tc == 2) {
            k_gemm_tc<<<dim3((unsigned)cdiv(rows, TN), (unsigned)cdiv(t, TMD)), 256, 0,
                         gg_stream>>>(g->xh, (const uint8_t *)w, out, t, rows, cols);
        } else {
            k_gemm_tc2<<<dim3((unsigned)cdiv(t, T2M), (unsigned)cdiv(rows, T2N)), 256, 0,
                          gg_stream>>>(g->xh, (const uint8_t *)w, out, t, rows, cols);
        }
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
    case GP_BF16_LINEAR: {
        /* x, w, out, rows, cols, tokens */
        int64_t t = hlit(r, 5, &bad), rows = hlit(r, 3, &bad), cols = hlit(r, 4, &bad);
        if (cols % 32 != 0) {
            bad = 1;
        }
        if (t == 1) {
            k_bf16_linear<<<(unsigned)cdiv(rows, ROWS_PER_BLOCK), W, 0, s>>>(dr, denv);
        } else if (t <= MT_MAX) {
            k_mt_gemv_bf16<<<(unsigned)cdiv(rows, ROWS_PER_BLOCK), W, 0, s>>>(
                (const float *)(intptr_t)hlit(r, 0, &bad), (const uint16_t *)(intptr_t)hlit(r, 1, &bad),
                (float *)(intptr_t)hlit(r, 2, &bad), (int)t, (int)rows, (int)cols);
        } else if (gg_tc && g->tc && cols % BHK == 0 && (size_t)t * cols <= g->xh_n) {
            const float *x = (const float *)(intptr_t)hlit(r, 0, &bad);
            size_t n = (size_t)t * cols;
            k_to_half<<<(unsigned)cdiv((int64_t)n / 4 + 1, 256), 256, 0, s>>>(x, g->xh, n);
            k_gemm_bh<<<dim3((unsigned)cdiv(t, T2M), (unsigned)cdiv(rows, T2N)), 256, 0, s>>>(
                g->xh, (const uint16_t *)(intptr_t)hlit(r, 1, &bad),
                (float *)(intptr_t)hlit(r, 2, &bad), (int)t, (int)rows, (int)cols);
        } else {
            k_gemm<2><<<dim3((unsigned)cdiv(rows, GN), (unsigned)cdiv(t, GM)), 256, 0, s>>>(
                (const float *)(intptr_t)hlit(r, 0, &bad), (const void *)(intptr_t)hlit(r, 1, &bad),
                (float *)(intptr_t)hlit(r, 2, &bad), (int)t, (int)rows, (int)cols);
        }
        break;
    }
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
        if (hlit(r, 8, &bad) == 1) {
            attn_launch(g, r, dr, denv, &bad);
        } else {
            /* A group: every query sees the keys of a window before it, as
             * the decode step does. */
            if (hlit(r, 7, &bad) % FD != 0 || hlit(r, 7, &bad) > 512) {
                bad = 1;
            }
            if (!(gg_tc && g->tc && (flash3_launch(r, dr, denv, &bad) == 0 ||
                                     flash_tc_launch(r, dr, denv, 1, &bad) == 0))) {
                k_flash_qc_mt<1><<<dim3((unsigned)hlit(r, 5, &bad),
                                        (unsigned)cdiv(hlit(r, 8, &bad), FQ)), 256, 0, s>>>(
                    dr, denv);
            }
        }
        break;
    case GP_ATTN_F32:
    case GP_ATTN_QC:
        attn_launch(g, r, dr, denv, &bad);
        break;
    case GP_HOT_SPLIT:
        k_hot_split<<<1, 1, 0, s>>>(dr, denv);
        break;
    case GP_HOT_MOE: {
        unsigned t = (unsigned)hlit(r, 15, &bad);
        unsigned k = (unsigned)hlit(r, 10, &bad) * t;
        k_hot_gu<<<dim3((unsigned)cdiv(hlit(r, 11, &bad), ROWS_PER_BLOCK), k), W, 0, s>>>(
            dr, denv);
        k_hot_gelu<<<k, T, 0, s>>>(dr, denv);
        k_hot_dn<<<dim3((unsigned)cdiv(hlit(r, 13, &bad), ROWS_PER_BLOCK), k), W, 0, s>>>(
            dr, denv);
        k_hot_sum<<<dim3((unsigned)cdiv(hlit(r, 13, &bad), T), t), T, 0, s>>>(dr, denv);
        break;
    }
    case GP_F32_LINEAR:
        k_f32_linear<<<(unsigned)cdiv(hlit(r, 3, &bad), ROWS_PER_BLOCK), W, 0, s>>>(dr, denv);
        break;
    case GP_DRAFT_HEAD: {
        /* The logits of the centroids come from a GP_F32_LINEAR before it. */
        if (hlit(r, 7, &bad) > 2048) {
            bad = 1;
        }
        k_draft_top<<<1, 1024, 0, s>>>(dr, denv);
        k_draft_sel<<<(unsigned)cdiv(hlit(r, 8, &bad) * hlit(r, 9, &bad), ROWS_PER_BLOCK), W, 0,
                      s>>>(dr, denv);
        k_draft_best<<<1, 1024, 0, s>>>(dr, denv);
        break;
    }
    case GP_ARGMAX:
        k_argmax<<<1, 1024, 0, s>>>(dr, denv);
        break;
    case GP_ADD_NORM:
        k_add_norm<<<(unsigned)hlit(r, 4, &bad), T, 0, s>>>(dr, denv);
        break;
    case GP_HOT_SPLIT_MT:
        k_hot_split_mt<<<(unsigned)cdiv(hlit(r, 5, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_INT4_LINEAR_MT:
        /* x, w, s, out, rows, cols, t */
        gemm_launch(g, r, dr, denv, 0, 1, 3, 4, 5, 6, &bad);
        break;
    case GP_INT4_MULTI4_MT:
        /* x, cols, t, then (w, s, out, rows) for up to four matrices */
        for (int m = 0, first = 1; m < 4; ++m) {
            if (r->v[3 + 4 * m] != 0) {
                gemm_launch(g, r, dr, denv, 0, 3 + 4 * m, 5 + 4 * m, 6 + 4 * m, 1, 2, &bad,
                            first);
                first = 0;
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
        if (hlit(r, 10, &bad) > MT_MAX && gg_tc && g->tc &&
            flash_tc_launch(r, dr, denv, 0, &bad) == 0) {
            /* the tensor cores */
        } else if (hlit(r, 10, &bad) > MT_MAX && hlit(r, 9, &bad) % FD == 0) {
            k_flash_qc_mt<0><<<dim3((unsigned)hlit(r, 7, &bad),
                                    (unsigned)cdiv(hlit(r, 10, &bad), FQ)), 256, 0, s>>>(dr, denv);
        } else {
            k_attn_qc_mt<<<dim3((unsigned)hlit(r, 7, &bad),
                                (unsigned)cdiv(hlit(r, 10, &bad), QT)), QT * 16, 0, s>>>(dr, denv);
        }
        break;
    case GP_ROUTER_MT: {
        int64_t t = hlit(r, 11, &bad), ex = hlit(r, 5, &bad);
        k_router_norm_mt<<<(unsigned)t, T, 0, s>>>(dr, denv);
        if (t <= MT_MAX) {
            k_router_logits_mt<<<dim3((unsigned)cdiv(ex, ROWS_PER_BLOCK), (unsigned)t), W, 0, s>>>(
                dr, denv);
        } else {
            k_gemm<1><<<dim3((unsigned)cdiv(ex, GN), (unsigned)cdiv(t, GM)), 256, 0, s>>>(
                (const float *)(intptr_t)r->v[12], (const void *)(intptr_t)r->v[2],
                (float *)(intptr_t)r->v[13], (int)t, (int)ex, (int)hlit(r, 4, &bad));
        }
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
        if (gg_tc && g->tc && cols % TK == 0 && inner % TK == 0 &&
            (size_t)t * cols <= g->xh_n && (size_t)pairs * inner <= g->xh_n) {
            size_t n1 = (size_t)t * cols, n2 = (size_t)pairs * inner;
            k_to_half<<<(unsigned)cdiv((int64_t)n1 / 4 + 1, 256), 256, 0, s>>>(h, g->xh, n1);
            k_moe_gemm_tc<<<dim3((unsigned)cdiv(gu_rows, TN), max_tiles), 128, 0, s>>>(
                dr, denv, g->xh, 1, gu, gu_rows, cols, act);
            k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
            k_to_half<<<(unsigned)cdiv((int64_t)n2 / 4 + 1, 256), 256, 0, s>>>(act2, g->xh, n2);
            k_moe_gemm_tc<<<dim3((unsigned)cdiv(dn_rows, TN), max_tiles), 128, 0, s>>>(
                dr, denv, g->xh, 0, dn, dn_rows, inner, de);
        } else {
            k_moe_gemm<<<dim3((unsigned)cdiv(gu_rows, GN), max_tiles), 256, 0, s>>>(
                dr, denv, h, 1, gu, gu_rows, cols, act);
            k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
            k_moe_gemm<<<dim3((unsigned)cdiv(dn_rows, GN), max_tiles), 256, 0, s>>>(
                dr, denv, act2, 0, dn, dn_rows, inner, de);
        }
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

void gg_set_tc(int on)
{
    gg_tc = on;    /* 0: float32, 1: k_gemm_tc2, 2: k_gemm_tc (for a test), 8: k_gemm_q8 */
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
    /* Bit 0 of the flags: the CUDA graph. Bit 1: no tensor cores for this
     * program (the 26B keeps float32 products; see gpu.py). */
    g->use_graph = use_graph & 1;
    g->tc = (use_graph & 2) ? 0 : 1;
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
    /* The float16 scratch of the products of large groups. */
    size_t need = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        size_t n = 0;
        if (r->op == GP_INT4_LINEAR_MT && r->v[6] > MT_MAX) {
            n = (size_t)r->v[6] * (size_t)r->v[5];
        } else if (r->op == GP_INT4_MULTI4_MT && r->v[2] > MT_MAX) {
            n = (size_t)r->v[2] * (size_t)r->v[1];
        } else if (r->op == GP_BF16_LINEAR && r->v[5] > MT_MAX) {
            n = (size_t)r->v[5] * (size_t)r->v[4];
        } else if (r->op == GP_MOE_GPU) {
            size_t a = (size_t)r->v[3] * (size_t)r->v[8];
            size_t b = (size_t)r->v[3] * (size_t)r->v[4] * (size_t)r->v[10];
            n = a > b ? a : b;
        }
        need = n > need ? n : need;
    }
    if (need > 0 && cudaMalloc(&g->xh, need * sizeof(__half)) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "gg_load: no memory for the float16 scratch");
        return NULL;
    }
    g->xh_n = need;
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
        cudaError_t err = cudaEventSynchronize(b);
        if (err != cudaSuccess) {
            snprintf(gg_error, sizeof(gg_error), "record %d (operation %d): %s", pc, r->op,
                     cudaGetErrorString(err));
            return -1;
        }
        CK(cudaEventElapsedTime(&ms[pc], a, b));
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return rc;
}

/* The output head: out = the Q6_K matrix w times the nx rows of x, with the
 * soft cap cap (0 for none). nx is at most HEAD_MAX. out has nx rows of rows
 * values. The kernel reads the head one time for all the rows. All pointers
 * are device addresses. The call does not wait for the GPU. */
int gg_q6k_head(const void *w, const float *x, float *out, int rows, int cols, float cap,
                int nx)
{
    if (nx < 1 || nx > HEAD_MAX) {
        snprintf(gg_error, sizeof(gg_error), "gg_q6k_head: 1 to %d rows of x", HEAD_MAX);
        return -1;
    }
    k_q6k_head<<<(unsigned)cdiv(rows, ROWS_PER_BLOCK), 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
        (const uint8_t *)w, x, out, rows, cols, cap, nx);
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
    cudaFree(g->xh);
    cudaFreeHost(g->henv);
    free(g->hcode);
    free(g);
    return 0;
}

}  /* extern "C" */
