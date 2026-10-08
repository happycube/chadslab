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
 * gg_q6k_head, or gg_q4_head for a file that keeps the head in Q4_0. The
 * launch of any other operation gives an error.
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
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The host clock, in s. */
static double gg_clock(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

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
    /* the int8 cache (NP_GEMMA_KV_INT8): GP_KV_WRITE, GP_ATTN_QC, GP_ATTN_QC_MT
     * with int8 values and a scale of max |x| / 127 for each group of 32 */
    GP_KV_WRITE8 = 58, GP_ATTN_Q8 = 59, GP_ATTN_Q8_MT = 60,
    /* int16 keys and int8 values (NP_GEMMA_KV_INT8 v) */
    GP_KV_WRITEV8 = 61, GP_ATTN_V8 = 62, GP_ATTN_V8_MT = 63,
    GP_INT4_LINEAR_MT = 36, GP_INT4_MULTI4_MT = 37, GP_GELU_MUL_ROWS = 38,
    GP_ATTN_QC_MT = 52,
    GP_ROUTER = 64, GP_ROUTER_MT = 66,
    /* the TQ6 cache of the Qwen models (np_gemma/tq6.py) */
    GP_KV_WRITETQ = 69, GP_ATTN_TQ = 70, GP_ATTN_TQ_MT = 71, GP_TQ_ROT = 72,
    GP_TO_HOST = 84, GP_CPU_JOIN = 85, GP_TO_DEV = 86, GP_HOT_SPLIT = 87,
    GP_HOT_MOE = 88, GP_MOE_GPU = 89, GP_FETCH = 90, GP_FETCH_WAIT = 91,
    GP_FETCH_DONE = 92, GP_HOT_SPLIT_MT = 93, GP_F32_LINEAR = 94, GP_DRAFT_HEAD = 95,
    GP_ARGMAX = 96, GP_ADD_NORM = 97, GP_COUNT = 98,
    /* Qwen3.5 (QWEN_PLAN.md, phase 4) */
    GP_ROUTER_TOPK = 103, GP_GDN = 104, GP_ATTN_PREP = 105, GP_SIGMUL = 106,
    GP_KQ_QUANT = 107, GP_KQ_LINEAR = 108, GP_KQ_HOT_MOE = 110, GP_KQ_MULTI = 111,
    GP_ADD_RMS = 112, GP_KQ_GROUP_MOE = 113,
    /* Qwen3.8 (QWEN38_PLAN.md, phase 5; csrc/hyperconn.c has the CPU forms) */
    GP_HC_NORM = 114, GP_HC_ACT = 115, GP_HC_MIX = 116, GP_HC_ADD = 117, GP_PLE_GATE = 118,
    GP_PLE_CONV = 119, GP_QSA_SELECT = 120, GP_ATTN_QSA = 121, GP_HC_CAT = 122,
    GP_CPU_START = 124, GP_CPU_WAIT = 125,
    /* the handoff to the CPU inside a graph (flags in pinned memory) */
    GP_SIGNAL = 126, GP_AWAIT = 127, GP_D2H = 128, GP_H2D = 129, GP_CPU_TASK = 130,
    GP_FFN_OUT = 131, GP_SOFTCAP = 132,
    /* The media encoders (np_gemma/gemma4_encoders.py, program form) */
    GP_ENC_LINEAR = 133, GP_ENC_RMS = 134, GP_ENC_GELU_MUL = 135, GP_ENC_ADD = 136,
    GP_ENC_ROPE2D = 137, GP_ENC_ATTN = 138, GP_ENC_SILU = 139, GP_ENC_MUL_VEC = 140,
    GP_ENC_GLU = 141, GP_ENC_DWCONV = 142, GP_ENC_LOCAL_ATTN = 143, GP_ENC_CLAMP = 144,
    GP_ENC_BIAS_CLAMP = 145, GP_ENC_LNORM = 146, GP_ENC_GELU = 147,
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

/* Where a run is (gg_where): the program, and the record (and its
 * operation) that the host queued last, in the segment [s0, s1) of its
 * launches. The kernels run after the host queues them, so a kernel that
 * fails shows its error at a later record: the record at fault is at or
 * before this one (since the last wait). NP_GEMMA_GPU_SYNC_CHECK=1: no
 * graphs, and a wait after each record, so that an error names the record
 * that made it (slow: for a replay of a crash). */
static const void *gg_where_prog;
static int gg_where_pc = -1, gg_where_op = -1, gg_where_s0 = -1, gg_where_s1 = -1;
static int gg_sync_chk = -1;

static int gg_sync_check(void)
{
    if (gg_sync_chk < 0) {
        const char *v = getenv("NP_GEMMA_GPU_SYNC_CHECK");
        gg_sync_chk = v != NULL && *v != 0 && strcmp(v, "0") != 0;
    }
    return gg_sync_chk;
}

static void gg_at(const void *prog, int pc, int op, int s0, int s1)
{
    gg_where_prog = prog;
    gg_where_pc = pc;
    gg_where_op = op;
    gg_where_s0 = s0;
    gg_where_s1 = s1;
}

/* NP_GEMMA_GPU_SYNC_CHECK: wait for the record pc; its error names it. */
#define GG_SYNC_AT(pc, op)                                                      \
    do {                                                                        \
        if (gg_sync_check()) {                                                  \
            cudaError_t err_ = cudaStreamSynchronize(gg_stream);                \
            if (err_ == cudaSuccess) {                                          \
                err_ = cudaGetLastError();                                      \
            }                                                                   \
            if (err_ != cudaSuccess) {                                          \
                snprintf(gg_error, sizeof(gg_error),                            \
                         "record %d (operation %d), NP_GEMMA_GPU_SYNC_CHECK: %s", \
                         (pc), (op), cudaGetErrorString(err_));                 \
                return -1;                                                      \
            }                                                                   \
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

/* The int8 x of the one-token products (k_kq_quant_x): 8 adjacent lanes
 * hold the 32 values of block bi, 4 values each (sub = the lane % 8). All
 * lanes of the warp must call it; a lane with live false gives zeros and
 * writes nothing. */
__device__ __forceinline__ void quant8_block(float4 v, bool live, size_t bi, int sub, int8_t *q,
                                             float *xs, float *xsum)
{
    if (!live) {
        v = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    float s2 = m / 127.0f;
    int a = s2 > 0.f ? __float2int_rn(v.x / s2) : 0, b = s2 > 0.f ? __float2int_rn(v.y / s2) : 0;
    int c = s2 > 0.f ? __float2int_rn(v.z / s2) : 0, d = s2 > 0.f ? __float2int_rn(v.w / s2) : 0;
    int sum = a + b + c + d;
    for (int o = 4; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    if (!live) {
        return;
    }
    char4 c4;
    c4.x = (signed char)a;
    c4.y = (signed char)b;
    c4.z = (signed char)c;
    c4.w = (signed char)d;
    *(char4 *)(q + bi * 32 + sub * 4) = c4;
    if (sub == 0) {
        xs[bi] = s2;
        xsum[bi] = s2 * (float)sum;
    }
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

/* Programmatic dependent launch. In a graph with programmatic edges
 * (gg_pdl_edges), a kernel can start before the kernel before it ends. Each
 * kernel waits for the data of the kernels before it (griddepcontrol.wait),
 * then lets the next kernel start (griddepcontrol.launch_dependents). The
 * next kernel starts only when every block of this kernel has passed that
 * point. Outside such a graph the two instructions do nothing. The
 * instructions need sm_90 or later; an older GPU (the 3090 is sm_86) runs
 * the graphs with ordinary edges (gg_pdl_on). */
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define PDL_START()                                                           \
    asm volatile("griddepcontrol.wait;\n" ::: "memory");                      \
    asm volatile("griddepcontrol.launch_dependents;\n" :::)
#else
#define PDL_START() do { } while (0)
#endif


/* The norm of each row of cols values: out = x s w, where s = 1 / sqrt(mean
 * of x^2 + eps). A null w gives out = x s. One block for each row. The
 * arguments give the operands of x, w, out, cols, and eps. Thus the kernel
 * serves GP_RMS_NORM and the first step of GP_RMS_NORM_MULTI4. */
__global__ void k_rms_norm(const gp_rec *r, const int64_t *e, int xk, int wk, int ok,
                           int ck, int ek)
{
    PDL_START();
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

/* GP_FFN_OUT: m, w1, ex, w2, w, x, out, rows, cols, eps, scale, wn, out2.
 * The end of a layer of the 26B in one pass (program.py, ffn_out):
 * f = norm(m) w1 + norm(ex) w2; out = (x + norm(f) w) scale; with wn,
 * out2 = norm(out) wn (the input norm of the next layer). The operations and
 * their order are those of k_rms_norm, k_add, and k_mul_s, so the values are
 * the same. out can be x. One block of 256 threads for each row; f stays in
 * shared memory (cols at most FFN_COLS). */
#define FFN_COLS 4096
__global__ void k_ffn_out(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float fs[FFN_COLS];
    int cols = DI(8);
    size_t ro = (size_t)blockIdx.x * cols;
    const float *m = DP(const float, 0) + ro, *w1 = DP(const float, 1);
    const float *ex = DP(const float, 2) + ro, *w2 = DP(const float, 3);
    const float *w = DP(const float, 4), *x = DP(const float, 5) + ro;
    float *out = DP(float, 6) + ro;
    float eps = df(r, e, 9), scale = df(r, e, 10);
    float sm = 0.f, se = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        sm += m[i] * m[i];
    }
    sm = block_sum(sm);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        se += ex[i] * ex[i];
    }
    se = block_sum(se);
    float s1 = 1.0f / sqrtf(sm / (float)cols + eps), s2 = 1.0f / sqrtf(se / (float)cols + eps);
    float sf = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float f = __fadd_rn(__fmul_rn(__fmul_rn(m[i], s1), w1[i]),
                            __fmul_rn(__fmul_rn(ex[i], s2), w2[i]));
        fs[i] = f;
        sf += f * f;
    }
    sf = block_sum(sf);
    float s3 = 1.0f / sqrtf(sf / (float)cols + eps);
    float so = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = __fmul_rn(__fadd_rn(x[i], __fmul_rn(__fmul_rn(fs[i], s3), w[i])), scale);
        out[i] = v;
        so += v * v;
    }
    const float *wn = DP(const float, 11);
    if (wn != NULL) {
        float *out2 = DP(float, 12) + ro;
        so = block_sum(so);
        float s4 = 1.0f / sqrtf(so / (float)cols + eps);
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            out2[i] = __fmul_rn(__fmul_rn(out[i], s4), wn[i]);
        }
    }
}

/* a, b, out, n */
__global__ void k_add(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(3)) {
        DP(float, 2)[i] = DP(const float, 0)[i] + DP(const float, 1)[i];
    }
}

/* x, s (float), out, n */
__global__ void k_mul_s(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(3)) {
        DP(float, 2)[i] = DP(const float, 0)[i] * df(r, e, 1);
    }
}

/* GP_ADD_NORM: o, w, x, out, rows, cols, eps, scale, w2, out2. For each row,
 * out = (x + o s w) scale, where s = 1 / sqrt(mean of o^2 + eps). This is
 * GP_RMS_NORM of o, GP_ADD, and GP_MUL_S in one pass, with the same
 * operations in the same order (as k_rms_norm), so the values are the same.
 * out can be x. With out2, the kernel also writes the norm of out with the
 * weights w2 there, as GP_RMS_NORM does. One block of 256 threads for each
 * row. */
__global__ void k_add_norm(const gp_rec *r, const int64_t *e)
{
    PDL_START();
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
    float *out2 = DP(float, 9);
    if (out2 != NULL) {
        /* Each thread reads back the values that it wrote. */
        const float *w2 = DP(const float, 8);
        out2 += (size_t)blockIdx.x * cols;
        float ss2 = 0.f;
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            ss2 += out[i] * out[i];
        }
        ss2 = block_sum(ss2);
        float s2 = 1.0f / sqrtf(ss2 / (float)cols + eps);
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            out2[i] = out[i] * s2 * w2[i];
        }
    }
}

/* k_add_norm with the values of a row in registers: each thread holds up to
 * AN_V float4 of the row, so the kernel reads o, x, and w once and does not
 * read out back. cols % 4 == 0, cols <= 4 AN_V blockDim.x, and 16-byte
 * addresses. With q, it also writes the int8 x of the final values (out2, or
 * out without it) for the product that follows (as k_kq_quant_x). */
#define AN_V 4

__global__ void k_add_norm_v(const gp_rec *r, const int64_t *e, int8_t *q, float *xs, float *xsum)
{
    PDL_START();
    int cols = DI(5), n4 = cols / 4;
    size_t row = blockIdx.x;
    const float4 *o = DP(const float4, 0) + row * n4;
    const float4 *w = DP(const float4, 1);
    const float4 *x = DP(const float4, 2) + row * n4;
    float4 *out = DP(float4, 3) + row * n4;
    float eps = df(r, e, 6), scale = df(r, e, 7);
    float4 v[AN_V], xa[AN_V], wa[AN_V];
    float ss = 0.f;
    /* All the loads start before the first sum of the block, so their
     * latencies overlap (one row is one block, and the kernel is bound by
     * the latency). */
    #pragma unroll
    for (int k = 0; k < AN_V; ++k) {
        int j = threadIdx.x + k * blockDim.x;
        float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
        v[k] = j < n4 ? o[j] : z;
        xa[k] = j < n4 ? x[j] : z;
        wa[k] = j < n4 ? w[j] : z;
    }
    #pragma unroll
    for (int k = 0; k < AN_V; ++k) {
        ss += v[k].x * v[k].x + v[k].y * v[k].y + v[k].z * v[k].z + v[k].w * v[k].w;
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)cols + eps);
    #define AN_F(c) __fmul_rn(__fadd_rn(xv.c, __fmul_rn(__fmul_rn(v[k].c, s), wv.c)), scale)
    #pragma unroll
    for (int k = 0; k < AN_V; ++k) {
        int j = threadIdx.x + k * blockDim.x;
        if (j < n4) {
            float4 xv = xa[k], wv = wa[k];
            v[k] = make_float4(AN_F(x), AN_F(y), AN_F(z), AN_F(w));
            out[j] = v[k];
        }
    }
    #undef AN_F
    float4 *out2 = DP(float4, 9);
    if (out2 != NULL) {
        const float4 *w2 = DP(const float4, 8);
        out2 += row * n4;
        #pragma unroll
        for (int k = 0; k < AN_V; ++k) {
            int j = threadIdx.x + k * blockDim.x;
            if (j < n4) {
                wa[k] = w2[j];
            }
        }
        float ss2 = 0.f;
        #pragma unroll
        for (int k = 0; k < AN_V; ++k) {
            int j = threadIdx.x + k * blockDim.x;
            if (j < n4) {
                ss2 += v[k].x * v[k].x + v[k].y * v[k].y + v[k].z * v[k].z + v[k].w * v[k].w;
            }
        }
        ss2 = block_sum(ss2);
        float s2 = 1.0f / sqrtf(ss2 / (float)cols + eps);
        #pragma unroll
        for (int k = 0; k < AN_V; ++k) {
            int j = threadIdx.x + k * blockDim.x;
            if (j < n4) {
                float4 wv = wa[k];
                v[k] = make_float4(v[k].x * s2 * wv.x, v[k].y * s2 * wv.y, v[k].z * s2 * wv.z,
                                   v[k].w * s2 * wv.w);
                out2[j] = v[k];
            }
        }
    }
    if (q != NULL) {
        /* cols % 32 == 0: a block of 32 is 8 adjacent threads. */
        #pragma unroll
        for (int k = 0; k < AN_V; ++k) {
            int j = threadIdx.x + k * blockDim.x;
            quant8_block(v[k], j < n4, (row * n4 + j) / 8, threadIdx.x % 8, q, xs, xsum);
        }
    }
}

/* k_gelu_mul_rows with 4 values for each thread, and with q the int8 x of
 * out (as k_add_norm_v). n = rows cols, n % 4 == 0, 16-byte addresses. */
__global__ void k_gelu_mul_rows_v(const gp_rec *r, const int64_t *e, int8_t *q, float *xs,
                                  float *xsum)
{
    PDL_START();
    size_t n4 = (size_t)DI(3) * DI(4) / 4, j = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool live = j < n4;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (live) {
        float4 g = DP(const float4, 0)[j], u = DP(const float4, 1)[j];
        #define GM_F(c) 0.5f * g.c * (1.0f + tanhf(0.7978845608028654f * \
                                                   (g.c + 0.044715f * g.c * g.c * g.c))) * u.c
        v = make_float4(GM_F(x), GM_F(y), GM_F(z), GM_F(w));
        #undef GM_F
        DP(float4, 2)[j] = v;
    }
    if (q != NULL) {
        quant8_block(v, live, j / 8, threadIdx.x % 8, q, xs, xsum);
    }
}

/* GP_COUNT: idx, counts, top_k, t, experts. counts[x] += the count of x in the first
 * t rows of idx (top_k values each): the selections of the router of a
 * group, for the cache of hot experts (HotCache.seed). */
__global__ void k_count(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int *idx = DP(const int, 0);
    int *counts = DP(int, 1);
    int n = DI(2) * DI(3);
    for (int p = blockIdx.x * blockDim.x + threadIdx.x; p < n; p += gridDim.x * blockDim.x) {
        atomicAdd(&counts[idx[p]], 1);
    }
}

/* src, dst, bytes */
__global__ void k_copy(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)di(r, e, 2)) {
        DP(uint8_t, 1)[i] = DP(const uint8_t, 0)[i];
    }
}

/* x, out, n. The tanh form of GELU, as ops.gelu_tanh. */
__global__ void k_gelu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
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
    PDL_START();
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
    PDL_START();
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
 * cache. The int16 form is the form of gemma_quant_group32_i16. Each group
 * of 32 values has a scale of max |x| / 32767. Each value is x / scale,
 * rounded to the nearest integer. quant32_i16_warp does a group with a warp:
 * lane l holds x[l]. */
__device__ __forceinline__ void quant32_i16_warp(float xl, int16_t *q, float *scale, int l)
{
    float amax = fabsf(xl);
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
    }
    float sc = amax > 0.f ? amax / 32767.0f : 1e-12f;
    int v = __float2int_rn(xl / sc);
    q[l] = (int16_t)max(-32767, min(32767, v));
    if (l == 0) {
        *scale = sc;
    }
}

/* A warp for each group of 32 values (KVW_WARPS warps in a block). A thread
 * for each group took 8.6 us for a layer of the 12B (64 groups): the loads
 * of a thread are not coalesced, and its loop is serial. */
#define KVW_WARPS 4

__global__ void k_kv_write(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int g = blockIdx.x * KVW_WARPS + threadIdx.x / 32, l = threadIdx.x % 32;
    int n = DI(8);
    if (g >= n / 32) {
        return;
    }
    size_t i = (size_t)g * 32 + l;
    float kl = DP(const float, 0)[i], vl = DP(const float, 1)[i];
    if (DP(float, 2)) {
        DP(float, 2)[i] = kl;
        DP(float, 3)[i] = vl;
    }
    if (DP(int16_t, 4)) {
        quant32_i16_warp(kl, DP(int16_t, 4) + (size_t)g * 32, DP(float, 5) + g, l);
        quant32_i16_warp(vl, DP(int16_t, 6) + (size_t)g * 32, DP(float, 7) + g, l);
    }
}

/* k_kv_write for the int8 cache: a scale of max |x| / 127 for each group of
 * 32 values, as gemma_quant_group32_i8 of the CPU. The int8 cache has no
 * float rows. */
template <bool K8>
__global__ void k_kv_write8(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int g = blockIdx.x * KVW_WARPS + threadIdx.x / 32, l = threadIdx.x % 32;
    int n = DI(8);
    if (g >= n / 32) {
        return;
    }
    size_t i = (size_t)g * 32 + l;
    for (int kv = 0; kv < 2; ++kv) {
        float x = DP(const float, kv)[i];
        float amax = fabsf(x);
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, o));
        }
        /* the keys of K8 false: int16, as k_kv_write */
        bool i8 = kv || K8;
        float lim = i8 ? 127.0f : 32767.0f;
        float sc = amax > 0.f ? amax / lim : 1e-12f;
        int v = max(-(int)lim, min((int)lim, __float2int_rn(x / sc)));
        if (i8) {
            DP(int8_t, kv ? 6 : 4)[i] = (int8_t)v;
        } else {
            DP(int16_t, 4)[i] = (int16_t)v;
        }
        if (l == 0) {
            DP(float, kv ? 7 : 5)[g] = sc;
        }
    }
}

/* The TQ6 cache of the Qwen models (np_gemma/tq6.py). The tables and the
 * order of the operations are those of tq6_* of the CPU, so both give the
 * same bits: a warp for each group of 32 values, lane l with value l. */
#include "tq6_tables.h"
__device__ const float tq6_cb_d[64] = TQ6_CODEBOOK;
__device__ const float tq6_edges_d[63] = TQ6_EDGES;

/* The rotation (inverse false) of the 32 values of a warp, or its inverse. */
__device__ __forceinline__ float tq6_rot_warp(float x, int l, bool inverse)
{
    bool neg = (TQ6_SIGNS >> l) & 1u;
    if (!inverse && neg) {
        x = -x;
    }
    #pragma unroll
    for (int h = 1; h < 32; h *= 2) {
        float p = __shfl_xor_sync(0xffffffff, x, h);
        x = (l & h) ? p - x : x + p;
    }
    x *= 0.17677669529663687f;
    return inverse && neg ? -x : x;
}

/* k_kv_write for the TQ6 cache: the operands of GP_KV_WRITE8; 24 bytes and
 * a norm for each group of 32 values (tq6_quant_group). */
__global__ void k_kv_write_tq(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int g = blockIdx.x * KVW_WARPS + threadIdx.x / 32, l = threadIdx.x % 32;
    int n = DI(8);
    if (g >= n / 32) {
        return;
    }
    for (int kv = 0; kv < 2; ++kv) {
        float y = tq6_rot_warp(DP(const float, kv)[(size_t)g * 32 + l], l, false);
        float s = __fmul_rn(y, y);
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(0xffffffff, s, o);
        }
        float nrm = sqrtf(s);
        float u = y * (1.0f / (nrm > 1e-30f ? nrm : 1e-30f));
        int lo = 0, hi = 63;
        while (lo < hi) {
            int m = (lo + hi) / 2;
            if (tq6_edges_d[m] < u) {
                lo = m + 1;
            } else {
                hi = m;
            }
        }
        int i8 = __shfl_sync(0xffffffff, lo, (l + 8) & 31);
        int i16 = __shfl_sync(0xffffffff, lo, (l + 16) & 31);
        int i24 = __shfl_sync(0xffffffff, lo, (l + 24) & 31);
        uint8_t *b = DP(uint8_t, kv ? 6 : 4) + (size_t)g * 24;
        if (l < 16) {
            b[l] = (uint8_t)((lo & 15) | ((i16 & 15) << 4));
        }
        if (l < 8) {
            b[16 + l] = (uint8_t)((lo >> 4) | ((i8 >> 4) << 2) | ((i16 >> 4) << 4) |
                                  ((i24 >> 4) << 6));
        }
        if (l == 0) {
            DP(float, kv ? 7 : 5)[g] = nrm;
        }
    }
}

/* GP_TQ_ROT: x, groups, inverse. The rotation of each group of 32 values of
 * x in place, or its inverse. */
__global__ void k_tq_rot(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int g = blockIdx.x * KVW_WARPS + threadIdx.x / 32, l = threadIdx.x % 32;
    if (g >= DI(1)) {
        return;
    }
    float *x = DP(float, 0) + (size_t)g * 32;
    x[l] = tq6_rot_warp(x[l], l, DI(2) != 0);
}

/* Values o to o + 7 (o a multiple of 8) of the TQ6 groups at p: the values
 * of the codebook, without the norm. Byte j < 16 of a group has the low 4
 * bits of indices j and j + 16; byte 16 + j has the high 2 bits of indices
 * j, j + 8, j + 16, j + 24. */
__device__ __forceinline__ void tq6x8(const uint8_t *p, size_t o, float *out,
                                      const float *cb = tq6_cb_d)
{
    const uint8_t *b = p + (o / 32) * 24;
    int w = (int)(o % 32);
    uint2 nb = *(const uint2 *)(b + (w & 15)), hb = *(const uint2 *)(b + 16);
    int ns = w >= 16 ? 4 : 0, hs = 2 * (w / 8);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        uint32_t nw = t < 4 ? nb.x : nb.y, hw = t < 4 ? hb.x : hb.y;
        int sh = 8 * (t % 4);
        uint32_t idx = ((nw >> (sh + ns)) & 15u) | (((hw >> (sh + hs)) & 3u) << 4);
        out[t] = cb[idx];
    }
}

/* ---------- matrices ---------- */

/* One row of int4 blocks against x. The row has cols / 32 blocks of 18
 * bytes: a float16 scale, then 16 bytes. The low 4 bits of byte i are weight
 * i, and the high 4 bits are weight i + 16. A value is the 4-bit number minus
 * 8. Four lanes share a block, so the lanes of a warp read adjacent values of
 * x. Return the sum to all lanes of the warp. */
/* The value nibble - 8 of the 4 bits of q at bit s, as a float: the bits
 * 2^23 + nibble, minus 2^23 + 8. That is exact, and it gives the float of
 * (float)((int)nibble - 8) with two full-rate operations, not a conversion
 * of an integer (a quarter of the rate). A group of 3 tokens was limited by
 * the arithmetic of these conversions. */
__device__ __forceinline__ float i4f(uint32_t q, int s)
{
    return __uint_as_float(((q >> s) & 15u) | 0x4B000000u) - 8388616.f;
}

/* The 8 products of the 4 bytes q (lane sub of a block) with x, in one
 * order. int4_row and k_mt_int4_rows use it, so a token of a group gets the
 * bits of a step. */
__device__ __forceinline__ float int4_part(uint32_t q, float4 xl, float4 xh)
{
    return i4f(q, 0) * xl.x
         + i4f(q, 8) * xl.y
         + i4f(q, 16) * xl.z
         + i4f(q, 24) * xl.w
         + i4f(q, 4) * xh.x
         + i4f(q, 12) * xh.y
         + i4f(q, 20) * xh.z
         + i4f(q, 28) * xh.w;
}

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
        float acc = int4_part(q, xl, xh);
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
    const uint8_t *wr = DP(const uint8_t, wk) + (size_t)row * (cols / 32) * 18;
    PDL_START();
    float v = int4_row(wr, DP(const float, xk), cols);
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
    const uint8_t *wr = DP(const uint8_t, m0 + 4 * m) + (size_t)row * (cols / 32) * 18;
    PDL_START();
    float v = int4_row(wr, DP(const float, xk), cols);
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
    PDL_START();
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
    PDL_START();
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
    PDL_START();
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

/* k_qkv_norm and then k_rope in one kernel (GP_QKV_NORM_ROPE): 128 threads
 * for a row, and thread t keeps the values t + 128 k (k < NK = head_dim /
 * 128) in registers. The sum of the squares and the values are those of the
 * two kernels: the same order, and the rope reads the same floats. The rope
 * pairs value i with value i + head_dim / 2, which is k with k + NK / 2. */
template <int NK>
__global__ void __launch_bounds__(128) k_qkv_norm_rope(const gp_rec *r, const int64_t *e,
                                                       qkv_ix ix, rope_ix rx)
{
    PDL_START();
    int row = blockIdx.x;
    int qr = DI(ix.qr), kr = DI(ix.kr), hd = DI(ix.hd);
    float *x;
    const float *w;
    int tok = -1;
    if (row < qr) {
        x = DP(float, ix.q) + (size_t)row * hd;
        w = DP(const float, ix.qw);
        tok = row / DI(rx.qh);
    } else if (row < qr + kr) {
        x = DP(float, ix.k) + (size_t)(row - qr) * hd;
        w = DP(const float, ix.kw);
        tok = (row - qr) / DI(rx.kh);
    } else {
        x = DP(float, ix.v) + (size_t)(row - qr - kr) * hd;
        w = NULL;
    }
    float v[NK];
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        v[k] = x[threadIdx.x + 128 * k];
        ss += v[k] * v[k];
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)hd + df(r, e, ix.eps));
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        int i = threadIdx.x + 128 * k;
        v[k] = w ? v[k] * s * w[i] : v[k] * s;
    }
    if (tok >= 0) {
        const float *c = DP(const float, rx.cos) + (size_t)tok * hd;
        const float *sn = DP(const float, rx.sin) + (size_t)tok * hd;
        #pragma unroll
        for (int k = 0; k < NK / 2; ++k) {
            int i = threadIdx.x + 128 * k;
            float a = v[k], b = v[k + NK / 2];
            v[k] = a * c[i] - b * sn[i];
            v[k + NK / 2] = b * c[i] + a * sn[i];
        }
    }
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        x[threadIdx.x + 128 * k] = v[k];
    }
}

/* The row stride of a float32 cache from its head stride hs: position-major
 * (positions, kv heads, hd) has hs == hd and rows of kvh hd values; the
 * head-major form (kv heads, positions, hd) has rows of hd. All the caches
 * are position-major now; the kernels take both. */
__host__ __device__ __forceinline__ size_t kv_rs(size_t hs, int kvh, int hd)
{
    return hs == (size_t)hd ? (size_t)kvh * hd : (size_t)hd;
}

/* k, v, kbuf, vbuf, head_stride, pos, tokens, kv_heads, head_dim. One block
 * for each head of each token. */
__global__ void k_kv_write_heads(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int kvh = DI(7), hd = DI(8);
    int j = blockIdx.x / kvh, h = blockIdx.x % kvh;
    size_t src = ((size_t)j * kvh + h) * hd;
    size_t hs = (size_t)di(r, e, 4);
    size_t dst = (size_t)h * hs + (size_t)(di(r, e, 5) + j) * kv_rs(hs, kvh, hd);
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
#define ATTN_KEYS 4          /* ATTN_KEYS * ATTN_REP = 32, the lanes of a warp */

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
    const int8_t *kq8, *vq8;     /* the int8 keys (kf8) and values (vf8) */
    const float *ks, *vs;        /* its scale for each group of 32 values */
    float *sc, *out;
    int qh, kvh, hd, n, window, i16;  /* i16: a quantized cache */
    int kf8, vf8;                /* the keys, the values in int8 */
    int64_t kp0, p;              /* the position of key row 0, and of the query */
    size_t hstride, rstride;     /* the distance of two heads and of two rows */
    const int32_t *rows;         /* GP_ATTN_QSA: the rows of the keys, or null */
    const uint8_t *kt, *vt;      /* tq: the TQ6 cache (tq6x8), q and out rotated */
    int tq;
    const float *cb;             /* tq: the codebook (in shared memory, or tq6_cb_d) */
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
 * the window of the query.
 *
 * GP_ATTN_QSA (Qwen3.8, csrc/qsa.c) reads the int16 cache on the rows that
 * QSA_SELECT gave. On the GPU, a record has one query (t = 1):
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, 1, pos,
 *     sel, cnt, maxsel
 *
 * cnt[0] is the count of the rows in sel, or -1 for the pos + 1 rows. */
__device__ attn_d attn_get(const gp_rec *r, const int64_t *e)
{
    attn_d a;
    memset(&a, 0, sizeof(a));
    a.q = DP(const float, 0);
    if (r->op == GP_ATTN_QC || r->op == GP_ATTN_QSA || r->op == GP_ATTN_Q8 ||
        r->op == GP_ATTN_V8 || r->op == GP_ATTN_TQ) {
        a.i16 = 1;
        a.tq = r->op == GP_ATTN_TQ || (r->op == GP_ATTN_QSA && DI(15) == 4);
        a.cb = tq6_cb_d;
        a.kt = DP(const uint8_t, 1);
        a.vt = DP(const uint8_t, 3);
        a.kf8 = r->op == GP_ATTN_Q8;
        a.vf8 = r->op == GP_ATTN_Q8 || r->op == GP_ATTN_V8;
        if (r->op == GP_ATTN_QSA) {
            /* operand 15 (or none): 0 int16, 1 int8, 2 int16 keys and int8
             * values (3 float32 and 4 TQ6 below) */
            int form = DI(15);
            a.kf8 = form == 1;
            a.vf8 = form >= 1;
        }
        a.kq = DP(const int16_t, 1);
        a.kq8 = DP(const int8_t, 1);
        a.vq8 = DP(const int8_t, 3);
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
        if (r->op == GP_ATTN_QSA) {
            int c = DP(const int32_t, 13)[0];
            a.p = di(r, e, 11);
            a.n = c < 0 ? (int)(a.p + 1) : c;
            a.rows = c < 0 ? NULL : DP(const int32_t, 12);
        }
        a.hstride = (size_t)a.hd;
        a.rstride = (size_t)a.kvh * a.hd;
        if (r->op == GP_ATTN_QSA && DI(15) == 3) {
            /* float32 rows: head stride operand 16 (kv_rs) */
            a.i16 = 0;
            a.kf8 = a.vf8 = 0;
            a.k = DP(const float, 1);
            a.v = DP(const float, 3);
            a.hstride = (size_t)di(r, e, 16);
            a.rstride = kv_rs(a.hstride, a.kvh, a.hd);
        }
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
        a.rstride = kv_rs(a.hstride, a.kvh, a.hd);
        a.k = DP(const float, 1) + (size_t)lo * a.rstride;
        a.v = DP(const float, 2) + (size_t)lo * a.rstride;
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
    size_t o = (size_t)(a.rows ? a.rows[j] : j) * a.rstride + (size_t)kv * a.hstride + i;
    if (a.tq) {
        tq6x8(key ? a.kt : a.vt, o, out, a.cb);
        float sc = (key ? a.ks : a.vs)[o / 32];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            out[t] *= sc;
        }
    } else if (key ? a.kf8 : a.vf8) {
        const int8_t *q = key ? a.kq8 : a.vq8;
        float sc = (key ? a.ks : a.vs)[o / 32];
        uint2 u = *(const uint2 *)(q + o);
        uint32_t w[2] = {u.x, u.y};
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            out[t] = (float)(int8_t)((w[t / 4] >> (8 * (t % 4))) & 0xff) * sc;
        }
    } else if (a.i16) {
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

/* The block has 256 threads. head_dim must be 256 or 512. The query heads
 * of a key head come in hg groups of at most ATTN_REP (12 heads of Qwen3.8:
 * 2 groups of 6); block x takes group x % hg of key head x / hg. qb is the
 * first query head of the block. */
__global__ void k_attn_part(const gp_rec *r, const int64_t *e, float *part, int hg)
{
    PDL_START();
    __shared__ float qs[ATTN_REP * 512];
    __shared__ float ps[ATTN_REP * ATTN_TILE];
    __shared__ float red[ATTN_REP * 512];
    __shared__ float cbs[64];
    attn_d a = attn_get(r, e);
    if (a.tq) {
        if (threadIdx.x < 64) {
            cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
        }
        a.cb = cbs;
    }
    int kv = blockIdx.x / hg, c = blockIdx.y;
    int n = a.n, hd = a.hd, rep = a.qh / a.kvh / hg;
    int qb = kv * (a.qh / a.kvh) + (blockIdx.x % hg) * rep;
    int len = attn_len(n);
    int j0 = c * len, j1 = min(n, j0 + len);
    if (j0 >= n) {
        return;
    }
    for (int i = threadIdx.x; i < rep * hd; i += blockDim.x) {
        qs[i] = a.q[(size_t)qb * hd + i];
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
        /* The 32 sums (ATTN_KEYS keys by ATTN_REP heads) over the lanes, as a
         * reduce-scatter: at each step a lane keeps half of its values and
         * sends the other half, so 31 shuffles leave value kk * ATTN_REP + h
         * on lane kk * ATTN_REP + h (a full sum for each value took 160). */
        float v[ATTN_KEYS * ATTN_REP];
        #pragma unroll
        for (int kk = 0; kk < ATTN_KEYS; ++kk) {
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                v[kk * ATTN_REP + h] = s[kk][h];
            }
        }
        #pragma unroll
        for (int off = 16, nv = 32; off > 0; off >>= 1, nv >>= 1) {
            bool up = (lane & off) != 0;
            #pragma unroll
            for (int x = 0; x < nv / 2; ++x) {
                float keep = up ? v[x + nv / 2] : v[x];
                float send = up ? v[x] : v[x + nv / 2];
                v[x] = keep + __shfl_xor_sync(0xffffffff, send, off);
            }
        }
        {
            int kk = lane / ATTN_REP, h = lane % ATTN_REP, j = jb + kk;
            if (h < rep && j < j1) {
                int64_t kp = a.rows ? a.rows[j] : a.kp0 + j;
                bool masked = kp > a.p || (a.window > 0 && a.p - kp >= a.window);
                a.sc[(size_t)(qb + h) * n + j] = masked ? -INFINITY : v[0];
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
        float *sc = a.sc + (size_t)(qb + h) * n;
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
            ps[h * ATTN_TILE + jj] = a.sc[(size_t)(qb + h) * n + t0 + jj];
        }
        __syncthreads();
        #pragma unroll 8
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
        part[((size_t)(qb + h) * ATTN_CHUNKS + c) * (hd + 2) + i] = red[x];
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int h = 0; h < ATTN_REP; ++h) {
            if (h < rep) {
                float *o = part + ((size_t)(qb + h) * ATTN_CHUNKS + c) * (hd + 2);
                o[hd] = m[h];
                o[hd + 1] = l[h];
            }
        }
    }
}

__device__ __forceinline__ int ft_len(int nl, int nb);

/* Block (h, x) joins the values x * 128 to x * 128 + 127 of query head h.
 * The weight exp(m_c - M) of each chunk goes to shared memory first. nb 0:
 * the chunks of attn_len; else the nb chunks of k_attn_fdtc (ft_len) of the
 * gridDim.x / qh queries of the record, head h of query h / qh (its n +
 * h / qh keys, its chunks), chunk stride nb. */
__global__ void k_attn_join(const gp_rec *r, const int64_t *e, const float *part, int nb)
{
    PDL_START();
    __shared__ float w[ATTN_CHUNKS];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x, hd = a.hd;
    int n = a.n, stride = ATTN_CHUNKS, len;
    if (nb > 0) {
        stride = nb;
        n = a.n + h / a.qh;
        len = ft_len(n, nb);
    } else {
        len = attn_len(a.n);
    }
    int nc = (n + len - 1) / len;
    const float *ph = part + (size_t)h * stride * (hd + 2);
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


/* ---------- the decode attention of one query over the int16 cache, in one
 * pass (flash decoding) ----------
 * For GP_ATTN_QC with head_dim 256 and 8 query heads for each key head
 * (Qwen3.5). Each warp takes FD_KEYS keys at a time of its own range of
 * keys: lane l keeps values 8 l to 8 l + 7 of each head. It loads the keys
 * and the values of the 4 rows together, computes the 32 scores (4 keys by 8
 * heads) with a reduce-scatter, and keeps the softmax running (the maximum m
 * and the sum l of each head). The sums of the values then take the new
 * weights at once. Each warp writes its part (FD_PARTS parts for each head);
 * k_attn_fd_join adds the parts. The keys and the values stream without the
 * phases of k_attn_part, and no scores go to memory. */
#define FD_KEYS 4
#define FD_BLOCKS 32                       /* blocks for each key head (32 was best of 32 to 256) */
#define FD_PARTS (FD_BLOCKS * 4)           /* 4 warps for each block */

__device__ __forceinline__ int fd_len(int n)
{
    int len = (n + FD_PARTS - 1) / FD_PARTS;
    len = len < 16 ? 16 : len;
    return (len + FD_KEYS - 1) / FD_KEYS * FD_KEYS;
}

__device__ __forceinline__ void fd_i8x8(uint2 u, float *out)
{
    uint32_t w[2] = {u.x, u.y};
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        out[t] = (float)(int8_t)((w[t / 4] >> (8 * (t % 4))) & 0xff);
    }
}

__device__ __forceinline__ void fd_i16x8(uint4 u, float *out)
{
    uint32_t w[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        out[2 * t] = (float)(int16_t)(w[t] & 0xffff);
        out[2 * t + 1] = (float)(int16_t)(w[t] >> 16);
    }
}

__global__ void __launch_bounds__(128) k_attn_fd(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    __shared__ __align__(16) float qs[8 * 256];
    __shared__ float pw[4][40];
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int n = a.n, len = fd_len(n);
    int part_id = blockIdx.y * 4 + warp;
    int j0 = part_id * len, j1 = min(n, j0 + len);
    for (int i = threadIdx.x; i < 8 * 256; i += blockDim.x) {
        qs[i] = a.q[(size_t)kv * 8 * 256 + i];
    }
    __syncthreads();
    if (j0 >= n) {
        return;
    }
    float acc[8][8];
    #pragma unroll
    for (int h = 0; h < 8; ++h) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[h][u] = 0.f;
        }
    }
    float m = -INFINITY, l = 0.f;          /* of head lane % 8 */
    size_t rs = a.rstride, off = (size_t)kv * 256 + 8 * lane;
    for (int jb = j0; jb < j1; jb += FD_KEYS) {
        uint4 kq[FD_KEYS], vq[FD_KEYS];
        float ks[FD_KEYS], vs[FD_KEYS];
        #pragma unroll
        for (int kk = 0; kk < FD_KEYS; ++kk) {
            int j = min(jb + kk, j1 - 1);
            size_t o = (size_t)j * rs + off;
            kq[kk] = *(const uint4 *)(a.kq + o);
            vq[kk] = *(const uint4 *)(a.vq + o);
            ks[kk] = a.ks[o / 32];
            vs[kk] = a.vs[o / 32];
        }
        float v[32];
        #pragma unroll
        for (int kk = 0; kk < FD_KEYS; ++kk) {
            float kf[8];
            fd_i16x8(kq[kk], kf);
            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                const float4 q0 = *(const float4 *)(qs + h * 256 + 8 * lane);
                const float4 q1 = *(const float4 *)(qs + h * 256 + 8 * lane + 4);
                float d = q0.x * kf[0] + q0.y * kf[1] + q0.z * kf[2] + q0.w * kf[3] +
                          q1.x * kf[4] + q1.y * kf[5] + q1.z * kf[6] + q1.w * kf[7];
                v[kk * 8 + h] = d * ks[kk];
            }
        }
        /* the reduce-scatter of k_attn_part: value kk * 8 + h to lane kk * 8 + h */
        #pragma unroll
        for (int o2 = 16, nv = 32; o2 > 0; o2 >>= 1, nv >>= 1) {
            bool up = (lane & o2) != 0;
            #pragma unroll
            for (int x = 0; x < nv / 2; ++x) {
                float keep = up ? v[x + nv / 2] : v[x];
                float send = up ? v[x] : v[x + nv / 2];
                v[x] = keep + __shfl_xor_sync(0xffffffff, send, o2);
            }
        }
        float sc = jb + lane / 8 < j1 ? v[0] : -INFINITY;
        /* the maximum and the sum of each head over the 4 keys: the lanes
         * h, h + 8, h + 16, h + 24 */
        float mx = fmaxf(sc, __shfl_xor_sync(0xffffffff, sc, 8));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 16));
        float mn = fmaxf(m, mx);
        float alpha = m == -INFINITY ? 0.f : expf(m - mn);
        float p = sc == -INFINITY ? 0.f : expf(sc - mn);
        float ps = p + __shfl_xor_sync(0xffffffff, p, 8);
        ps += __shfl_xor_sync(0xffffffff, ps, 16);
        l = l * alpha + ps;
        m = mn;
        pw[warp][lane] = p;
        if (lane < 8) {
            pw[warp][32 + lane] = alpha;
        }
        __syncwarp();
        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            float al = pw[warp][32 + h];
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[h][u] *= al;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < FD_KEYS; ++kk) {
            float vf[8];
            fd_i16x8(vq[kk], vf);
            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                float pp = pw[warp][kk * 8 + h] * vs[kk];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc[h][u] += pp * vf[u];
                }
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (int h = 0; h < 8; ++h) {
        float *o = part + ((size_t)(kv * 8 + h) * FD_PARTS + part_id) * 258;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            o[8 * lane + u] = acc[h][u];
        }
        if (lane == h) {
            o[256] = m;
            o[257] = l;
        }
    }
}

/* Block (h, x) joins the values x * 128 to x * 128 + 127 of query head h
 * from the parts of k_attn_fd (as k_attn_join). */
__global__ void k_attn_fd_join(const gp_rec *r, const int64_t *e, const float *part)
{
    PDL_START();
    __shared__ float w[FD_PARTS];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x;
    int len = fd_len(a.n);
    int nc = (a.n + len - 1) / len;
    const float *ph = part + (size_t)h * FD_PARTS * 258;
    float M = -INFINITY;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        M = fmaxf(M, ph[(size_t)c * 258 + 256]);
    }
    M = block_max(M);
    float wl = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float mc = ph[(size_t)c * 258 + 256];
        float wc = mc == -INFINITY ? 0.f : expf(mc - M);
        w[c] = wc;
        wl += wc * ph[(size_t)c * 258 + 257];
    }
    float wsum = block_sum(wl);
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    __syncthreads();
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < 256) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) {
            acc += w[c] * ph[(size_t)c * 258 + i];
        }
        a.out[(size_t)h * 256 + i] = acc * inv;
    }
}


/* k_attn_fd for other shapes: head_dim HD (256 or 512) and G query heads
 * for each key head (G divides 32; the Gemma 4 26B has HD 256 with G 2 and
 * HD 512 with G 8). A step of a warp takes 32 / G keys, so the scores are
 * 32 values (keys by heads) again. Lane l keeps values 8 l to 8 l + 7 of
 * each part of 256 values. The keys come first (the scores and the running
 * softmax), then the values, so fewer registers hold loads at a time. The
 * record gives the keys of the window, so no key needs a mask. */
template <int HD, int G, bool KQ8 = false, bool VQ8 = false>
__global__ void __launch_bounds__(128) k_attn_fdt(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int NP = HD / 256, K = 32 / G;
    __shared__ __align__(16) float qs[G * HD];
    __shared__ float pw[4][32 + G];
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int n = a.n, len = fd_len(n);
    len = (len + K - 1) / K * K;
    int part_id = blockIdx.y * 4 + warp;
    int j0 = part_id * len, j1 = min(n, j0 + len);
    for (int i = threadIdx.x; i < G * HD; i += blockDim.x) {
        qs[i] = a.q[(size_t)kv * G * HD + i];
    }
    __syncthreads();
    if (j0 >= n) {
        return;
    }
    float acc[G][NP][8];
    #pragma unroll
    for (int h = 0; h < G; ++h) {
        #pragma unroll
        for (int s = 0; s < NP; ++s) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[h][s][u] = 0.f;
            }
        }
    }
    float m = -INFINITY, l = 0.f;          /* of head lane % G */
    size_t rs = a.rstride, off = (size_t)kv * HD + 8 * lane;
    for (int jb = j0; jb < j1; jb += K) {
        float v[32];
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            int j = min(jb + kk, j1 - 1);
            #pragma unroll
            for (int h = 0; h < G; ++h) {
                v[kk * G + h] = 0.f;
            }
            #pragma unroll
            for (int s = 0; s < NP; ++s) {
                size_t o = (size_t)j * rs + off + 256 * s;
                float kf[8];
                if (KQ8) {
                    fd_i8x8(*(const uint2 *)(a.kq8 + o), kf);
                } else {
                    fd_i16x8(*(const uint4 *)(a.kq + o), kf);
                }
                float sk = a.ks[o / 32];
                #pragma unroll
                for (int h = 0; h < G; ++h) {
                    const float4 q0 = *(const float4 *)(qs + h * HD + 256 * s + 8 * lane);
                    const float4 q1 = *(const float4 *)(qs + h * HD + 256 * s + 8 * lane + 4);
                    float d = q0.x * kf[0] + q0.y * kf[1] + q0.z * kf[2] + q0.w * kf[3] +
                              q1.x * kf[4] + q1.y * kf[5] + q1.z * kf[6] + q1.w * kf[7];
                    v[kk * G + h] += d * sk;
                }
            }
        }
        /* the reduce-scatter: value kk * G + h to lane kk * G + h */
        #pragma unroll
        for (int o2 = 16, nv = 32; o2 > 0; o2 >>= 1, nv >>= 1) {
            bool up = (lane & o2) != 0;
            #pragma unroll
            for (int x = 0; x < nv / 2; ++x) {
                float keep = up ? v[x + nv / 2] : v[x];
                float send = up ? v[x] : v[x + nv / 2];
                v[x] = keep + __shfl_xor_sync(0xffffffff, send, o2);
            }
        }
        float sc = jb + lane / G < j1 ? v[0] : -INFINITY;
        /* the maximum and the sum of each head over the K keys: the lanes
         * h, h + G, h + 2 G, ... */
        float mx = sc;
        #pragma unroll
        for (int o2 = G; o2 < 32; o2 <<= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, o2));
        }
        float mn = fmaxf(m, mx);
        float alpha = m == -INFINITY ? 0.f : expf(m - mn);
        float p = sc == -INFINITY ? 0.f : expf(sc - mn);
        float ps = p;
        #pragma unroll
        for (int o2 = G; o2 < 32; o2 <<= 1) {
            ps += __shfl_xor_sync(0xffffffff, ps, o2);
        }
        l = l * alpha + ps;
        m = mn;
        pw[warp][lane] = p;
        if (lane < G) {
            pw[warp][32 + lane] = alpha;
        }
        __syncwarp();
        #pragma unroll
        for (int h = 0; h < G; ++h) {
            float al = pw[warp][32 + h];
            #pragma unroll
            for (int s = 0; s < NP; ++s) {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc[h][s][u] *= al;
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            int j = min(jb + kk, j1 - 1);
            #pragma unroll
            for (int s = 0; s < NP; ++s) {
                size_t o = (size_t)j * rs + off + 256 * s;
                float vf[8];
                if (VQ8) {
                    fd_i8x8(*(const uint2 *)(a.vq8 + o), vf);
                } else {
                    fd_i16x8(*(const uint4 *)(a.vq + o), vf);
                }
                float sv = a.vs[o / 32];
                #pragma unroll
                for (int h = 0; h < G; ++h) {
                    float pp = pw[warp][kk * G + h] * sv;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        acc[h][s][u] += pp * vf[u];
                    }
                }
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (int h = 0; h < G; ++h) {
        float *o = part + ((size_t)(kv * G + h) * FD_PARTS + part_id) * (HD + 2);
        #pragma unroll
        for (int s = 0; s < NP; ++s) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                o[256 * s + 8 * lane + u] = acc[h][s][u];
            }
        }
        if (lane == h) {
            o[HD] = m;
            o[HD + 1] = l;
        }
    }
}

/* k_attn_fd_join for the parts of k_attn_fdt: block (h, x) joins the values
 * x * 128 to x * 128 + 127 of query head h. */
template <int HD, int G>
__global__ void k_attn_fdt_join(const gp_rec *r, const int64_t *e, const float *part)
{
    PDL_START();
    __shared__ float w[FD_PARTS];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x;
    int len = fd_len(a.n);
    len = (len + 32 / G - 1) / (32 / G) * (32 / G);
    int nc = (a.n + len - 1) / len;
    const float *ph = part + (size_t)h * FD_PARTS * (HD + 2);
    float M = -INFINITY;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        M = fmaxf(M, ph[(size_t)c * (HD + 2) + HD]);
    }
    M = block_max(M);
    float wl = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float mc = ph[(size_t)c * (HD + 2) + HD];
        float wc = mc == -INFINITY ? 0.f : expf(mc - M);
        w[c] = wc;
        wl += wc * ph[(size_t)c * (HD + 2) + HD + 1];
    }
    float wsum = block_sum(wl);
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    __syncthreads();
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < HD) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) {
            acc += w[c] * ph[(size_t)c * (HD + 2) + i];
        }
        a.out[(size_t)h * HD + i] = acc * inv;
    }
}

/* GP_ATTN_QSA of a large group (t > 1): the operands of the CPU record
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim (256), t,
 *     pos, sel, cnt, maxsel
 *
 * Block (j, x): query j, key head x / hg, and query heads (x % hg) * R to
 * + R - 1 of that key head (R = q_heads / kv_heads / hg, at most 8). The 4
 * warps take every 4th key of the query (the rows of sel, or all the
 * positions to pos + j) with a softmax that runs: lane l keeps values 8 l
 * to 8 l + 7 of each head. Then the warps join in shared memory. */
#define AQ_W 4
template <int R>
__global__ void __launch_bounds__(128) k_attn_qsa_mt(const gp_rec *r, const int64_t *e, int hg)
{
    PDL_START();
    __shared__ __align__(16) float qs[R * 256];
    __shared__ float wm[AQ_W][R], wl[AQ_W][R];
    __shared__ float wacc[AQ_W][R][256];
    int nq = DI(7), nk = DI(8), maxsel = DI(14);
    int64_t pos = di(r, e, 11);
    int j = blockIdx.x, kv = blockIdx.y / hg, qb = kv * (nq / nk) + (blockIdx.y % hg) * R;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float *q = DP(const float, 0) + ((size_t)j * nq + qb) * 256;
    for (int i = threadIdx.x; i < R * 256; i += blockDim.x) {
        qs[i] = q[i];
    }
    __syncthreads();
    int c = DP(const int32_t, 13)[j];
    int n = c < 0 ? (int)(pos + j + 1) : c;
    const int32_t *rows = c < 0 ? NULL : DP(const int32_t, 12) + (size_t)j * maxsel;
    const int16_t *kq = DP(const int16_t, 1), *vq = DP(const int16_t, 3);
    const int8_t *kq8 = DP(const int8_t, 1), *vq8 = DP(const int8_t, 3);
    const float *ks = DP(const float, 2), *vs = DP(const float, 4);
    int form = DI(15);       /* 0 int16, 1 int8, 2 int16 keys and int8 values, 3 float32,
                              * 4 TQ6 */
    size_t rs = (size_t)nk * 256, off = (size_t)kv * 256 + 8 * lane;
    if (form == 3) {
        /* float32 rows: head stride operand 16 (kv_rs) */
        rs = kv_rs((size_t)di(r, e, 16), nk, 256);
        off = (size_t)kv * (size_t)di(r, e, 16) + 8 * lane;
    }
    float m[R], l[R], acc[R][8];
    #pragma unroll
    for (int h = 0; h < R; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[h][u] = 0.f;
        }
    }
    for (int x = warp; x < n; x += AQ_W) {
        size_t o = (size_t)(rows ? rows[x] : x) * rs + off;
        float kf[8], vf[8];
        float ksc = 1.f, vsc = 1.f;
        if (form == 3) {
            const float *kp = DP(const float, 1) + o, *vp = DP(const float, 3) + o;
            float4 a0 = *(const float4 *)kp, a1 = *(const float4 *)(kp + 4);
            float4 b0 = *(const float4 *)vp, b1 = *(const float4 *)(vp + 4);
            kf[0] = a0.x; kf[1] = a0.y; kf[2] = a0.z; kf[3] = a0.w;
            kf[4] = a1.x; kf[5] = a1.y; kf[6] = a1.z; kf[7] = a1.w;
            vf[0] = b0.x; vf[1] = b0.y; vf[2] = b0.z; vf[3] = b0.w;
            vf[4] = b1.x; vf[5] = b1.y; vf[6] = b1.z; vf[7] = b1.w;
        } else if (form == 4) {
            tq6x8(DP(const uint8_t, 1), o, kf);
            tq6x8(DP(const uint8_t, 3), o, vf);
            ksc = ks[o / 32];
            vsc = vs[o / 32];
        } else {
            if (form == 1) {
                fd_i8x8(*(const uint2 *)(kq8 + o), kf);
            } else {
                fd_i16x8(*(const uint4 *)(kq + o), kf);
            }
            if (form >= 1) {
                fd_i8x8(*(const uint2 *)(vq8 + o), vf);
            } else {
                fd_i16x8(*(const uint4 *)(vq + o), vf);
            }
            ksc = ks[o / 32];
            vsc = vs[o / 32];
        }
        #pragma unroll
        for (int h = 0; h < R; ++h) {
            const float4 q0 = *(const float4 *)(qs + h * 256 + 8 * lane);
            const float4 q1 = *(const float4 *)(qs + h * 256 + 8 * lane + 4);
            /* the scale of the key is that of the 32 values of the lane: before the sum */
            float d = (q0.x * kf[0] + q0.y * kf[1] + q0.z * kf[2] + q0.w * kf[3] +
                       q1.x * kf[4] + q1.y * kf[5] + q1.z * kf[6] + q1.w * kf[7]) * ksc;
            for (int o2 = 16; o2 > 0; o2 >>= 1) {
                d += __shfl_xor_sync(0xffffffff, d, o2);
            }
            float sc = d;
            float mn = fmaxf(m[h], sc);
            float alpha = expf(m[h] - mn), p = expf(sc - mn);
            l[h] = l[h] * alpha + p;
            m[h] = mn;
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[h][u] = acc[h][u] * alpha + p * vsc * vf[u];
            }
        }
    }
    #pragma unroll
    for (int h = 0; h < R; ++h) {
        if (lane == 0) {
            wm[warp][h] = m[h];
            wl[warp][h] = l[h];
        }
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            wacc[warp][h][8 * lane + u] = acc[h][u];
        }
    }
    __syncthreads();
    float *out = DP(float, 6) + ((size_t)j * nq + qb) * 256;
    for (int i = threadIdx.x; i < R * 256; i += blockDim.x) {
        int h = i / 256, d = i % 256;
        float M = -INFINITY;
        #pragma unroll
        for (int w = 0; w < AQ_W; ++w) {
            M = fmaxf(M, wm[w][h]);
        }
        float num = 0.f, den = 0.f;
        #pragma unroll
        for (int w = 0; w < AQ_W; ++w) {
            float f = wm[w][h] == -INFINITY ? 0.f : expf(wm[w][h] - M);
            num += f * wacc[w][h][d];
            den += f * wl[w][h];
        }
        out[i] = den > 0.f ? num / den : 0.f;
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
    PDL_START();
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
    PDL_START();
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

/* x (float32) to float16 with the clamps of an encoder linear (ENC_LINEAR). */
__global__ void k_enc_to_half(const float *x, __half *y, size_t n, float imin, float imax)
{
    PDL_START();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = __float2half_rn(fminf(fmaxf(x[i], imin), imax));
    }
}

/* The product of an encoder linear on the tensor cores: as k_gemm_bh, for
 * any cols that is a multiple of 8 (the 26B has 4304). The last step reads
 * zeros past cols. out = clamp(x W^T + b, omin, omax); b may be NULL. */
__global__ void __launch_bounds__(256) k_enc_gemm_tc(const __half *x, const uint16_t *w,
                                                     const float *bias, float *out, int t,
                                                     int rows, int cols, float omin, float omax)
{
    PDL_START();
    __shared__ __align__(16) __half as_[2][T2M][BHK + 8];
    __shared__ __align__(16) uint16_t bs[2][T2N][BHK + 8];
    int m0 = blockIdx.x * T2M, n0 = blockIdx.y * T2N;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    int steps = (cols + BHK - 1) / BHK;
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    auto load = [&](int st, int b) {
        int k0 = st * BHK;
        for (int e = threadIdx.x; e < T2M * BHK / 8; e += blockDim.x) {
            int m = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int q = min(m0 + m, t - 1);
            if (k0 + kk < cols) {
                cp_async16(&as_[b][m][kk], x + (size_t)q * cols + k0 + kk);
            } else {
                *(uint4 *)&as_[b][m][kk] = make_uint4(0, 0, 0, 0);
            }
        }
        for (int e = threadIdx.x; e < T2N * BHK / 8; e += blockDim.x) {
            int n = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int row = min(n0 + n, rows - 1);
            if (k0 + kk < cols) {
                cp_async16(&bs[b][n][kk], w + (size_t)row * cols + k0 + kk);
            } else {
                *(uint4 *)&bs[b][n][kk] = make_uint4(0, 0, 0, 0);
            }
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
            float b0 = (bias && n < rows) ? bias[n] : 0.f;
            float b1 = (bias && n + 1 < rows) ? bias[n + 1] : 0.f;
            if (m < t) {
                if (n < rows) out[(size_t)m * rows + n] = fminf(fmaxf(acc[i][j][0] + b0, omin), omax);
                if (n + 1 < rows) out[(size_t)m * rows + n + 1] = fminf(fmaxf(acc[i][j][1] + b1, omin), omax);
            }
            if (m + 8 < t) {
                if (n < rows) out[(size_t)(m + 8) * rows + n] = fminf(fmaxf(acc[i][j][2] + b0, omin), omax);
                if (n + 1 < rows) out[(size_t)(m + 8) * rows + n + 1] = fminf(fmaxf(acc[i][j][3] + b1, omin), omax);
            }
        }
    }
}


/* ---------- bfloat16 products of large groups on the tensor cores ----------
 * NP_GEMMA_DENSE=bf16 (Qwen3.8 with its dense matrices as they are): the
 * products of a prompt group ran in k_kq_gemm (float32, no tensor cores):
 * 4.6 s of 20 s for 8192 tokens. Here x goes to bfloat16 (k_to_bf16, round
 * to nearest even; bfloat16 has the range of float32, so no clamp), and
 * mma.sync m16n8k16 bf16 with float32 sums takes the bfloat16 rows of w as
 * they are. The tile is that of k_enc_gemm_tc: 128 tokens by 128 rows, 8
 * warps, steps of BHK columns in two buffers.
 *
 * out[(orow0 + m) * ldo + n] = sum_c x[(xrow0 + m) * ldx + c] w[n * cols + c]
 * for m < t, n < rows; xrow0 and orow0 (or null: 0) are read on the device
 * (the rows of the shared expert in the pairs of a group MoE). */
__device__ __forceinline__ void mma16816_bf16(float *c, const uint32_t *a, const uint32_t *b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* rows of n values from x (row stride ldx, from row xrow0 or its device
 * value) to bfloat16 rows of n values in y */
__device__ __forceinline__ uint16_t f2bf_rn(float v)
{
    uint32_t u = __float_as_uint(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* ylo (or null): the rest, x - hi, as bfloat16 too: hi + lo keeps about 16
 * bits of x (the X2 form of k_gemm_bf16_tc). */
__global__ void k_to_bf16(const float *x, size_t ldx, const int *xrow0, uint16_t *y, int rows, int n,
                          uint16_t *ylo = NULL)
{
    PDL_START();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * n) {
        return;
    }
    size_t m = i / n, c = i % n;
    size_t r0 = xrow0 != NULL ? (size_t)*xrow0 : 0;
    float v = x[(r0 + m) * ldx + c];
    uint16_t h = f2bf_rn(v);
    y[i] = h;
    if (ylo != NULL) {
        ylo[i] = f2bf_rn(v - __uint_as_float((uint32_t)h << 16));
    }
}

/* X2: x = hi + lo, two bfloat16 planes (xlo: lo), two products for each
 * fragment of w, in the same float32 sums: the bits of x of a float32
 * product but for about 16 of them (in place of 8). */
#define BF_SMEM(X2) ((size_t)2 * ((1 + (X2)) * T2M + T2N) * (BHK + 8) * 2)
/* W12: w holds KQ_BF12 rows (csrc/kquants.c; KQ_BF12 below): a step of BHK
 * = 32 columns is one group of each row. The step copies the bytes of the
 * group as they are (the sign and mantissa bytes, 32 a row; the gap nibbles,
 * 16; the word of the group exponent) into shared memory in place of bs, and
 * each thread makes the bfloat16 bits of its fragments of w from them
 * (bf12_frag): no copy of w in bfloat16, the bits of k_bf12_to_bf16. */
#define BF12_LO 48          /* bytes of a row of the lo bytes in shared memory (32 + 16: no bank conflict) */
/* Two values in the two 16-bit lanes of a word: the bytes b of lo and the
 * gaps of hi (>> sh) spread to the lanes, the exponent E - gap from E + 256
 * (no borrow out of a lane), and the zero code (gap 15, b 0x80) by
 * __vcmpeq2. */
__device__ __forceinline__ uint32_t bf12_frag(uint32_t lo, uint32_t hi, int sh, uint32_t E2)
{
    uint32_t b = __byte_perm(lo, 0, 0x4140);
    uint32_t gap = (__byte_perm(hi, 0, 0x4140) >> sh) & 0x000f000fu;
    uint32_t ex = (E2 - gap) & 0x00ff00ffu;
    uint32_t v = (b & 0x00800080u) << 8 | ex << 7 | (b & 0x007f007fu);
    return v & ~__vcmpeq2(b | gap << 8, 0x0f800f80u);
}

template <int X2, int W12 = 0>
__global__ void __launch_bounds__(256) k_gemm_bf16_tc(const uint16_t *x, const uint16_t *w, float *out,
                                                      size_t ldo, const int *orow0, int t, int rows,
                                                      int cols, const uint16_t *xlo)
{
    PDL_START();
    extern __shared__ __align__(16) uint16_t bfsm[];
    uint16_t (*as_)[T2M][BHK + 8] = (uint16_t (*)[T2M][BHK + 8])bfsm;
    uint16_t (*bs)[T2N][BHK + 8] = (uint16_t (*)[T2N][BHK + 8])(bfsm + 2 * T2M * (BHK + 8));
    uint16_t (*al_)[T2M][BHK + 8] = (uint16_t (*)[T2M][BHK + 8])(bfsm + 2 * (T2M + T2N) * (BHK + 8));
    /* W12: in the room of bs, for each of the two buffers: lo [T2N][BF12_LO],
     * hi [T2N][16], the exponent words [T2N] (2 x 8704 bytes of 2 x 10240) */
    const size_t w12_b = (size_t)T2N * (BF12_LO + 16 + 4);
    uint8_t *w12s = (uint8_t *)bs;
    const uint8_t *w8 = (const uint8_t *)w;
    const size_t rb12 = ((size_t)cols + cols / 2 + cols / 32 + 15) / 16 * 16;
    int m0 = blockIdx.x * T2M, n0 = blockIdx.y * T2N;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    int steps = (cols + BHK - 1) / BHK;
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    auto load = [&](int st, int b) {
        int k0 = st * BHK;
        for (int e = threadIdx.x; e < T2M * BHK / 8; e += blockDim.x) {
            int m = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int q = min(m0 + m, t - 1);
            if (k0 + kk < cols) {
                cp_async16(&as_[b][m][kk], x + (size_t)q * cols + k0 + kk);
                if (X2) {
                    cp_async16(&al_[b][m][kk], xlo + (size_t)q * cols + k0 + kk);
                }
            } else {
                *(uint4 *)&as_[b][m][kk] = make_uint4(0, 0, 0, 0);
                if (X2) {
                    *(uint4 *)&al_[b][m][kk] = make_uint4(0, 0, 0, 0);
                }
            }
        }
        if (W12) {
            /* the group st of each row: 2 x 16 bytes of lo, 16 of hi, the
             * aligned word of its exponent (cols % 32 == 0) */
            uint8_t *lo = w12s + b * w12_b, *hi = lo + T2N * BF12_LO, *ex = hi + T2N * 16;
            for (int e = threadIdx.x; e < T2N * 4; e += blockDim.x) {
                int n = e / 4, part = e % 4;
                const uint8_t *wr = w8 + (size_t)min(n0 + n, rows - 1) * rb12;
                if (part < 2) {
                    cp_async16(lo + n * BF12_LO + 16 * part, wr + k0 + 16 * part);
                } else if (part == 2) {
                    cp_async16(hi + n * 16, wr + cols + k0 / 2);
                } else {
                    cp_async4(ex + n * 4, wr + cols + cols / 2 + (st & ~3));
                }
            }
            cp_async_commit();
            return;
        }
        for (int e = threadIdx.x; e < T2N * BHK / 8; e += blockDim.x) {
            int n = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int row = min(n0 + n, rows - 1);
            if (k0 + kk < cols) {
                cp_async16(&bs[b][n][kk], w + (size_t)row * cols + k0 + kk);
            } else {
                *(uint4 *)&bs[b][n][kk] = make_uint4(0, 0, 0, 0);
            }
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
        /* W12: the fragments of w of both halves of the step (columns 2c,
         * 2c + 1, 2c + 8, 2c + 9 of each half: the gaps of the second half
         * are the high nibbles of the same hi bytes) */
        uint32_t bf12[2][4][2];
        if (W12) {
            const uint8_t *lo = w12s + b * w12_b, *hi = lo + T2N * BF12_LO, *ex = hi + T2N * 16;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int n = wn + j * 8 + g;
                const uint8_t *lr = lo + n * BF12_LO + 2 * c, *hr = hi + n * 16 + 2 * c;
                uint32_t E = (ex[n * 4 + (st & 3)] | 256u) * 0x00010001u;     /* E + 256 in each lane */
                uint32_t h0 = *(const uint16_t *)hr, h1 = *(const uint16_t *)(hr + 8);
                bf12[0][j][0] = bf12_frag(*(const uint16_t *)lr, h0, 0, E);
                bf12[0][j][1] = bf12_frag(*(const uint16_t *)(lr + 8), h1, 0, E);
                bf12[1][j][0] = bf12_frag(*(const uint16_t *)(lr + 16), h0, 4, E);
                bf12[1][j][1] = bf12_frag(*(const uint16_t *)(lr + 24), h1, 4, E);
            }
        }
        #pragma unroll
        for (int ks = 0; ks < BHK / 16; ++ks) {
            int kk = ks * 16;
            uint32_t bf[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (W12) {
                    bf[j][0] = bf12[ks][j][0];
                    bf[j][1] = bf12[ks][j][1];
                    continue;
                }
                const uint16_t *br = &bs[b][wn + j * 8 + g][kk + 2 * c];
                bf[j][0] = *(const uint32_t *)br;
                bf[j][1] = *(const uint32_t *)(br + 8);
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
                    mma16816_bf16(acc[i][j], a, bf[j]);
                }
                if (X2) {
                    a[0] = *(const uint32_t *)&al_[b][r0 + g][kk + 2 * c];
                    a[1] = *(const uint32_t *)&al_[b][r0 + g + 8][kk + 2 * c];
                    a[2] = *(const uint32_t *)&al_[b][r0 + g][kk + 2 * c + 8];
                    a[3] = *(const uint32_t *)&al_[b][r0 + g + 8][kk + 2 * c + 8];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        mma16816_bf16(acc[i][j], a, bf[j]);
                    }
                }
            }
        }
        __syncthreads();
    }
    size_t o0 = orow0 != NULL ? (size_t)*orow0 : 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = n0 + wn + j * 8 + 2 * c;
            int m = m0 + wm + i * 16 + g;
            if (m < t) {
                if (n < rows) out[(o0 + m) * ldo + n] = acc[i][j][0];
                if (n + 1 < rows) out[(o0 + m) * ldo + n + 1] = acc[i][j][1];
            }
            if (m + 8 < t) {
                if (n < rows) out[(o0 + m + 8) * ldo + n] = acc[i][j][2];
                if (n + 1 < rows) out[(o0 + m + 8) * ldo + n + 1] = acc[i][j][3];
            }
        }
    }
}

/* NP_GEMMA_GPU_BF16_TC: 1 (the default) x as two bfloat16 planes (about 16
 * bits); 8: one plane (8 bits of x, about 1.8 times as fast); 0: float32
 * products (k_kq_gemm). */
static int bf16_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_BF16_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return on;
}

/* x (t rows of cols float32 at x, stride ldx, from row xrow0 on the device or
 * 0) times the bfloat16 rows of w into out (stride ldo, from row orow0):
 * k_to_bf16 into the scratch xb (and xb + t cols for the lo plane), then
 * k_gemm_bf16_tc. */
static void gg_gemm_bf16(const float *x, size_t ldx, const int *xrow0, const uint16_t *w, float *out,
                         size_t ldo, const int *orow0, int t, int rows, int cols, uint16_t *xb,
                         cudaStream_t s, int w12 = 0)
{
    static int attr = 0;
    if (!attr) {
        cudaFuncSetAttribute(k_gemm_bf16_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        cudaFuncSetAttribute(k_gemm_bf16_tc<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        attr = 1;
    }
    int x2 = bf16_tc_on() != 8;
    uint16_t *xl = x2 ? xb + (size_t)t * cols : NULL;
    k_to_bf16<<<(unsigned)(((int64_t)t * cols + 255) / 256), 256, 0, s>>>(x, ldx, xrow0, xb, t, cols, xl);
    dim3 grid((unsigned)((t + T2M - 1) / T2M), (unsigned)((rows + T2N - 1) / T2N));
    if (x2 && w12) {
        k_gemm_bf16_tc<1, 1><<<grid, 256, BF_SMEM(1), s>>>(xb, w, out, ldo, orow0, t, rows, cols, xl);
    } else if (w12) {
        k_gemm_bf16_tc<0, 1><<<grid, 256, BF_SMEM(0), s>>>(xb, w, out, ldo, orow0, t, rows, cols, NULL);
    } else if (x2) {
        k_gemm_bf16_tc<1><<<grid, 256, BF_SMEM(1), s>>>(xb, w, out, ldo, orow0, t, rows, cols, xl);
    } else {
        k_gemm_bf16_tc<0><<<grid, 256, BF_SMEM(0), s>>>(xb, w, out, ldo, orow0, t, rows, cols, NULL);
    }
}

/* NP_GEMMA_GPU_BF12_FUSED: 1 (the default) the KQ_BF12 matrices of a large
 * group in k_gemm_bf16_tc<X2, 1> (the groups decoded into the fragments);
 * 0: chunks of bfloat16 rows in a scratch (gg_gemm_bf12), for a test. The
 * dense shapes of Qwen3.8 at 512 tokens (the 3090 at its 200 W cap): fused
 * 1.08-1.11 times the bfloat16 time, the scratch 1.07-1.35 (2.0 for 320 x
 * 10240); 2048 tokens: 1.10, 1.02-1.07. Without the decode (a test) the
 * fused kernel took 1.08: the cost is the three copies of a row of a step
 * (lo, hi, the exponent), not the decode. A decode of each tile once into
 * shared memory (an extra barrier a step) took 1.14. */
static int bf12_fused_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_BF12_FUSED");
        on = !(v && v[0] == '0');
    }
    return on;
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
    PDL_START();
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

/* x to the two int8 planes of the int16 form (NP_GEMMA_INT4_Q8=16 on the
 * GPU): q = round(x / s), s = max |x| / 16256 for each block of 32, and
 * q = 128 hi + lo with hi = round(q / 128) (|hi| <= 127, |lo| <= 64). A
 * product takes the int8 sums of hi and of lo on the tensor cores and
 * 128 sum(hi w) + sum(lo w) = sum(q w), exact in int32 (|q w| summed over a
 * block of 32 is at most 16256 * 8 * 32 < 2^22, so i2f_exact holds). The
 * values keep about 15 bits, as the int16 x of the CPU (32767 there).
 * xsum (or NULL): s times the sum of the 32 values of q. */
#define X2_QMAX 16256.0f
__global__ void k_quant_x2(const float *x, int8_t *qh, int8_t *ql, float *xs, float *xsum,
                           size_t blocks)
{
    PDL_START();
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t bi = tid / 8;
    int sub = threadIdx.x % 8;
    bool live = bi < blocks;
    float4 v = live ? *(const float4 *)(x + bi * 32 + sub * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    }
    float s2 = m / X2_QMAX;
    float vv[4] = {v.x, v.y, v.z, v.w};
    int q[4], h[4], sum = 0;
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        q[k] = s2 > 0.f ? __float2int_rn(vv[k] / s2) : 0;
        h[k] = __float2int_rn((float)q[k] * (1.0f / 128.0f));
        sum += q[k];
    }
    for (int o = 4; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    if (!live) {
        return;
    }
    *(char4 *)(qh + bi * 32 + sub * 4) = make_char4((signed char)h[0], (signed char)h[1],
                                                    (signed char)h[2], (signed char)h[3]);
    *(char4 *)(ql + bi * 32 + sub * 4) = make_char4(
        (signed char)(q[0] - 128 * h[0]), (signed char)(q[1] - 128 * h[1]),
        (signed char)(q[2] - 128 * h[2]), (signed char)(q[3] - 128 * h[3]));
    if (sub == 0) {
        xs[bi] = s2;
        if (xsum != NULL) {
            xsum[bi] = s2 * (float)sum;
        }
    }
}

/* The int32 sum of the int16 form (k_quant_x2): mma of hi, times 128, then
 * mma of lo into the same sums. */
__device__ __forceinline__ void mma16832_x2(int *ci, const uint32_t *ah, const uint32_t *al,
                                            const uint32_t *b);

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
/* X2 (the int16 form): the two planes of x in each row of a step, 2 buffers */
#define Q8SMEM_X2 ((size_t)2 * (T2M * (2 * Q8K + 16) + T2N * Q8RB + T2M * (Q8K / 32) * 4))

__device__ __forceinline__ void mma16832_x2(int *ci, const uint32_t *ah, const uint32_t *al,
                                            const uint32_t *b)
{
    mma16832(ci, ah, b);
    ci[0] *= 128;
    ci[1] *= 128;
    ci[2] *= 128;
    ci[3] *= 128;
    mma16832(ci, al, b);
}

__device__ __forceinline__ void cp_async8(void *smem, const void *gmem)
{
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(sa), "l"(gmem));
}

/* X2: the int16 form (k_quant_x2): xq the hi plane, xl the lo plane, a row
 * of a step holds the hi values and then the lo values; 2 buffers. */
template <int AL, int X2 = 0>
__global__ void __launch_bounds__(256) k_gemm_q8(const int8_t *xq, const float *xs,
                                                 const uint8_t *w, float *out, int t, int rows,
                                                 int cols, const int8_t *xl = NULL)
{
    PDL_START();
    /* NS buffers in dynamic shared memory: the copies of NS - 1 steps
     * run during the compute of a step. */
    constexpr int NS = X2 ? 2 : Q8NS, AW = Q8K * (1 + X2) + 16;
    extern __shared__ __align__(16) uint8_t q8sm[];
    int8_t (*as_)[T2M][AW] = (int8_t (*)[T2M][AW])q8sm;
    uint8_t (*bs)[T2N][Q8RB] = (uint8_t (*)[T2N][Q8RB])(q8sm + NS * T2M * AW);
    float (*ss)[T2M][Q8K / 32] = (float (*)[T2M][Q8K / 32])(q8sm + NS * (T2M * AW +
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
            if (X2) {
                cp_async16(&as_[b][m][Q8K + kk], xl + (size_t)q * cols + k0 + kk);
            }
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
    for (int st = 0; st < NS - 1; ++st) {
        if (st < steps) {
            load(st, st);
        } else {
            cp_async_commit();
        }
    }
    for (int st = 0; st < steps; ++st) {
        int b = st % NS;
        asm volatile("cp.async.wait_group %0;\n" :: "n"(NS - 2));
        __syncthreads();
        /* Buffer (st + NS - 1) % NS held step st - 1, which every warp
         * has passed. */
        if (st + NS - 1 < steps) {
            load(st + NS - 1, (st + NS - 1) % NS);
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
                uint32_t a[4], al[4];
                a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                if (X2) {
                    al[0] = *(const uint32_t *)&as_[b][r0 + g][Q8K + kk + 4 * c];
                    al[1] = *(const uint32_t *)&as_[b][r0 + g + 8][Q8K + kk + 4 * c];
                    al[2] = *(const uint32_t *)&as_[b][r0 + g][Q8K + kk + 4 * c + 16];
                    al[3] = *(const uint32_t *)&as_[b][r0 + g + 8][Q8K + kk + 4 * c + 16];
                }
                float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int ci[4] = {0, 0, 0, 0};
                    if (X2) {
                        mma16832_x2(ci, a, al, bf[j]);
                    } else {
                        mma16832(ci, a, bf[j]);
                    }
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
    PDL_START();
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
    PDL_START();
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

/* A small group of NT tokens (an MTP verify group) on an int4 matrix, with
 * the bits of int4_row (a decode step) for each token: the same lanes, and
 * the same order of the terms (int4_part, then the scale of the block).
 * k_mt_gemv_n multiplied each weight by the scale first, so a token of a
 * group did not get the bits of a step. A warp for each row: more rows for
 * each warp reuse x, but they leave too few warps for the small matrices of
 * the E4B (the groups were not faster). */
template <int NT>
__global__ void k_mt_int4_rows(const float *x, const uint8_t *w, float *out, int rows, int cols)
{
    PDL_START();
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
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            const float *xb = x + (size_t)j * cols + b * 32;
            float4 xl = *(const float4 *)(xb + 4 * sub);
            float4 xh = *(const float4 *)(xb + 16 + 4 * sub);
            float acc = int4_part(q, xl, xh);
            sum[j] += d * acc;
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
    PDL_START();
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
    PDL_START();
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
                /* The terms of k_bf16_linear, so a token of a group gets
                 * the bits of a decode step (an MTP verify group). */
                const float *xi = x + (size_t)j * cols + i * 8;
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    sum[j] += wv[2 * k] * xi[2 * k] + wv[2 * k + 1] * xi[2 * k + 1];
                }
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
    PDL_START();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)DI(3) * DI(4)) {
        float v = DP(const float, 0)[i];
        DP(float, 2)[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f *
                                                   (v + 0.044715f * v * v * v)))
                          * DP(const float, 1)[i];
    }
}

/* The last key that query j of a GP_ATTN_QC_MT record sees: pos + lim[j]
 * when the record has the array lim (operand 16: the tokens of an image see
 * each other), else its own position pos + j. */
__device__ __forceinline__ int64_t qc_last(const int *lim, int64_t pos, int j)
{
    return pos + (lim ? lim[j] : j);
}

/* The largest last key (less pos) of the queries j0 to jl. */
__device__ __forceinline__ int qc_tile_last(const int *lim, int j0, int jl)
{
    int m = jl;
    if (lim) {
        for (int a = j0; a <= jl; ++a) {
            m = max(m, lim[a]);
        }
    }
    return m;
}

/* The attention of a group of queries over the int16 cache, as FlashAttention
 * does it. The record GP_ATTN_QC_MT:
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, t, pos,
 *     base, window, lo, n, lim
 *
 * lim (optional, int32, t values) gives the last key of each query less pos
 * (qc_last). Without it query j sees the keys up to its own position.
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
    PDL_START();
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
    const int *lim = DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t pe = live ? qc_last(lim, pos, j) : p;
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
            bool ok = kk < kn && kp <= pe && (window == 0 || p - kp < window);
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
    PDL_START();
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
    const int *lim = F32H ? NULL : DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t pe = j < t ? qc_last(lim, pos, j) : p;
    /* The rows of head kv: the float cache (head-major) or the int16 cache
     * (row-major, from the head offset). */
    const float *kf = NULL, *vf = NULL;
    const int16_t *kq16 = NULL, *vq16 = NULL;
    const float *ks16 = NULL, *vs16 = NULL;
    size_t rowq = (size_t)kvh * hd, frs = hd;
    if (F32H) {
        kf = DP(const float, 1) + (size_t)kv * di(r, e, 10);
        vf = DP(const float, 2) + (size_t)kv * di(r, e, 10);
        frs = kv_rs((size_t)di(r, e, 10), kvh, hd);
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
                        kv_v = kf[(size_t)(k0 + a) * frs + d0 + d];
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
            bool ok = kk < kn && j < t && kp <= pe && (window == 0 || p - kp < window);
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
                        kv_v = vf[(size_t)(k0 + a) * frs + d0 + d];
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
    PDL_START();
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
    size_t rowq = (size_t)kvh * hd, frs = hd;
    if (F32H) {
        kf = DP(const float, 1) + (size_t)kv * di(r, e, 10);
        vf = DP(const float, 2) + (size_t)kv * di(r, e, 10);
        frs = kv_rs((size_t)di(r, e, 10), kvh, hd);
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
    const int *lim = F32H ? NULL : DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t p0 = pos + j0 + qg * 16 + g, p1 = p0 + 8;   /* the positions of rows g, g + 8 */
    bool live0 = j0 + qg * 16 + g < t, live1 = j0 + qg * 16 + g + 8 < t;
    int64_t e0 = live0 ? qc_last(lim, pos, j0 + qg * 16 + g) : p0;
    int64_t e1 = live1 ? qc_last(lim, pos, j0 + qg * 16 + g + 8) : p1;
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
                    kvk = kf[(size_t)(k0 + a) * frs + d];
                    kvv = vf[(size_t)(k0 + a) * frs + d];
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
                bool ok0 = live0 && kk < kn && kp <= e0 && (window == 0 || p0 - kp < window);
                bool ok1 = live1 && kk < kn && kp <= e1 && (window == 0 || p1 - kp < window);
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

/* GP_ATTN_QSA of a large group on the tensor cores (int8 or TQ6 keys and
 * values). k_attn_qsa_mt runs the dot of each query head with each key in
 * float32 (1.3 s of a bf16 prompt of 8192 tokens), and each block decodes
 * the keys of its query again (two blocks for each key head). But the query
 * heads of a key head read the same keys: they are the 16 rows (12 used) of
 * an mma.m16n8k16 tile. Block (j, kv): query j, all its heads of key head kv;
 * 4 warps, each with its own keys (16 at a time from the rows of sel, or all
 * positions to pos + j), decoded to float16 times their scale in the shared
 * memory of the warp: S = Q K^T (Q2: q as two float16 planes, hi + lo, so q
 * keeps about 22 bits), a softmax that runs, then O += P V (P float16, V by
 * ldmatrix.trans). The warps join at the end in shared memory. */
__device__ __forceinline__ void ldsm_x4_trans(uint32_t *d, const void *p)
{
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
}

#define QT_LD 264                    /* the row stride (halves) of the tiles */
#define QT_SMEM ((size_t)(2 * 16 + 4 * 16) * QT_LD * 2)
template <int Q2>
__global__ void __launch_bounds__(128) k_attn_qsa_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) __half qtsm[];
    __half (*qh)[QT_LD] = (__half (*)[QT_LD])qtsm;
    __half (*ql)[QT_LD] = (__half (*)[QT_LD])(qtsm + 16 * QT_LD);
    __shared__ float wm[4][16], wl[4][16];
    __shared__ float cbs[64];
    if (threadIdx.x < 64) {
        cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
    }
    int nq = DI(7), nk = DI(8), maxsel = DI(14), form = DI(15);
    int64_t pos = di(r, e, 11);
    int rep = nq / nk;
    int j = blockIdx.x, kv = blockIdx.y;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    __half (*ks_)[QT_LD] = (__half (*)[QT_LD])(qtsm + (2 + warp) * 16 * QT_LD);
    const float *q = DP(const float, 0) + ((size_t)j * nq + (size_t)kv * rep) * 256;
    for (int i = threadIdx.x; i < 16 * 256; i += blockDim.x) {
        int h = i / 256, d = i % 256;
        float v = h < rep ? q[(size_t)h * 256 + d] : 0.f;
        __half hi = __float2half_rn(v);
        qh[h][d] = hi;
        ql[h][d] = __float2half_rn(v - __half2float(hi));
    }
    __syncthreads();
    int cn = DP(const int32_t, 13)[j];
    int n = cn < 0 ? (int)(pos + j + 1) : cn;
    const int32_t *rows = cn < 0 ? NULL : DP(const int32_t, 12) + (size_t)j * maxsel;
    const uint8_t *kb = DP(const uint8_t, 1), *vb = DP(const uint8_t, 3);
    const float *ksc = DP(const float, 2), *vsc = DP(const float, 4);
    size_t rs = (size_t)nk * 256, off = (size_t)kv * 256 + 8 * lane;
    /* rows 16 keys of the warp from x0 (keys past n: 0) to its tile */
    auto decode = [&](const uint8_t *base, const float *sc, int x0) {
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            int x = x0 + rr;
            float f[8];
            if (x < n) {
                size_t o = (size_t)(rows ? rows[x] : x) * rs + off;
                if (form == 4) {
                    tq6x8(base, o, f, cbs);
                } else {
                    fd_i8x8(*(const uint2 *)((const int8_t *)base + o), f);
                }
                float s = sc[o / 32];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    f[u] *= s;
                }
            } else {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    f[u] = 0.f;
                }
            }
            uint4 pk = make_uint4(pack_h2(f[0], f[1]), pack_h2(f[2], f[3]), pack_h2(f[4], f[5]),
                                  pack_h2(f[6], f[7]));
            *(uint4 *)&ks_[rr][8 * lane] = pk;
        }
        __syncwarp();
    };
    float acc[32][4];
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;   /* rows g, g + 8 */
    for (int x0 = warp * 16; x0 < n; x0 += 64) {
        decode(kb, ksc, x0);
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int kk = 0; kk < 256; kk += 16) {
            uint32_t a[4], b[2][2];
            #pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                b[jn][0] = *(const uint32_t *)&ks_[jn * 8 + g][kk + 2 * c];
                b[jn][1] = *(const uint32_t *)&ks_[jn * 8 + g][kk + 2 * c + 8];
            }
            a[0] = *(const uint32_t *)&qh[g][kk + 2 * c];
            a[1] = *(const uint32_t *)&qh[g + 8][kk + 2 * c];
            a[2] = *(const uint32_t *)&qh[g][kk + 2 * c + 8];
            a[3] = *(const uint32_t *)&qh[g + 8][kk + 2 * c + 8];
            mma16816(s[0], a, b[0]);
            mma16816(s[1], a, b[1]);
            if (Q2) {
                a[0] = *(const uint32_t *)&ql[g][kk + 2 * c];
                a[1] = *(const uint32_t *)&ql[g + 8][kk + 2 * c];
                a[2] = *(const uint32_t *)&ql[g][kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&ql[g + 8][kk + 2 * c + 8];
                mma16816(s[0], a, b[0]);
                mma16816(s[1], a, b[1]);
            }
        }
        __syncwarp();
        /* the keys past n: no weight */
        #pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            int x = x0 + jn * 8 + 2 * c;
            if (x >= n) { s[jn][0] = s[jn][2] = -INFINITY; }
            if (x + 1 >= n) { s[jn][1] = s[jn][3] = -INFINITY; }
        }
        float t0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
        float t1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
        t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 1));
        t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, 1));
        t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, 2));
        float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);     /* finite: key x0 < n */
        float a0 = expf(m0 - n0), a1 = expf(m1 - n1);
        m0 = n0;
        m1 = n1;
        float p[2][4];
        #pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            p[jn][0] = expf(s[jn][0] - n0);
            p[jn][1] = expf(s[jn][1] - n0);
            p[jn][2] = expf(s[jn][2] - n1);
            p[jn][3] = expf(s[jn][3] - n1);
        }
        l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
        l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            acc[nt][0] *= a0;
            acc[nt][1] *= a0;
            acc[nt][2] *= a1;
            acc[nt][3] *= a1;
        }
        decode(vb, vsc, x0);
        uint32_t pa[4] = {pack_h2(p[0][0], p[0][1]), pack_h2(p[0][2], p[0][3]),
                          pack_h2(p[1][0], p[1][1]), pack_h2(p[1][2], p[1][3])};
        /* B = V (16 keys by 256 values): ldmatrix.trans of the rows (lane % 8)
         * + 8 ((lane / 8) & 1), the values n0 + 8 (lane / 16) */
        #pragma unroll
        for (int nt = 0; nt < 32; nt += 2) {
            uint32_t bv[4];
            ldsm_x4_trans(bv, &ks_[(lane % 8) + 8 * ((lane / 8) & 1)][nt * 8 + 8 * (lane / 16)]);
            mma16816(acc[nt], pa, bv);
            mma16816(acc[nt + 1], pa, bv + 2);
        }
        __syncwarp();
    }
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    if (c == 0) {
        wm[warp][g] = m0;
        wm[warp][g + 8] = m1;
        wl[warp][g] = l0;
        wl[warp][g + 8] = l1;
    }
    __syncthreads();
    /* the join: the weight of each warp for rows g and g + 8, then the sums
     * of the warps in turn in red (the tiles are free now) */
    float M0 = -INFINITY, M1 = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        M0 = fmaxf(M0, wm[w][g]);
        M1 = fmaxf(M1, wm[w][g + 8]);
    }
    float L0 = 0.f, L1 = 0.f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        L0 += wm[w][g] == -INFINITY ? 0.f : expf(wm[w][g] - M0) * wl[w][g];
        L1 += wm[w][g + 8] == -INFINITY ? 0.f : expf(wm[w][g + 8] - M1) * wl[w][g + 8];
    }
    float f0 = m0 == -INFINITY ? 0.f : expf(m0 - M0) / L0;
    float f1 = m1 == -INFINITY ? 0.f : expf(m1 - M1) / L1;
    float (*red)[256] = (float (*)[256])qtsm;
    for (int w = 0; w < 4; ++w) {
        if (warp == w) {
            #pragma unroll
            for (int nt = 0; nt < 32; ++nt) {
                int d = nt * 8 + 2 * c;
                if (w == 0) {
                    red[g][d] = acc[nt][0] * f0;
                    red[g][d + 1] = acc[nt][1] * f0;
                    red[g + 8][d] = acc[nt][2] * f1;
                    red[g + 8][d + 1] = acc[nt][3] * f1;
                } else {
                    red[g][d] += acc[nt][0] * f0;
                    red[g][d + 1] += acc[nt][1] * f0;
                    red[g + 8][d] += acc[nt][2] * f1;
                    red[g + 8][d + 1] += acc[nt][3] * f1;
                }
            }
        }
        __syncthreads();
    }
    float *out = DP(float, 6) + ((size_t)j * nq + (size_t)kv * rep) * 256;
    for (int i = threadIdx.x; i < rep * 256; i += blockDim.x) {
        out[i] = red[i / 256][i % 256];
    }
}

/* The decode attention of a global layer of the Gemma 4 26B (head_dim 512, 8
 * query heads for each key head: GP_ATTN_QC, GP_ATTN_Q8, GP_ATTN_V8) on the
 * tensor cores, for T = 1 to FT_TMAX queries in one record (operand 11 t:
 * an MTP verify group; query j sees n + j keys), in chunks (k_attn_join
 * with nb adds them). k_attn_fdt took 1.76 ms a layer at 100K
 * positions (about 250 GB/s of the int16 rows, 47% of a step), and a
 * verify group of 3 ran it for each query. RTX 3090, 100K rows
 * (scripts/bench_attn_fdtc.cu): one query 0.68 ms (int16; int8 0.48,
 * k16v8 0.62), a group of 3 0.82 ms (0.65, 0.72). The 26B at 100K: plain
 * decode 56 -> 87 tok/s, MTP (2 drafts) 121 tok/s, the same tokens.
 *
 * Block (kv, c), 8 warps: chunk c of the keys of key head kv (ft_len), 64
 * keys at a time; the rows of the 8 T heads of the queries (query j: rows
 * 8 j to 8 j + 7) are the rows of the m16 tiles. The rows of the cache go
 * from memory to the registers of the fragments, with the values in a free
 * order (each load whole sectors):
 *   S = Q K^T: warp w the keys 8 w .. 8 w + 7 (n8) over the 512 values;
 *   lane (g, cc) reads values 32 b + 8 cc .. + 7 of key g, the k16 of two
 *   steps (q as two float16 planes in shared memory, in the same order);
 *   the scores to shared memory; each warp the softmax that runs, the same
 *   in all (rows 8 j + g);
 *   O^T += V^T P^T: warp w the values 64 w .. 64 w + 63; lane (g, cc) reads
 *   keys 2 cc, 2 cc + 1, 2 cc + 8, 2 cc + 9 of each k16 at values 64 w +
 *   8 g .. + 7 (8 lanes: 128 bytes of a row): value 64 w + 8 g + 2 t is row
 *   g of tile t, + 1 row g + 8; P^T the scores of the lane (n8: the 8 heads
 *   of query j); 4 tiles of 16 values for each query (16 T floats a lane).
 * The chunks of query j depend on n + j in steps of 4096 keys only (a
 * group across a step: a second pass for the queries past it), and a tile
 * past the keys of a query changes nothing in it (all its scores -inf), so
 * a query of a group gets the numbers of a step (T 1) at its position: MTP
 * verifies with the numbers of the decode. The part of each head as
 * k_attn_part writes it (the sum of exp(score - m) v, m, l), head j * qh +
 * h, chunk stride gridDim.y. */
#define FT_TMAX 3
#define FT_KEYS 64
#define FT_SLD (FT_KEYS + 4)
#define FT_SMEM(T) ((size_t)(T) * 8 * 512 * 2 * 2 + (size_t)2 * (T) * 8 * FT_SLD * 4)

/* The keys of a chunk of nb chunks for nl keys (those of the last query):
 * from nl rounded up to 4096 keys, a multiple of FT_KEYS. */
__device__ __forceinline__ int ft_len(int nl, int nb)
{
    int nr = (nl + 4095) & ~4095;
    int len = (nr + nb - 1) / nb;
    return max(FT_KEYS, (len + FT_KEYS - 1) / FT_KEYS * FT_KEYS);
}

/* 8 values of the cache (B 16: int16, 8: int8) at o, raw */
template <int B>
__device__ __forceinline__ uint4 ft_ld(const void *p, size_t o)
{
    if (B == 16) {
        return *(const uint4 *)((const int16_t *)p + o);
    }
    uint2 u = *(const uint2 *)((const int8_t *)p + o);
    return make_uint4(u.x, u.y, 0u, 0u);
}

/* the 8 values of ft_ld times sc as 4 float16 pairs */
template <int B>
__device__ __forceinline__ void ft_h8(uint4 u, float sc, uint32_t *h)
{
    if (B == 16) {
        uint32_t w[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            h[t] = pack_h2((float)(int16_t)(w[t] & 0xffff) * sc, (float)(int16_t)(w[t] >> 16) * sc);
        }
    } else {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            uint32_t w = (t < 2 ? u.x : u.y) >> (16 * (t % 2));
            h[t] = pack_h2((float)(int8_t)(w & 0xff) * sc, (float)(int8_t)((w >> 8) & 0xff) * sc);
        }
    }
}

/* KB, VB: the bits of the keys and of the values (GP_ATTN_QC 16, 16;
 * GP_ATTN_Q8 8, 8; GP_ATTN_V8 16, 8). The loads of a batch (8 a lane) go
 * out before their use (through attn_kv8, a branch on the form in each,
 * one waited for the other). q: 16 bytes of a row (8 values) at chunk
 * x ^ 4 (row & 1) of the row, for the shared memory banks. */
template <int T, int KB, int VB>
__global__ void __launch_bounds__(256) k_attn_fdtc(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int MT = (T + 1) / 2;                /* m16 tiles of the 8 T rows */
    extern __shared__ __align__(16) unsigned char ftsm[];
    uint4 (*qh)[64] = (uint4 (*)[64])ftsm;                         /* [8 T][64] */
    uint4 (*ql)[64] = (uint4 (*)[64])(ftsm + (size_t)T * 8 * 1024);
    float (*ss)[T * 8][FT_SLD] = (float (*)[T * 8][FT_SLD])(ftsm + (size_t)T * 8 * 2048);
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, c = blockIdx.y, nb = gridDim.y;
    int n = a.n;
    if (c * ft_len(n, nb) >= n + T - 1) {
        return;
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, cc = lane % 4;
    int qhn = a.qh;
    /* q: rows 8 j + h (query j, head kv 8 + h) */
    for (int i = threadIdx.x; i < T * 8 * 128; i += 256) {
        int row = i / 128, d = 4 * (i % 128);
        float4 v = *(const float4 *)(a.q + ((size_t)(row / 8) * qhn + kv * 8 + row % 8) * 512 + d);
        float f[4] = {v.x, v.y, v.z, v.w};
        uint32_t hi[2], lo[2];
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            __half h0 = __float2half_rn(f[2 * t]), h1 = __float2half_rn(f[2 * t + 1]);
            hi[t] = pack_h2(__half2float(h0), __half2float(h1));
            lo[t] = pack_h2(f[2 * t] - __half2float(h0), f[2 * t + 1] - __half2float(h1));
        }
        int ch = (d / 8) ^ (4 * (row & 1));
        uint2 *ph = (uint2 *)&qh[row][ch] + (d % 8) / 4, *pl = (uint2 *)&ql[row][ch] + (d % 8) / 4;
        *ph = make_uint2(hi[0], hi[1]);
        *pl = make_uint2(lo[0], lo[1]);
    }
    __syncthreads();
    const void *kb = a.kq, *vb = a.vq;              /* operands 1, 3 (int16 or int8) */
    /* query j: chunks of ft_len(n + j); a group across a step of 4096 keys
     * takes a second pass for the queries past it */
    int L0 = ft_len(n, nb), L1 = ft_len(n + T - 1, nb), buf = 0;
    for (int ps = 0; ps < (L1 != L0 ? 2 : 1); ++ps) {
        int len = ps ? L1 : L0, act = 0, jl = 0;
        #pragma unroll
        for (int j = 0; j < T; ++j) {
            if (ft_len(n + j, nb) == len) {
                act |= 1 << j;
                jl = j;
            }
        }
        int nl = n + jl, j0 = c * len, j1 = min(nl, j0 + len);
        if (j0 >= nl) {
            continue;
        }
        float acc[4][T][4];
        float m[T], l[T];
        #pragma unroll
        for (int j = 0; j < T; ++j) {
            m[j] = -INFINITY;
            l[j] = 0.f;
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.f;
            }
        }
        for (int x0 = j0; x0 < j1; x0 += FT_KEYS, buf ^= 1) {
            /* S of keys x0 + 8 warp .. + 7; the keys past j1 read key j1 - 1
             * (their scores: -inf) */
            size_t ok = (size_t)min(x0 + 8 * warp + g, j1 - 1) * a.rstride + (size_t)kv * a.hstride;
            float s[MT][4];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                s[mt][0] = s[mt][1] = s[mt][2] = s[mt][3] = 0.f;
            }
            #pragma unroll 1
            for (int bb = 0; bb < 16; bb += 8) {
                uint4 rk[8];
                float ck[8];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    size_t o = ok + 32 * (bb + u) + 8 * cc;
                    rk[u] = ft_ld<KB>(kb, o);
                    ck[u] = a.ks[o / 32];
                }
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    uint32_t k[4];
                    ft_h8<KB>(rk[u], ck[u], k);
                    int ch = 4 * (bb + u) + cc;
                    #pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        int r0 = 16 * mt + g, r1 = r0 + 8;
                        bool h1 = r1 < 8 * T;
                        #pragma unroll
                        for (int pl = 0; pl < 2; ++pl) {
                            uint4 (*qp)[64] = pl ? ql : qh;
                            uint4 q0 = qp[r0][ch ^ (4 * (r0 & 1))];
                            uint4 q1 = h1 ? qp[r1][ch ^ (4 * (r1 & 1))] : make_uint4(0u, 0u, 0u, 0u);
                            uint32_t av[4] = {q0.x, q1.x, q0.y, q1.y};
                            uint32_t b0[2] = {k[0], k[1]};
                            mma16816(s[mt], av, b0);
                            av[0] = q0.z; av[1] = q1.z; av[2] = q0.w; av[3] = q1.w;
                            b0[0] = k[2]; b0[1] = k[3];
                            mma16816(s[mt], av, b0);
                        }
                    }
                }
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                int r0 = 16 * mt + g, r1 = r0 + 8;
                *(float2 *)&ss[buf][r0][8 * warp + 2 * cc] = make_float2(s[mt][0], s[mt][1]);
                if (r1 < 8 * T) {
                    *(float2 *)&ss[buf][r1][8 * warp + 2 * cc] = make_float2(s[mt][2], s[mt][3]);
                }
            }
            /* the values of the first two k16 steps, while the scores join */
            /* value k: key 16 (k / 4) + 2 cc + (k & 1) + 8 ((k >> 1) & 1) */
            auto vo = [&](int k) {
                int x = x0 + 16 * (k / 4) + 2 * cc + (k & 1) + 8 * ((k >> 1) & 1);
                return (size_t)min(x, j1 - 1) * a.rstride + (size_t)kv * a.hstride + 64 * warp + 8 * g;
            };
            uint4 rv[8];
            float cv[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k) {
                size_t o = vo(k);
                rv[k] = ft_ld<VB>(vb, o);
                cv[k] = a.vs[o / 32];
            }
            __syncthreads();
            /* the softmax of row 8 j + g: the keys 16 ks + 2 cc + {0, 1, 8, 9} of
             * the lane; B = P^T of each k16 step (b0 keys 2 cc, 2 cc + 1; b1 + 8) */
            uint32_t pb[T][4][2];
            #pragma unroll
            for (int j = 0; j < T; ++j) {
                int lim = (act >> j) & 1 ? min(j1, n + j) - x0 : -FT_KEYS;  /* its keys here */
                float sv[16];
                #pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    #pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        int kk = 16 * ks + 2 * cc + 8 * hf;
                        float2 v = *(const float2 *)&ss[buf][8 * j + g][kk];
                        sv[4 * ks + 2 * hf] = kk < lim ? v.x : -INFINITY;
                        sv[4 * ks + 2 * hf + 1] = kk + 1 < lim ? v.y : -INFINITY;
                    }
                }
                float t0 = sv[0];
                #pragma unroll
                for (int u = 1; u < 16; ++u) {
                    t0 = fmaxf(t0, sv[u]);
                }
                t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 1));
                t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 2));
                float n0 = fmaxf(m[j], t0);
                /* no key of query j yet (a tile past its keys): nothing changes */
                float a0 = n0 == -INFINITY ? 1.f : expf(m[j] - n0);
                m[j] = n0;
                float ps = 0.f;
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    sv[u] = n0 == -INFINITY ? 0.f : expf(sv[u] - n0);
                    ps += sv[u];
                }
                l[j] = l[j] * a0 + ps;
                #pragma unroll
                for (int ks = 0; ks < 4; ++ks) {
                    pb[j][ks][0] = pack_h2(sv[4 * ks], sv[4 * ks + 1]);
                    pb[j][ks][1] = pack_h2(sv[4 * ks + 2], sv[4 * ks + 3]);
                }
                /* the factor of the heads 2 cc and 2 cc + 1 (the columns of the
                 * fragments of O^T) from the lanes of their rows */
                float al0 = __shfl_sync(0xffffffff, a0, 8 * cc), al1 = __shfl_sync(0xffffffff, a0, 8 * cc + 4);
                #pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    acc[mt][j][0] *= al0;
                    acc[mt][j][1] *= al1;
                    acc[mt][j][2] *= al0;
                    acc[mt][j][3] *= al1;
                }
            }
            #pragma unroll
            for (int half = 0; half < 2; ++half) {
                if (half == 1) {
                    #pragma unroll
                    for (int k = 0; k < 8; ++k) {
                        size_t o = vo(8 + k);
                        rv[k] = ft_ld<VB>(vb, o);
                        cv[k] = a.vs[o / 32];
                    }
                }
                #pragma unroll
                for (int kh = 0; kh < 2; ++kh) {
                    int ks = 2 * half + kh;
                    uint32_t h[4][4];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        ft_h8<VB>(rv[4 * kh + k], cv[4 * kh + k], h[k]);
                    }
                    /* rows g (values 2 t) and g + 8 (2 t + 1) of tile t: the
                     * pairs of keys 2 cc, 2 cc + 1 and 2 cc + 8, 2 cc + 9 */
                    #pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        uint32_t av[4] = {__byte_perm(h[0][mt], h[1][mt], 0x5410),
                                          __byte_perm(h[0][mt], h[1][mt], 0x7632),
                                          __byte_perm(h[2][mt], h[3][mt], 0x5410),
                                          __byte_perm(h[2][mt], h[3][mt], 0x7632)};
                        #pragma unroll
                        for (int j = 0; j < T; ++j) {
                            mma16816(acc[mt][j], av, pb[j][ks]);
                        }
                    }
                }
            }
        }
        /* the part of each head: values 64 warp + 8 g + 2 mt (+ 1) of heads 2 cc,
         * 2 cc + 1 of query j */
        #pragma unroll
        for (int j = 0; j < T; ++j) {
            if (!((act >> j) & 1)) {
                continue;
            }
            float lj = l[j];
            lj += __shfl_xor_sync(0xffffffff, lj, 1);
            lj += __shfl_xor_sync(0xffffffff, lj, 2);
            float *p0 = part + ((size_t)(j * qhn + kv * 8 + 2 * cc) * nb + c) * 514;
            float *p1 = part + ((size_t)(j * qhn + kv * 8 + 2 * cc + 1) * nb + c) * 514;
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                int d = 64 * warp + 8 * g + 2 * mt;
                *(float2 *)&p0[d] = make_float2(acc[mt][j][0], acc[mt][j][2]);
                *(float2 *)&p1[d] = make_float2(acc[mt][j][1], acc[mt][j][3]);
            }
            if (warp == 0 && cc == 0) {
                float *o = part + ((size_t)(j * qhn + kv * 8 + g) * nb + c) * 514;
                o[512] = m[j];
                o[513] = lj;
            }
        }
    }
}

/* The decode attention of a sliding layer of the Gemma 4 26B (head_dim 256,
 * 2 query heads for each key head, a window of W keys) on the tensor cores,
 * for T = 1 to FT_TMAX queries in one record: operands 10 n (the rows to
 * query 0, from row 0 of the buffer), 11 t, 12 W, 13 base (the position of
 * row 0); query j sees rows max(0, n + j - W) to n + j - 1. k_attn_fdt took 26 us a layer (a step at
 * any length: the window), and a verify group of 3 ran it for each query
 * (25 layers: 2 ms of a cycle).
 *
 * Block (kv, c), 4 warps: chunk c of FS_L rows from the chunk of the first
 * row of query 0, 32 rows at a time; the 2 T heads of the queries (query
 * j, head h: row 2 j + h) are the rows of one m16 tile and the columns of
 * one n8 tile. The method of k_attn_fdtc: warp w the scores of rows 8 w ..
 * 8 w + 7 (lane (g, cc): values 32 b + 8 cc .. + 7 of row g), joined in
 * shared memory, each warp the softmax that runs, then O^T of values 64 w
 * .. 64 w + 63. The chunks and the tiles sit at multiples of FS_L and 32
 * positions (not rows: the buffer drops its old rows at other steps in a
 * decode and in MTP), a query masks the rows out of its window, a tile with
 * none of its rows changes nothing in it, and the join of query j reads its
 * own chunks only: each query gets the bits of a step (T 1) at its
 * position, MTP verifies with the numbers of the decode. Rows out of the
 * window of query 0 to the last are read as its first or last row (their
 * values finite; weight 0). */
#define FS_L 64
#define FS_SLD 36
#define FS_MAXC 64

template <int T, int KB, int VB>
__global__ void __launch_bounds__(128) k_attn_fdts(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int R = 2 * T;                       /* rows of the tile in use */
    __shared__ __align__(16) uint4 qh[R][32];
    __shared__ __align__(16) uint4 ql[R][32];
    __shared__ __align__(16) float ss[2][R][FS_SLD];
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, c = blockIdx.y, ncs = gridDim.y;
    int n = a.n, W = DI(12), base = DI(13), hi = n + T - 1, lo0 = max(0, n - W);
    int x_beg = ((lo0 + base) / FS_L + c) * FS_L - base;     /* rows; maybe < 0 */
    if (x_beg >= hi) {
        return;
    }
    int x_end = min(hi, x_beg + FS_L);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, cc = lane % 4;
    int qhn = a.qh;
    /* q: row 2 j + h (query j, head 2 kv + h) */
    for (int i = threadIdx.x; i < R * 64; i += 128) {
        int row = i / 64, d = 4 * (i % 64);
        float4 v = *(const float4 *)(a.q + ((size_t)(row / 2) * qhn + kv * 2 + row % 2) * 256 + d);
        float f[4] = {v.x, v.y, v.z, v.w};
        uint32_t hw[2], lw[2];
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            __half h0 = __float2half_rn(f[2 * t]), h1 = __float2half_rn(f[2 * t + 1]);
            hw[t] = pack_h2(__half2float(h0), __half2float(h1));
            lw[t] = pack_h2(f[2 * t] - __half2float(h0), f[2 * t + 1] - __half2float(h1));
        }
        int ch = (d / 8) ^ (4 * (row & 1));
        *((uint2 *)&qh[row][ch] + (d % 8) / 4) = make_uint2(hw[0], hw[1]);
        *((uint2 *)&ql[row][ch] + (d % 8) / 4) = make_uint2(lw[0], lw[1]);
    }
    __syncthreads();
    const void *kb = a.kq, *vb = a.vq;              /* operands 1, 3 (int16 or int8) */
    /* the window of the query of row g (lanes past the rows: none) */
    int jq = g / 2, wlo = g < R ? max(0, n + jq - W) : 0, whi = g < R ? n + jq : 0;
    float acc[4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    int buf = 0;
    for (int x0 = x_beg; x0 < x_end; x0 += 32, buf ^= 1) {
        size_t ok = (size_t)min(max(x0 + 8 * warp + g, lo0), hi - 1) * a.rstride +
                    (size_t)kv * a.hstride;
        uint4 rk[8];
        float ck[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            size_t o = ok + 32 * u + 8 * cc;
            rk[u] = ft_ld<KB>(kb, o);
            ck[u] = a.ks[o / 32];
        }
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            uint32_t k[4];
            ft_h8<KB>(rk[u], ck[u], k);
            int ch = 4 * u + cc;
            #pragma unroll
            for (int pl = 0; pl < 2; ++pl) {
                uint4 q0 = g < R ? (pl ? ql : qh)[g][ch ^ (4 * (g & 1))] : make_uint4(0u, 0u, 0u, 0u);
                uint32_t av[4] = {q0.x, 0u, q0.y, 0u};    /* rows 8-15: none */
                uint32_t b0[2] = {k[0], k[1]};
                mma16816(s, av, b0);
                av[0] = q0.z; av[2] = q0.w;
                b0[0] = k[2]; b0[1] = k[3];
                mma16816(s, av, b0);
            }
        }
        if (g < R) {
            *(float2 *)&ss[buf][g][8 * warp + 2 * cc] = make_float2(s[0], s[1]);
        }
        /* the values (row 16 (k / 4) + 2 cc + (k & 1) + 8 ((k >> 1) & 1)),
         * while the scores join */
        uint4 rv[8];
        float cv[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int x = x0 + 16 * (k / 4) + 2 * cc + (k & 1) + 8 * ((k >> 1) & 1);
            size_t o = (size_t)min(max(x, lo0), hi - 1) * a.rstride + (size_t)kv * a.hstride +
                       64 * warp + 8 * g;
            rv[k] = ft_ld<VB>(vb, o);
            cv[k] = a.vs[o / 32];
        }
        __syncthreads();
        /* the softmax of row g: rows x0 + 16 ks + 2 cc + {0, 1, 8, 9} */
        float sv[8];
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                int kk = 16 * ks + 2 * cc + 8 * hf, x = x0 + kk;
                float2 v = g < R ? *(const float2 *)&ss[buf][g][kk] : make_float2(0.f, 0.f);
                sv[4 * ks + 2 * hf] = x >= wlo && x < whi ? v.x : -INFINITY;
                sv[4 * ks + 2 * hf + 1] = x + 1 >= wlo && x + 1 < whi ? v.y : -INFINITY;
            }
        }
        float t0 = sv[0];
        #pragma unroll
        for (int u = 1; u < 8; ++u) {
            t0 = fmaxf(t0, sv[u]);
        }
        t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 1));
        t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 2));
        float n0 = fmaxf(m, t0);
        /* no row of the window yet (a tile out of it): nothing changes */
        float a0 = n0 == -INFINITY ? 1.f : expf(m - n0);
        m = n0;
        float ps = 0.f;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            sv[u] = n0 == -INFINITY ? 0.f : expf(sv[u] - n0);
            ps += sv[u];
        }
        l = l * a0 + ps;
        uint32_t pb[2][2];
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            pb[ks][0] = pack_h2(sv[4 * ks], sv[4 * ks + 1]);
            pb[ks][1] = pack_h2(sv[4 * ks + 2], sv[4 * ks + 3]);
        }
        /* the factor of the columns 2 cc and 2 cc + 1 from the lanes of their rows */
        float al0 = __shfl_sync(0xffffffff, a0, 8 * cc), al1 = __shfl_sync(0xffffffff, a0, 8 * cc + 4);
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            acc[mt][0] *= al0;
            acc[mt][1] *= al1;
            acc[mt][2] *= al0;
            acc[mt][3] *= al1;
        }
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t h[4][4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                ft_h8<VB>(rv[4 * ks + k], cv[4 * ks + k], h[k]);
            }
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                uint32_t av[4] = {__byte_perm(h[0][mt], h[1][mt], 0x5410),
                                  __byte_perm(h[0][mt], h[1][mt], 0x7632),
                                  __byte_perm(h[2][mt], h[3][mt], 0x5410),
                                  __byte_perm(h[2][mt], h[3][mt], 0x7632)};
                mma16816(acc[mt], av, pb[ks]);
            }
        }
    }
    l += __shfl_xor_sync(0xffffffff, l, 1);
    l += __shfl_xor_sync(0xffffffff, l, 2);
    /* the part of each head: values 64 warp + 8 g + 2 mt (+ 1) of the
     * columns 2 cc, 2 cc + 1 (query cc, heads 2 kv, 2 kv + 1) */
    if (2 * cc < R) {
        float *p0 = part + ((size_t)(cc * qhn + kv * 2) * ncs + c) * 258;
        float *p1 = part + ((size_t)(cc * qhn + kv * 2 + 1) * ncs + c) * 258;
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            int d = 64 * warp + 8 * g + 2 * mt;
            *(float2 *)&p0[d] = make_float2(acc[mt][0], acc[mt][2]);
            *(float2 *)&p1[d] = make_float2(acc[mt][1], acc[mt][3]);
        }
    }
    if (warp == 0 && cc == 0 && g < R) {
        float *o = part + ((size_t)(jq * qhn + kv * 2 + g % 2) * ncs + c) * 258;
        o[256] = m;
        o[257] = l;
    }
}

/* The join of k_attn_fdts: block (h, x) the values x * 128 to x * 128 + 127
 * of head h % qh of query h / qh, over the chunks of its window only (in
 * the order and on the threads of a step at its position). ncs: the chunk
 * stride. */
__global__ void k_attn_fdts_join(const gp_rec *r, const int64_t *e, const float *part, int ncs)
{
    PDL_START();
    __shared__ float w[FS_MAXC];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x, W = DI(12), base = DI(13);
    int hj = a.n + h / a.qh, lj = max(0, hj - W);
    int c0 = (max(0, a.n - W) + base) / FS_L, cj = (lj + base) / FS_L;
    int nc = (hj - 1 + base) / FS_L - cj + 1;
    const float *ph = part + ((size_t)h * ncs + (cj - c0)) * 258;
    float M = -INFINITY;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        M = fmaxf(M, ph[(size_t)c * 258 + 256]);
    }
    M = block_max(M);
    float wl = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float mc = ph[(size_t)c * 258 + 256];
        float wc = mc == -INFINITY ? 0.f : expf(mc - M);
        w[c] = wc;
        wl += wc * ph[(size_t)c * 258 + 257];
    }
    float wsum = block_sum(wl);
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    __syncthreads();
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < 256) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) {
            acc += w[c] * ph[(size_t)c * 258 + i];
        }
        a.out[(size_t)h * 256 + i] = acc * inv;
    }
}

/* NP_GEMMA_GPU_FDT_TC: 1 (the default) k_attn_fdtc for a global layer of the
 * 26B; 0 k_attn_fdt (a test; no record of a group, np_gemma/gpu.py). */
static int fdtc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_FDT_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return on;
}

/* The chunks of k_attn_fdtc for kvh key heads: one block for each place on
 * the GPU (of a step), all at once (512 blocks, 3.1 rounds on the 82 SMs of
 * the 3090, left the last round 12% full); NP_GEMMA_GPU_FDT_NB sets it. The
 * same for every T (a group verifies with the numbers of a step), and the
 * parts of FT_TMAX queries fit GG_PART_FLOATS. */
static int fdtc_chunks(int kvh)
{
    static int nb = -1;
    if (nb < 0) {
        const char *v = getenv("NP_GEMMA_GPU_FDT_NB");
        int dev = 0, sms = 0, occ = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_attn_fdtc<1, 16, 16>, 256, FT_SMEM(1));
        nb = v != NULL ? atoi(v) : sms * (occ > 0 ? occ : 1);
    }
    return max(1, min(ATTN_CHUNKS / FT_TMAX, nb / max(1, kvh)));
}

template <int T>
static void fdts_run(int op, dim3 grid, const gp_rec *dr, const int64_t *denv, float *part)
{
    if (op == GP_ATTN_QC) {
        k_attn_fdts<T, 16, 16><<<grid, 128, 0, gg_stream>>>(dr, denv, part);
    } else if (op == GP_ATTN_Q8) {
        k_attn_fdts<T, 8, 8><<<grid, 128, 0, gg_stream>>>(dr, denv, part);
    } else {
        k_attn_fdts<T, 16, 8><<<grid, 128, 0, gg_stream>>>(dr, denv, part);
    }
}

template <int T>
static void fdtc_run(int op, dim3 grid, const gp_rec *dr, const int64_t *denv, float *part)
{
    static int attr = 0;
    if (!attr) {
        cudaFuncSetAttribute(k_attn_fdtc<T, 16, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)FT_SMEM(T));
        cudaFuncSetAttribute(k_attn_fdtc<T, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)FT_SMEM(T));
        cudaFuncSetAttribute(k_attn_fdtc<T, 16, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)FT_SMEM(T));
        attr = 1;
    }
    if (op == GP_ATTN_QC) {
        k_attn_fdtc<T, 16, 16><<<grid, 256, FT_SMEM(T), gg_stream>>>(dr, denv, part);
    } else if (op == GP_ATTN_Q8) {
        k_attn_fdtc<T, 8, 8><<<grid, 256, FT_SMEM(T), gg_stream>>>(dr, denv, part);
    } else {
        k_attn_fdtc<T, 16, 8><<<grid, 256, FT_SMEM(T), gg_stream>>>(dr, denv, part);
    }
}

/* GP_ATTN_QSA of one query (a step, a row of a small group: the MTP layer
 * over all the positions, a QSA layer over its rows) on the tensor cores,
 * in the chunks of k_attn_part (k_attn_join adds them). k_attn_part takes
 * 8 query heads in a block, so the 12 heads of a key head of Qwen3.8 took
 * two blocks, each reading all the keys and values of the chunk: at 200K
 * positions 0.43 ms a row of the MTP layer, 18% of the decode. Block (kv,
 * c): chunk c, the heads of key head kv as the rows of one m16 tile, the
 * method of k_attn_qsa_tc (4 warps, each with its own 16 keys at a time of
 * the chunk; q as two float16 planes with Q2). It writes the part of each
 * head as k_attn_part does: the sum of exp(score - m) v, m, l. */
template <int Q2>
__global__ void __launch_bounds__(128) k_attn_part_tc(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    extern __shared__ __align__(16) __half qtsm[];
    __half (*qh)[QT_LD] = (__half (*)[QT_LD])qtsm;
    __half (*ql)[QT_LD] = (__half (*)[QT_LD])(qtsm + 16 * QT_LD);
    __shared__ float wm[4][16], wl[4][16];
    __shared__ float cbs[64];
    if (threadIdx.x < 64) {
        cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
    }
    int nq = DI(7), nk = DI(8), form = DI(15);
    int rep = nq / nk, kv = blockIdx.x, c = blockIdx.y;
    int cn = DP(const int32_t, 13)[0];
    int n = cn < 0 ? (int)(di(r, e, 11) + 1) : cn;
    int len = attn_len(n);
    int j0 = c * len, j1 = min(n, j0 + len);
    if (j0 >= n) {
        return;
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, cc = lane % 4;
    __half (*ks_)[QT_LD] = (__half (*)[QT_LD])(qtsm + (2 + warp) * 16 * QT_LD);
    const float *q = DP(const float, 0) + (size_t)kv * rep * 256;
    for (int i = threadIdx.x; i < 16 * 256; i += blockDim.x) {
        int h = i / 256, d = i % 256;
        float v = h < rep ? q[(size_t)h * 256 + d] : 0.f;
        __half hi = __float2half_rn(v);
        qh[h][d] = hi;
        ql[h][d] = __float2half_rn(v - __half2float(hi));
    }
    __syncthreads();
    const int32_t *rows = cn < 0 ? NULL : DP(const int32_t, 12);
    const uint8_t *kb = DP(const uint8_t, 1), *vb = DP(const uint8_t, 3);
    const float *ksc = DP(const float, 2), *vsc = DP(const float, 4);
    size_t rs = (size_t)nk * 256, off = (size_t)kv * 256 + 8 * lane;
    auto decode = [&](const uint8_t *base, const float *sc, int x0) {
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            int x = x0 + rr;
            float f[8];
            if (x < j1) {
                size_t o = (size_t)(rows ? rows[x] : x) * rs + off;
                if (form == 4) {
                    tq6x8(base, o, f, cbs);
                } else {
                    fd_i8x8(*(const uint2 *)((const int8_t *)base + o), f);
                }
                float s = sc[o / 32];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    f[u] *= s;
                }
            } else {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    f[u] = 0.f;
                }
            }
            *(uint4 *)&ks_[rr][8 * lane] = make_uint4(pack_h2(f[0], f[1]), pack_h2(f[2], f[3]),
                                                      pack_h2(f[4], f[5]), pack_h2(f[6], f[7]));
        }
        __syncwarp();
    };
    float acc[32][4];
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int x0 = j0 + warp * 16; x0 < j1; x0 += 64) {
        decode(kb, ksc, x0);
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int kk = 0; kk < 256; kk += 16) {
            uint32_t a[4], b[2][2];
            #pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                b[jn][0] = *(const uint32_t *)&ks_[jn * 8 + g][kk + 2 * cc];
                b[jn][1] = *(const uint32_t *)&ks_[jn * 8 + g][kk + 2 * cc + 8];
            }
            a[0] = *(const uint32_t *)&qh[g][kk + 2 * cc];
            a[1] = *(const uint32_t *)&qh[g + 8][kk + 2 * cc];
            a[2] = *(const uint32_t *)&qh[g][kk + 2 * cc + 8];
            a[3] = *(const uint32_t *)&qh[g + 8][kk + 2 * cc + 8];
            mma16816(s[0], a, b[0]);
            mma16816(s[1], a, b[1]);
            if (Q2) {
                a[0] = *(const uint32_t *)&ql[g][kk + 2 * cc];
                a[1] = *(const uint32_t *)&ql[g + 8][kk + 2 * cc];
                a[2] = *(const uint32_t *)&ql[g][kk + 2 * cc + 8];
                a[3] = *(const uint32_t *)&ql[g + 8][kk + 2 * cc + 8];
                mma16816(s[0], a, b[0]);
                mma16816(s[1], a, b[1]);
            }
        }
        __syncwarp();
        #pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            int x = x0 + jn * 8 + 2 * cc;
            if (x >= j1) { s[jn][0] = s[jn][2] = -INFINITY; }
            if (x + 1 >= j1) { s[jn][1] = s[jn][3] = -INFINITY; }
        }
        float t0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
        float t1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
        t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 1));
        t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, 1));
        t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, 2));
        float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);     /* finite: key x0 < j1 */
        float a0 = expf(m0 - n0), a1 = expf(m1 - n1);
        m0 = n0;
        m1 = n1;
        float p[2][4];
        #pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            p[jn][0] = expf(s[jn][0] - n0);
            p[jn][1] = expf(s[jn][1] - n0);
            p[jn][2] = expf(s[jn][2] - n1);
            p[jn][3] = expf(s[jn][3] - n1);
        }
        l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
        l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            acc[nt][0] *= a0;
            acc[nt][1] *= a0;
            acc[nt][2] *= a1;
            acc[nt][3] *= a1;
        }
        decode(vb, vsc, x0);
        uint32_t pa[4] = {pack_h2(p[0][0], p[0][1]), pack_h2(p[0][2], p[0][3]),
                          pack_h2(p[1][0], p[1][1]), pack_h2(p[1][2], p[1][3])};
        #pragma unroll
        for (int nt = 0; nt < 32; nt += 2) {
            uint32_t bv[4];
            ldsm_x4_trans(bv, &ks_[(lane % 8) + 8 * ((lane / 8) & 1)][nt * 8 + 8 * (lane / 16)]);
            mma16816(acc[nt], pa, bv);
            mma16816(acc[nt + 1], pa, bv + 2);
        }
        __syncwarp();
    }
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    if (cc == 0) {
        wm[warp][g] = m0;
        wm[warp][g + 8] = m1;
        wl[warp][g] = l0;
        wl[warp][g + 8] = l1;
    }
    __syncthreads();
    /* the join of the warps (k_attn_qsa_tc): rows g and g + 8, the weight
     * exp(m_w - M) of each warp; no division (k_attn_join divides) */
    float M0 = -INFINITY, M1 = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        M0 = fmaxf(M0, wm[w][g]);
        M1 = fmaxf(M1, wm[w][g + 8]);
    }
    float L0 = 0.f, L1 = 0.f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        L0 += wm[w][g] == -INFINITY ? 0.f : expf(wm[w][g] - M0) * wl[w][g];
        L1 += wm[w][g + 8] == -INFINITY ? 0.f : expf(wm[w][g + 8] - M1) * wl[w][g + 8];
    }
    float f0 = m0 == -INFINITY ? 0.f : expf(m0 - M0);
    float f1 = m1 == -INFINITY ? 0.f : expf(m1 - M1);
    float (*red)[256] = (float (*)[256])qtsm;
    for (int w = 0; w < 4; ++w) {
        if (warp == w) {
            #pragma unroll
            for (int nt = 0; nt < 32; ++nt) {
                int d = nt * 8 + 2 * cc;
                if (w == 0) {
                    red[g][d] = acc[nt][0] * f0;
                    red[g][d + 1] = acc[nt][1] * f0;
                    red[g + 8][d] = acc[nt][2] * f1;
                    red[g + 8][d + 1] = acc[nt][3] * f1;
                } else {
                    red[g][d] += acc[nt][0] * f0;
                    red[g][d + 1] += acc[nt][1] * f0;
                    red[g + 8][d] += acc[nt][2] * f1;
                    red[g + 8][d + 1] += acc[nt][3] * f1;
                }
            }
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < rep * 256; i += blockDim.x) {
        int h = i / 256, d = i % 256;
        part[((size_t)(kv * rep + h) * ATTN_CHUNKS + c) * 258 + d] = red[h][d];
    }
    if (warp == 0 && cc == 0) {
        /* lanes g: rows g and g + 8 (M, L are the same in each warp) */
        if (g < rep) {
            float *o = part + ((size_t)(kv * rep + g) * ATTN_CHUNKS + c) * 258;
            o[256] = M0;
            o[257] = L0;
        }
        if (g + 8 < rep) {
            float *o = part + ((size_t)(kv * rep + g + 8) * ATTN_CHUNKS + c) * 258;
            o[256] = M1;
            o[257] = L1;
        }
    }
}

/* GP_ATTN_QSA of a large group whose queries all see every position before
 * them (maxsel 0: cnt is -1 for each query; the dense attention of the MTP
 * layer) on the tensor cores. k_attn_qsa_tc decodes all the keys and values
 * again for each query: at 200K positions a group of 256 rows of the MTP
 * layer read 51 GB, 375 ms. Here block (j4, kv) takes the QD_W queries
 * QD_W j4 .. QD_W j4 + QD_W - 1, one for each warp (its query heads of key
 * head kv are the rows of its m16 tile, as in k_attn_qsa_tc), and the warps
 * share each tile of 16 keys and values: the threads of the block decode it
 * one time to shared memory for the QD_W queries. The math of each query is
 * that of k_attn_qsa_tc (q as two float16 planes with Q2, the softmax that
 * runs, P float16, V by ldmatrix.trans); one warp holds a query, so there is
 * no join. */
#define QD_W 4
#define QD_SMEM ((size_t)(QD_W * 2 * 16 + 2 * 16) * QT_LD * 2)
template <int Q2>
__global__ void __launch_bounds__(32 * QD_W) k_attn_dense_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) __half qdsm[];
    __shared__ float cbs[64];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    __half (*qh)[QT_LD] = (__half (*)[QT_LD])(qdsm + (size_t)(2 * warp) * 16 * QT_LD);
    __half (*ql)[QT_LD] = (__half (*)[QT_LD])(qdsm + (size_t)(2 * warp + 1) * 16 * QT_LD);
    __half (*kt)[QT_LD] = (__half (*)[QT_LD])(qdsm + (size_t)(2 * QD_W) * 16 * QT_LD);
    __half (*vt)[QT_LD] = (__half (*)[QT_LD])(qdsm + (size_t)(2 * QD_W + 1) * 16 * QT_LD);
    if (threadIdx.x < 64) {
        cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
    }
    int nq = DI(7), nk = DI(8), t = DI(10), form = DI(15);
    int64_t pos = di(r, e, 11);
    int rep = nq / nk, kv = blockIdx.y;
    int j = blockIdx.x * QD_W + warp;
    bool live = j < t;
    /* this warp's query to its planes */
    const float *q = DP(const float, 0) + ((size_t)j * nq + (size_t)kv * rep) * 256;
    for (int i = lane; i < 16 * 256; i += 32) {
        int h = i / 256, d = i % 256;
        float v = live && h < rep ? q[(size_t)h * 256 + d] : 0.f;
        __half hi = __float2half_rn(v);
        qh[h][d] = hi;
        ql[h][d] = __float2half_rn(v - __half2float(hi));
    }
    int n = live ? (int)(pos + j + 1) : 0;                    /* the keys of this query */
    int jl = min(t, (int)(blockIdx.x + 1) * QD_W) - 1;
    int nmax = (int)(pos + jl + 1);                           /* the keys of the block */
    const uint8_t *kb = DP(const uint8_t, 1), *vb = DP(const uint8_t, 3);
    const float *ksc = DP(const float, 2), *vsc = DP(const float, 4);
    size_t rs = (size_t)nk * 256;
    /* row tid / 8 of the tile: 8 values at (tid % 8) * 8 + 64 u, u = 0..3 */
    int dr = threadIdx.x / 8, dc = (threadIdx.x % 8) * 8;
    auto decode = [&](const uint8_t *base, const float *sc, int x0, __half (*tile)[QT_LD]) {
        int x = x0 + dr;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int d0 = dc + 64 * u;
            float f[8];
            if (x < nmax) {
                size_t o = (size_t)x * rs + (size_t)kv * 256 + d0;
                if (form == 4) {
                    tq6x8(base, o, f, cbs);
                } else {
                    fd_i8x8(*(const uint2 *)((const int8_t *)base + o), f);
                }
                float s = sc[o / 32];
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    f[w] *= s;
                }
            } else {
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    f[w] = 0.f;
                }
            }
            *(uint4 *)&tile[dr][d0] = make_uint4(pack_h2(f[0], f[1]), pack_h2(f[2], f[3]),
                                                 pack_h2(f[4], f[5]), pack_h2(f[6], f[7]));
        }
    };
    float acc[32][4];
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;   /* rows g, g + 8 */
    /* int8: the bytes and the scales of the next tile come to registers
     * before the compute of this one (in flight meanwhile; the loads of a
     * tile right before its use left the SM idle, one block of 4 warps) */
    uint2 rk[4], rv[4];
    float sk[4], sv[4];
    auto fetch = [&](int x0) {
        int x = x0 + dr;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            size_t o = (size_t)x * rs + (size_t)kv * 256 + dc + 64 * u;
            bool in = x < nmax;
            rk[u] = in ? *(const uint2 *)((const int8_t *)kb + o) : make_uint2(0u, 0u);
            rv[u] = in ? *(const uint2 *)((const int8_t *)vb + o) : make_uint2(0u, 0u);
            sk[u] = in ? ksc[o / 32] : 0.f;
            sv[u] = in ? vsc[o / 32] : 0.f;
        }
    };
    auto store = [&]() {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float f[8], h[8];
            fd_i8x8(rk[u], f);
            fd_i8x8(rv[u], h);
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                f[w] *= sk[u];
                h[w] *= sv[u];
            }
            *(uint4 *)&kt[dr][dc + 64 * u] = make_uint4(pack_h2(f[0], f[1]), pack_h2(f[2], f[3]),
                                                        pack_h2(f[4], f[5]), pack_h2(f[6], f[7]));
            *(uint4 *)&vt[dr][dc + 64 * u] = make_uint4(pack_h2(h[0], h[1]), pack_h2(h[2], h[3]),
                                                        pack_h2(h[4], h[5]), pack_h2(h[6], h[7]));
        }
    };
    if (form == 1) {
        fetch(0);
    }
    for (int x0 = 0; x0 < nmax; x0 += 16) {
        if (form == 1) {
            store();
            __syncthreads();
            if (x0 + 16 < nmax) {
                fetch(x0 + 16);
            }
        } else {
            decode(kb, ksc, x0, kt);
            decode(vb, vsc, x0, vt);
            __syncthreads();
        }
        if (x0 < n) {
            float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int kk = 0; kk < 256; kk += 16) {
                uint32_t a[4], b[2][2];
                #pragma unroll
                for (int jn = 0; jn < 2; ++jn) {
                    b[jn][0] = *(const uint32_t *)&kt[jn * 8 + g][kk + 2 * c];
                    b[jn][1] = *(const uint32_t *)&kt[jn * 8 + g][kk + 2 * c + 8];
                }
                a[0] = *(const uint32_t *)&qh[g][kk + 2 * c];
                a[1] = *(const uint32_t *)&qh[g + 8][kk + 2 * c];
                a[2] = *(const uint32_t *)&qh[g][kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&qh[g + 8][kk + 2 * c + 8];
                mma16816(s[0], a, b[0]);
                mma16816(s[1], a, b[1]);
                if (Q2) {
                    a[0] = *(const uint32_t *)&ql[g][kk + 2 * c];
                    a[1] = *(const uint32_t *)&ql[g + 8][kk + 2 * c];
                    a[2] = *(const uint32_t *)&ql[g][kk + 2 * c + 8];
                    a[3] = *(const uint32_t *)&ql[g + 8][kk + 2 * c + 8];
                    mma16816(s[0], a, b[0]);
                    mma16816(s[1], a, b[1]);
                }
            }
            /* the keys past n (this query): no weight */
            #pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                int x = x0 + jn * 8 + 2 * c;
                if (x >= n) { s[jn][0] = s[jn][2] = -INFINITY; }
                if (x + 1 >= n) { s[jn][1] = s[jn][3] = -INFINITY; }
            }
            float t0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
            float t1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 1));
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, 2));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, 1));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, 2));
            float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);     /* finite: key x0 < n */
            float a0 = expf(m0 - n0), a1 = expf(m1 - n1);
            m0 = n0;
            m1 = n1;
            float p[2][4];
            #pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                p[jn][0] = expf(s[jn][0] - n0);
                p[jn][1] = expf(s[jn][1] - n0);
                p[jn][2] = expf(s[jn][2] - n1);
                p[jn][3] = expf(s[jn][3] - n1);
            }
            l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
            l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
            #pragma unroll
            for (int nt = 0; nt < 32; ++nt) {
                acc[nt][0] *= a0;
                acc[nt][1] *= a0;
                acc[nt][2] *= a1;
                acc[nt][3] *= a1;
            }
            uint32_t pa[4] = {pack_h2(p[0][0], p[0][1]), pack_h2(p[0][2], p[0][3]),
                              pack_h2(p[1][0], p[1][1]), pack_h2(p[1][2], p[1][3])};
            #pragma unroll
            for (int nt = 0; nt < 32; nt += 2) {
                uint32_t bv[4];
                ldsm_x4_trans(bv, &vt[(lane % 8) + 8 * ((lane / 8) & 1)][nt * 8 + 8 * (lane / 16)]);
                mma16816(acc[nt], pa, bv);
                mma16816(acc[nt + 1], pa, bv + 2);
            }
        }
        __syncthreads();
    }
    if (!live) {
        return;
    }
    l0 += __shfl_xor_sync(0xffffffff, l0, 1);
    l0 += __shfl_xor_sync(0xffffffff, l0, 2);
    l1 += __shfl_xor_sync(0xffffffff, l1, 1);
    l1 += __shfl_xor_sync(0xffffffff, l1, 2);
    float f0 = 1.f / l0, f1 = 1.f / l1;
    float *out = DP(float, 6) + ((size_t)j * nq + (size_t)kv * rep) * 256;
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        int d = nt * 8 + 2 * c;
        if (g < rep) {
            out[(size_t)g * 256 + d] = acc[nt][0] * f0;
            out[(size_t)g * 256 + d + 1] = acc[nt][1] * f0;
        }
        if (g + 8 < rep) {
            out[(size_t)(g + 8) * 256 + d] = acc[nt][2] * f1;
            out[(size_t)(g + 8) * 256 + d + 1] = acc[nt][3] * f1;
        }
    }
}

/* NP_GEMMA_GPU_QSA_TC: 1 (the default) k_attn_qsa_tc with q as two float16
 * planes; 2 one plane; 0 k_attn_qsa_mt (float32). For the int8 and TQ6 forms
 * of the cache; the others keep k_attn_qsa_mt. */
static int qsa_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_QSA_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return on;
}

/* NP_GEMMA_GPU_PART_TC: 1 (the default) k_attn_part_tc for GP_ATTN_QSA of
 * one query (int8 or TQ6); 0 k_attn_part (a test). */
static int part_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_PART_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return on;
}

/* NP_GEMMA_GPU_DENSE_TC: 1 (the default) k_attn_dense_tc for a record of
 * maxsel 0; 0 k_attn_qsa_tc (a test). */
static int dense_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_DENSE_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return on;
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
    PDL_START();
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
    const size_t frs = kv_rs((size_t)hs, kvh, HD);
    bool live0 = j0 + g < t, live1 = j0 + g + 8 < t;
    int64_t p0 = pos + j0 + g, p1 = p0 + 8;
    int64_t e0 = p0, e1 = p1;
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
            size_t o2 = (size_t)(ok ? k0 + a : k0) * frs + d;
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
                bool ok0 = live0 && kk < kn && kp <= e0 && (window == 0 || p0 - kp < window);
                bool ok1 = live1 && kk < kn && kp <= e1 && (window == 0 || p1 - kp < window);
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

/* ---------- the attention of a large group over the int16 cache ----------
 * k_flash_qc_tc: k_flash_f32h for the record GP_ATTN_QC_MT (the int16 cache
 * of the 26B: rows of (kv heads, head_dim) int16 values and a float scale
 * for each 32 values, from row base). The differences:
 *
 * 1. cp.async copies the int16 rows (half the bytes of float32) and their
 *    scales. A row has HD + 8 values in shared memory, so the reads of the
 *    fragments of K and of V do not hit the same bank.
 * 2. A fragment of K is 16 values of a row: they have one scale. A value of
 *    V gets the scale of its row and of its block of 32.
 *
 * The queries, the keys, the values, and the weights p become float16, and
 * the sums are float32, as in k_flash_tc. The queries and the keys of the
 * 26B have a norm, so float16 has no overflow here. */
template <int HD>
struct flash_qc_dims {
    static constexpr int NS = HD / 256;
    static constexpr int LD = HD + 8;          /* int16 values of a row of K or V */
    static constexpr int NB = HD / 32;         /* scales of a row */
    static constexpr int QLD = HD + 8;
    static constexpr size_t smem(int hb, int st)
    {
        return (size_t)st * FK3 * (2 * LD * sizeof(int16_t) + 2 * NB * sizeof(float)) +
               (NS == 1 ? 0 : (size_t)hb * 16 * QLD * sizeof(__half));
    }
};

template <int HD, int HB, int ST>
__global__ void __launch_bounds__(32 * HB * (HD / 256))
k_flash_qc_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    typedef flash_qc_dims<HD> D;
    extern __shared__ __align__(16) uint8_t fsmq[];
    int16_t *kbuf = (int16_t *)fsmq;                          /* ST x FK3 x LD */
    int16_t *vbuf = kbuf + ST * FK3 * D::LD;                  /* ST x FK3 x LD */
    float *ksb = (float *)(vbuf + ST * FK3 * D::LD);          /* ST x FK3 x NB */
    float *vsb = ksb + ST * FK3 * D::NB;                      /* ST x FK3 x NB */
    __half *qs = (__half *)(vsb + ST * FK3 * D::NB);          /* HB x 16 x QLD (HD 512) */
    int qh = DI(7), kvh = DI(8), t = DI(10);
    int window = DI(13);
    int64_t pos = di(r, e, 11), base = di(r, e, 12);
    const float *q = DP(const float, 0);
    float *out = DP(float, 6);
    int G = qh / kvh, kv = blockIdx.x, j0 = blockIdx.y * 16;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int hl = warp / D::NS, dsl = warp % D::NS;
    int h = kv * G + blockIdx.z * HB + hl;
    int g = lane / 4, c = lane % 4;
    size_t rowq = (size_t)kvh * HD;
    const int16_t *kq = DP(const int16_t, 1) + (size_t)kv * HD;
    const int16_t *vq = DP(const int16_t, 3) + (size_t)kv * HD;
    const float *ks = DP(const float, 2) + (size_t)kv * D::NB;
    const float *vs = DP(const float, 4) + (size_t)kv * D::NB;
    bool live0 = j0 + g < t, live1 = j0 + g + 8 < t;
    int64_t p0 = pos + j0 + g, p1 = p0 + 8;
    int jl = min(t, j0 + 16) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int *lim = DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t e0 = live0 ? qc_last(lim, pos, j0 + g) : p0;
    int64_t e1 = live1 ? qc_last(lim, pos, j0 + g + 8) : p1;
    int nsteps = (int)((last - first) / FK3) + 1;

    auto load = [&](int stage, int64_t k0) {
        int kn = (int)min((int64_t)FK3, last - k0 + 1);
        int16_t *kd = kbuf + stage * FK3 * D::LD, *vd = vbuf + stage * FK3 * D::LD;
        float *ksd = ksb + stage * FK3 * D::NB, *vsd = vsb + stage * FK3 * D::NB;
        for (int x = threadIdx.x; x < FK3 * HD / 8; x += blockDim.x) {
            int a = x / (HD / 8), d = (x % (HD / 8)) * 8;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * rowq + d;
            cp_async16_z(kd + a * D::LD + d, kq + o2, ok ? 16 : 0);
            cp_async16_z(vd + a * D::LD + d, vq + o2, ok ? 16 : 0);
        }
        for (int x = threadIdx.x; x < FK3 * D::NB / 4; x += blockDim.x) {
            int a = x / (D::NB / 4), b = (x % (D::NB / 4)) * 4;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * (rowq / 32) + b;
            cp_async16_z(ksd + a * D::NB + b, ks + o2, ok ? 16 : 0);
            cp_async16_z(vsd + a * D::NB + b, vs + o2, ok ? 16 : 0);
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
        const int16_t *kd = kbuf + st * FK3 * D::LD, *vd = vbuf + st * FK3 * D::LD;
        const float *ksd = ksb + st * FK3 * D::NB, *vsd = vsb + st * FK3 * D::NB;
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
                int row = nt * 8 + g;
                const int16_t *kr = kd + row * D::LD + kk + 2 * c;
                float sk = ksd[row * D::NB + kk / 32];
                short2 x0 = *(const short2 *)kr, x1 = *(const short2 *)(kr + 8);
                uint32_t b[2] = {pack_h2((float)x0.x * sk, (float)x0.y * sk),
                                 pack_h2((float)x1.x * sk, (float)x1.y * sk)};
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
                bool ok0 = live0 && kk < kn && kp <= e0 && (window == 0 || p0 - kp < window);
                bool ok1 = live1 && kk < kn && kp <= e1 && (window == 0 || p1 - kp < window);
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
            const int16_t *vr = vd + (2 * c) * D::LD + d;
            const float *sr = vsd + (2 * c) * D::NB + d / 32;
            uint32_t b[2] = {pack_h2((float)vr[0] * sr[0], (float)vr[D::LD] * sr[D::NB]),
                             pack_h2((float)vr[8 * D::LD] * sr[8 * D::NB],
                                     (float)vr[9 * D::LD] * sr[9 * D::NB])};
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

/* k_flash_qc_h: k_flash_qc_tc for the attention of a prompt pass, with
 * the work of a block shared by more warps. A block takes HB query heads of
 * one key head and NQT tiles of 16 queries; a query head of HD values has NS
 * = HD / 256 warps for each tile, and warp dsl keeps the output values dsl
 * * 256 to + 255. The differences from k_flash_qc_tc:
 *
 * 1. The threads of the block convert the int16 keys and values of a step
 *    to float16 once (the keys by rows, the values by columns), and the
 *    warps read the fragments as two float16 values. In k_flash_qc_tc each
 *    warp converts each value that it reads. The copy of the next keys
 *    (cp.async) runs while the warps compute.
 * 2. With HD 512, each warp computes the scores over its 256 values only.
 *    The two warps then add the two parts in shared memory, the part of
 *    warp 0 first, so both have the same scores. k_flash_qc_tc computes
 *    the scores of all 512 values in both.
 * 3. Thus a warp keeps its 256 values of the queries in registers, and the
 *    queries need no shared memory.
 *
 * The float16 values are those of k_flash_qc_tc. With HD 512, the order of
 * the sums of the scores differs. The keys of a step are those of the
 * NQT tiles of the block; a tile masks the keys after its queries. */
template <int HD, int HB, int NQT, int FKS, bool KQ8 = false, bool VQ8 = false, bool TQ = false>
struct flash_h_dims {
    static constexpr int NS = HD / 256, WARPS = HB * NQT * NS;
    static constexpr int LD = HD + 8, NB = HD / 32, VLD = FKS + 8;
    /* the bytes of a row of the cache: int16, int8 (KQ8, VQ8), or TQ6 (3
     * bytes for 4 values) */
    static constexpr int RBK = TQ ? HD * 3 / 4 : (KQ8 ? HD : 2 * HD);
    static constexpr int RBV = TQ ? HD * 3 / 4 : (VQ8 ? HD : 2 * HD);
    /* the bytes of a staged row of keys or values */
    static constexpr int LDBK = TQ ? RBK + 16 : (KQ8 ? HD + 16 : 2 * (HD + 8));
    static constexpr int LDBV = TQ ? RBV + 16 : (VQ8 ? HD + 16 : 2 * (HD + 8));
    static constexpr size_t smem()
    {
        return (size_t)FKS * (LDBK + LDBV + 2 * NB * sizeof(float) + LD * sizeof(__half)) +
               (size_t)HD * VLD * sizeof(__half) +
               (NS == 2 ? (size_t)WARPS * 32 * (FKS / 2) * sizeof(float) : 0);
    }
};

/* The int16 at bit sh (0 or 16) of w as a float, exactly: the offset
 * binary value (q + 32768, the bits of q with the sign bit flipped) in the
 * mantissa of 2^23, minus 2^23 + 32768. Two full-rate operations, in place
 * of a conversion of an integer (a quarter of the rate), as i4f. */
__device__ __forceinline__ float i16f(uint32_t w, int sh)
{
    return __uint_as_float((((w >> sh) & 0xFFFFu) ^ 0x8000u) | 0x4B000000u) - 8421376.f;
}

/* The int8 at bit sh (0, 8, 16, 24) of w as a float, as i16f. */
__device__ __forceinline__ float i8f(uint32_t w, int sh)
{
    return __uint_as_float((((w >> sh) & 0xFFu) ^ 0x80u) | 0x4B000000u) - 8388736.f;
}

template <int HD, int HB, int NQT, int FKS, bool KQ8 = false, bool VQ8 = false, bool TQ = false>
__global__ void __launch_bounds__(32 * HB * NQT * (HD / 256))
k_flash_qc_h(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    typedef flash_h_dims<HD, HB, NQT, FKS, KQ8, VQ8, TQ> D;
    constexpr int NS = D::NS, LD = D::LD, NB = D::NB, VLD = D::VLD;
    constexpr int LDBK = D::LDBK, LDBV = D::LDBV;
    constexpr int RBK = D::RBK, RBV = D::RBV;          /* the bytes of a row of a head */
    extern __shared__ __align__(16) uint8_t fsm5[];
    uint8_t *kbuf = fsm5;                             /* FKS x LDBK bytes */
    uint8_t *vbuf = kbuf + FKS * LDBK;                /* FKS x LDBV bytes */
    float *ksb = (float *)(vbuf + FKS * LDBV);        /* FKS x NB */
    float *vsb = ksb + FKS * NB;                      /* FKS x NB */
    __half *kh = (__half *)(vsb + FKS * NB);          /* FKS x LD */
    __half *vt = kh + FKS * LD;                       /* HD x VLD */
    float *sx = (float *)(vt + HD * VLD);            /* NS 2: WARPS x 32 lanes x FKS / 2 */
    int qh = DI(7), kvh = DI(8), t = DI(10);
    int window = DI(13);
    int64_t pos = di(r, e, 11), base = di(r, e, 12);
    const float *q = DP(const float, 0);
    float *out = DP(float, 6);
    int G = qh / kvh, kv = blockIdx.x, j0 = blockIdx.y * 16 * NQT;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int dsl = warp % NS, qt = (warp / NS) % NQT, hl = warp / NS / NQT;
    int h = kv * G + blockIdx.z * HB + hl;
    int jq = j0 + qt * 16;                           /* the queries of this warp */
    int g = lane / 4, c = lane % 4;
    size_t rowq = (size_t)kvh * HD;
    const uint8_t *kq = DP(const uint8_t, 1) + (size_t)kv * RBK;
    const uint8_t *vq = DP(const uint8_t, 3) + (size_t)kv * RBV;
    const float *ks = DP(const float, 2) + (size_t)kv * NB;
    const float *vs = DP(const float, 4) + (size_t)kv * NB;
    bool live0 = jq + g < t, live1 = jq + g + 8 < t;
    int64_t p0 = pos + jq + g, p1 = p0 + 8;
    int jl = min(t, j0 + 16 * NQT) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int *lim = DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t e0 = live0 ? qc_last(lim, pos, jq + g) : p0;
    int64_t e1 = live1 ? qc_last(lim, pos, jq + g + 8) : p1;
    int nsteps = (int)((last - first) / FKS) + 1;

    auto load = [&](int64_t k0) {
        int kn = (int)min((int64_t)FKS, last - k0 + 1);
        /* 16 bytes at a time: 8 int16 or 16 int8 values */
        for (int x = threadIdx.x; x < FKS * RBK / 16; x += blockDim.x) {
            int a = x / (RBK / 16), d = (x % (RBK / 16)) * 16;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * (rowq / HD) * RBK + d;
            cp_async16_z(kbuf + a * LDBK + d, kq + o2, ok ? 16 : 0);
        }
        for (int x = threadIdx.x; x < FKS * RBV / 16; x += blockDim.x) {
            int a = x / (RBV / 16), d = (x % (RBV / 16)) * 16;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * (rowq / HD) * RBV + d;
            cp_async16_z(vbuf + a * LDBV + d, vq + o2, ok ? 16 : 0);
        }
        for (int x = threadIdx.x; x < FKS * NB / 4; x += blockDim.x) {
            int a = x / (NB / 4), b = (x % (NB / 4)) * 4;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * (rowq / 32) + b;
            cp_async16_z(ksb + a * NB + b, ks + o2, ok ? 16 : 0);
            cp_async16_z(vsb + a * NB + b, vs + o2, ok ? 16 : 0);
        }
        cp_async_commit();
    };
    load(first);

    /* The queries: rows g and g + 8, the values dsl * 256 to + 255. */
    const float *q0r = q + ((size_t)(jq + g) * qh + h) * HD + dsl * 256;
    const float *q1r = q + ((size_t)(jq + g + 8) * qh + h) * HD + dsl * 256;
    uint32_t qa[16][4];
    #pragma unroll
    for (int k2 = 0; k2 < 16; ++k2) {
        int d = k2 * 16 + 2 * c;
        qa[k2][0] = live0 ? pack_h2(q0r[d], q0r[d + 1]) : 0u;
        qa[k2][1] = live1 ? pack_h2(q1r[d], q1r[d + 1]) : 0u;
        qa[k2][2] = live0 ? pack_h2(q0r[d + 8], q0r[d + 9]) : 0u;
        qa[k2][3] = live1 ? pack_h2(q1r[d + 8], q1r[d + 9]) : 0u;
    }

    float o[32][4];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    constexpr int NT = FKS / 8, KC = FKS / 16;         /* tiles of 8 keys; chunks of 16 */
    float *mine = sx + (warp * 32 + lane) * (FKS / 2);
    const float *other = sx + ((warp ^ 1) * 32 + lane) * (FKS / 2);
    for (int s = 0; s < nsteps; ++s) {
        int64_t k0 = first + (int64_t)s * FKS;
        int kn = (int)min((int64_t)FKS, last - k0 + 1);
        cp_async_wait0();
        __syncthreads();
        /* The keys as float16 rows; the values as float16 columns (two
         * adjacent keys of a column in one word). A thread converts 8
         * values of a key row (one 16-byte load and store), or 8 values of
         * two adjacent keys; adjacent threads take adjacent pairs of keys,
         * so their stores to vt fall in different banks. */
        for (int x = threadIdx.x; x < FKS * HD / 8; x += blockDim.x) {
            int a = x / (HD / 8), d = (x % (HD / 8)) * 8;
            float sk = ksb[a * NB + d / 32];
            uint4 hv;
            uint32_t *h4 = (uint32_t *)&hv;
            if (TQ) {
                float f[8];
                tq6x8(kbuf + a * LDBK, d, f);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    h4[i] = pack_h2(f[2 * i] * sk, f[2 * i + 1] * sk);
                }
            } else if (KQ8) {
                uint2 u = *(const uint2 *)(kbuf + a * LDBK + d);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t w = i < 2 ? u.x : u.y;
                    h4[i] = pack_h2(i8f(w, 16 * (i % 2)) * sk, i8f(w, 16 * (i % 2) + 8) * sk);
                }
            } else {
                uint4 u = *(const uint4 *)(kbuf + a * LDBK + 2 * d);
                const uint32_t *w4 = (const uint32_t *)&u;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    h4[i] = pack_h2(i16f(w4[i], 0) * sk, i16f(w4[i], 16) * sk);
                }
            }
            *(uint4 *)(kh + a * LD + d) = hv;
        }
        for (int x = threadIdx.x; x < FKS / 2 * (HD / 8); x += blockDim.x) {
            int a = (x % (FKS / 2)) * 2, d = (x / (FKS / 2)) * 8;
            float s0 = vsb[a * NB + d / 32], s1 = vsb[(a + 1) * NB + d / 32];
            if (TQ) {
                float f0[8], f1[8];
                tq6x8(vbuf + a * LDBV, d, f0);
                tq6x8(vbuf + (a + 1) * LDBV, d, f1);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(uint32_t *)(vt + (d + i) * VLD + a) = pack_h2(f0[i] * s0, f1[i] * s1);
                }
            } else if (VQ8) {
                uint2 u0 = *(const uint2 *)(vbuf + a * LDBV + d);
                uint2 u1 = *(const uint2 *)(vbuf + (a + 1) * LDBV + d);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(uint32_t *)(vt + (d + i) * VLD + a) =
                        pack_h2(i8f(i < 4 ? u0.x : u0.y, 8 * (i % 4)) * s0,
                                i8f(i < 4 ? u1.x : u1.y, 8 * (i % 4)) * s1);
                }
            } else {
                uint4 u0 = *(const uint4 *)(vbuf + a * LDBV + 2 * d);
                uint4 u1 = *(const uint4 *)(vbuf + (a + 1) * LDBV + 2 * d);
                const uint32_t *w0 = (const uint32_t *)&u0, *w1 = (const uint32_t *)&u1;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(uint32_t *)(vt + (d + i) * VLD + a) =
                        pack_h2(i16f(w0[i / 2], 16 * (i % 2)) * s0,
                                i16f(w1[i / 2], 16 * (i % 2)) * s1);
                }
            }
        }
        __syncthreads();
        if (s + 1 < nsteps) {
            load(k0 + FKS);
        }
        /* S: 16 queries by FKS keys = NT tiles of 8 keys, over the values of
         * this warp. */
        float sc[NT][4];
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
        }
        #pragma unroll
        for (int k2 = 0; k2 < 16; ++k2) {
            int kk = dsl * 256 + k2 * 16 + 2 * c;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const __half *kr = kh + (nt * 8 + g) * LD + kk;
                uint32_t b[2] = {*(const uint32_t *)kr, *(const uint32_t *)(kr + 8)};
                mma16816(sc[nt], qa[k2], b);
            }
        }
        if (NS == 2) {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                *(float4 *)(mine + 4 * nt) = make_float4(sc[nt][0], sc[nt][1], sc[nt][2],
                                                         sc[nt][3]);
            }
            __syncthreads();
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    float ot = other[nt * 4 + u];
                    sc[nt][u] = dsl == 0 ? sc[nt][u] + ot : ot + sc[nt][u];
                }
            }
        }
        /* The mask and the online softmax, as in k_flash_tc. */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int kk = nt * 8 + 2 * c + u;
                int64_t kp = k0 + kk;
                bool ok0 = live0 && kk < kn && kp <= e0 && (window == 0 || p0 - kp < window);
                bool ok1 = live1 && kk < kn && kp <= e1 && (window == 0 || p1 - kp < window);
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
        /* pa[kc]: the A fragment of P for keys 16 kc to 16 kc + 15 (tiles
         * 2 kc and 2 kc + 1). */
        uint32_t pa[KC][4];
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            float x0 = sc[nt][0] == -INFINITY ? 0.f : expf(sc[nt][0] - mx0);
            float x1 = sc[nt][1] == -INFINITY ? 0.f : expf(sc[nt][1] - mx0);
            float x2 = sc[nt][2] == -INFINITY ? 0.f : expf(sc[nt][2] - mx1);
            float x3 = sc[nt][3] == -INFINITY ? 0.f : expf(sc[nt][3] - mx1);
            ls0 += x0 + x1;
            ls1 += x2 + x3;
            pa[nt / 2][(nt % 2) * 2 + 0] = pack_h2(x0, x1);
            pa[nt / 2][(nt % 2) * 2 + 1] = pack_h2(x2, x3);
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
        /* O += P V: b0 = V[2c, 2c + 1][d], b1 = V[2c + 8, 2c + 9][d]. */
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            o[nt][0] *= f0;
            o[nt][1] *= f0;
            o[nt][2] *= f1;
            o[nt][3] *= f1;
            const __half *vr = vt + (dsl * 256 + nt * 8 + g) * VLD + 2 * c;
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) {
                uint32_t b[2] = {*(const uint32_t *)(vr + 16 * kc),
                                 *(const uint32_t *)(vr + 16 * kc + 8)};
                mma16816(o[nt], pa[kc], b);
            }
        }
    }
    float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        int d = dsl * 256 + nt * 8 + 2 * c;
        if (live0) {
            *(float2 *)&out[((size_t)(jq + g) * qh + h) * HD + d] =
                make_float2(o[nt][0] * inv0, o[nt][1] * inv0);
        }
        if (live1) {
            *(float2 *)&out[((size_t)(jq + g + 8) * qh + h) * HD + d] =
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
    PDL_START();
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
    PDL_START();
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

/* A test of MTP for the 26B (gg_set_reuse): a value 1 + m lets token j > 0
 * of a group select only from the experts of token 0, the experts that the
 * GPU holds (operand 14, the slot of each expert, -1 for a cold expert), and
 * its own m best experts. With m = 0 the CPU runs only the cold experts of
 * token 0. The result is not the result of the model. */
__device__ int g_reuse = 0;

__global__ void k_router_top_mt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, experts = DI(5), top_k = DI(6);
    const float *lg = DP(const float, 13) + (size_t)j * experts;
    /* Only a small group (an MTP verify group), not a prompt pass. */
    if (g_reuse && j > 0 && DI(11) <= 16 && experts <= 256 && top_k <= 16) {
        __shared__ float masked[256], v0[16], vj[16];
        __shared__ int sel0[16], selj[16];
        const int *slots = DP(const int, 14);
        int keep = min(g_reuse - 1, top_k);
        /* The selection of token 0, as block 0 makes it, and the own
         * selection of token j, best first. */
        router_top_warp(DP(const float, 13), DP(const float, 3), v0, sel0, experts, top_k);
        router_top_warp(lg, DP(const float, 3), vj, selj, experts, top_k);
        __syncwarp();
        for (int x = threadIdx.x; x < experts; x += 32) {
            bool ok = slots != NULL && slots[x] >= 0;
            for (int k = 0; k < top_k; ++k) {
                ok = ok || sel0[k] == x || (k < keep && selj[k] == x);
            }
            masked[x] = ok ? lg[x] : -INFINITY;
        }
        __syncwarp();
        lg = masked;
    }
    router_top_warp(lg, DP(const float, 3), DP(float, 9) + (size_t)j * top_k,
                    DP(int, 10) + (size_t)j * top_k, experts, top_k);
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
    PDL_START();
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
    /* An expert of -1: a pair of the CPU (the mixed groups; see
     * GP_HOT_SPLIT_MT). It has no place, and k_moe_sum skips it. */
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        if (idx[p] >= 0) {
            atomicAdd(&cnt[idx[p]], 1);
        }
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
        if (ex < 0) {
            pair_of[p] = -1;
            continue;
        }
        int q = off[ex] + atomicAdd(&fill[ex], 1);
        pair_tok[q] = p / top_k;
        pair_of[p] = q;
    }
}

__global__ void k_moe_tiles(const gp_rec *r, const int64_t *e)
{
    PDL_START();
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
    PDL_START();
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
    PDL_START();
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
    PDL_START();
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
    PDL_START();
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
        if (q >= 0) {
            acc += val[j * top_k + s2] * de[(size_t)q * dn_rows + c];
        }
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
    PDL_START();
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
    PDL_START();
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
    PDL_START();
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
 * these experts (GP_MOE_N of the CPU interpreter). With 2 top_k + 1 values,
 * cold_idx[top_k + 1 + j] also gets idx[j]: the host reads the selection of
 * the step for the cache of hot experts (ModelGPU, HotCache). */
__device__ void hot_split_body(const gp_rec *r, const int64_t *e)
{
    const int *idx = DP(const int, 0);
    const float *val = DP(const float, 1);
    const int *map = DP(const int, 2);
    int *cold = DP(int, 3);
    float *cold_val = DP(float, 4);
    int top_k = DI(5), n = 0;
    /* Operands 7 and 8 (or none): zc (top_k ints) and nzc. The first nzc
     * cold experts go to the GPU, which reads them from the pinned copy of
     * the experts in host memory (zc[j] = 1; k_kqh_nvx), not to the CPU. */
    int *zc = r->tag[7] == GP_T_NONE ? NULL : DP(int, 7);
    int nzc = zc != NULL ? DI(8) : 0, z = 0;
    for (int j = 0; j < top_k; ++j) {
        int to_gpu = 0;
        if (map[idx[j]] < 0) {
            if (z < nzc) {
                to_gpu = 1;
                ++z;
            } else {
                cold[n] = idx[j];
                cold_val[n] = val[j];
                ++n;
            }
        }
        if (zc != NULL) {
            zc[j] = to_gpu;
        }
    }
    cold[top_k] = n;
    if (DI(6)) {
        for (int j = 0; j < top_k; ++j) {
            cold[top_k + 1 + j] = idx[j];
        }
    }
}

__global__ void k_hot_split(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    hot_split_body(r, e);
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
/* A KQ_Q4X group (csrc/kquants.c; the int4 experts of the Gemma 4 26B in
 * groups of 16 rows: for each block of 32 columns, 8 steps of 32 bytes of
 * codes q = w + 8, then the float16 scales of the 16 rows) on float32 x.
 * Lane l of a warp reads 8 bytes of each block: step l / 4, rows 2q, 2q + 1
 * (low 4 bits) and 2q + 8, 2q + 9 (high), q = l % 4. */
__device__ __forceinline__ float q4x_dot4(uint32_t a, float4 x)
{
    return (float)(int8_t)(a & 255) * x.x + (float)(int8_t)((a >> 8) & 255) * x.y +
           (float)(int8_t)((a >> 16) & 255) * x.z + (float)(int8_t)(a >> 24) * x.w;
}

/* The 16 rows of a KQ_Q4X group by the ROWS_PER_BLOCK warps of a block (the
 * hot experts of a step): warp w takes the blocks of 32 columns w, w + 8,
 * ..., and the warps add their sums in shared memory. Then
 * thread i < 16 stores row i into out. A block has many more loads in
 * flight than one warp for each group. */
__device__ void q4x_group_block(const uint8_t *wg, const float *x, int cols, float *out)
{
    __shared__ float red[ROWS_PER_BLOCK][16];
    int w = threadIdx.x / 32, lane = threadIdx.x % 32, st = lane / 4, q = lane % 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 2
    for (int b = w; b < cols / 32; b += ROWS_PER_BLOCK) {
        const uint8_t *blk = wg + (size_t)b * 288;
        uint2 c = *(const uint2 *)(blk + 8 * lane);
        float4 xv = *(const float4 *)(x + 32 * b + 4 * st);
        const __half *d = (const __half *)(blk + 256);
        acc[0] += __half2float(d[2 * q]) * q4x_dot4(__vsub4(c.x & 0x0f0f0f0fu, 0x08080808u), xv);
        acc[1] += __half2float(d[2 * q + 1]) * q4x_dot4(__vsub4(c.y & 0x0f0f0f0fu, 0x08080808u), xv);
        acc[2] += __half2float(d[2 * q + 8]) *
                  q4x_dot4(__vsub4((c.x >> 4) & 0x0f0f0f0fu, 0x08080808u), xv);
        acc[3] += __half2float(d[2 * q + 9]) *
                  q4x_dot4(__vsub4((c.y >> 4) & 0x0f0f0f0fu, 0x08080808u), xv);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int o = 4; o < 32; o <<= 1) {
            acc[i] += __shfl_xor_sync(0xffffffff, acc[i], o);
        }
    }
    if (lane < 4) {
        red[w][2 * lane] = acc[0];
        red[w][2 * lane + 1] = acc[1];
        red[w][2 * lane + 8] = acc[2];
        red[w][2 * lane + 9] = acc[3];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float s = 0.f;
        #pragma unroll
        for (int k = 0; k < ROWS_PER_BLOCK; ++k) {
            s += red[k][threadIdx.x];
        }
        out[threadIdx.x] = s;
    }
}

__device__ __forceinline__ int hot_slot(const gp_rec *r, const int64_t *e, int j)
{
    return DP(const int, 3)[DP(const int, 2)[j]];
}

__global__ void k_hot_gu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.y, slot = hot_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int rows = DI(11), cols = DI(12);
    size_t rb = (size_t)(cols / 32) * 18;
    if (DI(16) == 1) {
        /* KQ_Q4X experts (operand 16): block x is a group of 16 rows */
        int grp = blockIdx.x;
        if (slot < 0 || grp >= rows / 16) {
            return;
        }
        q4x_group_block(DP(const uint8_t, 4) + ((size_t)slot * rows + 16 * (size_t)grp) * rb,
                        DP(const float, 0) + (size_t)(j / DI(10)) * cols, cols,
                        DP(float, 6) + (size_t)j * rows + 16 * grp);
        return;
    }
    if (slot < 0 || row >= rows) {
        return;
    }
    float v = int4_row(DP(const uint8_t, 4) + ((size_t)slot * rows + row) * rb,
                       DP(const float, 0) + (size_t)(j / DI(10)) * cols, cols);
    if (threadIdx.x % 32 == 0) {
        DP(float, 6)[(size_t)j * rows + row] = v;
    }
}

__global__ void k_hot_gelu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
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
    PDL_START();
    int j = blockIdx.y, slot = hot_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int rows = DI(13), inner = DI(14);
    size_t rb = (size_t)(inner / 32) * 18;
    if (DI(16) == 1) {
        int grp = blockIdx.x;
        if (slot < 0 || grp >= rows / 16) {
            return;
        }
        q4x_group_block(DP(const uint8_t, 5) + ((size_t)slot * rows + 16 * (size_t)grp) * rb,
                        DP(const float, 7) + (size_t)j * inner, inner,
                        DP(float, 8) + (size_t)j * rows + 16 * grp);
        return;
    }
    if (slot < 0 || row >= rows) {
        return;
    }
    float v = int4_row(DP(const uint8_t, 5) + ((size_t)slot * rows + row) * rb,
                       DP(const float, 7) + (size_t)j * inner, inner);
    if (threadIdx.x % 32 == 0) {
        DP(float, 8)[(size_t)j * rows + row] = v;
    }
}

__global__ void k_hot_sum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
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
 * (the CPU part), and GP_HOT_MOE skips the other pairs (the GPU part).
 * Operand 8 (or none): gpu_idx gets idx with -1 for the experts that the
 * GPU does not hold, for GP_MOE_GPU (the mixed groups of a prompt). */
__global__ void k_hot_split_mt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= DI(5)) {
        return;
    }
    int x = DP(const int, 0)[p];
    bool cold = DP(const int, 2)[x] < 0;
    /* With operands 6 and 7 (nreal, top_k): the pairs of the padding rows
     * go to neither part. */
    int k = r->tag[7] == GP_T_NONE ? 0 : DI(7);
    if (k > 0 && DI(6) > 0 && p >= DI(6) * k) {
        cold = false;
    }
    /* Operands 9 and 10 (or none): zc (a flag for each pair) and nzc. The
     * first nzc cold experts of the group (in the order of their first
     * pair) go to the GPU with all their pairs (k_kqh_nvx reads them from
     * the pinned copy in host memory), not to the CPU. */
    int to_gpu = 0;
    if (cold && r->tag[9] != GP_T_NONE && DI(10) > 0) {
        const int *ix = DP(const int, 0), *map = DP(const int, 2);
        int lim = (k > 0 && DI(6) > 0) ? DI(6) * k : DI(5);
        int first = p;
        for (int q = 0; q < p; ++q) {
            if (ix[q] == x) {
                first = q;
                break;
            }
        }
        int rank = 0;
        for (int q = 0; q < first && q < lim; ++q) {
            if (map[ix[q]] >= 0) {
                continue;
            }
            int dup = 0;
            for (int u = 0; u < q; ++u) {
                if (ix[u] == ix[q]) {
                    dup = 1;
                    break;
                }
            }
            rank += !dup;
        }
        to_gpu = rank < DI(10);
    }
    if (r->tag[9] != GP_T_NONE) {
        DP(int, 9)[p] = to_gpu;
    }
    if (to_gpu) {
        cold = false;
    }
    DP(int, 3)[p] = cold ? x : -1;
    DP(float, 4)[p] = cold ? DP(const float, 1)[p] : 0.f;
    if (r->tag[8] != GP_T_NONE && DP(int, 8) != NULL) {
        DP(int, 8)[p] = DP(const int, 2)[x] >= 0 ? x : -1;
    }
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
    PDL_START();
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
    PDL_START();
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
    PDL_START();
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
    PDL_START();
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
/* GP_SOFTCAP: x, n, cap. x = tanh(x / cap) cap in place (the logits). */
__global__ void k_softcap(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    float *x = DP(float, 0);
    size_t n = (size_t)di(r, e, 1), i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float cap = df(r, e, 2);
    if (i < n) {
        x[i] = tanhf(x[i] / cap) * cap;
    }
}

__global__ void k_argmax(const gp_rec *r, const int64_t *e)
{
    PDL_START();
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

/* The row tok[0] of a Q4_0 table (the token table of the target, the blocks
 * of its head on the GPU), times scale, into out (cols values): the input of
 * an MTP draft step. The values are those of the host (dequantize_int4 or
 * q4_0_rows, then the embedding scale): (q - 8) d, then times scale, in
 * float32. The drafts of a round then run with no copy to the host between
 * them. */
__global__ void k_embed_q4(const uint8_t *table, const int *tok, float *out, int cols, float scale)
{
    PDL_START();
    const uint8_t *row = table + (size_t)tok[0] * (cols / 32) * 18;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint8_t *blk = row + (size_t)(c / 32) * 18;
        int j = c % 32;
        int q = j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4;
        float d = __half2float(__ushort_as_half((uint16_t)(blk[0] | (blk[1] << 8))));
        out[c] = ((float)(q - 8) * d) * scale;
    }
}

/* k_embed_q4 for a block of rows: block j writes the row tok[j] of the
 * Q4_0 table, times scale, into out row j; a negative token writes zeros
 * (the padding of a group). The values are those of Model.embed. */
__global__ void k_embed_q4_rows(const uint8_t *table, const int *tok, float *out, int cols,
                                float scale)
{
    PDL_START();
    int t = tok[blockIdx.x];
    float *o = out + (size_t)blockIdx.x * cols;
    if (t < 0) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            o[c] = 0.f;
        }
        return;
    }
    const uint8_t *row = table + (size_t)t * (cols / 32) * 18;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint8_t *blk = row + (size_t)(c / 32) * 18;
        int j = c % 32;
        int q = j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4;
        float d = __half2float(__ushort_as_half((uint16_t)(blk[0] | (blk[1] << 8))));
        o[c] = ((float)(q - 8) * d) * scale;
    }
}

/* The best token of each row of x (rows of vocab values), as k_argmax: at an
 * equal value the lower index wins, as np.argmax. A block for each row. The
 * greedy pick of a step reads these, not the logits. */
__global__ void k_argmax_rows(const float *x, int vocab, int *out)
{
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float *row = x + (size_t)blockIdx.x * vocab;
    float v = -INFINITY;
    int b = vocab;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (row[i] > v) {
            v = row[i];
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
        out[blockIdx.x] = bi[0];
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
            wv[4 * n + 0] = d * (float)sc[is + 0] * (float)(((a & 15) | ((hq & 3) << 4)) - 32);
            wv[4 * n + 1] = d * (float)sc[is + 2] * (float)(((bq & 15) | (((hq >> 2) & 3) << 4)) - 32);
            wv[4 * n + 2] = d * (float)sc[is + 4] * (float)(((a >> 4) | (((hq >> 4) & 3) << 4)) - 32);
            wv[4 * n + 3] = d * (float)sc[is + 6] * (float)(((bq >> 4) | (((hq >> 6) & 3) << 4)) - 32);
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
                v += __shfl_xor_sync(0xffffffff, v, o);
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
                    v += __shfl_xor_sync(0xffffffff, v, o);
                }
                if (l == 0 && row0 + r < rows) {
                    out[(size_t)k * rows + row0 + r] = cap > 0.f ? cap * tanhf(v / cap) : v;
                }
            }
        }
    }
}

/* The candidates of sampling (np_gemma/sampling.py, Sampler.sample_sparse):
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
        float w = __shfl_xor_sync(0xffffffff, v, o);
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
            float w = __shfl_xor_sync(0xffffffff, v, o);
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

/* ---------- Qwen3.5 on the GGUF formats (QWEN_PLAN.md, phase 4) ----------
 * The records of compile_qwen_step (np_gemma/qwen.py). The products read x
 * in float32: GP_KQ_QUANT does nothing on the GPU, and GP_KQ_LINEAR reads
 * operand 3 (the float rows). The weights are the blocks of the GGUF file
 * (the ggml types, csrc/kquants.c). One warp computes one row. */
#define KQ_F32 0
/* Q4_0: blocks of 32 values, 18 bytes: d (float16), then 16 bytes (value j
 * in the low 4 bits of byte j, value j + 16 in the high 4 bits). A value is
 * d (q - 8). The QAT files of Gemma 4 keep their matrices in Q4_0; the E4B
 * takes them as GP_KQ_LINEAR on the GPU (int8 x, kq_rows_i8). */
#define KQ_Q4_0 2
#define KQ_Q8_0 8
#define KQ_Q4_K 12
#define KQ_Q5_K 13
#define KQ_Q6_K 14
/* Q5_1: blocks of 32 values, 24 bytes: d, m (float16), the high bits
 * (32 bits), and the low 4 bits (value j < 16 in the low half of byte j,
 * value j + 16 in its high half). A value is d q + m. */
#define KQ_Q5_1 7
/* bfloat16 rows, and NVFP4 in the rows of csrc/kquants.c (KQ_NV4): the E2M1
 * codes (cols / 2 bytes; 16 for each block of 32 values: value j low, j + 16
 * high), the E4M3 scales (2 for each block), the float32 scale of the
 * matrix, zeros to 16 bytes. The codes start at a multiple of 16 bytes: a
 * fragment of the tensor cores is one aligned load. */
#define KQ_BF16 30
#define KQ_NV4 51
/* BF12 (csrc/kquants.c, plan-scripts/BF12_PLAN.md): the bfloat16 values of
 * a row in 12.25 bits. The sign and the 7 bits of the mantissa of each value
 * (cols bytes), the gap of its exponent below the exponent of its group of 32
 * (4 bits: value j of the group in the low half of byte j of the group's 16,
 * value j + 16 in the high half; cols / 2 bytes), the exponent of each group
 * (cols / 32 bytes), zeros to 16 bytes. The code gap 15 with the byte 0x80
 * (a negative zero) is the zero. */
#define KQ_BF12 57
/* The KQ_BF12 matrices that the steps and the small groups (1 to 8 tokens)
 * run on the tensor cores (k_kq_bf12_tc; the kernel by bt_cls): a choice by
 * the shape alone, so a token alone and in a verify group take the same
 * kernel. */
#define BT_MIN_ROWS 1024
__host__ __device__ __forceinline__ int bt_shape(int type, int rows, int cols)
{
    return type == KQ_BF12 && cols % 32 == 0;
}
/* NVFP4 in groups of 16 rows (KQ_NVX of csrc/kquants.c): a group is 16
 * bytes (the float32 scale of the matrix, zeros), then 288 bytes for each
 * block of 32 columns: 8 steps of 32 bytes (step s: values 4s .. 4s + 3;
 * byte 4r + u: row r in the low 4 bits, row r + 8 in the high 4 bits) and
 * 32 E4M3 scales (values 0-15 of the 16 rows, then values 16-31). The 4
 * bytes at 32 s + 4 r are the fragments of the tensor cores for rows r and
 * r + 8. A "row" is cols / 32 * 18 + 1 bytes. */
#define KQ_NVX 53
#define KQ_NVX_BB 288
/* Q8_0 in rows for the GPU (np_gemma/qwen_gpu.py): the int8 values of the
 * row, then the float16 scale of each 32 values, then zeros to a multiple
 * of 16 bytes. Thus the values of each row start at a multiple of 16
 * bytes. */
#define KQ_Q8_R 100

__host__ __device__ __forceinline__ size_t kq_row_bytes(int type, int cols)
{
    return type == KQ_F32 ? (size_t)cols * 4 :
           type == KQ_Q8_R ? ((size_t)cols / 32 * 34 + 15) / 16 * 16 :
           type == KQ_Q8_0 ? (size_t)cols / 32 * 34 :
           type == KQ_Q4_0 ? (size_t)cols / 32 * 18 :
           type == KQ_Q5_1 ? (size_t)cols / 32 * 24 :
           type == KQ_BF16 ? (size_t)cols * 2 :
           type == KQ_BF12 ? ((size_t)cols + cols / 2 + cols / 32 + 15) / 16 * 16 :
           type == KQ_NV4 ? ((size_t)cols / 2 + (size_t)cols / 16 + 4 + 15) / 16 * 16 :
           type == KQ_NVX ? (size_t)cols / 32 * 18 + 1 :
           type == KQ_Q4_K ? (size_t)cols / 256 * 144 : type == KQ_Q5_K ? (size_t)cols / 256 * 176 :
           (size_t)cols / 256 * 210;
}

__device__ __forceinline__ float kq_half(const uint8_t *p)
{
    return __half2float(__ushort_as_half((uint16_t)(p[0] | (p[1] << 8))));
}

__device__ __forceinline__ float kq_e4m3(uint8_t b)
{
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? (float)m * (1.f / 512.f) : __uint_as_float(((uint32_t)(e + 120) << 23) | ((uint32_t)m << 20));
    return (b & 0x80) ? -v : v;
}

/* Twice the value of an E2M1 code. */
__device__ __forceinline__ float kq_e2m1x2(int c)
{
    /* the bytes of the constant: 0, 1, 2, 3, 4, 6, 8, 12 */
    float v = (float)((0x0C08060403020100ull >> (8 * (c & 7))) & 0xff);
    return (c & 8) ? -v : v;
}

/* 4 E2M1 codes (4 bytes of codes, the low 4 bits of each byte) to 4 int8
 * (twice the values), as the dequantization of Marlin: prmt looks up the 4
 * magnitudes in a table of 8 bytes (0 1 2 3 4 6 8 12) at once, and the sign
 * bits negate the bytes (v ^ m) - m. */
__device__ __forceinline__ uint32_t kt_e2m1x4(uint32_t q)
{
    uint32_t c = q & 0x0f0f0f0fu;
    uint32_t i = c & 0x07070707u;
    uint32_t t = i | (i >> 4);                       /* bytes 0, 2: 2 indices of 4 bits */
    uint32_t sel = (t & 0xffu) | ((t >> 8) & 0xff00u);
    uint32_t mag = __byte_perm(0x03020100u, 0x0c080604u, sel);
    uint32_t m = ((c >> 3) & 0x01010101u) * 0xffu;
    return __vsub4(mag ^ m, m);
}

/* An E4M3 scale times 2^-8, with no branch: its bits go to float16 as they
 * are (the exponent bias of float16 is 8 more). */
__device__ __forceinline__ float kq_e4m3s(uint32_t b)
{
    return __half2float(__ushort_as_half((uint16_t)(((b & 0x7f) << 7) | ((b & 0x80) << 8))));
}

/* The sum of 4 int8 (a) times 4 floats. */
__device__ __forceinline__ float kq_dot4s8(uint32_t a, float4 x)
{
    return (float)(int8_t)(a & 255) * x.x + (float)(int8_t)((a >> 8) & 255) * x.y +
           (float)(int8_t)((a >> 16) & 255) * x.z + (float)(int8_t)(a >> 24) * x.w;
}

/* The 16 rows of the KQ_NVX group at wg with x (cols values), by a warp.
 * Lane l reads 8 bytes of each block: step l / 4, rows 2q, 2q + 1 (low 4
 * bits) and 2q + 8, 2q + 9 (high), q = l % 4. The lanes of each q add
 * their sums; lane q < 4 gets rows 2q, 2q + 1, 2q + 8, 2q + 9 in v. */
__device__ void kq_nvx_group(const uint8_t *wg, const float *x, int cols, float *v)
{
    int lane = threadIdx.x % 32, st = lane / 4, q = lane % 4, h = st / 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < cols / 32; ++b) {
        const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
        uint2 c = *(const uint2 *)(blk + 8 * lane);
        float4 xv = *(const float4 *)(x + 32 * b + 4 * st);
        uint32_t s01 = *(const uint16_t *)(blk + 256 + 16 * h + 2 * q);
        uint32_t s89 = *(const uint16_t *)(blk + 256 + 16 * h + 8 + 2 * q);
        acc[0] += kq_e4m3s(s01 & 255) * kq_dot4s8(kt_e2m1x4(c.x), xv);
        acc[1] += kq_e4m3s(s01 >> 8) * kq_dot4s8(kt_e2m1x4(c.y), xv);
        acc[2] += kq_e4m3s(s89 & 255) * kq_dot4s8(kt_e2m1x4(c.x >> 4), xv);
        acc[3] += kq_e4m3s(s89 >> 8) * kq_dot4s8(kt_e2m1x4(c.y >> 4), xv);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int o = 4; o < 32; o <<= 1) {
            acc[i] += __shfl_xor_sync(0xffffffff, acc[i], o);
        }
    }
    float gs = 128.f * *(const float *)wg;           /* 0.5 g, and the 2^8 of kq_e4m3s */
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = gs * acc[i];
    }
}

/* 8 values of row rin of the KQ_NVX group at wg from column c0 (a multiple
 * of 8), as float32. */
__device__ void kq_dequant8_nvx(const uint8_t *wg, int rin, int c0, float *v)
{
    const uint8_t *blk = wg + 16 + (size_t)(c0 / 32) * KQ_NVX_BB;
    int j = c0 % 32, sh = rin < 8 ? 0 : 4;
    float sc = 128.f * *(const float *)wg * kq_e4m3s(blk[256 + 16 * (j / 16) + rin]);
    uint32_t a = kt_e2m1x4(*(const uint32_t *)(blk + 32 * (j / 4) + 4 * (rin % 8)) >> sh);
    uint32_t b = kt_e2m1x4(*(const uint32_t *)(blk + 32 * (j / 4 + 1) + 4 * (rin % 8)) >> sh);
    for (int u = 0; u < 4; ++u) {
        v[u] = sc * (float)(int8_t)((a >> (8 * u)) & 255);
        v[4 + u] = sc * (float)(int8_t)((b >> (8 * u)) & 255);
    }
}

__device__ __forceinline__ float kq_bf16(uint16_t h)
{
    return __uint_as_float((uint32_t)h << 16);
}

/* The bfloat16 bits of n (8 or 16) values of a KQ_BF12 row from column c
 * (a multiple of n, in one half of a group of 32). */
template <int N>
__device__ __forceinline__ void kq_bf12_bits(const uint8_t *w, int cols, int c, uint32_t *bits)
{
    int g = c >> 5, h = (c >> 4) & 1;
    uint32_t lw[N / 4], hw[N / 4];
    #pragma unroll
    for (int k = 0; k < N / 4; ++k) {
        lw[k] = ((const uint32_t *)(w + c))[k];
        hw[k] = ((const uint32_t *)(w + cols + 16 * g + (c & 15)))[k];
    }
    uint32_t E2 = ((uint32_t)w[cols + cols / 2 + g] | 256u) * 0x00010001u;
    #pragma unroll
    for (int k = 0; k < N / 4; ++k) {
        uint32_t p0 = bf12_frag(lw[k], hw[k], 4 * h, E2), p1 = bf12_frag(lw[k] >> 16, hw[k] >> 16, 4 * h, E2);
        bits[4 * k] = p0 & 0xffff;
        bits[4 * k + 1] = p0 >> 16;
        bits[4 * k + 2] = p1 & 0xffff;
        bits[4 * k + 3] = p1 >> 16;
    }
}

/* The decode of the steps (kq_bf12_dec16) is bound by the instructions:
 * the 3090 at its power cap runs a memory-bound kernel at 500-900 MHz, and
 * Ampere runs the integer ones on one of its two datapaths. (The exponent
 * by an IMAD a value, on the FMA pipe, with a PRMT and a LOP3 for each
 * value, was not faster: 0.97-1.09 times bfloat16 for a token, against
 * 0.87-0.94 here.)
 * bf12_pair makes two values (the 16-bit lanes of a word: b the bytes, g
 * the gaps) in 4 instructions with no test of the zero code; E7 is
 * (E + 256) << 7 in each lane (no borrow out of a lane). The sign and the
 * mantissa: (b & 0x80) * 0xff + b = (b & 0x80) << 8 | (b & 0x7f). */
__device__ __forceinline__ uint32_t bf12_pair(uint32_t b, uint32_t g, uint32_t E7)
{
    uint32_t ex7 = (E7 - (g << 7)) & 0x7f807f80u;
    return ((b & 0x00800080u) * 0xffu + b) | ex7;
}

/* The bfloat16 bits of the 16 values of half h of a group of a KQ_BF12 row
 * from its words L (the 16 lo bytes), H (the 16 hi bytes of the group), and
 * E, as 8 words of two values (value 2i in the low lane of p[i]). A zero
 * code (b 0x80 with gap 15) is found 4 values at a time (a zero byte of
 * b ^ 0x80 | g ^ 15); a word with one takes the exact test (rare). */
__device__ __forceinline__ void kq_bf12_dec16(uint4 L, uint4 H, int E, int h, uint32_t *p)
{
    uint32_t E7 = (((uint32_t)E | 256u) << 7) * 0x00010001u;
    uint32_t lw[4] = {L.x, L.y, L.z, L.w}, hw[4] = {H.x, H.y, H.z, H.w};
    uint32_t any = 0, g4[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        g4[k] = (hw[k] >> (4 * h)) & 0x0f0f0f0fu;
        uint32_t zb = (lw[k] ^ 0x80808080u) | (g4[k] ^ 0x0f0f0f0fu);
        any |= (zb - 0x01010101u) & ~zb & 0x80808080u;
        p[2 * k] = bf12_pair(__byte_perm(lw[k], 0, 0x4140), __byte_perm(g4[k], 0, 0x4140), E7);
        p[2 * k + 1] = bf12_pair(__byte_perm(lw[k], 0, 0x4342), __byte_perm(g4[k], 0, 0x4342), E7);
    }
    if (any) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t c0 = __byte_perm(lw[k], 0, 0x4140) | __byte_perm(g4[k], 0, 0x4140) << 8;
            uint32_t c1 = __byte_perm(lw[k], 0, 0x4342) | __byte_perm(g4[k], 0, 0x4342) << 8;
            p[2 * k] &= ~__vcmpeq2(c0, 0x0f800f80u);
            p[2 * k + 1] &= ~__vcmpeq2(c1, 0x0f800f80u);
        }
    }
}

/* The sums of the 16 values of a KQ_BF12 row from column c (8 words of two
 * bfloat16 values) with the rows of x of NT tokens (t of them live): the
 * same order for each token, so a token alone (kq_row_part) and in a group
 * get the same bits. */
template <int NT>
__device__ __forceinline__ void kq_bf12_fma16(const uint32_t *p, const float *x, int cols, int c,
                                              int t, float *sum)
{
    float wv[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        wv[2 * i] = __uint_as_float(p[i] << 16);
        wv[2 * i + 1] = __uint_as_float(p[i] & 0xffff0000u);
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        if (j < t) {
            const float4 *x4 = (const float4 *)(x + (size_t)j * cols + c);
            float s = sum[j];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 a = x4[i];
                s = fmaf(wv[4 * i], a.x, s);
                s = fmaf(wv[4 * i + 1], a.y, s);
                s = fmaf(wv[4 * i + 2], a.z, s);
                s = fmaf(wv[4 * i + 3], a.w, s);
            }
            sum[j] = s;
        }
    }
}

/* The sums of part part of nparts of a KQ_BF12 row with x of NT tokens:
 * lane l takes 16 values (one half of a group of 32) of each part of 512
 * columns. The words of BF12_U parts are loaded before any is decoded (4
 * spilled in the 64 registers of k_kq_linear_bf12<1>). */
#define BF12_U 2
template <int NT>
__device__ __forceinline__ void kq_bf12_row(const uint8_t *w, const float *x, int cols, int t, int part,
                                            int nparts, float *sum)
{
    int lane = threadIdx.x % 32, h = lane & 1;
    for (int c0 = 16 * lane + 512 * part; c0 < cols; c0 += BF12_U * 512 * nparts) {
        uint4 L[BF12_U], H[BF12_U];
        int E[BF12_U];
        #pragma unroll
        for (int u = 0; u < BF12_U; ++u) {
            int c = c0 + u * 512 * nparts;
            if (c < cols) {
                L[u] = *(const uint4 *)(w + c);
                H[u] = *(const uint4 *)(w + cols + 16 * (c >> 5));
                E[u] = w[cols + cols / 2 + (c >> 5)];
            }
        }
        #pragma unroll
        for (int u = 0; u < BF12_U; ++u) {
            int c = c0 + u * 512 * nparts;
            if (c < cols) {
                uint32_t pw[8];
                kq_bf12_dec16(L[u], H[u], E[u], h, pw);
                kq_bf12_fma16<NT>(pw, x, cols, c, t, sum);
            }
        }
    }
}

__device__ __forceinline__ void kq_scale_min(const uint8_t *q, int j, int *sc, int *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float kq_dot4(uint32_t q, float4 x)
{
    return (float)(q & 255) * x.x + (float)((q >> 8) & 255) * x.y +
           (float)((q >> 16) & 255) * x.z + (float)(q >> 24) * x.w;
}

/* The product of one row with x (cols values), for part part of nparts of
 * the blocks of the row (part 0 of 1: the whole row). All the lanes of the
 * warp take part; each gets the sum. The parts let several warps share a
 * long row, so more loads are in flight (k_kq_linear_split). */
__device__ float kq_row_part(int type, const uint8_t *w, const float *x, int cols, int part,
                            int nparts)
{
    int lane = threadIdx.x % 32;
    float sum = 0.f;
    if (type == KQ_F32) {
        const float4 *w4 = (const float4 *)w, *x4 = (const float4 *)x;
        for (int k = lane + 32 * part; k < cols / 4; k += 32 * nparts) {
            float4 a = w4[k], b = x4[k];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else if (type == KQ_Q8_R) {
        /* lane l: 16 values (one 16-byte load) of each part of 512 */
        const __half *d = (const __half *)(w + cols);
        for (int c = 16 * lane + 512 * part; c < cols; c += 512 * nparts) {
            uint4 q = *(const uint4 *)(w + c);
            const float4 *x4 = (const float4 *)(x + c);
            uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            float acc = 0.f;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float4 xv = x4[u];
                acc += (float)(int8_t)(qq[u] & 255) * xv.x + (float)(int8_t)((qq[u] >> 8) & 255) * xv.y +
                       (float)(int8_t)((qq[u] >> 16) & 255) * xv.z + (float)(int8_t)(qq[u] >> 24) * xv.w;
            }
            sum += __half2float(d[c / 32]) * acc;
        }
    } else if (type == KQ_Q8_0) {
        /* 4 blocks for each pass: lane l takes 4 values of block l / 8. */
        int o = 4 * (lane % 8);
        for (int b = lane / 8 + 4 * part; b < cols / 32; b += 4 * nparts) {
            const uint8_t *blk = w + (size_t)b * 34;
            const uint16_t *q = (const uint16_t *)(blk + 2 + o);
            uint32_t a = q[0], c = q[1];
            float4 xv = *(const float4 *)(x + b * 32 + o);
            sum += kq_half(blk) * ((float)(int8_t)(a & 255) * xv.x +
                                   (float)(int8_t)(a >> 8) * xv.y +
                                   (float)(int8_t)(c & 255) * xv.z +
                                   (float)(int8_t)(c >> 8) * xv.w);
        }
    } else if (type == KQ_Q4_0) {
        /* As Q8_0: lane l takes 4 values of block l / 8 of 4 blocks (the
         * low 4 bits of 4 bytes, or their high 4 bits). */
        int o = 4 * (lane % 8), sh = o < 16 ? 0 : 4;
        for (int b = lane / 8 + 4 * part; b < cols / 32; b += 4 * nparts) {
            const uint8_t *blk = w + (size_t)b * 18;
            const uint16_t *q = (const uint16_t *)(blk + 2 + o % 16);
            uint32_t a = (uint32_t)q[0] | ((uint32_t)q[1] << 16);
            float4 xv = *(const float4 *)(x + b * 32 + o);
            sum += kq_half(blk) * ((float)((int)((a >> sh) & 15) - 8) * xv.x +
                                   (float)((int)((a >> (8 + sh)) & 15) - 8) * xv.y +
                                   (float)((int)((a >> (16 + sh)) & 15) - 8) * xv.z +
                                   (float)((int)((a >> (24 + sh)) & 15) - 8) * xv.w);
        }
    } else if (type == KQ_BF16) {
        const uint16_t *h = (const uint16_t *)w;
        for (int c = 8 * lane + 256 * part; c < cols; c += 256 * nparts) {
            uint4 q = *(const uint4 *)(h + c);
            uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            const float4 *x4 = (const float4 *)(x + c);
            float4 a = x4[0], b = x4[1];
            float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                sum += kq_bf16((uint16_t)(qq[u] & 0xffff)) * xv[2 * u] +
                       kq_bf16((uint16_t)(qq[u] >> 16)) * xv[2 * u + 1];
            }
        }
    } else if (type == KQ_BF12) {
        /* lane l: 16 values (one half of a group of 32) of each part of 512 */
        kq_bf12_row<1>(w, x, cols, 1, part, nparts, &sum);
    } else if (type == KQ_NV4) {
        /* As Q8_0: lane l takes 4 values of block l / 8 of 4 blocks. */
        float g = *(const float *)(w + cols / 2 + cols / 16);
        int o = 4 * (lane % 8);
        for (int b = lane / 8 + 4 * part; b < cols / 32; b += 4 * nparts) {
            float sc = 0.5f * g * kq_e4m3(w[cols / 2 + 2 * b + (o < 16 ? 0 : 1)]);
            const uint8_t *q = w + 16 * b + (o % 16);
            int sh = o < 16 ? 0 : 4;
            float4 xv = *(const float4 *)(x + b * 32 + o);
            sum += sc * (kq_e2m1x2((q[0] >> sh) & 15) * xv.x + kq_e2m1x2((q[1] >> sh) & 15) * xv.y +
                         kq_e2m1x2((q[2] >> sh) & 15) * xv.z + kq_e2m1x2((q[3] >> sh) & 15) * xv.w);
        }
    } else if (type == KQ_Q5_1) {
        /* As Q8_0: lane l takes 4 values of block l / 8 of 4 blocks. */
        int o = 4 * (lane % 8);
        for (int b = lane / 8 + 4 * part; b < cols / 32; b += 4 * nparts) {
            const uint8_t *blk = w + (size_t)b * 24;
            uint32_t qh = *(const uint32_t *)(blk + 4);
            uint32_t qs = *(const uint32_t *)(blk + 8 + o % 16);
            uint32_t q = o < 16 ? qs & 0x0f0f0f0fu : (qs >> 4) & 0x0f0f0f0fu;
            uint32_t h = qh >> o;
            q |= ((h & 1) << 4) | ((h & 2) << 11) | ((h & 4) << 18) | ((h & 8) << 25);
            float4 xv = *(const float4 *)(x + b * 32 + o);
            sum += kq_half(blk) * kq_dot4(q, xv) +
                   kq_half(blk + 2) * (xv.x + xv.y + xv.z + xv.w);
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        /* lane l: 4 bytes of the part pair c = l / 8: 4 values of part 2c
         * (the low 4 bits) and 4 of part 2c + 1 (the high 4 bits). */
        int five = type == KQ_Q5_K, c = lane / 8, sub = lane % 8;
        size_t bs = five ? 176 : 144;
        for (int b = part; b < cols / 256; b += nparts) {
            const uint8_t *blk = w + (size_t)b * bs;
            const uint8_t *qs = blk + (five ? 48 : 16);
            uint32_t q = *(const uint32_t *)(qs + 32 * c + 4 * sub);
            uint32_t lo = q & 0x0f0f0f0fu, hi = (q >> 4) & 0x0f0f0f0fu;
            if (five) {
                uint32_t h = *(const uint32_t *)(blk + 16 + 4 * sub);
                lo |= ((h >> (2 * c)) & 0x01010101u) << 4;
                hi |= ((h >> (2 * c + 1)) & 0x01010101u) << 4;
            }
            int s0, m0, s1, m1;
            kq_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            float d = kq_half(blk), dm = kq_half(blk + 2);
            const float *xb = x + b * 256 + 64 * c + 4 * sub;
            float4 xl = *(const float4 *)xb, xh = *(const float4 *)(xb + 32);
            float sl = xl.x + xl.y + xl.z + xl.w, sh = xh.x + xh.y + xh.z + xh.w;
            sum += d * ((float)s0 * kq_dot4(lo, xl) + (float)s1 * kq_dot4(hi, xh)) -
                   dm * ((float)m0 * sl + (float)m1 * sh);
        }
    } else {
        /* Q6_K: lane l takes the values l + 32 u of each half (as
         * k_q6k_head). */
        for (int b = part; b < cols / 256; b += nparts) {
            const uint8_t *ql = w + (size_t)b * 210;
            const uint8_t *qh = ql + 128;
            const int8_t *sc = (const int8_t *)(ql + 192);
            float d = kq_half(ql + 208);
            float acc = 0.f;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                int a = ql[64 * n + lane], bq = ql[64 * n + lane + 32], hq = qh[32 * n + lane];
                int is = 8 * n + lane / 16;
                const float *xb = x + b * 256 + 128 * n + lane;
                acc += (float)sc[is + 0] * (float)(((a & 15) | ((hq & 3) << 4)) - 32) * xb[0];
                acc += (float)sc[is + 2] * (float)(((bq & 15) | (((hq >> 2) & 3) << 4)) - 32) * xb[32];
                acc += (float)sc[is + 4] * (float)(((a >> 4) | (((hq >> 4) & 3) << 4)) - 32) * xb[64];
                acc += (float)sc[is + 6] * (float)(((bq >> 4) | (((hq >> 6) & 3) << 4)) - 32) * xb[96];
            }
            sum += d * acc;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, o);
    }
    return sum;
}

__device__ __forceinline__ float kq_row(int type, const uint8_t *w, const float *x, int cols)
{
    return kq_row_part(type, w, x, cols, 0, 1);
}

/* The rows of a block of the products of one token (4 was 3% faster than 8,
 * and 16 or 32 were slower). */
#define KQ_RPB 4

/* GP_KQ_LINEAR of a few tokens with a block of KQ_SPLIT warps for each row:
 * warp w takes the blocks w, w + KQ_SPLIT, ... of the row (kq_row_part), and
 * the warps add their sums in shared memory. A long row (the down matrix of
 * the E4B has 40 blocks of 256) then has KQ_SPLIT times the loads in flight
 * of one warp. Only for rows of 8192 values or more: a short row (10 blocks
 * of 256) was slower (331 to 227 GB/s), a long one faster (313 to 334). */
#define KQ_SPLIT 4
__global__ void __launch_bounds__(32 * KQ_SPLIT) k_kq_linear_split(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float red[KQ_SPLIT][MT_MAX];
    int row = blockIdx.x, w = threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), type = DI(5), t = DI(8);
    const uint8_t *wr = DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(type, cols);
    for (int j = 0; j < t; ++j) {
        float v = kq_row_part(type, wr, DP(const float, 3) + (size_t)j * cols, cols, w, KQ_SPLIT);
        if (threadIdx.x % 32 == 0) {
            red[w][j] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < t) {
        float s = 0.f;
        #pragma unroll
        for (int k = 0; k < KQ_SPLIT; ++k) {
            s += red[k][threadIdx.x];
        }
        DP(float, 9)[(size_t)threadIdx.x * rows + row] = s;
    }
}

/* NP_GEMMA_GPU_KQ_SPLIT=0 keeps k_kq_linear (a warp for each row), for a
 * test. */
static int kq_split_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_KQ_SPLIT");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* GP_KQ_LINEAR: xq, xs, xm, x, w, type, rows, cols, t, out. */
__global__ void k_kq_linear(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), type = DI(5), t = DI(8);
    if (row >= rows) {
        return;
    }
    const uint8_t *w = DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(type, cols);
    for (int j = 0; j < t; ++j) {
        float v = kq_row(type, w, DP(const float, 3) + (size_t)j * cols, cols);
        if (threadIdx.x % 32 == 0) {
            DP(float, 9)[(size_t)j * rows + row] = v;
        }
    }
}

__device__ __forceinline__ float qw_silu(float v)
{
    return v / (1.f + expf(-v));
}

/* GP_SIGMUL: x, g, out, n. out = x sigmoid(g). */
__global__ void k_sigmul(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < di(r, e, 3)) {
        DP(float, 2)[i] = DP(const float, 0)[i] / (1.f + expf(-DP(const float, 1)[i]));
    }
}

/* The largest value of the block and its index (the smallest index of
 * equal values). All threads get the result. */
__device__ void block_argmax(float v, int i, float *bv, int *bi)
{
    __shared__ float pv[32];
    __shared__ int pi[32];
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffff, v, o);
        int i2 = __shfl_xor_sync(0xffffffff, i, o);
        if (v2 > v || (v2 == v && i2 < i)) {
            v = v2;
            i = i2;
        }
    }
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) {
        pv[w] = v;
        pi[w] = i;
    }
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    v = lane < nw ? pv[lane] : -INFINITY;
    i = lane < nw ? pi[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(0xffffffff, v, o);
        int i2 = __shfl_xor_sync(0xffffffff, i, o);
        if (v2 > v || (v2 == v && i2 < i)) {
            v = v2;
            i = i2;
        }
    }
    *bv = v;
    *bi = i;
}

/* GP_ROUTER_TOPK: logits, t, experts, k, val, idx. The softmax of each row,
 * the k largest, and their weights divided by their sum. One block of 256
 * threads for each token; at most 1024 experts. */
/* r2 (or null): a GP_HOT_SPLIT of the selection of this one token, run at
 * the end by thread 0 (one kernel in place of two: NP_GEMMA_GPU_TOPK_SPLIT). */
__device__ void hot_split_body(const gp_rec *r, const int64_t *e);
__global__ void k_router_topk(const gp_rec *r, const int64_t *e, const gp_rec *r2 = NULL)
{
    PDL_START();
    int j = blockIdx.x, E = DI(2), k = DI(3);
    const float *l = DP(const float, 0) + (size_t)j * E;
    float p[4];
    float m = -INFINITY;
    for (int u = 0; u < 4; ++u) {
        int x = threadIdx.x + 256 * u;
        p[u] = x < E ? l[x] : -INFINITY;
        m = fmaxf(m, p[u]);
    }
    m = block_max(m);
    float s = 0.f;
    for (int u = 0; u < 4; ++u) {
        int x = threadIdx.x + 256 * u;
        p[u] = x < E ? expf(p[u] - m) : -1.f;
        s += x < E ? p[u] : 0.f;
    }
    s = block_sum(s);
    float vs = 0.f;
    float *val = DP(float, 4) + (size_t)j * k;
    int *idx = DP(int, 5) + (size_t)j * k;
    for (int s2 = 0; s2 < k; ++s2) {
        float bv = -1.f;
        int bi = 0x7fffffff;
        for (int u = 0; u < 4; ++u) {
            int x = threadIdx.x + 256 * u;
            if (p[u] >= 0.f && (p[u] > bv || (p[u] == bv && x < bi))) {
                bv = p[u];
                bi = x;
            }
        }
        block_argmax(bv, bi, &bv, &bi);
        if (threadIdx.x == 0) {
            idx[s2] = bi;
            val[s2] = bv / s;
        }
        vs += bv / s;
        if (bi % 256 == (int)threadIdx.x) {
            p[bi / 256] = -1.f;
        }
    }
    __syncthreads();
    if (threadIdx.x < (unsigned)k) {
        val[threadIdx.x] /= vs;
    }
    if (r2 != NULL) {
        __syncthreads();
        if (threadIdx.x == 0) {
            hot_split_body(r2, e);
        }
    }
}

/* GP_ATTN_PREP: qg, kk, vv, qn, kn, cos, sin, K, V, hs, pos, t, nq, nk, hd, rot, eps,
 * scale, qout, gate, kout (gp_attn_prep_body of the CPU; with a null K the
 * key goes to kout, for GP_KV_WRITE). One block of hd threads
 * for each (token, head). */
__global__ void k_attn_prep(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float ys[1024];
    int nq = DI(12), nk = DI(13), hd = DI(14), rot = DI(15), half = rot / 2;
    int j = blockIdx.x / (nq + nk), h = blockIdx.x % (nq + nk), d = threadIdx.x;
    int64_t hs = di(r, e, 9), pos = di(r, e, 10);
    const float *src, *w;
    float *dst, sc;
    if (h < nq) {
        src = DP(const float, 0) + ((size_t)j * nq + h) * 2 * hd;
        DP(float, 19)[((size_t)j * nq + h) * hd + d] = src[hd + d];
        dst = DP(float, 18) + ((size_t)j * nq + h) * hd;
        w = DP(const float, 3);
        sc = df(r, e, 17);
    } else {
        int kh = h - nq;
        src = DP(const float, 1) + ((size_t)j * nk + kh) * hd;
        if (DP(float, 7) == NULL) {
            dst = DP(float, 20) + ((size_t)j * nk + kh) * hd;
        } else {
            size_t rs = kv_rs((size_t)hs, nk, hd);
            dst = DP(float, 7) + (size_t)kh * hs + (size_t)(pos + j) * rs;
            DP(float, 8)[(size_t)kh * hs + (size_t)(pos + j) * rs + d] =
                DP(const float, 2)[((size_t)j * nk + kh) * hd + d];
        }
        w = DP(const float, 4);
        sc = 1.f;
    }
    float v = src[d];
    float ss = block_sum(v * v);
    float y = v * (1.f / sqrtf(ss / (float)hd + df(r, e, 16))) * w[d];
    ys[d] = y;
    __syncthreads();
    const float *c = DP(const float, 5) + (size_t)j * rot, *sn = DP(const float, 6) + (size_t)j * rot;
    if (d < half) {
        dst[d] = (ys[d] * c[d] - ys[d + half] * sn[d]) * sc;
    } else if (d < rot) {
        dst[d] = (ys[d] * c[d] + ys[d - half] * sn[d]) * sc;
    } else {
        dst[d] = y * sc;
    }
}

/* GP_GDN (gdn_body of csrc/deltanet.c): qkv, conv, conv_w, kernel, z, a, b,
 * A_log, dt_bias, norm_w, S, out, scratch, t, k_heads, v_heads, k_dim,
 * v_dim, eps, log, flags (1: tiled heads; 2: a sigmoid gate of the norm), nreal.
 *
 * Only the first nreal tokens change the state (0: all t); the other rows
 * pad a group to the size of its program. With a log (an MTP verify group)
 * conv and S do not change: the log gets, for each token, the input of the
 * convolution, and for each head k, the delta, and the decay, as on the CPU.
 * gg_gdn_commit applies the first n tokens.
 *
 * k_gdn_conv: the convolution of each channel, the tokens in order, into the
 * scratch; the last inputs go back to conv. */
__device__ __forceinline__ int gdn_nreal(const gp_rec *r, const int64_t *e)
{
    int t = DI(13), n = r->tag[21] == GP_T_NONE ? 0 : DI(21);
    return n > 0 && n < t ? n : t;
}

__global__ void k_gdn_conv(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int kernel = DI(3), t = DI(13);
    int cd = 2 * DI(14) * DI(16) + DI(15) * DI(17);
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cd) {
        return;
    }
    const float *qkv = DP(const float, 0);
    float *conv = DP(float, 1), *cv = DP(float, 12), *log = DP(float, 19);
    const float *w = DP(const float, 2) + (size_t)c * kernel;
    size_t lrow = (size_t)cd + (size_t)DI(15) * (DI(16) + DI(17) + 1);
    float hist[8];
    for (int j = 0; j < kernel - 1; ++j) {
        hist[j] = conv[(size_t)j * cd + c];
    }
    t = gdn_nreal(r, e);
    for (int i = 0; i < t; ++i) {
        if (log != NULL) {
            log[(size_t)i * lrow + c] = qkv[(size_t)i * cd + c];
        }
        float xin = qkv[(size_t)i * cd + c];
        float v = w[kernel - 1] * xin;
        for (int j = 0; j < kernel - 1; ++j) {
            v += w[j] * hist[j];
        }
        cv[(size_t)i * cd + c] = qw_silu(v);
        for (int j = 0; j < kernel - 2; ++j) {
            hist[j] = hist[j + 1];
        }
        hist[kernel - 2] = xin;
    }
    if (log == NULL) {
        for (int j = 0; j < kernel - 1; ++j) {
            conv[(size_t)j * cd + c] = hist[j];
        }
    }
}

/* k_gdn_heads: one block for each value head; thread e keeps column e of the
 * state of the head (KD values) in registers. KD = k_dim = v_dim = the
 * threads of the block. */
template <int KD>
__global__ void __launch_bounds__(KD) k_gdn_heads(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float qsh[KD], ksh[KD];
    int kh = DI(14), vh = DI(15), t = gdn_nreal(r, e), flags = DI(20), tiled = flags & 1;
    int kd = kh * KD, vd = vh * KD, cd = 2 * kd + vd;
    float *log = DP(float, 19);
    size_t lrow = (size_t)cd + (size_t)vh * (2 * KD + 1);
    int hv = blockIdx.x, c = threadIdx.x;
    int hk = tiled ? hv % kh : hv / (vh / kh);
    float *Sh = DP(float, 10) + (size_t)hv * KD * KD;
    float S[KD];
    #pragma unroll
    for (int d = 0; d < KD; ++d) {
        S[d] = Sh[(size_t)d * KD + c];
    }
    const float *cv = DP(const float, 12);
    float Aexp = expf(DP(const float, 7)[hv]), dtb = DP(const float, 8)[hv];
    float eps = df(r, e, 18);
    for (int i = 0; i < t; ++i) {
        const float *row = cv + (size_t)i * cd;
        float qv = row[hk * KD + c], kv = row[kd + hk * KD + c];
        float nq = block_sum(qv * qv), nk = block_sum(kv * kv);
        qsh[c] = qv * (1.f / sqrtf(nq + 1e-6f) / sqrtf((float)KD));
        ksh[c] = kv * (1.f / sqrtf(nk + 1e-6f));
        __syncthreads();
        float av = DP(const float, 5)[(size_t)i * vh + hv] + dtb;
        float sp = av > 20.f ? av : log1pf(expf(av));
        float decay = expf(-Aexp * sp);
        float beta = 1.f / (1.f + expf(-DP(const float, 6)[(size_t)i * vh + hv]));
        /* The updates of S use explicit roundings, as gg_gdn_commit does, so
         * a commit gives the state of plain steps. */
        float s1 = 0.f;
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            S[d] = __fmul_rn(S[d], decay);
            s1 += ksh[d] * S[d];
        }
        float delta = (row[2 * kd + hv * KD + c] - s1) * beta;
        if (log != NULL) {
            float *lg = log + (size_t)i * lrow + cd + (size_t)hv * (2 * KD + 1);
            lg[c] = ksh[c];
            lg[KD + c] = delta;
            if (c == 0) {
                lg[2 * KD] = decay;
            }
        }
        float o = 0.f;
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            S[d] = __fmaf_rn(ksh[d], delta, S[d]);
            o += qsh[d] * S[d];
        }
        float ss = block_sum(o * o);
        float inv = 1.f / sqrtf(ss / (float)KD + eps);
        size_t oi = (size_t)i * vd + (size_t)hv * KD + c;
        float zv = DP(const float, 4)[oi];
        float zg = (flags & 2) ? 1.f / (1.f + expf(-zv)) : qw_silu(zv);
        DP(float, 11)[oi] = o * inv * DP(const float, 9)[c] * zg;
        __syncthreads();
    }
    if (log == NULL) {
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            Sh[(size_t)d * KD + c] = S[d];
        }
    }
}

/* ---------- the DeltaNet of a large group (no log): parallel forms ----------
 * k_gdn_heads runs one block for each value head (48 blocks) and three block
 * sums for each token: about 4 us a token, 1.25 s of a bf16 prompt of 8192
 * tokens. But only the state needs the order of the tokens, and each column
 * of the state is on its own: the norms of q and k, the decay and beta do
 * not read it, and the norm of the output is a pass after it. So:
 *
 *   k_gdn_conv_par  the convolution, one thread for each (token, channel)
 *                   (the same sums, in the same order, as k_gdn_conv);
 *   k_gdn_prep      q and k of each head to their norms (in place in cv),
 *                   decay and beta to operands 5 and 6 (in place), and the
 *                   last kernel - 1 inputs to the state of the convolution;
 *   k_gdn_scan      the state: GS threads for each column, each with KD / GS
 *                   values of it, the tokens in steps of GDN_TC in shared
 *                   memory; the output without its norm to operand 11;
 *   k_gdn_out       the norm of the output and the gate.
 *
 * The sums of a column go in GS parts, so the bits differ a little from those
 * of the steps. A group with a log (the MTP verify) keeps k_gdn_heads. */
__global__ void k_gdn_conv_par(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int kernel = DI(3);
    int cd = 2 * DI(14) * DI(16) + DI(15) * DI(17);
    int c = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (c >= cd || i >= gdn_nreal(r, e)) {
        return;
    }
    const float *qkv = DP(const float, 0), *conv = DP(const float, 1);
    const float *w = DP(const float, 2) + (size_t)c * kernel;
    float v = w[kernel - 1] * qkv[(size_t)i * cd + c];
    for (int j = 0; j < kernel - 1; ++j) {
        int p = i - (kernel - 1) + j;
        float h = p >= 0 ? qkv[(size_t)p * cd + c] : conv[(size_t)(p + kernel - 1) * cd + c];
        v += w[j] * h;
    }
    DP(float, 12)[(size_t)i * cd + c] = qw_silu(v);
}

template <int KD>
__global__ void __launch_bounds__(KD) k_gdn_prep(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int kh = DI(14), vh = DI(15), t = gdn_nreal(r, e), kernel = DI(3);
    int kd = kh * KD, cd = 2 * kd + vh * KD;
    int i = blockIdx.x, x = blockIdx.y, c = threadIdx.x;
    if (x < kh) {
        if (i >= t) {
            return;
        }
        float *row = DP(float, 12) + (size_t)i * cd;
        float qv = row[x * KD + c], kv = row[kd + x * KD + c];
        float nq = block_sum(qv * qv), nk = block_sum(kv * kv);
        row[x * KD + c] = qv * (1.f / sqrtf(nq + 1e-6f) / sqrtf((float)KD));
        row[kd + x * KD + c] = kv * (1.f / sqrtf(nk + 1e-6f));
    } else if (x == kh) {
        if (i >= t || c >= vh) {
            return;
        }
        float Aexp = expf(DP(const float, 7)[c]);
        float *a = DP(float, 5) + (size_t)i * vh + c, *b = DP(float, 6) + (size_t)i * vh + c;
        float av = *a + DP(const float, 8)[c];
        float sp = av > 20.f ? av : log1pf(expf(av));
        *a = expf(-Aexp * sp);
        *b = 1.f / (1.f + expf(-*b));
    } else {
        /* the state of the convolution: the last kernel - 1 inputs (rows of
         * the old state when t < kernel - 1; row j reads row t + j > j) */
        const float *qkv = DP(const float, 0);
        float *conv = DP(float, 1);
        for (int ch = i * KD + c; ch < cd; ch += gridDim.x * KD) {
            for (int j = 0; j < kernel - 1; ++j) {
                int p = t - (kernel - 1) + j;
                conv[(size_t)j * cd + ch] = p >= 0 ? qkv[(size_t)p * cd + ch]
                                                   : conv[(size_t)(t + j) * cd + ch];
            }
        }
    }
}

#define GDN_TC 32
template <int KD, int GS>
__global__ void __launch_bounds__(128) k_gdn_scan(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    constexpr int NV = KD / GS, CB = 128 / GS;     /* values of a thread, columns of a block */
    __shared__ __align__(16) float qs[GDN_TC][KD], ks[GDN_TC][KD];
    __shared__ float dec[GDN_TC], bet[GDN_TC];
    int kh = DI(14), vh = DI(15), t = gdn_nreal(r, e), tiled = DI(20) & 1;
    int kd = kh * KD, vd = vh * KD, cd = 2 * kd + vd;
    int hv = blockIdx.x, part = threadIdx.x % GS;
    int col = blockIdx.y * CB + threadIdx.x / GS;
    int hk = tiled ? hv % kh : hv / (vh / kh);
    const float *cv = DP(const float, 12), *da = DP(const float, 5), *db = DP(const float, 6);
    float *Sh = DP(float, 10) + (size_t)hv * KD * KD, *out = DP(float, 11);
    /* thread value u: d = 4 GS (u / 4) + 4 part + u % 4 (float4 reads of q and
     * k with no bank conflicts) */
    float S[NV];
    #pragma unroll
    for (int u = 0; u < NV; ++u) {
        S[u] = Sh[(size_t)(4 * GS * (u / 4) + 4 * part + u % 4) * KD + col];
    }
    for (int i0 = 0; i0 < t; i0 += GDN_TC) {
        int n = min(GDN_TC, t - i0);
        __syncthreads();
        for (int z = threadIdx.x; z < n * KD / 4; z += 128) {
            int ii = z / (KD / 4), d4 = (z % (KD / 4)) * 4;
            const float *row = cv + (size_t)(i0 + ii) * cd;
            *(float4 *)&qs[ii][d4] = *(const float4 *)(row + hk * KD + d4);
            *(float4 *)&ks[ii][d4] = *(const float4 *)(row + kd + hk * KD + d4);
        }
        if (threadIdx.x < n) {
            dec[threadIdx.x] = da[(size_t)(i0 + threadIdx.x) * vh + hv];
            bet[threadIdx.x] = db[(size_t)(i0 + threadIdx.x) * vh + hv];
        }
        __syncthreads();
        for (int ii = 0; ii < n; ++ii) {
            float decay = dec[ii], beta = bet[ii];
            float vv = cv[(size_t)(i0 + ii) * cd + 2 * kd + hv * KD + col];
            float s0 = 0.f, s1 = 0.f;
            #pragma unroll
            for (int u4 = 0; u4 < NV / 4; ++u4) {
                float4 k4 = *(const float4 *)&ks[ii][4 * GS * u4 + 4 * part];
                S[4 * u4 + 0] = __fmul_rn(S[4 * u4 + 0], decay);
                S[4 * u4 + 1] = __fmul_rn(S[4 * u4 + 1], decay);
                S[4 * u4 + 2] = __fmul_rn(S[4 * u4 + 2], decay);
                S[4 * u4 + 3] = __fmul_rn(S[4 * u4 + 3], decay);
                s0 += k4.x * S[4 * u4 + 0] + k4.y * S[4 * u4 + 1];
                s1 += k4.z * S[4 * u4 + 2] + k4.w * S[4 * u4 + 3];
            }
            float sk = s0 + s1;
            #pragma unroll
            for (int o2 = 1; o2 < GS; o2 <<= 1) {
                sk += __shfl_xor_sync(0xffffffff, sk, o2);
            }
            float delta = (vv - sk) * beta;
            float o0 = 0.f, o1 = 0.f;
            #pragma unroll
            for (int u4 = 0; u4 < NV / 4; ++u4) {
                float4 k4 = *(const float4 *)&ks[ii][4 * GS * u4 + 4 * part];
                float4 q4 = *(const float4 *)&qs[ii][4 * GS * u4 + 4 * part];
                S[4 * u4 + 0] = __fmaf_rn(k4.x, delta, S[4 * u4 + 0]);
                S[4 * u4 + 1] = __fmaf_rn(k4.y, delta, S[4 * u4 + 1]);
                S[4 * u4 + 2] = __fmaf_rn(k4.z, delta, S[4 * u4 + 2]);
                S[4 * u4 + 3] = __fmaf_rn(k4.w, delta, S[4 * u4 + 3]);
                o0 += q4.x * S[4 * u4 + 0] + q4.y * S[4 * u4 + 1];
                o1 += q4.z * S[4 * u4 + 2] + q4.w * S[4 * u4 + 3];
            }
            float o = o0 + o1;
            #pragma unroll
            for (int o2 = 1; o2 < GS; o2 <<= 1) {
                o += __shfl_xor_sync(0xffffffff, o, o2);
            }
            if (part == 0) {
                out[(size_t)(i0 + ii) * vd + hv * KD + col] = o;
            }
        }
    }
    #pragma unroll
    for (int u = 0; u < NV; ++u) {
        Sh[(size_t)(4 * GS * (u / 4) + 4 * part + u % 4) * KD + col] = S[u];
    }
}

template <int KD>
__global__ void __launch_bounds__(KD) k_gdn_out(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int vh = DI(15), flags = DI(20);
    int i = blockIdx.x, hv = blockIdx.y, c = threadIdx.x;
    if (i >= gdn_nreal(r, e)) {
        return;
    }
    size_t oi = (size_t)i * vh * KD + (size_t)hv * KD + c;
    float *out = DP(float, 11);
    float o = out[oi];
    float ss = block_sum(o * o);
    float inv = 1.f / sqrtf(ss / (float)KD + df(r, e, 18));
    float zv = DP(const float, 4)[oi];
    float zg = (flags & 2) ? 1.f / (1.f + expf(-zv)) : qw_silu(zv);
    out[oi] = o * inv * DP(const float, 9)[c] * zg;
}

/* NP_GEMMA_GPU_GDN_PAR: the threads of a column of the state in k_gdn_scan
 * for a group of at least GDN_PAR_MIN tokens with no log (4 or 8; 0 keeps
 * k_gdn_heads). */
#define GDN_PAR_MIN 64
static int gdn_par_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_GDN_PAR");
        on = v == NULL ? 4 : atoi(v);
    }
    return on;
}

/* gg_gdn_commit: the first n tokens of a log to conv and S. */
__global__ void k_gdn_commit_conv(float *conv, const float *log, int n, int kernel, int cd,
                                  size_t lrow)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cd) {
        return;
    }
    float hist[8];
    for (int j = 0; j < kernel - 1; ++j) {
        hist[j] = conv[(size_t)j * cd + c];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < kernel - 2; ++j) {
            hist[j] = hist[j + 1];
        }
        hist[kernel - 2] = log[(size_t)i * lrow + c];
    }
    for (int j = 0; j < kernel - 1; ++j) {
        conv[(size_t)j * cd + c] = hist[j];
    }
}

template <int KD>
__global__ void __launch_bounds__(KD) k_gdn_commit_heads(float *S, const float *log, int n,
                                                         int cd, size_t lrow)
{
    int hv = blockIdx.x, c = threadIdx.x;
    float *Sh = S + (size_t)hv * KD * KD;
    float st[KD];
    #pragma unroll
    for (int d = 0; d < KD; ++d) {
        st[d] = Sh[(size_t)d * KD + c];
    }
    for (int i = 0; i < n; ++i) {
        const float *lg = log + (size_t)i * lrow + cd + (size_t)hv * (2 * KD + 1);
        float decay = lg[2 * KD], delta = lg[KD + c];
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            st[d] = __fmul_rn(st[d], decay);
        }
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            st[d] = __fmaf_rn(lg[d], delta, st[d]);
        }
    }
    #pragma unroll
    for (int d = 0; d < KD; ++d) {
        Sh[(size_t)d * KD + c] = st[d];
    }
}

/* GP_KQ_HOT_MOE: the experts that the GPU holds, and the shared expert.
 *
 *     h, val, idx, map, gate, up, down, act, act2, de, out, top_k, inner,
 *     hidden, gtype, dtype, t, sgate, sup, sdown, stype, slog
 *
 * gate, up, and down hold the experts of the slots (map gives the slot of
 * each expert, or -1), one after the other. Pair j < t top_k is slot
 * j % top_k of token j / top_k; pair t top_k + j is the shared expert of
 * token j (always on the GPU), with the weight sigmoid(slog[j]). The steps
 * are those of GP_HOT_MOE: gate and up, silu(gate) up, down, and the sum. */
/* The pool of HotCache (gpu.ExpertPool): expert blocks on the GPU in
 * segments of GG_POOL_K blocks that come and go (the room lent by the image
 * encoder, the free memory). A slot value KQH_WARM + i is block i: segment
 * i / GG_POOL_K, place i % GG_POOL_K, at a stride of its part (the bytes of
 * an expert of the layers that use the pool). gg_set_pool writes the table;
 * the graphs read it at run time, and a change adds segments or clears ones
 * that no slot uses. */
#define KQH_WARM (1 << 24)
#define GG_POOL_K 8
#define GG_POOL_SEG 512
__device__ const uint8_t *gg_pool_seg[3][GG_POOL_SEG];
__device__ size_t gg_pool_stride[3];

/* tab: the host table of the pool (3 GG_POOL_SEG segment addresses, part
 * by part, then the 3 strides) */
extern "C" int gg_set_pool(const int64_t *tab)
{
    if (cudaMemcpyToSymbol(gg_pool_seg, tab, sizeof(gg_pool_seg)) != cudaSuccess) {
        return -1;
    }
    size_t st[3] = {(size_t)tab[3 * GG_POOL_SEG], (size_t)tab[3 * GG_POOL_SEG + 1],
                    (size_t)tab[3 * GG_POOL_SEG + 2]};
    return cudaMemcpyToSymbol(gg_pool_stride, st, sizeof(st)) == cudaSuccess ? 0 : -1;
}

/* Fences (gpumm.GpuMem): gg_fence records an event on the stream of the
 * programs after the work queued so far (a run returns before the GPU is
 * done) and returns its number; gg_fence_done(n, wait) gives 1 when that
 * work is done (wait: wait for it), 0 if not yet, -1 on an error. The events
 * are a ring: a fence older than the ring is done. */
#define GG_FENCES 4096
static cudaEvent_t gg_fev[GG_FENCES];
static int64_t gg_fseq;

extern "C" int64_t gg_fence(void)
{
    if (gg_fseq == 0) {
        for (int k = 0; k < GG_FENCES; ++k) {
            if (cudaEventCreateWithFlags(&gg_fev[k], cudaEventDisableTiming) != cudaSuccess) {
                return -1;
            }
        }
    }
    int64_t n = ++gg_fseq;
    if (cudaEventRecord(gg_fev[n % GG_FENCES], gg_stream) != cudaSuccess) {
        return -1;
    }
    return n;
}

extern "C" int gg_fence_done(int64_t n, int wait)
{
    if (n <= 0 || gg_fseq - n >= GG_FENCES) {
        return 1;
    }
    cudaEvent_t ev = gg_fev[n % GG_FENCES];
    cudaError_t e = wait ? cudaEventSynchronize(ev) : cudaEventQuery(ev);
    if (e == cudaErrorNotReady) {
        return 0;
    }
    return e == cudaSuccess ? 1 : -1;
}

extern "C" int gg_pool_dims(int *k, int *seg)
{
    *k = GG_POOL_K;
    *seg = GG_POOL_SEG;
    return 0;
}

/* The bytes of slot of part (0 gate, 1 up, 2 down; per: the bytes of one
 * expert of that part): the store of the layer (operands 4, 5, 6), or a
 * block of the pool. */
__device__ __forceinline__ const uint8_t *kqh_wslot(const gp_rec *r, const int64_t *e, int slot,
                                                    int part, size_t per)
{
    if (slot >= KQH_WARM) {
        int i = slot - KQH_WARM;
        return gg_pool_seg[part][i / GG_POOL_K] + (size_t)(i % GG_POOL_K) * gg_pool_stride[part];
    }
    return DP(const uint8_t, 4 + part) + (size_t)slot * per;
}

__device__ __forceinline__ int kqh_slot(const gp_rec *r, const int64_t *e, int j)
{
    int tk = DI(16) * DI(11);
    return j >= tk ? 0 : DP(const int, 3)[DP(const int, 2)[j]];
}

/* Operands 22 and 23 (or none): zc (the flags of the pairs that read the
 * pinned copy of the experts in host memory, GP_HOT_SPLIT), and a table of
 * the device addresses of the gate, up, and down experts of the copy
 * (mapped: the GPU reads them over PCIe). */
__device__ __forceinline__ int kqh_zc(const gp_rec *r, const int64_t *e, int j)
{
    return r->tag[22] != GP_T_NONE && j < DI(16) * DI(11) && DP(const int, 22)[j] != 0;
}

/* A pair that the GPU computes: a hot expert, the shared expert, or a cold
 * expert of the copy (kqh_zc). */
__device__ __forceinline__ int kqh_on(const gp_rec *r, const int64_t *e, int j)
{
    return kqh_slot(r, e, j) >= 0 || kqh_zc(r, e, j);
}

__global__ void k_kqh_gu(const gp_rec *r, const int64_t *e, int jofs = 0)
{
    PDL_START();
    int j = blockIdx.y + jofs, slot = kqh_slot(r, e, j), up = blockIdx.z;
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int inner = DI(12), hidden = DI(13), k = DI(11), tk = DI(16) * k;
    if (slot < 0 || row >= inner) {
        return;
    }
    int shared = j >= tk;
    int type = shared ? DI(20) : DI(14);
    size_t rb = kq_row_bytes(type, hidden);
    const uint8_t *w = shared ? DP(const uint8_t, up ? 18 : 17)
                              : kqh_wslot(r, e, slot, up, (size_t)inner * rb);
    int tok = shared ? j - tk : j / k;
    if (type == KQ_NVX) {
        /* a warp for each group of 16 rows (row counts the groups) */
        if (row >= inner / 16) {
            return;
        }
        float v[4];
        kq_nvx_group(w + (size_t)16 * row * rb, DP(const float, 0) + (size_t)tok * hidden, hidden, v);
        int l = threadIdx.x % 32;
        if (l < 4) {
            float *o = DP(float, 7) + (size_t)j * 2 * inner + up * inner + 16 * row;
            o[2 * l] = v[0];
            o[2 * l + 1] = v[1];
            o[2 * l + 8] = v[2];
            o[2 * l + 9] = v[3];
        }
        return;
    }
    float v = kq_row(type, w + (size_t)row * rb, DP(const float, 0) + (size_t)tok * hidden, hidden);
    if (threadIdx.x % 32 == 0) {
        DP(float, 7)[(size_t)j * 2 * inner + up * inner + row] = v;
    }
}

/* RQ8_0 experts (gguf.RQ8_0, Qwen4GPU): the MoE kernels rotate the act of
 * each pair before the down product, as kquants.c kq_moe_rot (the program
 * rotates their x). A state of the process. */
static int gg_moe_rot = 0;

extern "C" void gg_set_moe_rot(int on)
{
    gg_moe_rot = on;
}

/* rot: the RQ8_0 experts (gg_moe_rot): the act rotated in each 32 values
 * (a warp a group; inner and blockDim are multiples of 32) */
__global__ void k_kqh_act(const gp_rec *r, const int64_t *e, int rot)
{
    PDL_START();
    int j = blockIdx.x;
    if (!kqh_on(r, e, j)) {
        return;
    }
    int inner = DI(12);
    const float *g = DP(const float, 7) + (size_t)j * 2 * inner;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = qw_silu(g[i]) * g[inner + i];
        if (rot) {
            v = tq6_rot_warp(v, i % 32, false);
        }
        DP(float, 8)[(size_t)j * inner + i] = v;
    }
}

__global__ void k_kqh_dn(const gp_rec *r, const int64_t *e, int jofs = 0)
{
    PDL_START();
    int j = blockIdx.y + jofs, slot = kqh_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int inner = DI(12), hidden = DI(13), tk = DI(16) * DI(11);
    if (slot < 0 || row >= hidden) {
        return;
    }
    int shared = j >= tk;
    int type = shared ? DI(20) : DI(15);
    size_t rb = kq_row_bytes(type, inner);
    const uint8_t *w = shared ? DP(const uint8_t, 19)
                              : kqh_wslot(r, e, slot, 2, (size_t)hidden * rb);
    if (type == KQ_NVX) {
        if (row >= hidden / 16) {
            return;
        }
        float v[4];
        kq_nvx_group(w + (size_t)16 * row * rb, DP(const float, 8) + (size_t)j * inner, inner, v);
        int l = threadIdx.x % 32;
        if (l < 4) {
            float *o = DP(float, 9) + (size_t)j * hidden + 16 * row;
            o[2 * l] = v[0];
            o[2 * l + 1] = v[1];
            o[2 * l + 8] = v[2];
            o[2 * l + 9] = v[3];
        }
        return;
    }
    float v = kq_row(type, w + (size_t)row * rb, DP(const float, 8) + (size_t)j * inner, inner);
    if (threadIdx.x % 32 == 0) {
        DP(float, 9)[(size_t)j * hidden + row] = v;
    }
}

/* The KQ_NVX experts of GP_KQ_HOT_MOE (the pairs j < t top_k): a block of KS
 * warps for each group of 16 rows (blockIdx.x), pair (blockIdx.y), and
 * matrix (blockIdx.z: gate or up; down with dn). Warp w takes the blocks
 * of 32 columns w, w + KS, ... of the group (kq_nvx_group for a part of
 * the columns), and the warps add their sums in shared memory. With a warp
 * for each whole group (k_kqh_gu) a step had about 250 KB of loads in
 * flight: 166 GB/s, 72 us for the gate and up of a layer. */
template <int KS>
__global__ void __launch_bounds__(32 * KS) k_kqh_nvx(const gp_rec *r, const int64_t *e, int dn)
{
    PDL_START();
    __shared__ float red[KS][16];
    int grp = blockIdx.x, j = blockIdx.y, up = blockIdx.z;
    int slot = kqh_slot(r, e, j), zc = slot < 0 && kqh_zc(r, e, j);
    if (slot < 0 && !zc) {
        return;
    }
    int inner = DI(12), hidden = DI(13), k = DI(11);
    int cols = dn ? inner : hidden;
    size_t rb = kq_row_bytes(KQ_NVX, cols);
    const uint8_t *w;
    if (zc) {
        /* expert idx[j] of the copy in host memory, over PCIe */
        size_t ex = (size_t)DP(const int, 2)[j];
        const int64_t *zt = DP(const int64_t, 23);
        w = dn ? (const uint8_t *)(intptr_t)zt[2] + ex * hidden * rb
               : (const uint8_t *)(intptr_t)zt[up ? 1 : 0] + ex * inner * rb;
    } else {
        w = dn ? kqh_wslot(r, e, slot, 2, (size_t)hidden * rb)
               : kqh_wslot(r, e, slot, up, (size_t)inner * rb);
    }
    const uint8_t *wg = w + (size_t)16 * grp * rb;
    const float *x = dn ? DP(const float, 8) + (size_t)j * inner
                        : DP(const float, 0) + (size_t)(j / k) * hidden;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, st = lane / 4, q = lane % 4, h = st / 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 2
    for (int b = warp; b < cols / 32; b += KS) {
        const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
        uint2 c = *(const uint2 *)(blk + 8 * lane);
        float4 xv = *(const float4 *)(x + 32 * b + 4 * st);
        uint32_t s01 = *(const uint16_t *)(blk + 256 + 16 * h + 2 * q);
        uint32_t s89 = *(const uint16_t *)(blk + 256 + 16 * h + 8 + 2 * q);
        acc[0] += kq_e4m3s(s01 & 255) * kq_dot4s8(kt_e2m1x4(c.x), xv);
        acc[1] += kq_e4m3s(s01 >> 8) * kq_dot4s8(kt_e2m1x4(c.y), xv);
        acc[2] += kq_e4m3s(s89 & 255) * kq_dot4s8(kt_e2m1x4(c.x >> 4), xv);
        acc[3] += kq_e4m3s(s89 >> 8) * kq_dot4s8(kt_e2m1x4(c.y >> 4), xv);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int o = 4; o < 32; o <<= 1) {
            acc[i] += __shfl_xor_sync(0xffffffff, acc[i], o);
        }
    }
    if (lane < 4) {
        red[warp][2 * lane] = acc[0];
        red[warp][2 * lane + 1] = acc[1];
        red[warp][2 * lane + 8] = acc[2];
        red[warp][2 * lane + 9] = acc[3];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sm = 0.f;
        #pragma unroll
        for (int i = 0; i < KS; ++i) {
            sm += red[i][threadIdx.x];
        }
        sm *= 128.f * *(const float *)wg;            /* 0.5 g, and the 2^8 of kq_e4m3s */
        if (dn) {
            DP(float, 9)[(size_t)j * hidden + 16 * grp + threadIdx.x] = sm;
        } else {
            DP(float, 7)[(size_t)j * 2 * inner + up * inner + 16 * grp + threadIdx.x] = sm;
        }
    }
}

static int kqh_nvx_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_KQH_NVX");
        on = !(v && v[0] == '0');
    }
    return on;
}

__global__ void k_kqh_sum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int c = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    int hidden = DI(13), k = DI(11), tk = DI(16) * k;
    if (c >= hidden) {
        return;
    }
    const float *val = DP(const float, 1), *de = DP(const float, 9);
    float acc = 0.f;
    for (int s2 = 0; s2 < k; ++s2) {
        int j = tok * k + s2;
        if (kqh_on(r, e, j)) {
            acc += val[j] * de[(size_t)j * hidden + c];
        }
    }
    float sw = 1.f / (1.f + expf(-DP(const float, 21)[tok]));
    acc += sw * de[(size_t)(tk + tok) * hidden + c];
    DP(float, 10)[(size_t)tok * hidden + c] = acc;
}

/* GP_ADD_RMS: x, o, w, h, rows, cols, eps. x += o, then h = rms_norm(x) w
 * (GP_ADD and GP_RMS_NORM of compile_qwen_step in one kernel). One block
 * for each row. */
__global__ void k_add_rms(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int cols = DI(5);
    float *x = DP(float, 0) + (size_t)blockIdx.x * cols;
    const float *o = DP(const float, 1) + (size_t)blockIdx.x * cols;
    const float *w = DP(const float, 2);
    float *h = DP(float, 3) + (size_t)blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[i] + o[i];
        x[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)cols + df(r, e, 6));
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        h[i] = x[i] * s * w[i];
    }
}

/* ---------- the gated residual and the n-gram layer of Qwen3.8 ----------
 * The records of csrc/hyperconn.c; the operands are the same. */

__device__ __forceinline__ float hc_sig(float v)
{
    return 1.f / (1.f + expf(-v));
}

/* GP_HC_NORM: x, w, out, t, groups, hid, eps. A block for each group of
 * each row. */
__device__ void row_quant8(const float *x, int n, size_t b0, int8_t *q, float *xs, float *xsum);
__global__ void k_hc_norm(const gp_rec *r, const int64_t *e, int8_t *q = NULL, float *qs = NULL,
                          float *qsum = NULL)
{
    PDL_START();
    int groups = DI(4), hid = DI(5);
    size_t o = (size_t)blockIdx.x * hid;
    const float *x = DP(const float, 0) + o;
    const float *w = DP(const float, 1) + (size_t)(blockIdx.x % groups) * hid;
    float *out = DP(float, 2) + o;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        ss += x[c] * x[c];
    }
    ss = block_sum(ss);
    float inv = 1.f / sqrtf(ss / (float)hid + df(r, e, 6));
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        out[c] = x[c] * inv * w[c];
    }
    if (q != NULL) {
        __syncthreads();
        row_quant8(out, hid, (size_t)blockIdx.x * (hid / 32), q, qs, qsum);
    }
}

/* The int8 x (quant8_block, the bits of k_kq_quant_x) of the n values at x
 * (written by this block before a __syncthreads), from block b0 of 32 on:
 * all the threads of the block call it. */
__device__ void row_quant8(const float *x, int n, size_t b0, int8_t *q, float *xs, float *xsum)
{
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nw = blockDim.x / 32, nb = n / 32;
    for (int base = warp * 4; base < nb; base += nw * 4) {
        int b = base + lane / 8;
        bool live = b < nb;
        float4 v = live ? *(const float4 *)(x + (size_t)b * 32 + 4 * (lane % 8))
                        : make_float4(0.f, 0.f, 0.f, 0.f);
        quant8_block(v, live, b0 + (size_t)b, lane % 8, q, xs, xsum);
    }
}

/* GP_HC_ACT: x, out, n, scale. out = silu(x * scale). */
__global__ void k_hc_act(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < di(r, e, 2)) {
        float v = DP(const float, 0)[i] * df(r, e, 3);
        DP(float, 1)[i] = v * hc_sig(v);
    }
}

/* GP_HC_MIX: hn, g, out, t, hc, hid. A thread for each value of out. */
__global__ void k_hc_mix(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)DI(3) * hid) {
        return;
    }
    int64_t j = i / hid, c = i % hid;
    const float *hn = DP(const float, 0), *g = DP(const float, 1);
    float s = 0.f;
    for (int k = 0; k < hc; ++k) {
        size_t x = ((size_t)j * hc + k) * hid + c;
        s += hc_sig(g[x]) * hn[x];
    }
    DP(float, 2)[i] = s / (float)hc;
}

/* GP_HC_ADD: H, out, inject, t, hc, hid, scale. A thread for each value of
 * H. */
__global__ void k_hc_add(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)DI(3) * hc * hid) {
        return;
    }
    int64_t rg = i / hid, j = rg / hc, c = i % hid;
    float w = 2.f * hc_sig(DP(const float, 2)[rg] * df(r, e, 6));
    DP(float, 0)[i] += DP(const float, 1)[j * hid + c] * w;
}

/* GP_HC_ADD (r) and the GP_HC_NORM of the same H that follows (r2), one
 * block for each stream of each row: the add, then the norm with the sums
 * of k_hc_norm in the same order (the same bits, one kernel in place of two;
 * NP_GEMMA_GPU_HC_FUSE). q: the int8 x of the norm (row_quant8). */
__global__ void k_hc_add_norm(const gp_rec *r, const int64_t *e, const gp_rec *r2, int8_t *q,
                              float *qs, float *qsum)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int rg = blockIdx.x, j = rg / hc;
    float wv = 2.f * hc_sig(DP(const float, 2)[rg] * df(r, e, 6));
    float *H = DP(float, 0) + (size_t)rg * hid;
    const float *o = DP(const float, 1) + (size_t)j * hid;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        H[c] += o[c] * wv;
        ss += H[c] * H[c];
    }
    {
        const gp_rec *r0 = r;
        r = r2;
        int groups = DI(4);
        const float *w = DP(const float, 1) + (size_t)(rg % groups) * hid;
        float *out = DP(float, 2) + (size_t)rg * hid;
        ss = block_sum(ss);
        float inv = 1.f / sqrtf(ss / (float)hid + df(r, e, 6));
        for (int c = threadIdx.x; c < hid; c += blockDim.x) {
            out[c] = H[c] * inv * w[c];
        }
        if (q != NULL) {
            __syncthreads();
            row_quant8(out, hid, (size_t)rg * (hid / 32), q, qs, qsum);
        }
        r = r0;
    }
}

/* GP_HC_ACT and GP_HC_MIX with the int8 x of their output for the next
 * product: 4 values for each thread, 8 threads for a block of 32. */
__global__ void k_hc_act_q(const gp_rec *r, const int64_t *e, int8_t *q, float *qs, float *qsum)
{
    PDL_START();
    int64_t n = di(r, e, 2), i4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool live = 4 * i4 < n;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (live) {
        float4 x = *(const float4 *)(DP(const float, 0) + 4 * i4);
        float sc = df(r, e, 3), a;
        a = x.x * sc; v.x = a * hc_sig(a);
        a = x.y * sc; v.y = a * hc_sig(a);
        a = x.z * sc; v.z = a * hc_sig(a);
        a = x.w * sc; v.w = a * hc_sig(a);
        *(float4 *)(DP(float, 1) + 4 * i4) = v;
    }
    quant8_block(v, live, (size_t)(i4 / 8), threadIdx.x % 8, q, qs, qsum);
}

__global__ void k_hc_mix_q(const gp_rec *r, const int64_t *e, int8_t *q, float *qs, float *qsum)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int64_t i4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool live = 4 * i4 < (int64_t)DI(3) * hid;
    float vv[4] = {0.f, 0.f, 0.f, 0.f};
    if (live) {
        const float *hn = DP(const float, 0), *g = DP(const float, 1);
        for (int u = 0; u < 4; ++u) {
            int64_t i = 4 * i4 + u, j = i / hid, c = i % hid;
            float sm = 0.f;
            for (int k = 0; k < hc; ++k) {
                size_t x = ((size_t)j * hc + k) * hid + c;
                sm += hc_sig(g[x]) * hn[x];
            }
            vv[u] = sm / (float)hc;
        }
    }
    float4 v = make_float4(vv[0], vv[1], vv[2], vv[3]);
    if (live) {
        *(float4 *)(DP(float, 2) + 4 * i4) = v;
    }
    quant8_block(v, live, (size_t)(i4 / 8), threadIdx.x % 8, q, qs, qsum);
}

static int hc_fuse_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_HC_FUSE");
        on = !(v && v[0] == '0');
    }
    return on;
}

static int topk_split_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_TOPK_SPLIT");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* GP_PLE_GATE: keyn, qn, value, gated, t, hc, hid. A block for each stream
 * of each row. */
__global__ void k_ple_gate(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hc = DI(5), hid = DI(6);
    size_t o = (size_t)blockIdx.x * hid;
    const float *k = DP(const float, 0) + o, *q = DP(const float, 1) + o;
    float s = 0.f;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        s += k[c] * q[c];
    }
    s = block_sum(s) / sqrtf((float)hid);
    float mag = sqrtf(fabsf(s) > 1e-6f ? fabsf(s) : 1e-6f);
    float g = hc_sig(s < 0.f ? -mag : (s > 0.f ? mag : 0.f));
    const float *v = DP(const float, 2) + (size_t)(blockIdx.x / hc) * hid;
    float *out = DP(float, 3) + o;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        out[c] = g * v[c];
    }
}

/* GP_PLE_CONV: gn, gated, H, state, w, t, channels, kernel, dilation. A
 * thread for each channel: the rows in order, then the new state. */
__global__ void k_ple_conv(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int t = DI(5), channels = DI(6), kernel = DI(7), dil = DI(8);
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    const float *gn = DP(const float, 0), *gated = DP(const float, 1), *w = DP(const float, 4);
    float *H = DP(float, 2), *state = DP(float, 3);
    int hist = (kernel - 1) * dil;
    for (int j = 0; j < t; ++j) {
        float v = 0.f;
        for (int k = 0; k < kernel; ++k) {
            int rr = j - (kernel - 1 - k) * dil;
            float x = rr >= 0 ? gn[(size_t)rr * channels + c]
                              : state[(size_t)(hist + rr) * channels + c];
            v += w[(size_t)c * kernel + k] * x;
        }
        size_t i = (size_t)j * channels + c;
        H[i] += gated[i] + v * hc_sig(v);
    }
    for (int rr = 0; rr < hist; ++rr) {
        int src = t - hist + rr;
        state[(size_t)rr * channels + c] = src >= 0 ? gn[(size_t)src * channels + c]
                                                    : state[(size_t)(hist + src) * channels + c];
    }
}

/* GP_HC_CAT: e, hn, out, t, hc, hid. For each stream: the row of e, then
 * the stream. */
__global__ void k_hc_cat(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)DI(3) * hc * 2 * hid) {
        return;
    }
    int64_t rg = i / (2 * hid), c = i % (2 * hid);
    DP(float, 2)[i] = c < hid ? DP(const float, 0)[(rg / hc) * hid + c]
                              : DP(const float, 1)[rg * hid + c - hid];
}

/* ---------- QSA (Qwen3.8): the selection of the keys (csrc/qsa.c) ----------
 * GP_QSA_SELECT: iq, ik, idxk, blk, qn, kn, cos, sin, pos, t, heads, d,
 * ratio, budget, rot, theta, eps, sel, cnt, maxsel, scratch, nbmax, qpos,
 * sec (qpos: the M-RoPE positions of the rows, see qsa_rpos). idxk and blk
 * keep float16 values, as on the CPU. Three
 * kernels: the raw keys to idxk; the key of each block that is complete now;
 * the selection of each query (one block of QSA_T threads for each query;
 * scratch has nbmax 64-bit keys for each query of a launch: the queries go
 * in launches of QSA_ROWS, so the scratch is QSA_ROWS * nbmax keys at most,
 * 0.27 GB at 262144 positions, in place of t * nbmax: 1.9 GB for a group of
 * 4096 rows at 225K, which took the last of the memory of the GPU). */
#define QSA_T 1024
#define QSA_ROWS 512
/* The start of the scratch: the queries of a launch after their norm and
 * RoPE as two float16 planes (k_qsa_qprep, for k_qsa_score_tc: heads * d
 * <= 8 * 256 values); the keys of the queries follow. */
#define QSA_QBUF ((size_t)QSA_ROWS * 8 * 256 * 2 * 2)

/* The queries of a launch of k_qsa_query (the rows of the scratch). */
extern "C" int gg_qsa_rows(void)
{
    return QSA_ROWS;
}

/* The bytes of the scratch before the keys (QSA_QBUF). */
extern "C" int64_t gg_qsa_extra(void)
{
    return (int64_t)QSA_QBUF;
}

/* RMS norm of d values (d <= 256) times w, then RoPE on the first rot
 * values; x in shared memory, the threads of the block (or of one warp: the
 * sums use the threads that call). */
__device__ void qsa_norm_rope_block(float *x, const float *w, int d, float eps, int rot,
                                    const float *c, const float *sn)
{
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    ss = block_sum(ss);
    float inv = 1.f / sqrtf(ss / (float)d + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        x[i] = x[i] * inv * w[i];
    }
    __syncthreads();
    int half = rot / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float a = x[i], b = x[i + half];
        x[i] = a * c[i] - b * sn[i];
        x[i + half] = b * c[i + half] + a * sn[i + half];
    }
    __syncthreads();
}

__global__ void k_qsa_copy(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int d = DI(11);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)DI(9) * d) {
        DP(__half, 2)[di(r, e, 8) * d + i] = __float2half_rn(DP(const float, 1)[i]);
    }
}

/* Block x makes the key of block pos / ratio + x, if it is complete. */
__global__ void k_qsa_blocks(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float x[256], c[256], sn[256];
    int64_t pos = di(r, e, 8), n = pos + DI(9);
    int d = DI(11), ratio = DI(12), rot = DI(14);
    int64_t b = pos / ratio + blockIdx.x;
    if (b >= n / ratio) {
        return;
    }
    const __half *idxk = DP(const __half, 2);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float sum = 0.f;
        for (int q = 0; q < ratio; ++q) {
            sum += __half2float(idxk[(size_t)(b * ratio + q) * d + i]);
        }
        x[i] = sum / (float)ratio;
    }
    double theta = (double)df(r, e, 15);
    const int32_t *qpos = DP(const int32_t, 22);
    int sec = DI(23);
    for (int i = threadIdx.x; i < rot / 2; i += blockDim.x) {
        int64_t p = b * ratio;
        if (qpos != NULL) {
            int a = (i % 3 == 1 && i < 3 * (sec & 255)) ? 1 : (i % 3 == 2 && i < 3 * (sec >> 8)) ? 2 : 0;
            p = qpos[p * 3 + a];
        }
        double f = (double)p / pow(theta, (double)(2 * i) / (double)rot);
        c[i] = c[i + rot / 2] = (float)cos(f);
        sn[i] = sn[i + rot / 2] = (float)sin(f);
    }
    __syncthreads();
    qsa_norm_rope_block(x, DP(const float, 5), d, df(r, e, 16), rot, c, sn);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        DP(__half, 3)[(size_t)b * d + i] = __float2half_rn(x[i]);
    }
}

/* The inclusive prefix sum of v over the block (QSA_T threads). */
__device__ int qsa_scan(int v, int *tmp)
{
    int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffff, v, o);
        v += lane >= o ? y : 0;
    }
    __syncthreads();
    if (lane == 31) {
        tmp[w] = v;
    }
    __syncthreads();
    if (w == 0) {
        int t = lane < QSA_T / 32 ? tmp[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(0xffffffff, t, o);
            t += lane >= o ? y : 0;
        }
        tmp[lane] = t;
    }
    __syncthreads();
    return v + (w > 0 ? tmp[w - 1] : 0);
}

/* k_qsa_qprep and k_qsa_score_tc: the scores of the blocks of a launch of
 * a large group on the tensor cores (4 heads of 128 values), in place of the
 * loop of k_qsa_query, which scores the blocks of one query in each block
 * with float32 dots and shuffles: it read all the block keys again for each
 * query (243 ms a layer for a group of 4096 rows at 196K positions, 12 s of
 * each 16K there). Block row k_qsa_qprep: query j0 + row after its norm and
 * RoPE, as two float16 planes (hi, lo: about 22 bits), at the start of the
 * scratch. */
__global__ void __launch_bounds__(128) k_qsa_qprep(const gp_rec *r, const int64_t *e, int j0)
{
    PDL_START();
    __shared__ float q[8 * 256];
    int jl = blockIdx.x, j = j0 + jl, heads = DI(10), d = DI(11), rot = DI(14);
    const float *iq = DP(const float, 0) + (size_t)j * heads * d;
    for (int i = threadIdx.x; i < heads * d; i += blockDim.x) {
        q[i] = iq[i];
    }
    __syncthreads();
    const float *cs = DP(const float, 6) + (size_t)j * rot, *sn = DP(const float, 7) + (size_t)j * rot;
    for (int h = 0; h < heads; ++h) {
        qsa_norm_rope_block(q + h * d, DP(const float, 4), d, df(r, e, 16), rot, cs, sn);
    }
    __half *qb = (__half *)DP(uint8_t, 20) + (size_t)jl * heads * d * 2;
    for (int i = threadIdx.x; i < heads * d; i += blockDim.x) {
        __half hi = __float2half_rn(q[i]);
        qb[i] = hi;
        qb[heads * d + i] = __float2half_rn(q[i] - __half2float(hi));
    }
}

/* Block (x, y): the 16 queries 16 y .. 16 y + 15 of the launch (rows rows),
 * the tiles of QS_KEYS block keys x, x + gridDim.x, ... up to the blocks of
 * its last query (the grid does not follow the position: the graph keeps
 * it). Warp w: queries 4 w .. 4 w + 3, their 4 heads the 16 rows of its m16
 * tile (row 4 query + head), the fragments of both planes in registers. For
 * each key: relu of each head, the sum of the heads (lanes 4 apart), times
 * 1 / sqrt(d); the key (score bits << 32 | block) of each query that sees
 * the block, as k_qsa_query writes it. */
#define QS_KEYS 128
#define QS_LD 136
#define QS_GX 16
__global__ void __launch_bounds__(128) k_qsa_score_tc(const gp_rec *r, const int64_t *e, int j0, int rows)
{
    PDL_START();
    __shared__ __align__(16) __half kt[QS_KEYS][QS_LD];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int heads = 4, d = 128;
    int ratio = DI(12);
    int64_t pos = di(r, e, 8), nbs = di(r, e, 21);
    int q0 = blockIdx.y * 16 + warp * 4;
    int jlast = min(rows, (int)blockIdx.y * 16 + 16) - 1;
    int64_t nbb = (pos + j0 + jlast + 1) / ratio;      /* the blocks of the last query */
    const __half *qb = (const __half *)DP(uint8_t, 20);
    size_t qrow = (size_t)heads * d * 2;
    uint32_t ah[8][4], al[8][4];
    {
        int qa = q0 + g / 4, ha = g % 4, qc = q0 + (g + 8) / 4, hc = (g + 8) % 4;
        const __half *pa = qb + (size_t)qa * qrow + (size_t)ha * d;
        const __half *pc = qb + (size_t)qc * qrow + (size_t)hc * d;
        bool va = qa < rows, vc = qc < rows;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int o = kk * 16 + 2 * c;
            ah[kk][0] = va ? *(const uint32_t *)(pa + o) : 0u;
            ah[kk][1] = vc ? *(const uint32_t *)(pc + o) : 0u;
            ah[kk][2] = va ? *(const uint32_t *)(pa + o + 8) : 0u;
            ah[kk][3] = vc ? *(const uint32_t *)(pc + o + 8) : 0u;
            al[kk][0] = va ? *(const uint32_t *)(pa + heads * d + o) : 0u;
            al[kk][1] = vc ? *(const uint32_t *)(pc + heads * d + o) : 0u;
            al[kk][2] = va ? *(const uint32_t *)(pa + heads * d + o + 8) : 0u;
            al[kk][3] = vc ? *(const uint32_t *)(pc + heads * d + o + 8) : 0u;
        }
    }
    const __half *blk = DP(const __half, 3);
    uint64_t *keys = (uint64_t *)(DP(uint8_t, 20) + QSA_QBUF);
    float scale = 1.f / sqrtf((float)d);
    int qa = q0 + g / 4, qc = q0 + (g + 8) / 4;
    int64_t nba = qa < rows ? (pos + j0 + qa + 1) / ratio : 0;
    int64_t nbc = qc < rows ? (pos + j0 + qc + 1) / ratio : 0;
    for (int64_t b0 = (int64_t)blockIdx.x * QS_KEYS; b0 < nbb; b0 += (int64_t)gridDim.x * QS_KEYS) {
        __syncthreads();
        for (int i = threadIdx.x; i < QS_KEYS * 16; i += blockDim.x) {
            int kr = i / 16, kc = (i % 16) * 8;
            int64_t b = b0 + kr;
            *(uint4 *)&kt[kr][kc] = b < nbb ? *(const uint4 *)(blk + (size_t)b * d + kc)
                                            : make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
        #pragma unroll 2
        for (int nt = 0; nt < QS_KEYS / 8; ++nt) {
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                uint32_t bb[2];
                bb[0] = *(const uint32_t *)&kt[nt * 8 + g][kk * 16 + 2 * c];
                bb[1] = *(const uint32_t *)&kt[nt * 8 + g][kk * 16 + 2 * c + 8];
                mma16816(sc, ah[kk], bb);
                mma16816(sc, al[kk], bb);
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                sc[u] = sc[u] > 0.f ? sc[u] : 0.f;
                sc[u] += __shfl_xor_sync(0xffffffff, sc[u], 4);
                sc[u] += __shfl_xor_sync(0xffffffff, sc[u], 8);
            }
            if ((g & 3) == 0) {
                int64_t b = b0 + nt * 8 + 2 * c;
                if (b < nba) {
                    keys[(size_t)qa * nbs + b] = ((uint64_t)__float_as_uint(sc[0] * scale) << 32) | (uint64_t)b;
                }
                if (b + 1 < nba) {
                    keys[(size_t)qa * nbs + b + 1] = ((uint64_t)__float_as_uint(sc[1] * scale) << 32) |
                                                     (uint64_t)(b + 1);
                }
                if (b < nbc) {
                    keys[(size_t)qc * nbs + b] = ((uint64_t)__float_as_uint(sc[2] * scale) << 32) | (uint64_t)b;
                }
                if (b + 1 < nbc) {
                    keys[(size_t)qc * nbs + b + 1] = ((uint64_t)__float_as_uint(sc[3] * scale) << 32) |
                                                     (uint64_t)(b + 1);
                }
            }
        }
    }
}

/* NP_GEMMA_GPU_QSA_SCORE_TC: 1 (the default) k_qsa_score_tc for a group (or
 * a step) with 4 heads of 128; 0 the loop of k_qsa_query. At 200K positions
 * the loop took 4 ms a layer for an MTP verify group of 4 rows (one SM for
 * each query): 28% of the decode. */
static int qsa_score_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_QSA_SCORE_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return on;
}

/* Block j selects the keys of query j0 + j (the method of qsa_select_body:
 * the scores of the blocks as keys (score bits << 32 | block), the top budget
 * keys by a radix select of 8 bits at a time, then the positions in order);
 * its keys are row j of the scratch. */
template <int SCORED>
__global__ void __launch_bounds__(QSA_T) k_qsa_query(const gp_rec *r, const int64_t *e, int j0)
{
    PDL_START();
    __shared__ float q[8 * 256];
    __shared__ int hist[256];
    __shared__ int tmp[32];
    __shared__ uint64_t prefix_s;
    __shared__ int need_s;
    int j = j0 + blockIdx.x, heads = DI(10), d = DI(11), ratio = DI(12), budget = DI(13);
    int rot = DI(14), maxsel = DI(19);
    int64_t pj = di(r, e, 8) + j, nb = (pj + 1) / ratio;
    int32_t *cnt = DP(int32_t, 18);
    if (nb <= budget) {
        if (threadIdx.x == 0) {
            cnt[j] = -1;
        }
        return;
    }
    uint64_t *keys = (uint64_t *)(DP(uint8_t, 20) + QSA_QBUF) + (size_t)blockIdx.x * di(r, e, 21);
    if (!SCORED) {
        /* (SCORED: k_qsa_score_tc wrote the keys) */
        const float *iq = DP(const float, 0) + (size_t)j * heads * d;
        for (int i = threadIdx.x; i < heads * d; i += blockDim.x) {
            q[i] = iq[i];
        }
        __syncthreads();
        const float *cs = DP(const float, 6) + (size_t)j * rot, *sn = DP(const float, 7) + (size_t)j * rot;
        for (int h = 0; h < heads; ++h) {
            qsa_norm_rope_block(q + h * d, DP(const float, 4), d, df(r, e, 16), rot, cs, sn);
        }
    }
    const __half *blk = DP(const __half, 3);
    float scale = 1.f / sqrtf((float)d);
    /* one warp for each block key: the lanes split the values */
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int64_t b = SCORED ? nb : warp; b < nb; b += QSA_T / 32) {
        const __half *kb = blk + (size_t)b * d;
        float score = 0.f;
        for (int h = 0; h < heads; ++h) {
            float dot = 0.f;
            for (int i = lane; i < d; i += 32) {
                dot += q[h * d + i] * __half2float(kb[i]);
            }
            for (int o = 16; o > 0; o >>= 1) {
                dot += __shfl_xor_sync(0xffffffff, dot, o);
            }
            score += dot > 0.f ? dot : 0.f;
        }
        if (lane == 0) {
            score *= scale;
            keys[b] = ((uint64_t)__float_as_uint(score) << 32) | (uint64_t)b;
        }
    }
    __syncthreads();
    /* The budget-th largest key: 8 bits at a time from the top. */
    if (threadIdx.x == 0) {
        prefix_s = 0;
        need_s = budget;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            hist[i] = 0;
        }
        __syncthreads();
        uint64_t prefix = prefix_s, mask = shift == 56 ? 0 : ~0ull << (shift + 8);
        for (int64_t b = threadIdx.x; b < nb; b += blockDim.x) {
            uint64_t k = keys[b];
            if ((k & mask) == prefix) {
                atomicAdd(&hist[(k >> shift) & 255], 1);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int need = need_s, digit = 255;
            while (hist[digit] < need) {
                need -= hist[digit];
                --digit;
            }
            need_s = need;
            prefix_s = prefix | ((uint64_t)digit << shift);
        }
        __syncthreads();
    }
    uint64_t thr = prefix_s;            /* the keys >= thr are the budget best */
    int32_t *out = DP(int32_t, 17) + (size_t)j * maxsel;
    int base = 0;
    for (int64_t b0 = 0; b0 < nb; b0 += blockDim.x) {
        int64_t b = b0 + threadIdx.x;
        int keep = b < nb && keys[b] >= thr;
        int incl = qsa_scan(keep, tmp);
        if (keep) {
            for (int q2 = 0; q2 < ratio; ++q2) {
                out[(base + incl - 1) * ratio + q2] = (int32_t)(b * ratio + q2);
            }
        }
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) {
            tmp[0] = incl;
        }
        __syncthreads();
        base += tmp[0];
        __syncthreads();
    }
    int c2 = base * ratio;
    for (int64_t p = nb * ratio + threadIdx.x; p <= pj; p += blockDim.x) {
        out[c2 + (p - nb * ratio)] = (int32_t)p;
    }
    if (threadIdx.x == 0) {
        cnt[j] = (int32_t)(c2 + (pj + 1 - nb * ratio));
    }
}

/* GP_KQ_MULTI: x, cols, t, n, then (w, type, rows, out) for n matrices on
 * the same rows of x: GP_KQ_LINEAR of each, in one launch. */
__global__ void k_kq_multi(const gp_rec *r, const int64_t *e, int skip_bt = 0)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int n = DI(3), cols = DI(1), t = DI(2), m = 0;
    while (m < n && row >= DI(6 + 4 * m)) {
        row -= DI(6 + 4 * m);
        ++m;
    }
    if (m >= n) {
        return;
    }
    int type = DI(5 + 4 * m), rows = DI(6 + 4 * m);
    if (skip_bt && bt_shape(type, rows, cols)) {
        return;         /* k_kq_bf12_tc */
    }
    const uint8_t *w = DP(const uint8_t, 4 + 4 * m) + (size_t)row * kq_row_bytes(type, cols);
    for (int j = 0; j < t; ++j) {
        float v = kq_row(type, w, DP(const float, 0) + (size_t)j * cols, cols);
        if (threadIdx.x % 32 == 0) {
            DP(float, 7 + 4 * m)[(size_t)j * rows + row] = v;
        }
    }
}

/* ---------- bfloat16 rows for a small group (an MTP verify group) ----------
 * kq_row_part of KQ_BF16 for up to NT tokens: the lanes take the columns of
 * kq_row_part (8 lane + 256 part, step 256 nparts), and each chunk of 8
 * weights is loaded and converted once for all the tokens, not once for
 * each token. Each token adds its terms in the order of one token, and the
 * sums of the lanes in the same order: res[j] has the bits of kq_row_part
 * of that token (a verify group keeps the values of the steps). On the
 * matrices of Qwen3.8 (bf16, NP_GEMMA_DENSE=bf16) 4 tokens took about 1.8
 * times the time of one; NP_GEMMA_GPU_BF16_NT=0 keeps the loop over the
 * tokens. */
template <int NT>
__device__ __forceinline__ void kq_row_bf16_nt(int type, const uint8_t *w, const float *x, int cols,
                                               int t, int part, int nparts, float *res)
{
    int lane = threadIdx.x % 32;
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    const uint16_t *h = (const uint16_t *)w;
    /* KQ_BF12: the order of kq_row_part (kq_bf12_row) */
    if (type == KQ_BF12) {
        kq_bf12_row<NT>(w, x, cols, t, part, nparts, sum);
    }
    for (int c = 8 * lane + 256 * part; type == KQ_BF16 && c < cols; c += 256 * nparts) {
        uint4 q = *(const uint4 *)(h + c);
        uint32_t qq[4] = {q.x, q.y, q.z, q.w};
        float wv[8];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            wv[2 * u] = kq_bf16((uint16_t)(qq[u] & 0xffff));
            wv[2 * u + 1] = kq_bf16((uint16_t)(qq[u] >> 16));
        }
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                const float4 *x4 = (const float4 *)(x + (size_t)j * cols + c);
                float4 a = x4[0], b = x4[1];
                float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    sum[j] += wv[2 * u] * xv[2 * u] + wv[2 * u + 1] * xv[2 * u + 1];
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        res[j] = v;
    }
}

/* KQ_BF12 rows (n rows of cols values) to bfloat16 rows in dst: a thread
 * for each 16 values. */
__global__ void k_bf12_to_bf16(const uint8_t *w, int n, int cols, uint16_t *dst)
{
    PDL_START();
    int per = cols / 16;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * per) {
        return;
    }
    int64_t row = i / per;
    int c = (int)(i % per) * 16;
    uint32_t b[16];
    kq_bf12_bits<16>(w + (size_t)row * kq_row_bytes(KQ_BF12, cols), cols, c, b);
    uint4 *o = (uint4 *)(dst + (size_t)row * cols + c);
    o[0] = make_uint4(b[0] | b[1] << 16, b[2] | b[3] << 16, b[4] | b[5] << 16, b[6] | b[7] << 16);
    o[1] = make_uint4(b[8] | b[9] << 16, b[10] | b[11] << 16, b[12] | b[13] << 16, b[14] | b[15] << 16);
}

/* As gg_gemm_bf16 for KQ_BF12 rows: k_gemm_bf16_tc<X2, 1> (bf12_fused_on),
 * else chunks of rows (at most w16_n values, a multiple of T2N rows) to
 * bfloat16 rows in the scratch w16 (k_bf12_to_bf16), then gg_gemm_bf16. */
static void gg_gemm_bf12(const float *x, size_t ldx, const int *xrow0, const uint8_t *w, float *out,
                         size_t ldo, const int *orow0, int t, int rows, int cols, uint16_t *xb,
                         uint16_t *w16, size_t w16_n, cudaStream_t s)
{
    if (bf12_fused_on()) {
        gg_gemm_bf16(x, ldx, xrow0, (const uint16_t *)w, out, ldo, orow0, t, rows, cols, xb, s, 1);
        return;
    }
    int step = (int)(w16_n / (size_t)cols);
    step = step >= T2N ? step / T2N * T2N : step;
    size_t rb = kq_row_bytes(KQ_BF12, cols);
    for (int r0 = 0; r0 < rows; r0 += step) {
        int n = rows - r0 < step ? rows - r0 : step;
        k_bf12_to_bf16<<<(unsigned)(((int64_t)n * (cols / 16) + 255) / 256), 256, 0, s>>>(
            w + (size_t)r0 * rb, n, cols, w16);
        gg_gemm_bf16(x, ldx, xrow0, w16, out + r0, ldo, orow0, t, n, cols, xb, s);
    }
}

/* The types of the small-group kernels of bfloat16 rows (kq_row_bf16_nt). */
__host__ __device__ __forceinline__ int kq_bf_type(int type)
{
    return type == KQ_BF16 || type == KQ_BF12;
}

static int bf16_nt_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_BF16_NT");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* GP_KQ_LINEAR of a KQ_BF16 matrix for 2 to NT tokens (as k_kq_linear). */
template <int NT>
__global__ void k_kq_linear_bf16(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), t = DI(8);
    if (row >= rows) {
        return;
    }
    float res[NT];
    kq_row_bf16_nt<NT>(DI(5), DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(DI(5), cols),
                       DP(const float, 3), cols, t, 0, 1, res);
    if (threadIdx.x % 32 == 0) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                DP(float, 9)[(size_t)j * rows + row] = res[j];
            }
        }
    }
}

/* As k_kq_linear_split (KQ_SPLIT warps for each row) for 2 to NT tokens. */
template <int NT>
__global__ void __launch_bounds__(32 * KQ_SPLIT) k_kq_linear_split_bf16(const gp_rec *r,
                                                                        const int64_t *e)
{
    PDL_START();
    __shared__ float red[KQ_SPLIT][NT];
    int row = blockIdx.x, w = threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), t = DI(8);
    float res[NT];
    kq_row_bf16_nt<NT>(DI(5), DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(DI(5), cols),
                       DP(const float, 3), cols, t, w, KQ_SPLIT, res);
    if (threadIdx.x % 32 == 0) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            red[w][j] = res[j];
        }
    }
    __syncthreads();
    if (threadIdx.x < t) {
        float sm = 0.f;
        #pragma unroll
        for (int k = 0; k < KQ_SPLIT; ++k) {
            sm += red[k][threadIdx.x];
        }
        DP(float, 9)[(size_t)threadIdx.x * rows + row] = sm;
    }
}

/* GP_KQ_LINEAR of a KQ_BF12 matrix for 1 to NT tokens (as k_kq_linear,
 * kq_bf12_row: the bits of kq_row for each token). k_kq_linear holds the
 * registers of every type (144: 12 warps an SM); this kernel holds at most
 * 64 for a token (BF12_MINB). One token, the 3090 at its power cap: 10240
 * x 2560 59 us (bfloat16 68, Q8_R 46), 10240 x 320 14 us (23), the head
 * 248320 x 2560 2.5 ms (2.1); 2 to 4 tokens 1.3-1.8 times bfloat16 (the
 * decode is bound by the integer instructions at 500-900 MHz). Now for
 * NP_GEMMA_GPU_BF12_TC=0 and groups of 9 to 16 tokens: k_kq_bf12_tc takes
 * 1 to 8 (the same time for 1 and for 4 tokens, below bfloat16). */
#define BF12_MINB 8         /* blocks of KQ_RPB warps an SM for a token: at most 64 registers */
template <int NT>
__global__ void __launch_bounds__(32 * KQ_RPB, NT == 1 ? BF12_MINB : 1) k_kq_linear_bf12(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), t = DI(8);
    if (row >= rows) {
        return;
    }
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    kq_bf12_row<NT>(DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(KQ_BF12, cols), DP(const float, 3),
                    cols, t, 0, 1, sum);
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        if (threadIdx.x % 32 == 0 && j < t) {
            DP(float, 9)[(size_t)j * rows + row] = v;
        }
    }
}

/* ---------- KQ_BF12 steps and small groups on the tensor cores ----------
 * The CUDA-core kernels pay an FMA and a load of x for each value and each
 * token (2 to 4 tokens: 1.3-1.8 times bfloat16). Here mma m16n8k8 tf32 takes
 * 8 tokens at once: the cost of a group is the decode of the weights, as for
 * one token. The bfloat16 values of w are exact in tf32; x goes in as two
 * tf32 parts (hi = x with the low 13 bits cleared, lo = x - hi: about 22
 * bits of x, two instructions). The tokens past t are zeros, and a column
 * of the mma does not depend on the others: a token alone and in a group
 * get the same bits.
 *
 * A block: KW warps and a tile of 16 rows (bt_cls); warp w takes the steps
 * of 64 columns w, w + KW, ...; the warps add their sums in shared memory in
 * a fixed order (by the shape alone). In a step thread (g, c) takes 16 columns of each of its
 * rows, 16c .. 16c + 15 (one half of a BF12 group: one load of the lo
 * bytes, one of the gap bytes, the exponent), as the k slots {c, c + 4} of
 * the 8 k-steps (k-step q: columns 16c + 2q, 16c + 2q + 1); the thread
 * (g', c) of x takes the same columns of token g'. GP_KQ_LINEAR (m < 0) or
 * the matrices of GP_KQ_MULTI of bt_shape (the blocks one after another). */
__device__ __forceinline__ void mma1688_tf32(float *c, const uint32_t *a, const uint32_t *b)
{
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* The 16 values of a BF12 row from column c0 (16 | c0) as 8 words of two
 * bfloat16 values (kq_bf12_dec16), or zeros past the row. */
__device__ __forceinline__ void bt_row16(const uint8_t *w, int cols, int c0, uint32_t *p)
{
    if (c0 >= cols) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            p[i] = 0;
        }
        return;
    }
    int g = c0 >> 5;
    kq_bf12_dec16(*(const uint4 *)(w + c0), *(const uint4 *)(w + cols + 16 * g),
                  w[cols + cols / 2 + g], (c0 >> 4) & 1, p);
}

/* A hot day (the GPU and the CPUs throttled), 1 / 2 / 4 tokens in us,
 * k_kq_bf12_tc against the CUDA-core kernels (bfloat16 in brackets):
 * 10240 x 2560 102 103 106 / 102 109 167 (126 126 129); 2560 x 6144 67 69
 * 71 / 70 86 129 (82 83 86); 320 x 10240 28 27 28 / 26 35 45 (28 29 32);
 * 10240 x 320 27 27 28 / 24 28 38 (35 30 33); the head 2126 2137 2157 /
 * 2136 2123 2910 (2735 2749 2767). One kernel for 1 to 8 tokens keeps the
 * bits of a step in a verify group (the CUDA-core kernel for a token alone
 * would save 2-3 us on the two small shapes only).
 *
 * The kernels by the shape (bt_cls): 0 the most, KW 8 warps a block, a
 * tile of 16 rows (two tiles, x shared by 32 rows, spilled and had too few
 * blocks: 10240 x 2560 129 us in place of 102, the head 3.06 ms in place of
 * 2.13); 1 fewer than BT_MIN_ROWS rows: also BT_KP blocks for each tile,
 * each a part of the columns (320 x 10240 had 20 blocks: 53 us for 1 to 4
 * tokens, the CUDA-core kernels 27 to 45); 2 at most 512 columns: KW 4 (8
 * steps or fewer of 64 columns: idle warps). */
#define BT_KP 8
__host__ __device__ __forceinline__ int bt_cls(int rows, int cols)
{
    return rows < BT_MIN_ROWS ? 1 : cols <= 512 ? 2 : 0;
}

/* A block of class 1 adds its sums of its part of the columns (part
 * blockIdx.y) to part[tile][y][128]; the last block of a tile (cnt) adds the
 * parts in their order (the same bits whichever block is last), writes out,
 * and sets cnt back to 0. */
template <int KW>
__global__ void __launch_bounds__(32 * KW, 3) k_kq_bf12_tc(const gp_rec *r, const int64_t *e, int multi,
                                                        int cls, float *part, int *cnt)
{
    PDL_START();
    __shared__ float red[KW][4][32];
    __shared__ int last;
    /* the matrix of this block */
    int blk = blockIdx.x, rows, cols, t;
    const uint8_t *w;
    const float *x;
    float *out;
    if (!multi) {
        rows = DI(6), cols = DI(7), t = DI(8);
        w = DP(const uint8_t, 4), x = DP(const float, 3), out = DP(float, 9);
    } else {
        int n = DI(3), m = 0;
        cols = DI(1), t = DI(2);
        for (;; ++m) {
            if (m >= n) {
                return;
            }
            int rm = DI(6 + 4 * m);
            if (!bt_shape(DI(5 + 4 * m), rm, cols) || bt_cls(rm, cols) != cls) {
                continue;
            }
            int nb = (rm + 15) / 16;
            if (blk < nb) {
                break;
            }
            blk -= nb;
        }
        rows = DI(6 + 4 * m);
        w = DP(const uint8_t, 4 + 4 * m), x = DP(const float, 0), out = DP(float, 7 + 4 * m);
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    int row0 = blk * 16, kp = gridDim.y;
    size_t rb = kq_row_bytes(KQ_BF12, cols);
    const uint8_t *w0 = w + (size_t)min(row0 + g, rows - 1) * rb;
    const uint8_t *w8 = w + (size_t)min(row0 + g + 8, rows - 1) * rb;
    const float *xg = g < t ? x + (size_t)g * cols : NULL;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int st = blockIdx.y * KW + warp; st * 64 < cols; st += KW * kp) {
        int c0 = st * 64 + 16 * c;
        float xr[16];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float4 v = xg != NULL && c0 < cols ? *(const float4 *)(xg + c0 + 4 * u)
                                               : make_float4(0.f, 0.f, 0.f, 0.f);
            xr[4 * u] = v.x, xr[4 * u + 1] = v.y, xr[4 * u + 2] = v.z, xr[4 * u + 3] = v.w;
        }
        uint32_t p0[8], p8[8];
        bt_row16(w0, cols, c0, p0);
        bt_row16(w8, cols, c0, p8);
        /* the sums of a step in the mma, then added in float32 (the mma
         * alone: 1.2e-5 of the largest output for rows of 8192 values
         * against float64; this way 6e-7, as the CUDA-core kernels) */
        float sa[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            /* x as tf32 hi + lo; k-step q: the values 2q (slot c) and 2q + 1 (slot c + 4) */
            uint32_t h0 = __float_as_uint(xr[2 * q]) & 0xffffe000u;
            uint32_t h1 = __float_as_uint(xr[2 * q + 1]) & 0xffffe000u;
            uint32_t bh[2] = {h0, h1};
            uint32_t bl[2] = {__float_as_uint(xr[2 * q] - __uint_as_float(h0)),
                              __float_as_uint(xr[2 * q + 1] - __uint_as_float(h1))};
            uint32_t a[4] = {p0[q] << 16, p8[q] << 16, p0[q] & 0xffff0000u, p8[q] & 0xffff0000u};
            mma1688_tf32(sa, a, bh);
            mma1688_tf32(sa, a, bl);
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc[k] += sa[k];
        }
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        red[warp][k][lane] = acc[k];
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
    float v[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = red[0][k][lane];
        #pragma unroll
        for (int ww = 1; ww < KW; ++ww) {
            v[k] += red[ww][k][lane];
        }
    }
    if (kp > 1) {
        float *pt = part + (size_t)blockIdx.x * kp * 128;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            pt[(size_t)blockIdx.y * 128 + 4 * lane + k] = v[k];
        }
        __threadfence();
        if (lane == 0) {
            last = atomicAdd(cnt + blockIdx.x, 1) == kp - 1;
        }
        __syncwarp();
        if (!last) {
            return;
        }
        __threadfence();
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = 0.f;
            for (int y = 0; y < kp; ++y) {
                v[k] += __ldcg(pt + (size_t)y * 128 + 4 * lane + k);
            }
        }
        if (lane == 0) {
            cnt[blockIdx.x] = 0;
        }
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        /* c0: (row g, token 2c), c1: (g, 2c + 1), c2: (g + 8, 2c), c3: (g + 8, 2c + 1) */
        int row = row0 + g + (k >= 2 ? 8 : 0), tok = 2 * c + (k & 1);
        if (row < rows && tok < t) {
            out[(size_t)tok * rows + row] = v[k];
        }
    }
}

/* Launch k_kq_bf12_tc for the matrices of class cls (blocks: their tiles). */
static void bt_launch(const gp_rec *dr, const int64_t *denv, int multi, int cls, int64_t blocks,
                      float *part, int *cnt, cudaStream_t s)
{
    if (cls == 1) {
        k_kq_bf12_tc<8><<<dim3((unsigned)blocks, BT_KP), 256, 0, s>>>(dr, denv, multi, 1, part, cnt);
    } else if (cls == 2) {
        k_kq_bf12_tc<4><<<(unsigned)blocks, 128, 0, s>>>(dr, denv, multi, 2, part, cnt);
    } else {
        k_kq_bf12_tc<8><<<(unsigned)blocks, 256, 0, s>>>(dr, denv, multi, 0, part, cnt);
    }
}

/* NP_GEMMA_GPU_BF12_TC=0: the CUDA-core kernels for the steps and small
 * groups of BF12 (k_kq_linear_bf12, k_kq_multi_bf16), for a test. */
static int bt_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_BF12_TC");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* GP_KQ_MULTI with KQ_BF16 (or KQ_BF12) matrices for 2 to NT tokens (as k_kq_multi); a
 * matrix of another type (the float32 alpha and beta of a DeltaNet layer)
 * takes kq_row for each token, as k_kq_multi. */
template <int NT>
__global__ void k_kq_multi_bf16(const gp_rec *r, const int64_t *e, int skip_bt = 0)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int n = DI(3), cols = DI(1), t = DI(2), m = 0;
    while (m < n && row >= DI(6 + 4 * m)) {
        row -= DI(6 + 4 * m);
        ++m;
    }
    if (m >= n) {
        return;
    }
    int rows = DI(6 + 4 * m), type = DI(5 + 4 * m);
    if (skip_bt && bt_shape(type, rows, cols)) {
        return;         /* k_kq_bf12_tc */
    }
    if (!kq_bf_type(type)) {
        const uint8_t *w = DP(const uint8_t, 4 + 4 * m) + (size_t)row * kq_row_bytes(type, cols);
        for (int j = 0; j < t; ++j) {
            float v = kq_row(type, w, DP(const float, 0) + (size_t)j * cols, cols);
            if (threadIdx.x % 32 == 0) {
                DP(float, 7 + 4 * m)[(size_t)j * rows + row] = v;
            }
        }
        return;
    }
    float res[NT];
    kq_row_bf16_nt<NT>(type, DP(const uint8_t, 4 + 4 * m) + (size_t)row * kq_row_bytes(type, cols),
                       DP(const float, 0), cols, t, 0, 1, res);
    if (threadIdx.x % 32 == 0) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                DP(float, 7 + 4 * m)[(size_t)j * rows + row] = res[j];
            }
        }
    }
}

/* ---------- the products of a step and of a small group with int8 x ----------
 * x as int8 with a scale for each 32 values (k_kq_quant_x, one time for a
 * record), and __dp4a on the int8 values of Q8_R: a lane reads 16 bytes of w
 * and 16 of x (not 64 of float32 x) and makes 4 dp4a and one float fma for
 * 16 values. The other types read x as before. A token alone and in a group
 * run the same code: the same bits. NP_GEMMA_GPU_I8X=0: float32 x for all. */
/* The types of kq_row_i8 (the others read x as float32). */
__host__ __device__ __forceinline__ int kq_i8_type(int type)
{
    return type == KQ_Q8_R || type == KQ_Q8_0 || type == KQ_Q4_K || type == KQ_Q5_K ||
           type == KQ_Q6_K || type == KQ_Q4_0;
}

/* kq_row_i8 for NT tokens (t <= NT of them live): the x of token j is at
 * x + j cols, xq + j cols, and xs + j cols / 32. The kernel unpacks each
 * weight word once for all the tokens. Each token adds its terms in the
 * order of one token, so res[j] has the bits of kq_row_i8 of that token (a
 * verify group of MTP gets the values of the decode steps). */
template <int NT, bool HALF = false>
__device__ void kq_rows_i8(int type, const uint8_t *w, const float *x, const int8_t *xq,
                           const float *xs, int cols, int t, float *res)
{
    if (!kq_i8_type(type) || xq == NULL) {
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                res[j] = kq_row(type, w, x + (size_t)j * cols, cols);
            }
        }
        return;
    }
    /* HALF: a row for each 16 lanes (the lanes 0 to 15 of the row, and the
     * strides of half a warp); k_kq_glu_i8. */
    int lane = threadIdx.x % (HALF ? 16 : 32), np = cols / 32;
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    if (type == KQ_Q8_0) {
        /* lane l: 16 values (half l % 2) of block l / 2 of the 16 blocks of a
         * step. The blocks are 34 bytes, so the loads take 2 bytes. */
        int hf = lane & 1;
        for (int b = lane >> 1; b < cols / 32; b += HALF ? 8 : 16) {
            const uint8_t *blk = w + (size_t)b * 34;
            const uint16_t *q = (const uint16_t *)(blk + 2 + 16 * hf);
            uint32_t qw[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                qw[k] = (uint32_t)q[2 * k] | ((uint32_t)q[2 * k + 1] << 16);
            }
            float d = kq_half(blk);
            #pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
                if (jt < t) {
                    uint4 xv = *(const uint4 *)(xq + (size_t)jt * cols + b * 32 + 16 * hf);
                    int si = __dp4a((int)qw[0], (int)xv.x, 0);
                    si = __dp4a((int)qw[1], (int)xv.y, si);
                    si = __dp4a((int)qw[2], (int)xv.z, si);
                    si = __dp4a((int)qw[3], (int)xv.w, si);
                    sum[jt] += (float)si * (d * xs[jt * np + b]);
                }
            }
        }
    } else if (type == KQ_Q4_0) {
        /* As Q8_0: lane l takes the 16 values hf = l % 2 of block l / 2 (the
         * low 4 bits of the 16 bytes for hf 0, the high 4 bits for hf 1).
         * dp4a of the codes q (0 to 15) and x, minus 8 times the sum of x:
         * the sum of (q - 8) x. */
        int hf = lane & 1;
        for (int b = lane >> 1; b < cols / 32; b += HALF ? 8 : 16) {
            const uint8_t *blk = w + (size_t)b * 18;
            const uint16_t *q = (const uint16_t *)(blk + 2);
            uint32_t qw[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                qw[k] = (((uint32_t)q[2 * k] | ((uint32_t)q[2 * k + 1] << 16)) >> (4 * hf)) &
                        0x0F0F0F0Fu;
            }
            float d = kq_half(blk);
            #pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
                if (jt < t) {
                    uint4 xv = *(const uint4 *)(xq + (size_t)jt * cols + b * 32 + 16 * hf);
                    int si = __dp4a((int)qw[0], (int)xv.x, 0);
                    si = __dp4a((int)qw[1], (int)xv.y, si);
                    si = __dp4a((int)qw[2], (int)xv.z, si);
                    si = __dp4a((int)qw[3], (int)xv.w, si);
                    int sx = __dp4a(0x01010101, (int)xv.x, 0);
                    sx = __dp4a(0x01010101, (int)xv.y, sx);
                    sx = __dp4a(0x01010101, (int)xv.z, sx);
                    sx = __dp4a(0x01010101, (int)xv.w, sx);
                    sum[jt] += (float)(si - 8 * sx) * (d * xs[jt * np + b]);
                }
            }
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        /* lane l = 8 sb + 2 c + hf: superblock 4 i + sb of step i, the 16
         * bytes hf of the 32 bytes c of the quants (16 values of part 2 c in
         * the low 4 bits, 16 of part 2 c + 1 in the high 4 bits). A warp
         * reads 4 superblocks for each step, 16 bytes for each lane. The
         * mins take the sum of the int8 x (dp4a with ones). */
        int five = type == KQ_Q5_K, hf = lane & 1, c = (lane >> 1) & 3, sbk = lane >> 3;
        int nbk = cols / 256;
        size_t bs = five ? 176 : 144;
        for (int b = sbk; b < nbk; b += HALF ? 2 : 4) {
            const uint8_t *blk = w + (size_t)b * bs;
            uint4 q = *(const uint4 *)(blk + (five ? 48 : 16) + 32 * c + 16 * hf);
            uint32_t qv[4] = {q.x, q.y, q.z, q.w}, lo[4], hi[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                lo[k] = qv[k] & 0x0f0f0f0fu;
                hi[k] = (qv[k] >> 4) & 0x0f0f0f0fu;
            }
            if (five) {
                uint4 h = *(const uint4 *)(blk + 16 + 16 * hf);
                uint32_t hv[4] = {h.x, h.y, h.z, h.w};
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    lo[k] |= ((hv[k] >> (2 * c)) & 0x01010101u) << 4;
                    hi[k] |= ((hv[k] >> (2 * c + 1)) & 0x01010101u) << 4;
                }
            }
            int s0, m0, s1, m1;
            kq_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            float d = kq_half(blk), dm = kq_half(blk + 2);
            int base = b * 256 + 64 * c + 16 * hf;
            #pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
                if (jt < t) {
                    const int8_t *xj = xq + (size_t)jt * cols;
                    uint4 xl = *(const uint4 *)(xj + base), xh = *(const uint4 *)(xj + base + 32);
                    float sl = xs[jt * np + base / 32], sh = xs[jt * np + base / 32 + 1];
                    int dl = __dp4a((int)lo[0], (int)xl.x, 0);
                    dl = __dp4a((int)lo[1], (int)xl.y, dl);
                    dl = __dp4a((int)lo[2], (int)xl.z, dl);
                    dl = __dp4a((int)lo[3], (int)xl.w, dl);
                    int dh = __dp4a((int)hi[0], (int)xh.x, 0);
                    dh = __dp4a((int)hi[1], (int)xh.y, dh);
                    dh = __dp4a((int)hi[2], (int)xh.z, dh);
                    dh = __dp4a((int)hi[3], (int)xh.w, dh);
                    int nl = __dp4a(0x01010101, (int)xl.x, 0);
                    nl = __dp4a(0x01010101, (int)xl.y, nl);
                    nl = __dp4a(0x01010101, (int)xl.z, nl);
                    nl = __dp4a(0x01010101, (int)xl.w, nl);
                    int nh = __dp4a(0x01010101, (int)xh.x, 0);
                    nh = __dp4a(0x01010101, (int)xh.y, nh);
                    nh = __dp4a(0x01010101, (int)xh.z, nh);
                    nh = __dp4a(0x01010101, (int)xh.w, nh);
                    sum[jt] += d * ((float)(s0 * dl) * sl + (float)(s1 * dh) * sh) -
                               dm * ((float)(m0 * nl) * sl + (float)(m1 * nh) * sh);
                }
            }
        }
    } else if (type == KQ_Q6_K) {
        /* lane l = 8 sb + 4 n + 2 u + hf: superblock 4 i + sb of step i,
         * half n, the 16 values hf of the quarters u (the low 4 bits of ql)
         * and u + 2 (the high 4 bits), with 2 bits of qh each. The code is
         * that value minus 32. The blocks are 210 bytes: 2-byte loads. */
        int hf = lane & 1, u = (lane >> 1) & 1, n = (lane >> 2) & 1, sbk = lane >> 3;
        for (int b = sbk; b < cols / 256; b += HALF ? 2 : 4) {
            const uint8_t *ql = w + (size_t)b * 210;
            const uint16_t *pl = (const uint16_t *)(ql + 64 * n + 32 * u + 16 * hf);
            const uint16_t *ph = (const uint16_t *)(ql + 128 + 32 * n + 16 * hf);
            const int8_t *sc = (const int8_t *)(ql + 192);
            float d = kq_half(ql + 208);
            uint32_t cl[4], ch[4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                uint32_t lq = (uint32_t)pl[2 * k] | ((uint32_t)pl[2 * k + 1] << 16);
                uint32_t hq = (uint32_t)ph[2 * k] | ((uint32_t)ph[2 * k + 1] << 16);
                cl[k] = __vsub4((lq & 0x0f0f0f0fu) | (((hq >> (2 * u)) & 0x03030303u) << 4),
                                0x20202020u);
                ch[k] = __vsub4(((lq >> 4) & 0x0f0f0f0fu) |
                                (((hq >> (2 * u + 4)) & 0x03030303u) << 4), 0x20202020u);
            }
            float fl = d * (float)sc[8 * n + hf + 2 * u], fh = d * (float)sc[8 * n + hf + 2 * u + 4];
            int base = b * 256 + 128 * n + 32 * u + 16 * hf;
            #pragma unroll
            for (int jt = 0; jt < NT; ++jt) {
                if (jt < t) {
                    const int8_t *xj = xq + (size_t)jt * cols;
                    uint4 xl = *(const uint4 *)(xj + base), xh = *(const uint4 *)(xj + base + 64);
                    int dl = __dp4a((int)cl[0], (int)xl.x, 0);
                    dl = __dp4a((int)cl[1], (int)xl.y, dl);
                    dl = __dp4a((int)cl[2], (int)xl.z, dl);
                    dl = __dp4a((int)cl[3], (int)xl.w, dl);
                    int dh = __dp4a((int)ch[0], (int)xh.x, 0);
                    dh = __dp4a((int)ch[1], (int)xh.y, dh);
                    dh = __dp4a((int)ch[2], (int)xh.z, dh);
                    dh = __dp4a((int)ch[3], (int)xh.w, dh);
                    sum[jt] += fl * xs[jt * np + base / 32] * (float)dl +
                               fh * xs[jt * np + base / 32 + 2] * (float)dh;
                }
            }
        }
    } else {
        const __half *d = (const __half *)(w + cols);
        for (int c = 16 * lane; c < cols; c += HALF ? 256 : 512) {
            uint4 q = *(const uint4 *)(w + c);
            float dc = __half2float(d[c / 32]);
            #pragma unroll
            for (int j = 0; j < NT; ++j) {
                if (j < t) {
                    uint4 xv = *(const uint4 *)(xq + (size_t)j * cols + c);
                    int si = __dp4a((int)q.x, (int)xv.x, 0);
                    si = __dp4a((int)q.y, (int)xv.y, si);
                    si = __dp4a((int)q.z, (int)xv.z, si);
                    si = __dp4a((int)q.w, (int)xv.w, si);
                    sum[j] += (float)si * (dc * xs[j * np + c / 32]);
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float v = sum[j];
        for (int o = HALF ? 8 : 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(0xffffffff, v, o);
        }
        res[j] = v;
    }
}

__device__ float kq_row_i8(int type, const uint8_t *w, const float *x, const int8_t *xq,
                           const float *xs, int cols)
{
    float v;
    kq_rows_i8<1>(type, w, x, xq, xs, cols, 1, &v);
    return v;
}

/* The products of row w on t tokens, NT tokens at a time (kq_rows_i8). A
 * kernel has one NT, so a step (NT 1) keeps the registers of one token. */
#define KQ_NT 4

template <int NT>
__device__ void kq_row_i8_t(int type, const uint8_t *w, const float *x, const int8_t *xq,
                            const float *xs, int cols, int t, float *out, size_t ostr)
{
    for (int j0 = 0; j0 < t; j0 += NT) {
        float res[NT];
        int n = t - j0 < NT ? t - j0 : NT;
        kq_rows_i8<NT>(type, w, x + (size_t)j0 * cols, xq + (size_t)j0 * cols,
                       xs + (size_t)j0 * (cols / 32), cols, n, res);
        if (threadIdx.x % 32 == 0) {
            for (int j = 0; j < n; ++j) {
                out[(size_t)(j0 + j) * ostr] = res[j];
            }
        }
    }
}

/* kq_row_i8_t for a group: each part of up to KQ_NT tokens takes the
 * kq_rows_i8 of its count (NT 0 in the kernels). */
__device__ void kq_row_i8_g(int type, const uint8_t *w, const float *x, const int8_t *xq,
                            const float *xs, int cols, int t, float *out, size_t ostr)
{
    for (int j0 = 0; j0 < t; j0 += KQ_NT) {
        float res[KQ_NT];
        int n = t - j0 < KQ_NT ? t - j0 : KQ_NT;
        const float *xj = x + (size_t)j0 * cols;
        const int8_t *qj = xq + (size_t)j0 * cols;
        const float *sj = xs + (size_t)j0 * (cols / 32);
        if (n == 1) {
            kq_rows_i8<1>(type, w, xj, qj, sj, cols, 1, res);
        } else if (n == 2) {
            kq_rows_i8<2>(type, w, xj, qj, sj, cols, 2, res);
        } else if (n == 3) {
            kq_rows_i8<3>(type, w, xj, qj, sj, cols, 3, res);
        } else {
            /* n == KQ_NT. A constant count lets the compiler drop the tests
             * of the live tokens (a runtime count made a group of 3 or 4
             * 1.5 to 2.5 times slower). */
            kq_rows_i8<KQ_NT>(type, w, xj, qj, sj, cols, KQ_NT, res);
        }
        if (threadIdx.x % 32 == 0) {
            for (int j = 0; j < n; ++j) {
                out[(size_t)(j0 + j) * ostr] = res[j];
            }
        }
    }
}



/* GP_KQ_LINEAR (t <= MT_MAX) with the int8 x of the scratch. FT >= 0 fixes
 * the type when the kernel compiles (KQ_Q4_0): the compiler then drops the
 * code of the other types, and the kernel takes fewer registers (the
 * general form took 92 to 110). The values are the same. */
template <int NT, int FT = -1>
__global__ void k_kq_linear_i8(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), type = FT >= 0 ? FT : DI(5), t = DI(8);
    if (row >= rows) {
        return;
    }
    const uint8_t *w = DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(type, cols);
    if (NT == 0) {
        kq_row_i8_g(type, w, DP(const float, 3), xq, xs, cols, t, DP(float, 9) + row,
                    (size_t)rows);
    } else {
        kq_row_i8_t<NT == 0 ? 1 : NT>(type, w, DP(const float, 3), xq, xs, cols, t,
                                       DP(float, 9) + row, (size_t)rows);
    }
}

/* GP_KQ_MULTI of the gate and the up matrix (the same rows), and the
 * GP_GELU_MUL_ROWS after it, in one kernel: out[j][r] = gelu(g) u, as
 * k_gelu_mul_rows. The lanes 0 to 15 of warp r take row r of the gate, the
 * lanes 16 to 31 row r of the up matrix (kq_rows_i8 HALF); the two halves
 * read the same x. NT 0: a group (the counts 1 to KQ_NT, as kq_row_i8_g). */
template <int NT, int FT = -1>
__global__ void k_kq_glu_i8(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs,
                            float *out)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32, hs = (threadIdx.x / 16) % 2;
    int cols = DI(1), t = DI(2), rows = DI(6);
    if (row >= rows) {
        return;
    }
    int type = FT >= 0 ? FT : DI(5 + 4 * hs);
    const uint8_t *w = DP(const uint8_t, 4 + 4 * hs) + (size_t)row * kq_row_bytes(type, cols);
    const float *x = DP(const float, 0);
    for (int j0 = 0; j0 < t; j0 += (NT == 0 ? KQ_NT : NT)) {
        float res[KQ_NT];
        int n = t - j0 < (NT == 0 ? KQ_NT : NT) ? t - j0 : (NT == 0 ? KQ_NT : NT);
        const float *xj = x + (size_t)j0 * cols;
        const int8_t *qj = xq + (size_t)j0 * cols;
        const float *sj = xs + (size_t)j0 * (cols / 32);
        if (NT == 1 || n == 1) {
            kq_rows_i8<1, true>(type, w, xj, qj, sj, cols, 1, res);
        } else if (n == 2) {
            kq_rows_i8<2, true>(type, w, xj, qj, sj, cols, 2, res);
        } else if (n == 3) {
            kq_rows_i8<3, true>(type, w, xj, qj, sj, cols, 3, res);
        } else {
            kq_rows_i8<KQ_NT, true>(type, w, xj, qj, sj, cols, KQ_NT, res);
        }
        for (int j = 0; j < n; ++j) {
            float gv = __shfl_sync(0xffffffff, res[j], 0), uv = __shfl_sync(0xffffffff, res[j], 16);
            if (threadIdx.x % 32 == 0) {
                out[(size_t)(j0 + j) * rows + row] =
                    0.5f * gv * (1.0f + tanhf(0.7978845608028654f * (gv + 0.044715f * gv * gv * gv)))
                    * uv;
            }
        }
    }
}

/* GP_KQ_MULTI with the int8 x of the scratch (FT as k_kq_linear_i8). */
template <int NT, int FT = -1>
__global__ void k_kq_multi_i8(const gp_rec *r, const int64_t *e, const int8_t *xq, const float *xs)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int n = DI(3), cols = DI(1), t = DI(2), m = 0;
    while (m < n && row >= DI(6 + 4 * m)) {
        row -= DI(6 + 4 * m);
        ++m;
    }
    if (m >= n) {
        return;
    }
    int type = FT >= 0 ? FT : DI(5 + 4 * m), rows = DI(6 + 4 * m);
    const uint8_t *w = DP(const uint8_t, 4 + 4 * m) + (size_t)row * kq_row_bytes(type, cols);
    if (NT == 0) {
        kq_row_i8_g(type, w, DP(const float, 0), xq, xs, cols, t, DP(float, 7 + 4 * m) + row,
                    (size_t)rows);
    } else {
        kq_row_i8_t<NT == 0 ? 1 : NT>(type, w, DP(const float, 0), xq, xs, cols, t,
                                       DP(float, 7 + 4 * m) + row, (size_t)rows);
    }
}

/* 8 values of a row from column c0 (a multiple of 8), as float32. */
__device__ void kq_dequant8(int type, const uint8_t *w, int cols, int c0, float *v)
{
    if (type == KQ_F32) {
        float4 a = *(const float4 *)((const float *)w + c0);
        float4 b = *(const float4 *)((const float *)w + c0 + 4);
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
        v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
    } else if (type == KQ_Q8_R) {
        float d = __half2float(((const __half *)(w + cols))[c0 / 32]);
        uint2 q = *(const uint2 *)(w + c0);
        for (int u = 0; u < 4; ++u) {
            v[u] = d * (float)(int8_t)((q.x >> (8 * u)) & 255);
            v[4 + u] = d * (float)(int8_t)((q.y >> (8 * u)) & 255);
        }
    } else if (type == KQ_Q8_0) {
        const uint8_t *blk = w + (size_t)(c0 / 32) * 34;
        float d = kq_half(blk);
        for (int u = 0; u < 8; ++u) {
            v[u] = d * (float)(int8_t)blk[2 + c0 % 32 + u];
        }
    } else if (type == KQ_Q4_0) {
        const uint8_t *blk = w + (size_t)(c0 / 32) * 18;
        float d = kq_half(blk);
        int j = c0 % 32, sh = j < 16 ? 0 : 4;
        for (int u = 0; u < 8; ++u) {
            v[u] = d * (float)((int)((blk[2 + j % 16 + u] >> sh) & 15) - 8);
        }
    } else if (type == KQ_BF12) {
        uint32_t bits[8];
        kq_bf12_bits<8>(w, cols, c0, bits);
        for (int u = 0; u < 8; ++u) {
            v[u] = __uint_as_float(bits[u] << 16);
        }
    } else if (type == KQ_BF16) {
        uint4 q = *(const uint4 *)((const uint16_t *)w + c0);
        uint32_t qq[4] = {q.x, q.y, q.z, q.w};
        for (int u = 0; u < 4; ++u) {
            v[2 * u] = kq_bf16((uint16_t)(qq[u] & 0xffff));
            v[2 * u + 1] = kq_bf16((uint16_t)(qq[u] >> 16));
        }
    } else if (type == KQ_NV4) {
        float g = *(const float *)(w + cols / 2 + cols / 16);
        int b = c0 / 32, j = c0 % 32, hi = j >= 16;
        float sc = 0.5f * g * kq_e4m3(w[cols / 2 + 2 * b + hi]);
        const uint8_t *q = w + 16 * b + (j % 16);
        for (int u = 0; u < 8; ++u) {
            v[u] = sc * kq_e2m1x2((q[u] >> (hi ? 4 : 0)) & 15);
        }
    } else if (type == KQ_Q5_1) {
        const uint8_t *blk = w + (size_t)(c0 / 32) * 24;
        float d = kq_half(blk), m = kq_half(blk + 2);
        uint32_t qh = *(const uint32_t *)(blk + 4);
        for (int u = 0; u < 8; ++u) {
            int j = c0 % 32 + u;
            int q = j < 16 ? blk[8 + j] & 15 : blk[8 + j - 16] >> 4;
            v[u] = d * (float)(q | (((qh >> j) & 1) << 4)) + m;
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        int five = type == KQ_Q5_K;
        const uint8_t *blk = w + (size_t)(c0 / 256) * (five ? 176 : 144);
        int q = c0 % 256, c = q / 64, off = q % 64, hi = off >= 32, l = off % 32;
        int j = 2 * c + hi, sc, m;
        kq_scale_min(blk + 4, j, &sc, &m);
        float d = kq_half(blk) * (float)sc, dm = kq_half(blk + 2) * (float)m;
        const uint8_t *qs = blk + (five ? 48 : 16) + 32 * c + l;
        for (int u = 0; u < 8; ++u) {
            int nib = hi ? qs[u] >> 4 : qs[u] & 15;
            if (five) {
                nib |= ((blk[16 + l + u] >> j) & 1) << 4;
            }
            v[u] = d * (float)nib - dm;
        }
    } else {
        const uint8_t *blk = w + (size_t)(c0 / 256) * 210;
        int q = c0 % 256, h = q / 128, rr = q % 128, u4 = rr / 32, l = rr % 32;
        const uint8_t *ql = blk + 64 * h + ((u4 & 1) ? 32 : 0) + l;
        const uint8_t *qh = blk + 128 + 32 * h + l;
        float d = kq_half(blk + 208) * (float)((const int8_t *)(blk + 192))[8 * h + l / 16 + 2 * u4];
        for (int u = 0; u < 8; ++u) {
            int lo = u4 >= 2 ? ql[u] >> 4 : ql[u] & 15;
            v[u] = d * (float)((lo | (((qh[u] >> (2 * u4)) & 3) << 4)) - 32);
        }
    }
}

/* A tile of the product of a group: rows r0 .. r0 + 63 of W (nrows rows,
 * rb bytes each) on n rows of x (row j at X + xmap[j] * xstride, or
 * X + j * xstride). O[j * ostride + r] gets row r, token j. 256 threads;
 * each computes 4 rows by 4 tokens. The weights come to shared memory as
 * float32, 32 columns at a time. */
#define KG_B 64
#define KG_K 32
__device__ void kq_tile(int type, const uint8_t *W, size_t rb, int nrows, int r0, int cols,
                        const float *X, size_t xstride, const int *xmap, int n, float *O,
                        size_t ostride)
{
    __shared__ __align__(16) float Ws[KG_K][KG_B + 4];
    __shared__ __align__(16) float Xs[KG_K][KG_B + 4];
    int tid = threadIdx.x, ty = tid / 16, tx = tid % 16;
    int lr = tid / 4, lc = (tid % 4) * 8;
    const uint8_t *wrow = r0 + lr < nrows ? W + (size_t)(r0 + lr) * rb : NULL;
    const float *xrow = lr < n ? X + (size_t)(xmap ? xmap[lr] : lr) * xstride : NULL;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int k0 = 0; k0 < cols; k0 += KG_K) {
        float v[8];
        if (wrow != NULL && type == KQ_NVX) {
            int row = r0 + lr;
            kq_dequant8_nvx(W + (size_t)(row & ~15) * rb, row & 15, k0 + lc, v);
        } else if (wrow != NULL) {
            kq_dequant8(type, wrow, cols, k0 + lc, v);
        } else {
            for (int u = 0; u < 8; ++u) {
                v[u] = 0.f;
            }
        }
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            Ws[lc + u][lr] = v[u];
        }
        if (xrow != NULL) {
            float4 a = *(const float4 *)(xrow + k0 + lc), b = *(const float4 *)(xrow + k0 + lc + 4);
            v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
            v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
        } else {
            for (int u = 0; u < 8; ++u) {
                v[u] = 0.f;
            }
        }
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            Xs[lc + u][lr] = v[u];
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < KG_K; ++kk) {
            float4 a = *(const float4 *)&Ws[kk][ty * 4];
            float4 b = *(const float4 *)&Xs[kk][tx * 4];
            float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += av[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int rr = r0 + ty * 4 + i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int jj = tx * 4 + j;
            if (rr < nrows && jj < n) {
                O[(size_t)jj * ostride + rr] = acc[i][j];
            }
        }
    }
}


/* ---------- the products of large groups on the tensor cores (Qwen3.8) ----------
 * QWEN38_PLAN.md, phase 5. As k_gemm_q8: x as int8 with a scale for each 32
 * values (kq_quant_x: xs, and xsum = xs * the sum of the 32 int8 values),
 * and mma.sync m16n8k32 on the int8 values of 32 columns. Each format of w
 * gives, for each block of 32 values of a row, int8 values q, a scale d and
 * a term mn with w = d q + (the same mn for all 32 values). So
 *
 *     sum over the block of w x = xs d (the int32 sum of q x) + mn xsum.
 *
 *     Q8_R   the rows of the GPU (int8 values, then float16 scales): mn = 0
 *     Q8_0   blocks of 34 bytes: mn = 0
 *     Q5_1   w = d q + m (q of 5 bits): mn = m
 *     Q4_K   w = d sc q - dmin mb (sub-blocks of 32 of a block of 256)
 *
 * A tile is BM rows of x (tokens or pairs) by 128 rows of w, with 8 warps;
 * the steps of 128 columns come to shared memory by cp.async in NS
 * buffers. A step of a row of w holds its bytes for the 128 columns (the
 * head of the block of 256 and 64 bytes of values for Q4_K). */
#define KT_BN 128
#define KT_K 128
#define KT_RB 144                           /* the bytes of a row of w in a step (136 used) */
enum { KT_Q8R = 0, KT_Q80 = 1, KT_Q51 = 2, KT_Q4K = 3, KT_NV4 = 4, KT_NVX = 5, KT_Q4X = 6, KT_Q6K = 7 };
/* KT_Q6K (the RQ6_K experts, RQ6_MIX_PLAN.md): a step of 128 columns is half
 * a Q6_K block; a row of a step holds its 64 bytes of low 4 bits, 32 of high
 * 2 bits, 8 int8 scales, and d (106 bytes). Block bk of the step: the
 * values - 32 as int8, a scale d sc for each 16 (the sums of the two halves,
 * as NVFP4). The blocks of 210 bytes are only 2-aligned: plain loads. */
/* KT_Q4X (the KQ_Q4X experts of the 26B): 8 groups of the 4 blocks of the
 * step, no head. cols can end in a part of a step (704: 22 blocks). */
#define KT_GB4 (4 * 288)
/* KT_NVX: a step of a tile holds the 8 groups of its 128 rows, each the 16
 * bytes of its head and the 4 blocks of the step (KT_GB bytes). */
#define KT_GB (16 + 4 * KQ_NVX_BB)

/* mma.sync m16n8k16 on int8 values: the two halves of a block of 32 of NVFP4
 * (a scale for each 16 values) have sums of their own. */
__device__ __forceinline__ void mma16816s8(int *c, uint32_t a0, uint32_t a1, uint32_t b0)
{
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a0), "r"(a1), "r"(b0));
}

__host__ __device__ __forceinline__ int kt_format(int type)
{
    return type == KQ_Q8_R ? KT_Q8R : type == KQ_Q8_0 ? KT_Q80 : type == KQ_Q5_1 ? KT_Q51 :
           type == KQ_Q4_K ? KT_Q4K : type == KQ_NV4 ? KT_NV4 : type == KQ_NVX ? KT_NVX :
           type == KQ_Q6_K ? KT_Q6K : -1;
}

/* x (n rows of cols values) to int8, with xs and xsum for each block of 32. */
__global__ void k_kq_quant_x(const float *x, int8_t *q, float *xs, float *xsum, size_t blocks)
{
    PDL_START();
    size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t bi = tid / 8;
    bool live = bi < blocks;
    float4 v = live ? *(const float4 *)(x + tid * 4) : make_float4(0.f, 0.f, 0.f, 0.f);
    quant8_block(v, live, bi, threadIdx.x % 8, q, xs, xsum);
}

/* Copy the bytes of row `row` of w for the step at column k0 to dst. */
template <int F>
__device__ __forceinline__ void kt_load_w(uint8_t *dst, const uint8_t *wrow, int cols, int k0)
{
    int lane = threadIdx.x % 32;       /* called by one warp for each row, lanes split the copies */
    if (F == KT_Q8R) {
        if (lane < 8) {
            cp_async16(dst + 16 * lane, wrow + k0 + 16 * lane);
        } else if (lane == 8) {
            cp_async8(dst + 128, wrow + cols + k0 / 16);
        }
    } else if (F == KT_Q80) {
        if (lane < 17) {
            cp_async8(dst + 8 * lane, wrow + (size_t)(k0 / 32) * 34 + 8 * lane);
        }
    } else if (F == KT_Q51) {
        if (lane < 6) {
            cp_async16(dst + 16 * lane, wrow + (size_t)(k0 / 32) * 24 + 16 * lane);
        }
    } else if (F == KT_Q6K) {
        /* 53 halfwords: ql (32), qh (16), the 8 scales (4), d (1) */
        const uint8_t *blk = wrow + (size_t)(k0 / 256) * 210;
        int h = (k0 / 128) % 2;
        for (int i = lane; i < 53; i += 32) {
            int o = i < 32 ? 64 * h + 2 * i : i < 48 ? 128 + 32 * h + 2 * (i - 32)
                  : i < 52 ? 192 + 8 * h + 2 * (i - 48) : 208;
            *(uint16_t *)(dst + 2 * i) = *(const uint16_t *)(blk + o);
        }
    } else if (F == KT_NV4) {
        /* the codes of the 4 blocks (64 bytes, 16-aligned), their 8 scales, the
         * scale of the matrix: 64 + 8 + 4 bytes of shared memory */
        if (lane < 4) {
            cp_async16(dst + 16 * lane, wrow + k0 / 2 + 16 * lane);
        } else if (lane == 4) {
            cp_async8(dst + 64, wrow + cols / 2 + k0 / 16);
        } else if (lane == 5) {
            cp_async4(dst + 72, wrow + cols / 2 + cols / 16);
        }
    } else {
        const uint8_t *blk = wrow + (size_t)(k0 / 256) * 144;
        if (lane == 0) {
            cp_async16(dst, blk);
        } else if (lane < 5) {
            cp_async16(dst + 16 * lane, blk + 16 + 64 * ((k0 / 128) % 2) + 16 * (lane - 1));
        }
    }
}

__device__ __forceinline__ uint32_t kt_hi4(uint32_t h)
{
    return ((h & 1) << 4) | ((h & 2) << 11) | ((h & 4) << 18) | ((h & 8) << 25);
}

/* The two registers of the fragment of B (values 4c..4c+3 and 16+4c.. of
 * block bk of the step) of a row in shared memory. */
template <int F>
__device__ __forceinline__ void kt_frag(const uint8_t *rp, int bk, int c, uint32_t *b)
{
    if (F == KT_Q8R) {
        b[0] = *(const uint32_t *)(rp + bk * 32 + 4 * c);
        b[1] = *(const uint32_t *)(rp + bk * 32 + 16 + 4 * c);
    } else if (F == KT_Q80) {
        const uint8_t *q = rp + bk * 34 + 2;
        b[0] = (uint32_t)*(const uint16_t *)(q + 4 * c) | ((uint32_t)*(const uint16_t *)(q + 4 * c + 2) << 16);
        b[1] = (uint32_t)*(const uint16_t *)(q + 16 + 4 * c) |
               ((uint32_t)*(const uint16_t *)(q + 16 + 4 * c + 2) << 16);
    } else if (F == KT_Q51) {
        const uint8_t *blk = rp + bk * 24;
        uint32_t qh = *(const uint32_t *)(blk + 4);
        uint32_t u = *(const uint32_t *)(blk + 8 + 4 * c);
        b[0] = (u & 0x0f0f0f0fu) | kt_hi4(qh >> (4 * c));
        b[1] = ((u >> 4) & 0x0f0f0f0fu) | kt_hi4(qh >> (16 + 4 * c));
    } else {
        const uint8_t *qs = rp + 16 + 32 * (bk / 2);
        int sh = 4 * (bk % 2);
        b[0] = (*(const uint32_t *)(qs + 4 * c) >> sh) & 0x0f0f0f0fu;
        b[1] = (*(const uint32_t *)(qs + 16 + 4 * c) >> sh) & 0x0f0f0f0fu;
    }
}

/* The scale d and the term mn of block bk of the step at k0 of a row. */
template <int F>
__device__ __forceinline__ void kt_scale(const uint8_t *rp, int bk, int k0, float *d, float *mn)
{
    if (F == KT_Q8R) {
        *d = __half2float(*(const __half *)(rp + 128 + 2 * bk));
        *mn = 0.f;
    } else if (F == KT_Q80) {
        *d = kq_half(rp + bk * 34);
        *mn = 0.f;
    } else if (F == KT_Q51) {
        *d = kq_half(rp + bk * 24);
        *mn = kq_half(rp + bk * 24 + 2);
    } else {
        int sc, m;
        kq_scale_min(rp + 4, 4 * ((k0 / 128) % 2) + bk, &sc, &m);
        *d = kq_half(rp) * (float)sc;
        *mn = -kq_half(rp + 2) * (float)m;
    }
}

/* A tile: out[m * ostride + n] for m < mcount (rows of x: xmap[m], or m)
 * and n0 <= n < n0 + 128 (n < rows) of w (rb bytes each). */
/* X2 (KT_Q4X only): the int16 form of x (k_quant_x2), xl its lo plane with
 * the rows of xq; a row of a step holds the hi values, then the lo values. */
template <int F, int BM, int NS, int X2 = 0>
__device__ void kt_tile(const uint8_t *w, size_t rb, int rows, int n0, int cols,
                        const int8_t *xq, const float *xs, const float *xsum, const int *xmap,
                        int mcount, float *out, size_t ostride, uint8_t *sm,
                        const int8_t *xl = NULL)
{
    const int MI = BM / 32;             /* the 16-row parts of a warp */
    constexpr int AW = KT_K * (1 + X2) + 16;
    int8_t (*as_)[BM][AW] = (int8_t (*)[BM][AW])sm;
    uint8_t (*bs)[KT_BN][KT_RB] = (uint8_t (*)[KT_BN][KT_RB])(sm + NS * BM * AW);
    float (*ss)[BM][8] = (float (*)[BM][8])(sm + NS * (BM * AW + KT_BN * KT_RB));
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * (BM / 2), wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    int nb = cols / 32, steps = F == KT_Q4X ? (cols + KT_K - 1) / KT_K : cols / KT_K;
    float acc[MI][4][4];
    #pragma unroll
    for (int i = 0; i < MI; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    auto xrow = [&](int m) {
        int mm = m < mcount ? m : mcount - 1;
        return xmap ? xmap[mm] : mm;
    };
    auto load = [&](int st, int b) {
        int k0 = st * KT_K;
        /* the blocks of the step (4, or fewer at the end of a KT_Q4X row) */
        int nbk = F == KT_Q4X ? min(4, nb - k0 / 32) : 4;
        for (int e = threadIdx.x; e < BM * KT_K / 16; e += blockDim.x) {
            int m = e / (KT_K / 16), kk = (e % (KT_K / 16)) * 16;
            if (F != KT_Q4X || kk < 32 * nbk) {
                cp_async16(&as_[b][m][kk], xq + (size_t)xrow(m) * cols + k0 + kk);
                if (X2) {
                    cp_async16(&as_[b][m][KT_K + kk], xl + (size_t)xrow(m) * cols + k0 + kk);
                }
            }
        }
        for (int e = threadIdx.x; e < BM * 4; e += blockDim.x) {
            int m = e / 4, bk = e % 4;
            if (F == KT_Q4X && bk >= nbk) {
                continue;
            }
            size_t o = (size_t)xrow(m) * nb + k0 / 32 + bk;
            cp_async4(&ss[b][m][bk], xs + o);
            cp_async4(&ss[b][m][4 + bk], xsum + o);
        }
        if (F == KT_Q4X) {
            uint8_t *gs = &bs[b][0][0];
            int nch = 18 * nbk;         /* 288 bytes a block */
            for (int e = threadIdx.x; e < 8 * nch; e += blockDim.x) {
                int gi = e / nch, ch = e % nch;
                int grp = min(n0 / 16 + gi, rows / 16 - 1);
                cp_async16(gs + gi * KT_GB4 + 16 * ch,
                           w + (size_t)grp * 16 * rb + (size_t)(k0 / 32) * 288 + 16 * ch);
            }
        } else if (F == KT_NVX) {
            /* 8 groups: the head, then the 4 blocks of the step */
            uint8_t *gs = &bs[b][0][0];
            for (int e = threadIdx.x; e < 8 * (KT_GB / 16); e += blockDim.x) {
                int gi = e / (KT_GB / 16), ch = e % (KT_GB / 16);
                int grp = min(n0 / 16 + gi, rows / 16 - 1);
                const uint8_t *src = w + (size_t)grp * 16 * rb +
                                     (ch == 0 ? 0 : 16 + (size_t)(k0 / 32) * KQ_NVX_BB + 16 * (ch - 1));
                cp_async16(gs + gi * KT_GB + 16 * ch, src);
            }
        } else {
            for (int n = warp; n < KT_BN; n += 8) {
                int row = min(n0 + n, rows - 1);
                kt_load_w<F>(bs[b][n], w + (size_t)row * rb, cols, k0);
            }
        }
        cp_async_commit();
    };
    for (int st = 0; st < NS - 1; ++st) {
        if (st < steps) {
            load(st, st);
        } else {
            cp_async_commit();
        }
    }
    for (int st = 0; st < steps; ++st) {
        int b = st % NS;
        if (NS == 2) {
            asm volatile("cp.async.wait_group 0;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 1;\n" ::);
        }
        __syncthreads();
        if (st + NS - 1 < steps) {
            load(st + NS - 1, (st + NS - 1) % NS);
        } else {
            cp_async_commit();
        }
        int k0 = st * KT_K;
        if (F == KT_Q4X) {
            int nbk = min(4, nb - k0 / 32);
            #pragma unroll
            for (int bk = 0; bk < 4; ++bk) {
                if (bk >= nbk) {
                    break;
                }
                uint32_t bf[4][2];
                float dw[4][2];
                #pragma unroll
                for (int jg = 0; jg < 2; ++jg) {
                    const uint8_t *blk = &bs[b][0][0] + (wn / 16 + jg) * KT_GB4 + 288 * bk;
                    uint32_t qa = *(const uint32_t *)(blk + 32 * c + 4 * g);
                    uint32_t qb = *(const uint32_t *)(blk + 32 * (4 + c) + 4 * g);
                    bf[2 * jg][0] = __vsub4(qa & 0x0f0f0f0fu, 0x08080808u);
                    bf[2 * jg][1] = __vsub4(qb & 0x0f0f0f0fu, 0x08080808u);
                    bf[2 * jg + 1][0] = __vsub4((qa >> 4) & 0x0f0f0f0fu, 0x08080808u);
                    bf[2 * jg + 1][1] = __vsub4((qb >> 4) & 0x0f0f0f0fu, 0x08080808u);
                    const __half *d = (const __half *)(blk + 256);
                    #pragma unroll
                    for (int sb = 0; sb < 2; ++sb) {
                        dw[2 * jg + sb][0] = __half2float(d[8 * sb + 2 * c]);
                        dw[2 * jg + sb][1] = __half2float(d[8 * sb + 2 * c + 1]);
                    }
                }
                #pragma unroll
                for (int i = 0; i < MI; ++i) {
                    int r0 = wm + i * 16, kk = bk * 32;
                    uint32_t a[4], al[4];
                    a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                    a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                    a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                    a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                    if (X2) {
                        al[0] = *(const uint32_t *)&as_[b][r0 + g][KT_K + kk + 4 * c];
                        al[1] = *(const uint32_t *)&as_[b][r0 + g + 8][KT_K + kk + 4 * c];
                        al[2] = *(const uint32_t *)&as_[b][r0 + g][KT_K + kk + 4 * c + 16];
                        al[3] = *(const uint32_t *)&as_[b][r0 + g + 8][KT_K + kk + 4 * c + 16];
                    }
                    float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int ci[4] = {0, 0, 0, 0};
                        if (X2) {
                            mma16832_x2(ci, a, al, bf[j]);
                        } else {
                            mma16832(ci, a, bf[j]);
                        }
                        acc[i][j][0] += i2f_exact(ci[0]) * s0 * dw[j][0];
                        acc[i][j][1] += i2f_exact(ci[1]) * s0 * dw[j][1];
                        acc[i][j][2] += i2f_exact(ci[2]) * s1 * dw[j][0];
                        acc[i][j][3] += i2f_exact(ci[3]) * s1 * dw[j][1];
                    }
                }
            }
            continue;
        }
        if (F == KT_NV4 || F == KT_NVX || F == KT_Q6K) {
            #pragma unroll
            for (int bk = 0; bk < 4; ++bk) {
                uint32_t bf[4][2];
                float dl[4][2], dh[4][2];
                if (F == KT_Q6K) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const uint8_t *rp = bs[b][wn + j * 8 + g];
                        const uint8_t *ql = rp + 32 * (bk & 1);
                        int sh = 4 * (bk >> 1);
                        uint32_t l0 = (*(const uint32_t *)(ql + 4 * c) >> sh) & 0x0f0f0f0fu;
                        uint32_t l1 = (*(const uint32_t *)(ql + 16 + 4 * c) >> sh) & 0x0f0f0f0fu;
                        uint32_t h0 = (*(const uint32_t *)(rp + 64 + 4 * c) >> (2 * bk)) & 0x03030303u;
                        uint32_t h1 = (*(const uint32_t *)(rp + 80 + 4 * c) >> (2 * bk)) & 0x03030303u;
                        bf[j][0] = __vsub4(l0 | (h0 << 4), 0x20202020u);
                        bf[j][1] = __vsub4(l1 | (h1 << 4), 0x20202020u);
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const uint8_t *rs = bs[b][wn + j * 8 + 2 * c + h];
                            float d = kq_half(rs + 104);
                            dl[j][h] = d * (float)(int8_t)rs[96 + 2 * bk];
                            dh[j][h] = d * (float)(int8_t)rs[96 + 2 * bk + 1];
                        }
                    }
                } else if (F == KT_NVX) {
                    /* rows wn .. wn + 31: 2 groups; a group gives the
                     * fragments of rows g and g + 8 from 2 loads */
                    #pragma unroll
                    for (int jg = 0; jg < 2; ++jg) {
                        const uint8_t *gb = &bs[b][0][0] + (wn / 16 + jg) * KT_GB;
                        const uint8_t *blk = gb + 16 + KQ_NVX_BB * bk;
                        float gg = 128.f * *(const float *)gb;
                        uint32_t qa = *(const uint32_t *)(blk + 32 * c + 4 * g);
                        uint32_t qb = *(const uint32_t *)(blk + 32 * (4 + c) + 4 * g);
                        bf[2 * jg][0] = kt_e2m1x4(qa);
                        bf[2 * jg][1] = kt_e2m1x4(qb);
                        bf[2 * jg + 1][0] = kt_e2m1x4(qa >> 4);
                        bf[2 * jg + 1][1] = kt_e2m1x4(qb >> 4);
                        #pragma unroll
                        for (int sb = 0; sb < 2; ++sb) {
                            #pragma unroll
                            for (int h = 0; h < 2; ++h) {
                                int rr = 8 * sb + 2 * c + h;
                                dl[2 * jg + sb][h] = gg * kq_e4m3s(blk[256 + rr]);
                                dh[2 * jg + sb][h] = gg * kq_e4m3s(blk[272 + rr]);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        uint32_t q = *(const uint32_t *)(bs[b][wn + j * 8 + g] + 16 * bk + 4 * c);
                        bf[j][0] = kt_e2m1x4(q);
                        bf[j][1] = kt_e2m1x4(q >> 4);
                        #pragma unroll
                        for (int h = 0; h < 2; ++h) {
                            const uint8_t *rs = bs[b][wn + j * 8 + 2 * c + h];
                            float gg = 0.5f * *(const float *)(rs + 72);
                            dl[j][h] = gg * kq_e4m3(rs[64 + 2 * bk]);
                            dh[j][h] = gg * kq_e4m3(rs[64 + 2 * bk + 1]);
                        }
                    }
                }
                #pragma unroll
                for (int i = 0; i < MI; ++i) {
                    int r0 = wm + i * 16, kk = bk * 32;
                    uint32_t a[4];
                    a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                    a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                    a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                    a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                    float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int cl[4] = {0, 0, 0, 0}, ch[4] = {0, 0, 0, 0};
                        mma16816s8(cl, a[0], a[1], bf[j][0]);
                        mma16816s8(ch, a[2], a[3], bf[j][1]);
                        acc[i][j][0] += s0 * (i2f_exact(cl[0]) * dl[j][0] + i2f_exact(ch[0]) * dh[j][0]);
                        acc[i][j][1] += s0 * (i2f_exact(cl[1]) * dl[j][1] + i2f_exact(ch[1]) * dh[j][1]);
                        acc[i][j][2] += s1 * (i2f_exact(cl[2]) * dl[j][0] + i2f_exact(ch[2]) * dh[j][0]);
                        acc[i][j][3] += s1 * (i2f_exact(cl[3]) * dl[j][1] + i2f_exact(ch[3]) * dh[j][1]);
                    }
                }
            }
            continue;
        }
        #pragma unroll
        for (int bk = 0; bk < 4; ++bk) {
            uint32_t bf[4][2];
            float dw[4][2], mw[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                kt_frag<F>(bs[b][wn + j * 8 + g], bk, c, bf[j]);
                kt_scale<F>(bs[b][wn + j * 8 + 2 * c], bk, k0, &dw[j][0], &mw[j][0]);
                kt_scale<F>(bs[b][wn + j * 8 + 2 * c + 1], bk, k0, &dw[j][1], &mw[j][1]);
            }
            #pragma unroll
            for (int i = 0; i < MI; ++i) {
                int r0 = wm + i * 16, kk = bk * 32;
                uint32_t a[4];
                a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c];
                a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c];
                a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 4 * c + 16];
                a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 4 * c + 16];
                float s0 = ss[b][r0 + g][bk], s1 = ss[b][r0 + g + 8][bk];
                float u0 = ss[b][r0 + g][4 + bk], u1 = ss[b][r0 + g + 8][4 + bk];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    int ci[4] = {0, 0, 0, 0};
                    mma16832(ci, a, bf[j]);
                    acc[i][j][0] += i2f_exact(ci[0]) * s0 * dw[j][0] + u0 * mw[j][0];
                    acc[i][j][1] += i2f_exact(ci[1]) * s0 * dw[j][1] + u0 * mw[j][1];
                    acc[i][j][2] += i2f_exact(ci[2]) * s1 * dw[j][0] + u1 * mw[j][0];
                    acc[i][j][3] += i2f_exact(ci[3]) * s1 * dw[j][1] + u1 * mw[j][1];
                }
            }
        }
    }
    #pragma unroll
    for (int i = 0; i < MI; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = n0 + wn + j * 8 + 2 * c;
            int m = wm + i * 16 + g;
            if (m < mcount) {
                if (n < rows) out[(size_t)m * ostride + n] = acc[i][j][0];
                if (n + 1 < rows) out[(size_t)m * ostride + n + 1] = acc[i][j][1];
            }
            if (m + 8 < mcount) {
                if (n < rows) out[(size_t)(m + 8) * ostride + n] = acc[i][j][2];
                if (n + 1 < rows) out[(size_t)(m + 8) * ostride + n + 1] = acc[i][j][3];
            }
        }
    }
}

#define KT_SMEM(BM, NS) ((size_t)(NS) * ((BM) * (KT_K + 16) + KT_BN * KT_RB + (BM) * 8 * 4))
#define KT_SMEM_X2(BM, NS) ((size_t)(NS) * ((BM) * (2 * KT_K + 16) + KT_BN * KT_RB + (BM) * 8 * 4))

/* GP_KQ_LINEAR of a large group on the tensor cores: blockIdx.x is the tile
 * of 128 tokens, blockIdx.y the tile of 128 rows. xq, xs, xsum: the scratch. */
template <int F>
__global__ void __launch_bounds__(256) k_kq_tc(const gp_rec *r, const int64_t *e, const int8_t *xq,
                                               const float *xs, const float *xsum)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t ktsm[];
    int rows = DI(6), cols = DI(7), t = DI(8), type = DI(5);
    int m0 = blockIdx.x * 128;
    kt_tile<F, 128, 2>(DP(const uint8_t, 4), kq_row_bytes(type, cols), rows, blockIdx.y * KT_BN, cols,
                       xq + (size_t)m0 * cols, xs + (size_t)m0 * (cols / 32),
                       xsum + (size_t)m0 * (cols / 32), NULL, min(128, t - m0),
                       DP(float, 9) + (size_t)m0 * rows, (size_t)rows, ktsm);
}

/* GP_KQ_LINEAR of a large group (t > MT_MAX). */
__global__ void __launch_bounds__(256) k_kq_gemm(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int type = DI(5), rows = DI(6), cols = DI(7), t = DI(8);
    int j0 = blockIdx.y * KG_B;
    kq_tile(type, DP(const uint8_t, 4), kq_row_bytes(type, cols), rows, blockIdx.x * KG_B, cols,
            DP(const float, 3) + (size_t)j0 * cols, (size_t)cols, NULL, min(KG_B, t - j0),
            DP(float, 9) + (size_t)j0 * rows, (size_t)rows);
}

/* GP_KQ_GROUP_MOE: the experts of a large group on the GPU.
 *
 *     h, val, idx, t, k, E, hidden, inner, tgate, tup, tdown, gtype, dtype,
 *     sgate, sup, sdown, stype, slog, work, act, act2, de, out, nreal
 *
 * tgate, tup, tdown give the device address of each expert (a hot slot, or
 * the buffer of the copies of GP_FETCH). The shared expert is expert E.
 * k_qmoe_sort sorts the pairs (token, expert) of the first nreal tokens by
 * expert, and makes the tiles of at most 64 pairs of one expert. Then the
 * gate and up rows, silu(gate) up, the down rows, and the sum. */
struct qmoe_w {
    int *cnt, *start, *ptok, *pof, *tiles;
};

__device__ __forceinline__ qmoe_w qmoe_work(const gp_rec *r, const int64_t *e)
{
    int E = DI(5), P = DI(3) * DI(4) + DI(3);
    qmoe_w w;
    w.cnt = DP(int, 18) + 8;
    w.start = w.cnt + E + 2;
    w.ptok = w.start + E + 2;
    w.pof = w.ptok + P;
    w.tiles = w.pof + P;
    return w;
}

/* k_qmoe_sort with a block of threads: the counts and the places of the
 * pairs with atomics in shared memory, the starts and the tiles on thread 0.
 * The pairs of an expert can come in another order than in k_qmoe_sort; a
 * pair has the same values in any place of its tiles. One thread took 5 ms
 * for a group of 2048 tokens of Qwen3.8 (20480 pairs, 512 experts). */
__global__ void __launch_bounds__(1024) k_qmoe_sort_par(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ int qs_sm[];
    int t = DI(3), k = DI(4), E = DI(5);
    int nr = DI(23);
    nr = nr > 0 && nr < t ? nr : t;
    const int *idx = DP(const int, 2);
    int *work = DP(int, 18);
    qmoe_w w = qmoe_work(r, e);
    int *scnt = qs_sm, *sfill = qs_sm + E + 2;
    for (int x = threadIdx.x; x <= E; x += blockDim.x) {
        scnt[x] = 0;
    }
    __syncthreads();
    for (int q = threadIdx.x; q < nr * k; q += blockDim.x) {
        if (idx[q] >= 0) {
            atomicAdd(&scnt[idx[q]], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        scnt[E] = DP(const void, 13) != NULL ? nr : 0;     /* no shared expert: sgate null */
        int a = 0, nt = 0;
        for (int x = 0; x <= E; ++x) {
            w.start[x] = a;
            sfill[x] = a;
            for (int s2 = 0; s2 < scnt[x]; s2 += KG_B) {
                w.tiles[3 * nt] = x;
                w.tiles[3 * nt + 1] = a + s2;
                w.tiles[3 * nt + 2] = min(KG_B, scnt[x] - s2);
                ++nt;
            }
            a += scnt[x];
        }
        w.start[E + 1] = a;
        work[0] = nt;
        work[1] = a;
    }
    __syncthreads();
    for (int x = threadIdx.x; x <= E; x += blockDim.x) {
        w.cnt[x] = scnt[x];
    }
    for (int q = threadIdx.x; q < t * k; q += blockDim.x) {
        if (q / k >= nr || idx[q] < 0) {
            w.pof[q] = -1;
            continue;
        }
        int pos = atomicAdd(&sfill[idx[q]], 1);
        w.ptok[pos] = q / k;
        w.pof[q] = pos;
    }
    const int shared = DP(const void, 13) != NULL;
    for (int j = threadIdx.x; j < t; j += blockDim.x) {
        int pos = -1;
        if (j < nr && shared) {
            pos = w.start[E] + j;
            w.ptok[pos] = j;
        }
        w.pof[t * k + j] = pos;
    }
}

static int qmoe_sort_par_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_QMOE_SORT_PAR");
        on = !(v && v[0] == '0');
    }
    return on;
}

__global__ void k_qmoe_sort(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    if (threadIdx.x != 0) {
        return;
    }
    int t = DI(3), k = DI(4), E = DI(5);
    int nr = DI(23);
    nr = nr > 0 && nr < t ? nr : t;
    const int *idx = DP(const int, 2);
    int *work = DP(int, 18);
    qmoe_w w = qmoe_work(r, e);
    for (int x = 0; x <= E; ++x) {
        w.cnt[x] = 0;
    }
    for (int q = 0; q < nr * k; ++q) {
        if (idx[q] >= 0) {          /* -1: an expert of the CPU (GP_MOE_PLAN) */
            w.cnt[idx[q]]++;
        }
    }
    w.cnt[E] = DP(const void, 13) != NULL ? nr : 0;
    int a = 0, nt = 0;
    for (int x = 0; x <= E; ++x) {
        w.start[x] = a;
        for (int s = 0; s < w.cnt[x]; s += KG_B) {
            w.tiles[3 * nt] = x;
            w.tiles[3 * nt + 1] = a + s;
            w.tiles[3 * nt + 2] = min(KG_B, w.cnt[x] - s);
            ++nt;
        }
        a += w.cnt[x];
        w.cnt[x] = 0;
    }
    w.start[E + 1] = a;
    for (int q = 0; q < t * k; ++q) {
        if (q / k >= nr || idx[q] < 0) {
            w.pof[q] = -1;
            continue;
        }
        int x = idx[q], pos = w.start[x] + w.cnt[x]++;
        w.ptok[pos] = q / k;
        w.pof[q] = pos;
    }
    for (int j = 0; j < t; ++j) {
        int pos = -1;
        if (j < nr && DP(const void, 13) != NULL) {
            pos = w.start[E] + w.cnt[E]++;
            w.ptok[pos] = j;
        }
        w.pof[t * k + j] = pos;
    }
    work[0] = nt;
    work[1] = a;
}

__global__ void __launch_bounds__(256) k_qmoe_gu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    int row0 = blockIdx.x * KG_B, up = row0 >= inner;
    int type = x < E ? DI(11) : DI(16);
    const uint8_t *W = x < E ? (const uint8_t *)(intptr_t)DP(const int64_t, up ? 9 : 8)[x]
                             : DP(const uint8_t, up ? 14 : 13);
    kq_tile(type, W, kq_row_bytes(type, hidden), inner, row0 - (up ? inner : 0), hidden,
            DP(const float, 0), (size_t)hidden, w.ptok + p0, n,
            DP(float, 19) + (size_t)p0 * 2 * inner + (up ? inner : 0), (size_t)2 * inner);
}

/* rot: as k_kqh_act (the warps cover whole groups: the count is a
 * multiple of 32) */
__global__ void k_qmoe_act(const gp_rec *r, const int64_t *e, int rot)
{
    PDL_START();
    int inner = DI(7);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)DP(const int, 18)[1] * inner) {
        return;
    }
    int64_t p = i / inner, c = i % inner;
    const float *a = DP(const float, 19) + p * 2 * inner;
    float v = qw_silu(a[c]) * a[inner + c];
    if (rot) {
        v = tq6_rot_warp(v, (int)(c % 32), false);
    }
    DP(float, 20)[i] = v;
}

__global__ void __launch_bounds__(256) k_qmoe_dn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    int type = x < E ? DI(12) : DI(16);
    const uint8_t *W = x < E ? (const uint8_t *)(intptr_t)DP(const int64_t, 10)[x]
                             : DP(const uint8_t, 15);
    kq_tile(type, W, kq_row_bytes(type, inner), hidden, blockIdx.x * KG_B, inner,
            DP(const float, 20) + (size_t)p0 * inner, (size_t)inner, NULL, n,
            DP(float, 21) + (size_t)p0 * hidden, (size_t)hidden);
}

__global__ void k_qmoe_sum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int c = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    int t = DI(3), k = DI(4), hidden = DI(6);
    if (c >= hidden) {
        return;
    }
    const float *val = DP(const float, 1), *de = DP(const float, 21);
    float acc = 0.f;
    int ps = w.pof[t * k + tok];
    /* no shared expert (sgate null: the copied experts of a mixed group,
     * after its hot experts): only the routed pairs */
    const int noshared = DP(const void, 13) == NULL;
    if (ps >= 0 || noshared) {
        if (ps >= 0) {
            acc = de[(size_t)ps * hidden + c] / (1.f + expf(-DP(const float, 17)[tok]));
        }
        for (int s2 = 0; s2 < k; ++s2) {
            int p = w.pof[tok * k + s2];
            if (p >= 0) {           /* -1: an expert of the CPU (GP_MOE_PLAN) */
                acc += val[tok * k + s2] * de[(size_t)p * hidden + c];
            }
        }
    }
    DP(float, 22)[(size_t)tok * hidden + c] = acc;
}


/* GP_KQ_GROUP_MOE on the tensor cores: the tiles of k_qmoe_sort (at most
 * KG_B pairs of one expert), 128 rows of w for each block. xq, xs, xsum:
 * the int8 rows of h (gate and up) or of act2 (down). The shared expert
 * (x == E) has the Q8_R rows of the GPU. */
template <int KFG>
__global__ void __launch_bounds__(256) k_qmoe_gu_tc(const gp_rec *r, const int64_t *e,
                                                    const int8_t *xq, const float *xs,
                                                    const float *xsum, int skip_shared = 0)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t ktsm[];
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    if (x == E && skip_shared) {
        return;         /* the shared expert in bfloat16 (gg_gemm_bf16) */
    }
    int row0 = blockIdx.x * KT_BN, up = row0 >= inner;
    float *out = DP(float, 19) + (size_t)p0 * 2 * inner + (up ? inner : 0);
    if (x < E) {
        const uint8_t *W = (const uint8_t *)(intptr_t)DP(const int64_t, up ? 9 : 8)[x];
        kt_tile<KFG, 64, 3>(W, kq_row_bytes(DI(11), hidden), inner, row0 - (up ? inner : 0), hidden,
                           xq, xs, xsum, w.ptok + p0, n, out, (size_t)2 * inner, ktsm);
    } else {
        kt_tile<KT_Q8R, 64, 3>(DP(const uint8_t, up ? 14 : 13), kq_row_bytes(KQ_Q8_R, hidden), inner,
                               row0 - (up ? inner : 0), hidden, xq, xs, xsum, w.ptok + p0, n, out,
                               (size_t)2 * inner, ktsm);
    }
}

template <int KFD>
__global__ void __launch_bounds__(256) k_qmoe_dn_tc(const gp_rec *r, const int64_t *e,
                                                    const int8_t *xq, const float *xs,
                                                    const float *xsum, int skip_shared = 0)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t ktsm[];
    qmoe_w w = qmoe_work(r, e);
    int tile = blockIdx.y;
    if (tile >= DP(const int, 18)[0]) {
        return;
    }
    int x = w.tiles[3 * tile], p0 = w.tiles[3 * tile + 1], n = w.tiles[3 * tile + 2];
    int E = DI(5), hidden = DI(6), inner = DI(7);
    if (x == E && skip_shared) {
        return;
    }
    float *out = DP(float, 21) + (size_t)p0 * hidden;
    const int8_t *xr = xq + (size_t)p0 * inner;
    const float *sr = xs + (size_t)p0 * (inner / 32), *ur = xsum + (size_t)p0 * (inner / 32);
    if (x < E) {
        const uint8_t *W = (const uint8_t *)(intptr_t)DP(const int64_t, 10)[x];
        kt_tile<KFD, 64, 3>(W, kq_row_bytes(DI(12), inner), hidden, blockIdx.x * KT_BN, inner, xr, sr,
                           ur, NULL, n, out, (size_t)hidden, ktsm);
    } else {
        kt_tile<KT_Q8R, 64, 3>(DP(const uint8_t, 15), kq_row_bytes(KQ_Q8_R, inner), hidden,
                               blockIdx.x * KT_BN, inner, xr, sr, ur, NULL, n, out, (size_t)hidden,
                               ktsm);
    }
}

/* GP_MOE_GPU with KQ_Q4X experts (operand 5 = 1): a tile of k_moe_tiles (at
 * most GM = 64 pairs of one expert) and 128 rows of its matrix, on the
 * tensor cores with int8 x (xq, xs, xsum of kt_quant). gather: row q of x
 * is row pair_tok[q] of h (gate and up); else x is (pairs, cols) (down). */
/* X2: the int16 form of x (k_quant_x2), xl its lo plane; 2 buffers. */
template <int X2>
__global__ void __launch_bounds__(256) k_moe_gemm_q4x(const gp_rec *r, const int64_t *e,
                                                      const int8_t *xq, const float *xs,
                                                      const float *xsum, int gather,
                                                      const int64_t *wtab, int rows, int cols,
                                                      float *out, const int8_t *xl)
{
    constexpr int NS = X2 ? 2 : 3;
    PDL_START();
    extern __shared__ __align__(16) uint8_t ktsm[];
    const int *tiles = DP(const int, 16);
    const int *off = DP(const int, 12);
    int tile = blockIdx.y;
    if (tile >= tiles[0]) {
        return;
    }
    int ex = tiles[1 + 2 * tile], q0 = tiles[2 + 2 * tile];
    int q1 = min(off[ex + 1], q0 + GM);
    size_t rb = (size_t)(cols / 32) * 18, nbx = (size_t)(cols / 32);
    const uint8_t *W = (const uint8_t *)(intptr_t)wtab[ex];
    if (gather) {
        kt_tile<KT_Q4X, 64, NS, X2>(W, rb, rows, blockIdx.x * KT_BN, cols, xq, xs, xsum,
                                    DP(const int, 14) + q0, q1 - q0, out + (size_t)q0 * rows,
                                    (size_t)rows, ktsm, xl);
    } else {
        kt_tile<KT_Q4X, 64, NS, X2>(W, rb, rows, blockIdx.x * KT_BN, cols,
                                    xq + (size_t)q0 * cols, xs + (size_t)q0 * nbx,
                                    xsum + (size_t)q0 * nbx, NULL, q1 - q0,
                                    out + (size_t)q0 * rows, (size_t)rows, ktsm,
                                    X2 ? xl + (size_t)q0 * cols : NULL);
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
    uint16_t *w16;       /* the bfloat16 rows of a chunk of a KQ_BF12 matrix (gg_gemm_bf12) */
    size_t w16_n;
    float *btp;          /* the parts of the sums of k_kq_bf12_tc of class 1, and their counts */
    int *btc;
    int8_t *kqx;         /* the int8 rows of the products of large groups (k_kq_tc) */
    size_t kqx_n;        /* its values; then xs and xsum, kqx_n / 32 floats each */
    int tc;              /* 1: the tensor cores for a large group */
    int i8;              /* 1: the int8 products of k_gemm_q8 for a large group, also
                          * with tc 0 (the 26B: int8 has no overflow, float16 has);
                          * 2: the int16 form (k_quant_x2), also for the experts */
    int atc;             /* 1: the attention of a large group on the tensor cores,
                          * also with tc 0 (the queries and keys have a norm) */
} gg_prog;

/* The size of the scratch of the attention: 16 query heads of 512 values,
 * or the same count of values in another form. */
#define GG_PART_FLOATS ((size_t)16 * ATTN_CHUNKS * (512 + 2))

/* ---------- the handoff to the CPU inside a graph ----------
 * A step can hand work to the CPU with no boundary record, so the step is
 * one graph and the host launches nothing between its layers:
 *
 *   GP_D2H (device source, host target, bytes) and GP_H2D (host source,
 *   device target, bytes): copies in the stream (copy nodes of the graph).
 *   GP_SIGNAL (flag, value, then up to 3 copies (device source, host
 *   target, bytes)): a kernel copies the data to pinned host memory (the
 *   GPU writes it through the map of the memory), then writes value to the
 *   flag, an int64 in pinned host memory.
 *   GP_AWAIT (flag, value, then a copy (host source, device target, bytes)
 *   or none, then a device int or none): a kernel waits until the flag is
 *   value or more, then copies the data from pinned host memory. The int is
 *   the length of the block of the CPU (the count of its experts): with 0,
 *   the kernel writes zeros to the target and does not wait, and the GPU
 *   goes on while the runner only marks the task done. A copy node of a graph costs more than
 *   a kernel, and it breaks the programmatic edges.
 *   GP_CPU_TASK (CPU program, flag in, flag out, value): no kernel. After
 *   the launch of its segment, the runner waits until flag in is value or
 *   more, runs the program, and writes value to flag out.
 *
 * The value is a slot that grows at each run (seq), so the flags need no
 * reset. The tasks run in their order, on the thread of the runner. */
/* Copy n bytes (a multiple of 4) with the threads of the block: 16 bytes
 * for each access when the addresses allow it (fewer PCIe transfers). The
 * loads do not use the caches (ld.global.cv): the source can be pinned host
 * memory that the CPU wrote since the last step, and a line of the last
 * step in a cache would give its old values. */
__device__ __forceinline__ void block_copy4(void *dst, const void *src, size_t n)
{
    if ((((uintptr_t)dst | (uintptr_t)src | n) & 15) == 0) {
        const float4 *s = (const float4 *)src;
        float4 *d = (float4 *)dst;
        for (size_t i = threadIdx.x; i < n / 16; i += blockDim.x) {
            d[i] = __ldcv(s + i);
        }
        return;
    }
    const float *s = (const float *)src;
    float *d = (float *)dst;
    for (size_t i = threadIdx.x; i < n / 4; i += blockDim.x) {
        d[i] = __ldcv(s + i);
    }
}

__global__ void k_signal(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    for (int k = 0; k < 3; ++k) {
        if (di(r, e, 2 + 3 * k) != 0) {
            block_copy4(DP(void, 3 + 3 * k), DP(const void, 2 + 3 * k), (size_t)di(r, e, 4 + 3 * k));
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *DP(volatile int64_t, 0) = di(r, e, 1);
        __threadfence_system();
    }
}

__global__ void k_await(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    if (di(r, e, 5) != 0 && *DP(const int, 5) == 0) {
        if (di(r, e, 2) != 0) {
            float *d = DP(float, 3);
            for (size_t i = threadIdx.x; i < (size_t)di(r, e, 4) / 4; i += blockDim.x) {
                d[i] = 0.f;
            }
        }
        return;
    }
    if (threadIdx.x == 0) {
        volatile int64_t *flag = DP(volatile int64_t, 0);
        int64_t v = di(r, e, 1);
        uint64_t t0;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        while (*flag < v) {
            __nanosleep(256);
            uint64_t t;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            if (t - t0 > 60ull * 1000000000ull) {
                __trap();           /* no CPU task in 60 s: an error, not a hang */
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if (di(r, e, 2) != 0) {
        block_copy4(DP(void, 3), DP(const void, 2), (size_t)di(r, e, 4));
    }
}

static int64_t hi(const gp_rec *r, const int64_t *e, int k);

/* The measures of the tasks (gg_task_stats): the time the runner waited for
 * the signals, the time of the CPU programs, the count. */
static double gg_ts_wait, gg_ts_cpu;
static int64_t gg_ts_n;

/* out gets the wait and CPU seconds and the count of the tasks since the
 * last call; then they start again. */
extern "C" int gg_task_stats(double *out)
{
    out[0] = gg_ts_wait;
    out[1] = gg_ts_cpu;
    out[2] = (double)gg_ts_n;
    gg_ts_wait = gg_ts_cpu = 0.0;
    gg_ts_n = 0;
    return 0;
}

/* Run a GP_CPU_TASK on the host: wait for flag in, run, write flag out. */
static int gg_cpu_task(const gp_rec *r, const int64_t *henv)
{
    volatile int64_t *fin = (volatile int64_t *)(intptr_t)r->v[1];
    volatile int64_t *fout = (volatile int64_t *)(intptr_t)r->v[2];
    int64_t v = hi(r, henv, 3);
    double t0 = gg_clock();
    while (*fin < v) {
        __builtin_ia32_pause();
        if (gg_clock() - t0 > 30.0) {
            snprintf(gg_error, sizeof(gg_error), "GP_CPU_TASK: no signal of the GPU in 30 s");
            return -1;
        }
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    double t1 = gg_clock();
    if (gg_cpu_run == NULL || gg_cpu_run((const int64_t *)(intptr_t)r->v[0], -1) != 0) {
        snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_TASK failed");
        return -1;
    }
    __atomic_thread_fence(__ATOMIC_RELEASE);
    *fout = v;
    double t2 = gg_clock();
    gg_ts_wait += t1 - t0;
    gg_ts_cpu += t2 - t1;
    ++gg_ts_n;
    return 0;
}

static int is_boundary(int op)
{
    return op == GP_TO_HOST || op == GP_CPU_JOIN || op == GP_TO_DEV || op == GP_FETCH ||
           op == GP_FETCH_WAIT || op == GP_FETCH_DONE || op == GP_CPU_START ||
           op == GP_CPU_WAIT;
}

/* ---------- a CPU program on a thread of its own ----------
 * GP_CPU_START (program, event): wait for the event, then a helper thread
 * runs the CPU program while the runner goes on (the copies of GP_FETCH,
 * the launches of the GPU). GP_CPU_WAIT: wait for the program. One program
 * at a time. */
static pthread_mutex_t gg_hmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gg_hcv = PTHREAD_COND_INITIALIZER;
static const int64_t *gg_hjob;
static int gg_hbusy, gg_hfail, gg_hon;
static double gg_hlast;       /* the time of the last program, in s */

static void *gg_helper(void *arg)
{
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&gg_hmu);
        while (gg_hjob == NULL) {
            pthread_cond_wait(&gg_hcv, &gg_hmu);
        }
        const int64_t *prog = gg_hjob;
        pthread_mutex_unlock(&gg_hmu);
        double t0 = gg_clock();
        int rc = gg_cpu_run == NULL ? -1 : gg_cpu_run(prog, -1);
        double dt = gg_clock() - t0;
        pthread_mutex_lock(&gg_hmu);
        gg_hlast = dt;
        gg_hjob = NULL;
        gg_hbusy = 0;
        gg_hfail |= rc != 0;
        pthread_cond_broadcast(&gg_hcv);
        pthread_mutex_unlock(&gg_hmu);
    }
    return NULL;
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

/* ---------- the measures of the mixed groups (gg_mix_stats) ----------
 * While they are on, a run keeps: the bytes and the time of each copy job
 * of GP_FETCH (host clock, after the job is on the stream); the time that
 * the stream waits in each GP_FETCH_WAIT, and in each GP_CPU_JOIN (timing
 * events on the stream: from the end of the work before to the next
 * record); and the time of each CPU program of GP_CPU_JOIN. ModelGPU.
 * plan_mix fits its model of the time to them. */
#define GG_MS_MAX 256
static int gg_ms_on;
static double gg_ms_copy_bytes, gg_ms_copy_s;
static int gg_ms_nf, gg_ms_nj;
static cudaEvent_t gg_ms_ev[4 * GG_MS_MAX];
static int gg_ms_ev_ok;
static double gg_ms_cpu[GG_MS_MAX];
static int gg_ms_kind[GG_MS_MAX];      /* 0: GP_FETCH_WAIT, 1: GP_CPU_JOIN */

/* on 1: clear the measures and keep them from now on. on 0: stop. */
extern "C" int gg_mix_stats_on(int on)
{
    if (on && !gg_ms_ev_ok) {
        for (int k = 0; k < 4 * GG_MS_MAX; ++k) {
            CK(cudaEventCreate(&gg_ms_ev[k]));
        }
        gg_ms_ev_ok = 1;
    }
    pthread_mutex_lock(&gg_mu);
    gg_ms_copy_bytes = gg_ms_copy_s = 0.0;
    pthread_mutex_unlock(&gg_mu);
    gg_ms_nf = gg_ms_nj = 0;
    gg_ms_on = on;
    return 0;
}

/* After a run: out gets the copy bytes and seconds, the counts nf and nj,
 * then for each wait in the order of the run its kind (0: GP_FETCH_WAIT,
 * 1: GP_CPU_JOIN), its time, and the time of the work of the stream between
 * the wait before and this one (0 for the first), then the nj CPU times
 * (times in seconds). Return the count of values, or -1. */
extern "C" int gg_mix_stats(double *out, int max)
{
    if (4 + 3 * gg_ms_nf + 4 * gg_ms_nj > max) {
        snprintf(gg_error, sizeof(gg_error), "gg_mix_stats: %d values do not fit",
                 4 + 3 * gg_ms_nf + 4 * gg_ms_nj);
        return -1;
    }
    CK(cudaStreamSynchronize(gg_stream));
    int n = 0;
    pthread_mutex_lock(&gg_mu);
    out[n++] = gg_ms_copy_bytes;
    out[n++] = gg_ms_copy_s;
    pthread_mutex_unlock(&gg_mu);
    out[n++] = gg_ms_nf;
    out[n++] = gg_ms_nj;
    for (int k = 0; k < gg_ms_nf + gg_ms_nj; ++k) {
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, gg_ms_ev[2 * k], gg_ms_ev[2 * k + 1]));
        out[n++] = gg_ms_kind[k];
        out[n++] = 1e-3 * ms;
        float gap = 0.f;
        if (k > 0) {
            CK(cudaEventElapsedTime(&gap, gg_ms_ev[2 * k - 1], gg_ms_ev[2 * k]));
        }
        out[n++] = 1e-3 * gap;
    }
    for (int k = 0; k < gg_ms_nj; ++k) {
        out[n++] = gg_ms_cpu[k];
    }
    return n;
}

/* ---------- copies from pageable host memory (the map of the model file) ----------
 * A cudaMemcpyAsync from pageable memory goes through a staging buffer of
 * the driver, and the call holds the driver while it copies: on the 2-socket
 * Xeon the copies of HotCache (2.8 MB experts of Qwen3.8 in the decode)
 * held back the steps, 32 tok/s against 36 with no copies. A stager copies
 * chunks of GG_STAGE bytes into two pinned buffers of its own (memcpy on
 * the thread of the worker) and gives each chunk to the stream as a copy
 * from pinned memory (DMA). NP_GEMMA_GPU_STAGE=0 keeps the plain call.
 * NP_GEMMA_GPU_COPY_CPU=c keeps the workers on CPU c (by default the last
 * CPU of the node of the GPU among the CPUs of the process at its start,
 * NP_GEMMA_START_CPUS of np_gemma: the balanced teams of gemma_run_task leave
 * the last CPU of a node free when they take fewer than its CPUs; the
 * mixed groups take OMP_NUM_THREADS - 2. -1: no binding). */
#define GG_STAGE (4u << 20)
typedef struct {
    uint8_t *buf[2];
    cudaEvent_t ev[2];
    int cur, init;
} gg_stager;

static int gg_stage_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_STAGE");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* The CPUs of the node of the GPU (local_cpulist) among those of the
 * process at its start (NP_GEMMA_START_CPUS of np_gemma; all the node's when
 * unset), ascending, in out (at most max); the count, or -1. */
static int gg_node_cpus(int *out, int max)
{
    int dev = 0;
    cudaGetDevice(&dev);
    char bus[32], path[128];
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) != cudaSuccess) {
        return -1;
    }
    for (char *q = bus; *q; ++q) {
        if (*q >= 'A' && *q <= 'F') {
            *q = (char)(*q - 'A' + 'a');
        }
    }
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", bus);
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    char line[256] = {0};
    if (fgets(line, sizeof(line), f) == NULL) {
        line[0] = 0;
    }
    fclose(f);
    static unsigned char node[4096], start[4096];
    memset(node, 0, sizeof(node));
    memset(start, 0, sizeof(start));
    for (char *q = line; *q && *q != '\n';) {
        int a = (int)strtol(q, &q, 10), b = a;
        if (*q == '-') {
            b = (int)strtol(q + 1, &q, 10);
        }
        for (int k = a; k <= b && k < 4096; ++k) {
            if (k >= 0) {
                node[k] = 1;
            }
        }
        if (*q == ',') {
            ++q;
        } else {
            break;
        }
    }
    const char *sc = getenv("NP_GEMMA_START_CPUS");
    int any = 0;
    for (const char *q = sc; q != NULL && *q;) {
        char *e;
        int k = (int)strtol(q, &e, 10);
        if (e == q) {
            break;
        }
        if (k >= 0 && k < 4096) {
            start[k] = 1;
            any = 1;
        }
        if (*e != ',') {
            break;
        }
        q = e + 1;
    }
    int n = 0;
    for (int k = 0; k < 4096 && n < max; ++k) {
        if (node[k] && (start[k] || !any)) {
            out[n++] = k;
        }
    }
    return n;
}

/* The CPUs of the threads around the teams (gemma_run_task) on the node of
 * the GPU: the copy workers the last of gg_node_cpus (NP_GEMMA_GPU_COPY_CPU
 * when set), the thread of the model (the thread that loads this library)
 * the first; -1 for none. np_gemma.gpu pins the model thread and keeps both
 * out of the teams (NP_GEMMA_RESERVED_CPUS); the master of the teams (the GPU
 * runner) takes the first CPU of their plan. */
extern "C" void gg_cpu_roles(int *copy_cpu, int *main_cpu)
{
    int cpus[1024];
    int n = gg_node_cpus(cpus, 1024);
    const char *v = getenv("NP_GEMMA_GPU_COPY_CPU");
    *copy_cpu = v ? atoi(v) : n > 0 ? cpus[n - 1] : -1;
    *main_cpu = n >= 3 ? cpus[0] : -1;
}

/* Keep the calling worker thread on CPU NP_GEMMA_GPU_COPY_CPU (see above). */
static void gg_worker_bind(void)
{
    int c, m;
    gg_cpu_roles(&c, &m);
    if (c < 0) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

/* 1 when src is registered (page-locked) host memory: a copy from it is
 * DMA with no staging (QwenGPU registers its copies of the experts). */
static int gg_is_pinned(const void *src)
{
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, src) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return a.type == cudaMemoryTypeHost;
}

/* Copy n bytes from host src (pageable, or registered) to device dst on
 * stream st. direct: a registered src goes as DMA with no staging (the
 * copies of a prompt group). The copies of HotCache in the decode stay
 * staged: DMA at the full rate of the bus delayed the small transfers of
 * the steps (Qwen3.8 MTP 66.7 -> 58.8 tok/s). */
static int gg_h2d_staged(gg_stager *sg, void *dst, const void *src, size_t n, cudaStream_t st,
                         int direct = 0)
{
    if (!gg_stage_on() || (direct && gg_is_pinned(src))) {
        return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, st) == cudaSuccess ? 0 : -1;
    }
    if (!sg->init) {
        for (int b = 0; b < 2; ++b) {
            if (cudaMallocHost(&sg->buf[b], GG_STAGE) != cudaSuccess ||
                cudaEventCreateWithFlags(&sg->ev[b], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventRecord(sg->ev[b], st) != cudaSuccess) {
                return -1;
            }
        }
        sg->init = 1;
    }
    for (size_t off = 0; off < n; off += GG_STAGE) {
        size_t len = n - off < GG_STAGE ? n - off : GG_STAGE;
        int b = sg->cur;
        if (cudaEventSynchronize(sg->ev[b]) != cudaSuccess) {
            return -1;
        }
        memcpy(sg->buf[b], (const uint8_t *)src + off, len);
        if (cudaMemcpyAsync((uint8_t *)dst + off, sg->buf[b], len, cudaMemcpyHostToDevice, st) !=
                cudaSuccess ||
            cudaEventRecord(sg->ev[b], st) != cudaSuccess) {
            return -1;
        }
        sg->cur ^= 1;
    }
    return 0;
}

/* NP_GEMMA_GPU_FETCH_PULL (16): the copies of a GP_FETCH job from
 * registered host memory as one kernel of that many blocks of 256 threads
 * that reads the host memory (the map of the pinned experts) and writes the
 * buffer, in place of a DMA copy for each range. On the 3090 (PCIe 3.0 x16)
 * the DMA engine moved 8.0 GB/s with copies of 1 MB to 512 MB (more streams
 * did not help); a kernel of 4 to 32 blocks reads 12.3 GB/s. 0: DMA. A job
 * with a range that is not registered or not 16-byte aligned takes DMA. */
static int gg_pull_blocks(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("NP_GEMMA_GPU_FETCH_PULL");
        n = v ? atoi(v) : 16;
    }
    return n;
}

__global__ void k_fetch_pull(const int64_t *__restrict__ rg, int count)
{
    const size_t step = (size_t)gridDim.x * blockDim.x;
    for (int k = 0; k < count; ++k) {
        const int4 *src = (const int4 *)(intptr_t)rg[3 * k];
        int4 *dst = (int4 *)(intptr_t)rg[3 * k + 1];
        size_t n = (size_t)rg[3 * k + 2] / 16;
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        /* two loads in flight a thread */
        for (; i + step < n; i += 2 * step) {
            int4 a = src[i], b = src[i + step];
            dst[i] = a;
            dst[i + step] = b;
        }
        if (i < n) {
            dst[i] = src[i];
        }
    }
}

/* The ranges of a job for k_fetch_pull (the device address of each
 * source), in rg (3 * count values; the ranges of 0 bytes left out); the
 * count, or -1 when the job must take DMA. */
static int gg_pull_ranges(const int64_t *ranges, int count, int64_t *rg)
{
    int m = 0;
    for (int k = 0; k < count; ++k) {
        const int64_t *q = ranges + 3 * k;
        if (q[2] == 0) {
            continue;
        }
        cudaPointerAttributes a;
        if (cudaPointerGetAttributes(&a, (const void *)(intptr_t)q[0]) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        int64_t src = (int64_t)(intptr_t)a.devicePointer;
        if (a.type != cudaMemoryTypeHost || src == 0 || (src | q[1] | q[2]) % 16 != 0) {
            return -1;
        }
        rg[3 * m] = src;
        rg[3 * m + 1] = q[1];
        rg[3 * m + 2] = q[2];
        ++m;
    }
    return m;
}

static void *gg_worker(void *arg)
{
    (void)arg;
    static gg_stager sg;
    static int64_t *prg, *prg_d;
    static int prg_cap;
    gg_worker_bind();
    for (;;) {
        pthread_mutex_lock(&gg_mu);
        while (gg_job_head == gg_job_tail) {
            pthread_cond_wait(&gg_cv, &gg_mu);
        }
        gg_job j = gg_jobs[gg_job_head % GG_FETCH_MAX];
        pthread_mutex_unlock(&gg_mu);
        int bad = cudaEventSynchronize(gg_freeev[j.b]) != cudaSuccess;
        double t0 = gg_clock(), nb = 0.0;
        int np_ = -1;
        if (!bad && gg_pull_blocks() > 0) {
            if (j.count > prg_cap) {
                free(prg);
                cudaFree(prg_d);
                prg = (int64_t *)malloc((size_t)j.count * 24);
                bad = prg == NULL || cudaMalloc((void **)&prg_d, (size_t)j.count * 24) != cudaSuccess;
                prg_cap = bad ? 0 : j.count;
            }
            np_ = bad ? -1 : gg_pull_ranges(j.ranges, j.count, prg);
        }
        if (np_ > 0) {
            /* prg is pageable: the call returns when the driver has the
             * values, so the next job can write prg */
            bad = cudaMemcpyAsync(prg_d, prg, (size_t)np_ * 24, cudaMemcpyHostToDevice, gg_copy) !=
                  cudaSuccess;
            if (!bad) {
                k_fetch_pull<<<gg_pull_blocks(), 256, 0, gg_copy>>>(prg_d, np_);
                bad = cudaGetLastError() != cudaSuccess;
            }
            for (int k = 0; k < np_; ++k) {
                nb += (double)prg[3 * k + 2];
            }
        }
        for (int k = 0; k < j.count && !bad && np_ < 0; ++k) {
            const int64_t *q = j.ranges + 3 * k;
            if (q[2] == 0) {
                continue;       /* a row of a fixed-size list that copies nothing */
            }
            bad = gg_h2d_staged(&sg, (void *)(intptr_t)q[1], (const void *)(intptr_t)q[0],
                                (size_t)q[2], gg_copy, 1) != 0;
            nb += (double)q[2];
        }
        bad = bad || cudaEventRecord(gg_ready[j.f], gg_copy) != cudaSuccess;
        pthread_mutex_lock(&gg_mu);
        gg_fetch_error |= bad;
        gg_recorded[j.f] = 1;
        ++gg_job_head;
        pthread_cond_broadcast(&gg_cv);
        pthread_mutex_unlock(&gg_mu);
        if (gg_ms_on && !bad && nb > 0.0) {
            /* The time of the copies of this job: the jobs do not overlap. */
            cudaEventSynchronize(gg_ready[j.f]);
            double dt = gg_clock() - t0;
            pthread_mutex_lock(&gg_mu);
            gg_ms_copy_bytes += nb;
            gg_ms_copy_s += dt;
            pthread_mutex_unlock(&gg_mu);
        }
    }
    return NULL;
}

/* ---------- the copies of the cache of hot experts ----------
 * gg_cache_copy queues a list of ranges (host address, device address,
 * bytes). A worker thread of its own copies them on a stream of its own,
 * then records an event. The host memory is the map of the model file, so a
 * copy call waits for the data; the worker does that wait, not the runner.
 * gg_cache_query gives 1 when the copies of a job are done. The caller keeps
 * the ranges until then. */
#define GG_CACHE_JOBS 64

typedef struct {
    const int64_t *ranges;
    int count, id;
} gg_cjob;

static pthread_mutex_t gg_cmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gg_ccv = PTHREAD_COND_INITIALIZER;
static gg_cjob gg_cjobs[GG_CACHE_JOBS];
static int gg_cjob_head, gg_cjob_tail, gg_cjob_next;
static int gg_cissued[GG_CACHE_JOBS], gg_cid[GG_CACHE_JOBS], gg_cerror;
static cudaEvent_t gg_cev[GG_CACHE_JOBS];
static cudaStream_t gg_cstream;
static int gg_cworker_on;

static void *gg_cworker(void *arg)
{
    (void)arg;
    static gg_stager sg;
    gg_worker_bind();
    for (;;) {
        pthread_mutex_lock(&gg_cmu);
        while (gg_cjob_head == gg_cjob_tail) {
            pthread_cond_wait(&gg_ccv, &gg_cmu);
        }
        gg_cjob j = gg_cjobs[gg_cjob_head % GG_CACHE_JOBS];
        pthread_mutex_unlock(&gg_cmu);
        int bad = 0;
        for (int k = 0; k < j.count && !bad; ++k) {
            const int64_t *q = j.ranges + 3 * k;
            bad = gg_h2d_staged(&sg, (void *)(intptr_t)q[1], (const void *)(intptr_t)q[0],
                                (size_t)q[2], gg_cstream) != 0;
        }
        int slot = j.id % GG_CACHE_JOBS;
        bad = bad || cudaEventRecord(gg_cev[slot], gg_cstream) != cudaSuccess;
        pthread_mutex_lock(&gg_cmu);
        gg_cerror |= bad;
        gg_cissued[slot] = 1;
        ++gg_cjob_head;
        pthread_cond_broadcast(&gg_ccv);
        pthread_mutex_unlock(&gg_cmu);
    }
    return NULL;
}

/* ---------- host memory that the GPU reads (zero copy) ----------
 * The read-only map of the model file cannot be registered with CUDA here
 * (cudaHostRegisterReadOnly: "operation not supported"); a kernel reads
 * registered host memory at 11 to 12 GB/s over the PCIe 3.0 x16 of the
 * 3090. */
/* Map n bytes of memory of the process at p for the GPU (an anonymous copy
 * of the experts, QwenGPU._pin_experts). Return the device address, or 0. */
extern "C" int64_t gg_host_register_rw(void *p, size_t n)
{
    cudaError_t err = cudaHostRegister(p, n, cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "cudaHostRegister of %zu bytes: %s", n,
                 cudaGetErrorString(err));
        cudaGetLastError();
        return 0;
    }
    void *d = NULL;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return (int64_t)(intptr_t)d;
}

/* The PCI bus id of the device ("0000:9e:00.0"), for its NUMA node. */
extern "C" int gg_pci_bus_id(char *out, int n)
{
    int dev = 0;
    cudaGetDevice(&dev);
    return cudaDeviceGetPCIBusId(out, n, dev) == cudaSuccess ? 0 : -1;
}

extern "C" int gg_host_unregister(void *p)
{
    return cudaHostUnregister(p) == cudaSuccess ? 0 : -1;
}

/* Queue a job. Return its id (0 or more), or -1. */
extern "C" int gg_cache_copy(const int64_t *ranges, int count)
{
    if (!gg_cworker_on) {
        CK(cudaStreamCreateWithFlags(&gg_cstream, cudaStreamNonBlocking));
        for (int k = 0; k < GG_CACHE_JOBS; ++k) {
            CK(cudaEventCreateWithFlags(&gg_cev[k], cudaEventDisableTiming));
        }
        pthread_t th;
        if (pthread_create(&th, NULL, gg_cworker, NULL) != 0) {
            snprintf(gg_error, sizeof(gg_error), "no worker thread for the cache copies");
            return -1;
        }
        pthread_detach(th);
        gg_cworker_on = 1;
    }
    pthread_mutex_lock(&gg_cmu);
    if (gg_cjob_tail - gg_cjob_head >= GG_CACHE_JOBS) {
        pthread_mutex_unlock(&gg_cmu);
        snprintf(gg_error, sizeof(gg_error), "too many cache copies in the queue");
        return -1;
    }
    int id = gg_cjob_next++;
    gg_cjob j = {ranges, count, id};
    gg_cissued[id % GG_CACHE_JOBS] = 0;
    gg_cid[id % GG_CACHE_JOBS] = id;
    gg_cjobs[gg_cjob_tail % GG_CACHE_JOBS] = j;
    ++gg_cjob_tail;
    pthread_cond_broadcast(&gg_ccv);
    pthread_mutex_unlock(&gg_cmu);
    return id;
}

/* 1: the copies of job id are on the GPU. 0: not yet. -1: an error. With
 * wait, wait for the job. */
extern "C" int gg_cache_query(int id, int wait)
{
    int slot = id % GG_CACHE_JOBS;
    pthread_mutex_lock(&gg_cmu);
    while (wait && !gg_cissued[slot]) {
        pthread_cond_wait(&gg_ccv, &gg_cmu);
    }
    int issued = gg_cissued[slot] && gg_cid[slot] == id, bad = gg_cerror;
    pthread_mutex_unlock(&gg_cmu);
    if (bad) {
        snprintf(gg_error, sizeof(gg_error), "a cache copy failed");
        return -1;
    }
    if (!issued) {
        return 0;
    }
    cudaError_t q = wait ? cudaEventSynchronize(gg_cev[slot]) : cudaEventQuery(gg_cev[slot]);
    if (q == cudaErrorNotReady) {
        return 0;
    }
    CK(q);
    return 1;
}

static int gg_fetch_init(void)
{
    if (gg_worker_on) {
        return 0;
    }
    /* the highest priority: the blocks of k_fetch_pull go before those of
     * the program's kernels */
    int lo = 0, hi_ = 0;
    CK(cudaDeviceGetStreamPriorityRange(&lo, &hi_));
    CK(cudaStreamCreateWithPriority(&gg_copy, cudaStreamNonBlocking, hi_));
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

/* NP_GEMMA_GPU_FD=0 keeps k_attn_part for the int16 cache of Qwen3.5 (a
 * test). */
static int fd_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_FD");
        on = !(v && v[0] == '0');
    }
    return on;
}

static int attn_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr,
                       const int64_t *denv, int *bad)
{
    int o = (r->op == GP_ATTN_QC || r->op == GP_ATTN_QSA || r->op == GP_ATTN_Q8 ||
             r->op == GP_ATTN_V8 || r->op == GP_ATTN_TQ) ? 7 : 5;
    /* (the operand of q_heads) */
    int64_t qh = hlit(r, o, bad), kvh = hlit(r, o + 1, bad), hd = hlit(r, o + 2, bad);
    int hg = qh % kvh == 0 ? (int)cdiv(qh / kvh, ATTN_REP) : 1;
    while (qh % kvh == 0 && (qh / kvh) % hg != 0) {
        ++hg;
    }
    if ((size_t)qh * ATTN_CHUNKS * (size_t)(hd + 2) > GG_PART_FLOATS ||
        qh % kvh != 0 || qh / kvh / hg > ATTN_REP || (hd != 256 && hd != 512) ||
        (r->op == GP_ATTN_QSA && hlit(r, 10, bad) != 1)) {
        *bad = 1;
        return 0;
    }
    if (r->op == GP_ATTN_QC && hd == 256 && qh == 8 * kvh && fd_on() &&
        (size_t)qh * FD_PARTS * 258 <= GG_PART_FLOATS) {
        k_attn_fd<<<dim3((unsigned)kvh, FD_BLOCKS), 128, 0, gg_stream>>>(dr, denv, g->part);
        k_attn_fd_join<<<dim3((unsigned)qh, 2), 128, 0, gg_stream>>>(dr, denv, g->part);
        return 0;
    }
    /* A global layer of the 26B on the tensor cores (k_attn_fdtc), with the
     * queries of a group (operand 11: t, np_gemma/gpu.py attn_rows_small). */
    bool qc = r->op == GP_ATTN_QC || r->op == GP_ATTN_Q8 || r->op == GP_ATTN_V8;
    int64_t t = qc && r->tag[11] != GP_T_NONE ? hlit(r, 11, bad) : 1;
    /* a sliding layer of the 26B (operand 12: the window; k_attn_fdts) */
    if (qc && r->tag[12] != GP_T_NONE) {
        int64_t win = hlit(r, 12, bad);
        int ncs = (int)((win + FT_TMAX - 1 + FS_L - 1) / FS_L + 1);
        if (!(fdtc_on() && hd == 256 && qh == 2 * kvh && t >= 1 && t <= FT_TMAX && win > 0 &&
              ncs <= FS_MAXC && (size_t)t * qh * ncs * 258 <= GG_PART_FLOATS)) {
            *bad = 1;
            return 0;
        }
        dim3 grid((unsigned)kvh, (unsigned)ncs);
        if (t == 1) {
            fdts_run<1>(r->op, grid, dr, denv, g->part);
        } else if (t == 2) {
            fdts_run<2>(r->op, grid, dr, denv, g->part);
        } else {
            fdts_run<3>(r->op, grid, dr, denv, g->part);
        }
        k_attn_fdts_join<<<dim3((unsigned)(t * qh), 2), 128, 0, gg_stream>>>(dr, denv, g->part, ncs);
        return 0;
    }
    if (qc && fdtc_on() && hd == 512 && qh == 8 * kvh && t >= 1 && t <= FT_TMAX &&
        (size_t)t * qh * fdtc_chunks((int)kvh) * (hd + 2) <= GG_PART_FLOATS) {
        dim3 grid((unsigned)kvh, (unsigned)fdtc_chunks((int)kvh));
        if (t == 1) {
            fdtc_run<1>(r->op, grid, dr, denv, g->part);
        } else if (t == 2) {
            fdtc_run<2>(r->op, grid, dr, denv, g->part);
        } else {
            fdtc_run<3>(r->op, grid, dr, denv, g->part);
        }
        k_attn_join<<<dim3((unsigned)(t * qh), (unsigned)cdiv(hd, 128)), 128, 0, gg_stream>>>(
            dr, denv, g->part, (int)grid.y);
        return 0;
    }
    if (t != 1) {
        *bad = 1;
        return 0;
    }
    /* The shapes of the Gemma 4 26B: one pass (k_attn_fdt). */
    if ((r->op == GP_ATTN_QC || r->op == GP_ATTN_Q8 || r->op == GP_ATTN_V8) && fd_on() &&
        (size_t)qh * FD_PARTS * (hd + 2) <= GG_PART_FLOATS &&
        ((hd == 256 && qh == 2 * kvh) || (hd == 512 && qh == 8 * kvh))) {
        dim3 grid((unsigned)kvh, FD_BLOCKS), jg((unsigned)qh, (unsigned)(hd / 128));
        bool q8 = r->op == GP_ATTN_Q8, v8 = r->op == GP_ATTN_V8;
        if (hd == 256) {
            if (q8) {
                k_attn_fdt<256, 2, true, true><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            } else if (v8) {
                k_attn_fdt<256, 2, false, true><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            } else {
                k_attn_fdt<256, 2><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            }
            k_attn_fdt_join<256, 2><<<jg, 128, 0, gg_stream>>>(dr, denv, g->part);
        } else {
            if (q8) {
                k_attn_fdt<512, 8, true, true><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            } else if (v8) {
                k_attn_fdt<512, 8, false, true><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            } else {
                k_attn_fdt<512, 8><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            }
            k_attn_fdt_join<512, 8><<<jg, 128, 0, gg_stream>>>(dr, denv, g->part);
        }
        return 0;
    }
    if (r->op == GP_ATTN_QSA && hd == 256 && qh / kvh <= 16 && qsa_tc_on() && part_tc_on() &&
        (hlit(r, 15, bad) == 1 || hlit(r, 15, bad) == 4)) {
        /* the heads of a key head in one tile (k_attn_part_tc) */
        static int attr = 0;
        if (!attr) {
            cudaFuncSetAttribute(k_attn_part_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)QT_SMEM);
            cudaFuncSetAttribute(k_attn_part_tc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)QT_SMEM);
            attr = 1;
        }
        dim3 grid((unsigned)kvh, ATTN_CHUNKS);
        if (qsa_tc_on() == 2) {
            k_attn_part_tc<0><<<grid, 128, QT_SMEM, gg_stream>>>(dr, denv, g->part);
        } else {
            k_attn_part_tc<1><<<grid, 128, QT_SMEM, gg_stream>>>(dr, denv, g->part);
        }
        k_attn_join<<<dim3((unsigned)qh, (unsigned)cdiv(hd, 128)), 128, 0, gg_stream>>>(
            dr, denv, g->part, 0);
        return 0;
    }
    k_attn_part<<<dim3((unsigned)(kvh * hg), ATTN_CHUNKS), 256, 0, gg_stream>>>(dr, denv, g->part,
                                                                              hg);
    k_attn_join<<<dim3((unsigned)qh, (unsigned)cdiv(hd, 128)), 128, 0, gg_stream>>>(
        dr, denv, g->part, 0);
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

template <int HD, int HB, int ST>
static void flash_qc_run(const gp_rec *dr, const int64_t *denv, int kvh, int t, int G)
{
    size_t smem = flash_qc_dims<HD>::smem(HB, ST);
    cudaFuncSetAttribute(k_flash_qc_tc<HD, HB, ST>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((unsigned)kvh, (unsigned)cdiv(t, 16), (unsigned)(G / HB));
    k_flash_qc_tc<HD, HB, ST><<<grid, 32 * HB * (HD / 256), smem, gg_stream>>>(dr, denv);
}

template <int HD, int HB, int NQT, int FKS = FK3, bool KQ8 = false, bool VQ8 = false,
          bool TQ = false>
static void flash_h_run(const gp_rec *dr, const int64_t *denv, int kvh, int t, int G)
{
    typedef flash_h_dims<HD, HB, NQT, FKS, KQ8, VQ8, TQ> D;
    size_t smem = D::smem();
    cudaFuncSetAttribute(k_flash_qc_h<HD, HB, NQT, FKS, KQ8, VQ8, TQ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((unsigned)kvh, (unsigned)cdiv(t, 16 * NQT), (unsigned)(G / HB));
    k_flash_qc_h<HD, HB, NQT, FKS, KQ8, VQ8, TQ><<<grid, 32 * D::WARPS, smem, gg_stream>>>(
        dr, denv);
}

/* NP_GEMMA_GPU_FLASH_FK: the keys of a step of k_flash_qc_h for a head of
 * 256 values (16 or 32). For the 40 layers of the 12B with a window and
 * 8192 tokens: 255 ms with 16, 235 ms with 32. */
static int flash_fk(void)
{
    static int fk = -1;
    if (fk < 0) {
        const char *v = getenv("NP_GEMMA_GPU_FLASH_FK");
        fk = v ? atoi(v) : 32;
    }
    return fk;
}

/* NP_GEMMA_GPU_FLASH512: the head of 512 values takes k_flash_qc_h with HB
 * query heads in a block (4, or 2 for a test; 0 keeps k_flash_qc_tc). The
 * kernel takes 254 registers, so an SM runs one block: 4 warps with HB 2,
 * 8 with HB 4. For the 8 global layers of the 12B and a prompt of 8192
 * tokens (chunks of 2048): 714 ms with k_flash_qc_tc, 974 ms with HB 2,
 * 475 ms with HB 4. A key head with fewer than 4 query heads keeps
 * k_flash_qc_tc. NP_GEMMA_GPU_FLASH256: the head of 256 values takes
 * k_flash_qc_h with 2 query heads and this many tiles of 16 queries in a
 * block (0 keeps k_flash_qc_tc). */
static int flash_env(const char *name, int dflt)
{
    const char *v = getenv(name);
    return v ? atoi(v) : dflt;
}

static int flash512_hb(void)
{
    static int hb = -1;
    if (hb < 0) {
        hb = flash_env("NP_GEMMA_GPU_FLASH512", 4);
    }
    return hb;
}

static int flash256_qt(void)
{
    static int qt = -1;
    if (qt < 0) {
        qt = flash_env("NP_GEMMA_GPU_FLASH256", 4);
    }
    return qt;
}

/* Launch k_flash_qc_tc for a GP_ATTN_QC_MT record. Return -1 when it does
 * not take the shape. NP_GEMMA_GPU_FLASH=1 selects k_flash_tc<0> for a test. */
static int flash_qc_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int *bad)
{
    static int old = -1;
    if (old < 0) {
        const char *v = getenv("NP_GEMMA_GPU_FLASH");
        old = v && v[0] == '1';
    }
    if (old) {
        return -1;
    }
    int64_t qh = hlit(r, 7, bad), kvh = hlit(r, 8, bad), hd = hlit(r, 9, bad);
    int64_t t = hlit(r, 10, bad);
    if (kvh <= 0 || qh % kvh) {
        return -1;
    }
    int G = (int)(qh / kvh);
    if (hd == 256 && G % 2 == 0 && flash256_qt() > 0) {
        switch (flash256_qt()) {
        case 1: flash_h_run<256, 2, 1>(dr, denv, (int)kvh, (int)t, G); break;
        case 2: flash_h_run<256, 2, 2>(dr, denv, (int)kvh, (int)t, G); break;
        default:
            if (flash_fk() == 32) {
                flash_h_run<256, 2, 4, 32>(dr, denv, (int)kvh, (int)t, G);
            } else {
                flash_h_run<256, 2, 4>(dr, denv, (int)kvh, (int)t, G);
            }
        }
        return 0;
    }
    if (hd == 256) {
        if (G % 4 == 0) {
            flash_qc_run<256, 4, 2>(dr, denv, (int)kvh, (int)t, G);
        } else if (G % 2 == 0) {
            flash_qc_run<256, 2, 2>(dr, denv, (int)kvh, (int)t, G);
        } else {
            flash_qc_run<256, 1, 2>(dr, denv, (int)kvh, (int)t, G);
        }
        return 0;
    }
    if (hd == 512) {
        if (flash512_hb() == 4 && G % 4 == 0) {
            flash_h_run<512, 4, 1>(dr, denv, (int)kvh, (int)t, G);
        } else if (flash512_hb() == 2 && G % 2 == 0) {
            flash_h_run<512, 2, 1>(dr, denv, (int)kvh, (int)t, G);
        } else if (G % 2 == 0) {
            flash_qc_run<512, 2, 1>(dr, denv, (int)kvh, (int)t, G);
        } else {
            flash_qc_run<512, 1, 1>(dr, denv, (int)kvh, (int)t, G);
        }
        return 0;
    }
    return -1;
}

/* k_flash_qc_h over the int8 values (GP_ATTN_Q8_MT; GP_ATTN_V8_MT with int16
 * keys: KQ8 false), the only kernel of a large group for them. Return -1 when it does not take the shape. */
template <bool KQ8>
static int flash_q8_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int *bad)
{
    int64_t qh = hlit(r, 7, bad), kvh = hlit(r, 8, bad), hd = hlit(r, 9, bad);
    int64_t t = hlit(r, 10, bad);
    if (kvh <= 0 || qh % kvh) {
        return -1;
    }
    int G = (int)(qh / kvh);
    if (hd == 256 && G % 2 == 0) {
        flash_h_run<256, 2, 4, 32, KQ8, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    if (hd == 512 && G % 4 == 0) {
        flash_h_run<512, 4, 1, FK3, KQ8, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    if (hd == 512 && G % 2 == 0) {
        flash_h_run<512, 2, 1, FK3, KQ8, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    return -1;
}

/* k_flash_qc_h over the TQ6 cache (GP_ATTN_TQ_MT: q and out rotated), the
 * only kernel of a large group for it. Return -1 when it does not take the
 * shape. */
static int flash_tq_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int *bad)
{
    int64_t qh = hlit(r, 7, bad), kvh = hlit(r, 8, bad), hd = hlit(r, 9, bad);
    int64_t t = hlit(r, 10, bad);
    if (kvh <= 0 || qh % kvh) {
        return -1;
    }
    int G = (int)(qh / kvh);
    if (hd == 256 && G % 2 == 0) {
        flash_h_run<256, 2, 4, 32, false, false, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    if (hd == 512 && G % 2 == 0) {
        flash_h_run<512, 2, 1, FK3, false, false, true>(dr, denv, (int)kvh, (int)t, G);
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
/* NP_GEMMA_GPU_MT_ROWS=0 keeps k_mt_gemv_n for the int4 matrices of a group
 * of 1 to 4 tokens (the form before k_mt_int4_rows). */
static int mt_rows_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_MT_ROWS");
        on = !(v && v[0] == '0');
    }
    return on;
}

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
    if (t <= gg_gemv_max && t <= 4 && mt_rows_on()) {
        /* A verify group: the bits of a step for each token (k_mt_int4_rows). */
        unsigned grid = (unsigned)cdiv(rows, ROWS_PER_BLOCK), blk = 32 * ROWS_PER_BLOCK;
        const uint8_t *wb = (const uint8_t *)w;
        switch (t) {
        case 1: k_mt_int4_rows<1><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols); break;
        case 2: k_mt_int4_rows<2><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols); break;
        case 3: k_mt_int4_rows<3><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols); break;
        default: k_mt_int4_rows<4><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols);
        }
    } else if (t <= gg_gemv_max) {
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
    } else if (gg_tc && ((gg_tc == 8 && g->tc) || g->i8) && cols % Q8K == 0 &&
               (size_t)t * cols <= g->xh_n) {
        /* int8 x: the scratch of xh holds the int8 values, then the scales.
         * i8 2 (the int16 form, k_quant_x2): the hi values, the lo values,
         * then the scales (gg_load gives xh the room). */
        size_t n = (size_t)t * cols;
        int8_t *xq = (int8_t *)g->xh;
        if (g->i8 == 2) {
            int8_t *xl = xq + n;
            float *xs = (float *)((char *)g->xh + ((2 * n + 255) & ~(size_t)255));
            if (quant) {
                k_quant_x2<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(
                    x, xq, xl, xs, NULL, n / 32);
            }
            dim3 grid((unsigned)cdiv(t, T2M), (unsigned)cdiv(rows, T2N));
            static int attr2 = 0;
            if (!attr2) {
                cudaFuncSetAttribute(k_gemm_q8<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)Q8SMEM_X2);
                cudaFuncSetAttribute(k_gemm_q8<0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)Q8SMEM_X2);
                attr2 = 1;
            }
            if ((cols / 32) % 8 == 0 && ((uintptr_t)w & 15) == 0) {
                k_gemm_q8<1, 1><<<grid, 256, Q8SMEM_X2, gg_stream>>>(
                    xq, xs, (const uint8_t *)w, out, t, rows, cols, xl);
            } else {
                k_gemm_q8<0, 1><<<grid, 256, Q8SMEM_X2, gg_stream>>>(
                    xq, xs, (const uint8_t *)w, out, t, rows, cols, xl);
            }
            return;
        }
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
/* NP_GEMMA_GPU_KQTC=0 keeps the float32 tiles for the products of large
 * groups of Qwen3.8 (a test). */
static int i8x_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_I8X");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* NP_GEMMA_GPU_MOE_F32=1 (NP_GEMMA_MOE_X16): the experts of a group
 * (GP_KQ_GROUP_MOE) in float32 (k_qmoe_gu, k_qmoe_dn: x not quantized), not
 * on the int8 tensor cores. */
static int moe_f32_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_MOE_F32");
        on = v && v[0] == '1';
    }
    return on;
}

static int kt_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_KQTC");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* The shared memory of the instances of the tensor-core kernels (once). */
static void kt_attr(void)
{
    static int done = 0;
    if (done) {
        return;
    }
    done = 1;
    int a = (int)KT_SMEM(128, 2), b = (int)KT_SMEM(64, 3);
    cudaFuncSetAttribute(k_kq_tc<KT_Q8R>, cudaFuncAttributeMaxDynamicSharedMemorySize, a);
#define KT_SET(F) \
    cudaFuncSetAttribute(k_qmoe_gu_tc<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, b); \
    cudaFuncSetAttribute(k_qmoe_dn_tc<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
    KT_SET(KT_Q8R) KT_SET(KT_Q80) KT_SET(KT_Q51) KT_SET(KT_Q4K) KT_SET(KT_NV4) KT_SET(KT_NVX)
    KT_SET(KT_Q6K)
#undef KT_SET
    cudaFuncSetAttribute(k_moe_gemm_q4x<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
    cudaFuncSetAttribute(k_moe_gemm_q4x<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)KT_SMEM_X2(64, 2));
}

/* The lo plane of the int16 form in the int8 scratch of g (i8 2): after
 * the int8 values and the xs and xsum of kqx_n values (gg_load). */
static int8_t *kt_lo(const gg_prog *g)
{
    return g->kqx + g->kqx_n + g->kqx_n / 32 * 8;
}

/* Quantize n values (a multiple of 32) at x to the int16 form (k_quant_x2)
 * in the int8 scratch of g: the hi plane, xs and xsum as kt_quant, and the
 * lo plane at kt_lo. */
static void kt_quant_x2(const gg_prog *g, const float *x, size_t n)
{
    float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
    k_quant_x2<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(
        x, g->kqx, kt_lo(g), xs, xsum, n / 32);
}

/* Quantize n values (a multiple of 32) at x to the int8 scratch of g. */
static void kt_quant(const gg_prog *g, const float *x, size_t n)
{
    int8_t *xq = g->kqx;
    float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
    k_kq_quant_x<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(x, xq, xs, xsum,
                                                                                   n / 32);
}

/* The x of record r if its launch quantizes x to int8 with kt_quant for a
 * product of one token (GP_KQ_LINEAR, GP_KQ_MULTI), else NULL. *n is the
 * count of values. The conditions are those of gg_launch. */
/* The fused gate, up and GELU (k_kq_glu_i8). NP_GEMMA_GPU_GLU=0 gives the
 * two records. */
static int glu_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_GLU");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* A record that the launch before did (k_kq_glu_i8 does the GELU record
 * after its GP_KQ_MULTI); gg_launch skips it. */
static const gp_rec *gg_skip;

static const float *kq_i8_input(const gg_prog *g, const gp_rec *r, size_t *n)
{
    int bad = 0;
    if (r->op == GP_KQ_LINEAR) {
        int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad);
        if (t <= MT_MAX && kq_i8_type((int)hlit(r, 5, &bad)) && i8x_on() && cols % 32 == 0 &&
            (size_t)(t * cols) <= g->kqx_n && !bad) {
            *n = (size_t)(t * cols);
            return (const float *)(intptr_t)hi(r, g->henv, 3);
        }
    } else if (r->op == GP_KQ_MULTI) {
        int64_t t = hlit(r, 2, &bad), cols = hlit(r, 1, &bad), q8r = 0;
        for (int m = 0; m < hlit(r, 3, &bad); ++m) {
            q8r |= kq_i8_type((int)hlit(r, 5 + 4 * m, &bad));
        }
        if (q8r && i8x_on() && cols % 32 == 0 && (size_t)(t * cols) <= g->kqx_n && !bad) {
            *n = (size_t)(t * cols);
            return (const float *)(intptr_t)hi(r, g->henv, 0);
        }
    }
    return NULL;
}

/* The x of a large int4 product that quantizes x with k_quant_q8 (the
 * int8 path of gemm_launch), and its count of values; else NULL. */
static const float *q8_gemm_input(const gg_prog *g, const gp_rec *r, size_t *n)
{
    int64_t cols, t;
    if (r->op == GP_INT4_LINEAR_MT) {
        cols = r->v[5];
        t = r->v[6];
    } else if (r->op == GP_INT4_MULTI4_MT) {
        cols = r->v[1];
        t = r->v[2];
    } else {
        return NULL;
    }
    if (r->tag[0] == GP_T_SLOT || t <= gg_gemv_max || !gg_tc ||
        !((gg_tc == 8 && g->tc) || g->i8) || cols % Q8K != 0 ||
        (size_t)(t * cols) > g->xh_n) {
        return NULL;
    }
    *n = (size_t)(t * cols);
    return (const float *)(intptr_t)r->v[0];
}

/* The int8 x that the launch before wrote (k_add_norm_v, k_gelu_mul_rows_v):
 * the source and the count of values. gg_launch clears it, so only the next
 * record can use it. */
static const float *gg_qx_src;
static size_t gg_qx_n;

static int an_v_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_AN_V");
        on = !(v && v[0] == '0');
    }
    return on;
}

static int qx_env_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_QX_FUSE");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* If the record after r (in the same program) quantizes out (n values), give
 * the int8 buffers to the kernel of r, and let that record skip kt_quant. */
static int8_t *qx_fuse(const gg_prog *g, const gp_rec *r, const void *out, size_t n,
                       float **xs, float **xsum)
{
    size_t m = 0;
    const gp_rec *nx = r + 1, *end = g->hcode + g->n_code;
    /* the GP_KQ_QUANT records between (the CPU form; nothing on the GPU) */
    while (nx < end && nx->op == GP_KQ_QUANT) {
        ++nx;
    }
    if (!qx_env_on() || out == NULL || nx >= end || n % 32 != 0) {
        return NULL;
    }
    if (q8_gemm_input(g, nx, &m) == (const float *)out && m == n) {
        if (g->i8 == 2) {
            return NULL;        /* the int16 form: gemm_launch quantizes x */
        }
        /* The layout of gemm_launch: the int8 values, then (at a multiple of
         * 256 bytes) the scales; the sums of k_kq_* after them are unused. */
        *xs = (float *)((char *)g->xh + ((n + 255) & ~(size_t)255));
        *xsum = *xs + n / 32;
        gg_qx_src = (const float *)out;
        gg_qx_n = n;
        return (int8_t *)g->xh;
    }
    if (kq_i8_input(g, nx, &m) != (const float *)out || m != n) {
        return NULL;
    }
    *xs = (float *)(g->kqx + g->kqx_n);
    *xsum = *xs + g->kqx_n / 32;
    gg_qx_src = (const float *)out;
    gg_qx_n = n;
    return g->kqx;
}

static int al16(int64_t p)
{
    return p % 16 == 0;
}

/* The kernels of the media encoders (defined with their entry points below). */
__global__ void k_enc_gemm(const float *x, const void *w, int wbf, const float *b, float *y,
                           int n, int m, int k, float imin, float imax, float omin, float omax);
__global__ void k_enc_rms(const float *x, const float *w, float *y, int rows, int d, float eps);
__global__ void k_enc_gelu_mul(const float *g, const float *u, float *y, int64_t n);
__global__ void k_enc_lnorm(const float *x, const float *w, const float *b, float *y, int rows,
                            int d, float eps);
__global__ void k_enc_gelu(const float *x, float *y, int64_t n, int use_erf);
__global__ void k_enc_add(float *x, const float *y, int64_t n, float sc);
__global__ void k_enc_silu(const float *x, float *y, int64_t n);
__global__ void k_enc_clamp(const float *x, float *y, int64_t n, float lo, float hi);
__global__ void k_enc_bias_clamp(float *y, const float *b, int rows, int cols, float lo, float hi);
__global__ void k_enc_mul_vec(const float *x, const float *vec, float *y, int rows, int cols);
__global__ void k_enc_glu(const float *x, float *y, int rows, int cols);
__global__ void k_enc_dwconv(const float *x, const float *w, float *y, int t, int c, int kw);
__global__ void k_enc_local_attn(const float *q, const float *k, const float *v, const float *r,
                                 const int *valid, float *o, int t, int heads, int hd, int span,
                                 float cap);
__global__ void k_enc_rope2d(float *x, const int *pos, const float *inv, int n, int heads, int hd);
__global__ void k_enc_attn(const float *q, const float *k, const float *v, float *o,
                           int n, int heads, int hd);

/* The float32 of a literal operand. */
static float hlitf(const gp_rec *r, int k, int *bad)
{
    uint32_t u = (uint32_t)hlit(r, k, bad);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

static int gg_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr, const int64_t *denv)
{
    int bad = 0;
    cudaStream_t s = gg_stream;
    const int T = 256;
    const int W = 32 * ROWS_PER_BLOCK;
    if (r->op == GP_KQ_QUANT) {
        return 0;       /* nothing on the GPU; the int8 x of qx_fuse stays */
    }
    const float *qx_src = gg_qx_src;
    size_t qx_n = gg_qx_n;
    gg_qx_src = NULL;
    if (r == gg_skip) {
        gg_skip = NULL;
        return 0;
    }
    gg_skip = NULL;
    switch (r->op) {
    /* ---- the media encoders: the operands of bf16_linear.c gp_step ---- */
    case GP_ENC_LINEAR: {
        int n = (int)hlit(r, 5, &bad), m = (int)hlit(r, 6, &bad), k = (int)hlit(r, 7, &bad);
        static int enc_tc = -1;
        if (enc_tc < 0) {
            const char *v = getenv("NP_GEMMA_GPU_ENC_TC");
            enc_tc = v == NULL || strcmp(v, "0") != 0;
        }
        if (enc_tc && gg_tc && g->tc && hlit(r, 2, &bad) && k % 8 == 0 && n > MT_MAX &&
            (size_t)n * k <= g->xh_n) {
            /* bfloat16 W on the tensor cores: x to float16 (with the input
             * clamps), then k_enc_gemm_tc (the bias and the output clamps) */
            size_t nk = (size_t)n * k;
            k_enc_to_half<<<(unsigned)cdiv((int64_t)nk, T), T, 0, s>>>(
                (const float *)(intptr_t)hi(r, g->henv, 0), g->xh, nk, hlitf(r, 8, &bad),
                hlitf(r, 9, &bad));
            k_enc_gemm_tc<<<dim3((unsigned)cdiv(n, T2M), (unsigned)cdiv(m, T2N)), 256, 0, s>>>(
                g->xh, (const uint16_t *)(intptr_t)hi(r, g->henv, 1),
                (const float *)(intptr_t)hi(r, g->henv, 3), (float *)(intptr_t)hi(r, g->henv, 4),
                n, m, k, hlitf(r, 10, &bad), hlitf(r, 11, &bad));
            break;
        }
        dim3 grid((unsigned)cdiv(m, 64), (unsigned)cdiv(n, 64));
        k_enc_gemm<<<grid, 256, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const void *)(intptr_t)hi(r, g->henv, 1),
            (int)hlit(r, 2, &bad), (const float *)(intptr_t)hi(r, g->henv, 3),
            (float *)(intptr_t)hi(r, g->henv, 4), n, m, k, hlitf(r, 8, &bad), hlitf(r, 9, &bad),
            hlitf(r, 10, &bad), hlitf(r, 11, &bad));
        break;
    }
    case GP_ENC_RMS: {
        int rows = (int)hlit(r, 3, &bad);
        k_enc_rms<<<(unsigned)cdiv(rows, 8), 256, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), rows, (int)hlit(r, 4, &bad), hlitf(r, 5, &bad));
        break;
    }
    case GP_ENC_LNORM: {
        int rows = (int)hlit(r, 4, &bad);
        k_enc_lnorm<<<(unsigned)cdiv(rows, 8), 256, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (const float *)(intptr_t)hi(r, g->henv, 2), (float *)(intptr_t)hi(r, g->henv, 3), rows,
            (int)hlit(r, 5, &bad), hlitf(r, 6, &bad));
        break;
    }
    case GP_ENC_GELU: {
        int64_t n = hlit(r, 2, &bad);
        k_enc_gelu<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 1), n,
            (int)hlit(r, 3, &bad));
        break;
    }
    case GP_ENC_GELU_MUL: {
        int64_t n = hlit(r, 3, &bad);
        k_enc_gelu_mul<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), n);
        break;
    }
    case GP_ENC_ADD: {
        int64_t n = hlit(r, 2, &bad);
        k_enc_add<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1), n,
            hlitf(r, 3, &bad));
        break;
    }
    case GP_ENC_CLAMP: {
        int64_t n = hlit(r, 2, &bad);
        k_enc_clamp<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 1), n,
            hlitf(r, 3, &bad), hlitf(r, 4, &bad));
        break;
    }
    case GP_ENC_BIAS_CLAMP: {
        int rows = (int)hlit(r, 2, &bad), cols = (int)hlit(r, 3, &bad);
        k_enc_bias_clamp<<<(unsigned)cdiv((int64_t)rows * cols, T), T, 0, s>>>(
            (float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            rows, cols, hlitf(r, 4, &bad), hlitf(r, 5, &bad));
        break;
    }
    case GP_ENC_SILU: {
        int64_t n = hlit(r, 2, &bad);
        k_enc_silu<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 1), n);
        break;
    }
    case GP_ENC_MUL_VEC: {
        int rows = (int)hlit(r, 3, &bad), cols = (int)hlit(r, 4, &bad);
        k_enc_mul_vec<<<(unsigned)cdiv((int64_t)rows * cols, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), rows, cols);
        break;
    }
    case GP_ENC_GLU: {
        int rows = (int)hlit(r, 2, &bad), cols = (int)hlit(r, 3, &bad);
        k_enc_glu<<<(unsigned)cdiv((int64_t)rows * cols, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 1),
            rows, cols);
        break;
    }
    case GP_ENC_DWCONV: {
        int t = (int)hlit(r, 3, &bad), c = (int)hlit(r, 4, &bad);
        k_enc_dwconv<<<(unsigned)cdiv((int64_t)t * c, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), t, c, (int)hlit(r, 5, &bad));
        break;
    }
    case GP_ENC_LOCAL_ATTN: {
        int t = (int)hlit(r, 6, &bad), heads = (int)hlit(r, 7, &bad);
        int span = (int)hlit(r, 9, &bad);
        if (span > 32) {
            bad = 1;
            break;
        }
        /* a warp for each (query, head): lane j < span takes key j */
        k_enc_local_attn<<<(unsigned)cdiv((int64_t)t * heads, 4), 128, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (const float *)(intptr_t)hi(r, g->henv, 2), (const float *)(intptr_t)hi(r, g->henv, 3),
            (const int *)(intptr_t)hi(r, g->henv, 4), (float *)(intptr_t)hi(r, g->henv, 5),
            t, heads, (int)hlit(r, 8, &bad), span, hlitf(r, 10, &bad));
        break;
    }
    case GP_ENC_ROPE2D: {
        int n = (int)hlit(r, 3, &bad), heads = (int)hlit(r, 4, &bad), hd = (int)hlit(r, 5, &bad);
        int64_t total = (int64_t)n * heads * 2 * (hd / 4);
        k_enc_rope2d<<<(unsigned)cdiv(total, T), T, 0, s>>>(
            (float *)(intptr_t)hi(r, g->henv, 0), (const int *)(intptr_t)hi(r, g->henv, 1),
            (const float *)(intptr_t)hi(r, g->henv, 2), n, heads, hd);
        break;
    }
    case GP_ENC_ATTN: {
        int n = (int)hlit(r, 4, &bad), heads = (int)hlit(r, 5, &bad), hd = (int)hlit(r, 6, &bad);
        if (hd > 128) {
            bad = 1;
            break;
        }
        size_t smem = (size_t)(2 * 32 * (hd + 1) + 4 * hd) * sizeof(float);
        k_enc_attn<<<dim3((unsigned)cdiv(n, 4), (unsigned)heads), 128, smem, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (const float *)(intptr_t)hi(r, g->henv, 2), (float *)(intptr_t)hi(r, g->henv, 3),
            n, heads, hd);
        break;
    }
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
        unsigned rows = (unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad) + hlit(r, 7, &bad));
        int64_t hd = hlit(r, 12, &bad);
        static int fuse = -1;
        if (fuse < 0) {
            const char *v = getenv("NP_GEMMA_GPU_QKV_FUSE");
            fuse = !(v && v[0] == '0');
        }
        if (fuse && hd == 256) {
            k_qkv_norm_rope<2><<<rows, 128, 0, s>>>(dr, denv, nx, rx);
        } else if (fuse && hd == 512) {
            k_qkv_norm_rope<4><<<rows, 128, 0, s>>>(dr, denv, nx, rx);
        } else {
            k_qkv_norm<<<rows, 128, 0, s>>>(dr, denv, nx);
            k_rope<<<(unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad)), 128, 0, s>>>(dr, denv,
                                                                                  rx);
        }
        break;
    }
    case GP_KV_WRITE:
        k_kv_write<<<(unsigned)cdiv(hlit(r, 8, &bad) / 32, KVW_WARPS), 32 * KVW_WARPS, 0, s>>>(
            dr, denv);
        break;
    case GP_KV_WRITE8:
    case GP_KV_WRITEV8: {
        unsigned grid = (unsigned)cdiv(hlit(r, 8, &bad) / 32, KVW_WARPS);
        if (r->op == GP_KV_WRITE8) {
            k_kv_write8<true><<<grid, 32 * KVW_WARPS, 0, s>>>(dr, denv);
        } else {
            k_kv_write8<false><<<grid, 32 * KVW_WARPS, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_KV_WRITETQ:
        k_kv_write_tq<<<(unsigned)cdiv(hlit(r, 8, &bad) / 32, KVW_WARPS), 32 * KVW_WARPS, 0,
                        s>>>(dr, denv);
        break;
    case GP_TQ_ROT:
        k_tq_rot<<<(unsigned)cdiv(hlit(r, 1, &bad), KVW_WARPS), 32 * KVW_WARPS, 0, s>>>(dr, denv);
        break;
    case GP_ATTN_TQ_MT:
        /* a large group over the TQ6 cache (k_flash_qc_h with TQ) */
        if (hlit(r, 10, &bad) <= MT_MAX || flash_tq_launch(r, dr, denv, &bad) != 0) {
            bad = 1;
        }
        break;
    case GP_ATTN_Q8_MT:
    case GP_ATTN_V8_MT:
        /* a large group over the int8 values; a small one has GP_ATTN_Q8 or
         * GP_ATTN_V8 for each query (SplitCompiler.attn_rows_small) */
        if (hlit(r, 10, &bad) <= MT_MAX ||
            (r->op == GP_ATTN_Q8_MT ? flash_q8_launch<true>(r, dr, denv, &bad)
                                    : flash_q8_launch<false>(r, dr, denv, &bad)) != 0) {
            bad = 1;
        }
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
    case GP_ATTN_QSA:
        if (hlit(r, 10, &bad) > 1) {
            int64_t nq = hlit(r, 7, &bad), nk = hlit(r, 8, &bad), rep = nk > 0 ? nq / nk : 0;
            int hg = rep == 12 ? 2 : (rep == 8 ? 1 : 0);
            int64_t form = hlit(r, 15, &bad);
            int tcq = qsa_tc_on();
            if (hlit(r, 9, &bad) != 256 || hg == 0 || nq % nk != 0) {
                bad = 1;
            } else if (tcq && (form == 1 || form == 4) && rep <= 16 && hlit(r, 14, &bad) == 0 &&
                       dense_tc_on()) {
                /* every query sees all the positions (k_attn_dense_tc) */
                static int dattr = 0;
                if (!dattr) {
                    cudaFuncSetAttribute(k_attn_dense_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)QD_SMEM);
                    cudaFuncSetAttribute(k_attn_dense_tc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)QD_SMEM);
                    dattr = 1;
                }
                dim3 grid((unsigned)cdiv(hlit(r, 10, &bad), QD_W), (unsigned)nk);
                if (tcq == 2) {
                    k_attn_dense_tc<0><<<grid, 32 * QD_W, QD_SMEM, s>>>(dr, denv);
                } else {
                    k_attn_dense_tc<1><<<grid, 32 * QD_W, QD_SMEM, s>>>(dr, denv);
                }
            } else if (tcq && (form == 1 || form == 4) && rep <= 16) {
                static int attr = 0;
                if (!attr) {
                    cudaFuncSetAttribute(k_attn_qsa_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)QT_SMEM);
                    cudaFuncSetAttribute(k_attn_qsa_tc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)QT_SMEM);
                    attr = 1;
                }
                dim3 grid((unsigned)hlit(r, 10, &bad), (unsigned)nk);
                if (tcq == 2) {
                    k_attn_qsa_tc<0><<<grid, 128, QT_SMEM, s>>>(dr, denv);
                } else {
                    k_attn_qsa_tc<1><<<grid, 128, QT_SMEM, s>>>(dr, denv);
                }
            } else if (hg == 2) {
                k_attn_qsa_mt<6><<<dim3((unsigned)hlit(r, 10, &bad), (unsigned)(nk * hg)), 128, 0, s>>>(
                    dr, denv, hg);
            } else {
                k_attn_qsa_mt<8><<<dim3((unsigned)hlit(r, 10, &bad), (unsigned)(nk * hg)), 128, 0, s>>>(
                    dr, denv, hg);
            }
            break;
        }
        attn_launch(g, r, dr, denv, &bad);
        break;
    case GP_ATTN_F32:
    case GP_ATTN_QC:
    case GP_ATTN_Q8:
    case GP_ATTN_V8:
    case GP_ATTN_TQ:
        attn_launch(g, r, dr, denv, &bad);
        break;
    case GP_HOT_SPLIT:
        k_hot_split<<<1, 1, 0, s>>>(dr, denv);
        break;
    case GP_HOT_MOE: {
        unsigned t = (unsigned)hlit(r, 15, &bad);
        unsigned k = (unsigned)hlit(r, 10, &bad) * t;
        /* KQ_Q4X (operand 16): a block for each group of 16 rows */
        int per = hlit(r, 16, &bad) == 1 ? 16 : ROWS_PER_BLOCK;
        k_hot_gu<<<dim3((unsigned)cdiv(hlit(r, 11, &bad), per), k), W, 0, s>>>(dr, denv);
        k_hot_gelu<<<k, T, 0, s>>>(dr, denv);
        k_hot_dn<<<dim3((unsigned)cdiv(hlit(r, 13, &bad), per), k), W, 0, s>>>(dr, denv);
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
    case GP_SOFTCAP:
        k_softcap<<<(unsigned)cdiv(hlit(r, 1, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_COUNT:
        k_count<<<4, T, 0, s>>>(dr, denv);
        break;
    case GP_ADD_NORM: {
        int64_t rows = hlit(r, 4, &bad), cols = hlit(r, 5, &bad);
        int64_t o2 = hi(r, g->henv, 9);
        int ok = cols % 4 == 0 && cols <= 4 * AN_V * T && an_v_on();
        for (int k = 0; k < 4 && ok; ++k) {
            ok = al16(hi(r, g->henv, k));
        }
        ok = ok && al16(o2) && al16(hi(r, g->henv, 8));
        if (ok) {
            float *xs = NULL, *xsum = NULL;
            int8_t *q = cols % 32 != 0 ? NULL :
                        qx_fuse(g, r, (const void *)(intptr_t)(o2 ? o2 : hi(r, g->henv, 3)),
                                (size_t)(rows * cols), &xs, &xsum);
            k_add_norm_v<<<(unsigned)rows, T, 0, s>>>(dr, denv, q, xs, xsum);
        } else {
            k_add_norm<<<(unsigned)rows, T, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_HOT_SPLIT_MT:
        k_hot_split_mt<<<(unsigned)cdiv(hlit(r, 5, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_INT4_LINEAR_MT: {
        /* x, w, s, out, rows, cols, t. The record before can have written
         * the int8 x already (qx_fuse). */
        size_t m = 0;
        const float *xq = q8_gemm_input(g, r, &m);
        gemm_launch(g, r, dr, denv, 0, 1, 3, 4, 5, 6, &bad,
                    !(xq != NULL && xq == qx_src && m == qx_n));
        break;
    }
    case GP_INT4_MULTI4_MT: {
        /* x, cols, t, then (w, s, out, rows) for up to four matrices */
        size_t m = 0;
        const float *xq = q8_gemm_input(g, r, &m);
        int first = !(xq != NULL && xq == qx_src && m == qx_n);
        for (int k = 0; k < 4; ++k) {
            if (r->v[3 + 4 * k] != 0) {
                gemm_launch(g, r, dr, denv, 0, 3 + 4 * k, 5 + 4 * k, 6 + 4 * k, 1, 2, &bad,
                            first);
                first = 0;
            }
        }
        break;
    }
    case GP_GELU_MUL_ROWS: {
        int64_t n = hlit(r, 3, &bad) * hlit(r, 4, &bad);
        if (n % 4 == 0 && an_v_on() && al16(hi(r, g->henv, 0)) && al16(hi(r, g->henv, 1)) &&
            al16(hi(r, g->henv, 2))) {
            float *xs = NULL, *xsum = NULL;
            int8_t *q = qx_fuse(g, r, (const void *)(intptr_t)hi(r, g->henv, 2), (size_t)n, &xs,
                                &xsum);
            k_gelu_mul_rows_v<<<(unsigned)cdiv(n / 4, T), T, 0, s>>>(dr, denv, q, xs, xsum);
        } else {
            k_gelu_mul_rows<<<(unsigned)cdiv(n, T), T, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_ATTN_QC_MT:
        if (hlit(r, 9, &bad) > 512 || hlit(r, 9, &bad) % 16 != 0) {
            bad = 1;
        }
        if (hlit(r, 10, &bad) > MT_MAX && gg_tc && (g->tc || g->atc) &&
            (flash_qc_launch(r, dr, denv, &bad) == 0 ||
             flash_tc_launch(r, dr, denv, 0, &bad) == 0)) {
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
        if (hlit(r, 5, &bad) == 1) {
            /* KQ_Q4X experts: int8 x on the tensor cores (k_moe_gemm_q4x), as
             * the int8 x of the prompt of the CPU (the flag tc of the program
             * is about the float16 products of the other layout) */
            if (!(kt_on() && gu_rows % KT_BN == 0 && dn_rows % KT_BN == 0 &&
                  cols % 32 == 0 && inner % 32 == 0 && (size_t)t * cols <= g->kqx_n &&
                  (size_t)pairs * inner <= g->kqx_n)) {
                snprintf(gg_error, sizeof(gg_error),
                         "GP_MOE_GPU with KQ_Q4X needs the tensor cores (kt %d, rows %d %d, "
                         "x %zu %zu of %zu)", kt_on(), gu_rows, dn_rows, (size_t)(t * cols),
                         (size_t)(pairs * inner), g->kqx_n);
                return -1;
            }
            kt_attr();
            float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
            if (g->i8 == 2) {
                /* the int16 form of h and of the GELU (k_quant_x2) */
                size_t sm = KT_SMEM_X2(64, 2);
                kt_quant_x2(g, h, (size_t)(t * cols));
                k_moe_gemm_q4x<1><<<dim3((unsigned)(gu_rows / KT_BN), max_tiles), 256, sm, s>>>(
                    dr, denv, g->kqx, xs, xsum, 1, gu, gu_rows, cols, act, kt_lo(g));
                k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
                kt_quant_x2(g, act2, (size_t)(pairs * inner));
                k_moe_gemm_q4x<1><<<dim3((unsigned)(dn_rows / KT_BN), max_tiles), 256, sm, s>>>(
                    dr, denv, g->kqx, xs, xsum, 0, dn, dn_rows, inner, de, kt_lo(g));
            } else {
                size_t sm = KT_SMEM(64, 3);
                kt_quant(g, h, (size_t)(t * cols));
                k_moe_gemm_q4x<0><<<dim3((unsigned)(gu_rows / KT_BN), max_tiles), 256, sm, s>>>(
                    dr, denv, g->kqx, xs, xsum, 1, gu, gu_rows, cols, act, NULL);
                k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
                kt_quant(g, act2, (size_t)(pairs * inner));
                k_moe_gemm_q4x<0><<<dim3((unsigned)(dn_rows / KT_BN), max_tiles), 256, sm, s>>>(
                    dr, denv, g->kqx, xs, xsum, 0, dn, dn_rows, inner, de, NULL);
            }
        } else if (gg_tc && g->tc && cols % TK == 0 && inner % TK == 0 &&
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
    case GP_KQ_QUANT:
        /* The GPU products read the float rows (operand 3 of GP_KQ_LINEAR). */
        return 0;
    case GP_KQ_LINEAR:
        if (hlit(r, 8, &bad) > MT_MAX && kt_on() && g->tc &&
            kt_format((int)hlit(r, 5, &bad)) == KT_Q8R && hlit(r, 7, &bad) % KT_K == 0 &&
            (size_t)hlit(r, 8, &bad) * hlit(r, 7, &bad) <= g->kqx_n) {
            /* The operand x (3) is a device address in the record on the device;
             * the host copy of the record has it too. */
            int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad), rows = hlit(r, 6, &bad);
            kt_attr();
            kt_quant(g, (const float *)(intptr_t)hi(r, g->henv, 3), (size_t)(t * cols));
            float *xs = (float *)(g->kqx + g->kqx_n);
            k_kq_tc<KT_Q8R><<<dim3((unsigned)cdiv(t, 128), (unsigned)cdiv(rows, KT_BN)), 256,
                              KT_SMEM(128, 2), s>>>(dr, denv, g->kqx, xs, xs + g->kqx_n / 32);
        } else if (hlit(r, 8, &bad) > MT_MAX && hlit(r, 5, &bad) == KQ_BF12 && bf16_tc_on() &&
                   hlit(r, 7, &bad) % 32 == 0 &&
                   (bf12_fused_on() || (g->w16 != NULL && (size_t)hlit(r, 7, &bad) <= g->w16_n)) &&
                   (size_t)hlit(r, 8, &bad) * hlit(r, 7, &bad) * (bf16_tc_on() != 8 ? 2 : 1) <= g->xh_n) {
            /* BF12 rows on the tensor cores: chunks of bfloat16 rows (gg_gemm_bf12) */
            int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad), rows = hlit(r, 6, &bad);
            gg_gemm_bf12((const float *)(intptr_t)hi(r, g->henv, 3), (size_t)cols, NULL,
                         (const uint8_t *)(intptr_t)hi(r, g->henv, 4),
                         (float *)(intptr_t)hi(r, g->henv, 9), (size_t)rows, NULL, (int)t, (int)rows,
                         (int)cols, (uint16_t *)g->xh, g->w16, g->w16_n, s);
        } else if (hlit(r, 8, &bad) > MT_MAX && hlit(r, 5, &bad) == KQ_BF16 && bf16_tc_on() &&
                   hlit(r, 7, &bad) % 8 == 0 &&
                   (size_t)hlit(r, 8, &bad) * hlit(r, 7, &bad) * (bf16_tc_on() != 8 ? 2 : 1) <= g->xh_n) {
            /* bfloat16 rows on the tensor cores (k_gemm_bf16_tc), x to bfloat16 in
             * the scratch xh */
            int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad), rows = hlit(r, 6, &bad);
            gg_gemm_bf16((const float *)(intptr_t)hi(r, g->henv, 3), (size_t)cols, NULL,
                         (const uint16_t *)(intptr_t)hi(r, g->henv, 4),
                         (float *)(intptr_t)hi(r, g->henv, 9), (size_t)rows, NULL, (int)t, (int)rows,
                         (int)cols, (uint16_t *)g->xh, s);
        } else if (hlit(r, 8, &bad) > MT_MAX) {
            if (hlit(r, 7, &bad) % KG_K != 0) {
                bad = 1;
            }
            k_kq_gemm<<<dim3((unsigned)cdiv(hlit(r, 6, &bad), KG_B),
                             (unsigned)cdiv(hlit(r, 8, &bad), KG_B)), 256, 0, s>>>(dr, denv);
        } else if (kq_i8_type((int)hlit(r, 5, &bad)) && i8x_on() && hlit(r, 7, &bad) % 32 == 0 &&
                   (size_t)(hlit(r, 8, &bad) * hlit(r, 7, &bad)) <= g->kqx_n) {
            size_t n = (size_t)(hlit(r, 8, &bad) * hlit(r, 7, &bad));
            const float *x = (const float *)(intptr_t)hi(r, g->henv, 3);
            if (x != qx_src || n != qx_n) {
                kt_quant(g, x, n);
            }
            dim3 gr((unsigned)cdiv(hlit(r, 6, &bad), KQ_RPB));
            const float *xs = (const float *)(g->kqx + g->kqx_n);
            /* A step keeps the registers of one token (NT 1). For a group,
             * one kernel with the counts 1 to KQ_NT (NT 0) was faster on the
             * E4B than a kernel for each count: a down matrix of 3 tokens took
             * 0.066 ms, not 0.103 ms. The terms of a token are the same. */
            int q40 = hlit(r, 5, &bad) == KQ_Q4_0;
            if (hlit(r, 8, &bad) <= 1) {
                if (q40) {
                    k_kq_linear_i8<1, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                } else {
                    k_kq_linear_i8<1><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                }
            } else if (q40) {
                k_kq_linear_i8<0, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else {
                k_kq_linear_i8<0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            }
        } else if (bt_on() && hlit(r, 8, &bad) <= 8 &&
                   bt_shape((int)hlit(r, 5, &bad), (int)hlit(r, 6, &bad), (int)hlit(r, 7, &bad))) {
            /* KQ_BF12 rows of a step or a small group on the tensor cores */
            int rows = (int)hlit(r, 6, &bad);
            bt_launch(dr, denv, 0, bt_cls(rows, (int)hlit(r, 7, &bad)), cdiv(rows, 16), g->btp, g->btc, s);
        } else if (hlit(r, 5, &bad) == KQ_BF12 && hlit(r, 8, &bad) <= 8 && bf16_nt_on() &&
                   !(kq_split_on() && hlit(r, 7, &bad) >= 8192) && hlit(r, 7, &bad) % 32 == 0) {
            /* KQ_BF12 rows of a step or a small group (k_kq_linear_bf12); long
             * rows keep the split kernels (k_kq_linear_split, _split_bf16) */
            int64_t t = hlit(r, 8, &bad), rows = hlit(r, 6, &bad);
            unsigned gb = (unsigned)cdiv(rows, KQ_RPB);
            if (t <= 1) {
                k_kq_linear_bf12<1><<<gb, 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else if (t <= 2) {
                k_kq_linear_bf12<2><<<gb, 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else if (t <= 4) {
                k_kq_linear_bf12<4><<<gb, 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else {
                k_kq_linear_bf12<8><<<gb, 32 * KQ_RPB, 0, s>>>(dr, denv);
            }
        } else if (kq_bf_type((int)hlit(r, 5, &bad)) && hlit(r, 8, &bad) >= 2 && hlit(r, 8, &bad) <= 8 &&
                   bf16_nt_on() && hlit(r, 7, &bad) % 16 == 0) {
            /* bfloat16 rows of a small group: the weights once for all the
             * tokens (k_kq_linear_bf16), as the split of a step */
            int64_t t = hlit(r, 8, &bad), rows = hlit(r, 6, &bad);
            int split = kq_split_on() && hlit(r, 7, &bad) >= 8192;
            if (split && t <= 4) {
                k_kq_linear_split_bf16<4><<<(unsigned)rows, 32 * KQ_SPLIT, 0, s>>>(dr, denv);
            } else if (split) {
                k_kq_linear_split_bf16<8><<<(unsigned)rows, 32 * KQ_SPLIT, 0, s>>>(dr, denv);
            } else if (t <= 2) {
                k_kq_linear_bf16<2><<<(unsigned)cdiv(rows, KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else if (t <= 4) {
                k_kq_linear_bf16<4><<<(unsigned)cdiv(rows, KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else {
                k_kq_linear_bf16<8><<<(unsigned)cdiv(rows, KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv);
            }
        } else if (kq_split_on() && hlit(r, 8, &bad) <= MT_MAX && hlit(r, 7, &bad) >= 8192) {
            k_kq_linear_split<<<(unsigned)hlit(r, 6, &bad), 32 * KQ_SPLIT, 0, s>>>(dr, denv);
        } else {
            k_kq_linear<<<(unsigned)cdiv(hlit(r, 6, &bad), KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv);
        }
        break;
    case GP_KQ_MULTI: {
        int64_t rows = 0, q8r = 0, t = hlit(r, 2, &bad), cols = hlit(r, 1, &bad);
        int q40 = 1;        /* all the matrices are Q4_0 (the kernels with FT) */
        for (int m = 0; m < hlit(r, 3, &bad); ++m) {
            rows += hlit(r, 6 + 4 * m, &bad);
            q8r |= kq_i8_type((int)hlit(r, 5 + 4 * m, &bad));
            q40 &= hlit(r, 5 + 4 * m, &bad) == KQ_Q4_0;
        }
        if (q8r && i8x_on() && cols % 32 == 0 && (size_t)(t * cols) <= g->kqx_n) {
            const float *x = (const float *)(intptr_t)hi(r, g->henv, 0);
            if (x != qx_src || (size_t)(t * cols) != qx_n) {
                kt_quant(g, x, (size_t)(t * cols));
            }
            dim3 gr((unsigned)cdiv(rows, KQ_RPB));
            const float *xs = (const float *)(g->kqx + g->kqx_n);
            /* the gate and the up matrix, then GELU_MUL_ROWS of their
             * outputs: one kernel (k_kq_glu_i8) */
            const gp_rec *nx = r + 1;
            if (glu_on() && hlit(r, 3, &bad) == 2 && nx < g->hcode + g->n_code &&
                nx->op == GP_GELU_MUL_ROWS && hlit(r, 6, &bad) == hlit(r, 10, &bad) &&
                kq_i8_type((int)hlit(r, 5, &bad)) && kq_i8_type((int)hlit(r, 9, &bad)) &&
                hi(nx, g->henv, 0) == hi(r, g->henv, 7) && hi(nx, g->henv, 1) == hi(r, g->henv, 11) &&
                hlit(nx, 3, &bad) == t && hlit(nx, 4, &bad) == hlit(r, 6, &bad)) {
                dim3 gg1((unsigned)cdiv(hlit(r, 6, &bad), KQ_RPB));
                float *out = (float *)(intptr_t)hi(nx, g->henv, 2);
                if (t <= 1 && q40) {
                    k_kq_glu_i8<1, KQ_Q4_0><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                } else if (t <= 1) {
                    k_kq_glu_i8<1><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                } else if (q40) {
                    k_kq_glu_i8<0, KQ_Q4_0><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                } else {
                    k_kq_glu_i8<0><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                }
                gg_skip = nx;
                break;
            }
            if (t <= 1 && q40) {
                k_kq_multi_i8<1, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else if (t <= 1) {
                k_kq_multi_i8<1><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else if (q40) {
                k_kq_multi_i8<0, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else {
                k_kq_multi_i8<0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            }
        } else {
            int anybf = 0, rest = 0;
            int64_t btb[3] = {0, 0, 0};     /* the blocks of k_kq_bf12_tc of each class */
            for (int m = 0; m < hlit(r, 3, &bad); ++m) {
                int ty = (int)hlit(r, 5 + 4 * m, &bad), rm = (int)hlit(r, 6 + 4 * m, &bad);
                if (bt_on() && t <= 8 && bt_shape(ty, rm, (int)cols)) {
                    btb[bt_cls(rm, (int)cols)] += cdiv(rm, 16);
                } else {
                    rest = 1;
                    anybf |= kq_bf_type(ty);
                }
            }
            int skip = btb[0] + btb[1] + btb[2] > 0;
            /* the KQ_BF12 matrices of bt_shape on the tensor cores */
            for (int k = 0; k < 3; ++k) {
                if (btb[k] > 0) {
                    bt_launch(dr, denv, 1, k, btb[k], g->btp, g->btc, s);
                }
            }
            if (!rest) {
                /* all of them there */
            } else if (anybf && t >= 2 && t <= 8 && bf16_nt_on() && cols % 16 == 0) {
                /* bfloat16 rows of a small group (k_kq_multi_bf16) */
                unsigned gm = (unsigned)cdiv(rows, KQ_RPB);
                if (t <= 2) {
                    k_kq_multi_bf16<2><<<gm, 32 * KQ_RPB, 0, s>>>(dr, denv, skip);
                } else if (t <= 4) {
                    k_kq_multi_bf16<4><<<gm, 32 * KQ_RPB, 0, s>>>(dr, denv, skip);
                } else {
                    k_kq_multi_bf16<8><<<gm, 32 * KQ_RPB, 0, s>>>(dr, denv, skip);
                }
            } else {
                k_kq_multi<<<(unsigned)cdiv(rows, KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv, skip);
            }
        }
        break;
    }
    case GP_ADD_RMS:
        k_add_rms<<<(unsigned)hlit(r, 4, &bad), T, 0, s>>>(dr, denv);
        break;
    case GP_KQ_GROUP_MOE: {
        int64_t t = hlit(r, 3, &bad), k = hlit(r, 4, &bad), E = hlit(r, 5, &bad);
        int64_t hidden = hlit(r, 6, &bad), inner = hlit(r, 7, &bad);
        unsigned tiles = (unsigned)(cdiv(t * k + t, KG_B) + E + 1);
        if (hidden % KG_B != 0 || inner % KG_B != 0) {
            bad = 1;
        }
        int fg = kt_format((int)hlit(r, 11, &bad)), fd = kt_format((int)hlit(r, 12, &bad));
        int64_t P = t * k + t;
        if (qmoe_sort_par_on() && (E + 2) * 8 <= 48 * 1024) {
            k_qmoe_sort_par<<<1, 1024, (size_t)(E + 2) * 8, s>>>(dr, denv);
        } else {
            k_qmoe_sort<<<1, 32, 0, s>>>(dr, denv);
        }
        /* the shared expert in bfloat16 (NP_GEMMA_DENSE=bf16): the routed
         * experts on the int8 tensor cores, the shared expert in gg_gemm_bf16
         * (its pairs are the rows start[E] .. start[E] + t of act, act2, de) */
        int stype = (int)hlit(r, 16, &bad);
        int sbf = (stype == KQ_BF16 ||
                   (stype == KQ_BF12 && hidden % 32 == 0 && inner % 32 == 0 &&
                    (bf12_fused_on() || (g->w16 != NULL &&
                                         (size_t)(hidden > inner ? hidden : inner) <= g->w16_n)))) &&
                  bf16_tc_on() && hidden % 16 == 0 && inner % 16 == 0 &&
                  2 * (size_t)(t * (hidden > inner ? hidden : inner)) <= g->xh_n;
        const int has_shared = hi(r, g->henv, 13) != 0;
        if (kt_on() && !moe_f32_on() && g->tc && fg >= 0 && fd >= 0 &&
            (kt_format((int)hlit(r, 16, &bad)) == KT_Q8R || sbf) &&
            hidden % KT_K == 0 && inner % KT_K == 0 && (size_t)(t * hidden) <= g->kqx_n &&
            (size_t)(P * inner) <= g->kqx_n) {
            const int *sstart = (const int *)(intptr_t)hi(r, g->henv, 18) + 8 + (E + 2) + E;
            float *act = (float *)(intptr_t)hi(r, g->henv, 19);
            kt_attr();
            float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
            size_t sm = KT_SMEM(64, 3);
            kt_quant(g, (const float *)(intptr_t)hi(r, g->henv, 0), (size_t)(t * hidden));
            dim3 ggu((unsigned)(2 * inner / KT_BN), tiles), gdn((unsigned)(hidden / KT_BN), tiles);
            switch (fg) {
            case KT_Q8R: k_qmoe_gu_tc<KT_Q8R><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q80: k_qmoe_gu_tc<KT_Q80><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q51: k_qmoe_gu_tc<KT_Q51><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_NV4: k_qmoe_gu_tc<KT_NV4><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_NVX: k_qmoe_gu_tc<KT_NVX><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q6K: k_qmoe_gu_tc<KT_Q6K><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            default: k_qmoe_gu_tc<KT_Q4K><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            }
            if (sbf && has_shared) {
                /* the gate and the up rows of the shared expert for the t rows of h */
                const float *h = (const float *)(intptr_t)hi(r, g->henv, 0);
                for (int u = 0; u < 2; ++u) {
                    const void *sw = (const void *)(intptr_t)hi(r, g->henv, 13 + u);
                    if (stype == KQ_BF12) {
                        gg_gemm_bf12(h, (size_t)hidden, NULL, (const uint8_t *)sw, act + u * inner,
                                     (size_t)(2 * inner), sstart, (int)t, (int)inner, (int)hidden,
                                     (uint16_t *)g->xh, g->w16, g->w16_n, s);
                    } else {
                        gg_gemm_bf16(h, (size_t)hidden, NULL, (const uint16_t *)sw, act + u * inner,
                                     (size_t)(2 * inner), sstart, (int)t, (int)inner, (int)hidden,
                                     (uint16_t *)g->xh, s);
                    }
                }
            }
            k_qmoe_act<<<(unsigned)cdiv(P * inner, T), T, 0, s>>>(dr, denv, gg_moe_rot);
            kt_quant(g, (const float *)(intptr_t)hi(r, g->henv, 20), (size_t)(P * inner));
            switch (fd) {
            case KT_Q8R: k_qmoe_dn_tc<KT_Q8R><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q80: k_qmoe_dn_tc<KT_Q80><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q51: k_qmoe_dn_tc<KT_Q51><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_NV4: k_qmoe_dn_tc<KT_NV4><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_NVX: k_qmoe_dn_tc<KT_NVX><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q6K: k_qmoe_dn_tc<KT_Q6K><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            default: k_qmoe_dn_tc<KT_Q4K><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            }
            if (sbf && has_shared) {
                /* the down rows of the shared expert: rows start[E] .. of act2 into de */
                if (stype == KQ_BF12) {
                    gg_gemm_bf12((const float *)(intptr_t)hi(r, g->henv, 20), (size_t)inner, sstart,
                                 (const uint8_t *)(intptr_t)hi(r, g->henv, 15),
                                 (float *)(intptr_t)hi(r, g->henv, 21), (size_t)hidden, sstart, (int)t,
                                 (int)hidden, (int)inner, (uint16_t *)g->xh, g->w16, g->w16_n, s);
                } else {
                    gg_gemm_bf16((const float *)(intptr_t)hi(r, g->henv, 20), (size_t)inner, sstart,
                                 (const uint16_t *)(intptr_t)hi(r, g->henv, 15),
                                 (float *)(intptr_t)hi(r, g->henv, 21), (size_t)hidden, sstart, (int)t,
                                 (int)hidden, (int)inner, (uint16_t *)g->xh, s);
                }
            }
        } else {
            k_qmoe_gu<<<dim3((unsigned)(2 * inner / KG_B), tiles), 256, 0, s>>>(dr, denv);
            k_qmoe_act<<<(unsigned)cdiv((t * k + t) * inner, T), T, 0, s>>>(dr, denv, gg_moe_rot);
            k_qmoe_dn<<<dim3((unsigned)(hidden / KG_B), tiles), 256, 0, s>>>(dr, denv);
        }
        k_qmoe_sum<<<dim3((unsigned)cdiv(hidden, T), (unsigned)t), T, 0, s>>>(dr, denv);
        break;
    }
    case GP_QSA_SELECT: {
        int64_t t = hlit(r, 9, &bad), d = hlit(r, 11, &bad), ratio = hlit(r, 12, &bad);
        if (d > 256 || hlit(r, 10, &bad) * d > 8 * 256 || ratio < 1) {
            bad = 1;
        }
        k_qsa_copy<<<(unsigned)cdiv(t * d, T), T, 0, s>>>(dr, denv);
        k_qsa_blocks<<<(unsigned)(t / (ratio > 0 ? ratio : 1) + 1), 128, 0, s>>>(dr, denv);
        int stc = qsa_score_tc_on() && hlit(r, 10, &bad) == 4 && d == 128;
        for (int64_t j0 = 0; j0 < t; j0 += QSA_ROWS) {
            unsigned rows = (unsigned)(t - j0 < QSA_ROWS ? t - j0 : QSA_ROWS);
            if (stc) {
                /* few queries (a step, an MTP verify group): more blocks on
                 * the keys, so the scan of all the blocks takes all the SMs */
                unsigned gy = (unsigned)cdiv(rows, 16);
                unsigned gx = gy >= 10 ? QS_GX : (gy >= 2 ? 64 : 128);
                k_qsa_qprep<<<rows, 128, 0, s>>>(dr, denv, (int)j0);
                k_qsa_score_tc<<<dim3(gx, gy), 128, 0, s>>>(dr, denv, (int)j0, (int)rows);
                k_qsa_query<1><<<rows, QSA_T, 0, s>>>(dr, denv, (int)j0);
            } else {
                k_qsa_query<0><<<rows, QSA_T, 0, s>>>(dr, denv, (int)j0);
            }
        }
        break;
    }
    case GP_HC_NORM: {
        int64_t rows = hlit(r, 3, &bad) * hlit(r, 4, &bad), hid = hlit(r, 5, &bad);
        float *qs = NULL, *qsum = NULL;
        int8_t *q = hc_fuse_on() && hid % 32 == 0 && al16(hi(r, g->henv, 2))
                        ? qx_fuse(g, r, (const void *)(intptr_t)hi(r, g->henv, 2),
                                  (size_t)(rows * hid), &qs, &qsum) : NULL;
        k_hc_norm<<<(unsigned)rows, 256, 0, s>>>(dr, denv, q, qs, qsum);
        break;
    }
    case GP_HC_ACT: {
        int64_t n = hlit(r, 2, &bad);
        float *qs = NULL, *qsum = NULL;
        int8_t *q = hc_fuse_on() && n % 32 == 0 && al16(hi(r, g->henv, 0)) && al16(hi(r, g->henv, 1))
                        ? qx_fuse(g, r, (const void *)(intptr_t)hi(r, g->henv, 1), (size_t)n, &qs,
                                  &qsum) : NULL;
        if (q != NULL) {
            k_hc_act_q<<<(unsigned)cdiv(n / 4, T), T, 0, s>>>(dr, denv, q, qs, qsum);
        } else {
            k_hc_act<<<(unsigned)cdiv(n, T), T, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_HC_MIX: {
        int64_t n = hlit(r, 3, &bad) * hlit(r, 5, &bad);
        float *qs = NULL, *qsum = NULL;
        int8_t *q = hc_fuse_on() && n % 32 == 0 && hlit(r, 5, &bad) % 4 == 0 && al16(hi(r, g->henv, 2))
                        ? qx_fuse(g, r, (const void *)(intptr_t)hi(r, g->henv, 2), (size_t)n, &qs,
                                  &qsum) : NULL;
        if (q != NULL) {
            k_hc_mix_q<<<(unsigned)cdiv(n / 4, T), T, 0, s>>>(dr, denv, q, qs, qsum);
        } else {
            k_hc_mix<<<(unsigned)cdiv(n, T), T, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_HC_ADD: {
        const gp_rec *nx = r + 1;
        if (hc_fuse_on() && nx < g->hcode + g->n_code && nx->op == GP_HC_NORM &&
            hi(nx, g->henv, 0) == hi(r, g->henv, 0) && hlit(nx, 3, &bad) == hlit(r, 3, &bad) &&
            hlit(nx, 4, &bad) == hlit(r, 4, &bad) && hlit(nx, 5, &bad) == hlit(r, 5, &bad)) {
            /* the add and the norm of the next record (k_hc_add_norm) */
            int64_t rows = hlit(r, 3, &bad) * hlit(r, 4, &bad), hid = hlit(r, 5, &bad);
            float *qs = NULL, *qsum = NULL;
            int8_t *q = hid % 32 == 0 && al16(hi(nx, g->henv, 2))
                            ? qx_fuse(g, nx, (const void *)(intptr_t)hi(nx, g->henv, 2),
                                      (size_t)(rows * hid), &qs, &qsum) : NULL;
            k_hc_add_norm<<<(unsigned)rows, 256, 0, s>>>(dr, denv, dr + 1, q, qs, qsum);
            gg_skip = nx;
            break;
        }
        k_hc_add<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad) * hlit(r, 5, &bad), T), T,
                   0, s>>>(dr, denv);
        break;
    }
    case GP_PLE_GATE:
        k_ple_gate<<<(unsigned)(hlit(r, 4, &bad) * hlit(r, 5, &bad)), 256, 0, s>>>(dr, denv);
        break;
    case GP_PLE_CONV:
        k_ple_conv<<<(unsigned)cdiv(hlit(r, 6, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_HC_CAT:
        k_hc_cat<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad) * 2 * hlit(r, 5, &bad), T),
                   T, 0, s>>>(dr, denv);
        break;
    case GP_SIGMUL:
        k_sigmul<<<(unsigned)cdiv(hlit(r, 3, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_ROUTER_TOPK:
        if (hlit(r, 2, &bad) > 1024 || hlit(r, 3, &bad) > 256) {
            bad = 1;
        }
        {
            const gp_rec *nx = r + 1;
            if (topk_split_on() && hlit(r, 1, &bad) == 1 && nx < g->hcode + g->n_code &&
                nx->op == GP_HOT_SPLIT && hi(nx, g->henv, 0) == hi(r, g->henv, 5) &&
                hi(nx, g->henv, 1) == hi(r, g->henv, 4)) {
                /* the hot split of the token in the same kernel */
                k_router_topk<<<1, 256, 0, s>>>(dr, denv, dr + 1);
                gg_skip = nx;
                break;
            }
        }
        k_router_topk<<<(unsigned)hlit(r, 1, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_ATTN_PREP: {
        int64_t hd = hlit(r, 14, &bad);
        if (hd > 1024 || hd % 32 != 0) {
            bad = 1;
        }
        k_attn_prep<<<(unsigned)(hlit(r, 11, &bad) * (hlit(r, 12, &bad) + hlit(r, 13, &bad))),
                      (unsigned)hd, 0, s>>>(dr, denv);
        break;
    }
    case GP_GDN: {
        /* k_dim = v_dim = 128 */
        if (hlit(r, 16, &bad) != 128 || hlit(r, 17, &bad) != 128 ||
            hlit(r, 3, &bad) > 9) {
            bad = 1;
        }
        int64_t cd = 2 * hlit(r, 14, &bad) * 128 + hlit(r, 15, &bad) * 128;
        int64_t t = hlit(r, 13, &bad), kh = hlit(r, 14, &bad), vh = hlit(r, 15, &bad);
        int gs = gdn_par_on();
        int nolog = r->tag[19] == GP_T_NONE || (r->tag[19] == GP_T_INT && r->v[19] == 0);
        if (nolog && t >= GDN_PAR_MIN && (gs == 4 || gs == 8)) {
            /* a large group with no log: the parallel forms */
            k_gdn_conv_par<<<dim3((unsigned)cdiv(cd, T), (unsigned)t), T, 0, s>>>(dr, denv);
            k_gdn_prep<128><<<dim3((unsigned)t, (unsigned)(kh + 2)), 128, 0, s>>>(dr, denv);
            if (gs == 4) {
                k_gdn_scan<128, 4><<<dim3((unsigned)vh, 4), 128, 0, s>>>(dr, denv);
            } else {
                k_gdn_scan<128, 8><<<dim3((unsigned)vh, 8), 128, 0, s>>>(dr, denv);
            }
            k_gdn_out<128><<<dim3((unsigned)t, (unsigned)vh), 128, 0, s>>>(dr, denv);
            break;
        }
        k_gdn_conv<<<(unsigned)cdiv(cd, T), T, 0, s>>>(dr, denv);
        k_gdn_heads<128><<<(unsigned)vh, 128, 0, s>>>(dr, denv);
        break;
    }
    case GP_FFN_OUT:
        if (hlit(r, 8, &bad) > FFN_COLS) {
            bad = 1;
        }
        k_ffn_out<<<(unsigned)hlit(r, 7, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_SIGNAL:
        k_signal<<<1, 256, 0, s>>>(dr, denv);
        break;
    case GP_AWAIT:
        k_await<<<1, 1024, 0, s>>>(dr, denv);
        break;
    case GP_D2H:
        CK(cudaMemcpyAsync((void *)(intptr_t)hlit(r, 1, &bad),
                           (const void *)(intptr_t)hlit(r, 0, &bad), (size_t)hlit(r, 2, &bad),
                           cudaMemcpyDeviceToHost, s));
        break;
    case GP_H2D:
        CK(cudaMemcpyAsync((void *)(intptr_t)hlit(r, 1, &bad),
                           (const void *)(intptr_t)hlit(r, 0, &bad), (size_t)hlit(r, 2, &bad),
                           cudaMemcpyHostToDevice, s));
        break;
    case GP_CPU_TASK:
        /* no kernel: the runner runs it (gg_exec) */
        break;
    case GP_KQ_HOT_MOE: {
        unsigned t = (unsigned)hlit(r, 16, &bad);
        unsigned pairs = (unsigned)hlit(r, 11, &bad) * t + t;
        int64_t inner = hlit(r, 12, &bad), hidden = hlit(r, 13, &bad);
        unsigned tk = (unsigned)hlit(r, 11, &bad) * t;
        if (kqh_nvx_on() && hlit(r, 14, &bad) == KQ_NVX && hlit(r, 15, &bad) == KQ_NVX &&
            inner % 16 == 0 && hidden % 16 == 0) {
            /* the KQ_NVX experts with a block for each group (k_kqh_nvx),
             * the shared expert (pairs tk on) with k_kqh_gu and k_kqh_dn */
            k_kqh_nvx<8><<<dim3((unsigned)(inner / 16), tk, 2), 32 * 8, 0, s>>>(dr, denv, 0);
            k_kqh_gu<<<dim3((unsigned)cdiv(inner, ROWS_PER_BLOCK), t, 2), W, 0, s>>>(dr, denv,
                                                                                 (int)tk);
            k_kqh_act<<<pairs, T, 0, s>>>(dr, denv, gg_moe_rot);
            k_kqh_nvx<4><<<dim3((unsigned)(hidden / 16), tk, 1), 32 * 4, 0, s>>>(dr, denv, 1);
            k_kqh_dn<<<dim3((unsigned)cdiv(hidden, ROWS_PER_BLOCK), t), W, 0, s>>>(dr, denv,
                                                                               (int)tk);
            k_kqh_sum<<<dim3((unsigned)cdiv(hidden, T), t), T, 0, s>>>(dr, denv);
            break;
        }
        k_kqh_gu<<<dim3((unsigned)cdiv(inner, ROWS_PER_BLOCK), pairs, 2), W, 0, s>>>(dr, denv);
        k_kqh_act<<<pairs, T, 0, s>>>(dr, denv, gg_moe_rot);
        k_kqh_dn<<<dim3((unsigned)cdiv(hidden, ROWS_PER_BLOCK), pairs), W, 0, s>>>(dr, denv);
        k_kqh_sum<<<dim3((unsigned)cdiv(hidden, T), t), T, 0, s>>>(dr, denv);
        break;
    }
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
        /* The first event of the measure goes before the wait of the host:
         * the copies are from pageable memory, so the worker records the job
         * only near the end of its copies, and the host waits here too. */
        int k = gg_ms_nf + gg_ms_nj;
        int ms = gg_ms_on && k < GG_MS_MAX;
        if (ms) {
            CK(cudaEventRecord(gg_ms_ev[2 * k], gg_stream));
        }
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
        if (ms) {
            CK(cudaStreamWaitEvent(gg_stream, gg_ready[f], 0));
            CK(cudaEventRecord(gg_ms_ev[2 * k + 1], gg_stream));
            gg_ms_kind[k] = 0;
            ++gg_ms_nf;
            return 0;
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
    case GP_CPU_JOIN: {
        int k = gg_ms_nf + gg_ms_nj;
        int ms = gg_ms_on && k < GG_MS_MAX;
        if (ms) {
            CK(cudaEventRecord(gg_ms_ev[2 * k], gg_stream));
        }
        CK(cudaEventSynchronize(g->ev[r->v[1]]));
        double t0 = gg_clock();
        if (gg_cpu_run == NULL || gg_cpu_run((const int64_t *)(intptr_t)r->v[0], -1) != 0) {
            snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_JOIN failed");
            return -1;
        }
        if (ms) {
            /* The event after the CPU program: the stream waited from the end
             * of its work before (the event above) to here. */
            gg_ms_cpu[gg_ms_nj] = gg_clock() - t0;
            CK(cudaEventRecord(gg_ms_ev[2 * k + 1], gg_stream));
            gg_ms_kind[k] = 1;
            ++gg_ms_nj;
        }
        return 0;
    }
    case GP_TO_DEV:
        CK(cudaMemcpyAsync((void *)(intptr_t)r->v[1], (const void *)(intptr_t)r->v[0],
                           (size_t)r->v[2], cudaMemcpyHostToDevice, gg_stream));
        return 0;
    case GP_CPU_START:
        CK(cudaEventSynchronize(g->ev[r->v[1]]));
        pthread_mutex_lock(&gg_hmu);
        if (!gg_hon) {
            pthread_t th;
            if (pthread_create(&th, NULL, gg_helper, NULL) != 0) {
                pthread_mutex_unlock(&gg_hmu);
                snprintf(gg_error, sizeof(gg_error), "no helper thread for GP_CPU_START");
                return -1;
            }
            pthread_detach(th);
            gg_hon = 1;
        }
        while (gg_hbusy) {
            pthread_cond_wait(&gg_hcv, &gg_hmu);
        }
        gg_hbusy = 1;
        gg_hjob = (const int64_t *)(intptr_t)r->v[0];
        pthread_cond_broadcast(&gg_hcv);
        pthread_mutex_unlock(&gg_hmu);
        return 0;
    case GP_CPU_WAIT: {
        /* The measure: as for GP_CPU_JOIN. */
        int k = gg_ms_nf + gg_ms_nj;
        int ms = gg_ms_on && k < GG_MS_MAX;
        if (ms) {
            CK(cudaEventRecord(gg_ms_ev[2 * k], gg_stream));
        }
        pthread_mutex_lock(&gg_hmu);
        while (gg_hbusy) {
            pthread_cond_wait(&gg_hcv, &gg_hmu);
        }
        int bad = gg_hfail;
        gg_hfail = 0;
        double dt = gg_hlast;
        pthread_mutex_unlock(&gg_hmu);
        if (ms) {
            gg_ms_cpu[gg_ms_nj] = dt;
            CK(cudaEventRecord(gg_ms_ev[2 * k + 1], gg_stream));
            gg_ms_kind[k] = 1;
            ++gg_ms_nj;
        }
        if (bad) {
            snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_START failed");
            return -1;
        }
        return 0;
    }
    default:
        return -1;
    }
}

/* Launch the kernels of a segment one at a time. With tasks, a GP_CPU_TASK
 * runs here, in its place (the launches without a graph); else the runner
 * runs it after the launches (gg_run_tasks). */
static int gg_launch_seg(gg_prog *g, const gg_seg *sg, int tasks)
{
    for (int pc = sg->start; pc < sg->end; ++pc) {
        const gp_rec *r = g->hcode + pc;
        gg_at(g, pc, r->op, sg->start, sg->end);
        if (r->op == GP_CPU_TASK && tasks) {
            if (gg_cpu_task(r, g->henv) != 0) {
                return -1;
            }
            continue;
        }
        if (gg_launch(g, r, g->dcode + pc, g->denv) != 0) {
            return -1;
        }
        GG_SYNC_AT(pc, r->op);
    }
    return 0;
}

/* Run the GP_CPU_TASK records from *from to to, in order (the graphs of
 * these records are on the stream). */
static int gg_run_tasks(gg_prog *g, int *from, int to)
{
    for (int pc = *from; pc < to; ++pc) {
        const gp_rec *r = g->hcode + pc;
        if (r->op == GP_CPU_TASK && gg_cpu_task(r, g->henv) != 0) {
            return -1;
        }
    }
    *from = to;
    return 0;
}

/* Run the segments and the boundary records in order. With the graph, the
 * first run records the graph of each segment. */
#define GG_SEG_FIRST 32
#define GG_SEG_MAX 160

/* NP_GEMMA_GPU_PDL=0 keeps the ordinary edges, for a test. A GPU before
 * sm_90 has no programmatic edges. */
static int gg_pdl_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("NP_GEMMA_GPU_PDL");
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        on = !(v && v[0] == '0') && major >= 9;
    }
    return on;
}

/* Change each edge from a kernel to a kernel into a programmatic edge. The
 * kernels start with PDL_START, so each waits for its data. */
static cudaError_t gg_pdl_edges(cudaGraph_t graph)
{
    size_t n = 0;
    cudaError_t err = cudaGraphGetEdges(graph, NULL, NULL, NULL, &n);
    if (err != cudaSuccess || n == 0) {
        return err;
    }
    cudaGraphNode_t *from = (cudaGraphNode_t *)malloc(n * sizeof(cudaGraphNode_t));
    cudaGraphNode_t *to = (cudaGraphNode_t *)malloc(n * sizeof(cudaGraphNode_t));
    cudaGraphEdgeData *ed = (cudaGraphEdgeData *)malloc(n * sizeof(cudaGraphEdgeData));
    err = cudaGraphGetEdges(graph, from, to, ed, &n);
    for (size_t i = 0; i < n && err == cudaSuccess; ++i) {
        cudaGraphNodeType ta, tb;
        if (cudaGraphNodeGetType(from[i], &ta) != cudaSuccess ||
            cudaGraphNodeGetType(to[i], &tb) != cudaSuccess ||
            ta != cudaGraphNodeTypeKernel || tb != cudaGraphNodeTypeKernel ||
            ed[i].type != cudaGraphDependencyTypeDefault) {
            continue;
        }
        err = cudaGraphRemoveDependencies(graph, &from[i], &to[i], &ed[i], 1);
        if (err == cudaSuccess) {
            cudaGraphEdgeData e;
            memset(&e, 0, sizeof(e));
            e.from_port = cudaGraphKernelNodePortProgrammatic;
            e.type = cudaGraphDependencyTypeProgrammatic;
            err = cudaGraphAddDependencies(graph, &from[i], &to[i], &e, 1);
        }
    }
    free(from);
    free(to);
    free(ed);
    return err;
}

/* Record the graph of a segment (the first run of a program). */
static int gg_capture(gg_prog *g, gg_seg *sg)
{
    CK(cudaStreamBeginCapture(gg_stream, cudaStreamCaptureModeThreadLocal));
    int rc = gg_launch_seg(g, sg, 0);
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(gg_stream, &graph);
    if (rc != 0) {
        return rc;
    }
    CK(err);
    if (gg_pdl_on()) {
        CK(gg_pdl_edges(graph));
    }
    CK(cudaGraphInstantiate(&sg->exec, graph, 0));
    CK(cudaGraphDestroy(graph));
    return 0;
}

/* The graphs of all the segments up to a boundary record go on the stream
 * first; then the runner runs their GP_CPU_TASK records, while the GPU runs
 * them. Thus the host launches nothing between the tasks.
 *
 * The first run records every graph before it launches one. A graph of the
 * GPU can wait for a task (GP_AWAIT), and the host runs the tasks only after
 * the launches; the first use of a kernel can make the driver wait for the
 * GPU (the lazy load of a module), so a capture after a launch could wait
 * for a task that waits for the capture. */
static int gg_exec(gg_prog *g)
{
    int pc = 0, k = 0, task = 0;
    int graph = g->use_graph && !gg_sync_check();
    if (graph) {
        for (int s = 0; s < g->n_seg; ++s) {
            if (g->seg[s].exec == NULL && gg_capture(g, g->seg + s) != 0) {
                return -1;
            }
        }
    }
    while (pc < g->n_code) {
        const gp_rec *r = g->hcode + pc;
        if (is_boundary(r->op)) {
            if (graph && gg_run_tasks(g, &task, pc) != 0) {
                return -1;
            }
            gg_at(g, pc, r->op, pc, pc + 1);
            if (gg_boundary(g, r) != 0) {
                return -1;
            }
            GG_SYNC_AT(pc, r->op);
            ++pc;
            task = pc;
            continue;
        }
        gg_seg *sg = g->seg + k++;
        if (!graph) {
            if (gg_launch_seg(g, sg, 1) != 0) {
                return -1;
            }
        } else {
            gg_at(g, sg->end - 1, g->hcode[sg->end - 1].op, sg->start, sg->end);
            CK(cudaGraphLaunch(sg->exec, gg_stream));
        }
        pc = sg->end;
    }
    if (graph && gg_run_tasks(g, &task, g->n_code) != 0) {
        return -1;
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

/* ---- The media encoders ---------------------------------------------------
 *
 * The kernels of the records GP_ENC_* (the encoders of images and audio,
 * np_gemma/gemma4_encoders.py, as programs). The arrays are row major. */

#define EG_T 64
#define EG_K 16

/* y (n, m) = clamp(clamp(x, imin, imax) W^T + b, omin, omax). x (n, k) is
 * float32 and W (m, k) bfloat16 (wbf 1) or float32; b may be NULL. Any n, m,
 * and k. A block of 256 threads makes a tile of 64 x 64 values of y. */
__global__ void k_enc_gemm(const float *x, const void *w, int wbf, const float *b, float *y,
                           int n, int m, int k, float imin, float imax, float omin, float omax)
{
    PDL_START();
    __shared__ float xs[EG_K][EG_T + 1];
    __shared__ float ws[EG_K][EG_T + 1];
    int r0 = blockIdx.y * EG_T, c0 = blockIdx.x * EG_T;
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int k0 = 0; k0 < k; k0 += EG_K) {
        for (int q = threadIdx.x; q < EG_T * EG_K; q += blockDim.x) {
            int rr = q / EG_K, kk = q % EG_K, kc = k0 + kk;
            int r = r0 + rr, c = c0 + rr;
            float xv = 0.f, wv = 0.f;
            if (r < n && kc < k) {
                xv = fminf(fmaxf(x[(size_t)r * k + kc], imin), imax);
            }
            if (c < m && kc < k) {
                size_t o = (size_t)c * k + kc;
                wv = wbf ? __uint_as_float((uint32_t)((const uint16_t *)w)[o] << 16)
                         : ((const float *)w)[o];
            }
            xs[kk][rr] = xv;
            ws[kk][rr] = wv;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < EG_K; ++kk) {
            float a[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = xs[kk][ty + 16 * i];
                bv[i] = ws[kk][tx + 16 * i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        int r = r0 + ty + 16 * i;
        for (int j = 0; j < 4; ++j) {
            int c = c0 + tx + 16 * j;
            if (r < n && c < m) {
                float v = acc[i][j] + (b ? b[c] : 0.f);
                y[(size_t)r * m + c] = fminf(fmaxf(v, omin), omax);
            }
        }
    }
}

/* y = x / sqrt(mean(x^2) + eps), times w when w is not NULL. One warp for
 * each row of d values. y may be x. */
__global__ void k_enc_rms(const float *x, const float *w, float *y, int rows, int d, float eps)
{
    PDL_START();
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const float *p = x + (size_t)row * d;
    float s = 0.f;
    for (int i = lane; i < d; i += 32) {
        s += p[i] * p[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, off);
    }
    float r = 1.0f / sqrtf(s / (float)d + eps);
    for (int i = lane; i < d; i += 32) {
        y[(size_t)row * d + i] = p[i] * r * (w ? w[i] : 1.0f);
    }
}

/* The LayerNorm of each row of d values, times w, plus b (either may be
 * NULL). One warp for each row. y may be x. */
__global__ void k_enc_lnorm(const float *x, const float *w, const float *b, float *y, int rows,
                            int d, float eps)
{
    PDL_START();
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const float *p = x + (size_t)row * d;
    float s = 0.f;
    for (int i = lane; i < d; i += 32) {
        s += p[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(0xffffffff, s, off);
    }
    float mean = s / (float)d, v = 0.f;
    for (int i = lane; i < d; i += 32) {
        float t = p[i] - mean;
        v += t * t;
    }
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(0xffffffff, v, off);
    }
    float r = 1.0f / sqrtf(v / (float)d + eps);
    for (int i = lane; i < d; i += 32) {
        y[(size_t)row * d + i] = (p[i] - mean) * r * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
}

/* y = gelu(x), n values: the tanh form, or the erf form (use_erf). */
__global__ void k_enc_gelu(const float *x, float *y, int64_t n, int use_erf)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = use_erf ? 0.5f * v * (1.0f + erff(v * 0.7071067811865476f))
                       : 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

/* y = gelu_tanh(g) * u, n values. */
__global__ void k_enc_gelu_mul(const float *g, const float *u, float *y, int64_t n)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        float t = tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v));
        y[i] = 0.5f * v * (1.0f + t) * u[i];
    }
}

/* x += sc y, n values. */
__global__ void k_enc_add(float *x, const float *y, int64_t n, float sc)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += sc * y[i];
    }
}

/* y = clamp(x, lo, hi), n values. */
__global__ void k_enc_clamp(const float *x, float *y, int64_t n, float lo, float hi)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fminf(fmaxf(x[i], lo), hi);
    }
}

/* y = clamp(y + b, lo, hi) in place; b may be NULL. */
__global__ void k_enc_bias_clamp(float *y, const float *b, int rows, int cols, float lo, float hi)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        y[i] = fminf(fmaxf(y[i] + (b ? b[i % cols] : 0.f), lo), hi);
    }
}

/* y = x sigmoid(x), n values. */
__global__ void k_enc_silu(const float *x, float *y, int64_t n)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* y = x vec for each row of cols values. */
__global__ void k_enc_mul_vec(const float *x, const float *vec, float *y, int rows, int cols)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        y[i] = x[i] * vec[i % cols];
    }
}

/* y (rows, cols) = a sigmoid(b), a and b the halves of each row of x. */
__global__ void k_enc_glu(const float *x, float *y, int rows, int cols)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        int64_t row = i / cols;
        int c = (int)(i % cols);
        const float *xa = x + row * 2 * cols;
        y[i] = xa[c] / (1.0f + expf(-xa[cols + c]));
    }
}

/* The causal depthwise conv: y[i, c] = sum_j w[c, j] x[i - kw + 1 + j, c]. */
__global__ void k_enc_dwconv(const float *x, const float *w, float *y, int t, int c, int kw)
{
    PDL_START();
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)t * c) {
        return;
    }
    int i = (int)(idx / c), ch = (int)(idx % c);
    float acc = 0.f;
    for (int j = 0; j < kw; ++j) {
        int src = i - kw + 1 + j;
        if (src >= 0) {
            acc += w[(size_t)ch * kw + j] * x[(size_t)src * c + ch];
        }
    }
    y[idx] = acc;
}

/* The local attention of gemma4a: query i sees the keys i - span + 1 .. i;
 * key j of that list adds q r[1 + j]; the scores get the soft cap, and an
 * invalid key -1e9. One warp for each (query, head); lane j < span scores
 * key j, then each lane sums its values of the output. */
__global__ void k_enc_local_attn(const float *q, const float *k, const float *v, const float *r,
                                 const int *valid, float *o, int t, int heads, int hd, int span,
                                 float cap)
{
    PDL_START();
    int64_t row = (int64_t)blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= (int64_t)t * heads) {
        return;
    }
    int i = (int)(row / heads), h = (int)(row % heads);
    const float *qi = q + row * hd;
    int kp = i - span + 1 + lane;
    float sc = -INFINITY;
    if (lane < span) {
        sc = -1e9f;
        if (kp >= 0 && valid[kp]) {
            const float *kk = k + ((size_t)kp * heads + h) * hd;
            const float *rj = r + ((size_t)(1 + lane) * heads + h) * hd;
            float a = 0.f;
            for (int d = 0; d < hd; ++d) {
                a += qi[d] * (kk[d] + rj[d]);
            }
            sc = tanhf(a / cap) * cap;
        }
    }
    float m = sc;
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
    }
    float p = lane < span ? expf(sc - m) : 0.f;
    float sum = p;
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, off);
    }
    p /= sum;
    for (int d0 = 0; d0 < hd; d0 += 32) {
        /* every lane takes each pass, so the shuffles agree */
        int d = d0 + lane;
        float acc = 0.f;
        for (int j = 0; j < span; ++j) {
            float pj = __shfl_sync(0xffffffff, p, j);
            int kj = i - span + 1 + j;
            if (kj >= 0 && d < hd) {
                acc += pj * v[((size_t)kj * heads + h) * hd + d];
            }
        }
        if (d < hd) {
            o[row * hd + d] = acc;
        }
    }
}

/* The axial 2D RoPE of gemma4v, in place on x (n, heads, hd): the first half
 * of each head turns with the x position and the second with the y
 * position; each half is a NEOX rotation (its first quarter with its second)
 * at the angles pos * inv[j], j < hd / 4. pos is (n, 2) int32. */
__global__ void k_enc_rope2d(float *x, const int *pos, const float *inv, int n, int heads, int hd)
{
    PDL_START();
    int q4 = hd / 4, h2 = hd / 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n * heads * 2 * q4;
    if (i >= total) {
        return;
    }
    int j = (int)(i % q4);
    int part = (int)((i / q4) % 2);
    int64_t row = i / (2 * q4);          /* patch * heads + head */
    int patch = (int)(row / heads);
    float ang = (float)pos[2 * patch + part] * inv[j];
    float c = cosf(ang), s = sinf(ang);
    float *p = x + row * hd + part * h2;
    float a = p[j], b = p[j + q4];
    p[j] = a * c - b * s;
    p[j + q4] = b * c + a * s;
}

/* The attention of every query over every key, scale 1. q, k, v, and o are
 * (n, heads, hd), hd at most 128. One warp for each (query, head); the 4
 * warps of a block take 4 queries of one head and share the tiles of 32 keys
 * and values in shared memory. Lane j scores key j of a tile; the softmax is
 * online. */
__global__ void k_enc_attn(const float *q, const float *k, const float *v, float *o,
                           int n, int heads, int hd)
{
    PDL_START();
    extern __shared__ float esm[];
    int ld = hd + 1;
    float *ks = esm, *vs = ks + 32 * ld;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float *qs = vs + 32 * ld + warp * hd;
    int h = blockIdx.y, qi = blockIdx.x * 4 + warp;
    bool live = qi < n;
    for (int d = lane; d < hd; d += 32) {
        qs[d] = live ? q[((size_t)qi * heads + h) * hd + d] : 0.f;
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float m = -INFINITY, l = 0.f;
    for (int k0 = 0; k0 < n; k0 += 32) {
        __syncthreads();
        for (int t = threadIdx.x; t < 32 * hd; t += blockDim.x) {
            int j = t / hd, d = t % hd, kj = k0 + j;
            size_t src = ((size_t)kj * heads + h) * hd + d;
            ks[j * ld + d] = kj < n ? k[src] : 0.f;
            vs[j * ld + d] = kj < n ? v[src] : 0.f;
        }
        __syncthreads();
        bool kv = k0 + lane < n;
        float sc = -INFINITY;
        if (kv) {
            sc = 0.f;
            for (int d = 0; d < hd; ++d) {
                sc += qs[d] * ks[lane * ld + d];
            }
        }
        float mt = sc;
        for (int off = 16; off > 0; off >>= 1) {
            mt = fmaxf(mt, __shfl_xor_sync(0xffffffff, mt, off));
        }
        float mn = fmaxf(m, mt);
        float p = kv ? expf(sc - mn) : 0.f;
        float scale = m == -INFINITY ? 0.f : expf(m - mn);
        float ps = p;
        for (int off = 16; off > 0; off >>= 1) {
            ps += __shfl_xor_sync(0xffffffff, ps, off);
        }
        l = l * scale + ps;
        for (int i = 0; i < 4; ++i) {
            acc[i] *= scale;
        }
        for (int j = 0; j < 32; ++j) {
            float pj = __shfl_sync(0xffffffff, p, j);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = lane + 32 * i;
                if (d < hd) {
                    acc[i] += pj * vs[j * ld + d];
                }
            }
        }
        m = mn;
    }
    if (live) {
        for (int i = 0; i < 4; ++i) {
            int d = lane + 32 * i;
            if (d < hd) {
                o[((size_t)qi * heads + h) * hd + d] = acc[i] / l;
            }
        }
    }
}

extern "C" {

const char *gg_last_error(void)
{
    return gg_error;
}

/* gg_where: the program (its handle), the record queued last and its
 * operation, the segment [s0, s1) of it (out: 4 ints), and whether the
 * runs wait after each record (NP_GEMMA_GPU_SYNC_CHECK). */
const void *gg_where(int *out)
{
    out[0] = gg_where_pc;
    out[1] = gg_where_op;
    out[2] = gg_where_s0;
    out[3] = gg_where_s1;
    out[4] = gg_sync_check();
    return gg_where_prog;
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

/* Clear the last error of the CUDA runtime in this thread. A failed call
 * that the caller handles (a cudaMalloc when the memory runs out) leaves its
 * error there, and the next run would report it (cudaGetLastError at the
 * end of gg_exec). Return the error it cleared. */
int gg_clear_error(void)
{
    return (int)cudaGetLastError();
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
/* n zero bytes at d, on the stream of the programs (a fresh cache:
 * _DevCache.attach). */
int gg_zero(void *d, size_t n)
{
    CK(cudaMemsetAsync(d, 0, n, gg_stream));
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

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
     * program (the 26B keeps float32 products; see gpu.py). Bit 2: the int8
     * products of the int4 matrices (k_gemm_q8) all the same; with bit 4,
     * the int16 form of x (k_quant_x2: two int8 planes) for them and for
     * the KQ_Q4X experts of a prompt (k_moe_gemm_q4x). */
    g->use_graph = use_graph & 1;
    g->tc = (use_graph & 2) ? 0 : 1;
    g->i8 = (use_graph & 4) ? ((use_graph & 16) ? 2 : 1) : 0;
    g->atc = (use_graph & 8) ? 1 : 0;
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
    /* A long run of records becomes several graphs: the first has
     * GG_SEG_FIRST records, the next ones GG_SEG_MAX. The GPU starts a graph
     * only when cudaGraphLaunch returns, about 1.5 us for each kernel. With a
     * small first graph, the GPU starts early, and the launches of the next
     * graphs run while it works. */
    int in_seg = 0, run = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        if (is_boundary(r->op)) {
            in_seg = 0;
            run = 0;
            if (r->op == GP_TO_HOST && r->v[9] + 1 > g->n_ev) {
                g->n_ev = (int)r->v[9] + 1;
            }
            continue;
        }
        if (in_seg && pc - g->seg[g->n_seg - 1].start >= (run == 1 ? GG_SEG_FIRST : GG_SEG_MAX)) {
            in_seg = 0;
        }
        if (!in_seg) {
            g->seg[g->n_seg].start = pc;
            ++g->n_seg;
            ++run;
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
        } else if (r->op == GP_ENC_LINEAR && r->v[2] && r->v[5] > MT_MAX) {
            n = (size_t)r->v[5] * (size_t)r->v[7];      /* the rows x cols of x */
        } else if (r->op == GP_KQ_GROUP_MOE && (r->v[16] == KQ_BF16 || r->v[16] == KQ_BF12)) {
            /* the two bfloat16 planes of h (or act2) for the shared expert */
            n = 2 * (size_t)r->v[3] * (size_t)(r->v[6] > r->v[7] ? r->v[6] : r->v[7]);
        } else if (r->op == GP_KQ_LINEAR && r->tag[8] != GP_T_SLOT && r->v[8] > MT_MAX &&
                   (r->v[5] == KQ_BF16 || r->v[5] == KQ_BF12)) {
            n = 2 * (size_t)r->v[8] * (size_t)r->v[7];  /* the two bfloat16 planes of x (k_gemm_bf16_tc) */
        } else if (r->op == GP_MOE_GPU) {
            size_t a = (size_t)r->v[3] * (size_t)r->v[8];
            size_t b = (size_t)r->v[3] * (size_t)r->v[4] * (size_t)r->v[10];
            n = a > b ? a : b;
        }
        need = n > need ? n : need;
    }
    /* the int16 form: 2 bytes for each value, then the scales */
    size_t xh_b = need * sizeof(__half) + (g->i8 == 2 ? need / 8 + 512 : 0);
    if (need > 0 && cudaMalloc(&g->xh, xh_b) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "gg_load: no memory for the float16 scratch");
        return NULL;
    }
    g->xh_n = need;
    /* The bfloat16 rows of the KQ_BF12 matrices of large groups (gg_gemm_bf12),
     * in chunks of at most 16M values (32 MB). */
    size_t wneed = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        size_t n = 0;
        if (r->op == GP_KQ_LINEAR && r->tag[8] != GP_T_SLOT && r->v[8] > MT_MAX && r->v[5] == KQ_BF12) {
            n = (size_t)r->v[6] * (size_t)r->v[7];
        } else if (r->op == GP_KQ_GROUP_MOE && r->v[16] == KQ_BF12) {
            n = (size_t)r->v[6] * (size_t)r->v[7];
        }
        wneed = n > wneed ? n : wneed;
    }
    wneed = bf12_fused_on() ? 0 : wneed < ((size_t)16 << 20) ? wneed : ((size_t)16 << 20);
    if (wneed > 0 && cudaMalloc(&g->w16, wneed * 2) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "gg_load: no memory for the bfloat16 rows of BF12");
        return NULL;
    }
    g->w16_n = wneed;
    /* The parts of k_kq_bf12_tc of class 1 (the tiles of its matrices). */
    size_t bneed = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        size_t n = 0;
        if (r->op == GP_KQ_LINEAR && bt_shape((int)r->v[5], (int)r->v[6], (int)r->v[7]) &&
            bt_cls((int)r->v[6], (int)r->v[7]) == 1) {
            n = (size_t)(r->v[6] + 15) / 16;
        } else if (r->op == GP_KQ_MULTI) {
            for (int m = 0; m < r->v[3]; ++m) {
                if (bt_shape((int)r->v[5 + 4 * m], (int)r->v[6 + 4 * m], (int)r->v[1]) &&
                    bt_cls((int)r->v[6 + 4 * m], (int)r->v[1]) == 1) {
                    n += (size_t)(r->v[6 + 4 * m] + 15) / 16;
                }
            }
        }
        bneed = n > bneed ? n : bneed;
    }
    if (bneed > 0 && (cudaMalloc(&g->btp, bneed * BT_KP * 128 * sizeof(float)) != cudaSuccess ||
                      cudaMalloc(&g->btc, bneed * sizeof(int)) != cudaSuccess ||
                      cudaMemset(g->btc, 0, bneed * sizeof(int)) != cudaSuccess)) {
        snprintf(gg_error, sizeof(gg_error), "gg_load: no memory for the parts of k_kq_bf12_tc");
        return NULL;
    }
    /* The int8 scratch of the products of large groups on the tensor cores. */
    size_t kneed = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        size_t n = 0;
        if (r->op == GP_KQ_LINEAR) {
            n = (size_t)r->v[8] * (size_t)r->v[7];
        } else if (r->op == GP_KQ_MULTI) {
            n = (size_t)r->v[2] * (size_t)r->v[1];
        } else if (r->op == GP_KQ_GROUP_MOE) {
            size_t t = (size_t)r->v[3], P = t * (size_t)r->v[4] + t;
            n = t * (size_t)r->v[6];
            n = P * (size_t)r->v[7] > n ? P * (size_t)r->v[7] : n;
        } else if (r->op == GP_MOE_GPU && r->v[5] == 1) {
            size_t t = (size_t)r->v[3], P = t * (size_t)r->v[4];
            n = t * (size_t)r->v[8];
            n = P * (size_t)r->v[10] > n ? P * (size_t)r->v[10] : n;
        }
        kneed = n > kneed ? n : kneed;
    }
    if (kneed > 0) {
        kneed = (kneed + 255) & ~(size_t)255;
        /* the int16 form: its lo plane after xs and xsum (kt_lo) */
        if (cudaMalloc(&g->kqx, kneed + kneed / 32 * 8 + (g->i8 == 2 ? kneed : 0)) != cudaSuccess) {
            snprintf(gg_error, sizeof(gg_error), "gg_load: no memory for the int8 scratch");
            return NULL;
        }
    }
    g->kqx_n = kneed;
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
/* The test of the reuse of experts in an MTP group (k_router_top_mt). */
int gg_set_reuse(int on)
{
    /* In the order of the stream, so the steps before keep their setting. */
    static int v;
    v = on;
    CK(cudaMemcpyToSymbolAsync(g_reuse, &v, sizeof(int), 0, cudaMemcpyHostToDevice, gg_stream));
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

/* Record the graphs of a program now and upload them to the device (the
 * first run does it otherwise). A graph does not depend on the values of
 * the environment (one graph serves every position), so the program can do
 * it when it is made: its memory is then taken before a caller gives the
 * rest of the memory to other buffers. */
int gg_prepare(void *handle)
{
    gg_prog *g = (gg_prog *)handle;
    if (!g->use_graph) {
        return 0;
    }
    for (int s = 0; s < g->n_seg; ++s) {
        if (g->seg[s].exec == NULL && gg_capture(g, g->seg + s) != 0) {
            return -1;
        }
        CK(cudaGraphUpload(g->seg[s].exec, gg_stream));
    }
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

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
        } else if (r->op == GP_CPU_TASK) {
            rc = gg_cpu_task(r, g->henv);
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
    dim3 grid((unsigned)cdiv(rows, ROWS_PER_BLOCK));
    if (nx == 1) {
        k_q6k_head<1><<<grid, 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
            (const uint8_t *)w, x, out, rows, cols, cap, nx);
    } else {
        k_q6k_head<0><<<grid, 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
            (const uint8_t *)w, x, out, rows, cols, cap, nx);
    }
    CK(cudaGetLastError());
    return 0;
}

/* The output head of a Q4_0 matrix w (k_q4_head), as gg_q6k_head. cols is a
 * multiple of 32 and x is 8-byte aligned. */
int gg_q4_head(const void *w, const float *x, float *out, int rows, int cols, float cap,
               int nx)
{
    if (nx < 1 || nx > HEAD_MAX) {
        snprintf(gg_error, sizeof(gg_error), "gg_q4_head: 1 to %d rows of x", HEAD_MAX);
        return -1;
    }
    if ((uintptr_t)x % 8 != 0 || cols % 32 != 0) {
        snprintf(gg_error, sizeof(gg_error), "gg_q4_head: x not 8-byte aligned or cols %d", cols);
        return -1;
    }
    const uint8_t *wb = (const uint8_t *)w;
#define Q4H(NXV, RV, UV) k_q4_head<NXV, RV, UV><<<dim3((unsigned)cdiv(rows, ROWS_PER_BLOCK * RV)), \
        32 * ROWS_PER_BLOCK, 0, gg_stream>>>(wb, x, out, rows, cols, cap, nx)
    /* The rows of a decode step and of an MTP verify group (1 to 4) have
     * their own kernels: the compiler then unrolls the loop over x. A warp
     * of a group does R rows of w, so that a load of x serves R rows. For the
     * 12B head (262144 x 3840) on an RTX 5060 Ti: 1.43, 1.58, 1.70, and
     * 1.89 ms for 1 to 4 rows of x, against 1.45, 1.93, 2.77, and 3.27 ms
     * with one row of w for each warp. */
    switch (nx) {
    case 1: Q4H(1, 1, 2); break;
    case 2: Q4H(2, 2, 4); break;
    case 3: Q4H(3, 4, 1); break;
    case 4: Q4H(4, 4, 1); break;
    default: Q4H(0, 1, 1);
    }
#undef Q4H
    CK(cudaGetLastError());
    return 0;
}

/* k_embed_q4: the row tok[0] (a device int) of the Q4_0 table, times scale,
 * into out. Device addresses; the call does not wait for the GPU. */
int gg_embed_q4(const void *table, const int *tok, float *out, int cols, float scale)
{
    k_embed_q4<<<1, 256, 0, gg_stream>>>((const uint8_t *)table, tok, out, cols, scale);
    CK(cudaGetLastError());
    return 0;
}

/* k_embed_q4_rows: rows rows of the Q4_0 table (tok: rows device ints, -1
 * for zeros) into out. Device addresses; the call does not wait. */
int gg_embed_q4_rows(const void *table, const int *tok, float *out, int rows, int cols,
                     float scale)
{
    k_embed_q4_rows<<<rows, 256, 0, gg_stream>>>((const uint8_t *)table, tok, out, cols, scale);
    CK(cudaGetLastError());
    return 0;
}

/* The best token of each of the rows of x (k_argmax_rows) into out (rows
 * ints). Device addresses; the call does not wait for the GPU. */
int gg_argmax_rows(const float *x, int rows, int vocab, int *out)
{
    k_argmax_rows<<<rows, 1024, 0, gg_stream>>>(x, vocab, out);
    CK(cudaGetLastError());
    return 0;
}

/* The K largest values of each of the rows of x (vocab values each), the row
 * max, and the sum of exp((x - max) inv_t) (k_topk_rows). ids and vals have
 * rows K entries, stat 2 rows. All are device addresses. The call does not
 * wait for the GPU. */
int gg_topk(const float *x, int rows, int vocab, int K, float inv_t, int *ids, float *vals,
            float *stat)
{
    if (rows < 1 || K < 1 || K > vocab) {
        snprintf(gg_error, sizeof(gg_error), "gg_topk: rows %d, K %d, vocab %d", rows, K, vocab);
        return -1;
    }
    k_topk_rows<<<rows, TOPK_T, 0, gg_stream>>>(x, vocab, K, inv_t, ids, vals, stat);
    CK(cudaGetLastError());
    return 0;
}

/* Apply the first n tokens of the log of a verify group to conv and S (the
 * device arrays of one linear layer), as gdn_commit of the CPU. */
int gg_gdn_commit(float *conv, float *S, const float *log, int n, int kernel, int k_heads,
                  int v_heads, int k_dim, int v_dim)
{
    if (k_dim != 128 || v_dim != 128 || kernel > 9) {
        snprintf(gg_error, sizeof(gg_error), "gg_gdn_commit: k_dim = v_dim = 128");
        return -1;
    }
    int cd = 2 * k_heads * k_dim + v_heads * v_dim;
    size_t lrow = (size_t)cd + (size_t)v_heads * (k_dim + v_dim + 1);
    k_gdn_commit_conv<<<(unsigned)cdiv(cd, 256), 256, 0, gg_stream>>>(conv, log, n, kernel, cd,
                                                                    lrow);
    k_gdn_commit_heads<128><<<(unsigned)v_heads, 128, 0, gg_stream>>>(S, log, n, cd, lrow);
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
    cudaFree(g->w16);
    cudaFree(g->btp);
    cudaFree(g->btc);
    cudaFree(g->kqx);
    cudaFreeHost(g->henv);
    free(g->hcode);
    free(g);
    return 0;
}

}  /* extern "C" */
