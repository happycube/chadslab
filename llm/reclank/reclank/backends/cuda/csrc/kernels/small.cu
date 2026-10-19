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
/* GP_BF16_ROUND: x, n. x rounded to bfloat16 values (to nearest even) in place. */
__global__ void k_bf16_round(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < DI(1)) {
        float *x = DP(float, 0);
        unsigned u = __float_as_uint(x[i]);
        x[i] = __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
    }
}

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
        amax = fmaxf(amax, __shfl_xor_sync(RC_FULL_MASK, amax, o));
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
        if (DP(int16_t, 6) == DP(int16_t, 4) && DP(float, 7) != DP(float, 5)) {
            /* keys only (ko_tab): the scale of the values is that of the
             * keys times 1 / c (operand 9) */
            if (l == 0) {
                DP(float, 7)[g] = DP(float, 5)[g] * df(r, e, 9);
            }
        } else {
            quant32_i16_warp(vl, DP(int16_t, 6) + (size_t)g * 32, DP(float, 7) + g, l);
        }
    }
}

/* k_kv_write for the int8 cache: a scale of max |x| / 127 for each group of
 * 32 values, as gemma_quant_group32_i8 of the CPU. The int8 cache has no
 * float rows. */
/* FP4-Local's FP8 cache (docs/FP4LOCAL_PLAN.md 2.9, vLLM's --kv-cache-dtype fp8): the bytes of the int8
 * cache are E4M3 codes and the scales 1 (K8 = e4m3_rne_satfinite(x / 1.0)); a state of the process
 * (rc_gpu_set_kv_e4m3) that the writes and every reader of the int8 cache take. */
__device__ int gg_kv_e4m3_d = 0;

extern "C" int rc_gpu_set_kv_e4m3(int on)
{
    return cudaMemcpyToSymbol(gg_kv_e4m3_d, &on, sizeof(int)) == cudaSuccess ? 0 : 1;
}

/* the E4M3 code of x (to nearest even, subnormals, saturating at 448: cvt.rn.satfinite.e4m3) */
__device__ __forceinline__ uint8_t rc_e4m3_code(float x)
{
    uint8_t s = (__float_as_uint(x) >> 24) & 0x80;
    float a = fabsf(x);
    if (!(a < 448.f)) {
        return s | (a == a ? 0x7E : 0x7F);          /* (NaN stays NaN) */
    }
    if (a < 0.015625f) {
        return s | (uint8_t)(int)rintf(a * 512.f);  /* subnormal codes 0-7, 8 = 2^-6 */
    }
    int e;
    float m = frexpf(a, &e);
    int q = (int)rintf((m * 2.f - 1.f) * 8.f);
    int E = e - 1 + 7;
    if (q == 8) {
        q = 0;
        ++E;
    }
    int c = (E << 3) | q;
    return s | (uint8_t)(c > 0x7E ? 0x7E : c);
}

__device__ __forceinline__ float rc_e4m3_val(uint32_t b)
{
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? (float)m * (1.f / 512.f) : __uint_as_float(((uint32_t)(e + 120) << 23) | ((uint32_t)m << 20));
    return (b & 0x80) ? -v : v;
}

/* a byte of the int8 cache: int8, or E4M3 with the FP8 cache */
__device__ __forceinline__ float rc_kv8_val(uint32_t b)
{
    return gg_kv_e4m3_d ? rc_e4m3_val(b & 0xff) : (float)(int8_t)(b & 0xff);
}

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
    if (gg_kv_e4m3_d) {
        for (int kv = 0; kv < 2; ++kv) {
            DP(uint8_t, kv ? 6 : 4)[i] = rc_e4m3_code(DP(const float, kv)[i]);
            if (l == 0) {
                DP(float, kv ? 7 : 5)[g] = 1.f;
            }
        }
        return;
    }
    for (int kv = 0; kv < 2; ++kv) {
        float x = DP(const float, kv)[i];
        float amax = fabsf(x);
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(RC_FULL_MASK, amax, o));
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

/* The TQ6 cache of the Qwen models (reclank/formats/tq6.py). The tables and the
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
        float p = __shfl_xor_sync(RC_FULL_MASK, x, h);
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
            s += __shfl_xor_sync(RC_FULL_MASK, s, o);
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
        int i8 = __shfl_sync(RC_FULL_MASK, lo, (l + 8) & 31);
        int i16 = __shfl_sync(RC_FULL_MASK, lo, (l + 16) & 31);
        int i24 = __shfl_sync(RC_FULL_MASK, lo, (l + 24) & 31);
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
    /* operand 3 (or none): the source (gpu35._fuse: the COPY before) */
    const float *src = r->tag[3] != GP_T_NONE ? DP(const float, 3) + (size_t)g * 32 : x;
    x[l] = tq6_rot_warp(src[l], l, DI(2) != 0);
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
