/* ---------- EXL3 MUL1 matrices (KQ_E3; EXL3_PLAN.md) ----------
 *
 * The form of backends/cpu/csrc/kernels/kq/exl3.c: a matrix of k inputs and
 * n outputs, K bits, is one block: the trellis of the file of ExLlamaV3
 * ((k / 16) (n / 16) tiles of 16 x 16 weights, 32 K bytes each), suh (k
 * float16), svh (n float16). The product y = x W:
 *
 *     xh = H (x * suh)          H: the Hadamard transform of each 128
 *     y' = xh W'                W': the MUL1 values of the codes
 *     y  = (H y') * svh
 *
 * Three kernels: k_e3_had_in (xh as int8, a scale and a sum for each 128),
 * k_e3_gemv (y' of a slice of the inputs: parts), k_e3_had_out (the sum of
 * the parts, the constant of MUL1, the transform, svh). A MUL1 value is
 * (1024 + s) k_inv + k_bias with s the byte sum of code * 0x83DCD12D, so
 * dp4a of the bytes of the product (unsigned) and of x splat in 4 bytes
 * (signed) gives q s in one instruction (the int8 GEMV of ExLlamaV3,
 * exl3_gemv_int8_kernel.cuh, MIT license; here with a scale for each 128
 * values and no tensor cores, so Pascal (dp4a, sm_61) runs it as it is).
 *
 * A warp takes a tile column (16 outputs) of a slice of the inputs; lane l
 * the stream positions 8 l to 8 l + 7 of each tile: rows 2 (l % 4) + {0, 1,
 * 8, 9} of columns l / 4 and l / 4 + 8 (tensor_core_perm of ExLlamaV3).
 * Its 8 codes are in the bits [8 l K + K - 16, 8 l K + 8 K) of the circular
 * stream of the tile (4 words at most): aligned by a funnel shift, then at
 * constant places. The parts of the slices go to their own rows (no
 * atomics: the same bits in every run). */

#define E3_BASE 70
#define E3_NT 8             /* rows of x of a block of k_e3_gemv */
#define E3_SLICE_MAX 32     /* blocks of 128 inputs in a slice (the shared memory) */

__host__ __device__ __forceinline__ int kq_e3_k(int type)
{
    return type > E3_BASE && type <= E3_BASE + 8 ? type - E3_BASE : 0;
}

#define E3_KINV 0.0067672729492187500f      /* float16 0x1eee */
#define E3_KBIAS (-10.3828125f)             /* float16 0xc931 */

__device__ __forceinline__ int e3_dp4a_us(uint32_t a, uint32_t b, int c)
{
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

/* Code m of a lane from its aligned window A0:A1:A2 (bit 0 the top of A0). */
template <int K, int m>
__device__ __forceinline__ uint32_t e3_code(uint32_t A0, uint32_t A1, uint32_t A2)
{
    constexpr int s = m * K, e = s + 16;
    if constexpr (e <= 32) {
        return (A0 >> (32 - e)) & 0xffffu;
    } else if constexpr (s < 32) {
        return __funnelshift_r(A1, A0, 64 - e) & 0xffffu;
    } else if constexpr (e <= 64) {
        return (A1 >> (64 - e)) & 0xffffu;
    } else if constexpr (s < 64) {
        return __funnelshift_r(A2, A1, 96 - e) & 0xffffu;
    } else {
        return (A2 >> (96 - e)) & 0xffffu;
    }
}

/* The 128-value Hadamard transform in shared memory v (128 threads; scaled
 * by 1 / sqrt(128)); returns the value of thread i. */
__device__ __forceinline__ float e3_fwht128(float *v, float a)
{
    int i = threadIdx.x;
    v[i] = a;
    __syncthreads();
    #pragma unroll
    for (int h = 1; h < 128; h <<= 1) {
        float u = v[i], w = v[i ^ h];
        __syncthreads();
        v[i] = (i & h) ? w - u : u + w;
        __syncthreads();
    }
    return v[i] * 0.08838834764831845f;
}

__device__ __forceinline__ float e3_block_max(float a, float *red)
{
    for (int o = 16; o > 0; o >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    }
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x / 32] = a;
    }
    __syncthreads();
    a = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    __syncthreads();
    return a;
}

__device__ __forceinline__ float e3_block_sum(float a, float *red)
{
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, o);
    }
    if (threadIdx.x % 32 == 0) {
        red[threadIdx.x / 32] = a;
    }
    __syncthreads();
    a = (red[0] + red[1]) + (red[2] + red[3]);
    __syncthreads();
    return a;
}

/* xh = H (x * suh) of block b (blockIdx.x) of row j (blockIdx.y) of x
 * (row stride ldx) as int8 xq (t x k), its scale xs and its sum xn (t x k /
 * 128). 128 threads. */
__global__ void __launch_bounds__(128) k_e3_had_in(const float *x, size_t ldx, const __half *suh, int k,
                                                   int8_t *xq, float *xs, int *xn)
{
    __shared__ float v[128];
    __shared__ float red[4];
    int b = blockIdx.x, j = blockIdx.y, i = threadIdx.x, nb = k / 128;
    float a = x[(size_t)j * ldx + 128 * b + i] * __half2float(suh[128 * b + i]);
    a = e3_fwht128(v, a);
    float m = e3_block_max(fabsf(a), red);
    float inv = m > 0.f ? 127.f / m : 0.f;
    int q = __float2int_rn(a * inv);
    q = max(-127, min(127, q));
    xq[(size_t)j * k + 128 * b + i] = (int8_t)q;
    float s = e3_block_sum((float)q, red);      /* exact: |sum| < 2^24 */
    if (i == 0) {
        xs[(size_t)j * nb + b] = m / 127.f;
        xn[(size_t)j * nb + b] = (int)s;
    }
}

/* The parts of y' (without the constant): block (ob + slice (n / 128), chunk
 * of NT rows); warp w the tile column 8 ob + w over the blocks of 128
 * inputs [b0, b1) of the slice. part (S x t x n): k_inv sum xs q s. The 8
 * tiles of a block of inputs: their words first (loads in flight), then the
 * codes and the products. */
template <int K, int NT>
__global__ void __launch_bounds__(256) k_e3_gemv(const uint8_t *wm, int k, int n, const int8_t *xq,
                                                 const float *xs, int t, int S, float *part)
{
    extern __shared__ int4 e3_sm[];
    const int nob = n / 128, nb = k / 128;
    const int ob = blockIdx.x % nob, s = blockIdx.x / nob;
    const int b0 = s * nb / S, b1 = (s + 1) * nb / S, nbs = b1 - b0;
    const int j0 = blockIdx.y * NT, nt = min(NT, t - j0);
    int8_t *xsh = (int8_t *)e3_sm;                      /* nt x nbs 128 */
    float *ssh = (float *)(xsh + NT * E3_SLICE_MAX * 128);  /* nt x nbs */
    const int sl = nbs * 128;
    for (int u = threadIdx.x; u < nt * sl / 16; u += 256) {
        int jj = u / (sl / 16), o = u % (sl / 16);
        ((int4 *)(xsh + jj * sl))[o] = ((const int4 *)(xq + (size_t)(j0 + jj) * k + 128 * b0))[o];
    }
    for (int u = threadIdx.x; u < nt * nbs; u += 256) {
        int jj = u / nbs;
        ssh[jj * nbs + u % nbs] = xs[(size_t)(j0 + jj) * nb + b0 + u % nbs];
    }
    __syncthreads();
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int tj = 8 * ob + warp, tn = n / 16;
    constexpr int W = 8 * K;                            /* words of a tile */
    const int first = 8 * lane * K + K - 16 + 256 * K;
    const int fw = first >> 5, off = first & 31;
    const int i0 = fw % W, i1 = (fw + 1) % W, i2 = (fw + 2) % W, i3 = (fw + 3) % W;
    const int r0 = 2 * (lane % 4);
    float facc[NT][2];
    #pragma unroll
    for (int jj = 0; jj < NT; ++jj) {
        facc[jj][0] = facc[jj][1] = 0.f;
    }
    const uint32_t *p = (const uint32_t *)wm + ((size_t)(8 * b0) * tn + tj) * W;
    const size_t step = (size_t)tn * W;                 /* the next tile row */
    for (int b = 0; b < nbs; ++b, p += 8 * step) {
        uint32_t a[8][4];
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            a[r][0] = __ldg(p + r * step + i0);
            a[r][1] = __ldg(p + r * step + i1);
            a[r][2] = __ldg(p + r * step + i2);
            a[r][3] = __ldg(p + r * step + i3);
        }
        int iacc[NT][2];
        #pragma unroll
        for (int jj = 0; jj < NT; ++jj) {
            iacc[jj][0] = iacc[jj][1] = 0;
        }
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            uint32_t A0 = __funnelshift_l(a[r][1], a[r][0], off), A1 = __funnelshift_l(a[r][2], a[r][1], off);
            uint32_t A2 = __funnelshift_l(a[r][3], a[r][2], off);
            uint32_t w0 = e3_code<K, 0>(A0, A1, A2) * 0x83DCD12Du;
            uint32_t w1 = e3_code<K, 1>(A0, A1, A2) * 0x83DCD12Du;
            uint32_t w2 = e3_code<K, 2>(A0, A1, A2) * 0x83DCD12Du;
            uint32_t w3 = e3_code<K, 3>(A0, A1, A2) * 0x83DCD12Du;
            uint32_t w4 = e3_code<K, 4>(A0, A1, A2) * 0x83DCD12Du;
            uint32_t w5 = e3_code<K, 5>(A0, A1, A2) * 0x83DCD12Du;
            uint32_t w6 = e3_code<K, 6>(A0, A1, A2) * 0x83DCD12Du;
            uint32_t w7 = e3_code<K, 7>(A0, A1, A2) * 0x83DCD12Du;
            const int8_t *xr = xsh + 128 * b + 16 * r + r0;
            #pragma unroll
            for (int jj = 0; jj < NT; ++jj) {
                if (NT == 1 || jj < nt) {
                    uint32_t lo = *(const uint16_t *)(xr + jj * sl);
                    uint32_t hi = *(const uint16_t *)(xr + jj * sl + 8);
                    uint32_t s0 = __byte_perm(lo, 0, 0x0000), s1 = __byte_perm(lo, 0, 0x1111);
                    uint32_t s2 = __byte_perm(hi, 0, 0x0000), s3 = __byte_perm(hi, 0, 0x1111);
                    iacc[jj][0] = e3_dp4a_us(w3, s3, e3_dp4a_us(w2, s2, e3_dp4a_us(w1, s1,
                                  e3_dp4a_us(w0, s0, iacc[jj][0]))));
                    iacc[jj][1] = e3_dp4a_us(w7, s3, e3_dp4a_us(w6, s2, e3_dp4a_us(w5, s1,
                                  e3_dp4a_us(w4, s0, iacc[jj][1]))));
                }
            }
        }
        #pragma unroll
        for (int jj = 0; jj < NT; ++jj) {
            if (NT == 1 || jj < nt) {
                float sc = ssh[jj * nbs + b];
                facc[jj][0] += sc * (float)iacc[jj][0];
                facc[jj][1] += sc * (float)iacc[jj][1];
            }
        }
    }
    #pragma unroll
    for (int jj = 0; jj < NT; ++jj) {
        #pragma unroll
        for (int c = 0; c < 2; ++c) {
            float v = facc[jj][c];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            facc[jj][c] = v;
        }
    }
    if (lane % 4 == 0) {
        const int c0 = lane / 4;
        #pragma unroll
        for (int jj = 0; jj < NT; ++jj) {
            if (jj < nt) {
                float *o = part + ((size_t)s * t + j0 + jj) * n + 16 * tj;
                o[c0] = E3_KINV * facc[jj][0];
                o[c0 + 8] = E3_KINV * facc[jj][1];
            }
        }
    }
}

/* y = (H (sum of the parts + c)) * svh of block ob (blockIdx.x) of row j
 * (blockIdx.y), into out (row stride ldo); cap > 0: cap tanh(y / cap) (the
 * softcap of a head). 128 threads. */
__global__ void __launch_bounds__(128) k_e3_had_out(const float *part, int S, int t, int n, const float *xs,
                                                    const int *xn, int nb, const __half *svh, float *out,
                                                    size_t ldo, float cap)
{
    __shared__ float v[128];
    __shared__ float red[4];
    int ob = blockIdx.x, j = blockIdx.y, i = threadIdx.x;
    float c = 0.f;
    for (int b = i; b < nb; b += 128) {
        c += xs[(size_t)j * nb + b] * (float)xn[(size_t)j * nb + b];
    }
    c = e3_block_sum(c, red) * (1024.f * E3_KINV + E3_KBIAS);
    float a = c;
    for (int s = 0; s < S; ++s) {
        a += part[((size_t)s * t + j) * n + 128 * ob + i];
    }
    a = e3_fwht128(v, a) * __half2float(svh[128 * ob + i]);
    if (cap > 0.f) {
        a = cap * tanhf(a / cap);
    }
    out[(size_t)j * ldo + 128 * ob + i] = a;
}

/* The slices of the inputs of a product: blocks enough for the GPU, at most
 * E3_SLICE_MAX blocks of 128 inputs in a slice, and parts of at most 16M
 * floats. */
static int e3_nt(int64_t t)
{
    return t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : E3_NT;
}

static int e3_slices(int64_t t, int k, int n)
{
    int nb = k / 128;
    int64_t base = (int64_t)(n / 128) * ((t + e3_nt(t) - 1) / e3_nt(t));
    int S = (int)((320 + base - 1) / base);
    int lo = (nb + E3_SLICE_MAX - 1) / E3_SLICE_MAX;
    if (S < lo) {
        S = lo;
    }
    int64_t cap = ((int64_t)16 << 20) / ((int64_t)t * n);
    if (S > cap && cap >= lo) {
        S = (int)cap;
    }
    return S < 1 ? 1 : S > nb ? nb : S;
}

/* The bytes of the scratch of a product: xq, xs, xn, the parts. */
static size_t e3_ws_bytes(int64_t t, int k, int n)
{
    size_t a = ((size_t)t * k + 255) & ~(size_t)255;
    size_t b = ((size_t)t * (k / 128) * 8 + 255) & ~(size_t)255;
    return a + b + (size_t)e3_slices(t, k, n) * t * n * 4 + 256;
}

/* y (t x n, row stride ldo) = x W for a KQ_E3 matrix w (K bits) on the rows
 * of x (t x k, row stride ldx). ws: e3_ws_bytes(t, k, n). */
static int e3_linear_launch(const uint8_t *w, int K, int n, int k, const float *x, size_t ldx, int64_t t,
                            float *out, size_t ldo, uint8_t *ws, size_t ws_n, float cap, cudaStream_t s)
{
    if (K < 1 || K > 8 || k % 128 || n % 128 || ws == NULL || ws_n < e3_ws_bytes(t, k, n)) {
        return -1;
    }
    const int nb = k / 128, S = e3_slices(t, k, n);
    int8_t *xq = (int8_t *)ws;
    float *xs = (float *)(ws + (((size_t)t * k + 255) & ~(size_t)255));
    int *xn = (int *)(xs + (size_t)t * nb);
    float *part = (float *)((uint8_t *)xs + (((size_t)t * nb * 8 + 255) & ~(size_t)255));
    const __half *suh = (const __half *)(w + (size_t)k * n * K / 8), *svh = suh + k;
    k_e3_had_in<<<dim3((unsigned)nb, (unsigned)t), 128, 0, s>>>(x, ldx, suh, k, xq, xs, xn);
    const int nt = e3_nt(t);
    dim3 g((unsigned)((n / 128) * S), (unsigned)((t + nt - 1) / nt));
    size_t sm = (size_t)nt * E3_SLICE_MAX * (128 + 4);
#define E3_CASE(KK) case KK: \
        if (nt == 1) k_e3_gemv<KK, 1><<<g, 256, sm, s>>>(w, k, n, xq, xs, (int)t, S, part); \
        else if (nt == 2) k_e3_gemv<KK, 2><<<g, 256, sm, s>>>(w, k, n, xq, xs, (int)t, S, part); \
        else if (nt == 4) k_e3_gemv<KK, 4><<<g, 256, sm, s>>>(w, k, n, xq, xs, (int)t, S, part); \
        else k_e3_gemv<KK, E3_NT><<<g, 256, sm, s>>>(w, k, n, xq, xs, (int)t, S, part); \
        break;
    switch (K) {
    E3_CASE(1) E3_CASE(2) E3_CASE(3) E3_CASE(4) E3_CASE(5) E3_CASE(6) E3_CASE(7) E3_CASE(8)
    }
#undef E3_CASE
    k_e3_had_out<<<dim3((unsigned)(n / 128), (unsigned)t), 128, 0, s>>>(part, S, (int)t, n, xs, xn, nb, svh,
                                                                       out, ldo, cap);
    return 0;
}

/* A KQ_E3 product outside the programs (a test; the head of a model): the
 * addresses are device addresses; the scratch is the library's own. */
static uint8_t *e3_lib_ws;
static size_t e3_lib_ws_n;

extern "C" int rc_gpu_e3_linear(const void *w, int K, int n, int k, const float *x, int t, float *out,
                                float cap)
{
    size_t need = e3_ws_bytes(t, k, n);
    if (need > e3_lib_ws_n) {
        if (e3_lib_ws != NULL) {
            cudaStreamSynchronize(gg_stream);
            cudaFree(e3_lib_ws);
        }
        e3_lib_ws = NULL;
        e3_lib_ws_n = 0;
        if (cudaMalloc(&e3_lib_ws, need) != cudaSuccess) {
            return -1;
        }
        e3_lib_ws_n = need;
    }
    if (e3_linear_launch((const uint8_t *)w, K, n, k, x, (size_t)k, t, out, (size_t)n, e3_lib_ws, e3_lib_ws_n,
                         cap, gg_stream) != 0) {
        return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}
