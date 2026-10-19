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
 * atomics: the same bits in every run).
 *
 * A large group (E3_TC_MIN rows or more, on the tensor cores) takes
 * k_e3_mma in place of k_e3_gemv: the same integers on mma.sync m16n8k32
 * (u8 x s8). s (0 to 1020) is l + 256 h with l a byte and h 0 to 3, so two
 * MMAs give the sums of l q and of h q exactly, and s q = l q + 256 h q.
 * The fragment of A (16 outputs x 32 inputs, the two tiles of a tile
 * column at tile rows r and r + 1) takes the 8 codes of a lane as they are:
 * the inputs of the MMA are those of the tiles in another order (k = 4 c + i
 * of a tile is its input 2 c + (i & 1) + 8 (i >> 1)), and the rows of x
 * go to shared memory in that order. The integer of each block of 128
 * inputs of an MMA step (two tile rows) is exact; the float sums run in
 * another order than those of the GEMV (the bits differ a little from
 * k_e3_gemv). */

#define E3_BASE 70
/* The largest int16 value of x16 (k_e3_had_in, k_e3b_had_in): the high byte
 * of 256 h + l (l in [-128, 127]) stays within [-127, 127]. */
#define E3_Q16 32639
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
    return rc_dp4a_us(a, b, c);     /* (port.cuh: a scalar form below sm_61) */
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

/* e3_fwht128 by one warp, no barrier: lane l holds the values l + 32 j (j
 * 0 to 3) in a; the strides 1 to 16 between the lanes, 32 and 64 within
 * them, in the order of e3_fwht128 (the same sums: the same bits). */
__device__ __forceinline__ void e3_fwht128_w(float *a)
{
    const int lane = threadIdx.x % 32;
    #pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float w = __shfl_xor_sync(0xffffffffu, a[j], h);
            a[j] = (lane & h) ? w - a[j] : a[j] + w;
        }
    }
    float b0 = a[0] + a[1], b1 = a[0] - a[1], b2 = a[2] + a[3], b3 = a[2] - a[3];
    a[0] = (b0 + b2) * 0.08838834764831845f;
    a[1] = (b1 + b3) * 0.08838834764831845f;
    a[2] = (b0 - b2) * 0.08838834764831845f;
    a[3] = (b1 - b3) * 0.08838834764831845f;
}

__device__ __forceinline__ float e3_warp_max(float a)
{
    for (int o = 16; o > 0; o >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    }
    return a;
}

__device__ __forceinline__ float e3_warp_sum(float a)
{
    for (int o = 16; o > 0; o >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, o);
    }
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
 * 128). 128 threads. perm: each 16 values of xq in the order of the inputs
 * of k_e3_mma (input 2 c + e + 8 h of a tile at 4 c + e + 2 h). */
__global__ void __launch_bounds__(128) k_e3_had_in(const float *x, size_t ldx, const __half *suh, int k,
                                                   int8_t *xq, float *xs, int *xn, int perm, int x16 = 0)
{
    __shared__ float v[128];
    __shared__ float red[4];
    int b = blockIdx.x, j = blockIdx.y, i = threadIdx.x, nb = k / 128;
    float a = x[(size_t)j * ldx + 128 * b + i] * __half2float(suh[128 * b + i]);
    a = e3_fwht128(v, a);
    float m = e3_block_max(fabsf(a), red);
    /* x16: int16 values as k_e3b_had_in x16 (the high bytes, then the low) */
    const float qm = x16 ? (float)E3_Q16 : 127.f;
    float inv = m > 0.f ? qm / m : 0.f;
    int q = __float2int_rn(a * inv);
    q = x16 ? max(-E3_Q16, min(E3_Q16, q)) : max(-127, min(127, q));
    int u = i % 16;
    int pos = perm ? (i & ~15) + 4 * ((u & 7) >> 1) + (u & 1) + 2 * (u >> 3) : i;
    if (x16) {
        int h = (q + 128) >> 8;
        xq[2 * ((size_t)j * k + 128 * b) + pos] = (int8_t)h;
        xq[2 * ((size_t)j * k + 128 * b) + 128 + pos] = (int8_t)(q - 256 * h);
    } else {
        xq[(size_t)j * k + 128 * b + pos] = (int8_t)q;
    }
    float s = e3_block_sum((float)q, red);      /* exact: |sum| < 2^24 */
    if (i == 0) {
        xs[(size_t)j * nb + b] = m / qm;
        xn[(size_t)j * nb + b] = (int)s;
    }
}

/* The parts of y' (without the constant): block (ob + slice (n / 128), chunk
 * of NT rows); warp w the tile column 8 ob + w over the blocks of 128
 * inputs [b0, b1) of the slice. part (S x t x n): k_inv sum xs q s. The 8
 * tiles of a block of inputs: their words first (loads in flight), then the
 * codes and the products.
 *
 * tw (a batch, NT 1): row j0 = blockIdx.y + it0 of x is item j0, the matrix
 * tw[j0] (null: none), its part at row j0 of t (the items); ik (or null):
 * the K of each item, which runs only in the GEMV of its K. */
template <int K, int NT>
__global__ void __launch_bounds__(256) k_e3_gemv(const uint8_t *wm, int k, int n, const int8_t *xq,
                                                 const float *xs, int t, int S, float *part,
                                                 const uint8_t *const *tw = nullptr, int it0 = 0,
                                                 const int *ik = nullptr)
{
    extern __shared__ int4 e3_sm[];
    const int nob = n / 128, nb = k / 128;
    const int ob = blockIdx.x % nob, s = blockIdx.x / nob;
    const int b0 = s * nb / S, b1 = (s + 1) * nb / S, nbs = b1 - b0;
    int j0 = blockIdx.y * NT, nt = min(NT, t - j0);
    if (tw != nullptr) {
        j0 = blockIdx.y + it0;
        nt = 1;
        wm = tw[j0];
        if (wm == nullptr || (ik != nullptr && ik[j0] != K)) {
            return;
        }
    }
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

#define E3_TC_MIN 32         /* the rows of a group from which k_e3_mma runs */
#define E3_BN 64             /* rows of x of a block of k_e3_mma */
#define E3_XS 144            /* the bytes of a row of x in shared memory (128 + 16: aligned, no bank conflicts) */
#define E3_XS16 272          /* (int16 x: 256 + 16) */

__device__ __forceinline__ void e3_mma_u8s8(int *c, const uint32_t *a, const uint32_t *b)
{
#if RC_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.u8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

/* The l and h bytes of the four codes m0 .. m0 + 3 of a lane (A0:A1:A2 its
 * aligned window), packed for a register of the A fragment. */
template <int K, int m0>
__device__ __forceinline__ void e3_lh(uint32_t A0, uint32_t A1, uint32_t A2, uint32_t &l, uint32_t &h)
{
    uint32_t s0 = e3_dp4a_us(e3_code<K, m0>(A0, A1, A2) * 0x83DCD12Du, 0x01010101u, 0);
    uint32_t s1 = e3_dp4a_us(e3_code<K, m0 + 1>(A0, A1, A2) * 0x83DCD12Du, 0x01010101u, 0);
    uint32_t s2 = e3_dp4a_us(e3_code<K, m0 + 2>(A0, A1, A2) * 0x83DCD12Du, 0x01010101u, 0);
    uint32_t s3 = e3_dp4a_us(e3_code<K, m0 + 3>(A0, A1, A2) * 0x83DCD12Du, 0x01010101u, 0);
    l = __byte_perm(__byte_perm(s0, s1, 0x0040), __byte_perm(s2, s3, 0x0040), 0x5410);
    h = __byte_perm(__byte_perm(s0, s1, 0x0051), __byte_perm(s2, s3, 0x0051), 0x5410);
}

/* The parts of y' of a large group (as k_e3_gemv): block (ob + slice (n /
 * 128), chunk of E3_BN rows); warp w the tile column 8 ob + w (16 outputs)
 * for the E3_BN rows, over the blocks of 128 inputs [b0, b1) of the slice.
 * Each block of inputs: its rows of x (xq in the order of the MMA inputs:
 * k_e3_had_in perm) in shared memory, two buffers, the next block read to
 * registers meanwhile; then 4 steps of two tile rows: the codes of the lane
 * to the l and h fragments, two MMAs for each 8 rows. */
/* The body of k_e3_mma for output block ob of slice s: the nt rows r0,
 * r0 + rs, .. of x (and of the part of the slice: part rows of t); a group
 * of the experts (k_e3g_mma) takes the rows of the pairs of a tile (gate 2 p,
 * up 2 p + 1 of the items). */
template <int K, bool X16 = false>
__device__ __forceinline__ void e3_mma_body(const uint8_t *wm, int k, int n, const int8_t *xq, const float *xs,
                                            int t, int S, float *part, int ob, int s, int r0, int rs, int nt)
{
    /* X16: each block of 128 values of a row of x is 256 bytes, the high
     * bytes then the low bytes of the int16 values (k_e3b_had_in x16) */
    constexpr int XB = X16 ? 256 : 128, XS = X16 ? E3_XS16 : E3_XS, NX = XB / 64;
    __shared__ __align__(16) uint8_t xsh2[2][E3_BN * XS];
    __shared__ float ssh2[2][E3_BN];
    const int nb = k / 128;
    const int b0 = s * nb / S, b1 = (s + 1) * nb / S;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int g = lane / 4, c = lane % 4;
    const int tj = 8 * ob + warp, tn = n / 16;
    constexpr int W = 8 * K;
    const int first = 8 * lane * K + K - 16 + 256 * K;
    const int fw = first >> 5, off = first & 31;
    const int i0 = fw % W, i1 = (fw + 1) % W, i2 = (fw + 2) % W, i3 = (fw + 3) % W;
    float facc[E3_BN / 8][4];
    #pragma unroll
    for (int q = 0; q < E3_BN / 8; ++q) {
        facc[q][0] = facc[q][1] = facc[q][2] = facc[q][3] = 0.f;
    }
    const size_t step = (size_t)tn * W;
    /* the rows of x of a block: E3_BN x XB / 16 int4, NX of them a thread */
    int4 nx[NX];
    float ns = 0.f;
    auto fetch = [&](int b) {
        #pragma unroll
        for (int e = 0; e < NX; ++e) {
            int u = threadIdx.x + 256 * e, jj = u / (XB / 16);
            nx[e] = jj < nt ? *((const int4 *)(xq + ((size_t)(r0 + jj * rs) * k + 128 * b) * (XB / 128)) +
                                u % (XB / 16))
                            : make_int4(0, 0, 0, 0);
        }
        ns = threadIdx.x < nt ? xs[(size_t)(r0 + (int)threadIdx.x * rs) * nb + b] : 0.f;
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int e = 0; e < NX; ++e) {
            int u = threadIdx.x + 256 * e;
            *(int4 *)(xsh2[buf] + (u / (XB / 16)) * XS + 16 * (u % (XB / 16))) = nx[e];
        }
        if (threadIdx.x < E3_BN) {
            ssh2[buf][threadIdx.x] = ns;
        }
    };
    /* X16: the rows of a block straight to shared memory (cp.async; the
     * registers of fetch spilled), the rows past nt zeros once */
    auto issue = [&](int b, int buf) {
        #pragma unroll
        for (int e = 0; e < NX; ++e) {
            int u = threadIdx.x + 256 * e, jj = u / (XB / 16);
            if (jj < nt) {
                cp_async16(xsh2[buf] + jj * XS + 16 * (u % (XB / 16)),
                           (const int4 *)(xq + ((size_t)(r0 + jj * rs) * k + 128 * b) * (XB / 128)) +
                               u % (XB / 16));
            }
        }
        if ((int)threadIdx.x < nt) {
            cp_async4(&ssh2[buf][threadIdx.x], xs + (size_t)(r0 + (int)threadIdx.x * rs) * nb + b);
        }
        cp_async_commit();
    };
    if (b0 < b1) {
        if constexpr (X16) {
            for (int u = threadIdx.x; u < 2 * E3_BN * (XB / 16); u += 256) {
                int bf = u / (E3_BN * (XB / 16)), v = u % (E3_BN * (XB / 16)), jj = v / (XB / 16);
                if (jj >= nt) {
                    *(int4 *)(xsh2[bf] + jj * XS + 16 * (v % (XB / 16))) = make_int4(0, 0, 0, 0);
                }
            }
            if ((int)threadIdx.x >= nt && threadIdx.x < E3_BN) {
                ssh2[0][threadIdx.x] = ssh2[1][threadIdx.x] = 0.f;
            }
            issue(b0, 0);
        } else {
            fetch(b0);
            store(0);
        }
    }
    for (int b = b0; b < b1; ++b) {
        const int buf = (b - b0) & 1;
        if constexpr (X16) {
#if RC_SM80
            asm volatile("cp.async.wait_group 0;\n" ::);
#endif
        }
        __syncthreads();
        if (b + 1 < b1) {
            if constexpr (X16) {
                issue(b + 1, buf ^ 1);
            } else {
                fetch(b + 1);
            }
        }
        const uint8_t *xsh = xsh2[buf];
        float sc[E3_BN / 8][2];
        #pragma unroll
        for (int q = 0; q < E3_BN / 8; ++q) {
            sc[q][0] = ssh2[buf][8 * q + 2 * c];
            sc[q][1] = ssh2[buf][8 * q + 2 * c + 1];
        }
        const uint32_t *p = (const uint32_t *)wm + ((size_t)(8 * b) * tn + tj) * W;
        #pragma unroll
        for (int pr = 0; pr < 4; ++pr) {
            uint32_t al[4], ah[4];
            #pragma unroll
            for (int h2 = 0; h2 < 2; ++h2) {
                const uint32_t *pt = p + (size_t)(2 * pr + h2) * step;
                uint32_t a0 = __ldg(pt + i0), a1 = __ldg(pt + i1), a2 = __ldg(pt + i2), a3 = __ldg(pt + i3);
                uint32_t A0 = __funnelshift_l(a1, a0, off), A1 = __funnelshift_l(a2, a1, off);
                uint32_t A2 = __funnelshift_l(a3, a2, off);
                e3_lh<K, 0>(A0, A1, A2, al[2 * h2], ah[2 * h2]);           /* output g */
                e3_lh<K, 4>(A0, A1, A2, al[2 * h2 + 1], ah[2 * h2 + 1]);   /* output g + 8 */
            }
            /* the integer of the 32 inputs of the step (exact), then the
             * scale of the block (few registers: two blocks an SM); X16:
             * s x = 256 s xh + s xl, a pass for the high bytes (the scale
             * times 256: exact), then one for the low bytes */
            #pragma unroll
            for (int pl = 0; pl < (X16 ? 2 : 1); ++pl) {
                const float f = X16 && pl == 0 ? 256.f : 1.f;
                #pragma unroll
                for (int q = 0; q < E3_BN / 8; ++q) {
                    const uint8_t *xr = xsh + (8 * q + g) * XS + 32 * pr + 4 * c + 128 * pl;
                    uint32_t bb[2] = {*(const uint32_t *)xr, *(const uint32_t *)(xr + 16)};
                    int il[4] = {0, 0, 0, 0}, ih[4] = {0, 0, 0, 0};
                    e3_mma_u8s8(il, al, bb);
                    e3_mma_u8s8(ih, ah, bb);
                    if constexpr (X16) {
                        facc[q][0] += f * sc[q][0] * (float)(il[0] + 256 * ih[0]);
                        facc[q][1] += f * sc[q][1] * (float)(il[1] + 256 * ih[1]);
                        facc[q][2] += f * sc[q][0] * (float)(il[2] + 256 * ih[2]);
                        facc[q][3] += f * sc[q][1] * (float)(il[3] + 256 * ih[3]);
                    } else {
                        facc[q][0] += sc[q][0] * (float)(il[0] + 256 * ih[0]);
                        facc[q][1] += sc[q][1] * (float)(il[1] + 256 * ih[1]);
                        facc[q][2] += sc[q][0] * (float)(il[2] + 256 * ih[2]);
                        facc[q][3] += sc[q][1] * (float)(il[3] + 256 * ih[3]);
                    }
                }
            }
        }
        if constexpr (!X16) {
            if (b + 1 < b1) {
                store(buf ^ 1);
            }
        }
    }
    #pragma unroll
    for (int q = 0; q < E3_BN / 8; ++q) {
        #pragma unroll
        for (int e = 0; e < 2; ++e) {
            int jj = 8 * q + 2 * c + e;
            if (jj < nt) {
                float *o = part + ((size_t)s * t + r0 + jj * rs) * n + 16 * tj;
                o[g] = E3_KINV * facc[q][e];
                o[g + 8] = E3_KINV * facc[q][2 + e];
            }
        }
    }
}

template <int K, bool X16 = false>
__global__ void __launch_bounds__(256, 2) k_e3_mma(const uint8_t *wm, int k, int n, const int8_t *xq,
                                                const float *xs, int t, int S, float *part)
{
    const int nob = n / 128, j0 = blockIdx.y * E3_BN;
    e3_mma_body<K, X16>(wm, k, n, xq, xs, t, S, part, blockIdx.x % nob, blockIdx.x / nob, j0, 1, min(E3_BN, t - j0));
}

/* The products of a group of the experts on the tensor cores: a block for
 * an output block (of gate, then up: mode 0; of down: mode 1) and a tile of
 * the sort (k_qmoe_sort: at most E3_BN = KG_B pairs of one expert; the
 * tiles after the counts, starts, ptok and pof of the work, their count
 * work[0]), the trellis of its first item (tw; ik: its K, which runs in the
 * kernel of its K). The rows of x and of part are the items: 2 p + m (mode
 * 0) or p (mode 1), ni of them; x in the order of the MMA inputs
 * (e3b_had_in_rows perm). Each tile row of an expert is decoded once for 64
 * pairs, where the GEMV of each item decoded it for each (3.4 s of 5.5 of a
 * prompt of 2048 tokens). */
template <int K, bool X16>
__global__ void __launch_bounds__(256, 2) k_e3g_mma(const gp_rec *r, const int64_t *e, int mode,
                                                 const uint8_t *const *tw, const int *ik, int k, int n,
                                                 const int8_t *xq, const float *xs, int ni, float *part)
{
    const int *work = DP(const int, 18);
    const int tile = blockIdx.y;
    if (tile >= work[0]) {
        return;
    }
    const int E = DI(5), P = DI(3) * DI(4) + DI(3);
    const int *tiles = work + 8 + 2 * (E + 2) + 2 * P;      /* (qmoe_work) */
    const int p0 = tiles[3 * tile + 1], nt = tiles[3 * tile + 2];
    const int nob = n / 128, m = mode == 0 ? (int)blockIdx.x / nob : 0, ob = blockIdx.x % nob;
    const int it = mode == 0 ? 2 * p0 + m : p0;
    const uint8_t *wm = tw[it];
    if (wm == nullptr || ik[it] != K) {
        return;
    }
    e3_mma_body<K, X16>(wm, k, n, xq, xs, ni, 1, part, ob, 0, it, mode == 0 ? 2 : 1, nt);
}

/* The slices of the inputs of a k_e3_mma product: two waves of blocks at
 * least, parts of at most 16M floats. */
static int e3_mma_slices(int64_t t, int k, int n)
{
    int nb = k / 128;
    int64_t base = (int64_t)(n / 128) * ((t + E3_BN - 1) / E3_BN);
    int S = (int)((164 + base - 1) / base);
    int64_t cap = ((int64_t)16 << 20) / ((int64_t)t * n);
    if (S > cap) {
        S = (int)cap;
    }
    return S < 1 ? 1 : S > nb ? nb : S;
}

static int e3_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_E3_TC");
        on = v == NULL || strcmp(v, "0") != 0;
    }
    return on;
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

/* A batch of products of one shape (k inputs, n outputs): item i (blockIdx.y
 * + i0) reads row xr[i] of x (row stride ldx) with the matrix whose trellis
 * is tw[i] (null: no product), suh su[i], svh sv[i]; its xh is row i of xq,
 * xs, xn, its output row i of out (row stride ldo). k_e3b_had_in, then
 * k_e3_gemv with tw, then k_e3b_had_out (GP_KQ_HOT_MOE of MUL1 experts). */
/* A warp for each block of 128 values (e3_fwht128_w): item j = (vector) / nb
 * + i0, block b = (vector) % nb; 4 warps a CUDA block, nv vectors. */
__global__ void __launch_bounds__(128) k_e3b_had_in(const float *x, size_t ldx, const int *xr,
                                                    const uint8_t *const *tw, const uint8_t *const *su, int k,
                                                    int i0, int8_t *xq, float *xs, int *xn, int perm, int64_t nv,
                                                    int x16 = 0)
{
    const int64_t vi = (int64_t)blockIdx.x * 4 + threadIdx.x / 32;
    const int nb = k / 128, lane = threadIdx.x % 32;
    if (vi >= nv) {
        return;
    }
    const int j = (int)(vi / nb) + i0, b = (int)(vi % nb);
    if (tw[j] == nullptr) {
        return;
    }
    const __half *suh = (const __half *)su[j];
    const float *xrow = x + (size_t)xr[j] * ldx + 128 * b;
    float a[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        a[q] = xrow[lane + 32 * q] * __half2float(suh[128 * b + lane + 32 * q]);
    }
    e3_fwht128_w(a);
    float m = e3_warp_max(fmaxf(fmaxf(fabsf(a[0]), fabsf(a[1])), fmaxf(fabsf(a[2]), fabsf(a[3]))));
    /* x16 (k_e3g_mma X16): int16 values to E3_Q16, each the high byte h and
     * the low byte l (both signed: v = 256 h + l), at 256 b of a row of 2 k
     * bytes: the 128 h, then the 128 l */
    const float qm = x16 ? (float)E3_Q16 : 127.f;
    float inv = m > 0.f ? qm / m : 0.f;
    float sum = 0.f;
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        int i = lane + 32 * q, v = __float2int_rn(a[q] * inv);
        v = x16 ? max(-E3_Q16, min(E3_Q16, v)) : max(-127, min(127, v));
        int u = i % 16;     /* perm: the order of the inputs of k_e3_mma, as k_e3_had_in */
        int pos = perm ? (i & ~15) + 4 * ((u & 7) >> 1) + (u & 1) + 2 * (u >> 3) : i;
        if (x16) {
            int h = (v + 128) >> 8;
            xq[2 * ((size_t)j * k + 128 * b) + pos] = (int8_t)h;
            xq[2 * ((size_t)j * k + 128 * b) + 128 + pos] = (int8_t)(v - 256 * h);
        } else {
            xq[(size_t)j * k + 128 * b + pos] = (int8_t)v;
        }
        sum += (float)v;
    }
    sum = e3_warp_sum(sum);        /* (exact: integers, |sum| < 2^23) */
    if (lane == 0) {
        xs[(size_t)j * nb + b] = m / qm;
        xn[(size_t)j * nb + b] = (int)sum;
    }
}

__global__ void __launch_bounds__(128) k_e3b_had_out(const float *part, int S, int ni, int n, const float *xs,
                                                     const int *xn, int nb, const uint8_t *const *tw,
                                                     const uint8_t *const *sv, int i0, float *out, size_t ldo,
                                                     int64_t nv)
{
    const int64_t vi = (int64_t)blockIdx.x * 4 + threadIdx.x / 32;
    const int nob = n / 128, lane = threadIdx.x % 32;
    if (vi >= nv) {
        return;
    }
    const int j = (int)(vi / nob) + i0, ob = (int)(vi % nob);
    if (tw[j] == nullptr) {
        return;
    }
    float c = 0.f;
    for (int b = lane; b < nb; b += 32) {
        c += xs[(size_t)j * nb + b] * (float)xn[(size_t)j * nb + b];
    }
    c = e3_warp_sum(c) * (1024.f * E3_KINV + E3_KBIAS);
    float a[4];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        a[q] = c;
        for (int s = 0; s < S; ++s) {
            a[q] += part[((size_t)s * ni + j) * n + 128 * ob + lane + 32 * q];
        }
    }
    e3_fwht128_w(a);
    const __half *svh = (const __half *)sv[j];
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        out[(size_t)j * ldo + 128 * ob + lane + 32 * q] = a[q] * __half2float(svh[128 * ob + lane + 32 * q]);
    }
}

/* The batch transforms of rows items: a warp for each block of 128 values,
 * a grid of x alone (4096 tokens of Qwen3.8 are 90112 items of gate and up:
 * more than the 65535 of gridDim.y). */
#define E3_ROWS_MAX 65535
static void e3b_had_in_rows(int nb, int rows, cudaStream_t s, const float *x, size_t ldx, const int *xr,
                            const uint8_t *const *tw, const uint8_t *const *su, int k, int8_t *xq, float *xs,
                            int *xn, int perm = 0, int x16 = 0)
{
    const int64_t nv = (int64_t)rows * nb;
    if (nv > 0) {
        k_e3b_had_in<<<(unsigned)((nv + 3) / 4), 128, 0, s>>>(x, ldx, xr, tw, su, k, 0, xq, xs, xn, perm, nv, x16);
    }
}

static void e3b_had_out_rows(int nob, int rows, cudaStream_t s, const float *part, int S, int ni, int n,
                             const float *xs, const int *xn, int nb, const uint8_t *const *tw,
                             const uint8_t *const *sv, float *out, size_t ldo)
{
    const int64_t nv = (int64_t)rows * nob;
    if (nv > 0) {
        k_e3b_had_out<<<(unsigned)((nv + 3) / 4), 128, 0, s>>>(part, S, ni, n, xs, xn, nb, tw, sv, 0, out, ldo, nv);
    }
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

/* The bytes of the scratch of a product: xq, xs, xn, the parts (of
 * k_e3_gemv or of k_e3_mma, the more). */
static size_t e3_ws_bytes(int64_t t, int k, int n, int x16 = 0)
{
    size_t a = ((size_t)(x16 ? 2 : 1) * t * k + 255) & ~(size_t)255;
    size_t b = ((size_t)t * (k / 128) * 8 + 255) & ~(size_t)255;
    int S = e3_slices(t, k, n);
    if (t >= E3_TC_MIN && e3_mma_slices(t, k, n) > S) {
        S = e3_mma_slices(t, k, n);
    }
    return a + b + (size_t)S * t * n * 4 + 256;
}

/* y (t x n, row stride ldo) = x W for a KQ_E3 matrix w (K bits) on the rows
 * of x (t x k, row stride ldx). ws: e3_ws_bytes(t, k, n). */
static int e3_linear_launch(const uint8_t *w, int K, int n, int k, const float *x, size_t ldx, int64_t t,
                            float *out, size_t ldo, uint8_t *ws, size_t ws_n, float cap, cudaStream_t s,
                            int tc = 0, int x16 = 0)
{
    if (K < 1 || K > 8 || k % 128 || n % 128 || ws == NULL || ws_n < e3_ws_bytes(t, k, n, x16)) {
        return -1;
    }
    /* tc: the tensor cores may run (sm_80 and later, not the Pascal mode);
     * x16 (a test of k_e3g_mma's int16 x: rc_gpu_e3_linear_x16): int16 x on
     * the tensor cores */
    const bool mma = tc && e3_tc_on() && t >= E3_TC_MIN;
    x16 = x16 && mma;
    const int nb = k / 128, S = mma ? e3_mma_slices(t, k, n) : e3_slices(t, k, n);
    int8_t *xq = (int8_t *)ws;
    float *xs = (float *)(ws + (((size_t)(x16 ? 2 : 1) * t * k + 255) & ~(size_t)255));
    int *xn = (int *)(xs + (size_t)t * nb);
    float *part = (float *)((uint8_t *)xs + (((size_t)t * nb * 8 + 255) & ~(size_t)255));
    const __half *suh = (const __half *)(w + (size_t)k * n * K / 8), *svh = suh + k;
    k_e3_had_in<<<dim3((unsigned)nb, (unsigned)t), 128, 0, s>>>(x, ldx, suh, k, xq, xs, xn, mma, x16);
    if (mma) {
        dim3 gm((unsigned)((n / 128) * S), (unsigned)((t + E3_BN - 1) / E3_BN));
#define E3_MCASE(KK) case KK: \
            if (x16) { \
                k_e3_mma<KK, true><<<gm, 256, 0, s>>>(w, k, n, xq, xs, (int)t, S, part); \
            } else { \
                k_e3_mma<KK><<<gm, 256, 0, s>>>(w, k, n, xq, xs, (int)t, S, part); \
            } \
            break;
        switch (K) {
        E3_MCASE(1) E3_MCASE(2) E3_MCASE(3) E3_MCASE(4) E3_MCASE(5) E3_MCASE(6) E3_MCASE(7) E3_MCASE(8)
        }
#undef E3_MCASE
        k_e3_had_out<<<dim3((unsigned)(n / 128), (unsigned)t), 128, 0, s>>>(part, S, (int)t, n, xs, xn, nb,
                                                                           svh, out, ldo, cap);
        return 0;
    }
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

static int e3_lib_linear(const void *w, int K, int n, int k, const float *x, int t, float *out, float cap,
                         int x16)
{
    size_t need = e3_ws_bytes(t, k, n, x16);
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
                         cap, gg_stream, gg_caps.tc, x16) != 0) {
        return -1;
    }
    return cudaGetLastError() == cudaSuccess ? 0 : -1;
}

extern "C" int rc_gpu_e3_linear(const void *w, int K, int n, int k, const float *x, int t, float *out,
                                float cap)
{
    return e3_lib_linear(w, K, n, k, x, t, out, cap, 0);
}

/* The same with int16 x on the tensor cores (t >= E3_TC_MIN): a check of the
 * int16 form of k_e3g_mma (scripts/check_exl3.py --gpu). */
extern "C" int rc_gpu_e3_linear_x16(const void *w, int K, int n, int k, const float *x, int t, float *out)
{
    return e3_lib_linear(w, K, n, k, x, t, out, 0.f, 1);
}

/* ---------- GP_KQ_HOT_MOE with MUL1 experts ----------
 *
 * The hot experts of a step or a small group (kqh_slot, kqh_wslot of
 * kernels/recurrent/deltanet.cu) and the shared expert, as batches of
 * products: item 2 j + part (gate, up) of pair j and item j of down.
 * k_e3h_setup finds the matrix of each item (the slot table and the pool;
 * a pair with no slot: none); then the transforms, the GEMV (the routed
 * items and the shared items: K of their own), the transforms of the
 * outputs into the act of the record (operand 7), k_kqh_act, the same for
 * down into de (operand 9), and k_kqh_sum. The pairs of the copy in host
 * memory (zc) are not taken here. */
__global__ void k_e3h_setup(const gp_rec *r, const int64_t *e, int P, int Kr, int Kd, int Ks,
                            const uint8_t **tw, const uint8_t **su, const uint8_t **sv, int *xr)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= P) {
        return;
    }
    int k = DI(11), inner = DI(12), hidden = DI(13), tk = DI(16) * k;
    int slot = kqh_slot(r, e, j), shared = j >= tk;
    int tok = shared ? j - tk : j / k;
    const size_t pgu = (size_t)hidden * inner * Kr / 8 + 2 * ((size_t)hidden + inner);
    const size_t pdn = (size_t)hidden * inner * Kd / 8 + 2 * ((size_t)hidden + inner);
    for (int part = 0; part < 3; ++part) {
        int it = part < 2 ? 2 * j + part : 2 * P + j;
        const uint8_t *w = NULL;
        if (slot >= 0) {
            w = shared ? DP(const uint8_t, 17 + part)
                       : kqh_wslot(r, e, slot, part, part < 2 ? pgu : pdn);
        }
        int K = shared ? Ks : (part < 2 ? Kr : Kd);
        int kin = part < 2 ? hidden : inner;
        tw[it] = w;
        su[it] = w != NULL ? w + (size_t)hidden * inner * K / 8 : NULL;
        sv[it] = w != NULL ? su[it] + 2 * (size_t)kin : NULL;
        xr[it] = part < 2 ? tok : j;
    }
}

/* The slices of a batch of ni items of k inputs and n outputs. */
static int e3b_slices(int ni, int k, int n)
{
    int nb = k / 128;
    int64_t base = (int64_t)(n / 128) * ni;
    int S = (int)((320 + base - 1) / base);
    int lo = (nb + E3_SLICE_MAX - 1) / E3_SLICE_MAX;
    S = S < lo ? lo : S;
    return S < 1 ? 1 : S > nb ? nb : S;
}

static size_t e3_al(size_t n)
{
    return (n + 255) & ~(size_t)255;
}

/* The products of a group of MUL1 experts on the tensor cores (k_e3g_mma)
 * take x as int16 (the high and the low bytes, two MMAs more for each step)
 * in place of int8; RECLANK_GPU_E3_X16=0: int8. Qwen3.8-Flash-Next 5.05 bpw,
 * a prompt of 1024 tokens and 128 steps: KL against the CPU program (int16
 * x everywhere) 2.6e-3, with int8 9.9e-3; pp8192 863 tok/s against 934
 * (pp2048 the same: the CPU's part sets the pace). The scratch of a group
 * of 4096 tokens 0.23 GB more. */
static int e3_x16_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_E3_X16");
        on = v == NULL || strcmp(v, "0") != 0;
    }
    return on;
}

/* The bytes of the scratch of a GP_KQ_HOT_MOE of P pairs; x16: of a
 * GP_KQ_GROUP_MOE with int16 x (the int16 rows of gate and up, which the
 * rows of down take after them). */
static size_t e3h_ws_bytes(int64_t P, int hidden, int inner, int x16 = 0)
{
    size_t a = e3_al((size_t)3 * P * 8) * 3 + 2 * e3_al((size_t)3 * P * 4);
    a += e3_al((size_t)(x16 ? 2 : 1) * 2 * P * hidden) + 2 * e3_al((size_t)2 * P * (hidden / 128) * 4);
    a += e3_al((size_t)e3b_slices((int)(2 * P), hidden, inner) * 2 * P * inner * 4);
    a += e3_al((size_t)P * inner) + 2 * e3_al((size_t)P * (inner / 128) * 4);
    a += e3_al((size_t)e3b_slices((int)P, inner, hidden) * P * hidden * 4);
    return a + 256;
}

/* A GEMV of items [i0, i0 + n) of a batch (ni items in all) of K bits. */
static void e3b_gemv(int K, const uint8_t *const *tw, int k, int n, const int8_t *xq, const float *xs, int ni,
                     int S, float *part, int i0, int cnt, cudaStream_t s, const int *ik = nullptr)
{
    if (cnt <= 0) {
        return;
    }
    size_t sm = (size_t)E3_SLICE_MAX * (128 + 4);
    for (int c = 0; c < cnt; c += E3_ROWS_MAX) {        /* (gridDim.y, as e3b_had_in_rows) */
        dim3 g((unsigned)((n / 128) * S), (unsigned)min(E3_ROWS_MAX, cnt - c));
#define E3_BCASE(KK) case KK: \
            k_e3_gemv<KK, 1><<<g, 256, sm, s>>>(NULL, k, n, xq, xs, ni, S, part, tw, i0 + c, ik); break;
        switch (K) {
        E3_BCASE(1) E3_BCASE(2) E3_BCASE(3) E3_BCASE(4) E3_BCASE(5) E3_BCASE(6) E3_BCASE(7) E3_BCASE(8)
        }
#undef E3_BCASE
    }
}

/* The launches of a GP_KQ_HOT_MOE of MUL1 experts (r its record, dr on the
 * device; x, act, act2, de, out the device addresses of operands 0, 7, 8,
 * 9, 10). ws: e3h_ws_bytes. */
static int e3_hot_moe_launch(const gp_rec *dr, const int64_t *denv, int t, int k, int hidden, int inner,
                             int Kr, int Kd, int Ks, int shared, const float *x, float *act, const float *act2,
                             float *de, int sum, int rot, uint8_t *ws, size_t ws_n, cudaStream_t s)
{
    const int P = t * k + (shared ? t : 0), tk = t * k;
    if (hidden % 128 || inner % 128 || ws == NULL || ws_n < e3h_ws_bytes(P, hidden, inner)) {
        return -1;
    }
    const int nh = hidden / 128, ni = inner / 128;
    uint8_t *p = ws;
    auto take = [&](size_t n) { uint8_t *q = p; p += e3_al(n); return q; };
    const uint8_t **tw = (const uint8_t **)take((size_t)3 * P * 8);
    const uint8_t **su = (const uint8_t **)take((size_t)3 * P * 8);
    const uint8_t **sv = (const uint8_t **)take((size_t)3 * P * 8);
    int *xr = (int *)take((size_t)3 * P * 4);
    take((size_t)3 * P * 4);                    /* (the K of each item: the group form) */
    int8_t *xq = (int8_t *)take((size_t)2 * P * hidden);
    float *xs = (float *)take((size_t)2 * P * nh * 4);
    int *xn = (int *)take((size_t)2 * P * nh * 4);
    const int S1 = e3b_slices(2 * P, hidden, inner), S2 = e3b_slices(P, inner, hidden);
    float *part = (float *)take((size_t)S1 * 2 * P * inner * 4);
    int8_t *aq = (int8_t *)take((size_t)P * inner);
    float *as = (float *)take((size_t)P * ni * 4);
    int *an = (int *)take((size_t)P * ni * 4);
    float *pdn = (float *)take((size_t)S2 * P * hidden * 4);
    k_e3h_setup<<<(unsigned)((P + 127) / 128), 128, 0, s>>>(dr, denv, P, Kr, Kd, Ks, tw, su, sv, xr);
    /* gate and up: items 0 .. 2 P - 1 */
    e3b_had_in_rows(nh, 2 * P, s, x, (size_t)hidden, xr, tw, su, hidden, xq, xs, xn);
    e3b_gemv(Kr, tw, hidden, inner, xq, xs, 2 * P, S1, part, 0, 2 * tk, s);
    e3b_gemv(Ks, tw, hidden, inner, xq, xs, 2 * P, S1, part, 2 * tk, 2 * (P - tk), s);
    e3b_had_out_rows(ni, 2 * P, s, part, S1, 2 * P, inner, xs, xn, nh, tw, sv, act, (size_t)inner);
    k_kqh_act<<<(unsigned)P, 256, 0, s>>>(dr, denv, rot);
    /* down: items 2 P .. 3 P - 1 of the tables, rows 0 .. P - 1 of its own */
    e3b_had_in_rows(ni, P, s, act2, (size_t)inner, xr + 2 * P, tw + 2 * P, su + 2 * P, inner, aq, as, an);
    e3b_gemv(Kd, tw + 2 * P, inner, hidden, aq, as, P, S2, pdn, 0, tk, s);
    e3b_gemv(Ks, tw + 2 * P, inner, hidden, aq, as, P, S2, pdn, tk, P - tk, s);
    e3b_had_out_rows(nh, P, s, pdn, S2, P, hidden, as, an, ni, tw + 2 * P, sv + 2 * P, de, (size_t)hidden);
    if (sum) {
        k_kqh_sum<<<dim3((unsigned)((hidden + 255) / 256), (unsigned)t), 256, 0, s>>>(dr, denv);
    }
    return 0;
}

/* ---------- GP_KQ_GROUP_MOE with MUL1 experts ----------
 *
 * After k_qmoe_sort (the pairs of each expert together: row p of the sort
 * is token ptok[p] of expert x, start[x] <= p < start[x + 1]; x = E the
 * shared expert): the batches of products on the rows of the sort, so the
 * rows of an expert run together (its tiles from L2). Phase 0: the gate and
 * up of row p into act (operand 19, row p: gate, then up); the runner then
 * runs k_qmoe_act (act2, operand 20); phase 1: down of row p into de
 * (operand 21), and k_qmoe_sum adds them up. A row past the sorted rows
 * (work[1]) has no matrix. */
__global__ void k_e3g_setup(const gp_rec *r, const int64_t *e, int P, int Kr, int Kd, int Ks,
                            const uint8_t **tw, const uint8_t **su, const uint8_t **sv, int *xr, int *ik)
{
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= P) {
        return;
    }
    int E = DI(5), hidden = DI(6), inner = DI(7);
    const int *work = DP(const int, 18);
    const int *start = work + 8 + E + 2, *ptok = start + E + 2;
    int x = -1;
    if (p < work[1]) {
        int lo = 0, hi = E;                     /* the expert x with start[x] <= p < start[x + 1] */
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (start[mid] <= p) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        x = lo;
    }
    for (int part = 0; part < 3; ++part) {
        int it = part < 2 ? 2 * p + part : 2 * P + p;
        const uint8_t *w = NULL;
        if (x >= 0) {
            w = x < E ? (const uint8_t *)(intptr_t)DP(const int64_t, 8 + part)[x] : DP(const uint8_t, 13 + part);
        }
        int K = x >= E ? Ks : (part < 2 ? Kr : Kd);
        int kin = part < 2 ? hidden : inner;
        tw[it] = w;
        su[it] = w != NULL ? w + (size_t)hidden * inner * K / 8 : NULL;
        sv[it] = w != NULL ? su[it] + 2 * (size_t)kin : NULL;
        xr[it] = part < 2 ? (x >= 0 ? ptok[p] : 0) : p;
        ik[it] = K;
    }
}

/* Phase 0 (setup, gate and up into act) or 1 (down of act2 into de) of a
 * GP_KQ_GROUP_MOE of P pairs (rows of the sort). ws: e3h_ws_bytes(P, ...). */
static void e3g_mma(int K, int mode, const gp_rec *dr, const int64_t *denv, const uint8_t *const *tw, const int *ik,
                    int k, int n, const int8_t *xq, const float *xs, int ni, float *part, unsigned tiles,
                    cudaStream_t s, int x16)
{
    dim3 g((unsigned)((mode == 0 ? 2 : 1) * (n / 128)), tiles);
#define E3_GCASE(KK) case KK: \
        if (x16) { \
            k_e3g_mma<KK, true><<<g, 256, 0, s>>>(dr, denv, mode, tw, ik, k, n, xq, xs, ni, part); \
        } else { \
            k_e3g_mma<KK, false><<<g, 256, 0, s>>>(dr, denv, mode, tw, ik, k, n, xq, xs, ni, part); \
        } \
        break;
    switch (K) {
    E3_GCASE(1) E3_GCASE(2) E3_GCASE(3) E3_GCASE(4) E3_GCASE(5) E3_GCASE(6) E3_GCASE(7) E3_GCASE(8)
    }
#undef E3_GCASE
}

/* tiles: the most tiles of the sort (the grid of k_e3g_mma); tc: the tensor
 * cores may run (as e3_linear_launch): the products of each tile of an
 * expert (k_e3g_mma), else the GEMV of each item. */
static int e3_group_moe_launch(int phase, const gp_rec *dr, const int64_t *denv, int P, int hidden, int inner,
                               int Kr, int Kd, int Ks, const float *x, float *act, const float *act2, float *de,
                               uint8_t *ws, size_t ws_n, cudaStream_t s, unsigned tiles = 0, int tc = 0)
{
    const int mma = tc && e3_tc_on() && tiles > 0, x16 = mma && e3_x16_on();
    if (hidden % 128 || inner % 128 || ws == NULL || ws_n < e3h_ws_bytes(P, hidden, inner, x16)) {
        return -1;
    }
    const int nh = hidden / 128, ni = inner / 128;
    uint8_t *p = ws;
    auto take = [&](size_t n) { uint8_t *q = p; p += e3_al(n); return q; };
    const uint8_t **tw = (const uint8_t **)take((size_t)3 * P * 8);
    const uint8_t **su = (const uint8_t **)take((size_t)3 * P * 8);
    const uint8_t **sv = (const uint8_t **)take((size_t)3 * P * 8);
    int *xr = (int *)take((size_t)3 * P * 4);
    int *ik = (int *)take((size_t)3 * P * 4);
    int8_t *xq = (int8_t *)take((size_t)(x16 ? 2 : 1) * 2 * P * hidden);
    float *xs = (float *)take((size_t)2 * P * nh * 4);
    int *xn = (int *)take((size_t)2 * P * nh * 4);
    const int S1 = mma ? 1 : e3b_slices(2 * P, hidden, inner), S2 = mma ? 1 : e3b_slices(P, inner, hidden);
    float *part = (float *)take((size_t)e3b_slices(2 * P, hidden, inner) * 2 * P * inner * 4);
    int8_t *aq = (int8_t *)take((size_t)P * inner);
    float *as = (float *)take((size_t)P * ni * 4);
    int *an = (int *)take((size_t)P * ni * 4);
    float *pdn = (float *)take((size_t)e3b_slices(P, inner, hidden) * P * hidden * 4);
    if (x16) {
        aq = xq;        /* (2 P inner of the 4 P hidden bytes: those of gate and up are done) */
    }
    if (phase == 0) {
        k_e3g_setup<<<(unsigned)((P + 127) / 128), 128, 0, s>>>(dr, denv, P, Kr, Kd, Ks, tw, su, sv, xr, ik);
        e3b_had_in_rows(nh, 2 * P, s, x, (size_t)hidden, xr, tw, su, hidden, xq, xs, xn, mma, x16);
        if (mma) {
            e3g_mma(Kr, 0, dr, denv, tw, ik, hidden, inner, xq, xs, 2 * P, part, tiles, s, x16);
            if (Ks != Kr) {
                e3g_mma(Ks, 0, dr, denv, tw, ik, hidden, inner, xq, xs, 2 * P, part, tiles, s, x16);
            }
            e3b_had_out_rows(ni, 2 * P, s, part, 1, 2 * P, inner, xs, xn, nh, tw, sv, act, (size_t)inner);
            return 0;
        }
        /* the shared rows end the sort (their place is on the device): the
         * GEMV of each K over all the rows, each item in that of its K */
        e3b_gemv(Kr, tw, hidden, inner, xq, xs, 2 * P, S1, part, 0, 2 * P, s, ik);
        if (Ks != Kr) {
            e3b_gemv(Ks, tw, hidden, inner, xq, xs, 2 * P, S1, part, 0, 2 * P, s, ik);
        }
        e3b_had_out_rows(ni, 2 * P, s, part, S1, 2 * P, inner, xs, xn, nh, tw, sv, act, (size_t)inner);
        return 0;
    }
    e3b_had_in_rows(ni, P, s, act2, (size_t)inner, xr + 2 * P, tw + 2 * P, su + 2 * P, inner, aq, as, an, mma,
                    x16);
    if (mma) {
        e3g_mma(Kd, 1, dr, denv, tw + 2 * P, ik + 2 * P, inner, hidden, aq, as, P, pdn, tiles, s, x16);
        if (Ks != Kd) {
            e3g_mma(Ks, 1, dr, denv, tw + 2 * P, ik + 2 * P, inner, hidden, aq, as, P, pdn, tiles, s, x16);
        }
        e3b_had_out_rows(nh, P, s, pdn, 1, P, hidden, as, an, ni, tw + 2 * P, sv + 2 * P, de, (size_t)hidden);
        return 0;
    }
    e3b_gemv(Kd, tw + 2 * P, inner, hidden, aq, as, P, S2, pdn, 0, P, s, ik + 2 * P);
    if (Ks != Kd) {
        e3b_gemv(Ks, tw + 2 * P, inner, hidden, aq, as, P, S2, pdn, 0, P, s, ik + 2 * P);
    }
    e3b_had_out_rows(nh, P, s, pdn, S2, P, hidden, as, an, ni, tw + 2 * P, sv + 2 * P, de, (size_t)hidden);
    return 0;
}
