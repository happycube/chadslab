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

/* The value nibble - 8 of the 4 bits of q at bit s (s <= 16) with no shift:
 * the nibble masked in place, with the bits of 2^23, is the float
 * 2^23 + n 2^s (exact while n 2^s < 2^23), and one FMA by the power of two
 * 2^-s less 2^(23-s) + 8 gives n - 8 exactly (one rounding of an exact
 * value). A LOP3 and an FFMA, where i4f takes a shift (half the rate on
 * Maxwell and Pascal), a LOP3 and an FADD: the same values, so the same bits
 * (MAXWELL_PLAN.md, section 3.2). */
template <int S>
__device__ __forceinline__ float i4m(uint32_t q)
{
    /* (one LOP3 with the mask and the bits in registers, below sm_70: the hot experts and the matrices of a
     * GP_INT4_MULTI4 slower on the M4000, 287 against 243 us for q, k, v) */
    const float f = __uint_as_float((q & (15u << S)) | 0x4B000000u);
    return fmaf(f, 1.f / (float)(1u << S), -(8388608.f / (float)(1u << S) + 8.f));
}

/* The 8 products of the 4 bytes q (lane sub of a block) with x, in one
 * order: the first product, then an FMA for each next one, written out, so
 * the bits do not hang on the products that nvcc fuses (a sum of products
 * lets it fuse either one, and the choice moved with the decode).
 * int4_row and k_mt_int4_rows use it, so a token of a group gets the bits
 * of a step. The high nibbles of bits 20, 24 and 28 take one shift of q by
 * 12 (to bits 8, 12 and 16) and the decode of i4m. */
__device__ __forceinline__ float int4_part(uint32_t q, float4 xl, float4 xh)
{
    const uint32_t h = q >> 12;
    float s = i4m<0>(q) * xl.x;
    s = fmaf(i4m<8>(q), xl.y, s);
    s = fmaf(i4m<16>(q), xl.z, s);
    s = fmaf(i4m<12>(h), xl.w, s);
    s = fmaf(i4m<4>(q), xh.x, s);
    s = fmaf(i4m<12>(q), xh.y, s);
    s = fmaf(i4m<8>(h), xh.z, s);
    return fmaf(i4m<16>(h), xh.w, s);
}

/* The 4 bytes of codes of lane sub of block b of an int4 row wr of cols values, and the float16 d of the
 * block: Q4_0 as stored (18-byte blocks), or with Q4S the codes of the row (16 bytes a block) and then the d
 * of each block, the same bytes moved (q4s_pack: the dense matrices of a GPU without dp4a; a lane's 4 bytes
 * at a multiple of 4, the 8 blocks of a warp's pass 128 bytes in a run). The same values, so the same bits. */
template <bool Q4S>
__device__ __forceinline__ uint32_t int4_blk(const uint8_t *wr, int cols, int b, int sub, float *d)
{
    if (Q4S) {
        *d = __half2float(((const __half *)(wr + cols / 2))[b]);
        return *(const uint32_t *)(wr + 16 * b + 4 * sub);
    }
    const uint8_t *blk = wr + (size_t)b * 18;
    *d = __half2float(__ushort_as_half(*(const uint16_t *)blk));
    const uint16_t *qp = (const uint16_t *)(blk + 2 + 4 * sub);
    return (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
}

/* int4_blk's loads alone: the codes, and the bits of d (int4_rows_p converts them where it uses them) */
template <bool Q4S>
__device__ __forceinline__ uint32_t int4_blk_raw(const uint8_t *wr, int cols, int b, int sub, uint16_t *dh)
{
    if (Q4S) {
        *dh = ((const uint16_t *)(wr + cols / 2))[b];
        return *(const uint32_t *)(wr + 16 * b + 4 * sub);
    }
    const uint8_t *blk = wr + (size_t)b * 18;
    *dh = *(const uint16_t *)blk;
    const uint16_t *qp = (const uint16_t *)(blk + 2 + 4 * sub);
    return (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
}

template <bool Q4S = false>
__device__ __forceinline__ float int4_row(const uint8_t *wr, const float *x, int cols)
{
    int lane = threadIdx.x % 32;
    int sub = lane & 3;
    int blocks = cols / 32;
    float sum = 0.f;
    for (int b = lane >> 2; b < blocks; b += 8) {
        float d;
        uint32_t q = int4_blk<Q4S>(wr, cols, b, sub, &d);
        float4 xl = *(const float4 *)(x + b * 32 + 4 * sub);
        float4 xh = *(const float4 *)(x + b * 32 + 16 + 4 * sub);
        float acc = int4_part(q, xl, xh);
        sum += d * acc;
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    return sum;
}

/* int4_row of R rows (wr[r]) at once, the loads of the next pass before the products of this one: two passes
 * of each row in flight, the x of a pass read once for the R rows. int4_row waits for the loads of each pass,
 * and for Q4S ptxas put the loads of d after the products' x (the hot experts on the Quadro M4000: two
 * latencies a pass). The same lanes and sums as int4_row: the same bits. (x rotated as int4_stage_x: no
 * faster here.) */
template <bool Q4S, int R>
__device__ __forceinline__ void int4_rows_p(const uint8_t *const *wr, const float *x, int cols, float *sum)
{
    const int lane = threadIdx.x % 32, sub = lane & 3, blocks = cols / 32;
    int b = lane >> 2;
    uint32_t q[R];
    uint16_t dh[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        sum[r] = 0.f;
        dh[r] = 0;
        q[r] = b < blocks ? int4_blk_raw<Q4S>(wr[r], cols, b, sub, &dh[r]) : 0u;
    }
    for (; b < blocks; b += 8) {
        const bool more = b + 8 < blocks;
        uint32_t qn[R];
        uint16_t dn[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            dn[r] = 0;
            qn[r] = more ? int4_blk_raw<Q4S>(wr[r], cols, b + 8, sub, &dn[r]) : 0u;
        }
        const float4 xl = *(const float4 *)(x + b * 32 + 4 * sub);
        const float4 xh = *(const float4 *)(x + b * 32 + 16 + 4 * sub);
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float acc = int4_part(q[r], xl, xh);
            sum[r] += __half2float(__ushort_as_half(dh[r])) * acc;
            q[r] = qn[r];
            dh[r] = dn[r];
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        for (int o = 16; o > 0; o >>= 1) {
            sum[r] += __shfl_xor_sync(RC_FULL_MASK, sum[r], o);
        }
    }
}

#define ROWS_PER_BLOCK 8


/* One warp for each row of an int4 matrix. The arguments give the operands
 * of x, w, out, rows, and cols: 0, 1, 3, 4, 5 for GP_INT4_LINEAR, and 3, 4,
 * 6, 7, 8 for the matrix of GP_GELU_MUL_INT4. The kernel reads the float16
 * scale of each block, not the float32 scales. reclank/backends/cuda/runtime.py checks that
 * the two scales are equal. */
template <bool Q4S = false>
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
    float v = int4_row<Q4S>(wr, DP(const float, xk), cols);
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

/* ---- the int4 rows of a step on a GPU without dp4a (Maxwell): int4_row's lanes, sums and order (the same
 * bits), the data by other paths. int4_row read 4 bytes of a block a lane as two 2-byte loads and x from
 * global memory, 32 bytes of x for each 4.5 of weights (the Quadro M4000: ~50 GB/s). Here the 8 blocks of a
 * pass of a warp (144 bytes; a row of an even count of blocks starts at a multiple of 4) load as 36 words,
 * lane l word l and lanes 0-3 also 32 + l, and go to the lanes by shuffles; x is staged once a block in
 * shared memory, float4 u of block b at 8 b + (u + 4 b) % 8 (the lanes of a quarter warp, blocks b and
 * b + 1 and subs 0-3, read 8 banks); a grid of a few blocks an SM loops over the rows. The M4000, a
 * 4096 x 2816 matrix: 141 -> 104 us (46 -> 62 GB/s); 2112 x 2816 81 -> 62. The matrices of a GP_INT4_MULTI4
 * take a launch each (a kernel that picked the matrix of each row: q, k, v 243 us, against 194 for the same
 * rows of one). ---- */
#define I4S_SMEM_MAX (48 * 1024)

__device__ __forceinline__ void int4_stage_x(const float *x, int cols, float *xs)
{
    for (int i = threadIdx.x; i < cols / 4; i += blockDim.x) {
        const int b = i / 8, u = i % 8;
        ((float4 *)xs)[8 * b + (u + 4 * b) % 8] = ((const float4 *)x)[i];
    }
}

/* word i of the 36 of a pass (w0: lane l's word l, w1: lanes 0-3's 32 + l) */
__device__ __forceinline__ uint32_t i4s_word(uint32_t w0, uint32_t w1, int i)
{
    const uint32_t a = __shfl_sync(RC_FULL_MASK, w0, i & 31), b = __shfl_sync(RC_FULL_MASK, w1, i & 31);
    return i < 32 ? a : b;
}

/* int4_row of a row (cols / 32 blocks, an even count; wr at a multiple of 4) with the staged x: the passes
 * of 8 blocks, then the last of fewer (its words and blocks checked; a check in each pass took 134 us for
 * 4096 x 2816, not 96) */
__device__ __forceinline__ float i4s_pass(uint32_t w0, uint32_t w1, const float *xs, int b, int iq, int sq,
                                          int id, int sd, int sub)
{
    const uint32_t q = __funnelshift_r(i4s_word(w0, w1, iq), i4s_word(w0, w1, iq + 1 < 36 ? iq + 1 : 35), sq);
    const float d = __half2float(__ushort_as_half((uint16_t)(i4s_word(w0, w1, id) >> sd)));
    const float4 *x4 = (const float4 *)xs + 8 * b;
    return d * int4_part(q, x4[(sub + 4 * b) % 8], x4[(4 + sub + 4 * b) % 8]);
}

__device__ __forceinline__ float int4_row_s(const uint8_t *wr, const float *xs, int cols)
{
    const int lane = threadIdx.x % 32, sub = lane & 3, bl = lane >> 2, nb = cols / 32;
    const int o = 18 * bl + 2 + 4 * sub, iq = o >> 2, sq = 8 * (o & 3);
    const int id = (18 * bl) >> 2, sd = 8 * ((18 * bl) & 3);
    const uint32_t *wp = (const uint32_t *)wr;
    float sum = 0.f;
    int b0 = 0;
    for (; b0 + 8 <= nb; b0 += 8, wp += 36) {
        const uint32_t w0 = wp[lane], w1 = lane < 4 ? wp[32 + lane] : 0u;
        sum += i4s_pass(w0, w1, xs, b0 + bl, iq, sq, id, sd, sub);
    }
    if (b0 < nb) {
        const int nw = 18 * (nb - b0) / 4;          /* (nb even: whole words) */
        const uint32_t w0 = lane < nw ? wp[lane] : 0u, w1 = lane < 4 && 32 + lane < nw ? wp[32 + lane] : 0u;
        const float v = i4s_pass(w0, w1, xs, b0 + bl < nb ? b0 + bl : nb - 1, iq, sq, id, sd, sub);
        if (b0 + bl < nb) {
            sum += v;
        }
    }
    for (int k = 16; k > 0; k >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, k);
    }
    return sum;
}

/* int4_row of two Q4S rows with the staged x: the loads in place of the shuffles (int4_blk), the x of a pass
 * read once for both, the loads of 2 passes first, then the sums of each row in its order (the same bits). The
 * M4000, 4096 x 2816: one row 4 passes 81 us, two rows 2 passes 72 (4: 76); 2112 x 2816 48 -> 45 */
__device__ __forceinline__ void int4_rows_sq(const uint8_t *w0, const uint8_t *w1, const float *xs, int cols,
                                             float *s0, float *s1)
{
    const int lane = threadIdx.x % 32, sub = lane & 3, nb = cols / 32;
    float a = 0.f, c = 0.f;
    int b = lane >> 2;
    for (; b + 8 < nb; b += 16) {
        float d0[2], d1[2];
        uint32_t q0[2], q1[2];
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            q0[u] = int4_blk<true>(w0, cols, b + 8 * u, sub, &d0[u]);
            q1[u] = int4_blk<true>(w1, cols, b + 8 * u, sub, &d1[u]);
        }
        #pragma unroll
        for (int u = 0; u < 2; ++u) {
            const int bb = b + 8 * u;
            const float4 *x4 = (const float4 *)xs + 8 * bb;
            const float4 xl = x4[(sub + 4 * bb) % 8], xh = x4[(4 + sub + 4 * bb) % 8];
            a += d0[u] * int4_part(q0[u], xl, xh);
            c += d1[u] * int4_part(q1[u], xl, xh);
        }
    }
    if (b < nb) {
        float d0, d1;
        const uint32_t q0 = int4_blk<true>(w0, cols, b, sub, &d0), q1 = int4_blk<true>(w1, cols, b, sub, &d1);
        const float4 *x4 = (const float4 *)xs + 8 * b;
        const float4 xl = x4[(sub + 4 * b) % 8], xh = x4[(4 + sub + 4 * b) % 8];
        a += d0 * int4_part(q0, xl, xh);
        c += d1 * int4_part(q1, xl, xh);
    }
    for (int k = 16; k > 0; k >>= 1) {
        a += __shfl_xor_sync(RC_FULL_MASK, a, k);
        c += __shfl_xor_sync(RC_FULL_MASK, c, k);
    }
    *s0 = a;
    *s1 = c;
}

/* k_int4_linear with int4_row_s, or with Q4S two rows a warp (int4_rows_sq) (operands as there) */
template <bool Q4S = false>
__global__ void __launch_bounds__(256) k_int4_linear_s(const gp_rec *r, const int64_t *e, int xk, int wk, int ok,
                                                       int rk, int ck)
{
    extern __shared__ __align__(16) float i4xs[];
    const int rows = DI(rk), cols = DI(ck), nw = blockDim.x / 32;
    /* (the operands once: the stores to out could alias the record, so nvcc read it again for each row) */
    const uint8_t *w = DP(const uint8_t, wk);
    float *out = DP(float, ok);
    int4_stage_x(DP(const float, xk), cols, i4xs);
    __syncthreads();
    const size_t rb = (size_t)(cols / 32) * 18;
    if (Q4S) {
        for (int row = 2 * (blockIdx.x * nw + threadIdx.x / 32); row < rows; row += 2 * gridDim.x * nw) {
            float v0, v1;
            int4_rows_sq(w + (size_t)row * rb, w + (size_t)min(row + 1, rows - 1) * rb, i4xs, cols, &v0, &v1);
            if (threadIdx.x % 32 == 0) {
                out[row] = v0;
                if (row + 1 < rows) {
                    out[row + 1] = v1;
                }
            }
        }
        return;
    }
    for (int row = blockIdx.x * nw + threadIdx.x / 32; row < rows; row += gridDim.x * nw) {
        const float v = int4_row_s(w + (size_t)row * rb, i4xs, cols);
        if (threadIdx.x % 32 == 0) {
            out[row] = v;
        }
    }
}

/* q4s_pack: the rows of an int4 matrix in place into Q4S (int4_blk): a warp a row (rb bytes, a multiple of 4
 * at a multiple of 4: an even count of blocks), the row in shared memory, the words of Q4S from its halves.
 * (A block a row of bytes took ~100 us for an expert's matrix of the 26B on the M4000: 1.6 ms of a step's
 * copies.) */
__global__ void __launch_bounds__(256) k_q4s_pack(uint8_t *w, int cols, int rows)
{
    extern __shared__ __align__(16) uint32_t q4row[];
    const int nb = cols / 32, rw = nb * 18 / 4, lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    const int row = blockIdx.x * (blockDim.x / 32) + wp;
    if (row >= rows) {
        return;
    }
    uint32_t *wr = (uint32_t *)(w + (size_t)row * nb * 18), *sr = q4row + (size_t)wp * rw;
    for (int i = lane; i < rw; i += 32) {
        sr[i] = wr[i];
    }
    __syncwarp();
    const uint16_t *sh = (const uint16_t *)sr;     /* block b at half 9 b: d, then the 8 halves of codes */
    for (int k = lane; k < 4 * nb; k += 32) {
        const int h = 9 * (k / 4) + 1 + 2 * (k % 4);
        wr[k] = (uint32_t)sh[h] | ((uint32_t)sh[h + 1] << 16);
    }
    for (int k = lane; k < nb / 2; k += 32) {
        wr[4 * nb + k] = (uint32_t)sh[18 * k] | ((uint32_t)sh[18 * k + 9] << 16);
    }
}

/* k_q4s_pack on the n bytes of rows of cols values at p (whole rows), on s */
static cudaError_t q4s_pack_launch(uint8_t *p, size_t n, int cols, cudaStream_t s)
{
    const size_t rb = (size_t)(cols / 32) * 18, rows = n / rb;
    if (rows == 0) {
        return cudaSuccess;
    }
    size_t nw = (48 * 1024) / rb;
    nw = nw < 1 ? 1 : nw > 8 ? 8 : nw;
    k_q4s_pack<<<(unsigned)((rows + nw - 1) / nw), (unsigned)(32 * nw), nw * rb, s>>>(p, cols, (int)rows);
    return cudaGetLastError();
}

/* The device ranges in Q4S (rc_gpu_q4s_pack), sorted by start */
#define Q4S_MAX 2048
static uintptr_t q4s_lo[Q4S_MAX], q4s_hi[Q4S_MAX];
static int q4s_n;

/* 1 when the int4 matrix at w is in Q4S */
static int q4s_is(const void *w)
{
    const uintptr_t p = (uintptr_t)w;
    int lo = 0, hi = q4s_n;
    while (lo < hi) {
        const int m = (lo + hi) / 2;
        if (q4s_hi[m] <= p) {
            lo = m + 1;
        } else {
            hi = m;
        }
    }
    return lo < q4s_n && q4s_lo[lo] <= p;
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
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    if (lane == 0) {
        DP(float, 2)[row] = sum;
    }
}
