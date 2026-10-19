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
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    return sum;
}

#define ROWS_PER_BLOCK 8


/* One warp for each row of an int4 matrix. The arguments give the operands
 * of x, w, out, rows, and cols: 0, 1, 3, 4, 5 for GP_INT4_LINEAR, and 3, 4,
 * 6, 7, 8 for the matrix of GP_GELU_MUL_INT4. The kernel reads the float16
 * scale of each block, not the float32 scales. reclank/backends/cuda/runtime.py checks that
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
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    if (lane == 0) {
        DP(float, 2)[row] = sum;
    }
}
