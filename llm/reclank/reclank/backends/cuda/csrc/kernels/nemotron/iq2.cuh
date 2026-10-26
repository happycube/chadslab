/* ---------- IQ2_XXS on the GPU (the experts of Nemotron 3.5 Bolt) ----------
 * ggml's 2.06-bit i-quant (backends/include/iq2_tables.h has the layout; the CPU's
 * kernels/kq/iq2.c). A block of 256 values is 66 bytes, so the stored blocks are only 2-byte
 * aligned (the loads of bytes and halves of nm_iq2_lane, the GGUF form KQ_IQ2_XXS); the Nemotron
 * kernels read the rows of KQ_IQ2R (below). A warp takes a block: lane l the 8 values of grid
 * index l % 4 of part l / 4. */
#include "iq2_tables.h"

#define KQ_IQ2_XXS 16

__device__ const uint64_t nm_iq2_grid[256] = IQ2XXS_GRID;

/* the sum of the 8 values of lane `lane` of the block blk times x (the 256 values of the block) */
__device__ __forceinline__ float nm_iq2_lane(const uint8_t *blk, const float *x, int lane)
{
    const int j = lane >> 2, l = lane & 3;
    const uint8_t *p = blk + 2 + 8 * j;
    const uint16_t *p16 = (const uint16_t *)p;
    const uint32_t aux = (uint32_t)p16[2] | ((uint32_t)p16[3] << 16);
    const uint64_t g = nm_iq2_grid[p[l]];
    const int s7 = (aux >> (7 * l)) & 127;
    const int sg = s7 | ((__popc(s7) & 1) << 7);
    const float db = __half2float(__ushort_as_half(*(const uint16_t *)blk)) * (float)(2 * (aux >> 28) + 1) * 0.125f;
    const float4 *x4 = (const float4 *)(x + 32 * j + 8 * l);
    const float4 a = x4[0], b = x4[1];
    const float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    float s = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float v = (float)((g >> (8 * k)) & 255);
        s += ((sg >> k) & 1) ? -v * xv[k] : v * xv[k];
    }
    return db * s;
}

/* ---------- KQ_IQ2R: the rows of this runtime (reclank/formats/iq2.py to_rows) ----------
 * The same bytes as the blocks, placed for aligned loads: a row of cols values is the 8-byte parts
 * of its blocks (block b, part j at 64 b + 8 j: the 4 grid indices, then the uint32 of the 4 x 7
 * sign bits and the 4-bit scale), then the fp16 d of each block (at cols / 4 + 2 b), zeros to a
 * multiple of 8 bytes. A part is one 8-byte load; a warp's 32 parts are 256 contiguous bytes. */
#define KQ_IQ2R 67
#define NM_Q2_RB(cols) (((cols) / 4 + (cols) / 128 + 7) / 8 * 8)

__host__ __device__ __forceinline__ size_t nm_q2_rb(int cols)
{
    return ((size_t)cols / 4 + (size_t)cols / 128 + 7) & ~(size_t)7;
}

/* part j of block b of a row: x the 4 grid indices, y the signs and the scale */
__device__ __forceinline__ uint2 nm_q2_part(const uint8_t *row, int b, int j)
{
    return *(const uint2 *)(row + 64 * (size_t)b + 8 * j);
}

/* the d of block b (as its fp16 bits) */
__device__ __forceinline__ uint16_t nm_q2_dh(const uint8_t *row, int cols, int b)
{
    return *(const uint16_t *)(row + cols / 4 + 2 * (size_t)b);
}

/* As nm_iq2_lane on block b of a KQ_IQ2R row (x: the 256 values of the block). */
__device__ __forceinline__ float nm_q2_lane(const uint8_t *row, int cols, int b, const float *x, int lane)
{
    const int j = lane >> 2, l = lane & 3;
    const uint2 pt = nm_q2_part(row, b, j);
    const uint32_t aux = pt.y;
    const uint64_t g = nm_iq2_grid[(pt.x >> (8 * l)) & 255];
    const int s7 = (aux >> (7 * l)) & 127;
    const int sg = s7 | ((__popc(s7) & 1) << 7);
    const float db = __half2float(__ushort_as_half(nm_q2_dh(row, cols, b))) * (float)(2 * (aux >> 28) + 1) * 0.125f;
    const float4 *x4 = (const float4 *)(x + 32 * j + 8 * l);
    const float4 a = x4[0], c = x4[1];
    const float xv[8] = {a.x, a.y, a.z, a.w, c.x, c.y, c.z, c.w};
    float s = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float v = (float)((g >> (8 * k)) & 255);
        s += ((sg >> k) & 1) ? -v * xv[k] : v * xv[k];
    }
    return db * s;
}

/* The sums over the lanes of a warp of N values (N a power of 2, at most 32): value j in the lanes
 * lane / (32 / N) == j. A transposed butterfly: at the offsets 16, 8, .. each lane keeps the half of
 * its values on its side and adds the other lane's (N - 1 shuffles), then the plain steps for the
 * offsets below 32 / N; the same pairs in the same order as the N plain sums (v += shfl_xor(v, o),
 * o = 16 .. 1: 5 N shuffles), so the same floats. (The shuffles go through the LSU pipe, which the
 * IQ2 products fill.) */
/* one step of nm_wsum: H values kept of 2 H, the partner at offset O (templates: the indices
 * constants; a loop kept v in local memory) */
template <int H, int O>
struct nm_wstep {
    static __device__ __forceinline__ void run(float *v, int lane)
    {
        const bool up = (lane & O) != 0;
#pragma unroll
        for (int i = 0; i < H; ++i) {
            const float lo = v[i], hi = v[i + H];
            v[i] = (up ? hi : lo) + __shfl_xor_sync(RC_FULL_MASK, up ? lo : hi, O);
        }
        nm_wstep<H / 2, O / 2>::run(v, lane);
    }
};

template <int O>
struct nm_wstep<0, O> {
    static __device__ __forceinline__ void run(float *, int)
    {
    }
};

template <int N>
__device__ __forceinline__ float nm_wsum(float *v, int lane)
{
    nm_wstep<N / 2, 16>::run(v, lane);
    float s = v[0];
#pragma unroll
    for (int o = 16 / N; o > 0; o >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, o);
    }
    return s;
}

/* The signs of a grid index as 8 bytes, 0xff where the value is negative (IQ2XXS_SIGNS64). */
__device__ const uint64_t nm_iq2_signs64[128] = IQ2XXS_SIGNS64;

/* The grid bytes g (8, 25 or 43) with their signs m (0xff: negative) as int8x4: (g ^ m) - m a
 * byte, as (g ^ m) + (m & 0x01010101) on the word (255 - g + 1 < 256: no carry to the next byte;
 * __vsub4 is several instructions from sm_70 on). */
__device__ __forceinline__ int nm_q2_sw(uint32_t g, uint32_t m)
{
    return (int)((g ^ m) + (m & 0x01010101u));
}

/* The 8 int8 weights of grid index l of a part (ix: its 4 indices, aux: its signs) as two int8x4
 * words; grid, sgn: the tables (global or shared). */
__device__ __forceinline__ void nm_q2_w8(uint32_t ix, uint32_t aux, int l, const uint64_t *grid, const uint64_t *sgn,
                                         int *w0, int *w1)
{
    const uint64_t g = grid[(ix >> (8 * l)) & 255], m = sgn[(aux >> (7 * l)) & 127];
    *w0 = nm_q2_sw((uint32_t)g, (uint32_t)m);
    *w1 = nm_q2_sw((uint32_t)(g >> 32), (uint32_t)(m >> 32));
}

/* the scale of a part: d (2 s + 1) / 8 */
__device__ __forceinline__ float nm_q2_db(uint16_t dh, uint32_t aux)
{
    return __half2float(__ushort_as_half(dh)) * (float)(2 * (aux >> 28) + 1) * 0.125f;
}

/* GP_NM_HOT_MOE's experts (operand 22, the tables: 0 or none, the slots): the key of pair p's expert
 * (its slot, or with the tables its id; < 0: not run here), and the rows from row of its up (op 4) or
 * down (5) matrix of m rows of rb bytes: the slot's in the stack, or the table's address of the expert
 * (NM_GROUP_MOE's pup / pdn: a hot slot or the host memory the GPU maps; a pair of idx -1 is another
 * GPU's). */
#define NM_TAB (r->tag[22] == GP_T_INT && r->v[22] != 0)
__device__ __forceinline__ int nm_hot_key(const gp_rec *r, const int64_t *e, int p)
{
    const int x = DP(const int, 1)[p];
    return NM_TAB ? x : DP(const int, 3)[x];
}

__device__ __forceinline__ const uint8_t *nm_hot_w(const gp_rec *r, const int64_t *e, int op, int p, int key,
                                                   int m, int row, size_t rb)
{
    if (NM_TAB) {
        return (const uint8_t *)(intptr_t)DP(const int64_t, op)[DP(const int, 1)[p]] + (size_t)row * rb;
    }
    return DP(const uint8_t, op) + ((size_t)key * m + row) * rb;
}

#define NM_TT 8                     /* the items of a pass: the decoded weights of a lane serve all */

/* A lane's 8 weights of block b of a KQ_IQ2R row (grid index l of part j) as two int8x4 words and
 * the scale. */
__device__ __forceinline__ void nm_iq2_w(const uint8_t *row, int cols, int b, int lane, int *w0, int *w1, float *db)
{
    const int j = lane >> 2, l = lane & 3;
    const uint2 pt = nm_q2_part(row, b, j);
    nm_q2_w8(pt.x, pt.y, l, nm_iq2_grid, nm_iq2_signs64, w0, w1);
    *db = nm_q2_db(nm_q2_dh(row, cols, b), pt.y);
}

/* acc[tt] += the dot of a KQ_IQ2R row (cols values) with the int8 rows qs[tt] (their scales of each
 * 32: ss[tt]) of n <= NM_TT items; each lane's weights decoded once a block for all of them; then the
 * sums of the lanes (every lane gets them). */
__device__ __forceinline__ void nm_row_tt(const uint8_t *row, int cols, const int8_t *const *qs,
                                          const float *const *ss, int n, int lane, float *acc)
{
    const int j = lane >> 2, l = lane & 3;
    for (int b = 0; b < cols / 256; ++b) {
        int w0, w1;
        float db;
        nm_iq2_w(row, cols, b, lane, &w0, &w1, &db);
        #pragma unroll
        for (int tt = 0; tt < NM_TT; ++tt) {
            if (tt < n) {
                const int2 xv = *(const int2 *)(qs[tt] + 256 * b + 32 * j + 8 * l);
                acc[tt] += (float)rc_dp4a(w1, xv.y, rc_dp4a(w0, xv.x, 0)) * db * ss[tt][8 * b + j];
            }
        }
    }
    #pragma unroll
    for (int tt = 0; tt < NM_TT; ++tt) {
        for (int o = 16; o > 0; o >>= 1) {
            acc[tt] += __shfl_xor_sync(RC_FULL_MASK, acc[tt], o);
        }
    }
}

/* ---------- int16 x as two int8 planes (RECLANK_NM_X16, the default) ----------
 * x of each 32 values: q = rint(x / s), s = max |x| / 32639, then hi = (q + 128) >> 8 (-127 .. 127)
 * and lo = q - 256 hi (-128 .. 127): q = 256 hi + lo, so a dot is 256 dot(w, hi) + dot(w, lo), two
 * dp4a for each 4 values: the precision of int16 x (the noise of int8 x was ~0.5% of a product). */
#define NM_Q16 32639.f

/* nm_row_tt with two planes (qls[tt] the low planes, or qls null: int8 x). */
__device__ __forceinline__ void nm_row_tt2(const uint8_t *row, int cols, const int8_t *const *qs,
                                           const int8_t *const *qls, const float *const *ss, int n, int lane,
                                           float *acc)
{
    const int j = lane >> 2, l = lane & 3;
    for (int b = 0; b < cols / 256; ++b) {
        int w0, w1;
        float db;
        nm_iq2_w(row, cols, b, lane, &w0, &w1, &db);
        #pragma unroll
        for (int tt = 0; tt < NM_TT; ++tt) {
            if (tt < n) {
                const size_t o = 256 * b + 32 * j + 8 * l;
                const int2 h = *(const int2 *)(qs[tt] + o);
                int s = rc_dp4a(w1, h.y, rc_dp4a(w0, h.x, 0));
                if (qls != NULL) {
                    const int2 q = *(const int2 *)(qls[tt] + o);
                    s = 256 * s + rc_dp4a(w1, q.y, rc_dp4a(w0, q.x, 0));
                }
                acc[tt] += (float)s * db * ss[tt][8 * b + j];
            }
        }
    }
    #pragma unroll
    for (int tt = 0; tt < NM_TT; ++tt) {
        for (int o = 16; o > 0; o >>= 1) {
            acc[tt] += __shfl_xor_sync(RC_FULL_MASK, acc[tt], o);
        }
    }
}

/* acc[r] += the dot of R KQ_IQ2R rows (row0, rb bytes apart; nb blocks) with the two planes of x
 * (qh, ql or null; xs the scales of each 32), a lane's 8 values of each block: the loads of the R
 * rows and of NM_LU blocks issued before the products (a warp a row waited on each block's bytes,
 * then on the grid). grid, sgn: the tables (global, or shared: k_nm_iq2_lin). The sums of the
 * lanes: the caller's. */
#define NM_LU 4
template <int R>
__device__ __forceinline__ void nm_iq2_rows_x2(const uint8_t *row0, size_t rb, int nb, const int8_t *qh,
                                               const int8_t *ql, const float *xs, int lane, float *acc,
                                               const uint64_t *grid, const uint64_t *sgn)
{
    const int j = lane >> 2, l = lane & 3;
    for (int b = 0; b < nb; b += NM_LU) {
        uint2 pt[NM_LU][R];
        uint16_t d[NM_LU][R];
#pragma unroll
        for (int u = 0; u < NM_LU; ++u) {
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint8_t *row = row0 + (size_t)r * rb;
                const int bb = b + u < nb ? b + u : b;
                pt[u][r] = nm_q2_part(row, bb, j);
                d[u][r] = nm_q2_dh(row, 256 * nb, bb);
            }
        }
#pragma unroll
        for (int u = 0; u < NM_LU; ++u) {
            if (b + u < nb) {
                const size_t o = 256 * (size_t)(b + u) + 32 * j + 8 * l;
                const int2 h = *(const int2 *)(qh + o);
                const int2 lo = ql != NULL ? *(const int2 *)(ql + o) : make_int2(0, 0);
                const float sx = xs[8 * (b + u) + j];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int w0, w1;
                    nm_q2_w8(pt[u][r].x, pt[u][r].y, l, grid, sgn, &w0, &w1);
                    int sm = rc_dp4a(w1, h.y, rc_dp4a(w0, h.x, 0));
                    if (ql != NULL) {
                        sm = 256 * sm + rc_dp4a(w1, lo.y, rc_dp4a(w0, lo.x, 0));
                    }
                    acc[r] += (float)sm * nm_q2_db(d[u][r], pt[u][r].y) * sx;
                }
            }
        }
    }
}

/* ---------- a lane a part (k_nm_q2_lin, the hot experts) ----------
 * Lane l takes part p = c0 + 32 u + l of a row (bytes 8p .. 8p + 7, the d of its block at cols / 4
 * + 2 (p / 8)): a warp's loads of a row are 256 contiguous bytes. nm_q2_load issues the loads of U
 * chunks of 32 parts of R rows (the weights do not wait for the kernel before: they may go before
 * griddepcontrol.wait), nm_q2_dot adds their products with the two planes of x (X2; else int8 x):
 * the int32 sum of a part (8 dp4a a plane: |sum| < 32 x 43 x 128 x 257), then one scale. */
template <int R, int U>
__device__ __forceinline__ void nm_q2_load(const uint8_t *row0, size_t rb, int cols, int c0, int lane,
                                           uint2 (&pt)[U][R], uint32_t (&dh)[U][R])
{
    const int np = cols / 32;
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int p = c0 + 32 * u + lane, pp = p < np ? p : 0;
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *row = row0 + (size_t)r * rb;
            pt[u][r] = __ldg((const uint2 *)(row + 8 * (size_t)pp));
            dh[u][r] = __ldg((const unsigned short *)(row + cols / 4 + 2 * (pp >> 3)));
        }
    }
}

/* the int8x4 words of grid index l of a part: ix its 4 indices, aux its signs and scale; the
 * signs from the bits (b: bit k of the 4 to byte k, (b * 0x204081) & 0x01010101; the mask b x 255)
 * and w = (g ^ m) + b (nm_q2_sw) */
__device__ __forceinline__ void nm_q2_wl(uint32_t ix, uint32_t aux, int l, const uint64_t *grid, int *w0, int *w1)
{
    const uint32_t s7 = (aux >> (7 * l)) & 127;
    const uint32_t b0 = ((s7 & 15) * 0x204081u) & 0x01010101u;
    const uint32_t b1 = (((s7 >> 4) | ((__popc(s7) & 1) << 3)) * 0x204081u) & 0x01010101u;
    const uint64_t g = grid[__byte_perm(ix, 0, 0x4440 + l)];
    *w0 = (int)(((uint32_t)g ^ (b0 * 255u)) + b0);
    *w1 = (int)(((uint32_t)(g >> 32) ^ (b1 * 255u)) + b1);
}

template <int R, int U, bool X2>
__device__ __forceinline__ void nm_q2_dot(int cols, int c0, int lane, const uint2 (&pt)[U][R], const uint32_t (&dh)[U][R],
                                          const int8_t *qh, const int8_t *ql, const float *xs, const uint64_t *grid,
                                          float *acc)
{
    const int np = cols / 32;
#pragma unroll
    for (int u = 0; u < U; ++u) {
        const int p = c0 + 32 * u + lane;
        if (p < np) {
            const int4 h0 = __ldg((const int4 *)(qh + 32 * (size_t)p)), h1 = __ldg((const int4 *)(qh + 32 * (size_t)p + 16));
            int4 l0 = make_int4(0, 0, 0, 0), l1 = l0;
            if (X2) {
                l0 = __ldg((const int4 *)(ql + 32 * (size_t)p));
                l1 = __ldg((const int4 *)(ql + 32 * (size_t)p + 16));
            }
            const int hx[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
            const int lx[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
            const float sx = __ldg(xs + p);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint32_t ix = pt[u][r].x, aux = pt[u][r].y;
                int sh = 0, sl = 0;
#pragma unroll
                for (int l = 0; l < 4; ++l) {
                    int w0, w1;
                    nm_q2_wl(ix, aux, l, grid, &w0, &w1);
                    sh = rc_dp4a(w1, hx[2 * l + 1], rc_dp4a(w0, hx[2 * l], sh));
                    if (X2) {
                        sl = rc_dp4a(w1, lx[2 * l + 1], rc_dp4a(w0, lx[2 * l], sl));
                    }
                }
                acc[r] += (float)(X2 ? 256 * sh + sl : sh) * nm_q2_db((uint16_t)dh[u][r], aux) * sx;
            }
        }
    }
}

/* acc[r] += the dot of R rows with x (ql null: int8 x), all chunks (nm_q2_load, nm_q2_dot, U = 1) */
template <int R>
__device__ __forceinline__ void nm_q2_rows(const uint8_t *row0, size_t rb, int cols, const int8_t *qh, const int8_t *ql,
                                           const float *xs, int lane, float *acc, const uint64_t *grid)
{
    uint2 pt[1][R];
    uint32_t dh[1][R];
    for (int c0 = 0; c0 < cols / 32; c0 += 32) {
        nm_q2_load<R, 1>(row0, rb, cols, c0, lane, pt, dh);
        if (ql != NULL) {
            nm_q2_dot<R, 1, true>(cols, c0, lane, pt, dh, qh, ql, xs, grid, acc);
        } else {
            nm_q2_dot<R, 1, false>(cols, c0, lane, pt, dh, qh, ql, xs, grid, acc);
        }
    }
}

/* the grid and the signs in shared memory (the lanes' lookups of the global tables, each lane a sector
 * of its own, filled the load queue of the L1: 30 sectors a request) */
#define NM_IQ2_SMEM_TABLES()                                                                       \
    __shared__ uint64_t nm_sg[256], nm_ss[128];                                                    \
    for (int i_ = threadIdx.x; i_ < 256; i_ += blockDim.x) {                                       \
        nm_sg[i_] = nm_iq2_grid[i_];                                                               \
        if (i_ < 128) {                                                                            \
            nm_ss[i_] = nm_iq2_signs64[i_];                                                        \
        }                                                                                          \
    }                                                                                              \
    __syncthreads()
