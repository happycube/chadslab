/* ---------- the KQ_IQ2R products in float32, for a GPU without dp4a (Maxwell, sm_52) ----------
 * The weights decoded to float (iq2.cuh's rows), x float: no int8 x and act. The grid goes to shared
 * memory first: a lane's index is its own, and a gather of 32 of them from the 2 KB table in global
 * memory replays for each line it touches (kq_row_part's KQ_IQ2R, nm_q2_lane, which reads it there: a
 * matrix of the shared expert, 4096 x 5376, 1.5 ms on the Quadro M4000; k_nm_q2_lin_f 0.24 ms).
 *
 * k_nm_q2_lin_f: GP_KQ_LINEAR of KQ_IQ2R rows (x, w, type, rows, cols, t, out at operands 3, 4, 5, 6,
 *     7, 8, 9), at most T tokens a pass: a warp for 2 rows, a lane the 8 values of grid index l % 4
 *     of part l / 4 of each block, the sums over the lanes after.
 * k_nm_hot_up_f, k_nm_hot_down_f: GP_NM_HOT_MOE's float form (its operands, the slots or with
 *     operand 22 the tables of nm_hot_key / nm_hot_w), 2 rows a warp; k_nm_hot_sum (nm.cu) after.
 * k_nm_gtile_f: GP_NM_GROUP_MOE's tiles (k_nm_gtile's form): a block for each expert and 32 rows of
 *     its up (or down) matrix, the 32 raw rows in shared memory once (each expert crosses PCIe once
 *     for the group), then for each 32 of its pairs and each 32 columns (one part of a block of
 *     every row) the rows decoded to float with their scale, the pairs' x, and a 2 x 2 patch of the
 *     products a thread. up: relu^2 into act (float, operand 7); down: times the weight into de. The
 *     Quadro M4000, a group of 1024 tokens on 304 experts (L 1024, inner 2048, k 9): 97 ms up, 97
 *     down, 199 GMAC/s, the same from mapped host memory (tried and slower: 4 x 4 patches of a
 *     quarter of the columns each, 157 GMAC/s; 4 x 2 of a half, 192). */

/* the grid in shared memory, for the block */
#define NMF_GRID()                                                                                 \
    __shared__ uint64_t nmf_grid[256];                                                             \
    nmf_grid_fill(nmf_grid);                                                                       \
    __syncthreads()

__device__ __forceinline__ void nmf_grid_fill(uint64_t *grid)
{
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        grid[i] = nm_iq2_grid[i];
    }
}

/* the 8 values of grid index l of the part pt with their signs (no scale). (Tried and slower on the
 * M4000, a matrix of the shared expert at one token: the bytes as 2^23 + b less 2^23, no I2F, 240 ->
 * 284 us; the grid as floats in shared memory, two float4 an index, 293 us.) */
__device__ __forceinline__ void nmf_w8(uint2 pt, int l, const uint64_t *grid, float *w)
{
    const uint64_t g = grid[(pt.x >> (8 * l)) & 255];
    const int s7 = (pt.y >> (7 * l)) & 127, sg = s7 | ((__popc(s7) & 1) << 7);
    const uint32_t lo = (uint32_t)g, hi = (uint32_t)(g >> 32);
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        const float a = (float)((lo >> (8 * u)) & 255), b = (float)((hi >> (8 * u)) & 255);
        w[u] = ((sg >> u) & 1) ? -a : a;
        w[u + 4] = ((sg >> (u + 4)) & 1) ? -b : b;
    }
}

__device__ __forceinline__ float nmf_dot8(const float *w, const float *x)
{
    const float4 a = *(const float4 *)x, c = *(const float4 *)(x + 4);
    return w[0] * a.x + w[1] * a.y + w[2] * a.z + w[3] * a.w + w[4] * c.x + w[5] * c.y + w[6] * c.z + w[7] * c.w;
}

__device__ __forceinline__ float nmf_wsum(float v)
{
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(RC_FULL_MASK, v, o);
    }
    return v;
}

/* the dots of the rows r0 and r1 (rb bytes, cols values) with x (float, cols), a warp's lanes: U blocks a
 * pass, the loads of the pass first, at offsets of pointers stepped once a pass (a block a pass, its
 * addresses from the block's index: a matrix of the shared expert, 4096 x 5376, 241 us on the M4000; 4
 * blocks a pass 176) */
template <int U>
__device__ __forceinline__ void nmf_rows2(const uint8_t *r0, const uint8_t *r1, int cols, const float *x,
                                          const uint64_t *grid, int lane, float *s0, float *s1)
{
    const int j = lane >> 2, l = lane & 3, nb = cols / 256;
    const uint8_t *p0 = r0 + 8 * j, *p1 = r1 + 8 * j;
    const uint16_t *d0 = (const uint16_t *)(r0 + cols / 4), *d1 = (const uint16_t *)(r1 + cols / 4);
    const float *xp = x + 32 * j + 8 * l;
    float a = 0.f, b = 0.f;
    int bk = 0;
    for (; U > 1 && bk + U <= nb; bk += U, p0 += 64 * U, p1 += 64 * U, xp += 256 * U) {
        uint2 pa[U], pb[U];
        uint32_t ha[(U + 1) / 2], hb[(U + 1) / 2];
        /* (the d of the pass in one load: cols / 4 is a multiple of 64, bk of U) */
        #pragma unroll
        for (int u = 0; u < (U + 1) / 2; ++u) {
            ha[u] = ((const uint32_t *)(d0 + bk))[u];
            hb[u] = ((const uint32_t *)(d1 + bk))[u];
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            pa[u] = *(const uint2 *)(p0 + 64 * u);
            pb[u] = *(const uint2 *)(p1 + 64 * u);
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            float wa[8], wb[8];
            nmf_w8(pa[u], l, grid, wa);
            nmf_w8(pb[u], l, grid, wb);
            a += nm_q2_db((uint16_t)(ha[u / 2] >> (16 * (u % 2))), pa[u].y) * nmf_dot8(wa, xp + 256 * u);
            b += nm_q2_db((uint16_t)(hb[u / 2] >> (16 * (u % 2))), pb[u].y) * nmf_dot8(wb, xp + 256 * u);
        }
    }
    for (; bk < nb; ++bk, p0 += 64, p1 += 64, xp += 256) {
        const uint2 pa = *(const uint2 *)p0, pb = *(const uint2 *)p1;
        float wa[8], wb[8];
        nmf_w8(pa, l, grid, wa);
        nmf_w8(pb, l, grid, wb);
        a += nm_q2_db(d0[bk], pa.y) * nmf_dot8(wa, xp);
        b += nm_q2_db(d1[bk], pb.y) * nmf_dot8(wb, xp);
    }
    *s0 = nmf_wsum(a);
    *s1 = nmf_wsum(b);
}

template <int T>
__global__ void __launch_bounds__(256) k_nm_q2_lin_f(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    NMF_GRID();
    const int rows = DI(6), cols = DI(7), t = DI(8), lane = threadIdx.x % 32;
    const int row = 2 * (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32);
    if (row >= rows) {
        return;
    }
    const size_t rb = nm_q2_rb(cols);
    const uint8_t *w0 = DP(const uint8_t, 4) + (size_t)row * rb, *w1 = row + 1 < rows ? w0 + rb : w0;
    const float *X = DP(const float, 3);
    float *out = DP(float, 9);
    if (T == 1 && t == 1) {
        float va, vb;
        nmf_rows2<4>(w0, w1, cols, X, nmf_grid, lane, &va, &vb);
        if (lane == 0) {
            out[row] = va;
            if (row + 1 < rows) {
                out[row + 1] = vb;
            }
        }
        return;
    }
    const int j = lane >> 2, l = lane & 3;
    for (int t0 = 0; t0 < t; t0 += T) {
        const int n = min(T, t - t0);
        float a[T], b[T];
        #pragma unroll
        for (int tt = 0; tt < T; ++tt) {
            a[tt] = b[tt] = 0.f;
        }
        for (int bk = 0; bk < cols / 256; ++bk) {
            const uint2 pa = nm_q2_part(w0, bk, j), pb = nm_q2_part(w1, bk, j);
            float wa[8], wb[8];
            nmf_w8(pa, l, nmf_grid, wa);
            nmf_w8(pb, l, nmf_grid, wb);
            const float da = nm_q2_db(nm_q2_dh(w0, cols, bk), pa.y), db = nm_q2_db(nm_q2_dh(w1, cols, bk), pb.y);
            const float *xp = X + (size_t)t0 * cols + 256 * bk + 32 * j + 8 * l;
            #pragma unroll
            for (int tt = 0; tt < T; ++tt) {
                if (tt < n) {
                    a[tt] += da * nmf_dot8(wa, xp + (size_t)tt * cols);
                    b[tt] += db * nmf_dot8(wb, xp + (size_t)tt * cols);
                }
            }
        }
        #pragma unroll
        for (int tt = 0; tt < T; ++tt) {
            if (tt < n) {
                const float va = nmf_wsum(a[tt]), vb = nmf_wsum(b[tt]);
                if (lane == 0) {
                    out[(size_t)(t0 + tt) * rows + row] = va;
                    if (row + 1 < rows) {
                        out[(size_t)(t0 + tt) * rows + row + 1] = vb;
                    }
                }
            }
        }
    }
}

/* GP_NM_HOT_MOE (float): up, 2 rows of the up matrix a warp at a time, NMF_HR times (grid: inner / (16
 * NMF_HR), pairs), NMF_HU blocks a pass (nmf_rows2), at most 64 registers (NMF_HB blocks an SM). The M4000,
 * 9 pairs in GPU memory: a block for 16 rows, a block a pass 538 us; 4 passes, 2 blocks 452 (1 pass 508;
 * 4 blocks a pass 81 registers, 3 blocks an SM: 624; 6 blocks an SM, spilled: 912). */
#ifndef NMF_HR
#define NMF_HR 4
#endif
#ifndef NMF_HB
#define NMF_HB 4
#endif
#ifndef NMF_HU
#define NMF_HU 2
#endif
__global__ void __launch_bounds__(256, NMF_HB) k_nm_hot_up_f(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    NMF_GRID();
    const int k = DI(9), L = DI(10), inner = DI(11), pr = blockIdx.y, lane = threadIdx.x % 32;
    const int key = nm_hot_key(r, e, pr);
    if (key < 0) {
        return;
    }
    const size_t rb = nm_q2_rb(L);
    const float *x = DP(const float, 0) + (size_t)(pr / k) * L;
    for (int row = 2 * (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32); row < inner;
         row += 2 * gridDim.x * (blockDim.x / 32)) {
        float s0, s1;
        nmf_rows2<NMF_HU>(nm_hot_w(r, e, 4, pr, key, inner, row, rb), nm_hot_w(r, e, 4, pr, key, inner, row + 1, rb), L, x,
                  nmf_grid, lane, &s0, &s1);
        if (lane == 0) {
            s0 = fmaxf(s0, 0.f);
            s1 = fmaxf(s1, 0.f);
            DP(float, 6)[(size_t)pr * inner + row] = s0 * s0;
            DP(float, 6)[(size_t)pr * inner + row + 1] = s1 * s1;
        }
    }
}

/* down, 2 rows of the down matrix a warp at a time (as up), times the pair's weight (zeros for a pair
 * not run here; k_nm_hot_sum then) */
__global__ void __launch_bounds__(256, NMF_HB) k_nm_hot_down_f(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int L = DI(10), inner = DI(11), pr = blockIdx.y, lane = threadIdx.x % 32;
    const int key = nm_hot_key(r, e, pr);
    if (key < 0) {
        for (int row = 2 * (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32); row < L;
             row += 2 * gridDim.x * (blockDim.x / 32)) {
            if (lane == 0) {
                DP(float, 7)[(size_t)pr * L + row] = 0.f;
                DP(float, 7)[(size_t)pr * L + row + 1] = 0.f;
            }
        }
        return;
    }
    NMF_GRID();
    const size_t rb = nm_q2_rb(inner);
    const float *x = DP(const float, 6) + (size_t)pr * inner, w = DP(const float, 2)[pr];
    for (int row = 2 * (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32); row < L;
         row += 2 * gridDim.x * (blockDim.x / 32)) {
        float s0, s1;
        nmf_rows2<NMF_HU>(nm_hot_w(r, e, 5, pr, key, L, row, rb), nm_hot_w(r, e, 5, pr, key, L, row + 1, rb), inner, x,
                  nmf_grid, lane, &s0, &s1);
        if (lane == 0) {
            DP(float, 7)[(size_t)pr * L + row] = s0 * w;
            DP(float, 7)[(size_t)pr * L + row + 1] = s1 * w;
        }
    }
}

#define NMF_BM 32                   /* the rows of a block of k_nm_gtile_f */
#define NMF_BN 32                   /* the pairs of a pass */

/* the dynamic shared memory of k_nm_gtile_f for rows of cols values */
__host__ __device__ static inline size_t nmf_gtile_smem(int cols)
{
    return 256 * 8 + 2 * 32 * (NMF_BM + 2) * 4 + NMF_BM * nm_q2_rb(cols);
}

template <bool DOWN>
__global__ void __launch_bounds__(256) k_nm_gtile_f(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t nmf_sm[];
    const int k = DI(13), L = DI(14), inner = DI(15), E = DI(17), tid = threadIdx.x;
    const int x = blockIdx.x, r0 = blockIdx.y * NMF_BM;
    const int rows = DOWN ? L : inner, cols = DOWN ? inner : L;
    const int *start = DP(const int, 12), *order = start + 2 * E + 1;
    const int n0 = start[x], n1 = start[x + 1];
    if (n0 == n1 || r0 >= rows) {
        return;
    }
    const size_t rb = nm_q2_rb(cols);
    uint64_t *grid = (uint64_t *)nmf_sm;                                        /* 256 */
    float (*wt)[NMF_BM + 2] = (float (*)[NMF_BM + 2])(grid + 256);              /* column, row */
    float (*xt)[NMF_BN + 2] = (float (*)[NMF_BN + 2])(&wt[32][0]);              /* column, pair */
    uint8_t *raw = (uint8_t *)&xt[32][0];                                       /* NMF_BM x rb */
    const int nrow = min(NMF_BM, rows - r0);
    nmf_grid_fill(grid);
    {
        /* (the rows: 8-byte loads, rb a multiple of 8) */
        const uint2 *s = (const uint2 *)((const uint8_t *)(intptr_t)DP(const int64_t, DOWN ? 4 : 3)[x] + (size_t)r0 * rb);
        for (size_t i = tid; i < nrow * rb / 8; i += blockDim.x) {
            ((uint2 *)raw)[i] = s[i];
        }
    }
    const float *X = DOWN ? DP(const float, 7) : DP(const float, 0);
    const int ty = tid / 16, tx = tid % 16;                     /* rows 2 ty, 2 ty + 1; pairs 2 tx, + 1 */
    /* (the x threads: 8 values of a pair from column k0 + 8 l, loaded a chunk ahead) */
    const int xq = tid - NMF_BM * 4, xpp = xq / 4, xl = xq % 4;
    for (int a0 = n0; a0 < n1; a0 += NMF_BN) {
        const int np = min(NMF_BN, n1 - a0);
        float c00 = 0.f, c01 = 0.f, c10 = 0.f, c11 = 0.f;
        const float *xr = NULL;
        float4 xa = make_float4(0.f, 0.f, 0.f, 0.f), xc = xa;
        if (tid >= NMF_BM * 4 && xpp < np) {
            const int pr = order[a0 + xpp];
            xr = X + (DOWN ? (size_t)pr * inner : (size_t)(pr / k) * L) + 8 * xl;
            xa = *(const float4 *)xr;
            xc = *(const float4 *)(xr + 4);
        }
        for (int k0 = 0; k0 < cols; k0 += 32) {
            __syncthreads();
            if (tid < NMF_BM * 4) {
                /* row rr's 8 values of grid index l of the part, times its scale */
                const int rr = tid / 4, l = tid % 4;
                float w[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                if (rr < nrow) {
                    const uint8_t *q2r = raw + rr * rb;
                    const uint2 pt = nm_q2_part(q2r, k0 / 256, (k0 % 256) / 32);
                    nmf_w8(pt, l, grid, w);
                    const float db = nm_q2_db(nm_q2_dh(q2r, cols, k0 / 256), pt.y);
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        w[u] *= db;
                    }
                }
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    wt[8 * l + u][rr] = w[u];
                }
            } else {
                const float v[8] = {xa.x, xa.y, xa.z, xa.w, xc.x, xc.y, xc.z, xc.w};
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    xt[8 * xl + u][xpp] = v[u];
                }
                if (xr != NULL && k0 + 32 < cols) {
                    xa = *(const float4 *)(xr + k0 + 32);
                    xc = *(const float4 *)(xr + k0 + 36);
                }
            }
            __syncthreads();
            #pragma unroll 8
            for (int c = 0; c < 32; ++c) {
                const float2 w = *(const float2 *)&wt[c][2 * ty], v = *(const float2 *)&xt[c][2 * tx];
                c00 += w.x * v.x;
                c01 += w.x * v.y;
                c10 += w.y * v.x;
                c11 += w.y * v.y;
            }
        }
        const float acc[2][2] = {{c00, c01}, {c10, c11}};
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int rr = 2 * ty + i;
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int pp = 2 * tx + jj;
                if (rr < nrow && pp < np) {
                    const int pr = order[a0 + pp];
                    if (DOWN) {
                        DP(float, 10)[(size_t)pr * L + r0 + rr] = acc[i][jj] * DP(const float, 2)[pr];
                    } else {
                        const float v = fmaxf(acc[i][jj], 0.f);
                        DP(float, 7)[(size_t)pr * inner + r0 + rr] = v * v;
                    }
                }
            }
        }
    }
}
