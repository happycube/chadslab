/* DeepSeek V4.1: the products of the hot experts (GP_DS_HOT_MOE; step.cu
 * gives the operands, the act and the combine), in a module of their own:
 * reclank_cuda.cu includes it after step.cu, and bench/ds_prefill.cu builds
 * it alone on operands dumped from a real run (RECLANK_DS_DUMP). A step and a
 * verify group: a block for each pair (k_ds_hot_gu, k_ds_hot_dn). A group of
 * a prompt: a block for each expert's rows (k_ds_hot_gu_g, k_ds_hot_dn_g; on
 * the tensor cores k_ds_hot_tc; k_ds_hot_tc6 on the items of k_ds_hot_lists).
 * The same bits in all of them: those of the CPU parts. */
#pragma once
#include "../frag.cuh"
#include "common.cuh"

__global__ void __launch_bounds__(256) k_ds_hot_gu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int pq = blockIdx.z, k = DI(9), H = DI(10), I = DI(11);
    int slot = DP(const int, 3)[DP(const int, 1)[pq]];
    if (slot < 0) {
        return;
    }
    int j = pq / k, mat = blockIdx.y, grp = blockIdx.x, nb = H / 32;
    const uint8_t *w = ds_slot_w(r, e, slot) + (mat ? di(r, e, 6) : 0) +
                       (size_t)grp * nb * MXX_BB;
    mxx_group_x2<8>(w, H, DP(const int8_t, 12) + (size_t)j * H, DP(const int8_t, 13) + (size_t)j * H,
                    DP(const float, 14) + (size_t)j * nb, DP(float, 15) + (size_t)pq * 2 * I + mat * I + 16 * grp);
}

__global__ void __launch_bounds__(256) k_ds_hot_dn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int pq = blockIdx.y, H = DI(10), I = DI(11);
    int slot = DP(const int, 3)[DP(const int, 1)[pq]];
    if (slot < 0) {
        return;
    }
    int grp = blockIdx.x, nb = I / 32;
    const uint8_t *w = ds_slot_w(r, e, slot) + di(r, e, 7) + (size_t)grp * nb * MXX_BB;
    mxx_group_x2<8>(w, I, DP(const int8_t, 17) + (size_t)pq * I, DP(const int8_t, 18) + (size_t)pq * I,
                    DP(const float, 19) + (size_t)pq * nb, DP(float, 20) + (size_t)pq * H + 16 * grp);
}

/* GP_DS_HOT_MOE for a group of a prompt (t > 8): a block for each (group of
 * 16 rows, matrix, expert); the experts the GPU does not hold and the ones no
 * pair takes leave at once; the others take their pairs (the scan of idx)
 * and the same rows of the expert for each of them in turn, so the rows are
 * read once from the memory and then from the cache (k_ds_hot_gu read them
 * again for every pair: 1.9 s of a group of 512 on the real prompt). The
 * operations of a pair are those of k_ds_hot_gu: the same bits. */
__global__ void __launch_bounds__(256) k_ds_hot_gu_g(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ex = blockIdx.z, k = DI(9), H = DI(10), I = DI(11), t = DI(8);
    int slot = DP(const int, 3)[ex];
    if (slot < 0) {
        return;
    }
    __shared__ int pl[DS_HOT_MAXP];
    __shared__ int np_;
    const int *idx = DP(const int, 1);
    for (int p0 = 0; p0 < t * k; p0 += DS_HOT_MAXP * k) {
        int n = ds_hot_pairs(idx, ex, p0, t * k, k, pl, &np_);
        if (n == 0) {
            continue;
        }
        int mat = blockIdx.y, grp = blockIdx.x, nb = H / 32;
        const uint8_t *w = ds_slot_w(r, e, slot) + (mat ? di(r, e, 6) : 0) +
                           (size_t)grp * nb * MXX_BB;
        for (int q = 0; q < n; ++q) {
            int pq = pl[q], j = pq / k;
            mxx_group_x2<8>(w, H, DP(const int8_t, 12) + (size_t)j * H, DP(const int8_t, 13) + (size_t)j * H,
                            DP(const float, 14) + (size_t)j * nb,
                            DP(float, 15) + (size_t)pq * 2 * I + mat * I + 16 * grp);
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(256) k_ds_hot_dn_g(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ex = blockIdx.y, k = DI(9), H = DI(10), I = DI(11), t = DI(8);
    int slot = DP(const int, 3)[ex];
    if (slot < 0) {
        return;
    }
    __shared__ int pl[DS_HOT_MAXP];
    __shared__ int np_;
    const int *idx = DP(const int, 1);
    for (int p0 = 0; p0 < t * k; p0 += DS_HOT_MAXP * k) {
        int n = ds_hot_pairs(idx, ex, p0, t * k, k, pl, &np_);
        if (n == 0) {
            continue;
        }
        int grp = blockIdx.x, nb = I / 32;
        const uint8_t *w = ds_slot_w(r, e, slot) + di(r, e, 7) + (size_t)grp * nb * MXX_BB;
        for (int q = 0; q < n; ++q) {
            int pq = pl[q];
            mxx_group_x2<8>(w, I, DP(const int8_t, 17) + (size_t)pq * I, DP(const int8_t, 18) + (size_t)pq * I,
                            DP(const float, 19) + (size_t)pq * nb, DP(float, 20) + (size_t)pq * H + 16 * grp);
            __syncthreads();
        }
    }
}

/* The same on the tensor cores (gg_caps.tc): a warp takes 16 rows of the
 * expert for 8 of its pairs at once, mma m16n8k32 on the int8 planes (the
 * weights 2 e2m1 from the KQ_MXX block: the 4 bytes at 32 c + 4 g are rows g
 * (low) and g + 8 (high) of values 4c .. 4c + 3, the A fragments; x the B
 * fragments), so for each block of 32 columns a = 128 sum(hi w) + sum(lo w),
 * exact in int32 as in k_mxx_rows_x2; then p = a 2^(e - 127) and f =
 * fma(p, xs, f) over the blocks in order, 0.5 f: the bits of the CPU. The
 * rows of the expert once for its pairs, 8 pairs a load. */
template <bool DN>
__global__ void __launch_bounds__(128) k_ds_hot_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ex = DN ? blockIdx.y : blockIdx.z, k = DI(9), H = DI(10), I = DI(11), t = DI(8);
    int slot = DP(const int, 3)[ex];
    if (slot < 0) {
        return;
    }
    __shared__ int pl[DS_HOT_MAXP];
    __shared__ int np_;
    const int *idx = DP(const int, 1);
    for (int p0 = 0; p0 < t * k; p0 += DS_HOT_MAXP * k) {
        int n = ds_hot_pairs(idx, ex, p0, t * k, k, pl, &np_);
        if (n == 0) {
            continue;
        }
        int cols = DN ? I : H, nb = cols / 32, grp = blockIdx.x, mat = DN ? 0 : blockIdx.y;
        const uint8_t *wg = ds_slot_w(r, e, slot) +
                            (DN ? di(r, e, 7) : (mat ? di(r, e, 6) : 0)) + (size_t)grp * nb * MXX_BB;
        const int8_t *qh = DP(const int8_t, DN ? 17 : 12), *ql = DP(const int8_t, DN ? 18 : 13);
        const float *xs = DP(const float, DN ? 19 : 14);
        float *out = DP(float, DN ? 20 : 15);
        int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
        for (int t0 = 8 * warp; t0 < n; t0 += 32) {
            /* the pair of this lane's B column (g), and of its C columns (2c, 2c + 1) */
            int pb = t0 + g < n ? pl[t0 + g] : -1;
            int pc0 = t0 + 2 * c < n ? pl[t0 + 2 * c] : -1, pc1 = t0 + 2 * c + 1 < n ? pl[t0 + 2 * c + 1] : -1;
            size_t xb = pb < 0 ? 0 : (size_t)(DN ? pb : pb / k) * cols;
            size_t s0 = pc0 < 0 ? 0 : (size_t)(DN ? pc0 : pc0 / k) * nb;
            size_t s1 = pc1 < 0 ? 0 : (size_t)(DN ? pc1 : pc1 / k) * nb;
            float f[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + (size_t)b * MXX_BB;
                uint32_t w0 = *(const uint32_t *)(blk + 32 * c + 4 * g);
                uint32_t w1 = *(const uint32_t *)(blk + 128 + 32 * c + 4 * g);
                uint32_t a[4] = {kt_e2m1x4(w0), kt_e2m1x4(w0 >> 4), kt_e2m1x4(w1), kt_e2m1x4(w1 >> 4)};
                uint32_t bh[2] = {0u, 0u}, bl[2] = {0u, 0u};
                if (pb >= 0) {
                    bh[0] = *(const uint32_t *)(qh + xb + 32 * b + 4 * c);
                    bh[1] = *(const uint32_t *)(qh + xb + 32 * b + 16 + 4 * c);
                    bl[0] = *(const uint32_t *)(ql + xb + 32 * b + 4 * c);
                    bl[1] = *(const uint32_t *)(ql + xb + 32 * b + 16 + 4 * c);
                }
                int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0};
                mma16832(ch, a, bh);
                mma16832(cl, a, bl);
                float e0 = e8m0f(blk[256 + g]), e1 = e8m0f(blk[256 + g + 8]);
                float x0 = pc0 >= 0 ? xs[s0 + b] : 0.f, x1 = pc1 >= 0 ? xs[s1 + b] : 0.f;
                f[0] = fmaf(__fmul_rn((float)(128 * ch[0] + cl[0]), e0), x0, f[0]);
                f[1] = fmaf(__fmul_rn((float)(128 * ch[1] + cl[1]), e0), x1, f[1]);
                f[2] = fmaf(__fmul_rn((float)(128 * ch[2] + cl[2]), e1), x0, f[2]);
                f[3] = fmaf(__fmul_rn((float)(128 * ch[3] + cl[3]), e1), x1, f[3]);
            }
            /* c0, c1: row g, pairs 2c, 2c + 1; c2, c3: row g + 8 */
            int rows[2] = {16 * grp + g, 16 * grp + g + 8};
            int pcs[2] = {pc0, pc1};
            for (int i = 0; i < 4; ++i) {
                int pq = pcs[i & 1], row = rows[i >> 1];
                if (pq < 0) {
                    continue;
                }
                float v = __fmul_rn(f[i], 0.5f);
                if (DN) {
                    out[(size_t)pq * H + row] = v;
                } else {
                    out[(size_t)pq * 2 * I + mat * I + row] = v;
                }
            }
        }
    }
}

/* The pairs of each expert that map holds, once for the kernels of a record (k_ds_hot_tc rescanned
 * all the pairs of the group in each of its blocks), in items of up to ch pairs of one expert (one
 * expert can take most of the pairs of a group: 327 of 430 on a real one, and its blocks ran its
 * pairs in turn while the others had finished): ex [nx] the expert of each item (in id order), off
 * [nx] its first pair in pl, cn [nx] its count of pairs, pl the pairs of each expert in turn (in any
 * order: each pair's sums are its own), nx (one int). One block of DS_HL_E threads, E <= DS_HL_E. */
#define DS_HL_E 1024
__global__ void __launch_bounds__(1024) k_ds_hot_lists(const int *idx, const int *map, int np, int E, int ch,
                                                       int *ex, int *off, int *cn, int *pl, int *nx)
{
    __shared__ int cnt[DS_HL_E], at[DS_HL_E];
    cnt[threadIdx.x] = 0;
    __syncthreads();
    for (int p = threadIdx.x; p < np; p += blockDim.x) {
        int xp = idx[p];
        if (xp >= 0 && map[xp] >= 0) {
            atomicAdd(&cnt[xp], 1);
        }
    }
    __syncthreads();
    /* each expert's first pair and first item: the sums of the counts before it (a scan) */
    int x = threadIdx.x, c = x < E ? cnt[x] : 0, ni = (c + ch - 1) / ch;
    __shared__ int sc[DS_HL_E], si[DS_HL_E];
    sc[x] = c;
    si[x] = ni;
    __syncthreads();
    for (int d = 1; d < DS_HL_E; d *= 2) {
        int a = x >= d ? sc[x - d] : 0, b = x >= d ? si[x - d] : 0;
        __syncthreads();
        sc[x] += a;
        si[x] += b;
        __syncthreads();
    }
    int o = sc[x] - c, n0 = si[x] - ni;
    at[x] = o;
    for (int q = 0; q < ni; ++q) {
        ex[n0 + q] = x;
        off[n0 + q] = o + q * ch;
        cn[n0 + q] = c - q * ch < ch ? c - q * ch : ch;
    }
    if (x == DS_HL_E - 1) {
        *nx = si[x];
    }
    __syncthreads();
    for (int p = threadIdx.x; p < np; p += blockDim.x) {
        int xp = idx[p];
        if (xp >= 0 && map[xp] >= 0) {
            pl[atomicAdd(&at[xp], 1)] = p;
        }
    }
}

/* k_ds_hot_tc's sums (the same bits) on the items of k_ds_hot_lists (no scan of the pairs; the
 * items z, z + Z, ...: blockIdx.z, or y for the down rows, of Z in the grid), so the pairs of one
 * busy expert run in parallel. A warp for RG groups of 16 rows (a block of 4 warps for 64 RG rows),
 * the item's pairs PT tiles of 8 at a time: the A fragments of a block of 32 columns decoded once
 * for the tiles (k_ds_hot_tc decoded them again for each tile of 8, in each of its warps). The
 * block copies the weights of its rows and the x of the pairs for KB blocks of 32 columns at a time
 * (a stage) to shared memory, S stages in flight (cp.async): the warps no longer wait on each
 * block's loads (7.7 of 11.4 cycles an instruction without). The items of up to 8 PT pairs; KB
 * divides the blocks of a row and is a multiple of 4. Dynamic shared memory: ds_tc6::smem. */
template <int PT, int RG, int KB, int S>
struct ds_tc6 {
    static constexpr int NP = 8 * PT;
    static constexpr int WB = 4 * RG * KB * MXX_BB;     /* the weights of a stage */
    static constexpr int XR = KB * 32 + 16;              /* a pair's x of a stage, one plane (padded) */
    static constexpr int SB = WB + 2 * NP * XR + NP * KB * 4;
    static constexpr size_t smem = (size_t)S * SB;
};

template <bool DN, int PT, int RG, int KB, int S>
__global__ void __launch_bounds__(128) k_ds_hot_tc6(const gp_rec *r, const int64_t *e, const int *exl,
                                                   const int *offl, const int *cnl, const int *pll,
                                                   const int *nxp)
{
    PDL_START();
    using C = ds_tc6<PT, RG, KB, S>;
    extern __shared__ __align__(16) uint8_t sm[];
    __shared__ int prow[C::NP];
    int k = DI(9), H = DI(10), I = DI(11), nx = *nxp;
    int cols = DN ? I : H, nb = cols / 32, nst = nb / KB, mat = DN ? 0 : blockIdx.y;
    int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, c = lane % 4;
    int G0 = blockIdx.x * 4 * RG;
    const int8_t *qh = DP(const int8_t, DN ? 17 : 12), *ql = DP(const int8_t, DN ? 18 : 13);
    const float *xs = DP(const float, DN ? 19 : 14);
    float *out = DP(float, DN ? 20 : 15);
    for (int z = DN ? blockIdx.y : blockIdx.z; z < nx; z += DN ? gridDim.y : gridDim.z) {
        const uint8_t *wg = ds_slot_w(r, e, DP(const int, 3)[exl[z]]) +
                            (DN ? di(r, e, 7) : (mat ? di(r, e, 6) : 0)) + (size_t)G0 * nb * MXX_BB;
        const int *pl = pll + offl[z];
        int nall = cnl[z];
        for (int q0 = 0; q0 < nall; q0 += C::NP) {
            int n = nall - q0 < C::NP ? nall - q0 : C::NP, nt = (n + 7) / 8;
            __syncthreads();       /* (the last item's stages and rows read) */
            if (tid < C::NP) {
                int pq = tid < n ? pl[q0 + tid] : 0;
                prow[tid] = DN ? pq : pq / k;
            }
            __syncthreads();
            auto issue = [&](int st) {
                if (st < nst) {
                    uint8_t *s = sm + (size_t)(st % S) * C::SB;
                    for (int i = tid; i < 4 * RG * KB * 17; i += 128) {
                        int m = i / (KB * 17), o = i % (KB * 17);
                        cp_async16(s + m * KB * MXX_BB + o * 16,
                                   wg + ((size_t)m * nb + (size_t)st * KB) * MXX_BB + o * 16);
                    }
                    for (int i = tid; i < 2 * n * 2 * KB; i += 128) {
                        int pln = i / (n * 2 * KB), q = i / (2 * KB) % n, o = i % (2 * KB);
                        cp_async16(s + C::WB + (pln * C::NP + q) * C::XR + o * 16,
                                   (pln ? ql : qh) + (size_t)prow[q] * cols + st * KB * 32 + o * 16);
                    }
                    for (int i = tid; i < n * KB / 4; i += 128) {
                        int q = i / (KB / 4), o = i % (KB / 4);
                        cp_async16(s + C::WB + 2 * C::NP * C::XR + q * KB * 4 + o * 16,
                                   xs + (size_t)prow[q] * nb + st * KB + o * 4);
                    }
                }
                cp_async_commit();
            };
            #pragma unroll
            for (int s = 0; s < S - 1; ++s) {
                issue(s);
            }
            float f[RG][PT][4];
            #pragma unroll
            for (int m = 0; m < RG; ++m) {
                #pragma unroll
                for (int u = 0; u < PT; ++u) {
                    f[m][u][0] = f[m][u][1] = f[m][u][2] = f[m][u][3] = 0.f;
                }
            }
            for (int st = 0; st < nst; ++st) {
                cp_async_wait<S - 2>();
                __syncthreads();
                issue(st + S - 1);
                const uint8_t *s = sm + (size_t)(st % S) * C::SB;
                const uint8_t *sx = s + C::WB;
                const float *sxs = (const float *)(s + C::WB + 2 * C::NP * C::XR);
                #pragma unroll
                for (int bb = 0; bb < KB; ++bb) {
                    uint32_t a[RG][4];
                    float e0[RG], e1[RG];
                    #pragma unroll
                    for (int m = 0; m < RG; ++m) {
                        const uint8_t *blk = s + ((warp * RG + m) * KB + bb) * MXX_BB;
                        uint32_t w0 = *(const uint32_t *)(blk + 32 * c + 4 * g);
                        uint32_t w1 = *(const uint32_t *)(blk + 128 + 32 * c + 4 * g);
                        a[m][0] = kt_e2m1x4(w0);
                        a[m][1] = kt_e2m1x4(w0 >> 4);
                        a[m][2] = kt_e2m1x4(w1);
                        a[m][3] = kt_e2m1x4(w1 >> 4);
                        e0[m] = e8m0f(blk[256 + g]);
                        e1[m] = e8m0f(blk[256 + g + 8]);
                    }
                    #pragma unroll
                    for (int u = 0; u < PT; ++u) {
                        if (u < nt) {
                            const uint8_t *xh = sx + (8 * u + g) * C::XR + 32 * bb + 4 * c;
                            const uint8_t *xl = xh + C::NP * C::XR;
                            uint32_t bh[2] = {*(const uint32_t *)xh, *(const uint32_t *)(xh + 16)};
                            uint32_t bl[2] = {*(const uint32_t *)xl, *(const uint32_t *)(xl + 16)};
                            float x0 = sxs[(8 * u + 2 * c) * KB + bb], x1 = sxs[(8 * u + 2 * c + 1) * KB + bb];
                            #pragma unroll
                            for (int m = 0; m < RG; ++m) {
                                int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0};
                                mma16832(ch, a[m], bh);
                                mma16832(cl, a[m], bl);
                                f[m][u][0] = fmaf(__fmul_rn((float)(128 * ch[0] + cl[0]), e0[m]), x0, f[m][u][0]);
                                f[m][u][1] = fmaf(__fmul_rn((float)(128 * ch[1] + cl[1]), e0[m]), x1, f[m][u][1]);
                                f[m][u][2] = fmaf(__fmul_rn((float)(128 * ch[2] + cl[2]), e1[m]), x0, f[m][u][2]);
                                f[m][u][3] = fmaf(__fmul_rn((float)(128 * ch[3] + cl[3]), e1[m]), x1, f[m][u][3]);
                            }
                        }
                    }
                }
            }
            cp_async_wait<0>();
            #pragma unroll
            for (int m = 0; m < RG; ++m) {
                #pragma unroll
                for (int u = 0; u < PT; ++u) {
                    int q = 8 * u + 2 * c;
                    int pcs[2] = {q < n ? pl[q0 + q] : -1, q + 1 < n ? pl[q0 + q + 1] : -1};
                    int row0 = 16 * (G0 + warp * RG + m) + g;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        int pq = pcs[i & 1], row = row0 + 8 * (i >> 1);
                        if (u >= nt || pq < 0) {
                            continue;
                        }
                        float v = __fmul_rn(f[m][u][i], 0.5f);
                        if (DN) {
                            out[(size_t)pq * H + row] = v;
                        } else {
                            out[(size_t)pq * 2 * I + mat * I + row] = v;
                        }
                    }
                }
            }
        }
    }
}

/* The form of k_ds_hot_tc6 that the runner takes (the fastest on the real groups of bench/ds_prefill.cu):
 * items of 24 pairs, 3 tiles of 8, 2 groups of 16 rows a warp, 4 blocks of columns a stage, 2 stages;
 * DS_TC6_Z items at a time in the grid. hl: the lists' scratch, ds_hot_hl_ints(tk, E) ints. The lists,
 * then the gate and up rows (dn 0) or the down rows (dn 1). */
#define DS_TC6_CH 24
#define DS_TC6_Z 128
typedef ds_tc6<3, 2, 4, 2> ds_tc6_run;

static size_t ds_hot_hl_ints(size_t tk, size_t E)
{
    return 3 * (tk + E) + tk + 1;
}

static int ds_hot_tc6_fits(int64_t H, int64_t I, int64_t E)
{
    return H % 128 == 0 && I % 128 == 0 && (H / 32) % 4 == 0 && (I / 32) % 4 == 0 && E <= DS_HL_E;
}

static void ds_hot_tc6(int dn, const gp_rec *dr, const int64_t *denv, const int *idx, const int *map, int *hl,
                       int64_t tk, int64_t E, int64_t H, int64_t I, cudaStream_t s)
{
    size_t ni = (size_t)(tk + E);
    int *ex = hl, *off = hl + ni, *cn = hl + 2 * ni, *pl = hl + 3 * ni, *nx = pl + tk;
    if (!dn) {
        k_ds_hot_lists<<<1, DS_HL_E, 0, s>>>(idx, map, (int)tk, (int)E, DS_TC6_CH, ex, off, cn, pl, nx);
        k_ds_hot_tc6<false, 3, 2, 4, 2><<<dim3((unsigned)(I / 128), 2, DS_TC6_Z), 128, ds_tc6_run::smem, s>>>(
            dr, denv, ex, off, cn, pl, nx);
    } else {
        k_ds_hot_tc6<true, 3, 2, 4, 2><<<dim3((unsigned)(H / 128), DS_TC6_Z), 128, ds_tc6_run::smem, s>>>(
            dr, denv, ex, off, cn, pl, nx);
    }
}
