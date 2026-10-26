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
    /* (bit 0 of the address: the expert in Q4S, int4_blk; fill_tables of reclank/backends/cuda/runtime.py) */
    const bool q4s = wtab[ex] & 1;
    const uint8_t *we = (const uint8_t *)(intptr_t)(wtab[ex] & ~(int64_t)1);
    float acc[4][4] = {{0.f}};
    int groups = cols / 32;
    for (int g = 0; g < groups; ++g) {
        /* x: a float4 a thread (the 32 values of GM rows); the int4 rows of w:
         * every thread dequantizes 8 values (4 bytes of a block: the low and
         * the high 4 bits), as w_block32. A thread for each row of w (64 of
         * the 256) took 51.7 ms for the gate and up of a group of 1024 tokens of
         * the 26B on the GTX 1070, 25.9 ms so; the sums are those of before
         * (the same bits). */
        for (int x = threadIdx.x; x < GM * 8; x += blockDim.x) {
            int qq = x / 8, k4 = (x % 8) * 4;
            int q = q0 + qq;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (q < qend) {
                int src = gather ? pair_tok[q] : q;
                v = *(const float4 *)(a + (size_t)src * cols + (size_t)g * 32 + k4);
            }
            xs[k4][qq] = v.x;
            xs[k4 + 1][qq] = v.y;
            xs[k4 + 2][qq] = v.z;
            xs[k4 + 3][qq] = v.w;
        }
        for (int x = threadIdx.x; x < GN * 4; x += blockDim.x) {
            int rr = x / 4, c = x % 4, n = n0 + rr;
            float v[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
            if (n < rows) {
                const uint8_t *wr = we + (size_t)n * groups * 18;
                const uint8_t *blk = q4s ? wr + 16 * g - 2 : wr + (size_t)g * 18;
                const uint8_t *dp = q4s ? wr + cols / 2 + 2 * g : blk;
                float d = __half2float(__ushort_as_half((uint16_t)(dp[0] | (dp[1] << 8))));
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    int bq = blk[2 + 4 * c + i];
                    v[i] = (float)((bq & 15) - 8) * d;
                    v[4 + i] = (float)((bq >> 4) - 8) * d;
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                ws[4 * c + i][rr] = v[i];
                ws[16 + 4 * c + i][rr] = v[4 + i];
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

/* k_moe_gemm without int8 products (Maxwell; MAXWELL_PLAN.md, section 3.4),
 * as k_gemm8 (gemm/groups.cu): a tile of the GM pairs of k_moe_tiles by 128
 * rows a block of 256 threads, 4 pairs by 8 rows a thread (rows 4 tx .. and
 * 64 + 4 tx ..: the lanes of a warp read runs of shared memory), the int4
 * block of each row decoded once a step by 2 threads (i4m, then the scale:
 * the values of k_moe_gemm), the loads of the next step in registers while
 * the tile works. The terms of each value add in the order of k_moe_gemm
 * with the same FMA: the same bits. */
/* TP the pairs of a tile: 64 (GM), or 32 for the last tile of an expert with at most 32 (a 1024-token prompt
 * of the 26B: the tiles of 64 were 64% full, of 64 and 32 ...); SPLIT: the 64 form leaves those to the 32 */
template <int TP = GM, bool SPLIT = false>
__global__ void __launch_bounds__(256, 2) k_moe_gemm8(const gp_rec *r, const int64_t *e, const float *a,
                                                      int gather, const int64_t *wtab, int rows, int cols,
                                                      float *out)
{
    PDL_START();
    constexpr int NL = TP / 32, PT = TP / 16;      /* the float4 of x a thread loads; its pairs */
    __shared__ __align__(16) float xs[32][TP + 4];
    __shared__ __align__(16) float ws[32][G8_LD];
    const int *tiles = DP(const int, 16);
    const int *off = DP(const int, 12);
    const int *pair_tok = DP(const int, 14);
    const int tile = blockIdx.y;
    if (tile >= tiles[0]) {
        return;
    }
    const int ex = tiles[1 + 2 * tile], q0 = tiles[2 + 2 * tile], qend = off[ex + 1];
    if ((TP == 32 && qend - q0 > 32) || (TP == 64 && SPLIT && qend - q0 <= 32)) {
        return;
    }
    const int n0 = blockIdx.x * G8_B;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    /* (bit 0 of the address: the expert in Q4S, as k_moe_gemm) */
    const bool q4s = wtab[ex] & 1;
    const uint8_t *we = (const uint8_t *)(intptr_t)(wtab[ex] & ~(int64_t)1);
    const int groups = cols / 32;
    const int wrow = tid / 2, half = tid & 1;
    const bool wok = n0 + wrow < rows;
    /* the rows of A of the 2 float4 of a thread (pair x / 8, columns 4 (x % 8)) */
    int src[NL];
    #pragma unroll
    for (int i = 0; i < NL; ++i) {
        int q = q0 + (tid + 256 * i) / 8;
        src[i] = q < qend ? (gather ? pair_tok[q] : q) : -1;
    }
    float4 lx[NL];
    uint32_t lw0, lw1;
    float ld;
    auto fetch = [&](int g) {
        #pragma unroll
        for (int i = 0; i < NL; ++i) {
            lx[i] = src[i] >= 0 ? *(const float4 *)(a + (size_t)src[i] * cols + (size_t)g * 32 + 4 * ((tid + 256 * i) % 8))
                                : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (wok && q4s) {
            const uint8_t *wr = we + (size_t)(n0 + wrow) * groups * 18;
            const uint32_t *c = (const uint32_t *)(wr + 16 * g + 8 * half);
            lw0 = c[0];
            lw1 = c[1];
            ld = __half2float(((const __half *)(wr + cols / 2))[g]);
        } else if (wok) {
            const uint8_t *blk = we + ((size_t)(n0 + wrow) * groups + g) * 18;
            const uint16_t *c = (const uint16_t *)(blk + 2 + 8 * half);
            lw0 = (uint32_t)c[0] | ((uint32_t)c[1] << 16);
            lw1 = (uint32_t)c[2] | ((uint32_t)c[3] << 16);
            ld = __half2float(__ushort_as_half(*(const uint16_t *)blk));
        } else {
            lw0 = lw1 = 0x88888888u;
            ld = 0.f;
        }
    };
    float acc[PT][8];
    #pragma unroll
    for (int u = 0; u < PT; ++u) {
        #pragma unroll
        for (int v = 0; v < 8; ++v) {
            acc[u][v] = 0.f;
        }
    }
    fetch(0);
    for (int g = 0; g < groups; ++g) {
        #pragma unroll
        for (int i = 0; i < NL; ++i) {
            int x = tid + 256 * i, qq = x / 8, k = 4 * (x % 8);
            xs[k][qq] = lx[i].x;
            xs[k + 1][qq] = lx[i].y;
            xs[k + 2][qq] = lx[i].z;
            xs[k + 3][qq] = lx[i].w;
        }
        {
            const int kb = 8 * half;
            const uint32_t h0 = lw0 >> 12, h1 = lw1 >> 12;
            ws[kb + 0][wrow] = i4m<0>(lw0) * ld;
            ws[kb + 1][wrow] = i4m<8>(lw0) * ld;
            ws[kb + 2][wrow] = i4m<16>(lw0) * ld;
            ws[kb + 3][wrow] = i4m<12>(h0) * ld;
            ws[kb + 4][wrow] = i4m<0>(lw1) * ld;
            ws[kb + 5][wrow] = i4m<8>(lw1) * ld;
            ws[kb + 6][wrow] = i4m<16>(lw1) * ld;
            ws[kb + 7][wrow] = i4m<12>(h1) * ld;
            ws[16 + kb + 0][wrow] = i4m<4>(lw0) * ld;
            ws[16 + kb + 1][wrow] = i4m<12>(lw0) * ld;
            ws[16 + kb + 2][wrow] = i4m<8>(h0) * ld;
            ws[16 + kb + 3][wrow] = i4m<16>(h0) * ld;
            ws[16 + kb + 4][wrow] = i4m<4>(lw1) * ld;
            ws[16 + kb + 5][wrow] = i4m<12>(lw1) * ld;
            ws[16 + kb + 6][wrow] = i4m<8>(h1) * ld;
            ws[16 + kb + 7][wrow] = i4m<16>(h1) * ld;
        }
        __syncthreads();
        if (g + 1 < groups) {
            fetch(g + 1);
        }
        #pragma unroll 4
        for (int k = 0; k < 32; ++k) {
            float av[PT];
            if constexpr (PT == 4) {
                const float4 a0 = *(const float4 *)&xs[k][4 * ty];
                av[0] = a0.x;
                av[1] = a0.y;
                av[2] = a0.z;
                av[3] = a0.w;
            } else {
                const float2 a0 = *(const float2 *)&xs[k][2 * ty];
                av[0] = a0.x;
                av[1] = a0.y;
            }
            const float4 b0 = *(const float4 *)&ws[k][4 * tx], b1 = *(const float4 *)&ws[k][64 + 4 * tx];
            const float bv[8] = {b0.x, b0.y, b0.z, b0.w, b1.x, b1.y, b1.z, b1.w};
            #pragma unroll
            for (int u = 0; u < PT; ++u) {
                #pragma unroll
                for (int v = 0; v < 8; ++v) {
                    acc[u][v] = fmaf(av[u], bv[v], acc[u][v]);
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int u = 0; u < PT; ++u) {
        int q = q0 + PT * ty + u;
        if (q >= qend) {
            continue;
        }
        #pragma unroll
        for (int hv = 0; hv < 2; ++hv) {
            int n = n0 + 64 * hv + 4 * tx;
            float *o = out + (size_t)q * rows + n;
            if (n + 3 < rows && (rows & 3) == 0) {
                *(float4 *)o = make_float4(acc[u][4 * hv], acc[u][4 * hv + 1], acc[u][4 * hv + 2], acc[u][4 * hv + 3]);
            } else {
                for (int v = 0; v < 4; ++v) {
                    if (n + v < rows) {
                        o[v] = acc[u][4 * hv + v];
                    }
                }
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
