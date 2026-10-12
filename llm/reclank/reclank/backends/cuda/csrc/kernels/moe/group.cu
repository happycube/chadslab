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
