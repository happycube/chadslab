/* ---------- the hot experts ----------
 * The GPU holds some experts of each layer (SPLIT_PLAN.md, the hot experts).
 * The array map of a layer gives the slot of each expert on the GPU, or -1.
 *
 * GP_HOT_SPLIT: idx, val, map, cold_idx, cold_val, top_k. Some selected
 * experts are not on the GPU. Write them to cold_idx, and their weights to
 * cold_val. cold_idx[top_k] gets their count. The CPU computes
 * these experts (GP_MOE_N of the CPU interpreter). With 2 top_k + 1 values,
 * cold_idx[top_k + 1 + j] also gets idx[j]: the host reads the selection of
 * the step for the cache of hot experts (ModelGPU, HotCache). */
__device__ void hot_split_body(const gp_rec *r, const int64_t *e)
{
    const int *idx = DP(const int, 0);
    const float *val = DP(const float, 1);
    const int *map = DP(const int, 2);
    int *cold = DP(int, 3);
    float *cold_val = DP(float, 4);
    int top_k = DI(5), n = 0;
    /* Operands 7 and 8 (or none): zc (top_k ints) and nzc. The first nzc
     * cold experts go to the GPU, which reads them from the pinned copy of
     * the experts in host memory (zc[j] = 1; k_kqh_nvx), not to the CPU. */
    int *zc = r->tag[7] == GP_T_NONE ? NULL : DP(int, 7);
    int nzc = zc != NULL ? DI(8) : 0, z = 0;
    for (int j = 0; j < top_k; ++j) {
        int to_gpu = 0;
        if (map[idx[j]] < 0) {
            if (z < nzc) {
                to_gpu = 1;
                ++z;
            } else {
                cold[n] = idx[j];
                cold_val[n] = val[j];
                ++n;
            }
        }
        if (zc != NULL) {
            zc[j] = to_gpu;
        }
    }
    cold[top_k] = n;
    if (DI(6)) {
        for (int j = 0; j < top_k; ++j) {
            cold[top_k + 1 + j] = idx[j];
        }
    }
}

__global__ void k_hot_split(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    hot_split_body(r, e);
}

/* The operands of GP_HOT_MOE:
 *
 *     h, val, idx, map, gu, dn, act, act2, de, out, top_k, gu_rows, cols,
 *     dn_rows, inner, t
 *
 * The record serves a group of t tokens too: h is (t, cols), and val and idx
 * are (t, top_k). Pair j is slot j % top_k of token j / top_k.
 *
 * The record computes the selected experts that the GPU holds. Four kernels
 * do the steps:
 *
 * 1. the gate and up rows into act;
 * 2. the GELU into act2;
 * 3. the down rows into de;
 * 4. the sum with the router weights into out.
 *
 * A block of an expert that the GPU does not hold does nothing. The arrays
 * gu and dn hold the int4 blocks of the experts of the layer, one expert
 * after the other. */
/* A KQ_Q4X group (backends/cpu/csrc/kernels/kq; the int4 experts of the Gemma 4 26B in
 * groups of 16 rows: for each block of 32 columns, 8 steps of 32 bytes of
 * codes q = w + 8, then the float16 scales of the 16 rows) on float32 x.
 * Lane l of a warp reads 8 bytes of each block: step l / 4, rows 2q, 2q + 1
 * (low 4 bits) and 2q + 8, 2q + 9 (high), q = l % 4. */
__device__ __forceinline__ float q4x_dot4(uint32_t a, float4 x)
{
    return (float)(int8_t)(a & 255) * x.x + (float)(int8_t)((a >> 8) & 255) * x.y +
           (float)(int8_t)((a >> 16) & 255) * x.z + (float)(int8_t)(a >> 24) * x.w;
}

/* The 16 rows of a KQ_Q4X group by the ROWS_PER_BLOCK warps of a block (the
 * hot experts of a step): warp w takes the blocks of 32 columns w, w + 8,
 * ..., and the warps add their sums in shared memory. Then
 * thread i < 16 stores row i into out. A block has many more loads in
 * flight than one warp for each group. */
__device__ void q4x_group_block(const uint8_t *wg, const float *x, int cols, float *out)
{
    __shared__ float red[ROWS_PER_BLOCK][16];
    int w = threadIdx.x / 32, lane = threadIdx.x % 32, st = lane / 4, q = lane % 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 2
    for (int b = w; b < cols / 32; b += ROWS_PER_BLOCK) {
        const uint8_t *blk = wg + (size_t)b * 288;
        uint2 c = *(const uint2 *)(blk + 8 * lane);
        float4 xv = *(const float4 *)(x + 32 * b + 4 * st);
        const __half *d = (const __half *)(blk + 256);
        acc[0] += __half2float(d[2 * q]) * q4x_dot4(__vsub4(c.x & 0x0f0f0f0fu, 0x08080808u), xv);
        acc[1] += __half2float(d[2 * q + 1]) * q4x_dot4(__vsub4(c.y & 0x0f0f0f0fu, 0x08080808u), xv);
        acc[2] += __half2float(d[2 * q + 8]) *
                  q4x_dot4(__vsub4((c.x >> 4) & 0x0f0f0f0fu, 0x08080808u), xv);
        acc[3] += __half2float(d[2 * q + 9]) *
                  q4x_dot4(__vsub4((c.y >> 4) & 0x0f0f0f0fu, 0x08080808u), xv);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int o = 4; o < 32; o <<= 1) {
            acc[i] += __shfl_xor_sync(RC_FULL_MASK, acc[i], o);
        }
    }
    if (lane < 4) {
        red[w][2 * lane] = acc[0];
        red[w][2 * lane + 1] = acc[1];
        red[w][2 * lane + 8] = acc[2];
        red[w][2 * lane + 9] = acc[3];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float s = 0.f;
        #pragma unroll
        for (int k = 0; k < ROWS_PER_BLOCK; ++k) {
            s += red[k][threadIdx.x];
        }
        out[threadIdx.x] = s;
    }
}

__device__ __forceinline__ int hot_slot(const gp_rec *r, const int64_t *e, int j)
{
    return DP(const int, 3)[DP(const int, 2)[j]];
}

/* The bytes of the expert of slot, part 0 (gate and up: operand 4) or 1
 * (down: operand 5), per bytes an expert: the store of the layer, or a block
 * of the pool of HotCache (slot >= KQH_WARM). */
__device__ __forceinline__ const uint8_t *hot_w(const gp_rec *r, const int64_t *e, int slot, int part,
                                                size_t per)
{
    if (slot >= KQH_WARM) {
        return pool_block(slot, part);
    }
    return DP(const uint8_t, 4 + part) + (size_t)slot * per;
}

__global__ void k_hot_gu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.y, slot = hot_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int rows = DI(11), cols = DI(12);
    size_t rb = (size_t)(cols / 32) * 18;
    if (DI(16) == 1) {
        /* KQ_Q4X experts (operand 16): block x is a group of 16 rows */
        int grp = blockIdx.x;
        if (slot < 0 || grp >= rows / 16) {
            return;
        }
        q4x_group_block(hot_w(r, e, slot, 0, (size_t)rows * rb) + 16 * (size_t)grp * rb,
                        DP(const float, 0) + (size_t)(j / DI(10)) * cols, cols,
                        DP(float, 6) + (size_t)j * rows + 16 * grp);
        return;
    }
    if (slot < 0 || row >= rows) {
        return;
    }
    float v = int4_row(hot_w(r, e, slot, 0, (size_t)rows * rb) + (size_t)row * rb,
                       DP(const float, 0) + (size_t)(j / DI(10)) * cols, cols);
    if (threadIdx.x % 32 == 0) {
        DP(float, 6)[(size_t)j * rows + row] = v;
    }
}

__global__ void k_hot_gelu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x;
    if (hot_slot(r, e, j) < 0) {
        return;
    }
    int inner = DI(14);
    const float *g = DP(const float, 6) + (size_t)j * 2 * inner;
    float *o = DP(float, 7) + (size_t)j * inner;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = g[i];
        o[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)))
               * g[inner + i];
    }
}

__global__ void k_hot_dn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.y, slot = hot_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int rows = DI(13), inner = DI(14);
    size_t rb = (size_t)(inner / 32) * 18;
    if (DI(16) == 1) {
        int grp = blockIdx.x;
        if (slot < 0 || grp >= rows / 16) {
            return;
        }
        q4x_group_block(hot_w(r, e, slot, 1, (size_t)rows * rb) + 16 * (size_t)grp * rb,
                        DP(const float, 7) + (size_t)j * inner, inner,
                        DP(float, 8) + (size_t)j * rows + 16 * grp);
        return;
    }
    if (slot < 0 || row >= rows) {
        return;
    }
    float v = int4_row(hot_w(r, e, slot, 1, (size_t)rows * rb) + (size_t)row * rb,
                       DP(const float, 7) + (size_t)j * inner, inner);
    if (threadIdx.x % 32 == 0) {
        DP(float, 8)[(size_t)j * rows + row] = v;
    }
}

__global__ void k_hot_sum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int tok = blockIdx.y;
    int rows = DI(13), top_k = DI(10);
    if (c >= rows) {
        return;
    }
    const float *val = DP(const float, 1);
    const float *de = DP(const float, 8);
    float acc = 0.f;
    for (int s2 = 0; s2 < top_k; ++s2) {
        int j = tok * top_k + s2;
        if (hot_slot(r, e, j) >= 0) {
            acc += val[j] * de[(size_t)j * rows + c];
        }
    }
    DP(float, 9)[(size_t)tok * rows + c] = acc;
}

/* GP_HOT_SPLIT_MT: idx, val, map, cold_idx, cold_val, pairs. Some pairs of
 * a group use an expert that the GPU does not hold. For such a pair, write
 * the expert and its weight. For the other pairs, write -1 and 0. GP_MOE_MT of the CPU skips the pairs of -1
 * (the CPU part), and GP_HOT_MOE skips the other pairs (the GPU part).
 * Operand 8 (or none): gpu_idx gets idx with -1 for the experts that the
 * GPU does not hold, for GP_MOE_GPU (the mixed groups of a prompt). */
__global__ void k_hot_split_mt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= DI(5)) {
        return;
    }
    int x = DP(const int, 0)[p];
    bool cold = DP(const int, 2)[x] < 0;
    /* With operands 6 and 7 (nreal, top_k): the pairs of the padding rows
     * go to neither part. */
    int k = r->tag[7] == GP_T_NONE ? 0 : DI(7);
    if (k > 0 && DI(6) > 0 && p >= DI(6) * k) {
        cold = false;
    }
    /* Operands 9 and 10 (or none): zc (a flag for each pair) and nzc. The
     * first nzc cold experts of the group (in the order of their first
     * pair) go to the GPU with all their pairs (k_kqh_nvx reads them from
     * the pinned copy in host memory), not to the CPU. */
    int to_gpu = 0;
    if (cold && r->tag[9] != GP_T_NONE && DI(10) > 0) {
        const int *ix = DP(const int, 0), *map = DP(const int, 2);
        int lim = (k > 0 && DI(6) > 0) ? DI(6) * k : DI(5);
        int first = p;
        for (int q = 0; q < p; ++q) {
            if (ix[q] == x) {
                first = q;
                break;
            }
        }
        int rank = 0;
        for (int q = 0; q < first && q < lim; ++q) {
            if (map[ix[q]] >= 0) {
                continue;
            }
            int dup = 0;
            for (int u = 0; u < q; ++u) {
                if (ix[u] == ix[q]) {
                    dup = 1;
                    break;
                }
            }
            rank += !dup;
        }
        to_gpu = rank < DI(10);
    }
    if (r->tag[9] != GP_T_NONE) {
        DP(int, 9)[p] = to_gpu;
    }
    if (to_gpu) {
        cold = false;
    }
    DP(int, 3)[p] = cold ? x : -1;
    DP(float, 4)[p] = cold ? DP(const float, 1)[p] : 0.f;
    if (r->tag[8] != GP_T_NONE && DP(int, 8) != NULL) {
        DP(int, 8)[p] = DP(const int, 2)[x] >= 0 ? x : -1;
    }
}
