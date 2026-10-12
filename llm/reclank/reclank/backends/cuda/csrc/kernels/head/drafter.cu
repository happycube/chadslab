/* ---------- the MTP drafter (reclank/backends/cuda/runtime.py, GPUDrafter) ----------
 * GP_F32_LINEAR: x, w, out, rows, cols. A float32 matrix on one row: one
 * warp for each row.
 *
 * GP_DRAFT_HEAD: u, clog, order, head, sel, top, token, n_cent, per, top_k,
 * hidden[, prob]. The centroid head of the drafter, as Assistant.logits
 * does it. The array clog holds the logits of the centroids. The kernels
 * select the top_k centroids. Then they compute the logits of the tokens of
 * those centroids: order holds per tokens for each centroid, and head is
 * the float32 table. Last, they write the best token, and (optional) prob
 * its softmax probability among those tokens (RECLANK_MTP_STOP). */
__global__ void k_f32_linear(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32, rows = DI(3), cols = DI(4);
    if (row >= rows) {
        return;
    }
    const float *w = DP(const float, 1) + (size_t)row * cols;
    const float *x = DP(const float, 0);
    float s2 = 0.f;
    for (int k = lane; k < cols; k += 32) {
        s2 += w[k] * x[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s2 += __shfl_xor_sync(RC_FULL_MASK, s2, o);
    }
    if (lane == 0) {
        DP(float, 2)[row] = s2;
    }
}

/* The top_k centroids: a bitonic sort of (logit, index) in shared memory,
 * largest first. One block of 1024 threads; n_cent is at most 2048. */
__global__ void k_draft_top(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float kv[2048];
    __shared__ int ki[2048];
    const float *clog = DP(const float, 1);
    int n = DI(7), top_k = DI(9);
    int size = 1;
    while (size < n) {
        size <<= 1;
    }
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
        kv[i] = i < n ? clog[i] : -INFINITY;
        ki[i] = i;
    }
    __syncthreads();
    for (int k = 2; k <= size; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < size; i += blockDim.x) {
                int l = i ^ j;
                if (l > i) {
                    bool desc = (i & k) == 0;
                    bool swap = desc ? (kv[i] < kv[l] || (kv[i] == kv[l] && ki[i] > ki[l]))
                                     : (kv[i] > kv[l] || (kv[i] == kv[l] && ki[i] < ki[l]));
                    if (swap) {
                        float tv = kv[i];
                        kv[i] = kv[l];
                        kv[l] = tv;
                        int ti = ki[i];
                        ki[i] = ki[l];
                        ki[l] = ti;
                    }
                }
            }
            __syncthreads();
        }
    }
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
        DP(int, 5)[i] = ki[i];
    }
}

/* The logit of each candidate token: one warp for each. */
__global__ void k_draft_sel(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int i = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32, per = DI(8), top_k = DI(9), hidden = DI(10);
    if (i >= top_k * per) {
        return;
    }
    int tok = DP(const int, 2)[(size_t)DP(const int, 5)[i / per] * per + i % per];
    const float *hrow = DP(const float, 3) + (size_t)tok * hidden;
    const float *u = DP(const float, 0);
    float s2 = 0.f;
    for (int k = lane; k < hidden; k += 32) {
        s2 += hrow[k] * u[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s2 += __shfl_xor_sync(RC_FULL_MASK, s2, o);
    }
    if (lane == 0) {
        DP(float, 4)[i] = s2;
    }
}

/* The best candidate: one block. At an equal logit, the first candidate. */
__global__ void k_draft_best(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    int per = DI(8), n = DI(9) * per;
    const float *sel = DP(const float, 4);
    float v = -INFINITY;
    int b = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (sel[i] > v) {
            v = sel[i];
            b = i;
        }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = b;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) {
            float ov = bv[threadIdx.x + s2];
            int oi = bi[threadIdx.x + s2];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int i = bi[0];
        DP(int, 6)[0] = DP(const int, 2)[(size_t)DP(const int, 5)[i / per] * per + i % per];
    }
    if (r->tag[11] != GP_T_NONE) {
        float m = bv[0], s = 0.f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            s += expf(sel[i] - m);
        }
        __syncthreads();
        bv[threadIdx.x] = s;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (threadIdx.x < s2) {
                bv[threadIdx.x] += bv[threadIdx.x + s2];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            DP(float, 11)[0] = 1.f / bv[0];
        }
    }
}

/* GP_ARGMAX: x, n, out[, prob]. out[0] gets the index of the largest of the
 * n values of x. At an equal value, the lower index wins, as np.argmax. prob
 * (optional) gets the softmax probability of that value, 1 / sum exp(x -
 * max): the confidence of an MTP draft (RECLANK_MTP_STOP). One block. */
/* GP_SOFTCAP: x, n, cap. x = tanh(x / cap) cap in place (the logits). */
__global__ void k_softcap(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    float *x = DP(float, 0);
    size_t n = (size_t)di(r, e, 1), i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float cap = df(r, e, 2);
    if (i < n) {
        x[i] = tanhf(x[i] / cap) * cap;
    }
}

__global__ void k_argmax(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float *x = DP(const float, 0);
    int n = DI(1);
    float v = -INFINITY;
    int b = n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        if (x[i] > v) {
            v = x[i];
            b = i;
        }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = b;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) {
            float ov = bv[threadIdx.x + s2];
            int oi = bi[threadIdx.x + s2];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        DP(int, 2)[0] = bi[0];
    }
    if (r->tag[3] != GP_T_NONE) {
        float m = bv[0], s = 0.f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            s += expf(x[i] - m);
        }
        __syncthreads();
        bv[threadIdx.x] = s;
        __syncthreads();
        for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
            if (threadIdx.x < s2) {
                bv[threadIdx.x] += bv[threadIdx.x + s2];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            DP(float, 3)[0] = 1.f / bv[0];
        }
    }
}

/* The row tok[0] of a Q4_0 table (the token table of the target, the blocks
 * of its head on the GPU), times scale, into out (cols values): the input of
 * an MTP draft step. The values are those of the host (dequantize_int4 or
 * q4_0_rows, then the embedding scale): (q - 8) d, then times scale, in
 * float32. The drafts of a round then run with no copy to the host between
 * them. */
__global__ void k_embed_q4(const uint8_t *table, const int *tok, float *out, int cols, float scale)
{
    PDL_START();
    const uint8_t *row = table + (size_t)tok[0] * (cols / 32) * 18;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint8_t *blk = row + (size_t)(c / 32) * 18;
        int j = c % 32;
        int q = j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4;
        float d = __half2float(__ushort_as_half((uint16_t)(blk[0] | (blk[1] << 8))));
        out[c] = ((float)(q - 8) * d) * scale;
    }
}

/* k_embed_q4 for a block of rows: block j writes the row tok[j] of the
 * Q4_0 table, times scale, into out row j; a negative token writes zeros
 * (the padding of a group). The values are those of Model.embed. */
__global__ void k_embed_q4_rows(const uint8_t *table, const int *tok, float *out, int cols,
                                float scale)
{
    PDL_START();
    int t = tok[blockIdx.x];
    float *o = out + (size_t)blockIdx.x * cols;
    if (t < 0) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            o[c] = 0.f;
        }
        return;
    }
    const uint8_t *row = table + (size_t)t * (cols / 32) * 18;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const uint8_t *blk = row + (size_t)(c / 32) * 18;
        int j = c % 32;
        int q = j < 16 ? blk[2 + j] & 15 : blk[2 + j - 16] >> 4;
        float d = __half2float(__ushort_as_half((uint16_t)(blk[0] | (blk[1] << 8))));
        o[c] = ((float)(q - 8) * d) * scale;
    }
}

/* The best token of each row of x (rows of vocab values), as k_argmax: at an
 * equal value the lower index wins, as np.argmax. A block for each row. The
 * greedy pick of a step reads these, not the logits. */
__global__ void k_argmax_rows(const float *x, int vocab, int *out)
{
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    const float *row = x + (size_t)blockIdx.x * vocab;
    float v = -INFINITY;
    int b = vocab;
    for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
        if (row[i] > v) {
            v = row[i];
            b = i;
        }
    }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = b;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if (threadIdx.x < s2) {
            float ov = bv[threadIdx.x + s2];
            int oi = bi[threadIdx.x + s2];
            if (ov > bv[threadIdx.x] || (ov == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = ov;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[blockIdx.x] = bi[0];
    }
}
