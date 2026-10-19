/* ---------- the LiLiCorr head of Nemotron 3.5 Bolt's drafter on the GPU ----------
 * (reclank/models/nemotron/lilicorr.py: LiLiCorr.lattice is the reference; the CPU's records in
 * backends/cpu/csrc/common/nemotron.c.) The candidates of each slot (NM_LTOPK), the attention of
 * its 2 layers over the 56 candidates with the slot biases (NM_LATT), the rows plus a bias and an
 * activation (NM_LROWS), [x, anchor, x anchor] (NM_LCAT), and the scores of the lattice
 * (NM_LSCORE). The matrices are the records of the drafter (KQ_NV4 rows, ENC_LINEAR of bfloat16). */

#define NM_LK 8                     /* the candidates of a slot at most */
#define NM_LN 64                    /* the rows of the head at most (slots x candidates) */

/* a before b: the larger logit, then the smaller index (the reference's lexsort) */
__device__ __forceinline__ bool nm_lt_before(float av, int ai, float bv, int bi)
{
    return av > bv || (av == bv && ai < bi);
}

/* the best (v, i) of the warp's heads */
__device__ __forceinline__ void nm_lt_warp_best(float *v, int *i)
{
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(RC_FULL_MASK, *v, o);
        const int oi = __shfl_xor_sync(RC_FULL_MASK, *i, o);
        if (nm_lt_before(ov, oi, *v, *i)) {
            *v = ov;
            *i = oi;
        }
    }
}

/* the k best of the sorted lists (tv, ti) of the lanes: k rounds, the winner's list moves up */
__device__ __forceinline__ void nm_lt_merge(float (&tv)[NM_LK], int (&ti)[NM_LK], int k, float *ov, int *oi)
{
    for (int q = 0; q < k; ++q) {
        float bv = tv[0];
        int bi = ti[0];
        nm_lt_warp_best(&bv, &bi);
        ov[q] = bv;
        oi[q] = bi;
        if (ti[0] == bi) {
#pragma unroll
            for (int u = 0; u < NM_LK - 1; ++u) {
                tv[u] = tv[u + 1];
                ti[u] = ti[u + 1];
            }
            tv[NM_LK - 1] = -INFINITY;
            ti[NM_LK - 1] = INT_MAX;
        }
    }
}

/* GP_NM_LTOPK: logits (S x V), S, V, k, cand (S x k int32), feat (S k x 5): the k best of each slot
 * (by logit, then index) and their features [log p, p, log p - log p0, rank / (k - 1), rank == 0]
 * (p of the softmax of the slot). A block a slot: each thread's k best, the warps', the block's. */
__global__ void __launch_bounds__(1024) k_nm_ltopk(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float red[32], wv[32][NM_LK];
    __shared__ int wi[32][NM_LK];
    const int s = blockIdx.x, V = DI(2), k = DI(3), tid = threadIdx.x, lane = tid % 32, w = tid / 32;
    const int nw = blockDim.x / 32;
    const float *x = DP(const float, 0) + (size_t)s * V;
    float tv[NM_LK];
    int ti[NM_LK];
#pragma unroll
    for (int q = 0; q < NM_LK; ++q) {
        tv[q] = -INFINITY;
        ti[q] = INT_MAX;
    }
    float mx = -INFINITY;
    for (int i = tid; i < V; i += blockDim.x) {
        float cv = x[i];
        int ci = i;
        mx = fmaxf(mx, cv);
        if (cv > tv[NM_LK - 1]) {
            /* (the indices of a thread rise: an equal value stays behind the earlier one) */
#pragma unroll
            for (int q = 0; q < NM_LK; ++q) {
                if (cv > tv[q]) {
                    const float t = tv[q];
                    const int u = ti[q];
                    tv[q] = cv;
                    ti[q] = ci;
                    cv = t;
                    ci = u;
                }
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, o));
    }
    if (lane == 0) {
        red[w] = mx;
    }
    __syncthreads();
    mx = red[0];
    for (int q = 1; q < nw; ++q) {
        mx = fmaxf(mx, red[q]);
    }
    float sum = 0.f;
    for (int i = tid; i < V; i += blockDim.x) {
        sum += expf(x[i] - mx);
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    __syncthreads();
    if (lane == 0) {
        red[w] = sum;
    }
    float ov[NM_LK];
    int oi[NM_LK];
    nm_lt_merge(tv, ti, k, ov, oi);
    if (lane == 0) {
        for (int q = 0; q < k; ++q) {
            wv[w][q] = ov[q];
            wi[w][q] = oi[q];
        }
    }
    __syncthreads();
    if (w != 0) {
        return;
    }
    sum = 0.f;
    for (int q = 0; q < nw; ++q) {
        sum += red[q];
    }
    const float lse = mx + logf(sum);
#pragma unroll
    for (int q = 0; q < NM_LK; ++q) {
        tv[q] = lane < nw && q < k ? wv[lane][q] : -INFINITY;
        ti[q] = lane < nw && q < k ? wi[lane][q] : INT_MAX;
    }
    nm_lt_merge(tv, ti, k, ov, oi);
    if (lane < k) {
        float v = ov[0];
        int ix = oi[0];
#pragma unroll
        for (int q = 1; q < NM_LK; ++q) {
            if (lane == q) {
                v = ov[q];
                ix = oi[q];
            }
        }
        const float p = expf(v - lse), lp = logf(p), lp0 = logf(expf(ov[0] - lse));
        DP(int, 4)[s * k + lane] = ix;
        float *f = DP(float, 5) + (size_t)(s * k + lane) * 5;
        f[0] = lp;
        f[1] = p;
        f[2] = lp - lp0;
        f[3] = (float)lane / (float)max(k - 1, 1);
        f[4] = lane == 0 ? 1.f : 0.f;
    }
}

/* GP_NM_LATT: qkv (n x 3H: q, k, v), bias (heads x n x n), out (n x H), n, heads, hd: each row's
 * softmax(q k / sqrt(hd) + bias) v over the n rows, a head. A block a (row, head): a warp a key. */
__global__ void __launch_bounds__(128) k_nm_latt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float qs[128], sc[NM_LN];
    const int i = blockIdx.x, h = blockIdx.y, n = DI(3), heads = DI(4), hd = DI(5);
    const int H = heads * hd, tid = threadIdx.x, lane = tid % 32, w = tid / 32;
    const float *qkv = DP(const float, 0);
    for (int d = tid; d < hd; d += blockDim.x) {
        qs[d] = qkv[(size_t)i * 3 * H + h * hd + d];
    }
    __syncthreads();
    const float rs = sqrtf((float)hd);
    for (int j = w; j < n; j += blockDim.x / 32) {
        const float *kr = qkv + (size_t)j * 3 * H + H + h * hd;
        float a = 0.f;
        for (int d = lane; d < hd; d += 32) {
            a += qs[d] * kr[d];
        }
        for (int o = 16; o > 0; o >>= 1) {
            a += __shfl_xor_sync(RC_FULL_MASK, a, o);
        }
        if (lane == 0) {
            sc[j] = a / rs + DP(const float, 1)[((size_t)h * n + i) * n + j];
        }
    }
    __syncthreads();
    if (w == 0) {
        float m = -INFINITY;
        for (int j = lane; j < n; j += 32) {
            m = fmaxf(m, sc[j]);
        }
        for (int o = 16; o > 0; o >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, o));
        }
        float z = 0.f;
        for (int j = lane; j < n; j += 32) {
            const float v = expf(sc[j] - m);
            sc[j] = v;
            z += v;
        }
        for (int o = 16; o > 0; o >>= 1) {
            z += __shfl_xor_sync(RC_FULL_MASK, z, o);
        }
        __syncwarp();
        for (int j = lane; j < n; j += 32) {
            sc[j] /= z;
        }
    }
    __syncthreads();
    for (int d = tid; d < hd; d += blockDim.x) {
        float a = 0.f;
        for (int j = 0; j < n; ++j) {
            a += sc[j] * qkv[(size_t)j * 3 * H + 2 * H + h * hd + d];
        }
        DP(float, 2)[(size_t)i * H + h * hd + d] = a;
    }
}

/* GP_NM_LROWS: x, b (cols, or none), y (or none), out, rows, cols, rep, act: out = act(x + b + y of
 * row r / rep) (act 1: silu) */
__global__ void k_nm_lrows(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int rows = DI(4), cols = DI(5), rep = DI(6), act = DI(7);
    const int64_t n = (int64_t)rows * cols, i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }
    const int c = (int)(i % cols), row = (int)(i / cols);
    const float *b = DP(const float, 1), *y = DP(const float, 2);
    float v = DP(const float, 0)[i];
    if (b != NULL) {
        v += b[c];
    }
    if (y != NULL) {
        v += y[(size_t)(row / rep) * cols + c];
    }
    DP(float, 3)[i] = act == 1 ? v / (1.f + expf(-v)) : v;
}

/* GP_NM_LCAT: x (n x H), a (H), out (n x 3H), n, H: [x, a, x a] */
__global__ void k_nm_lcat(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int n = DI(3), H = DI(4);
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)n * H) {
        return;
    }
    const int c = (int)(i % H), row = (int)(i / H);
    const float xv = DP(const float, 0)[i], av = DP(const float, 1)[c];
    float *o = DP(float, 2) + (size_t)row * 3 * H;
    o[c] = xv;
    o[H + c] = av;
    o[2 * H + c] = xv * av;
}

/* the norm of a row of D (a warp; max(|v|, eps)) and the dot of two */
__device__ __forceinline__ float nm_ls_dot(const float *a, const float *b, int D, int lane)
{
    float s = 0.f;
    for (int d = lane; d < D; d += 32) {
        s += a[d] * b[d];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, o);
    }
    return s;
}

/* GP_NM_LSCORE: ov, iv (S k x D: the out and in vectors), anc (D), scores (S x k x k), S, k, D, scale,
 * eps: the vectors L2-normed (max(|v|, eps)); scores[0][0][c] = scale in[0][c] anc (the rest of slot
 * 0: 0), scores[s][p][c] = scale out[s - 1][p] in[s][c]. A block a slot. */
__global__ void __launch_bounds__(256) k_nm_lscore(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float nin[NM_LK], nout[NM_LK];
    const int s = blockIdx.x, k = DI(5), D = DI(6), lane = threadIdx.x % 32, w = threadIdx.x / 32;
    const int nw = blockDim.x / 32;
    const float scale = df(r, e, 7), eps = df(r, e, 8);
    const float *ov = DP(const float, 0), *iv = DP(const float, 1), *anc = DP(const float, 2);
    const int np = s == 0 ? 1 : k;
    for (int q = w; q < k + np; q += nw) {
        const float *v = q < k ? iv + (size_t)(s * k + q) * D : s == 0 ? anc : ov + (size_t)((s - 1) * k + q - k) * D;
        const float nv = fmaxf(sqrtf(nm_ls_dot(v, v, D, lane)), eps);
        if (lane == 0) {
            if (q < k) {
                nin[q] = nv;
            } else {
                nout[q - k] = nv;
            }
        }
    }
    __syncthreads();
    float *sc = DP(float, 3) + (size_t)s * k * k;
    for (int q = w; q < k * k; q += nw) {
        const int p = q / k, c = q % k;
        float v = 0.f;
        if (p < np) {
            const float *a = s == 0 ? anc : ov + (size_t)((s - 1) * k + p) * D;
            v = scale * nm_ls_dot(a, iv + (size_t)(s * k + c) * D, D, lane) / (nout[p] * nin[c]);
        }
        if (lane == 0) {
            sc[q] = v;
        }
    }
}
