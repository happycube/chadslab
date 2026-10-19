/* ---------- GLM-5.3-Flash (docs/GLM53_PLAN.md) ----------
 *
 * GP_F16_GLINEAR: x (t x groups cols, float32), w (float16, rows x cols),
 * out (t x rows), rows, cols, t, groups. A float16 matrix in groups (block
 * diagonal): the rows in groups of rows / groups, group g on the columns
 * g cols .. (g + 1) cols - 1 of an x row. The kv_b of GLM's MLA: W_UK^T q
 * (64 groups of 512 x 256: the latent query of each head) and W_UV o (64
 * groups of 256 x 512: the values of each head from its latent output).
 *
 * k_f16g_gemv: a few tokens (t <= 8), a warp a row, each weight read once
 * for all of them. k_f16g_tile: a group, 64 x 64 tiles of out (rows x
 * tokens) in float32 FMA (a 64-row tile in one group; one group: any rows). */
template <int NT>
__global__ void __launch_bounds__(256) k_f16g_gemv(const gp_rec *r, const int64_t *e)
{
    int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    int rows = DI(3), cols = DI(4), t = DI(5), groups = DI(6) > 1 ? DI(6) : 1;
    if (row >= rows) {
        return;
    }
    const uint4 *w = (const uint4 *)(DP(const uint16_t, 1) + (size_t)row * cols);
    PDL_START();
    int g = row / (rows / groups);
    size_t xs = (size_t)groups * cols;
    const float *x = DP(const float, 0) + (size_t)g * cols;
    float sum[NT];
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        sum[j] = 0.f;
    }
    for (int i = lane; i < cols / 8; i += 32) {
        uint4 q = w[i];
        const __half2 *h = (const __half2 *)&q;
        float wv[8];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            float2 f = __half22float2(h[k]);
            wv[2 * k] = f.x;
            wv[2 * k + 1] = f.y;
        }
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                const float4 *xi = (const float4 *)(x + j * xs + 8 * i);
                float4 a = xi[0], b = xi[1];
                sum[j] += wv[0] * a.x + wv[1] * a.y + wv[2] * a.z + wv[3] * a.w +
                          wv[4] * b.x + wv[5] * b.y + wv[6] * b.z + wv[7] * b.w;
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        float s = sum[j];
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(RC_FULL_MASK, s, o);
        }
        if (lane == 0 && j < t) {
            DP(float, 2)[(size_t)j * rows + row] = s;
        }
    }
}

#define F16G_BM 64
#define F16G_BN 64
#define F16G_BK 16
__global__ void __launch_bounds__(256) k_f16g_tile(const gp_rec *r, const int64_t *e)
{
    __shared__ float ws[F16G_BK][F16G_BM + 4], xs_[F16G_BK][F16G_BN + 4];
    int rows = DI(3), cols = DI(4), t = DI(5), groups = DI(6) > 1 ? DI(6) : 1;
    int r0 = blockIdx.x * F16G_BM, j0 = blockIdx.y * F16G_BN;
    int g = r0 / (rows / groups);
    const uint16_t *w = DP(const uint16_t, 1);
    PDL_START();
    const float *x = DP(const float, 0) + (size_t)g * cols;
    size_t xstr = (size_t)groups * cols;
    int tr = threadIdx.x / 16, tc = threadIdx.x % 16;   /* 4 rows x 4 tokens a thread */
    float acc[4][4] = {};
    for (int k0 = 0; k0 < cols; k0 += F16G_BK) {
        /* 64 rows x 16 columns of w, 64 tokens x 16 columns of x */
        for (int z = threadIdx.x; z < F16G_BM * F16G_BK; z += 256) {
            int rr = z / F16G_BK, kk = z % F16G_BK;
            ws[kk][rr] = r0 + rr < rows ? __half2float(__ushort_as_half(w[(size_t)(r0 + rr) * cols + k0 + kk])) : 0.f;
            int jj = rr, j = j0 + jj;
            xs_[kk][jj] = j < t ? x[(size_t)j * xstr + k0 + kk] : 0.f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < F16G_BK; ++kk) {
            float a[4], b[4];
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                a[u] = ws[kk][tr * 4 + u];
                b[u] = xs_[kk][tc * 4 + u];
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                #pragma unroll
                for (int v = 0; v < 4; ++v) {
                    acc[u][v] += a[u] * b[v];
                }
            }
        }
        __syncthreads();
    }
    float *out = DP(float, 2);
    #pragma unroll
    for (int v = 0; v < 4; ++v) {
        int j = j0 + tc * 4 + v;
        if (j < t) {
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                if (r0 + tr * 4 + u < rows) {
                    out[(size_t)j * rows + r0 + tr * 4 + u] = acc[u][v];
                }
            }
        }
    }
}

/* ---------- the indexer of GLM's DSA ----------
 *
 * GP_GLM_POOL: k [t, hd] (wk x; normed in place), gate [t, hd] (the gate of
 * the pools, x times index_kpool_compress_gate), kstate, sstate [ratio, hd]
 * (the keys and gates of the open pool), ape [ratio, hd], pooled [*, hd] (the
 * cache of pooled keys), pos, ratio, hd, form (of pooled: 0 float32, 1
 * bfloat16), t, eps, bk, bs (or null: the state before the group, for a
 * verify group), ln_w, ln_b [hd]. k_glm_ln: the LayerNorm (with its bias) of
 * each key; k_glm_pool: each pool of ratio tokens that ends in the chunk, the
 * sum of its keys with the softmax over its tokens of gate + ape (the
 * columns apart), into pooled row end / ratio (no norm after:
 * Glm5NextTextIndexer.get_pooled_states); then k_ds_compress_state (the same
 * operands 0 to 3, 6 to 8, 10, 12, 13 as GP_DS_COMPRESS): the tokens after
 * the last end into the state. A block of hd threads (hd <= 512). */
__global__ void __launch_bounds__(512) k_glm_ln(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hd = DI(8), j = blockIdx.x, c = threadIdx.x;
    float *k = DP(float, 0) + (size_t)j * hd;
    float v = k[c];
    float mu = block_sum(v) / (float)hd;
    float d = v - mu;
    float var = block_sum(d * d) / (float)hd;
    k[c] = d / sqrtf(var + df(r, e, 11)) * DP(const float, 14)[c] + DP(const float, 15)[c];
}

__global__ void __launch_bounds__(512) k_glm_pool(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ratio = DI(7), hd = DI(8), t = DI(10), g = blockIdx.x, c = threadIdx.x;
    int64_t pos = di(r, e, 6);
    int64_t pe = ds_first_end(pos, ratio) + (int64_t)g * ratio;
    if (pe >= pos + t) {
        return;
    }
    const float *kv = DP(const float, 0), *sc = DP(const float, 1);
    const float *ks = DP(const float, 2), *ss = DP(const float, 3), *ape = DP(const float, 4);
    float kr[8], sr[8], m = -INFINITY;
    for (int i = 0; i < ratio; ++i) {
        int64_t q = pe - ratio + 1 + i;
        kr[i] = q < pos ? ks[(size_t)(q % ratio) * hd + c] : kv[(size_t)(q - pos) * hd + c];
        sr[i] = (q < pos ? ss[(size_t)(q % ratio) * hd + c] : sc[(size_t)(q - pos) * hd + c]) +
                ape[(size_t)i * hd + c];
        m = fmaxf(m, sr[i]);
    }
    float den = 0.f;
    for (int i = 0; i < ratio; ++i) {
        den += expf(sr[i] - m);
    }
    float v = 0.f;
    for (int i = 0; i < ratio; ++i) {
        v += (expf(sr[i] - m) / den) * kr[i];
    }
    size_t at = (size_t)(pe / ratio) * hd + c;
    if (DI(9) == 1) {
        uint32_t u = __float_as_uint(v);
        DP(uint16_t, 5)[at] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
    } else {
        DP(float, 5)[at] = v;
    }
}

/* GP_GLM_EXPAND: psel [t, kp] (the chosen pools of each query, ascending:
 * GP_DS_TOPK), pcnt [t], sel [t, maxk], cnt [t], pos, ratio, kp, maxk, t: the
 * tokens of the chosen pools, then the open tail of the query (the tokens of
 * position p = pos + j after its last complete pool: Glm5NextTextIndexer.
 * append_visible_tail), ascending. A block for each query. */
__global__ void __launch_bounds__(256) k_glm_expand(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, ratio = DI(5), kp = DI(6), maxk = DI(7);
    int64_t p = di(r, e, 4) + j;
    int np = DP(const int, 1)[j];
    const int *ps = DP(const int, 0) + (size_t)j * kp;
    int *sel = DP(int, 2) + (size_t)j * maxk;
    int tail = (int)((p + 1) % ratio), n = np * ratio;
    for (int u = threadIdx.x; u < n + tail; u += blockDim.x) {
        sel[u] = u < n ? ps[u / ratio] * ratio + u % ratio : (int)(p + 1 - tail + (u - n));
    }
    if (threadIdx.x == 0) {
        DP(int, 3)[j] = n + tail;
    }
}

/* ---------- mHC of a step: the mixes over many blocks, the coefficients once ----------
 *
 * GP_GLM_HC_MIX: h [t, n] (the streams side by side), fn [24, n], base [24],
 * scale [3], raw [t, 24], coef [t, 20] (post 4, then comb 16), part [t,
 * GLM_HC_NCH, 25] (scratch), t, n, eps, hc_eps, iters. raw = (h . fn_m) /
 * sqrt(mean h^2 + eps), as GP_DS_HC_MIX (its sums over GLM_HC_NCH chunks of
 * the columns, then in their order); coef: the post and comb of ds_hc_coef
 * (the Sinkhorn of comb, once for the token: GP_DS_HC_POST made them in each
 * of its blocks, thread 0 alone, 92 us of a step). k_glm_hc_part: a block for
 * each (chunk, token); k_glm_hc_coef: a warp for each token.
 *
 * GP_GLM_HC_POST: y [t, dim], res [t, 4, dim], coef [t, 20], out [t, 4,
 * dim] (may be res), t, dim: out_k = post_k y + sum_j comb_jk res_j. */
#define GLM_HC_NCH 32
__global__ void __launch_bounds__(256) k_glm_hc_part(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ch = blockIdx.x, j = blockIdx.y, n = DI(8), w = n / GLM_HC_NCH;
    const float *h = DP(const float, 0) + (size_t)j * n + (size_t)ch * w;
    const float *fn = DP(const float, 1) + (size_t)ch * w;
    float d[DS_NMIX + 1];
    #pragma unroll
    for (int m = 0; m <= DS_NMIX; ++m) {
        d[m] = 0.f;
    }
    for (int i = threadIdx.x; i < w; i += blockDim.x) {
        float x = h[i];
        d[DS_NMIX] += x * x;
        #pragma unroll
        for (int m = 0; m < DS_NMIX; ++m) {
            d[m] += x * fn[(size_t)m * n + i];
        }
    }
    /* each warp's sums by shuffles, then the warps' in order (one barrier: 25
     * block_sums took 24 us a record) */
    __shared__ float red[8][DS_NMIX + 1];
    int lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    #pragma unroll
    for (int m = 0; m <= DS_NMIX; ++m) {
        float v = d[m];
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(RC_FULL_MASK, v, o);
        }
        if (lane == 0) {
            red[wp][m] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x <= DS_NMIX) {
        float v = 0.f;
        for (int q = 0; q < (int)(blockDim.x / 32); ++q) {
            v += red[q][threadIdx.x];
        }
        DP(float, 6)[((size_t)j * GLM_HC_NCH + ch) * (DS_NMIX + 1) + threadIdx.x] = v;
    }
}

/* the coefficients with a lane for each entry of comb (j = lane / 4, k =
 * lane % 4: the row sums over the 4 lanes of a row, the column sums over the
 * 4 rows) and for each post (lanes 16 to 19): the math of ds_hc_coef */
__global__ void __launch_bounds__(32) k_glm_hc_coef(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, lane = threadIdx.x, n = DI(8);
    const float *part = DP(const float, 6) + (size_t)j * GLM_HC_NCH * (DS_NMIX + 1);
    __shared__ float raw[DS_NMIX];
    float ss = 0.f, d = 0.f;
    for (int c = 0; c < GLM_HC_NCH; ++c) {
        ss += part[(size_t)c * (DS_NMIX + 1) + DS_NMIX];
        if (lane < DS_NMIX) {
            d += part[(size_t)c * (DS_NMIX + 1) + lane];
        }
    }
    if (lane < DS_NMIX) {
        raw[lane] = d * (1.f / sqrtf(ss / (float)n + df(r, e, 9)));
        DP(float, 4)[(size_t)j * DS_NMIX + lane] = raw[lane];
    }
    __syncwarp();
    const float *sc = DP(const float, 3), *base = DP(const float, 2);
    const float eps = df(r, e, 10);
    const int iters = DI(11);
    int l = lane % 16;
    float v = raw[2 * DS_HC + l] * sc[2] + base[2 * DS_HC + l];
    float m = v;
    m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, 1));
    m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, 2));
    float ex = expf(v - m);
    float s = ex + __shfl_xor_sync(RC_FULL_MASK, ex, 1);
    s += __shfl_xor_sync(RC_FULL_MASK, s, 2);
    float cv = ex / s + eps;
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {
            float rsum = cv + __shfl_xor_sync(RC_FULL_MASK, cv, 1);
            rsum += __shfl_xor_sync(RC_FULL_MASK, rsum, 2);
            cv /= rsum + eps;
        }
        float csum = cv + __shfl_xor_sync(RC_FULL_MASK, cv, 4);
        csum += __shfl_xor_sync(RC_FULL_MASK, csum, 8);
        cv /= csum + eps;
    }
    float *co = DP(float, 5) + (size_t)j * 20;
    if (lane < 16) {
        co[DS_HC + lane] = cv;
    } else if (lane < 16 + DS_HC) {
        int k = lane - 16;
        co[k] = 2.f / (1.f + expf(-(raw[DS_HC + k] * sc[1] + base[DS_HC + k])));
    }
}

__global__ void __launch_bounds__(256) k_glm_hc_post(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.y, dim = DI(5), c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) {
        return;
    }
    const float *co = DP(const float, 2) + (size_t)j * 20;
    float y = DP(const float, 0)[(size_t)j * dim + c];
    const float *res = DP(const float, 1) + (size_t)j * DS_HC * dim + c;
    float rv[DS_HC];
    for (int k = 0; k < DS_HC; ++k) {
        rv[k] = res[(size_t)k * dim];
    }
    float *out = DP(float, 3) + (size_t)j * DS_HC * dim + c;
    for (int k = 0; k < DS_HC; ++k) {
        float o = co[k] * y;
        for (int i = 0; i < DS_HC; ++i) {
            o += co[DS_HC + DS_HC * i + k] * rv[i];
        }
        out[(size_t)k * dim] = o;
    }
}

/* ---------- the DSpark drafter of GLM-5.3 (models/glm53/dspark.py) ----------
 *
 * GP_GLM_ROPE: x [rows, heads, hd] in place, pos, rows, heads, hd, theta: the
 * neox RoPE (the halves rotated) of row j at position pos + j, the angles in
 * double (positions up to the context of a million). A block for each row.
 *
 * GP_GLM_RING_PUT: rows [t, n], ring [W, n], pos, t, W, n, nreal (or none):
 * the real rows j < nreal, the last W of them, into slot (pos + j) % W.
 *
 * GP_GLM_DSP_ATTN: q [nq, H, hd], ring K, V [W, H hd], block K, V [nq, H hd],
 * out [nq, H hd], pos, W, H, hd, nq: each query of the block over the context
 * positions max(0, pos - W) .. pos - 1 (their slots of the rings) and all nq
 * rows of the block (no causal mask), softmax(q k / sqrt(hd)) v. A block for
 * each (head, 4 queries); hd = 64. */
__global__ void k_glm_rope(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, H = DI(3), hd = DI(4), half = hd / 2;
    int64_t p = di(r, e, 1) + j;
    double theta = (double)df(r, e, 5);
    float *x = DP(float, 0) + (size_t)j * H * hd;
    for (int u = threadIdx.x; u < H * half; u += blockDim.x) {
        int h = u / half, i = u % half;
        double inv = pow(theta, -2.0 * i / (double)hd);
        double s, c;
        sincos((double)p * inv, &s, &c);
        float *xh = x + (size_t)h * hd;
        float a = xh[i], b = xh[i + half];
        xh[i] = (float)(a * c - b * s);
        xh[i + half] = (float)(b * c + a * s);
    }
}

__global__ void k_glm_ring_put(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int t = DI(3), W = DI(4), n = DI(5), j = blockIdx.y;
    int nreal = r->tag[6] == GP_T_NONE ? t : DI(6);
    if (nreal <= 0 || nreal > t) {
        nreal = t;
    }
    if (j >= nreal || j < nreal - W) {
        return;
    }
    int64_t slot = (di(r, e, 2) + j) % W;
    const float *src = DP(const float, 0) + (size_t)j * n;
    float *dst = DP(float, 1) + (size_t)slot * n;
    for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < n; c += gridDim.x * blockDim.x) {
        dst[c] = src[c];
    }
}

#define GLM_DSP_QB 4
#define GLM_DSP_MAXK (2048 + 16)
__global__ void __launch_bounds__(256) k_glm_dsp_attn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int h = blockIdx.x, q0 = blockIdx.y * GLM_DSP_QB, W = DI(7), H = DI(8), nq = DI(10);
    int64_t pos = di(r, e, 6);
    int nctx = (int)(pos < W ? pos : W), nk = nctx + nq;
    int64_t c0 = pos - nctx;
    const int hd = 64, HD = H * hd;
    __shared__ float qs[GLM_DSP_QB][64];
    __shared__ float sc[GLM_DSP_QB][GLM_DSP_MAXK];
    __shared__ float red[GLM_DSP_QB];
    const float *q = DP(const float, 0), *rk = DP(const float, 1), *rv = DP(const float, 2);
    const float *bk = DP(const float, 3), *bv = DP(const float, 4);
    for (int i = threadIdx.x; i < GLM_DSP_QB * hd; i += blockDim.x) {
        int a = i / hd, d = i % hd;
        qs[a][d] = q0 + a < nq ? q[(size_t)(q0 + a) * HD + (size_t)h * hd + d] * 0.125f : 0.f;   /* 1/sqrt(64) */
    }
    __syncthreads();
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int m = warp; m < nk; m += 8) {
        const float *kr = m < nctx ? rk + (size_t)((c0 + m) % W) * HD : bk + (size_t)(m - nctx) * HD;
        float k0 = kr[h * hd + 2 * lane], k1 = kr[h * hd + 2 * lane + 1];
        #pragma unroll
        for (int a = 0; a < GLM_DSP_QB; ++a) {
            float d = qs[a][2 * lane] * k0 + qs[a][2 * lane + 1] * k1;
            for (int o = 16; o > 0; o >>= 1) {
                d += __shfl_xor_sync(RC_FULL_MASK, d, o);
            }
            if (lane == 0) {
                sc[a][m] = d;
            }
        }
    }
    __syncthreads();
    if (warp < GLM_DSP_QB) {
        float mx = -INFINITY;
        for (int m = lane; m < nk; m += 32) {
            mx = fmaxf(mx, sc[warp][m]);
        }
        for (int o = 16; o > 0; o >>= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, o));
        }
        float sum = 0.f;
        for (int m = lane; m < nk; m += 32) {
            float p = expf(sc[warp][m] - mx);
            sc[warp][m] = p;
            sum += p;
        }
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
        }
        if (lane == 0) {
            red[warp] = 1.f / sum;
        }
    }
    __syncthreads();
    int a = threadIdx.x / hd, d = threadIdx.x % hd;     /* 4 queries x 64 dims */
    if (q0 + a >= nq) {
        return;
    }
    float o = 0.f;
    for (int m = 0; m < nk; ++m) {
        const float *vr = m < nctx ? rv + (size_t)((c0 + m) % W) * HD : bv + (size_t)(m - nctx) * HD;
        o += sc[a][m] * vr[h * hd + d];
    }
    DP(float, 5)[(size_t)(q0 + a) * HD + (size_t)h * hd + d] = o * red[a];
}

/* ---------- the MLA of a prompt group on the tensor cores ----------
 *
 * GP_GLM_ATTN: q [t, 64, 512] (the latent queries), lat [*, 512] (the
 * bfloat16 latents), sel [t, maxk], cnt [t], out [t, 64, 512], H (64), hd
 * (512), scale, maxk, t: each query's 64 heads over its selected latents
 * (the keys and the values), softmax(scale q k) k, no sink. k_ds_attn_g ran
 * it on the CUDA cores (510 ms a layer for 4096 queries at 8K). A block for
 * each query, 8 warps; the query's 64 heads (times scale) in shared memory as
 * float16, the latents 16 at a time from the cache (bfloat16 to float16:
 * exact at their sizes); a pass with a running softmax: warp w the scores of
 * heads 16 (w % 4) .. + 15 and keys 8 (w / 4) .. + 7 of the chunk (mma.sync
 * m16n8k16, float32 sums), the max and the sum of each head (one thread a
 * row), P float16, then warp w the output of its 16 heads and dims 256 (w /
 * 4) .. + 255 (V by ldmatrix.trans). */
#define GA_KC 16
#define GA_LD 520
#define GA_SMEM ((size_t)(64 + GA_KC) * GA_LD * 2 + 64 * (GA_KC + 1) * 4 + 64 * (GA_KC + 8) * 2)
__global__ void __launch_bounds__(256, 1) k_glm_attn_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) unsigned char gasm[];
    __half (*Qs)[GA_LD] = (__half (*)[GA_LD])gasm;
    __half (*Ks)[GA_LD] = (__half (*)[GA_LD])(gasm + 64 * GA_LD * 2);
    float (*Ss)[GA_KC + 1] = (float (*)[GA_KC + 1])(gasm + (64 + GA_KC) * GA_LD * 2);
    __half (*Ps)[GA_KC + 8] = (__half (*)[GA_KC + 8])(gasm + (64 + GA_KC) * GA_LD * 2 + 64 * (GA_KC + 1) * 4);
    __shared__ float rm[64], rl[64], rs[64];
    const int j = blockIdx.x, maxk = DI(8);
    const int nk = DP(const int, 3)[j];
    const int *sel = DP(const int, 2) + (size_t)j * maxk;
    const uint16_t *lat = DP(const uint16_t, 1);
    const float scale = df(r, e, 7);
    const float *q = DP(const float, 0) + (size_t)j * 64 * 512;
    for (int i = threadIdx.x; i < 64 * 512; i += 256) {
        Qs[i / 512][i % 512] = __float2half_rn(q[i] * scale);
    }
    if (threadIdx.x < 64) {
        rm[threadIdx.x] = -INFINITY;
        rl[threadIdx.x] = 0.f;
    }
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32, g = lane / 4, c = lane % 4;
    const int h0 = 16 * (w % 4), kt = 8 * (w / 4), d0 = 256 * (w / 4);
    float o[32][4];
    #pragma unroll
    for (int n = 0; n < 32; ++n) {
        o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f;
    }
    for (int m0 = 0; m0 < nk; m0 += GA_KC) {
        __syncthreads();
        for (int i = threadIdx.x; i < GA_KC * 128; i += 256) {
            int kk = i / 128, c4 = (i % 128) * 4;
            uint2 u = make_uint2(0u, 0u);
            if (m0 + kk < nk) {
                u = *(const uint2 *)(lat + (size_t)sel[m0 + kk] * 512 + c4);
            }
            Ks[kk][c4 + 0] = __float2half_rn(__uint_as_float(u.x << 16));
            Ks[kk][c4 + 1] = __float2half_rn(__uint_as_float(u.x & 0xffff0000u));
            Ks[kk][c4 + 2] = __float2half_rn(__uint_as_float(u.y << 16));
            Ks[kk][c4 + 3] = __float2half_rn(__uint_as_float(u.y & 0xffff0000u));
        }
        __syncthreads();
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll 8
        for (int k0 = 0; k0 < 512; k0 += 16) {
            uint32_t a[4], b[2];
            a[0] = *(const uint32_t *)&Qs[h0 + g][k0 + 2 * c];
            a[1] = *(const uint32_t *)&Qs[h0 + g + 8][k0 + 2 * c];
            a[2] = *(const uint32_t *)&Qs[h0 + g][k0 + 2 * c + 8];
            a[3] = *(const uint32_t *)&Qs[h0 + g + 8][k0 + 2 * c + 8];
            b[0] = *(const uint32_t *)&Ks[kt + g][k0 + 2 * c];
            b[1] = *(const uint32_t *)&Ks[kt + g][k0 + 2 * c + 8];
            mma16816(s, a, b);
        }
        Ss[h0 + g][kt + 2 * c] = s[0];
        Ss[h0 + g][kt + 2 * c + 1] = s[1];
        Ss[h0 + g + 8][kt + 2 * c] = s[2];
        Ss[h0 + g + 8][kt + 2 * c + 1] = s[3];
        __syncthreads();
        if (threadIdx.x < 64) {
            const int h = threadIdx.x;
            float mo = rm[h], mn = mo;
            for (int kk = 0; kk < GA_KC; ++kk) {
                if (m0 + kk < nk) {
                    mn = fmaxf(mn, Ss[h][kk]);
                }
            }
            float sc = expf(mo - mn), l = rl[h] * sc;
            for (int kk = 0; kk < GA_KC; ++kk) {
                __half ph = __float2half_rn(m0 + kk < nk ? expf(Ss[h][kk] - mn) : 0.f);
                Ps[h][kk] = ph;
                l += __half2float(ph);
            }
            rm[h] = mn;
            rl[h] = l;
            rs[h] = sc;
        }
        __syncthreads();
        const float sc0 = rs[h0 + g], sc1 = rs[h0 + g + 8];
        uint32_t a[4];
        a[0] = *(const uint32_t *)&Ps[h0 + g][2 * c];
        a[1] = *(const uint32_t *)&Ps[h0 + g + 8][2 * c];
        a[2] = *(const uint32_t *)&Ps[h0 + g][2 * c + 8];
        a[3] = *(const uint32_t *)&Ps[h0 + g + 8][2 * c + 8];
        const int mat = lane / 8, row = lane % 8;
        #pragma unroll
        for (int n = 0; n < 32; n += 2) {
            o[n][0] *= sc0;
            o[n][1] *= sc0;
            o[n][2] *= sc1;
            o[n][3] *= sc1;
            o[n + 1][0] *= sc0;
            o[n + 1][1] *= sc0;
            o[n + 1][2] *= sc1;
            o[n + 1][3] *= sc1;
            uint32_t bb[4];
            ldsm_x4_trans(bb, &Ks[(mat & 1) * 8 + row][d0 + 8 * n + (mat >> 1) * 8]);
            uint32_t b0[2] = {bb[0], bb[1]}, b1[2] = {bb[2], bb[3]};
            mma16816(o[n], a, b0);
            mma16816(o[n + 1], a, b1);
        }
    }
    __syncthreads();
    const float i0 = rl[h0 + g] > 0.f ? 1.f / rl[h0 + g] : 0.f, i1 = rl[h0 + g + 8] > 0.f ? 1.f / rl[h0 + g + 8] : 0.f;
    float *out = DP(float, 4) + (size_t)j * 64 * 512;
    #pragma unroll
    for (int n = 0; n < 32; ++n) {
        int d = d0 + 8 * n + 2 * c;
        *(float2 *)&out[(size_t)(h0 + g) * 512 + d] = make_float2(o[n][0] * i0, o[n][1] * i0);
        *(float2 *)&out[(size_t)(h0 + g + 8) * 512 + d] = make_float2(o[n][2] * i1, o[n][3] * i1);
    }
}
