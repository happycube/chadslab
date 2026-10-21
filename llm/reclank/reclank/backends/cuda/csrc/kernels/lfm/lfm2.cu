/* ---------- LFM2-MoE (docs/LFM25_24B_PLAN.md) ----------
 *
 * GP_SHORT_CONV: the double-gated short convolution (gp_short_conv_body of
 * the CPU). GP_ROUTER_SIG: the router with sigmoid scores and the expert
 * bias (gp_router_sig_body). k_attn64: the attention of one query over the
 * int16 or int8 cache for head_dim 64 (GP_ATTN_QC, GP_ATTN_Q8 from
 * attn_launch), split by keys, and k_attn64_join. */

/* GP_SHORT_CONV: in (t x 3 ch: B, C, x), state (kernel - 1 x ch), w (ch x
 * kernel), out (t x ch), t, ch, kernel, nreal (or none). One thread for each
 * channel, the tokens in order. Only the first nreal tokens (0: all t) give
 * the state; the other rows pad a group to the size of its program. */
__global__ void k_short_conv(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int t = DI(4), ch = DI(5), kernel = DI(6);
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) {
        return;
    }
    int nreal = r->tag[7] == GP_T_NONE ? 0 : DI(7);
    if (nreal <= 0 || nreal > t) {
        nreal = t;
    }
    const float *in = DP(const float, 0);
    float *state = DP(float, 1), *out = DP(float, 3);
    const float *w = DP(const float, 2) + (size_t)c * kernel;
    float hist[8];
    for (int q = 0; q < kernel - 1; ++q) {
        hist[q] = state[(size_t)q * ch + c];
    }
    for (int j = 0; j < t; ++j) {
        const float *row = in + (size_t)j * 3 * ch;
        float bx = row[c] * row[2 * ch + c];
        float y = 0.f;
        for (int q = 0; q < kernel - 1; ++q) {
            y += w[q] * hist[q];
        }
        y += w[kernel - 1] * bx;
        for (int q = 0; q + 1 < kernel - 1; ++q) {
            hist[q] = hist[q + 1];
        }
        if (kernel > 1) {
            hist[kernel - 2] = bx;
        }
        out[(size_t)j * ch + c] = row[ch + c] * y;
        if (j == nreal - 1) {
            for (int q = 0; q < kernel - 1; ++q) {
                state[(size_t)q * ch + c] = hist[q];
            }
        }
    }
}

/* GP_SHORT_CONV of a group: block (channels, row j). Row j reads B x of rows
 * j - kernel + 1 .. j: from the input rows, or from the state for the rows
 * before the group (the order of the sum of k_short_conv, so the same
 * values). The state does not change here (k_short_conv_state then). */
__device__ __forceinline__ float sc_bx(const float *in, const float *state, int j, int c, int ch,
                                       int kernel)
{
    if (j >= 0) {
        const float *row = in + (size_t)j * 3 * ch;
        return row[c] * row[2 * ch + c];
    }
    return state[(size_t)(kernel - 1 + j) * ch + c];
}

__global__ void k_short_conv_rows(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ch = DI(5), kernel = DI(6);
    int c = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y;
    if (c >= ch) {
        return;
    }
    const float *in = DP(const float, 0), *state = DP(const float, 1);
    const float *w = DP(const float, 2) + (size_t)c * kernel;
    float y = 0.f;
    for (int q = 0; q < kernel - 1; ++q) {
        y += w[q] * sc_bx(in, state, j - (kernel - 1) + q, c, ch, kernel);
    }
    y += w[kernel - 1] * sc_bx(in, state, j, c, ch, kernel);
    DP(float, 3)[(size_t)j * ch + c] = in[(size_t)j * 3 * ch + ch + c] * y;
}

/* The state after the first nreal rows of a group (0: all t): the last
 * kernel - 1 rows of B x, from the group or from the state before it. */
__global__ void k_short_conv_state(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int t = DI(4), ch = DI(5), kernel = DI(6);
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ch) {
        return;
    }
    int nreal = r->tag[7] == GP_T_NONE ? 0 : DI(7);
    if (nreal <= 0 || nreal > t) {
        nreal = t;
    }
    const float *in = DP(const float, 0);
    float *state = DP(float, 1);
    float v[8];
    for (int q = 0; q < kernel - 1; ++q) {
        v[q] = sc_bx(in, state, nreal - (kernel - 1) + q, c, ch, kernel);
    }
    for (int q = 0; q < kernel - 1; ++q) {
        state[(size_t)q * ch + c] = v[q];
    }
}

/* GP_ROUTER_SIG: logits, t, experts, k, val, idx, bias (or null), scale, eps
 * (or none: 1e-6; GLM-5.3 1e-20). One warp for each token: the scores
 * sigmoid(logit) in shared memory, the k largest of score + bias (the lower
 * index at an equal value), the weights the k scores over their sum + eps,
 * times scale. At most 1024 experts. */
__global__ void k_router_sig(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float s[1024], c[1024];
    int j = blockIdx.x, E = DI(2), k = DI(3), lane = threadIdx.x;
    const float *l = DP(const float, 0) + (size_t)j * E;
    const float *bias = r->tag[6] == GP_T_NONE ? NULL : DP(const float, 6);
    for (int x = lane; x < E; x += 32) {
        s[x] = 1.f / (1.f + expf(-l[x]));
        c[x] = s[x] + (bias != NULL ? bias[x] : 0.f);
    }
    __syncwarp();
    float *val = DP(float, 4) + (size_t)j * k;
    int *idx = DP(int, 5) + (size_t)j * k;
    float sum = 0.f;
    for (int s2 = 0; s2 < k; ++s2) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int x = lane; x < E; x += 32) {
            if (c[x] != -INFINITY && (c[x] > bv || (c[x] == bv && x < bi))) {
                bv = c[x];
                bi = x;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float v2 = __shfl_xor_sync(RC_FULL_MASK, bv, o);
            int i2 = __shfl_xor_sync(RC_FULL_MASK, bi, o);
            if (v2 > bv || (v2 == bv && i2 < bi)) {
                bv = v2;
                bi = i2;
            }
        }
        sum += s[bi];
        if (lane == 0) {
            idx[s2] = bi;
            val[s2] = s[bi];
        }
        __syncwarp();
        if (lane == 0) {
            c[bi] = -INFINITY;
        }
        __syncwarp();
    }
    float sc = df(r, e, 7), eps = r->tag[8] == GP_T_NONE ? 1e-6f : df(r, e, 8);
    if (lane < k) {
        val[lane] = val[lane] / (sum + eps) * sc;
    }
}

/* k_router_sig with a block of 256 for many experts (Nemotron: 240-496; one warp took 16 us): the same
 * choices (the larger score, the smaller index on a tie), the same sum in the order of the slots,
 * the same weights */
__global__ void __launch_bounds__(256) k_router_sig_b(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float s[1024], c[1024], rv[8];
    __shared__ int ri[8];
    const int j = blockIdx.x, E = DI(2), k = DI(3), tid = threadIdx.x, lane = tid % 32, wp = tid / 32;
    const float *l = DP(const float, 0) + (size_t)j * E;
    const float *bias = r->tag[6] == GP_T_NONE ? NULL : DP(const float, 6);
    for (int x = tid; x < E; x += blockDim.x) {
        s[x] = 1.f / (1.f + expf(-l[x]));
        c[x] = s[x] + (bias != NULL ? bias[x] : 0.f);
    }
    __syncthreads();
    float *val = DP(float, 4) + (size_t)j * k;
    int *idx = DP(int, 5) + (size_t)j * k;
    float sum = 0.f;
    for (int s2 = 0; s2 < k; ++s2) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int x = tid; x < E; x += blockDim.x) {
            if (c[x] != -INFINITY && (c[x] > bv || (c[x] == bv && x < bi))) {
                bv = c[x];
                bi = x;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float v2 = __shfl_xor_sync(RC_FULL_MASK, bv, o);
            int i2 = __shfl_xor_sync(RC_FULL_MASK, bi, o);
            if (v2 > bv || (v2 == bv && i2 < bi)) {
                bv = v2;
                bi = i2;
            }
        }
        if (lane == 0) {
            rv[wp] = bv;
            ri[wp] = bi;
        }
        __syncthreads();
        if (tid == 0) {
            bv = rv[0];
            bi = ri[0];
            for (int w = 1; w < (int)blockDim.x / 32; ++w) {
                if (rv[w] > bv || (rv[w] == bv && ri[w] < bi)) {
                    bv = rv[w];
                    bi = ri[w];
                }
            }
            sum += s[bi];
            idx[s2] = bi;
            val[s2] = s[bi];
            c[bi] = -INFINITY;
        }
        __syncthreads();
    }
    if (tid == 0) {
        const float sc = df(r, e, 7), eps = r->tag[8] == GP_T_NONE ? 1e-6f : df(r, e, 8);
        for (int q = 0; q < k; ++q) {
            val[q] = val[q] / (sum + eps) * sc;
        }
    }
}

/* k_router_sig_b's choices by a bitonic sort (E <= 1024; N the power of two from E up, N / 2 threads a
 * block): the experts by score down, the index up on a tie (k_router_sig_b's order of choice: a total
 * order), so the first k are its slots in its order; then the sum in the order of the slots: the same
 * choices, sum and weights (k_router_sig_b's 22 rounds took 24 us a token on a 3090, a rank of each
 * expert 20). Operands past k_router_sig_b's (or none): 9 counts (each choice of the first nreal rows
 * adds 1: GP_COUNT's), 10 slots, 11 cold, 12 count (GP_NM_SPLIT's: a pair's expert when its slot is
 * -1, else -1; their count; the grid one block for all the rows), 13 nreal. */
__device__ void router_sig_r_body(const gp_rec *r, const int64_t *e)
{
    __shared__ float s[1024], c[1024];
    __shared__ int id[1024], ncold;
    const int t = DI(1), E = DI(2), k = DI(3), tid = threadIdx.x, N = 2 * blockDim.x;
    const float *bias = r->tag[6] == GP_T_NONE ? NULL : DP(const float, 6);
    int *counts = r->tag[9] == GP_T_NONE ? NULL : DP(int, 9);
    const int *slots = r->tag[10] == GP_T_NONE ? NULL : DP(const int, 10);
    int nreal = r->tag[13] == GP_T_NONE ? t : DI(13);
    nreal = nreal > 0 && nreal < t ? nreal : t;
    const float sc = df(r, e, 7), eps = r->tag[8] == GP_T_NONE ? 1e-6f : df(r, e, 8);
    if (tid == 0) {
        ncold = 0;
    }
    const int j0 = slots != NULL ? 0 : blockIdx.x, j1 = slots != NULL ? t : blockIdx.x + 1;
    for (int j = j0; j < j1; ++j) {
        const float *l = DP(const float, 0) + (size_t)j * E;
        __syncthreads();
        for (int x = tid; x < N; x += blockDim.x) {
            if (x < E) {
                s[x] = 1.f / (1.f + expf(-l[x]));
                c[x] = s[x] + (bias != NULL ? bias[x] : 0.f);
            } else {
                c[x] = -INFINITY;
            }
            id[x] = x;
        }
        __syncthreads();
        for (int kk = 2; kk <= N; kk <<= 1) {
            for (int jj = kk >> 1; jj > 0; jj >>= 1) {
                const int i = (tid / jj) * 2 * jj + tid % jj, ix = i + jj;
                const float ca = c[i], cb = c[ix];
                const int ia = id[i], ib = id[ix];
                const bool bfirst = cb > ca || (cb == ca && ib < ia);
                if (bfirst == ((i & kk) == 0)) {
                    c[i] = cb;
                    c[ix] = ca;
                    id[i] = ib;
                    id[ix] = ia;
                }
                __syncthreads();
            }
        }
        float *val = DP(float, 4) + (size_t)j * k;
        int *idx = DP(int, 5) + (size_t)j * k;
        if (tid == 0) {
            float sum = 0.f;
            for (int q = 0; q < k; ++q) {
                sum += s[id[q]];
                idx[q] = id[q];
                val[q] = s[id[q]];
            }
            for (int q = 0; q < k; ++q) {
                val[q] = val[q] / (sum + eps) * sc;
            }
        }
        if (tid < k) {
            const int x = id[tid];
            if (counts != NULL && j < nreal) {
                atomicAdd(&counts[x], 1);
            }
            if (slots != NULL) {
                const int cx = j < nreal && slots[x] < 0 ? x : -1;
                DP(int, 11)[(size_t)j * k + tid] = cx;
                if (cx >= 0) {
                    atomicAdd(&ncold, 1);
                }
            }
        }
    }
    if (slots != NULL) {
        __syncthreads();
        if (tid == 0) {
            *DP(int, 12) = ncold;
        }
    }
}

__global__ void __launch_bounds__(512) k_router_sig_r(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    router_sig_r_body(r, e);
}

/* RECLANK_GPU_ROUTER_RANK=0: k_router_sig_b (a measure) */
static int router_rank_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_ROUTER_RANK");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* k_attn64: GP_ATTN_QC (int16, MODE 0) or GP_ATTN_Q8 (int8, MODE 1) with
 * head_dim 64: q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, 64, n; or
 * GP_ATTN_F32H of one token (MODE 2, the float32 rows of the first full
 * layer of the Qwen cache): q, K, V, scores, out, q_heads, kv_heads, 64, 1,
 * pos, head_stride (n = pos + 1; key p of head kv at kv hs + p kv_rs).
 * Block (kv, c)
 * takes the keys c len .. (c + 1) len - 1 of key head kv (len: n over
 * A64_NB blocks) for its rep <= 8 query heads; warp w takes every fourth of
 * them. Lane l holds values 2 l and 2 l + 1 of each query and of its output;
 * the score of a key is a warp sum; each warp keeps the online softmax (m,
 * l, the output) of each query head, and the block adds those of its warps
 * into part ((head, c): m, l, 64 values). k_attn64_join adds the parts of a
 * head. The sizes are literals of the graph: the count of the blocks is
 * fixed, the keys of each come from n at run time. */
#define A64_NB 64
#define A64_REP 8

template <int MODE>
__global__ void __launch_bounds__(128) k_attn64(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    __shared__ float red[4][A64_REP][66];
    int qh = MODE == 2 ? DI(5) : DI(7), kvh = MODE == 2 ? DI(6) : DI(8);
    int n = MODE == 2 ? (int)di(r, e, 9) + 1 : DI(10);
    int kv = blockIdx.x, cb = blockIdx.y, rep = qh / kvh;
    int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    int len = (n + A64_NB - 1) / A64_NB;
    int k0 = cb * len, k1 = min(n, k0 + len);
    const float *q = DP(const float, 0);
    const float *ks = MODE == 2 ? NULL : DP(const float, 2), *vs = MODE == 2 ? NULL : DP(const float, 4);
    size_t hs = MODE == 2 ? (size_t)di(r, e, 10) : 0, rs = MODE == 2 ? kv_rs(hs, kvh, 64) : 0;
    float qa[A64_REP], qb[A64_REP], m[A64_REP], l[A64_REP], oa[A64_REP], ob[A64_REP];
    #pragma unroll
    for (int h = 0; h < A64_REP; ++h) {
        const float *qr = q + (size_t)(kv * rep + (h < rep ? h : 0)) * 64;
        qa[h] = qr[2 * lane];
        qb[h] = qr[2 * lane + 1];
        m[h] = -INFINITY;
        l[h] = 0.f;
        oa[h] = 0.f;
        ob[h] = 0.f;
    }
    size_t row = (size_t)kvh * 64;
    for (int p = k0 + w; p < k1; p += 4) {
        size_t o = (size_t)p * row + (size_t)kv * 64 + 2 * lane;
        float ka, kb, va, vb;
        if (MODE == 2) {
            size_t of = (size_t)kv * hs + (size_t)p * rs + 2 * lane;
            float2 kk = *(const float2 *)(DP(const float, 1) + of);
            float2 vv = *(const float2 *)(DP(const float, 2) + of);
            ka = kk.x;
            kb = kk.y;
            va = vv.x;
            vb = vv.y;
        } else if (MODE == 1) {
            float ksc = ks[o / 32], vsc = vs[o / 32];
            const int8_t *kq = DP(const int8_t, 1), *vq = DP(const int8_t, 3);
            ka = (float)kq[o] * ksc;
            kb = (float)kq[o + 1] * ksc;
            va = (float)vq[o] * vsc;
            vb = (float)vq[o + 1] * vsc;
        } else {
            float ksc = ks[o / 32], vsc = vs[o / 32];
            short2 kk = *(const short2 *)(DP(const int16_t, 1) + o);
            short2 vv = *(const short2 *)(DP(const int16_t, 3) + o);
            ka = (float)kk.x * ksc;
            kb = (float)kk.y * ksc;
            va = (float)vv.x * vsc;
            vb = (float)vv.y * vsc;
        }
        #pragma unroll
        for (int h = 0; h < A64_REP; ++h) {
            if (h >= rep) {
                break;
            }
            float sd = qa[h] * ka + qb[h] * kb;
            for (int off = 16; off > 0; off >>= 1) {
                sd += __shfl_xor_sync(RC_FULL_MASK, sd, off);
            }
            float mn = fmaxf(m[h], sd);
            float f = m[h] == -INFINITY ? 0.f : expf(m[h] - mn);
            float pk = expf(sd - mn);
            l[h] = l[h] * f + pk;
            oa[h] = oa[h] * f + pk * va;
            ob[h] = ob[h] * f + pk * vb;
            m[h] = mn;
        }
    }
    for (int h = 0; h < rep; ++h) {
        red[w][h][2 + 2 * lane] = oa[h];
        red[w][h][3 + 2 * lane] = ob[h];
        if (lane == 0) {
            red[w][h][0] = m[h];
            red[w][h][1] = l[h];
        }
    }
    __syncthreads();
    /* thread x: value x % 64 of head x / 64 (two heads at a time) */
    for (int hv = threadIdx.x; hv < rep * 64; hv += blockDim.x) {
        int h = hv / 64, d = hv % 64;
        float M = -INFINITY;
        for (int ww = 0; ww < 4; ++ww) {
            M = fmaxf(M, red[ww][h][0]);
        }
        float L = 0.f, acc = 0.f;
        for (int ww = 0; ww < 4; ++ww) {
            float mw = red[ww][h][0];
            float f = mw == -INFINITY ? 0.f : expf(mw - M);
            L += f * red[ww][h][1];
            acc += f * red[ww][h][2 + d];
        }
        float *pp = part + ((size_t)(kv * rep + h) * A64_NB + cb) * 66;
        if (d == 0) {
            pp[0] = M;
            pp[1] = L;
        }
        pp[2 + d] = acc;
    }
}

/* Block h: value d (thread d) of query head h over the A64_NB parts, into
 * the output (operand oo: 6, or 4 for GP_ATTN_F32H). */
__global__ void k_attn64_join(const gp_rec *r, const int64_t *e, const float *part, int oo)
{
    PDL_START();
    int h = blockIdx.x, d = threadIdx.x;
    const float *pp = part + (size_t)h * A64_NB * 66;
    float M = -INFINITY;
    for (int c = 0; c < A64_NB; ++c) {
        M = fmaxf(M, pp[(size_t)c * 66]);
    }
    float L = 0.f, acc = 0.f;
    for (int c = 0; c < A64_NB; ++c) {
        float mc = pp[(size_t)c * 66];
        float f = mc == -INFINITY ? 0.f : expf(mc - M);
        L += f * pp[(size_t)c * 66 + 1];
        acc += f * pp[(size_t)c * 66 + 2 + d];
    }
    DP(float, oo)[(size_t)h * 64 + d] = L > 0.f ? acc / L : 0.f;
}
