/* ---------- DiffusionGemma (models/dgemma/gpu.py, docs/DIFFUSIONGEMMA_PLAN.md) ----------
 *
 * DG_KV_PUT   t rows of k and v into a layer's cache at the row in device memory (*row)
 * DG_ATTN     t queries over rows of a layer's cache: query i the rows [lo[i], hi[i]), GQA (query
 *             head h reads KV head h / (heads / kvh)), scale 1, float32 (an online softmax)
 * DG_SAMPLE   per row of logits: / the temperature, the end tokens banned (-inf) where ban[i],
 *             the max, the logsumexp, the entropy, the argmax and the sample of the uniform u[i]
 *             (the inverse of the cumulative sum, in the order of the block's chunks); the
 *             processed logits stay in place and their softmax goes to probs
 */

__global__ void k_dg_kv_put(const float *k, const float *v, float *K, float *V, const int *row, int t,
                            int kcols, int vcols)
{
    const int i = blockIdx.x;
    const size_t r = (size_t)(*row + i);
    for (int c = threadIdx.x; c < kcols; c += blockDim.x) {
        K[r * kcols + c] = k[(size_t)i * kcols + c];
    }
    for (int c = threadIdx.x; c < vcols; c += blockDim.x) {
        V[r * vcols + c] = v[(size_t)i * vcols + c];
    }
}

/* A block: 8 warps, QPW queries a warp (DG_QB = 8 QPW queries), one query head. Keys in tiles of
 * DG_KT rows of k and v in shared memory; each lane holds HD / 32 values of each of its queries
 * (lane + 32 j) and of their outputs. A key outside a query's range adds nothing to it. */
#define DG_QPW 4
#define DG_QB (8 * DG_QPW)

template <int HD>
__global__ void __launch_bounds__(256) k_dg_attn(const float *q, const float *K, const float *V, float *out,
                                                 const int *lo, const int *hi, int t, int heads, int kvh)
{
    constexpr int DJ = HD / 32;
    constexpr int DKT = HD >= 512 ? 16 : 32;
    extern __shared__ float dg_sm[];
    float *ks = dg_sm, *vs = dg_sm + DKT * HD;
    const int h = blockIdx.y, g = h / (heads / kvh);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q0 = blockIdx.x * DG_QB;
    float qr[DG_QPW][DJ], o[DG_QPW][DJ], m[DG_QPW], l[DG_QPW];
    int qlo[DG_QPW], qhi[DG_QPW];
    for (int a = 0; a < DG_QPW; ++a) {
        const int i = q0 + warp * DG_QPW + a;
        const int ok = i < t;
        qlo[a] = ok ? lo[i] : 0;
        qhi[a] = ok ? hi[i] : 0;
        m[a] = -INFINITY;
        l[a] = 0.f;
        for (int j = 0; j < DJ; ++j) {
            qr[a][j] = ok ? q[((size_t)i * heads + h) * HD + lane + 32 * j] : 0.f;
            o[a][j] = 0.f;
        }
    }
    /* the keys of the block: from the least lo to the largest hi of its queries */
    __shared__ int blo, bhi;
    if (threadIdx.x == 0) {
        int a = INT_MAX, b = 0;
        for (int i = q0; i < q0 + DG_QB && i < t; ++i) {
            a = min(a, lo[i]);
            b = max(b, hi[i]);
        }
        blo = a == INT_MAX ? 0 : a;
        bhi = b;
    }
    __syncthreads();
    const int k_lo = blo, k_hi = bhi;
    const size_t kstride = (size_t)kvh * HD;
    for (int k0 = k_lo; k0 < k_hi; k0 += DKT) {
        const int nk = min(DKT, k_hi - k0);
        __syncthreads();
        for (int e = threadIdx.x; e < nk * HD; e += blockDim.x) {
            const int r = e / HD, c = e % HD;
            ks[r * HD + c] = K[(size_t)(k0 + r) * kstride + (size_t)g * HD + c];
            vs[r * HD + c] = V[(size_t)(k0 + r) * kstride + (size_t)g * HD + c];
        }
        __syncthreads();
        for (int r = 0; r < nk; ++r) {
            const int key = k0 + r;
            float kv[DJ];
            for (int j = 0; j < DJ; ++j) {
                kv[j] = ks[r * HD + lane + 32 * j];
            }
            for (int a = 0; a < DG_QPW; ++a) {
                if (key < qlo[a] || key >= qhi[a]) {
                    continue;
                }
                float s = 0.f;
                for (int j = 0; j < DJ; ++j) {
                    s = fmaf(qr[a][j], kv[j], s);
                }
                for (int w = 16; w > 0; w >>= 1) {
                    s += __shfl_xor_sync(0xffffffffu, s, w);
                }
                const float mn = fmaxf(m[a], s);
                const float sc = expf(m[a] - mn), p = expf(s - mn);
                l[a] = l[a] * sc + p;
                for (int j = 0; j < DJ; ++j) {
                    o[a][j] = fmaf(p, vs[r * HD + lane + 32 * j], o[a][j] * sc);
                }
                m[a] = mn;
            }
        }
    }
    for (int a = 0; a < DG_QPW; ++a) {
        const int i = q0 + warp * DG_QPW + a;
        if (i >= t) {
            continue;
        }
        const float inv = l[a] > 0.f ? 1.f / l[a] : 0.f;
        for (int j = 0; j < DJ; ++j) {
            out[((size_t)i * heads + h) * HD + lane + 32 * j] = o[a][j] * inv;
        }
    }
}

static void dg_attn(const float *q, const float *K, const float *V, float *out, const int *lo, const int *hi,
                    int t, int heads, int kvh, int hd, cudaStream_t s)
{
    static int attr = 0;
    if (!attr) {
        cudaFuncSetAttribute(k_dg_attn<256>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * 32 * 256 * 4);
        cudaFuncSetAttribute(k_dg_attn<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, 2 * 16 * 512 * 4);
        attr = 1;
    }
    dim3 grid((unsigned)((t + DG_QB - 1) / DG_QB), (unsigned)heads);
    if (hd == 256) {
        k_dg_attn<256><<<grid, 256, 2 * 32 * 256 * 4, s>>>(q, K, V, out, lo, hi, t, heads, kvh);
    } else {
        k_dg_attn<512><<<grid, 256, 2 * 16 * 512 * 4, s>>>(q, K, V, out, lo, hi, t, heads, kvh);
    }
}

/* One block (1024 threads) a row. Thread j has the values j, j + 1024, ... (its chunk, in that
 * order); the sample is the first value of the cumulative sum, in the order of the chunks, past
 * u (the thread whose chunk holds it then walks it). praw (or null): each row's probability of
 * its argmax under the raw logits (the model's: no temperature, no ban), an online softmax of
 * the raw values in the first pass. */
#define DG_ST 1024

/* The merge of two online softmax sums (max m, sum s of exp(v - m)) into (m, s). */
__device__ __forceinline__ void dg_osm_merge(float &m, float &s, float om, float os)
{
    const float n = fmaxf(m, om);
    if (n == -INFINITY) {
        return;
    }
    s = (m == -INFINITY ? 0.f : s * expf(m - n)) + (om == -INFINITY ? 0.f : os * expf(om - n));
    m = n;
}

__global__ void __launch_bounds__(DG_ST) k_dg_sample(float *z, float *probs, int vocab, const float *temp,
                                                     const int *ban, const int *eos, int neos, const float *u,
                                                     float *ent, int *am, int *draw, float *pmax, float *praw)
{
    const int i = blockIdx.x, tid = threadIdx.x;
    float *zr = z + (size_t)i * vocab;
    float *pr = probs + (size_t)i * vocab;
    const float inv_t = 1.f / *temp;
    __shared__ float red[DG_ST / 32];
    __shared__ int redi[DG_ST / 32];
    __shared__ float s_max, s_sum, s_ez;
    __shared__ float scan[DG_ST];
    __shared__ int s_pick;
    __shared__ float rawm[DG_ST / 32], raws[DG_ST / 32], rawa[DG_ST / 32];
    /* 1: the processed logits, their max and its first index (and its raw value); the raw
     * logits' online softmax sum */
    float mx = -INFINITY, ma = -INFINITY, rm = -INFINITY, rs = 0.f;
    int mi = 0;
    const int banned = ban[i];
    const int raw = praw != NULL;
    for (int c = tid; c < vocab; c += DG_ST) {
        const float r = zr[c];
        if (raw) {
            if (r > rm) {
                rs = (rm == -INFINITY ? 0.f : rs * expf(rm - r)) + 1.f;
                rm = r;
            } else {
                rs += expf(r - rm);
            }
        }
        float v = r * inv_t;
        if (banned) {
            for (int e = 0; e < neos; ++e) {
                if (c == eos[e]) {
                    v = -INFINITY;
                }
            }
        }
        zr[c] = v;
        if (v > mx) {
            mx = v;
            mi = c;
            ma = r;
        }
    }
    for (int w = 16; w > 0; w >>= 1) {
        float om = __shfl_xor_sync(0xffffffffu, mx, w);
        int oi = __shfl_xor_sync(0xffffffffu, mi, w);
        float oa = __shfl_xor_sync(0xffffffffu, ma, w);
        float orm = __shfl_xor_sync(0xffffffffu, rm, w), ors = __shfl_xor_sync(0xffffffffu, rs, w);
        if (om > mx || (om == mx && oi < mi)) {
            mx = om;
            mi = oi;
            ma = oa;
        }
        dg_osm_merge(rm, rs, orm, ors);
    }
    if ((tid & 31) == 0) {
        red[tid >> 5] = mx;
        redi[tid >> 5] = mi;
        rawa[tid >> 5] = ma;
        rawm[tid >> 5] = rm;
        raws[tid >> 5] = rs;
    }
    __syncthreads();
    if (tid == 0) {
        float b = red[0], ba = rawa[0], bm = rawm[0], bs = raws[0];
        int bi = redi[0];
        for (int k = 1; k < DG_ST / 32; ++k) {
            if (red[k] > b || (red[k] == b && redi[k] < bi)) {
                b = red[k];
                bi = redi[k];
                ba = rawa[k];
            }
            dg_osm_merge(bm, bs, rawm[k], raws[k]);
        }
        s_max = b;
        am[i] = bi;
        if (raw) {
            praw[i] = expf(ba - bm) / bs;
        }
    }
    __syncthreads();
    const float M = s_max;
    /* 2: the sums of exp(z - M) and of exp(z - M) (z - M), each thread's own sum */
    float se = 0.f, sez = 0.f;
    for (int c = tid; c < vocab; c += DG_ST) {
        const float d = zr[c] - M;
        const float e = d == -INFINITY ? 0.f : expf(d);
        se += e;
        sez += d == -INFINITY ? 0.f : e * d;
    }
    scan[tid] = se;
    float a = se, b = sez;
    for (int w = 16; w > 0; w >>= 1) {
        a += __shfl_xor_sync(0xffffffffu, a, w);
        b += __shfl_xor_sync(0xffffffffu, b, w);
    }
    __shared__ float red2[DG_ST / 32];
    if ((tid & 31) == 0) {
        red[tid >> 5] = a;
        red2[tid >> 5] = b;
    }
    __syncthreads();
    if (tid == 0) {
        float sa = 0.f, sb = 0.f;
        for (int k = 0; k < DG_ST / 32; ++k) {
            sa += red[k];
            sb += red2[k];
        }
        s_sum = sa;
        s_ez = sb;
        /* the entropy: log S - sum e (z - M) / S; the argmax's probability 1 / S (or null) */
        ent[i] = logf(sa) - sb / sa;
        if (pmax != NULL) {
            pmax[i] = 1.f / sa;
        }
        /* the chunk of the sample: the inclusive prefix of the threads' sums past u S */
        const float target = u[i] * sa;
        float acc = 0.f;
        int pick = DG_ST - 1;
        for (int k = 0; k < DG_ST; ++k) {
            const float n = acc + scan[k];
            if (n > target) {
                pick = k;
                break;
            }
            acc = n;
        }
        s_pick = pick;
        scan[0] = acc;      /* the sum before the chunk */
    }
    __syncthreads();
    const float S = s_sum, invS = 1.f / S;
    /* 3: the probabilities; the thread of the chunk walks it for the sample */
    for (int c = tid; c < vocab; c += DG_ST) {
        const float d = zr[c] - M;
        pr[c] = d == -INFINITY ? 0.f : expf(d) * invS;
    }
    if (tid == s_pick) {
        const float target = u[i] * S;
        float acc = scan[0];
        int pick = -1, last = -1;
        for (int c = tid; c < vocab; c += DG_ST) {
            const float d = zr[c] - M;
            const float e = d == -INFINITY ? 0.f : expf(d);
            if (e > 0.f) {
                last = c;
            }
            acc += e;
            if (acc > target) {
                pick = c;
                break;
            }
        }
        draw[i] = pick >= 0 ? pick : (last >= 0 ? last : am[i]);
    }
}

static void dg_sample(float *z, float *probs, int t, int vocab, const float *temp, const int *ban, const int *eos,
                      int neos, const float *u, float *ent, int *am, int *draw, float *pmax, float *praw,
                      cudaStream_t s)
{
    k_dg_sample<<<(unsigned)t, DG_ST, 0, s>>>(z, probs, vocab, temp, ban, eos, neos, u, ent, am, draw, pmax, praw);
}

/* DG_MOE: the routed experts of a layer on the GPU (BF12 rows; models/dgemma/gpu.py, the
 * layers whose experts the GPU holds), float32 products (the CUDA cores):
 *   route  the pairs (token, slot) by expert, in the order of the pairs (one thread: a counting
 *          sort, the same order each run): start[E + 1], ptok (each sorted pair's token), ppos
 *          (each pair's sorted position)
 *   gu     an expert and 32 of its gate rows with the 32 up rows of the same index (gu holds the
 *          gate rows then the up rows of each expert) on tiles of 16 of its pairs, 32 columns
 *          at a time in shared memory: act = gelu_tanh(gate) up
 *   dn     an expert and 64 of its down rows on the act rows of its pairs: de
 *   sum    each token: the sum of val de over its slots, in the order of the slots */
#define DGM_TM 16
#define DGM_KC 32

/* The pairs by slot: map (or null: the identity) gives each expert's slot among the experts the
 * GPU holds (-1: the CPU's), cidx (or null) the CPU's ids (-1 for the GPU's pairs). One block of
 * 32 warps; warp w takes pairs [w cs, (w + 1) cs); the counts of each warp by slot (match_any),
 * then each slot's starts and each warp's base, then the positions in the order of the pairs
 * (a warp's lanes in order, its rounds in order): the same each run. */
#define DGM_MAXS 128

__global__ void __launch_bounds__(1024) k_dg_moe_route(const int *idx, int P, int nslot, const int *map,
                                                       int *start, int *ptok, int *ppos, int k, int *tiles,
                                                       int *cidx, int tm)
{
    __shared__ int cnt[32][DGM_MAXS + 1];
    __shared__ int tot[DGM_MAXS + 1];
    const int w = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int cs = (P + 31) / 32, p0 = w * cs, p1 = min(P, p0 + cs);
    for (int e = lane; e <= nslot; e += 32) {
        cnt[w][e] = 0;
    }
    __syncwarp();
    for (int b = p0; b < p1; b += 32) {
        const int p = b + lane;
        int sl = -1;
        if (p < p1) {
            const int e = idx[p];
            sl = map != NULL ? (e >= 0 ? map[e] : -1) : e;
            if (cidx != NULL) {
                cidx[p] = sl >= 0 ? -1 : e;
            }
        }
        const unsigned act = __ballot_sync(0xffffffffu, p < p1 && sl >= 0);
        if (p < p1 && sl >= 0) {
            const unsigned peers = __match_any_sync(act, sl);
            if (lane == __ffs(peers) - 1) {
                cnt[w][sl] += __popc(peers);
            }
        }
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < nslot; ++e) {
            start[e] = acc;
            for (int v = 0; v < 32; ++v) {
                const int c = cnt[v][e];
                cnt[v][e] = acc;        /* warp v's base for slot e */
                acc += c;
            }
        }
        start[nslot] = acc;
        tot[0] = acc;
    }
    __syncthreads();
    for (int b = p0; b < p1; b += 32) {
        const int p = b + lane;
        int sl = -1;
        if (p < p1) {
            const int e = idx[p];
            sl = map != NULL ? (e >= 0 ? map[e] : -1) : e;
        }
        const unsigned act = __ballot_sync(0xffffffffu, p < p1 && sl >= 0);
        if (p < p1) {
            if (sl >= 0) {
                const unsigned peers = __match_any_sync(act, sl);
                const int q = cnt[w][sl] + __popc(peers & ((1u << lane) - 1u));
                ptok[q] = p / k;
                ppos[p] = q;
                __syncwarp(peers);
                if (lane == __ffs(peers) - 1) {
                    cnt[w][sl] += __popc(peers);
                }
            } else {
                ppos[p] = -1;
            }
        }
        __syncwarp();
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nt = 0;
        for (int e = 0; e < nslot; ++e) {
            for (int q = start[e]; q < start[e + 1]; q += tm) {
                tiles[2 * nt] = e;
                tiles[2 * nt + 1] = q;
                ++nt;
            }
        }
        const int cap = (P + DGM_TM - 1) / DGM_TM + nslot;
        for (; nt < cap; ++nt) {
            tiles[2 * nt] = -1;
        }
    }
}

__device__ __forceinline__ float dg_gelu(float x)
{
    return 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

/* A block: a tile of DGM_TM = 16 pairs and 128 weight rows, 32 columns a stage, 128 threads;
 * shared memory holds x by column (xs[c][pair]) and the decoded rows by column (ws[c][row]), so
 * a thread reads its two pairs as a float2 and its rows as float4s.
 *   gu: rows 64 gate rows of the expert and the 64 up rows of the same index; thread (tp, tq):
 *       pairs 2 tp, 2 tp + 1 (tp 0..7), gate rows 4 tq .. 4 tq + 3 and the same up rows (tq 0..15):
 *       16 sums
 *   dn: 128 down rows; thread: pairs 2 tp, 2 tp + 1, rows 8 tq .. 8 tq + 7 (two float4s): 16 sums */
#define DGM_KS 32

__device__ __forceinline__ void dg_dec_rows(const uint8_t *w0, size_t rb, int cols, int k0, int nrows,
                                            float (*ws)[132], int tid)
{
    /* nrows x DGM_KS values: 16-value pieces (row, half), one a thread a round */
    for (int j = tid; j < nrows * 2; j += 128) {
        const int row = j >> 1, half = j & 1;
        uint32_t bits[16];
        kq_bf12_bits<16>(w0 + (size_t)row * rb, cols, k0 + 16 * half, bits);
        #pragma unroll
        for (int v = 0; v < 16; ++v) {
            ws[16 * half + v][row] = kq_bf16((uint16_t)bits[v]);
        }
    }
}

__global__ void __launch_bounds__(128) k_dg_moe_gu(const float *h, const uint8_t *gu, const int *start,
                                                   const int *ptok, const int *tiles, float *act, int hidden,
                                                   int inner, size_t rb)
{
    const int e = tiles[2 * blockIdx.x], r0 = blockIdx.y * 64;
    if (e < 0) {
        return;
    }
    const int s0 = tiles[2 * blockIdx.x + 1], n = min(DGM_TM, start[e + 1] - s0);
    __shared__ __align__(16) float xs[DGM_KS][DGM_TM + 4];
    __shared__ __align__(16) float ws[DGM_KS][132];        /* rows 0..63 gate, 64..127 up */
    const int tid = threadIdx.x, tp = tid & 7, tq = tid >> 3;
    const uint8_t *we = gu + (size_t)e * 2 * inner * rb;
    float g[2][4], u[2][4];
    for (int a = 0; a < 2; ++a) {
        for (int m = 0; m < 4; ++m) {
            g[a][m] = 0.f;
            u[a][m] = 0.f;
        }
    }
    for (int k0 = 0; k0 < hidden; k0 += DGM_KS) {
        __syncthreads();
        for (int q = tid; q < DGM_TM * DGM_KS; q += 128) {
            const int i = q / DGM_KS, c = q % DGM_KS;
            xs[c][i] = i < n ? h[(size_t)ptok[s0 + i] * hidden + k0 + c] : 0.f;
        }
        /* the 64 gate rows r0.., then the 64 up rows inner + r0.. */
        for (int j = tid; j < 128 * 2; j += 128) {
            const int row = j >> 1, half = j & 1;
            const int src = row < 64 ? r0 + row : inner + r0 + (row - 64);
            uint32_t bits[16];
            kq_bf12_bits<16>(we + (size_t)src * rb, hidden, k0 + 16 * half, bits);
            #pragma unroll
            for (int v = 0; v < 16; ++v) {
                ws[16 * half + v][row] = kq_bf16((uint16_t)bits[v]);
            }
        }
        __syncthreads();
        #pragma unroll 4
        for (int c = 0; c < DGM_KS; ++c) {
            const float2 xv = *(const float2 *)&xs[c][2 * tp];
            const float4 gw = *(const float4 *)&ws[c][4 * tq], uw = *(const float4 *)&ws[c][64 + 4 * tq];
            const float x2[2] = {xv.x, xv.y};
            const float gv[4] = {gw.x, gw.y, gw.z, gw.w}, uv[4] = {uw.x, uw.y, uw.z, uw.w};
            #pragma unroll
            for (int a = 0; a < 2; ++a) {
                #pragma unroll
                for (int m = 0; m < 4; ++m) {
                    g[a][m] = fmaf(x2[a], gv[m], g[a][m]);
                    u[a][m] = fmaf(x2[a], uv[m], u[a][m]);
                }
            }
        }
    }
    for (int a = 0; a < 2; ++a) {
        const int i = 2 * tp + a;
        if (i < n) {
            float *ar = act + (size_t)(s0 + i) * inner + r0 + 4 * tq;
            for (int m = 0; m < 4; ++m) {
                ar[m] = dg_gelu(g[a][m]) * u[a][m];
            }
        }
    }
}

__global__ void __launch_bounds__(128) k_dg_moe_dn(const float *act, const uint8_t *dn, const int *start,
                                                   const int *tiles, float *de, int hidden, int inner, size_t rb)
{
    const int e = tiles[2 * blockIdx.x], r0 = blockIdx.y * 128;
    if (e < 0) {
        return;
    }
    const int s0 = tiles[2 * blockIdx.x + 1], n = min(DGM_TM, start[e + 1] - s0);
    __shared__ __align__(16) float xs[DGM_KS][DGM_TM + 4];
    __shared__ __align__(16) float ws[DGM_KS][132];
    const int tid = threadIdx.x, tp = tid & 7, tq = tid >> 3;
    const uint8_t *we = dn + ((size_t)e * hidden + r0) * rb;
    float o[2][8];
    for (int a = 0; a < 2; ++a) {
        for (int m = 0; m < 8; ++m) {
            o[a][m] = 0.f;
        }
    }
    for (int k0 = 0; k0 < inner; k0 += DGM_KS) {
        __syncthreads();
        for (int q = tid; q < DGM_TM * DGM_KS; q += 128) {
            const int i = q / DGM_KS, c = q % DGM_KS;
            xs[c][i] = i < n ? act[(size_t)(s0 + i) * inner + k0 + c] : 0.f;
        }
        dg_dec_rows(we, rb, inner, k0, 128, ws, tid);
        __syncthreads();
        #pragma unroll 4
        for (int c = 0; c < DGM_KS; ++c) {
            const float2 xv = *(const float2 *)&xs[c][2 * tp];
            const float4 w0 = *(const float4 *)&ws[c][8 * tq], w1 = *(const float4 *)&ws[c][8 * tq + 4];
            const float x2[2] = {xv.x, xv.y};
            const float wv[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
            #pragma unroll
            for (int a = 0; a < 2; ++a) {
                #pragma unroll
                for (int m = 0; m < 8; ++m) {
                    o[a][m] = fmaf(x2[a], wv[m], o[a][m]);
                }
            }
        }
    }
    for (int a = 0; a < 2; ++a) {
        const int i = 2 * tp + a;
        if (i < n) {
            float *dr_ = de + (size_t)(s0 + i) * hidden + r0 + 8 * tq;
            for (int m = 0; m < 8; ++m) {
                dr_[m] = o[a][m];
            }
        }
    }
}

/* The tensor cores (mma m16n8k16, bfloat16 in, float32 sums): x in three bfloat16 planes
 * (hi + mid + lo: the 24 bits of float32), each plane times the bfloat16 values of the BF12 rows,
 * so the products are those of float32 (the sums in another order). A block of 4 warps: 16
 * pairs by 128 weight rows, 64 columns a stage in shared memory (rows of 72 halves: no bank
 * conflicts for the fragments).
 *   gu: rows 64 gate + the 64 up rows of the same index; warp w: gate rows 16 w .. 16 w + 15 and
 *       those up rows (two n blocks of 8 each), act = gelu(gate) up at the end
 *   dn: 128 down rows; warp w: rows 32 w .. 32 w + 31 (four n blocks) */
#define DGT_KS 64
#define DGT_LD 72

__device__ __forceinline__ void dg_split3(float v, uint16_t &h, uint16_t &m, uint16_t &l)
{
    h = f2bf_rn(v);
    float r = v - __uint_as_float((uint32_t)h << 16);
    m = f2bf_rn(r);
    r -= __uint_as_float((uint32_t)m << 16);
    l = f2bf_rn(r);
}

/* Tiles of DGT_TM = 32 pairs (two m blocks; the experts the GPU holds have ~30 pairs a layer, so
 * a decoded row serves twice the products of 16). The next stage's global loads (each thread its
 * row's 96 bytes of BF12, 64 columns, and 16 values of x) go out before the current stage's
 * products, in registers, then make the shared tiles. */
#define DGT_TM 32

struct dg_raw {
    uint4 lo[4], hi[2];
    uint32_t E;
    float4 x[4];
};

/* the next stage's bytes: row w of cols (64 columns from k0) and x (the pairs of the tile) */
__device__ __forceinline__ void dg_raw_load(dg_raw &r, const uint8_t *w, int cols, int k0, const float *x, size_t ld,
                                            const int *ptok, int s0, int n, int tid)
{
    const uint4 *lo = (const uint4 *)(w + k0), *hi = (const uint4 *)(w + cols + k0 / 2);
    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        r.lo[q] = lo[q];
    }
    r.hi[0] = hi[0];
    r.hi[1] = hi[1];
    const uint8_t *E = w + cols + cols / 2 + k0 / 32;
    r.E = (uint32_t)E[0] | (uint32_t)E[1] << 8;
    const int i = tid >> 2, c = (tid & 3) * 16;
    if (i < n) {
        const float4 *xr = (const float4 *)(x + (ptok != NULL ? (size_t)ptok[s0 + i] : (size_t)(s0 + i)) * ld + k0 + c);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            r.x[q] = xr[q];
        }
    } else {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            r.x[q] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
}

/* the shared tiles from a stage's bytes: the thread's row of 64 bfloat16 values, and its 16 values
 * of x in the three planes */
__device__ __forceinline__ void dg_raw_store(const dg_raw &r, uint16_t (*ws)[DGT_LD], uint16_t (*xs)[DGT_TM][DGT_LD],
                                             int tid)
{
    const uint32_t *lw = (const uint32_t *)r.lo, *hw = (const uint32_t *)r.hi;
    #pragma unroll
    for (int g = 0; g < 2; ++g) {
        const uint32_t E2 = (((r.E >> (8 * g)) & 0xffu) | 256u) * 0x00010001u;
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            uint32_t p[8];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t l = lw[8 * g + 4 * h + k], hh = hw[4 * g + k];
                p[2 * k] = bf12_frag(l, hh, 4 * h, E2);
                p[2 * k + 1] = bf12_frag(l >> 16, hh >> 16, 4 * h, E2);
            }
            uint16_t *dst = ws[tid] + 32 * g + 16 * h;
            *(uint4 *)dst = make_uint4(p[0], p[1], p[2], p[3]);
            *(uint4 *)(dst + 8) = make_uint4(p[4], p[5], p[6], p[7]);
        }
    }
    const int i = tid >> 2, c = (tid & 3) * 16;
    const float *v = (const float *)r.x;
    #pragma unroll
    for (int q = 0; q < 2; ++q) {
        uint32_t ph[4], pm[4], pl[4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            uint16_t h0, m0, l0, h1, m1, l1;
            dg_split3(v[8 * q + 2 * u], h0, m0, l0);
            dg_split3(v[8 * q + 2 * u + 1], h1, m1, l1);
            ph[u] = (uint32_t)h0 | (uint32_t)h1 << 16;
            pm[u] = (uint32_t)m0 | (uint32_t)m1 << 16;
            pl[u] = (uint32_t)l0 | (uint32_t)l1 << 16;
        }
        *(uint4 *)&xs[0][i][c + 8 * q] = make_uint4(ph[0], ph[1], ph[2], ph[3]);
        *(uint4 *)&xs[1][i][c + 8 * q] = make_uint4(pm[0], pm[1], pm[2], pm[3]);
        *(uint4 *)&xs[2][i][c + 8 * q] = make_uint4(pl[0], pl[1], pl[2], pl[3]);
    }
}

/* the products of k step kk: two m blocks (pairs 0..15, 16..31) by NB n blocks; each plane's 16
 * products from zero, then into the float32 sums (lo + mid, then hi) */
template <int NB>
__device__ __forceinline__ void dg_mma2(uint16_t (*xs)[DGT_TM][DGT_LD], uint16_t (*ws)[DGT_LD], int kk,
                                        const int *nb, float (*acc)[NB][4], int lane)
{
    const int g = lane >> 2, tg = lane & 3, c = kk + 2 * tg;
    uint32_t b[NB][2];
    #pragma unroll
    for (int j = 0; j < NB; ++j) {
        b[j][0] = *(const uint32_t *)&ws[nb[j] + g][c];
        b[j][1] = *(const uint32_t *)&ws[nb[j] + g][c + 8];
    }
    #pragma unroll
    for (int mb = 0; mb < 2; ++mb) {
        uint32_t a[3][4];
        #pragma unroll
        for (int pl = 0; pl < 3; ++pl) {
            a[pl][0] = *(const uint32_t *)&xs[pl][16 * mb + g][c];
            a[pl][1] = *(const uint32_t *)&xs[pl][16 * mb + g + 8][c];
            a[pl][2] = *(const uint32_t *)&xs[pl][16 * mb + g][c + 8];
            a[pl][3] = *(const uint32_t *)&xs[pl][16 * mb + g + 8][c + 8];
        }
        #pragma unroll
        for (int j = 0; j < NB; ++j) {
            float pp[3][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int pl = 0; pl < 3; ++pl) {
                mma16816_bf16(pp[pl], a[pl], b[j]);
            }
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                acc[mb][j][q] += (pp[2][q] + pp[1][q]) + pp[0][q];
            }
        }
    }
}

__global__ void __launch_bounds__(128) k_dg_moe_gu_tc(const float *h, const uint8_t *gu, const int *start,
                                                      const int *ptok, const int *tiles, float *act, int hidden,
                                                      int inner, size_t rb)
{
    const int e = tiles[2 * blockIdx.x], r0 = blockIdx.y * 64;
    if (e < 0) {
        return;
    }
    const int s0 = tiles[2 * blockIdx.x + 1], n = min(DGT_TM, start[e + 1] - s0);
    __shared__ __align__(16) uint16_t xs[3][DGT_TM][DGT_LD];
    __shared__ __align__(16) uint16_t ws[128][DGT_LD];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const int wrow = tid < 64 ? r0 + tid : inner + r0 + (tid - 64);
    const uint8_t *wr = gu + ((size_t)e * 2 * inner + wrow) * rb;
    const int nb[4] = {16 * w, 16 * w + 8, 64 + 16 * w, 64 + 16 * w + 8};
    float acc[2][4][4];
    for (int mb = 0; mb < 2; ++mb) {
        for (int j = 0; j < 4; ++j) {
            for (int q = 0; q < 4; ++q) {
                acc[mb][j][q] = 0.f;
            }
        }
    }
    dg_raw r;
    dg_raw_load(r, wr, hidden, 0, h, (size_t)hidden, ptok, s0, n, tid);
    for (int k0 = 0; k0 < hidden; k0 += DGT_KS) {
        __syncthreads();
        dg_raw_store(r, ws, xs, tid);
        __syncthreads();
        if (k0 + DGT_KS < hidden) {
            dg_raw_load(r, wr, hidden, k0 + DGT_KS, h, (size_t)hidden, ptok, s0, n, tid);
        }
        #pragma unroll
        for (int kk = 0; kk < DGT_KS; kk += 16) {
            dg_mma2<4>(xs, ws, kk, nb, acc, lane);
        }
    }
    const int g = lane >> 2, tg = lane & 3;
    for (int mb = 0; mb < 2; ++mb) {
        for (int j = 0; j < 2; ++j) {
            const int col = r0 + 16 * w + 8 * j + 2 * tg;
            for (int hh = 0; hh < 2; ++hh) {
                const int i = 16 * mb + g + 8 * hh;
                if (i < n) {
                    float *ar = act + (size_t)(s0 + i) * inner + col;
                    ar[0] = dg_gelu(acc[mb][j][2 * hh]) * acc[mb][2 + j][2 * hh];
                    ar[1] = dg_gelu(acc[mb][j][2 * hh + 1]) * acc[mb][2 + j][2 * hh + 1];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(128) k_dg_moe_dn_tc(const float *act, const uint8_t *dn, const int *start,
                                                      const int *tiles, float *de, int hidden, int inner, size_t rb)
{
    const int e = tiles[2 * blockIdx.x], r0 = blockIdx.y * 128;
    if (e < 0) {
        return;
    }
    const int s0 = tiles[2 * blockIdx.x + 1], n = min(DGT_TM, start[e + 1] - s0);
    __shared__ __align__(16) uint16_t xs[3][DGT_TM][DGT_LD];
    __shared__ __align__(16) uint16_t ws[128][DGT_LD];
    const int tid = threadIdx.x, w = tid >> 5, lane = tid & 31;
    const uint8_t *wr = dn + ((size_t)e * hidden + r0 + tid) * rb;
    const int nb[4] = {32 * w, 32 * w + 8, 32 * w + 16, 32 * w + 24};
    float acc[2][4][4];
    for (int mb = 0; mb < 2; ++mb) {
        for (int j = 0; j < 4; ++j) {
            for (int q = 0; q < 4; ++q) {
                acc[mb][j][q] = 0.f;
            }
        }
    }
    dg_raw r;
    dg_raw_load(r, wr, inner, 0, act, (size_t)inner, NULL, s0, n, tid);
    for (int k0 = 0; k0 < inner; k0 += DGT_KS) {
        __syncthreads();
        dg_raw_store(r, ws, xs, tid);
        __syncthreads();
        if (k0 + DGT_KS < inner) {
            dg_raw_load(r, wr, inner, k0 + DGT_KS, act, (size_t)inner, NULL, s0, n, tid);
        }
        #pragma unroll
        for (int kk = 0; kk < DGT_KS; kk += 16) {
            dg_mma2<4>(xs, ws, kk, nb, acc, lane);
        }
    }
    const int g = lane >> 2, tg = lane & 3;
    for (int mb = 0; mb < 2; ++mb) {
        for (int j = 0; j < 4; ++j) {
            const int col = r0 + 32 * w + 8 * j + 2 * tg;
            for (int hh = 0; hh < 2; ++hh) {
                const int i = 16 * mb + g + 8 * hh;
                if (i < n) {
                    float *dr_ = de + (size_t)(s0 + i) * hidden + col;
                    dr_[0] = acc[mb][j][2 * hh];
                    dr_[1] = acc[mb][j][2 * hh + 1];
                }
            }
        }
    }
}

static int dg_moe_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_DG_MOE_TC");
        on = v == NULL ? 1 : atoi(v) != 0;
    }
    return on && gg_caps.tc;
}

__global__ void k_dg_moe_sum(const float *de, const float *val, const int *ppos, float *out, int k, int hidden)
{
    const int tok = blockIdx.x;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float s = 0.f;
        for (int j = 0; j < k; ++j) {
            const int p = tok * k + j;
            if (ppos[p] >= 0) {
                s = fmaf(val[p], de[(size_t)ppos[p] * hidden + c], s);
            }
        }
        out[(size_t)tok * hidden + c] = s;
    }
}

static void dg_moe(const float *h, const int *idx, const float *val, const uint8_t *gu, const uint8_t *dn, float *out,
                   int *start, int *ptok, int *ppos, float *act, float *de, int t, int k, int E, int hidden,
                   int inner, const int *map, int *cidx, cudaStream_t s)
{
    const int P = t * k;
    /* the tile table after start (E + 1 ints): its room is in the start buffer (DG_MOE's operand 6
     * holds E + 1 + 2 (P / DGM_TM + 1 + E) ints) */
    int *tiles = start + E + 1;
    const unsigned nt = (unsigned)((P + DGM_TM - 1) / DGM_TM + E);
    const int tc = dg_moe_tc_on() && inner % DGT_KS == 0 && hidden % DGT_KS == 0;
    k_dg_moe_route<<<1, 1024, 0, s>>>(idx, P, E, map, start, ptok, ppos, k, tiles, cidx, tc ? DGT_TM : DGM_TM);
    if (E > 0) {
        const size_t rg = kq_row_bytes(KQ_BF12, hidden), rd = kq_row_bytes(KQ_BF12, inner);
        if (tc) {
            k_dg_moe_gu_tc<<<dim3(nt, (unsigned)(inner / 64)), 128, 0, s>>>(h, gu, start, ptok, tiles, act, hidden,
                                                                            inner, rg);
            k_dg_moe_dn_tc<<<dim3(nt, (unsigned)(hidden / 128)), 128, 0, s>>>(act, dn, start, tiles, de, hidden,
                                                                              inner, rd);
        } else {
            k_dg_moe_gu<<<dim3(nt, (unsigned)(inner / 64)), 128, 0, s>>>(h, gu, start, ptok, tiles, act, hidden,
                                                                         inner, rg);
            k_dg_moe_dn<<<dim3(nt, (unsigned)(hidden / 128)), 128, 0, s>>>(act, dn, start, tiles, de, hidden,
                                                                           inner, rd);
        }
    }
    k_dg_moe_sum<<<(unsigned)t, 256, 0, s>>>(de, val, ppos, out, k, hidden);
}

/* DG_SPLIT: the CPU's ids of a layer's pairs (cidx: the expert, or -1 when the GPU holds it: map
 * gives its slot), before the GPU's experts, so that the CPU can start on them */
__global__ void k_dg_split(const int *idx, const int *map, int *cidx, int P)
{
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p < P) {
        const int e = idx[p];
        cidx[p] = e >= 0 && map[e] >= 0 ? -1 : e;
    }
}

/* DG_ATTN, tiled (the default; RECLANK_DG_ATTN_TILE=0: k_dg_attn above): float32 on the CUDA
 * cores, the same sums but in another order. A block: BR rows of (query, head) that read one KV
 * head (row r: query q0 + r / rep, head g rep + r % rep), one of DGA_NS parts of their keys
 * (split-K: the block's keys [least lo, largest hi) in DGA_NS equal parts; k_dg_attn_comb joins
 * the parts). 256 threads; keys in tiles of DGA_BK = 32:
 *   S = Q K^T, 32 dims a stage (Q's rows and K's keys in shared memory, read as float4): thread
 *     (ry, kx) the rows ry RPT .. + RPT - 1 and the keys 2 kx, 2 kx + 1 (RPT = BR / 16)
 *   the online softmax of each row (its 16 threads by shuffles), P into shared memory
 *   O += P V: the thread's rows and the dims kx 4 + 64 j (V's tile of 32 keys in shared memory)
 * D 256: BR 64; D 512: BR 32 (the 64 sums of O a thread either way). */
#define DGA_BK 32
#define DGA_DC 32
#define DGA_NS 4
#define DGA_LD (DGA_DC + 4)

template <int D, int BR>
__global__ void __launch_bounds__(256) k_dg_attn_t(const float *q, const float *K, const float *V, float *part,
                                                   const int *lo, const int *hi, int t, int heads, int kvh)
{
    constexpr int RPT = BR / 16, NJ = D / 64;
    extern __shared__ __align__(16) float dga_sm[];
    float *qs = dga_sm;                         /* [BR][DGA_LD] */
    float *ks = qs + BR * DGA_LD;               /* [DGA_BK][DGA_LD] */
    float *vs = ks + DGA_BK * DGA_LD;           /* [DGA_BK][D] */
    float *ps = vs + DGA_BK * D;                /* [DGA_BK][BR] */
    __shared__ int rlo[BR], rhi[BR], blk[2];
    const int g = blockIdx.y, sp = blockIdx.z, rep = heads / kvh;
    const int tid = threadIdx.x, ry = tid >> 4, kx = tid & 15;
    const int row0 = blockIdx.x * BR, nrows = t * rep;
    if (tid < BR) {
        const int rr = row0 + tid;
        rlo[tid] = rr < nrows ? lo[rr / rep] : 0;
        rhi[tid] = rr < nrows ? hi[rr / rep] : 0;
    }
    __syncthreads();
    if (tid == 0) {
        int a = INT_MAX, b = 0;
        for (int r = 0; r < BR; ++r) {
            if (row0 + r < nrows) {
                a = min(a, rlo[r]);
                b = max(b, rhi[r]);
            }
        }
        if (a == INT_MAX) {
            a = 0;
        }
        const int len = max(0, b - a), ch = (len + DGA_NS - 1) / DGA_NS;
        blk[0] = min(b, a + sp * ch);
        blk[1] = min(b, a + (sp + 1) * ch);
    }
    __syncthreads();
    const int ks0 = blk[0], ke = blk[1];
    float o[RPT][NJ][4], m[RPT], l[RPT];
    #pragma unroll
    for (int i = 0; i < RPT; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
            o[i][j][0] = o[i][j][1] = o[i][j][2] = o[i][j][3] = 0.f;
        }
    }
    const size_t kst = (size_t)kvh * D;
    for (int k0 = ks0; k0 < ke; k0 += DGA_BK) {
        const int nk = min(DGA_BK, ke - k0);
        float s[RPT][2];
        #pragma unroll
        for (int i = 0; i < RPT; ++i) {
            s[i][0] = s[i][1] = 0.f;
        }
        for (int d0 = 0; d0 < D; d0 += DGA_DC) {
            __syncthreads();
            for (int e = tid; e < BR * (DGA_DC / 4); e += 256) {
                const int r = e / (DGA_DC / 4), c = (e % (DGA_DC / 4)) * 4, rr = row0 + r;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (rr < nrows) {
                    v = *(const float4 *)(q + ((size_t)(rr / rep) * heads + g * rep + rr % rep) * D + d0 + c);
                }
                *(float4 *)(qs + r * DGA_LD + c) = v;
            }
            for (int e = tid; e < DGA_BK * (DGA_DC / 4); e += 256) {
                const int r = e / (DGA_DC / 4), c = (e % (DGA_DC / 4)) * 4;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (r < nk) {
                    v = *(const float4 *)(K + (size_t)(k0 + r) * kst + (size_t)g * D + d0 + c);
                }
                *(float4 *)(ks + r * DGA_LD + c) = v;
            }
            __syncthreads();
            #pragma unroll
            for (int d = 0; d < DGA_DC; d += 4) {
                const float4 k0v = *(const float4 *)(ks + (2 * kx) * DGA_LD + d);
                const float4 k1v = *(const float4 *)(ks + (2 * kx + 1) * DGA_LD + d);
                #pragma unroll
                for (int i = 0; i < RPT; ++i) {
                    const float4 qv = *(const float4 *)(qs + (ry * RPT + i) * DGA_LD + d);
                    s[i][0] = fmaf(qv.x, k0v.x, s[i][0]);
                    s[i][0] = fmaf(qv.y, k0v.y, s[i][0]);
                    s[i][0] = fmaf(qv.z, k0v.z, s[i][0]);
                    s[i][0] = fmaf(qv.w, k0v.w, s[i][0]);
                    s[i][1] = fmaf(qv.x, k1v.x, s[i][1]);
                    s[i][1] = fmaf(qv.y, k1v.y, s[i][1]);
                    s[i][1] = fmaf(qv.z, k1v.z, s[i][1]);
                    s[i][1] = fmaf(qv.w, k1v.w, s[i][1]);
                }
            }
        }
        /* V's tile, then the softmax of the tile's scores and P */
        for (int e = tid; e < DGA_BK * (D / 4); e += 256) {
            const int r = e / (D / 4), c = (e % (D / 4)) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (r < nk) {
                v = *(const float4 *)(V + (size_t)(k0 + r) * kst + (size_t)g * D + c);
            }
            *(float4 *)(vs + r * D + c) = v;
        }
        #pragma unroll
        for (int i = 0; i < RPT; ++i) {
            const int r = ry * RPT + i;
            float sv[2];
            float mx = -INFINITY;
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                const int key = k0 + 2 * kx + u;
                const int ok = 2 * kx + u < nk && key >= rlo[r] && key < rhi[r];
                sv[u] = ok ? s[i][u] : -INFINITY;
                mx = fmaxf(mx, sv[u]);
            }
            for (int w = 8; w > 0; w >>= 1) {
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, w));
            }
            const float mn = fmaxf(m[i], mx);
            float p[2] = {0.f, 0.f}, sc = 1.f;
            if (mn != -INFINITY) {
                sc = m[i] == -INFINITY ? 0.f : expf(m[i] - mn);
                p[0] = sv[0] == -INFINITY ? 0.f : expf(sv[0] - mn);
                p[1] = sv[1] == -INFINITY ? 0.f : expf(sv[1] - mn);
            }
            float ps_ = p[0] + p[1];
            for (int w = 8; w > 0; w >>= 1) {
                ps_ += __shfl_xor_sync(0xffffffffu, ps_, w);
            }
            l[i] = l[i] * sc + ps_;
            m[i] = mn;
            #pragma unroll
            for (int j = 0; j < NJ; ++j) {
                o[i][j][0] *= sc;
                o[i][j][1] *= sc;
                o[i][j][2] *= sc;
                o[i][j][3] *= sc;
            }
            ps[(2 * kx) * BR + r] = p[0];
            ps[(2 * kx + 1) * BR + r] = p[1];
        }
        __syncthreads();
        for (int j = 0; j < nk; ++j) {
            float pv[RPT];
            #pragma unroll
            for (int i = 0; i < RPT; ++i) {
                pv[i] = ps[j * BR + ry * RPT + i];
            }
            #pragma unroll
            for (int jj = 0; jj < NJ; ++jj) {
                const float4 v = *(const float4 *)(vs + j * D + kx * 4 + 64 * jj);
                #pragma unroll
                for (int i = 0; i < RPT; ++i) {
                    o[i][jj][0] = fmaf(pv[i], v.x, o[i][jj][0]);
                    o[i][jj][1] = fmaf(pv[i], v.y, o[i][jj][1]);
                    o[i][jj][2] = fmaf(pv[i], v.z, o[i][jj][2]);
                    o[i][jj][3] = fmaf(pv[i], v.w, o[i][jj][3]);
                }
            }
        }
    }
    /* the part: O (not divided), then m and l, of each row (query, head) */
    const size_t rows_all = (size_t)t * heads;
    #pragma unroll
    for (int i = 0; i < RPT; ++i) {
        const int rr = row0 + ry * RPT + i;
        if (rr >= nrows) {
            continue;
        }
        const size_t row = (size_t)(rr / rep) * heads + g * rep + rr % rep;
        float *po = part + ((size_t)sp * rows_all + row) * D;
        #pragma unroll
        for (int jj = 0; jj < NJ; ++jj) {
            *(float4 *)(po + kx * 4 + 64 * jj) = make_float4(o[i][jj][0], o[i][jj][1], o[i][jj][2], o[i][jj][3]);
        }
        if (kx == 0) {
            float *ml = part + (size_t)DGA_NS * rows_all * D + ((size_t)sp * rows_all + row) * 2;
            ml[0] = m[i];
            ml[1] = l[i];
        }
    }
}

/* the parts of each row joined: O = sum_s e^(m_s - M) O_s / sum_s e^(m_s - M) l_s */
__global__ void k_dg_attn_comb(const float *part, float *out, int rows, int D)
{
    const int row = blockIdx.x;
    const float *ml = part + (size_t)DGA_NS * rows * D;
    float M = -INFINITY;
    for (int s = 0; s < DGA_NS; ++s) {
        M = fmaxf(M, ml[((size_t)s * rows + row) * 2]);
    }
    float w[DGA_NS], L = 0.f;
    for (int s = 0; s < DGA_NS; ++s) {
        const float ms = ml[((size_t)s * rows + row) * 2];
        w[s] = ms == -INFINITY ? 0.f : expf(ms - M);
        L += w[s] * ml[((size_t)s * rows + row) * 2 + 1];
    }
    const float inv = L > 0.f ? 1.f / L : 0.f;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float a = 0.f;
        for (int s = 0; s < DGA_NS; ++s) {
            if (w[s] != 0.f) {
                a = fmaf(w[s], part[((size_t)s * rows + row) * D + c], a);
            }
        }
        out[(size_t)row * D + c] = a * inv;
    }
}

static int dg_attn_tile_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_DG_ATTN_TILE");
        on = v == NULL ? 1 : atoi(v) != 0;
    }
    return on;
}

/* the bytes of DG_ATTN's scratch (operand 10) for t queries of heads x hd: DGA_NS parts of O, m, l */
static size_t dg_attn_part_floats(int t, int heads, int hd)
{
    return (size_t)DGA_NS * t * heads * (hd + 2);
}

static void dg_attn_tiled(const float *q, const float *K, const float *V, float *out, float *part, const int *lo,
                          const int *hi, int t, int heads, int kvh, int hd, cudaStream_t s)
{
    static int attr = 0;
    const int smem256 = (64 * DGA_LD + DGA_BK * DGA_LD + DGA_BK * 256 + DGA_BK * 64) * 4;
    const int smem512 = (32 * DGA_LD + DGA_BK * DGA_LD + DGA_BK * 512 + DGA_BK * 32) * 4;
    if (!attr) {
        cudaFuncSetAttribute(k_dg_attn_t<256, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem256);
        cudaFuncSetAttribute(k_dg_attn_t<512, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem512);
        attr = 1;
    }
    const int rep = heads / kvh;
    if (hd == 256) {
        dim3 grid((unsigned)((t * rep + 63) / 64), (unsigned)kvh, DGA_NS);
        k_dg_attn_t<256, 64><<<grid, 256, smem256, s>>>(q, K, V, part, lo, hi, t, heads, kvh);
    } else {
        dim3 grid((unsigned)((t * rep + 31) / 32), (unsigned)kvh, DGA_NS);
        k_dg_attn_t<512, 32><<<grid, 256, smem512, s>>>(q, K, V, part, lo, hi, t, heads, kvh);
    }
    k_dg_attn_comb<<<(unsigned)(t * heads), 128, 0, s>>>(part, out, t * heads, hd);
}
