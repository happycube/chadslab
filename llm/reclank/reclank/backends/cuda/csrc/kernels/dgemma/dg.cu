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
 * u (the thread whose chunk holds it then walks it). */
#define DG_ST 1024

__global__ void __launch_bounds__(DG_ST) k_dg_sample(float *z, float *probs, int vocab, const float *temp,
                                                     const int *ban, const int *eos, int neos, const float *u,
                                                     float *ent, int *am, int *draw)
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
    /* 1: the processed logits, their max and its first index */
    float mx = -INFINITY;
    int mi = 0;
    const int banned = ban[i];
    for (int c = tid; c < vocab; c += DG_ST) {
        float v = zr[c] * inv_t;
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
        }
    }
    for (int w = 16; w > 0; w >>= 1) {
        float om = __shfl_xor_sync(0xffffffffu, mx, w);
        int oi = __shfl_xor_sync(0xffffffffu, mi, w);
        if (om > mx || (om == mx && oi < mi)) {
            mx = om;
            mi = oi;
        }
    }
    if ((tid & 31) == 0) {
        red[tid >> 5] = mx;
        redi[tid >> 5] = mi;
    }
    __syncthreads();
    if (tid == 0) {
        float b = red[0];
        int bi = redi[0];
        for (int k = 1; k < DG_ST / 32; ++k) {
            if (red[k] > b || (red[k] == b && redi[k] < bi)) {
                b = red[k];
                bi = redi[k];
            }
        }
        s_max = b;
        am[i] = bi;
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
        /* the entropy: log S - sum e (z - M) / S */
        ent[i] = logf(sa) - sb / sa;
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
                      int neos, const float *u, float *ent, int *am, int *draw, cudaStream_t s)
{
    k_dg_sample<<<(unsigned)t, DG_ST, 0, s>>>(z, probs, vocab, temp, ban, eos, neos, u, ent, am, draw);
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

__global__ void k_dg_moe_route(const int *idx, int P, int E, int *start, int *ptok, int *ppos, int k,
                               int *tiles)
{
    if (threadIdx.x != 0) {
        return;
    }
    for (int e = 0; e <= E; ++e) {
        start[e] = 0;
    }
    for (int p = 0; p < P; ++p) {
        start[idx[p] + 1] += 1;
    }
    for (int e = 0; e < E; ++e) {
        start[e + 1] += start[e];
    }
    /* the fill: the next free position of each expert (start[e] moves on, then back) */
    for (int p = 0; p < P; ++p) {
        const int e = idx[p];
        const int q = start[e]++;
        ptok[q] = p / k;
        ppos[p] = q;
    }
    for (int e = E; e > 0; --e) {
        start[e] = start[e - 1];
    }
    start[0] = 0;
    /* the tiles: (expert, first pair) of each DGM_TM pairs of an expert; -1 past the last */
    int nt = 0;
    for (int e = 0; e < E; ++e) {
        for (int p0 = start[e]; p0 < start[e + 1]; p0 += DGM_TM) {
            tiles[2 * nt] = e;
            tiles[2 * nt + 1] = p0;
            ++nt;
        }
    }
    const int cap = (P + DGM_TM - 1) / DGM_TM + E;
    for (; nt < cap; ++nt) {
        tiles[2 * nt] = -1;
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

__global__ void k_dg_moe_sum(const float *de, const float *val, const int *ppos, float *out, int k, int hidden)
{
    const int tok = blockIdx.x;
    for (int c = threadIdx.x; c < hidden; c += blockDim.x) {
        float s = 0.f;
        for (int j = 0; j < k; ++j) {
            const int p = tok * k + j;
            s = fmaf(val[p], de[(size_t)ppos[p] * hidden + c], s);
        }
        out[(size_t)tok * hidden + c] = s;
    }
}

static void dg_moe(const float *h, const int *idx, const float *val, const uint8_t *gu, const uint8_t *dn, float *out,
                   int *start, int *ptok, int *ppos, float *act, float *de, int t, int k, int E, int hidden,
                   int inner, cudaStream_t s)
{
    const int P = t * k;
    /* the tile table after start (E + 1 ints): its room is in the start buffer (DG_MOE's operand 6
     * holds E + 1 + 2 (P / DGM_TM + 1 + E) ints) */
    int *tiles = start + E + 1;
    const unsigned nt = (unsigned)((P + DGM_TM - 1) / DGM_TM + E);
    k_dg_moe_route<<<1, 32, 0, s>>>(idx, P, E, start, ptok, ppos, k, tiles);
    k_dg_moe_gu<<<dim3(nt, (unsigned)(inner / 64)), 128, 0, s>>>(
        h, gu, start, ptok, tiles, act, hidden, inner, kq_row_bytes(KQ_BF12, hidden));
    k_dg_moe_dn<<<dim3(nt, (unsigned)(hidden / 128)), 128, 0, s>>>(
        act, dn, start, tiles, de, hidden, inner, kq_row_bytes(KQ_BF12, inner));
    k_dg_moe_sum<<<(unsigned)t, 256, 0, s>>>(de, val, ppos, out, k, hidden);
}
