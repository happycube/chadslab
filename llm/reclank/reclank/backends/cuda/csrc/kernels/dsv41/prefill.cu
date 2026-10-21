/* DeepSeek V4.1: the kernels of a prompt group that take most of its GPU
 * time, in a module of their own: reclank_cuda.cu includes it after step.cu,
 * and a bench builds it alone in seconds (bench/ds_prefill.cu, on operands
 * dumped from a real run: RECLANK_DS_DUMP). The attention of a group
 * (k_ds_attn_g; k_ds_attn_g2 its bits faster; k_ds_attn_tc on the tensor
 * cores); the experts' are in hot.cu. */
#pragma once
#include "../frag.cuh"
#include "common.cuh"

/* GP_DS_ATTN for a group (t > 8, no split): the operands of k_ds_attn; a
 * block for each DS_GHB heads of a query, one pass over its keys with an
 * online softmax: each chunk of DS_GKC keys decoded once into shared memory
 * (from any form of the caches) for the heads, a warp for each key's
 * scores, then thread h of the first DS_GHB the max and the sum of head h,
 * then every thread its 2 columns of the outputs of the heads. The sink
 * joins the sum at the end (model.py sparse_attn). */
#define DS_GHB 16
#define DS_GKC 4
__global__ void __launch_bounds__(256) k_ds_attn_g(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int win = DI(3), hd = DI(10), maxk = DI(12), h0 = blockIdx.x * DS_GHB, j = blockIdx.y, H = DI(9);
    int64_t pos = di(r, e, 2), p = pos + j;
    int64_t lo = p - win + 1 > 0 ? p - win + 1 : 0;
    int nr = lo < pos ? (int)(pos - lo) : 0;
    int g0 = lo > pos ? (int)(lo - pos) : 0, ng = j - g0 + 1;
    int mode = r->tag[16] != GP_T_NONE ? DI(16) : 0;
    if (mode & 2) {
        /* bounded replay (V4.1 3.2.2): the window starts at operand 18 (the
         * first position of the replay) */
        int64_t w0 = di(r, e, 18);
        lo = lo > w0 ? lo : w0;
        nr = lo < pos ? (int)(pos - lo) : 0;
        g0 = lo > pos ? (int)(lo - pos) : 0;
        ng = j - g0 + 1;
    }
    if (mode & 1) {
        nr = (int)(pos < win ? pos : win);
        lo = pos - nr;
        g0 = 0;
        ng = DI(14);
    }
    const void *ckv = DP(const void, 4);
    int form = ds_form(r, e, 17);
    const int *sel = DP(const int, 5) + (size_t)j * maxk;
    int nk = nr + ng + (ckv ? DP(const int, 6)[j] : 0);
    const void *ring = DP(const void, 1);
    const float *kvq = DP(const float, 13);
    float scale = df(r, e, 11);
    __shared__ __align__(16) float qs[DS_GHB][512];
    __shared__ __align__(16) float ks[DS_GKC][512];
    __shared__ float ps[DS_GHB][DS_GKC];
    __shared__ float mrun[DS_GHB], lrun[DS_GHB], corr[DS_GHB];
    const float *q = DP(const float, 0) + ((size_t)j * H + h0) * hd;
    for (int i = threadIdx.x; i < DS_GHB * hd; i += blockDim.x) {
        qs[i / hd][i % hd] = q[i];
    }
    if (threadIdx.x < DS_GHB) {
        mrun[threadIdx.x] = -INFINITY;
        lrun[threadIdx.x] = 0.f;
    }
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    float acc[DS_GHB][2];
    #pragma unroll
    for (int h = 0; h < DS_GHB; ++h) {
        acc[h][0] = acc[h][1] = 0.f;
    }
    int nq = hd / 4;
    for (int m0 = 0; m0 < nk; m0 += DS_GKC) {
        int nc = nk - m0 < DS_GKC ? nk - m0 : DS_GKC;
        __syncthreads();
        for (int idx = threadIdx.x; idx < nc * nq; idx += blockDim.x) {
            int kk = idx / nq, c4 = 4 * (idx % nq);
            ds_krow kr = ds_akey(m0 + kk, nr, ng, lo, g0, win, ring, kvq, ckv, sel, form);
            *(float4 *)&ks[kk][c4] = ds_rld4(kr.p, kr.row, c4, hd, kr.form);
        }
        __syncthreads();
        if (warp < nc) {
            float d[DS_GHB];
            #pragma unroll
            for (int h = 0; h < DS_GHB; ++h) {
                d[h] = 0.f;
            }
            for (int c = 4 * lane; c < hd; c += 128) {
                float4 k4 = *(const float4 *)&ks[warp][c];
                #pragma unroll
                for (int h = 0; h < DS_GHB; ++h) {
                    float4 q4 = *(const float4 *)&qs[h][c];
                    d[h] += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
                }
            }
            #pragma unroll
            for (int h = 0; h < DS_GHB; ++h) {
                for (int o = 16; o > 0; o >>= 1) {
                    d[h] += __shfl_xor_sync(RC_FULL_MASK, d[h], o);
                }
            }
            if (lane == 0) {
                #pragma unroll
                for (int h = 0; h < DS_GHB; ++h) {
                    ps[h][warp] = d[h] * scale;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x < DS_GHB) {
            int h = threadIdx.x;
            float mx = mrun[h];
            for (int k = 0; k < nc; ++k) {
                mx = fmaxf(mx, ps[h][k]);
            }
            float cr = mrun[h] == -INFINITY ? 0.f : expf(mrun[h] - mx);
            float l = lrun[h] * cr;
            for (int k = 0; k < nc; ++k) {
                float pv = expf(ps[h][k] - mx);
                ps[h][k] = pv;
                l += pv;
            }
            mrun[h] = mx;
            lrun[h] = l;
            corr[h] = cr;
        }
        __syncthreads();
        #pragma unroll
        for (int h = 0; h < DS_GHB; ++h) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                int col = threadIdx.x + 256 * cc;
                if (col < hd) {
                    float a = acc[h][cc] * corr[h];
                    for (int k = 0; k < nc; ++k) {
                        a += ps[h][k] * ks[k][col];
                    }
                    acc[h][cc] = a;
                }
            }
        }
    }
    __syncthreads();
    float *out = DP(float, 8) + ((size_t)j * H + h0) * hd;
    #pragma unroll
    for (int h = 0; h < DS_GHB; ++h) {
        float den = lrun[h] + expf(DP(const float, 7)[h0 + h] - mrun[h]);
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
            int col = threadIdx.x + 256 * cc;
            if (col < hd) {
                out[(size_t)h * hd + col] = acc[h][cc] / den;
            }
        }
    }
}

/* k_ds_attn_g's sums (the same bits) in rounds of DS_G2KC keys (hd 512): the keys of a round decoded
 * once into shared memory; warp w the scores of heads 4 (w % 4) .. + 3 and keys 8 (w / 4) .. + 7, lane
 * L the sums of columns 4 L + 128 i (k_ds_attn_g's lane L) with its heads' q in registers, then the
 * 32 sums of the warp's (head, key) by a butterfly that leaves each in one lane (the tree of
 * k_ds_attn_g's xor shuffles: the same bits); the max and the correction of each chunk of DS_GKC
 * keys in order (16 threads), the exps (a thread each), and every thread its 2 columns of the outputs
 * of the heads, the chunks in order (a = a corr, then a += p k): 5 barriers a round, not 4 a chunk,
 * and no q from shared memory. */
#define DS_G2KC 16
/* a step of the butterfly over v (32 sums a lane, half of them left): the values chosen, not their
 * addresses (selp: the array stays in registers) */
#define DS_BFLY(v, W)                                                                                \
    do {                                                                                             \
        const int up_ = (lane & (W)) != 0;                                                           \
        _Pragma("unroll") for (int s_ = 0; s_ < (W); ++s_)                                           \
        {                                                                                            \
            float lo_ = v[s_], hi_ = v[s_ + (W)], keep_, send_;                                      \
            asm("{.reg .pred p; setp.ne.s32 p, %4, 0; selp.f32 %0, %2, %3, p; selp.f32 %1, %3, %2, p;}" \
                : "=f"(keep_), "=f"(send_) : "f"(hi_), "f"(lo_), "r"(up_));                           \
            v[s_] = keep_ + __shfl_xor_sync(RC_FULL_MASK, send_, (W));                               \
        }                                                                                            \
    } while (0)
__global__ void __launch_bounds__(256, 1) k_ds_attn_g2(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int win = DI(3), maxk = DI(12), h0 = blockIdx.x * DS_GHB, j = blockIdx.y, H = DI(9);
    const int hd = 512;
    int64_t pos = di(r, e, 2), p = pos + j;
    int64_t lo = p - win + 1 > 0 ? p - win + 1 : 0;
    int nr = lo < pos ? (int)(pos - lo) : 0;
    int g0 = lo > pos ? (int)(lo - pos) : 0, ng = j - g0 + 1;
    int mode = r->tag[16] != GP_T_NONE ? DI(16) : 0;
    if (mode & 2) {
        int64_t w0 = di(r, e, 18);
        lo = lo > w0 ? lo : w0;
        nr = lo < pos ? (int)(pos - lo) : 0;
        g0 = lo > pos ? (int)(lo - pos) : 0;
        ng = j - g0 + 1;
    }
    if (mode & 1) {
        nr = (int)(pos < win ? pos : win);
        lo = pos - nr;
        g0 = 0;
        ng = DI(14);
    }
    const void *ckv = DP(const void, 4);
    int form = ds_form(r, e, 17);
    const int *sel = DP(const int, 5) + (size_t)j * maxk;
    int nk = nr + ng + (ckv ? DP(const int, 6)[j] : 0);
    const void *ring = DP(const void, 1);
    const float *kvq = DP(const float, 13);
    float scale = df(r, e, 11);
    __shared__ __align__(16) float ks[DS_G2KC][512];
    __shared__ __align__(16) float ps[DS_G2KC][DS_GHB];          /* [key][head] */
    __shared__ float cmx[DS_G2KC / DS_GKC][DS_GHB], ccr[DS_G2KC / DS_GKC][DS_GHB];
    __shared__ float mrun[DS_GHB], lrun[DS_GHB];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, hg = warp % 4, kh = warp / 4;
    const float *q = DP(const float, 0) + ((size_t)j * H + h0 + 4 * hg) * hd;
    float4 qr[4][4];
    #pragma unroll
    for (int a = 0; a < 4; ++a) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            qr[a][i] = *(const float4 *)(q + (size_t)a * hd + 4 * lane + 128 * i);
        }
    }
    if (threadIdx.x < DS_GHB) {
        mrun[threadIdx.x] = -INFINITY;
        lrun[threadIdx.x] = 0.f;
    }
    float acc[DS_GHB][2];
    #pragma unroll
    for (int h = 0; h < DS_GHB; ++h) {
        acc[h][0] = acc[h][1] = 0.f;
    }
    const int nq = hd / 4;
    for (int m0 = 0; m0 < nk; m0 += DS_G2KC) {
        int nkr = nk - m0 < DS_G2KC ? nk - m0 : DS_G2KC, nch = (nkr + DS_GKC - 1) / DS_GKC;
        __syncthreads();
        for (int idx = threadIdx.x; idx < nkr * nq; idx += blockDim.x) {
            int kk = idx / nq, c4 = 4 * (idx % nq);
            ds_krow kr = ds_akey(m0 + kk, nr, ng, lo, g0, win, ring, kvq, ckv, sel, form);
            *(float4 *)&ks[kk][c4] = ds_rld4(kr.p, kr.row, c4, hd, kr.form);
        }
        __syncthreads();
        /* the scores: v[8 a + kk] the lane's sum of head 4 hg + a, key 8 kh + kk */
        float v[32];
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            float d[4] = {0.f, 0.f, 0.f, 0.f};
            if (8 * kh + kk < nkr) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    float4 k4 = *(const float4 *)&ks[8 * kh + kk][4 * lane + 128 * i];
                    #pragma unroll
                    for (int a = 0; a < 4; ++a) {
                        float4 q4 = qr[a][i];
                        d[a] += q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
                    }
                }
            }
            #pragma unroll
            for (int a = 0; a < 4; ++a) {
                v[8 * a + kk] = d[a];
            }
        }
        /* the butterfly: at xor o the lanes with bit o keep the upper half of their sums, the others
         * the lower, each adding the partner's of the half it keeps (lane L ends with sum L) */
        DS_BFLY(v, 16);
        DS_BFLY(v, 8);
        DS_BFLY(v, 4);
        DS_BFLY(v, 2);
        DS_BFLY(v, 1);
        {
            int a = lane / 8, kk = lane % 8;
            if (8 * kh + kk < nkr) {
                ps[8 * kh + kk][4 * hg + a] = v[0] * scale;
            }
        }
        __syncthreads();
        if (threadIdx.x < DS_GHB) {
            int h = threadIdx.x;
            float m = mrun[h];
            for (int c = 0; c < nch; ++c) {
                int k1 = DS_GKC * c + DS_GKC < nkr ? DS_GKC * c + DS_GKC : nkr;
                float mx = m;
                for (int k = DS_GKC * c; k < k1; ++k) {
                    mx = fmaxf(mx, ps[k][h]);
                }
                ccr[c][h] = m == -INFINITY ? 0.f : expf(m - mx);
                cmx[c][h] = mx;
                m = mx;
            }
            mrun[h] = m;
        }
        __syncthreads();
        {
            int kk = threadIdx.x / DS_GHB, h = threadIdx.x % DS_GHB;
            if (kk < nkr) {
                ps[kk][h] = expf(ps[kk][h] - cmx[kk / DS_GKC][h]);
            }
        }
        __syncthreads();
        if (threadIdx.x < DS_GHB) {
            int h = threadIdx.x;
            float l = lrun[h];
            for (int c = 0; c < nch; ++c) {
                int k1 = DS_GKC * c + DS_GKC < nkr ? DS_GKC * c + DS_GKC : nkr;
                l = l * ccr[c][h];
                for (int k = DS_GKC * c; k < k1; ++k) {
                    l += ps[k][h];
                }
            }
            lrun[h] = l;
        }
        for (int c = 0; c < nch; ++c) {
            int k1 = DS_GKC * c + DS_GKC < nkr ? DS_GKC * c + DS_GKC : nkr;
            #pragma unroll
            for (int h = 0; h < DS_GHB; ++h) {
                float cr = ccr[c][h];
                acc[h][0] = acc[h][0] * cr;
                acc[h][1] = acc[h][1] * cr;
            }
            for (int k = DS_GKC * c; k < k1; ++k) {
                float2 kv = *(const float2 *)&ks[k][2 * threadIdx.x];
                #pragma unroll
                for (int h4 = 0; h4 < DS_GHB / 4; ++h4) {
                    float4 p4 = *(const float4 *)&ps[k][4 * h4];
                    float pp[4] = {p4.x, p4.y, p4.z, p4.w};
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        acc[4 * h4 + b][0] += pp[b] * kv.x;
                        acc[4 * h4 + b][1] += pp[b] * kv.y;
                    }
                }
            }
        }
    }
    __syncthreads();
    float *out = DP(float, 8) + ((size_t)j * H + h0) * hd;
    #pragma unroll
    for (int h = 0; h < DS_GHB; ++h) {
        float den = lrun[h] + expf(DP(const float, 7)[h0 + h] - mrun[h]);
        *(float2 *)&out[(size_t)h * hd + 2 * threadIdx.x] = make_float2(acc[h][0] / den, acc[h][1] / den);
    }
}

/* GP_DS_ATTN for a group on the tensor cores (hd 512, 64 heads a block; RECLANK_DS_ATTN_TC=1):
 * k_glm_attn_tc's form with k_ds_attn_g's keys and sink. A block for each query, 8 warps; its 64
 * heads (times scale) in shared memory as float16, its keys DS_TKC at a time decoded from any form
 * of the caches to float16; a pass with a running softmax: warp w the scores of heads 16 (w % 4) ..
 * + 15 and keys 8 (w / 4) .. + 7 (mma.sync m16n8k16, float32 sums), the max and the sum of each
 * head (one thread a row), P float16, then warp w the output of its 16 heads and dims 256 (w / 4)
 * .. + 255 (V by ldmatrix.trans); the sink joins the sum at the end. Not the bits of k_ds_attn_g
 * (q, the keys and P rounded to float16). */
#define DS_TKC 16
#define DS_TLD 520
#define DS_TSMEM ((size_t)(64 + DS_TKC) * DS_TLD * 2 + 64 * (DS_TKC + 1) * 4 + 64 * (DS_TKC + 8) * 2)
__global__ void __launch_bounds__(256, 1) k_ds_attn_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) unsigned char dtsm[];
    __half (*Qs)[DS_TLD] = (__half (*)[DS_TLD])dtsm;
    __half (*Ks)[DS_TLD] = (__half (*)[DS_TLD])(dtsm + 64 * DS_TLD * 2);
    float (*Ss)[DS_TKC + 1] = (float (*)[DS_TKC + 1])(dtsm + (64 + DS_TKC) * DS_TLD * 2);
    __half (*Ps)[DS_TKC + 8] = (__half (*)[DS_TKC + 8])(dtsm + (64 + DS_TKC) * DS_TLD * 2 + 64 * (DS_TKC + 1) * 4);
    __shared__ float rm[64], rl[64], rs[64];
    int win = DI(3), maxk = DI(12), hb = blockIdx.x * 64, j = blockIdx.y, H = DI(9);
    const int hd = 512;
    int64_t pos = di(r, e, 2), p = pos + j;
    int64_t lo = p - win + 1 > 0 ? p - win + 1 : 0;
    int nr = lo < pos ? (int)(pos - lo) : 0;
    int g0 = lo > pos ? (int)(lo - pos) : 0, ng = j - g0 + 1;
    int mode = r->tag[16] != GP_T_NONE ? DI(16) : 0;
    if (mode & 2) {
        int64_t w0 = di(r, e, 18);
        lo = lo > w0 ? lo : w0;
        nr = lo < pos ? (int)(pos - lo) : 0;
        g0 = lo > pos ? (int)(lo - pos) : 0;
        ng = j - g0 + 1;
    }
    if (mode & 1) {
        nr = (int)(pos < win ? pos : win);
        lo = pos - nr;
        g0 = 0;
        ng = DI(14);
    }
    const void *ckv = DP(const void, 4);
    int form = ds_form(r, e, 17);
    const int *sel = DP(const int, 5) + (size_t)j * maxk;
    int nk = nr + ng + (ckv ? DP(const int, 6)[j] : 0);
    const void *ring = DP(const void, 1);
    const float *kvq = DP(const float, 13);
    const float scale = df(r, e, 11);
    const float *q = DP(const float, 0) + ((size_t)j * H + hb) * hd;
    for (int i = threadIdx.x; i < 64 * 128; i += 256) {
        float4 v = *(const float4 *)(q + 4 * i);
        __half *d = &Qs[i / 128][4 * (i % 128)];
        d[0] = __float2half_rn(v.x * scale);
        d[1] = __float2half_rn(v.y * scale);
        d[2] = __float2half_rn(v.z * scale);
        d[3] = __float2half_rn(v.w * scale);
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
    for (int m0 = 0; m0 < nk; m0 += DS_TKC) {
        __syncthreads();
        for (int i = threadIdx.x; i < DS_TKC * 128; i += 256) {
            int kk = i / 128, c4 = (i % 128) * 4;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (m0 + kk < nk) {
                ds_krow kr = ds_akey(m0 + kk, nr, ng, lo, g0, win, ring, kvq, ckv, sel, form);
                v = ds_rld4(kr.p, kr.row, c4, hd, kr.form);
            }
            __half2 *d = (__half2 *)&Ks[kk][c4];
            d[0] = __floats2half2_rn(v.x, v.y);
            d[1] = __floats2half2_rn(v.z, v.w);
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
            for (int kk = 0; kk < DS_TKC; ++kk) {
                if (m0 + kk < nk) {
                    mn = fmaxf(mn, Ss[h][kk]);
                }
            }
            float sc = expf(mo - mn), l = rl[h] * sc;
            for (int kk = 0; kk < DS_TKC; ++kk) {
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
    const float *sink = DP(const float, 7) + hb;
    const float i0 = 1.f / (rl[h0 + g] + expf(sink[h0 + g] - rm[h0 + g]));
    const float i1 = 1.f / (rl[h0 + g + 8] + expf(sink[h0 + g + 8] - rm[h0 + g + 8]));
    float *out = DP(float, 8) + ((size_t)j * H + hb) * hd;
    #pragma unroll
    for (int n = 0; n < 32; ++n) {
        int d = d0 + 8 * n + 2 * c;
        *(float2 *)&out[(size_t)(h0 + g) * 512 + d] = make_float2(o[n][0] * i0, o[n][1] * i0);
        *(float2 *)&out[(size_t)(h0 + g + 8) * 512 + d] = make_float2(o[n][2] * i1, o[n][3] * i1);
    }
}
