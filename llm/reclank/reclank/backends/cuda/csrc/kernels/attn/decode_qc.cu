/* ---------- the decode attention of one query over the quantized cache, in
 * one pass (flash decoding) ----------
 * The keys of each key head are cut into parts (FD_PARTS for each head, of
 * fd_len keys); each part keeps its softmax running (the maximum m and the
 * sum l of each head) and writes its sums of the values, m and l; a join
 * kernel adds the parts. The keys and the values stream without the phases
 * of k_attn_part, and no scores go to memory. The kernels: k_attn_fdq
 * (Qwen3.5 / 3.6: head_dim 256, 8 query heads a key head), k_attn_fds and
 * k_attn_fdg (the sliding and the global layers of the Gemma 4 26B), and
 * k_attn_fdt (the int8 forms of the 26B). */
#define FD_KEYS 4
#define FD_BLOCKS 32                       /* blocks for each key head (32 was best of 32 to 256) */
#define FD_PARTS (FD_BLOCKS * 4)           /* 4 warps for each block */

__device__ __forceinline__ int fd_len(int n)
{
    int len = (n + FD_PARTS - 1) / FD_PARTS;
    len = len < 16 ? 16 : len;
    return (len + FD_KEYS - 1) / FD_KEYS * FD_KEYS;
}

/* a if mask, else b: by bits, since nvcc makes up ? v[i] : v[j] into
 * v[up ? i : j] (an index it does not know: v in local memory) */
__device__ __forceinline__ float fd_sel(unsigned mask, float a, float b)
{
    return __uint_as_float((__float_as_uint(a) & mask) | (__float_as_uint(b) & ~mask));
}

/* the reduce-scatter of 32 sums over the warp: lane x gets the sum of
 * value x. A step of it for NV values: lane bit NV / 2 picks the half. The
 * steps are templates: in a loop nvcc kept a counter and v in local memory. */
template <int NV>
__device__ __forceinline__ void fd_scatter_step(float *v, int lane)
{
    unsigned up = (lane & (NV / 2)) ? ~0u : 0u;
    #pragma unroll
    for (int x = 0; x < NV / 2; ++x) {
        float keep = fd_sel(up, v[x + NV / 2], v[x]);
        float send = fd_sel(up, v[x], v[x + NV / 2]);
        v[x] = keep + __shfl_xor_sync(RC_FULL_MASK, send, NV / 2);
    }
}

__device__ __forceinline__ void fd_scatter32(float *v, int lane)
{
    fd_scatter_step<32>(v, lane);
    fd_scatter_step<16>(v, lane);
    fd_scatter_step<8>(v, lane);
    fd_scatter_step<4>(v, lane);
    fd_scatter_step<2>(v, lane);
}

__device__ __forceinline__ void fd_i8x8(uint2 u, float *out)
{
    uint32_t w[2] = {u.x, u.y};
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        out[t] = rc_kv8_val((w[t / 4] >> (8 * (t % 4))) & 0xff);
    }
}

__device__ __forceinline__ void fd_i16x8(uint4 u, float *out)
{
    uint32_t w[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        out[2 * t] = (float)(int16_t)(w[t] & 0xffff);
        out[2 * t + 1] = (float)(int16_t)(w[t] >> 16);
    }
}

/* k_attn_fdt: decode attention of head_dim HD (256 or 512) and G query heads
 * for each key head (G divides 32; the Gemma 4 26B has HD 256 with G 2 and
 * HD 512 with G 8). A warp takes GH of the G heads (a divisor of G; the
 * blocks of the G / GH groups of a key head are adjacent in x, so they read
 * the same keys at about the same time): a step of a warp takes 32 / GH
 * keys, so the scores are 32 values (keys by heads) again. Lane l keeps
 * values 8 l to 8 l + 7 of each part of 256 values. The keys come first (the
 * scores and the running softmax), then the values, so fewer registers hold
 * loads at a time. The record gives the keys of the window, so no key needs a
 * mask. The int16 caches of the 26B take k_attn_fds and k_attn_fdg; this
 * kernel serves the int8 forms. */
template <int HD, int G, bool KQ8 = false, bool VQ8 = false, int GH = G>
__global__ void __launch_bounds__(128) k_attn_fdt(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int NP = HD / 256, K = 32 / GH, HG = G / GH;
    __shared__ __align__(16) float qs[GH * HD];
    __shared__ float pw[4][32 + GH];
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x / HG, hg = blockIdx.x % HG, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int n = a.n, len = fd_len(n);
    len = (len + K - 1) / K * K;
    int part_id = blockIdx.y * 4 + warp;
    int j0 = part_id * len, j1 = min(n, j0 + len);
    for (int i = threadIdx.x; i < GH * HD; i += blockDim.x) {
        qs[i] = a.q[((size_t)kv * G + hg * GH) * HD + i];
    }
    __syncthreads();
    if (j0 >= n) {
        return;
    }
    float acc[GH][NP][8];
    #pragma unroll
    for (int h = 0; h < GH; ++h) {
        #pragma unroll
        for (int s = 0; s < NP; ++s) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[h][s][u] = 0.f;
            }
        }
    }
    float m = -INFINITY, l = 0.f;          /* of head lane % GH */
    size_t rs = a.rstride, off = (size_t)kv * HD + 8 * lane;
    for (int jb = j0; jb < j1; jb += K) {
        float v[32];
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            int j = min(jb + kk, j1 - 1);
            #pragma unroll
            for (int h = 0; h < GH; ++h) {
                v[kk * GH + h] = 0.f;
            }
            #pragma unroll
            for (int s = 0; s < NP; ++s) {
                size_t o = (size_t)j * rs + off + 256 * s;
                float kf[8];
                if (KQ8) {
                    fd_i8x8(*(const uint2 *)(a.kq8 + o), kf);
                } else {
                    fd_i16x8(*(const uint4 *)(a.kq + o), kf);
                }
                float sk = a.ks[o / 32];
                #pragma unroll
                for (int h = 0; h < GH; ++h) {
                    const float4 q0 = *(const float4 *)(qs + h * HD + 256 * s + 8 * lane);
                    const float4 q1 = *(const float4 *)(qs + h * HD + 256 * s + 8 * lane + 4);
                    float d = q0.x * kf[0] + q0.y * kf[1] + q0.z * kf[2] + q0.w * kf[3] +
                              q1.x * kf[4] + q1.y * kf[5] + q1.z * kf[6] + q1.w * kf[7];
                    v[kk * GH + h] += d * sk;
                }
            }
        }
        /* the reduce-scatter: value kk * GH + h to lane kk * GH + h */
        fd_scatter32(v, lane);
        float sc = jb + lane / GH < j1 ? v[0] : -INFINITY;
        /* the maximum and the sum of each head over the K keys: the lanes
         * h, h + GH, h + 2 GH, ... */
        float mx = sc;
        #pragma unroll
        for (int o2 = GH; o2 < 32; o2 <<= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, o2));
        }
        float mn = fmaxf(m, mx);
        float alpha = m == -INFINITY ? 0.f : expf(m - mn);
        float p = sc == -INFINITY ? 0.f : expf(sc - mn);
        float ps = p;
        #pragma unroll
        for (int o2 = GH; o2 < 32; o2 <<= 1) {
            ps += __shfl_xor_sync(RC_FULL_MASK, ps, o2);
        }
        l = l * alpha + ps;
        m = mn;
        pw[warp][lane] = p;
        if (lane < GH) {
            pw[warp][32 + lane] = alpha;
        }
        __syncwarp();
        #pragma unroll
        for (int h = 0; h < GH; ++h) {
            float al = pw[warp][32 + h];
            #pragma unroll
            for (int s = 0; s < NP; ++s) {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc[h][s][u] *= al;
                }
            }
        }
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            int j = min(jb + kk, j1 - 1);
            #pragma unroll
            for (int s = 0; s < NP; ++s) {
                size_t o = (size_t)j * rs + off + 256 * s;
                float vf[8];
                if (VQ8) {
                    fd_i8x8(*(const uint2 *)(a.vq8 + o), vf);
                } else {
                    fd_i16x8(*(const uint4 *)(a.vq + o), vf);
                }
                float sv = a.vs[o / 32];
                if (!VQ8 && a.vq == a.kq && a.vs != a.ks && ko_turned(8 * lane + 256 * s)) {
                    /* keys only (ko_tab): the turned values from the partner */
                    int i = 8 * lane + 256 * s;
                    size_t po = o - i + ko_partner(i);
                    float y[8];
                    ko_i16x8(*(const uint4 *)(a.kq + po), a.vs[po / 32], y);
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        vf[u] *= sv;
                    }
                    ko_rot8(vf, y, j, i);
                    sv = 1.f;
                }
                #pragma unroll
                for (int h = 0; h < GH; ++h) {
                    float pp = pw[warp][kk * GH + h] * sv;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        acc[h][s][u] += pp * vf[u];
                    }
                }
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (int h = 0; h < GH; ++h) {
        float *o = part + ((size_t)(kv * G + hg * GH + h) * FD_PARTS + part_id) * (HD + 2);
        #pragma unroll
        for (int s = 0; s < NP; ++s) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                o[256 * s + 8 * lane + u] = acc[h][s][u];
            }
        }
        if (lane == h) {
            o[HD] = m;
            o[HD + 1] = l;
        }
    }
}

/* The join of the parts of k_attn_fdt, fds, fdg and fdq: block (h, x) joins
 * the values x * 128 to x * 128 + 127 of query head h (GH: the heads of a
 * warp of k_attn_fdt; 8 for the others, whose parts have fd_len keys). */
template <int HD, int G, int GH = G>
__global__ void k_attn_fdt_join(const gp_rec *r, const int64_t *e, const float *part)
{
    PDL_START();
    __shared__ float w[FD_PARTS];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x;
    int len = fd_len(a.n);
    len = (len + 32 / GH - 1) / (32 / GH) * (32 / GH);
    int nc = (a.n + len - 1) / len;
    const float *ph = part + (size_t)h * FD_PARTS * (HD + 2);
    float M = -INFINITY;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        M = fmaxf(M, ph[(size_t)c * (HD + 2) + HD]);
    }
    M = block_max(M);
    float wl = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float mc = ph[(size_t)c * (HD + 2) + HD];
        float wc = mc == -INFINITY ? 0.f : expf(mc - M);
        w[c] = wc;
        wl += wc * ph[(size_t)c * (HD + 2) + HD + 1];
    }
    float wsum = block_sum(wl);
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    __syncthreads();
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < HD) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) {
            acc += w[c] * ph[(size_t)c * (HD + 2) + i];
        }
        a.out[(size_t)h * HD + i] = acc * inv;
    }
}

/* k_attn_fdq: Qwen3.5 / 3.6 (head_dim 256, 8 query heads for each key
 * head, int16 keys and values) in the form of k_attn_fds: block (key head,
 * part) of 8 warps; warp w has heads 4 (w & 1) to that + 3 (q of these in
 * registers: lane l, values 8 l to 8 l + 7) and the keys 2 (w >> 1) + 8 i and
 * the one after of the part, read from global memory (the two warps of a set
 * of keys read the same rows at about the same time: L1), with a softmax of
 * its own; the 4 sets of keys join at the end. The scores of a step: 8 sums
 * (2 keys by 4 heads), 3 steps of the reduce-scatter and 2 sums (lane: key
 * (lane & 7) / 4, head lane & 3). KQ8, VQ8: int8 keys, values (GP_ATTN_Q8,
 * GP_ATTN_V8; Qwen's default cache is int8, which went to k_attn_part: 1.6
 * ms a layer at 8192 tokens on the GTX 1070). The kernel before (int16
 * only) read q of the 8 heads from shared memory for each key and had 213
 * registers. */
template <bool KQ8, bool VQ8>
__global__ void __launch_bounds__(256, 2) k_attn_fdq(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int HD = 256, G = 8;
    __shared__ float red[6][32][33];
    __shared__ float wm[4][G], wl[4][G];
    const float *aq, *aks, *avs;
    const int16_t *akq, *avq;
    const int8_t *akq8, *avq8;
    int n;
    size_t rs;
    {
        /* the operands of GP_ATTN_QC */
        aq = DP(const float, 0);
        akq = DP(const int16_t, 1);
        aks = DP(const float, 2);
        avq = DP(const int16_t, 3);
        akq8 = DP(const int8_t, 1);
        avq8 = DP(const int8_t, 3);
        avs = DP(const float, 4);
        n = DI(10);
        rs = (size_t)DI(8) * DI(9);
    }
    int kv = blockIdx.x, w = threadIdx.x / 32, lane = threadIdx.x % 32;
    int hg = w & 1, kset = w >> 1;
    int len = fd_len(n);
    int part_id = blockIdx.y;
    int j0 = part_id * len, j1 = min(n, j0 + len);
    if (j0 >= n) {
        return;
    }
    size_t off = (size_t)kv * HD + 8 * lane;
    float q[4][8];
    #pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        const float *qp = aq + ((size_t)kv * G + 4 * hg + hh) * HD + 8 * lane;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            q[hh][u] = qp[u];
        }
    }
    float acc[4][8];
    #pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[hh][u] = 0.f;
        }
    }
    float m = -INFINITY, l = 0.f;                    /* of head 4 hg + (lane & 3) */
    for (int jb = j0 + 2 * kset; jb < j1; jb += 8) {
        float v[8], vf[2][8], sv[2];
        #pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            size_t o = (size_t)min(jb + kk, j1 - 1) * rs + off;
            float kf[8];
            if (KQ8) {
                fd_i8x8(*(const uint2 *)(akq8 + o), kf);
            } else {
                fd_i16x8(*(const uint4 *)(akq + o), kf);
            }
            if (VQ8) {
                fd_i8x8(*(const uint2 *)(avq8 + o), vf[kk]);
            } else {
                fd_i16x8(*(const uint4 *)(avq + o), vf[kk]);
            }
            float sk = aks[o / 32];
            sv[kk] = avs[o / 32];
            #pragma unroll
            for (int hh = 0; hh < 4; ++hh) {
                float d = 0.f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    d += q[hh][u] * kf[u];
                }
                v[kk * 4 + hh] = d * sk;
            }
        }
        fd_scatter_step<8>(v, lane);
        fd_scatter_step<4>(v, lane);
        fd_scatter_step<2>(v, lane);
        float sc = v[0] + __shfl_xor_sync(RC_FULL_MASK, v[0], 8);
        sc += __shfl_xor_sync(RC_FULL_MASK, sc, 16);
        sc = jb + (lane & 7) / 4 < j1 ? sc : -INFINITY;
        float mx = fmaxf(sc, __shfl_xor_sync(RC_FULL_MASK, sc, 4));
        float mn = fmaxf(m, mx);
        float alpha = m == -INFINITY ? 0.f : expf(m - mn);
        float p = sc == -INFINITY ? 0.f : expf(sc - mn);
        float ps = p + __shfl_xor_sync(RC_FULL_MASK, p, 4);
        l = l * alpha + ps;
        m = mn;
        #pragma unroll
        for (int hh = 0; hh < 4; ++hh) {
            float al = __shfl_sync(RC_FULL_MASK, alpha, hh);
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[hh][u] *= al;
            }
            #pragma unroll
            for (int kk = 0; kk < 2; ++kk) {
                float pp = __shfl_sync(RC_FULL_MASK, p, kk * 4 + hh) * sv[kk];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc[hh][u] += pp * vf[kk][u];
                }
            }
        }
    }
    /* the 4 sets of keys join: sets 1-3 through shared memory */
    if (kset > 0) {
        #pragma unroll
        for (int hh = 0; hh < 4; ++hh) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                red[w - 2][hh * 8 + u][lane] = acc[hh][u];
            }
        }
    }
    if (lane < 4) {
        wm[kset][4 * hg + lane] = m;
        wl[kset][4 * hg + lane] = l;
    }
    __syncthreads();
    if (kset == 0) {
        #pragma unroll
        for (int hh = 0; hh < 4; ++hh) {
            int h = 4 * hg + hh;
            float M = fmaxf(fmaxf(wm[0][h], wm[1][h]), fmaxf(wm[2][h], wm[3][h]));
            float c[4], L = 0.f;
            #pragma unroll
            for (int ks = 0; ks < 4; ++ks) {
                c[ks] = wm[ks][h] == -INFINITY ? 0.f : expf(wm[ks][h] - M);
                L += c[ks] * wl[ks][h];
            }
            float *o = part + ((size_t)(kv * G + h) * FD_PARTS + part_id) * (HD + 2);
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                float x = c[0] * acc[hh][u];
                #pragma unroll
                for (int ks = 1; ks < 4; ++ks) {
                    x += c[ks] * red[2 * ks + hg - 2][hh * 8 + u][lane];
                }
                o[8 * lane + u] = x;
            }
            if (lane == 0) {
                o[HD] = M;
                o[HD + 1] = L;
            }
        }
    }
}

/* k_attn_fdt for a sliding layer of the 26B (head_dim 256, 2 query heads for
 * each key head, int16 keys and values of their own): block (key head, part)
 * of 4 warps; warp w takes keys 4 w to 4 w + 3 of each 16 of the part, with
 * q of both heads in registers (lane l: values 8 l to 8 l + 7) and a softmax
 * of its own; the 4 warps join at the end. The 8 scores of a step: 3 steps
 * of the reduce-scatter and 2 sums (lane: key (lane & 7) / 2, head lane &
 * 1). No shared memory in the loop and few registers (78), so many blocks an
 * SM: 53 us at 1024 keys on the GTX 1070, where k_attn_fdt (16 keys a step,
 * 231 registers, 2 blocks an SM) took 104 us. */
__global__ void __launch_bounds__(128) k_attn_fds(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int HD = 256, G = 2;
    __shared__ float red[4][G][HD];
    __shared__ float wm[4][G], wl[4][G];
    const float *aq, *aks, *avs;
    const int16_t *akq, *avq;
    int n;
    size_t rs;
    {
        /* the operands of GP_ATTN_QC */
        aq = DP(const float, 0);
        akq = DP(const int16_t, 1);
        aks = DP(const float, 2);
        avq = DP(const int16_t, 3);
        avs = DP(const float, 4);
        n = DI(10);
        rs = (size_t)DI(8) * DI(9);
    }
    if (avq == akq && avs != aks) {
        __trap();                                    /* keys only: a global layer (k_attn_fdg) */
    }
    int kv = blockIdx.x, w = threadIdx.x / 32, lane = threadIdx.x % 32;
    int len = fd_len(n);
    int part_id = blockIdx.y;
    int j0 = part_id * len, j1 = min(n, j0 + len);
    if (j0 >= n) {
        return;
    }
    size_t off = (size_t)kv * HD + 8 * lane;
    float q[G][8];
    #pragma unroll
    for (int h = 0; h < G; ++h) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            q[h][u] = aq[((size_t)kv * G + h) * HD + 8 * lane + u];
        }
    }
    float acc[G][8];
    #pragma unroll
    for (int h = 0; h < G; ++h) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[h][u] = 0.f;
        }
    }
    float m = -INFINITY, l = 0.f;                    /* of head lane & 1 */
    #pragma unroll 2
    for (int jb = j0 + 4 * w; jb < j1; jb += 16) {
        float v[8], vf[4][8], sv[4];
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            size_t o = (size_t)min(jb + kk, j1 - 1) * rs + off;
            float kf[8];
            fd_i16x8(*(const uint4 *)(akq + o), kf);
            fd_i16x8(*(const uint4 *)(avq + o), vf[kk]);
            float sk = aks[o / 32];
            sv[kk] = avs[o / 32];
            #pragma unroll
            for (int h = 0; h < G; ++h) {
                float d = 0.f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    d += q[h][u] * kf[u];
                }
                v[kk * G + h] = d * sk;
            }
        }
        fd_scatter_step<8>(v, lane);
        fd_scatter_step<4>(v, lane);
        fd_scatter_step<2>(v, lane);
        float sc = v[0] + __shfl_xor_sync(RC_FULL_MASK, v[0], 8);
        sc += __shfl_xor_sync(RC_FULL_MASK, sc, 16);
        sc = jb + (lane & 7) / 2 < j1 ? sc : -INFINITY;
        float mx = fmaxf(sc, __shfl_xor_sync(RC_FULL_MASK, sc, 2));
        mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, 4));
        float mn = fmaxf(m, mx);
        float alpha = m == -INFINITY ? 0.f : expf(m - mn);
        float p = sc == -INFINITY ? 0.f : expf(sc - mn);
        float ps = p + __shfl_xor_sync(RC_FULL_MASK, p, 2);
        ps += __shfl_xor_sync(RC_FULL_MASK, ps, 4);
        l = l * alpha + ps;
        m = mn;
        #pragma unroll
        for (int h = 0; h < G; ++h) {
            float al = __shfl_sync(RC_FULL_MASK, alpha, h);
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[h][u] *= al;
            }
            #pragma unroll
            for (int kk = 0; kk < 4; ++kk) {
                float pp = __shfl_sync(RC_FULL_MASK, p, kk * G + h) * sv[kk];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc[h][u] += pp * vf[kk][u];
                }
            }
        }
    }
    /* the 4 warps join */
    #pragma unroll
    for (int h = 0; h < G; ++h) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            red[w][h][8 * lane + u] = acc[h][u];
        }
    }
    if (lane < G) {
        wm[w][lane] = m;
        wl[w][lane] = l;
    }
    __syncthreads();
    #pragma unroll
    for (int i = 0; i < G * HD / 128; ++i) {
        int x = threadIdx.x + 128 * i, h = x / HD, d = x % HD;
        float M = fmaxf(fmaxf(wm[0][h], wm[1][h]), fmaxf(wm[2][h], wm[3][h]));
        float o = 0.f, L = 0.f;
        #pragma unroll
        for (int ww = 0; ww < 4; ++ww) {
            float c = wm[ww][h] == -INFINITY ? 0.f : expf(wm[ww][h] - M);
            o += c * red[ww][h][d];
            L += c * wl[ww][h];
        }
        float *po = part + ((size_t)(kv * G + h) * FD_PARTS + part_id) * (HD + 2);
        po[d] = o;
        if (d == 0) {
            po[HD] = M;
            po[HD + 1] = L;
        }
    }
}

/* k_attn_fdt for a global layer of the 26B (head_dim 512, 8 query heads for
 * each key head, int16 keys): block (key head, part) of 8 warps takes the
 * keys of its part in tiles of 16, which the 256 threads copy to shared
 * memory (with keys only, ko_tab with 64 pairs, the turned values too, as
 * floats). Warp w has half w & 1 of the values (lane l: 8 l to 8 l + 7 of
 * it), heads 4 ((w >> 1) & 1) to that + 3 (q of these in registers: 32), and
 * keys 8 (w >> 2) to that + 7 of the tile: each int16 it reads from shared
 * memory goes into 4 heads. The scores (32 sums, a reduce-scatter; the two
 * halves added in shared memory), then warp 0 keeps the running softmax of
 * the 8 heads, then each warp adds its values; the warps of the two sets of
 * keys join at the end. Values of their own (not keys only) come from global
 * memory. On the GTX 1070 at 8192 tokens: 0.33 ms a layer; k_attn_fdt took
 * 2 ms (255 registers and v in local memory, see fd_scatter32), and a warp
 * for each head that read the whole tile 0.65 ms (the reads of shared
 * memory). */
#define FDG_T 16
__global__ void __launch_bounds__(256, 2) k_attn_fdg(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int HD = 512, G = 8;
    __shared__ __align__(16) int16_t kt[FDG_T][HD];   /* the keys */
    __shared__ __align__(16) float vt[FDG_T][128];    /* the turned values: 0-63, 256-319 */
    __shared__ float kst[FDG_T][HD / 32], vst[FDG_T][HD / 32];
    __shared__ float ps[2][FDG_T][G + 1];             /* the scores of each half; then the weights */
    __shared__ float al[G], ml[2][G];
    const float *aq, *aks, *avs;
    const int16_t *akq, *avq;
    int n;
    size_t rs;
    {
        /* the operands of GP_ATTN_QC (attn_get: its attn_d stayed in local
         * memory, 104 bytes) */
        aq = DP(const float, 0);
        akq = DP(const int16_t, 1);
        aks = DP(const float, 2);
        avq = DP(const int16_t, 3);
        avs = DP(const float, 4);
        n = DI(10);
        rs = (size_t)DI(8) * DI(9);
    }
    int kv = blockIdx.x, w = threadIdx.x / 32, lane = threadIdx.x % 32, t = threadIdx.x;
    int half = w & 1, hg = (w >> 1) & 1, ksub = w >> 2;
    int len = fd_len(n);
    int part_id = blockIdx.y;
    int j0 = part_id * len, j1 = min(n, j0 + len);
    if (j0 >= n) {
        return;
    }
    bool ko = avq == akq && avs != aks && g_ko.rd == 64 && g_ko.half == 256;
    const float2 *khi = g_ko.hi, *klo = g_ko.lo;
    size_t hb = (size_t)kv * HD;
    int d0 = 256 * half + 8 * lane;                  /* the values of the lane */
    float q[4][8];
    #pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        const float *qp = aq + (hb * G + (size_t)(4 * hg + hh) * HD) + d0;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            q[hh][u] = qp[u];
        }
    }
    float acc[4][8];
    #pragma unroll
    for (int hh = 0; hh < 4; ++hh) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[hh][u] = 0.f;
        }
    }
    float m = -INFINITY, l = 0.f;                    /* warp 0, lane h < 8 */
    bool turned = ko && lane < 8;                    /* the values 0-63 or 256-319 */
    for (int jt = j0; jt < j1; jt += FDG_T) {
        float2 c4[4];
        if (ko) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int x = t + 256 * i, j = min(jt + x / 64, j1 - 1), f = x % 64;
                float2 p = khi[(j >> 8) * 64 + f], qq = klo[(j & 255) * 64 + f];
                c4[i] = make_float2(p.x * qq.x - p.y * qq.y, p.x * qq.y + p.y * qq.x);
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int x = t + 256 * i, k = x / 64, c = x % 64;
            int j = min(jt + k, j1 - 1);
            *(uint4 *)&kt[k][8 * c] = *(const uint4 *)(akq + (size_t)j * rs + hb + 8 * c);
        }
        {
            int j = min(jt + t / 16, j1 - 1);
            size_t o = ((size_t)j * rs + hb) / 32 + t % 16;
            kst[t / 16][t % 16] = aks[o];
            vst[t / 16][t % 16] = avs[o];
        }
        __syncthreads();
        if (ko) {
            /* the turned values: pair f of key k (4 a thread) */
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int x = t + 256 * i, k = x / 64, f = x % 64;
                float xv = (float)kt[k][f] * vst[k][f / 32], yv = (float)kt[k][256 + f] * vst[k][8 + f / 32];
                vt[k][f] = xv * c4[i].x + yv * c4[i].y;
                vt[k][64 + f] = yv * c4[i].x - xv * c4[i].y;
            }
        }
        /* the scores of keys 8 ksub to 8 ksub + 7 for heads 4 hg to 4 hg + 3 (this half) */
        float v[32];
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int k = 8 * ksub + kk;
            float x[8];
            fd_i16x8(*(const uint4 *)&kt[k][d0], x);
            float sk = kst[k][d0 / 32];
            #pragma unroll
            for (int hh = 0; hh < 4; ++hh) {
                float d = 0.f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    d += q[hh][u] * x[u];
                }
                v[kk * 4 + hh] = d * sk;
            }
        }
        fd_scatter32(v, lane);
        ps[half][8 * ksub + lane / 4][4 * hg + lane % 4] = v[0];
        __syncthreads();
        /* the running softmax: warp 0, lane h + 8 c for keys 4 c to 4 c + 3 of head h */
        if (w == 0) {
            int h = lane % 8, c = lane / 8;
            float sc[4], mx = -INFINITY;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int k = 4 * c + i;
                sc[i] = jt + k < j1 ? ps[0][k][h] + ps[1][k][h] : -INFINITY;
                mx = fmaxf(mx, sc[i]);
            }
            mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, 8));
            mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, 16));
            float mn = fmaxf(m, mx), sum = 0.f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float p = sc[i] == -INFINITY ? 0.f : expf(sc[i] - mn);
                ps[0][4 * c + i][h] = p;
                sum += p;
            }
            sum += __shfl_xor_sync(RC_FULL_MASK, sum, 8);
            sum += __shfl_xor_sync(RC_FULL_MASK, sum, 16);
            float alpha = m == -INFINITY ? 0.f : expf(m - mn);
            l = l * alpha + sum;
            m = mn;
            if (lane < 8) {
                al[h] = alpha;
            }
        }
        __syncthreads();
        #pragma unroll
        for (int hh = 0; hh < 4; ++hh) {
            float aa = al[4 * hg + hh];
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[hh][u] *= aa;
            }
        }
        /* the values of keys 8 ksub to 8 ksub + 7 */
        #pragma unroll 4
        for (int kk = 0; kk < 8; ++kk) {
            int k = 8 * ksub + kk;
            float x[8];
            if (turned) {
                const float4 *s4 = (const float4 *)&vt[k][64 * half + 8 * lane];
                float4 a0 = s4[0], a1 = s4[1];
                x[0] = a0.x; x[1] = a0.y; x[2] = a0.z; x[3] = a0.w;
                x[4] = a1.x; x[5] = a1.y; x[6] = a1.z; x[7] = a1.w;
            } else if (ko) {
                fd_i16x8(*(const uint4 *)&kt[k][d0], x);
                float sv = vst[k][d0 / 32];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    x[u] *= sv;
                }
            } else {
                size_t o = (size_t)min(jt + k, j1 - 1) * rs + hb + d0;
                fd_i16x8(*(const uint4 *)(avq + o), x);
                float sv = avs[o / 32];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    x[u] *= sv;
                }
            }
            #pragma unroll
            for (int hh = 0; hh < 4; ++hh) {
                float p = ps[0][k][4 * hg + hh];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc[hh][u] += p * x[u];
                }
            }
        }
        __syncthreads();
    }
    /* the two sets of keys join: the warps of keys 8-15 through shared memory */
    float *red = (float *)&kt[0][0];                 /* 4 warps by 32 lanes by 32 */
    if (ksub == 1) {
        #pragma unroll
        for (int hh = 0; hh < 4; ++hh) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                red[((w - 4) * 32 + hh * 8 + u) * 32 + lane] = acc[hh][u];
            }
        }
    }
    if (w == 0 && lane < 8) {
        ml[0][lane] = m;
        ml[1][lane] = l;
    }
    __syncthreads();
    if (ksub == 0) {
        #pragma unroll
        for (int hh = 0; hh < 4; ++hh) {
            int h = 4 * hg + hh;
            float *o = part + ((size_t)(kv * G + h) * FD_PARTS + part_id) * (HD + 2);
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                o[d0 + u] = acc[hh][u] + red[(w * 32 + hh * 8 + u) * 32 + lane];
            }
            if (half == 0 && lane == 0) {
                o[HD] = ml[0][h];
                o[HD + 1] = ml[1][h];
            }
        }
    }
}

/* GP_ATTN_QSA of a large group (t > 1): the operands of the CPU record
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim (256), t,
 *     pos, sel, cnt, maxsel
 *
 * Block (j, x): query j, key head x / hg, and query heads (x % hg) * R to
 * + R - 1 of that key head (R = q_heads / kv_heads / hg, at most 8). The 4
 * warps take every 4th key of the query (the rows of sel, or all the
 * positions to pos + j) with a softmax that runs: lane l keeps values 8 l
 * to 8 l + 7 of each head. Then the warps join in shared memory. */
#define AQ_W 4
template <int R>
__global__ void __launch_bounds__(128) k_attn_qsa_mt(const gp_rec *r, const int64_t *e, int hg)
{
    PDL_START();
    __shared__ __align__(16) float qs[R * 256];
    __shared__ float wm[AQ_W][R], wl[AQ_W][R];
    __shared__ float wacc[AQ_W][R][256];
    int nq = DI(7), nk = DI(8), maxsel = DI(14);
    int64_t pos = di(r, e, 11);
    int j = blockIdx.x, kv = blockIdx.y / hg, qb = kv * (nq / nk) + (blockIdx.y % hg) * R;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const float *q = DP(const float, 0) + ((size_t)j * nq + qb) * 256;
    for (int i = threadIdx.x; i < R * 256; i += blockDim.x) {
        qs[i] = q[i];
    }
    __syncthreads();
    int c = DP(const int32_t, 13)[j];
    int n = c < 0 ? (int)(pos + j + 1) : c;
    const int32_t *rows = c < 0 ? NULL : DP(const int32_t, 12) + (size_t)j * maxsel;
    const int16_t *kq = DP(const int16_t, 1), *vq = DP(const int16_t, 3);
    const int8_t *kq8 = DP(const int8_t, 1), *vq8 = DP(const int8_t, 3);
    const float *ks = DP(const float, 2), *vs = DP(const float, 4);
    int form = DI(15);       /* 0 int16, 1 int8, 2 int16 keys and int8 values, 3 float32,
                              * 4 TQ6 */
    size_t rs = (size_t)nk * 256, off = (size_t)kv * 256 + 8 * lane;
    if (form == 3) {
        /* float32 rows: head stride operand 16 (kv_rs) */
        rs = kv_rs((size_t)di(r, e, 16), nk, 256);
        off = (size_t)kv * (size_t)di(r, e, 16) + 8 * lane;
    }
    float m[R], l[R], acc[R][8];
    #pragma unroll
    for (int h = 0; h < R; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[h][u] = 0.f;
        }
    }
    for (int x = warp; x < n; x += AQ_W) {
        size_t o = (size_t)(rows ? rows[x] : x) * rs + off;
        float kf[8], vf[8];
        float ksc = 1.f, vsc = 1.f;
        if (form == 3) {
            const float *kp = DP(const float, 1) + o, *vp = DP(const float, 3) + o;
            float4 a0 = *(const float4 *)kp, a1 = *(const float4 *)(kp + 4);
            float4 b0 = *(const float4 *)vp, b1 = *(const float4 *)(vp + 4);
            kf[0] = a0.x; kf[1] = a0.y; kf[2] = a0.z; kf[3] = a0.w;
            kf[4] = a1.x; kf[5] = a1.y; kf[6] = a1.z; kf[7] = a1.w;
            vf[0] = b0.x; vf[1] = b0.y; vf[2] = b0.z; vf[3] = b0.w;
            vf[4] = b1.x; vf[5] = b1.y; vf[6] = b1.z; vf[7] = b1.w;
        } else if (form == 4) {
            tq6x8(DP(const uint8_t, 1), o, kf);
            tq6x8(DP(const uint8_t, 3), o, vf);
            ksc = ks[o / 32];
            vsc = vs[o / 32];
        } else {
            if (form == 1) {
                fd_i8x8(*(const uint2 *)(kq8 + o), kf);
            } else {
                fd_i16x8(*(const uint4 *)(kq + o), kf);
            }
            if (form >= 1) {
                fd_i8x8(*(const uint2 *)(vq8 + o), vf);
            } else {
                fd_i16x8(*(const uint4 *)(vq + o), vf);
            }
            ksc = ks[o / 32];
            vsc = vs[o / 32];
        }
        #pragma unroll
        for (int h = 0; h < R; ++h) {
            const float4 q0 = *(const float4 *)(qs + h * 256 + 8 * lane);
            const float4 q1 = *(const float4 *)(qs + h * 256 + 8 * lane + 4);
            /* the scale of the key is that of the 32 values of the lane: before the sum */
            float d = (q0.x * kf[0] + q0.y * kf[1] + q0.z * kf[2] + q0.w * kf[3] +
                       q1.x * kf[4] + q1.y * kf[5] + q1.z * kf[6] + q1.w * kf[7]) * ksc;
            for (int o2 = 16; o2 > 0; o2 >>= 1) {
                d += __shfl_xor_sync(RC_FULL_MASK, d, o2);
            }
            float sc = d;
            float mn = fmaxf(m[h], sc);
            float alpha = expf(m[h] - mn), p = expf(sc - mn);
            l[h] = l[h] * alpha + p;
            m[h] = mn;
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[h][u] = acc[h][u] * alpha + p * vsc * vf[u];
            }
        }
    }
    #pragma unroll
    for (int h = 0; h < R; ++h) {
        if (lane == 0) {
            wm[warp][h] = m[h];
            wl[warp][h] = l[h];
        }
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            wacc[warp][h][8 * lane + u] = acc[h][u];
        }
    }
    __syncthreads();
    float *out = DP(float, 6) + ((size_t)j * nq + qb) * 256;
    for (int i = threadIdx.x; i < R * 256; i += blockDim.x) {
        int h = i / 256, d = i % 256;
        float M = -INFINITY;
        #pragma unroll
        for (int w = 0; w < AQ_W; ++w) {
            M = fmaxf(M, wm[w][h]);
        }
        float num = 0.f, den = 0.f;
        #pragma unroll
        for (int w = 0; w < AQ_W; ++w) {
            float f = wm[w][h] == -INFINITY ? 0.f : expf(wm[w][h] - M);
            num += f * wacc[w][h][d];
            den += f * wl[w][h];
        }
        out[i] = den > 0.f ? num / den : 0.f;
    }
}
