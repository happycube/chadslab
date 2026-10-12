/* ---------- the decode attention of one query over the int16 cache, in one
 * pass (flash decoding) ----------
 * For GP_ATTN_QC with head_dim 256 and 8 query heads for each key head
 * (Qwen3.5). Each warp takes FD_KEYS keys at a time of its own range of
 * keys: lane l keeps values 8 l to 8 l + 7 of each head. It loads the keys
 * and the values of the 4 rows together, computes the 32 scores (4 keys by 8
 * heads) with a reduce-scatter, and keeps the softmax running (the maximum m
 * and the sum l of each head). The sums of the values then take the new
 * weights at once. Each warp writes its part (FD_PARTS parts for each head);
 * k_attn_fd_join adds the parts. The keys and the values stream without the
 * phases of k_attn_part, and no scores go to memory. */
#define FD_KEYS 4
#define FD_BLOCKS 32                       /* blocks for each key head (32 was best of 32 to 256) */
#define FD_PARTS (FD_BLOCKS * 4)           /* 4 warps for each block */

__device__ __forceinline__ int fd_len(int n)
{
    int len = (n + FD_PARTS - 1) / FD_PARTS;
    len = len < 16 ? 16 : len;
    return (len + FD_KEYS - 1) / FD_KEYS * FD_KEYS;
}

__device__ __forceinline__ void fd_i8x8(uint2 u, float *out)
{
    uint32_t w[2] = {u.x, u.y};
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        out[t] = (float)(int8_t)((w[t / 4] >> (8 * (t % 4))) & 0xff);
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

__global__ void __launch_bounds__(128) k_attn_fd(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    __shared__ __align__(16) float qs[8 * 256];
    __shared__ float pw[4][40];
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int n = a.n, len = fd_len(n);
    int part_id = blockIdx.y * 4 + warp;
    int j0 = part_id * len, j1 = min(n, j0 + len);
    for (int i = threadIdx.x; i < 8 * 256; i += blockDim.x) {
        qs[i] = a.q[(size_t)kv * 8 * 256 + i];
    }
    __syncthreads();
    if (j0 >= n) {
        return;
    }
    float acc[8][8];
    #pragma unroll
    for (int h = 0; h < 8; ++h) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[h][u] = 0.f;
        }
    }
    float m = -INFINITY, l = 0.f;          /* of head lane % 8 */
    size_t rs = a.rstride, off = (size_t)kv * 256 + 8 * lane;
    for (int jb = j0; jb < j1; jb += FD_KEYS) {
        uint4 kq[FD_KEYS], vq[FD_KEYS];
        float ks[FD_KEYS], vs[FD_KEYS];
        #pragma unroll
        for (int kk = 0; kk < FD_KEYS; ++kk) {
            int j = min(jb + kk, j1 - 1);
            size_t o = (size_t)j * rs + off;
            kq[kk] = *(const uint4 *)(a.kq + o);
            vq[kk] = *(const uint4 *)(a.vq + o);
            ks[kk] = a.ks[o / 32];
            vs[kk] = a.vs[o / 32];
        }
        float v[32];
        #pragma unroll
        for (int kk = 0; kk < FD_KEYS; ++kk) {
            float kf[8];
            fd_i16x8(kq[kk], kf);
            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                const float4 q0 = *(const float4 *)(qs + h * 256 + 8 * lane);
                const float4 q1 = *(const float4 *)(qs + h * 256 + 8 * lane + 4);
                float d = q0.x * kf[0] + q0.y * kf[1] + q0.z * kf[2] + q0.w * kf[3] +
                          q1.x * kf[4] + q1.y * kf[5] + q1.z * kf[6] + q1.w * kf[7];
                v[kk * 8 + h] = d * ks[kk];
            }
        }
        /* the reduce-scatter of k_attn_part: value kk * 8 + h to lane kk * 8 + h */
        #pragma unroll
        for (int o2 = 16, nv = 32; o2 > 0; o2 >>= 1, nv >>= 1) {
            bool up = (lane & o2) != 0;
            #pragma unroll
            for (int x = 0; x < nv / 2; ++x) {
                float keep = up ? v[x + nv / 2] : v[x];
                float send = up ? v[x] : v[x + nv / 2];
                v[x] = keep + __shfl_xor_sync(RC_FULL_MASK, send, o2);
            }
        }
        float sc = jb + lane / 8 < j1 ? v[0] : -INFINITY;
        /* the maximum and the sum of each head over the 4 keys: the lanes
         * h, h + 8, h + 16, h + 24 */
        float mx = fmaxf(sc, __shfl_xor_sync(RC_FULL_MASK, sc, 8));
        mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, 16));
        float mn = fmaxf(m, mx);
        float alpha = m == -INFINITY ? 0.f : expf(m - mn);
        float p = sc == -INFINITY ? 0.f : expf(sc - mn);
        float ps = p + __shfl_xor_sync(RC_FULL_MASK, p, 8);
        ps += __shfl_xor_sync(RC_FULL_MASK, ps, 16);
        l = l * alpha + ps;
        m = mn;
        pw[warp][lane] = p;
        if (lane < 8) {
            pw[warp][32 + lane] = alpha;
        }
        __syncwarp();
        #pragma unroll
        for (int h = 0; h < 8; ++h) {
            float al = pw[warp][32 + h];
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[h][u] *= al;
            }
        }
        #pragma unroll
        for (int kk = 0; kk < FD_KEYS; ++kk) {
            float vf[8];
            fd_i16x8(vq[kk], vf);
            #pragma unroll
            for (int h = 0; h < 8; ++h) {
                float pp = pw[warp][kk * 8 + h] * vs[kk];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    acc[h][u] += pp * vf[u];
                }
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (int h = 0; h < 8; ++h) {
        float *o = part + ((size_t)(kv * 8 + h) * FD_PARTS + part_id) * 258;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            o[8 * lane + u] = acc[h][u];
        }
        if (lane == h) {
            o[256] = m;
            o[257] = l;
        }
    }
}

/* Block (h, x) joins the values x * 128 to x * 128 + 127 of query head h
 * from the parts of k_attn_fd (as k_attn_join). */
__global__ void k_attn_fd_join(const gp_rec *r, const int64_t *e, const float *part)
{
    PDL_START();
    __shared__ float w[FD_PARTS];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x;
    int len = fd_len(a.n);
    int nc = (a.n + len - 1) / len;
    const float *ph = part + (size_t)h * FD_PARTS * 258;
    float M = -INFINITY;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        M = fmaxf(M, ph[(size_t)c * 258 + 256]);
    }
    M = block_max(M);
    float wl = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float mc = ph[(size_t)c * 258 + 256];
        float wc = mc == -INFINITY ? 0.f : expf(mc - M);
        w[c] = wc;
        wl += wc * ph[(size_t)c * 258 + 257];
    }
    float wsum = block_sum(wl);
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    __syncthreads();
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < 256) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) {
            acc += w[c] * ph[(size_t)c * 258 + i];
        }
        a.out[(size_t)h * 256 + i] = acc * inv;
    }
}


/* k_attn_fd for other shapes: head_dim HD (256 or 512) and G query heads
 * for each key head (G divides 32; the Gemma 4 26B has HD 256 with G 2 and
 * HD 512 with G 8). A step of a warp takes 32 / G keys, so the scores are
 * 32 values (keys by heads) again. Lane l keeps values 8 l to 8 l + 7 of
 * each part of 256 values. The keys come first (the scores and the running
 * softmax), then the values, so fewer registers hold loads at a time. The
 * record gives the keys of the window, so no key needs a mask. */
template <int HD, int G, bool KQ8 = false, bool VQ8 = false>
__global__ void __launch_bounds__(128) k_attn_fdt(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int NP = HD / 256, K = 32 / G;
    __shared__ __align__(16) float qs[G * HD];
    __shared__ float pw[4][32 + G];
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int n = a.n, len = fd_len(n);
    len = (len + K - 1) / K * K;
    int part_id = blockIdx.y * 4 + warp;
    int j0 = part_id * len, j1 = min(n, j0 + len);
    for (int i = threadIdx.x; i < G * HD; i += blockDim.x) {
        qs[i] = a.q[(size_t)kv * G * HD + i];
    }
    __syncthreads();
    if (j0 >= n) {
        return;
    }
    float acc[G][NP][8];
    #pragma unroll
    for (int h = 0; h < G; ++h) {
        #pragma unroll
        for (int s = 0; s < NP; ++s) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[h][s][u] = 0.f;
            }
        }
    }
    float m = -INFINITY, l = 0.f;          /* of head lane % G */
    size_t rs = a.rstride, off = (size_t)kv * HD + 8 * lane;
    for (int jb = j0; jb < j1; jb += K) {
        float v[32];
        #pragma unroll
        for (int kk = 0; kk < K; ++kk) {
            int j = min(jb + kk, j1 - 1);
            #pragma unroll
            for (int h = 0; h < G; ++h) {
                v[kk * G + h] = 0.f;
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
                for (int h = 0; h < G; ++h) {
                    const float4 q0 = *(const float4 *)(qs + h * HD + 256 * s + 8 * lane);
                    const float4 q1 = *(const float4 *)(qs + h * HD + 256 * s + 8 * lane + 4);
                    float d = q0.x * kf[0] + q0.y * kf[1] + q0.z * kf[2] + q0.w * kf[3] +
                              q1.x * kf[4] + q1.y * kf[5] + q1.z * kf[6] + q1.w * kf[7];
                    v[kk * G + h] += d * sk;
                }
            }
        }
        /* the reduce-scatter: value kk * G + h to lane kk * G + h */
        #pragma unroll
        for (int o2 = 16, nv = 32; o2 > 0; o2 >>= 1, nv >>= 1) {
            bool up = (lane & o2) != 0;
            #pragma unroll
            for (int x = 0; x < nv / 2; ++x) {
                float keep = up ? v[x + nv / 2] : v[x];
                float send = up ? v[x] : v[x + nv / 2];
                v[x] = keep + __shfl_xor_sync(RC_FULL_MASK, send, o2);
            }
        }
        float sc = jb + lane / G < j1 ? v[0] : -INFINITY;
        /* the maximum and the sum of each head over the K keys: the lanes
         * h, h + G, h + 2 G, ... */
        float mx = sc;
        #pragma unroll
        for (int o2 = G; o2 < 32; o2 <<= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, o2));
        }
        float mn = fmaxf(m, mx);
        float alpha = m == -INFINITY ? 0.f : expf(m - mn);
        float p = sc == -INFINITY ? 0.f : expf(sc - mn);
        float ps = p;
        #pragma unroll
        for (int o2 = G; o2 < 32; o2 <<= 1) {
            ps += __shfl_xor_sync(RC_FULL_MASK, ps, o2);
        }
        l = l * alpha + ps;
        m = mn;
        pw[warp][lane] = p;
        if (lane < G) {
            pw[warp][32 + lane] = alpha;
        }
        __syncwarp();
        #pragma unroll
        for (int h = 0; h < G; ++h) {
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
                for (int h = 0; h < G; ++h) {
                    float pp = pw[warp][kk * G + h] * sv;
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
    for (int h = 0; h < G; ++h) {
        float *o = part + ((size_t)(kv * G + h) * FD_PARTS + part_id) * (HD + 2);
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

/* k_attn_fd_join for the parts of k_attn_fdt: block (h, x) joins the values
 * x * 128 to x * 128 + 127 of query head h. */
template <int HD, int G>
__global__ void k_attn_fdt_join(const gp_rec *r, const int64_t *e, const float *part)
{
    PDL_START();
    __shared__ float w[FD_PARTS];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x;
    int len = fd_len(a.n);
    len = (len + 32 / G - 1) / (32 / G) * (32 / G);
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
