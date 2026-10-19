/* ---------- the attention of a large group in float32 tiles (Pascal) ----------
 * k_flash_f32t: GP_ATTN_QC_MT, GP_ATTN_Q8_MT, GP_ATTN_V8_MT, and GP_ATTN_TQ_MT
 * (the int16, int8, int16-key int8-value, or TQ6 cache; the record of
 * k_flash_qc_mt; TQ6: the queries and the output in the rotated form of
 * tq6.py, GP_TQ_ROT before and after), and
 * GP_ATTN_F32H of a group (F32H: the float32 cache of the E4B) for a GPU
 * without the tensor cores. FlashAttention-2 in float32 with register tiles:
 * Pascal runs float16 at 1/64 of float32, and the probabilities of the
 * softmax keep their float32 bits.
 *
 * Block (h, tile): query head h and BQ queries (64 for head_dim 256, 32 for
 * 512), 256 threads; tx = thread % 16, ty = thread / 16. The tiles of the
 * last queries (the most keys) start first. For each tile of BK = 64 keys:
 *
 * 1. S = Q K^T: thread (tx, ty) holds QPT queries (ty QPT + u) by 4 keys
 *    (4 tx + v). The block reads 64 dims of the queries and of the keys
 *    (decoded from the cache) at a time to shared memory, both transposed
 *    (dim-major), so a thread reads its queries and its keys of a dim with
 *    one vector load each.
 * 2. The masks and the online softmax of each query: its 64 scores are in
 *    the 16 threads of a half warp (shuffles).
 * 3. O += P V: P to shared memory (key-major, in the place of the queries);
 *    thread (tx, ty) holds QPT queries by head_dim / 16 dims (64 c + 4 tx +
 *    w), 64 floats; the block reads 64 dims of the values at a time.
 *
 * 35 KB of shared memory. The queries come scaled (the record's form). */
#define F32T_BK 64
#define F32T_DC 64                 /* the dims of a step of S, and of P V */

template <int N>
__device__ __forceinline__ void f32t_ld(const float *p, float *v)
{
    if (N == 4) {
        float4 a = *(const float4 *)p;
        v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    } else {
        float2 a = *(const float2 *)p;
        v[0] = a.x; v[1] = a.y;
    }
}

/* Value w of the TQ6 group at b (tq6x8, one value): the low 4 bits of its
 * index in byte w % 16 (the high half for w >= 16), the high 2 bits in byte
 * 16 + w % 8. */
__device__ __forceinline__ float tq6_one(const uint8_t *b, int w, const float *cb)
{
    int lo = (b[w & 15] >> (w >= 16 ? 4 : 0)) & 15;
    int hi = (b[16 + (w & 7)] >> (2 * (w / 8))) & 3;
    return cb[lo | (hi << 4)];
}

template <int HD, bool KQ8, bool VQ8, bool F32H = false, bool TQ = false>
__global__ void __launch_bounds__(256) k_flash_f32t(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    constexpr int QPT = HD == 256 ? 4 : 2, BQ = 16 * QPT, NC = HD / F32T_DC;
    constexpr int QS = BQ + 4, KS = F32T_BK + 4;      /* the row strides (floats; 16-byte rows) */
    /* qp: the queries [dim][query] in S, then P [key][query]; kv_s: the keys
     * [dim][key] in S, then the values [key][dim] */
    __shared__ __align__(16) float qp[F32T_DC][QS];
    __shared__ __align__(16) float kv_s[F32T_DC][KS];
    __shared__ float cbs[TQ ? 64 : 1];          /* TQ6: the codebook */
    if (TQ) {
        if (threadIdx.x < 64) {
            cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
        }
        __syncthreads();
    }
    /* the operands of GP_ATTN_F32H: q, k, v, scores, out, q_heads, kv_heads,
     * head_dim, t, pos, head_stride, window (its cache from position 0) */
    const int o0 = F32H ? 5 : 7;
    const float *q = DP(const float, 0);
    float *out = DP(float, F32H ? 4 : 6);
    const int qh = DI(o0), kvh = DI(o0 + 1), t = DI(o0 + 3), window = F32H ? DI(11) : DI(13);
    const int64_t pos = di(r, e, F32H ? 9 : 11), base = F32H ? 0 : di(r, e, 12);
    const int h = blockIdx.x, kv = h / (qh / kvh);
    const int j0 = (gridDim.y - 1 - blockIdx.y) * BQ;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int jl = min(t, j0 + BQ) - 1;
    const int *lim = F32H ? NULL : DP(const int, 16);
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int64_t last = pos + qc_tile_last(lim, j0, jl);
    const size_t rowq = (size_t)kvh * HD;
    const void *kq = DP(const void, 1), *vq = DP(const void, F32H ? 2 : 3);
    const float *ksc = F32H ? NULL : DP(const float, 2), *vsc = F32H ? NULL : DP(const float, 4);
    /* F32H: row p of head kv at kv hs + p rs (kv_rs) */
    const size_t hs = F32H ? (size_t)di(r, e, 10) : 0, frs = F32H ? kv_rs(hs, kvh, HD) : 0;
    auto cache = [&](const void *b, const float *sc, bool i8, int64_t row, int d) {
        if (F32H) {
            return ((const float *)b)[(size_t)kv * hs + (size_t)row * frs + d];
        }
        size_t o = (size_t)row * rowq + (size_t)kv * HD + d;
        if (TQ) {   /* 24 bytes and a norm for each group of 32 values */
            return tq6_one((const uint8_t *)b + (o / 32) * 24, (int)(o % 32), cbs) * sc[o / 32];
        }
        float v = i8 ? (float)((const int8_t *)b)[o] : (float)((const int16_t *)b)[o];
        return v * sc[o / 32];
    };
    /* keys only (ko_tab): the values are the keys, turned back */
    const bool kon = !F32H && !TQ && !KQ8 && !VQ8 && kq == vq && ksc != vsc;
    int jq[QPT];
    int64_t pe[QPT];
    #pragma unroll
    for (int u = 0; u < QPT; ++u) {
        jq[u] = j0 + ty * QPT + u;
        pe[u] = jq[u] < t ? qc_last(lim, pos, jq[u]) : pos + jq[u];
    }
    float o[QPT][NC * 4], m[QPT], l[QPT];
    #pragma unroll
    for (int u = 0; u < QPT; ++u) {
        m[u] = -INFINITY;
        l[u] = 0.f;
        #pragma unroll
        for (int c = 0; c < NC * 4; ++c) {
            o[u][c] = 0.f;
        }
    }
    for (int64_t k0 = first; k0 <= last; k0 += F32T_BK) {
        const int kn = (int)min((int64_t)F32T_BK, last - k0 + 1);
        float s[QPT][4];
        #pragma unroll
        for (int u = 0; u < QPT; ++u) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                s[u][v] = 0.f;
            }
        }
        for (int d0 = 0; d0 < HD; d0 += F32T_DC) {
            /* the queries and the keys of 64 dims, transposed */
            for (int x = tid; x < BQ * F32T_DC; x += 256) {
                int a = x / F32T_DC, d = x % F32T_DC, jj = j0 + a;
                qp[d][a] = jj < t ? q[((size_t)jj * qh + h) * HD + d0 + d] : 0.f;
            }
            if (F32H || TQ || KQ8) {
                for (int x = tid; x < F32T_BK * F32T_DC; x += 256) {
                    int a = x / F32T_DC, d = x % F32T_DC;
                    kv_s[d][a] = a < kn ? cache(kq, ksc, KQ8, k0 - base + a, d0 + d) : 0.f;
                }
            } else {
                /* int16 keys: 8 values (one 16-byte load, one scale) a thread,
                 * the values of cache() (k_flash_f32w: 267 -> 327 tok/s at 8192) */
                for (int x = tid; x < F32T_BK * F32T_DC / 8; x += 256) {
                    int a = x / (F32T_DC / 8), d = (x % (F32T_DC / 8)) * 8;
                    float k8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    if (a < kn) {
                        size_t oo = (size_t)(k0 - base + a) * rowq + (size_t)kv * HD + d0 + d;
                        ko_i16x8(*(const uint4 *)((const int16_t *)kq + oo), ksc[oo / 32], k8);
                    }
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        kv_s[d + w][a] = k8[w];
                    }
                }
            }
            __syncthreads();
            #pragma unroll 8
            for (int d = 0; d < F32T_DC; ++d) {
                float a[QPT], b[4];
                f32t_ld<QPT>(&qp[d][ty * QPT], a);
                f32t_ld<4>(&kv_s[d][4 * tx], b);
                #pragma unroll
                for (int u = 0; u < QPT; ++u) {
                    #pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        s[u][v] = fmaf(a[u], b[v], s[u][v]);
                    }
                }
            }
            __syncthreads();
        }
        /* the masks and the online softmax; P to qp [key][query] */
        #pragma unroll
        for (int u = 0; u < QPT; ++u) {
            const int64_t p = pos + jq[u];
            float mt = m[u];
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                int kk = 4 * tx + v;
                int64_t kp = k0 + kk;
                bool ok = kk < kn && jq[u] < t && kp <= pe[u] && (window == 0 || p - kp < window);
                s[u][v] = ok ? s[u][v] : -INFINITY;
                mt = fmaxf(mt, s[u][v]);
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                mt = fmaxf(mt, __shfl_xor_sync(RC_FULL_MASK, mt, off));
            }
            float scale = (m[u] == -INFINITY || mt == -INFINITY) ? (m[u] == -INFINITY ? 0.f : 1.f)
                                                                 : expf(m[u] - mt);
            float ls = 0.f;
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                float pk = s[u][v] == -INFINITY ? 0.f : expf(s[u][v] - mt);
                qp[4 * tx + v][ty * QPT + u] = pk;
                ls += pk;
            }
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                ls += __shfl_xor_sync(RC_FULL_MASK, ls, off);
            }
            if (mt != -INFINITY) {
                l[u] = l[u] * scale + ls;
                #pragma unroll
                for (int c = 0; c < NC * 4; ++c) {
                    o[u][c] *= scale;
                }
                m[u] = mt;
            }
        }
        /* O += P V, 64 dims of the values at a time */
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            if (F32H || TQ || VQ8) {
                for (int x = tid; x < F32T_BK * F32T_DC; x += 256) {
                    int a = x / F32T_DC, d = x % F32T_DC;
                    int dd = c * F32T_DC + d;
                    float v = a < kn ? cache(vq, vsc, VQ8, k0 - base + a, dd) : 0.f;
                    if (kon && a < kn && ko_turned(dd & ~7)) {
                        int pd = ko_partner(dd & ~7) + (dd & 7);
                        v = ko_rot1(v, cache(vq, vsc, false, k0 - base + a, pd), k0 + a, dd);
                    }
                    kv_s[a][d] = v;
                }
            } else {
                /* int16 values: 8 a thread; the keys only: the partner's 8 too (ko_rot8) */
                for (int x = tid; x < F32T_BK * F32T_DC / 8; x += 256) {
                    int a = x / (F32T_DC / 8), d = (x % (F32T_DC / 8)) * 8;
                    int dd = c * F32T_DC + d;
                    float v8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    if (a < kn) {
                        size_t oo = (size_t)(k0 - base + a) * rowq + (size_t)kv * HD + dd;
                        ko_i16x8(*(const uint4 *)((const int16_t *)vq + oo), vsc[oo / 32], v8);
                        if (kon && ko_turned(dd)) {
                            size_t op = (size_t)(k0 - base + a) * rowq + (size_t)kv * HD + ko_partner(dd);
                            float y8[8];
                            ko_i16x8(*(const uint4 *)((const int16_t *)vq + op), vsc[op / 32], y8);
                            ko_rot8(v8, y8, k0 + a, dd);
                        }
                    }
                    *(float4 *)&kv_s[a][d] = make_float4(v8[0], v8[1], v8[2], v8[3]);
                    *(float4 *)&kv_s[a][d + 4] = make_float4(v8[4], v8[5], v8[6], v8[7]);
                }
            }
            __syncthreads();
            #pragma unroll 8
            for (int kk = 0; kk < F32T_BK; ++kk) {
                float pa[QPT], vb[4];
                f32t_ld<QPT>(&qp[kk][ty * QPT], pa);
                f32t_ld<4>(&kv_s[kk][4 * tx], vb);
                #pragma unroll
                for (int u = 0; u < QPT; ++u) {
                    #pragma unroll
                    for (int w = 0; w < 4; ++w) {
                        o[u][c * 4 + w] = fmaf(pa[u], vb[w], o[u][c * 4 + w]);
                    }
                }
            }
            __syncthreads();
        }
    }
    #pragma unroll
    for (int u = 0; u < QPT; ++u) {
        if (jq[u] >= t) {
            continue;
        }
        float inv = l[u] > 0.f ? 1.0f / l[u] : 0.f;
        #pragma unroll
        for (int c = 0; c < NC; ++c) {
            float *op = out + ((size_t)jq[u] * qh + h) * HD + c * F32T_DC + 4 * tx;
            #pragma unroll
            for (int w = 0; w < 4; ++w) {
                op[w] = o[u][c * 4 + w] * inv;
            }
        }
    }
}

/* k_flash_f32w: k_flash_f32t for head_dim 512 (the global layers of the 26B)
 * with more uses of each load of shared memory. k_flash_f32t loads 6 floats
 * for 8 products in S and in P V (2 queries by 4 keys or dims a thread), so
 * shared memory, not the arithmetic, set its rate (0.59 TFLOPS on the GTX
 * 1070 at 8192 tokens). Here, with W_BQ = 32 queries of head h a block:
 *
 * 1. S for W_BK = 128 keys: warp w holds queries 4w..4w+3, lane l keys
 *    4l..4l+3 (16 products for 8 floats); 32 dims of the queries and keys a
 *    step, transposed in shared memory. The softmax of a query is in its warp.
 * 2. P to shared memory [key][query], and the factor of each query's online
 *    softmax (rs: the rows of P V are not those of S).
 * 3. O += P V: thread (tr, td) = (tid / 64, tid % 64) holds queries
 *    8tr..8tr+7 by dims 8td..8td+7 (64 products for 16 floats); the values
 *    come W_VK = 8 keys of all 512 dims at a time.
 *
 * The rows of the softmax start again each 128 keys (64 in k_flash_f32t): the
 * same values but not the same bits. 40 KB of shared memory. */
#define W_BQ 32
#define W_BK 128
#define W_DC 32
#define W_VK 8

template <bool KQ8, bool VQ8>
__global__ void __launch_bounds__(256) k_flash_f32w(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    constexpr int HD = 512;
    /* region A: S (qs [W_DC][W_BQ + 4], ks [W_DC][W_BK + 4]) or the values
     * (vs [W_VK][HD + 4]); region B: P [W_BK][W_BQ + 4]; rs, ls of the rows */
    constexpr int A_S = W_DC * (W_BQ + 4) + W_DC * (W_BK + 4), A_V = W_VK * (HD + 4);
    __shared__ __align__(16) float sa[A_S > A_V ? A_S : A_V];
    __shared__ __align__(16) float ps[W_BK][W_BQ + 4];
    __shared__ float rs[W_BQ], ls[W_BQ];
    float (*qs)[W_BQ + 4] = (float (*)[W_BQ + 4])sa;
    float (*ks)[W_BK + 4] = (float (*)[W_BK + 4])(sa + W_DC * (W_BQ + 4));
    float (*vs)[HD + 4] = (float (*)[HD + 4])sa;
    const float *q = DP(const float, 0);
    float *out = DP(float, 6);
    const int qh = DI(7), kvh = DI(8), t = DI(10), window = DI(13);
    const int64_t pos = di(r, e, 11), base = di(r, e, 12);
    const int h = blockIdx.x, kv = h / (qh / kvh);
    const int j0 = (gridDim.y - 1 - blockIdx.y) * W_BQ;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int tr = tid / 64, td = tid % 64;
    const int jl = min(t, j0 + W_BQ) - 1;
    const int *lim = DP(const int, 16);
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int64_t last = pos + qc_tile_last(lim, j0, jl);
    const size_t rowq = (size_t)kvh * HD;
    const void *kq = DP(const void, 1), *vq = DP(const void, 3);
    const float *ksc = DP(const float, 2), *vsc = DP(const float, 4);
    auto cache = [&](const void *b, const float *sc, bool i8, int64_t row, int d) {
        size_t o = (size_t)row * rowq + (size_t)kv * HD + d;
        float v = i8 ? (float)((const int8_t *)b)[o] : (float)((const int16_t *)b)[o];
        return v * sc[o / 32];
    };
    const bool kon = !KQ8 && !VQ8 && kq == vq && ksc != vsc;
    /* the queries of S (warp w: 4w + u) */
    int jq[4];
    int64_t pe[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        jq[u] = j0 + 4 * warp + u;
        pe[u] = jq[u] < t ? qc_last(lim, pos, jq[u]) : pos + jq[u];
    }
    float m[4], l[4];
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        m[u] = -INFINITY;
        l[u] = 0.f;
    }
    float o[8][8];
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            o[a][c] = 0.f;
        }
    }
    for (int64_t k0 = first; k0 <= last; k0 += W_BK) {
        const int kn = (int)min((int64_t)W_BK, last - k0 + 1);
        float s[4][4];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                s[u][v] = 0.f;
            }
        }
        for (int d0 = 0; d0 < HD; d0 += W_DC) {
            for (int x = tid; x < W_BQ * W_DC; x += 256) {
                int a = x / W_DC, d = x % W_DC, jj = j0 + a;
                qs[d][a] = jj < t ? q[((size_t)jj * qh + h) * HD + d0 + d] : 0.f;
            }
            if (KQ8) {
                for (int x = tid; x < W_BK * W_DC; x += 256) {
                    int a = x / W_DC, d = x % W_DC;
                    ks[d][a] = a < kn ? cache(kq, ksc, KQ8, k0 - base + a, d0 + d) : 0.f;
                }
            } else {
                /* int16 keys: 8 values (one 16-byte load, one scale) a thread */
                for (int x = tid; x < W_BK * W_DC / 8; x += 256) {
                    int a = x / (W_DC / 8), d = (x % (W_DC / 8)) * 8;
                    float kv8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    if (a < kn) {
                        size_t oo = (size_t)(k0 - base + a) * rowq + (size_t)kv * HD + d0 + d;
                        ko_i16x8(*(const uint4 *)((const int16_t *)kq + oo), ksc[oo / 32], kv8);
                    }
                    #pragma unroll
                    for (int w = 0; w < 8; ++w) {
                        ks[d + w][a] = kv8[w];
                    }
                }
            }
            __syncthreads();
            #pragma unroll 8
            for (int d = 0; d < W_DC; ++d) {
                float a[4], b[4];
                f32t_ld<4>(&qs[d][4 * warp], a);
                f32t_ld<4>(&ks[d][4 * lane], b);
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    #pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        s[u][v] = fmaf(a[u], b[v], s[u][v]);
                    }
                }
            }
            __syncthreads();
        }
        /* the masks and the online softmax of the warp's 4 queries; P and the
         * factor of each row to shared memory */
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int64_t p = pos + jq[u];
            float mt = m[u];
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                int kk = 4 * lane + v;
                int64_t kp = k0 + kk;
                bool ok = kk < kn && jq[u] < t && kp <= pe[u] && (window == 0 || p - kp < window);
                s[u][v] = ok ? s[u][v] : -INFINITY;
                mt = fmaxf(mt, s[u][v]);
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                mt = fmaxf(mt, __shfl_xor_sync(RC_FULL_MASK, mt, off));
            }
            float scale = (m[u] == -INFINITY || mt == -INFINITY) ? (m[u] == -INFINITY ? 0.f : 1.f)
                                                                 : expf(m[u] - mt);
            float lsum = 0.f;
            #pragma unroll
            for (int v = 0; v < 4; ++v) {
                float pk = s[u][v] == -INFINITY ? 0.f : expf(s[u][v] - mt);
                ps[4 * lane + v][4 * warp + u] = pk;
                lsum += pk;
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                lsum += __shfl_xor_sync(RC_FULL_MASK, lsum, off);
            }
            if (mt != -INFINITY) {
                l[u] = l[u] * scale + lsum;
                m[u] = mt;
            } else {
                scale = 1.f;
            }
            if (lane == 0) {
                rs[4 * warp + u] = scale;
            }
        }
        __syncthreads();
        /* O += P V */
        #pragma unroll
        for (int a = 0; a < 8; ++a) {
            float sc = rs[8 * tr + a];
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                o[a][c] *= sc;
            }
        }
        for (int k1 = 0; k1 < kn; k1 += W_VK) {
            if (VQ8) {
                for (int x = tid; x < W_VK * HD; x += 256) {
                    int a = x / HD, dd = x % HD;
                    vs[a][dd] = k1 + a < kn ? cache(vq, vsc, VQ8, k0 - base + k1 + a, dd) : 0.f;
                }
            } else {
                /* int16 values: 8 a thread; the keys only (kon): the 8 of the
                 * partner too, turned back together (ko_rot8) */
                for (int x = tid; x < W_VK * HD / 8; x += 256) {
                    int a = x / (HD / 8), dd = (x % (HD / 8)) * 8;
                    float v8[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
                    if (k1 + a < kn) {
                        int64_t row = k0 - base + k1 + a;
                        size_t oo = (size_t)row * rowq + (size_t)kv * HD + dd;
                        ko_i16x8(*(const uint4 *)((const int16_t *)vq + oo), vsc[oo / 32], v8);
                        if (kon && ko_turned(dd)) {
                            size_t op = (size_t)row * rowq + (size_t)kv * HD + ko_partner(dd);
                            float y8[8];
                            ko_i16x8(*(const uint4 *)((const int16_t *)vq + op), vsc[op / 32], y8);
                            ko_rot8(v8, y8, k0 + k1 + a, dd);
                        }
                    }
                    *(float4 *)&vs[a][dd] = make_float4(v8[0], v8[1], v8[2], v8[3]);
                    *(float4 *)&vs[a][dd + 4] = make_float4(v8[4], v8[5], v8[6], v8[7]);
                }
            }
            __syncthreads();
            #pragma unroll
            for (int kk = 0; kk < W_VK; ++kk) {
                float pa[8], vb[8];
                f32t_ld<4>(&ps[k1 + kk][8 * tr], pa);
                f32t_ld<4>(&ps[k1 + kk][8 * tr + 4], pa + 4);
                f32t_ld<4>(&vs[kk][8 * td], vb);
                f32t_ld<4>(&vs[kk][8 * td + 4], vb + 4);
                #pragma unroll
                for (int a = 0; a < 8; ++a) {
                    #pragma unroll
                    for (int c = 0; c < 8; ++c) {
                        o[a][c] = fmaf(pa[a], vb[c], o[a][c]);
                    }
                }
            }
            __syncthreads();
        }
    }
    if (lane == 0) {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            ls[4 * warp + u] = l[u];
        }
    }
    __syncthreads();
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        int jj = j0 + 8 * tr + a;
        if (jj >= t) {
            continue;
        }
        float lv = ls[8 * tr + a];
        float inv = lv > 0.f ? 1.0f / lv : 0.f;
        float *op = out + ((size_t)jj * qh + h) * HD + 8 * td;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            op[c] = o[a][c] * inv;
        }
    }
}

/* RECLANK_GPU_F32W=0 keeps k_flash_f32t for head_dim 512 (a test). */
static int f32w_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_F32W");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* RECLANK_GPU_F32T=0 keeps k_flash_qc_mt (int16 cache) and the queries one by
 * one (the int8 caches of Qwen) on a GPU without the tensor cores (a test). */
static int f32t_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_F32T");
        on = !(v && v[0] == '0');
    }
    return on;
}
