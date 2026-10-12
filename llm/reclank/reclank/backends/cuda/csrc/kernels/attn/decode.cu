/* ---------- attention ---------- */

/* The operands of the norm of the query, the key, and the value. GP_QKV_NORM
 * has them in this order. GP_QKV_NORM_ROPE has head_dim and eps at 12 and
 * 13. */
struct qkv_ix {
    int q, qw, qr, k, kw, kr, v, vr, hd, eps;
};

/* The norm of each head of q, k, and v, in place. One block for each row of
 * head_dim values. A row of v has no weight. */
__global__ void k_qkv_norm(const gp_rec *r, const int64_t *e, qkv_ix ix)
{
    PDL_START();
    int row = blockIdx.x;
    int qr = DI(ix.qr), kr = DI(ix.kr), hd = DI(ix.hd);
    float *x;
    const float *w;
    if (row < qr) {
        x = DP(float, ix.q) + (size_t)row * hd;
        w = DP(const float, ix.qw);
    } else if (row < qr + kr) {
        x = DP(float, ix.k) + (size_t)(row - qr) * hd;
        w = DP(const float, ix.kw);
    } else {
        x = DP(float, ix.v) + (size_t)(row - qr - kr) * hd;
        w = NULL;
    }
    float ss = 0.f;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)hd + df(r, e, ix.eps));
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        x[i] = w ? x[i] * s * w[i] : x[i] * s;
    }
}

/* The operands of the rope. GP_ROPE has them in this order. */
struct rope_ix {
    int q, qr, qh, k, kr, kh, cos, sin, hd;
};

/* The rope of each head of q and k, in place. One block for each row. As
 * gemma_rope_body. */
__global__ void k_rope(const gp_rec *r, const int64_t *e, rope_ix ix)
{
    PDL_START();
    int row = blockIdx.x;
    int qr = DI(ix.qr), hd = DI(ix.hd), d = hd / 2;
    float *x;
    int tok;
    if (row < qr) {
        x = DP(float, ix.q) + (size_t)row * hd;
        tok = row / DI(ix.qh);
    } else {
        x = DP(float, ix.k) + (size_t)(row - qr) * hd;
        tok = (row - qr) / DI(ix.kh);
    }
    const float *c = DP(const float, ix.cos) + (size_t)tok * hd;
    const float *s = DP(const float, ix.sin) + (size_t)tok * hd;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float a = x[i], b = x[i + d];
        x[i] = a * c[i] - b * s[i];
        x[i + d] = b * c[i] + a * s[i];
    }
}

/* k_qkv_norm and then k_rope in one kernel (GP_QKV_NORM_ROPE): 128 threads
 * for a row, and thread t keeps the values t + 128 k (k < NK = head_dim /
 * 128) in registers. The sum of the squares and the values are those of the
 * two kernels: the same order, and the rope reads the same floats. The rope
 * pairs value i with value i + head_dim / 2, which is k with k + NK / 2. */
template <int NK>
__global__ void __launch_bounds__(128) k_qkv_norm_rope(const gp_rec *r, const int64_t *e,
                                                       qkv_ix ix, rope_ix rx)
{
    PDL_START();
    int row = blockIdx.x;
    int qr = DI(ix.qr), kr = DI(ix.kr), hd = DI(ix.hd);
    float *x;
    const float *w;
    int tok = -1;
    if (row < qr) {
        x = DP(float, ix.q) + (size_t)row * hd;
        w = DP(const float, ix.qw);
        tok = row / DI(rx.qh);
    } else if (row < qr + kr) {
        x = DP(float, ix.k) + (size_t)(row - qr) * hd;
        w = DP(const float, ix.kw);
        tok = (row - qr) / DI(rx.kh);
    } else {
        x = DP(float, ix.v) + (size_t)(row - qr - kr) * hd;
        w = NULL;
    }
    float v[NK];
    float ss = 0.f;
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        v[k] = x[threadIdx.x + 128 * k];
        ss += v[k] * v[k];
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)hd + df(r, e, ix.eps));
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        int i = threadIdx.x + 128 * k;
        v[k] = w ? v[k] * s * w[i] : v[k] * s;
    }
    if (tok >= 0) {
        const float *c = DP(const float, rx.cos) + (size_t)tok * hd;
        const float *sn = DP(const float, rx.sin) + (size_t)tok * hd;
        #pragma unroll
        for (int k = 0; k < NK / 2; ++k) {
            int i = threadIdx.x + 128 * k;
            float a = v[k], b = v[k + NK / 2];
            v[k] = a * c[i] - b * sn[i];
            v[k + NK / 2] = b * c[i] + a * sn[i];
        }
    }
    #pragma unroll
    for (int k = 0; k < NK; ++k) {
        x[threadIdx.x + 128 * k] = v[k];
    }
}

/* The row stride of a float32 cache from its head stride hs: position-major
 * (positions, kv heads, hd) has hs == hd and rows of kvh hd values; the
 * head-major form (kv heads, positions, hd) has rows of hd. All the caches
 * are position-major now; the kernels take both. */
__host__ __device__ __forceinline__ size_t kv_rs(size_t hs, int kvh, int hd)
{
    return hs == (size_t)hd ? (size_t)kvh * hd : (size_t)hd;
}

/* k, v, kbuf, vbuf, head_stride, pos, tokens, kv_heads, head_dim. One block
 * for each head of each token. */
__global__ void k_kv_write_heads(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int kvh = DI(7), hd = DI(8);
    int j = blockIdx.x / kvh, h = blockIdx.x % kvh;
    size_t src = ((size_t)j * kvh + h) * hd;
    size_t hs = (size_t)di(r, e, 4);
    size_t dst = (size_t)h * hs + (size_t)(di(r, e, 5) + j) * kv_rs(hs, kvh, hd);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        DP(float, 2)[dst + i] = DP(const float, 0)[src + i];
        DP(float, 3)[dst + i] = DP(const float, 1)[src + i];
    }
}

/* The attention in two kernels, with the method of FlashDecoding. A decode
 * step has few heads, so one block for each head leaves most of the GPU idle.
 * Here the keys of each key and value head go to chunks of len keys. The
 * value len is n / ATTN_CHUNKS, but at least ATTN_MIN_KEYS. Thus a short context gets
 * a few small chunks, and a long context gets ATTN_CHUNKS chunks.
 *
 * The launch
 * has ATTN_CHUNKS blocks for each head. A block past the last chunk does
 * nothing.
 *
 * The kernel k_attn_part: block (kv, c) takes chunk c of the keys of key and
 * value head kv. It does this for all the query heads of that head (rep
 * heads), so it reads each key and value row one time. For each query head h and chunk c,
 * it writes hd + 2 values to the scratch of the program:
 *
 * - the sum of exp(score - m) times the value rows;
 * - m, the maximum of the scores of the chunk;
 * - l, the sum of exp(score - m).
 *
 * The kernel k_attn_join: block h adds the parts of head h. Part c gets the
 * weight exp(m_c - M), where M is the largest m_c. The kernel then divides by
 * the sum of the weighted l_c. */
#define ATTN_CHUNKS 256
#define ATTN_MIN_KEYS 32
#define ATTN_TILE 128
#define ATTN_REP 8
#define ATTN_KEYS 4          /* ATTN_KEYS * ATTN_REP = 32, the lanes of a warp */

/* The keys of a chunk, and the count of chunks, for n keys. */
__device__ __forceinline__ int attn_len(int n)
{
    return max(ATTN_MIN_KEYS, (n + ATTN_CHUNKS - 1) / ATTN_CHUNKS);
}

/* The operands of one attention, for the three layouts of the cache. */
struct attn_d {
    const float *q;
    const float *k, *v;          /* the float cache */
    const int16_t *kq, *vq;      /* the int16 cache */
    const int8_t *kq8, *vq8;     /* the int8 keys (kf8) and values (vf8) */
    const float *ks, *vs;        /* its scale for each group of 32 values */
    float *sc, *out;
    int qh, kvh, hd, n, window, i16;  /* i16: a quantized cache */
    int kf8, vf8;                /* the keys, the values in int8 */
    int64_t kp0, p;              /* the position of key row 0, and of the query */
    size_t hstride, rstride;     /* the distance of two heads and of two rows */
    const int32_t *rows;         /* GP_ATTN_QSA: the rows of the keys, or null */
    const uint8_t *kt, *vt;      /* tq: the TQ6 cache (tq6x8), q and out rotated */
    int tq;
    const float *cb;             /* tq: the codebook (in shared memory, or tq6_cb_d) */
};

/* GP_ATTN_F32H reads the E4B cache, which has the shape (heads, positions,
 * head_dim). Its operands:
 *
 *     q, k, v, scores, out, q_heads, kv_heads, head_dim, tokens (1), pos,
 *     head_stride, window, slide
 *
 * The pointers k and v point at position 0. With slide, the keys start at
 * the first position of the window.
 *
 * GP_ATTN_F32 reads the float cache of the 26B model, which has the shape
 * (rows, kv_heads, head_dim). Its operands:
 *
 *     q, k, v, scores, out, q_heads, kv_heads, head_dim, n, pos, base, window
 *
 * The pointers k and v point at the first key row. Its position is base.
 *
 * GP_ATTN_QC reads the int16 cache of the 26B model, with the same shape.
 * Its operands:
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, n
 *
 * The pointers point at the first key row. Every one of the n rows is in
 * the window of the query.
 *
 * GP_ATTN_QSA (Qwen3.8, backends/cpu/csrc/common/qsa.c) reads the int16 cache on the rows that
 * QSA_SELECT gave. On the GPU, a record has one query (t = 1):
 *
 *     q, kq, ks, vq, vs, scores, out, q_heads, kv_heads, head_dim, 1, pos,
 *     sel, cnt, maxsel
 *
 * cnt[0] is the count of the rows in sel, or -1 for the pos + 1 rows. */
__device__ attn_d attn_get(const gp_rec *r, const int64_t *e)
{
    attn_d a;
    memset(&a, 0, sizeof(a));
    a.q = DP(const float, 0);
    if (r->op == GP_ATTN_QC || r->op == GP_ATTN_QSA || r->op == GP_ATTN_Q8 ||
        r->op == GP_ATTN_V8 || r->op == GP_ATTN_TQ) {
        a.i16 = 1;
        a.tq = r->op == GP_ATTN_TQ || (r->op == GP_ATTN_QSA && DI(15) == 4);
        a.cb = tq6_cb_d;
        a.kt = DP(const uint8_t, 1);
        a.vt = DP(const uint8_t, 3);
        a.kf8 = r->op == GP_ATTN_Q8;
        a.vf8 = r->op == GP_ATTN_Q8 || r->op == GP_ATTN_V8;
        if (r->op == GP_ATTN_QSA) {
            /* operand 15 (or none): 0 int16, 1 int8, 2 int16 keys and int8
             * values (3 float32 and 4 TQ6 below) */
            int form = DI(15);
            a.kf8 = form == 1;
            a.vf8 = form >= 1;
        }
        a.kq = DP(const int16_t, 1);
        a.kq8 = DP(const int8_t, 1);
        a.vq8 = DP(const int8_t, 3);
        a.ks = DP(const float, 2);
        a.vq = DP(const int16_t, 3);
        a.vs = DP(const float, 4);
        a.sc = DP(float, 5);
        a.out = DP(float, 6);
        a.qh = DI(7);
        a.kvh = DI(8);
        a.hd = DI(9);
        a.n = DI(10);
        a.window = 0;
        a.kp0 = 0;
        a.p = a.n;
        if (r->op == GP_ATTN_QSA) {
            int c = DP(const int32_t, 13)[0];
            a.p = di(r, e, 11);
            a.n = c < 0 ? (int)(a.p + 1) : c;
            a.rows = c < 0 ? NULL : DP(const int32_t, 12);
        }
        a.hstride = (size_t)a.hd;
        a.rstride = (size_t)a.kvh * a.hd;
        if (r->op == GP_ATTN_QSA && DI(15) == 3) {
            /* float32 rows: head stride operand 16 (kv_rs) */
            a.i16 = 0;
            a.kf8 = a.vf8 = 0;
            a.k = DP(const float, 1);
            a.v = DP(const float, 3);
            a.hstride = (size_t)di(r, e, 16);
            a.rstride = kv_rs(a.hstride, a.kvh, a.hd);
        }
        return a;
    }
    a.sc = DP(float, 3);
    a.out = DP(float, 4);
    a.qh = DI(5);
    a.kvh = DI(6);
    a.hd = DI(7);
    a.p = di(r, e, 9);
    a.window = DI(11);
    if (r->op == GP_ATTN_F32H) {
        int64_t lo = (DI(12) && a.window > 0) ? a.p - a.window + 1 : 0;
        if (lo < 0) {
            lo = 0;
        }
        a.kp0 = lo;
        a.n = (int)(a.p + 1 - lo);
        a.hstride = (size_t)di(r, e, 10);
        a.rstride = kv_rs(a.hstride, a.kvh, a.hd);
        a.k = DP(const float, 1) + (size_t)lo * a.rstride;
        a.v = DP(const float, 2) + (size_t)lo * a.rstride;
    } else {
        a.kp0 = di(r, e, 10);
        a.n = DI(8);
        a.hstride = (size_t)a.hd;
        a.rstride = (size_t)a.kvh * a.hd;
        a.k = DP(const float, 1);
        a.v = DP(const float, 2);
    }
    return a;
}

/* Read 8 values as floats: values i to i + 7 of row j of head kv. With key
 * 1 they come from the key cache, and with key 0 from the value cache. The
 * index i is a multiple of 8. Thus one load of 16 bytes (int16) or two loads
 * of 16 bytes (float) read them. Wide loads keep more bytes in flight. A
 * load of one value for each lane reached only about 58 GB/s. */
__device__ __forceinline__ void attn_kv8(const attn_d &a, int key, int kv, int j, int i,
                                         float *out)
{
    size_t o = (size_t)(a.rows ? a.rows[j] : j) * a.rstride + (size_t)kv * a.hstride + i;
    if (a.tq) {
        tq6x8(key ? a.kt : a.vt, o, out, a.cb);
        float sc = (key ? a.ks : a.vs)[o / 32];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            out[t] *= sc;
        }
    } else if (key ? a.kf8 : a.vf8) {
        const int8_t *q = key ? a.kq8 : a.vq8;
        float sc = (key ? a.ks : a.vs)[o / 32];
        uint2 u = *(const uint2 *)(q + o);
        uint32_t w[2] = {u.x, u.y};
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            out[t] = (float)(int8_t)((w[t / 4] >> (8 * (t % 4))) & 0xff) * sc;
        }
    } else if (a.i16) {
        const int16_t *q = key ? a.kq : a.vq;
        float sc = (key ? a.ks : a.vs)[o / 32];
        uint4 u = *(const uint4 *)(q + o);
        uint32_t w[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            out[2 * t] = (float)(int16_t)(w[t] & 0xffff) * sc;
            out[2 * t + 1] = (float)(int16_t)(w[t] >> 16) * sc;
        }
        if (!key && a.vq == a.kq && a.vs != a.ks && ko_turned(i)) {
            /* keys only (ko_tab): a turned value from its key and the partner */
            size_t po = o - i + ko_partner(i);
            float y[8];
            ko_i16x8(*(const uint4 *)(a.kq + po), a.vs[po / 32], y);
            ko_rot8(out, y, a.kp0 + (a.rows ? a.rows[j] : j), i);
        }
    } else {
        const float *p = (key ? a.k : a.v) + o;
        float4 x = *(const float4 *)p, y = *(const float4 *)(p + 4);
        out[0] = x.x; out[1] = x.y; out[2] = x.z; out[3] = x.w;
        out[4] = y.x; out[5] = y.y; out[6] = y.z; out[7] = y.w;
    }
}

/* The block has 256 threads. head_dim must be 256 or 512. The query heads
 * of a key head come in hg groups of at most ATTN_REP (12 heads of Qwen3.8:
 * 2 groups of 6); block x takes group x % hg of key head x / hg. qb is the
 * first query head of the block. */
__global__ void k_attn_part(const gp_rec *r, const int64_t *e, float *part, int hg)
{
    PDL_START();
    __shared__ float qs[ATTN_REP * 512];
    __shared__ float ps[ATTN_REP * ATTN_TILE];
    __shared__ float red[ATTN_REP * 512];
    __shared__ float cbs[64];
    attn_d a = attn_get(r, e);
    if (a.tq) {
        if (threadIdx.x < 64) {
            cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
        }
        a.cb = cbs;
    }
    int kv = blockIdx.x / hg, c = blockIdx.y;
    int n = a.n, hd = a.hd, rep = a.qh / a.kvh / hg;
    int qb = kv * (a.qh / a.kvh) + (blockIdx.x % hg) * rep;
    int len = attn_len(n);
    int j0 = c * len, j1 = min(n, j0 + len);
    if (j0 >= n) {
        return;
    }
    for (int i = threadIdx.x; i < rep * hd; i += blockDim.x) {
        qs[i] = a.q[(size_t)qb * hd + i];
        red[i] = 0.f;
    }
    __syncthreads();
    /* The scores: each warp takes ATTN_KEYS keys at a time. Lane l reads
     * values 8 l to 8 l + 7 of each part of 256 values of each key. The loads
     * of the keys come first, and the sums of the lanes of the keys follow
     * each other, so more loads are in flight. */
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, nw = blockDim.x / 32;
    for (int jb = j0 + warp * ATTN_KEYS; jb < j1; jb += nw * ATTN_KEYS) {
        float s[ATTN_KEYS][ATTN_REP];
        #pragma unroll
        for (int kk = 0; kk < ATTN_KEYS; ++kk) {
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                s[kk][h] = 0.f;
            }
        }
        for (int base = 0; base < hd; base += 256) {
            int i = base + 8 * lane;
            float kv8[ATTN_KEYS][8];
            #pragma unroll
            for (int kk = 0; kk < ATTN_KEYS; ++kk) {
                if (jb + kk < j1) {
                    attn_kv8(a, 1, kv, jb + kk, i, kv8[kk]);
                } else {
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        kv8[kk][u] = 0.f;
                    }
                }
            }
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                if (h < rep) {
                    const float *qh = qs + h * hd + i;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        float qv = qh[u];
                        #pragma unroll
                        for (int kk = 0; kk < ATTN_KEYS; ++kk) {
                            s[kk][h] += qv * kv8[kk][u];
                        }
                    }
                }
            }
        }
        /* The 32 sums (ATTN_KEYS keys by ATTN_REP heads) over the lanes, as a
         * reduce-scatter: at each step a lane keeps half of its values and
         * sends the other half, so 31 shuffles leave value kk * ATTN_REP + h
         * on lane kk * ATTN_REP + h (a full sum for each value took 160). */
        float v[ATTN_KEYS * ATTN_REP];
        #pragma unroll
        for (int kk = 0; kk < ATTN_KEYS; ++kk) {
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                v[kk * ATTN_REP + h] = s[kk][h];
            }
        }
        #pragma unroll
        for (int off = 16, nv = 32; off > 0; off >>= 1, nv >>= 1) {
            bool up = (lane & off) != 0;
            #pragma unroll
            for (int x = 0; x < nv / 2; ++x) {
                float keep = up ? v[x + nv / 2] : v[x];
                float send = up ? v[x] : v[x + nv / 2];
                v[x] = keep + __shfl_xor_sync(RC_FULL_MASK, send, off);
            }
        }
        {
            int kk = lane / ATTN_REP, h = lane % ATTN_REP, j = jb + kk;
            if (h < rep && j < j1) {
                int64_t kp = a.rows ? a.rows[j] : a.kp0 + j;
                bool masked = kp > a.p || (a.window > 0 && a.p - kp >= a.window);
                a.sc[(size_t)(qb + h) * n + j] = masked ? -INFINITY : v[0];
            }
        }
    }
    __syncthreads();
    float m[ATTN_REP], l[ATTN_REP];
    #pragma unroll
    for (int h = 0; h < ATTN_REP; ++h) {
        m[h] = -INFINITY;
        l[h] = 0.f;
        if (h >= rep) {
            continue;
        }
        float *sc = a.sc + (size_t)(qb + h) * n;
        float mh = -INFINITY;
        for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
            mh = fmaxf(mh, sc[j]);
        }
        mh = block_max(mh);
        float lh = 0.f;
        for (int j = j0 + threadIdx.x; j < j1; j += blockDim.x) {
            float x = mh == -INFINITY ? 0.f : expf(sc[j] - mh);
            sc[j] = x;
            lh += x;
        }
        l[h] = block_sum(lh);
        m[h] = mh;
    }
    __syncthreads();
    /* The value rows. Thread t keeps values 8 d to 8 d + 7 of every query
     * head, where d = t mod (hd / 8). The threads with the same d form
     * groups, and group g takes the keys g, g + groups, and so on, of each
     * tile. At the end, the groups add their sums in shared memory. */
    int nd = hd / 8, groups = blockDim.x / nd;
    int d = threadIdx.x % nd, grp = threadIdx.x / nd;
    float acc[ATTN_REP][8];
    #pragma unroll
    for (int h = 0; h < ATTN_REP; ++h) {
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[h][u] = 0.f;
        }
    }
    for (int t0 = j0; t0 < j1; t0 += ATTN_TILE) {
        int tn = min(ATTN_TILE, j1 - t0);
        for (int x = threadIdx.x; x < rep * tn; x += blockDim.x) {
            int h = x / tn, jj = x % tn;
            ps[h * ATTN_TILE + jj] = a.sc[(size_t)(qb + h) * n + t0 + jj];
        }
        __syncthreads();
        #pragma unroll 8
        for (int jj = grp; jj < tn; jj += groups) {
            float vv[8];
            attn_kv8(a, 0, kv, t0 + jj, 8 * d, vv);
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                if (h < rep) {
                    float p = ps[h * ATTN_TILE + jj];
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        acc[h][u] += p * vv[u];
                    }
                }
            }
        }
        __syncthreads();
    }
    /* The groups add their sums one after the other, in the order of the
     * group. An atomic add gives the order of arrival, and the router of the
     * 26B turns such a difference in the last bit into another expert. */
    for (int gi = 0; gi < groups; ++gi) {
        if (grp == gi) {
            #pragma unroll
            for (int h = 0; h < ATTN_REP; ++h) {
                if (h < rep) {
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        red[h * hd + 8 * d + u] += acc[h][u];
                    }
                }
            }
        }
        __syncthreads();
    }
    for (int x = threadIdx.x; x < rep * hd; x += blockDim.x) {
        int h = x / hd, i = x % hd;
        part[((size_t)(qb + h) * ATTN_CHUNKS + c) * (hd + 2) + i] = red[x];
    }
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int h = 0; h < ATTN_REP; ++h) {
            if (h < rep) {
                float *o = part + ((size_t)(qb + h) * ATTN_CHUNKS + c) * (hd + 2);
                o[hd] = m[h];
                o[hd + 1] = l[h];
            }
        }
    }
}

__device__ __forceinline__ int ft_len(int nl, int nb);

/* Block (h, x) joins the values x * 128 to x * 128 + 127 of query head h.
 * The weight exp(m_c - M) of each chunk goes to shared memory first. nb 0:
 * the chunks of attn_len; else the nb chunks of k_attn_fdtc (ft_len) of the
 * gridDim.x / qh queries of the record, head h of query h / qh (its n +
 * h / qh keys, its chunks), chunk stride nb. */
__global__ void k_attn_join(const gp_rec *r, const int64_t *e, const float *part, int nb)
{
    PDL_START();
    __shared__ float w[ATTN_CHUNKS];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x, hd = a.hd;
    int n = a.n, stride = ATTN_CHUNKS, len;
    if (nb > 0) {
        stride = nb;
        n = a.n + h / a.qh;
        len = ft_len(n, nb);
    } else {
        len = attn_len(a.n);
    }
    int nc = (n + len - 1) / len;
    const float *ph = part + (size_t)h * stride * (hd + 2);
    float M = -INFINITY;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        M = fmaxf(M, ph[(size_t)c * (hd + 2) + hd]);
    }
    M = block_max(M);
    float wl = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float mc = ph[(size_t)c * (hd + 2) + hd];
        float wc = mc == -INFINITY ? 0.f : expf(mc - M);
        w[c] = wc;
        wl += wc * ph[(size_t)c * (hd + 2) + hd + 1];
    }
    float wsum = block_sum(wl);
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    __syncthreads();
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < hd) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) {
            acc += w[c] * ph[(size_t)c * (hd + 2) + i];
        }
        a.out[(size_t)h * hd + i] = acc * inv;
    }
}
