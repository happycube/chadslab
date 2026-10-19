/* ---------- small operations of other models (Qwen3.5) ---------- */

/* The router of a MoE layer as softmax, then the top k, then the k weights
 * divided by their sum (Qwen3.5, Mixtral). At an equal value the lower index
 * wins. One token for each thread. */
static void gp_router_topk_body(const float *logits, int t, int experts, int k, float *val,
                                int32_t *idx, const int32_t *map, float bias, float swap,
                                int32_t *sel)
{
    /* map (or NULL): the choice takes the logit + bias of each expert with
     * map[x] >= 0 (on the GPU), the weights the plain softmax (k_router_topk) */
    const float wb = map != NULL ? expf(bias) : 1.f;
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        const float *l = logits + (size_t)j * experts;
        float m = -INFINITY;
        for (int x = 0; x < experts; ++x) {
            m = l[x] > m ? l[x] : m;
        }
        float p[1024];
        float sum = 0.f;
        for (int x = 0; x < experts; ++x) {
            p[x] = expf(l[x] - m);
            sum += p[x];
            if (map != NULL && map[x] >= 0) {
                p[x] *= wb;
            }
        }
        float vs = 0.f;
        for (int s2 = 0; s2 < k; ++s2) {
            int b = -1;
            for (int x = 0; x < experts; ++x) {
                if (p[x] >= 0.f && (b < 0 || p[x] > p[b])) {
                    b = x;
                }
            }
            const float pb = map != NULL ? expf(l[b] - m) : p[b];
            idx[(size_t)j * k + s2] = b;
            val[(size_t)j * k + s2] = pb / sum;
            vs += pb / sum;
            p[b] = -1.f;
        }
        if (map != NULL && swap > 0.f) {
            /* RECLANK_GPU_HOT_SWAP (k_router_topk): sel gets the choice; from
             * the lightest, a chosen expert with map < 0 gives its place to
             * the best unchosen one with map >= 0 of at least swap times its
             * weight; the first with none ends it */
            int32_t *ij = idx + (size_t)j * k;
            float *vj = val + (size_t)j * k;
            if (sel != NULL) {
                memcpy(sel + (size_t)j * k, ij, (size_t)k * sizeof(*ij));
            }
            for (int s2 = k - 1; s2 >= 0; --s2) {
                if (map[ij[s2]] >= 0) {
                    continue;
                }
                int b = -1;
                for (int x = 0; x < experts; ++x) {
                    if (p[x] >= 0.f && map[x] >= 0 && (b < 0 || p[x] > p[b])) {
                        b = x;
                    }
                }
                if (b < 0) {
                    break;
                }
                const float cp = expf(l[b] - m) / sum;
                if (cp < swap * vj[s2]) {
                    break;
                }
                vs += cp - vj[s2];
                ij[s2] = b;
                vj[s2] = cp;
                p[b] = -1.f;
            }
        }
        for (int s2 = 0; s2 < k; ++s2) {
            val[(size_t)j * k + s2] /= vs;
        }
    }
}

/* The router of LFM2-MoE: the score of each expert is sigmoid(logit); the
 * choice takes the k largest of score + bias[x] (bias: the expert bias,
 * float32, for the choice only), the weights are the k scores divided by
 * their sum + eps (1e-6; GLM-5.3 1e-20), times scale (transformers
 * Lfm2MoeTopKRouter with norm_topk_prob, Glm5NextTextTopkRouter). At an
 * equal value the lower index wins. One token for each thread. */
static void gp_router_sig_body(const float *logits, int t, int experts, int k, float *val,
                               int32_t *idx, const float *bias, float scale, float eps)
{
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        const float *l = logits + (size_t)j * experts;
        float s[1024], c[1024];
        for (int x = 0; x < experts; ++x) {
            s[x] = 1.f / (1.f + expf(-l[x]));
            c[x] = s[x] + (bias != NULL ? bias[x] : 0.f);
        }
        float sum = 0.f;
        for (int s2 = 0; s2 < k; ++s2) {
            int b = -1;
            for (int x = 0; x < experts; ++x) {
                if (c[x] != -INFINITY && (b < 0 || c[x] > c[b])) {
                    b = x;
                }
            }
            idx[(size_t)j * k + s2] = b;
            val[(size_t)j * k + s2] = s[b];
            sum += s[b];
            c[b] = -INFINITY;
        }
        for (int s2 = 0; s2 < k; ++s2) {
            val[(size_t)j * k + s2] = val[(size_t)j * k + s2] / (sum + eps) * scale;
        }
    }
}

/* The double-gated short convolution of LFM2 (Lfm2MoeShortConv), for t
 * tokens: in (t x 3 ch) holds B, C, x of each token (the in_proj output, in
 * that order); bx = B * x goes through the causal depthwise convolution of
 * kernel taps w (ch x kernel, the last tap on the newest row) with the last
 * kernel - 1 rows of bx in state (kernel - 1 x ch, the oldest first), and
 * out (t x ch) = C * conv. state gets the last kernel - 1 rows of bx of the
 * first nreal tokens (0: all t; the other rows pad a group of the GPU). One
 * block of channels for each thread. */
static void gp_short_conv_body(const float *in, float *state, const float *w, float *out, int t,
                               int ch, int kernel, int nreal)
{
    if (nreal <= 0 || nreal > t) {
        nreal = t;
    }
    #pragma omp for schedule(static)
    for (int c0 = 0; c0 < ch; c0 += 64) {
        int c1 = c0 + 64 < ch ? c0 + 64 : ch;
        for (int c = c0; c < c1; ++c) {
            float hist[16];
            for (int q = 0; q < kernel - 1; ++q) {
                hist[q] = state[(size_t)q * ch + c];
            }
            const float *wc = w + (size_t)c * kernel;
            for (int j = 0; j < t; ++j) {
                const float *r = in + (size_t)j * 3 * ch;
                float bx = r[c] * r[2 * ch + c];
                float y = 0.f;
                for (int q = 0; q < kernel - 1; ++q) {
                    y += wc[q] * hist[q];
                }
                y += wc[kernel - 1] * bx;
                for (int q = 0; q + 1 < kernel - 1; ++q) {
                    hist[q] = hist[q + 1];
                }
                if (kernel > 1) {
                    hist[kernel - 2] = bx;
                }
                out[(size_t)j * ch + c] = r[ch + c] * y;
                if (j == nreal - 1) {
                    for (int q = 0; q < kernel - 1; ++q) {
                        state[(size_t)q * ch + c] = hist[q];
                    }
                }
            }
        }
    }
}

/* out = x * sigmoid(g), for n values. */
/* GP_Q16: n float32 values (n a multiple of 128) as int16 q, then a
 * float32 scale s for each 128 (out: 2 n bytes of q, then 4 n / 128 of s);
 * a value is q s, s = max |x| / 32767 of its 128. GP_DQ16 back. The
 * transfers of the mixed groups of Qwen at half the bytes (gpu35._moe_mix:
 * 42 MB a layer of a group of 4096 rows each way, the copies to the GPU
 * at 3 GB/s next to those of the experts). */
static void gp_q16_body(const float *x, uint8_t *out, int64_t n)
{
    int16_t *q = (int16_t *)out;
    float *sc = (float *)(out + 2 * n);
    #pragma omp for schedule(static)
    for (int64_t b = 0; b < n / 128; ++b) {
        const float *v = x + 128 * b;
        float m = 0.f;
        for (int i = 0; i < 128; ++i) {
            m = fmaxf(m, fabsf(v[i]));
        }
        const float inv = m > 0.f ? 32767.f / m : 0.f;
        for (int i = 0; i < 128; ++i) {
            q[128 * b + i] = (int16_t)lrintf(v[i] * inv);
        }
        sc[b] = m / 32767.f;
    }
}

static void gp_dq16_body(const uint8_t *in, float *x, int64_t n)
{
    const int16_t *q = (const int16_t *)in;
    const float *sc = (const float *)(in + 2 * n);
    #pragma omp for schedule(static)
    for (int64_t b = 0; b < n / 128; ++b) {
        const float s = sc[b];
        for (int i = 0; i < 128; ++i) {
            x[128 * b + i] = (float)q[128 * b + i] * s;
        }
    }
}

static void gp_sigmul_body(const float *x, const float *g, float *out, int64_t n)
{
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        out[i] = x[i] / (1.f + expf(-g[i]));
    }
}

/* The steps before the attention of a gated attention with partial RoPE
 * (Qwen3.5), for t tokens from pos:
 *
 * - qg (t x nq x 2 hd): the query and the gate of each head; with a null
 *   gate (LFM2, no output gate), qg is t x nq x hd, the query alone;
 * - the query: rms_norm with qn, RoPE on the first rot values, times scale,
 *   into qout (t x nq x hd); the gate into gate (t x nq x hd);
 * - kk, vv (t x nk x hd): the key gets rms_norm with kn and RoPE; the key and
 *   the value go to the cache K, V (nk heads, hs values between heads) at
 *   the positions pos to pos + t - 1. With a null K (the int16 cache), the
 *   key goes to kout (t x nk x hd) instead.
 *
 * cos and sin have rot values for each token (the two halves the same). */
static void gp_attn_prep_body(const float *qg, const float *kk, const float *vv,
                              const float *qn, const float *kn, const float *cos,
                              const float *sin, float *K, float *V, int64_t hs, int64_t pos,
                              int t, int nq, int nk, int hd, int rot, float eps, float scale,
                              float *qout, float *gate, float *kout)
{
    int half = rot / 2;
    #pragma omp for schedule(static)
    for (int x = 0; x < t * (nq + nk); ++x) {
        int j = x / (nq + nk), h = x % (nq + nk);
        const float *src;
        const float *w;
        float *dst;
        float sc;
        if (h < nq) {
            if (gate != NULL) {
                src = qg + ((size_t)j * nq + h) * 2 * hd;
                memcpy(gate + ((size_t)j * nq + h) * hd, src + hd, (size_t)hd * 4);
            } else {
                src = qg + ((size_t)j * nq + h) * hd;       /* no gate (LFM2): the query alone */
            }
            dst = qout + ((size_t)j * nq + h) * hd;
            w = qn;
            sc = scale;
        } else {
            int kh = h - nq;
            src = kk + ((size_t)j * nk + kh) * hd;
            if (K == NULL) {
                /* The int16 cache: the key goes to kout, and GP_KV_WRITE
                 * stores the key and the value. */
                dst = kout + ((size_t)j * nk + kh) * hd;
            } else {
                size_t rs = gp_kv_rs((size_t)hs, nk, hd);
                dst = K + (size_t)kh * hs + (size_t)(pos + j) * rs;
                memcpy(V + (size_t)kh * hs + (size_t)(pos + j) * rs,
                       vv + ((size_t)j * nk + kh) * hd, (size_t)hd * 4);
            }
            w = kn;
            sc = 1.f;
        }
        float ss = 0.f;
        for (int d = 0; d < hd; ++d) {
            ss += src[d] * src[d];
        }
        /* no norm weights (Nemotron): the rows as they are */
        float inv = w != NULL ? 1.f / sqrtf(ss / (float)hd + eps) : 1.f;
        float y[1024];
        for (int d = 0; d < hd; ++d) {
            y[d] = w != NULL ? src[d] * inv * w[d] : src[d];
        }
        const float *c = cos + (size_t)j * rot, *sn = sin + (size_t)j * rot;
        for (int d = 0; d < half; ++d) {
            float a = y[d], b = y[d + half];
            dst[d] = (a * c[d] - b * sn[d]) * sc;
            dst[d + half] = (b * c[d + half] + a * sn[d + half]) * sc;
        }
        for (int d = rot; d < hd; ++d) {
            dst[d] = y[d] * sc;
        }
    }
}

/* GP_NARG, GP_MAGIC, gp_rec, and the codes GP_*: the table of
 * reclank/ir/opcodes.py. */
#include "opcodes.h"

/* Set by a record that cannot run its shape (the int8 cache has one
 * kernel); rc_run then returns -3. */
static volatile int gp_fail;


int rc_gp_record_size(void)
{
    return (int)sizeof(gp_rec);
}

static inline int64_t gp_i(const gp_rec *r, const int64_t *e, int k)
{
    return r->tag[k] == GP_T_SLOT ? e[r->v[k]] : r->v[k];
}

static inline float gp_f(const gp_rec *r, const int64_t *e, int k)
{
    uint32_t u = (uint32_t)gp_i(r, e, k);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

#define GP_P(T, k) ((T *)(intptr_t)gp_i(r, e, (k)))
#define GP_I(k) ((int)gp_i(r, e, (k)))

/* out += d * v, one multiply and one add for each value, as NumPy does. A fused
 * multiply and add rounds one time and gives other bits. */
__attribute__((optimize("fp-contract=off")))
static void gp_add_scaled(float *out, const float *d, float v, int n)
{
    for (int c = 0; c < n; ++c) {
        float p = d[c] * v;
        out[c] = out[c] + p;
    }
}

/* The experts of one token, as Model._moe_one_token does it. The kernels run
 * the selected experts in the order of their index. Then the code adds their
 * outputs with the router weights, in the same order.
 *
 * GP_MOE gives the count of experts as operand 3. GP_MOE_N gives the address
 * of an int32 that holds the count. The GPU writes that count with the list
 * of the experts that it does not hold (reclank/backends/cuda/runtime.py). A count of 0 gives
 * an output of zeros. */
/* ---------- an expert device (reclank/backends/cuda/devices.py) ----------
 * A second GPU, driven by a second copy of the CUDA library in this process,
 * holds experts the GPU of the step does not: the cold experts of a step
 * that it holds go to it (post: its job on its stream, and back at once),
 * the others to the CPU (GP_MOE_N), then wait adds its sum. er_ctx is made
 * by devices.py (the same layout). */
typedef struct {
    int (*post)(int layer, int n, const int32_t *ids, const float *w, const float *h);
    int (*wait)(float *out);
    const uint8_t *held;         /* [layer][experts] */
    int64_t experts;
    int64_t alive;
    /* a group of t tokens (Nemotron: GP_NM_REMOTE_POST), idx the expert of each of its t k pairs
     * or -1; or null */
    int (*post_t)(int layer, int t, int k, const int32_t *idx, const float *w, const float *x);
} er_ctx;
#define ER_MAXK 16
#define ER_MAXP 256

/* REMOTE_POST: h, val, idx, count (an int32 in memory), ctx, layer, hidden
 * (unused), lidx, lval, lcount, posted. The pairs of the experts the device
 * holds go to it (posted gets 1); the others to lidx, lval and lcount, for
 * GP_MOE_N. */
static void gp_remote_post(const gp_rec *r, const int64_t *e)
{
    #pragma omp single
    {
        const float *h = GP_P(const float, 0);
        const float *val = GP_P(const float, 1);
        const int32_t *idx = GP_P(const int32_t, 2);
        /* (the count and the ids in pinned memory that the GPU writes: the count read once and at most
         * ER_MAXK, an id outside the layer's experts not posted) */
        int cnt = *GP_P(const volatile int32_t, 3);
        cnt = cnt < 0 ? 0 : cnt < ER_MAXK ? cnt : ER_MAXK;
        er_ctx *ctx = GP_P(er_ctx, 4);
        int layer = GP_I(5);
        int32_t *lidx = GP_P(int32_t, 7);
        float *lval = GP_P(float, 8);
        int32_t *lcnt = GP_P(int32_t, 9);
        int64_t *posted = GP_P(int64_t, 10);
        const uint8_t *held = ctx->held + (size_t)layer * ctx->experts;
        int32_t ids[ER_MAXK];
        float w[ER_MAXK];
        int n = 0, m = 0;
        for (int j = 0; j < cnt; ++j) {
            int x = idx[j];
            if (x < 0 || x >= ctx->experts) {
                continue;
            }
            if (ctx->alive && held[x] && n < ER_MAXK) {
                ids[n] = x;
                w[n] = val[j];
                ++n;
            } else {
                lidx[m] = x;
                lval[m] = val[j];
                ++m;
            }
        }
        *posted = 0;
        if (n > 0) {
            if (ctx->post(layer, n, ids, w, h) == 0) {
                *posted = 1;
            } else {
                /* (the device refused: its experts to the CPU) */
                for (int j = 0; j < n; ++j) {
                    lidx[m] = ids[j];
                    lval[m] = w[j];
                    ++m;
                }
            }
        }
        *lcnt = m;
    }
}

/* NM_REMOTE_POST: x (the latent rows), cold (the expert of each pair, or -1:
 * GP_NM_SPLIT), val, pairs, ctx, layer, lcold, posted, k. The cold pairs of
 * the experts the device holds go to it (posted gets 1), lcold is cold
 * without them, for GP_NM_MOE: one token (pairs = k) by post, a group by
 * post_t. */
static void gp_nm_remote_post(const gp_rec *r, const int64_t *e)
{
    #pragma omp single
    {
        const float *x = GP_P(const float, 0);
        const int32_t *cold = GP_P(const int32_t, 1);
        const float *val = GP_P(const float, 2);
        int np = GP_I(3);
        er_ctx *ctx = GP_P(er_ctx, 4);
        int layer = GP_I(5);
        int32_t *lcold = GP_P(int32_t, 6);
        int64_t *posted = GP_P(int64_t, 7);
        int k = r->tag[8] == GP_T_NONE ? np : GP_I(8);
        static int dbg = -1;
        if (dbg < 0) {
            const char *v = getenv("RECLANK_EX_DEBUG");
            dbg = v != NULL && atoi(v) != 0;
        }
        const uint8_t *held = ctx->held + (size_t)layer * ctx->experts;
        *posted = 0;
        if (np > k && (np > ER_MAXP || ctx->post_t == NULL)) {
            /* (a group larger than a job of the device: all its pairs to the CPU) */
            for (int p = 0; p < np; ++p) {
                lcold[p] = cold[p];
            }
            goto done;
        }
        if (np > k) {
            /* a group: its pairs of the device's experts in place, the others -1 */
            int32_t gidx[ER_MAXP];
            int n = 0;
            for (int p = 0; p < np; ++p) {
                int c = cold[p];
                lcold[p] = c;
                gidx[p] = -1;
                if (c >= 0 && c < ctx->experts && ctx->alive && ctx->post_t != NULL && np <= ER_MAXP && held[c]) {
                    gidx[p] = c;
                    lcold[p] = -1;
                    ++n;
                }
            }
            if (dbg) {
                fprintf(stderr, "[cpu] post layer %d: %d rows, %d of %d pairs to the device (thread %d)\n",
                        layer, np / k, n, np, omp_get_thread_num());
            }
            if (n > 0) {
                if (ctx->post_t(layer, np / k, k, gidx, val, x) == 0) {
                    *posted = 1;
                } else {
                    fprintf(stderr, "[reclank] NM_REMOTE_POST layer %d: the expert device refused a group of %d rows\n",
                            layer, np / k);
                    for (int p = 0; p < np; ++p) {
                        lcold[p] = cold[p];
                    }
                }
            }
            goto done;
        }
        int32_t ids[ER_MAXK], at[ER_MAXK];
        float w[ER_MAXK];
        int n = 0;
        for (int p = 0; p < np; ++p) {
            int c = cold[p];
            lcold[p] = c;
            if (c >= 0 && c < ctx->experts && ctx->alive && held[c] && n < ER_MAXK) {
                ids[n] = c;
                w[n] = val[p];
                at[n] = p;
                lcold[p] = -1;
                ++n;
            }
        }
        if (n > 0) {
            if (ctx->post(layer, n, ids, w, x) == 0) {
                *posted = 1;
            } else {
                /* (the device refused: its pairs to the CPU) */
                for (int j = 0; j < n; ++j) {
                    lcold[at[j]] = ids[j];
                }
            }
        }
    done:;
    }
}

/* REMOTE_JOIN: ctx, posted, out. After GP_MOE_N wrote the CPU's sum to out:
 * add the device's sum (wait for its job). */
static void gp_remote_join(const gp_rec *r, const int64_t *e)
{
    #pragma omp single
    {
        er_ctx *ctx = GP_P(er_ctx, 0);
        if (*GP_P(const int64_t, 1)) {
            static int dbg = -1;
            if (dbg < 0) {
                const char *v = getenv("RECLANK_EX_DEBUG");
                dbg = v != NULL && atoi(v) != 0;
            }
            if (dbg) {
                fprintf(stderr, "[cpu] join (thread %d)\n", omp_get_thread_num());
            }
            if (ctx->wait(GP_P(float, 2)) != 0) {
                fprintf(stderr, "[reclank] the expert device failed: its experts go to the CPU from now on\n");
                ctx->alive = 0;
            }
        }
    }
}

static void gp_moe_one(const gp_rec *r, const int64_t *e)
{
    const float *h = GP_P(const float, 0);
    const float *val = GP_P(const float, 1);
    const int32_t *idx = GP_P(const int32_t, 2);
    /* GP_MOE_N: the count in pinned memory that the GPU writes, read once by the thread of the
     * single below and given to all (copyprivate: two reads could differ), at most 64 */
    int top_k = r->op == GP_MOE_N ? 0 : GP_I(3);
    const uint8_t *gu_w = GP_P(const uint8_t, 4);
    const float *gu_s = GP_P(const float, 5);
    const uint8_t *dn_w = GP_P(const uint8_t, 6);
    const float *dn_s = GP_P(const float, 7);
    int gu_rows = GP_I(8);
    int cols = GP_I(9);
    int dn_rows = GP_I(10);
    int inner = GP_I(11);
    int32_t *ids = GP_P(int32_t, 12);
    float *act = GP_P(float, 13);
    float *act2 = GP_P(float, 14);
    float *de = GP_P(float, 15);
    float *out = GP_P(float, 16);
    #pragma omp single copyprivate(top_k)
    {
        if (r->op == GP_MOE_N) {
            int kc = *GP_P(const volatile int32_t, 3);
            top_k = kc < 0 ? 0 : kc < 64 ? kc : 64;
        }
        /* np.unique: the router gives distinct experts, so a sort is enough. */
        for (int j = 0; j < top_k; ++j) {
            ids[j] = idx[j];
        }
        for (int j = 1; j < top_k; ++j) {
            int32_t x = ids[j];
            int k = j - 1;
            while (k >= 0 && ids[k] > x) {
                ids[k + 1] = ids[k];
                --k;
            }
            ids[k + 1] = x;
        }
    }
    gemma_moe_gemv_gelu_body(gu_w, gu_s, h, ids, top_k, act, act2, gu_rows, cols, 0, inner);
    gemma_int4_moe_gemv_body(dn_w, dn_s, act2, ids, top_k, de, dn_rows, inner, inner);
    #pragma omp single
    {
        for (int c = 0; c < dn_rows; ++c) {
            out[c] = 0.0f;
        }
        for (int j = 0; j < top_k; ++j) {
            int s = 0;
            while (s < top_k && idx[s] != ids[j]) {
                ++s;
            }
            if (s < top_k) {
                gp_add_scaled(out, de + (size_t)j * (size_t)dn_rows, val[s], dn_rows);
            }
        }
    }
}

/* The key rows of query row j of a group: the first row lo and the count n.
 * The first query has the position pos, and row 0 of the buffer has the
 * position base. A sliding layer (window > 0) reads only the window. This is
 * the rule of Model._attention for a group. */
static inline void gp_rows(int64_t pos, int64_t base, int window, int j,
                           int *lo, int *n)
{
    int64_t p = pos + j;
    int64_t l = window > 0 ? p - window + 1 - base : 0;
    if (l < 0) {
        l = 0;
    }
    *lo = (int)l;
    *n = (int)(p + 1 - base - l);
}

/* The experts of a group of tokens, as Model._moe_mt does it.
 *
 * A pair is one (expert, token). The code sorts the pairs by expert with a
 * stable sort, so the pairs of one expert keep the order of their tokens.
 * The kernels then read each expert one time for all of its pairs. Last, each
 * token adds the outputs of its pairs in the order of the expert index, with
 * the router weights. */
static void gp_moe_group(const gp_rec *r, const int64_t *e)
{
    const float *h = GP_P(const float, 0);
    const float *val = GP_P(const float, 1);
    const int32_t *idx = GP_P(const int32_t, 2);
    int tokens = GP_I(3);
    int top_k = GP_I(4);
    const uint8_t *gu_w = GP_P(const uint8_t, 5);
    const float *gu_s = GP_P(const float, 6);
    const uint8_t *dn_w = GP_P(const uint8_t, 7);
    const float *dn_s = GP_P(const float, 8);
    int gu_rows = GP_I(9);
    int cols = GP_I(10);
    int dn_rows = GP_I(11);
    int inner = GP_I(12);
    int32_t *order = GP_P(int32_t, 13);    /* the flat index of each pair */
    int32_t *ids = GP_P(int32_t, 14);      /* the expert of each job */
    int32_t *poff = GP_P(int32_t, 15);     /* the first pair of each job */
    int32_t *xi = GP_P(int32_t, 16);       /* the token of each pair */
    int32_t *xi2 = GP_P(int32_t, 17);      /* the pair itself */
    int32_t *jobs_out = GP_P(int32_t, 18);
    float *act = GP_P(float, 19);
    float *act2 = GP_P(float, 20);
    float *de = GP_P(float, 21);
    float *out = GP_P(float, 22);
    const int pairs = tokens * top_k;
    #pragma omp single
    {
        /* A stable sort of the flat indices by expert (counting: a group of
         * a prompt has thousands of pairs, where the insertion sort of a
         * small group took time of the square of them). A pair with a
         * negative expert index is not for the CPU: the GPU computes it
         * (reclank/backends/cuda/runtime.py). The sort puts such pairs
         * first. */
        int ne = 0, s0 = 0;
        for (int f = 0; f < pairs; ++f) {
            if (idx[f] < 0) {
                ++s0;
            } else if (idx[f] + 1 > ne) {
                ne = idx[f] + 1;
            }
        }
        int *at = (int *)calloc((size_t)ne + 1, sizeof(int));
        for (int f = 0; f < pairs; ++f) {
            if (idx[f] >= 0) {
                ++at[idx[f] + 1];
            }
        }
        at[0] = s0;
        for (int x = 0; x < ne; ++x) {
            at[x + 1] += at[x];
        }
        int nn = 0;
        for (int f = 0; f < pairs; ++f) {
            if (idx[f] < 0) {
                order[nn++] = f;
            } else {
                order[at[idx[f]]++] = f;
            }
        }
        free(at);
        int jobs = 0;
        for (int p = s0; p < pairs; ++p) {
            int32_t ex = idx[order[p]];
            if (jobs == 0 || ids[jobs - 1] != ex) {
                ids[jobs] = ex;
                poff[jobs] = p;
                ++jobs;
            }
            xi[p] = order[p] / top_k;
            xi2[p] = p;
        }
        poff[jobs] = pairs;
        jobs_out[0] = jobs;
    }
    const int jobs = jobs_out[0];
    gemma_moe_gemv_gelu_mt_body(gu_w, gu_s, h, ids, poff, xi, jobs, act, act2,
                                gu_rows, cols, cols, inner);
    gemma_int4_moe_gemv_mt_body(dn_w, dn_s, act2, ids, poff, xi2, jobs, de,
                                dn_rows, inner, inner);
    #pragma omp single
    {
        for (int t = 0; t < tokens; ++t) {
            float *o = out + (size_t)t * (size_t)dn_rows;
            for (int c = 0; c < dn_rows; ++c) {
                o[c] = 0.0f;
            }
        }
        /* The pairs are in the order of the expert index. The pairs of one
         * token therefore come in that order too. */
        for (int p = 0; p < pairs; ++p) {
            int f = order[p];
            if (idx[f] < 0) {
                continue;
            }
            int t = f / top_k;
            gp_add_scaled(out + (size_t)t * (size_t)dn_rows,
                          de + (size_t)p * (size_t)dn_rows, val[f], dn_rows);
        }
    }
}
