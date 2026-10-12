/* ---------- the gated residual and the n-gram layer of Qwen3.8 ----------
 * The records of backends/cpu/csrc/common/hyperconn.c; the operands are the same. */

__device__ __forceinline__ float hc_sig(float v)
{
    return 1.f / (1.f + expf(-v));
}

/* GP_HC_NORM: x, w, out, t, groups, hid, eps. A block for each group of
 * each row. */
__device__ void row_quant8(const float *x, int n, size_t b0, int8_t *q, float *xs, float *xsum);
__global__ void k_hc_norm(const gp_rec *r, const int64_t *e, int8_t *q = NULL, float *qs = NULL,
                          float *qsum = NULL)
{
    PDL_START();
    int groups = DI(4), hid = DI(5);
    size_t o = (size_t)blockIdx.x * hid;
    const float *x = DP(const float, 0) + o;
    const float *w = DP(const float, 1) + (size_t)(blockIdx.x % groups) * hid;
    float *out = DP(float, 2) + o;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        ss += x[c] * x[c];
    }
    ss = block_sum(ss);
    float inv = 1.f / sqrtf(ss / (float)hid + df(r, e, 6));
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        out[c] = x[c] * inv * w[c];
    }
    if (q != NULL) {
        __syncthreads();
        row_quant8(out, hid, (size_t)blockIdx.x * (hid / 32), q, qs, qsum);
    }
}

/* The int8 x (quant8_block, the bits of k_kq_quant_x) of the n values at x
 * (written by this block before a __syncthreads), from block b0 of 32 on:
 * all the threads of the block call it. */
__device__ void row_quant8(const float *x, int n, size_t b0, int8_t *q, float *xs, float *xsum)
{
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nw = blockDim.x / 32, nb = n / 32;
    for (int base = warp * 4; base < nb; base += nw * 4) {
        int b = base + lane / 8;
        bool live = b < nb;
        float4 v = live ? *(const float4 *)(x + (size_t)b * 32 + 4 * (lane % 8))
                        : make_float4(0.f, 0.f, 0.f, 0.f);
        quant8_block(v, live, b0 + (size_t)b, lane % 8, q, xs, xsum);
    }
}

/* GP_HC_ACT: x, out, n, scale. out = silu(x * scale). */
__global__ void k_hc_act(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < di(r, e, 2)) {
        float v = DP(const float, 0)[i] * df(r, e, 3);
        DP(float, 1)[i] = v * hc_sig(v);
    }
}

/* GP_HC_MIX: hn, g, out, t, hc, hid. A thread for each value of out. */
__global__ void k_hc_mix(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)DI(3) * hid) {
        return;
    }
    int64_t j = i / hid, c = i % hid;
    const float *hn = DP(const float, 0), *g = DP(const float, 1);
    float s = 0.f;
    for (int k = 0; k < hc; ++k) {
        size_t x = ((size_t)j * hc + k) * hid + c;
        s += hc_sig(g[x]) * hn[x];
    }
    DP(float, 2)[i] = s / (float)hc;
}

/* GP_HC_ADD: H, out, inject, t, hc, hid, scale. A thread for each value of
 * H. */
__global__ void k_hc_add(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)DI(3) * hc * hid) {
        return;
    }
    int64_t rg = i / hid, j = rg / hc, c = i % hid;
    float w = 2.f * hc_sig(DP(const float, 2)[rg] * df(r, e, 6));
    float o = DP(const float, 1)[j * hid + c];
    if (r->tag[7] != GP_T_NONE) {
        o += DP(const float, 7)[j * hid + c];     /* the add before (gpu35._fuse: the MoE parts) */
    }
    DP(float, 0)[i] += o * w;
}

/* GP_HC_ADD (r) and the GP_HC_NORM of the same H that follows (r2), one
 * block for each stream of each row: the add, then the norm with the sums
 * of k_hc_norm in the same order (the same bits, one kernel in place of two;
 * RECLANK_GPU_HC_FUSE). q: the int8 x of the norm (row_quant8). */
__global__ void k_hc_add_norm(const gp_rec *r, const int64_t *e, const gp_rec *r2, int8_t *q,
                              float *qs, float *qsum)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int rg = blockIdx.x, j = rg / hc;
    float wv = 2.f * hc_sig(DP(const float, 2)[rg] * df(r, e, 6));
    float *H = DP(float, 0) + (size_t)rg * hid;
    const float *o = DP(const float, 1) + (size_t)j * hid;
    const float *o2 = r->tag[7] != GP_T_NONE ? DP(const float, 7) + (size_t)j * hid : NULL;
    float ss = 0.f;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        H[c] += (o2 != NULL ? o[c] + o2[c] : o[c]) * wv;
        ss += H[c] * H[c];
    }
    {
        const gp_rec *r0 = r;
        r = r2;
        int groups = DI(4);
        const float *w = DP(const float, 1) + (size_t)(rg % groups) * hid;
        float *out = DP(float, 2) + (size_t)rg * hid;
        ss = block_sum(ss);
        float inv = 1.f / sqrtf(ss / (float)hid + df(r, e, 6));
        for (int c = threadIdx.x; c < hid; c += blockDim.x) {
            out[c] = H[c] * inv * w[c];
        }
        if (q != NULL) {
            __syncthreads();
            row_quant8(out, hid, (size_t)rg * (hid / 32), q, qs, qsum);
        }
        r = r0;
    }
}

/* GP_HC_ACT and GP_HC_MIX with the int8 x of their output for the next
 * product: 4 values for each thread, 8 threads for a block of 32. */
__global__ void k_hc_act_q(const gp_rec *r, const int64_t *e, int8_t *q, float *qs, float *qsum)
{
    PDL_START();
    int64_t n = di(r, e, 2), i4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool live = 4 * i4 < n;
    float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
    if (live) {
        float4 x = *(const float4 *)(DP(const float, 0) + 4 * i4);
        float sc = df(r, e, 3), a;
        a = x.x * sc; v.x = a * hc_sig(a);
        a = x.y * sc; v.y = a * hc_sig(a);
        a = x.z * sc; v.z = a * hc_sig(a);
        a = x.w * sc; v.w = a * hc_sig(a);
        *(float4 *)(DP(float, 1) + 4 * i4) = v;
    }
    quant8_block(v, live, (size_t)(i4 / 8), threadIdx.x % 8, q, qs, qsum);
}

__global__ void k_hc_mix_q(const gp_rec *r, const int64_t *e, int8_t *q, float *qs, float *qsum)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int64_t i4 = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    bool live = 4 * i4 < (int64_t)DI(3) * hid;
    float vv[4] = {0.f, 0.f, 0.f, 0.f};
    if (live) {
        const float *hn = DP(const float, 0), *g = DP(const float, 1);
        for (int u = 0; u < 4; ++u) {
            int64_t i = 4 * i4 + u, j = i / hid, c = i % hid;
            float sm = 0.f;
            for (int k = 0; k < hc; ++k) {
                size_t x = ((size_t)j * hc + k) * hid + c;
                sm += hc_sig(g[x]) * hn[x];
            }
            vv[u] = sm / (float)hc;
        }
    }
    float4 v = make_float4(vv[0], vv[1], vv[2], vv[3]);
    if (live) {
        *(float4 *)(DP(float, 2) + 4 * i4) = v;
    }
    quant8_block(v, live, (size_t)(i4 / 8), threadIdx.x % 8, q, qs, qsum);
}

static int hc_fuse_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_HC_FUSE");
        on = !(v && v[0] == '0');
    }
    return on;
}

static int topk_split_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_TOPK_SPLIT");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* GP_PLE_GATE: keyn, qn, value, gated, t, hc, hid. A block for each stream
 * of each row. */
__global__ void k_ple_gate(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hc = DI(5), hid = DI(6);
    size_t o = (size_t)blockIdx.x * hid;
    const float *k = DP(const float, 0) + o, *q = DP(const float, 1) + o;
    float s = 0.f;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        s += k[c] * q[c];
    }
    s = block_sum(s) / sqrtf((float)hid);
    float mag = sqrtf(fabsf(s) > 1e-6f ? fabsf(s) : 1e-6f);
    float g = hc_sig(s < 0.f ? -mag : (s > 0.f ? mag : 0.f));
    const float *v = DP(const float, 2) + (size_t)(blockIdx.x / hc) * hid;
    float *out = DP(float, 3) + o;
    for (int c = threadIdx.x; c < hid; c += blockDim.x) {
        out[c] = g * v[c];
    }
}

/* GP_PLE_CONV: gn, gated, H, state, w, t, channels, kernel, dilation. A
 * thread for each channel: the rows in order, then the new state. */
__global__ void k_ple_conv(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int t = DI(5), channels = DI(6), kernel = DI(7), dil = DI(8);
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    const float *gn = DP(const float, 0), *gated = DP(const float, 1), *w = DP(const float, 4);
    float *H = DP(float, 2), *state = DP(float, 3);
    int hist = (kernel - 1) * dil;
    for (int j = 0; j < t; ++j) {
        float v = 0.f;
        for (int k = 0; k < kernel; ++k) {
            int rr = j - (kernel - 1 - k) * dil;
            float x = rr >= 0 ? gn[(size_t)rr * channels + c]
                              : state[(size_t)(hist + rr) * channels + c];
            v += w[(size_t)c * kernel + k] * x;
        }
        size_t i = (size_t)j * channels + c;
        H[i] += gated[i] + v * hc_sig(v);
    }
    for (int rr = 0; rr < hist; ++rr) {
        int src = t - hist + rr;
        state[(size_t)rr * channels + c] = src >= 0 ? gn[(size_t)src * channels + c]
                                                    : state[(size_t)(hist + src) * channels + c];
    }
}

/* GP_HC_CAT: e, hn, out, t, hc, hid. For each stream: the row of e, then
 * the stream. */
__global__ void k_hc_cat(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hc = DI(4), hid = DI(5);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)DI(3) * hc * 2 * hid) {
        return;
    }
    int64_t rg = i / (2 * hid), c = i % (2 * hid);
    DP(float, 2)[i] = c < hid ? DP(const float, 0)[(rg / hc) * hid + c]
                              : DP(const float, 1)[rg * hid + c - hid];
}

/* GP_HC_PRE: hn, wd, wu, wi, out, inj, part, t, hc, hid, lr, scale. The input
 * of a block from the normed streams hn (t x hc x hid), in place of the
 * records KQ_LINEAR (down: lr x hc hid, BF12), HC_ACT, KQ_LINEAR (up: hc hid
 * x lr, BF12), HC_MIX, and KQ_LINEAR (inject: hc x hc hid, float32, or none;
 * gpu35._fuse_hc). Two kernels, t <= HC_PRE_T:
 *
 * k_hc_pre_down: block (rows 8 b .. 8 b + 7 of down and inject, stream s) with
 *   the hid values of stream s of each token in shared memory; a warp a row,
 *   the sum over the columns of stream s into part[s][j][row].
 * k_hc_pre_up: lo = the sum of the parts, loa = silu(lo scale) in shared
 *   memory; a warp for each column c of out: the hc rows c + s hid of up on
 *   loa, then out[j][c] = mean over s of sigmoid(g_s) hn[j][s][c]; block 0
 *   writes inj (the sum of the parts of the inject rows).
 *
 * The 7 launches of a site became 3 (k_hc_add_norm, these two), and the
 * products read their x from shared memory, not each row its 40 KB. */
#define HC_PRE_T 4

/* The x of the products in shared memory: float4 q of the chunk of 16
 * values k at 4 k + (q ^ (k / 2 % 4)). The lanes of a warp take the chunks
 * k = lane (+ 32 i); the 8 lanes of a phase of a float4 load then cover the
 * 32 banks (in order, 4 lanes on the same 4 banks). */
__device__ __forceinline__ int hc_sw4(int f)
{
    return (f & ~3) | ((f ^ (f >> 3)) & 3);
}

/* kq_bf12_fma16 with x in that order (the same products and sums) */
template <int NT>
__device__ __forceinline__ void hc_fma16(const uint32_t *p, const float *x, int cols, int c, int t, float *sum)
{
    float wv[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        wv[2 * i] = __uint_as_float(p[i] << 16);
        wv[2 * i + 1] = __uint_as_float(p[i] & 0xffff0000u);
    }
    const int k = c >> 4, sw = (k >> 1) & 3;
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        if (j < t) {
            const float4 *x4 = (const float4 *)(x + (size_t)j * cols) + 4 * k;
            float s = sum[j];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 a = x4[i ^ sw];
                s = fmaf(wv[4 * i], a.x, s);
                s = fmaf(wv[4 * i + 1], a.y, s);
                s = fmaf(wv[4 * i + 2], a.z, s);
                s = fmaf(wv[4 * i + 3], a.w, s);
            }
            sum[j] = s;
        }
    }
}

__global__ void __launch_bounds__(256) k_hc_pre_down(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ float hx[];               /* t x hid */
    const int t = DI(7), hc = DI(8), hid = DI(9), lr = DI(10), s = blockIdx.y;
    const int ni = DP(const float, 3) != NULL ? hc : 0, rows = lr + ni, HD = hc * hid;
    const float *hn = DP(const float, 0);
    for (int i = threadIdx.x; i < t * hid / 4; i += 256) {
        int j = i / (hid / 4), c = i % (hid / 4);
        ((float4 *)hx)[j * (hid / 4) + hc_sw4(c)] = ((const float4 *)(hn + ((size_t)j * hc + s) * hid))[c];
    }
    __syncthreads();
    const int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    float sum[HC_PRE_T] = {0.f, 0.f, 0.f, 0.f};
    if (row < lr) {
        const uint8_t *w = DP(const uint8_t, 1) + (size_t)row * kq_row_bytes(KQ_BF12, HD);
        const int h = lane & 1;
        for (int c0 = s * hid + 16 * lane; c0 < (s + 1) * hid; c0 += 512) {
            uint4 L = *(const uint4 *)(w + c0);
            uint4 H = *(const uint4 *)(w + HD + 16 * (c0 >> 5));
            uint32_t pw[8];
            kq_bf12_dec16(L, H, w[HD + HD / 2 + (c0 >> 5)], h, pw);
            hc_fma16<HC_PRE_T>(pw, hx, hid, c0 - s * hid, t, sum);
        }
    } else {
        const float *w = DP(const float, 3) + (size_t)(row - lr) * HD + (size_t)s * hid;
        for (int c = 4 * lane; c < hid; c += 128) {
            float4 a = *(const float4 *)(w + c);
            #pragma unroll
            for (int j = 0; j < HC_PRE_T; ++j) {
                if (j < t) {
                    float4 b = ((const float4 *)(hx + (size_t)j * hid))[hc_sw4(c / 4)];
                    sum[j] += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
            }
        }
    }
    #pragma unroll
    for (int j = 0; j < HC_PRE_T; ++j) {
        float v = sum[j];
        for (int o = 16; o > 0; o >>= 1) {
            v += __shfl_xor_sync(RC_FULL_MASK, v, o);
        }
        if (lane == 0 && j < t) {
            DP(float, 6)[((size_t)s * t + j) * rows + row] = v;
        }
    }
}

__global__ void __launch_bounds__(256) k_hc_pre_up(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ float la[];               /* t x lr */
    const int t = DI(7), hc = DI(8), hid = DI(9), lr = DI(10);
    const int ni = DP(const float, 3) != NULL ? hc : 0, rows = lr + ni, HD = hc * hid;
    const float scale = df(r, e, 11);
    const float *part = DP(const float, 6);
    for (int i = threadIdx.x; i < t * lr; i += 256) {
        int j = i / lr, q = i % lr;
        float v = 0.f;
        for (int s = 0; s < hc; ++s) {
            v += part[((size_t)s * t + j) * rows + q];
        }
        v *= scale;
        la[j * lr + 4 * hc_sw4(q / 4) + q % 4] = v * hc_sig(v);
    }
    if (blockIdx.x == 0 && ni > 0) {
        for (int i = threadIdx.x; i < t * ni; i += 256) {
            int j = i / ni, q = i % ni;
            float v = 0.f;
            for (int s = 0; s < hc; ++s) {
                v += part[((size_t)s * t + j) * rows + lr + q];
            }
            DP(float, 5)[(size_t)j * hc + q] = v;
        }
    }
    __syncthreads();
    const int c = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32, h = lane & 1;
    if (c >= hid) {
        return;
    }
    const size_t rb = kq_row_bytes(KQ_BF12, lr);
    const float *hn = DP(const float, 0);
    float acc[HC_PRE_T] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < hc; ++s) {
        const uint8_t *w = DP(const uint8_t, 2) + (size_t)(s * hid + c) * rb;
        float g[HC_PRE_T] = {0.f, 0.f, 0.f, 0.f};
        for (int c0 = 16 * lane; c0 < lr; c0 += 512) {
            uint4 L = *(const uint4 *)(w + c0);
            uint4 H = *(const uint4 *)(w + lr + 16 * (c0 >> 5));
            uint32_t pw[8];
            kq_bf12_dec16(L, H, w[lr + lr / 2 + (c0 >> 5)], h, pw);
            hc_fma16<HC_PRE_T>(pw, la, lr, c0, t, g);
        }
        #pragma unroll
        for (int j = 0; j < HC_PRE_T; ++j) {
            float v = g[j];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (j < t) {
                acc[j] += hc_sig(v) * hn[((size_t)j * hc + s) * hid + c];
            }
        }
    }
    if (lane == 0) {
        for (int j = 0; j < t; ++j) {
            DP(float, 4)[(size_t)j * hid + c] = acc[j] / (float)hc;
        }
    }
    (void)HD;
}
