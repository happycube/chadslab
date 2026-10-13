/* DeepSeek V4.1 Flash on the GPU (DEEPSEEK_V41_PLAN.md 3, Phase 3): the records
 * of a decode step other than the products (types.cu). The forms are those of
 * the CPU model in near mode (reclank/models/deepseek41/model.py): float32
 * activations, and the roundings that are part of the model as the reference
 * does them (the window rows to E4M3 with power-of-two scales for each 32,
 * the compressed rows to FP4 with E4M3 scales for each 16, the index keys and
 * queries to FP4 with E8M0 scales for each 32).
 *
 * The records read the position from the environment at run time (a slot),
 * so one graph serves every step; the records of the compressor do nothing
 * on a step that does not end a group of ratio tokens. The grids of the
 * indexer and the top-k are those of the longest cache (maxn) and stop at
 * the entries of the step.
 *
 * The hyper-connections have 4 streams (hc): the 24 mixes of a sublayer are
 * kept raw ([t, 24], k_ds_hc_mix), and each record that uses them makes the
 * coefficients it needs (ds_hc_coef: the sigmoids and the Sinkhorn of 4 x 4).
 */
#define DS_HC 4
#define DS_NMIX ((2 + DS_HC) * DS_HC)

/* The caches (the window rings, the compressed rows, the index keys): a row of
 * hd values in one of these forms. The records take the form of the caches
 * (1 bfloat16, 2 the smallest), each cache its own form of it (ds_cf):
 *   0  float32 (the rows of a group, kvq; not a cache);
 *   1  the bfloat16 bits (the three caches);
 *   2  FP8: hd E4M3 codes, then an E8M0 scale for each 32 (the window rows);
 *   4  FP4: hd / 2 bytes of E2M1 codes (value 2i in the low 4 bits), then an
 *      E4M3 scale for each 16 (the compressed rows: the form of the model);
 *   5  FP4: the same, an E8M0 scale for each 32 (the index keys).
 * Each holds the values the model puts there exactly (E4M3 times a power of
 * two; E2M1 times an E4M3 scale; E2M1 times a power of two; the sign of a
 * zero too), so the bits of the model do not change. */
#define DS_CW 0                 /* the kinds of cache, for ds_cf */
#define DS_CC 1
#define DS_CK 2
__device__ __forceinline__ int ds_cf(int form, int kind)
{
    return form <= 1 ? form : kind == DS_CW ? 2 : kind == DS_CC ? 4 : 5;
}

__device__ __forceinline__ size_t ds_rb(int hd, int form)
{
    switch (form) {
    case 0: return (size_t)hd * 4;
    case 1: return (size_t)hd * 2;
    case 2: return (size_t)hd + hd / 32;
    case 4: return (size_t)hd / 2 + hd / 16;
    default: return (size_t)hd / 2 + hd / 32;
    }
}

/* The value of an E2M1 code: twice the values 0, 0.5, 1, 1.5, 2, 3, 4, 6 are
 * the nibbles of 0xC8643210 (no table in local memory); the sign of a zero
 * is kept. */
__device__ __forceinline__ float ds_e2m1f(uint32_t c)
{
    float v = (float)((0xC8643210u >> (4 * (c & 7))) & 15u) * 0.5f;
    return __uint_as_float(__float_as_uint(v) | ((c & 8u) << 28));
}

/* the scale of value col of a row of FP8 or FP4 */
__device__ __forceinline__ float ds_bscale(const uint8_t *rw, int hd, int col, int form)
{
    if (form == 2) {
        return __int_as_float((int)rw[hd + col / 32] << 23);
    }
    if (form == 4) {
        return e4m3f(rw[hd / 2 + col / 16]);
    }
    return __int_as_float((int)rw[hd / 2 + col / 32] << 23);
}

/* value col of row of a cache */
__device__ __forceinline__ float ds_rld(const void *base, size_t row, int col, int hd, int form)
{
    if (form == 0) {
        return ((const float *)base)[row * hd + col];
    }
    if (form == 1) {
        return __uint_as_float((uint32_t)((const uint16_t *)base)[row * hd + col] << 16);
    }
    const uint8_t *rw = (const uint8_t *)base + row * ds_rb(hd, form);
    float q = form == 2 ? e4m3f(rw[col]) : ds_e2m1f(rw[col / 2] >> (4 * (col & 1)));
    return __fmul_rn(q, ds_bscale(rw, hd, col, form));
}

/* the 4 values from col (a multiple of 4) */
__device__ __forceinline__ float4 ds_rld4(const void *base, size_t row, int col, int hd, int form)
{
    if (form == 0) {
        return *(const float4 *)((const float *)base + row * hd + col);
    }
    if (form == 1) {
        uint2 u = *(const uint2 *)((const uint16_t *)base + row * hd + col);
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16),
                           __uint_as_float(u.y & 0xffff0000u));
    }
    const uint8_t *rw = (const uint8_t *)base + row * ds_rb(hd, form);
    float sc = ds_bscale(rw, hd, col, form);
    float q0, q1, q2, q3;
    if (form == 2) {
        uint32_t q = *(const uint32_t *)(rw + col);
        q0 = e4m3f(q & 255), q1 = e4m3f((q >> 8) & 255), q2 = e4m3f((q >> 16) & 255), q3 = e4m3f(q >> 24);
    } else {
        uint32_t q = *(const uint16_t *)(rw + col / 2);
        q0 = ds_e2m1f(q), q1 = ds_e2m1f(q >> 4), q2 = ds_e2m1f(q >> 8), q3 = ds_e2m1f(q >> 12);
    }
    return make_float4(__fmul_rn(q0, sc), __fmul_rn(q1, sc), __fmul_rn(q2, sc), __fmul_rn(q3, sc));
}

/* The E4M3 code of x, a value E4M3 holds exactly (the sign of a zero kept). */
__device__ __forceinline__ uint32_t ds_e4m3_code(float x)
{
    uint32_t sg = signbit(x) ? 0x80u : 0u;
    float a = fabsf(x);
    if (a == 0.f) {
        return sg;
    }
    int ex;
    frexpf(a, &ex);
    int e = ex - 1;
    if (e < -6) {
        return sg | (uint32_t)(int)(a * 512.f);            /* a subnormal: m 2^-9 */
    }
    return sg | (uint32_t)((e + 7) << 3) | (uint32_t)(int)((a / ldexpf(1.f, e) - 1.f) * 8.f);
}

/* The E2M1 code of y, an E2M1 value (the sign of a zero kept). */
__device__ __forceinline__ uint32_t ds_e2m1_code(float y)
{
    float a = fabsf(y);
    uint32_t c = a < 2.f ? (uint32_t)(a * 2.f) : a < 4.f ? (uint32_t)a + 2u : a == 4.f ? 6u : 7u;
    return c | (signbit(y) ? 8u : 0u);
}

/* Value col of row (all the lanes of a warp call it, a block of the FP8 and
 * FP4 forms in adjacent lanes): v the value, q the value over the block's
 * scale (its code), scale_byte the scale's byte (E8M0 or E4M3). */
__device__ __forceinline__ void ds_rst(void *base, size_t row, int col, int hd, int form, float v, float q,
                                       uint32_t scale_byte)
{
    if (form == 0) {
        ((float *)base)[row * hd + col] = v;
        return;
    }
    if (form == 1) {
        ((uint16_t *)base)[row * hd + col] = (uint16_t)(__float_as_uint(v) >> 16);
        return;
    }
    uint8_t *rw = (uint8_t *)base + row * ds_rb(hd, form);
    int bs = form == 4 ? 16 : 32;
    if (form == 2) {
        rw[col] = (uint8_t)ds_e4m3_code(q);
        if (col % 32 == 0) {
            rw[hd + col / 32] = (uint8_t)scale_byte;
        }
        return;
    }
    uint32_t c = ds_e2m1_code(q);
    uint32_t hi = __shfl_down_sync(RC_FULL_MASK, c, 1);
    if ((col & 1) == 0) {
        rw[col / 2] = (uint8_t)(c | (hi << 4));
    }
    if (col % bs == 0) {
        rw[hd / 2 + col / bs] = (uint8_t)scale_byte;
    }
}

/* the bytes of value col of row (and of its block's scale): a copy between
 * rows of the same form (the window rows: forms 1 and 2) */
__device__ __forceinline__ void ds_rcopy(void *dst, size_t drow, const void *src, size_t srow, int col, int hd,
                                         int form)
{
    size_t rb = ds_rb(hd, form);
    uint8_t *d = (uint8_t *)dst + drow * rb;
    const uint8_t *sp = (const uint8_t *)src + srow * rb;
    if (form <= 1) {
        int w = form == 0 ? 4 : 2;
        for (int k = 0; k < w; ++k) {
            d[col * w + k] = sp[col * w + k];
        }
        return;
    }
    d[col] = sp[col];
    if (col % 32 == 0) {
        d[hd + col / 32] = sp[hd + col / 32];
    }
}

/* the operand k of a record as the form of a cache (none: 0) */
__device__ __forceinline__ int ds_form(const gp_rec *r, const int64_t *e, int k)
{
    return r->tag[k] != GP_T_NONE ? (int)di(r, e, k) : 0;
}

/* kq_round_e4m3 of the CPU (mxx.c): nearest, ties to even, saturated at 448. */
__device__ __forceinline__ float ds_round_e4m3(float v)
{
    float a = fabsf(v);
    if (a == 0.f) {
        return v;
    }
    int ex;
    frexpf(a, &ex);
    ex -= 1;
    if (ex < -6) {
        ex = -6;
    }
    float step = ldexpf(1.f, ex - 3);
    float q = rintf(a / step) * step;
    q = fminf(q, 448.f);
    return v < 0.f ? -q : q;
}

/* fast_round_scale of the reference: 2^ceil(log2 v) from the bits (kq_pow2_ceil). */
__device__ __forceinline__ float ds_pow2_ceil(float v)
{
    uint32_t u = __float_as_uint(v);
    int ex = (int)((u >> 23) & 0xff) - 127 + ((u & 0x7fffff) != 0);
    return __uint_as_float((uint32_t)(ex + 127) << 23);
}

/* E2M1 of |v| <= 6: nearest of 0, 0.5, 1, 1.5, 2, 3, 4, 6, ties to the even
 * code (model.py _e2m1_round). */
__device__ __forceinline__ float ds_round_e2m1(float v)
{
    float a = fabsf(v), q;
    if (a < 2.f) {
        q = rintf(a * 2.f) * 0.5f;
    } else if (a < 4.f) {
        q = rintf(a);
    } else {
        q = rintf(a * 0.5f) * 2.f;
    }
    return copysignf(q, v);
}

/* fp4_act_quant of the reference on blocks of bs (16 or 32) adjacent lanes:
 * the value of this lane after the rounding. All lanes of the warp call it. */
__device__ __forceinline__ void ds_fp4_parts(float v, int bs, bool e4m3_scale, float *yq, float *sc)
{
    float amax = fabsf(v);
    for (int o = bs / 2; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(RC_FULL_MASK, amax, o));
    }
    float s;
    if (e4m3_scale) {
        s = ds_round_e4m3(fmaxf(amax, 6.f * 0.001953125f) / 6.f);
    } else {
        s = ds_pow2_ceil(__fmul_rn(fmaxf(amax, 6.f * __uint_as_float(0x00800000u)), 0.16666667f));
    }
    *yq = ds_round_e2m1(fminf(fmaxf(v / s, -6.f), 6.f));
    *sc = s;
}

/* the value after the rounding (the E2M1 value times the scale) */
__device__ __forceinline__ float ds_fp4(float v, int bs, bool e4m3_scale)
{
    float y, s;
    ds_fp4_parts(v, bs, e4m3_scale, &y, &s);
    return __fmul_rn(y, s);
}

/* act_quant of the reference (E4M3, a power-of-two scale for each 32) on the
 * 32 lanes of a warp: the value of this lane after the rounding. */
__device__ __forceinline__ float ds_act_e4m3(float v)
{
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(RC_FULL_MASK, amax, o));
    }
    float sc = ds_pow2_ceil(__fmul_rn(fmaxf(amax, 1e-4f), 1.f / 448.f));
    float y = fminf(fmaxf(v / sc, -448.f), 448.f);
    return __fmul_rn(ds_round_e4m3(y), sc);
}

/* The rotation of the pair (a, b) by (c, s), without contractions (the
 * float32 operations of model.py rope). Lane parity odd gives the second. */
__device__ __forceinline__ float ds_rot(float a, float b, float c, float s, int odd)
{
    return odd ? __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c)) : __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
}

__device__ __forceinline__ float ds_sigmoid(float v)
{
    return 1.f / (1.f + expf(-v));
}

/* The coefficients of a sublayer from its raw mixes (model.py hc_mixes): pre
 * p, post q, and comb [j][k] after the Sinkhorn. */
__device__ void ds_hc_coef(const float *raw, const float *sc, const float *base, float eps, int iters,
                           float *p, float *q, float *comb)
{
    for (int k = 0; k < DS_HC; ++k) {
        if (p) {
            p[k] = ds_sigmoid(raw[k] * sc[0] + base[k]) + eps;
        }
        if (q) {
            q[k] = 2.f * ds_sigmoid(raw[DS_HC + k] * sc[1] + base[DS_HC + k]);
        }
    }
    if (!comb) {
        return;
    }
    float c[DS_HC][DS_HC];
    for (int j = 0; j < DS_HC; ++j) {
        float m = -INFINITY;
        for (int k = 0; k < DS_HC; ++k) {
            c[j][k] = raw[2 * DS_HC + DS_HC * j + k] * sc[2] + base[2 * DS_HC + DS_HC * j + k];
            m = fmaxf(m, c[j][k]);
        }
        float s = 0.f;
        for (int k = 0; k < DS_HC; ++k) {
            c[j][k] = expf(c[j][k] - m);
            s += c[j][k];
        }
        for (int k = 0; k < DS_HC; ++k) {
            c[j][k] = c[j][k] / s + eps;
        }
    }
    for (int it = 0; it < iters; ++it) {
        if (it > 0) {
            for (int j = 0; j < DS_HC; ++j) {
                float s = 0.f;
                for (int k = 0; k < DS_HC; ++k) {
                    s += c[j][k];
                }
                for (int k = 0; k < DS_HC; ++k) {
                    c[j][k] /= s + eps;
                }
            }
        }
        for (int k = 0; k < DS_HC; ++k) {
            float s = 0.f;
            for (int j = 0; j < DS_HC; ++j) {
                s += c[j][k];
            }
            for (int j = 0; j < DS_HC; ++j) {
                c[j][k] /= s + eps;
            }
        }
    }
    for (int j = 0; j < DS_HC; ++j) {
        for (int k = 0; k < DS_HC; ++k) {
            comb[DS_HC * j + k] = c[j][k];
        }
    }
}

/* GP_DS_HC_MIX: h [t, n] (the streams side by side), fn [24, n], raw [t, 24],
 * t, n, eps. raw = (h . fn_m) / sqrt(mean h^2 + eps). A block for each (m, j). */
__global__ void __launch_bounds__(256) k_ds_hc_mix(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int m = blockIdx.x, j = blockIdx.y, n = DI(4);
    const float *h = DP(const float, 0) + (size_t)j * n;
    const float *f = DP(const float, 1) + (size_t)m * n;
    float ss = 0.f, d = 0.f;
    for (int i = 4 * threadIdx.x; i < n; i += 4 * blockDim.x) {
        float4 a = *(const float4 *)(h + i), b = *(const float4 *)(f + i);
        ss += a.x * a.x + a.y * a.y + a.z * a.z + a.w * a.w;
        d += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
    }
    ss = block_sum(ss);
    d = block_sum(d);
    if (threadIdx.x == 0) {
        DP(float, 2)[(size_t)j * gridDim.x + m] = d * (1.f / sqrtf(ss / (float)n + df(r, e, 5)));
    }
}

/* GP_DS_HC_PRE: h [t, 4, dim], raw [t, 24] (null: stream 0, the first layer),
 * sc [3], base [24], w [dim] (null: no norm), out [t, dim], t, dim, eps,
 * hc_eps. out = the norm of sum_k p_k h_k, times w. A block for each token. */
__global__ void __launch_bounds__(1024) k_ds_hc_pre(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, dim = DI(7);
    __shared__ float p[DS_HC];
    const float *raw = DP(const float, 1);
    if (threadIdx.x == 0) {
        if (raw) {
            ds_hc_coef(raw + (size_t)j * DS_NMIX, DP(const float, 2), DP(const float, 3), df(r, e, 9), 0, p,
                       NULL, NULL);
        } else {
            p[0] = 1.f;
            p[1] = p[2] = p[3] = 0.f;
        }
    }
    __syncthreads();
    const float *h = DP(const float, 0) + (size_t)j * DS_HC * dim;
    const float *w = DP(const float, 4);
    float *out = DP(float, 5) + (size_t)j * dim;
    float ss = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        float x = p[0] * h[c];
        for (int k = 1; k < DS_HC; ++k) {
            x += p[k] * h[(size_t)k * dim + c];
        }
        out[c] = x;
        ss += x * x;
    }
    if (!w) {
        return;                 /* no weight: the mix alone (the input of DSpark's confidence) */
    }
    ss = block_sum(ss);
    float s = 1.f / sqrtf(ss / (float)dim + df(r, e, 8));
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        out[c] = out[c] * s * w[c];
    }
}

/* GP_DS_HC_POST: y [t, dim], y2 [t, dim] (or null: added to y), res [t, 4,
 * dim], raw [t, 24], sc, base, out [t, 4, dim] (may be res), t, dim, hc_eps,
 * iters. out_k = q_k y + sum_j comb_jk res_j. */
__global__ void __launch_bounds__(256) k_ds_hc_post(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.y, dim = DI(8), c = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float q[DS_HC], comb[DS_HC * DS_HC];
    if (threadIdx.x == 0) {
        ds_hc_coef(DP(const float, 3) + (size_t)j * DS_NMIX, DP(const float, 4), DP(const float, 5),
                   df(r, e, 9), DI(10), NULL, q, comb);
    }
    __syncthreads();
    if (c >= dim) {
        return;
    }
    float y = DP(const float, 0)[(size_t)j * dim + c];
    const float *y2 = DP(const float, 1);
    if (y2) {
        y += y2[(size_t)j * dim + c];
    }
    const float *res = DP(const float, 2) + (size_t)j * DS_HC * dim + c;
    float rv[DS_HC];
    for (int k = 0; k < DS_HC; ++k) {
        rv[k] = res[(size_t)k * dim];
    }
    float *out = DP(float, 6) + (size_t)j * DS_HC * dim + c;
    for (int k = 0; k < DS_HC; ++k) {
        float o = q[k] * y;
        for (int i = 0; i < DS_HC; ++i) {
            o += comb[DS_HC * i + k] * rv[i];
        }
        out[(size_t)k * dim] = o;
    }
}

/* GP_DS_ROPE: x [t, nh, hd] in place, cos, sin [positions, rd / 2], pos, t, nh,
 * hd, rd, inverse, delta: the last rd values of each head (adjacent pairs)
 * turned by the angle of position pos + delta + j (inverse: back). A block
 * of rd / 2 threads for each (head, token). */
__global__ void k_ds_rope(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hh = blockIdx.x, j = blockIdx.y, nh = DI(5), hd = DI(6), rd = DI(7), i = threadIdx.x;
    int64_t p = di(r, e, 3) + DI(9) + j;
    float c = DP(const float, 1)[p * (rd / 2) + i], s = DP(const float, 2)[p * (rd / 2) + i];
    if (DI(8)) {
        s = -s;
    }
    float *x = DP(float, 0) + ((size_t)j * nh + hh) * hd + hd - rd + 2 * i;
    float a = x[0], b = x[1];
    x[0] = ds_rot(a, b, c, s, 0);
    x[1] = ds_rot(a, b, c, s, 1);
}

/* GP_DS_FP4: x [n] in place, n, bs (16: E4M3 scales, 32: E8M0 scales). */
__global__ void __launch_bounds__(256) k_ds_fp4(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t n = di(r, e, 1), i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int bs = DI(2);
    float *x = DP(float, 0);
    float v = i < n ? x[i] : 0.f;       /* n is a multiple of bs: a block is live or not */
    v = ds_fp4(v, bs, bs == 16);
    if (i < n) {
        x[i] = v;
    }
}

/* GP_DS_WIN_WRITE: kv [t, hd] (after kv_norm), cos, sin, kvq [t, hd], pos, t,
 * win, hd, rd: the last rd values turned (position pos + j), then act_quant,
 * into row j of kvq (the window rows of the group; GP_DS_RING_PUT puts them
 * in the ring after the attention). A block of hd threads for each token. */
__global__ void __launch_bounds__(512) k_ds_win_write(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, hd = DI(7), rd = DI(8), c = threadIdx.x;
    int64_t p = di(r, e, 4) + j;
    const float *kv = DP(const float, 0) + (size_t)j * hd;
    float v = kv[c];
    int rc = c - (hd - rd);
    if (rc >= 0) {
        int i = rc / 2;
        float cs = DP(const float, 1)[p * (rd / 2) + i], sn = DP(const float, 2)[p * (rd / 2) + i];
        v = ds_rot(kv[c & ~1], kv[c | 1], cs, sn, c & 1);
    }
    DP(float, 3)[(size_t)j * hd + c] = ds_act_e4m3(v);
}

/* GP_DS_RING_PUT: kvq [t, hd], ring [win, hd], pos, t, win, hd, backup [t,
 * hd] (or null), form: the last min(t, win) rows into the slots (pos + j) %
 * win; with backup (a verify group of DSpark), the rows they replace first
 * (in the form of the ring). */
__global__ void __launch_bounds__(512) k_ds_ring_put(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, t = DI(3), win = DI(4), hd = DI(5), form = ds_cf(ds_form(r, e, 7), DS_CW), c = threadIdx.x;
    if (j < t - win) {
        return;
    }
    int64_t p = di(r, e, 2) + j;
    void *ring = DP(void, 1), *bk = DP(void, 6);
    size_t slot = (size_t)(p % win);
    if (bk) {
        ds_rcopy(bk, (size_t)j, ring, slot, c, hd, form);
    }
    float v = DP(const float, 0)[(size_t)j * hd + c];
    float q = v;
    uint32_t sb = 0;
    if (form >= 2) {
        /* the act_quant rows: a power of two for each 32 (a warp) that keeps
         * them in E4M3 (exact: a power of two at most the reference's) */
        float amax = fabsf(v);
        for (int o = 16; o > 0; o >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(RC_FULL_MASK, amax, o));
        }
        float sc = amax > 0.f ? ds_pow2_ceil(amax / 448.f) : 1.f;
        q = v / sc;
        sb = (__float_as_uint(sc) >> 23) & 255;
    }
    ds_rst(ring, slot, c, hd, form, v, q, sb);
}

/* GP_DS_RING_RESTORE: ring, backup [t, hd], pos, t, keep, win, hd, form: the rows
 * of the tokens from keep on (the drafts that the verify group refused)
 * back to what GP_DS_RING_PUT replaced. */
__global__ void __launch_bounds__(512) k_ds_ring_restore(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, t = DI(3), win = DI(5), hd = DI(6);
    if (j < DI(4) || j < t - win) {
        return;
    }
    int64_t p = di(r, e, 2) + j;
    ds_rcopy(DP(void, 0), (size_t)(p % win), DP(const void, 1), (size_t)j, threadIdx.x, hd,
             ds_cf(ds_form(r, e, 7), DS_CW));
}

/* GP_DS_STREAM_MEAN: h [t, hc, dim], out [t, stride], t, dim, stride, off,
 * hc: out[j, off + c] = the mean of the streams (the input of DSpark from its
 * target layers, model.py forward). */
__global__ void __launch_bounds__(256) k_ds_stream_mean(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.y, dim = DI(3), hc = DI(6), c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= dim) {
        return;
    }
    const float *h = DP(const float, 0) + (size_t)j * hc * dim + c;
    float s = 0.f;
    for (int k = 0; k < hc; ++k) {
        s += h[(size_t)k * dim];
    }
    DP(float, 1)[(size_t)j * DI(4) + DI(5) + c] = s / (float)hc;
}

/* The first position at or after pos that ends a group of ratio tokens. */
__device__ __forceinline__ int64_t ds_first_end(int64_t pos, int ratio)
{
    return pos + (ratio - 1 - pos % ratio);
}

/* GP_DS_COMPRESS: kv [t, hd] (wkv x), score [t, hd] (wgate x; null for
 * ratio 1), kstate, sstate [ratio, hd], w [hd] (the norm), latent [nmax, hd],
 * pos, ratio, hd, eps, t, nmax. Ratio 1: latent row j is the norm of kv row
 * j. Else latent row g is the group of ratio tokens that ends at the g-th end
 * inside the chunk (its rows before pos from the state): the norm of the
 * softmax-weighted sum of its rows (model.py compressor); a row past the
 * ends of the chunk is left. Then (k_ds_compress_state) the tokens after the
 * last end go into the state. A block of hd threads for each latent row. */
__global__ void __launch_bounds__(512) k_ds_compress(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ratio = DI(7), hd = DI(8), t = DI(10), g = blockIdx.x, c = threadIdx.x;
    int64_t pos = di(r, e, 6);
    const float *kv = DP(const float, 0);
    float v;
    if (ratio == 1) {
        if (g >= t) {
            return;
        }
        v = kv[(size_t)g * hd + c];
    } else {
        int64_t pe = ds_first_end(pos, ratio) + (int64_t)g * ratio;
        if (pe >= pos + t) {
            return;
        }
        const float *sc = DP(const float, 1), *ks = DP(const float, 2), *ss = DP(const float, 3);
        float kr[8], sr[8], m = -INFINITY;
        for (int i = 0; i < ratio; ++i) {
            int64_t q = pe - ratio + 1 + i;
            kr[i] = q < pos ? ks[(size_t)(q % ratio) * hd + c] : kv[(size_t)(q - pos) * hd + c];
            sr[i] = q < pos ? ss[(size_t)(q % ratio) * hd + c] : sc[(size_t)(q - pos) * hd + c];
            m = fmaxf(m, sr[i]);
        }
        float den = 0.f;
        for (int i = 0; i < ratio; ++i) {
            den += expf(sr[i] - m);
        }
        v = 0.f;
        for (int i = 0; i < ratio; ++i) {
            v += kr[i] * (expf(sr[i] - m) / den);
        }
    }
    float s = block_sum(v * v);
    s = 1.f / sqrtf(s / (float)hd + df(r, e, 9));
    DP(float, 5)[(size_t)g * hd + c] = v * s * DP(const float, 4)[c];
}

/* Operands 12, 13 (or null): the state before this group, kept for
 * GP_DS_COMPRESS_FIX (a verify group of DSpark). */
__global__ void __launch_bounds__(512) k_ds_compress_state(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ratio = DI(7), hd = DI(8), t = DI(10), c = threadIdx.x;
    int64_t pos = di(r, e, 6), end = pos + t;
    float *bk = DP(float, 12), *bs = DP(float, 13);
    if (bk) {
        for (int i = 0; i < ratio; ++i) {
            bk[(size_t)i * hd + c] = DP(const float, 2)[(size_t)i * hd + c];
            bs[(size_t)i * hd + c] = DP(const float, 3)[(size_t)i * hd + c];
        }
    }
    int64_t last = end - end % ratio - 1;      /* the last end before pos + t */
    for (int64_t q = (last + 1 > pos ? last + 1 : pos); q < end; ++q) {
        DP(float, 2)[(size_t)(q % ratio) * hd + c] = DP(const float, 0)[(size_t)(q - pos) * hd + c];
        DP(float, 3)[(size_t)(q % ratio) * hd + c] = DP(const float, 1)[(size_t)(q - pos) * hd + c];
    }
}

/* GP_DS_COMPRESS_FIX: kstate, sstate, the state before the group (bk, bs),
 * kv [t, hd], score [t, hd] (of the group), pos, keep, ratio, hd: the state
 * as after the first keep tokens of the group (the rows of the open group at
 * pos + keep: from the group, or from before it). A block of hd threads. */
__global__ void __launch_bounds__(512) k_ds_compress_fix(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ratio = DI(8), hd = DI(9), c = threadIdx.x;
    int64_t pos = di(r, e, 6), P = pos + DI(7);
    for (int64_t q = P - P % ratio; q < P; ++q) {
        int sl = (int)(q % ratio);
        bool in = q >= pos;
        DP(float, 0)[(size_t)sl * hd + c] = in ? DP(const float, 4)[(size_t)(q - pos) * hd + c]
                                                : DP(const float, 2)[(size_t)sl * hd + c];
        DP(float, 1)[(size_t)sl * hd + c] = in ? DP(const float, 5)[(size_t)(q - pos) * hd + c]
                                                : DP(const float, 3)[(size_t)sl * hd + c];
    }
}

/* GP_DS_CMP_WRITE: latent [nmax, hd], k [nmax, ixd] (wk latent), knorm
 * [ixd], cos, sin (the compressed positions), ckv [*, hd], kcache [*, ixd],
 * pos, ratio, hd, ixd, rd, eps, t, nmax, form. For each latent row of the chunk
 * (the group that ends at pe: entry pe / ratio, position pe + 1 - ratio): the
 * latent turned and rounded to FP4 (E4M3 scales of 16) into ckv; the key
 * normed, turned, rounded to FP4 (E8M0 scales of 32) into kcache. A block of
 * hd threads for each latent row. */
__global__ void __launch_bounds__(512) k_ds_cmp_write(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ratio = DI(8), hd = DI(9), ixd = DI(10), rd = DI(11), t = DI(13), g = blockIdx.x, c = threadIdx.x;
    int64_t pos = di(r, e, 7);
    int64_t pe = ds_first_end(pos, ratio) + (int64_t)g * ratio;
    if (pe >= pos + t) {
        return;
    }
    int64_t slot = pe / ratio, p = pe + 1 - ratio;
    const float *cs = DP(const float, 3) + p * (rd / 2), *sn = DP(const float, 4) + p * (rd / 2);
    __shared__ float lat[512], kn[512];
    lat[c] = DP(const float, 0)[(size_t)g * hd + c];
    float kv = c < ixd ? DP(const float, 1)[(size_t)g * ixd + c] : 0.f;
    float ss = block_sum(kv * kv);
    if (c < ixd) {
        kn[c] = kv * (1.f / sqrtf(ss / (float)ixd + df(r, e, 12))) * DP(const float, 2)[c];
    }
    __syncthreads();
    float v = lat[c];
    int rc = c - (hd - rd);
    if (rc >= 0) {
        v = ds_rot(lat[c & ~1], lat[c | 1], cs[rc / 2], sn[rc / 2], c & 1);
    }
    int form = ds_form(r, e, 15);
    float y, sc;
    ds_fp4_parts(v, 16, true, &y, &sc);
    ds_rst(DP(void, 5), slot, c, hd, ds_cf(form, DS_CC), __fmul_rn(y, sc), y, ds_e4m3_code(sc));
    if (c < ixd) {              /* whole warps: ixd is a multiple of 32 */
        v = kn[c];
        rc = c - (ixd - rd);
        if (rc >= 0) {
            v = ds_rot(kn[c & ~1], kn[c | 1], cs[rc / 2], sn[rc / 2], c & 1);
        }
        ds_fp4_parts(v, 32, false, &y, &sc);
        ds_rst(DP(void, 6), slot, c, ixd, ds_cf(form, DS_CK), __fmul_rn(y, sc), y, (__float_as_uint(sc) >> 23) & 255);
    }
}

/* GP_DS_IDX_SCORE: q [t, H, D] (turned, FP4), w [t, H] (weights_proj x),
 * wscale, kcache [maxn, D], pos, ratio, mask [t, mstride] (bytes for each
 * block of mbs entries, or null), mbs, out [t, maxn], H, D, maxn, t, mstride, form,
 * c0. For query j and the entries u < (pos + j + 1) / ratio: out_ju = sum_h
 * relu(q_jh . k_u) w_jh wscale, -inf where the mask is 0. A warp for each
 * entry, a lane for each head (H = 32, D <= 128); blockIdx.y the query. A
 * chunk (DS_TOPK_MERGE): the entries c0 .. c0 + maxn alone, entry u at
 * out_j(u - c0) (c0 0: all of them). */
__global__ void __launch_bounds__(256) k_ds_idx_score(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int D = DI(10), j = blockIdx.y;
    int64_t maxn = di(r, e, 11), n = (di(r, e, 4) + j + 1) / DI(5), c0 = di(r, e, 15);
    if (n > c0 + maxn) {
        n = c0 + maxn;
    }
    __shared__ float qs[32][129];
    __shared__ __align__(16) float krow[8][128];
    const float *q = DP(const float, 0) + (size_t)j * 32 * D;
    for (int i = threadIdx.x; i < 32 * D; i += blockDim.x) {
        qs[i / D][i % D] = q[i];
    }
    __syncthreads();
    int lane = threadIdx.x % 32;
    float wh = DP(const float, 1)[(size_t)j * 32 + lane] * df(r, e, 2);
    const void *kc = DP(const void, 3);
    int form = ds_cf(ds_form(r, e, 14), DS_CK);
    const uint8_t *mask = DP(const uint8_t, 6);
    if (mask) {
        mask += (size_t)j * di(r, e, 13);
    }
    int mbs = DI(7);
    float *out = DP(float, 8) + (size_t)j * maxn;
    int64_t nw = (int64_t)gridDim.x * (blockDim.x / 32);
    for (int64_t u = c0 + (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32; u < n; u += nw) {
        if (mask && !mask[u / mbs]) {
            if (lane == 0) {
                out[u - c0] = -INFINITY;
            }
            continue;
        }
        /* the key row decoded once by the warp (4 values a lane), then read
         * by every lane (a head each) */
        float *kr = krow[threadIdx.x / 32];
        if (4 * lane < D) {
            *(float4 *)(kr + 4 * lane) = ds_rld4(kc, (size_t)u, 4 * lane, D, form);
        }
        __syncwarp();
        float d = 0.f;
        for (int i = 0; i < D; i += 4) {
            float4 kv = *(const float4 *)(kr + i);
            d += qs[lane][i] * kv.x + qs[lane][i + 1] * kv.y + qs[lane][i + 2] * kv.z + qs[lane][i + 3] * kv.w;
        }
        __syncwarp();
        d = fmaxf(d, 0.f) * wh;
        for (int o = 16; o > 0; o >>= 1) {
            d += __shfl_xor_sync(RC_FULL_MASK, d, o);
        }
        if (lane == 0) {
            out[u - c0] = d;
        }
    }
}

/* GP_DS_CAND: scores [t, maxn], pos, ratio, bs, out [t, nbs], nbs, t, maxn, c0:
 * for query j, the max of each block of bs of its entries (the entries past
 * (pos + j + 1) / ratio: -inf), the last block +inf (model.py candidates). A
 * chunk (DS_TOPK_MERGE): the scores of the entries c0 .. c0 + maxn (c0 a
 * multiple of bs), their blocks from c0 / bs on at out_j0. */
__global__ void __launch_bounds__(256) k_ds_cand(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.y, bs = DI(3);
    int64_t n = (di(r, e, 1) + j + 1) / DI(2), nb = (n + bs - 1) / bs, c0 = di(r, e, 8), b0 = c0 / bs;
    int64_t b1 = nb < b0 + di(r, e, 5) ? nb : b0 + di(r, e, 5);
    const float *s = DP(const float, 0) + (size_t)j * di(r, e, 7);
    float *out = DP(float, 4) + (size_t)j * di(r, e, 5);
    for (int64_t b = b0 + (int64_t)blockIdx.x * blockDim.x + threadIdx.x; b < b1;
         b += (int64_t)gridDim.x * blockDim.x) {
        float m = -INFINITY;
        for (int64_t u = b * bs; u < (b + 1) * bs && u < n; ++u) {
            m = fmaxf(m, s[u - c0]);
        }
        out[b - b0] = b == nb - 1 ? INFINITY : m;
    }
}

__device__ __forceinline__ uint32_t ds_key(float v)
{
    uint32_t u = __float_as_uint(v);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/* The exclusive prefix sum of v over a block of 1024 threads; *total gets the
 * sum. All threads call it. */
__device__ int ds_scan1024(int v, int *total)
{
    __shared__ int ws[32];
    int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(RC_FULL_MASK, x, o);
        if (lane >= o) {
            x += y;
        }
    }
    __syncthreads();
    if (lane == 31) {
        ws[w] = x;
    }
    __syncthreads();
    if (w == 0) {
        int s = ws[lane];
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(RC_FULL_MASK, s, o);
            if (lane >= o) {
                s += y;
            }
        }
        ws[lane] = s;
    }
    __syncthreads();
    int before = (w > 0 ? ws[w - 1] : 0) + x - v;
    *total = ws[31];
    return before;
}

/* GP_DS_TOPK: scores [t, stride], pos, ratio, bs, k, sel [t, k] (int32), cnt
 * [t] (int32), mask [t, stride] (bytes, or null), stride, t. For query j,
 * over its n = ceil(((pos + j + 1) / ratio) / bs) scores: the min(k, n)
 * largest, ties to the lower index (a stable sort of model.py), as indices
 * in ascending order; the mask gets 1 for a chosen score above -inf, else 0.
 * A block of 1024 threads for each query: a radix select of the k-th key (4
 * passes of 8 bits), then the indices in order. */
__global__ void __launch_bounds__(1024) k_ds_topk(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int bs = DI(3), k = DI(4), j = blockIdx.x;
    int64_t stride = di(r, e, 8);
    int64_t n = ((di(r, e, 1) + j + 1) / DI(2) + bs - 1) / bs;
    const float *s = DP(const float, 0) + (size_t)j * stride;
    int *sel = DP(int, 5) + (size_t)j * k;
    uint8_t *mask = DP(uint8_t, 7);
    if (mask) {
        mask += (size_t)j * stride;
    }
    int kk = (int)(n < k ? n : k);
    if (threadIdx.x == 0) {
        DP(int, 6)[j] = kk;
    }
    if (kk == 0) {
        return;
    }
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s, need_s;
    uint32_t prefix = 0, need = (uint32_t)kk;       /* the key's bits above the pass; the rank left */
    for (int pass = 0; pass < 4; ++pass) {
        int shift = 24 - 8 * pass;
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            hist[i] = 0;
        }
        __syncthreads();
        uint32_t hmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t u = threadIdx.x; u < n; u += blockDim.x) {
            uint32_t key = ds_key(s[u]);
            if ((key & hmask) == prefix) {
                atomicAdd(&hist[(key >> shift) & 255], 1);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            uint32_t acc = 0;
            int d = 255;
            for (; d > 0; --d) {
                if (acc + (uint32_t)hist[d] >= need) {
                    break;
                }
                acc += (uint32_t)hist[d];
            }
            prefix_s = prefix | ((uint32_t)d << shift);
            need_s = need - acc;
        }
        __syncthreads();
        prefix = prefix_s;
        need = need_s;
        __syncthreads();
    }
    /* prefix: the key of the kk-th score; need: how many of the scores of
     * that key are taken (the first ones) */
    int out_base = 0, eq_base = 0;
    for (int64_t c0 = 0; c0 < n; c0 += blockDim.x) {
        int64_t u = c0 + threadIdx.x;
        uint32_t key = u < n ? ds_key(s[u]) : 0u;
        int gt = u < n && key > prefix, eq = u < n && key == prefix;
        int etot, otot;
        int er = ds_scan1024(eq, &etot);
        int take = gt || (eq && (uint32_t)(eq_base + er) < need);
        int at = ds_scan1024(take, &otot);
        if (take) {
            sel[out_base + at] = (int)u;
        }
        if (mask && u < n) {
            mask[u] = take && s[u] > -INFINITY;
        }
        out_base += otot;
        eq_base += etot;
    }
}

/* GP_DS_TOPK_MERGE: chunk [t, cs] (the scores of the units c0 .. c0 + cs),
 * pos, ratio, bs, k, sel [t, k] (int32), cnt [t] (int32), vals [t, k], cs, t,
 * c0, first, mask [t, mstride] (bytes, or null), mstride, last. The top k of
 * DS_TOPK over a row in chunks (DS_TOPK needs all the scores of the row at
 * once: [t, n]): for query j (n = ceil(((pos + j + 1) / ratio) / bs) units,
 * as DS_TOPK), the k best of its k best so far (sel, vals, cnt; none when
 * first) and the units of the chunk below n, ties to the lower index. The
 * ones so far are all below the chunk, so the choice is DS_TOPK's over the
 * units up to the end of the chunk; in ascending order of index. last: the
 * mask of DS_TOPK (1 for a chosen score above -inf, else 0). The radix
 * select of k_ds_topk over the row (the ones so far, then the chunk). */
#define DS_TOPK_MAXK 2048
__global__ void __launch_bounds__(1024) k_ds_topk_merge(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int bs = DI(3), k = DI(4), j = blockIdx.x;
    int64_t cs = di(r, e, 8), c0 = di(r, e, 10);
    int64_t n = ((di(r, e, 1) + j + 1) / DI(2) + bs - 1) / bs;
    int64_t nc = n - c0 < 0 ? 0 : (n - c0 < cs ? n - c0 : cs);
    const float *s = DP(const float, 0) + (size_t)j * cs;
    int *sel = DP(int, 5) + (size_t)j * k, *cnt = DP(int, 6) + j;
    float *vals = DP(float, 7) + (size_t)j * k;
    uint8_t *mask = DI(14) ? DP(uint8_t, 12) : NULL;
    if (mask) {
        mask += (size_t)j * di(r, e, 13);
    }
    __shared__ float rv[DS_TOPK_MAXK];
    __shared__ int ri[DS_TOPK_MAXK];
    __shared__ int hist[256];
    __shared__ uint32_t prefix_s, need_s;
    int had = DI(11) ? 0 : *cnt;
    for (int i = threadIdx.x; i < had; i += blockDim.x) {
        rv[i] = vals[i];
        ri[i] = sel[i];
    }
    __syncthreads();
    int64_t m = had + nc;
    int kk = (int)(m < k ? m : k);
#define DS_MV(u) ((u) < had ? rv[(u)] : s[(u) - had])
    if (kk > 0) {
        uint32_t prefix = 0, need = (uint32_t)kk;
        for (int pass = 0; pass < 4; ++pass) {
            int shift = 24 - 8 * pass;
            for (int i = threadIdx.x; i < 256; i += blockDim.x) {
                hist[i] = 0;
            }
            __syncthreads();
            uint32_t hmask = pass == 0 ? 0u : (0xffffffffu << (shift + 8));
            for (int64_t u = threadIdx.x; u < m; u += blockDim.x) {
                uint32_t key = ds_key(DS_MV(u));
                if ((key & hmask) == prefix) {
                    atomicAdd(&hist[(key >> shift) & 255], 1);
                }
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                uint32_t acc = 0;
                int d = 255;
                for (; d > 0; --d) {
                    if (acc + (uint32_t)hist[d] >= need) {
                        break;
                    }
                    acc += (uint32_t)hist[d];
                }
                prefix_s = prefix | ((uint32_t)d << shift);
                need_s = need - acc;
            }
            __syncthreads();
            prefix = prefix_s;
            need = need_s;
            __syncthreads();
        }
        int out_base = 0, eq_base = 0;
        for (int64_t q0 = 0; q0 < m; q0 += blockDim.x) {
            int64_t u = q0 + threadIdx.x;
            float v = u < m ? DS_MV(u) : 0.f;
            uint32_t key = u < m ? ds_key(v) : 0u;
            int gt = u < m && key > prefix, eq = u < m && key == prefix;
            int etot, otot;
            int er = ds_scan1024(eq, &etot);
            int take = gt || (eq && (uint32_t)(eq_base + er) < need);
            int at = ds_scan1024(take, &otot);
            if (take) {
                sel[out_base + at] = u < had ? ri[u] : (int)(c0 + u - had);
                vals[out_base + at] = v;
            }
            out_base += otot;
            eq_base += etot;
        }
    }
#undef DS_MV
    if (threadIdx.x == 0) {
        *cnt = kk;
    }
    if (mask) {
        __syncthreads();
        for (int64_t u = threadIdx.x; u < n; u += blockDim.x) {
            mask[u] = 0;
        }
        __syncthreads();
        for (int i = threadIdx.x; i < kk; i += blockDim.x) {
            if (vals[i] > -INFINITY) {
                mask[sel[i]] = 1;
            }
        }
    }
}

/* GP_DS_ATTN: q [t, H, hd], ring [win, hd], pos, win, ckv [*, hd] (or null),
 * sel [t, maxk] (int32), cnt [t] (int32), sink [H], out [t, H, hd], H, hd,
 * scale, maxk, kvq [t, hd], t, partials (or null), mode (1: DSpark; 2: the
 * window from operand 18 on, bounded replay), the form
 * of the caches (1 bfloat16, 2 the smallest: ds_cf). Query j (position p = pos + j) over its window
 * (the positions p - win + 1 .. p: before pos from the ring, from pos on the
 * rows of the group in kvq) and its selected compressed entries: softmax
 * with a sink for each head (model.py sparse_attn); the keys are the values.
 * A block for each DS_AHB heads of a query. */
#define DS_AHB 4
#define DS_AMAX (128 + 2048)
/* Key m of a query: a row of the ring (form) or of the compressed cache (its
 * form), or of the group (kvq, float32): its base, row and form. */
struct ds_krow {
    const void *p;
    size_t row;
    int form;
};

__device__ __forceinline__ ds_krow ds_akey(int m, int nr, int ng, int64_t lo, int g0, int win, const void *ring,
                                          const float *kvq, const void *ckv, const int *sel, int form)
{
    if (m < nr) {
        return {ring, (size_t)((lo + m) % win), ds_cf(form, DS_CW)};
    }
    if (m < nr + ng) {
        return {kvq, (size_t)(g0 + m - nr), 0};
    }
    return {ckv, (size_t)sel[m - nr - ng], ds_cf(form, DS_CC)};
}

/* Operand 15 (or null): the partials of a split over the keys (a step:
 * blockIdx.z of DS_ASPLIT ranges of the keys), [t, H, DS_ASPLIT, hd + 2]
 * (the max, the sum of the exps, the output not yet divided); then
 * k_ds_attn_join adds them with the sink. */
#define DS_ASPLIT 8
__global__ void __launch_bounds__(256) k_ds_attn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int win = DI(3), hd = DI(10), maxk = DI(12), h0 = blockIdx.x * DS_AHB, j = blockIdx.y, H = DI(9);
    int64_t pos = di(r, e, 2), p = pos + j;
    int64_t lo = p - win + 1 > 0 ? p - win + 1 : 0;
    int nr = lo < pos ? (int)(pos - lo) : 0;                  /* from the ring */
    int g0 = lo > pos ? (int)(lo - pos) : 0, ng = j - g0 + 1;  /* from the group */
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
        /* a DSpark block (model.py DSparkAttention): every query over the
         * last min(win, pos) positions in the ring and all the rows of the
         * block */
        nr = (int)(pos < win ? pos : win);
        lo = pos - nr;
        g0 = 0;
        ng = DI(14);
    }
    const void *ckv = DP(const void, 4);
    int form = ds_form(r, e, 17);
    const int *sel = DP(const int, 5) + (size_t)j * maxk;
    int nsel = ckv ? DP(const int, 6)[j] : 0;
    int nk = nr + ng + nsel;
    float *part = DP(float, 15);
    int ns = part ? gridDim.z : 1, z = blockIdx.z;
    int chunk = (nk + ns - 1) / ns, m0 = z * chunk, m1 = m0 + chunk < nk ? m0 + chunk : nk;
    const void *ring = DP(const void, 1);
    const float *kvq = DP(const float, 13);
    float scale = df(r, e, 11);
    __shared__ float qs[DS_AHB][512];
    __shared__ float sc[DS_AHB][DS_AMAX];
    __shared__ float red[DS_AHB][2];
    const float *q = DP(const float, 0) + ((size_t)j * H + h0) * hd;
    for (int i = threadIdx.x; i < DS_AHB * hd; i += blockDim.x) {
        qs[i / hd][i % hd] = q[i];
    }
    __syncthreads();
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarp = blockDim.x / 32;
    for (int m = m0 + warp; m < m1; m += nwarp) {
        ds_krow kr = ds_akey(m, nr, ng, lo, g0, win, ring, kvq, ckv, sel, form);
        float d[DS_AHB] = {0.f};
        for (int i = 4 * lane; i < hd; i += 128) {
            float4 kv = ds_rld4(kr.p, kr.row, i, hd, kr.form);
            for (int a = 0; a < DS_AHB; ++a) {
                d[a] += qs[a][i] * kv.x + qs[a][i + 1] * kv.y + qs[a][i + 2] * kv.z + qs[a][i + 3] * kv.w;
            }
        }
        for (int a = 0; a < DS_AHB; ++a) {
            for (int o = 16; o > 0; o >>= 1) {
                d[a] += __shfl_xor_sync(RC_FULL_MASK, d[a], o);
            }
            if (lane == 0) {
                sc[a][m - m0] = d[a] * scale;
            }
        }
    }
    __syncthreads();
    /* a warp for each head: the max, the exps, the sum (with the sink when
     * the keys are not split) */
    if (warp < DS_AHB) {
        float mx = -INFINITY;
        for (int m = lane; m < m1 - m0; m += 32) {
            mx = fmaxf(mx, sc[warp][m]);
        }
        for (int o = 16; o > 0; o >>= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, o));
        }
        float sum = 0.f;
        for (int m = lane; m < m1 - m0; m += 32) {
            float pv = expf(sc[warp][m] - mx);
            sc[warp][m] = pv;
            sum += pv;
        }
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
        }
        if (lane == 0) {
            red[warp][0] = part ? sum : sum + expf(DP(const float, 7)[h0 + warp] - mx);
            red[warp][1] = mx;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o[DS_AHB] = {0.f};
        for (int m = m0; m < m1; ++m) {
            ds_krow kr = ds_akey(m, nr, ng, lo, g0, win, ring, kvq, ckv, sel, form);
            float kv = ds_rld(kr.p, kr.row, i, hd, kr.form);
            for (int a = 0; a < DS_AHB; ++a) {
                o[a] += sc[a][m - m0] * kv;
            }
        }
        for (int a = 0; a < DS_AHB; ++a) {
            if (part) {
                float *pp = part + (((size_t)j * H + h0 + a) * ns + z) * (hd + 2);
                pp[2 + i] = o[a];
                if (i == 0) {
                    pp[0] = red[a][1];
                    pp[1] = red[a][0];
                }
            } else {
                DP(float, 8)[((size_t)j * H + h0 + a) * hd + i] = o[a] / red[a][0];
            }
        }
    }
}

/* The keys of query j of GP_DS_ATTN (the window, the group's rows, the
 * selected compressed entries): lo, the rows from the ring (nr), the first
 * row of the group (g0) and its rows (ng); the count of the selected
 * entries (nsel). */
struct ds_aspan {
    int64_t lo;
    int nr, g0, ng, nsel;
};

__device__ __forceinline__ ds_aspan ds_attn_span(const gp_rec *r, const int64_t *e, int j)
{
    int win = DI(3);
    int64_t pos = di(r, e, 2), p = pos + j;
    ds_aspan sp;
    sp.lo = p - win + 1 > 0 ? p - win + 1 : 0;
    sp.nr = sp.lo < pos ? (int)(pos - sp.lo) : 0;
    sp.g0 = sp.lo > pos ? (int)(sp.lo - pos) : 0;
    sp.ng = j - sp.g0 + 1;
    int mode = r->tag[16] != GP_T_NONE ? DI(16) : 0;
    if (mode & 2) {
        int64_t w0 = di(r, e, 18);
        sp.lo = sp.lo > w0 ? sp.lo : w0;
        sp.nr = sp.lo < pos ? (int)(pos - sp.lo) : 0;
        sp.g0 = sp.lo > pos ? (int)(sp.lo - pos) : 0;
        sp.ng = j - sp.g0 + 1;
    }
    if (mode & 1) {
        sp.nr = (int)(pos < win ? pos : win);
        sp.lo = pos - sp.nr;
        sp.g0 = 0;
        sp.ng = DI(14);
    }
    sp.nsel = DP(const void, 4) ? DP(const int, 6)[j] : 0;
    return sp;
}

/* GP_DS_ATTN of a step or a verify group (operand 15: the keys split,
 * operand 19: the keys buffer), first k_ds_attn_keys: the keys of each query
 * decoded once (ds_rld4, every form) into float32 rows of operand 19 [t,
 * win + maxk, hd], a thread for each 4 values (k_ds_attn decoded every key
 * in each of the 16 blocks of a query, twice: 60% of its instructions). */
__global__ void __launch_bounds__(256) k_ds_attn_keys(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.y, hd = DI(10), win = DI(3), maxk = DI(12), nq = hd / 4;
    ds_aspan sp = ds_attn_span(r, e, j);
    int nk = sp.nr + sp.ng + sp.nsel;
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)nk * nq) {
        return;
    }
    int m = (int)(idx / nq), c4 = 4 * (int)(idx % nq);
    ds_krow kr = ds_akey(m, sp.nr, sp.ng, sp.lo, sp.g0, win, DP(const void, 1), DP(const float, 13),
                         DP(const void, 4), DP(const int, 5) + (size_t)j * maxk, ds_form(r, e, 17));
    float *out = DP(float, 19) + ((size_t)j * (win + maxk) + m) * hd;
    *(float4 *)(out + c4) = ds_rld4(kr.p, kr.row, c4, hd, kr.form);
}

/* Then k_ds_attn_s: the work and the arithmetic of k_ds_attn (with the keys
 * split) for DS_SHB heads a block, the keys float32 from operand 19 (L2):
 * the scores of key m by warp (m - m0) % 8 over the same columns of each
 * lane, the outputs of a column over m in order (the same values as the
 * decode of k_ds_attn: the same bits). */
#define DS_SHB 4
__global__ void __launch_bounds__(256) k_ds_attn_s(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int win = DI(3), hd = DI(10), maxk = DI(12), h0 = blockIdx.x * DS_SHB, j = blockIdx.y, H = DI(9);
    ds_aspan sp = ds_attn_span(r, e, j);
    int nk = sp.nr + sp.ng + sp.nsel;
    float *part = DP(float, 15);
    int ns = gridDim.z, z = blockIdx.z;
    int chunk = (nk + ns - 1) / ns, m0 = z * chunk, m1 = m0 + chunk < nk ? m0 + chunk : nk;
    const float *keys = DP(const float, 19) + (size_t)j * (win + maxk) * hd;
    float scale = df(r, e, 11);
    __shared__ __align__(16) float qs[DS_SHB][512];
    __shared__ float sc[DS_SHB][(DS_AMAX + DS_ASPLIT - 1) / DS_ASPLIT];
    __shared__ float red[DS_SHB][2];
    const float *q = DP(const float, 0) + ((size_t)j * H + h0) * hd;
    for (int i = threadIdx.x; i < DS_SHB * hd; i += blockDim.x) {
        qs[i / hd][i % hd] = q[i];
    }
    __syncthreads();
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nwarp = blockDim.x / 32;
    for (int m = m0 + warp; m < m1; m += nwarp) {
        const float *kr = keys + (size_t)m * hd;
        float d[DS_SHB];
#pragma unroll
        for (int a = 0; a < DS_SHB; ++a) {
            d[a] = 0.f;
        }
        for (int i = 4 * lane; i < hd; i += 128) {
            float4 kv = *(const float4 *)(kr + i);
#pragma unroll
            for (int a = 0; a < DS_SHB; ++a) {
                d[a] += qs[a][i] * kv.x + qs[a][i + 1] * kv.y + qs[a][i + 2] * kv.z + qs[a][i + 3] * kv.w;
            }
        }
#pragma unroll
        for (int a = 0; a < DS_SHB; ++a) {
            for (int o = 16; o > 0; o >>= 1) {
                d[a] += __shfl_xor_sync(RC_FULL_MASK, d[a], o);
            }
            if (lane == 0) {
                sc[a][m - m0] = d[a] * scale;
            }
        }
    }
    __syncthreads();
    if (warp < DS_SHB) {
        float mx = -INFINITY;
        for (int m = lane; m < m1 - m0; m += 32) {
            mx = fmaxf(mx, sc[warp][m]);
        }
        for (int o = 16; o > 0; o >>= 1) {
            mx = fmaxf(mx, __shfl_xor_sync(RC_FULL_MASK, mx, o));
        }
        float sum = 0.f;
        for (int m = lane; m < m1 - m0; m += 32) {
            float pv = expf(sc[warp][m] - mx);
            sc[warp][m] = pv;
            sum += pv;
        }
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
        }
        if (lane == 0) {
            red[warp][0] = sum;
            red[warp][1] = mx;
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o[DS_SHB];
#pragma unroll
        for (int a = 0; a < DS_SHB; ++a) {
            o[a] = 0.f;
        }
#pragma unroll 8
        for (int m = m0; m < m1; ++m) {         /* (unrolled: the loads ahead, the sums in order) */
            float kv = keys[(size_t)m * hd + i];
#pragma unroll
            for (int a = 0; a < DS_SHB; ++a) {
                o[a] += sc[a][m - m0] * kv;
            }
        }
#pragma unroll
        for (int a = 0; a < DS_SHB; ++a) {
            float *pp = part + (((size_t)j * H + h0 + a) * ns + z) * (hd + 2);
            pp[2 + i] = o[a];
            if (i == 0) {
                pp[0] = red[a][1];
                pp[1] = red[a][0];
            }
        }
    }
}

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

/* The partials of the split of GP_DS_ATTN: out = sum_z o_z e^(mx_z - M) /
 * (sum_z s_z e^(mx_z - M) + e^(sink - M)). A block for each (head, token). */
__global__ void __launch_bounds__(512) k_ds_attn_join(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int hh = blockIdx.x, j = blockIdx.y, H = DI(9), hd = DI(10), ns = DS_ASPLIT;
    const float *pp = DP(const float, 15) + ((size_t)j * H + hh) * ns * (hd + 2);
    float M = -INFINITY;
    for (int z = 0; z < ns; ++z) {
        M = fmaxf(M, pp[(size_t)z * (hd + 2)]);
    }
    float den = expf(DP(const float, 7)[hh] - M);
    for (int z = 0; z < ns; ++z) {
        float mz = pp[(size_t)z * (hd + 2)];
        if (mz > -INFINITY) {
            den += pp[(size_t)z * (hd + 2) + 1] * expf(mz - M);
        }
    }
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float o = 0.f;
        for (int z = 0; z < ns; ++z) {
            float mz = pp[(size_t)z * (hd + 2)];
            if (mz > -INFINITY) {
                o += pp[(size_t)z * (hd + 2) + 2 + i] * expf(mz - M);
            }
        }
        DP(float, 8)[((size_t)j * H + hh) * hd + i] = o / den;
    }
}

/* GP_DS_ROUTER: logits [t, E], bias [E], ids [t, k] (int32), val [t, k], t,
 * E, k, scale, bias_vl [E] or NULL, vis [t] (int32) or NULL. s =
 * sqrt(softplus(logit)); the k best of s + bias (bias_vl for a token of an
 * image span: vis[j] != 0; ties to the lower expert); the weights s / sum s
 * times scale (model.py Gate). A block of 256 threads for each token. */
__global__ void __launch_bounds__(256) k_ds_router(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, E = DI(5), k = DI(6);
    __shared__ float s[512], b[512];
    __shared__ float bv[8];
    __shared__ int bi[8], pick[16];
    const float *lg = DP(const float, 0) + (size_t)j * E, *bias = DP(const float, 1);
    if (DP(const void, 9) && DP(const int, 9)[j]) {
        bias = DP(const float, 8);
    }
    for (int i = threadIdx.x; i < E; i += blockDim.x) {
        float l = lg[i];
        s[i] = sqrtf(fmaxf(l, 0.f) + log1pf(expf(-fabsf(l))));
        b[i] = s[i] + bias[i];
    }
    __syncthreads();
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    for (int c = 0; c < k; ++c) {
        float v = -INFINITY;
        int vi = E;
        for (int i = threadIdx.x; i < E; i += blockDim.x) {
            if (b[i] > v) {
                v = b[i];
                vi = i;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float v2 = __shfl_xor_sync(RC_FULL_MASK, v, o);
            int i2 = __shfl_xor_sync(RC_FULL_MASK, vi, o);
            if (v2 > v || (v2 == v && i2 < vi)) {
                v = v2;
                vi = i2;
            }
        }
        if (lane == 0) {
            bv[warp] = v;
            bi[warp] = vi;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            for (int w = 1; w < (int)(blockDim.x / 32); ++w) {
                if (bv[w] > bv[0] || (bv[w] == bv[0] && bi[w] < bi[0])) {
                    bv[0] = bv[w];
                    bi[0] = bi[w];
                }
            }
            pick[c] = bi[0];
            b[bi[0]] = -INFINITY;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        float sum = 0.f;
        for (int c = 0; c < k; ++c) {
            sum += s[pick[c]];
        }
        sum += 1e-20f;
        for (int c = 0; c < k; ++c) {
            DP(int, 2)[(size_t)j * k + c] = pick[c];
            DP(float, 3)[(size_t)j * k + c] = s[pick[c]] / sum * df(r, e, 7);
        }
    }
}

/* GP_DS_SWIGLU: g, u, out, n, limit: silu(min(g, limit)) clamp(u, -limit, limit). */
__global__ void __launch_bounds__(256) k_ds_swiglu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= di(r, e, 3)) {
        return;
    }
    float lim = df(r, e, 4);
    float g = fminf(DP(const float, 0)[i], lim), u = fminf(fmaxf(DP(const float, 1)[i], -lim), lim);
    DP(float, 2)[i] = g / (1.f + expf(-g)) * u;
}

/* GP_DS_ENGRAM: kv [t, (hc + 1) dim] (the keys of the streams, then the
 * value), h [t, hc, dim] in place, wq [hc, dim] (q_weight k_weight), t, dim,
 * hc, eps. The gate of stream s: sigmoid of the signed sqrt of (h . wq .
 * key) / rms(h) / rms(key) / sqrt(dim); h += gate value (model.py engram). A
 * block for each (stream, token). */
__global__ void __launch_bounds__(1024) k_ds_engram(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int sidx = blockIdx.x, j = blockIdx.y, dim = DI(4), hc = DI(5);
    const float *kv = DP(const float, 0) + (size_t)j * (hc + 1) * dim;
    const float *key = kv + (size_t)sidx * dim, *val = kv + (size_t)hc * dim;
    const float *wq = DP(const float, 2) + (size_t)sidx * dim;
    float *h = DP(float, 1) + ((size_t)j * hc + sidx) * dim;
    float sh = 0.f, sk = 0.f, sd = 0.f;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        float a = h[c], b = key[c];
        sh += a * a;
        sk += b * b;
        sd += a * wq[c] * b;
    }
    sh = block_sum(sh);
    sk = block_sum(sk);
    sd = block_sum(sd);
    float eps = df(r, e, 6);
    float rstd = 1.f / sqrtf(sh / (float)dim + eps) / sqrtf(sk / (float)dim + eps);
    float dot = sd * rstd * rsqrtf((float)dim);
    float g = ds_sigmoid(copysignf(sqrtf(fmaxf(fabsf(dot), 1e-6f)), dot));
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        h[c] += g * val[c];
    }
}

/* ---------- the hot experts (GP_DS_HOT_MOE, GP_DS_COMBINE) ----------
 * The experts of a layer that the GPU holds: map[e] is the slot of expert e
 * in store (or -1), a slot of per bytes: w1 (KQ_MXX, inner x hidden), w3 at
 * off3, w2 (hidden x inner) at off2. The operations are those of the CPU
 * parts (moe_part.c, flag 128): x to int16 (k_quant_x2, the bits of
 * kq_quant_part16), the gate and up rows (mxx_group_x2), the act
 * silu(min(g, 10)) clamp(u, -10, 10) times the weight of the pair (rc_dexp of
 * the CPU), the act to int16, the down rows: the de row of a pair is the
 * CPU's bit for bit, and GP_DS_COMBINE adds the rows of a token in the
 * order of the router as the CPU does, so the output does not depend on
 * which experts the GPU holds.
 *
 * GP_DS_HOT_MOE: x [t, H], idx [t, k], val [t, k], map [E], store, per,
 * off3, off2, t, k, H, I, qh, ql [t, H], xs [t, H / 32], act [t k, 2 I],
 * act2 [t k, I], aqh, aql [t k, I], axs [t k, I / 32], de [t k, H], E (the experts
 * of the layer: the grid of the grouped kernels of a prompt). */
__global__ void __launch_bounds__(256) k_ds_hot_gu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int pq = blockIdx.z, k = DI(9), H = DI(10), I = DI(11);
    int slot = DP(const int, 3)[DP(const int, 1)[pq]];
    if (slot < 0) {
        return;
    }
    int j = pq / k, mat = blockIdx.y, grp = blockIdx.x, nb = H / 32;
    const uint8_t *w = DP(const uint8_t, 4) + (size_t)slot * di(r, e, 5) + (mat ? di(r, e, 6) : 0) +
                       (size_t)grp * nb * MXX_BB;
    mxx_group_x2<8>(w, H, DP(const int8_t, 12) + (size_t)j * H, DP(const int8_t, 13) + (size_t)j * H,
                    DP(const float, 14) + (size_t)j * nb, DP(float, 15) + (size_t)pq * 2 * I + mat * I + 16 * grp);
}

__global__ void __launch_bounds__(256) k_ds_hot_act(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int pq = blockIdx.y, I = DI(11), i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) {
        return;
    }
    float *o = DP(float, 16) + (size_t)pq * I;
    if (DP(const int, 3)[DP(const int, 1)[pq]] < 0) {
        o[i] = 0.f;
        return;
    }
    const float *a = DP(const float, 15) + (size_t)pq * 2 * I;
    float g = fminf(a[i], 10.f), u = fminf(fmaxf(a[I + i], -10.f), 10.f);
    float sl = __fdiv_rn(g, __fadd_rn(1.f, rc_dexp(-g)));
    o[i] = __fmul_rn(__fmul_rn(sl, u), DP(const float, 2)[pq]);
}

__global__ void __launch_bounds__(256) k_ds_hot_dn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int pq = blockIdx.y, H = DI(10), I = DI(11);
    int slot = DP(const int, 3)[DP(const int, 1)[pq]];
    if (slot < 0) {
        return;
    }
    int grp = blockIdx.x, nb = I / 32;
    const uint8_t *w = DP(const uint8_t, 4) + (size_t)slot * di(r, e, 5) + di(r, e, 7) + (size_t)grp * nb * MXX_BB;
    mxx_group_x2<8>(w, I, DP(const int8_t, 17) + (size_t)pq * I, DP(const int8_t, 18) + (size_t)pq * I,
                    DP(const float, 19) + (size_t)pq * nb, DP(float, 20) + (size_t)pq * H + 16 * grp);
}

/* GP_DS_HOT_MOE for a group of a prompt (t > 8): a block for each (group of
 * 16 rows, matrix, expert); the experts the GPU does not hold and the ones no
 * pair takes leave at once; the others take their pairs (the scan of idx)
 * and the same rows of the expert for each of them in turn, so the rows are
 * read once from the memory and then from the cache (k_ds_hot_gu read them
 * again for every pair: 1.9 s of a group of 512 on the real prompt). The
 * operations of a pair are those of k_ds_hot_gu: the same bits. */
#define DS_HOT_MAXP 3072

/* The pairs of expert ex among the pairs [p0, p0 + DS_HOT_MAXP k) of idx
 * [t k] into pl (an expert at most once a token: at most DS_HOT_MAXP), in any
 * order (each pair's sums are its own); their count. Every thread calls it,
 * for the chunks of the pairs in turn (a group of any size). */
__device__ int ds_hot_pairs(const int *idx, int ex, int p0, int np, int k, int *pl, int *cnt)
{
    __syncthreads();            /* the chunk before is done with pl */
    if (threadIdx.x == 0) {
        *cnt = 0;
    }
    __syncthreads();
    int p1 = p0 + DS_HOT_MAXP * k < np ? p0 + DS_HOT_MAXP * k : np;
    for (int p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
        if (idx[p] == ex) {
            int at = atomicAdd(cnt, 1);
            if (at < DS_HOT_MAXP) {
                pl[at] = p;
            }
        }
    }
    __syncthreads();
    return *cnt < DS_HOT_MAXP ? *cnt : DS_HOT_MAXP;
}
__global__ void __launch_bounds__(256) k_ds_hot_gu_g(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ex = blockIdx.z, k = DI(9), H = DI(10), I = DI(11), t = DI(8);
    int slot = DP(const int, 3)[ex];
    if (slot < 0) {
        return;
    }
    __shared__ int pl[DS_HOT_MAXP];
    __shared__ int np_;
    const int *idx = DP(const int, 1);
    for (int p0 = 0; p0 < t * k; p0 += DS_HOT_MAXP * k) {
        int n = ds_hot_pairs(idx, ex, p0, t * k, k, pl, &np_);
        if (n == 0) {
            continue;
        }
        int mat = blockIdx.y, grp = blockIdx.x, nb = H / 32;
        const uint8_t *w = DP(const uint8_t, 4) + (size_t)slot * di(r, e, 5) + (mat ? di(r, e, 6) : 0) +
                           (size_t)grp * nb * MXX_BB;
        for (int q = 0; q < n; ++q) {
            int pq = pl[q], j = pq / k;
            mxx_group_x2<8>(w, H, DP(const int8_t, 12) + (size_t)j * H, DP(const int8_t, 13) + (size_t)j * H,
                            DP(const float, 14) + (size_t)j * nb,
                            DP(float, 15) + (size_t)pq * 2 * I + mat * I + 16 * grp);
            __syncthreads();
        }
    }
}

__global__ void __launch_bounds__(256) k_ds_hot_dn_g(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ex = blockIdx.y, k = DI(9), H = DI(10), I = DI(11), t = DI(8);
    int slot = DP(const int, 3)[ex];
    if (slot < 0) {
        return;
    }
    __shared__ int pl[DS_HOT_MAXP];
    __shared__ int np_;
    const int *idx = DP(const int, 1);
    for (int p0 = 0; p0 < t * k; p0 += DS_HOT_MAXP * k) {
        int n = ds_hot_pairs(idx, ex, p0, t * k, k, pl, &np_);
        if (n == 0) {
            continue;
        }
        int grp = blockIdx.x, nb = I / 32;
        const uint8_t *w = DP(const uint8_t, 4) + (size_t)slot * di(r, e, 5) + di(r, e, 7) + (size_t)grp * nb * MXX_BB;
        for (int q = 0; q < n; ++q) {
            int pq = pl[q];
            mxx_group_x2<8>(w, I, DP(const int8_t, 17) + (size_t)pq * I, DP(const int8_t, 18) + (size_t)pq * I,
                            DP(const float, 19) + (size_t)pq * nb, DP(float, 20) + (size_t)pq * H + 16 * grp);
            __syncthreads();
        }
    }
}

/* The same on the tensor cores (gg_caps.tc): a warp takes 16 rows of the
 * expert for 8 of its pairs at once, mma m16n8k32 on the int8 planes (the
 * weights 2 e2m1 from the KQ_MXX block: the 4 bytes at 32 c + 4 g are rows g
 * (low) and g + 8 (high) of values 4c .. 4c + 3, the A fragments; x the B
 * fragments), so for each block of 32 columns a = 128 sum(hi w) + sum(lo w),
 * exact in int32 as in k_mxx_rows_x2; then p = a 2^(e - 127) and f =
 * fma(p, xs, f) over the blocks in order, 0.5 f: the bits of the CPU. The
 * rows of the expert once for its pairs, 8 pairs a load. */
template <bool DN>
__global__ void __launch_bounds__(128) k_ds_hot_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ex = DN ? blockIdx.y : blockIdx.z, k = DI(9), H = DI(10), I = DI(11), t = DI(8);
    int slot = DP(const int, 3)[ex];
    if (slot < 0) {
        return;
    }
    __shared__ int pl[DS_HOT_MAXP];
    __shared__ int np_;
    const int *idx = DP(const int, 1);
    for (int p0 = 0; p0 < t * k; p0 += DS_HOT_MAXP * k) {
        int n = ds_hot_pairs(idx, ex, p0, t * k, k, pl, &np_);
        if (n == 0) {
            continue;
        }
        int cols = DN ? I : H, nb = cols / 32, grp = blockIdx.x, mat = DN ? 0 : blockIdx.y;
        const uint8_t *wg = DP(const uint8_t, 4) + (size_t)slot * di(r, e, 5) +
                            (DN ? di(r, e, 7) : (mat ? di(r, e, 6) : 0)) + (size_t)grp * nb * MXX_BB;
        const int8_t *qh = DP(const int8_t, DN ? 17 : 12), *ql = DP(const int8_t, DN ? 18 : 13);
        const float *xs = DP(const float, DN ? 19 : 14);
        float *out = DP(float, DN ? 20 : 15);
        int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
        for (int t0 = 8 * warp; t0 < n; t0 += 32) {
            /* the pair of this lane's B column (g), and of its C columns (2c, 2c + 1) */
            int pb = t0 + g < n ? pl[t0 + g] : -1;
            int pc0 = t0 + 2 * c < n ? pl[t0 + 2 * c] : -1, pc1 = t0 + 2 * c + 1 < n ? pl[t0 + 2 * c + 1] : -1;
            size_t xb = pb < 0 ? 0 : (size_t)(DN ? pb : pb / k) * cols;
            size_t s0 = pc0 < 0 ? 0 : (size_t)(DN ? pc0 : pc0 / k) * nb;
            size_t s1 = pc1 < 0 ? 0 : (size_t)(DN ? pc1 : pc1 / k) * nb;
            float f[4] = {0.f, 0.f, 0.f, 0.f};
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + (size_t)b * MXX_BB;
                uint32_t w0 = *(const uint32_t *)(blk + 32 * c + 4 * g);
                uint32_t w1 = *(const uint32_t *)(blk + 128 + 32 * c + 4 * g);
                uint32_t a[4] = {kt_e2m1x4(w0), kt_e2m1x4(w0 >> 4), kt_e2m1x4(w1), kt_e2m1x4(w1 >> 4)};
                uint32_t bh[2] = {0u, 0u}, bl[2] = {0u, 0u};
                if (pb >= 0) {
                    bh[0] = *(const uint32_t *)(qh + xb + 32 * b + 4 * c);
                    bh[1] = *(const uint32_t *)(qh + xb + 32 * b + 16 + 4 * c);
                    bl[0] = *(const uint32_t *)(ql + xb + 32 * b + 4 * c);
                    bl[1] = *(const uint32_t *)(ql + xb + 32 * b + 16 + 4 * c);
                }
                int ch[4] = {0, 0, 0, 0}, cl[4] = {0, 0, 0, 0};
                mma16832(ch, a, bh);
                mma16832(cl, a, bl);
                float e0 = e8m0f(blk[256 + g]), e1 = e8m0f(blk[256 + g + 8]);
                float x0 = pc0 >= 0 ? xs[s0 + b] : 0.f, x1 = pc1 >= 0 ? xs[s1 + b] : 0.f;
                f[0] = fmaf(__fmul_rn((float)(128 * ch[0] + cl[0]), e0), x0, f[0]);
                f[1] = fmaf(__fmul_rn((float)(128 * ch[1] + cl[1]), e0), x1, f[1]);
                f[2] = fmaf(__fmul_rn((float)(128 * ch[2] + cl[2]), e1), x0, f[2]);
                f[3] = fmaf(__fmul_rn((float)(128 * ch[3] + cl[3]), e1), x1, f[3]);
            }
            /* c0, c1: row g, pairs 2c, 2c + 1; c2, c3: row g + 8 */
            int rows[2] = {16 * grp + g, 16 * grp + g + 8};
            int pcs[2] = {pc0, pc1};
            for (int i = 0; i < 4; ++i) {
                int pq = pcs[i & 1], row = rows[i >> 1];
                if (pq < 0) {
                    continue;
                }
                float v = __fmul_rn(f[i], 0.5f);
                if (DN) {
                    out[(size_t)pq * H + row] = v;
                } else {
                    out[(size_t)pq * 2 * I + mat * I + row] = v;
                }
            }
        }
    }
}

/* GP_DS_COMBINE: idx [t, k], map [E], de [t k, H], rows (pinned: the de rows
 * of the CPU: a step, those of its cold pairs in their order (GP_HOT_SPLIT);
 * a group, row p for each cold pair p (GP_HOT_SPLIT_MT)), out [t, H], k, H, t, mt
 * (1: the rows of GP_HOT_SPLIT_MT also when t is 1). */
__global__ void __launch_bounds__(256) k_ds_combine(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int c = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y, k = DI(5), H = DI(6);
    if (c >= H) {
        return;
    }
    const int *idx = DP(const int, 0), *map = DP(const int, 1);
    const float *de = DP(const float, 2), *rows = DP(const float, 3);
    float acc = 0.f;
    int n = 0;
    /* the CPU's rows in the order of its cold pairs (GP_HOT_SPLIT: a step), or
     * row p for pair p (GP_HOT_SPLIT_MT: operand 8 set; a lane of one token of
     * a verify group too) */
    int packed = DI(7) == 1 && !DI(8);
    for (int s = 0; s < k; ++s) {
        int p = tok * k + s;
        acc += map[idx[p]] >= 0 ? de[(size_t)p * H + c] : __ldcv(rows + (size_t)(packed ? n++ : p) * H + c);
    }
    DP(float, 4)[(size_t)tok * H + c] = acc;
}


/* ---------- DSpark (DEEPSEEK_V41_PLAN.md Phase 5) ---------- */

/* GP_DS_GATHER: table [V, n] (float32), ids (int32), i, out [n], n: out =
 * the row of token ids[i] (the Markov embedding of a draft token). */
__global__ void __launch_bounds__(256) k_ds_gather(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int n = DI(4);
    int64_t tok = DP(const int, 1)[DI(2)];
    for (int c = threadIdx.x; c < n; c += blockDim.x) {
        DP(float, 3)[c] = DP(const float, 0)[tok * n + c];
    }
}

/* GP_DS_MARKOV: logits [n, V], bias [V], ids [n + 1] (int32), i, V: row i of
 * the logits plus the bias of the Markov head; ids[i + 1] = its argmax (the
 * first of the largest, as torch). One block of 1024 threads. */
__global__ void __launch_bounds__(1024) k_ds_markov(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int i = DI(3), V = DI(4);
    float *lg = DP(float, 0) + (size_t)i * V;
    const float *b = DP(const float, 1);
    float best = -INFINITY;
    int bi = V;
    for (int c = threadIdx.x; c < V; c += blockDim.x) {
        float v = lg[c] + b[c];
        lg[c] = v;
        if (v > best) {
            best = v;
            bi = c;
        }
    }
    __shared__ float sv[32];
    __shared__ int si[32];
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(RC_FULL_MASK, best, o);
        int i2 = __shfl_xor_sync(RC_FULL_MASK, bi, o);
        if (v2 > best || (v2 == best && i2 < bi)) {
            best = v2;
            bi = i2;
        }
    }
    if (threadIdx.x % 32 == 0) {
        sv[threadIdx.x / 32] = best;
        si[threadIdx.x / 32] = bi;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)(blockDim.x / 32); ++w) {
            if (sv[w] > sv[0] || (sv[w] == sv[0] && si[w] < si[0])) {
                sv[0] = sv[w];
                si[0] = si[w];
            }
        }
        DP(int, 2)[i + 1] = si[0];
    }
}

/* GP_DS_CONF: x [n, dim] (the input of the head before its norm), me [n, rk]
 * (the Markov embeddings), w [dim + rk] (float32), out [n], n, dim, rk: the
 * logit of the confidence of each draft position (model.py
 * DSparkConfidenceHead). A block for each position. */
__global__ void __launch_bounds__(256) k_ds_conf(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, dim = DI(5), rk = DI(6);
    const float *x = DP(const float, 0) + (size_t)j * dim, *me = DP(const float, 1) + (size_t)j * rk;
    const float *w = DP(const float, 2);
    float s = 0.f;
    for (int c = threadIdx.x; c < dim + rk; c += blockDim.x) {
        s += (c < dim ? x[c] : me[c - dim]) * w[c];
    }
    s = block_sum(s);
    if (threadIdx.x == 0) {
        DP(float, 3)[j] = s;
    }
}


/* ---------- the inputs of a run on the GPU ---------- */

/* GP_DS_EMBED: table (the BF12 rows of the embedding, in pinned host memory:
 * read through the map), ids [t] (int32), h [t, hc, dim], t, dim, hc: the
 * row of each token, decoded (kq_bf12_bits: the values of the CPU model's
 * embed_rows), into every stream. A block for each token. */
__global__ void __launch_bounds__(256) k_ds_embed(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, dim = DI(4), hc = DI(5);
    size_t rb = (size_t)dim + dim / 2 + dim / 32;
    const uint8_t *row = DP(const uint8_t, 0) + (size_t)DP(const int, 1)[j] * rb;
    float *h = DP(float, 2) + (size_t)j * hc * dim;
    for (int c = 8 * threadIdx.x; c < dim; c += 8 * blockDim.x) {
        uint32_t bits[8];
        kq_bf12_bits<8>(row, dim, c, bits);
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            float v = __uint_as_float(bits[u] << 16);
            for (int s2 = 0; s2 < hc; ++s2) {
                h[(size_t)s2 * dim + c + u] = v;
            }
        }
    }
}

/* GP_DS_EMBED_ROWS: rows [t, dim], vis [t] (int32), h [t, hc, dim], t, dim,
 * hc: the row of each token of an image span (vis[j] != 0: the aligner's
 * row, or a learned delimiter) in place of its embedding, into every stream
 * (model.py merge_image_embeddings). A block for each token. */
__global__ void __launch_bounds__(256) k_ds_embed_rows(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, dim = DI(4), hc = DI(5);
    if (!DP(const int, 1)[j]) {
        return;
    }
    const float *row = DP(const float, 0) + (size_t)j * dim;
    float *h = DP(float, 2) + (size_t)j * hc * dim;
    for (int c = threadIdx.x; c < dim; c += blockDim.x) {
        float v = row[c];
        for (int s2 = 0; s2 < hc; ++s2) {
            h[(size_t)s2 * dim + c] = v;
        }
    }
}

/* GP_DS_ENGRAM_ROWS: codes [n, w] (E4M3), scales [n, w / 32] (E8M0), out [n,
 * w], n, w: the values of the rows of an engram table (the 24 rows of each
 * token, gathered from the shards by the host). */
__global__ void __launch_bounds__(256) k_ds_engram_rows(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t n = di(r, e, 3), w = DI(4), i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * w) {
        return;
    }
    int64_t row = i / w, c = i % w;
    float sc = __int_as_float((int)DP(const uint8_t, 1)[row * (w / 32) + c / 32] << 23);
    DP(float, 2)[i] = __fmul_rn(e4m3f(DP(const uint8_t, 0)[i]), sc);
}

/* GP_DS_ROWS_AT: src [t, w], dst [n, w], pos, base, t, n, w (float32): row j
 * of the group (position pos + j) into row pos + j - base of dst when that
 * is in 0 .. n - 1 (the encoder's outputs of the last tokens of a prompt, the
 * input of the decoder's bounded replay). */
__global__ void __launch_bounds__(256) k_ds_rows_at(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t t = di(r, e, 4), n = di(r, e, 5), w = di(r, e, 6), i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= t * w) {
        return;
    }
    int64_t j = i / w, d = di(r, e, 2) + j - di(r, e, 3);
    if (d >= 0 && d < n) {
        DP(float, 1)[d * w + i % w] = DP(const float, 0)[i];
    }
}
