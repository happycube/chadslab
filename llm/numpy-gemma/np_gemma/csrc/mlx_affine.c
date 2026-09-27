/* The MLX affine weight format (mlx-community, OptiQ): products for one
 * token, a small group, and the experts of a MoE layer.
 *
 * This file is part of the cops library: bf16_linear.c includes it, and it
 * uses the helpers there (bf16_to_f32).
 *
 * The weights are MLX affine blocks: for each row, groups of 64 values,
 * w = scale * q + bias, with q of 4 or 8 bits packed in uint32 words from
 * the low bits up, and scale and bias in bfloat16.
 *
 * The products quantize x to int8 for each group of 64 (ma_quant_x): xq,
 * the scale xs = max |x| / 127, and xsum = xs * sum(xq). Then for each row
 * and group
 *
 *     y += scale * xs * sum(q * xq) + bias * xsum
 *
 * The weights are unsigned (0..15 or 0..255) and xq is signed, which is the
 * operation of the VNNI instruction vpdpbusd (u8 x s8 -> s32).
 *
 * For 4-bit weights, byte j of a group holds the values 2j (low 4 bits) and
 * 2j + 1 (high 4 bits). The kernel does not reorder the weights; ma_quant_x
 * puts x in the same order. A pair of groups (128 values, 64 bytes of
 * weights) is one register: the low 4 bits of its 64 bytes are the even
 * values of the two groups, and the high 4 bits the odd values. Thus xq of
 * 4 bits has, for each pair of groups A and B: the even values of A, of B,
 * then the odd values of A, of B (128 bytes).
 */
#define MA_G 64          /* the values of a group */
#define MA_MAX_T 16      /* the most tokens of one product */

/* Quantize t rows of x (cols values each, cols % 128 == 0 for 4 bits). xq
 * has cols bytes for each row, in the order of the bits (see the top).
 * xs and xsum have cols / 64 values for each row. */
static void ma_quant_rows(const float *x, int t, int cols, int bits, int8_t *xq, float *xs,
                       float *xsum)
{
    int ng = cols / MA_G;
    for (int r = 0; r < t; ++r) {
        const float *xr = x + (size_t)r * cols;
        int8_t *qr = xq + (size_t)r * cols;
        for (int g = 0; g < ng; ++g) {
            const float *v = xr + g * MA_G;
            float m = 0.f, s = 0.f;
            for (int i = 0; i < MA_G; ++i) {
                float a = fabsf(v[i]);
                m = a > m ? a : m;
                s += v[i];
            }
            float sc = m / 127.f, inv = m > 0.f ? 127.f / m : 0.f;
            int32_t qs = 0;
            (void)s;
            xs[(size_t)r * ng + g] = sc;
            for (int i = 0; i < MA_G; ++i) {
                int q = (int)lrintf(v[i] * inv);
                int8_t qv = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
                qs += qv;
                int dst;
                if (bits == 8) {
                    dst = g * MA_G + i;
                } else {
                    /* pair p = g / 2, in the pair: A (g even) or B (g odd) */
                    int p = g / 2, b = g % 2, odd = i % 2;
                    dst = p * 128 + odd * 64 + b * 32 + i / 2;
                }
                qr[dst] = qv;
            }
            /* The sum of the quantized values, not of x: then the product is
             * exactly xq . w. The weights are s q + b with q >= 0, so s q is
             * far from zero mean, and the error of xq cancels only with the
             * bias term of the same xq. */
            xsum[(size_t)r * ng + g] = sc * (float)qs;
        }
    }
}

void ma_quant_x(const float *x, int t, int cols, int bits, int8_t *xq, float *xs, float *xsum)
{
    ma_quant_rows(x, t, cols, bits, xq, xs, xsum);
}

/* The sum over the groups of bias * sum(x) for one row. */
static inline float ma_bias_dot(const uint16_t *b, const float *xsum, int ng)
{
    float s = 0.f;
    for (int g = 0; g < ng; ++g) {
        s += bf16_to_f32(b[g]) * xsum[g];
    }
    return s;
}

/* One row of weights (w, s, b) on t rows of quantized x. out[j] gets the
 * product of token j. */
static void ma_row_dot(const uint32_t *w, const uint16_t *s, const uint16_t *b, int bits, int cols,
                    const int8_t *xq, const float *xs, const float *xsum, int t, float *out,
                    size_t ostride)
{
    int ng = cols / MA_G;
    const uint8_t *wb = (const uint8_t *)w;
#if defined(__AVX512VNNI__)
    __m512 acc[MA_MAX_T];
    for (int j = 0; j < t; ++j) {
        acc[j] = _mm512_setzero_ps();
    }
    if (bits == 8) {
        for (int g = 0; g < ng; ++g) {
            __m512i wv = _mm512_loadu_si512((const void *)(wb + (size_t)g * MA_G));
            float sg = bf16_to_f32(s[g]);
            for (int j = 0; j < t; ++j) {
                __m512i xv = _mm512_loadu_si512((const void *)(xq + (size_t)j * cols + g * MA_G));
                __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wv, xv);
                acc[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is),
                                         _mm512_set1_ps(sg * xs[(size_t)j * ng + g]), acc[j]);
            }
        }
    } else {
        const __m512i m4 = _mm512_set1_epi8(0x0f);
        for (int p = 0; p < ng / 2; ++p) {
            __m512i wv = _mm512_loadu_si512((const void *)(wb + (size_t)p * 64));
            __m512i lo = _mm512_and_si512(wv, m4);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(wv, 4), m4);
            float sa = bf16_to_f32(s[2 * p]), sb = bf16_to_f32(s[2 * p + 1]);
            for (int j = 0; j < t; ++j) {
                const int8_t *xp = xq + (size_t)j * cols + (size_t)p * 128;
                __m512i ev = _mm512_loadu_si512((const void *)xp);
                __m512i od = _mm512_loadu_si512((const void *)(xp + 64));
                __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, ev);
                is = _mm512_dpbusd_epi32(is, hi, od);
                const float *xsj = xs + (size_t)j * ng + 2 * p;
                __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sa * xsj[0]),
                                                 _mm512_set1_ps(sb * xsj[1]));
                acc[j] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc[j]);
            }
        }
    }
    for (int j = 0; j < t; ++j) {
        out[(size_t)j * ostride] = _mm512_reduce_add_ps(acc[j]) +
                                   ma_bias_dot(b, xsum + (size_t)j * ng, ng);
    }
#else
    for (int j = 0; j < t; ++j) {
        const int8_t *xr = xq + (size_t)j * cols;
        float acc = 0.f;
        for (int g = 0; g < ng; ++g) {
            int32_t is = 0;
            if (bits == 8) {
                for (int i = 0; i < MA_G; ++i) {
                    is += (int32_t)wb[(size_t)g * MA_G + i] * xr[g * MA_G + i];
                }
            } else {
                int p = g / 2, bsel = g % 2;
                for (int i = 0; i < 32; ++i) {
                    uint8_t byte = wb[(size_t)g * 32 + i];
                    is += (int32_t)(byte & 15) * xr[p * 128 + bsel * 32 + i];
                    is += (int32_t)(byte >> 4) * xr[p * 128 + 64 + bsel * 32 + i];
                }
            }
            acc += bf16_to_f32(s[g]) * xs[(size_t)j * ng + g] * (float)is;
        }
        out[(size_t)j * ostride] = acc + ma_bias_dot(b, xsum + (size_t)j * ng, ng);
    }
#endif
}

/* out (t x rows) = x W^T for a matrix of rows x cols. x is quantized with
 * the bits of W (ma_quant_x). */
void ma_linear(const uint32_t *w, const uint16_t *s, const uint16_t *b, int bits, int rows,
               int cols, const int8_t *xq, const float *xs, const float *xsum, int t, float *out)
{
    int ng = cols / MA_G;
    size_t wrow = (size_t)cols * bits / 32;
    /* ma_row_dot keeps MA_MAX_T sums: more tokens go in blocks. */
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        for (int j0 = 0; j0 < t; j0 += MA_MAX_T) {
            int tj = t - j0 < MA_MAX_T ? t - j0 : MA_MAX_T;
            ma_row_dot(w + (size_t)r * wrow, s + (size_t)r * ng, b + (size_t)r * ng, bits, cols,
                       xq + (size_t)j0 * cols, xs + (size_t)j0 * ng, xsum + (size_t)j0 * ng, tj,
                       out + (size_t)j0 * rows + r, (size_t)rows);
        }
    }
}

static inline float ma_silu(float v)
{
    return v / (1.f + expf(-v));
}

/* One matrix of the MoE step: data, scales, biases, bits. */
typedef struct {
    const uint32_t *w;
    const uint16_t *s, *b;
    int bits;
} ma_mat;

/* The experts of one token (the decode step), with the shared expert.
 *
 * h (hidden) is quantized for 4 and 8 bits: hq4/hq8, hs, hsum (the same
 * scales and sums for both orders). ids and val give the k experts and
 * their weights. gate, up, down are the stacked expert matrices (the
 * address of expert e is the base plus e times the size of one). sg, su, sd
 * are the shared expert of models that have one (Qwen3.5), or null; shared_w
 * is its weight (for Qwen3.5 sigmoid(gate . h), which the caller computes). act (2 * (k + 1) * inner), aq (2 * (k + 1) * inner
 * bytes), as, asum are scratch. out gets the sum (hidden values).
 *
 * One parallel region: the gate and up rows of all the experts, then the
 * activations and their quantization, then the down rows. */
void ma_moe_step(const int8_t *hq4, const int8_t *hq8, const float *hs, const float *hsum,
                 const int32_t *ids, const float *val, int k,
                 const uint32_t *gw, const uint16_t *gs, const uint16_t *gb, int gbits,
                 const uint32_t *uw, const uint16_t *us, const uint16_t *ub, int ubits,
                 const uint32_t *dw, const uint16_t *ds, const uint16_t *db, int dbits,
                 const uint32_t *sgw, const uint16_t *sgs, const uint16_t *sgb, int sgbits,
                 const uint32_t *suw, const uint16_t *sus, const uint16_t *sub, int subits,
                 const uint32_t *sdw, const uint16_t *sds, const uint16_t *sdb, int sdbits,
                 float shared_w, int hidden, int inner, float *act, int8_t *aq, float *as,
                 float *asum, float *de, float *out)
{
    int ne = sgw != NULL ? k + 1 : k;      /* the experts, and the shared one */
    int ngh = hidden / MA_G, ngi = inner / MA_G;
    size_t gsz_w = (size_t)inner * hidden / 32, gsz_s = (size_t)inner * ngh;
    size_t dsz_w = (size_t)hidden * inner / 32, dsz_s = (size_t)hidden * ngi;
    #pragma omp parallel
    {
        /* 1. gate and up: 2 * inner rows for each expert. */
        #pragma omp for schedule(static)
        for (int x = 0; x < ne * 2 * inner; ++x) {
            int e = x / (2 * inner), rr = x % (2 * inner), isup = rr >= inner, r = rr % inner;
            ma_mat m;
            if (e < k) {
                size_t ex = (size_t)ids[e];
                if (!isup) {
                    m.w = gw + ex * gsz_w * gbits; m.s = gs + ex * gsz_s; m.b = gb + ex * gsz_s;
                    m.bits = gbits;
                } else {
                    m.w = uw + ex * gsz_w * ubits; m.s = us + ex * gsz_s; m.b = ub + ex * gsz_s;
                    m.bits = ubits;
                }
            } else if (!isup) {
                m.w = sgw; m.s = sgs; m.b = sgb; m.bits = sgbits;
            } else {
                m.w = suw; m.s = sus; m.b = sub; m.bits = subits;
            }
            size_t wrow = (size_t)hidden * m.bits / 32;
            ma_row_dot(m.w + (size_t)r * wrow, m.s + (size_t)r * ngh, m.b + (size_t)r * ngh, m.bits,
                    hidden, m.bits == 4 ? hq4 : hq8, hs, hsum, 1,
                    act + (size_t)e * 2 * inner + rr, 1);
        }
        /* 2. silu(gate) * up, and its quantization, for each expert. */
        #pragma omp for schedule(static)
        for (int e = 0; e < ne; ++e) {
            float *a = act + (size_t)e * 2 * inner;
            for (int i = 0; i < inner; ++i) {
                a[i] = ma_silu(a[i]) * a[inner + i];
            }
            int bits = e < k ? dbits : sdbits;
            ma_quant_rows(a, 1, inner, bits, aq + (size_t)e * inner, as + (size_t)e * ngi,
                       asum + (size_t)e * ngi);
        }
        /* 3. down: hidden rows for each expert. */
        #pragma omp for schedule(static)
        for (int x = 0; x < ne * hidden; ++x) {
            int e = x / hidden, r = x % hidden;
            ma_mat m;
            if (e < k) {
                size_t ex = (size_t)ids[e];
                m.w = dw + ex * dsz_w * dbits; m.s = ds + ex * dsz_s; m.b = db + ex * dsz_s;
                m.bits = dbits;
            } else {
                m.w = sdw; m.s = sds; m.b = sdb; m.bits = sdbits;
            }
            size_t wrow = (size_t)inner * m.bits / 32;
            ma_row_dot(m.w + (size_t)r * wrow, m.s + (size_t)r * ngi, m.b + (size_t)r * ngi, m.bits,
                    inner, aq + (size_t)e * inner, as + (size_t)e * ngi, asum + (size_t)e * ngi, 1,
                    de + (size_t)e * hidden + r, 1);
        }
        /* 4. the weighted sum. */
        #pragma omp for schedule(static)
        for (int c = 0; c < hidden; ++c) {
            float v = ne > k ? shared_w * de[(size_t)k * hidden + c] : 0.f;
            for (int e = 0; e < k; ++e) {
                v += val[e] * de[(size_t)e * hidden + c];
            }
            out[c] = v;
        }
    }
}
