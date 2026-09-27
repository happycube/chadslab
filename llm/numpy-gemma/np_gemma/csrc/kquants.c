/* The GGUF block formats of llama.cpp: products for one token, a small
 * group, and the experts of a MoE layer.
 *
 * This file is part of the cops library: bf16_linear.c includes it, and it
 * uses the helpers there (fp16_to_f32) and in moe.c.
 *
 * The formats (the numbers are the ggml types):
 *
 *     KQ_F32   0   float32 rows.
 *     KQ_Q8_0  8   blocks of 32: d (fp16), 32 int8. w = d * q.
 *     KQ_Q4_K 12   blocks of 256 (144 bytes): d, dmin (fp16), 12 bytes of
 *                  6-bit scales and mins for 8 parts of 32, 128 bytes of
 *                  4-bit values. w = d * sc * q - dmin * m.
 *     KQ_Q5_K 13   Q4_K with a fifth bit (32 more bytes, qh).
 *     KQ_Q6_K 14   blocks of 256 (210 bytes): 128 bytes of the low 4 bits,
 *                  64 bytes of the high 2 bits, 16 int8 scales (one for
 *                  each 16 values), d. w = d * sc * (q - 32).
 *
 * The products quantize x to int8 in its natural order, with one scale xs
 * for each 32 values and the sum xm of the int8 values of each 16 values
 * (kq_quant_x). The same x serves all the formats. For each part of a
 * block:
 *
 *     Q8_0:  y += d * xs * sum(q * xq)
 *     Q4_K:  y += d * sc * xs * sum(q * xq) - dmin * m * xs * sum(xq)
 *     Q6_K:  y += d * sc * xs * (sum(q * xq) - 32 * sum(xq))
 *
 * The values of the K formats are not negative, which is the operation of
 * the VNNI instruction vpdpbusd (u8 x s8 -> s32). Q8_0 is signed: the
 * product uses |q| and x with the sign of q.
 *
 * All the products of one row use one function for one token, so a token
 * gives the same bits alone and in a group (an MTP verify group).
 */
#define KQ_F32 0
#define KQ_Q8_0 8
#define KQ_Q4_K 12
#define KQ_Q5_K 13
#define KQ_Q6_K 14

/* The bytes of one row of cols values. */
static inline size_t kq_row_bytes(int type, int cols)
{
    switch (type) {
    case KQ_F32: return (size_t)cols * 4;
    case KQ_Q8_0: return (size_t)cols / 32 * 34;
    case KQ_Q4_K: return (size_t)cols / 256 * 144;
    case KQ_Q5_K: return (size_t)cols / 256 * 176;
    case KQ_Q6_K: return (size_t)cols / 256 * 210;
    }
    return 0;
}

static inline float kq_h(const uint8_t *p)
{
    return fp16_to_f32((uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)));
}

/* Quantize the part g (32 values) of one row of x. */
static void kq_quant_part(const float *xr, int g, int8_t *qr, float *xs, float *xm)
{
    const float *v = xr + g * 32;
    float m = 0.f;
    for (int i = 0; i < 32; ++i) {
        float a = fabsf(v[i]);
        m = a > m ? a : m;
    }
    float inv = m > 0.f ? 127.f / m : 0.f;
    xs[g] = m / 127.f;
    int32_t s0 = 0, s1 = 0;
    for (int i = 0; i < 32; ++i) {
        int q = (int)lrintf(v[i] * inv);
        int8_t qv = (int8_t)(q > 127 ? 127 : (q < -127 ? -127 : q));
        qr[g * 32 + i] = qv;
        if (i < 16) {
            s0 += qv;
        } else {
            s1 += qv;
        }
    }
    xm[2 * g] = (float)s0;
    xm[2 * g + 1] = (float)s1;
}

/* Quantize t rows of x (cols % 32 == 0), inside a parallel region: xq (t x
 * cols), xs (t x cols / 32), xm (t x cols / 16). */
static void kq_quant_body(const float *x, int t, int cols, int8_t *xq, float *xs, float *xm)
{
    int np = cols / 32;
    #pragma omp for schedule(static)
    for (int x2 = 0; x2 < t * np; ++x2) {
        int r = x2 / np, g = x2 % np;
        kq_quant_part(x + (size_t)r * cols, g, xq + (size_t)r * cols, xs + (size_t)r * np,
                      xm + (size_t)r * 2 * np);
    }
}

void kq_quant_x(const float *x, int t, int cols, int8_t *xq, float *xs, float *xm)
{
    #pragma omp parallel
    kq_quant_body(x, t, cols, xq, xs, xm);
}

/* The 6-bit scale and min of part j of a Q4_K or Q5_K block (ggml,
 * get_scale_min_k4). */
static inline void kq_scale_min(const uint8_t *q, int j, int *sc, int *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

/* The float values of block b of a row (the reference, and the path
 * without VNNI). out gets 32 (Q8_0) or 256 values. */
static void kq_block_values(int type, const uint8_t *row, int b, float *out)
{
    if (type == KQ_Q8_0) {
        const uint8_t *blk = row + (size_t)b * 34;
        float d = kq_h(blk);
        for (int i = 0; i < 32; ++i) {
            out[i] = d * (float)(int8_t)blk[2 + i];
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        int five = type == KQ_Q5_K;
        const uint8_t *blk = row + (size_t)b * (five ? 176 : 144);
        float d = kq_h(blk), dm = kq_h(blk + 2);
        const uint8_t *qh = blk + 16, *qs = blk + (five ? 48 : 16);
        for (int c = 0; c < 4; ++c) {
            int s0, m0, s1, m1;
            kq_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            for (int l = 0; l < 32; ++l) {
                int lo = qs[32 * c + l] & 15, hi = qs[32 * c + l] >> 4;
                if (five) {
                    lo |= ((qh[l] >> (2 * c)) & 1) << 4;
                    hi |= ((qh[l] >> (2 * c + 1)) & 1) << 4;
                }
                out[64 * c + l] = d * s0 * lo - dm * m0;
                out[64 * c + 32 + l] = d * s1 * hi - dm * m1;
            }
        }
    } else if (type == KQ_Q6_K) {
        const uint8_t *blk = row + (size_t)b * 210;
        float d = kq_h(blk + 208);
        const int8_t *sc = (const int8_t *)(blk + 192);
        for (int h = 0; h < 2; ++h) {
            const uint8_t *ql = blk + 64 * h, *qh = blk + 128 + 32 * h;
            for (int l = 0; l < 32; ++l) {
                int q1 = (ql[l] & 15) | (((qh[l] >> 0) & 3) << 4);
                int q2 = (ql[l + 32] & 15) | (((qh[l] >> 2) & 3) << 4);
                int q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                int q4 = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                float *o = out + 128 * h;
                o[l] = d * sc[8 * h + l / 16] * (q1 - 32);
                o[l + 32] = d * sc[8 * h + 2 + l / 16] * (q2 - 32);
                o[l + 64] = d * sc[8 * h + 4 + l / 16] * (q3 - 32);
                o[l + 96] = d * sc[8 * h + 6 + l / 16] * (q4 - 32);
            }
        }
    }
}

#if defined(__AVX512VNNI__)
static inline __m512i kq_two256(const int8_t *a, const int8_t *b)
{
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a)),
                              _mm256_loadu_si256((const __m256i *)b), 1);
}

static float kq_dot_q8_0(const uint8_t *w, int cols, const int8_t *xq, const float *xs)
{
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < cols / 32; i += 2) {
        const uint8_t *b0 = w + (size_t)i * 34, *b1 = b0 + 34;
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b1 + 2));
        __m512i xv = _mm512_loadu_si512((const void *)(xq + (size_t)i * 32));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(wv), sx);
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(kq_h(b0) * xs[i]),
                                         _mm512_set1_ps(kq_h(b1) * xs[i + 1]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static float kq_dot_q45k(const uint8_t *w, int cols, int five, const int8_t *xq, const float *xs,
                         const float *xm)
{
    const __m512i m4 = _mm512_set1_epi8(0x0f), one = _mm512_set1_epi8(1);
    size_t bs = five ? 176 : 144;
    __m512 acc = _mm512_setzero_ps();
    float mins = 0.f;
    for (int b = 0; b < cols / 256; ++b) {
        const uint8_t *blk = w + (size_t)b * bs;
        float d = kq_h(blk), dm = kq_h(blk + 2);
        const uint8_t *qs = blk + (five ? 48 : 16);
        const float *xsb = xs + b * 8, *xmb = xm + b * 16;
        float f[8];
        for (int j = 0; j < 8; ++j) {
            int sc, m;
            kq_scale_min(blk + 4, j, &sc, &m);
            f[j] = d * (float)sc * xsb[j];
            mins += dm * (float)m * xsb[j] * (xmb[2 * j] + xmb[2 * j + 1]);
        }
        __m512i qhv = five ? _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 16)))
                           : _mm512_setzero_si512();
        for (int p = 0; p < 2; ++p) {
            /* 64 bytes: parts 4p (low 4 bits of the first 32 bytes), 4p + 1
             * (their high 4 bits), 4p + 2 and 4p + 3 (the next 32 bytes). */
            __m512i wv = _mm512_loadu_si512((const void *)(qs + 64 * p));
            __m512i lo = _mm512_and_si512(wv, m4);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(wv, 4), m4);
            if (five) {
                __m512i c = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(4 * p),
                                                    _mm512_set1_epi16(4 * p + 2));
                __m512i c1 = _mm512_add_epi16(c, _mm512_set1_epi16(1));
                lo = _mm512_or_si512(lo, _mm512_slli_epi16(
                    _mm512_and_si512(_mm512_srlv_epi16(qhv, c), one), 4));
                hi = _mm512_or_si512(hi, _mm512_slli_epi16(
                    _mm512_and_si512(_mm512_srlv_epi16(qhv, c1), one), 4));
            }
            const int8_t *xb = xq + (size_t)b * 256 + 128 * p;
            __m512i ia = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo, kq_two256(xb, xb + 64));
            __m512i ib = _mm512_dpbusd_epi32(_mm512_setzero_si512(), hi,
                                             kq_two256(xb + 32, xb + 96));
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia),
                                  _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(f[4 * p]),
                                                       _mm512_set1_ps(f[4 * p + 2])), acc);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ib),
                                  _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(f[4 * p + 1]),
                                                       _mm512_set1_ps(f[4 * p + 3])), acc);
        }
    }
    return _mm512_reduce_add_ps(acc) - mins;
}

static float kq_dot_q6k(const uint8_t *w, int cols, const int8_t *xq, const float *xs,
                        const float *xm)
{
    const __m512i m4 = _mm512_set1_epi8(0x0f), three = _mm512_set1_epi8(3);
    const __m512i cA = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(0),
                                               _mm512_set1_epi16(2));
    const __m512i cB = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(4),
                                               _mm512_set1_epi16(6));
    /* lane i of a product of 64 values belongs to the part of 16 i / 4 */
    const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i iB = _mm512_add_epi32(iA, _mm512_set1_epi32(4));
    const __m512i i8 = _mm512_set1_epi32(8);
    __m512 acc = _mm512_setzero_ps(), macc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
        const uint8_t *blk = w + (size_t)b * 210;
        float d = kq_h(blk + 208);
        const int8_t *sc = (const int8_t *)(blk + 192);
        float fs[16];
        for (int g = 0; g < 16; ++g) {
            fs[g] = d * (float)sc[g] * xs[b * 8 + g / 2];
        }
        __m512 fv = _mm512_loadu_ps(fs);
        macc = _mm512_fmadd_ps(fv, _mm512_loadu_ps(xm + b * 16), macc);
        for (int h = 0; h < 2; ++h) {
            __m512i wv = _mm512_loadu_si512((const void *)(blk + 64 * h));
            __m512i qhv = _mm512_broadcast_i64x4(
                _mm256_loadu_si256((const __m256i *)(blk + 128 + 32 * h)));
            __m512i A = _mm512_or_si512(_mm512_and_si512(wv, m4), _mm512_slli_epi16(
                _mm512_and_si512(_mm512_srlv_epi16(qhv, cA), three), 4));
            __m512i B = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(wv, 4), m4),
                _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(qhv, cB), three), 4));
            const int8_t *xb = xq + (size_t)b * 256 + 128 * h;
            __m512i pa = _mm512_dpbusd_epi32(_mm512_setzero_si512(), A,
                                             _mm512_loadu_si512((const void *)xb));
            __m512i pb = _mm512_dpbusd_epi32(_mm512_setzero_si512(), B,
                                             _mm512_loadu_si512((const void *)(xb + 64)));
            __m512i ha = h ? _mm512_add_epi32(iA, i8) : iA, hb = h ? _mm512_add_epi32(iB, i8) : iB;
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pa), _mm512_permutexvar_ps(ha, fv), acc);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(pb), _mm512_permutexvar_ps(hb, fv), acc);
        }
    }
    return _mm512_reduce_add_ps(acc) - 32.f * _mm512_reduce_add_ps(macc);
}

static float kq_dot_f32(const float *w, int cols, const float *x)
{
    __m512 acc = _mm512_setzero_ps();
    for (int c = 0; c < cols; c += 16) {
        acc = _mm512_fmadd_ps(_mm512_loadu_ps(w + c), _mm512_loadu_ps(x + c), acc);
    }
    return _mm512_reduce_add_ps(acc);
}
#endif

/* One row of type type on one token: xq, xs, xm (quantized), or x for F32. */
static float kq_dot1(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                     const float *xm, const float *x)
{
#if defined(__AVX512VNNI__)
    switch (type) {
    case KQ_F32: return kq_dot_f32((const float *)w, cols, x);
    case KQ_Q8_0: return kq_dot_q8_0(w, cols, xq, xs);
    case KQ_Q4_K: return kq_dot_q45k(w, cols, 0, xq, xs, xm);
    case KQ_Q5_K: return kq_dot_q45k(w, cols, 1, xq, xs, xm);
    case KQ_Q6_K: return kq_dot_q6k(w, cols, xq, xs, xm);
    }
    return 0.f;
#else
    (void)xm;
    if (type == KQ_F32) {
        float s = 0.f;
        for (int c = 0; c < cols; ++c) {
            s += ((const float *)w)[c] * x[c];
        }
        return s;
    }
    int bv = type == KQ_Q8_0 ? 32 : 256;
    float v[256], s = 0.f;
    for (int b = 0; b < cols / bv; ++b) {
        kq_block_values(type, w, b, v);
        for (int i = 0; i < bv; ++i) {
            int c = b * bv + i;
            s += v[i] * xs[c / 32] * (float)xq[c];
        }
    }
    return s;
#endif
}

/* One row on n tokens. out[j * ostride] gets token j. */
static void kq_row(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                   const float *xm, const float *x, int n, float *out, size_t ostride)
{
    for (int j = 0; j < n; ++j) {
        out[(size_t)j * ostride] = kq_dot1(w, type, cols, xq + (size_t)j * cols,
                                           xs + (size_t)j * (cols / 32),
                                           xm + (size_t)j * (cols / 16),
                                           x ? x + (size_t)j * cols : NULL);
    }
}

/* out (t x rows) = x W^T, inside a parallel region. x is the float input
 * (for F32), and xq, xs, xm its quantization (kq_quant_body). */
static void kq_linear_body(const uint8_t *w, int type, int rows, int cols, const int8_t *xq,
                           const float *xs, const float *xm, const float *x, int t, float *out)
{
    size_t rb = kq_row_bytes(type, cols);
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; ++r) {
        kq_row(w + (size_t)r * rb, type, cols, xq, xs, xm, x, t, out + r, (size_t)rows);
    }
}

void kq_linear(const uint8_t *w, int type, int rows, int cols, const int8_t *xq, const float *xs,
               const float *xm, const float *x, int t, float *out)
{
    #pragma omp parallel
    kq_linear_body(w, type, rows, cols, xq, xs, xm, x, t, out);
}

/* The float values of rows ids of a matrix (the embeddings). */
void kq_rows(const uint8_t *w, int type, int cols, const int64_t *ids, int n, float *out)
{
    size_t rb = kq_row_bytes(type, cols);
    int bv = type == KQ_Q8_0 ? 32 : 256;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const uint8_t *row = w + (size_t)ids[i] * rb;
        if (type == KQ_F32) {
            memcpy(out + (size_t)i * cols, row, (size_t)cols * 4);
            continue;
        }
        for (int b = 0; b < cols / bv; ++b) {
            kq_block_values(type, row, b, out + (size_t)i * cols + (size_t)b * bv);
        }
    }
}

/* The size of the scratch of kq_moe_body, in bytes. */
size_t kq_moe_scratch(int t, int k, int experts, int hidden, int inner)
{
    size_t P = (size_t)t * (k + 1);
    size_t n = 0;
    n += (size_t)(experts + 2) * 4 * 3 + 64;           /* counts, starts, used */
    n += P * 4 * 2;                                    /* pair token, pair of */
    n += P * hidden + P * (hidden / 32) * 4 + P * (hidden / 16) * 4;
    n += P * 2 * inner * 4;                            /* gate and up */
    n += P * inner + P * (inner / 32) * 4 + P * (inner / 16) * 4;
    n += P * hidden * 4;                               /* down outputs */
    return n + 64 * 16;
}

/* One matrix of the MoE: data and type. */
typedef struct {
    const uint8_t *w;
    int type;
} kq_mat;

static inline kq_mat kq_mat_of(const int64_t *d)
{
    kq_mat m;
    m.w = (const uint8_t *)(intptr_t)d[0];
    m.type = (int)d[1];
    return m;
}

/* The experts of t tokens in the GGUF formats, as ma_moe_body. h is
 * quantized (hq, hs, hm; kq_quant_body). mats has 6 descriptors (w, type):
 * gate, up, down of the stacked experts, then of the shared expert (w = 0
 * for none). The weight of the shared expert is sigmoid(shared_logit). */
static void kq_moe_body(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                        const float *val, int t, int k, int experts, const int64_t *mats,
                        const float *shared_logit, int hidden, int inner, uint8_t *scratch,
                        float *out)
{
    kq_mat G = kq_mat_of(mats), U = kq_mat_of(mats + 2), D = kq_mat_of(mats + 4);
    kq_mat SG = kq_mat_of(mats + 6), SU = kq_mat_of(mats + 8), SD = kq_mat_of(mats + 10);
    int shared = SG.w != NULL;
    int ne = experts + shared;
    int P = t * k + (shared ? t : 0);
    int nph = hidden / 32, npi = inner / 32;
    uint8_t *p = scratch;
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P * 4);
    int *nused_p = (int *)ma_take(&p, 64);
    int8_t *xq = (int8_t *)ma_take(&p, (size_t)P * hidden);
    float *xs = (float *)ma_take(&p, (size_t)P * nph * 4);
    float *xm = (float *)ma_take(&p, (size_t)P * nph * 2 * 4);
    float *act = (float *)ma_take(&p, (size_t)P * 2 * inner * 4);
    int8_t *aq = (int8_t *)ma_take(&p, (size_t)P * inner);
    float *as = (float *)ma_take(&p, (size_t)P * npi * 4);
    float *am = (float *)ma_take(&p, (size_t)P * npi * 2 * 4);
    float *de = (float *)ma_take(&p, (size_t)P * hidden * 4);
    moe_sort_pairs(ids, t, k, experts, shared, cnt, start, used, pair_tok, pair_of, nused_p);
    int nu = *nused_p;
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        int j = pair_tok[q];
        memcpy(xq + (size_t)q * hidden, hq + (size_t)j * hidden, (size_t)hidden);
        memcpy(xs + (size_t)q * nph, hs + (size_t)j * nph, (size_t)nph * 4);
        memcpy(xm + (size_t)q * 2 * nph, hm + (size_t)j * 2 * nph, (size_t)nph * 8);
    }
    /* Gate and up: 2 * inner rows of each used expert, in tasks of 4 rows. */
    #pragma omp for schedule(static)
    for (int x = 0; x < nu * 2 * inner / 4; ++x) {
        int e = used[x / (2 * inner / 4)], rr = (x % (2 * inner / 4)) * 4, isup = rr >= inner;
        int r = rr % inner;
        kq_mat m = e < experts ? (isup ? U : G) : (isup ? SU : SG);
        size_t rb = kq_row_bytes(m.type, hidden);
        const uint8_t *w = m.w + (e < experts ? (size_t)e * inner * rb : 0);
        int s0 = start[e];
        int n = (e + 1 < ne ? start[e + 1] : P) - s0;
        for (int i = 0; i < 4; ++i) {
            kq_row(w + (size_t)(r + i) * rb, m.type, hidden, xq + (size_t)s0 * hidden,
                   xs + (size_t)s0 * nph, xm + (size_t)s0 * 2 * nph, NULL, n,
                   act + (size_t)s0 * 2 * inner + rr + i, (size_t)2 * inner);
        }
    }
    /* silu(gate) * up, and its quantization for down. */
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        float *a = act + (size_t)q * 2 * inner;
        for (int i = 0; i < inner; ++i) {
            a[i] = ma_silu(a[i]) * a[inner + i];
        }
        for (int g = 0; g < npi; ++g) {
            kq_quant_part(a, g, aq + (size_t)q * inner, as + (size_t)q * npi,
                          am + (size_t)q * 2 * npi);
        }
    }
    /* Down: hidden rows of each used expert, in tasks of 4 rows. */
    #pragma omp for schedule(static)
    for (int x = 0; x < nu * hidden / 4; ++x) {
        int e = used[x / (hidden / 4)], r = (x % (hidden / 4)) * 4;
        kq_mat m = e < experts ? D : SD;
        size_t rb = kq_row_bytes(m.type, inner);
        const uint8_t *w = m.w + (e < experts ? (size_t)e * hidden * rb : 0);
        int s0 = start[e];
        int n = (e + 1 < ne ? start[e + 1] : P) - s0;
        for (int i = 0; i < 4; ++i) {
            kq_row(w + (size_t)(r + i) * rb, m.type, inner, aq + (size_t)s0 * inner,
                   as + (size_t)s0 * npi, am + (size_t)s0 * 2 * npi, NULL, n,
                   de + (size_t)s0 * hidden + r + i, (size_t)hidden);
        }
    }
    moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
}

void kq_moe(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
            const float *val, int t, int k, int experts, const int64_t *mats,
            const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out);
}
