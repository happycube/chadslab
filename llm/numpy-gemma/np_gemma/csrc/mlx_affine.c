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
#define MA_TB 128        /* the tokens of a block of the prompt pass */

/* Quantize t rows of x (cols values each, cols % 128 == 0 for 4 bits). xq
 * has cols bytes for each row, in the order of the bits (see the top).
 * xs and xsum have cols / 64 values for each row. */
/* Quantize group g of one row of x for bits: see the top. */
static void ma_quant_group(const float *xr, int g, int bits, int8_t *qr, float *xs, float *xsum)
{
    const float *v = xr + g * MA_G;
    float m = 0.f;
    for (int i = 0; i < MA_G; ++i) {
        float a = fabsf(v[i]);
        m = a > m ? a : m;
    }
    float sc = m / 127.f, inv = m > 0.f ? 127.f / m : 0.f;
    int32_t qs = 0;
    xs[g] = sc;
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
     * exactly xq . w. The weights are s q + b with q >= 0, so s q is far
     * from zero mean, and the error of xq cancels only with the bias term
     * of the same xq. */
    xsum[g] = sc * (float)qs;
}

static void ma_quant_rows(const float *x, int t, int cols, int bits, int8_t *xq, float *xs,
                          float *xsum)
{
    int ng = cols / MA_G;
    for (int r = 0; r < t; ++r) {
        for (int g = 0; g < ng; ++g) {
            ma_quant_group(x + (size_t)r * cols, g, bits, xq + (size_t)r * cols,
                           xs + (size_t)r * ng, xsum + (size_t)r * ng);
        }
    }
}

void ma_quant_x(const float *x, int t, int cols, int bits, int8_t *xq, float *xs, float *xsum)
{
    ma_quant_rows(x, t, cols, bits, xq, xs, xsum);
}

/* The quantization of t rows of x for 4 bits (xq4) and for 8 bits (xq8),
 * inside a parallel region. A null xq4 or xq8 skips that order. xs and xsum
 * are the same for both. */
static void ma_quant_body(const float *x, int t, int cols, int8_t *xq4, int8_t *xq8, float *xs,
                          float *xsum)
{
    int ng = cols / MA_G;
    /* One group for each step, so one row uses all the threads. */
    #pragma omp for schedule(static)
    for (int x2 = 0; x2 < t * ng; ++x2) {
        int r = x2 / ng, g = x2 % ng;
        const float *xr = x + (size_t)r * cols;
        if (xq4 != NULL) {
            ma_quant_group(xr, g, 4, xq4 + (size_t)r * cols, xs + (size_t)r * ng,
                           xsum + (size_t)r * ng);
        }
        if (xq8 != NULL) {
            ma_quant_group(xr, g, 8, xq8 + (size_t)r * cols, xs + (size_t)r * ng,
                           xsum + (size_t)r * ng);
        }
    }
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

#if defined(__AVX512VNNI__)
/* One row on one token (the decode step). The scales of the row times the
 * scales of x, and the bias term, come 16 groups at a time. */
static inline float ma_row_dot1(const uint8_t *wb, const uint16_t *s, const uint16_t *b, int bits,
                                int cols, const int8_t *xq, const float *xs, const float *xsum)
{
    int ng = cols / MA_G;
    float sc[256];
    __m512 bacc = _mm512_setzero_ps();
    for (int g0 = 0; g0 < ng; g0 += 16) {
        __mmask16 m = ng - g0 >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << (ng - g0)) - 1);
        __m512 sv = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, s + g0)), 16));
        __m512 bv = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, b + g0)), 16));
        _mm512_mask_storeu_ps(sc + g0, m, _mm512_mul_ps(sv, _mm512_maskz_loadu_ps(m, xs + g0)));
        bacc = _mm512_fmadd_ps(bv, _mm512_maskz_loadu_ps(m, xsum + g0), bacc);
    }
    /* One sum in the order of the groups: the tiles (ma_tile4) add in the
     * same order, so a token gives the same bits alone and in a group. An
     * MTP verify group then gives the values of the plain decode. */
    __m512 a0 = _mm512_setzero_ps();
    if (bits == 8) {
        for (int g = 0; g < ng; ++g) {
            __m512i w0 = _mm512_loadu_si512((const void *)(wb + (size_t)g * MA_G));
            __m512i i0 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w0,
                                             _mm512_loadu_si512((const void *)(xq + g * MA_G)));
            a0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i0), _mm512_set1_ps(sc[g]), a0);
        }
    } else {
        const __m512i m4 = _mm512_set1_epi8(0x0f);
        /* The scale of the pair: 8 lanes of group 2p, 8 lanes of group 2p + 1. */
        for (int p = 0; p < ng / 2; ++p) {
            __m512i wv = _mm512_loadu_si512((const void *)(wb + (size_t)p * 64));
            __m512i lo = _mm512_and_si512(wv, m4);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(wv, 4), m4);
            const int8_t *xp = xq + (size_t)p * 128;
            __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), lo,
                                             _mm512_loadu_si512((const void *)xp));
            is = _mm512_dpbusd_epi32(is, hi, _mm512_loadu_si512((const void *)(xp + 64)));
            __m512 pv = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sc[2 * p]),
                                             _mm512_set1_ps(sc[2 * p + 1]));
            a0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), pv, a0);
        }
    }
    return _mm512_reduce_add_ps(a0) + _mm512_reduce_add_ps(bacc);
}
#endif

#if defined(__AVX512VNNI__)
/* A tile of 4 rows by 4 tokens (the prompt pass, a group of an expert).
 * wr[i] are the rows (ng groups each), with their scales sr[i] and biases
 * br[i]. The tokens are rows of xq (cols bytes each), with xs and xsum
 * (ng values each). out[i * ostr_r + j * ostr_t] gets row i, token j.
 *
 * First the scales of each (row, token) pair: sc = s * xs, 16 groups at a
 * time. Then each load of x serves the 4 rows. For 4 bits, the scale of a
 * pair of groups (8 lanes each) comes from permutexvar of 16 scales. */
static void ma_tile4(const uint8_t *const wr[4], const uint16_t *const sr[4],
                     const uint16_t *const br[4], int bits, int cols, const int8_t *xq,
                     const float *xs, const float *xsum, int nt, float *out, size_t ostr_r,
                     size_t ostr_t)
{
    int ng = cols / MA_G;
    /* Always 4 tokens: a short tile repeats its last token, so the loops
     * have fixed bounds and the 16 sums stay in registers. */
    const int8_t *xr[4];
    const float *xsr[4], *xmr[4];
    for (int j = 0; j < 4; ++j) {
        int jj = j < nt ? j : nt - 1;
        xr[j] = xq + (size_t)jj * cols;
        xsr[j] = xs + (size_t)jj * ng;
        xmr[j] = xsum + (size_t)jj * ng;
    }
    float sc[4][4][256];
    float bias[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            __m512 bacc = _mm512_setzero_ps();
            for (int g0 = 0; g0 < ng; g0 += 16) {
                __mmask16 m = ng - g0 >= 16 ? (__mmask16)0xffff
                                            : (__mmask16)((1u << (ng - g0)) - 1);
                __m512 sv = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, sr[i] + g0)), 16));
                __m512 bv = _mm512_castsi512_ps(_mm512_slli_epi32(
                    _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(m, br[i] + g0)), 16));
                _mm512_mask_storeu_ps(sc[i][j] + g0, m,
                                      _mm512_mul_ps(sv, _mm512_maskz_loadu_ps(m, xsr[j] + g0)));
                bacc = _mm512_fmadd_ps(bv, _mm512_maskz_loadu_ps(m, xmr[j] + g0), bacc);
            }
            bias[i][j] = _mm512_reduce_add_ps(bacc);
        }
    }
    __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
    __m512 a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512 a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    __m512 a30 = a00, a31 = a00, a32 = a00, a33 = a00;
#define MA_ACC8(A, I, J, W, X, G) \
    A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(_mm512_setzero_si512(), W, X)), \
                        _mm512_set1_ps(sc[I][J][G]), A)
#define MA_ACC4(A, I, J, LO, HI, EV, OD) \
    A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32( \
            _mm512_dpbusd_epi32(_mm512_setzero_si512(), LO, EV), HI, OD)), \
        _mm512_permutexvar_ps(pq, _mm512_loadu_ps(sc[I][J] + g16)), A)
    if (bits == 8) {
        for (int g = 0; g < ng; ++g) {
            size_t o = (size_t)g * MA_G;
            __m512i w0 = _mm512_loadu_si512((const void *)(wr[0] + o));
            __m512i w1 = _mm512_loadu_si512((const void *)(wr[1] + o));
            __m512i w2 = _mm512_loadu_si512((const void *)(wr[2] + o));
            __m512i w3 = _mm512_loadu_si512((const void *)(wr[3] + o));
            __m512i x0 = _mm512_loadu_si512((const void *)(xr[0] + o));
            __m512i x1 = _mm512_loadu_si512((const void *)(xr[1] + o));
            __m512i x2 = _mm512_loadu_si512((const void *)(xr[2] + o));
            __m512i x3 = _mm512_loadu_si512((const void *)(xr[3] + o));
            MA_ACC8(a00, 0, 0, w0, x0, g); MA_ACC8(a01, 0, 1, w0, x1, g);
            MA_ACC8(a02, 0, 2, w0, x2, g); MA_ACC8(a03, 0, 3, w0, x3, g);
            MA_ACC8(a10, 1, 0, w1, x0, g); MA_ACC8(a11, 1, 1, w1, x1, g);
            MA_ACC8(a12, 1, 2, w1, x2, g); MA_ACC8(a13, 1, 3, w1, x3, g);
            MA_ACC8(a20, 2, 0, w2, x0, g); MA_ACC8(a21, 2, 1, w2, x1, g);
            MA_ACC8(a22, 2, 2, w2, x2, g); MA_ACC8(a23, 2, 3, w2, x3, g);
            MA_ACC8(a30, 3, 0, w3, x0, g); MA_ACC8(a31, 3, 1, w3, x1, g);
            MA_ACC8(a32, 3, 2, w3, x2, g); MA_ACC8(a33, 3, 3, w3, x3, g);
        }
    } else {
        const __m512i m4 = _mm512_set1_epi8(0x0f);
        for (int p = 0; p < ng / 2; ++p) {
            size_t o = (size_t)p * 64, ox = (size_t)p * 128;
            int g16 = (2 * p) & ~15, q = p % 8;
            __m512i pq = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(2 * q),
                                                 _mm512_set1_epi32(2 * q + 1));
            __m512i v0 = _mm512_loadu_si512((const void *)(wr[0] + o));
            __m512i v1 = _mm512_loadu_si512((const void *)(wr[1] + o));
            __m512i v2 = _mm512_loadu_si512((const void *)(wr[2] + o));
            __m512i v3 = _mm512_loadu_si512((const void *)(wr[3] + o));
            __m512i l0 = _mm512_and_si512(v0, m4), h0 = _mm512_and_si512(_mm512_srli_epi16(v0, 4), m4);
            __m512i l1 = _mm512_and_si512(v1, m4), h1 = _mm512_and_si512(_mm512_srli_epi16(v1, 4), m4);
            __m512i l2 = _mm512_and_si512(v2, m4), h2 = _mm512_and_si512(_mm512_srli_epi16(v2, 4), m4);
            __m512i l3 = _mm512_and_si512(v3, m4), h3 = _mm512_and_si512(_mm512_srli_epi16(v3, 4), m4);
            for (int j = 0; j < 4; ++j) {
                __m512i ev = _mm512_loadu_si512((const void *)(xr[j] + ox));
                __m512i od = _mm512_loadu_si512((const void *)(xr[j] + ox + 64));
                if (j == 0) {
                    MA_ACC4(a00, 0, 0, l0, h0, ev, od); MA_ACC4(a10, 1, 0, l1, h1, ev, od);
                    MA_ACC4(a20, 2, 0, l2, h2, ev, od); MA_ACC4(a30, 3, 0, l3, h3, ev, od);
                } else if (j == 1) {
                    MA_ACC4(a01, 0, 1, l0, h0, ev, od); MA_ACC4(a11, 1, 1, l1, h1, ev, od);
                    MA_ACC4(a21, 2, 1, l2, h2, ev, od); MA_ACC4(a31, 3, 1, l3, h3, ev, od);
                } else if (j == 2) {
                    MA_ACC4(a02, 0, 2, l0, h0, ev, od); MA_ACC4(a12, 1, 2, l1, h1, ev, od);
                    MA_ACC4(a22, 2, 2, l2, h2, ev, od); MA_ACC4(a32, 3, 2, l3, h3, ev, od);
                } else {
                    MA_ACC4(a03, 0, 3, l0, h0, ev, od); MA_ACC4(a13, 1, 3, l1, h1, ev, od);
                    MA_ACC4(a23, 2, 3, l2, h2, ev, od); MA_ACC4(a33, 3, 3, l3, h3, ev, od);
                }
            }
        }
    }
#undef MA_ACC8
#undef MA_ACC4
    __m512 acc[4][4] = {{a00, a01, a02, a03}, {a10, a11, a12, a13},
                        {a20, a21, a22, a23}, {a30, a31, a32, a33}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < nt; ++j) {
            out[(size_t)i * ostr_r + (size_t)j * ostr_t] =
                _mm512_reduce_add_ps(acc[i][j]) + bias[i][j];
        }
    }
}

/* 4 rows (r0 .. r0 + 3) of a matrix on n tokens, in tiles of 4 tokens. */
static void ma_rows4(const uint32_t *w, const uint16_t *s, const uint16_t *b, int bits, int r0,
                     int cols, const int8_t *xq, const float *xs, const float *xsum, int n,
                     float *out, size_t ostr_r, size_t ostr_t)
{
    int ng = cols / MA_G;
    size_t wrow = (size_t)cols * bits / 32;
    const uint8_t *wr[4];
    const uint16_t *sr[4], *brr[4];
    for (int i = 0; i < 4; ++i) {
        wr[i] = (const uint8_t *)(w + (size_t)(r0 + i) * wrow);
        sr[i] = s + (size_t)(r0 + i) * ng;
        brr[i] = b + (size_t)(r0 + i) * ng;
    }
    for (int j0 = 0; j0 < n; j0 += 4) {
        int nt = n - j0 < 4 ? n - j0 : 4;
        ma_tile4(wr, sr, brr, bits, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * ng,
                 xsum + (size_t)j0 * ng, nt, out + (size_t)j0 * ostr_t, ostr_r, ostr_t);
    }
}
#endif

/* One row of weights (w, s, b) on t rows of quantized x. out[j] gets the
 * product of token j. */
static void ma_row_dot(const uint32_t *w, const uint16_t *s, const uint16_t *b, int bits, int cols,
                    const int8_t *xq, const float *xs, const float *xsum, int t, float *out,
                    size_t ostride)
{
    int ng = cols / MA_G;
    const uint8_t *wb = (const uint8_t *)w;
#if defined(__AVX512VNNI__)
    if (ng <= 256) {
        /* A few tokens: each one alone, for the same bits as the tiles. */
        for (int j = 0; j < t; ++j) {
            out[(size_t)j * ostride] = ma_row_dot1(wb, s, b, bits, cols, xq + (size_t)j * cols,
                                                  xs + (size_t)j * ng, xsum + (size_t)j * ng);
        }
        return;
    }
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
static void ma_linear_body(const uint32_t *w, const uint16_t *s, const uint16_t *b, int bits,
                           int rows, int cols, const int8_t *xq, const float *xs,
                           const float *xsum, int t, float *out)
{
    int ng = cols / MA_G;
    size_t wrow = (size_t)cols * bits / 32;
#if defined(__AVX512VNNI__)
    if (t >= 4 && rows % 4 == 0 && ng <= 256) {
        /* The tiles of 4 rows by 4 tokens (the prompt pass), in blocks of
         * MA_TB tokens: the x of a block stays in L2, and with the static
         * split each thread keeps its rows for all the blocks. */
        for (int j0 = 0; j0 < t; j0 += MA_TB) {
            int nb = t - j0 < MA_TB ? t - j0 : MA_TB;
            #pragma omp for schedule(static) nowait
            for (int r4 = 0; r4 < rows / 4; ++r4) {
                ma_rows4(w, s, b, bits, 4 * r4, cols, xq + (size_t)j0 * cols,
                         xs + (size_t)j0 * ng, xsum + (size_t)j0 * ng, nb,
                         out + (size_t)j0 * rows + 4 * r4, 1, (size_t)rows);
            }
        }
        #pragma omp barrier
        return;
    }
#endif
    /* ma_row_dot keeps MA_MAX_T sums: more tokens go in blocks. */
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; ++r) {
        for (int j0 = 0; j0 < t; j0 += MA_MAX_T) {
            int tj = t - j0 < MA_MAX_T ? t - j0 : MA_MAX_T;
            ma_row_dot(w + (size_t)r * wrow, s + (size_t)r * ng, b + (size_t)r * ng, bits, cols,
                       xq + (size_t)j0 * cols, xs + (size_t)j0 * ng, xsum + (size_t)j0 * ng, tj,
                       out + (size_t)j0 * rows + r, (size_t)rows);
        }
    }
}

void ma_linear(const uint32_t *w, const uint16_t *s, const uint16_t *b, int bits, int rows,
               int cols, const int8_t *xq, const float *xs, const float *xsum, int t, float *out)
{
    #pragma omp parallel
    ma_linear_body(w, s, b, bits, rows, cols, xq, xs, xsum, t, out);
}

static inline float ma_silu(float v)
{
    return v / (1.f + expf(-v));
}

/* One matrix: data, scales, biases, bits. */
typedef struct {
    const uint32_t *w;
    const uint16_t *s, *b;
    int bits;
} ma_mat;

static inline ma_mat ma_mat_of(const int64_t *d)
{
    ma_mat m;
    m.w = (const uint32_t *)(intptr_t)d[0];
    m.s = (const uint16_t *)(intptr_t)d[1];
    m.b = (const uint16_t *)(intptr_t)d[2];
    m.bits = (int)d[3];
    return m;
}

/* Expert e of a stack of matrices of rows x cols. */
static inline ma_mat ma_expert(ma_mat m, int e, int rows, int cols)
{
    size_t ng = (size_t)rows * (cols / MA_G);
    m.w += (size_t)e * rows * cols / 32 * m.bits;
    m.s += (size_t)e * ng;
    m.b += (size_t)e * ng;
    return m;
}

/* One row of m on n rows of quantized x, in blocks of MA_MAX_T. */
static void ma_rows_dot(ma_mat m, int r, int cols, const int8_t *xq, const float *xs,
                        const float *xsum, int n, float *out, size_t ostride)
{
    int ng = cols / MA_G;
    size_t wrow = (size_t)cols * m.bits / 32;
    for (int j0 = 0; j0 < n; j0 += MA_MAX_T) {
        int tj = n - j0 < MA_MAX_T ? n - j0 : MA_MAX_T;
        ma_row_dot(m.w + (size_t)r * wrow, m.s + (size_t)r * ng, m.b + (size_t)r * ng, m.bits,
                   cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * ng, xsum + (size_t)j0 * ng,
                   tj, out + (size_t)j0 * ostride, ostride);
    }
}

/* Rows r .. r + 3 of m on n rows of x: the tiles for 4 tokens or more, else
 * one row at a time. out[i + j * ostride] gets row r + i, token j. */
static void ma_rows_any(ma_mat m, int r, int cols, const int8_t *xq, const float *xs,
                        const float *xsum, int n, float *out, size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (n >= 4 && cols / MA_G <= 256) {
        ma_rows4(m.w, m.s, m.b, m.bits, r, cols, xq, xs, xsum, n, out, 1, ostride);
        return;
    }
#endif
    for (int i = 0; i < 4; ++i) {
        ma_rows_dot(m, r + i, cols, xq, xs, xsum, n, out + i, ostride);
    }
}

/* The size of the scratch of ma_moe_body, in bytes. */
size_t ma_moe_scratch(int t, int k, int experts, int hidden, int inner)
{
    size_t P = (size_t)t * (k + 1);
    size_t n = 0;
    n += (size_t)(experts + 2) * 4 * 4;               /* counts, starts, used */
    n += P * 4 * 3;                                    /* pair token, pair of, order */
    n += P * hidden * 2;                               /* gathered x, 4 and 8 bits */
    n += P * (hidden / MA_G) * 4 * 2;                  /* its scales and sums */
    n += P * 2 * inner * 4;                            /* gate and up */
    n += P * inner;                                    /* activations in int8 */
    n += P * (inner / MA_G) * 4 * 2;
    n += P * hidden * 4;                               /* down outputs */
    return n + 64 * 16;
}

static inline void *ma_take(uint8_t **p, size_t n)
{
    void *r = *p;
    *p += (n + 63) & ~(size_t)63;
    return r;
}

/* The experts of t tokens in the MLX affine format, with an optional shared
 * expert, inside a parallel region.
 *
 * h is quantized for 4 and 8 bits (hq4, hq8, hs, hsum; t rows of hidden).
 * ids and val (t x k) give the experts of each token and their weights.
 * mats has 6 descriptors (w, s, b, bits): gate, up, down of the stacked
 * experts, then gate, up, down of the shared expert (w = 0 for none).
 * shared_logit (t values, or null) gives the gate of the shared expert:
 * its weight is sigmoid(shared_logit), as in Qwen3.5.
 *
 * The pairs (token, expert) go in the order of the experts. Then each row of
 * an expert runs one time for all of its tokens: for one token that is the
 * decode step, for a group it reads each expert one time. out gets t rows. */
static void ma_moe_body(const int8_t *hq4, const int8_t *hq8, const float *hs, const float *hsum,
                        const int32_t *ids, const float *val, int t, int k, int experts,
                        const int64_t *mats, const float *shared_logit, int hidden, int inner,
                        uint8_t *scratch, float *out)
{
    ma_mat G = ma_mat_of(mats), U = ma_mat_of(mats + 4), D = ma_mat_of(mats + 8);
    ma_mat SG = ma_mat_of(mats + 12), SU = ma_mat_of(mats + 16), SD = ma_mat_of(mats + 20);
    int shared = SG.w != NULL;
    int ne = experts + shared;                         /* the shared expert is the last */
    int P = t * k + (shared ? t : 0);
    int ngh = hidden / MA_G, ngi = inner / MA_G;
    uint8_t *p = scratch;
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P * 4);  /* token slot -> sorted pair */
    int *nused_p = (int *)ma_take(&p, 64);
    int8_t *x4 = (int8_t *)ma_take(&p, (size_t)P * hidden);
    int8_t *x8 = (int8_t *)ma_take(&p, (size_t)P * hidden);
    float *xs = (float *)ma_take(&p, (size_t)P * ngh * 4);
    float *xm = (float *)ma_take(&p, (size_t)P * ngh * 4);
    float *act = (float *)ma_take(&p, (size_t)P * 2 * inner * 4);
    int8_t *aq = (int8_t *)ma_take(&p, (size_t)P * inner);
    float *as = (float *)ma_take(&p, (size_t)P * ngi * 4);
    float *am = (float *)ma_take(&p, (size_t)P * ngi * 4);
    float *de = (float *)ma_take(&p, (size_t)P * hidden * 4);
    moe_sort_pairs(ids, t, k, experts, shared, cnt, start, used, pair_tok, pair_of, nused_p);
    int nu = *nused_p;
    /* The input rows of the pairs, in the sorted order. */
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        int j = pair_tok[q];
        memcpy(x4 + (size_t)q * hidden, hq4 + (size_t)j * hidden, (size_t)hidden);
        memcpy(x8 + (size_t)q * hidden, hq8 + (size_t)j * hidden, (size_t)hidden);
        memcpy(xs + (size_t)q * ngh, hs + (size_t)j * ngh, (size_t)ngh * 4);
        memcpy(xm + (size_t)q * ngh, hsum + (size_t)j * ngh, (size_t)ngh * 4);
    }
    /* Gate and up: 2 * inner rows of each used expert, on its pairs, in
     * tasks of 4 rows. */
    #pragma omp for schedule(static)
    for (int x = 0; x < nu * 2 * inner / 4; ++x) {
        int e = used[x / (2 * inner / 4)], rr = (x % (2 * inner / 4)) * 4, isup = rr >= inner;
        int r = rr % inner;
        ma_mat m = e < experts ? ma_expert(isup ? U : G, e, inner, hidden) : (isup ? SU : SG);
        int s0 = start[e];
        int n = (e + 1 < ne ? start[e + 1] : P) - s0;
        ma_rows_any(m, r, hidden, (m.bits == 4 ? x4 : x8) + (size_t)s0 * hidden,
                    xs + (size_t)s0 * ngh, xm + (size_t)s0 * ngh, n,
                    act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
    }
    /* silu(gate) * up, and its quantization for down. */
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        float *a = act + (size_t)q * 2 * inner;
        for (int i = 0; i < inner; ++i) {
            a[i] = ma_silu(a[i]) * a[inner + i];
        }
        int bits = (shared && q >= start[experts]) ? SD.bits : D.bits;
        ma_quant_rows(a, 1, inner, bits, aq + (size_t)q * inner, as + (size_t)q * ngi,
                      am + (size_t)q * ngi);
    }
    /* Down: hidden rows of each used expert, in tasks of 4 rows. */
    #pragma omp for schedule(static)
    for (int x = 0; x < nu * hidden / 4; ++x) {
        int e = used[x / (hidden / 4)], r = (x % (hidden / 4)) * 4;
        ma_mat m = e < experts ? ma_expert(D, e, hidden, inner) : SD;
        int s0 = start[e];
        int n = (e + 1 < ne ? start[e + 1] : P) - s0;
        ma_rows_any(m, r, inner, aq + (size_t)s0 * inner, as + (size_t)s0 * ngi,
                    am + (size_t)s0 * ngi, n, de + (size_t)s0 * hidden + r, (size_t)hidden);
    }
    moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
}

void ma_moe(const int8_t *hq4, const int8_t *hq8, const float *hs, const float *hsum,
            const int32_t *ids, const float *val, int t, int k, int experts,
            const int64_t *mats, const float *shared_logit, int hidden, int inner,
            uint8_t *scratch, float *out)
{
    #pragma omp parallel
    ma_moe_body(hq4, hq8, hs, hsum, ids, val, t, k, experts, mats, shared_logit, hidden, inner,
                scratch, out);
}
