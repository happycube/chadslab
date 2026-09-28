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
 *     KQ_Q5_1  7   blocks of 32 (24 bytes): d, m (fp16), 32 fifth bits,
 *                  16 bytes of 4-bit values. w = d * q + m.
 *     KQ_IQ4_NL 20 blocks of 32 (18 bytes): d, 16 bytes of 4-bit codes into
 *                  a table of 16 values (only kq_rows: the n-gram table).
 *     KQ_Q6_K 14   blocks of 256 (210 bytes): 128 bytes of the low 4 bits,
 *                  64 bytes of the high 2 bits, 16 int8 scales (one for
 *                  each 16 values), d. w = d * sc * (q - 32).
 *     KQ_BF16 30   bfloat16 rows (the safetensors checkpoints).
 *     KQ_Q8X16 60  Q8_0 in groups of 16 rows (this runtime; kq_pack_q8x16 makes
 *                  it from Q8_0 for the CPU): for each group and each block
 *                  of 32 columns, 576 bytes: the 16 fp16 scales of the rows,
 *                  the 16 sums of the int8 values of the block of each row
 *                  (int16), then 8 steps of 64 bytes; step k holds values
 *                  4k to 4k + 3 of the 16 rows (row r in bytes 4r to 4r + 3).
 *                  A lane of vpdpbusd is a row: no sums across lanes, and an
 *                  int32 sum for each block (see kq_x16_body).
 *     KQ_NV4  51   NVFP4 in rows for this runtime (np_gemma/st_qwen4.py
 *                  repacks the checkpoint of NVIDIA ModelOpt). A row: the
 *                  4-bit E2M1 codes (cols / 2 bytes; block b of 32 values in
 *                  bytes 16b .. 16b + 15, value j in the low 4 bits of byte
 *                  j, value j + 16 in the high 4 bits), the E4M3 scales
 *                  (cols / 16 bytes; 2 for each block: values 0-15, 16-31),
 *                  the float32 scale of the matrix, and zeros to a multiple
 *                  of 16 bytes. So the codes of a row start at a multiple of
 *                  16 bytes, and the scales at a multiple of 4 (the loads of
 *                  the GPU; kq_nv4_* give the places). w = g * s *
 *                  e2m1(code); 2 e2m1 is an integer (0, 1, 2, 3, 4, 6, 8,
 *                  12 and the negatives).
 *     KQ_BF16X16 61, KQ_F32X16 62  bfloat16 or float32 in groups of 16 rows
 *                  (this runtime, only in memory; kq_pack_x16f makes them
 *                  for the CPU): for each column, the values of the 16 rows
 *                  (32 or 64 bytes). The rows go to a multiple of 16 with
 *                  zeros. A lane is a row, x is float32 (not int8), and a
 *                  step is one fma for each token (kq_x16f_body). Only for
 *                  64 rows or more (Qwen4CPU.KP): the smaller matrices stay
 *                  in rows, on tasks of a row and 16 tokens.
 *     KQ_NVX  53   NVFP4 in groups of 16 rows (this runtime; the GGUF of
 *                  scripts/convert_nvfp4_gguf.py; kq_nvx_pack). A group: 16
 *                  bytes (the float32 scale of the matrix, zeros), then for
 *                  each block of 32 columns 288 bytes: 8 steps of 32 bytes
 *                  and 32 E4M3 scales. Step s holds values 4s to 4s + 3 of
 *                  the 16 rows: byte 4r + u (r < 8) has value 4s + u of row
 *                  r in its low 4 bits, and that of row r + 8 in its high 4
 *                  bits. The scales: values 0-15 of rows 0 to 15, then
 *                  values 16-31. The CPU: a step is one vpdpbusd, a lane for
 *                  each row (kq_nvx_rows). The GPU: the 4 bytes of a lane
 *                  are a fragment of the tensor cores for 2 rows. A "row" is
 *                  cols / 32 * 18 + 1 bytes (a group is 16 of them).
 *
 * The products quantize x to int8 in its natural order, with one scale xs
 * for each 32 values and xm, xs times the sum of the int8 values of each 16
 * values
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
#define KQ_Q5_1 7
#define KQ_IQ4_NL 20
#define KQ_BF16 30
#define KQ_NV4 51
#define KQ_Q8X16 60
#define KQ_NVX 53
#define KQ_BF16X16 61
#define KQ_F32X16 62
#define KQ_NVX_BB 288

/* The bytes of one row of cols values. */
static inline size_t kq_row_bytes(int type, int cols)
{
    switch (type) {
    case KQ_F32: return (size_t)cols * 4;
    case KQ_Q8_0: return (size_t)cols / 32 * 34;
    case KQ_Q5_1: return (size_t)cols / 32 * 24;
    case KQ_IQ4_NL: return (size_t)cols / 32 * 18;
    case KQ_Q4_K: return (size_t)cols / 256 * 144;
    case KQ_Q5_K: return (size_t)cols / 256 * 176;
    case KQ_Q6_K: return (size_t)cols / 256 * 210;
    case KQ_BF16: return (size_t)cols * 2;
    case KQ_NV4: return ((size_t)cols / 2 + (size_t)cols / 16 + 4 + 15) / 16 * 16;
    case KQ_Q8X16: return (size_t)cols / 32 * 36;      /* a group of 16 rows: 16 times this */
    case KQ_NVX: return (size_t)cols / 32 * 18 + 1;    /* a group of 16 rows: 16 times this */
    case KQ_BF16X16: return (size_t)cols * 2;          /* a group: 16 times this */
    case KQ_F32X16: return (size_t)cols * 4;
    }
    return 0;
}

/* The places in a KQ_NV4 row: the codes of block b, its two scales, and the
 * scale of the matrix. */
static inline const uint8_t *kq_nv4_codes(const uint8_t *w, int b)
{
    return w + (size_t)16 * b;
}

static inline const uint8_t *kq_nv4_scales(const uint8_t *w, int cols, int b)
{
    return w + (size_t)cols / 2 + 2 * (size_t)b;
}

static inline float kq_nv4_g(const uint8_t *w, int cols)
{
    float g;
    memcpy(&g, w + (size_t)cols / 2 + (size_t)cols / 16, 4);
    return g;
}

/* Twice the value of an E2M1 code (NVFP4): an integer. */
static const int8_t kq_e2m1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

/* An FP8 E4M3 value (the scales of NVFP4). */
static inline float kq_e4m3(uint8_t b)
{
    int e = (b >> 3) & 15, m = b & 7;
    float v;
    if (e == 0) {
        v = (float)m * (1.f / 512.f);          /* m / 8 * 2^-6 */
    } else {
        uint32_t bits = ((uint32_t)(e - 7 + 127) << 23) | ((uint32_t)m << 20);
        memcpy(&v, &bits, 4);
    }
    return (b & 0x80) ? -v : v;
}

/* The E4M3 values of the 256 codes (made when the library loads). */
static float kq_e4m3_tab[256];

__attribute__((constructor)) static void kq_e4m3_init(void)
{
    for (int b = 0; b < 256; ++b) {
        kq_e4m3_tab[b] = kq_e4m3((uint8_t)b);
    }
}

/* The float values of row rin (0 to 15) of the KQ_NVX group at wg. */
static void kq_nvx_values(const uint8_t *wg, int rin, int cols, float *out)
{
    static const int8_t e2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    float g;
    memcpy(&g, wg, 4);
    int sh = rin < 8 ? 0 : 4, r8 = rin % 8;
    for (int b = 0; b < cols / 32; ++b) {
        const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
        for (int v = 0; v < 32; ++v) {
            int c = (blk[32 * (v / 4) + 4 * r8 + v % 4] >> sh) & 15;
            float sc = kq_e4m3_tab[blk[256 + 16 * (v / 16) + rin]];
            out[32 * b + v] = 0.5f * g * sc * (float)e2[c];
        }
    }
}

static inline float kq_bf16(uint16_t h)
{
    uint32_t bits = (uint32_t)h << 16;
    float v;
    memcpy(&v, &bits, 4);
    return v;
}

static inline float kq_h(const uint8_t *p)
{
    uint16_t v = (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
#if defined(__F16C__)
    return _cvtsh_ss(v);
#else
    return fp16_to_f32(v);
#endif
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
    /* The sums times the scale: the terms of the mins (Q4_K, Q5_K) and of
     * the offset 32 (Q6_K) are then products with the scales of the row. */
    xm[2 * g] = xs[g] * (float)s0;
    xm[2 * g + 1] = xs[g] * (float)s1;
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
static const int8_t kq_iq4nl_values[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                            1, 13, 25, 38, 53, 69, 89, 113};

/* The values of a block of 32 of a type of blocks of 32. */
static inline int kq_block32(int type)
{
    return type == KQ_Q8_0 || type == KQ_Q5_1 || type == KQ_IQ4_NL || type == KQ_BF16 ||
           type == KQ_NV4;
}

static void kq_block_values(int type, const uint8_t *row, int cols, int b, float *out)
{
    if (type == KQ_BF16) {
        const uint16_t *h = (const uint16_t *)row + (size_t)b * 32;
        for (int i = 0; i < 32; ++i) {
            out[i] = kq_bf16(h[i]);
        }
    } else if (type == KQ_NV4) {
        float g = kq_nv4_g(row, cols);
        const uint8_t *q = kq_nv4_codes(row, b), *sc = kq_nv4_scales(row, cols, b);
        float s0 = 0.5f * g * kq_e4m3(sc[0]), s1 = 0.5f * g * kq_e4m3(sc[1]);
        for (int j = 0; j < 16; ++j) {
            out[j] = s0 * (float)kq_e2m1x2[q[j] & 15];
            out[j + 16] = s1 * (float)kq_e2m1x2[q[j] >> 4];
        }
    } else if (type == KQ_Q5_1) {
        const uint8_t *blk = row + (size_t)b * 24;
        float d = kq_h(blk), m = kq_h(blk + 2);
        uint32_t qh;
        memcpy(&qh, blk + 4, 4);
        for (int j = 0; j < 16; ++j) {
            int lo = (blk[8 + j] & 15) | (((qh >> j) & 1) << 4);
            int hi = (blk[8 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
            out[j] = d * (float)lo + m;
            out[j + 16] = d * (float)hi + m;
        }
    } else if (type == KQ_IQ4_NL) {
        const uint8_t *blk = row + (size_t)b * 18;
        float d = kq_h(blk);
        for (int j = 0; j < 16; ++j) {
            out[j] = d * (float)kq_iq4nl_values[blk[2 + j] & 15];
            out[j + 16] = d * (float)kq_iq4nl_values[blk[2 + j] >> 4];
        }
    } else if (type == KQ_Q8_0) {
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
static inline __attribute__((always_inline)) __m512i kq_two256(const int8_t *a, const int8_t *b)
{
    return _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)a)),
                              _mm256_loadu_si256((const __m256i *)b), 1);
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

#if defined(__AVX512VNNI__)
/* The scales of one row that do not depend on x: ds gets d (Q8_0), d * sc
 * (Q4_K, Q5_K: one for each 32 values; Q6_K: one for each 16), and dm gets
 * dmin * m (Q4_K, Q5_K). */
static void kq_row_scales(const uint8_t *w, int type, int cols, float *ds, float *dm)
{
    if (type == KQ_Q8_0) {
        for (int i = 0; i < cols / 32; ++i) {
            ds[i] = kq_h(w + (size_t)i * 34);
        }
    } else if (type == KQ_Q5_1) {
        /* w = d q + m: the term m sum(x) is the mins term of Q4_K with -m. */
        for (int i = 0; i < cols / 32; ++i) {
            ds[i] = kq_h(w + (size_t)i * 24);
            dm[i] = -kq_h(w + (size_t)i * 24 + 2);
        }
    } else if (type == KQ_Q6_K) {
        for (int b = 0; b < cols / 256; ++b) {
            const uint8_t *blk = w + (size_t)b * 210;
            __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)(blk + 192))));
            _mm512_storeu_ps(ds + 16 * b, _mm512_mul_ps(_mm512_set1_ps(kq_h(blk + 208)), sc));
        }
    } else {
        size_t bs = type == KQ_Q5_K ? 176 : 144;
        for (int b = 0; b < cols / 256; ++b) {
            const uint8_t *blk = w + (size_t)b * bs;
            /* The 6-bit scales and mins with 32-bit masks (as ggml): the
             * bytes of u are the 8 scales, then the 8 mins. */
            uint32_t u[4];
            memcpy(u, blk + 4, 12);
            const uint32_t k1 = 0x3f3f3f3f, k2 = 0x0f0f0f0f, k3 = 0x03030303;
            u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
            uint32_t mins = u[1] & k1;
            u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
            u[2] = mins;
            u[0] &= k1;
            __m512 v = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)u)));
            __m512 dd = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(kq_h(blk)),
                                             _mm512_set1_ps(kq_h(blk + 2)));
            v = _mm512_mul_ps(dd, v);
            /* lanes 0 .. 7 to ds, 8 .. 15 to dm (masked stores) */
            _mm512_mask_storeu_ps(ds + 8 * b, 0x00ff, v);
            _mm512_mask_storeu_ps(dm + 8 * b - 8, 0xff00, v);
        }
    }
}

/* The scales of one row on one token: S gets the scale of each part (ds *
 * xs, in the order of the parts), and the return value is the term that
 * the product subtracts at the end: the mins of Q4_K and Q5_K (dm times
 * xm), 32 times the sums of Q6_K (ds times xm). kq_dot1 and the tiles both
 * use it, so they give the same bits. */
static float kq_prep(int type, int cols, const float *ds, const float *dm, const float *xs,
                     const float *xm, float *S)
{
    /* two lanes for each value of 8 (a scale of each 32 for 2 sums of 16) */
    const __m512i twice = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    if (type == KQ_Q6_K) {
        __m512 macc = _mm512_setzero_ps();
        for (int b = 0; b < cols / 256; ++b) {
            __m512 dv = _mm512_loadu_ps(ds + 16 * b);
            __m512 xv = _mm512_permutexvar_ps(twice, _mm512_maskz_loadu_ps(0xff, xs + 8 * b));
            _mm512_storeu_ps(S + 16 * b, _mm512_mul_ps(dv, xv));
            macc = _mm512_fmadd_ps(dv, _mm512_loadu_ps(xm + 16 * b), macc);
        }
        return 32.f * _mm512_reduce_add_ps(macc);
    }
    int np = cols / 32;
    for (int j = 0; j < np; j += 16) {
        __mmask16 m = np - j >= 16 ? (__mmask16)0xffff : (__mmask16)((1u << (np - j)) - 1);
        _mm512_mask_storeu_ps(S + j, m, _mm512_mul_ps(_mm512_maskz_loadu_ps(m, ds + j),
                                                      _mm512_maskz_loadu_ps(m, xs + j)));
    }
    if (type == KQ_Q8_0) {
        return 0.f;
    }
    __m512 macc = _mm512_setzero_ps();
    for (int j = 0; j < np; j += 8) {
        __mmask16 m = np - j >= 8 ? (__mmask16)0xffff : (__mmask16)((1u << (2 * (np - j))) - 1);
        __m512 dv = _mm512_permutexvar_ps(twice, _mm512_maskz_loadu_ps(np - j >= 8 ? (__mmask16)0xff : (__mmask16)((1u << (np - j)) - 1), dm + j));
        macc = _mm512_fmadd_ps(dv, _mm512_maskz_loadu_ps(m, xm + 2 * j), macc);
    }
    return _mm512_reduce_add_ps(macc);
}

/* The 4-bit (and fifth bit) values of part pair p of a Q4_K or Q5_K block,
 * in the lanes of kq_dot_q45k. */
static inline __attribute__((always_inline)) void kq_q45_values(const uint8_t *blk, int five, int p, __m512i *lo, __m512i *hi)
{
    const __m512i m4 = _mm512_set1_epi8(0x0f), one = _mm512_set1_epi8(1);
    const uint8_t *qs = blk + (five ? 48 : 16);
    __m512i wv = _mm512_loadu_si512((const void *)(qs + 64 * p));
    __m512i l = _mm512_and_si512(wv, m4), h = _mm512_and_si512(_mm512_srli_epi16(wv, 4), m4);
    if (five) {
        __m512i qhv = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 16)));
        __m512i c = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(4 * p),
                                            _mm512_set1_epi16(4 * p + 2));
        __m512i c1 = _mm512_add_epi16(c, _mm512_set1_epi16(1));
        l = _mm512_or_si512(l, _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(qhv, c), one), 4));
        h = _mm512_or_si512(h, _mm512_slli_epi16(_mm512_and_si512(_mm512_srlv_epi16(qhv, c1), one), 4));
    }
    *lo = l;
    *hi = h;
}

/* The 6-bit values of half h of a Q6_K block, in the lanes of kq_dot_q6k. */
static inline __attribute__((always_inline)) void kq_q6_values(const uint8_t *blk, int h, __m512i *A, __m512i *B)
{
    const __m512i m4 = _mm512_set1_epi8(0x0f), three = _mm512_set1_epi8(3);
    const __m512i cA = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(0),
                                               _mm512_set1_epi16(2));
    const __m512i cB = _mm512_mask_blend_epi16(0xffff0000u, _mm512_set1_epi16(4),
                                               _mm512_set1_epi16(6));
    __m512i wv = _mm512_loadu_si512((const void *)(blk + 64 * h));
    __m512i qhv = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(blk + 128 + 32 * h)));
    *A = _mm512_or_si512(_mm512_and_si512(wv, m4), _mm512_slli_epi16(
        _mm512_and_si512(_mm512_srlv_epi16(qhv, cA), three), 4));
    *B = _mm512_or_si512(_mm512_and_si512(_mm512_srli_epi16(wv, 4), m4), _mm512_slli_epi16(
        _mm512_and_si512(_mm512_srlv_epi16(qhv, cB), three), 4));
}


/* The 5-bit values of two Q5_1 blocks (64 values in their order). The fifth
 * bits of the two blocks are the 64 bits of a mask. */
static inline __attribute__((always_inline)) __m512i kq_q51_values(const uint8_t *b0,
                                                                   const uint8_t *b1)
{
    const __m128i m4 = _mm_set1_epi8(0x0f);
    __m128i qa = _mm_loadu_si128((const __m128i *)(b0 + 8));
    __m128i qb = _mm_loadu_si128((const __m128i *)(b1 + 8));
    __m512i v = _mm512_castsi128_si512(_mm_and_si128(qa, m4));
    v = _mm512_inserti32x4(v, _mm_and_si128(_mm_srli_epi16(qa, 4), m4), 1);
    v = _mm512_inserti32x4(v, _mm_and_si128(qb, m4), 2);
    v = _mm512_inserti32x4(v, _mm_and_si128(_mm_srli_epi16(qb, 4), m4), 3);
    uint32_t ha, hb;
    memcpy(&ha, b0 + 4, 4);
    memcpy(&hb, b1 + 4, 4);
    __mmask64 hi = (__mmask64)ha | ((__mmask64)hb << 32);
    return _mm512_mask_or_epi32(v, 0xffff, v, _mm512_maskz_mov_epi8(hi, _mm512_set1_epi8(16)));
}

/* The products of one row on one token. S and corr come from kq_prep, as
 * in the tiles. */
static float kq_dot_q5_1(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < cols / 32; i += 2) {
        const uint8_t *b0 = w + (size_t)i * 24;
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), kq_q51_values(b0, b0 + 24),
                                         _mm512_loadu_si512((const void *)(xq + (size_t)i * 32)));
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(S[i]), _mm512_set1_ps(S[i + 1]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static float kq_dot_q8_0(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    __m512 acc = _mm512_setzero_ps();
    for (int i = 0; i < cols / 32; i += 2) {
        const uint8_t *b0 = w + (size_t)i * 34, *b1 = b0 + 34;
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b1 + 2));
        __m512i xv = _mm512_loadu_si512((const void *)(xq + (size_t)i * 32));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(wv), sx);
        __m512 sc = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(S[i]), _mm512_set1_ps(S[i + 1]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc);
    }
    return _mm512_reduce_add_ps(acc);
}

static float kq_dot_q45k(const uint8_t *w, int cols, int five, const int8_t *xq, const float *S)
{
    size_t bs = five ? 176 : 144;
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
        const float *f = S + 8 * b;
        for (int p = 0; p < 2; ++p) {
            /* 64 bytes: parts 4p (low 4 bits of the first 32 bytes), 4p + 1
             * (their high 4 bits), 4p + 2 and 4p + 3 (the next 32 bytes). */
            __m512i lo, hi;
            kq_q45_values(w + (size_t)b * bs, five, p, &lo, &hi);
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
    return _mm512_reduce_add_ps(acc);
}

static float kq_dot_q6k(const uint8_t *w, int cols, const int8_t *xq, const float *S)
{
    /* lane i of a product of 64 values belongs to the part of 16 i / 4 */
    const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
    const __m512i iB = _mm512_add_epi32(iA, _mm512_set1_epi32(4));
    const __m512i i8 = _mm512_set1_epi32(8);
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 256; ++b) {
        __m512 fv = _mm512_loadu_ps(S + 16 * b);
        for (int h = 0; h < 2; ++h) {
            __m512i A, B;
            kq_q6_values(w + (size_t)b * 210, h, &A, &B);
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
    return _mm512_reduce_add_ps(acc);
}

#define KQ_S 528           /* the most parts of a row, and a pad of 16 */

/* A tile of 4 rows (wr, with their kq_row_scales ds and dm) by 4 tokens (the prompt pass, a group of an
 * expert), for the formats with int8 x. Each (row, token) pair adds in the
 * order of kq_dot1, so the tile gives the same bits. A short tile repeats
 * its last token. out[i * ostr_r + j * ostr_t] gets row i, token j. */
static void kq_tile4(const uint8_t *const wr[4], float ds[4][KQ_S], float dm[4][KQ_S],
                     int type, int cols, const int8_t *xq, const float *xs, const float *xm,
                     int nt, float *out, size_t ostr_r, size_t ostr_t)
{
    const int8_t *xr[4];
    float S[4][4][KQ_S];
    float corr[4][4];
    for (int j = 0; j < 4; ++j) {
        int jj = j < nt ? j : nt - 1;
        xr[j] = xq + (size_t)jj * cols;
        for (int i = 0; i < 4; ++i) {
            corr[i][j] = kq_prep(type, cols, ds[i], dm[i], xs + (size_t)jj * (cols / 32),
                                 xm + (size_t)jj * (cols / 16), S[i][j]);
        }
    }
    __m512 a00 = _mm512_setzero_ps(), a01 = a00, a02 = a00, a03 = a00;
    __m512 a10 = a00, a11 = a00, a12 = a00, a13 = a00;
    __m512 a20 = a00, a21 = a00, a22 = a00, a23 = a00;
    __m512 a30 = a00, a31 = a00, a32 = a00, a33 = a00;
    const __m512i z = _mm512_setzero_si512();
/* acc += (w . x) * the scales of S[I][J] at base, permuted by idx */
#define KQ_ACC(A, I, J, W, X, IDX, BASE) \
    A = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_dpbusd_epi32(z, W, X)), \
                        _mm512_permutexvar_ps(IDX, _mm512_loadu_ps(S[I][J] + (BASE))), A)
    if (type == KQ_Q8_0) {
        for (int q = 0; q < cols / 64; ++q) {
            int base = (2 * q) & ~15, o = (2 * q) & 15;
            __m512i idx = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o),
                                                  _mm512_set1_epi32(o + 1));
            __m512i w0, w1, w2, w3;
            __mmask64 n0, n1, n2, n3;
#define KQ8_W(I) { const uint8_t *b0 = wr[I] + (size_t)q * 68; \
        __m512i wv = kq_two256((const int8_t *)(b0 + 2), (const int8_t *)(b0 + 36)); \
        n##I = _mm512_movepi8_mask(wv); w##I = _mm512_abs_epi8(wv); }
            KQ8_W(0) KQ8_W(1) KQ8_W(2) KQ8_W(3)
#undef KQ8_W
#define KQ8_J(J, A0, A1, A2, A3) { \
        __m512i xv = _mm512_loadu_si512((const void *)(xr[J] + (size_t)q * 64)); \
        KQ_ACC(A0, 0, J, w0, _mm512_mask_sub_epi8(xv, n0, z, xv), idx, base); \
        KQ_ACC(A1, 1, J, w1, _mm512_mask_sub_epi8(xv, n1, z, xv), idx, base); \
        KQ_ACC(A2, 2, J, w2, _mm512_mask_sub_epi8(xv, n2, z, xv), idx, base); \
        KQ_ACC(A3, 3, J, w3, _mm512_mask_sub_epi8(xv, n3, z, xv), idx, base); }
            KQ8_J(0, a00, a10, a20, a30) KQ8_J(1, a01, a11, a21, a31)
            KQ8_J(2, a02, a12, a22, a32) KQ8_J(3, a03, a13, a23, a33)
#undef KQ8_J
        }
    } else if (type == KQ_Q5_1) {
        for (int q = 0; q < cols / 64; ++q) {
            int base = (2 * q) & ~15, o = (2 * q) & 15;
            __m512i idx = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o),
                                                  _mm512_set1_epi32(o + 1));
            __m512i w0 = kq_q51_values(wr[0] + (size_t)q * 48, wr[0] + (size_t)q * 48 + 24);
            __m512i w1 = kq_q51_values(wr[1] + (size_t)q * 48, wr[1] + (size_t)q * 48 + 24);
            __m512i w2 = kq_q51_values(wr[2] + (size_t)q * 48, wr[2] + (size_t)q * 48 + 24);
            __m512i w3 = kq_q51_values(wr[3] + (size_t)q * 48, wr[3] + (size_t)q * 48 + 24);
#define KQ51_J(J, A0, A1, A2, A3) { \
        __m512i xv = _mm512_loadu_si512((const void *)(xr[J] + (size_t)q * 64)); \
        KQ_ACC(A0, 0, J, w0, xv, idx, base); KQ_ACC(A1, 1, J, w1, xv, idx, base); \
        KQ_ACC(A2, 2, J, w2, xv, idx, base); KQ_ACC(A3, 3, J, w3, xv, idx, base); }
            KQ51_J(0, a00, a10, a20, a30) KQ51_J(1, a01, a11, a21, a31)
            KQ51_J(2, a02, a12, a22, a32) KQ51_J(3, a03, a13, a23, a33)
#undef KQ51_J
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        int five = type == KQ_Q5_K;
        size_t bs = five ? 176 : 144;
        for (int b = 0; b < cols / 256; ++b) {
            int base = (8 * b) & ~15, o = (8 * b) & 15;
            for (int p = 0; p < 2; ++p) {
                __m512i ia = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(o + 4 * p),
                                                     _mm512_set1_epi32(o + 4 * p + 2));
                __m512i ib = _mm512_add_epi32(ia, _mm512_set1_epi32(1));
                __m512i l0, h0, l1, h1, l2, h2, l3, h3;
                kq_q45_values(wr[0] + (size_t)b * bs, five, p, &l0, &h0);
                kq_q45_values(wr[1] + (size_t)b * bs, five, p, &l1, &h1);
                kq_q45_values(wr[2] + (size_t)b * bs, five, p, &l2, &h2);
                kq_q45_values(wr[3] + (size_t)b * bs, five, p, &l3, &h3);
#define KQ4_J(J, A0, A1, A2, A3) { \
        const int8_t *xb = xr[J] + (size_t)b * 256 + 128 * p; \
        __m512i xa = kq_two256(xb, xb + 64), xc = kq_two256(xb + 32, xb + 96); \
        KQ_ACC(A0, 0, J, l0, xa, ia, base); KQ_ACC(A0, 0, J, h0, xc, ib, base); \
        KQ_ACC(A1, 1, J, l1, xa, ia, base); KQ_ACC(A1, 1, J, h1, xc, ib, base); \
        KQ_ACC(A2, 2, J, l2, xa, ia, base); KQ_ACC(A2, 2, J, h2, xc, ib, base); \
        KQ_ACC(A3, 3, J, l3, xa, ia, base); KQ_ACC(A3, 3, J, h3, xc, ib, base); }
                KQ4_J(0, a00, a10, a20, a30) KQ4_J(1, a01, a11, a21, a31)
                KQ4_J(2, a02, a12, a22, a32) KQ4_J(3, a03, a13, a23, a33)
#undef KQ4_J
            }
        }
    } else {
        const __m512i iA = _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0);
        for (int b = 0; b < cols / 256; ++b) {
            for (int h = 0; h < 2; ++h) {
                __m512i ha = _mm512_add_epi32(iA, _mm512_set1_epi32(8 * h));
                __m512i hb = _mm512_add_epi32(ha, _mm512_set1_epi32(4));
                __m512i A0, B0, A1, B1, A2, B2, A3, B3;
                kq_q6_values(wr[0] + (size_t)b * 210, h, &A0, &B0);
                kq_q6_values(wr[1] + (size_t)b * 210, h, &A1, &B1);
                kq_q6_values(wr[2] + (size_t)b * 210, h, &A2, &B2);
                kq_q6_values(wr[3] + (size_t)b * 210, h, &A3, &B3);
#define KQ6_J(J, C0, C1, C2, C3) { \
        const int8_t *xb = xr[J] + (size_t)b * 256 + 128 * h; \
        __m512i x0 = _mm512_loadu_si512((const void *)xb); \
        __m512i x1 = _mm512_loadu_si512((const void *)(xb + 64)); \
        KQ_ACC(C0, 0, J, A0, x0, ha, 16 * b); KQ_ACC(C0, 0, J, B0, x1, hb, 16 * b); \
        KQ_ACC(C1, 1, J, A1, x0, ha, 16 * b); KQ_ACC(C1, 1, J, B1, x1, hb, 16 * b); \
        KQ_ACC(C2, 2, J, A2, x0, ha, 16 * b); KQ_ACC(C2, 2, J, B2, x1, hb, 16 * b); \
        KQ_ACC(C3, 3, J, A3, x0, ha, 16 * b); KQ_ACC(C3, 3, J, B3, x1, hb, 16 * b); }
                KQ6_J(0, a00, a10, a20, a30) KQ6_J(1, a01, a11, a21, a31)
                KQ6_J(2, a02, a12, a22, a32) KQ6_J(3, a03, a13, a23, a33)
#undef KQ6_J
            }
        }
    }
#undef KQ_ACC
    __m512 acc[4][4] = {{a00, a01, a02, a03}, {a10, a11, a12, a13},
                        {a20, a21, a22, a23}, {a30, a31, a32, a33}};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < nt; ++j) {
            out[(size_t)i * ostr_r + (size_t)j * ostr_t] =
                _mm512_reduce_add_ps(acc[i][j]) - corr[i][j];
        }
    }
}

/* The tiles apply to a format with int8 x and a row of at most KQ_S - 16
 * parts. */
static inline int kq_tiles(int type, int cols)
{
    int parts = type == KQ_Q6_K ? cols / 16 : cols / 32;
    return type != KQ_F32 && type != KQ_IQ4_NL && type != KQ_BF16 && type != KQ_NV4 &&
           parts <= KQ_S - 16 &&
           (!kq_block32(type) || cols % 64 == 0);
}
#endif

#if defined(__AVX512VNNI__)
/* One row on one token, with the scales of the row (kq_row_scales). */
static float kq_dot_scaled(const uint8_t *w, int type, int cols, const float *ds,
                           const float *dm, const int8_t *xq, const float *xs, const float *xm)
{
    float S[KQ_S];
    float corr = kq_prep(type, cols, ds, dm, xs, xm, S);
    switch (type) {
    case KQ_Q8_0: return kq_dot_q8_0(w, cols, xq, S) - corr;
    case KQ_Q5_1: return kq_dot_q5_1(w, cols, xq, S) - corr;
    case KQ_Q4_K: return kq_dot_q45k(w, cols, 0, xq, S) - corr;
    case KQ_Q5_K: return kq_dot_q45k(w, cols, 1, xq, S) - corr;
    }
    return kq_dot_q6k(w, cols, xq, S) - corr;
}
#endif

/* One row of type type on one token: xq, xs, xm (quantized), or x for F32. */
#if defined(__AVX512VNNI__)
/* BF16 row, int8 x: sum over the blocks of xs * sum(w * xq). */
static float kq_dot_bf16(const uint8_t *w, int cols, const int8_t *xq, const float *xs)
{
    const uint16_t *h = (const uint16_t *)w;
    __m512 acc = _mm512_setzero_ps();
    for (int b = 0; b < cols / 32; ++b) {
        __m512i hw = _mm512_loadu_si512((const void *)(h + 32 * b));
        __m512 w0 = _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm512_castsi512_si256(hw)), 16));
        __m512 w1 = _mm512_castsi512_ps(_mm512_slli_epi32(
            _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64(hw, 1)), 16));
        __m128i xa = _mm_loadu_si128((const __m128i *)(xq + 32 * b));
        __m128i xb = _mm_loadu_si128((const __m128i *)(xq + 32 * b + 16));
        __m512 x0 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(xa));
        __m512 x1 = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(xb));
        __m512 p = _mm512_fmadd_ps(w1, x1, _mm512_mul_ps(w0, x0));
        acc = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[b]), acc);
    }
    return _mm512_reduce_add_ps(acc);
}

/* NVFP4 row (KQ_NV4), int8 x. For each block: the codes to int8 (twice the
 * values), vpdpbusd with |w| and x with the sign of w (8 sums of 4 values:
 * the first 4 of values 0-15, the last 4 of 16-31), each half times its
 * scale and xs. */
static float kq_dot_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs)
{
    float g = kq_nv4_g(w, cols);
    const __m128i lut = _mm_loadu_si128((const __m128i *)kq_e2m1x2);
    const __m128i m4 = _mm_set1_epi8(15);
    const __m512i lut5 = _mm512_broadcast_i32x4(lut), m45 = _mm512_set1_epi8(15);
    /* the scale of each of the 16 sums: 4 for each half block */
    const __m512i sidx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    __m512 acc5 = _mm512_setzero_ps();
    int nb = cols / 32, b = 0;
    for (; b + 2 <= nb; b += 2) {
        const uint8_t *s0p = kq_nv4_scales(w, cols, b);
        __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(w, b));
        /* lanes of 128 bits: the low codes of block 0, its high codes, the
         * low codes of block 1, its high codes (the order of x) */
        __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
        q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
        __m512i codes = _mm512_and_si512(
            _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
        __m512i wv = _mm512_shuffle_epi8(lut5, codes);
        __m512i xv = _mm512_loadu_si512((const void *)(xq + 32 * b));
        __mmask64 neg = _mm512_movepi8_mask(wv);
        __m512i sx = _mm512_mask_sub_epi8(xv, neg, _mm512_setzero_si512(), xv);
        __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), _mm512_abs_epi8(wv), sx);
        __m128 s4 = _mm_setr_ps(kq_e4m3_tab[s0p[0]] * xs[b], kq_e4m3_tab[s0p[1]] * xs[b],
                                kq_e4m3_tab[s0p[2]] * xs[b + 1], kq_e4m3_tab[s0p[3]] * xs[b + 1]);
        __m512 sc = _mm512_permutexvar_ps(sidx, _mm512_castps128_ps512(s4));
        acc5 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), sc, acc5);
    }
    __m256 acc = _mm256_setzero_ps();
    for (; b < nb; ++b) {
        const uint8_t *sp = kq_nv4_scales(w, cols, b);
        __m128i q = _mm_loadu_si128((const __m128i *)kq_nv4_codes(w, b));
        __m128i lo = _mm_shuffle_epi8(lut, _mm_and_si128(q, m4));
        __m128i hi = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(q, 4), m4));
        __m256i wv = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
        __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + 32 * b));
        __m256i sx = _mm256_sign_epi8(xv, wv);
        __m256i is = _mm256_dpbusd_epi32(_mm256_setzero_si256(), _mm256_abs_epi8(wv), sx);
        float s0 = kq_e4m3_tab[sp[0]] * xs[b], s1 = kq_e4m3_tab[sp[1]] * xs[b];
        __m256 sc = _mm256_setr_ps(s0, s0, s0, s0, s1, s1, s1, s1);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(is), sc, acc);
    }
    __m128 r = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    r = _mm_hadd_ps(r, r);
    r = _mm_hadd_ps(r, r);
    return 0.5f * g * (_mm_cvtss_f32(r) + _mm512_reduce_add_ps(acc5));
}
#endif

#if defined(__AVX512VNNI__)
#define KQ_NV4_MAX 4096
static void kq_row_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs, int n,
                       float *out, size_t ostride);
#endif

static float kq_dot1(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                     const float *xm, const float *x)
{
#if defined(__AVX512VNNI__)
    if (type == KQ_F32) {
        return kq_dot_f32((const float *)w, cols, x);
    }
    if (type == KQ_BF16) {
        return kq_dot_bf16(w, cols, xq, xs);
    }
    if (type == KQ_NV4) {
        /* With cols a multiple of 64, the sums and their order are those
         * of kq_row_nv4 (a group): the same bits. */
        return kq_dot_nv4(w, cols, xq, xs);
    }
    if (kq_tiles(type, cols)) {
        float ds[KQ_S], dm[KQ_S];
        kq_row_scales(w, type, cols, ds, dm);
        return kq_dot_scaled(w, type, cols, ds, dm, xq, xs, xm);
    }
#endif
    (void)xm;
    if (type == KQ_F32) {
        float s = 0.f;
        for (int c = 0; c < cols; ++c) {
            s += ((const float *)w)[c] * x[c];
        }
        return s;
    }
    int bv = kq_block32(type) ? 32 : 256;
    float v[256], s = 0.f;
    for (int b = 0; b < cols / bv; ++b) {
        kq_block_values(type, w, cols, b, v);
        for (int i = 0; i < bv; ++i) {
            int c = b * bv + i;
            s += v[i] * xs[c / 32] * (float)xq[c];
        }
    }
    return s;
}

/* One row on n tokens. out[j * ostride] gets token j. */
#if defined(__AVX512VNNI__)
/* NVFP4 on several tokens: the row to |w| (int8), its signs, and a scale for
 * each 16 values, one time; then for each token 64 values at a time. The
 * sums are those of kq_dot_nv4 (the same int32 sums of 16 values, the same
 * scales), so a group gives the bits of steps. */
static void kq_row_nv4(const uint8_t *w, int cols, const int8_t *xq, const float *xs, int n,
                       float *out, size_t ostride)
{
    __attribute__((aligned(64))) int8_t aw[KQ_NV4_MAX];
    __attribute__((aligned(64))) float sc[KQ_NV4_MAX / 16];
    __mmask64 negs[KQ_NV4_MAX / 64];
    float g = kq_nv4_g(w, cols);
    int nb = cols / 32;
    const __m512i lut5 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
    const __m512i m45 = _mm512_set1_epi8(15);
    for (int b = 0; b < nb; b += 2) {
        /* as kq_dot_nv4: two blocks to 64 int8 in the order of x */
        const uint8_t *sp = kq_nv4_scales(w, cols, b);
        __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(w, b));
        __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
        q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
        __m512i codes = _mm512_and_si512(
            _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
        __m512i wv = _mm512_shuffle_epi8(lut5, codes);
        negs[b / 2] = _mm512_movepi8_mask(wv);
        _mm512_store_si512((void *)(aw + 32 * b), _mm512_abs_epi8(wv));
        sc[2 * b] = kq_e4m3_tab[sp[0]];
        sc[2 * b + 1] = kq_e4m3_tab[sp[1]];
        sc[2 * b + 2] = kq_e4m3_tab[sp[2]];
        sc[2 * b + 3] = kq_e4m3_tab[sp[3]];
    }
    const __m512i sidx = _mm512_setr_epi32(0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3);
    for (int j = 0; j < n; ++j) {
        const int8_t *xr = xq + (size_t)j * cols;
        const float *sr = xs + (size_t)j * nb;
        __m512 acc = _mm512_setzero_ps();
        for (int c = 0; c < cols; c += 64) {
            __m512i xv = _mm512_loadu_si512((const void *)(xr + c));
            __m512i sx = _mm512_mask_sub_epi8(xv, negs[c / 64], _mm512_setzero_si512(), xv);
            __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(),
                                             _mm512_load_si512((const void *)(aw + c)), sx);
            int b = c / 32;
            __m128 s4 = _mm_setr_ps(sc[2 * b] * sr[b], sc[2 * b + 1] * sr[b],
                                    sc[2 * b + 2] * sr[b + 1], sc[2 * b + 3] * sr[b + 1]);
            acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is),
                                  _mm512_permutexvar_ps(sidx, _mm512_castps128_ps512(s4)), acc);
        }
        out[(size_t)j * ostride] = 0.5f * g * _mm512_reduce_add_ps(acc);
    }
}
#endif

static void kq_row(const uint8_t *w, int type, int cols, const int8_t *xq, const float *xs,
                   const float *xm, const float *x, int n, float *out, size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (type == KQ_NV4 && n > 1 && cols % 64 == 0 && cols <= KQ_NV4_MAX) {
        kq_row_nv4(w, cols, xq, xs, n, out, ostride);
        return;
    }
    if (n > 1 && type != KQ_F32 && kq_tiles(type, cols)) {
        /* The scales of the row one time for all the tokens. */
        float ds[KQ_S], dm[KQ_S];
        kq_row_scales(w, type, cols, ds, dm);
        for (int j = 0; j < n; ++j) {
            out[(size_t)j * ostride] = kq_dot_scaled(
                w, type, cols, ds, dm, xq + (size_t)j * cols, xs + (size_t)j * (cols / 32),
                xm + (size_t)j * (cols / 16));
        }
        return;
    }
#endif
    for (int j = 0; j < n; ++j) {
        out[(size_t)j * ostride] = kq_dot1(w, type, cols, xq + (size_t)j * cols,
                                           xs + (size_t)j * (cols / 32),
                                           xm + (size_t)j * (cols / 16),
                                           x ? x + (size_t)j * cols : NULL);
    }
}

/* Rows r .. r + 3 (at w, rb bytes each) on n tokens: tiles of 4 tokens for
 * n >= 4, else one row at a time. out[i + j * ostride] gets row r + i,
 * token j. */
#if defined(__AVX512VNNI__)
/* NVFP4, rows r .. r + 3 on n tokens (the experts of a group): the 4 rows
 * to |w| (int8), the signs, and the E4M3 scales in the order of the 16 sums
 * of a chunk of 64 values, one time; then tiles of 4 rows by 4 tokens, so a
 * chunk of a row and of a token is loaded one time for 4 products. The sums,
 * the scales, and their order are those of kq_dot_nv4: the bits of steps. */
static void kq_rows4_nv4(const uint8_t *w, size_t rb, int cols, const int8_t *xq,
                         const float *xs, int n, float *out, size_t ostride)
{
    __attribute__((aligned(64))) int8_t aw[4][KQ_NV4_MAX];
    __attribute__((aligned(64))) float wsc[4][KQ_NV4_MAX / 64][16];
    __mmask64 negs[4][KQ_NV4_MAX / 64];
    float g[4];
    int nb = cols / 32, nc = cols / 64;
    const __m512i lut5 = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1x2));
    const __m512i m45 = _mm512_set1_epi8(15);
    for (int r = 0; r < 4; ++r) {
        const uint8_t *wr = w + (size_t)r * rb;
        g[r] = kq_nv4_g(wr, cols);
        for (int b = 0; b < nb; b += 2) {
            const uint8_t *sp = kq_nv4_scales(wr, cols, b);
            __m256i q01 = _mm256_loadu_si256((const __m256i *)kq_nv4_codes(wr, b));
            __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(q01), q01, 1);
            q = _mm512_permutexvar_epi64(_mm512_setr_epi64(0, 1, 0, 1, 2, 3, 2, 3), q);
            __m512i codes = _mm512_and_si512(
                _mm512_mask_blend_epi64(0xcc, q, _mm512_srli_epi16(q, 4)), m45);
            __m512i wv = _mm512_shuffle_epi8(lut5, codes);
            negs[r][b / 2] = _mm512_movepi8_mask(wv);
            _mm512_store_si512((void *)(aw[r] + 32 * b), _mm512_abs_epi8(wv));
            float s4[4] = {kq_e4m3_tab[sp[0]], kq_e4m3_tab[sp[1]], kq_e4m3_tab[sp[2]],
                           kq_e4m3_tab[sp[3]]};
            for (int l = 0; l < 16; ++l) {
                wsc[r][b / 2][l] = s4[l / 4];
            }
        }
    }
    int j = 0;
    for (; j + 4 <= n; j += 4) {
        __m512 acc[4][4];
        for (int r = 0; r < 4; ++r) {
            for (int t = 0; t < 4; ++t) {
                acc[r][t] = _mm512_setzero_ps();
            }
        }
        for (int c = 0; c < nc; ++c) {
            __m512i xv[4];
            __m512 xsv[4];
            for (int t = 0; t < 4; ++t) {
                const float *sr = xs + (size_t)(j + t) * nb;
                xv[t] = _mm512_loadu_si512((const void *)(xq + (size_t)(j + t) * cols + 64 * c));
                xsv[t] = _mm512_mask_blend_ps(0xff00, _mm512_set1_ps(sr[2 * c]),
                                              _mm512_set1_ps(sr[2 * c + 1]));
            }
            for (int r = 0; r < 4; ++r) {
                __m512i wa = _mm512_load_si512((const void *)(aw[r] + 64 * c));
                __m512 ws = _mm512_load_ps(wsc[r][c]);
                for (int t = 0; t < 4; ++t) {
                    __m512i sx = _mm512_mask_sub_epi8(xv[t], negs[r][c], _mm512_setzero_si512(), xv[t]);
                    __m512i is = _mm512_dpbusd_epi32(_mm512_setzero_si512(), wa, sx);
                    acc[r][t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(is), _mm512_mul_ps(ws, xsv[t]),
                                                acc[r][t]);
                }
            }
        }
        for (int r = 0; r < 4; ++r) {
            for (int t = 0; t < 4; ++t) {
                out[(size_t)(j + t) * ostride + r] = 0.5f * g[r] * _mm512_reduce_add_ps(acc[r][t]);
            }
        }
    }
    for (; j < n; ++j) {                    /* the last tokens: one at a time */
        for (int r = 0; r < 4; ++r) {
            out[(size_t)j * ostride + r] = kq_dot_nv4(w + (size_t)r * rb, cols, xq + (size_t)j * cols,
                                                      xs + (size_t)j * nb);
        }
    }
}
#endif

static void kq_rows4(const uint8_t *w, size_t rb, int type, int cols, const int8_t *xq,
                     const float *xs, const float *xm, const float *x, int n, float *out,
                     size_t ostride)
{
#if defined(__AVX512VNNI__)
    if (type == KQ_NV4 && n > 1 && cols % 64 == 0 && cols <= KQ_NV4_MAX) {
        kq_rows4_nv4(w, rb, cols, xq, xs, n, out, ostride);
        return;
    }
    if (n >= 4 && kq_tiles(type, cols)) {
        const uint8_t *wr[4] = {w, w + rb, w + 2 * rb, w + 3 * rb};
        float ds[4][KQ_S], dm[4][KQ_S];
        for (int i = 0; i < 4; ++i) {
            kq_row_scales(wr[i], type, cols, ds[i], dm[i]);
        }
        for (int j0 = 0; j0 < n; j0 += 4) {
            int nt = n - j0 < 4 ? n - j0 : 4;
            kq_tile4(wr, ds, dm, type, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * (cols / 32),
                     xm + (size_t)j0 * (cols / 16), nt, out + (size_t)j0 * ostride, 1, ostride);
        }
        return;
    }
#endif
    for (int i = 0; i < 4; ++i) {
        kq_row(w + (size_t)i * rb, type, cols, xq, xs, xm, x, n, out + i, ostride);
    }
}

/* out (t x rows) = x W^T, inside a parallel region. x is the float input
 * (for F32), and xq, xs, xm its quantization (kq_quant_body). */
#if defined(__AVX512VNNI__)
#define KQ_X16_BB 576
#define KQ_X16_XL 98304                  /* the bytes of a small x (kq_x16_body) */
/* KQ_Q8X16: rows g * 16 .. + 15 (the group at wg) on the tokens j0 .. j1 - 1,
 * 8 tokens at a time. xu is x as uint8 (x + 128). vpdpbusd (u8 x s8) on x + 128
 * and w gives sum(x w) + 128 sum(w): the int32 sum of a block starts at
 * -128 times the sums of the rows (made when the rows were packed), the same
 * for all the tokens. For each block: the 8 steps of the group in registers;
 * for each token, 8 vpdpbusd with 4 values of x in each lane, then the scales
 * of the 16 rows times xs. out gets the 16 rows of each token (a stride of
 * rows). */
/* NT tokens from j (a count fixed at compile time: the sums stay in
 * registers). */
#define KQ_X16_NT(NT)                                                                             \
static inline void kq_x16_tok##NT(const uint8_t *wg, int nb, int cols, const uint8_t *xu,        \
                                  const float *xs, int j, float *out, size_t rows)               \
{                                                                                                 \
    __m512 f[NT];                                                                                 \
    for (int tt = 0; tt < NT; ++tt) {                                                             \
        f[tt] = _mm512_setzero_ps();                                                              \
    }                                                                                             \
    for (int b = 0; b < nb; ++b) {                                                                \
        const uint8_t *blk = wg + (size_t)b * KQ_X16_BB;                                          \
        __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)blk));                     \
        __m512i c0 = _mm512_slli_epi32(_mm512_cvtepi16_epi32(                                     \
            _mm256_sub_epi16(_mm256_setzero_si256(), _mm256_loadu_si256((const __m256i *)(blk + 32)))), 7); \
        __m512i W[8];                                                                             \
        for (int k = 0; k < 8; ++k) {                                                             \
            W[k] = _mm512_loadu_si512((const void *)(blk + 64 + 64 * k));                         \
        }                                                                                         \
        _Pragma("GCC unroll 8")                                                                   \
        for (int tt = 0; tt < NT; ++tt) {                                                         \
            const uint8_t *xb = xu + (size_t)(j + tt) * cols + 32 * b;                            \
            __m512i acc = c0;                                                                     \
            for (int k = 0; k < 8; ++k) {                                                         \
                int32_t x4;                                                                       \
                memcpy(&x4, xb + 4 * k, 4);                                                       \
                acc = _mm512_dpbusd_epi32(acc, _mm512_set1_epi32(x4), W[k]);                      \
            }                                                                                     \
            f[tt] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc),                                      \
                                    _mm512_mul_ps(d, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b])), \
                                    f[tt]);                                                       \
        }                                                                                         \
    }                                                                                             \
    for (int tt = 0; tt < NT; ++tt) {                                                             \
        _mm512_storeu_ps(out + (size_t)(j + tt) * rows, f[tt]);                                   \
    }                                                                                             \
}
KQ_X16_NT(8)
KQ_X16_NT(4)
KQ_X16_NT(2)
KQ_X16_NT(1)

/* KQ_Q8X16: rows g * 16 .. + 15 (the group at wg) on the tokens j0 .. j1 - 1,
 * 8 tokens at a time, then 4, 2, 1 (counts fixed at compile time: a
 * variable count made 4 tokens on 10240 x 320 take 1.9 times 1 token). xu
 * is x as uint8 (x + 128). vpdpbusd (u8 x s8) on x + 128 and w gives sum(x
 * w) + 128 sum(w): the int32 sum of a block starts at -128 times the sums
 * of the rows (made when the rows were packed), the same for all the
 * tokens. For each block: the 8 steps of the group in registers; for each
 * token, 8 vpdpbusd with 4 values of x in each lane, then the scales of the
 * 16 rows times xs. out gets the 16 rows of each token (a stride of rows).
 * Each token has the same operations in all the counts: the same bits. */
static void kq_x16_group(const uint8_t *wg, int cols, const uint8_t *xu, const float *xs, int j0,
                         int j1, float *out, size_t rows)
{
    int nb = cols / 32, j = j0;
    for (; j + 8 <= j1; j += 8) {
        kq_x16_tok8(wg, nb, cols, xu, xs, j, out, rows);
    }
    if (j + 4 <= j1) {
        kq_x16_tok4(wg, nb, cols, xu, xs, j, out, rows);
        j += 4;
    }
    if (j + 2 <= j1) {
        kq_x16_tok2(wg, nb, cols, xu, xs, j, out, rows);
        j += 2;
    }
    if (j < j1) {
        kq_x16_tok1(wg, nb, cols, xu, xs, j, out, rows);
    }
}

/* out (t x rows) of a KQ_Q8X16 matrix, inside a parallel region: x + 128 once
 * (a buffer of the call), then blocks of MA_TB tokens (x in L2) with the
 * groups of 16 rows over the threads. One token has the sums of a group: the
 * same bits. */
static void kq_x16_body(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        int t, float *out)
{
    size_t n = (size_t)t * cols;
    size_t gb = (size_t)cols / 32 * KQ_X16_BB;
    if (n <= KQ_X16_XL) {
        /* a small x (a step, a verify group): each thread makes x + 128 on
         * its stack, so the record has one barrier, not four (and no
         * malloc). Two halves of the columns as tasks, for a matrix of few
         * groups (hc_*_down: 20 groups for 18 threads), were not faster. */
        __attribute__((aligned(64))) uint8_t xl[KQ_X16_XL];
        for (size_t i = 0; i < n; i += 64) {
            if (i + 64 <= n) {
                __m512i v = _mm512_loadu_si512((const void *)(xq + i));
                _mm512_store_si512((void *)(xl + i), _mm512_xor_si512(v, _mm512_set1_epi8((char)0x80)));
            } else {
                for (size_t k = i; k < n; ++k) {
                    xl[k] = (uint8_t)xq[k] ^ 0x80;
                }
            }
        }
        #pragma omp for schedule(static)
        for (int g = 0; g < rows / 16; ++g) {
            kq_x16_group(w + (size_t)g * gb, cols, xl, xs, 0, t, out + 16 * g, (size_t)rows);
        }
        return;
    }
    uint8_t *xu = NULL;
    #pragma omp single copyprivate(xu)
    xu = (uint8_t *)aligned_alloc(64, (n + 63) / 64 * 64);
    #pragma omp for schedule(static)
    for (size_t i = 0; i < n; i += 64) {
        if (i + 64 <= n) {
            __m512i v = _mm512_loadu_si512((const void *)(xq + i));
            _mm512_storeu_si512((void *)(xu + i), _mm512_xor_si512(v, _mm512_set1_epi8((char)0x80)));
        } else {
            for (size_t k = i; k < n; ++k) {
                xu[k] = (uint8_t)xq[k] ^ 0x80;
            }
        }
    }
    for (int j0 = 0; j0 < t; j0 += MA_TB) {
        int j1 = t - j0 < MA_TB ? t : j0 + MA_TB;
        #pragma omp for schedule(static) nowait
        for (int g = 0; g < rows / 16; ++g) {
            kq_x16_group(w + (size_t)g * gb, cols, xu, xs, j0, j1, out + 16 * g, (size_t)rows);
        }
    }
    #pragma omp barrier
    #pragma omp single
    free(xu);
}
#endif

/* KQ_NVX: -12 times the sum of each 16 int8 values of n rows of x (the
 * term of the codes + 12 of kq_nvx_rows). */
static void kq_nvx_xsum(const int8_t *xq, int n, int cols, int32_t *xn)
{
    for (int j = 0; j < n; ++j) {
        for (int h = 0; h < cols / 16; ++h) {
            const int8_t *p = xq + (size_t)j * cols + 16 * h;
            int32_t sum = 0;
            for (int u = 0; u < 16; ++u) {
                sum += p[u];
            }
            xn[(size_t)j * (cols / 16) + h] = -12 * sum;
        }
    }
}

#if defined(__AVX512VNNI__)
/* E2M1 codes to twice their values plus 12 (0 to 24: the unsigned operand
 * of vpdpbusd). */
/* 16 tokens for each decode of a block (8: 1.5% slower at 512 tokens, 7% at
 * 2048; the experts of a prompt are near the rate of the memory). */
#define KQ_NVX_TB 16
static const uint8_t kq_e2m1u[16] = {12, 13, 14, 15, 16, 18, 20, 24, 12, 11, 10, 9, 8, 6, 4, 0};

/* 16 E4M3 scales as float32 times 2^-8: the bits go to float16 as they are
 * (the exponent bias of float16 is 8 more), exact for all the codes. */
static inline __m512 kq_e4m3x16(const uint8_t *p)
{
    __m256i b = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)p));
    __m256i h = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x7f)), 7),
                                _mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x80)), 8));
    return _mm512_cvtph_ps(h);
}

/* The 16 rows of the KQ_NVX group at wg on n tokens: out[j * ostride + r]
 * gets row r, token j. xn: kq_nvx_xsum of the tokens. For each block, 8
 * tokens at a time: the 8 steps to u8 (codes + 12) in registers; for each
 * token, 4 vpdpbusd for each half (a lane is a row; x broadcast, signed),
 * from -12 sum(x), which removes the 12. A token alone and in a group has
 * the same operations: the same bits. */
static void kq_nvx_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    int nb = cols / 32, nh = cols / 16;
    float g;
    memcpy(&g, wg, 4);
    const __m512 gs = _mm512_set1_ps(128.f * g);        /* 0.5 g, and 2^8 of kq_e4m3x16 */
    const __m512i lut = _mm512_broadcast_i32x4(_mm_loadu_si128((const __m128i *)kq_e2m1u));
    const __m256i m4 = _mm256_set1_epi8(15);
    for (int j = 0; j < n; j += KQ_NVX_TB) {
        int nt = n - j < KQ_NVX_TB ? n - j : KQ_NVX_TB;
        __m512 f[KQ_NVX_TB];
        for (int tt = 0; tt < KQ_NVX_TB; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int b = 0; b < nb; ++b) {
            const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
            __m512i W[8];
            for (int k = 0; k < 8; ++k) {
                __m256i q = _mm256_loadu_si256((const __m256i *)(blk + 32 * k));
                __m256i lo = _mm256_and_si256(q, m4), hi = _mm256_and_si256(_mm256_srli_epi16(q, 4), m4);
                W[k] = _mm512_shuffle_epi8(lut, _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
            }
            __m512 s0 = kq_e4m3x16(blk + 256), s1 = kq_e4m3x16(blk + 272);
            for (int tt = 0; tt < nt; ++tt) {
                const int8_t *xb = xq + (size_t)(j + tt) * cols + 32 * b;
                const int32_t *nx = xn + (size_t)(j + tt) * nh + 2 * b;
                __m512i a0 = _mm512_set1_epi32(nx[0]), a1 = _mm512_set1_epi32(nx[1]);
                for (int k = 0; k < 4; ++k) {
                    int32_t x0, x1;
                    memcpy(&x0, xb + 4 * k, 4);
                    memcpy(&x1, xb + 16 + 4 * k, 4);
                    a0 = _mm512_dpbusd_epi32(a0, W[k], _mm512_set1_epi32(x0));
                    a1 = _mm512_dpbusd_epi32(a1, W[4 + k], _mm512_set1_epi32(x1));
                }
                __m512 p = _mm512_fmadd_ps(_mm512_cvtepi32_ps(a1), s1,
                                           _mm512_mul_ps(_mm512_cvtepi32_ps(a0), s0));
                f[tt] = _mm512_fmadd_ps(p, _mm512_set1_ps(xs[(size_t)(j + tt) * nb + b]), f[tt]);
            }
        }
        for (int tt = 0; tt < nt; ++tt) {
            _mm512_storeu_ps(out + (size_t)(j + tt) * ostride, _mm512_mul_ps(f[tt], gs));
        }
    }
}
#else
/* Without VNNI: the same sums in C (not the bits of the VNNI path). */
static void kq_nvx_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride)
{
    static const int8_t e2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    int nb = cols / 32;
    float g;
    memcpy(&g, wg, 4);
    (void)xn;
    for (int j = 0; j < n; ++j) {
        for (int r = 0; r < 16; ++r) {
            float f = 0.f;
            for (int b = 0; b < nb; ++b) {
                const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
                float p = 0.f;
                for (int h = 0; h < 2; ++h) {
                    int32_t a = 0;
                    for (int v = 16 * h; v < 16 * h + 16; ++v) {
                        int c = (blk[32 * (v / 4) + 4 * (r % 8) + v % 4] >> (r < 8 ? 0 : 4)) & 15;
                        a += e2[c] * xq[(size_t)j * cols + 32 * b + v];
                    }
                    p += (float)a * kq_e4m3_tab[blk[256 + 16 * h + r]];
                }
                f += p * xs[(size_t)j * nb + b];
            }
            out[(size_t)j * ostride + r] = 0.5f * g * f;
        }
    }
}
#endif

/* KQ_NVX: out (t x rows), inside a parallel region; blocks of MA_TB tokens
 * with the groups of 16 rows over the threads. */
static void kq_nvx_body(const uint8_t *w, int rows, int cols, const int8_t *xq, const float *xs,
                        int t, float *out)
{
    int32_t *xn = NULL;
    size_t gb = 16 * kq_row_bytes(KQ_NVX, cols);
    #pragma omp single copyprivate(xn)
    xn = (int32_t *)malloc((size_t)t * (cols / 16) * 4);
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        kq_nvx_xsum(xq + (size_t)j * cols, 1, cols, xn + (size_t)j * (cols / 16));
    }
    for (int j0 = 0; j0 < t; j0 += MA_TB) {
        int nt = t - j0 < MA_TB ? t - j0 : MA_TB;
        #pragma omp for schedule(static) nowait
        for (int g = 0; g < rows / 16; ++g) {
            kq_nvx_rows(w + (size_t)g * gb, cols, xq + (size_t)j0 * cols, xs + (size_t)j0 * (cols / 32),
                        xn + (size_t)j0 * (cols / 16), nt, out + (size_t)j0 * rows + 16 * g,
                        (size_t)rows);
        }
    }
    #pragma omp barrier
    #pragma omp single
    free(xn);
}

#if defined(__AVX512F__)
#ifndef KQ_X16F_GB
#define KQ_X16F_GB 4
#endif

/* ng (at most 4) groups of 16 rows (gb bytes apart; the last has nr rows)
 * on at most 4 tokens j0 .. j1 - 1: the groups give independent sums, so
 * the fma of a token do not wait on each other (one group on one token
 * waits 4 cycles for each column). A missing group repeats group 0 and is
 * not stored. For each row and token: one fma for each column, in order, as
 * kq_x16f_group16: the same bits. */
static void kq_x16f_groups4(const uint8_t *wg, size_t gb, int ng, int nr, int bf, int cols,
                            size_t xstride, const float *x, int j0, int j1, float *out,
                            size_t ostride)
{
    int nt = j1 - j0;
    const uint8_t *gp[4];
    for (int g = 0; g < 4; ++g) {
        gp[g] = wg + (g < ng ? g : 0) * gb;
    }
    __m512 f[4][4];
    for (int g = 0; g < 4; ++g) {
        for (int tt = 0; tt < 4; ++tt) {
            f[g][tt] = _mm512_setzero_ps();
        }
    }
    const float *xr = x + (size_t)j0 * xstride;
    for (int c = 0; c < cols; ++c) {
        __m512 wv[4];
        #pragma GCC unroll 4
        for (int g = 0; g < 4; ++g) {
            wv[g] = bf ? _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                             _mm256_loadu_si256((const __m256i *)(gp[g] + (size_t)32 * c))), 16))
                       : _mm512_loadu_ps((const float *)(gp[g] + (size_t)64 * c));
        }
        #pragma GCC unroll 4
        for (int tt = 0; tt < 4; ++tt) {
            if (tt < nt) {
                __m512 xv = _mm512_set1_ps(xr[(size_t)tt * xstride + c]);
                #pragma GCC unroll 4
                for (int g = 0; g < 4; ++g) {
                    f[g][tt] = _mm512_fmadd_ps(wv[g], xv, f[g][tt]);
                }
            }
        }
    }
    for (int tt = 0; tt < nt; ++tt) {
        for (int g = 0; g < ng; ++g) {
            int n = g == ng - 1 ? nr : 16;
            __mmask16 mk = (__mmask16)(n >= 16 ? 0xffff : (1u << n) - 1);
            _mm512_mask_storeu_ps(out + (size_t)(j0 + tt) * ostride + 16 * g, mk, f[g][tt]);
        }
    }
}

/* One group of 16 rows on tokens j0 .. j1 - 1, 16 tokens at a time, with
 * loops of fixed counts (the 16 sums stay in registers; each broadcast is
 * an operand of its fma). A batch of fewer than 16 tokens repeats its last
 * token and does not store it. */
static void kq_x16f_group16(const uint8_t *wg, int bf, int cols, size_t xstride, const float *x,
                            int j0, int j1, int nr, float *out, size_t ostride)
{
    __mmask16 mk = (__mmask16)(nr >= 16 ? 0xffff : (1u << nr) - 1);
    for (int j = j0; j < j1; j += 16) {
        int nt = j1 - j < 16 ? j1 - j : 16;
        const float *xp[16];
        for (int tt = 0; tt < 16; ++tt) {
            xp[tt] = x + (size_t)(j + (tt < nt ? tt : nt - 1)) * xstride;
        }
        __m512 f[16];
        for (int tt = 0; tt < 16; ++tt) {
            f[tt] = _mm512_setzero_ps();
        }
        for (int c = 0; c < cols; ++c) {
            __m512 wv = bf ? _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                 _mm256_loadu_si256((const __m256i *)(wg + (size_t)32 * c))), 16))
                           : _mm512_loadu_ps((const float *)(wg + (size_t)64 * c));
            #pragma GCC unroll 16
            for (int tt = 0; tt < 16; ++tt) {
                f[tt] = _mm512_fmadd_ps(wv, _mm512_set1_ps(xp[tt][c]), f[tt]);
            }
        }
        #pragma GCC unroll 16
        for (int tt = 0; tt < 16; ++tt) {
            if (tt < nt) {
                _mm512_mask_storeu_ps(out + (size_t)(j + tt) * ostride, mk, f[tt]);
            }
        }
    }
}
#else
/* Without AVX-512: the same sums in C (not the bits of the AVX-512 path). */
static void kq_x16f_group(const uint8_t *wg, int bf, int cols, size_t xstride, const float *x,
                          int j0, int j1, int nr, float *out, size_t ostride)
{
    for (int j = j0; j < j1; ++j) {
        for (int r = 0; r < nr; ++r) {
            float s = 0.f;
            for (int c = 0; c < cols; ++c) {
                float v;
                if (bf) {
                    uint32_t u = (uint32_t)((const uint16_t *)wg)[16 * c + r] << 16;
                    memcpy(&v, &u, 4);
                } else {
                    v = ((const float *)wg)[16 * c + r];
                }
                s += v * x[(size_t)j * xstride + c];
            }
            out[(size_t)j * ostride + r] = s;
        }
    }
}
#endif

/* out (t x rows) of a KQ_BF16X16 or KQ_F32X16 matrix, inside a parallel
 * region. A task: at most 4 tokens, 4 groups or 1 (kq_x16f_groups4); more
 * tokens, KQ_X16F_GB groups on 16 tokens, so x comes from the cache of the
 * core for all of them (kq_x16f_group16). Tiles of 4 groups by 64 tokens in
 * chunks of 256 columns, with x as [column][token], were slower (44 to 48
 * ms, not 28, for 10240 x 2560 on 512 tokens). */
static void kq_x16f_body(const uint8_t *w, int type, int rows, int cols, const float *x, int t,
                         float *out)
{
    int ng = (rows + 15) / 16, bf = type == KQ_BF16X16;
    size_t gb = (size_t)16 * kq_row_bytes(type, cols);
    int u = 1, tk = 16;
#if defined(__AVX512F__)
    if (t <= 4) {
        /* 4 groups in a task when there are enough for all the threads */
        u = ng >= 4 * omp_get_num_threads() ? 4 : 1, tk = 4;
    } else {
        u = KQ_X16F_GB;
    }
#endif
    int nu = (ng + u - 1) / u, nj = (t + tk - 1) / tk;
    #pragma omp for schedule(static)
    for (int e = 0; e < nu * nj; ++e) {
        int g = u * (e / nj), j0 = tk * (e % nj), j1 = j0 + tk < t ? j0 + tk : t;
        int n = ng - g < u ? ng - g : u;                   /* the groups of the task */
        int last = rows - 16 * (g + n - 1);                /* the rows of its last group */
        last = last < 16 ? last : 16;
        const uint8_t *wg = w + (size_t)g * gb;
        float *o = out + (size_t)16 * g;
#if defined(__AVX512F__)
        if (t <= 4) {
            kq_x16f_groups4(wg, gb, n, last, bf, cols, (size_t)cols, x, j0, j1, o, (size_t)rows);
            continue;
        }
        for (int gi = 0; gi < n; ++gi) {
            kq_x16f_group16(wg + gi * gb, bf, cols, (size_t)cols, x, j0, j1, gi == n - 1 ? last : 16,
                            o + 16 * gi, (size_t)rows);
        }
#else
        kq_x16f_group(wg, bf, cols, (size_t)cols, x, j0, j1, last, o, (size_t)rows);
#endif
    }
}

static void kq_linear_body(const uint8_t *w, int type, int rows, int cols, const int8_t *xq,
                           const float *xs, const float *xm, const float *x, int t, float *out)
{
    if (type == KQ_BF16X16 || type == KQ_F32X16) {
        kq_x16f_body(w, type, rows, cols, x, t, out);
        return;
    }
    if (type == KQ_NVX) {
        kq_nvx_body(w, rows, cols, xq, xs, t, out);
        return;
    }
    size_t rb = kq_row_bytes(type, cols);
#if defined(__AVX512VNNI__)
    if (type == KQ_Q8X16) {
        kq_x16_body(w, rows, cols, xq, xs, t, out);
        return;
    }
    if (t >= 4 && rows % 4 == 0 && kq_tiles(type, cols)) {
        /* Blocks of MA_TB tokens, as ma_linear_body. */
        for (int j0 = 0; j0 < t; j0 += MA_TB) {
            int nb = t - j0 < MA_TB ? t - j0 : MA_TB;
            #pragma omp for schedule(static) nowait
            for (int r4 = 0; r4 < rows / 4; ++r4) {
                kq_rows4(w + (size_t)4 * r4 * rb, rb, type, cols, xq + (size_t)j0 * cols,
                         xs + (size_t)j0 * (cols / 32), xm + (size_t)j0 * (cols / 16), NULL, nb,
                         out + (size_t)j0 * rows + 4 * r4, (size_t)rows);
            }
        }
        #pragma omp barrier
        return;
    }
#endif
    if (rows < 64 && t > 1) {
        /* few rows (the small float32 matrices): tasks of a row and 16
         * tokens, so the tokens take the threads; the sums of each token
         * are those of kq_row */
        int nj = (t + 15) / 16;
        #pragma omp for schedule(static)
        for (int e = 0; e < rows * nj; ++e) {
            int r = e / nj, j0 = 16 * (e % nj), n = t - j0 < 16 ? t - j0 : 16;
            kq_row(w + (size_t)r * rb, type, cols, xq + (size_t)j0 * cols,
                   xs + (size_t)j0 * (cols / 32), xm + (size_t)j0 * (cols / 16),
                   x ? x + (size_t)j0 * cols : NULL, n, out + (size_t)j0 * rows + r, (size_t)rows);
        }
        return;
    }
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

/* ---------- the conversions of the safetensors loader (np_gemma/st_qwen4.py) ---------- */

/* One row of cols float values to Q8_0 blocks (as quantize_row_q8_0 of ggml). */
static void kq_row_to_q8_0(const float *x, int cols, uint8_t *dst)
{
    for (int b = 0; b < cols / 32; ++b) {
        const float *v = x + 32 * b;
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) {
            amax = fabsf(v[i]) > amax ? fabsf(v[i]) : amax;
        }
        float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        uint8_t *blk = dst + (size_t)b * 34;
        uint16_t hd = _cvtss_sh(d, 0);
        memcpy(blk, &hd, 2);
        for (int i = 0; i < 32; ++i) {
            blk[2 + i] = (uint8_t)(int8_t)lrintf(v[i] * id);
        }
    }
}

/* rows x cols values (bf16 if bf16, else float32) to Q8_0 rows. */
void kq_to_q8_0(const void *src, int bf16, int64_t rows, int cols, uint8_t *dst)
{
    #pragma omp parallel
    {
        float *tmp = (float *)malloc((size_t)cols * 4);
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            const float *x;
            if (bf16) {
                const uint16_t *h = (const uint16_t *)src + (size_t)r * cols;
                for (int c = 0; c < cols; ++c) {
                    tmp[c] = kq_bf16(h[c]);
                }
                x = tmp;
            } else {
                x = (const float *)src + (size_t)r * cols;
            }
            kq_row_to_q8_0(x, cols, dst + (size_t)r * kq_row_bytes(KQ_Q8_0, cols));
        }
        free(tmp);
    }
}

/* The NVFP4 matrices of n experts (ModelOpt: w, rows x cols / 2 bytes, value
 * 2i in the low 4 bits of byte i; s, rows x cols / 16 E4M3 scales; g, the
 * float32 scale) to KQ_NV4 rows: dst gets n matrices of rows rows. wp and sp
 * hold the addresses of w and s of each expert. */
void kq_nv4_pack(const int64_t *wp, const int64_t *sp, const float *g, int n, int rows, int cols,
                 uint8_t *dst)
{
    size_t rb = kq_row_bytes(KQ_NV4, cols);
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < (int64_t)n * rows; ++x) {
        int e = (int)(x / rows), r = (int)(x % rows);
        const uint8_t *w = (const uint8_t *)(intptr_t)wp[e] + (size_t)r * (cols / 2);
        const uint8_t *s = (const uint8_t *)(intptr_t)sp[e] + (size_t)r * (cols / 16);
        uint8_t *o = dst + (size_t)x * rb;
        memset(o, 0, rb);
        memcpy(o + (size_t)cols / 2, s, (size_t)cols / 16);       /* the scales, in order */
        memcpy(o + (size_t)cols / 2 + (size_t)cols / 16, &g[e], 4);
        for (int b = 0; b < cols / 32; ++b) {
            const uint8_t *src = w + (size_t)b * 16;     /* 32 values, 2 in each byte */
            uint8_t *q = o + (size_t)16 * b;
            for (int j = 0; j < 16; ++j) {
                int lo = (src[j / 2] >> (4 * (j % 2))) & 15;             /* value j */
                int hi = (src[8 + j / 2] >> (4 * (j % 2))) & 15;         /* value j + 16 */
                q[j] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
}

/* The NVFP4 matrices of n experts (as kq_nv4_pack; rows a multiple of 16)
 * to KQ_NVX groups: dst gets n matrices of rows / 16 groups. */
void kq_nvx_pack(const int64_t *wp, const int64_t *sp, const float *g, int n, int rows, int cols,
                 uint8_t *dst)
{
    size_t gb = 16 * kq_row_bytes(KQ_NVX, cols);
    int ng = rows / 16;
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < (int64_t)n * ng; ++x) {
        int e = (int)(x / ng), g0 = 16 * (int)(x % ng);
        const uint8_t *w = (const uint8_t *)(intptr_t)wp[e];
        const uint8_t *s = (const uint8_t *)(intptr_t)sp[e];
        uint8_t *o = dst + (size_t)x * gb;
        memset(o, 0, gb);
        memcpy(o, &g[e], 4);
        for (int b = 0; b < cols / 32; ++b) {
            uint8_t *blk = o + 16 + (size_t)b * KQ_NVX_BB;
            for (int r = 0; r < 16; ++r) {
                const uint8_t *wr = w + (size_t)(g0 + r) * (cols / 2) + 16 * b;
                for (int v = 0; v < 32; ++v) {
                    int c = (wr[v / 2] >> (4 * (v % 2))) & 15;      /* value 2i: the low 4 bits */
                    blk[32 * (v / 4) + 4 * (r % 8) + v % 4] |= (uint8_t)(c << (r < 8 ? 0 : 4));
                }
                const uint8_t *sr = s + (size_t)(g0 + r) * (cols / 16) + 2 * b;
                blk[256 + r] = sr[0];
                blk[272 + r] = sr[1];
            }
        }
    }
}

/* Copy n rows of bytes bytes from the addresses addrs (rows of a memory map
 * of a file: the n-gram table of the safetensors checkpoint). Many threads
 * take the page faults at the same time, so the disk has many reads in flight
 * (a loop of one thread waits for each read). */
void kq_gather(const int64_t *addrs, int64_t n, int bytes, uint8_t *out)
{
    #pragma omp parallel for num_threads(64) schedule(dynamic, 16)
    for (int64_t i = 0; i < n; ++i) {
        memcpy(out + (size_t)i * bytes, (const void *)(intptr_t)addrs[i], (size_t)bytes);
    }
}

/* float32 or bfloat16 rows (bf) to KQ_F32X16 or KQ_BF16X16: dst gets
 * (rows + 15) / 16 groups; the rows past rows are zeros. */
void kq_pack_x16f(const uint8_t *src, int bf, int64_t rows, int cols, uint8_t *dst)
{
    int es = bf ? 2 : 4;
    int64_t ng = (rows + 15) / 16;
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < ng; ++g) {
        uint8_t *o = dst + (size_t)g * 16 * cols * es;
        for (int r = 0; r < 16; ++r) {
            int64_t row = 16 * g + r;
            for (int c = 0; c < cols; ++c) {
                if (row < rows) {
                    memcpy(o + ((size_t)16 * c + r) * es, src + ((size_t)row * cols + c) * es, es);
                } else {
                    memset(o + ((size_t)16 * c + r) * es, 0, es);
                }
            }
        }
    }
}

/* Q8_0 rows (rows a multiple of 16) to KQ_Q8X16 (see the formats). */
void kq_pack_q8x16(const uint8_t *q8, int64_t rows, int cols, uint8_t *dst)
{
    int nb = cols / 32;
    size_t rb = (size_t)nb * 34;
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < rows / 16; ++g) {
        for (int b = 0; b < nb; ++b) {
            uint8_t *o = dst + ((size_t)g * nb + b) * 576;
            for (int r = 0; r < 16; ++r) {
                const uint8_t *blk = q8 + (size_t)(16 * g + r) * rb + (size_t)b * 34;
                memcpy(o + 2 * r, blk, 2);
                int16_t sum = 0;
                for (int k = 0; k < 8; ++k) {
                    for (int u = 0; u < 4; ++u) {
                        o[64 + 64 * k + 4 * r + u] = blk[2 + 4 * k + u];
                        sum += (int8_t)blk[2 + 4 * k + u];
                    }
                }
                memcpy(o + 32 + 2 * r, &sum, 2);
            }
        }
    }
}

/* The float values of rows ids of a matrix (the embeddings). */
void kq_rows(const uint8_t *w, int type, int cols, const int64_t *ids, int n, float *out)
{
    size_t rb = kq_row_bytes(type, cols);
    int bv = kq_block32(type) ? 32 : 256;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const uint8_t *row = w + (size_t)ids[i] * rb;
        if (type == KQ_NVX) {
            kq_nvx_values(w + (size_t)(ids[i] & ~(int64_t)15) * rb, (int)(ids[i] & 15), cols,
                          out + (size_t)i * cols);
            continue;
        }
        if (type == KQ_F32) {
            memcpy(out + (size_t)i * cols, row, (size_t)cols * 4);
            continue;
        }
        for (int b = 0; b < cols / bv; ++b) {
            kq_block_values(type, row, cols, b, out + (size_t)i * cols + (size_t)b * bv);
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
    n += P * (hidden / 16) * 4 + P * (inner / 16) * 4;  /* the sums of KQ_NVX */
    return n + 64 * 16;
}

/* rows rows (a multiple of 16) from row r of a matrix on n tokens, as
 * kq_rows4: KQ_NVX in groups of 16 (xn: kq_nvx_xsum of the tokens), the
 * other types 4 rows at a time. */
static void kq_rows_n(const uint8_t *w, size_t rb, int type, int nr, int cols, const int8_t *xq,
                      const float *xs, const float *xm, const int32_t *xn, int n, float *out,
                      size_t ostride)
{
    for (int i = 0; i < nr; i += type == KQ_NVX ? 16 : 4) {
        if (type == KQ_NVX) {
            kq_nvx_rows(w + (size_t)i * rb, cols, xq, xs, xn, n, out + i, ostride);
        } else {
            kq_rows4(w + (size_t)i * rb, rb, type, cols, xq, xs, xm, NULL, n, out + i, ostride);
        }
    }
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
 * for none). The weight of the shared expert is sigmoid(shared_logit).
 * kcount (or null) holds k, for one token. */
static void kq_moe_body(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                        const float *val, int t, int k, int experts, const int64_t *mats,
                        const float *shared_logit, int hidden, int inner, uint8_t *scratch,
                        float *out, const int32_t *kcount)
{
    if (kcount != NULL) {
        /* One token with a count of experts in memory: the cold experts of a
         * GPU step (GP_HOT_SPLIT writes the count). */
        k = *kcount;
    }
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
    int32_t *xn = (int32_t *)ma_take(&p, (size_t)P * (hidden / 16) * 4);
    int32_t *an = (int32_t *)ma_take(&p, (size_t)P * (inner / 16) * 4);
    /* KQ_NVX: tasks of 16 rows (a group) for all the matrices of the layer */
    int ngu = G.type == KQ_NVX ? 16 : 4, ndn = D.type == KQ_NVX ? 16 : 4;
    moe_sort_pairs(ids, t, k, experts, shared, cnt, start, used, pair_tok, pair_of, nused_p);
    int nu = *nused_p;
    P = start[ne];          /* the pairs with an expert */
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        int j = pair_tok[q];
        memcpy(xq + (size_t)q * hidden, hq + (size_t)j * hidden, (size_t)hidden);
        memcpy(xs + (size_t)q * nph, hs + (size_t)j * nph, (size_t)nph * 4);
        memcpy(xm + (size_t)q * 2 * nph, hm + (size_t)j * 2 * nph, (size_t)nph * 8);
        if (ngu == 16) {
            kq_nvx_xsum(xq + (size_t)q * hidden, 1, hidden, xn + (size_t)q * (hidden / 16));
        }
    }
    /* A group: the experts have different counts of tokens, so the tasks
     * go to the threads as they finish. With a static split, the threads
     * were idle about 28% of the time. A decode step: the same work in each
     * task, a static split. Each thread sets the schedule of its own
     * loops. */
    omp_set_schedule(t > 1 ? omp_sched_dynamic : omp_sched_static, t > 1 ? 8 : 0);
    /* Gate and up: 2 * inner rows of each used expert, in tasks of 4 rows
     * (16 for KQ_NVX). */
    #pragma omp for schedule(runtime)
    for (int x = 0; x < nu * 2 * inner / ngu; ++x) {
        int e = used[x / (2 * inner / ngu)], rr = (x % (2 * inner / ngu)) * ngu, isup = rr >= inner;
        int r = rr % inner;
        kq_mat m = e < experts ? (isup ? U : G) : (isup ? SU : SG);
        size_t rb = kq_row_bytes(m.type, hidden);
        const uint8_t *w = m.w + (e < experts ? (size_t)e * inner * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        kq_rows_n(w + (size_t)r * rb, rb, m.type, ngu, hidden, xq + (size_t)s0 * hidden,
                  xs + (size_t)s0 * nph, xm + (size_t)s0 * 2 * nph, xn + (size_t)s0 * (hidden / 16), n,
                  act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
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
        if (ndn == 16) {
            kq_nvx_xsum(aq + (size_t)q * inner, 1, inner, an + (size_t)q * (inner / 16));
        }
    }
    /* Down: hidden rows of each used expert, in tasks of 4 rows (16 for
     * KQ_NVX). */
    #pragma omp for schedule(runtime)
    for (int x = 0; x < nu * hidden / ndn; ++x) {
        int e = used[x / (hidden / ndn)], r = (x % (hidden / ndn)) * ndn;
        kq_mat m = e < experts ? D : SD;
        size_t rb = kq_row_bytes(m.type, inner);
        const uint8_t *w = m.w + (e < experts ? (size_t)e * hidden * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        kq_rows_n(w + (size_t)r * rb, rb, m.type, ndn, inner, aq + (size_t)s0 * inner,
                  as + (size_t)s0 * npi, am + (size_t)s0 * 2 * npi, an + (size_t)s0 * (inner / 16), n,
                  de + (size_t)s0 * hidden + r, (size_t)hidden);
    }
    moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
}

void kq_moe(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
            const float *val, int t, int k, int experts, const int64_t *mats,
            const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL);
}
