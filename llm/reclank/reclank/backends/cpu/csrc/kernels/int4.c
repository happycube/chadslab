/* ---------- 4-bit weights ----------
 * W is packed as unsigned bytes. One byte holds two 4-bit values. A group of
 * 32 values uses 16 bytes. For group g, byte j holds:
 *     low nibble  = value g*32 + j
 *     high nibble = value g*32 + 16 + j
 * A value is a signed 4-bit number (the nibble minus 8). A group has one
 * float32 scale. The group size is fixed at 32.
 *
 * The kernel converts the nibbles to float32 and uses a fused multiply and
 * add. The kernel does not write the values to a buffer first. A scalar loop
 * is too slow. The kernel then becomes compute bound.
 */

/* Turn packed nibbles into signed bytes. A value is a 4-bit two's complement
 * number, so nibble - 8 sign-extends it. Do this in the byte lanes before
 * the widening. The operation is then 2 instructions for 16 values, not 2
 * instructions for each widened vector. */
#if GEMMA_X86
static inline __m128i i4_sign_bytes(__m128i nib, __m128i b8)
{
    return _mm_sub_epi8(nib, b8);
}
#endif

#if GEMMA_X86 && defined(__AVX512F__)
/* Return the dot product of one packed 4-bit row and one float32 row.
 * scales holds one value for each group of 32 columns.
 */
static inline float dot_i4_f32(const uint8_t *w, const float *scales,
                               const float *x, int n)
{
    __m512 acc = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        __m128i b = _mm_loadu_si128((const __m128i *)(w + (size_t)g * 18 + 2));
        __m512 lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            i4_sign_bytes(_mm_and_si128(b, mask), b8)));
        __m512 hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
            i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8)));
        __m512 p = _mm512_fmadd_ps(
            _mm512_loadu_ps(x + (size_t)g * 32), lo,
            _mm512_mul_ps(_mm512_loadu_ps(x + (size_t)g * 32 + 16), hi));
        acc = _mm512_fmadd_ps(p, _mm512_set1_ps(scales[g]), acc);
    }
    return _mm512_reduce_add_ps(acc);
}
#elif GEMMA_X86
/* Return the dot product of one packed 4-bit row and one float32 row. Use AVX2.
 * The helper hsum256_ps is above, in the int8 part. */
static inline float dot_i4_f32(const uint8_t *w, const float *scales,
                               const float *x, int n)
{
    __m256 acc = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        __m128i b = _mm_loadu_si128((const __m128i *)(w + (size_t)g * 18 + 2));
        __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
        __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
        __m256 l0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        __m256 l1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        __m256 h0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        __m256 h1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        __m256 p0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + (size_t)g * 32), l0,
                                    _mm256_mul_ps(_mm256_loadu_ps(x + (size_t)g * 32 + 8), l1));
        __m256 p1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + (size_t)g * 32 + 16), h0,
                                    _mm256_mul_ps(_mm256_loadu_ps(x + (size_t)g * 32 + 24), h1));
        acc = _mm256_fmadd_ps(_mm256_add_ps(p0, p1), _mm256_set1_ps(scales[g]), acc);
    }
    return hsum256_ps(acc);
}
#else
/* Return the dot product of one packed 4-bit row and one float32 row. Use scalar code. */
static inline float dot_i4_f32(const uint8_t *w, const float *scales,
                               const float *x, int n)
{
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
        int g = k / 32;
        int p = k % 32;
        int byte = w[(size_t)g * 18 + 2 + (p & 15)];
        int nib = (p < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
        acc += x[k] * (float)(nib - 8) * scales[g];
    }
    return acc;
}
#endif

/* ---------- four 4-bit rows for each x block ----------
 * The x row uses 4 bytes for each weight of a row. Thus x traffic controls the
 * time for a matrix with many columns. This loop reads four weight rows for
 * each x block. The x traffic falls by four times.
 */

#if GEMMA_X86 && defined(__AVX512F__)
/* Put the 32 values of 16 packed bytes in two float32 vectors. */
static inline void i4_pair(__m128i b, __m128i mask, __m128i b8,
                           __m512 *lo, __m512 *hi)
{
    *lo = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
        i4_sign_bytes(_mm_and_si128(b, mask), b8)));
    *hi = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
        i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8)));
}

static inline void dot4_i4_f32(const uint8_t *w, int stride, const float *scales,
                               const float *x, int n, float *r)
{
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        __m512 xlo = _mm512_loadu_ps(x + (size_t)g * 32);
        __m512 xhi = _mm512_loadu_ps(x + (size_t)g * 32 + 16);
        const uint8_t *p = w + (size_t)g * 18 + 2;
        __m128i b0 = _mm_loadu_si128((const __m128i *)(p));
        __m128i b1 = _mm_loadu_si128((const __m128i *)(p + stride));
        __m128i b2 = _mm_loadu_si128((const __m128i *)(p + (size_t)2 * stride));
        __m128i b3 = _mm_loadu_si128((const __m128i *)(p + (size_t)3 * stride));
        __m512 l0, h0, l1, h1, l2, h2, l3, h3;
        i4_pair(b0, mask, b8, &l0, &h0);
        i4_pair(b1, mask, b8, &l1, &h1);
        i4_pair(b2, mask, b8, &l2, &h2);
        i4_pair(b3, mask, b8, &l3, &h3);
        a0 = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l0, _mm512_mul_ps(xhi, h0)),
                             _mm512_set1_ps(scales[g]), a0);
        a1 = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l1, _mm512_mul_ps(xhi, h1)),
                             _mm512_set1_ps(scales[(size_t)groups + g]), a1);
        a2 = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l2, _mm512_mul_ps(xhi, h2)),
                             _mm512_set1_ps(scales[(size_t)2 * groups + g]), a2);
        a3 = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l3, _mm512_mul_ps(xhi, h3)),
                             _mm512_set1_ps(scales[(size_t)3 * groups + g]), a3);
    }
    r[0] = _mm512_reduce_add_ps(a0);
    r[1] = _mm512_reduce_add_ps(a1);
    r[2] = _mm512_reduce_add_ps(a2);
    r[3] = _mm512_reduce_add_ps(a3);
}
#elif GEMMA_X86
/* Put the 32 values of 16 packed bytes in four float32 vectors. */
static inline void i4_pair256(__m128i b, __m128i mask, __m128i b8,
                              __m256 *lo0, __m256 *lo1, __m256 *hi0, __m256 *hi1)
{
    __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
    __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
    *lo0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
    *lo1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
    *hi0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
    *hi1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
}

static inline __m256 i4_row256(const uint8_t *p, __m256 x0, __m256 x1,
                               __m256 x2, __m256 x3, __m128i mask, __m128i b8)
{
    __m256 l0, l1, h0, h1;
    i4_pair256(_mm_loadu_si128((const __m128i *)p), mask, b8, &l0, &l1, &h0, &h1);
    return _mm256_add_ps(_mm256_fmadd_ps(x0, l0, _mm256_mul_ps(x1, l1)),
                         _mm256_fmadd_ps(x2, h0, _mm256_mul_ps(x3, h1)));
}

static inline void dot4_i4_f32(const uint8_t *w, int stride, const float *scales,
                               const float *x, int n, float *r)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        __m256 x0 = _mm256_loadu_ps(x + (size_t)g * 32);
        __m256 x1 = _mm256_loadu_ps(x + (size_t)g * 32 + 8);
        __m256 x2 = _mm256_loadu_ps(x + (size_t)g * 32 + 16);
        __m256 x3 = _mm256_loadu_ps(x + (size_t)g * 32 + 24);
        const uint8_t *p = w + (size_t)g * 18 + 2;
        a0 = _mm256_fmadd_ps(i4_row256(p, x0, x1, x2, x3, mask, b8),
                             _mm256_set1_ps(scales[g]), a0);
        a1 = _mm256_fmadd_ps(i4_row256(p + stride, x0, x1, x2, x3, mask, b8),
                             _mm256_set1_ps(scales[(size_t)groups + g]), a1);
        a2 = _mm256_fmadd_ps(i4_row256(p + (size_t)2 * stride, x0, x1, x2, x3, mask, b8),
                             _mm256_set1_ps(scales[(size_t)2 * groups + g]), a2);
        a3 = _mm256_fmadd_ps(i4_row256(p + (size_t)3 * stride, x0, x1, x2, x3, mask, b8),
                             _mm256_set1_ps(scales[(size_t)3 * groups + g]), a3);
    }
    r[0] = hsum256_ps(a0);
    r[1] = hsum256_ps(a1);
    r[2] = hsum256_ps(a2);
    r[3] = hsum256_ps(a3);
}
#else
static inline void dot4_i4_f32(const uint8_t *w, int stride, const float *scales,
                               const float *x, int n, float *r)
{
    int groups = n / 32;
    for (int j = 0; j < 4; ++j) {
        r[j] = dot_i4_f32(w + (size_t)j * stride, scales + (size_t)j * groups, x, n);
    }
}
#endif

/* ---------- int8 x for the int4 products of a few tokens (AVX2) ----------
 * AVX2 has no VNNI and 8 float lanes, so the float products above take a
 * decode to float32 and 4 fma for each 32 weights and each row. Here x is
 * int8 (a scale for each 32 values, as kq_quant_part), and a group of 32 is
 * one vpmaddubsw and one vpmaddwd (the sign of w moves to x), then one fma
 * with the scale of w times the scale of x. llama.cpp does the same on
 * AVX2.
 *
 * A body that uses it quantizes its x rows first (i4q_begin: one thread,
 * then the barrier of omp single) into the scratch of its team; the dot
 * kernels find the int8 row of an x row by its address (i4q_find). An x
 * row that is not in the scratch takes the float kernel. The kernel of one
 * token and the kernel of a group do the same steps for each token, so a
 * verify group gives the bits of the steps. RECLANK_I4Q8=0 turns it off. */
#if GEMMA_X86 && !defined(__AVX512F__)
#define I4Q_TEAMS 8

typedef struct {
    const float *base;      /* the first x row, or NULL */
    size_t stride;          /* floats from one x row to the next */
    int rows, cols;
    int8_t *q;              /* rows x cols */
    float *s;               /* rows x cols / 32 */
    int32_t *n;             /* -8 times the sum of each 32 (KQ_Q4X), stride cols / 16 */
    size_t cap;             /* the values that q holds */
    int16_t *q16;           /* RECLANK_DECODE_X16: the int16 rows too (KQ_Q4X) */
    float *s16;
    int8_t *h16;            /* and their planes (kq_planes16_to): one row */
    uint8_t *l16;
    int32_t *n16;
    size_t cap16;
} i4q_reg;

static i4q_reg i4q_regs[I4Q_TEAMS];

static int i4q_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_I4Q8");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* The KQ_Q4X products of a step (and of a verify group) take int16 x
 * (gemma_quant_group32_i16, kq_q4x_rows16), as the prompt with int16 x:
 * twice the instructions of int8 x. The default; RECLANK_DECODE_X16=0
 * keeps int8 x. */
static int i4q16_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_DECODE_X16");
        on = !(v && v[0] == '0');
    }
    return on;
}

static inline float gemma_quant_group32_i16(const float *x, int16_t *q);
static void kq_q4x_rows16(const uint8_t *wg, int cols, const int16_t *xq, const float *xs, int n,
                          float *out, size_t ostride);
static void kq_planes16_to(const int16_t *xq, int cols, int8_t *kp_h, uint8_t *kp_l, int32_t *kp_n);
static void kq_q4x_rows_p16(const uint8_t *wg, int cols, const int8_t *kp_h, const uint8_t *kp_l,
                            const int32_t *kp_n, const float *xs, float *out);

/* The scratch of this team: the part of rc_run_parts, or 0. */
static inline i4q_reg *i4q_cur(void)
{
    int lvl = omp_get_level();
    int p = lvl > 1 ? omp_get_ancestor_thread_num(lvl - 1) : 0;
    return &i4q_regs[p % I4Q_TEAMS];
}

/* 32 values of x to int8: s = max |x| / 127, q = round(x / s). */
static inline void i4q_quant32(const float *x, int8_t *q, float *s)
{
    const __m256 sign = _mm256_set1_ps(-0.0f);
    __m256 v0 = _mm256_loadu_ps(x), v1 = _mm256_loadu_ps(x + 8);
    __m256 v2 = _mm256_loadu_ps(x + 16), v3 = _mm256_loadu_ps(x + 24);
    __m256 m = _mm256_max_ps(_mm256_max_ps(_mm256_andnot_ps(sign, v0), _mm256_andnot_ps(sign, v1)),
                             _mm256_max_ps(_mm256_andnot_ps(sign, v2), _mm256_andnot_ps(sign, v3)));
    __m128 h = _mm_max_ps(_mm256_castps256_ps128(m), _mm256_extractf128_ps(m, 1));
    h = _mm_max_ps(h, _mm_movehl_ps(h, h));
    h = _mm_max_ss(h, _mm_shuffle_ps(h, h, 1));
    float mx = _mm_cvtss_f32(h);
    float inv = mx > 0.f ? 127.f / mx : 0.f;
    *s = mx / 127.f;
    __m256 iv = _mm256_set1_ps(inv);
    __m256i i0 = _mm256_cvtps_epi32(_mm256_mul_ps(v0, iv)), i1 = _mm256_cvtps_epi32(_mm256_mul_ps(v1, iv));
    __m256i i2 = _mm256_cvtps_epi32(_mm256_mul_ps(v2, iv)), i3 = _mm256_cvtps_epi32(_mm256_mul_ps(v3, iv));
    __m256i a = _mm256_packs_epi32(i0, i1), b = _mm256_packs_epi32(i2, i3);
    __m256i c = _mm256_packs_epi16(a, b);
    c = _mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
    _mm256_storeu_si256((__m256i *)q, c);
}

/* Quantize rows x rows of cols values (stride floats apart) into the scratch
 * of the team. All the threads of the team call it. */
static void i4q_begin(const float *x, size_t stride, int rows, int cols)
{
    i4q_reg *R = i4q_cur();
    #pragma omp single
    {
        R->base = NULL;
        if (i4q_on() && cols % 32 == 0 && rows > 0) {
            size_t need = (size_t)rows * (size_t)cols;
            if (need > R->cap) {
                free(R->q);
                free(R->s);
                free(R->n);
                R->q = (int8_t *)malloc(need);
                R->s = (float *)malloc(need / 32 * sizeof(float));
                R->n = (int32_t *)malloc(need / 16 * sizeof(int32_t));
                R->cap = R->q && R->s && R->n ? need : 0;
            }
            if (R->cap >= need) {
                for (int r = 0; r < rows; ++r) {
                    const float *xr = x + (size_t)r * stride;
                    int8_t *qr = R->q + (size_t)r * cols;
                    float *sr = R->s + (size_t)r * (cols / 32);
                    int32_t *nr = R->n + (size_t)r * (cols / 16);
                    for (int g = 0; g < cols / 32; ++g) {
                        i4q_quant32(xr + 32 * g, qr + 32 * g, sr + g);
                        int32_t sum = 0;
                        for (int u = 0; u < 32; ++u) {
                            sum += qr[32 * g + u];
                        }
                        nr[g] = -8 * sum;
                    }
                }
                if (i4q16_on() && R->cap16 < need) {
                    free(R->q16);
                    free(R->s16);
                    free(R->h16);
                    free(R->l16);
                    free(R->n16);
                    R->q16 = (int16_t *)malloc(need * sizeof(int16_t));
                    R->s16 = (float *)malloc(need / 32 * sizeof(float));
                    R->h16 = (int8_t *)malloc(need + 64);
                    R->l16 = (uint8_t *)malloc(need + 64);
                    R->n16 = (int32_t *)malloc((need / 32 + 16) * sizeof(int32_t));
                    R->cap16 = R->q16 && R->s16 && R->h16 && R->l16 && R->n16 ? need : 0;
                }
                if (i4q16_on() && R->cap16 >= need) {
                    for (int r = 0; r < rows; ++r) {
                        for (int g = 0; g < cols / 32; ++g) {
                            R->s16[(size_t)r * (cols / 32) + g] = gemma_quant_group32_i16(
                                x + (size_t)r * stride + 32 * g, R->q16 + (size_t)r * cols + 32 * g);
                        }
                    }
                    if (rows == 1) {
                        kq_planes16_to(R->q16, cols, R->h16, R->l16, R->n16);
                    }
                }
                R->base = x;
                R->stride = stride;
                R->rows = rows;
                R->cols = cols;
            }
        }
    }
}

/* The int8 row and its scales for the x row x of n values, or NULL. xn
 * (if not NULL) gets the sums for KQ_Q4X. */
static inline const int8_t *i4q_find2(const float *x, int n, const float **xs,
                                      const int32_t **xn)
{
    const i4q_reg *R = i4q_cur();
    if (R->base == NULL || R->cols != n || x < R->base) {
        return NULL;
    }
    size_t d = (size_t)(x - R->base);
    size_t r = R->stride ? d / R->stride : 0;
    if ((R->stride ? d % R->stride : d) != 0 || r >= (size_t)R->rows) {
        return NULL;
    }
    *xs = R->s + r * (size_t)(n / 32);
    if (xn != NULL) {
        *xn = R->n + r * (size_t)(n / 16);
    }
    return R->q + r * (size_t)n;
}

static inline const int8_t *i4q_find(const float *x, int n, const float **xs)
{
    return i4q_find2(x, n, xs, NULL);
}

/* The int16 row (RECLANK_DECODE_X16) and its scales for the x row x, or NULL. */
static inline const int16_t *i4q_find16(const float *x, int n, const float **xs)
{
    const i4q_reg *R = i4q_cur();
    const float *s8;
    const int8_t *q = i4q_find2(x, n, &s8, NULL);
    if (q == NULL || !i4q16_on() || R->cap16 < (size_t)R->rows * (size_t)R->cols) {
        return NULL;
    }
    size_t r = (size_t)(q - R->q) / (size_t)n;
    *xs = R->s16 + r * (size_t)(n / 32);
    return R->q16 + r * (size_t)n;
}

/* The 32 signed values (-8 to 7) of a Q4_0 group, in the order of x. */
static inline __m256i i4q_w32(const uint8_t *p)
{
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m128i b = _mm_loadu_si128((const __m128i *)p);
    __m256i w = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(b, 4), m4), _mm_and_si128(b, m4));
    return _mm256_sub_epi8(w, _mm256_set1_epi8(8));
}

/* The int32 sums (8 lanes) of 32 products of w (signed, abs aw) and x. */
static inline __m256 i4q_dot32(__m256i aw, __m256i w, __m256i x)
{
    __m256i p = _mm256_maddubs_epi16(aw, _mm256_sign_epi8(x, w));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}

/* One row, one token (the tail rows). */
static inline float dot_i4_q8(const uint8_t *w, const float *scales, const int8_t *xq,
                              const float *xs, int n)
{
    __m256 acc = _mm256_setzero_ps();
    for (int g = 0; g < n / 32; ++g) {
        __m256i wv = i4q_w32(w + (size_t)g * 18 + 2);
        __m256i xv = _mm256_loadu_si256((const __m256i *)(xq + (size_t)g * 32));
        acc = _mm256_fmadd_ps(i4q_dot32(_mm256_abs_epi8(wv), wv, xv),
                              _mm256_set1_ps(scales[g] * xs[g]), acc);
    }
    return hsum256_ps(acc);
}

/* Four rows (stride bytes apart) for NT tokens (a constant 1 or 2, so the
 * sums stay in registers). Each (row, token) adds its groups in the order
 * of dot_i4_q8. */
static inline __attribute__((always_inline)) void dot4_i4_q8_n(
    const uint8_t *w, int stride, const float *scales, const int8_t *const *xq,
    const float *const *xs, const int NT, int n, float *r)
{
    int groups = n / 32;
    __m256 a[2][4];
    for (int t = 0; t < NT; ++t) {
        for (int j = 0; j < 4; ++j) {
            a[t][j] = _mm256_setzero_ps();
        }
    }
    for (int g = 0; g < groups; ++g) {
        __m256i xv[2];
        for (int t = 0; t < NT; ++t) {
            xv[t] = _mm256_loadu_si256((const __m256i *)(xq[t] + (size_t)g * 32));
        }
        for (int j = 0; j < 4; ++j) {
            __m256i wv = i4q_w32(w + (size_t)j * stride + (size_t)g * 18 + 2);
            __m256i aw = _mm256_abs_epi8(wv);
            float sw = scales[(size_t)j * groups + g];
            for (int t = 0; t < NT; ++t) {
                a[t][j] = _mm256_fmadd_ps(i4q_dot32(aw, wv, xv[t]), _mm256_set1_ps(sw * xs[t][g]),
                                          a[t][j]);
            }
        }
    }
    for (int t = 0; t < NT; ++t) {
        for (int j = 0; j < 4; ++j) {
            r[t * 4 + j] = hsum256_ps(a[t][j]);
        }
    }
}
#endif

/* The four-row and one-row dots of the bodies below: on AVX2 the int8 x of
 * i4q_begin when the body quantized x, else the float kernels. */
#if GEMMA_X86 && !defined(__AVX512F__)
static inline void I4_DOT4(const uint8_t *w, int stride, const float *scales, const float *x,
                           int n, float *r)
{
    const float *xs;
    const int8_t *q = i4q_find(x, n, &xs);
    if (q != NULL) {
        dot4_i4_q8_n(w, stride, scales, &q, &xs, 1, n, r);
    } else {
        dot4_i4_f32(w, stride, scales, x, n, r);
    }
}

static inline float I4_DOT1(const uint8_t *w, const float *scales, const float *x, int n)
{
    const float *xs;
    const int8_t *q = i4q_find(x, n, &xs);
    return q != NULL ? dot_i4_q8(w, scales, q, xs, n) : dot_i4_f32(w, scales, x, n);
}
#define I4Q_BEGIN(x, stride, rows, cols) i4q_begin((x), (stride), (rows), (cols))
#else
#define I4_DOT4 dot4_i4_f32
#define I4_DOT1 dot_i4_f32
#define I4Q_BEGIN(x, stride, rows, cols) ((void)0)
#endif

/* ---------- int4 tile for a small group of tokens ----------
 * A mixture-of-experts layer gives a small group of tokens to each expert.
 * The group is often smaller than the token block of the multi-level GEMM. The
 * one-row dot then reads x again for each weight row. This tile reads the x
 * block one time for I4T_MR rows. It also decodes each weight group one time
 * for I4T_TB tokens.
 */

#if GEMMA_X86 && defined(__AVX512F__)
#define I4T_MR 16
#define I4T_TB 16
#elif GEMMA_X86
#define I4T_MR 8
#define I4T_TB 8
#else
#define I4T_MR 1
#define I4T_TB 1
#endif

#if GEMMA_X86 && defined(__AVX512F__)
static inline void gemma_int4_gemm_tile(const uint8_t *w, const float *scales,
                                        const float *xt, float *out,
                                        int rows, int cols, int tokens,
                                        int i0, int t0)
{
    __m512 acc[I4T_MR];
    for (int r = 0; r < I4T_MR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    const int groups = cols / 32;
    const int stride = (cols / 32) * 18;
    /* Mask the token lanes that pass the end of the group. */
    int ntok = tokens - t0;
    if (ntok > I4T_TB) {
        ntok = I4T_TB;
    }
    const __mmask16 km = ntok >= I4T_TB ? (__mmask16)0xFFFF : (__mmask16)((1u << ntok) - 1);
    for (int g = 0; g < groups; ++g) {
        float wf[I4T_MR][32];
        for (int r = 0; r < I4T_MR; ++r) {
            const uint8_t *p = w + (size_t)(i0 + r) * stride + (size_t)g * 18 + 2;
            const float sc = scales[(size_t)(i0 + r) * groups + g];
            __m128i b = _mm_loadu_si128((const __m128i *)p);
            __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
            __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
            _mm512_storeu_ps(wf[r] + 0, _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), _mm512_set1_ps(sc)));
            _mm512_storeu_ps(wf[r] + 16, _mm512_mul_ps(
                _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), _mm512_set1_ps(sc)));
        }
        for (int k = 0; k < 32; ++k) {
            __m512 xv = _mm512_maskz_loadu_ps(km, xt + (size_t)(g * 32 + k) * tokens + t0);
            for (int r = 0; r < I4T_MR; ++r) {
                acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(wf[r][k]), xv, acc[r]);
            }
        }
    }
    float tmp[I4T_TB];
    for (int r = 0; r < I4T_MR; ++r) {
        _mm512_storeu_ps(tmp, acc[r]);
        for (int t = 0; t < ntok; ++t) {
            out[(size_t)(t0 + t) * rows + i0 + r] = tmp[t];
        }
    }
}
#elif GEMMA_X86
static inline void gemma_int4_gemm_tile(const uint8_t *w, const float *scales,
                                        const float *xt, float *out,
                                        int rows, int cols, int tokens,
                                        int i0, int t0)
{
    __m256 acc[I4T_MR];
    for (int r = 0; r < I4T_MR; ++r) {
        acc[r] = _mm256_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    const int groups = cols / 32;
    const int stride = (cols / 32) * 18;
    for (int g = 0; g < groups; ++g) {
        float wf[I4T_MR][32];
        for (int r = 0; r < I4T_MR; ++r) {
            const uint8_t *p = w + (size_t)(i0 + r) * stride + (size_t)g * 18 + 2;
            const float sc = scales[(size_t)(i0 + r) * groups + g];
            __m128i b = _mm_loadu_si128((const __m128i *)p);
            __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
            __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
            __m256 scv = _mm256_set1_ps(sc);
            _mm256_storeu_ps(wf[r] + 0, _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), scv));
            _mm256_storeu_ps(wf[r] + 8, _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))), scv));
            _mm256_storeu_ps(wf[r] + 16, _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), scv));
            _mm256_storeu_ps(wf[r] + 24, _mm256_mul_ps(
                _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))), scv));
        }
        for (int k = 0; k < 32; ++k) {
            __m256 xv = _mm256_loadu_ps(xt + (size_t)(g * 32 + k) * tokens + t0);
            for (int r = 0; r < I4T_MR; ++r) {
                acc[r] = _mm256_fmadd_ps(_mm256_set1_ps(wf[r][k]), xv, acc[r]);
            }
        }
    }
    float tmp[I4T_TB];
    for (int r = 0; r < I4T_MR; ++r) {
        _mm256_storeu_ps(tmp, acc[r]);
        for (int t = 0; t < I4T_TB; ++t) {
            out[(size_t)(t0 + t) * rows + i0 + r] = tmp[t];
        }
    }
}
#else
static inline void gemma_int4_gemm_tile(const uint8_t *w, const float *scales,
                                        const float *xt, float *out,
                                        int rows, int cols, int tokens,
                                        int i0, int t0)
{
    (void)w; (void)scales; (void)xt; (void)out; (void)rows; (void)cols;
    (void)tokens; (void)i0; (void)t0;
}
#endif

void rc_int4_gemm_tile_run(const uint8_t *w, const float *scales,
                              const float *x, const float *xt, float *out,
                              int rows, int cols, int tokens)
{
    const int stride = (cols / 32) * 18;
    const int groups = cols / 32;
    const int mr = rows / I4T_MR * I4T_MR;
#if GEMMA_X86 && defined(__AVX512F__)
    /* The tile masks the token lanes past the end of the group. Thus it
     * handles every token count. */
    const int tb = tokens;
#else
    const int tb = tokens / I4T_TB * I4T_TB;
#endif
#if GEMMA_X86
    if (tb > 0 && mr > 0) {
        const int nt = (tb + I4T_TB - 1) / I4T_TB;
        #pragma omp parallel for schedule(static) collapse(2)
        for (int bi = 0; bi < mr / I4T_MR; ++bi) {
            for (int bt = 0; bt < nt; ++bt) {
                gemma_int4_gemm_tile(w, scales, xt, out, rows, cols, tokens,
                                     bi * I4T_MR, bt * I4T_TB);
            }
        }
    }
#endif
    /* The rows that do not fill a tile and the token tail use the four-row
     * dot. */
    const int blocks = (rows + 3) / 4;
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < blocks; ++b) {
        const int i = b * 4;
        const int nrow = rows - i < 4 ? rows - i : 4;
        for (int t = 0; t < tokens; ++t) {
            if (i < mr && t < tb) {
                continue;
            }
            if (nrow == 4) {
                float r[4];
                dot4_i4_f32(w + (size_t)i * stride, stride, scales + (size_t)i * groups,
                            x + (size_t)t * cols, cols, r);
                out[(size_t)t * rows + i + 0] = r[0];
                out[(size_t)t * rows + i + 1] = r[1];
                out[(size_t)t * rows + i + 2] = r[2];
                out[(size_t)t * rows + i + 3] = r[3];
            } else {
                for (int j = 0; j < nrow; ++j) {
                    out[(size_t)t * rows + i + j] =
                        dot_i4_f32(w + (size_t)(i + j) * stride,
                                   scales + (size_t)(i + j) * groups,
                                   x + (size_t)t * cols, cols);
                }
            }
        }
    }
}

static int gemma_int4_rows4 = 1;

/* Select the four-row loop (1) or the one-row loop (0). Use this for a test. */
void rc_int4_set_rows4(int on)
{
    gemma_int4_rows4 = on ? 1 : 0;
}

/* backends/cpu/csrc/kernels/kq (included later): the 16 rows of a KQ_Q4X group on float32 x,
 * and on int8 x. */
static void kq_q4x_rows_f(const uint8_t *wg, int cols, const float *x, size_t xstride, int n,
                          float *out, size_t ostride);
static void kq_q4x_rows(const uint8_t *wg, int cols, const int8_t *xq, const float *xs,
                        const int32_t *xn, int n, float *out, size_t ostride);

/* The 16 rows of a KQ_Q4X group on n rows of x (xstride == cols, the rows
 * of one i4q_begin): on AVX2 the int8 x of i4q_begin, else float32 x. */
static inline void q4x_rows_x(const uint8_t *wg, int cols, const float *x, size_t xstride, int n,
                              float *out, size_t ostride)
{
#if GEMMA_X86 && !defined(__AVX512F__)
    const float *xs, *xs_last;
    const int32_t *xn;
    const int8_t *q = xstride == (size_t)cols ? i4q_find2(x, cols, &xs, &xn) : NULL;
    /* the n rows must all be in the scratch (they follow each other there) */
    if (q != NULL && (n == 1 || i4q_find(x + (size_t)(n - 1) * xstride, cols, &xs_last) != NULL)) {
        const float *s16;
        const int16_t *q16 = i4q_find16(x, cols, &s16);
        if (q16 != NULL) {
            const i4q_reg *R = i4q_cur();
            if (n == 1 && R->rows == 1) {
                /* a step: the planes of i4q_begin */
                kq_q4x_rows_p16(wg, cols, R->h16, R->l16, R->n16, s16, out);
            } else {
                kq_q4x_rows16(wg, cols, q16, s16, n, out, ostride);
            }
            return;
        }
        kq_q4x_rows(wg, cols, q, xs, xn, n, out, ostride);
        return;
    }
#endif
    kq_q4x_rows_f(wg, cols, x, xstride, n, out, ostride);
}

/* scales NULL: w is the KQ_Q4X copy of the matrix (ops.q4x_pack_model;
 * program.py gives it to the records of one token on the CPU): groups of 16
 * rows, float32 x, the same products. */
static void gemma_int4_linear_body(const uint8_t *w, const float *scales, const float *x, float *out,
                       int rows, int cols, int tokens, int group)
{
    if (scales == NULL) {
        size_t gb = (size_t)16 * (cols / 32) * 18;
        I4Q_BEGIN(x, (size_t)cols, tokens, cols);
        #pragma omp for schedule(static)
        for (int g = 0; g < rows / 16; ++g) {
            q4x_rows_x(w + (size_t)g * gb, cols, x, (size_t)cols, tokens, out + 16 * g,
                       (size_t)rows);
        }
        return;
    }
    /* The fast dot uses a group of 32 values. ops.linear_int4 sends only that
     * group size. */
    (void)group;
    int groups = cols / 32;
    int stride = (cols / 32) * 18;
    if (gemma_int4_rows4 && tokens == 1) {
        I4Q_BEGIN(x, (size_t)cols, 1, cols);
        int blocks = (rows + 3) / 4;
        #pragma omp for schedule(static)
        for (int b = 0; b < blocks; ++b) {
            int i = b * 4;
            int left = rows - i;
            if (left >= 4) {
                float r[4];
                I4_DOT4(w + (size_t)i * (size_t)stride, stride,
                        scales + (size_t)i * (size_t)groups, x, cols, r);
                out[i] = r[0];
                out[i + 1] = r[1];
                out[i + 2] = r[2];
                out[i + 3] = r[3];
            } else {
                for (int j = 0; j < left; ++j) {
                    out[i + j] = I4_DOT1(w + (size_t)(i + j) * (size_t)stride,
                                         scales + (size_t)(i + j) * (size_t)groups,
                                         x, cols);
                }
            }
        }
        return;
    }
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wi = w + (size_t)i * (size_t)stride;
        const float *si = scales + (size_t)i * (size_t)groups;
        for (int t = 0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i4_f32(wi, si, x + (size_t)t * (size_t)cols, cols);
        }
    }
}

void rc_int4_linear(const uint8_t *w, const float *scales, const float *x, float *out,
                       int rows, int cols, int tokens, int group)
{
    #pragma omp parallel
    gemma_int4_linear_body(w, scales, x, out, rows, cols, tokens, group);
}

/* ---------- four int4 matrices on one x row ----------
 * The query, the key, and the value projection of one attention layer share
 * the x row. Three calls then start three OpenMP regions for the same x. This
 * kernel runs up to four matrices in one region. A null weight pointer skips
 * a matrix. The MLP gate and up projection can use the same kernel.
 */
static void gemma_int4_multi4_body(const uint8_t *w0, const float *s0, float *o0, int rows0,
                       const uint8_t *w1, const float *s1, float *o1, int rows1,
                       const uint8_t *w2, const float *s2, float *o2, int rows2,
                       const uint8_t *w3, const float *s3, float *o3, int rows3,
                       const float *x, int cols)
{
    int groups = cols / 32;
    size_t stride = (size_t)groups * 18;
    /* a matrix with s NULL is a KQ_Q4X copy: a task is a group of 16 rows */
    int b0 = w0 ? (s0 ? (rows0 + 3) / 4 : rows0 / 16) : 0;
    int b1 = w1 ? (s1 ? (rows1 + 3) / 4 : rows1 / 16) : 0;
    int b2 = w2 ? (s2 ? (rows2 + 3) / 4 : rows2 / 16) : 0;
    int b3 = w3 ? (s3 ? (rows3 + 3) / 4 : rows3 / 16) : 0;
    long total = (long)b0 + (long)b1 + (long)b2 + (long)b3;
    I4Q_BEGIN(x, (size_t)cols, 1, cols);
    #pragma omp for schedule(static)
    for (long t = 0; t < total; ++t) {
        const uint8_t *w = w0;
        const float *s = s0;
        float *o = o0;
        int rows = rows0;
        long u = t;
        if (u >= b0) {
            u -= b0;
            if (u < b1) {
                w = w1; s = s1; o = o1; rows = rows1;
            } else {
                u -= b1;
                if (u < b2) {
                    w = w2; s = s2; o = o2; rows = rows2;
                } else {
                    u -= b2;
                    w = w3; s = s3; o = o3; rows = rows3;
                }
            }
        }
        if (s == NULL) {
            q4x_rows_x(w + (size_t)u * 16 * stride, cols, x, (size_t)cols, 1, o + 16 * u,
                       (size_t)rows);
            continue;
        }
        int i = (int)u * 4;
        if (i + 4 <= rows) {
            float r[4];
            I4_DOT4(w + (size_t)i * stride, (int)stride,
                    s + (size_t)i * (size_t)groups, x, cols, r);
            o[i] = r[0];
            o[i + 1] = r[1];
            o[i + 2] = r[2];
            o[i + 3] = r[3];
        } else {
            for (int q = i; q < rows; ++q) {
                o[q] = I4_DOT1(w + (size_t)q * stride,
                               s + (size_t)q * (size_t)groups, x, cols);
            }
        }
    }
}

void rc_int4_multi4(const uint8_t *w0, const float *s0, float *o0, int rows0,
                       const uint8_t *w1, const float *s1, float *o1, int rows1,
                       const uint8_t *w2, const float *s2, float *o2, int rows2,
                       const uint8_t *w3, const float *s3, float *o3, int rows3,
                       const float *x, int cols)
{
    #pragma omp parallel
    gemma_int4_multi4_body(w0, s0, o0, rows0, w1, s1, o1, rows1, w2, s2, o2, rows2, w3, s3, o3, rows3, x, cols);
}

/* ---------- int4 mixture of experts ----------
 * A mixture-of-experts layer selects a small set of experts for each token.
 * A one-row call for each expert then starts one OpenMP region for each
 * expert. The thread team is small when the expert has few rows, and the
 * main thread does the region work many times.
 *
 * This kernel takes the list of selected experts. The parallel loop covers
 * the full work of all the experts at once. Thus the thread team is large,
 * and the caller starts one region for the whole layer.
 *
 * W holds one matrix for each expert in the Q4_0 block layout. scales holds
 * one float32 scale for each group of 32 columns. ids gives the matrix index
 * for each job. Job j computes
 *     out[j * rows + r] = sum_k W[ids[j]][r][k] * x[j * xstride + k].
 * An xstride of 0 gives the same x to every job. That is a decode step.
 */
static void gemma_int4_moe_gemv_body(const uint8_t *w, const float *scales,
                         const float *x, const int *ids, int jobs,
                         float *out, int rows, int cols, int xstride)
{
    int groups = cols / 32;
    size_t stride = (size_t)groups * 18;
    size_t expert_bytes = (size_t)rows * stride;
    size_t expert_scales = (size_t)rows * (size_t)groups;
    int blocks = (rows + 3) / 4;
    long total = (long)jobs * (long)blocks;
    I4Q_BEGIN(x, (size_t)xstride, xstride ? jobs : 1, cols);
    #pragma omp for schedule(static)
    for (long t = 0; t < total; ++t) {
        int j = (int)(t / blocks);
        int i = (int)(t % blocks) * 4;
        int e = ids[j];
        const uint8_t *wj = w + (size_t)e * expert_bytes;
        const float *sj = scales + (size_t)e * expert_scales;
        const float *xj = x + (size_t)j * (size_t)xstride;
        float *oj = out + (size_t)j * (size_t)rows;
        int left = rows - i;
        if (left >= 4) {
            float r[4];
            I4_DOT4(wj + (size_t)i * stride, (int)stride,
                    sj + (size_t)i * (size_t)groups, xj, cols, r);
            oj[i] = r[0];
            oj[i + 1] = r[1];
            oj[i + 2] = r[2];
            oj[i + 3] = r[3];
        } else {
            for (int q = i; q < rows; ++q) {
                oj[q] = I4_DOT1(wj + (size_t)q * stride,
                                sj + (size_t)q * (size_t)groups, xj, cols);
            }
        }
    }
}

void rc_int4_moe_gemv(const uint8_t *w, const float *scales,
                         const float *x, const int *ids, int jobs,
                         float *out, int rows, int cols, int xstride)
{
    #pragma omp parallel
    gemma_int4_moe_gemv_body(w, scales, x, ids, jobs, out, rows, cols, xstride);
}

/* ---------- int4 GEMV for a small group of tokens ----------
 * An MTP verify step runs the target on two to eight tokens. The prompt
 * kernels are tuned for a large token block, and a decode kernel reads the
 * weights again for each token. These kernels read a weight block one time
 * for all the tokens.
 *
 * The steps for one token are the steps of the one-token kernel, in the same
 * order. Thus each token gets the same result, bit for bit, as a decode step.
 * The MTP decode then gives the same text as the plain decode.
 *
 * xs gives the address of the x row of each token. The dot writes
 * r[t * 4 + j] for token t and weight row j.
 */
#define I4MT_T 4
#define I4MT_MAX 16

#if GEMMA_X86 && defined(__AVX512F__)
static inline void dot4_i4_f32_mt(const uint8_t *w, int stride, const float *scales,
                                  const float *const *xs, int nt, int n, float *r)
{
    __m512 a[I4MT_T][4];
    for (int t = 0; t < nt; ++t) {
        a[t][0] = _mm512_setzero_ps();
        a[t][1] = _mm512_setzero_ps();
        a[t][2] = _mm512_setzero_ps();
        a[t][3] = _mm512_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 32;
    for (int g = 0; g < groups; ++g) {
        const uint8_t *p = w + (size_t)g * 18 + 2;
        __m512 l0, h0, l1, h1, l2, h2, l3, h3;
        i4_pair(_mm_loadu_si128((const __m128i *)(p)), mask, b8, &l0, &h0);
        i4_pair(_mm_loadu_si128((const __m128i *)(p + stride)), mask, b8, &l1, &h1);
        i4_pair(_mm_loadu_si128((const __m128i *)(p + (size_t)2 * stride)), mask, b8, &l2, &h2);
        i4_pair(_mm_loadu_si128((const __m128i *)(p + (size_t)3 * stride)), mask, b8, &l3, &h3);
        __m512 s0 = _mm512_set1_ps(scales[g]);
        __m512 s1 = _mm512_set1_ps(scales[(size_t)groups + g]);
        __m512 s2 = _mm512_set1_ps(scales[(size_t)2 * groups + g]);
        __m512 s3 = _mm512_set1_ps(scales[(size_t)3 * groups + g]);
        for (int t = 0; t < nt; ++t) {
            __m512 xlo = _mm512_loadu_ps(xs[t] + (size_t)g * 32);
            __m512 xhi = _mm512_loadu_ps(xs[t] + (size_t)g * 32 + 16);
            a[t][0] = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l0, _mm512_mul_ps(xhi, h0)), s0, a[t][0]);
            a[t][1] = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l1, _mm512_mul_ps(xhi, h1)), s1, a[t][1]);
            a[t][2] = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l2, _mm512_mul_ps(xhi, h2)), s2, a[t][2]);
            a[t][3] = _mm512_fmadd_ps(_mm512_fmadd_ps(xlo, l3, _mm512_mul_ps(xhi, h3)), s3, a[t][3]);
        }
    }
    for (int t = 0; t < nt; ++t) {
        r[t * 4 + 0] = _mm512_reduce_add_ps(a[t][0]);
        r[t * 4 + 1] = _mm512_reduce_add_ps(a[t][1]);
        r[t * 4 + 2] = _mm512_reduce_add_ps(a[t][2]);
        r[t * 4 + 3] = _mm512_reduce_add_ps(a[t][3]);
    }
}
#else
/* AVX2 and scalar: run the one-token dot for each token. The four weight
 * rows stay in the L1 cache, so the memory reads them one time. */
static inline void dot4_i4_f32_mt(const uint8_t *w, int stride, const float *scales,
                                  const float *const *xs, int nt, int n, float *r)
{
    for (int t = 0; t < nt; ++t) {
        dot4_i4_f32(w, stride, scales, xs[t], n, r + t * 4);
    }
}
#endif

/* Rows i to i + 3 (or fewer at the end) of one matrix for nt tokens. out has
 * the row stride ostride for each token. */
static inline void i4_rows_mt(const uint8_t *w, const float *s, int rows, int cols,
                              int i, const float *const *xs, float *const *outs, int nt)
{
    int groups = cols / 32;
    size_t stride = (size_t)groups * 18;
#if GEMMA_X86 && !defined(__AVX512F__)
    {
        /* int8 x of i4q_begin: 2 tokens at a time; each token has the steps
         * of the one-token kernel */
        const int8_t *q[I4MT_MAX];
        const float *qs[I4MT_MAX];
        int all = nt <= I4MT_MAX;
        for (int t = 0; t < nt && all; ++t) {
            q[t] = i4q_find(xs[t], cols, &qs[t]);
            all = q[t] != NULL;
        }
        if (all) {
            for (int t0 = 0; t0 < nt; t0 += 2) {
                int n = nt - t0 < 2 ? nt - t0 : 2;
                if (i + 4 <= rows) {
                    float r[8];
                    if (n == 2) {
                        dot4_i4_q8_n(w + (size_t)i * stride, (int)stride, s + (size_t)i * groups,
                                     q + t0, qs + t0, 2, cols, r);
                    } else {
                        dot4_i4_q8_n(w + (size_t)i * stride, (int)stride, s + (size_t)i * groups,
                                     q + t0, qs + t0, 1, cols, r);
                    }
                    for (int t = 0; t < n; ++t) {
                        float *o = outs[t0 + t];
                        o[i] = r[t * 4];
                        o[i + 1] = r[t * 4 + 1];
                        o[i + 2] = r[t * 4 + 2];
                        o[i + 3] = r[t * 4 + 3];
                    }
                } else {
                    for (int t = 0; t < n; ++t) {
                        for (int k = i; k < rows; ++k) {
                            outs[t0 + t][k] = dot_i4_q8(w + (size_t)k * stride, s + (size_t)k * groups,
                                                        q[t0 + t], qs[t0 + t], cols);
                        }
                    }
                }
            }
            return;
        }
    }
#endif
    for (int t0 = 0; t0 < nt; t0 += I4MT_T) {
        int n = nt - t0 < I4MT_T ? nt - t0 : I4MT_T;
        if (i + 4 <= rows) {
            float r[I4MT_T * 4];
            dot4_i4_f32_mt(w + (size_t)i * stride, (int)stride,
                           s + (size_t)i * (size_t)groups, xs + t0, n, cols, r);
            for (int t = 0; t < n; ++t) {
                float *o = outs[t0 + t];
                o[i] = r[t * 4];
                o[i + 1] = r[t * 4 + 1];
                o[i + 2] = r[t * 4 + 2];
                o[i + 3] = r[t * 4 + 3];
            }
        } else {
            for (int t = 0; t < n; ++t) {
                for (int q = i; q < rows; ++q) {
                    outs[t0 + t][q] = dot_i4_f32(w + (size_t)q * stride,
                                                 s + (size_t)q * (size_t)groups,
                                                 xs[t0 + t], cols);
                }
            }
        }
    }
}


/* x is (tokens, cols) and out is (tokens, rows). */
/* ldo: the values between the rows of two tokens in out (0: rows); a part
 * of a step in parts writes its rows of each token into the output of all
 * the rows (reclank/models/gemma4/parts.py). */
static void gemma_int4_linear_mt_body(const uint8_t *w, const float *scales, const float *x,
                          float *out, int rows, int cols, int tokens, int ldo)
{
    size_t ld = ldo > 0 ? (size_t)ldo : (size_t)rows;
    if (scales == NULL) {
        /* a KQ_Q4X copy (see gemma_int4_linear_body): each token has the
         * operations of a step, so a verify group gives the bits of steps */
        size_t gb = (size_t)16 * (cols / 32) * 18;
        I4Q_BEGIN(x, (size_t)cols, tokens, cols);
        #pragma omp for schedule(static)
        for (int g = 0; g < rows / 16; ++g) {
            q4x_rows_x(w + (size_t)g * gb, cols, x, (size_t)cols, tokens, out + 16 * g, ld);
        }
        return;
    }
    const float *xs[I4MT_MAX];
    float *outs[I4MT_MAX];
    for (int t = 0; t < tokens; ++t) {
        xs[t] = x + (size_t)t * (size_t)cols;
        outs[t] = out + (size_t)t * ld;
    }
    int blocks = (rows + 3) / 4;
    I4Q_BEGIN(x, (size_t)cols, tokens, cols);
    #pragma omp for schedule(static)
    for (int b = 0; b < blocks; ++b) {
        i4_rows_mt(w, scales, rows, cols, b * 4, xs, outs, tokens);
    }
}

void rc_int4_linear_mt(const uint8_t *w, const float *scales, const float *x,
                          float *out, int rows, int cols, int tokens)
{
    #pragma omp parallel
    gemma_int4_linear_mt_body(w, scales, x, out, rows, cols, tokens, 0);
}

/* Up to four matrices on the same x rows. Matrix k writes (tokens, rows_k). */
static void gemma_int4_multi4_mt_body(const uint8_t *w0, const float *s0, float *o0, int rows0,
                          const uint8_t *w1, const float *s1, float *o1, int rows1,
                          const uint8_t *w2, const float *s2, float *o2, int rows2,
                          const uint8_t *w3, const float *s3, float *o3, int rows3,
                          const float *x, int cols, int tokens, const int *ldo)
{
    /* ldo (or NULL): the values between the rows of two tokens in the output
     * of each matrix (0: its rows), as gemma_int4_linear_mt_body */
    const uint8_t *ws[4] = {w0, w1, w2, w3};
    const float *ss[4] = {s0, s1, s2, s3};
    float *os[4] = {o0, o1, o2, o3};
    int rs[4] = {rows0, rows1, rows2, rows3};
    size_t ld[4];
    for (int k = 0; k < 4; ++k) {
        ld[k] = ldo != NULL && ldo[k] > 0 ? (size_t)ldo[k] : (size_t)rs[k];
    }
    long nb[4];
    long total = 0;
    for (int k = 0; k < 4; ++k) {
        /* s NULL: a KQ_Q4X copy, a task is a group of 16 rows */
        nb[k] = ws[k] ? (ss[k] ? (rs[k] + 3) / 4 : rs[k] / 16) : 0;
        total += nb[k];
    }
    const float *xs[I4MT_MAX];
    for (int t = 0; t < tokens; ++t) {
        xs[t] = x + (size_t)t * (size_t)cols;
    }
    I4Q_BEGIN(x, (size_t)cols, tokens, cols);
    #pragma omp for schedule(static)
    for (long u = 0; u < total; ++u) {
        int k = 0;
        long v = u;
        while (v >= nb[k]) {
            v -= nb[k];
            ++k;
        }
        if (ss[k] == NULL) {
            q4x_rows_x(ws[k] + (size_t)v * 16 * (cols / 32) * 18, cols, x, (size_t)cols, tokens,
                       os[k] + 16 * v, ld[k]);
            continue;
        }
        float *outs[I4MT_MAX];
        for (int t = 0; t < tokens; ++t) {
            outs[t] = os[k] + (size_t)t * ld[k];
        }
        i4_rows_mt(ws[k], ss[k], rs[k], cols, (int)v * 4, xs, outs, tokens);
    }
}

void rc_int4_multi4_mt(const uint8_t *w0, const float *s0, float *o0, int rows0,
                          const uint8_t *w1, const float *s1, float *o1, int rows1,
                          const uint8_t *w2, const float *s2, float *o2, int rows2,
                          const uint8_t *w3, const float *s3, float *o3, int rows3,
                          const float *x, int cols, int tokens)
{
    #pragma omp parallel
    gemma_int4_multi4_mt_body(w0, s0, o0, rows0, w1, s1, o1, rows1, w2, s2, o2, rows2, w3, s3, o3, rows3, x, cols, tokens,
                              NULL);
}

/* The selected experts of a small group of tokens. Job j is one expert,
 * ids[j], with the pairs poff[j] to poff[j + 1] - 1. Pair p reads the x row
 * xi[p] (the stride is xstride) and writes the out row p. Each expert is read
 * one time for all of its tokens. */
static void gemma_int4_moe_gemv_mt_body(const uint8_t *w, const float *scales, const float *x,
                            const int *ids, const int *poff, const int *xi, int jobs,
                            float *out, int rows, int cols, int xstride)
{
    int groups = cols / 32;
    size_t expert_bytes = (size_t)rows * (size_t)groups * 18;
    size_t expert_scales = (size_t)rows * (size_t)groups;
    int blocks = (rows + 3) / 4;
    long total = (long)jobs * (long)blocks;
    int xrows = 1;
    for (int p = 0; p < poff[jobs]; ++p) {
        xrows = xi[p] + 1 > xrows ? xi[p] + 1 : xrows;
    }
    I4Q_BEGIN(x, (size_t)xstride, xrows, cols);
    #pragma omp for schedule(static)
    for (long u = 0; u < total; ++u) {
        int j = (int)(u / blocks);
        int i = (int)(u % blocks) * 4;
        int e = ids[j];
        int np = poff[j + 1] - poff[j];
        const float *xs[I4MT_MAX];
        float *outs[I4MT_MAX];
        for (int q = 0; q < np; ++q) {
            int p = poff[j] + q;
            xs[q] = x + (size_t)xi[p] * (size_t)xstride;
            outs[q] = out + (size_t)p * (size_t)rows;
        }
        i4_rows_mt(w + (size_t)e * expert_bytes, scales + (size_t)e * expert_scales,
                   rows, cols, i, xs, outs, np);
    }
}

void rc_int4_moe_gemv_mt(const uint8_t *w, const float *scales, const float *x,
                            const int *ids, const int *poff, const int *xi, int jobs,
                            float *out, int rows, int cols, int xstride)
{
    #pragma omp parallel
    gemma_int4_moe_gemv_mt_body(w, scales, x, ids, poff, xi, jobs, out, rows, cols, xstride);
}

void rc_gelu_mul(const float *x, float *out, int rows, int inner);

/* The gate and up projection of the experts for a group of tokens. Then the
 * GELU and the multiply. act holds 2 * inner values for each pair, and out
 * holds inner values for each pair. */
void rc_moe_gemv_gelu_mt(const uint8_t *w, const float *scales, const float *x,
                            const int *ids, const int *poff, const int *xi, int jobs,
                            float *act, float *out, int rows, int cols, int xstride,
                            int inner)
{
    rc_int4_moe_gemv_mt(w, scales, x, ids, poff, xi, jobs, act, rows, cols, xstride);
    rc_gelu_mul(act, out, poff[jobs], inner);
}

/* ---------- int4 GEMM for a long prompt ----------
 * The one-row dot decodes the packed weights again for each token. This GEMM
 * decodes a row block to a float32 A panel one time and then uses the panel
 * for every token block. The decode cost then falls by the number of token
 * blocks. The A panel holds the group scale, so the micro kernel is a plain
 * float32 multiply and add.
 */

#if GEMMA_X86 && defined(__AVX512F__)
#define I4_MC 64
#define I4_KC 128
#define I4_NC 64
#define I4_MR 16
#define I4_NR 16
#else
/* AVX2 has 8 float lanes. The A panel and the B panel are float32, so keep
 * them small enough for the L1 cache. */
#define I4_MC 64
#define I4_KC 64
#define I4_NC 32
#define I4_MR 8
#define I4_NR 8
#endif

/* Decode ng groups (32 columns each) of one packed 4-bit row to float32. The
 * function applies the group scale. The A panel is column-major: the values of
 * one column are contiguous with a stride of lda. The micro kernel then reads
 * the rows of one column as one vector. */
static inline void i4_decode_row(const uint8_t *w, const float *scales,
                                 int ng, int lda, float *dst)
{
#if !GEMMA_X86
    for (int g = 0; g < ng; ++g) {
        const uint8_t *q = w + (size_t)g * 18 + 2;
        float sc = scales[g];
        float *col = dst + (size_t)g * 32 * lda;
        for (int j = 0; j < 16; ++j) {
            col[(size_t)j * lda] = (float)((int)(q[j] & 15) - 8) * sc;
            col[(size_t)(j + 16) * lda] = (float)((int)(q[j] >> 4) - 8) * sc;
        }
    }
#else
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    for (int g = 0; g < ng; ++g) {
        __m128i b = _mm_loadu_si128((const __m128i *)(w + (size_t)g * 18 + 2));
        float sc = scales[g];
        __m128i lo = i4_sign_bytes(_mm_and_si128(b, mask), b8);
        __m128i hi = i4_sign_bytes(_mm_and_si128(_mm_srli_epi16(b, 4), mask), b8);
        float buf[32];
#if defined(__AVX512F__)
        _mm512_storeu_ps(buf + 0, _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(lo)), _mm512_set1_ps(sc)));
        _mm512_storeu_ps(buf + 16, _mm512_mul_ps(
            _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(hi)), _mm512_set1_ps(sc)));
#else
        _mm256_storeu_ps(buf + 0, _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo)), _mm256_set1_ps(sc)));
        _mm256_storeu_ps(buf + 8, _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8))),
            _mm256_set1_ps(sc)));
        _mm256_storeu_ps(buf + 16, _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi)), _mm256_set1_ps(sc)));
        _mm256_storeu_ps(buf + 24, _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8))),
            _mm256_set1_ps(sc)));
#endif
        float *col = dst + (size_t)g * 32 * lda;
        for (int j = 0; j < 32; ++j) {
            col[(size_t)j * lda] = buf[j];
        }
    }
#endif
}

#if GEMMA_X86
/* One micro tile for an already decoded float32 A panel. */
static inline void gemma_ml_micro_f32(const float *a, int lda, int mi,
                                      const float *b, int ldb, int nj, int kc,
                                      float *out, int rows, int m0, int n0, int add)
{
#if defined(__AVX512F__)
    __m512 acc[I4_NR];
    for (int n = 0; n < I4_NR; ++n) {
        acc[n] = _mm512_setzero_ps();
    }
    for (int k = 0; k < kc; ++k) {
        __m512 av = _mm512_loadu_ps(a + (size_t)k * lda + mi);
        const float *brow = b + (size_t)k * ldb + nj;
        for (int n = 0; n < I4_NR; ++n) {
            acc[n] = _mm512_fmadd_ps(av, _mm512_set1_ps(brow[n]), acc[n]);
        }
    }
    for (int n = 0; n < I4_NR; ++n) {
        float *op = out + (size_t)(n0 + nj + n) * (size_t)rows + (m0 + mi);
        __m512 v = acc[n];
        if (add) {
            v = _mm512_add_ps(_mm512_loadu_ps(op), v);
        }
        _mm512_storeu_ps(op, v);
    }
#else
    __m256 acc[I4_NR];
    for (int n = 0; n < I4_NR; ++n) {
        acc[n] = _mm256_setzero_ps();
    }
    for (int k = 0; k < kc; ++k) {
        __m256 av = _mm256_loadu_ps(a + (size_t)k * lda + mi);
        const float *brow = b + (size_t)k * ldb + nj;
        for (int n = 0; n < I4_NR; ++n) {
            acc[n] = _mm256_fmadd_ps(av, _mm256_set1_ps(brow[n]), acc[n]);
        }
    }
    for (int n = 0; n < I4_NR; ++n) {
        float *op = out + (size_t)(n0 + nj + n) * (size_t)rows + (m0 + mi);
        __m256 v = acc[n];
        if (add) {
            v = _mm256_add_ps(_mm256_loadu_ps(op), v);
        }
        _mm256_storeu_ps(op, v);
    }
#endif
}
#endif

void rc_int4_gemm(const uint8_t *w, const float *scales, const float *x,
                     const float *xt, float *out, int rows, int cols, int tokens)
{
#if GEMMA_X86
    int mb = rows / I4_MC;
    int nb = tokens / I4_NC;
    int kb = (cols + I4_KC - 1) / I4_KC;
    #pragma omp parallel
    {
        float *abuf = (float *)malloc((size_t)I4_MC * I4_KC * sizeof(float));
        float *bbuf = (float *)malloc((size_t)I4_KC * I4_NC * sizeof(float));
        if (abuf != NULL && bbuf != NULL) {
            #pragma omp for schedule(static)
            for (int bi = 0; bi < mb; ++bi) {
                int m0 = bi * I4_MC;
                for (int ki = 0; ki < kb; ++ki) {
                    int k0 = ki * I4_KC;
                    int kc = cols - k0 < I4_KC ? cols - k0 : I4_KC;
                    int kfast = kc & ~31;
                    int g0 = k0 / 32;
                    /* Decode the A panel one time for all the token blocks. */
                    for (int mm = 0; mm < I4_MC; ++mm) {
                        const uint8_t *wi = w + (size_t)(m0 + mm) * ((size_t)(cols / 32) * 18);
                        const float *si = scales + (size_t)(m0 + mm) * (size_t)(cols / 32);
                        i4_decode_row(wi + (size_t)g0 * 18, si + g0, kfast / 32,
                                      I4_MC, abuf + mm);
                    }
                    /* The columns that do not fill a group of 32 use scalar code. */
                    for (int kk = kfast; kk < kc; ++kk) {
                        int g = (k0 + kk) / 32;
                        int p = (k0 + kk) % 32;
                        for (int mm = 0; mm < I4_MC; ++mm) {
                            const uint8_t *wi = w + (size_t)(m0 + mm) * ((size_t)(cols / 32) * 18);
                            const float *si = scales + (size_t)(m0 + mm) * (size_t)(cols / 32);
                            int byte = wi[(size_t)g * 18 + 2 + (p & 15)];
                            int nib = (p < 16) ? (byte & 0x0F) : ((byte >> 4) & 0x0F);
                            abuf[(size_t)kk * I4_MC + mm] = (float)(nib - 8) * si[g];
                        }
                    }
                    for (int ni = 0; ni < nb; ++ni) {
                        int n0 = ni * I4_NC;
                        for (int kk = 0; kk < kc; ++kk) {
                            memcpy(bbuf + (size_t)kk * I4_NC,
                                   xt + (size_t)(k0 + kk) * (size_t)tokens + n0,
                                   I4_NC * sizeof(float));
                        }
                        for (int mi = 0; mi < I4_MC; mi += I4_MR) {
                            for (int njj = 0; njj < I4_NC; njj += I4_NR) {
                                gemma_ml_micro_f32(abuf, I4_MC, mi, bbuf, I4_NC, njj,
                                                   kc, out, rows, m0, n0, ki > 0);
                            }
                        }
                    }
                }
            }
        }
        free(abuf);
        free(bbuf);
    }
    /* The rows and the tokens that do not fill a block use the one-row dot. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const uint8_t *wi = w + (size_t)i * ((size_t)(cols / 32) * 18);
        const float *si = scales + (size_t)i * (size_t)(cols / 32);
        int t0 = (i < mb * I4_MC) ? nb * I4_NC : 0;
        for (int t = t0; t < tokens; ++t) {
            out[(size_t)t * (size_t)rows + i] =
                dot_i4_f32(wi, si, x + (size_t)t * (size_t)cols, cols);
        }
    }
#else
    (void)w; (void)scales; (void)x; (void)xt; (void)out;
    (void)rows; (void)cols; (void)tokens;
#endif
}
