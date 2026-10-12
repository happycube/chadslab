/* ---------- the compressed-tensors packed layout ----------
 *
 * The E4B mobile-ct checkpoint is a compressed-tensors file, and it is not the
 * layout of the int4 kernels above. The file stores a quantized weight as
 * int32 words. Element k of a row starts at bit k * bits, counted from the
 * start of the row. When bits divides 32 no element crosses a word: for 4
 * bits, word k / 8 holds the value in nibble k % 8 and the low nibble comes
 * first.
 *
 * The value is a two's complement number. The packing added a bias of
 * 2 ** (bits - 1) first, so subtract 8 from a 4-bit value and 2 from a 2-bit
 * value.
 *
 * These matrices use the "channel" strategy of compressed-tensors. One
 * float32 scale covers the whole row, so the scale multiplies the finished
 * dot product. That is one multiply for each row, not one for each group of
 * 32 columns as in the layout above.
 *
 *   out[t][r] = scale[r] * sum over k of x[t][k] * q[r][k]
 *
 * The kernel reads the words where the file put them, and it never writes a
 * float32 copy of a weight. So a token reads 4 bits for each weight instead
 * of the 32 bits of the float32 copy. That is the point of the kernel: the
 * file layout is the runtime layout, and a memory map over the file is then
 * enough.
 *
 * The scale is applied after the sum. Scaling each group and then summing, as
 * the NumPy path does, rounds differently. The difference is in the last bits
 * of the float32 mantissa; check_ct_kernel.py measures it.
 */

#if GEMMA_X86
/* Put the 8 values of one 4-bit word into 8 signed bytes, in element order. */
static inline __m128i ct_sign4(__m128i w)
{
    const __m128i mask = _mm_set1_epi8(0x0F);
    __m128i lo = _mm_and_si128(w, mask);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(w, 4), mask);
    /* lo holds the low nibble of each byte and hi holds the high nibble. The
     * byte interleave then puts them in the order of the elements. */
    return _mm_sub_epi8(_mm_unpacklo_epi8(lo, hi), _mm_set1_epi8(8));
}

/* Put the 16 values of one 2-bit word into 16 signed bytes, in element order. */
static inline __m128i ct_sign2(__m128i w)
{
    const __m128i mask = _mm_set1_epi8(0x03);
    __m128i t0 = _mm_and_si128(w, mask);
    __m128i t1 = _mm_and_si128(_mm_srli_epi16(w, 2), mask);
    __m128i t2 = _mm_and_si128(_mm_srli_epi16(w, 4), mask);
    __m128i t3 = _mm_and_si128(_mm_srli_epi16(w, 6), mask);
    /* Byte j of the word holds four values: t0 holds the one at 4j + 0, t1 the
     * one at 4j + 1, and so on. The first interleave pairs the values of each
     * byte, and the second interleave orders the four bytes. The word has only
     * four useful bytes, so a single 16-bit interleave of the low half holds
     * all 16 values. */
    __m128i a = _mm_unpacklo_epi8(t0, t1);
    __m128i b = _mm_unpacklo_epi8(t2, t3);
    return _mm_sub_epi8(_mm_unpacklo_epi16(a, b), _mm_set1_epi8(2));
}
#endif

/* Return one 4-bit value of a row as a signed int. Use it for a tail. */
static inline int ct_q4(const uint32_t *w, int k)
{
    return (int)((w[(size_t)k >> 3] >> ((k & 7) * 4)) & 0x0Fu) - 8;
}

/* Return one 2-bit value of a row as a signed int. Use it for a tail. */
static inline int ct_q2(const uint32_t *w, int k)
{
    return (int)((w[(size_t)k >> 4] >> ((k & 15) * 2)) & 0x03u) - 2;
}

/* The dot product of one 4-bit row with one float32 row. */
#if GEMMA_X86 && defined(__AVX512F__)
static inline float ct_dot4_f32(const uint32_t *w, const float *x, int n)
{
    __m512 acc = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        /* Two words hold 16 values, and 16 float32 values fill one zmm. */
        __m128i w2 = _mm_loadl_epi64((const __m128i *)(w + (size_t)g * 2));
        __m128i q = _mm_sub_epi8(
            _mm_unpacklo_epi8(_mm_and_si128(w2, mask),
                              _mm_and_si128(_mm_srli_epi16(w2, 4), mask)), b8);
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)),
                              _mm512_loadu_ps(x + (size_t)g * 16), acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (int k = groups * 16; k < n; ++k)
        s += x[k] * (float)ct_q4(w, k);
    return s;
}
#elif GEMMA_X86
static inline float ct_dot4_f32(const uint32_t *w, const float *x, int n)
{
    __m256 acc = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        __m128i w2 = _mm_loadl_epi64((const __m128i *)(w + (size_t)g * 2));
        __m128i q = _mm_sub_epi8(
            _mm_unpacklo_epi8(_mm_and_si128(w2, mask),
                              _mm_and_si128(_mm_srli_epi16(w2, 4), mask)), b8);
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),
                              _mm256_loadu_ps(x + (size_t)g * 16), acc);
        acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))),
            _mm256_loadu_ps(x + (size_t)g * 16 + 8), acc);
    }
    float s = hsum256_ps(acc);
    for (int k = groups * 16; k < n; ++k)
        s += x[k] * (float)ct_q4(w, k);
    return s;
}
#else
static inline float ct_dot4_f32(const uint32_t *w, const float *x, int n)
{
    float s = 0.0f;
    for (int k = 0; k < n; ++k)
        s += x[k] * (float)ct_q4(w, k);
    return s;
}
#endif

/* The dot product of one 2-bit row with one float32 row. */
#if GEMMA_X86 && defined(__AVX512F__)
static inline float ct_dot2_f32(const uint32_t *w, const float *x, int n)
{
    __m512 acc = _mm512_setzero_ps();
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        /* One word holds 16 values, and 16 float32 values fill one zmm. */
        __m128i q = ct_sign2(_mm_cvtsi32_si128((int)w[g]));
        acc = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q)),
                              _mm512_loadu_ps(x + (size_t)g * 16), acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (int k = groups * 16; k < n; ++k)
        s += x[k] * (float)ct_q2(w, k);
    return s;
}
#elif GEMMA_X86
static inline float ct_dot2_f32(const uint32_t *w, const float *x, int n)
{
    __m256 acc = _mm256_setzero_ps();
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        __m128i q = ct_sign2(_mm_cvtsi32_si128((int)w[g]));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),
                              _mm256_loadu_ps(x + (size_t)g * 16), acc);
        acc = _mm256_fmadd_ps(
            _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8))),
            _mm256_loadu_ps(x + (size_t)g * 16 + 8), acc);
    }
    float s = hsum256_ps(acc);
    for (int k = groups * 16; k < n; ++k)
        s += x[k] * (float)ct_q2(w, k);
    return s;
}
#else
static inline float ct_dot2_f32(const uint32_t *w, const float *x, int n)
{
    float s = 0.0f;
    for (int k = 0; k < n; ++k)
        s += x[k] * (float)ct_q2(w, k);
    return s;
}
#endif

/* Four 4-bit rows for each x block.
 *
 * The one-row dot above reads the whole x row again for each output row. For a
 * matrix of 10240 rows and 2560 columns that is 105 MB of x traffic against
 * 13 MB of weights, so x controls the time. The x values are the same for every
 * row, so this loop keeps one x block in a register and uses it for four
 * weight rows. The x traffic falls by four times, and the four independent
 * accumulators give the pipeline more work to overlap. This is the same
 * change the int4 kernel of the 12B model uses.
 *
 * out holds the four dot products. The scale is not applied here, because it
 * is different for each row.
 */
#if GEMMA_X86 && defined(__AVX512F__)
static inline void ct_dot4x4_f32(const uint32_t *w, size_t stride,
                                 const float *x, int n, float *out)
{
    __m512 a0 = _mm512_setzero_ps();
    __m512 a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps();
    __m512 a3 = _mm512_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        __m512 xv = _mm512_loadu_ps(x + (size_t)g * 16);
        __m128i w0 = _mm_loadl_epi64((const __m128i *)(w + (size_t)g * 2));
        __m128i w1 = _mm_loadl_epi64((const __m128i *)(w + stride + (size_t)g * 2));
        __m128i w2 = _mm_loadl_epi64((const __m128i *)(w + 2 * stride + (size_t)g * 2));
        __m128i w3 = _mm_loadl_epi64((const __m128i *)(w + 3 * stride + (size_t)g * 2));
        __m128i q0 = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(w0, mask),
                                   _mm_and_si128(_mm_srli_epi16(w0, 4), mask)), b8);
        __m128i q1 = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(w1, mask),
                                   _mm_and_si128(_mm_srli_epi16(w1, 4), mask)), b8);
        __m128i q2 = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(w2, mask),
                                   _mm_and_si128(_mm_srli_epi16(w2, 4), mask)), b8);
        __m128i q3 = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(w3, mask),
                                   _mm_and_si128(_mm_srli_epi16(w3, 4), mask)), b8);
        a0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q0)), xv, a0);
        a1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q1)), xv, a1);
        a2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q2)), xv, a2);
        a3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(q3)), xv, a3);
    }
    out[0] = _mm512_reduce_add_ps(a0);
    out[1] = _mm512_reduce_add_ps(a1);
    out[2] = _mm512_reduce_add_ps(a2);
    out[3] = _mm512_reduce_add_ps(a3);
    int tail = groups * 16;
    for (int k = tail; k < n; ++k) {
        float xk = x[k];
        out[0] += xk * (float)ct_q4(w, k);
        out[1] += xk * (float)ct_q4(w + stride, k);
        out[2] += xk * (float)ct_q4(w + 2 * stride, k);
        out[3] += xk * (float)ct_q4(w + 3 * stride, k);
    }
}
#elif GEMMA_X86
static inline void ct_dot4x4_f32(const uint32_t *w, size_t stride,
                                 const float *x, int n, float *out)
{
    __m256 a0 = _mm256_setzero_ps();
    __m256 a1 = _mm256_setzero_ps();
    __m256 a2 = _mm256_setzero_ps();
    __m256 a3 = _mm256_setzero_ps();
    __m256 a4 = _mm256_setzero_ps();
    __m256 a5 = _mm256_setzero_ps();
    __m256 a6 = _mm256_setzero_ps();
    __m256 a7 = _mm256_setzero_ps();
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    int groups = n / 16;
    for (int g = 0; g < groups; ++g) {
        __m256 xlo = _mm256_loadu_ps(x + (size_t)g * 16);
        __m256 xhi = _mm256_loadu_ps(x + (size_t)g * 16 + 8);
        for (int j = 0; j < 4; ++j) {
            __m128i wj = _mm_loadl_epi64((const __m128i *)(w + (size_t)j * stride + (size_t)g * 2));
            __m128i q = _mm_sub_epi8(_mm_unpacklo_epi8(_mm_and_si128(wj, mask),
                                  _mm_and_si128(_mm_srli_epi16(wj, 4), mask)), b8);
            __m256 lo = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q));
            __m256 hi = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(q, 8)));
            switch (j) {
            case 0: a0 = _mm256_fmadd_ps(lo, xlo, a0); a4 = _mm256_fmadd_ps(hi, xhi, a4); break;
            case 1: a1 = _mm256_fmadd_ps(lo, xlo, a1); a5 = _mm256_fmadd_ps(hi, xhi, a5); break;
            case 2: a2 = _mm256_fmadd_ps(lo, xlo, a2); a6 = _mm256_fmadd_ps(hi, xhi, a6); break;
            default: a3 = _mm256_fmadd_ps(lo, xlo, a3); a7 = _mm256_fmadd_ps(hi, xhi, a7); break;
            }
        }
    }
    out[0] = hsum256_ps(_mm256_add_ps(a0, a4));
    out[1] = hsum256_ps(_mm256_add_ps(a1, a5));
    out[2] = hsum256_ps(_mm256_add_ps(a2, a6));
    out[3] = hsum256_ps(_mm256_add_ps(a3, a7));
    int tail = groups * 16;
    for (int k = tail; k < n; ++k) {
        float xk = x[k];
        out[0] += xk * (float)ct_q4(w, k);
        out[1] += xk * (float)ct_q4(w + stride, k);
        out[2] += xk * (float)ct_q4(w + 2 * stride, k);
        out[3] += xk * (float)ct_q4(w + 3 * stride, k);
    }
}
#else
static inline void ct_dot4x4_f32(const uint32_t *w, size_t stride,
                                 const float *x, int n, float *out)
{
    for (int j = 0; j < 4; ++j)
        out[j] = ct_dot4_f32(w + (size_t)j * stride, x, n);
}
#endif

static int gemma_ct_rows4 = 1;

/* Select the four-row loop (1) or the one-row loop (0). Use this for a test. */
void rc_ct_set_rows4(int on)
{
    gemma_ct_rows4 = on ? 1 : 0;
}

/* Multiply x by one matrix in the packed layout. bits is 4 or 2. */
static void ct_linear_run(const uint32_t *w, const float *scale, const float *x,
                          float *out, int rows, int cols, int tokens, int bits)
{
    /* A 4-bit row uses cols / 8 words and a 2-bit row cols / 16. The caller
     * checks that cols is a multiple of 16, so the vector loops below cover
     * the row and the scalar tail never runs. */
    size_t words = (size_t)(cols / 16) * (bits == 4 ? 2 : 1);
    int groups = gemma_ct_rows4 ? rows / 4 : 0;
    if (bits == 4) {
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < groups; ++b) {
            int r = b * 4;
            for (int t = 0; t < tokens; ++t) {
                float d[4];
                ct_dot4x4_f32(w + (size_t)r * words, words,
                              x + (size_t)t * (size_t)cols, cols, d);
                for (int j = 0; j < 4; ++j)
                    out[(size_t)t * (size_t)rows + r + j] = scale[r + j] * d[j];
            }
        }
        #pragma omp parallel for schedule(static)
        for (int r = groups * 4; r < rows; ++r) {
            const float s = scale[r];
            for (int t = 0; t < tokens; ++t)
                out[(size_t)t * (size_t)rows + r] =
                    s * ct_dot4_f32(w + (size_t)r * words,
                                    x + (size_t)t * (size_t)cols, cols);
        }
        return;
    }
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        const uint32_t *wr = w + (size_t)r * words;
        const float s = scale[r];
        for (int t = 0; t < tokens; ++t) {
            const float *xt = x + (size_t)t * (size_t)cols;
            out[(size_t)t * (size_t)rows + r] = s * ct_dot2_f32(wr, xt, cols);
        }
    }
}

/* Multiply x by W. W is a weight of the compressed-tensors file, packed.
 *
 * w       the int32 words of the weight, one row after another
 * scale   one float32 value for each row
 * x       the activations, tokens rows of cols values
 * out     tokens rows of rows values
 * bits    4 or 2
 *
 * cols must be a multiple of 16 and must equal the value count that the
 * packed words hold. Every matrix of the E4B text model satisfies this.
 */
void rc_ct_linear(const uint32_t *w, const float *scale, const float *x,
                     float *out, int rows, int cols, int tokens, int bits)
{
    if (bits != 4 && bits != 2)
        return;
    ct_linear_run(w, scale, x, out, rows, cols, tokens, bits);
}
