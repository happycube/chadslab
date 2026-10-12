/* ---------- int8 mixture of experts for a decode step ----------
 * The int4 expert kernel holds four rows in its float lanes, so a decode step
 * reads each selected expert with the plain multiply. This kernel holds eight
 * rows in the int32 lanes of vpdpbusd instead. One call quantizes the row of
 * each job and runs every job in one parallel region.
 *
 * w has the shape (experts, rows, groups, 18). ids gives the selected expert
 * of each job. x holds one row for each job, with a stride of xstride. A
 * stride of 0 gives the same row to every job.
 */
void rc_int4_q8_moe_gemv(const uint8_t *w, const float *scales,
                            const float *x, const int32_t *ids, int jobs,
                            float *out, int rows, int cols, int xstride)
{
    const int groups = cols / 32;
    const size_t wstride = (size_t)groups * 18;
    const int nrb = (rows + I4QG_MR - 1) / I4QG_MR;
    /* One slice of the scratch for each job. */
    if (jobs <= 0 || !gemma_i4q_scratch(cols * jobs)) {
        return;
    }
    int8_t *qx = g_i4qx;
    float *sx = g_i4qs;
    int32_t *sumx = g_i4qm;
    const size_t qstride = (size_t)cols;
    const size_t sstride = (size_t)groups;
    for (int j = 0; j < jobs; ++j) {
        const float *xj = x + (xstride != 0 ? (size_t)j * (size_t)xstride : 0);
        for (int g = 0; g < groups; ++g) {
            sx[(size_t)j * sstride + g] = gemma_quant_group32(
                xj + (size_t)g * 32, qx + (size_t)j * qstride + (size_t)g * 32,
                &sumx[(size_t)j * sstride + g]);
        }
    }
    #pragma omp parallel for schedule(static) collapse(2)
    for (int j = 0; j < jobs; ++j) {
        for (int rb = 0; rb < nrb; ++rb) {
            const size_t e = (size_t)ids[j];
            gemma_int4_q8_gemv_block(w + e * (size_t)rows * wstride,
                                     scales + e * (size_t)rows * (size_t)groups,
                                     qx + (size_t)j * qstride,
                                     sx + (size_t)j * sstride,
                                     sumx + (size_t)j * sstride,
                                     out + (size_t)j * (size_t)rows,
                                     rows, cols, rb);
        }
    }
}

/* ---------- int8 mixture of experts for a prompt ----------
 * The model gives each expert a different group of tokens. A call for each
 * expert then starts a small parallel region for each expert. A group of
 * sixteen tokens gives little work for one region, so a machine with many
 * cores gains little.
 *
 * These two kernels take the list of experts. One parallel region covers the
 * full work of all the experts. Thus the thread team is large and the caller
 * starts two regions for the whole layer.
 *
 * Every expert uses its own slice of the scratch and the output. off holds the
 * token offset of each expert and ntok holds its token count. The slices are
 * not padded. A tile may read a few tokens past a slice, so the caller leaves a
 * slack of one token block.
 */

/* Quantize the rows of every expert to the transposed int8 layout. src maps a
 * destination row to a row of x. A null src gives the identity. */
void rc_quantize_q8_t_moe(const float *x, const int32_t *src, int8_t *qxt,
                             float *sx, int32_t *sumx, int cols, int stride,
                             const int32_t *off, const int32_t *ntok, int ne)
{
    const int groups = cols / 32;
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < ne; ++e) {
        for (int t = 0; t < ntok[e]; ++t) {
            const int row = off[e] + t;
            /* src maps a destination row to a row of x. A null src is the
             * identity. Thus the caller needs no separate gather. */
            const float *xt = x + (size_t)(src != NULL ? src[row] : row)
                              * (size_t)cols;
            int8_t q[32];
            for (int g = 0; g < groups; ++g) {
                int32_t s;
                float sc = gemma_quant_group32(xt + (size_t)g * 32, q, &s);
                sx[(size_t)g * (size_t)stride + row] = sc;
                sumx[(size_t)g * (size_t)stride + row] = s;
                for (int q4 = 0; q4 < 8; ++q4) {
                    memcpy(qxt + (((size_t)g * 8 + q4) * (size_t)stride + row) * 4,
                           q + q4 * 4, 4);
                }
            }
        }
    }
}

/* Multiply the selected experts of one layer by their input rows. w holds one
 * matrix for each expert in the Q4_0 block layout. scales holds one float32
 * scale for each group of 32 columns. eid gives the matrix index of a job. */
/* ---------- the experts of a prompt with float32 activations ----------
 *
 * The float path of the experts ran one call for each expert. A layer then
 * made about 220 small calls, and each call paid for its own region. This
 * code runs the float tile of every expert in one region, as
 * rc_int4_q8_moe_run does with the int8 tile. The activations stay
 * float32, so the result has no error from a quantization. */

/* Gather the rows of each expert into one transposed buffer. Row r of the
 * expert scratch is row src[r] of x. A null src is the identity. xt is
 * (cols, stride). The value of column c of scratch row r is at
 * c * stride + r. */
void rc_gather_t_moe(const float *x, const int32_t *src, float *xt, int cols,
                        int stride, int rows_total)
{
    #pragma omp parallel for schedule(static)
    for (int c0 = 0; c0 < cols; c0 += 64) {
        int c1 = c0 + 64 < cols ? c0 + 64 : cols;
        for (int r = 0; r < rows_total; ++r) {
            const float *xr = x + (size_t)(src != NULL ? src[r] : r) * (size_t)cols;
            for (int c = c0; c < c1; ++c) {
                xt[(size_t)c * (size_t)stride + r] = xr[c];
            }
        }
    }
}

#if GEMMA_X86 && defined(__AVX512F__)
/* The float tile of one expert: I4T_MR rows and up to I4T_TB tokens. The
 * tokens of the expert start at column base of xt. The row length of xt is
 * ld. */
static inline void gemma_int4_f32_moe_tile(const uint8_t *w, const float *scales,
                                           const float *xt, float *out, int rows,
                                           int cols, int ntok_e, int ld, int base,
                                           int i0, int t0)
{
    __m512 acc[I4T_MR];
    for (int r = 0; r < I4T_MR; ++r) {
        acc[r] = _mm512_setzero_ps();
    }
    const __m128i mask = _mm_set1_epi8(0x0F);
    const __m128i b8 = _mm_set1_epi8(8);
    const int groups = cols / 32;
    const int wstride = groups * 18;
    int nrow = rows - i0;
    if (nrow > I4T_MR) {
        nrow = I4T_MR;
    }
    int ntok = ntok_e - t0;
    if (ntok > I4T_TB) {
        ntok = I4T_TB;
    }
    const __mmask16 km = ntok >= I4T_TB ? (__mmask16)0xFFFF : (__mmask16)((1u << ntok) - 1);
    for (int g = 0; g < groups; ++g) {
        float wf[I4T_MR][32];
        for (int r = 0; r < nrow; ++r) {
            const uint8_t *p = w + (size_t)(i0 + r) * wstride + (size_t)g * 18 + 2;
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
            __m512 xv = _mm512_maskz_loadu_ps(km, xt + (size_t)(g * 32 + k) * (size_t)ld
                                                  + base + t0);
            for (int r = 0; r < nrow; ++r) {
                acc[r] = _mm512_fmadd_ps(_mm512_set1_ps(wf[r][k]), xv, acc[r]);
            }
        }
    }
    float tmp[I4T_TB];
    for (int r = 0; r < nrow; ++r) {
        _mm512_storeu_ps(tmp, acc[r]);
        for (int t = 0; t < ntok; ++t) {
            out[(size_t)(base + t0 + t) * (size_t)rows + (size_t)i0 + r] = tmp[t];
        }
    }
}
#else
static inline void gemma_int4_f32_moe_tile(const uint8_t *w, const float *scales,
                                           const float *xt, float *out, int rows,
                                           int cols, int ntok_e, int ld, int base,
                                           int i0, int t0)
{
    const int groups = cols / 32;
    const int wstride = groups * 18;
    for (int r = 0; r < I4T_MR && i0 + r < rows; ++r) {
        for (int t = 0; t < I4T_TB && t0 + t < ntok_e; ++t) {
            float acc = 0.0f;
            for (int g = 0; g < groups; ++g) {
                const uint8_t *b = w + (size_t)(i0 + r) * wstride + (size_t)g * 18 + 2;
                const float sc = scales[(size_t)(i0 + r) * groups + g];
                for (int k = 0; k < 32; ++k) {
                    int nib = k < 16 ? (b[k] & 0x0F) : (b[k - 16] >> 4);
                    acc += (float)(nib - 8) * sc
                           * xt[(size_t)(g * 32 + k) * (size_t)ld + base + t0 + t];
                }
            }
            out[(size_t)(base + t0 + t) * (size_t)rows + (size_t)i0 + r] = acc;
        }
    }
}
#endif

/* Run the float tile of every selected expert in one region. The arguments
 * follow rc_int4_q8_moe_run. */
void rc_int4_f32_moe_run(const uint8_t *w, const float *scales, const float *xt,
                            float *out, int rows, int cols, int stride,
                            const int32_t *off, const int32_t *ntok,
                            const int32_t *eid, int ne)
{
    const int groups = cols / 32;
    const size_t expert_bytes = (size_t)rows * (size_t)groups * 18;
    const size_t expert_scales = (size_t)rows * (size_t)groups;
    const int nrb = (rows + I4T_MR - 1) / I4T_MR;
    long *start = (long *)malloc((size_t)(ne + 1) * sizeof(long));
    long total = 0;
    for (int e = 0; e < ne; ++e) {
        start[e] = total;
        total += (long)nrb * (long)((ntok[e] + I4T_TB - 1) / I4T_TB);
    }
    start[ne] = total;
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total; ++t) {
        int lo = 0, hi = ne - 1, e = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (start[mid] <= t) {
                e = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        const int ntb = (ntok[e] + I4T_TB - 1) / I4T_TB;
        const long u = t - start[e];
        gemma_int4_f32_moe_tile(w + (size_t)eid[e] * expert_bytes,
                                scales + (size_t)eid[e] * expert_scales, xt, out, rows,
                                cols, ntok[e], stride, off[e], (int)(u / ntb) * I4T_MR,
                                (int)(u % ntb) * I4T_TB);
    }
    free(start);
}

/* ---------- the experts of a prompt with int16 activations ----------
 *
 * The float tile of the experts is limited by the work of the multiply: it
 * changes each weight to float32 for each token block. The int16 tile keeps
 * the integer multiply of the int8 tile, at half its rate. Its error is about
 * 250 times smaller than the error of int8. These two functions follow
 * rc_quantize_q8_t_moe and rc_int4_q8_moe_run. */

/* Quantize the rows of each expert to int16 in the layout of the tile. Row r
 * of the expert scratch is row src[r] of x (a null src is the identity). */
void rc_quantize_q16_t_moe(const float *x, const int32_t *src, int16_t *qxt,
                              float *sx, int cols, int stride, int rows_total)
{
    const int groups = cols / 32;
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < rows_total; ++r) {
        const float *xt = x + (size_t)(src != NULL ? src[r] : r) * (size_t)cols;
        int16_t q[32];
        for (int g = 0; g < groups; ++g) {
            sx[(size_t)g * (size_t)stride + r] = gemma_quant_group32_i16(xt + (size_t)g * 32, q);
            for (int k2 = 0; k2 < 16; ++k2) {
                int16_t *dst = qxt + (((size_t)g * 16 + k2) * (size_t)stride + r) * 2;
                dst[0] = q[k2 * 2];
                dst[1] = q[k2 * 2 + 1];
            }
        }
    }
}

void rc_int4_q16_moe_run(const uint8_t *w, const float *scales, const int16_t *qxt,
                            const float *sx, float *out, int rows, int cols, int stride,
                            const int32_t *off, const int32_t *ntok, const int32_t *eid,
                            int ne)
{
    const int groups = cols / 32;
    const size_t expert_bytes = (size_t)rows * (size_t)groups * 18;
    const size_t expert_scales = (size_t)rows * (size_t)groups;
    const int nrb = (rows + I4Q_MR - 1) / I4Q_MR;
    long *start = (long *)malloc((size_t)(ne + 1) * sizeof(long));
    long total = 0;
    for (int e = 0; e < ne; ++e) {
        start[e] = total;
        total += (long)nrb * (long)((ntok[e] + I4Q_TB - 1) / I4Q_TB);
    }
    start[ne] = total;
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total; ++t) {
        int lo = 0, hi = ne - 1, e = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (start[mid] <= t) {
                e = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        const int ntb = (ntok[e] + I4Q_TB - 1) / I4Q_TB;
        const long u = t - start[e];
        gemma_int4_q16_tile(w + (size_t)eid[e] * expert_bytes,
                            scales + (size_t)eid[e] * expert_scales, qxt, sx, out, rows,
                            cols, ntok[e], stride, off[e], (int)(u / ntb) * I4Q_MR,
                            (int)(u % ntb) * I4Q_TB);
    }
    free(start);
}

void rc_int4_q8_moe_run(const uint8_t *w, const float *scales,
                           const int8_t *qxt, const float *sx,
                           const int32_t *sumx, float *out,
                           int rows, int cols, int stride,
                           const int32_t *off, const int32_t *ntok,
                           const int32_t *eid, int ne)
{
    const int groups = cols / 32;
    const size_t wstride = (size_t)groups * 18;
    const size_t expert_bytes = (size_t)rows * wstride;
    const size_t expert_scales = (size_t)rows * (size_t)groups;
    const int nrb = (rows + I4Q_MR - 1) / I4Q_MR;
    const int tb = gemma_i4q_tb8 ? I4Q2_TB : I4Q_TB;
    /* Count the tasks of each expert. A task is one row block and one token
     * block. The task list spans every expert. Thus one parallel region covers
     * the whole layer and the thread team stays large. Parallel over the
     * experts alone is not enough, because an expert is a long serial chain. */
    long *start = (long *)malloc((size_t)(ne + 1) * sizeof(long));
    long total = 0;
    for (int e = 0; e < ne; ++e) {
        start[e] = total;
        total += (long)nrb * (long)((ntok[e] + tb - 1) / tb);
    }
    start[ne] = total;
    #pragma omp parallel for schedule(static)
    for (long t = 0; t < total; ++t) {
        int lo = 0;
        int hi = ne - 1;
        int e = 0;
        while (lo <= hi) {
            int mid = (lo + hi) >> 1;
            if (start[mid] <= t) {
                e = mid;
                lo = mid + 1;
            } else {
                hi = mid - 1;
            }
        }
        const int ntb = (ntok[e] + tb - 1) / tb;
        const long u = t - start[e];
        const int rb = (int)(u / ntb);
        const int t0 = (int)(u % ntb) * tb;
        const uint8_t *we = w + (size_t)eid[e] * expert_bytes;
        const float *se = scales + (size_t)eid[e] * expert_scales;
#if GEMMA_X86
        if (gemma_i4q_tb8) {
            gemma_int4_q8_tile_narrow(we, se, qxt, sx, sumx, out, rows, cols,
                                      ntok[e], stride, off[e], rb * I4Q2_MR, t0);
            continue;
        }
#endif
        gemma_int4_q8_tile(we, se, qxt, sx, sumx, out, rows, cols,
                           ntok[e], stride, off[e], rb * I4Q_MR, t0);
    }
    free(start);
}


/* Apply the GELU to the gate half of x and multiply by the up half. x has two
 * inner values in each row: the gate first, then the up. out has one inner
 * value in each row. One pass avoids the temporaries of the NumPy path. */
static void gemma_gelu_mul_body(const float *x, float *out, int rows, int inner)
{
    const float c = 0.7978845608028654f;
#if GEMMA_X86 && defined(__AVX512F__)
    const __m512 cv = _mm512_set1_ps(c);
    const __m512 half = _mm512_set1_ps(0.5f);
    const __m512 one = _mm512_set1_ps(1.0f);
    const __m512 k3 = _mm512_set1_ps(0.044715f);
    const int nv = inner & ~15;
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const float *g = x + (size_t)i * 2 * (size_t)inner;
        const float *u = g + inner;
        float *o = out + (size_t)i * (size_t)inner;
        for (int j = 0; j < nv; j += 16) {
            const __m512 v = _mm512_loadu_ps(g + j);
            const __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
            const __m512 t = _mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v));
            const __m512 r = _mm512_mul_ps(_mm512_mul_ps(half, v),
                                           _mm512_add_ps(one, gemma_tanh_ps(t)));
            _mm512_storeu_ps(o + j, _mm512_mul_ps(r, _mm512_loadu_ps(u + j)));
        }
        for (int j = nv; j < inner; ++j) {
            const float v = g[j];
            o[j] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))) * u[j];
        }
    }
#else
    #pragma omp for schedule(static)
    for (int i = 0; i < rows; ++i) {
        const float *g = x + (size_t)i * 2 * (size_t)inner;
        const float *u = g + inner;
        float *o = out + (size_t)i * (size_t)inner;
        for (int j = GELU_MUL_VEC(g, u, o, inner); j < inner; ++j) {
            float v = g[j];
            o[j] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v))) * u[j];
        }
    }
#endif
}

void rc_gelu_mul(const float *x, float *out, int rows, int inner)
{
    #pragma omp parallel if(rows >= 8)
    gemma_gelu_mul_body(x, out, rows, inner);
}

/* Add the weighted expert output to the rows of out. de holds one row for each
 * (expert, token) pair. rows[j] gives the token of row j of de and w[j] gives
 * the router weight. A token may appear in several experts, so the code
 * parallelizes over the hidden axis. Then two jobs never write the same
 * element. */
void rc_moe_scatter(float *out, const float *de, const int32_t *rows,
                       const float *w, int n, int hidden)
{
    const int db = 64;
    const int nblk = (hidden + db - 1) / db;
    #pragma omp parallel for schedule(static)
    for (int blk = 0; blk < nblk; ++blk) {
        const int d0 = blk * db;
        int dn = hidden - d0;
        if (dn > db) {
            dn = db;
        }
        for (int j = 0; j < n; ++j) {
            const float *src = de + (size_t)j * (size_t)hidden + d0;
            float *dst = out + (size_t)rows[j] * (size_t)hidden + d0;
            const float wj = w[j];
            for (int d = 0; d < dn; ++d) {
                dst[d] += src[d] * wj;
            }
        }
    }
}
