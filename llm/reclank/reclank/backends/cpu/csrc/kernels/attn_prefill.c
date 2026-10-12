/* ---------- flash attention for the prompt ----------
 *
 * The plain path builds the whole score matrix for one chunk: (heads, tokens,
 * keys). It masks the matrix, normalizes it, and multiplies by V. At a long
 * context that matrix is large, so the path moves a lot of memory, and it
 * computes the scores that the causal mask hides and, for a sliding layer, the
 * scores that the window hides.
 *
 * These kernels keep the scores of one block at a time and they read only the
 * keys that the block can see. The sliding window makes the point: 25 of the
 * 30 layers of this model slide with a window of 1024, so at a long context
 * they read the window and not the whole context.
 *
 * The key is held transposed, so the score of a row over a block of keys is a
 * vector over the keys. That removes the horizontal sum. The value stays in
 * the natural layout, so the weighted sum is a vector over the head dimension.
 *
 * The online softmax holds a running maximum m, a running sum l, and the
 * running weighted sum acc. A new maximum rescales acc and l by exp(m - mnew).
 *
 * There are three versions. The AVX-512 version, the AVX2 version, and a
 * straight C version. The straight C version is the reference and the fallback
 * for a target with no AVX2. Set the version with
 * rc_attn_prefill_set_impl, or leave the default of 0 to take the best
 * version that the build gives.
 */

/* Transpose the key: kt[h][d * ld + j] = k[(lo_key + j) * kv_heads + h][d].
 * The block over d keeps one group of cache lines hot while the loop walks the
 * keys. */
static void gemma_attn_transpose_k(const float *k, float *kt, int nvis, int ld,
                                   int kv_heads, int hd, int lo_key)
{
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < kv_heads; ++h) {
        float *dsth = kt + (size_t)h * (size_t)hd * (size_t)ld;
        for (int jb = 0; jb < nvis; jb += 16) {
            const int je = jb + 16 < nvis ? jb + 16 : nvis;
            for (int db = 0; db < hd; db += 32) {
                const int de = db + 32 < hd ? db + 32 : hd;
                for (int j = jb; j < je; ++j) {
                    const float *src = k + ((size_t)(lo_key + j) * kv_heads + h) * (size_t)hd;
                    for (int d = db; d < de; ++d) {
                        dsth[(size_t)d * ld + j] = src[d];
                    }
                }
            }
        }
    }
}

/* ---------- the straight C version ---------- */

/* Walk one query row at a time. Sum over the head dimension with a plain loop.
 * This is the reference for the two vector versions. */
void rc_attn_prefill_scalar(const float *q, const float *k, const float *v,
                               const int32_t *positions, int base, int window,
                               float *out, int t, int n, int q_heads,
                               int kv_heads, int hd)
{
    if (t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    const long rows = (long)t * n_rep;
    #pragma omp parallel for schedule(static)
    for (int h = 0; h < kv_heads; ++h) {
        for (long row = 0; row < rows; ++row) {
            const int tok = (int)(row / n_rep);
            const int g = (int)(row - (long)tok * n_rep);
            const int pos = positions[tok];
            const float *qr = q + ((size_t)tok * q_heads + h * n_rep + g) * (size_t)hd;
            float *op = out + ((size_t)tok * q_heads + h * n_rep + g) * (size_t)hd;
            int jlo = 0;
            if (window > 0) {
                jlo = pos - window + 1 - base;
                if (jlo < 0) {
                    jlo = 0;
                }
            }
            int jhi = pos - base + 1;
            if (jhi > n) {
                jhi = n;
            }
            for (int d = 0; d < hd; ++d) {
                op[d] = 0.0f;
            }
            float m = -INFINITY;
            float l = 0.0f;
            for (int j = jlo; j < jhi; ++j) {
                const float *kr = k + ((size_t)j * kv_heads + h) * (size_t)hd;
                float s = 0.0f;
                for (int d = 0; d < hd; ++d) {
                    s += qr[d] * kr[d];
                }
                const float mn = m > s ? m : s;
                float alpha = 0.0f;
                if (m == -INFINITY && mn == -INFINITY) {
                    alpha = 1.0f;
                } else if (m != -INFINITY && mn != -INFINITY) {
                    alpha = expf(m - mn);
                }
                const float p = expf(s - mn);
                l = l * alpha + p;
                const float *vr = v + ((size_t)j * kv_heads + h) * (size_t)hd;
                for (int d = 0; d < hd; ++d) {
                    op[d] = op[d] * alpha + p * vr[d];
                }
                m = mn;
            }
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            for (int d = 0; d < hd; ++d) {
                op[d] *= inv;
            }
        }
    }
}

/* ---------- the AVX-512 version ---------- */

#if GEMMA_X86 && defined(__AVX512F__)

/* exp(x) for sixteen floats. The code splits x into an integer part and a
 * fraction in [-0.5, 0.5]. A degree six polynomial gives 2^frac and the scale
 * instruction applies 2^int. The argument is a score less the running maximum,
 * so it is at most zero, and the clamp stops a masked score of -inf. */
static inline __m512 gemma_attn_exp_avx512(__m512 x)
{
    x = _mm512_max_ps(x, _mm512_set1_ps(-87.0f));
    const __m512 t = _mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f));
    const __m512 n = _mm512_roundscale_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m512 f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_set1_ps(0.000154035303933816f);
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.00133335581464284f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.00961812910762848f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.05550410866482158f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.24022650695910071f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(0.69314718055994529f));
    p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, n);
}

/* The key block, the row block, and the row micro tile. Sixteen keys fill one
 * vector, so two vectors hold a block of 32 keys. A micro tile of eight rows
 * then needs sixteen score vectors and two key vectors. Eight rows for one key
 * vector is the point: the transposed key block stays in the cache and the
 * kernel reads it eight times fewer than a four row tile. */
#define ATTN5_NC 32
#define ATTN5_NB 2
#define ATTN5_MR 8
#define ATTN5_MC 64

/* The tasks of rc_attn_prefill_avx512: the threads of the enclosing OpenMP region split
 * them (an orphaned omp for). kt is the transposed keys. */
/* limit (null, or t int32): the last key position of each query in place of
 * its position (the tokens of one image see each other, Model._media_limit);
 * the window still counts from the position. limit does not decrease over
 * the tokens. */
static void gemma_attn_prefill_avx512_tasks(const float *q, const float *v, const float *kt, const int32_t *positions, int base, int window, float *out, int n, int q_heads, int kv_heads, int hd, int n_rep, int lo_key, int ld, long rows, long last, int nrb, int tasks, const int32_t *limit)
{
    float *acc = (float *)malloc((size_t)ATTN5_MC * hd * sizeof(float));
    float *sp = (float *)malloc((size_t)ATTN5_MC * ATTN5_NC * sizeof(float));
    if (acc != NULL && sp != NULL) {
        const __m512i lane = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7,
                                               8, 9, 10, 11, 12, 13, 14, 15);
        const __m512 ninf = _mm512_set1_ps(-INFINITY);
        #pragma omp for schedule(static)
        for (int task = 0; task < tasks; ++task) {
            const int h = task / nrb;
            const long r0 = (long)(task % nrb) * ATTN5_MC;
            const int tok0 = (int)(r0 / n_rep);
            long t1r = r0 + ATTN5_MC - 1;
            if (t1r > last) {
                t1r = last;
            }
            const int tok1 = (int)(t1r / n_rep);
            const int pos_max = limit != NULL ? limit[tok1] : positions[tok1];
            int jhi = pos_max - base + 1;
            if (jhi > n) {
                jhi = n;
            }
            int jlo = 0;
            if (window > 0) {
                jlo = positions[tok0] - window + 1 - base;
                if (jlo < 0) {
                    jlo = 0;
                }
            }
            if (jlo < lo_key) {
                jlo = lo_key;
            }
            float mrow[ATTN5_MC];
            float lrow[ATTN5_MC];
            for (int i = 0; i < ATTN5_MC; ++i) {
                float *aci = acc + (size_t)i * hd;
                for (int d = 0; d < hd; ++d) {
                    aci[d] = 0.0f;
                }
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            const float *kth = kt + (size_t)h * hd * ld;
            for (int j0 = jlo; j0 < jhi; j0 += ATTN5_NC) {
                const int jn = jhi - j0 < ATTN5_NC ? jhi - j0 : ATTN5_NC;
                /* The score of ATTN5_MR rows over ATTN5_NC keys. The key
                 * vector is read one time for the whole micro tile. */
                for (int i0 = 0; i0 < ATTN5_MC; i0 += ATTN5_MR) {
                    __m512 sv[ATTN5_MR][ATTN5_NB];
                    for (int i = 0; i < ATTN5_MR; ++i) {
                        for (int b = 0; b < ATTN5_NB; ++b) {
                            sv[i][b] = _mm512_setzero_ps();
                        }
                    }
                    /* The row pointers stay fixed for the whole key
                     * block, so the inner loop needs no index product. */
                    const float *qrp[ATTN5_MR];
                    for (int i = 0; i < ATTN5_MR; ++i) {
                        long row = r0 + i0 + i;
                        if (row > last) {
                            row = last;
                        }
                        const int tok = (int)(row / n_rep);
                        const int g = (int)(row - (long)tok * n_rep);
                        qrp[i] = q + ((size_t)tok * q_heads + h * n_rep + g) * hd;
                    }
                    for (int d = 0; d < hd; ++d) {
                        const float *kd = kth + (size_t)d * ld + (j0 - lo_key);
                        __m512 kv[ATTN5_NB];
                        for (int b = 0; b < ATTN5_NB; ++b) {
                            kv[b] = _mm512_loadu_ps(kd + b * 16);
                        }
                        for (int i = 0; i < ATTN5_MR; ++i) {
                            const __m512 qb = _mm512_set1_ps(qrp[i][d]);
                            for (int b = 0; b < ATTN5_NB; ++b) {
                                sv[i][b] = _mm512_fmadd_ps(qb, kv[b], sv[i][b]);
                            }
                        }
                    }
                    for (int i = 0; i < ATTN5_MR; ++i) {
                        long row = r0 + i0 + i;
                        if (row > last) {
                            row = last;
                        }
                        const int tok = (int)(row / n_rep);
                        const int pos = positions[tok];
                        const int lpos = limit != NULL ? limit[tok] : pos;
                        float *srow = sp + (size_t)(i0 + i) * ATTN5_NC;
                        for (int b = 0; b < ATTN5_NB; ++b) {
                            const int jb = j0 + b * 16;
                            __mmask16 keep = 0;
                            if (jb < jhi) {
                                const __m512i kp = _mm512_add_epi32(
                                    _mm512_set1_epi32(base + jb), lane);
                                keep = _mm512_cmp_epi32_mask(kp,
                                    _mm512_set1_epi32(lpos), _MM_CMPINT_LE);
                                if (window > 0) {
                                    const __m512i dist = _mm512_sub_epi32(
                                        _mm512_set1_epi32(pos), kp);
                                    keep = _mm512_kand(keep, _mm512_cmp_epi32_mask(
                                        dist, _mm512_set1_epi32(window), _MM_CMPINT_LT));
                                }
                                const int lim = jhi - jb;
                                if (lim < 16) {
                                    keep = _mm512_kand(keep,
                                        (__mmask16)((1u << lim) - 1u));
                                }
                            }
                            _mm512_storeu_ps(srow + b * 16,
                                _mm512_mask_blend_ps(keep, ninf, sv[i][b]));
                        }
                    }
                }
                /* The online softmax and the weighted sum. */
                for (int i = 0; i < ATTN5_MC; ++i) {
                    float *srow = sp + (size_t)i * ATTN5_NC;
                    const int nv = (jn + 15) / 16;
                    __m512 mx = ninf;
                    for (int b = 0; b < nv; ++b) {
                        mx = _mm512_max_ps(mx, _mm512_loadu_ps(srow + b * 16));
                    }
                    const float smax = _mm512_reduce_max_ps(mx);
                    const float mnew = mrow[i] > smax ? mrow[i] : smax;
                    float alpha = 0.0f;
                    if (mrow[i] == -INFINITY && mnew == -INFINITY) {
                        alpha = 1.0f;
                    } else if (mrow[i] != -INFINITY && mnew != -INFINITY) {
                        alpha = expf(mrow[i] - mnew);
                    }
                    const __m512 mv = _mm512_set1_ps(mnew);
                    const __m512 av = _mm512_set1_ps(alpha);
                    __m512 lsum = _mm512_setzero_ps();
                    for (int b = 0; b < nv; ++b) {
                        const __m512 sv2 = _mm512_loadu_ps(srow + b * 16);
                        __m512 pv = gemma_attn_exp_avx512(_mm512_sub_ps(sv2, mv));
                        pv = _mm512_maskz_mov_ps(
                            _mm512_cmp_ps_mask(sv2, ninf, _CMP_GT_OQ), pv);
                        _mm512_storeu_ps(srow + b * 16, pv);
                        lsum = _mm512_add_ps(lsum, pv);
                    }
                    lrow[i] = lrow[i] * alpha + _mm512_reduce_add_ps(lsum);
                    mrow[i] = mnew;
                    float *aci = acc + (size_t)i * hd;
                    for (int d0 = 0; d0 < hd; d0 += 64) {
                        const int nb = (hd - d0) >= 64 ? 4 : (hd - d0) / 16;
                        __m512 a[4];
                        for (int b = 0; b < nb; ++b) {
                            a[b] = _mm512_mul_ps(_mm512_loadu_ps(aci + d0 + b * 16), av);
                        }
                        for (int j = 0; j < jn; ++j) {
                            const __m512 pv = _mm512_set1_ps(srow[j]);
                            const float *vr = v +
                                ((size_t)(j0 + j) * kv_heads + h) * hd + d0;
                            for (int b = 0; b < nb; ++b) {
                                a[b] = _mm512_fmadd_ps(pv,
                                    _mm512_loadu_ps(vr + b * 16), a[b]);
                            }
                        }
                        for (int b = 0; b < nb; ++b) {
                            _mm512_storeu_ps(aci + d0 + b * 16, a[b]);
                        }
                    }
                }
            }
            for (int i = 0; i < ATTN5_MC; ++i) {
                const long row = r0 + i;
                if (row >= rows) {
                    break;
                }
                const int tok = (int)(row / n_rep);
                const int g = (int)(row - (long)tok * n_rep);
                const float inv = lrow[i] > 0.0f ? 1.0f / lrow[i] : 0.0f;
                const float *aci = acc + (size_t)i * hd;
                float *op = out + ((size_t)tok * q_heads + h * n_rep + g) * hd;
                for (int d = 0; d < hd; ++d) {
                    op[d] = aci[d] * inv;
                }
            }
        }
    }
    free(acc);
    free(sp);
}

void rc_attn_prefill_avx512(const float *q, const float *k, const float *v,
                               const int32_t *positions, int base, int window,
                               float *out, int t, int n, int q_heads,
                               int kv_heads, int hd)
{
    if (t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads || (hd % 16) != 0) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    int lo_key = 0;
    if (window > 0) {
        lo_key = positions[0] - window + 1 - base;
        if (lo_key < 0) {
            lo_key = 0;
        }
    }
    const int nvis = n - lo_key;
    if (nvis <= 0) {
        memset(out, 0, (size_t)t * q_heads * hd * sizeof(float));
        return;
    }
    const int ld = ((nvis + 15) / 16) * 16;
    float *kt = (float *)malloc((size_t)kv_heads * hd * ld * sizeof(float));
    if (kt == NULL) {
        return;
    }
    gemma_attn_transpose_k(k, kt, nvis, ld, kv_heads, hd, lo_key);

    const long rows = (long)t * n_rep;
    const long last = rows - 1;
    const int nrb = (int)((rows + ATTN5_MC - 1) / ATTN5_MC);
    const int tasks = kv_heads * nrb;
    #pragma omp parallel
    gemma_attn_prefill_avx512_tasks(q, v, kt, positions, base, window, out, n, q_heads, kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb, tasks, NULL);
    free(kt);
}

#endif

/* ---------- the AVX2 version ---------- */

#if GEMMA_X86

/* exp(x) for eight floats. The same split as the AVX-512 version. The scaling
 * by 2^int is a shift of the exponent field. */
static inline __m256 gemma_attn_exp_avx2(__m256 x)
{
    x = _mm256_max_ps(x, _mm256_set1_ps(-87.0f));
    const __m256 t = _mm256_mul_ps(x, _mm256_set1_ps(1.44269504088896341f));
    const __m256 n = _mm256_round_ps(t, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    const __m256 f = _mm256_sub_ps(t, n);
    __m256 p = _mm256_set1_ps(0.000154035303933816f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.00133335581464284f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.00961812910762848f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.05550410866482158f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.24022650695910071f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.69314718055994529f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));
    const __m256i ni = _mm256_cvtps_epi32(n);
    const __m256i e = _mm256_slli_epi32(
        _mm256_add_epi32(ni, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

static inline float gemma_attn_hmax8(__m256 x)
{
    __m128 lo = _mm256_castps256_ps128(x);
    const __m128 hi = _mm256_extractf128_ps(x, 1);
    lo = _mm_max_ps(lo, hi);
    lo = _mm_max_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_max_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

static inline float gemma_attn_hsum8(__m256 x)
{
    __m128 lo = _mm256_castps256_ps128(x);
    const __m128 hi = _mm256_extractf128_ps(x, 1);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo));
    lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}

/* Eight keys fill one vector, so two vectors hold a block of 16 keys. A micro
 * tile of four rows needs eight score vectors and two key vectors. */
#define ATTN2_NC 16
#define ATTN2_NB 2
#define ATTN2_MR 4
#define ATTN2_MC 32

/* The tasks of rc_attn_prefill_avx2: the threads of the enclosing OpenMP region split
 * them (an orphaned omp for). kt is the transposed keys. */
static void gemma_attn_prefill_avx2_tasks(const float *q, const float *v, const float *kt, const int32_t *positions, int base, int window, float *out, int n, int q_heads, int kv_heads, int hd, int n_rep, int lo_key, int ld, long rows, long last, int nrb, int tasks, const int32_t *limit)
{
    float *acc = (float *)malloc((size_t)ATTN2_MC * hd * sizeof(float));
    float *sp = (float *)malloc((size_t)ATTN2_MC * ATTN2_NC * sizeof(float));
    if (acc != NULL && sp != NULL) {
        const __m256i lane = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        const __m256 ninf = _mm256_set1_ps(-INFINITY);
        #pragma omp for schedule(static)
        for (int task = 0; task < tasks; ++task) {
            const int h = task / nrb;
            const long r0 = (long)(task % nrb) * ATTN2_MC;
            const int tok0 = (int)(r0 / n_rep);
            long t1r = r0 + ATTN2_MC - 1;
            if (t1r > last) {
                t1r = last;
            }
            const int tok1 = (int)(t1r / n_rep);
            const int pos_max = limit != NULL ? limit[tok1] : positions[tok1];
            int jhi = pos_max - base + 1;
            if (jhi > n) {
                jhi = n;
            }
            int jlo = 0;
            if (window > 0) {
                jlo = positions[tok0] - window + 1 - base;
                if (jlo < 0) {
                    jlo = 0;
                }
            }
            if (jlo < lo_key) {
                jlo = lo_key;
            }
            float mrow[ATTN2_MC];
            float lrow[ATTN2_MC];
            for (int i = 0; i < ATTN2_MC; ++i) {
                float *aci = acc + (size_t)i * hd;
                for (int d = 0; d < hd; ++d) {
                    aci[d] = 0.0f;
                }
                mrow[i] = -INFINITY;
                lrow[i] = 0.0f;
            }
            const float *kth = kt + (size_t)h * hd * ld;
            for (int j0 = jlo; j0 < jhi; j0 += ATTN2_NC) {
                const int jn = jhi - j0 < ATTN2_NC ? jhi - j0 : ATTN2_NC;
                for (int i0 = 0; i0 < ATTN2_MC; i0 += ATTN2_MR) {
                    __m256 sv[ATTN2_MR][ATTN2_NB];
                    for (int i = 0; i < ATTN2_MR; ++i) {
                        for (int b = 0; b < ATTN2_NB; ++b) {
                            sv[i][b] = _mm256_setzero_ps();
                        }
                    }
                    const float *qrp[ATTN2_MR];
                    for (int i = 0; i < ATTN2_MR; ++i) {
                        long row = r0 + i0 + i;
                        if (row > last) {
                            row = last;
                        }
                        const int tok = (int)(row / n_rep);
                        const int g = (int)(row - (long)tok * n_rep);
                        qrp[i] = q + ((size_t)tok * q_heads + h * n_rep + g) * hd;
                    }
                    for (int d = 0; d < hd; ++d) {
                        const float *kd = kth + (size_t)d * ld + (j0 - lo_key);
                        __m256 kv[ATTN2_NB];
                        for (int b = 0; b < ATTN2_NB; ++b) {
                            kv[b] = _mm256_loadu_ps(kd + b * 8);
                        }
                        for (int i = 0; i < ATTN2_MR; ++i) {
                            const __m256 qb = _mm256_set1_ps(qrp[i][d]);
                            for (int b = 0; b < ATTN2_NB; ++b) {
                                sv[i][b] = _mm256_fmadd_ps(qb, kv[b], sv[i][b]);
                            }
                        }
                    }
                    for (int i = 0; i < ATTN2_MR; ++i) {
                        long row = r0 + i0 + i;
                        if (row > last) {
                            row = last;
                        }
                        const int tok = (int)(row / n_rep);
                        const int pos = positions[tok];
                        const int lpos = limit != NULL ? limit[tok] : pos;
                        float *srow = sp + (size_t)(i0 + i) * ATTN2_NC;
                        for (int b = 0; b < ATTN2_NB; ++b) {
                            const int jb = j0 + b * 8;
                            __m256i keep = _mm256_setzero_si256();
                            if (jb < jhi) {
                                const __m256i kp = _mm256_add_epi32(
                                    _mm256_set1_epi32(base + jb), lane);
                                const __m256i pq = _mm256_set1_epi32(pos);
                                keep = _mm256_cmpgt_epi32(
                                    _mm256_set1_epi32(lpos + 1), kp);
                                if (window > 0) {
                                    const __m256i dist = _mm256_sub_epi32(pq, kp);
                                    keep = _mm256_and_si256(keep, _mm256_cmpgt_epi32(
                                        _mm256_set1_epi32(window), dist));
                                }
                                const int lim = jhi - jb;
                                if (lim < 8) {
                                    keep = _mm256_and_si256(keep,
                                        _mm256_cmpgt_epi32(_mm256_set1_epi32(lim), lane));
                                }
                            }
                            _mm256_storeu_ps(srow + b * 8,
                                _mm256_blendv_ps(ninf, sv[i][b],
                                    _mm256_castsi256_ps(keep)));
                        }
                    }
                }
                /* The softmax of each row, then the values for two rows at a
                 * time: a load of 4 value vectors serves both rows. Each row
                 * keeps its operations and their order. */
                float alpha_row[ATTN2_MC];
                for (int i = 0; i < ATTN2_MC; ++i) {
                    float *srow = sp + (size_t)i * ATTN2_NC;
                    const int nv = (jn + 7) / 8;
                    __m256 mx = ninf;
                    for (int b = 0; b < nv; ++b) {
                        mx = _mm256_max_ps(mx, _mm256_loadu_ps(srow + b * 8));
                    }
                    const float smax = gemma_attn_hmax8(mx);
                    const float mnew = mrow[i] > smax ? mrow[i] : smax;
                    float alpha = 0.0f;
                    if (mrow[i] == -INFINITY && mnew == -INFINITY) {
                        alpha = 1.0f;
                    } else if (mrow[i] != -INFINITY && mnew != -INFINITY) {
                        alpha = expf(mrow[i] - mnew);
                    }
                    const __m256 mv = _mm256_set1_ps(mnew);
                    __m256 lsum = _mm256_setzero_ps();
                    for (int b = 0; b < nv; ++b) {
                        const __m256 sv2 = _mm256_loadu_ps(srow + b * 8);
                        __m256 pv = gemma_attn_exp_avx2(_mm256_sub_ps(sv2, mv));
                        pv = _mm256_and_ps(pv, _mm256_cmp_ps(sv2, ninf, _CMP_GT_OQ));
                        _mm256_storeu_ps(srow + b * 8, pv);
                        lsum = _mm256_add_ps(lsum, pv);
                    }
                    lrow[i] = lrow[i] * alpha + gemma_attn_hsum8(lsum);
                    mrow[i] = mnew;
                    alpha_row[i] = alpha;
                }
                for (int i = 0; i < ATTN2_MC; i += 2) {
                    const float *s0 = sp + (size_t)i * ATTN2_NC;
                    const float *s1 = s0 + ATTN2_NC;
                    float *ac0 = acc + (size_t)i * hd;
                    float *ac1 = ac0 + hd;
                    const __m256 av0 = _mm256_set1_ps(alpha_row[i]);
                    const __m256 av1 = _mm256_set1_ps(alpha_row[i + 1]);
                    for (int d0 = 0; d0 < hd; d0 += 32) {
                        const int nb = (hd - d0) >= 32 ? 4 : (hd - d0) / 8;
                        __m256 a0[4], a1[4];
                        for (int b = 0; b < nb; ++b) {
                            a0[b] = _mm256_mul_ps(_mm256_loadu_ps(ac0 + d0 + b * 8), av0);
                            a1[b] = _mm256_mul_ps(_mm256_loadu_ps(ac1 + d0 + b * 8), av1);
                        }
                        for (int j = 0; j < jn; ++j) {
                            const __m256 p0 = _mm256_set1_ps(s0[j]);
                            const __m256 p1 = _mm256_set1_ps(s1[j]);
                            const float *vr = v +
                                ((size_t)(j0 + j) * kv_heads + h) * hd + d0;
                            for (int b = 0; b < nb; ++b) {
                                const __m256 vv = _mm256_loadu_ps(vr + b * 8);
                                a0[b] = _mm256_fmadd_ps(p0, vv, a0[b]);
                                a1[b] = _mm256_fmadd_ps(p1, vv, a1[b]);
                            }
                        }
                        for (int b = 0; b < nb; ++b) {
                            _mm256_storeu_ps(ac0 + d0 + b * 8, a0[b]);
                            _mm256_storeu_ps(ac1 + d0 + b * 8, a1[b]);
                        }
                    }
                }
            }
            for (int i = 0; i < ATTN2_MC; ++i) {
                const long row = r0 + i;
                if (row >= rows) {
                    break;
                }
                const int tok = (int)(row / n_rep);
                const int g = (int)(row - (long)tok * n_rep);
                const float inv = lrow[i] > 0.0f ? 1.0f / lrow[i] : 0.0f;
                const float *aci = acc + (size_t)i * hd;
                float *op = out + ((size_t)tok * q_heads + h * n_rep + g) * hd;
                for (int d = 0; d < hd; ++d) {
                    op[d] = aci[d] * inv;
                }
            }
        }
    }
    free(acc);
    free(sp);
}

void rc_attn_prefill_avx2(const float *q, const float *k, const float *v,
                             const int32_t *positions, int base, int window,
                             float *out, int t, int n, int q_heads,
                             int kv_heads, int hd)
{
    if (t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads || (hd % 8) != 0) {
        return;
    }
    const int n_rep = q_heads / kv_heads;
    int lo_key = 0;
    if (window > 0) {
        lo_key = positions[0] - window + 1 - base;
        if (lo_key < 0) {
            lo_key = 0;
        }
    }
    const int nvis = n - lo_key;
    if (nvis <= 0) {
        memset(out, 0, (size_t)t * q_heads * hd * sizeof(float));
        return;
    }
    const int ld = ((nvis + 7) / 8) * 8;
    float *kt = (float *)malloc((size_t)kv_heads * hd * ld * sizeof(float));
    if (kt == NULL) {
        return;
    }
    gemma_attn_transpose_k(k, kt, nvis, ld, kv_heads, hd, lo_key);

    const long rows = (long)t * n_rep;
    const long last = rows - 1;
    const int nrb = (int)((rows + ATTN2_MC - 1) / ATTN2_MC);
    const int tasks = kv_heads * nrb;
    #pragma omp parallel
    gemma_attn_prefill_avx2_tasks(q, v, kt, positions, base, window, out, n, q_heads, kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb, tasks, NULL);
    free(kt);
}

#endif

/* ---------- the dispatch ---------- */

static int gemma_attn_impl = 0;

/* Select the implementation. 0 takes the best one that the build gives, 1 the
 * straight C version, 2 the AVX2 version, and 3 the AVX-512 version. A version
 * that the build does not have falls back to the best one. A test uses this. */
void rc_attn_prefill_set_impl(int impl)
{
    gemma_attn_impl = impl;
}

void rc_attn_prefill(const float *q, const float *k, const float *v,
                        const int32_t *positions, int base, int window,
                        float *out, int t, int n, int q_heads, int kv_heads, int hd)
{
    if (gemma_attn_impl == 1) {
        rc_attn_prefill_scalar(q, k, v, positions, base, window, out,
                                  t, n, q_heads, kv_heads, hd);
        return;
    }
#if GEMMA_X86 && defined(__AVX512F__)
    if (gemma_attn_impl == 2) {
        rc_attn_prefill_avx2(q, k, v, positions, base, window, out,
                                t, n, q_heads, kv_heads, hd);
        return;
    }
    rc_attn_prefill_avx512(q, k, v, positions, base, window, out,
                              t, n, q_heads, kv_heads, hd);
#elif GEMMA_X86
    rc_attn_prefill_avx2(q, k, v, positions, base, window, out,
                            t, n, q_heads, kv_heads, hd);
#else
    rc_attn_prefill_scalar(q, k, v, positions, base, window, out,
                              t, n, q_heads, kv_heads, hd);
#endif
}

/* rc_attn_prefill inside an OpenMP region (a program record): every thread
 * of the region calls it. One thread makes the transposed keys (kt), and the
 * threads split the tasks of the AVX-512 or AVX2 version (an orphaned omp
 * for). The scalar version runs on one thread. */
static void gemma_attn_prefill_region_lim(const float *q, const float *k, const float *v,
                                          const int32_t *positions, int base, int window,
                                          float *out, int t, int n, int q_heads, int kv_heads,
                                          int hd, const int32_t *limit)
{
#if GEMMA_X86
    int wide = 0;           /* 16 floats (AVX-512) or 8 (AVX2) */
#if defined(__AVX512F__)
    /* a head of 8k values that is not 16k (72 in the 26B vision) takes AVX2 */
    wide = (gemma_attn_impl == 2 || hd % 16 != 0) ? 8 : 16;
#else
    wide = 8;
#endif
    if (gemma_attn_impl == 1 || t <= 0 || n <= 0 || hd <= 0 || q_heads < kv_heads ||
        (hd % wide) != 0) {
        #pragma omp single
        rc_attn_prefill_scalar(q, k, v, positions, base, window, out, t, n, q_heads,
                                  kv_heads, hd);
        return;
    }
    const int n_rep = q_heads / kv_heads;
    int lo_key = 0;
    if (window > 0) {
        lo_key = positions[0] - window + 1 - base;
        if (lo_key < 0) {
            lo_key = 0;
        }
    }
    const int nvis = n - lo_key;
    if (nvis <= 0) {
        #pragma omp single
        memset(out, 0, (size_t)t * q_heads * hd * sizeof(float));
        return;
    }
    const int ld = ((nvis + wide - 1) / wide) * wide;
    const long rows = (long)t * n_rep;
    const long last = rows - 1;
    float *kt = NULL;
    #pragma omp single copyprivate(kt)
    {
        kt = (float *)malloc((size_t)kv_heads * hd * ld * sizeof(float));
        if (kt != NULL) {
            gemma_attn_transpose_k(k, kt, nvis, ld, kv_heads, hd, lo_key);
        }
    }
    if (kt == NULL) {
        return;
    }
#if defined(__AVX512F__)
    if (wide == 16) {
        const int nrb = (int)((rows + ATTN5_MC - 1) / ATTN5_MC);
        gemma_attn_prefill_avx512_tasks(q, v, kt, positions, base, window, out, n, q_heads,
                                        kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb,
                                        kv_heads * nrb, limit);
    } else
#endif
    {
        const int nrb = (int)((rows + ATTN2_MC - 1) / ATTN2_MC);
        gemma_attn_prefill_avx2_tasks(q, v, kt, positions, base, window, out, n, q_heads,
                                      kv_heads, hd, n_rep, lo_key, ld, rows, last, nrb,
                                      kv_heads * nrb, limit);
    }
    #pragma omp single
    free(kt);
#else
    #pragma omp single
    rc_attn_prefill_scalar(q, k, v, positions, base, window, out, t, n, q_heads, kv_heads, hd);
#endif
}

/* limit: see gemma_attn_prefill_avx512_tasks (the AVX-512 and AVX2 versions) */
void rc_attn_prefill_region(const float *q, const float *k, const float *v,
                               const int32_t *positions, int base, int window,
                               float *out, int t, int n, int q_heads, int kv_heads, int hd)
{
    gemma_attn_prefill_region_lim(q, k, v, positions, base, window, out, t, n, q_heads, kv_heads,
                                  hd, NULL);
}
