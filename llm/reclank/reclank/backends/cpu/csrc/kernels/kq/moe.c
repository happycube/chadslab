/* n values (a multiple of 32) rounded to int16 in each 32: q = rint(x / s),
 * s = max |x| / 32767, then q s (the test of int16 activations). */
static void kq_round_i16(float *x, int n)
{
    for (int g = 0; g < n; g += 32) {
        float m = 0.f;
        for (int i = 0; i < 32; ++i) {
            m = fmaxf(m, fabsf(x[g + i]));
        }
        float s = m / 32767.f;
        for (int i = 0; i < 32 && s > 0.f; ++i) {
            x[g + i] = (float)lrintf(x[g + i] / s) * s;
        }
    }
}

/* The exact path of kq_moe_body for a type other than KQ_Q4X (a test): nr
 * rows of w (cols values; rc_kq_rows to float) times the n float rows of x
 * (stride xst) into out[j * ost + i]. */
static void kq_rows_f_any(const uint8_t *w, int type, int nr, int cols, const float *x, size_t xst,
                          int n, float *out, size_t ost)
{
    float wf[16 * cols];
    int64_t ids[16];
    for (int i = 0; i < nr; ++i) {
        ids[i] = i;
    }
    rc_kq_rows(w, type, cols, ids, nr, wf);
    for (int j = 0; j < n; ++j) {
        const float *xr = x + (size_t)j * xst;
        for (int i = 0; i < nr; ++i) {
            const float *wr = wf + (size_t)i * cols;
            float acc = 0.f;
            for (int c = 0; c < cols; ++c) {
                acc = fmaf(wr[c], xr[c], acc);
            }
            out[(size_t)j * ost + i] = acc;
        }
    }
}

/* nr rows of KQ_BF12 (rb bytes each) on n float32 rows of x (xst apart):
 * out[j * ost + i]. The experts in BF12 (an overlay of the MTP layer from the
 * bfloat16 of the original: the values of bfloat16, but the zeroed ones)
 * read the float32 rows, not int8 or int16 ones. */
#if defined(__AVX512F__)
/* A BF12 row as the float32 values that kq_dot_bf12 multiplies (its decode,
 * stored). */
static void kq_bf12_row_f(const uint8_t *w, int cols, float *out)
{
    const uint8_t *lo = w, *hi = w + cols, *E = w + cols + cols / 2;
    const __m512i m7f = _mm512_set1_epi16(0x7f), m80 = _mm512_set1_epi16(0x80);
    const __m512i g15 = _mm512_set1_epi16(15), m15 = _mm512_set1_epi16(15);
    for (int g = 0; g < cols / 32; ++g) {
        __m512i l = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(lo + 32 * g)));
        __m256i h16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(hi + 16 * g)));
        __m512i gp = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(h16, _mm512_castsi512_si256(m15))),
                                        _mm256_srli_epi16(h16, 4), 1);
        __m512i e = _mm512_sub_epi16(_mm512_set1_epi16(E[g]), gp);
        __m512i b = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(l, m80), 8),
                                                    _mm512_slli_epi16(e, 7)), _mm512_and_si512(l, m7f));
        b = _mm512_maskz_mov_epi16(~(_mm512_cmpeq_epi16_mask(gp, g15) & _mm512_cmpeq_epi16_mask(l, m80)), b);
        _mm512_storeu_ps(out + 32 * g, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                             _mm512_castsi512_si256(b)), 16)));
        _mm512_storeu_ps(out + 32 * g + 16, _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                             _mm512_extracti64x4_epi64(b, 1)), 16)));
    }
}

/* The dot of a decoded BF12 row (kq_bf12_row_f) and x in the order of
 * kq_dot_bf12: the same bits. */
static float kq_dot_bf12_f(const float *wf, int cols, const float *x)
{
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    for (int g = 0; g < cols / 32; ++g) {
        a0 = _mm512_fmadd_ps(_mm512_loadu_ps(wf + 32 * g), _mm512_loadu_ps(x + 32 * g), a0);
        a1 = _mm512_fmadd_ps(_mm512_loadu_ps(wf + 32 * g + 16), _mm512_loadu_ps(x + 32 * g + 16), a1);
    }
    return _mm512_reduce_add_ps(_mm512_add_ps(a0, a1));
}
#endif

#if defined(__AVX512F__)
/* The bfloat16 bits of group g of a BF12 row (kq_dot_bf12's decode). */
static inline __m512i kq_bf12_group(const uint8_t *w, int cols, int g)
{
    const uint8_t *lo = w, *hi = w + cols, *E = w + cols + cols / 2;
    const __m512i m7f = _mm512_set1_epi16(0x7f), m80 = _mm512_set1_epi16(0x80);
    const __m512i g15 = _mm512_set1_epi16(15), m15 = _mm512_set1_epi16(15);
    __m512i l = _mm512_cvtepu8_epi16(_mm256_loadu_si256((const __m256i *)(lo + 32 * g)));
    __m256i h16 = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(hi + 16 * g)));
    __m512i gp = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(h16, _mm512_castsi512_si256(m15))),
                                    _mm256_srli_epi16(h16, 4), 1);
    __m512i e = _mm512_sub_epi16(_mm512_set1_epi16(E[g]), gp);
    __m512i b = _mm512_or_si512(_mm512_or_si512(_mm512_slli_epi16(_mm512_and_si512(l, m80), 8),
                                                _mm512_slli_epi16(e, 7)), _mm512_and_si512(l, m7f));
    return _mm512_maskz_mov_epi16(~(_mm512_cmpeq_epi16_mask(gp, g15) & _mm512_cmpeq_epi16_mask(l, m80)), b);
}

/* Four BF12 rows (rb bytes apart) on one float row x: out[i * ost], each the
 * sums of kq_dot_bf12 in its order (the same bits); x read once for the four,
 * four chains at a time. */
static void kq_dot4_bf12(const uint8_t *w, size_t rb, int cols, const float *x, float *out, size_t ost)
{
    __m512 a[8];
    for (int i = 0; i < 8; ++i) {
        a[i] = _mm512_setzero_ps();
    }
    for (int g = 0; g < cols / 32; ++g) {
        __m512 x0 = _mm512_loadu_ps(x + 32 * g), x1 = _mm512_loadu_ps(x + 32 * g + 16);
        #pragma GCC unroll 4
        for (int i = 0; i < 4; ++i) {
            __m512i b = kq_bf12_group(w + (size_t)i * rb, cols, g);
            a[2 * i] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                           _mm512_castsi512_si256(b)), 16)), x0, a[2 * i]);
            a[2 * i + 1] = _mm512_fmadd_ps(_mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepu16_epi32(
                                               _mm512_extracti64x4_epi64(b, 1)), 16)), x1, a[2 * i + 1]);
        }
    }
    for (int i = 0; i < 4; ++i) {
        out[(size_t)i * ost] = _mm512_reduce_add_ps(_mm512_add_ps(a[2 * i], a[2 * i + 1]));
    }
}
#endif

static void kq_rows_bf12_f(const uint8_t *w, size_t rb, int nr, int cols, const float *x, size_t xst,
                           int n, float *out, size_t ost)
{
#if defined(__AVX512F__)
    static int dot4 = -1;       /* RECLANK_BF12_DOT4=0: one row at a time (a check of the bits) */
    if (dot4 < 0) {
        const char *v = getenv("RECLANK_BF12_DOT4");
        dot4 = v ? atoi(v) != 0 : 1;
    }
    if (dot4 && n == 1 && nr % 4 == 0) {
        /* one token (a decode step): four rows at a time */
        for (int i = 0; i < nr; i += 4) {
            kq_dot4_bf12(w + (size_t)i * rb, rb, cols, x, out + i, 1);
        }
        return;
    }
    if (n > 1 && cols <= 8192) {
        /* each row decoded once for the n rows of x (the BF12 routed experts of
         * a group: one decode a row, not one for each token) */
        float wf[8192] __attribute__((aligned(64)));
        for (int i = 0; i < nr; ++i) {
            kq_bf12_row_f(w + (size_t)i * rb, cols, wf);
            for (int j = 0; j < n; ++j) {
                out[(size_t)j * ost + i] = kq_dot_bf12_f(wf, cols, x + (size_t)j * xst);
            }
        }
        return;
    }
#endif
    for (int j = 0; j < n; ++j) {
        const float *xr = x + (size_t)j * xst;
        for (int i = 0; i < nr; ++i) {
            out[(size_t)j * ost + i] = kq_dot1(w + (size_t)i * rb, KQ_BF12, cols, NULL, NULL, NULL, xr);
        }
    }
}

static int kq_mats_bf12(const int64_t *mats)
{
    for (int i = 1; i < 12; i += 2) {
        if (mats[i - 1] != 0 && (int)mats[i] == KQ_BF12) {
            return 1;
        }
    }
    return 0;
}

#define KQ_A4_MAXC 8192      /* (the largest hidden of the A4 rows: a row on the stack) */
static void kq_moe_body(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                        const float *val, int t, int k, int experts, const int64_t *mats,
                        const float *shared_logit, int hidden, int inner, uint8_t *scratch,
                        float *out, const int32_t *kcount, int act_flags, const float *hf,
                        const int64_t *mats1, const int32_t *slot1, const int32_t *slot0)
{
    int gelu = act_flags & 1, exact = (act_flags & 2) && hf != NULL;
    /* bit 4 with exact: the float rows and the GELU rounded to int16 in each
     * 32 values (the scale max / 32767), then the float products: the values
     * of int16 activations (a test of their precision against int8) */
    int emu16 = (act_flags & 16) && exact;
    /* bit 5: x and the GELU as int16 (kq_quant_part16, two int8 planes in
     * kq_t2_tile; KQ_NVX: kq_nvx_rows2), for the types of kq_x16s_ok; else int8 */
    int x2 = (act_flags & 32) && hf != NULL && !exact;
    /* bit 6: the rows form (Qwen GPU, GP_MOE_COMBINE): no sum, the row of each
     * pair. x16 for the types of kq_x16s_ok (KQ_NVX too: kq_nvx_rows2), as the
     * hot experts of the GPU */
    const int rowsf = (act_flags & 64) != 0;
    for (int i = 1; i < 12; i += 2) {
        if (mats[i - 1] != 0 && !kq_x16s_ok((int)mats[i], i % 6 == 5 ? inner : hidden)) {
            x2 = 0;
        }
    }
    /* KQ_BF12 matrices read float32 rows: x (hf) and the act */
    const int bf12 = kq_mats_bf12(mats) && hf != NULL;
    if (kq_mats_bf12(mats) && hf == NULL) {
        fprintf(stderr, "kq_moe_body: KQ_BF12 experts need the float32 rows of x (hf)\n");
        abort();
    }
#ifdef KQ_Q4X_Q16
    int q16 = (act_flags & 4) && hf != NULL;
#else
    const int q16 = 0;
#endif
    if (kcount != NULL) {
        /* One token with a count of experts in memory: the cold experts of a
         * GPU step (GP_HOT_SPLIT writes the count). Read once, and at most the
         * k of the record (the scratch is for that many: pinned memory that
         * the GPU writes) */
        int kc = *(const volatile int32_t *)kcount;
        k = kc < 0 ? 0 : kc < k ? kc : k;
    }
    if (kq_e3_bits((int)mats[1])) {
        /* the MUL1 experts of an EXL3 model (kernels/kq/exl3_moe.c) */
        kq_e3_moe_body(ids, val, t, k, experts, mats, shared_logit, hidden, inner, out, hf, mats1,
                       slot1, slot0, act_flags, scratch);
        return;
    }
    if ((act_flags & 8) && hf != NULL && t <= 4 && !exact && !q16) {
        /* float rows of a few tokens: the form with few barriers; x16 for
         * the types of kq_x16s_ok (KQ_NVX too, which the tiles of this body
         * do not take) */
        int x2s = (act_flags & 32) != 0;
        for (int i = 1; i < 12; i += 2) {
            if (mats[i - 1] != 0 && !kq_x16s_ok((int)mats[i], i % 6 == 5 ? inner : hidden)) {
                x2s = 0;
            }
        }
        kq_moe_small_body(ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                          out, gelu, hf, mats1, slot1, slot0, x2s, (act_flags & 64) != 0);
        return;
    }
    /* mats1 (or null): a copy of the experts in the memory of NUMA node 1
     * (QwenGPU, RECLANK_GPU_NUMA_COPY): a thread on a CPU of node 1 reads
     * that copy for the experts it has (all, or slot1[e] >= 0), the others
     * mats, so the reads are local. The threads share the tasks as before;
     * the values are the same bytes. */
    if (kq_gprof_on < 0) {
        const char *v = getenv("RECLANK_MOE_GPROF");
        kq_gprof_on = v && v[0] == '1';
    }
    const int gp = kq_gprof_on;
    double *gt = kq_gt[omp_get_thread_num() & 255];
    double glast = 0;
    if (gp) {
        gt[0] = omp_get_wtime();
    }
    int on1 = mats1 != NULL && kq_my_node() == 1;
    kq_mat G = kq_mat_of(mats), U = kq_mat_of(mats + 2), D = kq_mat_of(mats + 4);
    kq_mat SG = kq_mat_of(mats + 6), SU = kq_mat_of(mats + 8), SD = kq_mat_of(mats + 10);
    kq_mat G1 = G, U1 = U, D1 = D;
    if (mats1 != NULL) {
        G1 = kq_mat_of(mats1);
        U1 = kq_mat_of(mats1 + 2);
        D1 = kq_mat_of(mats1 + 4);
    }
    int shared = SG.w != NULL;
    /* act bit 2048 (FP4-Local, docs/FP4LOCAL_PLAN.md 2.2): the A4 contract of b12x for the prompt
     * rows on KQ_NVX experts: x to NVFP4 with the layer's gs1 (the heads of the gate matrices, bytes
     * 4-7), kq_nvx_rows_a4 (alpha = g / gs1), the act kq_b12x_act, requantized with the expert's gs2
     * (the heads of its down matrix), the down rows y = bf16(alpha2 sum), the sum of w y */
    const int a4 = (act_flags & 2048) && hf != NULL && G.type == KQ_NVX && U.type == KQ_NVX && D.type == KQ_NVX &&
                   !shared;
    if (a4) {
        x2 = 0;
    }
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
    int32_t *xn = (int32_t *)ma_take(&p, (size_t)P * (hidden / 16) * 8);    /* (2 planes with x2) */
    int32_t *an = (int32_t *)ma_take(&p, (size_t)P * (inner / 16) * 8);
    /* the int16 rows of h and of the GELU (act bit 2) */
    int16_t *x16 = (int16_t *)ma_take(&p, (size_t)P * hidden * 2);
    float *xs16 = (float *)ma_take(&p, (size_t)P * nph * 4);
    int16_t *a16 = (int16_t *)ma_take(&p, (size_t)P * inner * 2);
    float *as16 = (float *)ma_take(&p, (size_t)P * npi * 4);
#if !defined(__AVX512VNNI__) && defined(__AVX2__) && defined(KQ_Q4X_Q16)
    /* AVX2: the planes of x16 and a16 (kq_planes16_to), for the experts of
     * one token (kq_q4x_rows_p16; the bits of kq_q4x_rows16) */
#define KQ16_PLANES 1
    int8_t *xh16 = (int8_t *)ma_take(&p, (size_t)P * hidden);
    uint8_t *xl16 = (uint8_t *)ma_take(&p, (size_t)P * hidden);
    int32_t *xn16 = (int32_t *)ma_take(&p, (size_t)P * nph * 4);
    int8_t *ah16 = (int8_t *)ma_take(&p, (size_t)P * inner);
    uint8_t *al16 = (uint8_t *)ma_take(&p, (size_t)P * inner);
    int32_t *an16 = (int32_t *)ma_take(&p, (size_t)P * npi * 4);
#endif
    /* x2 (act bit 5): the low planes of h and of the GELU as int16
     * (kq_quant_part16); xq and aq hold the high planes */
    int8_t *xq2 = NULL, *aq2 = NULL;
    if (x2) {
        xq2 = (int8_t *)ma_take(&p, (size_t)P * hidden);
        aq2 = (int8_t *)ma_take(&p, (size_t)P * inner);
    }
    int *utmp = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    /* KQ_NVX: tasks of 16 rows (a group) for all the matrices of the layer */
    int ngu = G.type == KQ_NVX || G.type == KQ_Q4X ? 16 : 4;
    int ndn = D.type == KQ_NVX || D.type == KQ_Q4X ? 16 : 4;
    /* the sums of xn and an of a pair: one plane, or with x2 (KQ_NVX) both */
    const size_t xns = (size_t)(hidden / 16) * (x2 && ngu == 16 ? 2 : 1);
    const size_t ans = (size_t)(inner / 16) * (x2 && ndn == 16 ? 2 : 1);
    moe_sort_pairs(ids, t, k, experts, shared, cnt, start, used, pair_tok, pair_of, nused_p);
    int nu = *nused_p;
    /* a partial copy on node 1 (strict): the threads of node 1 take only the
     * experts of the copy (kq_used_split: those last in used; a slot for
     * each thread, its tasks by kq_task_or_end) */
    const int strict = (mats1 != NULL && slot1 != NULL) || slot0 != NULL;
    #pragma omp single
    {
        if (strict) {
            kq_used_split3(used, nu, experts, slot0, slot1, mats1 != NULL, utmp, &nused_p[1], &nused_p[6]);
        } else {
            nused_p[1] = nu;
            nused_p[6] = 0;
        }
        nused_p[2] = nused_p[3] = nused_p[4] = nused_p[5] = nused_p[7] = nused_p[8] = 0;
    }
    P = start[ne];          /* the pairs with an expert */
    if (gp) {
        gt[1] = omp_get_wtime();
    }
    /* int16: a matrix that is not KQ_Q4X (a shared expert, another type)
     * takes int8 rows, quantized here from hf and from the GELU */
    int q8u = q16 && (G.type != KQ_Q4X || (U.w != NULL && U.type != KQ_Q4X) ||
                      (shared && (SG.type != KQ_Q4X || SU.type != KQ_Q4X)));
    int q8d = q16 && (D.type != KQ_Q4X || (shared && SD.type != KQ_Q4X));
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        int j = pair_tok[q];
        if (q16) {
            for (int g = 0; g < nph; ++g) {
                xs16[(size_t)q * nph + g] = gemma_quant_group32_i16(
                    hf + (size_t)j * hidden + 32 * g, x16 + (size_t)q * hidden + 32 * g);
            }
#ifdef KQ16_PLANES
            kq_planes16_to(x16 + (size_t)q * hidden, hidden, xh16 + (size_t)q * hidden,
                           xl16 + (size_t)q * hidden, xn16 + (size_t)q * nph);
#endif
            if (!q8u) {
                continue;
            }
            for (int g = 0; g < nph; ++g) {
                kq_quant_part(hf + (size_t)j * hidden, g, xq + (size_t)q * hidden,
                              xs + (size_t)q * nph, xm + (size_t)q * 2 * nph);
            }
            if (ngu == 16) {
                kq_x16_xsum(G.type, xq + (size_t)q * hidden, hidden, xn + (size_t)q * xns);
            }
            continue;
        }
        if (a4) {
            float gs1;
            memcpy(&gs1, G.w + 4, 4);
            float xr[KQ_A4_MAXC];
            for (int i = 0; i < hidden; ++i) {
                xr[i] = kq_bf16r(hf[(size_t)j * hidden + i]);
            }
            kq_a4_quant(xr, hidden, gs1, xq + (size_t)q * hidden, xm + (size_t)q * (hidden / 16));
            kq_nvx_xsum(xq + (size_t)q * hidden, 1, hidden, xn + (size_t)q * xns);
            continue;
        }
        if (bf12 && !exact) {
            /* the float32 rows for the KQ_BF12 matrices (kq_rows_bf12_f), in de
             * as for exact; the int8 rows below for the others */
            memcpy(de + (size_t)q * hidden, hf + (size_t)j * hidden, (size_t)hidden * 4);
        }
        if (exact) {
            /* the float32 rows: in de, which the down matrix writes later */
            memcpy(de + (size_t)q * hidden, hf + (size_t)j * hidden, (size_t)hidden * 4);
            if (emu16) {
                kq_round_i16(de + (size_t)q * hidden, hidden);
            }
            continue;
        }
        memcpy(xq + (size_t)q * hidden, hq + (size_t)j * hidden, (size_t)hidden);
        memcpy(xs + (size_t)q * nph, hs + (size_t)j * nph, (size_t)nph * 4);
        memcpy(xm + (size_t)q * 2 * nph, hm + (size_t)j * 2 * nph, (size_t)nph * 8);
        if (x2) {
            kq_quant_row16(hf + (size_t)j * hidden, hidden, xq + (size_t)q * hidden,
                           xq2 + (size_t)q * hidden, xs + (size_t)q * nph, xm + (size_t)q * 2 * nph);
        }
        if (ngu == 16) {
            kq_x16_xsum(G.type, xq + (size_t)q * hidden, hidden, xn + (size_t)q * xns);
            if (x2) {
                kq_x16_xsum(G.type, xq2 + (size_t)q * hidden, hidden, xn + (size_t)q * xns + hidden / 16);
            }
        }
    }
    /* A group: the experts have different counts of tokens, so the tasks
     * go to the threads as they finish. With a static split, the threads
     * were idle about 28% of the time. A decode step: the same work in each
     * task, a static split. Each thread sets the schedule of its own
     * loops. */
    if (gp) {
        gt[2] = omp_get_wtime();
    }
    omp_set_schedule(strict ? omp_sched_static : t > 1 ? omp_sched_dynamic : omp_sched_static,
                     strict ? 1 : t > 1 ? 8 : 0);
    /* Gate and up: 2 * inner rows of each used expert, in tasks of 4 rows
     * (16 for KQ_NVX). */
    const int ngt = nu * (2 * inner / ngu), nagt = nused_p[1] * (2 * inner / ngu);
    const int nabgt = (nused_p[1] + nused_p[6]) * (2 * inner / ngu);
    #pragma omp for schedule(runtime)
    for (int x0 = 0; x0 < (strict ? omp_get_num_threads() : ngt); ++x0) {
      for (int x = strict ? kq_task_or_end(&nused_p[2], nagt, &nused_p[3], nabgt, &nused_p[7], ngt, on1) : x0;
           x < (strict ? INT_MAX : x0 + 1);
           x = strict ? kq_task_or_end(&nused_p[2], nagt, &nused_p[3], nabgt, &nused_p[7], ngt, on1) : x + 1) {
        int e = used[x / (2 * inner / ngu)], rr = (x % (2 * inner / ngu)) * ngu, isup = rr >= inner;
        int r = rr % inner;
        int c1 = on1 && e < experts && (slot1 == NULL || slot1[e] >= 0);
        if ((on1 && slot1 != NULL && !c1) || (!on1 && slot0 != NULL && e < experts && slot0[e] < 0)) {
            __atomic_add_fetch(&kq_strict_bad, 1, __ATOMIC_RELAXED);
        }
        /* the index in the copy read: node 1's, else node 0's (slot0) */
        int ex = c1 ? (slot1 != NULL ? slot1[e] : e) : (slot0 != NULL && e < experts ? slot0[e] : e);
        kq_mat m = e < experts ? (isup ? (c1 ? U1 : U) : (c1 ? G1 : G)) : (isup ? SU : SG);
        size_t rb, estride = inner;
        if (e < experts && U.w == NULL) {
            /* G holds the gate rows and then the up rows of each expert (the
             * KQ_Q4X copy of a gate and up stack: one copy for the CPU and
             * the GPU) */
            m = c1 ? G1 : G;
            r = rr;
            estride = 2 * (size_t)inner;
        }
        rb = kq_row_bytes(m.type, hidden);
        const uint8_t *w = m.w + (e < experts ? (size_t)ex * estride * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        if (exact && m.type == KQ_Q4X) {
            for (int i = 0; i < ngu; i += 16) {
                kq_q4x_rows_f(w + (size_t)(r + i) * rb, hidden, de + (size_t)s0 * hidden, (size_t)hidden,
                              n, act + (size_t)s0 * 2 * inner + rr + i, (size_t)2 * inner);
            }
            continue;
        }
        if (m.type == KQ_BF12 && (bf12 || exact)) {
            kq_rows_bf12_f(w + (size_t)r * rb, rb, ngu, hidden, de + (size_t)s0 * hidden,
                           (size_t)hidden, n, act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
            continue;
        }
        if (a4) {
            kq_nvx_rows_a4(w + (size_t)r * rb, hidden, xq + (size_t)s0 * hidden, xm + (size_t)s0 * (hidden / 16),
                           xn + (size_t)s0 * xns, n, act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
            continue;
        }
        if (exact) {
            kq_rows_f_any(w + (size_t)r * rb, m.type, ngu, hidden, de + (size_t)s0 * hidden,
                          (size_t)hidden, n, act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
            continue;
        }
#ifdef KQ_Q4X_Q16
        if (q16 && m.type == KQ_Q4X) {
#ifdef KQ16_PLANES
            if (n == 1) {
                for (int i = 0; i < ngu; i += 16) {
                    kq_q4x_rows_p16(w + (size_t)(r + i) * rb, hidden, xh16 + (size_t)s0 * hidden,
                                    xl16 + (size_t)s0 * hidden, xn16 + (size_t)s0 * nph,
                                    xs16 + (size_t)s0 * nph, act + (size_t)s0 * 2 * inner + rr + i);
                }
                continue;
            }
#endif
            for (int i = 0; i < ngu; i += 16) {
                kq_q4x_rows16(w + (size_t)(r + i) * rb, hidden, x16 + (size_t)s0 * hidden,
                              xs16 + (size_t)s0 * nph, n, act + (size_t)s0 * 2 * inner + rr + i,
                              (size_t)2 * inner);
            }
            continue;
        }
#endif
        kq_rows_n2(w + (size_t)r * rb, rb, m.type, ngu, hidden, xq + (size_t)s0 * hidden,
                   xs + (size_t)s0 * nph, xm + (size_t)s0 * 2 * nph, xn + (size_t)s0 * xns,
                   x2 ? xq2 + (size_t)s0 * hidden : NULL, n,
                   act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
        if (gp) {
            glast = omp_get_wtime();
        }
      }
    }
    if (gp) {
        gt[3] = glast;
        gt[4] = omp_get_wtime();
    }
    /* silu(gate) * up (gelu: the tanh GELU of the Gemma 4 26B), and its
     * quantization for down. */
    if (a4) {
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        float *a = act + (size_t)q * 2 * inner;
        for (int i = 0; i < inner; ++i) {
            a[i] = kq_b12x_act(a[i], a[inner + i]);
        }
        int lo = 0, hi = ne;            /* the expert of pair q: start[lo] <= q < start[lo + 1] */
        while (hi - lo > 1) {
            int mid = (lo + hi) / 2;
            if (start[mid] <= q) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        float gs2;
        memcpy(&gs2, D.w + (size_t)lo * hidden * kq_row_bytes(KQ_NVX, inner) + 4, 4);
        kq_a4_quant(a, inner, gs2, aq + (size_t)q * inner, am + (size_t)q * (inner / 16));
        kq_nvx_xsum(aq + (size_t)q * inner, 1, inner, an + (size_t)q * ans);
    }
    } else {
    #pragma omp for schedule(static)
    for (int q = 0; q < P; ++q) {
        float *a = act + (size_t)q * 2 * inner;
        if (gelu) {
            int i = 0;
#if GEMMA_X86 && defined(__AVX512F__)
            const __m512 cv = _mm512_set1_ps(0.7978845608028654f), half = _mm512_set1_ps(0.5f);
            const __m512 one = _mm512_set1_ps(1.0f), k3 = _mm512_set1_ps(0.044715f);
            for (; i + 16 <= inner; i += 16) {
                __m512 v = _mm512_loadu_ps(a + i);
                __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
                __m512 th = gemma_tanh_ps(_mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v)));
                __m512 r = _mm512_mul_ps(_mm512_mul_ps(half, v), _mm512_add_ps(one, th));
                _mm512_storeu_ps(a + i, _mm512_mul_ps(r, _mm512_loadu_ps(a + inner + i)));
            }
#else
            i = GELU_MUL_VEC(a, a + inner, a, inner);
#endif
            for (; i < inner; ++i) {
                float v = a[i];
                a[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) *
                       a[inner + i];
            }
        } else {
            for (int i = 0; i < inner; ++i) {
                a[i] = ma_silu(a[i]) * a[inner + i];
            }
        }
        if (kq_moe_rot) {
            for (int g = 0; g < inner / 32; ++g) {
                tq6_rot32(a + 32 * g, 0);
            }
        }
        if (exact) {
            if (emu16) {
                kq_round_i16(a, inner);
            }
            continue;       /* the down matrix reads a (float32) */
        }
        if (q16) {
            for (int g = 0; g < npi; ++g) {
                as16[(size_t)q * npi + g] = gemma_quant_group32_i16(
                    a + 32 * g, a16 + (size_t)q * inner + 32 * g);
            }
#ifdef KQ16_PLANES
            kq_planes16_to(a16 + (size_t)q * inner, inner, ah16 + (size_t)q * inner,
                           al16 + (size_t)q * inner, an16 + (size_t)q * npi);
#endif
            if (!q8d) {
                continue;
            }
        }
        if (x2) {
            kq_quant_row16(a, inner, aq + (size_t)q * inner, aq2 + (size_t)q * inner,
                           as + (size_t)q * npi, am + (size_t)q * 2 * npi);
        } else {
            for (int g = 0; g < npi; ++g) {
                kq_quant_part(a, g, aq + (size_t)q * inner, as + (size_t)q * npi,
                              am + (size_t)q * 2 * npi);
            }
        }
        if (ndn == 16) {
            kq_x16_xsum(D.type, aq + (size_t)q * inner, inner, an + (size_t)q * ans);
            if (x2) {
                kq_x16_xsum(D.type, aq2 + (size_t)q * inner, inner, an + (size_t)q * ans + inner / 16);
            }
        }
    }
    }       /* (a4) */
    if (gp) {
        gt[5] = omp_get_wtime();
    }
    /* Down: hidden rows of each used expert, in tasks of 4 rows (16 for
     * KQ_NVX). */
    const int ndt = nu * (hidden / ndn), nadt = nused_p[1] * (hidden / ndn);
    const int nabdt = (nused_p[1] + nused_p[6]) * (hidden / ndn);
    #pragma omp for schedule(runtime)
    for (int x0 = 0; x0 < (strict ? omp_get_num_threads() : ndt); ++x0) {
      for (int x = strict ? kq_task_or_end(&nused_p[4], nadt, &nused_p[5], nabdt, &nused_p[8], ndt, on1) : x0;
           x < (strict ? INT_MAX : x0 + 1);
           x = strict ? kq_task_or_end(&nused_p[4], nadt, &nused_p[5], nabdt, &nused_p[8], ndt, on1) : x + 1) {
        int e = used[x / (hidden / ndn)], r = (x % (hidden / ndn)) * ndn;
        int c1 = on1 && e < experts && (slot1 == NULL || slot1[e] >= 0);
        if ((on1 && slot1 != NULL && !c1) || (!on1 && slot0 != NULL && e < experts && slot0[e] < 0)) {
            __atomic_add_fetch(&kq_strict_bad, 1, __ATOMIC_RELAXED);
        }
        /* the index in the copy read: node 1's, else node 0's (slot0) */
        int ex = c1 ? (slot1 != NULL ? slot1[e] : e) : (slot0 != NULL && e < experts ? slot0[e] : e);
        kq_mat m = e < experts ? (c1 ? D1 : D) : SD;
        size_t rb = kq_row_bytes(m.type, inner);
        const uint8_t *w = m.w + (e < experts ? (size_t)ex * hidden * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        if (exact && m.type == KQ_Q4X) {
            for (int i = 0; i < ndn; i += 16) {
                kq_q4x_rows_f(w + (size_t)(r + i) * rb, inner, act + (size_t)s0 * 2 * inner,
                              (size_t)2 * inner, n, de + (size_t)s0 * hidden + r + i, (size_t)hidden);
            }
            continue;
        }
        if (m.type == KQ_BF12 && (bf12 || exact)) {
            kq_rows_bf12_f(w + (size_t)r * rb, rb, ndn, inner, act + (size_t)s0 * 2 * inner,
                           (size_t)2 * inner, n, de + (size_t)s0 * hidden + r, (size_t)hidden);
            continue;
        }
        if (a4) {
            kq_nvx_rows_a4(w + (size_t)r * rb, inner, aq + (size_t)s0 * inner, am + (size_t)s0 * (inner / 16),
                           an + (size_t)s0 * ans, n, de + (size_t)s0 * hidden + r, (size_t)hidden);
            continue;
        }
        if (exact) {
            kq_rows_f_any(w + (size_t)r * rb, m.type, ndn, inner, act + (size_t)s0 * 2 * inner,
                          (size_t)2 * inner, n, de + (size_t)s0 * hidden + r, (size_t)hidden);
            continue;
        }
#ifdef KQ_Q4X_Q16
        if (q16 && m.type == KQ_Q4X) {
#ifdef KQ16_PLANES
            if (n == 1) {
                for (int i = 0; i < ndn; i += 16) {
                    kq_q4x_rows_p16(w + (size_t)(r + i) * rb, inner, ah16 + (size_t)s0 * inner,
                                    al16 + (size_t)s0 * inner, an16 + (size_t)s0 * npi,
                                    as16 + (size_t)s0 * npi, de + (size_t)s0 * hidden + r + i);
                }
                continue;
            }
#endif
            for (int i = 0; i < ndn; i += 16) {
                kq_q4x_rows16(w + (size_t)(r + i) * rb, inner, a16 + (size_t)s0 * inner,
                              as16 + (size_t)s0 * npi, n, de + (size_t)s0 * hidden + r + i,
                              (size_t)hidden);
            }
            continue;
        }
#endif
        kq_rows_n2(w + (size_t)r * rb, rb, m.type, ndn, inner, aq + (size_t)s0 * inner,
                   as + (size_t)s0 * npi, am + (size_t)s0 * 2 * npi, an + (size_t)s0 * ans,
                   x2 ? aq2 + (size_t)s0 * inner : NULL, n,
                   de + (size_t)s0 * hidden + r, (size_t)hidden);
        if (gp) {
            glast = omp_get_wtime();
        }
      }
    }
    if (gp) {
        gt[6] = glast;
        gt[7] = omp_get_wtime();
    }
    if (a4) {
        /* y = bf16(alpha2 sum) of each pair (the down rows of b12x's A4) */
        #pragma omp for schedule(static)
        for (int q = 0; q < P; ++q) {
            float *d = de + (size_t)q * hidden;
            for (int i = 0; i < hidden; ++i) {
                d[i] = kq_bf16r(d[i]);
            }
        }
    }
    if (rowsf) {
        moe_rows(de, pair_of, t, k, hidden, out);
    } else {
        moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
    }
    if (gp) {
        gt[8] = omp_get_wtime();
        #pragma omp barrier
        #pragma omp master
        {
            int nt = omp_get_num_threads() < 256 ? omp_get_num_threads() : 256;
            double ref = kq_gt[0][0], amax = ref, m3 = 0, x3 = 0, m6 = 0, x6 = 0;
            for (int q = 0; q < nt; ++q) {
                ref = kq_gt[q][0] < ref ? kq_gt[q][0] : ref;
                amax = kq_gt[q][0] > amax ? kq_gt[q][0] : amax;
            }
            for (int q = 0; q < nt; ++q) {
                m3 += (kq_gt[q][3] - ref) / nt;
                m6 += (kq_gt[q][6] - ref) / nt;
                x3 = kq_gt[q][3] - ref > x3 ? kq_gt[q][3] - ref : x3;
                x6 = kq_gt[q][6] - ref > x6 ? kq_gt[q][6] - ref : x6;
            }
            double v[11] = {amax - ref, gt[1] - ref, gt[2] - ref, m3, x3, gt[4] - ref, gt[5] - ref,
                            m6, x6, gt[7] - ref, gt[8] - ref};
            for (int i = 0; i < 11; ++i) {
                kq_gprof_s[i] += v[i];
            }
            kq_gprof_s[11] += 1;
        }
    }
}

void rc_kq_moe(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
            const float *val, int t, int k, int experts, const int64_t *mats,
            const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL, 0, NULL, NULL, NULL, NULL);
}

/* rc_kq_moe with a copy of the experts on NUMA node 1 (mats1; slot1: the slot
 * of each expert in it, or -1; null: all), as GP_KQ_MOE. */
void rc_kq_moe_numa(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                 const float *val, int t, int k, int experts, const int64_t *mats,
                 const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out,
                 const int64_t *mats1, const int32_t *slot1)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL, 0, NULL, mats1, slot1, NULL);
}

/* rc_kq_moe_numa with act flags and the float rows hf (a test: act bit 3 with
 * t <= 4 runs kq_moe_small_body). */
void rc_kq_moe_numa_act(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                     const float *val, int t, int k, int experts, const int64_t *mats,
                     const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out,
                     int act, const float *hf, const int64_t *mats1, const int32_t *slot1,
                     const int32_t *slot0)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL, act, hf, mats1, slot1, slot0);
}

/* rc_kq_moe with gelu 1: the tanh GELU of the gate in place of SiLU (the Gemma 4
 * 26B). */
void rc_kq_moe_act(const int8_t *hq, const float *hs, const float *hm, const int32_t *ids,
                const float *val, int t, int k, int experts, const int64_t *mats,
                const float *shared_logit, int hidden, int inner, uint8_t *scratch, float *out,
                int act_flags, const float *hf)
{
    #pragma omp parallel
    kq_moe_body(hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
                out, NULL, act_flags, hf, NULL, NULL, NULL);
}
