/* ---------- programs in parts ----------
 * SPLIT_PLAN.md, phase 1. A step can run as several programs, one for each
 * part of the machine. A part is, for example, one NUMA node. The function
 * rc_run_parts runs each part in its own team of threads. A part computes
 * a range of the rows of the large operations, so each output value still
 * comes from one thread.
 *
 * A barrier across the parts (GP_XBAR) comes before an operation that reads
 * the output of a different part. The barrier is an int64 array of the
 * caller, with a flag for each part: b[8 (p + 1)], on a cache line of its
 * own. A flag counts the barriers that its part passed.
 *
 * At a barrier, a part stores its count plus one in its flag (a release
 * store), then waits until the flag of each other part is that count or
 * more (acquire loads). Only a part writes its flag, so no line takes an
 * atomic read-modify-write of all the parts. A count grows at each barrier,
 * so a flag needs no reset within a run; rc_run_parts sets the flags to
 * 0 before a run.
 *
 * The OpenMP barrier at the start makes sure that each thread of the team
 * finished its writes. The release store then makes those writes visible to
 * the other parts. */
/* The measures of the barriers (rc_xbar_stats): for each part (the
 * thread of the outer region), the time its team waited for the other
 * parts, the time of its team barrier, and the count. */
#define GP_XS_PARTS 8
static double gp_xs_wait[GP_XS_PARTS], gp_xs_team[GP_XS_PARTS];
static int64_t gp_xs_n[GP_XS_PARTS];

int rc_xbar_stats(double *out)
{
    for (int p = 0; p < GP_XS_PARTS; ++p) {
        out[3 * p] = gp_xs_wait[p];
        out[3 * p + 1] = gp_xs_team[p];
        out[3 * p + 2] = (double)gp_xs_n[p];
        gp_xs_wait[p] = gp_xs_team[p] = 0.0;
        gp_xs_n[p] = 0;
    }
    return GP_XS_PARTS;
}

/* The count of the barriers that a thread passed. Each thread of a part
 * passes the same barriers, so each finds the same count; the runners of
 * the parts set it to 0 at the start of a run, as the flags. */
static __thread int64_t gp_xcnt;

/* The prefetch of a thread during the wait of a barrier: at most this many
 * bytes (L2 is 1 MB on the Xeon of the parts). */
#define GP_XPF_MAX (512 * 1024)

/* After the team barrier, the first thread of the team stores the count of
 * the part (release), and every thread waits for the flags of the other
 * parts itself (acquire). The barrier of the team makes the writes of the
 * team come before the store, so a thread that sees the flags of the other
 * parts sees their writes: no second team barrier.
 *
 * pf and pf_bytes (or null) are the weights of the next operation of the
 * part (parts.py, _xbar_prefetch). During the wait, each thread prefetches
 * its share of them (the share of the static schedule of the next kernel),
 * so a part that waits loads its next rows into its caches. */
static void gp_xbar(int64_t *b, int n, const uint8_t *pf, int64_t pf_bytes)
{
    double t0 = omp_get_wtime();
    int p = omp_get_ancestor_thread_num(1);
    int64_t v = ++gp_xcnt;
    #pragma omp barrier
    const int first = omp_get_thread_num() == 0;
    double t1 = omp_get_wtime();
    if (first) {
        __atomic_store_n(b + 8 * (p + 1), v, __ATOMIC_RELEASE);
    }
    const uint8_t *q = NULL, *qe = NULL;
    if (pf != NULL && pf_bytes > 0) {
        size_t nt = (size_t)omp_get_num_threads(), t = (size_t)omp_get_thread_num();
        size_t per = ((size_t)pf_bytes + nt - 1) / nt, lo = per * t;
        size_t len = lo >= (size_t)pf_bytes ? 0 : (size_t)pf_bytes - lo;
        len = len < per ? len : per;
        len = len < GP_XPF_MAX ? len : GP_XPF_MAX;
        q = pf + lo;
        qe = q + len;
    }
    for (int k = 0; k < n; ++k) {
        if (k == p) {
            continue;
        }
        while (__atomic_load_n(b + 8 * (k + 1), __ATOMIC_ACQUIRE) < v) {
#if GEMMA_X86
            if (q < qe) {
                for (int j = 0; j < 8 && q < qe; ++j, q += 64) {
                    _mm_prefetch((const char *)q, _MM_HINT_T1);
                }
            } else {
                _mm_pause();
            }
#endif
        }
    }
    if (first && p >= 0 && p < GP_XS_PARTS) {
        gp_xs_wait[p] += omp_get_wtime() - t1;
        gp_xs_team[p] += t1 - t0;
        ++gp_xs_n[p];
    }
}

/* The experts of one token, for the rows of one part. The part holds a copy
 * of some rows of each expert:
 *
 * - ni gate rows from inner row a0, and the ni up rows that go with them;
 * - nd down rows from output row c0.
 *
 * The steps:
 *
 * 1. Sort the selected experts, as gp_moe_one does.
 * 2. Run the gate and the up rows of the part and the GELU. Copy the result
 *    to columns a0 to a0 + ni of the shared activation (top_k, inner).
 * 3. Wait at the barrier until all parts wrote their columns.
 * 4. Run the down rows of the part on the whole activation.
 * 5. Add the outputs of the experts for rows c0 to c0 + nd of the output.
 *    Use the order of the expert index, as gp_moe_one does.
 *
 * Each value comes from the same instructions as in gp_moe_one, so the bits
 * are the same. This needs a0 and ni to be multiples of 16. The GELU uses 16
 * values in a vector and the rest one at a time. */
static void gp_moe_part(const gp_rec *r, const int64_t *e)
{
    const float *h = GP_P(const float, 0);
    const float *val = GP_P(const float, 1);
    const int32_t *idx = GP_P(const int32_t, 2);
    int top_k = GP_I(3);
    const uint8_t *gu_w = GP_P(const uint8_t, 4);
    const float *gu_s = GP_P(const float, 5);
    const uint8_t *dn_w = GP_P(const uint8_t, 6);
    const float *dn_s = GP_P(const float, 7);
    int gu_rows = GP_I(8);      /* 2 ni */
    int cols = GP_I(9);
    int nd = GP_I(10);
    int ni = GP_I(11);
    int32_t *ids = GP_P(int32_t, 12);
    float *act = GP_P(float, 13);     /* (top_k, 2 ni), private */
    float *act2p = GP_P(float, 14);   /* (top_k, ni), private */
    float *act2 = GP_P(float, 15);    /* (top_k, inner), shared */
    int a0 = GP_I(16);
    int inner = GP_I(17);
    float *de = GP_P(float, 18);      /* (top_k, nd), private */
    float *out = GP_P(float, 19);     /* (hidden), shared */
    int c0 = GP_I(20);
    int64_t *bar = GP_P(int64_t, 21);
    int nparts = GP_I(22);
    if (gu_s == NULL) {
        /* The KQ_Q4X copies of the rows of this part (reclank/models/gemma4/parts.py):
         * the operations of GP_KQ_MOE with float32 activations (the step of
         * one part), row by row: the same bits. gu_w holds 2 ni rows for
         * each expert (the gate rows, then the up rows), dn_w nd rows. */
        size_t rbh = (size_t)(cols / 32) * 18, rbi = (size_t)(inner / 32) * 18;
        int ggu = gu_rows / 16, gdn = nd / 16;
        /* AVX2: int8 x, as GP_KQ_MOE on that CPU (q4x_rows_x) */
        I4Q_BEGIN(h, (size_t)cols, 1, cols);
        #pragma omp for schedule(static)
        for (int x = 0; x < top_k * ggu; ++x) {
            int j = x / ggu, g = x % ggu;
            q4x_rows_x(gu_w + ((size_t)idx[j] * gu_rows + 16 * g) * rbh, cols, h, (size_t)cols, 1,
                       act + (size_t)j * gu_rows + 16 * g, (size_t)gu_rows);
        }
        #pragma omp for schedule(static)
        for (int j = 0; j < top_k; ++j) {
            const float *a = act + (size_t)j * gu_rows;
            float *o = act2 + (size_t)j * inner + a0;
            int i = 0;
#if GEMMA_X86 && defined(__AVX512F__)
            /* the GELU of kq_moe_body, 16 values at a time */
            const __m512 cv = _mm512_set1_ps(0.7978845608028654f), half = _mm512_set1_ps(0.5f);
            const __m512 one = _mm512_set1_ps(1.0f), k3 = _mm512_set1_ps(0.044715f);
            for (; i + 16 <= ni; i += 16) {
                __m512 v = _mm512_loadu_ps(a + i);
                __m512 v3 = _mm512_mul_ps(_mm512_mul_ps(v, v), v);
                __m512 th = gemma_tanh_ps(_mm512_mul_ps(cv, _mm512_fmadd_ps(k3, v3, v)));
                __m512 rr = _mm512_mul_ps(_mm512_mul_ps(half, v), _mm512_add_ps(one, th));
                _mm512_storeu_ps(o + i, _mm512_mul_ps(rr, _mm512_loadu_ps(a + ni + i)));
            }
#else
            i = GELU_MUL_VEC(a, a + ni, o, ni);
#endif
            for (; i < ni; ++i) {
                float v = a[i];
                o[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) *
                       a[ni + i];
            }
        }
        gp_xbar(bar, nparts, NULL, 0);
        I4Q_BEGIN(act2, (size_t)inner, top_k, inner);
        #pragma omp for schedule(static)
        for (int x = 0; x < top_k * gdn; ++x) {
            int j = x / gdn, g = x % gdn;
            q4x_rows_x(dn_w + ((size_t)idx[j] * nd + 16 * g) * rbi, inner, act2 + (size_t)j * inner,
                       (size_t)inner, 1, de + (size_t)j * nd + 16 * g, (size_t)nd);
        }
        /* the sum of moe_combine: from 0, the experts in the order of the
         * router */
        #pragma omp for schedule(static)
        for (int c = 0; c < nd; ++c) {
            float o = 0.f;
            for (int j = 0; j < top_k; ++j) {
                o += val[j] * de[(size_t)j * nd + c];
            }
            out[c0 + c] = o;
        }
        return;
    }
    #pragma omp single
    {
        for (int j = 0; j < top_k; ++j) {
            ids[j] = idx[j];
        }
        for (int j = 1; j < top_k; ++j) {
            int32_t x = ids[j];
            int k = j - 1;
            while (k >= 0 && ids[k] > x) {
                ids[k + 1] = ids[k];
                --k;
            }
            ids[k + 1] = x;
        }
    }
    gemma_moe_gemv_gelu_body(gu_w, gu_s, h, ids, top_k, act, act2p, gu_rows, cols, 0, ni);
    #pragma omp single
    for (int j = 0; j < top_k; ++j) {
        memcpy(act2 + (size_t)j * (size_t)inner + a0, act2p + (size_t)j * (size_t)ni,
               (size_t)ni * sizeof(float));
    }
    gp_xbar(bar, nparts, NULL, 0);
    gemma_int4_moe_gemv_body(dn_w, dn_s, act2, ids, top_k, de, nd, inner, inner);
    #pragma omp single
    {
        for (int c = 0; c < nd; ++c) {
            out[c0 + c] = 0.0f;
        }
        for (int j = 0; j < top_k; ++j) {
            int s = 0;
            while (idx[s] != ids[j]) {
                ++s;
            }
            gp_add_scaled(out + c0, de + (size_t)j * (size_t)nd, val[s], nd);
        }
    }
}
