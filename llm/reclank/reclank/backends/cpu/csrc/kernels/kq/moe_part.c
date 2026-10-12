/* ---------- the cold experts in parts (GP_KQ_MOE_PART) ----------
 *
 * The GPU+parts mode of QwenGPU (RECLANK_GPU_PARTS): the cold experts of a
 * GPU step (or of a small group) run in a part for each NUMA node, each on
 * the CPUs of its node (GP_RUN_PARTS, run.c). The parts run in one team of nparts x team threads (GP_RUN_PARTS: a
 * flat team, so OpenMP keeps its threads; nested teams were made again at
 * each call: 5.4 ms). Part p computes rows of every cold expert: the gate and the up rows a0 to
 * a0 + ni (of inner) and the down rows c0 to c0 + nd (of hidden), from a
 * copy of those rows in the memory of its node (or from the full experts
 * there). The parts exchange the act through act2 (one row of inner values
 * for each pair, shared) at a barrier of the team; each writes its rows c0
 * to c0 + nd of out.
 *
 * Each value comes from the operations of kq_moe_small_body (the int8 rows
 * of the tokens, kq_rows_n2 on groups of rows aligned as there, the act, the
 * rotation of the RQ types, the int8 act, the sum of moe_combine), so the
 * output has the bits of GP_KQ_MOE.
 *
 * pm (int64, 12 values): for gate, up, and down the address of the rows of
 * expert 0, the type, the bytes from an expert to the next, and the first
 * row (the rows of the part in the full experts: a0 or c0; in a copy of the
 * rows of the part: 0). */

static size_t kq_moe_part_scratch_bytes(int t, int k, int experts, int hidden, int inner, int ni, int nd)
{
    size_t P = (size_t)t * k, ne = (size_t)experts + 1, nph = hidden / 32, npi = inner / 32;
    size_t a = 0;
#define KQ_PT(n) a += ((size_t)(n) + 63) & ~(size_t)63
    KQ_PT(ne * 4); KQ_PT(ne * 4); KQ_PT(ne * 4);                  /* cnt, start, used */
    KQ_PT(P * 4); KQ_PT(P * 4); KQ_PT(P * 4); KQ_PT(64);          /* pair_tok, pair_of, exp_of, nused */
    KQ_PT(P * 4);                                                  /* pair_q (flag 128) */
    KQ_PT((size_t)t * hidden); KQ_PT((size_t)t * hidden);         /* tq, tq2 */
    KQ_PT((size_t)t * nph * 4); KQ_PT((size_t)t * nph * 8);       /* ts, tm */
    KQ_PT(P * hidden); KQ_PT(P * hidden);                          /* xq, xq2 */
    KQ_PT(P * nph * 4); KQ_PT(P * nph * 8); KQ_PT(P * (hidden / 16) * 8);   /* xs, xm, xn (2 planes) */
    KQ_PT(P * 2 * ni * 4);                                         /* act (gate, up of the part) */
    KQ_PT(P * inner * 4);                                          /* the act rows, rotated */
    KQ_PT(P * inner); KQ_PT(P * inner);                            /* aq, aq2 */
    KQ_PT(P * npi * 4); KQ_PT(P * npi * 8); KQ_PT(P * (inner / 16) * 8);    /* as, am, an (2 planes) */
    KQ_PT(P * nd * 4);                                             /* de */
#undef KQ_PT
    return a + 64;
}

size_t rc_kq_moe_part_scratch(int t, int k, int experts, int hidden, int inner, int ni, int nd)
{
    return kq_moe_part_scratch_bytes(t, k, experts, hidden, inner, ni, nd);
}

/* A barrier of the threads of one part (a counter and a generation of the
 * part, spinning on its socket): the barriers across the parts (the whole
 * team) cost the wake of the other socket each time. */
static int64_t kq_pbar[8][16] __attribute__((aligned(64)));

static inline void kq_part_barrier(int part, int team)
{
    int64_t *cnt = &kq_pbar[part][0], *gen = &kq_pbar[part][8];
    int64_t g = __atomic_load_n(gen, __ATOMIC_ACQUIRE);
    if (__atomic_add_fetch(cnt, 1, __ATOMIC_ACQ_REL) == team) {
        __atomic_store_n(cnt, 0, __ATOMIC_RELAXED);
        __atomic_store_n(gen, g + 1, __ATOMIC_RELEASE);
        return;
    }
    while (__atomic_load_n(gen, __ATOMIC_ACQUIRE) == g) {
#if GEMMA_X86
        _mm_pause();
#endif
    }
}

/* The counters of the units of phases B and E of each part (dynamic split:
 * kq_part_grab), on lines of their own. */
static int64_t kq_pdyn[8][16] __attribute__((aligned(64)));

/* The dynamic split of the units of B and E for a group of 32 tokens or more
 * (RECLANK_PART_DYN, default 1; 0: the static split by cost of
 * kq_part_units): each thread takes the next chunk of units of its part
 * (RECLANK_PART_CHUNK units, default 16; 4 took 7% more time in B on the real
 * prompt) until none is left. The static split left the threads of a part
 * waiting for the slowest at each barrier (0.65 s of 4.5 on the real prompt
 * of 1414 tokens): the cost of a unit is not the count of its pairs alone
 * (its rows are read and decoded once whatever it is), and the threads of a
 * socket do not run at one speed. Each unit gives its rows alone: the same
 * bits. */
static int kq_part_dyn(int *chunk)
{
    static int on = -1, ch = 16;
    if (on < 0) {
        const char *v = getenv("RECLANK_PART_DYN"), *c = getenv("RECLANK_PART_CHUNK");
        ch = c && atoi(c) > 0 ? atoi(c) : 16;
        on = v ? atoi(v) != 0 : 1;
    }
    *chunk = ch;
    return on;
}

/* The next chunk [*u0, *u1) of the tot units of counter c of the part; 0 when
 * none is left. */
static inline int kq_part_grab(int64_t *c, int64_t tot, int ch, int64_t *u0, int64_t *u1)
{
    int64_t a = __atomic_fetch_add(c, ch, __ATOMIC_RELAXED);
    if (a >= tot) {
        return 0;
    }
    *u0 = a;
    *u1 = a + ch < tot ? a + ch : tot;
    return 1;
}

/* The static share [*lo, *hi) of n tasks for thread lt of a team of team
 * threads (the split of omp for schedule(static)). */
static inline void kq_part_share(int n, int lt, int team, int *lo, int *hi)
{
    int ch = (n + team - 1) / team;
    *lo = lt * ch < n ? lt * ch : n;
    *hi = (lt + 1) * ch < n ? (lt + 1) * ch : n;
}

/* GP_KQ_MOE_PART in the flat team of GP_RUN_PARTS (rc_run_task): nparts x
 * team threads, thread tid of part tid / team. Each part has its matrices
 * (pm + 12 p), its rows (gu[2 p], gu[2 p + 1]: a0, ni; dn: c0, nd), and its
 * scratch (scr[p], in the memory of its node). The phases end at barriers
 * of the part (kq_part_barrier); the exchange of the act after phase C is
 * the one barrier of the whole team. */
/* RECLANK_PART_PROF=1: the time of each phase of each thread of the parts
 * (a measure of where a group of a prompt goes): kq_pp[thread][phase],
 * rc_part_prof gives the most and the mean over the threads. The phases: 0 A
 * (the sort, the int16 rows of the tokens), 1 its wait, 2 the rows of the
 * pairs, 3 its wait, 4 B (gate, up), 5 its wait, 6 C (the act), 7 its wait,
 * 8 D (the int16 act), 9 its wait, 10 E (down), 11 its wait, 12 F (out). */
#define KQ_PP_N 13
static int kq_pp_on = -1;
static double kq_pp[256][16];
static double kq_pp_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}
#define KQ_PP(k)                                                                                   \
    do {                                                                                           \
        if (pp) {                                                                                  \
            double now_ = kq_pp_now();                                                             \
            kq_pp[tid & 255][k] += now_ - pp_t;                                                    \
            pp_t = now_;                                                                           \
        }                                                                                          \
    } while (0)

void rc_part_prof(double *out, int reset)
{
    for (int k = 0; k < KQ_PP_N; ++k) {
        double mx = 0.0, sum = 0.0;
        int n = 0;
        for (int i = 0; i < 256; ++i) {
            if (kq_pp[i][k] > 0.0) {
                mx = kq_pp[i][k] > mx ? kq_pp[i][k] : mx;
                sum += kq_pp[i][k];
                ++n;
            }
        }
        out[k] = mx;
        out[16 + k] = n ? sum / n : 0.0;
    }
    if (reset) {
        memset(kq_pp, 0, sizeof(kq_pp));
    }
}

/* The row groups [*qa, *qb) of an expert (its pairs from s0, n of them) that
 * thread lt of a team of team takes: the units (an expert, a group of rows;
 * per of them for each expert) are split by their cost, the pairs of their
 * expert (P in all), in the order of the experts then their rows. A split by
 * the count of units gave the thread with the units of a popular expert (a
 * prompt) twice the work: a group of 512 tokens with a skewed choice took
 * 169 ms a layer, 81 with an even one. With one pair each (a step) the split
 * is the one by count. */
static inline void kq_part_units(int64_t per, int64_t P, int lt, int team, int64_t s0, int64_t n, int64_t *qa,
                                 int64_t *qb)
{
    int64_t C = per * P, clo = C * lt / team, chi = C * (lt + 1) / team, B = per * s0;
    int64_t a = clo > B ? (clo - B + n - 1) / n : 0, b = chi > B ? (chi - B + n - 1) / n : 0;
    *qa = a < per ? a : per;
    *qb = b < per ? b : per;
}

static void kq_moe_part_body(const float *hf, const int32_t *ids, const float *val, int t, int k0,
                             int experts, const int64_t *pm_all, int hidden, int inner, const int32_t *gu,
                             const int32_t *dn, float *act2, float *out, int nparts, int team,
                             const int32_t *kcount, int flags, const int64_t *scr)
{
    const int tid = omp_get_thread_num();
    if (kq_pp_on < 0) {
        const char *v = getenv("RECLANK_PART_PROF");
        kq_pp_on = v && v[0] == '1';
    }
    const int pp = kq_pp_on;
    double pp_t = pp ? kq_pp_now() : 0.0;
    if (omp_get_num_threads() != nparts * team || nparts > 8) {
        #pragma omp single
        fprintf(stderr, "kq_moe_part_body: a team of %d threads, not %d x %d\n", omp_get_num_threads(),
                nparts, team);
        return;
    }
    const int part = tid / team, lt = tid % team;
    const int k = kcount != NULL ? *kcount : k0;    /* the cold experts of one token (GP_HOT_SPLIT) */
    const int a0 = gu[2 * part], ni = gu[2 * part + 1], c0 = dn[2 * part], nd = dn[2 * part + 1];
    const int64_t *pm = pm_all + 12 * part;
    const uint8_t *Gw = (const uint8_t *)(intptr_t)pm[0], *Uw = (const uint8_t *)(intptr_t)pm[4];
    const uint8_t *Dw = (const uint8_t *)(intptr_t)pm[8];
    const int Gt = (int)pm[1], Ut = (int)pm[5], Dt = (int)pm[9];
    const size_t Gs = (size_t)pm[2], Us = (size_t)pm[6], Ds = (size_t)pm[10];
    const int Gr = (int)pm[3], Ur = (int)pm[7], Dr = (int)pm[11];
    const size_t rbg = kq_row_bytes(Gt, hidden), rbu = kq_row_bytes(Ut, hidden);
    /* int16 x (act bit 5) for the types of kq_x16s_ok, as kq_moe_small_body
     * (KQ_NVX too: the sums of its two planes in xn and an) */
    const int gelu = flags & 1;
    /* KQ_MXX has only the int16 form (kq_mxx_rows2): it takes it without bit 5 too */
    const int mxx = Gt == KQ_MXX || Ut == KQ_MXX || Dt == KQ_MXX;
    const int x2 = ((flags & 32) != 0 || mxx) && kq_x16s_ok(Gt, hidden) && kq_x16s_ok(Ut, hidden) &&
                   kq_x16s_ok(Dt, inner);
    /* flag 256 (8-bit rows: the low planes 0) on KQ_MXX: the high planes alone
     * (kq_mxx_rows1, the same bits) */
    static int one_on = -1;     /* RECLANK_X8_ONE=0: both planes all the same (a check) */
    if (one_on < 0) {
        const char *v = getenv("RECLANK_X8_ONE");
        one_on = v ? atoi(v) != 0 : 1;
    }
    const int one = one_on && x2 && (flags & 256) && Gt == KQ_MXX && Ut == KQ_MXX && Dt == KQ_MXX;
    const size_t rbd = kq_row_bytes(Dt, inner);
    const int ne = experts, nph = hidden / 32, npi = inner / 32;
    int P = t * k;
    uint8_t *p = (uint8_t *)(intptr_t)scr[part];
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P * 4);
    int *exp_of = (int *)ma_take(&p, (size_t)P * 4);
    int *nused_p = (int *)ma_take(&p, 64);
    int *pair_q = (int *)ma_take(&p, (size_t)P * 4);
    int8_t *tq = (int8_t *)ma_take(&p, (size_t)t * hidden);
    int8_t *tq2 = (int8_t *)ma_take(&p, (size_t)t * hidden);
    float *ts = (float *)ma_take(&p, (size_t)t * nph * 4);
    float *tm = (float *)ma_take(&p, (size_t)t * nph * 8);
    int8_t *xq = (int8_t *)ma_take(&p, (size_t)P * hidden);
    int8_t *xq2 = (int8_t *)ma_take(&p, (size_t)P * hidden);
    float *xs = (float *)ma_take(&p, (size_t)P * nph * 4);
    float *xm = (float *)ma_take(&p, (size_t)P * nph * 8);
    int32_t *xn = (int32_t *)ma_take(&p, (size_t)P * (hidden / 16) * 8);
    float *act = (float *)ma_take(&p, (size_t)P * 2 * ni * 4);
    float *arow = (float *)ma_take(&p, (size_t)P * inner * 4);
    int8_t *aq = (int8_t *)ma_take(&p, (size_t)P * inner);
    int8_t *aq2 = (int8_t *)ma_take(&p, (size_t)P * inner);
    float *as = (float *)ma_take(&p, (size_t)P * npi * 4);
    float *am = (float *)ma_take(&p, (size_t)P * npi * 8);
    int32_t *an = (int32_t *)ma_take(&p, (size_t)P * (inner / 16) * 8);
    float *de = (float *)ma_take(&p, (size_t)P * nd * 4);
    /* (units of 32 rows for KQ_MXX, kq_mxx_rows32 feeding two groups from each
     * broadcast of x, were slower on a prompt: 16 here) */
    const int ngu = Gt == KQ_NVX || Gt == KQ_Q4X || Gt == KQ_MXX ? 16 : 4;
    const int ndn = Dt == KQ_NVX || Dt == KQ_Q4X || Dt == KQ_MXX ? 16 : kq_ndn_rows();
    /* flag 128 (DeepSeek V4.1): the act is silu(min(g, 10)) clamp(u, -10, 10)
     * times the weight of the pair (the reference's Expert), and the down rows
     * are summed as they are */
    const int dsv = (flags & 128) != 0;
    /* the sums of xn and an of a pair: one plane, or with x2 both */
    const size_t xns = (size_t)(hidden / 16) * (x2 && ngu >= 16 ? 2 : 1);
    const size_t ans = (size_t)(inner / 16) * (x2 && ndn >= 16 ? 2 : 1);
    int ch;
    const int dyn = kq_part_dyn(&ch) && t >= 32;
    int64_t *cnt_b = &kq_pdyn[part][0], *cnt_e = &kq_pdyn[part][8];
    int lo, hi;
    /* A: the int8 rows of the tokens (their groups of 32 shared by the threads
     * but the first) and the sort of the pairs (moe_sort_pairs: the first
     * thread); then the rows of the pairs (shared), in the scratch of the
     * part (kq_moe_small_body). One thread for all of it took 8 us a token. */
    if (lt == 0 || team == 1) {
        for (int e = 0; e <= ne; ++e) {
            cnt[e] = 0;
        }
        for (int q = 0; q < P; ++q) {
            if (ids[q] >= 0) {
                cnt[ids[q]]++;
            }
        }
        int a = 0, nu = 0;
        for (int e = 0; e < ne; ++e) {
            start[e] = a;
            a += cnt[e];
            if (cnt[e] > 0) {
                used[nu++] = e;
            }
            cnt[e] = 0;
        }
        start[ne] = a;
        for (int q = 0; q < P; ++q) {
            int e = ids[q];
            if (e < 0) {
                pair_of[q] = -1;
                continue;
            }
            int pos = start[e] + cnt[e]++;
            pair_tok[pos] = q / k;
            pair_of[q] = pos;
            exp_of[pos] = e;
            pair_q[pos] = q;
        }
        *nused_p = nu;
        __atomic_store_n(cnt_b, 0, __ATOMIC_RELAXED);
        __atomic_store_n(cnt_e, 0, __ATOMIC_RELAXED);
    }
    if (lt > 0 || team == 1) {
        int qt = team > 1 ? team - 1 : 1, ql = team > 1 ? lt - 1 : 0;
        kq_part_share(t * nph, ql, qt, &lo, &hi);
        for (int x = lo; x < hi; ++x) {
            int j = x / nph, g = x % nph;
            if (x2) {
                kq_quant_part16(hf + (size_t)j * hidden, g, tq + (size_t)j * hidden, tq2 + (size_t)j * hidden,
                                ts + (size_t)j * nph, tm + (size_t)j * 2 * nph);
                if (flags & 256) {
                    kq_round8(tq + (size_t)j * hidden + 32 * g, tq2 + (size_t)j * hidden + 32 * g, 32);
                }
            } else {
                kq_quant_part(hf + (size_t)j * hidden, g, tq + (size_t)j * hidden, ts + (size_t)j * nph,
                              tm + (size_t)j * 2 * nph);
            }
        }
    }
    KQ_PP(0);
    kq_part_barrier(part, team);
    KQ_PP(1);
    kq_part_share(start[ne], lt, team, &lo, &hi);
    for (int q = lo; q < hi; ++q) {
        int j = pair_tok[q];
        memcpy(xq + (size_t)q * hidden, tq + (size_t)j * hidden, (size_t)hidden);
        memcpy(xs + (size_t)q * nph, ts + (size_t)j * nph, (size_t)nph * 4);
        memcpy(xm + (size_t)q * 2 * nph, tm + (size_t)j * 2 * nph, (size_t)nph * 8);
        if (x2 && !one) {
            memcpy(xq2 + (size_t)q * hidden, tq2 + (size_t)j * hidden, (size_t)hidden);
        }
        if (ngu >= 16) {
            kq_x16_xsum(Gt, xq + (size_t)q * hidden, hidden, xn + (size_t)q * xns);
            if (x2 && !one) {
                kq_x16_xsum(Gt, xq2 + (size_t)q * hidden, hidden, xn + (size_t)q * xns + hidden / 16);
            }
        }
    }
    KQ_PP(2);
    kq_part_barrier(part, team);
    KQ_PP(3);
    const int nu = *nused_p;
    P = start[ne];
    /* B: the gate and the up rows of the part */
    const int per_gu = 2 * ni / ngu;
    /* unit qq of the expert of used[ui] */
#define KQ_GU_UNIT(ui, qq)                                                                         \
    do {                                                                                           \
        int e = used[ui], s0 = start[e], n = start[e + 1] - s0;                                    \
        int rr = (int)(qq) * ngu, isup = rr >= ni, r = rr % ni;                                    \
        const uint8_t *w = isup ? Uw + (size_t)e * Us + (size_t)(Ur + r) * rbu                     \
                                : Gw + (size_t)e * Gs + (size_t)(Gr + r) * rbg;                    \
        kq_rows_n2(w, isup ? rbu : rbg, isup ? Ut : Gt, ngu, hidden, xq + (size_t)s0 * hidden,     \
                   xs + (size_t)s0 * nph, xm + (size_t)s0 * 2 * nph, xn + (size_t)s0 * xns,        \
                   x2 && !one ? xq2 + (size_t)s0 * hidden : NULL, n, act + (size_t)s0 * 2 * ni + rr, \
                   (size_t)2 * ni);                                                                \
    } while (0)
    if (dyn) {
        int64_t u0, u1;
        while (kq_part_grab(cnt_b, (int64_t)nu * per_gu, ch, &u0, &u1)) {
            for (int64_t u = u0; u < u1; ++u) {
                KQ_GU_UNIT((int)(u / per_gu), u % per_gu);
            }
        }
    } else {
        for (int ui = 0; ui < nu; ++ui) {
            int e = used[ui], s0 = start[e], n = start[e + 1] - s0;
            int64_t qa, qb;
            kq_part_units(per_gu, P, lt, team, s0, n, &qa, &qb);
            for (int64_t qq = qa; qq < qb; ++qq) {
                KQ_GU_UNIT(ui, qq);
            }
        }
    }
#undef KQ_GU_UNIT
    KQ_PP(4);
    kq_part_barrier(part, team);
    KQ_PP(5);
    /* C: the act of the columns of the part, into the shared rows */
    kq_part_share(P, lt, team, &lo, &hi);
    for (int q = lo; q < hi; ++q) {
        const float *a = act + (size_t)q * 2 * ni;
        float *o = act2 + (size_t)q * inner + a0;
        if (dsv) {
            float wq = val[pair_q[q]];
            for (int i = 0; i < ni; ++i) {
                float g = fminf(a[i], 10.f), u = fminf(fmaxf(a[ni + i], -10.f), 10.f);
                o[i] = ma_silu(g) * u * wq;
            }
            continue;
        }
        for (int i = 0; i < ni; ++i) {
            float v = a[i];
            o[i] = gelu ? 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) * a[ni + i]
                        : ma_silu(v) * a[ni + i];
        }
    }
    KQ_PP(6);
    #pragma omp barrier
    KQ_PP(7);
    /* D: the whole act of each pair: rotated (RQ types), then int8 */
    for (int q = lo; q < hi; ++q) {
        float *a = arow + (size_t)q * inner;
        memcpy(a, act2 + (size_t)q * inner, (size_t)inner * 4);
        if (kq_moe_rot) {
            for (int g = 0; g < inner / 32; ++g) {
                tq6_rot32(a + 32 * g, 0);
            }
        }
        if (x2) {
            kq_quant_row16(a, inner, aq + (size_t)q * inner, aq2 + (size_t)q * inner, as + (size_t)q * npi,
                           am + (size_t)q * 2 * npi);
            if (flags & 256) {
                kq_round8(aq + (size_t)q * inner, aq2 + (size_t)q * inner, inner);
            }
        } else {
            for (int g = 0; g < npi; ++g) {
                kq_quant_part(a, g, aq + (size_t)q * inner, as + (size_t)q * npi, am + (size_t)q * 2 * npi);
            }
        }
        if (ndn >= 16) {
            kq_x16_xsum(Dt, aq + (size_t)q * inner, inner, an + (size_t)q * ans);
            if (x2 && !one) {
                kq_x16_xsum(Dt, aq2 + (size_t)q * inner, inner, an + (size_t)q * ans + inner / 16);
            }
        }
    }
    KQ_PP(8);
    kq_part_barrier(part, team);
    KQ_PP(9);
    /* E: the down rows of the part */
    const int per_dn = nd / ndn;
#define KQ_DN_UNIT(ui, qq)                                                                         \
    do {                                                                                           \
        int e = used[ui], s0 = start[e], n = start[e + 1] - s0;                                    \
        int r = (int)(qq) * ndn;                                                                   \
        const uint8_t *w = Dw + (size_t)e * Ds + (size_t)(Dr + r) * rbd;                           \
        kq_rows_n2(w, rbd, Dt, ndn, inner, aq + (size_t)s0 * inner, as + (size_t)s0 * npi,         \
                   am + (size_t)s0 * 2 * npi, an + (size_t)s0 * ans,                               \
                   x2 && !one ? aq2 + (size_t)s0 * inner : NULL, n, de + (size_t)s0 * nd + r, (size_t)nd); \
    } while (0)
    if (dyn) {
        int64_t u0, u1;
        while (kq_part_grab(cnt_e, (int64_t)nu * per_dn, ch, &u0, &u1)) {
            for (int64_t u = u0; u < u1; ++u) {
                KQ_DN_UNIT((int)(u / per_dn), u % per_dn);
            }
        }
    } else {
        for (int ui = 0; ui < nu; ++ui) {
            int e = used[ui], s0 = start[e], n = start[e + 1] - s0;
            int64_t qa, qb;
            kq_part_units(per_dn, P, lt, team, s0, n, &qa, &qb);
            for (int64_t qq = qa; qq < qb; ++qq) {
                KQ_DN_UNIT(ui, qq);
            }
        }
    }
#undef KQ_DN_UNIT
    KQ_PP(10);
    kq_part_barrier(part, team);
    KQ_PP(11);
    /* F: the rows c0 to c0 + nd of out: the sum of moe_combine (from 0, the
     * experts in the order of the router) */
    if (flags & 64) {
        /* the rows form (GP_MOE_COMBINE): the de row of each cold pair, the
         * columns c0 to c0 + nd of row j k + sl of out */
        kq_part_share(t * k, lt, team, &lo, &hi);
        for (int q = lo; q < hi; ++q) {
            if (pair_of[q] >= 0) {
                memcpy(out + (size_t)q * hidden + c0, de + (size_t)pair_of[q] * nd, (size_t)nd * 4);
            }
        }
        KQ_PP(12);
        return;
    }
    kq_part_share(t * nd, lt, team, &lo, &hi);
    for (int x = lo; x < hi; ++x) {
        int j = x / nd, c = x % nd;
        float o = 0.f;
        for (int sl = 0; sl < k; ++sl) {
            int pos = pair_of[j * k + sl];
            if (pos >= 0) {
                o += dsv ? de[(size_t)pos * nd + c] : val[j * k + sl] * de[(size_t)pos * nd + c];
            }
        }
        out[(size_t)j * hidden + c0 + c] = o;
    }
    KQ_PP(12);
    /* (the end of the region of GP_RUN_PARTS is the last barrier) */
    (void)exp_of;
}
