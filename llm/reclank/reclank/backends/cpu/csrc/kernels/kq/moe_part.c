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
 * rows of the part: 0).
 *
 * The KQ_E3 experts (kernels/kq/exl3.c) take the same phases: their rows are
 * the outputs, the columns of the trellis. The outputs of a part (128-aligned:
 * the transform of the outputs mixes blocks of 128) are a run of tile columns
 * in each row of tiles, so a copy of them (moe_parts.py) is the block of a
 * k x n_part matrix: those tiles, all of suh, its part of svh; a full block
 * (the experts of the weightkeeper on each node) gives the first output as
 * the first row. The n of the block in memory comes from the bytes of an
 * expert. The rows of a part of gate and up (gu) are a range of the outputs
 * of both, gate then up (2 inner; on two nodes all of gate and all of up:
 * the even split that 128-aligned rows of inner alone cannot give, 384 and
 * 256 of 640). Each part makes xh of every pair for the matrices it has
 * (each expert has its own suh: A2 is per pair, not per token), its
 * columns (B), their output transform into act2 (C: P x 2 inner, shared),
 * then the act of the whole row and xh of it for down (D), its down
 * columns (E) and their output transform into de (E2), then F. */

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
    KQ_PT(P * hidden * 4);                                         /* xf (KQ_BF12: the float rows) */
#undef KQ_PT
    return a + 64;
}

/* The scratch of the KQ_E3 experts in parts (the pairs as above, then the
 * rows of each pair: R = 2 rows with int16 x, the two planes). */
static size_t kq_e3_part_scratch_bytes(int t, int k, int experts, int hidden, int inner, int ni, int nd)
{
    size_t P = (size_t)t * k, ne = (size_t)experts + 1, nh = hidden / 128, nib = inner / 128, R = 2;
    size_t a = 0;
#define KQ_PT(n) a += ((size_t)(n) + 63) & ~(size_t)63
    KQ_PT(ne * 4); KQ_PT(ne * 4); KQ_PT(ne * 4);
    KQ_PT(P * 4); KQ_PT(P * 4); KQ_PT(P * 4); KQ_PT(64);
    KQ_PT(P * 4);
    for (int m = 0; m < 2; ++m) {
        KQ_PT(R * P * hidden); KQ_PT(R * P * nh * 4); KQ_PT(R * P * nh * 4);   /* xh of gate, up */
    }
    KQ_PT(R * P * ni * 4);                                                      /* y' of the part */
    KQ_PT(P * inner * 4);                                                       /* the act */
    KQ_PT(R * P * inner); KQ_PT(R * P * nib * 4); KQ_PT(R * P * nib * 4);       /* xh of down */
    KQ_PT(R * P * nd * 4);                                                      /* y' of down */
    KQ_PT(P * nd * 4);                                                          /* de */
#undef KQ_PT
    return a + 64;
}

size_t rc_kq_moe_part_scratch(int t, int k, int experts, int hidden, int inner, int ni, int nd)
{
    size_t a = kq_moe_part_scratch_bytes(t, k, experts, hidden, inner, ni, nd);
    size_t b = hidden % 128 == 0 && inner % 128 == 0 ? kq_e3_part_scratch_bytes(t, k, experts, hidden, inner, ni, nd)
                                                    : 0;
    return a > b ? a : b;
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

/* RECLANK_PART_FUSE=0: the phases as before (fuse of kq_moe_part_body off; a check of the bits). */
static int kq_part_fuse_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_PART_FUSE");
        on = v ? atoi(v) != 0 : 1;
    }
    return on;
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
 * bits. RECLANK_PART_DYN_MIN: the fewest tokens for it (32; GLM53GPU sets 3:
 * GLM-5.3's experts in a verify group of 3, 5, 9 tokens 4%, 11%, 16% faster
 * than the static split; a token alone 15% slower: the units of 8 experts,
 * each of one row, all of a cost, which the static split balances). */
static int kq_part_dyn(int *chunk, int *tmin)
{
    static int on = -1, ch = 16, mn = 32;
    if (on < 0) {
        const char *v = getenv("RECLANK_PART_DYN"), *c = getenv("RECLANK_PART_CHUNK");
        const char *m = getenv("RECLANK_PART_DYN_MIN");
        ch = c && atoi(c) > 0 ? atoi(c) : 16;
        mn = m && atoi(m) > 0 ? atoi(m) : 32;
        on = v ? atoi(v) != 0 : 1;
    }
    *chunk = ch;
    *tmin = mn;
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

/* The act of one column (phase C's operations): flag 128 silu(min(g, 10)) clamp(u, -10, 10) times the
 * weight of the pair, flag 512 without the weight, else gelu or silu of g times u. */
static inline float kq_part_act(float g, float u, int dsv, int clamp, int gelu, float wq)
{
    if (dsv) {
        float gg = fminf(g, 10.f), uu = fminf(fmaxf(u, -10.f), 10.f);
        return ma_silu(gg) * uu * wq;
    }
    if (clamp) {
        return ma_silu(fminf(g, 10.f)) * fminf(fmaxf(u, -10.f), 10.f);
    }
    return gelu ? 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * g * g * g))) * u
                : ma_silu(g) * u;
}

/* flag 4096 (DeepSeek V4.1 steps, RECLANK_DS_SPEC): a prefetch for the next layer's cold experts, not
 * their rows. The CPU waits for the GPU's signal of each layer (its attention) with nothing to do; this
 * task runs right after the last layer's, on its input hf (the FFN input of the layer before, in the
 * host buffer of the signals): the router of the next layer on it (gate: bf16 [experts, hidden], bias:
 * the selection's), sqrt(softplus) + bias, the top k, those that map (the GPU's slots, its host copy)
 * does not hold, at most RECLANK_DS_SPEC_N (6); then each thread reads the rows it would take of them
 * in the real task (the split of kq_part_units for one pair each, fuse's units of 16 rows of gate and
 * up, then the down rows), so the task finds them in its caches (the same task, its experts from the
 * caches: 1.27x on a real step). The reads stop when the next layer's signal comes (cur, next: the flag
 * the last task wrote and the next task's flag in; with next >= cur the GPU has the next task's inputs
 * out): a prefetch of all of them took 477 us, the wait for the signal 309, and held the task back (a
 * step 34.1 -> 39.1 ms). The likeliest experts first, the gate and up rows of all, then the down rows.
 * The task after compares its experts with the guess (rc_ds_spec_stats). */
static float kq_spec_sel[1024];
static int32_t kq_spec_list[16], kq_spec_rank[16];
static int kq_spec_n = -1;
static int64_t kq_spec_st[4];              /* the guessed experts, the task's, the hits, the tasks */
static volatile uint64_t kq_spec_sink;

void rc_ds_spec_stats(int64_t *out, int reset)
{
    for (int i = 0; i < 4; ++i) {
        out[i] = kq_spec_st[i];
    }
    if (reset) {
        memset(kq_spec_st, 0, sizeof(kq_spec_st));
    }
}

static inline uint64_t kq_spec_touch(const uint8_t *p, size_t n)
{
    uint64_t s = 0;
    for (size_t o = 0; o < n; o += 64) {
        s += *(const volatile uint64_t *)(p + o);
    }
    return s;
}

static void kq_moe_spec_body(const float *hf, int k, int experts, const int64_t *pm_all, int hidden, int inner,
                             const int32_t *gu, const int32_t *dn, int nparts, int team, const uint16_t *gate,
                             const float *bias, const int32_t *map, const int64_t *cur, const int64_t *next)
{
    static int maxn = -1, dn_on = 1;
    if (maxn < 0) {
        const char *v = getenv("RECLANK_DS_SPEC_N"), *d = getenv("RECLANK_DS_SPEC_DN");
        maxn = v && atoi(v) >= 0 ? atoi(v) : 6;
        maxn = maxn < 16 ? maxn : 16;
        dn_on = d ? atoi(d) != 0 : 1;
    }
    const int tid = omp_get_thread_num(), nth = nparts * team;
    if (experts > 1024 || k > 16 || omp_get_num_threads() != nth) {
        return;
    }
    int lo, hi;
    kq_part_share(experts, tid, nth, &lo, &hi);
    for (int e = lo; e < hi; ++e) {
        const uint16_t *w = gate + (size_t)e * hidden;
        float a = 0.f;
        #pragma omp simd reduction(+ : a)
        for (int c = 0; c < hidden; ++c) {
            uint32_t u = (uint32_t)w[c] << 16;
            float f;
            memcpy(&f, &u, 4);
            a += f * hf[c];
        }
        float sp = a > 20.f ? a : log1pf(expf(a));
        kq_spec_sel[e] = sqrtf(sp) + bias[e];
    }
    #pragma omp barrier
    if (tid == 0) {
        /* the top k (ties: the lower id), the cold ones in id order (the order of the task's used) */
        int top[16], n = 0;
        for (int j = 0; j < k; ++j) {
            int b = -1;
            for (int e = 0; e < experts; ++e) {
                int taken = 0;
                for (int q = 0; q < j; ++q) {
                    taken |= top[q] == e;
                }
                if (!taken && (b < 0 || kq_spec_sel[e] > kq_spec_sel[b])) {
                    b = e;
                }
            }
            top[j] = b;
        }
        for (int j = 0; j < k; ++j) {
            if (top[j] >= 0 && map[top[j]] < 0 && n < maxn) {
                kq_spec_rank[n] = top[j];
                kq_spec_list[n++] = top[j];
            }
        }
        for (int a = 1; a < n; ++a) {
            for (int b = a; b > 0 && kq_spec_list[b - 1] > kq_spec_list[b]; --b) {
                int x = kq_spec_list[b];
                kq_spec_list[b] = kq_spec_list[b - 1];
                kq_spec_list[b - 1] = x;
            }
        }
        kq_spec_n = n;
        kq_spec_st[0] += n;
    }
    #pragma omp barrier
    const int n = kq_spec_n, part = tid / team, lt = tid % team;
    const int ni = gu[2 * part + 1], nd = dn[2 * part + 1];
    const int64_t *pm = pm_all + 12 * part;
    const uint8_t *Gw = (const uint8_t *)(intptr_t)pm[0], *Uw = (const uint8_t *)(intptr_t)pm[4];
    const uint8_t *Dw = (const uint8_t *)(intptr_t)pm[8];
    const size_t Gs = (size_t)pm[2], Us = (size_t)pm[6], Ds = (size_t)pm[10];
    const int Gr = (int)pm[3], Ur = (int)pm[7], Dr = (int)pm[11];
    const size_t rbg = kq_row_bytes((int)pm[1], hidden), rbu = kq_row_bytes((int)pm[5], hidden);
    const size_t rbd = kq_row_bytes((int)pm[9], inner);
    /* the signal of the next task: then the reads stop (checked each unit of 16 rows) */
    const int64_t want = cur != NULL ? __atomic_load_n(cur, __ATOMIC_ACQUIRE) : 0;
#define KQ_SPEC_STOP() (next != NULL && __atomic_load_n(next, __ATOMIC_ACQUIRE) >= want)
    uint64_t sink = 0;
    for (int m = 0; m < 2 && (m == 0 || dn_on); ++m) {
        for (int r = 0; r < n; ++r) {
            /* (the rank's expert; the split of the task: its place in id order) */
            const int e = kq_spec_rank[r];
            int s0 = 0;
            while (kq_spec_list[s0] != e) {
                ++s0;
            }
            int64_t qa, qb;
            kq_part_units(m == 0 ? ni / 16 : nd / 16, n, lt, team, s0, 1, &qa, &qb);
            for (int64_t q = qa; q < qb; ++q) {
                if (KQ_SPEC_STOP()) {
                    goto done;
                }
                if (m == 0) {
                    sink += kq_spec_touch(Gw + (size_t)e * Gs + (size_t)(Gr + 16 * q) * rbg, 16 * rbg);
                    sink += kq_spec_touch(Uw + (size_t)e * Us + (size_t)(Ur + 16 * q) * rbu, 16 * rbu);
                } else {
                    sink += kq_spec_touch(Dw + (size_t)e * Ds + (size_t)(Dr + 16 * q) * rbd, 16 * rbd);
                }
            }
        }
    }
done:
#undef KQ_SPEC_STOP
    kq_spec_sink += sink;
}

static void kq_moe_part_body(const float *hf, const int32_t *ids, const float *val, int t, int k0,
                             int experts, const int64_t *pm_all, int hidden, int inner, const int32_t *gu,
                             const int32_t *dn, float *act2, float *out, int nparts, int team,
                             const int32_t *kcount, int flags, const int64_t *scr, const uint16_t *spec_gate,
                             const float *spec_bias, const int32_t *spec_map, const int64_t *spec_cur,
                             const int64_t *spec_next)
{
    if (flags & 4096) {
        kq_moe_spec_body(hf, k0, experts, pm_all, hidden, inner, gu, dn, nparts, team, spec_gate, spec_bias,
                         spec_map, spec_cur, spec_next);
        return;
    }
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
    /* k: the pairs of a token, k0 or (kcount: a step's cold experts, GP_HOT_SPLIT) the count in pinned
     * memory that the GPU writes. The first thread of the part reads it once (phase A) and the others
     * take that value after A's barrier (nused_p[1]): two reads could differ (the GPU's next signal,
     * an AWAIT that did not wait), and threads with other counts read pair_of past what the sort
     * wrote (a general protection fault in gp_step). The scratch is laid out for k0 (the size of
     * rc_kq_moe_part_scratch), so every thread places the arrays alike whatever the count. */
    int k = k0;
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
    int P = t * k0;
    uint8_t *p = (uint8_t *)(intptr_t)scr[part];
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P * 4);
    int *exp_of = (int *)ma_take(&p, (size_t)P * 4);
    int *nused_p = (int *)ma_take(&p, 64);
    int *pair_q = (int *)ma_take(&p, (size_t)P * 4);
    /* KQ_E3: K of each matrix, the n of its blocks in memory (Gs = k n K / 8
     * + 2 (k + n)), the rows of each pair (R: 2 with int16 x) */
    const int e3 = kq_e3_bits(Gt) != 0;
    const int Kg = e3 ? kq_e3_bits(Gt) : 0, Ku = e3 ? kq_e3_bits(Ut) : 0, Kd = e3 ? kq_e3_bits(Dt) : 0;
    const int nG = e3 ? (int)((Gs - 2 * (size_t)hidden) / ((size_t)hidden * Kg / 8 + 2)) : 0;
    const int nU = e3 ? (int)((Us - 2 * (size_t)hidden) / ((size_t)hidden * Ku / 8 + 2)) : 0;
    const int nD = e3 ? (int)((Ds - 2 * (size_t)inner) / ((size_t)inner * Kd / 8 + 2)) : 0;
    const int R = e3 && (kq_e3_x16() || (flags & 32)) ? 2 : 1;
    const int nh = hidden / 128, nib = inner / 128;
    /* KQ_E3: the outputs a0 to a0 + ni of gate then up (2 inner): the gate
     * columns e3g0 to e3g1, the up columns e3u0 to e3u1; output g of the range
     * is column g - e3g0 + Gr of gate (g < inner) or g - inner - e3u0 + Ur of up
     * (the names: KQ_E3_COL runs inside the loops of the units, whose u0 is a
     * chunk) */
    const int e3g0 = a0 < inner ? a0 : inner, e3g1 = a0 + ni < inner ? a0 + ni : inner;
    const int e3u0 = (a0 > inner ? a0 : inner) - inner, e3u1 = (a0 + ni > inner ? a0 + ni : inner) - inner;
    const int has[2] = {e3g1 > e3g0, e3u1 > e3u0};
#define KQ_E3_COL(g) ((g) < inner ? (g) - e3g0 + Gr : (g) - inner - e3u0 + Ur)
    if (e3 && (!Ku || !Kd || a0 % 128 || ni % 128 || nd % 128 || Gr % 128 || Ur % 128 || Dr % 128 ||
               kq_e3_bytes(Kg, hidden, nG) != Gs || kq_e3_bytes(Ku, hidden, nU) != Us ||
               kq_e3_bytes(Kd, inner, nD) != Ds)) {
        #pragma omp single
        fprintf(stderr, "kq_moe_part_body: KQ_E3 parts need 128-aligned rows and whole blocks\n");
        return;
    }
    int8_t *e3x[2] = {NULL, NULL}, *e3a = NULL;
    float *e3s[2] = {NULL, NULL}, *e3y[2] = {NULL, NULL}, *e3as = NULL, *e3yd = NULL;
    int32_t *e3n[2] = {NULL, NULL}, *e3an = NULL;
    if (e3) {
        for (int m = 0; m < 2; ++m) {
            e3x[m] = (int8_t *)ma_take(&p, (size_t)R * P * hidden);
            e3s[m] = (float *)ma_take(&p, (size_t)R * P * nh * 4);
            e3n[m] = (int32_t *)ma_take(&p, (size_t)R * P * nh * 4);
        }
        e3y[0] = (float *)ma_take(&p, (size_t)R * P * ni * 4);
    }
    float *e3act = e3 ? (float *)ma_take(&p, (size_t)P * inner * 4) : NULL;
    if (e3) {
        e3a = (int8_t *)ma_take(&p, (size_t)R * P * inner);
        e3as = (float *)ma_take(&p, (size_t)R * P * nib * 4);
        e3an = (int32_t *)ma_take(&p, (size_t)R * P * nib * 4);
        e3yd = (float *)ma_take(&p, (size_t)R * P * nd * 4);
    }
    float *e3de = e3 ? (float *)ma_take(&p, (size_t)P * nd * 4) : NULL;
    /* the rows of the other types (none for KQ_E3: act and de are its own) */
    int8_t *tq = NULL, *tq2 = NULL, *xq = NULL, *xq2 = NULL, *aq = NULL, *aq2 = NULL;
    float *ts = NULL, *tm = NULL, *xs = NULL, *xm = NULL, *arow = NULL, *as = NULL, *am = NULL;
    float *act = e3act, *de = e3de, *xf = NULL;
    int32_t *xn = NULL, *an = NULL;
    if (!e3) {
        tq = (int8_t *)ma_take(&p, (size_t)t * hidden);
        tq2 = (int8_t *)ma_take(&p, (size_t)t * hidden);
        ts = (float *)ma_take(&p, (size_t)t * nph * 4);
        tm = (float *)ma_take(&p, (size_t)t * nph * 8);
        xq = (int8_t *)ma_take(&p, (size_t)P * hidden);
        xq2 = (int8_t *)ma_take(&p, (size_t)P * hidden);
        xs = (float *)ma_take(&p, (size_t)P * nph * 4);
        xm = (float *)ma_take(&p, (size_t)P * nph * 8);
        xn = (int32_t *)ma_take(&p, (size_t)P * (hidden / 16) * 8);
        act = (float *)ma_take(&p, (size_t)P * 2 * ni * 4);
        arow = (float *)ma_take(&p, (size_t)P * inner * 4);
        aq = (int8_t *)ma_take(&p, (size_t)P * inner);
        aq2 = (int8_t *)ma_take(&p, (size_t)P * inner);
        as = (float *)ma_take(&p, (size_t)P * npi * 4);
        am = (float *)ma_take(&p, (size_t)P * npi * 8);
        an = (int32_t *)ma_take(&p, (size_t)P * (inner / 16) * 8);
        de = (float *)ma_take(&p, (size_t)P * nd * 4);
        /* KQ_BF12 (the routed experts of the bfloat16 checkpoint in the
         * weightkeeper, reclank/models/qwen/keep.py) reads the float32 rows: x of
         * the pairs (xf) and the act (arow), kq_rows_bf12_f as kq_moe_body */
        xf = (float *)ma_take(&p, (size_t)P * hidden * 4);
    }
    const int gbf = Gt == KQ_BF12, ubf = Ut == KQ_BF12, dbf = Dt == KQ_BF12;
    /* (units of 32 rows for KQ_MXX, kq_mxx_rows32 feeding two groups from each
     * broadcast of x, were slower on a prompt: 16 here) */
    /* (KQ_E3: a unit is a tile column, 16 outputs) */
    const int ngu = e3 || Gt == KQ_NVX || Gt == KQ_Q4X || Gt == KQ_MXX ? 16 : 4;
    const int ndn = e3 || Dt == KQ_NVX || Dt == KQ_Q4X || Dt == KQ_MXX ? 16 : kq_ndn_rows();
    /* flag 128 (DeepSeek V4.1): the act is silu(min(g, 10)) clamp(u, -10, 10)
     * times the weight of the pair (the reference's Expert), and the down rows
     * are summed as they are */
    const int dsv = (flags & 128) != 0;
    /* flag 512 (GLM-5.3): the act is silu(min(g, 10)) clamp(u, -10, 10), the
     * weights of the pairs as usual */
    const int clamp = (flags & 512) != 0;
    /* flag 1024 (FP4-Local, docs/FP4LOCAL_PLAN.md): the A16 contract of b12x on KQ_NVX experts: the
     * float rows of x (bfloat16 values), the exact products (kq_nvx_rows_f), g and u to bfloat16 and
     * h = bf16(bf16(g sigma(g)) u) (kq_b12x_act), the raw down sums D2, and each pair's row y =
     * bf16(bf16(D2) bf16(w g_down)), which the sum adds as they are */
    const int b12x = (flags & 1024) != 0 && Gt == KQ_NVX && Ut == KQ_NVX && Dt == KQ_NVX && !e3;
    /* the sums of xn and an of a pair: one plane, or with x2 both */
    const size_t xns = (size_t)(hidden / 16) * (x2 && ngu >= 16 ? 2 : 1);
    const size_t ans = (size_t)(inner / 16) * (x2 && ndn >= 16 ? 2 : 1);
    /* fuse (the integer rows of 16-row groups, not KQ_E3, b12x or KQ_BF12): a unit of B is 16 rows of
     * gate and the same 16 of up, whose act its thread makes at once: no barrier nor phase C, the same
     * operations on each value (the same bits). A DeepSeek V4.1 step 38.7 -> 37.5 ms (C's act took a
     * thread for each pair, 27 us a layer). Tried, slower: D in units of a pair and a group of 32 (all
     * the threads: E 0.4 ms slower a step, its rows written by many cores), and for a token alone its
     * int16 row for every pair (no rows of the pairs nor their barrier: B 2.3 ms slower a step). Not
     * for the groups of the dynamic split (a prompt: 3% slower) */
    int ch, tmin;
    const int dyn = kq_part_dyn(&ch, &tmin) && t >= tmin;
    const int fuse = !e3 && !b12x && !gbf && !ubf && ngu == 16 && ni % 16 == 0 && !dyn && kq_part_fuse_on();
    int64_t *cnt_b = &kq_pdyn[part][0], *cnt_e = &kq_pdyn[part][8];
    int lo, hi;
    /* A: the int8 rows of the tokens (their groups of 32 shared by the threads
     * but the first) and the sort of the pairs (moe_sort_pairs: the first
     * thread); then the rows of the pairs (shared), in the scratch of the
     * part (kq_moe_small_body). One thread for all of it took 8 us a token. */
    if (lt == 0 || team == 1) {
        int kc = kcount != NULL ? *(const volatile int32_t *)kcount : k0;
        kc = kc < 0 ? 0 : kc < k0 ? kc : k0;
        nused_p[1] = kc;
        k = kc;
        P = t * kc;
        for (int e = 0; e <= ne; ++e) {
            cnt[e] = 0;
        }
        /* each id read once and checked (pair_of holds them until the second pass) */
        for (int q = 0; q < P; ++q) {
            int e = ids[q];
            e = e >= 0 && e < ne ? e : -1;
            pair_of[q] = e;
            if (e >= 0) {
                cnt[e]++;
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
            int e = pair_of[q];
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
        if (part == 0 && kcount != NULL && kq_spec_n >= 0) {
            /* (the guess of the prefetch before: flag 4096) */
            int h = 0;
            for (int a = 0; a < nu; ++a) {
                for (int b = 0; b < kq_spec_n; ++b) {
                    h += used[a] == kq_spec_list[b];
                }
            }
            kq_spec_st[1] += nu;
            kq_spec_st[2] += h;
            kq_spec_st[3] += 1;
            kq_spec_n = -1;
        }
        __atomic_store_n(cnt_b, 0, __ATOMIC_RELAXED);
        __atomic_store_n(cnt_e, 0, __ATOMIC_RELAXED);
    }
    if ((lt > 0 || team == 1) && !e3 && !b12x) {
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
    k = nused_p[1];         /* (the first thread's count: see k above) */
    KQ_PP(1);
    /* the block of expert e of matrix m (0 gate, 1 up, 2 down; KQ_E3) */
#define KQ_E3_BLK(m, e) ((m) == 0 ? Gw + (size_t)(e) * Gs : (m) == 1 ? Uw + (size_t)(e) * Us : Dw + (size_t)(e) * Ds)
    if (e3) {
        /* KQ_E3: xh of each pair for gate and up (the suh of its expert), the
         * rows R pr and R pr + 1 of the pair */
        kq_part_share(start[ne] * 2 * nh, lt, team, &lo, &hi);
        for (int x = lo; x < hi; ++x) {
            int pr = x / (2 * nh), m = (x / nh) % 2, b = x % nh;
            if (!has[m]) {
                continue;
            }
            const uint8_t *suh = KQ_E3_BLK(m, exp_of[pr]) + (size_t)hidden * (m ? nU * Ku : nG * Kg) / 8;
            const float *xr = hf + (size_t)pair_tok[pr] * hidden;
            const size_t r0 = (size_t)R * pr;
            if (R == 2) {
                kq_e3_had_in16(xr, suh, b, e3x[m] + r0 * hidden, e3s[m] + r0 * nh, e3n[m] + r0 * nh,
                               e3x[m] + (r0 + 1) * hidden, e3s[m] + (r0 + 1) * nh, e3n[m] + (r0 + 1) * nh);
            } else {
                kq_e3_had_in(xr, suh, b, e3x[m] + r0 * hidden, e3s[m] + r0 * nh, e3n[m] + r0 * nh);
            }
        }
        hi = lo;        /* (no rows of the other types) */
    } else {
        kq_part_share(start[ne], lt, team, &lo, &hi);
    }
    for (int q = lo; q < hi && b12x; ++q) {
        const float *hr = hf + (size_t)pair_tok[q] * hidden;
        float *xr = xf + (size_t)q * hidden;
        for (int i = 0; i < hidden; ++i) {
            xr[i] = kq_bf16r(hr[i]);
        }
    }
    for (int q = lo; q < hi && !b12x; ++q) {
        int j = pair_tok[q];
        memcpy(xq + (size_t)q * hidden, tq + (size_t)j * hidden, (size_t)hidden);
        memcpy(xs + (size_t)q * nph, ts + (size_t)j * nph, (size_t)nph * 4);
        memcpy(xm + (size_t)q * 2 * nph, tm + (size_t)j * 2 * nph, (size_t)nph * 8);
        if (gbf || ubf) {
            memcpy(xf + (size_t)q * hidden, hf + (size_t)j * hidden, (size_t)hidden * 4);
        }
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
    const int per_gu = (e3 || fuse ? ni : 2 * ni) / ngu;     /* (KQ_E3, fuse: ni outputs of gate and up together) */
    /* unit qq of the expert of used[ui] */
#define KQ_GU_UNIT(ui, qq)                                                                         \
    do {                                                                                           \
        int e = used[ui], s0 = start[e], n = start[e + 1] - s0;                                    \
        int rr = (int)(qq) * ngu, isup = rr >= ni, r = rr % ni;                                    \
        if (e3) {                                                                                  \
            /* the tile column of output rr of the part; yp from rr - col, so the column     \
             * lands at rr (the outputs of the part: e3y rows of ni) */                      \
            int g_ = a0 + rr, m3 = g_ >= inner, col = KQ_E3_COL(g_);                              \
            for (int a_ = R * s0; a_ < R * (s0 + n); a_ += 8) { /* (the chunks of kq_e3_moe_body) */ \
                int m_ = R * (s0 + n) - a_ < 8 ? R * (s0 + n) - a_ : 8;                            \
                kq_e3_cols(KQ_E3_BLK(m3, e), m3 ? Ku : Kg, hidden, m3 ? nU : nG, col / 16,         \
                           e3x[m3] + (size_t)a_ * hidden, e3s[m3] + (size_t)a_ * nh, m_,           \
                           e3y[0] + (size_t)a_ * ni + rr - col, (size_t)ni);                     \
            }                                                                                      \
            break;                                                                                 \
        }                                                                                          \
        const uint8_t *w = isup ? Uw + (size_t)e * Us + (size_t)(Ur + r) * rbu                     \
                                : Gw + (size_t)e * Gs + (size_t)(Gr + r) * rbg;                    \
        if (isup ? ubf : gbf) {                                                                    \
            kq_rows_bf12_f(w, isup ? rbu : rbg, ngu, hidden, xf + (size_t)s0 * hidden, (size_t)hidden, \
                           n, act + (size_t)s0 * 2 * ni + rr, (size_t)2 * ni);                     \
            break;                                                                                 \
        }                                                                                          \
        if (b12x) {                                                                                \
            kq_nvx_rows_f(w, hidden, xf + (size_t)s0 * hidden, (size_t)hidden, n,                  \
                          act + (size_t)s0 * 2 * ni + rr, (size_t)2 * ni, 0);                      \
            break;                                                                                 \
        }                                                                                          \
        if (fuse) {                                                                                \
            /* gate rows r .. r + 15, then up's, then the act of those columns of the n pairs */  \
            const int r16 = (int)(qq) * ngu;                                                       \
            const int8_t *xq_ = xq + (size_t)s0 * hidden;                                          \
            const float *xs_ = xs + (size_t)s0 * nph, *xm_ = xm + (size_t)s0 * 2 * nph;            \
            const int8_t *xl_ = x2 && !one ? xq2 + (size_t)s0 * hidden : NULL;                     \
            const int32_t *xn_ = xn + (size_t)s0 * xns;                                            \
            kq_rows_n2(Gw + (size_t)e * Gs + (size_t)(Gr + r16) * rbg, rbg, Gt, ngu, hidden, xq_, xs_, xm_, xn_, \
                       xl_, n, act + (size_t)s0 * 2 * ni + r16, (size_t)2 * ni);                    \
            kq_rows_n2(Uw + (size_t)e * Us + (size_t)(Ur + r16) * rbu, rbu, Ut, ngu, hidden, xq_, xs_, xm_, xn_, \
                       xl_, n, act + (size_t)s0 * 2 * ni + ni + r16, (size_t)2 * ni);               \
            for (int q_ = s0; q_ < s0 + n; ++q_) {                                                 \
                const float *a_ = act + (size_t)q_ * 2 * ni, wq_ = dsv ? val[pair_q[q_]] : 0.f;    \
                float *o_ = act2 + (size_t)q_ * inner + a0;                                        \
                for (int i_ = r16; i_ < r16 + ngu; ++i_) {                                         \
                    o_[i_] = kq_part_act(a_[i_], a_[ni + i_], dsv, clamp, gelu, wq_);              \
                }                                                                                  \
            }                                                                                      \
            break;                                                                                 \
        }                                                                                          \
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
    if (!fuse) {
        kq_part_barrier(part, team);
    }
    KQ_PP(5);
    /* C: the act of the columns of the part, into the shared rows (fuse: made in B) */
    kq_part_share(P, lt, team, &lo, &hi);
    if (e3) {
        /* KQ_E3: y = (H (y' + c)) svh of the outputs of the part (the two
         * planes of a row added), into the row of the pair in act2 (gate,
         * then up) */
        for (int q = lo; q < hi; ++q) {
            const int e = exp_of[q];
            float c[2] = {0.f, 0.f};
            for (int m = 0; m < 2; ++m) {
                if (has[m]) {
                    c[m] = kq_e3_const(e3s[m] + (size_t)R * q * nh, e3n[m] + (size_t)R * q * nh, hidden);
                    if (R == 2) {
                        c[m] += kq_e3_const(e3s[m] + (size_t)(R * q + 1) * nh, e3n[m] + (size_t)(R * q + 1) * nh,
                                            hidden);
                    }
                }
            }
            float *yy = e3y[0] + (size_t)R * q * ni;
            if (R == 2) {
                for (int i = 0; i < ni; ++i) {
                    yy[i] += yy[ni + i];
                }
            }
            for (int j = 0; j < ni; j += 128) {
                const int g = a0 + j, m = g >= inner, col = KQ_E3_COL(g);
                const uint8_t *svh = KQ_E3_BLK(m, e) + (size_t)hidden * (m ? nU * Ku : nG * Kg) / 8 +
                                     2 * (size_t)hidden + 2 * (size_t)col;
                kq_e3_had_out(yy + j, svh, 0, c[m], act2 + (size_t)q * 2 * inner + g);
            }
        }
    }
    for (int q = lo; q < hi && !e3 && !fuse; ++q) {
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
        if (clamp) {
            for (int i = 0; i < ni; ++i) {
                o[i] = ma_silu(fminf(a[i], 10.f)) * fminf(fmaxf(a[ni + i], -10.f), 10.f);
            }
            continue;
        }
        if (b12x) {
            for (int i = 0; i < ni; ++i) {
                o[i] = kq_b12x_act(a[i], a[ni + i]);
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
    /* D: the whole act of each pair: rotated (RQ types), then int8 (KQ_E3:
     * xh of it for down, the suh of its expert) */
    if (e3) {
        /* KQ_E3: units of a pair and a block of 128: the act of its columns,
         * then xh of the block (a pair a thread left 12 of 18 threads waiting) */
        int dlo, dhi;
        kq_part_share(P * nib, lt, team, &dlo, &dhi);
        for (int x = dlo; x < dhi; ++x) {
            const int q = x / nib, b = x % nib;
            const uint8_t *suh = KQ_E3_BLK(2, exp_of[q]) + (size_t)inner * nD * Kd / 8;
            const float *gr = act2 + (size_t)q * 2 * inner;
            float *ar = act + (size_t)q * inner;
            for (int i = 128 * b; i < 128 * (b + 1); ++i) {
                float v = gr[i];
                ar[i] = clamp ? ma_silu(fminf(v, 10.f)) * fminf(fmaxf(gr[inner + i], -10.f), 10.f)
                        : gelu ? 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) *
                                     gr[inner + i]
                               : ma_silu(v) * gr[inner + i];
            }
            const size_t r0 = (size_t)R * q;
            if (R == 2) {
                kq_e3_had_in16(ar, suh, b, e3a + r0 * inner, e3as + r0 * nib, e3an + r0 * nib,
                               e3a + (r0 + 1) * inner, e3as + (r0 + 1) * nib, e3an + (r0 + 1) * nib);
            } else {
                kq_e3_had_in(ar, suh, b, e3a + r0 * inner, e3as + r0 * nib, e3an + r0 * nib);
            }
        }
    }
    for (int q = lo; q < hi && !e3; ++q) {
        float *a = arow + (size_t)q * inner;
        memcpy(a, act2 + (size_t)q * inner, (size_t)inner * 4);
        if (b12x) {
            continue;           /* (the float rows of the act, as they are) */
        }
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
        if (e3) {                                                                                  \
            for (int a_ = R * s0; a_ < R * (s0 + n); a_ += 8) {                                    \
                int m_ = R * (s0 + n) - a_ < 8 ? R * (s0 + n) - a_ : 8;                            \
                kq_e3_cols(KQ_E3_BLK(2, e), Kd, inner, nD, (Dr + r) / 16, e3a + (size_t)a_ * inner,  \
                           e3as + (size_t)a_ * nib, m_, e3yd + (size_t)a_ * nd - Dr, (size_t)nd);  \
            }                                                                                      \
            break;                                                                                 \
        }                                                                                          \
        const uint8_t *w = Dw + (size_t)e * Ds + (size_t)(Dr + r) * rbd;                           \
        if (dbf) {                                                                                 \
            kq_rows_bf12_f(w, rbd, ndn, inner, arow + (size_t)s0 * inner, (size_t)inner, n,        \
                           de + (size_t)s0 * nd + r, (size_t)nd);                                  \
            break;                                                                                 \
        }                                                                                          \
        if (b12x) {                                                                                \
            kq_nvx_rows_f(w, inner, arow + (size_t)s0 * inner, (size_t)inner, n,                   \
                          de + (size_t)s0 * nd + r, (size_t)nd, 1);                                \
            break;                                                                                 \
        }                                                                                          \
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
    if (e3) {
        /* E2 (KQ_E3): de = (H (y' + c)) svh of the down outputs of the part */
        const int nob = nd / 128;
        kq_part_share(P * nob, lt, team, &lo, &hi);
        for (int x = lo; x < hi; ++x) {
            const int q = x / nob, ob = x % nob;
            const uint8_t *svh = KQ_E3_BLK(2, exp_of[q]) + (size_t)inner * nD * Kd / 8 + 2 * (size_t)inner +
                                 2 * (size_t)Dr;
            float *yy = e3yd + (size_t)R * q * nd;
            float c = kq_e3_const(e3as + (size_t)R * q * nib, e3an + (size_t)R * q * nib, inner);
            if (R == 2) {
                for (int i = 128 * ob; i < 128 * (ob + 1); ++i) {
                    yy[i] += yy[nd + i];
                }
                c += kq_e3_const(e3as + (size_t)(R * q + 1) * nib, e3an + (size_t)(R * q + 1) * nib, inner);
            }
            kq_e3_had_out(yy, svh, ob, c, de + (size_t)q * nd);
        }
        kq_part_barrier(part, team);
    }
#undef KQ_E3_BLK
#undef KQ_E3_COL
    /* F: the rows c0 to c0 + nd of out: the sum of moe_combine (from 0, the
     * experts in the order of the router) */
    if (flags & 64) {
        /* the rows form (GP_MOE_COMBINE): the de row of each cold pair, the
         * columns c0 to c0 + nd of row j k + sl of out */
        kq_part_share(t * k, lt, team, &lo, &hi);
        for (int q = lo; q < hi; ++q) {
            if (pair_of[q] >= P) {
                continue;           /* (not a pair of this run: see k above) */
            }
            if (pair_of[q] >= 0 && b12x) {
                /* y = bf16(bf16(D2) bf16(w g_down)): g_down in the header of the expert's group */
                float gd;
                memcpy(&gd, Dw + (size_t)exp_of[pair_of[q]] * Ds + (size_t)Dr * rbd, 4);
                const float ws = kq_bf16r(val[q] * gd), *d = de + (size_t)pair_of[q] * nd;
                float *o = out + (size_t)q * hidden + c0;
                for (int c = 0; c < nd; ++c) {
                    o[c] = kq_bf16r(kq_bf16r(d[c]) * ws);
                }
            } else if (pair_of[q] >= 0) {
                memcpy(out + (size_t)q * hidden + c0, de + (size_t)pair_of[q] * nd, (size_t)nd * 4);
            }
        }
        KQ_PP(12);
        return;
    }
    if (flags & 2048) {
        /* flag 2048 (DeepSeek V4.1, GP_DS_HC_STEP): out is float64 [t, hidden], each token's sum of its
         * pairs in float64 (exact for these few terms in practice), which the GPU adds to the sum of
         * its own pairs in float64: the float of the total does not depend on where an expert ran */
        double *od = (double *)out;
        kq_part_share(t * nd, lt, team, &lo, &hi);
        for (int x = lo; x < hi; ++x) {
            int j = x / nd, c = x % nd;
            double o = 0.0;
            for (int sl = 0; sl < k; ++sl) {
                int pos = pair_of[j * k + sl];
                if (pos >= 0 && pos < P) {
                    o += (double)(dsv ? de[(size_t)pos * nd + c] : val[j * k + sl] * de[(size_t)pos * nd + c]);
                }
            }
            od[(size_t)j * hidden + c0 + c] = o;
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
            pos = pos < P ? pos : -1;
            if (pos >= 0 && b12x) {
                float gd;
                memcpy(&gd, Dw + (size_t)exp_of[pos] * Ds + (size_t)Dr * rbd, 4);
                o += kq_bf16r(kq_bf16r(de[(size_t)pos * nd + c]) * kq_bf16r(val[j * k + sl] * gd));
            } else if (pos >= 0) {
                o += dsv ? de[(size_t)pos * nd + c] : val[j * k + sl] * de[(size_t)pos * nd + c];
            }
        }
        out[(size_t)j * hidden + c0 + c] = o;
    }
    KQ_PP(12);
    /* (the end of the region of GP_RUN_PARTS is the last barrier) */
    (void)exp_of;
}
