/* The parts of a MoE layer that do not depend on the weight format: the
 * sort of the pairs (token, expert) by expert, and the weighted sum of the
 * outputs of the experts. The formats (kquants.c) use them.
 *
 * This file is part of the cops library: bf16_linear.c includes it.
 */

/* Sort the pairs of t tokens by expert, inside a parallel region (one
 * thread does it). ids (t x k) gives the experts of each token. With a
 * shared expert, it is expert number `experts`, with one pair for each
 * token. The arrays have (experts + 2) values (cnt, start, used) and
 * P = t * k (+ t) values (pair_tok, pair_of). start[e] is the first pair of
 * expert e; used has the experts with pairs; pair_tok gives the token of
 * each sorted pair; pair_of gives the sorted pair of each slot (j * k + s,
 * then t * k + j for the shared expert). *nused gets the count of used.
 * start[ne] gets the count of pairs. A negative id is no expert (an expert
 * of a GPU step that the GPU computes): its pair_of is -1. Each id is read
 * once, and one outside 0 .. experts - 1 is no expert too (the ids are in
 * pinned memory that the GPU writes: a second read could differ, and a count
 * or a position past the arrays would follow); pair_of holds the checked ids
 * between the two passes. */
static void moe_sort_pairs(const int32_t *ids, int t, int k, int experts, int shared, int *cnt,
                           int *start, int *used, int *pair_tok, int *pair_of, int *nused)
{
    int ne = experts + shared;
    #pragma omp single
    {
        for (int e = 0; e <= ne; ++e) {
            cnt[e] = 0;
        }
        for (int q = 0; q < t * k; ++q) {
            int e = ids[q];
            e = e >= 0 && e < experts ? e : -1;
            pair_of[q] = e;
            if (e >= 0) {
                cnt[e]++;
            }
        }
        if (shared) {
            cnt[experts] = t;
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
        for (int q = 0; q < t * k; ++q) {
            int e = pair_of[q];
            if (e < 0) {
                pair_of[q] = -1;
                continue;
            }
            int pos = start[e] + cnt[e]++;
            pair_tok[pos] = q / k;
            pair_of[q] = pos;
        }
        if (shared) {
            for (int j = 0; j < t; ++j) {
                int pos = start[experts] + cnt[experts]++;
                pair_tok[pos] = j;
                pair_of[t * k + j] = pos;
            }
        }
        *nused = nu;
    }
}

/* The output of each token: the sum of the outputs of its experts (de, one
 * row of hidden values for each sorted pair) times val, plus the shared
 * expert times sigmoid(shared_logit). In blocks of 64 columns, so one token
 * uses all the threads. */
static void moe_combine(const float *de, const int *pair_of, const float *val,
                        const float *shared_logit, int shared, int t, int k, int hidden,
                        float *out)
{
    int nb = hidden / 64;
    #pragma omp for schedule(static)
    for (int x = 0; x < t * nb; ++x) {
        int j = x / nb, c0 = (x % nb) * 64;
        float *o = out + (size_t)j * hidden;
        /* (no shared_logit: an ungated shared expert, weight 1 (GLM-5.3)) */
        float sw = !shared ? 0.f : shared_logit != NULL ? 1.f / (1.f + expf(-shared_logit[j])) : 1.f;
        const float *sd = shared ? de + (size_t)pair_of[t * k + j] * hidden : NULL;
        for (int c = c0; c < c0 + 64; ++c) {
            o[c] = shared ? sw * sd[c] : 0.f;
        }
        for (int sl = 0; sl < k; ++sl) {
            if (pair_of[j * k + sl] < 0) {
                continue;
            }
            const float *d = de + (size_t)pair_of[j * k + sl] * hidden;
            float w = val[j * k + sl];
            for (int c = c0; c < c0 + 64; ++c) {
                o[c] += w * d[c];
            }
        }
    }
}

/* The rows form of the output (act bit 64; Qwen GPU, GP_MOE_COMBINE): the
 * de row of each pair (slot sl of token j) to row j k + sl of out, with no
 * sum; the GPU adds them with its own in the order of the router. */
static void moe_rows(const float *de, const int *pair_of, int t, int k, int hidden, float *out)
{
    #pragma omp for schedule(static)
    for (int p = 0; p < t * k; ++p) {
        if (pair_of[p] >= 0) {
            memcpy(out + (size_t)p * hidden, de + (size_t)pair_of[p] * hidden, (size_t)hidden * 4);
        }
    }
}

/* GP_MOE_PLAN: split the experts of a group of tokens between the GPU and
 * the CPU (Qwen4GPU, the mixed groups of a prompt; QWEN38_PLAN.md, phase 5).
 *
 *     ip, nreal, t, k, E, slots, desc, cpu_a, cpu_b, gpu_c, tab, gidx, cidx,
 *     ranges, stats
 *
 * ip has the experts of each token (t x k; the rows from *nreal on are
 * padding). slots gives the hot slot of each expert, or -1. desc has 12
 * values for the parts gate, up, down: the device address of hot slot 0 of
 * each part, the bytes of one expert of each part, the host address of
 * expert 0 of each part (the map of the file), the device address of place
 * 0 of the buffer of the copies of each part, and then cap, the places of
 * the buffer (13 values; the host fills them after the program exists).
 * With 16 values and desc[13] > 0, desc[13], desc[14], and desc[15] take
 * the place of cpu_a, cpu_b, and gpu_c: the host measures them as the
 * groups run (QwenGPU.calibrate_mix). desc[16] (19 values): the host
 * address of the table of the pool of HotCache (gpu.ExpertPool: the
 * strides, the count of segments, then the segment addresses of each
 * part; deltanet.cu rc_gpu_set_pool), or
 * 0: a slot of MOE_WARM + i is block i there. desc[17]: the host address of
 * the blocks of the pool for the copies (int32, cap of them; desc[9 .. 11]
 * unused), or 0: the copies go to desc[9 .. 11] + place * bytes.
 * Prefetch (24 values; QwenGPU.mix): desc[19] the host address of the place
 * of each expert that was copied before the plan (int32 E: an index of the
 * blocks of desc[17], or -1), or 0; desc[20] the places those take (the
 * copies go after them); desc[21] (or 0) gidx3: ip with -1 but for the
 * prefetched experts (the GPU runs them after their copies, before the
 * copies of the plan); desc[22] (or 0) the tokens of each expert (int32 E,
 * for the prediction of the next group); stats[5] and [6]: the prefetched
 * experts used and copied, stats[7] the experts the plan would copy with no
 * prefetch (the size of the next prediction: the prefetched experts with a
 * token are all used, so their count grows with the prediction), with the
 * cost of a copy desc[23] (or gpu_c): a prefetch runs while the GPU works.
 * desc[24] (25 values): 0, or the host address of a table of the host
 * address of each expert of each part (3 x E; the experts split over the
 * NUMA nodes, QwenGPU._numa_split). */
#define MOE_WARM (1 << 24)
#define MOE_POOL_K 8            /* deltanet.cu GG_POOL_K */

/* pt: the 3 strides, the count n of segments, then the 3 n segment
 * addresses, part by part (gpumm.ExpertPool.table) */
static inline int64_t moe_pool_addr(const int64_t *pt, int p, int64_t i)
{
    return pt[4 + p * pt[3] + i / MOE_POOL_K] + (i % MOE_POOL_K) * pt[p];
}
/*
 *
 * The hot experts run on the GPU at no cost. Of the other experts that the
 * tokens use, the GPU takes the m with the most tokens: a copy costs about
 * the same time for any count of tokens (gpu_c ns), and the CPU costs
 * cpu_a + cpu_b * tokens ns for each expert. m (at most cap) gives the
 * smallest max(GPU time, CPU time). The outputs:
 *
 *   tab     3 x E device addresses of the experts on the GPU (0 for none),
 *           the tables of GP_KQ_GROUP_MOE;
 *   gidx    ip with -1 for the experts of the CPU (and for the padding);
 *   cidx    ip with -1 for the experts of the GPU;
 *   gidx2   (or null) the copied experts; gidx then has only the hot ones;
 *   ranges  3 E rows (host address, device address, bytes) of the copies,
 *           for GP_FETCH (the rest are zero rows);
 *   stats   the copied experts, their tokens, the CPU experts, their
 *           tokens, the hot experts. */
static int moe_plan_cmp(const void *a, const void *b)
{
    const int *x = (const int *)a, *y = (const int *)b;
    return x[1] != y[1] ? (y[1] > x[1] ? 1 : -1) : x[0] - y[0];   /* most tokens first */
}

static void moe_plan_body(const int32_t *ip, const int64_t *nreal, int t, int k, int E,
                          const int32_t *slots, const int64_t *desc, int64_t cpu_a, int64_t cpu_b,
                          int64_t gpu_c, int64_t *tab, int32_t *gidx, int32_t *cidx,
                          int64_t *ranges, int64_t *stats, int32_t *gidx2)
{
    #pragma omp single
    {
        int cap = (int)desc[12];
        const int64_t *pt = (const int64_t *)(intptr_t)desc[16];
        const int32_t *places = pt != NULL ? (const int32_t *)(intptr_t)desc[17] : NULL;
        const int32_t *pre = places != NULL ? (const int32_t *)(intptr_t)desc[19] : NULL;
        int npre = pre != NULL ? (int)desc[20] : 0;
        int32_t *gidx3 = pre != NULL ? (int32_t *)(intptr_t)desc[21] : NULL;
        int32_t *cnt_out = places != NULL ? (int32_t *)(intptr_t)desc[22] : NULL;
        /* desc[24] (25 values): the host address of each expert of each part
         * (3 x E int64; a split of the experts over the NUMA nodes), or 0 */
        const int64_t *srct = (const int64_t *)(intptr_t)desc[24];
        int cap_all = cap;
        cap -= npre;            /* the copies of the plan go after the prefetched experts */
        if (desc[13] > 0) {
            cpu_a = desc[13];
            cpu_b = desc[14];
            gpu_c = desc[15];
        }
        int nr = (int)*nreal;
        nr = nr < 0 ? 0 : (nr > t ? t : nr);
        int *cnt = (int *)calloc((size_t)E, sizeof(int));
        int *order = (int *)malloc((size_t)E * 2 * sizeof(int));
        uint8_t *gpu = (uint8_t *)calloc((size_t)E, 1);
        for (int q = 0; q < nr * k; ++q) {
            cnt[ip[q]]++;
        }
        if (cnt_out != NULL) {
            memcpy(cnt_out, cnt, (size_t)E * sizeof(int));
        }
        int n = 0, hot = 0, pre_used = 0;
        int64_t cpu_all = 0, cpu_pre = 0;
        for (int e = 0; e < E; ++e) {
            if (cnt[e] == 0) {
                continue;
            }
            if (slots[e] >= 0) {
                gpu[e] = 1;
                ++hot;
                continue;
            }
            if (pre != NULL && pre[e] >= 0) {
                gpu[e] = 3;             /* prefetched: on the GPU at no cost */
                ++pre_used;
                cpu_pre += cpu_a + cpu_b * cnt[e];
                continue;
            }
            order[2 * n] = e;
            order[2 * n + 1] = cnt[e];
            cpu_all += cpu_a + cpu_b * cnt[e];
            ++n;
        }
        qsort(order, (size_t)n, 2 * sizeof(int), moe_plan_cmp);
        int best = 0;
        int64_t best_t = cpu_all, moved = 0;
        for (int m = 1; m <= n && m <= cap; ++m) {
            moved += cpu_a + cpu_b * order[2 * (m - 1) + 1];
            int64_t g = gpu_c * m, c = cpu_all - moved, tm = g > c ? g : c;
            if (tm < best_t) {
                best_t = tm;
                best = m;
            }
        }
        int64_t gpu_pairs = 0, cpu_pairs = 0;
        for (int i = 0; i < n; ++i) {
            if (i < best) {
                gpu[order[2 * i]] = 2;              /* copied */
                gpu_pairs += order[2 * i + 1];
            } else {
                cpu_pairs += order[2 * i + 1];
            }
        }
        memset(tab, 0, (size_t)3 * E * sizeof(int64_t));
        memset(ranges, 0, (size_t)3 * E * 3 * sizeof(int64_t));
        int row = 0;
        for (int p = 0; p < 3; ++p) {
            int64_t nb = desc[3 + p];
            int r = 0;                              /* the place in the buffer, in the order of the experts */
            for (int e = 0; e < E; ++e) {
                if (gpu[e] == 1) {
                    tab[(size_t)p * E + e] = slots[e] >= MOE_WARM
                        ? moe_pool_addr(pt, p, slots[e] - MOE_WARM)
                        : desc[p] + (int64_t)slots[e] * nb;
                } else if (gpu[e] == 3) {
                    tab[(size_t)p * E + e] = moe_pool_addr(pt, p, places[pre[e]]);
                } else if (gpu[e] == 2) {
                    int64_t dst = places != NULL ? moe_pool_addr(pt, p, places[npre + r])
                                                 : desc[9 + p] + (int64_t)r * nb;
                    tab[(size_t)p * E + e] = dst;
                    int64_t src = srct != NULL ? srct[(size_t)p * E + e] : desc[6 + p] + (int64_t)e * nb;
                    int64_t *last = row > 0 ? ranges + 3 * (row - 1) : NULL;
                    if (last != NULL && last[0] + last[2] == src && last[1] + last[2] == dst) {
                        last[2] += nb;              /* the next expert of a run */
                    } else {
                        ranges[3 * row] = src;
                        ranges[3 * row + 1] = dst;
                        ranges[3 * row + 2] = nb;
                        ++row;
                    }
                    ++r;
                }
            }
        }
        for (int q = 0; q < t * k; ++q) {
            int e = q / k < nr ? ip[q] : -1;
            if (gidx2 != NULL) {
                /* the hot experts (the GPU starts them during the copies), the
                 * prefetched ones (gidx3, or with the copied ones), and the
                 * copied ones apart */
                gidx[q] = e >= 0 && gpu[e] == 1 ? e : -1;
                gidx2[q] = e >= 0 && (gpu[e] == 2 || (gpu[e] == 3 && gidx3 == NULL)) ? e : -1;
                if (gidx3 != NULL) {
                    gidx3[q] = e >= 0 && gpu[e] == 3 ? e : -1;
                }
            } else {
                gidx[q] = e >= 0 && gpu[e] ? e : -1;
            }
            cidx[q] = e >= 0 && !gpu[e] ? e : -1;
        }
        {
            /* RECLANK_MIX_PLAN_LOG=path: a line for each plan (a test):
             * desc, the hot count, then expert:tokens of the copied ones */
            static FILE *lf;
            static int lon = -1;
            if (lon < 0) {
                const char *v = getenv("RECLANK_MIX_PLAN_LOG");
                lf = v ? fopen(v, "a") : NULL;
                lon = lf != NULL;
            }
            if (lon) {
                fprintf(lf, "%lld %d", (long long)(intptr_t)desc, hot);
                for (int i = 0; i < n; ++i) {
                    fprintf(lf, " %d:%d%s", order[2 * i], order[2 * i + 1], i < best ? "" : "c");
                }
                fprintf(lf, "\n");
                fflush(lf);
            }
        }
        stats[0] = best;
        stats[1] = gpu_pairs;
        stats[2] = n - best;
        stats[3] = cpu_pairs;
        stats[4] = hot;
        if (pre != NULL) {
            stats[5] = pre_used;
            stats[6] = npre;
            /* the copies with no prefetch: the same choice over all the
             * experts that are not hot */
            int m_all = 0;
            {
                int na = 0;
                int *oa = (int *)malloc((size_t)E * 2 * sizeof(int));
                for (int e = 0; e < E; ++e) {
                    if (cnt[e] > 0 && slots[e] < 0) {
                        oa[2 * na] = e;
                        oa[2 * na + 1] = cnt[e];
                        ++na;
                    }
                }
                qsort(oa, (size_t)na, 2 * sizeof(int), moe_plan_cmp);
                int64_t all = cpu_all + cpu_pre, mv = 0, bt = all;
                int64_t pc = desc[23] > 0 ? desc[23] : gpu_c;
                for (int m = 1; m <= na && m <= cap_all; ++m) {
                    mv += cpu_a + cpu_b * oa[2 * (m - 1) + 1];
                    int64_t g = pc * m, c = all - mv, tm = g > c ? g : c;
                    if (tm < bt) {
                        bt = tm;
                        m_all = m;
                    }
                }
                free(oa);
            }
            stats[7] = m_all;
        }
        free(cnt);
        free(order);
        free(gpu);
    }
}

/* ---------- helpers of the formats (kquants.c) ---------- */

#define MA_TB 128        /* the tokens of a block of the prompt pass */

/* The count of parts of 32 values of x up to which one thread quantizes x
 * (kq_quant_body). When the threads of the team write the parts, the
 * products after it read lines that many cores wrote. Each product was
 * slower by about 200 us on the E4B (gate 10240 x 2560 Q4_K: 418 us, 245 us
 * with one thread). One thread does 80 groups (2560 values) in about 2 us.
 * RECLANK_QUANT_SINGLE sets it (0: the team always). */
static int quant_single_max(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_QUANT_SINGLE");
        n = v ? atoi(v) : 512;
    }
    return n;
}

/* exp(x) with the operations of rc_dexp of the GPU (kernels/quant/kquants.cu),
 * so the SiLU of the act has the same bits on both (the expf of glibc and
 * of CUDA differ in the last bit for 13% of the gates): the range of
 * Cephes (x clamped to [-87, 88]), n = rint(x log2 e), r = x - n ln 2 in two
 * parts, a polynomial of degree 7 by fmaf, then 2^n by the exponent bits.
 * Each step is one operation (fmaf, or a product, or a sum), so no
 * contraction of the compiler changes it. About 1 ulp. */
static inline float rc_dexp(float x)
{
    x = fminf(fmaxf(x, -87.0f), 88.0f);
    float n = rintf(x * 1.44269504088896341f);
    float r = fmaf(n, -0.693359375f, x);
    r = fmaf(n, 2.12194440e-4f, r);
    float p = fmaf(1.9875691500e-4f, r, 1.3981999507e-3f);
    p = fmaf(p, r, 8.3334519073e-3f);
    p = fmaf(p, r, 4.1665795894e-2f);
    p = fmaf(p, r, 1.6666665459e-1f);
    p = fmaf(p, r, 5.0000001201e-1f);
    float z = r * r;
    float y = fmaf(p, z, r) + 1.0f;
    int32_t b = ((int32_t)n + 127) << 23;
    float sc;
    memcpy(&sc, &b, 4);
    return y * sc;
}

static inline float ma_silu(float v)
{
    return v / (1.f + rc_dexp(-v));
}

/* n bytes of a scratch buffer at *p (64-byte aligned steps). */
static inline void *ma_take(uint8_t **p, size_t n)
{
    void *r = *p;
    *p += (n + 63) & ~(size_t)63;
    return r;
}
