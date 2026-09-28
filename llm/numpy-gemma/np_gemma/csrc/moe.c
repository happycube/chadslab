/* The parts of a MoE layer that do not depend on the weight format: the
 * sort of the pairs (token, expert) by expert, and the weighted sum of the
 * outputs of the experts. The formats (mlx_affine.c, kquants.c) use them.
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
 * of a GPU step that the GPU computes): its pair_of is -1. */
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
            if (ids[q] >= 0) {
                cnt[ids[q]]++;
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
            int e = ids[q];
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
        float sw = shared ? 1.f / (1.f + expf(-shared_logit[j])) : 0.f;
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
                          int64_t *ranges, int64_t *stats)
{
    #pragma omp single
    {
        int cap = (int)desc[12];
        int nr = (int)*nreal;
        nr = nr < 0 ? 0 : (nr > t ? t : nr);
        int *cnt = (int *)calloc((size_t)E, sizeof(int));
        int *order = (int *)malloc((size_t)E * 2 * sizeof(int));
        uint8_t *gpu = (uint8_t *)calloc((size_t)E, 1);
        for (int q = 0; q < nr * k; ++q) {
            cnt[ip[q]]++;
        }
        int n = 0, hot = 0;
        int64_t cpu_all = 0;
        for (int e = 0; e < E; ++e) {
            if (cnt[e] == 0) {
                continue;
            }
            if (slots[e] >= 0) {
                gpu[e] = 1;
                ++hot;
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
                    tab[(size_t)p * E + e] = desc[p] + (int64_t)slots[e] * nb;
                } else if (gpu[e] == 2) {
                    int64_t dst = desc[9 + p] + (int64_t)r * nb;
                    tab[(size_t)p * E + e] = dst;
                    int64_t src = desc[6 + p] + (int64_t)e * nb;
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
            gidx[q] = e >= 0 && gpu[e] ? e : -1;
            cidx[q] = e >= 0 && !gpu[e] ? e : -1;
        }
        stats[0] = best;
        stats[1] = gpu_pairs;
        stats[2] = n - best;
        stats[3] = cpu_pairs;
        stats[4] = hot;
        free(cnt);
        free(order);
        free(gpu);
    }
}
