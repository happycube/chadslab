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
 * then t * k + j for the shared expert). *nused gets the count of used. */
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
            cnt[ids[q]]++;
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
        for (int q = 0; q < t * k; ++q) {
            int e = ids[q];
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
            const float *d = de + (size_t)pair_of[j * k + sl] * hidden;
            float w = val[j * k + sl];
            for (int c = c0; c < c0 + 64; ++c) {
                o[c] += w * d[c];
            }
        }
    }
}
