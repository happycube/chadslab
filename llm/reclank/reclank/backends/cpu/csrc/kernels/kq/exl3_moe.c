/* ---------- the MUL1 experts (KQ_E3; kernels/kq/exl3.c) ----------
 *
 * GP_KQ_MOE with KQ_E3 matrices: the pairs of a token and an expert (and
 * the shared expert of each token), on the float rows hf. Each expert has
 * its own suh, so x is transformed for each pair and each of gate and up.
 * The pairs of an expert are together (moe_sort_pairs), so a task of a
 * tile column decodes the tiles once for all the tokens of the expert.
 *
 *   1. xh of each pair for gate and for up (kq_e3_had_in);
 *   2. tasks of an expert, a matrix (gate, up), and a tile column: y';
 *   3. for each pair the transforms of the outputs, silu(gate) up, and xh
 *      of that for down;
 *   4. tasks of an expert and a tile column of down: y';
 *   5. for each pair and block of 128 outputs the transform: de;
 *   6. out = the sum of de with the weights (moe_combine).
 *
 * The address of expert e: the stack of mats (slot0: at slot0[e], or on
 * node 1 when slot0[e] < 0: the stack of mats1 at slot1[e]). */

static const uint8_t *kq_e3_expert(kq_mat m, kq_mat m1, int e, size_t per, const int32_t *slot1,
                                   const int32_t *slot0)
{
    int ex = slot0 != NULL ? slot0[e] : e;
    if (ex < 0) {
        return m1.w + (size_t)(slot1 != NULL ? slot1[e] : e) * per;
    }
    return m.w + (size_t)ex * per;
}

static void kq_e3_moe_body(const int32_t *ids, const float *val, int t, int k, int experts,
                           const int64_t *mats, const float *shared_logit, int hidden, int inner,
                           float *out, const float *hf, const int64_t *mats1, const int32_t *slot1,
                           const int32_t *slot0)
{
    kq_mat M[2][3], M1[3];
    for (int part = 0; part < 3; ++part) {
        M[0][part] = kq_mat_of(mats + 2 * part);
        M[1][part] = kq_mat_of(mats + 6 + 2 * part);
        M1[part] = mats1 != NULL ? kq_mat_of(mats1 + 2 * part) : M[0][part];
    }
    if (hf == NULL) {
        fprintf(stderr, "kq_e3_moe_body: KQ_E3 experts need the float32 rows of x (hf)\n");
        abort();
    }
    const int shared = M[1][0].w != NULL;
    int Kb[2][3];
    for (int s = 0; s < 2; ++s) {
        for (int part = 0; part < 3; ++part) {
            Kb[s][part] = (s == 0 || shared) ? kq_e3_bits(M[s][part].type) : 1;
            if (!Kb[s][part]) {
                fprintf(stderr, "kq_e3_moe_body: every matrix must be KQ_E3\n");
                abort();
            }
        }
    }
    size_t per[3];
    for (int part = 0; part < 3; ++part) {
        per[part] = part < 2 ? kq_e3_bytes(Kb[0][part], hidden, inner)
                             : kq_e3_bytes(Kb[0][part], inner, hidden);
    }
    const int ne = experts + shared, P0 = t * k + (shared ? t : 0);
    const int nh = hidden / 128, ni = inner / 128;
    /* the scratch: the pairs sorted by expert, then the rows of each pair */
    uint8_t *mem = NULL;
    size_t nb = (size_t)(ne + 1) * 4 * 3 + (size_t)P0 * 4 * 3 + 64 + 16 * 64 +
                (size_t)P0 * (2 * hidden + inner) + (size_t)P0 * (2 * nh + ni) * 8 +
                (size_t)P0 * (4 * inner + 2 * hidden) * 4;
    #pragma omp single copyprivate(mem)
    mem = (uint8_t *)malloc(nb);
    uint8_t *p = mem;
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P0 * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P0 * 4);
    int *exp_of = (int *)ma_take(&p, (size_t)P0 * 4);
    int *nused_p = (int *)ma_take(&p, 64);
    int8_t *xq[2], *aq;
    float *xs[2], *as, *yg[2], *yd, *act, *de;
    int32_t *xn[2], *an;
    for (int part = 0; part < 2; ++part) {
        xq[part] = (int8_t *)ma_take(&p, (size_t)P0 * hidden);
        xs[part] = (float *)ma_take(&p, (size_t)P0 * nh * 4);
        xn[part] = (int32_t *)ma_take(&p, (size_t)P0 * nh * 4);
        yg[part] = (float *)ma_take(&p, (size_t)P0 * inner * 4);
    }
    aq = (int8_t *)ma_take(&p, (size_t)P0 * inner);
    as = (float *)ma_take(&p, (size_t)P0 * ni * 4);
    an = (int32_t *)ma_take(&p, (size_t)P0 * ni * 4);
    act = (float *)ma_take(&p, (size_t)P0 * 2 * inner * 4);
    yd = (float *)ma_take(&p, (size_t)P0 * hidden * 4);
    de = (float *)ma_take(&p, (size_t)P0 * hidden * 4);
    moe_sort_pairs(ids, t, k, experts, shared, cnt, start, used, pair_tok, pair_of, nused_p);
    const int nu = *nused_p, P = start[ne];
    #pragma omp single
    for (int u = 0; u < nu; ++u) {
        for (int a = start[used[u]]; a < start[used[u]] + cnt[used[u]]; ++a) {
            exp_of[a] = used[u];
        }
    }
    /* the matrix of expert e and its K, part 0 gate, 1 up, 2 down */
    #define E3_W(e, part) ((e) >= experts ? M[1][part].w \
                           : kq_e3_expert(M[0][part], M1[part], e, per[part], slot1, slot0))
    #define E3_K(e, part) Kb[(e) >= experts][part]
    /* 1. xh of each pair for gate and up */
    #pragma omp for schedule(static)
    for (int q = 0; q < P * 2 * nh; ++q) {
        int pr = q / (2 * nh), part = (q / nh) % 2, b = q % nh, e = exp_of[pr];
        kq_e3_had_in(hf + (size_t)pair_tok[pr] * hidden,
                     E3_W(e, part) + (size_t)hidden * inner * E3_K(e, part) / 8, b,
                     xq[part] + (size_t)pr * hidden, xs[part] + (size_t)pr * nh,
                     xn[part] + (size_t)pr * nh);
    }
    /* 2. gate and up: tasks of an expert, a matrix, a tile column; the pairs
     * of the expert in chunks of 8 */
    const int tci = inner / 16;
    #pragma omp for schedule(dynamic, 1)
    for (int q = 0; q < nu * 2 * tci; ++q) {
        int e = used[q / (2 * tci)], part = (q / tci) % 2, tj = q % tci;
        int a0 = start[e], n = cnt[e];
        for (int a = a0; a < a0 + n; a += 8) {
            int m = a0 + n - a < 8 ? a0 + n - a : 8;
            kq_e3_cols(E3_W(e, part), E3_K(e, part), hidden, inner, tj, xq[part] + (size_t)a * hidden,
                       xs[part] + (size_t)a * nh, m, yg[part] + (size_t)a * inner, (size_t)inner);
        }
    }
    /* 3. the outputs of gate and up, silu(gate) up, and its xh for down */
    #pragma omp for schedule(static)
    for (int pr = 0; pr < P; ++pr) {
        int e = exp_of[pr];
        float *a = act + (size_t)pr * 2 * inner;
        for (int part = 0; part < 2; ++part) {
            const uint8_t *svh = E3_W(e, part) + (size_t)hidden * inner * E3_K(e, part) / 8 +
                                 2 * (size_t)hidden;
            float c = kq_e3_const(xs[part] + (size_t)pr * nh, xn[part] + (size_t)pr * nh, hidden);
            for (int ob = 0; ob < ni; ++ob) {
                kq_e3_had_out(yg[part] + (size_t)pr * inner, svh, ob, c, a + part * inner);
            }
        }
        for (int i = 0; i < inner; ++i) {
            a[i] = ma_silu(a[i]) * a[inner + i];
        }
        const uint8_t *suh = E3_W(e, 2) + (size_t)inner * hidden * E3_K(e, 2) / 8;
        for (int b = 0; b < ni; ++b) {
            kq_e3_had_in(a, suh, b, aq + (size_t)pr * inner, as + (size_t)pr * ni, an + (size_t)pr * ni);
        }
    }
    /* 4. down */
    const int tch = hidden / 16;
    #pragma omp for schedule(dynamic, 1)
    for (int q = 0; q < nu * tch; ++q) {
        int e = used[q / tch], tj = q % tch;
        int a0 = start[e], n = cnt[e];
        for (int a = a0; a < a0 + n; a += 8) {
            int m = a0 + n - a < 8 ? a0 + n - a : 8;
            kq_e3_cols(E3_W(e, 2), E3_K(e, 2), inner, hidden, tj, aq + (size_t)a * inner,
                       as + (size_t)a * ni, m, yd + (size_t)a * hidden, (size_t)hidden);
        }
    }
    /* 5. the outputs of down */
    #pragma omp for schedule(static)
    for (int q = 0; q < P * nh; ++q) {
        int pr = q / nh, e = exp_of[pr];
        const uint8_t *svh = E3_W(e, 2) + (size_t)inner * hidden * E3_K(e, 2) / 8 + 2 * (size_t)inner;
        kq_e3_had_out(yd + (size_t)pr * hidden, svh, q % nh,
                      kq_e3_const(as + (size_t)pr * ni, an + (size_t)pr * ni, inner),
                      de + (size_t)pr * hidden);
    }
    /* 6. out */
    moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
    #pragma omp barrier
    #pragma omp single
    free(mem);
    #undef E3_W
    #undef E3_K
}
