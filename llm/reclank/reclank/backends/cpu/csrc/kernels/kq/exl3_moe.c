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
 *   6. out = the sum of de with the weights (moe_combine), or with act bit 64
 *      the row of each pair (moe_rows: the rows form of the Qwen GPU).
 *
 * Act bit 32 (as RECLANK_E3_X16=1): x and the act as int16 in two int8
 * planes (kq_e3_had_in16), the products of each plane. Act bit 512
 * (GLM-5.3): the act is silu(min(gate, 10)) clamp(up, -10, 10).
 *
 * The address of expert e: the stack of mats (slot0: at slot0[e], or on
 * node 1 when slot0[e] < 0: the stack of mats1 at slot1[e]); a thread of
 * node 1 (on1) takes the copy in mats1 when it has the expert (the
 * weightkeeper's experts on every node, keep_exl3: each node reads its own
 * memory). */

static const uint8_t *kq_e3_expert(kq_mat m, kq_mat m1, int e, size_t per, const int32_t *slot1,
                                   const int32_t *slot0, int on1)
{
    if (on1 && (slot1 == NULL || slot1[e] >= 0)) {
        return m1.w + (size_t)(slot1 != NULL ? slot1[e] : e) * per;
    }
    int ex = slot0 != NULL ? slot0[e] : e;
    if (ex < 0) {
        return m1.w + (size_t)(slot1 != NULL ? slot1[e] : e) * per;
    }
    return m.w + (size_t)ex * per;
}

static void kq_e3_moe_body(const int32_t *ids, const float *val, int t, int k, int experts,
                           const int64_t *mats, const float *shared_logit, int hidden, int inner,
                           float *out, const float *hf, const int64_t *mats1, const int32_t *slot1,
                           const int32_t *slot0, int act_flags, uint8_t *scratch)
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
    /* the scratch of the record (rc_kq_moe_scratch: kq_e3_moe_bytes), or
     * (none) a malloc: a malloc of each call faulted its pages in and
     * zeroed them each time (a decode step of the GPU: the page locks of
     * the kernel took a sixth of the CPU) */
    uint8_t *mem = scratch;
    /* x16 (act bit 32, or RECLANK_E3_X16=1): xh as int16 in two int8 planes
     * (kq_e3_had_in16), as two rows of the pair (the high plane, then the
     * low one): the products of 2 to 8 rows decode the codes of a block once
     * (kq_e3_cols_many), so the second plane costs its vpdpbusd, not a
     * second decode; the two rows add up before the transform of the outputs */
    const int x16 = kq_e3_x16() || (act_flags & 32) != 0;
    const int clamp = (act_flags & 512) != 0;
    const int R = x16 ? 2 : 1;
    size_t nb = (size_t)(ne + 1) * 4 * 3 + (size_t)P0 * 4 * 3 + 64 + 16 * 64 +
                R * ((size_t)P0 * (2 * hidden + inner) + (size_t)P0 * (2 * nh + ni) * 8) +
                (size_t)P0 * (2 * inner + 2 * hidden) * 4 + R * (size_t)P0 * (2 * inner + hidden) * 4 + 16 * 64;
    if (scratch == NULL) {
        #pragma omp single copyprivate(mem)
        mem = (uint8_t *)malloc(nb);
    }
    uint8_t *p = mem;
    int *cnt = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *start = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *used = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *pair_tok = (int *)ma_take(&p, (size_t)P0 * 4);
    int *pair_of = (int *)ma_take(&p, (size_t)P0 * 4);
    int *exp_of = (int *)ma_take(&p, (size_t)P0 * 4);
    int *nused_p = (int *)ma_take(&p, 64);
    /* row r of a pair pr is row R pr + r (r 0: int8, or the high plane) */
    int8_t *xq[2], *aq;
    float *xs[2], *as, *yg[2], *yd, *act, *de;
    int32_t *xn[2], *an;
    for (int part = 0; part < 2; ++part) {
        xq[part] = (int8_t *)ma_take(&p, (size_t)R * P0 * hidden);
        xs[part] = (float *)ma_take(&p, (size_t)R * P0 * nh * 4);
        xn[part] = (int32_t *)ma_take(&p, (size_t)R * P0 * nh * 4);
        yg[part] = (float *)ma_take(&p, (size_t)R * P0 * inner * 4);
    }
    aq = (int8_t *)ma_take(&p, (size_t)R * P0 * inner);
    as = (float *)ma_take(&p, (size_t)R * P0 * ni * 4);
    an = (int32_t *)ma_take(&p, (size_t)R * P0 * ni * 4);
    act = (float *)ma_take(&p, (size_t)P0 * 2 * inner * 4);
    yd = (float *)ma_take(&p, (size_t)R * P0 * hidden * 4);
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
    const int on1 = mats1 != NULL && kq_my_node() == 1;
    #define E3_W(e, part) ((e) >= experts ? M[1][part].w \
                           : kq_e3_expert(M[0][part], M1[part], e, per[part], slot1, slot0, on1))
    #define E3_K(e, part) Kb[(e) >= experts][part]
    /* 1. xh of each pair for gate and up */
    #pragma omp for schedule(static)
    for (int q = 0; q < P * 2 * nh; ++q) {
        int pr = q / (2 * nh), part = (q / nh) % 2, b = q % nh, e = exp_of[pr];
        const uint8_t *suh = E3_W(e, part) + (size_t)hidden * inner * E3_K(e, part) / 8;
        const size_t r0 = (size_t)R * pr;
        if (x16) {
            kq_e3_had_in16(hf + (size_t)pair_tok[pr] * hidden, suh, b, xq[part] + r0 * hidden,
                           xs[part] + r0 * nh, xn[part] + r0 * nh, xq[part] + (r0 + 1) * hidden,
                           xs[part] + (r0 + 1) * nh, xn[part] + (r0 + 1) * nh);
        } else {
            kq_e3_had_in(hf + (size_t)pair_tok[pr] * hidden, suh, b, xq[part] + r0 * hidden,
                         xs[part] + r0 * nh, xn[part] + r0 * nh);
        }
    }
    /* 2. gate and up: tasks of an expert, a matrix, a tile column; the rows
     * of the pairs of the expert in chunks of 8 */
    const int tci = inner / 16;
    #pragma omp for schedule(dynamic, 1)
    for (int q = 0; q < nu * 2 * tci; ++q) {
        int e = used[q / (2 * tci)], part = (q / tci) % 2, tj = q % tci;
        int a0 = R * start[e], a1 = R * (start[e] + cnt[e]);
        for (int a = a0; a < a1; a += 8) {
            int m = a1 - a < 8 ? a1 - a : 8;
            kq_e3_cols(E3_W(e, part), E3_K(e, part), hidden, inner, tj, xq[part] + (size_t)a * hidden,
                       xs[part] + (size_t)a * nh, m, yg[part] + (size_t)a * inner, (size_t)inner);
        }
    }
    /* 3. the outputs of gate and up, silu(gate) up, and its xh for down */
    #pragma omp for schedule(static)
    for (int pr = 0; pr < P; ++pr) {
        int e = exp_of[pr];
        float *a = act + (size_t)pr * 2 * inner;
        const size_t r0 = (size_t)R * pr;
        for (int part = 0; part < 2; ++part) {
            const uint8_t *svh = E3_W(e, part) + (size_t)hidden * inner * E3_K(e, part) / 8 +
                                 2 * (size_t)hidden;
            float *yy = yg[part] + r0 * inner;
            float c = kq_e3_const(xs[part] + r0 * nh, xn[part] + r0 * nh, hidden);
            if (x16) {
                const float *yl = yy + inner;
                for (int i = 0; i < inner; ++i) {
                    yy[i] += yl[i];
                }
                c += kq_e3_const(xs[part] + (r0 + 1) * nh, xn[part] + (r0 + 1) * nh, hidden);
            }
            for (int ob = 0; ob < ni; ++ob) {
                kq_e3_had_out(yy, svh, ob, c, a + part * inner);
            }
        }
        if (clamp) {
            for (int i = 0; i < inner; ++i) {
                a[i] = ma_silu(fminf(a[i], 10.f)) * fminf(fmaxf(a[inner + i], -10.f), 10.f);
            }
        } else {
            for (int i = 0; i < inner; ++i) {
                a[i] = ma_silu(a[i]) * a[inner + i];
            }
        }
        const uint8_t *suh = E3_W(e, 2) + (size_t)inner * hidden * E3_K(e, 2) / 8;
        for (int b = 0; b < ni; ++b) {
            if (x16) {
                kq_e3_had_in16(a, suh, b, aq + r0 * inner, as + r0 * ni, an + r0 * ni, aq + (r0 + 1) * inner,
                               as + (r0 + 1) * ni, an + (r0 + 1) * ni);
            } else {
                kq_e3_had_in(a, suh, b, aq + r0 * inner, as + r0 * ni, an + r0 * ni);
            }
        }
    }
    /* 4. down */
    const int tch = hidden / 16;
    #pragma omp for schedule(dynamic, 1)
    for (int q = 0; q < nu * tch; ++q) {
        int e = used[q / tch], tj = q % tch;
        int a0 = R * start[e], a1 = R * (start[e] + cnt[e]);
        for (int a = a0; a < a1; a += 8) {
            int m = a1 - a < 8 ? a1 - a : 8;
            kq_e3_cols(E3_W(e, 2), E3_K(e, 2), inner, hidden, tj, aq + (size_t)a * inner,
                       as + (size_t)a * ni, m, yd + (size_t)a * hidden, (size_t)hidden);
        }
    }
    /* 5. the outputs of down */
    #pragma omp for schedule(static)
    for (int q = 0; q < P * nh; ++q) {
        int pr = q / nh, e = exp_of[pr];
        const size_t r0 = (size_t)R * pr;
        const uint8_t *svh = E3_W(e, 2) + (size_t)inner * hidden * E3_K(e, 2) / 8 + 2 * (size_t)inner;
        float *yy = yd + r0 * hidden;
        float c = kq_e3_const(as + r0 * ni, an + r0 * ni, inner);
        if (x16) {
            const float *yl = yy + hidden + 128 * (q % nh);
            for (int i = 0; i < 128; ++i) {
                yy[128 * (q % nh) + i] += yl[i];
            }
            c += kq_e3_const(as + (r0 + 1) * ni, an + (r0 + 1) * ni, inner);
        }
        kq_e3_had_out(yy, svh, q % nh, c, de + (size_t)pr * hidden);
    }
    /* 6. out */
    if (act_flags & 64) {
        moe_rows(de, pair_of, t, k, hidden, out);
    } else {
        moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
    }
    #pragma omp barrier
    if (scratch == NULL) {
        #pragma omp single
        free(mem);
    }
    #undef E3_W
    #undef E3_K
}
