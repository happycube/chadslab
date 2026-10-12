/* kq_moe_body for a few tokens with float rows (act bit 3: a decode step or a
 * verify group of the cold experts of a GPU step, t <= 4). The barriers of
 * kq_moe_body cost about 35 us of a layer of 240 on the 2-socket Xeon (the
 * quantization of x, the sort, the copy of the rows of the pairs, the act,
 * each with its barrier). Here:
 *
 * - one single: the int8 rows of the t tokens from hf, the sort of the
 *   pairs, the rows of the pairs, and the counters of the used experts;
 * - the gate and up tasks with no barrier after them: the thread that ends
 *   the last task of an expert computes its act and the int8 act (the code
 *   of kq_moe_body), then marks the expert ready;
 * - the down tasks with no barrier before them: a task waits for its expert;
 * - one barrier before the sum (moe_combine).
 *
 * Each pair has the operations of kq_moe_body: the same bits. */
static void kq_rows_bf12_f(const uint8_t *w, size_t rb, int nr, int cols, const float *x, size_t xst,
                           int n, float *out, size_t ost);

static void kq_moe_small_body(const int32_t *ids, const float *val, int t, int k, int experts,
                              const int64_t *mats, const float *shared_logit, int hidden,
                              int inner, uint8_t *scratch, float *out, int gelu, const float *hf,
                              const int64_t *mats1, const int32_t *slot1, const int32_t *slot0, int x2,
                              int rows)
{
    /* mats1 (or null): the copy of the experts on node 1: all of them (slot1
     * null), or a partial copy, expert e in slot slot1[e] of mats1 (-1: not
     * there). The threads of node 1 read it for those experts, and the
     * experts with a copy come last in used (the second half of the team
     * takes them: kq_share_range) */
    if (kq_mprof_on < 0) {
        const char *v = getenv("RECLANK_MOE_PROF");
        kq_mprof_on = v != NULL && atoi(v) != 0;
    }
    const int mprof = kq_mprof_on && omp_get_num_threads() <= 256;
    const int mtid = omp_get_thread_num();
    if (mprof) {
        kq_mt[0][mtid] = omp_get_wtime();
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
    int ne = experts + shared;
    int P = t * k + (shared ? t : 0);
    int nph = hidden / 32, npi = inner / 32;
    /* KQ_BF12 matrices read the float32 rows (hf, the act): kq_rows_bf12_f */
    const int bf12 = (G.type == KQ_BF12 || U.type == KQ_BF12 || D.type == KQ_BF12 ||
                      (shared && (SG.type == KQ_BF12 || SU.type == KQ_BF12 || SD.type == KQ_BF12))) &&
                     hf != NULL;
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
    int *done = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int *ready = (int *)ma_take(&p, (size_t)(ne + 1) * 4);
    int8_t *tq = (int8_t *)ma_take(&p, (size_t)t * hidden);
    float *ts = (float *)ma_take(&p, (size_t)t * nph * 4);
    float *tm = (float *)ma_take(&p, (size_t)t * nph * 2 * 4);
    /* x2: the low planes of x as int16 (kq_quant_part16) of the tokens, the
     * pairs, and the act; tq, xq, aq hold the high planes */
    int8_t *tq2 = NULL, *xq2 = NULL, *aq2 = NULL;
    if (x2) {
        tq2 = (int8_t *)ma_take(&p, (size_t)t * hidden);
        xq2 = (int8_t *)ma_take(&p, (size_t)P * hidden);
        aq2 = (int8_t *)ma_take(&p, (size_t)P * inner);
    }
    int ngu = G.type == KQ_NVX || G.type == KQ_Q4X ? 16 : 4;
    int ndn = D.type == KQ_NVX || D.type == KQ_Q4X ? 16 : kq_ndn_rows();
    /* the sums of xn and an of a pair: one plane, or with x2 (KQ_NVX) both */
    const size_t xns = (size_t)(hidden / 16) * (x2 && ngu == 16 ? 2 : 1);
    const size_t ans = (size_t)(inner / 16) * (x2 && ndn == 16 ? 2 : 1);
    #pragma omp single
    {
        double sb0 = mprof ? omp_get_wtime() : 0.0;
        for (int j = 0; j < t; ++j) {
            if (x2) {
                kq_quant_row16(hf + (size_t)j * hidden, hidden, tq + (size_t)j * hidden,
                               tq2 + (size_t)j * hidden, ts + (size_t)j * nph, tm + (size_t)j * 2 * nph);
                continue;
            }
            for (int g = 0; g < nph; ++g) {
                kq_quant_part(hf + (size_t)j * hidden, g, tq + (size_t)j * hidden,
                              ts + (size_t)j * nph, tm + (size_t)j * 2 * nph);
            }
        }
        double sb1 = mprof ? omp_get_wtime() : 0.0;
        /* moe_sort_pairs */
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
        /* the experts with a copy on node 1 last (kq_used_split); the
         * counters of the dynamic loops (kq_next_task) */
        double sb2 = mprof ? omp_get_wtime() : 0.0;
        if ((mats1 != NULL && slot1 != NULL) || slot0 != NULL) {
            kq_used_split3(used, nu, experts, slot0, slot1, mats1 != NULL, pair_tok, &nused_p[1],
                           &nused_p[6]);
        } else {
            nused_p[1] = nu;
            nused_p[6] = 0;
        }
        nused_p[2] = nused_p[3] = nused_p[4] = nused_p[5] = nused_p[7] = nused_p[8] = 0;
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
        *nused_p = nu;
        double sb3 = mprof ? omp_get_wtime() : 0.0;
        /* the rows of the pairs, and the counters of the used experts */
        for (int q = 0; q < start[ne]; ++q) {
            int j = pair_tok[q];
            if (bf12) {
                /* the float32 rows for the KQ_BF12 matrices (in de, which the
                 * down matrix of the pair writes after its gate and up) */
                memcpy(de + (size_t)q * hidden, hf + (size_t)j * hidden, (size_t)hidden * 4);
            }
            memcpy(xq + (size_t)q * hidden, tq + (size_t)j * hidden, (size_t)hidden);
            memcpy(xs + (size_t)q * nph, ts + (size_t)j * nph, (size_t)nph * 4);
            memcpy(xm + (size_t)q * 2 * nph, tm + (size_t)j * 2 * nph, (size_t)nph * 8);
            if (x2) {
                memcpy(xq2 + (size_t)q * hidden, tq2 + (size_t)j * hidden, (size_t)hidden);
            }
            if (ngu == 16) {
                kq_x16_xsum(G.type, xq + (size_t)q * hidden, hidden, xn + (size_t)q * xns);
                if (x2) {
                    kq_x16_xsum(G.type, xq2 + (size_t)q * hidden, hidden, xn + (size_t)q * xns + hidden / 16);
                }
            }
        }
        for (int u = 0; u < nu; ++u) {
            done[used[u]] = 0;
            ready[used[u]] = 0;
        }
        if (mprof) {
            double sb4 = omp_get_wtime();
            kq_mprof[14] += sb4 - sb0;      /* the body of the single region */
            kq_mprof[15] += nu;             /* the experts of the call (the shared one too) */
            kq_msb[0] += sb1 - sb0;
            kq_msb[1] += sb2 - sb1;
            kq_msb[2] += sb3 - sb2;
            kq_msb[3] += sb4 - sb3;
        }
    }
    int nu = *nused_p;
    P = start[ne];
    if (mprof) {
        kq_mt[1][mtid] = omp_get_wtime();
    }
    static int dyn = -1;
    if (dyn < 0) {
        const char *v = getenv("RECLANK_MOE_DYN");
        dyn = v ? atoi(v) : 0;
    }
    const int per_gu = 2 * inner / ngu, per_dn = hidden / ndn;
    /* one token: a static split with the share of each node (kq_share_range);
     * a group: dynamic (its experts have different counts of tokens). A
     * partial copy on node 1 (strict): the threads of node 1 take only the
     * experts of the copy (the last nA.. of used): kq_share_range_strict,
     * or a slot for each thread that takes its tasks with kq_next_task. */
    const int st1 = t == 1 && !dyn;
    const int strict = (mats1 != NULL && slot1 != NULL) || slot0 != NULL;
    const int nA = nused_p[1], nAB = nused_p[1] + nused_p[6];
    const int qd = strict && !st1;      /* the dynamic loops by kq_next_task */
    omp_set_schedule(qd ? omp_sched_static : t > 1 || dyn ? omp_sched_dynamic : omp_sched_static,
                     qd ? 1 : t > 1 ? 8 : (dyn ? dyn : 0));
    int glo = 0, ghi = 0, dlo = 0, dhi = 0;
    if (st1 && strict) {
        kq_share_range_strict(nu * per_gu, nA * per_gu, nAB * per_gu, &glo, &ghi);
        kq_share_range_strict(nu * per_dn, nA * per_dn, nAB * per_dn, &dlo, &dhi);
    } else if (st1) {
        kq_share_range(nu * per_gu, &glo, &ghi);
        kq_share_range(nu * per_dn, &dlo, &dhi);
    }
    double tbusy = 0.0;
    int pn = 0;     /* this thread's tasks of the expert of the run, not yet in done */
    #pragma omp for schedule(runtime) nowait
    for (int x0 = 0; x0 < (st1 || qd ? omp_get_num_threads() : nu * per_gu); ++x0) {
      int xa = st1 ? glo : x0, xb = st1 ? ghi : x0 + 1;
      if (qd) {
          xa = kq_task_or_end(&nused_p[2], nA * per_gu, &nused_p[3], nAB * per_gu, &nused_p[7], nu * per_gu, on1);
          xb = INT_MAX;
      }
      for (int x = xa; x < xb;
           x = qd ? kq_task_or_end(&nused_p[2], nA * per_gu, &nused_p[3], nAB * per_gu, &nused_p[7], nu * per_gu, on1) : x + 1) {
        double tc = (kq_calib_on || mprof) ? omp_get_wtime() : 0.0;
        int e = used[x / per_gu], rr = (x % per_gu) * ngu, isup = rr >= inner;
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
            m = c1 ? G1 : G;
            r = rr;
            estride = 2 * (size_t)inner;
        }
        rb = kq_row_bytes(m.type, hidden);
        const uint8_t *w = m.w + (e < experts ? (size_t)ex * estride * rb : 0);
        int s0 = start[e];
        int n = start[e + 1] - s0;
        if (m.type == KQ_BF12) {
            kq_rows_bf12_f(w + (size_t)r * rb, rb, ngu, hidden, de + (size_t)s0 * hidden,
                           (size_t)hidden, n, act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
        } else {
            kq_rows_n2(w + (size_t)r * rb, rb, m.type, ngu, hidden, xq + (size_t)s0 * hidden,
                       xs + (size_t)s0 * nph, xm + (size_t)s0 * 2 * nph, xn + (size_t)s0 * xns,
                       x2 ? xq2 + (size_t)s0 * hidden : NULL, n,
                       act + (size_t)s0 * 2 * inner + rr, (size_t)2 * inner);
        }
        /* done[e]: one add for a run of tasks of expert e (a static range:
         * the next task of the range on the same expert waits), not one for
         * each task from 24 threads on two sockets (the same speed of a
         * decode step on the 2-socket Xeon: the kernels take the time) */
        int add = 1;
        if (!qd) {
            ++pn;
            if (x + 1 < xb && used[(x + 1) / per_gu] == e) {
                if (kq_calib_on || mprof) {
                    tbusy += omp_get_wtime() - tc;
                }
                continue;
            }
            add = pn;
            pn = 0;
        }
        if (__atomic_add_fetch(&done[e], add, __ATOMIC_ACQ_REL) == per_gu) {
            /* the last task of expert e: its act and int8 act */
            for (int q = s0; q < s0 + n; ++q) {
                float *a = act + (size_t)q * 2 * inner;
                if (gelu) {
                    for (int i = 0; i < inner; ++i) {
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
            __atomic_store_n(&ready[e], 1, __ATOMIC_RELEASE);
        }
        if (kq_calib_on || mprof) {
            tbusy += omp_get_wtime() - tc;
        }
      }
    }
    if (mprof) {
        kq_mt[2][mtid] = omp_get_wtime();
        if (mtid < 256) {
            kq_mbusy[mtid][0] += tbusy;     /* the kernels of gate and up */
        }
    }
    #pragma omp for schedule(runtime) nowait
    for (int x0 = 0; x0 < (st1 || qd ? omp_get_num_threads() : nu * per_dn); ++x0) {
      int xa = st1 ? dlo : x0, xb = st1 ? dhi : x0 + 1;
      if (qd) {
          xa = kq_task_or_end(&nused_p[4], nA * per_dn, &nused_p[5], nAB * per_dn, &nused_p[8], nu * per_dn, on1);
          xb = INT_MAX;
      }
      for (int x = xa; x < xb;
           x = qd ? kq_task_or_end(&nused_p[4], nA * per_dn, &nused_p[5], nAB * per_dn, &nused_p[8], nu * per_dn, on1) : x + 1) {
        int e = used[x / per_dn], r = (x % per_dn) * ndn;
        while (!__atomic_load_n(&ready[e], __ATOMIC_ACQUIRE)) {
#if GEMMA_X86
            _mm_pause();
#endif
        }
        double tc = (kq_calib_on || mprof) ? omp_get_wtime() : 0.0;
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
        if (m.type == KQ_BF12) {
            kq_rows_bf12_f(w + (size_t)r * rb, rb, ndn, inner, act + (size_t)s0 * 2 * inner,
                           (size_t)2 * inner, n, de + (size_t)s0 * hidden + r, (size_t)hidden);
        } else {
            kq_rows_n2(w + (size_t)r * rb, rb, m.type, ndn, inner, aq + (size_t)s0 * inner,
                       as + (size_t)s0 * npi, am + (size_t)s0 * 2 * npi, an + (size_t)s0 * ans,
                       x2 ? aq2 + (size_t)s0 * inner : NULL, n,
                       de + (size_t)s0 * hidden + r, (size_t)hidden);
        }
        if (kq_calib_on || mprof) {
            tbusy += omp_get_wtime() - tc;
        }
      }
    }
    if (kq_calib_on && omp_get_thread_num() < 512) {
        kq_calib_busy[omp_get_thread_num()] += tbusy;
    }
    if (mprof && mtid < 256) {
        kq_mbusy[mtid][1] += tbusy;         /* the kernels of gate, up and down */
    }
    if (mprof) {
        kq_mt[3][mtid] = omp_get_wtime();
    }
    #pragma omp barrier
    if (mprof) {
        kq_mt[4][mtid] = omp_get_wtime();
    }
    if (rows) {
        moe_rows(de, pair_of, t, k, hidden, out);       /* (act bit 64: no shared expert here) */
    } else {
        moe_combine(de, pair_of, val, shared_logit, shared, t, k, hidden, out);
    }
    if (mprof) {
        double t5 = omp_get_wtime();
        #pragma omp barrier
        #pragma omp master
        {
            int n = omp_get_num_threads();
            double in0 = 1e30, in1 = 0, gm = 0, gx = 0, dm = 0, dx = 0, bw = 0;
            for (int q = 0; q < n; ++q) {
                in0 = fmin(in0, kq_mt[0][q]);
                in1 = fmax(in1, kq_mt[0][q]);
            }
            double s1 = kq_mt[1][0];
            for (int q = 0; q < n; ++q) {
                double g = kq_mt[2][q] - s1, d = kq_mt[3][q] - s1;
                gm += g / n;
                gx = fmax(gx, g);
                dm += d / n;
                dx = fmax(dx, d);
                bw += (kq_mt[4][q] - kq_mt[3][q]) / n;
            }
            kq_mprof[0] += in1 - in0;
            kq_mprof[1] += s1 - in1;
            kq_mprof[2] += gm;
            kq_mprof[3] += gx;
            kq_mprof[4] += dm;
            kq_mprof[5] += dx;
            kq_mprof[6] += t5 - in0;
            for (int q = 0; q < n; ++q) {
                kq_mthr[q][0] += kq_mt[0][q] - in0;
                kq_mthr[q][1] += kq_mt[2][q] - s1;
                kq_mthr[q][2] += kq_mt[3][q] - s1;
            }
            for (int hf = 0; hf < 2; ++hf) {
                int q0 = hf * (n / 2), q1 = hf ? n : n / 2;
                double ar = 0, gu = 0, dn = 0;
                for (int q = q0; q < q1; ++q) {
                    ar += (kq_mt[0][q] - in0) / (q1 - q0);
                    gu += (kq_mt[2][q] - s1) / (q1 - q0);
                    dn += (kq_mt[3][q] - s1) / (q1 - q0);
                }
                kq_mprof[8 + 3 * hf] += ar;
                kq_mprof[9 + 3 * hf] += gu;
                kq_mprof[10 + 3 * hf] += dn;
            }
            (void)bw;
            ++kq_mprof_n;
        }
    }
}

/* Measure the rate of the threads of each half of a team of nth threads
 * bound spread (node 0, node 1) on the cold experts of a decode step: reps
 * calls of kq_moe_small_body on one token with ncold random experts of the
 * layers of mats (nl tables of 12 int64; mats1: the copies of node 1, or
 * null), with the current share. out[0], out[1]: the mean compute time of a
 * thread of each half (s). */
void rc_kq_calib_nodes(const int64_t *mats, const int64_t *mats1, int nl, int experts, int hidden,
                    int inner, int ncold, int reps, int nth, double *out)
{
    size_t sb = rc_kq_moe_scratch(1, ncold, experts, hidden, inner);
    uint8_t *scratch = (uint8_t *)aligned_alloc(64, (sb + 63) / 64 * 64);
    float *hf = (float *)malloc((size_t)hidden * 4), *o = (float *)malloc((size_t)hidden * 4);
    int32_t *ids = (int32_t *)malloc((size_t)ncold * 4);
    float *val = (float *)malloc((size_t)ncold * 4);
    uint32_t seed = 12345u;
    for (int i = 0; i < hidden; ++i) {
        seed = seed * 1664525u + 1013904223u;
        hf[i] = ((float)(seed >> 8) / 16777216.0f - 0.5f) * 0.2f;
    }
    for (int i = 0; i < 512; ++i) {
        kq_calib_busy[i] = 0.0;
    }
    kq_calib_on = 1;
    for (int rp = 0; rp < reps; ++rp) {
        for (int c = 0; c < ncold; ++c) {
            int e, dup;
            do {
                seed = seed * 1664525u + 1013904223u;
                e = (int)((seed >> 8) % (uint32_t)experts);
                dup = 0;
                for (int d = 0; d < c; ++d) {
                    dup |= ids[d] == e;
                }
            } while (dup);
            ids[c] = e;
            val[c] = 1.0f / ncold;
        }
        const int64_t *m = mats + (size_t)(rp % nl) * 12;
        const int64_t *m1 = mats1 != NULL ? mats1 + (size_t)(rp % nl) * 12 : NULL;
        #pragma omp parallel num_threads(nth) proc_bind(spread)
        kq_moe_small_body(ids, val, 1, ncold, experts, m, NULL, hidden, inner, scratch, o, 0, hf, m1,
                          NULL, NULL, 0, 0);
    }
    kq_calib_on = 0;
    double b0 = 0.0, b1 = 0.0;
    int half = nth / 2;
    for (int i = 0; i < nth && i < 512; ++i) {
        if (i < half) {
            b0 += kq_calib_busy[i];
        } else {
            b1 += kq_calib_busy[i];
        }
    }
    out[0] = half > 0 ? b0 / half / reps : 0.0;
    out[1] = nth - half > 0 ? b1 / (nth - half) / reps : 0.0;
    free(scratch);
    free(hf);
    free(o);
    free(ids);
    free(val);
}

/* mats: the up matrix with no data: the gate matrix holds 2 inner rows for
 * each expert (its gate rows, then its up rows).
 * act: bit 0 the tanh GELU of the gate (the Gemma 4 26B) in place of SiLU;
 * bit 1 float32 activations (hf: the float32 rows of h; KQ_Q4X matrices
 * only): no quantization, as the decode of the 26B; bit 2 int16 activations
 * (hf; KQ_Q4X; the prompt with RECLANK_INT4_Q8=16): the rows of h and of
 * the GELU become int16 (kq_q4x_rows16); bit 3 (hf: the float rows of the
 * tokens, t <= 4, no int16): kq_moe_small_body, with few barriers. */
/* RECLANK_MOE_GPROF=1: the phases of kq_moe_body (a prompt group), summed
 * over the calls (rc_kq_moe_gprof, us from the first thread to arrive): 0 the
 * spread of the arrival, 1 the end of the sort, 2 of the copy of the rows,
 * 3 4 the last gate and up task of each thread (mean, max), 5 their
 * barrier, 6 the end of the act, 7 8 the last down task (mean, max), 9 its
 * barrier, 10 the end of the sum; 11 the calls. */
static int kq_gprof_on = -1;
static double kq_gt[256][9];
static double kq_gprof_s[12];

void rc_kq_moe_gprof(double *out)
{
    double n = kq_gprof_s[11] > 0 ? kq_gprof_s[11] : 1;
    for (int i = 0; i < 11; ++i) {
        out[i] = 1e6 * kq_gprof_s[i] / n;
        kq_gprof_s[i] = 0;
    }
    out[11] = kq_gprof_s[11];
    kq_gprof_s[11] = 0;
}
