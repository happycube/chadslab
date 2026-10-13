/* ---------- the program interpreter ----------
 * PERF_PLAN.md, phase 2. Python builds a program (reclank/ir/program.py) and
 * this code runs it. The program is one int64 array:
 *
 *     int64  magic, env count, record count, 0
 *     int64  env[env count]         the slots: parameters and variables
 *     record code[record count]
 *
 * A record has an operation code, flags, and GP_NARG operands. A tag gives
 * the kind of each operand. An operand is an integer literal, a float literal,
 * or a slot of the environment. An address is an integer, and a float literal
 * holds the bits of a float32.
 *
 * rc_run opens one OpenMP region for the whole program. Each thread copies
 * the environment. A scalar operation writes the private copy of each thread,
 * so the scalar operations need no barrier and no thread writes shared data.
 * A kernel operation calls the body of a kernel. The loops of a body are an
 * orphaned "omp for", which ends with a barrier. A one-row operation runs in
 * an "omp single", which also ends with a barrier.
 *
 * limit runs only the first limit records. A check compares the buffers after
 * each prefix with the Python interpreter, and the first difference names the
 * faulty operation. A limit below zero runs every record.
 */

static void gp_step(const gp_rec *r, int64_t *e)
{
    switch (r->op) {
    case GP_ROUTER_TOPK:
        /* logits, t, experts, k, val, idx, [map, bias, [swap, sel]] */
        gp_router_topk_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_I(3), GP_P(float, 4),
                            GP_P(int32_t, 5),
                            r->tag[6] == GP_T_NONE ? NULL : GP_P(const int32_t, 6),
                            r->tag[6] == GP_T_NONE ? 0.f : gp_f(r, e, 7),
                            r->tag[8] == GP_T_NONE ? 0.f : gp_f(r, e, 8),
                            r->tag[9] == GP_T_NONE ? NULL : GP_P(int32_t, 9));
        break;
    case GP_GDN:
        /* qkv, conv, conv_w, kernel, z, a, b, A_log, dt_bias, norm_w, S, out, scratch,
         * t, k_heads, v_heads, k_dim, v_dim, eps, log (null, or an MTP verify group),
         * tiled (the order of the value heads) */
        gdn_body(GP_P(const float, 0), GP_P(float, 1), GP_P(const float, 2), GP_I(3),
                 GP_P(const float, 4), GP_P(const float, 5), GP_P(const float, 6),
                 GP_P(const float, 7), GP_P(const float, 8), GP_P(const float, 9), GP_P(float, 10),
                 GP_P(float, 11), GP_P(float, 12), GP_I(13), GP_I(14), GP_I(15), GP_I(16),
                 GP_I(17), gp_f(r, e, 18), GP_P(float, 19), GP_I(20));
        break;
    case GP_HC_NORM:
        /* x, w, out, t, groups, hid, eps */
        hc_norm_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_I(3), GP_I(4),
                     GP_I(5), gp_f(r, e, 6));
        break;
    case GP_HC_ACT:
        /* x, out, n, scale */
        hc_act_body(GP_P(const float, 0), GP_P(float, 1), gp_i(r, e, 2), gp_f(r, e, 3));
        break;
    case GP_HC_MIX:
        /* hn, g, out, t, hc, hid */
        hc_mix_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_I(3), GP_I(4),
                    GP_I(5));
        break;
    case GP_HC_ADD:
        /* H, out, inject, t, hc, hid, scale */
        hc_add_body(GP_P(float, 0), GP_P(const float, 1), GP_P(const float, 2), GP_I(3), GP_I(4),
                    GP_I(5), gp_f(r, e, 6));
        break;
    case GP_PLE_GATE:
        /* keyn, qn, value, gated, t, hc, hid */
        ple_gate_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2),
                      GP_P(float, 3), GP_I(4), GP_I(5), GP_I(6));
        break;
    case GP_PLE_CONV:
        /* gn, gated, H, state, w, t, channels, kernel, dilation */
        ple_conv_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_P(float, 3),
                      GP_P(const float, 4), GP_I(5), GP_I(6), GP_I(7), GP_I(8));
        break;
    case GP_MOE_PLAN:
        /* ip, nreal, t, k, E, slots, desc, cpu_a, cpu_b, gpu_c, tab, gidx, cidx, ranges,
         * stats (moe.c) */
        moe_plan_body(GP_P(const int32_t, 0), GP_P(const int64_t, 1), GP_I(2), GP_I(3), GP_I(4),
                      GP_P(const int32_t, 5), GP_P(const int64_t, 6), gp_i(r, e, 7),
                      gp_i(r, e, 8), gp_i(r, e, 9), GP_P(int64_t, 10), GP_P(int32_t, 11),
                      GP_P(int32_t, 12), GP_P(int64_t, 13), GP_P(int64_t, 14), GP_P(int32_t, 15));
        break;
    case GP_HC_CAT:
        /* e, hn, out, t, hc, hid */
        hc_cat_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_I(3), GP_I(4),
                    GP_I(5));
        break;
    case GP_QSA_SELECT:
        /* iq, ik, idxk, blk, qn, kn, cos, sin, pos, t, heads, d, ratio, budget, rot, theta,
         * eps, sel, cnt, maxsel, scratch, nbmax, qpos (or 0), sec */
        qsa_select_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(uint16_t, 2),
                        GP_P(uint16_t, 3),
                        GP_P(const float, 4), GP_P(const float, 5), GP_P(const float, 6),
                        GP_P(const float, 7), gp_i(r, e, 8), GP_I(9), GP_I(10), GP_I(11), GP_I(12),
                        GP_I(13), GP_I(14), gp_f(r, e, 15), gp_f(r, e, 16), GP_P(int32_t, 17),
                        GP_P(int32_t, 18), GP_I(19), GP_P(uint8_t, 20), gp_i(r, e, 21),
                        GP_P(const int32_t, 22), GP_I(23));
        break;
    case GP_ATTN_QSA:
        /* q, kq, ks, vq, vs, scores, out, nq, nk, hd, t, pos, sel, cnt, maxsel,
         * form (0 int16, 1 int8, 2 int16 keys and int8 values, 3 float32
         * rows), hs (form 3: the head stride) */
        attn_qsa_body(GP_P(const float, 0), GP_P(const void, 1), GP_P(const float, 2),
                      GP_P(const void, 3), GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                      GP_I(7), GP_I(8), GP_I(9), GP_I(10), gp_i(r, e, 11), GP_P(const int32_t, 12),
                      GP_P(const int32_t, 13), GP_I(14), GP_I(15), gp_i(r, e, 16));
        break;
    case GP_KQ_QUANT:
        /* x, t, cols, xq, xs, xm, then (or none) ids and k: only the rows
         * with a selected expert */
        if (GP_P(const int32_t, 6) != NULL) {
            kq_quant_rows_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_P(int8_t, 3),
                               GP_P(float, 4), GP_P(float, 5), GP_P(const int32_t, 6), GP_I(7));
        } else {
            kq_quant_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_P(int8_t, 3),
                          GP_P(float, 4), GP_P(float, 5));
        }
        break;
    case GP_KQ_LINEAR:
        /* xq, xs, xm, x, w, type, rows, cols, t, out */
        kq_linear_body(GP_P(const uint8_t, 4), GP_I(5), GP_I(6), GP_I(7), GP_P(const int8_t, 0),
                       GP_P(const float, 1), GP_P(const float, 2), GP_P(const float, 3), GP_I(8),
                       GP_P(float, 9));
        break;
    case GP_KQ_MOE:
        /* hq, hs, hm, ids, val, t, k, experts, mats, shared_logit, hidden, inner, scratch,
         * out, kcount (null, or the count of experts of one token), act, hf, mats1 (the
         * copy of node 1, or null), slot1 (a partial mats1: the slot of each expert, or -1;
         * null: a full copy), slot0 (null: mats holds every expert; else the slot of each
         * expert in mats, or -1: a split of the experts over the nodes) */
        kq_moe_body(GP_P(const int8_t, 0), GP_P(const float, 1), GP_P(const float, 2),
                    GP_P(const int32_t, 3), GP_P(const float, 4), GP_I(5), GP_I(6), GP_I(7),
                    GP_P(const int64_t, 8), GP_P(const float, 9), GP_I(10), GP_I(11),
                    GP_P(uint8_t, 12), GP_P(float, 13),
                    GP_P(const int32_t, 14), GP_I(15), GP_P(const float, 16),
                    GP_P(const int64_t, 17), GP_P(const int32_t, 18), GP_P(const int32_t, 19));
        break;
    case GP_KQ_MOE_PART:
        /* hf, ids, val, t, k, experts, pm (nparts x 12), hidden, inner, gu (nparts x 2:
         * a0, ni), dn (c0, nd), act2, out, nparts, team, kcount (or null), flags, scr
         * (the scratch of each part): the flat team of GP_RUN_PARTS
         * (kernels/kq/moe_part.c) */
        kq_moe_part_body(GP_P(const float, 0), GP_P(const int32_t, 1), GP_P(const float, 2), GP_I(3),
                         GP_I(4), GP_I(5), GP_P(const int64_t, 6), GP_I(7), GP_I(8),
                         GP_P(const int32_t, 9), GP_P(const int32_t, 10), GP_P(float, 11), GP_P(float, 12),
                         GP_I(13), GP_I(14), GP_P(const int32_t, 15), GP_I(16), GP_P(const int64_t, 17));
        break;
    case GP_RUN_PARTS:
        /* the first record of a program of rc_run_task (run.c runs it) */
        break;
    case GP_ATTN_PREP:
        /* qg, kk, vv, qn, kn, cos, sin, K, V, hs, pos, t, nq, nk, hd, rot, eps, scale,
         * qout, gate */
        gp_attn_prep_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2),
                          GP_P(const float, 3), GP_P(const float, 4), GP_P(const float, 5),
                          GP_P(const float, 6), GP_P(float, 7), GP_P(float, 8), gp_i(r, e, 9),
                          gp_i(r, e, 10), GP_I(11), GP_I(12), GP_I(13), GP_I(14), GP_I(15),
                          gp_f(r, e, 16), gp_f(r, e, 17), GP_P(float, 18), GP_P(float, 19),
                          GP_P(float, 20));
        break;
    case GP_SIGMUL:
        /* x, g, out, n */
        gp_sigmul_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2),
                       gp_i(r, e, 3));
        break;
    /* ---- scalar operations: every thread, private copy ---- */
    case GP_S_MOV: e[r->v[0]] = gp_i(r, e, 1); break;
    case GP_S_ADD: e[r->v[0]] = gp_i(r, e, 1) + gp_i(r, e, 2); break;
    case GP_S_SUB: e[r->v[0]] = gp_i(r, e, 1) - gp_i(r, e, 2); break;
    case GP_S_MUL: e[r->v[0]] = gp_i(r, e, 1) * gp_i(r, e, 2); break;
    case GP_S_MAX: {
        int64_t a = gp_i(r, e, 1), b = gp_i(r, e, 2);
        e[r->v[0]] = a > b ? a : b;
        break;
    }
    case GP_S_MIN: {
        int64_t a = gp_i(r, e, 1), b = gp_i(r, e, 2);
        e[r->v[0]] = a < b ? a : b;
        break;
    }
    /* ---- one-row operations ---- */
    case GP_RMS_NORM: {
        /* x, w (0 for none), out, rows, cols, eps */
        const float *x = GP_P(const float, 0);
        const float *w = GP_P(const float, 1);
        float *out = GP_P(float, 2);
        int rows = GP_I(3), cols = GP_I(4);
        float eps = gp_f(r, e, 5);
        if (rows < 8) {
            #pragma omp single
            for (int i = 0; i < rows; ++i) {
                gemma_rms_norm_row(x + (size_t)i * (size_t)cols, w,
                                   out + (size_t)i * (size_t)cols, cols, eps);
            }
        } else {
            /* the rows of a prompt over the team, as rc_rms_norm */
            gemma_rms_norm_body(x, w, out, rows, cols, eps);
        }
        break;
    }
    case GP_ADD: {
        /* a, b, out, n: the team adds (each value as one thread did) */
        const float *a = GP_P(const float, 0);
        const float *b = GP_P(const float, 1);
        float *out = GP_P(float, 2);
        int n = GP_I(3);
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            out[i] = a[i] + b[i];
        }
        break;
    }
    case GP_MUL_S: {
        /* x, s (float), out, n */
        const float *x = GP_P(const float, 0);
        float sc = gp_f(r, e, 1);
        float *out = GP_P(float, 2);
        int n = GP_I(3);
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            out[i] = x[i] * sc;
        }
        break;
    }
    case GP_COPY: {
        /* src, dst, bytes */
        const void *src = GP_P(const void, 0);
        void *dst = GP_P(void, 1);
        size_t n = (size_t)gp_i(r, e, 2);
        #pragma omp single
        memcpy(dst, src, n);
        break;
    }
    case GP_GELU:
        /* Operands: x, out, n. As ops.gelu_tanh. */
        gemma_gelu_body(GP_P(const float, 0), GP_P(float, 1), GP_I(2));
        break;
    case GP_MUL: {
        /* Operands: a, b, out, rows, cols, b_stride. out = a * b for each
         * value. Row r of b starts at r * b_stride. Thus b can be a slice of a
         * wider array, for example the per-layer input of the E4B model. */
        const float *a = GP_P(const float, 0);
        const float *b = GP_P(const float, 1);
        float *out = GP_P(float, 2);
        int rows = GP_I(3), cols = GP_I(4);
        size_t bs = (size_t)gp_i(r, e, 5);
        /* the values over the team (a row of a prompt has 15360) */
        #pragma omp for schedule(static)
        for (long ic = 0; ic < (long)rows * cols; ++ic) {
            const long i = ic / cols, c = ic % cols;
            out[(size_t)ic] = a[(size_t)ic] * b[(size_t)i * bs + (size_t)c];
        }
        break;
    }
    case GP_BF16_LINEAR:
        /* Operands: x, w, out, rows, cols, tokens. A bfloat16 matrix. */
        gemma_bf16_linear_body(GP_P(const uint16_t, 1), GP_P(const float, 0),
                               GP_P(float, 2), GP_I(3), GP_I(4), GP_I(5));
        break;
    /* ---- int4 matrices ---- */
    case GP_INT4_LINEAR:
        /* x, w, s, out, rows, cols */
        gemma_int4_linear_body(GP_P(const uint8_t, 1), GP_P(const float, 2),
                               GP_P(const float, 0), GP_P(float, 3),
                               GP_I(4), GP_I(5), 1, 32);
        break;
    case GP_INT4_MULTI4:
        /* x, cols, then (w, s, out, rows) four times */
        gemma_int4_multi4_body(GP_P(const uint8_t, 2), GP_P(const float, 3), GP_P(float, 4), GP_I(5),
                               GP_P(const uint8_t, 6), GP_P(const float, 7), GP_P(float, 8), GP_I(9),
                               GP_P(const uint8_t, 10), GP_P(const float, 11), GP_P(float, 12), GP_I(13),
                               GP_P(const uint8_t, 14), GP_P(const float, 15), GP_P(float, 16), GP_I(17),
                               GP_P(const float, 0), GP_I(1));
        break;
    case GP_RMS_NORM_MULTI4:
        /* x, wn, scratch, cols, eps, then (w, s, out, rows) four times */
        gemma_rms_norm_multi4_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2),
                                   GP_I(3), gp_f(r, e, 4),
                                   GP_P(const uint8_t, 5), GP_P(const float, 6), GP_P(float, 7), GP_I(8),
                                   GP_P(const uint8_t, 9), GP_P(const float, 10), GP_P(float, 11), GP_I(12),
                                   GP_P(const uint8_t, 13), GP_P(const float, 14), GP_P(float, 15), GP_I(16),
                                   GP_P(const uint8_t, 17), GP_P(const float, 18), GP_P(float, 19), GP_I(20));
        break;
    case GP_GELU_MUL_INT4:
        /* g, u, inner, scratch, w, s, out, rows, cols */
        gemma_gelu_mul_int4_body(GP_P(const float, 0), GP_P(const float, 1), GP_I(2),
                                 GP_P(float, 3), GP_P(const uint8_t, 4), GP_P(const float, 5),
                                 GP_P(float, 6), GP_I(7), GP_I(8));
        break;
    case GP_INT4_LINEAR_MT:
        /* Operands: x, w, s, out, rows, cols, tokens[, ldo: the values between
         * the rows of two tokens in out (a part of a group in parts)]. */
        gemma_int4_linear_mt_body(GP_P(const uint8_t, 1), GP_P(const float, 2),
                                  GP_P(const float, 0), GP_P(float, 3),
                                  GP_I(4), GP_I(5), GP_I(6),
                                  r->tag[7] == GP_T_NONE ? 0 : GP_I(7));
        break;
    case GP_INT4_MULTI4_MT: {
        /* Operands: x, cols, tokens, then (w, s, out, rows) four times[, then
         * the ldo of each out, as GP_INT4_LINEAR_MT]. */
        int ldo[4] = {0, 0, 0, 0};
        if (r->tag[19] != GP_T_NONE) {
            for (int k = 0; k < 4; ++k) {
                ldo[k] = GP_I(19 + k);
            }
        }
        gemma_int4_multi4_mt_body(GP_P(const uint8_t, 3), GP_P(const float, 4), GP_P(float, 5), GP_I(6),
                                  GP_P(const uint8_t, 7), GP_P(const float, 8), GP_P(float, 9), GP_I(10),
                                  GP_P(const uint8_t, 11), GP_P(const float, 12), GP_P(float, 13), GP_I(14),
                                  GP_P(const uint8_t, 15), GP_P(const float, 16), GP_P(float, 17), GP_I(18),
                                  GP_P(const float, 0), GP_I(1), GP_I(2), ldo);
        break;
    }
    case GP_GELU_MUL_ROWS: {
        /* Operands: g, u, out, rows, inner. One call for each row, as
         * ops.gelu_mul_rows does, so the tail of each row stays the same. */
        const float *g = GP_P(const float, 0);
        const float *u = GP_P(const float, 1);
        float *out = GP_P(float, 2);
        int rows = GP_I(3), inner = GP_I(4);
        #pragma omp single
        for (int i = 0; i < rows; ++i) {
            rc_gelu_mul_pair(g + (size_t)i * (size_t)inner, u + (size_t)i * (size_t)inner,
                                out + (size_t)i * (size_t)inner, inner);
        }
        break;
    }
    /* ---- attention ---- */
    case GP_QKV_NORM_ROPE:
        /* q, q_w, q_rows, k, k_w, k_rows, v, v_rows, cos, sin, q_heads,
         * k_heads, head_dim, eps */
        gemma_qkv_norm_rope_body(GP_P(float, 0), GP_P(const float, 1), GP_I(2),
                                 GP_P(float, 3), GP_P(const float, 4), GP_I(5),
                                 GP_P(float, 6), GP_I(7),
                                 GP_P(const float, 8), GP_P(const float, 9),
                                 GP_I(10), GP_I(11), GP_I(12), gp_f(r, e, 13));
        break;
    case GP_KV_WRITE: {
        /* Operands: k, v, kd, vd, kqd, ksd, vqd, vsd, n.
         * Store the rows of the float cache and of its int16 copy. A cache
         * with no int16 copy gives null addresses. */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        float *kd = GP_P(float, 2);
        float *vd = GP_P(float, 3);
        int16_t *kqd = GP_P(int16_t, 4);
        float *ksd = GP_P(float, 5);
        int16_t *vqd = GP_P(int16_t, 6);
        float *vsd = GP_P(float, 7);
        int n = GP_I(8);
        if (kd != NULL) {
            #pragma omp single nowait
            {
                memcpy(kd, k, (size_t)n * sizeof(float));
                memcpy(vd, v, (size_t)n * sizeof(float));
            }
        }
        if (kqd != NULL) {
            /* the groups over the team (the rows of a prompt block): each
             * group as one thread did it */
            #pragma omp for schedule(static)
            for (int g = 0; g < n / 32; ++g) {
                ksd[g] = gemma_quant_group32_i16(k + (size_t)g * 32, kqd + (size_t)g * 32);
                vsd[g] = gemma_quant_group32_i16(v + (size_t)g * 32, vqd + (size_t)g * 32);
            }
        } else {
            #pragma omp barrier
        }
        break;
    }
    case GP_KV_WRITE8: {
        /* Operands: k, v, kd, vd, kqd, ksd, vqd, vsd, n.
         * GP_KV_WRITE for the int8 cache (RECLANK_KV_INT8). */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        float *kd = GP_P(float, 2);
        float *vd = GP_P(float, 3);
        int8_t *kqd = GP_P(int8_t, 4);
        float *ksd = GP_P(float, 5);
        int8_t *vqd = GP_P(int8_t, 6);
        float *vsd = GP_P(float, 7);
        int n = GP_I(8);
        #pragma omp single
        {
            if (kd != NULL) {
                memcpy(kd, k, (size_t)n * sizeof(float));
                memcpy(vd, v, (size_t)n * sizeof(float));
            }
            for (int g = 0; g < n / 32; ++g) {
                ksd[g] = gemma_quant_group32_i8(k + (size_t)g * 32, kqd + (size_t)g * 32);
                vsd[g] = gemma_quant_group32_i8(v + (size_t)g * 32, vqd + (size_t)g * 32);
            }
        }
        break;
    }
    case GP_KV_WRITEV8: {
        /* Operands: those of GP_KV_WRITE8. The keys int16, the values int8,
         * no float rows (KVCache kv="k16v8"). */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        int16_t *kqd = GP_P(int16_t, 4);
        float *ksd = GP_P(float, 5);
        int8_t *vqd = GP_P(int8_t, 6);
        float *vsd = GP_P(float, 7);
        int n = GP_I(8);
        #pragma omp single
        for (int g = 0; g < n / 32; ++g) {
            ksd[g] = gemma_quant_group32_i16(k + (size_t)g * 32, kqd + (size_t)g * 32);
            vsd[g] = gemma_quant_group32_i8(v + (size_t)g * 32, vqd + (size_t)g * 32);
        }
        break;
    }
    case GP_KV_WRITETQ: {
        /* Operands: those of GP_KV_WRITE8; the bytes of the TQ6 groups and
         * their norms (tq6_quant_group), no float rows. */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        uint8_t *kqd = GP_P(uint8_t, 4);
        float *ksd = GP_P(float, 5);
        uint8_t *vqd = GP_P(uint8_t, 6);
        float *vsd = GP_P(float, 7);
        int n = GP_I(8);
        #pragma omp for schedule(static)
        for (int g = 0; g < n / 32; ++g) {
            ksd[g] = tq6_quant_group(k + (size_t)g * 32, kqd + (size_t)g * 24);
            vsd[g] = tq6_quant_group(v + (size_t)g * 32, vqd + (size_t)g * 24);
        }
        break;
    }
    case GP_TQ_ROT: {
        /* Operands: x, groups, inverse. The rotation of TQ6 (tq6_rot32) of
         * each group of 32 values of x, in place, or its inverse. */
        float *x = GP_P(float, 0);
        int groups = GP_I(1), inv = GP_I(2);
        #pragma omp for schedule(static)
        for (int g = 0; g < groups; ++g) {
            tq6_rot32(x + (size_t)g * 32, inv);
        }
        break;
    }
    case GP_ATTN_TQ: {
        /* Operands: those of GP_ATTN_QC, over the TQ6 cache; q and out in
         * the rotated form (GP_TQ_ROT). */
        int n1 = (int)GP_I(10);
        if (!gemma_attn_split_tq6_body(GP_P(const float, 0), GP_P(const uint8_t, 1),
                                       GP_P(const float, 2), GP_P(const uint8_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), NULL, &n1, n1, 1)) {
            gp_fail = 1;
        }
        break;
    }
    case GP_ATTN_TQ_MT: {
        /* Operands: those of GP_ATTN_QC_MT, over the TQ6 cache (GP_ATTN_TQ). */
        int tokens = GP_I(10), window = GP_I(13);
        int64_t pos = gp_i(r, e, 11), base = gp_i(r, e, 12);
        int *lo = GP_P(int, 14);
        int *n = GP_P(int, 15);
        #pragma omp single
        for (int j = 0; j < tokens; ++j) {
            gp_rows(pos, base, window, j, lo + j, n + j);
        }
        int nmax = 0;
        for (int j = 0; j < tokens; ++j) {
            nmax = n[j] > nmax ? n[j] : nmax;
        }
        if (!gemma_attn_split_tq6_body(GP_P(const float, 0), GP_P(const uint8_t, 1),
                                       GP_P(const float, 2), GP_P(const uint8_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens)) {
            gp_fail = 1;
        }
        break;
    }
    case GP_ATTN_Q8:
    case GP_ATTN_V8: {
        /* Operands: those of GP_ATTN_QC, over the int8 cache (GP_ATTN_V8: the
         * keys int16). */
        int n1 = (int)GP_I(10);
        if (r->op == GP_ATTN_V8) {
            if (!gemma_attn_split_v8_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                          GP_P(const float, 2), GP_P(const int8_t, 3),
                                          GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                          GP_I(7), GP_I(8), GP_I(9), NULL, &n1, n1, 1)) {
                gp_fail = 1;
            }
        } else if (!gemma_attn_split_i8_body(GP_P(const float, 0), GP_P(const int8_t, 1),
                                      GP_P(const float, 2), GP_P(const int8_t, 3),
                                      GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                      GP_I(7), GP_I(8), GP_I(9), NULL, &n1, n1, 1)) {
            gp_fail = 1;
        }
        break;
    }
    case GP_ATTN_Q8_MT:
    case GP_ATTN_V8_MT: {
        /* Operands: those of GP_ATTN_QC_MT, over the int8 cache (GP_ATTN_V8_MT:
         * the keys int16). */
        int tokens = GP_I(10), window = GP_I(13);
        int64_t pos = gp_i(r, e, 11), base = gp_i(r, e, 12);
        int *lo = GP_P(int, 14);
        int *n = GP_P(int, 15);
        #pragma omp single
        for (int j = 0; j < tokens; ++j) {
            gp_rows(pos, base, window, j, lo + j, n + j);
        }
        int nmax = 0;
        for (int j = 0; j < tokens; ++j) {
            nmax = n[j] > nmax ? n[j] : nmax;
        }
        int ok = r->op == GP_ATTN_V8_MT ?
            gemma_attn_split_v8_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                     GP_P(const float, 2), GP_P(const int8_t, 3),
                                     GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                     GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens) :
            gemma_attn_split_i8_body(GP_P(const float, 0), GP_P(const int8_t, 1),
                                     GP_P(const float, 2), GP_P(const int8_t, 3),
                                     GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                     GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens);
        if (!ok) {
            gp_fail = 1;
        }
        break;
    }
    case GP_ATTN_QC: {
        /* Operands: q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
         * head_dim, n.
         * The attention of one float32 query over the int16 cache: split by
         * keys (gemma_attn_split_i16_body), or one head for each thread. */
        int n1 = (int)GP_I(10);
        if (!gemma_attn_split_i16_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                       GP_P(const float, 2), GP_P(const int16_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), NULL, &n1, n1, 1)) {
            gemma_attn_decode_i16_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                       GP_P(const float, 2), GP_P(const int16_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), GP_I(10));
        }
        break;
    }
    case GP_ATTN_F32:
        /* Operands: q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
         * pos, base, window.
         * The attention of one query over the float cache. k and v point at
         * the row of position base. */
        gemma_attn_decode_f32_body(GP_P(const float, 0), GP_P(const float, 1),
                                   GP_P(const float, 2), GP_P(float, 3), GP_P(float, 4),
                                   GP_I(5), GP_I(6), GP_I(7), GP_I(8),
                                   GP_I(7), GP_I(7), (long)GP_I(6) * GP_I(7),
                                   (long)GP_I(6) * GP_I(7), GP_I(9), GP_I(10), GP_I(11));
        break;
    case GP_ATTN_QC_MT: {
        /* Operands: q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
         * head_dim, tokens, pos, base, window, lo, n.
         * The attention of a group of float32 queries over the int16 cache.
         * The cache addresses point at buffer row 0. lo and n are scratch. */
        int tokens = GP_I(10), window = GP_I(13);
        int64_t pos = gp_i(r, e, 11), base = gp_i(r, e, 12);
        int *lo = GP_P(int, 14);
        int *n = GP_P(int, 15);
        #pragma omp single
        for (int j = 0; j < tokens; ++j) {
            gp_rows(pos, base, window, j, lo + j, n + j);
        }
        int nmax = 0;
        for (int j = 0; j < tokens; ++j) {
            nmax = n[j] > nmax ? n[j] : nmax;
        }
        /* The scores buffer of a group holds tokens * q_heads * (pos +
         * tokens) values (program.step_params), so the split form fits. */
        if (!gemma_attn_split_i16_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                       GP_P(const float, 2), GP_P(const int16_t, 3),
                                       GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                       GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens)) {
            gemma_attn_decode_i16_mt_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                          GP_P(const float, 2), GP_P(const int16_t, 3),
                                          GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                          GP_I(7), GP_I(8), GP_I(9), lo, n, nmax, tokens);
        }
        break;
    }
    case GP_ATTN_F32_MT: {
        /* Operands: q, k, v, scores, out, q_heads, kv_heads, head_dim,
         * tokens, pos, base, window.
         * The attention of a group of queries over the float cache. The code
         * runs one query at a time, as Model._attend_one does it. k and v
         * point at buffer row 0. */
        const float *q = GP_P(const float, 0);
        const float *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2);
        float *out = GP_P(float, 4);
        int q_heads = GP_I(5), kv_heads = GP_I(6), head_dim = GP_I(7);
        int tokens = GP_I(8), window = GP_I(11);
        int64_t pos = gp_i(r, e, 9), base = gp_i(r, e, 10);
        long row = (long)kv_heads * head_dim;
        size_t qd = (size_t)q_heads * (size_t)head_dim;
        for (int j = 0; j < tokens; ++j) {
            int lo, n;
            gp_rows(pos, base, window, j, &lo, &n);
            gemma_attn_decode_f32_body(q + j * qd, k + (size_t)lo * (size_t)row,
                                       v + (size_t)lo * (size_t)row, GP_P(float, 3),
                                       out + j * qd, q_heads, kv_heads, head_dim, n,
                                       head_dim, head_dim, row, row,
                                       (int)(pos + j), (int)(base + lo), window);
        }
        break;
    }
    case GP_QKV_NORM:
        /* Operands: q, q_w, q_rows, k, k_w, k_rows, v, v_rows, head_dim, eps.
         *
         * A layer that reuses the key and the value of an earlier layer gives
         * null for k and v. */
        gemma_qkv_norm_body(GP_P(float, 0), GP_P(const float, 1), GP_I(2),
                            GP_P(float, 3), GP_P(const float, 4), GP_I(5),
                            GP_P(float, 6), GP_I(7), GP_I(8), gp_f(r, e, 9));
        break;
    case GP_ROPE:
        /* Operands: q, q_rows, q_heads, k, k_rows, k_heads, cos, sin,
         * head_dim. */
        gemma_rope_body(GP_P(float, 0), GP_I(1), GP_I(2), GP_P(float, 3), GP_I(4),
                        GP_I(5), GP_P(const float, 6), GP_P(const float, 7), GP_I(8));
        break;
    case GP_KV_WRITE_HEADS: {
        /* Operands: k, v, kbuf, vbuf, head_stride, pos, tokens, kv_heads,
         * head_dim. A float32 cache (the E4B): each head of each token goes
         * to its place (gp_kv_rs: position-major or head-major). */
        const float *k = GP_P(const float, 0);
        const float *v = GP_P(const float, 1);
        float *kb = GP_P(float, 2);
        float *vb = GP_P(float, 3);
        size_t hs = (size_t)gp_i(r, e, 4);
        int64_t pos = gp_i(r, e, 5);
        int tokens = GP_I(6), kv_heads = GP_I(7), head_dim = GP_I(8);
        #pragma omp single
        for (int j = 0; j < tokens; ++j) {
            for (int h = 0; h < kv_heads; ++h) {
                size_t src = ((size_t)j * (size_t)kv_heads + (size_t)h) * (size_t)head_dim;
                size_t dst = (size_t)h * hs + (size_t)(pos + j) * gp_kv_rs(hs, kv_heads, head_dim);
                memcpy(kb + dst, k + src, (size_t)head_dim * sizeof(float));
                memcpy(vb + dst, v + src, (size_t)head_dim * sizeof(float));
            }
        }
        break;
    }
    case GP_ATTN_F32H: {
        /* Operands: q, k, v, scores, out, q_heads, kv_heads, head_dim,
         * tokens, pos, head_stride, window, slide.
         * The attention of one or more queries over the E4B cache. The code
         * runs one query at a time, as E4B.attention does it. k and v point at
         * position 0.
         * With slide, a sliding layer reads only the rows of its window. */
        const float *q = GP_P(const float, 0);
        const float *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2);
        float *out = GP_P(float, 4);
        int q_heads = GP_I(5), kv_heads = GP_I(6), head_dim = GP_I(7);
        int tokens = GP_I(8), window = GP_I(11), slide = GP_I(12);
        int64_t pos = gp_i(r, e, 9);
        long hs = (long)gp_i(r, e, 10);
        long rs = (long)gp_kv_rs((size_t)hs, kv_heads, head_dim);
        size_t qd = (size_t)q_heads * (size_t)head_dim;
        for (int j = 0; j < tokens; ++j) {
            int64_t p = pos + j;
            int64_t lo = (slide && window > 0) ? p - window + 1 : 0;
            if (lo < 0) {
                lo = 0;
            }
            gemma_attn_decode_f32_body(q + j * qd, k + (size_t)lo * (size_t)rs,
                                       v + (size_t)lo * (size_t)rs, GP_P(float, 3),
                                       out + j * qd, q_heads, kv_heads, head_dim,
                                       (int)(p + 1 - lo), hs, hs, rs, rs,
                                       (int)p, (int)lo, window);
        }
        break;
    }
    /* ---- mixture of experts ---- */
    case GP_ROUTER:
        /* x, scale, proj, per_expert, hidden, experts, top_k, eps, hscale,
         * val, idx, r, logits */
        gemma_router_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2),
                          GP_P(const float, 3), GP_I(4), GP_I(5), GP_I(6),
                          gp_f(r, e, 7), gp_f(r, e, 8), GP_P(float, 9), GP_P(int, 10),
                          GP_P(float, 11), GP_P(float, 12));
        break;
    case GP_MOE:
    case GP_MOE_N:
        gp_moe_one(r, e);
        break;
    case GP_ROUTER_MT:
        /* Operands: x, scale, proj, per_expert, hidden, experts, top_k, eps,
         * hscale, val, idx, tokens, r, logits. */
        gemma_router_mt_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2),
                             GP_P(const float, 3), GP_I(4), GP_I(5), GP_I(6),
                             gp_f(r, e, 7), gp_f(r, e, 8), GP_P(float, 9), GP_P(int, 10),
                             GP_I(11), GP_P(float, 12), GP_P(float, 13));
        break;
    case GP_MOE_MT:
        gp_moe_group(r, e);
        break;
    /* ---- programs in parts ---- */
    case GP_XBAR:
        /* bar, parts, then (or null) the weights to prefetch and their bytes */
        gp_xbar(GP_P(int64_t, 0), GP_I(1), GP_P(const uint8_t, 2), gp_i(r, e, 3));
        break;
    case GP_MOE_PART:
        gp_moe_part(r, e);
        break;
    case GP_ATTN_QC_H: {
        /* Operands: q, kq, ks, vq, vs, scores, out, q_heads, kv_heads,
         * head_dim, n, kv_stride, ks_stride, rep.
         * GP_ATTN_QC for some heads. The strides are the values of a whole
         * cache row. rep is the query heads of a key head in the layer. As
         * GP_ATTN_QC: split by keys over the threads of the team, or one
         * head for each thread. The choice is that of GP_ATTN_QC for the
         * whole layer (rep heads of an item), so each head has its bits: a
         * part with fewer query heads of a key head (parts.head_split) does
         * not take the split form where the layer does not. */
        const int hd = GP_I(9);
        const size_t kvs = (size_t)gp_i(r, e, 11);
        if (GP_I(13) * hd > AS_MAXQ ||
            kvs % (size_t)hd != 0 || (size_t)gp_i(r, e, 12) != kvs / 32 ||
            !gemma_attn_split_i16_rows_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                            GP_P(const float, 2), GP_P(const int16_t, 3),
                                            GP_P(const float, 4), GP_P(float, 6), GP_I(7),
                                            GP_I(8), hd, GP_I(10), (int)(kvs / (size_t)hd))) {
            gemma_attn_decode_i16_s_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                         GP_P(const float, 2), GP_P(const int16_t, 3),
                                         GP_P(const float, 4), GP_P(float, 5), GP_P(float, 6),
                                         GP_I(7), GP_I(8), hd, GP_I(10), kvs,
                                         (size_t)gp_i(r, e, 12));
        }
        break;
    }
    case GP_ATTN_F32_H:
        /* Operands: q, k, v, scores, out, q_heads, kv_heads, head_dim, n,
         * pos, base, window, row.
         * GP_ATTN_F32 for some heads. row is the value count of a whole
         * cache row. */
        gemma_attn_decode_f32_body(GP_P(const float, 0), GP_P(const float, 1),
                                   GP_P(const float, 2), GP_P(float, 3), GP_P(float, 4),
                                   GP_I(5), GP_I(6), GP_I(7), GP_I(8),
                                   GP_I(7), GP_I(7), (long)gp_i(r, e, 12),
                                   (long)gp_i(r, e, 12), GP_I(9), GP_I(10), GP_I(11));
        break;
    /* ---- the media encoders: rows split over the threads ---- */
    case GP_ENC_LINEAR: {
        /* x, w, wbf, b, y, n, m, k, imin, imax, omin, omax, scratch, x16:
         * y (n, m) = clamp(clamp(x, imin, imax) W^T + b, omin, omax). The
         * scratch (n k floats, or 0) holds the clamped x. x16 (or 0) is W in
         * groups of 16 rows (KQ_BF16X16 or KQ_F32X16, rc_kq_pack_x16f): the
         * product of many rows (kq_x16f_body). */
        const float *x = GP_P(const float, 0);
        const float *b = GP_P(const float, 3);
        float *y = GP_P(float, 4);
        int n = GP_I(5), m = GP_I(6), k = GP_I(7);
        float imin = gp_f(r, e, 8), imax = gp_f(r, e, 9);
        float omin = gp_f(r, e, 10), omax = gp_f(r, e, 11);
        float *xs = GP_P(float, 12);
        if (xs != NULL) {
            int64_t nk = (int64_t)n * k;
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < nk; ++i) {
                xs[i] = fminf(fmaxf(x[i], imin), imax);
            }
            x = xs;
        }
        const uint8_t *x16 = GP_P(const uint8_t, 13);
        if (x16 != NULL) {
            kq_x16f_body(x16, GP_I(2) ? KQ_BF16X16 : KQ_F32X16, m, k, x, n, y);
        } else if (GP_I(2)) {
            gemma_bf16_linear_body(GP_P(const uint16_t, 1), x, y, m, k, n);
        } else {
            const float *w = GP_P(const float, 1);
            #pragma omp for schedule(static)
            for (int c = 0; c < m; ++c) {
                const float *wc = w + (size_t)c * k;
                for (int t = 0; t < n; ++t) {
                    const float *xt = x + (size_t)t * k;
                    float acc = 0.0f;
                    for (int q = 0; q < k; ++q) {
                        acc += xt[q] * wc[q];
                    }
                    y[(size_t)t * m + c] = acc;
                }
            }
        }
        if (b != NULL || omin > -INFINITY || omax < INFINITY) {
            int64_t nm = (int64_t)n * m;
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < nm; ++i) {
                float v = y[i] + (b != NULL ? b[i % m] : 0.0f);
                y[i] = fminf(fmaxf(v, omin), omax);
            }
        }
        break;
    }
    case GP_ENC_RMS: {
        /* x, w (0 for none), y, rows, cols, eps */
        const float *x = GP_P(const float, 0);
        const float *w = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int rows = GP_I(3), cols = GP_I(4);
        float eps = gp_f(r, e, 5);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            gemma_rms_norm_row(x + (size_t)i * cols, w, y + (size_t)i * cols, cols, eps);
        }
        break;
    }
    case GP_ENC_LNORM: {
        /* x, w, b, y, rows, cols, eps: the LayerNorm of each row (w and b may
         * be 0). y may be x. */
        const float *x = GP_P(const float, 0);
        const float *w = GP_P(const float, 1);
        const float *b = GP_P(const float, 2);
        float *y = GP_P(float, 3);
        int rows = GP_I(4), cols = GP_I(5);
        float eps = gp_f(r, e, 6);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const float *p = x + (size_t)i * cols;
            float *q = y + (size_t)i * cols;
            float s = 0.f;
            for (int j = 0; j < cols; ++j) {
                s += p[j];
            }
            float mean = s / (float)cols, v = 0.f;
            for (int j = 0; j < cols; ++j) {
                float d = p[j] - mean;
                v += d * d;
            }
            float rs = 1.0f / sqrtf(v / (float)cols + eps);
            for (int j = 0; j < cols; ++j) {
                float t = (p[j] - mean) * rs;
                q[j] = t * (w != NULL ? w[j] : 1.0f) + (b != NULL ? b[j] : 0.0f);
            }
        }
        break;
    }
    case GP_Q6K_LINEAR:
        /* w, x, out, rows, cols, tokens: as rc_q6k_linear. A row has the
         * bits of that call, so a range of rows gives the same values. */
        gemma_q6k_linear_body(GP_P(const uint8_t, 0), GP_P(const float, 1), GP_P(float, 2),
                              GP_I(3), GP_I(4), GP_I(5));
        break;
    case GP_KQ_QUANT16:
        /* x, t, cols, xq (int16, t x cols), xs (t x cols / 32) */
        kq_quant16_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_P(int16_t, 3), GP_P(float, 4));
        break;
    case GP_KQ_LINEAR16:
        /* w (KQ_Q4X), rows, cols, xq, xs, t, out (t x rows) */
        kq_linear16_body(GP_P(const uint8_t, 0), GP_I(1), GP_I(2), GP_P(const int16_t, 3),
                         GP_P(const float, 4), GP_I(5), GP_P(float, 6));
        #pragma omp barrier
        break;
    case GP_ATTN_PREFILL_QC:
        /* q, kq, ks, vq, vs (the cache from row 0), positions (t int32),
         * base, window, out, t, n (the buffer rows after the block), q
         * heads, kv heads, head_dim, limit (null, or the last key of each
         * query: media) */
        gemma_attn_prefill_qc_body(GP_P(const float, 0), GP_P(const int16_t, 1),
                                   GP_P(const float, 2), GP_P(const int16_t, 3),
                                   GP_P(const float, 4), GP_P(const int32_t, 5), GP_I(6),
                                   GP_I(7), GP_P(float, 8), GP_I(9), GP_I(10), GP_I(11),
                                   GP_I(12), GP_I(13), GP_P(const int32_t, 14));
        #pragma omp barrier
        break;
    case GP_PART_PREFILL: {
        /* q (t, qh, hd), k, v (t, kvh, hd): the block after the norms and the
         * rope; out (t, qh, hd); kq, ks, vq, vs: the int16 cache of the part,
         * rows of nk = g1 - g0 heads; kf, vf: (rows, nk, hd) float scratch;
         * qp, op: (t, nq, hd) float scratch; positions (t int32); ip (int32):
         * t, start (the buffer row of the block), rows (the buffer rows after
         * the block), base, window, qh, kvh, hd, g0, g1 (the key and value
         * heads of the part), h0, h1 (its query heads, nq = h1 - h0); limit
         * (null, or the last key of each query: media).
         *
         * As KVCache.write, KVCache.read, and rc_attn_prefill for the
         * heads g0 to g1 - 1 and their query heads: each value has the
         * operations of the run of all heads, so the bits are the same. */
        const int32_t *ip = GP_P(const int32_t, 13);
        const int t = ip[0], start = ip[1], rows = ip[2], base = ip[3], window = ip[4];
        const int qh = ip[5], kvh = ip[6], hd = ip[7], g0 = ip[8], g1 = ip[9];
        const int h0 = ip[10], h1 = ip[11];
        const int nk = g1 - g0, nq = h1 - h0, gpr = hd / 32;
        const float *q = GP_P(const float, 0), *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2);
        float *out = GP_P(float, 3);
        int16_t *kq = GP_P(int16_t, 4), *vq = GP_P(int16_t, 6);
        float *ks = GP_P(float, 5), *vs = GP_P(float, 7);
        float *kf = GP_P(float, 8), *vf = GP_P(float, 9);
        float *qp = GP_P(float, 10), *op = GP_P(float, 11);
        const long ng = (long)nk * gpr;
        /* the rows of the block into the cache of the part */
        #pragma omp for schedule(static)
        for (long i = 0; i < (long)t * ng; ++i) {
            const long j = i / ng, r = i % ng;
            const size_t src = ((size_t)j * kvh + g0) * hd + (size_t)r * 32;
            const size_t row = (size_t)(start + j) * ng + r;
            ks[row] = gemma_quant_group32_i16(k + src, kq + row * 32);
            vs[row] = gemma_quant_group32_i16(v + src, vq + row * 32);
        }
        /* the query heads of the part, adjacent */
        #pragma omp for schedule(static)
        for (long i = 0; i < (long)t * nq; ++i) {
            const long j = i / nq, h = i % nq;
            memcpy(qp + (size_t)i * hd, q + ((size_t)j * qh + (size_t)h0 + h) * hd,
                   (size_t)hd * sizeof(float));
        }
        /* the attention over the int16 cache of the part (no float copy) */
        (void)kf;
        (void)vf;
        if (rc_attn_prefill_qc_ok(nq, nk, hd)) {
            gemma_attn_prefill_qc_body(qp, kq, ks, vq, vs, GP_P(const int32_t, 12), base, window,
                                       op, t, rows, nq, nk, hd, GP_P(const int32_t, 14));
        } else {
            #pragma omp for schedule(static)
            for (long g = 0; g < (long)rows * ng; ++g) {
                const float sk = ks[g], sv = vs[g];
                for (int e = 0; e < 32; ++e) {
                    kf[g * 32 + e] = (float)kq[g * 32 + e] * sk;
                    vf[g * 32 + e] = (float)vq[g * 32 + e] * sv;
                }
            }
            gemma_attn_prefill_region_lim(qp, kf, vf, GP_P(const int32_t, 12), base, window, op,
                                          t, rows, nq, nk, hd, GP_P(const int32_t, 14));
        }
        #pragma omp barrier
        #pragma omp for schedule(static)
        for (long i = 0; i < (long)t * nq; ++i) {
            const long j = i / nq, h = i % nq;
            memcpy(out + ((size_t)j * qh + (size_t)h0 + h) * hd, op + (size_t)i * hd,
                   (size_t)hd * sizeof(float));
        }
        break;
    }
    case GP_ENC_GELU: {
        /* x, y, n, erf: y = gelu(x), the tanh form (erf 0) or the erf form */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2);
        int use_erf = GP_I(3);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            float v = x[i];
            y[i] = use_erf ? 0.5f * v * (1.0f + erff(v * 0.7071067811865476f))
                           : 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
        }
        break;
    }
    case GP_ENC_GELU_MUL: {
        /* g, u, y, n: y = gelu_tanh(g) u, in blocks of 4096 values */
        const float *g = GP_P(const float, 0);
        const float *u = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int64_t n = gp_i(r, e, 3);
        int64_t nb = (n + 4095) / 4096;
        #pragma omp for schedule(static)
        for (int64_t bi = 0; bi < nb; ++bi) {
            int64_t o = bi * 4096, len = n - o < 4096 ? n - o : 4096;
            rc_gelu_mul_pair(g + o, u + o, y + o, (int)len);
        }
        break;
    }
    case GP_ENC_ADD: {
        /* x, y, n, s: x += s y */
        float *x = GP_P(float, 0);
        const float *y = GP_P(const float, 1);
        int64_t n = gp_i(r, e, 2);
        float sc = gp_f(r, e, 3);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            x[i] += sc * y[i];
        }
        break;
    }
    case GP_ENC_CLAMP: {
        /* x, y, n, lo, hi: y = clamp(x, lo, hi) */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2);
        float lo = gp_f(r, e, 3), hi = gp_f(r, e, 4);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            y[i] = fminf(fmaxf(x[i], lo), hi);
        }
        break;
    }
    case GP_ENC_BIAS_CLAMP: {
        /* y, b (or 0), rows, cols, lo, hi: y = clamp(y + b, lo, hi) in place */
        float *y = GP_P(float, 0);
        const float *b = GP_P(const float, 1);
        int rows = GP_I(2), cols = GP_I(3);
        float lo = gp_f(r, e, 4), hi = gp_f(r, e, 5);
        int64_t n = (int64_t)rows * cols;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            y[i] = fminf(fmaxf(y[i] + (b != NULL ? b[i % cols] : 0.0f), lo), hi);
        }
        break;
    }
    case GP_ENC_SILU: {
        /* x, y, n: y = x sigmoid(x) */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            y[i] = x[i] / (1.0f + expf(-x[i]));
        }
        break;
    }
    case GP_ENC_SILU_MUL: {
        /* g, u, y, n: y = silu(g) u */
        const float *g = GP_P(const float, 0);
        const float *u = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int64_t n = gp_i(r, e, 3);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            y[i] = g[i] / (1.0f + expf(-g[i])) * u[i];
        }
        break;
    }
    case GP_ENC_MUL_VEC: {
        /* x, vec, y, rows, cols: y = x vec (vec of cols values for each row) */
        const float *x = GP_P(const float, 0);
        const float *vec = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int rows = GP_I(3), cols = GP_I(4);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            for (int c = 0; c < cols; ++c) {
                y[(size_t)i * cols + c] = x[(size_t)i * cols + c] * vec[c];
            }
        }
        break;
    }
    case GP_ENC_GLU: {
        /* x (rows, 2 cols), y (rows, cols), rows, cols: y = a sigmoid(b) with
         * a the first half of a row and b the second */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int rows = GP_I(2), cols = GP_I(3);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            const float *xa = x + (size_t)i * 2 * cols;
            for (int c = 0; c < cols; ++c) {
                y[(size_t)i * cols + c] = xa[c] / (1.0f + expf(-xa[cols + c]));
            }
        }
        break;
    }
    case GP_ENC_DWCONV: {
        /* x (t, c), w (c, kw), y (t, c), t, c, kw: the causal depthwise conv
         * y[i, c] = sum_j w[c, j] x[i - kw + 1 + j, c] (0 before the start) */
        const float *x = GP_P(const float, 0);
        const float *w = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int t = GP_I(3), c = GP_I(4), kw = GP_I(5);
        #pragma omp for schedule(static)
        for (int i = 0; i < t; ++i) {
            for (int ch = 0; ch < c; ++ch) {
                float acc = 0.0f;
                for (int j = 0; j < kw; ++j) {
                    int src = i - kw + 1 + j;
                    if (src >= 0) {
                        acc += w[(size_t)ch * kw + j] * x[(size_t)src * c + ch];
                    }
                }
                y[(size_t)i * c + ch] = acc;
            }
        }
        break;
    }
    case GP_ENC_LOCAL_ATTN: {
        /* q, k, v (t, heads, hd), r (span + 1, heads, hd), valid (t int32), o,
         * t, heads, hd, span, cap: query i sees the keys i - span + 1 .. i;
         * key j of that list adds q r[1 + j] (the relative position); the
         * scores get the soft cap, and an invalid key -1e9 (gemma4a). */
        const float *q = GP_P(const float, 0), *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2), *rr = GP_P(const float, 3);
        const int32_t *valid = GP_P(const int32_t, 4);
        float *o = GP_P(float, 5);
        int t = GP_I(6), heads = GP_I(7), hd = GP_I(8), span = GP_I(9);
        float cap = gp_f(r, e, 10);
        int64_t rows = (int64_t)t * heads;
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            int i = (int)(row / heads), h = (int)(row % heads);
            const float *qi = q + row * hd;
            float sc[64];
            float m = -INFINITY;
            for (int j = 0; j < span && j < 64; ++j) {
                int kp = i - span + 1 + j;
                float sv = -1e9f;
                if (kp >= 0 && valid[kp]) {
                    const float *kk = k + ((size_t)kp * heads + h) * hd;
                    const float *rj = rr + ((size_t)(1 + j) * heads + h) * hd;
                    float a = 0.0f;
                    for (int d = 0; d < hd; ++d) {
                        a += qi[d] * (kk[d] + rj[d]);
                    }
                    sv = tanhf(a / cap) * cap;
                }
                sc[j] = sv;
                m = fmaxf(m, sv);
            }
            float sum = 0.0f;
            for (int j = 0; j < span; ++j) {
                sc[j] = expf(sc[j] - m);
                sum += sc[j];
            }
            float *oi = o + row * hd;
            for (int d = 0; d < hd; ++d) {
                oi[d] = 0.0f;
            }
            for (int j = 0; j < span; ++j) {
                int kp = i - span + 1 + j;
                if (kp < 0) {
                    continue;
                }
                float pj = sc[j] / sum;
                const float *vv = v + ((size_t)kp * heads + h) * hd;
                for (int d = 0; d < hd; ++d) {
                    oi[d] += pj * vv[d];
                }
            }
        }
        break;
    }
    case GP_ENC_ROPE2D: {
        /* x (n, heads, hd), pos (n, 2) int32, inv (hd / 4), n, heads, hd: the
         * axial 2D RoPE of gemma4v in place (see k_enc_rope2d of gpu.cu). */
        float *x = GP_P(float, 0);
        const int32_t *pos = GP_P(const int32_t, 1);
        const float *inv = GP_P(const float, 2);
        int n = GP_I(3), heads = GP_I(4), hd = GP_I(5), q4 = hd / 4, h2 = hd / 2;
        int64_t rows = (int64_t)n * heads;
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            int patch = (int)(row / heads);
            for (int part = 0; part < 2; ++part) {
                float *p = x + row * hd + part * h2;
                for (int j = 0; j < q4; ++j) {
                    float ang = (float)pos[2 * patch + part] * inv[j];
                    float c = cosf(ang), sn = sinf(ang);
                    float a = p[j], bb = p[j + q4];
                    p[j] = a * c - bb * sn;
                    p[j + q4] = bb * c + a * sn;
                }
            }
        }
        break;
    }
    case GP_ENC_ATTN: {
        /* q, k, v, o, n, heads, hd, positions: the attention of every query
         * over every key, scale 1. positions (n int32) holds n - 1 for each
         * query, so the causal limit of the prompt kernel keeps every key. */
        rc_attn_prefill_region(GP_P(const float, 0), GP_P(const float, 1),
                                  GP_P(const float, 2), GP_P(const int32_t, 7), 0, 0,
                                  GP_P(float, 3), GP_I(4), GP_I(4), GP_I(5), GP_I(5), GP_I(6));
        break;
    }
    default:
        /* A code of a GPU record does nothing here; a code that the table
         * does not have is an error (rc_run returns -4). */
        if (!gp_op_known(r->op)) {
            gp_fail = 2;
        }
        break;
    }
}

/* Run the records of one program. The caller opens the parallel region. */
static void gp_exec(const int64_t *prog, int limit)
{
    const int n_env = (int)prog[1];
    int n_code = (int)prog[2];
    const int64_t *env0 = prog + 4;
    const gp_rec *code = (const gp_rec *)(env0 + n_env);
    if (limit >= 0 && limit < n_code) {
        n_code = limit;
    }
    int64_t *e = (int64_t *)malloc((size_t)(n_env > 0 ? n_env : 1) * sizeof(int64_t));
    memcpy(e, env0, (size_t)n_env * sizeof(int64_t));
    for (int pc = 0; pc < n_code; ++pc) {
        gp_step(code + pc, e);
    }
    free(e);
}

/* gp_exec with the time of each record: the first thread of the team adds
 * the ms of record pc to ms[pc]. No barrier is added. Most records end with
 * the barrier of their omp for or omp single, so the time of the first
 * thread is the time of the team; a record with no barrier gives only the
 * share of the first thread, and the next record with a barrier gets the
 * rest. */
static void gp_exec_prof(const int64_t *prog, double *ms)
{
    const int n_env = (int)prog[1];
    const int n_code = (int)prog[2];
    const int64_t *env0 = prog + 4;
    const gp_rec *code = (const gp_rec *)(env0 + n_env);
    int64_t *e = (int64_t *)malloc((size_t)(n_env > 0 ? n_env : 1) * sizeof(int64_t));
    memcpy(e, env0, (size_t)n_env * sizeof(int64_t));
    const int first = omp_get_thread_num() == 0;
    double t0 = omp_get_wtime();
    for (int pc = 0; pc < n_code; ++pc) {
        gp_step(code + pc, e);
        if (first) {
            double t1 = omp_get_wtime();
            ms[pc] += (t1 - t0) * 1e3;
            t0 = t1;
        }
    }
    free(e);
}

/* Run a program with a barrier after each record, and the time of each
 * record in ms (CPU_PLAN.md, phase 0). ms has one value for each record.
 * The barriers add a little time; the times show where the step goes. */
int rc_profile(const int64_t *prog, double *ms)
{
    if (prog[0] != GP_MAGIC) {
        return -1;
    }
    #pragma omp parallel
    {
        const int n_env = (int)prog[1];
        const int n_code = (int)prog[2];
        const int64_t *env0 = prog + 4;
        const gp_rec *code = (const gp_rec *)(env0 + n_env);
        int64_t *e = (int64_t *)malloc((size_t)(n_env > 0 ? n_env : 1) * sizeof(int64_t));
        memcpy(e, env0, (size_t)n_env * sizeof(int64_t));
        #pragma omp barrier
        double t0 = omp_get_wtime();
        for (int pc = 0; pc < n_code; ++pc) {
            gp_step(code + pc, e);
            #pragma omp barrier
            #pragma omp master
            {
                double t1 = omp_get_wtime();
                ms[pc] = (t1 - t0) * 1e3;
                t0 = t1;
            }
        }
        free(e);
    }
    return 0;
}

int rc_run(const int64_t *prog, int limit)
{
    if (prog[0] != GP_MAGIC) {
        return -1;
    }
    gp_fail = 0;
    int was = gp_planned;
    gp_planned = 1;
    #pragma omp parallel
    gp_exec(prog, limit);
    gp_planned = was;
    return gp_fail == 2 ? -4 : gp_fail ? -3 : 0;
}
