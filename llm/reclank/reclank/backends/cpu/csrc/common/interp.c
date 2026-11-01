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

/* ---------- 2D to stereo (ENC_RESIZE, ENC_DIBR) ----------
 * The float operations of the GPU's kernels (encoders.cu) and of the NumPy
 * references, one by one: no contraction into FMAs, so the CPU, the GPU and
 * NumPy give the same bits. */
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")

/* the source pixels and weights of output d of a bilinear resize of n_in to
 * n_out (torch's upsample_bilinear2d, float): mode 1 align_corners */
static void gp_lerp_src(int64_t d, int64_t n_in, int64_t n_out, int mode, int64_t *i0, int64_t *i1, float *l0,
                        float *l1)
{
    float src;
    if (mode == 1) {
        float scale = n_out > 1 ? (float)(n_in - 1) / (float)(n_out - 1) : 0.f;
        src = scale * (float)d;
    } else {
        float scale = (float)n_in / (float)n_out;
        src = scale * ((float)d + 0.5f) - 0.5f;
        if (src < 0.f) {
            src = 0.f;
        }
    }
    int64_t a = (int64_t)src;
    if (a > n_in - 1) {
        a = n_in - 1;
    }
    *i0 = a;
    *i1 = a < n_in - 1 ? a + 1 : a;
    *l1 = src - (float)a;
    *l0 = 1.0f - *l1;
}

static float gp_lerp2(float ly0, float ly1, float lx0, float lx1, float a, float b, float c, float d)
{
    float u = lx0 * a, v = lx1 * b, p = lx0 * c, q = lx1 * d;
    float top = u + v, bot = p + q;
    return ly0 * top + ly1 * bot;
}

/* float32 -> uint32 of the same order, and back */
static uint32_t gp_orderable(float f)
{
    uint32_t u;
    memcpy(&u, &f, 4);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static float gp_unorder(uint32_t u)
{
    uint32_t b = (u & 0x80000000u) ? (u & 0x7fffffffu) : ~u;
    float f;
    memcpy(&f, &b, 4);
    return f;
}

/* source pixel j of a row: its box, and the segment to j + 1 */
static void gp_dibr_splat(const float *sr, uint64_t *kr, int64_t W, int64_t j, float sign, float gap)
{
    float sj = sr[j];
    float p = (float)j + sign * sj;
    uint64_t kb = ((uint64_t)gp_orderable(sj) << 32) | ((uint64_t)j << 1);
    int64_t lo = (int64_t)ceilf(p - 0.5f), hi = (int64_t)ceilf(p + 0.5f);
    for (int64_t X = lo; X < hi; ++X) {
        if (X >= 0 && X < W && kr[X] < kb) {
            kr[X] = kb;
        }
    }
    if (j + 1 >= W) {
        return;
    }
    float sb = sr[j + 1];
    float p1 = (float)(j + 1) + sign * sb;
    float span = p1 - p;
    if (!(span > 0.f && span <= 1.0f + gap)) {
        return;
    }
    lo = (int64_t)ceilf(p);
    hi = (int64_t)ceilf(p1);
    float ds = sb - sj;
    for (int64_t X = lo; X < hi; ++X) {
        if (X < 0 || X >= W) {
            continue;
        }
        float t = ((float)X - p) / span;
        float sx = sj + t * ds;
        uint64_t k = ((uint64_t)gp_orderable(sx) << 32) | ((uint64_t)j << 1) | 1u;
        if (kr[X] < k) {
            kr[X] = k;
        }
    }
}

/* target pixel X of row y: the colour of its winner into tmp, its s into zb */
static void gp_dibr_resolve(const float *img, const float *s, const uint64_t *keys, float *zb, float *tmp, int64_t W,
                            int64_t y, int64_t X, float sign)
{
    uint64_t k = keys[y * W + X];
    float *o = tmp + (y * W + X) * 3;
    if (k == 0) {
        o[0] = o[1] = o[2] = 0.f;
        return;
    }
    zb[y * W + X] = gp_unorder((uint32_t)(k >> 32));
    int64_t j = (int64_t)((k >> 1) & 0x7fffffffu);
    const float *a = img + (y * W + j) * 3;
    if (!(k & 1u)) {
        o[0] = a[0], o[1] = a[1], o[2] = a[2];
        return;
    }
    const float *sr = s + y * W;
    float p0 = (float)j + sign * sr[j], p1 = (float)(j + 1) + sign * sr[j + 1];
    float t = ((float)X - p0) / (p1 - p0);
    const float *b = a + 3;
    for (int c = 0; c < 3; ++c) {
        float d = b[c] - a[c];
        float td = t * d;
        o[c] = a[c] + td;
    }
}

/* a hole pixel: the colour and s of its farther drawn neighbour in the row
 * (the nearest drawn pixel on each side; the smaller key; the only one at an
 * edge; none: black, s = -inf) */
static void gp_dibr_stretch(const uint64_t *keys, float *zb, float *tmp, int64_t W, int64_t y, int64_t X)
{
    const uint64_t *kr = keys + y * W;
    if (kr[X]) {
        return;
    }
    int64_t L = X - 1, R = X + 1;
    while (L >= 0 && !kr[L]) {
        --L;
    }
    while (R < W && !kr[R]) {
        ++R;
    }
    int64_t src = -1;
    if (L >= 0 && (R >= W || kr[L] <= kr[R])) {
        src = L;
    } else if (R < W) {
        src = R;
    }
    float *o = tmp + (y * W + X) * 3;
    if (src < 0) {
        o[0] = o[1] = o[2] = 0.f;
        zb[y * W + X] = -INFINITY;
        return;
    }
    const float *a = tmp + (y * W + src) * 3;
    o[0] = a[0], o[1] = a[1], o[2] = a[2];
    zb[y * W + X] = zb[y * W + src];
}

/* a hole pixel's fill into o: mode 0 the 5-tap horizontal mean of tmp
 * (clamped at the row's ends); mode 1 the mean of the 3 x 3 neighbours (in the
 * image) that are holes or have zb <= its zb + tol, in row order */
static void gp_dibr_fill(const uint64_t *keys, const float *zb, const float *tmp, float *o, int64_t H, int64_t W,
                         int64_t y, int64_t X, int mode, float tol)
{
    float acc[3] = {0.f, 0.f, 0.f};
    if (mode == 0) {
        for (int d = -2; d <= 2; ++d) {
            int64_t xx = X + d < 0 ? 0 : X + d > W - 1 ? W - 1 : X + d;
            const float *t = tmp + (y * W + xx) * 3;
            for (int c = 0; c < 3; ++c) {
                acc[c] = acc[c] + t[c];
            }
        }
        for (int c = 0; c < 3; ++c) {
            o[c] = acc[c] / 5.0f;
        }
        return;
    }
    float lim = zb[y * W + X] + tol;
    float cnt = 0.f;
    for (int64_t dy = -1; dy <= 1; ++dy) {
        int64_t yy = y + dy;
        if (yy < 0 || yy >= H) {
            continue;
        }
        for (int64_t dx = -1; dx <= 1; ++dx) {
            int64_t xx = X + dx;
            if (xx < 0 || xx >= W) {
                continue;
            }
            int64_t q = yy * W + xx;
            if (keys[q] && !(zb[q] <= lim)) {
                continue;
            }
            const float *t = tmp + q * 3;
            for (int c = 0; c < 3; ++c) {
                acc[c] = acc[c] + t[c];
            }
            cnt = cnt + 1.0f;
        }
    }
    for (int c = 0; c < 3; ++c) {
        o[c] = acc[c] / cnt;
    }
}

static float gp_taps2(const float *x, int64_t w, int64_t C, int64_t c, const int32_t *iy, const float *wy,
                      int64_t ky, const int32_t *ix, const float *wx, int64_t kx)
{
    float acc = 0.f;
    for (int64_t k = 0; k < ky; ++k) {
        const float *row = x + (int64_t)iy[k] * w * C + c;
        float h = 0.f;
        for (int64_t j = 0; j < kx; ++j) {
            float p = wx[j] * row[(int64_t)ix[j] * C];
            h = h + p;
        }
        float q = wy[k] * h;
        acc = acc + q;
    }
    return acc;
}

/* ---------- frame interpolation (ENC_WARP, ENC_PRELU, ENC_MIX) ----------
 * The float operations of encoders.cu's kernels and of models/vfi/ref.py one
 * by one (no contraction): the three give the same bits (ENC_MIX's expf
 * apart). */

/* pixel (i, j) of ENC_WARP: torch's CUDA grid_sampler_2d (bilinear, border,
 * align_corners): the grid plus the flow over (size - 1) / 2, unnormalized,
 * clipped, the four corners in bounds summed nw, ne, sw, se */
static void vfi_warp_px(const float *x, const float *fl, float *y, const float *gx, const float *gy, int64_t i,
                        int64_t j, int64_t h, int64_t w, int64_t C, int64_t xs, int64_t xo, int64_t fs, int64_t fo,
                        int64_t ys, int64_t yo, float dx, float dy)
{
    const float *f = fl + (i * w + j) * fs + fo;
    float gxv = gx[j] + f[0] / dx, gyv = gy[i] + f[1] / dy;
    float ix = ((gxv + 1.f) / 2.f) * (float)(w - 1), iy = ((gyv + 1.f) / 2.f) * (float)(h - 1);
    ix = fminf((float)(w - 1), fmaxf(ix, 0.f));
    iy = fminf((float)(h - 1), fmaxf(iy, 0.f));
    float xw = floorf(ix), yn = floorf(iy), xe = xw + 1.f, ysth = yn + 1.f;
    float nw = (xe - ix) * (ysth - iy), ne = (ix - xw) * (ysth - iy);
    float sw = (xe - ix) * (iy - yn), se = (ix - xw) * (iy - yn);
    int64_t x0 = (int64_t)xw, y0 = (int64_t)yn, x1 = x0 + 1, y1 = y0 + 1;
    int okw = x0 >= 0 && x0 < w, oke = x1 >= 0 && x1 < w, okn = y0 >= 0 && y0 < h, oks = y1 >= 0 && y1 < h;
    const float *xb = x + xo;
    float *o = y + (i * w + j) * ys + yo;
    for (int64_t c = 0; c < C; ++c) {
        float acc = 0.f;
        if (okw && okn) {
            acc = acc + xb[(y0 * w + x0) * xs + c] * nw;
        }
        if (oke && okn) {
            acc = acc + xb[(y0 * w + x1) * xs + c] * ne;
        }
        if (okw && oks) {
            acc = acc + xb[(y1 * w + x0) * xs + c] * sw;
        }
        if (oke && oks) {
            acc = acc + xb[(y1 * w + x1) * xs + c] * se;
        }
        o[c] = acc;
    }
}

static void vfi_prelu_row(const float *x, float *y, const float *a, int64_t cols)
{
    for (int64_t c = 0; c < cols; ++c) {
        float v = x[c];
        y[c] = v > 0.f ? v : v * a[c];
    }
}

/* a row of ENC_MIX: a s + b (1 - s), s = 1 / (1 + exp(-m)) */
static void vfi_mix_row(const float *r, float *o, int64_t ao, int64_t bo, int64_t mo, int64_t C)
{
    float s = 1.f / (1.f + expf(-r[mo]));
    float t = 1.f - s;
    for (int64_t c = 0; c < C; ++c) {
        float u = r[ao + c] * s, v = r[bo + c] * t;
        o[c] = u + v;
    }
}

#pragma GCC pop_options

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
         * (the scratch of each part), then with flag 4096 (a prefetch) the next layer's
         * gate, bias, map, and the flags that stop it (the last task's out, the next's
         * in): the flat team of GP_RUN_PARTS (kernels/kq/moe_part.c) */
        kq_moe_part_body(GP_P(const float, 0), GP_P(const int32_t, 1), GP_P(const float, 2), GP_I(3),
                         GP_I(4), GP_I(5), GP_P(const int64_t, 6), GP_I(7), GP_I(8),
                         GP_P(const int32_t, 9), GP_P(const int32_t, 10), GP_P(float, 11), GP_P(float, 12),
                         GP_I(13), GP_I(14), GP_P(const int32_t, 15), GP_I(16), GP_P(const int64_t, 17),
                         GP_P(const uint16_t, 18), GP_P(const float, 19), GP_P(const int32_t, 20),
                         GP_P(const int64_t, 21), GP_P(const int64_t, 22));
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
    case GP_ROUTER_SIG:
        /* logits, t, experts, k, val, idx, bias (or null), scale, eps (or none: 1e-6) */
        gp_router_sig_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_I(3), GP_P(float, 4),
                           GP_P(int32_t, 5), GP_P(const float, 6), gp_f(r, e, 7),
                           r->tag[8] == GP_T_NONE ? 1e-6f : gp_f(r, e, 8));
        break;
    case GP_SHORT_CONV:
        /* in (t x 3 ch: B, C, x), state (kernel - 1 x ch), w (ch x kernel), out, t, ch,
         * kernel, nreal (or none: all) */
        gp_short_conv_body(GP_P(const float, 0), GP_P(float, 1), GP_P(const float, 2),
                           GP_P(float, 3), GP_I(4), GP_I(5), GP_I(6),
                           r->tag[7] == GP_T_NONE ? 0 : GP_I(7));
        break;
    case GP_Q16:
        /* x, out (int16, then the scales), n */
        gp_q16_body(GP_P(const float, 0), GP_P(uint8_t, 1), gp_i(r, e, 2));
        break;
    case GP_DQ16:
        /* in (of GP_Q16), x, n */
        gp_dq16_body(GP_P(const uint8_t, 0), GP_P(float, 1), gp_i(r, e, 2));
        break;
    case GP_NM_MAMBA:
        /* zxd, conv state, conv w, conv b, A_log, D, dt_bias, norm w, ssm state, y, scratch,
         * t, heads, hdim, groups, nstate, kernel, eps, dt_min, nreal (or none: all) */
        nm_mamba_body(GP_P(const float, 0), GP_P(float, 1), GP_P(const float, 2),
                      GP_P(const float, 3), GP_P(const float, 4), GP_P(const float, 5),
                      GP_P(const float, 6), GP_P(const float, 7), GP_P(float, 8), GP_P(float, 9),
                      GP_P(float, 10), GP_I(11), GP_I(12), GP_I(13), GP_I(14), GP_I(15), GP_I(16),
                      gp_f(r, e, 17), gp_f(r, e, 18), r->tag[19] == GP_T_NONE ? 0 : GP_I(19));
        break;
    case GP_RELU2:
        /* x, out, n */
        nm_relu2_body(GP_P(const float, 0), GP_P(float, 1), gp_i(r, e, 2));
        break;
    case GP_NM_MOE:
        /* x (t x L), ids, val, t, k, up, up type, down, down type, L, inner, scratch, out */
        /* (13, or none: x and the act as int16 in two planes; 14, 15 or none: the stacks' copies on node 1) */
        nm_moe_body(GP_P(const float, 0), GP_P(const int32_t, 1), GP_P(const float, 2), GP_I(3),
                    GP_I(4), GP_P(const uint8_t, 5), GP_I(6), GP_P(const uint8_t, 7), GP_I(8),
                    GP_I(9), GP_I(10), GP_P(uint8_t, 11), GP_P(float, 12),
                    r->tag[13] == GP_T_NONE ? 0 : GP_I(13),
                    r->tag[14] != GP_T_NONE ? GP_P(const uint8_t, 14) : NULL,
                    r->tag[15] != GP_T_NONE ? GP_P(const uint8_t, 15) : NULL);
        break;
    case GP_NM_QX2:
        /* x, t, cols, xh, xl, xs, xm (kq_quant_part16) */
        kq_quant_x2(GP_P(const float, 0), GP_I(1), GP_I(2), GP_P(int8_t, 3), GP_P(int8_t, 4), GP_P(float, 5),
                    GP_P(float, 6));
        break;
    case GP_NM_LIN2:
        /* xh, xl, xs, w, type, rows, cols, t, out */
        /* (9 or none: x as int16 for a prompt group, the tiled form) */
        nm_lin2_body(GP_P(const int8_t, 0), GP_P(const int8_t, 1), GP_P(const float, 2), GP_P(const uint8_t, 3),
                     GP_I(4), GP_I(5), GP_I(6), GP_I(7), GP_P(float, 8),
                     r->tag[9] == GP_T_NONE ? NULL : GP_P(int16_t, 9));
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
    case GP_NM_DPREP:
        /* q (or none), k, v, qn, kn, K, V, qout, t, nq, nk, hd, pos, ring, theta, eps */
        nm_dprep_body(r->tag[0] == GP_T_NONE ? NULL : GP_P(const float, 0), GP_P(const float, 1),
                      GP_P(const float, 2), GP_P(const float, 3), GP_P(const float, 4), GP_P(float, 5),
                      GP_P(float, 6), GP_P(float, 7), GP_I(8), GP_I(9), GP_I(10), GP_I(11), gp_i(r, e, 12),
                      GP_I(13), gp_f(r, e, 14), gp_f(r, e, 15));
        break;
    case GP_NM_DATTN:
        /* q, K, V, sinks, out, t, nq, nk, hd, pos0, ring, window, scratch (t nq (window + t) floats) */
        nm_dattn_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2), GP_P(const float, 3),
                      GP_P(float, 4), GP_I(5), GP_I(6), GP_I(7), GP_I(8), gp_i(r, e, 9), GP_I(10), GP_I(11),
                      GP_P(float, 12));
        break;
    case GP_NM_DCONV:
        /* h, dyn, base, out, t, cols, kern, group, side */
        nm_dconv_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2), GP_P(float, 3), GP_I(4),
                      GP_I(5), GP_I(6), GP_I(7), GP_I(8));
        break;
    case GP_NM_LTOPK:
        /* logits, S, V, k, cand, feat */
        if (GP_I(3) > 16) {
            gp_fail = 2;
            break;
        }
        nm_ltopk_body(GP_P(const float, 0), GP_I(1), GP_I(2), GP_I(3), GP_P(int, 4), GP_P(float, 5));
        break;
    case GP_NM_LATT:
        /* qkv, bias, out, n, heads, hd */
        if (GP_I(3) > 64) {
            gp_fail = 2;
            break;
        }
        nm_latt_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_I(3), GP_I(4), GP_I(5));
        break;
    case GP_NM_LROWS:
        /* x, b, y, out, rows, cols, rep, act */
        nm_lrows_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2), GP_P(float, 3), GP_I(4),
                      GP_I(5), GP_I(6), GP_I(7));
        break;
    case GP_NM_LCAT:
        nm_lcat_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(float, 2), GP_I(3), GP_I(4));
        break;
    case GP_NM_LSCORE:
        /* ov, iv, anc, scores, S, k, D, scale, eps */
        nm_lscore_body(GP_P(const float, 0), GP_P(const float, 1), GP_P(const float, 2), GP_P(float, 3), GP_I(4),
                       GP_I(5), GP_I(6), gp_f(r, e, 7), gp_f(r, e, 8));
        break;
    case GP_NM_ADDNORM: {
        /* h, a, b (or none), w, out, rows, cols, eps: h += a + b (the sum of the block's outputs first,
         * as ADD o, b, o then ADD h, o, h), then out = the norm of h (gemma_rms_norm_row) */
        float *h = GP_P(float, 0);
        const float *a = GP_P(const float, 1);
        const float *b = r->tag[2] == GP_T_NONE ? NULL : GP_P(const float, 2);
        const float *w = GP_P(const float, 3);
        float *out = GP_P(float, 4);
        int rows = GP_I(5), cols = GP_I(6);
        float eps = gp_f(r, e, 7);
        #pragma omp for schedule(static)
        for (int i = 0; i < rows; ++i) {
            float *hr = h + (size_t)i * cols;
            const float *ar = a + (size_t)i * cols, *br = b != NULL ? b + (size_t)i * cols : NULL;
            for (int c = 0; c < cols; ++c) {
                hr[c] += br != NULL ? ar[c] + br[c] : ar[c];
            }
            gemma_rms_norm_row(hr, w, out + (size_t)i * cols, cols, eps);
        }
        break;
    }
    case GP_MX8_ACT: {
        /* x, n: each 32 values as dynamic MXFP8 (FP4-Local's vision attention: b12x's quantized
         * mode): e = clamp(ceil(log2(amax / 448)), -127, 127), x = e4m3(x / 2^e) 2^e */
        float *x = GP_P(float, 0);
        int n = GP_I(1);
        #pragma omp for schedule(static)
        for (int g = 0; g < n / 32; ++g) {
            float *v = x + (size_t)32 * g, amax = 0.f;
            for (int i = 0; i < 32; ++i) {
                amax = fmaxf(amax, fabsf(v[i]));
            }
            int ex = 0;
            if (amax > 0.f) {
                int p;
                float m = frexpf(amax / 448.f, &p);
                ex = m == 0.5f ? p - 1 : p;
                ex = ex < -127 ? -127 : ex > 127 ? 127 : ex;
            }
            for (int i = 0; i < 32; ++i) {
                float q = kq_e4m3_rne(fabsf(ldexpf(v[i], -ex)));
                v[i] = ldexpf(v[i] < 0.f ? -q : q, ex);
            }
        }
        break;
    }
    case GP_BF16_ROUND: {
        /* x, n: x rounded to bfloat16 values (to nearest even) in place */
        float *x = GP_P(float, 0);
        int n = GP_I(1);
        #pragma omp for schedule(static)
        for (int i = 0; i < n; ++i) {
            x[i] = kq_bf16r(x[i]);
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
            if (rc_kv_e4m3) {
                /* the FP8 cache of FP4-Local: E4M3 codes, the scales 1 */
                rc_kv_e4m3_codes(k, n, (uint8_t *)kqd);
                rc_kv_e4m3_codes(v, n, (uint8_t *)vqd);
                for (int g = 0; g < n / 32; ++g) {
                    ksd[g] = vsd[g] = 1.f;
                }
            } else {
                for (int g = 0; g < n / 32; ++g) {
                    ksd[g] = gemma_quant_group32_i8(k + (size_t)g * 32, kqd + (size_t)g * 32);
                    vsd[g] = gemma_quant_group32_i8(v + (size_t)g * 32, vqd + (size_t)g * 32);
                }
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
    case GP_REMOTE_POST:
        gp_remote_post(r, e);
        break;
    case GP_REMOTE_JOIN:
        gp_remote_join(r, e);
        break;
    case GP_NM_REMOTE_POST:
        gp_nm_remote_post(r, e);
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
         * product of many rows (kq_x16f_body). wbf 3: F20, w and x16 its
         * KQ_F20X16 (rc_kq_pack_f20x16); wbf 4: BF12, w and x16 its
         * KQ_BF12X16 (rc_kq_pack_bf12x16). */
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
            int wbf = (int)GP_I(2);
            kq_x16f_body(x16, wbf == 4 ? KQ_BF12X16 : wbf == 3 ? KQ_F20X16 : wbf ? KQ_BF16X16 : KQ_F32X16, m, k, x,
                         n, y);
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
        /* g, u, y, n, erf: y = gelu(g) u, the tanh form (erf 0, in blocks of
           4096 values) or the erf form */
        const float *g = GP_P(const float, 0);
        const float *u = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        int64_t n = gp_i(r, e, 3);
        if (GP_I(4)) {
            #pragma omp for schedule(static)
            for (int64_t i = 0; i < n; ++i) {
                float v = g[i];
                y[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f)) * u[i];
            }
            break;
        }
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
    case GP_ENC_ATTN_BIAS: {
        /* q, k, v, o, bias (types, heads, g, g), wtype (n / g), n, heads, hd,
         * g, scale: each query over the g rows of its window, the scores
         * times scale plus the bias of the window's type; g at most 64 */
        const float *q = GP_P(const float, 0);
        const float *k = GP_P(const float, 1);
        const float *v = GP_P(const float, 2);
        float *o = GP_P(float, 3);
        const float *bias = GP_P(const float, 4);
        const int32_t *wt = GP_P(const int32_t, 5);
        int n = GP_I(6), heads = GP_I(7), hd = GP_I(8), g = GP_I(9);
        float scale = (float)gp_f(r, e, 10);
        int64_t stride = (int64_t)heads * hd;
        #pragma omp for schedule(static)
        for (int64_t wh = 0; wh < (int64_t)(n / g) * heads; ++wh) {
            int64_t w = wh / heads;
            int h = (int)(wh % heads);
            const float *bb = bias + ((int64_t)wt[w] * heads + h) * g * g;
            for (int i = 0; i < g; ++i) {
                const float *qi = q + (w * g + i) * stride + (int64_t)h * hd;
                float s[64], mx = -INFINITY, sum = 0.0f;
                for (int j = 0; j < g; ++j) {
                    const float *kj = k + (w * g + j) * stride + (int64_t)h * hd;
                    float acc = 0.0f;
                    for (int d = 0; d < hd; ++d) {
                        acc += qi[d] * kj[d];
                    }
                    s[j] = acc * scale + bb[(int64_t)i * g + j];
                    mx = s[j] > mx ? s[j] : mx;
                }
                for (int j = 0; j < g; ++j) {
                    s[j] = expf(s[j] - mx);
                    sum += s[j];
                }
                float inv = 1.0f / sum;
                float *oi = o + (w * g + i) * stride + (int64_t)h * hd;
                for (int d = 0; d < hd; ++d) {
                    oi[d] = 0.0f;
                }
                for (int j = 0; j < g; ++j) {
                    const float *vj = v + (w * g + j) * stride + (int64_t)h * hd;
                    float p = s[j] * inv;
                    for (int d = 0; d < hd; ++d) {
                        oi[d] += p * vj[d];
                    }
                }
            }
        }
        break;
    }
    case GP_ENC_GATHER: {
        /* x, idx (n, blocks) int32, y, n, blocks, cols: y row i, block j =
         * x row idx[i * blocks + j] (cols values), zeros for an index < 0 */
        const float *x = GP_P(const float, 0);
        const int32_t *idx = GP_P(const int32_t, 1);
        float *y = GP_P(float, 2);
        int64_t n = gp_i(r, e, 3), blocks = gp_i(r, e, 4), cols = gp_i(r, e, 5);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n * blocks; ++i) {
            float *dst = y + i * cols;
            if (idx[i] < 0) {
                memset(dst, 0, (size_t)cols * sizeof(float));
            } else {
                memcpy(dst, x + (int64_t)idx[i] * cols, (size_t)cols * sizeof(float));
            }
        }
        break;
    }
    case GP_ENC_GNORM: {
        /* x, w, b, y, rows, cols, groups, eps, silu: the mean and variance of
         * each group of channels over all the rows (double sums, from the
         * threads' parts), then (x - mean) rstd w + b, SiLU if silu */
        static double *acc = NULL;
        const float *x = GP_P(const float, 0), *w = GP_P(const float, 1), *b = GP_P(const float, 2);
        float *y = GP_P(float, 3);
        int64_t rows = gp_i(r, e, 4);
        int cols = GP_I(5), groups = GP_I(6), silu = GP_I(8), cg = cols / groups;
        float eps = gp_f(r, e, 7);
        #pragma omp single
        {
            acc = (double *)calloc((size_t)groups * 2, sizeof(double));
        }
        double *mine = (double *)calloc((size_t)groups, sizeof(double));
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            const float *p = x + i * cols;
            for (int c = 0; c < cols; ++c) {
                mine[c / cg] += p[c];
            }
        }
        #pragma omp critical
        for (int g = 0; g < groups; ++g) {
            acc[g] += mine[g];
        }
        #pragma omp barrier
        for (int g = 0; g < groups; ++g) {
            mine[g] = 0.0;
        }
        double count = (double)rows * cg;
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            const float *p = x + i * cols;
            for (int c = 0; c < cols; ++c) {
                double d = p[c] - acc[c / cg] / count;
                mine[c / cg] += d * d;
            }
        }
        #pragma omp critical
        for (int g = 0; g < groups; ++g) {
            acc[groups + g] += mine[g];
        }
        #pragma omp barrier
        free(mine);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            const float *p = x + i * cols;
            float *q = y + i * cols;
            for (int c = 0; c < cols; ++c) {
                int g = c / cg;
                float mean = (float)(acc[g] / count);
                float rs = 1.0f / sqrtf((float)(acc[groups + g] / count) + eps);
                float v = (p[c] - mean) * rs * (w ? w[c] : 1.0f) + (b ? b[c] : 0.0f);
                q[c] = silu ? v / (1.0f + expf(-v)) : v;
            }
        }
        #pragma omp barrier
        #pragma omp single
        {
            free(acc);
            acc = NULL;
        }
        break;
    }
    case GP_ENC_ROPE_TAB: {
        /* x, cos, sin, n, heads, hd: the pairs (2j, 2j + 1) of each head of
         * row i turn by the angle of (i, j) (cos and sin: n x hd / 2) */
        float *x = GP_P(float, 0);
        const float *cs = GP_P(const float, 1), *sn = GP_P(const float, 2);
        int64_t n = gp_i(r, e, 3);
        int heads = GP_I(4), hd = GP_I(5), h2 = hd / 2, neox = GP_I(6);
        int step = neox ? 1 : 2, other = neox ? h2 : 1;  /* pair j: (j step, j step + other) */
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            const float *ci = cs + i * h2, *si = sn + i * h2;
            for (int h = 0; h < heads; ++h) {
                float *p = x + (i * heads + h) * hd;
                for (int j = 0; j < h2; ++j) {
                    float a = p[j * step], b = p[j * step + other];
                    p[j * step] = a * ci[j] - b * si[j];
                    p[j * step + other] = b * ci[j] + a * si[j];
                }
            }
        }
        break;
    }
    case GP_ENC_SNAKE: {
        /* x, alpha, beta, y, rows, cols: y = x + (beta + 1e-9)^-1 sin(alpha x)^2 */
        const float *x = GP_P(const float, 0), *al = GP_P(const float, 1), *be = GP_P(const float, 2);
        float *y = GP_P(float, 3);
        int64_t rows = gp_i(r, e, 4);
        int cols = GP_I(5);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            for (int c = 0; c < cols; ++c) {
                float v = x[i * cols + c], s = sinf(al[c] * v);
                y[i * cols + c] = v + 1.0f / (be[c] + 1e-9f) * (s * s);
            }
        }
        break;
    }
    case GP_ENC_ELU: {
        /* x, y, n: y = x above 0, else expm1(x) (torch's ELU of alpha 1) */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            float v = x[i];
            y[i] = v > 0.0f ? v : expm1f(v);
        }
        break;
    }
    case GP_ENC_RESIZE: {
        /* x, y, n, h, w, H, W, C, mode [, xs, xo, ys, yo]: n images of h x w
         * pixels (C values a pixel: the columns xo.. of rows of xs; by
         * default xs = C, xo = 0) to H x W (the columns yo.. of rows of ys).
         * mode 0: bilinear, align_corners False (torch's: src = (d + 0.5) in /
         * out - 0.5, at least 0); 1: bilinear, align_corners True (src = d
         * (in - 1) / (out - 1)); the weights and the sum in torch's order,
         * h0 (w0 x00 + w1 x01) + h1 (w0 x10 + w1 x11). 2: the mean of each
         * block of h / H x w / W pixels (integer factors), summed row by row. */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2), h = gp_i(r, e, 3), w = gp_i(r, e, 4), H = gp_i(r, e, 5), W = gp_i(r, e, 6);
        int64_t C = gp_i(r, e, 7);
        int mode = (int)gp_i(r, e, 8);
        int strided = r->tag[9] != GP_T_NONE;
        int64_t xs = strided ? gp_i(r, e, 9) : C, xo = strided ? gp_i(r, e, 10) : 0;
        int64_t ys = strided ? gp_i(r, e, 11) : C, yo = strided ? gp_i(r, e, 12) : 0;
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < n * H; ++row) {
            int64_t im = row / H, oy = row % H;
            const float *xi = x + im * h * w * xs + xo;
            float *yr = y + row * W * ys + yo;
            if (mode == 2) {
                int64_t fy = h / H, fx = w / W;
                for (int64_t ox = 0; ox < W; ++ox) {
                    for (int64_t c = 0; c < C; ++c) {
                        float acc = 0.f;
                        for (int64_t a = 0; a < fy; ++a) {
                            for (int64_t b = 0; b < fx; ++b) {
                                acc += xi[((oy * fy + a) * w + ox * fx + b) * xs + c];
                            }
                        }
                        yr[ox * ys + c] = acc / (float)(fy * fx);
                    }
                }
                continue;
            }
            int64_t y0, y1;
            float ly0, ly1;
            gp_lerp_src(oy, h, H, mode, &y0, &y1, &ly0, &ly1);
            for (int64_t ox = 0; ox < W; ++ox) {
                int64_t x0, x1;
                float lx0, lx1;
                gp_lerp_src(ox, w, W, mode, &x0, &x1, &lx0, &lx1);
                const float *p00 = xi + (y0 * w + x0) * xs, *p01 = xi + (y0 * w + x1) * xs;
                const float *p10 = xi + (y1 * w + x0) * xs, *p11 = xi + (y1 * w + x1) * xs;
                for (int64_t c = 0; c < C; ++c) {
                    yr[ox * ys + c] = gp_lerp2(ly0, ly1, lx0, lx1, p00[c], p01[c], p10[c], p11[c]);
                }
            }
        }
        break;
    }
    case GP_ENC_WARP: {
        /* x, flow, y, gx, gy, h, w, C, xs, xo, fs, fo, ys, yo, dx, dy: RIFE's
         * backward warp (vfi_warp_px) a pixel at a time */
        const float *x = GP_P(const float, 0), *fl = GP_P(const float, 1);
        float *y = GP_P(float, 2);
        const float *gx = GP_P(const float, 3), *gy = GP_P(const float, 4);
        int64_t h = gp_i(r, e, 5), w = gp_i(r, e, 6), C = gp_i(r, e, 7), xs = gp_i(r, e, 8), xo = gp_i(r, e, 9);
        int64_t fs = gp_i(r, e, 10), fo = gp_i(r, e, 11), ys = gp_i(r, e, 12), yo = gp_i(r, e, 13);
        float dx = gp_f(r, e, 14), dy = gp_f(r, e, 15);
        #pragma omp for schedule(static)
        for (int64_t p = 0; p < h * w; ++p) {
            vfi_warp_px(x, fl, y, gx, gy, p / w, p % w, h, w, C, xs, xo, fs, fo, ys, yo, dx, dy);
        }
        break;
    }
    case GP_ENC_PRELU: {
        /* x, y, rows, cols, a: y = x above 0, else a[col] x (torch's leaky_relu / prelu) */
        const float *x = GP_P(const float, 0), *a = GP_P(const float, 4);
        float *y = GP_P(float, 1);
        int64_t rows = gp_i(r, e, 2), cols = gp_i(r, e, 3);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            vfi_prelu_row(x + i * cols, y + i * cols, a, cols);
        }
        break;
    }
    case GP_ENC_MIX: {
        /* x, y, n, xs, ao, bo, mo, C, ys, yo: a s + b (1 - s), s = sigmoid(m) */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2), xs = gp_i(r, e, 3), ao = gp_i(r, e, 4), bo = gp_i(r, e, 5), mo = gp_i(r, e, 6);
        int64_t C = gp_i(r, e, 7), ys = gp_i(r, e, 8), yo = gp_i(r, e, 9);
        #pragma omp for schedule(static)
        for (int64_t p = 0; p < n; ++p) {
            vfi_mix_row(x + p * xs, y + p * ys + yo, ao, bo, mo, C);
        }
        break;
    }
    case GP_ENC_DIBR: {
        /* img (H W 3), s (H W), keys (2 H W: uint64), zb (H W), tmp (H W 3),
         * out, H, W, off, stride, sign, gap, mode, iters, tol: one eye. Source
         * pixel j of a row lands at j + sign s_j; the segment to j + 1 is drawn
         * unless it folds or stretches past 1 + gap pixels, its pixels X in
         * [p_j, p_j+1) interpolated at t = (X - p_j) / (p_j+1 - p_j); each
         * pixel is also a box [p_j - 0.5, p_j + 0.5). The nearest wins: the
         * largest key (s made orderable: larger s is nearer in both eyes; the
         * source index; segment over box). A hole takes its farther side's
         * colour (the nearest drawn pixel left or right, the one of smaller s;
         * the only one at an edge); then mode 0: a 5-tap horizontal mean over
         * the hole pixels; mode 1: iters Jacobi steps, each hole pixel the
         * mean of its 3 x 3 neighbours that are holes or not nearer than its
         * background by tol. The eye into out[off + y stride + 3 x + c].
         * warp_ref.py is this in NumPy, the GPU's kernels the same float
         * operations. */
        const float *img = GP_P(const float, 0);
        const float *s = GP_P(const float, 1);
        uint64_t *keys = GP_P(uint64_t, 2);
        float *zb = GP_P(float, 3);
        float *tmp = GP_P(float, 4);
        float *out = GP_P(float, 5);
        int64_t H = gp_i(r, e, 6), W = gp_i(r, e, 7), off = gp_i(r, e, 8), stride = gp_i(r, e, 9);
        float sign = (float)gp_i(r, e, 10), gap = gp_f(r, e, 11);
        int mode = (int)gp_i(r, e, 12), iters = (int)gp_i(r, e, 13);
        float tol = gp_f(r, e, 14);
        #pragma omp for schedule(static)
        for (int64_t y = 0; y < H; ++y) {
            uint64_t *kr = keys + y * W;
            const float *sr = s + y * W;
            for (int64_t X = 0; X < W; ++X) {
                kr[X] = 0;
            }
            for (int64_t j = 0; j < W; ++j) {
                gp_dibr_splat(sr, kr, W, j, sign, gap);
            }
        }
        #pragma omp for schedule(static)
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t X = 0; X < W; ++X) {
                gp_dibr_resolve(img, s, keys, zb, tmp, W, y, X, sign);
            }
        }
        #pragma omp for schedule(static)
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t X = 0; X < W; ++X) {
                gp_dibr_stretch(keys, zb, tmp, W, y, X);
            }
        }
        #pragma omp for schedule(static)
        for (int64_t y = 0; y < H; ++y) {
            for (int64_t X = 0; X < W; ++X) {
                const float *t = tmp + (y * W + X) * 3;
                float *o = out + off + y * stride + X * 3;
                o[0] = t[0], o[1] = t[1], o[2] = t[2];
            }
        }
        int steps = mode == 0 ? 1 : iters;
        for (int it = 0; it < steps; ++it) {
            #pragma omp for schedule(static)
            for (int64_t y = 0; y < H; ++y) {
                for (int64_t X = 0; X < W; ++X) {
                    if (keys[y * W + X] == 0) {
                        gp_dibr_fill(keys, zb, tmp, out + off + y * stride + X * 3, H, W, y, X, mode, tol);
                    }
                }
            }
            if (mode == 1) {
                #pragma omp for schedule(static)
                for (int64_t y = 0; y < H; ++y) {
                    for (int64_t X = 0; X < W; ++X) {
                        if (keys[y * W + X] == 0) {
                            const float *o = out + off + y * stride + X * 3;
                            float *t = tmp + (y * W + X) * 3;
                            t[0] = o[0], t[1] = o[1], t[2] = o[2];
                        }
                    }
                }
            }
        }
        break;
    }
    case GP_ENC_RESIZE_TAPS: {
        /* x, y, n, h, w, H, W, C, ky, iy (H ky int32), wy (H ky), kx, ix (W kx), wx (W kx): out
         * pixel (Y, X) channel c = sum over k of wy[Y k] (sum over j of wx[X j] x[iy[Y k],
         * ix[X j], c]), each sum in order (OpenCV's separable resize: the columns first) */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2), h = gp_i(r, e, 3), w = gp_i(r, e, 4), H = gp_i(r, e, 5), W = gp_i(r, e, 6);
        int64_t C = gp_i(r, e, 7), ky = gp_i(r, e, 8), kx = gp_i(r, e, 11);
        const int32_t *iy = GP_P(const int32_t, 9), *ix = GP_P(const int32_t, 12);
        const float *wy = GP_P(const float, 10), *wx = GP_P(const float, 13);
        (void)h;
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < n * H; ++row) {
            int64_t im = row / H, oy = row % H;
            const float *xi = x + im * h * w * C;
            float *yr = y + row * W * C;
            for (int64_t ox = 0; ox < W; ++ox) {
                for (int64_t c = 0; c < C; ++c) {
                    yr[ox * C + c] = gp_taps2(xi, w, C, c, iy + oy * ky, wy + oy * ky, ky, ix + ox * kx, wx + ox * kx, kx);
                }
            }
        }
        break;
    }
    case GP_ENC_UNPACK_U8: {
        /* x (bytes), y, n, d: y[i] = byte i of x / d */
        const uint8_t *x = GP_P(const uint8_t, 0);
        float *y = GP_P(float, 1);
        int64_t n = gp_i(r, e, 2);
        float d = gp_f(r, e, 3);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            y[i] = (float)x[i] / d;
        }
        break;
    }
    case GP_ENC_PACK_U8: {
        /* x, y, n: byte i of y = x[i] rounded to nearest (even), clamped to 0..255 */
        const float *x = GP_P(const float, 0);
        uint8_t *y = GP_P(uint8_t, 1);
        int64_t n = gp_i(r, e, 2);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            float v = nearbyintf(x[i]);
            y[i] = (uint8_t)(v < 0.f ? 0.f : v > 255.f ? 255.f : v);
        }
        break;
    }
    case GP_ENC_DUPUP: {
        /* x, out, h, w, cin, cout, ft, fs: out (fs h x fs w pixels, cout channels) += x (h x w,
         * cin): pixel (y, x) channel o reads x's pixel (y / fs, x / fs), channel (o F + (ft - 1)
         * fs^2 + (y % fs) fs + x % fs) / rep (F = ft fs^2, rep = cout F / cin): diffusers'
         * DupUp3D of one frame (first_chunk: the last of its temporal copies) */
        const float *x = GP_P(const float, 0);
        float *y = GP_P(float, 1);
        int64_t h = gp_i(r, e, 2), w = gp_i(r, e, 3), cin = gp_i(r, e, 4), cout = gp_i(r, e, 5);
        int64_t ft = gp_i(r, e, 6), fs = gp_i(r, e, 7);
        int64_t F = ft * fs * fs, rep = cout * F / cin, W2 = fs * w, rows = fs * h * W2;
        #pragma omp for schedule(static)
        for (int64_t p = 0; p < rows; ++p) {
            int64_t py = p / W2, px = p % W2;
            const float *src = x + ((py / fs) * w + px / fs) * cin;
            int64_t ph = (ft - 1) * fs * fs + (py % fs) * fs + px % fs;
            float *dst = y + p * cout;
            for (int64_t o = 0; o < cout; ++o) {
                dst[o] += src[(o * F + ph) / rep];
            }
        }
        break;
    }
    case GP_ENC_F20_PACK: {
        /* x (rows, cols) float32, y (rows, F20 rows), rows, cols: kq_f20_encode_row (the bits of
         * rc_kq_f32_to_f20's rows) */
        const float *x = GP_P(const float, 0);
        uint8_t *y = GP_P(uint8_t, 1);
        int64_t rows = gp_i(r, e, 2);
        int cols = (int)gp_i(r, e, 3);
        size_t rb = kq_row_bytes(KQ_F20, cols);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            kq_f20_encode_row(x + i * cols, cols, y + i * rb);
        }
        break;
    }
    case GP_ENC_F20_UNPACK: {
        /* x (rows, F20 rows), y (rows, cols) float32, rows, cols: kq_f20_decode_row */
        const uint8_t *x = GP_P(const uint8_t, 0);
        float *y = GP_P(float, 1);
        int64_t rows = gp_i(r, e, 2);
        int cols = (int)gp_i(r, e, 3);
        size_t rb = kq_row_bytes(KQ_F20, cols);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < rows; ++i) {
            kq_f20_decode_row(x + i * rb, cols, y + i * cols);
        }
        break;
    }
    case GP_ENC_SCATTER: {
        /* x, idx (n) int32, y, n, cols: y row idx[i] = x row i (none for an
         * index < 0); the other rows of y stay */
        const float *x = GP_P(const float, 0);
        const int32_t *idx = GP_P(const int32_t, 1);
        float *y = GP_P(float, 2);
        int64_t n = gp_i(r, e, 3), cols = gp_i(r, e, 4);
        #pragma omp for schedule(static)
        for (int64_t i = 0; i < n; ++i) {
            if (idx[i] >= 0) {
                memcpy(y + (int64_t)idx[i] * cols, x + i * cols, (size_t)cols * sizeof(float));
            }
        }
        break;
    }
    case GP_ENC_ATTN_KV: {
        /* q, k, v, o, lo, hi, n, heads, hd: query i over the rows [lo[i],
         * hi[i]) of k and v (rows of heads x hd), scale 1. A (query, head)
         * a task: the scores of a tile of keys, the softmax online. */
        const float *q = GP_P(const float, 0), *k = GP_P(const float, 1), *v = GP_P(const float, 2);
        float *o = GP_P(float, 3);
        const int32_t *lo = GP_P(const int32_t, 4), *hi = GP_P(const int32_t, 5);
        int n = GP_I(6), heads = GP_I(7), hd = GP_I(8), kvh = GP_I(10) ? GP_I(10) : GP_I(7);
        int64_t stride = (int64_t)heads * hd, kstride = (int64_t)kvh * hd;
        #pragma omp for schedule(dynamic, 4)
        for (int64_t t = 0; t < (int64_t)n * heads; ++t) {
            enum { KT = 128 };
            float sc[KT];
            int i = (int)(t / heads), h = (int)(t % heads);
            const float *qi = q + i * stride + (int64_t)h * hd;
            float *oi = o + i * stride + (int64_t)h * hd;
            float m = -INFINITY, l = 0.0f;
            for (int d = 0; d < hd; ++d) {
                oi[d] = 0.0f;
            }
            for (int k0 = lo[i]; k0 < hi[i]; k0 += KT) {
                int nk = hi[i] - k0 < KT ? hi[i] - k0 : KT;
                float mt = -INFINITY;
                for (int j = 0; j < nk; ++j) {
                    const float *kj = k + (int64_t)(k0 + j) * kstride + (int64_t)(h / (heads / kvh)) * hd;
                    float a = 0.0f;
                    for (int d = 0; d < hd; ++d) {
                        a += qi[d] * kj[d];
                    }
                    sc[j] = a;
                    mt = a > mt ? a : mt;
                }
                float mn = m > mt ? m : mt;
                float f = m == -INFINITY ? 0.0f : expf(m - mn);
                l *= f;
                for (int d = 0; d < hd; ++d) {
                    oi[d] *= f;
                }
                for (int j = 0; j < nk; ++j) {
                    float p = expf(sc[j] - mn);
                    const float *vj = v + (int64_t)(k0 + j) * kstride + (int64_t)(h / (heads / kvh)) * hd;
                    l += p;
                    for (int d = 0; d < hd; ++d) {
                        oi[d] += p * vj[d];
                    }
                }
                m = mn;
            }
            float inv = l > 0.0f ? 1.0f / l : 0.0f;
            for (int d = 0; d < hd; ++d) {
                oi[d] *= inv;
            }
        }
        break;
    }
    case GP_ENC_ATTN_REL: {
        /* q, k, v, o, lo, hi, rel (heads, 2 span - 1), n, heads, hd, span: query i over the
         * keys [lo[i], hi[i]), scale 1, each score plus rel[h, j - i + span - 1] (T5's relative
         * position bias of head h at the distance j - i; |j - i| < span). A task: a block of
         * QB queries and a head. The keys of the block's ranges transposed (kt: hd rows of
         * the keys, zeros to a multiple of JC) and their values in rows (vt); the scores of
         * JC keys of a query in a vector, the sums of v of 4 queries at once. */
        enum { QB = 32, JC = 64, DMAX = 256 };
        const float *q = GP_P(const float, 0), *k = GP_P(const float, 1), *v = GP_P(const float, 2);
        float *o = GP_P(float, 3);
        const int32_t *lo = GP_P(const int32_t, 4), *hi = GP_P(const int32_t, 5);
        const float *rel = GP_P(const float, 6);
        int n = GP_I(7), heads = GP_I(8), hd = GP_I(9), span = GP_I(10);
        int64_t stride = (int64_t)heads * hd, width = 2 * (int64_t)span - 1;
        int64_t nqb = (n + QB - 1) / QB, ld = (n + JC - 1) / JC * JC;
        if (hd > DMAX) {
            gp_fail = 3;
            break;
        }
        float *kt = (float *)malloc((size_t)ld * hd * sizeof(float));
        float *vt = (float *)malloc((size_t)n * hd * sizeof(float));
        float *s = (float *)malloc((size_t)QB * ld * sizeof(float));
        #pragma omp for schedule(dynamic, 1)
        for (int64_t t = 0; t < nqb * heads; ++t) {
            int i0 = (int)(t / heads) * QB, h = (int)(t % heads);
            int i1 = i0 + QB < n ? i0 + QB : n;
            int k0 = lo[i0], k1 = hi[i0];
            for (int i = i0 + 1; i < i1; ++i) {
                k0 = lo[i] < k0 ? lo[i] : k0;
                k1 = hi[i] > k1 ? hi[i] : k1;
            }
            int nk = k1 - k0, nkc = (nk + JC - 1) / JC * JC;
            for (int j = 0; j < nk; ++j) {
                const float *kj = k + (int64_t)(k0 + j) * stride + (int64_t)h * hd;
                memcpy(vt + (int64_t)j * hd, v + (int64_t)(k0 + j) * stride + (int64_t)h * hd,
                       (size_t)hd * sizeof(float));
                for (int d = 0; d < hd; ++d) {
                    kt[(int64_t)d * nkc + j] = kj[d];
                }
            }
            for (int d = 0; d < hd; ++d) {
                for (int j = nk; j < nkc; ++j) {
                    kt[(int64_t)d * nkc + j] = 0.0f;
                }
            }
            /* the scores: each query's vector of JC keys over the values of the head */
            for (int jc = 0; jc < nkc; jc += JC) {
                for (int i = i0; i < i1; ++i) {
                    const float *qi = q + (int64_t)i * stride + (int64_t)h * hd;
                    float acc[JC];
                    for (int jj = 0; jj < JC; ++jj) {
                        acc[jj] = 0.0f;
                    }
                    for (int d = 0; d < hd; ++d) {
                        const float qd = qi[d], *kd = kt + (int64_t)d * nkc + jc;
                        for (int jj = 0; jj < JC; ++jj) {
                            acc[jj] += qd * kd[jj];
                        }
                    }
                    memcpy(s + (int64_t)(i - i0) * ld + jc, acc, sizeof(acc));
                }
            }
            /* the bias, the softmax: the weights of the keys out of a query's range 0 */
            float inv[QB];
            for (int i = i0; i < i1; ++i) {
                float *si = s + (int64_t)(i - i0) * ld;
                const float *rb = rel + h * width + (span - 1) - i + k0;
                int a = lo[i] - k0, z = hi[i] - k0;
                float mx = -INFINITY, sum = 0.0f;
                for (int j = a; j < z; ++j) {
                    si[j] += rb[j];
                    mx = si[j] > mx ? si[j] : mx;
                }
                for (int j = 0; j < a; ++j) {
                    si[j] = 0.0f;
                }
                for (int j = a; j < z; ++j) {
                    si[j] = expf(si[j] - mx);
                    sum += si[j];
                }
                for (int j = z; j < nk; ++j) {
                    si[j] = 0.0f;
                }
                inv[i - i0] = sum > 0.0f ? 1.0f / sum : 0.0f;
            }
            /* the sums of v: 4 queries a pass over vt */
            for (int i = i0; i < i1; i += 4) {
                int m = i1 - i < 4 ? i1 - i : 4;
                float acc[4][DMAX];
                for (int r = 0; r < 4; ++r) {
                    for (int d = 0; d < hd; ++d) {
                        acc[r][d] = 0.0f;
                    }
                }
                const float *s0 = s + (int64_t)(i - i0) * ld;
                for (int j = 0; j < nk; ++j) {
                    const float *vj = vt + (int64_t)j * hd;
                    const float p0 = s0[j], p1 = m > 1 ? s0[ld + j] : 0.0f;
                    const float p2 = m > 2 ? s0[2 * ld + j] : 0.0f, p3 = m > 3 ? s0[3 * ld + j] : 0.0f;
                    for (int d = 0; d < hd; ++d) {
                        acc[0][d] += p0 * vj[d];
                        acc[1][d] += p1 * vj[d];
                        acc[2][d] += p2 * vj[d];
                        acc[3][d] += p3 * vj[d];
                    }
                }
                for (int r = 0; r < m; ++r) {
                    float *oi = o + (int64_t)(i + r) * stride + (int64_t)h * hd;
                    for (int d = 0; d < hd; ++d) {
                        oi[d] = acc[r][d] * inv[i + r - i0];
                    }
                }
            }
        }
        free(kt);
        free(vt);
        free(s);
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
    case GP_ENC_ROPE1D: {
        /* x (n, heads, hd), pos (n) int32, inv (hd / 2), n, heads, hd: the 1D
         * RoPE of EmbeddingGemma 2 in place, NEOX halves (see k_enc_rope1d). */
        float *x = GP_P(float, 0);
        const int32_t *pos = GP_P(const int32_t, 1);
        const float *inv = GP_P(const float, 2);
        int n = GP_I(3), heads = GP_I(4), hd = GP_I(5), h2 = hd / 2;
        int64_t rows = (int64_t)n * heads;
        #pragma omp for schedule(static)
        for (int64_t row = 0; row < rows; ++row) {
            float *p = x + row * hd;
            float base = (float)pos[row / heads];
            for (int j = 0; j < h2; ++j) {
                float ang = base * inv[j];
                float c = cosf(ang), sn = sinf(ang);
                float a = p[j], bb = p[j + h2];
                p[j] = a * c - bb * sn;
                p[j + h2] = bb * c + a * sn;
            }
        }
        break;
    }
    case GP_ENC_ATTN_RANGE: {
        /* q, k, v, o, lo, hi, n, heads, kv_heads, hd: the attention of query
         * i over the keys [lo[i], hi[i]), scale 1, query head h on KV head
         * h / (heads / kv_heads) (see k_enc_attn_range). The prompt kernel
         * with a limit: key j counts when pos - window < j <= limit, so pos =
         * lo + window - 1 and limit = hi - 1 for a window of n. lo and hi do
         * not decrease from a query to the next (the kernel's blocks of
         * queries read the keys from the lo of their first to the hi of their
         * last). */
        static int32_t *pw = NULL, *lim = NULL;
        int n = GP_I(6);
        const int32_t *lo = GP_P(const int32_t, 4), *hi = GP_P(const int32_t, 5);
        #pragma omp single
        {
            pw = (int32_t *)malloc((size_t)n * sizeof(int32_t));
            lim = (int32_t *)malloc((size_t)n * sizeof(int32_t));
            for (int i = 0; i < n; ++i) {
                pw[i] = lo[i] + n - 1;
                lim[i] = hi[i] - 1;
            }
        }
        gemma_attn_prefill_region_lim(GP_P(const float, 0), GP_P(const float, 1),
                                      GP_P(const float, 2), pw, 0, n, GP_P(float, 3), n, n,
                                      GP_I(7), GP_I(8), GP_I(9), lim);
        #pragma omp barrier
        #pragma omp single
        {
            free(pw);
            free(lim);
            pw = lim = NULL;
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
        nm_last_record = pc == n_code - 1;
        gp_step(code + pc, e);
    }
    nm_last_record = 0;
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
    /* word 3 of the header: the team of this program (program.Program
     * threads: a program on the CPU in a process of the GPU, whose
     * OMP_NUM_THREADS is 1), or 0 for the default */
    const int n = prog[3] > 0 ? (int)prog[3] : omp_get_max_threads();
    #pragma omp parallel num_threads(n)
    gp_exec(prog, limit);
    gp_planned = was;
    return gp_fail == 2 ? -4 : gp_fail ? -3 : 0;
}
