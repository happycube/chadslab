/* The launch of kernel K<F> (k_qmoe_*_dp4a) for the format f of the tiles
 * of dp4a (kd_format), with the last argument a. */
#define KD_SWITCH(f, K, X2, grid, a, xl)                                                  \
    switch (f) {                                                                          \
    case KD_Q5K: K<KD_Q5K, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break; \
    case KT_Q8R: K<KT_Q8R, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break; \
    case KT_Q80: K<KT_Q80, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break; \
    case KT_Q51: K<KT_Q51, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break; \
    case KT_Q40: K<KT_Q40, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break; \
    case KT_NV4: K<KT_NV4, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break; \
    case KT_NVX: K<KT_NVX, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break; \
    case KT_Q6K: K<KT_Q6K, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break; \
    default: K<KT_Q4K, X2><<<grid, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, a, xl); break;     \
    }

/* RECLANK_DS_ROUTE_FUSE=0: GP_DS_ROUTER, GP_HOT_SPLIT and GP_SIGNAL of a step as three kernels. */
static int ds_route_fuse_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RECLANK_DS_ROUTE_FUSE");
        v = s ? atoi(s) != 0 : 1;
    }
    return v;
}

static int ds_attn_g2_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RECLANK_DS_ATTN_G2");
        v = s ? atoi(s) != 0 : 1;
    }
    return v;
}

/* RECLANK_DS_HOT_TC6: the hot experts by ds_hot_tc6 (the lists, then the held experts' items) for the
 * groups of a prompt (bit 0), the steps (bit 1: t 1) and the verify groups (bit 2: t 2 .. 8; a pair of
 * lanes' steps too); default 5: a step's experts take a pair or two each, and k_ds_hot_gu reads their
 * weights as fast (34.2 ms a step either way); a verify group of 4 to 8 rows 1-4% faster */
static int ds_hot_tc6_on(int64_t t)
{
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RECLANK_DS_HOT_TC6");
        v = s ? atoi(s) : 5;
    }
    return (v >> (t > 8 ? 0 : t == 1 ? 1 : 2)) & 1;
}

static int ds_attn_tc_on(void)
{
    static int v = -1;
    if (v < 0) {
        const char *s = getenv("RECLANK_DS_ATTN_TC");
        v = s ? atoi(s) != 0 : 0;
    }
    return v;
}

/* an expert device's copy (expert_dev.cu, Nemotron) with RECLANK_EX_HG=0: its groups by the rows
 * kernels, not k_nm_hg_* (a test) */
static int gg_nm_hg_off;

/* GP_DS_STAGE_PULL beside the program's kernels: the stream's event after the pick, the pull on gg_side
 * after it, then the pull's event gg_side_ev[ev]; GP_DS_STAGE_JOIN makes the stream wait for that event
 * (in a graph: a branch, joined in the same segment: rc_gpu_load). Made by a thread when it loads or first runs
 * such a program. */
#define GG_SIDE_EV 4
static __thread cudaStream_t gg_side;            /* (each thread its own: the lanes, rc_gpu_lane) */
static __thread cudaEvent_t gg_side_fork, gg_side_ev[GG_SIDE_EV];

static int gg_side_init(void)
{
    if (gg_side != NULL) {
        return 0;
    }
    CK(cudaStreamCreateWithFlags(&gg_side, cudaStreamNonBlocking));
    CK(cudaEventCreateWithFlags(&gg_side_fork, cudaEventDisableTiming));
    for (int k = 0; k < GG_SIDE_EV; ++k) {
        CK(cudaEventCreateWithFlags(&gg_side_ev[k], cudaEventDisableTiming));
    }
    return 0;
}

static int gg_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr, const int64_t *denv)
{
    int bad = 0;
    cudaStream_t s = gg_stream;
    const int T = 256;
    const int W = 32 * ROWS_PER_BLOCK;
    if (r->op == GP_KQ_QUANT) {
        return 0;       /* nothing on the GPU; the int8 x of qx_fuse stays */
    }
    const float *qx_src = gg_qx_src;
    size_t qx_n = gg_qx_n;
    gg_qx_src = NULL;
    if (gg_skip != NULL && r <= gg_skip && r > gg_skip - 4) {
        /* (the records up to gg_skip, which a kernel before ran) */
        if (r == gg_skip) {
            gg_skip = NULL;
        }
        return 0;
    }
    gg_skip = NULL;
    switch (r->op) {
    /* ---- the media encoders: the operands of bf16_linear.c gp_step ---- */
    case GP_ENC_LINEAR: {
        int n = (int)hlit(r, 5, &bad), m = (int)hlit(r, 6, &bad), k = (int)hlit(r, 7, &bad);
        if (enc_tc_on() && gg_tc && g->tc && hlit(r, 2, &bad) && k % 8 == 0 && n > MT_MAX &&
            (size_t)n * k <= g->xh_n) {
            /* bfloat16 W on the tensor cores: x to float16 (with the input
             * clamps), then k_enc_gemm_tc (the bias and the output clamps) */
            size_t nk = (size_t)n * k;
            k_enc_to_half<<<(unsigned)cdiv((int64_t)nk, T), T, 0, s>>>(
                (const float *)(intptr_t)hi(r, g->henv, 0), g->xh, nk, hlitf(r, 8, &bad),
                hlitf(r, 9, &bad));
            k_enc_gemm_tc<<<dim3((unsigned)cdiv(n, T2M), (unsigned)cdiv(m, T2N)), 256, 0, s>>>(
                g->xh, (const uint16_t *)(intptr_t)hi(r, g->henv, 1),
                (const float *)(intptr_t)hi(r, g->henv, 3), (float *)(intptr_t)hi(r, g->henv, 4),
                n, m, k, hlitf(r, 10, &bad), hlitf(r, 11, &bad));
            break;
        }
        dim3 grid((unsigned)cdiv(m, 64), (unsigned)cdiv(n, 64));
        k_enc_gemm<<<grid, 256, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const void *)(intptr_t)hi(r, g->henv, 1),
            (int)hlit(r, 2, &bad), (const float *)(intptr_t)hi(r, g->henv, 3),
            (float *)(intptr_t)hi(r, g->henv, 4), n, m, k, hlitf(r, 8, &bad), hlitf(r, 9, &bad),
            hlitf(r, 10, &bad), hlitf(r, 11, &bad));
        break;
    }
    case GP_ENC_RMS: {
        int rows = (int)hlit(r, 3, &bad);
        k_enc_rms<<<(unsigned)cdiv(rows, 8), 256, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), rows, (int)hlit(r, 4, &bad), hlitf(r, 5, &bad));
        break;
    }
    case GP_ENC_LNORM: {
        int rows = (int)hlit(r, 4, &bad);
        k_enc_lnorm<<<(unsigned)cdiv(rows, 8), 256, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (const float *)(intptr_t)hi(r, g->henv, 2), (float *)(intptr_t)hi(r, g->henv, 3), rows,
            (int)hlit(r, 5, &bad), hlitf(r, 6, &bad));
        break;
    }
    case GP_ENC_GELU: {
        int64_t n = hlit(r, 2, &bad);
        k_enc_gelu<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 1), n,
            (int)hlit(r, 3, &bad));
        break;
    }
    case GP_ENC_GELU_MUL: {
        int64_t n = hlit(r, 3, &bad);
        k_enc_gelu_mul<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), n);
        break;
    }
    case GP_ENC_SILU_MUL: {
        int64_t n = hlit(r, 3, &bad);
        k_enc_silu_mul<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), n);
        break;
    }
    case GP_ENC_ADD: {
        int64_t n = hlit(r, 2, &bad);
        k_enc_add<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1), n,
            hlitf(r, 3, &bad));
        break;
    }
    case GP_ENC_CLAMP: {
        int64_t n = hlit(r, 2, &bad);
        k_enc_clamp<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 1), n,
            hlitf(r, 3, &bad), hlitf(r, 4, &bad));
        break;
    }
    case GP_ENC_BIAS_CLAMP: {
        int rows = (int)hlit(r, 2, &bad), cols = (int)hlit(r, 3, &bad);
        k_enc_bias_clamp<<<(unsigned)cdiv((int64_t)rows * cols, T), T, 0, s>>>(
            (float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            rows, cols, hlitf(r, 4, &bad), hlitf(r, 5, &bad));
        break;
    }
    case GP_ENC_SILU: {
        int64_t n = hlit(r, 2, &bad);
        k_enc_silu<<<(unsigned)cdiv(n, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 1), n);
        break;
    }
    case GP_ENC_MUL_VEC: {
        int rows = (int)hlit(r, 3, &bad), cols = (int)hlit(r, 4, &bad);
        k_enc_mul_vec<<<(unsigned)cdiv((int64_t)rows * cols, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), rows, cols);
        break;
    }
    case GP_ENC_GLU: {
        int rows = (int)hlit(r, 2, &bad), cols = (int)hlit(r, 3, &bad);
        k_enc_glu<<<(unsigned)cdiv((int64_t)rows * cols, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 1),
            rows, cols);
        break;
    }
    case GP_ENC_DWCONV: {
        int t = (int)hlit(r, 3, &bad), c = (int)hlit(r, 4, &bad);
        k_enc_dwconv<<<(unsigned)cdiv((int64_t)t * c, T), T, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (float *)(intptr_t)hi(r, g->henv, 2), t, c, (int)hlit(r, 5, &bad));
        break;
    }
    case GP_ENC_LOCAL_ATTN: {
        int t = (int)hlit(r, 6, &bad), heads = (int)hlit(r, 7, &bad);
        int span = (int)hlit(r, 9, &bad);
        if (span > 32) {
            bad = 1;
            break;
        }
        /* a warp for each (query, head): lane j < span takes key j */
        k_enc_local_attn<<<(unsigned)cdiv((int64_t)t * heads, 4), 128, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (const float *)(intptr_t)hi(r, g->henv, 2), (const float *)(intptr_t)hi(r, g->henv, 3),
            (const int *)(intptr_t)hi(r, g->henv, 4), (float *)(intptr_t)hi(r, g->henv, 5),
            t, heads, (int)hlit(r, 8, &bad), span, hlitf(r, 10, &bad));
        break;
    }
    case GP_ENC_ROPE2D: {
        int n = (int)hlit(r, 3, &bad), heads = (int)hlit(r, 4, &bad), hd = (int)hlit(r, 5, &bad);
        int64_t total = (int64_t)n * heads * 2 * (hd / 4);
        k_enc_rope2d<<<(unsigned)cdiv(total, T), T, 0, s>>>(
            (float *)(intptr_t)hi(r, g->henv, 0), (const int *)(intptr_t)hi(r, g->henv, 1),
            (const float *)(intptr_t)hi(r, g->henv, 2), n, heads, hd);
        break;
    }
    case GP_ENC_ATTN: {
        int n = (int)hlit(r, 4, &bad), heads = (int)hlit(r, 5, &bad), hd = (int)hlit(r, 6, &bad);
        if (hd > 128) {
            bad = 1;
            break;
        }
        size_t nq = (size_t)n * heads * hd;
        if (enc_tc_on() && gg_tc && g->tc && hd % 8 == 0 && 3 * nq <= g->xh_n) {
            /* on the tensor cores: q, k, v to float16 in xh, then k_enc_attn_tc */
            const float *src[3] = {(const float *)(intptr_t)hi(r, g->henv, 0),
                                   (const float *)(intptr_t)hi(r, g->henv, 1),
                                   (const float *)(intptr_t)hi(r, g->henv, 2)};
            for (int i = 0; i < 3; ++i) {
                k_enc_to_half<<<(unsigned)cdiv((int64_t)nq, T), T, 0, s>>>(
                    src[i], g->xh + i * nq, nq, -65504.f, 65504.f);
            }
            dim3 grid((unsigned)cdiv(n, EA_Q), (unsigned)heads);
            const __half *qh = g->xh, *kh = g->xh + nq, *vh = g->xh + 2 * nq;
            float *out = (float *)(intptr_t)hi(r, g->henv, 3);
            int hp = (hd + 15) / 16 * 16;
            if (hp <= 32) {
                k_enc_attn_tc<32><<<grid, 128, 0, s>>>(qh, kh, vh, out, n, heads, hd);
            } else if (hp <= 64) {
                k_enc_attn_tc<64><<<grid, 128, 0, s>>>(qh, kh, vh, out, n, heads, hd);
            } else if (hp <= 80) {
                k_enc_attn_tc<80><<<grid, 128, 0, s>>>(qh, kh, vh, out, n, heads, hd);
            } else if (hp <= 96) {
                k_enc_attn_tc<96><<<grid, 128, 0, s>>>(qh, kh, vh, out, n, heads, hd);
            } else {
                k_enc_attn_tc<128><<<grid, 128, 0, s>>>(qh, kh, vh, out, n, heads, hd);
            }
            break;
        }
        size_t smem = (size_t)(2 * 32 * (hd + 1) + 4 * hd) * sizeof(float);
        k_enc_attn<<<dim3((unsigned)cdiv(n, 4), (unsigned)heads), 128, smem, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), (const float *)(intptr_t)hi(r, g->henv, 1),
            (const float *)(intptr_t)hi(r, g->henv, 2), (float *)(intptr_t)hi(r, g->henv, 3),
            n, heads, hd);
        break;
    }
    case GP_S_MOV: case GP_S_ADD: case GP_S_SUB: case GP_S_MUL: case GP_S_MAX:
    case GP_S_MIN:
        return 0;
    case GP_RMS_NORM:
        k_rms_norm<<<(unsigned)hlit(r, 3, &bad), T, 0, s>>>(dr, denv, 0, 1, 2, 4, 5);
        break;
    case GP_MX8_ACT:
        k_mx8_act<<<(unsigned)cdiv(hlit(r, 1, &bad) / 32, 8), 256, 0, s>>>(dr, denv);
        break;
    case GP_B12X_GLU:
        k_b12x_glu<<<(unsigned)cdiv(hlit(r, 3, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_MOE_B12X_OUT:
        k_moe_b12x_out<<<dim3((unsigned)cdiv(hlit(r, 6, &bad), T), (unsigned)hlit(r, 5, &bad)), T, 0, s>>>(dr, denv);
        break;
    case GP_BF16_ROUND:
        k_bf16_round<<<(unsigned)cdiv(hlit(r, 1, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_ADD:
        k_add<<<(unsigned)cdiv(hlit(r, 3, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_MUL_S:
        k_mul_s<<<(unsigned)cdiv(hlit(r, 3, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_COPY:
        k_copy<<<(unsigned)cdiv(hlit(r, 2, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_GELU:
        k_gelu<<<(unsigned)cdiv(hlit(r, 2, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_MUL:
        k_mul<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_INT4_LINEAR:
        k_int4_linear<<<(unsigned)cdiv(hlit(r, 4, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv, 0, 1, 3, 4, 5);
        break;
    case GP_INT4_MULTI4:
        k_int4_multi4<<<(unsigned)cdiv(multi4_rows(r, 2, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv, 0, 1, 2);
        break;
    case GP_RMS_NORM_MULTI4:
        /* x, wn, scratch, cols, eps, then the matrices from operand 5. */
        k_rms_norm<<<1, T, 0, s>>>(dr, denv, 0, 1, 2, 3, 4);
        k_int4_multi4<<<(unsigned)cdiv(multi4_rows(r, 5, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv, 2, 3, 5);
        break;
    case GP_GELU_MUL_INT4:
        /* g, u, inner, scratch, w, s, out, rows, cols */
        k_gelu_mul<<<(unsigned)cdiv(hlit(r, 2, &bad), T), T, 0, s>>>(dr, denv);
        k_int4_linear<<<(unsigned)cdiv(hlit(r, 7, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv, 3, 4, 6, 7, 8);
        break;
    case GP_BF16_LINEAR: {
        /* x, w, out, rows, cols, tokens */
        int64_t t = hlit(r, 5, &bad), rows = hlit(r, 3, &bad), cols = hlit(r, 4, &bad);
        if (cols % 32 != 0) {
            bad = 1;
        }
        if (t == 1) {
            k_bf16_linear<<<(unsigned)cdiv(rows, ROWS_PER_BLOCK), W, 0, s>>>(dr, denv);
        } else if (t <= MT_MAX) {
            k_mt_gemv_bf16<<<(unsigned)cdiv(rows, ROWS_PER_BLOCK), W, 0, s>>>(
                (const float *)(intptr_t)hlit(r, 0, &bad), (const uint16_t *)(intptr_t)hlit(r, 1, &bad),
                (float *)(intptr_t)hlit(r, 2, &bad), (int)t, (int)rows, (int)cols);
        } else if (gg_tc && g->tc && cols % BHK == 0 && (size_t)t * cols <= g->xh_n) {
            const float *x = (const float *)(intptr_t)hlit(r, 0, &bad);
            size_t n = (size_t)t * cols;
            k_to_half<<<(unsigned)cdiv((int64_t)n / 4 + 1, 256), 256, 0, s>>>(x, g->xh, n);
            k_gemm_bh<<<dim3((unsigned)cdiv(t, T2M), (unsigned)cdiv(rows, T2N)), 256, 0, s>>>(
                g->xh, (const uint16_t *)(intptr_t)hlit(r, 1, &bad),
                (float *)(intptr_t)hlit(r, 2, &bad), (int)t, (int)rows, (int)cols);
        } else {
            k_gemm<2><<<dim3((unsigned)cdiv(rows, GN), (unsigned)cdiv(t, GM)), 256, 0, s>>>(
                (const float *)(intptr_t)hlit(r, 0, &bad), (const void *)(intptr_t)hlit(r, 1, &bad),
                (float *)(intptr_t)hlit(r, 2, &bad), (int)t, (int)rows, (int)cols);
        }
        break;
    }
    case GP_QKV_NORM: {
        qkv_ix ix = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
        k_qkv_norm<<<(unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad) + hlit(r, 7, &bad)),
                     128, 0, s>>>(dr, denv, ix);
        break;
    }
    case GP_ROPE: {
        rope_ix ix = {0, 1, 2, 3, 4, 5, 6, 7, 8};
        k_rope<<<(unsigned)(hlit(r, 1, &bad) + hlit(r, 4, &bad)), 128, 0, s>>>(dr, denv, ix);
        break;
    }
    case GP_QKV_NORM_ROPE: {
        /* q, q_w, q_rows, k, k_w, k_rows, v, v_rows, cos, sin, q_heads,
         * k_heads, head_dim, eps */
        qkv_ix nx = {0, 1, 2, 3, 4, 5, 6, 7, 12, 13};
        rope_ix rx = {0, 2, 10, 3, 5, 11, 8, 9, 12};
        unsigned rows = (unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad) + hlit(r, 7, &bad));
        int64_t hd = hlit(r, 12, &bad);
        static int fuse = -1;
        if (fuse < 0) {
            const char *v = getenv("RECLANK_GPU_QKV_FUSE");
            fuse = !(v && v[0] == '0');
        }
        if (fuse && hd == 256) {
            k_qkv_norm_rope<2><<<rows, 128, 0, s>>>(dr, denv, nx, rx);
        } else if (fuse && hd == 512) {
            k_qkv_norm_rope<4><<<rows, 128, 0, s>>>(dr, denv, nx, rx);
        } else {
            k_qkv_norm<<<rows, 128, 0, s>>>(dr, denv, nx);
            k_rope<<<(unsigned)(hlit(r, 2, &bad) + hlit(r, 5, &bad)), 128, 0, s>>>(dr, denv,
                                                                                  rx);
        }
        break;
    }
    case GP_KV_WRITE:
        k_kv_write<<<(unsigned)cdiv(hlit(r, 8, &bad) / 32, KVW_WARPS), 32 * KVW_WARPS, 0, s>>>(
            dr, denv);
        break;
    case GP_KV_WRITE8:
    case GP_KV_WRITEV8: {
        unsigned grid = (unsigned)cdiv(hlit(r, 8, &bad) / 32, KVW_WARPS);
        if (r->op == GP_KV_WRITE8) {
            k_kv_write8<true><<<grid, 32 * KVW_WARPS, 0, s>>>(dr, denv);
        } else {
            k_kv_write8<false><<<grid, 32 * KVW_WARPS, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_KV_WRITETQ:
        k_kv_write_tq<<<(unsigned)cdiv(hlit(r, 8, &bad) / 32, KVW_WARPS), 32 * KVW_WARPS, 0,
                        s>>>(dr, denv);
        break;
    case GP_TQ_ROT:
        k_tq_rot<<<(unsigned)cdiv(hlit(r, 1, &bad), KVW_WARPS), 32 * KVW_WARPS, 0, s>>>(dr, denv);
        break;
    case GP_ATTN_TQ_MT:
        /* a large group over the TQ6 cache (k_flash_qc_h with TQ; no tensor
         * cores, Pascal: the float32 tiles) */
        if (hlit(r, 10, &bad) <= MT_MAX) {
            bad = 1;
        } else if (!gg_caps.tc) {
            if (!f32t_on() || f32t_launch<false>(r, dr, denv, 0, 0, s, &bad, 1) != 0) {
                bad = 1;
            }
        } else if (flash_tq_launch(r, dr, denv, &bad) != 0) {
            bad = 1;
        }
        break;
    case GP_ATTN_Q8_MT:
    case GP_ATTN_V8_MT:
        /* a large group over the int8 values; a small one has GP_ATTN_Q8 or
         * GP_ATTN_V8 for each query (SplitCompiler.attn_rows_small) */
        if (hlit(r, 10, &bad) <= MT_MAX) {
            bad = 1;
        } else if (!gg_caps.tc) {
            /* no tensor cores (Pascal): the float32 tiles */
            if (!f32t_on() || f32t_launch<false>(r, dr, denv, r->op == GP_ATTN_Q8_MT, 1, s, &bad) != 0) {
                bad = 1;
            }
        } else if ((r->op == GP_ATTN_Q8_MT ? flash_q8_launch<true>(r, dr, denv, &bad)
                                           : flash_q8_launch<false>(r, dr, denv, &bad)) != 0) {
            bad = 1;
        }
        break;
    case GP_KV_WRITE_HEADS:
        k_kv_write_heads<<<(unsigned)(hlit(r, 6, &bad) * hlit(r, 7, &bad)), 128, 0, s>>>(
            dr, denv);
        break;
    case GP_ATTN_F32H:
        if (hlit(r, 8, &bad) == 1) {
            attn_launch(g, r, dr, denv, &bad);
        } else {
            /* A group: every query sees the keys of a window before it, as
             * the decode step does. */
            if (hlit(r, 7, &bad) % FD != 0 || hlit(r, 7, &bad) > 512) {
                bad = 1;
            }
            if (gg_tc && g->tc && (flash3_launch(r, dr, denv, &bad) == 0 ||
                                   flash_tc_launch(r, dr, denv, 1, &bad) == 0)) {
                /* the tensor cores */
            } else if (!gg_caps.tc && f32t_on() && hlit(r, 8, &bad) > MT_MAX &&
                       f32t_launch<true>(r, dr, denv, 0, 0, s, &bad) == 0) {
                /* no tensor cores (Pascal): the float32 tiles */
            } else {
                k_flash_qc_mt<1><<<dim3((unsigned)hlit(r, 5, &bad),
                                        (unsigned)cdiv(hlit(r, 8, &bad), FQ)), 256, 0, s>>>(
                    dr, denv);
            }
        }
        break;
    case GP_ATTN_QSA:
        if (hlit(r, 10, &bad) > 1) {
            int64_t nq = hlit(r, 7, &bad), nk = hlit(r, 8, &bad), rep = nk > 0 ? nq / nk : 0;
            int hg = rep == 12 ? 2 : (rep == 8 ? 1 : 0);
            int64_t form = hlit(r, 15, &bad);
            int tcq = qsa_tc_on();
            if (hlit(r, 9, &bad) != 256 || hg == 0 || nq % nk != 0) {
                bad = 1;
            } else if (tcq && (form == 1 || form == 4) && rep <= 16 && hlit(r, 14, &bad) == 0 &&
                       dense_tc_on()) {
                /* every query sees all the positions (k_attn_dense_tc) */
                static int dattr = 0;
                if (!dattr) {
                    cudaFuncSetAttribute(k_attn_dense_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)QD_SMEM);
                    cudaFuncSetAttribute(k_attn_dense_tc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)QD_SMEM);
                    dattr = 1;
                }
                dim3 grid((unsigned)cdiv(hlit(r, 10, &bad), QD_W), (unsigned)nk);
                if (tcq == 2) {
                    k_attn_dense_tc<0><<<grid, 32 * QD_W, QD_SMEM, s>>>(dr, denv);
                } else {
                    k_attn_dense_tc<1><<<grid, 32 * QD_W, QD_SMEM, s>>>(dr, denv);
                }
            } else if (tcq && (form == 1 || form == 4) && rep <= 16) {
                static int attr = 0;
                if (!attr) {
                    cudaFuncSetAttribute(k_attn_qsa_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)QT_SMEM);
                    cudaFuncSetAttribute(k_attn_qsa_tc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         (int)QT_SMEM);
                    attr = 1;
                }
                dim3 grid((unsigned)hlit(r, 10, &bad), (unsigned)nk);
                if (tcq == 2) {
                    k_attn_qsa_tc<0><<<grid, 128, QT_SMEM, s>>>(dr, denv);
                } else {
                    k_attn_qsa_tc<1><<<grid, 128, QT_SMEM, s>>>(dr, denv);
                }
            } else if (hg == 2) {
                k_attn_qsa_mt<6><<<dim3((unsigned)hlit(r, 10, &bad), (unsigned)(nk * hg)), 128, 0, s>>>(
                    dr, denv, hg);
            } else {
                k_attn_qsa_mt<8><<<dim3((unsigned)hlit(r, 10, &bad), (unsigned)(nk * hg)), 128, 0, s>>>(
                    dr, denv, hg);
            }
            break;
        }
        attn_launch(g, r, dr, denv, &bad);
        break;
    case GP_ATTN_F32:
    case GP_ATTN_QC:
    case GP_ATTN_Q8:
    case GP_ATTN_V8:
    case GP_ATTN_TQ:
        attn_launch(g, r, dr, denv, &bad);
        break;
    case GP_HOT_SPLIT:
        k_hot_split<<<1, 1, 0, s>>>(dr, denv);
        break;
    case GP_HOT_MOE: {
        unsigned t = (unsigned)hlit(r, 15, &bad);
        unsigned k = (unsigned)hlit(r, 10, &bad) * t;
        /* KQ_Q4X (operand 16): a block for each group of 16 rows */
        int per = hlit(r, 16, &bad) == 1 ? 16 : ROWS_PER_BLOCK;
        /* int4 rows: x in shared memory, HOT_XR rows a warp (k_hot_gu_xr) */
        const int xr = hlit(r, 16, &bad) != 1 && hot_xr_on();
        if (xr) {
            k_hot_gu_xr<<<dim3((unsigned)cdiv(hlit(r, 11, &bad), ROWS_PER_BLOCK * HOT_XR), k), W,
                          (size_t)hlit(r, 12, &bad) * 4, s>>>(dr, denv);
        } else {
            k_hot_gu<<<dim3((unsigned)cdiv(hlit(r, 11, &bad), per), k), W, 0, s>>>(dr, denv);
        }
        k_hot_gelu<<<k, T, 0, s>>>(dr, denv);
        if (xr) {
            k_hot_dn_xr<<<dim3((unsigned)cdiv(hlit(r, 13, &bad), ROWS_PER_BLOCK * HOT_XR), k), W,
                          (size_t)hlit(r, 14, &bad) * 4, s>>>(dr, denv);
        } else {
            k_hot_dn<<<dim3((unsigned)cdiv(hlit(r, 13, &bad), per), k), W, 0, s>>>(dr, denv);
        }
        k_hot_sum<<<dim3((unsigned)cdiv(hlit(r, 13, &bad), T), t), T, 0, s>>>(dr, denv);
        break;
    }
    case GP_F32_LINEAR:
        k_f32_linear<<<(unsigned)cdiv(hlit(r, 3, &bad), ROWS_PER_BLOCK), W, 0, s>>>(dr, denv);
        break;
    case GP_DRAFT_HEAD: {
        /* The logits of the centroids come from a GP_F32_LINEAR before it. */
        if (hlit(r, 7, &bad) > 2048) {
            bad = 1;
        }
        k_draft_top<<<1, 1024, 0, s>>>(dr, denv);
        k_draft_sel<<<(unsigned)cdiv(hlit(r, 8, &bad) * hlit(r, 9, &bad), ROWS_PER_BLOCK), W, 0,
                      s>>>(dr, denv);
        k_draft_best<<<1, 1024, 0, s>>>(dr, denv);
        break;
    }
    case GP_ARGMAX:
        k_argmax<<<1, 1024, 0, s>>>(dr, denv);
        break;
    case GP_SOFTCAP:
        k_softcap<<<(unsigned)cdiv(hlit(r, 1, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_COUNT:
        k_count<<<4, T, 0, s>>>(dr, denv);
        break;
    case GP_ADD_NORM: {
        int64_t rows = hlit(r, 4, &bad), cols = hlit(r, 5, &bad);
        int64_t o2 = hi(r, g->henv, 9);
        int ok = cols % 4 == 0 && cols <= 4 * AN_V * T && an_v_on();
        for (int k = 0; k < 4 && ok; ++k) {
            ok = al16(hi(r, g->henv, k));
        }
        ok = ok && al16(o2) && al16(hi(r, g->henv, 8));
        if (ok) {
            float *xs = NULL, *xsum = NULL;
            int8_t *q = cols % 32 != 0 ? NULL :
                        qx_fuse(g, r, (const void *)(intptr_t)(o2 ? o2 : hi(r, g->henv, 3)),
                                (size_t)(rows * cols), &xs, &xsum);
            k_add_norm_v<<<(unsigned)rows, T, 0, s>>>(dr, denv, q, xs, xsum);
        } else {
            k_add_norm<<<(unsigned)rows, T, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_HOT_SPLIT_MT:
        k_hot_split_mt<<<(unsigned)cdiv(hlit(r, 5, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_INT4_LINEAR_MT: {
        /* x, w, s, out, rows, cols, t. The record before can have written
         * the int8 x already (qx_fuse). */
        size_t m = 0;
        const float *xq = q8_gemm_input(g, r, &m);
        gemm_launch(g, r, dr, denv, 0, 1, 3, 4, 5, 6, &bad,
                    !(xq != NULL && xq == qx_src && m == qx_n));
        break;
    }
    case GP_INT4_MULTI4_MT: {
        /* x, cols, t, then (w, s, out, rows) for up to four matrices */
        size_t m = 0;
        const float *xq = q8_gemm_input(g, r, &m);
        int first = !(xq != NULL && xq == qx_src && m == qx_n);
        for (int k = 0; k < 4; ++k) {
            if (r->v[3 + 4 * k] != 0) {
                gemm_launch(g, r, dr, denv, 0, 3 + 4 * k, 5 + 4 * k, 6 + 4 * k, 1, 2, &bad,
                            first);
                first = 0;
            }
        }
        break;
    }
    case GP_GELU_MUL_ROWS: {
        int64_t n = hlit(r, 3, &bad) * hlit(r, 4, &bad);
        if (n % 4 == 0 && an_v_on() && al16(hi(r, g->henv, 0)) && al16(hi(r, g->henv, 1)) &&
            al16(hi(r, g->henv, 2))) {
            float *xs = NULL, *xsum = NULL;
            int8_t *q = qx_fuse(g, r, (const void *)(intptr_t)hi(r, g->henv, 2), (size_t)n, &xs,
                                &xsum);
            k_gelu_mul_rows_v<<<(unsigned)cdiv(n / 4, T), T, 0, s>>>(dr, denv, q, xs, xsum);
        } else {
            k_gelu_mul_rows<<<(unsigned)cdiv(n, T), T, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_ATTN_QC_MT:
        if (hlit(r, 9, &bad) > 512 || hlit(r, 9, &bad) % 16 != 0) {
            bad = 1;
        }
        if (hlit(r, 10, &bad) > MT_MAX && gg_tc && (g->tc || g->atc) &&
            (flash_qc_launch(r, dr, denv, &bad) == 0 ||
             flash_tc_launch(r, dr, denv, 0, &bad) == 0)) {
            /* the tensor cores */
        } else if (hlit(r, 10, &bad) > MT_MAX && !gg_caps.tc && f32t_on() &&
                   f32t_launch<false>(r, dr, denv, 0, 0, s, &bad) == 0) {
            /* no tensor cores (Pascal): the float32 tiles */
        } else if (hlit(r, 10, &bad) > MT_MAX && hlit(r, 9, &bad) % FD == 0) {
            k_flash_qc_mt<0><<<dim3((unsigned)hlit(r, 7, &bad),
                                    (unsigned)cdiv(hlit(r, 10, &bad), FQ)), 256, 0, s>>>(dr, denv);
        } else {
            k_attn_qc_mt<<<dim3((unsigned)hlit(r, 7, &bad),
                                (unsigned)cdiv(hlit(r, 10, &bad), QT)), QT * 16, 0, s>>>(dr, denv);
        }
        break;
    case GP_ROUTER_MT: {
        int64_t t = hlit(r, 11, &bad), ex = hlit(r, 5, &bad);
        k_router_norm_mt<<<(unsigned)t, T, 0, s>>>(dr, denv);
        if (t <= MT_MAX) {
            k_router_logits_mt<<<dim3((unsigned)cdiv(ex, ROWS_PER_BLOCK), (unsigned)t), W, 0, s>>>(
                dr, denv);
        } else {
            k_gemm<1><<<dim3((unsigned)cdiv(ex, GN), (unsigned)cdiv(t, GM)), 256, 0, s>>>(
                (const float *)(intptr_t)r->v[12], (const void *)(intptr_t)r->v[2],
                (float *)(intptr_t)r->v[13], (int)t, (int)ex, (int)hlit(r, 4, &bad));
        }
        k_router_top_mt<<<(unsigned)t, 32, 0, s>>>(dr, denv);
        break;
    }
    case GP_MOE_GPU: {
        /* The pointers of the weights and of the scratch are literals. */
        int64_t t = hlit(r, 3, &bad), k = hlit(r, 4, &bad), pairs = t * k;
        int gu_rows = (int)hlit(r, 7, &bad), cols = (int)hlit(r, 8, &bad);
        int dn_rows = (int)hlit(r, 9, &bad), inner = (int)hlit(r, 10, &bad);
        unsigned max_tiles = (unsigned)(cdiv(pairs, GM) + 128);
        const int64_t *gu = (const int64_t *)(intptr_t)hlit(r, 21, &bad);
        const int64_t *dn = (const int64_t *)(intptr_t)hlit(r, 22, &bad);
        const float *h = (const float *)(intptr_t)hlit(r, 0, &bad);
        float *act = (float *)(intptr_t)hlit(r, 17, &bad);
        float *act2 = (float *)(intptr_t)hlit(r, 18, &bad);
        float *de = (float *)(intptr_t)hlit(r, 19, &bad);
        k_moe_sort<<<1, 1024, 0, s>>>(dr, denv);
        k_moe_tiles<<<1, 1, 0, s>>>(dr, denv);
        if (hlit(r, 5, &bad) == 1) {
            /* KQ_Q4X experts: int8 x on the tensor cores (k_moe_gemm_q4x), as
             * the int8 x of the prompt of the CPU (the flag tc of the program
             * is about the float16 products of the other layout) */
            int kd = !kt_on() && kt_dp4a_on();      /* Pascal: the tiles on dp4a */
            if (!((kt_on() || kd) && gu_rows % KT_BN == 0 && dn_rows % KT_BN == 0 &&
                  cols % 32 == 0 && inner % 32 == 0 && (size_t)t * cols <= g->kqx_n &&
                  (size_t)pairs * inner <= g->kqx_n)) {
                snprintf(gg_error, sizeof(gg_error),
                         "GP_MOE_GPU with KQ_Q4X needs the tensor cores or dp4a (kt %d, rows %d %d, "
                         "x %zu %zu of %zu)", kt_on(), gu_rows, dn_rows, (size_t)(t * cols),
                         (size_t)(pairs * inner), g->kqx_n);
                return -1;
            }
            float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
            if (kd) {
                int x2 = g->i8 == 2;
                dim3 ggu((unsigned)(gu_rows / KT_BN), max_tiles), gdn((unsigned)(dn_rows / KT_BN), max_tiles);
                if (x2) {
                    kt_quant_x2(g, h, (size_t)(t * cols));
                    k_moe_gemm_q4x_dp4a<1><<<ggu, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, 1, gu,
                                                               gu_rows, cols, act, kt_lo(g));
                } else {
                    kt_quant(g, h, (size_t)(t * cols));
                    k_moe_gemm_q4x_dp4a<0><<<ggu, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, 1, gu,
                                                               gu_rows, cols, act, NULL);
                }
                k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
                if (x2) {
                    kt_quant_x2(g, act2, (size_t)(pairs * inner));
                    k_moe_gemm_q4x_dp4a<1><<<gdn, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, 0, dn,
                                                               dn_rows, inner, de, kt_lo(g));
                } else {
                    kt_quant(g, act2, (size_t)(pairs * inner));
                    k_moe_gemm_q4x_dp4a<0><<<gdn, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, 0, dn,
                                                               dn_rows, inner, de, NULL);
                }
            } else if (g->i8 == 2) {
                /* the int16 form of h and of the GELU (k_quant_x2) */
                size_t sm = KT_SMEM_X2(64, 2);
                kt_attr();
                kt_quant_x2(g, h, (size_t)(t * cols));
                k_moe_gemm_q4x<1><<<dim3((unsigned)(gu_rows / KT_BN), max_tiles), 256, sm, s>>>(
                    dr, denv, g->kqx, xs, xsum, 1, gu, gu_rows, cols, act, kt_lo(g));
                k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
                kt_quant_x2(g, act2, (size_t)(pairs * inner));
                k_moe_gemm_q4x<1><<<dim3((unsigned)(dn_rows / KT_BN), max_tiles), 256, sm, s>>>(
                    dr, denv, g->kqx, xs, xsum, 0, dn, dn_rows, inner, de, kt_lo(g));
            } else {
                size_t sm = KT_SMEM(64, 3);
                kt_attr();
                kt_quant(g, h, (size_t)(t * cols));
                k_moe_gemm_q4x<0><<<dim3((unsigned)(gu_rows / KT_BN), max_tiles), 256, sm, s>>>(
                    dr, denv, g->kqx, xs, xsum, 1, gu, gu_rows, cols, act, NULL);
                k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
                kt_quant(g, act2, (size_t)(pairs * inner));
                k_moe_gemm_q4x<0><<<dim3((unsigned)(dn_rows / KT_BN), max_tiles), 256, sm, s>>>(
                    dr, denv, g->kqx, xs, xsum, 0, dn, dn_rows, inner, de, NULL);
            }
        } else if (gg_tc && g->tc && cols % TK == 0 && inner % TK == 0 &&
            (size_t)t * cols <= g->xh_n && (size_t)pairs * inner <= g->xh_n) {
            size_t n1 = (size_t)t * cols, n2 = (size_t)pairs * inner;
            k_to_half<<<(unsigned)cdiv((int64_t)n1 / 4 + 1, 256), 256, 0, s>>>(h, g->xh, n1);
            k_moe_gemm_tc<<<dim3((unsigned)cdiv(gu_rows, TN), max_tiles), 128, 0, s>>>(
                dr, denv, g->xh, 1, gu, gu_rows, cols, act);
            k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
            k_to_half<<<(unsigned)cdiv((int64_t)n2 / 4 + 1, 256), 256, 0, s>>>(act2, g->xh, n2);
            k_moe_gemm_tc<<<dim3((unsigned)cdiv(dn_rows, TN), max_tiles), 128, 0, s>>>(
                dr, denv, g->xh, 0, dn, dn_rows, inner, de);
        } else if (!kt_on() && kt_dp4a_on() && moe_q40_mode() > 0 && cols % 32 == 0 && inner % 32 == 0 &&
                   (size_t)t * cols <= g->kqx_n && (size_t)pairs * inner <= g->kqx_n &&
                   (moe_q40_mode() == 1 || g->kqx_lo)) {
            /* Pascal: the int4 experts on dp4a (k_moe_gemm_q40_dp4a), x in the
             * int16 form (x2) or int8 */
            float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
            const int x2 = moe_q40_mode() == 2;
            dim3 ggu((unsigned)cdiv(gu_rows, KT_BN), max_tiles), gdn((unsigned)cdiv(dn_rows, KT_BN), max_tiles);
            if (x2 && q40_tile_on()) {
                /* the tile of its own (k_moe_q40_x2): the bits of kt_tile_dp4a */
                kt_quant_x2(g, h, (size_t)(t * cols));
                k_moe_q40_x2<<<ggu, 256, 0, s>>>(dr, denv, g->kqx, xs, 1, gu, gu_rows, cols, act, kt_lo(g));
            } else if (x2) {
                kt_quant_x2(g, h, (size_t)(t * cols));
                k_moe_gemm_q40_dp4a<1><<<ggu, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, 1, gu, gu_rows, cols, act,
                                                           kt_lo(g));
            } else {
                kt_quant(g, h, (size_t)(t * cols));
                k_moe_gemm_q40_dp4a<0><<<ggu, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, 1, gu, gu_rows, cols, act,
                                                           NULL);
            }
            k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
            if (x2 && q40_tile_on()) {
                kt_quant_x2(g, act2, (size_t)(pairs * inner));
                k_moe_q40_x2<<<gdn, 256, 0, s>>>(dr, denv, g->kqx, xs, 0, dn, dn_rows, inner, de, kt_lo(g));
            } else if (x2) {
                kt_quant_x2(g, act2, (size_t)(pairs * inner));
                k_moe_gemm_q40_dp4a<1><<<gdn, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, 0, dn, dn_rows, inner, de,
                                                           kt_lo(g));
            } else {
                kt_quant(g, act2, (size_t)(pairs * inner));
                k_moe_gemm_q40_dp4a<0><<<gdn, 256, 0, s>>>(dr, denv, g->kqx, xs, xsum, 0, dn, dn_rows, inner, de,
                                                           NULL);
            }
        } else if (gemm8_on() && cols % 32 == 0 && inner % 32 == 0) {
            /* no int8 products (Maxwell): the float tile of k_gemm8 */
            k_moe_gemm8<<<dim3((unsigned)cdiv(gu_rows, G8_B), max_tiles), 256, 0, s>>>(
                dr, denv, h, 1, gu, gu_rows, cols, act);
            k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
            k_moe_gemm8<<<dim3((unsigned)cdiv(dn_rows, G8_B), max_tiles), 256, 0, s>>>(
                dr, denv, act2, 0, dn, dn_rows, inner, de);
        } else {
            k_moe_gemm<<<dim3((unsigned)cdiv(gu_rows, GN), max_tiles), 256, 0, s>>>(
                dr, denv, h, 1, gu, gu_rows, cols, act);
            k_moe_gelu<<<(unsigned)cdiv(pairs * inner, T), T, 0, s>>>(dr, denv);
            k_moe_gemm<<<dim3((unsigned)cdiv(dn_rows, GN), max_tiles), 256, 0, s>>>(
                dr, denv, act2, 0, dn, dn_rows, inner, de);
        }
        k_moe_sum<<<(unsigned)cdiv(t * dn_rows, T), T, 0, s>>>(dr, denv);
        break;
    }
    case GP_ROUTER:
        k_router_norm<<<1, T, 0, s>>>(dr, denv);
        k_router_logits<<<(unsigned)cdiv(hlit(r, 5, &bad), ROWS_PER_BLOCK), W, 0, s>>>(
            dr, denv);
        k_router_top<<<1, 32, 0, s>>>(dr, denv);
        break;
    case GP_KQ_QUANT:
        /* The GPU products read the float rows (operand 3 of GP_KQ_LINEAR). */
        return 0;
    case GP_KQ_LINEAR:
        if (hlit(r, 5, &bad) == KQ_MXF8) {
            /* an MXFP8 matrix of FP4-Local (kernels/quant/mxf8.cu): x in bfloat16, out rounded; operand 10 (or
             * none): a factor of every output (Super's FP8 tensor scale, the GP_MUL after it folded in) */
            const float sc = r->tag[10] != GP_T_NONE ? hlitf(r, 10, &bad) : 1.f;
            if (bad || mxf8_linear_launch((const float *)(intptr_t)hi(r, g->henv, 3),
                                          (const uint8_t *)(intptr_t)hi(r, g->henv, 4), (int)hlit(r, 6, &bad),
                                          (int)hlit(r, 7, &bad), (int)hlit(r, 8, &bad),
                                          (float *)(intptr_t)hi(r, g->henv, 9), sc, g->w16, g->w16_n,
                                          (uint16_t *)g->xh, g->xh_n, s) != 0) {
                bad = 1;
            }
            break;
        }
        if (hlit(r, 5, &bad) == KQ_NV4 && !bad &&
            nv4_linear_launch((const float *)(intptr_t)hi(r, g->henv, 3), (const uint8_t *)(intptr_t)hi(r, g->henv, 4),
                              (int)hlit(r, 6, &bad), (int)hlit(r, 7, &bad), (int)hlit(r, 8, &bad),
                              (float *)(intptr_t)hi(r, g->henv, 9), s) == 0) {
            /* NVFP4 rows of a step or a few tokens (kernels/quant/nv4.cu: FP4-Local's MTP head) */
            break;
        }
        if (kq_e3_k((int)hlit(r, 5, &bad))) {
            /* an EXL3 MUL1 matrix (kernels/quant/exl3.cu): x is operand 3 */
            int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad), rows = hlit(r, 6, &bad);
            if (bad || e3_linear_launch((const uint8_t *)(intptr_t)hi(r, g->henv, 4),
                                        kq_e3_k((int)hlit(r, 5, &bad)), (int)rows, (int)cols,
                                        (const float *)(intptr_t)hi(r, g->henv, 3), (size_t)cols, t,
                                        (float *)(intptr_t)hi(r, g->henv, 9), (size_t)rows, g->e3, g->e3_n,
                                        0.f, s, g->tc) != 0) {
                bad = 1;
            }
            break;
        }
        if (hlit(r, 8, &bad) > MT_MAX && kt_on() && g->tc &&
            kt_format((int)hlit(r, 5, &bad)) == KT_Q8R && hlit(r, 7, &bad) % KT_K == 0 &&
            (size_t)hlit(r, 8, &bad) * hlit(r, 7, &bad) <= g->kqx_n) {
            /* The operand x (3) is a device address in the record on the device;
             * the host copy of the record has it too. */
            int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad), rows = hlit(r, 6, &bad);
            kt_attr();
            kt_quant(g, (const float *)(intptr_t)hi(r, g->henv, 3), (size_t)(t * cols));
            float *xs = (float *)(g->kqx + g->kqx_n);
            k_kq_tc<KT_Q8R><<<dim3((unsigned)cdiv(t, 128), (unsigned)cdiv(rows, KT_BN)), 256,
                              KT_SMEM(128, 2), s>>>(dr, denv, g->kqx, xs, xs + g->kqx_n / 32);
        } else if (hlit(r, 8, &bad) > MT_MAX && hlit(r, 5, &bad) == KQ_BF12 && bf16_tc_on() &&
                   hlit(r, 7, &bad) % 32 == 0 &&
                   (bf12_fused_on() || (g->w16 != NULL && (size_t)hlit(r, 7, &bad) <= g->w16_n)) &&
                   (size_t)hlit(r, 8, &bad) * hlit(r, 7, &bad) * (bf16_tc_on() != 8 ? 2 : 1) <= g->xh_n) {
            /* BF12 rows on the tensor cores: chunks of bfloat16 rows (gg_gemm_bf12) */
            int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad), rows = hlit(r, 6, &bad);
            gg_gemm_bf12((const float *)(intptr_t)hi(r, g->henv, 3), (size_t)cols, NULL,
                         (const uint8_t *)(intptr_t)hi(r, g->henv, 4),
                         (float *)(intptr_t)hi(r, g->henv, 9), (size_t)rows, NULL, (int)t, (int)rows,
                         (int)cols, (uint16_t *)g->xh, g->w16, g->w16_n, s);
        } else if (hlit(r, 8, &bad) > MT_MAX && hlit(r, 5, &bad) == KQ_BF16 && bf16_tc_on() &&
                   hlit(r, 7, &bad) % 8 == 0 &&
                   (size_t)hlit(r, 8, &bad) * hlit(r, 7, &bad) * (bf16_tc_on() != 8 ? 2 : 1) <= g->xh_n) {
            /* bfloat16 rows on the tensor cores (k_gemm_bf16_tc), x to bfloat16 in
             * the scratch xh */
            int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad), rows = hlit(r, 6, &bad);
            gg_gemm_bf16((const float *)(intptr_t)hi(r, g->henv, 3), (size_t)cols, NULL,
                         (const uint16_t *)(intptr_t)hi(r, g->henv, 4),
                         (float *)(intptr_t)hi(r, g->henv, 9), (size_t)rows, NULL, (int)t, (int)rows,
                         (int)cols, (uint16_t *)g->xh, s);
        } else if (hlit(r, 8, &bad) > MT_MAX && kt_dp4a_on() && g->tcreq &&
                   kd_format((int)hlit(r, 5, &bad)) >= 0 && hlit(r, 7, &bad) % KT_K == 0 &&
                   (kd_format((int)hlit(r, 5, &bad)) != KT_NVX || hlit(r, 6, &bad) % 16 == 0) &&
                   (size_t)hlit(r, 8, &bad) * hlit(r, 7, &bad) <= g->kqx_n) {
            /* no tensor cores (Pascal): the int8 tiles on dp4a */
            int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad), rows = hlit(r, 6, &bad);
            const float *x = (const float *)(intptr_t)hi(r, g->henv, 3);
            float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
            const int x2 = kq_x2_on();
            const int8_t *xl = x2 ? kt_lo(g) : NULL;
            if (x2) {
                kt_quant_x2(g, x, (size_t)(t * cols));
            } else {
                kt_quant(g, x, (size_t)(t * cols));
            }
            dim3 gr((unsigned)cdiv(t, KD_BM), (unsigned)cdiv(rows, KT_BN));
            int f = kd_format((int)hlit(r, 5, &bad));
            if (x2) {
                KD_SWITCH(f, k_kq_dp4a, 1, gr, 0, xl);
            } else {
                KD_SWITCH(f, k_kq_dp4a, 0, gr, 0, xl);
            }
        } else if (hlit(r, 8, &bad) > MT_MAX) {
            if (hlit(r, 7, &bad) % KG_K != 0) {
                bad = 1;
            }
            k_kq_gemm<<<dim3((unsigned)cdiv(hlit(r, 6, &bad), KG_B),
                             (unsigned)cdiv(hlit(r, 8, &bad), KG_B)), 256, 0, s>>>(dr, denv);
        } else if (kq_i8_type((int)hlit(r, 5, &bad)) && i8x_on() && hlit(r, 7, &bad) % 32 == 0 &&
                   (size_t)(hlit(r, 8, &bad) * hlit(r, 7, &bad)) <= g->kqx_n) {
            size_t n = (size_t)(hlit(r, 8, &bad) * hlit(r, 7, &bad));
            const float *x = (const float *)(intptr_t)hi(r, g->henv, 3);
            if (x != qx_src || n != qx_n) {
                kt_quant(g, x, n);
            }
            dim3 gr((unsigned)cdiv(hlit(r, 6, &bad), KQ_RPB));
            const float *xs = (const float *)(g->kqx + g->kqx_n);
            /* A step keeps the registers of one token (NT 1). For a group,
             * one kernel with the counts 1 to KQ_NT (NT 0) was faster on the
             * E4B than a kernel for each count: a down matrix of 3 tokens took
             * 0.066 ms, not 0.103 ms. The terms of a token are the same. */
            int q40 = hlit(r, 5, &bad) == KQ_Q4_0;
            int64_t tg = hlit(r, 8, &bad);
            if (tg <= 1) {
                if (q40) {
                    k_kq_linear_i8<1, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                } else {
                    k_kq_linear_i8<1><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                }
            } else if (q40 && kq_r2(tg, hlit(r, 6, &bad))) {
                /* two rows a warp (k_kq_linear_i8x2) */
                dim3 g2((unsigned)cdiv(hlit(r, 6, &bad), 2 * KQ_RPB));
                if (tg >= KQ_NT && tg <= KQ_G8) {
                    k_kq_linear_i8x2<KQ_G8><<<g2, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                } else {
                    k_kq_linear_i8x2<0><<<g2, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                }
            } else if (tg > KQ_NT && tg <= KQ_G8) {
                /* 5 to 8 tokens in one part (KQ_G8) */
                if (q40) {
                    k_kq_linear_i8<KQ_G8, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                } else {
                    k_kq_linear_i8<KQ_G8><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                }
            } else if (q40) {
                k_kq_linear_i8<0, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else {
                k_kq_linear_i8<0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            }
        } else if (bt_on() && hlit(r, 8, &bad) <= 8 &&
                   bt_shape((int)hlit(r, 5, &bad), (int)hlit(r, 6, &bad), (int)hlit(r, 7, &bad)) &&
                   /* (one token on a matrix of at most 4096 x 4096: k_kq_linear_bf12 below, Super's latent
                    * projections 18.2 -> 13.6 us, its router 11.2 -> 9.3; 4 tokens 18.6 against 25.5 here) */
                   !(hlit(r, 8, &bad) == 1 && hlit(r, 6, &bad) <= 4096 && hlit(r, 7, &bad) <= 4096) &&
                   !(hlit(r, 7, &bad) >= 4096 && hlit(r, 6, &bad) >= 16384 && hlit(r, 5, &bad) == KQ_BF12 &&
                     bf16_nt_on() && !(kq_split_on() && hlit(r, 7, &bad) >= 8192) && hlit(r, 7, &bad) % 32 == 0)) {
            /* (long rows, many of them, as DeepSeek V4.1's head of 129280 x 5120: the CUDA-core kernels
             * below for 1 to 8 tokens, 1.26 ms against 1.81 on the tensor cores for one, 1.39 against
             * 1.80 for two, 2.61 against 1.90 for eight; and a token in a verify group gets the bits
             * of a token alone, which the tensor cores' order does not give) */
            /* KQ_BF12 rows of a step or a small group on the tensor cores */
            int rows = (int)hlit(r, 6, &bad);
            bt_launch(dr, denv, 0, bt_cls(rows, (int)hlit(r, 7, &bad)), cdiv(rows, 16), g->btp, g->btc, s);
        } else if (hlit(r, 5, &bad) == KQ_BF12 && hlit(r, 8, &bad) <= 8 && bf16_nt_on() &&
                   !(kq_split_on() && hlit(r, 7, &bad) >= 8192) && hlit(r, 7, &bad) % 32 == 0) {
            /* KQ_BF12 rows of a step or a small group (k_kq_linear_bf12); long
             * rows keep the split kernels (k_kq_linear_split, _split_bf16) */
            int64_t t = hlit(r, 8, &bad), rows = hlit(r, 6, &bad);
            unsigned gb = (unsigned)cdiv(rows, KQ_RPB);
            if (t >= 2 && rows >= 16384) {
                /* many rows (a head): 4 rows a warp, x once for them (k_kq_linear_bf12_r; R = 2: 3.27
                 * ms for 8 tokens, 4: 2.61) */
                unsigned gr = (unsigned)cdiv(rows, KQ_RPB * 4);
#define BF12_R(NN) k_kq_linear_bf12_r<NN, 4><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv)
                if (t <= 2) {
                    BF12_R(2);
                } else if (t <= 4) {
                    BF12_R(4);
                } else {
                    BF12_R(8);
                }
#undef BF12_R
            } else if (t <= 1) {
                k_kq_linear_bf12<1><<<gb, 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else if (t <= 2) {
                k_kq_linear_bf12<2><<<gb, 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else if (t <= 4) {
                k_kq_linear_bf12<4><<<gb, 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else {
                k_kq_linear_bf12<8><<<gb, 32 * KQ_RPB, 0, s>>>(dr, denv);
            }
        } else if ((kq_bf_type((int)hlit(r, 5, &bad)) || hlit(r, 5, &bad) == KQ_F32) && hlit(r, 8, &bad) >= 2 &&
                   hlit(r, 8, &bad) <= 8 && bf16_nt_on() && hlit(r, 7, &bad) % 16 == 0) {
            /* bfloat16 (or float32) rows of a small group: the weights once for all
             * the tokens (k_kq_linear_bf16), as the split of a step */
            int64_t t = hlit(r, 8, &bad), rows = hlit(r, 6, &bad);
            int split = kq_split_on() && hlit(r, 7, &bad) >= 8192;
            if (split && t <= 4) {
                k_kq_linear_split_bf16<4><<<(unsigned)rows, 32 * KQ_SPLIT, 0, s>>>(dr, denv);
            } else if (split) {
                k_kq_linear_split_bf16<8><<<(unsigned)rows, 32 * KQ_SPLIT, 0, s>>>(dr, denv);
            } else if (t <= 2) {
                k_kq_linear_bf16<2><<<(unsigned)cdiv(rows, KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else if (t <= 4) {
                k_kq_linear_bf16<4><<<(unsigned)cdiv(rows, KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv);
            } else {
                k_kq_linear_bf16<8><<<(unsigned)cdiv(rows, KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv);
            }
        } else if (kq_split_on() && hlit(r, 8, &bad) <= MT_MAX && hlit(r, 7, &bad) >= 8192) {
            k_kq_linear_split<<<(unsigned)hlit(r, 6, &bad), 32 * KQ_SPLIT, 0, s>>>(dr, denv);
        } else {
            k_kq_linear<<<(unsigned)cdiv(hlit(r, 6, &bad), KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv);
        }
        break;
    case GP_KQ_MULTI: {
        int64_t rows = 0, q8r = 0, t = hlit(r, 2, &bad), cols = hlit(r, 1, &bad);
        int q40 = 1;        /* all the matrices are Q4_0 (the kernels with FT) */
        for (int m = 0; m < hlit(r, 3, &bad); ++m) {
            rows += hlit(r, 6 + 4 * m, &bad);
            q8r |= kq_i8_type((int)hlit(r, 5 + 4 * m, &bad));
            q40 &= hlit(r, 5 + 4 * m, &bad) == KQ_Q4_0;
        }
        if (q8r && i8x_on() && cols % 32 == 0 && (size_t)(t * cols) <= g->kqx_n) {
            const float *x = (const float *)(intptr_t)hi(r, g->henv, 0);
            if (x != qx_src || (size_t)(t * cols) != qx_n) {
                kt_quant(g, x, (size_t)(t * cols));
            }
            dim3 gr((unsigned)cdiv(rows, KQ_RPB));
            const float *xs = (const float *)(g->kqx + g->kqx_n);
            /* the gate and the up matrix, then GELU_MUL_ROWS of their
             * outputs: one kernel (k_kq_glu_i8) */
            const gp_rec *nx = r + 1;
            if (glu_on() && hlit(r, 3, &bad) == 2 && nx < g->hcode + g->n_code &&
                nx->op == GP_GELU_MUL_ROWS && hlit(r, 6, &bad) == hlit(r, 10, &bad) &&
                kq_i8_type((int)hlit(r, 5, &bad)) && kq_i8_type((int)hlit(r, 9, &bad)) &&
                hi(nx, g->henv, 0) == hi(r, g->henv, 7) && hi(nx, g->henv, 1) == hi(r, g->henv, 11) &&
                hlit(nx, 3, &bad) == t && hlit(nx, 4, &bad) == hlit(r, 6, &bad)) {
                dim3 gg1((unsigned)cdiv(hlit(r, 6, &bad), KQ_RPB));
                float *out = (float *)(intptr_t)hi(nx, g->henv, 2);
                if (t <= 1 && q40) {
                    k_kq_glu_i8<1, KQ_Q4_0><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                } else if (t <= 1) {
                    k_kq_glu_i8<1><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                } else if (q40 && kq_r2(t, hlit(r, 6, &bad))) {
                    dim3 g2((unsigned)cdiv(hlit(r, 6, &bad), 2 * KQ_RPB));
                    if (t >= KQ_NT && t <= KQ_G8) {
                        k_kq_glu_i8x2<KQ_G8><<<g2, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                    } else {
                        k_kq_glu_i8x2<0><<<g2, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                    }
                } else if (t > KQ_NT && t <= KQ_G8 && q40) {
                    k_kq_glu_i8<KQ_G8, KQ_Q4_0><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                } else if (t > KQ_NT && t <= KQ_G8) {
                    k_kq_glu_i8<KQ_G8><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                } else if (q40) {
                    k_kq_glu_i8<0, KQ_Q4_0><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                } else {
                    k_kq_glu_i8<0><<<gg1, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs, out);
                }
                gg_skip = nx;
                break;
            }
            if (t <= 1 && q40) {
                k_kq_multi_i8<1, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else if (t <= 1) {
                k_kq_multi_i8<1><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else if (q40 && kq_r2(t, rows)) {
                dim3 g2((unsigned)cdiv(rows, 2 * KQ_RPB));
                if (t >= KQ_NT && t <= KQ_G8) {
                    k_kq_multi_i8x2<KQ_G8><<<g2, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                } else {
                    k_kq_multi_i8x2<0><<<g2, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
                }
            } else if (t > KQ_NT && t <= KQ_G8 && q40) {
                k_kq_multi_i8<KQ_G8, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else if (t > KQ_NT && t <= KQ_G8) {
                k_kq_multi_i8<KQ_G8><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else if (q40) {
                k_kq_multi_i8<0, KQ_Q4_0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            } else {
                k_kq_multi_i8<0><<<gr, 32 * KQ_RPB, 0, s>>>(dr, denv, g->kqx, xs);
            }
        } else {
            int anybf = 0, rest = 0;
            int64_t btb[3] = {0, 0, 0};     /* the blocks of k_kq_bf12_tc of each class */
            for (int m = 0; m < hlit(r, 3, &bad); ++m) {
                int ty = (int)hlit(r, 5 + 4 * m, &bad), rm = (int)hlit(r, 6 + 4 * m, &bad);
                if (bt_on() && t <= 8 && bt_shape(ty, rm, (int)cols)) {
                    btb[bt_cls(rm, (int)cols)] += cdiv(rm, 16);
                } else {
                    rest = 1;
                    anybf |= kq_bf_type(ty);
                }
            }
            int skip = btb[0] + btb[1] + btb[2] > 0;
            /* the KQ_BF12 matrices of bt_shape on the tensor cores */
            for (int k = 0; k < 3; ++k) {
                if (btb[k] > 0) {
                    bt_launch(dr, denv, 1, k, btb[k], g->btp, g->btc, s);
                }
            }
            if (!rest) {
                /* all of them there */
            } else if (anybf && t >= 2 && t <= 8 && bf16_nt_on() && cols % 16 == 0) {
                /* bfloat16 rows of a small group (k_kq_multi_bf16) */
                unsigned gm = (unsigned)cdiv(rows, KQ_RPB);
                if (t <= 2) {
                    k_kq_multi_bf16<2><<<gm, 32 * KQ_RPB, 0, s>>>(dr, denv, skip);
                } else if (t <= 4) {
                    k_kq_multi_bf16<4><<<gm, 32 * KQ_RPB, 0, s>>>(dr, denv, skip);
                } else {
                    k_kq_multi_bf16<8><<<gm, 32 * KQ_RPB, 0, s>>>(dr, denv, skip);
                }
            } else {
                k_kq_multi<<<(unsigned)cdiv(rows, KQ_RPB), 32 * KQ_RPB, 0, s>>>(dr, denv, skip);
            }
        }
        break;
    }
    case GP_ADD_RMS:
        k_add_rms<<<(unsigned)hlit(r, 4, &bad), T, 0, s>>>(dr, denv);
        break;
    case GP_KQ_GROUP_MOE: {
        int64_t t = hlit(r, 3, &bad), k = hlit(r, 4, &bad), E = hlit(r, 5, &bad);
        int64_t hidden = hlit(r, 6, &bad), inner = hlit(r, 7, &bad);
        unsigned tiles = (unsigned)(cdiv(t * k + t, KG_B) + E + 1);
        if (hidden % KG_B != 0 || inner % KG_B != 0) {
            bad = 1;
        }
        int fg = kt_format((int)hlit(r, 11, &bad)), fd = kt_format((int)hlit(r, 12, &bad));
        int64_t P = t * k + t;
        if (qmoe_sort_par_on() && (E + 2) * 8 <= 48 * 1024) {
            k_qmoe_sort_par<<<1, 1024, (size_t)(E + 2) * 8, s>>>(dr, denv);
        } else {
            k_qmoe_sort<<<1, 32, 0, s>>>(dr, denv);
        }
        if (kq_e3_k((int)hlit(r, 11, &bad)) && kq_e3_k((int)hlit(r, 12, &bad))) {
            /* MUL1 experts (kernels/quant/exl3.cu): the batches of products on
             * the rows of the sort, k_qmoe_act between, k_qmoe_sum after */
            const int has_sh = hi(r, g->henv, 13) != 0;
            const int Kr = kq_e3_k((int)hlit(r, 11, &bad)), Kd = kq_e3_k((int)hlit(r, 12, &bad));
            const int Ks = has_sh ? kq_e3_k((int)hlit(r, 16, &bad)) : Kr;
            const float *hx = (const float *)(intptr_t)hi(r, g->henv, 0);
            float *act = (float *)(intptr_t)hi(r, g->henv, 19), *de = (float *)(intptr_t)hi(r, g->henv, 21);
            const float *act2 = (const float *)(intptr_t)hi(r, g->henv, 20);
            if (bad || !Ks ||
                e3_group_moe_launch(0, dr, denv, (int)P, (int)hidden, (int)inner, Kr, Kd, Ks, hx, act, act2, de,
                                    g->e3, g->e3_n, s, tiles, g->tc) != 0) {
                bad = 1;
                break;
            }
            k_qmoe_act<<<(unsigned)cdiv(P * inner, T), T, 0, s>>>(dr, denv, gg_moe_rot & ~4);
            if (e3_group_moe_launch(1, dr, denv, (int)P, (int)hidden, (int)inner, Kr, Kd, Ks, hx, act, act2, de,
                                    g->e3, g->e3_n, s, tiles, g->tc) != 0) {
                bad = 1;
                break;
            }
            k_qmoe_sum<<<dim3((unsigned)cdiv(hidden, T), (unsigned)t), T, 0, s>>>(dr, denv);
            break;
        }
        /* the shared expert in bfloat16 (RECLANK_DENSE=bf16): the routed
         * experts on the int8 tensor cores, the shared expert in gg_gemm_bf16
         * (its pairs are the rows start[E] .. start[E] + t of act, act2, de) */
        int stype = (int)hlit(r, 16, &bad);
        int sbf = (stype == KQ_BF16 ||
                   (stype == KQ_BF12 && hidden % 32 == 0 && inner % 32 == 0 &&
                    (bf12_fused_on() || (g->w16 != NULL &&
                                         (size_t)(hidden > inner ? hidden : inner) <= g->w16_n)))) &&
                  bf16_tc_on() && hidden % 16 == 0 && inner % 16 == 0 &&
                  2 * (size_t)(t * (hidden > inner ? hidden : inner)) <= g->xh_n;
        const int has_shared = hi(r, g->henv, 13) != 0;
        if ((gg_moe_rot & 8) && fg == KT_NVX && fd == KT_NVX && !has_shared && g->tc && kt_on() &&
            hidden % KT_K == 0 && inner % KT_K == 0 && (size_t)(t * hidden) <= g->kqx_n &&
            (size_t)(P * inner) <= g->kqx_n) {
            /* FP4-Local's prompt rows (kernels/moe/b12x.cu): b12x's A4 */
            float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
            size_t sm = KT_SMEM(64, 3);
            kt_attr();
            dim3 ggu((unsigned)(2 * inner / KT_BN), tiles), gdn((unsigned)(hidden / KT_BN), tiles);
            k_a4_quant_h<<<(unsigned)cdiv(t * (hidden / 16), 256), 256, 0, s>>>(dr, denv, g->kqx, xs, xsum);
            k_qmoe_gu_tc<KT_NVX, 1><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, 0);
            k_qmoe_act_a4<<<(unsigned)cdiv(P * inner, T), T, 0, s>>>(dr, denv);
            k_a4_quant_pairs<<<(unsigned)cdiv(P * (inner / 16), 256), 256, 0, s>>>(dr, denv, g->kqx, xs, xsum);
            k_qmoe_dn_tc<KT_NVX, 1><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, 0);
            k_qmoe_y_a4<<<(unsigned)cdiv(P * hidden, T), T, 0, s>>>(dr, denv);
            k_qmoe_sum<<<dim3((unsigned)cdiv(hidden, T), (unsigned)t), T, 0, s>>>(dr, denv);
            break;
        }
        /* (no shared expert, LFM2: the routed experts alone on the tensor cores) */
        if (kt_on() && !moe_f32_on() && g->tc && fg >= 0 && fd >= 0 &&
            (kt_format((int)hlit(r, 16, &bad)) == KT_Q8R || sbf || !has_shared) &&
            hidden % KT_K == 0 && inner % KT_K == 0 && (size_t)(t * hidden) <= g->kqx_n &&
            (size_t)(P * inner) <= g->kqx_n) {
            const int *sstart = (const int *)(intptr_t)hi(r, g->henv, 18) + 8 + (E + 2) + E;
            float *act = (float *)(intptr_t)hi(r, g->henv, 19);
            kt_attr();
            float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
            size_t sm = KT_SMEM(64, 3);
            kt_quant(g, (const float *)(intptr_t)hi(r, g->henv, 0), (size_t)(t * hidden));
            dim3 ggu((unsigned)(2 * inner / KT_BN), tiles), gdn((unsigned)(hidden / KT_BN), tiles);
            switch (fg) {
            case KT_Q8R: k_qmoe_gu_tc<KT_Q8R><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q80: k_qmoe_gu_tc<KT_Q80><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q51: k_qmoe_gu_tc<KT_Q51><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q40: k_qmoe_gu_tc<KT_Q40><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_NV4: k_qmoe_gu_tc<KT_NV4><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_NVX: k_qmoe_gu_tc<KT_NVX><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q6K: k_qmoe_gu_tc<KT_Q6K><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            default: k_qmoe_gu_tc<KT_Q4K><<<ggu, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            }
            if (sbf && has_shared) {
                /* the gate and the up rows of the shared expert for the t rows of h */
                const float *h = (const float *)(intptr_t)hi(r, g->henv, 0);
                for (int u = 0; u < 2; ++u) {
                    const void *sw = (const void *)(intptr_t)hi(r, g->henv, 13 + u);
                    if (stype == KQ_BF12) {
                        gg_gemm_bf12(h, (size_t)hidden, NULL, (const uint8_t *)sw, act + u * inner,
                                     (size_t)(2 * inner), sstart, (int)t, (int)inner, (int)hidden,
                                     (uint16_t *)g->xh, g->w16, g->w16_n, s);
                    } else {
                        gg_gemm_bf16(h, (size_t)hidden, NULL, (const uint16_t *)sw, act + u * inner,
                                     (size_t)(2 * inner), sstart, (int)t, (int)inner, (int)hidden,
                                     (uint16_t *)g->xh, s);
                    }
                }
            }
            k_qmoe_act<<<(unsigned)cdiv(P * inner, T), T, 0, s>>>(dr, denv, gg_moe_rot & ~4);
            kt_quant(g, (const float *)(intptr_t)hi(r, g->henv, 20), (size_t)(P * inner));
            switch (fd) {
            case KT_Q8R: k_qmoe_dn_tc<KT_Q8R><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q80: k_qmoe_dn_tc<KT_Q80><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q51: k_qmoe_dn_tc<KT_Q51><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q40: k_qmoe_dn_tc<KT_Q40><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_NV4: k_qmoe_dn_tc<KT_NV4><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_NVX: k_qmoe_dn_tc<KT_NVX><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            case KT_Q6K: k_qmoe_dn_tc<KT_Q6K><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            default: k_qmoe_dn_tc<KT_Q4K><<<gdn, 256, sm, s>>>(dr, denv, g->kqx, xs, xsum, sbf); break;
            }
            if (sbf && has_shared) {
                /* the down rows of the shared expert: rows start[E] .. of act2 into de */
                if (stype == KQ_BF12) {
                    gg_gemm_bf12((const float *)(intptr_t)hi(r, g->henv, 20), (size_t)inner, sstart,
                                 (const uint8_t *)(intptr_t)hi(r, g->henv, 15),
                                 (float *)(intptr_t)hi(r, g->henv, 21), (size_t)hidden, sstart, (int)t,
                                 (int)hidden, (int)inner, (uint16_t *)g->xh, g->w16, g->w16_n, s);
                } else {
                    gg_gemm_bf16((const float *)(intptr_t)hi(r, g->henv, 20), (size_t)inner, sstart,
                                 (const uint16_t *)(intptr_t)hi(r, g->henv, 15),
                                 (float *)(intptr_t)hi(r, g->henv, 21), (size_t)hidden, sstart, (int)t,
                                 (int)hidden, (int)inner, (uint16_t *)g->xh, s);
                }
            }
        } else if (kt_dp4a_on() && g->tcreq &&
                   kd_format((int)hlit(r, 11, &bad)) >= 0 && kd_format((int)hlit(r, 12, &bad)) >= 0 &&
                   (kd_format(stype) >= 0 || !has_shared) && hidden % KT_K == 0 &&
                   inner % KT_K == 0 && (size_t)(t * hidden) <= g->kqx_n &&
                   (size_t)(P * inner) <= g->kqx_n) {
            /* no tensor cores (Pascal): the int8 tiles of k_qmoe_*_tc on dp4a;
             * the shared expert in a launch of its own format. With
             * RECLANK_GPU_MOE_F32 (the float32 x of k_qmoe_gu: RECLANK_MOE_X16
             * of Qwen) x in the int16 form (two int8 planes). */
            float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
            const int x2 = moe_f32_on();
            const int8_t *xl = x2 ? kt_lo(g) : NULL;
            if (x2) {
                kt_quant_x2(g, (const float *)(intptr_t)hi(r, g->henv, 0), (size_t)(t * hidden));
            } else {
                kt_quant(g, (const float *)(intptr_t)hi(r, g->henv, 0), (size_t)(t * hidden));
            }
            dim3 ggu((unsigned)(2 * inner / KT_BN), tiles), gdn((unsigned)(hidden / KT_BN), tiles);
            int fs = has_shared ? kd_format(stype) : -1;
            for (int sh = 0; sh < 2; ++sh) {
                int f = sh ? fs : kd_format((int)hlit(r, 11, &bad));
                if (f < 0) {
                    continue;
                }
                if (x2) {
                    KD_SWITCH(f, k_qmoe_gu_dp4a, 1, ggu, sh, xl);
                } else {
                    KD_SWITCH(f, k_qmoe_gu_dp4a, 0, ggu, sh, xl);
                }
            }
            k_qmoe_act<<<(unsigned)cdiv(P * inner, T), T, 0, s>>>(dr, denv, gg_moe_rot & ~4);
            if (x2) {
                kt_quant_x2(g, (const float *)(intptr_t)hi(r, g->henv, 20), (size_t)(P * inner));
            } else {
                kt_quant(g, (const float *)(intptr_t)hi(r, g->henv, 20), (size_t)(P * inner));
            }
            for (int sh = 0; sh < 2; ++sh) {
                int f = sh ? fs : kd_format((int)hlit(r, 12, &bad));
                if (f < 0) {
                    continue;
                }
                if (x2) {
                    KD_SWITCH(f, k_qmoe_dn_dp4a, 1, gdn, sh, xl);
                } else {
                    KD_SWITCH(f, k_qmoe_dn_dp4a, 0, gdn, sh, xl);
                }
            }
        } else {
            k_qmoe_gu<<<dim3((unsigned)(2 * inner / KG_B), tiles), 256, 0, s>>>(dr, denv);
            k_qmoe_act<<<(unsigned)cdiv((t * k + t) * inner, T), T, 0, s>>>(dr, denv, gg_moe_rot & ~4);
            k_qmoe_dn<<<dim3((unsigned)(hidden / KG_B), tiles), 256, 0, s>>>(dr, denv);
        }
        k_qmoe_sum<<<dim3((unsigned)cdiv(hidden, T), (unsigned)t), T, 0, s>>>(dr, denv);
        break;
    }
    case GP_QSA_SELECT: {
        int64_t t = hlit(r, 9, &bad), d = hlit(r, 11, &bad), ratio = hlit(r, 12, &bad);
        if (d > 256 || hlit(r, 10, &bad) * d > 8 * 256 || ratio < 1) {
            bad = 1;
        }
        k_qsa_copy<<<(unsigned)cdiv(t * d, T), T, 0, s>>>(dr, denv);
        k_qsa_blocks<<<(unsigned)(t / (ratio > 0 ? ratio : 1) + 1), 128, 0, s>>>(dr, denv);
        int stc = qsa_score_tc_on() && hlit(r, 10, &bad) == 4 && d == 128;
        for (int64_t j0 = 0; j0 < t; j0 += QSA_ROWS) {
            unsigned rows = (unsigned)(t - j0 < QSA_ROWS ? t - j0 : QSA_ROWS);
            if (stc) {
                /* few queries (a step, an MTP verify group): more blocks on
                 * the keys, so the scan of all the blocks takes all the SMs */
                unsigned gy = (unsigned)cdiv(rows, 16);
                unsigned gx = gy >= 10 ? QS_GX : (gy >= 2 ? 64 : 128);
                k_qsa_qprep<<<rows, 128, 0, s>>>(dr, denv, (int)j0);
                k_qsa_score_tc<<<dim3(gx, gy), 128, 0, s>>>(dr, denv, (int)j0, (int)rows);
                k_qsa_query<1><<<rows, QSA_T, 0, s>>>(dr, denv, (int)j0);
            } else {
                k_qsa_query<0><<<rows, QSA_T, 0, s>>>(dr, denv, (int)j0);
            }
        }
        break;
    }
    case GP_HC_NORM: {
        int64_t rows = hlit(r, 3, &bad) * hlit(r, 4, &bad), hid = hlit(r, 5, &bad);
        float *qs = NULL, *qsum = NULL;
        int8_t *q = hc_fuse_on() && hid % 32 == 0 && al16(hi(r, g->henv, 2))
                        ? qx_fuse(g, r, (const void *)(intptr_t)hi(r, g->henv, 2),
                                  (size_t)(rows * hid), &qs, &qsum) : NULL;
        k_hc_norm<<<(unsigned)rows, 256, 0, s>>>(dr, denv, q, qs, qsum);
        break;
    }
    case GP_HC_PRE: {
        /* the input of a block from the normed streams (k_hc_pre_down, k_hc_pre_up) */
        int64_t t = hlit(r, 7, &bad), hc = hlit(r, 8, &bad), hid = hlit(r, 9, &bad), lr = hlit(r, 10, &bad);
        int64_t rows = lr + (hi(r, g->henv, 3) != 0 ? hc : 0);
        if (bad || t > HC_PRE_T || hid % 512 != 0 || lr % 32 != 0 || t * hid * 4 > 48 * 1024) {
            bad = 1;
            break;
        }
        k_hc_pre_down<<<dim3((unsigned)cdiv(rows, 8), (unsigned)hc), 256, (size_t)(t * hid * 4), s>>>(dr, denv);
        k_hc_pre_up<<<(unsigned)cdiv(hid, 8), 256, (size_t)(t * lr * 4), s>>>(dr, denv);
        break;
    }
    case GP_HC_ACT: {
        int64_t n = hlit(r, 2, &bad);
        float *qs = NULL, *qsum = NULL;
        int8_t *q = hc_fuse_on() && n % 32 == 0 && al16(hi(r, g->henv, 0)) && al16(hi(r, g->henv, 1))
                        ? qx_fuse(g, r, (const void *)(intptr_t)hi(r, g->henv, 1), (size_t)n, &qs,
                                  &qsum) : NULL;
        if (q != NULL) {
            k_hc_act_q<<<(unsigned)cdiv(n / 4, T), T, 0, s>>>(dr, denv, q, qs, qsum);
        } else {
            k_hc_act<<<(unsigned)cdiv(n, T), T, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_HC_MIX: {
        int64_t n = hlit(r, 3, &bad) * hlit(r, 5, &bad);
        float *qs = NULL, *qsum = NULL;
        int8_t *q = hc_fuse_on() && n % 32 == 0 && hlit(r, 5, &bad) % 4 == 0 && al16(hi(r, g->henv, 2))
                        ? qx_fuse(g, r, (const void *)(intptr_t)hi(r, g->henv, 2), (size_t)n, &qs,
                                  &qsum) : NULL;
        if (q != NULL) {
            k_hc_mix_q<<<(unsigned)cdiv(n / 4, T), T, 0, s>>>(dr, denv, q, qs, qsum);
        } else {
            k_hc_mix<<<(unsigned)cdiv(n, T), T, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_HC_ADD: {
        const gp_rec *nx = r + 1;
        if (hc_fuse_on() && nx < g->hcode + g->n_code && nx->op == GP_HC_NORM &&
            hi(nx, g->henv, 0) == hi(r, g->henv, 0) && hlit(nx, 3, &bad) == hlit(r, 3, &bad) &&
            hlit(nx, 4, &bad) == hlit(r, 4, &bad) && hlit(nx, 5, &bad) == hlit(r, 5, &bad)) {
            /* the add and the norm of the next record (k_hc_add_norm) */
            int64_t rows = hlit(r, 3, &bad) * hlit(r, 4, &bad), hid = hlit(r, 5, &bad);
            float *qs = NULL, *qsum = NULL;
            int8_t *q = hid % 32 == 0 && al16(hi(nx, g->henv, 2))
                            ? qx_fuse(g, nx, (const void *)(intptr_t)hi(nx, g->henv, 2),
                                      (size_t)(rows * hid), &qs, &qsum) : NULL;
            k_hc_add_norm<<<(unsigned)rows, 256, 0, s>>>(dr, denv, dr + 1, q, qs, qsum);
            gg_skip = nx;
            break;
        }
        k_hc_add<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad) * hlit(r, 5, &bad), T), T,
                   0, s>>>(dr, denv);
        break;
    }
    case GP_PLE_GATE:
        k_ple_gate<<<(unsigned)(hlit(r, 4, &bad) * hlit(r, 5, &bad)), 256, 0, s>>>(dr, denv);
        break;
    case GP_PLE_CONV:
        k_ple_conv<<<(unsigned)cdiv(hlit(r, 6, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_HC_CAT:
        k_hc_cat<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad) * 2 * hlit(r, 5, &bad), T),
                   T, 0, s>>>(dr, denv);
        break;
    case GP_SIGMUL:
        k_sigmul<<<(unsigned)cdiv(hlit(r, 3, &bad), T), T, 0, s>>>(dr, denv);
        break;
    case GP_Q16:
    case GP_DQ16:
        if (hlit(r, 2, &bad) % 128 != 0) {
            bad = 1;
            break;
        }
        if (r->op == GP_Q16) {
            k_q16<<<(unsigned)cdiv(hlit(r, 2, &bad) / 128, 8), 256, 0, s>>>(dr, denv);
        } else {
            k_dq16<<<(unsigned)cdiv(hlit(r, 2, &bad) / 128, 8), 256, 0, s>>>(dr, denv);
        }
        break;
    case GP_ROUTER_TOPK:
        if (hlit(r, 2, &bad) > 1024 || hlit(r, 3, &bad) > 256) {
            bad = 1;
        }
        {
            const gp_rec *nx = r + 1;
            if (topk_split_on() && hlit(r, 1, &bad) == 1 && nx < g->hcode + g->n_code &&
                nx->op == GP_HOT_SPLIT && hi(nx, g->henv, 0) == hi(r, g->henv, 5) &&
                hi(nx, g->henv, 1) == hi(r, g->henv, 4)) {
                /* the hot split of the token in the same kernel */
                k_router_topk<<<1, 256, 0, s>>>(dr, denv, dr + 1);
                gg_skip = nx;
                break;
            }
        }
        k_router_topk<<<(unsigned)hlit(r, 1, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_SHORT_CONV:
        if (hlit(r, 6, &bad) > 8) {
            bad = 1;
        }
        if (hlit(r, 4, &bad) == 1) {
            k_short_conv<<<(unsigned)cdiv(hlit(r, 5, &bad), 128), 128, 0, s>>>(dr, denv);
        } else {
            /* a group: every row at once, then the state */
            k_short_conv_rows<<<dim3((unsigned)cdiv(hlit(r, 5, &bad), 128), (unsigned)hlit(r, 4, &bad)),
                                128, 0, s>>>(dr, denv);
            k_short_conv_state<<<(unsigned)cdiv(hlit(r, 5, &bad), 128), 128, 0, s>>>(dr, denv);
        }
        break;
    case GP_NM_MAMBA: {
        /* (kernels/nemotron/nm.cu) the state rows of 128 values: 4 a lane */
        int64_t t = hlit(r, 11, &bad), heads = hlit(r, 12, &bad), hdim = hlit(r, 13, &bad);
        int64_t groups = hlit(r, 14, &bad), nstate = hlit(r, 15, &bad), kern = hlit(r, 16, &bad);
        if (nstate != 128 || kern > 9 || groups <= 0 || heads % groups != 0) {
            bad = 1;
            break;
        }
        int64_t conv = heads * hdim + 2 * groups * nstate;
        if (t == 1 && hdim <= 64 && hdim % 8 == 0 && kern >= 2 && r->tag[23] == GP_T_NONE) {
            /* a step (nreal 1): the conv and the scan in one kernel, the conv's state rows with the norm */
            k_nm_mamba1<<<(unsigned)heads, 256, 0, s>>>(dr, denv);
            k_nm_gnorm<true><<<(unsigned)groups, 256, 0, s>>>(dr, denv);
            break;
        }
        k_nm_conv<<<(unsigned)cdiv(conv, 256), 256, 0, s>>>(dr, denv);
        k_nm_scan<<<(unsigned)cdiv(heads * hdim, 4), 128, 0, s>>>(dr, denv);
        k_nm_gnorm<false><<<(unsigned)(t * groups), 256, 0, s>>>(dr, denv);
        break;
    }
    case GP_RELU2:
        k_nm_relu2<<<(unsigned)cdiv(hlit(r, 2, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_NM_ATTN: {
        int64_t hd = hlit(r, 6, &bad);
        if (hd % 32 != 0 || hd > 256) {
            bad = 1;
            break;
        }
        k_nm_attn<<<(unsigned)cdiv(hlit(r, 7, &bad) * hlit(r, 4, &bad), 4), 128, 0, s>>>(dr, denv);
        break;
    }
    case GP_NM_SPLIT:
        k_nm_split<<<1, 256, 0, s>>>(dr, denv);
        break;
    case GP_NM_PLAN:
        if (hlit(r, 8, &bad) > NM_PLAN_E || hlit(r, 5, &bad) / hlit(r, 6, &bad) > NM_PLAN_T) {
            bad = 1;
            break;
        }
        k_nm_plan<<<1, 1024, 0, s>>>(dr, denv);
        break;
    case GP_NM_HOT_MOE: {
        int64_t k = hlit(r, 9, &bad), L = hlit(r, 10, &bad), inner = hlit(r, 11, &bad), t = hlit(r, 12, &bad);
        if (L % 256 != 0 || inner % 256 != 0) {
            bad = 1;
            break;
        }
        if (r->tag[13] != GP_T_NONE && t >= 2 && t <= NM_HG && t * k <= 128 && nm_hg_on() && !gg_nm_hg_off) {
            /* a small group by expert (k_nm_hg_*: each expert's rows decoded once for its pairs) */
            k_nm_hot_qx<<<(unsigned)cdiv(t * L / 32, 128), 128, 0, s>>>(dr, denv);
            switch (nm_hg_form()) {
            case 22: nm_hg_launch<2, 2>(t, k, L, inner, s, dr, denv); break;
            case 24: nm_hg_launch<2, 4>(t, k, L, inner, s, dr, denv); break;
            case 44: nm_hg_launch<4, 4>(t, k, L, inner, s, dr, denv); break;
            case 82: nm_hg_launch<8, 2>(t, k, L, inner, s, dr, denv); break;
            case 81: nm_hg_launch<8, 1>(t, k, L, inner, s, dr, denv); break;
            default: nm_hg_launch<4, 2>(t, k, L, inner, s, dr, denv); break;
            }
            k_nm_hg_sum<<<dim3((unsigned)cdiv(L, 256), (unsigned)t), 256, 0, s>>>(dr, denv);
            break;
        } else if (r->tag[13] != GP_T_NONE) {
            /* int8 x and act (dp4a) */
            k_nm_hot_qx<<<(unsigned)cdiv(t * L / 32, 128), 128, 0, s>>>(dr, denv);
            k_nm_hot_up8<<<dim3((unsigned)cdiv(inner, 4 * NM_HR), (unsigned)(t * k)), 128, 0, s>>>(dr, denv);
            k_nm_hot_qa<<<(unsigned)cdiv(t * k * inner / 32, 128), 128, 0, s>>>(dr, denv);
            /* (the sum of the slots in it) */
            if (k > NM_KMAX || L % NM_HR != 0 || inner % NM_HR != 0) {
                bad = 1;
                break;
            }
            k_nm_hot_down8<<<dim3((unsigned)(L / NM_HR), (unsigned)t), (unsigned)(32 * k), 0, s>>>(dr, denv);
        } else {
            k_nm_hot_up<<<dim3((unsigned)cdiv(inner, 4), (unsigned)(t * k)), 128, 0, s>>>(dr, denv);
            k_nm_hot_down<<<dim3((unsigned)cdiv(L, 4), (unsigned)(t * k)), 128, 0, s>>>(dr, denv);
            k_nm_hot_sum<<<dim3((unsigned)cdiv(L, 256), (unsigned)t), 256, 0, s>>>(dr, denv);
        }
        if (r->tag[19] == GP_T_SLOT || (r->tag[19] == GP_T_INT && r->v[19] != 0)) {
            /* (the sums as NM_QX's x too: here a kernel of its own) */
            k_nm_hot_oq<<<(unsigned)cdiv(t * L / 32, 128), 128, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_NM_HOT_NV4: {
        /* x, idx, val, slots, up, down, act, de, out, k, L, inner, t (GP_NM_HOT_MOE's); 13, 14 or none:
         * the device address of each expert's up and down rows (the slots mark the hot ones) */
        int64_t k = hlit(r, 9, &bad), L = hlit(r, 10, &bad), inner = hlit(r, 11, &bad), t = hlit(r, 12, &bad);
        if (L % 32 != 0 || inner % 32 != 0) {
            bad = 1;
            break;
        }
        if (nm_hot_b_on() && L <= 1024 && inner <= 3072) {
            /* (up: x of L <= 1024 columns, one block a lane; down: the act of inner <= 3072, three) */
            k_nm_hot_nv4b<false, 8, 1><<<dim3((unsigned)cdiv(inner, 64), (unsigned)(t * k)), 256,
                                         (size_t)L * 4, s>>>(dr, denv);
            k_nm_hot_nv4b<true, 4, 3><<<dim3((unsigned)cdiv(L, 32), (unsigned)(t * k)), 256,
                                        (size_t)inner * 4, s>>>(dr, denv);
        } else {
            k_nm_hot_up_nv4<<<dim3((unsigned)cdiv(inner, 8), (unsigned)(t * k)), 256, 0, s>>>(dr, denv);
            k_nm_hot_down_nv4<<<dim3((unsigned)cdiv(L, 8), (unsigned)(t * k)), 256, 0, s>>>(dr, denv);
        }
        k_nm_hot_sum<<<dim3((unsigned)cdiv(L, 256), (unsigned)t), 256, 0, s>>>(dr, denv);
        break;
    }
    case GP_NM_KV:
        k_nm_kv<<<(unsigned)cdiv(hlit(r, 5, &bad) * hlit(r, 4, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_NM_ATTN2: {
        int64_t nq = hlit(r, 4, &bad), nk = hlit(r, 5, &bad), t = hlit(r, 7, &bad);
        if (hlit(r, 6, &bad) != NM_HD || nq != NM_QH * nk || (t * nm_nch((int)t) > NM_PARTS && t < NM_PARTS)) {
            bad = 1;
            break;
        }
        if (t >= NM_FA_MIN && nm_fa_on()) {
            /* a group: the tensor cores (k_nm_fattn: 8 tokens a block, no chunks to merge) */
            k_nm_fattn<<<dim3((unsigned)cdiv(t, 8), (unsigned)nk), 256, 0, s>>>(dr, denv);
            break;
        }
        k_nm_attn2<<<dim3((unsigned)nm_nch((int)t), (unsigned)nk, (unsigned)t), 256, 0, s>>>(dr, denv);
        k_nm_attn_merge<<<dim3((unsigned)nq, (unsigned)t), NM_HD, 0, s>>>(dr, denv);
        break;
    }
    case GP_NM_GROUP_MOE: {
        int64_t k = hlit(r, 13, &bad), L = hlit(r, 14, &bad), inner = hlit(r, 15, &bad);
        int64_t t = hlit(r, 16, &bad), E = hlit(r, 17, &bad);
        /* operand 21: the type of the experts (none: KQ_IQ2R; KQ_NV4: Super's, k_nm_gtile_nv4) */
        const int64_t xtype = r->tag[21] != GP_T_NONE ? hlit(r, 21, &bad) : KQ_IQ2R;
        if (xtype == KQ_NV4) {
            const int x2 = r->tag[19] != GP_T_NONE && hlit(r, 19, &bad) != 0;
            if (L % 32 != 0 || L > 1024 || inner % 32 != 0 || E > 4096 || !nm_tc_on()) {
                bad = 1;
                break;
            }
            const size_t su = nm_gtile_nv4_smem((int)L), sd = nm_gtile_nv4_smem((int)inner);
            static size_t set_u = 0, set_d = 0;      /* (the opt-in of the dynamic shared memory, once a size) */
            if (su > 48 * 1024 && su > set_u) {
                cudaFuncSetAttribute(k_nm_gtile_nv4<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)su);
                cudaFuncSetAttribute(k_nm_gtile_nv4<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)su);
                set_u = su;
            }
            if (sd > 48 * 1024 && sd > set_d) {
                cudaFuncSetAttribute(k_nm_gtile_nv4<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sd);
                cudaFuncSetAttribute(k_nm_gtile_nv4<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sd);
                set_d = sd;
            }
            k_nm_gqx<<<(unsigned)cdiv(t * L / 32, 128), 128, 0, s>>>(dr, denv);
            k_nm_gsort<<<1, 1024, 0, s>>>(dr, denv);
            dim3 gu((unsigned)E, (unsigned)cdiv(inner, GM_BM)), gd((unsigned)E, (unsigned)cdiv(L, GM_BM));
            /* (the up tiles quantize the act themselves: operand 7 unused). Operand 22 (or none): the experts in
             * the GPU's memory (the stage buffers: k_nm_gtile_nv4c, 64 x 64 tiles) */
            if (r->tag[22] != GP_T_NONE && hlit(r, 22, &bad) == 2 && L % 64 == 0 && inner % 64 == 0) {
                /* the experts in NV4T (the stage buffers by k_nm_tile_pull): k_nm_gtile_nv4t */
                dim3 cu((unsigned)(inner / 64), (unsigned)E), cd((unsigned)(L / 64), (unsigned)E);
                if (x2) {
                    k_nm_gtile_nv4t<false, true><<<cu, 256, 0, s>>>(dr, denv);
                    k_nm_gtile_nv4t<true, true><<<cd, 256, 0, s>>>(dr, denv);
                } else {
                    k_nm_gtile_nv4t<false, false><<<cu, 256, 0, s>>>(dr, denv);
                    k_nm_gtile_nv4t<true, false><<<cd, 256, 0, s>>>(dr, denv);
                }
            } else if (r->tag[22] != GP_T_NONE && hlit(r, 22, &bad) != 0 && L % 64 == 0 && inner % 64 == 0 && nm_gc_on()) {
                dim3 cu((unsigned)(inner / 64), (unsigned)E), cd((unsigned)(L / 64), (unsigned)E);
                if (x2) {
                    k_nm_gtile_nv4c<false, true><<<cu, 256, 0, s>>>(dr, denv);
                    k_nm_gtile_nv4c<true, true><<<cd, 256, 0, s>>>(dr, denv);
                } else {
                    k_nm_gtile_nv4c<false, false><<<cu, 256, 0, s>>>(dr, denv);
                    k_nm_gtile_nv4c<true, false><<<cd, 256, 0, s>>>(dr, denv);
                }
            } else if (x2) {
                k_nm_gtile_nv4<false, true><<<gu, 256, su, s>>>(dr, denv);
                k_nm_gtile_nv4<true, true><<<gd, 256, sd, s>>>(dr, denv);
            } else {
                k_nm_gtile_nv4<false, false><<<gu, 256, su, s>>>(dr, denv);
                k_nm_gtile_nv4<true, false><<<gd, 256, sd, s>>>(dr, denv);
            }
            k_nm_gsum<<<dim3((unsigned)cdiv(L, 256), (unsigned)t), 256, 0, s>>>(dr, denv);
            break;
        }
        if (L % 256 != 0 || L > 1024 || inner % 256 != 0 || inner > 256 * NM_GMAXB || E > 4096) {
            bad = 1;
            break;
        }
        k_nm_gqx<<<(unsigned)cdiv(t * L / 32, 128), 128, 0, s>>>(dr, denv);
        k_nm_gsort<<<1, 1024, 0, s>>>(dr, denv);
        {
            /* the tiles (k_nm_gtile): the raw rows of 32 of an expert in shared memory once */
            const int x2 = r->tag[19] != GP_T_NONE && hlit(r, 19, &bad) != 0;
            const size_t su = nm_gtile_smem((int)L), sd = nm_gtile_smem((int)inner);
            dim3 gu((unsigned)E, (unsigned)cdiv(inner, GM_BM)), gd((unsigned)E, (unsigned)cdiv(L, GM_BM));
            if (nm_tc_on()) {
                /* (the up tiles quantize the act themselves, operands 8, 9, 20: no k_nm_gqa, which would
                 * quantize operand 7 over them, the float act these tiles no longer write) */
                if (x2) {
                    k_nm_gtile_tc<false, true><<<gu, 256, su, s>>>(dr, denv);
                    k_nm_gtile_tc<true, true><<<gd, 256, sd, s>>>(dr, denv);
                } else {
                    k_nm_gtile_tc<false, false><<<gu, 256, su, s>>>(dr, denv);
                    k_nm_gtile_tc<true, false><<<gd, 256, sd, s>>>(dr, denv);
                }
            } else if (x2) {
                k_nm_gtile<false, true><<<gu, 256, su, s>>>(dr, denv);
                k_nm_gqa<<<(unsigned)cdiv(t * k * inner / 32, 128), 128, 0, s>>>(dr, denv);
                k_nm_gtile<true, true><<<gd, 256, sd, s>>>(dr, denv);
            } else {
                k_nm_gtile<false, false><<<gu, 256, su, s>>>(dr, denv);
                k_nm_gqa<<<(unsigned)cdiv(t * k * inner / 32, 128), 128, 0, s>>>(dr, denv);
                k_nm_gtile<true, false><<<gd, 256, sd, s>>>(dr, denv);
            }
        }
        k_nm_gsum<<<dim3((unsigned)cdiv(L, 256), (unsigned)t), 256, 0, s>>>(dr, denv);
        break;
    }
    case GP_NM_GEMM8: {
        /* (RECLANK_NM_KS: the column shares of k_nm_gemm8_tc8, a test) */
        /* q, s, w, type, rows, cols, t, out (kernels/nemotron/nm.cu) */
        int64_t type = hlit(r, 3, &bad), rows = hlit(r, 4, &bad), cols = hlit(r, 5, &bad), t = hlit(r, 6, &bad);
        dim3 gr((unsigned)cdiv(rows, G8_BM), (unsigned)cdiv(t, G8_BN));
        const int x2 = r->tag[8] != GP_T_NONE && hlit(r, 8, &bad) != 0;
        if (cols % (type == KQ_IQ2R ? 256 : 32) != 0) {
            bad = 1;
        } else if (nm_tc_on() && t <= 8 && cols % 128 == 0 && (type == KQ_NV4 || type == KQ_IQ2R || type == KQ_Q8_0)) {
            /* rows on at most 8 tokens: k_nm_t8v (the weights streamed; the column shares of
             * nm_t8_ks), else k_nm_t8 */
            bool done = false;
            if (nm_t8v_on()) {
                if (type == KQ_NV4) {
                    done = x2 ? nm_t8v_launch<KQ_NV4, true>(rows, cols, t, s, dr, denv, g->nv8, g->nv8_n)
                              : nm_t8v_launch<KQ_NV4, false>(rows, cols, t, s, dr, denv, g->nv8, g->nv8_n);
                } else if (type == KQ_IQ2R) {
                    done = x2 ? nm_t8v_launch<KQ_IQ2R, true>(rows, cols, t, s, dr, denv, g->nv8, g->nv8_n)
                              : nm_t8v_launch<KQ_IQ2R, false>(rows, cols, t, s, dr, denv, g->nv8, g->nv8_n);
                } else {
                    done = x2 ? nm_t8v_launch<KQ_Q8_0, true>(rows, cols, t, s, dr, denv, g->nv8, g->nv8_n)
                              : nm_t8v_launch<KQ_Q8_0, false>(rows, cols, t, s, dr, denv, g->nv8, g->nv8_n);
                }
            }
            if (done) {
                break;
            }
            const int64_t blocks = cdiv(rows, NV8_BM);
            int ks = blocks >= 120 ? 1 : blocks >= 60 ? 2 : 4;
            float *part = g->nv8;
            if ((size_t)ks * t * rows > g->nv8_n) {
                ks = 1;
            }
            dim3 gv((unsigned)blocks, (unsigned)ks);
            if (type == KQ_NV4) {
                if (x2) k_nm_t8<KQ_NV4, true><<<gv, 256, 0, s>>>(dr, denv, part);
                else k_nm_t8<KQ_NV4, false><<<gv, 256, 0, s>>>(dr, denv, part);
            } else if (type == KQ_IQ2R) {
                if (x2) k_nm_t8<KQ_IQ2R, true><<<gv, 256, 0, s>>>(dr, denv, part);
                else k_nm_t8<KQ_IQ2R, false><<<gv, 256, 0, s>>>(dr, denv, part);
            } else {
                if (x2) k_nm_t8<KQ_Q8_0, true><<<gv, 256, 0, s>>>(dr, denv, part);
                else k_nm_t8<KQ_Q8_0, false><<<gv, 256, 0, s>>>(dr, denv, part);
            }
            if (ks > 1) {
                k_nm_t8_sum<<<(unsigned)cdiv(t * rows, 256), 256, 0, s>>>(dr, denv, part, ks);
            }
        } else if (!nm_tc_on() && t <= 8 && nm_t8d_on() && (type != KQ_IQ2R || cols % 256 == 0)) {
            /* no tensor cores (Pascal): a group of at most 8 tokens with dp4a, the blocks of a lane
             * decoded once for all the tokens (k_nm_t8d; RECLANK_NM_T8D: the rows of a warp) */
            const int R8 = nm_t8d_on();
            const unsigned nbk = (unsigned)cdiv(rows, 8 * R8), cap = (unsigned)(nv4_sms() * 8);
            const unsigned nb8 = nbk < cap ? nbk : cap;
            if (type == KQ_NV4) {
                if (R8 == 1) k_nm_t8d<KQ_NV4, 1><<<nb8, 256, 0, s>>>(dr, denv);
                else if (R8 == 4) k_nm_t8d<KQ_NV4, 4><<<nb8, 256, 0, s>>>(dr, denv);
                else k_nm_t8d<KQ_NV4, 2><<<nb8, 256, 0, s>>>(dr, denv);
            } else if (type == KQ_IQ2R) {
                if (R8 == 1) k_nm_t8d<KQ_IQ2R, 1><<<nb8, 256, 0, s>>>(dr, denv);
                else if (R8 == 4) k_nm_t8d<KQ_IQ2R, 4><<<nb8, 256, 0, s>>>(dr, denv);
                else k_nm_t8d<KQ_IQ2R, 2><<<nb8, 256, 0, s>>>(dr, denv);
            } else {
                if (R8 == 1) k_nm_t8d<KQ_Q8_0, 1><<<nb8, 256, 0, s>>>(dr, denv);
                else if (R8 == 4) k_nm_t8d<KQ_Q8_0, 4><<<nb8, 256, 0, s>>>(dr, denv);
                else k_nm_t8d<KQ_Q8_0, 2><<<nb8, 256, 0, s>>>(dr, denv);
            }
        } else if (nm_tc_on() && t <= 8) {
            /* a group of at most 8 tokens: one n8 tile */
            /* the column shares of a row group: 4 warps a group for the narrow matrices */
            const int64_t groups = cdiv(rows, 16);
            const int ks = nm_ks_env() > 0 ? nm_ks_env() : groups >= 1024 ? 1 : groups >= 512 ? 2 : 4;
            dim3 g8((unsigned)cdiv(rows, G8S_BM / ks), (unsigned)ks);
            if (type == KQ_IQ2R) {
                if (x2) k_nm_gemm8_tc8<KQ_IQ2R, true><<<g8, 128, 0, s>>>(dr, denv);
                else k_nm_gemm8_tc8<KQ_IQ2R, false><<<g8, 128, 0, s>>>(dr, denv);
            } else if (type == KQ_NV4) {
                if (x2) k_nm_gemm8_tc8<KQ_NV4, true><<<g8, 128, 0, s>>>(dr, denv);
                else k_nm_gemm8_tc8<KQ_NV4, false><<<g8, 128, 0, s>>>(dr, denv);
            } else if (type == KQ_Q8_0) {
                if (x2) k_nm_gemm8_tc8<KQ_Q8_0, true><<<g8, 128, 0, s>>>(dr, denv);
                else k_nm_gemm8_tc8<KQ_Q8_0, false><<<g8, 128, 0, s>>>(dr, denv);
            } else {
                bad = 1;
            }
        } else if (nm_tc_on() && type == KQ_NV4 && rows % 64 == 0 && cols % 64 == 0 && nm_gc_on()) {
            /* KQ_NV4 rows of a group: 64 x 64 tiles (k_nm_gemm8_nv4c) */
            dim3 gc((unsigned)(rows / 64), (unsigned)cdiv(t, 64));
            if (x2) k_nm_gemm8_nv4c<true><<<gc, 256, 0, s>>>(dr, denv);
            else k_nm_gemm8_nv4c<false><<<gc, 256, 0, s>>>(dr, denv);
        } else if (nm_tc_on()) {
            /* the tensor cores (sm_80 up) */
            if (type == KQ_IQ2R) {
                if (x2) k_nm_gemm8_tc<KQ_IQ2R, true><<<gr, 256, 0, s>>>(dr, denv);
                else k_nm_gemm8_tc<KQ_IQ2R, false><<<gr, 256, 0, s>>>(dr, denv);
            } else if (type == KQ_NV4) {
                if (x2) k_nm_gemm8_tc<KQ_NV4, true><<<gr, 256, 0, s>>>(dr, denv);
                else k_nm_gemm8_tc<KQ_NV4, false><<<gr, 256, 0, s>>>(dr, denv);
            } else if (type == KQ_Q8_0) {
                if (x2) k_nm_gemm8_tc<KQ_Q8_0, true><<<gr, 256, 0, s>>>(dr, denv);
                else k_nm_gemm8_tc<KQ_Q8_0, false><<<gr, 256, 0, s>>>(dr, denv);
            } else {
                bad = 1;
            }
        } else if (r->tag[8] != GP_T_NONE && hlit(r, 8, &bad) != 0) {
            /* x as two int8 planes (operand 8: the low one) */
            if (type == KQ_IQ2R) {
                k_nm_gemm8k<KQ_IQ2R, true><<<gr, 256, 0, s>>>(dr, denv);
            } else if (type == KQ_NV4) {
                k_nm_gemm8k<KQ_NV4, true><<<gr, 256, 0, s>>>(dr, denv);
            } else if (type == KQ_Q8_0) {
                k_nm_gemm8k<KQ_Q8_0, true><<<gr, 256, 0, s>>>(dr, denv);
            } else {
                bad = 1;
            }
        } else if (type == KQ_IQ2R) {
            k_nm_gemm8k<KQ_IQ2R, false><<<gr, 256, 0, s>>>(dr, denv);
        } else if (type == KQ_NV4) {
            k_nm_gemm8k<KQ_NV4, false><<<gr, 256, 0, s>>>(dr, denv);
        } else if (type == KQ_Q8_0) {
            k_nm_gemm8k<KQ_Q8_0, false><<<gr, 256, 0, s>>>(dr, denv);
        } else {
            bad = 1;
        }
        break;
    }
    case GP_NM_QX:
        k_nm_qx<<<(unsigned)cdiv(hlit(r, 1, &bad) * hlit(r, 2, &bad) / 32, 128), 128, 0, s>>>(dr, denv);
        break;
    case GP_NM_DPREP: {
        /* (no q: a null address) */
        const int64_t nh = (r->tag[0] == GP_T_NONE || hlit(r, 0, &bad) == 0 ? 0 : hlit(r, 9, &bad)) + hlit(r, 10, &bad);
        const int64_t hd = hlit(r, 11, &bad);
        if (hd > 512 || hd % 32 != 0) {
            bad = 1;
            break;
        }
        k_nm_dprep<<<(unsigned)(hlit(r, 8, &bad) * nh), (unsigned)hd, 0, s>>>(dr, denv);
        break;
    }
    case GP_NM_DATTN:
        if (hlit(r, 8, &bad) != 128 || hlit(r, 11, &bad) + hlit(r, 5, &bad) > NM_DWIN) {
            bad = 1;
            break;
        }
        k_nm_dattn<<<(unsigned)(hlit(r, 5, &bad) * hlit(r, 6, &bad)), 128, 0, s>>>(dr, denv);
        break;
    case GP_NM_DCONV:
        k_nm_dconv<<<(unsigned)cdiv(hlit(r, 4, &bad) * hlit(r, 5, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_NM_LTOPK:
        if (hlit(r, 3, &bad) > NM_LK) {
            bad = 1;
            break;
        }
        k_nm_ltopk<<<(unsigned)hlit(r, 1, &bad), 1024, 0, s>>>(dr, denv);
        break;
    case GP_NM_LATT:
        if (hlit(r, 3, &bad) > NM_LN || hlit(r, 5, &bad) > 128) {
            bad = 1;
            break;
        }
        k_nm_latt<<<dim3((unsigned)hlit(r, 3, &bad), (unsigned)hlit(r, 4, &bad)), 128, 0, s>>>(dr, denv);
        break;
    case GP_NM_LROWS:
        k_nm_lrows<<<(unsigned)cdiv(hlit(r, 4, &bad) * hlit(r, 5, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_NM_LCAT:
        k_nm_lcat<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_NM_LSCORE:
        if (hlit(r, 5, &bad) > NM_LK) {
            bad = 1;
            break;
        }
        k_nm_lscore<<<(unsigned)hlit(r, 4, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_NM_ADDNORM:
        k_nm_addnorm<<<(unsigned)hlit(r, 5, &bad), 1024, 0, s>>>(dr, denv);
        break;
    case GP_NM_IQ2_LIN:
        if (hlit(r, 4, &bad) % 256 != 0) {
            bad = 1;
            break;
        }
        if (hlit(r, 5, &bad) > MT_MAX && hlit(r, 4, &bad) <= 256 * NM_LMAXB) {
            k_nm_iq2_gemm<<<(unsigned)cdiv(hlit(r, 3, &bad), NM_GROWS), 256, 0, s>>>(dr, denv);
        } else if (nm_q2_form() == 0) {
            k_nm_iq2_lin<<<(unsigned)cdiv(hlit(r, 3, &bad), 4 * NM_LR), 128, 0, s>>>(dr, denv);
        } else {
            /* (RECLANK_NM_Q2 = R U B: the rows of a warp, the chunks in flight, the blocks an SM; a test) */
            const int f = nm_q2_form() % 1000, R = f / 100, U = (f / 10) % 10, B = f % 10;
            const bool x2 = r->tag[7] == GP_T_SLOT || (r->tag[7] == GP_T_INT && r->v[7] != 0);    /* (none: T_INT 0) */
            unsigned nb = (unsigned)cdiv(hlit(r, 3, &bad), 8 * R), cap = (unsigned)(nv4_sms() * B);
            nb = nb < cap || B == 0 ? nb : cap;
#define Q2L(RR, UU) else if (R == RR && U == UU) nm_q2_lin_launch<RR, UU>(x2, nb, s, dr, denv)
            if (0) {}
            Q2L(1, 1); Q2L(1, 2); Q2L(1, 4); Q2L(2, 1); Q2L(2, 2); Q2L(2, 4); Q2L(4, 1); Q2L(4, 2); Q2L(8, 1);
            else bad = 1;
#undef Q2L
        }
        break;
    case GP_ROUTER_SIG:
        if (hlit(r, 2, &bad) > 1024 || hlit(r, 3, &bad) > 32) {
            bad = 1;
        }
        if (hlit(r, 2, &bad) > 128 && router_rank_on()) {
            /* (operand 10, the split: one block for all the rows) */
            int n = 64;
            while (n < hlit(r, 2, &bad)) {
                n *= 2;
            }
            if (r->tag[10] != GP_T_NONE && hlit(r, 1, &bad) > 32) {
                bad = 1;
                break;
            }
            const gp_rec *n1 = r + 1;
            if (r->tag[10] != GP_T_NONE && n1 < g->hcode + g->n_code && n1->op == GP_SIGNAL && ds_route_fuse_on()) {
                /* (a step's router and its split, and the signal after: k_nm_route_signal, one kernel) */
                k_nm_route_signal<<<1, (unsigned)(n / 2), 0, s>>>(dr, denv, dr + 1);
                gg_skip = n1;
                break;
            }
            k_router_sig_r<<<r->tag[10] != GP_T_NONE ? 1u : (unsigned)hlit(r, 1, &bad), (unsigned)(n / 2), 0, s>>>(dr, denv);
        } else if (r->tag[9] != GP_T_NONE || r->tag[10] != GP_T_NONE) {
            bad = 1;            /* (the fused count and split: k_router_sig_r only) */
        } else if (hlit(r, 2, &bad) > 128) {
            k_router_sig_b<<<(unsigned)hlit(r, 1, &bad), 256, 0, s>>>(dr, denv);
        } else {
            k_router_sig<<<(unsigned)hlit(r, 1, &bad), 32, 0, s>>>(dr, denv);
        }
        break;
    case GP_ATTN_PREP: {
        int64_t hd = hlit(r, 14, &bad);
        if (hd > 1024 || hd % 32 != 0) {
            bad = 1;
        }
        k_attn_prep<<<(unsigned)(hlit(r, 11, &bad) * (hlit(r, 12, &bad) + hlit(r, 13, &bad))),
                      (unsigned)hd, 0, s>>>(dr, denv);
        break;
    }
    case GP_GDN: {
        /* k_dim = v_dim = 128 */
        if (hlit(r, 16, &bad) != 128 || hlit(r, 17, &bad) != 128 ||
            hlit(r, 3, &bad) > 9) {
            bad = 1;
        }
        int64_t cd = 2 * hlit(r, 14, &bad) * 128 + hlit(r, 15, &bad) * 128;
        int64_t t = hlit(r, 13, &bad), kh = hlit(r, 14, &bad), vh = hlit(r, 15, &bad);
        int gs = gdn_par_on();
        int nolog = r->tag[19] == GP_T_NONE || (r->tag[19] == GP_T_INT && r->v[19] == 0);
        if (nolog && t >= GDN_PAR_MIN && (gs == 4 || gs == 8)) {
            /* a large group with no log: the parallel forms */
            k_gdn_conv_par<<<dim3((unsigned)cdiv(cd, T), (unsigned)t), T, 0, s>>>(dr, denv);
            k_gdn_prep<128><<<dim3((unsigned)t, (unsigned)(kh + 2)), 128, 0, s>>>(dr, denv);
            if (hlit(r, 20, &bad) & 4) {
                /* KDA: the decay of each channel */
                if (gs == 4) {
                    k_gdn_scan<128, 4, true, 16><<<dim3((unsigned)vh, 4), 128, 0, s>>>(dr, denv);
                } else {
                    k_gdn_scan<128, 8, true, 16><<<dim3((unsigned)vh, 8), 128, 0, s>>>(dr, denv);
                }
            } else if (gs == 4) {
                k_gdn_scan<128, 4><<<dim3((unsigned)vh, 4), 128, 0, s>>>(dr, denv);
            } else {
                k_gdn_scan<128, 8><<<dim3((unsigned)vh, 8), 128, 0, s>>>(dr, denv);
            }
            k_gdn_out<128><<<dim3((unsigned)t, (unsigned)vh), 128, 0, s>>>(dr, denv);
            break;
        }
        k_gdn_conv<<<(unsigned)cdiv(cd, T), T, 0, s>>>(dr, denv);
        k_gdn_heads<128><<<(unsigned)vh, 128, 0, s>>>(dr, denv);
        break;
    }
    case GP_FFN_OUT:
        if (hlit(r, 8, &bad) > FFN_COLS) {
            bad = 1;
        }
        k_ffn_out<<<(unsigned)hlit(r, 7, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_SIGNAL:
        k_signal<<<1, 256, 0, s>>>(dr, denv);
        break;
    case GP_AWAIT:
        k_await<<<1, 1024, 0, s>>>(dr, denv);
        break;
    case GP_D2H:
        CK(cudaMemcpyAsync((void *)(intptr_t)hlit(r, 1, &bad),
                           (const void *)(intptr_t)hlit(r, 0, &bad), (size_t)hlit(r, 2, &bad),
                           cudaMemcpyDeviceToHost, s));
        break;
    case GP_H2D:
        CK(cudaMemcpyAsync((void *)(intptr_t)hlit(r, 1, &bad),
                           (const void *)(intptr_t)hlit(r, 0, &bad), (size_t)hlit(r, 2, &bad),
                           cudaMemcpyHostToDevice, s));
        break;
    case GP_CPU_TASK:
        /* no kernel: the runner runs it (gg_exec) */
        break;
    case GP_MOE_COMBINE: {
        /* the routed experts of the GPU (its de) and the rows of the CPU */
        /* (the contract's combine unless operand 12 says the experts are not its: FP4-Local's MTP
         * layer from the BF12 file) */
        int64_t t = hlit(r, 9, &bad), hidden = hlit(r, 8, &bad);
        const int plain = r->tag[12] != GP_T_NONE && hlit(r, 12, &bad) != 0;
        k_moe_combine<<<dim3((unsigned)cdiv(hidden, T), (unsigned)t), T, 0, s>>>(dr, denv,
                                                                            plain ? 0 : (gg_moe_rot >> 2) & 1);
        break;
    }
    case GP_DS_MXX_LINEAR: {
        /* x (t x cols float32), w (KQ_MXX), out (t x rows), rows, cols, t, the
         * planes hi, lo and xs of x (kernels/dsv41/types.cu) */
        int64_t rows = hlit(r, 3, &bad), cols = hlit(r, 4, &bad), t = hlit(r, 5, &bad);
        if (bad || rows % 16 != 0 || cols % 32 != 0 || cols / 32 > MXX_NB) {
            bad = 1;
            break;
        }
        size_t blocks = (size_t)t * (cols / 32);
        int8_t *qh = (int8_t *)(intptr_t)hi(r, g->henv, 6), *ql = (int8_t *)(intptr_t)hi(r, g->henv, 7);
        float *xs = (float *)(intptr_t)hi(r, g->henv, 8);
        k_quant_x2<<<(unsigned)cdiv(blocks * 8, 256), 256, 0, s>>>(
            (const float *)(intptr_t)hi(r, g->henv, 0), qh, ql, xs, NULL, blocks);
        k_mxx_rows_x2<8><<<dim3((unsigned)(rows / 16), (unsigned)t), 32 * 8, 0, s>>>(
            (const uint8_t *)(intptr_t)hi(r, g->henv, 1), (int)cols, qh, ql, xs,
            (float *)(intptr_t)hi(r, g->henv, 2), (int)rows);
        break;
    }
    case GP_F16_GLINEAR: {
        /* x (t x groups cols), w (float16), out (t x rows), rows, cols, t, groups */
        int64_t rows = hlit(r, 3, &bad), cols = hlit(r, 4, &bad), t = hlit(r, 5, &bad);
        int64_t groups = hlit(r, 6, &bad) > 1 ? hlit(r, 6, &bad) : 1;
        if (bad || cols % F16G_BK != 0 || cols % 8 != 0 || rows % groups != 0 ||
            (groups > 1 && (rows / groups) % F16G_BM != 0)) {
            bad = 1;
            break;
        }
        if (t > 8) {
            k_f16g_tile<<<dim3((unsigned)cdiv(rows, F16G_BM), (unsigned)cdiv(t, F16G_BN)), 256, 0, s>>>(dr, denv);
        } else if (t > 4) {
            k_f16g_gemv<8><<<(unsigned)cdiv(rows, 8), 256, 0, s>>>(dr, denv);
        } else if (t > 1) {
            k_f16g_gemv<4><<<(unsigned)cdiv(rows, 8), 256, 0, s>>>(dr, denv);
        } else {
            k_f16g_gemv<1><<<(unsigned)cdiv(rows, 8), 256, 0, s>>>(dr, denv);
        }
        break;
    }
    case GP_GLM_POOL: {
        /* (operand indices of GP_DS_COMPRESS for k_ds_compress_state) */
        int64_t t = hlit(r, 10, &bad), hd = hlit(r, 8, &bad), ratio = hlit(r, 7, &bad);
        bad |= hd > 512 || hd % 32 != 0 || ratio > 8 || ratio < 1;
        if (bad) {
            break;
        }
        k_glm_ln<<<(unsigned)t, (unsigned)hd, 0, s>>>(dr, denv);
        k_glm_pool<<<(unsigned)(t / ratio + 1), (unsigned)hd, 0, s>>>(dr, denv);
        k_ds_compress_state<<<1, (unsigned)hd, 0, s>>>(dr, denv);
        break;
    }
    case GP_GLM_HC_MIX:
        bad |= hlit(r, 8, &bad) % GLM_HC_NCH != 0;
        k_glm_hc_part<<<dim3(GLM_HC_NCH, (unsigned)hlit(r, 7, &bad)), 256, 0, s>>>(dr, denv);
        k_glm_hc_coef<<<(unsigned)hlit(r, 7, &bad), 32, 0, s>>>(dr, denv);
        break;
    case GP_GLM_HC_POST:
        k_glm_hc_post<<<dim3((unsigned)cdiv(hlit(r, 5, &bad), 256), (unsigned)hlit(r, 4, &bad)), 256, 0, s>>>(dr, denv);
        break;
    case GP_GLM_ROPE:
        k_glm_rope<<<(unsigned)hlit(r, 2, &bad), 128, 0, s>>>(dr, denv);
        break;
    case GP_GLM_RING_PUT:
        k_glm_ring_put<<<dim3((unsigned)cdiv(hlit(r, 5, &bad), 1024), (unsigned)hlit(r, 3, &bad)), 256, 0, s>>>(dr, denv);
        break;
    case GP_GLM_DSP_ATTN:
        bad |= hlit(r, 9, &bad) != 64 || hlit(r, 7, &bad) + hlit(r, 10, &bad) > GLM_DSP_MAXK;
        k_glm_dsp_attn<<<dim3((unsigned)hlit(r, 8, &bad), (unsigned)cdiv(hlit(r, 10, &bad), GLM_DSP_QB)), 256, 0, s>>>(
            dr, denv);
        break;
    case GP_GLM_ATTN:
        /* q, lat (bfloat16), sel, cnt, out, H, hd, scale, maxk, t */
        bad |= hlit(r, 5, &bad) != 64 || hlit(r, 6, &bad) != 512 || !gg_caps.tc;
        if (bad) {
            break;
        }
        cudaFuncSetAttribute(k_glm_attn_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)GA_SMEM);
        k_glm_attn_tc<<<(unsigned)hlit(r, 9, &bad), 256, GA_SMEM, s>>>(dr, denv);
        break;
    case GP_GLM_EXPAND:
        bad |= hlit(r, 6, &bad) * hlit(r, 5, &bad) + hlit(r, 5, &bad) - 1 > hlit(r, 7, &bad);
        k_glm_expand<<<(unsigned)hlit(r, 8, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_FP8_LINEAR: {
        /* x (t x groups cols float32), codes, scales, out (t x rows), rows, cols, t,
         * groups (0 or 1: one) */
        int64_t rows = hlit(r, 4, &bad), cols = hlit(r, 5, &bad), t = hlit(r, 6, &bad);
        int64_t groups = hlit(r, 7, &bad) > 1 ? hlit(r, 7, &bad) : 1;
        if (bad || cols % 128 != 0 || rows % groups != 0 || (rows / groups) % 32 != 0) {
            bad = 1;
            break;
        }
        if (t > 8 && gg_caps.tc && (rows / groups) % F8T_BM == 0) {
            k_fp8b32_tc<<<dim3((unsigned)(rows / F8T_BM), (unsigned)cdiv(t, F8T_BN)), 128, 0, s>>>(
                (const uint8_t *)(intptr_t)hi(r, g->henv, 1), (const uint8_t *)(intptr_t)hi(r, g->henv, 2),
                (int)rows, (int)cols, (const float *)(intptr_t)hi(r, g->henv, 0), (int)t,
                (float *)(intptr_t)hi(r, g->henv, 3), (int)groups);
            break;
        }
        {
            /* a step: a warp a row; a few tokens (a verify group): each weight
             * read once for them */
            const uint8_t *fw = (const uint8_t *)(intptr_t)hi(r, g->henv, 1);
            const uint8_t *fs = (const uint8_t *)(intptr_t)hi(r, g->henv, 2);
            const float *fx = (const float *)(intptr_t)hi(r, g->henv, 0);
            float *fo = (float *)(intptr_t)hi(r, g->henv, 3);
            /* R rows a warp while that leaves 128 blocks or more (the loads in
             * flight); the tokens 2, 4 or 8 at a time */
            int R = rows >= 32 * 128 ? 4 : rows >= 16 * 128 ? 2 : 1;
            unsigned nblk = (unsigned)cdiv(rows, 8 * R);
#define FP8_GEMV(RR, NN) k_fp8b32_gemv<RR, NN><<<nblk, 256, 0, s>>>(fw, fs, (int)rows, (int)cols, fx, (int)t, fo, (int)groups)
#define FP8_GEMV_T(NN) \
    do { \
        if (R == 4) { \
            FP8_GEMV(4, NN); \
        } else if (R == 2) { \
            FP8_GEMV(2, NN); \
        } else { \
            FP8_GEMV(1, NN); \
        } \
    } while (0)
            if (t == 1) {
                nblk = (unsigned)cdiv(rows, 8);
                FP8_GEMV(1, 1);
            } else if (t == 2) {
                FP8_GEMV_T(2);
            } else if (t <= 4) {
                FP8_GEMV_T(4);
            } else {
                FP8_GEMV_T(8);
            }
#undef FP8_GEMV_T
#undef FP8_GEMV
        }
        break;
    }
    /* the records of a step (kernels/dsv41/step.cu): the kernels read their
     * operands from the record (the position is a slot) */
    case GP_DS_HC_MIX:
        k_ds_hc_mix<<<dim3(DS_NMIX, (unsigned)hlit(r, 3, &bad)), 256, 0, s>>>(dr, denv);
        bad |= hlit(r, 4, &bad) % 4 != 0;
        break;
    case GP_DS_HC_PRE:
        /* (with operand 14, coef: t blocks more for the post's coefficients) */
        k_ds_hc_pre<<<(unsigned)hlit(r, 6, &bad) * (r->tag[14] != GP_T_NONE && r->v[14] != 0 ? 2 : 1), 1024, 0,
                      s>>>(dr, denv);
        break;
    case GP_DS_HC_POST:
        k_ds_hc_post<<<dim3((unsigned)cdiv(hlit(r, 8, &bad), 256), (unsigned)hlit(r, 7, &bad)), 256, 0, s>>>(dr,
                                                                                                           denv);
        break;
    case GP_DS_HC_STEP: {
        int64_t dim = hlit(r, 19, &bad);
        bad |= dim % DS_HCS_C != 0 || dim / DS_HCS_C > DS_HCS_NB;
        k_ds_hc_step<<<dim3((unsigned)(dim / DS_HCS_C), (unsigned)hlit(r, 18, &bad)), DS_HCS_T, 0, s>>>(dr, denv);
        break;
    }
    case GP_DS_ROPE:
        k_ds_rope<<<dim3((unsigned)hlit(r, 5, &bad), (unsigned)hlit(r, 4, &bad)), (unsigned)hlit(r, 7, &bad) / 2, 0,
                    s>>>(dr, denv);
        break;
    case GP_DS_FP4:
        bad |= hlit(r, 1, &bad) % hlit(r, 2, &bad) != 0;
        k_ds_fp4<<<(unsigned)cdiv(hlit(r, 1, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_WIN_WRITE:
        bad |= hlit(r, 7, &bad) > 512 || hlit(r, 7, &bad) % 32 != 0;
        k_ds_win_write<<<(unsigned)hlit(r, 5, &bad), (unsigned)hlit(r, 7, &bad), 0, s>>>(dr, denv);
        break;
    case GP_DS_RING_PUT:
        bad |= hlit(r, 5, &bad) > 512;
        k_ds_ring_put<<<(unsigned)hlit(r, 3, &bad), (unsigned)hlit(r, 5, &bad), 0, s>>>(dr, denv);
        break;
    case GP_DS_COMPRESS:
        bad |= hlit(r, 8, &bad) > 512 || hlit(r, 8, &bad) % 32 != 0 || hlit(r, 7, &bad) > 8;
        k_ds_compress<<<(unsigned)hlit(r, 11, &bad), (unsigned)hlit(r, 8, &bad), 0, s>>>(dr, denv);
        if (hlit(r, 7, &bad) > 1) {
            k_ds_compress_state<<<1, (unsigned)hlit(r, 8, &bad), 0, s>>>(dr, denv);
        }
        break;
    case GP_DS_CMP_WRITE:
        bad |= hlit(r, 9, &bad) > 512 || hlit(r, 9, &bad) % 32 != 0 || hlit(r, 10, &bad) % 32 != 0 ||
               hlit(r, 10, &bad) > hlit(r, 9, &bad);
        k_ds_cmp_write<<<(unsigned)hlit(r, 14, &bad), (unsigned)hlit(r, 9, &bad), 0, s>>>(dr, denv);
        break;
    case GP_DS_IDX_SCORE: {
        bad |= hlit(r, 9, &bad) != 32 || hlit(r, 10, &bad) > 128 || hlit(r, 10, &bad) % 4 != 0;
        int64_t nbk = cdiv(hlit(r, 11, &bad), 8);
        int64_t lim = hlit(r, 12, &bad) > 1 ? 256 : 1024;
        k_ds_idx_score<<<dim3((unsigned)(nbk < lim ? nbk : lim), (unsigned)hlit(r, 12, &bad)), 256, 0, s>>>(dr,
                                                                                                             denv);
        break;
    }
    case GP_DS_CAND: {
        int64_t nbk = cdiv(hlit(r, 5, &bad), 256);
        k_ds_cand<<<dim3((unsigned)(nbk < 64 ? nbk : 64), (unsigned)hlit(r, 6, &bad)), 256, 0, s>>>(dr, denv);
        break;
    }
    case GP_DS_TOPK:
        k_ds_topk<<<(unsigned)hlit(r, 9, &bad), 1024, 0, s>>>(dr, denv);
        break;
    case GP_DS_TOPK_MERGE:
        bad |= hlit(r, 4, &bad) > DS_TOPK_MAXK || hlit(r, 3, &bad) < 1;
        k_ds_topk_merge<<<(unsigned)hlit(r, 9, &bad), 1024, 0, s>>>(dr, denv);
        break;
    case GP_DS_ATTN:
        bad |= hlit(r, 9, &bad) % DS_AHB != 0 || hlit(r, 10, &bad) > 512 || hlit(r, 10, &bad) % 128 != 0 ||
               hlit(r, 3, &bad) + hlit(r, 12, &bad) > DS_AMAX;
        if (r->tag[15] != GP_T_NONE && r->v[15] != 0) {
            /* a step: the keys in DS_ASPLIT ranges, then the join; with a keys
             * buffer (operand 19) the keys of each query decoded once first */
            if (r->tag[19] != GP_T_NONE && r->v[19] != 0) {
                bad |= hlit(r, 9, &bad) % DS_SHB != 0;
                int64_t nkq = (hlit(r, 3, &bad) + hlit(r, 12, &bad)) * (hlit(r, 10, &bad) / 4);
                k_ds_attn_keys<<<dim3((unsigned)cdiv(nkq, 256), (unsigned)hlit(r, 14, &bad)), 256, 0, s>>>(dr, denv);
                k_ds_attn_s<<<dim3((unsigned)(hlit(r, 9, &bad) / DS_SHB), (unsigned)hlit(r, 14, &bad), DS_ASPLIT), 256,
                              0, s>>>(dr, denv);
            } else {
                k_ds_attn<<<dim3((unsigned)(hlit(r, 9, &bad) / DS_AHB), (unsigned)hlit(r, 14, &bad), DS_ASPLIT), 256, 0,
                            s>>>(dr, denv);
            }
            k_ds_attn_join<<<dim3((unsigned)hlit(r, 9, &bad), (unsigned)hlit(r, 14, &bad)), 512, 0, s>>>(dr, denv);
            break;
        }
        if (hlit(r, 14, &bad) > 8 && hlit(r, 9, &bad) % DS_GHB == 0) {
            /* a group of a prompt: one pass, the keys decoded once for 16 heads (hd 512: in rounds of
             * DS_G2KC keys, k_ds_attn_g2; RECLANK_DS_ATTN_G2=0: k_ds_attn_g) */
            if (hlit(r, 10, &bad) == 512 && hlit(r, 9, &bad) % 64 == 0 && ds_attn_tc_on()) {
                cudaFuncSetAttribute(k_ds_attn_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DS_TSMEM);
                k_ds_attn_tc<<<dim3((unsigned)(hlit(r, 9, &bad) / 64), (unsigned)hlit(r, 14, &bad)), 256, DS_TSMEM,
                               s>>>(dr, denv);
            } else if (hlit(r, 10, &bad) == 512 && ds_attn_g2_on()) {
                k_ds_attn_g2<<<dim3((unsigned)(hlit(r, 9, &bad) / DS_GHB), (unsigned)hlit(r, 14, &bad)), 256, 0, s>>>(
                    dr, denv);
            } else {
                k_ds_attn_g<<<dim3((unsigned)(hlit(r, 9, &bad) / DS_GHB), (unsigned)hlit(r, 14, &bad)), 256, 0, s>>>(
                    dr, denv);
            }
            if (!bad && ds_dump_attn(g, r) != 0) {
                return -1;
            }
            break;
        }
        k_ds_attn<<<dim3((unsigned)(hlit(r, 9, &bad) / DS_AHB), (unsigned)hlit(r, 14, &bad)), 256, 0, s>>>(dr,
                                                                                                          denv);
        break;
    case GP_DS_ROUTER: {
        bad |= hlit(r, 5, &bad) > 512 || hlit(r, 6, &bad) > 16;
        const gp_rec *n1 = r + 1, *n2 = r + 2;
        if (hlit(r, 4, &bad) == 1 && n2 < g->hcode + g->n_code && n1->op == GP_HOT_SPLIT &&
            n2->op == GP_SIGNAL && hi(n1, g->henv, 0) == hi(r, g->henv, 2) && ds_route_fuse_on()) {
            /* a step: the router, the hot split and the signal in one kernel */
            k_ds_route_signal<<<1, 256, 0, s>>>(dr, denv, dr + 1, dr + 2);
            gg_skip = n2;
            break;
        }
        k_ds_router<<<(unsigned)hlit(r, 4, &bad), 256, 0, s>>>(dr, denv);
        break;
    }
    case GP_DS_SWIGLU:
        k_ds_swiglu<<<(unsigned)cdiv(hlit(r, 3, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_HOT_MOE: {
        int64_t t = hlit(r, 8, &bad), k = hlit(r, 9, &bad), H = hlit(r, 10, &bad), I = hlit(r, 11, &bad);
        if (bad || H % 32 != 0 || I % 32 != 0 || H / 32 > MXX_NB || I / 32 > MXX_NB) {
            bad = 1;
            break;
        }
        size_t xb = (size_t)t * (H / 32), ab = (size_t)t * k * (I / 32);
        /* operand 22: 8-bit activations (the measure; the CPU parts' flag 256) */
        int x8 = r->tag[22] != GP_T_NONE && r->v[22] != 0;
        if (x8) {
            k_quant_x2r<<<(unsigned)cdiv(xb * 8, 256), 256, 0, s>>>(
                (const float *)(intptr_t)hi(r, g->henv, 0), (int8_t *)(intptr_t)hi(r, g->henv, 12),
                (int8_t *)(intptr_t)hi(r, g->henv, 13), (float *)(intptr_t)hi(r, g->henv, 14), xb);
        } else {
            k_quant_x2<<<(unsigned)cdiv(xb * 8, 256), 256, 0, s>>>(
                (const float *)(intptr_t)hi(r, g->henv, 0), (int8_t *)(intptr_t)hi(r, g->henv, 12),
                (int8_t *)(intptr_t)hi(r, g->henv, 13), (float *)(intptr_t)hi(r, g->henv, 14), NULL, xb);
        }
        /* a group of a prompt: a block for each expert's rows, its pairs in turn
         * (the rows once; the pairs in chunks of DS_HOT_MAXP tokens); a step and
         * a verify group: a block for each pair */
        int grouped = t > 8;
        int64_t E = grouped ? hlit(r, 21, &bad) : 0;
        int tc6 = gg_caps.tc && ds_hot_tc6_on(t) && r->tag[21] == GP_T_INT &&
                  ds_hot_tc6_fits(H, I, r->v[21]) && ds_hot_hl_ints((size_t)(t * k), (size_t)r->v[21]) <= g->hl_n;
        if (tc6) {
            ds_hot_tc6(0, dr, denv, (const int *)(intptr_t)hi(r, g->henv, 1), (const int *)(intptr_t)hi(r, g->henv, 3),
                       g->hl, t * k, r->v[21], H, I, s);
        } else if (grouped && gg_caps.tc) {
            k_ds_hot_tc<false><<<dim3((unsigned)(I / 16), 2, (unsigned)E), 128, 0, s>>>(dr, denv);
        } else if (grouped) {
            k_ds_hot_gu_g<<<dim3((unsigned)(I / 16), 2, (unsigned)E), 256, 0, s>>>(dr, denv);
        } else {
            k_ds_hot_gu<<<dim3((unsigned)(I / 16), 2, (unsigned)(t * k)), 256, 0, s>>>(dr, denv);
        }
        k_ds_hot_act<<<dim3((unsigned)cdiv(I, 256), (unsigned)(t * k)), 256, 0, s>>>(dr, denv);
        if (x8) {
            k_quant_x2r<<<(unsigned)cdiv(ab * 8, 256), 256, 0, s>>>(
                (const float *)(intptr_t)hi(r, g->henv, 16), (int8_t *)(intptr_t)hi(r, g->henv, 17),
                (int8_t *)(intptr_t)hi(r, g->henv, 18), (float *)(intptr_t)hi(r, g->henv, 19), ab);
        } else {
            k_quant_x2<<<(unsigned)cdiv(ab * 8, 256), 256, 0, s>>>(
                (const float *)(intptr_t)hi(r, g->henv, 16), (int8_t *)(intptr_t)hi(r, g->henv, 17),
                (int8_t *)(intptr_t)hi(r, g->henv, 18), (float *)(intptr_t)hi(r, g->henv, 19), NULL, ab);
        }
        if (tc6) {
            ds_hot_tc6(1, dr, denv, NULL, NULL, g->hl, t * k, r->v[21], H, I, s);
        } else if (grouped && gg_caps.tc) {
            k_ds_hot_tc<true><<<dim3((unsigned)(H / 16), (unsigned)E), 128, 0, s>>>(dr, denv);
        } else if (grouped) {
            k_ds_hot_dn_g<<<dim3((unsigned)(H / 16), (unsigned)E), 256, 0, s>>>(dr, denv);
        } else {
            k_ds_hot_dn<<<dim3((unsigned)(H / 16), (unsigned)(t * k)), 256, 0, s>>>(dr, denv);
        }
        if (ds_dump_hot(g, r) != 0) {
            return -1;
        }
        break;
    }
    case GP_DS_STAGE_PICK:
        if (hlit(r, 5, &bad) > DS_STAGE_E || hlit(r, 6, &bad) > DS_STAGE_N) {
            bad = 1;
            break;
        }
        k_ds_stage_pick<<<1, 1024, 0, s>>>(dr, denv);
        break;
    case GP_DS_STAGE_PULL: {
        int ev = (int)hlit(r, 8, &bad);
        if (bad || ev < 0 || ev >= GG_SIDE_EV || gg_side_init() != 0) {
            bad = 1;
            break;
        }
        CK(cudaEventRecord(gg_side_fork, s));
        CK(cudaStreamWaitEvent(gg_side, gg_side_fork, 0));
        k_ds_stage_pull<<<(unsigned)hlit(r, 7, &bad), 256, 0, gg_side>>>(dr, denv);
        CK(cudaGetLastError());
        CK(cudaEventRecord(gg_side_ev[ev], gg_side));
        break;
    }
    case GP_DS_STAGE_JOIN: {
        int ev = (int)hlit(r, 0, &bad);
        if (bad || ev < 0 || ev >= GG_SIDE_EV || gg_side == NULL) {
            bad = 1;
            break;
        }
        CK(cudaStreamWaitEvent(s, gg_side_ev[ev], 0));
        break;
    }
    case GP_DS_COMBINE:
        k_ds_combine<<<dim3((unsigned)cdiv(hlit(r, 6, &bad), 256), (unsigned)hlit(r, 7, &bad)), 256, 0, s>>>(dr,
                                                                                                           denv);
        break;
    case GP_DS_RING_RESTORE:
        bad |= hlit(r, 6, &bad) > 512;
        k_ds_ring_restore<<<(unsigned)hlit(r, 3, &bad), (unsigned)hlit(r, 6, &bad), 0, s>>>(dr, denv);
        break;
    case GP_DS_STREAM_MEAN:
        k_ds_stream_mean<<<dim3((unsigned)cdiv(hlit(r, 3, &bad), 256), (unsigned)hlit(r, 2, &bad)), 256, 0, s>>>(
            dr, denv);
        break;
    case GP_DS_COMPRESS_FIX:
        bad |= hlit(r, 9, &bad) > 512;
        k_ds_compress_fix<<<1, (unsigned)hlit(r, 9, &bad), 0, s>>>(dr, denv);
        break;
    case GP_DS_GATHER:
        k_ds_gather<<<1, 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_MARKOV:
        k_ds_markov<<<1, 1024, 0, s>>>(dr, denv);
        break;
    case GP_DS_CONF:
        k_ds_conf<<<(unsigned)hlit(r, 4, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_EMBED:
        k_ds_embed<<<(unsigned)hlit(r, 3, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_EMBED_ROWS:
        k_ds_embed_rows<<<(unsigned)hlit(r, 3, &bad), 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_ENGRAM_ROWS:
        k_ds_engram_rows<<<(unsigned)cdiv(hlit(r, 3, &bad) * hlit(r, 4, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_ROWS_AT:
        k_ds_rows_at<<<(unsigned)cdiv(hlit(r, 4, &bad) * hlit(r, 6, &bad), 256), 256, 0, s>>>(dr, denv);
        break;
    case GP_DS_ENGRAM:
        bad |= hlit(r, 5, &bad) != DS_HC;
        k_ds_engram<<<dim3((unsigned)hlit(r, 5, &bad), (unsigned)hlit(r, 3, &bad)), 1024, 0, s>>>(dr, denv);
        break;
    case GP_KQ_HOT_MOE: {
        unsigned t = (unsigned)hlit(r, 16, &bad);
        unsigned pairs = (unsigned)hlit(r, 11, &bad) * t + t;
        int64_t inner = hlit(r, 12, &bad), hidden = hlit(r, 13, &bad);
        unsigned tk = (unsigned)hlit(r, 11, &bad) * t;
        if (kq_e3_k((int)hlit(r, 14, &bad)) && kq_e3_k((int)hlit(r, 15, &bad))) {
            /* MUL1 experts (kernels/quant/exl3.cu): batches of products */
            const int shared = hi(r, g->henv, 17) != 0;
            const int Ks = shared ? kq_e3_k((int)hlit(r, 20, &bad)) : 1;
            if (bad || !Ks || r->tag[22] != GP_T_NONE ||
                e3_hot_moe_launch(dr, denv, (int)t, (int)hlit(r, 11, &bad), (int)hidden, (int)inner,
                                  kq_e3_k((int)hlit(r, 14, &bad)), kq_e3_k((int)hlit(r, 15, &bad)), Ks, shared,
                                  (const float *)(intptr_t)hi(r, g->henv, 0), (float *)(intptr_t)hi(r, g->henv, 7),
                                  (const float *)(intptr_t)hi(r, g->henv, 8), (float *)(intptr_t)hi(r, g->henv, 9),
                                  hi(r, g->henv, 10) != 0, gg_moe_rot, g->e3, g->e3_n, s) != 0) {
                bad = 1;
            }
            break;
        }
        if (kqh_nvx_on() && hlit(r, 14, &bad) == KQ_NVX && hlit(r, 15, &bad) == KQ_NVX &&
            inner % 16 == 0 && hidden % 16 == 0) {
            /* the KQ_NVX experts with a block for each group (k_kqh_nvx),
             * the shared expert (pairs tk on) with k_kqh_gu and k_kqh_dn;
             * on int8 or int16 x, the operations of the CPU (k_kqh_nvx_q) */
            /* the contract of FP4-Local (gg_moe_rot 4): float32 x, k_kqh_nvx */
            const int ct = (gg_moe_rot >> 2) & 1;
            const int nx = ct ? 0 : kqh_x_mode();
            if (nx > 0 && hidden % 32 == 0 && inner % 32 == 0 && hidden / 32 <= KQH_NVX_NB &&
                inner / 32 <= KQH_NVX_NB && (size_t)t * hidden <= g->kqx_n &&
                (size_t)pairs * inner <= g->kqx_n && (nx == 1 || g->kqx_lo)) {
                const float *xs = (const float *)(g->kqx + g->kqx_n);
                const int8_t *xl = kt_lo(g);
                const float *hx = (const float *)(intptr_t)hi(r, g->henv, 0);
                const float *ax = (const float *)(intptr_t)hi(r, g->henv, 8);
                if (nx == 2) {
                    kt_quant_x2(g, hx, (size_t)t * hidden);
                    k_kqh_nvx_q<8, true><<<dim3((unsigned)(inner / 16), tk, 2), 32 * 8, 0, s>>>(
                        dr, denv, 0, g->kqx, xl, xs);
                } else {
                    kt_quant(g, hx, (size_t)t * hidden);
                    k_kqh_nvx_q<8, false><<<dim3((unsigned)(inner / 16), tk, 2), 32 * 8, 0, s>>>(
                        dr, denv, 0, g->kqx, xl, xs);
                }
                k_kqh_gu<<<dim3((unsigned)cdiv(inner, ROWS_PER_BLOCK), t, 2), W, 0, s>>>(dr, denv,
                                                                                     (int)tk);
                k_kqh_act<<<pairs, T, 0, s>>>(dr, denv, gg_moe_rot);
                if (nx == 2) {
                    kt_quant_x2(g, ax, (size_t)pairs * inner);
                    k_kqh_nvx_q<4, true><<<dim3((unsigned)(hidden / 16), tk, 1), 32 * 4, 0, s>>>(
                        dr, denv, 1, g->kqx, xl, xs);
                } else {
                    kt_quant(g, ax, (size_t)pairs * inner);
                    k_kqh_nvx_q<4, false><<<dim3((unsigned)(hidden / 16), tk, 1), 32 * 4, 0, s>>>(
                        dr, denv, 1, g->kqx, xl, xs);
                }
                k_kqh_dn<<<dim3((unsigned)cdiv(hidden, ROWS_PER_BLOCK), t), W, 0, s>>>(dr, denv,
                                                                                   (int)tk);
                if (hi(r, g->henv, 10) != 0) {      /* (none: GP_MOE_COMBINE sums) */
                    k_kqh_sum<<<dim3((unsigned)cdiv(hidden, T), t), T, 0, s>>>(dr, denv);
                }
                break;
            }
            if (ct && hidden / 16 <= KQH_CT_NH && inner / 16 <= KQH_CT_NH) {
                /* the contract with the operations of the CPU (k_kqh_nvx_ct) */
                k_kqh_nvx_ct<<<dim3((unsigned)(inner / 16), tk, 2), KQH_CT_T, 0, s>>>(dr, denv, 0);
            } else {
                k_kqh_nvx<8><<<dim3((unsigned)(inner / 16), tk, 2), 32 * 8, 0, s>>>(dr, denv, 0, ct);
            }
            k_kqh_gu<<<dim3((unsigned)cdiv(inner, ROWS_PER_BLOCK), t, 2), W, 0, s>>>(dr, denv,
                                                                                 (int)tk);
            k_kqh_act<<<pairs, T, 0, s>>>(dr, denv, gg_moe_rot);
            if (ct && hidden / 16 <= KQH_CT_NH && inner / 16 <= KQH_CT_NH) {
                k_kqh_nvx_ct<<<dim3((unsigned)(hidden / 16), tk, 1), KQH_CT_T, 0, s>>>(dr, denv, 1);
            } else {
                k_kqh_nvx<4><<<dim3((unsigned)(hidden / 16), tk, 1), 32 * 4, 0, s>>>(dr, denv, 1, ct);
            }
            k_kqh_dn<<<dim3((unsigned)cdiv(hidden, ROWS_PER_BLOCK), t), W, 0, s>>>(dr, denv,
                                                                               (int)tk);
            if (hi(r, g->henv, 10) != 0) {      /* (none: GP_MOE_COMBINE sums) */
                k_kqh_sum<<<dim3((unsigned)cdiv(hidden, T), t), T, 0, s>>>(dr, denv, ct);
            }
            break;
        }
        const int kx = kqh_x_mode(), gt = (int)hlit(r, 14, &bad), dt = (int)hlit(r, 15, &bad);
        const int st = (int)hlit(r, 20, &bad);
        if (kx > 0 && r->tag[22] == GP_T_NONE && hidden % 32 == 0 && inner % 32 == 0 &&
            (size_t)t * hidden <= g->kqx_n && (size_t)pairs * inner <= g->kqx_n &&
            (kx == 2 ? g->kqx_lo && kq_x2_type(gt) && kq_x2_type(dt)    /* (a shared expert of another type: kq_row on float32 x) */
                     : i8x_on() && kq_i8_type(gt) && kq_i8_type(dt) &&
                       (kq_i8_type(st) || hi(r, g->henv, 17) == 0))) {     /* (no shared expert: LFM2) */
            /* int16 x and act as two int8 planes (kq_row_x2), or int8 (kq_rows_i8) */
            const float *xs = (const float *)(g->kqx + g->kqx_n);
            const int8_t *xl = kx == 2 ? kt_lo(g) : NULL;
            const float *hx = (const float *)(intptr_t)hi(r, g->henv, 0);
            const float *ax = (const float *)(intptr_t)hi(r, g->henv, 8);
            /* int16: the routed experts of Q8_0 and Q6_K with the operations of
             * the CPU (k_kqh_x2c), the shared expert with kq_row_x2 */
            const int cx = kx == 2 && kq_x2c_type(gt, (int)hidden) && kq_x2c_type(dt, (int)inner);
            if (kx == 2) {
                kt_quant_x2(g, hx, (size_t)t * hidden);
            } else {
                kt_quant(g, hx, (size_t)t * hidden);
            }
            if (cx) {
                k_kqh_x2c<0><<<dim3((unsigned)cdiv(inner, 2 * ROWS_PER_BLOCK), tk, 2), W, 0, s>>>(
                    dr, denv, g->kqx, xl, xs);
                k_kqh_gu_i8<<<dim3((unsigned)cdiv(inner, ROWS_PER_BLOCK), t, 2), W, 0, s>>>(
                    dr, denv, g->kqx, xl, xs, (int)tk);
            } else {
                k_kqh_gu_i8<<<dim3((unsigned)cdiv(inner, ROWS_PER_BLOCK), pairs, 2), W, 0, s>>>(
                    dr, denv, g->kqx, xl, xs);
            }
            k_kqh_act<<<pairs, T, 0, s>>>(dr, denv, gg_moe_rot & ~4);     /* (the contract: KQ_NVX only) */
            if (kx == 2) {
                kt_quant_x2(g, ax, (size_t)pairs * inner);
            } else {
                kt_quant(g, ax, (size_t)pairs * inner);
            }
            if (cx) {
                k_kqh_x2c<1><<<dim3((unsigned)cdiv(hidden, 2 * ROWS_PER_BLOCK), tk, 1), W, 0, s>>>(
                    dr, denv, g->kqx, xl, xs);
                k_kqh_dn_i8<<<dim3((unsigned)cdiv(hidden, ROWS_PER_BLOCK), t), W, 0, s>>>(
                    dr, denv, g->kqx, xl, xs, (int)tk);
            } else {
                k_kqh_dn_i8<<<dim3((unsigned)cdiv(hidden, ROWS_PER_BLOCK), pairs), W, 0, s>>>(
                    dr, denv, g->kqx, xl, xs);
            }
            if (hi(r, g->henv, 10) != 0) {      /* (none: GP_MOE_COMBINE sums) */
                k_kqh_sum<<<dim3((unsigned)cdiv(hidden, T), t), T, 0, s>>>(dr, denv);
            }
            break;
        }
        /* the routed pairs of Q4_K and Q5_K experts: k_kqh_k (x in registers, rows a warp);
         * the shared expert (the pairs from tk) with k_kqh_gu and k_kqh_dn */
        const int kk = r->tag[22] == GP_T_NONE && kqh_k_ok(gt, (int)hidden) && kqh_k_ok(dt, (int)inner);
        if (kk) {
            kqh_k_launch(dr, denv, gt, (int)hidden, 0, tk, inner, s);
            k_kqh_gu<<<dim3((unsigned)cdiv(inner, ROWS_PER_BLOCK), t, 2), W, 0, s>>>(dr, denv, (int)tk);
        } else {
            k_kqh_gu<<<dim3((unsigned)cdiv(inner, ROWS_PER_BLOCK), pairs, 2), W, 0, s>>>(dr, denv);
        }
        k_kqh_act<<<pairs, T, 0, s>>>(dr, denv, gg_moe_rot & ~4);
        if (kk) {
            kqh_k_launch(dr, denv, dt, (int)inner, 1, tk, hidden, s);
            k_kqh_dn<<<dim3((unsigned)cdiv(hidden, ROWS_PER_BLOCK), t), W, 0, s>>>(dr, denv, (int)tk);
        } else {
            k_kqh_dn<<<dim3((unsigned)cdiv(hidden, ROWS_PER_BLOCK), pairs), W, 0, s>>>(dr, denv);
        }
        if (hi(r, g->henv, 10) != 0) {      /* (none: GP_MOE_COMBINE sums) */
            k_kqh_sum<<<dim3((unsigned)cdiv(hidden, T), t), T, 0, s>>>(dr, denv);
        }
        break;
    }
    default:
        snprintf(gg_error, sizeof(gg_error), gp_op_known(r->op) ? "no GPU kernel for operation %d"
                 : "operation %d is not in the table (reclank/ir/opcodes.py)", r->op);
        return -1;
    }
    if (bad) {
        snprintf(gg_error, sizeof(gg_error),
                 "operation %d: a size is not a literal, or the form is not supported", r->op);
        return -1;
    }
    return 0;
}
