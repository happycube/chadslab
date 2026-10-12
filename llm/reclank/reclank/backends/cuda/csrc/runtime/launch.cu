/* A value that sets the size of a launch. It must be a literal, because the
 * graph keeps the size of each launch. */
static int64_t hlit(const gp_rec *r, int k, int *bad)
{
    if (r->tag[k] == GP_T_SLOT) {
        *bad = 1;
    }
    return r->v[k];
}

static int64_t cdiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

/* The rows of up to four matrices from operand m0, as k_int4_multi4 reads
 * them. */
static int64_t multi4_rows(const gp_rec *r, int m0, int *bad)
{
    int64_t rows = 0;
    for (int m = 0; m < 4; ++m) {
        if (r->v[m0 + 4 * m] != 0) {
            rows += hlit(r, m0 + 3 + 4 * m, bad);
        }
    }
    return rows;
}

/* RECLANK_GPU_FD=0 keeps k_attn_part for the int16 cache of Qwen3.5 (a
 * test). */
static int fd_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_FD");
        on = !(v && v[0] == '0');
    }
    return on;
}

static int attn_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr,
                       const int64_t *denv, int *bad)
{
    int o = (r->op == GP_ATTN_QC || r->op == GP_ATTN_QSA || r->op == GP_ATTN_Q8 ||
             r->op == GP_ATTN_V8 || r->op == GP_ATTN_TQ) ? 7 : 5;
    /* (the operand of q_heads) */
    int64_t qh = hlit(r, o, bad), kvh = hlit(r, o + 1, bad), hd = hlit(r, o + 2, bad);
    int hg = qh % kvh == 0 ? (int)cdiv(qh / kvh, ATTN_REP) : 1;
    while (qh % kvh == 0 && (qh / kvh) % hg != 0) {
        ++hg;
    }
    if ((size_t)qh * ATTN_CHUNKS * (size_t)(hd + 2) > GG_PART_FLOATS ||
        qh % kvh != 0 || qh / kvh / hg > ATTN_REP || (hd != 256 && hd != 512) ||
        (r->op == GP_ATTN_QSA && hlit(r, 10, bad) != 1)) {
        *bad = 1;
        return 0;
    }
    if (r->op == GP_ATTN_QC && hd == 256 && qh == 8 * kvh && fd_on() &&
        (size_t)qh * FD_PARTS * 258 <= GG_PART_FLOATS) {
        k_attn_fd<<<dim3((unsigned)kvh, FD_BLOCKS), 128, 0, gg_stream>>>(dr, denv, g->part);
        k_attn_fd_join<<<dim3((unsigned)qh, 2), 128, 0, gg_stream>>>(dr, denv, g->part);
        return 0;
    }
    /* A global layer of the 26B on the tensor cores (k_attn_fdtc), with the
     * queries of a group (operand 11: t, reclank/backends/cuda/runtime.py attn_rows_small). */
    bool qc = r->op == GP_ATTN_QC || r->op == GP_ATTN_Q8 || r->op == GP_ATTN_V8;
    int64_t t = qc && r->tag[11] != GP_T_NONE ? hlit(r, 11, bad) : 1;
    /* a sliding layer of the 26B (operand 12: the window; k_attn_fdts) */
    if (qc && r->tag[12] != GP_T_NONE) {
        int64_t win = hlit(r, 12, bad);
        int ncs = (int)((win + FT_TMAX - 1 + FS_L - 1) / FS_L + 1);
        if (!(fdtc_on() && hd == 256 && qh == 2 * kvh && t >= 1 && t <= FT_TMAX && win > 0 &&
              ncs <= FS_MAXC && (size_t)t * qh * ncs * 258 <= GG_PART_FLOATS)) {
            *bad = 1;
            return 0;
        }
        dim3 grid((unsigned)kvh, (unsigned)ncs);
        if (t == 1) {
            fdts_run<1>(r->op, grid, dr, denv, g->part);
        } else if (t == 2) {
            fdts_run<2>(r->op, grid, dr, denv, g->part);
        } else if (t == 3) {
            fdts_run<3>(r->op, grid, dr, denv, g->part);
        } else {
            fdts_run<4>(r->op, grid, dr, denv, g->part);
        }
        k_attn_fdts_join<<<dim3((unsigned)(t * qh), 2), 128, 0, gg_stream>>>(dr, denv, g->part, ncs);
        return 0;
    }
    if (qc && fdtc_on() && hd == 512 && qh == 8 * kvh && t >= 1 && t <= FT_TMAX &&
        (size_t)t * qh * fdtc_chunks((int)kvh) * (hd + 2) <= GG_PART_FLOATS) {
        dim3 grid((unsigned)kvh, (unsigned)fdtc_chunks((int)kvh));
        if (t == 1) {
            fdtc_run<1>(r->op, grid, dr, denv, g->part);
        } else if (t == 2) {
            fdtc_run<2>(r->op, grid, dr, denv, g->part);
        } else if (t == 3) {
            fdtc_run<3>(r->op, grid, dr, denv, g->part);
        } else {
            fdtc_run<4>(r->op, grid, dr, denv, g->part);
        }
        k_attn_join<<<dim3((unsigned)(t * qh), (unsigned)cdiv(hd, 128)), 128, 0, gg_stream>>>(
            dr, denv, g->part, (int)grid.y);
        return 0;
    }
    if (t != 1) {
        *bad = 1;
        return 0;
    }
    /* The shapes of the Gemma 4 26B: one pass (k_attn_fdt). */
    if ((r->op == GP_ATTN_QC || r->op == GP_ATTN_Q8 || r->op == GP_ATTN_V8) && fd_on() &&
        (size_t)qh * FD_PARTS * (hd + 2) <= GG_PART_FLOATS &&
        ((hd == 256 && qh == 2 * kvh) || (hd == 512 && qh == 8 * kvh))) {
        dim3 grid((unsigned)kvh, FD_BLOCKS), jg((unsigned)qh, (unsigned)(hd / 128));
        bool q8 = r->op == GP_ATTN_Q8, v8 = r->op == GP_ATTN_V8;
        if (hd == 256) {
            if (q8) {
                k_attn_fdt<256, 2, true, true><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            } else if (v8) {
                k_attn_fdt<256, 2, false, true><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            } else {
                k_attn_fdt<256, 2><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            }
            k_attn_fdt_join<256, 2><<<jg, 128, 0, gg_stream>>>(dr, denv, g->part);
        } else {
            if (q8) {
                k_attn_fdt<512, 8, true, true><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            } else if (v8) {
                k_attn_fdt<512, 8, false, true><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            } else {
                k_attn_fdt<512, 8><<<grid, 128, 0, gg_stream>>>(dr, denv, g->part);
            }
            k_attn_fdt_join<512, 8><<<jg, 128, 0, gg_stream>>>(dr, denv, g->part);
        }
        return 0;
    }
    if (r->op == GP_ATTN_QSA && hd == 256 && qh / kvh <= 16 && qsa_tc_on() && part_tc_on() &&
        (hlit(r, 15, bad) == 1 || hlit(r, 15, bad) == 4)) {
        /* the heads of a key head in one tile (k_attn_part_tc) */
        static int attr = 0;
        if (!attr) {
            cudaFuncSetAttribute(k_attn_part_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)QT_SMEM);
            cudaFuncSetAttribute(k_attn_part_tc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)QT_SMEM);
            attr = 1;
        }
        dim3 grid((unsigned)kvh, ATTN_CHUNKS);
        if (qsa_tc_on() == 2) {
            k_attn_part_tc<0><<<grid, 128, QT_SMEM, gg_stream>>>(dr, denv, g->part);
        } else {
            k_attn_part_tc<1><<<grid, 128, QT_SMEM, gg_stream>>>(dr, denv, g->part);
        }
        k_attn_join<<<dim3((unsigned)qh, (unsigned)cdiv(hd, 128)), 128, 0, gg_stream>>>(
            dr, denv, g->part, 0);
        return 0;
    }
    k_attn_part<<<dim3((unsigned)(kvh * hg), ATTN_CHUNKS), 256, 0, gg_stream>>>(dr, denv, g->part,
                                                                              hg);
    k_attn_join<<<dim3((unsigned)qh, (unsigned)cdiv(hd, 128)), 128, 0, gg_stream>>>(
        dr, denv, g->part, 0);
    return 0;
}

/* The largest group for k_mt_gemv. A larger group uses k_gemm. A test can
 * lower it with rc_gpu_set_gemv_max to check k_gemm with a small group. */
static int gg_gemv_max = MT_MAX;

/* 1: the tensor cores for the int4 products of a large group (k_gemm_tc,
 * k_moe_gemm_tc). 0: the float32 kernels (k_gemm, k_moe_gemm). */
static int gg_tc = 1;

template <int HD, int HB, int ST>
static void flash3_run(const gp_rec *dr, const int64_t *denv, int kvh, int t, int G)
{
    typedef flash3_dims<HD> D;
    size_t smem = (size_t)ST * FK3 * (D::KLD + D::VLD) * sizeof(float) +
                  (D::NS == 1 ? 0 : (size_t)HB * 16 * D::QLD * sizeof(__half));
    cudaFuncSetAttribute(k_flash_f32h<HD, HB, ST>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((unsigned)kvh, (unsigned)cdiv(t, 16), (unsigned)(G / HB));
    k_flash_f32h<HD, HB, ST><<<grid, 32 * HB * D::NS, smem, gg_stream>>>(dr, denv);
}

/* Launch k_flash_f32h for a GP_ATTN_F32H record. Return -1 when it does not
 * take the shape. RECLANK_GPU_FLASH=1 selects k_flash_tc<1> for a test. */
static int flash3_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int *bad)
{
    static int old = -1;
    if (old < 0) {
        const char *v = getenv("RECLANK_GPU_FLASH");
        old = v && v[0] == '1';
    }
    if (old) {
        return -1;
    }
    int64_t qh = hlit(r, 5, bad), kvh = hlit(r, 6, bad), hd = hlit(r, 7, bad);
    int64_t t = hlit(r, 8, bad);
    if (kvh <= 0 || qh % kvh) {
        return -1;
    }
    int G = (int)(qh / kvh);
    if (hd == 256) {
        if (G % 4 == 0) {
            flash3_run<256, 4, 2>(dr, denv, (int)kvh, (int)t, G);
        } else if (G % 2 == 0) {
            flash3_run<256, 2, 2>(dr, denv, (int)kvh, (int)t, G);
        } else {
            flash3_run<256, 1, 2>(dr, denv, (int)kvh, (int)t, G);
        }
        return 0;
    }
    if (hd == 512) {
        if (G % 2 == 0) {
            flash3_run<512, 2, 1>(dr, denv, (int)kvh, (int)t, G);
        } else {
            flash3_run<512, 1, 1>(dr, denv, (int)kvh, (int)t, G);
        }
        return 0;
    }
    return -1;
}

template <int HD, int HB, int ST>
static void flash_qc_run(const gp_rec *dr, const int64_t *denv, int kvh, int t, int G)
{
    size_t smem = flash_qc_dims<HD>::smem(HB, ST);
    cudaFuncSetAttribute(k_flash_qc_tc<HD, HB, ST>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((unsigned)kvh, (unsigned)cdiv(t, 16), (unsigned)(G / HB));
    k_flash_qc_tc<HD, HB, ST><<<grid, 32 * HB * (HD / 256), smem, gg_stream>>>(dr, denv);
}

template <int HD, int HB, int NQT, int FKS = FK3, bool KQ8 = false, bool VQ8 = false,
          bool TQ = false>
static void flash_h_run(const gp_rec *dr, const int64_t *denv, int kvh, int t, int G)
{
    typedef flash_h_dims<HD, HB, NQT, FKS, KQ8, VQ8, TQ> D;
    size_t smem = D::smem();
    cudaFuncSetAttribute(k_flash_qc_h<HD, HB, NQT, FKS, KQ8, VQ8, TQ>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    dim3 grid((unsigned)kvh, (unsigned)cdiv(t, 16 * NQT), (unsigned)(G / HB));
    k_flash_qc_h<HD, HB, NQT, FKS, KQ8, VQ8, TQ><<<grid, 32 * D::WARPS, smem, gg_stream>>>(
        dr, denv);
}

/* RECLANK_GPU_FLASH_FK: the keys of a step of k_flash_qc_h for a head of
 * 256 values (16 or 32). For the 40 layers of the 12B with a window and
 * 8192 tokens: 255 ms with 16, 235 ms with 32. */
static int flash_fk(void)
{
    static int fk = -1;
    if (fk < 0) {
        const char *v = getenv("RECLANK_GPU_FLASH_FK");
        fk = v ? atoi(v) : 32;
    }
    return fk;
}

/* RECLANK_GPU_FLASH512: the head of 512 values takes k_flash_qc_h with HB
 * query heads in a block (4, or 2 for a test; 0 keeps k_flash_qc_tc). The
 * kernel takes 254 registers, so an SM runs one block: 4 warps with HB 2,
 * 8 with HB 4. For the 8 global layers of the 12B and a prompt of 8192
 * tokens (chunks of 2048): 714 ms with k_flash_qc_tc, 974 ms with HB 2,
 * 475 ms with HB 4. A key head with fewer than 4 query heads keeps
 * k_flash_qc_tc. RECLANK_GPU_FLASH256: the head of 256 values takes
 * k_flash_qc_h with 2 query heads and this many tiles of 16 queries in a
 * block (0 keeps k_flash_qc_tc). */
static int flash_env(const char *name, int dflt)
{
    const char *v = getenv(name);
    return v ? atoi(v) : dflt;
}

static int flash512_hb(void)
{
    static int hb = -1;
    if (hb < 0) {
        hb = flash_env("RECLANK_GPU_FLASH512", 4);
    }
    return hb;
}

static int flash256_qt(void)
{
    static int qt = -1;
    if (qt < 0) {
        qt = flash_env("RECLANK_GPU_FLASH256", 4);
    }
    return qt;
}

/* Launch k_flash_qc_tc for a GP_ATTN_QC_MT record. Return -1 when it does
 * not take the shape. RECLANK_GPU_FLASH=1 selects k_flash_tc<0> for a test. */
static int flash_qc_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int *bad)
{
    static int old = -1;
    if (old < 0) {
        const char *v = getenv("RECLANK_GPU_FLASH");
        old = v && v[0] == '1';
    }
    if (old) {
        return -1;
    }
    int64_t qh = hlit(r, 7, bad), kvh = hlit(r, 8, bad), hd = hlit(r, 9, bad);
    int64_t t = hlit(r, 10, bad);
    if (kvh <= 0 || qh % kvh) {
        return -1;
    }
    int G = (int)(qh / kvh);
    if (hd == 256 && G % 2 == 0 && flash256_qt() > 0) {
        switch (flash256_qt()) {
        case 1: flash_h_run<256, 2, 1>(dr, denv, (int)kvh, (int)t, G); break;
        case 2: flash_h_run<256, 2, 2>(dr, denv, (int)kvh, (int)t, G); break;
        default:
            if (flash_fk() == 32) {
                flash_h_run<256, 2, 4, 32>(dr, denv, (int)kvh, (int)t, G);
            } else {
                flash_h_run<256, 2, 4>(dr, denv, (int)kvh, (int)t, G);
            }
        }
        return 0;
    }
    if (hd == 256) {
        if (G % 4 == 0) {
            flash_qc_run<256, 4, 2>(dr, denv, (int)kvh, (int)t, G);
        } else if (G % 2 == 0) {
            flash_qc_run<256, 2, 2>(dr, denv, (int)kvh, (int)t, G);
        } else {
            flash_qc_run<256, 1, 2>(dr, denv, (int)kvh, (int)t, G);
        }
        return 0;
    }
    if (hd == 512) {
        if (flash512_hb() == 4 && G % 4 == 0) {
            flash_h_run<512, 4, 1>(dr, denv, (int)kvh, (int)t, G);
        } else if (flash512_hb() == 2 && G % 2 == 0) {
            flash_h_run<512, 2, 1>(dr, denv, (int)kvh, (int)t, G);
        } else if (G % 2 == 0) {
            flash_qc_run<512, 2, 1>(dr, denv, (int)kvh, (int)t, G);
        } else {
            flash_qc_run<512, 1, 1>(dr, denv, (int)kvh, (int)t, G);
        }
        return 0;
    }
    return -1;
}

/* Launch k_flash_f32t for a record of GP_ATTN_QC_MT (KQ8 0, VQ8 0),
 * GP_ATTN_Q8_MT (1, 1), GP_ATTN_V8_MT (0, 1), or GP_ATTN_F32H of a group
 * (f32h). Return -1 for a head_dim it does not take. */
template <bool F32H>
static int f32t_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int kq8, int vq8,
                       cudaStream_t s, int *bad, int tq = 0)
{
    int o0 = F32H ? 5 : 7;
    int64_t qh = hlit(r, o0, bad), kvh = hlit(r, o0 + 1, bad), hd = hlit(r, o0 + 2, bad);
    int64_t t = hlit(r, o0 + 3, bad);
    if (kvh <= 0 || qh % kvh || (hd != 256 && hd != 512)) {
        return -1;
    }
    dim3 grid((unsigned)qh, (unsigned)cdiv(t, hd == 256 ? 64 : 32));
#define F32T_K(HD) \
    if (F32H) { \
        k_flash_f32t<HD, false, false, true><<<grid, 256, 0, s>>>(dr, denv); \
    } else if (tq) { \
        k_flash_f32t<HD, false, false, false, true><<<grid, 256, 0, s>>>(dr, denv); \
    } else if (kq8) { \
        k_flash_f32t<HD, true, true><<<grid, 256, 0, s>>>(dr, denv); \
    } else if (vq8) { \
        k_flash_f32t<HD, false, true><<<grid, 256, 0, s>>>(dr, denv); \
    } else { \
        k_flash_f32t<HD, false, false><<<grid, 256, 0, s>>>(dr, denv); \
    }
    if (hd == 256) {
        F32T_K(256)
    } else {
        F32T_K(512)
    }
#undef F32T_K
    return 0;
}

/* k_flash_qc_h over the int8 values (GP_ATTN_Q8_MT; GP_ATTN_V8_MT with int16
 * keys: KQ8 false), the only kernel of a large group for them. Return -1 when it does not take the shape. */
template <bool KQ8>
static int flash_q8_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int *bad)
{
    int64_t qh = hlit(r, 7, bad), kvh = hlit(r, 8, bad), hd = hlit(r, 9, bad);
    int64_t t = hlit(r, 10, bad);
    if (kvh <= 0 || qh % kvh) {
        return -1;
    }
    int G = (int)(qh / kvh);
    if (hd == 256 && G % 2 == 0) {
        flash_h_run<256, 2, 4, 32, KQ8, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    if (hd == 512 && G % 4 == 0) {
        flash_h_run<512, 4, 1, FK3, KQ8, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    if (hd == 512 && G % 2 == 0) {
        flash_h_run<512, 2, 1, FK3, KQ8, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    return -1;
}

/* k_flash_qc_h over the TQ6 cache (GP_ATTN_TQ_MT: q and out rotated), the
 * only kernel of a large group for it. Return -1 when it does not take the
 * shape. */
static int flash_tq_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int *bad)
{
    int64_t qh = hlit(r, 7, bad), kvh = hlit(r, 8, bad), hd = hlit(r, 9, bad);
    int64_t t = hlit(r, 10, bad);
    if (kvh <= 0 || qh % kvh) {
        return -1;
    }
    int G = (int)(qh / kvh);
    if (hd == 256 && G % 2 == 0) {
        flash_h_run<256, 2, 4, 32, false, false, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    if (hd == 512 && G % 2 == 0) {
        flash_h_run<512, 2, 1, FK3, false, false, true>(dr, denv, (int)kvh, (int)t, G);
        return 0;
    }
    return -1;
}

static int flash_tc_launch(const gp_rec *r, const gp_rec *dr, const int64_t *denv, int f32h,
                           int *bad)
{
    int o0 = f32h ? 5 : 7;
    int64_t qh = hlit(r, o0, bad), hd = hlit(r, o0 + 2, bad), t = hlit(r, o0 + 3, bad);
    if (hd != 256 && hd != 512) {
        return -1;
    }
    size_t smem = ((size_t)(FQ2 + FK2) * (hd + 8) + (size_t)hd * (FK2 + 8)) * sizeof(__half);
    unsigned warps = 2 * (unsigned)(hd / 256);
    dim3 grid((unsigned)qh, (unsigned)cdiv(t, FQ2));
    if (f32h) {
        cudaFuncSetAttribute(k_flash_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        k_flash_tc<1><<<grid, 32 * warps, smem, gg_stream>>>(dr, denv);
    } else {
        cudaFuncSetAttribute(k_flash_tc<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        k_flash_tc<0><<<grid, 32 * warps, smem, gg_stream>>>(dr, denv);
    }
    return 0;
}

/* The product of an int4 matrix and a group of rows. The operands give x,
 * w, out, rows, cols, and t. The pointers must be literals, because the
 * kernel takes them as arguments: the compiler of a group passes arrays,
 * not slots. */
/* RECLANK_GPU_MT_ROWS=0 keeps k_mt_gemv_n for the int4 matrices of a group
 * of 1 to 4 tokens (the form before k_mt_int4_rows). */
static int mt_rows_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_MT_ROWS");
        on = !(v && v[0] == '0');
    }
    return on;
}

static void gemm_launch(const gg_prog *g, const gp_rec *r, const gp_rec *dr,
                        const int64_t *denv, int xk, int wk, int ok, int rk, int ck, int tk,
                        int *bad, int quant = 1)
{
    if (r->tag[xk] == GP_T_SLOT || r->tag[wk] == GP_T_SLOT || r->tag[ok] == GP_T_SLOT) {
        *bad = 1;
        return;
    }
    const float *x = (const float *)(intptr_t)r->v[xk];
    const void *w = (const void *)(intptr_t)r->v[wk];
    float *out = (float *)(intptr_t)r->v[ok];
    int rows = (int)hlit(r, rk, bad), cols = (int)hlit(r, ck, bad), t = (int)hlit(r, tk, bad);
    if (t <= gg_gemv_max && t <= 8 && mt_rows_on()) {
        /* A verify group: the bits of a step for each token (k_mt_int4_rows),
         * R rows a warp (mt_rows_r). Up to 8 tokens it is as fast as
         * k_mt_gemv_n with 2 rows a warp (5 to 7 a little faster, 8 4%
         * slower), so a verify group of up to 8 rows (7 drafts) verifies
         * with the numbers of the decode. */
        int R = mt_rows_r(t, rows);
        unsigned grid = (unsigned)cdiv(rows, ROWS_PER_BLOCK * R), blk = 32 * ROWS_PER_BLOCK;
        const uint8_t *wb = (const uint8_t *)w;
        switch (t * 2 + (R == 2)) {
#define GG_RT(n) case 2 * n: k_mt_int4_rows<n><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols); break; \
                 case 2 * n + 1: k_mt_int4_rows<n, 2><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols); break;
        GG_RT(1) GG_RT(2) GG_RT(3) GG_RT(4) GG_RT(5) GG_RT(6) GG_RT(7) GG_RT(8)
#undef GG_RT
        default: *bad = 1;
        }
    } else if (t <= gg_gemv_max) {
        int R = t <= MT_MAX ? mt_rows_r(t, rows) : 1;
        unsigned grid = (unsigned)cdiv(rows, ROWS_PER_BLOCK * R), blk = 32 * ROWS_PER_BLOCK;
        const uint8_t *wb = (const uint8_t *)w;
        switch (t * 2 + (R == 2)) {
#define GG_NT(n) case 2 * n: k_mt_gemv_n<n><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols); break; \
                 case 2 * n + 1: k_mt_gemv_n<n, 2><<<grid, blk, 0, gg_stream>>>(x, wb, out, rows, cols); break;
        GG_NT(1) GG_NT(2) GG_NT(3) GG_NT(4) GG_NT(5) GG_NT(6) GG_NT(7) GG_NT(8)
        GG_NT(9) GG_NT(10) GG_NT(11) GG_NT(12) GG_NT(13) GG_NT(14) GG_NT(15) GG_NT(16)
#undef GG_NT
        default:
            k_mt_gemv<<<cdiv(rows, ROWS_PER_BLOCK), blk, 0, gg_stream>>>(x, wb, out, t, rows, cols);
        }
    } else if (gg_tc && ((gg_tc == 8 && g->tc) || g->i8) && cols % Q8K == 0 &&
               (size_t)t * cols <= g->xh_n) {
        /* int8 x: the scratch of xh holds the int8 values, then the scales.
         * i8 2 (the int16 form, k_quant_x2): the hi values, the lo values,
         * then the scales (rc_gpu_load gives xh the room). */
        size_t n = (size_t)t * cols;
        int8_t *xq = (int8_t *)g->xh;
        if (g->i8 == 2) {
            int8_t *xl = xq + n;
            float *xs = (float *)((char *)g->xh + ((2 * n + 255) & ~(size_t)255));
            if (quant) {
                k_quant_x2<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(
                    x, xq, xl, xs, NULL, n / 32);
            }
            dim3 grid((unsigned)cdiv(t, T2M), (unsigned)cdiv(rows, T2N));
            static int attr2 = 0;
            if (!attr2) {
                cudaFuncSetAttribute(k_gemm_q8<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)Q8SMEM_X2);
                cudaFuncSetAttribute(k_gemm_q8<0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     (int)Q8SMEM_X2);
                attr2 = 1;
            }
            if ((cols / 32) % 8 == 0 && ((uintptr_t)w & 15) == 0) {
                k_gemm_q8<1, 1><<<grid, 256, Q8SMEM_X2, gg_stream>>>(
                    xq, xs, (const uint8_t *)w, out, t, rows, cols, xl);
            } else {
                k_gemm_q8<0, 1><<<grid, 256, Q8SMEM_X2, gg_stream>>>(
                    xq, xs, (const uint8_t *)w, out, t, rows, cols, xl);
            }
            return;
        }
        float *xs = (float *)((char *)g->xh + ((n + 255) & ~(size_t)255));
        /* quant 0: the scratch already holds x as int8, from the matrix
         * before this one in GP_INT4_MULTI4_MT. */
        if (quant) {
            k_quant_q8<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(
                x, xq, xs, n / 32);
        }
        dim3 grid((unsigned)cdiv(t, T2M), (unsigned)cdiv(rows, T2N));
        static int attr = 0;
        if (!attr) {
            cudaFuncSetAttribute(k_gemm_q8<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)Q8SMEM);
            cudaFuncSetAttribute(k_gemm_q8<0>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 (int)Q8SMEM);
            attr = 1;
        }
        if ((cols / 32) % 8 == 0 && ((uintptr_t)w & 15) == 0) {
            /* The rows of w start at a multiple of 16 bytes. */
            k_gemm_q8<1><<<grid, 256, Q8SMEM, gg_stream>>>(xq, xs, (const uint8_t *)w, out, t,
                                                          rows, cols);
        } else {
            k_gemm_q8<0><<<grid, 256, Q8SMEM, gg_stream>>>(xq, xs, (const uint8_t *)w, out, t,
                                                          rows, cols);
        }
    } else if (gg_tc && g->tc && cols % TK == 0 && (size_t)t * cols <= g->xh_n) {
        size_t n = (size_t)t * cols;
        if (quant) {
            k_to_half<<<(unsigned)cdiv((int64_t)n / 4 + 1, 256), 256, 0, gg_stream>>>(x, g->xh, n);
        }
        if (gg_tc == 2) {
            k_gemm_tc<<<dim3((unsigned)cdiv(rows, TN), (unsigned)cdiv(t, TMD)), 256, 0,
                         gg_stream>>>(g->xh, (const uint8_t *)w, out, t, rows, cols);
        } else {
            k_gemm_tc2<<<dim3((unsigned)cdiv(t, T2M), (unsigned)cdiv(rows, T2N)), 256, 0,
                          gg_stream>>>(g->xh, (const uint8_t *)w, out, t, rows, cols);
        }
    } else if (!gg_caps.tc && g->i8 && dp4a_on() && cols % 32 == 0 && ((uintptr_t)w & 1) == 0 &&
               (size_t)t * cols <= g->xh_n) {
        /* no tensor cores (Pascal): int8 x as for k_gemm_q8, on dp4a (the
         * layout of the scratch of k_gemm_q8) */
        size_t n = (size_t)t * cols;
        int8_t *xq = (int8_t *)g->xh, *xl = NULL;
        float *xs;
        if (g->i8 == 2) {
            xl = xq + n;
            xs = (float *)((char *)g->xh + ((2 * n + 255) & ~(size_t)255));
            if (quant) {
                k_quant_x2<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(
                    x, xq, xl, xs, NULL, n / 32);
            }
        } else {
            xs = (float *)((char *)g->xh + ((n + 255) & ~(size_t)255));
            if (quant) {
                k_quant_q8<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(
                    x, xq, xs, n / 32);
            }
        }
        dim3 grid((unsigned)cdiv(t, DP_BM), (unsigned)cdiv(rows, DP_BN));
        if (xl != NULL) {
            k_gemm_dp4a<1><<<grid, 256, 0, gg_stream>>>(xq, xs, (const uint8_t *)w, out, t, rows,
                                                       cols, xl);
        } else {
            k_gemm_dp4a<0><<<grid, 256, 0, gg_stream>>>(xq, xs, (const uint8_t *)w, out, t, rows,
                                                       cols, NULL);
        }
    } else {
        k_gemm<0><<<dim3((unsigned)cdiv(rows, GN), (unsigned)cdiv(t, GM)), 256, 0, gg_stream>>>(
            x, w, out, t, rows, cols);
    }
}

/* Launch the kernels of one record. Return 0, or -1 for an operation that
 * this file does not have or a size that is not a literal. */
/* RECLANK_GPU_KQTC=0 keeps the float32 tiles for the products of large
 * groups of Qwen3.8 (a test). */
static int i8x_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_I8X");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* RECLANK_GPU_MOE_F32=1 (RECLANK_MOE_X16): the experts of a group
 * (GP_KQ_GROUP_MOE) in float32 (k_qmoe_gu, k_qmoe_dn: x not quantized), not
 * on the int8 tensor cores. */
static int moe_f32_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_MOE_F32");
        on = v && v[0] == '1';
    }
    return on;
}

static int kt_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_KQTC");
        on = !(v && v[0] == '0');
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}

/* The shared memory of the instances of the tensor-core kernels (once). */
static void kt_attr(void)
{
    static int done = 0;
    if (done) {
        return;
    }
    done = 1;
    int a = (int)KT_SMEM(128, 2), b = (int)KT_SMEM(64, 3);
    cudaFuncSetAttribute(k_kq_tc<KT_Q8R>, cudaFuncAttributeMaxDynamicSharedMemorySize, a);
#define KT_SET(F) \
    cudaFuncSetAttribute(k_qmoe_gu_tc<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, b); \
    cudaFuncSetAttribute(k_qmoe_dn_tc<F>, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
    KT_SET(KT_Q8R) KT_SET(KT_Q80) KT_SET(KT_Q51) KT_SET(KT_Q4K) KT_SET(KT_NV4) KT_SET(KT_NVX)
    KT_SET(KT_Q6K)
#undef KT_SET
    cudaFuncSetAttribute(k_moe_gemm_q4x<0>, cudaFuncAttributeMaxDynamicSharedMemorySize, b);
    cudaFuncSetAttribute(k_moe_gemm_q4x<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)KT_SMEM_X2(64, 2));
}

/* The lo plane of the int16 form in the int8 scratch of g (i8 2): after
 * the int8 values and the xs and xsum of kqx_n values (rc_gpu_load). */
static int8_t *kt_lo(const gg_prog *g)
{
    return g->kqx + g->kqx_n + g->kqx_n / 32 * 8;
}

/* Quantize n values (a multiple of 32) at x to the int16 form (k_quant_x2)
 * in the int8 scratch of g: the hi plane, xs and xsum as kt_quant, and the
 * lo plane at kt_lo. */
static void kt_quant_x2(const gg_prog *g, const float *x, size_t n)
{
    float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
    k_quant_x2<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(
        x, g->kqx, kt_lo(g), xs, xsum, n / 32);
}

/* Quantize n values (a multiple of 32) at x to the int8 scratch of g. */
static void kt_quant(const gg_prog *g, const float *x, size_t n)
{
    int8_t *xq = g->kqx;
    float *xs = (float *)(g->kqx + g->kqx_n), *xsum = xs + g->kqx_n / 32;
    k_kq_quant_x<<<(unsigned)cdiv((int64_t)(n / 32) * 8, 256), 256, 0, gg_stream>>>(x, xq, xs, xsum,
                                                                                   n / 32);
}

/* The x of record r if its launch quantizes x to int8 with kt_quant for a
 * product of one token (GP_KQ_LINEAR, GP_KQ_MULTI), else NULL. *n is the
 * count of values. The conditions are those of gg_launch. */
/* The fused gate, up and GELU (k_kq_glu_i8). RECLANK_GPU_GLU=0 gives the
 * two records. */
static int glu_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_GLU");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* A record that the launch before did (k_kq_glu_i8 does the GELU record
 * after its GP_KQ_MULTI); gg_launch skips it. */
static const gp_rec *gg_skip;

static const float *kq_i8_input(const gg_prog *g, const gp_rec *r, size_t *n)
{
    int bad = 0;
    if (r->op == GP_KQ_LINEAR) {
        int64_t t = hlit(r, 8, &bad), cols = hlit(r, 7, &bad);
        if (t <= MT_MAX && kq_i8_type((int)hlit(r, 5, &bad)) && i8x_on() && cols % 32 == 0 &&
            (size_t)(t * cols) <= g->kqx_n && !bad) {
            *n = (size_t)(t * cols);
            return (const float *)(intptr_t)hi(r, g->henv, 3);
        }
    } else if (r->op == GP_KQ_MULTI) {
        int64_t t = hlit(r, 2, &bad), cols = hlit(r, 1, &bad), q8r = 0;
        for (int m = 0; m < hlit(r, 3, &bad); ++m) {
            q8r |= kq_i8_type((int)hlit(r, 5 + 4 * m, &bad));
        }
        if (q8r && i8x_on() && cols % 32 == 0 && (size_t)(t * cols) <= g->kqx_n && !bad) {
            *n = (size_t)(t * cols);
            return (const float *)(intptr_t)hi(r, g->henv, 0);
        }
    }
    return NULL;
}

/* The x of a large int4 product that quantizes x with k_quant_q8 (the
 * int8 path of gemm_launch), and its count of values; else NULL. */
static const float *q8_gemm_input(const gg_prog *g, const gp_rec *r, size_t *n)
{
    int64_t cols, t;
    if (r->op == GP_INT4_LINEAR_MT) {
        cols = r->v[5];
        t = r->v[6];
    } else if (r->op == GP_INT4_MULTI4_MT) {
        cols = r->v[1];
        t = r->v[2];
    } else {
        return NULL;
    }
    if (r->tag[0] == GP_T_SLOT || t <= gg_gemv_max || !gg_tc ||
        !((gg_tc == 8 && g->tc) || g->i8) || cols % Q8K != 0 ||
        (size_t)(t * cols) > g->xh_n) {
        return NULL;
    }
    *n = (size_t)(t * cols);
    return (const float *)(intptr_t)r->v[0];
}

/* The int8 x that the launch before wrote (k_add_norm_v, k_gelu_mul_rows_v):
 * the source and the count of values. gg_launch clears it, so only the next
 * record can use it. */
static const float *gg_qx_src;
static size_t gg_qx_n;

static int an_v_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_AN_V");
        on = !(v && v[0] == '0');
    }
    return on;
}

static int qx_env_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_QX_FUSE");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* If the record after r (in the same program) quantizes out (n values), give
 * the int8 buffers to the kernel of r, and let that record skip kt_quant. */
static int8_t *qx_fuse(const gg_prog *g, const gp_rec *r, const void *out, size_t n,
                       float **xs, float **xsum)
{
    size_t m = 0;
    const gp_rec *nx = r + 1, *end = g->hcode + g->n_code;
    /* the GP_KQ_QUANT records between (the CPU form; nothing on the GPU) */
    while (nx < end && nx->op == GP_KQ_QUANT) {
        ++nx;
    }
    if (!qx_env_on() || out == NULL || nx >= end || n % 32 != 0) {
        return NULL;
    }
    if (q8_gemm_input(g, nx, &m) == (const float *)out && m == n) {
        if (g->i8 == 2) {
            return NULL;        /* the int16 form: gemm_launch quantizes x */
        }
        /* The layout of gemm_launch: the int8 values, then (at a multiple of
         * 256 bytes) the scales; the sums of k_kq_* after them are unused. */
        *xs = (float *)((char *)g->xh + ((n + 255) & ~(size_t)255));
        *xsum = *xs + n / 32;
        gg_qx_src = (const float *)out;
        gg_qx_n = n;
        return (int8_t *)g->xh;
    }
    if (kq_i8_input(g, nx, &m) != (const float *)out || m != n) {
        return NULL;
    }
    *xs = (float *)(g->kqx + g->kqx_n);
    *xsum = *xs + g->kqx_n / 32;
    gg_qx_src = (const float *)out;
    gg_qx_n = n;
    return g->kqx;
}

static int al16(int64_t p)
{
    return p % 16 == 0;
}

/* The kernels of the media encoders (defined with their entry points below). */
__global__ void k_enc_gemm(const float *x, const void *w, int wbf, const float *b, float *y,
                           int n, int m, int k, float imin, float imax, float omin, float omax);
__global__ void k_enc_rms(const float *x, const float *w, float *y, int rows, int d, float eps);
__global__ void k_enc_gelu_mul(const float *g, const float *u, float *y, int64_t n);
__global__ void k_enc_lnorm(const float *x, const float *w, const float *b, float *y, int rows,
                            int d, float eps);
__global__ void k_enc_gelu(const float *x, float *y, int64_t n, int use_erf);
__global__ void k_enc_add(float *x, const float *y, int64_t n, float sc);
__global__ void k_enc_silu(const float *x, float *y, int64_t n);
__global__ void k_enc_clamp(const float *x, float *y, int64_t n, float lo, float hi);
__global__ void k_enc_bias_clamp(float *y, const float *b, int rows, int cols, float lo, float hi);
__global__ void k_enc_mul_vec(const float *x, const float *vec, float *y, int rows, int cols);
__global__ void k_enc_glu(const float *x, float *y, int rows, int cols);
__global__ void k_enc_dwconv(const float *x, const float *w, float *y, int t, int c, int kw);
__global__ void k_enc_local_attn(const float *q, const float *k, const float *v, const float *r,
                                 const int *valid, float *o, int t, int heads, int hd, int span,
                                 float cap);
__global__ void k_enc_rope2d(float *x, const int *pos, const float *inv, int n, int heads, int hd);
__global__ void k_enc_attn(const float *q, const float *k, const float *v, float *o,
                           int n, int heads, int hd);

/* The float32 of a literal operand. */
static float hlitf(const gp_rec *r, int k, int *bad)
{
    uint32_t u = (uint32_t)hlit(r, k, bad);
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}
