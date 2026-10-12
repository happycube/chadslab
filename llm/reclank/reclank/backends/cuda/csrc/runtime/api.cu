extern "C" {

const char *rc_gpu_last_error(void)
{
    return gg_error;
}

/* rc_gpu_where: the program (its handle), the record queued last and its
 * operation, the segment [s0, s1) of it (out: 4 ints), and whether the
 * runs wait after each record (RECLANK_GPU_SYNC_CHECK). */
const void *rc_gpu_where(int *out)
{
    out[0] = gg_where_pc;
    out[1] = gg_where_op;
    out[2] = gg_where_s0;
    out[3] = gg_where_s1;
    out[4] = gg_sync_check();
    return gg_where_prog;
}

int rc_gpu_init(int device)
{
    CK(cudaSetDevice(device));
    gg_caps_init(device);
    if (!gg_caps.tc) {
        gg_tc = 0;     /* the float32 products of a group: no tensor cores */
    }
    if (gg_stream == NULL) {
        CK(cudaStreamCreateWithFlags(&gg_stream, cudaStreamNonBlocking));
    }
    return 0;
}

/* The capabilities (rc_caps, runtime/caps.cu) as ints in their order: n of
 * them to out. Return the count of fields. */
int rc_gpu_caps_get(int *out, int n)
{
    const int *c = (const int *)&gg_caps;
    int all = (int)(sizeof(gg_caps) / sizeof(int));
    for (int i = 0; i < n && i < all; ++i) {
        out[i] = c[i];
    }
    return all;
}

void rc_gpu_set_cpu_runner(void *fn)
{
    gg_cpu_run = (gg_cpu_runner)fn;
}

void rc_gpu_set_gemv_max(int t)
{
    gg_gemv_max = t < MT_MAX ? t : MT_MAX;
}

void rc_gpu_set_tc(int on)
{
    gg_tc = on;    /* 0: float32, 1: k_gemm_tc2, 2: k_gemm_tc (for a test), 8: k_gemm_q8 */
}

int rc_gpu_mem_info(size_t *free_b, size_t *total_b)
{
    CK(cudaMemGetInfo(free_b, total_b));
    return 0;
}

/* Clear the last error of the CUDA runtime in this thread. A failed call
 * that the caller handles (a cudaMalloc when the memory runs out) leaves its
 * error there, and the next run would report it (cudaGetLastError at the
 * end of gg_exec). Return the error it cleared. */
int rc_gpu_clear_error(void)
{
    return (int)cudaGetLastError();
}

void *rc_gpu_malloc(size_t n)
{
    void *p = NULL;
    if (cudaMalloc(&p, n) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "cudaMalloc of %zu bytes failed", n);
        return NULL;
    }
    return p;
}

int rc_gpu_free(void *p)
{
    CK(cudaFree(p));
    return 0;
}

/* Pinned host memory, for the copies of the handoff to the CPU. */
void *rc_gpu_host_alloc(size_t n)
{
    void *p = NULL;
    if (cudaMallocHost(&p, n) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "cudaMallocHost of %zu bytes failed", n);
        return NULL;
    }
    memset(p, 0, n);
    return p;
}

int rc_gpu_host_free(void *p)
{
    CK(cudaFreeHost(p));
    return 0;
}

/* Copies on the stream of the programs. A copy to the host waits for the
 * stream. */
int rc_gpu_h2d(void *d, const void *h, size_t n)
{
    CK(cudaMemcpyAsync(d, h, n, cudaMemcpyHostToDevice, gg_stream));
    return 0;
}

int rc_gpu_d2h(void *h, const void *d, size_t n)
{
    CK(cudaMemcpyAsync(h, d, n, cudaMemcpyDeviceToHost, gg_stream));
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

/* A copy on the GPU. The areas can overlap. */
/* n zero bytes at d, on the stream of the programs (a fresh cache:
 * _DevCache.attach). */
int rc_gpu_zero(void *d, size_t n)
{
    CK(cudaMemsetAsync(d, 0, n, gg_stream));
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

int rc_gpu_d2d(void *d, const void *src, size_t n)
{
    if ((const char *)d < (const char *)src + n && (const char *)src < (const char *)d + n) {
        void *tmp = NULL;
        CK(cudaMallocAsync(&tmp, n, gg_stream));
        CK(cudaMemcpyAsync(tmp, src, n, cudaMemcpyDeviceToDevice, gg_stream));
        CK(cudaMemcpyAsync(d, tmp, n, cudaMemcpyDeviceToDevice, gg_stream));
        CK(cudaFreeAsync(tmp, gg_stream));
        return 0;
    }
    CK(cudaMemcpyAsync(d, src, n, cudaMemcpyDeviceToDevice, gg_stream));
    return 0;
}

int rc_gpu_sync(void)
{
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

/* The scratch of the programs (xh, w16, btp, btc, kqx, e3 of gg_prog) is
 * shared: the programs run one at a time on gg_stream, and no record reads
 * what a run before (of any program) left there. A program takes the newest
 * scratch of a kind when it is large enough, else a new one of its size,
 * which the programs after it take; each scratch is freed with the last
 * program that holds it (a group of a prompt of each size held its own: tens
 * of MB each). Each program keeps its own sizes (xh_n, kqx_n: the places in
 * it), so a larger scratch is the same to it. */
enum { GG_SH_XH, GG_SH_W16, GG_SH_BTP, GG_SH_BTC, GG_SH_KQX, GG_SH_E3, GG_SH_KINDS };
typedef struct gg_shared {
    void *p;
    size_t n;
    int refs, kind;
    struct gg_shared *next;
} gg_shared;
static gg_shared *gg_sh;        /* the newest first */

/* A scratch of kind of at least n bytes (zeroed when new and zero is 1), or
 * NULL (no memory). */
static void *gg_scratch_get(int kind, size_t n, int zero)
{
    for (gg_shared *e = gg_sh; e != NULL; e = e->next) {
        if (e->kind == kind) {
            if (e->n >= n) {
                ++e->refs;
                return e->p;
            }
            break;              /* (the newest of its kind is too small) */
        }
    }
    void *p = NULL;
    if (cudaMalloc(&p, n) != cudaSuccess || (zero && cudaMemset(p, 0, n) != cudaSuccess)) {
        cudaFree(p);
        return NULL;
    }
    gg_shared *e = (gg_shared *)calloc(1, sizeof(gg_shared));
    e->p = p;
    e->n = n;
    e->refs = 1;
    e->kind = kind;
    e->next = gg_sh;
    gg_sh = e;
    return p;
}

/* A program lets go of its scratch p (NULL: none). */
static void gg_scratch_put(void *p)
{
    if (p == NULL) {
        return;
    }
    for (gg_shared **at = &gg_sh; *at != NULL; at = &(*at)->next) {
        gg_shared *e = *at;
        if (e->p == p) {
            if (--e->refs == 0) {
                cudaFree(e->p);
                *at = e->next;
                free(e);
            }
            return;
        }
    }
}

/* Load a program whose addresses are device addresses. Return a handle, or
 * NULL. use_graph 0 launches the kernels one at a time in each run. */
void *rc_gpu_load(const int64_t *prog, int use_graph)
{
    if (prog[0] != GP_MAGIC) {
        snprintf(gg_error, sizeof(gg_error), "not a program");
        return NULL;
    }
    gg_prog *g = (gg_prog *)calloc(1, sizeof(gg_prog));
    g->n_env = (int)prog[1];
    g->n_code = (int)prog[2];
    /* Bit 0 of the flags: the CUDA graph. Bit 1: no tensor cores for this
     * program (the 26B keeps float32 products; see gpu.py). Bit 2: the int8
     * products of the int4 matrices (k_gemm_q8) all the same; with bit 4,
     * the int16 form of x (k_quant_x2: two int8 planes) for them and for
     * the KQ_Q4X experts of a prompt (k_moe_gemm_q4x). */
    g->use_graph = use_graph & 1;
    g->tcreq = (use_graph & 2) ? 0 : 1;
    g->tc = g->tcreq && gg_caps.tc;
    /* (without the tensor cores, i8 takes k_gemm_dp4a; the tensor-core
     * kernels of i8 test gg_tc and kt_on) */
    g->i8 = (use_graph & 4) ? ((use_graph & 16) ? 2 : 1) : 0;
    g->atc = (use_graph & 8) && gg_caps.tc ? 1 : 0;
    size_t env_b = (size_t)(g->n_env > 0 ? g->n_env : 1) * sizeof(int64_t);
    size_t code_b = (size_t)g->n_code * sizeof(gp_rec);
    g->hcode = (gp_rec *)malloc(code_b);
    memcpy(g->hcode, prog + 4 + g->n_env, code_b);
    if (cudaMallocHost(&g->henv, env_b) != cudaSuccess ||
        cudaMalloc(&g->denv, env_b) != cudaSuccess ||
        cudaMalloc(&g->dcode, code_b) != cudaSuccess ||
        cudaMemcpy(g->dcode, g->hcode, code_b, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMalloc(&g->part, GG_PART_FLOATS * sizeof(float)) != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_load: no memory");
        return NULL;
    }
    memcpy(g->henv, prog + 4, (size_t)g->n_env * sizeof(int64_t));
    /* The segments, and one event for each GP_TO_HOST. */
    g->seg = (gg_seg *)calloc((size_t)g->n_code + 1, sizeof(gg_seg));
    /* A long run of records becomes several graphs: the first has
     * GG_SEG_FIRST records, the next ones GG_SEG_MAX. The GPU starts a graph
     * only when cudaGraphLaunch returns, about 1.5 us for each kernel. With a
     * small first graph, the GPU starts early, and the launches of the next
     * graphs run while it works. */
    int in_seg = 0, run = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        if (is_boundary(r->op)) {
            in_seg = 0;
            run = 0;
            if (r->op == GP_TO_HOST && r->v[9] + 1 > g->n_ev) {
                g->n_ev = (int)r->v[9] + 1;
            }
            continue;
        }
        if (in_seg && pc - g->seg[g->n_seg - 1].start >= (run == 1 ? GG_SEG_FIRST : GG_SEG_MAX)) {
            in_seg = 0;
        }
        if (!in_seg) {
            g->seg[g->n_seg].start = pc;
            ++g->n_seg;
            ++run;
            in_seg = 1;
        }
        g->seg[g->n_seg - 1].end = pc + 1;
    }
    /* The float16 scratch of the products of large groups. */
    size_t need = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        size_t n = 0;
        if (r->op == GP_INT4_LINEAR_MT && r->v[6] > MT_MAX) {
            n = (size_t)r->v[6] * (size_t)r->v[5];
        } else if (r->op == GP_INT4_MULTI4_MT && r->v[2] > MT_MAX) {
            n = (size_t)r->v[2] * (size_t)r->v[1];
        } else if (r->op == GP_BF16_LINEAR && r->v[5] > MT_MAX) {
            n = (size_t)r->v[5] * (size_t)r->v[4];
        } else if (r->op == GP_ENC_LINEAR && r->v[2] && r->v[5] > MT_MAX) {
            n = (size_t)r->v[5] * (size_t)r->v[7];      /* the rows x cols of x */
        } else if (r->op == GP_KQ_GROUP_MOE && (r->v[16] == KQ_BF16 || r->v[16] == KQ_BF12)) {
            /* the two bfloat16 planes of h (or act2) for the shared expert */
            n = 2 * (size_t)r->v[3] * (size_t)(r->v[6] > r->v[7] ? r->v[6] : r->v[7]);
        } else if (r->op == GP_KQ_LINEAR && r->tag[8] != GP_T_SLOT && r->v[8] > MT_MAX &&
                   (r->v[5] == KQ_BF16 || r->v[5] == KQ_BF12)) {
            n = 2 * (size_t)r->v[8] * (size_t)r->v[7];  /* the two bfloat16 planes of x (k_gemm_bf16_tc) */
        } else if (r->op == GP_MOE_GPU) {
            size_t a = (size_t)r->v[3] * (size_t)r->v[8];
            size_t b = (size_t)r->v[3] * (size_t)r->v[4] * (size_t)r->v[10];
            n = a > b ? a : b;
        }
        need = n > need ? n : need;
    }
    /* the int16 form: 2 bytes for each value, then the scales */
    size_t xh_b = need * sizeof(__half) + (g->i8 == 2 ? need / 8 + 512 : 0);
    if (need > 0 && (g->xh = (__half *)gg_scratch_get(GG_SH_XH, xh_b, 0)) == NULL) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_load: no memory for the float16 scratch");
        return NULL;
    }
    g->xh_n = need;
    /* The bfloat16 rows of the KQ_BF12 matrices of large groups (gg_gemm_bf12),
     * in chunks of at most 16M values (32 MB). */
    size_t wneed = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        size_t n = 0;
        if (r->op == GP_KQ_LINEAR && r->tag[8] != GP_T_SLOT && r->v[8] > MT_MAX && r->v[5] == KQ_BF12) {
            n = (size_t)r->v[6] * (size_t)r->v[7];
        } else if (r->op == GP_KQ_GROUP_MOE && r->v[16] == KQ_BF12) {
            n = (size_t)r->v[6] * (size_t)r->v[7];
        }
        wneed = n > wneed ? n : wneed;
    }
    wneed = bf12_fused_on() ? 0 : wneed < ((size_t)16 << 20) ? wneed : ((size_t)16 << 20);
    if (wneed > 0 && (g->w16 = (uint16_t *)gg_scratch_get(GG_SH_W16, wneed * 2, 0)) == NULL) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_load: no memory for the bfloat16 rows of BF12");
        return NULL;
    }
    g->w16_n = wneed;
    /* The parts of k_kq_bf12_tc of class 1 (the tiles of its matrices). */
    size_t bneed = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        size_t n = 0;
        if (r->op == GP_KQ_LINEAR && bt_shape((int)r->v[5], (int)r->v[6], (int)r->v[7]) &&
            bt_cls((int)r->v[6], (int)r->v[7]) == 1) {
            n = (size_t)(r->v[6] + 15) / 16;
        } else if (r->op == GP_KQ_MULTI) {
            for (int m = 0; m < r->v[3]; ++m) {
                if (bt_shape((int)r->v[5 + 4 * m], (int)r->v[6 + 4 * m], (int)r->v[1]) &&
                    bt_cls((int)r->v[6 + 4 * m], (int)r->v[1]) == 1) {
                    n += (size_t)(r->v[6 + 4 * m] + 15) / 16;
                }
            }
        }
        bneed = n > bneed ? n : bneed;
    }
    /* (btc: counters that the kernels leave at 0) */
    if (bneed > 0 && ((g->btp = (float *)gg_scratch_get(GG_SH_BTP, bneed * BT_KP * 128 * sizeof(float), 0)) == NULL ||
                      (g->btc = (int *)gg_scratch_get(GG_SH_BTC, bneed * sizeof(int), 1)) == NULL)) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_load: no memory for the parts of k_kq_bf12_tc");
        return NULL;
    }
    /* The int8 scratch of the products of large groups on the tensor cores. */
    size_t kneed = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        size_t n = 0;
        if (r->op == GP_KQ_LINEAR) {
            n = (size_t)r->v[8] * (size_t)r->v[7];
        } else if (r->op == GP_KQ_MULTI) {
            n = (size_t)r->v[2] * (size_t)r->v[1];
        } else if (r->op == GP_KQ_GROUP_MOE) {
            size_t t = (size_t)r->v[3], P = t * (size_t)r->v[4] + t;
            n = t * (size_t)r->v[6];
            n = P * (size_t)r->v[7] > n ? P * (size_t)r->v[7] : n;
        } else if (r->op == GP_KQ_HOT_MOE) {
            /* the int8 x of the t tokens, then the int8 act of the pairs */
            size_t t = (size_t)r->v[16], P = t * (size_t)r->v[11] + t;
            n = t * (size_t)r->v[13];
            n = P * (size_t)r->v[12] > n ? P * (size_t)r->v[12] : n;
        } else if (r->op == GP_MOE_GPU && r->v[5] == 1) {
            size_t t = (size_t)r->v[3], P = t * (size_t)r->v[4];
            n = t * (size_t)r->v[8];
            n = P * (size_t)r->v[10] > n ? P * (size_t)r->v[10] : n;
        }
        kneed = n > kneed ? n : kneed;
    }
    if (kneed > 0) {
        kneed = (kneed + 255) & ~(size_t)255;
        /* the int16 form: its lo plane after xs and xsum (kt_lo); without
         * the tensor cores the tiles of dp4a take it for RECLANK_MOE_X16 */
        /* the hot experts take the int16 form too (kq_row_x2) */
        int hot = 0;
        for (int pc = 0; pc < g->n_code; ++pc) {
            hot |= g->hcode[pc].op == GP_KQ_HOT_MOE;
        }
        g->kqx_lo = g->i8 == 2 || !gg_caps.tc || hot;
        if ((g->kqx = (int8_t *)gg_scratch_get(GG_SH_KQX, kneed + kneed / 32 * 8 + (g->kqx_lo ? kneed : 0), 0)) ==
            NULL) {
            snprintf(gg_error, sizeof(gg_error), "rc_gpu_load: no memory for the int8 scratch");
            return NULL;
        }
    }
    g->kqx_n = kneed;
    /* The scratch of the KQ_E3 products (kernels/quant/exl3.cu). */
    size_t eneed = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        if (r->op == GP_KQ_LINEAR && kq_e3_k((int)r->v[5]) && r->tag[8] != GP_T_SLOT) {
            size_t n = e3_ws_bytes(r->v[8], (int)r->v[7], (int)r->v[6]);
            eneed = n > eneed ? n : eneed;
        }
    }
    if (eneed > 0 && (g->e3 = (uint8_t *)gg_scratch_get(GG_SH_E3, eneed, 0)) == NULL) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_load: no memory for the scratch of the EXL3 products");
        return NULL;
    }
    g->e3_n = eneed;
    g->ev = (cudaEvent_t *)calloc((size_t)g->n_ev + 1, sizeof(cudaEvent_t));
    for (int k = 0; k < g->n_ev; ++k) {
        if (cudaEventCreateWithFlags(&g->ev[k], cudaEventDisableTiming) != cudaSuccess) {
            snprintf(gg_error, sizeof(gg_error), "rc_gpu_load: no event");
            return NULL;
        }
    }
    return g;
}

/* Run a program. env holds the values of the slots after the bind; the run
 * computes the scalar slots itself. The run does not wait for the GPU. */
/* The tables of the caches with keys only (ko_tab, kernels/operands.cu):
 * hi (nhi rows) and lo (256 rows) of rd (cos, sin) pairs, device addresses;
 * hi null turns them off. */
int rc_gpu_ko_table(const void *hi, const void *lo, int rd, int half, int nhi)
{
    ko_tab t = {(const float2 *)hi, (const float2 *)lo, rd, half, nhi};
    CK(cudaStreamSynchronize(gg_stream));
    CK(cudaMemcpyToSymbol(g_ko, &t, sizeof(t)));
    return 0;
}

/* The test of the reuse of experts in an MTP group (k_router_top_mt). */
int rc_gpu_set_reuse(int on)
{
    /* In the order of the stream, so the steps before keep their setting. */
    static int v;
    v = on;
    CK(cudaMemcpyToSymbolAsync(g_reuse, &v, sizeof(int), 0, cudaMemcpyHostToDevice, gg_stream));
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

/* Record the graphs of a program now and upload them to the device (the
 * first run does it otherwise). A graph does not depend on the values of
 * the environment (one graph serves every position), so the program can do
 * it when it is made: its memory is then taken before a caller gives the
 * rest of the memory to other buffers. */
int rc_gpu_prepare(void *handle)
{
    gg_prog *g = (gg_prog *)handle;
    if (!g->use_graph || gg_sync_check()) {
        return 0;
    }
    for (int s = 0; s < g->n_seg; ++s) {
        if (g->seg[s].exec == NULL && gg_capture(g, g->seg + s) != 0) {
            return -1;
        }
        CK(cudaGraphUpload(g->seg[s].exec, gg_stream));
    }
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

int rc_gpu_run(void *handle, const int64_t *env)
{
    gg_prog *g = (gg_prog *)handle;
    memcpy(g->henv, env, (size_t)g->n_env * sizeof(int64_t));
    gg_scalars(g);
    CK(cudaMemcpyAsync(g->denv, g->henv, (size_t)g->n_env * sizeof(int64_t),
                       cudaMemcpyHostToDevice, gg_stream));
    return gg_exec(g);
}

/* Run a program without the graph and measure each record. The array ms
 * gets the time of each record on the GPU, in milliseconds. A boundary
 * record gets the time of its copies and of its CPU program. This is for a
 * profile: the events add time between the kernels. */
int rc_gpu_profile(void *handle, const int64_t *env, float *ms)
{
    gg_prog *g = (gg_prog *)handle;
    memcpy(g->henv, env, (size_t)g->n_env * sizeof(int64_t));
    gg_scalars(g);
    CK(cudaMemcpyAsync(g->denv, g->henv, (size_t)g->n_env * sizeof(int64_t),
                       cudaMemcpyHostToDevice, gg_stream));
    cudaEvent_t a, b;
    CK(cudaEventCreate(&a));
    CK(cudaEventCreate(&b));
    int rc = 0;
    for (int pc = 0; pc < g->n_code && rc == 0; ++pc) {
        const gp_rec *r = g->hcode + pc;
        CK(cudaEventRecord(a, gg_stream));
        if (is_boundary(r->op)) {
            rc = gg_boundary(g, r);
        } else if (r->op == GP_CPU_TASK) {
            rc = gg_cpu_task(r, g->henv);
        } else {
            rc = gg_launch(g, r, g->dcode + pc, g->denv);
        }
        CK(cudaEventRecord(b, gg_stream));
        cudaError_t err = cudaEventSynchronize(b);
        if (err != cudaSuccess) {
            snprintf(gg_error, sizeof(gg_error), "record %d (operation %d): %s", pc, r->op,
                     cudaGetErrorString(err));
            return -1;
        }
        CK(cudaEventElapsedTime(&ms[pc], a, b));
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return rc;
}

/* The output head: out = the Q6_K matrix w times the nx rows of x, with the
 * soft cap cap (0 for none). nx is at most HEAD_MAX. out has nx rows of rows
 * values. The kernel reads the head one time for all the rows. All pointers
 * are device addresses. The call does not wait for the GPU. */
int rc_gpu_q6k_head(const void *w, const float *x, float *out, int rows, int cols, float cap,
                int nx)
{
    if (nx < 1 || nx > HEAD_MAX) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_q6k_head: 1 to %d rows of x", HEAD_MAX);
        return -1;
    }
    dim3 grid((unsigned)cdiv(rows, ROWS_PER_BLOCK));
    if (nx == 1) {
        k_q6k_head<1><<<grid, 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
            (const uint8_t *)w, x, out, rows, cols, cap, nx);
    } else {
        k_q6k_head<0><<<grid, 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
            (const uint8_t *)w, x, out, rows, cols, cap, nx);
    }
    CK(cudaGetLastError());
    return 0;
}

/* The output head of a Q4_0 matrix w (k_q4_head), as rc_gpu_q6k_head. cols is a
 * multiple of 32 and x is 8-byte aligned. */
int rc_gpu_q4_head(const void *w, const float *x, float *out, int rows, int cols, float cap,
               int nx)
{
    if (nx < 1 || nx > HEAD_MAX) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_q4_head: 1 to %d rows of x", HEAD_MAX);
        return -1;
    }
    if ((uintptr_t)x % 8 != 0 || cols % 32 != 0) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_q4_head: x not 8-byte aligned or cols %d", cols);
        return -1;
    }
    const uint8_t *wb = (const uint8_t *)w;
#define Q4H(NXV, RV, UV) k_q4_head<NXV, RV, UV><<<dim3((unsigned)cdiv(rows, ROWS_PER_BLOCK * RV)), \
        32 * ROWS_PER_BLOCK, 0, gg_stream>>>(wb, x, out, rows, cols, cap, nx)
    /* The rows of a decode step and of an MTP verify group (1 to 4) have
     * their own kernels: the compiler then unrolls the loop over x. A warp
     * of a group does R rows of w, so that a load of x serves R rows. For the
     * 12B head (262144 x 3840) on an RTX 5060 Ti: 1.43, 1.58, 1.70, and
     * 1.89 ms for 1 to 4 rows of x, against 1.45, 1.93, 2.77, and 3.27 ms
     * with one row of w for each warp. */
    switch (nx) {
    case 1: Q4H(1, 1, 2); break;
    case 2: Q4H(2, 2, 4); break;
    case 3: Q4H(3, 4, 1); break;
    case 4: Q4H(4, 4, 1); break;
    default: Q4H(0, 1, 1);
    }
#undef Q4H
    CK(cudaGetLastError());
    return 0;
}

/* k_embed_q4: the row tok[0] (a device int) of the Q4_0 table, times scale,
 * into out. Device addresses; the call does not wait for the GPU. */
int rc_gpu_embed_q4(const void *table, const int *tok, float *out, int cols, float scale)
{
    k_embed_q4<<<1, 256, 0, gg_stream>>>((const uint8_t *)table, tok, out, cols, scale);
    CK(cudaGetLastError());
    return 0;
}

/* k_embed_q4_rows: rows rows of the Q4_0 table (tok: rows device ints, -1
 * for zeros) into out. Device addresses; the call does not wait. */
int rc_gpu_embed_q4_rows(const void *table, const int *tok, float *out, int rows, int cols,
                     float scale)
{
    k_embed_q4_rows<<<rows, 256, 0, gg_stream>>>((const uint8_t *)table, tok, out, cols, scale);
    CK(cudaGetLastError());
    return 0;
}

/* The best token of each of the rows of x (k_argmax_rows) into out (rows
 * ints). Device addresses; the call does not wait for the GPU. */
int rc_gpu_argmax_rows(const float *x, int rows, int vocab, int *out)
{
    k_argmax_rows<<<rows, 1024, 0, gg_stream>>>(x, vocab, out);
    CK(cudaGetLastError());
    return 0;
}

/* The K largest values of each of the rows of x (vocab values each), the row
 * max, and the sum of exp((x - max) inv_t) (k_topk_rows). ids and vals have
 * rows K entries, stat 2 rows. All are device addresses. The call does not
 * wait for the GPU. */
int rc_gpu_topk(const float *x, int rows, int vocab, int K, float inv_t, int *ids, float *vals,
            float *stat)
{
    if (rows < 1 || K < 1 || K > vocab) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_topk: rows %d, K %d, vocab %d", rows, K, vocab);
        return -1;
    }
    k_topk_rows<<<rows, TOPK_T, 0, gg_stream>>>(x, vocab, K, inv_t, ids, vals, stat);
    CK(cudaGetLastError());
    return 0;
}

/* Apply the first n tokens of the log of a verify group to conv and S (the
 * device arrays of one linear layer), as gdn_commit of the CPU. */
int rc_gpu_gdn_commit(float *conv, float *S, const float *log, int n, int kernel, int k_heads,
                  int v_heads, int k_dim, int v_dim)
{
    if (k_dim != 128 || v_dim != 128 || kernel > 9) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_gdn_commit: k_dim = v_dim = 128");
        return -1;
    }
    int cd = 2 * k_heads * k_dim + v_heads * v_dim;
    size_t lrow = (size_t)cd + (size_t)v_heads * (k_dim + v_dim + 1);
    k_gdn_commit_conv<<<(unsigned)cdiv(cd, 256), 256, 0, gg_stream>>>(conv, log, n, kernel, cd,
                                                                    lrow);
    k_gdn_commit_heads<128><<<(unsigned)v_heads, 128, 0, gg_stream>>>(S, log, n, cd, lrow);
    CK(cudaGetLastError());
    return 0;
}

int rc_gpu_unload(void *handle)
{
    gg_prog *g = (gg_prog *)handle;
    for (int k = 0; k < g->n_seg; ++k) {
        if (g->seg[k].exec != NULL) {
            cudaGraphExecDestroy(g->seg[k].exec);
        }
    }
    for (int k = 0; k < g->n_ev; ++k) {
        cudaEventDestroy(g->ev[k]);
    }
    free(g->seg);
    free(g->ev);
    cudaFree(g->denv);
    cudaFree(g->dcode);
    cudaFree(g->part);
    gg_scratch_put(g->xh);
    gg_scratch_put(g->w16);
    gg_scratch_put(g->btp);
    gg_scratch_put(g->btc);
    gg_scratch_put(g->kqx);
    gg_scratch_put(g->e3);
    cudaFreeHost(g->henv);
    free(g->hcode);
    free(g);
    return 0;
}

}  /* extern "C" */
