/* ---------- an expert device (reclank/backends/cuda/devices.py) ----------
 * A second GPU that holds experts the GPU of the steps does not. The process
 * loads a second copy of this library for it (its own state: its device, its
 * stream), uploads its experts once (rc_gpu_ex_store), and the CPU part of
 * each step calls rc_gpu_ex_post and rc_gpu_ex_wait through function pointers
 * (GP_REMOTE_POST, GP_REMOTE_JOIN: backends/cpu/csrc/common/ops_qwen.c): the
 * cold experts it holds run here while the CPU runs the others. The int4
 * experts with the kernels of the hot experts (hot_rows_xr: the same rows),
 * a pair's weights by pointer. The weightkeeper's KQ_Q4X experts of the
 * Gemma 4 26B (rc_gpu_ex_init's q4x) as they are, with the kernel of the
 * primary's hot experts in that form (q4x_group_block: float32 x, no dp4a).
 *
 * Qwen (rc_gpu_ex_kq_init): the K-quant experts of each layer by slot (gate, up, down), a job a
 * GP_KQ_HOT_MOE record of its tokens with no shared expert, through gg_launch with a program of
 * zeros (no int8 scratch: the float x kernels, k_kqh_k / k_kqh_gu / k_kqh_dn, read the record only).
 *
 * Nemotron (rc_gpu_ex_nm_init; GP_NM_REMOTE_POST): the IQ2_XXS experts of
 * each layer by slot, as NemotronGPU's hot stacks; a job is a GP_NM_HOT_MOE
 * record of one token (its pairs the experts posted) through gg_launch, the
 * kernels of the primary's hot experts. Super's NVFP4 experts (nv4: KQ_NV4
 * rows, relu^2) the same way by GP_NM_HOT_NV4 (float x, any arch: the 1070
 * beside the 3060). */
#define EX_MAXK 16
#define EX_W 256                 /* the staging: [pairs][2] pointers, then w at 256, then h at 320 */
#define EX_H 320
#define EXG_TMAX 16              /* a group (an MTP verify group, MT_CPU rows): at most 16 tokens */
#define EXG_PMAX 128             /* and 128 pairs; its staging: the pointers, each pair's token, w, h */
#define EXG_TOK (EXG_PMAX * 16)
#define EXG_W (EXG_TOK + EXG_PMAX * 4)
#define EXG_H (EXG_W + EXG_PMAX * 4)
#define EXK_GRAPHS 32            /* Qwen: the most graphs of one-token jobs */

typedef struct {
    int dev, layers, hidden, inner, dn_rows;
    int q4x;                     /* Gemma: the experts in KQ_Q4X, not Q4_0 */
    uint8_t **gu, **dn;          /* the experts of each layer on this GPU (or null) */
    int32_t **slots;             /* each layer: expert -> its place in gu and dn, or -1 (host) */
    size_t gu_per, dn_per;
    uint8_t *stage_h, *stage_d;  /* pinned and device: pointers, w, h */
    float *act, *act2, *de, *out_d, *out_h;
    cudaEvent_t done;
    int pending;
    /* Nemotron: the latent L, the most pairs and inner of a job, each layer's inner and its
     * slots on this GPU (device), the record of a job and the int8 planes of its x and act */
    int nm, nv4, L, kmax, imax, nout;
    int *inner_l, *experts_l;
    /* Super's extras (rc_gpu_ex_nm_hosts): cold experts this GPU does not hold, which the CPU's task posts too;
     * a job copies their rows from each layer's stacks in host memory (hup, hdn: the keeper's, registered by
     * the primary: DMA) into xst (xmax experts), and runs every pair from its job's slots and tables (jt_h,
     * jt_d: the slots, the up and the down addresses); each layer's slots on the host (hslots) and the bytes
     * of an expert's rows (upp_l, dnp_l) */
    const uint8_t **hup, **hdn;
    int32_t **hslots;
    size_t *upp_l, *dnp_l;
    uint8_t *xst, *jt_h, *jt_d;
    int xmax, jt_e;
    /* Qwen: the up blocks of each layer and the types of its gate (and up) and down */
    int kq;
    uint8_t **up;
    int *gtype_l, *dtype_l;
    int32_t **slots_d;
    gp_rec *rec_d;
    int64_t *env_d;
    int8_t *qx, *qa, *qxl, *qal;
    float *sx, *sa;
    /* Qwen: the graphs of a one-token job (its copies and kernels, which read the record from
     * rec_d), one for each gate type, down type and count of pairs (kq_key) */
    cudaGraphExec_t kq_exec[EXK_GRAPHS];
    int kq_key[EXK_GRAPHS], kq_ng;
} ex_dev;
#define EXN_TMAX 8               /* a Nemotron job: at most 8 tokens (a verify group), 128 pairs */
#define EXN_TK 128
#define EXN_PMAX 256             /* (Super's, top 22: 8 tokens, 176 pairs; ER_MAXP) */
#define EXN_IDS 256              /* its staging: the record, idx, w, x */
#define EXN_VAL (EXN_IDS + 4 * EXN_PMAX)
#define EXN_X (EXN_VAL + 4 * EXN_PMAX)

static ex_dev gg_ex;

/* a prompt group's share of the experts on this GPU (see rc_gpu_ex_nm_gpost_in) */
typedef struct {
    int tmax, tkmax, npmax, nrmax, emax;
    float *x, *val, *sx, *sa, *de, *out_d, *out_h, *act;
    int32_t *idx, *work;
    int8_t *qx, *qxl, *qa, *qal;
    int64_t **tu_h, **td_h, **tu_d, **td_d;   /* each layer's expert addresses (E each) */
    uint8_t *tab_ok;
    const uint8_t **hup, **hdn;           /* each layer's stacks in host memory (rc_gpu_ex_nm_ghosts) */
    gp_rec *rec_h, *rec_d;
    int nout, pending;
    /* the stage (RECLANK_EX_STAGE, rc_gpu_ex_nm_gstage): two buffers of sc experts' rows, their copies on cs by DMA, a
     * chunk's ready and a buffer's free events; the job's local ids of its experts and their tables (host, device) */
    int sc;
    size_t sper;
    uint8_t *sbuf[2];
    cudaStream_t cs;
    cudaEvent_t ready[2], sfree[2];
    int32_t *lid, *lcnt, *lcold, *lidx_h, *lslot_d;
    int sfail;
    int64_t *ltu_h, *ltd_h, *ltu_d, *ltd_d;
    /* each layer's cold share (rc_gpu_ex_nm_gmask: the primary's split, the most routed first); the experts of the
     * next layer copied ahead (pf_layer's pf_n, pf_ex: the first of its next job's chunks) */
    int32_t **gm;
    int *gm_n;
    int pf_layer, pf_n, *pf_ex;
} ex_grp;
static ex_grp gg_exg;

/* Super's steps: the cold experts of the next MoE layer that the primary predicts (its router on the residual entering
 * the layer before), which the CPU's task of that layer hands here (rc_gpu_ex_nm_prefetch): copied ahead by DMA on cs
 * into one of two buffers of nmax experts, from the host's stacks (rc_gpu_ex_nm_ghosts); stg (the host's, devices.py:
 * layers x em) marks them, the CPU's task of their layer posts their pairs here, and the job reads their rows in the
 * buffer (ready: their copies; free: the job that read them). copied, used: the experts copied, and read by a job */
#define EXD_MAX 16
typedef struct {
    int nmax, em, next;
    size_t per;
    uint8_t *stg;
    cudaStream_t cs;
    struct {
        uint8_t *buf;
        int layer, n;
        int32_t ex[EXD_MAX];
        cudaEvent_t ready, free, done[EXD_MAX];
    } b[2];
    int64_t copied, used, late;
} ex_dst;
static ex_dst gg_exd;

/* a job's small inputs (its record, pairs and x; its tables) read from pinned host memory by a kernel, not copied
 * by DMA: with the decode stage the copy engine carries the copies ahead, and a DMA of the job's queued behind them
 * (the GTX 1070: its uploads' p90 from 28 us to ~2 ms after their issue) */
__global__ void k_ex_pull16(uint4 *dst, const uint4 *src, int n16)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += gridDim.x * blockDim.x) {
        dst[i] = src[i];
    }
}

static int ex_h2d(void *dst, const void *src, size_t n)
{
    if (gg_exd.b[0].buf != NULL && n % 16 == 0 && n <= ((size_t)1 << 20)) {
        void *sd = NULL;
        if (cudaHostGetDevicePointer(&sd, (void *)src, 0) == cudaSuccess && sd != NULL) {
            const int n16 = (int)(n / 16);
            k_ex_pull16<<<(unsigned)(n16 < 256 * 8 ? cdiv(n16, 256) : 8), 256, 0, gg_stream>>>((uint4 *)dst,
                                                                                            (const uint4 *)sd, n16);
            return cudaGetLastError() == cudaSuccess ? 0 : -1;
        }
        cudaGetLastError();
    }
    return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, gg_stream) == cudaSuccess ? 0 : -1;
}

/* the decode stage's buffer of layer's experts (-1: none), and expert e's place in buffer b (-1: none) */
static int ex_dst_find(int layer)
{
    for (int b = 0; b < 2; ++b) {
        if (gg_exd.b[b].buf != NULL && gg_exd.b[b].layer == layer && gg_exd.b[b].n > 0) {
            return b;
        }
    }
    return -1;
}

static int ex_dst_slot(int b, int e)
{
    for (int q = 0; q < gg_exd.b[b].n; ++q) {
        if (gg_exd.b[b].ex[q] == e) {
            return q;
        }
    }
    return -1;
}

/* The CUDA driver keeps one current context for each thread, which the two
 * copies of the library share: each entry point here leaves the thread's
 * context as it found it (the primary's next call is on its own device). */
#include <dlfcn.h>
typedef int (*ex_ctx_get_t)(void **);
typedef int (*ex_ctx_set_t)(void *);
static ex_ctx_get_t ex_ctx_get;
static ex_ctx_set_t ex_ctx_set;
static void *ex_own;             /* this copy's context (rc_gpu_ex_open) */

/* RECLANK_EX_DEBUG=1: each Nemotron group job (layer, rows, pairs, thread, the contexts) to
 * stderr, its wait (the time; past 1 s the state of the event and the stream each second); 2: also a
 * stream synchronize after the launch (the time, the error); every error */
#include <sys/syscall.h>
#include <unistd.h>
#include <time.h>
static int ex_dbg = -1;
static int ex_debug(void)
{
    if (ex_dbg < 0) {
        const char *v = getenv("RECLANK_EX_DEBUG");
        ex_dbg = v != NULL ? atoi(v) : 0;
    }
    return ex_dbg;
}
static double ex_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void *ex_enter(void)
{
    if (ex_ctx_get == NULL) {
        void *h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (h == NULL) {
            h = dlopen("libcuda.so.1", RTLD_NOW);
        }
        ex_ctx_get = (ex_ctx_get_t)dlsym(h, "cuCtxGetCurrent");
        ex_ctx_set = (ex_ctx_set_t)dlsym(h, "cuCtxSetCurrent");
    }
    void *ctx = NULL;
    ex_ctx_get(&ctx);
    if (ex_own != NULL) {
        ex_ctx_set(ex_own);
    }
    return ctx;
}

static int ex_leave(void *ctx, int rc)
{
    ex_ctx_set(ctx);
    return rc;
}

/* rc_gpu_init of this copy on device, leaving the thread's context as it was;
 * free gets the free bytes of the device. */
extern "C" int rc_gpu_ex_open(int device, size_t *free)
{
    void *ctx = ex_enter();
    int rc = rc_gpu_init(device);
    size_t total = 0;
    if (rc == 0 && cudaMemGetInfo(free, &total) != cudaSuccess) {
        rc = -1;
    }
    gg_ex.dev = device;
    ex_ctx_get(&ex_own);
    return ex_leave(ctx, rc);
}

__global__ void k_ex_gu(const int64_t *ptrs, const float *x, int rows, int cols, float *act)
{
    int j = blockIdx.y;
    hot_rows_xr((const uint8_t *)(intptr_t)ptrs[2 * j], x, rows, cols, act + (size_t)j * rows);
}

/* k_ex_gu of a group: pair j on the row of its token */
__global__ void k_ex_gu_t(const int64_t *ptrs, const int32_t *tok, const float *x, int rows, int cols, float *act)
{
    int j = blockIdx.y;
    hot_rows_xr((const uint8_t *)(intptr_t)ptrs[2 * j], x + (size_t)tok[j] * cols, rows, cols,
                act + (size_t)j * rows);
}

__global__ void k_ex_gelu(const float *act, int inner, float *act2)
{
    int j = blockIdx.x;
    const float *g = act + (size_t)j * 2 * inner;
    float *o = act2 + (size_t)j * inner;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = g[i];
        o[i] = 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v))) * g[inner + i];
    }
}

__global__ void k_ex_dn(const int64_t *ptrs, const float *act2, int rows, int inner, float *de)
{
    int j = blockIdx.y;
    hot_rows_xr((const uint8_t *)(intptr_t)ptrs[2 * j + 1], act2 + (size_t)j * inner, rows, inner,
                de + (size_t)j * rows);
}

/* k_ex_gu (tok null) and k_ex_gu_t on KQ_Q4X experts: block x a group of 16 rows */
__global__ void k_ex_gu_q4x(const int64_t *ptrs, const int32_t *tok, const float *x, int rows, int cols,
                            float *act)
{
    int j = blockIdx.y, grp = blockIdx.x;
    q4x_group_block((const uint8_t *)(intptr_t)ptrs[2 * j] + (size_t)grp * (cols / 32) * 288,
                    x + (tok != NULL ? (size_t)tok[j] * cols : 0), cols, act + (size_t)j * rows + 16 * grp);
}

__global__ void k_ex_dn_q4x(const int64_t *ptrs, const float *act2, int rows, int inner, float *de)
{
    int j = blockIdx.y, grp = blockIdx.x;
    q4x_group_block((const uint8_t *)(intptr_t)ptrs[2 * j + 1] + (size_t)grp * (inner / 32) * 288,
                    act2 + (size_t)j * inner, inner, de + (size_t)j * rows + 16 * grp);
}

/* The gate and up rows, the GELU and the down rows of n pairs (their pointers pd; td: each pair's
 * token on the rows hd, or null: all on the one row hd) into act, act2 and de */
static void ex_rows(ex_dev *x, const int64_t *pd, const int32_t *td, const float *hd, int n)
{
    int gu_rows = 2 * x->inner;
    if (x->q4x) {
        k_ex_gu_q4x<<<dim3((unsigned)(gu_rows / 16), n), 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
            pd, td, hd, gu_rows, x->hidden, x->act);
    } else if (td != NULL) {
        k_ex_gu_t<<<dim3((unsigned)cdiv(gu_rows, ROWS_PER_BLOCK * HOT_XR), n), 32 * ROWS_PER_BLOCK,
                    (size_t)x->hidden * 4, gg_stream>>>(pd, td, hd, gu_rows, x->hidden, x->act);
    } else {
        k_ex_gu<<<dim3((unsigned)cdiv(gu_rows, ROWS_PER_BLOCK * HOT_XR), n), 32 * ROWS_PER_BLOCK,
                  (size_t)x->hidden * 4, gg_stream>>>(pd, hd, gu_rows, x->hidden, x->act);
    }
    k_ex_gelu<<<n, 256, 0, gg_stream>>>(x->act, x->inner, x->act2);
    if (x->q4x) {
        k_ex_dn_q4x<<<dim3((unsigned)(x->dn_rows / 16), n), 32 * ROWS_PER_BLOCK, 0, gg_stream>>>(
            pd, x->act2, x->dn_rows, x->inner, x->de);
    } else {
        k_ex_dn<<<dim3((unsigned)cdiv(x->dn_rows, ROWS_PER_BLOCK * HOT_XR), n), 32 * ROWS_PER_BLOCK,
                  (size_t)x->inner * 4, gg_stream>>>(pd, x->act2, x->dn_rows, x->inner, x->de);
    }
}

__global__ void k_ex_sum(const float *w, const float *de, int n, int rows, float *out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= rows) {
        return;
    }
    float acc = 0.f;
    for (int j = 0; j < n; ++j) {
        acc += w[j] * de[(size_t)j * rows + c];
    }
    out[c] = acc;
}

/* k_ex_sum of a group: the pairs of token blockIdx.y */
__global__ void k_ex_sum_t(const float *w, const int32_t *tok, const float *de, int n, int rows, float *out)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x, t = blockIdx.y;
    if (c >= rows) {
        return;
    }
    float acc = 0.f;
    for (int j = 0; j < n; ++j) {
        if (tok[j] == t) {
            acc += w[j] * de[(size_t)j * rows + c];
        }
    }
    out[(size_t)t * rows + c] = acc;
}

/* The shapes and the buffers (after rc_gpu_init of this copy's device); q4x: the experts in KQ_Q4X. */
static int rc_gpu_ex_init_in(int layers, int hidden, int inner, int dn_rows, size_t gu_per, size_t dn_per,
                             int q4x)
{
    ex_dev *x = &gg_ex;
    if (q4x && ((2 * inner) % 16 != 0 || dn_rows % 16 != 0)) {
        return -1;
    }
    x->q4x = q4x;
    x->layers = layers;
    x->hidden = hidden;
    x->inner = inner;
    x->dn_rows = dn_rows;
    x->gu_per = gu_per;
    x->dn_per = dn_per;
    x->gu = (uint8_t **)calloc(layers, sizeof(uint8_t *));
    x->dn = (uint8_t **)calloc(layers, sizeof(uint8_t *));
    x->slots = (int32_t **)calloc(layers, sizeof(int32_t *));
    size_t stage = EXG_H + (size_t)EXG_TMAX * hidden * 4;      /* (a step's fits: EX_H < EXG_H) */
    CK(cudaHostAlloc((void **)&x->stage_h, stage, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&x->stage_d, stage));
    CK(cudaMalloc((void **)&x->act, (size_t)EXG_PMAX * 2 * inner * 4));
    CK(cudaMalloc((void **)&x->act2, (size_t)EXG_PMAX * inner * 4));
    CK(cudaMalloc((void **)&x->de, (size_t)EXG_PMAX * dn_rows * 4));
    CK(cudaMalloc((void **)&x->out_d, (size_t)EXG_TMAX * dn_rows * 4));
    CK(cudaHostAlloc((void **)&x->out_h, (size_t)EXG_TMAX * dn_rows * 4, cudaHostAllocDefault));
    CK(cudaEventCreateWithFlags(&x->done, cudaEventDisableTiming));
    return 0;
}

/* The experts of a layer this GPU holds: count of them (gu_per and dn_per
 * bytes each, in the order of their places), and slots: expert -> place, or
 * -1 (experts ints). */
static int rc_gpu_ex_store_in(int layer, int count, const void *gu, const void *dn, const int32_t *slots, int experts)
{
    ex_dev *x = &gg_ex;
    CK(cudaMalloc((void **)&x->gu[layer], (size_t)count * x->gu_per));
    CK(cudaMalloc((void **)&x->dn[layer], (size_t)count * x->dn_per));
    CK(cudaMemcpy(x->gu[layer], gu, (size_t)count * x->gu_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x->dn[layer], dn, (size_t)count * x->dn_per, cudaMemcpyHostToDevice));
    x->slots[layer] = (int32_t *)malloc((size_t)experts * sizeof(int32_t));
    memcpy(x->slots[layer], slots, (size_t)experts * sizeof(int32_t));
    return 0;
}

/* Nemotron: layers, L (the latent), kmax (top-k) and imax (the largest inner); the buffers of
 * a job of one token. */
static int rc_gpu_ex_nm_init_in(int layers, int L, int kmax, int imax, int nv4)
{
    ex_dev *x = &gg_ex;
    /* (KQ_NV4: NM_HOT_NV4's columns a multiple of 32, Super's inner 2688; top 22) */
    if (nv4 ? (L % 32 != 0 || imax % 32 != 0 || EXN_TMAX * kmax > EXN_PMAX)
            : (kmax > NM_KMAX || L % 256 != 0 || imax % 256 != 0)) {
        return -1;
    }
    x->nm = 1;
    x->nv4 = nv4;
    gg_nm_hg_off = getenv("RECLANK_EX_HG") != NULL && atoi(getenv("RECLANK_EX_HG")) == 0;
    x->layers = layers;
    x->L = L;
    x->kmax = kmax;
    x->imax = imax;
    x->dn_rows = L;
    x->hidden = L;
    x->gu = (uint8_t **)calloc(layers, sizeof(uint8_t *));
    x->dn = (uint8_t **)calloc(layers, sizeof(uint8_t *));
    x->slots_d = (int32_t **)calloc(layers, sizeof(int32_t *));
    x->inner_l = (int *)calloc(layers, sizeof(int));
    x->experts_l = (int *)calloc(layers, sizeof(int));
    x->hslots = (int32_t **)calloc(layers, sizeof(int32_t *));
    x->upp_l = (size_t *)calloc(layers, sizeof(size_t));
    x->dnp_l = (size_t *)calloc(layers, sizeof(size_t));
    const size_t T = EXN_TMAX, P = (size_t)EXN_TMAX * kmax;
    size_t stage = EXN_X + T * L * 4;
    CK(cudaHostAlloc((void **)&x->stage_h, stage, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&x->stage_d, stage));
    CK(cudaMalloc((void **)&x->act, P * imax * 4));
    CK(cudaMalloc((void **)&x->de, P * L * 4));
    CK(cudaMalloc((void **)&x->out_d, T * L * 4));
    CK(cudaHostAlloc((void **)&x->out_h, T * L * 4, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&x->qx, T * L));
    CK(cudaMalloc((void **)&x->qxl, T * L));
    CK(cudaMalloc((void **)&x->sx, T * L / 32 * 4));
    CK(cudaMalloc((void **)&x->qa, P * imax));
    CK(cudaMalloc((void **)&x->qal, P * imax));
    CK(cudaMalloc((void **)&x->sa, P * imax / 32 * 4));
    CK(cudaMalloc((void **)&x->env_d, 64));
    CK(cudaMemset(x->env_d, 0, 64));
    x->rec_d = (gp_rec *)x->stage_d;
    CK(cudaEventCreateWithFlags(&x->done, cudaEventDisableTiming));
    return 0;
}

/* Qwen: layers, hidden, inner, kmax (top-k); the buffers of a job of up to EXG_TMAX tokens
 * (GP_KQ_HOT_MOE's pairs: t k routed, t of the shared expert, unused here). */
static int rc_gpu_ex_kq_init_in(int layers, int hidden, int inner, int kmax)
{
    ex_dev *x = &gg_ex;
    if (kmax > NM_KMAX || EXG_TMAX * (kmax + 1) > 4 * EXG_PMAX) {
        return -1;
    }
    x->kq = 1;
    x->layers = layers;
    x->hidden = hidden;
    x->inner = inner;
    x->dn_rows = hidden;
    x->kmax = kmax;
    x->gu = (uint8_t **)calloc(layers, sizeof(uint8_t *));
    x->up = (uint8_t **)calloc(layers, sizeof(uint8_t *));
    x->dn = (uint8_t **)calloc(layers, sizeof(uint8_t *));
    x->slots_d = (int32_t **)calloc(layers, sizeof(int32_t *));
    x->experts_l = (int *)calloc(layers, sizeof(int));
    x->gtype_l = (int *)calloc(layers, sizeof(int));
    x->dtype_l = (int *)calloc(layers, sizeof(int));
    const size_t T = EXG_TMAX, P = (size_t)EXG_TMAX * (kmax + 1);
    size_t stage = EXN_X + T * hidden * 4;
    CK(cudaHostAlloc((void **)&x->stage_h, stage, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&x->stage_d, stage));
    CK(cudaMalloc((void **)&x->act, P * 2 * inner * 4));
    CK(cudaMalloc((void **)&x->act2, P * inner * 4));
    CK(cudaMalloc((void **)&x->de, P * hidden * 4));
    CK(cudaMalloc((void **)&x->out_d, T * hidden * 4));
    CK(cudaHostAlloc((void **)&x->out_h, T * hidden * 4, cudaHostAllocDefault));
    CK(cudaMalloc((void **)&x->env_d, 64));
    CK(cudaMemset(x->env_d, 0, 64));
    x->rec_d = (gp_rec *)x->stage_d;
    CK(cudaEventCreateWithFlags(&x->done, cudaEventDisableTiming));
    return 0;
}

/* Qwen: the count experts of a layer this GPU holds (gate, up, down blocks of g_per, u_per,
 * d_per bytes each, by slot; types gtype for gate and up, dtype for down), slots: expert -> slot,
 * or -1 (experts ints; on the device one more, -1). */
static int rc_gpu_ex_kq_store_in(int layer, int count, const void *gate, const void *up, const void *down,
                                 size_t g_per, size_t u_per, size_t d_per, int gtype, int dtype,
                                 const int32_t *slots, int experts)
{
    ex_dev *x = &gg_ex;
    CK(cudaMalloc((void **)&x->gu[layer], (size_t)count * g_per));
    CK(cudaMalloc((void **)&x->up[layer], (size_t)count * u_per));
    CK(cudaMalloc((void **)&x->dn[layer], (size_t)count * d_per));
    CK(cudaMemcpy(x->gu[layer], gate, (size_t)count * g_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x->up[layer], up, (size_t)count * u_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x->dn[layer], down, (size_t)count * d_per, cudaMemcpyHostToDevice));
    const int32_t none = -1;
    CK(cudaMalloc((void **)&x->slots_d[layer], (size_t)(experts + 1) * sizeof(int32_t)));
    CK(cudaMemcpy(x->slots_d[layer], slots, (size_t)experts * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x->slots_d[layer] + experts, &none, sizeof(int32_t), cudaMemcpyHostToDevice));
    x->experts_l[layer] = experts;
    x->gtype_l[layer] = gtype;
    x->dtype_l[layer] = dtype;
    return 0;
}

/* RECLANK_EX_GRAPH (1): a one-token Qwen job as a graph (ex_kq_capture); 0, its copies and launches */
static int ex_graph_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_EX_GRAPH");
        on = v == NULL || atoi(v) != 0;
    }
    return on;
}

/* The graph of a one-token job of key (the copy of nin bytes of the staging, the kernels of rec, the copy
 * of its row back): the copies and launches of a job took the CPU 36 us before its own experts (nsys; the
 * M4000, Qwen3.6), a graph's one launch a median of 25; both cards' decode +1-2%. Its kernels read the
 * record from rec_d, so the graph serves every layer of the same types and count; -1 when it cannot be
 * made (the caller launches the job). */
static int ex_kq_capture(ex_dev *x, gg_prog *stub, const gp_rec *rec, size_t nin, int key)
{
    cudaGraph_t gr;
    cudaGraphExec_t ge;
    CK(cudaStreamBeginCapture(gg_stream, cudaStreamCaptureModeThreadLocal));
    cudaMemcpyAsync(x->stage_d, x->stage_h, nin, cudaMemcpyHostToDevice, gg_stream);
    int rc = gg_launch(stub, rec, x->rec_d, x->env_d);
    cudaMemcpyAsync(x->out_h, x->out_d, (size_t)x->hidden * 4, cudaMemcpyDeviceToHost, gg_stream);
    cudaError_t ce = cudaStreamEndCapture(gg_stream, &gr);
    if (rc != 0 || ce != cudaSuccess) {
        if (ce == cudaSuccess) {
            cudaGraphDestroy(gr);
        }
        cudaGetLastError();
        return -1;
    }
    ce = cudaGraphInstantiate(&ge, gr, 0);
    cudaGraphDestroy(gr);
    if (ce != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    x->kq_exec[x->kq_ng] = ge;
    x->kq_key[x->kq_ng++] = key;
    return 0;
}

/* Qwen: t tokens of k pairs each (idx: the expert of each pair, or -1; weights w) on their rows h:
 * one GP_KQ_HOT_MOE record (no shared expert), its out the sum of each token (t rows) */
static int rc_gpu_ex_kq_job(int layer, int t, int k, const int32_t *idx, const float *w, const float *h)
{
    static gg_prog stub;        /* (zeros: no int8 scratch, no environment: the record's literals) */
    ex_dev *x = &gg_ex;
    if (t <= 0 || t > EXG_TMAX || k <= 0 || k > x->kmax || t * k > EXN_TK || x->gu[layer] == NULL) {
        return -1;
    }
    int32_t *si = (int32_t *)(x->stage_h + EXN_IDS);
    for (int p = 0; p < t * k; ++p) {
        si[p] = idx[p] >= 0 ? idx[p] : x->experts_l[layer];
        if (si[p] < 0 || si[p] > x->experts_l[layer]) {
            return -1;
        }
    }
    gp_rec rec;
    memset(&rec, 0, sizeof(rec));
    rec.op = GP_KQ_HOT_MOE;
    uint8_t *sd = x->stage_d;
    int64_t v[22] = {(int64_t)(intptr_t)(sd + EXN_X), (int64_t)(intptr_t)(sd + EXN_VAL),
                     (int64_t)(intptr_t)(sd + EXN_IDS), (int64_t)(intptr_t)x->slots_d[layer],
                     (int64_t)(intptr_t)x->gu[layer], (int64_t)(intptr_t)x->up[layer],
                     (int64_t)(intptr_t)x->dn[layer], (int64_t)(intptr_t)x->act, (int64_t)(intptr_t)x->act2,
                     (int64_t)(intptr_t)x->de, (int64_t)(intptr_t)x->out_d, k, x->inner, x->hidden,
                     x->gtype_l[layer], x->dtype_l[layer], t, 0, 0, 0, 0, 0};
    for (int i = 0; i < 22; ++i) {
        rec.tag[i] = GP_T_INT;
        rec.v[i] = v[i];
    }
    memcpy(x->stage_h, &rec, sizeof(rec));
    memcpy(x->stage_h + EXN_VAL, w, (size_t)t * k * 4);
    memcpy(x->stage_h + EXN_X, h, (size_t)t * x->hidden * 4);
    if (t == 1 && ex_graph_on()) {
        const int key = x->gtype_l[layer] | x->dtype_l[layer] << 8 | k << 16;
        int gi = 0;
        while (gi < x->kq_ng && x->kq_key[gi] != key) {
            ++gi;
        }
        if (gi == x->kq_ng && (gi == EXK_GRAPHS || ex_kq_capture(x, &stub, &rec, EXN_X + (size_t)x->hidden * 4,
                                                                 key) != 0)) {
            gi = -1;
        }
        if (gi >= 0) {
            CK(cudaGraphLaunch(x->kq_exec[gi], gg_stream));
            x->nout = x->hidden;
            CK(cudaEventRecord(x->done, gg_stream));
            x->pending = 1;
            return 0;
        }
    }
    CK(cudaMemcpyAsync(x->stage_d, x->stage_h, EXN_X + (size_t)t * x->hidden * 4, cudaMemcpyHostToDevice,
                       gg_stream));
    if (gg_launch(&stub, &rec, x->rec_d, x->env_d) != 0) {
        fprintf(stderr, "[ex] layer %d t %d k %d: gg_launch refused the record: %s\n", layer, t, k, gg_error);
        return -1;
    }
    CK(cudaGetLastError());
    x->nout = t * x->hidden;
    CK(cudaMemcpyAsync(x->out_h, x->out_d, (size_t)x->nout * 4, cudaMemcpyDeviceToHost, gg_stream));
    CK(cudaEventRecord(x->done, gg_stream));
    x->pending = 1;
    return 0;
}

/* Nemotron: the count experts of a layer this GPU holds (up_per and dn_per bytes each, by slot),
 * inner, and slots: expert -> slot, or -1 (experts ints; on the device one more, -1: the pairs of
 * a group that are not this GPU's). */
static int rc_gpu_ex_nm_store_in(int layer, int count, const void *up, const void *dn, size_t up_per,
                                 size_t dn_per, int inner, const int32_t *slots, int experts)
{
    ex_dev *x = &gg_ex;
    if (inner > x->imax || inner % (x->nv4 ? 32 : 256) != 0) {
        return -1;
    }
    CK(cudaMalloc((void **)&x->gu[layer], (size_t)count * up_per));
    CK(cudaMalloc((void **)&x->dn[layer], (size_t)count * dn_per));
    CK(cudaMemcpy(x->gu[layer], up, (size_t)count * up_per, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x->dn[layer], dn, (size_t)count * dn_per, cudaMemcpyHostToDevice));
    const int32_t none = -1;
    CK(cudaMalloc((void **)&x->slots_d[layer], (size_t)(experts + 1) * sizeof(int32_t)));
    CK(cudaMemcpy(x->slots_d[layer], slots, (size_t)experts * sizeof(int32_t), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(x->slots_d[layer] + experts, &none, sizeof(int32_t), cudaMemcpyHostToDevice));
    x->inner_l[layer] = inner;
    x->experts_l[layer] = experts;
    x->upp_l[layer] = up_per;
    x->dnp_l[layer] = dn_per;
    x->hslots[layer] = (int32_t *)malloc((size_t)experts * sizeof(int32_t));
    memcpy(x->hslots[layer], slots, (size_t)experts * sizeof(int32_t));
    return 0;
}

/* Super's extras of a layer (see ex_dev): its stacks in host memory (up, dn: experts rows of up_per and dn_per
 * bytes each), xmax the most extras of a job (the stage, made once, and the job's tables) */
static int rc_gpu_ex_nm_hosts_in(int layer, const void *up, const void *dn, int xmax)
{
    ex_dev *x = &gg_ex;
    if (!x->nv4 || layer < 0 || layer >= x->layers || x->gu[layer] == NULL || xmax <= 0) {
        return -1;
    }
    if (x->hup == NULL) {
        x->hup = (const uint8_t **)calloc(x->layers, sizeof(uint8_t *));
        x->hdn = (const uint8_t **)calloc(x->layers, sizeof(uint8_t *));
    }
    const size_t per = x->upp_l[layer] + x->dnp_l[layer];
    if (x->xst == NULL) {
        CK(cudaMalloc((void **)&x->xst, (size_t)xmax * per));
        x->xmax = xmax;
    } else if (xmax > x->xmax) {
        return -1;
    }
    const int E = x->experts_l[layer];
    if (x->jt_h == NULL || E > x->jt_e) {
        if (x->jt_h != NULL) {
            return -1;
        }
        x->jt_e = E < 1024 ? 1024 : E;
        const size_t jb = (size_t)(x->jt_e + 1) * 4 + 2 * (size_t)x->jt_e * 8 + 64;
        CK(cudaHostAlloc((void **)&x->jt_h, jb, cudaHostAllocDefault));
        CK(cudaMalloc((void **)&x->jt_d, jb));
    }
    x->hup[layer] = (const uint8_t *)up;
    x->hdn[layer] = (const uint8_t *)dn;
    return 0;
}

/* n experts of a layer into its slots anew (an expert device's refresh, devices.py: between prompts,
 * no job in flight): part p of expert i (p: Gemma's gate_up, Qwen's gate, Nemotron's up; Qwen's up;
 * the down) from the host address src[3 i + p] (0: none) of per[p] bytes into slot slot[i] of the
 * part's stack, then the layer's slot table (experts values: the device's of Qwen and Nemotron, its
 * last entry staying -1; the host's of Gemma's posts). The layer keeps its count of slots. */
static int rc_gpu_ex_put_in(int layer, int n, const int32_t *slot, const int64_t *src, const size_t *per,
                            const int32_t *slots, int experts)
{
    ex_dev *x = &gg_ex;
    uint8_t **stk[3] = {x->gu, x->up, x->dn};
    if (layer < 0 || layer >= x->layers || x->gu == NULL || x->gu[layer] == NULL) {
        return -1;
    }
    for (int i = 0; i < n; ++i) {
        for (int p = 0; p < 3; ++p) {
            if (per[p] == 0 || src[3 * i + p] == 0) {
                continue;
            }
            if (stk[p] == NULL || stk[p][layer] == NULL) {
                return -1;
            }
            CK(cudaMemcpyAsync(stk[p][layer] + (size_t)slot[i] * per[p], (const void *)(intptr_t)src[3 * i + p],
                               per[p], cudaMemcpyHostToDevice, gg_stream));
        }
    }
    if (x->slots_d != NULL && x->slots_d[layer] != NULL) {
        CK(cudaMemcpyAsync(x->slots_d[layer], slots, (size_t)experts * sizeof(int32_t), cudaMemcpyHostToDevice,
                           gg_stream));
    }
    CK(cudaStreamSynchronize(gg_stream));
    if (x->slots != NULL && x->slots[layer] != NULL) {
        memcpy(x->slots[layer], slots, (size_t)experts * sizeof(int32_t));
    }
    if (x->hslots != NULL && x->hslots[layer] != NULL) {
        memcpy(x->hslots[layer], slots, (size_t)experts * sizeof(int32_t));
    }
    if (gg_exg.tab_ok != NULL) {
        gg_exg.tab_ok[layer] = 0;       /* (the group jobs' tables of the layer made again) */
    }
    return 0;
}

/* Nemotron: t tokens of k pairs each (idx: the expert of each pair, or -1 for those this GPU does
 * not run; weights w) on their latent rows h: one GP_NM_HOT_MOE record (int8 x and act in two
 * planes, as the primary's NM_HOT_MOE; t 2-8 by expert, k_nm_hg_*), or (nv4) GP_NM_HOT_NV4's
 * (float x; the experts by slot in the stacks: no tables) */
static int rc_gpu_ex_nm_job(int layer, int t, int k, const int32_t *idx, const float *w, const float *h)
{
    ex_dev *x = &gg_ex;
    if (t <= 0 || t > EXN_TMAX || k <= 0 || k > x->kmax || t * k > (x->nv4 ? EXN_PMAX : EXN_TK) ||
        x->gu[layer] == NULL) {
        return -1;
    }
    int32_t *si = (int32_t *)(x->stage_h + EXN_IDS);
    int posted = 0;
    for (int p = 0; p < t * k; ++p) {
        si[p] = idx[p] >= 0 ? idx[p] : x->experts_l[layer];
        if (si[p] < 0 || si[p] > x->experts_l[layer]) {
            fprintf(stderr, "[ex] layer %d pair %d: expert %d of %d\n", layer, p, si[p], x->experts_l[layer]);
            return -1;
        }
        posted += idx[p] >= 0;
    }
    const int dbg = ex_debug() && t > 1;
    double t0 = 0;
    if (dbg) {
        void *cur = NULL;
        int dev = -1;
        ex_ctx_get(&cur);
        cudaGetDevice(&dev);
        t0 = ex_ms();
        fprintf(stderr, "[ex] job layer %d t %d k %d posted %d inner %d tid %ld ctx %p own %p dev %d stream %p\n",
                layer, t, k, posted, x->inner_l[layer], (long)syscall(SYS_gettid), cur, ex_own, dev, (void *)gg_stream);
    }
    gp_rec rec;
    memset(&rec, 0, sizeof(rec));
    rec.op = x->nv4 ? GP_NM_HOT_NV4 : GP_NM_HOT_MOE;
    uint8_t *sd = x->stage_d;
    int64_t v[19] = {(int64_t)(intptr_t)(sd + EXN_X), (int64_t)(intptr_t)(sd + EXN_IDS),
                     (int64_t)(intptr_t)(sd + EXN_VAL), (int64_t)(intptr_t)x->slots_d[layer],
                     (int64_t)(intptr_t)x->gu[layer], (int64_t)(intptr_t)x->dn[layer],
                     (int64_t)(intptr_t)x->act, (int64_t)(intptr_t)x->de, (int64_t)(intptr_t)x->out_d,
                     k, x->L, x->inner_l[layer], t,
                     (int64_t)(intptr_t)x->qx, (int64_t)(intptr_t)x->sx, (int64_t)(intptr_t)x->qa,
                     (int64_t)(intptr_t)x->sa, (int64_t)(intptr_t)x->qxl, (int64_t)(intptr_t)x->qal};
    /* (no dp4a, Maxwell: operands 0-12 only, NM_HOT_MOE's float form on the float rows of x; NM_HOT_NV4's
     * are those) */
    const int nop = gg_caps.dp4a && !x->nv4 ? 19 : 13;
    for (int i = 0; i < nop; ++i) {
        rec.tag[i] = GP_T_INT;
        rec.v[i] = v[i];
    }
    const int db = x->nv4 && x->jt_h != NULL ? ex_dst_find(layer) : -1;
    int dused = 0;
    if (x->nv4 && x->jt_h != NULL && ((x->hup != NULL && x->hup[layer] != NULL) || db >= 0)) {
        /* the job's slots (the layer's, each extra 0) and each expert's addresses (its slot's, an extra's place in
         * xst or in the decode stage): the extras' rows copied from the host's stacks first (DMA, this stream), the
         * staged ones' after their copies ahead (the stage's ready) */
        const int E = x->experts_l[layer];
        const size_t upp = x->upp_l[layer], dnp = x->dnp_l[layer];
        const int32_t *hs = x->hslots[layer];
        int32_t *js = (int32_t *)x->jt_h;
        const size_t tu_off = ((size_t)(x->jt_e + 1) * 4 + 63) / 64 * 64, td_off = tu_off + (size_t)x->jt_e * 8;
        int64_t *tu = (int64_t *)(x->jt_h + tu_off), *td = (int64_t *)(x->jt_h + td_off);
        memcpy(js, hs, (size_t)E * 4);
        js[E] = -1;
        int nx = 0;
        for (int p = 0; p < t * k; ++p) {
            const int ex = idx[p];
            if (ex < 0 || ex >= E) {
                continue;
            }
            const int q = db >= 0 && hs[ex] < 0 && js[ex] < 0 ? ex_dst_slot(db, ex) : -1;
            if (hs[ex] >= 0) {
                tu[ex] = (int64_t)(intptr_t)(x->gu[layer] + (size_t)hs[ex] * upp);
                td[ex] = (int64_t)(intptr_t)(x->dn[layer] + (size_t)hs[ex] * dnp);
            } else if (q >= 0) {
                uint8_t *src = gg_exd.b[db].buf + (size_t)q * gg_exd.per;
                tu[ex] = (int64_t)(intptr_t)src;
                td[ex] = (int64_t)(intptr_t)(src + upp);
                js[ex] = 0;
                CK(cudaStreamWaitEvent(gg_stream, gg_exd.b[db].done[q], 0));   /* (ended: settle) */
                ++dused;
            } else if (js[ex] < 0) {
                if (x->hup == NULL || x->hup[layer] == NULL || nx >= x->xmax) {
                    return -1;              /* (more extras than the stage, or none staged: the CPU runs them) */
                }
                uint8_t *dst = x->xst + (size_t)nx++ * (upp + dnp);
                CK(cudaMemcpyAsync(dst, x->hup[layer] + (size_t)ex * upp, upp, cudaMemcpyHostToDevice, gg_stream));
                CK(cudaMemcpyAsync(dst + upp, x->hdn[layer] + (size_t)ex * dnp, dnp, cudaMemcpyHostToDevice, gg_stream));
                tu[ex] = (int64_t)(intptr_t)dst;
                td[ex] = (int64_t)(intptr_t)(dst + upp);
                js[ex] = 0;
            }
        }
        if (ex_h2d(x->jt_d, x->jt_h, td_off + (size_t)E * 8) != 0) {
            return -1;
        }
        rec.v[3] = (int64_t)(intptr_t)x->jt_d;
        rec.tag[13] = rec.tag[14] = GP_T_INT;
        rec.v[13] = (int64_t)(intptr_t)(x->jt_d + tu_off);
        rec.v[14] = (int64_t)(intptr_t)(x->jt_d + td_off);
    }
    memcpy(x->stage_h, &rec, sizeof(rec));
    memcpy(x->stage_h + EXN_VAL, w, (size_t)t * k * 4);
    memcpy(x->stage_h + EXN_X, h, (size_t)t * x->L * 4);
    if (ex_h2d(x->stage_d, x->stage_h, EXN_X + (size_t)t * x->L * 4) != 0) {
        return -1;
    }
    gg_exd.used += dused;
    if (gg_launch(NULL, &rec, x->rec_d, x->env_d) != 0) {
        fprintf(stderr, "[ex] layer %d t %d k %d: gg_launch refused the record\n", layer, t, k);
        return -1;
    }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "[ex] layer %d t %d k %d: launch: %s\n", layer, t, k, cudaGetErrorString(le));
    }
    CK(le);
    if (dused > 0) {
        CK(cudaEventRecord(gg_exd.b[db].free, gg_stream));
    }
    if (dbg && ex_debug() >= 2) {
        cudaError_t se = cudaStreamSynchronize(gg_stream);
        fprintf(stderr, "[ex] job layer %d t %d: kernels %.2f ms: %s\n", layer, t, ex_ms() - t0, cudaGetErrorString(se));
        CK(se);
    }
    x->nout = t * x->L;
    CK(cudaMemcpyAsync(x->out_h, x->out_d, (size_t)x->nout * 4, cudaMemcpyDeviceToHost, gg_stream));
    CK(cudaEventRecord(x->done, gg_stream));
    x->pending = 1;
    return 0;
}

/* the decode stage (gg_exd): two buffers of nmax experts, stg the host's marks (layers x em); after
 * rc_gpu_ex_nm_ghosts (each layer's stacks and the bytes of its experts) */
static int rc_gpu_ex_nm_dstage_in(int nmax, uint8_t *stg, int em)
{
    ex_dev *x = &gg_ex;
    ex_dst *d = &gg_exd;
    if (!x->nv4 || nmax <= 0 || nmax > EXD_MAX || stg == NULL || em <= 0 || x->upp_l == NULL) {
        return -1;
    }
    size_t per = 0;
    int emax = 0;
    for (int l = 0; l < x->layers; ++l) {
        const size_t p = x->upp_l[l] + x->dnp_l[l];
        per = p > per ? p : per;
        emax = x->experts_l[l] > emax ? x->experts_l[l] : emax;
    }
    if (per == 0 || emax > em) {
        return -1;
    }
    if (d->b[0].buf == NULL) {
        CK(cudaStreamCreateWithFlags(&d->cs, cudaStreamNonBlocking));
        for (int b = 0; b < 2; ++b) {
            if (cudaMalloc((void **)&d->b[b].buf, (size_t)nmax * per) != cudaSuccess) {
                cudaGetLastError();
                snprintf(gg_error, sizeof(gg_error), "expert device: no memory for the decode stage (2 x %d experts)",
                         nmax);
                return -1;
            }
            CK(cudaEventCreateWithFlags(&d->b[b].ready, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&d->b[b].free, cudaEventDisableTiming));
            for (int q = 0; q < EXD_MAX; ++q) {
                CK(cudaEventCreateWithFlags(&d->b[b].done[q], cudaEventDisableTiming));
            }
            CK(cudaEventRecord(d->b[b].free, gg_stream));
            CK(cudaEventRecord(d->b[b].ready, d->cs));
            d->b[b].layer = -1;
            d->b[b].n = 0;
        }
    }
    if (x->jt_h == NULL) {
        /* (the job's slots and addresses, as rc_gpu_ex_nm_hosts's) */
        x->jt_e = emax < 1024 ? 1024 : emax;
        const size_t jb = (size_t)(x->jt_e + 1) * 4 + 2 * (size_t)x->jt_e * 8 + 64;
        CK(cudaHostAlloc((void **)&x->jt_h, jb, cudaHostAllocDefault));
        CK(cudaMalloc((void **)&x->jt_d, jb));
    }
    d->nmax = nmax;
    d->per = per;
    d->stg = stg;
    d->em = em;
    return 0;
}

/* er_ctx prefetch (the CPU's task of the MoE layer before): n experts of layer, the most likely first, copied ahead
 * into the stage's next buffer (its old experts unmarked; after the job that read them); those this GPU holds, or
 * past nmax, not */
static int rc_gpu_ex_nm_prefetch_in(int layer, int n, const int32_t *ids)
{
    ex_dev *x = &gg_ex;
    ex_dst *d = &gg_exd;
    ex_grp *gp = &gg_exg;
    if (d->b[0].buf == NULL || layer < 0 || layer >= x->layers || gp->hup == NULL || gp->hup[layer] == NULL ||
        x->gu[layer] == NULL || x->hslots[layer] == NULL || x->jt_h == NULL) {
        return -1;
    }
    const int b = d->next;
    d->next ^= 1;
    for (int q = 0; q < d->b[b].n; ++q) {
        if (d->b[b].ex[q] >= 0) {
            d->stg[(size_t)d->b[b].layer * d->em + d->b[b].ex[q]] = 0;
        }
    }
    d->b[b].n = 0;
    d->b[b].layer = layer;
    const int E = x->experts_l[layer];
    const size_t upp = x->upp_l[layer], dnp = x->dnp_l[layer];
    if (upp + dnp > d->per) {
        return -1;
    }
    const int32_t *hs = x->hslots[layer];
    uint8_t *mk = d->stg + (size_t)layer * d->em;
    CK(cudaStreamWaitEvent(d->cs, d->b[b].free, 0));
    int q = 0;
    for (int i = 0; i < n && q < d->nmax; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= E || hs[e] >= 0 || mk[e]) {
            continue;
        }
        uint8_t *dst = d->b[b].buf + (size_t)q * d->per;
        CK(cudaMemcpyAsync(dst, gp->hup[layer] + (size_t)e * upp, upp, cudaMemcpyHostToDevice, d->cs));
        CK(cudaMemcpyAsync(dst + upp, gp->hdn[layer] + (size_t)e * dnp, dnp, cudaMemcpyHostToDevice, d->cs));
        CK(cudaEventRecord(d->b[b].done[q], d->cs));
        d->b[b].ex[q++] = e;
        mk[e] = 1;
    }
    d->b[b].n = q;
    d->copied += q;
    CK(cudaEventRecord(d->b[b].ready, d->cs));
    return 0;
}

/* er_ctx settle (the CPU's task of layer, before it splits the pairs): of the experts copied ahead for layer, those
 * whose copies have not ended unmarked (their pairs the CPU's: no job waits for a copy; it still lands, unread) */
static int rc_gpu_ex_nm_settle_in(int layer)
{
    ex_dst *d = &gg_exd;
    const int b = ex_dst_find(layer);
    if (b < 0) {
        return 0;
    }
    for (int q = 0; q < d->b[b].n; ++q) {
        const int e = d->b[b].ex[q];
        if (e >= 0 && cudaEventQuery(d->b[b].done[q]) != cudaSuccess) {
            d->stg[(size_t)layer * d->em + e] = 0;
            d->b[b].ex[q] = -1;
            ++d->late;
        }
    }
    cudaGetLastError();
    return 0;
}

/* ---------- a share of a prompt group's experts (Super: models/nemotron/gpu.py XSPLIT) ----------
 * The primary's staged group gives this GPU the pairs of a share of each MoE layer's experts (those it holds and some
 * of the cold ones): the group's t rows of x, its t k pairs (the expert of each pair this GPU runs, or -1) and their
 * weights, in the primary's pinned buffers (GP_NM_REMOTE_POST, its group form). The job runs them as this GPU alone
 * runs a prompt group: NM_GROUP_MOE's records on k_nm_gtile_nv4p (dp4a), in mc chunks of the experts and windows of
 * win pairs, each expert from its slot when this GPU holds it, else from the host's stacks through the map (the
 * keeper's pages, registered portable by the primary); the sums of each row's pairs (t x L) back to pinned memory,
 * which rc_gpu_ex_wait adds to the CPU's (zeros: no pair of the CPU's). */

static void ex_grp_free(ex_grp *gp)
{
    cudaFree(gp->x);
    cudaFree(gp->val);
    cudaFree(gp->sx);
    cudaFree(gp->sa);
    cudaFree(gp->de);
    cudaFree(gp->out_d);
    cudaFreeHost(gp->out_h);
    cudaFree(gp->act);
    cudaFree(gp->idx);
    cudaFree(gp->work);
    cudaFree(gp->qx);
    cudaFree(gp->qxl);
    cudaFree(gp->qa);
    cudaFree(gp->qal);
    cudaFreeHost(gp->rec_h);
    cudaFreeHost(gp->lidx_h);
    gp->lidx_h = NULL;
    cudaFree(gp->rec_d);
    gp->x = gp->val = gp->sx = gp->sa = gp->de = gp->out_d = gp->out_h = gp->act = NULL;
    gp->idx = gp->work = NULL;
    gp->qx = gp->qxl = gp->qa = gp->qal = NULL;
    gp->rec_h = gp->rec_d = NULL;
    gp->tmax = gp->tkmax = gp->npmax = gp->nrmax = 0;
}

/* the job's buffers for t rows, tk pairs, a window of np pairs and nrec records (made again when larger) */
static int ex_grp_room(ex_dev *x, ex_grp *gp, int t, int tk, int np, int nrec)
{
    if (t <= gp->tmax && tk <= gp->tkmax && np <= gp->npmax && nrec <= gp->nrmax) {
        return 0;
    }
    t = t > gp->tmax ? t : gp->tmax;
    tk = tk > gp->tkmax ? tk : gp->tkmax;
    np = np > gp->npmax ? np : gp->npmax;
    nrec = nrec > gp->nrmax ? nrec : gp->nrmax;
    ex_grp_free(gp);
    const size_t L = (size_t)x->L, Im = (size_t)x->imax, Em = (size_t)gp->emax;
    if (cudaMalloc((void **)&gp->x, (size_t)t * L * 4) != cudaSuccess ||
        cudaMalloc((void **)&gp->val, (size_t)tk * 4) != cudaSuccess ||
        cudaMalloc((void **)&gp->idx, (size_t)tk * 4) != cudaSuccess ||
        cudaMalloc((void **)&gp->qx, (size_t)t * L) != cudaSuccess ||
        cudaMalloc((void **)&gp->qxl, (size_t)t * L) != cudaSuccess ||
        cudaMalloc((void **)&gp->sx, (size_t)t * L / 32 * 4) != cudaSuccess ||
        cudaMalloc((void **)&gp->act, 32 * 4) != cudaSuccess ||
        cudaMalloc((void **)&gp->qa, (size_t)np * Im) != cudaSuccess ||
        cudaMalloc((void **)&gp->qal, (size_t)np * Im) != cudaSuccess ||
        cudaMalloc((void **)&gp->sa, (size_t)np * Im / 32 * 4) != cudaSuccess ||
        cudaMalloc((void **)&gp->de, (size_t)np * L * 4) != cudaSuccess ||
        cudaMalloc((void **)&gp->out_d, (size_t)t * L * 4) != cudaSuccess ||
        cudaHostAlloc((void **)&gp->out_h, (size_t)t * L * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc((void **)&gp->work, (2 * Em + 1 + 2 * (size_t)tk) * 4) != cudaSuccess ||
        cudaHostAlloc((void **)&gp->rec_h, (size_t)nrec * sizeof(gp_rec), cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void **)&gp->lidx_h, (size_t)tk * 4, cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc((void **)&gp->rec_d, (size_t)nrec * sizeof(gp_rec)) != cudaSuccess) {
        cudaGetLastError();
        ex_grp_free(gp);
        snprintf(gg_error, sizeof(gg_error), "expert device: no memory for a group job of %d rows", t);
        return -1;
    }
    gp->tmax = t;
    gp->tkmax = tk;
    gp->npmax = np;
    gp->nrmax = nrec;
    return 0;
}

/* each layer's stacks in host memory, for the group jobs' cold experts (the tables made at the layer's first job, and
 * again after its slots change: rc_gpu_ex_put) */
static int rc_gpu_ex_nm_ghosts_in(int layer, const void *up, const void *dn, int experts, int inner, size_t up_per,
                                  size_t dn_per)
{
    ex_dev *x = &gg_ex;
    ex_grp *gp = &gg_exg;
    if (!x->nv4 || layer < 0 || layer >= x->layers || experts <= 0 || inner > x->imax || inner % 32 != 0) {
        return -1;
    }
    if (x->experts_l[layer] == 0) {
        /* (a layer of none of this GPU's experts: its shape for the group jobs) */
        x->experts_l[layer] = experts;
        x->inner_l[layer] = inner;
        x->upp_l[layer] = up_per;
        x->dnp_l[layer] = dn_per;
    } else if (x->experts_l[layer] != experts || x->inner_l[layer] != inner || x->upp_l[layer] != up_per ||
               x->dnp_l[layer] != dn_per) {
        return -1;
    }
    if (gp->hup == NULL) {
        gp->hup = (const uint8_t **)calloc(x->layers, sizeof(uint8_t *));
        gp->hdn = (const uint8_t **)calloc(x->layers, sizeof(uint8_t *));
        gp->tab_ok = (uint8_t *)calloc(x->layers, 1);
        gp->tu_h = (int64_t **)calloc(x->layers, sizeof(int64_t *));
        gp->td_h = (int64_t **)calloc(x->layers, sizeof(int64_t *));
        gp->tu_d = (int64_t **)calloc(x->layers, sizeof(int64_t *));
        gp->td_d = (int64_t **)calloc(x->layers, sizeof(int64_t *));
    }
    if (gp->tu_h[layer] == NULL) {
        const size_t tb = (size_t)experts * 8;
        if (cudaHostAlloc((void **)&gp->tu_h[layer], tb, cudaHostAllocDefault) != cudaSuccess ||
            cudaHostAlloc((void **)&gp->td_h[layer], tb, cudaHostAllocDefault) != cudaSuccess ||
            cudaMalloc((void **)&gp->tu_d[layer], tb) != cudaSuccess ||
            cudaMalloc((void **)&gp->td_d[layer], tb) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
    }
    gp->emax = experts > gp->emax ? experts : gp->emax;
    gp->hup[layer] = (const uint8_t *)up;
    gp->hdn[layer] = (const uint8_t *)dn;
    gp->tab_ok[layer] = 0;
    return 0;
}

/* RECLANK_EX_STAGE=0: a group job's cold experts read through the map (k_nm_gtile_nv4p) as before;
 * RECLANK_EX_STAGE_EXPERTS: the experts of a stage buffer (48: 2 x 149 MB for Super; a prompt on both cards,
 * 32 / 48 / 64: 1K 274 / 274 / 242 tok/s, 2K 391 / 403 / 402) */
static int ex_stage_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_EX_STAGE");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* RECLANK_EX_PREFETCH=0: no chunks of the next layer copied ahead */
static int ex_prefetch_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_EX_PREFETCH");
        on = !(v && v[0] == '0');
    }
    return on;
}

static int ex_stage_n(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_EX_STAGE_EXPERTS");
        n = v ? atoi(v) : 48;
        n = n < 1 ? 1 : n;
    }
    return n;
}

/* the stage's buffers and tables (once, for experts of per bytes) */
static int ex_stage_alloc(ex_dev *x, ex_grp *gp, size_t per)
{
    if (gp->sbuf[0] != NULL) {
        return per <= gp->sper ? 0 : -1;
    }
    if (gp->sfail) {
        return -1;
    }
    gp->sc = ex_stage_n();
    gp->sper = per;
    const size_t em = (size_t)gp->emax;
    if (cudaMalloc((void **)&gp->sbuf[0], (size_t)gp->sc * per) != cudaSuccess ||
        cudaMalloc((void **)&gp->sbuf[1], (size_t)gp->sc * per) != cudaSuccess ||
        cudaStreamCreateWithFlags(&gp->cs, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc((void **)&gp->ltu_d, em * 8) != cudaSuccess ||
        cudaMalloc((void **)&gp->ltd_d, em * 8) != cudaSuccess ||
        cudaHostAlloc((void **)&gp->ltu_h, em * 8, cudaHostAllocDefault) != cudaSuccess ||
        cudaHostAlloc((void **)&gp->ltd_h, em * 8, cudaHostAllocDefault) != cudaSuccess ||
        cudaMalloc((void **)&gp->lslot_d, em * 4) != cudaSuccess || cudaMemset(gp->lslot_d, 0, em * 4) != cudaSuccess) {
        cudaGetLastError();
        cudaFree(gp->sbuf[0]);
        cudaFree(gp->sbuf[1]);
        cudaFree(gp->ltu_d);
        cudaFree(gp->ltd_d);
        cudaFree(gp->lslot_d);
        cudaFreeHost(gp->ltu_h);
        cudaFreeHost(gp->ltd_h);
        if (gp->cs != NULL) {
            cudaStreamDestroy(gp->cs);
        }
        gp->sbuf[0] = gp->sbuf[1] = NULL;
        gp->ltu_d = gp->ltd_d = gp->ltu_h = gp->ltd_h = NULL;
        gp->lslot_d = NULL;
        gp->cs = NULL;
        cudaGetLastError();
        gp->sfail = 1;
        fprintf(stderr, "[ex] expert device: no memory for the stage (2 x %d experts of %.1f MB): its share mapped\n",
                gp->sc, per / 1e6);
        return -1;
    }
    for (int b = 0; b < 2; ++b) {
        CK(cudaEventCreateWithFlags(&gp->ready[b], cudaEventDisableTiming));
        CK(cudaEventCreateWithFlags(&gp->sfree[b], cudaEventDisableTiming));
        CK(cudaEventRecord(gp->sfree[b], gg_stream));
    }
    gp->lid = (int32_t *)malloc(em * 4);
    gp->lcnt = (int32_t *)malloc(em * 4);
    gp->lcold = (int32_t *)malloc(em * 4);
    (void)x;
    return 0;
}

/* a layer's cold share for the stage (the experts of a split prompt group this GPU runs and does not hold, the most
 * routed first: the job before copies the first 2 sc ahead); n 0: none */
static int rc_gpu_ex_nm_gmask_in(int layer, int n, const int32_t *experts)
{
    ex_dev *x = &gg_ex;
    ex_grp *gp = &gg_exg;
    if (layer < 0 || layer >= x->layers || n < 0 || n > x->experts_l[layer]) {
        return -1;
    }
    if (gp->gm == NULL) {
        gp->gm = (int32_t **)calloc(x->layers, sizeof(int32_t *));
        gp->gm_n = (int *)calloc(x->layers, sizeof(int));
        gp->pf_layer = -1;
    }
    if (gp->gm[layer] == NULL && n > 0) {
        gp->gm[layer] = (int32_t *)malloc((size_t)x->experts_l[layer] * 4);
    }
    if (n > 0) {
        memcpy(gp->gm[layer], experts, (size_t)n * 4);
    }
    gp->gm_n[layer] = n;
    return 0;
}

/* The group job with its cold experts staged: the share's experts by local ids, the cold ones in chunks of sc (each a
 * range of the ids: NM_GROUP_MOE's operand 23 chunks) and those this GPU holds one group; each cold chunk's rows copied
 * by DMA from the host's stacks into stage buffer c % 2 on the stream cs (after the buffer's last chunk ran), the
 * chunk's records after its copies; every expert in the GPU's memory then (operand 22 = 3, a slot table of zeros:
 * k_nm_gtile_nv4pc). The DMA ~7.8 GB/s beside the primary's copies, the map's small reads ~5-6.5. */
static int ex_gstage_job(ex_dev *x, ex_grp *gp, int layer, int t, int k, const int32_t *idx, const float *w,
                         const float *xh, int win, int x16)
{
    const int E = x->experts_l[layer], L = x->L, inner = x->inner_l[layer], tk = t * k;
    const size_t upp = x->upp_l[layer], dnp = x->dnp_l[layer], per = upp + dnp;
    if (ex_stage_alloc(x, gp, per) != 0) {
        return -1;
    }
    const int sc = gp->sc;
    const int32_t *hs = x->gu[layer] != NULL ? x->hslots[layer] : NULL;
    /* the local ids in the order of the job's groups: the chunks the job before copied ahead, the held experts (their
     * work beside the copies of the next chunks), the other chunks (x's planes and the sort with the first group, the
     * sums with the last: NM_GROUP_MOE's x0 0 and x0 + nx = El) */
    int32_t *lid = gp->lid, *lcnt = gp->lcnt, *lcold = gp->lcold, *lheld = gp->lcnt;   /* (lheld until the counts) */
    for (int e = 0; e < E; ++e) {
        lid[e] = -1;
    }
    for (int p = 0; p < tk; ++p) {
        if (idx[p] >= 0 && idx[p] < E) {
            lid[idx[p]] = -2;
        }
    }
    int nh = 0, nc = 0;
    for (int e = 0; e < E; ++e) {
        if (lid[e] == -2 && hs != NULL && hs[e] >= 0) {
            lheld[nh++] = e;
            lid[e] = -3;
        }
    }
    /* the cold ones: those the job before copied ahead at their places (pairs or not), then the others of the pairs */
    const int npre = gp->pf_layer == layer ? gp->pf_n : 0;
    for (int pass = 0; pass < 2; ++pass) {
        const int n = pass == 0 ? npre : E;
        for (int i = 0; i < n; ++i) {
            const int e = pass == 0 ? gp->pf_ex[i] : i;
            if (pass == 0 && (e < 0 || e >= E || lid[e] >= 0 || (hs != NULL && hs[e] >= 0))) {
                break;      /* (held since: the rest copied again) */
            }
            if (pass == 1 && lid[e] != -2) {
                continue;
            }
            lcold[nc] = e;
            lid[e] = nc++;
        }
    }
    /* the leading chunks copied ahead: each whole, its experts at their places */
    int pf = 0;
    for (int q = 0; q < nc && q < npre && gp->pf_ex[q] == lcold[q]; ++q) {
        if ((q + 1) % sc == 0 || q + 1 == nc) {
            pf = q / sc + 1;
        }
    }
    gp->pf_layer = -1;
    const int P = pf * sc < nc ? pf * sc : nc, El = nh + nc;
    if (El == 0) {
        return -1;
    }
    for (int q = 0; q < nc; ++q) {
        const int id = q < P ? q : nh + q;
        uint8_t *dst = gp->sbuf[(q / sc) % 2] + (size_t)(q % sc) * per;
        gp->ltu_h[id] = (int64_t)(intptr_t)dst;
        gp->ltd_h[id] = (int64_t)(intptr_t)(dst + upp);
        lid[lcold[q]] = id;
    }
    for (int i = 0; i < nh; ++i) {
        const int e = lheld[i];
        gp->ltu_h[P + i] = (int64_t)(intptr_t)(x->gu[layer] + (size_t)hs[e] * upp);
        gp->ltd_h[P + i] = (int64_t)(intptr_t)(x->dn[layer] + (size_t)hs[e] * dnp);
        lid[e] = P + i;
    }
    for (int i = 0; i < El; ++i) {
        lcnt[i] = 0;
    }
    for (int p = 0; p < tk; ++p) {
        if (idx[p] >= 0 && idx[p] < E) {
            ++lcnt[lid[idx[p]]];
        }
    }
    /* the groups (gch: the chunk, -1 the held); each its passes of win pairs */
    const int ncc = (nc + sc - 1) / sc, ng = (nh > 0) + ncc;
    int gx0[64], gnx[64], gps[64], gch[64];
    if (ng > 64) {
        return -1;
    }
    int gi = 0, nrec = 0;
    for (int c = 0; c < ncc; ++c) {
        if (c == pf && nh > 0) {
            gx0[gi] = P;
            gnx[gi] = nh;
            gch[gi++] = -1;
        }
        gx0[gi] = (c < pf ? 0 : nh) + c * sc;
        gnx[gi] = nc - c * sc < sc ? nc - c * sc : sc;
        gch[gi++] = c;
    }
    if (pf >= ncc && nh > 0) {
        gx0[gi] = P;
        gnx[gi] = nh;
        gch[gi++] = -1;
    }
    for (int g = 0; g < ng; ++g) {
        int np = 0;
        for (int i = gx0[g]; i < gx0[g] + gnx[g]; ++i) {
            np += lcnt[i];
        }
        gps[g] = win > 0 ? (np + win - 1) / win : 1;
        gps[g] = gps[g] < 1 ? 1 : gps[g];
        if (gps[g] > 255) {
            return -1;
        }
        nrec += gps[g];
    }
    const int npw = win > 0 ? win : tk;
    if (ex_grp_room(x, gp, t, tk, npw, nrec) != 0) {
        return -1;
    }
    /* the job's pairs by local id (pinned: their copy is async) */
    int32_t *lidx = gp->lidx_h;
    for (int p = 0; p < tk; ++p) {
        lidx[p] = idx[p] >= 0 && idx[p] < E ? lid[idx[p]] : -1;
    }
    const double t0 = ex_debug() ? ex_ms() : 0;
    CK(cudaMemcpyAsync(gp->x, xh, (size_t)t * L * 4, cudaMemcpyHostToDevice, gg_stream));
    CK(cudaMemcpyAsync(gp->idx, lidx, (size_t)tk * 4, cudaMemcpyHostToDevice, gg_stream));
    CK(cudaMemcpyAsync(gp->val, w, (size_t)tk * 4, cudaMemcpyHostToDevice, gg_stream));
    CK(cudaMemcpyAsync(gp->ltu_d, gp->ltu_h, (size_t)El * 8, cudaMemcpyHostToDevice, gg_stream));
    CK(cudaMemcpyAsync(gp->ltd_d, gp->ltd_h, (size_t)El * 8, cudaMemcpyHostToDevice, gg_stream));
    gp_rec rec;
    memset(&rec, 0, sizeof(rec));
    rec.op = GP_NM_GROUP_MOE;
    int64_t v[23] = {(int64_t)(intptr_t)gp->x, (int64_t)(intptr_t)gp->idx, (int64_t)(intptr_t)gp->val,
                     (int64_t)(intptr_t)gp->ltu_d, (int64_t)(intptr_t)gp->ltd_d, (int64_t)(intptr_t)gp->qx,
                     (int64_t)(intptr_t)gp->sx, (int64_t)(intptr_t)gp->lslot_d, (int64_t)(intptr_t)gp->qa,
                     (int64_t)(intptr_t)gp->sa, (int64_t)(intptr_t)gp->de, (int64_t)(intptr_t)gp->out_d,
                     (int64_t)(intptr_t)gp->work, k, L, inner, t, El, t,
                     x16 ? (int64_t)(intptr_t)gp->qxl : 0, x16 ? (int64_t)(intptr_t)gp->qal : 0, KQ_NV4, 3};
    for (int i = 0; i < 23; ++i) {
        rec.tag[i] = GP_T_INT;
        rec.v[i] = v[i];
    }
    if (!x16) {
        rec.tag[19] = rec.tag[20] = GP_T_NONE;
    }
    int n = 0;
    for (int g = 0; g < ng; ++g) {
        for (int ps = 0; ps < gps[g]; ++ps) {
            gp->rec_h[n] = rec;
            gp->rec_h[n].tag[23] = GP_T_INT;
            gp->rec_h[n].v[23] = (int64_t)gx0[g] | (int64_t)gnx[g] << 16 | (int64_t)ps << 32 |
                                 (int64_t)(win > 0 ? win : 0) << 40;
            ++n;
        }
    }
    CK(cudaMemcpyAsync(gp->rec_d, gp->rec_h, (size_t)n * sizeof(gp_rec), cudaMemcpyHostToDevice, gg_stream));
    /* each group: a chunk's copies (after its buffer's last chunk) and its records (after its copies) */
    n = 0;
    for (int g = 0; g < ng; ++g) {
        const int c = gch[g];
        if (c >= 0) {
            const int b = c % 2;
            if (c >= pf) {
                CK(cudaStreamWaitEvent(gp->cs, gp->sfree[b], 0));
                for (int q = c * sc; q < c * sc + gnx[g]; ++q) {
                    const int ex = lcold[q];
                    uint8_t *dst = gp->sbuf[b] + (size_t)(q % sc) * per;
                    const uint8_t *su = gp->hup[layer] + (size_t)ex * upp, *sd = gp->hdn[layer] + (size_t)ex * dnp;
                    CK(cudaMemcpyAsync(dst, su, upp, cudaMemcpyHostToDevice, gp->cs));
                    CK(cudaMemcpyAsync(dst + upp, sd, dnp, cudaMemcpyHostToDevice, gp->cs));
                }
                CK(cudaEventRecord(gp->ready[b], gp->cs));
            }
            CK(cudaStreamWaitEvent(gg_stream, gp->ready[b], 0));
        }
        for (int ps = 0; ps < gps[g]; ++ps, ++n) {
            if (gg_launch(NULL, gp->rec_h + n, gp->rec_d + n, x->env_d) != 0) {
                fprintf(stderr, "[ex] staged group layer %d t %d: gg_launch refused record %d\n", layer, t, n);
                return -1;
            }
        }
        if (c >= 0) {
            CK(cudaEventRecord(gp->sfree[c % 2], gg_stream));
        }
    }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "[ex] staged group layer %d t %d: launch: %s\n", layer, t, cudaGetErrorString(le));
    }
    CK(le);
    gp->nout = t * L;
    CK(cudaMemcpyAsync(gp->out_h, gp->out_d, (size_t)gp->nout * 4, cudaMemcpyDeviceToHost, gg_stream));
    CK(cudaEventRecord(x->done, gg_stream));
    gp->pending = 1;
    if (ex_debug()) {
        fprintf(stderr, "[ex] staged group layer %d t %d k %d: %d held, %d cold in %d chunks (%d ahead), %d records, "
                        "posted in %.2f ms; win %d x16 %d\n",
                layer, t, k, nh, nc, ncc, pf, nrec, ex_ms() - t0, win, x16);
    }
    return 0;
}

static int rc_gpu_ex_nm_gstage(ex_dev *x, ex_grp *gp, int layer, int t, int k, const int32_t *idx, const float *w,
                               const float *xh, int win, int x16)
{
    if (ex_gstage_job(x, gp, layer, t, k, idx, w, xh, win, x16) != 0) {
        return -1;
    }
    const int sc = gp->sc;
    /* the next layer of a share: its first two chunks into the buffers once this job's last chunks free them (the
     * copies beside the primary's work between the two layers: its next job finds them there) */
    int nxl = -1;
    for (int l = layer + 1; l < x->layers && gp->gm != NULL; ++l) {
        if (gp->gm[l] != NULL && gp->gm_n[l] > 0) {
            nxl = l;
            break;
        }
    }
    int npf = 0;
    if (nxl >= 0 && ex_prefetch_on() && x->upp_l[nxl] + x->dnp_l[nxl] <= gp->sper) {
        const size_t up2 = x->upp_l[nxl], dn2 = x->dnp_l[nxl], per2 = up2 + dn2;
        const int32_t *hs2 = x->gu[nxl] != NULL ? x->hslots[nxl] : NULL;
        if (gp->pf_ex == NULL) {
            gp->pf_ex = (int *)malloc((size_t)2 * sc * sizeof(int));
        }
        int q = 0;
        for (int i = 0; i < gp->gm_n[nxl] && q < 2 * sc; ++i) {
            const int ex = gp->gm[nxl][i];
            if (ex < 0 || ex >= x->experts_l[nxl] || (hs2 != NULL && hs2[ex] >= 0)) {
                continue;
            }
            const int b = (q / sc) % 2;
            if (q % sc == 0) {
                CK(cudaStreamWaitEvent(gp->cs, gp->sfree[b], 0));
            }
            gp->pf_ex[q] = ex;
            uint8_t *dst = gp->sbuf[b] + (size_t)(q % sc) * per2;
            CK(cudaMemcpyAsync(dst, gp->hup[nxl] + (size_t)ex * up2, up2, cudaMemcpyHostToDevice, gp->cs));
            CK(cudaMemcpyAsync(dst + up2, gp->hdn[nxl] + (size_t)ex * dn2, dn2, cudaMemcpyHostToDevice, gp->cs));
            ++q;
            if (q % sc == 0) {
                CK(cudaEventRecord(gp->ready[b], gp->cs));
            }
        }
        if (q % sc != 0) {
            CK(cudaEventRecord(gp->ready[(q / sc) % 2], gp->cs));
        }
        npf = (q + sc - 1) / sc;
        gp->pf_layer = nxl;
        gp->pf_n = q;
    }
    if (ex_debug()) {
        fprintf(stderr, "[ex] staged group layer %d: layer %d's %d chunks ahead\n", layer, nxl, npf);
    }
    return 0;
}

static int rc_gpu_ex_nm_gpost_in(int layer, int t, int k, const int32_t *idx, const float *w, const float *xh, int mc,
                                 int win, int x16)
{
    ex_dev *x = &gg_ex;
    ex_grp *gp = &gg_exg;
    if (!x->nv4 || layer < 0 || layer >= x->layers || gp->hup == NULL || gp->hup[layer] == NULL || t <= 0 || k <= 0 ||
        mc <= 0 || x->experts_l[layer] % mc != 0 || x->pending || gp->pending) {
        return -1;
    }
    const int E = x->experts_l[layer], L = x->L, inner = x->inner_l[layer], tk = t * k, Ec = E / mc;
    const int np = win > 0 ? win : tk, passes = win > 0 ? (tk + win - 1) / win : 1, nrec = mc * passes;
    if (passes > 255 || E > gp->emax || ex_grp_room(x, gp, t, tk, np, nrec) != 0) {
        return -1;
    }
    if (ex_stage_on() && !gg_caps.tc && ex_stage_alloc(x, gp, x->upp_l[layer] + x->dnp_l[layer]) == 0) {
        return rc_gpu_ex_nm_gstage(x, gp, layer, t, k, idx, w, xh, win, x16);   /* (else mapped: no room for it) */
    }
    const double t0 = ex_debug() ? ex_ms() : 0;
    int64_t *tu = gp->tu_h[layer], *td = gp->td_h[layer];
    if (!gp->tab_ok[layer]) {
        /* (each expert's rows: its slot's here, else the host's stacks through the map: the device address of a
         * page registered portable, cudaHostGetDevicePointer of the stack's start) */
        const size_t upp = x->upp_l[layer], dnp = x->dnp_l[layer];
        void *du = NULL, *dd = NULL;
        if (cudaHostGetDevicePointer(&du, (void *)gp->hup[layer], 0) != cudaSuccess ||
            cudaHostGetDevicePointer(&dd, (void *)gp->hdn[layer], 0) != cudaSuccess) {
            cudaGetLastError();
            fprintf(stderr, "[ex] layer %d: the host's stacks are not mapped for this GPU\n", layer);
            return -1;
        }
        const int32_t *hs = x->gu[layer] != NULL ? x->hslots[layer] : NULL;
        for (int e = 0; e < E; ++e) {
            if (hs != NULL && hs[e] >= 0) {
                tu[e] = (int64_t)(intptr_t)(x->gu[layer] + (size_t)hs[e] * upp);
                td[e] = (int64_t)(intptr_t)(x->dn[layer] + (size_t)hs[e] * dnp);
            } else {
                tu[e] = (int64_t)(intptr_t)((uint8_t *)du + (size_t)e * upp);
                td[e] = (int64_t)(intptr_t)((uint8_t *)dd + (size_t)e * dnp);
            }
        }
        CK(cudaMemcpyAsync(gp->tu_d[layer], tu, (size_t)E * 8, cudaMemcpyHostToDevice, gg_stream));
        CK(cudaMemcpyAsync(gp->td_d[layer], td, (size_t)E * 8, cudaMemcpyHostToDevice, gg_stream));
        gp->tab_ok[layer] = 1;
    }
    CK(cudaMemcpyAsync(gp->x, xh, (size_t)t * L * 4, cudaMemcpyHostToDevice, gg_stream));
    CK(cudaMemcpyAsync(gp->idx, idx, (size_t)tk * 4, cudaMemcpyHostToDevice, gg_stream));
    CK(cudaMemcpyAsync(gp->val, w, (size_t)tk * 4, cudaMemcpyHostToDevice, gg_stream));
    /* the records: the primary's of a mapped group on dp4a (gpu.py NmEmit._moe: operands 0-23, 23 the chunk and window) */
    gp_rec rec;
    memset(&rec, 0, sizeof(rec));
    rec.op = GP_NM_GROUP_MOE;
    int64_t v[22] = {(int64_t)(intptr_t)gp->x, (int64_t)(intptr_t)gp->idx, (int64_t)(intptr_t)gp->val,
                     (int64_t)(intptr_t)gp->tu_d[layer], (int64_t)(intptr_t)gp->td_d[layer], (int64_t)(intptr_t)gp->qx,
                     (int64_t)(intptr_t)gp->sx, (int64_t)(intptr_t)gp->act, (int64_t)(intptr_t)gp->qa,
                     (int64_t)(intptr_t)gp->sa, (int64_t)(intptr_t)gp->de, (int64_t)(intptr_t)gp->out_d,
                     (int64_t)(intptr_t)gp->work, k, L, inner, t, E, t,
                     x16 ? (int64_t)(intptr_t)gp->qxl : 0, x16 ? (int64_t)(intptr_t)gp->qal : 0, KQ_NV4};
    for (int i = 0; i < 22; ++i) {
        rec.tag[i] = GP_T_INT;
        rec.v[i] = v[i];
    }
    if (!x16) {
        rec.tag[19] = rec.tag[20] = GP_T_NONE;
    }
    rec.tag[22] = GP_T_NONE;
    if (!gg_caps.tc && x->gu[layer] != NULL && x->slots_d[layer] != NULL) {
        /* (its held experts by k_nm_gtile_nv4pc: operand 22 = 3, operand 7 the layer's slots) */
        rec.tag[22] = GP_T_INT;
        rec.v[22] = 3;
        rec.tag[7] = GP_T_INT;
        rec.v[7] = (int64_t)(intptr_t)x->slots_d[layer];
    }
    int n = 0;
    for (int c = 0; c < mc; ++c) {
        for (int ps = 0; ps < passes; ++ps) {
            gp->rec_h[n] = rec;
            if (mc > 1 || win > 0) {
                gp->rec_h[n].tag[23] = GP_T_INT;
                gp->rec_h[n].v[23] = (int64_t)(c * Ec) | (int64_t)Ec << 16 | (int64_t)ps << 32 | (int64_t)(win > 0 ? win : 0) << 40;
            }
            ++n;
        }
    }
    CK(cudaMemcpyAsync(gp->rec_d, gp->rec_h, (size_t)n * sizeof(gp_rec), cudaMemcpyHostToDevice, gg_stream));
    for (int i = 0; i < n; ++i) {
        if (gg_launch(NULL, gp->rec_h + i, gp->rec_d + i, x->env_d) != 0) {
            fprintf(stderr, "[ex] group layer %d t %d: gg_launch refused record %d\n", layer, t, i);
            return -1;
        }
    }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "[ex] group layer %d t %d: launch: %s\n", layer, t, cudaGetErrorString(le));
    }
    CK(le);
    gp->nout = t * L;
    CK(cudaMemcpyAsync(gp->out_h, gp->out_d, (size_t)gp->nout * 4, cudaMemcpyDeviceToHost, gg_stream));
    CK(cudaEventRecord(x->done, gg_stream));
    gp->pending = 1;
    if (ex_debug()) {
        fprintf(stderr, "[ex] group job layer %d t %d k %d: %d records (%d chunks, %d passes) posted in %.2f ms\n",
                layer, t, k, n, mc, passes, ex_ms() - t0);
    }
    return 0;
}

/* Start the n experts ids (weights w) of a layer on the row h: the copy of
 * the job, then the kernels, on this copy's stream; rc_gpu_ex_wait ends it.
 * The caller (a CPU thread of the step) goes on with its own experts. */
static int rc_gpu_ex_post_in(int layer, int n, const int32_t *ids, const float *w, const float *h)
{
    ex_dev *x = &gg_ex;
    if (x->nm) {
        return rc_gpu_ex_nm_job(layer, 1, n, ids, w, h);
    }
    if (x->kq) {
        return rc_gpu_ex_kq_job(layer, 1, n, ids, w, h);
    }
    if (n <= 0 || n > EX_MAXK || x->gu[layer] == NULL) {
        return -1;
    }
    int64_t *ptrs = (int64_t *)x->stage_h;
    for (int j = 0; j < n; ++j) {
        int s = x->slots[layer][ids[j]];
        ptrs[2 * j] = (int64_t)(intptr_t)(x->gu[layer] + (size_t)s * x->gu_per);
        ptrs[2 * j + 1] = (int64_t)(intptr_t)(x->dn[layer] + (size_t)s * x->dn_per);
    }
    memcpy(x->stage_h + EX_W, w, (size_t)n * 4);
    memcpy(x->stage_h + EX_H, h, (size_t)x->hidden * 4);
    CK(cudaMemcpyAsync(x->stage_d, x->stage_h, EX_H + (size_t)x->hidden * 4, cudaMemcpyHostToDevice, gg_stream));
    const int64_t *pd = (const int64_t *)x->stage_d;
    const float *wd = (const float *)(x->stage_d + EX_W), *hd = (const float *)(x->stage_d + EX_H);
    ex_rows(x, pd, NULL, hd, n);
    k_ex_sum<<<(unsigned)cdiv(x->dn_rows, 256), 256, 0, gg_stream>>>(wd, x->de, n, x->dn_rows, x->out_d);
    CK(cudaMemcpyAsync(x->out_h, x->out_d, (size_t)x->dn_rows * 4, cudaMemcpyDeviceToHost, gg_stream));
    CK(cudaEventRecord(x->done, gg_stream));
    x->nout = x->dn_rows;
    x->pending = 1;
    return 0;
}

/* A group of t tokens of k pairs each (idx: the expert of each pair, or -1 for those this GPU does
 * not run; weights w) on their rows h (t x hidden): the kernels of a step, each pair on the row of
 * its token, then the sum of each token's pairs (t rows). -1 when a pair is not this GPU's (the
 * caller gives them all to the CPU). */
static int rc_gpu_ex_group_job(int layer, int t, int k, const int32_t *idx, const float *w, const float *h)
{
    ex_dev *x = &gg_ex;
    if (t <= 0 || t > EXG_TMAX || x->gu[layer] == NULL) {
        return -1;
    }
    int64_t *ptrs = (int64_t *)x->stage_h;
    int32_t *tok = (int32_t *)(x->stage_h + EXG_TOK);
    float *ws = (float *)(x->stage_h + EXG_W);
    int n = 0;
    for (int p = 0; p < t * k; ++p) {
        if (idx[p] < 0) {
            continue;
        }
        int s = x->slots[layer][idx[p]];
        if (s < 0 || n >= EXG_PMAX) {
            return -1;
        }
        ptrs[2 * n] = (int64_t)(intptr_t)(x->gu[layer] + (size_t)s * x->gu_per);
        ptrs[2 * n + 1] = (int64_t)(intptr_t)(x->dn[layer] + (size_t)s * x->dn_per);
        tok[n] = p / k;
        ws[n] = w[p];
        ++n;
    }
    if (n == 0) {
        return -1;
    }
    memcpy(x->stage_h + EXG_H, h, (size_t)t * x->hidden * 4);
    CK(cudaMemcpyAsync(x->stage_d, x->stage_h, EXG_H + (size_t)t * x->hidden * 4, cudaMemcpyHostToDevice,
                       gg_stream));
    const int64_t *pd = (const int64_t *)x->stage_d;
    const int32_t *td = (const int32_t *)(x->stage_d + EXG_TOK);
    const float *wd = (const float *)(x->stage_d + EXG_W), *hd = (const float *)(x->stage_d + EXG_H);
    ex_rows(x, pd, td, hd, n);
    k_ex_sum_t<<<dim3((unsigned)cdiv(x->dn_rows, 256), t), 256, 0, gg_stream>>>(wd, td, x->de, n, x->dn_rows,
                                                                                 x->out_d);
    CK(cudaGetLastError());
    x->nout = t * x->dn_rows;
    CK(cudaMemcpyAsync(x->out_h, x->out_d, (size_t)x->nout * 4, cudaMemcpyDeviceToHost, gg_stream));
    CK(cudaEventRecord(x->done, gg_stream));
    x->pending = 1;
    return 0;
}

/* Wait for the job of rc_gpu_ex_post (or of a group) and add its sum to out (dn_rows floats; a
 * group's t rows). */
static int rc_gpu_ex_wait_in(float *out)
{
    ex_dev *x = &gg_ex;
    ex_grp *gp = &gg_exg;
    if (gp->pending) {
        /* (a prompt group's share: its sums added to the CPU's) */
        const double t0 = ex_debug() ? ex_ms() : 0;
        cudaError_t we = cudaEventSynchronize(x->done);
        gp->pending = 0;
        if (we != cudaSuccess) {
            fprintf(stderr, "[ex] group wait: %s\n", cudaGetErrorString(we));
        }
        CK(we);
        for (int c = 0; c < gp->nout; ++c) {
            out[c] += gp->out_h[c];
        }
        if (ex_debug()) {
            fprintf(stderr, "[ex] group wait (%d rows): %.2f ms\n", gp->nout / x->L, ex_ms() - t0);
        }
        return 0;
    }
    if (!x->pending) {
        return 0;
    }
    double t0 = ex_debug() ? ex_ms() : 0;
    cudaError_t we;
    if (ex_debug() && x->nm && x->nout > x->L) {
        /* (a group: poll, and say where it waits) */
        double next = t0 + 1000;
        while ((we = cudaEventQuery(x->done)) == cudaErrorNotReady) {
            double now = ex_ms();
            if (now > next) {
                cudaError_t sq = cudaStreamQuery(gg_stream);
                fprintf(stderr, "[ex] wait (%d rows) %.0f ms: the event %s, the stream %s, tid %ld\n",
                        x->nout / x->L, now - t0, cudaGetErrorString(we), cudaGetErrorString(sq),
                        (long)syscall(SYS_gettid));
                next = now + 1000;
            }
        }
    } else {
        we = cudaEventSynchronize(x->done);
    }
    if (we != cudaSuccess || (ex_debug() && x->nm && x->nout > x->L)) {
        fprintf(stderr, "[ex] wait (%d rows): %.2f ms: %s\n", x->nout / (x->nm ? x->L : x->dn_rows), ex_debug() ? ex_ms() - t0 : 0.0,
                cudaGetErrorString(we));
    }
    CK(we);
    x->pending = 0;
    for (int c = 0; c < x->nout; ++c) {
        out[c] += x->out_h[c];
    }
    return 0;
}

extern "C" int rc_gpu_ex_init(int layers, int hidden, int inner, int dn_rows, size_t gu_per, size_t dn_per,
                              int q4x)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_init_in(layers, hidden, inner, dn_rows, gu_per, dn_per, q4x));
}

extern "C" int rc_gpu_ex_store(int layer, int count, const void *gu, const void *dn, const int32_t *slots, int experts)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_store_in(layer, count, gu, dn, slots, experts));
}

extern "C" int rc_gpu_ex_nm_init(int layers, int L, int kmax, int imax, int nv4)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_init_in(layers, L, kmax, imax, nv4));
}

extern "C" int rc_gpu_ex_nm_store(int layer, int count, const void *up, const void *dn, size_t up_per,
                                  size_t dn_per, int inner, const int32_t *slots, int experts)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_store_in(layer, count, up, dn, up_per, dn_per, inner, slots, experts));
}

extern "C" int rc_gpu_ex_nm_hosts(int layer, const void *up, const void *dn, int xmax)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_hosts_in(layer, up, dn, xmax));
}

extern "C" int rc_gpu_ex_put(int layer, int n, const int32_t *slot, const int64_t *src, const size_t *per,
                             const int32_t *slots, int experts)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_put_in(layer, n, slot, src, per, slots, experts));
}

/* er_ctx post_t (GP_NM_REMOTE_POST: a group of t tokens): a Nemotron job, or a Gemma one */
extern "C" int rc_gpu_ex_post_t(int layer, int t, int k, const int32_t *idx, const float *w, const float *x)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, gg_ex.nm ? rc_gpu_ex_nm_job(layer, t, k, idx, w, x)
                         : gg_ex.kq ? rc_gpu_ex_kq_job(layer, t, k, idx, w, x)
                         : rc_gpu_ex_group_job(layer, t, k, idx, w, x));
}

extern "C" int rc_gpu_ex_nm_post_t(int layer, int t, int k, const int32_t *idx, const float *w, const float *x)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_job(layer, t, k, idx, w, x));
}

/* er_ctx post_g (GP_NM_REMOTE_POST's group form): a share of a prompt group's experts */
extern "C" int rc_gpu_ex_nm_gpost(int layer, int t, int k, const int32_t *idx, const float *w, const float *x, int mc,
                                  int win, int x16)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_gpost_in(layer, t, k, idx, w, x, mc, win, x16));
}

extern "C" int rc_gpu_ex_nm_dstage(int nmax, uint8_t *stg, int em)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_dstage_in(nmax, stg, em));
}

extern "C" int rc_gpu_ex_nm_prefetch(int layer, int n, const int32_t *ids)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_prefetch_in(layer, n, ids));
}

extern "C" int rc_gpu_ex_nm_settle(int layer)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_settle_in(layer));
}

/* the decode stage's counts: the experts copied ahead, and read by a job */
extern "C" void rc_gpu_ex_nm_dstats(int64_t *out)
{
    out[0] = gg_exd.copied;
    out[1] = gg_exd.used;
    out[2] = gg_exd.late;
}

/* ---------- Super's steps on the primary: its share of the predicted experts copied ahead (models/nemotron/gpu.py PF3)
 * The CPU's task of a MoE layer, after its pairs (rc_gpu_pf3_issue, er_ctx pf3): the next MoE layer's predicted cold
 * experts that the expert device does not take, copied by DMA on this GPU's own stream into the next of two stage
 * buffers, from the host's stacks (the keeper's pages, registered); meta (pinned, the step's NM_STAGED reads it): each
 * buffer's layer, seq, n, its experts and a flag each, set to seq by a one-thread kernel after that expert's copy on
 * the stream (a host function's flag came late: its thread beside the CPU's team) */
#define PF3_MAX 16
typedef struct {
    int dev, K, next, layers;
    int32_t seq;
    cudaStream_t s;
    int32_t *meta[2];
    uint8_t *stage[2];
    size_t per;
    const uint8_t **hup, **hdn;
    size_t *upp, *dnp;
    int64_t copied;
} pf3_t;
static pf3_t gg_pf3;

__global__ void k_pf3_flag(volatile int32_t *flag, int32_t val)
{
    __threadfence_system();
    *flag = val;
    __threadfence_system();
}

extern "C" int rc_gpu_pf3_init(int K, int32_t *meta0, int32_t *meta1, void *stage0, void *stage1, size_t per,
                               int layers)
{
    pf3_t *f = &gg_pf3;
    if (K <= 0 || K > PF3_MAX || meta0 == NULL || meta1 == NULL || stage0 == NULL || stage1 == NULL || layers <= 0) {
        return -1;
    }
    CK(cudaGetDevice(&f->dev));
    if (f->s == NULL) {
        CK(cudaStreamCreateWithFlags(&f->s, cudaStreamNonBlocking));
    }
    if (f->hup == NULL) {
        f->hup = (const uint8_t **)calloc((size_t)layers, sizeof(uint8_t *));
        f->hdn = (const uint8_t **)calloc((size_t)layers, sizeof(uint8_t *));
        f->upp = (size_t *)calloc((size_t)layers, sizeof(size_t));
        f->dnp = (size_t *)calloc((size_t)layers, sizeof(size_t));
        f->layers = layers;
    } else if (layers > f->layers) {
        return -1;
    }
    f->K = K;
    f->meta[0] = meta0;
    f->meta[1] = meta1;
    f->stage[0] = (uint8_t *)stage0;
    f->stage[1] = (uint8_t *)stage1;
    f->per = per;
    for (int b = 0; b < 2; ++b) {
        ((volatile int32_t *)f->meta[b])[0] = -1;
    }
    return 0;
}

extern "C" int rc_gpu_pf3_hosts(int layer, const void *up, const void *dn, size_t upp, size_t dnp)
{
    pf3_t *f = &gg_pf3;
    if (f->hup == NULL || layer < 0 || layer >= f->layers || upp + dnp > f->per || upp % 16 != 0) {
        return -1;
    }
    f->hup[layer] = (const uint8_t *)up;
    f->hdn[layer] = (const uint8_t *)dn;
    f->upp[layer] = upp;
    f->dnp[layer] = dnp;
    return 0;
}

extern "C" int rc_gpu_pf3_issue(int layer, int n, const int32_t *ids)
{
    pf3_t *f = &gg_pf3;
    if (f->s == NULL || layer < 0 || layer >= f->layers || f->hup[layer] == NULL || n <= 0) {
        return -1;
    }
    const size_t upp = f->upp[layer], dnp = f->dnp[layer];
    int prev = -1;
    cudaGetDevice(&prev);
    if (prev != f->dev) {
        cudaSetDevice(f->dev);
    }
    const int b = f->next;
    f->next ^= 1;
    int32_t *m = f->meta[b];
    volatile int32_t *vm = (volatile int32_t *)m;
    const int K = f->K;
    vm[0] = -1;
    __sync_synchronize();
    const int32_t seq = ++f->seq;
    n = n < K ? n : K;
    for (int q = 0; q < n; ++q) {
        vm[3 + q] = ids[q];
    }
    vm[2] = n;
    vm[1] = seq;
    __sync_synchronize();
    vm[0] = layer;
    int rc = 0;
    for (int q = 0; q < n && rc == 0; ++q) {
        uint8_t *dst = f->stage[b] + (size_t)q * f->per;
        const uint8_t *su = f->hup[layer] + (size_t)ids[q] * upp, *sd = f->hdn[layer] + (size_t)ids[q] * dnp;
        if (cudaMemcpyAsync(dst, su, upp, cudaMemcpyHostToDevice, f->s) != cudaSuccess ||
            cudaMemcpyAsync(dst + upp, sd, dnp, cudaMemcpyHostToDevice, f->s) != cudaSuccess) {
            cudaGetLastError();
            rc = -1;
            break;
        }
        k_pf3_flag<<<1, 1, 0, f->s>>>(vm + 3 + K + q, seq);
        if (cudaGetLastError() != cudaSuccess) {
            rc = -1;
        }
        ++f->copied;
    }
    if (prev >= 0 && prev != f->dev) {
        cudaSetDevice(prev);
    }
    return rc;
}

/* the primary's copies ahead: those issued */
extern "C" int64_t rc_gpu_pf3_copied(void)
{
    return gg_pf3.copied;
}

extern "C" int rc_gpu_ex_nm_gmask(int layer, int n, const int32_t *experts)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_gmask_in(layer, n, experts));
}

extern "C" int rc_gpu_ex_nm_ghosts(int layer, const void *up, const void *dn, int experts, int inner, size_t up_per,
                                   size_t dn_per)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_ghosts_in(layer, up, dn, experts, inner, up_per, dn_per));
}

extern "C" int rc_gpu_ex_kq_init(int layers, int hidden, int inner, int kmax)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_kq_init_in(layers, hidden, inner, kmax));
}

extern "C" int rc_gpu_ex_kq_store(int layer, int count, const void *gate, const void *up, const void *down,
                                  size_t g_per, size_t u_per, size_t d_per, int gtype, int dtype,
                                  const int32_t *slots, int experts)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_kq_store_in(layer, count, gate, up, down, g_per, u_per, d_per, gtype, dtype,
                                               slots, experts));
}

extern "C" int rc_gpu_ex_post(int layer, int n, const int32_t *ids, const float *w, const float *h)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_post_in(layer, n, ids, w, h));
}

extern "C" int rc_gpu_ex_wait(float *out)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_wait_in(out));
}
