/* ---------- an expert device (reclank/backends/cuda/devices.py) ----------
 * A second GPU that holds experts the GPU of the steps does not. The process
 * loads a second copy of this library for it (its own state: its device, its
 * stream), uploads its experts once (rc_gpu_ex_store), and the CPU part of
 * each step calls rc_gpu_ex_post and rc_gpu_ex_wait through function pointers
 * (GP_REMOTE_POST, GP_REMOTE_JOIN: backends/cpu/csrc/common/ops_qwen.c): the
 * cold experts it holds run here while the CPU runs the others. The int4
 * experts with the kernels of the hot experts (hot_rows_xr: the same rows),
 * a pair's weights by pointer.
 *
 * Qwen (rc_gpu_ex_kq_init): the K-quant experts of each layer by slot (gate, up, down), a job a
 * GP_KQ_HOT_MOE record of its tokens with no shared expert, through gg_launch with a program of
 * zeros (no int8 scratch: the float x kernels, k_kqh_k / k_kqh_gu / k_kqh_dn, read the record only).
 *
 * Nemotron (rc_gpu_ex_nm_init; GP_NM_REMOTE_POST): the IQ2_XXS experts of
 * each layer by slot, as NemotronGPU's hot stacks; a job is a GP_NM_HOT_MOE
 * record of one token (its pairs the experts posted) through gg_launch, the
 * kernels of the primary's hot experts. */
#define EX_MAXK 16
#define EX_W 256                 /* the staging: [pairs][2] pointers, then w at 256, then h at 320 */
#define EX_H 320
#define EXG_TMAX 16              /* a group (an MTP verify group, MT_CPU rows): at most 16 tokens */
#define EXG_PMAX 128             /* and 128 pairs; its staging: the pointers, each pair's token, w, h */
#define EXG_TOK (EXG_PMAX * 16)
#define EXG_W (EXG_TOK + EXG_PMAX * 4)
#define EXG_H (EXG_W + EXG_PMAX * 4)

typedef struct {
    int dev, layers, hidden, inner, dn_rows;
    uint8_t **gu, **dn;          /* the experts of each layer on this GPU (or null) */
    int32_t **slots;             /* each layer: expert -> its place in gu and dn, or -1 (host) */
    size_t gu_per, dn_per;
    uint8_t *stage_h, *stage_d;  /* pinned and device: pointers, w, h */
    float *act, *act2, *de, *out_d, *out_h;
    cudaEvent_t done;
    int pending;
    /* Nemotron: the latent L, the most pairs and inner of a job, each layer's inner and its
     * slots on this GPU (device), the record of a job and the int8 planes of its x and act */
    int nm, L, kmax, imax, nout;
    int *inner_l, *experts_l;
    /* Qwen: the up blocks of each layer and the types of its gate (and up) and down */
    int kq;
    uint8_t **up;
    int *gtype_l, *dtype_l;
    int32_t **slots_d;
    gp_rec *rec_d;
    int64_t *env_d;
    int8_t *qx, *qa, *qxl, *qal;
    float *sx, *sa;
} ex_dev;
#define EXN_TMAX 8               /* a Nemotron job: at most 8 tokens (a verify group), 128 pairs */
#define EXN_TK 128
#define EXN_IDS 256              /* its staging: the record, idx, w, x */
#define EXN_VAL (EXN_IDS + 4 * EXN_TK)
#define EXN_X (EXN_VAL + 4 * EXN_TK)

static ex_dev gg_ex;

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

/* The shapes and the buffers (after rc_gpu_init of this copy's device). */
static int rc_gpu_ex_init_in(int layers, int hidden, int inner, int dn_rows, size_t gu_per, size_t dn_per)
{
    ex_dev *x = &gg_ex;
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
static int rc_gpu_ex_nm_init_in(int layers, int L, int kmax, int imax)
{
    ex_dev *x = &gg_ex;
    if (kmax > NM_KMAX || L % 256 != 0 || imax % 256 != 0) {
        return -1;
    }
    x->nm = 1;
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
    if (inner > x->imax || inner % 256 != 0) {
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
    return 0;
}

/* Nemotron: t tokens of k pairs each (idx: the expert of each pair, or -1 for those this GPU does
 * not run; weights w) on their latent rows h: one GP_NM_HOT_MOE record (int8 x and act in two
 * planes, as the primary's NM_HOT_MOE; t 2-8 by expert, k_nm_hg_*) */
static int rc_gpu_ex_nm_job(int layer, int t, int k, const int32_t *idx, const float *w, const float *h)
{
    ex_dev *x = &gg_ex;
    if (t <= 0 || t > EXN_TMAX || k <= 0 || k > x->kmax || t * k > EXN_TK || x->gu[layer] == NULL) {
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
    rec.op = GP_NM_HOT_MOE;
    uint8_t *sd = x->stage_d;
    int64_t v[19] = {(int64_t)(intptr_t)(sd + EXN_X), (int64_t)(intptr_t)(sd + EXN_IDS),
                     (int64_t)(intptr_t)(sd + EXN_VAL), (int64_t)(intptr_t)x->slots_d[layer],
                     (int64_t)(intptr_t)x->gu[layer], (int64_t)(intptr_t)x->dn[layer],
                     (int64_t)(intptr_t)x->act, (int64_t)(intptr_t)x->de, (int64_t)(intptr_t)x->out_d,
                     k, x->L, x->inner_l[layer], t,
                     (int64_t)(intptr_t)x->qx, (int64_t)(intptr_t)x->sx, (int64_t)(intptr_t)x->qa,
                     (int64_t)(intptr_t)x->sa, (int64_t)(intptr_t)x->qxl, (int64_t)(intptr_t)x->qal};
    for (int i = 0; i < 19; ++i) {
        rec.tag[i] = GP_T_INT;
        rec.v[i] = v[i];
    }
    memcpy(x->stage_h, &rec, sizeof(rec));
    memcpy(x->stage_h + EXN_VAL, w, (size_t)t * k * 4);
    memcpy(x->stage_h + EXN_X, h, (size_t)t * x->L * 4);
    CK(cudaMemcpyAsync(x->stage_d, x->stage_h, EXN_X + (size_t)t * x->L * 4, cudaMemcpyHostToDevice, gg_stream));
    if (gg_launch(NULL, &rec, x->rec_d, x->env_d) != 0) {
        fprintf(stderr, "[ex] layer %d t %d k %d: gg_launch refused the record\n", layer, t, k);
        return -1;
    }
    cudaError_t le = cudaGetLastError();
    if (le != cudaSuccess) {
        fprintf(stderr, "[ex] layer %d t %d k %d: launch: %s\n", layer, t, k, cudaGetErrorString(le));
    }
    CK(le);
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
    int gu_rows = 2 * x->inner;
    k_ex_gu<<<dim3((unsigned)cdiv(gu_rows, ROWS_PER_BLOCK * HOT_XR), n), 32 * ROWS_PER_BLOCK, (size_t)x->hidden * 4,
              gg_stream>>>(pd, hd, gu_rows, x->hidden, x->act);
    k_ex_gelu<<<n, 256, 0, gg_stream>>>(x->act, x->inner, x->act2);
    k_ex_dn<<<dim3((unsigned)cdiv(x->dn_rows, ROWS_PER_BLOCK * HOT_XR), n), 32 * ROWS_PER_BLOCK, (size_t)x->inner * 4,
              gg_stream>>>(pd, x->act2, x->dn_rows, x->inner, x->de);
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
    int gu_rows = 2 * x->inner;
    k_ex_gu_t<<<dim3((unsigned)cdiv(gu_rows, ROWS_PER_BLOCK * HOT_XR), n), 32 * ROWS_PER_BLOCK,
                (size_t)x->hidden * 4, gg_stream>>>(pd, td, hd, gu_rows, x->hidden, x->act);
    k_ex_gelu<<<n, 256, 0, gg_stream>>>(x->act, x->inner, x->act2);
    k_ex_dn<<<dim3((unsigned)cdiv(x->dn_rows, ROWS_PER_BLOCK * HOT_XR), n), 32 * ROWS_PER_BLOCK, (size_t)x->inner * 4,
              gg_stream>>>(pd, x->act2, x->dn_rows, x->inner, x->de);
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

extern "C" int rc_gpu_ex_init(int layers, int hidden, int inner, int dn_rows, size_t gu_per, size_t dn_per)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_init_in(layers, hidden, inner, dn_rows, gu_per, dn_per));
}

extern "C" int rc_gpu_ex_store(int layer, int count, const void *gu, const void *dn, const int32_t *slots, int experts)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_store_in(layer, count, gu, dn, slots, experts));
}

extern "C" int rc_gpu_ex_nm_init(int layers, int L, int kmax, int imax)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_init_in(layers, L, kmax, imax));
}

extern "C" int rc_gpu_ex_nm_store(int layer, int count, const void *up, const void *dn, size_t up_per,
                                  size_t dn_per, int inner, const int32_t *slots, int experts)
{
    void *ctx = ex_enter();
    return ex_leave(ctx, rc_gpu_ex_nm_store_in(layer, count, up, dn, up_per, dn_per, inner, slots, experts));
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
