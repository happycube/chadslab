/* ---------- host memory that the GPU reads (zero copy) ----------
 * The read-only map of the model file cannot be registered with CUDA here
 * (cudaHostRegisterReadOnly: "operation not supported"); a kernel reads
 * registered host memory at 11 to 12 GB/s over the PCIe 3.0 x16 of the
 * 3090. */
/* The host memory registered with CUDA (docs/GPU_SWAP_PLAN.md 3.6): a context's reset drops every
 * registration, and the memory itself stays (an anonymous map: rc_gpu_host_alloc; a weightkeeper's
 * segment: rc_gpu_host_register_*), so the list registers it again in the next context
 * (gg_reg_again). own: the library mapped it (rc_gpu_host_alloc, freed by rc_gpu_host_free). */
typedef struct gg_reg {
    void *p;
    size_t n;
    unsigned flags;
    int own;
    struct gg_reg *next;
} gg_reg;
static gg_reg *gg_regs;
static pthread_mutex_t gg_reg_mu = PTHREAD_MUTEX_INITIALIZER;
static int gg_reg_ep = 1;           /* the epoch of the registrations */
static double gg_reg_secs;          /* the time of the last gg_reg_again */

static void gg_reg_add(void *p, size_t n, unsigned flags, int own)
{
    gg_reg *r = (gg_reg *)calloc(1, sizeof(gg_reg));
    r->p = p;
    r->n = n;
    r->flags = flags;
    r->own = own;
    pthread_mutex_lock(&gg_reg_mu);
    r->next = gg_regs;
    gg_regs = r;
    pthread_mutex_unlock(&gg_reg_mu);
}

/* Forget the registration at p; its bytes (0: none) to *n. */
static void gg_reg_del(void *p, size_t *n)
{
    pthread_mutex_lock(&gg_reg_mu);
    for (gg_reg **at = &gg_regs; *at != NULL; at = &(*at)->next) {
        if ((*at)->p == p) {
            gg_reg *r = *at;
            *at = r->next;
            if (n != NULL) {
                *n = r->n;
            }
            free(r);
            break;
        }
    }
    pthread_mutex_unlock(&gg_reg_mu);
}

/* In a new context: every registration of the list again (as it was made). */
static int gg_reg_again(void)
{
    if (gg_reg_ep == gg_epoch) {
        return 0;
    }
    double t0 = gg_clock();
    pthread_mutex_lock(&gg_reg_mu);
    for (gg_reg *r = gg_regs; r != NULL; r = r->next) {
        cudaError_t err = cudaHostRegister(r->p, r->n, r->flags);
        if (err != cudaSuccess) {
            pthread_mutex_unlock(&gg_reg_mu);
            snprintf(gg_error, sizeof(gg_error), "cudaHostRegister again of %zu bytes: %s", r->n,
                     cudaGetErrorString(err));
            cudaGetLastError();
            return -1;
        }
    }
    gg_reg_ep = gg_epoch;
    pthread_mutex_unlock(&gg_reg_mu);
    gg_reg_secs = gg_clock() - t0;
    return 0;
}

/* Map n bytes of memory of the process at p for the GPU (an anonymous copy
 * of the experts, QwenGPU._pin_experts). Return the device address, or 0. */
extern "C" int64_t rc_gpu_host_register_rw(void *p, size_t n)
{
    cudaError_t err = cudaHostRegister(p, n, cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "cudaHostRegister of %zu bytes: %s", n,
                 cudaGetErrorString(err));
        cudaGetLastError();
        return 0;
    }
    gg_reg_add(p, n, cudaHostRegisterMapped | cudaHostRegisterPortable, 0);
    void *d = NULL;
    if (cudaHostGetDevicePointer(&d, p, 0) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return (int64_t)(intptr_t)d;
}

/* Page-lock n bytes of a read-only map at p for the copies to the GPU (the
 * segments of the weightkeeper: a sealed memfd, mapped PROT_READ), so that they
 * are DMA, not a memcpy through a staging buffer. 0, or -1 (the copies then
 * stage). */
extern "C" int rc_gpu_host_register_ro(void *p, size_t n)
{
    cudaError_t err = cudaHostRegister(p, n, cudaHostRegisterReadOnly | cudaHostRegisterPortable);
    if (err != cudaSuccess) {
        snprintf(gg_error, sizeof(gg_error), "cudaHostRegister (read only) of %zu bytes: %s", n,
                 cudaGetErrorString(err));
        cudaGetLastError();
        return -1;
    }
    gg_reg_add(p, n, cudaHostRegisterReadOnly | cudaHostRegisterPortable, 0);
    return 0;
}

/* The device address of host memory at p that CUDA has page-locked (registered: read by kernels over the
 * link), or 0. */
extern "C" int64_t rc_gpu_dev_addr(const void *p)
{
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, p) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return a.type == cudaMemoryTypeHost ? (int64_t)(intptr_t)a.devicePointer : 0;
}

/* The PCI bus id of the device ("0000:9e:00.0"), for its NUMA node. */
extern "C" int rc_gpu_pci_bus_id(char *out, int n)
{
    int dev = 0;
    cudaGetDevice(&dev);
    return cudaDeviceGetPCIBusId(out, n, dev) == cudaSuccess ? 0 : -1;
}

extern "C" int rc_gpu_host_unregister(void *p)
{
    gg_reg_del(p, NULL);
    return cudaHostUnregister(p) == cudaSuccess ? 0 : -1;
}

/* Queue a job: its copies, then the packs into Q4S (gg_cjob). Return its id (0 or more), or -1. */
extern "C" int rc_gpu_cache_copy_q4s(const int64_t *ranges, int count, const int64_t *packs, int npacks)
{
    if (gg_cache_ep != gg_epoch) {
        CK(cudaStreamCreateWithFlags(&gg_cstream, cudaStreamNonBlocking));
        for (int k = 0; k < GG_CACHE_JOBS; ++k) {
            CK(cudaEventCreateWithFlags(&gg_cev[k], cudaEventDisableTiming));
        }
        gg_cache_ep = gg_epoch;
    }
    if (!gg_cworker_on) {
        pthread_t th;
        if (pthread_create(&th, NULL, gg_cworker, NULL) != 0) {
            snprintf(gg_error, sizeof(gg_error), "no worker thread for the cache copies");
            return -1;
        }
        pthread_detach(th);
        gg_cworker_on = 1;
    }
    pthread_mutex_lock(&gg_cmu);
    if (gg_cjob_tail - gg_cjob_head >= GG_CACHE_JOBS) {
        pthread_mutex_unlock(&gg_cmu);
        snprintf(gg_error, sizeof(gg_error), "too many cache copies in the queue");
        return -1;
    }
    int id = gg_cjob_next++;
    gg_cjob j = {ranges, count, id, packs, npacks};
    gg_cissued[id % GG_CACHE_JOBS] = 0;
    gg_cid[id % GG_CACHE_JOBS] = id;
    gg_cjobs[gg_cjob_tail % GG_CACHE_JOBS] = j;
    ++gg_cjob_tail;
    pthread_cond_broadcast(&gg_ccv);
    pthread_mutex_unlock(&gg_cmu);
    return id;
}

/* Queue a job of copies. Return its id (0 or more), or -1. */
extern "C" int rc_gpu_cache_copy(const int64_t *ranges, int count)
{
    return rc_gpu_cache_copy_q4s(ranges, count, NULL, 0);
}

/* 1: the copies of job id are on the GPU. 0: not yet. -1: an error. With
 * wait, wait for the job. */
extern "C" int rc_gpu_cache_query(int id, int wait)
{
    int slot = id % GG_CACHE_JOBS;
    pthread_mutex_lock(&gg_cmu);
    while (wait && !gg_cissued[slot]) {
        pthread_cond_wait(&gg_ccv, &gg_cmu);
    }
    int issued = gg_cissued[slot] && gg_cid[slot] == id, bad = gg_cerror;
    pthread_mutex_unlock(&gg_cmu);
    if (bad) {
        snprintf(gg_error, sizeof(gg_error), "a cache copy failed");
        return -1;
    }
    if (!issued) {
        return 0;
    }
    cudaError_t q = wait ? cudaEventSynchronize(gg_cev[slot]) : cudaEventQuery(gg_cev[slot]);
    if (q == cudaErrorNotReady) {
        return 0;
    }
    CK(q);
    return 1;
}

static int gg_fetch_init(void)
{
    if (gg_worker_on && gg_fetch_ep == gg_epoch) {
        return 0;
    }
    if (gg_fetch_ep != gg_epoch) {
        /* the highest priority: the blocks of k_fetch_pull go before those of
         * the program's kernels */
        int lo = 0, hi_ = 0;
        CK(cudaDeviceGetStreamPriorityRange(&lo, &hi_));
        CK(cudaStreamCreateWithPriority(&gg_copy, cudaStreamNonBlocking, hi_));
        for (int k = 0; k < GG_FETCH_MAX; ++k) {
            CK(cudaEventCreateWithFlags(&gg_ready[k], cudaEventDisableTiming));
        }
        for (int k = 0; k < 2; ++k) {
            CK(cudaEventCreateWithFlags(&gg_freeev[k], cudaEventDisableTiming));
            CK(cudaEventRecord(gg_freeev[k], gg_stream));
        }
        gg_fetch_ep = gg_epoch;
    }
    if (gg_worker_on) {
        return 0;
    }
    pthread_t th;
    if (pthread_create(&th, NULL, gg_worker, NULL) != 0) {
        snprintf(gg_error, sizeof(gg_error), "no worker thread for the copies");
        return -1;
    }
    pthread_detach(th);
    gg_worker_on = 1;
    return 0;
}

static int64_t hi(const gp_rec *r, const int64_t *e, int k)
{
    return r->tag[k] == GP_T_SLOT ? e[r->v[k]] : r->v[k];
}
