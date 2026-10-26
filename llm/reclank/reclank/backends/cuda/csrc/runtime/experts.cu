/* ---------- the copy of the weights of the experts ----------
 * The operands of GP_FETCH are ranges, count, fetch f, and buffer b. ranges is a host array of
 * count triples (host address, device address, bytes). A worker thread
 * copies the ranges to the GPU on a stream of its own, and records the event
 * ready[f]. The host memory is not pinned: it is the map of the model file.
 * Thus the copy call waits until the data is in the buffers of the driver. The worker thread does that
 * wait, and the runner goes on with the launches. Before the copy, the
 * worker waits for the event free[b]: the last use of buffer b is done.
 *
 * GP_FETCH_WAIT f: the stream of the program waits for ready[f].
 * GP_FETCH_DONE b: the stream of the program records free[b]. */
#define GG_FETCH_MAX 256

typedef struct {
    const int64_t *ranges;
    int count;
    int f, b;
    int mode, c0, c1;       /* mode 1: KQ_NV4 rows into NV4T (k_nm_tile_pull; the columns of even / odd ranges) */
} gg_job;

static pthread_mutex_t gg_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gg_cv = PTHREAD_COND_INITIALIZER;
static gg_job gg_jobs[GG_FETCH_MAX];
static int gg_job_head, gg_job_tail;
static int gg_recorded[GG_FETCH_MAX];
static int gg_fetch_error;
static char gg_fetch_why[160];     /* the CUDA error of the first failed copy */
static cudaEvent_t gg_ready[GG_FETCH_MAX], gg_freeev[2];
static cudaStream_t gg_copy;
static int gg_worker_on;

/* ---------- the measures of the mixed groups (rc_gpu_mix_stats) ----------
 * While they are on, a run keeps: the bytes and the time of each copy job
 * of GP_FETCH (host clock, after the job is on the stream); the time that
 * the stream waits in each GP_FETCH_WAIT, and in each GP_CPU_JOIN (timing
 * events on the stream: from the end of the work before to the next
 * record); and the time of each CPU program of GP_CPU_JOIN. ModelGPU.
 * plan_mix fits its model of the time to them. */
#define GG_MS_MAX 256
static int gg_ms_on;
static double gg_ms_copy_bytes, gg_ms_copy_s;
static int gg_ms_nf, gg_ms_nj;
static cudaEvent_t gg_ms_ev[4 * GG_MS_MAX];
static int gg_ms_ev_ok;
static double gg_ms_cpu[GG_MS_MAX];
static int gg_ms_kind[GG_MS_MAX];      /* 0: GP_FETCH_WAIT, 1: GP_CPU_JOIN */

/* on 1: clear the measures and keep them from now on. on 0: stop. */
extern "C" int rc_gpu_mix_stats_on(int on)
{
    if (on && !gg_ms_ev_ok) {
        for (int k = 0; k < 4 * GG_MS_MAX; ++k) {
            CK(cudaEventCreate(&gg_ms_ev[k]));
        }
        gg_ms_ev_ok = 1;
    }
    pthread_mutex_lock(&gg_mu);
    gg_ms_copy_bytes = gg_ms_copy_s = 0.0;
    pthread_mutex_unlock(&gg_mu);
    gg_ms_nf = gg_ms_nj = 0;
    gg_ms_on = on;
    return 0;
}

/* After a run: out gets the copy bytes and seconds, the counts nf and nj,
 * then for each wait in the order of the run its kind (0: GP_FETCH_WAIT,
 * 1: GP_CPU_JOIN), its time, and the time of the work of the stream between
 * the wait before and this one (0 for the first), then the nj CPU times
 * (times in seconds). Return the count of values, or -1. */
extern "C" int rc_gpu_mix_stats(double *out, int max)
{
    if (4 + 3 * gg_ms_nf + 4 * gg_ms_nj > max) {
        snprintf(gg_error, sizeof(gg_error), "rc_gpu_mix_stats: %d values do not fit",
                 4 + 3 * gg_ms_nf + 4 * gg_ms_nj);
        return -1;
    }
    CK(cudaStreamSynchronize(gg_stream));
    int n = 0;
    pthread_mutex_lock(&gg_mu);
    out[n++] = gg_ms_copy_bytes;
    out[n++] = gg_ms_copy_s;
    pthread_mutex_unlock(&gg_mu);
    out[n++] = gg_ms_nf;
    out[n++] = gg_ms_nj;
    for (int k = 0; k < gg_ms_nf + gg_ms_nj; ++k) {
        float ms = 0.f;
        CK(cudaEventElapsedTime(&ms, gg_ms_ev[2 * k], gg_ms_ev[2 * k + 1]));
        out[n++] = gg_ms_kind[k];
        out[n++] = 1e-3 * ms;
        float gap = 0.f;
        if (k > 0) {
            CK(cudaEventElapsedTime(&gap, gg_ms_ev[2 * k - 1], gg_ms_ev[2 * k]));
        }
        out[n++] = 1e-3 * gap;
    }
    for (int k = 0; k < gg_ms_nj; ++k) {
        out[n++] = gg_ms_cpu[k];
    }
    return n;
}

/* ---------- copies from pageable host memory (the map of the model file) ----------
 * A cudaMemcpyAsync from pageable memory goes through a staging buffer of
 * the driver, and the call holds the driver while it copies: on the 2-socket
 * Xeon the copies of HotCache (2.8 MB experts of Qwen3.8 in the decode)
 * held back the steps, 32 tok/s against 36 with no copies. A stager copies
 * chunks of GG_STAGE bytes into two pinned buffers of its own (memcpy on
 * the thread of the worker) and gives each chunk to the stream as a copy
 * from pinned memory (DMA). RECLANK_GPU_STAGE=0 keeps the plain call.
 * RECLANK_GPU_COPY_CPU=c keeps the workers on CPU c (by default the last
 * CPU of the node of the GPU among the CPUs of the process at its start,
 * RECLANK_START_CPUS of reclank: the balanced teams of gemma_run_task leave
 * the last CPU of a node free when they take fewer than its CPUs; the
 * mixed groups take OMP_NUM_THREADS - 2. -1: no binding). */
#define GG_STAGE (4u << 20)
typedef struct {
    uint8_t *buf[2];
    cudaEvent_t ev[2];
    int cur, init;
} gg_stager;

static int gg_stage_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_STAGE");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* The CPUs of the node of the GPU (local_cpulist) among those of the
 * process at its start (RECLANK_START_CPUS of reclank; all the node's when
 * unset), ascending, in out (at most max); the count, or -1. */
static int gg_node_cpus(int *out, int max)
{
    int dev = 0;
    cudaGetDevice(&dev);
    char bus[32], path[128];
    if (cudaDeviceGetPCIBusId(bus, sizeof(bus), dev) != cudaSuccess) {
        return -1;
    }
    for (char *q = bus; *q; ++q) {
        if (*q >= 'A' && *q <= 'F') {
            *q = (char)(*q - 'A' + 'a');
        }
    }
    snprintf(path, sizeof(path), "/sys/bus/pci/devices/%s/local_cpulist", bus);
    FILE *f = fopen(path, "r");
    if (f == NULL) {
        return -1;
    }
    char line[256] = {0};
    if (fgets(line, sizeof(line), f) == NULL) {
        line[0] = 0;
    }
    fclose(f);
    static unsigned char node[4096], start[4096];
    memset(node, 0, sizeof(node));
    memset(start, 0, sizeof(start));
    for (char *q = line; *q && *q != '\n';) {
        int a = (int)strtol(q, &q, 10), b = a;
        if (*q == '-') {
            b = (int)strtol(q + 1, &q, 10);
        }
        for (int k = a; k <= b && k < 4096; ++k) {
            if (k >= 0) {
                node[k] = 1;
            }
        }
        if (*q == ',') {
            ++q;
        } else {
            break;
        }
    }
    const char *sc = getenv("RECLANK_START_CPUS");
    int any = 0;
    for (const char *q = sc; q != NULL && *q;) {
        char *e;
        int k = (int)strtol(q, &e, 10);
        if (e == q) {
            break;
        }
        if (k >= 0 && k < 4096) {
            start[k] = 1;
            any = 1;
        }
        if (*e != ',') {
            break;
        }
        q = e + 1;
    }
    int n = 0;
    for (int k = 0; k < 4096 && n < max; ++k) {
        if (node[k] && (start[k] || !any)) {
            out[n++] = k;
        }
    }
    return n;
}

/* The CPUs of the threads around the teams (gemma_run_task) on the node of
 * the GPU: the copy workers the last of gg_node_cpus (RECLANK_GPU_COPY_CPU
 * when set), the thread of the model (the thread that loads this library)
 * the first; -1 for none. reclank.backends.cuda.runtime pins the model thread and keeps both
 * out of the teams (RECLANK_RESERVED_CPUS); the master of the teams (the GPU
 * runner) takes the first CPU of their plan. */
extern "C" void rc_gpu_cpu_roles(int *copy_cpu, int *main_cpu)
{
    int cpus[1024];
    int n = gg_node_cpus(cpus, 1024);
    const char *v = getenv("RECLANK_GPU_COPY_CPU");
    *copy_cpu = v ? atoi(v) : n > 0 ? cpus[n - 1] : -1;
    *main_cpu = n >= 3 ? cpus[0] : -1;
}

/* Keep the calling worker thread on CPU RECLANK_GPU_COPY_CPU (see above). */
static void gg_worker_bind(void)
{
    int c, m;
    rc_gpu_cpu_roles(&c, &m);
    if (c < 0) {
        return;
    }
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(c, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
}

/* 1 when src is registered (page-locked) host memory: a copy from it is
 * DMA with no staging (QwenGPU registers its copies of the experts). */
static int gg_is_pinned(const void *src)
{
    cudaPointerAttributes a;
    if (cudaPointerGetAttributes(&a, src) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return a.type == cudaMemoryTypeHost;
}

/* Copy n bytes from host src (pageable, or registered) to device dst on
 * stream st. direct: a registered src goes as DMA with no staging (the
 * copies of a prompt group). The copies of HotCache in the decode stay
 * staged: DMA at the full rate of the bus delayed the small transfers of
 * the steps (Qwen3.8 MTP 66.7 -> 58.8 tok/s). */
static int gg_h2d_staged(gg_stager *sg, void *dst, const void *src, size_t n, cudaStream_t st,
                         int direct = 0)
{
    if (!gg_stage_on() || (direct && gg_is_pinned(src))) {
        return cudaMemcpyAsync(dst, src, n, cudaMemcpyHostToDevice, st) == cudaSuccess ? 0 : -1;
    }
    if (!sg->init) {
        for (int b = 0; b < 2; ++b) {
            if (cudaMallocHost(&sg->buf[b], GG_STAGE) != cudaSuccess ||
                cudaEventCreateWithFlags(&sg->ev[b], cudaEventDisableTiming) != cudaSuccess ||
                cudaEventRecord(sg->ev[b], st) != cudaSuccess) {
                return -1;
            }
        }
        sg->init = 1;
    }
    for (size_t off = 0; off < n; off += GG_STAGE) {
        size_t len = n - off < GG_STAGE ? n - off : GG_STAGE;
        int b = sg->cur;
        if (cudaEventSynchronize(sg->ev[b]) != cudaSuccess) {
            return -1;
        }
        memcpy(sg->buf[b], (const uint8_t *)src + off, len);
        if (cudaMemcpyAsync((uint8_t *)dst + off, sg->buf[b], len, cudaMemcpyHostToDevice, st) !=
                cudaSuccess ||
            cudaEventRecord(sg->ev[b], st) != cudaSuccess) {
            return -1;
        }
        sg->cur ^= 1;
    }
    return 0;
}

/* RECLANK_GPU_FETCH_PULL (8): the copies of a GP_FETCH job from
 * registered host memory as one kernel of that many blocks of 256 threads
 * that reads the host memory (the map of the pinned experts) and writes the
 * buffer, in place of a DMA copy for each range. On the 3090 (PCIe 3.0 x16)
 * the DMA engine moved 8.0 GB/s with copies of 1 MB to 512 MB (more streams
 * did not help); a kernel of 4 to 32 blocks reads 12.3 GB/s. 0: DMA. A job
 * with a range that is not registered or not 16-byte aligned takes DMA. In
 * the mixed groups 8 blocks leave room on the bus and in the SMs for the
 * copies on the path of each layer (the part of the CPU, TO_DEV): Qwen3.8
 * EXL3, a prompt of 4096 tokens, 882 tok/s against 792 with 16. A GPU of
 * fewer than 24 SMs takes DMA by default: there the blocks of the kernel are
 * a share of the SMs that the experts of the group need. The GTX 1070 (15
 * SMs), Qwen3.6, a prompt of 4096 tokens: DMA 331 tok/s (the copies at 8.2
 * GB/s, hidden), 4 blocks 317, 8 blocks 307, 16 blocks 287. */
static int gg_pull_blocks(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_GPU_FETCH_PULL");
        n = v ? atoi(v) : gg_caps.sm_count > 0 && gg_caps.sm_count < 24 ? 0 : 8;
    }
    return n;
}

/* RECLANK_GPU_TILE_PULL (16): the blocks of k_nm_tile_pull (the copies of a stage in NV4T) */
static int gg_tile_blocks(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_GPU_TILE_PULL");
        n = v ? atoi(v) : 16;
        n = n > 0 ? n : 16;
    }
    return n;
}

__global__ void k_fetch_pull(const int64_t *__restrict__ rg, int count)
{
    const size_t step = (size_t)gridDim.x * blockDim.x;
    for (int k = 0; k < count; ++k) {
        const int4 *src = (const int4 *)(intptr_t)rg[3 * k];
        int4 *dst = (int4 *)(intptr_t)rg[3 * k + 1];
        size_t n = (size_t)rg[3 * k + 2] / 16;
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        /* two loads in flight a thread */
        for (; i + step < n; i += 2 * step) {
            int4 a = src[i], b = src[i + step];
            dst[i] = a;
            dst[i + step] = b;
        }
        if (i < n) {
            dst[i] = src[i];
        }
    }
}

/* The ranges of a job for k_fetch_pull (the device address of each
 * source), in rg (3 * count values; the ranges of 0 bytes left out); the
 * count, or -1 when the job must take DMA. dev: a source in the GPU's own
 * memory too (k_nm_tile_pull: the hot experts' slots); keep: the ranges of 0
 * bytes kept (their place: the even / odd columns of k_nm_tile_pull). */
static int gg_pull_ranges(const int64_t *ranges, int count, int64_t *rg, int dev = 0, int keep = 0)
{
    int m = 0;
    for (int k = 0; k < count; ++k) {
        const int64_t *q = ranges + 3 * k;
        if (q[2] == 0) {
            if (keep) {
                rg[3 * m] = rg[3 * m + 1] = rg[3 * m + 2] = 0;
                ++m;
            }
            continue;
        }
        cudaPointerAttributes a;
        if (cudaPointerGetAttributes(&a, (const void *)(intptr_t)q[0]) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        int64_t src = (int64_t)(intptr_t)a.devicePointer;
        if (dev && a.type == cudaMemoryTypeDevice) {
            src = q[0];
        } else if (a.type != cudaMemoryTypeHost) {
            return -1;
        }
        if (src == 0 || (src | q[1] | q[2]) % 16 != 0) {
            return -1;
        }
        rg[3 * m] = src;
        rg[3 * m + 1] = q[1];
        rg[3 * m + 2] = q[2];
        ++m;
    }
    return m;
}

static void *gg_worker(void *arg)
{
    (void)arg;
    cudaSetDevice(gg_device);
    static gg_stager sg;
    static int64_t *prg, *prg_d;
    static int prg_cap;
    gg_worker_bind();
    for (;;) {
        pthread_mutex_lock(&gg_mu);
        while (gg_job_head == gg_job_tail) {
            pthread_cond_wait(&gg_cv, &gg_mu);
        }
        gg_job j = gg_jobs[gg_job_head % GG_FETCH_MAX];
        pthread_mutex_unlock(&gg_mu);
        int bad = cudaEventSynchronize(gg_freeev[j.b]) != cudaSuccess;
        double t0 = gg_clock(), nb = 0.0;
        int np_ = -1;
        if (!bad && (gg_pull_blocks() > 0 || j.mode == 1)) {
            if (j.count > prg_cap) {
                free(prg);
                cudaFree(prg_d);
                prg = (int64_t *)malloc((size_t)j.count * 24);
                bad = prg == NULL || cudaMalloc((void **)&prg_d, (size_t)j.count * 24) != cudaSuccess;
                prg_cap = bad ? 0 : j.count;
            }
            np_ = bad ? -1 : gg_pull_ranges(j.ranges, j.count, prg, j.mode == 1, j.mode == 1);
            if (j.mode == 1 && np_ < 0) {
                bad = 1;            /* (NV4T has no DMA form) */
            }
        }
        if (np_ > 0) {
            /* prg is pageable: the call returns when the driver has the
             * values, so the next job can write prg */
            bad = cudaMemcpyAsync(prg_d, prg, (size_t)np_ * 24, cudaMemcpyHostToDevice, gg_copy) !=
                  cudaSuccess;
            if (!bad && j.mode == 1) {
                /* (16 rows of the widest matrix in shared memory: opted in once) */
                static size_t tp_set = 0;
                const size_t sm = 16 * kq_row_bytes(KQ_NV4, j.c0 > j.c1 ? j.c0 : j.c1);
                if (sm > 48 * 1024 && sm > tp_set) {
                    cudaFuncSetAttribute(k_nm_tile_pull, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
                    tp_set = sm;
                }
                k_nm_tile_pull<<<gg_tile_blocks(), 256, sm, gg_copy>>>(prg_d, np_, j.c0, j.c1);
                bad = cudaGetLastError() != cudaSuccess;
            } else if (!bad) {
                k_fetch_pull<<<gg_pull_blocks(), 256, 0, gg_copy>>>(prg_d, np_);
                bad = cudaGetLastError() != cudaSuccess;
            }
            for (int k = 0; k < np_; ++k) {
                nb += (double)prg[3 * k + 2];
            }
        }
        for (int k = 0; k < j.count && !bad && np_ < 0; ++k) {
            const int64_t *q = j.ranges + 3 * k;
            if (q[2] == 0) {
                continue;       /* a row of a fixed-size list that copies nothing */
            }
            bad = gg_h2d_staged(&sg, (void *)(intptr_t)q[1], (const void *)(intptr_t)q[0],
                                (size_t)q[2], gg_copy, 1) != 0;
            nb += (double)q[2];
        }
        bad = bad || cudaEventRecord(gg_ready[j.f], gg_copy) != cudaSuccess;
        pthread_mutex_lock(&gg_mu);
        if (bad && !gg_fetch_error) {
            cudaError_t ce = cudaGetLastError();
            snprintf(gg_fetch_why, sizeof(gg_fetch_why), "%s (job of %d ranges, buffer %d)",
                     cudaGetErrorString(ce), j.count, j.b);
        }
        gg_fetch_error |= bad;
        gg_recorded[j.f] = 1;
        ++gg_job_head;
        pthread_cond_broadcast(&gg_cv);
        pthread_mutex_unlock(&gg_mu);
        if (gg_ms_on && !bad && nb > 0.0) {
            /* The time of the copies of this job: the jobs do not overlap. */
            cudaEventSynchronize(gg_ready[j.f]);
            double dt = gg_clock() - t0;
            pthread_mutex_lock(&gg_mu);
            gg_ms_copy_bytes += nb;
            gg_ms_copy_s += dt;
            pthread_mutex_unlock(&gg_mu);
        }
    }
    return NULL;
}

/* ---------- the copies of the cache of hot experts ----------
 * rc_gpu_cache_copy queues a list of ranges (host address, device address,
 * bytes). A worker thread of its own copies them on a stream of its own,
 * then records an event. The host memory is the map of the model file, so a
 * copy call waits for the data; the worker does that wait, not the runner.
 * rc_gpu_cache_query gives 1 when the copies of a job are done. The caller keeps
 * the ranges until then. packs (rc_gpu_cache_copy_q4s: device address, bytes,
 * cols): the int4 rows the copies wrote, into Q4S after them on the stream
 * (k_q4s_pack: the hot experts of a GPU without dp4a), before the event. */
#define GG_CACHE_JOBS 64

typedef struct {
    const int64_t *ranges;
    int count, id;
    const int64_t *packs;
    int npacks;
} gg_cjob;

static pthread_mutex_t gg_cmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gg_ccv = PTHREAD_COND_INITIALIZER;
static gg_cjob gg_cjobs[GG_CACHE_JOBS];
static int gg_cjob_head, gg_cjob_tail, gg_cjob_next;
static int gg_cissued[GG_CACHE_JOBS], gg_cid[GG_CACHE_JOBS], gg_cerror;
static cudaEvent_t gg_cev[GG_CACHE_JOBS];
static cudaStream_t gg_cstream;
static int gg_cworker_on;

static void *gg_cworker(void *arg)
{
    (void)arg;
    cudaSetDevice(gg_device);
    static gg_stager sg;
    gg_worker_bind();
    for (;;) {
        pthread_mutex_lock(&gg_cmu);
        while (gg_cjob_head == gg_cjob_tail) {
            pthread_cond_wait(&gg_ccv, &gg_cmu);
        }
        gg_cjob j = gg_cjobs[gg_cjob_head % GG_CACHE_JOBS];
        pthread_mutex_unlock(&gg_cmu);
        int bad = 0;
        for (int k = 0; k < j.count && !bad; ++k) {
            const int64_t *q = j.ranges + 3 * k;
            /* (a page-locked source by DMA: the weightkeeper's segments that the caller registered; no
             * memcpy by this thread beside the CPU's experts) */
            bad = gg_h2d_staged(&sg, (void *)(intptr_t)q[1], (const void *)(intptr_t)q[0],
                                (size_t)q[2], gg_cstream, 1) != 0;
        }
        for (int k = 0; k < j.npacks && !bad; ++k) {
            const int64_t *q = j.packs + 3 * k;
            bad = q4s_pack_launch((uint8_t *)(intptr_t)q[0], (size_t)q[1], (int)q[2], gg_cstream) != cudaSuccess;
        }
        int slot = j.id % GG_CACHE_JOBS;
        bad = bad || cudaEventRecord(gg_cev[slot], gg_cstream) != cudaSuccess;
        pthread_mutex_lock(&gg_cmu);
        gg_cerror |= bad;
        gg_cissued[slot] = 1;
        ++gg_cjob_head;
        pthread_cond_broadcast(&gg_ccv);
        pthread_mutex_unlock(&gg_cmu);
    }
    return NULL;
}
