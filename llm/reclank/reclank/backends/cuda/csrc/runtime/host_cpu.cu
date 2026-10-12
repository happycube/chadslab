/* ---------- the host side ---------- */

/* The runner of a CPU program: gemma_run of the CPU library. Python gives its
 * address with rc_gpu_set_cpu_runner. */
typedef int (*gg_cpu_runner)(const int64_t *prog, int limit);
static gg_cpu_runner gg_cpu_run;

/* A segment: the kernel records from start to end - 1, and their graph. */
typedef struct {
    int start, end;
    cudaGraphExec_t exec;
} gg_seg;

typedef struct {
    int n_env, n_code;
    int64_t *henv;       /* the environment on the host */
    int64_t *denv;       /* the environment on the GPU */
    gp_rec *hcode;       /* the records, with device addresses */
    gp_rec *dcode;
    int use_graph;
    float *part;         /* the scratch of the attention */
    int n_seg;
    gg_seg *seg;         /* the segments, in the order of the records */
    int n_ev;
    cudaEvent_t *ev;     /* the events of the GP_TO_HOST records */
    __half *xh;          /* the float16 copy of the input of a product */
    size_t xh_n;
    uint16_t *w16;       /* the bfloat16 rows of a chunk of a KQ_BF12 matrix (gg_gemm_bf12) */
    size_t w16_n;
    float *btp;          /* the parts of the sums of k_kq_bf12_tc of class 1, and their counts */
    int *btc;
    int8_t *kqx;         /* the int8 rows of the products of large groups (k_kq_tc) */
    size_t kqx_n;        /* its values; then xs and xsum, kqx_n / 32 floats each */
    int tc;              /* 1: the tensor cores for a large group */
    int tcreq;           /* the program asks for tc (the caps aside): the int8 tiles of
                          * dp4a take its place without the tensor cores */
    int i8;              /* 1: the int8 products of k_gemm_q8 for a large group, also
                          * with tc 0 (the 26B: int8 has no overflow, float16 has);
                          * 2: the int16 form (k_quant_x2), also for the experts */
    int atc;             /* 1: the attention of a large group on the tensor cores,
                          * also with tc 0 (the queries and keys have a norm) */
    int kqx_lo;          /* 1: kqx has the lo plane of the int16 form (kt_lo) */
    uint8_t *e3;         /* the scratch of the KQ_E3 products (e3_ws_bytes) */
    size_t e3_n;
} gg_prog;

/* The size of the scratch of the attention: 24 query heads of 512 values
 * (the parts of 4 queries of a global layer of the Gemma 4 26B, 16 heads
 * in 85 chunks: k_attn_fdtc T 4), or the same count of values in another
 * form. */
#define GG_PART_FLOATS ((size_t)24 * ATTN_CHUNKS * (512 + 2))

/* ---------- the handoff to the CPU inside a graph ----------
 * A step can hand work to the CPU with no boundary record, so the step is
 * one graph and the host launches nothing between its layers:
 *
 *   GP_D2H (device source, host target, bytes) and GP_H2D (host source,
 *   device target, bytes): copies in the stream (copy nodes of the graph).
 *   GP_SIGNAL (flag, value, then up to 3 copies (device source, host
 *   target, bytes)): a kernel copies the data to pinned host memory (the
 *   GPU writes it through the map of the memory), then writes value to the
 *   flag, an int64 in pinned host memory.
 *   GP_AWAIT (flag, value, then a copy (host source, device target, bytes)
 *   or none, then a device int or none): a kernel waits until the flag is
 *   value or more, then copies the data from pinned host memory. The int is
 *   the length of the block of the CPU (the count of its experts): with 0,
 *   the kernel writes zeros to the target and does not wait, and the GPU
 *   goes on while the runner only marks the task done. A copy node of a graph costs more than
 *   a kernel, and it breaks the programmatic edges.
 *   GP_CPU_TASK (CPU program, flag in, flag out, value): no kernel. After
 *   the launch of its segment, the runner waits until flag in is value or
 *   more, runs the program, and writes value to flag out.
 *
 * The value is a slot that grows at each run (seq), so the flags need no
 * reset. The tasks run in their order, on the thread of the runner. */
/* Copy n bytes (a multiple of 4) with the threads of the block: 16 bytes
 * for each access when the addresses allow it (fewer PCIe transfers). The
 * loads do not use the caches (ld.global.cv): the source can be pinned host
 * memory that the CPU wrote since the last step, and a line of the last
 * step in a cache would give its old values. */
__device__ __forceinline__ void block_copy4(void *dst, const void *src, size_t n)
{
    if ((((uintptr_t)dst | (uintptr_t)src | n) & 15) == 0) {
        const float4 *s = (const float4 *)src;
        float4 *d = (float4 *)dst;
        for (size_t i = threadIdx.x; i < n / 16; i += blockDim.x) {
            d[i] = __ldcv(s + i);
        }
        return;
    }
    const float *s = (const float *)src;
    float *d = (float *)dst;
    for (size_t i = threadIdx.x; i < n / 4; i += blockDim.x) {
        d[i] = __ldcv(s + i);
    }
}

__global__ void k_signal(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    for (int k = 0; k < 3; ++k) {
        if (di(r, e, 2 + 3 * k) != 0) {
            block_copy4(DP(void, 3 + 3 * k), DP(const void, 2 + 3 * k), (size_t)di(r, e, 4 + 3 * k));
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        *DP(volatile int64_t, 0) = di(r, e, 1);
        __threadfence_system();
    }
}

__global__ void k_await(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    if (di(r, e, 5) != 0 && *DP(const int, 5) == 0) {
        if (di(r, e, 2) != 0) {
            float *d = DP(float, 3);
            for (size_t i = threadIdx.x; i < (size_t)di(r, e, 4) / 4; i += blockDim.x) {
                d[i] = 0.f;
            }
        }
        return;
    }
    if (threadIdx.x == 0) {
        volatile int64_t *flag = DP(volatile int64_t, 0);
        int64_t v = di(r, e, 1);
        uint64_t t0;
        t0 = rc_globaltimer();
        while (*flag < v) {
            rc_nanosleep(256);
            uint64_t t;
            t = rc_globaltimer();
            if (t - t0 > 60ull * 1000000000ull) {
                __trap();           /* no CPU task in 60 s: an error, not a hang */
            }
        }
    }
    __syncthreads();
    __threadfence_system();
    if (di(r, e, 2) != 0) {
        block_copy4(DP(void, 3), DP(const void, 2), (size_t)di(r, e, 4));
    }
}

static int64_t hi(const gp_rec *r, const int64_t *e, int k);

/* The measures of the tasks (rc_gpu_task_stats): the time the runner waited for
 * the signals, the time of the CPU programs, the count. */
static double gg_ts_wait, gg_ts_cpu;
static int64_t gg_ts_n;

/* out gets the wait and CPU seconds and the count of the tasks since the
 * last call; then they start again. */
extern "C" int rc_gpu_task_stats(double *out)
{
    out[0] = gg_ts_wait;
    out[1] = gg_ts_cpu;
    out[2] = (double)gg_ts_n;
    gg_ts_wait = gg_ts_cpu = 0.0;
    gg_ts_n = 0;
    return 0;
}

/* Run a GP_CPU_TASK on the host: wait for flag in, run, write flag out. */
static int gg_cpu_task(const gp_rec *r, const int64_t *henv)
{
    volatile int64_t *fin = (volatile int64_t *)(intptr_t)r->v[1];
    volatile int64_t *fout = (volatile int64_t *)(intptr_t)r->v[2];
    int64_t v = hi(r, henv, 3);
    double t0 = gg_clock();
    while (*fin < v) {
        __builtin_ia32_pause();
        if (gg_clock() - t0 > 30.0) {
            snprintf(gg_error, sizeof(gg_error), "GP_CPU_TASK: no signal of the GPU in 30 s");
            return -1;
        }
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    double t1 = gg_clock();
    if (gg_cpu_run == NULL || gg_cpu_run((const int64_t *)(intptr_t)r->v[0], -1) != 0) {
        snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_TASK failed");
        return -1;
    }
    __atomic_thread_fence(__ATOMIC_RELEASE);
    *fout = v;
    double t2 = gg_clock();
    gg_ts_wait += t1 - t0;
    gg_ts_cpu += t2 - t1;
    ++gg_ts_n;
    return 0;
}

static int is_boundary(int op)
{
    return op == GP_TO_HOST || op == GP_CPU_JOIN || op == GP_TO_DEV || op == GP_FETCH ||
           op == GP_FETCH_WAIT || op == GP_FETCH_DONE || op == GP_CPU_START ||
           op == GP_CPU_WAIT;
}

/* ---------- a CPU program on a thread of its own ----------
 * GP_CPU_START (program, event): wait for the event, then a helper thread
 * runs the CPU program while the runner goes on (the copies of GP_FETCH,
 * the launches of the GPU). GP_CPU_WAIT: wait for the program. One program
 * at a time. */
static pthread_mutex_t gg_hmu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gg_hcv = PTHREAD_COND_INITIALIZER;
static const int64_t *gg_hjob;
static int gg_hbusy, gg_hfail, gg_hon;
static double gg_hlast;       /* the time of the last program, in s */

static void *gg_helper(void *arg)
{
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&gg_hmu);
        while (gg_hjob == NULL) {
            pthread_cond_wait(&gg_hcv, &gg_hmu);
        }
        const int64_t *prog = gg_hjob;
        pthread_mutex_unlock(&gg_hmu);
        double t0 = gg_clock();
        int rc = gg_cpu_run == NULL ? -1 : gg_cpu_run(prog, -1);
        double dt = gg_clock() - t0;
        pthread_mutex_lock(&gg_hmu);
        gg_hlast = dt;
        gg_hjob = NULL;
        gg_hbusy = 0;
        gg_hfail |= rc != 0;
        pthread_cond_broadcast(&gg_hcv);
        pthread_mutex_unlock(&gg_hmu);
    }
    return NULL;
}
