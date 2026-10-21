/* ---------- the host side ---------- */
#include <unistd.h>  /* (usleep: RECLANK_CPU_TASK_DELAY_US) */

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
    float *nv8;          /* the column shares of k_nm_t8 */
    size_t nv8_n;
    int *hl;             /* the lists of the DeepSeek hot experts (ds_hot_tc6) */
    size_t hl_n;         /* (ints) */
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

/* GP_SIGNAL's operand 11 (or none): the flag out of the CPU task that last read the pinned arrays
 * this signal writes. An AWAIT with a count of 0 does not wait for its task, so the GPU can reach the
 * next signal while the runner still runs that task (late: the runner was busy); the copies would
 * then change its input under it (its count, its experts: a CPU program read past its arrays, a
 * general protection fault in gp_step). The signal waits for that task first (it is done long
 * before, but for that case: one read of the flag). */
__device__ __forceinline__ void signal_wait(const gp_rec *r, const int64_t *e)
{
    if (r->tag[11] == GP_T_NONE || di(r, e, 11) == 0) {
        return;
    }
    if (threadIdx.x == 0) {
        volatile int64_t *prev = DP(volatile int64_t, 11);
        int64_t v = di(r, e, 1);
        uint64_t t0 = rc_globaltimer();
        while (*prev < v) {
            rc_nanosleep(256);
            if (rc_globaltimer() - t0 > 60ull * 1000000000ull) {
                __trap();           /* the task before never ended in 60 s: an error, not a hang */
            }
        }
    }
    __syncthreads();
}

__device__ void signal_body(const gp_rec *r, const int64_t *e, int k0 = 0)
{
    for (int k = k0; k < 3; ++k) {
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

__global__ void k_signal(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    signal_wait(r, e);
    signal_body(r, e);
}

/* A Nemotron step's GP_ROUTER_SIG (with its split: k_router_sig_r's operands 10-12) and the GP_SIGNAL after it
 * in one block (as k_ds_route_signal): the latent rows' copy first (ready before the router: their writes
 * over PCIe in flight while it sorts), then the router, then the copies of its outputs and the flag. */
__global__ void __launch_bounds__(512) k_nm_route_signal(const gp_rec *r, const int64_t *e, const gp_rec *rg)
{
    PDL_START();
    signal_wait(rg, e);
    {
        const gp_rec *r = rg;
        if (di(r, e, 2) != 0) {
            block_copy4(DP(void, 3), DP(const void, 2), (size_t)di(r, e, 4));
        }
    }
    router_sig_r_body(r, e);
    __threadfence();
    __syncthreads();
    signal_body(rg, e, 1);
}

/* GP_DS_ROUTER of one token, the GP_HOT_SPLIT after it and the GP_SIGNAL after that in one block (a
 * DeepSeek V4.1 step: three kernels of one block each, 7.9 + 4.5 + 10 us and their gaps, on the way
 * to the CPU's task of every layer). The same operations in the same order. */
__global__ void __launch_bounds__(256) k_ds_route_signal(const gp_rec *r, const int64_t *e, const gp_rec *rs,
                                                         const gp_rec *rg)
{
    PDL_START();
    /* the input rows (copy 0: x of the experts, ready before the router) go first, their writes over
     * PCIe in flight while the router and the split run; the copies of the split's outputs after */
    const gp_rec *r0 = r;
    signal_wait(rg, e);
    {
        const gp_rec *r = rg;
        if (di(r, e, 2) != 0) {
            block_copy4(DP(void, 3), DP(const void, 2), (size_t)di(r, e, 4));
        }
    }
    ds_router_body(r0, e, 0);
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) {
        hot_split_body(rs, e);
        __threadfence();
    }
    __syncthreads();
    signal_body(rg, e, 1);
}

/* operand 6 (or none): rows the CPU's are added to (dst = add + src: the GP_ADD after it folded in; with
 * no cold pair add + 0, its bits) */
__global__ void k_await(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const float *add = r->tag[6] != GP_T_NONE ? DP(const float, 6) : NULL;
    if (di(r, e, 5) != 0 && *DP(const int, 5) == 0) {
        if (di(r, e, 2) != 0) {
            float *d = DP(float, 3);
            for (size_t i = threadIdx.x; i < (size_t)di(r, e, 4) / 4; i += blockDim.x) {
                d[i] = add != NULL ? add[i] + 0.f : 0.f;
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
    if (di(r, e, 2) != 0 && add != NULL) {
        float *d = DP(float, 3);
        const float *src = DP(const float, 2);
        for (size_t i = threadIdx.x; i < (size_t)di(r, e, 4) / 4; i += blockDim.x) {
            d[i] = add[i] + src[i];
        }
    } else if (di(r, e, 2) != 0) {
        block_copy4(DP(void, 3), DP(const void, 2), (size_t)di(r, e, 4));
    }
}

static int64_t hi(const gp_rec *r, const int64_t *e, int k);

/* The measures of the tasks (rc_gpu_task_stats): the time the runner waited for
 * the signals, the time of the CPU programs, the count. */
static double gg_ts_wait, gg_ts_cpu;
static int64_t gg_ts_n;
/* rc_gpu_task_log: while on, the clock at each task's wait, start and end (a measure) */
#define GG_TL_MAX 65536
static double *gg_tl;
static int gg_tl_n, gg_tl_on;

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

/* on 1: the log of the tasks cleared and kept from now on (0: stopped). out (or NULL): the log so far,
 * 3 clocks a task (the wait's start, the program's start and end, seconds), at most max tasks; the count. */
extern "C" int rc_gpu_task_log(int on, double *out, int max)
{
    int n = gg_tl_n < max ? gg_tl_n : max;
    if (out != NULL && gg_tl != NULL) {
        memcpy(out, gg_tl, (size_t)n * 3 * sizeof(double));
    }
    if (on && gg_tl == NULL) {
        gg_tl = (double *)malloc((size_t)GG_TL_MAX * 3 * sizeof(double));
    }
    if (on) {
        gg_tl_n = 0;
    }
    gg_tl_on = on && gg_tl != NULL;
    return n;
}

/* ---------- lane B (docs/LANES_PLAN.md) ----------
 * rc_gpu_lane(1) gives the calling thread a stream of its own (gg_tstream)
 * and makes it lane B: it runs a step program of its own at the same time as
 * the model thread runs lane A's. The CPU programs of the cold experts run in
 * the parts, OpenMP teams of the thread that runs them, and the teams of a
 * second thread put the threads over the CPUs (server/worker.py); so lane B
 * never runs a CPU program: at each GP_CPU_TASK it waits for its signal, puts
 * the program in the mailbox, and waits; the model thread runs it while it
 * waits for its own signals (gg_cpu_task) and, after its run, until lane B is
 * done (rc_gpu_lane_serve). One program at a time on the one team. */
static cudaStream_t gg_lane_stream;
static __thread int gg_lane_b;
static const int64_t *volatile gg_lb_prog;
static int gg_lb_state;         /* 0 empty, 1 a program to run, 2 done, -1 it failed */
static double gg_lb_cpu;        /* the measures of lane B's programs (rc_gpu_lane_stats) */
static int64_t gg_lb_n;

/* The model thread: run the program of lane B's mailbox, if there is one. */
static inline void gg_lane_serve_one(void)
{
    if (__atomic_load_n(&gg_lb_state, __ATOMIC_ACQUIRE) != 1) {
        return;
    }
    double t0 = gg_clock();
    int rc = gg_cpu_run == NULL ? -1 : gg_cpu_run(gg_lb_prog, -1);
    gg_lb_cpu += gg_clock() - t0;
    ++gg_lb_n;
    __atomic_store_n(&gg_lb_state, rc == 0 ? 2 : -1, __ATOMIC_RELEASE);
}

/* The seconds of lane B's CPU programs (run by the model thread) and their count since the last
 * call, to out[0] and out[1] (rc_gpu_task_stats has the model thread's own). */
extern "C" int rc_gpu_lane_stats(double *out)
{
    out[0] = gg_lb_cpu;
    out[1] = (double)gg_lb_n;
    gg_lb_cpu = 0.0;
    gg_lb_n = 0;
    return 0;
}

extern "C" int rc_gpu_lane(int on)
{
    if (on && gg_lane_stream == NULL) {
        CK(cudaStreamCreateWithFlags(&gg_lane_stream, cudaStreamNonBlocking));
    }
    gg_tstream = on ? gg_lane_stream : NULL;
    gg_lane_b = on != 0;
    return 0;
}

/* The model thread, after its run: lane B's programs until *done (lane B's
 * thread sets it after its run). */
extern "C" int rc_gpu_lane_serve(const int64_t *done)
{
    double t0 = gg_clock();
    while (__atomic_load_n(done, __ATOMIC_ACQUIRE) == 0) {
        gg_lane_serve_one();
        __builtin_ia32_pause();
        if (gg_clock() - t0 > 60.0) {
            snprintf(gg_error, sizeof(gg_error), "rc_gpu_lane_serve: lane B not done in 60 s");
            return -1;
        }
    }
    return 0;
}

/* Wait for the work of the calling thread's stream. */
extern "C" int rc_gpu_lane_sync(void)
{
    CK(cudaStreamSynchronize(gg_stream));
    return 0;
}

/* Run a GP_CPU_TASK on the host: wait for flag in, run, write flag out. The
 * model thread runs lane B's programs while it waits; lane B's thread hands
 * its own to the model thread. */
static int gg_cpu_task(const gp_rec *r, const int64_t *henv)
{
    volatile int64_t *fin = (volatile int64_t *)(intptr_t)r->v[1];
    volatile int64_t *fout = (volatile int64_t *)(intptr_t)r->v[2];
    int64_t v = hi(r, henv, 3);
    double t0 = gg_clock();
    while (*fin < v) {
        if (!gg_lane_b) {
            gg_lane_serve_one();
        }
        __builtin_ia32_pause();
        if (gg_clock() - t0 > 30.0) {
            snprintf(gg_error, sizeof(gg_error), "GP_CPU_TASK%s: no signal of the GPU in 30 s (flag %lld, "
                     "waits for %lld; lane B's mailbox %d)", gg_lane_b ? " of lane B" : "",
                     (long long)*fin, (long long)v, __atomic_load_n(&gg_lb_state, __ATOMIC_RELAXED));
            return -1;
        }
    }
    __atomic_thread_fence(__ATOMIC_ACQUIRE);
    {
        /* RECLANK_CPU_TASK_DELAY_US: the runner late by that much before each task (a test of the
         * order of the signals: the GPU then reaches the next signal first after an AWAIT that does
         * not wait, and signal_wait must keep that task's input) */
        static int delay = -1;
        if (delay < 0) {
            const char *dv = getenv("RECLANK_CPU_TASK_DELAY_US");
            delay = dv ? atoi(dv) : 0;
        }
        if (delay > 0) {
            usleep((useconds_t)delay);
        }
    }
    double t1 = gg_clock();
    if (gg_lane_b) {
        gg_lb_prog = (const int64_t *)(intptr_t)r->v[0];
        __atomic_store_n(&gg_lb_state, 1, __ATOMIC_RELEASE);
        int st;
        while ((st = __atomic_load_n(&gg_lb_state, __ATOMIC_ACQUIRE)) == 1) {
            __builtin_ia32_pause();
            if (gg_clock() - t1 > 30.0) {
                snprintf(gg_error, sizeof(gg_error), "GP_CPU_TASK of lane B: not run in 30 s");
                return -1;
            }
        }
        __atomic_store_n(&gg_lb_state, 0, __ATOMIC_RELAXED);
        if (st < 0) {
            snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_TASK of lane B failed");
            return -1;
        }
    } else if (gg_cpu_run == NULL || gg_cpu_run((const int64_t *)(intptr_t)r->v[0], -1) != 0) {
        snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_TASK failed");
        return -1;
    }
    __atomic_thread_fence(__ATOMIC_RELEASE);
    *fout = v;
    if (!gg_lane_b) {           /* (the measures: the model thread's tasks) */
        double t2 = gg_clock();
        gg_ts_wait += t1 - t0;
        gg_ts_cpu += t2 - t1;
        ++gg_ts_n;
        if (gg_tl_on && gg_tl_n < GG_TL_MAX) {
            gg_tl[3 * gg_tl_n] = t0;
            gg_tl[3 * gg_tl_n + 1] = t1;
            gg_tl[3 * gg_tl_n + 2] = t2;
            ++gg_tl_n;
        }
    }
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

/* The helper keeps the team of OpenMP threads of its programs while they
 * come one after the other (the layers of a prompt group); after this long
 * with no program it lets them go (gg_cpu_run of a null program). The
 * threads of a team stay with the thread that opened it, and while the
 * process has more of them than CPUs, libgomp spins only briefly before a
 * thread sleeps: a team of 25 kept by the helper after a prompt put the 35
 * threads of the parts (GP_CPU_TASK, every layer of a decode step) to sleep
 * between the layers, to be woken at each. */
#define GG_HELPER_IDLE_NS 20000000L

static void *gg_helper(void *arg)
{
    (void)arg;
    cudaSetDevice(gg_device);
    int team = 0;       /* the helper has kept a team since its last program */
    for (;;) {
        pthread_mutex_lock(&gg_hmu);
        while (gg_hjob == NULL) {
            if (!team) {
                pthread_cond_wait(&gg_hcv, &gg_hmu);
                continue;
            }
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += GG_HELPER_IDLE_NS;
            if (ts.tv_nsec >= 1000000000L) {
                ts.tv_sec += 1;
                ts.tv_nsec -= 1000000000L;
            }
            if (pthread_cond_timedwait(&gg_hcv, &gg_hmu, &ts) != 0 && gg_hjob == NULL) {
                pthread_mutex_unlock(&gg_hmu);
                if (gg_cpu_run != NULL) {
                    gg_cpu_run(NULL, -1);
                }
                team = 0;
                pthread_mutex_lock(&gg_hmu);
            }
        }
        team = 1;
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
