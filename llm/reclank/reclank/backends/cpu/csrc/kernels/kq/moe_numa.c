/* One matrix of the MoE: data and type. */
typedef struct {
    const uint8_t *w;
    int type;
} kq_mat;

static inline kq_mat kq_mat_of(const int64_t *d)
{
    kq_mat m;
    m.w = (const uint8_t *)(intptr_t)d[0];
    m.type = (int)d[1];
    return m;
}

/* The experts of t tokens in the GGUF formats. h is
 * quantized (hq, hs, hm; kq_quant_body). mats has 6 descriptors (w, type):
 * gate, up, down of the stacked experts, then of the shared expert (w = 0
 * for none). The weight of the shared expert is sigmoid(shared_logit).
 * kcount (or null) holds k, for one token. */
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
extern int sched_getcpu(void);

/* The NUMA node of each CPU (/sys/devices/system/node), read once. */
static int kq_cpu_node_tab[4096];
static pthread_once_t kq_cpu_node_once = PTHREAD_ONCE_INIT;

static void kq_cpu_node_init(void)
{
    for (int node = 0; node < 64; ++node) {
        char path[96];
        snprintf(path, sizeof(path), "/sys/devices/system/node/node%d/cpulist", node);
        FILE *f = fopen(path, "r");
        if (f == NULL) {
            continue;
        }
        char line[1024];
        if (fgets(line, sizeof(line), f) != NULL) {
            char *q = line;
            while (*q && *q != '\n') {
                int a = (int)strtol(q, &q, 10), b = a;
                if (*q == '-') {
                    b = (int)strtol(q + 1, &q, 10);
                }
                for (int c = a; c <= b && c < 4096; ++c) {
                    if (c >= 0) {
                        kq_cpu_node_tab[c] = node;
                    }
                }
                if (*q == ',') {
                    ++q;
                }
            }
        }
        fclose(f);
    }
}

/* The node of the CPU of the calling thread (0 without /sys). */
static int kq_my_node(void)
{
    pthread_once(&kq_cpu_node_once, kq_cpu_node_init);
    int c = sched_getcpu();
    return c >= 0 && c < 4096 ? kq_cpu_node_tab[c] : 0;
}

/* The share of the tasks of a decode step (t = 1, static) for the threads
 * of the first half of the team (bound spread: the CPUs of node 0): the
 * nodes can stream at different rates (6 memory channels on node 0 and 4 on
 * node 1 on the 2-socket Xeon). rc_kq_calib_nodes measures it at the start. */
static float kq_node0_share = 0.5f;

/* RQ8_0 experts (gguf.RQ8_0: the Q8_0 blocks of rows rotated in each 32
 * values): the act of each pair is rotated the same way before the down
 * product (the program rotates the x of gate and up). A state of the
 * process, set by the model (Qwen4CPU). */
static int kq_moe_rot = 0;

/* RECLANK_MOE_PROF=1: the times of the phases of kq_moe_small_body (the
 * cold experts of a GPU step), summed over the calls (rc_kq_moe_prof): the
 * spread of the arrival of the threads, the single region, the end of the
 * gate and up tasks and of the down tasks of each thread (mean and max from
 * the end of the single), the wait at the last barrier, and the sum. */
static int kq_mprof_on = -1;
static double kq_mprof[16];
static long kq_mprof_n;
static double kq_mt[5][256];
static double kq_mthr[256][3];
static double kq_msb[4];
static double kq_mbusy[256][2];    /* per thread: the time in the kernels of gate/up, of all (s, summed) */           /* the single region: quant, sort, split, rows (s, summed) */     /* per thread: the sums of arrival, end of gate/up, end of down */

/* out: 0 the spread of the arrival, 1 the single region, 2 3 the end of gate
 * and up (mean, max), 4 5 the end of down (mean, max), 6 all, 7 the calls;
 * 8 .. 13 by half of the team (node 0, node 1): the arrival after the
 * first thread, the end of gate and up, the end of down (means). */
/* The times of each thread of the last call (us, from the end of the single
 * region): arrival (as from the first thread), the end of gate and up, the
 * end of down; n threads. */
/* The means per thread since the last call (us): arrival, end of gate/up,
 * end of down, for n threads; calls the count of calls (from rc_kq_moe_prof's
 * count at that time). */
void rc_kq_moe_prof_thread_means(double *out, int n, double calls)
{
    for (int q = 0; q < n && q < 256; ++q) {
        for (int i = 0; i < 3; ++i) {
            out[3 * q + i] = calls > 0 ? 1e6 * kq_mthr[q][i] / calls : 0.0;
            kq_mthr[q][i] = 0.0;
        }
    }
}

int rc_kq_moe_prof_threads(double *out, int n)
{
    double in0 = 1e30;
    for (int q = 0; q < n; ++q) {
        in0 = fmin(in0, kq_mt[0][q]);
    }
    for (int q = 0; q < n; ++q) {
        out[3 * q] = 1e6 * (kq_mt[0][q] - in0);
        out[3 * q + 1] = 1e6 * (kq_mt[2][q] - kq_mt[1][0]);
        out[3 * q + 2] = 1e6 * (kq_mt[3][q] - kq_mt[1][0]);
    }
    return n;
}

/* The parts of the single region of kq_moe_small_body (us a call): the
 * quantization of x, the sort of the pairs, the split by node
 * (kq_used_split3), the rows of the pairs; then cleared. */
void rc_kq_moe_prof_single(double *out, double calls)
{
    for (int i = 0; i < 4; ++i) {
        out[i] = calls > 0 ? 1e6 * kq_msb[i] / calls : 0.0;
        kq_msb[i] = 0.0;
    }
}

/* The time in the kernels of each thread (us a call): gate and up, all; then
 * cleared. */
void rc_kq_moe_prof_busy(double *out, int n, double calls)
{
    for (int q = 0; q < n && q < 256; ++q) {
        out[2 * q] = calls > 0 ? 1e6 * kq_mbusy[q][0] / calls : 0.0;
        out[2 * q + 1] = calls > 0 ? 1e6 * kq_mbusy[q][1] / calls : 0.0;
        kq_mbusy[q][0] = kq_mbusy[q][1] = 0.0;
    }
}

void rc_kq_moe_prof(double *out)
{
    for (int i = 0; i < 16; ++i) {
        out[i] = kq_mprof_n ? kq_mprof[i] / (double)kq_mprof_n : 0.0;
        kq_mprof[i] = 0.0;
    }
    out[7] = (double)kq_mprof_n;
    kq_mprof_n = 0;
}

void rc_kq_set_moe_rot(int on)
{
    kq_moe_rot = on;
}
static int kq_calib_on;
static double kq_calib_busy[512];       /* the compute time of each thread */

void rc_kq_set_node0_share(float s)
{
    kq_node0_share = s > 0.05f && s < 0.95f ? s : 0.5f;
}

float rc_kq_get_node0_share(void)
{
    return kq_node0_share;
}

/* The range [lo, hi) of n tasks of the calling thread: the first half of
 * the team takes kq_node0_share of them, evenly, the second half the rest.
 * With an odd team or one thread, an even split. */
/* The experts on the nodes (a machine of two): node 0 holds the expert
 * stacks of mats, all of them (slot0 null) or those with slot0[e] >= 0 at
 * that slot (a split: the experts do not fit one node); node 1 the copy
 * mats1, all (slot1 null) or those with slot1[e] >= 0. The threads of each
 * node take only the tasks of the experts on their node: used in three
 * classes (kq_used_split3), A on node 0 only (and the shared expert), B on
 * both, C on node 1 only; node 0's threads A then B, node 1's C then B. */
/* The static split of a step: node 0's half the first n0 tasks, n0 = its
 * share clamped to [nA, nAB] (nAB = nA + nB), node 1's half the rest. */
static void kq_share_range_strict(int n, int nA, int nAB, int *lo, int *hi)
{
    int nth = omp_get_num_threads(), tid = omp_get_thread_num();
    if (nth < 2 || nth % 2 != 0) {
        *lo = (int)((int64_t)n * tid / nth);
        *hi = (int)((int64_t)n * (tid + 1) / nth);
        return;
    }
    int half = nth / 2, n0 = (int)(n * kq_node0_share + 0.5f);
    n0 = n0 < nA ? nA : n0 > nAB ? nAB : n0;
    if (tid < half) {
        *lo = (int)((int64_t)n0 * tid / half);
        *hi = (int)((int64_t)n0 * (tid + 1) / half);
    } else {
        int r = tid - half;
        *lo = n0 + (int)((int64_t)(n - n0) * r / half);
        *hi = n0 + (int)((int64_t)(n - n0) * (r + 1) / half);
    }
}

/* The next task of a dynamic loop: tasks 0 .. nA - 1 (counter ca), nA ..
 * nAB - 1 (cb), nAB .. n - 1 (cc); node 0's threads ca then cb, node 1's cc
 * then cb. -1 when there are none left for this thread. */
static inline int kq_next_task(int *ca, int nA, int *cb, int nAB, int *cc, int n, int node1)
{
    int x;
    if (!node1) {
        x = __atomic_fetch_add(ca, 1, __ATOMIC_RELAXED);
        if (x < nA) {
            return x;
        }
    } else {
        x = nAB + __atomic_fetch_add(cc, 1, __ATOMIC_RELAXED);
        if (x < n) {
            return x;
        }
    }
    x = nA + __atomic_fetch_add(cb, 1, __ATOMIC_RELAXED);
    return x < nAB ? x : -1;
}

/* The tasks of experts not on its node that a thread ran (a check of the
 * split; rc_kq_moe_strict_violations reads and clears it). */
static long kq_strict_bad;

long rc_kq_moe_strict_violations(void)
{
    long n = __atomic_exchange_n(&kq_strict_bad, 0, __ATOMIC_RELAXED);
    return n;
}

/* kq_next_task, INT_MAX in place of -1 (the end of a loop "x < INT_MAX"). */
static inline int kq_task_or_end(int *ca, int nA, int *cb, int nAB, int *cc, int n, int node1)
{
    int x = kq_next_task(ca, nA, cb, nAB, cc, n, node1);
    return x < 0 ? INT_MAX : x;
}

/* Stable partition of used (nu experts) in the classes A, B, C (see above;
 * has1: there is a copy on node 1); *nA, *nB their counts. tmp: nu ints. */
static void kq_used_split3(int *used, int nu, int experts, const int32_t *slot0, const int32_t *slot1,
                           int has1, int *tmp, int *nA, int *nB)
{
    int w = 0;
    for (int cls = 0; cls < 3; ++cls) {
        for (int u = 0; u < nu; ++u) {
            int e = used[u];
            int on0 = e >= experts || slot0 == NULL || slot0[e] >= 0;
            int on1 = e < experts && has1 && (slot1 == NULL || slot1[e] >= 0);
            int c = on0 && !on1 ? 0 : on0 ? 1 : 2;
            if (c == cls) {
                tmp[w++] = e;
            }
        }
        if (cls == 0) {
            *nA = w;
        } else if (cls == 1) {
            *nB = w - *nA;
        }
    }
    memcpy(used, tmp, (size_t)nu * 4);
}

static void kq_share_range(int n, int *lo, int *hi)
{
    int nth = omp_get_num_threads(), tid = omp_get_thread_num();
    if (nth < 2 || nth % 2 != 0 || kq_node0_share == 0.5f) {
        *lo = (int)((int64_t)n * tid / nth);
        *hi = (int)((int64_t)n * (tid + 1) / nth);
        return;
    }
    int half = nth / 2, n0 = (int)(n * kq_node0_share + 0.5f);
    if (tid < half) {
        *lo = (int)((int64_t)n0 * tid / half);
        *hi = (int)((int64_t)n0 * (tid + 1) / half);
    } else {
        int r = tid - half;
        *lo = n0 + (int)((int64_t)(n - n0) * r / half);
        *hi = n0 + (int)((int64_t)(n - n0) * (r + 1) / half);
    }
}
