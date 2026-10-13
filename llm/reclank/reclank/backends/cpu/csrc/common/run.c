/* The CPU programs that a GPU program runs (GP_CPU_TASK, GP_CPU_JOIN: the
 * cold experts of a step, gg_set_cpu_runner): a team of gp_task_threads
 * threads bound spread over the places. On the 2-socket Xeon (48 cores),
 * Qwen3.8 on the GPU decoded 26.0 tok/s with 24 threads spread, 21.3 with
 * all 48 bound close (OMP_PROC_BIND close, the default of reclank): the
 * threads of the GPU runner and of Python keep cores, and the experts read
 * the memory of both nodes. 0 (the default) takes OMP_NUM_THREADS.
 * rc_set_task_threads sets it (reclank/backends/cuda/runtime.py, RECLANK_GPU_CPU_THREADS). */
static int gp_task_threads;
void rc_set_task_threads(int n)
{
    gp_task_threads = n;
}

/* The team of rc_run_task on a machine of two or more NUMA nodes, pinned
 * with the same count of threads on each node (as far as the CPUs of a node
 * allow; RECLANK_TEAM_BALANCE, default 1), out of RECLANK_RESERVED_CPUS. proc_bind(spread) over the 40
 * CPUs of taskset 0-19,24-43 put 24 threads as 10 on node 0 and 14 on node 1
 * (it takes the places in pairs, then one by one), while kq_share_range
 * gives the tasks of node 0 to the first half of the team. Here thread i <
 * n0 takes a CPU of node 0 (evenly spaced in the allowed CPUs of the node),
 * the others those of node 1, ...; the allowed CPUs are those of the places
 * of OpenMP (OMP_PLACES, made from the affinity of the process when the
 * runtime starts; the main thread of Python is pinned later). A thread pins itself
 * when sched_getcpu is not its CPU: one call at the first region of a team
 * size, then none. 0: proc_bind(spread) only. */
/* The CPUs of the places of OpenMP (all of them; 0 when there are none). */
static int gp_place_cpus(cpu_set_t *m)
{
    CPU_ZERO(m);
    int np = omp_get_num_places(), any = 0;
    for (int p = 0; p < np; ++p) {
        int k = omp_get_place_num_procs(p);
        int ids[256];
        if (k <= 0 || k > 256) {
            continue;
        }
        omp_get_place_proc_ids(p, ids);
        for (int i = 0; i < k; ++i) {
            if (ids[i] >= 0 && ids[i] < CPU_SETSIZE) {
                CPU_SET(ids[i], m);
                any = 1;
            }
        }
    }
    return any;
}

/* The CPUs of the places of OpenMP (the affinity of the process when the
 * runtime started: the taskset, before OMP_PROC_BIND binds the main thread)
 * into out (at most max); returns their count. */
int rc_place_cpus(int *out, int max)
{
    cpu_set_t m;
    int n = 0;
    if (!gp_place_cpus(&m)) {
        return 0;
    }
    for (int c = 0; c < CPU_SETSIZE && n < max; ++c) {
        if (CPU_ISSET(c, &m)) {
            out[n++] = c;
        }
    }
    return n;
}

static int gp_team_balance_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_TEAM_BALANCE");
        on = !(v && v[0] == '0');
    }
    return on;
}

#define GP_TEAM_MAX 256
static int gp_team_cpu[GP_TEAM_MAX];
static int gp_team_n = -1;          /* the size gp_team_cpu is made for; 0: no plan */

/* gp_team_cpu for n threads; return 0 when there is no plan (one node, too
 * few CPUs, or no mask). Called by the master before the region. */
static int gp_team_plan(int n)
{
    if (n == gp_team_n) {
        return gp_team_n > 0;
    }
    gp_team_n = 0;
    cpu_set_t gp_start_mask;
    if (!gp_team_balance_on() || n < 2 || n > GP_TEAM_MAX || !gp_place_cpus(&gp_start_mask)) {
        return 0;
    }
    pthread_once(&kq_cpu_node_once, kq_cpu_node_init);
    /* the allowed CPUs of each node, in order */
    static int cpus[64][GP_TEAM_MAX];
    int cnt[64] = {0}, nodes = 0, total = 0;
    /* RECLANK_RESERVED_CPUS ("1,19"): the CPUs of the main thread and of the
     * copy workers (reclank.backends.cuda.runtime), out of the teams */
    static unsigned char rsv[4096];
    memset(rsv, 0, sizeof(rsv));
    const char *rv = getenv("RECLANK_RESERVED_CPUS");
    for (const char *q = rv; q != NULL && *q;) {
        char *e;
        int k = (int)strtol(q, &e, 10);
        if (e == q) {
            break;
        }
        if (k >= 0 && k < 4096) {
            rsv[k] = 1;
        }
        if (*e != ',') {
            break;
        }
        q = e + 1;
    }
    for (int c = 0; c < 4096 && c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &gp_start_mask) || rsv[c]) {
            continue;
        }
        int nd = kq_cpu_node_tab[c];
        if (nd < 0 || nd >= 64 || cnt[nd] >= GP_TEAM_MAX) {
            continue;
        }
        cpus[nd][cnt[nd]++] = c;
        ++total;
        nodes = nd + 1 > nodes ? nd + 1 : nodes;
    }
    int used = 0;
    for (int nd = 0; nd < nodes; ++nd) {
        used += cnt[nd] > 0;
    }
    if (used < 2 || n > total) {
        return 0;
    }
    /* the same count of threads on each node (n / nodes), a node with too
     * few CPUs giving the rest to the others (the prompt team of 38 with
     * CPUs 1 and 19 reserved: 18 and 20) */
    int want[64] = {0}, left = n, room = 0;
    for (int nd = 0; nd < nodes; ++nd) {
        room += cnt[nd] > 0;
    }
    while (left > 0 && room > 0) {
        int share = (left + room - 1) / room, given = 0;
        for (int nd = 0; nd < nodes && left > 0; ++nd) {
            int free_nd = cnt[nd] - want[nd];
            if (free_nd <= 0) {
                continue;
            }
            int g = share < free_nd ? share : free_nd;
            g = g < left ? g : left;
            want[nd] += g;
            left -= g;
            given += g;
        }
        room = 0;
        for (int nd = 0; nd < nodes; ++nd) {
            room += cnt[nd] > want[nd];
        }
        if (given == 0) {
            break;
        }
    }
    int k = 0;
    for (int nd = 0; nd < nodes; ++nd) {
        int m = want[nd];
        for (int i = 0; i < m && k < n; ++i) {
            gp_team_cpu[k++] = cpus[nd][(int)((int64_t)i * cnt[nd] / m)];
        }
    }
    if (k != n) {
        return 0;
    }
    gp_team_n = n;
    return 1;
}

/* In the region: pin this thread to its CPU of the plan, if it is not there. */
static inline void gp_team_pin(void)
{
    int tid = omp_get_thread_num(), want = gp_team_cpu[tid];
    if (sched_getcpu() != want) {
        cpu_set_t m;
        CPU_ZERO(&m);
        CPU_SET(want, &m);
        sched_setaffinity(0, sizeof(m), &m);
    }
}

/* A program that starts with GP_RUN_PARTS: nparts, team, cpus (int32,
 * nparts x team). The program runs in one flat team of nparts x team
 * threads, thread i pinned to cpus[i] (the GPU+parts mode of QwenGPU: the
 * first team CPUs of each NUMA node; GP_KQ_MOE_PART gives each part its
 * share). A flat team: OpenMP keeps its threads from a run to the next
 * (nested teams of rc_run_parts were made again at each run: 5.4 ms). */
static int gp_run_parts_task(const int64_t *prog, const gp_rec *r, int limit)
{
    const int nparts = (int)r->v[0], team = (int)r->v[1];
    const int32_t *cpus = (const int32_t *)(intptr_t)r->v[2];
    if (nparts < 1 || team < 1) {
        return -1;
    }
    gp_fail = 0;
    int was = gp_planned;
    gp_planned = 1;         /* a planned team (rc_team_warn) */
    #pragma omp parallel num_threads(nparts * team)
    {
        int want = cpus[omp_get_thread_num()];
        if (want >= 0 && sched_getcpu() != want) {
            cpu_set_t m;
            CPU_ZERO(&m);
            CPU_SET(want, &m);
            sched_setaffinity(0, sizeof(m), &m);
        }
        gp_exec(prog, limit);
    }
    gp_planned = was;
    return gp_fail == 2 ? -4 : gp_fail ? -3 : 0;
}

int rc_run_task(const int64_t *prog, int limit)
{
    if (prog[0] != GP_MAGIC) {
        return -1;
    }
    if (prog[2] >= 1 && ((const gp_rec *)(prog + 4 + prog[1]))->op == GP_RUN_PARTS) {
        return gp_run_parts_task(prog, (const gp_rec *)(prog + 4 + prog[1]), limit);
    }
    /* word 3 of the header: the team of this program, or 0 (program.Program
     * threads; the CPU part of a mixed prompt group takes more threads than
     * the step) */
    int n = prog[3] > 0 ? (int)prog[3] : gp_task_threads;
    if (n <= 0) {
        return rc_run(prog, limit);
    }
    gp_fail = 0;
    int plan = gp_team_plan(n);
    int was = gp_planned;
    gp_planned = 1;
    #pragma omp parallel num_threads(n) proc_bind(spread)
    {
        if (plan && omp_get_num_threads() == n) {
            gp_team_pin();
        }
        gp_exec(prog, limit);
    }
    gp_planned = was;
    return gp_fail == 2 ? -4 : gp_fail ? -3 : 0;
}

/* Run nparts programs at the same time. Each program runs in a team of team
 * threads. The array progs holds the address of each program. The outer
 * region has one thread for each part, spread over the places of OMP_PLACES.
 * Each of those threads opens a team on the places near it.
 *
 * A team of 0 divides the threads of OMP_NUM_THREADS by the parts. The array
 * bar is the barrier of the programs (see gp_xbar). This function sets it to
 * 0 first. */
/* rc_run_parts with the time of each record of each part (gp_exec_prof):
 * part p adds to ms[p * stride + pc]. With rc_xbar_stats, this gives the
 * time of each operation of each part and the wait at the barriers. */
int rc_run_parts_prof(const int64_t *const *progs, int nparts, int team, int64_t *bar,
                         double *ms, int stride)
{
    for (int p = 0; p < nparts; ++p) {
        if (progs[p][0] != GP_MAGIC) {
            return -1;
        }
    }
    if (team <= 0) {
        team = omp_get_max_threads() / nparts;
        if (team < 1) {
            team = 1;
        }
    }
    for (int p = 0; p <= nparts; ++p) {
        bar[8 * p] = 0;
    }
    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(nparts) proc_bind(spread)
    {
        const int p = omp_get_thread_num();
        #pragma omp parallel num_threads(team) proc_bind(close)
        {
            gp_xcnt = 0;
            gp_exec_prof(progs[p], ms + (size_t)p * (size_t)stride);
        }
    }
    return 0;
}

int rc_run_parts(const int64_t *const *progs, int nparts, int team, int64_t *bar)
{
    for (int p = 0; p < nparts; ++p) {
        if (progs[p][0] != GP_MAGIC) {
            return -1;
        }
    }
    if (team <= 0) {
        team = omp_get_max_threads() / nparts;
        if (team < 1) {
            team = 1;
        }
    }
    for (int p = 0; p <= nparts; ++p) {
        bar[8 * p] = 0;
    }
    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(nparts) proc_bind(spread)
    {
        const int64_t *prog = progs[omp_get_thread_num()];
        #pragma omp parallel num_threads(team) proc_bind(close)
        {
            gp_xcnt = 0;
            gp_exec(prog, -1);
        }
    }
    return 0;
}

extern int sched_getcpu(void);

/* The CPU of each thread of rc_run_parts, with the same teams: cpus[p *
 * team + t] for thread t of part p, or -1 for a thread that did not run. The
 * threads are bound (OMP_PLACES, OMP_PROC_BIND), so a run of the parts uses
 * these CPUs too. reclank/runtime/numa.py gives each part the node of its CPUs.
 * Return the team. */
int rc_part_cpus(int nparts, int team, int *cpus)
{
    if (team <= 0) {
        team = omp_get_max_threads() / nparts;
        if (team < 1) {
            team = 1;
        }
    }
    for (int i = 0; i < nparts * team; ++i) {
        cpus[i] = -1;
    }
    omp_set_max_active_levels(2);
    #pragma omp parallel num_threads(nparts) proc_bind(spread)
    {
        const int p = omp_get_thread_num();
        #pragma omp parallel num_threads(team) proc_bind(close)
        {
            const int t = omp_get_thread_num();
            if (t < team) {
                cpus[p * team + t] = sched_getcpu();
            }
        }
    }
    return team;
}
