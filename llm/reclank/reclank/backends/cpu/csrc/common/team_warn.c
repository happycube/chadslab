/* ---- teams outside the planned ones (rc_team_warn) ----
 *
 * Every parallel region of this library starts in GOMP_parallel; the build
 * links it with -Wl,--wrap=GOMP_parallel, so each comes here first. The
 * planned teams (rc_run_task: the CPU experts of a step and of a mixed
 * group; rc_run_parts; rc_run: the CPU model) mark themselves
 * (gp_planned). Any other region of a team of gp_team_warn_min threads or
 * more at the outer level, once warnings are on (rc_team_warn: the
 * servers turn them on after the load), is reported once for each place of
 * the call: on the 2-socket Xeon such a team of 40 threads (rc_kq_rows of the 64
 * n-gram rows of an MTP verify group) waited 5.7 ms a call for cores that the
 * team of the CPU experts spins on. rc_team_warn_stats gives the places
 * and the counts. */
static __thread int gp_planned;
static int gp_team_warn_min;
#define GP_WARN_SITES 64
static void *gp_warn_site[GP_WARN_SITES];
static long gp_warn_count[GP_WARN_SITES];
static int gp_warn_nsites;
static pthread_mutex_t gp_warn_mu = PTHREAD_MUTEX_INITIALIZER;

void rc_team_warn(int min_threads)
{
    gp_team_warn_min = min_threads;
}

/* The places (return addresses) and counts of the reported teams; returns
 * their number. */
int rc_team_warn_stats(void **sites, long *counts, int max)
{
    pthread_mutex_lock(&gp_warn_mu);
    int n = gp_warn_nsites < max ? gp_warn_nsites : max;
    for (int i = 0; i < n; ++i) {
        sites[i] = gp_warn_site[i];
        counts[i] = gp_warn_count[i];
    }
    pthread_mutex_unlock(&gp_warn_mu);
    return n;
}

static void gp_team_report(void *ra, unsigned n)
{
    int first = 0;
    pthread_mutex_lock(&gp_warn_mu);
    int i = 0;
    while (i < gp_warn_nsites && gp_warn_site[i] != ra) {
        ++i;
    }
    if (i == gp_warn_nsites && i < GP_WARN_SITES) {
        gp_warn_site[i] = ra;
        gp_warn_nsites++;
        first = 1;
    }
    if (i < GP_WARN_SITES) {
        gp_warn_count[i]++;
    }
    pthread_mutex_unlock(&gp_warn_mu);
    if (first) {
        Dl_info d;
        const char *s = "?";
        long off = 0;
        if (dladdr(ra, &d) && d.dli_sname != NULL) {
            s = d.dli_sname;
            off = (long)((char *)ra - (char *)d.dli_saddr);
        }
        fprintf(stderr, "[reclank] WARNING: an OpenMP team of %u threads outside the planned teams, "
                "from %s+%#lx (cpu %d); it competes with the cores of the CPU experts "
                "(reported once a place; rc_team_warn)\n", n, s, off, sched_getcpu());
    }
}

extern void __real_GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads,
                                 unsigned flags);
void __wrap_GOMP_parallel(void (*fn)(void *), void *data, unsigned num_threads, unsigned flags)
{
    if (gp_team_warn_min > 0 && !gp_planned && omp_get_level() == 0) {
        unsigned n = num_threads ? num_threads : (unsigned)omp_get_max_threads();
        if ((int)n >= gp_team_warn_min) {
            gp_team_report(__builtin_return_address(0), n);
        }
    }
    __real_GOMP_parallel(fn, data, num_threads, flags);
}
