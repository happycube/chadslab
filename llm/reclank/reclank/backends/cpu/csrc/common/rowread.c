/* Rows of tables in files read straight into memory (DeepSeek V4.1's engram rows: each token's 24 rows of a table
 * of 98 GB on the NVMe, and their scales, at random). Fancy indexing of a map of the file faults one row at a time,
 * and each fault reads around it (128 KB: 175 KB read from the NVMe a row of 264 bytes, the page cache filled with
 * rows no one asks for). Here the reads go through the page cache (its pages stay for the next prompt), many in
 * flight, from a pool of workers made once (rc_rows_pool_new) on the CPUs given (the process's spare ones, not the
 * one CPU the calling thread is bound to): each with an io_uring of qd entries (raw syscalls, IORING_OP_READ; the
 * reads of pages not cached go to the ring's io-wq workers, made on the worker's CPUs), or, when io_uring cannot
 * be set up (ENOSYS, EPERM, kernel.io_uring_disabled) or is not asked for, by pread one read after another. The
 * real 16K prompt's 1.54M reads on 47 CPUs, cold: 4 rings of 64 2.84 s and 10.9 CPU-s, 32 pread workers 4.03 s and
 * 30 CPU-s, 64 3.24 s and 56 CPU-s (the NVMe's ~540K reads a second). rc_rows_pread: the reads by threads made for
 * the call (the first form). */
#include <errno.h>
#include <linux/io_uring.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef struct {
    const int *fd;
    const int64_t *off, *len, *dst;
    int64_t n, k0, step;
    const int *cpus;
    int ncpu, pin, err;
} rc_rowread_job;

static void rc_rowread_pin(const int *cpus, int ncpu)
{
    if (ncpu > 0) {
        cpu_set_t set;
        CPU_ZERO(&set);
        for (int c = 0; c < ncpu; ++c) {
            CPU_SET(cpus[c], &set);
        }
        pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
}

/* read k (from its byte got on) by pread; 0 or an errno */
static int rc_rowread_one(const int *fd, const int64_t *off, const int64_t *len, const int64_t *dst, int64_t k,
                          int64_t got)
{
    uint8_t *d = (uint8_t *)(intptr_t)dst[k];
    while (got < len[k]) {
        ssize_t r = pread(fd[k], d + got, (size_t)(len[k] - got), (off_t)(off[k] + got));
        if (r <= 0) {
            return r < 0 ? errno : EIO;
        }
        got += r;
    }
    return 0;
}

static void *rc_rowread_worker(void *arg)
{
    rc_rowread_job *j = (rc_rowread_job *)arg;
    if (j->pin) {
        rc_rowread_pin(j->cpus, j->ncpu);
    }
    for (int64_t k = j->k0; k < j->n && !j->err; k += j->step) {
        j->err = rc_rowread_one(j->fd, j->off, j->len, j->dst, k, 0);
    }
    return NULL;
}

#define RC_ROWREAD_MAX 256

/* n reads: len[k] bytes of file fd[k] at off[k] into the address dst[k], by `threads` workers made for the call (at
 * most RC_ROWREAD_MAX; worker w takes the reads k = w + threads m). 0, or the errno of a failed read. */
int rc_rows_pread(const int *fd, const int64_t *off, const int64_t *len, const int64_t *dst, int64_t n, int threads,
                  const int *cpus, int ncpu)
{
    if (n <= 0) {
        return 0;
    }
    threads = threads < 1 ? 1 : threads > RC_ROWREAD_MAX ? RC_ROWREAD_MAX : threads;
    if (threads > n) {
        threads = (int)n;
    }
    rc_rowread_job jobs[RC_ROWREAD_MAX];
    pthread_t tid[RC_ROWREAD_MAX];
    int made[RC_ROWREAD_MAX] = {0};
    for (int t = 0; t < threads; ++t) {
        jobs[t] = (rc_rowread_job){fd, off, len, dst, n, t, threads, cpus, ncpu, t > 0, 0};
        made[t] = t > 0 && pthread_create(&tid[t], NULL, rc_rowread_worker, &jobs[t]) == 0;
    }
    /* (the caller's share on its own CPU; a worker that did not start, its share after) */
    for (int t = 0; t < threads; ++t) {
        if (!made[t]) {
            jobs[t].pin = 0;
            rc_rowread_worker(&jobs[t]);
        }
    }
    int err = 0;
    for (int t = 0; t < threads; ++t) {
        if (made[t]) {
            pthread_join(tid[t], NULL);
        }
        err = err ? err : jobs[t].err;
    }
    return err;
}

/* ---- the pool: workers made once, each with an io_uring (or pread) */

typedef struct {
    int fd;
    unsigned *sq_tail, *sq_mask, *sq_array, *cq_head, *cq_tail, *cq_mask;
    struct io_uring_sqe *sqes;
    struct io_uring_cqe *cqes;
} rc_ring;

/* an io_uring of entries by raw syscalls; 0, or an errno (ENOSYS, EPERM: no io_uring here) */
static int rc_ring_init(rc_ring *r, unsigned entries)
{
    struct io_uring_params p;
    memset(&p, 0, sizeof(p));
    int fd = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (fd < 0) {
        return errno;
    }
    size_t sq_sz = p.sq_off.array + p.sq_entries * sizeof(unsigned);
    size_t cq_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    uint8_t *sq = mmap(0, sq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    uint8_t *cq = mmap(0, cq_sz, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    void *sqes = mmap(0, p.sq_entries * sizeof(struct io_uring_sqe), PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (sq == MAP_FAILED || cq == MAP_FAILED || sqes == MAP_FAILED) {
        close(fd);
        return ENOMEM;
    }
    r->fd = fd;
    r->sq_tail = (unsigned *)(sq + p.sq_off.tail);
    r->sq_mask = (unsigned *)(sq + p.sq_off.ring_mask);
    r->sq_array = (unsigned *)(sq + p.sq_off.array);
    r->cq_head = (unsigned *)(cq + p.cq_off.head);
    r->cq_tail = (unsigned *)(cq + p.cq_off.tail);
    r->cq_mask = (unsigned *)(cq + p.cq_off.ring_mask);
    r->cqes = (struct io_uring_cqe *)(cq + p.cq_off.cqes);
    r->sqes = (struct io_uring_sqe *)sqes;
    return 0;
}

/* the reads k = k0 + step m through ring r, qd in flight (the rest of a short read by pread); 0 or an errno. The SQEs
 * written and not yet taken by the kernel (queued) go again with the next io_uring_enter, only those it took count
 * in flight (it takes fewer under memory pressure: EAGAIN, EBUSY, a short count). On an error no read is added, and
 * the ring is left with nothing queued or in flight: the reads in flight are reaped (none writes into dst after the
 * return, no CQE is left for the next job); *broken when that cannot be done (an io_uring_enter that fails for good
 * with SQEs queued or reads in flight: the worker gives up its ring for pread). */
static int rc_ring_reads(rc_ring *r, int qd, const int *fd, const int64_t *off, const int64_t *len, const int64_t *dst,
                         int64_t n, int64_t k0, int64_t step, int *broken)
{
    int64_t next = k0, inflight = 0, queued = 0;
    int err = 0;
    *broken = 0;
    while ((next < n && !err) || queued > 0 || inflight > 0) {
        unsigned tail = *r->sq_tail;
        while (!err && next < n && inflight + queued < qd) {
            struct io_uring_sqe *e = &r->sqes[tail & *r->sq_mask];
            memset(e, 0, sizeof(*e));
            e->opcode = IORING_OP_READ;
            e->fd = fd[next];
            e->off = (uint64_t)off[next];
            e->addr = (uint64_t)dst[next];
            e->len = (unsigned)len[next];
            e->user_data = (uint64_t)next;
            r->sq_array[tail & *r->sq_mask] = tail & *r->sq_mask;
            ++tail;
            ++queued;
            next += step;
        }
        __atomic_store_n(r->sq_tail, tail, __ATOMIC_RELEASE);
        /* (wait for a completion only with reads in flight or about to be) */
        int ret = (int)syscall(__NR_io_uring_enter, r->fd, (unsigned)queued, inflight + queued > 0 ? 1 : 0,
                               IORING_ENTER_GETEVENTS, NULL, 0);
        if (ret >= 0) {
            queued -= ret;
            inflight += ret;
        } else if (errno == EAGAIN || errno == EBUSY) {
            /* (the kernel took none: reap what completed, then again; with none in flight, after a pause) */
            if (inflight == 0) {
                sched_yield();
            }
        } else if (errno != EINTR) {
            err = err ? err : errno;
            if (queued > 0 || inflight > 0) {
                /* (the ring cannot be drained: its worker closes it, the ring's exit cancels and waits for the
                 * reads in flight) */
                *broken = 1;
                return err;
            }
        }
        unsigned head = *r->cq_head;
        while (head != __atomic_load_n(r->cq_tail, __ATOMIC_ACQUIRE)) {
            struct io_uring_cqe *c = &r->cqes[head & *r->cq_mask];
            int64_t k = (int64_t)c->user_data;
            if (c->res < 0) {
                err = err ? err : -c->res;
            } else if (c->res < len[k]) {
                int e2 = rc_rowread_one(fd, off, len, dst, k, c->res);
                err = err ? err : e2;
            }
            --inflight;
            ++head;
        }
        __atomic_store_n(r->cq_head, head, __ATOMIC_RELEASE);
    }
    return err;
}

typedef struct rc_rows_pool rc_rows_pool;
typedef struct {
    rc_rows_pool *pool;
    int index, uring;
    rc_ring ring;
    pthread_t tid;
} rc_rows_worker;

struct rc_rows_pool {
    int nw, qd, uring;
    pthread_mutex_t job;            /* one read of the pool at a time (another thread's waits) */
    pthread_mutex_t mu;
    pthread_cond_t go, done;
    int64_t gen;
    int active, left, err, ready;
    const int *fd;
    const int64_t *off, *len, *dst;
    int64_t n;
    int *cpus, ncpu;
    rc_rows_worker w[];
};

static void *rc_pool_main(void *arg)
{
    rc_rows_worker *w = (rc_rows_worker *)arg;
    rc_rows_pool *p = w->pool;
    rc_rowread_pin(p->cpus, p->ncpu);
    pthread_setname_np(pthread_self(), "rc-rows");
    /* (the ring made here: its io-wq workers on this thread's CPUs) */
    w->uring = p->uring && rc_ring_init(&w->ring, (unsigned)p->qd) == 0;
    pthread_mutex_lock(&p->mu);
    ++p->ready;
    pthread_cond_broadcast(&p->done);
    int64_t seen = p->gen;
    for (;;) {
        while (p->gen == seen) {
            pthread_cond_wait(&p->go, &p->mu);
        }
        seen = p->gen;
        if (w->index >= p->active) {
            continue;
        }
        const int *fd = p->fd;
        const int64_t *off = p->off, *len = p->len, *dst = p->dst;
        int64_t n = p->n, step = p->active;
        pthread_mutex_unlock(&p->mu);
        int err = 0;
        if (w->uring) {
            int broken = 0;
            err = rc_ring_reads(&w->ring, p->qd, fd, off, len, dst, n, w->index, step, &broken);
            if (broken) {
                /* (a ring that could not be drained: closed, its worker by pread from now on) */
                close(w->ring.fd);
                w->uring = 0;
            }
        } else {
            for (int64_t k = w->index; k < n && !err; k += step) {
                err = rc_rowread_one(fd, off, len, dst, k, 0);
            }
        }
        pthread_mutex_lock(&p->mu);
        p->err = p->err ? p->err : err;
        if (--p->left == 0) {
            pthread_cond_broadcast(&p->done);
        }
    }
    return NULL;
}

/* A pool of nw workers (at most RC_ROWREAD_MAX) on cpus, each with an io_uring of qd entries when uring (else, or
 * when io_uring cannot be set up, by pread); NULL when no worker starts. *mode: 1 when every worker has its ring,
 * else 0 (pread). */
void *rc_rows_pool_new(int nw, int uring, int qd, const int *cpus, int ncpu, int *mode)
{
    nw = nw < 1 ? 1 : nw > RC_ROWREAD_MAX ? RC_ROWREAD_MAX : nw;
    rc_rows_pool *p = calloc(1, sizeof(rc_rows_pool) + (size_t)nw * sizeof(rc_rows_worker));
    p->qd = qd < 1 ? 1 : qd;
    p->uring = uring;
    p->cpus = malloc((size_t)(ncpu > 0 ? ncpu : 1) * sizeof(int));
    memcpy(p->cpus, cpus, (size_t)(ncpu > 0 ? ncpu : 0) * sizeof(int));
    p->ncpu = ncpu;
    pthread_mutex_init(&p->job, NULL);
    pthread_mutex_init(&p->mu, NULL);
    pthread_cond_init(&p->go, NULL);
    pthread_cond_init(&p->done, NULL);
    int made = 0;
    pthread_mutex_lock(&p->mu);
    for (int i = 0; i < nw; ++i) {
        p->w[i].pool = p;
        p->w[i].index = i;
        if (pthread_create(&p->w[i].tid, NULL, rc_pool_main, &p->w[i]) != 0) {
            break;
        }
        ++made;
    }
    p->nw = made;
    while (p->ready < made) {
        pthread_cond_wait(&p->done, &p->mu);
    }
    int all = made > 0;
    for (int i = 0; i < made; ++i) {
        all &= p->w[i].uring;
    }
    pthread_mutex_unlock(&p->mu);
    if (made == 0) {
        free(p->cpus);
        free(p);
        return NULL;
    }
    if (mode) {
        *mode = uring && all ? 1 : 0;
    }
    return p;
}

/* The n reads of rc_rows_pread by the first `workers` of the pool, worker w the reads k = w + workers m (one job of
 * the pool at a time: a caller of another thread waits); 0, or the errno of a failed read. */
int rc_rows_pool_read(void *handle, const int *fd, const int64_t *off, const int64_t *len, const int64_t *dst,
                      int64_t n, int workers)
{
    rc_rows_pool *p = (rc_rows_pool *)handle;
    if (n <= 0) {
        return 0;
    }
    pthread_mutex_lock(&p->job);
    pthread_mutex_lock(&p->mu);
    int a = workers < 1 ? 1 : workers > p->nw ? p->nw : workers;
    p->active = (int64_t)a > n ? (int)n : a;
    p->fd = fd;
    p->off = off;
    p->len = len;
    p->dst = dst;
    p->n = n;
    p->err = 0;
    p->left = p->active;
    ++p->gen;
    pthread_cond_broadcast(&p->go);
    while (p->left > 0) {
        pthread_cond_wait(&p->done, &p->mu);
    }
    int err = p->err;
    pthread_mutex_unlock(&p->mu);
    pthread_mutex_unlock(&p->job);
    return err;
}
