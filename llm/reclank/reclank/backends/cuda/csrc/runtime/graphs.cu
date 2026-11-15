/* Run a record that moves work between the GPU and the CPU.
 *
 * GP_TO_HOST: (device source, host target, bytes) three times, then the
 * index of an event. Null sources are not copied.
 * GP_CPU_JOIN: the address of a CPU program, then the index of the event.
 * GP_TO_DEV: host source, device target, bytes. */
static int gg_boundary(gg_prog *g, const gp_rec *r)
{
    switch (r->op) {
    case GP_FETCH: {
        if (gg_fetch_init() != 0) {
            return -1;
        }
        gg_job j;
        j.ranges = (const int64_t *)(intptr_t)r->v[0];
        j.count = (int)r->v[1];
        j.f = (int)r->v[2];
        j.b = (int)r->v[3];
        /* operands 4-6 (or none): the mode (1: KQ_NV4 rows into NV4T; 2: DMA, no pull kernel), the columns of the
         * even and odd ranges */
        j.mode = r->tag[4] != GP_T_NONE ? (int)r->v[4] : 0;
        j.c0 = r->tag[5] != GP_T_NONE ? (int)r->v[5] : 0;
        j.c1 = r->tag[6] != GP_T_NONE ? (int)r->v[6] : 0;
        pthread_mutex_lock(&gg_mu);
        gg_recorded[j.f] = 0;
        gg_jobs[gg_job_tail % GG_FETCH_MAX] = j;
        ++gg_job_tail;
        pthread_cond_broadcast(&gg_cv);
        pthread_mutex_unlock(&gg_mu);
        return 0;
    }
    case GP_FETCH_WAIT: {
        int f = (int)r->v[0];
        /* The first event of the measure goes before the wait of the host:
         * the copies are from pageable memory, so the worker records the job
         * only near the end of its copies, and the host waits here too. */
        int k = gg_ms_nf + gg_ms_nj;
        int ms = gg_ms_on && k < GG_MS_MAX;
        if (ms) {
            CK(cudaEventRecord(gg_ms_ev[2 * k], gg_stream));
        }
        pthread_mutex_lock(&gg_mu);
        while (!gg_recorded[f]) {
            pthread_cond_wait(&gg_cv, &gg_mu);
        }
        int bad = gg_fetch_error;
        pthread_mutex_unlock(&gg_mu);
        if (bad) {
            snprintf(gg_error, sizeof(gg_error), "a copy of the weights of the experts failed: %s", gg_fetch_why);
            return -1;
        }
        if (ms) {
            CK(cudaStreamWaitEvent(gg_stream, gg_ready[f], 0));
            CK(cudaEventRecord(gg_ms_ev[2 * k + 1], gg_stream));
            gg_ms_kind[k] = 0;
            ++gg_ms_nf;
            return 0;
        }
        CK(cudaStreamWaitEvent(gg_stream, gg_ready[f], 0));
        return 0;
    }
    case GP_FETCH_DONE:
        CK(cudaEventRecord(gg_freeev[r->v[0]], gg_stream));
        return 0;
    case GP_TO_HOST:
        for (int k = 0; k < 3; ++k) {
            if (r->v[3 * k] != 0) {
                CK(cudaMemcpyAsync((void *)(intptr_t)r->v[3 * k + 1],
                                   (const void *)(intptr_t)r->v[3 * k],
                                   (size_t)r->v[3 * k + 2], cudaMemcpyDeviceToHost, gg_stream));
            }
        }
        CK(cudaEventRecord(g->ev[r->v[9]], gg_stream));
        return 0;
    case GP_CPU_JOIN: {
        int k = gg_ms_nf + gg_ms_nj;
        int ms = gg_ms_on && k < GG_MS_MAX;
        if (ms) {
            CK(cudaEventRecord(gg_ms_ev[2 * k], gg_stream));
        }
        CK(cudaEventSynchronize(g->ev[r->v[1]]));
        double t0 = gg_clock();
        if (gg_cpu_run == NULL || gg_cpu_run((const int64_t *)(intptr_t)r->v[0], -1) != 0) {
            snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_JOIN failed");
            return -1;
        }
        if (ms) {
            /* The event after the CPU program: the stream waited from the end
             * of its work before (the event above) to here. */
            gg_ms_cpu[gg_ms_nj] = gg_clock() - t0;
            CK(cudaEventRecord(gg_ms_ev[2 * k + 1], gg_stream));
            gg_ms_kind[k] = 1;
            ++gg_ms_nj;
        }
        return 0;
    }
    case GP_TO_DEV:
        CK(cudaMemcpyAsync((void *)(intptr_t)r->v[1], (const void *)(intptr_t)r->v[0],
                           (size_t)r->v[2], cudaMemcpyHostToDevice, gg_stream));
        return 0;
    case GP_CPU_START:
        CK(cudaEventSynchronize(g->ev[r->v[1]]));
        pthread_mutex_lock(&gg_hmu);
        if (!gg_hon) {
            pthread_t th;
            if (pthread_create(&th, NULL, gg_helper, NULL) != 0) {
                pthread_mutex_unlock(&gg_hmu);
                snprintf(gg_error, sizeof(gg_error), "no helper thread for GP_CPU_START");
                return -1;
            }
            pthread_detach(th);
            gg_hon = 1;
        }
        while (gg_hbusy) {
            pthread_cond_wait(&gg_hcv, &gg_hmu);
        }
        gg_hbusy = 1;
        gg_hjob = (const int64_t *)(intptr_t)r->v[0];
        pthread_cond_broadcast(&gg_hcv);
        pthread_mutex_unlock(&gg_hmu);
        return 0;
    case GP_CPU_WAIT: {
        /* The measure: as for GP_CPU_JOIN. */
        int k = gg_ms_nf + gg_ms_nj;
        int ms = gg_ms_on && k < GG_MS_MAX;
        if (ms) {
            CK(cudaEventRecord(gg_ms_ev[2 * k], gg_stream));
        }
        pthread_mutex_lock(&gg_hmu);
        while (gg_hbusy) {
            pthread_cond_wait(&gg_hcv, &gg_hmu);
        }
        int bad = gg_hfail;
        gg_hfail = 0;
        double dt = gg_hlast;
        pthread_mutex_unlock(&gg_hmu);
        if (ms) {
            gg_ms_cpu[gg_ms_nj] = dt;
            CK(cudaEventRecord(gg_ms_ev[2 * k + 1], gg_stream));
            gg_ms_kind[k] = 1;
            ++gg_ms_nj;
        }
        if (bad) {
            snprintf(gg_error, sizeof(gg_error), "the CPU program of a GP_CPU_START failed");
            return -1;
        }
        return 0;
    }
    default:
        return -1;
    }
}

/* Launch the kernels of a segment one at a time. With tasks, a GP_CPU_TASK
 * runs here, in its place (the launches without a graph); else the runner
 * runs it after the launches (gg_run_tasks). */
static int gg_launch_seg(gg_prog *g, const gg_seg *sg, int tasks)
{
    for (int pc = sg->start; pc < sg->end; ++pc) {
        const gp_rec *r = g->hcode + pc;
        gg_at(g, pc, r->op, sg->start, sg->end);
        if (r->op == GP_CPU_TASK && tasks) {
            if (gg_cpu_task(r, g->henv) != 0) {
                return -1;
            }
            continue;
        }
        if (gg_launch(g, r, g->dcode + pc, g->denv) != 0) {
            return -1;
        }
        GG_SYNC_AT(pc, r->op);
    }
    return 0;
}

/* Run the GP_CPU_TASK records from *from to to, in order (the graphs of
 * these records are on the stream). */
static int gg_run_tasks(gg_prog *g, int *from, int to)
{
    for (int pc = *from; pc < to; ++pc) {
        const gp_rec *r = g->hcode + pc;
        if (r->op == GP_CPU_TASK && gg_cpu_task(r, g->henv) != 0) {
            return -1;
        }
    }
    *from = to;
    return 0;
}

/* Run the segments and the boundary records in order. With the graph, the
 * first run records the graph of each segment. */
#define GG_SEG_FIRST 32
#define GG_SEG_MAX 160

/* RECLANK_GPU_PDL=0 keeps the ordinary edges, for a test. A GPU before
 * sm_90 has no programmatic edges. */
static int gg_pdl_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_PDL");
        on = !(v && v[0] == '0') && gg_caps.pdl;
    }
    return on;
}

/* Change each edge from a kernel to a kernel into a programmatic edge. The
 * kernels start with PDL_START, so each waits for its data. */
#if CUDART_VERSION >= 13000
#define RC_GRAPH_GET_EDGES cudaGraphGetEdges
#define RC_GRAPH_REMOVE_DEPS cudaGraphRemoveDependencies
#define RC_GRAPH_ADD_DEPS cudaGraphAddDependencies
#elif CUDART_VERSION >= 12030
#define RC_GRAPH_GET_EDGES cudaGraphGetEdges_v2
#define RC_GRAPH_REMOVE_DEPS cudaGraphRemoveDependencies_v2
#define RC_GRAPH_ADD_DEPS cudaGraphAddDependencies_v2
#endif

#if defined(RC_GRAPH_GET_EDGES)
static cudaError_t gg_pdl_edges(cudaGraph_t graph)
{
    size_t n = 0;
    cudaError_t err = RC_GRAPH_GET_EDGES(graph, NULL, NULL, NULL, &n);
    if (err != cudaSuccess || n == 0) {
        return err;
    }
    cudaGraphNode_t *from = (cudaGraphNode_t *)malloc(n * sizeof(cudaGraphNode_t));
    cudaGraphNode_t *to = (cudaGraphNode_t *)malloc(n * sizeof(cudaGraphNode_t));
    cudaGraphEdgeData *ed = (cudaGraphEdgeData *)malloc(n * sizeof(cudaGraphEdgeData));
    err = RC_GRAPH_GET_EDGES(graph, from, to, ed, &n);
    for (size_t i = 0; i < n && err == cudaSuccess; ++i) {
        cudaGraphNodeType ta, tb;
        if (cudaGraphNodeGetType(from[i], &ta) != cudaSuccess ||
            cudaGraphNodeGetType(to[i], &tb) != cudaSuccess ||
            ta != cudaGraphNodeTypeKernel || tb != cudaGraphNodeTypeKernel ||
            ed[i].type != cudaGraphDependencyTypeDefault) {
            continue;
        }
        err = RC_GRAPH_REMOVE_DEPS(graph, &from[i], &to[i], &ed[i], 1);
        if (err == cudaSuccess) {
            cudaGraphEdgeData e;
            memset(&e, 0, sizeof(e));
            e.from_port = cudaGraphKernelNodePortProgrammatic;
            e.type = cudaGraphDependencyTypeProgrammatic;
            err = RC_GRAPH_ADD_DEPS(graph, &from[i], &to[i], &e, 1);
        }
    }
    free(from);
    free(to);
    free(ed);
    return err;
}
#else
/* A toolkit before 12.3 has no edge data in the graph API: no programmatic
 * edges (PDL off), the ordinary edges stay. */
static cudaError_t gg_pdl_edges(cudaGraph_t graph)
{
    (void)graph;
    return cudaSuccess;
}
#endif

/* ---------- the chains: the layers of a kind on one graph, tail launches from one layer to the next ----------
 * A program's records mark its layers (GP_F_LAYER on a layer's first record, GP_F_LAYER_END on the first after the
 * last). A run of layers with no boundary record inside is a chain (gg_find_links). Each layer's records are
 * captured as a graph whose kernels read their records in one area, dbody, instead of the layer's place in dcode,
 * and k_tail_next goes last. Two layers whose graphs are the same (the same nodes: kernels, grids, blocks, shared
 * memory and the bytes of every argument; gg_graph_cmp) run on one graph, the body of their kind: what differs is
 * only the records, which k_tail_next copies into dbody before it tail-launches the next layer's body. A kernel
 * that takes a layer's pointer as an argument (not through its record) makes that layer's graph differ, and the
 * layer gets a body of its own: the bits never change, only the memory of the graphs. The host launches
 * k_tail_start and the chain's first body; the stream counts the tail launches as part of that launch.
 * RECLANK_GPU_TAIL=0: the graphs as before; RECLANK_GPU_TAIL_LOG=1: each chain's bodies, and what made a body
 * differ from the one most like it. */
static int gg_tail_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_TAIL");
#if CUDART_VERSION >= 12040
        on = !(v && v[0] == '0');
#else
        (void)v;
        on = 0;     /* (no cudaFuncGetParamInfo: no compare of the graphs) */
#endif
    }
    return on;
}

/* RECLANK_GPU_TAIL_MIN: the fewest layers of a chain (a shorter run of layers stays in ordinary graphs);
 * RECLANK_GPU_TAIL_PLAIN=1: each chain an ordinary graph (the cuts at the layers kept: a measure) */
static int gg_tail_min(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_GPU_TAIL_MIN");
        n = v ? atoi(v) : 1;
        n = n < 1 ? 1 : n;
    }
    return n;
}

/* A program that fetches weights into a stage (GP_FETCH) has no chains: on the RTX 3060, Super's staged group
 * of 1024 rows with its Mamba layers chained (40 chains of 1-2 layers) ran 172 -> 164 tok/s, its pull kernels
 * (k_nm_tile_pull, on the stream of the copies beside the layers) 5.42 -> 5.64 s, though none of a toy's
 * launches, events, priorities or overlaps differed; and those layers got no body to share (the tensor cores'
 * products take their pointers). RECLANK_GPU_TAIL_FETCH=1 chains them anyway (a measure). */
static int gg_tail_fetch(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_TAIL_FETCH");
        on = v && v[0] == '1';
    }
    return on;
}

static int gg_tail_plain(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_TAIL_PLAIN");
        on = v && v[0] == '1';
    }
    return on;
}

static int gg_tail_log(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_TAIL_LOG");
        on = v && v[0] == '1';
    }
    return on;
}

/* The first layer of a chain into dbody; the chain's first and last links into the state. */
__global__ void k_tail_start(const gg_link *lk, int first, int last, int *st, const gp_rec *dcode, gp_rec *dbody)
{
    const int4 *s = (const int4 *)(dcode + lk[first].pc);
    int4 *d = (int4 *)dbody;
    const int n = lk[first].n * (int)(sizeof(gp_rec) / sizeof(int4));
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        d[i] = s[i];
    }
    if (threadIdx.x == 0) {
        st[0] = first;
        st[1] = last;
    }
}

/* The last node of each body: the next layer of the chain into dbody (the layer before has finished: this kernel
 * comes after all of its kernels), then its body by a tail launch (it starts when this graph ends). */
__global__ void k_tail_next(const gg_link *lk, int *st, const gp_rec *dcode, gp_rec *dbody, const cudaGraphExec_t *ex,
                            int *err)
{
    const int c = st[0] + 1, last = st[1];
    if (c <= last) {
        const int4 *s = (const int4 *)(dcode + lk[c].pc);
        int4 *d = (int4 *)dbody;
        const int n = lk[c].n * (int)(sizeof(gp_rec) / sizeof(int4));
        for (int i = threadIdx.x; i < n; i += blockDim.x) {
            d[i] = s[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        st[0] = c;
        if (c <= last) {
            const cudaError_t e = cudaGraphLaunch(ex[lk[c].body], cudaStreamGraphTailLaunch);
            if (e != cudaSuccess && *(volatile int *)err == 0) {
                *(volatile int *)err = (c + 1) << 16 | (int)e;
            }
        }
    }
}

/* The layers of a chain (at load): g->hlink, g->n_link, g->body_recs, and for each run of chained layers its first
 * record's run[pc] = 1 + the run's index, its end, first and last links in ends/l0s/l1s (n_code entries each). */
static int gg_find_links(gg_prog *g, int *run, int *ends, int *l0s, int *l1s)
{
    g->hlink = (gg_link *)calloc((size_t)g->n_code + 1, sizeof(gg_link));
    g->n_link = 0;
    g->body_recs = 0;
    for (int pc = 0; pc < g->n_code && !gg_tail_fetch(); ++pc) {
        if (g->hcode[pc].op == GP_FETCH) {
            return 0;
        }
    }
    int nrun = 0, cur = -1, prev_end = -1;     /* cur: the first record of the open run */
    for (int pc = 0; pc < g->n_code; ++pc) {
        if (!(g->hcode[pc].flags & GP_F_LAYER)) {
            continue;
        }
        int end = pc + 1;
        while (end < g->n_code && !(g->hcode[end].flags & (GP_F_LAYER | GP_F_LAYER_END))) {
            ++end;
        }
        int ok = end < g->n_code;   /* (a mark after the last layer) */
        for (int q = pc; q < end && ok; ++q) {
            const int op = g->hcode[q].op;
            ok = !is_boundary(op) && op != GP_DS_STAGE_PULL && op != GP_DS_STAGE_JOIN;
        }
        if (ok) {
            if (cur < 0 || prev_end != pc) {
                cur = pc;
                run[pc] = 1 + nrun++;
                l0s[pc] = g->n_link;
            }
            gg_link *lk = g->hlink + g->n_link++;
            lk->pc = pc;
            lk->n = end - pc;
            lk->body = -1;
            g->body_recs = lk->n > g->body_recs ? lk->n : g->body_recs;
            ends[cur] = end;
            l1s[cur] = g->n_link - 1;
            prev_end = end;
        } else {
            cur = -1;
            prev_end = -1;
        }
        pc = end - 1;
    }
    /* runs of fewer than RECLANK_GPU_TAIL_MIN layers: not chains */
    for (int pc = 0; pc < g->n_code; ++pc) {
        if (run[pc] && l1s[pc] - l0s[pc] + 1 < gg_tail_min()) {
            run[pc] = 0;
        }
    }
    int kept = 0;
    for (int pc = 0; pc < g->n_code; ++pc) {
        kept += run[pc] != 0;
    }
    return kept;
}

/* The device side of the chains (at load, when gg_find_links found some). */
static int gg_chain_alloc(gg_prog *g)
{
    g->bodyg = (cudaGraph_t *)calloc((size_t)g->n_link, sizeof(cudaGraph_t));
    g->body = (cudaGraphExec_t *)calloc((size_t)g->n_link, sizeof(cudaGraphExec_t));
    void *h = NULL;
    if (cudaMalloc(&g->dlink, (size_t)g->n_link * sizeof(gg_link)) != cudaSuccess ||
        cudaMalloc(&g->dbody, (size_t)g->body_recs * sizeof(gp_rec)) != cudaSuccess ||
        cudaMalloc(&g->dst, 2 * sizeof(int)) != cudaSuccess ||
        cudaMalloc(&g->dbodyx, (size_t)g->n_link * sizeof(cudaGraphExec_t)) != cudaSuccess ||
        cudaHostAlloc(&h, sizeof(int), cudaHostAllocMapped) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    g->herr = (int *)h;
    *g->herr = 0;
    if (cudaHostGetDevicePointer((void **)&g->herr_d, h, 0) != cudaSuccess) {
        cudaGetLastError();
        return -1;
    }
    return 0;
}

static void gg_chain_free(gg_prog *g)
{
    for (int k = 0; k < g->n_body; ++k) {
        if (g->body != NULL && g->body[k] != NULL) {
            cudaGraphExecDestroy(g->body[k]);
        }
        if (g->bodyg != NULL && g->bodyg[k] != NULL) {
            cudaGraphDestroy(g->bodyg[k]);
        }
    }
    free(g->body);
    free(g->bodyg);
    free(g->hlink);
    cudaFree(g->dlink);
    cudaFree(g->dbody);
    cudaFree(g->dst);
    cudaFree(g->dbodyx);
    if (g->herr != NULL) {
        cudaFreeHost(g->herr);
    }
}

#if CUDART_VERSION >= 12040
/* A node's dependencies (13.0 gained the edges' data, NULL for none). */
#if CUDART_VERSION >= 13000
#define RC_NODE_DEPS(n, p, c) cudaGraphNodeGetDependencies((n), (p), NULL, (c))
#else
#define RC_NODE_DEPS(n, p, c) cudaGraphNodeGetDependencies((n), (p), (c))
#endif

/* -1 when graphs a and b are the same, else the index of their first node that differs (why: what differs). */
static int gg_graph_cmp(cudaGraph_t a, cudaGraph_t b, char *why, size_t nwhy)
{
    size_t na = 0, nb = 0;
    if (cudaGraphGetNodes(a, NULL, &na) != cudaSuccess || cudaGraphGetNodes(b, NULL, &nb) != cudaSuccess) {
        cudaGetLastError();
        snprintf(why, nwhy, "no nodes");
        return 0;
    }
    if (na != nb) {
        snprintf(why, nwhy, "%zu nodes vs %zu", na, nb);
        return 0;
    }
    cudaGraphNode_t *xa = (cudaGraphNode_t *)malloc(2 * na * sizeof(cudaGraphNode_t)), *xb = xa + na;
    int at = -1;
    if (cudaGraphGetNodes(a, xa, &na) != cudaSuccess || cudaGraphGetNodes(b, xb, &nb) != cudaSuccess) {
        at = 0;
        snprintf(why, nwhy, "no nodes");
    }
    for (size_t i = 0; i < na && at < 0; ++i) {
        cudaGraphNodeType ta, tb;
        size_t da = 0, db = 0;
        if (cudaGraphNodeGetType(xa[i], &ta) != cudaSuccess || cudaGraphNodeGetType(xb[i], &tb) != cudaSuccess ||
            ta != tb) {
            at = (int)i;
            snprintf(why, nwhy, "node types");
            break;
        }
        /* the same edges: each node's dependencies, by index */
        if (RC_NODE_DEPS(xa[i], NULL, &da) != cudaSuccess ||
            RC_NODE_DEPS(xb[i], NULL, &db) != cudaSuccess || da != db || da > 8) {
            at = (int)i;
            snprintf(why, nwhy, "dependencies (%zu vs %zu)", da, db);
            break;
        }
        if (da > 0) {
            cudaGraphNode_t pa[8], pb[8];
            RC_NODE_DEPS(xa[i], pa, &da);
            RC_NODE_DEPS(xb[i], pb, &db);
            for (size_t k = 0; k < da && at < 0; ++k) {
                size_t ia = 0, ib = 0;
                while (ia < na && xa[ia] != pa[k]) {
                    ++ia;
                }
                while (ib < nb && xb[ib] != pb[k]) {
                    ++ib;
                }
                if (ia != ib) {
                    at = (int)i;
                    snprintf(why, nwhy, "dependencies");
                }
            }
        }
        if (at >= 0) {
            break;
        }
        if (ta == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams ka, kb;
            if (cudaGraphKernelNodeGetParams(xa[i], &ka) != cudaSuccess ||
                cudaGraphKernelNodeGetParams(xb[i], &kb) != cudaSuccess) {
                at = (int)i;
                snprintf(why, nwhy, "kernel params");
                break;
            }
            const char *name = NULL;
            if (cudaFuncGetName(&name, ka.func) != cudaSuccess || name == NULL) {
                name = "?";
            }
            if (ka.func != kb.func) {
                at = (int)i;
                const char *nb2 = NULL;
                if (cudaFuncGetName(&nb2, kb.func) != cudaSuccess || nb2 == NULL) {
                    nb2 = "?";
                }
                snprintf(why, nwhy, "kernel %s vs %s", name, nb2);
            } else if (ka.gridDim.x != kb.gridDim.x || ka.gridDim.y != kb.gridDim.y || ka.gridDim.z != kb.gridDim.z) {
                at = (int)i;
                snprintf(why, nwhy, "%s: grid %ux%ux%u vs %ux%ux%u", name, ka.gridDim.x, ka.gridDim.y, ka.gridDim.z,
                         kb.gridDim.x, kb.gridDim.y, kb.gridDim.z);
            } else if (ka.blockDim.x != kb.blockDim.x || ka.blockDim.y != kb.blockDim.y ||
                       ka.blockDim.z != kb.blockDim.z || ka.sharedMemBytes != kb.sharedMemBytes) {
                at = (int)i;
                snprintf(why, nwhy, "%s: block or shared memory", name);
            } else if (ka.kernelParams == NULL || kb.kernelParams == NULL) {
                at = (int)i;
                snprintf(why, nwhy, "%s: no kernelParams", name);
            } else {
                for (size_t j = 0;; ++j) {
                    size_t off = 0, sz = 0;
                    if (cudaFuncGetParamInfo(ka.func, j, &off, &sz) != cudaSuccess) {
                        cudaGetLastError();
                        break;
                    }
                    if (memcmp(ka.kernelParams[j], kb.kernelParams[j], sz) != 0) {
                        at = (int)i;
                        snprintf(why, nwhy, "%s: argument %zu", name, j);
                        break;
                    }
                }
            }
        } else if (ta == cudaGraphNodeTypeMemset) {
            cudaMemsetParams ma, mb;
            if (cudaGraphMemsetNodeGetParams(xa[i], &ma) != cudaSuccess ||
                cudaGraphMemsetNodeGetParams(xb[i], &mb) != cudaSuccess || ma.dst != mb.dst || ma.pitch != mb.pitch ||
                ma.value != mb.value || ma.elementSize != mb.elementSize || ma.width != mb.width ||
                ma.height != mb.height) {
                at = (int)i;
                snprintf(why, nwhy, "a memset");
            }
        } else if (ta == cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms ca, cb;
            memset(&ca, 0, sizeof(ca));
            memset(&cb, 0, sizeof(cb));
            if (cudaGraphMemcpyNodeGetParams(xa[i], &ca) != cudaSuccess ||
                cudaGraphMemcpyNodeGetParams(xb[i], &cb) != cudaSuccess || memcmp(&ca, &cb, sizeof(ca)) != 0) {
                at = (int)i;
                snprintf(why, nwhy, "a memcpy");
            }
        } else if (ta != cudaGraphNodeTypeEmpty) {
            at = (int)i;
            snprintf(why, nwhy, "a node of type %d", (int)ta);
        }
    }
    free(xa);
    cudaGetLastError();
    return at;
}

/* A chained layer's records as a graph (not made into an exec): its kernels read their records in dbody, then
 * k_tail_next. 1: the layer cannot be a body (a record fused with one past the layer's end). */
static int gg_capture_link(gg_prog *g, const gg_link *lk, cudaGraph_t *out)
{
    CK(cudaStreamBeginCapture(gg_stream, cudaStreamCaptureModeThreadLocal));
    int rc = 0;
    for (int pc = lk->pc; pc < lk->pc + lk->n && rc == 0; ++pc) {
        const gp_rec *r = g->hcode + pc;
        gg_at(g, pc, r->op, lk->pc, lk->pc + lk->n);
        rc = gg_launch(g, r, g->dbody + (pc - lk->pc), g->denv);
    }
    if (rc == 0 && gg_skip != NULL) {
        rc = 1;
    }
    if (rc == 0) {
        k_tail_next<<<1, 256, 0, gg_stream>>>(g->dlink, g->dst, g->dcode, g->dbody, g->dbodyx, g->herr_d);
    }
    cudaGraph_t graph = NULL;
    cudaError_t err = cudaStreamEndCapture(gg_stream, &graph);
    if (rc != 0 || err != cudaSuccess) {
        if (graph != NULL) {
            cudaGraphDestroy(graph);
        }
        if (rc == 0) {
            snprintf(gg_error, sizeof(gg_error), "the capture of a chained layer: %s", cudaGetErrorString(err));
            rc = -1;
        }
        return rc;
    }
    *out = graph;
    return 0;
}

static int gg_capture(gg_prog *g, gg_seg *sg);

/* The bodies of a chain's layers (its first run or prepare): each layer's graph, the same as a body's before or
 * a new body; the new ones made for device launch and uploaded. A layer that cannot be a body, or a body that
 * cannot be made, and the chain's records are one ordinary graph instead. */
static int gg_capture_chain(gg_prog *g, gg_seg *sg)
{
    const gp_rec *skip0 = gg_skip;
    const float *qx0 = gg_qx_src;
    const size_t qxn0 = gg_qx_n;
    const int nb0 = g->n_body;
    int rc = gg_tail_plain() ? 1 : 0;
    char why[256];
    for (int l = sg->link0; l <= sg->link1 && rc == 0; ++l) {
        gg_link *lk = g->hlink + l;
        cudaGraph_t gr = NULL;
        rc = gg_capture_link(g, lk, &gr);
        if (rc != 0) {
            break;
        }
        int k = 0, best = -1, best_at = -1;
        char best_why[256] = "";
        for (; k < g->n_body; ++k) {
            const int at = gg_graph_cmp(g->bodyg[k], gr, why, sizeof(why));
            if (at < 0) {
                break;
            }
            if (at > best_at) {
                best_at = at;
                best = k;
                snprintf(best_why, sizeof(best_why), "%s", why);
            }
        }
        if (k < g->n_body) {
            cudaGraphDestroy(gr);
            lk->body = k;
            continue;
        }
        g->bodyg[g->n_body] = gr;
        lk->body = g->n_body++;
        if (gg_tail_log()) {
            const int layer = (g->hcode[lk->pc].flags >> GP_F_LAYER_SHIFT) - 1;
            size_t nn = 0;
            cudaGraphGetNodes(gr, NULL, &nn);
            if (best >= 0) {
                fprintf(stderr, "tail: layer %d (%d records, %zu nodes): body %d; body %d differs at node %d: %s\n",
                        layer, lk->n, nn, lk->body, best, best_at, best_why);
            } else {
                fprintf(stderr, "tail: layer %d (%d records, %zu nodes): body %d\n", layer, lk->n, nn, lk->body);
            }
        }
    }
    for (int k = nb0; k < g->n_body && rc == 0; ++k) {
        cudaError_t e = cudaGraphInstantiateWithFlags(&g->body[k], g->bodyg[k], cudaGraphInstantiateFlagDeviceLaunch);
        if (e == cudaSuccess) {
            e = cudaGraphUpload(g->body[k], gg_stream);
        }
        if (e != cudaSuccess) {
            if (e == cudaErrorMemoryAllocation) {
                snprintf(gg_error, sizeof(gg_error), "cudaGraphInstantiate of a body: out of memory");
                rc = -1;
            } else {
                if (gg_tail_log()) {
                    fprintf(stderr, "tail: a body for device launch failed (%s): an ordinary graph\n",
                            cudaGetErrorString(e));
                }
                rc = 1;
            }
            cudaGetLastError();
        }
    }
    if (rc == 0) {
        if (cudaMemcpy(g->dlink, g->hlink, (size_t)g->n_link * sizeof(gg_link), cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(g->dbodyx, g->body, (size_t)g->n_body * sizeof(cudaGraphExec_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            snprintf(gg_error, sizeof(gg_error), "the tables of a chain: %s", cudaGetErrorString(cudaGetLastError()));
            return -1;
        }
        sg->exec = g->body[g->hlink[sg->link0].body];
        if (gg_tail_log()) {
            fprintf(stderr, "tail: a chain of %d layers (records %d-%d): %d bodies in the program\n",
                    sg->link1 - sg->link0 + 1, sg->start, sg->end - 1, g->n_body);
        }
        return 0;
    }
    /* the bodies made here go */
    for (int k = nb0; k < g->n_body; ++k) {
        if (g->body[k] != NULL) {
            cudaGraphExecDestroy(g->body[k]);
            g->body[k] = NULL;
        }
        cudaGraphDestroy(g->bodyg[k]);
        g->bodyg[k] = NULL;
    }
    g->n_body = nb0;
    for (int l = sg->link0; l <= sg->link1; ++l) {
        g->hlink[l].body = -1;
    }
    if (rc < 0) {
        return rc;
    }
    gg_skip = skip0;
    gg_qx_src = qx0;
    gg_qx_n = qxn0;
    sg->chain = 0;
    if (gg_tail_log()) {
        fprintf(stderr, "tail: records %d-%d as an ordinary graph\n", sg->start, sg->end - 1);
    }
    return gg_capture(g, sg);
}
#else
static int gg_capture(gg_prog *g, gg_seg *sg);
static int gg_capture_chain(gg_prog *g, gg_seg *sg)
{
    sg->chain = 0;
    return gg_capture(g, sg);
}
#endif

/* Record the graph of a segment (the first run of a program). */
static int gg_capture(gg_prog *g, gg_seg *sg)
{
    if (sg->chain) {
        return gg_capture_chain(g, sg);
    }
    CK(cudaStreamBeginCapture(gg_stream, cudaStreamCaptureModeThreadLocal));
    int rc = gg_launch_seg(g, sg, 0);
    cudaGraph_t graph;
    cudaError_t err = cudaStreamEndCapture(gg_stream, &graph);
    if (rc != 0) {
        return rc;
    }
    CK(err);
    if (gg_pdl_on()) {
        CK(gg_pdl_edges(graph));
    }
    CK(cudaGraphInstantiate(&sg->exec, graph, 0));
    CK(cudaGraphDestroy(graph));
    return 0;
}

/* The graphs of all the segments up to a boundary record go on the stream
 * first; then the runner runs their GP_CPU_TASK records, while the GPU runs
 * them. Thus the host launches nothing between the tasks.
 *
 * The first run records every graph before it launches one. A graph of the
 * GPU can wait for a task (GP_AWAIT), and the host runs the tasks only after
 * the launches; the first use of a kernel can make the driver wait for the
 * GPU (the lazy load of a module), so a capture after a launch could wait
 * for a task that waits for the capture. */
static int gg_exec(gg_prog *g)
{
    int pc = 0, k = 0, task = 0;
    int graph = g->use_graph && !gg_sync_check();
    if (g->herr != NULL && *(volatile int *)g->herr != 0) {
        const int v = *(volatile int *)g->herr;
        snprintf(gg_error, sizeof(gg_error), "a tail launch of a chain failed at its link %d: %s", (v >> 16) - 1,
                 cudaGetErrorString((cudaError_t)(v & 0xffff)));
        *(volatile int *)g->herr = 0;
        return -1;
    }
    if (graph) {
        for (int s = 0; s < g->n_seg; ++s) {
            if (g->seg[s].exec == NULL && gg_capture(g, g->seg + s) != 0) {
                return -1;
            }
        }
    }
    while (pc < g->n_code) {
        const gp_rec *r = g->hcode + pc;
        if (is_boundary(r->op)) {
            if (graph && gg_run_tasks(g, &task, pc) != 0) {
                return -1;
            }
            gg_at(g, pc, r->op, pc, pc + 1);
            if (gg_boundary(g, r) != 0) {
                return -1;
            }
            GG_SYNC_AT(pc, r->op);
            ++pc;
            task = pc;
            continue;
        }
        gg_seg *sg = g->seg + k++;
        if (!graph) {
            if (gg_launch_seg(g, sg, 1) != 0) {
                return -1;
            }
        } else if (sg->chain) {
            /* (the chain's first layer into dbody, then its body: the others by tail launches) */
            gg_at(g, sg->end - 1, g->hcode[sg->end - 1].op, sg->start, sg->end);
            k_tail_start<<<1, 256, 0, gg_stream>>>(g->dlink, sg->link0, sg->link1, g->dst, g->dcode, g->dbody);
            CK(cudaGraphLaunch(sg->exec, gg_stream));
        } else {
            gg_at(g, sg->end - 1, g->hcode[sg->end - 1].op, sg->start, sg->end);
            CK(cudaGraphLaunch(sg->exec, gg_stream));
        }
        pc = sg->end;
    }
    if (graph && gg_run_tasks(g, &task, g->n_code) != 0) {
        return -1;
    }
    CK(cudaGetLastError());
    return 0;
}

/* Run the scalar operations on the host environment. */
static void gg_scalars(gg_prog *g)
{
    int64_t *e = g->henv;
    for (int pc = 0; pc < g->n_code; ++pc) {
        const gp_rec *r = g->hcode + pc;
        switch (r->op) {
        case GP_S_MOV: e[r->v[0]] = hi(r, e, 1); break;
        case GP_S_ADD: e[r->v[0]] = hi(r, e, 1) + hi(r, e, 2); break;
        case GP_S_SUB: e[r->v[0]] = hi(r, e, 1) - hi(r, e, 2); break;
        case GP_S_MUL: e[r->v[0]] = hi(r, e, 1) * hi(r, e, 2); break;
        case GP_S_MAX: {
            int64_t a = hi(r, e, 1), b = hi(r, e, 2);
            e[r->v[0]] = a > b ? a : b;
            break;
        }
        case GP_S_MIN: {
            int64_t a = hi(r, e, 1), b = hi(r, e, 2);
            e[r->v[0]] = a < b ? a : b;
            break;
        }
        default:
            break;
        }
    }
}
