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
        /* operands 4-6 (or none): the mode (1: KQ_NV4 rows into NV4T), the columns of the even and odd ranges */
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

/* Record the graph of a segment (the first run of a program). */
static int gg_capture(gg_prog *g, gg_seg *sg)
{
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
