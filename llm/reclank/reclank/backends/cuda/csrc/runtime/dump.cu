/* RECLANK_DS_DUMP=dir: the real operands of chosen records of a run without
 * the graph (DSV41GPU(graph=False)) to files, for the bench of the kernel
 * module kernels/dsv41/prefill.cu (bench/ds_prefill.cu). After the record ran:
 * dir/<kind>_<n>/rec.bin (the record, its slots resolved) and op_KK.bin, the
 * bytes of each operand that points at device memory (inputs and outputs).
 * Kinds: attn (GP_DS_ATTN of a group), hot (GP_DS_HOT_MOE: a group, or a step), of
 * RECLANK_DS_DUMP_T rows (1024); n: the occurrences in RECLANK_DS_DUMP_ATTN
 * and RECLANK_DS_DUMP_HOT (lists, from 0). */
#include <sys/stat.h>

static int ds_dump_want(const char *kind, const char *list_env, int t)
{
    static int seen[2];
    const char *dir = getenv("RECLANK_DS_DUMP"), *list = getenv(list_env), *te = getenv("RECLANK_DS_DUMP_T");
    if (dir == NULL || list == NULL || t != (te ? atoi(te) : 1024)) {
        return -1;
    }
    int n = seen[kind[0] == 'h']++;
    for (const char *p = list; *p;) {
        if (atoi(p) == n) {
            return n;
        }
        p = strchr(p, ',');
        if (p == NULL) {
            break;
        }
        ++p;
    }
    return -1;
}

static size_t ds_rb_h(int hd, int form)
{
    switch (form) {
    case 0: return (size_t)hd * 4;
    case 1: return (size_t)hd * 2;
    case 2: return (size_t)hd + hd / 32;
    case 4: return (size_t)hd / 2 + hd / 16;
    default: return (size_t)hd / 2 + hd / 32;
    }
}

static int ds_cf_h(int form, int kind)
{
    return form <= 1 ? form : kind == 0 ? 2 : kind == 1 ? 4 : 5;
}

/* the record (slots resolved) and nops operands of nb[i] bytes each */
static int ds_dump_write(const gg_prog *g, const gp_rec *r, const char *kind, int n, const int *ops,
                         const size_t *nb, int nops)
{
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s_%d", getenv("RECLANK_DS_DUMP"), kind, n);
    mkdir(getenv("RECLANK_DS_DUMP"), 0775);
    mkdir(path, 0775);
    gp_rec rr = *r;
    for (int k = 0; k < GP_NARG; ++k) {
        if (rr.tag[k] == GP_T_SLOT) {
            rr.v[k] = g->henv[r->v[k]];
            rr.tag[k] = GP_T_INT;
        }
    }
    char fn[1100];
    snprintf(fn, sizeof(fn), "%s/rec.bin", path);
    FILE *f = fopen(fn, "wb");
    if (f == NULL) {
        snprintf(gg_error, sizeof(gg_error), "RECLANK_DS_DUMP: cannot write %s", fn);
        return -1;
    }
    fwrite(&rr, sizeof(rr), 1, f);
    fclose(f);
    for (int i = 0; i < nops; ++i) {
        if (rr.tag[ops[i]] == GP_T_NONE || rr.v[ops[i]] == 0 || nb[i] == 0) {
            continue;
        }
        void *h = malloc(nb[i]);
        CK(cudaMemcpy(h, (const void *)(intptr_t)rr.v[ops[i]], nb[i], cudaMemcpyDeviceToHost));
        snprintf(fn, sizeof(fn), "%s/op_%02d.bin", path, ops[i]);
        f = fopen(fn, "wb");
        fwrite(h, 1, nb[i], f);
        fclose(f);
        free(h);
    }
    fprintf(stderr, "[ds_dump] %s\n", path);
    return 0;
}

/* GP_DS_ATTN of a group: q, ring, ckv (the rows its selection reaches), sel, cnt, sink, out, kvq */
static int ds_dump_attn(const gg_prog *g, const gp_rec *r)
{
    const int64_t *e = g->henv;
    int t = (int)hi(r, e, 14), n = ds_dump_want("attn", "RECLANK_DS_DUMP_ATTN", t);
    if (n < 0 || g->use_graph) {
        return 0;
    }
    CK(cudaStreamSynchronize(gg_stream));
    int H = (int)hi(r, e, 9), hd = (int)hi(r, e, 10), maxk = (int)hi(r, e, 12), win = (int)hi(r, e, 3);
    int form = r->tag[17] != GP_T_NONE ? (int)hi(r, e, 17) : 0;
    size_t rows = 0;
    int has_c = r->tag[4] != GP_T_NONE && hi(r, e, 4) != 0;
    if (has_c) {
        int *sel = (int *)malloc((size_t)t * maxk * 4), *cnt = (int *)malloc((size_t)t * 4);
        CK(cudaMemcpy(sel, (const void *)(intptr_t)hi(r, e, 5), (size_t)t * maxk * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cnt, (const void *)(intptr_t)hi(r, e, 6), (size_t)t * 4, cudaMemcpyDeviceToHost));
        for (int j = 0; j < t; ++j) {
            for (int m = 0; m < cnt[j]; ++m) {
                rows = (size_t)sel[(size_t)j * maxk + m] + 1 > rows ? (size_t)sel[(size_t)j * maxk + m] + 1 : rows;
            }
        }
        free(sel);
        free(cnt);
    }
    int ops[] = {0, 1, 4, 5, 6, 7, 8, 13};
    size_t nb[] = {(size_t)t * H * hd * 4, (size_t)win * ds_rb_h(hd, ds_cf_h(form, 0)),
                   rows * ds_rb_h(hd, ds_cf_h(form, 1)), has_c ? (size_t)t * maxk * 4 : 0, has_c ? (size_t)t * 4 : 0,
                   (size_t)H * 4, (size_t)t * H * hd * 4, (size_t)t * hd * 4};
    return ds_dump_write(g, r, "attn", n, ops, nb, 8);
}

/* GP_DS_HOT_MOE (a group, a step, a verify group): idx, val, map, the slots of store and stage that map
 * takes, the planes of x and of the activations, act and de (the outputs) */
static int ds_dump_hot(const gg_prog *g, const gp_rec *r)
{
    const int64_t *e = g->henv;
    int t = (int)hi(r, e, 8), n = ds_dump_want("hot", "RECLANK_DS_DUMP_HOT", t);
    if (n < 0 || g->use_graph) {
        return 0;
    }
    CK(cudaStreamSynchronize(gg_stream));
    int k = (int)hi(r, e, 9), H = (int)hi(r, e, 10), I = (int)hi(r, e, 11), E = (int)hi(r, e, 21);
    size_t per = (size_t)hi(r, e, 5);
    int *map = (int *)malloc((size_t)E * 4);
    CK(cudaMemcpy(map, (const void *)(intptr_t)hi(r, e, 3), (size_t)E * 4, cudaMemcpyDeviceToHost));
    size_t ns = 0, nst = 0;
    for (int x = 0; x < E; ++x) {
        if (map[x] >= (1 << 20)) {
            nst = (size_t)(map[x] - (1 << 20)) + 1 > nst ? (size_t)(map[x] - (1 << 20)) + 1 : nst;
        } else if (map[x] >= 0) {
            ns = (size_t)map[x] + 1 > ns ? (size_t)map[x] + 1 : ns;
        }
    }
    free(map);
    size_t tk = (size_t)t * k;
    int ops[] = {1, 2, 3, 4, 23, 12, 13, 14, 15, 17, 18, 19, 20};
    size_t nb[] = {tk * 4, tk * 4, (size_t)E * 4, ns * per, nst * per, (size_t)t * H, (size_t)t * H,
                   (size_t)t * (H / 32) * 4, tk * 2 * I * 4, tk * I, tk * I, tk * (I / 32) * 4, tk * H * 4};
    return ds_dump_write(g, r, "hot", n, ops, nb, 13);
}
