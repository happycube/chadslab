/* The bench of the kernel modules kernels/dsv41/prefill.cu and hot.cu on real
 * operands (runtime/dump.cu: RECLANK_DS_DUMP of a run without the graph). It
 * builds the modules alone (seconds, not the library's minutes):
 *
 *   bench/ds_prefill.sh attn DIR [reps]   each attention kernel on the dumped
 *       record: its time, its outputs against the dumped ones (the kernel of the
 *       run) and against float64 on sampled queries
 *   bench/ds_prefill.sh hot DIR [reps]    the hot experts' kernels (gate and up,
 *       down) on the dumped record (a group of a prompt, or a step or verify group:
 *       RECLANK_DS_DUMP_T): time, outputs against the dumped ones (the pairs of map)
 *
 * The record's operands that point at device memory get copies of the dumped
 * bytes; the others are its values (the slots resolved). */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include "opcodes.h"
#include "port.cuh"
#define PDL_START() do { } while (0)
#include "kernels/operands.cu"
#include "kernels/frag.cuh"
#include "kernels/dsv41/types.cu"
#include "kernels/dsv41/prefill.cu"
#include "kernels/dsv41/hot.cu"

#define CKB(x)                                                                         \
    do {                                                                               \
        cudaError_t err_ = (x);                                                        \
        if (err_ != cudaSuccess) {                                                     \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err_)); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static std::vector<char> readf(const char *dir, const char *name, bool must = true)
{
    char fn[1024];
    snprintf(fn, sizeof(fn), "%s/%s", dir, name);
    std::vector<char> b;
    FILE *f = fopen(fn, "rb");
    if (f == NULL) {
        if (must) {
            fprintf(stderr, "no %s\n", fn);
            exit(1);
        }
        return b;
    }
    fseek(f, 0, SEEK_END);
    b.resize((size_t)ftell(f));
    fseek(f, 0, SEEK_SET);
    if (fread(b.data(), 1, b.size(), f) != b.size()) {
        exit(1);
    }
    fclose(f);
    return b;
}

struct Dump {
    gp_rec rec;
    std::vector<char> host[GP_NARG];
    void *dev[GP_NARG] = {};
    gp_rec *drec;
    int64_t *denv;
};

static void load(Dump &d, const char *dir)
{
    std::vector<char> r = readf(dir, "rec.bin");
    memcpy(&d.rec, r.data(), sizeof(gp_rec));
    for (int k = 0; k < GP_NARG; ++k) {
        char name[32];
        snprintf(name, sizeof(name), "op_%02d.bin", k);
        d.host[k] = readf(dir, name, false);
        if (!d.host[k].empty()) {
            CKB(cudaMalloc(&d.dev[k], d.host[k].size()));
            CKB(cudaMemcpy(d.dev[k], d.host[k].data(), d.host[k].size(), cudaMemcpyHostToDevice));
            d.rec.v[k] = (int64_t)(intptr_t)d.dev[k];
        }
    }
    CKB(cudaMalloc(&d.drec, sizeof(gp_rec)));
    CKB(cudaMemcpy(d.drec, &d.rec, sizeof(gp_rec), cudaMemcpyHostToDevice));
    CKB(cudaMalloc(&d.denv, 64));
}

template <typename F>
static float timeit(F launch, int reps)
{
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    launch();
    CKB(cudaDeviceSynchronize());
    cudaEventRecord(a);
    for (int i = 0; i < reps; ++i) {
        launch();
    }
    cudaEventRecord(b);
    CKB(cudaEventSynchronize(b));
    CKB(cudaGetLastError());
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    return ms / reps;
}

/* the keys of query j as the kernels decode them (float32), for the float64 reference */
__global__ void k_bench_keys(const gp_rec *r, const int64_t *e, int j, float *out, int *nko)
{
    int win = DI(3), maxk = DI(12);
    const int hd = 512;
    int64_t pos = di(r, e, 2), p = pos + j;
    int64_t lo = p - win + 1 > 0 ? p - win + 1 : 0;
    int nr = lo < pos ? (int)(pos - lo) : 0;
    int g0 = lo > pos ? (int)(lo - pos) : 0, ng = j - g0 + 1;
    int mode = r->tag[16] != GP_T_NONE ? DI(16) : 0;
    if (mode & 2) {
        int64_t w0 = di(r, e, 18);
        lo = lo > w0 ? lo : w0;
        nr = lo < pos ? (int)(pos - lo) : 0;
        g0 = lo > pos ? (int)(lo - pos) : 0;
        ng = j - g0 + 1;
    }
    const void *ckv = DP(const void, 4);
    int form = ds_form(r, e, 17);
    const int *sel = DP(const int, 5) + (size_t)j * maxk;
    int nk = nr + ng + (ckv ? DP(const int, 6)[j] : 0);
    if (threadIdx.x == 0) {
        *nko = nk;
    }
    for (int i = threadIdx.x; i < nk * 128; i += blockDim.x) {
        int m = i / 128, c4 = 4 * (i % 128);
        ds_krow kr = ds_akey(m, nr, ng, lo, g0, win, DP(const void, 1), DP(const float, 13), ckv, sel, form);
        *(float4 *)(out + (size_t)m * hd + c4) = ds_rld4(kr.p, kr.row, c4, hd, kr.form);
    }
}

struct Err {
    double max_abs = 0, se = 0, ss = 0;
    size_t n = 0, same = 0;
    void add(double a, double ref)
    {
        double d = fabs(a - ref);
        max_abs = d > max_abs ? d : max_abs;
        se += d * d;
        ss += ref * ref;
        ++n;
    }
    double db() const { return se > 0 ? 10 * log10(se / ss) : -999; }
};

static void attn(const char *dir, int reps)
{
    Dump d;
    load(d, dir);
    const gp_rec &R = d.rec;
    int t = (int)R.v[14], H = (int)R.v[9], hd = (int)R.v[10];
    printf("attn %s: t %d, H %d, hd %d, pos %lld, win %d, maxk %d, form %d, compressed %s\n", dir, t, H, hd,
           (long long)R.v[2], (int)R.v[3], (int)R.v[12], R.tag[17] != GP_T_NONE ? (int)R.v[17] : 0,
           d.dev[4] ? "yes" : "no");
    size_t n = (size_t)t * H * hd;
    const float *ref = (const float *)d.host[8].data();
    float *out = (float *)d.dev[8];
    std::vector<float> h(n);
    /* float64 on sampled queries (every 64th, all heads) */
    float *dk;
    int *dnk;
    CKB(cudaMalloc(&dk, (size_t)4096 * hd * 4));
    CKB(cudaMalloc(&dnk, 4));
    std::vector<int> qs;
    for (int j = 0; j < t; j += 64) {
        qs.push_back(j);
    }
    qs.push_back(t - 1);
    std::vector<std::vector<double>> ref64(qs.size());
    const float *q = (const float *)d.host[0].data(), *sink = (const float *)d.host[7].data();
    float scale;
    memcpy(&scale, &R.v[11], 4);
    for (size_t qi = 0; qi < qs.size(); ++qi) {
        int j = qs[qi], nk;
        k_bench_keys<<<1, 256>>>(d.drec, d.denv, j, dk, dnk);
        CKB(cudaMemcpy(&nk, dnk, 4, cudaMemcpyDeviceToHost));
        std::vector<float> K((size_t)nk * hd);
        CKB(cudaMemcpy(K.data(), dk, K.size() * 4, cudaMemcpyDeviceToHost));
        ref64[qi].assign((size_t)H * hd, 0.0);
        for (int hh = 0; hh < H; ++hh) {
            const float *qq = q + ((size_t)j * H + hh) * hd;
            std::vector<double> s(nk);
            double mx = -1e300;
            for (int m = 0; m < nk; ++m) {
                double a = 0;
                for (int c = 0; c < hd; ++c) {
                    a += (double)qq[c] * K[(size_t)m * hd + c];
                }
                s[m] = a * scale;
                mx = s[m] > mx ? s[m] : mx;
            }
            double den = exp((double)sink[hh] - mx);
            for (int m = 0; m < nk; ++m) {
                s[m] = exp(s[m] - mx);
                den += s[m];
            }
            for (int m = 0; m < nk; ++m) {
                for (int c = 0; c < hd; ++c) {
                    ref64[qi][(size_t)hh * hd + c] += s[m] / den * K[(size_t)m * hd + c];
                }
            }
        }
    }
    Err base;
    for (size_t qi = 0; qi < qs.size(); ++qi) {
        for (size_t i = 0; i < (size_t)H * hd; ++i) {
            base.add(ref[(size_t)qs[qi] * H * hd + i], ref64[qi][i]);
        }
    }
    printf("  the run's kernel against float64: %.1f dB, max |d| %.3g\n", base.db(), base.max_abs);
    auto check = [&](const char *name, float ms) {
        CKB(cudaMemcpy(h.data(), out, n * 4, cudaMemcpyDeviceToHost));
        Err er, e64;
        for (size_t i = 0; i < n; ++i) {
            er.same += memcmp(&h[i], &ref[i], 4) == 0;
            er.add(h[i], ref[i]);
        }
        for (size_t qi = 0; qi < qs.size(); ++qi) {
            for (size_t i = 0; i < (size_t)H * hd; ++i) {
                e64.add(h[(size_t)qs[qi] * H * hd + i], ref64[qi][i]);
            }
        }
        printf("  %-14s %8.3f ms   same bits %.4f%%   vs run %.1f dB (max |d| %.3g)   vs float64 %.1f dB\n", name, ms,
               100.0 * er.same / n, er.db(), er.max_abs, e64.db());
    };
    dim3 g16((unsigned)(H / DS_GHB), (unsigned)t), g64((unsigned)(H / 64), (unsigned)t);
    CKB(cudaMemset(out, 0, n * 4));
    check("k_ds_attn_g", timeit([&] { k_ds_attn_g<<<g16, 256>>>(d.drec, d.denv); }, reps));
    CKB(cudaMemset(out, 0, n * 4));
    check("k_ds_attn_g2", timeit([&] { k_ds_attn_g2<<<g16, 256>>>(d.drec, d.denv); }, reps));
    CKB(cudaFuncSetAttribute(k_ds_attn_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DS_TSMEM));
    CKB(cudaMemset(out, 0, n * 4));
    check("k_ds_attn_tc", timeit([&] { k_ds_attn_tc<<<g64, 256, DS_TSMEM>>>(d.drec, d.denv); }, reps));
}

static void hot(const char *dir, int reps)
{
    Dump d;
    load(d, dir);
    const gp_rec &R = d.rec;
    int t = (int)R.v[8], k = (int)R.v[9], H = (int)R.v[10], I = (int)R.v[11], E = (int)R.v[21];
    const int *idx = (const int *)d.host[1].data(), *map = (const int *)d.host[3].data();
    size_t tk = (size_t)t * k, held = 0;
    int experts = 0;
    for (int x = 0; x < E; ++x) {
        experts += map[x] >= 0;
    }
    std::vector<char> on(tk);
    for (size_t p = 0; p < tk; ++p) {
        on[p] = idx[p] >= 0 && map[idx[p]] >= 0;
        held += on[p];
    }
    printf("hot %s: t %d, k %d, H %d, I %d, experts held %d, their pairs %zu\n", dir, t, k, H, I, experts, held);
    /* (per pair: gate and up 2 I H, down H I multiply-adds) */
    double gmac = held * 3.0 * H * I / 1e9;
    auto cmp = [&](int op, size_t rowlen, const char *name, float ms, double macs) {
        std::vector<float> h(tk * rowlen);
        CKB(cudaMemcpy(h.data(), d.dev[op], h.size() * 4, cudaMemcpyDeviceToHost));
        const float *ref = (const float *)d.host[op].data();
        size_t same = 0, nn = 0;
        for (size_t p = 0; p < tk; ++p) {
            if (!on[p]) {
                continue;
            }
            for (size_t i = 0; i < rowlen; ++i) {
                same += memcmp(&h[p * rowlen + i], &ref[p * rowlen + i], 4) == 0;
                ++nn;
            }
        }
        printf("  %-22s %8.3f ms   %.1f TMAC/s   same bits %.4f%% of %zu\n", name, ms, macs / ms,
               100.0 * same / nn, nn);
    };
    /* the run's kernels: a group of a prompt k_ds_hot_tc, a step and a verify group (t <= 8) a block
     * for each pair */
    if (t > 8) {
        CKB(cudaMemset(d.dev[15], 0, tk * 2 * I * 4));
        cmp(15, 2 * (size_t)I, "k_ds_hot_tc gate, up",
            timeit([&] { k_ds_hot_tc<false><<<dim3((unsigned)(I / 16), 2, (unsigned)E), 128>>>(d.drec, d.denv); },
                   reps), gmac * 2 / 3);
        CKB(cudaMemset(d.dev[20], 0, tk * H * 4));
        cmp(20, (size_t)H, "k_ds_hot_tc down",
            timeit([&] { k_ds_hot_tc<true><<<dim3((unsigned)(H / 16), (unsigned)E), 128>>>(d.drec, d.denv); }, reps),
            gmac / 3);
    } else {
        CKB(cudaMemset(d.dev[15], 0, tk * 2 * I * 4));
        cmp(15, 2 * (size_t)I, "k_ds_hot_gu",
            timeit([&] { k_ds_hot_gu<<<dim3((unsigned)(I / 16), 2, (unsigned)tk), 256>>>(d.drec, d.denv); }, reps),
            gmac * 2 / 3);
        CKB(cudaMemset(d.dev[20], 0, tk * H * 4));
        cmp(20, (size_t)H, "k_ds_hot_dn",
            timeit([&] { k_ds_hot_dn<<<dim3((unsigned)(H / 16), (unsigned)tk), 256>>>(d.drec, d.denv); }, reps),
            gmac / 3);
    }
    /* the lists once (items of up to ch pairs of an expert), then the items alone, Z at a time in
     * the grid */
    int *lex, *loff, *lcn, *lpl, *lnx;
    size_t ni = tk + E;
    CKB(cudaMalloc(&lex, ni * 4));
    CKB(cudaMalloc(&loff, ni * 4));
    CKB(cudaMalloc(&lcn, ni * 4));
    CKB(cudaMalloc(&lpl, tk * 4));
    CKB(cudaMalloc(&lnx, 4));
    auto lists = [&](int ch) {
        float ms = timeit([&] {
            k_ds_hot_lists<<<1, 1024>>>((const int *)d.dev[1], (const int *)d.dev[3], (int)tk, E, ch, lex, loff,
                                        lcn, lpl, lnx); }, reps);
        int nx;
        CKB(cudaMemcpy(&nx, lnx, 4, cudaMemcpyDeviceToHost));
        printf("  k_ds_hot_lists ch %-4d   %8.3f ms   %d items\n", ch, ms, nx);
    };
#define HOT6(PT, RG, KB, S, Z)                                                                          \
    do {                                                                                                \
        using C_ = ds_tc6<PT, RG, KB, S>;                                                               \
        unsigned z_ = (Z);                                                                              \
        char nm_[64];                                                                                   \
        CKB(cudaFuncSetAttribute(k_ds_hot_tc6<false, PT, RG, KB, S>,                                    \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C_::smem));          \
        CKB(cudaFuncSetAttribute(k_ds_hot_tc6<true, PT, RG, KB, S>,                                     \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)C_::smem));          \
        CKB(cudaMemset(d.dev[15], 0, tk * 2 * I * 4));                                                  \
        snprintf(nm_, sizeof(nm_), "tc6<%d,%d,%d,%d> Z %u gate, up", PT, RG, KB, S, z_);                \
        cmp(15, 2 * (size_t)I, nm_,                                                                     \
            timeit([&] { k_ds_hot_tc6<false, PT, RG, KB, S><<<dim3((unsigned)(I / (64 * RG)), 2, z_), 128, \
                                                               C_::smem>>>(                             \
                             d.drec, d.denv, lex, loff, lcn, lpl, lnx); }, reps), gmac * 2 / 3);        \
        CKB(cudaMemset(d.dev[20], 0, tk * H * 4));                                                      \
        snprintf(nm_, sizeof(nm_), "tc6<%d,%d,%d,%d> Z %u down", PT, RG, KB, S, z_);                    \
        cmp(20, (size_t)H, nm_,                                                                         \
            timeit([&] { k_ds_hot_tc6<true, PT, RG, KB, S><<<dim3((unsigned)(H / (64 * RG)), z_), 128,  \
                                                              C_::smem>>>(                              \
                             d.drec, d.denv, lex, loff, lcn, lpl, lnx); }, reps), gmac / 3);            \
    } while (0)
    lists(DS_TC6_CH);
    HOT6(3, 2, 4, 2, DS_TC6_Z);
    if (t <= 8) {
        HOT6(1, 2, 4, 2, DS_TC6_Z);
        HOT6(1, 1, 4, 2, DS_TC6_Z);
        HOT6(1, 1, 4, 3, DS_TC6_Z);
    }
    /* the runner's call (the lists in its time) */
    int *hl;
    CKB(cudaMalloc(&hl, ds_hot_hl_ints(tk, E) * 4));
    CKB(cudaMemset(d.dev[15], 0, tk * 2 * I * 4));
    cmp(15, 2 * (size_t)I, "ds_hot_tc6 gate, up",
        timeit([&] { ds_hot_tc6(0, d.drec, d.denv, (const int *)d.dev[1], (const int *)d.dev[3], hl, tk, E, H, I, 0); },
               reps), gmac * 2 / 3);
    CKB(cudaMemset(d.dev[20], 0, tk * H * 4));
    cmp(20, (size_t)H, "ds_hot_tc6 down",
        timeit([&] { ds_hot_tc6(1, d.drec, d.denv, NULL, NULL, hl, tk, E, H, I, 0); }, reps), gmac / 3);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "ds_prefill attn|hot DIR [reps]\n");
        return 1;
    }
    int reps = argc > 3 ? atoi(argv[3]) : 10;
    if (!strcmp(argv[1], "attn")) {
        attn(argv[2], reps);
    } else {
        hot(argv[2], reps);
    }
    return 0;
}
