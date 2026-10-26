/* The bench of the kernel module kernels/nemotron/iq2_f32.cu (the float KQ_IQ2R products of a GPU
 * without dp4a) on random KQ_IQ2R rows: each kernel's outputs against float64 on the CPU (the same
 * rows decoded there) and its time. It builds the module alone (seconds, not the library's minutes):
 *
 *   bench/nm_f32.sh lin [reps]     k_nm_q2_lin_f (GP_KQ_LINEAR): the shared expert's matrices of
 *                                  Nemotron 3.5 Bolt (4096 x 5376, 5376 x 4096), 1, 2 and 8 tokens
 *   bench/nm_f32.sh hot [reps]     k_nm_hot_up_f, k_nm_hot_down_f (GP_NM_HOT_MOE, the tables of zero
 *                                  copy): a step's 9 pairs of a layer of 304 experts, and 4 tokens;
 *                                  the cold experts in mapped host memory (COLD of 100, default 88)
 *   bench/nm_f32.sh group [reps]   k_nm_gtile_f (GP_NM_GROUP_MOE): a group of 1024 tokens (T) on the
 *                                  experts in device memory, then in mapped host memory
 *
 * The pairs' sort (k_nm_gsort) and the sums of each token (k_nm_gsum) are the host's here. */
#include <algorithm>
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
#include "kernels/nemotron/iq2.cuh"
#include "kernels/nemotron/iq2_f32.cu"

#define CKB(x)                                                                         \
    do {                                                                               \
        cudaError_t err_ = (x);                                                        \
        if (err_ != cudaSuccess) {                                                     \
            fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(err_)); \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

static const uint64_t grid_h[256] = IQ2XXS_GRID;
static uint64_t rng = 0x9e3779b97f4a7c15ull;

static uint32_t rnd32()
{
    rng ^= rng << 13;
    rng ^= rng >> 7;
    rng ^= rng << 17;
    return (uint32_t)(rng >> 16);
}

static float rndf(float lo, float hi)
{
    return lo + (hi - lo) * (float)(rnd32() & 0xffffff) / 16777216.f;
}

/* n random KQ_IQ2R rows of cols values (iq2.cuh's layout), d in [1e-3, 1e-2) */
static std::vector<uint8_t> q2_rows(int n, int cols)
{
    const size_t rb = nm_q2_rb(cols);
    std::vector<uint8_t> w((size_t)n * rb, 0);
    for (int i = 0; i < n; ++i) {
        uint8_t *row = w.data() + (size_t)i * rb;
        for (int b = 0; b < cols / 256; ++b) {
            for (int j = 0; j < 8; ++j) {
                const uint32_t ix = rnd32(), aux = rnd32();
                memcpy(row + 64 * b + 8 * j, &ix, 4);
                memcpy(row + 64 * b + 8 * j + 4, &aux, 4);
            }
            const __half d = __float2half(rndf(1e-3f, 1e-2f));
            memcpy(row + cols / 4 + 2 * b, &d, 2);
        }
    }
    return w;
}

/* row i of rows w (cols values) decoded, float64 */
static void q2_decode(const uint8_t *w, int cols, double *out)
{
    for (int b = 0; b < cols / 256; ++b) {
        __half dh;
        memcpy(&dh, w + cols / 4 + 2 * b, 2);
        const double d = (double)__half2float(dh);
        for (int j = 0; j < 8; ++j) {
            uint32_t ix, aux;
            memcpy(&ix, w + 64 * b + 8 * j, 4);
            memcpy(&aux, w + 64 * b + 8 * j + 4, 4);
            const double db = d * (double)(2 * (aux >> 28) + 1) * 0.125;
            for (int l = 0; l < 4; ++l) {
                const uint64_t g = grid_h[(ix >> (8 * l)) & 255];
                const int s7 = (aux >> (7 * l)) & 127, sg = s7 | ((__builtin_popcount(s7) & 1) << 7);
                for (int u = 0; u < 8; ++u) {
                    const double v = (double)((g >> (8 * u)) & 255);
                    out[256 * b + 32 * j + 8 * l + u] = db * (((sg >> u) & 1) ? -v : v);
                }
            }
        }
    }
}

static double dot(const double *a, const float *x, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += a[i] * (double)x[i];
    }
    return s;
}

template <typename T>
static T *dev(const std::vector<T> &h)
{
    T *d = NULL;
    CKB(cudaMalloc(&d, h.size() * sizeof(T) + 16));
    CKB(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

/* the bytes of h in mapped host memory: its device address */
static void *mapped(const void *h, size_t n)
{
    void *p = NULL, *d = NULL;
    CKB(cudaHostAlloc(&p, n, cudaHostAllocMapped));
    memcpy(p, h, n);
    CKB(cudaHostGetDevicePointer(&d, p, 0));
    return d;
}

struct Rec {
    gp_rec h;
    Rec() { memset(&h, 0, sizeof h); }
    Rec &set(int k, int64_t v) { h.tag[k] = GP_T_INT; h.v[k] = v; return *this; }
    Rec &ptr(int k, const void *p) { return set(k, (int64_t)(intptr_t)p); }
    gp_rec *up() const
    {
        gp_rec *d = NULL;
        CKB(cudaMalloc(&d, sizeof h));
        CKB(cudaMemcpy(d, &h, sizeof h, cudaMemcpyHostToDevice));
        return d;
    }
};

static int64_t *env_d;

struct Timer {
    cudaEvent_t a, b;
    Timer() { cudaEventCreate(&a); cudaEventCreate(&b); }
    void start() { cudaEventRecord(a); }
    float ms() { cudaEventRecord(b); cudaEventSynchronize(b); float v; cudaEventElapsedTime(&v, a, b); return v; }
};

/* the largest relative error of got against want (relative to want's largest value) */
static double rel_err(const std::vector<float> &got, const std::vector<double> &want)
{
    double m = 0.0, e = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        m = fmax(m, fabs(want[i]));
        e = fmax(e, fabs((double)got[i] - want[i]));
    }
    return m > 0 ? e / m : e;
}

static int bench_lin(int reps)
{
    const int shapes[2][2] = {{4096, 5376}, {5376, 4096}};
    int bad = 0;
    for (const auto &sh : shapes) {
        const int rows = sh[0], cols = sh[1];
        std::vector<uint8_t> w = q2_rows(rows, cols);
        uint8_t *w_d = dev(w);
        std::vector<double> wf((size_t)rows * cols);
        for (int i = 0; i < rows; ++i) {
            q2_decode(w.data() + (size_t)i * nm_q2_rb(cols), cols, wf.data() + (size_t)i * cols);
        }
        for (int t : {1, 2, 8}) {
            std::vector<float> x((size_t)t * cols);
            for (auto &v : x) {
                v = rndf(-1.f, 1.f);
            }
            float *x_d = dev(x), *o_d = NULL;
            CKB(cudaMalloc(&o_d, (size_t)t * rows * 4));
            gp_rec *r = Rec().ptr(3, x_d).ptr(4, w_d).set(5, KQ_IQ2R).set(6, rows).set(7, cols).set(8, t).ptr(9, o_d).up();
            const unsigned g = (unsigned)((rows + 15) / 16);
            auto launch = [&]() {
                if (t <= 1) {
                    k_nm_q2_lin_f<1><<<g, 256>>>(r, env_d);
                } else if (t <= 2) {
                    k_nm_q2_lin_f<2><<<g, 256>>>(r, env_d);
                } else if (t <= 4) {
                    k_nm_q2_lin_f<4><<<g, 256>>>(r, env_d);
                } else {
                    k_nm_q2_lin_f<8><<<g, 256>>>(r, env_d);
                }
            };
            launch();
            CKB(cudaDeviceSynchronize());
            Timer tm;
            tm.start();
            for (int i = 0; i < reps; ++i) {
                launch();
            }
            const float ms = tm.ms() / reps;
            CKB(cudaGetLastError());
            std::vector<float> got((size_t)t * rows);
            CKB(cudaMemcpy(got.data(), o_d, got.size() * 4, cudaMemcpyDeviceToHost));
            std::vector<double> want((size_t)t * rows);
            for (int j = 0; j < t; ++j) {
                for (int i = 0; i < rows; ++i) {
                    want[(size_t)j * rows + i] = dot(wf.data() + (size_t)i * cols, x.data() + (size_t)j * cols, cols);
                }
            }
            const double e = rel_err(got, want);
            const double gb = (double)rows * nm_q2_rb(cols) / 1e9;
            printf("lin %5d x %5d, %d tokens: %8.1f us, %6.1f GB/s of weights; error %.2e%s\n", rows, cols, t,
                   1000 * ms, gb / (ms / 1000), e, e > 1e-5 ? "  FAIL" : "");
            bad |= e > 1e-5;
            cudaFree(x_d);
            cudaFree(o_d);
            cudaFree(r);
        }
        cudaFree(w_d);
    }
    return bad;
}

/* E experts of up (inner x L) and down (L x inner) rows: the tables of their device addresses, a cold
 * share of them in mapped host memory */
struct Experts {
    int E, L, inner;
    std::vector<uint8_t> up, dn;
    std::vector<int64_t> pup, pdn;
    Experts(int E_, int L_, int inner_, int cold) : E(E_), L(L_), inner(inner_)
    {
        const size_t pu = (size_t)inner * nm_q2_rb(L), pd = (size_t)L * nm_q2_rb(inner);
        up = q2_rows(E * inner, L);
        dn = q2_rows(E * L, inner);
        uint8_t *up_d = dev(up), *dn_d = dev(dn);
        uint8_t *up_m = (uint8_t *)mapped(up.data(), up.size()), *dn_m = (uint8_t *)mapped(dn.data(), dn.size());
        pup.resize(E);
        pdn.resize(E);
        for (int x = 0; x < E; ++x) {
            const bool c = (int)(rnd32() % 100) < cold;
            pup[x] = (int64_t)(intptr_t)((c ? up_m : up_d) + x * pu);
            pdn[x] = (int64_t)(intptr_t)((c ? dn_m : dn_d) + x * pd);
        }
    }
    /* pair (expert x, weight w) on the latent row lat: its down sum times w (float64), its act */
    void ref(int x, float w, const float *lat, std::vector<double> &act, std::vector<double> &out) const
    {
        std::vector<double> row(std::max(L, inner));
        std::vector<float> af(inner);
        act.assign(inner, 0.0);
        for (int i = 0; i < inner; ++i) {
            q2_decode(up.data() + ((size_t)x * inner + i) * nm_q2_rb(L), L, row.data());
            const double v = fmax(dot(row.data(), lat, L), 0.0);
            act[i] = v * v;
            af[i] = (float)act[i];
        }
        out.assign(L, 0.0);
        for (int i = 0; i < L; ++i) {
            q2_decode(dn.data() + ((size_t)x * L + i) * nm_q2_rb(inner), inner, row.data());
            out[i] = (double)w * dot(row.data(), af.data(), inner);
        }
    }
};

/* t tokens' k distinct experts of E, their weights */
static void route(int t, int k, int E, std::vector<int> &idx, std::vector<float> &val)
{
    idx.resize((size_t)t * k);
    val.resize((size_t)t * k);
    for (int j = 0; j < t; ++j) {
        for (int q = 0; q < k; ++q) {
            int x;
            bool dup;
            do {
                x = (int)(rnd32() % E);
                dup = false;
                for (int q2 = 0; q2 < q; ++q2) {
                    dup |= idx[(size_t)j * k + q2] == x;
                }
            } while (dup);
            idx[(size_t)j * k + q] = x;
            val[(size_t)j * k + q] = rndf(0.05f, 0.3f);
        }
    }
}

static int bench_hot(int reps)
{
    const int E = 304, L = 1024, inner = 2048, k = 9;
    const int cold = getenv("COLD") ? atoi(getenv("COLD")) : 88;
    Experts ex(E, L, inner, cold);
    int64_t *pup_d = dev(ex.pup), *pdn_d = dev(ex.pdn);
    int bad = 0;
    for (int t : {1, 4}) {
        std::vector<int> idx;
        std::vector<float> val, lat((size_t)t * L);
        route(t, k, E, idx, val);
        for (auto &v : lat) {
            v = rndf(-1.f, 1.f);
        }
        int *idx_d = dev(idx);
        float *val_d = dev(val), *lat_d = dev(lat), *act_d = NULL, *de_d = NULL;
        CKB(cudaMalloc(&act_d, (size_t)t * k * inner * 4));
        CKB(cudaMalloc(&de_d, (size_t)t * k * L * 4));
        /* (the float form: operands 0-12, then 22 the tables) */
        gp_rec *r = Rec().ptr(0, lat_d).ptr(1, idx_d).ptr(2, val_d).ptr(4, pup_d).ptr(5, pdn_d).ptr(6, act_d)
                        .ptr(7, de_d).set(9, k).set(10, L).set(11, inner).set(12, t).set(22, 1).up();
        auto launch = [&]() {
            k_nm_hot_up_f<<<dim3(inner / (16 * NMF_HR), t * k), 256>>>(r, env_d);
            k_nm_hot_down_f<<<dim3(L / (16 * NMF_HR), t * k), 256>>>(r, env_d);
        };
        launch();
        CKB(cudaDeviceSynchronize());
        Timer tm;
        tm.start();
        for (int i = 0; i < reps; ++i) {
            launch();
        }
        const float ms = tm.ms() / reps;
        CKB(cudaGetLastError());
        std::vector<float> got((size_t)t * k * L);
        CKB(cudaMemcpy(got.data(), de_d, got.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<double> want(got.size()), act, out;
        for (int p = 0; p < t * k; ++p) {
            ex.ref(idx[p], val[p], lat.data() + (size_t)(p / k) * L, act, out);
            for (int i = 0; i < L; ++i) {
                want[(size_t)p * L + i] = out[i];
            }
        }
        const double e = rel_err(got, want);
        const double mb = (double)t * k * ((size_t)inner * nm_q2_rb(L) + (size_t)L * nm_q2_rb(inner)) / 1e6;
        printf("hot %d tokens x %d pairs (%d%% cold, mapped): %8.1f us, %.1f MB of experts, %5.1f GB/s; error %.2e%s\n",
               t, k, cold, 1000 * ms, mb, mb / ms, e, e > 1e-5 ? "  FAIL" : "");
        bad |= e > 1e-5;
    }
    return bad;
}

static int bench_group(int reps)
{
    const int E = 304, L = 1024, inner = 2048, k = 9;
    const int t = getenv("T") ? atoi(getenv("T")) : 1024;
    int bad = 0;
    for (int cold : {0, 100}) {
        Experts ex(E, L, inner, cold);
        int64_t *pup_d = dev(ex.pup), *pdn_d = dev(ex.pdn);
        std::vector<int> idx;
        std::vector<float> val, lat((size_t)t * L);
        route(t, k, E, idx, val);
        for (auto &v : lat) {
            v = rndf(-1.f, 1.f);
        }
        /* (k_nm_gsort's work: E + 1 starts, E counts, the pairs by expert) */
        std::vector<int> work(2 * E + 1 + (size_t)t * k, 0);
        for (int p = 0; p < t * k; ++p) {
            ++work[E + 1 + idx[p]];
        }
        for (int x = 0, s = 0; x < E; ++x) {
            work[x] = s;
            s += work[E + 1 + x];
            work[E + 1 + x] = 0;
        }
        work[E] = t * k;
        for (int p = 0; p < t * k; ++p) {
            const int x = idx[p];
            work[2 * E + 1 + work[x] + work[E + 1 + x]++] = p;
        }
        int *idx_d = dev(idx), *work_d = dev(work);
        float *val_d = dev(val), *lat_d = dev(lat), *act_d = NULL, *de_d = NULL;
        CKB(cudaMalloc(&act_d, (size_t)t * k * inner * 4));
        CKB(cudaMalloc(&de_d, (size_t)t * k * L * 4));
        gp_rec *r = Rec().ptr(0, lat_d).ptr(1, idx_d).ptr(2, val_d).ptr(3, pup_d).ptr(4, pdn_d).ptr(7, act_d)
                        .ptr(10, de_d).ptr(12, work_d).set(13, k).set(14, L).set(15, inner).set(16, t).set(17, E)
                        .set(18, t).up();
        const size_t su = nmf_gtile_smem(L), sd = nmf_gtile_smem(inner);
        auto launch = [&](int which) {
            if (which & 1) {
                k_nm_gtile_f<false><<<dim3(E, inner / NMF_BM), 256, su>>>(r, env_d);
            }
            if (which & 2) {
                k_nm_gtile_f<true><<<dim3(E, L / NMF_BM), 256, sd>>>(r, env_d);
            }
        };
        launch(3);
        CKB(cudaDeviceSynchronize());
        float ms[2];
        for (int w = 0; w < 2; ++w) {
            Timer tm;
            tm.start();
            for (int i = 0; i < reps; ++i) {
                launch(1 << w);
            }
            ms[w] = tm.ms() / reps;
        }
        CKB(cudaGetLastError());
        /* (16 pairs checked: their act and de) */
        std::vector<float> ga((size_t)t * k * inner), gd((size_t)t * k * L);
        CKB(cudaMemcpy(ga.data(), act_d, ga.size() * 4, cudaMemcpyDeviceToHost));
        CKB(cudaMemcpy(gd.data(), de_d, gd.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<float> sa, sd2;
        std::vector<double> wa, wd, act, out;
        for (int s = 0; s < 16; ++s) {
            const int p = (int)(rnd32() % (t * k));
            ex.ref(idx[p], val[p], lat.data() + (size_t)(p / k) * L, act, out);
            sa.insert(sa.end(), ga.begin() + (size_t)p * inner, ga.begin() + (size_t)(p + 1) * inner);
            wa.insert(wa.end(), act.begin(), act.end());
            sd2.insert(sd2.end(), gd.begin() + (size_t)p * L, gd.begin() + (size_t)(p + 1) * L);
            wd.insert(wd.end(), out.begin(), out.end());
        }
        const double ea = rel_err(sa, wa), ed = rel_err(sd2, wd);
        const double macs = 2.0 * t * k * inner * L;
        printf("group %d tokens, experts %s: up %7.2f ms, down %7.2f ms, %5.0f GMAC/s (%.0f tok/s of the experts "
               "of 40 layers); error act %.2e de %.2e%s\n", t, cold ? "mapped (host)" : "on the GPU", ms[0], ms[1],
               macs / ((ms[0] + ms[1]) / 1000) / 1e9, t / (40 * (ms[0] + ms[1]) / 1000), ea, ed,
               ea > 1e-5 || ed > 1e-5 ? "  FAIL" : "");
        bad |= ea > 1e-5 || ed > 1e-5;
    }
    return bad;
}

int main(int argc, char **argv)
{
    const char *what = argc > 1 ? argv[1] : "lin";
    const int reps = argc > 2 ? atoi(argv[2]) : 20;
    CKB(cudaSetDeviceFlags(cudaDeviceMapHost));
    CKB(cudaMalloc(&env_d, 64 * sizeof(int64_t)));
    CKB(cudaMemset(env_d, 0, 64 * sizeof(int64_t)));
    cudaDeviceProp pr;
    CKB(cudaGetDeviceProperties(&pr, 0));
    printf("%s (sm_%d%d), %d SMs\n", pr.name, pr.major, pr.minor, pr.multiProcessorCount);
    int bad = 0;
    if (!strcmp(what, "lin") || !strcmp(what, "all")) {
        bad |= bench_lin(reps);
    }
    if (!strcmp(what, "hot") || !strcmp(what, "all")) {
        bad |= bench_hot(reps);
    }
    if (!strcmp(what, "group") || !strcmp(what, "all")) {
        bad |= bench_group(reps);
    }
    return bad;
}
