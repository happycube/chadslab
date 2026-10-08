// The three zero rules of BF12 on the GPU (plan-scripts/BF12_PLAN.md): one warp
// a row, float x. ./bf12_zero_gpu DATA (the data of fmt_make_data.py and
// bf12_zero_make.py).
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#define RPB 8
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)
template <int MODE>
__global__ void k_bf12(const uint8_t *w, size_t rb, int rows, int cols, const float *x, float *out)
{
    int row = blockIdx.x * RPB + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const uint8_t *lo = w + (size_t)row * rb, *hi = lo + cols;
    float acc = 0.f;
    for (int t = lane; t < cols / 16; t += 32) {
        int g = t >> 1, h = t & 1;
        uint4 L = *(const uint4 *)(lo + 32 * g + 16 * h), H = *(const uint4 *)(hi + 16 * g);
        const uint8_t *lb = (const uint8_t *)&L, *hb = (const uint8_t *)&H;
        const float4 *xv = (const float4 *)(x + 32 * g + 16 * h);
        int E = hi[cols / 2 + g];
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            float4 v = xv[q];
            float xs[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int j = 4 * q + i, gap = (hb[j] >> (4 * h)) & 15, b = lb[j];
                uint32_t bits = ((uint32_t)(b & 0x80) << 8 | (uint32_t)((E - gap) & 255) << 7 | (b & 0x7f)) << 16;
                if (MODE == 1) bits = gap == 15 ? 0u : bits;
                if (MODE == 2) bits = (gap == 15 && b == 0x80) ? 0u : bits;
                acc = fmaf(__uint_as_float(bits), xs[i], acc);
            }
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) out[row] = acc;
}
static std::vector<uint8_t> rd(std::string p)
{
    FILE *f = fopen(p.c_str(), "rb"); if (!f) { printf("no %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET); std::vector<uint8_t> v(s);
    if (fread(v.data(), 1, s, f) != (size_t)s) exit(1); fclose(f); return v;
}
int main(int argc, char **argv)
{
    std::string Z = std::string(argc > 1 ? argv[1] : "/tmp/np_gemma_fmt") + "/", F2 = Z;
    struct K { const char *k; int r, c; } ks[] = {{"qkv", 10240, 2560}, {"ssm_out", 2560, 6144}, {"v_proj", 512, 2560}};
    const char *nm[3] = {"none (flush)", "gap15 = 0", "neg0 = 0"};
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (auto &K : ks) {
        int r = K.r, c = K.c; size_t rb = (c + c / 2 + c / 32 + 15) / 16 * 16;
        auto hx = rd(F2 + K.k + ".x.bin");
        float *x, *out; CK(cudaMalloc(&x, hx.size())); CK(cudaMalloc(&out, r * 4));
        CK(cudaMemcpy(x, hx.data(), hx.size(), cudaMemcpyHostToDevice));
        uint8_t *w[3]; size_t nb = 0; int cp = 0; double err[3];
        for (int m = 0; m < 3; ++m) {
            auto hw = rd(Z + K.k + ".z" + std::to_string(m) + ".bin"); nb = hw.size();
            cp = std::max(1, (int)(400000000 / nb)); if (cp > 64) cp = 64;
            CK(cudaMalloc(&w[m], nb * cp));
            for (int i = 0; i < cp; ++i) CK(cudaMemcpy(w[m] + i * nb, hw.data(), nb, cudaMemcpyHostToDevice));
            auto hy = rd(Z + K.k + ".zy" + std::to_string(m) + ".bin"); const float *y = (const float *)hy.data();
            dim3 g((r + RPB - 1) / RPB);
            if (m == 0) k_bf12<0><<<g, 32 * RPB>>>(w[m], rb, r, c, x, out);
            else if (m == 1) k_bf12<1><<<g, 32 * RPB>>>(w[m], rb, r, c, x, out);
            else k_bf12<2><<<g, 32 * RPB>>>(w[m], rb, r, c, x, out);
            std::vector<float> ho(r); CK(cudaMemcpy(ho.data(), out, r * 4, cudaMemcpyDeviceToHost));
            double mx = 0, ym = 0; for (int i = 0; i < r; ++i) { mx = std::max(mx, (double)fabs(ho[i] - y[i])); ym = std::max(ym, (double)fabs(y[i])); }
            err[m] = mx / ym;
        }
        std::vector<float> ts[3];
        for (int rnd = 0; rnd < 30; ++rnd) for (int m = 0; m < 3; ++m) {
            dim3 g((r + RPB - 1) / RPB); const int IT = 100;
            auto L = [&](int i) { const uint8_t *wi = w[m] + (size_t)(i % cp) * nb;
                if (m == 0) k_bf12<0><<<g, 32 * RPB>>>(wi, rb, r, c, x, out);
                else if (m == 1) k_bf12<1><<<g, 32 * RPB>>>(wi, rb, r, c, x, out);
                else k_bf12<2><<<g, 32 * RPB>>>(wi, rb, r, c, x, out); };
            for (int i = 0; i < 5; ++i) L(i);
            cudaEventRecord(e0); for (int i = 0; i < IT; ++i) L(i); cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
            float ms; cudaEventElapsedTime(&ms, e0, e1); ts[m].push_back(ms * 1e3f / IT);
        }
        printf("\n%s %d x %d (%d copies cycled, DRAM)\n", K.k, r, c, cp);
        std::vector<float> b = ts[0]; std::sort(b.begin(), b.end()); float base = b[b.size() / 2];
        for (int m = 0; m < 3; ++m) {
            std::vector<float> t = ts[m]; std::sort(t.begin(), t.end());
            float med = t[t.size() / 2];
            printf("  %-14s median %7.2f us (p10 %6.2f, p90 %6.2f)  x%.3f  %4.0f GB/s  max rel diff %.1e\n", nm[m], med,
                   t[t.size() / 10], t[t.size() * 9 / 10], med / base, nb / med / 1e3, err[m]);
        }
        for (int m = 0; m < 3; ++m) cudaFree(w[m]);
        cudaFree(x); cudaFree(out);
    }
    return 0;
}
