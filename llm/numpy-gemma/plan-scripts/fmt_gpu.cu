// Test kernels (plan-scripts/BF12_PLAN.md; not runtime code): BF12, RQ12, RQ10 on the GPU, one
// warp a row, float x; against kq_row (Q8_0, RQ8_0, BF16) of gpu.cu (included
// read-only). Layouts: make_data.py.
#include "gpu.cu"   // nvcc -I ../np_gemma/csrc (run_fmt.sh)
#include <vector>
#include <string>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

// lane l of a pass: group 16 p + l / 2, half l & 1 (16 values)
template <int FMT>   // 0 BF12, 12 RQ12, 10 RQ10
__global__ void k_fmt(const uint8_t *w, size_t rb, int rows, int cols, const float *x, float *out)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) return;
    const uint8_t *lo = w + (size_t)row * rb, *hi = lo + cols;
    int ng = cols / 32;
    float acc = 0.f;
    for (int t = lane; t < 2 * ng; t += 32) {
        int g = t >> 1, h = t & 1;
        uint4 L = *(const uint4 *)(lo + 32 * g + 16 * h);
        const uint8_t *lb = (const uint8_t *)&L;
        const float4 *xv = (const float4 *)(x + 32 * g + 16 * h);
        float xs[16];
        #pragma unroll
        for (int i = 0; i < 4; ++i) { float4 v = xv[i]; xs[4*i] = v.x; xs[4*i+1] = v.y; xs[4*i+2] = v.z; xs[4*i+3] = v.w; }
        if (FMT == 0 || FMT == 12) {
            uint4 Hv = *(const uint4 *)(hi + 16 * g);
            const uint8_t *hb = (const uint8_t *)&Hv;
            if (FMT == 0) {
                int E = hi[cols / 2 + g];
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    int gap = (hb[j] >> (4 * h)) & 15, b = lb[j];
                    uint32_t bits = ((uint32_t)(b & 0x80) << 8 | (uint32_t)((E - gap) & 255) << 7 | (b & 0x7f)) << 16;
                    bits = (gap == 15 && b == 0x80) ? 0u : bits;      // neg0: the zero code
                    acc = fmaf(__uint_as_float(bits), xs[j], acc);
                }
            } else {
                float d = kq_half(hi + cols / 2 + 2 * g), s = 0.f;
                #pragma unroll
                for (int j = 0; j < 16; ++j) s = fmaf((float)(lb[j] + 256 * ((hb[j] >> (4 * h)) & 15) - 2048), xs[j], s);
                acc = fmaf(d, s, acc);
            }
        } else {
            uint2 Hv = *(const uint2 *)(hi + 8 * g);
            const uint8_t *hb = (const uint8_t *)&Hv;
            float d = kq_half(hi + cols / 4 + 2 * g), s = 0.f;
            #pragma unroll
            for (int j = 0; j < 16; ++j) {
                int v = 16 * h + j;   // value in the group: byte v % 8, bits 2 (v / 8)
                s = fmaf((float)(lb[j] + 256 * ((hb[v & 7] >> (2 * (v >> 3))) & 3) - 512), xs[j], s);
            }
            acc = fmaf(d, s, acc);
        }
    }
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffff, acc, o);
    if (lane == 0) out[row] = acc;
}

__global__ void k_ref(int type, const uint8_t *w, size_t rb, int rows, int cols, const float *x, float *out)
{
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    if (row >= rows) return;
    float v = kq_row(type, w + (size_t)row * rb, x, cols);
    if (threadIdx.x % 32 == 0) out[row] = v;
}

static std::string DIR = "/tmp/np_gemma_fmt/";   // argv[1] (the data of fmt_make_data.py)
static std::vector<uint8_t> rd(const std::string &k, const char *n)
{
    FILE *f = fopen((DIR + k + "." + n + ".bin").c_str(), "rb"); if (!f) { printf("no %s.%s\n", k.c_str(), n); exit(1); }
    fseek(f, 0, SEEK_END); long s = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> v(s); if (fread(v.data(), 1, s, f) != (size_t)s) exit(1); fclose(f); return v;
}

int main(int argc, char **argv)
{
    if (argc > 1) DIR = std::string(argv[1]) + "/";
    struct K { const char *k; int rows, cols, copies; size_t rb12, rb10, rbbf; };
    K ks[] = {{"qkv", 10240, 2560, 8, 0, 0, 0}, {"ssm_out", 2560, 6144, 8, 0, 0, 0}, {"v_proj", 512, 2560, 64, 0, 0, 0}};
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (auto &K : ks) {
        int r = K.rows, c = K.cols;
        auto pad = [](size_t n) { return (n + 15) / 16 * 16; };
        size_t rbf[3] = {pad(c + c / 2 + c / 32), pad(c + c / 2 + c / 16), pad(c + c / 4 + c / 16)};
        auto hx = rd(K.k, "x"), hxr = rd(K.k, "xr"), hy = rd(K.k, "y");
        float *x, *xr, *out; CK(cudaMalloc(&x, hx.size())); CK(cudaMalloc(&xr, hxr.size())); CK(cudaMalloc(&out, r * 4));
        CK(cudaMemcpy(x, hx.data(), hx.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(xr, hxr.data(), hxr.size(), cudaMemcpyHostToDevice));
        const float *y = (const float *)hy.data();
        printf("\n%s: %d x %d, %d copies cycled; output error against the exact bf16 product\n", K.k, r, c, K.copies);
        printf("  %-22s %10s %9s %9s %8s\n", "kernel", "error", "us", "GB/s", "bytes");
        struct V { const char *name, *file; int kind; size_t rb; bool rot; };
        V vs[] = {{"Q8_0 (kq_row, float x)", "q8", KQ_Q8_0, (size_t)c / 32 * 34, false},
                  {"RQ8_0 (kq_row, float x)", "rq8", KQ_Q8_0, (size_t)c / 32 * 34, true},
                  {"bf16 (kq_row, float x)", "bf16", KQ_BF16, (size_t)c * 2, false},
                  {"BF12 (test, float x)", "bf12", -1, rbf[0], false},
                  {"RQ12 (test, float x)", "rq12", -12, rbf[1], true},
                  {"RQ10 (test, float x)", "rq10", -10, rbf[2], true}};
        for (auto &v : vs) {
            auto hw = rd(K.k, v.file);
            size_t nb = hw.size();
            uint8_t *w; CK(cudaMalloc(&w, nb * K.copies));
            for (int i = 0; i < K.copies; ++i) CK(cudaMemcpy(w + i * nb, hw.data(), nb, cudaMemcpyHostToDevice));
            const float *xx = v.rot ? xr : x;
            auto launch = [&](int i) {
                const uint8_t *wi = w + (size_t)(i % K.copies) * nb;
                dim3 grid((r + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK);
                if (v.kind >= 0) k_ref<<<grid, 32 * ROWS_PER_BLOCK>>>(v.kind, wi, v.rb, r, c, xx, out);
                else if (v.kind == -1) k_fmt<0><<<grid, 32 * ROWS_PER_BLOCK>>>(wi, v.rb, r, c, xx, out);
                else if (v.kind == -12) k_fmt<12><<<grid, 32 * ROWS_PER_BLOCK>>>(wi, v.rb, r, c, xx, out);
                else k_fmt<10><<<grid, 32 * ROWS_PER_BLOCK>>>(wi, v.rb, r, c, xx, out);
            };
            launch(0); CK(cudaDeviceSynchronize());
            std::vector<float> ho(r); CK(cudaMemcpy(ho.data(), out, r * 4, cudaMemcpyDeviceToHost));
            double num = 0, den = 0; for (int i = 0; i < r; ++i) { double dd = ho[i] - (double)y[i]; num += dd * dd; den += (double)y[i] * y[i]; }
            const int IT = 400;
            for (int i = 0; i < 20; ++i) launch(i);
            cudaEventRecord(e0); for (int i = 0; i < IT; ++i) launch(i); cudaEventRecord(e1);
            CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            double us = ms * 1e3 / IT;
            printf("  %-22s %7.2f dB %9.1f %9.0f %7.1fM\n", v.name, 10 * log10(num / den), us, nb / us / 1e3, nb / 1e6);
            cudaFree(w);
        }
        cudaFree(x); cudaFree(xr); cudaFree(out);
    }
    return 0;
}
