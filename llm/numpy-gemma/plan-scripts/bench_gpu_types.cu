// Microbenchmark (scratch, not in the repo): the hot-expert products of one
// token (10 pairs) for each expert type, with the device code of gpu.cu
// (included read-only). Warp per row (kq_row: the generic path of k_kqh_gu /
// k_kqh_dn), KS warps per row (kq_row_part: a "block" form for any type), and
// for KQ_NVX warp per group (kq_nvx_group) and the block form of k_kqh_nvx.
#include "gpu.cu"   // nvcc -I ../np_gemma/csrc
#include <vector>
#include <random>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

__global__ void b_warp_row(int type, const uint8_t *w, size_t esz, size_t rb, const int *ids,
                           int rows, const float *x, int cols, float *out)
{
    int j = blockIdx.y, row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    if (row >= rows) return;
    float v = kq_row(type, w + (size_t)ids[j] * esz + (size_t)row * rb, x, cols);
    if (threadIdx.x % 32 == 0) out[(size_t)j * rows + row] = v;
}

template <int KS>
__global__ void b_block_row(int type, const uint8_t *w, size_t esz, size_t rb, const int *ids,
                            int rows, const float *x, int cols, float *out)
{
    __shared__ float red[KS];
    int j = blockIdx.y, row = blockIdx.x, warp = threadIdx.x / 32;
    float v = kq_row_part(type, w + (size_t)ids[j] * esz + (size_t)row * rb, x, cols, warp, KS);
    if (threadIdx.x % 32 == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) { float s = 0; for (int i = 0; i < KS; ++i) s += red[i]; out[(size_t)j * rows + row] = s; }
}

__global__ void b_nvx_warp(const uint8_t *w, size_t esz, size_t rb, const int *ids, int rows,
                           const float *x, int cols, float *out)
{
    int j = blockIdx.y, grp = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    if (grp >= rows / 16) return;
    float v[4];
    kq_nvx_group(w + (size_t)ids[j] * esz + (size_t)16 * grp * rb, x, cols, v);
    int l = threadIdx.x % 32;
    if (l < 4) { float *o = out + (size_t)j * rows + 16 * grp; o[2*l] = v[0]; o[2*l+1] = v[1]; o[2*l+8] = v[2]; o[2*l+9] = v[3]; }
}

template <int KS>   // the loop of k_kqh_nvx
__global__ void __launch_bounds__(32 * KS) b_nvx_block(const uint8_t *w, size_t esz, size_t rb, const int *ids,
                                                       int rows, const float *x, int cols, float *out)
{
    __shared__ float red[KS][16];
    int grp = blockIdx.x, j = blockIdx.y;
    const uint8_t *wg = w + (size_t)ids[j] * esz + (size_t)16 * grp * rb;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, st = lane / 4, q = lane % 4, h = st / 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 2
    for (int b = warp; b < cols / 32; b += KS) {
        const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
        uint2 c = *(const uint2 *)(blk + 8 * lane);
        float4 xv = *(const float4 *)(x + 32 * b + 4 * st);
        uint32_t s01 = *(const uint16_t *)(blk + 256 + 16 * h + 2 * q);
        uint32_t s89 = *(const uint16_t *)(blk + 256 + 16 * h + 8 + 2 * q);
        acc[0] += kq_e4m3s(s01 & 255) * kq_dot4s8(kt_e2m1x4(c.x), xv);
        acc[1] += kq_e4m3s(s01 >> 8) * kq_dot4s8(kt_e2m1x4(c.y), xv);
        acc[2] += kq_e4m3s(s89 & 255) * kq_dot4s8(kt_e2m1x4(c.x >> 4), xv);
        acc[3] += kq_e4m3s(s89 >> 8) * kq_dot4s8(kt_e2m1x4(c.y >> 4), xv);
    }
    for (int i = 0; i < 4; ++i) for (int o = 4; o < 32; o <<= 1) acc[i] += __shfl_xor_sync(0xffffffff, acc[i], o);
    if (lane < 4) { red[warp][2*lane] = acc[0]; red[warp][2*lane+1] = acc[1]; red[warp][2*lane+8] = acc[2]; red[warp][2*lane+9] = acc[3]; }
    __syncthreads();
    if (threadIdx.x < 16) { float s = 0; for (int i = 0; i < KS; ++i) s += red[i][threadIdx.x]; out[(size_t)j * rows + 16 * grp + threadIdx.x] = s; }
}

static size_t rbytes(int t, int cols) { return kq_row_bytes(t, cols); }

static std::vector<uint8_t> make(int t, int rows, int cols, int nexp, std::mt19937 &g)
{
    size_t n = (size_t)nexp * rows * rbytes(t, cols);
    std::vector<uint8_t> a(n);
    for (auto &b : a) b = (uint8_t)g();
    uint16_t h = 0x1419;   // a small float16
    size_t bs = t == KQ_Q8_0 ? 34 : t == KQ_Q6_K ? 210 : t == KQ_Q4_K ? 144 : t == KQ_Q5_1 ? 24 : 0;
    if (bs) {
        size_t off = t == KQ_Q6_K ? 208 : 0;
        for (size_t i = 0; i + bs <= n; i += bs) { memcpy(&a[i + off], &h, 2); if (t == KQ_Q4_K || t == KQ_Q5_1) memcpy(&a[i + 2], &h, 2); }
    } else if (t == KQ_NVX) {
        size_t gsz = 16 + (size_t)cols / 32 * 288;
        for (size_t gi = 0; gi + gsz <= n; gi += gsz) {
            float one = 0.01f; memcpy(&a[gi], &one, 4); memset(&a[gi + 4], 0, 12);
            for (int b = 0; b < cols / 32; ++b) memset(&a[gi + 16 + (size_t)b * 288 + 256], 0x38, 32);
        }
    }
    return a;
}

int main()
{
    const int H = 2560, I = 640, NEXP = 256, PAIRS = 10, ITERS = 300;
    std::mt19937 g(1);
    float *x; CK(cudaMalloc(&x, H * 4)); std::vector<float> hx(H); for (auto &v : hx) v = (g() % 2000) / 1000.f - 1.f;
    CK(cudaMemcpy(x, hx.data(), H * 4, cudaMemcpyHostToDevice));
    float *out; CK(cudaMalloc(&out, (size_t)PAIRS * H * 4));
    int *ids; CK(cudaMalloc(&ids, (size_t)ITERS * PAIRS * 4));
    std::vector<int> hid(ITERS * PAIRS); for (auto &v : hid) v = g() % NEXP;
    CK(cudaMemcpy(ids, hid.data(), hid.size() * 4, cudaMemcpyHostToDevice));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    struct M { const char *name; int type, rows, cols; };
    M ms[] = {{"gate (640x2560)", 0, I, H}, {"down (2560x640)", 0, H, I}};
    int types[] = {KQ_Q8_0, KQ_Q6_K, KQ_NVX, KQ_Q4_K, KQ_Q5_1};
    const char *tn[] = {"Q8_0", "Q6_K", "NVX", "Q4_K", "Q5_1"};
    printf("RTX: %d pairs per launch, %d experts in the slots, GB/s of the weight bytes\n", PAIRS, NEXP);
    printf("%-16s %-5s %8s %12s %12s %12s %12s\n", "matrix", "type", "MB/exp", "warp/row", "4 warps/row", "8 warps/row", "nvx block");
    for (auto &m : ms) for (int ti = 0; ti < 5; ++ti) {
        int t = types[ti];
        if (t == KQ_Q6_K && m.cols % 256) { printf("%-16s %-5s   (cols %% 256 != 0: no Q6_K)\n", m.name, tn[ti]); continue; }
        if (t == KQ_Q4_K && m.cols % 256) continue;
        size_t rb = rbytes(t, m.cols), esz = (size_t)m.rows * rb;
        auto hw = make(t, m.rows, m.cols, NEXP, g);
        uint8_t *w; CK(cudaMalloc(&w, hw.size())); CK(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice));
        double gbs[4] = {0, 0, 0, 0};
        for (int v = 0; v < 4; ++v) {
            if (v == 3 && t != KQ_NVX) continue;
            auto launch = [&](int it) {
                const int *id = ids + (size_t)it * PAIRS;
                if (t == KQ_NVX && v == 0) b_nvx_warp<<<dim3((m.rows / 16 + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, PAIRS), 32 * ROWS_PER_BLOCK>>>(w, esz, rb, id, m.rows, x, m.cols, out);
                else if (v == 0) b_warp_row<<<dim3(m.rows / ROWS_PER_BLOCK, PAIRS), 32 * ROWS_PER_BLOCK>>>(t, w, esz, rb, id, m.rows, x, m.cols, out);
                else if (v == 1) b_block_row<4><<<dim3(m.rows, PAIRS), 128>>>(t, w, esz, rb, id, m.rows, x, m.cols, out);
                else if (v == 2) b_block_row<8><<<dim3(m.rows, PAIRS), 256>>>(t, w, esz, rb, id, m.rows, x, m.cols, out);
                else if (m.cols == H) b_nvx_block<8><<<dim3(m.rows / 16, PAIRS), 256>>>(w, esz, rb, id, m.rows, x, m.cols, out);
                else b_nvx_block<4><<<dim3(m.rows / 16, PAIRS), 128>>>(w, esz, rb, id, m.rows, x, m.cols, out);
            };
            if (t == KQ_NVX && (v == 1 || v == 2)) continue;   // kq_row_part has no NVX groups
            for (int it = 0; it < 20; ++it) launch(it);
            CK(cudaDeviceSynchronize());
            cudaEventRecord(e0);
            for (int it = 0; it < ITERS; ++it) launch(it);
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms_; cudaEventElapsedTime(&ms_, e0, e1);
            gbs[v] = (double)PAIRS * esz * ITERS / (ms_ / 1e3) / 1e9;
        }
        printf("%-16s %-5s %8.2f", m.name, tn[ti], esz / 1e6);
        for (int v = 0; v < 4; ++v) if (gbs[v] > 0) printf(" %12.0f", gbs[v]); else printf(" %12s", "-");
        printf("\n");
        cudaFree(w);
    }
    return 0;
}
