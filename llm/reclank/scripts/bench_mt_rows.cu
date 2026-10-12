// k_mt_int4_rows, the int4 kernel of a small group (an MTP verify group of
// 2 to 8 tokens), with one and two rows a warp: each row of each token must
// get the bits of a step (k_mt_int4_rows<1> on that token alone), and the
// time of both, on int4 matrices of the shapes of the Gemma 4 26B
// (mt_rows_r takes 2 rows a warp from 2048 rows).
//
//     nvcc -O3 -arch=native -I ../reclank/backends/include -I ../reclank/backends/cuda/csrc \
//          -o bench_mt_rows bench_mt_rows.cu
//     ./bench_mt_rows
#include "../reclank/backends/cuda/csrc/reclank_cuda.cu"
#include <vector>
#include <cstdio>
#include <cstring>
#include <random>

template <int NT, int R>
static void go(const float *x, const uint8_t *w, float *o, int rows, int cols)
{
    unsigned grid = (unsigned)((rows + ROWS_PER_BLOCK * R - 1) / (ROWS_PER_BLOCK * R));
    k_mt_int4_rows<NT, R><<<grid, 256>>>(x, w, o, rows, cols);
}

template <int NT, int R>
static float timeit(const float *x, const uint8_t *w, float *o, int rows, int cols)
{
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int i = 0; i < 50; ++i) go<NT, R>(x, w, o, rows, cols);
    cudaEventRecord(e0);
    for (int i = 0; i < 400; ++i) go<NT, R>(x, w, o, rows, cols);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms;
    cudaEventElapsedTime(&ms, e0, e1);
    return ms * 2.5f;     // us a call
}

/* 1 when the rows of token j of out (NT tokens) are those of a step on it */
template <int NT>
static int as_step(const float *dx, const uint8_t *dw, const float *out, float *d1, int rows, int cols)
{
    std::vector<float> a((size_t)NT * rows), b(rows);
    cudaMemcpy(a.data(), out, a.size() * 4, cudaMemcpyDeviceToHost);
    int same = 1;
    for (int j = 0; j < NT; ++j) {
        go<1, 1>(dx + (size_t)j * cols, dw, d1, rows, cols);
        cudaMemcpy(b.data(), d1, b.size() * 4, cudaMemcpyDeviceToHost);
        same &= memcmp(a.data() + (size_t)j * rows, b.data(), rows * 4) == 0;
    }
    return same;
}

template <int NT>
static int one(const float *dx, const uint8_t *dw, float *d0, float *d1, int rows, int cols)
{
    go<NT, 1>(dx, dw, d0, rows, cols);
    int s1 = as_step<NT>(dx, dw, d0, d1, rows, cols);
    go<NT, 2>(dx, dw, d0, rows, cols);
    int s2 = as_step<NT>(dx, dw, d0, d1, rows, cols);
    printf("  t %d: R1 %5.1f us, R2 %5.1f us; as a step: %s\n", NT, timeit<NT, 1>(dx, dw, d0, rows, cols),
           timeit<NT, 2>(dx, dw, d0, rows, cols), s1 && s2 ? "same bits" : "DIFFERENT");
    return !(s1 && s2);
}

int main()
{
    const int shapes[][2] = {{4096, 2816}, {2816, 4096}, {8192, 2816}, {2816, 8192}, {2112, 2816}, {1024, 2816}};
    std::mt19937 rng(3);
    int bad = 0;
    for (auto &s : shapes) {
        int rows = s[0], cols = s[1];
        size_t nb = (size_t)rows * cols / 32 * 18;
        std::vector<uint8_t> w(nb);
        for (size_t i = 0; i < nb; ++i) w[i] = rng() & 255;
        for (size_t b = 0; b < nb / 18; ++b) {     // finite float16 scales
            uint16_t h = 0x2000 | (rng() & 0x0fff);
            memcpy(&w[b * 18], &h, 2);
        }
        std::vector<float> x(8 * (size_t)cols);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto &v : x) v = nd(rng);
        uint8_t *dw;
        float *dx, *d0, *d1;
        cudaMalloc(&dw, nb);
        cudaMalloc(&dx, x.size() * 4);
        cudaMalloc(&d0, 8 * (size_t)rows * 4);
        cudaMalloc(&d1, (size_t)rows * 4);
        cudaMemcpy(dw, w.data(), nb, cudaMemcpyHostToDevice);
        cudaMemcpy(dx, x.data(), x.size() * 4, cudaMemcpyHostToDevice);
        printf("rows %d cols %d\n", rows, cols);
        bad += one<2>(dx, dw, d0, d1, rows, cols);
        bad += one<3>(dx, dw, d0, d1, rows, cols);
        bad += one<4>(dx, dw, d0, d1, rows, cols);
        bad += one<5>(dx, dw, d0, d1, rows, cols);
        bad += one<6>(dx, dw, d0, d1, rows, cols);
        bad += one<8>(dx, dw, d0, d1, rows, cols);
        cudaFree(dw);
        cudaFree(dx);
        cudaFree(d0);
        cudaFree(d1);
    }
    printf("%s\n", bad ? "FAIL" : "PASS");
    return bad;
}
