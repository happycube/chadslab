// SPLIT_PLAN.md, phase 0: the rate of an int4 GEMV on the GPU.
//
// The decode step reads each weight one time, so its speed is the rate at
// which the GEMV reads the weights. This test runs the matrices of the GPU
// part of the 26B for one token: about 1.6 GB of Q4_0 blocks. A block holds
// 32 weights in 18 bytes: a float16 scale and 16 bytes of 4-bit values. The
// test reports the time and the rate against the 448 GB/s of the memory of
// the RTX 5060 Ti.
//
// A warp computes one row. The lanes first copy the row to shared memory with
// loads of 4 bytes. A block of 18 bytes is not aligned for wider loads. Four
// lanes then share a block, so the lanes of a warp read adjacent values of x.
// A shuffle adds the sums of the lanes at the end.
//
// A first form, with one block for each lane, gave only about 58 GB/s. Each
// read of x by a warp then touched 32 cache lines. The test does not check
// the result against the CPU. That is for the kernels of phase 3.
//
//     nvcc -O3 -arch=native -o gpu_gemv_rate scripts/gpu_gemv_rate.cu
//     ./gpu_gemv_rate [megabytes]
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define PER_BLOCK 8
#define MAX_ROW_WORDS 1024

__global__ void gemv_q4_0(const uint8_t *w, const float *x, float *out, int rows, int cols)
{
    __shared__ uint32_t stage[PER_BLOCK][MAX_ROW_WORDS];
    int wid = threadIdx.x / 32;
    int row = blockIdx.x * PER_BLOCK + wid;
    int lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    int blocks = cols / 32;
    int words = blocks * 18 / 4;
    const uint32_t *wr = (const uint32_t *)(w + (size_t)row * blocks * 18);
    uint32_t *sw = stage[wid];
    for (int i = lane; i < words; i += 32) {
        sw[i] = wr[i];
    }
    __syncwarp();
    const uint8_t *sb = (const uint8_t *)sw;
    /* Four lanes share a block, and a warp does 8 blocks at a time. Lane sub
     * reads bytes 4 sub to 4 sub + 3 of the 16 bytes of values. The low 4 bits
     * of byte i are weight i, and the high 4 bits are weight i + 16. */
    int sub = lane & 3;
    float sum = 0.f;
    for (int b = lane >> 2; b < blocks; b += 8) {
        const uint8_t *blk = sb + b * 18;
        float d = __half2float(__ushort_as_half(*(const uint16_t *)blk));
        const uint16_t *qp = (const uint16_t *)(blk + 2 + 4 * sub);
        uint32_t q = (uint32_t)qp[0] | ((uint32_t)qp[1] << 16);
        float4 xl = *(const float4 *)(x + b * 32 + 4 * sub);
        float4 xh = *(const float4 *)(x + b * 32 + 16 + 4 * sub);
        float acc = (float)((int)(q & 15) - 8) * xl.x
                  + (float)((int)((q >> 8) & 15) - 8) * xl.y
                  + (float)((int)((q >> 16) & 15) - 8) * xl.z
                  + (float)((int)((q >> 24) & 15) - 8) * xl.w
                  + (float)((int)((q >> 4) & 15) - 8) * xh.x
                  + (float)((int)((q >> 12) & 15) - 8) * xh.y
                  + (float)((int)((q >> 20) & 15) - 8) * xh.z
                  + (float)((int)((q >> 28) & 15) - 8) * xh.w;
        sum += d * acc;
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, o);
    }
    if (lane == 0) {
        out[row] = sum;
    }
}

int main(int argc, char **argv)
{
    size_t mb = argc > 1 ? (size_t)atoi(argv[1]) : 1600;
    const int cols = 2816;
    size_t row_bytes = (size_t)(cols / 32) * 18;
    int rows = (int)(mb * 1000000 / row_bytes);
    size_t bytes = (size_t)rows * row_bytes;
    uint8_t *w;
    float *x, *out;
    if (cudaMalloc(&w, bytes) != cudaSuccess) {
        printf("no memory for %zu MB\n", mb);
        return 1;
    }
    cudaMalloc(&x, cols * 4);
    cudaMalloc(&out, (size_t)rows * 4);
    cudaMemset(w, 0x11, bytes);
    cudaMemset(x, 0, cols * 4);
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    int per_block = PER_BLOCK;
    int grid = (rows + per_block - 1) / per_block;
    for (int rep = 0; rep < 3; rep++) {
        cudaEventRecord(a);
        for (int it = 0; it < 10; it++) {
            gemv_q4_0<<<grid, 32 * per_block>>>(w, x, out, rows, cols);
        }
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        ms /= 10;
        printf("%zu MB of Q4_0 blocks, %d rows of %d: %.2f ms, %.0f GB/s\n",
               bytes / 1000000, rows, cols, ms, bytes / (ms * 1e6));
    }
    return 0;
}
