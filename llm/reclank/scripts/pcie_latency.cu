// SPLIT_PLAN.md, phase 0: the cost of the link between the CPU and the GPU.
//
// The hidden state of the 26B has 2816 float32 values (11 KB). The operation
// split sends it from the GPU to the CPU and back one time in each layer. This test
// measures that round trip in two ways:
//
// 1. A kernel, an asynchronous copy to pinned host memory, a sync of the
//    stream, and a copy back. This is the simple way.
// 2. One kernel that stays on the GPU. It writes the state to mapped host
//    memory and sets a flag. The CPU waits for the flag, changes the state,
//    and sets the flag again. The kernel waits for that. This way has no
//    launch and no sync in the loop.
//
// Then it measures the rate of a large copy in each direction.
//
//     nvcc -O2 -arch=native -o pcie_latency scripts/pcie_latency.cu
//     ./pcie_latency
#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>

#define N 2816

__global__ void touch(float *x)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        x[i] += 1.f;
    }
}

__global__ void pingpong(float *h, volatile int *flag, int iters)
{
    for (int it = 0; it < iters; it++) {
        for (int i = threadIdx.x; i < N; i += blockDim.x) {
            h[i] += 1.f;
        }
        __threadfence_system();
        __syncthreads();
        if (threadIdx.x == 0) {
            flag[0] = 2 * it + 1;
            while (flag[0] != 2 * it + 2) {
            }
        }
        __syncthreads();
    }
}

static double now()
{
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

int main()
{
    float *d, *h;
    cudaMalloc(&d, N * 4);
    cudaHostAlloc(&h, N * 4, cudaHostAllocMapped);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const int R = 2000;

    // 1. Copy and sync. The first pass warms up.
    for (int w = 0; w < 2; w++) {
        double t = now();
        for (int r = 0; r < R; r++) {
            touch<<<11, 256, 0, s>>>(d);
            cudaMemcpyAsync(h, d, N * 4, cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
            h[0] += 1;
            cudaMemcpyAsync(d, h, N * 4, cudaMemcpyHostToDevice, s);
        }
        cudaStreamSynchronize(s);
        if (w) {
            printf("copy and stream sync:        %.1f us for each round trip\n",
                   (now() - t) / R * 1e6);
        }
    }

    // 2. One kernel that polls mapped host memory.
    int *flag;
    cudaHostAlloc(&flag, 64, cudaHostAllocMapped);
    float *hd;
    int *fd;
    cudaHostGetDevicePointer(&hd, h, 0);
    cudaHostGetDevicePointer(&fd, flag, 0);
    for (int w = 0; w < 2; w++) {
        flag[0] = 0;
        volatile int *vf = flag;
        pingpong<<<1, 256, 0, s>>>(hd, fd, R);
        double t = now();
        for (int it = 0; it < R; it++) {
            while (vf[0] != 2 * it + 1) {
            }
            h[1] += 1;
            __sync_synchronize();
            vf[0] = 2 * it + 2;
        }
        cudaStreamSynchronize(s);
        if (w) {
            printf("kernel that polls host:      %.1f us for each round trip\n",
                   (now() - t) / R * 1e6);
        }
    }

    // 3. The rate of a large copy.
    size_t B = 256u << 20;
    float *db, *hb;
    cudaMalloc(&db, B);
    cudaHostAlloc(&hb, B, 0);
    double t = now();
    for (int i = 0; i < 4; i++) {
        cudaMemcpy(db, hb, B, cudaMemcpyHostToDevice);
    }
    printf("host to GPU, 256 MB, pinned: %.1f GB/s\n", 4.0 * B / (now() - t) / 1e9);
    t = now();
    for (int i = 0; i < 4; i++) {
        cudaMemcpy(hb, db, B, cudaMemcpyDeviceToHost);
    }
    printf("GPU to host, 256 MB, pinned: %.1f GB/s\n", 4.0 * B / (now() - t) / 1e9);
    return 0;
}
