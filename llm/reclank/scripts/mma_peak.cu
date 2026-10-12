// SPLIT_PLAN.md, phase 5: the peak rate of the tensor cores of this GPU.
//
// Each warp runs many mma.sync instructions in a loop, with no memory
// access. The test prints the rate of three forms: float16 inputs with
// float32 sums, float16 inputs with float16 sums, and int8 inputs with int32
// sums. On the RTX 5060 Ti of jackal: 37, 38, and 204 T(FL)OPS.
//
//     nvcc -O3 -arch=native -o mma_peak scripts/mma_peak.cu && ./mma_peak
#include <cstdint>
#include <cstdio>
#include <cuda_fp16.h>
__global__ void k(float *out, int iters, int acc16) {
  uint32_t a[4] = {0x3c003c00u, 0x3c003c00u, 0x3c003c00u, 0x3c003c00u}, b[2] = {0x3c003c00u, 0x3c003c00u};
  float c[8][4] = {}; uint32_t h[8][2] = {};
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      if (acc16) asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};" : "+r"(h[j][0]), "+r"(h[j][1]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
      else asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+f"(c[j][0]),"+f"(c[j][1]),"+f"(c[j][2]),"+f"(c[j][3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
  }
  float s = 0; for (int j = 0; j < 8; ++j) s += c[j][0] + h[j][0]; if (s == 1.2345f) out[0] = s;
}
__global__ void k8(float *out, int iters) {
  uint32_t a[4] = {0x01010101u,0x01010101u,0x01010101u,0x01010101u}, b[2] = {0x01010101u,0x01010101u};
  int c[8][4] = {};
  for (int i = 0; i < iters; ++i) {
    #pragma unroll
    for (int j = 0; j < 8; ++j)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};" : "+r"(c[j][0]),"+r"(c[j][1]),"+r"(c[j][2]),"+r"(c[j][3]) : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
  }
  int s = 0; for (int j = 0; j < 8; ++j) s += c[j][0]; if (s == 12345) out[0] = s;
}
int main() {
  float *o; cudaMalloc(&o, 4); cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int blocks = 36 * 16, threads = 256, iters = 4096;
  for (int mode = 0; mode < 3; ++mode) {
    for (int r = 0; r < 2; ++r) {
      cudaEventRecord(a);
      if (mode < 2) k<<<blocks, threads>>>(o, iters, mode); else k8<<<blocks, threads>>>(o, iters);
      cudaEventRecord(b); cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
      double flop = (double)blocks * threads / 32 * iters * 8 * 2 * 16 * 8 * (mode == 2 ? 32 : 16);
      if (r) printf("%s: %.1f T(FL)OPS\n", mode == 0 ? "fp16 in, fp32 sum" : mode == 1 ? "fp16 in, fp16 sum" : "int8 in, int32 sum", flop / ms / 1e9);
    }
  }
}
