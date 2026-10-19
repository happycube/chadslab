/* Multiply x by W for the Gemma 4 NumPy runtime.
 *
 * This file gives these kernels:
 *   1. A bfloat16 GEMV kernel. It reads four output rows in one loop. Thus the
 *      loop loads x one time for four rows.
 *   2. A bfloat16 GEMM kernel for a prompt with many tokens.
 *   3. An int8 kernel for weights with float32 activations. A one-row dot
 *      serves a decode step. A K-vectorized tile and a multi-level GEMM serve
 *      a prompt.
 *   4. An int4 kernel for weights with float32 activations. A four-row dot
 *      serves a decode step. A multi-level GEMM decodes a row block to a
 *      float32 panel and serves a prompt.
 *   5. An int4 kernel that quantizes the activations to int8. The pointwise
 *      loop then uses integer multiply and add. The lanes of the accumulator
 *      hold the tokens, so the kernel sums over k with no horizontal sum. The
 *      tile serves a prompt. Set RECLANK_INT4_Q8=1 to select it.
 *   6. A float32 kernel for a comparison.
 *   7. A Q6_K kernel for the tied output head. It decodes a 210-byte block in
 *      the registers. It reads the weights in place. Thus the load step does
 *      no dequantize of the head.
 *   8. An RMSNorm kernel and a GELU kernel. The model calls these functions
 *      for each layer. A NumPy call on a small array costs more than the work.
 *   9. A flash attention kernel for a prompt. It keeps the scores of one block
 *      at a time and it reads only the keys that the block can see. It has an
 *      AVX-512 version, an AVX2 version, and a straight C version.
 *
 * Each kernel has an AVX-512 version and an AVX2 version. The code selects the
 * AVX-512 version at run time. If the CPU does not give AVX-512, the code uses
 * the AVX2 version. Every target machine gives AVX2.
 *
 * Build this file with:
 *   cc -O3 -mavx2 -mfma -fopenmp -shared -fPIC -o libgemma.so bf16_linear.c
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE         /* sched_setaffinity, cpu_set_t (gp_team_pin) */
#endif
#include <sched.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <dlfcn.h>
#include <pthread.h>
