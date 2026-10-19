/* The program interpreter of reclank on a CUDA GPU.
 *
 * SPLIT_PLAN.md, phase 3. The CPU interpreter (gemma_run in bf16_linear.c)
 * runs the records of a program in one OpenMP region. This file runs the
 * same records on the GPU. The Python side (reclank/backends/cuda/runtime.py) copies the
 * weights and the buffers of the program to the GPU. It changes each host
 * address of the records to the device address of the same data.
 *
 * Each kernel gets the address of its record and of the environment in
 * device memory. The kernel reads its operands from the record, as gp_step
 * does on the CPU. Thus the arguments of each launch stay the same from one
 * step to the next. Only the environment changes: the position, and the
 * addresses of buffers that can move, such as the cache. The first run
 * records the launches in a CUDA graph. Each later run writes the
 * environment and starts the graph with one call.
 *
 * The scalar operations (codes 1 to 15) run on the host before the launches.
 * Each of them writes a new slot of the environment, so their order does not
 * depend on the kernels.
 *
 * The kernels do not give the bits of the CPU kernels, because they add the
 * values in a different order. scripts/check_gpu.py compares the two.
 *
 * The file covers the operations of a decode step of one token of the E4B
 * model, and of the 26B model without the experts. The output head is
 * rc_gpu_q6k_head, or rc_gpu_q4_head for a file that keeps the head in Q4_0. The
 * launch of any other operation gives an error.
 *
 * A step of the 26B keeps its experts on the CPU (SPLIT_PLAN.md, phase 4).
 * Three records of the program move the work to the CPU and back:
 *
 * - GP_TO_HOST copies the input of the experts to pinned host memory and
 *   records an event.
 * - GP_CPU_JOIN waits for that event and runs a CPU program (gemma_run of
 *   the CPU library) that computes the experts.
 * - GP_TO_DEV copies the output of the experts back to the GPU.
 *
 * The runner launches the kernels after a GP_TO_HOST before it runs the
 * GP_CPU_JOIN. Thus the GPU computes the dense feed-forward part while the
 * CPU computes the experts. The kernels between two such records are one
 * segment, and each segment has its own CUDA graph.
 */
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* The host clock, in s. */
static double gg_clock(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* GP_NARG, GP_MAGIC, gp_rec, and the codes GP_*: the table of
 * reclank/ir/opcodes.py. */
#include "opcodes.h"
#include "port.cuh"

/* The stream of the programs: the calling thread's (gg_tstream: lane B of
 * docs/LANES_PLAN.md, rc_gpu_lane), else the model's. */
static cudaStream_t gg_main_stream;
/* The device of this copy of the library (rc_gpu_init): the threads it starts
 * select it first, as CUDA gives a new thread device 0 (the 3060 and the 1070
 * of a box both visible, the steps on the 3060: a copy worker on the 1070). */
static int gg_device;
static __thread cudaStream_t gg_tstream;
#define gg_stream (gg_tstream != NULL ? gg_tstream : gg_main_stream)
static char gg_error[256];

#define CK(x)                                                                   \
    do {                                                                        \
        cudaError_t err_ = (x);                                                 \
        if (err_ != cudaSuccess) {                                              \
            snprintf(gg_error, sizeof(gg_error), "%s: %s", #x,                  \
                     cudaGetErrorString(err_));                                 \
            return -1;                                                          \
        }                                                                       \
    } while (0)

/* Where a run is (rc_gpu_where): the program, and the record (and its
 * operation) that the host queued last, in the segment [s0, s1) of its
 * launches. The kernels run after the host queues them, so a kernel that
 * fails shows its error at a later record: the record at fault is at or
 * before this one (since the last wait). RECLANK_GPU_SYNC_CHECK=1: no
 * graphs, and a wait after each record, so that an error names the record
 * that made it (slow: for a replay of a crash). */
static const void *gg_where_prog;
static int gg_where_pc = -1, gg_where_op = -1, gg_where_s0 = -1, gg_where_s1 = -1;
static int gg_sync_chk = -1;

static int gg_sync_check(void)
{
    if (gg_sync_chk < 0) {
        const char *v = getenv("RECLANK_GPU_SYNC_CHECK");
        gg_sync_chk = v != NULL && *v != 0 && strcmp(v, "0") != 0;
    }
    return gg_sync_chk;
}

static void gg_at(const void *prog, int pc, int op, int s0, int s1)
{
    gg_where_prog = prog;
    gg_where_pc = pc;
    gg_where_op = op;
    gg_where_s0 = s0;
    gg_where_s1 = s1;
}

/* RECLANK_GPU_SYNC_CHECK: wait for the record pc; its error names it. */
#define GG_SYNC_AT(pc, op)                                                      \
    do {                                                                        \
        if (gg_sync_check()) {                                                  \
            cudaError_t err_ = cudaStreamSynchronize(gg_stream);                \
            if (err_ == cudaSuccess) {                                          \
                err_ = cudaGetLastError();                                      \
            }                                                                   \
            if (err_ != cudaSuccess) {                                          \
                snprintf(gg_error, sizeof(gg_error),                            \
                         "record %d (operation %d), RECLANK_GPU_SYNC_CHECK: %s", \
                         (pc), (op), cudaGetErrorString(err_));                 \
                return -1;                                                      \
            }                                                                   \
        }                                                                       \
    } while (0)
