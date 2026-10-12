/* The vendor layer (REFACTOR_PLAN.md, section 4.6): the CUDA names that HIP
 * spells another way, in one place, so a HIP build later changes this file
 * and not the kernels. On NVIDIA each one is the CUDA call of before.
 *
 * RC_FULL_MASK   the mask of all the lanes of a warp (shuffles)
 * RC_WARP        the threads of a warp (32; 64 on AMD CDNA)
 * rc_globaltimer the GPU clock in ns
 * rc_nanosleep   sleep about ns (sm_70; a short spin before it)
 * RC_SM80        1 where the inline PTX of the tensor-core kernels (mma.sync,
 *                ldmatrix, cp.async) assembles: the host pass and sm_80 up.
 *                Below it (Pascal) an mma or ldmatrix helper traps, a
 *                cp.async is a plain load and store, and its commit and wait do nothing;
 *                the host runs those kernels only with caps.tc. RC_NO_SM80
 *                (RECLANK_GPU_CC below 80, build.py) takes the Pascal forms
 *                on any GPU: a tensor-core kernel the host runs traps. */
#ifndef RECLANK_PORT_CUH
#define RECLANK_PORT_CUH

#define RC_FULL_MASK 0xffffffffu
#define RC_WARP 32

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800) && !defined(RC_NO_SM80)
#define RC_SM80 1
#else
#define RC_SM80 0
#endif

__device__ __forceinline__ unsigned long long rc_globaltimer(void)
{
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__device__ __forceinline__ void rc_nanosleep(unsigned ns)
{
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 700
    __nanosleep(ns);
#else
    unsigned long long t0 = rc_globaltimer();
    while (rc_globaltimer() - t0 < ns) {
    }
#endif
}

#endif
