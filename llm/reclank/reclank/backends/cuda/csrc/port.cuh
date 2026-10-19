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
 *                on any GPU: a tensor-core kernel the host runs traps.
 * rc_dp4a        c + the sum of the products of the 4 signed bytes of a and b
 *                (dp4a, sm_61); rc_dp4a_us: a's bytes unsigned. Below sm_61, or
 *                with RC_NO_DP4A (RECLANK_GPU_CC below 61, build.py: the Maxwell
 *                tier on a newer GPU), a scalar form, for correctness only (about
 *                12 slots for 4 products where a float FMA is 1 a product: no path
 *                picks it for speed). That form traps unless rc_dp4a_emu_d is set
 *                (RECLANK_GPU_DP4A_EMU=1, caps.cu): a dp4a kernel there is a wrong
 *                dispatch, and the switch keeps the slow form for the families
 *                that have no float path yet (MAXWELL_PLAN.md, section 3.1). */
#ifndef RECLANK_PORT_CUH
#define RECLANK_PORT_CUH

#define RC_FULL_MASK 0xffffffffu
#define RC_WARP 32

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800) && !defined(RC_NO_SM80)
#define RC_SM80 1
#else
#define RC_SM80 0
#endif

#if (!defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 610) && !defined(RC_NO_DP4A)
#define RC_DP4A 1
#else
#define RC_DP4A 0
#endif

__device__ int rc_dp4a_emu_d;      /* 1: the scalar dp4a runs, 0: it traps (RECLANK_GPU_DP4A_EMU) */

__device__ __forceinline__ int rc_dp4a(int a, int b, int c)
{
#if RC_DP4A
    return __dp4a(a, b, c);
#else
    if (!rc_dp4a_emu_d) {
        __trap();
    }
    const char4 x = *(const char4 *)&a, y = *(const char4 *)&b;
    return c + x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
#endif
}

__device__ __forceinline__ int rc_dp4a_us(uint32_t a, uint32_t b, int c)
{
#if RC_DP4A
    int d;
    asm("dp4a.u32.s32 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
#else
    if (!rc_dp4a_emu_d) {
        __trap();
    }
    const uchar4 x = *(const uchar4 *)&a;
    const char4 y = *(const char4 *)&b;
    return c + x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
#endif
}

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
