/* The tensor-core fragments and the E2M1 decode that the kernels share
 * (included first by reclank_cuda.cu; alone by a bench of a kernel module,
 * kernels/dsv41/prefill.cu). The layouts of the fragments: the comments of
 * kernels/gemm/int4_tc.cu (m16n8k16) and kernels/gemm/int4_q8.cu (m16n8k32).
 * And the copies to shared memory (cp.async). */
#pragma once

/* m16n8k16, float16, float32 sums */
__device__ __forceinline__ void mma16816(float *c, const uint32_t *a, const uint32_t *b)
{
#if RC_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

/* m16n8k8, tf32, float32 sums */
__device__ __forceinline__ void mma1688_tf32(float *c, const uint32_t *a, const uint32_t *b)
{
#if RC_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

/* m16n8k32, int8, int32 sums */
__device__ __forceinline__ void mma16832(int *c, const uint32_t *a, const uint32_t *b)
{
#if RC_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

/* 4 8 x 8 tiles of float16 from shared memory, transposed */
__device__ __forceinline__ void ldsm_x4_trans(uint32_t *d, const void *p)
{
#if RC_SM80
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
#else
    __trap();
#endif
}

/* 4 E2M1 codes (4 bytes of codes, the low 4 bits of each byte) to 4 int8
 * (twice the values), as the dequantization of Marlin: prmt looks up the 4
 * magnitudes in a table of 8 bytes (0 1 2 3 4 6 8 12) at once, and the sign
 * bits negate the bytes (v ^ m) - m. */
__device__ __forceinline__ uint32_t kt_e2m1x4(uint32_t q)
{
    uint32_t c = q & 0x0f0f0f0fu;
    uint32_t i = c & 0x07070707u;
    uint32_t t = i | (i >> 4);                       /* bytes 0, 2: 2 indices of 4 bits */
    uint32_t sel = (t & 0xffu) | ((t >> 8) & 0xff00u);
    uint32_t mag = __byte_perm(0x03020100u, 0x0c080604u, sel);
    uint32_t m = ((c >> 3) & 0x01010101u) * 0xffu;
    return __vsub4(mag ^ m, m);
}

__device__ __forceinline__ void cp_async4(void *smem, const void *gmem)
{
#if RC_SM80
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa), "l"(gmem));
#else
    *(uint32_t *)smem = *(const uint32_t *)gmem;     /* (cp.async: aligned) */
#endif
}

__device__ __forceinline__ void cp_async16(void *smem, const void *gmem)
{
#if RC_SM80
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
#else
    *(int4 *)smem = *(const int4 *)gmem;
#endif
}

__device__ __forceinline__ void cp_async_commit()
{
#if RC_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait1()
{
#if RC_SM80
    asm volatile("cp.async.wait_group 1;\n" ::);
#endif
}

__device__ __forceinline__ void cp_async_wait0()
{
#if RC_SM80
    asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

/* wait until at most N groups of copies are in flight */
template <int N>
__device__ __forceinline__ void cp_async_wait()
{
#if RC_SM80
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}
