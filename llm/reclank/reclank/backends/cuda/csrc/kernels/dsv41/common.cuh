/* DeepSeek V4.1: what step.cu and the kernel module prefill.cu share (the
 * forms of the caches and their readers, the keys of a query, the slots of
 * the experts and the pairs of an expert in a group). Included by step.cu and
 * by prefill.cu (and so by a bench of prefill.cu alone); needs types.cu's
 * e4m3f before it. */
#pragma once

/* The caches (the window rings, the compressed rows, the index keys): a row of
 * hd values in one of these forms. The records take the form of the caches
 * (1 bfloat16, 2 the smallest), each cache its own form of it (ds_cf):
 *   0  float32 (the rows of a group, kvq; not a cache);
 *   1  the bfloat16 bits (the three caches);
 *   2  FP8: hd E4M3 codes, then an E8M0 scale for each 32 (the window rows);
 *   4  FP4: hd / 2 bytes of E2M1 codes (value 2i in the low 4 bits), then an
 *      E4M3 scale for each 16 (the compressed rows: the form of the model);
 *   5  FP4: the same, an E8M0 scale for each 32 (the index keys).
 * Each holds the values the model puts there exactly (E4M3 times a power of
 * two; E2M1 times an E4M3 scale; E2M1 times a power of two; the sign of a
 * zero too), so the bits of the model do not change. */
#define DS_CW 0                 /* the kinds of cache, for ds_cf */
#define DS_CC 1
#define DS_CK 2
__device__ __forceinline__ int ds_cf(int form, int kind)
{
    return form <= 1 ? form : kind == DS_CW ? 2 : kind == DS_CC ? 4 : 5;
}

__device__ __forceinline__ size_t ds_rb(int hd, int form)
{
    switch (form) {
    case 0: return (size_t)hd * 4;
    case 1: return (size_t)hd * 2;
    case 2: return (size_t)hd + hd / 32;
    case 4: return (size_t)hd / 2 + hd / 16;
    default: return (size_t)hd / 2 + hd / 32;
    }
}

/* The value of an E2M1 code: twice the values 0, 0.5, 1, 1.5, 2, 3, 4, 6 are
 * the nibbles of 0xC8643210 (no table in local memory); the sign of a zero
 * is kept. */
__device__ __forceinline__ float ds_e2m1f(uint32_t c)
{
    float v = (float)((0xC8643210u >> (4 * (c & 7))) & 15u) * 0.5f;
    return __uint_as_float(__float_as_uint(v) | ((c & 8u) << 28));
}

/* the scale of value col of a row of FP8 or FP4 */
__device__ __forceinline__ float ds_bscale(const uint8_t *rw, int hd, int col, int form)
{
    if (form == 2) {
        return __int_as_float((int)rw[hd + col / 32] << 23);
    }
    if (form == 4) {
        return e4m3f(rw[hd / 2 + col / 16]);
    }
    return __int_as_float((int)rw[hd / 2 + col / 32] << 23);
}

/* value col of row of a cache */
__device__ __forceinline__ float ds_rld(const void *base, size_t row, int col, int hd, int form)
{
    if (form == 0) {
        return ((const float *)base)[row * hd + col];
    }
    if (form == 1) {
        return __uint_as_float((uint32_t)((const uint16_t *)base)[row * hd + col] << 16);
    }
    const uint8_t *rw = (const uint8_t *)base + row * ds_rb(hd, form);
    float q = form == 2 ? e4m3f(rw[col]) : ds_e2m1f(rw[col / 2] >> (4 * (col & 1)));
    return __fmul_rn(q, ds_bscale(rw, hd, col, form));
}

/* the 4 values from col (a multiple of 4) */
__device__ __forceinline__ float4 ds_rld4(const void *base, size_t row, int col, int hd, int form)
{
    if (form == 0) {
        return *(const float4 *)((const float *)base + row * hd + col);
    }
    if (form == 1) {
        uint2 u = *(const uint2 *)((const uint16_t *)base + row * hd + col);
        return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16),
                           __uint_as_float(u.y & 0xffff0000u));
    }
    const uint8_t *rw = (const uint8_t *)base + row * ds_rb(hd, form);
    float sc = ds_bscale(rw, hd, col, form);
    float q0, q1, q2, q3;
    if (form == 2) {
        uint32_t q = *(const uint32_t *)(rw + col);
        q0 = e4m3f(q & 255), q1 = e4m3f((q >> 8) & 255), q2 = e4m3f((q >> 16) & 255), q3 = e4m3f(q >> 24);
    } else {
        uint32_t q = *(const uint16_t *)(rw + col / 2);
        q0 = ds_e2m1f(q), q1 = ds_e2m1f(q >> 4), q2 = ds_e2m1f(q >> 8), q3 = ds_e2m1f(q >> 12);
    }
    return make_float4(__fmul_rn(q0, sc), __fmul_rn(q1, sc), __fmul_rn(q2, sc), __fmul_rn(q3, sc));
}

/* the operand k of a record as the form of a cache (none: 0) */
__device__ __forceinline__ int ds_form(const gp_rec *r, const int64_t *e, int k)
{
    return r->tag[k] != GP_T_NONE ? (int)di(r, e, k) : 0;
}

/* Key m of a query: a row of the ring (form) or of the compressed cache (its
 * form), or of the group (kvq, float32): its base, row and form. */
struct ds_krow {
    const void *p;
    size_t row;
    int form;
};

__device__ __forceinline__ ds_krow ds_akey(int m, int nr, int ng, int64_t lo, int g0, int win, const void *ring,
                                          const float *kvq, const void *ckv, const int *sel, int form)
{
    if (m < nr) {
        return {ring, (size_t)((lo + m) % win), ds_cf(form, DS_CW)};
    }
    if (m < nr + ng) {
        return {kvq, (size_t)(g0 + m - nr), 0};
    }
    return {ckv, (size_t)sel[m - nr - ng], ds_cf(form, DS_CC)};
}

#define DS_STAGE_SLOT0 (1 << 20)

/* The bytes of the expert in slot of map: of store (operand 4), or of the stage (operand 23). */
__device__ __forceinline__ const uint8_t *ds_slot_w(const gp_rec *r, const int64_t *e, int slot)
{
    return slot >= DS_STAGE_SLOT0 ? DP(const uint8_t, 23) + (size_t)(slot - DS_STAGE_SLOT0) * di(r, e, 5)
                                  : DP(const uint8_t, 4) + (size_t)slot * di(r, e, 5);
}

#define DS_HOT_MAXP 3072

/* The pairs of expert ex among the pairs [p0, p0 + DS_HOT_MAXP k) of idx
 * [t k] into pl (an expert at most once a token: at most DS_HOT_MAXP), in any
 * order (each pair's sums are its own); their count. Every thread calls it,
 * for the chunks of the pairs in turn (a group of any size). */
__device__ int ds_hot_pairs(const int *idx, int ex, int p0, int np, int k, int *pl, int *cnt)
{
    __syncthreads();            /* the chunk before is done with pl */
    if (threadIdx.x == 0) {
        *cnt = 0;
    }
    __syncthreads();
    int p1 = p0 + DS_HOT_MAXP * k < np ? p0 + DS_HOT_MAXP * k : np;
    for (int p = p0 + threadIdx.x; p < p1; p += blockDim.x) {
        if (idx[p] == ex) {
            int at = atomicAdd(cnt, 1);
            if (at < DS_HOT_MAXP) {
                pl[at] = p;
            }
        }
    }
    __syncthreads();
    return *cnt < DS_HOT_MAXP ? *cnt : DS_HOT_MAXP;
}
