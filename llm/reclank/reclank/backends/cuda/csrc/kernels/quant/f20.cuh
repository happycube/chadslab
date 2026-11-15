/* F20 rows (backends/cpu/csrc/kernels/kq/formats.c KQ_F20; k a multiple of 32): 16 significant bits of a
 * float32 value in 20 bits (a word of the sign and 15 bits of the mantissa, a nibble of the gap of its exponent
 * under the largest of its group of 32), and that largest exponent in a byte a group. The bytes of a row, the
 * float32 value of a word (f20_val), 8 values at kc (f20_load8, f20_dec8), and a warp's group of 32 values
 * into a row (f20_store_warp). The media encoders' weights and caches (kernels/enc/encoders.cu) and the
 * cache of the first full attention layer of Qwen3.5 / 3.6 (k_attn_prep, attn_kv8, k_flash_f32h). */

__host__ __device__ __forceinline__ size_t f20_row_bytes(int k)
{
    return ((size_t)k * 2 + (size_t)k / 2 + (size_t)k / 32 + 15) / 16 * 16;
}

__device__ __forceinline__ float f20_val(uint32_t lo, int gap, int E)
{
    uint32_t u = gap == 15 && lo == 0x8000u ? 0u
                 : ((lo & 0x8000u) << 16) | ((uint32_t)((E - gap) & 255) << 23) | ((lo & 0x7fffu) << 8);
    return __uint_as_float(u);
}

/* The words of 8 values of an F20 row at kc (a multiple of 8; f20_load8: their lo, gap bytes, the
 * nibble's shift, E) and their float32 values (f20_dec8): apart, so that a kernel's next loads
 * wait for nothing before its products. */
struct F20Words8 {
    uint4 l;
    uint2 h;
    int sh, E;
};

__device__ __forceinline__ F20Words8 f20_load8(const uint8_t *w, int k, int kc)
{
    const int g = kc >> 5, j = kc & 31;
    F20Words8 r;
    r.l = *(const uint4 *)(w + 2 * (size_t)kc);
    r.h = *(const uint2 *)(w + 2 * (size_t)k + 16 * g + (j & 15));
    r.sh = 4 * (j >> 4);
    r.E = w[2 * (size_t)k + k / 2 + g];
    return r;
}

__device__ __forceinline__ void f20_dec8(const F20Words8 &f, float *dst)
{
    const uint32_t l[4] = {f.l.x, f.l.y, f.l.z, f.l.w};
    const uint32_t h[2] = {f.h.x >> f.sh, f.h.y >> f.sh};
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
        dst[q] = f20_val((l[q >> 1] >> (16 * (q & 1))) & 0xffffu, (h[q >> 2] >> (8 * (q & 3))) & 15, f.E);
    }
}

/* Value c of a row of k values (c % 32 the lane: the 32 lanes of the warp hold the group c / 32, all of them
 * here) into the F20 row w: the bits of rc_kq_f32_to_f20 (the pad bytes of the row are not written). */
__device__ __forceinline__ void f20_store_warp(float x, uint8_t *w, int k, int c)
{
    const int lane = c & 31, g = c >> 5;
    uint32_t v = __float_as_uint(x);
    if (((v >> 23) & 255) == 255) {
        v = 0;
    } else {
        v = (v + 0x7fu + ((v >> 8) & 1u)) & 0xffffff00u;
        if (((v >> 23) & 255) == 255) {
            v = 0;
        }
    }
    const int e = (v >> 23) & 255;
    int E = e;
    for (int sh = 16; sh; sh >>= 1) {
        E = max(E, __shfl_xor_sync(0xffffffffu, E, sh));
    }
    int sgn = v >> 31, m = (v >> 8) & 0x7fff, gap = E - e;
    if (gap > 15) {
        gap = 15;
        sgn = 1;
        m = 0;
    }
    ((uint16_t *)w)[32 * g + lane] = (uint16_t)(sgn << 15 | m);
    const int up = __shfl_down_sync(0xffffffffu, gap, 16);
    if (lane < 16) {
        w[2 * (size_t)k + 16 * g + lane] = (uint8_t)(gap | up << 4);
    }
    if (lane == 0) {
        w[2 * (size_t)k + k / 2 + g] = (uint8_t)E;
    }
}
