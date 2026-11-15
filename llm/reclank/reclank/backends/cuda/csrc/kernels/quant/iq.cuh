/* ---------- the other block formats of Unsloth's UD files: Q2_K, Q3_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
 * IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS ----------
 * Ported from ggml (ggml/src/ggml-common.h, ggml/src/ggml-quants.c; MIT License; Copyright (c)
 * 2023-2026 The ggml authors). See THIRD_PARTY_NOTICES.md. The layouts: reclank/formats/gguf.py
 * (_BLOCK_DT, dequant), backends/include/iq_tables.h and iq2_tables.h.
 *
 * Every format gives, for each part of 32 values, int8 values q and a scale for each 16 (the same for
 * both halves but in Q2_K, Q3_K, IQ2_XS, IQ2_S and IQ1_M): w = d q, and for Q2_K a min for each 16
 * (w = d q - m: the products take m times the sum of x). iq_words makes two words of 4 values (values
 * 4c .. 4c + 3 and 16 + 4c .. 16 + 4c + 3) of part j of a block; iq_scales the scales of the two halves;
 * iq_mins the mins. The products of a step (kq_rows_iq: dp4a on the int8 x), of a few tokens on float
 * x (kq_iq_part), the float tiles (kq_iq_dequant8), the tensor-core tiles (kt_tile) and the dp4a tiles
 * of Pascal (kt_tile_dp4a) all read the values with these, so they get the same values. IQ1_S and
 * IQ1_M: the values 8 g +- 1 (g of the grid in -1, 0, 1; the delta +-1/8) and the scale / 8.
 *
 * blk is a block of 256 values (part j < 8), or for IQ4_NL the first block of 32 (part j is the block
 * j after it). The blocks of 98, 110, 66, 74, 82, 50 and 18 bytes are only 2-aligned: halfword
 * loads. */
#include "iq_tables.h"

#define KQ_Q2_K 10
#define KQ_Q3_K 11
#define KQ_IQ2_XS 17
#define KQ_IQ3_XXS 18
#define KQ_IQ1_S 19
#define KQ_IQ4_NL 20
#define KQ_IQ3_S 21
#define KQ_IQ2_S 22
#define KQ_IQ4_XS 23
#define KQ_IQ1_M 29

__device__ const uint32_t kq_iq3xxs_grid[256] = IQ3XXS_GRID;
__device__ const uint32_t kq_iq3s_grid[512] = IQ3S_GRID;
__device__ const uint64_t kq_iq2xs_grid[512] = IQ2XS_GRID;
__device__ const uint64_t kq_iq2s_grid[1024] = IQ2S_GRID;
__device__ const uint64_t kq_iq1s_grid[2048] = IQ1S_GRID;

/* (constexpr: the kernels of one type, k_kq_ft) */
__host__ __device__ constexpr bool kq_iq_c(int type)
{
    return type == KQ_Q2_K || type == KQ_Q3_K || type == KQ_IQ1_S || type == KQ_IQ1_M || type == KQ_IQ2_XXS ||
           type == KQ_IQ2_XS || type == KQ_IQ2_S || type == KQ_IQ3_XXS || type == KQ_IQ3_S || type == KQ_IQ4_NL ||
           type == KQ_IQ4_XS;
}

__host__ __device__ __forceinline__ int kq_iq_type(int type)
{
    return type == KQ_Q2_K || type == KQ_Q3_K || type == KQ_IQ1_S || type == KQ_IQ1_M || type == KQ_IQ2_XXS ||
           type == KQ_IQ2_XS || type == KQ_IQ2_S || type == KQ_IQ3_XXS || type == KQ_IQ3_S || type == KQ_IQ4_NL ||
           type == KQ_IQ4_XS;
}

/* the bytes of a block of 256 values (IQ4_NL: of 32) */
__host__ __device__ constexpr int iq_bb(int type)
{
    return type == KQ_IQ4_NL ? 18 : type == KQ_IQ4_XS ? 136 : type == KQ_IQ3_XXS ? 98 : type == KQ_Q2_K ? 84 :
           type == KQ_IQ2_XXS ? 66 : type == KQ_IQ2_XS ? 74 : type == KQ_IQ2_S ? 82 : type == KQ_IQ1_S ? 50 :
           type == KQ_IQ1_M ? 56 : 110;
}

__host__ __device__ __forceinline__ size_t kq_iq_row_bytes(int type, int cols)
{
    return (size_t)cols / (type == KQ_IQ4_NL ? 32 : 256) * iq_bb(type);
}

/* a scale of its own for each 16 values (the tiles: the sums of the two halves) */
__host__ __device__ constexpr bool iq_split(int type)
{
    return type == KQ_Q2_K || type == KQ_Q3_K || type == KQ_IQ2_XS || type == KQ_IQ2_S || type == KQ_IQ1_M;
}

__device__ __forceinline__ float iq_half(const uint8_t *p)
{
    return __half2float(__ushort_as_half(*(const uint16_t *)p));
}

__device__ __forceinline__ uint32_t iq_u32(const uint8_t *p)
{
    const uint16_t *h = (const uint16_t *)p;
    return (uint32_t)h[0] | ((uint32_t)h[1] << 16);
}

/* The int8 values (ggml kvalues_iq4nl) of the 4 codes in the low 4 bits of the bytes of q: byte n of
 * the table by two byte_perm (the low 3 bits of the code select in each half of the table), its
 * bit 3 the half. */
__device__ __forceinline__ uint32_t iq4_lut(uint32_t q)
{
    uint32_t s = (q & 0x7u) | ((q >> 4) & 0x70u) | ((q >> 8) & 0x700u) | ((q >> 12) & 0x7000u);
    uint32_t lo = __byte_perm(0xbfad9881u, 0xf6eaddcfu, s), hi = __byte_perm(0x26190d01u, 0x71594535u, s);
    uint32_t m = ((q >> 3) & 0x01010101u) * 0xffu;      /* (not __vcmpne4: several instructions on sm_86) */
    return (lo & ~m) | (hi & m);
}

/* the 4 values of a grid word g (bytes 1 to 62) with the signs of 4 bits (bit k: value k negative): a
 * negative byte is ~g + 1, so (g ^ 0xff e) + e with e 1 in those bytes; ~g is at most 0xfe (no grid byte
 * is 0), so no carry crosses a byte. (__vsub4 and __vcmpne4 are several instructions each on sm_86.) */
__device__ __forceinline__ uint32_t iq_signed(uint32_t g, uint32_t s4)
{
    uint32_t e = (s4 * 0x00204081u) & 0x01010101u;     /* bit k to byte k */
    return (g ^ (e * 0xffu)) + e;
}

/* the bytes of v (each 0 to 0x7f above -0x80 .. 0x7f as int8 with the bias 0x80) minus c in each byte, with no
 * borrow across bytes: biased by 0x80, the subtraction, the bias off (Q3_K's -4, IQ1's -1) */
__device__ __forceinline__ uint32_t iq_sub_bytes(uint32_t v, uint32_t c)
{
    return ((v ^ 0x80808080u) - c) ^ 0x80808080u;
}

/* the 8 signs of a group of IQ2_XXS, IQ2_XS or IQ3_XXS: the 7 bits and their parity (IQ2XXS_SIGN) */
__device__ __forceinline__ uint32_t iq_sign8(uint32_t s7)
{
    return s7 | ((uint32_t)(__popc(s7) & 1) << 7);
}

/* the half h of 8 grid bytes with the signs s8 (bit k: byte k negative) */
__device__ __forceinline__ uint32_t iq_signed8(uint64_t g, uint32_t s8, int h)
{
    return iq_signed((uint32_t)(g >> (32 * h)), (s8 >> (4 * h)) & 15);
}

/* the half h of 8 values of IQ1S_GRID (-1, 0, 1 as int8) as 8 g + 1, or 8 g - 1 (neg) */
__device__ __forceinline__ uint32_t iq1_word(uint64_t g, int h, bool neg)
{
    uint32_t w = ((uint32_t)(g >> (32 * h)) << 3) & 0xf8f8f8f8u;
    return neg ? iq_sub_bytes(w, 0x01010101u) : (w | 0x01010101u);
}

/* the 2-bit values of part j from the 64 bytes qs (Q2_K; Q3_K's low bits): byte 32 (j / 4) + i at 2 (j % 4) */
__device__ __forceinline__ void iq_2bit(const uint8_t *qs, int j, int c, uint32_t *b)
{
    const uint8_t *q = qs + 32 * (j / 4);
    int sh = 2 * (j % 4);
    b[0] = (iq_u32(q + 4 * c) >> sh) & 0x03030303u;
    b[1] = (iq_u32(q + 16 + 4 * c) >> sh) & 0x03030303u;
}

template <int T>
__device__ __forceinline__ void iq_words(const uint8_t *blk, int j, int c, uint32_t *b)
{
    const int h = c % 2;
    if (T == KQ_IQ4_NL || T == KQ_IQ4_XS) {
        uint32_t u = iq_u32((T == KQ_IQ4_NL ? blk + 18 * j + 2 : blk + 8 + 16 * j) + 4 * c);
        b[0] = iq4_lut(u & 0x0f0f0f0fu);
        b[1] = iq4_lut((u >> 4) & 0x0f0f0f0fu);
    } else if (T == KQ_Q2_K) {
        iq_2bit(blk + 16, j, c, b);
    } else if (T == KQ_Q3_K) {
        /* value i of part j: 2 bits as Q2_K's, and bit j of hm[i] (its 0 is -4) */
        uint32_t l[2];
        iq_2bit(blk + 32, j, c, l);
        uint32_t h0 = (iq_u32(blk + 4 * c) >> j) & 0x01010101u, h1 = (iq_u32(blk + 16 + 4 * c) >> j) & 0x01010101u;
        b[0] = iq_sub_bytes(l[0] | (h0 << 2), 0x04040404u);
        b[1] = iq_sub_bytes(l[1] | (h1 << 2), 0x04040404u);
    } else if (T == KQ_IQ3_XXS) {
        /* indices 8 j + c and 8 j + 4 + c; the signs of values 8 l .. 8 l + 7 at 7 l of the word of j */
        const uint8_t *qs = blk + 2 + 8 * j;
        uint32_t ss = iq_u32(blk + 66 + 4 * j);
        uint32_t s0 = iq_sign8((ss >> (7 * (c / 2))) & 127), s1 = iq_sign8((ss >> (7 * (2 + c / 2))) & 127);
        b[0] = iq_signed(kq_iq3xxs_grid[qs[c]], (s0 >> (4 * h)) & 15);
        b[1] = iq_signed(kq_iq3xxs_grid[qs[4 + c]], (s1 >> (4 * h)) & 15);
    } else if (T == KQ_IQ3_S) {
        /* 9-bit indices (bit 8 in qh[j]); the signs of value 8 m + k in bit k of signs[m] */
        const uint8_t *qs = blk + 2 + 8 * j;
        uint32_t qh = blk[66 + j];
        const uint8_t *sg = blk + 74 + 4 * j;
        uint32_t i0 = qs[c] | (((qh >> c) & 1) << 8), i1 = qs[4 + c] | (((qh >> (4 + c)) & 1) << 8);
        b[0] = iq_signed(kq_iq3s_grid[i0], (sg[c / 2] >> (4 * h)) & 15);
        b[1] = iq_signed(kq_iq3s_grid[i1], (sg[2 + c / 2] >> (4 * h)) & 15);
    } else if (T == KQ_IQ2_XXS) {
        /* the group of j (8 bytes at 2 + 8 j): 4 grid indices (8 values each), then a word of the signs
         * of group l at 7 l and the scale in bits 28-31 */
        const uint8_t *p = blk + 2 + 8 * j;
        uint32_t aux = iq_u32(p + 4);
        int l0 = c / 2, l1 = 2 + c / 2;
        b[0] = iq_signed8(nm_iq2_grid[p[l0]], iq_sign8((aux >> (7 * l0)) & 127), h);
        b[1] = iq_signed8(nm_iq2_grid[p[l1]], iq_sign8((aux >> (7 * l1)) & 127), h);
    } else if (T == KQ_IQ2_XS) {
        /* entry e (8 values): the uint16 at 2 + 2 e, 9 bits of index and 7 of signs */
        const uint16_t *qs = (const uint16_t *)(blk + 2);
        uint32_t q0 = qs[4 * j + c / 2], q1 = qs[4 * j + 2 + c / 2];
        b[0] = iq_signed8(kq_iq2xs_grid[q0 & 511], iq_sign8(q0 >> 9), h);
        b[1] = iq_signed8(kq_iq2xs_grid[q1 & 511], iq_sign8(q1 >> 9), h);
    } else if (T == KQ_IQ2_S) {
        /* entry e: index qs[e] and 2 bits of qh[e / 4] at 2 (e % 4), the 8 signs signs[e] */
        int e0 = 4 * j + c / 2, e1 = 4 * j + 2 + c / 2;
        uint32_t qh = blk[66 + j];
        uint32_t i0 = blk[2 + e0] | (((qh >> (2 * (c / 2))) & 3) << 8);
        uint32_t i1 = blk[2 + e1] | (((qh >> (2 * (2 + c / 2))) & 3) << 8);
        b[0] = iq_signed8(kq_iq2s_grid[i0], blk[34 + e0], h);
        b[1] = iq_signed8(kq_iq2s_grid[i1], blk[34 + e1], h);
    } else if (T == KQ_IQ1_S) {
        /* entry 4 j + m: index qs[e] and 3 bits of the word qh[j] at 3 m; its bit 15 the sign of the delta */
        uint32_t qh = *(const uint16_t *)(blk + 34 + 2 * j);
        int m0 = c / 2, m1 = 2 + c / 2;
        uint32_t i0 = blk[2 + 4 * j + m0] | (((qh >> (3 * m0)) & 7) << 8);
        uint32_t i1 = blk[2 + 4 * j + m1] | (((qh >> (3 * m1)) & 7) << 8);
        bool neg = (qh >> 15) & 1;
        b[0] = iq1_word(kq_iq1s_grid[i0], h, neg);
        b[1] = iq1_word(kq_iq1s_grid[i1], h, neg);
    } else {
        /* IQ1_M: entry e: index qs[e] and the low 3 bits of the nibble e % 2 of qh[e / 2], its bit 3 the sign
         * of the delta */
        int e0 = 4 * j + c / 2, e1 = 4 * j + 2 + c / 2;
        uint32_t n0 = (blk[32 + e0 / 2] >> (4 * (e0 % 2))) & 15, n1 = (blk[32 + e1 / 2] >> (4 * (e1 % 2))) & 15;
        b[0] = iq1_word(kq_iq1s_grid[blk[e0] | ((n0 & 7) << 8)], h, n0 & 8);
        b[1] = iq1_word(kq_iq1s_grid[blk[e1] | ((n1 & 7) << 8)], h, n1 & 8);
    }
}

/* the 6-bit scale si (of 16) of a Q3_K block, minus 32 */
__device__ __forceinline__ int iq_q3k_scale(const uint8_t *sc, int si)
{
    return (int)(((sc[si % 8] >> (4 * (si / 8))) & 0xF) | (((sc[8 + si % 4] >> (2 * (si / 4))) & 3) << 4)) - 32;
}

/* the float16 scale of an IQ1_M block: the top 4 bits of its 4 scale words */
__device__ __forceinline__ float iq1m_d(const uint8_t *blk)
{
    const uint16_t *sc = (const uint16_t *)(blk + 48);
    uint32_t d = (sc[0] >> 12) | ((sc[1] >> 8) & 0xf0u) | ((sc[2] >> 4) & 0xf00u) | (sc[3] & 0xf000u);
    return __half2float(__ushort_as_half((uint16_t)d));
}

template <int T>
__device__ __forceinline__ void iq_scales(const uint8_t *blk, int j, float *d0, float *d1)
{
    if (T == KQ_IQ4_NL) {
        *d0 = *d1 = iq_half(blk + 18 * j);
    } else if (T == KQ_IQ4_XS) {
        uint32_t sh = *(const uint16_t *)(blk + 2);
        int ls = (int)(((blk[4 + j / 2] >> (4 * (j % 2))) & 0xF) | (((sh >> (2 * j)) & 3) << 4));
        *d0 = *d1 = iq_half(blk) * (float)(ls - 32);
    } else if (T == KQ_Q2_K) {
        float d = iq_half(blk + 80);
        *d0 = d * (float)(blk[2 * j] & 15);
        *d1 = d * (float)(blk[2 * j + 1] & 15);
    } else if (T == KQ_Q3_K) {
        float d = iq_half(blk + 108);
        *d0 = d * (float)iq_q3k_scale(blk + 96, 2 * j);
        *d1 = d * (float)iq_q3k_scale(blk + 96, 2 * j + 1);
    } else if (T == KQ_IQ3_XXS) {
        *d0 = *d1 = iq_half(blk) * (0.5f + (float)(iq_u32(blk + 66 + 4 * j) >> 28)) * 0.5f;
    } else if (T == KQ_IQ3_S) {
        *d0 = *d1 = iq_half(blk) * (float)(1 + 2 * ((blk[106 + j / 2] >> (4 * (j % 2))) & 0xF));
    } else if (T == KQ_IQ2_XXS) {
        *d0 = *d1 = iq_half(blk) * (float)(2 * (iq_u32(blk + 6 + 8 * j) >> 28) + 1) * 0.125f;
    } else if (T == KQ_IQ2_XS || T == KQ_IQ2_S) {
        /* the scale of each 16: the low half of byte j for the first, the high half for the second */
        uint32_t s = blk[(T == KQ_IQ2_XS ? 66 : 74) + j];
        float d = iq_half(blk) * 0.25f;
        *d0 = d * (0.5f + (float)(s & 15));
        *d1 = d * (0.5f + (float)(s >> 4));
    } else if (T == KQ_IQ1_S) {
        uint32_t qh = *(const uint16_t *)(blk + 34 + 2 * j);
        *d0 = *d1 = iq_half(blk) * (float)(2 * ((qh >> 12) & 7) + 1) * 0.125f;
    } else {
        /* IQ1_M: the 3 bits of scale k = 2 j + h in the word k / 4 at 3 (k % 4) */
        const uint16_t *sc = (const uint16_t *)(blk + 48);
        float d = iq1m_d(blk) * 0.125f;
        int k0 = 2 * j, k1 = 2 * j + 1;
        *d0 = d * (float)(2 * ((sc[k0 / 4] >> (3 * (k0 % 4))) & 7) + 1);
        *d1 = d * (float)(2 * ((sc[k1 / 4] >> (3 * (k1 % 4))) & 7) + 1);
    }
}

/* the mins of the two halves of part j (Q2_K; 0 for the others) */
template <int T>
__device__ __forceinline__ void iq_mins(const uint8_t *blk, int j, float *m0, float *m1)
{
    if (T == KQ_Q2_K) {
        float dm = iq_half(blk + 82);
        *m0 = dm * (float)(blk[2 * j] >> 4);
        *m1 = dm * (float)(blk[2 * j + 1] >> 4);
    } else {
        *m0 = *m1 = 0.f;
    }
}

/* the block and the part of part s (of 32 values) of a row */
template <int T>
__device__ __forceinline__ const uint8_t *iq_part(const uint8_t *w, int s, int *j)
{
    if (T == KQ_IQ4_NL) {
        *j = s;
        return w;
    }
    *j = s % 8;
    return w + (size_t)(s / 8) * iq_bb(T);
}

/* The products of a row with the int8 x of NT tokens (kq_rows_i8): lane l takes the parts l, l + 32
 * (l + 16 for HALF) ..., 8 dp4a for each token (Q2_K: 8 more for the sums of x of its mins). sum gets
 * the lane's terms. */
template <int T, int NT, bool HALF>
__device__ __forceinline__ void kq_rows_iq_t(const uint8_t *w, const int8_t *xq, const float *xs, int cols,
                                             int t, float *sum)
{
    int lane = threadIdx.x % (HALF ? 16 : 32), np = cols / 32;
    for (int s = lane; s < np; s += HALF ? 16 : 32) {
        int j;
        const uint8_t *blk = iq_part<T>(w, s, &j);
        uint32_t q[8];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            uint32_t b[2];
            iq_words<T>(blk, j, c, b);
            q[c] = b[0];
            q[4 + c] = b[1];
        }
        float d0, d1, m0, m1;
        iq_scales<T>(blk, j, &d0, &d1);
        iq_mins<T>(blk, j, &m0, &m1);
        #pragma unroll
        for (int jt = 0; jt < NT; ++jt) {
            if (jt < t) {
                const int8_t *xj = xq + (size_t)jt * cols + 32 * s;
                uint4 xl = *(const uint4 *)xj, xh = *(const uint4 *)(xj + 16);
                int dl = rc_dp4a((int)q[0], (int)xl.x, 0);
                dl = rc_dp4a((int)q[1], (int)xl.y, dl);
                dl = rc_dp4a((int)q[2], (int)xl.z, dl);
                dl = rc_dp4a((int)q[3], (int)xl.w, dl);
                int dh = rc_dp4a((int)q[4], (int)xh.x, 0);
                dh = rc_dp4a((int)q[5], (int)xh.y, dh);
                dh = rc_dp4a((int)q[6], (int)xh.z, dh);
                dh = rc_dp4a((int)q[7], (int)xh.w, dh);
                float v = d0 * (float)dl + d1 * (float)dh;
                if (T == KQ_Q2_K) {
                    int nl = rc_dp4a(0x01010101, (int)xl.x, 0);
                    nl = rc_dp4a(0x01010101, (int)xl.y, nl);
                    nl = rc_dp4a(0x01010101, (int)xl.z, nl);
                    nl = rc_dp4a(0x01010101, (int)xl.w, nl);
                    int nh = rc_dp4a(0x01010101, (int)xh.x, 0);
                    nh = rc_dp4a(0x01010101, (int)xh.y, nh);
                    nh = rc_dp4a(0x01010101, (int)xh.z, nh);
                    nh = rc_dp4a(0x01010101, (int)xh.w, nh);
                    v -= m0 * (float)nl + m1 * (float)nh;
                }
                sum[jt] += xs[jt * np + s] * v;
            }
        }
    }
}

/* kq_rows_iq_t of the type: a call, so the generic kernels of the other types (kq_rows_i8) keep their
 * registers (as kq_iq2r_part). */
#define IQ_DISPATCH(type, F, ...)                                                                    \
    switch (type) {                                                                                  \
    case KQ_Q2_K: F<KQ_Q2_K>(__VA_ARGS__); break;                                                    \
    case KQ_Q3_K: F<KQ_Q3_K>(__VA_ARGS__); break;                                                    \
    case KQ_IQ1_S: F<KQ_IQ1_S>(__VA_ARGS__); break;                                                  \
    case KQ_IQ1_M: F<KQ_IQ1_M>(__VA_ARGS__); break;                                                  \
    case KQ_IQ2_XXS: F<KQ_IQ2_XXS>(__VA_ARGS__); break;                                              \
    case KQ_IQ2_XS: F<KQ_IQ2_XS>(__VA_ARGS__); break;                                                \
    case KQ_IQ2_S: F<KQ_IQ2_S>(__VA_ARGS__); break;                                                  \
    case KQ_IQ3_XXS: F<KQ_IQ3_XXS>(__VA_ARGS__); break;                                              \
    case KQ_IQ3_S: F<KQ_IQ3_S>(__VA_ARGS__); break;                                                  \
    case KQ_IQ4_NL: F<KQ_IQ4_NL>(__VA_ARGS__); break;                                                \
    default: F<KQ_IQ4_XS>(__VA_ARGS__); break;                                                       \
    }

template <int NT, bool HALF>
struct kq_rows_iq_f {
    template <int T>
    static __device__ __forceinline__ void run(const uint8_t *w, const int8_t *xq, const float *xs, int cols,
                                               int t, float *sum)
    {
        kq_rows_iq_t<T, NT, HALF>(w, xq, xs, cols, t, sum);
    }
};

template <int NT, bool HALF>
static __device__ __noinline__ void kq_rows_iq(int type, const uint8_t *w, const int8_t *xq, const float *xs,
                                               int cols, int t, float *sum)
{
    using R = kq_rows_iq_f<NT, HALF>;
    IQ_DISPATCH(type, R::template run, w, xq, xs, cols, t, sum)
}

/* The sum of 4 int8 values (a) times 4 floats. */
__device__ __forceinline__ float iq_dot4f(uint32_t a, float4 x)
{
    return (float)(int8_t)(a & 255) * x.x + (float)(int8_t)((a >> 8) & 255) * x.y +
           (float)(int8_t)((a >> 16) & 255) * x.z + (float)(int8_t)(a >> 24) * x.w;
}

template <int T>
__device__ __forceinline__ void kq_iq_part_t(const uint8_t *w, const float *x, int cols, int part, int nparts,
                                             int lane, float *out)
{
    float sum = 0.f;
    for (int s = lane + 32 * part; s < cols / 32; s += 32 * nparts) {
        int j;
        const uint8_t *blk = iq_part<T>(w, s, &j);
        float d0, d1, m0, m1, a0 = 0.f, a1 = 0.f, x0 = 0.f, x1 = 0.f;
        iq_scales<T>(blk, j, &d0, &d1);
        iq_mins<T>(blk, j, &m0, &m1);
        const float4 *x4 = (const float4 *)(x + 32 * s);
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            uint32_t b[2];
            iq_words<T>(blk, j, c, b);
            a0 += iq_dot4f(b[0], x4[c]);
            a1 += iq_dot4f(b[1], x4[4 + c]);
            if (T == KQ_Q2_K) {
                x0 += x4[c].x + x4[c].y + x4[c].z + x4[c].w;
                x1 += x4[4 + c].x + x4[4 + c].y + x4[4 + c].z + x4[4 + c].w;
            }
        }
        sum += d0 * a0 + d1 * a1 - (m0 * x0 + m1 * x1);
    }
    *out = sum;
}

/* kq_row_part of these types on float32 x (Maxwell; RECLANK_GPU_I8X=0): lane l takes the parts
 * l + 32 part, ... (all the lanes get their own terms; the caller adds them). */
static __device__ __noinline__ float kq_iq_part(int type, const uint8_t *w, const float *x, int cols, int part,
                                                int nparts, int lane)
{
    float v;
    IQ_DISPATCH(type, kq_iq_part_t, w, x, cols, part, nparts, lane, &v)
    return v;
}

template <int T>
__device__ __forceinline__ void kq_iq_dequant8_t(const uint8_t *w, int c0, float *v)
{
    int j, o = c0 % 32;
    const uint8_t *blk = iq_part<T>(w, c0 / 32, &j);
    uint32_t b0[2], b1[2];
    iq_words<T>(blk, j, (o % 16) / 4, b0);
    iq_words<T>(blk, j, (o % 16) / 4 + 1, b1);
    float d0, d1, m0, m1;
    iq_scales<T>(blk, j, &d0, &d1);
    iq_mins<T>(blk, j, &m0, &m1);
    uint32_t a = o < 16 ? b0[0] : b0[1], b = o < 16 ? b1[0] : b1[1];
    float d = o < 16 ? d0 : d1, m = o < 16 ? m0 : m1;
    for (int u = 0; u < 4; ++u) {
        v[u] = d * (float)(int8_t)((a >> (8 * u)) & 255) - m;
        v[4 + u] = d * (float)(int8_t)((b >> (8 * u)) & 255) - m;
    }
}

/* 8 values of a row from column c0 (a multiple of 8) as float32 (kq_dequant8: the float tiles) */
static __device__ __noinline__ void kq_iq_dequant8(int type, const uint8_t *w, int c0, float *v)
{
    IQ_DISPATCH(type, kq_iq_dequant8_t, w, c0, v)
}
