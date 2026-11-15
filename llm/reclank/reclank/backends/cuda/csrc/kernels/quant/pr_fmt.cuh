/* ---------- ggml's i-quants and Q3_K on the GPU (Qwen3.8-27B's ShapeLearn GGUF files) ----------
 * The block formats ported from ggml (ggml/src/ggml-common.h, ggml/src/ggml-quants.c; MIT License;
 * Copyright (c) 2023-2026 The ggml authors). See THIRD_PARTY_NOTICES.md.
 *
 * byteshape's files of Qwen3.8-27B choose a type for each tensor: IQ4_XS, IQ3_S, IQ3_XXS, Q4_K, Q6_K,
 * Q3_K, Q5_K, IQ2_XXS, IQ2_XS, Q8_0, Q2_K. Each of the new ones is an int8 vector of each 32 values
 * (each 4 of them one word, the operand of dp4a) times a float scale of each 16 or 32. A row goes to
 * the GPU in the form below (reclank/formats/pr_rows.py to_rows; QwenGPU.dense): the same bits as the
 * file, in planes, so each part of 32 values is aligned loads.
 *
 *   KQ_PR_IQ4X (107)  the GGUF blocks as they are (136 bytes: d, the high 2 bits of the 8 scales,
 *                   their low 4 bits, 128 bytes of 4-bit codes; the codes of part j at 8 + 16 j, 8-aligned)
 *   KQ_PR_IQ3S (101)  IQ3_S: the low 8 bits of the grid indices (cols / 4 bytes: 8 a part), the signs
 *                   (cols / 8: a bit a value), the 9th bits (cols / 32: a byte a part), the 4-bit
 *                   scales (cols / 64), the fp16 d of each 256 (cols / 128); zeros to 16 bytes
 *   KQ_PR_IQ3X (102)  IQ3_XXS: the 8-bit grid indices (cols / 4), the uint32 of each part (cols / 8:
 *                   four 7-bit sign indices, the scale in bits 28-31), d (cols / 128); zeros to 16
 *   KQ_PR_Q3K (103)   Q3_K: the 2-bit codes (cols / 4: word u of part p, bits 8 b + 2 s: value 16 u + 4 s + b),
 *                   the high bits inverted (cols / 8: bit 8 b + s of part p's word: value 4 s + b, 1
 *                   where ggml's high bit is 0), the scales less 32 as int8 (cols / 16, one a 16), d
 *                   (cols / 128); zeros to 16. value = d sc (code - 4) = d sc (q2 | (hinv ? 0xfc : 0)).
 *   KQ_PR_IQ2S (104)  IQ2_XS: the uint16 codes (cols / 4: a 9-bit grid index and a 7-bit sign index, 8
 *                   values each), the 4-bit scales of each 16 (cols / 32: a byte a part), d (cols / 128)
 *   KQ_PR_Q6K (106)   Q6_K (the 27B's head: k_q6k_head took its 20 blocks a row at 106 GB/s): the low 4 bits
 *                   (cols / 2: 16 bytes a part, word k the values 4k .. 4k+3 in its low 4 bits, 16+4k .. in
 *                   the high), the high 2 bits with the top one inverted (cols / 4: word u of part p, bits
 *                   8 b + 2 s: value 16 u + 4 s + b), the int8 scales (cols / 16), d (cols / 128); zeros to
 *                   16. e = low | high << 4 is the code less 32 in 6 bits; the int8 value e | (e & 0x20) * 6.
 *   KQ_PR_IQ2X (105)  IQ2_XXS in the rows of iq2.cuh's KQ_IQ2R (Nemotron Bolt's form; a type of its own so
 *                   Bolt's IQ2R paths stay as they are)
 *
 * pr_part<F> decodes part p of a row (its 32 values) to 8 words (word k: values 4k .. 4k+3 as int8)
 * and the float scales of values 0-15 (dl) and 16-31 (dh). The kernels (pr.cu) take the int8 x of
 * k_kq_quant_x (a scale for each 32).
 *
 * These are the Pascal forms (a GPU without the tensor cores: QwenGPU.dense makes them there): master's
 * kernels of the ggml blocks (iq.cuh: k_kq_ft for a step, the dp4a tiles for a prompt) read 2-aligned
 * blocks with halfword loads; these rows take 16-byte loads (a P6000 at 150 W: a step 76 -> 55 ms, a
 * verify group of 8 188 -> 101 ms, prompts 60 -> 200 tok/s). The ids are their own (101-107) so the
 * kernels of the ggml ids stay as they are on the other GPUs. */
#include "iq_tables.h"

#define KQ_PR_IQ4X 107
#define KQ_PR_IQ3S 101
#define KQ_PR_IQ3X 102
#define KQ_PR_Q3K 103
#define KQ_PR_IQ2S 104
#define KQ_PR_IQ2X 105
#define KQ_PR_Q6K 106

__host__ __device__ __forceinline__ int kq_pr_type(int type)
{
    return type == KQ_PR_IQ4X || type == KQ_PR_IQ3S || type == KQ_PR_IQ3X || type == KQ_PR_Q3K || type == KQ_PR_IQ2S ||
           type == KQ_PR_IQ2X || type == KQ_PR_Q6K;
}

__host__ __device__ __forceinline__ size_t pr_r16(size_t n)
{
    return (n + 15) & ~(size_t)15;
}

/* the bytes of a row of cols values (a multiple of 256) of a type of kq_pr_type */
__host__ __device__ __forceinline__ size_t pr_row_bytes(int type, int cols)
{
    const size_t c = (size_t)cols;
    return type == KQ_PR_IQ4X ? c / 256 * 136 :
           type == KQ_PR_IQ3S ? pr_r16(c / 4 + c / 8 + c / 32 + c / 64 + c / 128) :
           type == KQ_PR_IQ3X ? pr_r16(c / 4 + c / 8 + c / 128) :
           type == KQ_PR_Q3K ? pr_r16(c / 4 + c / 8 + c / 16 + c / 128) :
           type == KQ_PR_IQ2X ? nm_q2_rb(cols) : type == KQ_PR_Q6K ? pr_r16(c / 2 + c / 4 + c / 16 + c / 128) :
           pr_r16(c / 4 + c / 32 + c / 128);
}

__device__ const uint32_t pr_iq3xxs_grid[256] = IQ3XXS_GRID;
__device__ const uint32_t pr_iq3s_grid[512] = IQ3S_GRID;
__device__ const uint64_t pr_iq2xs_grid[512] = IQ2XS_GRID;
/* (the IQ2_XXS grid and the 64-bit sign masks of a 7-bit sign index: nm_iq2_grid, nm_iq2_signs64 of iq2.cuh) */

/* the 16 values of IQ4_NL / IQ4_XS's codes (ggml kvalues_iq4nl), 4 to a word */
#define PR_IQ4_T0 0xbfad9881u      /* -127 -104  -83  -65 */
#define PR_IQ4_T1 0xf6eaddcfu      /*  -49  -35  -22  -10 */
#define PR_IQ4_T2 0x26190d01u      /*    1   13   25   38 */
#define PR_IQ4_T3 0x71594535u      /*   53   69   89  113 */

/* The int8 values of the 8 codes of q (4 bits each): .x of the low 4 bits of its bytes, .y of the high
 * (ggml's get_int_from_table_16: __byte_perm takes 3 bits of each selector, a second picks the half) */
__device__ __forceinline__ int2 pr_lut16(uint32_t q)
{
    const uint32_t sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const uint32_t r0 = __byte_perm(__byte_perm(PR_IQ4_T0, PR_IQ4_T1, q), __byte_perm(PR_IQ4_T2, PR_IQ4_T3, q), sel);
    const uint32_t r1 = __byte_perm(__byte_perm(PR_IQ4_T0, PR_IQ4_T1, q >> 16), __byte_perm(PR_IQ4_T2, PR_IQ4_T3, q >> 16),
                                    sel >> 16);
    return make_int2((int)__byte_perm(r0, r1, 0x6420), (int)__byte_perm(r0, r1, 0x7531));
}

/* The 4 magnitudes of g (bytes < 128) with the signs of the 4 bits of s4 (bit k: byte k negative).
 * PR_SIGN_MUL 1 (the default): bit k to byte k by a multiply; 0: the byte replicated, a mask of each bit,
 * __vcmpne4 and __vsub4 (both emulated on Pascal: IQ3_S 210 GB/s against 232). */
#ifndef PR_SIGN_MUL
#define PR_SIGN_MUL 1
#endif
__device__ __forceinline__ uint32_t pr_sign4(uint32_t g, uint32_t s4)
{
    if (PR_SIGN_MUL) {
        const uint32_t t = (s4 * 0x00204081u) & 0x01010101u;      /* bit k to bit 0 of byte k */
        const uint32_t m = t * 0xffu;
        return (g ^ m) + t;
    }
    const uint32_t m = __vcmpne4(__byte_perm(s4, 0, 0) & 0x08040201u, 0);
    return __vsub4(g ^ m, m);
}

/* g with the signs s4 by the 16 masks at mt (the tables of mode 0). PR_SIGN_TAB 0 (the default): pr_sign4 (its two
 * 32-bit multiplies are 3 XMAD each on Pascal); 1: the masks of the table, one load and 3 logic ops (the bench,
 * IQ3_S one token: 179 GB/s and 78 registers against 232 and 55: more registers for the loads in flight) */
#ifndef PR_SIGN_TAB
#define PR_SIGN_TAB 0
#endif
__device__ __forceinline__ uint32_t pr_sign4t(uint32_t g, uint32_t s4, const uint32_t *mt)
{
    if (!PR_SIGN_TAB) {
        return pr_sign4(g, s4);
    }
    const uint32_t m = mt[s4];
    return (g ^ m) + (m & 0x01010101u);
}

/* g (8 bytes) with the 64-bit sign mask m (0xff a negative byte): the two words */
__device__ __forceinline__ void pr_sign8(uint64_t g, uint64_t m, uint32_t *w0, uint32_t *w1)
{
    const uint32_t g0 = (uint32_t)g, g1 = (uint32_t)(g >> 32), m0 = (uint32_t)m, m1 = (uint32_t)(m >> 32);
    *w0 = (g0 ^ m0) + (m0 & 0x01010101u);
    *w1 = (g1 ^ m1) + (m1 & 0x01010101u);
}

/* The tables of the IQ3 types in shared memory (the mode PR_IQ3X_SIGNED, PR_IQ3S_SIGNED):
 * 1  the signed tables: entry i | s4 << 8 (IQ3_XXS, 4096 words) or i | s4 << 9 (IQ3_S, 8192 words) is
 *    grid entry i with the signs s4 (one load gives 4 signed values; 16 KB, 32 KB);
 * 0  the grid alone (256, 512 words) and the signs by arithmetic (pr_sign4);
 * 2  the grid, then the 64-bit masks of the 8 signs of a sign index (IQ3_XXS: its 7-bit index, 128
 *    masks; IQ3_S: a byte of signs, 256 masks): one 8-byte load signs 8 values (pr_sign8; 2 KB, 4 KB).
 * A P6000, real rows of the 27B (scratch bench mvb.cu), one token, IQ3_S gate / down, IQ3_XXS gate: 1 150 /
 * 137 / 232 GB/s (the 32 KB left 3 blocks an SM), 0 232 / 187 / 262, 2 159 / 145 / 254 (the 8-byte loads'
 * bank conflicts). */
#ifndef PR_IQ3X_SIGNED
#define PR_IQ3X_SIGNED 0
#endif
#ifndef PR_IQ3S_SIGNED
#define PR_IQ3S_SIGNED 0
#endif
/* (mode 0: the grid, then the 16 byte masks of 4 signs, pr_sign4t: 16 words in 16 banks, no conflict) */
#define PR_IQ3X_TAB (PR_IQ3X_SIGNED == 1 ? 4096 : PR_IQ3X_SIGNED == 2 ? 256 + 256 : 256 + 16)
#define PR_IQ3S_TAB (PR_IQ3S_SIGNED == 1 ? 8192 : PR_IQ3S_SIGNED == 2 ? 512 + 512 : 512 + 16)
__device__ __forceinline__ uint64_t pr_mask8(uint32_t b)
{
    uint64_t m = 0;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        m |= ((b >> k) & 1u) ? (uint64_t)0xff << (8 * k) : 0;
    }
    return m;
}
__device__ __forceinline__ void pr_tables_fill(uint32_t *t3x, uint32_t *t3s)
{
    if (t3x != NULL) {
        for (int e = threadIdx.x; e < PR_IQ3X_TAB; e += blockDim.x) {
            if (PR_IQ3X_SIGNED == 1) {
                t3x[e] = pr_sign4(pr_iq3xxs_grid[e & 255], (uint32_t)e >> 8);
            } else if (e < 256) {
                t3x[e] = pr_iq3xxs_grid[e];
            } else if (PR_IQ3X_SIGNED == 0) {
                t3x[e] = (uint32_t)pr_mask8((uint32_t)(e - 256));
            } else {
                const uint64_t m = nm_iq2_signs64[(e - 256) >> 1];
                t3x[e] = (e & 1) ? (uint32_t)(m >> 32) : (uint32_t)m;
            }
        }
    }
    if (t3s != NULL) {
        for (int e = threadIdx.x; e < PR_IQ3S_TAB; e += blockDim.x) {
            if (PR_IQ3S_SIGNED == 1) {
                t3s[e] = pr_sign4(pr_iq3s_grid[e & 511], (uint32_t)e >> 9);
            } else if (e < 512) {
                t3s[e] = pr_iq3s_grid[e];
            } else if (PR_IQ3S_SIGNED == 0) {
                t3s[e] = (uint32_t)pr_mask8((uint32_t)(e - 512));
            } else {
                const uint64_t m = pr_mask8((uint32_t)(e - 512) >> 1);
                t3s[e] = (e & 1) ? (uint32_t)(m >> 32) : (uint32_t)m;
            }
        }
    }
}

/* The 8-bit sign byte of IQ2/IQ3_XXS's 7-bit sign index (the parity of the 7 as bit 7). */
__device__ __forceinline__ uint32_t pr_sg8(uint32_t s7)
{
    return s7 | ((__popc(s7) & 1u) << 7);
}

/* ---- the parts ----
 * Each decoder reads the planes of a row (or of the step of a tile in shared memory, with the same
 * relative layout: qmoe_dp4a.cu kd_load_w) through pointers to each plane and the part p within them.
 * IQ3 parts take a signed table (t != NULL: pr_tables_fill) or the grid of global memory. */

/* IQ4_XS part j of the block blk (136 bytes) */
__device__ __forceinline__ void pr_part_iq4xs(const uint8_t *blk, int j, uint32_t *w, float *d)
{
    const uint32_t h0 = *(const uint32_t *)blk, sl = *(const uint32_t *)(blk + 4);
    const uint2 qa = *(const uint2 *)(blk + 8 + 16 * j), qb = *(const uint2 *)(blk + 16 + 16 * j);
    const int ls = (int)(((sl >> (4 * j)) & 15u) | (((h0 >> (16 + 2 * j)) & 3u) << 4));
    const uint32_t qq[4] = {qa.x, qa.y, qb.x, qb.y};
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int2 v = pr_lut16(qq[k]);
        w[k] = (uint32_t)v.x;
        w[4 + k] = (uint32_t)v.y;
    }
    *d = __half2float(__ushort_as_half((uint16_t)(h0 & 0xffffu))) * (float)(ls - 32);
}

/* IQ3_S (KQ_PR_IQ3S): q the 8 low index bytes, sg the 32 sign bits, qh the 9th bits, sc the 4-bit scale */
__device__ __forceinline__ void pr_part_iq3s(uint2 q, uint32_t sg, uint32_t qh, const uint32_t *t, uint32_t *w)
{
    const uint32_t qi[2] = {q.x, q.y};
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t i = ((qi[k / 4] >> (8 * (k % 4))) & 255u) | (((qh >> k) & 1u) << 8);
        const uint32_t s4 = (sg >> (4 * k)) & 15u;
        if (t != NULL && PR_IQ3S_SIGNED == 2) {
            if (k % 2 == 0) {
                const uint2 m = ((const uint2 *)(t + 512))[(sg >> (4 * k)) & 255u];
                const uint32_t i1 = ((qi[k / 4] >> (8 * (k % 4) + 8)) & 255u) | (((qh >> (k + 1)) & 1u) << 8);
                w[k] = (t[i] ^ m.x) + (m.x & 0x01010101u);
                w[k + 1] = (t[i1] ^ m.y) + (m.y & 0x01010101u);
            }
        } else {
            w[k] = t == NULL ? pr_sign4(__ldg(&pr_iq3s_grid[i]), s4) : PR_IQ3S_SIGNED ? t[i | (s4 << 9)] :
                   pr_sign4t(t[i], s4, t + 512);
        }
    }
}

/* IQ3_XXS (KQ_PR_IQ3X): q the 8 index bytes, aux the signs and the scale */
__device__ __forceinline__ void pr_part_iq3x(uint2 q, uint32_t aux, const uint32_t *t, uint32_t *w)
{
    const uint32_t qi[2] = {q.x, q.y};
    if (t != NULL && PR_IQ3X_SIGNED == 2) {
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint2 m = ((const uint2 *)(t + 256))[(aux >> (7 * l)) & 127u];
            const uint32_t i0 = (qi[l / 2] >> (16 * (l % 2))) & 255u, i1 = (qi[l / 2] >> (16 * (l % 2) + 8)) & 255u;
            w[2 * l] = (t[i0] ^ m.x) + (m.x & 0x01010101u);
            w[2 * l + 1] = (t[i1] ^ m.y) + (m.y & 0x01010101u);
        }
        return;
    }
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t sg = pr_sg8((aux >> (7 * l)) & 127u);
        #pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int k = 2 * l + h;
            const uint32_t i = (qi[k / 4] >> (8 * (k % 4))) & 255u, s4 = (sg >> (4 * h)) & 15u;
            w[k] = t == NULL ? pr_sign4(__ldg(&pr_iq3xxs_grid[i]), s4) : PR_IQ3X_SIGNED ? t[i | (s4 << 8)] :
                   pr_sign4t(t[i], s4, t + 256);
        }
    }
}

/* Q3_K (KQ_PR_Q3K): c the two words of 2-bit codes, hi the inverted high bits */
__device__ __forceinline__ void pr_part_q3k(uint2 c, uint32_t hi, uint32_t *w)
{
    const uint32_t cw[2] = {c.x, c.y};
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint32_t q2 = (cw[k / 4] >> (2 * (k % 4))) & 0x03030303u;
        const uint32_t hb = (hi >> k) & 0x01010101u;
        w[k] = q2 | (hb * 0xfcu);
    }
}

/* IQ2_XS (KQ_PR_IQ2S): c the 4 uint16 codes of the part */
__device__ __forceinline__ void pr_part_iq2s(uint2 c, uint32_t *w)
{
    const uint32_t cw[2] = {c.x, c.y};
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t code = (cw[l / 2] >> (16 * (l % 2))) & 0xffffu;
        pr_sign8(__ldg(&pr_iq2xs_grid[code & 511u]), __ldg(&nm_iq2_signs64[code >> 9]), &w[2 * l], &w[2 * l + 1]);
    }
}

/* IQ2_XXS (KQ_IQ2R): pt the part (x the 4 grid indices, y the signs and the scale) */
__device__ __forceinline__ void pr_part_iq2x(uint2 pt, uint32_t *w)
{
    #pragma unroll
    for (int l = 0; l < 4; ++l) {
        pr_sign8(__ldg(&nm_iq2_grid[(pt.x >> (8 * l)) & 255u]), __ldg(&nm_iq2_signs64[(pt.y >> (7 * l)) & 127u]),
                 &w[2 * l], &w[2 * l + 1]);
    }
}

/* Part p of a row w (cols values) of type `type` (kq_pr_type): the 8 words and the scales of values
 * 0-15 (dl) and 16-31 (dh). t3x, t3s: the signed tables (or NULL: the grids of global memory). */
__device__ __forceinline__ void pr_part(int type, const uint8_t *w, int cols, int p, const uint32_t *t3x,
                                        const uint32_t *t3s, uint32_t *wd, float *dl, float *dh)
{
    const int c4 = cols / 4;
    if (type == KQ_PR_IQ4X) {
        pr_part_iq4xs(w + (size_t)(p >> 3) * 136, p & 7, wd, dl);
        *dh = *dl;
    } else if (type == KQ_PR_IQ3S) {
        const uint8_t *hp = w + c4 + cols / 8, *sp = hp + cols / 32, *dp = sp + cols / 64;
        pr_part_iq3s(((const uint2 *)w)[p], ((const uint32_t *)(w + c4))[p], hp[p], t3s, wd);
        const uint32_t sc = (sp[p >> 1] >> (4 * (p & 1))) & 15u;
        *dl = *dh = __half2float(((const __half *)dp)[p >> 3]) * (float)(2 * sc + 1);
    } else if (type == KQ_PR_IQ3X) {
        const uint32_t aux = ((const uint32_t *)(w + c4))[p];
        pr_part_iq3x(((const uint2 *)w)[p], aux, t3x, wd);
        *dl = *dh = __half2float(((const __half *)(w + c4 + cols / 8))[p >> 3]) * (float)(2 * (aux >> 28) + 1) * 0.25f;
    } else if (type == KQ_PR_Q3K) {
        pr_part_q3k(((const uint2 *)w)[p], ((const uint32_t *)(w + c4))[p], wd);
        const int8_t *sc = (const int8_t *)(w + c4 + cols / 8);
        const float d = __half2float(((const __half *)(w + c4 + cols / 8 + cols / 16))[p >> 3]);
        *dl = d * (float)sc[2 * p];
        *dh = d * (float)sc[2 * p + 1];
    } else if (type == KQ_PR_Q6K) {
        const uint4 lo = ((const uint4 *)w)[p];
        const uint2 hi = ((const uint2 *)(w + cols / 2))[p];
        const uint32_t L[4] = {lo.x, lo.y, lo.z, lo.w}, H[2] = {hi.x, hi.y};
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint32_t e = ((L[k & 3] >> (4 * (k >> 2))) & 0x0f0f0f0fu) |
                               (((H[k >> 2] >> (2 * (k & 3))) & 0x03030303u) << 4);
            wd[k] = e | ((e & 0x20202020u) * 6u);
        }
        const int8_t *sc = (const int8_t *)(w + cols / 2 + c4);
        const float d = __half2float(((const __half *)(w + cols / 2 + c4 + cols / 16))[p >> 3]);
        *dl = d * (float)sc[2 * p];
        *dh = d * (float)sc[2 * p + 1];
    } else if (type == KQ_PR_IQ2S) {
        pr_part_iq2s(((const uint2 *)w)[p], wd);
        const uint32_t sc = w[c4 + p];
        const float d = 0.25f * __half2float(((const __half *)(w + c4 + cols / 32))[p >> 3]);
        *dl = d * (0.5f + (float)(sc & 15u));
        *dh = d * (0.5f + (float)(sc >> 4));
    } else {
        const uint2 pt = nm_q2_part(w, p >> 3, p & 7);
        pr_part_iq2x(pt, wd);
        *dl = *dh = __half2float(__ushort_as_half(nm_q2_dh(w, cols, p >> 3))) * (float)(2 * (pt.y >> 28) + 1) * 0.125f;
    }
}

/* 8 values of a row from column c0 (a multiple of 8), as float32 (kq_dequant8, kq_row_part: the paths on
 * float x; a call, so the generic kernels keep their registers) */
static __device__ __noinline__ void kq_pr_dequant8(int type, const uint8_t *w, int cols, int c0, float *v)
{
    uint32_t wd[8];
    float dl, dh;
    pr_part(type, w, cols, c0 / 32, NULL, NULL, wd, &dl, &dh);
    const int k0 = (c0 % 32) / 4;
    const float d = k0 < 4 ? dl : dh;
    #pragma unroll
    for (int u = 0; u < 4; ++u) {
        v[u] = d * (float)(int8_t)((wd[k0] >> (8 * u)) & 255u);
        v[4 + u] = d * (float)(int8_t)((wd[k0 + 1] >> (8 * u)) & 255u);
    }
}

/* kq_row_part on float x: the sums of the parts part, part + nparts, ... of 32 values (each lane its own
 * part of 8 values; the caller adds the lanes) */
static __device__ __noinline__ float kq_pr_row_part(int type, const uint8_t *w, const float *x, int cols, int part,
                                                    int nparts, int lane)
{
    float sum = 0.f;
    for (int c0 = 8 * lane + 256 * part; c0 < cols; c0 += 256 * nparts) {
        float v[8];
        kq_pr_dequant8(type, w, cols, c0, v);
        const float4 a = *(const float4 *)(x + c0), b = *(const float4 *)(x + c0 + 4);
        sum += v[0] * a.x + v[1] * a.y + v[2] * a.z + v[3] * a.w + v[4] * b.x + v[5] * b.y + v[6] * b.z + v[7] * b.w;
    }
    return sum;
}
