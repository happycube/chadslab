/* ---------- Qwen3.5 on the GGUF formats (QWEN_PLAN.md, phase 4) ----------
 * The records of compile_qwen_step (reclank/models/qwen/qwen35.py). The products read x
 * in float32: GP_KQ_QUANT does nothing on the GPU, and GP_KQ_LINEAR reads
 * operand 3 (the float rows). The weights are the blocks of the GGUF file
 * (the ggml types, backends/cpu/csrc/kernels/kq). One warp computes one row. */
#define KQ_F32 0
/* Q4_0: blocks of 32 values, 18 bytes: d (float16), then 16 bytes (value j
 * in the low 4 bits of byte j, value j + 16 in the high 4 bits). A value is
 * d (q - 8). The QAT files of Gemma 4 keep their matrices in Q4_0; the E4B
 * takes them as GP_KQ_LINEAR on the GPU (int8 x, kq_rows_i8). */
#define KQ_Q4_0 2
#define KQ_Q8_0 8
#define KQ_Q4_K 12
#define KQ_Q5_K 13
#define KQ_Q6_K 14
/* Q5_1: blocks of 32 values, 24 bytes: d, m (float16), the high bits
 * (32 bits), and the low 4 bits (value j < 16 in the low half of byte j,
 * value j + 16 in its high half). A value is d q + m. */
#define KQ_Q5_1 7
/* bfloat16 rows, and NVFP4 in the rows of backends/cpu/csrc/kernels/kq (KQ_NV4): the E2M1
 * codes (cols / 2 bytes; 16 for each block of 32 values: value j low, j + 16
 * high), the E4M3 scales (2 for each block), the float32 scale of the
 * matrix, zeros to 16 bytes. The codes start at a multiple of 16 bytes: a
 * fragment of the tensor cores is one aligned load. */
#define KQ_BF16 30
#define KQ_NV4 51
/* BF12 (backends/cpu/csrc/kernels/kq, plan-scripts/BF12_PLAN.md): the bfloat16 values of
 * a row in 12.25 bits. The sign and the 7 bits of the mantissa of each value
 * (cols bytes), the gap of its exponent below the exponent of its group of 32
 * (4 bits: value j of the group in the low half of byte j of the group's 16,
 * value j + 16 in the high half; cols / 2 bytes), the exponent of each group
 * (cols / 32 bytes), zeros to 16 bytes. The code gap 15 with the byte 0x80
 * (a negative zero) is the zero. */
#define KQ_BF12 57
/* The KQ_BF12 matrices that the steps and the small groups (1 to 8 tokens)
 * run on the tensor cores (k_kq_bf12_tc; the kernel by bt_cls): a choice by
 * the shape alone, so a token alone and in a verify group take the same
 * kernel. */
#define BT_MIN_ROWS 1024
__host__ __device__ __forceinline__ int bt_shape(int type, int rows, int cols)
{
    return type == KQ_BF12 && cols % 32 == 0;
}
/* NVFP4 in groups of 16 rows (KQ_NVX of backends/cpu/csrc/kernels/kq): a group is 16
 * bytes (the float32 scale of the matrix, zeros), then 288 bytes for each
 * block of 32 columns: 8 steps of 32 bytes (step s: values 4s .. 4s + 3;
 * byte 4r + u: row r in the low 4 bits, row r + 8 in the high 4 bits) and
 * 32 E4M3 scales (values 0-15 of the 16 rows, then values 16-31). The 4
 * bytes at 32 s + 4 r are the fragments of the tensor cores for rows r and
 * r + 8. A "row" is cols / 32 * 18 + 1 bytes. */
#define KQ_NVX 53
#define KQ_NVX_BB 288
/* KQ_MXF8 (FP4-Local): an MXFP8 row, its cols E4M3 codes, then the E8M0 scale of each 32, zeros to
 * 16 bytes (kernels/quant/mxf8.cu) */
#define KQ_MXF8 66
/* Q8_0 in rows for the GPU (reclank/models/qwen/gpu35.py): the int8 values of the
 * row, then the float16 scale of each 32 values, then zeros to a multiple
 * of 16 bytes. Thus the values of each row start at a multiple of 16
 * bytes. */
#define KQ_Q8_R 100

__host__ __device__ __forceinline__ size_t kq_row_bytes(int type, int cols)
{
    return type == KQ_F32 ? (size_t)cols * 4 :
           type == KQ_Q8_R ? ((size_t)cols / 32 * 34 + 15) / 16 * 16 :
           type == KQ_Q8_0 ? (size_t)cols / 32 * 34 :
           type == KQ_Q4_0 ? (size_t)cols / 32 * 18 :
           type == KQ_Q5_1 ? (size_t)cols / 32 * 24 :
           type == KQ_BF16 ? (size_t)cols * 2 :
           type == KQ_BF12 ? ((size_t)cols + cols / 2 + cols / 32 + 15) / 16 * 16 :
           type == KQ_NV4 ? ((size_t)cols / 2 + (size_t)cols / 16 + 4 + 15) / 16 * 16 :
           type == KQ_NVX ? (size_t)cols / 32 * 18 + 1 :
           type == KQ_MXF8 ? ((size_t)cols + cols / 32 + 15) / 16 * 16 :
           type == KQ_IQ2_XXS ? (size_t)cols / 256 * 66 :
           type == KQ_IQ2R ? nm_q2_rb(cols) :
           type == KQ_Q4_K ? (size_t)cols / 256 * 144 : type == KQ_Q5_K ? (size_t)cols / 256 * 176 :
           (size_t)cols / 256 * 210;
}

__device__ __forceinline__ float kq_half(const uint8_t *p)
{
    return __half2float(__ushort_as_half((uint16_t)(p[0] | (p[1] << 8))));
}

/* float32 rounded to bfloat16 (to nearest even), as float32 */
__device__ __forceinline__ float rc_bf16r_k(float v)
{
    unsigned u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

__device__ __forceinline__ float kq_e4m3(uint8_t b)
{
    int e = (b >> 3) & 15, m = b & 7;
    float v = e == 0 ? (float)m * (1.f / 512.f) : __uint_as_float(((uint32_t)(e + 120) << 23) | ((uint32_t)m << 20));
    return (b & 0x80) ? -v : v;
}

/* Twice the value of an E2M1 code. */
__device__ __forceinline__ float kq_e2m1x2(int c)
{
    /* the bytes of the constant: 0, 1, 2, 3, 4, 6, 8, 12 */
    float v = (float)((0x0C08060403020100ull >> (8 * (c & 7))) & 0xff);
    return (c & 8) ? -v : v;
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

/* An E4M3 scale times 2^-8, with no branch: its bits go to float16 as they
 * are (the exponent bias of float16 is 8 more). */
__device__ __forceinline__ float kq_e4m3s(uint32_t b)
{
    return __half2float(__ushort_as_half((uint16_t)(((b & 0x7f) << 7) | ((b & 0x80) << 8))));
}

/* The sum of 4 int8 (a) times 4 floats. */
__device__ __forceinline__ float kq_dot4s8(uint32_t a, float4 x)
{
    return (float)(int8_t)(a & 255) * x.x + (float)(int8_t)((a >> 8) & 255) * x.y +
           (float)(int8_t)((a >> 16) & 255) * x.z + (float)(int8_t)(a >> 24) * x.w;
}

/* The 16 rows of the KQ_NVX group at wg with x (cols values), by a warp.
 * Lane l reads 8 bytes of each block: step l / 4, rows 2q, 2q + 1 (low 4
 * bits) and 2q + 8, 2q + 9 (high), q = l % 4. The lanes of each q add
 * their sums; lane q < 4 gets rows 2q, 2q + 1, 2q + 8, 2q + 9 in v. */
__device__ void kq_nvx_group(const uint8_t *wg, const float *x, int cols, float *v)
{
    int lane = threadIdx.x % 32, st = lane / 4, q = lane % 4, h = st / 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int b = 0; b < cols / 32; ++b) {
        const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
        uint2 c = *(const uint2 *)(blk + 8 * lane);
        float4 xv = *(const float4 *)(x + 32 * b + 4 * st);
        uint32_t s01 = *(const uint16_t *)(blk + 256 + 16 * h + 2 * q);
        uint32_t s89 = *(const uint16_t *)(blk + 256 + 16 * h + 8 + 2 * q);
        acc[0] += kq_e4m3s(s01 & 255) * kq_dot4s8(kt_e2m1x4(c.x), xv);
        acc[1] += kq_e4m3s(s01 >> 8) * kq_dot4s8(kt_e2m1x4(c.y), xv);
        acc[2] += kq_e4m3s(s89 & 255) * kq_dot4s8(kt_e2m1x4(c.x >> 4), xv);
        acc[3] += kq_e4m3s(s89 >> 8) * kq_dot4s8(kt_e2m1x4(c.y >> 4), xv);
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        for (int o = 4; o < 32; o <<= 1) {
            acc[i] += __shfl_xor_sync(RC_FULL_MASK, acc[i], o);
        }
    }
    float gs = 128.f * *(const float *)wg;           /* 0.5 g, and the 2^8 of kq_e4m3s */
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = gs * acc[i];
    }
}

/* 8 values of row rin of the KQ_NVX group at wg from column c0 (a multiple
 * of 8), as float32. */
__device__ void kq_dequant8_nvx(const uint8_t *wg, int rin, int c0, float *v)
{
    const uint8_t *blk = wg + 16 + (size_t)(c0 / 32) * KQ_NVX_BB;
    int j = c0 % 32, sh = rin < 8 ? 0 : 4;
    float sc = 128.f * *(const float *)wg * kq_e4m3s(blk[256 + 16 * (j / 16) + rin]);
    uint32_t a = kt_e2m1x4(*(const uint32_t *)(blk + 32 * (j / 4) + 4 * (rin % 8)) >> sh);
    uint32_t b = kt_e2m1x4(*(const uint32_t *)(blk + 32 * (j / 4 + 1) + 4 * (rin % 8)) >> sh);
    for (int u = 0; u < 4; ++u) {
        v[u] = sc * (float)(int8_t)((a >> (8 * u)) & 255);
        v[4 + u] = sc * (float)(int8_t)((b >> (8 * u)) & 255);
    }
}

__device__ __forceinline__ float kq_bf16(uint16_t h)
{
    return __uint_as_float((uint32_t)h << 16);
}

/* The bfloat16 bits of n (8 or 16) values of a KQ_BF12 row from column c
 * (a multiple of n, in one half of a group of 32). */
template <int N>
__device__ __forceinline__ void kq_bf12_bits(const uint8_t *w, int cols, int c, uint32_t *bits)
{
    int g = c >> 5, h = (c >> 4) & 1;
    uint32_t lw[N / 4], hw[N / 4];
    #pragma unroll
    for (int k = 0; k < N / 4; ++k) {
        lw[k] = ((const uint32_t *)(w + c))[k];
        hw[k] = ((const uint32_t *)(w + cols + 16 * g + (c & 15)))[k];
    }
    uint32_t E2 = ((uint32_t)w[cols + cols / 2 + g] | 256u) * 0x00010001u;
    #pragma unroll
    for (int k = 0; k < N / 4; ++k) {
        uint32_t p0 = bf12_frag(lw[k], hw[k], 4 * h, E2), p1 = bf12_frag(lw[k] >> 16, hw[k] >> 16, 4 * h, E2);
        bits[4 * k] = p0 & 0xffff;
        bits[4 * k + 1] = p0 >> 16;
        bits[4 * k + 2] = p1 & 0xffff;
        bits[4 * k + 3] = p1 >> 16;
    }
}

/* The decode of the steps (kq_bf12_dec16) is bound by the instructions:
 * the 3090 at its power cap runs a memory-bound kernel at 500-900 MHz, and
 * Ampere runs the integer ones on one of its two datapaths. (The exponent
 * by an IMAD a value, on the FMA pipe, with a PRMT and a LOP3 for each
 * value, was not faster: 0.97-1.09 times bfloat16 for a token, against
 * 0.87-0.94 here.)
 * bf12_pair makes two values (the 16-bit lanes of a word: b the bytes, g
 * the gaps) in 4 instructions with no test of the zero code; E7 is
 * (E + 256) << 7 in each lane (no borrow out of a lane). The sign and the
 * mantissa: (b & 0x80) * 0xff + b = (b & 0x80) << 8 | (b & 0x7f). */
__device__ __forceinline__ uint32_t bf12_pair(uint32_t b, uint32_t g, uint32_t E7)
{
    uint32_t ex7 = (E7 - (g << 7)) & 0x7f807f80u;
    return ((b & 0x00800080u) * 0xffu + b) | ex7;
}

/* The bfloat16 bits of the 16 values of half h of a group of a KQ_BF12 row
 * from its words L (the 16 lo bytes), H (the 16 hi bytes of the group), and
 * E, as 8 words of two values (value 2i in the low lane of p[i]). A zero
 * code (b 0x80 with gap 15) is found 4 values at a time (a zero byte of
 * b ^ 0x80 | g ^ 15); a word with one takes the exact test (rare). */
__device__ __forceinline__ void kq_bf12_dec16(uint4 L, uint4 H, int E, int h, uint32_t *p)
{
    uint32_t E7 = (((uint32_t)E | 256u) << 7) * 0x00010001u;
    uint32_t lw[4] = {L.x, L.y, L.z, L.w}, hw[4] = {H.x, H.y, H.z, H.w};
    uint32_t any = 0, g4[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        g4[k] = (hw[k] >> (4 * h)) & 0x0f0f0f0fu;
        uint32_t zb = (lw[k] ^ 0x80808080u) | (g4[k] ^ 0x0f0f0f0fu);
        any |= (zb - 0x01010101u) & ~zb & 0x80808080u;
        p[2 * k] = bf12_pair(__byte_perm(lw[k], 0, 0x4140), __byte_perm(g4[k], 0, 0x4140), E7);
        p[2 * k + 1] = bf12_pair(__byte_perm(lw[k], 0, 0x4342), __byte_perm(g4[k], 0, 0x4342), E7);
    }
    if (any) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            uint32_t c0 = __byte_perm(lw[k], 0, 0x4140) | __byte_perm(g4[k], 0, 0x4140) << 8;
            uint32_t c1 = __byte_perm(lw[k], 0, 0x4342) | __byte_perm(g4[k], 0, 0x4342) << 8;
            p[2 * k] &= ~__vcmpeq2(c0, 0x0f800f80u);
            p[2 * k + 1] &= ~__vcmpeq2(c1, 0x0f800f80u);
        }
    }
}

/* The sums of the 16 values of a KQ_BF12 row from column c (8 words of two
 * bfloat16 values) with the rows of x of NT tokens (t of them live): the
 * same order for each token, so a token alone (kq_row_part) and in a group
 * get the same bits. */
template <int NT>
__device__ __forceinline__ void kq_bf12_fma16(const uint32_t *p, const float *x, int cols, int c,
                                              int t, float *sum)
{
    float wv[16];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        wv[2 * i] = __uint_as_float(p[i] << 16);
        wv[2 * i + 1] = __uint_as_float(p[i] & 0xffff0000u);
    }
    #pragma unroll
    for (int j = 0; j < NT; ++j) {
        if (j < t) {
            const float4 *x4 = (const float4 *)(x + (size_t)j * cols + c);
            float s = sum[j];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float4 a = x4[i];
                s = fmaf(wv[4 * i], a.x, s);
                s = fmaf(wv[4 * i + 1], a.y, s);
                s = fmaf(wv[4 * i + 2], a.z, s);
                s = fmaf(wv[4 * i + 3], a.w, s);
            }
            sum[j] = s;
        }
    }
}

/* The sums of part part of nparts of a KQ_BF12 row with x of NT tokens:
 * lane l takes 16 values (one half of a group of 32) of each part of 512
 * columns. The words of BF12_U parts are loaded before any is decoded (4
 * spilled in the 64 registers of k_kq_linear_bf12<1>). */
#define BF12_U 2
template <int NT>
__device__ __forceinline__ void kq_bf12_row(const uint8_t *w, const float *x, int cols, int t, int part,
                                            int nparts, float *sum)
{
    int lane = threadIdx.x % 32, h = lane & 1;
    for (int c0 = 16 * lane + 512 * part; c0 < cols; c0 += BF12_U * 512 * nparts) {
        uint4 L[BF12_U], H[BF12_U];
        int E[BF12_U];
        #pragma unroll
        for (int u = 0; u < BF12_U; ++u) {
            int c = c0 + u * 512 * nparts;
            if (c < cols) {
                L[u] = *(const uint4 *)(w + c);
                H[u] = *(const uint4 *)(w + cols + 16 * (c >> 5));
                E[u] = w[cols + cols / 2 + (c >> 5)];
            }
        }
        #pragma unroll
        for (int u = 0; u < BF12_U; ++u) {
            int c = c0 + u * 512 * nparts;
            if (c < cols) {
                uint32_t pw[8];
                kq_bf12_dec16(L[u], H[u], E[u], h, pw);
                kq_bf12_fma16<NT>(pw, x, cols, c, t, sum);
            }
        }
    }
}

__device__ __forceinline__ void kq_scale_min(const uint8_t *q, int j, int *sc, int *m)
{
    if (j < 4) {
        *sc = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

__device__ __forceinline__ float kq_dot4(uint32_t q, float4 x)
{
    return (float)(q & 255) * x.x + (float)((q >> 8) & 255) * x.y +
           (float)((q >> 16) & 255) * x.z + (float)(q >> 24) * x.w;
}

/* The product of one row with x (cols values), for part part of nparts of
 * the blocks of the row (part 0 of 1: the whole row). All the lanes of the
 * warp take part; each gets the sum. The parts let several warps share a
 * long row, so more loads are in flight (k_kq_linear_split). */
__device__ float kq_row_part(int type, const uint8_t *w, const float *x, int cols, int part,
                            int nparts)
{
    int lane = threadIdx.x % 32;
    float sum = 0.f;
    if (type == KQ_F32) {
        const float4 *w4 = (const float4 *)w, *x4 = (const float4 *)x;
        for (int k = lane + 32 * part; k < cols / 4; k += 32 * nparts) {
            float4 a = w4[k], b = x4[k];
            sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
    } else if (type == KQ_Q8_R) {
        /* lane l: 16 values (one 16-byte load) of each part of 512 */
        const __half *d = (const __half *)(w + cols);
        for (int c = 16 * lane + 512 * part; c < cols; c += 512 * nparts) {
            uint4 q = *(const uint4 *)(w + c);
            const float4 *x4 = (const float4 *)(x + c);
            uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            float acc = 0.f;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float4 xv = x4[u];
                acc += (float)(int8_t)(qq[u] & 255) * xv.x + (float)(int8_t)((qq[u] >> 8) & 255) * xv.y +
                       (float)(int8_t)((qq[u] >> 16) & 255) * xv.z + (float)(int8_t)(qq[u] >> 24) * xv.w;
            }
            sum += __half2float(d[c / 32]) * acc;
        }
    } else if (type == KQ_Q8_0) {
        /* 4 blocks for each pass: lane l takes 4 values of block l / 8. */
        int o = 4 * (lane % 8);
        for (int b = lane / 8 + 4 * part; b < cols / 32; b += 4 * nparts) {
            const uint8_t *blk = w + (size_t)b * 34;
            const uint16_t *q = (const uint16_t *)(blk + 2 + o);
            uint32_t a = q[0], c = q[1];
            float4 xv = *(const float4 *)(x + b * 32 + o);
            sum += kq_half(blk) * ((float)(int8_t)(a & 255) * xv.x +
                                   (float)(int8_t)(a >> 8) * xv.y +
                                   (float)(int8_t)(c & 255) * xv.z +
                                   (float)(int8_t)(c >> 8) * xv.w);
        }
    } else if (type == KQ_Q4_0) {
        /* As Q8_0: lane l takes 4 values of block l / 8 of 4 blocks (the
         * low 4 bits of 4 bytes, or their high 4 bits). */
        int o = 4 * (lane % 8), sh = o < 16 ? 0 : 4;
        for (int b = lane / 8 + 4 * part; b < cols / 32; b += 4 * nparts) {
            const uint8_t *blk = w + (size_t)b * 18;
            const uint16_t *q = (const uint16_t *)(blk + 2 + o % 16);
            uint32_t a = (uint32_t)q[0] | ((uint32_t)q[1] << 16);
            float4 xv = *(const float4 *)(x + b * 32 + o);
            sum += kq_half(blk) * ((float)((int)((a >> sh) & 15) - 8) * xv.x +
                                   (float)((int)((a >> (8 + sh)) & 15) - 8) * xv.y +
                                   (float)((int)((a >> (16 + sh)) & 15) - 8) * xv.z +
                                   (float)((int)((a >> (24 + sh)) & 15) - 8) * xv.w);
        }
    } else if (type == KQ_BF16) {
        const uint16_t *h = (const uint16_t *)w;
        for (int c = 8 * lane + 256 * part; c < cols; c += 256 * nparts) {
            uint4 q = *(const uint4 *)(h + c);
            uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            const float4 *x4 = (const float4 *)(x + c);
            float4 a = x4[0], b = x4[1];
            float xv[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                sum += kq_bf16((uint16_t)(qq[u] & 0xffff)) * xv[2 * u] +
                       kq_bf16((uint16_t)(qq[u] >> 16)) * xv[2 * u + 1];
            }
        }
    } else if (type == KQ_BF12) {
        /* lane l: 16 values (one half of a group of 32) of each part of 512 */
        kq_bf12_row<1>(w, x, cols, 1, part, nparts, &sum);
    } else if (type == KQ_MXF8) {
        /* lane l: 16 values (one load) of each part of 512; the products on x in bfloat16 values
         * (FP4-Local's a16), each 16 summed, times the 2^(s - 127) of their 32 */
        for (int c = 16 * lane + 512 * part; c < cols; c += 512 * nparts) {
            uint4 q = *(const uint4 *)(w + c);
            uint32_t qq[4] = {q.x, q.y, q.z, q.w};
            const float4 *x4 = (const float4 *)(x + c);
            float acc = 0.f;
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                float4 xv = x4[u];
                acc += kq_e4m3((uint8_t)(qq[u] & 255)) * rc_bf16r_k(xv.x) +
                       kq_e4m3((uint8_t)((qq[u] >> 8) & 255)) * rc_bf16r_k(xv.y) +
                       kq_e4m3((uint8_t)((qq[u] >> 16) & 255)) * rc_bf16r_k(xv.z) +
                       kq_e4m3((uint8_t)(qq[u] >> 24)) * rc_bf16r_k(xv.w);
            }
            sum += acc * __int_as_float((int)w[cols + c / 32] << 23);
        }
    } else if (type == KQ_NV4) {
        /* As Q8_0: lane l takes 4 values of block l / 8 of 4 blocks. */
        float g = *(const float *)(w + cols / 2 + cols / 16);
        int o = 4 * (lane % 8);
        for (int b = lane / 8 + 4 * part; b < cols / 32; b += 4 * nparts) {
            float sc = 0.5f * g * kq_e4m3(w[cols / 2 + 2 * b + (o < 16 ? 0 : 1)]);
            const uint8_t *q = w + 16 * b + (o % 16);
            int sh = o < 16 ? 0 : 4;
            float4 xv = *(const float4 *)(x + b * 32 + o);
            sum += sc * (kq_e2m1x2((q[0] >> sh) & 15) * xv.x + kq_e2m1x2((q[1] >> sh) & 15) * xv.y +
                         kq_e2m1x2((q[2] >> sh) & 15) * xv.z + kq_e2m1x2((q[3] >> sh) & 15) * xv.w);
        }
    } else if (type == KQ_Q5_1) {
        /* As Q8_0: lane l takes 4 values of block l / 8 of 4 blocks. */
        int o = 4 * (lane % 8);
        for (int b = lane / 8 + 4 * part; b < cols / 32; b += 4 * nparts) {
            const uint8_t *blk = w + (size_t)b * 24;
            uint32_t qh = *(const uint32_t *)(blk + 4);
            uint32_t qs = *(const uint32_t *)(blk + 8 + o % 16);
            uint32_t q = o < 16 ? qs & 0x0f0f0f0fu : (qs >> 4) & 0x0f0f0f0fu;
            uint32_t h = qh >> o;
            q |= ((h & 1) << 4) | ((h & 2) << 11) | ((h & 4) << 18) | ((h & 8) << 25);
            float4 xv = *(const float4 *)(x + b * 32 + o);
            sum += kq_half(blk) * kq_dot4(q, xv) +
                   kq_half(blk + 2) * (xv.x + xv.y + xv.z + xv.w);
        }
    } else if (type == KQ_IQ2_XXS) {
        /* a block for the warp: lane l the 8 values of grid index l % 4 of part l / 4 */
        for (int b = part; b < cols / 256; b += nparts) {
            sum += nm_iq2_lane(w + (size_t)b * 66, x + (size_t)b * 256, lane);
        }
    } else if (type == KQ_Q4_K || type == KQ_Q5_K) {
        /* lane l: 4 bytes of the part pair c = l / 8: 4 values of part 2c
         * (the low 4 bits) and 4 of part 2c + 1 (the high 4 bits). */
        int five = type == KQ_Q5_K, c = lane / 8, sub = lane % 8;
        size_t bs = five ? 176 : 144;
        for (int b = part; b < cols / 256; b += nparts) {
            const uint8_t *blk = w + (size_t)b * bs;
            const uint8_t *qs = blk + (five ? 48 : 16);
            uint32_t q = *(const uint32_t *)(qs + 32 * c + 4 * sub);
            uint32_t lo = q & 0x0f0f0f0fu, hi = (q >> 4) & 0x0f0f0f0fu;
            if (five) {
                uint32_t h = *(const uint32_t *)(blk + 16 + 4 * sub);
                lo |= ((h >> (2 * c)) & 0x01010101u) << 4;
                hi |= ((h >> (2 * c + 1)) & 0x01010101u) << 4;
            }
            int s0, m0, s1, m1;
            kq_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            float d = kq_half(blk), dm = kq_half(blk + 2);
            const float *xb = x + b * 256 + 64 * c + 4 * sub;
            float4 xl = *(const float4 *)xb, xh = *(const float4 *)(xb + 32);
            float sl = xl.x + xl.y + xl.z + xl.w, sh = xh.x + xh.y + xh.z + xh.w;
            sum += d * ((float)s0 * kq_dot4(lo, xl) + (float)s1 * kq_dot4(hi, xh)) -
                   dm * ((float)m0 * sl + (float)m1 * sh);
        }
    } else {
        /* Q6_K: lane l takes the values l + 32 u of each half (as
         * k_q6k_head). */
        for (int b = part; b < cols / 256; b += nparts) {
            const uint8_t *ql = w + (size_t)b * 210;
            const uint8_t *qh = ql + 128;
            const int8_t *sc = (const int8_t *)(ql + 192);
            float d = kq_half(ql + 208);
            float acc = 0.f;
            #pragma unroll
            for (int n = 0; n < 2; ++n) {
                int a = ql[64 * n + lane], bq = ql[64 * n + lane + 32], hq = qh[32 * n + lane];
                int is = 8 * n + lane / 16;
                const float *xb = x + b * 256 + 128 * n + lane;
                acc += (float)sc[is + 0] * (float)(((a & 15) | ((hq & 3) << 4)) - 32) * xb[0];
                acc += (float)sc[is + 2] * (float)(((bq & 15) | (((hq >> 2) & 3) << 4)) - 32) * xb[32];
                acc += (float)sc[is + 4] * (float)(((a >> 4) | (((hq >> 4) & 3) << 4)) - 32) * xb[64];
                acc += (float)sc[is + 6] * (float)(((bq >> 4) | (((hq >> 6) & 3) << 4)) - 32) * xb[96];
            }
            sum += d * acc;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    return sum;
}

__device__ __forceinline__ float kq_row(int type, const uint8_t *w, const float *x, int cols)
{
    return kq_row_part(type, w, x, cols, 0, 1);
}

/* The rows of a block of the products of one token (4 was 3% faster than 8,
 * and 16 or 32 were slower). */
#define KQ_RPB 4

/* GP_KQ_LINEAR of a few tokens with a block of KQ_SPLIT warps for each row:
 * warp w takes the blocks w, w + KQ_SPLIT, ... of the row (kq_row_part), and
 * the warps add their sums in shared memory. A long row (the down matrix of
 * the E4B has 40 blocks of 256) then has KQ_SPLIT times the loads in flight
 * of one warp. Only for rows of 8192 values or more: a short row (10 blocks
 * of 256) was slower (331 to 227 GB/s), a long one faster (313 to 334). */
#define KQ_SPLIT 4
__global__ void __launch_bounds__(32 * KQ_SPLIT) k_kq_linear_split(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float red[KQ_SPLIT][MT_MAX];
    int row = blockIdx.x, w = threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), type = DI(5), t = DI(8);
    const uint8_t *wr = DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(type, cols);
    for (int j = 0; j < t; ++j) {
        float v = kq_row_part(type, wr, DP(const float, 3) + (size_t)j * cols, cols, w, KQ_SPLIT);
        if (threadIdx.x % 32 == 0) {
            red[w][j] = v;
        }
    }
    __syncthreads();
    if (threadIdx.x < t) {
        float s = 0.f;
        #pragma unroll
        for (int k = 0; k < KQ_SPLIT; ++k) {
            s += red[k][threadIdx.x];
        }
        DP(float, 9)[(size_t)threadIdx.x * rows + row] = s;
    }
}

/* RECLANK_GPU_KQ_SPLIT=0 keeps k_kq_linear (a warp for each row), for a
 * test. */
static int kq_split_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_KQ_SPLIT");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* GP_KQ_LINEAR: xq, xs, xm, x, w, type, rows, cols, t, out. */
__global__ void k_kq_linear(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int rows = DI(6), cols = DI(7), type = DI(5), t = DI(8);
    if (row >= rows) {
        return;
    }
    const uint8_t *w = DP(const uint8_t, 4) + (size_t)row * kq_row_bytes(type, cols);
    for (int j = 0; j < t; ++j) {
        float v = kq_row(type, w, DP(const float, 3) + (size_t)j * cols, cols);
        if (threadIdx.x % 32 == 0) {
            DP(float, 9)[(size_t)j * rows + row] = v;
        }
    }
}

/* exp(x) with the operations of rc_dexp of the CPU (common/moe.c): the SiLU
 * of the act of the experts gets the same bits on both. */
__device__ __forceinline__ float rc_dexp(float x)
{
    x = fminf(fmaxf(x, -87.0f), 88.0f);
    float n = rintf(__fmul_rn(x, 1.44269504088896341f));
    float r = fmaf(n, -0.693359375f, x);
    r = fmaf(n, 2.12194440e-4f, r);
    float p = fmaf(1.9875691500e-4f, r, 1.3981999507e-3f);
    p = fmaf(p, r, 8.3334519073e-3f);
    p = fmaf(p, r, 4.1665795894e-2f);
    p = fmaf(p, r, 1.6666665459e-1f);
    p = fmaf(p, r, 5.0000001201e-1f);
    float z = __fmul_rn(r, r);
    float y = __fadd_rn(fmaf(p, z, r), 1.0f);
    return __fmul_rn(y, __int_as_float(((int)n + 127) << 23));
}

__device__ __forceinline__ float qw_silu(float v)
{
    return __fdiv_rn(v, __fadd_rn(1.f, rc_dexp(-v)));
}

/* GP_SIGMUL: x, g, out, n. out = x sigmoid(g). */
__global__ void k_sigmul(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < di(r, e, 3)) {
        DP(float, 2)[i] = DP(const float, 0)[i] / (1.f + expf(-DP(const float, 1)[i]));
    }
}

/* GP_Q16: x, out, n (a multiple of 128): the int16 values, then a float32
 * scale for each 128 (ops_qwen.c gp_q16_body); a warp for each 128. GP_DQ16:
 * in, x, n. */
__global__ void k_q16(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int64_t n = di(r, e, 2), b = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (b >= n / 128) {
        return;
    }
    float4 v = ((const float4 *)(DP(const float, 0) + 128 * b))[lane];
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, o));
    }
    const float inv = m > 0.f ? 32767.f / m : 0.f;
    uint8_t *out = DP(uint8_t, 1);
    ((short4 *)((int16_t *)out + 128 * b))[lane] =
        make_short4((short)__float2int_rn(v.x * inv), (short)__float2int_rn(v.y * inv),
                    (short)__float2int_rn(v.z * inv), (short)__float2int_rn(v.w * inv));
    if (lane == 0) {
        ((float *)(out + 2 * n))[b] = m / 32767.f;
    }
}

__global__ void k_dq16(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int64_t n = di(r, e, 2), b = (int64_t)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    if (b >= n / 128) {
        return;
    }
    const uint8_t *in = DP(const uint8_t, 0);
    const float s = ((const float *)(in + 2 * n))[b];
    short4 q = ((const short4 *)((const int16_t *)in + 128 * b))[lane];
    ((float4 *)(DP(float, 1) + 128 * b))[lane] = make_float4(q.x * s, q.y * s, q.z * s, q.w * s);
}

/* The largest value of the block and its index (the smallest index of
 * equal values). All threads get the result. */
__device__ void block_argmax(float v, int i, float *bv, int *bi)
{
    __shared__ float pv[32];
    __shared__ int pi[32];
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(RC_FULL_MASK, v, o);
        int i2 = __shfl_xor_sync(RC_FULL_MASK, i, o);
        if (v2 > v || (v2 == v && i2 < i)) {
            v = v2;
            i = i2;
        }
    }
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) {
        pv[w] = v;
        pi[w] = i;
    }
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    v = lane < nw ? pv[lane] : -INFINITY;
    i = lane < nw ? pi[lane] : 0x7fffffff;
    for (int o = 16; o > 0; o >>= 1) {
        float v2 = __shfl_xor_sync(RC_FULL_MASK, v, o);
        int i2 = __shfl_xor_sync(RC_FULL_MASK, i, o);
        if (v2 > v || (v2 == v && i2 < i)) {
            v = v2;
            i = i2;
        }
    }
    *bv = v;
    *bi = i;
}

/* GP_ROUTER_TOPK: logits, t, experts, k, val, idx. The softmax of each row,
 * the k largest, and their weights divided by their sum. One block of 256
 * threads for each token; at most 1024 experts. */
/* r2 (or null): a GP_HOT_SPLIT of the selection of this one token, run at
 * the end by thread 0 (one kernel in place of two: RECLANK_GPU_TOPK_SPLIT). */
__device__ void hot_split_body(const gp_rec *r, const int64_t *e);

__global__ void k_router_topk(const gp_rec *r, const int64_t *e, const gp_rec *r2 = NULL)
{
    PDL_START();
    int j = blockIdx.x, E = DI(2), k = DI(3);
    const float *l = DP(const float, 0) + (size_t)j * E;
    /* operands 6 and 7 (or none): map (the slot of each expert on the GPU,
     * or -1) and bias: the choice takes the logit + bias of each expert on
     * the GPU (RECLANK_GPU_RESIDENT_BIAS), the weights the plain softmax.
     * Operands 8 and 9 (or none): swap and sel: from the lightest, a chosen
     * expert not on the GPU gives its place to the best expert on the GPU
     * not chosen when that one has at least swap times its weight (the first
     * that has none ends it), and sel (or null) gets the choice before the
     * swaps, for HotCache (RECLANK_GPU_HOT_SWAP). */
    const int *map = r->tag[6] == GP_T_NONE ? NULL : DP(const int, 6);
    const float wb = map != NULL ? expf(df(r, e, 7)) : 1.f;
    const float swap = map != NULL && r->tag[8] != GP_T_NONE ? df(r, e, 8) : 0.f;
    int *sel = map != NULL && r->tag[9] != GP_T_NONE ? DP(int, 9) : NULL;
    float p[4];
    float m = -INFINITY;
    for (int u = 0; u < 4; ++u) {
        int x = threadIdx.x + 256 * u;
        p[u] = x < E ? l[x] : -INFINITY;
        m = fmaxf(m, p[u]);
    }
    m = block_max(m);
    float s = 0.f;
    for (int u = 0; u < 4; ++u) {
        int x = threadIdx.x + 256 * u;
        p[u] = x < E ? expf(p[u] - m) : -1.f;
        s += x < E ? p[u] : 0.f;
        if (map != NULL && x < E && map[x] >= 0) {
            p[u] *= wb;             /* (the sum s has the plain values) */
        }
    }
    s = block_sum(s);
    float vs = 0.f;
    float *val = DP(float, 4) + (size_t)j * k;
    int *idx = DP(int, 5) + (size_t)j * k;
    for (int s2 = 0; s2 < k; ++s2) {
        float bv = -1.f;
        int bi = 0x7fffffff;
        for (int u = 0; u < 4; ++u) {
            int x = threadIdx.x + 256 * u;
            if (p[u] >= 0.f && (p[u] > bv || (p[u] == bv && x < bi))) {
                bv = p[u];
                bi = x;
            }
        }
        block_argmax(bv, bi, &bv, &bi);
        if (map != NULL) {
            bv = expf(l[bi] - m);   /* the plain value of the choice */
        }
        if (threadIdx.x == 0) {
            idx[s2] = bi;
            val[s2] = bv / s;
        }
        vs += bv / s;
        if (bi % 256 == (int)threadIdx.x) {
            p[bi / 256] = -1.f;
        }
    }
    __syncthreads();
    if (swap > 0.f) {
        if (sel != NULL && threadIdx.x < (unsigned)k) {
            sel[(size_t)j * k + threadIdx.x] = idx[threadIdx.x];
        }
        for (int s2 = k - 1; s2 >= 0; --s2) {
            const int c = idx[s2];
            const float cv = val[s2];
            if (map[c] >= 0) {
                continue;
            }
            float bv = -1.f;
            int bi = 0x7fffffff;
            for (int u = 0; u < 4; ++u) {
                int x = threadIdx.x + 256 * u;
                if (x < E && p[u] >= 0.f && map[x] >= 0 && (p[u] > bv || (p[u] == bv && x < bi))) {
                    bv = p[u];
                    bi = x;
                }
            }
            block_argmax(bv, bi, &bv, &bi);
            if (bv < 0.f) {
                break;
            }
            const float cp = expf(l[bi] - m) / s;
            if (cp < swap * cv) {
                break;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                idx[s2] = bi;
                val[s2] = cp;
            }
            vs += cp - cv;
            if (bi % 256 == (int)threadIdx.x) {
                p[bi / 256] = -1.f;
            }
            __syncthreads();
        }
    }
    if (threadIdx.x < (unsigned)k) {
        val[threadIdx.x] /= vs;
    }
    if (r2 != NULL) {
        __syncthreads();
        if (threadIdx.x == 0) {
            hot_split_body(r2, e);
        }
    }
}

/* GP_ATTN_PREP: qg, kk, vv, qn, kn, cos, sin, K, V, hs, pos, t, nq, nk, hd, rot, eps,
 * scale, qout, gate, kout (gp_attn_prep_body of the CPU; with a null K the
 * key goes to kout, for GP_KV_WRITE; with a null gate, LFM2, qg holds the
 * query alone, hd values a head). One block of hd threads
 * for each (token, head). */
__global__ void k_attn_prep(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float ys[1024];
    int nq = DI(12), nk = DI(13), hd = DI(14), rot = DI(15), half = rot / 2;
    int j = blockIdx.x / (nq + nk), h = blockIdx.x % (nq + nk), d = threadIdx.x;
    int64_t hs = di(r, e, 9), pos = di(r, e, 10);
    const float *src, *w;
    float *dst, sc;
    if (h < nq) {
        if (DP(float, 19) != NULL) {
            src = DP(const float, 0) + ((size_t)j * nq + h) * 2 * hd;
            DP(float, 19)[((size_t)j * nq + h) * hd + d] = src[hd + d];
        } else {
            src = DP(const float, 0) + ((size_t)j * nq + h) * hd;
        }
        dst = DP(float, 18) + ((size_t)j * nq + h) * hd;
        w = DP(const float, 3);
        sc = df(r, e, 17);
    } else {
        int kh = h - nq;
        src = DP(const float, 1) + ((size_t)j * nk + kh) * hd;
        if (DP(float, 7) == NULL) {
            dst = DP(float, 20) + ((size_t)j * nk + kh) * hd;
        } else {
            size_t rs = kv_rs((size_t)hs, nk, hd);
            dst = DP(float, 7) + (size_t)kh * hs + (size_t)(pos + j) * rs;
            DP(float, 8)[(size_t)kh * hs + (size_t)(pos + j) * rs + d] =
                DP(const float, 2)[((size_t)j * nk + kh) * hd + d];
        }
        w = DP(const float, 4);
        sc = 1.f;
    }
    float v = src[d];
    /* no norm weights (Nemotron): the rows as they are (w is null for the whole block) */
    float y = v;
    if (w != NULL) {
        float ss = block_sum(v * v);
        y = v * (1.f / sqrtf(ss / (float)hd + df(r, e, 16))) * w[d];
    }
    ys[d] = y;
    __syncthreads();
    const float *c = DP(const float, 5) + (size_t)j * rot, *sn = DP(const float, 6) + (size_t)j * rot;
    if (d < half) {
        dst[d] = (ys[d] * c[d] - ys[d + half] * sn[d]) * sc;
    } else if (d < rot) {
        dst[d] = (ys[d] * c[d] + ys[d - half] * sn[d]) * sc;
    } else {
        dst[d] = y * sc;
    }
}

/* GP_GDN (gdn_body of backends/cpu/csrc/common/deltanet.c): qkv, conv, conv_w, kernel, z, a, b,
 * A_log, dt_bias, norm_w, S, out, scratch, t, k_heads, v_heads, k_dim,
 * v_dim, eps, log, flags (1: tiled heads; 2: a sigmoid gate of the norm), nreal.
 *
 * Only the first nreal tokens change the state (0: all t); the other rows
 * pad a group to the size of its program. With a log (an MTP verify group)
 * conv and S do not change: the log gets, for each token, the input of the
 * convolution, and for each head k, the delta, and the decay, as on the CPU.
 * rc_gpu_gdn_commit applies the first n tokens.
 *
 * k_gdn_conv: the convolution of each channel, the tokens in order, into the
 * scratch; the last inputs go back to conv. */
__device__ __forceinline__ int gdn_nreal(const gp_rec *r, const int64_t *e)
{
    int t = DI(13), n = r->tag[21] == GP_T_NONE ? 0 : DI(21);
    return n > 0 && n < t ? n : t;
}

__global__ void k_gdn_conv(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int kernel = DI(3), t = DI(13);
    int cd = 2 * DI(14) * DI(16) + DI(15) * DI(17);
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cd) {
        return;
    }
    const float *qkv = DP(const float, 0);
    float *conv = DP(float, 1), *cv = DP(float, 12), *log = DP(float, 19);
    const float *w = DP(const float, 2) + (size_t)c * kernel;
    /* (flag 4, KDA: a decay for each channel in the log, 3 k_dim values a head) */
    size_t lrow = (size_t)cd + (size_t)DI(15) * ((DI(20) & 4) ? 3 * DI(16) : DI(16) + DI(17) + 1);
    float hist[8];
    for (int j = 0; j < kernel - 1; ++j) {
        hist[j] = conv[(size_t)j * cd + c];
    }
    t = gdn_nreal(r, e);
    for (int i = 0; i < t; ++i) {
        if (log != NULL) {
            log[(size_t)i * lrow + c] = qkv[(size_t)i * cd + c];
        }
        float xin = qkv[(size_t)i * cd + c];
        float v = w[kernel - 1] * xin;
        for (int j = 0; j < kernel - 1; ++j) {
            v += w[j] * hist[j];
        }
        cv[(size_t)i * cd + c] = qw_silu(v);
        for (int j = 0; j < kernel - 2; ++j) {
            hist[j] = hist[j + 1];
        }
        hist[kernel - 2] = xin;
    }
    if (log == NULL) {
        for (int j = 0; j < kernel - 1; ++j) {
            conv[(size_t)j * cd + c] = hist[j];
        }
    }
}

/* k_gdn_heads: one block for each value head; thread e keeps column e of the
 * state of the head (KD values) in registers. KD = k_dim = v_dim = the
 * threads of the block. */
template <int KD>
__global__ void __launch_bounds__(KD) k_gdn_heads(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float qsh[KD], ksh[KD], dsh[KD];
    int kh = DI(14), vh = DI(15), t = gdn_nreal(r, e), flags = DI(20), tiled = flags & 1;
    int ch = (flags & 4) != 0;          /* KDA: a decay for each channel (see k_gdn_scan) */
    int kd = kh * KD, vd = vh * KD, cd = 2 * kd + vd;
    float *log = DP(float, 19);
    const int lh = ch ? 3 * KD : 2 * KD + 1;     /* the log of a head: k, delta, the decay(s) */
    size_t lrow = (size_t)cd + (size_t)vh * lh;
    int hv = blockIdx.x, c = threadIdx.x;
    int hk = tiled ? hv % kh : hv / (vh / kh);
    float *Sh = DP(float, 10) + (size_t)hv * KD * KD;
    float S[KD];
    #pragma unroll
    for (int d = 0; d < KD; ++d) {
        S[d] = Sh[(size_t)d * KD + c];
    }
    const float *cv = DP(const float, 12);
    float Aexp = expf(DP(const float, 7)[hv]), dtb = ch ? 0.f : DP(const float, 8)[hv];
    float eps = df(r, e, 18);
    for (int i = 0; i < t; ++i) {
        const float *row = cv + (size_t)i * cd;
        float qv = row[hk * KD + c], kv = row[kd + hk * KD + c];
        float nq = block_sum(qv * qv), nk = block_sum(kv * kv);
        qsh[c] = qv * (1.f / sqrtf(nq + 1e-6f) / sqrtf((float)KD));
        ksh[c] = kv * (1.f / sqrtf(nk + 1e-6f));
        float decay = 1.f;
        if (ch) {
            float g = Aexp * (DP(const float, 5)[(size_t)i * vd + (size_t)hv * KD + c] +
                              DP(const float, 8)[(size_t)hv * KD + c]);
            dsh[c] = expf(-5.f / (1.f + expf(-g)));
        } else {
            float av = DP(const float, 5)[(size_t)i * vh + hv] + dtb;
            float sp = av > 20.f ? av : log1pf(expf(av));
            decay = expf(-Aexp * sp);
        }
        __syncthreads();
        float beta = 1.f / (1.f + expf(-DP(const float, 6)[(size_t)i * vh + hv]));
        /* The updates of S use explicit roundings, as rc_gpu_gdn_commit does, so
         * a commit gives the state of plain steps. */
        float s1 = 0.f;
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            S[d] = __fmul_rn(S[d], ch ? dsh[d] : decay);
            s1 += ksh[d] * S[d];
        }
        float delta = (row[2 * kd + hv * KD + c] - s1) * beta;
        if (log != NULL) {
            float *lg = log + (size_t)i * lrow + cd + (size_t)hv * lh;
            lg[c] = ksh[c];
            lg[KD + c] = delta;
            if (ch) {
                lg[2 * KD + c] = dsh[c];
            } else if (c == 0) {
                lg[2 * KD] = decay;
            }
        }
        float o = 0.f;
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            S[d] = __fmaf_rn(ksh[d], delta, S[d]);
            o += qsh[d] * S[d];
        }
        float ss = block_sum(o * o);
        float inv = 1.f / sqrtf(ss / (float)KD + eps);
        size_t oi = (size_t)i * vd + (size_t)hv * KD + c;
        float zv = DP(const float, 4)[oi];
        float zg = (flags & 2) ? 1.f / (1.f + expf(-zv)) : qw_silu(zv);
        DP(float, 11)[oi] = o * inv * DP(const float, 9)[c] * zg;
        __syncthreads();
    }
    /* The padding rows (after nreal) get zeros: the products after read
     * them, and their buffer can hold anything (the packed temporaries of
     * ir/alloc.py; a NaN there made a router choose no expert). */
    for (int i = t; i < DI(13); ++i) {
        DP(float, 11)[(size_t)i * vd + (size_t)hv * KD + c] = 0.f;
    }
    if (log == NULL) {
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            Sh[(size_t)d * KD + c] = S[d];
        }
    }
}
