/* ---------- KQ_BF12 steps and small groups on the tensor cores ----------
 * The CUDA-core kernels pay an FMA and a load of x for each value and each
 * token (2 to 4 tokens: 1.3-1.8 times bfloat16). Here mma m16n8k8 tf32 takes
 * 8 tokens at once: the cost of a group is the decode of the weights, as for
 * one token. The bfloat16 values of w are exact in tf32; x goes in as two
 * tf32 parts (hi = x with the low 13 bits cleared, lo = x - hi: about 22
 * bits of x, two instructions). The tokens past t are zeros, and a column
 * of the mma does not depend on the others: a token alone and in a group
 * get the same bits.
 *
 * A block: KW warps and a tile of 16 rows (bt_cls); warp w takes the steps
 * of 64 columns w, w + KW, ...; the warps add their sums in shared memory in
 * a fixed order (by the shape alone). In a step thread (g, c) takes 16 columns of each of its
 * rows, 16c .. 16c + 15 (one half of a BF12 group: one load of the lo
 * bytes, one of the gap bytes, the exponent), as the k slots {c, c + 4} of
 * the 8 k-steps (k-step q: columns 16c + 2q, 16c + 2q + 1); the thread
 * (g', c) of x takes the same columns of token g'. GP_KQ_LINEAR (m < 0) or
 * the matrices of GP_KQ_MULTI of bt_shape (the blocks one after another). */
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

/* The 16 values of a BF12 row from column c0 (16 | c0) as 8 words of two
 * bfloat16 values (kq_bf12_dec16), or zeros past the row. */
__device__ __forceinline__ void bt_row16(const uint8_t *w, int cols, int c0, uint32_t *p)
{
    if (c0 >= cols) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            p[i] = 0;
        }
        return;
    }
    int g = c0 >> 5;
    kq_bf12_dec16(*(const uint4 *)(w + c0), *(const uint4 *)(w + cols + 16 * g),
                  w[cols + cols / 2 + g], (c0 >> 4) & 1, p);
}

/* A hot day (the GPU and the CPUs throttled), 1 / 2 / 4 tokens in us,
 * k_kq_bf12_tc against the CUDA-core kernels (bfloat16 in brackets):
 * 10240 x 2560 102 103 106 / 102 109 167 (126 126 129); 2560 x 6144 67 69
 * 71 / 70 86 129 (82 83 86); 320 x 10240 28 27 28 / 26 35 45 (28 29 32);
 * 10240 x 320 27 27 28 / 24 28 38 (35 30 33); the head 2126 2137 2157 /
 * 2136 2123 2910 (2735 2749 2767). One kernel for 1 to 8 tokens keeps the
 * bits of a step in a verify group (the CUDA-core kernel for a token alone
 * would save 2-3 us on the two small shapes only).
 *
 * The kernels by the shape (bt_cls): 0 the most, KW 8 warps a block, a
 * tile of 16 rows (two tiles, x shared by 32 rows, spilled and had too few
 * blocks: 10240 x 2560 129 us in place of 102, the head 3.06 ms in place of
 * 2.13); 1 fewer than BT_MIN_ROWS rows: also BT_KP blocks for each tile,
 * each a part of the columns (320 x 10240 had 20 blocks: 53 us for 1 to 4
 * tokens, the CUDA-core kernels 27 to 45); 2 at most 512 columns: KW 4 (8
 * steps or fewer of 64 columns: idle warps). */
#define BT_KP 8
__host__ __device__ __forceinline__ int bt_cls(int rows, int cols)
{
    return rows < BT_MIN_ROWS ? 1 : cols <= 512 ? 2 : 0;
}

/* A block of class 1 adds its sums of its part of the columns (part
 * blockIdx.y) to part[tile][y][128]; the last block of a tile (cnt) adds the
 * parts in their order (the same bits whichever block is last), writes out,
 * and sets cnt back to 0. */
template <int KW>
__global__ void __launch_bounds__(32 * KW, 3) k_kq_bf12_tc(const gp_rec *r, const int64_t *e, int multi,
                                                        int cls, float *part, int *cnt)
{
    PDL_START();
    __shared__ float red[KW][4][32];
    __shared__ int last;
    /* the matrix of this block */
    int blk = blockIdx.x, rows, cols, t;
    const uint8_t *w;
    const float *x;
    float *out;
    if (!multi) {
        rows = DI(6), cols = DI(7), t = DI(8);
        w = DP(const uint8_t, 4), x = DP(const float, 3), out = DP(float, 9);
    } else {
        int n = DI(3), m = 0;
        cols = DI(1), t = DI(2);
        for (;; ++m) {
            if (m >= n) {
                return;
            }
            int rm = DI(6 + 4 * m);
            if (!bt_shape(DI(5 + 4 * m), rm, cols) || bt_cls(rm, cols) != cls) {
                continue;
            }
            int nb = (rm + 15) / 16;
            if (blk < nb) {
                break;
            }
            blk -= nb;
        }
        rows = DI(6 + 4 * m);
        w = DP(const uint8_t, 4 + 4 * m), x = DP(const float, 0), out = DP(float, 7 + 4 * m);
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    int row0 = blk * 16, kp = gridDim.y;
    size_t rb = kq_row_bytes(KQ_BF12, cols);
    const uint8_t *w0 = w + (size_t)min(row0 + g, rows - 1) * rb;
    const uint8_t *w8 = w + (size_t)min(row0 + g + 8, rows - 1) * rb;
    const float *xg = g < t ? x + (size_t)g * cols : NULL;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int st = blockIdx.y * KW + warp; st * 64 < cols; st += KW * kp) {
        int c0 = st * 64 + 16 * c;
        float xr[16];
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float4 v = xg != NULL && c0 < cols ? *(const float4 *)(xg + c0 + 4 * u)
                                               : make_float4(0.f, 0.f, 0.f, 0.f);
            xr[4 * u] = v.x, xr[4 * u + 1] = v.y, xr[4 * u + 2] = v.z, xr[4 * u + 3] = v.w;
        }
        uint32_t p0[8], p8[8];
        bt_row16(w0, cols, c0, p0);
        bt_row16(w8, cols, c0, p8);
        /* the sums of a step in the mma, then added in float32 (the mma
         * alone: 1.2e-5 of the largest output for rows of 8192 values
         * against float64; this way 6e-7, as the CUDA-core kernels) */
        float sa[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            /* x as tf32 hi + lo; k-step q: the values 2q (slot c) and 2q + 1 (slot c + 4) */
            uint32_t h0 = __float_as_uint(xr[2 * q]) & 0xffffe000u;
            uint32_t h1 = __float_as_uint(xr[2 * q + 1]) & 0xffffe000u;
            uint32_t bh[2] = {h0, h1};
            uint32_t bl[2] = {__float_as_uint(xr[2 * q] - __uint_as_float(h0)),
                              __float_as_uint(xr[2 * q + 1] - __uint_as_float(h1))};
            uint32_t a[4] = {p0[q] << 16, p8[q] << 16, p0[q] & 0xffff0000u, p8[q] & 0xffff0000u};
            mma1688_tf32(sa, a, bh);
            mma1688_tf32(sa, a, bl);
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            acc[k] += sa[k];
        }
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        red[warp][k][lane] = acc[k];
    }
    __syncthreads();
    if (warp != 0) {
        return;
    }
    float v[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        v[k] = red[0][k][lane];
        #pragma unroll
        for (int ww = 1; ww < KW; ++ww) {
            v[k] += red[ww][k][lane];
        }
    }
    if (kp > 1) {
        float *pt = part + (size_t)blockIdx.x * kp * 128;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            pt[(size_t)blockIdx.y * 128 + 4 * lane + k] = v[k];
        }
        __threadfence();
        if (lane == 0) {
            last = atomicAdd(cnt + blockIdx.x, 1) == kp - 1;
        }
        __syncwarp();
        if (!last) {
            return;
        }
        __threadfence();
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            v[k] = 0.f;
            for (int y = 0; y < kp; ++y) {
                v[k] += __ldcg(pt + (size_t)y * 128 + 4 * lane + k);
            }
        }
        if (lane == 0) {
            cnt[blockIdx.x] = 0;
        }
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        /* c0: (row g, token 2c), c1: (g, 2c + 1), c2: (g + 8, 2c), c3: (g + 8, 2c + 1) */
        int row = row0 + g + (k >= 2 ? 8 : 0), tok = 2 * c + (k & 1);
        if (row < rows && tok < t) {
            out[(size_t)tok * rows + row] = v[k];
        }
    }
}

/* Launch k_kq_bf12_tc for the matrices of class cls (blocks: their tiles). */
static void bt_launch(const gp_rec *dr, const int64_t *denv, int multi, int cls, int64_t blocks,
                      float *part, int *cnt, cudaStream_t s)
{
    if (cls == 1) {
        k_kq_bf12_tc<8><<<dim3((unsigned)blocks, BT_KP), 256, 0, s>>>(dr, denv, multi, 1, part, cnt);
    } else if (cls == 2) {
        k_kq_bf12_tc<4><<<(unsigned)blocks, 128, 0, s>>>(dr, denv, multi, 2, part, cnt);
    } else {
        k_kq_bf12_tc<8><<<(unsigned)blocks, 256, 0, s>>>(dr, denv, multi, 0, part, cnt);
    }
}

/* RECLANK_GPU_BF12_TC=0: the CUDA-core kernels for the steps and small
 * groups of BF12 (k_kq_linear_bf12, k_kq_multi_bf16), for a test. */
static int bt_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_BF12_TC");
        on = !(v && v[0] == '0');
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}

/* GP_KQ_MULTI with KQ_BF16 (or KQ_BF12) matrices for 2 to NT tokens (as k_kq_multi); a
 * matrix of another type (the float32 alpha and beta of a DeltaNet layer)
 * takes kq_row for each token, as k_kq_multi. */
template <int NT>
__global__ void k_kq_multi_bf16(const gp_rec *r, const int64_t *e, int skip_bt = 0)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int n = DI(3), cols = DI(1), t = DI(2), m = 0;
    while (m < n && row >= DI(6 + 4 * m)) {
        row -= DI(6 + 4 * m);
        ++m;
    }
    if (m >= n) {
        return;
    }
    int rows = DI(6 + 4 * m), type = DI(5 + 4 * m);
    if (skip_bt && bt_shape(type, rows, cols)) {
        return;         /* k_kq_bf12_tc */
    }
    if (!kq_bf_type(type)) {
        const uint8_t *w = DP(const uint8_t, 4 + 4 * m) + (size_t)row * kq_row_bytes(type, cols);
        for (int j = 0; j < t; ++j) {
            float v = kq_row(type, w, DP(const float, 0) + (size_t)j * cols, cols);
            if (threadIdx.x % 32 == 0) {
                DP(float, 7 + 4 * m)[(size_t)j * rows + row] = v;
            }
        }
        return;
    }
    float res[NT];
    kq_row_bf16_nt<NT>(type, DP(const uint8_t, 4 + 4 * m) + (size_t)row * kq_row_bytes(type, cols),
                       DP(const float, 0), cols, t, 0, 1, res);
    if (threadIdx.x % 32 == 0) {
        #pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                DP(float, 7 + 4 * m)[(size_t)j * rows + row] = res[j];
            }
        }
    }
}
