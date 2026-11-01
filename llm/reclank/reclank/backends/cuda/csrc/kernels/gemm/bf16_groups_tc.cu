/* ---------- bfloat16 products of large groups on the tensor cores ----------
 * RECLANK_DENSE=bf16 (Qwen3.8 with its dense matrices as they are): the
 * products of a prompt group ran in k_kq_gemm (float32, no tensor cores):
 * 4.6 s of 20 s for 8192 tokens. Here x goes to bfloat16 (k_to_bf16, round
 * to nearest even; bfloat16 has the range of float32, so no clamp), and
 * mma.sync m16n8k16 bf16 with float32 sums takes the bfloat16 rows of w as
 * they are. The tile is that of k_enc_gemm_tc: 128 tokens by 128 rows, 8
 * warps, steps of BHK columns in two buffers.
 *
 * out[(orow0 + m) * ldo + n] = sum_c x[(xrow0 + m) * ldx + c] w[n * cols + c]
 * for m < t, n < rows; xrow0 and orow0 (or null: 0) are read on the device
 * (the rows of the shared expert in the pairs of a group MoE). */
__device__ __forceinline__ void mma16816_bf16(float *c, const uint32_t *a, const uint32_t *b)
{
#if RC_SM80
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
#else
    __trap();
#endif
}

/* rows of n values from x (row stride ldx, from row xrow0 or its device
 * value) to bfloat16 rows of n values in y */
__device__ __forceinline__ uint16_t f2bf_rn(float v)
{
    uint32_t u = __float_as_uint(v);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

/* ylo (or null): the rest, x - hi, as bfloat16 too: hi + lo keeps about 16
 * bits of x (the X2 form of k_gemm_bf16_tc). */
__global__ void k_to_bf16(const float *x, size_t ldx, const int *xrow0, uint16_t *y, int rows, int n,
                          uint16_t *ylo = NULL)
{
    PDL_START();
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * n) {
        return;
    }
    size_t m = i / n, c = i % n;
    size_t r0 = xrow0 != NULL ? (size_t)*xrow0 : 0;
    float v = x[(r0 + m) * ldx + c];
    uint16_t h = f2bf_rn(v);
    y[i] = h;
    if (ylo != NULL) {
        ylo[i] = f2bf_rn(v - __uint_as_float((uint32_t)h << 16));
    }
}

/* X2: x = hi + lo, two bfloat16 planes (xlo: lo), two products for each
 * fragment of w, in the same float32 sums: the bits of x of a float32
 * product but for about 16 of them (in place of 8). */
#define BF_SMEM(X2) ((size_t)2 * ((1 + (X2)) * T2M + T2N) * (BHK + 8) * 2)
/* W12: w holds KQ_BF12 rows (backends/cpu/csrc/kernels/kq; KQ_BF12 below): a step of BHK
 * = 32 columns is one group of each row. The step copies the bytes of the
 * group as they are (the sign and mantissa bytes, 32 a row; the gap nibbles,
 * 16; the word of the group exponent) into shared memory in place of bs, and
 * each thread makes the bfloat16 bits of its fragments of w from them
 * (bf12_frag): no copy of w in bfloat16, the bits of k_bf12_to_bf16. */
#define BF12_LO 48          /* bytes of a row of the lo bytes in shared memory (32 + 16: no bank conflict) */
/* Two values in the two 16-bit lanes of a word: the bytes b of lo and the
 * gaps of hi (>> sh) spread to the lanes, the exponent E - gap from E + 256
 * (no borrow out of a lane), and the zero code (gap 15, b 0x80) by
 * __vcmpeq2. */
__device__ __forceinline__ uint32_t bf12_frag(uint32_t lo, uint32_t hi, int sh, uint32_t E2)
{
    uint32_t b = __byte_perm(lo, 0, 0x4140);
    uint32_t gap = (__byte_perm(hi, 0, 0x4140) >> sh) & 0x000f000fu;
    uint32_t ex = (E2 - gap) & 0x00ff00ffu;
    uint32_t v = (b & 0x00800080u) << 8 | ex << 7 | (b & 0x007f007fu);
    return v & ~__vcmpeq2(b | gap << 8, 0x0f800f80u);
}

/* NF: blockIdx.x is the tile of rows of w (else of tokens), so the blocks that
 * read the same tokens run together and x comes from DRAM one time: for many
 * more tokens than rows (MuQ's convs, 32000 x 4608 into 512: 8.45 -> 7.28 ms;
 * the bits are the same). */
/* GX: the rows of x are gathered (ENC_GATHER_MM, an im2col without its
 * copy): row m of x is gnb blocks of cols / gnb values, block j row gidx[m gnb
 * + j] of the bfloat16 rows x (and xlo) of cols / gnb values, or zeros where it
 * is < 0; cols / gnb a multiple of BHK, so a step reads one block. */
/* F16: x and w are float16 values (mma16816 in place of mma16816_bf16: the
 * checkpoint's own float16 matrices of ENC_LINEAR wbf 2), each product from
 * zero; the rest as is. */
template <int F16>
__device__ __forceinline__ void mma_x(float *c, const uint32_t *a, const uint32_t *b)
{
    if (F16) {
        /* from zero, then added rounded to nearest: the tensor cores truncate as
         * they add into c, a bias of about -1e-7 a product (Whisper's fc2) */
        float z[4] = {0.f, 0.f, 0.f, 0.f};
        mma16816(z, a, b);
        c[0] += z[0];
        c[1] += z[1];
        c[2] += z[2];
        c[3] += z[3];
    } else {
        mma16816_bf16(c, a, b);
    }
}

template <int X2, int W12 = 0, int NF = 0, int GX = 0, int F16 = 0>
__global__ void __launch_bounds__(256) k_gemm_bf16_tc(const uint16_t *x, const uint16_t *w, float *out,
                                                      size_t ldo, const int *orow0, int t, int rows,
                                                      int cols, const uint16_t *xlo, const int *gidx = NULL,
                                                      int gnb = 1, const float *xinv = NULL)
{
    PDL_START();
    extern __shared__ __align__(16) uint16_t bfsm[];
    uint16_t (*as_)[T2M][BHK + 8] = (uint16_t (*)[T2M][BHK + 8])bfsm;
    uint16_t (*bs)[T2N][BHK + 8] = (uint16_t (*)[T2N][BHK + 8])(bfsm + 2 * T2M * (BHK + 8));
    uint16_t (*al_)[T2M][BHK + 8] = (uint16_t (*)[T2M][BHK + 8])(bfsm + 2 * (T2M + T2N) * (BHK + 8));
    /* W12: in the room of bs, for each of the two buffers: lo [T2N][BF12_LO],
     * hi [T2N][16], the exponent words [T2N] (2 x 8704 bytes of 2 x 10240) */
    const size_t w12_b = (size_t)T2N * (BF12_LO + 16 + 4);
    uint8_t *w12s = (uint8_t *)bs;
    const uint8_t *w8 = (const uint8_t *)w;
    const size_t rb12 = ((size_t)cols + cols / 2 + cols / 32 + 15) / 16 * 16;
    int m0 = (NF ? blockIdx.y : blockIdx.x) * T2M, n0 = (NF ? blockIdx.x : blockIdx.y) * T2N;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int wm = (warp / 4) * 64, wn = (warp % 4) * 32;
    int g = lane / 4, c = lane % 4;
    int steps = (cols + BHK - 1) / BHK;
    float acc[4][4][4];
    /* F16: x's blocks of BHK columns are scaled (k_to_f16_blocks), so the sums of
     * each step go into tot times 1 / the block's scale (xinv), rounded to
     * nearest: the float32 sums of the float16 products truncate, which over
     * thousands of columns drifts */
    float tot[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
            tot[i][j][0] = tot[i][j][1] = tot[i][j][2] = tot[i][j][3] = 0.f;
        }
    }
    const int gc = cols / gnb;      /* GX: the values of a block */
    auto load = [&](int st, int b) {
        int k0 = st * BHK;
        for (int e = threadIdx.x; e < T2M * BHK / 8; e += blockDim.x) {
            int m = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int q = min(m0 + m, t - 1);
            if (GX) {
                int blk = k0 / gc, src = gidx[(size_t)q * gnb + blk];
                if (src >= 0) {
                    size_t o = (size_t)src * gc + (k0 - blk * gc) + kk;
                    cp_async16(&as_[b][m][kk], x + o);
                    if (X2) {
                        cp_async16(&al_[b][m][kk], xlo + o);
                    }
                } else {
                    *(uint4 *)&as_[b][m][kk] = make_uint4(0, 0, 0, 0);
                    if (X2) {
                        *(uint4 *)&al_[b][m][kk] = make_uint4(0, 0, 0, 0);
                    }
                }
                continue;
            }
            if (k0 + kk < cols) {
                cp_async16(&as_[b][m][kk], x + (size_t)q * cols + k0 + kk);
                if (X2) {
                    cp_async16(&al_[b][m][kk], xlo + (size_t)q * cols + k0 + kk);
                }
            } else {
                *(uint4 *)&as_[b][m][kk] = make_uint4(0, 0, 0, 0);
                if (X2) {
                    *(uint4 *)&al_[b][m][kk] = make_uint4(0, 0, 0, 0);
                }
            }
        }
        if (W12) {
            /* the group st of each row: 2 x 16 bytes of lo, 16 of hi, the
             * aligned word of its exponent (cols % 32 == 0) */
            uint8_t *lo = w12s + b * w12_b, *hi = lo + T2N * BF12_LO, *ex = hi + T2N * 16;
            for (int e = threadIdx.x; e < T2N * 4; e += blockDim.x) {
                int n = e / 4, part = e % 4;
                const uint8_t *wr = w8 + (size_t)min(n0 + n, rows - 1) * rb12;
                if (part < 2) {
                    cp_async16(lo + n * BF12_LO + 16 * part, wr + k0 + 16 * part);
                } else if (part == 2) {
                    cp_async16(hi + n * 16, wr + cols + k0 / 2);
                } else {
                    cp_async4(ex + n * 4, wr + cols + cols / 2 + (st & ~3));
                }
            }
            cp_async_commit();
            return;
        }
        for (int e = threadIdx.x; e < T2N * BHK / 8; e += blockDim.x) {
            int n = e / (BHK / 8), kk = (e % (BHK / 8)) * 8;
            int row = min(n0 + n, rows - 1);
            if (k0 + kk < cols) {
                cp_async16(&bs[b][n][kk], w + (size_t)row * cols + k0 + kk);
            } else {
                *(uint4 *)&bs[b][n][kk] = make_uint4(0, 0, 0, 0);
            }
        }
        cp_async_commit();
    };
    load(0, 0);
    for (int st = 0; st < steps; ++st) {
        int b = st & 1;
        if (st + 1 < steps) {
            load(st + 1, b ^ 1);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();
        /* W12: the fragments of w of both halves of the step (columns 2c,
         * 2c + 1, 2c + 8, 2c + 9 of each half: the gaps of the second half
         * are the high nibbles of the same hi bytes) */
        uint32_t bf12[2][4][2];
        if (W12) {
            const uint8_t *lo = w12s + b * w12_b, *hi = lo + T2N * BF12_LO, *ex = hi + T2N * 16;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int n = wn + j * 8 + g;
                const uint8_t *lr = lo + n * BF12_LO + 2 * c, *hr = hi + n * 16 + 2 * c;
                uint32_t E = (ex[n * 4 + (st & 3)] | 256u) * 0x00010001u;     /* E + 256 in each lane */
                uint32_t h0 = *(const uint16_t *)hr, h1 = *(const uint16_t *)(hr + 8);
                bf12[0][j][0] = bf12_frag(*(const uint16_t *)lr, h0, 0, E);
                bf12[0][j][1] = bf12_frag(*(const uint16_t *)(lr + 8), h1, 0, E);
                bf12[1][j][0] = bf12_frag(*(const uint16_t *)(lr + 16), h0, 4, E);
                bf12[1][j][1] = bf12_frag(*(const uint16_t *)(lr + 24), h1, 4, E);
            }
        }
        #pragma unroll
        for (int ks = 0; ks < BHK / 16; ++ks) {
            int kk = ks * 16;
            uint32_t bf[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (W12) {
                    bf[j][0] = bf12[ks][j][0];
                    bf[j][1] = bf12[ks][j][1];
                    continue;
                }
                const uint16_t *br = &bs[b][wn + j * 8 + g][kk + 2 * c];
                bf[j][0] = *(const uint32_t *)br;
                bf[j][1] = *(const uint32_t *)(br + 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int r0 = wm + i * 16;
                uint32_t a[4];
                a[0] = *(const uint32_t *)&as_[b][r0 + g][kk + 2 * c];
                a[1] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 2 * c];
                a[2] = *(const uint32_t *)&as_[b][r0 + g][kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&as_[b][r0 + g + 8][kk + 2 * c + 8];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma_x<F16>(acc[i][j], a, bf[j]);
                }
                if (X2) {
                    a[0] = *(const uint32_t *)&al_[b][r0 + g][kk + 2 * c];
                    a[1] = *(const uint32_t *)&al_[b][r0 + g + 8][kk + 2 * c];
                    a[2] = *(const uint32_t *)&al_[b][r0 + g][kk + 2 * c + 8];
                    a[3] = *(const uint32_t *)&al_[b][r0 + g + 8][kk + 2 * c + 8];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        mma_x<F16>(acc[i][j], a, bf[j]);
                    }
                }
            }
        }
        if (F16) {
            /* the step's sums of each row, times 1 / the scale of the row's block of x */
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int ra = min(m0 + wm + i * 16 + g, t - 1), rb = min(m0 + wm + i * 16 + g + 8, t - 1);
                float ia = __ldg(&xinv[(size_t)ra * steps + st]), ib = __ldg(&xinv[(size_t)rb * steps + st]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    tot[i][j][0] += acc[i][j][0] * ia;
                    tot[i][j][1] += acc[i][j][1] * ia;
                    tot[i][j][2] += acc[i][j][2] * ib;
                    tot[i][j][3] += acc[i][j][3] * ib;
                    acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
                }
            }
        }
        __syncthreads();
    }
    if (F16) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    acc[i][j][u] = tot[i][j][u];
                }
            }
        }
    }
    size_t o0 = orow0 != NULL ? (size_t)*orow0 : 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            int n = n0 + wn + j * 8 + 2 * c;
            int m = m0 + wm + i * 16 + g;
            if (m < t) {
                if (n < rows) out[(o0 + m) * ldo + n] = acc[i][j][0];
                if (n + 1 < rows) out[(o0 + m) * ldo + n + 1] = acc[i][j][1];
            }
            if (m + 8 < t) {
                if (n < rows) out[(o0 + m + 8) * ldo + n] = acc[i][j][2];
                if (n + 1 < rows) out[(o0 + m + 8) * ldo + n + 1] = acc[i][j][3];
            }
        }
    }
}

/* RECLANK_GPU_BF16_TC: 1 (the default) x as two bfloat16 planes (about 16
 * bits); 8: one plane (8 bits of x, about 1.8 times as fast); 0: float32
 * products (k_kq_gemm). */
static int bf16_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_BF16_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}

/* x (t rows of cols float32 at x, stride ldx, from row xrow0 on the device or
 * 0) times the bfloat16 rows of w into out (stride ldo, from row orow0):
 * k_to_bf16 into the scratch xb (and xb + t cols for the lo plane), then
 * k_gemm_bf16_tc. */
/* x1: x is bfloat16 values already (the low plane would be zeros: one product a fragment, the same sums);
 * xready: xb holds its bfloat16 rows already (an earlier call on the same x: no k_to_bf16) */
static void gg_gemm_bf16(const float *x, size_t ldx, const int *xrow0, const uint16_t *w, float *out,
                         size_t ldo, const int *orow0, int t, int rows, int cols, uint16_t *xb,
                         cudaStream_t s, int w12 = 0, int x1 = 0, int xready = 0)
{
    static int attr = 0;
    if (!attr) {
        cudaFuncSetAttribute(k_gemm_bf16_tc<1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        cudaFuncSetAttribute(k_gemm_bf16_tc<1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        cudaFuncSetAttribute(k_gemm_bf16_tc<1, 1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        attr = 1;
    }
    int x2 = bf16_tc_on() != 8 && !x1;
    uint16_t *xl = x2 ? xb + (size_t)t * cols : NULL;
    if (!xready) {
        k_to_bf16<<<(unsigned)(((int64_t)t * cols + 255) / 256), 256, 0, s>>>(x, ldx, xrow0, xb, t, cols, xl);
    }
    dim3 grid((unsigned)((t + T2M - 1) / T2M), (unsigned)((rows + T2N - 1) / T2N));
    /* many more tokens than rows (x far past the L2, w in it): the tiles of
     * rows of w fastest (NF); at 1500 x 1024 the order of tokens is faster */
    if (x2 && w12 && t >= 16 * rows) {
        k_gemm_bf16_tc<1, 1, 1><<<dim3(grid.y, grid.x), 256, BF_SMEM(1), s>>>(xb, w, out, ldo, orow0, t, rows,
                                                                            cols, xl);
    } else if (x2 && w12) {
        k_gemm_bf16_tc<1, 1><<<grid, 256, BF_SMEM(1), s>>>(xb, w, out, ldo, orow0, t, rows, cols, xl);
    } else if (w12) {
        k_gemm_bf16_tc<0, 1><<<grid, 256, BF_SMEM(0), s>>>(xb, w, out, ldo, orow0, t, rows, cols, NULL);
    } else if (x2) {
        k_gemm_bf16_tc<1><<<grid, 256, BF_SMEM(1), s>>>(xb, w, out, ldo, orow0, t, rows, cols, xl);
    } else {
        k_gemm_bf16_tc<0><<<grid, 256, BF_SMEM(0), s>>>(xb, w, out, ldo, orow0, t, rows, cols, NULL);
    }
}

/* Each block of BHK columns of each row of x (rows of cols float32), clamped
 * to [imin, imax], times a power of two s that takes its largest value to
 * [2^13, 2^14), as two float16 planes: hi, and lo = x s - hi (about 22 bits of
 * x; a block's small values stay clear of the float16 subnormals, though a
 * few channels of the row are far larger), and inv[row steps + block] = 1 /
 * s. A warp a block (BHK = 32 columns: a column a lane). The F16 form of
 * k_gemm_bf16_tc. */
__global__ void k_to_f16_blocks(const float *x, int rows, int cols, __half *hi, __half *lo, float *inv,
                                float imin, float imax)
{
    PDL_START();
    int steps = (cols + BHK - 1) / BHK;
    size_t wb = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) / 32;
    int lane = threadIdx.x % 32;
    if (wb >= (size_t)rows * steps) {
        return;
    }
    size_t r = wb / steps;
    int c = (int)(wb % steps) * BHK + lane;
    float v = c < cols ? fminf(fmaxf(x[r * cols + c], imin), imax) : 0.f;
    float m = fabsf(v);
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, off));
    }
    int e = 0;
    if (m > 0.f && isfinite(m)) {
        frexpf(m, &e);                       /* m = f 2^e, f in [0.5, 1) */
    }
    v *= ldexpf(1.f, 14 - e);
    if (c < cols) {
        __half h = __float2half_rn(v);
        hi[r * cols + c] = h;
        lo[r * cols + c] = __float2half_rn(v - __half2float(h));
    }
    if (lane == 0) {
        inv[wb] = ldexpf(1.f, e - 14);
    }
}

__global__ void k_enc_bias_clamp(float *y, const float *b, int rows, int cols, float lo, float hi);

/* ENC_LINEAR of float16 rows w (wbf 2: a checkpoint's own float16 matrices):
 * x (t rows of cols float32) to its two float16 planes in xb (2 t cols
 * halves, then the t steps floats of the blocks' 1 / s: k_to_f16_blocks),
 * k_gemm_bf16_tc<1, 0, NF, 0, 1> (two products a fragment in float32 sums)
 * into out (t, rows), then the bias (or NULL) and the clamps omin, omax
 * (k_enc_bias_clamp). cols a multiple of 8. */
static void gg_gemm_f16(const float *x, const uint16_t *w, const float *bias, float *out, int t, int rows,
                        int cols, uint16_t *xb, float imin, float imax, float omin, float omax, cudaStream_t s)
{
    static int attr = 0;
    if (!attr) {
        cudaFuncSetAttribute(k_gemm_bf16_tc<1, 0, 0, 0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        cudaFuncSetAttribute(k_gemm_bf16_tc<1, 0, 1, 0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        attr = 1;
    }
    size_t nk = (size_t)t * cols;
    int steps = (cols + BHK - 1) / BHK;
    uint16_t *xl = xb + nk;
    float *inv = (float *)(xb + 2 * nk);
    size_t warps = (size_t)t * steps;
    k_to_f16_blocks<<<(unsigned)((warps + 7) / 8), 256, 0, s>>>(x, t, cols, (__half *)xb, (__half *)xl, inv, imin,
                                                                imax);
    dim3 grid((unsigned)((t + T2M - 1) / T2M), (unsigned)((rows + T2N - 1) / T2N));
    if (t >= 16 * rows) {
        k_gemm_bf16_tc<1, 0, 1, 0, 1><<<dim3(grid.y, grid.x), 256, BF_SMEM(1), s>>>(xb, w, out, rows, NULL, t,
                                                                                   rows, cols, xl, NULL, 1, inv);
    } else {
        k_gemm_bf16_tc<1, 0, 0, 0, 1><<<grid, 256, BF_SMEM(1), s>>>(xb, w, out, rows, NULL, t, rows, cols, xl,
                                                                    NULL, 1, inv);
    }
    size_t nm = (size_t)t * rows;
    k_enc_bias_clamp<<<(unsigned)((nm + 255) / 256), 256, 0, s>>>(out, bias, t, rows, omin, omax);
}

/* ENC_GATHER_MM: the product of the KQ_BF12 rows w with the rows of
 * ENC_GATHER of x (src rows of gc values, idx (t, nb), -1 zeros), without the
 * gathered rows: x (src x gc, 9 times fewer values than the im2col of a conv
 * 3x3) to its bfloat16 planes in xb (xready: an earlier record did, the same
 * x), then k_gemm_bf16_tc<X2, 1, NF, 1>. The bits of ENC_GATHER then the
 * product (MuQ's convs: 4.4 ms of 32 a clip, and 1.3 GB, went to the copies). */
static void gg_gemm_bf12_gather(const float *x, int src, int gc, const int *idx, int nb, const uint8_t *w,
                                float *out, int t, int rows, uint16_t *xb, cudaStream_t s, int xready)
{
    static int attr = 0;
    if (!attr) {
        cudaFuncSetAttribute(k_gemm_bf16_tc<1, 1, 0, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        cudaFuncSetAttribute(k_gemm_bf16_tc<1, 1, 1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)BF_SMEM(1));
        attr = 1;
    }
    int x2 = bf16_tc_on() != 8;
    int cols = nb * gc;
    uint16_t *xl = x2 ? xb + (size_t)src * gc : NULL;
    if (!xready) {
        k_to_bf16<<<(unsigned)(((int64_t)src * gc + 255) / 256), 256, 0, s>>>(x, gc, NULL, xb, src, gc, xl);
    }
    dim3 grid((unsigned)((t + T2M - 1) / T2M), (unsigned)((rows + T2N - 1) / T2N));
    const uint16_t *w16 = (const uint16_t *)w;
    if (x2 && t >= 16 * rows) {
        k_gemm_bf16_tc<1, 1, 1, 1><<<dim3(grid.y, grid.x), 256, BF_SMEM(1), s>>>(xb, w16, out, rows, NULL, t, rows,
                                                                                cols, xl, idx, nb);
    } else if (x2) {
        k_gemm_bf16_tc<1, 1, 0, 1><<<grid, 256, BF_SMEM(1), s>>>(xb, w16, out, rows, NULL, t, rows, cols, xl,
                                                                 idx, nb);
    } else {
        k_gemm_bf16_tc<0, 1, 0, 1><<<grid, 256, BF_SMEM(0), s>>>(xb, w16, out, rows, NULL, t, rows, cols, NULL,
                                                                 idx, nb);
    }
}

/* RECLANK_GPU_BF12_FUSED: 1 (the default) the KQ_BF12 matrices of a large
 * group in k_gemm_bf16_tc<X2, 1> (the groups decoded into the fragments);
 * 0: chunks of bfloat16 rows in a scratch (gg_gemm_bf12), for a test. The
 * dense shapes of Qwen3.8 at 512 tokens (the 3090 at its 200 W cap): fused
 * 1.08-1.11 times the bfloat16 time, the scratch 1.07-1.35 (2.0 for 320 x
 * 10240); 2048 tokens: 1.10, 1.02-1.07. Without the decode (a test) the
 * fused kernel took 1.08: the cost is the three copies of a row of a step
 * (lo, hi, the exponent), not the decode. A decode of each tile once into
 * shared memory (an extra barrier a step) took 1.14. */
static int bf12_fused_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_BF12_FUSED");
        on = !(v && v[0] == '0');
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}
