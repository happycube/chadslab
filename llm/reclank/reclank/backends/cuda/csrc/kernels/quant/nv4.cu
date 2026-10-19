/* KQ_NV4 rows (FP4-Local's MTP head, fp4local_st.mtp_head_nvfp4: the drafts only) for a step or a few
 * tokens: the generic k_kq_linear took a warp a row and a byte a value (2.7 ms for the 248320 rows of
 * the head, 133 GB/s). Here a warp takes R rows; each lane 4 bytes of codes (8 values: 4 of the low
 * half of a block of 32 and the 4 of its high half) of each row a step, 8 blocks a step; x (t <= NT
 * rows) staged in shared memory. A row: the codes (byte i of block b: column 32 b + i low, 32 b + 16 +
 * i high), an E4M3 scale for each 16 (low, high), the float32 g; w = 0.5 g e4m3(s) e2m1x2(c). The sums
 * are float32 in an order of their own (the drafts do not need the bits of another kernel). */
template <int R, int NT>
__global__ void __launch_bounds__(256) k_nv4_gemv(const uint8_t *w, int rows, int cols, const float *x, int t,
                                                 float *out)
{
    extern __shared__ __align__(16) float nxs[];       /* NT x cols */
    __shared__ float e2t[16];                          /* twice the E2M1 value of each code */
    if (threadIdx.x < 16) {
        e2t[threadIdx.x] = kq_e2m1x2(threadIdx.x);
    }
    for (int i = threadIdx.x; i < t * cols / 4; i += 256) {
        ((float4 *)nxs)[i] = ((const float4 *)x)[i];
    }
    __syncthreads();
    const int row0 = (blockIdx.x * 8 + threadIdx.x / 32) * R, lane = threadIdx.x % 32;
    if (row0 >= rows) {
        return;
    }
    const size_t rb = kq_row_bytes(KQ_NV4, cols);
    const int k0 = 4 * (lane % 4);          /* the bytes k0 .. k0 + 3 of a block */
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            acc[r][j] = 0.f;
        }
    }
    const int nb = cols / 32;
#pragma unroll 2
    for (int b = lane / 4; b < nb; b += 8) {
        uint32_t q[R];
        float s0[R], s1[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint8_t *wr = w + (size_t)(row0 + r < rows ? row0 + r : row0) * rb;
            q[r] = *(const uint32_t *)(wr + 16 * b + k0);
            const uint16_t sc = *(const uint16_t *)(wr + cols / 2 + 2 * b);
            s0[r] = kq_e4m3s(sc & 0xff);        /* (2^-8 e4m3: the 256 goes to g below) */
            s1[r] = kq_e4m3s(sc >> 8);
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            if (j < t) {
                const float4 xl = *(const float4 *)(nxs + (size_t)j * cols + 32 * b + k0);
                const float4 xh = *(const float4 *)(nxs + (size_t)j * cols + 32 * b + 16 + k0);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t c = q[r];
                    float pl = e2t[c & 15] * xl.x + e2t[(c >> 8) & 15] * xl.y + e2t[(c >> 16) & 15] * xl.z +
                               e2t[(c >> 24) & 15] * xl.w;
                    float ph = e2t[(c >> 4) & 15] * xh.x + e2t[(c >> 12) & 15] * xh.y +
                               e2t[(c >> 20) & 15] * xh.z + e2t[(c >> 28) & 15] * xh.w;
                    acc[r][j] += s0[r] * pl + s1[r] * ph;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        const float g = row0 + r < rows ? *(const float *)(w + (size_t)(row0 + r) * rb + cols / 2 + cols / 16) : 0.f;
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            float v = acc[r][j];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (lane == 0 && j < t && row0 + r < rows) {
                out[(size_t)j * rows + row0 + r] = (128.f * g) * v;      /* 0.5 g, and the 2^8 of kq_e4m3s */
            }
        }
    }
}

#define NV4_R 8             /* the rows of a warp */
#define NV4_SW 4096         /* the columns of a slice of x (k_nv4_gemv2) */

/* The same product, a block of 16 bytes a lane: lane l takes block b0 + l of each of the R rows of
 * its warp (the 512 bytes of 32 blocks of a row a load of the warp, R rows of them in flight), the
 * warps of the grid loop over the groups of R rows (x staged once a block). x in shared memory with
 * the 8 float4 of each block of 32 rotated by the block (chunk c of block b at 4 ((c + b) % 8)): the
 * lanes of a quarter warp read 8 different banks. */
template <int R, int NT, int U>
__global__ void __launch_bounds__(256) k_nv4_gemv2(const uint8_t *w, int rows, int cols, const float *x, int t,
                                                  float *out)
{
    constexpr int SW = 1024 * U;                       /* the columns of a slice: U blocks a lane */
    extern __shared__ __align__(16) float nxs[];       /* NT x SW, rotated */
    __shared__ float e2t[16];
    if (threadIdx.x < 16) {
        e2t[threadIdx.x] = kq_e2m1x2(threadIdx.x);
    }
    const int lane = threadIdx.x % 32, nw = blockDim.x / 32;
    const size_t rb = kq_row_bytes(KQ_NV4, cols);
    const int ngr = (rows + R - 1) / R;
    /* x in slices of SW columns (a wide x: twice the blocks an SM); a warp's rows the same in each
     * slice, the sums of a later slice added to out */
    for (int s0 = 0; s0 < cols; s0 += SW) {
    if (s0 > 0) {
        __syncthreads();
    }
    for (int i = threadIdx.x; i < t * SW / 4; i += blockDim.x) {
        const int j = i / (SW / 4), c4 = i % (SW / 4), b = c4 / 8, c = c4 % 8;
        ((float4 *)nxs)[(size_t)j * (SW / 4) + 8 * b + (c + b) % 8] = ((const float4 *)(x + (size_t)j * cols + s0))[c4];
    }
    __syncthreads();
    for (int gr = blockIdx.x * nw + threadIdx.x / 32; gr < ngr; gr += gridDim.x * nw) {
        const int row0 = gr * R;
        float acc[R][NT];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                acc[r][j] = 0.f;
            }
        }
        /* U blocks a lane a pass (the slice), every load first */
        {
            const int b0 = s0 / 32;
            uint4 q[R][U];
            uint16_t sc[R][U];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const uint8_t *wr = w + (size_t)(row0 + r < rows ? row0 + r : row0) * rb;
#pragma unroll
                for (int u = 0; u < U; ++u) {
                    const int b = b0 + lane + 32 * u;
                    q[r][u] = *(const uint4 *)(wr + 16 * b);
                    sc[r][u] = *(const uint16_t *)(wr + cols / 2 + 2 * b);
                }
            }
#pragma unroll
            for (int u = 0; u < U; ++u) {
                const int b = lane + 32 * u;        /* (of the slice) */
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    if (j < t) {
                        const float *xb = nxs + (size_t)j * SW + 32 * b;
                        float pl[R], ph[R];
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            pl[r] = ph[r] = 0.f;
                        }
#pragma unroll
                        for (int c = 0; c < 8; ++c) {
                            const float4 xv = *(const float4 *)(xb + 4 * ((c + b) % 8));
#pragma unroll
                            for (int r = 0; r < R; ++r) {
                                const uint32_t v = c % 4 == 0 ? q[r][u].x : c % 4 == 1 ? q[r][u].y : c % 4 == 2 ? q[r][u].z : q[r][u].w;
                                const int sh = c < 4 ? 0 : 4;
                                const float d = e2t[(v >> sh) & 15] * xv.x + e2t[(v >> (8 + sh)) & 15] * xv.y +
                                                e2t[(v >> (16 + sh)) & 15] * xv.z + e2t[(v >> (24 + sh)) & 15] * xv.w;
                                if (c < 4) {
                                    pl[r] += d;
                                } else {
                                    ph[r] += d;
                                }
                            }
                        }
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            acc[r][j] += kq_e4m3s(sc[r][u] & 0xff) * pl[r] + kq_e4m3s(sc[r][u] >> 8) * ph[r];
                        }
                    }
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const float g = row0 + r < rows ? *(const float *)(w + (size_t)(row0 + r) * rb + cols / 2 + cols / 16) : 0.f;
#pragma unroll
            for (int j = 0; j < NT; ++j) {
                float v = acc[r][j];
                for (int o = 16; o > 0; o >>= 1) {
                    v += __shfl_xor_sync(RC_FULL_MASK, v, o);
                }
                if (lane == 0 && j < t && row0 + r < rows) {
                    float *o = out + (size_t)j * rows + row0 + r;
                    *o = s0 == 0 ? (128.f * g) * v : *o + (128.f * g) * v;
                }
            }
        }
    }
    }
}

static int nv4_sms(void)
{
    static int n = 0;
    if (n == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev);
        n = n > 0 ? n : 1;
    }
    return n;
}

/* RECLANK_GPU_NV4_GEMV=0: the generic kernel */
static int nv4_gemv_off(void)
{
    static int off = -1;
    if (off < 0) {
        const char *v = getenv("RECLANK_GPU_NV4_GEMV");
        off = v && v[0] == '0';
    }
    return off;
}

/* RECLANK_GPU_NV4_GEMV2=0: the first form (a word a lane) */
static int nv4_gemv2_off(void)
{
    static int off = -1;
    if (off < 0) {
        const char *v = getenv("RECLANK_GPU_NV4_GEMV2");
        off = v && v[0] == '0';
    }
    return off;
}

/* GP_KQ_LINEAR of KQ_NV4 rows for t <= 8 (else 1: the generic kernel) */
static int nv4_linear_launch(const float *x, const uint8_t *w, int rows, int cols, int t, float *out, cudaStream_t s)
{
    if (t > 8 || cols % 256 != 0 || nv4_gemv_off()) {
        return 1;
    }
    if (t > 4 && (cols % NV4_SW != 0 || nv4_gemv2_off())) {
        return 1;
    }
    if (t <= 4 && (size_t)t * cols * 4 > 48 * 1024 && (cols % NV4_SW != 0 || nv4_gemv2_off())) {
        return 1;
    }
    const size_t sm = (size_t)t * cols * 4;
    if (cols % NV4_SW == 0 && t <= 8 && !nv4_gemv2_off()) {
        /* 2 rows a warp, the grid 4 blocks an SM; slices of x of 32 KB at most (Pascal's 48) */
        const int nth = 256, per = nth / 32 * 2;
        unsigned nblk = (unsigned)((rows + per - 1) / per), cap = (unsigned)(nv4_sms() * 4);
        nblk = nblk < cap ? nblk : cap;
        if (t == 1) {
            k_nv4_gemv2<2, 1, 4><<<nblk, nth, (size_t)1 * 4096 * 4, s>>>(w, rows, cols, x, t, out);
        } else if (t == 2) {
            k_nv4_gemv2<2, 2, 4><<<nblk, nth, (size_t)2 * 4096 * 4, s>>>(w, rows, cols, x, t, out);
        } else if (t <= 4) {
            k_nv4_gemv2<2, 4, 2><<<nblk, nth, (size_t)4 * 2048 * 4, s>>>(w, rows, cols, x, t, out);
        } else {
            k_nv4_gemv2<2, 8, 1><<<nblk, nth, (size_t)8 * 1024 * 4, s>>>(w, rows, cols, x, t, out);
        }
        return 0;
    }
    const unsigned nblk = (unsigned)((rows + 8 * NV4_R - 1) / (8 * NV4_R));
    if (t == 1) {
        k_nv4_gemv<NV4_R, 1><<<nblk, 256, sm, s>>>(w, rows, cols, x, t, out);
    } else if (t == 2) {
        k_nv4_gemv<NV4_R, 2><<<nblk, 256, sm, s>>>(w, rows, cols, x, t, out);
    } else {
        k_nv4_gemv<NV4_R, 4><<<nblk, 256, sm, s>>>(w, rows, cols, x, t, out);
    }
    return 0;
}
