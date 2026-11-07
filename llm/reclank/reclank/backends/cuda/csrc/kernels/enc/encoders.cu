/* ---- The media encoders ---------------------------------------------------
 *
 * The kernels of the records GP_ENC_* (the encoders of images and audio,
 * reclank/encoders/runtime.py, as programs). The arrays are row major. */

#define EG_T 64
#define EG_K 16

/* F20 rows (backends/cpu/csrc/kernels/kq/formats.c KQ_F20; k a multiple of 32): the bytes of a
 * row, and the float32 values of row w at columns kc .. kc + 3 (kc a multiple of 4) or kc. */
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

__device__ __forceinline__ float f20_at(const uint8_t *w, int k, int kc)
{
    const int g = kc >> 5, j = kc & 31;
    const int gap = (w[2 * (size_t)k + 16 * g + (j & 15)] >> (4 * (j >> 4))) & 15;
    return f20_val(((const uint16_t *)w)[kc], gap, w[2 * (size_t)k + k / 2 + g]);
}

/* y (n, m) = clamp(clamp(x, imin, imax) W^T + b, omin, omax). x (n, k) is
 * float32 and W (m, k) bfloat16 (wbf 1), float16 (wbf 2), F20 rows (wbf 3) or float32; b may be
 * NULL. Any n, m, and k (F20: k a multiple of 32). A block of 256 threads makes a tile of 64 x 64
 * values of y. */
__global__ void k_enc_gemm(const float *x, const void *w, int wbf, const float *b, float *y,
                           int n, int m, int k, float imin, float imax, float omin, float omax)
{
    PDL_START();
    __shared__ float xs[EG_K][EG_T + 1];
    __shared__ float ws[EG_K][EG_T + 1];
    int r0 = blockIdx.y * EG_T, c0 = blockIdx.x * EG_T;
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int k0 = 0; k0 < k; k0 += EG_K) {
        for (int q = threadIdx.x; q < EG_T * EG_K; q += blockDim.x) {
            int rr = q / EG_K, kk = q % EG_K, kc = k0 + kk;
            int r = r0 + rr, c = c0 + rr;
            float xv = 0.f, wv = 0.f;
            if (r < n && kc < k) {
                xv = fminf(fmaxf(x[(size_t)r * k + kc], imin), imax);
            }
            if (c < m && kc < k) {
                size_t o = (size_t)c * k + kc;
                wv = wbf == 3 ? f20_at((const uint8_t *)w + (size_t)c * f20_row_bytes(k), k, kc)
                     : wbf == 2 ? __half2float(((const __half *)w)[o])
                     : wbf ? __uint_as_float((uint32_t)((const uint16_t *)w)[o] << 16)
                           : ((const float *)w)[o];
            }
            xs[kk][rr] = xv;
            ws[kk][rr] = wv;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < EG_K; ++kk) {
            float a[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = xs[kk][ty + 16 * i];
                bv[i] = ws[kk][tx + 16 * i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        int r = r0 + ty + 16 * i;
        for (int j = 0; j < 4; ++j) {
            int c = c0 + tx + 16 * j;
            if (r < n && c < m) {
                float v = acc[i][j] + (b ? b[c] : 0.f);
                y[(size_t)r * m + c] = fminf(fmaxf(v, omin), omax);
            }
        }
    }
}

/* k_enc_gemm of float32 W (wbf 0) in tiles of 16 ETM x 16 ETN values of y
 * (ETM, ETN: 8 or 4): a block of 256 threads, ETM x ETN values a thread, k in
 * steps of 8 through shared memory (two buffers, the next step's values
 * loaded into registers while this one's products run; float4 loads where k
 * is a multiple of 4). Each value of y is k_enc_gemm's: fmaf(x, w, acc) over
 * k in order from 0 (the zeros past k change nothing), + b, the clamps: the
 * same bits. F20: W F20 rows (k a multiple of 32), thread tid < BN loading 8 values of k of row
 * tid of W's tile (f20_load8), decoded to their float32 (f20_dec8) as it stores them: the bits of
 * float32 W of the decoded values. */
#define ESG_K 8
template <int ETM, int ETN, bool F20>
__global__ void __launch_bounds__(256) k_enc_sgemm(const float *x, const void *w, const float *b, float *y,
                                                   int n, int m, int k, float imin, float imax, float omin,
                                                   float omax)
{
    PDL_START();
    constexpr int BM = 16 * ETM, BN = 16 * ETN;
    __shared__ __align__(16) float xs[2][ESG_K][BM];
    __shared__ __align__(16) float ws[2][ESG_K][BN];
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    /* the loads: thread tid reads 4 values of k (lk .. lk + 3) of row lr of the tiles */
    const int lr = tid / 2, lk = (tid % 2) * 4;
    const bool v4 = (k % 4) == 0;
    float acc[ETM][ETN];
    #pragma unroll
    for (int i = 0; i < ETM; ++i) {
        #pragma unroll
        for (int j = 0; j < ETN; ++j) {
            acc[i][j] = 0.f;
        }
    }
    float xr[4], wr[4];
    /* F20: the words of row tid of W's tile (zeros past m: lo 0, gap 0, E 0), decoded in store */
    F20Words8 wf = {make_uint4(0u, 0u, 0u, 0u), make_uint2(0u, 0u), 0, 0};
    const uint8_t *wrow = nullptr;
    if (F20 && tid < BN && c0 + tid < m) {
        wrow = (const uint8_t *)w + (size_t)(c0 + tid) * f20_row_bytes(k);
    }
    auto load1 = [&](const float *src, int rows, int r, int kc, float *dst) {
        if (v4 && kc + 3 < k) {
            float4 a = r < rows ? *(const float4 *)(src + (size_t)r * k + kc) : make_float4(0.f, 0.f, 0.f, 0.f);
            dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
        } else {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                dst[q] = r < rows && kc + q < k ? src[(size_t)r * k + kc + q] : 0.f;
            }
        }
    };
    auto load = [&](int k0) {
        if (lr < BM) {
            load1(x, n, r0 + lr, k0 + lk, xr);
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                xr[q] = fminf(fmaxf(xr[q], imin), imax);
            }
        }
        if constexpr (F20) {
            if (wrow != nullptr) {
                wf = f20_load8(wrow, k, k0);
            }
        } else if (lr < BN) {
            load1((const float *)w, m, c0 + lr, k0 + lk, wr);
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (lr < BM) {
                xs[buf][lk + q][lr] = xr[q];
            }
            if (!F20 && lr < BN) {
                ws[buf][lk + q][lr] = wr[q];
            }
        }
        if constexpr (F20) {
            if (tid < BN) {
                float w8[8];
                f20_dec8(wf, w8);
                #pragma unroll
                for (int q = 0; q < ESG_K; ++q) {
                    ws[buf][q][tid] = w8[q];
                }
            }
        }
    };
    /* the thread's rows: groups of 4 at ty 4 (and BM / 2 + ty 4); its columns the same by tx */
    auto row = [&](int i) { return (i / 4) * (BM / 2) + ty * 4 + i % 4; };
    auto col = [&](int j) { return (j / 4) * (BN / 2) + tx * 4 + j % 4; };
    load(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < k; k0 += ESG_K) {
        const bool more = k0 + ESG_K < k;
        if (more) {
            load(k0 + ESG_K);
        }
        #pragma unroll
        for (int kk = 0; kk < ESG_K; ++kk) {
            float a[ETM], bv[ETN];
            #pragma unroll
            for (int i = 0; i < ETM; i += 4) {
                float4 v = *(const float4 *)&xs[buf][kk][row(i)];
                a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
            }
            #pragma unroll
            for (int j = 0; j < ETN; j += 4) {
                float4 v = *(const float4 *)&ws[buf][kk][col(j)];
                bv[j] = v.x; bv[j + 1] = v.y; bv[j + 2] = v.z; bv[j + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < ETM; ++i) {
                #pragma unroll
                for (int j = 0; j < ETN; ++j) {
                    acc[i][j] = fmaf(a[i], bv[j], acc[i][j]);
                }
            }
        }
        if (more) {
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }
    #pragma unroll
    for (int i = 0; i < ETM; ++i) {
        const int r = r0 + row(i);
        if (r >= n) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < ETN; ++j) {
            const int c = c0 + col(j);
            if (c < m) {
                float v = acc[i][j] + (b ? b[c] : 0.f);
                y[(size_t)r * m + c] = fminf(fmaxf(v, omin), omax);
            }
        }
    }
}

/* k_enc_sgemm in the largest tiles of which there are enough for the SMs
 * (128 x 128, 64 x 128, 64 x 64); f20: W F20 rows. */
template <bool F20>
static void enc_sgemm_t(const float *x, const void *w, const float *b, float *y, int n, int m, int k, float imin,
                        float imax, float omin, float omax, int sms, cudaStream_t s)
{
    auto blocks = [&](int bm, int bn) { return (long)((n + bm - 1) / bm) * ((m + bn - 1) / bn); };
    if (blocks(128, 128) >= sms) {
        k_enc_sgemm<8, 8, F20><<<dim3((unsigned)((m + 127) / 128), (unsigned)((n + 127) / 128)), 256, 0, s>>>(
            x, w, b, y, n, m, k, imin, imax, omin, omax);
    } else if (blocks(64, 128) >= sms) {
        k_enc_sgemm<4, 8, F20><<<dim3((unsigned)((m + 127) / 128), (unsigned)((n + 63) / 64)), 256, 0, s>>>(
            x, w, b, y, n, m, k, imin, imax, omin, omax);
    } else {
        k_enc_sgemm<4, 4, F20><<<dim3((unsigned)((m + 63) / 64), (unsigned)((n + 63) / 64)), 256, 0, s>>>(
            x, w, b, y, n, m, k, imin, imax, omin, omax);
    }
}

static void enc_sgemm(const float *x, const void *w, int f20, const float *b, float *y, int n, int m, int k,
                      float imin, float imax, float omin, float omax, int sms, cudaStream_t s)
{
    if (f20) {
        enc_sgemm_t<true>(x, w, b, y, n, m, k, imin, imax, omin, omax, sms, s);
    } else {
        enc_sgemm_t<false>(x, w, b, y, n, m, k, imin, imax, omin, omax, sms, s);
    }
}

/* GP_KQ_LINEAR of KQ_F20 rows (the float32 dense matrices of an NVFP4 Qwen3.6 on the GPU,
 * models/qwen/gpu35.py QwenGPU.dense): out (t, rows) = x (t, cols) W^T. A few tokens (T of them a
 * pass) by k_kq_f20_gemv: a warp a row, each lane 8 values of each 256 of the row (f20_load8,
 * f20_dec8), float32 sums, the lanes' sums added by shuffles; more tokens by k_enc_sgemm's F20
 * tiles (ENC_LINEAR wbf 3, no bias, no clamps). */
template <int T>
__global__ void __launch_bounds__(256) k_kq_f20_gemv(const uint8_t *w, int rows, int cols, const float *x, float *out)
{
    const int lane = threadIdx.x & 31, row = blockIdx.x * 8 + (threadIdx.x >> 5);
    if (row >= rows) {
        return;
    }
    const uint8_t *wr = w + (size_t)row * f20_row_bytes(cols);
    float acc[T];
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        acc[t] = 0.f;
    }
    for (int kc = lane * 8; kc < cols; kc += 256) {
        float v[8];
        f20_dec8(f20_load8(wr, cols, kc), v);
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            const float4 a = *(const float4 *)(x + (size_t)t * cols + kc);
            const float4 b = *(const float4 *)(x + (size_t)t * cols + kc + 4);
            float s = acc[t];
            s = fmaf(v[0], a.x, s);
            s = fmaf(v[1], a.y, s);
            s = fmaf(v[2], a.z, s);
            s = fmaf(v[3], a.w, s);
            s = fmaf(v[4], b.x, s);
            s = fmaf(v[5], b.y, s);
            s = fmaf(v[6], b.z, s);
            acc[t] = fmaf(v[7], b.w, s);
        }
    }
    #pragma unroll
    for (int t = 0; t < T; ++t) {
        float s = acc[t];
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(0xffffffffu, s, o);
        }
        acc[t] = s;
    }
    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < T; ++t) {
            out[(size_t)t * rows + row] = acc[t];
        }
    }
}

static int kq_f20_linear(const float *x, const uint8_t *w, int rows, int cols, int t, float *out, cudaStream_t s)
{
    if (cols % 32 != 0 || ((uintptr_t)x | (uintptr_t)w) % 16 != 0) {
        return 1;
    }
    if (t > MT_MAX) {
        enc_sgemm(x, w, 1, NULL, out, t, rows, cols, -INFINITY, INFINITY, -INFINITY, INFINITY, enc_sms(), s);
        return 0;
    }
    const unsigned nb = (unsigned)((rows + 7) / 8);
    for (int r0 = 0; r0 < t;) {
        const int rest = t - r0, nr = rest >= 8 ? 8 : rest >= 4 ? 4 : rest >= 2 ? 2 : 1;
        const float *xr = x + (size_t)r0 * cols;
        float *yr = out + (size_t)r0 * rows;
        if (nr == 8) {
            k_kq_f20_gemv<8><<<nb, 256, 0, s>>>(w, rows, cols, xr, yr);
        } else if (nr == 4) {
            k_kq_f20_gemv<4><<<nb, 256, 0, s>>>(w, rows, cols, xr, yr);
        } else if (nr == 2) {
            k_kq_f20_gemv<2><<<nb, 256, 0, s>>>(w, rows, cols, xr, yr);
        } else {
            k_kq_f20_gemv<1><<<nb, 256, 0, s>>>(w, rows, cols, xr, yr);
        }
        r0 += nr;
    }
    return 0;
}

/* ENC_CONV_MM: k_enc_sgemm with x's rows gathered (an im2col without its buffer) and an epilogue.
 * Row r of the implicit x is the blocks of cin values of rows idx[r taps + t] of src (t = 0 ..
 * taps - 1; zeros for an index < 0), so k = taps cin. Each value of y is fmaf(x, w, acc) over k
 * in order from 0, + b (k_enc_gemm's, k_enc_sgemm's: the bits of ENC_GATHER + ENC_LINEAR), then
 * + res (ENC_ADD's) when res is not NULL, then x above 0 or x slope[c] (ENC_PRELU's) when slope
 * is not NULL. scat (not NULL): y's values in place of the ENC_GATHER of its rows of blk values
 * (a deconv's phases to its pixels): block j = r (m / blk) + c / blk to y's row scat[j] (none for
 * -1). Tiles of 16 ETM x 16 ETN (ETN 1, 2, 4, 8: the narrow outputs of a small conv). */
template <int ETM, int ETN>
__global__ void __launch_bounds__(256) k_enc_conv_mm(const float *src, const int *idx, const float *w,
                                                     const float *b, const float *res, const float *slope,
                                                     float *y, int n, int m, int taps, int cin, const int *scat,
                                                     int blk)
{
    PDL_START();
    constexpr int BM = 16 * ETM, BN = 16 * ETN;
    __shared__ __align__(16) float xs[2][ESG_K][BM];
    __shared__ __align__(16) float ws[2][ESG_K][BN < 4 ? 4 : BN];
    const int k = taps * cin;
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int lr = tid / 2, lk = (tid % 2) * 4;
    const bool v4 = (cin % 4) == 0;
    float acc[ETM][ETN];
    #pragma unroll
    for (int i = 0; i < ETM; ++i) {
        #pragma unroll
        for (int j = 0; j < ETN; ++j) {
            acc[i][j] = 0.f;
        }
    }
    float xr[4], wr[4];
    /* the gathered x: value kc .. kc + 3 of row r (one tap when cin is a multiple of 4) */
    auto loadx = [&](int r, int kc) {
        if (r >= n) {
            xr[0] = xr[1] = xr[2] = xr[3] = 0.f;
            return;
        }
        if (v4 && kc < k) {
            const int t = kc / cin, c = kc - t * cin;
            const int sr = idx[(size_t)r * taps + t];
            float4 a = sr >= 0 ? *(const float4 *)(src + (size_t)sr * cin + c) : make_float4(0.f, 0.f, 0.f, 0.f);
            xr[0] = a.x; xr[1] = a.y; xr[2] = a.z; xr[3] = a.w;
            return;
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int kk = kc + q;
            float v = 0.f;
            if (kk < k) {
                const int t = kk / cin, c = kk - t * cin;
                const int sr = idx[(size_t)r * taps + t];
                v = sr >= 0 ? src[(size_t)sr * cin + c] : 0.f;
            }
            xr[q] = v;
        }
    };
    auto loadw = [&](int c, int kc) {
        const bool w4 = (k % 4) == 0;
        if (w4 && kc + 3 < k) {
            float4 a = c < m ? *(const float4 *)(w + (size_t)c * k + kc) : make_float4(0.f, 0.f, 0.f, 0.f);
            wr[0] = a.x; wr[1] = a.y; wr[2] = a.z; wr[3] = a.w;
        } else {
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                wr[q] = c < m && kc + q < k ? w[(size_t)c * k + kc + q] : 0.f;
            }
        }
    };
    auto load = [&](int k0) {
        if (lr < BM) {
            loadx(r0 + lr, k0 + lk);
        }
        if (lr < BN) {
            loadw(c0 + lr, k0 + lk);
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            if (lr < BM) {
                xs[buf][lk + q][lr] = xr[q];
            }
            if (lr < BN) {
                ws[buf][lk + q][lr] = wr[q];
            }
        }
    };
    auto row = [&](int i) { return (i / 4) * (BM / 2) + ty * 4 + i % 4; };
    auto col = [&](int j) { return ETN >= 4 ? (j / 4) * (BN / 2) + tx * 4 + j % 4 : tx * ETN + j; };
    load(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < k; k0 += ESG_K) {
        const bool more = k0 + ESG_K < k;
        if (more) {
            load(k0 + ESG_K);
        }
        #pragma unroll
        for (int kk = 0; kk < ESG_K; ++kk) {
            float a[ETM], bv[ETN];
            #pragma unroll
            for (int i = 0; i < ETM; i += 4) {
                float4 v = *(const float4 *)&xs[buf][kk][row(i)];
                a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
            }
            if constexpr (ETN >= 4) {
                #pragma unroll
                for (int j = 0; j < ETN; j += 4) {
                    float4 v = *(const float4 *)&ws[buf][kk][col(j)];
                    bv[j] = v.x; bv[j + 1] = v.y; bv[j + 2] = v.z; bv[j + 3] = v.w;
                }
            } else {
                #pragma unroll
                for (int j = 0; j < ETN; ++j) {
                    bv[j] = ws[buf][kk][col(j)];
                }
            }
            #pragma unroll
            for (int i = 0; i < ETM; ++i) {
                #pragma unroll
                for (int j = 0; j < ETN; ++j) {
                    acc[i][j] = fmaf(a[i], bv[j], acc[i][j]);
                }
            }
        }
        if (more) {
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }
    #pragma unroll
    for (int i = 0; i < ETM; ++i) {
        const int r = r0 + row(i);
        if (r >= n) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < ETN; ++j) {
            const int c = c0 + col(j);
            if (c < m) {
                float v = __fadd_rn(acc[i][j], b ? b[c] : 0.f);
                if (res != nullptr) {
                    v = __fadd_rn(v, res[(size_t)r * m + c]);
                }
                if (slope != nullptr) {
                    v = v > 0.f ? v : __fmul_rn(v, slope[c]);
                }
                if (scat != nullptr) {
                    const int d = scat[(size_t)r * (m / blk) + c / blk];
                    if (d >= 0) {
                        y[(size_t)d * blk + c % blk] = v;
                    }
                } else {
                    y[(size_t)r * m + c] = v;
                }
            }
        }
    }
}

/* ENC_CONV_MM's tall tiles of 128 x 128 when cin is a multiple of 16 (SeedVR2's video VAE: 128 to 512 channels):
 * k in steps of 16, each one tap (the tap and its channel offset carried from step to step: no division a
 * load, the row's index read once a tap), each thread 8 values of k of a row of x and of W a step (two float4
 * each). The same bits as k_enc_conv_mm: each value fmaf over k in order from 0, then its epilogue; 1.2 to
 * 1.3x its time on the VAE's shapes (11.5-13.4 -> 15.1-15.8 TFLOPS on the 3090). */
__global__ void __launch_bounds__(256) k_enc_conv_mm16(const float *src, const int *idx, const float *w,
                                                       const float *b, const float *res, const float *slope,
                                                       float *y, int n, int m, int taps, int cin, const int *scat,
                                                       int blk)
{
    PDL_START();
    constexpr int ETM = 8, ETN = 8, BM = 128, BN = 128, BK = 16, PER = BK / 2;
    __shared__ __align__(16) float xs[2][BK][BM];
    __shared__ __align__(16) float ws[2][BK][BN];
    const int k = taps * cin;
    const int r0 = blockIdx.y * BM, c0 = blockIdx.x * BN;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int lr = tid / 2, lk = (tid % 2) * PER;
    const int xrow = r0 + lr, wcol = c0 + lr;
    const int *irow = idx + (size_t)(xrow < n ? xrow : 0) * taps;
    const float *wp = w + (size_t)(wcol < m ? wcol : 0) * k + lk;
    float acc[ETM][ETN];
    #pragma unroll
    for (int i = 0; i < ETM; ++i) {
        #pragma unroll
        for (int j = 0; j < ETN; ++j) {
            acc[i][j] = 0.f;
        }
    }
    float xr[PER], wr[PER];
    /* the next load's tap t, channel offset coff and source row sr (-1: zeros) */
    int t = 0, coff = lk, sr = xrow < n ? irow[0] : -1;
    auto load = [&](int k0) {
        const float *sp = src + (size_t)(sr >= 0 ? sr : 0) * cin + coff;
        #pragma unroll
        for (int q = 0; q < PER; q += 4) {
            float4 a = sr >= 0 ? *(const float4 *)(sp + q) : make_float4(0.f, 0.f, 0.f, 0.f);
            xr[q] = a.x; xr[q + 1] = a.y; xr[q + 2] = a.z; xr[q + 3] = a.w;
        }
        #pragma unroll
        for (int q = 0; q < PER; q += 4) {
            float4 a = wcol < m ? *(const float4 *)(wp + k0 + q) : make_float4(0.f, 0.f, 0.f, 0.f);
            wr[q] = a.x; wr[q + 1] = a.y; wr[q + 2] = a.z; wr[q + 3] = a.w;
        }
        coff += BK;
        if (coff >= cin) {
            coff -= cin;
            ++t;
            sr = (xrow < n && t < taps) ? irow[t] : -1;
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int q = 0; q < PER; ++q) {
            xs[buf][lk + q][lr] = xr[q];
            ws[buf][lk + q][lr] = wr[q];
        }
    };
    auto row = [&](int i) { return (i / 4) * (BM / 2) + ty * 4 + i % 4; };
    auto col = [&](int j) { return (j / 4) * (BN / 2) + tx * 4 + j % 4; };
    load(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < k; k0 += BK) {
        const bool more = k0 + BK < k;
        if (more) {
            load(k0 + BK);
        }
        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float a[ETM], bv[ETN];
            #pragma unroll
            for (int i = 0; i < ETM; i += 4) {
                float4 v = *(const float4 *)&xs[buf][kk][row(i)];
                a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
            }
            #pragma unroll
            for (int j = 0; j < ETN; j += 4) {
                float4 v = *(const float4 *)&ws[buf][kk][col(j)];
                bv[j] = v.x; bv[j + 1] = v.y; bv[j + 2] = v.z; bv[j + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < ETM; ++i) {
                #pragma unroll
                for (int j = 0; j < ETN; ++j) {
                    acc[i][j] = fmaf(a[i], bv[j], acc[i][j]);
                }
            }
        }
        if (more) {
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }
    #pragma unroll
    for (int i = 0; i < ETM; ++i) {
        const int r = r0 + row(i);
        if (r >= n) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < ETN; ++j) {
            const int c = c0 + col(j);
            if (c < m) {
                float v = __fadd_rn(acc[i][j], b ? b[c] : 0.f);
                if (res != nullptr) {
                    v = __fadd_rn(v, res[(size_t)r * m + c]);
                }
                if (slope != nullptr) {
                    v = v > 0.f ? v : __fmul_rn(v, slope[c]);
                }
                if (scat != nullptr) {
                    const int d = scat[(size_t)r * (m / blk) + c / blk];
                    if (d >= 0) {
                        y[(size_t)d * blk + c % blk] = v;
                    }
                } else {
                    y[(size_t)r * m + c] = v;
                }
            }
        }
    }
}

/* ENC_CONV_MM of bfloat16 W (wbf 1) on the tensor cores: the implicit im2col's x (float32 rows of src) rounded
 * to bfloat16 (to nearest even) as it is gathered, W's bfloat16 rows (rounded once by the program), mma.sync
 * m16n8k16 with float32 sums; then the epilogue of k_enc_conv_mm (+ b, + res, PReLU, scat). Not the bits of the
 * float32 records: the products of bfloat16 values (SeedVR2's official VAE runs in bfloat16). Tiles of 128 rows x
 * 128 outputs, 8 warps of 64 x 32, k in steps of 32 (cin a multiple of 32: a step within one tap, the tap and
 * its channel offset carried from step to step); W by cp.async, x through registers while a step's products run. */
#define CTC_M 128
#define CTC_N 128
#define CTC_K 32
__device__ __forceinline__ void mma16816_bf16_enc(float *c, const uint32_t *a, const uint32_t *b)
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

__device__ __forceinline__ uint32_t enc_bf2(float lo, float hi)
{
    uint32_t a = __float_as_uint(lo), b = __float_as_uint(hi);
    a = (a + 0x7fffu + ((a >> 16) & 1u)) >> 16;
    b = (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
    return a | (b << 16);
}

__device__ __forceinline__ void mma16816_f16_enc(float *c, const uint32_t *a, const uint32_t *b)
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

/* two float32 values to float16 (to nearest even), saturated at +-65504 (a NaN stays NaN) */
__device__ __forceinline__ uint32_t enc_h2(float lo, float hi)
{
    const __half2 h = __floats2half2_rn(fminf(fmaxf(lo, -65504.f), 65504.f), fminf(fmaxf(hi, -65504.f), 65504.f));
    return *(const uint32_t *)&h;
}

/* k_enc_conv_tc's forms. XM 0: src float32 rows, converted as they are loaded; XM 1: src's rows already 16-bit
 * (uint16: ENC_BF16_PACK's, the same conversion: the same bits as the float32 rows they were packed from; a
 * causal conv's cache, SeedVR2's VAE). H: float16 (W the W16H rows: float16 of W ws, ws a power of two of each
 * output row after the rows, the products divided by it, exactly), else bfloat16. GATHER: x's rows by idx (the
 * implicit im2col), else src's rows themselves (ENC_LINEAR wbf 7: taps 1, cin k; its input and output clamps). */
template <int XM, bool H, bool GATHER>
__global__ void __launch_bounds__(256) k_enc_conv_tc(const void *src, const int *idx, const uint16_t *w,
                                                     const float *b, const float *res, const float *slope,
                                                     float *y, int n, int m, int taps, int cin, const int *scat,
                                                     int blk, float imin, float imax, float omin, float omax)
{
    PDL_START();
    __shared__ __align__(16) uint16_t as_[2][CTC_M][CTC_K + 8];
    __shared__ __align__(16) uint16_t bs[2][CTC_N][CTC_K + 8];
    const int k = taps * cin, steps = k / CTC_K;
    const int r0 = blockIdx.y * CTC_M, c0 = blockIdx.x * CTC_N;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = (warp / 4) * 64, wn = (warp % 4) * 32, g = lane / 4, cq = lane % 4;
    const float *ws = H ? (const float *)(w + (size_t)m * k) : nullptr;
    /* x's loads: thread tid 16 values of k (a half of the step) of row lr */
    const int lr = tid / 2, half = (tid % 2) * 16, xrow = r0 + lr;
    const int *irow = GATHER ? idx + (size_t)(xrow < n ? xrow : 0) * taps : nullptr;
    int t = 0, coff = half, sr = xrow < n ? (GATHER ? irow[0] : xrow) : -1;
    uint32_t xr[8];
    auto cvt = [&](float lo, float hi) -> uint32_t {
        if (!GATHER) {
            lo = fminf(fmaxf(lo, imin), imax);
            hi = fminf(fmaxf(hi, imin), imax);
        }
        return H ? enc_h2(lo, hi) : enc_bf2(lo, hi);
    };
    auto loadx = [&]() {
        if (XM == 1) {
            const uint16_t *sp = (const uint16_t *)src + (size_t)(sr >= 0 ? sr : 0) * cin + coff;
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            const uint4 a = sr >= 0 ? *(const uint4 *)sp : z, c = sr >= 0 ? *(const uint4 *)(sp + 8) : z;
            xr[0] = a.x, xr[1] = a.y, xr[2] = a.z, xr[3] = a.w;
            xr[4] = c.x, xr[5] = c.y, xr[6] = c.z, xr[7] = c.w;
        } else {
            const float *sp = (const float *)src + (size_t)(sr >= 0 ? sr : 0) * cin + coff;
            #pragma unroll
            for (int q = 0; q < 4; ++q) {
                float4 a = sr >= 0 ? *(const float4 *)(sp + 4 * q) : make_float4(0.f, 0.f, 0.f, 0.f);
                xr[2 * q] = cvt(a.x, a.y);
                xr[2 * q + 1] = cvt(a.z, a.w);
            }
        }
        coff += CTC_K;
        if (coff >= cin) {
            coff -= cin;
            ++t;
            sr = (xrow < n && t < taps) ? (GATHER ? irow[t] : xrow) : -1;
        }
    };
    auto storex = [&](int buf) {
        *(uint4 *)&as_[buf][lr][half] = make_uint4(xr[0], xr[1], xr[2], xr[3]);
        *(uint4 *)&as_[buf][lr][half + 8] = make_uint4(xr[4], xr[5], xr[6], xr[7]);
    };
    auto loadw = [&](int st, int buf) {
        const int k0 = st * CTC_K;
        #pragma unroll
        for (int e = tid; e < CTC_N * CTC_K / 8; e += 256) {
            const int nn = e / (CTC_K / 8), kk = (e % (CTC_K / 8)) * 8;
            if (c0 + nn < m) {
                cp_async16(&bs[buf][nn][kk], w + (size_t)(c0 + nn) * k + k0 + kk);
            } else {
                *(uint4 *)&bs[buf][nn][kk] = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        cp_async_commit();
    };
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    loadw(0, 0);
    loadx();
    storex(0);
    cp_async_wait0();
    __syncthreads();
    int buf = 0;
    for (int st = 0; st < steps; ++st) {
        const bool more = st + 1 < steps;
        if (more) {
            loadw(st + 1, buf ^ 1);
            loadx();
        }
        #pragma unroll
        for (int ks = 0; ks < CTC_K / 16; ++ks) {
            const int kk = ks * 16;
            uint32_t bf[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint16_t *br = &bs[buf][wn + j * 8 + g][kk + 2 * cq];
                bf[j][0] = *(const uint32_t *)br;
                bf[j][1] = *(const uint32_t *)(br + 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int ra = wm + i * 16;
                uint32_t a[4];
                a[0] = *(const uint32_t *)&as_[buf][ra + g][kk + 2 * cq];
                a[1] = *(const uint32_t *)&as_[buf][ra + g + 8][kk + 2 * cq];
                a[2] = *(const uint32_t *)&as_[buf][ra + g][kk + 2 * cq + 8];
                a[3] = *(const uint32_t *)&as_[buf][ra + g + 8][kk + 2 * cq + 8];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    if (H) {
                        mma16816_f16_enc(acc[i][j], a, bf[j]);
                    } else {
                        mma16816_bf16_enc(acc[i][j], a, bf[j]);
                    }
                }
            }
        }
        if (more) {
            storex(buf ^ 1);
            cp_async_wait0();
            __syncthreads();
            buf ^= 1;
        }
    }
    auto put = [&](int r, int c, float v0) {
        if (r >= n || c >= m) {
            return;
        }
        if (H) {
            v0 = __fmul_rn(v0, __frcp_rn(ws[c]));       /* (a power of two: exact) */
        }
        float v = __fadd_rn(v0, b ? b[c] : 0.f);
        if (res != nullptr) {
            v = __fadd_rn(v, res[(size_t)r * m + c]);
        }
        if (slope != nullptr) {
            v = v > 0.f ? v : __fmul_rn(v, slope[c]);
        }
        if (!GATHER) {
            v = fminf(fmaxf(v, omin), omax);
        }
        if (scat != nullptr) {
            const int d = scat[(size_t)r * (m / blk) + c / blk];
            if (d >= 0) {
                y[(size_t)d * blk + c % blk] = v;
            }
        } else {
            y[(size_t)r * m + c] = v;
        }
    };
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + wn + j * 8 + 2 * cq, r = r0 + wm + i * 16 + g;
            put(r, c, acc[i][j][0]);
            put(r, c + 1, acc[i][j][1]);
            put(r + 8, c, acc[i][j][2]);
            put(r + 8, c + 1, acc[i][j][3]);
        }
    }
}

/* ENC_LINEAR of W16H rows (wbf 7: any float32 W stored as float16, encoders.runtime.to_w16) on the tensor cores:
 * k_enc_conv_tc without the gather (k a multiple of 32, x's rows 16-byte aligned) */
void enc_linear_w16h(const float *x, const uint16_t *w, const float *b, float *y, int n, int m, int k, float imin,
                     float imax, float omin, float omax, cudaStream_t s)
{
    const dim3 grid((unsigned)((m + CTC_N - 1) / CTC_N), (unsigned)((n + CTC_M - 1) / CTC_M));
    k_enc_conv_tc<0, true, false><<<grid, 256, 0, s>>>(x, nullptr, w, b, nullptr, nullptr, y, n, m, 1, k, nullptr, 1,
                                                       imin, imax, omin, omax);
}

/* I16 rows (W16I's, and x's: ENC_I16_PACK): values in groups of 64 along the row, each group an int16 q of each
 * value with a float32 scale s (the value q s): q = rint(v 32639 / mx) (mx the group's largest |v|; mx 0, or not
 * finite: q 0, s 0), s = mx / 32639; q as two int8, hi = floor((q + 128) / 256) and lo = q - 256 hi. A row of k
 * values: hi (k bytes), lo (k bytes), the k / 64 scales, padded to 16 bytes (i16_row_bytes). */
__host__ __device__ __forceinline__ size_t i16_row_bytes(int k)
{
    return ((size_t)2 * k + 4 * (size_t)(k / 64) + 15) / 16 * 16;
}

/* the group's inverse (32639 / mx) and scale (mx / 32639) of its largest |v| */
__device__ __forceinline__ void i16_scale(float mx, float &inv, float &sc)
{
    const bool ok = mx > 0.f && mx <= 3.402823466e38f;
    inv = ok ? __fdiv_rn(32639.f, mx) : 0.f;
    sc = ok ? __fdiv_rn(mx, 32639.f) : 0.f;
}

/* four values to their hi and lo bytes (packed in two words) */
__device__ __forceinline__ void i16_q4(float a, float b, float c, float d, float inv, uint32_t &h, uint32_t &l)
{
    const float v[4] = {a, b, c, d};
    h = l = 0u;
    #pragma unroll
    for (int e = 0; e < 4; ++e) {
        int q = __float2int_rn(__fmul_rn(v[e], inv));
        q = q > 32639 ? 32639 : q < -32639 ? -32639 : q;
        const int hi = (q + 128) >> 8, lo = q - (hi << 8);
        h |= (uint32_t)(hi & 255) << (8 * e);
        l |= (uint32_t)(lo & 255) << (8 * e);
    }
}

/* four 8 x 16-byte matrices of shared memory (ldmatrix x4, b16): the int8 fragments of m16n8k32 */
__device__ __forceinline__ void ldsm_x4_enc(uint32_t *d, const void *p)
{
#if RC_SM80
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
                 : "r"(a));
#else
    __trap();
#endif
}

__device__ __forceinline__ void mma16832_s8_enc(int *c, const uint32_t *a, const uint32_t *b)
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

/* ENC_I16_PACK: float32 rows x (cols a multiple of 64) to I16 rows y: two threads a group (32 values each) */
__global__ void k_enc_i16_pack(const float *x, uint8_t *y, int64_t rows, int cols)
{
    PDL_START();
    const int ng = cols / 64;
    const int64_t gi = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 1;
    const int part = threadIdx.x & 1;
    const bool live = gi < rows * ng;
    const int64_t r = live ? gi / ng : 0;
    const int grp = live ? (int)(gi % ng) : 0;
    const float *xp = x + r * cols + grp * 64 + part * 32;
    float v[32];
    float mx = 0.f;
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
        const float4 a = live ? *(const float4 *)(xp + 4 * q) : make_float4(0.f, 0.f, 0.f, 0.f);
        v[4 * q] = a.x, v[4 * q + 1] = a.y, v[4 * q + 2] = a.z, v[4 * q + 3] = a.w;
        mx = fmaxf(mx, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
    }
    mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
    if (!live) {
        return;
    }
    float inv, sc;
    i16_scale(mx, inv, sc);
    uint8_t *o = y + r * i16_row_bytes(cols);
    uint32_t h[8], l[8];
    #pragma unroll
    for (int q = 0; q < 8; ++q) {
        i16_q4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3], inv, h[q], l[q]);
    }
    const int off = grp * 64 + part * 32;
    *(uint4 *)(o + off) = make_uint4(h[0], h[1], h[2], h[3]);
    *(uint4 *)(o + off + 16) = make_uint4(h[4], h[5], h[6], h[7]);
    *(uint4 *)(o + cols + off) = make_uint4(l[0], l[1], l[2], l[3]);
    *(uint4 *)(o + cols + off + 16) = make_uint4(l[4], l[5], l[6], l[7]);
    if (part == 0) {
        ((float *)(o + 2 * (size_t)cols))[grp] = sc;
    }
}

/* ENC_CONV_MM of W16I rows on the tensor cores (wbf 5: x float32, each group of 64 of a gathered row quantized
 * as it is loaded, as ENC_I16_PACK; wbf 6: x's rows I16 already, ENC_I16_PACK's: the same bits), and ENC_LINEAR's
 * wbf 8 (GATHER false: src's rows, the input and output clamps). Each group of 64 k: mma.sync m16n8k32 s8 of hi
 * hi, hi lo and lo hi with int32 sums (lo lo left out), then the float32 sum += (256 hh + (hl + lh)) 256 s_x s_w.
 * Tiles of 128 rows x 64 outputs, 8 warps of 32 x 32, a group a buffer (x's one, W's two: 41.5 KB); cin (k) a
 * multiple of 64. */
#define CI_M 128
#define CI_N 64
#define CI_K 64
template <int XM, bool GATHER>
__global__ void __launch_bounds__(256) k_enc_conv_i16(const void *src, const int *idx, const uint8_t *w,
                                                      const float *b, const float *res, const float *slope,
                                                      float *y, int n, int m, int taps, int cin, const int *scat,
                                                      int blk, float imin, float imax, float omin, float omax)
{
    PDL_START();
    __shared__ __align__(16) uint8_t xh[CI_M][CI_K + 16], xl[CI_M][CI_K + 16];
    __shared__ __align__(16) uint8_t wh[2][CI_N][CI_K + 16], wl[2][CI_N][CI_K + 16];
    __shared__ float xs[CI_M], wsc[2][CI_N];
    const int k = taps * cin, groups = k / CI_K;
    const size_t wrb = i16_row_bytes(k), xrb = i16_row_bytes(cin);
    const int r0 = blockIdx.y * CI_M, c0 = blockIdx.x * CI_N;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = (warp / 2) * 32, wn = (warp % 2) * 32, g = lane / 4, cq = lane % 4;
    /* x: thread tid the 32 values part of a group of row lr; W: row tid / 4, the 32 bytes part of its hi or lo */
    const int lr = tid / 2, part = tid % 2, xrow = r0 + lr;
    const int wr = tid / 4, wpl = (tid / 2) % 2, wrow = c0 + wr;
    const int *irow = GATHER ? idx + (size_t)(xrow < n ? xrow : 0) * taps : nullptr;
    int t = 0, coff = 0, sr = xrow < n ? (GATHER ? irow[0] : xrow) : -1;
    uint32_t hr[8], lr_[8];
    float sxr = 0.f;
    auto loadx = [&]() {
        if (XM == 1) {
            const uint8_t *row = (const uint8_t *)src + (size_t)(sr >= 0 ? sr : 0) * xrb;
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            const uint4 h0 = sr >= 0 ? *(const uint4 *)(row + coff + part * 32) : z;
            const uint4 h1 = sr >= 0 ? *(const uint4 *)(row + coff + part * 32 + 16) : z;
            const uint4 l0 = sr >= 0 ? *(const uint4 *)(row + cin + coff + part * 32) : z;
            const uint4 l1 = sr >= 0 ? *(const uint4 *)(row + cin + coff + part * 32 + 16) : z;
            hr[0] = h0.x, hr[1] = h0.y, hr[2] = h0.z, hr[3] = h0.w, hr[4] = h1.x, hr[5] = h1.y, hr[6] = h1.z;
            hr[7] = h1.w;
            lr_[0] = l0.x, lr_[1] = l0.y, lr_[2] = l0.z, lr_[3] = l0.w, lr_[4] = l1.x, lr_[5] = l1.y;
            lr_[6] = l1.z, lr_[7] = l1.w;
            sxr = sr >= 0 ? ((const float *)(row + 2 * (size_t)cin))[coff / 64] : 0.f;
        } else {
            const float *sp = (const float *)src + (size_t)(sr >= 0 ? sr : 0) * cin + coff + part * 32;
            float v[32];
            float mx = 0.f;
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                float4 a = sr >= 0 ? *(const float4 *)(sp + 4 * q) : make_float4(0.f, 0.f, 0.f, 0.f);
                if (!GATHER) {
                    a.x = fminf(fmaxf(a.x, imin), imax), a.y = fminf(fmaxf(a.y, imin), imax);
                    a.z = fminf(fmaxf(a.z, imin), imax), a.w = fminf(fmaxf(a.w, imin), imax);
                }
                v[4 * q] = a.x, v[4 * q + 1] = a.y, v[4 * q + 2] = a.z, v[4 * q + 3] = a.w;
                mx = fmaxf(mx, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
            }
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
            float inv;
            i16_scale(mx, inv, sxr);
            #pragma unroll
            for (int q = 0; q < 8; ++q) {
                i16_q4(v[4 * q], v[4 * q + 1], v[4 * q + 2], v[4 * q + 3], inv, hr[q], lr_[q]);
            }
        }
        coff += CI_K;
        if (coff >= cin) {
            coff = 0;
            ++t;
            sr = (xrow < n && t < taps) ? (GATHER ? irow[t] : xrow) : -1;
        }
    };
    auto storex = [&]() {
        *(uint4 *)&xh[lr][part * 32] = make_uint4(hr[0], hr[1], hr[2], hr[3]);
        *(uint4 *)&xh[lr][part * 32 + 16] = make_uint4(hr[4], hr[5], hr[6], hr[7]);
        *(uint4 *)&xl[lr][part * 32] = make_uint4(lr_[0], lr_[1], lr_[2], lr_[3]);
        *(uint4 *)&xl[lr][part * 32 + 16] = make_uint4(lr_[4], lr_[5], lr_[6], lr_[7]);
        if (part == 0) {
            xs[lr] = sxr;
        }
    };
    auto loadw = [&](int gr, int buf) {
        const int k0 = gr * CI_K;
        uint8_t *dst = wpl ? &wl[buf][wr][part * 32] : &wh[buf][wr][part * 32];
        if (wrow < m) {
            const uint8_t *row = w + (size_t)wrow * wrb + (wpl ? k : 0) + k0 + part * 32;
            cp_async16(dst, row);
            cp_async16(dst + 16, row + 16);
            if (tid % 4 == 0) {
                cp_async4(&wsc[buf][wr], w + (size_t)wrow * wrb + 2 * (size_t)k + 4 * (size_t)gr);
            }
        } else {
            const uint4 z = make_uint4(0u, 0u, 0u, 0u);
            *(uint4 *)dst = z, *(uint4 *)(dst + 16) = z;
            if (tid % 4 == 0) {
                wsc[buf][wr] = 0.f;
            }
        }
        cp_async_commit();
    };
    float acc[2][4][4];
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    loadw(0, 0);
    loadx();
    storex();
    cp_async_wait0();
    __syncthreads();
    int buf = 0;
    for (int gr = 0; gr < groups; ++gr) {
        const bool more = gr + 1 < groups;
        if (more) {
            loadw(gr + 1, buf ^ 1);
            loadx();
        }
        int hh[2][4][4], cr[2][4][4];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    hh[i][j][e] = cr[i][j][e] = 0;
                }
            }
        }
        /* ldmatrix: lane l the row l % 8 of its matrix l / 8 */
        const int lq = lane / 8, l8 = lane % 8;
        #pragma unroll
        for (int ks = 0; ks < CI_K / 32; ++ks) {
            const int kk = ks * 32;
            uint32_t bh[4][2], bl[4][2];
            #pragma unroll
            for (int j = 0; j < 4; j += 2) {
                /* matrices (j, bytes 0-15), (j, 16-31), (j + 1, 0-15), (j + 1, 16-31) */
                const int nn = wn + (j + lq / 2) * 8 + l8, kb = kk + (lq % 2) * 16;
                uint32_t t[4];
                ldsm_x4_enc(t, &wh[buf][nn][kb]);
                bh[j][0] = t[0], bh[j][1] = t[1], bh[j + 1][0] = t[2], bh[j + 1][1] = t[3];
                ldsm_x4_enc(t, &wl[buf][nn][kb]);
                bl[j][0] = t[0], bl[j][1] = t[1], bl[j + 1][0] = t[2], bl[j + 1][1] = t[3];
            }
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                /* matrices (rows 0-7, bytes 0-15), (8-15, 0-15), (0-7, 16-31), (8-15, 16-31): a0..a3 */
                const int ra = wm + i * 16 + (lq % 2) * 8 + l8, kb = kk + (lq / 2) * 16;
                uint32_t ah[4], al[4];
                ldsm_x4_enc(ah, &xh[ra][kb]);
                ldsm_x4_enc(al, &xl[ra][kb]);
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma16832_s8_enc(hh[i][j], ah, bh[j]);
                    mma16832_s8_enc(cr[i][j], ah, bl[j]);
                    mma16832_s8_enc(cr[i][j], al, bh[j]);
                }
            }
        }
        /* the group into the float32 sums: (256 hh + cr) 256 s_x s_w (256 hh + cr below 2^31: |hh| <= 127^2 64) */
        float sx[2][2], sw[4][2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            sx[i][0] = __fmul_rn(xs[wm + i * 16 + g], 256.f);
            sx[i][1] = __fmul_rn(xs[wm + i * 16 + g + 8], 256.f);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            sw[j][0] = wsc[buf][wn + j * 8 + 2 * cq];
            sw[j][1] = wsc[buf][wn + j * 8 + 2 * cq + 1];
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float f = __fmul_rn(sx[i][e >> 1], sw[j][e & 1]);
                    acc[i][j][e] = fmaf((float)(hh[i][j][e] * 256 + cr[i][j][e]), f, acc[i][j][e]);
                }
            }
        }
        if (more) {
            __syncthreads();            /* (x's buffer read: its next group goes in) */
            storex();
            cp_async_wait0();
            __syncthreads();
            buf ^= 1;
        }
    }
    auto put = [&](int r, int c, float v0) {
        if (r >= n || c >= m) {
            return;
        }
        float v = __fadd_rn(v0, b ? b[c] : 0.f);
        if (res != nullptr) {
            v = __fadd_rn(v, res[(size_t)r * m + c]);
        }
        if (slope != nullptr) {
            v = v > 0.f ? v : __fmul_rn(v, slope[c]);
        }
        if (!GATHER) {
            v = fminf(fmaxf(v, omin), omax);
        }
        if (scat != nullptr) {
            const int d = scat[(size_t)r * (m / blk) + c / blk];
            if (d >= 0) {
                y[(size_t)d * blk + c % blk] = v;
            }
        } else {
            y[(size_t)r * m + c] = v;
        }
    };
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + wn + j * 8 + 2 * cq, r = r0 + wm + i * 16 + g;
            put(r, c, acc[i][j][0]);
            put(r, c + 1, acc[i][j][1]);
            put(r + 8, c, acc[i][j][2]);
            put(r + 8, c + 1, acc[i][j][3]);
        }
    }
}

/* ENC_LINEAR of W16I rows (wbf 8: encoders.runtime.to_w16 "i16") on the tensor cores: k_enc_conv_i16 without the
 * gather (k a multiple of 64, x's rows 16-byte aligned) */
void enc_linear_w16i(const float *x, const uint8_t *w, const float *b, float *y, int n, int m, int k, float imin,
                     float imax, float omin, float omax, cudaStream_t s)
{
    const dim3 grid((unsigned)((m + CI_N - 1) / CI_N), (unsigned)((n + CI_M - 1) / CI_M));
    k_enc_conv_i16<0, false><<<grid, 256, 0, s>>>(x, nullptr, w, b, nullptr, nullptr, y, n, m, 1, k, nullptr, 1,
                                                  imin, imax, omin, omax);
}

/* ENC_LINEAR of F20 rows on the tensor cores (wbf 5): x (with the input clamps) and W's F20 values (decoded as
 * they are loaded) rounded to bfloat16, mma.sync m16n8k16 with float32 sums, + b, the output clamps. Not the bits
 * of k_enc_sgemm (wbf 3): the products of bfloat16 values (SeedVR2's official DiT runs under bfloat16 autocast).
 * k_enc_conv_tc's tiles (128 x 128, 8 warps of 64 x 32, k in steps of 32: k a multiple of 32). */
__global__ void __launch_bounds__(256) k_enc_gemm_f20tc(const float *x, const uint8_t *w, const float *b, float *y,
                                                        int n, int m, int k, float imin, float imax, float omin,
                                                        float omax)
{
    PDL_START();
    __shared__ __align__(16) uint16_t as_[2][CTC_M][CTC_K + 8];
    __shared__ __align__(16) uint16_t bs[2][CTC_N][CTC_K + 8];
    const int steps = k / CTC_K;
    const int r0 = blockIdx.y * CTC_M, c0 = blockIdx.x * CTC_N;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int wm = (warp / 4) * 64, wn = (warp % 4) * 32, g = lane / 4, cq = lane % 4;
    const int lr = tid / 2, half = (tid % 2) * 16, xrow = r0 + lr, wcol = c0 + lr;
    const float *xp = x + (size_t)(xrow < n ? xrow : 0) * k + half;
    const uint8_t *wrow = w + (size_t)(wcol < m ? wcol : 0) * f20_row_bytes(k);
    uint32_t xr[8], wr[8];
    auto load = [&](int st) {
        const int k0 = st * CTC_K;
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            float4 a = xrow < n ? *(const float4 *)(xp + k0 + 4 * q) : make_float4(0.f, 0.f, 0.f, 0.f);
            xr[2 * q] = enc_bf2(fminf(fmaxf(a.x, imin), imax), fminf(fmaxf(a.y, imin), imax));
            xr[2 * q + 1] = enc_bf2(fminf(fmaxf(a.z, imin), imax), fminf(fmaxf(a.w, imin), imax));
        }
        float wv[16];
        if (wcol < m) {
            const F20Words8 f0 = f20_load8(wrow, k, k0 + half), f1 = f20_load8(wrow, k, k0 + half + 8);
            f20_dec8(f0, wv);
            f20_dec8(f1, wv + 8);
        } else {
            #pragma unroll
            for (int q = 0; q < 16; ++q) {
                wv[q] = 0.f;
            }
        }
        #pragma unroll
        for (int q = 0; q < 8; ++q) {
            wr[q] = enc_bf2(wv[2 * q], wv[2 * q + 1]);
        }
    };
    auto store = [&](int buf) {
        *(uint4 *)&as_[buf][lr][half] = make_uint4(xr[0], xr[1], xr[2], xr[3]);
        *(uint4 *)&as_[buf][lr][half + 8] = make_uint4(xr[4], xr[5], xr[6], xr[7]);
        *(uint4 *)&bs[buf][lr][half] = make_uint4(wr[0], wr[1], wr[2], wr[3]);
        *(uint4 *)&bs[buf][lr][half + 8] = make_uint4(wr[4], wr[5], wr[6], wr[7]);
    };
    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
        }
    }
    load(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int st = 0; st < steps; ++st) {
        const bool more = st + 1 < steps;
        if (more) {
            load(st + 1);
        }
        #pragma unroll
        for (int ks = 0; ks < CTC_K / 16; ++ks) {
            const int kk = ks * 16;
            uint32_t bf[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const uint16_t *br = &bs[buf][wn + j * 8 + g][kk + 2 * cq];
                bf[j][0] = *(const uint32_t *)br;
                bf[j][1] = *(const uint32_t *)(br + 8);
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int ra = wm + i * 16;
                uint32_t a[4];
                a[0] = *(const uint32_t *)&as_[buf][ra + g][kk + 2 * cq];
                a[1] = *(const uint32_t *)&as_[buf][ra + g + 8][kk + 2 * cq];
                a[2] = *(const uint32_t *)&as_[buf][ra + g][kk + 2 * cq + 8];
                a[3] = *(const uint32_t *)&as_[buf][ra + g + 8][kk + 2 * cq + 8];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    mma16816_bf16_enc(acc[i][j], a, bf[j]);
                }
            }
        }
        if (more) {
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int c = c0 + wn + j * 8 + 2 * cq, r = r0 + wm + i * 16 + g;
            #pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int rr = r + (e >> 1) * 8, cc = c + (e & 1);
                if (rr < n && cc < m) {
                    const float v = acc[i][j][e] + (b ? b[cc] : 0.f);
                    y[(size_t)rr * m + cc] = fminf(fmaxf(v, omin), omax);
                }
            }
        }
    }
}

/* ENC_CONV_MM of the narrow outputs (m = BN: 32 or 64; cin a multiple of KC; no scat): one column tile
 * of all of m, 256 rows a block, each thread NTM x NTN of y (8 x 4 or 8 x 8: three or four shared loads
 * of 128 bits for 32 or 64 FMAs, against k_enc_conv_mm's 8 x 2 / 8 x 4). A chunk of KC (8 or 16) of k
 * is one tap, so a row's gathered values of a chunk are KC / 4 float4 loads. The same bits as
 * k_enc_conv_mm: each value fmaf over k in order from 0, then its epilogue. */
template <int NTM, int NTN, int BN, int KC>
__global__ void __launch_bounds__(256, 2) k_enc_conv_mm_nar(const float *src, const int *idx, const float *w,
                                                         const float *b, const float *res, const float *slope,
                                                         float *y, int n, int taps, int cin)
{
    PDL_START();
    constexpr int CT = BN / NTN, BM = NTM * (256 / CT), XV = KC / 4, WV = BN * KC / 4;
    static_assert(BM == 256, "256 rows a block");
    __shared__ __align__(16) float xs[2][KC][BM];
    __shared__ __align__(16) float ws[2][KC][BN];
    const int k = taps * cin, m = BN;
    const int r0 = blockIdx.x * BM;
    const int tid = threadIdx.x, tx = tid % CT, ty = tid / CT;
    float acc[NTM][NTN];
    #pragma unroll
    for (int i = 0; i < NTM; ++i) {
        #pragma unroll
        for (int j = 0; j < NTN; ++j) {
            acc[i][j] = 0.f;
        }
    }
    float4 xv[XV], wv[(WV + 255) / 256];
    auto load = [&](int k0) {
        const int r = r0 + tid;
        int sr = -1;
        if (r < n) {
            const int t = k0 / cin;
            sr = idx[(size_t)r * taps + t];
        }
        const float4 *p = (const float4 *)(src + (size_t)(sr < 0 ? 0 : sr) * cin + (k0 % cin));
        #pragma unroll
        for (int q = 0; q < XV; ++q) {
            xv[q] = sr >= 0 ? p[q] : make_float4(0.f, 0.f, 0.f, 0.f);
        }
        #pragma unroll
        for (int q = 0; q < (WV + 255) / 256; ++q) {
            const int f = tid + q * 256;
            if (f < WV) {
                wv[q] = *(const float4 *)(w + (size_t)(f / XV) * k + k0 + (f % XV) * 4);
            }
        }
    };
    auto store = [&](int buf) {
        #pragma unroll
        for (int q = 0; q < XV; ++q) {
            xs[buf][4 * q][tid] = xv[q].x; xs[buf][4 * q + 1][tid] = xv[q].y;
            xs[buf][4 * q + 2][tid] = xv[q].z; xs[buf][4 * q + 3][tid] = xv[q].w;
        }
        #pragma unroll
        for (int q = 0; q < (WV + 255) / 256; ++q) {
            const int f = tid + q * 256;
            if (f < WV) {
                const int c = f / XV, h = (f % XV) * 4;
                ws[buf][h][c] = wv[q].x; ws[buf][h + 1][c] = wv[q].y; ws[buf][h + 2][c] = wv[q].z;
                ws[buf][h + 3][c] = wv[q].w;
            }
        }
    };
    load(0);
    store(0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < k; k0 += KC) {
        const bool more = k0 + KC < k;
        if (more) {
            load(k0 + KC);
        }
        #pragma unroll
        for (int kk = 0; kk < KC; ++kk) {
            float a[NTM], bv[NTN];
            #pragma unroll
            for (int i = 0; i < NTM; i += 4) {
                float4 v = *(const float4 *)&xs[buf][kk][ty * NTM + i];
                a[i] = v.x; a[i + 1] = v.y; a[i + 2] = v.z; a[i + 3] = v.w;
            }
            #pragma unroll
            for (int j = 0; j < NTN; j += 4) {
                float4 v = *(const float4 *)&ws[buf][kk][tx * NTN + j];
                bv[j] = v.x; bv[j + 1] = v.y; bv[j + 2] = v.z; bv[j + 3] = v.w;
            }
            #pragma unroll
            for (int i = 0; i < NTM; ++i) {
                #pragma unroll
                for (int j = 0; j < NTN; ++j) {
                    acc[i][j] = fmaf(a[i], bv[j], acc[i][j]);
                }
            }
        }
        if (more) {
            store(buf ^ 1);
            __syncthreads();
            buf ^= 1;
        }
    }
    #pragma unroll
    for (int i = 0; i < NTM; ++i) {
        const int r = r0 + ty * NTM + i;
        if (r >= n) {
            continue;
        }
        #pragma unroll
        for (int j = 0; j < NTN; ++j) {
            const int c = tx * NTN + j;
            float v = __fadd_rn(acc[i][j], b ? b[c] : 0.f);
            if (res != nullptr) {
                v = __fadd_rn(v, res[(size_t)r * m + c]);
            }
            if (slope != nullptr) {
                v = v > 0.f ? v : __fmul_rn(v, slope[c]);
            }
            y[(size_t)r * m + c] = v;
        }
    }
}

/* ENC_CONV_MM's launch: rows of 128 (64 when there are too few blocks for the SMs), the narrowest
 * column tile of 16, 32, 64 or 128 that wastes the fewest columns of m. m 32 or 64 with cin a
 * multiple of 8 and no scat: k_enc_conv_mm_nar (RECLANK_CONV_MM2=0: not); tall tiles of 128 columns with cin a
 * multiple of 16: k_enc_conv_mm16 (RECLANK_CONV_MM16=0: not). wbf 1: bfloat16 W, k_enc_conv_tc. */
static void enc_conv_mm(const float *src, const int *idx, const float *w, const float *b, const float *res,
                        const float *slope, float *y, int n, int m, int taps, int cin, const int *scat, int blk,
                        int sms, cudaStream_t s, int wbf)
{
    if (wbf) {
        /* 16-bit W on the tensor cores (the program gives it only with cin a multiple of 32): wbf 1 bfloat16 W,
         * 2 and x's rows bfloat16 (uint16); 3 W16H (float16) rows, 4 and x's rows float16; 5 W16I (int16 with
         * scales) rows, 6 and x's rows I16 */
        const dim3 grid((unsigned)((m + CTC_N - 1) / CTC_N), (unsigned)((n + CTC_M - 1) / CTC_M));
        const uint16_t *w16 = (const uint16_t *)w;
        const float inf = INFINITY;
#define CONV_TC(XM, H) k_enc_conv_tc<XM, H, true><<<grid, 256, 0, s>>>(src, idx, w16, b, res, slope, y, n, m, taps, \
                                                                       cin, scat, blk, -inf, inf, -inf, inf)
        if (wbf == 5 || wbf == 6) {
            /* W16I rows: int16 with scales (cin a multiple of 64); 6: x's rows I16 */
            const dim3 gi((unsigned)((m + CI_N - 1) / CI_N), (unsigned)((n + CI_M - 1) / CI_M));
            if (wbf == 5) {
                k_enc_conv_i16<0, true><<<gi, 256, 0, s>>>(src, idx, (const uint8_t *)w, b, res, slope, y, n, m, taps,
                                                           cin, scat, blk, -inf, inf, -inf, inf);
            } else {
                k_enc_conv_i16<1, true><<<gi, 256, 0, s>>>(src, idx, (const uint8_t *)w, b, res, slope, y, n, m, taps,
                                                           cin, scat, blk, -inf, inf, -inf, inf);
            }
        } else if (wbf == 1) {
            CONV_TC(0, false);
        } else if (wbf == 2) {
            CONV_TC(1, false);
        } else if (wbf == 3) {
            CONV_TC(0, true);
        } else {
            CONV_TC(1, true);
        }
#undef CONV_TC
        return;
    }
    static int nar = -1;
    if (nar < 0) {
        const char *e = getenv("RECLANK_CONV_MM2");
        nar = e ? atoi(e) != 0 : 1;
    }
    static int kc = -1;
    if (kc < 0) {
        const char *e = getenv("RECLANK_CONV_MM2_KC");
        kc = e ? atoi(e) : 16;
    }
    if (nar && scat == nullptr && (m == 32 || m == 64) && cin % 8 == 0) {
        const unsigned gx = (unsigned)((n + 255) / 256);
#define ECN(TN_, BN_, KC_) k_enc_conv_mm_nar<8, TN_, BN_, KC_><<<gx, 256, 0, s>>>(src, idx, w, b, res, slope, y, n, \
                                                                               taps, cin)
        const bool k16 = kc == 16 && cin % 16 == 0;
        if (m == 32) {
            if (k16) ECN(4, 32, 16); else ECN(4, 32, 8);
        } else {
            if (k16) ECN(8, 64, 16); else ECN(8, 64, 8);
        }
#undef ECN
        return;
    }
    int bn = 128;
    long bw = -1;
    for (int c : {128, 64, 32, 16}) {          /* (a tie keeps the wider tile) */
        const long waste = (long)((m + c - 1) / c) * c - m;
        if (bw < 0 || waste < bw) {
            bw = waste;
            bn = c;
        }
    }
    const bool tall = (long)((n + 127) / 128) * ((m + bn - 1) / bn) >= sms;
    const unsigned gx = (unsigned)((m + bn - 1) / bn);
#define ECM(ETM, ETN) k_enc_conv_mm<ETM, ETN><<<dim3(gx, (unsigned)((n + 16 * ETM - 1) / (16 * ETM))), 256, 0, s>>>( \
        src, idx, w, b, res, slope, y, n, m, taps, cin, scat, blk)
    static int k16 = -1;
    if (k16 < 0) {
        const char *e = getenv("RECLANK_CONV_MM16");
        k16 = e ? atoi(e) != 0 : 1;
    }
    if (tall && bn == 128 && k16 && cin % 16 == 0) {
        k_enc_conv_mm16<<<dim3(gx, (unsigned)((n + 127) / 128)), 256, 0, s>>>(src, idx, w, b, res, slope, y, n, m,
                                                                              taps, cin, scat, blk);
    } else if (tall) {
        if (bn == 16) ECM(8, 1); else if (bn == 32) ECM(8, 2); else if (bn == 64) ECM(8, 4); else ECM(8, 8);
    } else {
        if (bn == 16) ECM(4, 1); else if (bn == 32) ECM(4, 2); else if (bn == 64) ECM(4, 4); else ECM(4, 8);
    }
#undef ECM
}

/* ENC_CONV3: a 3 x 3 conv (zero padding) of an H x W image of pixel rows by direct tiles: a block
 * 8 x 32 pixels of all M outputs; for each chunk of 8 input channels the tile with its border (10 x
 * 34 pixels) and the chunk's 9 x 8 x M weights in shared memory, then each thread 8 pixels of a row
 * times NTN outputs (each input value read once a block, against 9 times from L2 / DRAM by the
 * implicit im2col of ENC_CONV_MM). The input channels are ng groups (offset, row stride, channels:
 * a multiple of 8) of x in order (Real-ESRGAN's dense blocks read their concatenation in place);
 * up: x is the H / 2 x W / 2 image, read upsampled 2x nearest. W rows (M, 9 cin) in im2col's order
 * (tap, channel). Each value: the sum over the chunks, then the channels, then the taps (float32,
 * fmaf; not ENC_CONV_MM's order), + b, then x rs + res when res is not NULL, then LeakyReLU of slope
 * when slope is not NULL. */
#define EC3_TH 8
#define EC3_TW 32
#define EC3_CK 8
template <int NTN, int M>
__global__ void __launch_bounds__(256, 2) k_enc_conv3(const float *x, const int *groups, const float *w,
                                                      const float *b, const float *res, const float *slope,
                                                      float *y, int H, int W, int cin, int up, int ng, float rs)
{
    PDL_START();
    constexpr int HP = EC3_TH + 2, WP = EC3_TW + 2, CG = M / NTN;
    static_assert(CG == 8, "eight column groups");
    __shared__ __align__(16) float xsm[EC3_CK][HP][WP];
    __shared__ __align__(16) float wsm[EC3_CK][9][M];
    const int tid = threadIdx.x, cg = tid % CG, pg = tid / CG;
    const int ty = pg / 4, xs0 = (pg % 4) * 8;
    const int X0 = blockIdx.x * EC3_TW, Y0 = blockIdx.y * EC3_TH;
    const int sw = up ? W / 2 : W;
    float acc[8][NTN];
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        #pragma unroll
        for (int j = 0; j < NTN; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int c0 = 0; c0 < cin; c0 += EC3_CK) {
        int off = 0, ld = 0, gc = c0;
        for (int g = 0; g < ng; ++g) {
            if (gc < groups[3 * g + 2]) {
                off = groups[3 * g];
                ld = groups[3 * g + 1];
                break;
            }
            gc -= groups[3 * g + 2];
        }
        __syncthreads();
        for (int e = tid; e < HP * WP * 2; e += 256) {
            const int pix = e >> 1, half = e & 1, hy = pix / WP, hx = pix - hy * WP;
            const int Y = Y0 + hy - 1, X = X0 + hx - 1;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (Y >= 0 && Y < H && X >= 0 && X < W) {
                const long sp = up ? (long)(Y >> 1) * sw + (X >> 1) : (long)Y * W + X;
                v = *(const float4 *)(x + off + sp * ld + gc + half * 4);
            }
            xsm[half * 4][hy][hx] = v.x;
            xsm[half * 4 + 1][hy][hx] = v.y;
            xsm[half * 4 + 2][hy][hx] = v.z;
            xsm[half * 4 + 3][hy][hx] = v.w;
        }
        for (int e = tid; e < M * 18; e += 256) {
            const int t = (e / M) >> 1, half = (e / M) & 1, o = e % M;
            const float4 v = *(const float4 *)(w + (size_t)o * 9 * cin + t * cin + c0 + half * 4);
            wsm[half * 4][t][o] = v.x;
            wsm[half * 4 + 1][t][o] = v.y;
            wsm[half * 4 + 2][t][o] = v.z;
            wsm[half * 4 + 3][t][o] = v.w;
        }
        __syncthreads();
        #pragma unroll 2
        for (int cc = 0; cc < EC3_CK; ++cc) {
            #pragma unroll
            for (int dy = 0; dy < 3; ++dy) {
                float a[10];
                #pragma unroll
                for (int q = 0; q < 10; ++q) {
                    a[q] = xsm[cc][ty + dy][xs0 + q];
                }
                #pragma unroll
                for (int dx = 0; dx < 3; ++dx) {
                    float bv[NTN];
                    #pragma unroll
                    for (int j = 0; j < NTN; j += 4) {
                        const float4 v = *(const float4 *)&wsm[cc][dy * 3 + dx][cg * NTN + j];
                        bv[j] = v.x; bv[j + 1] = v.y; bv[j + 2] = v.z; bv[j + 3] = v.w;
                    }
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        #pragma unroll
                        for (int j = 0; j < NTN; ++j) {
                            acc[i][j] = fmaf(a[i + dx], bv[j], acc[i][j]);
                        }
                    }
                }
            }
        }
    }
    const int Y = Y0 + ty;
    if (Y >= H) {
        return;
    }
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int X = X0 + xs0 + i;
        if (X >= W) {
            continue;
        }
        const size_t p = (size_t)Y * W + X;
        #pragma unroll
        for (int j = 0; j < NTN; ++j) {
            const int o = cg * NTN + j;
            float v = __fadd_rn(acc[i][j], b ? b[o] : 0.f);
            if (res != nullptr) {
                v = __fadd_rn(__fmul_rn(v, rs), res[p * M + o]);
            }
            if (slope != nullptr) {
                v = v > 0.f ? v : __fmul_rn(v, slope[o]);
            }
            y[p * M + o] = v;
        }
    }
}

static void enc_conv3(const float *x, const int *groups, const float *w, const float *b, const float *res,
                      const float *slope, float *y, int H, int W, int cin, int m, int up, int ng, float rs,
                      cudaStream_t s)
{
    const dim3 grid((unsigned)((W + EC3_TW - 1) / EC3_TW), (unsigned)((H + EC3_TH - 1) / EC3_TH));
    if (m == 32) {
        k_enc_conv3<4, 32><<<grid, 256, 0, s>>>(x, groups, w, b, res, slope, y, H, W, cin, up, ng, rs);
    } else {
        k_enc_conv3<8, 64><<<grid, 256, 0, s>>>(x, groups, w, b, res, slope, y, H, W, cin, up, ng, rs);
    }
}

/* y = x / sqrt(mean(x^2) + eps), times w when w is not NULL. One warp for
 * each row of d values. y may be x. */
__global__ void k_enc_rms(const float *x, const float *w, float *y, int rows, int d, float eps)
{
    PDL_START();
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const float *p = x + (size_t)row * d;
    float s = 0.f;
    for (int i = lane; i < d; i += 32) {
        s += p[i] * p[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, off);
    }
    float r = 1.0f / sqrtf(s / (float)d + eps);
    for (int i = lane; i < d; i += 32) {
        y[(size_t)row * d + i] = p[i] * r * (w ? w[i] : 1.0f);
    }
}

/* The LayerNorm of each row of d values, times w, plus b (either may be
 * NULL). One warp for each row. y may be x. */
__global__ void k_enc_lnorm(const float *x, const float *w, const float *b, float *y, int rows,
                            int d, float eps)
{
    PDL_START();
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const float *p = x + (size_t)row * d;
    float s = 0.f;
    #pragma unroll 8
    for (int i = lane; i < d; i += 32) {
        s += p[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, off);
    }
    float mean = s / (float)d, v = 0.f;
    #pragma unroll 8
    for (int i = lane; i < d; i += 32) {
        float t = p[i] - mean;
        v += t * t;
    }
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(RC_FULL_MASK, v, off);
    }
    float r = 1.0f / sqrtf(v / (float)d + eps);
    #pragma unroll 8
    for (int i = lane; i < d; i += 32) {
        y[(size_t)row * d + i] = (p[i] - mean) * r * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
}

/* k_enc_lnorm with a block of 256 threads a row (few rows: a decoder's step,
 * where a warp a row waits on its loads three times), the values of x, w and
 * b in registers: d at most 256 ELN_V. */
#define ELN_V 8
__global__ void __launch_bounds__(256) k_enc_lnorm_row(const float *x, const float *w, const float *b, float *y,
                                                      int d, float eps)
{
    PDL_START();
    __shared__ float red[8];
    int row = blockIdx.x, t = threadIdx.x, warp = t / 32, lane = t % 32;
    const float *p = x + (size_t)row * d;
    float xv[ELN_V], wv[ELN_V], bv[ELN_V];
    float s = 0.f;
    #pragma unroll
    for (int a = 0; a < ELN_V; ++a) {
        int i = t + 256 * a;
        bool in = i < d;
        xv[a] = in ? p[i] : 0.f;
        wv[a] = in && w ? w[i] : 1.f;
        bv[a] = in && b ? b[i] : 0.f;
        s += xv[a];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, off);
    }
    if (lane == 0) {
        red[warp] = s;
    }
    __syncthreads();
    s = 0.f;
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        s += red[a];
    }
    float mean = s / (float)d, v = 0.f;
    #pragma unroll
    for (int a = 0; a < ELN_V; ++a) {
        float u = t + 256 * a < d ? xv[a] - mean : 0.f;
        v += u * u;
    }
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(RC_FULL_MASK, v, off);
    }
    __syncthreads();
    if (lane == 0) {
        red[warp] = v;
    }
    __syncthreads();
    v = 0.f;
    #pragma unroll
    for (int a = 0; a < 8; ++a) {
        v += red[a];
    }
    float r = 1.0f / sqrtf(v / (float)d + eps);
    #pragma unroll
    for (int a = 0; a < ELN_V; ++a) {
        int i = t + 256 * a;
        if (i < d) {
            y[(size_t)row * d + i] = (xv[a] - mean) * r * wv[a] + bv[a];
        }
    }
}

/* y = gelu(x), n values: the tanh form, or the erf form (use_erf). */
__global__ void k_enc_gelu(const float *x, float *y, int64_t n, int use_erf)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = use_erf ? 0.5f * v * (1.0f + erff(v * 0.7071067811865476f))
                       : 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

/* y = gelu(g) * u, n values: the tanh form, or the erf form (use_erf). */
__global__ void k_enc_gelu_mul(const float *g, const float *u, float *y, int64_t n, int use_erf)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        float t = use_erf ? erff(v * 0.7071067811865476f)
                          : tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v));
        y[i] = 0.5f * v * (1.0f + t) * u[i];
    }
}

/* y = silu(g) * u, n values. */
__global__ void k_enc_silu_mul(const float *g, const float *u, float *y, int64_t n)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        y[i] = v / (1.0f + expf(-v)) * u[i];
    }
}

/* x += sc y, n values. */
__global__ void k_enc_add(float *x, const float *y, int64_t n, float sc)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += sc * y[i];
    }
}

/* y = clamp(x, lo, hi), n values. */
__global__ void k_enc_clamp(const float *x, float *y, int64_t n, float lo, float hi)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fminf(fmaxf(x[i], lo), hi);
    }
}

/* y = clamp(y + b, lo, hi) in place; b may be NULL. */
__global__ void k_enc_bias_clamp(float *y, const float *b, int rows, int cols, float lo, float hi)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        y[i] = fminf(fmaxf(y[i] + (b ? b[i % cols] : 0.f), lo), hi);
    }
}

/* y = x sigmoid(x), n values. */
__global__ void k_enc_silu(const float *x, float *y, int64_t n)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* y = x vec for each row of cols values. */
__global__ void k_enc_mul_vec(const float *x, const float *vec, float *y, int rows, int cols)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        y[i] = x[i] * vec[i % cols];
    }
}

/* y (rows, cols) = a sigmoid(b), a and b the halves of each row of x. */
__global__ void k_enc_glu(const float *x, float *y, int rows, int cols)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        int64_t row = i / cols;
        int c = (int)(i % cols);
        const float *xa = x + row * 2 * cols;
        y[i] = xa[c] / (1.0f + expf(-xa[cols + c]));
    }
}

/* The causal depthwise conv: y[i, c] = sum_j w[c, j] x[i - kw + 1 + j, c]. */
__global__ void k_enc_dwconv(const float *x, const float *w, float *y, int t, int c, int kw)
{
    PDL_START();
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)t * c) {
        return;
    }
    int i = (int)(idx / c), ch = (int)(idx % c);
    float acc = 0.f;
    for (int j = 0; j < kw; ++j) {
        int src = i - kw + 1 + j;
        if (src >= 0) {
            acc += w[(size_t)ch * kw + j] * x[(size_t)src * c + ch];
        }
    }
    y[idx] = acc;
}

/* The local attention of gemma4a: query i sees the keys i - span + 1 .. i;
 * key j of that list adds q r[1 + j]; the scores get the soft cap, and an
 * invalid key -1e9. One warp for each (query, head); lane j < span scores
 * key j, then each lane sums its values of the output. */
__global__ void k_enc_local_attn(const float *q, const float *k, const float *v, const float *r,
                                 const int *valid, float *o, int t, int heads, int hd, int span,
                                 float cap)
{
    PDL_START();
    int64_t row = (int64_t)blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= (int64_t)t * heads) {
        return;
    }
    int i = (int)(row / heads), h = (int)(row % heads);
    const float *qi = q + row * hd;
    int kp = i - span + 1 + lane;
    float sc = -INFINITY;
    if (lane < span) {
        sc = -1e9f;
        if (kp >= 0 && valid[kp]) {
            const float *kk = k + ((size_t)kp * heads + h) * hd;
            const float *rj = r + ((size_t)(1 + lane) * heads + h) * hd;
            float a = 0.f;
            for (int d = 0; d < hd; ++d) {
                a += qi[d] * (kk[d] + rj[d]);
            }
            sc = tanhf(a / cap) * cap;
        }
    }
    float m = sc;
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, off));
    }
    float p = lane < span ? expf(sc - m) : 0.f;
    float sum = p;
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, off);
    }
    p /= sum;
    for (int d0 = 0; d0 < hd; d0 += 32) {
        /* every lane takes each pass, so the shuffles agree */
        int d = d0 + lane;
        float acc = 0.f;
        for (int j = 0; j < span; ++j) {
            float pj = __shfl_sync(RC_FULL_MASK, p, j);
            int kj = i - span + 1 + j;
            if (kj >= 0 && d < hd) {
                acc += pj * v[((size_t)kj * heads + h) * hd + d];
            }
        }
        if (d < hd) {
            o[row * hd + d] = acc;
        }
    }
}

/* The axial 2D RoPE of gemma4v, in place on x (n, heads, hd): the first half
 * of each head turns with the x position and the second with the y
 * position; each half is a NEOX rotation (its first quarter with its second)
 * at the angles pos * inv[j], j < hd / 4. pos is (n, 2) int32. */
__global__ void k_enc_rope2d(float *x, const int *pos, const float *inv, int n, int heads, int hd)
{
    PDL_START();
    int q4 = hd / 4, h2 = hd / 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n * heads * 2 * q4;
    if (i >= total) {
        return;
    }
    int j = (int)(i % q4);
    int part = (int)((i / q4) % 2);
    int64_t row = i / (2 * q4);          /* patch * heads + head */
    int patch = (int)(row / heads);
    float ang = (float)pos[2 * patch + part] * inv[j];
    float c = cosf(ang), s = sinf(ang);
    float *p = x + row * hd + part * h2;
    float a = p[j], b = p[j + q4];
    p[j] = a * c - b * s;
    p[j + q4] = b * c + a * s;
}

/* The attention of each query over the g rows of its window (rows w g to
 * w g + g), the scores times scale plus bias[wt[w], head] (types, heads, g,
 * g): Swin's window attention with its relative position bias and the mask of
 * a shifted window. q k v o are (n, heads, hd). One block of 64 threads for
 * each (window, head); thread i takes query i (g at most 64, hd at most 128);
 * K and V of the window in shared memory. */
__global__ void k_enc_attn_bias(const float *q, const float *k, const float *v, float *o, const float *bias,
                                const int *wt, int heads, int hd, int g, float scale)
{
    PDL_START();
    extern __shared__ float esm[];
    float *ks = esm, *vs = esm + g * hd;
    int w = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    int64_t stride = (int64_t)heads * hd;
    for (int t = threadIdx.x; t < g * hd; t += blockDim.x) {
        int j = t / hd, d = t % hd;
        int64_t at = ((int64_t)w * g + j) * stride + (int64_t)h * hd + d;
        ks[t] = k[at];
        vs[t] = v[at];
    }
    __syncthreads();
    if (i >= g) {
        return;
    }
    const float *qi = q + ((int64_t)w * g + i) * stride + (int64_t)h * hd;
    float qr[128];
    for (int d = 0; d < hd; ++d) {
        qr[d] = qi[d];
    }
    const float *bb = bias + (((int64_t)wt[w] * heads + h) * g + i) * g;
    float s[64], mx = -INFINITY, sum = 0.0f;
    for (int j = 0; j < g; ++j) {
        float acc = 0.0f;
        for (int d = 0; d < hd; ++d) {
            acc += qr[d] * ks[j * hd + d];
        }
        s[j] = acc * scale + bb[j];
        mx = fmaxf(mx, s[j]);
    }
    for (int j = 0; j < g; ++j) {
        s[j] = expf(s[j] - mx);
        sum += s[j];
    }
    float inv = 1.0f / sum;
    float *oi = o + ((int64_t)w * g + i) * stride + (int64_t)h * hd;
    for (int d = 0; d < hd; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < g; ++j) {
            acc += s[j] * vs[j * hd + d];
        }
        oi[d] = acc * inv;
    }
}

/* GroupNorm (ENC_GNORM) in three kernels: k_enc_gnorm_part, a block of
 * EGN_R rows of a group: its sums of x and x^2 (double) to part (groups x
 * chunks x 2); k_enc_gnorm_stats, a block a group: its mean and rstd (float)
 * after the parts; k_enc_gnorm_apply, a thread a value. */
#define EGN_R 256
__global__ void k_enc_gnorm_part(const float *x, int64_t rows, int cols, int cg, double *part)
{
    PDL_START();
    __shared__ double rs[8], rq[8];
    int g = blockIdx.x, ch = blockIdx.y, nch = gridDim.y;
    int64_t i = (int64_t)ch * EGN_R + threadIdx.x;
    double s = 0.0, q = 0.0;
    if (i < rows) {
        const float *p = x + i * cols + (int64_t)g * cg;
        for (int c = 0; c < cg; ++c) {
            double v = p[c];
            s += v;
            q += v * v;
        }
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, off);
        q += __shfl_xor_sync(RC_FULL_MASK, q, off);
    }
    if (threadIdx.x % 32 == 0) {
        rs[threadIdx.x / 32] = s;
        rq[threadIdx.x / 32] = q;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < EGN_R / 32; ++w) {
            s += rs[w];
            q += rq[w];
        }
        part[((size_t)g * nch + ch) * 2] = s;
        part[((size_t)g * nch + ch) * 2 + 1] = q;
    }
}

__global__ void k_enc_gnorm_stats(const double *part, int nch, double count, float eps, float *stats)
{
    PDL_START();
    __shared__ double rs[8], rq[8];
    int g = blockIdx.x;
    double s = 0.0, q = 0.0;
    for (int ch = threadIdx.x; ch < nch; ch += blockDim.x) {
        s += part[((size_t)g * nch + ch) * 2];
        q += part[((size_t)g * nch + ch) * 2 + 1];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, off);
        q += __shfl_xor_sync(RC_FULL_MASK, q, off);
    }
    if (threadIdx.x % 32 == 0) {
        rs[threadIdx.x / 32] = s;
        rq[threadIdx.x / 32] = q;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < (int)blockDim.x / 32; ++w) {
            s += rs[w];
            q += rq[w];
        }
        double mean = s / count, var = q / count - mean * mean;
        stats[2 * g] = (float)mean;
        stats[2 * g + 1] = (float)(1.0 / sqrt((var > 0.0 ? var : 0.0) + (double)eps));
    }
}

__global__ void k_enc_gnorm_apply(const float *x, const float *w, const float *b, float *y, int64_t n, int cols,
                                  int cg, const float *stats, int silu)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    int c = (int)(t % cols), g = c / cg;
    float v = (x[t] - stats[2 * g]) * stats[2 * g + 1] * (w ? w[c] : 1.f) + (b ? b[c] : 0.f);
    y[t] = silu ? v / (1.f + expf(-v)) : v;
}

/* RoPE of the pairs (2j, 2j + 1) of each head of row i at the angle of (i,
 * j): cos and sin (n, hd / 2). A thread a pair. */
__global__ void k_enc_rope_tab(float *x, const float *cs, const float *sn, int64_t n, int heads, int hd, int neox)
{
    PDL_START();
    int h2 = hd / 2;
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * heads * h2) {
        return;
    }
    int j = (int)(t % h2);
    int64_t i = t / ((int64_t)heads * h2);
    /* the pair (2j, 2j + 1), or (j, j + hd / 2) of NEOX */
    float *p = neox ? x + (t / h2) * hd + j : x + 2 * t;
    int other = neox ? h2 : 1;
    float c = cs[i * h2 + j], s = sn[i * h2 + j];
    float a = p[0], b = p[other];
    p[0] = a * c - b * s;
    p[other] = b * c + a * s;
}

/* y = x + (beta + 1e-9)^-1 sin(alpha x)^2, alpha and beta of the column: a thread a value. */
__global__ void k_enc_snake(const float *x, const float *al, const float *be, float *y, int64_t n, int cols)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    int c = (int)(t % cols);
    float v = x[t], s = sinf(al[c] * v);
    y[t] = v + 1.0f / (be[c] + 1e-9f) * (s * s);
}

/* y = x above 0, else expm1(x) (the ELU of alpha 1): a thread a value. */
__global__ void k_enc_elu(const float *x, float *y, int64_t n)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = v > 0.0f ? v : expm1f(v);
    }
}

/* y row idx[i] = x row i (none for an index < 0): n rows, one CUDA block each
 * (the new keys and values of a decoder's step into its cache). */
__global__ void k_enc_scatter(const float *x, const int *idx, float *y, int n, int cols)
{
    PDL_START();
    int i = blockIdx.x;
    if (i >= n || idx[i] < 0) {
        return;
    }
    const float *src = x + (int64_t)i * cols;
    float *dst = y + (int64_t)idx[i] * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[c] = src[c];
    }
}

/* The attention of query i over the rows [lo[i], hi[i]) of k and v (rows of
 * heads x hd; a cache of any rows), scale 1: a decoder's step. The keys of a
 * range in S parts of EKV_C: a block of EKV_T threads for each (query, head,
 * part) scores its keys (a key a warp, a lane hd / 32 of its values), then
 * thread t sums the values d = t % hd of the keys j = t / hd mod EKV_T / hd;
 * it writes its max, its sum of weights and its values (hd + 2 floats) to
 * part, and k_enc_attn_kv_join adds the S parts of a (query, head). hd a
 * multiple of 32 that divides EKV_T. */
#define EKV_T 128
#define EKV_C 128
template <int NI>
__global__ void __launch_bounds__(EKV_T) k_enc_attn_kv_part(const float *q, const float *k, const float *v,
                                                          const int *lo, const int *hi, int heads, float *part,
                                                          int kv_heads)
{
    PDL_START();
    constexpr int hd = 32 * NI, G = EKV_T / hd, NW = EKV_T / 32;
    __shared__ float sc[EKV_C];
    __shared__ float red[NW];
    __shared__ float acc_g[EKV_T];
    int i = blockIdx.x, h = blockIdx.y, S = gridDim.z;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    size_t stride = (size_t)heads * hd, kstride = (size_t)kv_heads * hd;
    int kh = h / (heads / kv_heads);
    float *out = part + (((size_t)i * heads + h) * S + blockIdx.z) * (hd + 2);
    int k0 = lo[i] + blockIdx.z * EKV_C, nk = min(EKV_C, hi[i] - k0);
    int d = threadIdx.x % hd, g = threadIdx.x / hd;
    if (nk <= 0) {
        if (threadIdx.x < hd) {
            out[2 + threadIdx.x] = 0.f;
        }
        if (threadIdx.x == 0) {
            out[0] = -INFINITY;
            out[1] = 0.f;
        }
        return;
    }
    const float *qi = q + i * stride + (size_t)h * hd;
    float qr[NI];
    #pragma unroll
    for (int a = 0; a < NI; ++a) {
        qr[a] = qi[lane + 32 * a];
    }
    float mt = -INFINITY;
    for (int j = warp; j < nk; j += NW) {
        const float *kj = k + (size_t)(k0 + j) * kstride + (size_t)kh * hd;
        float s = 0.f;
        #pragma unroll
        for (int a = 0; a < NI; ++a) {
            s += qr[a] * kj[lane + 32 * a];
        }
        for (int off = 16; off > 0; off >>= 1) {
            s += __shfl_xor_sync(RC_FULL_MASK, s, off);
        }
        if (lane == 0) {
            sc[j] = s;
        }
        mt = fmaxf(mt, s);
    }
    if (lane == 0) {
        red[warp] = mt;
    }
    __syncthreads();
    float mx = red[0];
    #pragma unroll
    for (int w = 1; w < NW; ++w) {
        mx = fmaxf(mx, red[w]);
    }
    __syncthreads();
    float ps = 0.f;
    for (int j = threadIdx.x; j < nk; j += EKV_T) {
        float p = expf(sc[j] - mx);
        sc[j] = p;
        ps += p;
    }
    for (int off = 16; off > 0; off >>= 1) {
        ps += __shfl_xor_sync(RC_FULL_MASK, ps, off);
    }
    if (lane == 0) {
        red[warp] = ps;
    }
    __syncthreads();
    float acc = 0.f;
    for (int j = g; j < nk; j += G) {
        acc += sc[j] * v[(size_t)(k0 + j) * kstride + (size_t)kh * hd + d];
    }
    acc_g[threadIdx.x] = acc;
    __syncthreads();
    if (g == 0) {
        float t = 0.f;
        #pragma unroll
        for (int b = 0; b < G; ++b) {
            t += acc_g[b * hd + d];
        }
        out[2 + d] = t;
    }
    if (threadIdx.x == 0) {
        float l = 0.f;
        #pragma unroll
        for (int w = 0; w < NW; ++w) {
            l += red[w];
        }
        out[0] = mx;
        out[1] = l;
    }
}

/* o of (query i, head h) = the S parts of k_enc_attn_kv_part, each scaled to
 * the largest max: a block of hd threads each. */
__global__ void k_enc_attn_kv_join(const float *part, float *o, int heads, int hd, int S)
{
    PDL_START();
    int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const float *p = part + ((size_t)i * heads + h) * S * (hd + 2);
    float mx = -INFINITY;
    for (int s = 0; s < S; ++s) {
        mx = fmaxf(mx, p[s * (hd + 2)]);
    }
    float l = 0.f, acc = 0.f;
    for (int s = 0; s < S; ++s) {
        const float *ps = p + s * (hd + 2);
        if (ps[0] == -INFINITY) {
            continue;
        }
        float f = expf(ps[0] - mx);
        l += f * ps[1];
        acc += f * ps[2 + d];
    }
    o[((size_t)i * heads + h) * hd + d] = l > 0.f ? acc / l : 0.f;
}
template __global__ void k_enc_attn_kv_part<1>(const float *, const float *, const float *, const int *,
                                               const int *, int, float *, int);
template __global__ void k_enc_attn_kv_part<2>(const float *, const float *, const float *, const int *,
                                               const int *, int, float *, int);
template __global__ void k_enc_attn_kv_part<4>(const float *, const float *, const float *, const int *,
                                               const int *, int, float *, int);

/* y (NR, m) = clamp(clamp(x, imin, imax) W^T + b, omin, omax) of NR rows of
 * x and float16 W (wbf 2; the products of a decoder's step, k_enc_gemm's
 * values in another order): a warp a row of W, a lane 8 values of it a step
 * (16 bytes), the rows of x from L1. k a multiple of 8, x and W aligned to 16
 * bytes. */
template <int NR>
__global__ void __launch_bounds__(256) k_enc_gemv_f16(const float *x, const __half *w, const float *b, float *y,
                                                    int m, int k, float imin, float imax, float omin, float omax)
{
    PDL_START();
    int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= m) {
        return;
    }
    const uint4 *wr = (const uint4 *)(w + (size_t)row * k);
    float acc[NR];
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        acc[r] = 0.f;
    }
    int k8 = k / 8;
    #pragma unroll 4
    for (int j = lane; j < k8; j += 32) {
        uint4 u = __ldg(wr + j);
        const __half2 *h2 = (const __half2 *)&u;
        float wf[8];
        #pragma unroll
        for (int a = 0; a < 4; ++a) {
            float2 f = __half22float2(h2[a]);
            wf[2 * a] = f.x;
            wf[2 * a + 1] = f.y;
        }
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            const float4 *xr = (const float4 *)(x + (size_t)r * k) + 2 * j;
            float4 p0 = xr[0], p1 = xr[1];
            float xv[8] = {p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w};
            #pragma unroll
            for (int a = 0; a < 8; ++a) {
                acc[r] += fminf(fmaxf(xv[a], imin), imax) * wf[a];
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        float t = acc[r];
        for (int off = 16; off > 0; off >>= 1) {
            t += __shfl_xor_sync(RC_FULL_MASK, t, off);
        }
        if (lane == 0) {
            y[(size_t)r * m + row] = fminf(fmaxf(t + (b ? b[row] : 0.f), omin), omax);
        }
    }
}
template __global__ void k_enc_gemv_f16<1>(const float *, const __half *, const float *, float *, int, int, float,
                                           float, float, float);
template __global__ void k_enc_gemv_f16<2>(const float *, const __half *, const float *, float *, int, int, float,
                                           float, float, float);
template __global__ void k_enc_gemv_f16<4>(const float *, const __half *, const float *, float *, int, int, float,
                                           float, float, float);
template __global__ void k_enc_gemv_f16<8>(const float *, const __half *, const float *, float *, int, int, float,
                                           float, float, float);

/* y block i (cols values) = x row idx[i], zeros for an index < 0: nb blocks,
 * one CUDA block each (the window partitions and shifts of Swin, patch merging,
 * the im2col of a convolution). */
__global__ void k_enc_gather(const float *x, const int *idx, float *y, int64_t nb, int cols)
{
    PDL_START();
    int64_t i = blockIdx.x;
    if (i >= nb) {
        return;
    }
    int src = idx[i];
    float *dst = y + i * cols;
    if (src < 0) {
        for (int c = threadIdx.x; c < cols; c += blockDim.x) {
            dst[c] = 0.0f;
        }
        return;
    }
    const float *p = x + (int64_t)src * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[c] = p[c];
    }
}

/* The 1D RoPE of EmbeddingGemma 2's text model, in place on x (n, heads,
 * hd): NEOX halves (value j turns with j + hd / 2) at the angles pos[row] *
 * inv[j], j < hd / 2. pos is (n) int32. */
__global__ void k_enc_rope1d(float *x, const int *pos, const float *inv, int n, int heads, int hd)
{
    PDL_START();
    int h2 = hd / 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n * heads * h2;
    if (i >= total) {
        return;
    }
    int j = (int)(i % h2);
    int64_t row = i / h2;                /* token * heads + head */
    float ang = (float)pos[row / heads] * inv[j];
    float c = cosf(ang), s = sinf(ang);
    float *p = x + row * hd;
    float a = p[j], b = p[j + h2];
    p[j] = a * c - b * s;
    p[j + h2] = b * c + a * s;
}

/* The attention of each query i over the keys [lo[i], hi[i]), scale 1, with
 * KV heads shared by heads / kv_heads query heads (query head h reads KV head
 * h / (heads / kv_heads)). q and o are (n, heads, hd), k and v (n, kv_heads,
 * hd); hd a multiple of 32 of at most 512. The ranges give a bidirectional
 * window, every key, or the inputs packed in one pass (each its own keys).
 * One warp for each (query, head); the EAR_W warps of a block take EAR_W
 * queries of one head over the union of their ranges, a tile of EAR_K keys
 * in shared memory (then its values in the same place). Lane j scores key j
 * of a tile; the softmax is online. */
#define EAR_W 8
#define EAR_K 32
/* NI = hd / 32, the values of a head in a lane: one kernel each (runner.cu), so
 * that the loops of the values are unrolled without the tests of d < hd (MuQ,
 * hd 64: 3 times as fast; the same products in the same order, the same bits) */
template <int NI>
__global__ void __launch_bounds__(32 * EAR_W) k_enc_attn_range_t(const float *q, const float *k, const float *v, float *o,
                                 const int *lo, const int *hi, int n, int heads, int kv_heads,
                                 int hd_)
{
    PDL_START();
    constexpr int hd = 32 * NI;     /* hd_ == hd: the loops of a head unrolled */
    extern __shared__ float esm[];
    __shared__ int blo, bhi;
    int ld = hd + 1;
    float *ts = esm;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float *qs = ts + EAR_K * ld + warp * hd;
    int h = blockIdx.y, kh = h / (heads / kv_heads);
    int q0 = blockIdx.x * EAR_W, qi = q0 + warp;
    bool live = qi < n;
    for (int d = lane; d < hd; d += 32) {
        qs[d] = live ? q[((size_t)qi * heads + h) * hd + d] : 0.f;
    }
    int mylo = live ? lo[qi] : 0, myhi = live ? hi[qi] : 0;
    if (threadIdx.x == 0) {
        int a = 0x7fffffff, b = 0;
        for (int w = 0; w < EAR_W && q0 + w < n; ++w) {
            a = min(a, lo[q0 + w]);
            b = max(b, hi[q0 + w]);
        }
        blo = a;
        bhi = b;
    }
    __syncthreads();
    float acc[NI];
    #pragma unroll
    for (int i = 0; i < NI; ++i) {
        acc[i] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int k0 = blo; k0 < bhi; k0 += EAR_K) {
        __syncthreads();
        for (int t = threadIdx.x; t < EAR_K * hd; t += blockDim.x) {
            int j = t / hd, d = t % hd, kj = k0 + j;
            ts[j * ld + d] = kj < bhi ? k[((size_t)kj * kv_heads + kh) * hd + d] : 0.f;
        }
        __syncthreads();
        int kj = k0 + lane;
        bool ok = kj >= mylo && kj < myhi;
        float sc = -INFINITY;
        if (ok) {
            sc = 0.f;
            for (int d = 0; d < hd; ++d) {
                sc += qs[d] * ts[lane * ld + d];
            }
        }
        float mt = sc;
        for (int off = 16; off > 0; off >>= 1) {
            mt = fmaxf(mt, __shfl_xor_sync(RC_FULL_MASK, mt, off));
        }
        float mn = fmaxf(m, mt);
        float p = ok ? expf(sc - mn) : 0.f;
        float scale = m == -INFINITY ? 0.f : expf(m - mn);
        float ps = p;
        for (int off = 16; off > 0; off >>= 1) {
            ps += __shfl_xor_sync(RC_FULL_MASK, ps, off);
        }
        l = l * scale + ps;
        #pragma unroll
        for (int i = 0; i < NI; ++i) {
            acc[i] *= scale;
        }
        __syncthreads();
        for (int t = threadIdx.x; t < EAR_K * hd; t += blockDim.x) {
            int j = t / hd, d = t % hd, vj = k0 + j;
            ts[j * ld + d] = vj < bhi ? v[((size_t)vj * kv_heads + kh) * hd + d] : 0.f;
        }
        __syncthreads();
        for (int j = 0; j < EAR_K; ++j) {
            float pj = __shfl_sync(RC_FULL_MASK, p, j);
            #pragma unroll
            for (int i = 0; i < NI; ++i) {
                acc[i] += pj * ts[j * ld + lane + 32 * i];
            }
        }
        m = mn;
    }
    if (live) {
        float inv_l = l > 0.f ? 1.f / l : 0.f;
        #pragma unroll
        for (int i = 0; i < NI; ++i) {
            o[((size_t)qi * heads + h) * hd + lane + 32 * i] = acc[i] * inv_l;
        }
    }
}

/* The attention of every query over every key, scale 1. q, k, v, and o are
 * (n, heads, hd), hd at most 128. One warp for each (query, head); the 4
 * warps of a block take 4 queries of one head and share the tiles of 32 keys
 * and values in shared memory. Lane j scores key j of a tile; the softmax is
 * online. */
__global__ void k_enc_attn(const float *q, const float *k, const float *v, float *o,
                           int n, int heads, int hd)
{
    PDL_START();
    extern __shared__ float esm[];
    int ld = hd + 1;
    float *ks = esm, *vs = ks + 32 * ld;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float *qs = vs + 32 * ld + warp * hd;
    int h = blockIdx.y, qi = blockIdx.x * 4 + warp;
    bool live = qi < n;
    for (int d = lane; d < hd; d += 32) {
        qs[d] = live ? q[((size_t)qi * heads + h) * hd + d] : 0.f;
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float m = -INFINITY, l = 0.f;
    for (int k0 = 0; k0 < n; k0 += 32) {
        __syncthreads();
        for (int t = threadIdx.x; t < 32 * hd; t += blockDim.x) {
            int j = t / hd, d = t % hd, kj = k0 + j;
            size_t src = ((size_t)kj * heads + h) * hd + d;
            ks[j * ld + d] = kj < n ? k[src] : 0.f;
            vs[j * ld + d] = kj < n ? v[src] : 0.f;
        }
        __syncthreads();
        bool kv = k0 + lane < n;
        float sc = -INFINITY;
        if (kv) {
            sc = 0.f;
            for (int d = 0; d < hd; ++d) {
                sc += qs[d] * ks[lane * ld + d];
            }
        }
        float mt = sc;
        for (int off = 16; off > 0; off >>= 1) {
            mt = fmaxf(mt, __shfl_xor_sync(RC_FULL_MASK, mt, off));
        }
        float mn = fmaxf(m, mt);
        float p = kv ? expf(sc - mn) : 0.f;
        float scale = m == -INFINITY ? 0.f : expf(m - mn);
        float ps = p;
        for (int off = 16; off > 0; off >>= 1) {
            ps += __shfl_xor_sync(RC_FULL_MASK, ps, off);
        }
        l = l * scale + ps;
        for (int i = 0; i < 4; ++i) {
            acc[i] *= scale;
        }
        for (int j = 0; j < 32; ++j) {
            float pj = __shfl_sync(RC_FULL_MASK, p, j);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = lane + 32 * i;
                if (d < hd) {
                    acc[i] += pj * vs[j * ld + d];
                }
            }
        }
        m = mn;
    }
    if (live) {
        for (int i = 0; i < 4; ++i) {
            int d = lane + 32 * i;
            if (d < hd) {
                o[((size_t)qi * heads + h) * hd + d] = acc[i] / l;
            }
        }
    }
}

/* k_enc_attn on the tensor cores (mma.sync m16n8k16): q, k, and v are
 * float16 (n, heads, hd), from k_enc_to_half; the scores, the softmax, and
 * the sums are float32, the weights p float16 (as k_flash_tc). hd is a
 * multiple of 8 of at most HP, HP a multiple of 16 (the values past hd are
 * zeros). A block of 4 warps takes EA_Q queries of one head; warp w takes the
 * 16 queries 16 w, with their fragments of q in registers. For each step of
 * EA_K keys: K and V to shared memory, S = Q K^T, the online softmax on the
 * fragments of S (the 4 lanes of a row share its maximum and its sum), then
 * O += P V (the fragments of S are those of P; V by ldmatrix.trans). */
#define EA_K 64

__device__ __forceinline__ void ldsm_x2_trans(uint32_t *d, const void *p)
{
#if RC_SM80
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(a));
#else
    __trap();
#endif
}

template <int HP>
__global__ void __launch_bounds__(128) k_enc_attn_tc(const __half *q, const __half *k,
                                                     const __half *v, float *o, int n, int heads,
                                                     int hd)
{
    PDL_START();
    constexpr int LD = HP + 8;                /* the row stride (halves): 16 bytes apart */
    __shared__ __align__(16) __half ks[EA_K * LD];
    __shared__ __align__(16) __half vs[EA_K * LD];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = lane / 4, c = lane % 4;
    int h = blockIdx.y;
    size_t rs = (size_t)heads * hd;           /* the row stride of q, k, v */
    int r0 = blockIdx.x * EA_Q + warp * 16 + g, r1 = r0 + 8;
    bool live0 = r0 < n, live1 = r1 < n;
    const __half *q0 = q + (size_t)(live0 ? r0 : 0) * rs + (size_t)h * hd;
    const __half *q1 = q + (size_t)(live1 ? r1 : 0) * rs + (size_t)h * hd;
    uint32_t qa[HP / 16][4];
    #pragma unroll
    for (int kk = 0; kk < HP / 16; ++kk) {
        int d0 = kk * 16 + 2 * c, d1 = d0 + 8;
        qa[kk][0] = d0 < hd ? *(const uint32_t *)&q0[d0] : 0u;
        qa[kk][1] = d0 < hd ? *(const uint32_t *)&q1[d0] : 0u;
        qa[kk][2] = d1 < hd ? *(const uint32_t *)&q0[d1] : 0u;
        qa[kk][3] = d1 < hd ? *(const uint32_t *)&q1[d1] : 0u;
    }
    float acc[HP / 8][4];
    #pragma unroll
    for (int i = 0; i < HP / 8; ++i) {
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    }
    const float L2E = 1.4426950408889634f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int k0 = 0; k0 < n; k0 += EA_K) {
        __syncthreads();
        /* 8 halves (16 bytes) at a time; zeros past n and past hd */
        for (int x = threadIdx.x; x < EA_K * (HP / 8); x += blockDim.x) {
            int j = x / (HP / 8), d = (x % (HP / 8)) * 8, kj = k0 + j;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = kv;
            if (kj < n && d < hd) {
                size_t src = (size_t)kj * rs + (size_t)h * hd + d;
                kv = *(const uint4 *)&k[src];
                vv = *(const uint4 *)&v[src];
            }
            *(uint4 *)&ks[j * LD + d] = kv;
            *(uint4 *)&vs[j * LD + d] = vv;
        }
        __syncthreads();
        float sc[EA_K / 8][4];
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
        }
        #pragma unroll
        for (int kk = 0; kk < HP / 16; ++kk) {
            #pragma unroll
            for (int nt = 0; nt < EA_K / 8; ++nt) {
                uint32_t b[2];
                b[0] = *(const uint32_t *)&ks[(nt * 8 + g) * LD + kk * 16 + 2 * c];
                b[1] = *(const uint32_t *)&ks[(nt * 8 + g) * LD + kk * 16 + 2 * c + 8];
                mma16816(sc[nt], qa[kk], b);
            }
        }
        /* sc[nt][0..1]: row g, keys nt * 8 + 2c and + 1; sc[nt][2..3]: row g + 8.
         * Each step has a key, so the maxima are finite. */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                bool ok = k0 + nt * 8 + 2 * c + u < n;
                sc[nt][u] = ok ? sc[nt][u] * L2E : -INFINITY;
                sc[nt][2 + u] = ok ? sc[nt][2 + u] * L2E : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][u]);
                mx1 = fmaxf(mx1, sc[nt][2 + u]);
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(RC_FULL_MASK, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(RC_FULL_MASK, mx1, off));
        }
        float f0 = exp2f(m0 - mx0), f1 = exp2f(m1 - mx1);   /* 0 at the first step */
        float ls0 = 0.f, ls1 = 0.f;
        uint32_t pa[EA_K / 8][2];
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            float e0 = exp2f(sc[nt][0] - mx0), e1 = exp2f(sc[nt][1] - mx0);
            float e2 = exp2f(sc[nt][2] - mx1), e3 = exp2f(sc[nt][3] - mx1);
            ls0 += e0 + e1;
            ls1 += e2 + e3;
            __half2 h01 = __floats2half2_rn(e0, e1), h23 = __floats2half2_rn(e2, e3);
            pa[nt][0] = *(uint32_t *)&h01;
            pa[nt][1] = *(uint32_t *)&h23;
        }
        for (int off = 1; off < 4; off <<= 1) {
            ls0 += __shfl_xor_sync(RC_FULL_MASK, ls0, off);
            ls1 += __shfl_xor_sync(RC_FULL_MASK, ls1, off);
        }
        l0 = l0 * f0 + ls0;
        l1 = l1 * f1 + ls1;
        m0 = mx0;
        m1 = mx1;
        #pragma unroll
        for (int nt = 0; nt < HP / 8; ++nt) {
            acc[nt][0] *= f0;
            acc[nt][1] *= f0;
            acc[nt][2] *= f1;
            acc[nt][3] *= f1;
        }
        #pragma unroll
        for (int kk = 0; kk < EA_K / 16; ++kk) {
            /* the A fragment of P: keys 16 kk + 2c (tile 2 kk), + 8 (tile 2 kk + 1) */
            uint32_t a[4] = {pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1]};
            const __half *vrow = &vs[(kk * 16 + (lane % 16)) * LD];
            #pragma unroll
            for (int nt = 0; nt < HP / 8; ++nt) {
                uint32_t b[2];
                ldsm_x2_trans(b, vrow + nt * 8);
                mma16816(acc[nt], a, b);
            }
        }
    }
    float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    float *o0 = o + (size_t)r0 * rs + (size_t)h * hd, *o1 = o + (size_t)r1 * rs + (size_t)h * hd;
    #pragma unroll
    for (int nt = 0; nt < HP / 8; ++nt) {
        int d = nt * 8 + 2 * c;
        if (d < hd) {
            if (live0) {
                *(float2 *)&o0[d] = make_float2(acc[nt][0] * inv0, acc[nt][1] * inv0);
            }
            if (live1) {
                *(float2 *)&o1[d] = make_float2(acc[nt][2] * inv1, acc[nt][3] * inv1);
            }
        }
    }
}

/* ENC_ATTN_X2: ENC_ATTN_RANGE on the tensor cores: q, k, v as two bfloat16
 * planes each (hi, then lo = x - hi: k_to_bf16 into xh, q's planes nE values
 * apart, k's and v's nEk: n heads hd each, or more keys than queries in
 * ENC_ATTN_X2C; the range of float32, about 16 bits), the scores q k^T of
 * three products a fragment (hi hi, hi lo, lo hi), the weights p as two
 * bfloat16 planes too, and p v of three products; the scores, the softmax
 * and the sums float32. Query
 * i reads the keys [lo[i], hi[i]) (lo and hi do not decrease from a query to
 * the next); a block of 4 warps takes EA_Q queries of one head and steps of
 * EA_K keys over the union of their ranges, each row masked to its own. A row
 * with no key gives zeros. hd a multiple of 8 of at most HP (64 or 128; the
 * four planes of a step in dynamic shared memory, EAX_SMEM: 36 KB, 68 KB). */
template <int HP>
__global__ void __launch_bounds__(128) k_enc_attn_x2(const uint16_t *qp, const uint16_t *kp, const uint16_t *vp,
                                                     size_t nE, size_t nEk, float *o, const int *lo, const int *hi,
                                                     int n, int heads, int hd)
{
    PDL_START();
    constexpr int LD = HP + 8;
    extern __shared__ __align__(16) uint16_t eax_sm[];      /* EAX_SMEM(HP): the four planes of a step */
    uint16_t *ksh = eax_sm, *ksl = ksh + EA_K * LD, *vsh = ksl + EA_K * LD, *vsl = vsh + EA_K * LD;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = lane / 4, c = lane % 4;
    int h = blockIdx.y;
    size_t rs = (size_t)heads * hd;
    int b0 = blockIdx.x * EA_Q, b1 = min(b0 + EA_Q, n) - 1;
    int klo = lo[b0], khi = hi[b1];                    /* the union of the block's ranges */
    int r0 = b0 + warp * 16 + g, r1 = r0 + 8;
    bool live0 = r0 < n, live1 = r1 < n;
    int lo0 = live0 ? lo[r0] : 0, hi0 = live0 ? hi[r0] : 0;
    int lo1 = live1 ? lo[r1] : 0, hi1 = live1 ? hi[r1] : 0;
    uint32_t qh[HP / 16][4], ql[HP / 16][4];
    #pragma unroll
    for (int pl = 0; pl < 2; ++pl) {
        const uint16_t *q0 = qp + pl * nE + (size_t)(live0 ? r0 : 0) * rs + (size_t)h * hd;
        const uint16_t *q1 = qp + pl * nE + (size_t)(live1 ? r1 : 0) * rs + (size_t)h * hd;
        #pragma unroll
        for (int kk = 0; kk < HP / 16; ++kk) {
            int d0 = kk * 16 + 2 * c, d1 = d0 + 8;
            uint32_t *qa = pl ? ql[kk] : qh[kk];
            qa[0] = d0 < hd ? *(const uint32_t *)&q0[d0] : 0u;
            qa[1] = d0 < hd ? *(const uint32_t *)&q1[d0] : 0u;
            qa[2] = d1 < hd ? *(const uint32_t *)&q0[d1] : 0u;
            qa[3] = d1 < hd ? *(const uint32_t *)&q1[d1] : 0u;
        }
    }
    float acc[HP / 8][4];
    #pragma unroll
    for (int i = 0; i < HP / 8; ++i) {
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    }
    const float L2E = 1.4426950408889634f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int k0 = klo; k0 < khi; k0 += EA_K) {
        __syncthreads();
        for (int x = threadIdx.x; x < EA_K * (HP / 8); x += blockDim.x) {
            int j = x / (HP / 8), d = (x % (HP / 8)) * 8, kj = k0 + j;
            uint4 z = make_uint4(0u, 0u, 0u, 0u), a = z, b = z, e = z, f = z;
            if (kj < khi && d < hd) {
                size_t src = (size_t)kj * rs + (size_t)h * hd + d;
                a = *(const uint4 *)&kp[src];
                b = *(const uint4 *)&kp[nEk + src];
                e = *(const uint4 *)&vp[src];
                f = *(const uint4 *)&vp[nEk + src];
            }
            *(uint4 *)&ksh[j * LD + d] = a;
            *(uint4 *)&ksl[j * LD + d] = b;
            *(uint4 *)&vsh[j * LD + d] = e;
            *(uint4 *)&vsl[j * LD + d] = f;
        }
        __syncthreads();
        float sc[EA_K / 8][4];
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
        }
        #pragma unroll
        for (int kk = 0; kk < HP / 16; ++kk) {
            #pragma unroll
            for (int nt = 0; nt < EA_K / 8; ++nt) {
                int at = (nt * 8 + g) * LD + kk * 16 + 2 * c;
                uint32_t bh[2] = {*(const uint32_t *)&ksh[at], *(const uint32_t *)&ksh[at + 8]};
                uint32_t bl[2] = {*(const uint32_t *)&ksl[at], *(const uint32_t *)&ksl[at + 8]};
                mma16816_bf16(sc[nt], qh[kk], bh);
                mma16816_bf16(sc[nt], qh[kk], bl);
                mma16816_bf16(sc[nt], ql[kk], bh);
            }
        }
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int kj = k0 + nt * 8 + 2 * c + u;
                sc[nt][u] = kj >= lo0 && kj < hi0 ? sc[nt][u] * L2E : -INFINITY;
                sc[nt][2 + u] = kj >= lo1 && kj < hi1 ? sc[nt][2 + u] * L2E : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][u]);
                mx1 = fmaxf(mx1, sc[nt][2 + u]);
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(RC_FULL_MASK, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(RC_FULL_MASK, mx1, off));
        }
        /* a row with no key yet keeps max -inf: its weights and its factor are 0 and 1 */
        float s0 = mx0 == -INFINITY ? 0.f : mx0, s1 = mx1 == -INFINITY ? 0.f : mx1;
        float f0 = m0 == -INFINITY ? 1.f : exp2f(m0 - s0), f1 = m1 == -INFINITY ? 1.f : exp2f(m1 - s1);
        float ls0 = 0.f, ls1 = 0.f;
        uint32_t ph[EA_K / 8][2], pl[EA_K / 8][2];
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            float e0 = exp2f(sc[nt][0] - s0), e1 = exp2f(sc[nt][1] - s0);
            float e2 = exp2f(sc[nt][2] - s1), e3 = exp2f(sc[nt][3] - s1);
            ls0 += e0 + e1;
            ls1 += e2 + e3;
            uint16_t h0 = f2bf_rn(e0), h1 = f2bf_rn(e1), h2 = f2bf_rn(e2), h3 = f2bf_rn(e3);
            uint16_t q0 = f2bf_rn(e0 - __uint_as_float((uint32_t)h0 << 16));
            uint16_t q1 = f2bf_rn(e1 - __uint_as_float((uint32_t)h1 << 16));
            uint16_t q2 = f2bf_rn(e2 - __uint_as_float((uint32_t)h2 << 16));
            uint16_t q3 = f2bf_rn(e3 - __uint_as_float((uint32_t)h3 << 16));
            ph[nt][0] = (uint32_t)h0 | ((uint32_t)h1 << 16);
            ph[nt][1] = (uint32_t)h2 | ((uint32_t)h3 << 16);
            pl[nt][0] = (uint32_t)q0 | ((uint32_t)q1 << 16);
            pl[nt][1] = (uint32_t)q2 | ((uint32_t)q3 << 16);
        }
        for (int off = 1; off < 4; off <<= 1) {
            ls0 += __shfl_xor_sync(RC_FULL_MASK, ls0, off);
            ls1 += __shfl_xor_sync(RC_FULL_MASK, ls1, off);
        }
        l0 = l0 * f0 + ls0;
        l1 = l1 * f1 + ls1;
        m0 = mx0;
        m1 = mx1;
        #pragma unroll
        for (int nt = 0; nt < HP / 8; ++nt) {
            acc[nt][0] *= f0;
            acc[nt][1] *= f0;
            acc[nt][2] *= f1;
            acc[nt][3] *= f1;
        }
        #pragma unroll
        for (int kk = 0; kk < EA_K / 16; ++kk) {
            uint32_t ah[4] = {ph[2 * kk][0], ph[2 * kk][1], ph[2 * kk + 1][0], ph[2 * kk + 1][1]};
            uint32_t al[4] = {pl[2 * kk][0], pl[2 * kk][1], pl[2 * kk + 1][0], pl[2 * kk + 1][1]};
            int row = (kk * 16 + (lane % 16)) * LD;
            #pragma unroll
            for (int nt = 0; nt < HP / 8; ++nt) {
                uint32_t bh[2], bl[2];
                ldsm_x2_trans(bh, &vsh[row + nt * 8]);
                ldsm_x2_trans(bl, &vsl[row + nt * 8]);
                mma16816_bf16(acc[nt], ah, bh);
                mma16816_bf16(acc[nt], ah, bl);
                mma16816_bf16(acc[nt], al, bh);
            }
        }
    }
    float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    float *o0 = o + (size_t)r0 * rs + (size_t)h * hd, *o1 = o + (size_t)r1 * rs + (size_t)h * hd;
    #pragma unroll
    for (int nt = 0; nt < HP / 8; ++nt) {
        int d = nt * 8 + 2 * c;
        if (d < hd) {
            if (live0) {
                o0[d] = acc[nt][0] * inv0;
                o0[d + 1] = acc[nt][1] * inv0;
            }
            if (live1) {
                o1[d] = acc[nt][2] * inv1;
                o1[d + 1] = acc[nt][3] * inv1;
            }
        }
    }
}

/* ENC_ATTN_TILE: ENC_ATTN_RANGE (no shared KV heads) in float32 on the CUDA
 * cores in register-blocked tiles, for heads of at most 128 (ENC_ATTN_X2's
 * float32 form). A block of NT threads takes BQ queries of one head and steps
 * of EAT_K keys over the union of their ranges (lo and hi do not decrease
 * from a query to the next), each row masked to its own; the steps start at
 * the first key of the query's group of EAT_Q (the same steps for every BQ: a
 * step of no key of a row changes nothing of it). Q stays in shared memory
 * (its float4s of odd rows swapped in pairs: the two rows a warp reads in
 * different banks); a step's K, then its V, in one buffer, the next one
 * loaded into registers while this one's products run; the weights P in a
 * third. HALF (heads of 128 in 48 KB): K and V in two halves of 64 columns
 * each (K's float4s of key j at d4 ^ (j % 8), P's swapped as Q's), the scores
 * summed over the first half, then the second. Thread (ty, tx) of NT / 16 x
 * 16 holds rows ty + NT / 16 i (i < BQ 16 / NT): the scores of keys tx and tx
 * + 16 of a step, and the values of columns 64 g + 4 tx .. + 3 (g < HP / 64).
 * A score is fmaf over d in order from 0, the softmax online with expf, a
 * step's sum of weights the tree of k_enc_attn_range_t's lanes, p v in order
 * of the keys: every form the same bits, and those of k_enc_attn_range_t
 * where the queries of its blocks and of these start at the same key (every
 * query over every key). A row with no key gives zeros. hd a multiple of 4 of
 * at most HP (64 or 128; zeros past hd); EAT_SMEM(HP, BQ, HALF). */
#define EAT_Q 128
#define EAT_K 32
template <int HP, int BQ, int NT, int MINB, bool HALF>
__global__ void __launch_bounds__(NT, MINB) k_enc_attn_tile(const float *q, const float *k, const float *v,
                                                            float *o, const int *lo, const int *hi, int n,
                                                            int heads, int hd)
{
    PDL_START();
    constexpr int TY = NT / 16, RW = BQ / TY, NG = HP / 64;
    constexpr int LK = HALF ? 64 : HP + 4, LP = HALF ? EAT_K : EAT_K + 4;
    constexpr int KW = HALF ? 64 : HP;                  /* the columns of a load */
    constexpr int NL = EAT_K * KW / 4 / NT;
    static_assert(!HALF || HP == 128, "HALF: heads of 128");
    extern __shared__ __align__(16) float eat_sm[];
    float *qs = eat_sm, *kvs = qs + BQ * HP, *ps = kvs + EAT_K * LK;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16, par = ty & 1;
    const int h = blockIdx.y;
    const size_t rs = (size_t)heads * hd;
    const int b0 = blockIdx.x * BQ, b1 = min(b0 + BQ, n) - 1;
    /* the steps from the first key of the group of EAT_Q; the keys of every row: [lo[b1], hi[b0]) */
    const int klo = lo[b0 / EAT_Q * EAT_Q], khi = hi[b1], alo = lo[b1], ahi = hi[b0];
    for (int f = tid; f < BQ * HP / 4; f += NT) {
        int r = f / (HP / 4), d4 = f % (HP / 4);
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        if (b0 + r < n && 4 * d4 < hd) {
            a = *(const float4 *)&q[(size_t)(b0 + r) * rs + (size_t)h * hd + 4 * d4];
        }
        *(float4 *)&qs[r * HP + 4 * (d4 ^ (r & 1))] = a;
    }
    /* float4 f = tid + NT u of a step's K or V (columns c0 ..): key f / (KW / 4), columns 4 (f % (KW / 4)) */
    float4 st[NL];
    auto load = [&](const float *src, int k0, int c0) {
        #pragma unroll
        for (int u = 0; u < NL; ++u) {
            int f = tid + NT * u, j = f / (KW / 4), d = c0 + 4 * (f % (KW / 4));
            st[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k0 + j < khi && d < hd) {
                st[u] = *(const float4 *)&src[(size_t)(k0 + j) * rs + (size_t)h * hd + d];
            }
        }
    };
    auto store = [&](bool kx) {
        #pragma unroll
        for (int u = 0; u < NL; ++u) {
            int f = tid + NT * u, j = f / (KW / 4), d4 = f % (KW / 4);
            *(float4 *)&kvs[j * LK + 4 * (HALF && kx ? d4 ^ (j & 7) : d4)] = st[u];
        }
    };
    float acc[RW][4 * NG];
    float m[RW], l[RW];
    #pragma unroll
    for (int i = 0; i < RW; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
        #pragma unroll
        for (int c = 0; c < 4 * NG; ++c) {
            acc[i][c] = 0.f;
        }
    }
    /* the rows' float4s of Q (and of P in HALF): even d4 at qe + 4 d4, odd at qo + 4 d4 */
    const float *qe = qs + ty * HP + 4 * par, *qo = qs + ty * HP - 4 * par;
    const float *pe = ps + ty * LP + (HALF ? 4 * par : 0), *po = ps + ty * LP - (HALF ? 4 * par : 0);
    /* K's float4 d4 of key tx (and tx + 16): kvs[ko[d4 % 8] + 32 (d4 / 8)] in HALF, else kvs[ko[0] + 4 d4] */
    int ko[HALF ? 8 : 1];
    #pragma unroll
    for (int b = 0; b < (HALF ? 8 : 1); ++b) {
        ko[b] = tx * LK + (HALF ? 4 * (b ^ (tx & 7)) : 0);
    }
    float sc[RW][2];
    /* the scores' sums over the float4s [d0, d0 + nd) of the buffer (columns 4 dq0 .. of the head) */
    auto scores = [&](int dq0, auto nd_c) {
        constexpr int ND = decltype(nd_c)::value;
        #pragma unroll
        for (int d4 = 0; d4 < ND; ++d4) {
            const float *ka_p = kvs + (HALF ? ko[d4 % 8] + 32 * (d4 / 8) : ko[0] + 4 * d4);
            float4 ka = *(const float4 *)ka_p, kb = *(const float4 *)(ka_p + 16 * LK);
            const int dg = dq0 + d4;
            #pragma unroll
            for (int i = 0; i < RW; ++i) {
                float4 a = *(const float4 *)&(dg & 1 ? qo : qe)[i * TY * HP + 4 * dg];
                sc[i][0] = fmaf(a.x, ka.x, sc[i][0]);
                sc[i][0] = fmaf(a.y, ka.y, sc[i][0]);
                sc[i][0] = fmaf(a.z, ka.z, sc[i][0]);
                sc[i][0] = fmaf(a.w, ka.w, sc[i][0]);
                sc[i][1] = fmaf(a.x, kb.x, sc[i][1]);
                sc[i][1] = fmaf(a.y, kb.y, sc[i][1]);
                sc[i][1] = fmaf(a.z, kb.z, sc[i][1]);
                sc[i][1] = fmaf(a.w, kb.w, sc[i][1]);
            }
        }
    };
    /* p v of the step over the columns of group g (the buffer holds V's columns from 64 g, or all) */
    auto values = [&](int g0, auto ng_c) {
        constexpr int VG = decltype(ng_c)::value;
        #pragma unroll
        for (int j4 = 0; j4 < EAT_K / 4; ++j4) {
            float pr[RW][4];
            #pragma unroll
            for (int i = 0; i < RW; ++i) {
                float4 a = *(const float4 *)&(j4 & 1 ? po : pe)[i * TY * LP + 4 * j4];
                pr[i][0] = a.x; pr[i][1] = a.y; pr[i][2] = a.z; pr[i][3] = a.w;
            }
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                float vv[4 * VG];
                #pragma unroll
                for (int g = 0; g < VG; ++g) {
                    float4 b = *(const float4 *)&kvs[(4 * j4 + jj) * LK + (HALF ? 0 : 64 * g) + 4 * tx];
                    vv[4 * g] = b.x; vv[4 * g + 1] = b.y; vv[4 * g + 2] = b.z; vv[4 * g + 3] = b.w;
                }
                #pragma unroll
                for (int i = 0; i < RW; ++i) {
                    #pragma unroll
                    for (int c = 0; c < 4 * VG; ++c) {
                        acc[i][4 * g0 + c] += pr[i][jj] * vv[c];
                    }
                }
            }
        }
    };
    if (klo < khi) {
        load(k, klo, 0);
        store(true);
    }
    __syncthreads();
    for (int k0 = klo; k0 < khi; k0 += EAT_K) {
        const bool more = k0 + EAT_K < khi;
        #pragma unroll
        for (int i = 0; i < RW; ++i) {
            sc[i][0] = sc[i][1] = 0.f;
        }
        if constexpr (HALF) {
            load(k, k0, 64);
            scores(0, std::integral_constant<int, 16>());
            __syncthreads();
            store(true);
            __syncthreads();
            load(v, k0, 0);
            scores(16, std::integral_constant<int, 16>());
        } else {
            load(v, k0, 0);
            scores(0, std::integral_constant<int, HP / 4>());
        }
        /* the rows' own ranges where the step is not inside every row's */
        if (k0 < alo || k0 + EAT_K > ahi) {
            #pragma unroll
            for (int i = 0; i < RW; ++i) {
                int r = b0 + ty + TY * i, rl = r < n ? lo[r] : 0, rh = r < n ? hi[r] : 0;
                if (k0 + tx < rl || k0 + tx >= rh) {
                    sc[i][0] = -INFINITY;
                }
                if (k0 + tx + 16 < rl || k0 + tx + 16 >= rh) {
                    sc[i][1] = -INFINITY;
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < RW; ++i) {
            float mt = fmaxf(sc[i][0], sc[i][1]);
            for (int off = 8; off > 0; off >>= 1) {
                mt = fmaxf(mt, __shfl_xor_sync(RC_FULL_MASK, mt, off));
            }
            float mn = fmaxf(m[i], mt);
            float p0 = sc[i][0] == -INFINITY ? 0.f : expf(sc[i][0] - mn);
            float p1 = sc[i][1] == -INFINITY ? 0.f : expf(sc[i][1] - mn);
            float scale = m[i] == -INFINITY ? 0.f : expf(m[i] - mn);
            /* the lanes' tree of k_enc_attn_range_t: key j with j + 16, then the halves' xor 8, 4, 2, 1 */
            float s = p0 + p1;
            for (int off = 8; off > 0; off >>= 1) {
                s += __shfl_xor_sync(RC_FULL_MASK, s, off);
            }
            l[i] = l[i] * scale + s;
            #pragma unroll
            for (int c = 0; c < 4 * NG; ++c) {
                acc[i][c] *= scale;
            }
            m[i] = mn;
            int r = ty + TY * i;
            if constexpr (HALF) {
                ps[r * LP + 4 * ((tx / 4) ^ (r & 1)) + tx % 4] = p0;
                ps[r * LP + 4 * ((tx / 4 + 4) ^ (r & 1)) + tx % 4] = p1;
            } else {
                ps[r * LP + tx] = p0;
                ps[r * LP + tx + 16] = p1;
            }
        }
        __syncthreads();
        store(false);
        __syncthreads();
        if constexpr (HALF) {
            load(v, k0, 64);
            values(0, std::integral_constant<int, 1>());
            __syncthreads();
            store(false);
            __syncthreads();
            if (more) {
                load(k, k0 + EAT_K, 0);
            }
            values(1, std::integral_constant<int, 1>());
        } else {
            if (more) {
                load(k, k0 + EAT_K, 0);
            }
            values(0, std::integral_constant<int, NG>());
        }
        __syncthreads();
        if (more) {
            store(true);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < RW; ++i) {
        int r = b0 + ty + TY * i;
        if (r >= n) {
            continue;
        }
        float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
        #pragma unroll
        for (int g = 0; g < NG; ++g) {
            int d = 64 * g + 4 * tx;
            if (d < hd) {
                *(float4 *)&o[(size_t)r * rs + (size_t)h * hd + d] =
                    make_float4(acc[i][4 * g] * inv, acc[i][4 * g + 1] * inv, acc[i][4 * g + 2] * inv,
                                acc[i][4 * g + 3] * inv);
            }
        }
    }
}

/* ENC_ATTN_TILE (and ENC_ATTN_X2 without tensor cores) in the form of the device (the same
 * bits): 128 queries a block where a block takes its shared memory (98.5 KB for heads of 128,
 * 58.5 KB for 64), else 64 in at most 48 KB (heads of 128: K and V in halves; Pascal, Maxwell).
 * RECLANK_GPU_ENC_TILE_FORM=1, 2: one of them; 3: 64 queries of 256 threads. */
static void enc_attn_tile(const float *q, const float *k, const float *v, float *o, const int *lo, const int *hi,
                          int n, int heads, int hd, cudaStream_t s)
{
    static int form = -1;
    if (form < 0) {
        const char *e = getenv("RECLANK_GPU_ENC_TILE_FORM");
        form = e ? atoi(e) : 0;
    }
    int f = form;
    if (f < 1 || f > 3) {
        f = (size_t)gg_caps.smem_block_max >= (hd <= 64 ? EAT_SMEM(64, 128, false) : EAT_SMEM(128, 128, false))
                ? 1 : 2;
    }
#define EAT_LAUNCH(HP, BQ, NT, MINB, HALF)                                                                   \
    {                                                                                                        \
        static int set = 0;                                                                                  \
        size_t sm = EAT_SMEM(HP, BQ, HALF);                                                                  \
        if (sm > 48 * 1024 && !set) {                                                                        \
            cudaFuncSetAttribute(k_enc_attn_tile<HP, BQ, NT, MINB, HALF>,                                    \
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);                      \
            set = 1;                                                                                         \
        }                                                                                                    \
        k_enc_attn_tile<HP, BQ, NT, MINB, HALF><<<dim3((unsigned)((n + BQ - 1) / BQ), (unsigned)heads), NT,  \
                                                  sm, s>>>(q, k, v, o, lo, hi, n, heads, hd);                \
    }
    if (hd <= 64 && f == 1) {
        EAT_LAUNCH(64, 128, 256, 1, false)
    } else if (hd <= 64 && f == 2) {
        EAT_LAUNCH(64, 64, 128, 2, false)
    } else if (hd <= 64) {
        EAT_LAUNCH(64, 64, 256, 2, false)
    } else if (f == 1) {
        EAT_LAUNCH(128, 128, 256, 1, false)
    } else if (f == 2) {
        EAT_LAUNCH(128, 64, 128, 2, true)
    } else {
        EAT_LAUNCH(128, 64, 256, 2, true)
    }
#undef EAT_LAUNCH
}

/* ENC_ATTN_BLOCKS: block-sparse attention (FlashVSR's: Block-Sparse-Attention's blocks of 128 queries and 128
 * keys), k_enc_attn_tile's float32 form (HP 128, BQ 128, NT 256: its order of the sums, its online softmax) over
 * the key blocks of a list instead of a range: a block of threads takes query block blockIdx.x of head
 * blockIdx.y and steps of EAT_K keys through the blocks bl[(h nqb + qb) nsel ..] (ascending, -1 ends them), the
 * same keys for its 128 rows (no row's own mask). The lists of the call's blocks: those of query blocks qb0 ..
 * of a list of nqb_all (a tile of the rows). K and V float32, or float16 (F16: uint16 rows, each value widened as
 * it is loaded). A query block with no key block gives zeros. hd a multiple of 4 of at most 128. */
template <bool F16>
__global__ void __launch_bounds__(256, 1) k_enc_attn_blocks(const float *q, const void *kp, const void *vp,
                                                             float *o, const int *bl, int nqb_all, int qb0,
                                                             int heads, int hd, int nsel)
{
    PDL_START();
    constexpr int HP = 128, BQ = 128, NT = 256;
    constexpr int TY = NT / 16, RW = BQ / TY, NG = HP / 64;
    constexpr int LK = HP + 4, LP = EAT_K + 4, KW = HP;
    constexpr int NL = EAT_K * KW / 4 / NT;
    extern __shared__ __align__(16) float eab_sm[];
    float *qs = eab_sm, *kvs = qs + BQ * HP, *ps = kvs + EAT_K * LK;
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16, par = ty & 1;
    const int h = blockIdx.y, qb = blockIdx.x;
    const size_t rs = (size_t)heads * hd;
    const int b0 = qb * BQ;
    const int *L = bl + ((size_t)h * nqb_all + qb0 + qb) * nsel;
    int nb = 0;
    while (nb < nsel && L[nb] >= 0) {
        ++nb;
    }
    const int nst = 4 * nb;                 /* steps of EAT_K: 4 a key block */
    for (int f = tid; f < BQ * HP / 4; f += NT) {
        int r = f / (HP / 4), d4 = f % (HP / 4);
        float4 a = make_float4(0.f, 0.f, 0.f, 0.f);
        if (4 * d4 < hd) {
            a = *(const float4 *)&q[(size_t)(b0 + r) * rs + (size_t)h * hd + 4 * d4];
        }
        *(float4 *)&qs[r * HP + 4 * (d4 ^ (r & 1))] = a;
    }
    float4 st[NL];
    auto kstep = [&](int i) { return L[i >> 2] * 128 + EAT_K * (i & 3); };
    auto load = [&](const void *src, int k0) {
        #pragma unroll
        for (int u = 0; u < NL; ++u) {
            int f = tid + NT * u, j = f / (KW / 4), d = 4 * (f % (KW / 4));
            st[u] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (d < hd) {
                size_t at = (size_t)(k0 + j) * rs + (size_t)h * hd + d;
                if constexpr (F16) {
                    const __half2 *p2 = (const __half2 *)((const __half *)src + at);
                    float2 a = __half22float2(p2[0]), b = __half22float2(p2[1]);
                    st[u] = make_float4(a.x, a.y, b.x, b.y);
                } else {
                    st[u] = *(const float4 *)((const float *)src + at);
                }
            }
        }
    };
    auto store = [&]() {
        #pragma unroll
        for (int u = 0; u < NL; ++u) {
            int f = tid + NT * u, j = f / (KW / 4), d4 = f % (KW / 4);
            *(float4 *)&kvs[j * LK + 4 * d4] = st[u];
        }
    };
    float acc[RW][4 * NG];
    float m[RW], l[RW];
    #pragma unroll
    for (int i = 0; i < RW; ++i) {
        m[i] = -INFINITY;
        l[i] = 0.f;
        #pragma unroll
        for (int c = 0; c < 4 * NG; ++c) {
            acc[i][c] = 0.f;
        }
    }
    const float *qe = qs + ty * HP + 4 * par, *qo = qs + ty * HP - 4 * par;
    const int ko = tx * LK;
    float sc[RW][2];
    if (nst > 0) {
        load(kp, kstep(0));
        store();
    }
    __syncthreads();
    for (int i0 = 0; i0 < nst; ++i0) {
        const int k0 = kstep(i0);
        const bool more = i0 + 1 < nst;
        #pragma unroll
        for (int i = 0; i < RW; ++i) {
            sc[i][0] = sc[i][1] = 0.f;
        }
        load(vp, k0);
        #pragma unroll
        for (int d4 = 0; d4 < HP / 4; ++d4) {
            const float *ka_p = kvs + ko + 4 * d4;
            float4 ka = *(const float4 *)ka_p, kb = *(const float4 *)(ka_p + 16 * LK);
            #pragma unroll
            for (int i = 0; i < RW; ++i) {
                float4 a = *(const float4 *)&(d4 & 1 ? qo : qe)[i * TY * HP + 4 * d4];
                sc[i][0] = fmaf(a.x, ka.x, sc[i][0]);
                sc[i][0] = fmaf(a.y, ka.y, sc[i][0]);
                sc[i][0] = fmaf(a.z, ka.z, sc[i][0]);
                sc[i][0] = fmaf(a.w, ka.w, sc[i][0]);
                sc[i][1] = fmaf(a.x, kb.x, sc[i][1]);
                sc[i][1] = fmaf(a.y, kb.y, sc[i][1]);
                sc[i][1] = fmaf(a.z, kb.z, sc[i][1]);
                sc[i][1] = fmaf(a.w, kb.w, sc[i][1]);
            }
        }
        #pragma unroll
        for (int i = 0; i < RW; ++i) {
            float mt = fmaxf(sc[i][0], sc[i][1]);
            for (int off = 8; off > 0; off >>= 1) {
                mt = fmaxf(mt, __shfl_xor_sync(RC_FULL_MASK, mt, off));
            }
            float mn = fmaxf(m[i], mt);
            float p0 = expf(sc[i][0] - mn), p1 = expf(sc[i][1] - mn);
            float scale = m[i] == -INFINITY ? 0.f : expf(m[i] - mn);
            float sum = p0 + p1;
            for (int off = 8; off > 0; off >>= 1) {
                sum += __shfl_xor_sync(RC_FULL_MASK, sum, off);
            }
            l[i] = l[i] * scale + sum;
            #pragma unroll
            for (int c = 0; c < 4 * NG; ++c) {
                acc[i][c] *= scale;
            }
            m[i] = mn;
            int r = ty + TY * i;
            ps[r * LP + tx] = p0;
            ps[r * LP + tx + 16] = p1;
        }
        __syncthreads();
        store();
        __syncthreads();
        if (more) {
            load(kp, kstep(i0 + 1));
        }
        #pragma unroll
        for (int j4 = 0; j4 < EAT_K / 4; ++j4) {
            float pr[RW][4];
            #pragma unroll
            for (int i = 0; i < RW; ++i) {
                float4 a = *(const float4 *)&ps[(ty + TY * i) * LP + 4 * j4];
                pr[i][0] = a.x; pr[i][1] = a.y; pr[i][2] = a.z; pr[i][3] = a.w;
            }
            #pragma unroll
            for (int jj = 0; jj < 4; ++jj) {
                float vv[4 * NG];
                #pragma unroll
                for (int g = 0; g < NG; ++g) {
                    float4 b = *(const float4 *)&kvs[(4 * j4 + jj) * LK + 64 * g + 4 * tx];
                    vv[4 * g] = b.x; vv[4 * g + 1] = b.y; vv[4 * g + 2] = b.z; vv[4 * g + 3] = b.w;
                }
                #pragma unroll
                for (int i = 0; i < RW; ++i) {
                    #pragma unroll
                    for (int c = 0; c < 4 * NG; ++c) {
                        acc[i][c] += pr[i][jj] * vv[c];
                    }
                }
            }
        }
        __syncthreads();
        if (more) {
            store();
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i = 0; i < RW; ++i) {
        int r = b0 + ty + TY * i;
        float inv = l[i] > 0.f ? 1.f / l[i] : 0.f;
        #pragma unroll
        for (int g = 0; g < NG; ++g) {
            int d = 64 * g + 4 * tx;
            if (d < hd) {
                *(float4 *)&o[(size_t)r * rs + (size_t)h * hd + d] =
                    make_float4(acc[i][4 * g] * inv, acc[i][4 * g + 1] * inv, acc[i][4 * g + 2] * inv,
                                acc[i][4 * g + 3] * inv);
            }
        }
    }
}

/* ENC_BLOCK_MASK: FlashVSR's generate_draft_block_mask on the device, then the lists of ENC_ATTN_BLOCKS. qm (nq, heads
 * hd): the query blocks' mean rows; km: the key blocks' mean rows, block j at row kr[j] (nk of them); local (nb x nb
 * uint8): the spatial window of a block of a slice (block b of a slice: b % nb). p (heads, nq, nk) the scores
 * (fmaf over d in order), -inf outside the window, softmax over the keys (expf); each head's and query slice's
 * (nb rows) threshold the (k + 1)-th largest of its nb nk values (k = min(nb nk - 1, topk): a radix select of
 * their bits, non-negative floats); bl[h][q0 + q][..] the ids[j] of the keys above it, in order, -1 after (nsel a
 * row). */
__global__ void k_bmask_scores(const float *qm, const float *km, const int *kr, const uint8_t *local, float *p, int nq,
                               int nk, int nb, int heads, int hd)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (int64_t)heads * nq * nk) {
        return;
    }
    int j = (int)(t % nk), q = (int)((t / nk) % nq), h = (int)(t / ((int64_t)nk * nq));
    float s = -INFINITY;
    if (local[(size_t)(q % nb) * nb + j % nb]) {
        const float *a = qm + (size_t)q * heads * hd + (size_t)h * hd;
        const float *b = km + (size_t)kr[j] * heads * hd + (size_t)h * hd;
        s = 0.f;
        for (int d = 0; d < hd; ++d) {
            s = fmaf(a[d], b[d], s);
        }
    }
    p[t] = s;
}

/* a block a row of p: its softmax in place */
__global__ void __launch_bounds__(256) k_bmask_softmax(float *p, int nk)
{
    PDL_START();
    __shared__ float red[32];
    float *r = p + (size_t)blockIdx.x * nk;
    const int tid = threadIdx.x, lane = tid % 32, wid = tid / 32;
    float m = -INFINITY;
    for (int j = tid; j < nk; j += 256) {
        m = fmaxf(m, r[j]);
    }
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, off));
    }
    if (lane == 0) {
        red[wid] = m;
    }
    __syncthreads();
    m = red[0];
    for (int w = 1; w < 8; ++w) {
        m = fmaxf(m, red[w]);
    }
    __syncthreads();
    float sum = 0.f;
    for (int j = tid; j < nk; j += 256) {
        float e = r[j] == -INFINITY ? 0.f : expf(r[j] - m);
        r[j] = e;
        sum += e;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, off);
    }
    if (lane == 0) {
        red[wid] = sum;
    }
    __syncthreads();
    sum = 0.f;
    for (int w = 0; w < 8; ++w) {
        sum += red[w];
    }
    float inv = 1.f / sum;
    for (int j = tid; j < nk; j += 256) {
        r[j] = r[j] * inv;
    }
}

/* a block a head's query slice (n = nb nk values of p from base): thr[blockIdx.x] the (k + 1)-th largest, by 4
 * passes of 8 bits over the values' bits (non-negative: their order) */
__global__ void __launch_bounds__(1024) k_bmask_select(const float *p, float *thr, int64_t n, int64_t k)
{
    PDL_START();
    __shared__ unsigned hist[256];
    __shared__ unsigned prefix_s, want_s;
    const unsigned *v = (const unsigned *)(p + (size_t)blockIdx.x * n);
    const int tid = threadIdx.x;
    unsigned prefix = 0, mask = 0;
    unsigned long long want = (unsigned long long)k + 1;       /* the rank from the top */
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int b = tid; b < 256; b += 1024) {
            hist[b] = 0;
        }
        __syncthreads();
        for (int64_t i = tid; i < n; i += 1024) {
            unsigned x = v[i];
            if ((x & mask) == prefix) {
                atomicAdd(&hist[(x >> shift) & 255], 1u);
            }
        }
        __syncthreads();
        if (tid == 0) {
            unsigned long long acc = 0;
            int b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= want) {
                    break;
                }
                acc += hist[b];
            }
            prefix_s = prefix | ((unsigned)b << shift);
            want_s = (unsigned)(want - acc);
        }
        __syncthreads();
        prefix = prefix_s;
        want = want_s;
        mask |= 255u << shift;
        __syncthreads();
    }
    if (tid == 0) {
        thr[blockIdx.x] = __uint_as_float(prefix);
    }
}

/* a warp a row (h, q) of p: the ids of its keys above the threshold of its slice, in order, then -1 */
__global__ void k_bmask_lists(const float *p, const float *thr, const int *ids, int *bl, int nq, int nk, int nb,
                              int heads, int q0, int nq_all, int nsel)
{
    PDL_START();
    int row = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) / 32), lane = threadIdx.x % 32;
    if (row >= heads * nq) {
        return;
    }
    int h = row / nq, q = row % nq;
    float t = thr[h * (nq / nb) + q / nb];
    const float *r = p + (size_t)row * nk;
    int *out = bl + ((size_t)h * nq_all + q0 + q) * nsel;
    int at = 0;
    for (int j0 = 0; j0 < nk; j0 += 32) {
        int j = j0 + lane;
        bool on = j < nk && r[j] > t;
        unsigned bal = __ballot_sync(RC_FULL_MASK, on);
        if (on) {
            int pos = at + __popc(bal & ((1u << lane) - 1u));
            if (pos < nsel) {
                out[pos] = ids[j];
            }
        }
        at += __popc(bal);
    }
    for (int j = at + lane; j < nsel; j += 32) {
        out[j] = -1;
    }
}

static void enc_block_mask(const float *qm, const float *km, const int *kr, const uint8_t *local, const int *ids,
                           float *p, float *thr, int *bl, int nq, int nk, int nb, int heads, int hd, int64_t topk,
                           int q0, int nq_all, int nsel, cudaStream_t s)
{
    int64_t tot = (int64_t)heads * nq * nk, n = (int64_t)nb * nk;
    int64_t k = topk < n - 1 ? topk : n - 1;
    k_bmask_scores<<<(unsigned)((tot + 255) / 256), 256, 0, s>>>(qm, km, kr, local, p, nq, nk, nb, heads, hd);
    k_bmask_softmax<<<(unsigned)(heads * nq), 256, 0, s>>>(p, nk);
    k_bmask_select<<<(unsigned)(heads * (nq / nb)), 1024, 0, s>>>(p, thr, n, k);
    k_bmask_lists<<<(unsigned)((heads * nq * 32 + 255) / 256), 256, 0, s>>>(p, thr, ids, bl, nq, nk, nb, heads, q0,
                                                                           nq_all, nsel);
}

/* ENC_ATTN_BLOCKS on the tensor cores (f16 2; sm_80, hd 128): k_enc_attn_tc's form (mma.sync m16n8k16: the
 * scores, the online softmax and the sums float32, q and the weights p float16) over the key blocks of a list: a
 * block of 8 warps takes query block blockIdx.x of head blockIdx.y, warp w its rows 16 w .. 16 w + 15 (their q
 * float16 in registers), and steps of EAB_K keys through the blocks bl[(h nqb_all + qb0 + qb) nsel ..] (-1 ends
 * them); K and V float16 by cp.async into two buffers (the next step's copies while a step's products run), K's
 * fragments by ldmatrix, V's by ldmatrix.trans. A query block with no key block gives zeros. */
#define EAB_K 64
#define EAB_LD (128 + 8)                                    /* the row stride (halves): 16 bytes apart */
#define EAB_SMEM ((size_t)4 * EAB_K * EAB_LD * sizeof(__half))  /* K and V, two buffers: 68 KB */

__global__ void __launch_bounds__(256, 1) k_enc_attn_blocks_tc(const float *q, const __half *k, const __half *v,
                                                                float *o, const int *bl, int nqb_all, int qb0,
                                                                int heads, int nsel)
{
    PDL_START();
    constexpr int HP = 128, LD = EAB_LD;
    extern __shared__ __align__(16) __half eabt_sm[];
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, c = lane % 4;
    const int h = blockIdx.y, qb = blockIdx.x;
    const size_t rs = (size_t)heads * HP;
    const int *L = bl + ((size_t)h * nqb_all + qb0 + qb) * nsel;
    int nb = 0;
    while (nb < nsel && L[nb] >= 0) {
        ++nb;
    }
    const int nst = 2 * nb;                 /* steps of EAB_K: 2 a key block */
    auto issue = [&](int i) {
        const int kr = L[i >> 1] * 128 + EAB_K * (i & 1);
        __half *ks = eabt_sm + (i & 1) * 2 * EAB_K * LD, *vs = ks + EAB_K * LD;
        #pragma unroll
        for (int u = 0; u < EAB_K * HP / 8 / 256; ++u) {
            int f = tid + 256 * u, j = f / (HP / 8), d = 8 * (f % (HP / 8));
            size_t at = (size_t)(kr + j) * rs + (size_t)h * HP + d;
            cp_async16(&ks[j * LD + d], k + at);
            cp_async16(&vs[j * LD + d], v + at);
        }
        cp_async_commit();
    };
    if (nst > 0) {
        issue(0);
    }
    const int r0 = qb * 128 + warp * 16 + g, r1 = r0 + 8;
    const float *q0 = q + (size_t)r0 * rs + (size_t)h * HP, *q1 = q + (size_t)r1 * rs + (size_t)h * HP;
    uint32_t qa[HP / 16][4];
    #pragma unroll
    for (int kk = 0; kk < HP / 16; ++kk) {
        int d0 = kk * 16 + 2 * c, d1 = d0 + 8;
        __half2 a0 = __float22half2_rn(*(const float2 *)&q0[d0]), a1 = __float22half2_rn(*(const float2 *)&q1[d0]);
        __half2 a2 = __float22half2_rn(*(const float2 *)&q0[d1]), a3 = __float22half2_rn(*(const float2 *)&q1[d1]);
        qa[kk][0] = *(uint32_t *)&a0;
        qa[kk][1] = *(uint32_t *)&a1;
        qa[kk][2] = *(uint32_t *)&a2;
        qa[kk][3] = *(uint32_t *)&a3;
    }
    float acc[HP / 8][4];
    #pragma unroll
    for (int i = 0; i < HP / 8; ++i) {
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    }
    const float L2E = 1.4426950408889634f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int i = 0; i < nst; ++i) {
        if (i + 1 < nst) {
            issue(i + 1);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();
        const __half *ks = eabt_sm + (i & 1) * 2 * EAB_K * LD, *vs = ks + EAB_K * LD;
        float sc[EAB_K / 8][4];
        #pragma unroll
        for (int nt = 0; nt < EAB_K / 8; ++nt) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
        }
        #pragma unroll
        for (int kk = 0; kk < HP / 16; kk += 2) {
            #pragma unroll
            for (int nt = 0; nt < EAB_K / 8; ++nt) {
                /* keys 8 nt + (lane % 8); the matrices: d 16 kk, + 8, 16 (kk + 1), + 8 */
                uint32_t b[4];
                ldsm_x4_enc(b, &ks[(nt * 8 + (lane & 7)) * LD + kk * 16 + (lane >> 3) * 8]);
                mma16816(sc[nt], qa[kk], b);
                mma16816(sc[nt], qa[kk + 1], b + 2);
            }
        }
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < EAB_K / 8; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                sc[nt][u] *= L2E;
                sc[nt][2 + u] *= L2E;
                mx0 = fmaxf(mx0, sc[nt][u]);
                mx1 = fmaxf(mx1, sc[nt][2 + u]);
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(RC_FULL_MASK, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(RC_FULL_MASK, mx1, off));
        }
        float f0 = exp2f(m0 - mx0), f1 = exp2f(m1 - mx1);   /* 0 at the first step */
        float ls0 = 0.f, ls1 = 0.f;
        uint32_t pa[EAB_K / 8][2];
        #pragma unroll
        for (int nt = 0; nt < EAB_K / 8; ++nt) {
            float e0 = exp2f(sc[nt][0] - mx0), e1 = exp2f(sc[nt][1] - mx0);
            float e2 = exp2f(sc[nt][2] - mx1), e3 = exp2f(sc[nt][3] - mx1);
            ls0 += e0 + e1;
            ls1 += e2 + e3;
            __half2 h01 = __floats2half2_rn(e0, e1), h23 = __floats2half2_rn(e2, e3);
            pa[nt][0] = *(uint32_t *)&h01;
            pa[nt][1] = *(uint32_t *)&h23;
        }
        for (int off = 1; off < 4; off <<= 1) {
            ls0 += __shfl_xor_sync(RC_FULL_MASK, ls0, off);
            ls1 += __shfl_xor_sync(RC_FULL_MASK, ls1, off);
        }
        l0 = l0 * f0 + ls0;
        l1 = l1 * f1 + ls1;
        m0 = mx0;
        m1 = mx1;
        #pragma unroll
        for (int nt = 0; nt < HP / 8; ++nt) {
            acc[nt][0] *= f0;
            acc[nt][1] *= f0;
            acc[nt][2] *= f1;
            acc[nt][3] *= f1;
        }
        #pragma unroll
        for (int kk = 0; kk < EAB_K / 16; ++kk) {
            uint32_t a[4] = {pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1]};
            #pragma unroll
            for (int nt = 0; nt < HP / 8; nt += 2) {
                /* keys 16 kk + (lane % 16); d 8 nt, then 8 (nt + 1) */
                uint32_t b[4];
                ldsm_x4_trans(b, &vs[(kk * 16 + (lane & 15)) * LD + nt * 8 + (lane >> 4) * 8]);
                mma16816(acc[nt], a, b);
                mma16816(acc[nt + 1], a, b + 2);
            }
        }
        __syncthreads();                    /* (the buffer free for step i + 2's copies) */
    }
    float inv0 = nst > 0 ? 1.0f / l0 : 0.f, inv1 = nst > 0 ? 1.0f / l1 : 0.f;
    float *o0 = o + (size_t)r0 * rs + (size_t)h * HP, *o1 = o + (size_t)r1 * rs + (size_t)h * HP;
    #pragma unroll
    for (int nt = 0; nt < HP / 8; ++nt) {
        int d = nt * 8 + 2 * c;
        *(float2 *)&o0[d] = make_float2(acc[nt][0] * inv0, acc[nt][1] * inv0);
        *(float2 *)&o1[d] = make_float2(acc[nt][2] * inv1, acc[nt][3] * inv1);
    }
}

/* ENC_ATTN_BLOCKS: nq / 128 query blocks of each head (98.5 KB of shared memory a block; f16 2 on the tensor
 * cores: 68 KB, hd 128, else as f16 1) */
static void enc_attn_blocks(const float *q, const void *k, const void *v, float *o, const int *bl, int nq, int heads,
                            int hd, int nsel, int f16, int qb0, int nqb_all, cudaStream_t s, int *bad)
{
    if (f16 == 2 && gg_caps.tc && hd == 128 && (size_t)gg_caps.smem_block_max >= EAB_SMEM) {
        static int set_tc = 0;
        if (!set_tc) {
            cudaFuncSetAttribute(k_enc_attn_blocks_tc, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)EAB_SMEM);
            set_tc = 1;
        }
        k_enc_attn_blocks_tc<<<dim3((unsigned)(nq / 128), (unsigned)heads), 256, EAB_SMEM, s>>>(
            q, (const __half *)k, (const __half *)v, o, bl, nqb_all, qb0, heads, nsel);
        return;
    }
    size_t sm = EAT_SMEM(128, 128, false);
    if ((size_t)gg_caps.smem_block_max < sm) {
        *bad = 1;           /* (a block of 98.5 KB: Volta and later) */
        return;
    }
    static int set[2] = {0, 0};
    dim3 grid((unsigned)(nq / 128), (unsigned)heads);
    if (f16) {
        if (!set[1]) {
            cudaFuncSetAttribute(k_enc_attn_blocks<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
            set[1] = 1;
        }
        k_enc_attn_blocks<true><<<grid, 256, sm, s>>>(q, k, v, o, bl, nqb_all, qb0, heads, hd, nsel);
    } else {
        if (!set[0]) {
            cudaFuncSetAttribute(k_enc_attn_blocks<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)sm);
            set[0] = 1;
        }
        k_enc_attn_blocks<false><<<grid, 256, sm, s>>>(q, k, v, o, bl, nqb_all, qb0, heads, hd, nsel);
    }
}

/* y (fs h x fs w pixels, cout channels) += the channels of x (h x w, cin) by
 * the pixel's phase: pixel (py, px) channel o reads x's pixel (py / fs, px /
 * fs), channel (o F + (ft - 1) fs^2 + (py % fs) fs + px % fs) / rep (F = ft
 * fs^2, rep = cout F / cin): DupUp3D of one frame. A thread a value. */
__global__ void k_enc_dupup(const float *x, float *y, int h, int w, int cin, int cout, int ft, int fs)
{
    PDL_START();
    int64_t W2 = (int64_t)fs * w, n = (int64_t)fs * h * W2 * cout;
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    int64_t p = t / cout, py = p / W2, px = p % W2;
    int o = (int)(t % cout), F = ft * fs * fs, rep = cout * F / cin;
    int ph = (ft - 1) * fs * fs + (int)(py % fs) * fs + (int)(px % fs);
    y[t] += x[((py / fs) * w + px / fs) * cin + (o * F + ph) / rep];
}

/* ENC_BF16_PACK: n float32 values to bfloat16 (h 0: enc_bf2's rounding, to nearest even) or float16 (h 1:
 * enc_h2's, saturated) as uint16: a thread 2 values */
__global__ void k_enc_bf16_pack(const float *x, uint16_t *y, int64_t n, int h)
{
    PDL_START();
    const int64_t i = 2 * ((int64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (i + 1 < n) {
        *(uint32_t *)(y + i) = h ? enc_h2(x[i], x[i + 1]) : enc_bf2(x[i], x[i + 1]);
    } else if (i < n) {
        y[i] = (uint16_t)(h ? enc_h2(x[i], 0.f) : enc_bf2(x[i], 0.f));
    }
}

/* float32 rows x (cols a multiple of 32) to F20 rows y: the bits of rc_kq_f32_to_f20 (kq/convert.c
 * kq_f20_encode_row). A warp a group of 32 values: each rounded to 16 significant bits (to nearest,
 * even; not finite, or rounded to Inf: 0), the group's largest exponent E, each value's gap to it (over
 * 15: the zero code); the row's padding bytes 0. */
__global__ void k_enc_f20_pack(const float *x, uint8_t *y, int64_t rows, int cols)
{
    PDL_START();
    const int ng = cols >> 5, lane = threadIdx.x & 31;
    const int64_t wi = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (wi >= rows * ng) {
        return;
    }
    const int64_t r = wi / ng;
    const int g = (int)(wi % ng);
    const size_t rb = f20_row_bytes(cols);
    uint8_t *o = y + r * rb;
    uint32_t v = __float_as_uint(x[r * cols + 32 * g + lane]);
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
    ((uint16_t *)o)[32 * g + lane] = (uint16_t)(sgn << 15 | m);
    const int up = __shfl_down_sync(0xffffffffu, gap, 16);
    if (lane < 16) {
        o[2 * (size_t)cols + 16 * g + lane] = (uint8_t)(gap | up << 4);
    }
    if (lane == 0) {
        o[2 * (size_t)cols + cols / 2 + g] = (uint8_t)E;
    }
    if (g == ng - 1 && lane == 0) {
        for (size_t b = 2 * (size_t)cols + cols / 2 + ng; b < rb; ++b) {
            o[b] = 0;
        }
    }
}

/* F20 rows x to float32 rows y (cols values; f20_at: the bits of kq_f20_bits). A thread a value. */
__global__ void k_enc_f20_unpack(const uint8_t *x, float *y, int64_t rows, int cols)
{
    PDL_START();
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * cols) {
        return;
    }
    const int64_t r = t / cols;
    y[t] = f20_at(x + r * f20_row_bytes(cols), cols, (int)(t % cols));
}

/* F20 rows x (rows, cols) to the two bfloat16 planes of their values (f20_at), y the hi plane and ylo
 * the lo one: the bits of k_to_bf16 of their float32 rows (ENC_ATTN_X2C's cache rows). A thread a value. */
__global__ void k_f20_to_bf16(const uint8_t *x, uint16_t *y, uint16_t *ylo, int64_t rows, int cols)
{
    PDL_START();
    const int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * cols) {
        return;
    }
    const int64_t r = t / cols;
    const float v = f20_at(x + r * f20_row_bytes(cols), cols, (int)(t % cols));
    const uint16_t h = f2bf_rn(v);
    y[t] = h;
    ylo[t] = f2bf_rn(v - __uint_as_float((uint32_t)h << 16));
}

/* y (cols, rows) = x (rows, cols) transposed: tiles of 32 x 32 through shared
 * memory (the values of ENC_ATTN_WIDE as the rows of a matrix). */
__global__ void k_transpose(const float *x, float *y, int rows, int cols)
{
    PDL_START();
    __shared__ float tile[32][33];
    int c0 = blockIdx.x * 32, r0 = blockIdx.y * 32;
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        int r = r0 + i, c = c0 + threadIdx.x;
        if (r < rows && c < cols) {
            tile[i][threadIdx.x] = x[(size_t)r * cols + c];
        }
    }
    __syncthreads();
    for (int i = threadIdx.y; i < 32; i += blockDim.y) {
        int c = c0 + i, r = r0 + threadIdx.x;
        if (r < rows && c < cols) {
            y[(size_t)c * rows + r] = tile[threadIdx.x][i];
        }
    }
}

/* Each row of s (rows, n) to its softmax in place: exp(s - max) / sum, a
 * block of 256 threads a row. */
__global__ void k_softmax_rows(float *s, int n)
{
    PDL_START();
    __shared__ float red[32];
    float *row = s + (size_t)blockIdx.x * n;
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, nw = blockDim.x / 32;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        mx = fmaxf(mx, row[j]);
    }
    for (int o = 16; o > 0; o /= 2) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    }
    if (lane == 0) {
        red[warp] = mx;
    }
    __syncthreads();
    mx = red[0];
    for (int i = 1; i < nw; ++i) {
        mx = fmaxf(mx, red[i]);
    }
    __syncthreads();
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        float v = expf(row[j] - mx);
        row[j] = v;
        sum += v;
    }
    for (int o = 16; o > 0; o /= 2) {
        sum += __shfl_xor_sync(0xffffffffu, sum, o);
    }
    if (lane == 0) {
        red[warp] = sum;
    }
    __syncthreads();
    sum = 0.f;
    for (int i = 0; i < nw; ++i) {
        sum += red[i];
    }
    float inv = 1.0f / sum;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        row[j] *= inv;
    }
}

/* ---------- 2D to stereo: ENC_RESIZE, ENC_DIBR, ENC_PACK_U8 ----------
 * The float operations of the CPU's (interp.c) one by one with the _rn
 * intrinsics (no contraction), so both give the same bits. */
__device__ __forceinline__ void enc_lerp_src(int64_t d, int64_t n_in, int64_t n_out, int mode, int64_t *i0,
                                             int64_t *i1, float *l0, float *l1)
{
    float src;
    if (mode == 1) {
        float scale = n_out > 1 ? __fdiv_rn((float)(n_in - 1), (float)(n_out - 1)) : 0.f;
        src = __fmul_rn(scale, (float)d);
    } else {
        float scale = __fdiv_rn((float)n_in, (float)n_out);
        src = __fsub_rn(__fmul_rn(scale, __fadd_rn((float)d, 0.5f)), 0.5f);
        if (src < 0.f) {
            src = 0.f;
        }
    }
    int64_t a = (int64_t)src;
    if (a > n_in - 1) {
        a = n_in - 1;
    }
    *i0 = a;
    *i1 = a < n_in - 1 ? a + 1 : a;
    *l1 = __fsub_rn(src, (float)a);
    *l0 = __fsub_rn(1.0f, *l1);
}

/* y (n images of H x W pixels of C: the columns yo.. of rows of ys) of x (n of h x w: the columns
 * xo.. of rows of xs): a thread a value */
__global__ void k_enc_resize(const float *x, float *y, int64_t n, int64_t h, int64_t w, int64_t H, int64_t W,
                             int64_t C, int mode, int64_t xs, int64_t xo, int64_t ys, int64_t yo)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * H * W * C) {
        return;
    }
    int64_t c = t % C, px = t / C, ox = px % W, oy = (px / W) % H, im = px / (W * H);
    const float *xi = x + im * h * w * xs + xo + c;
    float *yt = y + px * ys + yo + c;
    if (mode == 2) {
        int64_t fy = h / H, fx = w / W;
        float acc = 0.f;
        for (int64_t a = 0; a < fy; ++a) {
            for (int64_t b = 0; b < fx; ++b) {
                acc = __fadd_rn(acc, xi[((oy * fy + a) * w + ox * fx + b) * xs]);
            }
        }
        *yt = __fdiv_rn(acc, (float)(fy * fx));
        return;
    }
    int64_t y0, y1, x0, x1;
    float ly0, ly1, lx0, lx1;
    enc_lerp_src(oy, h, H, mode, &y0, &y1, &ly0, &ly1);
    enc_lerp_src(ox, w, W, mode, &x0, &x1, &lx0, &lx1);
    float top = __fadd_rn(__fmul_rn(lx0, xi[(y0 * w + x0) * xs]), __fmul_rn(lx1, xi[(y0 * w + x1) * xs]));
    float bot = __fadd_rn(__fmul_rn(lx0, xi[(y1 * w + x0) * xs]), __fmul_rn(lx1, xi[(y1 * w + x1) * xs]));
    *yt = __fadd_rn(__fmul_rn(ly0, top), __fmul_rn(ly1, bot));
}

/* ---------- frame interpolation: ENC_WARP, ENC_PRELU, ENC_MIX ----------
 * The float operations of interp.c's one by one with the _rn intrinsics (no
 * contraction): the same bits (ENC_MIX's expf apart). */

/* ENC_WARP: a thread a pixel (torch's CUDA grid_sampler_2d: bilinear, border (zeros: padding "zeros", not
 * clipped), align_corners) */
__global__ void k_enc_warp(const float *x, const float *fl, float *y, const float *gx, const float *gy, int64_t h,
                           int64_t w, int64_t C, int64_t xs, int64_t xo, int64_t fs, int64_t fo, int64_t ys,
                           int64_t yo, float dx, float dy, int zeros)
{
    PDL_START();
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= h * w) {
        return;
    }
    int64_t i = p / w, j = p % w;
    const float *f = fl + p * fs + fo;
    float gxv = __fadd_rn(gx[j], __fdiv_rn(f[0], dx)), gyv = __fadd_rn(gy[i], __fdiv_rn(f[1], dy));
    float ix = __fmul_rn(__fdiv_rn(__fadd_rn(gxv, 1.f), 2.f), (float)(w - 1));
    float iy = __fmul_rn(__fdiv_rn(__fadd_rn(gyv, 1.f), 2.f), (float)(h - 1));
    if (!zeros) {
        ix = fminf((float)(w - 1), fmaxf(ix, 0.f));
        iy = fminf((float)(h - 1), fmaxf(iy, 0.f));
    }
    float xw = floorf(ix), yn = floorf(iy), xe = __fadd_rn(xw, 1.f), ysth = __fadd_rn(yn, 1.f);
    float nw = __fmul_rn(__fsub_rn(xe, ix), __fsub_rn(ysth, iy));
    float ne = __fmul_rn(__fsub_rn(ix, xw), __fsub_rn(ysth, iy));
    float sw = __fmul_rn(__fsub_rn(xe, ix), __fsub_rn(iy, yn));
    float se = __fmul_rn(__fsub_rn(ix, xw), __fsub_rn(iy, yn));
    int64_t x0 = (int64_t)xw, y0 = (int64_t)yn, x1 = x0 + 1, y1 = y0 + 1;
    bool okw = x0 >= 0 && x0 < w, oke = x1 >= 0 && x1 < w, okn = y0 >= 0 && y0 < h, oks = y1 >= 0 && y1 < h;
    const float *xb = x + xo;
    float *o = y + p * ys + yo;
    for (int64_t c = 0; c < C; ++c) {
        float acc = 0.f;
        if (okw && okn) {
            acc = __fadd_rn(acc, __fmul_rn(xb[(y0 * w + x0) * xs + c], nw));
        }
        if (oke && okn) {
            acc = __fadd_rn(acc, __fmul_rn(xb[(y0 * w + x1) * xs + c], ne));
        }
        if (okw && oks) {
            acc = __fadd_rn(acc, __fmul_rn(xb[(y1 * w + x0) * xs + c], sw));
        }
        if (oke && oks) {
            acc = __fadd_rn(acc, __fmul_rn(xb[(y1 * w + x1) * xs + c], se));
        }
        o[c] = acc;
    }
}

/* ---------- RIFE's block inputs and output without the full-resolution buffer (ENC_RIFE_IN, ENC_RIFE_OUT) ----------
 * The values the im2col program computes at full resolution (models/vfi/rife.py: the flow F, the
 * frames and heads warped by it, the upsampled mask and features: its buffer A) computed only at
 * the pixels a block's input resize reads, each by the same _rn operations in the same order as
 * k_vfi_resize, k_enc_mul_vec, k_enc_add, k_enc_warp and k_enc_mix: the same bits. */

/* the bilinear upsample (mode 0) of column c of a block's tmp (hs x ws pixels of 13) at (py, px) */
__device__ __forceinline__ float rife_up(const float *tmp, int64_t hs, int64_t ws, int64_t Hp, int64_t Wp, int c,
                                         int64_t py, int64_t px)
{
    int64_t y0, y1, x0, x1;
    float ly0, ly1, lx0, lx1;
    enc_lerp_src(py, hs, Hp, 0, &y0, &y1, &ly0, &ly1);
    enc_lerp_src(px, ws, Wp, 0, &x0, &x1, &lx0, &lx1);
    float top = __fadd_rn(__fmul_rn(lx0, tmp[(y0 * ws + x0) * 13 + c]), __fmul_rn(lx1, tmp[(y0 * ws + x1) * 13 + c]));
    float bot = __fadd_rn(__fmul_rn(lx0, tmp[(y1 * ws + x0) * 13 + c]), __fmul_rn(lx1, tmp[(y1 * ws + x1) * 13 + c]));
    return __fadd_rn(__fmul_rn(ly0, top), __fmul_rn(ly1, bot));
}

struct RifeTmps {
    const float *t[5];
    int s[5];               /* each block's scale (its tmp: Hp / s x Wp / s) */
    int n;                  /* the blocks done */
};

/* the flow at (py, px) after the blocks of tm: s0 up(tmp0) (k_enc_mul_vec), + s_k up(tmp_k) (k_enc_add) */
__device__ __forceinline__ void rife_flow(const RifeTmps &tm, int64_t Hp, int64_t Wp, int64_t py, int64_t px,
                                          float *f)
{
    for (int k = 0; k < tm.n; ++k) {
        const int64_t hs = Hp / tm.s[k], ws = Wp / tm.s[k];
        const float sk = (float)tm.s[k];
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float d = rife_up(tm.t[k], hs, ws, Hp, Wp, c, py, px);
            f[c] = k == 0 ? __fmul_rn(d, sk) : __fadd_rn(f[c], __fmul_rn(sk, d));
        }
    }
}

/* k_enc_warp's sample of C channels of img (rows of C) at (i, j) plus the flow (fx, fy), into o */
__device__ __forceinline__ void rife_warp(const float *img, int C, const float *gx, const float *gy, int64_t h,
                                          int64_t w, int64_t i, int64_t j, float fx, float fy, float dx, float dy,
                                          float *o)
{
    float gxv = __fadd_rn(gx[j], __fdiv_rn(fx, dx)), gyv = __fadd_rn(gy[i], __fdiv_rn(fy, dy));
    float ix = __fmul_rn(__fdiv_rn(__fadd_rn(gxv, 1.f), 2.f), (float)(w - 1));
    float iy = __fmul_rn(__fdiv_rn(__fadd_rn(gyv, 1.f), 2.f), (float)(h - 1));
    ix = fminf((float)(w - 1), fmaxf(ix, 0.f));
    iy = fminf((float)(h - 1), fmaxf(iy, 0.f));
    float xw = floorf(ix), yn = floorf(iy), xe = __fadd_rn(xw, 1.f), ysth = __fadd_rn(yn, 1.f);
    float nw = __fmul_rn(__fsub_rn(xe, ix), __fsub_rn(ysth, iy));
    float ne = __fmul_rn(__fsub_rn(ix, xw), __fsub_rn(ysth, iy));
    float sw = __fmul_rn(__fsub_rn(xe, ix), __fsub_rn(iy, yn));
    float se = __fmul_rn(__fsub_rn(ix, xw), __fsub_rn(iy, yn));
    int64_t x0 = (int64_t)xw, y0 = (int64_t)yn, x1 = x0 + 1, y1 = y0 + 1;
    bool okw = x0 >= 0 && x0 < w, oke = x1 >= 0 && x1 < w, okn = y0 >= 0 && y0 < h, oks = y1 >= 0 && y1 < h;
    for (int c = 0; c < C; ++c) {
        float acc = 0.f;
        if (okw && okn) {
            acc = __fadd_rn(acc, __fmul_rn(img[(y0 * w + x0) * C + c], nw));
        }
        if (oke && okn) {
            acc = __fadd_rn(acc, __fmul_rn(img[(y0 * w + x1) * C + c], ne));
        }
        if (okw && oks) {
            acc = __fadd_rn(acc, __fmul_rn(img[(y1 * w + x0) * C + c], sw));
        }
        if (oke && oks) {
            acc = __fadd_rn(acc, __fmul_rn(img[(y1 * w + x1) * C + c], se));
        }
        o[c] = acc;
    }
}

/* A's 28 values at the full-resolution pixel (py, px) before block tm.n: the frames and heads
 * warped by the flow (0:14), 0 (t's column: the bias puts t), the last block's mask and features
 * (15:24), the flow (24:28) */
__device__ __forceinline__ void rife_a(const float *x0, const float *x1, const float *f0, const float *f1,
                                       const RifeTmps &tm, const float *gx, const float *gy, int64_t Hp, int64_t Wp,
                                       float dx, float dy, int64_t py, int64_t px, float *a)
{
    float fl[4];
    rife_flow(tm, Hp, Wp, py, px, fl);
    rife_warp(x0, 3, gx, gy, Hp, Wp, py, px, fl[0], fl[1], dx, dy, a + 0);
    rife_warp(x1, 3, gx, gy, Hp, Wp, py, px, fl[2], fl[3], dx, dy, a + 3);
    rife_warp(f0, 4, gx, gy, Hp, Wp, py, px, fl[0], fl[1], dx, dy, a + 6);
    rife_warp(f1, 4, gx, gy, Hp, Wp, py, px, fl[2], fl[3], dx, dy, a + 10);
    a[14] = 0.f;
    const int k = tm.n - 1;
    const int64_t hs = Hp / tm.s[k], ws = Wp / tm.s[k];
    #pragma unroll
    for (int c = 0; c < 9; ++c) {
        a[15 + c] = rife_up(tm.t[k], hs, ws, Hp, Wp, 4 + c, py, px);
    }
    #pragma unroll
    for (int c = 0; c < 4; ++c) {
        a[24 + c] = fl[c];
    }
}

/* ENC_RIFE_IN: block tm.n's input (hs x ws pixels of 28) = A resized (mode 0) to 1 / s, times m,
 * plus tb: a thread a pixel. With s 1 the resize reads one pixel (its other taps' weights are 0). */
__global__ void k_enc_rife_in(const float *x0, const float *x1, const float *f0, const float *f1, RifeTmps tm,
                              const float *gx, const float *gy, float *xs, const float *m, const float *tb,
                              int64_t Hp, int64_t Wp, int s, float dx, float dy)
{
    PDL_START();
    const int64_t hs = Hp / s, ws = Wp / s;
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= hs * ws) {
        return;
    }
    const int64_t oy = p / ws, ox = p % ws;
    int64_t yy[2], xx[2];
    float ly[2], lx[2];
    enc_lerp_src(oy, Hp, hs, 0, &yy[0], &yy[1], &ly[0], &ly[1]);
    enc_lerp_src(ox, Wp, ws, 0, &xx[0], &xx[1], &lx[0], &lx[1]);
    float row[2][28];
    for (int r = 0; r < 2; ++r) {
        float a0[28], a1[28];
        rife_a(x0, x1, f0, f1, tm, gx, gy, Hp, Wp, dx, dy, yy[r], xx[0], a0);
        if (lx[1] != 0.f || s != 1) {
            rife_a(x0, x1, f0, f1, tm, gx, gy, Hp, Wp, dx, dy, yy[r], xx[1], a1);
        }
        #pragma unroll
        for (int c = 0; c < 28; ++c) {
            row[r][c] = (lx[1] != 0.f || s != 1) ? __fadd_rn(__fmul_rn(lx[0], a0[c]), __fmul_rn(lx[1], a1[c]))
                                                 : __fmul_rn(lx[0], a0[c]);
        }
        if (ly[1] == 0.f && s == 1) {
            #pragma unroll
            for (int c = 0; c < 28; ++c) {
                row[1][c] = 0.f;
            }
            break;
        }
    }
    float *o = xs + p * 28;
    #pragma unroll
    for (int c = 0; c < 28; ++c) {
        float v = __fadd_rn(__fmul_rn(ly[0], row[0][c]), __fmul_rn(ly[1], row[1][c]));
        v = __fmul_rn(v, m[c]);
        o[c] = __fadd_rn(v, tb[c]);
    }
}

/* ENC_RIFE_OUT: the output a full-resolution pixel: the frames warped by the last flow, merged by
 * the last mask (k_enc_mix); w0, w1 (when not NULL) the warped frames (the cut probe) */
__global__ void k_enc_rife_out(const float *x0, const float *x1, RifeTmps tm, const float *gx, const float *gy,
                               float *out, float *w0, float *w1, int64_t Hp, int64_t Wp, float dx, float dy)
{
    PDL_START();
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= Hp * Wp) {
        return;
    }
    const int64_t py = p / Wp, px = p % Wp;
    float fl[4], a[3], b[3];
    rife_flow(tm, Hp, Wp, py, px, fl);
    rife_warp(x0, 3, gx, gy, Hp, Wp, py, px, fl[0], fl[1], dx, dy, a);
    rife_warp(x1, 3, gx, gy, Hp, Wp, py, px, fl[2], fl[3], dx, dy, b);
    const int k = tm.n - 1;
    const float msk = rife_up(tm.t[k], Hp / tm.s[k], Wp / tm.s[k], Hp, Wp, 4, py, px);
    float sg = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-msk)));
    float t = __fsub_rn(1.f, sg);
    for (int c = 0; c < 3; ++c) {
        out[p * 3 + c] = __fadd_rn(__fmul_rn(a[c], sg), __fmul_rn(b[c], t));
        if (w0 != nullptr) {
            w0[p * 3 + c] = a[c];
            w1[p * 3 + c] = b[c];
        }
    }
}

/* the launches of ENC_RIFE_IN and ENC_RIFE_OUT from their records (hlit, hi: runner.cu's) */
static void enc_rife_in(const gp_rec *r, const int64_t *e, int *bad, cudaStream_t s)
{
    /* x0, x1, f0, f1, t0 .. t3, xs, m, tb, gx, gy, Hp, Wp, s, n, s0 .. s3, dx, dy */
    RifeTmps tm;
    tm.n = (int)hlit(r, 16, bad);
    for (int k = 0; k < 5; ++k) {
        tm.t[k] = k < 4 ? (const float *)(intptr_t)hi(r, e, 4 + k) : nullptr;
        tm.s[k] = k < 4 ? (int)hlit(r, 17 + k, bad) : 1;
    }
    const int64_t Hp = hlit(r, 13, bad), Wp = hlit(r, 14, bad);
    const int sc = (int)hlit(r, 15, bad);
    if (tm.n < 1 || tm.n > 4 || sc < 1) {
        *bad = 1;
        return;
    }
    const int64_t n = (Hp / sc) * (Wp / sc);
    k_enc_rife_in<<<(unsigned)((n + 127) / 128), 128, 0, s>>>(
        (const float *)(intptr_t)hi(r, e, 0), (const float *)(intptr_t)hi(r, e, 1), (const float *)(intptr_t)hi(r, e, 2),
        (const float *)(intptr_t)hi(r, e, 3), tm, (const float *)(intptr_t)hi(r, e, 11),
        (const float *)(intptr_t)hi(r, e, 12), (float *)(intptr_t)hi(r, e, 8), (const float *)(intptr_t)hi(r, e, 9),
        (const float *)(intptr_t)hi(r, e, 10), Hp, Wp, sc, hlitf(r, 21, bad), hlitf(r, 22, bad));
}

static void enc_rife_out(const gp_rec *r, const int64_t *e, int *bad, cudaStream_t s)
{
    /* x0, x1, t0 .. t4, out, w0, w1, gx, gy, Hp, Wp, n, s0 .. s4, dx, dy */
    RifeTmps tm;
    tm.n = (int)hlit(r, 14, bad);
    for (int k = 0; k < 5; ++k) {
        tm.t[k] = (const float *)(intptr_t)hi(r, e, 2 + k);
        tm.s[k] = (int)hlit(r, 15 + k, bad);
    }
    const int64_t Hp = hlit(r, 12, bad), Wp = hlit(r, 13, bad);
    if (tm.n != 5) {
        *bad = 1;
        return;
    }
    k_enc_rife_out<<<(unsigned)((Hp * Wp + 127) / 128), 128, 0, s>>>(
        (const float *)(intptr_t)hi(r, e, 0), (const float *)(intptr_t)hi(r, e, 1), tm,
        (const float *)(intptr_t)hi(r, e, 10), (const float *)(intptr_t)hi(r, e, 11), (float *)(intptr_t)hi(r, e, 7),
        (float *)(intptr_t)hi(r, e, 8), (float *)(intptr_t)hi(r, e, 9), Hp, Wp, hlitf(r, 20, bad),
        hlitf(r, 21, bad));
}

/* ENC_PRELU: a thread a value */
__global__ void k_enc_prelu(const float *x, float *y, int64_t rows, int64_t cols, const float *a)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= rows * cols) {
        return;
    }
    float v = x[t];
    y[t] = v > 0.f ? v : __fmul_rn(v, a[t % cols]);
}

/* ENC_MIX: a thread a pixel */
__global__ void k_enc_mix(const float *x, float *y, int64_t n, int64_t xs, int64_t ao, int64_t bo, int64_t mo,
                          int64_t C, int64_t ys, int64_t yo)
{
    PDL_START();
    int64_t p = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) {
        return;
    }
    const float *r = x + p * xs;
    float s = __fdiv_rn(1.f, __fadd_rn(1.f, expf(-r[mo])));
    float t = __fsub_rn(1.f, s);
    for (int64_t c = 0; c < C; ++c) {
        y[p * ys + yo + c] = __fadd_rn(__fmul_rn(r[ao + c], s), __fmul_rn(r[bo + c], t));
    }
}

/* ENC_UNPACK_U8P: a thread a value of the (Hp Wp, C) rows; the bytes of an h x w frame, zeros past it
 * (ENC_UNPACK_U8's division: the bits of an unpack of the frame padded with zero bytes) */
__global__ void k_enc_unpack_u8p(const unsigned char *x, float *y, int64_t h, int64_t w, int64_t Hp, int64_t Wp,
                                 int64_t C, float d)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= Hp * Wp * C) {
        return;
    }
    int64_t p = t / C, c = t % C, yy = p / Wp, xx = p % Wp;
    y[t] = __fdiv_rn((yy < h && xx < w) ? (float)x[(yy * w + xx) * C + c] : 0.f, d);
}

/* ENC_PACK_U8C: a thread a byte of the h x w frame: the value of its pixel of the (Hp Wp, C) rows
 * times s, rounded and clamped (ENC_MUL_VEC then ENC_PACK_U8 of the rows, cropped) */
__global__ void k_enc_pack_u8c(const float *x, unsigned char *y, int64_t h, int64_t w, int64_t Wp, int64_t C,
                               float s)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= h * w * C) {
        return;
    }
    int64_t p = t / C, c = t % C;
    float v = rintf(__fmul_rn(x[((p / w) * Wp + p % w) * C + c], s));
    y[t] = (unsigned char)(v < 0.f ? 0.f : v > 255.f ? 255.f : v);
}

/* ENC_RESAMPLE_U8: one pass of PIL's 8-bit resample along an axis of bytes (ported from Pillow's
 * src/libImaging/Resample.c, ImagingResampleHorizontal_8bpc and ImagingResampleVertical_8bpc; MIT-CMU
 * License; Copyright (c) 1997-2011 by Secret Labs AB, Copyright (c) 1995-2011 by Fredrik Lundh and
 * contributors, Copyright (c) 2010 by Jeffrey 'Alex' Clark and contributors; see THIRD_PARTY_NOTICES.md). Output value (a, o, b), a < outer, o < n,
 * b < inner: 2^21 + the sum over j < bounds[2 o + 1] of x[a sa + (bounds[2 o] + j) so + b] k[o ks + j]
 * in int32 (normalize_coeffs_8bpc's coefficients), shifted right by 22 and clamped to 0..255 */
__global__ void k_enc_resample_u8(const unsigned char *x, unsigned char *y, const int *bounds, const int *k,
                                  int64_t outer, int64_t n, int64_t inner, int64_t sa, int64_t so, int64_t ks)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= outer * n * inner) {
        return;
    }
    int64_t b = t % inner, o = (t / inner) % n, a = t / (inner * n);
    const unsigned char *src = x + a * sa + (int64_t)bounds[2 * o] * so + b;
    const int *kr = k + o * ks;
    int ss = 1 << 21;
    for (int j = 0; j < bounds[2 * o + 1]; ++j) {
        ss += (int)src[j * so] * kr[j];
    }
    int v = ss >> 22;
    y[t] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
}

/* ENC_THUMB_U8: the pixels of rows x cols (a thread a byte) */
__global__ void k_enc_take_u8(const unsigned char *x, unsigned char *y, const int *rows, const int *cols,
                              int64_t nr, int64_t nc, int64_t w, int64_t C)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nr * nc * C) {
        return;
    }
    int64_t c = t % C, j = (t / C) % nc, i = t / (C * nc);
    y[t] = x[((int64_t)rows[i] * w + cols[j]) * C + c];
}

/* ENC_THUMB_U8: the sum of the three channels over block (i, j): rows ry0[i] .. ry1[i] - 1, columns
 * rx0[j] .. rx1[j] - 1 (a thread block a block; integers, so the order does not matter) */
__global__ void k_enc_block_sums_u8(const unsigned char *x, int *y, const int *ry0, const int *ry1, const int *rx0,
                                    const int *rx1, int64_t nx, int64_t w)
{
    PDL_START();
    int i = blockIdx.x / (int)nx, j = blockIdx.x % (int)nx;
    int y0 = ry0[i], x0 = rx0[j], bw = rx1[j] - x0, n = (ry1[i] - y0) * bw;
    int s = 0;
    for (int q = threadIdx.x; q < n; q += blockDim.x) {
        const unsigned char *p = x + ((int64_t)(y0 + q / bw) * w + x0 + q % bw) * 3;
        s += (int)p[0] + (int)p[1] + (int)p[2];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_down_sync(0xffffffffu, s, off);
    }
    __shared__ int part[32];
    if ((threadIdx.x & 31) == 0) {
        part[threadIdx.x >> 5] = s;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int tot = 0;
        for (int q = 0; q < (int)(blockDim.x >> 5); ++q) {
            tot += part[q];
        }
        y[blockIdx.x] = tot;
    }
}

__device__ __forceinline__ unsigned enc_orderable(float f)
{
    unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ float enc_unorder(unsigned u)
{
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7fffffffu) : ~u);
}

/* a thread a source pixel: its box and the segment to its right neighbour (keys zeroed before) */
__global__ void k_dibr_splat(const float *s, unsigned long long *keys, int64_t H, int64_t W, float sign, float gap)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * W) {
        return;
    }
    int64_t y = t / W, j = t % W;
    const float *sr = s + y * W;
    unsigned long long *kr = keys + y * W;
    float sj = sr[j];
    float p = __fadd_rn((float)j, __fmul_rn(sign, sj));
    unsigned long long kb = ((unsigned long long)enc_orderable(sj) << 32) | ((unsigned long long)j << 1);
    int64_t lo = (int64_t)ceilf(__fsub_rn(p, 0.5f)), hi = (int64_t)ceilf(__fadd_rn(p, 0.5f));
    for (int64_t X = lo; X < hi; ++X) {
        if (X >= 0 && X < W) {
            atomicMax(kr + X, kb);
        }
    }
    if (j + 1 >= W) {
        return;
    }
    float sb = sr[j + 1];
    float p1 = __fadd_rn((float)(j + 1), __fmul_rn(sign, sb));
    float span = __fsub_rn(p1, p);
    if (!(span > 0.f && span <= __fadd_rn(1.0f, gap))) {
        return;
    }
    lo = (int64_t)ceilf(p);
    hi = (int64_t)ceilf(p1);
    float ds = __fsub_rn(sb, sj);
    for (int64_t X = lo; X < hi; ++X) {
        if (X < 0 || X >= W) {
            continue;
        }
        float tt = __fdiv_rn(__fsub_rn((float)X, p), span);
        float sx = __fadd_rn(sj, __fmul_rn(tt, ds));
        unsigned long long k = ((unsigned long long)enc_orderable(sx) << 32) | ((unsigned long long)j << 1) | 1ull;
        atomicMax(kr + X, k);
    }
}

/* a thread a target pixel: the winner's colour into tmp, its s into zb */
__global__ void k_dibr_resolve(const float *img, const float *s, const unsigned long long *keys, float *zb,
                               float *tmp, int64_t H, int64_t W, float sign)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * W) {
        return;
    }
    int64_t y = t / W, X = t % W;
    unsigned long long k = keys[t];
    float *o = tmp + t * 3;
    if (k == 0) {
        o[0] = o[1] = o[2] = 0.f;
        return;
    }
    zb[t] = enc_unorder((unsigned)(k >> 32));
    int64_t j = (int64_t)((k >> 1) & 0x7fffffffull);
    const float *a = img + (y * W + j) * 3;
    if (!(k & 1ull)) {
        o[0] = a[0], o[1] = a[1], o[2] = a[2];
        return;
    }
    const float *sr = s + y * W;
    float p0 = __fadd_rn((float)j, __fmul_rn(sign, sr[j]));
    float p1 = __fadd_rn((float)(j + 1), __fmul_rn(sign, sr[j + 1]));
    float tt = __fdiv_rn(__fsub_rn((float)X, p0), __fsub_rn(p1, p0));
    const float *b = a + 3;
    for (int c = 0; c < 3; ++c) {
        o[c] = __fadd_rn(a[c], __fmul_rn(tt, __fsub_rn(b[c], a[c])));
    }
}

/* a thread a target pixel: a hole takes its farther drawn neighbour's colour and s */
__global__ void k_dibr_stretch(const unsigned long long *keys, float *zb, float *tmp, int64_t H, int64_t W)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * W || keys[t]) {
        return;
    }
    int64_t y = t / W, X = t % W;
    const unsigned long long *kr = keys + y * W;
    int64_t L = X - 1, R = X + 1;
    while (L >= 0 && !kr[L]) {
        --L;
    }
    while (R < W && !kr[R]) {
        ++R;
    }
    int64_t src = -1;
    if (L >= 0 && (R >= W || kr[L] <= kr[R])) {
        src = L;
    } else if (R < W) {
        src = R;
    }
    float *o = tmp + t * 3;
    if (src < 0) {
        o[0] = o[1] = o[2] = 0.f;
        zb[t] = -INFINITY;
        return;
    }
    const float *a = tmp + (y * W + src) * 3;
    o[0] = a[0], o[1] = a[1], o[2] = a[2];
    zb[t] = zb[y * W + src];
}

/* a thread a pixel: tmp into the eye's place in out (all: 0; the holes: 1, back: 2) */
__global__ void k_dibr_copy(const unsigned long long *keys, float *tmp, float *out, int64_t H, int64_t W,
                            int64_t off, int64_t stride, int what)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * W || (what && keys[t])) {
        return;
    }
    float *o = out + off + (t / W) * stride + (t % W) * 3, *a = tmp + t * 3;
    if (what == 2) {
        a[0] = o[0], a[1] = o[1], a[2] = o[2];
    } else {
        o[0] = a[0], o[1] = a[1], o[2] = a[2];
    }
}

/* a thread a hole pixel: its fill (interp.c's gp_dibr_fill) into out */
__global__ void k_dibr_fill(const unsigned long long *keys, const float *zb, const float *tmp, float *out, int64_t H,
                            int64_t W, int64_t off, int64_t stride, int mode, float tol)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= H * W || keys[t]) {
        return;
    }
    int64_t y = t / W, X = t % W;
    float *o = out + off + y * stride + X * 3;
    float acc[3] = {0.f, 0.f, 0.f};
    if (mode == 0) {
        for (int d = -2; d <= 2; ++d) {
            int64_t xx = X + d < 0 ? 0 : X + d > W - 1 ? W - 1 : X + d;
            const float *a = tmp + (y * W + xx) * 3;
            for (int c = 0; c < 3; ++c) {
                acc[c] = __fadd_rn(acc[c], a[c]);
            }
        }
        for (int c = 0; c < 3; ++c) {
            o[c] = __fdiv_rn(acc[c], 5.0f);
        }
        return;
    }
    float lim = __fadd_rn(zb[t], tol);
    float cnt = 0.f;
    for (int64_t dy = -1; dy <= 1; ++dy) {
        int64_t yy = y + dy;
        if (yy < 0 || yy >= H) {
            continue;
        }
        for (int64_t dx = -1; dx <= 1; ++dx) {
            int64_t xx = X + dx;
            if (xx < 0 || xx >= W) {
                continue;
            }
            int64_t q = yy * W + xx;
            if (keys[q] && !(zb[q] <= lim)) {
                continue;
            }
            const float *a = tmp + q * 3;
            for (int c = 0; c < 3; ++c) {
                acc[c] = __fadd_rn(acc[c], a[c]);
            }
            cnt = __fadd_rn(cnt, 1.0f);
        }
    }
    for (int c = 0; c < 3; ++c) {
        o[c] = __fdiv_rn(acc[c], cnt);
    }
}

/* byte i of y = x[i] rounded (to nearest even) and clamped to 0..255 */
__global__ void k_enc_pack_u8(const float *x, unsigned char *y, int64_t n)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) {
        return;
    }
    float v = rintf(x[t]);
    y[t] = (unsigned char)(v < 0.f ? 0.f : v > 255.f ? 255.f : v);
}

/* y of x by each axis's taps (ENC_RESIZE_TAPS): a thread a value, the columns' sum first */
__global__ void k_enc_resize_taps(const float *x, float *y, int64_t n, int64_t h, int64_t w, int64_t H, int64_t W,
                                  int64_t C, int64_t ky, const int *iy, const float *wy, int64_t kx, const int *ix,
                                  const float *wx)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n * H * W * C) {
        return;
    }
    int64_t c = t % C, px = t / C, ox = px % W, oy = (px / W) % H, im = px / (W * H);
    const float *xi = x + im * h * w * C + c;
    float acc = 0.f;
    for (int64_t k = 0; k < ky; ++k) {
        const float *row = xi + (int64_t)iy[oy * ky + k] * w * C;
        float hs = 0.f;
        for (int64_t j = 0; j < kx; ++j) {
            hs = __fadd_rn(hs, __fmul_rn(wx[ox * kx + j], row[(int64_t)ix[ox * kx + j] * C]));
        }
        acc = __fadd_rn(acc, __fmul_rn(wy[oy * ky + k], hs));
    }
    y[t] = acc;
}

/* y[i] = byte i of x / d (ENC_UNPACK_U8) */
__global__ void k_enc_unpack_u8(const unsigned char *x, float *y, int64_t n, float d)
{
    PDL_START();
    int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (t < n) {
        y[t] = __fdiv_rn((float)x[t], d);
    }
}
