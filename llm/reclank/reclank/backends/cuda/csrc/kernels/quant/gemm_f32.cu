/* GP_KQ_LINEAR of a large group in float32, for a GPU without dp4a (Maxwell): the KQ_NV4, KQ_IQ2R and KQ_Q8_R
 * rows of Nemotron's prompt groups. The generic k_kq_gemm (kq_tile: 64 x 64, a thread 4 x 4, no second
 * buffer) took ~370 GMAC/s on the Quadro M4000, 30% of its float32 peak: a group of 1024 tokens of Bolt
 * 18.9 s of its 31. Here a block takes 128 rows x BN tokens (128, or 64 for a smaller group), a thread
 * 8 x BN / 16 (rows ty 4 + i and 64 + ty 4 + i, tokens tx 4 + j and 64 + tx 4 + j), 16 columns a step; the
 * bytes of the next step load before the products of this one and decode after them, into the other of
 * two buffers in shared memory. The M4000, NV4 18560 x 4096 of 1024 tokens: 108 ms, 721 GMAC/s (with the
 * decode right after the loads 621). The sums are float32 in an order of their own. */
#define KF_BM 128
#define KF_BK 16
#define KF_LD (KF_BM + 4)

/* the decoders (row(w, cols): a row's own; IQG: the IQ2 grid in shared memory): load(row, cols, c0) the bytes of the 8 values c0 .. c0 + 7 (c0 a multiple of 8), decode()
 * the values */
struct kf_nv4 {
    struct raw {
        uint2 q;
        uint32_t s;
    };
    float g;
    __device__ void row(const uint8_t *w, int cols)
    {
        g = 0.5f * *(const float *)(w + cols / 2 + cols / 16);
    }
    __device__ raw load(const uint8_t *w, int cols, int c0) const
    {
        const int b = c0 / 32, j = c0 % 32;
        raw r;
        r.q = *(const uint2 *)(w + 16 * b + (j % 16));
        r.s = w[cols / 2 + 2 * b + (j >= 16)];
        return r;
    }
    __device__ void decode(const raw &r, int c0, const uint64_t *, float *v) const
    {
        const int sh = (c0 % 32) >= 16 ? 4 : 0;
        const float sc = g * kq_e4m3((uint8_t)r.s);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const uint32_t q = u < 4 ? r.q.x >> (8 * u) : r.q.y >> (8 * (u - 4));
            v[u] = sc * kq_e2m1x2((q >> sh) & 15);
        }
    }
};

/* (iq2.cuh's rows: part j of block b, the d after the parts; the grid in shared memory) */
struct kf_iq2r {
    struct raw {
        uint2 pt;
        uint32_t dh;
    };
    __device__ void row(const uint8_t *, int) {}
    __device__ raw load(const uint8_t *w, int cols, int c0) const
    {
        raw r;
        r.pt = nm_q2_part(w, c0 / 256, (c0 % 256) / 32);
        r.dh = nm_q2_dh(w, cols, c0 / 256);
        return r;
    }
    __device__ void decode(const raw &r, int c0, const uint64_t *grid, float *v) const
    {
        const int l = (c0 % 32) / 8;
        const uint64_t g = grid[(r.pt.x >> (8 * l)) & 255];
        const int s7 = (r.pt.y >> (7 * l)) & 127, sg = s7 | ((__popc(s7) & 1) << 7);
        const float db = nm_q2_db((uint16_t)r.dh, r.pt.y);
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float q = db * (float)((g >> (8 * u)) & 255);
            v[u] = ((sg >> u) & 1) ? -q : q;
        }
    }
};

struct kf_q8r {
    struct raw {
        uint2 q;
        uint32_t dh;
    };
    __device__ void row(const uint8_t *, int) {}
    __device__ raw load(const uint8_t *w, int cols, int c0) const
    {
        raw r;
        r.q = *(const uint2 *)(w + c0);
        r.dh = ((const uint16_t *)(w + cols))[c0 / 32];
        return r;
    }
    __device__ void decode(const raw &r, int, const uint64_t *, float *v) const
    {
        const float d = __half2float(__ushort_as_half((uint16_t)r.dh));
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            v[u] = d * (float)(int8_t)((r.q.x >> (8 * u)) & 255);
            v[4 + u] = d * (float)(int8_t)((r.q.y >> (8 * u)) & 255);
        }
    }
};

/* W rows of rb bytes (rows x cols), X t x cols, O t x rows */
template <class DQ, int BN, bool IQG>
__global__ void __launch_bounds__(256, 2) k_kq_gemm_f(const uint8_t *W, size_t rb, int rows, int cols, const float *X,
                                                    int t, float *O)
{
    constexpr int KTN = BN / 16, KXL = BN * KF_BK / 256;      /* a thread's tokens; its x of a step */
    __shared__ __align__(16) float Ws[2][KF_BK][KF_LD];
    __shared__ __align__(16) float Xs[2][KF_BK][BN + 4];
    __shared__ uint64_t kf_grid[IQG ? 256 : 1];
    const int tid = threadIdx.x, tx = tid % 16, ty = tid / 16;
    const int r0 = blockIdx.x * KF_BM, j0 = blockIdx.y * BN;
    if (IQG) {
        kf_grid[tid] = nm_iq2_grid[tid];
    }
    /* the loads: thread tid takes row tid % 128 (columns 8 (tid / 128) .. + 7 of each step) and token
     * tid % BN (columns KXL (tid / BN) .. + KXL - 1) */
    const int lr = tid % 128, lk = 8 * (tid / 128), xr = tid % BN, xk = KXL * (tid / BN);
    const uint8_t *wrow = r0 + lr < rows ? W + (size_t)(r0 + lr) * rb : NULL;
    const float *xrow = j0 + xr < t ? X + (size_t)(j0 + xr) * cols : NULL;
    DQ dq;
    if (wrow != NULL) {
        dq.row(wrow, cols);
    }
    typename DQ::raw wq;
    float4 xv[KXL / 4];
    auto load = [&](int k0) {
        if (wrow != NULL) {
            wq = dq.load(wrow, cols, k0 + lk);
        }
#pragma unroll
        for (int u = 0; u < KXL / 4; ++u) {
            xv[u] = xrow != NULL ? *(const float4 *)(xrow + k0 + xk + 4 * u) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
    };
    auto store = [&](int buf, int k0) {
        float wv[8];
        if (wrow != NULL) {
            dq.decode(wq, k0 + lk, kf_grid, wv);
        } else {
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                wv[u] = 0.f;
            }
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            Ws[buf][lk + u][lr] = wv[u];
        }
#pragma unroll
        for (int u = 0; u < KXL / 4; ++u) {
            Xs[buf][xk + 4 * u][xr] = xv[u].x;
            Xs[buf][xk + 4 * u + 1][xr] = xv[u].y;
            Xs[buf][xk + 4 * u + 2][xr] = xv[u].z;
            Xs[buf][xk + 4 * u + 3][xr] = xv[u].w;
        }
    };
    float acc[8][KTN];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
#pragma unroll
        for (int j = 0; j < KTN; ++j) {
            acc[i][j] = 0.f;
        }
    }
    if (IQG) {
        __syncthreads();
    }
    load(0);
    store(0, 0);
    __syncthreads();
    int buf = 0;
    for (int k0 = 0; k0 < cols; k0 += KF_BK) {
        const bool more = k0 + KF_BK < cols;
        if (more) {
            load(k0 + KF_BK);
        }
#pragma unroll
        for (int kk = 0; kk < KF_BK; ++kk) {
            const float4 a0 = *(const float4 *)&Ws[buf][kk][4 * ty], a1 = *(const float4 *)&Ws[buf][kk][64 + 4 * ty];
            const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
            float b[KTN];
            const float4 b0 = *(const float4 *)&Xs[buf][kk][4 * tx];
            b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
            if (KTN == 8) {
                const float4 b1 = *(const float4 *)&Xs[buf][kk][64 + 4 * tx];
                b[KTN - 4] = b1.x; b[KTN - 3] = b1.y; b[KTN - 2] = b1.z; b[KTN - 1] = b1.w;
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < KTN; ++j) {
                    acc[i][j] += a[i] * b[j];
                }
            }
        }
        if (more) {
            store(buf ^ 1, k0 + KF_BK);
            __syncthreads();
            buf ^= 1;
        }
    }
    /* O[token][row]: a thread's 4 rows a float4 */
#pragma unroll
    for (int j = 0; j < KTN; ++j) {
        const int tok = j0 + (j < 4 ? 4 * tx + j : 64 + 4 * tx + j - 4);
        if (tok >= t) {
            continue;
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = r0 + 64 * h + 4 * ty;
            float *o = O + (size_t)tok * rows + row;
            if (row + 3 < rows) {
                *(float4 *)o = make_float4(acc[4 * h][j], acc[4 * h + 1][j], acc[4 * h + 2][j], acc[4 * h + 3][j]);
            } else {
                for (int i = 0; i < 4 && row + i < rows; ++i) {
                    o[i] = acc[4 * h + i][j];
                }
            }
        }
    }
}

/* the float32 GEMM of KQ_LINEAR (t > MT_MAX) for type, when it has a decoder here (else 1: k_kq_gemm) */
template <class DQ, bool IQG>
static void kq_gemm_f_go(const float *x, const uint8_t *w, size_t rb, int rows, int cols, int t, float *out,
                         cudaStream_t s)
{
    if (t <= 64) {
        k_kq_gemm_f<DQ, 64, IQG><<<dim3((unsigned)((rows + KF_BM - 1) / KF_BM), (unsigned)((t + 63) / 64)), 256, 0, s>>>(
            w, rb, rows, cols, x, t, out);
    } else {
        k_kq_gemm_f<DQ, 128, IQG><<<dim3((unsigned)((rows + KF_BM - 1) / KF_BM), (unsigned)((t + 127) / 128)), 256, 0, s>>>(
            w, rb, rows, cols, x, t, out);
    }
}

static int kq_gemm_f_launch(int type, const float *x, const uint8_t *w, size_t rb, int rows, int cols, int t,
                            float *out, cudaStream_t s)
{
    if (cols % KF_BK != 0) {
        return 1;
    }
    if (type == KQ_NV4) {
        kq_gemm_f_go<kf_nv4, false>(x, w, rb, rows, cols, t, out, s);
    } else if (type == KQ_IQ2R && cols % 256 == 0) {
        kq_gemm_f_go<kf_iq2r, true>(x, w, rb, rows, cols, t, out, s);
    } else if (type == KQ_Q8_R) {
        kq_gemm_f_go<kf_q8r, false>(x, w, rb, rows, cols, t, out, s);
    } else {
        return 1;
    }
    return 0;
}
