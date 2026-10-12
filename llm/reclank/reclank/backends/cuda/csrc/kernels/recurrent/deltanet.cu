/* ---------- the DeltaNet of a large group (no log): parallel forms ----------
 * k_gdn_heads runs one block for each value head (48 blocks) and three block
 * sums for each token: about 4 us a token, 1.25 s of a bf16 prompt of 8192
 * tokens. But only the state needs the order of the tokens, and each column
 * of the state is on its own: the norms of q and k, the decay and beta do
 * not read it, and the norm of the output is a pass after it. So:
 *
 *   k_gdn_conv_par  the convolution, one thread for each (token, channel)
 *                   (the same sums, in the same order, as k_gdn_conv);
 *   k_gdn_prep      q and k of each head to their norms (in place in cv),
 *                   decay and beta to operands 5 and 6 (in place), and the
 *                   last kernel - 1 inputs to the state of the convolution;
 *   k_gdn_scan      the state: GS threads for each column, each with KD / GS
 *                   values of it, the tokens in steps of GDN_TC in shared
 *                   memory; the output without its norm to operand 11;
 *   k_gdn_out       the norm of the output and the gate.
 *
 * The sums of a column go in GS parts, so the bits differ a little from those
 * of the steps. A group with a log (the MTP verify) keeps k_gdn_heads. */
__global__ void k_gdn_conv_par(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int kernel = DI(3);
    int cd = 2 * DI(14) * DI(16) + DI(15) * DI(17);
    int c = blockIdx.x * blockDim.x + threadIdx.x, i = blockIdx.y;
    if (c >= cd || i >= gdn_nreal(r, e)) {
        return;
    }
    const float *qkv = DP(const float, 0), *conv = DP(const float, 1);
    const float *w = DP(const float, 2) + (size_t)c * kernel;
    float v = w[kernel - 1] * qkv[(size_t)i * cd + c];
    for (int j = 0; j < kernel - 1; ++j) {
        int p = i - (kernel - 1) + j;
        float h = p >= 0 ? qkv[(size_t)p * cd + c] : conv[(size_t)(p + kernel - 1) * cd + c];
        v += w[j] * h;
    }
    DP(float, 12)[(size_t)i * cd + c] = qw_silu(v);
}

template <int KD>
__global__ void __launch_bounds__(KD) k_gdn_prep(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int kh = DI(14), vh = DI(15), t = gdn_nreal(r, e), kernel = DI(3);
    int kd = kh * KD, cd = 2 * kd + vh * KD;
    int i = blockIdx.x, x = blockIdx.y, c = threadIdx.x;
    if (x < kh) {
        if (i >= t) {
            return;
        }
        float *row = DP(float, 12) + (size_t)i * cd;
        float qv = row[x * KD + c], kv = row[kd + x * KD + c];
        float nq = block_sum(qv * qv), nk = block_sum(kv * kv);
        row[x * KD + c] = qv * (1.f / sqrtf(nq + 1e-6f) / sqrtf((float)KD));
        row[kd + x * KD + c] = kv * (1.f / sqrtf(nk + 1e-6f));
    } else if (x == kh) {
        if (i >= t || c >= vh) {
            return;
        }
        float Aexp = expf(DP(const float, 7)[c]);
        float *a = DP(float, 5) + (size_t)i * vh + c, *b = DP(float, 6) + (size_t)i * vh + c;
        float av = *a + DP(const float, 8)[c];
        float sp = av > 20.f ? av : log1pf(expf(av));
        *a = expf(-Aexp * sp);
        *b = 1.f / (1.f + expf(-*b));
    } else {
        /* the state of the convolution: the last kernel - 1 inputs (rows of
         * the old state when t < kernel - 1; row j reads row t + j > j) */
        const float *qkv = DP(const float, 0);
        float *conv = DP(float, 1);
        for (int ch = i * KD + c; ch < cd; ch += gridDim.x * KD) {
            for (int j = 0; j < kernel - 1; ++j) {
                int p = t - (kernel - 1) + j;
                conv[(size_t)j * cd + ch] = p >= 0 ? qkv[(size_t)p * cd + ch]
                                                   : conv[(size_t)(t + j) * cd + ch];
            }
        }
    }
}

#define GDN_TC 32
template <int KD, int GS>
__global__ void __launch_bounds__(128) k_gdn_scan(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    constexpr int NV = KD / GS, CB = 128 / GS;     /* values of a thread, columns of a block */
    __shared__ __align__(16) float qs[GDN_TC][KD], ks[GDN_TC][KD];
    __shared__ float dec[GDN_TC], bet[GDN_TC];
    int kh = DI(14), vh = DI(15), t = gdn_nreal(r, e), tiled = DI(20) & 1;
    int kd = kh * KD, vd = vh * KD, cd = 2 * kd + vd;
    int hv = blockIdx.x, part = threadIdx.x % GS;
    int col = blockIdx.y * CB + threadIdx.x / GS;
    int hk = tiled ? hv % kh : hv / (vh / kh);
    const float *cv = DP(const float, 12), *da = DP(const float, 5), *db = DP(const float, 6);
    float *Sh = DP(float, 10) + (size_t)hv * KD * KD, *out = DP(float, 11);
    /* thread value u: d = 4 GS (u / 4) + 4 part + u % 4 (float4 reads of q and
     * k with no bank conflicts) */
    float S[NV];
    #pragma unroll
    for (int u = 0; u < NV; ++u) {
        S[u] = Sh[(size_t)(4 * GS * (u / 4) + 4 * part + u % 4) * KD + col];
    }
    for (int i0 = 0; i0 < t; i0 += GDN_TC) {
        int n = min(GDN_TC, t - i0);
        __syncthreads();
        for (int z = threadIdx.x; z < n * KD / 4; z += 128) {
            int ii = z / (KD / 4), d4 = (z % (KD / 4)) * 4;
            const float *row = cv + (size_t)(i0 + ii) * cd;
            *(float4 *)&qs[ii][d4] = *(const float4 *)(row + hk * KD + d4);
            *(float4 *)&ks[ii][d4] = *(const float4 *)(row + kd + hk * KD + d4);
        }
        if (threadIdx.x < n) {
            dec[threadIdx.x] = da[(size_t)(i0 + threadIdx.x) * vh + hv];
            bet[threadIdx.x] = db[(size_t)(i0 + threadIdx.x) * vh + hv];
        }
        __syncthreads();
        for (int ii = 0; ii < n; ++ii) {
            float decay = dec[ii], beta = bet[ii];
            float vv = cv[(size_t)(i0 + ii) * cd + 2 * kd + hv * KD + col];
            float s0 = 0.f, s1 = 0.f;
            #pragma unroll
            for (int u4 = 0; u4 < NV / 4; ++u4) {
                float4 k4 = *(const float4 *)&ks[ii][4 * GS * u4 + 4 * part];
                S[4 * u4 + 0] = __fmul_rn(S[4 * u4 + 0], decay);
                S[4 * u4 + 1] = __fmul_rn(S[4 * u4 + 1], decay);
                S[4 * u4 + 2] = __fmul_rn(S[4 * u4 + 2], decay);
                S[4 * u4 + 3] = __fmul_rn(S[4 * u4 + 3], decay);
                s0 += k4.x * S[4 * u4 + 0] + k4.y * S[4 * u4 + 1];
                s1 += k4.z * S[4 * u4 + 2] + k4.w * S[4 * u4 + 3];
            }
            float sk = s0 + s1;
            #pragma unroll
            for (int o2 = 1; o2 < GS; o2 <<= 1) {
                sk += __shfl_xor_sync(RC_FULL_MASK, sk, o2);
            }
            float delta = (vv - sk) * beta;
            float o0 = 0.f, o1 = 0.f;
            #pragma unroll
            for (int u4 = 0; u4 < NV / 4; ++u4) {
                float4 k4 = *(const float4 *)&ks[ii][4 * GS * u4 + 4 * part];
                float4 q4 = *(const float4 *)&qs[ii][4 * GS * u4 + 4 * part];
                S[4 * u4 + 0] = __fmaf_rn(k4.x, delta, S[4 * u4 + 0]);
                S[4 * u4 + 1] = __fmaf_rn(k4.y, delta, S[4 * u4 + 1]);
                S[4 * u4 + 2] = __fmaf_rn(k4.z, delta, S[4 * u4 + 2]);
                S[4 * u4 + 3] = __fmaf_rn(k4.w, delta, S[4 * u4 + 3]);
                o0 += q4.x * S[4 * u4 + 0] + q4.y * S[4 * u4 + 1];
                o1 += q4.z * S[4 * u4 + 2] + q4.w * S[4 * u4 + 3];
            }
            float o = o0 + o1;
            #pragma unroll
            for (int o2 = 1; o2 < GS; o2 <<= 1) {
                o += __shfl_xor_sync(RC_FULL_MASK, o, o2);
            }
            if (part == 0) {
                out[(size_t)(i0 + ii) * vd + hv * KD + col] = o;
            }
        }
    }
    #pragma unroll
    for (int u = 0; u < NV; ++u) {
        Sh[(size_t)(4 * GS * (u / 4) + 4 * part + u % 4) * KD + col] = S[u];
    }
}

template <int KD>
__global__ void __launch_bounds__(KD) k_gdn_out(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int vh = DI(15), flags = DI(20);
    int i = blockIdx.x, hv = blockIdx.y, c = threadIdx.x;
    size_t oi = (size_t)i * vh * KD + (size_t)hv * KD + c;
    float *out = DP(float, 11);
    if (i >= gdn_nreal(r, e)) {
        out[oi] = 0.f;          /* a padding row: zeros (see k_gdn_heads) */
        return;
    }
    float o = out[oi];
    float ss = block_sum(o * o);
    float inv = 1.f / sqrtf(ss / (float)KD + df(r, e, 18));
    float zv = DP(const float, 4)[oi];
    float zg = (flags & 2) ? 1.f / (1.f + expf(-zv)) : qw_silu(zv);
    out[oi] = o * inv * DP(const float, 9)[c] * zg;
}

/* RECLANK_GPU_GDN_PAR: the threads of a column of the state in k_gdn_scan
 * for a group of at least GDN_PAR_MIN tokens with no log (4 or 8; 0 keeps
 * k_gdn_heads). */
#define GDN_PAR_MIN 64
static int gdn_par_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_GDN_PAR");
        on = v == NULL ? 4 : atoi(v);
    }
    return on;
}

/* rc_gpu_gdn_commit: the first n tokens of a log to conv and S. */
__global__ void k_gdn_commit_conv(float *conv, const float *log, int n, int kernel, int cd,
                                  size_t lrow)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= cd) {
        return;
    }
    float hist[8];
    for (int j = 0; j < kernel - 1; ++j) {
        hist[j] = conv[(size_t)j * cd + c];
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < kernel - 2; ++j) {
            hist[j] = hist[j + 1];
        }
        hist[kernel - 2] = log[(size_t)i * lrow + c];
    }
    for (int j = 0; j < kernel - 1; ++j) {
        conv[(size_t)j * cd + c] = hist[j];
    }
}

template <int KD>
__global__ void __launch_bounds__(KD) k_gdn_commit_heads(float *S, const float *log, int n,
                                                         int cd, size_t lrow)
{
    int hv = blockIdx.x, c = threadIdx.x;
    float *Sh = S + (size_t)hv * KD * KD;
    float st[KD];
    #pragma unroll
    for (int d = 0; d < KD; ++d) {
        st[d] = Sh[(size_t)d * KD + c];
    }
    for (int i = 0; i < n; ++i) {
        const float *lg = log + (size_t)i * lrow + cd + (size_t)hv * (2 * KD + 1);
        float decay = lg[2 * KD], delta = lg[KD + c];
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            st[d] = __fmul_rn(st[d], decay);
        }
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            st[d] = __fmaf_rn(lg[d], delta, st[d]);
        }
    }
    #pragma unroll
    for (int d = 0; d < KD; ++d) {
        Sh[(size_t)d * KD + c] = st[d];
    }
}

/* GP_KQ_HOT_MOE: the experts that the GPU holds, and the shared expert.
 *
 *     h, val, idx, map, gate, up, down, act, act2, de, out, top_k, inner,
 *     hidden, gtype, dtype, t, sgate, sup, sdown, stype, slog
 *
 * gate, up, and down hold the experts of the slots (map gives the slot of
 * each expert, or -1), one after the other. Pair j < t top_k is slot
 * j % top_k of token j / top_k; pair t top_k + j is the shared expert of
 * token j (always on the GPU), with the weight sigmoid(slog[j]). The steps
 * are those of GP_HOT_MOE: gate and up, silu(gate) up, down, and the sum. */
/* The pool of HotCache (gpu.ExpertPool): expert blocks on the GPU in
 * segments of GG_POOL_K blocks that come and go (the room lent by the image
 * encoder, the free memory). A slot value KQH_WARM + i is block i: segment
 * i / GG_POOL_K, place i % GG_POOL_K, at a stride of its part (the bytes of
 * an expert of the layers that use the pool). rc_gpu_set_pool writes the table;
 * the graphs read it at run time, and a change adds segments or clears ones
 * that no slot uses. The host sets the count of segments (enough for the
 * memory of the GPU); the table lives in device memory. */
/* (KQH_WARM, GG_POOL_K and the table of the device: kernels/operands.cu) */
static const uint8_t **gg_pool_tab;             /* the device table */
static int gg_pool_cap;                          /* its segments */

/* tab: the host table of the pool (moe.c moe_pool_addr): the 3 strides, the
 * count n of segments, then 3 n segment addresses, part by part */
extern "C" int rc_gpu_set_pool(const int64_t *tab)
{
    const int n = (int)tab[3];
    if (n <= 0) {
        return -1;
    }
    if (n > gg_pool_cap) {
        const uint8_t **d = NULL;
        if (cudaMalloc((void **)&d, (size_t)3 * n * sizeof(*d)) != cudaSuccess) {
            return -1;
        }
        if (gg_pool_tab != NULL) {
            cudaFree(gg_pool_tab);
        }
        gg_pool_tab = d;
        gg_pool_cap = n;
        if (cudaMemcpyToSymbol(gg_pool_seg, &d, sizeof(d)) != cudaSuccess) {
            return -1;
        }
    }
    if (cudaMemcpy(gg_pool_tab, tab + 4, (size_t)3 * n * sizeof(int64_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        return -1;
    }
    size_t st[3] = {(size_t)tab[0], (size_t)tab[1], (size_t)tab[2]};
    if (cudaMemcpyToSymbol(gg_pool_nseg, &n, sizeof(n)) != cudaSuccess) {
        return -1;
    }
    return cudaMemcpyToSymbol(gg_pool_stride, st, sizeof(st)) == cudaSuccess ? 0 : -1;
}

/* Fences (gpumm.GpuMem): rc_gpu_fence records an event on the stream of the
 * programs after the work queued so far (a run returns before the GPU is
 * done) and returns its number; rc_gpu_fence_done(n, wait) gives 1 when that
 * work is done (wait: wait for it), 0 if not yet, -1 on an error. The events
 * are a ring: a fence older than the ring is done. */
#define GG_FENCES 4096
static cudaEvent_t gg_fev[GG_FENCES];
static int64_t gg_fseq;

extern "C" int64_t rc_gpu_fence(void)
{
    if (gg_fseq == 0) {
        for (int k = 0; k < GG_FENCES; ++k) {
            if (cudaEventCreateWithFlags(&gg_fev[k], cudaEventDisableTiming) != cudaSuccess) {
                return -1;
            }
        }
    }
    int64_t n = ++gg_fseq;
    if (cudaEventRecord(gg_fev[n % GG_FENCES], gg_stream) != cudaSuccess) {
        return -1;
    }
    return n;
}

extern "C" int rc_gpu_fence_done(int64_t n, int wait)
{
    if (n <= 0 || gg_fseq - n >= GG_FENCES) {
        return 1;
    }
    cudaEvent_t ev = gg_fev[n % GG_FENCES];
    cudaError_t e = wait ? cudaEventSynchronize(ev) : cudaEventQuery(ev);
    if (e == cudaErrorNotReady) {
        return 0;
    }
    return e == cudaSuccess ? 1 : -1;
}

extern "C" int rc_gpu_pool_dims(int *k)
{
    *k = GG_POOL_K;
    return 0;
}

/* The bytes of slot of part (0 gate, 1 up, 2 down; per: the bytes of one
 * expert of that part): the store of the layer (operands 4, 5, 6), or a
 * block of the pool. */
__device__ __forceinline__ const uint8_t *kqh_wslot(const gp_rec *r, const int64_t *e, int slot,
                                                    int part, size_t per)
{
    if (slot >= KQH_WARM) {
        return pool_block(slot, part);
    }
    return DP(const uint8_t, 4 + part) + (size_t)slot * per;
}

__device__ __forceinline__ int kqh_slot(const gp_rec *r, const int64_t *e, int j)
{
    int tk = DI(16) * DI(11);
    return j >= tk ? 0 : DP(const int, 3)[DP(const int, 2)[j]];
}

/* Operands 22 and 23 (or none): zc (the flags of the pairs that read the
 * pinned copy of the experts in host memory, GP_HOT_SPLIT), and a table of
 * the device addresses of the gate, up, and down experts of the copy
 * (mapped: the GPU reads them over PCIe). */
__device__ __forceinline__ int kqh_zc(const gp_rec *r, const int64_t *e, int j)
{
    return r->tag[22] != GP_T_NONE && j < DI(16) * DI(11) && DP(const int, 22)[j] != 0;
}

/* A pair that the GPU computes: a hot expert, the shared expert, or a cold
 * expert of the copy (kqh_zc). */
__device__ __forceinline__ int kqh_on(const gp_rec *r, const int64_t *e, int j)
{
    return kqh_slot(r, e, j) >= 0 || kqh_zc(r, e, j);
}

__global__ void k_kqh_gu(const gp_rec *r, const int64_t *e, int jofs = 0)
{
    PDL_START();
    int j = blockIdx.y + jofs, slot = kqh_slot(r, e, j), up = blockIdx.z;
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int inner = DI(12), hidden = DI(13), k = DI(11), tk = DI(16) * k;
    if (slot < 0 || row >= inner) {
        return;
    }
    int shared = j >= tk;
    int type = shared ? DI(20) : DI(14);
    size_t rb = kq_row_bytes(type, hidden);
    const uint8_t *w = shared ? DP(const uint8_t, up ? 18 : 17)
                              : kqh_wslot(r, e, slot, up, (size_t)inner * rb);
    int tok = shared ? j - tk : j / k;
    if (type == KQ_NVX) {
        /* a warp for each group of 16 rows (row counts the groups) */
        if (row >= inner / 16) {
            return;
        }
        float v[4];
        kq_nvx_group(w + (size_t)16 * row * rb, DP(const float, 0) + (size_t)tok * hidden, hidden, v);
        int l = threadIdx.x % 32;
        if (l < 4) {
            float *o = DP(float, 7) + (size_t)j * 2 * inner + up * inner + 16 * row;
            o[2 * l] = v[0];
            o[2 * l + 1] = v[1];
            o[2 * l + 8] = v[2];
            o[2 * l + 9] = v[3];
        }
        return;
    }
    float v = kq_row(type, w + (size_t)row * rb, DP(const float, 0) + (size_t)tok * hidden, hidden);
    if (threadIdx.x % 32 == 0) {
        DP(float, 7)[(size_t)j * 2 * inner + up * inner + row] = v;
    }
}

/* RQ8_0 experts (gguf.RQ8_0, Qwen4GPU): the MoE kernels rotate the act of
 * each pair before the down product, as kquants.c kq_moe_rot (the program
 * rotates their x). A state of the process. */
static int gg_moe_rot = 0;

extern "C" void rc_gpu_set_moe_rot(int on)
{
    gg_moe_rot = on;
}

/* rot: the RQ8_0 experts (gg_moe_rot): the act rotated in each 32 values
 * (a warp a group; inner and blockDim are multiples of 32) */
__global__ void k_kqh_act(const gp_rec *r, const int64_t *e, int rot)
{
    PDL_START();
    int j = blockIdx.x;
    if (!kqh_on(r, e, j)) {
        return;
    }
    int inner = DI(12);
    const float *g = DP(const float, 7) + (size_t)j * 2 * inner;
    for (int i = threadIdx.x; i < inner; i += blockDim.x) {
        float v = qw_silu(g[i]) * g[inner + i];
        if (rot) {
            v = tq6_rot_warp(v, i % 32, false);
        }
        DP(float, 8)[(size_t)j * inner + i] = v;
    }
}

__global__ void k_kqh_dn(const gp_rec *r, const int64_t *e, int jofs = 0)
{
    PDL_START();
    int j = blockIdx.y + jofs, slot = kqh_slot(r, e, j);
    int row = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int inner = DI(12), hidden = DI(13), tk = DI(16) * DI(11);
    if (slot < 0 || row >= hidden) {
        return;
    }
    int shared = j >= tk;
    int type = shared ? DI(20) : DI(15);
    size_t rb = kq_row_bytes(type, inner);
    const uint8_t *w = shared ? DP(const uint8_t, 19)
                              : kqh_wslot(r, e, slot, 2, (size_t)hidden * rb);
    if (type == KQ_NVX) {
        if (row >= hidden / 16) {
            return;
        }
        float v[4];
        kq_nvx_group(w + (size_t)16 * row * rb, DP(const float, 8) + (size_t)j * inner, inner, v);
        int l = threadIdx.x % 32;
        if (l < 4) {
            float *o = DP(float, 9) + (size_t)j * hidden + 16 * row;
            o[2 * l] = v[0];
            o[2 * l + 1] = v[1];
            o[2 * l + 8] = v[2];
            o[2 * l + 9] = v[3];
        }
        return;
    }
    float v = kq_row(type, w + (size_t)row * rb, DP(const float, 8) + (size_t)j * inner, inner);
    if (threadIdx.x % 32 == 0) {
        DP(float, 9)[(size_t)j * hidden + row] = v;
    }
}

/* The KQ_NVX experts of GP_KQ_HOT_MOE (the pairs j < t top_k): a block of KS
 * warps for each group of 16 rows (blockIdx.x), pair (blockIdx.y), and
 * matrix (blockIdx.z: gate or up; down with dn). Warp w takes the blocks
 * of 32 columns w, w + KS, ... of the group (kq_nvx_group for a part of
 * the columns), and the warps add their sums in shared memory. With a warp
 * for each whole group (k_kqh_gu) a step had about 250 KB of loads in
 * flight: 166 GB/s, 72 us for the gate and up of a layer. */
template <int KS>
__global__ void __launch_bounds__(32 * KS) k_kqh_nvx(const gp_rec *r, const int64_t *e, int dn)
{
    PDL_START();
    __shared__ float red[KS][16];
    int grp = blockIdx.x, j = blockIdx.y, up = blockIdx.z;
    int slot = kqh_slot(r, e, j), zc = slot < 0 && kqh_zc(r, e, j);
    if (slot < 0 && !zc) {
        return;
    }
    int inner = DI(12), hidden = DI(13), k = DI(11);
    int cols = dn ? inner : hidden;
    size_t rb = kq_row_bytes(KQ_NVX, cols);
    const uint8_t *w;
    if (zc) {
        /* expert idx[j] of the copy in host memory, over PCIe */
        size_t ex = (size_t)DP(const int, 2)[j];
        const int64_t *zt = DP(const int64_t, 23);
        w = dn ? (const uint8_t *)(intptr_t)zt[2] + ex * hidden * rb
               : (const uint8_t *)(intptr_t)zt[up ? 1 : 0] + ex * inner * rb;
    } else {
        w = dn ? kqh_wslot(r, e, slot, 2, (size_t)hidden * rb)
               : kqh_wslot(r, e, slot, up, (size_t)inner * rb);
    }
    const uint8_t *wg = w + (size_t)16 * grp * rb;
    const float *x = dn ? DP(const float, 8) + (size_t)j * inner
                        : DP(const float, 0) + (size_t)(j / k) * hidden;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, st = lane / 4, q = lane % 4, h = st / 4;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    #pragma unroll 2
    for (int b = warp; b < cols / 32; b += KS) {
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
    if (lane < 4) {
        red[warp][2 * lane] = acc[0];
        red[warp][2 * lane + 1] = acc[1];
        red[warp][2 * lane + 8] = acc[2];
        red[warp][2 * lane + 9] = acc[3];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float sm = 0.f;
        #pragma unroll
        for (int i = 0; i < KS; ++i) {
            sm += red[i][threadIdx.x];
        }
        sm *= 128.f * *(const float *)wg;            /* 0.5 g, and the 2^8 of kq_e4m3s */
        if (dn) {
            DP(float, 9)[(size_t)j * hidden + 16 * grp + threadIdx.x] = sm;
        } else {
            DP(float, 7)[(size_t)j * 2 * inner + up * inner + 16 * grp + threadIdx.x] = sm;
        }
    }
}

/* The KQ_NVX experts of GP_KQ_HOT_MOE on int8 x (xq, xs of kt_quant), or
 * int16 x as two int8 planes (X2: xq the high plane, xl the low one, xs of
 * kt_quant_x2), with the operations of the CPU (kq_nvx_rows): for each block
 * of 32 columns the exact sums a0, a1 of each half (the codes as twice
 * their values, int8, by dp4a; X2: 128 hi + lo), p = a1 s1 + a0 s0 (one
 * fma), then for each row f = fma(p, xs, f) over the blocks in their order,
 * and f 128 g. The warps take the blocks as k_kqh_nvx; each p goes to
 * shared memory, and one thread for each row adds them in order. The bits
 * of the CPU, for a token alone and in a group (scripts/check_nvx_hot.py).
 * RECLANK_GPU_KQH_X chooses (x2, the default, as RECLANK_MOE_X16 of the CPU;
 * i8; f32: k_kqh_nvx, float32 x, not the bits of the CPU). */
#define KQH_NVX_NB 128      /* the most blocks of 32 columns (4096 columns) */
template <int KS, bool X2>
__global__ void __launch_bounds__(32 * KS) k_kqh_nvx_q(const gp_rec *r, const int64_t *e, int dn,
                                                       const int8_t *xq, const int8_t *xl, const float *xs)
{
    PDL_START();
    __shared__ float pb[KQH_NVX_NB][16];
    int grp = blockIdx.x, j = blockIdx.y, up = blockIdx.z;
    int slot = kqh_slot(r, e, j), zc = slot < 0 && kqh_zc(r, e, j);
    if (slot < 0 && !zc) {
        return;
    }
    int inner = DI(12), hidden = DI(13), k = DI(11);
    int cols = dn ? inner : hidden, nb = cols / 32;
    size_t rb = kq_row_bytes(KQ_NVX, cols);
    const uint8_t *w;
    if (zc) {
        size_t ex = (size_t)DP(const int, 2)[j];
        const int64_t *zt = DP(const int64_t, 23);
        w = dn ? (const uint8_t *)(intptr_t)zt[2] + ex * hidden * rb
               : (const uint8_t *)(intptr_t)zt[up ? 1 : 0] + ex * inner * rb;
    } else {
        w = dn ? kqh_wslot(r, e, slot, 2, (size_t)hidden * rb)
               : kqh_wslot(r, e, slot, up, (size_t)inner * rb);
    }
    const uint8_t *wg = w + (size_t)16 * grp * rb;
    /* the int8 row: the act of pair j (down), else token j / k */
    size_t x0 = dn ? (size_t)j * inner : (size_t)(j / k) * hidden;
    const int8_t *xh = xq + x0;
    const float *xsr = xs + x0 / 32;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, st = lane / 4, q = lane % 4;
    for (int b = warp; b < nb; b += KS) {
        const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
        uint2 c = *(const uint2 *)(blk + 8 * lane);
        int xv = *(const int *)(xh + 32 * b + 4 * st);
        int wv[4] = {(int)kt_e2m1x4(c.x), (int)kt_e2m1x4(c.y), (int)kt_e2m1x4(c.x >> 4),
                     (int)kt_e2m1x4(c.y >> 4)};
        int a[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] = __dp4a(wv[i], xv, 0);
        }
        if (X2) {
            int xlv = *(const int *)(xl + x0 + 32 * b + 4 * st);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = 128 * a[i] + __dp4a(wv[i], xlv, 0);
            }
        }
        /* the 4 steps of each half (lane bits 2 and 3), then the other half */
        int o[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            a[i] += __shfl_xor_sync(RC_FULL_MASK, a[i], 4);
            a[i] += __shfl_xor_sync(RC_FULL_MASK, a[i], 8);
            o[i] = __shfl_xor_sync(RC_FULL_MASK, a[i], 16);
        }
        if (lane < 4) {
            /* rows 2q, 2q + 1 (low 4 bits), 2q + 8, 2q + 9 (high) */
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int row = 2 * q + (i & 1) + (i >> 1) * 8;
                float s0 = kq_e4m3s(blk[256 + row]), s1 = kq_e4m3s(blk[272 + row]);
                pb[b][row] = fmaf((float)o[i], s1, __fmul_rn((float)a[i], s0));
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        float f = 0.f;
        for (int b = 0; b < nb; ++b) {
            f = fmaf(pb[b][threadIdx.x], xsr[b], f);
        }
        f = __fmul_rn(f, __fmul_rn(128.f, *(const float *)wg));     /* 0.5 g, and 2^8 of kq_e4m3s */
        if (dn) {
            DP(float, 9)[(size_t)j * hidden + 16 * grp + threadIdx.x] = f;
        } else {
            DP(float, 7)[(size_t)j * 2 * inner + up * inner + 16 * grp + threadIdx.x] = f;
        }
    }
}

static int kqh_nvx_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_KQH_NVX");
        on = !(v && v[0] == '0');
    }
    return on;
}

__global__ void k_kqh_sum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int c = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    int hidden = DI(13), k = DI(11), tk = DI(16) * k;
    if (c >= hidden) {
        return;
    }
    const float *val = DP(const float, 1), *de = DP(const float, 9);
    float acc = 0.f;
    for (int s2 = 0; s2 < k; ++s2) {
        int j = tok * k + s2;
        if (kqh_on(r, e, j)) {
            acc += val[j] * de[(size_t)j * hidden + c];
        }
    }
    float sw = 1.f / (1.f + expf(-DP(const float, 21)[tok]));
    acc += sw * de[(size_t)(tk + tok) * hidden + c];
    DP(float, 10)[(size_t)tok * hidden + c] = acc;
}

#define KQH_TOPK_MAX 16      /* the most experts of a token (k_moe_combine) */
/* GP_MOE_COMBINE: val, idx, slots, zc, de, rows, out, k, hidden, t, slog,
 * cold. The output of the experts of t tokens in the order of the router,
 * wherever each one ran: slot s of token j (pair p = j k + s) adds val[p]
 * times its row, the de of GP_KQ_HOT_MOE for a hot pair (slots[idx[p]] >=
 * 0, or zc[p]: the copy in host memory), else the row of the CPU in rows
 * (pinned host memory: the KQ_MOE rows form, act bit 64): row p of a
 * group, row n of a step (its n-th cold slot: GP_HOT_SPLIT lists them in
 * their order). A pair of a group with no cold expert there (cold[p] < 0,
 * the padding rows) adds nothing. Then the shared expert (row t k + j of de)
 * times sigmoid(slog[j]), as k_kqh_sum. So a token has the same bits
 * whichever experts the GPU holds (the hot and the cold experts give the
 * same bits: scripts/check_nvx_hot.py). */
__global__ void k_moe_combine(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int c = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    int k = DI(7), hidden = DI(8), t = DI(9);
    if (c >= hidden) {
        return;
    }
    const float *val = DP(const float, 0), *de = DP(const float, 4), *rows = DP(const float, 5);
    const int *idx = DP(const int, 1), *slots = DP(const int, 2);
    const int *zc = r->tag[3] != GP_T_NONE ? DP(const int, 3) : NULL;
    const int *cold = r->tag[11] != GP_T_NONE ? DP(const int, 11) : NULL;
    float v[KQH_TOPK_MAX];
    int n = 0;
    #pragma unroll 4
    for (int s = 0; s < k; ++s) {
        int p = tok * k + s;
        if (slots[idx[p]] >= 0 || (zc != NULL && zc[p])) {
            v[s] = de[(size_t)p * hidden + c];
        } else if (t > 1 && cold != NULL && cold[p] < 0) {
            v[s] = 0.f;
        } else {
            v[s] = __ldcv(rows + (size_t)(t == 1 ? n++ : p) * hidden + c);
        }
    }
    float acc = 0.f;
    for (int s = 0; s < k; ++s) {
        acc += val[tok * k + s] * v[s];
    }
    float sw = 1.f / (1.f + expf(-DP(const float, 10)[tok]));
    acc += sw * de[(size_t)(t * k + tok) * hidden + c];
    DP(float, 6)[(size_t)tok * hidden + c] = acc;
}

/* GP_ADD_RMS: x, o, w, h, rows, cols, eps. x += o, then h = rms_norm(x) w
 * (GP_ADD and GP_RMS_NORM of compile_qwen_step in one kernel). One block
 * for each row. */
__global__ void k_add_rms(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int cols = DI(5);
    float *x = DP(float, 0) + (size_t)blockIdx.x * cols;
    const float *o = DP(const float, 1) + (size_t)blockIdx.x * cols;
    const float *w = DP(const float, 2);
    float *h = DP(float, 3) + (size_t)blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float v = x[i] + o[i];
        x[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss);
    float s = 1.0f / sqrtf(ss / (float)cols + df(r, e, 6));
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        h[i] = x[i] * s * w[i];
    }
}
