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
 * of the steps. A group with a log (the MTP verify) keeps k_gdn_heads.
 *
 * Flag 4 (KDA, GLM-5.3): a decay for each channel of the keys, not each
 * head. Operand 5 is [t, vh KD] (f_b f_a x), dt_bias [vh KD], and the decay
 * exp(-5 sigmoid(exp(A_log[h]) (a + dt_bias))) scales row d of the state of
 * head h (k_gdn_scan<.., CH>: the decays of a step in shared memory too, in
 * steps of 16 tokens). */
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
        if (i >= t) {
            return;
        }
        if (DI(20) & 4) {
            /* KDA: the decay of each channel */
            float *a = DP(float, 5) + (size_t)i * vh * KD;
            for (int ch = c; ch < vh * KD; ch += KD) {
                float g = expf(DP(const float, 7)[ch / KD]) * (a[ch] + DP(const float, 8)[ch]);
                a[ch] = expf(-5.f / (1.f + expf(-g)));
            }
            if (c < vh) {
                float *b = DP(float, 6) + (size_t)i * vh + c;
                *b = 1.f / (1.f + expf(-*b));
            }
            return;
        }
        if (c >= vh) {
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
template <int KD, int GS, bool CH = false, int TC = GDN_TC>
__global__ void __launch_bounds__(128) k_gdn_scan(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    constexpr int NV = KD / GS, CB = 128 / GS;     /* values of a thread, columns of a block */
    __shared__ __align__(16) float qs[TC][KD], ks[TC][KD], dch[CH ? TC : 1][KD];
    __shared__ float dec[TC], bet[TC];
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
    for (int i0 = 0; i0 < t; i0 += TC) {
        int n = min(TC, t - i0);
        __syncthreads();
        for (int z = threadIdx.x; z < n * KD / 4; z += 128) {
            int ii = z / (KD / 4), d4 = (z % (KD / 4)) * 4;
            const float *row = cv + (size_t)(i0 + ii) * cd;
            *(float4 *)&qs[ii][d4] = *(const float4 *)(row + hk * KD + d4);
            *(float4 *)&ks[ii][d4] = *(const float4 *)(row + kd + hk * KD + d4);
            if (CH) {
                *(float4 *)&dch[ii][d4] = *(const float4 *)(da + (size_t)(i0 + ii) * vd + hv * KD + d4);
            }
        }
        if (threadIdx.x < n && !CH) {
            dec[threadIdx.x] = da[(size_t)(i0 + threadIdx.x) * vh + hv];
        }
        if (threadIdx.x < n) {
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
                float4 d4 = CH ? *(const float4 *)&dch[ii][4 * GS * u4 + 4 * part]
                               : make_float4(decay, decay, decay, decay);
                S[4 * u4 + 0] = __fmul_rn(S[4 * u4 + 0], d4.x);
                S[4 * u4 + 1] = __fmul_rn(S[4 * u4 + 1], d4.y);
                S[4 * u4 + 2] = __fmul_rn(S[4 * u4 + 2], d4.z);
                S[4 * u4 + 3] = __fmul_rn(S[4 * u4 + 3], d4.w);
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
                                                         int cd, size_t lrow, int ch)
{
    int hv = blockIdx.x, c = threadIdx.x;
    float *Sh = S + (size_t)hv * KD * KD;
    float st[KD];
    #pragma unroll
    for (int d = 0; d < KD; ++d) {
        st[d] = Sh[(size_t)d * KD + c];
    }
    for (int i = 0; i < n; ++i) {
        /* a head: k, delta, then its decay (or with ch, the decay of each row) */
        const float *lg = log + (size_t)i * lrow + cd + (size_t)hv * (ch ? 3 * KD : 2 * KD + 1);
        float delta = lg[KD + c];
        #pragma unroll
        for (int d = 0; d < KD; ++d) {
            st[d] = __fmul_rn(st[d], lg[2 * KD + (ch ? d : 0)]);
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
    /* the shared expert: none when its gate matrix (operand 17) is null (LFM2) */
    return j >= tk ? (DP(const uint8_t, 17) != NULL ? 0 : -1) : DP(const int, 3)[DP(const int, 2)[j]];
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

/* The act of the MoE kernels (k_kqh_act, k_qmoe_act), bits of a state of
 * the process: 1, RQ8_0 experts (gguf.RQ8_0, Qwen4GPU): the act of each pair
 * rotated before the down product, as kquants.c kq_moe_rot (the program
 * rotates their x); 2, GLM-5.3: the SwiGLU clamped, silu(min(g, 10))
 * clamp(u, -10, 10), as act bit 512 of the CPU (the experts and the shared
 * expert). */
static int gg_moe_rot = 0;

extern "C" void rc_gpu_set_moe_rot(int on)
{
    gg_moe_rot = (gg_moe_rot & ~1) | (on ? 1 : 0);
}

extern "C" void rc_gpu_set_moe_clamp(int on)
{
    gg_moe_rot = (gg_moe_rot & ~2) | (on ? 2 : 0);
}

/* 4: the contract of FP4-Local (docs/FP4LOCAL_PLAN.md, b12x's W4A16): the KQ_NVX hot experts on
 * float32 x (bfloat16 values: BF16_ROUND before the MoE), the act kq_b12x_act, each pair's down
 * row y = bf16(bf16(D2) bf16(w g)) (k_kqh_nvx), and the combine of the rows as they are, bf16(sum),
 * the shared expert bf16(bf16(sigmoid(bf16(gate))) bf16(s)), out = bf16(routed + shared) */
extern "C" void rc_gpu_set_moe_contract(int on)
{
    /* on: bit 0 the A16 contract of steps and verify groups (4), bit 1 the A4 contract of the prompt
     * groups (8: GP_KQ_GROUP_MOE, kernels/moe/b12x.cu) */
    gg_moe_rot = (gg_moe_rot & ~12) | ((on & 1) ? 4 : 0) | ((on & 2) ? 8 : 0);
}

__device__ __forceinline__ float rc_bf16r(float v)
{
    unsigned u = __float_as_uint(v);
    return __uint_as_float((u + 0x7FFFu + ((u >> 16) & 1u)) & 0xFFFF0000u);
}

__device__ __forceinline__ float moe_b12x_act(float g, float u)
{
    g = rc_bf16r(g);
    u = rc_bf16r(u);
    /* the operations of kq_b12x_act of the CPU (rc_dexp, IEEE products): the same bits */
    return rc_bf16r(__fmul_rn(rc_bf16r(__fmul_rn(g, __frcp_rn(__fadd_rn(1.f, rc_dexp(-g))))), u));
}

/* the contract's MoE output from the routed sum acc (of the y rows) and the shared expert's down
 * row s with its gate logit sl (has: there is one) */
__device__ __forceinline__ float moe_b12x_out(float acc, int has, float sl, float s)
{
    float routed = rc_bf16r(acc);
    if (!has) {
        return routed;
    }
    float sg = rc_bf16r(__frcp_rn(1.f + expf(-rc_bf16r(sl))));
    return rc_bf16r(routed + rc_bf16r(sg * rc_bf16r(s)));
}

__device__ __forceinline__ float moe_act(float g, float u, int act)
{
    if (act & 4) {
        return moe_b12x_act(g, u);
    }
    if (act & 2) {
        g = fminf(g, 10.f);
        u = fminf(fmaxf(u, -10.f), 10.f);
    }
    return qw_silu(g) * u;
}

/* rot: the bits of gg_moe_rot: 1, the act rotated in each 32 values (a warp
 * a group; inner and blockDim are multiples of 32); 2, clamped */
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
        float v = moe_act(g[i], g[inner + i], rot);
        if (rot & 1) {
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

/* The routed pairs (j < t top_k) of GP_KQ_HOT_MOE with Q4_K or Q5_K experts
 * of NB blocks a row (gate and up: blockIdx.z; DN: down): a warp does
 * KQH_K_R rows of the expert with x of each lane in registers (the 8 values
 * of each block it reads in kq_row_part, and their sums), loaded once, in
 * place of a warp a row that reads all of x for each row (8 KB of x for the
 * 1.15 KB of a Q4_K row of 2048). STAGE: the rows of the warp first go to
 * shared memory with 16-byte loads (when they are 16-byte aligned). Each
 * row has the operations of kq_row (the same bits). Qwen3.6 on the GTX 1070,
 * 12 pairs a layer: gate or up 125 -> 86 us, down (STAGE) 158 -> 64 us. */
#define KQH_K_R 4
template <int FIVE, int NB, int DN, int STAGE>
__global__ void __launch_bounds__(32 * ROWS_PER_BLOCK) k_kqh_k(const gp_rec *r, const int64_t *e)
{
    constexpr int BS = FIVE ? 176 : 144, RB = NB * BS, NV = KQH_K_R * RB / 16;
    constexpr int PER = STAGE ? (NV + 31) / 32 : 1;
    __shared__ __align__(16) uint8_t sm[STAGE ? ROWS_PER_BLOCK : 1][STAGE ? KQH_K_R * RB : 16];
    PDL_START();
    int j = blockIdx.y, up = DN ? 2 : blockIdx.z, slot = kqh_slot(r, e, j);
    int inner = DI(12), hidden = DI(13), k = DI(11);
    int rows = DN ? hidden : inner;
    int row0 = (blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32) * KQH_K_R;
    if (slot < 0 || row0 >= rows) {
        return;
    }
    int l = threadIdx.x % 32, c = l / 8, sub = l % 8;
    const uint8_t *wr = kqh_wslot(r, e, slot, up, (size_t)rows * RB) + (size_t)row0 * RB;
    int nr = min(KQH_K_R, rows - row0);
    const int st = STAGE && ((uintptr_t)wr & 15) == 0;     /* (else the direct loads) */
    if (st) {
        uint8_t *buf = sm[threadIdx.x / 32];
        const uint4 *src = (const uint4 *)wr;
        int nv = nr * RB / 16;
        #pragma unroll
        for (int p = 0; p < PER; ++p) {
            int i = l + 32 * p;
            if (i < nv) {
                ((uint4 *)buf)[i] = __ldg(src + i);
            }
        }
        wr = buf;
    }
    const float *xt = DN ? DP(const float, 8) + (size_t)j * inner : DP(const float, 0) + (size_t)(j / k) * hidden;
    float4 xl[NB], xh[NB];
    float sl[NB], sh[NB];
    #pragma unroll
    for (int b = 0; b < NB; ++b) {
        const float *xb = xt + b * 256 + 64 * c + 4 * sub;
        xl[b] = *(const float4 *)xb;
        xh[b] = *(const float4 *)(xb + 32);
        sl[b] = xl[b].x + xl[b].y + xl[b].z + xl[b].w;
        sh[b] = xh[b].x + xh[b].y + xh[b].z + xh[b].w;
    }
    if (st) {
        __syncwarp();
    }
    float *out = DN ? DP(float, 9) + (size_t)j * hidden : DP(float, 7) + (size_t)j * 2 * inner + up * inner;
    #pragma unroll
    for (int rr = 0; rr < KQH_K_R; ++rr) {
        if (rr >= nr) {
            break;
        }
        const uint8_t *w = wr + (size_t)rr * RB;
        float sum = 0.f;
        #pragma unroll
        for (int b = 0; b < NB; ++b) {
            const uint8_t *blk = w + b * BS;
            const uint8_t *qs = blk + (FIVE ? 48 : 16);
            uint32_t q = *(const uint32_t *)(qs + 32 * c + 4 * sub);
            uint32_t lo = q & 0x0f0f0f0fu, hi = (q >> 4) & 0x0f0f0f0fu;
            if (FIVE) {
                uint32_t h = *(const uint32_t *)(blk + 16 + 4 * sub);
                lo |= ((h >> (2 * c)) & 0x01010101u) << 4;
                hi |= ((h >> (2 * c + 1)) & 0x01010101u) << 4;
            }
            int s0, m0, s1, m1;
            kq_scale_min(blk + 4, 2 * c, &s0, &m0);
            kq_scale_min(blk + 4, 2 * c + 1, &s1, &m1);
            float d = kq_half(blk), dm = kq_half(blk + 2);
            sum += d * ((float)s0 * kq_dot4(lo, xl[b]) + (float)s1 * kq_dot4(hi, xh[b])) -
                   dm * ((float)m0 * sl[b] + (float)m1 * sh[b]);
        }
        for (int o = 16; o > 0; o >>= 1) {
            sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
        }
        if (l == 0) {
            out[row0 + rr] = sum;
        }
    }
}

/* RECLANK_GPU_KQH_K (1): k_kqh_k for the routed pairs of Q4_K and Q5_K experts
 * whose rows are 2, 4, 8, or 16 blocks; 0 keeps k_kqh_gu and k_kqh_dn (a test). */
static int kqh_k_ok(int type, int cols)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_KQH_K");
        on = v == NULL || atoi(v) != 0;
    }
    int nb = cols / 256;
    return on && (type == KQ_Q4_K || type == KQ_Q5_K) && cols % 256 == 0 &&
           (nb == 2 || nb == 4 || nb == 8 || nb == 16);
}

/* k_kqh_k for the tk routed pairs: gate and up (dn 0, rows inner, cols
 * hidden) or down (dn 1, rows hidden, cols inner; staged: its rows are short) */
static void kqh_k_launch(const gp_rec *dr, const int64_t *denv, int type, int cols, int dn, unsigned tk,
                         int64_t rows, cudaStream_t s)
{
    dim3 grid((unsigned)((rows + ROWS_PER_BLOCK * KQH_K_R - 1) / (ROWS_PER_BLOCK * KQH_K_R)), tk, dn ? 1 : 2);
    const int five = type == KQ_Q5_K, nb = cols / 256, W = 32 * ROWS_PER_BLOCK;
#define KQH_K(F, N) \
    if (five == F && nb == N) { \
        if (dn) { \
            k_kqh_k<F, N, 1, (N <= 4)><<<grid, W, 0, s>>>(dr, denv); \
        } else { \
            k_kqh_k<F, N, 0, 0><<<grid, W, 0, s>>>(dr, denv); \
        } \
        return; \
    }
    KQH_K(0, 2) KQH_K(0, 4) KQH_K(0, 8) KQH_K(0, 16)
    KQH_K(1, 2) KQH_K(1, 4) KQH_K(1, 8) KQH_K(1, 16)
#undef KQH_K
}

/* The KQ_NVX experts of GP_KQ_HOT_MOE (the pairs j < t top_k): a block of KS
 * warps for each group of 16 rows (blockIdx.x), pair (blockIdx.y), and
 * matrix (blockIdx.z: gate or up; down with dn). Warp w takes the blocks
 * of 32 columns w, w + KS, ... of the group (kq_nvx_group for a part of
 * the columns), and the warps add their sums in shared memory. With a warp
 * for each whole group (k_kqh_gu) a step had about 250 KB of loads in
 * flight: 166 GB/s, 72 us for the gate and up of a layer. */
template <int KS>
__global__ void __launch_bounds__(32 * KS) k_kqh_nvx(const gp_rec *r, const int64_t *e, int dn, int ct = 0)
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
        if (ct && dn) {
            /* the contract (gg_moe_rot 4): y = bf16(bf16(D2) bf16(w g)) of the pair */
            float d2 = sm * 128.f, wg_ = rc_bf16r(DP(const float, 1)[j] * *(const float *)wg);
            sm = rc_bf16r(rc_bf16r(d2) * wg_);
        } else {
            sm *= 128.f * *(const float *)wg;        /* 0.5 g, and the 2^8 of kq_e4m3s */
        }
        if (dn) {
            DP(float, 9)[(size_t)j * hidden + 16 * grp + threadIdx.x] = sm;
        } else {
            DP(float, 7)[(size_t)j * 2 * inner + up * inner + 16 * grp + threadIdx.x] = sm;
        }
    }
}

/* The KQ_NVX experts of GP_KQ_HOT_MOE with FP4-Local's A16 contract (gg_moe_rot 4), with the operations
 * of the CPU's kq_nvx_rows_f in their order, so a hot and a cold expert give the same bits: for each
 * row and half block of 16 columns four float32 sums a_u (fma over the 4 steps, w = the E2M1 value),
 * t = (a0 + a1) + (a2 + a3) (the threads, into shared memory), then a thread for each row f = fma(t,
 * s 2^-8, f) over the halves in order, f 256 g (gate, up) or the pair's y = bf16(bf16(f 256) bf16(w g))
 * (down). x: the float rows (bfloat16 values). */
#define KQH_CT_NH 320       /* the most halves of a row (5120 columns) */
#define KQH_CT_T 256        /* (the threads of a block: the loads of more tasks in flight) */
__global__ void __launch_bounds__(KQH_CT_T) k_kqh_nvx_ct(const gp_rec *r, const int64_t *e, int dn)
{
    PDL_START();
    __shared__ float th[16][KQH_CT_NH + 1];
    __shared__ uint8_t scb[16][KQH_CT_NH + 4];     /* (the scale byte of each row and half: the serial sums
                                                     * below read shared memory, not a global load a step) */
    int grp = blockIdx.x, j = blockIdx.y, up = blockIdx.z;
    int slot = kqh_slot(r, e, j), zc = slot < 0 && kqh_zc(r, e, j);
    if (slot < 0 && !zc) {
        return;
    }
    int inner = DI(12), hidden = DI(13), k = DI(11);
    int cols = dn ? inner : hidden, nh = cols / 16;
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
    const float *x = dn ? DP(const float, 8) + (size_t)j * inner : DP(const float, 0) + (size_t)(j / k) * hidden;
    /* the E2M1 values of the 16 codes (sign bit 8), in shared memory (an array of the thread indexed
     * by the code lands in local memory) */
    __shared__ float e2s[16];
    if (threadIdx.x < 16) {
        const int c = threadIdx.x & 7, m = c & 1, ex = c >> 1;
        const float v = ex == 0 ? 0.5f * (float)m : ldexpf(1.f + 0.5f * (float)m, ex - 1);
        e2s[threadIdx.x] = threadIdx.x & 8 ? -v : v;
    }
    __syncthreads();
#pragma unroll 2
    for (int q = threadIdx.x; q < 16 * nh; q += blockDim.x) {
        int row = q % 16, h = q / 16, b = h / 2;
        const uint8_t *blk = wg + 16 + (size_t)b * KQ_NVX_BB;
        int sh = row < 8 ? 0 : 4, r8 = row % 8, k0 = 4 * (h % 2);
        float a[4] = {0.f, 0.f, 0.f, 0.f};
        /* (the 4 codes of a step in one word, the 4 values of x in one float4: the loads of the
         * step; the products and sums in the order of the CPU's kq_nvx_rows_f) */
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int kk = k0 + i;
            const uint32_t cw = *(const uint32_t *)(blk + 32 * kk + 4 * r8);
            const float4 xv = *(const float4 *)(x + 32 * b + 4 * kk);
            const float xs4[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int u = 0; u < 4; ++u) {
                a[u] = fmaf(e2s[(cw >> (8 * u + sh)) & 15], xs4[u], a[u]);
            }
        }
        th[row][h] = __fadd_rn(__fadd_rn(a[0], a[1]), __fadd_rn(a[2], a[3]));
        scb[row][h] = blk[256 + 16 * (h % 2) + row];
    }
    __syncthreads();
    if (threadIdx.x < 16) {
        int row = threadIdx.x;
        float f = 0.f;
#pragma unroll 8
        for (int h = 0; h < nh; ++h) {
            f = fmaf(th[row][h], kq_e4m3s(scb[row][h]), f);
        }
        float g = *(const float *)wg;
        if (dn) {
            float wq = rc_bf16r(__fmul_rn(DP(const float, 1)[j], g));
            DP(float, 9)[(size_t)j * hidden + 16 * grp + row] = rc_bf16r(__fmul_rn(rc_bf16r(__fmul_rn(f, 256.f)), wq));
        } else {
            DP(float, 7)[(size_t)j * 2 * inner + up * inner + 16 * grp + row] = __fmul_rn(f, __fmul_rn(256.f, g));
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
            a[i] = rc_dp4a(wv[i], xv, 0);
        }
        if (X2) {
            int xlv = *(const int *)(xl + x0 + 32 * b + 4 * st);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = 128 * a[i] + rc_dp4a(wv[i], xlv, 0);
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

__global__ void k_kqh_sum(const gp_rec *r, const int64_t *e, int ct = 0)
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
            acc += ct ? de[(size_t)j * hidden + c] : val[j] * de[(size_t)j * hidden + c];
        }
    }
    if (ct) {
        int has = DP(const uint8_t, 17) != NULL;
        DP(float, 10)[(size_t)tok * hidden + c] = moe_b12x_out(
            acc, has, has ? DP(const float, 21)[tok] : 0.f, has ? de[(size_t)(tk + tok) * hidden + c] : 0.f);
        return;
    }
    if (DP(const uint8_t, 17) != NULL) {
        float sw = 1.f / (1.f + expf(-DP(const float, 21)[tok]));
        acc += sw * de[(size_t)(tk + tok) * hidden + c];
    }
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
__global__ void k_moe_combine(const gp_rec *r, const int64_t *e, int ct)
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
    if (ct) {
        /* the contract: the rows are the y of each pair (weighted, rounded) */
        for (int s = 0; s < k; ++s) {
            acc += v[s];
        }
        int has = r->tag[10] != GP_T_NONE && DP(const float, 10) != NULL;
        DP(float, 6)[(size_t)tok * hidden + c] = moe_b12x_out(
            acc, has, has ? DP(const float, 10)[tok] : 0.f, has ? de[(size_t)(t * k + tok) * hidden + c] : 0.f);
        return;
    }
    for (int s = 0; s < k; ++s) {
        acc += val[tok * k + s] * v[s];
    }
    if (r->tag[10] != GP_T_NONE && DP(const float, 10) != NULL) {     /* (no shared expert: LFM2) */
        float sw = 1.f / (1.f + expf(-DP(const float, 10)[tok]));
        acc += sw * de[(size_t)(t * k + tok) * hidden + c];
    }
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
