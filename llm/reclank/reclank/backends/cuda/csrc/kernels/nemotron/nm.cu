/* ---------- Nemotron 3.5 Bolt on the GPU (docs/NEMOTRON_BOLT_PLAN.md, P4) ----------
 * The records of compile_nemotron_step (reclank/models/nemotron/kq.py) as the CPU runs them
 * (backends/cpu/csrc/common/nemotron.c has the math). Float32 math everywhere (Pascal's fp16 is
 * slow), no tensor cores, no cp.async: every kernel runs on sm_61.
 *
 * GP_NM_MAMBA: k_nm_conv (a thread for each channel: the causal conv over the rows of the state and
 * the group, silu, then the state's new rows), k_nm_scan (a warp for each value of each head: the
 * 128 values of its state row, 4 a lane; the tokens in order), k_nm_gnorm (a block for each token
 * and group: the gated RMSNorm).
 * GP_RELU2: relu(x)^2.
 * GP_NM_ATTN: q (t x nq x hd, scaled), K, V (the float rows of the cache, position-major: [n, nk,
 * hd]), out, nq, nk, hd, t, pos: a warp for each token and query head, the positions up to its
 * own (no RoPE: the rows as they are), an online softmax.
 * GP_NM_SPLIT: idx, slots (the GPU slot of each expert, or -1), cold (the pairs the CPU computes:
 * the expert, or -1), count (the cold pairs), t * k, k, nreal.
 * GP_NM_HOT_MOE: x (t x L), idx, val, slots, up, down (the hot experts' blocks by slot), act
 * (t k x inner), de (t k x L), out (t x L), k, L, inner, t: the hot pairs' up rows, relu^2, the
 * down rows times the weight (zero rows for the cold pairs), the sum of each token's pairs. */

/* The x of the int8 products (NM_QX) from the kernel that makes it: the 32 values of a warp (lane l:
 * value l of a 32-group, at index at of the rows) to q, its scale s[at / 32], ql (the two planes; NULL:
 * int8 x in q); the bits of nm_qx32 / nm_qx32_2. GP_NM_ADDNORM (operands 8-10), GP_NM_MAMBA (20-22),
 * GP_RELU2 (3-5) and GP_NM_HOT_MOE (19-21) take q, s, ql: their output also as NM_QX's. */
__device__ __forceinline__ void nm_qx_warp(float v, size_t at, int8_t *q, float *s, int8_t *ql)
{
    float m = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, o));
    }
    if (ql != NULL) {
        const float iv = m > 0.f ? NM_Q16 / m : 0.f;
        const int qv = __float2int_rn(v * iv), hi = (qv + 128) >> 8;
        q[at] = (int8_t)hi;
        ql[at] = (int8_t)(qv - 256 * hi);
        if (at % 32 == 0) {
            s[at / 32] = m / NM_Q16;
        }
    } else {
        const float iv = m > 0.f ? 127.f / m : 0.f;
        q[at] = (int8_t)__float2int_rn(v * iv);
        if (at % 32 == 0) {
            s[at / 32] = m / 127.f;
        }
    }
}

/* operand i of a record as an int8 / float pointer, NULL when none */
#define NM_OPQ(i) (r->tag[i] != GP_T_NONE && DP(int8_t, i) != NULL ? DP(int8_t, i) : (int8_t *)NULL)

/* GP_NM_MAMBA: zxd, conv, cw, cb, A_log, D, dt_bias, nw, ssm, y, scr, t, heads, hdim, groups,
 * nstate, kernel, eps, dt_min, nreal (, q, s, ql, sd). sd (more than one token): the new conv rows
 * and state sd bytes from the ones read (both), which stay as they were (the verify group of spec
 * decoding and its replay: the states in two sets, no copies); none or 0: in place. */
__device__ __forceinline__ int64_t nm_sd(const gp_rec *r, const int64_t *e)
{
    return r->tag[23] == GP_T_NONE ? 0 : di(r, e, 23) / 4;
}

__global__ void k_nm_conv(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int t = DI(11), heads = DI(12), hdim = DI(13), groups = DI(14), nstate = DI(15), kern = DI(16);
    const int inner = heads * hdim, conv = inner + 2 * groups * nstate;
    int nreal = r->tag[19] == GP_T_NONE ? t : DI(19);
    if (nreal <= 0 || nreal > t) {
        nreal = t;
    }
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv) {
        return;
    }
    const size_t rs = (size_t)inner + conv + heads;
    const float *zxd = DP(const float, 0), *w = DP(const float, 2) + (size_t)c * kern;
    float *state = DP(float, 1), *scr = DP(float, 10);
    const float bias = DP(const float, 3)[c];
    float st[8];
    for (int k = 0; k < kern - 1; ++k) {
        st[k] = state[(size_t)k * conv + c];
    }
    for (int j = 0; j < t; ++j) {
        float s = bias;
        for (int k = 0; k < kern; ++k) {
            int q = j + k - (kern - 1);
            s += (q < 0 ? st[q + kern - 1] : zxd[(size_t)q * rs + inner + c]) * w[k];
        }
        scr[(size_t)j * conv + c] = s / (1.f + expf(-s));
    }
    float *out = state + nm_sd(r, e);
    for (int k = 0; k < kern - 1; ++k) {
        int q = nreal + k - (kern - 1);
        float v = q < 0 ? st[q + kern - 1] : zxd[(size_t)q * rs + inner + c];
        out[(size_t)k * conv + c] = v;
    }
}

__global__ void k_nm_scan(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int t = DI(11), heads = DI(12), hdim = DI(13), groups = DI(14), nstate = DI(15);
    const int inner = heads * hdim, conv = inner + 2 * groups * nstate;
    int nreal = r->tag[19] == GP_T_NONE ? t : DI(19);
    if (nreal <= 0 || nreal > t) {
        nreal = t;
    }
    const int wid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (wid >= heads * hdim) {
        return;
    }
    const int h = wid / hdim, d = wid % hdim, g = h / (heads / groups);
    const size_t rs = (size_t)inner + conv + heads;
    const float *zxd = DP(const float, 0), *scr = DP(const float, 10);
    float *S = DP(float, 8) + ((size_t)h * hdim + d) * nstate;
    float *y = DP(float, 9);
    const float A = -expf(DP(const float, 4)[h]), Dh = DP(const float, 5)[h], dtb = DP(const float, 6)[h];
    const float dt_min = df(r, e, 18);
    float4 s4 = ((const float4 *)S)[lane];
    for (int j = 0; j < t; ++j) {
        if (j >= nreal) {
            if (lane == 0) {
                y[(size_t)j * inner + (size_t)h * hdim + d] = 0.f;
            }
            continue;
        }
        const float *xr = scr + (size_t)j * conv;
        float raw = zxd[(size_t)j * rs + inner + conv + h] + dtb;
        float dt = fmaxf(raw > 20.f ? raw : log1pf(expf(raw)), dt_min);
        float dA = expf(dt * A), xv = xr[(size_t)h * hdim + d], dx = dt * xv;
        float4 B = ((const float4 *)(xr + inner + (size_t)g * nstate))[lane];
        float4 C = ((const float4 *)(xr + inner + (size_t)groups * nstate + (size_t)g * nstate))[lane];
        s4.x = s4.x * dA + dx * B.x;
        s4.y = s4.y * dA + dx * B.y;
        s4.z = s4.z * dA + dx * B.z;
        s4.w = s4.w * dA + dx * B.w;
        float acc = s4.x * C.x + s4.y * C.y + s4.z * C.z + s4.w * C.w;
        for (int o = 16; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(RC_FULL_MASK, acc, o);
        }
        if (lane == 0) {
            y[(size_t)j * inner + (size_t)h * hdim + d] = acc + Dh * xv;
        }
    }
    ((float4 *)(S + nm_sd(r, e)))[lane] = s4;
}

/* GP_NM_MAMBA of one token: the conv and the scan in one kernel, a block for each head (its 64 x
 * channels and its group's B and C through the conv, silu, in shared memory; then a warp for each 8
 * values of hdim, as k_nm_scan). The conv's state rows are left: k_nm_gnorm<true> writes them after
 * (the blocks of a group read the old rows of B and C). The same sums as k_nm_conv and k_nm_scan. */
__global__ void __launch_bounds__(256) k_nm_mamba1(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int heads = DI(12), hdim = DI(13), groups = DI(14), nstate = DI(15), kern = DI(16);
    const int inner = heads * hdim, conv = inner + 2 * groups * nstate;
    const int h = blockIdx.x, g = h / (heads / groups), lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    __shared__ __align__(16) float xc[64], Bc[128], Cc[128];
    const float *zxd = DP(const float, 0), *state = DP(const float, 1), *cw = DP(const float, 2), *cb = DP(const float, 3);
    for (int i = threadIdx.x; i < hdim + 2 * nstate; i += blockDim.x) {
        const int c = i < hdim ? h * hdim + i : i < hdim + nstate ? inner + g * nstate + (i - hdim)
                                                                  : inner + groups * nstate + g * nstate + (i - hdim - nstate);
        float sv = cb[c];
        for (int k = 0; k < kern; ++k) {
            sv += (k < kern - 1 ? state[(size_t)k * conv + c] : zxd[inner + c]) * cw[(size_t)c * kern + k];
        }
        sv = sv / (1.f + expf(-sv));
        if (i < hdim) {
            xc[i] = sv;
        } else if (i < hdim + nstate) {
            Bc[i - hdim] = sv;
        } else {
            Cc[i - hdim - nstate] = sv;
        }
    }
    __syncthreads();
    const float A = -expf(DP(const float, 4)[h]), Dh = DP(const float, 5)[h], dtb = DP(const float, 6)[h];
    const float raw = zxd[inner + conv + h] + dtb;
    const float dt = fmaxf(raw > 20.f ? raw : log1pf(expf(raw)), df(r, e, 18)), dA = expf(dt * A);
    const float4 B = ((const float4 *)Bc)[lane], C = ((const float4 *)Cc)[lane];
    float *y = DP(float, 9);
    const int per = hdim / (blockDim.x / 32);
    for (int d = wp * per; d < (wp + 1) * per; ++d) {
        float *S = DP(float, 8) + ((size_t)h * hdim + d) * nstate;
        float4 s4 = ((const float4 *)S)[lane];
        const float xv = xc[d], dx = dt * xv;
        s4.x = s4.x * dA + dx * B.x;
        s4.y = s4.y * dA + dx * B.y;
        s4.z = s4.z * dA + dx * B.z;
        s4.w = s4.w * dA + dx * B.w;
        float acc = s4.x * C.x + s4.y * C.y + s4.z * C.z + s4.w * C.w;
        for (int o = 16; o > 0; o >>= 1) {
            acc += __shfl_xor_sync(RC_FULL_MASK, acc, o);
        }
        ((float4 *)S)[lane] = s4;
        if (lane == 0) {
            y[(size_t)h * hdim + d] = acc + Dh * xv;
        }
    }
}

/* ST: one token, after k_nm_mamba1: the blocks also move the conv's state rows (the old rows less the
 * first, then the token's xBC) */
template <bool ST>
__global__ void k_nm_gnorm(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    if (ST) {
        const int heads = DI(12), hdim = DI(13), groups = DI(14), nstate = DI(15), kern = DI(16);
        const int inner = heads * hdim, conv = inner + 2 * groups * nstate;
        float *state = DP(float, 1);
        const float *zxd = DP(const float, 0);
        for (int c = blockIdx.x * blockDim.x + threadIdx.x; c < conv; c += gridDim.x * blockDim.x) {
            for (int k = 0; k < kern - 2; ++k) {
                state[(size_t)k * conv + c] = state[(size_t)(k + 1) * conv + c];
            }
            state[(size_t)(kern - 2) * conv + c] = zxd[inner + c];
        }
    }
    const int heads = DI(12), hdim = DI(13), groups = DI(14), nstate = DI(15);
    const int inner = heads * hdim, conv = inner + 2 * groups * nstate, gs = inner / groups;
    const int j = blockIdx.x / groups, g = blockIdx.x % groups;
    const size_t rs = (size_t)inner + conv + heads;
    float *y = DP(float, 9) + (size_t)j * inner + (size_t)g * gs;
    const float *z = DP(const float, 0) + (size_t)j * rs + (size_t)g * gs;
    const float *nw = DP(const float, 7) + (size_t)g * gs;
    float ss = 0.f;
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {
        float v = y[i] * (z[i] / (1.f + expf(-z[i])));
        y[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss);
    const float inv = 1.f / sqrtf(ss / (float)gs + df(r, e, 17));
    int8_t *q = NM_OPQ(20);
    for (int i = threadIdx.x; i < gs; i += blockDim.x) {
        const float v = y[i] * inv * nw[i];
        y[i] = v;
        if (q != NULL) {
            /* (gs and blockDim multiples of 32: a warp's 32 values one group) */
            nm_qx_warp(v, (size_t)j * inner + (size_t)g * gs + i, q, DP(float, 21), NM_OPQ(22));
        }
    }
}

/* GP_RELU2: x, out, n (, q, s, ql: n a multiple of 32) */
__global__ void k_nm_relu2(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < di(r, e, 2)) {
        float v = fmaxf(DP(const float, 0)[i], 0.f);
        DP(float, 1)[i] = v * v;
        if (NM_OPQ(3) != NULL) {
            nm_qx_warp(v * v, (size_t)i, NM_OPQ(3), DP(float, 4), NM_OPQ(5));
        }
    }
}

/* GP_NM_ATTN: q, K, V, out, nq, nk, hd, t, pos (hd a multiple of 32, at most 256) */
__global__ void k_nm_attn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int nq = DI(4), nk = DI(5), hd = DI(6), t = DI(7);
    const int64_t pos = di(r, e, 8);
    const int wid = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (wid >= t * nq) {
        return;
    }
    const int j = wid / nq, h = wid % nq, kh = h / (nq / nk), per = hd / 32;
    const float *q = DP(const float, 0) + ((size_t)j * nq + h) * hd;
    const float *K = DP(const float, 1), *V = DP(const float, 2);
    float qv[8], acc[8];
    for (int u = 0; u < per; ++u) {
        qv[u] = q[lane + 32 * u];
        acc[u] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    const int64_t n = pos + j + 1;
    for (int64_t p = 0; p < n; ++p) {
        const float *kr = K + ((size_t)p * nk + kh) * hd, *vr = V + ((size_t)p * nk + kh) * hd;
        float s = 0.f;
        for (int u = 0; u < per; ++u) {
            s += qv[u] * kr[lane + 32 * u];
        }
        for (int o = 16; o > 0; o >>= 1) {
            s += __shfl_xor_sync(RC_FULL_MASK, s, o);
        }
        float m2 = fmaxf(m, s), c = expf(m - m2), pw = expf(s - m2);
        l = l * c + pw;
        for (int u = 0; u < per; ++u) {
            acc[u] = acc[u] * c + pw * vr[lane + 32 * u];
        }
        m = m2;
    }
    float *out = DP(float, 3) + ((size_t)j * nq + h) * hd;
    for (int u = 0; u < per; ++u) {
        out[lane + 32 * u] = acc[u] / l;
    }
}

/* GP_NM_SPLIT: idx, slots, cold, count, tk, k, nreal */
__global__ void k_nm_split(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int *idx = DP(const int, 0), *slots = DP(const int, 1);
    int *cold = DP(int, 2), *count = DP(int, 3);
    const int tk = DI(4), k = DI(5), nreal = DI(6);
    __shared__ int n;
    if (threadIdx.x == 0) {
        n = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < tk; p += blockDim.x) {
        int x = idx[p], c = (p / k < nreal && slots[x] < 0) ? x : -1;
        cold[p] = c;
        if (c >= 0) {
            atomicAdd(&n, 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        *count = n;
    }
}

/* GP_NM_HOT_MOE: x, idx, val, slots, up, down, act, de, out, k, L, inner, t; a warp for each row
 * of each pair (grid: rows / 4, pairs) */
__global__ void k_nm_hot_up(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int k = DI(9), L = DI(10), inner = DI(11), pr = blockIdx.y;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int slot = DP(const int, 3)[DP(const int, 1)[pr]];
    if (row >= inner || slot < 0) {
        return;
    }
    const size_t rb = nm_q2_rb(L);
    const uint8_t *w = DP(const uint8_t, 4) + ((size_t)slot * inner + row) * rb;
    float v = nm_q2_row_f(w, DP(const float, 0) + (size_t)(pr / k) * L, L);
    if (threadIdx.x % 32 == 0) {
        v = fmaxf(v, 0.f);
        DP(float, 6)[(size_t)pr * inner + row] = v * v;
    }
}

__global__ void k_nm_hot_down(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int L = DI(10), inner = DI(11), pr = blockIdx.y;
    const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (row >= L) {
        return;
    }
    const int slot = DP(const int, 3)[DP(const int, 1)[pr]];
    float v = 0.f;
    if (slot >= 0) {
        const size_t rb = nm_q2_rb(inner);
        const uint8_t *w = DP(const uint8_t, 5) + ((size_t)slot * L + row) * rb;
        v = nm_q2_row_f(w, DP(const float, 6) + (size_t)pr * inner, inner) * DP(const float, 2)[pr];
    }
    if (threadIdx.x % 32 == 0) {
        DP(float, 7)[(size_t)pr * L + row] = v;
    }
}

__global__ void k_nm_hot_sum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int k = DI(9), L = DI(10), j = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= L) {
        return;
    }
    const float *de = DP(const float, 7) + (size_t)j * k * L;
    float s = 0.f;
    for (int q = 0; q < k; ++q) {
        s += de[(size_t)q * L + c];
    }
    DP(float, 8)[(size_t)j * L + c] = s;
}

/* GP_NM_QX: x, rows, cols, q, s: the float rows as int8 with a scale of each 32 (max |x| / 127,
 * rounded to the nearest even); a thread for each 32. */
__device__ __forceinline__ void nm_qx32(const float *x, int8_t *q, float *s)
{
    float v[32], m = 0.f;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        float4 a = ((const float4 *)x)[u];
        v[4 * u] = a.x; v[4 * u + 1] = a.y; v[4 * u + 2] = a.z; v[4 * u + 3] = a.w;
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        m = fmaxf(m, fabsf(v[i]));
    }
    const float iv = m > 0.f ? 127.f / m : 0.f;
    *s = m / 127.f;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        int a = __float2int_rn(v[4 * u] * iv), b = __float2int_rn(v[4 * u + 1] * iv);
        int c = __float2int_rn(v[4 * u + 2] * iv), d = __float2int_rn(v[4 * u + 3] * iv);
        ((uint32_t *)q)[u] = (uint32_t)(a & 255) | ((uint32_t)(b & 255) << 8) | ((uint32_t)(c & 255) << 16) |
                             ((uint32_t)(d & 255) << 24);
    }
}

/* The two planes of 32 values (iq2.cuh): hi to qh, lo to ql, the scale to *s. */
__device__ __forceinline__ void nm_qx32_2(const float *x, int8_t *qh, int8_t *ql, float *s)
{
    float v[32], m = 0.f;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        float4 a = ((const float4 *)x)[u];
        v[4 * u] = a.x; v[4 * u + 1] = a.y; v[4 * u + 2] = a.z; v[4 * u + 3] = a.w;
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        m = fmaxf(m, fabsf(v[i]));
    }
    const float iv = m > 0.f ? NM_Q16 / m : 0.f;
    *s = m / NM_Q16;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        uint32_t h = 0, l = 0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int q = __float2int_rn(v[4 * u + k] * iv), hi = (q + 128) >> 8, lo = q - 256 * hi;
            h |= (uint32_t)(hi & 255) << (8 * k);
            l |= (uint32_t)(lo & 255) << (8 * k);
        }
        ((uint32_t *)qh)[u] = h;
        ((uint32_t *)ql)[u] = l;
    }
}

/* GP_NM_ADDNORM: h, a, b (or none), w, out, rows, cols, eps: h += a + b, out = h s w with s = 1 /
 * sqrt(mean of h^2 + eps) (k_rms_norm's): the two ADD records and the RMS_NORM of a block boundary in
 * one pass, a block of 1024 a row (k_rms_norm's 256 took 10 us a row of 4096) */
__global__ void __launch_bounds__(1024) k_nm_addnorm(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int cols = DI(6);
    float *h = DP(float, 0) + (size_t)blockIdx.x * cols;
    const float *a = DP(const float, 1) + (size_t)blockIdx.x * cols;
    const float *b = r->tag[2] != GP_T_NONE && DP(const float, 2) != NULL ? DP(const float, 2) + (size_t)blockIdx.x * cols : NULL;
    const float *w = DP(const float, 3);
    float *out = DP(float, 4) + (size_t)blockIdx.x * cols;
    float ss = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = h[i] + (b != NULL ? a[i] + b[i] : a[i]);
        h[i] = v;
        ss += v * v;
    }
    ss = block_sum(ss);
    const float s = 1.0f / sqrtf(ss / (float)cols + df(r, e, 7));
    int8_t *q = NM_OPQ(8);
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        const float v = h[i] * s * w[i];
        out[i] = v;
        if (q != NULL) {
            /* (cols a multiple of 32) */
            nm_qx_warp(v, (size_t)blockIdx.x * cols + i, q, DP(float, 9), NM_OPQ(10));
        }
    }
}

/* GP_NM_QX: x, rows, cols, q, s, ql (or none: int8 x in q; with ql: the two planes, hi in q) */
__global__ void k_nm_qx(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int64_t n = (int64_t)DI(1) * DI(2) / 32;
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n) {
        return;
    }
    if (r->tag[5] != GP_T_NONE && DP(int8_t, 5) != NULL) {
        nm_qx32_2(DP(const float, 0) + 32 * b, DP(int8_t, 3) + 32 * b, DP(int8_t, 5) + 32 * b, DP(float, 4) + b);
    } else {
        nm_qx32(DP(const float, 0) + 32 * b, DP(int8_t, 3) + 32 * b, DP(float, 4) + b);
    }
}

/* GP_NM_IQ2_LIN: q, s (of NM_QX), w (IQ2_XXS rows), rows, cols, t, out: a warp for each NM_LR rows
 * (nm_iq2_rows_x2), the tokens in turn. */
#define NM_LR 2
__global__ void k_nm_iq2_lin(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    NM_IQ2_SMEM_TABLES();
    const int rows = DI(3), cols = DI(4), t = DI(5), lane = threadIdx.x % 32;
    const int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * NM_LR;
    if (row >= rows) {
        return;
    }
    const size_t rb = nm_q2_rb(cols);
    const uint8_t *w = DP(const uint8_t, 2) + (size_t)row * rb;
    for (int j = 0; j < t; ++j) {
        const int8_t *q = DP(const int8_t, 0) + (size_t)j * cols;
        const float *s = DP(const float, 1) + (size_t)j * (cols / 32);
        const int8_t *ql = r->tag[7] != GP_T_NONE && DP(const int8_t, 7) != NULL ?
                           DP(const int8_t, 7) + (size_t)j * cols : NULL;
        float acc[NM_LR] = {};
        if (row + NM_LR <= rows) {
            nm_iq2_rows_x2<NM_LR>(w, rb, cols / 256, q, ql, s, lane, acc, nm_sg, nm_ss);
        } else {
            for (int rr = 0; rr < rows - row; ++rr) {
                nm_iq2_rows_x2<1>(w + (size_t)rr * rb, rb, cols / 256, q, ql, s, lane, acc + rr, nm_sg, nm_ss);
            }
        }
#pragma unroll
        for (int rr = 0; rr < NM_LR; ++rr) {
            float v = acc[rr];
            for (int o = 16; o > 0; o >>= 1) {
                v += __shfl_xor_sync(RC_FULL_MASK, v, o);
            }
            if (lane == 0 && row + rr < rows) {
                DP(float, 6)[(size_t)j * rows + row + rr] = v;
            }
        }
    }
}

/* the form of GP_NM_IQ2_LIN (RECLANK_NM_Q2: R U B digits; 0: k_nm_iq2_lin) */
static int nm_q2_form(void)
{
    static int f = -1;
    if (f < 0) {
        const char *v = getenv("RECLANK_NM_Q2");
        f = v ? atoi(v) : 410;
    }
    return f;
}

/* GP_NM_IQ2_LIN for a few tokens, a lane a part (nm_q2_load, nm_q2_dot): blocks of 256 threads
 * over the groups of R rows (each warp looping), the grid table in shared memory; the table and the
 * first weights before the wait for the kernel before (PDL). */
template <int R, int U, bool X2>
__global__ void __launch_bounds__(256) k_nm_q2_lin(const gp_rec *r, const int64_t *e)
{
    __shared__ uint64_t sg[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sg[i] = nm_iq2_grid[i];
    }
    const int rows = DI(3), cols = DI(4), t = DI(5), lane = threadIdx.x % 32;
    const size_t rb = nm_q2_rb(cols);
    const int warps = gridDim.x * (blockDim.x / 32), np = cols / 32;
    const uint8_t *W = DP(const uint8_t, 2);
    bool waited = false;
    for (int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R; ; row += warps * R) {
        const int nr = min(R, rows - row);              /* (the last group: rows repeated, not stored) */
        const uint8_t *w = W + (size_t)min(row, rows - 1) * rb;
        const size_t rbs = nr == R ? rb : 0;
        uint2 pt[U][R];
        uint32_t dh[U][R];
        if (row < rows) {
            nm_q2_load<R, U>(w, rbs, cols, 0, lane, pt, dh);
        }
        if (!waited) {
            PDL_START();
            __syncthreads();
            waited = true;
        }
        if (row >= rows) {
            break;
        }
        for (int j = 0; j < t; ++j) {
            const int8_t *q = DP(const int8_t, 0) + (size_t)j * cols;
            const float *s = DP(const float, 1) + (size_t)j * (cols / 32);
            const int8_t *ql = X2 ? DP(const int8_t, 7) + (size_t)j * cols : NULL;
            float acc[R] = {};
            for (int c0 = 0; c0 < np; c0 += 32 * U) {
                if (c0 > 0 || j > 0) {
                    nm_q2_load<R, U>(w, rbs, cols, c0, lane, pt, dh);
                }
                nm_q2_dot<R, U, X2>(cols, c0, lane, pt, dh, q, ql, s, sg, acc);
            }
            const float v = nm_wsum<R>(acc, lane);
            const int rr = lane / (32 / R);
            if (lane % (32 / R) == 0 && rr < nr) {
                DP(float, 6)[(size_t)j * rows + row + rr] = v;
            }
        }
    }
}

template <int R, int U>
static void nm_q2_lin_launch(bool x2, unsigned nb, cudaStream_t s, const gp_rec *dr, const int64_t *denv)
{
    if (x2) {
        k_nm_q2_lin<R, U, true><<<nb, 256, 0, s>>>(dr, denv);
    } else {
        k_nm_q2_lin<R, U, false><<<nb, 256, 0, s>>>(dr, denv);
    }
}

/* the rows of a warp of the hot experts (k_nm_hot_up8, k_nm_hot_down8) */
#define NM_HR 4

/* The hot experts on int8 x (GP_NM_HOT_MOE with operands 13-16: qx, sx (t x L), qa, sa (t k x
 * inner)): x quantized once, the up rows, relu^2, the act quantized, the down rows. */
__global__ void k_nm_hot_up8(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ uint64_t sg[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sg[i] = nm_iq2_grid[i];
    }
    __syncthreads();
    const int k = DI(9), L = DI(10), inner = DI(11), pr = blockIdx.y, lane = threadIdx.x % 32;
    const int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * NM_HR;
    const int slot = DP(const int, 3)[DP(const int, 1)[pr]];
    if (row >= inner || slot < 0) {
        return;
    }
    const size_t rb = nm_q2_rb(L);
    const uint8_t *w = DP(const uint8_t, 4) + ((size_t)slot * inner + row) * rb;
    const int8_t *q = DP(const int8_t, 13) + (size_t)(pr / k) * L;
    const float *s = DP(const float, 14) + (size_t)(pr / k) * (L / 32);
    const int8_t *ql = r->tag[17] != GP_T_NONE && DP(const int8_t, 17) != NULL ?
                       DP(const int8_t, 17) + (size_t)(pr / k) * L : NULL;
    float acc[NM_HR] = {};
    nm_q2_rows<NM_HR>(w, rb, L, q, ql, s, lane, acc, sg);       /* (inner: a multiple of NM_HR) */
    float v = nm_wsum<NM_HR>(acc, lane);
    if (lane % (32 / NM_HR) == 0) {
        v = fmaxf(v, 0.f);
        DP(float, 6)[(size_t)pr * inner + row + lane / (32 / NM_HR)] = v * v;
    }
}

__global__ void k_nm_hot_qa(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int k = DI(9), inner = DI(11), t = DI(12);
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x, n = (int64_t)t * k * inner / 32;
    if (b >= n) {
        return;
    }
    const int pr = (int)(b / (inner / 32));
    if (DP(const int, 3)[DP(const int, 1)[pr]] < 0) {
        return;
    }
    if (r->tag[18] != GP_T_NONE && DP(int8_t, 18) != NULL) {
        nm_qx32_2(DP(const float, 6) + 32 * b, DP(int8_t, 15) + 32 * b, DP(int8_t, 18) + 32 * b, DP(float, 16) + b);
    } else {
        nm_qx32(DP(const float, 6) + 32 * b, DP(int8_t, 15) + 32 * b, DP(float, 16) + b);
    }
}

/* the down rows of the k pairs of token blockIdx.y (a warp a pair: blockDim k warps, NM_LR rows a
 * block), times their weights, summed in the order of the slots (k_nm_hot_sum's) through shared
 * memory: straight to the sum (operand 8). (The global tables of the grid: blocks of a few rows
 * each filling the shared copy cost more than the lookups.) */
#define NM_KMAX 16
__global__ void k_nm_hot_down8(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float part[NM_KMAX][NM_HR];
    __shared__ uint64_t sg[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sg[i] = nm_iq2_grid[i];
    }
    __syncthreads();
    const int k = DI(9), L = DI(10), inner = DI(11), j = blockIdx.y, lane = threadIdx.x % 32, q = threadIdx.x / 32;
    const int row = blockIdx.x * NM_HR;
    const size_t rb = nm_q2_rb(inner);
    const int pr = j * k + q, slot = DP(const int, 3)[DP(const int, 1)[pr]];
    float acc[NM_HR] = {};
    if (slot >= 0) {
        const uint8_t *w = DP(const uint8_t, 5) + ((size_t)slot * L + row) * rb;
        const int8_t *qa = DP(const int8_t, 15) + (size_t)pr * inner;
        const float *sa = DP(const float, 16) + (size_t)pr * (inner / 32);
        const int8_t *ql = r->tag[18] != GP_T_NONE && DP(const int8_t, 18) != NULL ?
                           DP(const int8_t, 18) + (size_t)pr * inner : NULL;
        nm_q2_rows<NM_HR>(w, rb, inner, qa, ql, sa, lane, acc, sg);
    }
    const float wv = slot >= 0 ? DP(const float, 2)[pr] : 0.f;
    const float v = nm_wsum<NM_HR>(acc, lane);
    if (lane % (32 / NM_HR) == 0) {
        part[q][lane / (32 / NM_HR)] = v * wv;
    }
    __syncthreads();
    if (threadIdx.x < NM_HR) {
        float sum = 0.f;
        for (int qq = 0; qq < k; ++qq) {
            sum += part[qq][threadIdx.x];
        }
        DP(float, 8)[(size_t)j * L + row + threadIdx.x] = sum;
    }
}

__global__ void k_nm_hot_qx(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int L = DI(10), t = DI(12);
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < (int64_t)t * L / 32) {
        if (r->tag[17] != GP_T_NONE && DP(int8_t, 17) != NULL) {
            nm_qx32_2(DP(const float, 0) + 32 * b, DP(int8_t, 13) + 32 * b, DP(int8_t, 17) + 32 * b, DP(float, 14) + b);
        } else {
            nm_qx32(DP(const float, 0) + 32 * b, DP(int8_t, 13) + 32 * b, DP(float, 14) + b);
        }
    }
}

/* ---------- the hot experts of a small group by expert (GP_NM_HOT_MOE, int8 x, 2 <= t <= NM_HG) ----------
 * The pairs of a verify group share experts (the tokens of a block are alike: 40 experts for the 72
 * pairs of a layer on the code prompt): a block for each pair (grid y), and only the first pair of
 * each expert works, for all the pairs of that expert (nm_hg_pairs: at most one a token). Each warp
 * decodes the weights of its R rows once a part and adds their products with the x of P pairs at a
 * time (more: the weights decoded again); the sum of each pair in the order of the one-pair kernels
 * (the same floats). k_nm_hg_up: relu^2 to act (then k_nm_hot_qa); k_nm_hg_down: times the weight,
 * to de (operand 7); k_nm_hg_sum: the slots of each token in order. 4 warps a block, at most 102
 * registers (NM_HGB: the spills of more pairs a pass cost more than the decodes they save). The
 * kernels are bound by the LSU pipe (the grid lookups, the x loads, the shuffles: nm_wsum); R = 4,
 * P = 2 on the 3090: the hot experts of a verify group 7.0 -> 5.95 ms. */
#define NM_HG 8
#ifndef NM_HGB
#define NM_HGB 5                    /* the blocks of an SM (the registers: 102) */
#endif

/* the pairs of the expert of pair p in the order of the pairs to pr[] (shared), their count (0: pair p
 * is not the first of its expert, or a cold one); blockDim >= t k (<= 128 pairs) */
__device__ __forceinline__ int nm_hg_pairs(const gp_rec *r, const int64_t *e, int p, int np, int *pr)
{
    __shared__ uint32_t mk[4];
    __shared__ int before;
    const int i = threadIdx.x;
    const int *slots = DP(const int, 3), *idx = DP(const int, 1);
    const int sp = slots[idx[p]];
    const int si = i < np ? slots[idx[i]] : -1;
    if (i == 0) {
        before = 0;
    }
    __syncthreads();
    if (i < p && si == sp) {
        before = 1;
    }
    const uint32_t b = __ballot_sync(RC_FULL_MASK, i >= p && i < np && si == sp);
    if (i < 128 && i % 32 == 0) {
        mk[i / 32] = b;
    }
    __syncthreads();
    if (sp < 0 || before) {
        return 0;
    }
    int n = 0;
    for (int w = 0; w < 4; ++w) {
        for (uint32_t m = mk[w]; m; m &= m - 1) {
            if (i == 0 && n < NM_HG) {
                pr[n] = 32 * w + __ffs(m) - 1;
            }
            ++n;
        }
    }
    __syncthreads();
    return min(n, NM_HG);
}

/* acc[m][r] += the products of R rows (row0, rb apart) with the x of n <= P pairs (x of pair m: qh +
 * xo[m] (shared), its scales xs + xo[m] / 32): the weights of a part decoded once for all */
template <int R, int P>
__device__ __forceinline__ void nm_hg_rows(const uint8_t *row0, size_t rb, int cols, int n, const int *xo,
                                           const int8_t *qh, const int8_t *ql, const float *xs, int lane,
                                           const uint64_t *grid, float (&acc)[P][R])
{
    const int np = cols / 32;
    for (int c0 = 0; c0 < np; c0 += 32) {
        uint2 pt[1][R];
        uint32_t dh[1][R];
        nm_q2_load<R, 1>(row0, rb, cols, c0, lane, pt, dh);
        const int p = c0 + lane;
        if (p >= np) {
            continue;
        }
        int w[R][8];
        float db[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                nm_q2_wl(pt[0][r].x, pt[0][r].y, l, grid, &w[r][2 * l], &w[r][2 * l + 1]);
            }
            db[r] = nm_q2_db((uint16_t)dh[0][r], pt[0][r].y);
        }
#pragma unroll
        for (int m = 0; m < P; ++m) {
            if (m < n) {
                const int8_t *q = qh + xo[m] + 32 * p;
                const int4 h0 = __ldg((const int4 *)q), h1 = __ldg((const int4 *)(q + 16));
                int4 l0 = make_int4(0, 0, 0, 0), l1 = l0;
                if (ql != NULL) {
                    l0 = __ldg((const int4 *)(ql + xo[m] + 32 * p));
                    l1 = __ldg((const int4 *)(ql + xo[m] + 32 * p + 16));
                }
                const int hx[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                const int lx[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
                const float sx = __ldg(xs + xo[m] / 32 + p);
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    int sh = 0, sl = 0;
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        sh = rc_dp4a(w[r][2 * l + 1], hx[2 * l + 1], rc_dp4a(w[r][2 * l], hx[2 * l], sh));
                        sl = rc_dp4a(w[r][2 * l + 1], lx[2 * l + 1], rc_dp4a(w[r][2 * l], lx[2 * l], sl));
                    }
                    acc[m][r] += (float)(ql != NULL ? 256 * sh + sl : sh) * db[r] * sx;
                }
            }
        }
    }
}

/* R rows a warp, 4 warps a block; P pairs a pass */
template <int R, int P>
__global__ void __launch_bounds__(128, NM_HGB) k_nm_hg_up(const gp_rec *r, const int64_t *e)
{
    __shared__ uint64_t sg[256];
    __shared__ int pr[NM_HG], xo[NM_HG];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sg[i] = nm_iq2_grid[i];
    }
    PDL_START();
    const int k = DI(9), L = DI(10), inner = DI(11), t = DI(12), lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    const int n = nm_hg_pairs(r, e, blockIdx.y, t * k, pr);
    if (n == 0) {
        return;
    }
    if (threadIdx.x < n) {
        xo[threadIdx.x] = pr[threadIdx.x] / k * L;
    }
    __syncthreads();
    const int8_t *ql = r->tag[17] != GP_T_NONE && DP(const int8_t, 17) != NULL ? DP(const int8_t, 17) : NULL;
    const int row = (blockIdx.x * 4 + wp) * R;
    const int slot = DP(const int, 3)[DP(const int, 1)[blockIdx.y]];
    const size_t rb = nm_q2_rb(L);
    for (int m0 = 0; m0 < n; m0 += P) {
        float acc[P][R] = {};
        nm_hg_rows<R, P>(DP(const uint8_t, 4) + ((size_t)slot * inner + row) * rb, rb, L, min(P, n - m0), xo + m0,
                         DP(const int8_t, 13), ql, DP(const float, 14), lane, sg, acc);
        /* (value m R + rr in the lanes (m R + rr) 32 / (P R)) */
        float v = nm_wsum<P * R>(&acc[0][0], lane);
        const int m = lane / (32 / (P * R)) / R, rr = lane / (32 / (P * R)) % R;
        if (lane % (32 / (P * R)) == 0 && m0 + m < n) {
            v = fmaxf(v, 0.f);
            DP(float, 6)[(size_t)pr[m0 + m] * inner + row + rr] = v * v;
        }
    }
}

template <int R, int P>
__global__ void __launch_bounds__(128, NM_HGB) k_nm_hg_down(const gp_rec *r, const int64_t *e)
{
    __shared__ uint64_t sg[256];
    __shared__ int pr[NM_HG], xo[NM_HG];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        sg[i] = nm_iq2_grid[i];
    }
    PDL_START();
    const int k = DI(9), L = DI(10), inner = DI(11), t = DI(12), lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    const int n = nm_hg_pairs(r, e, blockIdx.y, t * k, pr);
    if (n == 0) {
        return;
    }
    if (threadIdx.x < n) {
        xo[threadIdx.x] = pr[threadIdx.x] * inner;
    }
    __syncthreads();
    const int8_t *ql = r->tag[18] != GP_T_NONE && DP(const int8_t, 18) != NULL ? DP(const int8_t, 18) : NULL;
    const int row = (blockIdx.x * 4 + wp) * R;
    const int slot = DP(const int, 3)[DP(const int, 1)[blockIdx.y]];
    const size_t rb = nm_q2_rb(inner);
    for (int m0 = 0; m0 < n; m0 += P) {
        float acc[P][R] = {};
        nm_hg_rows<R, P>(DP(const uint8_t, 5) + ((size_t)slot * L + row) * rb, rb, inner, min(P, n - m0), xo + m0,
                         DP(const int8_t, 15), ql, DP(const float, 16), lane, sg, acc);
        const float v = nm_wsum<P * R>(&acc[0][0], lane);
        const int m = lane / (32 / (P * R)) / R, rr = lane / (32 / (P * R)) % R;
        if (lane % (32 / (P * R)) == 0 && m0 + m < n) {
            const int q = pr[m0 + m];
            DP(float, 7)[(size_t)q * L + row + rr] = v * DP(const float, 2)[q];
        }
    }
}

/* the form of k_nm_hg_*: RECLANK_NM_HGF "RP" (rows a warp, pairs a pass; a test), default 42 */
static int nm_hg_form(void)
{
    static int f = -1;
    if (f < 0) {
        const char *v = getenv("RECLANK_NM_HGF");
        f = v ? atoi(v) : 42;
    }
    return f;
}

template <int R, int P>
static void nm_hg_launch(int64_t t, int64_t k, int64_t L, int64_t inner, cudaStream_t s, const gp_rec *dr,
                         const int64_t *denv)
{
    k_nm_hg_up<R, P><<<dim3((unsigned)(inner / (4 * R)), (unsigned)(t * k)), 128, 0, s>>>(dr, denv);
    k_nm_hot_qa<<<(unsigned)((t * k * inner / 32 + 127) / 128), 128, 0, s>>>(dr, denv);
    k_nm_hg_down<R, P><<<dim3((unsigned)(L / (4 * R)), (unsigned)(t * k)), 128, 0, s>>>(dr, denv);
}

__global__ void k_nm_hg_sum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int k = DI(9), L = DI(10), j = blockIdx.y;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= L) {
        return;
    }
    const int *slots = DP(const int, 3), *idx = DP(const int, 1) + (size_t)j * k;
    const float *de = DP(const float, 7) + (size_t)j * k * L;
    float s = 0.f;
    for (int q = 0; q < k; ++q) {
        if (slots[idx[q]] >= 0) {
            s += de[(size_t)q * L + c];
        }
    }
    DP(float, 8)[(size_t)j * L + c] = s;
    if (NM_OPQ(19) != NULL) {
        nm_qx_warp(s, (size_t)j * L + c, NM_OPQ(19), DP(float, 20), NM_OPQ(21));
    }
}

/* the sums of GP_NM_HOT_MOE (operand 8) to q, s, ql (19-21) where the kernels of the form do not */
__global__ void k_nm_hot_oq(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int L = DI(10), t = DI(12);
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < (int64_t)t * L / 32) {
        if (NM_OPQ(21) != NULL) {
            nm_qx32_2(DP(const float, 8) + 32 * b, DP(int8_t, 19) + 32 * b, DP(int8_t, 21) + 32 * b, DP(float, 20) + b);
        } else {
            nm_qx32(DP(const float, 8) + 32 * b, DP(int8_t, 19) + 32 * b, DP(float, 20) + b);
        }
    }
}

/* ---------- the attention on the bfloat16 cache ----------
 * GP_NM_KV: k, v (t x per float32), K, V (the cache rows, bfloat16 bits, position-major), per,
 * t, pos: rounded to the nearest even (as torch's bfloat16).
 * GP_NM_ATTN2: q (t x nq x hd, raw), K, V, out, nq, nk, hd (128), t, pos, scratch, scale. A block
 * for each token, KV head and chunk of the positions the token sees (pos + j + 1, in NM_NCH(t)
 * chunks: the grid does not depend on pos, so a graph holds it): its 16 query heads over tiles of
 * 32 keys in shared memory, an online softmax; then k_nm_attn_merge adds the chunks of each query
 * head. Float32 math (Pascal). */
#define NM_HD 128
#define NM_QH 16
#define NM_TILE 32
#define NM_PARTS 256
__host__ __device__ __forceinline__ int nm_nch(int t)
{
    return t >= NM_PARTS ? 1 : (t == 1 ? 128 : NM_PARTS / t);
}

__device__ __forceinline__ uint16_t nm_bf16(float x)
{
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

__global__ void k_nm_kv(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int per = DI(4), t = DI(5);
    const int64_t pos = di(r, e, 6), i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)t * per) {
        return;
    }
    const size_t o = (size_t)pos * per + i;
    DP(uint16_t, 2)[o] = nm_bf16(DP(const float, 0)[i]);
    DP(uint16_t, 3)[o] = nm_bf16(DP(const float, 1)[i]);
}

__global__ void __launch_bounds__(256) k_nm_attn2(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float qs[NM_QH][NM_HD];
    __shared__ float ks[NM_TILE][NM_HD + 1];
    __shared__ float vs[NM_TILE][NM_HD];
    __shared__ float ss[NM_QH][NM_TILE];
    const int nq = DI(4), nk = DI(5), t = DI(7), nch = nm_nch(t);
    const int64_t pos = di(r, e, 8);
    const int c = blockIdx.x, kh = blockIdx.y, j = blockIdx.z, tid = threadIdx.x;
    const float scale = df(r, e, 10);
    const int64_t n = pos + j + 1, cs = (n + nch - 1) / nch, p0 = (int64_t)c * cs;
    const int64_t p1 = p0 + cs < n ? p0 + cs : n;
    const size_t rowe = (size_t)nk * NM_HD;           /* the bfloat16 values of a position */
    const uint16_t *K = DP(const uint16_t, 1) + (size_t)kh * NM_HD, *V = DP(const uint16_t, 2) + (size_t)kh * NM_HD;
    for (int x = tid; x < NM_QH * NM_HD; x += 256) {
        int h = x / NM_HD, d = x % NM_HD;
        qs[h][d] = DP(const float, 0)[((size_t)j * nq + kh * NM_QH + h) * NM_HD + d] * scale;
    }
    const int h = tid / 16, dl = tid % 16;            /* this thread: head h, dims dl + 16 u */
    float acc[8], m = -INFINITY, l = 0.f;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        acc[u] = 0.f;
    }
    for (int64_t b = p0; b < p1; b += NM_TILE) {
        __syncthreads();
        /* the tile: 32 rows of 128 keys and values, 16 values a thread of each */
        {
            int pr = tid / 8, d0 = (tid % 8) * 16;
            int64_t p = b + pr;
            if (p < p1) {
                const uint4 *ka = (const uint4 *)(K + (size_t)p * rowe + d0);
                const uint4 *va = (const uint4 *)(V + (size_t)p * rowe + d0);
                uint4 kq[2] = {ka[0], ka[1]}, vq[2] = {va[0], va[1]};
                const uint32_t *kw = (const uint32_t *)kq, *vw = (const uint32_t *)vq;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    ks[pr][d0 + 2 * u] = __uint_as_float(kw[u] << 16);
                    ks[pr][d0 + 2 * u + 1] = __uint_as_float(kw[u] & 0xffff0000u);
                    vs[pr][d0 + 2 * u] = __uint_as_float(vw[u] << 16);
                    vs[pr][d0 + 2 * u + 1] = __uint_as_float(vw[u] & 0xffff0000u);
                }
            } else {
                #pragma unroll
                for (int u = 0; u < 16; ++u) {
                    ks[pr][d0 + u] = 0.f;
                    vs[pr][d0 + u] = 0.f;
                }
            }
        }
        __syncthreads();
        /* the scores: 16 heads x 32 keys, 2 a thread */
        #pragma unroll
        for (int w = 0; w < 2; ++w) {
            int x = tid + 256 * w, hh = x / NM_TILE, pp = x % NM_TILE;
            float s = 0.f;
            #pragma unroll 16
            for (int d = 0; d < NM_HD; ++d) {
                s += qs[hh][d] * ks[pp][d];
            }
            ss[hh][pp] = b + pp < p1 ? s : -INFINITY;
        }
        __syncthreads();
        float tm = -INFINITY;
        #pragma unroll 8
        for (int pp = 0; pp < NM_TILE; ++pp) {
            tm = fmaxf(tm, ss[h][pp]);
        }
        const float m2 = fmaxf(m, tm), cr = expf(m - m2);
        l *= cr;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            acc[u] *= cr;
        }
        for (int pp = 0; pp < NM_TILE; ++pp) {
            const float w = expf(ss[h][pp] - m2);
            l += w;
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                acc[u] += w * vs[pp][dl + 16 * u];
            }
        }
        m = m2;
    }
    /* the partial of (token, chunk) for this head: acc, then m and l */
    float *part = DP(float, 9) + ((size_t)(j * nch + c) * nq + kh * NM_QH + h) * (NM_HD + 2);
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        part[dl + 16 * u] = acc[u];
    }
    if (dl == 0) {
        part[NM_HD] = m;
        part[NM_HD + 1] = l;
    }
}

__global__ void k_nm_attn_merge(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int nq = DI(4), t = DI(7), nch = nm_nch(t), d = threadIdx.x;
    const int h = blockIdx.x, j = blockIdx.y;
    const float *base = DP(const float, 9);
    float M = -INFINITY;
    for (int c = 0; c < nch; ++c) {
        M = fmaxf(M, base[((size_t)(j * nch + c) * nq + h) * (NM_HD + 2) + NM_HD]);
    }
    float num = 0.f, den = 0.f;
    for (int c = 0; c < nch; ++c) {
        const float *p = base + ((size_t)(j * nch + c) * nq + h) * (NM_HD + 2);
        const float lc = p[NM_HD + 1];
        if (lc > 0.f) {
            const float w = expf(p[NM_HD] - M);
            num += w * p[d];
            den += w * lc;
        }
    }
    DP(float, 3)[((size_t)j * nq + h) * NM_HD + d] = num / den;
}

/* ---------- the experts of a prompt group (GP_NM_GROUP_MOE) ----------
 * Every pair of the group on the GPU: the experts it holds from their slots, the others straight
 * from the host memory the GPU maps (zero copy: each expert crosses PCIe once for the group, at
 * ~11-12 GB/s on PCIe 3.0 x16), so no buffer of copies takes the memory of a small card.
 * Operands: x (t x L), idx, val, pup, pdn (the device address of each expert's up and down rows,
 * int64), qx, sx (t x L int8, t x L / 32), act (t k x inner), qa, sa, de (t k x L), out (t x L),
 * work (int32: E + 1 starts, E counts, t k pairs), k, L, inner, t, E, nreal. A pair of idx -1 is
 * left out (another GPU runs it: NemotronGPU's expert device, models/nemotron/gpu.py).
 * k_nm_gsort (one block): the pairs of the real tokens by expert. k_nm_gup: a block for each
 * expert and tile of 32 up rows: the tile into shared memory once, then every pair of the expert
 * (dp4a on int8 x), relu^2. k_nm_gqa: the act to int8. k_nm_gdown: as k_nm_gup on the down rows,
 * times the weight. k_nm_gsum: the slots of each token in their order. */
#define NM_GROWS 32
#define NM_GMAXB 10                 /* the blocks of 256 of a row: inner <= 2560 */

__global__ void k_nm_gqx(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int L = DI(14), t = DI(16);
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < (int64_t)t * L / 32) {
        if (r->tag[19] != GP_T_NONE && DP(int8_t, 19) != NULL) {
            nm_qx32_2(DP(const float, 0) + 32 * b, DP(int8_t, 5) + 32 * b, DP(int8_t, 19) + 32 * b, DP(float, 6) + b);
        } else {
            nm_qx32(DP(const float, 0) + 32 * b, DP(int8_t, 5) + 32 * b, DP(float, 6) + b);
        }
    }
}

__global__ void __launch_bounds__(1024) k_nm_gsort(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int k = DI(13), t = DI(16), E = DI(17);
    int nreal = DI(18);
    nreal = nreal > 0 && nreal < t ? nreal : t;
    const int *idx = DP(const int, 1);
    int *start = DP(int, 12), *fill = start + E + 1, *order = fill + E;
    for (int x = threadIdx.x; x < E; x += blockDim.x) {
        fill[x] = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < nreal * k; p += blockDim.x) {
        if (idx[p] >= 0) {
            atomicAdd(&fill[idx[p]], 1);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int s = 0;
        for (int x = 0; x < E; ++x) {
            start[x] = s;
            s += fill[x];
            fill[x] = 0;
        }
        start[E] = s;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < nreal * k; p += blockDim.x) {
        int x = idx[p];
        if (x >= 0) {
            order[start[x] + atomicAdd(&fill[x], 1)] = p;
        }
    }
}

/* the tile: nrow rows of rb bytes from src (8-byte aligned) into dst */
__device__ __forceinline__ void nm_tile_load(uint8_t *dst, const uint8_t *src, size_t nbytes)
{
    const uint2 *s = (const uint2 *)src;
    uint2 *d = (uint2 *)dst;
    for (size_t i = threadIdx.x; i < nbytes / 8; i += blockDim.x) {
        d[i] = s[i];
    }
}

__global__ void __launch_bounds__(256) k_nm_gup(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ __align__(16) uint8_t tile[NM_GROWS * NM_Q2_RB(1024)];      /* L <= 1024 */
    const int k = DI(13), L = DI(14), inner = DI(15), E = DI(17);
    const int x = blockIdx.x, r0 = blockIdx.y * NM_GROWS;
    const int *start = DP(const int, 12), *order = start + 2 * E + 1;
    const int n0 = start[x], n1 = start[x + 1];
    if (n0 == n1 || r0 >= inner) {
        return;
    }
    const size_t rb = nm_q2_rb(L);
    const int nrow = min(NM_GROWS, inner - r0);
    nm_tile_load(tile, (const uint8_t *)(intptr_t)DP(const int64_t, 3)[x] + (size_t)r0 * rb, nrow * rb);
    __syncthreads();
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int8_t *qs[NM_TT], *qls[NM_TT];
    const float *ss[NM_TT];
    int prs[NM_TT];
    const bool x2 = r->tag[19] != GP_T_NONE && DP(const int8_t, 19) != NULL;
    for (int a0 = n0; a0 < n1; a0 += NM_TT) {
        const int n = min(NM_TT, n1 - a0);
        for (int tt = 0; tt < NM_TT; ++tt) {
            prs[tt] = order[a0 + min(tt, n - 1)];
            qs[tt] = DP(const int8_t, 5) + (size_t)(prs[tt] / k) * L;
            ss[tt] = DP(const float, 6) + (size_t)(prs[tt] / k) * (L / 32);
            if (x2) {
                qls[tt] = DP(const int8_t, 19) + (size_t)(prs[tt] / k) * L;
            }
        }
        for (int rr = w; rr < nrow; rr += 8) {
            float acc[NM_TT];
            #pragma unroll
            for (int tt = 0; tt < NM_TT; ++tt) {
                acc[tt] = 0.f;
            }
            nm_row_tt2(tile + rr * rb, L, qs, x2 ? qls : NULL, ss, n, lane, acc);
            #pragma unroll
            for (int tt = 0; tt < NM_TT; ++tt) {
                if (lane == tt && tt < n) {
                    float v = fmaxf(acc[tt], 0.f);
                    DP(float, 7)[(size_t)prs[tt] * inner + r0 + rr] = v * v;
                }
            }
        }
    }
}

__global__ void k_nm_gqa(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int k = DI(13), inner = DI(15), t = DI(16);
    int nreal = DI(18);
    nreal = nreal > 0 && nreal < t ? nreal : t;
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b < (int64_t)nreal * k * inner / 32) {
        if (r->tag[20] != GP_T_NONE && DP(int8_t, 20) != NULL) {
            nm_qx32_2(DP(const float, 7) + 32 * b, DP(int8_t, 8) + 32 * b, DP(int8_t, 20) + 32 * b, DP(float, 9) + b);
        } else {
            nm_qx32(DP(const float, 7) + 32 * b, DP(int8_t, 8) + 32 * b, DP(float, 9) + b);
        }
    }
}

__global__ void __launch_bounds__(256) k_nm_gdown(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ __align__(16) uint8_t tile[NM_GROWS * NM_Q2_RB(256 * NM_GMAXB)];
    const int L = DI(14), inner = DI(15), E = DI(17);
    const int x = blockIdx.x, r0 = blockIdx.y * NM_GROWS;
    const int *start = DP(const int, 12), *order = start + 2 * E + 1;
    const int n0 = start[x], n1 = start[x + 1];
    if (n0 == n1 || r0 >= L) {
        return;
    }
    const size_t rb = nm_q2_rb(inner);
    const int nrow = min(NM_GROWS, L - r0);
    nm_tile_load(tile, (const uint8_t *)(intptr_t)DP(const int64_t, 4)[x] + (size_t)r0 * rb, nrow * rb);
    __syncthreads();
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int8_t *qs[NM_TT], *qls[NM_TT];
    const float *ss[NM_TT];
    int prs[NM_TT];
    const bool x2 = r->tag[20] != GP_T_NONE && DP(const int8_t, 20) != NULL;
    for (int a0 = n0; a0 < n1; a0 += NM_TT) {
        const int n = min(NM_TT, n1 - a0);
        for (int tt = 0; tt < NM_TT; ++tt) {
            prs[tt] = order[a0 + min(tt, n - 1)];
            qs[tt] = DP(const int8_t, 8) + (size_t)prs[tt] * inner;
            ss[tt] = DP(const float, 9) + (size_t)prs[tt] * (inner / 32);
            if (x2) {
                qls[tt] = DP(const int8_t, 20) + (size_t)prs[tt] * inner;
            }
        }
        for (int rr = w; rr < nrow; rr += 8) {
            float acc[NM_TT];
            #pragma unroll
            for (int tt = 0; tt < NM_TT; ++tt) {
                acc[tt] = 0.f;
            }
            nm_row_tt2(tile + rr * rb, inner, qs, x2 ? qls : NULL, ss, n, lane, acc);
            #pragma unroll
            for (int tt = 0; tt < NM_TT; ++tt) {
                if (lane == tt && tt < n) {
                    DP(float, 10)[(size_t)prs[tt] * L + r0 + rr] = acc[tt] * DP(const float, 2)[prs[tt]];
                }
            }
        }
    }
}

__global__ void k_nm_gsum(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int k = DI(13), L = DI(14), t = DI(16), j = blockIdx.y;
    int nreal = DI(18);
    nreal = nreal > 0 && nreal < t ? nreal : t;
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= L) {
        return;
    }
    float s = 0.f;
    if (j < nreal) {
        const float *de = DP(const float, 10) + (size_t)j * k * L;
        const int *idx = DP(const int, 1) + (size_t)j * k;
        for (int q = 0; q < k; ++q) {
            if (idx[q] >= 0) {
                s += de[(size_t)q * L + c];
            }
        }
    }
    DP(float, 11)[(size_t)j * L + c] = s;
}

/* GP_NM_IQ2_LIN of a prompt group (t > 16): a block for each tile of 32 rows, the tile in shared
 * memory once (cols <= 5376: 1392 bytes a row, 44 KB); warp w takes rows w, w + 8, ...,
 * NM_TT tokens at a time (nm_row_tt: each lane's weights decoded once for them). */
#define NM_LMAXB 21
__global__ void __launch_bounds__(256) k_nm_iq2_gemm(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ __align__(16) uint8_t tile[NM_GROWS * NM_Q2_RB(256 * NM_LMAXB)];
    const int rows = DI(3), cols = DI(4), t = DI(5), r0 = blockIdx.x * NM_GROWS;
    const size_t rb = nm_q2_rb(cols);
    const int nrow = min(NM_GROWS, rows - r0);
    const uint8_t *src = DP(const uint8_t, 2) + (size_t)r0 * rb;
    for (size_t i = threadIdx.x; i < nrow * rb / 8; i += blockDim.x) {
        ((uint2 *)tile)[i] = ((const uint2 *)src)[i];
    }
    __syncthreads();
    const int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int8_t *qs[NM_TT];
    const float *ss[NM_TT];
    for (int j0 = 0; j0 < t; j0 += NM_TT) {
        const int n = min(NM_TT, t - j0);
        for (int tt = 0; tt < NM_TT; ++tt) {
            const int j = j0 + min(tt, n - 1);
            qs[tt] = DP(const int8_t, 0) + (size_t)j * cols;
            ss[tt] = DP(const float, 1) + (size_t)j * (cols / 32);
        }
        for (int rr = w; rr < nrow; rr += 8) {
            float acc[NM_TT];
            #pragma unroll
            for (int tt = 0; tt < NM_TT; ++tt) {
                acc[tt] = 0.f;
            }
            nm_row_tt(tile + rr * rb, cols, qs, ss, n, lane, acc);
            #pragma unroll
            for (int tt = 0; tt < NM_TT; ++tt) {
                if (lane == tt && tt < n) {
                    DP(float, 6)[(size_t)(j0 + tt) * rows + r0 + rr] = acc[tt];
                }
            }
        }
    }
}

/* ---------- the tiled int8 product of a prompt group (GP_NM_GEMM8) ----------
 * q, s (the int8 rows of x and their scales of each 32: NM_QX), w, type (KQ_IQ2R, KQ_NV4 or
 * KQ_Q8_0), rows, cols, t, out (t x rows). A block of 256 threads makes a tile of 64 rows x 64
 * tokens; for each 32 columns it decodes the 64 rows to int8 in shared memory once (the 8 values
 * of a grid index of IQ2_XXS, 4 E2M1 codes through kt_e2m1x4, or the Q8_0 bytes) with the scale of
 * each (two for NVFP4: a scale for each 16), loads the 64 int8 rows of x, then each thread adds a
 * 4 x 4 patch with dp4a (8 for each 32, as two halves for NVFP4). Float32 sums of the scaled int32
 * of each 32 (16). Pascal: 48 KB of shared memory is plenty, dp4a on sm_61. */
#define G8_BM 64
#define G8_BN 64
template <int F, bool X2>
__global__ void __launch_bounds__(256) k_nm_gemm8(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ int ws[G8_BM][9];            /* 8 words of int8 a row, padded */
    __shared__ int xs[G8_BN][9];
    __shared__ int xl[X2 ? G8_BN : 1][9];   /* the low plane of x (operand 8) */
    __shared__ float wsc[G8_BM][2], xsc[G8_BN];
    const int rows = DI(4), cols = DI(5), t = DI(6), tid = threadIdx.x;
    const int r0 = blockIdx.x * G8_BM, t0 = blockIdx.y * G8_BN;
    const size_t rb = kq_row_bytes(F, cols);
    const uint8_t *W = DP(const uint8_t, 2);
    const int8_t *Q = DP(const int8_t, 0);
    const float *S = DP(const float, 1);
    const int ty = tid / 16, tx = tid % 16;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int k0 = 0; k0 < cols; k0 += 32) {
        /* the weights: 64 rows x 32 values, a quarter row a thread */
        {
            const int rr = tid / 4, l = tid % 4, row = r0 + rr;
            int w0 = 0, w1 = 0;
            float sc0 = 0.f, sc1 = 0.f;
            if (row < rows) {
                const uint8_t *wr = W + (size_t)row * rb;
                if (F == KQ_IQ2R) {
                    const uint8_t *q2r = wr;
                    const uint2 pt = nm_q2_part(q2r, k0 / 256, (k0 % 256) / 32);
                    nm_q2_w8(pt.x, pt.y, l, nm_iq2_grid, nm_iq2_signs64, &w0, &w1);
                    sc0 = nm_q2_db(nm_q2_dh(q2r, cols, k0 / 256), pt.y);
                } else if (F == KQ_NV4) {
                    /* the 16 code bytes of the block: value j low, j + 16 high; this thread the
                     * bytes 4l .. 4l + 3: values 4l .. 4l + 3 (word l) and 16 + 4l .. (word l + 4) */
                    const uint32_t qw = *(const uint32_t *)(wr + (size_t)(k0 / 32) * 16 + 4 * l);
                    w0 = (int)kt_e2m1x4(qw & 0x0f0f0f0fu);
                    w1 = (int)kt_e2m1x4((qw >> 4) & 0x0f0f0f0fu);
                    const float g = *(const float *)(wr + cols / 2 + cols / 16);
                    sc0 = 0.5f * g * kq_e4m3(wr[cols / 2 + 2 * (k0 / 32)]);
                    sc1 = 0.5f * g * kq_e4m3(wr[cols / 2 + 2 * (k0 / 32) + 1]);
                } else {
                    const uint8_t *blk = wr + (size_t)(k0 / 32) * 34;
                    const uint16_t *b16 = (const uint16_t *)(blk + 2 + 8 * l);
                    w0 = (int)((uint32_t)b16[0] | ((uint32_t)b16[1] << 16));
                    w1 = (int)((uint32_t)b16[2] | ((uint32_t)b16[3] << 16));
                    sc0 = __half2float(__ushort_as_half(*(const uint16_t *)blk));
                }
            }
            if (F == KQ_NV4) {
                ws[rr][l] = w0;
                ws[rr][l + 4] = w1;
            } else {
                ws[rr][2 * l] = w0;
                ws[rr][2 * l + 1] = w1;
            }
            if (l == 0) {
                wsc[rr][0] = sc0;
                wsc[rr][1] = F == KQ_NV4 ? sc1 : sc0;
            }
        }
        /* x: 64 tokens x 32 int8, two words a thread */
        {
            const int tt = tid / 4, l = tid % 4, tok = t0 + tt;
            int2 v = make_int2(0, 0), v2 = make_int2(0, 0);
            if (tok < t) {
                v = *(const int2 *)(Q + (size_t)tok * cols + k0 + 8 * l);
                if (X2) {
                    v2 = *(const int2 *)(DP(const int8_t, 8) + (size_t)tok * cols + k0 + 8 * l);
                }
            }
            xs[tt][2 * l] = v.x;
            xs[tt][2 * l + 1] = v.y;
            if (X2) {
                xl[tt][2 * l] = v2.x;
                xl[tt][2 * l + 1] = v2.y;
            }
            if (l == 0) {
                xsc[tt] = tok < t ? S[(size_t)tok * (cols / 32) + k0 / 32] : 0.f;
            }
        }
        __syncthreads();
        int wv[4][8], xv[4][8], xw[X2 ? 4 : 1][8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                wv[i][u] = ws[ty * 4 + i][u];
                xv[i][u] = xs[tx * 4 + i][u];
                if (X2) {
                    xw[i][u] = xl[tx * 4 + i][u];
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float a0 = wsc[ty * 4 + i][0], a1 = wsc[ty * 4 + i][1];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                int lo = 0, hi = 0;
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    lo = rc_dp4a(wv[i][u], xv[j][u], lo);
                    hi = rc_dp4a(wv[i][u + 4], xv[j][u + 4], hi);
                }
                if (X2) {
                    /* q = 256 hi + lo: the high plane's sums times 256, plus the low plane's */
                    int lo2 = 0, hi2 = 0;
                    #pragma unroll
                    for (int u = 0; u < 4; ++u) {
                        lo2 = rc_dp4a(wv[i][u], xw[j][u], lo2);
                        hi2 = rc_dp4a(wv[i][u + 4], xw[j][u + 4], hi2);
                    }
                    lo = 256 * lo + lo2;
                    hi = 256 * hi + hi2;
                }
                const float xsj = xsc[tx * 4 + j];
                acc[i][j] += ((float)lo * a0 + (float)hi * a1) * xsj;
            }
        }
        __syncthreads();
    }
    float *out = DP(float, 7);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = r0 + ty * 4 + i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int tok = t0 + tx * 4 + j;
            if (row < rows && tok < t) {
                out[(size_t)tok * rows + row] = acc[i][j];
            }
        }
    }
}

/* ---------- the experts of a prompt group as tiles (k_nm_gup8 / k_nm_gdown8) ----------
 * A block for each expert and 32 rows of its up (or down) matrix: the 32 raw IQ2_XXS rows into
 * shared memory once (from the GPU's slot or the mapped host memory: each expert crosses PCIe once),
 * then for each 32 of the expert's pairs and each 32 columns: the rows decoded to int8 (one grid
 * index a thread), the pairs' x (one or two planes), and a 2 x 2 patch of dot products a thread
 * (dp4a). up: relu^2 into act; down: times the weight into de. Same sums for any count of pairs. */
#define GM_BM 32
#define GM_BN 32
template <bool DOWN, bool X2>
__global__ void __launch_bounds__(256) k_nm_gtile(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t gm_sm[];
    const int k = DI(13), L = DI(14), inner = DI(15), E = DI(17), tid = threadIdx.x;
    const int x = blockIdx.x, r0 = blockIdx.y * GM_BM;
    const int rows = DOWN ? L : inner, cols = DOWN ? inner : L;
    const int *start = DP(const int, 12), *order = start + 2 * E + 1;
    const int n0 = start[x], n1 = start[x + 1];
    if (n0 == n1 || r0 >= rows) {
        return;
    }
    const size_t rb = nm_q2_rb(cols);
    uint8_t *raw = gm_sm;                                         /* GM_BM x rb */
    int (*ws)[9] = (int (*)[9])(gm_sm + ((GM_BM * rb + 15) & ~(size_t)15));
    int (*xh)[9] = ws + GM_BM;
    int (*xo)[9] = xh + GM_BN;
    float *wsc = (float *)(xo + GM_BN), *xsc = wsc + GM_BM;
    const int nrow = min(GM_BM, rows - r0);
    nm_tile_load(raw, (const uint8_t *)(intptr_t)DP(const int64_t, DOWN ? 4 : 3)[x] + (size_t)r0 * rb, nrow * rb);
    const int8_t *Qh = DOWN ? DP(const int8_t, 8) : DP(const int8_t, 5);
    const int8_t *Ql = X2 ? (DOWN ? DP(const int8_t, 20) : DP(const int8_t, 19)) : NULL;
    const float *S = DOWN ? DP(const float, 9) : DP(const float, 6);
    const int ty = tid / 16, tx = tid % 16;                     /* rows 2 ty, 2 ty + 1; pairs 2 tx, + 1 */
    for (int a0 = n0; a0 < n1; a0 += GM_BN) {
        const int np = min(GM_BN, n1 - a0);
        float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
        for (int k0 = 0; k0 < cols; k0 += 32) {
            __syncthreads();
            if (tid < GM_BM * 4) {
                const int rr = tid / 4, l = tid % 4;
                int w0 = 0, w1 = 0;
                float db = 0.f;
                if (rr < nrow) {
                    const uint8_t *q2r = raw + rr * rb;
                    const uint2 pt = nm_q2_part(q2r, k0 / 256, (k0 % 256) / 32);
                    nm_q2_w8(pt.x, pt.y, l, nm_iq2_grid, nm_iq2_signs64, &w0, &w1);
                    db = nm_q2_db(nm_q2_dh(q2r, cols, k0 / 256), pt.y);
                }
                ws[rr][2 * l] = w0;
                ws[rr][2 * l + 1] = w1;
                if (l == 0) {
                    wsc[rr] = db;
                }
            } else {
                const int q = tid - GM_BM * 4, pp = q / 4, l = q % 4;     /* 32 pairs x 4 */
                int2 h = make_int2(0, 0), o = make_int2(0, 0);
                float sc = 0.f;
                if (pp < np) {
                    const int pr = order[a0 + pp];
                    const size_t row = DOWN ? (size_t)pr : (size_t)(pr / k);
                    h = *(const int2 *)(Qh + row * cols + k0 + 8 * l);
                    if (X2) {
                        o = *(const int2 *)(Ql + row * cols + k0 + 8 * l);
                    }
                    sc = S[row * (cols / 32) + k0 / 32];
                }
                xh[pp][2 * l] = h.x;
                xh[pp][2 * l + 1] = h.y;
                if (X2) {
                    xo[pp][2 * l] = o.x;
                    xo[pp][2 * l + 1] = o.y;
                }
                if (l == 0) {
                    xsc[pp] = sc;
                }
            }
            __syncthreads();
            #pragma unroll
            for (int i = 0; i < 2; ++i) {
                const int rr = 2 * ty + i;
                #pragma unroll
                for (int jj = 0; jj < 2; ++jj) {
                    const int pp = 2 * tx + jj;
                    int sh = 0, so = 0;
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        sh = rc_dp4a(ws[rr][u], xh[pp][u], sh);
                        if (X2) {
                            so = rc_dp4a(ws[rr][u], xo[pp][u], so);
                        }
                    }
                    const int s = X2 ? 256 * sh + so : sh;
                    acc[i][jj] += (float)s * wsc[rr] * xsc[pp];
                }
            }
        }
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int rr = 2 * ty + i;
            #pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                const int pp = 2 * tx + jj;
                if (rr < nrow && pp < np) {
                    const int pr = order[a0 + pp];
                    if (DOWN) {
                        DP(float, 10)[(size_t)pr * L + r0 + rr] = acc[i][jj] * DP(const float, 2)[pr];
                    } else {
                        const float v = fmaxf(acc[i][jj], 0.f);
                        DP(float, 7)[(size_t)pr * inner + r0 + rr] = v * v;
                    }
                }
            }
        }
    }
}

template <bool DOWN, bool X2>
__global__ void __launch_bounds__(256) k_nm_gtile_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) uint8_t gm_sm[];
    const int k = DI(13), L = DI(14), inner = DI(15), E = DI(17), tid = threadIdx.x;
    const int x = blockIdx.x, r0 = blockIdx.y * GM_BM;
    const int rows = DOWN ? L : inner, cols = DOWN ? inner : L;
    const int *start = DP(const int, 12), *order = start + 2 * E + 1;
    const int n0 = start[x], n1 = start[x + 1];
    if (n0 == n1 || r0 >= rows) {
        return;
    }
    const size_t rb = nm_q2_rb(cols);
    uint8_t *raw = gm_sm;                                         /* GM_BM x rb */
    int (*ws)[9] = (int (*)[9])(gm_sm + ((GM_BM * rb + 15) & ~(size_t)15));
    int (*xh)[9] = ws + GM_BM;
    int (*xo)[9] = xh + GM_BN;
    float *wsc = (float *)(xo + GM_BN), *xsc = wsc + GM_BM;
    const int nrow = min(GM_BM, rows - r0);
    nm_tile_load(raw, (const uint8_t *)(intptr_t)DP(const int64_t, DOWN ? 4 : 3)[x] + (size_t)r0 * rb, nrow * rb);
    const int8_t *Qh = DOWN ? DP(const int8_t, 8) : DP(const int8_t, 5);
    const int8_t *Ql = X2 ? (DOWN ? DP(const int8_t, 20) : DP(const int8_t, 19)) : NULL;
    const float *S = DOWN ? DP(const float, 9) : DP(const float, 6);
    /* warp w: rows 16 (w % 2), pairs 8 (w / 2): one m16n8k32 a step (a second on the low plane) */
    const int warp = tid / 32, lane = tid % 32, g = lane >> 2, c4 = lane & 3;
    const int wm = 16 * (warp % 2), wn = 8 * (warp / 2);
    for (int a0 = n0; a0 < n1; a0 += GM_BN) {
        const int np = min(GM_BN, n1 - a0);
        float acc[4] = {0.f, 0.f, 0.f, 0.f};
        for (int k0 = 0; k0 < cols; k0 += 32) {
            __syncthreads();
            if (tid < GM_BM * 4) {
                const int rr = tid / 4, l = tid % 4;
                int w0 = 0, w1 = 0;
                float db = 0.f;
                if (rr < nrow) {
                    const uint8_t *q2r = raw + rr * rb;
                    const uint2 pt = nm_q2_part(q2r, k0 / 256, (k0 % 256) / 32);
                    nm_q2_w8(pt.x, pt.y, l, nm_iq2_grid, nm_iq2_signs64, &w0, &w1);
                    db = nm_q2_db(nm_q2_dh(q2r, cols, k0 / 256), pt.y);
                }
                ws[rr][2 * l] = w0;
                ws[rr][2 * l + 1] = w1;
                if (l == 0) {
                    wsc[rr] = db;
                }
            } else {
                const int q = tid - GM_BM * 4, pp = q / 4, l = q % 4;     /* 32 pairs x 4 */
                int2 h = make_int2(0, 0), o = make_int2(0, 0);
                float sc = 0.f;
                if (pp < np) {
                    const int pr = order[a0 + pp];
                    const size_t row = DOWN ? (size_t)pr : (size_t)(pr / k);
                    h = *(const int2 *)(Qh + row * cols + k0 + 8 * l);
                    if (X2) {
                        o = *(const int2 *)(Ql + row * cols + k0 + 8 * l);
                    }
                    sc = S[row * (cols / 32) + k0 / 32];
                }
                xh[pp][2 * l] = h.x;
                xh[pp][2 * l + 1] = h.y;
                if (X2) {
                    xo[pp][2 * l] = o.x;
                    xo[pp][2 * l + 1] = o.y;
                }
                if (l == 0) {
                    xsc[pp] = sc;
                }
            }
            __syncthreads();
            {
                const uint32_t a[4] = {(uint32_t)ws[wm + g][c4], (uint32_t)ws[wm + g + 8][c4],
                                       (uint32_t)ws[wm + g][4 + c4], (uint32_t)ws[wm + g + 8][4 + c4]};
                const uint32_t b[2] = {(uint32_t)xh[wn + g][c4], (uint32_t)xh[wn + g][4 + c4]};
                int ci[4] = {0, 0, 0, 0};
                mma16832(ci, a, b);
                if (X2) {
                    const uint32_t bl[2] = {(uint32_t)xo[wn + g][c4], (uint32_t)xo[wn + g][4 + c4]};
                    int cl[4] = {0, 0, 0, 0};
                    mma16832(cl, a, bl);
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        ci[q] = 256 * ci[q] + cl[q];
                    }
                }
                const int pj = wn + 2 * c4;
                acc[0] += (float)ci[0] * wsc[wm + g] * xsc[pj];
                acc[1] += (float)ci[1] * wsc[wm + g] * xsc[pj + 1];
                acc[2] += (float)ci[2] * wsc[wm + g + 8] * xsc[pj];
                acc[3] += (float)ci[3] * wsc[wm + g + 8] * xsc[pj + 1];
            }
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int rr = wm + g + (q >= 2 ? 8 : 0), pp = wn + 2 * c4 + (q & 1);
            if (rr < nrow && pp < np) {
                const int pr = order[a0 + pp];
                if (DOWN) {
                    DP(float, 10)[(size_t)pr * L + r0 + rr] = acc[q] * DP(const float, 2)[pr];
                } else {
                    const float v = fmaxf(acc[q], 0.f);
                    DP(float, 7)[(size_t)pr * inner + r0 + rr] = v * v;
                }
            }
        }
    }
}

__host__ static size_t nm_gtile_smem(int cols)
{
    const size_t rb = nm_q2_rb(cols);
    return ((GM_BM * rb + 15) & ~(size_t)15) + (size_t)(GM_BM + 2 * GM_BN) * 9 * 4 + (GM_BM + GM_BN) * 4;
}

/* k_nm_gemm8 with 128 columns a step (G8_BK): a quarter of the barriers, 4 parts of 32 (8 for the
 * scales of NVFP4) in flight a step. The weights: 64 rows x 128 values, 4 quarter-rows a thread
 * (one 32-part each of the 4); x: 64 tokens x 128 (x 2 planes), 8 words a thread. */
#define G8_BK 128
/* the word w of row r of k_nm_gemm8k's tiles: the 16-byte chunks of a row in the order chunk ^ (r / 4
 * % 8), so the 16 rows of tokens a warp reads (4 apart) are in other banks; read 4 words at a time
 * (the rows padded to 33 words and read a word at a time took 96 loads for 256 dp4a a part) */
#define G8_SW(r, w) (((((w) >> 2) ^ (((r) >> 2) & 7)) << 2) | ((w) & 3))
template <int F, bool X2>
__global__ void __launch_bounds__(256, 2) k_nm_gemm8k(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ __align__(16) int ws[G8_BM][G8_BK / 4];
    __shared__ __align__(16) int xs[G8_BN][G8_BK / 4];
    __shared__ __align__(16) int xl[X2 ? G8_BN : 1][G8_BK / 4];
    __shared__ float wsc[G8_BM][G8_BK / 16], xsc[G8_BN][G8_BK / 32];
    const int rows = DI(4), cols = DI(5), t = DI(6), tid = threadIdx.x;
    const int r0 = blockIdx.x * G8_BM, t0 = blockIdx.y * G8_BN;
    const size_t rb = kq_row_bytes(F, cols);
    const uint8_t *W = DP(const uint8_t, 2);
    const int8_t *Q = DP(const int8_t, 0);
    const float *S = DP(const float, 1);
    const int ty = tid / 16, tx = tid % 16;
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int k0 = 0; k0 < cols; k0 += G8_BK) {
        const int nk = min(G8_BK, cols - k0);           /* (a multiple of 32) */
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int rr = tid / 4, l = tid % 4, row = r0 + rr, kk = k0 + 32 * part;
            int w0 = 0, w1 = 0;
            float sc0 = 0.f, sc1 = 0.f;
            if (row < rows && 32 * part < nk) {
                const uint8_t *wr = W + (size_t)row * rb;
                if (F == KQ_IQ2R) {
                    const uint8_t *q2r = wr;
                    const uint2 pt = nm_q2_part(q2r, kk / 256, (kk % 256) / 32);
                    nm_q2_w8(pt.x, pt.y, l, nm_iq2_grid, nm_iq2_signs64, &w0, &w1);
                    sc0 = sc1 = nm_q2_db(nm_q2_dh(q2r, cols, kk / 256), pt.y);
                } else if (F == KQ_NV4) {
                    const uint32_t qw = *(const uint32_t *)(wr + (size_t)(kk / 32) * 16 + 4 * l);
                    w0 = (int)kt_e2m1x4(qw & 0x0f0f0f0fu);
                    w1 = (int)kt_e2m1x4((qw >> 4) & 0x0f0f0f0fu);
                    const float g = *(const float *)(wr + cols / 2 + cols / 16);
                    sc0 = 0.5f * g * kq_e4m3(wr[cols / 2 + 2 * (kk / 32)]);
                    sc1 = 0.5f * g * kq_e4m3(wr[cols / 2 + 2 * (kk / 32) + 1]);
                } else {
                    const uint8_t *blk = wr + (size_t)(kk / 32) * 34;
                    const uint16_t *b16 = (const uint16_t *)(blk + 2 + 8 * l);
                    w0 = (int)((uint32_t)b16[0] | ((uint32_t)b16[1] << 16));
                    w1 = (int)((uint32_t)b16[2] | ((uint32_t)b16[3] << 16));
                    sc0 = sc1 = __half2float(__ushort_as_half(*(const uint16_t *)blk));
                }
            }
            if (F == KQ_NV4) {
                ws[rr][G8_SW(rr, 8 * part + l)] = w0;
                ws[rr][G8_SW(rr, 8 * part + l + 4)] = w1;
            } else {
                ws[rr][G8_SW(rr, 8 * part + 2 * l)] = w0;
                ws[rr][G8_SW(rr, 8 * part + 2 * l + 1)] = w1;
            }
            if (l == 0) {
                wsc[rr][2 * part] = sc0;
                wsc[rr][2 * part + 1] = sc1;
            }
        }
        {
            const int tt = tid / 4, l = tid % 4, tok = t0 + tt;
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                int2 v = make_int2(0, 0), v2 = make_int2(0, 0);
                float sc = 0.f;
                if (tok < t && 32 * part < nk) {
                    const size_t o = (size_t)tok * cols + k0 + 32 * part + 8 * l;
                    v = *(const int2 *)(Q + o);
                    if (X2) {
                        v2 = *(const int2 *)(DP(const int8_t, 8) + o);
                    }
                    sc = S[(size_t)tok * (cols / 32) + (k0 + 32 * part) / 32];
                }
                xs[tt][G8_SW(tt, 8 * part + 2 * l)] = v.x;
                xs[tt][G8_SW(tt, 8 * part + 2 * l + 1)] = v.y;
                if (X2) {
                    xl[tt][G8_SW(tt, 8 * part + 2 * l)] = v2.x;
                    xl[tt][G8_SW(tt, 8 * part + 2 * l + 1)] = v2.y;
                }
                if (l == 0) {
                    xsc[tt][part] = sc;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            if (32 * part >= nk) {
                break;
            }
            /* a half (16 values) at a time: the int32 sums of each half of each row and token (the
             * two planes: 256 hi + lo), then the scales (the fragments of a whole part held 222
             * registers: one block an SM) */
            int cs[2][4][4];
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                int wv[4][4], xv[4][4], xw[X2 ? 4 : 1][4];
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int rw = ty * 4 + i, rx = tx * 4 + i;
                    const int4 a = *(const int4 *)&ws[rw][G8_SW(rw, 8 * part + 4 * h)];
                    const int4 b = *(const int4 *)&xs[rx][G8_SW(rx, 8 * part + 4 * h)];
                    wv[i][0] = a.x; wv[i][1] = a.y; wv[i][2] = a.z; wv[i][3] = a.w;
                    xv[i][0] = b.x; xv[i][1] = b.y; xv[i][2] = b.z; xv[i][3] = b.w;
                    if (X2) {
                        const int4 c = *(const int4 *)&xl[rx][G8_SW(rx, 8 * part + 4 * h)];
                        xw[i][0] = c.x; xw[i][1] = c.y; xw[i][2] = c.z; xw[i][3] = c.w;
                    }
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        int hi = 0;
                        #pragma unroll
                        for (int u = 0; u < 4; ++u) {
                            hi = rc_dp4a(wv[i][u], xv[j][u], hi);
                        }
                        if (X2) {
                            int lo = 0;
                            #pragma unroll
                            for (int u = 0; u < 4; ++u) {
                                lo = rc_dp4a(wv[i][u], xw[j][u], lo);
                            }
                            hi = 256 * hi + lo;
                        }
                        cs[h][i][j] = hi;
                    }
                }
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float a0 = wsc[ty * 4 + i][2 * part], a1 = wsc[ty * 4 + i][2 * part + 1];
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += ((float)cs[0][i][j] * a0 + (float)cs[1][i][j] * a1) * xsc[tx * 4 + j][part];
                }
            }
        }
        __syncthreads();
    }
    float *out = DP(float, 7);
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = r0 + ty * 4 + i;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int tok = t0 + tx * 4 + j;
            if (row < rows && tok < t) {
                out[(size_t)tok * rows + row] = acc[i][j];
            }
        }
    }
}

/* ---------- the tensor-core forms (sm_80 up: Ampere, Ada, Hopper, Blackwell) ----------
 * k_nm_gemm8_tc: k_nm_gemm8k's tiles (64 rows x 64 tokens, 128 columns a step, the same decode into
 * the same shared-memory words) with mma.sync on int8: warp w the rows 16 (w % 4) and the tokens 32
 * (w / 4), 4 tiles of 8 tokens; IQ2_XXS and Q8_0 one m16n8k32 a plane for each 32 (one scale), NVFP4
 * two m16n8k16 (a scale for each 16). With two planes a second mma on the low one: 256 hi + lo. The
 * int32 sums of each part times the scales, as the dp4a form (the same sums: the same bits? no:
 * the same integers, the float products in another order). */
template <int F, bool X2>
__global__ void __launch_bounds__(256) k_nm_gemm8_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ uint32_t ws[G8_BM][G8_BK / 4 + 1];
    __shared__ uint32_t xs[G8_BN][G8_BK / 4 + 1];
    __shared__ uint32_t xl[X2 ? G8_BN : 1][G8_BK / 4 + 1];
    __shared__ float wsc[G8_BM][G8_BK / 16], xsc[G8_BN][G8_BK / 32];
    const int rows = DI(4), cols = DI(5), t = DI(6), tid = threadIdx.x;
    const int r0 = blockIdx.x * G8_BM, t0 = blockIdx.y * G8_BN;
    const size_t rb = kq_row_bytes(F, cols);
    const uint8_t *W = DP(const uint8_t, 2);
    const int8_t *Q = DP(const int8_t, 0);
    const float *S = DP(const float, 1);
    const int warp = tid / 32, lane = tid % 32, g = lane >> 2, c4 = lane & 3;
    const int wm = 16 * (warp % 4), wn = 32 * (warp / 4);
    float acc[4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int k0 = 0; k0 < cols; k0 += G8_BK) {
        const int nk = min(G8_BK, cols - k0);
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            const int rr = tid / 4, l = tid % 4, row = r0 + rr, kk = k0 + 32 * part;
            uint32_t w0 = 0, w1 = 0;
            float sc0 = 0.f, sc1 = 0.f;
            if (row < rows && 32 * part < nk) {
                const uint8_t *wr = W + (size_t)row * rb;
                if (F == KQ_IQ2R) {
                    const uint8_t *q2r = wr;
                    const uint2 pt = nm_q2_part(q2r, kk / 256, (kk % 256) / 32);
                    nm_q2_w8(pt.x, pt.y, l, nm_iq2_grid, nm_iq2_signs64, (int *)&w0, (int *)&w1);
                    sc0 = sc1 = nm_q2_db(nm_q2_dh(q2r, cols, kk / 256), pt.y);
                } else if (F == KQ_NV4) {
                    const uint32_t qw = *(const uint32_t *)(wr + (size_t)(kk / 32) * 16 + 4 * l);
                    w0 = kt_e2m1x4(qw & 0x0f0f0f0fu);
                    w1 = kt_e2m1x4((qw >> 4) & 0x0f0f0f0fu);
                    const float gs = *(const float *)(wr + cols / 2 + cols / 16);
                    sc0 = 0.5f * gs * kq_e4m3(wr[cols / 2 + 2 * (kk / 32)]);
                    sc1 = 0.5f * gs * kq_e4m3(wr[cols / 2 + 2 * (kk / 32) + 1]);
                } else {
                    const uint8_t *blk = wr + (size_t)(kk / 32) * 34;
                    const uint16_t *b16 = (const uint16_t *)(blk + 2 + 8 * l);
                    w0 = (uint32_t)b16[0] | ((uint32_t)b16[1] << 16);
                    w1 = (uint32_t)b16[2] | ((uint32_t)b16[3] << 16);
                    sc0 = sc1 = __half2float(__ushort_as_half(*(const uint16_t *)blk));
                }
            }
            if (F == KQ_NV4) {
                ws[rr][8 * part + l] = w0;
                ws[rr][8 * part + l + 4] = w1;
            } else {
                ws[rr][8 * part + 2 * l] = w0;
                ws[rr][8 * part + 2 * l + 1] = w1;
            }
            if (l == 0) {
                wsc[rr][2 * part] = sc0;
                wsc[rr][2 * part + 1] = sc1;
            }
        }
        {
            const int tt = tid / 4, l = tid % 4, tok = t0 + tt;
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                int2 v = make_int2(0, 0), v2 = make_int2(0, 0);
                float sc = 0.f;
                if (tok < t && 32 * part < nk) {
                    const size_t o = (size_t)tok * cols + k0 + 32 * part + 8 * l;
                    v = *(const int2 *)(Q + o);
                    if (X2) {
                        v2 = *(const int2 *)(DP(const int8_t, 8) + o);
                    }
                    sc = S[(size_t)tok * (cols / 32) + (k0 + 32 * part) / 32];
                }
                xs[tt][8 * part + 2 * l] = (uint32_t)v.x;
                xs[tt][8 * part + 2 * l + 1] = (uint32_t)v.y;
                if (X2) {
                    xl[tt][8 * part + 2 * l] = (uint32_t)v2.x;
                    xl[tt][8 * part + 2 * l + 1] = (uint32_t)v2.y;
                }
                if (l == 0) {
                    xsc[tt][part] = sc;
                }
            }
        }
        __syncthreads();
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            if (32 * part >= nk) {
                break;
            }
            const int o = 8 * part;
            const float sa0 = wsc[wm + g][2 * part], sa1 = wsc[wm + g + 8][2 * part];
            const float sb0 = wsc[wm + g][2 * part + 1], sb1 = wsc[wm + g + 8][2 * part + 1];
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int tok = wn + 8 * nt + g;           /* the column of B this lane loads */
                const int cj = wn + 8 * nt + 2 * c4;       /* the columns of C this lane holds */
                const float x0 = xsc[cj][part], x1 = xsc[cj + 1][part];
                if (F == KQ_NV4) {
                    /* the halves: values 0-15 (words 0-3), 16-31 (words 4-7), each its scale */
                    int ch[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
                    #pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        mma16816s8(ch[hf], ws[wm + g][o + 4 * hf + c4], ws[wm + g + 8][o + 4 * hf + c4],
                                   xs[tok][o + 4 * hf + c4]);
                        if (X2) {
                            int cl[4] = {0, 0, 0, 0};
                            mma16816s8(cl, ws[wm + g][o + 4 * hf + c4], ws[wm + g + 8][o + 4 * hf + c4],
                                       xl[tok][o + 4 * hf + c4]);
                            #pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                ch[hf][q] = 256 * ch[hf][q] + cl[q];
                            }
                        }
                    }
                    acc[nt][0] += ((float)ch[0][0] * sa0 + (float)ch[1][0] * sb0) * x0;
                    acc[nt][1] += ((float)ch[0][1] * sa0 + (float)ch[1][1] * sb0) * x1;
                    acc[nt][2] += ((float)ch[0][2] * sa1 + (float)ch[1][2] * sb1) * x0;
                    acc[nt][3] += ((float)ch[0][3] * sa1 + (float)ch[1][3] * sb1) * x1;
                } else {
                    const uint32_t a[4] = {ws[wm + g][o + c4], ws[wm + g + 8][o + c4], ws[wm + g][o + 4 + c4],
                                           ws[wm + g + 8][o + 4 + c4]};
                    const uint32_t b[2] = {xs[tok][o + c4], xs[tok][o + 4 + c4]};
                    int ci[4] = {0, 0, 0, 0};
                    mma16832(ci, a, b);
                    if (X2) {
                        const uint32_t bl[2] = {xl[tok][o + c4], xl[tok][o + 4 + c4]};
                        int cl[4] = {0, 0, 0, 0};
                        mma16832(cl, a, bl);
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            ci[q] = 256 * ci[q] + cl[q];
                        }
                    }
                    acc[nt][0] += (float)ci[0] * sa0 * x0;
                    acc[nt][1] += (float)ci[1] * sa0 * x1;
                    acc[nt][2] += (float)ci[2] * sa1 * x0;
                    acc[nt][3] += (float)ci[3] * sa1 * x1;
                }
            }
        }
        __syncthreads();
    }
    float *out = DP(float, 7);
    #pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        const int cj = t0 + wn + 8 * nt + 2 * c4;
        const int ra = r0 + wm + g, rbb = ra + 8;
        if (ra < rows && cj < t) out[(size_t)cj * rows + ra] = acc[nt][0];
        if (ra < rows && cj + 1 < t) out[(size_t)(cj + 1) * rows + ra] = acc[nt][1];
        if (rbb < rows && cj < t) out[(size_t)cj * rows + rbb] = acc[nt][2];
        if (rbb < rows && cj + 1 < t) out[(size_t)(cj + 1) * rows + rbb] = acc[nt][3];
    }
}

/* k_nm_gemm8_tc8: a group of at most 8 tokens (a verify group of the drafter's block; the drafter's
 * own block): one n8 tile. Each warp its own 16 rows and its own copy of the x tile, 128 columns a
 * step, so no barrier of the block (4 warps: a block of 64 rows); the decode of k_nm_gemm8_tc (each
 * weight decoded once for the 8 tokens; the raw words of a step loaded before the decode), the same
 * mma forms. (k_nm_gemm8_tc's 64 x 64 tiles did 8 times the work; a first form with the block's
 * barriers and 180 registers ran at 19% of the DRAM rate.) */
#define G8S_BM 64
static int nm_ks_env(void)
{
    static int v = -1;
    if (v < 0) {
        const char *x = getenv("RECLANK_NM_KS");
        v = x ? atoi(x) : 0;
    }
    return v;
}
template <int F, bool X2>
__global__ void __launch_bounds__(128, 4) k_nm_gemm8_tc8(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ uint32_t wsa[4][16][G8_BK / 4 + 1];
    __shared__ uint32_t xsa[4][8][G8_BK / 4 + 1];
    __shared__ uint32_t xla[4][X2 ? 8 : 1][G8_BK / 4 + 1];
    __shared__ float wsca[4][16][G8_BK / 16], xsca[4][8][G8_BK / 32];
    const int rows = DI(4), cols = DI(5), t = DI(6), tid = threadIdx.x;
    const int warp = tid / 32, lane = tid % 32, g = lane >> 2, c4 = lane & 3;
    /* KS (gridDim.y: 1, 2 or 4) warps a row group, each a share of the columns (the narrow
     * matrices: more warps), their sums added in the order of the shares at the end */
    const int KS = gridDim.y, ks = warp % KS;
    const int r0 = blockIdx.x * (G8S_BM / KS) + 16 * (warp / KS);
    const int kc = (cols / G8_BK + KS - 1) / KS * G8_BK, kbeg = ks * kc, kend = min(cols, kbeg + kc);
    __shared__ float part_s[4][4][32];
    uint32_t (*ws)[G8_BK / 4 + 1] = wsa[warp];
    uint32_t (*xs)[G8_BK / 4 + 1] = xsa[warp];
    uint32_t (*xl)[G8_BK / 4 + 1] = xla[warp];
    float (*wsc)[G8_BK / 16] = wsca[warp];
    float (*xsc)[G8_BK / 32] = xsca[warp];
    const size_t rb = kq_row_bytes(F, cols);
    const uint8_t *W = DP(const uint8_t, 2);
    const int8_t *Q = DP(const int8_t, 0);
    const float *S = DP(const float, 1);
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    const bool live = r0 < rows;
    const float g2 = F == KQ_NV4 && live ? 0.5f * *(const float *)(W + (size_t)min(r0 + lane / 2, rows - 1) * rb + cols / 2 + cols / 16) : 0.f;
    /* 8 items a lane (16 rows x 4 parts x 4 lanes: rr = 2 it + lane / 16); the raw words of the next
     * step loaded before this step's decode and products */
    uint32_t raw[8], aux[8], nraw[8], naux[8];
    uint16_t hd[8], nhd[8];
    auto fetch = [&](int k0, uint32_t *rw, uint32_t *ax, uint16_t *h) {
        const int nk = min(G8_BK, kend - k0);
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int rr = 2 * it + lane / 16, part = (lane / 4) % 4, l = lane % 4, row = r0 + rr, kk = k0 + 32 * part;
            rw[it] = ax[it] = 0;
            h[it] = 0;
            if (live && row < rows && 32 * part < nk && k0 < kend) {
                const uint8_t *wr = W + (size_t)row * rb;
                if (F == KQ_IQ2R) {
                    const uint2 pt = nm_q2_part(wr, kk / 256, (kk % 256) / 32);
                    rw[it] = (pt.x >> (8 * l)) & 255;
                    ax[it] = pt.y;
                    h[it] = nm_q2_dh(wr, cols, kk / 256);
                } else if (F == KQ_NV4) {
                    rw[it] = *(const uint32_t *)(wr + (size_t)(kk / 32) * 16 + 4 * l);
                    ax[it] = *(const uint16_t *)(wr + cols / 2 + 2 * (kk / 32));
                } else {
                    const uint8_t *blk = wr + (size_t)(kk / 32) * 34;
                    const uint16_t *b16 = (const uint16_t *)(blk + 2 + 8 * l);
                    rw[it] = (uint32_t)b16[0] | ((uint32_t)b16[1] << 16);
                    ax[it] = (uint32_t)b16[2] | ((uint32_t)b16[3] << 16);
                    h[it] = *(const uint16_t *)blk;
                }
            }
        }
    };
    (void)nraw;
    (void)naux;
    (void)nhd;
    for (int k0 = kbeg; k0 < kend; k0 += G8_BK) {
        const int nk = min(G8_BK, kend - k0);
        fetch(k0, raw, aux, hd);
        {
            const int tt = lane / 4, l = lane % 4;
            #pragma unroll
            for (int part = 0; part < 4; ++part) {
                int2 v = make_int2(0, 0), v2 = make_int2(0, 0);
                float sc = 0.f;
                if (live && tt < t && 32 * part < nk) {
                    const size_t o = (size_t)tt * cols + k0 + 32 * part + 8 * l;
                    v = *(const int2 *)(Q + o);
                    if (X2) {
                        v2 = *(const int2 *)(DP(const int8_t, 8) + o);
                    }
                    sc = S[(size_t)tt * (cols / 32) + (k0 + 32 * part) / 32];
                }
                xs[tt][8 * part + 2 * l] = (uint32_t)v.x;
                xs[tt][8 * part + 2 * l + 1] = (uint32_t)v.y;
                if (X2) {
                    xl[tt][8 * part + 2 * l] = (uint32_t)v2.x;
                    xl[tt][8 * part + 2 * l + 1] = (uint32_t)v2.y;
                }
                if (l == 0) {
                    xsc[tt][part] = sc;
                }
            }
        }
        #pragma unroll
        for (int it = 0; it < 8; ++it) {
            const int rr = 2 * it + lane / 16, part = (lane / 4) % 4, l = lane % 4, row = r0 + rr;
            uint32_t w0 = 0, w1 = 0;
            float sc0 = 0.f, sc1 = 0.f;
            if (live && row < rows && 32 * part < nk) {
                if (F == KQ_IQ2R) {
                    const uint64_t gg = nm_iq2_grid[raw[it]], m = nm_iq2_signs64[(aux[it] >> (7 * l)) & 127];
                    w0 = __vsub4((uint32_t)gg ^ (uint32_t)m, (uint32_t)m);
                    w1 = __vsub4((uint32_t)(gg >> 32) ^ (uint32_t)(m >> 32), (uint32_t)(m >> 32));
                    sc0 = sc1 = __half2float(__ushort_as_half(hd[it])) * (float)(2 * (aux[it] >> 28) + 1) * 0.125f;
                } else if (F == KQ_NV4) {
                    w0 = kt_e2m1x4(raw[it] & 0x0f0f0f0fu);
                    w1 = kt_e2m1x4((raw[it] >> 4) & 0x0f0f0f0fu);
                    const float gr = __shfl_sync(RC_FULL_MASK, g2, 2 * rr);
                    sc0 = gr * kq_e4m3(aux[it] & 0xff);
                    sc1 = gr * kq_e4m3(aux[it] >> 8);
                } else {
                    w0 = raw[it];
                    w1 = aux[it];
                    sc0 = sc1 = __half2float(__ushort_as_half(hd[it]));
                }
            }
            if (F == KQ_NV4) {
                ws[rr][8 * part + l] = w0;
                ws[rr][8 * part + l + 4] = w1;
            } else {
                ws[rr][8 * part + 2 * l] = w0;
                ws[rr][8 * part + 2 * l + 1] = w1;
            }
            if (l == 0) {
                wsc[rr][2 * part] = sc0;
                wsc[rr][2 * part + 1] = sc1;
            }
        }
        __syncwarp();
        #pragma unroll
        for (int part = 0; part < 4; ++part) {
            if (32 * part >= nk) {
                break;
            }
            const int o = 8 * part;
            const float sa0 = wsc[g][2 * part], sa1 = wsc[g + 8][2 * part];
            const float sb0 = wsc[g][2 * part + 1], sb1 = wsc[g + 8][2 * part + 1];
            const int tok = g, cj = 2 * c4;
            const float x0 = xsc[cj][part], x1 = xsc[cj + 1][part];
            if (F == KQ_NV4) {
                int ch[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
                #pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    mma16816s8(ch[hf], ws[g][o + 4 * hf + c4], ws[g + 8][o + 4 * hf + c4], xs[tok][o + 4 * hf + c4]);
                    if (X2) {
                        int cl[4] = {0, 0, 0, 0};
                        mma16816s8(cl, ws[g][o + 4 * hf + c4], ws[g + 8][o + 4 * hf + c4], xl[tok][o + 4 * hf + c4]);
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            ch[hf][q] = 256 * ch[hf][q] + cl[q];
                        }
                    }
                }
                acc[0] += ((float)ch[0][0] * sa0 + (float)ch[1][0] * sb0) * x0;
                acc[1] += ((float)ch[0][1] * sa0 + (float)ch[1][1] * sb0) * x1;
                acc[2] += ((float)ch[0][2] * sa1 + (float)ch[1][2] * sb1) * x0;
                acc[3] += ((float)ch[0][3] * sa1 + (float)ch[1][3] * sb1) * x1;
            } else {
                const uint32_t a[4] = {ws[g][o + c4], ws[g + 8][o + c4], ws[g][o + 4 + c4], ws[g + 8][o + 4 + c4]};
                const uint32_t b[2] = {xs[tok][o + c4], xs[tok][o + 4 + c4]};
                int ci[4] = {0, 0, 0, 0};
                mma16832(ci, a, b);
                if (X2) {
                    const uint32_t bl[2] = {xl[tok][o + c4], xl[tok][o + 4 + c4]};
                    int cl[4] = {0, 0, 0, 0};
                    mma16832(cl, a, bl);
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        ci[q] = 256 * ci[q] + cl[q];
                    }
                }
                acc[0] += (float)ci[0] * sa0 * x0;
                acc[1] += (float)ci[1] * sa0 * x1;
                acc[2] += (float)ci[2] * sa1 * x0;
                acc[3] += (float)ci[3] * sa1 * x1;
            }
        }
        __syncwarp();
    }
    if (KS > 1) {
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            part_s[warp][q][lane] = acc[q];
        }
        __syncthreads();
        if (ks != 0) {
            return;
        }
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            float v = 0.f;
            for (int j = 0; j < KS; ++j) {
                v += part_s[warp + j][q][lane];
            }
            acc[q] = v;
        }
    }
    if (!live) {
        return;
    }
    float *out = DP(float, 7);
    const int cj = 2 * c4, ra = r0 + g, rbb = ra + 8;
    if (ra < rows && cj < t) out[(size_t)cj * rows + ra] = acc[0];
    if (ra < rows && cj + 1 < t) out[(size_t)(cj + 1) * rows + ra] = acc[1];
    if (rbb < rows && cj < t) out[(size_t)cj * rows + rbb] = acc[2];
    if (rbb < rows && cj + 1 < t) out[(size_t)(cj + 1) * rows + rbb] = acc[3];
}

/* k_nm_t8: KQ_NV4, IQ2_XXS or Q8_0 rows on a group of at most 8 tokens (the drafter's verify group, its block)
 * on the tensor cores. A block of 8 warps takes 128 rows (16 a warp); x (the two planes of the 8
 * tokens) staged for the block 512 columns at a time; each warp decodes its rows 128 columns a step:
 * lane l the 32 code bytes of row l / 2, parts 2 (l % 2) and 2 (l % 2) + 1 (two 16-byte loads), as
 * int8 words in the order of the values (kt_e2m1x4), rows 36 words apart (the A fragments without
 * bank conflicts, 16-byte stores), and its 4 scales from one 4-byte load. gridDim.y > 1: the columns
 * in shares, each block's sums to part (gridDim.y x t x rows), added in order by k_nm_t8_sum. */
#define NV8_BM 128
#define NV8_XK 512
#define NV8_WS 36
template <int F, bool X2>
__global__ void __launch_bounds__(256) k_nm_t8(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    __shared__ __align__(16) uint32_t xh[8][NV8_XK / 4 + 4];
    __shared__ __align__(16) uint32_t xlo[X2 ? 8 : 1][NV8_XK / 4 + 4];
    __shared__ float xsc[8][NV8_XK / 32];
    __shared__ __align__(16) uint32_t wsa[8][16][NV8_WS];
    __shared__ float wsca[8][16][8];
    const int rows = DI(4), cols = DI(5), t = DI(6), tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane >> 2, c4 = lane & 3, r0 = blockIdx.x * NV8_BM + 16 * warp;
    const int ks = gridDim.y, kc = ((cols + NV8_XK - 1) / NV8_XK + ks - 1) / ks * NV8_XK;
    const int kbeg = blockIdx.y * kc, kend = min(cols, kbeg + kc);
    const size_t rb = kq_row_bytes(F, cols);
    const uint8_t *W = DP(const uint8_t, 2);
    const int8_t *Q = DP(const int8_t, 0), *QL = X2 ? DP(const int8_t, 8) : NULL;
    const float *S = DP(const float, 1);
    uint32_t (*ws)[NV8_WS] = wsa[warp];
    float (*wsc)[8] = wsca[warp];
    const int rr = lane / 2, hh = lane % 2, row = min(r0 + rr, rows - 1);
    const uint8_t *wr = W + (size_t)row * rb;
    const float g2 = F == KQ_NV4 ? 0.5f * *(const float *)(wr + cols / 2 + cols / 16) : 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int c0 = kbeg; c0 < kend; c0 += NV8_XK) {
        __syncthreads();
        {
            /* x: 8 tokens x 512 bytes a plane, a 16-byte piece a thread */
            const int tt = tid / 32, w4 = tid % 32;
            int4 v = make_int4(0, 0, 0, 0), v2 = make_int4(0, 0, 0, 0);
            if (tt < t && c0 + 16 * w4 < kend) {
                v = *(const int4 *)(Q + (size_t)tt * cols + c0 + 16 * w4);
                if (X2) {
                    v2 = *(const int4 *)(QL + (size_t)tt * cols + c0 + 16 * w4);
                }
            }
            *(int4 *)&xh[tt][4 * w4] = v;
            if (X2) {
                *(int4 *)&xlo[tt][4 * w4] = v2;
            }
            if (tid < 8 * (NV8_XK / 32)) {
                const int t2 = tid / (NV8_XK / 32), p2 = tid % (NV8_XK / 32);
                xsc[t2][p2] = t2 < t && c0 + 32 * p2 < kend ? S[(size_t)t2 * (cols / 32) + c0 / 32 + p2] : 0.f;
            }
        }
        __syncthreads();
        for (int k0 = c0; k0 < min(c0 + NV8_XK, kend); k0 += 128) {
            /* the decode: row rr, parts 2 hh and 2 hh + 1 (32 values each), as int8 words in the order
             * of the values, the scales of each part's halves */
            const bool rl = r0 + rr < rows;
            if (F == KQ_NV4) {
                const int b0 = k0 / 32 + 2 * hh;
                const uint4 qa = *(const uint4 *)(wr + (size_t)b0 * 16), qb = *(const uint4 *)(wr + (size_t)(b0 + 1) * 16);
                const uint32_t s4 = *(const uint32_t *)(wr + cols / 2 + 2 * b0);
                const uint32_t qq[2][4] = {{qa.x, qa.y, qa.z, qa.w}, {qb.x, qb.y, qb.z, qb.w}};
                #pragma unroll
                for (int pp = 0; pp < 2; ++pp) {
                    const int prt = 2 * hh + pp;
                    uint4 lo, hi;
                    lo.x = kt_e2m1x4(qq[pp][0]); lo.y = kt_e2m1x4(qq[pp][1]); lo.z = kt_e2m1x4(qq[pp][2]); lo.w = kt_e2m1x4(qq[pp][3]);
                    hi.x = kt_e2m1x4(qq[pp][0] >> 4); hi.y = kt_e2m1x4(qq[pp][1] >> 4); hi.z = kt_e2m1x4(qq[pp][2] >> 4); hi.w = kt_e2m1x4(qq[pp][3] >> 4);
                    *(uint4 *)&ws[rr][8 * prt] = lo;
                    *(uint4 *)&ws[rr][8 * prt + 4] = hi;
                    wsc[rr][2 * prt] = rl ? g2 * kq_e4m3((s4 >> (16 * pp)) & 0xff) : 0.f;
                    wsc[rr][2 * prt + 1] = rl ? g2 * kq_e4m3((s4 >> (16 * pp + 8)) & 0xff) : 0.f;
                }
            } else if (F == KQ_IQ2R) {
                const float d = __half2float(__ushort_as_half(nm_q2_dh(wr, cols, k0 / 256))) * 0.125f;
                #pragma unroll
                for (int pp = 0; pp < 2; ++pp) {
                    const int prt = 2 * hh + pp, j = (k0 % 256) / 32 + prt;
                    const uint2 pt = nm_q2_part(wr, k0 / 256, j);
                    const uint32_t ix = pt.x, aux = pt.y;
                    uint32_t wv[8];
                    #pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        const uint64_t gg = nm_iq2_grid[(ix >> (8 * l)) & 255], m = nm_iq2_signs64[(aux >> (7 * l)) & 127];
                        wv[2 * l] = __vsub4((uint32_t)gg ^ (uint32_t)m, (uint32_t)m);
                        wv[2 * l + 1] = __vsub4((uint32_t)(gg >> 32) ^ (uint32_t)(m >> 32), (uint32_t)(m >> 32));
                    }
                    *(uint4 *)&ws[rr][8 * prt] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
                    *(uint4 *)&ws[rr][8 * prt + 4] = make_uint4(wv[4], wv[5], wv[6], wv[7]);
                    wsc[rr][2 * prt] = rl ? d * (float)(2 * (aux >> 28) + 1) : 0.f;
                }
            } else {
                #pragma unroll
                for (int pp = 0; pp < 2; ++pp) {
                    const int prt = 2 * hh + pp;
                    const uint8_t *blk = wr + (size_t)(k0 / 32 + prt) * 34;
                    const uint16_t *b16 = (const uint16_t *)(blk + 2);
                    uint32_t wv[8];
                    #pragma unroll
                    for (int l = 0; l < 8; ++l) {
                        wv[l] = (uint32_t)b16[2 * l] | ((uint32_t)b16[2 * l + 1] << 16);
                    }
                    *(uint4 *)&ws[rr][8 * prt] = make_uint4(wv[0], wv[1], wv[2], wv[3]);
                    *(uint4 *)&ws[rr][8 * prt + 4] = make_uint4(wv[4], wv[5], wv[6], wv[7]);
                    wsc[rr][2 * prt] = rl ? __half2float(__ushort_as_half(*(const uint16_t *)blk)) : 0.f;
                }
            }
            __syncwarp();
            const int xo = (k0 - c0) / 4;
            #pragma unroll
            for (int prt = 0; prt < 4; ++prt) {
                const int o = 8 * prt;
                const float sa0 = wsc[g][2 * prt], sa1 = wsc[g + 8][2 * prt];
                const float x0 = xsc[2 * c4][(k0 - c0) / 32 + prt], x1 = xsc[2 * c4 + 1][(k0 - c0) / 32 + prt];
                if (F == KQ_NV4) {
                    const float sb0 = wsc[g][2 * prt + 1], sb1 = wsc[g + 8][2 * prt + 1];
                    int ch[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
                    #pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        mma16816s8(ch[hf], ws[g][o + 4 * hf + c4], ws[g + 8][o + 4 * hf + c4], xh[g][xo + o + 4 * hf + c4]);
                        if (X2) {
                            int cl[4] = {0, 0, 0, 0};
                            mma16816s8(cl, ws[g][o + 4 * hf + c4], ws[g + 8][o + 4 * hf + c4], xlo[g][xo + o + 4 * hf + c4]);
                            #pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                ch[hf][q] = 256 * ch[hf][q] + cl[q];
                            }
                        }
                    }
                    acc[0] += ((float)ch[0][0] * sa0 + (float)ch[1][0] * sb0) * x0;
                    acc[1] += ((float)ch[0][1] * sa0 + (float)ch[1][1] * sb0) * x1;
                    acc[2] += ((float)ch[0][2] * sa1 + (float)ch[1][2] * sb1) * x0;
                    acc[3] += ((float)ch[0][3] * sa1 + (float)ch[1][3] * sb1) * x1;
                } else {
                    const uint32_t a[4] = {ws[g][o + c4], ws[g + 8][o + c4], ws[g][o + 4 + c4], ws[g + 8][o + 4 + c4]};
                    const uint32_t bb[2] = {xh[g][xo + o + c4], xh[g][xo + o + 4 + c4]};
                    int ci[4] = {0, 0, 0, 0};
                    mma16832(ci, a, bb);
                    if (X2) {
                        const uint32_t bl[2] = {xlo[g][xo + o + c4], xlo[g][xo + o + 4 + c4]};
                        int cl[4] = {0, 0, 0, 0};
                        mma16832(cl, a, bl);
                        #pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            ci[q] = 256 * ci[q] + cl[q];
                        }
                    }
                    acc[0] += (float)ci[0] * sa0 * x0;
                    acc[1] += (float)ci[1] * sa0 * x1;
                    acc[2] += (float)ci[2] * sa1 * x0;
                    acc[3] += (float)ci[3] * sa1 * x1;
                }
            }
            __syncwarp();
        }
    }
    float *out = ks > 1 ? part + (size_t)blockIdx.y * t * rows : DP(float, 7);
    const int cj = 2 * c4, ra = r0 + g, rbb = ra + 8;
    if (ra < rows && cj < t) out[(size_t)cj * rows + ra] = acc[0];
    if (ra < rows && cj + 1 < t) out[(size_t)(cj + 1) * rows + ra] = acc[1];
    if (rbb < rows && cj < t) out[(size_t)cj * rows + rbb] = acc[2];
    if (rbb < rows && cj + 1 < t) out[(size_t)(cj + 1) * rows + rbb] = acc[3];
}

/* the shares of k_nm_t8 added in their order into out (operand 7) */
__global__ void k_nm_t8_sum(const gp_rec *r, const int64_t *e, const float *part, int ks)
{
    PDL_START();
    const int64_t n = (int64_t)DI(6) * DI(4), i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = 0.f;
        for (int j = 0; j < ks; ++j) {
            v += part[(size_t)j * n + i];
        }
        DP(float, 7)[i] = v;
    }
}

/* ---------- k_nm_t8v: rows on a group of at most 8 tokens, the weights streamed ----------
 * As k_nm_t8 (a block of 8 warps, 16 rows a warp; the tensor cores; x the two planes of NM_QX), with
 * the latency of the weights hidden: each warp copies the raw bytes of its 16 rows for a step of 128
 * columns to shared memory (cp.async, coalesced; T8_ST steps in flight), and each lane decodes from
 * there just the values of its A fragments into registers (rows g, g + 8: per 32 columns 2 words a
 * row; NVFP4 2 halves of 16 for the scales). The block's column share (at most T8_XK) of x is
 * staged once. The shares (gridDim.y, nm_t8_ks) so that every matrix fills the GPU; their sums to
 * part, added in order by k_nm_t8_sum. k_nm_t8 took 40-130 us a product (the lanes' loads of their
 * own rows: 34 2-byte loads a step for Q8_0, the IQ2 grid from global memory, 1-2 blocks an SM). */
#ifndef T8_XK
#define T8_XK 512
#endif

/* the bytes of a row of a stage of S steps, padded to words = rem mod m (the lanes of a fragment in
 * other banks: NVFP4, Q8_0 4 mod 8 (and 16-byte rows), IQ2 2 mod 4) */
__host__ __device__ constexpr int t8_pad(int need, int m, int rem)
{
    return (((need + 3) / 4 - rem + m - 1) / m * m + rem) * 4;
}

template <int F, int S>
__host__ __device__ constexpr int t8_rb()
{
    return F == KQ_NV4 ? t8_pad(72 * S, 8, 4) : F == KQ_IQ2R ? t8_pad(32 * S + 4 * ((S + 1) / 2), 4, 2)
                                                             : t8_pad(136 * S, 8, 4);
}

/* XK: the most columns of a share (x staged once: NVFP4 2048, a share of the long rows' in_proj
 * half its columns; the others 512, two blocks an SM) */
template <int F>
__host__ __device__ constexpr int t8_xk()
{
    return F == KQ_NV4 ? 2048 : T8_XK;
}

template <int F, bool X2, int S, int ST>
__host__ __device__ constexpr size_t t8_smem()
{
    return (size_t)(X2 ? 2 : 1) * 8 * (t8_xk<F>() + 16) + 8 * (t8_xk<F>() / 32) * 4 + (F == KQ_IQ2R ? 256 * 8 : 0) +
           (size_t)8 * ST * 16 * t8_rb<F, S>();
}

/* the shares of the columns of a product (rows x cols on at most 8 tokens) and the stages of 128 S
 * columns of each (spb): at most T8_XK columns a share, and about 2 blocks an SM */
static int nm_t8_ks(int64_t rows, int64_t cols, int *spb, int S = 1, int xk = T8_XK)
{
    const int nst = (int)(cols / (128 * S)), bm = (int)((rows + 127) / 128);
    int ks = (nst + xk / (128 * S) - 1) / (xk / (128 * S));
    const int want = (2 * nv4_sms() + bm - 1) / bm;
    ks = ks > want ? ks : (want < nst ? want : nst);
    *spb = (nst + ks - 1) / ks;
    return (nst + *spb - 1) / *spb;
}

__device__ __forceinline__ void nm_cp(void *smem, const void *gmem, int bytes)
{
#if RC_SM80
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    if (bytes == 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(sa), "l"(gmem));
    } else if (bytes == 8) {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(sa), "l"(gmem));
    } else {
        asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" :: "r"(sa), "l"(gmem));
    }
#else
    memcpy(smem, gmem, bytes);
#endif
}

template <int N>
__device__ __forceinline__ void nm_cp_wait()
{
#if RC_SM80
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#endif
}

/* the end of a group of nm_cp (nothing below sm_80, where nm_cp is a plain
 * copy: a bare commit_group broke the real sm_61 build) */
__device__ __forceinline__ void nm_cp_commit()
{
#if RC_SM80
    asm volatile("cp.async.commit_group;\n" ::);
#endif
}

/* E2M1 codes to int8 (kt_e2m1x4's values; the signs as (v ^ m) + (m & 1): no __vsub4) */
__device__ __forceinline__ uint32_t nm_e2m1x4(uint32_t q)
{
    const uint32_t c = q & 0x0f0f0f0fu, i = c & 0x07070707u, t = i | (i >> 4);
    const uint32_t sel = (t & 0xffu) | ((t >> 8) & 0xff00u);
    const uint32_t mag = __byte_perm(0x03020100u, 0x0c080604u, sel);
    const uint32_t b = (c >> 3) & 0x01010101u;
    return (mag ^ (b * 0xffu)) + b;
}

/* the int8x4 word of half h of grid index l of an IQ2 part (ix, aux) */
__device__ __forceinline__ uint32_t nm_q2_half(uint32_t ix, uint32_t aux, int l, int h, const uint64_t *grid)
{
    const uint32_t s7 = (aux >> (7 * l)) & 127, s8 = s7 | ((__popc(s7) & 1) << 7);
    const uint32_t b = (((s8 >> (4 * h)) & 15) * 0x204081u) & 0x01010101u;
    const uint64_t gg = grid[(ix >> (8 * l)) & 255];
    return ((uint32_t)(h ? gg >> 32 : gg) ^ (b * 255u)) + b;
}

/* two E4M3 bytes (the low 16 bits of b) as floats, 2^-8 times their values (kq_e4m3s: the bits in a
 * float16; the 256 goes to the row's scale, so the products are those of kq_e4m3) */
__device__ __forceinline__ float2 nm_e4m3x2(uint32_t b)
{
    const uint32_t h = ((b & 0x7fu) << 7) | ((b & 0x80u) << 8) | ((b & 0x7f00u) << 15) | ((b & 0x8000u) << 16);
    return __half22float2(*(const __half2 *)&h);
}

/* a Q8_0 word at byte off (2-aligned) of a row of the stage */
__device__ __forceinline__ uint32_t nm_q8w(const uint8_t *row, int off)
{
    if (off & 2) {
        return __byte_perm(*(const uint32_t *)(row + off - 2), *(const uint32_t *)(row + off + 2), 0x5432);
    }
    return *(const uint32_t *)(row + off);
}

/* S: the steps of 128 columns a stage (a copy of 128 S columns of each of the warp's rows: longer
 * runs of each row for the memory), ST: the stages in flight */
template <int F, bool X2, int S, int ST>
__global__ void __launch_bounds__(256) k_nm_t8v(const gp_rec *r, const int64_t *e, float *part, int spb)
{
    extern __shared__ __align__(16) uint8_t t8s[];
    constexpr int XK = t8_xk<F>(), RB = t8_rb<F, S>(), XW = XK / 4 + 4;
    uint32_t *xh = (uint32_t *)t8s, *xl = xh + 8 * XW;
    float *xsc = (float *)(t8s + (size_t)(X2 ? 2 : 1) * 8 * (XK + 16));
    uint64_t *sg = (uint64_t *)(xsc + 8 * (XK / 32));
    uint8_t *raw = (uint8_t *)(sg + (F == KQ_IQ2R ? 256 : 0));
    PDL_START();
    const int rows = DI(4), cols = DI(5), t = DI(6), tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane >> 2, c4 = lane & 3, r0 = blockIdx.x * 128 + 16 * warp;
    const int kbeg = blockIdx.y * spb * 128 * S, kend = min(cols, kbeg + spb * 128 * S), ns = (kend - kbeg) / (128 * S);
    const size_t rb = kq_row_bytes(F, cols);
    const uint8_t *W = DP(const uint8_t, 2);
    uint8_t *mine = raw + (size_t)warp * ST * 16 * RB;
    /* the copy of stage s of the warp's rows (128 S columns) to buffer s % ST */
    auto issue = [&](int s) {
        const int k0 = kbeg + 128 * S * s;
        uint8_t *st = mine + (size_t)(s % ST) * 16 * RB;
        constexpr int ND = F == KQ_IQ2R ? (S + 1) / 2 : 0;
        constexpr int PER = F == KQ_NV4 ? 4 * S + 1 : F == KQ_IQ2R ? 4 * S + ND : 17 * S;
        for (int i = lane; i < 16 * PER; i += 32) {
            const int rr = i / PER, j = i % PER;
            const uint8_t *wr = W + (size_t)min(r0 + rr, rows - 1) * rb;
            uint8_t *d = st + rr * RB;
            if (F == KQ_NV4) {
                if (j < 4 * S) {
                    nm_cp(d + 16 * j, wr + k0 / 2 + 16 * j, 16);
                } else {
                    for (int u = 0; u < S; ++u) {
                        nm_cp(d + 64 * S + 8 * u, wr + cols / 2 + k0 / 16 + 8 * u, 8);
                    }
                }
            } else if (F == KQ_IQ2R) {
                if (j < 4 * S) {
                    nm_cp(d + 8 * j, wr + k0 / 4 + 8 * j, 8);
                } else {
                    nm_cp(d + 32 * S + 4 * (j - 4 * S), wr + ((cols / 4 + 2 * (k0 / 256 + (j - 4 * S))) & ~3), 4);
                }
            } else {
                nm_cp(d + 8 * j, wr + (size_t)(k0 / 32) * 34 + 8 * j, 8);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < ns) {
            issue(s);
        }
        nm_cp_commit();
    }
    /* x: the share's columns of the 8 tokens, once */
    {
        const int8_t *Q = DP(const int8_t, 0), *QL = X2 ? DP(const int8_t, 8) : NULL;
        const float *Sx = DP(const float, 1);
        const int kc = kend - kbeg, n16 = kc / 16;
        for (int i = tid; i < 8 * n16; i += 256) {
            const int tt = i / n16, c = i % n16;
            int4 v = make_int4(0, 0, 0, 0), v2 = v;
            if (tt < t) {
                v = *(const int4 *)(Q + (size_t)tt * cols + kbeg + 16 * c);
                if (X2) {
                    v2 = *(const int4 *)(QL + (size_t)tt * cols + kbeg + 16 * c);
                }
            }
            *(int4 *)&xh[tt * XW + 4 * c] = v;
            if (X2) {
                *(int4 *)&xl[tt * XW + 4 * c] = v2;
            }
        }
        for (int i = tid; i < 8 * (kc / 32); i += 256) {
            const int tt = i / (kc / 32), p = i % (kc / 32);
            xsc[tt * (XK / 32) + p] = tt < t ? Sx[(size_t)tt * (cols / 32) + kbeg / 32 + p] : 0.f;
        }
        if (F == KQ_IQ2R) {
            for (int i = tid; i < 256; i += 256) {
                sg[i] = nm_iq2_grid[i];
            }
        }
    }
    __syncthreads();
    const int ra = min(r0 + g, rows - 1), rbr = min(r0 + g + 8, rows - 1);
    float g2a = 0.f, g2b = 0.f;
    if (F == KQ_NV4) {
        g2a = 128.f * *(const float *)(W + (size_t)ra * rb + cols / 2 + cols / 16);    /* (0.5 x 256) */
        g2b = 128.f * *(const float *)(W + (size_t)rbr * rb + cols / 2 + cols / 16);
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < ns; ++s) {
        if (s + ST - 1 < ns) {
            issue(s + ST - 1);
        }
        nm_cp_commit();
        nm_cp_wait<ST - 1>();
        __syncwarp();
        const uint8_t *st = mine + (size_t)(s % ST) * 16 * RB, *sa_ = st + g * RB, *sb_ = st + (g + 8) * RB;
#pragma unroll
        for (int u = 0; u < S; ++u) {
            const int k0 = kbeg + 128 * (S * s + u), xo = (k0 - kbeg) / 4;
            float dA = 0.f, dB = 0.f;
            uint2 scA = make_uint2(0, 0), scB = scA;
            if (F == KQ_IQ2R) {
                const int sh = ((cols / 4 + 2 * (k0 / 256)) & 2) * 8, dj = S == 1 ? 0 : u / 2;
                dA = __half2float(__ushort_as_half((uint16_t)(*(const uint32_t *)(sa_ + 32 * S + 4 * dj) >> sh))) * 0.125f;
                dB = __half2float(__ushort_as_half((uint16_t)(*(const uint32_t *)(sb_ + 32 * S + 4 * dj) >> sh))) * 0.125f;
            } else if (F == KQ_NV4) {
                scA = *(const uint2 *)(sa_ + 64 * S + 8 * u);
                scB = *(const uint2 *)(sb_ + 64 * S + 8 * u);
            }
#pragma unroll
            for (int p = 0; p < 4; ++p) {
                const int o = 8 * p;
                const float x0 = xsc[(2 * c4) * (XK / 32) + (k0 - kbeg) / 32 + p];
                const float x1 = xsc[(2 * c4 + 1) * (XK / 32) + (k0 - kbeg) / 32 + p];
                if (F == KQ_NV4) {
                    const uint32_t wa = *(const uint32_t *)(sa_ + 64 * u + 16 * p + 4 * c4);
                    const uint32_t wb = *(const uint32_t *)(sb_ + 64 * u + 16 * p + 4 * c4);
                    const uint32_t sA = p < 2 ? scA.x >> (16 * p) : scA.y >> (16 * (p - 2));
                    const uint32_t sB = p < 2 ? scB.x >> (16 * p) : scB.y >> (16 * (p - 2));
                    int ch[2][4] = {{0, 0, 0, 0}, {0, 0, 0, 0}};
#pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        const uint32_t a0 = nm_e2m1x4(hf ? wa >> 4 : wa), a1 = nm_e2m1x4(hf ? wb >> 4 : wb);
                        mma16816s8(ch[hf], a0, a1, xh[g * XW + xo + o + 4 * hf + c4]);
                        if (X2) {
                            int cl[4] = {0, 0, 0, 0};
                            mma16816s8(cl, a0, a1, xl[g * XW + xo + o + 4 * hf + c4]);
#pragma unroll
                            for (int q = 0; q < 4; ++q) {
                                ch[hf][q] = 256 * ch[hf][q] + cl[q];
                            }
                        }
                    }
                    const float2 fA = nm_e4m3x2(sA), fB = nm_e4m3x2(sB);
                    const float sa0 = g2a * fA.x, sb0 = g2a * fA.y, sa1 = g2b * fB.x, sb1 = g2b * fB.y;
                    acc[0] += ((float)ch[0][0] * sa0 + (float)ch[1][0] * sb0) * x0;
                    acc[1] += ((float)ch[0][1] * sa0 + (float)ch[1][1] * sb0) * x1;
                    acc[2] += ((float)ch[0][2] * sa1 + (float)ch[1][2] * sb1) * x0;
                    acc[3] += ((float)ch[0][3] * sa1 + (float)ch[1][3] * sb1) * x1;
                } else {
                    uint32_t a[4];
                    float sa0, sa1;
                    if (F == KQ_IQ2R) {
                        const uint2 pa = *(const uint2 *)(sa_ + 32 * u + 8 * p), pb = *(const uint2 *)(sb_ + 32 * u + 8 * p);
                        const int l0 = c4 >> 1, h = c4 & 1;
                        a[0] = nm_q2_half(pa.x, pa.y, l0, h, sg);
                        a[1] = nm_q2_half(pb.x, pb.y, l0, h, sg);
                        a[2] = nm_q2_half(pa.x, pa.y, l0 + 2, h, sg);
                        a[3] = nm_q2_half(pb.x, pb.y, l0 + 2, h, sg);
                        sa0 = dA * (float)(2 * (pa.y >> 28) + 1);
                        sa1 = dB * (float)(2 * (pb.y >> 28) + 1);
                    } else {
                        const int b0 = 136 * u + 34 * p;
                        a[0] = nm_q8w(sa_, b0 + 2 + 4 * c4);
                        a[1] = nm_q8w(sb_, b0 + 2 + 4 * c4);
                        a[2] = nm_q8w(sa_, b0 + 18 + 4 * c4);
                        a[3] = nm_q8w(sb_, b0 + 18 + 4 * c4);
                        sa0 = __half2float(__ushort_as_half(*(const uint16_t *)(sa_ + b0)));
                        sa1 = __half2float(__ushort_as_half(*(const uint16_t *)(sb_ + b0)));
                    }
                    const uint32_t bb[2] = {xh[g * XW + xo + o + c4], xh[g * XW + xo + o + 4 + c4]};
                    int ci[4] = {0, 0, 0, 0};
                    mma16832(ci, a, bb);
                    if (X2) {
                        const uint32_t bl[2] = {xl[g * XW + xo + o + c4], xl[g * XW + xo + o + 4 + c4]};
                        int cl[4] = {0, 0, 0, 0};
                        mma16832(cl, a, bl);
#pragma unroll
                        for (int q = 0; q < 4; ++q) {
                            ci[q] = 256 * ci[q] + cl[q];
                        }
                    }
                    acc[0] += (float)ci[0] * sa0 * x0;
                    acc[1] += (float)ci[1] * sa0 * x1;
                    acc[2] += (float)ci[2] * sa1 * x0;
                    acc[3] += (float)ci[3] * sa1 * x1;
                }
            }
        }
        __syncwarp();
    }
    float *out = gridDim.y > 1 ? part + (size_t)blockIdx.y * t * rows : DP(float, 7);
    const int cj = 2 * c4, ya = r0 + g, yb = ya + 8;
    if (ya < rows && cj < t) out[(size_t)cj * rows + ya] = acc[0];
    if (ya < rows && cj + 1 < t) out[(size_t)(cj + 1) * rows + ya] = acc[1];
    if (yb < rows && cj < t) out[(size_t)cj * rows + yb] = acc[2];
    if (yb < rows && cj + 1 < t) out[(size_t)(cj + 1) * rows + yb] = acc[3];
}

/* the form of the products of 2-8 rows: k_nm_t8v unless RECLANK_NM_T8V=0 (k_nm_t8, a test) */
static int nm_t8v_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_NM_T8V");
        on = !(v && v[0] == '0');
    }
    return on;
}

template <int F, bool X2, int S, int ST>
static void nm_t8v_go(dim3 gv, cudaStream_t s, const gp_rec *dr, const int64_t *denv, float *part, int spb)
{
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(k_nm_t8v<F, X2, S, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)t8_smem<F, X2, S, ST>());
        attr = true;
    }
    k_nm_t8v<F, X2, S, ST><<<gv, 256, t8_smem<F, X2, S, ST>(), s>>>(dr, denv, part, spb);
}

/* the form of k_nm_t8v: RECLANK_NM_T8F "S ST" (a test), else nm_t8_form's */
static int nm_t8_form(int type)
{
    static int f = -1;
    if (f < 0) {
        const char *v = getenv("RECLANK_NM_T8F");
        f = v ? atoi(v) : 0;
    }
    return f ? f : 22;
}

/* GP_NM_GEMM8 of rows x cols on t <= 8 tokens by k_nm_t8v: false when the shares do not fit part */
template <int F, bool X2>
static bool nm_t8v_launch(int64_t rows, int64_t cols, int64_t t, cudaStream_t s, const gp_rec *dr,
                          const int64_t *denv, float *part, size_t part_n)
{
    const int f = nm_t8_form(F), S = f / 10;
    if (cols % (128 * S) != 0) {
        return false;
    }
    int spb = 0;
    const int ks = nm_t8_ks(rows, cols, &spb, S, t8_xk<F>());
    if (ks > 1 && (size_t)ks * t * rows > part_n) {
        return false;
    }
    dim3 gv((unsigned)((rows + 127) / 128), (unsigned)ks);
    switch (f) {
    case 12: nm_t8v_go<F, X2, 1, 2>(gv, s, dr, denv, part, spb); break;
    case 14: nm_t8v_go<F, X2, 1, 4>(gv, s, dr, denv, part, spb); break;
    case 22: nm_t8v_go<F, X2, 2, 2>(gv, s, dr, denv, part, spb); break;
    case 23: nm_t8v_go<F, X2, 2, 3>(gv, s, dr, denv, part, spb); break;
    case 42: nm_t8v_go<F, X2, 4, 2>(gv, s, dr, denv, part, spb); break;
    default: nm_t8v_go<F, X2, 1, 3>(gv, s, dr, denv, part, spb); break;
    }
    if (ks > 1) {
        k_nm_t8_sum<<<(unsigned)((t * rows + 255) / 256), 256, 0, s>>>(dr, denv, part, ks);
    }
    return true;
}

/* ---------- k_nm_t8d: NVFP4, IQ2_XXS or Q8_0 rows on a group of at most 8 tokens with dp4a (no tensor cores) ----------
 * The products of a verify group on Pascal (k_nm_gemm8k's tiles of 64 tokens did 8: 92 of the 103 ms
 * of a group). As k_nm_q2_lin: warps over groups of R rows, a lane a block of 32 values; each lane
 * decodes its block of each row once into 8 int8 words (the scales: Q8_0 one, NVFP4 one a half)
 * and adds their products with the two planes of x of every token (dp4a, the int32 sums of a block,
 * then its scales). x from global memory (a group's x: 8 rows of 2 planes). The sums of the lanes in
 * a transposed butterfly (nm_wsum). IQ2_XXS: a lane a part (KQ_IQ2R), the grid in shared memory; its
 * sums those of k_nm_q2_lin (the int32 sum of a part, the same order of the parts, the same tree):
 * the same floats, the part decoded once for all the tokens. */
/* the raw words of block bb of R rows from row (k_nm_t8d: NVFP4 4 words of codes and the 2 scales;
 * IQ2 the part and d; Q8_0 9 aligned words) */
template <int F, int R>
__device__ __forceinline__ void nm_t8d_fetch(const uint8_t *W, size_t rb, int row, int rows, int cols, int bb,
                                             uint32_t (&rw)[R][9])
{
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
        const uint8_t *wr = W + (size_t)min(row + rr, rows - 1) * rb;
        if (F == KQ_IQ2R) {
            const uint2 pt = __ldg((const uint2 *)(wr + 8 * (size_t)bb));
            rw[rr][0] = pt.x;
            rw[rr][1] = pt.y;
            rw[rr][2] = __ldg((const unsigned short *)(wr + cols / 4 + 2 * (size_t)(bb >> 3)));
        } else if (F == KQ_NV4) {
            const uint4 q = __ldg((const uint4 *)(wr + 16 * (size_t)bb));
            rw[rr][0] = q.x;
            rw[rr][1] = q.y;
            rw[rr][2] = q.z;
            rw[rr][3] = q.w;
            rw[rr][4] = __ldg((const unsigned short *)(wr + cols / 2 + 2 * (size_t)bb));
        } else {
            const size_t o = 34 * (size_t)bb;
#pragma unroll
            for (int k = 0; k < 9; ++k) {
                rw[rr][k] = __ldg((const uint32_t *)(wr + (o & ~(size_t)3)) + k);
            }
        }
    }
}

template <int F, int R>
__global__ void __launch_bounds__(256) k_nm_t8d(const gp_rec *r, const int64_t *e)
{
    __shared__ uint64_t sg[F == KQ_IQ2R ? 256 : 1];
    if (F == KQ_IQ2R) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            sg[i] = nm_iq2_grid[i];
        }
        __syncthreads();
    }
    PDL_START();
    const int rows = DI(4), cols = DI(5), t = DI(6), lane = threadIdx.x % 32;
    const size_t rb = kq_row_bytes(F, cols);
    const uint8_t *W = DP(const uint8_t, 2);
    const int8_t *Q = DP(const int8_t, 0), *QL = NM_OPQ(8);
    const float *XS = DP(const float, 1);
    const int nb = cols / 32, warps = gridDim.x * (blockDim.x / 32);
    for (int row = (blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32) * R; row < rows; row += warps * R) {
        float g2[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            g2[rr] = F == KQ_NV4 ? 0.5f * *(const float *)(W + (size_t)min(row + rr, rows - 1) * rb + cols / 2 + cols / 16) : 1.f;
        }
        float acc[8 * R];
#pragma unroll
        for (int i = 0; i < 8 * R; ++i) {
            acc[i] = 0.f;
        }
        /* the raw words of a lane's block of each row, the next chunk's loaded before the products of
         * this one (NVFP4: 4 words of codes, the 2 scales; IQ2: the part, d; Q8_0: 9 aligned words) */
        uint32_t rw[R][9];
        if (lane < nb) {
            nm_t8d_fetch<F, R>(W, rb, row, rows, cols, lane, rw);
        }
        for (int c0 = 0; c0 < nb; c0 += 32) {
            const int b = c0 + lane;
            if (b >= nb) {
                break;
            }
            uint32_t w[R][8];
            float sa[R], sb[R];
#pragma unroll
            for (int rr = 0; rr < R; ++rr) {
                if (F == KQ_IQ2R) {
#pragma unroll
                    for (int l = 0; l < 4; ++l) {
                        int w0, w1;
                        nm_q2_wl(rw[rr][0], rw[rr][1], l, sg, &w0, &w1);
                        w[rr][2 * l] = (uint32_t)w0;
                        w[rr][2 * l + 1] = (uint32_t)w1;
                    }
                    sa[rr] = sb[rr] = nm_q2_db((uint16_t)rw[rr][2], rw[rr][1]);
                } else if (F == KQ_NV4) {
#pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        w[rr][k] = nm_e2m1x4(rw[rr][k]);
                        w[rr][4 + k] = nm_e2m1x4(rw[rr][k] >> 4);
                    }
                    sa[rr] = g2[rr] * kq_e4m3(rw[rr][4] & 0xff);
                    sb[rr] = g2[rr] * kq_e4m3((rw[rr][4] >> 8) & 0xff);
                } else {
                    /* (the block starts 2-aligned) */
                    if (((34 * b) & 3) == 0) {
                        sa[rr] = __half2float(__ushort_as_half((uint16_t)rw[rr][0]));
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            w[rr][k] = __byte_perm(rw[rr][k], rw[rr][k + 1], 0x5432);
                        }
                    } else {
                        sa[rr] = __half2float(__ushort_as_half((uint16_t)(rw[rr][0] >> 16)));
#pragma unroll
                        for (int k = 0; k < 8; ++k) {
                            w[rr][k] = rw[rr][k + 1];
                        }
                    }
                    sb[rr] = sa[rr];
                }
            }
            if (b + 32 < nb) {
                nm_t8d_fetch<F, R>(W, rb, row, rows, cols, b + 32, rw);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                if (j >= t) {
                    break;
                }
                const int8_t *qh = Q + (size_t)j * cols + 32 * (size_t)b;
                const int4 h0 = __ldg((const int4 *)qh), h1 = __ldg((const int4 *)(qh + 16));
                int4 l0 = make_int4(0, 0, 0, 0), l1 = l0;
                if (QL != NULL) {
                    const int8_t *ql = QL + (size_t)j * cols + 32 * (size_t)b;
                    l0 = __ldg((const int4 *)ql);
                    l1 = __ldg((const int4 *)(ql + 16));
                }
                const int hx[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
                const int lx[8] = {l0.x, l0.y, l0.z, l0.w, l1.x, l1.y, l1.z, l1.w};
                const float sx = __ldg(XS + (size_t)j * nb + b);
#pragma unroll
                for (int rr = 0; rr < R; ++rr) {
                    if (F == KQ_NV4) {
                        /* (a scale a half) */
                        int h_a = 0, h_b = 0, l_a = 0, l_b = 0;
#pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            h_a = rc_dp4a((int)w[rr][i], hx[i], h_a);
                            h_b = rc_dp4a((int)w[rr][4 + i], hx[4 + i], h_b);
                            l_a = rc_dp4a((int)w[rr][i], lx[i], l_a);
                            l_b = rc_dp4a((int)w[rr][4 + i], lx[4 + i], l_b);
                        }
                        const float ca = (float)(QL != NULL ? 256 * h_a + l_a : h_a);
                        const float cb = (float)(QL != NULL ? 256 * h_b + l_b : h_b);
                        acc[j * R + rr] += (ca * sa[rr] + cb * sb[rr]) * sx;
                    } else {
                        int sh = 0, sl = 0;
#pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            sh = rc_dp4a((int)w[rr][i], hx[i], sh);
                            if (QL != NULL) {
                                sl = rc_dp4a((int)w[rr][i], lx[i], sl);
                            }
                        }
                        acc[j * R + rr] += (float)(QL != NULL ? 256 * sh + sl : sh) * sa[rr] * sx;
                    }
                }
            }
        }
        /* (value j R + rr in the lanes (j R + rr) 32 / (8 R)) */
        const float v = nm_wsum<8 * R>(acc, lane);
        const int k = lane / (32 / (8 * R)), j = k / R, rr = k % R;
        if (lane % (32 / (8 * R)) == 0 && j < t && row + rr < rows) {
            DP(float, 7)[(size_t)j * rows + row + rr] = v;
        }
    }
}

/* the rows of a warp of k_nm_t8d (RECLANK_NM_T8D; 0: k_nm_gemm8k, a test), default 2 */
static int nm_t8d_on(void)
{
    static int r = -1;
    if (r < 0) {
        const char *v = getenv("RECLANK_NM_T8D");
        r = v ? atoi(v) : 2;
    }
    return r;
}

/* The tensor-core forms of the prompt groups: a GPU with them (caps.tc, sm_80 up), unless
 * RECLANK_NM_TC=0 (the dp4a forms of Pascal). */
static int nm_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_NM_TC");
        on = !(v && v[0] == '0');
    }
    return on && gg_caps.tc;
}

/* the hot experts of a small group by expert (k_nm_hg_*), unless RECLANK_NM_HG=0 (a test) */
static int nm_hg_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_NM_HG");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* ---------- the LiLiCorr drafter (reclank/models/nemotron/lilicorr.py; the CPU's common/nemotron.c) ----------
 * GP_NM_DPREP: a block of hd threads for each head row (q heads first, then k), the norm, the NEOX rope
 * (the angle in double precision, as the CPU), q to qout, k and v to the ring.
 * GP_NM_DATTN: a block of 4 warps for each token and query head: a warp for each key (the lanes 4
 * values of the row), the scores in shared memory, the softmax with the sink, then a thread for each
 * value of the head.
 * GP_NM_DCONV: a thread for each value of each token. */
__global__ void k_nm_dprep(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const float *q = r->tag[0] == GP_T_NONE ? NULL : DP(const float, 0);
    const int nq = DI(9), nk = DI(10), hd = DI(11), ring = DI(13), half = hd / 2;
    const int64_t pos = di(r, e, 12);
    const int nh = (q != NULL ? nq : 0) + nk, j = blockIdx.x / nh, h = blockIdx.x % nh, d = threadIdx.x;
    const bool isq = q != NULL && h < nq;
    const int kh = isq ? 0 : h - (q != NULL ? nq : 0);
    const float *src = isq ? q + ((size_t)j * nq + h) * hd : DP(const float, 1) + ((size_t)j * nk + kh) * hd;
    const float *w = isq ? DP(const float, 3) : DP(const float, 4);
    const size_t slot = (size_t)((pos + j) % ring);
    float *dst = isq ? DP(float, 7) + ((size_t)j * nq + h) * hd : DP(float, 5) + (slot * nk + kh) * hd;
    __shared__ float y[512];
    const float v = src[d];
    const float ss = block_sum(v * v);
    y[d] = v * (1.f / sqrtf(ss / (float)hd + df(r, e, 15))) * w[d];
    __syncthreads();
    if (d < half) {
        const double a = (double)(pos + j) * pow((double)df(r, e, 14), -2.0 * d / hd);
        const float c = (float)cos(a), sn = (float)sin(a);
        dst[d] = y[d] * c - y[d + half] * sn;
        dst[d + half] = y[d + half] * c + y[d] * sn;
    }
    if (!isq) {
        DP(float, 6)[(slot * nk + kh) * hd + d] = DP(const float, 2)[((size_t)j * nk + kh) * hd + d];
    }
}

#define NM_DWIN 1040                /* the most keys of a query (window + block) */
__global__ void __launch_bounds__(128) k_nm_dattn(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int t = DI(5), nq = DI(6), nk = DI(7), hd = DI(8), ring = DI(10), window = DI(11);
    const int64_t pos0 = di(r, e, 9);
    const int j = blockIdx.x / nq, h = blockIdx.x % nq, kh = h / (nq / nk), lane = threadIdx.x % 32, wp = threadIdx.x / 32;
    const int64_t p = pos0 + j, lo = p - window + 1 > 0 ? p - window + 1 : 0, hi = pos0 + t - 1;
    const int n = (int)(hi - lo + 1);
    __shared__ float sc[NM_DWIN];
    __shared__ float red[2];
    const float *K = DP(const float, 1), *V = DP(const float, 2);
    const float4 qv = ((const float4 *)(DP(const float, 0) + ((size_t)j * nq + h) * hd))[lane];   /* (hd 128) */
    const float scale = 1.f / sqrtf((float)hd), sink = DP(const float, 3)[h];
    for (int i = wp; i < n; i += 4) {
        const float4 kv = ((const float4 *)(K + ((size_t)((lo + i) % ring) * nk + kh) * hd))[lane];
        float d = qv.x * kv.x + qv.y * kv.y + qv.z * kv.z + qv.w * kv.w;
        for (int o = 16; o > 0; o >>= 1) {
            d += __shfl_xor_sync(RC_FULL_MASK, d, o);
        }
        if (lane == 0) {
            sc[i] = d * scale;
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = sink;
        for (int i = 0; i < n; ++i) {
            m = fmaxf(m, sc[i]);
        }
        float den = expf(sink - m);
        for (int i = 0; i < n; ++i) {
            sc[i] = expf(sc[i] - m);
            den += sc[i];
        }
        red[0] = den;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < hd; d += blockDim.x) {
        float o = 0.f;
        for (int i = 0; i < n; ++i) {
            o += sc[i] * V[((size_t)((lo + i) % ring) * nk + kh) * hd + d];
        }
        DP(float, 4)[((size_t)j * nq + h) * hd + d] = o / red[0];
    }
}

__global__ void k_nm_dconv(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const int t = DI(4), cols = DI(5), kern = DI(6), group = DI(7), side = DI(8), ng = cols / group;
    const int64_t x = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= (int64_t)t * cols) {
        return;
    }
    const int j = (int)(x / cols), c = (int)(x % cols), g = c / group;
    const float *h = DP(const float, 0), *dyn = DP(const float, 1), *base = DP(const float, 2);
    float sum = 0.f;
    for (int tap = 0; tap < kern && tap <= j; ++tap) {
        const float w = base[((size_t)side * kern + tap) * cols + c] + dyn[(size_t)j * 2 * kern * ng + ((size_t)side * kern + tap) * ng + g];
        sum += w * h[(size_t)(j - tap) * cols + c];
    }
    DP(float, 3)[x] = sum;
}
