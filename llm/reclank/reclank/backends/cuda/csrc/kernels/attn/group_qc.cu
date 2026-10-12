/* ---------- the attention of a large group over the int16 cache ----------
 * k_flash_qc_tc: k_flash_f32h for the record GP_ATTN_QC_MT (the int16 cache
 * of the 26B: rows of (kv heads, head_dim) int16 values and a float scale
 * for each 32 values, from row base). The differences:
 *
 * 1. cp.async copies the int16 rows (half the bytes of float32) and their
 *    scales. A row has HD + 8 values in shared memory, so the reads of the
 *    fragments of K and of V do not hit the same bank.
 * 2. A fragment of K is 16 values of a row: they have one scale. A value of
 *    V gets the scale of its row and of its block of 32.
 *
 * The queries, the keys, the values, and the weights p become float16, and
 * the sums are float32, as in k_flash_tc. The queries and the keys of the
 * 26B have a norm, so float16 has no overflow here. */
template <int HD>
struct flash_qc_dims {
    static constexpr int NS = HD / 256;
    static constexpr int LD = HD + 8;          /* int16 values of a row of K or V */
    static constexpr int NB = HD / 32;         /* scales of a row */
    static constexpr int QLD = HD + 8;
    static constexpr size_t smem(int hb, int st)
    {
        return (size_t)st * FK3 * (2 * LD * sizeof(int16_t) + 2 * NB * sizeof(float)) +
               (NS == 1 ? 0 : (size_t)hb * 16 * QLD * sizeof(__half));
    }
};

template <int HD, int HB, int ST>
__global__ void __launch_bounds__(32 * HB * (HD / 256))
k_flash_qc_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    typedef flash_qc_dims<HD> D;
    extern __shared__ __align__(16) uint8_t fsmq[];
    int16_t *kbuf = (int16_t *)fsmq;                          /* ST x FK3 x LD */
    int16_t *vbuf = kbuf + ST * FK3 * D::LD;                  /* ST x FK3 x LD */
    float *ksb = (float *)(vbuf + ST * FK3 * D::LD);          /* ST x FK3 x NB */
    float *vsb = ksb + ST * FK3 * D::NB;                      /* ST x FK3 x NB */
    __half *qs = (__half *)(vsb + ST * FK3 * D::NB);          /* HB x 16 x QLD (HD 512) */
    int qh = DI(7), kvh = DI(8), t = DI(10);
    int window = DI(13);
    int64_t pos = di(r, e, 11), base = di(r, e, 12);
    const float *q = DP(const float, 0);
    float *out = DP(float, 6);
    int G = qh / kvh, kv = blockIdx.x, j0 = blockIdx.y * 16;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int hl = warp / D::NS, dsl = warp % D::NS;
    int h = kv * G + blockIdx.z * HB + hl;
    int g = lane / 4, c = lane % 4;
    size_t rowq = (size_t)kvh * HD;
    const int16_t *kq = DP(const int16_t, 1) + (size_t)kv * HD;
    const int16_t *vq = DP(const int16_t, 3) + (size_t)kv * HD;
    const float *ks = DP(const float, 2) + (size_t)kv * D::NB;
    const float *vs = DP(const float, 4) + (size_t)kv * D::NB;
    /* keys only (ko_tab): the values are the keys, turned back */
    const bool kon = DP(const int16_t, 1) == DP(const int16_t, 3) && DP(const float, 2) != DP(const float, 4);
    bool live0 = j0 + g < t, live1 = j0 + g + 8 < t;
    int64_t p0 = pos + j0 + g, p1 = p0 + 8;
    int jl = min(t, j0 + 16) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int *lim = DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t e0 = live0 ? qc_last(lim, pos, j0 + g) : p0;
    int64_t e1 = live1 ? qc_last(lim, pos, j0 + g + 8) : p1;
    int nsteps = (int)((last - first) / FK3) + 1;

    auto load = [&](int stage, int64_t k0) {
        int kn = (int)min((int64_t)FK3, last - k0 + 1);
        int16_t *kd = kbuf + stage * FK3 * D::LD, *vd = vbuf + stage * FK3 * D::LD;
        float *ksd = ksb + stage * FK3 * D::NB, *vsd = vsb + stage * FK3 * D::NB;
        for (int x = threadIdx.x; x < FK3 * HD / 8; x += blockDim.x) {
            int a = x / (HD / 8), d = (x % (HD / 8)) * 8;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * rowq + d;
            cp_async16_z(kd + a * D::LD + d, kq + o2, ok ? 16 : 0);
            cp_async16_z(vd + a * D::LD + d, vq + o2, ok ? 16 : 0);
        }
        for (int x = threadIdx.x; x < FK3 * D::NB / 4; x += blockDim.x) {
            int a = x / (D::NB / 4), b = (x % (D::NB / 4)) * 4;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * (rowq / 32) + b;
            cp_async16_z(ksd + a * D::NB + b, ks + o2, ok ? 16 : 0);
            cp_async16_z(vsd + a * D::NB + b, vs + o2, ok ? 16 : 0);
        }
        cp_async_commit();
    };
    load(0, first);

    /* The queries: rows g and g + 8 are the queries j0 + g and j0 + g + 8. */
    const float *q0r = q + ((size_t)(j0 + g) * qh + h) * HD;
    const float *q1r = q + ((size_t)(j0 + g + 8) * qh + h) * HD;
    uint32_t qa[D::NS == 1 ? HD / 16 : 1][4];
    if (D::NS == 1) {
        #pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            int k2 = D::NS == 1 ? kk / 16 : 0;
            int d = kk + 2 * c;
            qa[k2][0] = live0 ? pack_h2(q0r[d], q0r[d + 1]) : 0u;
            qa[k2][1] = live1 ? pack_h2(q1r[d], q1r[d + 1]) : 0u;
            qa[k2][2] = live0 ? pack_h2(q0r[d + 8], q0r[d + 9]) : 0u;
            qa[k2][3] = live1 ? pack_h2(q1r[d + 8], q1r[d + 9]) : 0u;
        }
    } else {
        for (int x = threadIdx.x; x < HB * 16 * HD; x += blockDim.x) {
            int hh = x / (16 * HD), a = (x / HD) % 16, d = x % HD;
            int jj = j0 + a;
            qs[(hh * 16 + a) * D::QLD + d] = __float2half_rn(
                jj < t ? q[((size_t)jj * qh + kv * G + blockIdx.z * HB + hh) * HD + d] : 0.f);
        }
    }
    const __half *qw = qs + hl * 16 * D::QLD;

    float o[32][4];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int s = 0; s < nsteps; ++s) {
        int64_t k0 = first + (int64_t)s * FK3;
        int kn = (int)min((int64_t)FK3, last - k0 + 1);
        int st = ST == 2 ? (s & 1) : 0;
        if (ST == 2 && s + 1 < nsteps) {
            load(st ^ 1, k0 + FK3);
            cp_async_wait1();
        } else {
            cp_async_wait0();
        }
        __syncthreads();
        const int16_t *kd = kbuf + st * FK3 * D::LD, *vd = vbuf + st * FK3 * D::LD;
        const float *ksd = ksb + st * FK3 * D::NB, *vsd = vsb + st * FK3 * D::NB;
        /* S: 16 queries by FK3 keys = 2 tiles of 8 keys. */
        float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int kk = 0; kk < HD; kk += 16) {
            uint32_t a[4];
            if (D::NS == 1) {
                int k2 = D::NS == 1 ? kk / 16 : 0;
                a[0] = qa[k2][0];
                a[1] = qa[k2][1];
                a[2] = qa[k2][2];
                a[3] = qa[k2][3];
            } else {
                a[0] = *(const uint32_t *)&qw[g * D::QLD + kk + 2 * c];
                a[1] = *(const uint32_t *)&qw[(g + 8) * D::QLD + kk + 2 * c];
                a[2] = *(const uint32_t *)&qw[g * D::QLD + kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&qw[(g + 8) * D::QLD + kk + 2 * c + 8];
            }
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                int row = nt * 8 + g;
                const int16_t *kr = kd + row * D::LD + kk + 2 * c;
                float sk = ksd[row * D::NB + kk / 32];
                short2 x0 = *(const short2 *)kr, x1 = *(const short2 *)(kr + 8);
                uint32_t b[2] = {pack_h2((float)x0.x * sk, (float)x0.y * sk),
                                 pack_h2((float)x1.x * sk, (float)x1.y * sk)};
                mma16816(sc[nt], a, b);
            }
        }
        /* The mask and the online softmax, as in k_flash_tc. */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int kk = nt * 8 + 2 * c + u;
                int64_t kp = k0 + kk;
                bool ok0 = live0 && kk < kn && kp <= e0 && (window == 0 || p0 - kp < window);
                bool ok1 = live1 && kk < kn && kp <= e1 && (window == 0 || p1 - kp < window);
                sc[nt][u] = ok0 ? sc[nt][u] : -INFINITY;
                sc[nt][2 + u] = ok1 ? sc[nt][2 + u] : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][u]);
                mx1 = fmaxf(mx1, sc[nt][2 + u]);
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(RC_FULL_MASK, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(RC_FULL_MASK, mx1, off));
        }
        float sc0 = (mx0 == -INFINITY || m0 == -INFINITY) ? (m0 == -INFINITY ? 0.f : 1.f)
                                                            : expf(m0 - mx0);
        float sc1 = (mx1 == -INFINITY || m1 == -INFINITY) ? (m1 == -INFINITY ? 0.f : 1.f)
                                                            : expf(m1 - mx1);
        float ls0 = 0.f, ls1 = 0.f;
        uint32_t pa[4];
        #pragma unroll
        for (int nt = 0; nt < 2; ++nt) {
            float e0 = sc[nt][0] == -INFINITY ? 0.f : expf(sc[nt][0] - mx0);
            float e1 = sc[nt][1] == -INFINITY ? 0.f : expf(sc[nt][1] - mx0);
            float e2 = sc[nt][2] == -INFINITY ? 0.f : expf(sc[nt][2] - mx1);
            float e3 = sc[nt][3] == -INFINITY ? 0.f : expf(sc[nt][3] - mx1);
            ls0 += e0 + e1;
            ls1 += e2 + e3;
            pa[nt * 2 + 0] = pack_h2(e0, e1);
            pa[nt * 2 + 1] = pack_h2(e2, e3);
        }
        for (int off = 1; off < 4; off <<= 1) {
            ls0 += __shfl_xor_sync(RC_FULL_MASK, ls0, off);
            ls1 += __shfl_xor_sync(RC_FULL_MASK, ls1, off);
        }
        if (mx0 != -INFINITY) {
            l0 = l0 * sc0 + ls0;
            m0 = mx0;
        }
        if (mx1 != -INFINITY) {
            l1 = l1 * sc1 + ls1;
            m1 = mx1;
        }
        float f0 = mx0 == -INFINITY ? 1.f : sc0, f1 = mx1 == -INFINITY ? 1.f : sc1;
        /* O += P V. A fragment of B: b0 = V[2c, 2c + 1][d], b1 = V[2c + 8, 2c + 9][d]. */
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            o[nt][0] *= f0;
            o[nt][1] *= f0;
            o[nt][2] *= f1;
            o[nt][3] *= f1;
            int d = dsl * 256 + nt * 8 + g;
            const int16_t *vr = vd + (2 * c) * D::LD + d;
            const float *sr = vsd + (2 * c) * D::NB + d / 32;
            uint32_t b[2];
            if (kon && ko_turned(d & ~7)) {
                int pd = ko_partner(d & ~7) + (d & 7);
                auto val = [&](int key) {
                    return ko_rot1((float)vd[key * D::LD + d] * vsd[key * D::NB + d / 32],
                                   (float)vd[key * D::LD + pd] * vsd[key * D::NB + pd / 32],
                                   k0 + key, d);
                };
                b[0] = pack_h2(val(2 * c), val(2 * c + 1));
                b[1] = pack_h2(val(2 * c + 8), val(2 * c + 9));
            } else {
                b[0] = pack_h2((float)vr[0] * sr[0], (float)vr[D::LD] * sr[D::NB]);
                b[1] = pack_h2((float)vr[8 * D::LD] * sr[8 * D::NB], (float)vr[9 * D::LD] * sr[9 * D::NB]);
            }
            mma16816(o[nt], pa, b);
        }
        __syncthreads();
        if (ST == 1 && s + 1 < nsteps) {
            load(0, k0 + FK3);
        }
    }
    float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        int d = dsl * 256 + nt * 8 + 2 * c;
        if (live0) {
            *(float2 *)&out[((size_t)(j0 + g) * qh + h) * HD + d] =
                make_float2(o[nt][0] * inv0, o[nt][1] * inv0);
        }
        if (live1) {
            *(float2 *)&out[((size_t)(j0 + g + 8) * qh + h) * HD + d] =
                make_float2(o[nt][2] * inv1, o[nt][3] * inv1);
        }
    }
}

/* k_flash_qc_h: k_flash_qc_tc for the attention of a prompt pass, with
 * the work of a block shared by more warps. A block takes HB query heads of
 * one key head and NQT tiles of 16 queries; a query head of HD values has NS
 * = HD / 256 warps for each tile, and warp dsl keeps the output values dsl
 * * 256 to + 255. The differences from k_flash_qc_tc:
 *
 * 1. The threads of the block convert the int16 keys and values of a step
 *    to float16 once (the keys by rows, the values by columns), and the
 *    warps read the fragments as two float16 values. In k_flash_qc_tc each
 *    warp converts each value that it reads. The copy of the next keys
 *    (cp.async) runs while the warps compute.
 * 2. With HD 512, each warp computes the scores over its 256 values only.
 *    The two warps then add the two parts in shared memory, the part of
 *    warp 0 first, so both have the same scores. k_flash_qc_tc computes
 *    the scores of all 512 values in both.
 * 3. Thus a warp keeps its 256 values of the queries in registers, and the
 *    queries need no shared memory.
 *
 * The float16 values are those of k_flash_qc_tc. With HD 512, the order of
 * the sums of the scores differs. The keys of a step are those of the
 * NQT tiles of the block; a tile masks the keys after its queries. */
template <int HD, int HB, int NQT, int FKS, bool KQ8 = false, bool VQ8 = false, bool TQ = false>
struct flash_h_dims {
    static constexpr int NS = HD / 256, WARPS = HB * NQT * NS;
    static constexpr int LD = HD + 8, NB = HD / 32, VLD = FKS + 8;
    /* the bytes of a row of the cache: int16, int8 (KQ8, VQ8), or TQ6 (3
     * bytes for 4 values) */
    static constexpr int RBK = TQ ? HD * 3 / 4 : (KQ8 ? HD : 2 * HD);
    static constexpr int RBV = TQ ? HD * 3 / 4 : (VQ8 ? HD : 2 * HD);
    /* the bytes of a staged row of keys or values */
    static constexpr int LDBK = TQ ? RBK + 16 : (KQ8 ? HD + 16 : 2 * (HD + 8));
    static constexpr int LDBV = TQ ? RBV + 16 : (VQ8 ? HD + 16 : 2 * (HD + 8));
    static constexpr size_t smem()
    {
        return (size_t)FKS * (LDBK + LDBV + 2 * NB * sizeof(float) + LD * sizeof(__half)) +
               (size_t)HD * VLD * sizeof(__half) +
               (NS == 2 ? (size_t)WARPS * 32 * (FKS / 2) * sizeof(float) : 0);
    }
};

/* The int16 at bit sh (0 or 16) of w as a float, exactly: the offset
 * binary value (q + 32768, the bits of q with the sign bit flipped) in the
 * mantissa of 2^23, minus 2^23 + 32768. Two full-rate operations, in place
 * of a conversion of an integer (a quarter of the rate), as i4f. */
__device__ __forceinline__ float i16f(uint32_t w, int sh)
{
    return __uint_as_float((((w >> sh) & 0xFFFFu) ^ 0x8000u) | 0x4B000000u) - 8421376.f;
}

/* The int8 at bit sh (0, 8, 16, 24) of w as a float, as i16f. */
__device__ __forceinline__ float i8f(uint32_t w, int sh)
{
    return __uint_as_float((((w >> sh) & 0xFFu) ^ 0x80u) | 0x4B000000u) - 8388736.f;
}

template <int HD, int HB, int NQT, int FKS, bool KQ8 = false, bool VQ8 = false, bool TQ = false>
__global__ void __launch_bounds__(32 * HB * NQT * (HD / 256))
k_flash_qc_h(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    typedef flash_h_dims<HD, HB, NQT, FKS, KQ8, VQ8, TQ> D;
    constexpr int NS = D::NS, LD = D::LD, NB = D::NB, VLD = D::VLD;
    constexpr int LDBK = D::LDBK, LDBV = D::LDBV;
    constexpr int RBK = D::RBK, RBV = D::RBV;          /* the bytes of a row of a head */
    extern __shared__ __align__(16) uint8_t fsm5[];
    uint8_t *kbuf = fsm5;                             /* FKS x LDBK bytes */
    uint8_t *vbuf = kbuf + FKS * LDBK;                /* FKS x LDBV bytes */
    float *ksb = (float *)(vbuf + FKS * LDBV);        /* FKS x NB */
    float *vsb = ksb + FKS * NB;                      /* FKS x NB */
    __half *kh = (__half *)(vsb + FKS * NB);          /* FKS x LD */
    __half *vt = kh + FKS * LD;                       /* HD x VLD */
    float *sx = (float *)(vt + HD * VLD);            /* NS 2: WARPS x 32 lanes x FKS / 2 */
    int qh = DI(7), kvh = DI(8), t = DI(10);
    int window = DI(13);
    int64_t pos = di(r, e, 11), base = di(r, e, 12);
    const float *q = DP(const float, 0);
    float *out = DP(float, 6);
    int G = qh / kvh, kv = blockIdx.x, j0 = blockIdx.y * 16 * NQT;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int dsl = warp % NS, qt = (warp / NS) % NQT, hl = warp / NS / NQT;
    int h = kv * G + blockIdx.z * HB + hl;
    int jq = j0 + qt * 16;                           /* the queries of this warp */
    int g = lane / 4, c = lane % 4;
    size_t rowq = (size_t)kvh * HD;
    const uint8_t *kq = DP(const uint8_t, 1) + (size_t)kv * RBK;
    const uint8_t *vq = DP(const uint8_t, 3) + (size_t)kv * RBV;
    const float *ks = DP(const float, 2) + (size_t)kv * NB;
    const float *vs = DP(const float, 4) + (size_t)kv * NB;
    /* keys only (ko_tab): the values are the staged keys, turned back */
    const bool kon = !KQ8 && !VQ8 && !TQ && DP(const uint8_t, 1) == DP(const uint8_t, 3) &&
                     DP(const float, 2) != DP(const float, 4);
    bool live0 = jq + g < t, live1 = jq + g + 8 < t;
    int64_t p0 = pos + jq + g, p1 = p0 + 8;
    int jl = min(t, j0 + 16 * NQT) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int *lim = DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t e0 = live0 ? qc_last(lim, pos, jq + g) : p0;
    int64_t e1 = live1 ? qc_last(lim, pos, jq + g + 8) : p1;
    int nsteps = (int)((last - first) / FKS) + 1;

    auto load = [&](int64_t k0) {
        int kn = (int)min((int64_t)FKS, last - k0 + 1);
        /* 16 bytes at a time: 8 int16 or 16 int8 values */
        for (int x = threadIdx.x; x < FKS * RBK / 16; x += blockDim.x) {
            int a = x / (RBK / 16), d = (x % (RBK / 16)) * 16;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * (rowq / HD) * RBK + d;
            cp_async16_z(kbuf + a * LDBK + d, kq + o2, ok ? 16 : 0);
        }
        for (int x = threadIdx.x; x < (kon ? 0 : FKS * RBV / 16); x += blockDim.x) {
            int a = x / (RBV / 16), d = (x % (RBV / 16)) * 16;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * (rowq / HD) * RBV + d;
            cp_async16_z(vbuf + a * LDBV + d, vq + o2, ok ? 16 : 0);
        }
        for (int x = threadIdx.x; x < FKS * NB / 4; x += blockDim.x) {
            int a = x / (NB / 4), b = (x % (NB / 4)) * 4;
            bool ok = a < kn;
            size_t o2 = (size_t)((ok ? k0 + a : k0) - base) * (rowq / 32) + b;
            cp_async16_z(ksb + a * NB + b, ks + o2, ok ? 16 : 0);
            cp_async16_z(vsb + a * NB + b, vs + o2, ok ? 16 : 0);
        }
        cp_async_commit();
    };
    load(first);

    /* The queries: rows g and g + 8, the values dsl * 256 to + 255. */
    const float *q0r = q + ((size_t)(jq + g) * qh + h) * HD + dsl * 256;
    const float *q1r = q + ((size_t)(jq + g + 8) * qh + h) * HD + dsl * 256;
    uint32_t qa[16][4];
    #pragma unroll
    for (int k2 = 0; k2 < 16; ++k2) {
        int d = k2 * 16 + 2 * c;
        qa[k2][0] = live0 ? pack_h2(q0r[d], q0r[d + 1]) : 0u;
        qa[k2][1] = live1 ? pack_h2(q1r[d], q1r[d + 1]) : 0u;
        qa[k2][2] = live0 ? pack_h2(q0r[d + 8], q0r[d + 9]) : 0u;
        qa[k2][3] = live1 ? pack_h2(q1r[d + 8], q1r[d + 9]) : 0u;
    }

    float o[32][4];
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    constexpr int NT = FKS / 8, KC = FKS / 16;         /* tiles of 8 keys; chunks of 16 */
    float *mine = sx + (warp * 32 + lane) * (FKS / 2);
    const float *other = sx + ((warp ^ 1) * 32 + lane) * (FKS / 2);
    for (int s = 0; s < nsteps; ++s) {
        int64_t k0 = first + (int64_t)s * FKS;
        int kn = (int)min((int64_t)FKS, last - k0 + 1);
        cp_async_wait0();
        __syncthreads();
        /* The keys as float16 rows; the values as float16 columns (two
         * adjacent keys of a column in one word). A thread converts 8
         * values of a key row (one 16-byte load and store), or 8 values of
         * two adjacent keys; adjacent threads take adjacent pairs of keys,
         * so their stores to vt fall in different banks. */
        for (int x = threadIdx.x; x < FKS * HD / 8; x += blockDim.x) {
            int a = x / (HD / 8), d = (x % (HD / 8)) * 8;
            float sk = ksb[a * NB + d / 32];
            uint4 hv;
            uint32_t *h4 = (uint32_t *)&hv;
            if (TQ) {
                float f[8];
                tq6x8(kbuf + a * LDBK, d, f);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    h4[i] = pack_h2(f[2 * i] * sk, f[2 * i + 1] * sk);
                }
            } else if (KQ8) {
                uint2 u = *(const uint2 *)(kbuf + a * LDBK + d);
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    uint32_t w = i < 2 ? u.x : u.y;
                    h4[i] = pack_h2(i8f(w, 16 * (i % 2)) * sk, i8f(w, 16 * (i % 2) + 8) * sk);
                }
            } else {
                uint4 u = *(const uint4 *)(kbuf + a * LDBK + 2 * d);
                const uint32_t *w4 = (const uint32_t *)&u;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    h4[i] = pack_h2(i16f(w4[i], 0) * sk, i16f(w4[i], 16) * sk);
                }
            }
            *(uint4 *)(kh + a * LD + d) = hv;
        }
        for (int x = threadIdx.x; x < FKS / 2 * (HD / 8); x += blockDim.x) {
            int a = (x % (FKS / 2)) * 2, d = (x / (FKS / 2)) * 8;
            float s0 = vsb[a * NB + d / 32], s1 = vsb[(a + 1) * NB + d / 32];
            if (TQ) {
                float f0[8], f1[8];
                tq6x8(vbuf + a * LDBV, d, f0);
                tq6x8(vbuf + (a + 1) * LDBV, d, f1);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(uint32_t *)(vt + (d + i) * VLD + a) = pack_h2(f0[i] * s0, f1[i] * s1);
                }
            } else if (VQ8) {
                uint2 u0 = *(const uint2 *)(vbuf + a * LDBV + d);
                uint2 u1 = *(const uint2 *)(vbuf + (a + 1) * LDBV + d);
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    *(uint32_t *)(vt + (d + i) * VLD + a) =
                        pack_h2(i8f(i < 4 ? u0.x : u0.y, 8 * (i % 4)) * s0,
                                i8f(i < 4 ? u1.x : u1.y, 8 * (i % 4)) * s1);
                }
            } else {
                const uint8_t *vb2 = kon ? kbuf : vbuf;      /* (LDBK == LDBV here) */
                uint4 u0 = *(const uint4 *)(vb2 + a * LDBV + 2 * d);
                uint4 u1 = *(const uint4 *)(vb2 + (a + 1) * LDBV + 2 * d);
                if (kon && ko_turned(d)) {
                    int pd = ko_partner(d);
                    float f0[8], f1[8], y0[8], y1[8];
                    ko_i16x8(u0, s0, f0);
                    ko_i16x8(u1, s1, f1);
                    ko_i16x8(*(const uint4 *)(vb2 + a * LDBV + 2 * pd), vsb[a * NB + pd / 32], y0);
                    ko_i16x8(*(const uint4 *)(vb2 + (a + 1) * LDBV + 2 * pd),
                             vsb[(a + 1) * NB + pd / 32], y1);
                    ko_rot8(f0, y0, k0 + a, d);
                    ko_rot8(f1, y1, k0 + a + 1, d);
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        *(uint32_t *)(vt + (d + i) * VLD + a) = pack_h2(f0[i], f1[i]);
                    }
                } else {
                    const uint32_t *w0 = (const uint32_t *)&u0, *w1 = (const uint32_t *)&u1;
                    #pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        *(uint32_t *)(vt + (d + i) * VLD + a) =
                            pack_h2(i16f(w0[i / 2], 16 * (i % 2)) * s0,
                                    i16f(w1[i / 2], 16 * (i % 2)) * s1);
                    }
                }
            }
        }
        __syncthreads();
        if (s + 1 < nsteps) {
            load(k0 + FKS);
        }
        /* S: 16 queries by FKS keys = NT tiles of 8 keys, over the values of
         * this warp. */
        float sc[NT][4];
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
        }
        #pragma unroll
        for (int k2 = 0; k2 < 16; ++k2) {
            int kk = dsl * 256 + k2 * 16 + 2 * c;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const __half *kr = kh + (nt * 8 + g) * LD + kk;
                uint32_t b[2] = {*(const uint32_t *)kr, *(const uint32_t *)(kr + 8)};
                mma16816(sc[nt], qa[k2], b);
            }
        }
        if (NS == 2) {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                *(float4 *)(mine + 4 * nt) = make_float4(sc[nt][0], sc[nt][1], sc[nt][2],
                                                         sc[nt][3]);
            }
            __syncthreads();
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                #pragma unroll
                for (int u = 0; u < 4; ++u) {
                    float ot = other[nt * 4 + u];
                    sc[nt][u] = dsl == 0 ? sc[nt][u] + ot : ot + sc[nt][u];
                }
            }
        }
        /* The mask and the online softmax, as in k_flash_tc. */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                int kk = nt * 8 + 2 * c + u;
                int64_t kp = k0 + kk;
                bool ok0 = live0 && kk < kn && kp <= e0 && (window == 0 || p0 - kp < window);
                bool ok1 = live1 && kk < kn && kp <= e1 && (window == 0 || p1 - kp < window);
                sc[nt][u] = ok0 ? sc[nt][u] : -INFINITY;
                sc[nt][2 + u] = ok1 ? sc[nt][2 + u] : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][u]);
                mx1 = fmaxf(mx1, sc[nt][2 + u]);
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(RC_FULL_MASK, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(RC_FULL_MASK, mx1, off));
        }
        float sc0 = (mx0 == -INFINITY || m0 == -INFINITY) ? (m0 == -INFINITY ? 0.f : 1.f)
                                                            : expf(m0 - mx0);
        float sc1 = (mx1 == -INFINITY || m1 == -INFINITY) ? (m1 == -INFINITY ? 0.f : 1.f)
                                                            : expf(m1 - mx1);
        float ls0 = 0.f, ls1 = 0.f;
        /* pa[kc]: the A fragment of P for keys 16 kc to 16 kc + 15 (tiles
         * 2 kc and 2 kc + 1). */
        uint32_t pa[KC][4];
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            float x0 = sc[nt][0] == -INFINITY ? 0.f : expf(sc[nt][0] - mx0);
            float x1 = sc[nt][1] == -INFINITY ? 0.f : expf(sc[nt][1] - mx0);
            float x2 = sc[nt][2] == -INFINITY ? 0.f : expf(sc[nt][2] - mx1);
            float x3 = sc[nt][3] == -INFINITY ? 0.f : expf(sc[nt][3] - mx1);
            ls0 += x0 + x1;
            ls1 += x2 + x3;
            pa[nt / 2][(nt % 2) * 2 + 0] = pack_h2(x0, x1);
            pa[nt / 2][(nt % 2) * 2 + 1] = pack_h2(x2, x3);
        }
        for (int off = 1; off < 4; off <<= 1) {
            ls0 += __shfl_xor_sync(RC_FULL_MASK, ls0, off);
            ls1 += __shfl_xor_sync(RC_FULL_MASK, ls1, off);
        }
        if (mx0 != -INFINITY) {
            l0 = l0 * sc0 + ls0;
            m0 = mx0;
        }
        if (mx1 != -INFINITY) {
            l1 = l1 * sc1 + ls1;
            m1 = mx1;
        }
        float f0 = mx0 == -INFINITY ? 1.f : sc0, f1 = mx1 == -INFINITY ? 1.f : sc1;
        /* O += P V: b0 = V[2c, 2c + 1][d], b1 = V[2c + 8, 2c + 9][d]. */
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            o[nt][0] *= f0;
            o[nt][1] *= f0;
            o[nt][2] *= f1;
            o[nt][3] *= f1;
            const __half *vr = vt + (dsl * 256 + nt * 8 + g) * VLD + 2 * c;
            #pragma unroll
            for (int kc = 0; kc < KC; ++kc) {
                uint32_t b[2] = {*(const uint32_t *)(vr + 16 * kc),
                                 *(const uint32_t *)(vr + 16 * kc + 8)};
                mma16816(o[nt], pa[kc], b);
            }
        }
    }
    float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        int d = dsl * 256 + nt * 8 + 2 * c;
        if (live0) {
            *(float2 *)&out[((size_t)(jq + g) * qh + h) * HD + d] =
                make_float2(o[nt][0] * inv0, o[nt][1] * inv0);
        }
        if (live1) {
            *(float2 *)&out[((size_t)(jq + g + 8) * qh + h) * HD + d] =
                make_float2(o[nt][2] * inv1, o[nt][3] * inv1);
        }
    }
}

/* GP_ROUTER_MT: x, scale, proj, per_expert, hidden, experts, top_k, eps,
 * hscale, val, idx, t, r, logits. The kernels do the steps of GP_ROUTER for
 * each token. First the norm of the rows into r. Then the logits, as a
 * product with proj. Last, the top experts of each token. */
__global__ void k_router_norm_mt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, hidden = DI(4);
    const float *x = DP(const float, 0) + (size_t)j * hidden;
    const float *scale = DP(const float, 1);
    float *rv = DP(float, 12) + (size_t)j * hidden;
    float ss = 0.f;
    for (int k = threadIdx.x; k < hidden; k += blockDim.x) {
        ss += x[k] * x[k];
    }
    ss = block_sum(ss);
    float inv = 1.0f / sqrtf(ss / (float)hidden + df(r, e, 7));
    float hs = df(r, e, 8);
    for (int k = threadIdx.x; k < hidden; k += blockDim.x) {
        rv[k] = x[k] * inv * scale[k] * hs;
    }
}

__device__ void router_top_warp(const float *logits, const float *per_expert, float *val,
                                int *idx, int experts, int top_k);

/* The logits of a small group: one warp for each expert and token. */
__global__ void k_router_logits_mt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ex = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int j = blockIdx.y, lane = threadIdx.x % 32;
    int hidden = DI(4), experts = DI(5);
    if (ex >= experts) {
        return;
    }
    const float *pe = DP(const float, 2) + (size_t)ex * hidden;
    const float *rv = DP(const float, 12) + (size_t)j * hidden;
    float s2 = 0.f;
    for (int k = lane; k < hidden; k += 32) {
        s2 += rv[k] * pe[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s2 += __shfl_xor_sync(RC_FULL_MASK, s2, o);
    }
    if (lane == 0) {
        DP(float, 13)[(size_t)j * experts + ex] = s2;
    }
}

/* A test of MTP for the 26B (rc_gpu_set_reuse): a value 1 + m lets token j > 0
 * of a group select only from the experts of token 0, the experts that the
 * GPU holds (operand 14, the slot of each expert, -1 for a cold expert), and
 * its own m best experts. With m = 0 the CPU runs only the cold experts of
 * token 0. The result is not the result of the model. */
__device__ int g_reuse = 0;

__global__ void k_router_top_mt(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int j = blockIdx.x, experts = DI(5), top_k = DI(6);
    const float *lg = DP(const float, 13) + (size_t)j * experts;
    /* Only a small group (an MTP verify group), not a prompt pass. */
    if (g_reuse && j > 0 && DI(11) <= 16 && experts <= 256 && top_k <= 16) {
        __shared__ float masked[256], v0[16], vj[16];
        __shared__ int sel0[16], selj[16];
        const int *slots = DP(const int, 14);
        int keep = min(g_reuse - 1, top_k);
        /* The selection of token 0, as block 0 makes it, and the own
         * selection of token j, best first. */
        router_top_warp(DP(const float, 13), DP(const float, 3), v0, sel0, experts, top_k);
        router_top_warp(lg, DP(const float, 3), vj, selj, experts, top_k);
        __syncwarp();
        for (int x = threadIdx.x; x < experts; x += 32) {
            bool ok = slots != NULL && slots[x] >= 0;
            for (int k = 0; k < top_k; ++k) {
                ok = ok || sel0[k] == x || (k < keep && selj[k] == x);
            }
            masked[x] = ok ? lg[x] : -INFINITY;
        }
        __syncwarp();
        lg = masked;
    }
    router_top_warp(lg, DP(const float, 3), DP(float, 9) + (size_t)j * top_k,
                    DP(int, 10) + (size_t)j * top_k, experts, top_k);
}
