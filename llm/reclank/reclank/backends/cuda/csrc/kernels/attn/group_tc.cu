/* ---------- the attention of a large group on the tensor cores ----------
 * k_flash_tc: FlashAttention-2 with mma.sync m16n8k16. The queries, the
 * keys, the values, and the weights p become float16. The sums are float32.
 *
 * Block (h, tile) takes query head h and FQ2 queries. Warp w takes the 16
 * queries (w % 2) * 16 of the tile, and the 256 output values (w / 2) * 256
 * of each query. A head of 512 values has 4 warps. Each warp computes the
 * scores of its queries itself.
 *
 * For each step of FK2 keys:
 *
 * 1. S = Q K^T: the keys come to shared memory as float16.
 * 2. The online softmax on the fragments of S. The 4 lanes that hold a row
 *    find its maximum and its sum with shuffles.
 * 3. O += P V: the fragments of S are the fragments of A. The values come
 *    to shared memory with the keys as columns (Vt), so a fragment of B is
 *    two adjacent values.
 *
 * The record is GP_ATTN_QC_MT (F32H 0) or GP_ATTN_F32H (F32H 1). The kernel
 * k_flash_qc_mt takes the same records. head_dim is 256 or 512. */
#define FQ2 32
#define FK2 16

template <int F32H>
__global__ void k_flash_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __half fsm[];
    int o0 = F32H ? 5 : 7;
    int qh = DI(o0), kvh = DI(o0 + 1), hd = DI(o0 + 2), t = DI(o0 + 3);
    int window = F32H ? DI(11) : DI(13);
    int64_t pos = di(r, e, F32H ? 9 : 11), base = F32H ? 0 : di(r, e, 12);
    const float *q = DP(const float, 0);
    float *out = DP(float, F32H ? 4 : 6);
    int h = blockIdx.x, kv = h / (qh / kvh);
    int j0 = blockIdx.y * FQ2;
    int ld = hd + 8;
    __half *qs = fsm;                      /* FQ2 x ld */
    __half *ks = qs + FQ2 * ld;            /* FK2 x ld */
    __half *vt = ks + FK2 * ld;            /* hd x (FK2 + 8) */
    const int vld = FK2 + 8;
    const float *kf = NULL, *vf = NULL;
    const int16_t *kq16 = NULL, *vq16 = NULL;
    const float *ks16 = NULL, *vs16 = NULL;
    size_t rowq = (size_t)kvh * hd, frs = hd;
    if (F32H) {
        kf = DP(const float, 1) + (size_t)kv * di(r, e, 10);
        vf = DP(const float, 2) + (size_t)kv * di(r, e, 10);
        frs = kv_rs((size_t)di(r, e, 10), kvh, hd);
    } else {
        kq16 = DP(const int16_t, 1) + (size_t)kv * hd;
        vq16 = DP(const int16_t, 3) + (size_t)kv * hd;
        ks16 = DP(const float, 2) + (size_t)kv * hd / 32;
        vs16 = DP(const float, 4) + (size_t)kv * hd / 32;
    }
    for (int x = threadIdx.x; x < FQ2 * hd; x += blockDim.x) {
        int a = x / hd, d = x % hd;
        int jj = j0 + a;
        qs[a * ld + d] = __float2half_rn(jj < t ? q[((size_t)jj * qh + h) * hd + d] : 0.f);
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int qg = warp % 2, dsl = warp / 2;
    int g = lane / 4, c = lane % 4;
    int jl = min(t, j0 + FQ2) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < base) {
        first = base;
    }
    const int *lim = F32H ? NULL : DP(const int, 16);
    int64_t last = pos + qc_tile_last(lim, j0, jl);
    int64_t p0 = pos + j0 + qg * 16 + g, p1 = p0 + 8;   /* the positions of rows g, g + 8 */
    bool live0 = j0 + qg * 16 + g < t, live1 = j0 + qg * 16 + g + 8 < t;
    int64_t e0 = live0 ? qc_last(lim, pos, j0 + qg * 16 + g) : p0;
    int64_t e1 = live1 ? qc_last(lim, pos, j0 + qg * 16 + g + 8) : p1;
    float o[32][4];
    for (int i = 0; i < 32; ++i) {
        o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int64_t k0 = first; k0 <= last; k0 += FK2) {
        int kn = (int)min((int64_t)FK2, last - k0 + 1);
        __syncthreads();
        for (int x = threadIdx.x; x < FK2 * hd; x += blockDim.x) {
            int a = x / hd, d = x % hd;
            float kvk = 0.f, kvv = 0.f;
            if (a < kn) {
                if (F32H) {
                    kvk = kf[(size_t)(k0 + a) * frs + d];
                    kvv = vf[(size_t)(k0 + a) * frs + d];
                } else {
                    size_t o2 = (size_t)(k0 - base + a) * rowq + d;
                    kvk = (float)kq16[o2] * ks16[o2 / 32];
                    kvv = (float)vq16[o2] * vs16[o2 / 32];
                }
            }
            ks[a * ld + d] = __float2half_rn(kvk);
            vt[d * vld + a] = __float2half_rn(kvv);
        }
        __syncthreads();
        /* S: 16 queries by FK2 keys = 2 tiles of 8 keys. */
        float sc[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        const __half *qa = qs + (qg * 16) * ld;
        for (int kk = 0; kk < hd; kk += 16) {
            uint32_t a[4];
            a[0] = *(const uint32_t *)&qa[g * ld + kk + 2 * c];
            a[1] = *(const uint32_t *)&qa[(g + 8) * ld + kk + 2 * c];
            a[2] = *(const uint32_t *)&qa[g * ld + kk + 2 * c + 8];
            a[3] = *(const uint32_t *)&qa[(g + 8) * ld + kk + 2 * c + 8];
            #pragma unroll
            for (int nt = 0; nt < 2; ++nt) {
                uint32_t b[2];
                b[0] = *(const uint32_t *)&ks[(nt * 8 + g) * ld + kk + 2 * c];
                b[1] = *(const uint32_t *)&ks[(nt * 8 + g) * ld + kk + 2 * c + 8];
                mma16816(sc[nt], a, b);
            }
        }
        /* The mask and the online softmax. sc[nt][0..1] are row g, keys
         * nt * 8 + 2c and + 1; sc[nt][2..3] are row g + 8. */
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
            __half2 h01 = __floats2half2_rn(e0, e1), h23 = __floats2half2_rn(e2, e3);
            pa[nt * 2 + 0] = *(uint32_t *)&h01;
            pa[nt * 2 + 1] = *(uint32_t *)&h23;
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
        /* The A fragment of P: a0 = row g, keys 2c.. (tile 0); a1 = row g+8
         * (tile 0); a2 = row g, keys 8 + 2c (tile 1); a3 = row g + 8 (tile 1). */
        uint32_t a[4] = {pa[0], pa[1], pa[2], pa[3]};
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            o[nt][0] *= (mx0 == -INFINITY ? 1.f : sc0);
            o[nt][1] *= (mx0 == -INFINITY ? 1.f : sc0);
            o[nt][2] *= (mx1 == -INFINITY ? 1.f : sc1);
            o[nt][3] *= (mx1 == -INFINITY ? 1.f : sc1);
            int d = dsl * 256 + nt * 8 + g;
            uint32_t b[2];
            b[0] = *(const uint32_t *)&vt[d * vld + 2 * c];
            b[1] = *(const uint32_t *)&vt[d * vld + 2 * c + 8];
            mma16816(o[nt], a, b);
        }
    }
    float inv0 = l0 > 0.f ? 1.0f / l0 : 0.f, inv1 = l1 > 0.f ? 1.0f / l1 : 0.f;
    int ja = j0 + qg * 16 + g;
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        int d = dsl * 256 + nt * 8 + 2 * c;
        if (live0) {
            out[((size_t)ja * qh + h) * hd + d] = o[nt][0] * inv0;
            out[((size_t)ja * qh + h) * hd + d + 1] = o[nt][1] * inv0;
        }
        if (live1) {
            out[((size_t)(ja + 8) * qh + h) * hd + d] = o[nt][2] * inv1;
            out[((size_t)(ja + 8) * qh + h) * hd + d + 1] = o[nt][3] * inv1;
        }
    }
}
