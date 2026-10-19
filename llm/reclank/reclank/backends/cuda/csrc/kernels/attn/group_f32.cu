/* ---------- the attention of a large group over a float32 cache ----------
 * k_flash_f32h: FlashAttention-2 for the record GP_ATTN_F32H (the cache of
 * the E4B model). It gives the result of k_flash_tc<1>, but it is faster:
 *
 * 1. Block (kv, tile, z) takes key and value head kv, the 16 queries of the
 *    tile, and HB query heads of that key head. The query heads share the
 *    keys and the values in shared memory, so the block reads them one time
 *    for HB heads.
 * 2. The keys and the values come to shared memory as float32 with
 *    cp.async. With ST 2, the copy of the next step of FK3 keys runs during
 *    the compute of this step.
 * 3. The fragments become float16 in registers. The rows of K have HD + 8
 *    values and the rows of V have HD + 4, so the reads of a warp do not
 *    hit the same bank.
 * 4. For HD 256, the fragments of the queries stay in registers. For HD 512
 *    they are in shared memory as float16.
 *
 * Warp w takes query head w / NS and the 256 output values (w % NS) * 256.
 * NS is HD / 256. */
#define FK3 16

__device__ __forceinline__ void cp_async16_z(void *smem, const void *gmem, int bytes)
{
#if RC_SM80
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(sa), "l"(gmem), "r"(bytes));
#else
    if (bytes == 16) {
        *(int4 *)smem = *(const int4 *)gmem;
    } else {
        memset(smem, 0, 16);
        memcpy(smem, gmem, bytes);
    }
#endif
}

__device__ __forceinline__ uint32_t pack_h2(float a, float b)
{
    __half2 h = __floats2half2_rn(a, b);
    return *(uint32_t *)&h;
}

/* GP_ATTN_QSA of a large group on the tensor cores (int8 or TQ6 keys and
 * values). k_attn_qsa_mt runs the dot of each query head with each key in
 * float32 (1.3 s of a bf16 prompt of 8192 tokens), and each block decodes
 * the keys of its query again (two blocks for each key head). But the query
 * heads of a key head read the same keys: they are the 16 rows (12 used) of
 * an mma.m16n8k16 tile. Block (j, kv): query j, all its heads of key head kv;
 * 4 warps, each with its own keys (16 at a time from the rows of sel, or all
 * positions to pos + j), decoded to float16 times their scale in the shared
 * memory of the warp: S = Q K^T (Q2: q as two float16 planes, hi + lo, so q
 * keeps about 22 bits), a softmax that runs, then O += P V (P float16, V by
 * ldmatrix.trans). The warps join at the end in shared memory. */
__device__ __forceinline__ void ldsm_x4_trans(uint32_t *d, const void *p)
{
#if RC_SM80
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(a));
#else
    __trap();
#endif
}

#define QT_LD 264                    /* the row stride (halves) of the tiles */
#define QT_SMEM ((size_t)(2 * 16 + 4 * 16) * QT_LD * 2)
template <int Q2>
__global__ void __launch_bounds__(128) k_attn_qsa_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) __half qtsm[];
    __half (*qh)[QT_LD] = (__half (*)[QT_LD])qtsm;
    __half (*ql)[QT_LD] = (__half (*)[QT_LD])(qtsm + 16 * QT_LD);
    __shared__ float wm[4][16], wl[4][16];
    __shared__ float cbs[64];
    if (threadIdx.x < 64) {
        cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
    }
    int nq = DI(7), nk = DI(8), maxsel = DI(14), form = DI(15);
    int64_t pos = di(r, e, 11);
    int rep = nq / nk;
    int j = blockIdx.x, kv = blockIdx.y;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    __half (*ks_)[QT_LD] = (__half (*)[QT_LD])(qtsm + (2 + warp) * 16 * QT_LD);
    const float *q = DP(const float, 0) + ((size_t)j * nq + (size_t)kv * rep) * 256;
    for (int i = threadIdx.x; i < 16 * 256; i += blockDim.x) {
        int h = i / 256, d = i % 256;
        float v = h < rep ? q[(size_t)h * 256 + d] : 0.f;
        __half hi = __float2half_rn(v);
        qh[h][d] = hi;
        ql[h][d] = __float2half_rn(v - __half2float(hi));
    }
    __syncthreads();
    int cn = DP(const int32_t, 13)[j];
    int n = cn < 0 ? (int)(pos + j + 1) : cn;
    const int32_t *rows = cn < 0 ? NULL : DP(const int32_t, 12) + (size_t)j * maxsel;
    const uint8_t *kb = DP(const uint8_t, 1), *vb = DP(const uint8_t, 3);
    const float *ksc = DP(const float, 2), *vsc = DP(const float, 4);
    size_t rs = (size_t)nk * 256, off = (size_t)kv * 256 + 8 * lane;
    /* rows 16 keys of the warp from x0 (keys past n: 0) to its tile */
    auto decode = [&](const uint8_t *base, const float *sc, int x0) {
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            int x = x0 + rr;
            float f[8];
            if (x < n) {
                size_t o = (size_t)(rows ? rows[x] : x) * rs + off;
                if (form == 4) {
                    tq6x8(base, o, f, cbs);
                } else {
                    fd_i8x8(*(const uint2 *)((const int8_t *)base + o), f);
                }
                float s = sc[o / 32];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    f[u] *= s;
                }
            } else {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    f[u] = 0.f;
                }
            }
            uint4 pk = make_uint4(pack_h2(f[0], f[1]), pack_h2(f[2], f[3]), pack_h2(f[4], f[5]),
                                  pack_h2(f[6], f[7]));
            *(uint4 *)&ks_[rr][8 * lane] = pk;
        }
        __syncwarp();
    };
    float acc[32][4];
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;   /* rows g, g + 8 */
    for (int x0 = warp * 16; x0 < n; x0 += 64) {
        decode(kb, ksc, x0);
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int kk = 0; kk < 256; kk += 16) {
            uint32_t a[4], b[2][2];
            #pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                b[jn][0] = *(const uint32_t *)&ks_[jn * 8 + g][kk + 2 * c];
                b[jn][1] = *(const uint32_t *)&ks_[jn * 8 + g][kk + 2 * c + 8];
            }
            a[0] = *(const uint32_t *)&qh[g][kk + 2 * c];
            a[1] = *(const uint32_t *)&qh[g + 8][kk + 2 * c];
            a[2] = *(const uint32_t *)&qh[g][kk + 2 * c + 8];
            a[3] = *(const uint32_t *)&qh[g + 8][kk + 2 * c + 8];
            mma16816(s[0], a, b[0]);
            mma16816(s[1], a, b[1]);
            if (Q2) {
                a[0] = *(const uint32_t *)&ql[g][kk + 2 * c];
                a[1] = *(const uint32_t *)&ql[g + 8][kk + 2 * c];
                a[2] = *(const uint32_t *)&ql[g][kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&ql[g + 8][kk + 2 * c + 8];
                mma16816(s[0], a, b[0]);
                mma16816(s[1], a, b[1]);
            }
        }
        __syncwarp();
        /* the keys past n: no weight */
        #pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            int x = x0 + jn * 8 + 2 * c;
            if (x >= n) { s[jn][0] = s[jn][2] = -INFINITY; }
            if (x + 1 >= n) { s[jn][1] = s[jn][3] = -INFINITY; }
        }
        float t0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
        float t1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
        t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 1));
        t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(RC_FULL_MASK, t1, 1));
        t1 = fmaxf(t1, __shfl_xor_sync(RC_FULL_MASK, t1, 2));
        float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);     /* finite: key x0 < n */
        float a0 = expf(m0 - n0), a1 = expf(m1 - n1);
        m0 = n0;
        m1 = n1;
        float p[2][4];
        #pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            p[jn][0] = expf(s[jn][0] - n0);
            p[jn][1] = expf(s[jn][1] - n0);
            p[jn][2] = expf(s[jn][2] - n1);
            p[jn][3] = expf(s[jn][3] - n1);
        }
        l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
        l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            acc[nt][0] *= a0;
            acc[nt][1] *= a0;
            acc[nt][2] *= a1;
            acc[nt][3] *= a1;
        }
        decode(vb, vsc, x0);
        uint32_t pa[4] = {pack_h2(p[0][0], p[0][1]), pack_h2(p[0][2], p[0][3]),
                          pack_h2(p[1][0], p[1][1]), pack_h2(p[1][2], p[1][3])};
        /* B = V (16 keys by 256 values): ldmatrix.trans of the rows (lane % 8)
         * + 8 ((lane / 8) & 1), the values n0 + 8 (lane / 16) */
        #pragma unroll
        for (int nt = 0; nt < 32; nt += 2) {
            uint32_t bv[4];
            ldsm_x4_trans(bv, &ks_[(lane % 8) + 8 * ((lane / 8) & 1)][nt * 8 + 8 * (lane / 16)]);
            mma16816(acc[nt], pa, bv);
            mma16816(acc[nt + 1], pa, bv + 2);
        }
        __syncwarp();
    }
    l0 += __shfl_xor_sync(RC_FULL_MASK, l0, 1);
    l0 += __shfl_xor_sync(RC_FULL_MASK, l0, 2);
    l1 += __shfl_xor_sync(RC_FULL_MASK, l1, 1);
    l1 += __shfl_xor_sync(RC_FULL_MASK, l1, 2);
    if (c == 0) {
        wm[warp][g] = m0;
        wm[warp][g + 8] = m1;
        wl[warp][g] = l0;
        wl[warp][g + 8] = l1;
    }
    __syncthreads();
    /* the join: the weight of each warp for rows g and g + 8, then the sums
     * of the warps in turn in red (the tiles are free now) */
    float M0 = -INFINITY, M1 = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        M0 = fmaxf(M0, wm[w][g]);
        M1 = fmaxf(M1, wm[w][g + 8]);
    }
    float L0 = 0.f, L1 = 0.f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        L0 += wm[w][g] == -INFINITY ? 0.f : expf(wm[w][g] - M0) * wl[w][g];
        L1 += wm[w][g + 8] == -INFINITY ? 0.f : expf(wm[w][g + 8] - M1) * wl[w][g + 8];
    }
    float f0 = m0 == -INFINITY ? 0.f : expf(m0 - M0) / L0;
    float f1 = m1 == -INFINITY ? 0.f : expf(m1 - M1) / L1;
    float (*red)[256] = (float (*)[256])qtsm;
    for (int w = 0; w < 4; ++w) {
        if (warp == w) {
            #pragma unroll
            for (int nt = 0; nt < 32; ++nt) {
                int d = nt * 8 + 2 * c;
                if (w == 0) {
                    red[g][d] = acc[nt][0] * f0;
                    red[g][d + 1] = acc[nt][1] * f0;
                    red[g + 8][d] = acc[nt][2] * f1;
                    red[g + 8][d + 1] = acc[nt][3] * f1;
                } else {
                    red[g][d] += acc[nt][0] * f0;
                    red[g][d + 1] += acc[nt][1] * f0;
                    red[g + 8][d] += acc[nt][2] * f1;
                    red[g + 8][d + 1] += acc[nt][3] * f1;
                }
            }
        }
        __syncthreads();
    }
    float *out = DP(float, 6) + ((size_t)j * nq + (size_t)kv * rep) * 256;
    for (int i = threadIdx.x; i < rep * 256; i += blockDim.x) {
        out[i] = red[i / 256][i % 256];
    }
}

/* The decode attention of a global layer of the Gemma 4 26B (head_dim 512, 8
 * query heads for each key head: GP_ATTN_QC, GP_ATTN_Q8, GP_ATTN_V8) on the
 * tensor cores, for T = 1 to FT_TMAX queries in one record (operand 11 t:
 * an MTP verify group; query j sees n + j keys), in chunks (k_attn_join
 * with nb adds them). k_attn_fdt took 1.76 ms a layer at 100K
 * positions (about 250 GB/s of the int16 rows, 47% of a step), and a
 * verify group of 3 ran it for each query. RTX 3090, 100K rows
 * (scripts/bench_attn_fdtc.cu): one query 0.68 ms (int16; int8 0.48,
 * k16v8 0.62), a group of 3 0.82 ms (0.65, 0.72). The 26B at 100K: plain
 * decode 56 -> 87 tok/s, MTP (2 drafts) 121 tok/s, the same tokens. T 4
 * fills the two m16 tiles of T 3 (98 KB of shared memory); a larger group
 * takes records of at most FT_TMAX queries (gpu.py attn_rows_small).
 *
 * Block (kv, c), 8 warps: chunk c of the keys of key head kv (ft_len), 32
 * keys at a time (FT_TK: the tiles of the 56 blocks of the RTX 3060, 1.8
 * MB, stay in its L2 of 3 MB until O reads them again; tiles of 64, 3.5 MB,
 * read most rows from memory twice); the rows of the 8 T heads of the
 * queries (query j: rows 8 j to 8 j + 7) are the rows of the m16 tiles. The
 * rows of the cache go from memory to the registers of the fragments, with
 * the values in a free order (each load whole sectors):
 *   S = Q K^T: warp w the keys 8 (w & 3) .. + 7 (n8) over half of the 512
 *   values: 8 blocks of 32 (bt = w >> 2: blocks bt and bt + 8, then 2 .. 7
 *   or 10 .. 15); lane (g, cc) reads values 32 b + 8 cc .. + 7 of key g, the
 *   k16 of two steps (q as two float16 planes in shared memory, in the same
 *   order); the scores of the two halves to shared memory; each warp the
 *   softmax that runs, the same in all (rows 8 j + g, the sum of the halves);
 *   O^T += V^T P^T: warp w the values 64 w .. 64 w + 63; lane (g, cc) reads
 *   keys 2 cc, 2 cc + 1, 2 cc + 8, 2 cc + 9 of each k16 at values 64 w +
 *   8 g .. + 7 (8 lanes: 128 bytes of a row): value 64 w + 8 g + 2 t is row
 *   g of tile t, + 1 row g + 8; P^T the scores of the lane (n8: the 8 heads
 *   of query j); 4 tiles of 16 values for each query (16 T floats a lane).
 * A cache of keys only (ko_tab of at most 64 pairs): the turned values (0 ..
 * 63 and 256 .. 319: warps 0 and 4 in O) turn back in S, where a lane has
 * both of a pair (blocks bt and bt + 8: each warp one block of pairs), to
 * shared memory (rot), and warps 0 and 4 read them there. When they turned
 * in O (16 keys a lane, a partner row read after each key, two tables for
 * each value) the other 6 warps waited for those two at the barrier: on
 * the RTX 3060 (100 W) at 128K positions 2.31 ms, now 1.53 ms.
 * The chunks of query j depend on n + j in steps of 4096 keys only (a
 * group across a step: a second pass for the queries past it), and a tile
 * past the keys of a query changes nothing in it (all its scores -inf), so
 * a query of a group gets the numbers of a step (T 1) at its position: MTP
 * verifies with the numbers of the decode. The part of each head as
 * k_attn_part writes it (the sum of exp(score - m) v, m, l), head j * qh +
 * h, chunk stride gridDim.y. */
#define FT_TMAX 4
#define FT_KEYS 64
#define FT_TK 32
#define FT_SLD (FT_TK + 4)
/* q: 2 planes of rows of FT_QLD chunks of 16 bytes (T 1: 1088 bytes, the 8
 * rows of a load in different banks; else chunk x of a row at x ^ 4 (row &
 * 1), which takes registers for the addresses but fits T 4 in 99 KB), the
 * scores (2 tiles of 2 halves), rot (2 tiles; R: int16 keys and values) */
#define FT_QLD(T) ((T) == 1 ? 64 + 4 : 64)
#define FT_SMEM(T, R) ((size_t)(T) * 8 * FT_QLD(T) * 16 * 2 + (size_t)4 * (T) * 8 * FT_SLD * 4 + \
                       ((R) ? (size_t)2 * FT_TK * 256 : 0))

/* The keys of a chunk of nb chunks for nl keys (those of the last query):
 * from nl rounded up to 4096 keys, a multiple of FT_KEYS. */
__device__ __forceinline__ int ft_len(int nl, int nb)
{
    int nr = (nl + 4095) & ~4095;
    int len = (nr + nb - 1) / nb;
    return max(FT_KEYS, (len + FT_KEYS - 1) / FT_KEYS * FT_KEYS);
}

/* 8 values of the cache (B 16: int16, 8: int8) at o, raw */
template <int B>
__device__ __forceinline__ uint4 ft_ld(const void *p, size_t o)
{
    if (B == 16) {
        return *(const uint4 *)((const int16_t *)p + o);
    }
    uint2 u = *(const uint2 *)((const int8_t *)p + o);
    return make_uint4(u.x, u.y, 0u, 0u);
}

/* the 8 values of ft_ld times sc as 4 float16 pairs */
template <int B>
__device__ __forceinline__ void ft_h8(uint4 u, float sc, uint32_t *h)
{
    if (B == 16) {
        uint32_t w[4] = {u.x, u.y, u.z, u.w};
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            h[t] = pack_h2((float)(int16_t)(w[t] & 0xffff) * sc, (float)(int16_t)(w[t] >> 16) * sc);
        }
    } else {
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            uint32_t w = (t < 2 ? u.x : u.y) >> (16 * (t % 2));
            h[t] = pack_h2(rc_kv8_val(w & 0xff) * sc, rc_kv8_val((w >> 8) & 0xff) * sc);
        }
    }
}

/* KB, VB: the bits of the keys and of the values (GP_ATTN_QC 16, 16;
 * GP_ATTN_Q8 8, 8; GP_ATTN_V8 16, 8). The loads of a batch (8 a lane) go
 * out before their use (through attn_kv8, a branch on the form in each,
 * one waited for the other). q: FT_QLD. */
template <int T, int KB, int VB>
__global__ void __launch_bounds__(256, T == 1 ? 2 : 1) k_attn_fdtc(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int MT = (T + 1) / 2;                /* m16 tiles of the 8 T rows */
    extern __shared__ __align__(16) unsigned char ftsm[];
    constexpr int QLD = FT_QLD(T);
    uint4 (*qh)[QLD] = (uint4 (*)[QLD])ftsm;                       /* [8 T][QLD] */
    uint4 (*ql)[QLD] = qh + T * 8;
    float (*ss)[2][T * 8][FT_SLD] = (float (*)[2][T * 8][FT_SLD])(ql + T * 8);
    /* rot[buf][key][i ^ ft_rsw(key)]: block i of the turned values of a key
     * (8 values: 8 i .. of 0 .. 63, then of 256 .. 319), turned back, as
     * float16; ft_rsw(key) = 2 ((key >> 1) & 3) ^ 4 (key & 1) puts the 8 lanes
     * of a store or a load in different banks */
    uint4 (*rot)[FT_TK][16] = (uint4 (*)[FT_TK][16])(ss + 2);
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, c = blockIdx.y, nb = gridDim.y;
    int n = a.n;
    if (c * ft_len(n, nb) >= n + T - 1) {
        return;
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, cc = lane % 4;
    int qhn = a.qh;
    /* q: rows 8 j + h (query j, head kv 8 + h) */
    for (int i = threadIdx.x; i < T * 8 * 128; i += 256) {
        int row = i / 128, d = 4 * (i % 128);
        float4 v = *(const float4 *)(a.q + ((size_t)(row / 8) * qhn + kv * 8 + row % 8) * 512 + d);
        float f[4] = {v.x, v.y, v.z, v.w};
        uint32_t hi[2], lo[2];
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            __half h0 = __float2half_rn(f[2 * t]), h1 = __float2half_rn(f[2 * t + 1]);
            hi[t] = pack_h2(__half2float(h0), __half2float(h1));
            lo[t] = pack_h2(f[2 * t] - __half2float(h0), f[2 * t + 1] - __half2float(h1));
        }
        int ch = T == 1 ? d / 8 : (d / 8) ^ (4 * (row & 1));
        uint2 *ph = (uint2 *)&qh[row][ch] + (d % 8) / 4, *pl = (uint2 *)&ql[row][ch] + (d % 8) / 4;
        *ph = make_uint2(hi[0], hi[1]);
        *pl = make_uint2(lo[0], lo[1]);
    }
    __syncthreads();
    const void *kb = a.kq, *vb = a.vq;              /* operands 1, 3 (int16 or int8) */
    const bool kon = KB == 16 && kb == vb && a.vs != a.ks;   /* keys only (ko_tab) */
    /* rot: the turned pairs in blocks 0, 1, 8 and 9 (values 0 .. 63 and 256 .. 319) */
    const bool rt = KB == 16 && VB == 16 && kon && g_ko.half == 256 && g_ko.rd <= 64;
    /* query j: chunks of ft_len(n + j); a group across a step of 4096 keys
     * takes a second pass for the queries past it */
    int L0 = ft_len(n, nb), L1 = ft_len(n + T - 1, nb), buf = 0;
    for (int ps = 0; ps < (L1 != L0 ? 2 : 1); ++ps) {
        int len = ps ? L1 : L0, act = 0, jl = 0;
        #pragma unroll
        for (int j = 0; j < T; ++j) {
            if (ft_len(n + j, nb) == len) {
                act |= 1 << j;
                jl = j;
            }
        }
        int nl = n + jl, j0 = c * len, j1 = min(nl, j0 + len);
        if (j0 >= nl) {
            continue;
        }
        float acc[4][T][4];
        float m[T], l[T];
        #pragma unroll
        for (int j = 0; j < T; ++j) {
            m[j] = -INFINITY;
            l[j] = 0.f;
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                acc[mt][j][0] = acc[mt][j][1] = acc[mt][j][2] = acc[mt][j][3] = 0.f;
            }
        }
        for (int x0 = j0; x0 < j1; x0 += FT_TK, buf ^= 1) {
            /* S of keys x0 + 8 (warp & 3) .. + 7 (key kx of the lane; the keys
             * past j1 read key j1 - 1, their scores -inf) over the 8 blocks
             * of 32 values of half bt: blocks bt, bt + 8 (a turned block and
             * its partner, in the same lane), then 2 .. 7 (bt 0) or 10 .. 15 */
            int bt = warp >> 2, kr = 8 * (warp & 3) + g, kx = min(x0 + kr, j1 - 1);
            size_t ok = (size_t)kx * a.rstride + (size_t)kv * a.hstride;
            float s[MT][4];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                s[mt][0] = s[mt][1] = s[mt][2] = s[mt][3] = 0.f;
            }
            uint4 rk[8];
            float ck[8];
            const float *ksp = a.ks + ok / 32;      /* (rows of 32 scales) */
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                int b = u < 2 ? bt + 8 * u : 8 * bt + u;
                rk[u] = ft_ld<KB>(kb, ok + 8 * cc + 32 * b);
                ck[u] = ksp[b];
            }
            #pragma unroll
            for (int u = 0; u < 8; ++u) {
                uint32_t k[4];
                ft_h8<KB>(rk[u], ck[u], k);
                int ch = 4 * (u < 2 ? bt + 8 * u : 8 * bt + u) + cc;
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    int r0 = 16 * mt + g, r1 = r0 + 8;
                    bool h1 = r1 < 8 * T;
                    #pragma unroll
                    for (int pl = 0; pl < 2; ++pl) {
                        uint4 (*qp)[QLD] = pl ? ql : qh;
                        uint4 q0 = qp[r0][T == 1 ? ch : ch ^ (4 * (r0 & 1))];
                        uint4 q1 = h1 ? qp[r1][ch ^ (4 * (r1 & 1))] : make_uint4(0u, 0u, 0u, 0u);
                        uint32_t av[4] = {q0.x, q1.x, q0.y, q1.y};
                        uint32_t b0[2] = {k[0], k[1]};
                        mma16816(s[mt], av, b0);
                        av[0] = q0.z; av[1] = q1.z; av[2] = q0.w; av[3] = q1.w;
                        b0[0] = k[2]; b0[1] = k[3];
                        mma16816(s[mt], av, b0);
                    }
                }
            }
            /* keys only: the values of blocks bt and bt + 8 of key kx (values
             * i and i + 256 of the lane, turned back) to rot, for warps 0
             * and 4 */
            if (rt) {
                int i = 32 * bt + 8 * cc;
                float f[8], y[8];
                ko_i16x8(rk[0], a.vs[ok / 32 + bt], f);
                ko_i16x8(rk[1], a.vs[ok / 32 + bt + 8], y);
                if (ko_turned(i)) {
                    #pragma unroll
                    for (int u = 0; u < 8; ++u) {
                        float2 cs = ko_cs(kx, i + u);
                        float v0 = f[u] * cs.x + y[u] * cs.y, v1 = y[u] * cs.x - f[u] * cs.y;
                        f[u] = v0;
                        y[u] = v1;
                    }
                }
                uint32_t h0[4], h1[4];
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    h0[t] = pack_h2(f[2 * t], f[2 * t + 1]);
                    h1[t] = pack_h2(y[2 * t], y[2 * t + 1]);
                }
                int sw = 2 * ((g >> 1) & 3) ^ 4 * (g & 1);     /* ft_rsw(kr) */
                rot[buf][kr][(4 * bt + cc) ^ sw] = make_uint4(h0[0], h0[1], h0[2], h0[3]);
                rot[buf][kr][(8 + 4 * bt + cc) ^ sw] = make_uint4(h1[0], h1[1], h1[2], h1[3]);
            }
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                int r0 = 16 * mt + g, r1 = r0 + 8;
                *(float2 *)&ss[buf][bt][r0][kr - g + 2 * cc] = make_float2(s[mt][0], s[mt][1]);
                if (r1 < 8 * T) {
                    *(float2 *)&ss[buf][bt][r1][kr - g + 2 * cc] = make_float2(s[mt][2], s[mt][3]);
                }
            }
            /* the values of the tile, while the scores join (warps 0 and 4
             * of a cache of keys only: from rot) */
            /* value k: key 16 (k / 4) + 2 cc + (k & 1) + 8 ((k >> 1) & 1) */
            auto vk = [&](int k) { return 16 * (k / 4) + 2 * cc + (k & 1) + 8 * ((k >> 1) & 1); };
            auto vo = [&](int k) {
                return (size_t)min(x0 + vk(k), j1 - 1) * a.rstride + (size_t)kv * a.hstride + 64 * warp + 8 * g;
            };
            const bool vr = rt && (warp & 3) == 0;
            uint4 rv[8];
            float cv[8];
            if (!vr) {
                #pragma unroll
                for (int k = 0; k < 8; ++k) {
                    size_t o = vo(k);
                    rv[k] = ft_ld<VB>(vb, o);
                    cv[k] = a.vs[o / 32];
                }
            }
            /* keys only without rot (ko_tab: more than 64 pairs): a turned
             * block from its key and the partner */
            auto vh8 = [&](int k, uint4 u, float sc, uint32_t *h) {
                int i = 64 * warp + 8 * g;
                if (ko_turned(i)) {
                    int x = min(x0 + vk(k), j1 - 1);
                    size_t po = vo(k) - i + ko_partner(i);
                    float f[8], y[8];
                    ko_i16x8(u, sc, f);
                    ko_i16x8(*(const uint4 *)((const int16_t *)vb + po), a.vs[po / 32], y);
                    ko_rot8(f, y, x, i);
                    #pragma unroll
                    for (int t = 0; t < 4; ++t) {
                        h[t] = pack_h2(f[2 * t], f[2 * t + 1]);
                    }
                } else {
                    ft_h8<VB>(u, sc, h);
                }
            };
            __syncthreads();
            /* the softmax of row 8 j + g: the keys 16 ks + 2 cc + {0, 1, 8, 9} of
             * the lane (the sum of the scores of the two halves); B = P^T of
             * each k16 step (b0 keys 2 cc, 2 cc + 1; b1 + 8) */
            uint32_t pb[T][2][2];
            #pragma unroll
            for (int j = 0; j < T; ++j) {
                int lim = (act >> j) & 1 ? min(j1, n + j) - x0 : -FT_TK;  /* its keys here */
                float sv[8];
                #pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    #pragma unroll
                    for (int hf = 0; hf < 2; ++hf) {
                        int kk = 16 * ks + 2 * cc + 8 * hf;
                        float2 v = *(const float2 *)&ss[buf][0][8 * j + g][kk];
                        float2 w = *(const float2 *)&ss[buf][1][8 * j + g][kk];
                        sv[4 * ks + 2 * hf] = kk < lim ? v.x + w.x : -INFINITY;
                        sv[4 * ks + 2 * hf + 1] = kk + 1 < lim ? v.y + w.y : -INFINITY;
                    }
                }
                float t0 = sv[0];
                #pragma unroll
                for (int u = 1; u < 8; ++u) {
                    t0 = fmaxf(t0, sv[u]);
                }
                t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 1));
                t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 2));
                float n0 = fmaxf(m[j], t0);
                /* no key of query j yet (a tile past its keys): nothing changes */
                float a0 = n0 == -INFINITY ? 1.f : expf(m[j] - n0);
                m[j] = n0;
                float ps = 0.f;
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    sv[u] = n0 == -INFINITY ? 0.f : expf(sv[u] - n0);
                    ps += sv[u];
                }
                l[j] = l[j] * a0 + ps;
                #pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    pb[j][ks][0] = pack_h2(sv[4 * ks], sv[4 * ks + 1]);
                    pb[j][ks][1] = pack_h2(sv[4 * ks + 2], sv[4 * ks + 3]);
                }
                /* the factor of the heads 2 cc and 2 cc + 1 (the columns of the
                 * fragments of O^T) from the lanes of their rows */
                float al0 = __shfl_sync(RC_FULL_MASK, a0, 8 * cc), al1 = __shfl_sync(RC_FULL_MASK, a0, 8 * cc + 4);
                #pragma unroll
                for (int mt = 0; mt < 4; ++mt) {
                    acc[mt][j][0] *= al0;
                    acc[mt][j][1] *= al1;
                    acc[mt][j][2] *= al0;
                    acc[mt][j][3] *= al1;
                }
            }
            /* O^T += V^T P^T, hk(k, h) the 4 float16 pairs of value k */
            auto vmma = [&](auto hk) {
                #pragma unroll
                for (int ks = 0; ks < 2; ++ks) {
                    uint32_t h[4][4];
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) {
                        hk(4 * ks + k, h[k]);
                    }
                    /* rows g (values 2 t) and g + 8 (2 t + 1) of tile t: the
                     * pairs of keys 2 cc, 2 cc + 1 and 2 cc + 8, 2 cc + 9 */
                    #pragma unroll
                    for (int mt = 0; mt < 4; ++mt) {
                        uint32_t av[4] = {__byte_perm(h[0][mt], h[1][mt], 0x5410),
                                          __byte_perm(h[0][mt], h[1][mt], 0x7632),
                                          __byte_perm(h[2][mt], h[3][mt], 0x5410),
                                          __byte_perm(h[2][mt], h[3][mt], 0x7632)};
                        #pragma unroll
                        for (int j = 0; j < T; ++j) {
                            mma16816(acc[mt][j], av, pb[j][ks]);
                        }
                    }
                }
            };
            if (vr) {
                /* ft_rsw of key vk(k): 2 cc ^ 4 (k & 1) */
                vmma([&](int k, uint32_t *h) {
                    uint4 u = rot[buf][vk(k)][(8 * bt + g) ^ (2 * cc) ^ (4 * (k & 1))];
                    h[0] = u.x; h[1] = u.y; h[2] = u.z; h[3] = u.w;
                });
            } else if (VB == 16 && kon && !rt) {
                vmma([&](int k, uint32_t *h) { vh8(k, rv[k], cv[k], h); });
            } else {
                vmma([&](int k, uint32_t *h) { ft_h8<VB>(rv[k], cv[k], h); });
            }
        }
        /* the part of each head: values 64 warp + 8 g + 2 mt (+ 1) of heads 2 cc,
         * 2 cc + 1 of query j */
        #pragma unroll
        for (int j = 0; j < T; ++j) {
            if (!((act >> j) & 1)) {
                continue;
            }
            float lj = l[j];
            lj += __shfl_xor_sync(RC_FULL_MASK, lj, 1);
            lj += __shfl_xor_sync(RC_FULL_MASK, lj, 2);
            float *p0 = part + ((size_t)(j * qhn + kv * 8 + 2 * cc) * nb + c) * 514;
            float *p1 = part + ((size_t)(j * qhn + kv * 8 + 2 * cc + 1) * nb + c) * 514;
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                int d = 64 * warp + 8 * g + 2 * mt;
                *(float2 *)&p0[d] = make_float2(acc[mt][j][0], acc[mt][j][2]);
                *(float2 *)&p1[d] = make_float2(acc[mt][j][1], acc[mt][j][3]);
            }
            if (warp == 0 && cc == 0) {
                float *o = part + ((size_t)(j * qhn + kv * 8 + g) * nb + c) * 514;
                o[512] = m[j];
                o[513] = lj;
            }
        }
    }
}

/* The decode attention of a sliding layer of the Gemma 4 26B (head_dim 256,
 * 2 query heads for each key head, a window of W keys) on the tensor cores,
 * for T = 1 to FT_TMAX queries in one record: operands 10 n (the rows to
 * query 0, from row 0 of the buffer), 11 t, 12 W, 13 base (the position of
 * row 0); query j sees rows max(0, n + j - W) to n + j - 1. k_attn_fdt took 26 us a layer (a step at
 * any length: the window), and a verify group of 3 ran it for each query
 * (25 layers: 2 ms of a cycle).
 *
 * Block (kv, c), 4 warps: chunk c of FS_L rows from the chunk of the first
 * row of query 0, 32 rows at a time; the 2 T heads of the queries (query
 * j, head h: row 2 j + h) are the rows of one m16 tile and the columns of
 * one n8 tile (T 4: all 8 columns). The method of k_attn_fdtc: warp w the scores of rows 8 w ..
 * 8 w + 7 (lane (g, cc): values 32 b + 8 cc .. + 7 of row g), joined in
 * shared memory, each warp the softmax that runs, then O^T of values 64 w
 * .. 64 w + 63. The chunks and the tiles sit at multiples of FS_L and 32
 * positions (not rows: the buffer drops its old rows at other steps in a
 * decode and in MTP), a query masks the rows out of its window, a tile with
 * none of its rows changes nothing in it, and the join of query j reads its
 * own chunks only: each query gets the bits of a step (T 1) at its
 * position, MTP verifies with the numbers of the decode. Rows out of the
 * window of query 0 to the last are read as its first or last row (their
 * values finite; weight 0). */
#define FS_L 64
#define FS_SLD 36
#define FS_MAXC 64

template <int T, int KB, int VB>
__global__ void __launch_bounds__(128) k_attn_fdts(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    constexpr int R = 2 * T;                       /* rows of the tile in use */
    __shared__ __align__(16) uint4 qh[R][32];
    __shared__ __align__(16) uint4 ql[R][32];
    __shared__ __align__(16) float ss[2][R][FS_SLD];
    attn_d a = attn_get(r, e);
    int kv = blockIdx.x, c = blockIdx.y, ncs = gridDim.y;
    int n = a.n, W = DI(12), base = DI(13), hi = n + T - 1, lo0 = max(0, n - W);
    int x_beg = ((lo0 + base) / FS_L + c) * FS_L - base;     /* rows; maybe < 0 */
    if (x_beg >= hi) {
        return;
    }
    int x_end = min(hi, x_beg + FS_L);
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, cc = lane % 4;
    int qhn = a.qh;
    /* q: row 2 j + h (query j, head 2 kv + h) */
    for (int i = threadIdx.x; i < R * 64; i += 128) {
        int row = i / 64, d = 4 * (i % 64);
        float4 v = *(const float4 *)(a.q + ((size_t)(row / 2) * qhn + kv * 2 + row % 2) * 256 + d);
        float f[4] = {v.x, v.y, v.z, v.w};
        uint32_t hw[2], lw[2];
        #pragma unroll
        for (int t = 0; t < 2; ++t) {
            __half h0 = __float2half_rn(f[2 * t]), h1 = __float2half_rn(f[2 * t + 1]);
            hw[t] = pack_h2(__half2float(h0), __half2float(h1));
            lw[t] = pack_h2(f[2 * t] - __half2float(h0), f[2 * t + 1] - __half2float(h1));
        }
        int ch = (d / 8) ^ (4 * (row & 1));
        *((uint2 *)&qh[row][ch] + (d % 8) / 4) = make_uint2(hw[0], hw[1]);
        *((uint2 *)&ql[row][ch] + (d % 8) / 4) = make_uint2(lw[0], lw[1]);
    }
    __syncthreads();
    const void *kb = a.kq, *vb = a.vq;              /* operands 1, 3 (int16 or int8) */
    /* the window of the query of row g (lanes past the rows: none) */
    int jq = g / 2, wlo = g < R ? max(0, n + jq - W) : 0, whi = g < R ? n + jq : 0;
    float acc[4][4];
    #pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
        acc[mt][0] = acc[mt][1] = acc[mt][2] = acc[mt][3] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    int buf = 0;
    for (int x0 = x_beg; x0 < x_end; x0 += 32, buf ^= 1) {
        size_t ok = (size_t)min(max(x0 + 8 * warp + g, lo0), hi - 1) * a.rstride +
                    (size_t)kv * a.hstride;
        uint4 rk[8];
        float ck[8];
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            size_t o = ok + 32 * u + 8 * cc;
            rk[u] = ft_ld<KB>(kb, o);
            ck[u] = a.ks[o / 32];
        }
        float s[4] = {0.f, 0.f, 0.f, 0.f};
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            uint32_t k[4];
            ft_h8<KB>(rk[u], ck[u], k);
            int ch = 4 * u + cc;
            #pragma unroll
            for (int pl = 0; pl < 2; ++pl) {
                uint4 q0 = g < R ? (pl ? ql : qh)[g][ch ^ (4 * (g & 1))] : make_uint4(0u, 0u, 0u, 0u);
                uint32_t av[4] = {q0.x, 0u, q0.y, 0u};    /* rows 8-15: none */
                uint32_t b0[2] = {k[0], k[1]};
                mma16816(s, av, b0);
                av[0] = q0.z; av[2] = q0.w;
                b0[0] = k[2]; b0[1] = k[3];
                mma16816(s, av, b0);
            }
        }
        if (g < R) {
            *(float2 *)&ss[buf][g][8 * warp + 2 * cc] = make_float2(s[0], s[1]);
        }
        /* the values (row 16 (k / 4) + 2 cc + (k & 1) + 8 ((k >> 1) & 1)),
         * while the scores join */
        uint4 rv[8];
        float cv[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            int x = x0 + 16 * (k / 4) + 2 * cc + (k & 1) + 8 * ((k >> 1) & 1);
            size_t o = (size_t)min(max(x, lo0), hi - 1) * a.rstride + (size_t)kv * a.hstride +
                       64 * warp + 8 * g;
            rv[k] = ft_ld<VB>(vb, o);
            cv[k] = a.vs[o / 32];
        }
        __syncthreads();
        /* the softmax of row g: rows x0 + 16 ks + 2 cc + {0, 1, 8, 9} */
        float sv[8];
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            #pragma unroll
            for (int hf = 0; hf < 2; ++hf) {
                int kk = 16 * ks + 2 * cc + 8 * hf, x = x0 + kk;
                float2 v = g < R ? *(const float2 *)&ss[buf][g][kk] : make_float2(0.f, 0.f);
                sv[4 * ks + 2 * hf] = x >= wlo && x < whi ? v.x : -INFINITY;
                sv[4 * ks + 2 * hf + 1] = x + 1 >= wlo && x + 1 < whi ? v.y : -INFINITY;
            }
        }
        float t0 = sv[0];
        #pragma unroll
        for (int u = 1; u < 8; ++u) {
            t0 = fmaxf(t0, sv[u]);
        }
        t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 1));
        t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 2));
        float n0 = fmaxf(m, t0);
        /* no row of the window yet (a tile out of it): nothing changes */
        float a0 = n0 == -INFINITY ? 1.f : expf(m - n0);
        m = n0;
        float ps = 0.f;
        #pragma unroll
        for (int u = 0; u < 8; ++u) {
            sv[u] = n0 == -INFINITY ? 0.f : expf(sv[u] - n0);
            ps += sv[u];
        }
        l = l * a0 + ps;
        uint32_t pb[2][2];
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            pb[ks][0] = pack_h2(sv[4 * ks], sv[4 * ks + 1]);
            pb[ks][1] = pack_h2(sv[4 * ks + 2], sv[4 * ks + 3]);
        }
        /* the factor of the columns 2 cc and 2 cc + 1 from the lanes of their rows */
        float al0 = __shfl_sync(RC_FULL_MASK, a0, 8 * cc), al1 = __shfl_sync(RC_FULL_MASK, a0, 8 * cc + 4);
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            acc[mt][0] *= al0;
            acc[mt][1] *= al1;
            acc[mt][2] *= al0;
            acc[mt][3] *= al1;
        }
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            uint32_t h[4][4];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                ft_h8<VB>(rv[4 * ks + k], cv[4 * ks + k], h[k]);
            }
            #pragma unroll
            for (int mt = 0; mt < 4; ++mt) {
                uint32_t av[4] = {__byte_perm(h[0][mt], h[1][mt], 0x5410),
                                  __byte_perm(h[0][mt], h[1][mt], 0x7632),
                                  __byte_perm(h[2][mt], h[3][mt], 0x5410),
                                  __byte_perm(h[2][mt], h[3][mt], 0x7632)};
                mma16816(acc[mt], av, pb[ks]);
            }
        }
    }
    l += __shfl_xor_sync(RC_FULL_MASK, l, 1);
    l += __shfl_xor_sync(RC_FULL_MASK, l, 2);
    /* the part of each head: values 64 warp + 8 g + 2 mt (+ 1) of the
     * columns 2 cc, 2 cc + 1 (query cc, heads 2 kv, 2 kv + 1) */
    if (2 * cc < R) {
        float *p0 = part + ((size_t)(cc * qhn + kv * 2) * ncs + c) * 258;
        float *p1 = part + ((size_t)(cc * qhn + kv * 2 + 1) * ncs + c) * 258;
        #pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            int d = 64 * warp + 8 * g + 2 * mt;
            *(float2 *)&p0[d] = make_float2(acc[mt][0], acc[mt][2]);
            *(float2 *)&p1[d] = make_float2(acc[mt][1], acc[mt][3]);
        }
    }
    if (warp == 0 && cc == 0 && g < R) {
        float *o = part + ((size_t)(jq * qhn + kv * 2 + g % 2) * ncs + c) * 258;
        o[256] = m;
        o[257] = l;
    }
}

/* The join of k_attn_fdts: block (h, x) the values x * 128 to x * 128 + 127
 * of head h % qh of query h / qh, over the chunks of its window only (in
 * the order and on the threads of a step at its position). ncs: the chunk
 * stride. */
__global__ void k_attn_fdts_join(const gp_rec *r, const int64_t *e, const float *part, int ncs)
{
    PDL_START();
    __shared__ float w[FS_MAXC];
    attn_d a = attn_get(r, e);
    int h = blockIdx.x, W = DI(12), base = DI(13);
    int hj = a.n + h / a.qh, lj = max(0, hj - W);
    int c0 = (max(0, a.n - W) + base) / FS_L, cj = (lj + base) / FS_L;
    int nc = (hj - 1 + base) / FS_L - cj + 1;
    const float *ph = part + ((size_t)h * ncs + (cj - c0)) * 258;
    float M = -INFINITY;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        M = fmaxf(M, ph[(size_t)c * 258 + 256]);
    }
    M = block_max(M);
    float wl = 0.f;
    for (int c = threadIdx.x; c < nc; c += blockDim.x) {
        float mc = ph[(size_t)c * 258 + 256];
        float wc = mc == -INFINITY ? 0.f : expf(mc - M);
        w[c] = wc;
        wl += wc * ph[(size_t)c * 258 + 257];
    }
    float wsum = block_sum(wl);
    float inv = wsum > 0.f ? 1.0f / wsum : 0.f;
    __syncthreads();
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i < 256) {
        float acc = 0.f;
        for (int c = 0; c < nc; ++c) {
            acc += w[c] * ph[(size_t)c * 258 + i];
        }
        a.out[(size_t)h * 256 + i] = acc * inv;
    }
}

/* RECLANK_GPU_FDT_TC: 1 (the default) k_attn_fdtc for a global layer of the
 * 26B; 0 k_attn_fdt (a test; no record of a group, reclank/backends/cuda/runtime.py). */
static int fdtc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_FDT_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}

/* The chunks of k_attn_fdtc for kvh key heads: one block for each place on
 * the GPU (of a step), all at once (512 blocks, 3.1 rounds on the 82 SMs of
 * the 3090, left the last round 12% full); RECLANK_GPU_FDT_NB sets it. The
 * same for every T (a group verifies with the numbers of a step), and the
 * parts of FT_TMAX queries fit GG_PART_FLOATS. The bound stays that of 3
 * queries (ATTN_CHUNKS / 3: 85), from before T 4: the same chunks, the same
 * bits of a decode step. */
static int fdtc_chunks(int kvh)
{
    static int nb = -1;
    if (nb < 0) {
        const char *v = getenv("RECLANK_GPU_FDT_NB");
        int dev = 0, sms = 0, occ = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_attn_fdtc<1, 16, 16>, 256, FT_SMEM(1, 1));
        nb = v != NULL ? atoi(v) : sms * (occ > 0 ? occ : 1);
    }
    return max(1, min(ATTN_CHUNKS / 3, nb / max(1, kvh)));
}

template <int T>
static void fdts_run(int op, dim3 grid, const gp_rec *dr, const int64_t *denv, float *part)
{
    if (op == GP_ATTN_QC) {
        k_attn_fdts<T, 16, 16><<<grid, 128, 0, gg_stream>>>(dr, denv, part);
    } else if (op == GP_ATTN_Q8) {
        k_attn_fdts<T, 8, 8><<<grid, 128, 0, gg_stream>>>(dr, denv, part);
    } else {
        k_attn_fdts<T, 16, 8><<<grid, 128, 0, gg_stream>>>(dr, denv, part);
    }
}

template <int T>
static void fdtc_run(int op, dim3 grid, const gp_rec *dr, const int64_t *denv, float *part)
{
    static int attr = 0;
    if (!attr) {
        cudaFuncSetAttribute(k_attn_fdtc<T, 16, 16>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)FT_SMEM(T, 1));
        cudaFuncSetAttribute(k_attn_fdtc<T, 8, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)FT_SMEM(T, 0));
        cudaFuncSetAttribute(k_attn_fdtc<T, 16, 8>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)FT_SMEM(T, 0));
        attr = 1;
    }
    if (op == GP_ATTN_QC) {
        k_attn_fdtc<T, 16, 16><<<grid, 256, FT_SMEM(T, 1), gg_stream>>>(dr, denv, part);
    } else if (op == GP_ATTN_Q8) {
        k_attn_fdtc<T, 8, 8><<<grid, 256, FT_SMEM(T, 0), gg_stream>>>(dr, denv, part);
    } else {
        k_attn_fdtc<T, 16, 8><<<grid, 256, FT_SMEM(T, 0), gg_stream>>>(dr, denv, part);
    }
}

/* GP_ATTN_QSA of one query (a step, a row of a small group: the MTP layer
 * over all the positions, a QSA layer over its rows) on the tensor cores,
 * in the chunks of k_attn_part (k_attn_join adds them). k_attn_part takes
 * 8 query heads in a block, so the 12 heads of a key head of Qwen3.8 took
 * two blocks, each reading all the keys and values of the chunk: at 200K
 * positions 0.43 ms a row of the MTP layer, 18% of the decode. Block (kv,
 * c): chunk c, the heads of key head kv as the rows of one m16 tile, the
 * method of k_attn_qsa_tc (4 warps, each with its own 16 keys at a time of
 * the chunk; q as two float16 planes with Q2). It writes the part of each
 * head as k_attn_part does: the sum of exp(score - m) v, m, l. */
template <int Q2>
__global__ void __launch_bounds__(128) k_attn_part_tc(const gp_rec *r, const int64_t *e, float *part)
{
    PDL_START();
    extern __shared__ __align__(16) __half qtsm[];
    __half (*qh)[QT_LD] = (__half (*)[QT_LD])qtsm;
    __half (*ql)[QT_LD] = (__half (*)[QT_LD])(qtsm + 16 * QT_LD);
    __shared__ float wm[4][16], wl[4][16];
    __shared__ float cbs[64];
    if (threadIdx.x < 64) {
        cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
    }
    int nq = DI(7), nk = DI(8), form = DI(15);
    int rep = nq / nk, kv = blockIdx.x, c = blockIdx.y;
    int cn = DP(const int32_t, 13)[0];
    int n = cn < 0 ? (int)(di(r, e, 11) + 1) : cn;
    int len = attn_len(n);
    int j0 = c * len, j1 = min(n, j0 + len);
    if (j0 >= n) {
        return;
    }
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, cc = lane % 4;
    __half (*ks_)[QT_LD] = (__half (*)[QT_LD])(qtsm + (2 + warp) * 16 * QT_LD);
    const float *q = DP(const float, 0) + (size_t)kv * rep * 256;
    for (int i = threadIdx.x; i < 16 * 256; i += blockDim.x) {
        int h = i / 256, d = i % 256;
        float v = h < rep ? q[(size_t)h * 256 + d] : 0.f;
        __half hi = __float2half_rn(v);
        qh[h][d] = hi;
        ql[h][d] = __float2half_rn(v - __half2float(hi));
    }
    __syncthreads();
    const int32_t *rows = cn < 0 ? NULL : DP(const int32_t, 12);
    const uint8_t *kb = DP(const uint8_t, 1), *vb = DP(const uint8_t, 3);
    const float *ksc = DP(const float, 2), *vsc = DP(const float, 4);
    size_t rs = (size_t)nk * 256, off = (size_t)kv * 256 + 8 * lane;
    auto decode = [&](const uint8_t *base, const float *sc, int x0) {
        #pragma unroll
        for (int rr = 0; rr < 16; ++rr) {
            int x = x0 + rr;
            float f[8];
            if (x < j1) {
                size_t o = (size_t)(rows ? rows[x] : x) * rs + off;
                if (form == 4) {
                    tq6x8(base, o, f, cbs);
                } else {
                    fd_i8x8(*(const uint2 *)((const int8_t *)base + o), f);
                }
                float s = sc[o / 32];
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    f[u] *= s;
                }
            } else {
                #pragma unroll
                for (int u = 0; u < 8; ++u) {
                    f[u] = 0.f;
                }
            }
            *(uint4 *)&ks_[rr][8 * lane] = make_uint4(pack_h2(f[0], f[1]), pack_h2(f[2], f[3]),
                                                      pack_h2(f[4], f[5]), pack_h2(f[6], f[7]));
        }
        __syncwarp();
    };
    float acc[32][4];
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int x0 = j0 + warp * 16; x0 < j1; x0 += 64) {
        decode(kb, ksc, x0);
        float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
        #pragma unroll
        for (int kk = 0; kk < 256; kk += 16) {
            uint32_t a[4], b[2][2];
            #pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                b[jn][0] = *(const uint32_t *)&ks_[jn * 8 + g][kk + 2 * cc];
                b[jn][1] = *(const uint32_t *)&ks_[jn * 8 + g][kk + 2 * cc + 8];
            }
            a[0] = *(const uint32_t *)&qh[g][kk + 2 * cc];
            a[1] = *(const uint32_t *)&qh[g + 8][kk + 2 * cc];
            a[2] = *(const uint32_t *)&qh[g][kk + 2 * cc + 8];
            a[3] = *(const uint32_t *)&qh[g + 8][kk + 2 * cc + 8];
            mma16816(s[0], a, b[0]);
            mma16816(s[1], a, b[1]);
            if (Q2) {
                a[0] = *(const uint32_t *)&ql[g][kk + 2 * cc];
                a[1] = *(const uint32_t *)&ql[g + 8][kk + 2 * cc];
                a[2] = *(const uint32_t *)&ql[g][kk + 2 * cc + 8];
                a[3] = *(const uint32_t *)&ql[g + 8][kk + 2 * cc + 8];
                mma16816(s[0], a, b[0]);
                mma16816(s[1], a, b[1]);
            }
        }
        __syncwarp();
        #pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            int x = x0 + jn * 8 + 2 * cc;
            if (x >= j1) { s[jn][0] = s[jn][2] = -INFINITY; }
            if (x + 1 >= j1) { s[jn][1] = s[jn][3] = -INFINITY; }
        }
        float t0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
        float t1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
        t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 1));
        t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(RC_FULL_MASK, t1, 1));
        t1 = fmaxf(t1, __shfl_xor_sync(RC_FULL_MASK, t1, 2));
        float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);     /* finite: key x0 < j1 */
        float a0 = expf(m0 - n0), a1 = expf(m1 - n1);
        m0 = n0;
        m1 = n1;
        float p[2][4];
        #pragma unroll
        for (int jn = 0; jn < 2; ++jn) {
            p[jn][0] = expf(s[jn][0] - n0);
            p[jn][1] = expf(s[jn][1] - n0);
            p[jn][2] = expf(s[jn][2] - n1);
            p[jn][3] = expf(s[jn][3] - n1);
        }
        l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
        l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
        #pragma unroll
        for (int nt = 0; nt < 32; ++nt) {
            acc[nt][0] *= a0;
            acc[nt][1] *= a0;
            acc[nt][2] *= a1;
            acc[nt][3] *= a1;
        }
        decode(vb, vsc, x0);
        uint32_t pa[4] = {pack_h2(p[0][0], p[0][1]), pack_h2(p[0][2], p[0][3]),
                          pack_h2(p[1][0], p[1][1]), pack_h2(p[1][2], p[1][3])};
        #pragma unroll
        for (int nt = 0; nt < 32; nt += 2) {
            uint32_t bv[4];
            ldsm_x4_trans(bv, &ks_[(lane % 8) + 8 * ((lane / 8) & 1)][nt * 8 + 8 * (lane / 16)]);
            mma16816(acc[nt], pa, bv);
            mma16816(acc[nt + 1], pa, bv + 2);
        }
        __syncwarp();
    }
    l0 += __shfl_xor_sync(RC_FULL_MASK, l0, 1);
    l0 += __shfl_xor_sync(RC_FULL_MASK, l0, 2);
    l1 += __shfl_xor_sync(RC_FULL_MASK, l1, 1);
    l1 += __shfl_xor_sync(RC_FULL_MASK, l1, 2);
    if (cc == 0) {
        wm[warp][g] = m0;
        wm[warp][g + 8] = m1;
        wl[warp][g] = l0;
        wl[warp][g + 8] = l1;
    }
    __syncthreads();
    /* the join of the warps (k_attn_qsa_tc): rows g and g + 8, the weight
     * exp(m_w - M) of each warp; no division (k_attn_join divides) */
    float M0 = -INFINITY, M1 = -INFINITY;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        M0 = fmaxf(M0, wm[w][g]);
        M1 = fmaxf(M1, wm[w][g + 8]);
    }
    float L0 = 0.f, L1 = 0.f;
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        L0 += wm[w][g] == -INFINITY ? 0.f : expf(wm[w][g] - M0) * wl[w][g];
        L1 += wm[w][g + 8] == -INFINITY ? 0.f : expf(wm[w][g + 8] - M1) * wl[w][g + 8];
    }
    float f0 = m0 == -INFINITY ? 0.f : expf(m0 - M0);
    float f1 = m1 == -INFINITY ? 0.f : expf(m1 - M1);
    float (*red)[256] = (float (*)[256])qtsm;
    for (int w = 0; w < 4; ++w) {
        if (warp == w) {
            #pragma unroll
            for (int nt = 0; nt < 32; ++nt) {
                int d = nt * 8 + 2 * cc;
                if (w == 0) {
                    red[g][d] = acc[nt][0] * f0;
                    red[g][d + 1] = acc[nt][1] * f0;
                    red[g + 8][d] = acc[nt][2] * f1;
                    red[g + 8][d + 1] = acc[nt][3] * f1;
                } else {
                    red[g][d] += acc[nt][0] * f0;
                    red[g][d + 1] += acc[nt][1] * f0;
                    red[g + 8][d] += acc[nt][2] * f1;
                    red[g + 8][d + 1] += acc[nt][3] * f1;
                }
            }
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < rep * 256; i += blockDim.x) {
        int h = i / 256, d = i % 256;
        part[((size_t)(kv * rep + h) * ATTN_CHUNKS + c) * 258 + d] = red[h][d];
    }
    if (warp == 0 && cc == 0) {
        /* lanes g: rows g and g + 8 (M, L are the same in each warp) */
        if (g < rep) {
            float *o = part + ((size_t)(kv * rep + g) * ATTN_CHUNKS + c) * 258;
            o[256] = M0;
            o[257] = L0;
        }
        if (g + 8 < rep) {
            float *o = part + ((size_t)(kv * rep + g + 8) * ATTN_CHUNKS + c) * 258;
            o[256] = M1;
            o[257] = L1;
        }
    }
}

/* GP_ATTN_QSA of a large group whose queries all see every position before
 * them (maxsel 0: cnt is -1 for each query; the dense attention of the MTP
 * layer) on the tensor cores. k_attn_qsa_tc decodes all the keys and values
 * again for each query: at 200K positions a group of 256 rows of the MTP
 * layer read 51 GB, 375 ms. Here block (j4, kv) takes the QD_W queries
 * QD_W j4 .. QD_W j4 + QD_W - 1, one for each warp (its query heads of key
 * head kv are the rows of its m16 tile, as in k_attn_qsa_tc), and the warps
 * share each tile of 16 keys and values: the threads of the block decode it
 * one time to shared memory for the QD_W queries. The math of each query is
 * that of k_attn_qsa_tc (q as two float16 planes with Q2, the softmax that
 * runs, P float16, V by ldmatrix.trans); one warp holds a query, so there is
 * no join. */
#define QD_W 4
#define QD_SMEM ((size_t)(QD_W * 2 * 16 + 2 * 16) * QT_LD * 2)
template <int Q2>
__global__ void __launch_bounds__(32 * QD_W) k_attn_dense_tc(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    extern __shared__ __align__(16) __half qdsm[];
    __shared__ float cbs[64];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    __half (*qh)[QT_LD] = (__half (*)[QT_LD])(qdsm + (size_t)(2 * warp) * 16 * QT_LD);
    __half (*ql)[QT_LD] = (__half (*)[QT_LD])(qdsm + (size_t)(2 * warp + 1) * 16 * QT_LD);
    __half (*kt)[QT_LD] = (__half (*)[QT_LD])(qdsm + (size_t)(2 * QD_W) * 16 * QT_LD);
    __half (*vt)[QT_LD] = (__half (*)[QT_LD])(qdsm + (size_t)(2 * QD_W + 1) * 16 * QT_LD);
    if (threadIdx.x < 64) {
        cbs[threadIdx.x] = tq6_cb_d[threadIdx.x];
    }
    int nq = DI(7), nk = DI(8), t = DI(10), form = DI(15);
    int64_t pos = di(r, e, 11);
    int rep = nq / nk, kv = blockIdx.y;
    int j = blockIdx.x * QD_W + warp;
    bool live = j < t;
    /* this warp's query to its planes */
    const float *q = DP(const float, 0) + ((size_t)j * nq + (size_t)kv * rep) * 256;
    for (int i = lane; i < 16 * 256; i += 32) {
        int h = i / 256, d = i % 256;
        float v = live && h < rep ? q[(size_t)h * 256 + d] : 0.f;
        __half hi = __float2half_rn(v);
        qh[h][d] = hi;
        ql[h][d] = __float2half_rn(v - __half2float(hi));
    }
    int n = live ? (int)(pos + j + 1) : 0;                    /* the keys of this query */
    int jl = min(t, (int)(blockIdx.x + 1) * QD_W) - 1;
    int nmax = (int)(pos + jl + 1);                           /* the keys of the block */
    const uint8_t *kb = DP(const uint8_t, 1), *vb = DP(const uint8_t, 3);
    const float *ksc = DP(const float, 2), *vsc = DP(const float, 4);
    size_t rs = (size_t)nk * 256;
    /* row tid / 8 of the tile: 8 values at (tid % 8) * 8 + 64 u, u = 0..3 */
    int dr = threadIdx.x / 8, dc = (threadIdx.x % 8) * 8;
    auto decode = [&](const uint8_t *base, const float *sc, int x0, __half (*tile)[QT_LD]) {
        int x = x0 + dr;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int d0 = dc + 64 * u;
            float f[8];
            if (x < nmax) {
                size_t o = (size_t)x * rs + (size_t)kv * 256 + d0;
                if (form == 4) {
                    tq6x8(base, o, f, cbs);
                } else {
                    fd_i8x8(*(const uint2 *)((const int8_t *)base + o), f);
                }
                float s = sc[o / 32];
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    f[w] *= s;
                }
            } else {
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    f[w] = 0.f;
                }
            }
            *(uint4 *)&tile[dr][d0] = make_uint4(pack_h2(f[0], f[1]), pack_h2(f[2], f[3]),
                                                 pack_h2(f[4], f[5]), pack_h2(f[6], f[7]));
        }
    };
    float acc[32][4];
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;   /* rows g, g + 8 */
    /* int8: the bytes and the scales of the next tile come to registers
     * before the compute of this one (in flight meanwhile; the loads of a
     * tile right before its use left the SM idle, one block of 4 warps) */
    uint2 rk[4], rv[4];
    float sk[4], sv[4];
    auto fetch = [&](int x0) {
        int x = x0 + dr;
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            size_t o = (size_t)x * rs + (size_t)kv * 256 + dc + 64 * u;
            bool in = x < nmax;
            rk[u] = in ? *(const uint2 *)((const int8_t *)kb + o) : make_uint2(0u, 0u);
            rv[u] = in ? *(const uint2 *)((const int8_t *)vb + o) : make_uint2(0u, 0u);
            sk[u] = in ? ksc[o / 32] : 0.f;
            sv[u] = in ? vsc[o / 32] : 0.f;
        }
    };
    auto store = [&]() {
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            float f[8], h[8];
            fd_i8x8(rk[u], f);
            fd_i8x8(rv[u], h);
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                f[w] *= sk[u];
                h[w] *= sv[u];
            }
            *(uint4 *)&kt[dr][dc + 64 * u] = make_uint4(pack_h2(f[0], f[1]), pack_h2(f[2], f[3]),
                                                        pack_h2(f[4], f[5]), pack_h2(f[6], f[7]));
            *(uint4 *)&vt[dr][dc + 64 * u] = make_uint4(pack_h2(h[0], h[1]), pack_h2(h[2], h[3]),
                                                        pack_h2(h[4], h[5]), pack_h2(h[6], h[7]));
        }
    };
    if (form == 1) {
        fetch(0);
    }
    for (int x0 = 0; x0 < nmax; x0 += 16) {
        if (form == 1) {
            store();
            __syncthreads();
            if (x0 + 16 < nmax) {
                fetch(x0 + 16);
            }
        } else {
            decode(kb, ksc, x0, kt);
            decode(vb, vsc, x0, vt);
            __syncthreads();
        }
        if (x0 < n) {
            float s[2][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
            #pragma unroll
            for (int kk = 0; kk < 256; kk += 16) {
                uint32_t a[4], b[2][2];
                #pragma unroll
                for (int jn = 0; jn < 2; ++jn) {
                    b[jn][0] = *(const uint32_t *)&kt[jn * 8 + g][kk + 2 * c];
                    b[jn][1] = *(const uint32_t *)&kt[jn * 8 + g][kk + 2 * c + 8];
                }
                a[0] = *(const uint32_t *)&qh[g][kk + 2 * c];
                a[1] = *(const uint32_t *)&qh[g + 8][kk + 2 * c];
                a[2] = *(const uint32_t *)&qh[g][kk + 2 * c + 8];
                a[3] = *(const uint32_t *)&qh[g + 8][kk + 2 * c + 8];
                mma16816(s[0], a, b[0]);
                mma16816(s[1], a, b[1]);
                if (Q2) {
                    a[0] = *(const uint32_t *)&ql[g][kk + 2 * c];
                    a[1] = *(const uint32_t *)&ql[g + 8][kk + 2 * c];
                    a[2] = *(const uint32_t *)&ql[g][kk + 2 * c + 8];
                    a[3] = *(const uint32_t *)&ql[g + 8][kk + 2 * c + 8];
                    mma16816(s[0], a, b[0]);
                    mma16816(s[1], a, b[1]);
                }
            }
            /* the keys past n (this query): no weight */
            #pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                int x = x0 + jn * 8 + 2 * c;
                if (x >= n) { s[jn][0] = s[jn][2] = -INFINITY; }
                if (x + 1 >= n) { s[jn][1] = s[jn][3] = -INFINITY; }
            }
            float t0 = fmaxf(fmaxf(s[0][0], s[0][1]), fmaxf(s[1][0], s[1][1]));
            float t1 = fmaxf(fmaxf(s[0][2], s[0][3]), fmaxf(s[1][2], s[1][3]));
            t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 1));
            t0 = fmaxf(t0, __shfl_xor_sync(RC_FULL_MASK, t0, 2));
            t1 = fmaxf(t1, __shfl_xor_sync(RC_FULL_MASK, t1, 1));
            t1 = fmaxf(t1, __shfl_xor_sync(RC_FULL_MASK, t1, 2));
            float n0 = fmaxf(m0, t0), n1 = fmaxf(m1, t1);     /* finite: key x0 < n */
            float a0 = expf(m0 - n0), a1 = expf(m1 - n1);
            m0 = n0;
            m1 = n1;
            float p[2][4];
            #pragma unroll
            for (int jn = 0; jn < 2; ++jn) {
                p[jn][0] = expf(s[jn][0] - n0);
                p[jn][1] = expf(s[jn][1] - n0);
                p[jn][2] = expf(s[jn][2] - n1);
                p[jn][3] = expf(s[jn][3] - n1);
            }
            l0 = l0 * a0 + p[0][0] + p[0][1] + p[1][0] + p[1][1];
            l1 = l1 * a1 + p[0][2] + p[0][3] + p[1][2] + p[1][3];
            #pragma unroll
            for (int nt = 0; nt < 32; ++nt) {
                acc[nt][0] *= a0;
                acc[nt][1] *= a0;
                acc[nt][2] *= a1;
                acc[nt][3] *= a1;
            }
            uint32_t pa[4] = {pack_h2(p[0][0], p[0][1]), pack_h2(p[0][2], p[0][3]),
                              pack_h2(p[1][0], p[1][1]), pack_h2(p[1][2], p[1][3])};
            #pragma unroll
            for (int nt = 0; nt < 32; nt += 2) {
                uint32_t bv[4];
                ldsm_x4_trans(bv, &vt[(lane % 8) + 8 * ((lane / 8) & 1)][nt * 8 + 8 * (lane / 16)]);
                mma16816(acc[nt], pa, bv);
                mma16816(acc[nt + 1], pa, bv + 2);
            }
        }
        __syncthreads();
    }
    if (!live) {
        return;
    }
    l0 += __shfl_xor_sync(RC_FULL_MASK, l0, 1);
    l0 += __shfl_xor_sync(RC_FULL_MASK, l0, 2);
    l1 += __shfl_xor_sync(RC_FULL_MASK, l1, 1);
    l1 += __shfl_xor_sync(RC_FULL_MASK, l1, 2);
    float f0 = 1.f / l0, f1 = 1.f / l1;
    float *out = DP(float, 6) + ((size_t)j * nq + (size_t)kv * rep) * 256;
    #pragma unroll
    for (int nt = 0; nt < 32; ++nt) {
        int d = nt * 8 + 2 * c;
        if (g < rep) {
            out[(size_t)g * 256 + d] = acc[nt][0] * f0;
            out[(size_t)g * 256 + d + 1] = acc[nt][1] * f0;
        }
        if (g + 8 < rep) {
            out[(size_t)(g + 8) * 256 + d] = acc[nt][2] * f1;
            out[(size_t)(g + 8) * 256 + d + 1] = acc[nt][3] * f1;
        }
    }
}

/* RECLANK_GPU_QSA_TC: 1 (the default) k_attn_qsa_tc with q as two float16
 * planes; 2 one plane; 0 k_attn_qsa_mt (float32). For the int8 and TQ6 forms
 * of the cache; the others keep k_attn_qsa_mt. */
static int qsa_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_QSA_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}

/* RECLANK_GPU_PART_TC: 1 (the default) k_attn_part_tc for GP_ATTN_QSA of
 * one query (int8 or TQ6); 0 k_attn_part (a test). */
static int part_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_PART_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}

/* RECLANK_GPU_DENSE_TC: 1 (the default) k_attn_dense_tc for a record of
 * maxsel 0; 0 k_attn_qsa_tc (a test). */
static int dense_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_DENSE_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}


template <int HD>
struct flash3_dims {
    static constexpr int NS = HD / 256;
    static constexpr int KLD = HD + 8;
    static constexpr int VLD = HD + 4;
    static constexpr int QLD = HD + 8;
};

template <int HD, int HB, int ST>
__global__ void __launch_bounds__(32 * HB * (HD / 256))
k_flash_f32h(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    typedef flash3_dims<HD> D;
    extern __shared__ float fsm3[];
    float *kbuf = fsm3;                                   /* ST x FK3 x KLD */
    float *vbuf = kbuf + ST * FK3 * D::KLD;               /* ST x FK3 x VLD */
    __half *qs = (__half *)(vbuf + ST * FK3 * D::VLD);    /* HB x 16 x QLD (HD 512) */
    int qh = DI(5), kvh = DI(6), t = DI(8);
    int window = DI(11);
    int64_t pos = di(r, e, 9), hs = di(r, e, 10);
    const float *q = DP(const float, 0);
    float *out = DP(float, 4);
    int G = qh / kvh, kv = blockIdx.x, j0 = blockIdx.y * 16;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int hl = warp / D::NS, dsl = warp % D::NS;
    int h = kv * G + blockIdx.z * HB + hl;
    int g = lane / 4, c = lane % 4;
    const float *kf = DP(const float, 1) + (size_t)kv * hs;
    const float *vf = DP(const float, 2) + (size_t)kv * hs;
    const size_t frs = kv_rs((size_t)hs, kvh, HD);
    bool live0 = j0 + g < t, live1 = j0 + g + 8 < t;
    int64_t p0 = pos + j0 + g, p1 = p0 + 8;
    int64_t e0 = p0, e1 = p1;
    int jl = min(t, j0 + 16) - 1;
    int64_t first = window > 0 ? pos + j0 - window + 1 : 0;
    if (first < 0) {
        first = 0;
    }
    int64_t last = pos + jl;
    int nsteps = (int)((last - first) / FK3) + 1;

    auto load = [&](int stage, int64_t k0) {
        int kn = (int)min((int64_t)FK3, last - k0 + 1);
        float *kd = kbuf + stage * FK3 * D::KLD, *vd = vbuf + stage * FK3 * D::VLD;
        for (int x = threadIdx.x; x < FK3 * HD / 4; x += blockDim.x) {
            int a = x / (HD / 4), d = (x % (HD / 4)) * 4;
            bool ok = a < kn;
            size_t o2 = (size_t)(ok ? k0 + a : k0) * frs + d;
            cp_async16_z(kd + a * D::KLD + d, kf + o2, ok ? 16 : 0);
            cp_async16_z(vd + a * D::VLD + d, vf + o2, ok ? 16 : 0);
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
        const float *kd = kbuf + st * FK3 * D::KLD;
        const float *vd = vbuf + st * FK3 * D::VLD;
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
                const float *kr = kd + (nt * 8 + g) * D::KLD + kk + 2 * c;
                float2 x0 = *(const float2 *)kr, x1 = *(const float2 *)(kr + 8);
                uint32_t b[2] = {pack_h2(x0.x, x0.y), pack_h2(x1.x, x1.y)};
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
            const float *vr = vd + (2 * c) * D::VLD + d;
            uint32_t b[2] = {pack_h2(vr[0], vr[D::VLD]),
                             pack_h2(vr[8 * D::VLD], vr[9 * D::VLD])};
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
