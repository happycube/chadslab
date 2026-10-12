/* ---------- QSA (Qwen3.8): the selection of the keys (backends/cpu/csrc/common/qsa.c) ----------
 * GP_QSA_SELECT: iq, ik, idxk, blk, qn, kn, cos, sin, pos, t, heads, d,
 * ratio, budget, rot, theta, eps, sel, cnt, maxsel, scratch, nbmax, qpos,
 * sec (qpos: the M-RoPE positions of the rows, see qsa_rpos). idxk and blk
 * keep float16 values, as on the CPU. Three
 * kernels: the raw keys to idxk; the key of each block that is complete now;
 * the selection of each query (one block of QSA_T threads for each query;
 * scratch has nbmax 64-bit keys for each query of a launch: the queries go
 * in launches of QSA_ROWS, so the scratch is QSA_ROWS * nbmax keys at most,
 * 0.27 GB at 262144 positions, in place of t * nbmax: 1.9 GB for a group of
 * 4096 rows at 225K, which took the last of the memory of the GPU). */
#define QSA_T 1024
#define QSA_ROWS 512
/* The start of the scratch: the queries of a launch after their norm and
 * RoPE as two float16 planes (k_qsa_qprep, for k_qsa_score_tc: heads * d
 * <= 8 * 256 values); the keys of the queries follow. */
#define QSA_QBUF ((size_t)QSA_ROWS * 8 * 256 * 2 * 2)

/* The queries of a launch of k_qsa_query (the rows of the scratch). */
extern "C" int rc_gpu_qsa_rows(void)
{
    return QSA_ROWS;
}

/* The bytes of the scratch before the keys (QSA_QBUF). */
extern "C" int64_t rc_gpu_qsa_extra(void)
{
    return (int64_t)QSA_QBUF;
}

/* RMS norm of d values (d <= 256) times w, then RoPE on the first rot
 * values; x in shared memory, the threads of the block (or of one warp: the
 * sums use the threads that call). */
__device__ void qsa_norm_rope_block(float *x, const float *w, int d, float eps, int rot,
                                    const float *c, const float *sn)
{
    float ss = 0.f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        ss += x[i] * x[i];
    }
    ss = block_sum(ss);
    float inv = 1.f / sqrtf(ss / (float)d + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        x[i] = x[i] * inv * w[i];
    }
    __syncthreads();
    int half = rot / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float a = x[i], b = x[i + half];
        x[i] = a * c[i] - b * sn[i];
        x[i + half] = b * c[i + half] + a * sn[i + half];
    }
    __syncthreads();
}

__global__ void k_qsa_copy(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int d = DI(11);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)DI(9) * d) {
        DP(__half, 2)[di(r, e, 8) * d + i] = __float2half_rn(DP(const float, 1)[i]);
    }
}

/* Block x makes the key of block pos / ratio + x, if it is complete. */
__global__ void k_qsa_blocks(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    __shared__ float x[256], c[256], sn[256];
    int64_t pos = di(r, e, 8), n = pos + DI(9);
    int d = DI(11), ratio = DI(12), rot = DI(14);
    int64_t b = pos / ratio + blockIdx.x;
    if (b >= n / ratio) {
        return;
    }
    const __half *idxk = DP(const __half, 2);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        float sum = 0.f;
        for (int q = 0; q < ratio; ++q) {
            sum += __half2float(idxk[(size_t)(b * ratio + q) * d + i]);
        }
        x[i] = sum / (float)ratio;
    }
    double theta = (double)df(r, e, 15);
    const int32_t *qpos = DP(const int32_t, 22);
    int sec = DI(23);
    for (int i = threadIdx.x; i < rot / 2; i += blockDim.x) {
        int64_t p = b * ratio;
        if (qpos != NULL) {
            int a = (i % 3 == 1 && i < 3 * (sec & 255)) ? 1 : (i % 3 == 2 && i < 3 * (sec >> 8)) ? 2 : 0;
            p = qpos[p * 3 + a];
        }
        double f = (double)p / pow(theta, (double)(2 * i) / (double)rot);
        c[i] = c[i + rot / 2] = (float)cos(f);
        sn[i] = sn[i + rot / 2] = (float)sin(f);
    }
    __syncthreads();
    qsa_norm_rope_block(x, DP(const float, 5), d, df(r, e, 16), rot, c, sn);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        DP(__half, 3)[(size_t)b * d + i] = __float2half_rn(x[i]);
    }
}

/* The inclusive prefix sum of v over the block (QSA_T threads). */
__device__ int qsa_scan(int v, int *tmp)
{
    int lane = threadIdx.x % 32, w = threadIdx.x / 32;
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(RC_FULL_MASK, v, o);
        v += lane >= o ? y : 0;
    }
    __syncthreads();
    if (lane == 31) {
        tmp[w] = v;
    }
    __syncthreads();
    if (w == 0) {
        int t = lane < QSA_T / 32 ? tmp[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            int y = __shfl_up_sync(RC_FULL_MASK, t, o);
            t += lane >= o ? y : 0;
        }
        tmp[lane] = t;
    }
    __syncthreads();
    return v + (w > 0 ? tmp[w - 1] : 0);
}

/* k_qsa_qprep and k_qsa_score_tc: the scores of the blocks of a launch of
 * a large group on the tensor cores (4 heads of 128 values), in place of the
 * loop of k_qsa_query, which scores the blocks of one query in each block
 * with float32 dots and shuffles: it read all the block keys again for each
 * query (243 ms a layer for a group of 4096 rows at 196K positions, 12 s of
 * each 16K there). Block row k_qsa_qprep: query j0 + row after its norm and
 * RoPE, as two float16 planes (hi, lo: about 22 bits), at the start of the
 * scratch. */
__global__ void __launch_bounds__(128) k_qsa_qprep(const gp_rec *r, const int64_t *e, int j0)
{
    PDL_START();
    __shared__ float q[8 * 256];
    int jl = blockIdx.x, j = j0 + jl, heads = DI(10), d = DI(11), rot = DI(14);
    const float *iq = DP(const float, 0) + (size_t)j * heads * d;
    for (int i = threadIdx.x; i < heads * d; i += blockDim.x) {
        q[i] = iq[i];
    }
    __syncthreads();
    const float *cs = DP(const float, 6) + (size_t)j * rot, *sn = DP(const float, 7) + (size_t)j * rot;
    for (int h = 0; h < heads; ++h) {
        qsa_norm_rope_block(q + h * d, DP(const float, 4), d, df(r, e, 16), rot, cs, sn);
    }
    __half *qb = (__half *)DP(uint8_t, 20) + (size_t)jl * heads * d * 2;
    for (int i = threadIdx.x; i < heads * d; i += blockDim.x) {
        __half hi = __float2half_rn(q[i]);
        qb[i] = hi;
        qb[heads * d + i] = __float2half_rn(q[i] - __half2float(hi));
    }
}

/* Block (x, y): the 16 queries 16 y .. 16 y + 15 of the launch (rows rows),
 * the tiles of QS_KEYS block keys x, x + gridDim.x, ... up to the blocks of
 * its last query (the grid does not follow the position: the graph keeps
 * it). Warp w: queries 4 w .. 4 w + 3, their 4 heads the 16 rows of its m16
 * tile (row 4 query + head), the fragments of both planes in registers. For
 * each key: relu of each head, the sum of the heads (lanes 4 apart), times
 * 1 / sqrt(d); the key (score bits << 32 | block) of each query that sees
 * the block, as k_qsa_query writes it. */
#define QS_KEYS 128
#define QS_LD 136
#define QS_GX 16
__global__ void __launch_bounds__(128) k_qsa_score_tc(const gp_rec *r, const int64_t *e, int j0, int rows)
{
    PDL_START();
    __shared__ __align__(16) __half kt[QS_KEYS][QS_LD];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane / 4, c = lane % 4;
    const int heads = 4, d = 128;
    int ratio = DI(12);
    int64_t pos = di(r, e, 8), nbs = di(r, e, 21);
    int q0 = blockIdx.y * 16 + warp * 4;
    int jlast = min(rows, (int)blockIdx.y * 16 + 16) - 1;
    int64_t nbb = (pos + j0 + jlast + 1) / ratio;      /* the blocks of the last query */
    const __half *qb = (const __half *)DP(uint8_t, 20);
    size_t qrow = (size_t)heads * d * 2;
    uint32_t ah[8][4], al[8][4];
    {
        int qa = q0 + g / 4, ha = g % 4, qc = q0 + (g + 8) / 4, hc = (g + 8) % 4;
        const __half *pa = qb + (size_t)qa * qrow + (size_t)ha * d;
        const __half *pc = qb + (size_t)qc * qrow + (size_t)hc * d;
        bool va = qa < rows, vc = qc < rows;
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int o = kk * 16 + 2 * c;
            ah[kk][0] = va ? *(const uint32_t *)(pa + o) : 0u;
            ah[kk][1] = vc ? *(const uint32_t *)(pc + o) : 0u;
            ah[kk][2] = va ? *(const uint32_t *)(pa + o + 8) : 0u;
            ah[kk][3] = vc ? *(const uint32_t *)(pc + o + 8) : 0u;
            al[kk][0] = va ? *(const uint32_t *)(pa + heads * d + o) : 0u;
            al[kk][1] = vc ? *(const uint32_t *)(pc + heads * d + o) : 0u;
            al[kk][2] = va ? *(const uint32_t *)(pa + heads * d + o + 8) : 0u;
            al[kk][3] = vc ? *(const uint32_t *)(pc + heads * d + o + 8) : 0u;
        }
    }
    const __half *blk = DP(const __half, 3);
    uint64_t *keys = (uint64_t *)(DP(uint8_t, 20) + QSA_QBUF);
    float scale = 1.f / sqrtf((float)d);
    int qa = q0 + g / 4, qc = q0 + (g + 8) / 4;
    int64_t nba = qa < rows ? (pos + j0 + qa + 1) / ratio : 0;
    int64_t nbc = qc < rows ? (pos + j0 + qc + 1) / ratio : 0;
    for (int64_t b0 = (int64_t)blockIdx.x * QS_KEYS; b0 < nbb; b0 += (int64_t)gridDim.x * QS_KEYS) {
        __syncthreads();
        for (int i = threadIdx.x; i < QS_KEYS * 16; i += blockDim.x) {
            int kr = i / 16, kc = (i % 16) * 8;
            int64_t b = b0 + kr;
            *(uint4 *)&kt[kr][kc] = b < nbb ? *(const uint4 *)(blk + (size_t)b * d + kc)
                                            : make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
        #pragma unroll 2
        for (int nt = 0; nt < QS_KEYS / 8; ++nt) {
            float sc[4] = {0.f, 0.f, 0.f, 0.f};
            #pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                uint32_t bb[2];
                bb[0] = *(const uint32_t *)&kt[nt * 8 + g][kk * 16 + 2 * c];
                bb[1] = *(const uint32_t *)&kt[nt * 8 + g][kk * 16 + 2 * c + 8];
                mma16816(sc, ah[kk], bb);
                mma16816(sc, al[kk], bb);
            }
            #pragma unroll
            for (int u = 0; u < 4; ++u) {
                sc[u] = sc[u] > 0.f ? sc[u] : 0.f;
                sc[u] += __shfl_xor_sync(RC_FULL_MASK, sc[u], 4);
                sc[u] += __shfl_xor_sync(RC_FULL_MASK, sc[u], 8);
            }
            if ((g & 3) == 0) {
                int64_t b = b0 + nt * 8 + 2 * c;
                if (b < nba) {
                    keys[(size_t)qa * nbs + b] = ((uint64_t)__float_as_uint(sc[0] * scale) << 32) | (uint64_t)b;
                }
                if (b + 1 < nba) {
                    keys[(size_t)qa * nbs + b + 1] = ((uint64_t)__float_as_uint(sc[1] * scale) << 32) |
                                                     (uint64_t)(b + 1);
                }
                if (b < nbc) {
                    keys[(size_t)qc * nbs + b] = ((uint64_t)__float_as_uint(sc[2] * scale) << 32) | (uint64_t)b;
                }
                if (b + 1 < nbc) {
                    keys[(size_t)qc * nbs + b + 1] = ((uint64_t)__float_as_uint(sc[3] * scale) << 32) |
                                                     (uint64_t)(b + 1);
                }
            }
        }
    }
}

/* RECLANK_GPU_QSA_SCORE_TC: 1 (the default) k_qsa_score_tc for a group (or
 * a step) with 4 heads of 128; 0 the loop of k_qsa_query. At 200K positions
 * the loop took 4 ms a layer for an MTP verify group of 4 rows (one SM for
 * each query): 28% of the decode. */
static int qsa_score_tc_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_GPU_QSA_SCORE_TC");
        on = v == NULL ? 1 : atoi(v);
    }
    return gg_caps.tc ? on : 0;   /* the tensor cores: sm_80 */
}

/* Block j selects the keys of query j0 + j (the method of qsa_select_body:
 * the scores of the blocks as keys (score bits << 32 | block), the top budget
 * keys by a radix select of 8 bits at a time, then the positions in order);
 * its keys are row j of the scratch. */
template <int SCORED>
__global__ void __launch_bounds__(QSA_T) k_qsa_query(const gp_rec *r, const int64_t *e, int j0)
{
    PDL_START();
    __shared__ float q[8 * 256];
    __shared__ int hist[256];
    __shared__ int tmp[32];
    __shared__ uint64_t prefix_s;
    __shared__ int need_s;
    int j = j0 + blockIdx.x, heads = DI(10), d = DI(11), ratio = DI(12), budget = DI(13);
    int rot = DI(14), maxsel = DI(19);
    int64_t pj = di(r, e, 8) + j, nb = (pj + 1) / ratio;
    int32_t *cnt = DP(int32_t, 18);
    if (nb <= budget) {
        if (threadIdx.x == 0) {
            cnt[j] = -1;
        }
        return;
    }
    uint64_t *keys = (uint64_t *)(DP(uint8_t, 20) + QSA_QBUF) + (size_t)blockIdx.x * di(r, e, 21);
    if (!SCORED) {
        /* (SCORED: k_qsa_score_tc wrote the keys) */
        const float *iq = DP(const float, 0) + (size_t)j * heads * d;
        for (int i = threadIdx.x; i < heads * d; i += blockDim.x) {
            q[i] = iq[i];
        }
        __syncthreads();
        const float *cs = DP(const float, 6) + (size_t)j * rot, *sn = DP(const float, 7) + (size_t)j * rot;
        for (int h = 0; h < heads; ++h) {
            qsa_norm_rope_block(q + h * d, DP(const float, 4), d, df(r, e, 16), rot, cs, sn);
        }
    }
    const __half *blk = DP(const __half, 3);
    float scale = 1.f / sqrtf((float)d);
    /* one warp for each block key: the lanes split the values */
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int64_t b = SCORED ? nb : warp; b < nb; b += QSA_T / 32) {
        const __half *kb = blk + (size_t)b * d;
        float score = 0.f;
        for (int h = 0; h < heads; ++h) {
            float dot = 0.f;
            for (int i = lane; i < d; i += 32) {
                dot += q[h * d + i] * __half2float(kb[i]);
            }
            for (int o = 16; o > 0; o >>= 1) {
                dot += __shfl_xor_sync(RC_FULL_MASK, dot, o);
            }
            score += dot > 0.f ? dot : 0.f;
        }
        if (lane == 0) {
            score *= scale;
            keys[b] = ((uint64_t)__float_as_uint(score) << 32) | (uint64_t)b;
        }
    }
    __syncthreads();
    /* The budget-th largest key: 8 bits at a time from the top. */
    if (threadIdx.x == 0) {
        prefix_s = 0;
        need_s = budget;
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        for (int i = threadIdx.x; i < 256; i += blockDim.x) {
            hist[i] = 0;
        }
        __syncthreads();
        uint64_t prefix = prefix_s, mask = shift == 56 ? 0 : ~0ull << (shift + 8);
        for (int64_t b = threadIdx.x; b < nb; b += blockDim.x) {
            uint64_t k = keys[b];
            if ((k & mask) == prefix) {
                atomicAdd(&hist[(k >> shift) & 255], 1);
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int need = need_s, digit = 255;
            while (hist[digit] < need) {
                need -= hist[digit];
                --digit;
            }
            need_s = need;
            prefix_s = prefix | ((uint64_t)digit << shift);
        }
        __syncthreads();
    }
    uint64_t thr = prefix_s;            /* the keys >= thr are the budget best */
    int32_t *out = DP(int32_t, 17) + (size_t)j * maxsel;
    int base = 0;
    for (int64_t b0 = 0; b0 < nb; b0 += blockDim.x) {
        int64_t b = b0 + threadIdx.x;
        int keep = b < nb && keys[b] >= thr;
        int incl = qsa_scan(keep, tmp);
        if (keep) {
            for (int q2 = 0; q2 < ratio; ++q2) {
                out[(base + incl - 1) * ratio + q2] = (int32_t)(b * ratio + q2);
            }
        }
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) {
            tmp[0] = incl;
        }
        __syncthreads();
        base += tmp[0];
        __syncthreads();
    }
    int c2 = base * ratio;
    for (int64_t p = nb * ratio + threadIdx.x; p <= pj; p += blockDim.x) {
        out[c2 + (p - nb * ratio)] = (int32_t)p;
    }
    if (threadIdx.x == 0) {
        cnt[j] = (int32_t)(c2 + (pj + 1 - nb * ratio));
    }
}

/* GP_KQ_MULTI: x, cols, t, n, then (w, type, rows, out) for n matrices on
 * the same rows of x: GP_KQ_LINEAR of each, in one launch. */
__global__ void k_kq_multi(const gp_rec *r, const int64_t *e, int skip_bt = 0)
{
    PDL_START();
    int row = blockIdx.x * KQ_RPB + threadIdx.x / 32;
    int n = DI(3), cols = DI(1), t = DI(2), m = 0;
    while (m < n && row >= DI(6 + 4 * m)) {
        row -= DI(6 + 4 * m);
        ++m;
    }
    if (m >= n) {
        return;
    }
    int type = DI(5 + 4 * m), rows = DI(6 + 4 * m);
    if (skip_bt && bt_shape(type, rows, cols)) {
        return;         /* k_kq_bf12_tc */
    }
    const uint8_t *w = DP(const uint8_t, 4 + 4 * m) + (size_t)row * kq_row_bytes(type, cols);
    for (int j = 0; j < t; ++j) {
        float v = kq_row(type, w, DP(const float, 0) + (size_t)j * cols, cols);
        if (threadIdx.x % 32 == 0) {
            DP(float, 7 + 4 * m)[(size_t)j * rows + row] = v;
        }
    }
}
