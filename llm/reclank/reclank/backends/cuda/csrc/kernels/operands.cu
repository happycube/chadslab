/* ---------- the operands ---------- */

__device__ __forceinline__ int64_t di(const gp_rec *r, const int64_t *e, int k)
{
    return r->tag[k] == GP_T_SLOT ? e[r->v[k]] : r->v[k];
}

__device__ __forceinline__ float df(const gp_rec *r, const int64_t *e, int k)
{
    return __uint_as_float((uint32_t)di(r, e, k));
}

#define DP(T, k) ((T *)(intptr_t)di(r, e, (k)))
#define DI(k) ((int)di(r, e, (k)))

/* The pool of HotCache (gpumm.ExpertPool): expert blocks on the GPU in
 * segments of GG_POOL_K blocks that come and go. A slot value KQH_WARM + i is
 * block i: segment i / GG_POOL_K, place i % GG_POOL_K, at the stride of its
 * part (0, 1, 2: gate, up, down of Qwen; gate and up, down of Gemma). The
 * graphs read the table at run time (rc_gpu_set_pool, recurrent/deltanet.cu). */
#define KQH_WARM (1 << 24)
#define GG_POOL_K 8
__device__ const uint8_t *const *gg_pool_seg;   /* [3][gg_pool_nseg] */
__device__ int gg_pool_nseg;
__device__ size_t gg_pool_stride[3];

/* the bytes of part of the block of slot (slot >= KQH_WARM) */
__device__ __forceinline__ const uint8_t *pool_block(int slot, int part)
{
    int i = slot - KQH_WARM;
    return gg_pool_seg[part * gg_pool_nseg + i / GG_POOL_K] + (size_t)(i % GG_POOL_K) * gg_pool_stride[part];
}

/* ---------- the caches with keys only (GPUKV konly) ----------
 * A global layer of Gemma 4 (k_eq_v) has K = c rope(V): its value is the
 * raw key with a norm of no weight, its key that row times the k_norm
 * weight, a constant c, then turned. The cache then keeps the int16 keys
 * and their scales (ks), and for the values only the scales vs = ks / c;
 * the record gives the address of the keys for the values (vq == kq). A
 * value is the key turned back: value i of a turned pair (i < rd, its
 * partner i + half; or half <= i < half + rd, its partner i - half) is
 *
 *     v_i = k_i cos + k_{i + half} sin,  v_{i + half} = k_{i + half} cos - k_i sin
 *
 * (the angle of position p and frequency i, each k with its vs), and every
 * other value is k_i with its vs. The cos and sin of p theta_f come from
 * two tables: hi[p >> 8][f] (of (p >> 8) 256 theta_f) and lo[p & 255][f],
 * made in float64 (rc_gpu_ko_table), joined by the sum of the angles. */
struct ko_tab {
    const float2 *hi, *lo;      /* (cos, sin), rd of each row */
    int rd, half, nhi;          /* the turned pairs, head_dim / 2, the rows of hi */
};
__device__ ko_tab g_ko;

__device__ __forceinline__ float2 ko_cs(int64_t p, int f)
{
    float2 a = g_ko.hi[(p >> 8) * g_ko.rd + f], b = g_ko.lo[(p & 255) * g_ko.rd + f];
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

/* true when the 8 values from i (a multiple of 8) are turned */
__device__ __forceinline__ bool ko_turned(int i)
{
    return (i < g_ko.half ? i : i - g_ko.half) < g_ko.rd;
}

/* the index of the partner of the 8 values from i */
__device__ __forceinline__ int ko_partner(int i)
{
    return i < g_ko.half ? i + g_ko.half : i - g_ko.half;
}

/* x: the 8 values from i of a key row of position p (times their vs), y:
 * those of the partner (times theirs). x becomes the 8 values of V. */
__device__ __forceinline__ void ko_rot8(float *x, const float *y, int64_t p, int i)
{
    int f0 = i < g_ko.half ? i : i - g_ko.half;
    float sg = i < g_ko.half ? 1.f : -1.f;
    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        float2 cs = ko_cs(p, f0 + u);
        x[u] = x[u] * cs.x + sg * y[u] * cs.y;
    }
}

/* one value: x the value i of a key row of position p (times its vs), y
 * that of its partner */
__device__ __forceinline__ float ko_rot1(float x, float y, int64_t p, int i)
{
    int f = i < g_ko.half ? i : i - g_ko.half;
    float2 cs = ko_cs(p, f);
    return x * cs.x + (i < g_ko.half ? 1.f : -1.f) * y * cs.y;
}

/* the 8 int16 values of u (one 16-byte load) times sc */
__device__ __forceinline__ void ko_i16x8(uint4 u, float sc, float *out)
{
    uint32_t w[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        out[2 * t] = (float)(int16_t)(w[t] & 0xffff) * sc;
        out[2 * t + 1] = (float)(int16_t)(w[t] >> 16) * sc;
    }
}

/* The sum of v over the block. All threads get the result. The block has at
 * most 1024 threads. */
__device__ float block_sum(float v)
{
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(RC_FULL_MASK, v, o);
    }
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) {
        part[w] = v;
    }
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    v = lane < nw ? part[lane] : 0.f;
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(RC_FULL_MASK, v, o);
    }
    return v;
}

/* The int8 x of the one-token products (k_kq_quant_x): 8 adjacent lanes
 * hold the 32 values of block bi, 4 values each (sub = the lane % 8). All
 * lanes of the warp must call it; a lane with live false gives zeros and
 * writes nothing. */
__device__ __forceinline__ void quant8_block(float4 v, bool live, size_t bi, int sub, int8_t *q,
                                             float *xs, float *xsum)
{
    if (!live) {
        v = make_float4(0.f, 0.f, 0.f, 0.f);
    }
    float m = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
    for (int o = 4; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, o));
    }
    float s2 = m / 127.0f;
    int a = s2 > 0.f ? __float2int_rn(v.x / s2) : 0, b = s2 > 0.f ? __float2int_rn(v.y / s2) : 0;
    int c = s2 > 0.f ? __float2int_rn(v.z / s2) : 0, d = s2 > 0.f ? __float2int_rn(v.w / s2) : 0;
    int sum = a + b + c + d;
    for (int o = 4; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    if (!live) {
        return;
    }
    char4 c4;
    c4.x = (signed char)a;
    c4.y = (signed char)b;
    c4.z = (signed char)c;
    c4.w = (signed char)d;
    *(char4 *)(q + bi * 32 + sub * 4) = c4;
    if (sub == 0) {
        xs[bi] = s2;
        xsum[bi] = s2 * (float)sum;
    }
}

__device__ float block_max(float v)
{
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(RC_FULL_MASK, v, o));
    }
    int w = threadIdx.x / 32, lane = threadIdx.x % 32;
    __syncthreads();
    if (lane == 0) {
        part[w] = v;
    }
    __syncthreads();
    int nw = (blockDim.x + 31) / 32;
    v = lane < nw ? part[lane] : -INFINITY;
    for (int o = 16; o > 0; o >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(RC_FULL_MASK, v, o));
    }
    return v;
}
