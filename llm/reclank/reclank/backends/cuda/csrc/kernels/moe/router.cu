/* ---------- the router ----------
 * The operands of GP_ROUTER:
 *
 *     x, scale, proj, per_expert, hidden, experts, top_k, eps, hscale, val,
 *     idx, r, logits
 *
 * Three kernels do the three steps of gemma_router_body:
 *
 * 1. the norm of x into r;
 * 2. the logit of each expert;
 * 3. the softmax, and the choice of the top_k experts. */
__global__ void k_router_norm(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    const float *x = DP(const float, 0);
    const float *scale = DP(const float, 1);
    float *rv = DP(float, 11);
    int hidden = DI(4);
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

__global__ void k_router_logits(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int ex = blockIdx.x * ROWS_PER_BLOCK + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int hidden = DI(4);
    if (ex >= DI(5)) {
        return;
    }
    const float *pe = DP(const float, 2) + (size_t)ex * hidden;
    const float *rv = DP(const float, 11);
    float s = 0.f;
    for (int k = lane; k < hidden; k += 32) {
        s += rv[k] * pe[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, o);
    }
    if (lane == 0) {
        DP(float, 12)[ex] = s;
    }
}

/* One warp: the softmax, the top_k experts, and their weights. Each lane
 * keeps the logits of experts lane, lane + 32, and so on. At an equal
 * probability, the expert with the lower index wins, as in
 * gemma_router_body. At most 256 experts. */
__device__ void router_top_warp(const float *logits, const float *per_expert, float *val,
                                int *idx, int experts, int top_k)
{
    int lane = threadIdx.x;
    float p[8];
    float m = -INFINITY;
    for (int k = 0; k < 8; ++k) {
        int x = lane + 32 * k;
        p[k] = x < experts ? logits[x] : -INFINITY;
        m = fmaxf(m, p[k]);
    }
    for (int o = 16; o > 0; o >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, o));
    }
    float sum = 0.f;
    for (int k = 0; k < 8; ++k) {
        p[k] = lane + 32 * k < experts ? expf(p[k] - m) : 0.f;
        sum += p[k];
    }
    for (int o = 16; o > 0; o >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, o);
    }
    float invs = 1.0f / sum;
    for (int k = 0; k < 8; ++k) {
        p[k] = lane + 32 * k < experts ? p[k] * invs : -2.0f;
    }
    float vs = 0.f;
    for (int j = 0; j < top_k; ++j) {
        /* The best expert of this lane, then of the warp. */
        float bv = -3.0f;
        int bx = 1 << 30;
        for (int k = 0; k < 8; ++k) {
            if (p[k] > bv) {
                bv = p[k];
                bx = lane + 32 * k;
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            float ov = __shfl_xor_sync(RC_FULL_MASK, bv, o);
            int ox = __shfl_xor_sync(RC_FULL_MASK, bx, o);
            if (ov > bv || (ov == bv && ox < bx)) {
                bv = ov;
                bx = ox;
            }
        }
        if (lane == 0) {
            val[j] = bv;
            idx[j] = bx;
        }
        vs += bv;
        if ((bx & 31) == lane) {
            p[bx / 32] = -1.0f;
        }
    }
    if (lane == 0) {
        float invv = 1.0f / vs;
        for (int j = 0; j < top_k; ++j) {
            val[j] = val[j] * invv * per_expert[idx[j]];
        }
    }
}

__global__ void k_router_top(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    router_top_warp(DP(const float, 12), DP(const float, 3), DP(float, 9), DP(int, 10),
                    DI(5), DI(6));
}
