/* FP4-Local's contract on the prompt rows (docs/FP4LOCAL_PLAN.md 2.2: b12x's A4 prefill), for
 * GP_KQ_GROUP_MOE on KQ_NVX experts (gg_moe_rot 8): the rows of h quantized to NVFP4 with the
 * layer's gs1 (the threshold quantizer of quantize_block_fp4), the int8 tensor cores on the codes
 * (twice the E2M1 values) with a scale for each 16 of x (kt_tile A4), g, u = bf16(alpha1 acc), the
 * act bf16(bf16(g sigma(g)) u), requantized with the expert's gs2, the down product, each pair's row
 * y = bf16(alpha2 acc), and the sum of w y in float32 (k_qmoe_sum). The A4 global scales are in
 * bytes 4-7 of the heads of the KQ_NVX groups (gate and up: gs1, down: gs2; fp4local_st). The shared
 * expert runs outside (KQ_LINEAR, B12X_GLU), and MOE_B12X_OUT adds it to the routed sum.
 *
 * The scales of x go where k_kq_quant_x puts xs and xsum (a float for each block of 32): the scale
 * of the first 16 of a block in xs, that of the second in xsum. */

/* the nearest E4M3 value of v (to nearest even; subnormals; saturating at 448), v >= 0 */
__device__ __forceinline__ float rc_e4m3_rne(float v)
{
    if (!(v > 0.f)) {
        return 0.f;
    }
    if (v >= 448.f) {
        return 448.f;
    }
    if (v < 0.015625f) {
        return rintf(v * 512.f) * (1.f / 512.f);        /* the subnormals: steps of 2^-9 */
    }
    int e;
    float m = frexpf(v, &e);            /* v = m 2^e, m in [0.5, 1) */
    float q = rintf((m * 2.f - 1.f) * 8.f);
    return ldexpf(1.f + q * 0.125f, e - 1);
}

/* the A4 form of 16 values of x (x in bfloat16 values): their scale, and twice the E2M1 code of
 * each (int8) */
__device__ __forceinline__ float rc_a4_16(const float *x, float gs, int8_t *q)
{
    float v[16], amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = rc_bf16r(x[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    float sf = rc_e4m3_rne(fminf(__fdiv_rn(__fmul_rn(amax, gs), 6.f), 448.f));
    uint32_t w[4] = {0, 0, 0, 0};
    if (sf > 0.f && gs != 0.f) {
        float vs = __fdiv_rn(sf, gs);
        float t1 = vs * 0.25f, t2 = vs * 0.75f, t3 = vs * 1.25f, t4 = vs * 1.75f, t5 = vs * 2.5f, t6 = vs * 3.5f,
              t7 = vs * 5.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            float a = fabsf(v[i]);
            int c = a > t7 ? 12 : a >= t6 ? 8 : a > t5 ? 6 : a >= t4 ? 4 : a > t3 ? 3 : a >= t2 ? 2 : a > t1 ? 1 : 0;
            c = v[i] < 0.f ? -c : c;
            w[i / 4] |= (uint32_t)(uint8_t)(int8_t)c << (8 * (i % 4));
        }
    }
    *(uint4 *)q = make_uint4(w[0], w[1], w[2], w[3]);
    return sf;
}

/* the A4 global scale in a KQ_NVX matrix's head */
__device__ __forceinline__ float nvx_a4_gs(const uint8_t *w)
{
    return *(const float *)(w + 4);
}

/* the expert of pair row p of the sort (start[x] <= p < start[x + 1]) */
__device__ __forceinline__ int qmoe_expert_of(const int *start, int E, int p)
{
    int lo = 0, hi = E + 1;
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if (start[m] <= p) {
            lo = m;
        } else {
            hi = m;
        }
    }
    return lo;
}

/* the rows of h of GP_KQ_GROUP_MOE in the A4 form (gs1 of the gate matrix of the first tile's expert;
 * every expert of the layer has the same) */
__global__ void k_a4_quant_h(const gp_rec *r, const int64_t *e, int8_t *q, float *slo, float *shi)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int t = DI(3), E = DI(5), hidden = DI(6);
    int nt = DP(const int, 18)[0];
    if (nt == 0 || w.tiles[0] >= E) {
        return;
    }
    float gs1 = nvx_a4_gs((const uint8_t *)(intptr_t)DP(const int64_t, 8)[w.tiles[0]]);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int nh = hidden / 16;
    if (i >= (int64_t)t * nh) {
        return;
    }
    int row = (int)(i / nh), h = (int)(i % nh);
    float sf = rc_a4_16(DP(const float, 0) + (size_t)row * hidden + 16 * h, gs1, q + (size_t)row * hidden + 16 * h);
    (h % 2 ? shi : slo)[(size_t)row * (hidden / 32) + h / 2] = sf;
}

/* the act of the routed pairs: g, u = bf16(acc 0.5 / gs1) (the tiles' acc is 2 g sum(q sf E2M1 s)),
 * h = bf16(bf16(g sigma(g)) u) */
__global__ void k_qmoe_act_a4(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int E = DI(5), inner = DI(7);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)w.start[E] * inner) {
        return;
    }
    int p = (int)(i / inner), c = (int)(i % inner);
    int x = qmoe_expert_of(w.start, E, p);
    float sc = 0.5f / nvx_a4_gs((const uint8_t *)(intptr_t)DP(const int64_t, 8)[x]);
    const float *a = DP(const float, 19) + (size_t)p * 2 * inner;
    DP(float, 20)[i] = moe_b12x_act(a[c] * sc, a[inner + c] * sc);
}

/* the act of the routed pairs in the A4 form with the gs2 of each pair's expert */
__global__ void k_a4_quant_pairs(const gp_rec *r, const int64_t *e, int8_t *q, float *slo, float *shi)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int E = DI(5), inner = DI(7), nh = inner / 16;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)w.start[E] * nh) {
        return;
    }
    int p = (int)(i / nh), h = (int)(i % nh);
    int x = qmoe_expert_of(w.start, E, p);
    float gs2 = nvx_a4_gs((const uint8_t *)(intptr_t)DP(const int64_t, 10)[x]);
    float sf = rc_a4_16(DP(const float, 20) + (size_t)p * inner + 16 * h, gs2, q + (size_t)p * inner + 16 * h);
    (h % 2 ? shi : slo)[(size_t)p * (inner / 32) + h / 2] = sf;
}

/* y = bf16(acc 0.5 / gs2) of each routed pair's down row (in place in de) */
__global__ void k_qmoe_y_a4(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    qmoe_w w = qmoe_work(r, e);
    int E = DI(5), hidden = DI(6);
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (int64_t)w.start[E] * hidden) {
        return;
    }
    int p = (int)(i / hidden);
    int x = qmoe_expert_of(w.start, E, p);
    float sc = 0.5f / nvx_a4_gs((const uint8_t *)(intptr_t)DP(const int64_t, 10)[x]);
    float *d = DP(float, 21);
    d[i] = rc_bf16r(d[i] * sc);
}

/* GP_MX8_ACT: x, n. Each 32 values as dynamic MXFP8 in place (FP4-Local's vision attention): e =
 * clamp(ceil(log2(amax / 448)), -127, 127), x = e4m3(x / 2^e) 2^e (a warp a group, as the CPU). */
__global__ void k_mx8_act(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int g = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, l = threadIdx.x % 32;
    if (g >= DI(1) / 32) {
        return;
    }
    float *x = DP(float, 0) + (size_t)32 * g;
    float v = x[l], amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(RC_FULL_MASK, amax, o));
    }
    int ex = 0;
    if (amax > 0.f) {
        int p;
        float m = frexpf(__fdiv_rn(amax, 448.f), &p);
        ex = m == 0.5f ? p - 1 : p;
        ex = ex < -127 ? -127 : ex > 127 ? 127 : ex;
    }
    float q = rc_e4m3_rne(fabsf(ldexpf(v, -ex)));
    x[l] = ldexpf(v < 0.f ? -q : q, ex);
}

/* GP_B12X_GLU: g, u, out, n. out = bf16(bf16(g' sigma(g')) u') with g' = bf16(g), u' = bf16(u) (the
 * act of the contract: the shared expert of a prompt group). */
__global__ void k_b12x_glu(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)DI(3)) {
        DP(float, 2)[i] = moe_b12x_act(DP(const float, 0)[i], DP(const float, 1)[i]);
    }
}

/* GP_MOE_B12X_OUT: routed (the GPU's sum of w y), cpu (its part, or none), s (the shared expert's down
 * rows, or none), slog (its gate logits), out, t, hidden. out = bf16(bf16(routed + cpu) + bf16(
 * bf16(sigmoid(bf16(slog))) bf16(s))) (moe_b12x_out). */
__global__ void k_moe_b12x_out(const gp_rec *r, const int64_t *e)
{
    PDL_START();
    int c = blockIdx.x * blockDim.x + threadIdx.x, tok = blockIdx.y;
    int hidden = DI(6);
    if (c >= hidden) {
        return;
    }
    size_t o = (size_t)tok * hidden + c;
    float acc = DP(const float, 0)[o];
    if (r->tag[1] != GP_T_NONE && DP(const float, 1) != NULL) {
        acc += DP(const float, 1)[o];
    }
    int has = r->tag[2] != GP_T_NONE && DP(const float, 2) != NULL;
    DP(float, 4)[o] = moe_b12x_out(acc, has, has ? DP(const float, 3)[tok] : 0.f, has ? DP(const float, 2)[o] : 0.f);
}
