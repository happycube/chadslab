/* ---- The media encoders ---------------------------------------------------
 *
 * The kernels of the records GP_ENC_* (the encoders of images and audio,
 * reclank/encoders/runtime.py, as programs). The arrays are row major. */

#define EG_T 64
#define EG_K 16

/* y (n, m) = clamp(clamp(x, imin, imax) W^T + b, omin, omax). x (n, k) is
 * float32 and W (m, k) bfloat16 (wbf 1) or float32; b may be NULL. Any n, m,
 * and k. A block of 256 threads makes a tile of 64 x 64 values of y. */
__global__ void k_enc_gemm(const float *x, const void *w, int wbf, const float *b, float *y,
                           int n, int m, int k, float imin, float imax, float omin, float omax)
{
    PDL_START();
    __shared__ float xs[EG_K][EG_T + 1];
    __shared__ float ws[EG_K][EG_T + 1];
    int r0 = blockIdx.y * EG_T, c0 = blockIdx.x * EG_T;
    int tx = threadIdx.x % 16, ty = threadIdx.x / 16;
    float acc[4][4];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            acc[i][j] = 0.f;
        }
    }
    for (int k0 = 0; k0 < k; k0 += EG_K) {
        for (int q = threadIdx.x; q < EG_T * EG_K; q += blockDim.x) {
            int rr = q / EG_K, kk = q % EG_K, kc = k0 + kk;
            int r = r0 + rr, c = c0 + rr;
            float xv = 0.f, wv = 0.f;
            if (r < n && kc < k) {
                xv = fminf(fmaxf(x[(size_t)r * k + kc], imin), imax);
            }
            if (c < m && kc < k) {
                size_t o = (size_t)c * k + kc;
                wv = wbf ? __uint_as_float((uint32_t)((const uint16_t *)w)[o] << 16)
                         : ((const float *)w)[o];
            }
            xs[kk][rr] = xv;
            ws[kk][rr] = wv;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < EG_K; ++kk) {
            float a[4], bv[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = xs[kk][ty + 16 * i];
                bv[i] = ws[kk][tx + 16 * i];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[i][j] += a[i] * bv[j];
                }
            }
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        int r = r0 + ty + 16 * i;
        for (int j = 0; j < 4; ++j) {
            int c = c0 + tx + 16 * j;
            if (r < n && c < m) {
                float v = acc[i][j] + (b ? b[c] : 0.f);
                y[(size_t)r * m + c] = fminf(fmaxf(v, omin), omax);
            }
        }
    }
}

/* y = x / sqrt(mean(x^2) + eps), times w when w is not NULL. One warp for
 * each row of d values. y may be x. */
__global__ void k_enc_rms(const float *x, const float *w, float *y, int rows, int d, float eps)
{
    PDL_START();
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const float *p = x + (size_t)row * d;
    float s = 0.f;
    for (int i = lane; i < d; i += 32) {
        s += p[i] * p[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, off);
    }
    float r = 1.0f / sqrtf(s / (float)d + eps);
    for (int i = lane; i < d; i += 32) {
        y[(size_t)row * d + i] = p[i] * r * (w ? w[i] : 1.0f);
    }
}

/* The LayerNorm of each row of d values, times w, plus b (either may be
 * NULL). One warp for each row. y may be x. */
__global__ void k_enc_lnorm(const float *x, const float *w, const float *b, float *y, int rows,
                            int d, float eps)
{
    PDL_START();
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= rows) {
        return;
    }
    const float *p = x + (size_t)row * d;
    float s = 0.f;
    for (int i = lane; i < d; i += 32) {
        s += p[i];
    }
    for (int off = 16; off > 0; off >>= 1) {
        s += __shfl_xor_sync(RC_FULL_MASK, s, off);
    }
    float mean = s / (float)d, v = 0.f;
    for (int i = lane; i < d; i += 32) {
        float t = p[i] - mean;
        v += t * t;
    }
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor_sync(RC_FULL_MASK, v, off);
    }
    float r = 1.0f / sqrtf(v / (float)d + eps);
    for (int i = lane; i < d; i += 32) {
        y[(size_t)row * d + i] = (p[i] - mean) * r * (w ? w[i] : 1.0f) + (b ? b[i] : 0.0f);
    }
}

/* y = gelu(x), n values: the tanh form, or the erf form (use_erf). */
__global__ void k_enc_gelu(const float *x, float *y, int64_t n, int use_erf)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i];
        y[i] = use_erf ? 0.5f * v * (1.0f + erff(v * 0.7071067811865476f))
                       : 0.5f * v * (1.0f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    }
}

/* y = gelu_tanh(g) * u, n values. */
__global__ void k_enc_gelu_mul(const float *g, const float *u, float *y, int64_t n)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        float t = tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v));
        y[i] = 0.5f * v * (1.0f + t) * u[i];
    }
}

/* y = silu(g) * u, n values. */
__global__ void k_enc_silu_mul(const float *g, const float *u, float *y, int64_t n)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = g[i];
        y[i] = v / (1.0f + expf(-v)) * u[i];
    }
}

/* x += sc y, n values. */
__global__ void k_enc_add(float *x, const float *y, int64_t n, float sc)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        x[i] += sc * y[i];
    }
}

/* y = clamp(x, lo, hi), n values. */
__global__ void k_enc_clamp(const float *x, float *y, int64_t n, float lo, float hi)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = fminf(fmaxf(x[i], lo), hi);
    }
}

/* y = clamp(y + b, lo, hi) in place; b may be NULL. */
__global__ void k_enc_bias_clamp(float *y, const float *b, int rows, int cols, float lo, float hi)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        y[i] = fminf(fmaxf(y[i] + (b ? b[i % cols] : 0.f), lo), hi);
    }
}

/* y = x sigmoid(x), n values. */
__global__ void k_enc_silu(const float *x, float *y, int64_t n)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = x[i] / (1.0f + expf(-x[i]));
    }
}

/* y = x vec for each row of cols values. */
__global__ void k_enc_mul_vec(const float *x, const float *vec, float *y, int rows, int cols)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        y[i] = x[i] * vec[i % cols];
    }
}

/* y (rows, cols) = a sigmoid(b), a and b the halves of each row of x. */
__global__ void k_enc_glu(const float *x, float *y, int rows, int cols)
{
    PDL_START();
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (int64_t)rows * cols) {
        int64_t row = i / cols;
        int c = (int)(i % cols);
        const float *xa = x + row * 2 * cols;
        y[i] = xa[c] / (1.0f + expf(-xa[cols + c]));
    }
}

/* The causal depthwise conv: y[i, c] = sum_j w[c, j] x[i - kw + 1 + j, c]. */
__global__ void k_enc_dwconv(const float *x, const float *w, float *y, int t, int c, int kw)
{
    PDL_START();
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)t * c) {
        return;
    }
    int i = (int)(idx / c), ch = (int)(idx % c);
    float acc = 0.f;
    for (int j = 0; j < kw; ++j) {
        int src = i - kw + 1 + j;
        if (src >= 0) {
            acc += w[(size_t)ch * kw + j] * x[(size_t)src * c + ch];
        }
    }
    y[idx] = acc;
}

/* The local attention of gemma4a: query i sees the keys i - span + 1 .. i;
 * key j of that list adds q r[1 + j]; the scores get the soft cap, and an
 * invalid key -1e9. One warp for each (query, head); lane j < span scores
 * key j, then each lane sums its values of the output. */
__global__ void k_enc_local_attn(const float *q, const float *k, const float *v, const float *r,
                                 const int *valid, float *o, int t, int heads, int hd, int span,
                                 float cap)
{
    PDL_START();
    int64_t row = (int64_t)blockIdx.x * 4 + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= (int64_t)t * heads) {
        return;
    }
    int i = (int)(row / heads), h = (int)(row % heads);
    const float *qi = q + row * hd;
    int kp = i - span + 1 + lane;
    float sc = -INFINITY;
    if (lane < span) {
        sc = -1e9f;
        if (kp >= 0 && valid[kp]) {
            const float *kk = k + ((size_t)kp * heads + h) * hd;
            const float *rj = r + ((size_t)(1 + lane) * heads + h) * hd;
            float a = 0.f;
            for (int d = 0; d < hd; ++d) {
                a += qi[d] * (kk[d] + rj[d]);
            }
            sc = tanhf(a / cap) * cap;
        }
    }
    float m = sc;
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor_sync(RC_FULL_MASK, m, off));
    }
    float p = lane < span ? expf(sc - m) : 0.f;
    float sum = p;
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(RC_FULL_MASK, sum, off);
    }
    p /= sum;
    for (int d0 = 0; d0 < hd; d0 += 32) {
        /* every lane takes each pass, so the shuffles agree */
        int d = d0 + lane;
        float acc = 0.f;
        for (int j = 0; j < span; ++j) {
            float pj = __shfl_sync(RC_FULL_MASK, p, j);
            int kj = i - span + 1 + j;
            if (kj >= 0 && d < hd) {
                acc += pj * v[((size_t)kj * heads + h) * hd + d];
            }
        }
        if (d < hd) {
            o[row * hd + d] = acc;
        }
    }
}

/* The axial 2D RoPE of gemma4v, in place on x (n, heads, hd): the first half
 * of each head turns with the x position and the second with the y
 * position; each half is a NEOX rotation (its first quarter with its second)
 * at the angles pos * inv[j], j < hd / 4. pos is (n, 2) int32. */
__global__ void k_enc_rope2d(float *x, const int *pos, const float *inv, int n, int heads, int hd)
{
    PDL_START();
    int q4 = hd / 4, h2 = hd / 2;
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)n * heads * 2 * q4;
    if (i >= total) {
        return;
    }
    int j = (int)(i % q4);
    int part = (int)((i / q4) % 2);
    int64_t row = i / (2 * q4);          /* patch * heads + head */
    int patch = (int)(row / heads);
    float ang = (float)pos[2 * patch + part] * inv[j];
    float c = cosf(ang), s = sinf(ang);
    float *p = x + row * hd + part * h2;
    float a = p[j], b = p[j + q4];
    p[j] = a * c - b * s;
    p[j + q4] = b * c + a * s;
}

/* The attention of every query over every key, scale 1. q, k, v, and o are
 * (n, heads, hd), hd at most 128. One warp for each (query, head); the 4
 * warps of a block take 4 queries of one head and share the tiles of 32 keys
 * and values in shared memory. Lane j scores key j of a tile; the softmax is
 * online. */
__global__ void k_enc_attn(const float *q, const float *k, const float *v, float *o,
                           int n, int heads, int hd)
{
    PDL_START();
    extern __shared__ float esm[];
    int ld = hd + 1;
    float *ks = esm, *vs = ks + 32 * ld;
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    float *qs = vs + 32 * ld + warp * hd;
    int h = blockIdx.y, qi = blockIdx.x * 4 + warp;
    bool live = qi < n;
    for (int d = lane; d < hd; d += 32) {
        qs[d] = live ? q[((size_t)qi * heads + h) * hd + d] : 0.f;
    }
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    float m = -INFINITY, l = 0.f;
    for (int k0 = 0; k0 < n; k0 += 32) {
        __syncthreads();
        for (int t = threadIdx.x; t < 32 * hd; t += blockDim.x) {
            int j = t / hd, d = t % hd, kj = k0 + j;
            size_t src = ((size_t)kj * heads + h) * hd + d;
            ks[j * ld + d] = kj < n ? k[src] : 0.f;
            vs[j * ld + d] = kj < n ? v[src] : 0.f;
        }
        __syncthreads();
        bool kv = k0 + lane < n;
        float sc = -INFINITY;
        if (kv) {
            sc = 0.f;
            for (int d = 0; d < hd; ++d) {
                sc += qs[d] * ks[lane * ld + d];
            }
        }
        float mt = sc;
        for (int off = 16; off > 0; off >>= 1) {
            mt = fmaxf(mt, __shfl_xor_sync(RC_FULL_MASK, mt, off));
        }
        float mn = fmaxf(m, mt);
        float p = kv ? expf(sc - mn) : 0.f;
        float scale = m == -INFINITY ? 0.f : expf(m - mn);
        float ps = p;
        for (int off = 16; off > 0; off >>= 1) {
            ps += __shfl_xor_sync(RC_FULL_MASK, ps, off);
        }
        l = l * scale + ps;
        for (int i = 0; i < 4; ++i) {
            acc[i] *= scale;
        }
        for (int j = 0; j < 32; ++j) {
            float pj = __shfl_sync(RC_FULL_MASK, p, j);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int d = lane + 32 * i;
                if (d < hd) {
                    acc[i] += pj * vs[j * ld + d];
                }
            }
        }
        m = mn;
    }
    if (live) {
        for (int i = 0; i < 4; ++i) {
            int d = lane + 32 * i;
            if (d < hd) {
                o[((size_t)qi * heads + h) * hd + d] = acc[i] / l;
            }
        }
    }
}

/* k_enc_attn on the tensor cores (mma.sync m16n8k16): q, k, and v are
 * float16 (n, heads, hd), from k_enc_to_half; the scores, the softmax, and
 * the sums are float32, the weights p float16 (as k_flash_tc). hd is a
 * multiple of 8 of at most HP, HP a multiple of 16 (the values past hd are
 * zeros). A block of 4 warps takes EA_Q queries of one head; warp w takes the
 * 16 queries 16 w, with their fragments of q in registers. For each step of
 * EA_K keys: K and V to shared memory, S = Q K^T, the online softmax on the
 * fragments of S (the 4 lanes of a row share its maximum and its sum), then
 * O += P V (the fragments of S are those of P; V by ldmatrix.trans). */
#define EA_K 64

__device__ __forceinline__ void ldsm_x2_trans(uint32_t *d, const void *p)
{
#if RC_SM80
    unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(a));
#else
    __trap();
#endif
}

template <int HP>
__global__ void __launch_bounds__(128) k_enc_attn_tc(const __half *q, const __half *k,
                                                     const __half *v, float *o, int n, int heads,
                                                     int hd)
{
    PDL_START();
    constexpr int LD = HP + 8;                /* the row stride (halves): 16 bytes apart */
    __shared__ __align__(16) __half ks[EA_K * LD];
    __shared__ __align__(16) __half vs[EA_K * LD];
    int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    int g = lane / 4, c = lane % 4;
    int h = blockIdx.y;
    size_t rs = (size_t)heads * hd;           /* the row stride of q, k, v */
    int r0 = blockIdx.x * EA_Q + warp * 16 + g, r1 = r0 + 8;
    bool live0 = r0 < n, live1 = r1 < n;
    const __half *q0 = q + (size_t)(live0 ? r0 : 0) * rs + (size_t)h * hd;
    const __half *q1 = q + (size_t)(live1 ? r1 : 0) * rs + (size_t)h * hd;
    uint32_t qa[HP / 16][4];
    #pragma unroll
    for (int kk = 0; kk < HP / 16; ++kk) {
        int d0 = kk * 16 + 2 * c, d1 = d0 + 8;
        qa[kk][0] = d0 < hd ? *(const uint32_t *)&q0[d0] : 0u;
        qa[kk][1] = d0 < hd ? *(const uint32_t *)&q1[d0] : 0u;
        qa[kk][2] = d1 < hd ? *(const uint32_t *)&q0[d1] : 0u;
        qa[kk][3] = d1 < hd ? *(const uint32_t *)&q1[d1] : 0u;
    }
    float acc[HP / 8][4];
    #pragma unroll
    for (int i = 0; i < HP / 8; ++i) {
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    }
    const float L2E = 1.4426950408889634f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int k0 = 0; k0 < n; k0 += EA_K) {
        __syncthreads();
        /* 8 halves (16 bytes) at a time; zeros past n and past hd */
        for (int x = threadIdx.x; x < EA_K * (HP / 8); x += blockDim.x) {
            int j = x / (HP / 8), d = (x % (HP / 8)) * 8, kj = k0 + j;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u), vv = kv;
            if (kj < n && d < hd) {
                size_t src = (size_t)kj * rs + (size_t)h * hd + d;
                kv = *(const uint4 *)&k[src];
                vv = *(const uint4 *)&v[src];
            }
            *(uint4 *)&ks[j * LD + d] = kv;
            *(uint4 *)&vs[j * LD + d] = vv;
        }
        __syncthreads();
        float sc[EA_K / 8][4];
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            sc[nt][0] = sc[nt][1] = sc[nt][2] = sc[nt][3] = 0.f;
        }
        #pragma unroll
        for (int kk = 0; kk < HP / 16; ++kk) {
            #pragma unroll
            for (int nt = 0; nt < EA_K / 8; ++nt) {
                uint32_t b[2];
                b[0] = *(const uint32_t *)&ks[(nt * 8 + g) * LD + kk * 16 + 2 * c];
                b[1] = *(const uint32_t *)&ks[(nt * 8 + g) * LD + kk * 16 + 2 * c + 8];
                mma16816(sc[nt], qa[kk], b);
            }
        }
        /* sc[nt][0..1]: row g, keys nt * 8 + 2c and + 1; sc[nt][2..3]: row g + 8.
         * Each step has a key, so the maxima are finite. */
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            #pragma unroll
            for (int u = 0; u < 2; ++u) {
                bool ok = k0 + nt * 8 + 2 * c + u < n;
                sc[nt][u] = ok ? sc[nt][u] * L2E : -INFINITY;
                sc[nt][2 + u] = ok ? sc[nt][2 + u] * L2E : -INFINITY;
                mx0 = fmaxf(mx0, sc[nt][u]);
                mx1 = fmaxf(mx1, sc[nt][2 + u]);
            }
        }
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(RC_FULL_MASK, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(RC_FULL_MASK, mx1, off));
        }
        float f0 = exp2f(m0 - mx0), f1 = exp2f(m1 - mx1);   /* 0 at the first step */
        float ls0 = 0.f, ls1 = 0.f;
        uint32_t pa[EA_K / 8][2];
        #pragma unroll
        for (int nt = 0; nt < EA_K / 8; ++nt) {
            float e0 = exp2f(sc[nt][0] - mx0), e1 = exp2f(sc[nt][1] - mx0);
            float e2 = exp2f(sc[nt][2] - mx1), e3 = exp2f(sc[nt][3] - mx1);
            ls0 += e0 + e1;
            ls1 += e2 + e3;
            __half2 h01 = __floats2half2_rn(e0, e1), h23 = __floats2half2_rn(e2, e3);
            pa[nt][0] = *(uint32_t *)&h01;
            pa[nt][1] = *(uint32_t *)&h23;
        }
        for (int off = 1; off < 4; off <<= 1) {
            ls0 += __shfl_xor_sync(RC_FULL_MASK, ls0, off);
            ls1 += __shfl_xor_sync(RC_FULL_MASK, ls1, off);
        }
        l0 = l0 * f0 + ls0;
        l1 = l1 * f1 + ls1;
        m0 = mx0;
        m1 = mx1;
        #pragma unroll
        for (int nt = 0; nt < HP / 8; ++nt) {
            acc[nt][0] *= f0;
            acc[nt][1] *= f0;
            acc[nt][2] *= f1;
            acc[nt][3] *= f1;
        }
        #pragma unroll
        for (int kk = 0; kk < EA_K / 16; ++kk) {
            /* the A fragment of P: keys 16 kk + 2c (tile 2 kk), + 8 (tile 2 kk + 1) */
            uint32_t a[4] = {pa[2 * kk][0], pa[2 * kk][1], pa[2 * kk + 1][0], pa[2 * kk + 1][1]};
            const __half *vrow = &vs[(kk * 16 + (lane % 16)) * LD];
            #pragma unroll
            for (int nt = 0; nt < HP / 8; ++nt) {
                uint32_t b[2];
                ldsm_x2_trans(b, vrow + nt * 8);
                mma16816(acc[nt], a, b);
            }
        }
    }
    float inv0 = 1.0f / l0, inv1 = 1.0f / l1;
    float *o0 = o + (size_t)r0 * rs + (size_t)h * hd, *o1 = o + (size_t)r1 * rs + (size_t)h * hd;
    #pragma unroll
    for (int nt = 0; nt < HP / 8; ++nt) {
        int d = nt * 8 + 2 * c;
        if (d < hd) {
            if (live0) {
                *(float2 *)&o0[d] = make_float2(acc[nt][0] * inv0, acc[nt][1] * inv0);
            }
            if (live1) {
                *(float2 *)&o1[d] = make_float2(acc[nt][2] * inv1, acc[nt][3] * inv1);
            }
        }
    }
}
