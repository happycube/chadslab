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
