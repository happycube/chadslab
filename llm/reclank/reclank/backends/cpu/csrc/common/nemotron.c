/* ---------- Nemotron 3.5 Bolt (NemotronH): the Mamba-2 mixer, relu^2, the latent MoE ----------
 * docs/NEMOTRON_BOLT_PLAN.md; reclank/models/nemotron/model.py is the reference (its names).
 *
 * GP_NM_MAMBA: t rows of in_proj's output, each [z (inner) | xBC (conv) | dt (heads)]: the causal
 * conv of the kernel over xBC with the conv rows of the state, + bias, silu; then for each head
 * (16 heads a group of B and C) and each real token the recurrence of the state (heads x hdim x
 * nstate, float32): dt = max(softplus(dt + dt_bias), dt_min), h = exp(dt A) h + dt x B,
 * y = h C + D x; then the gated RMSNorm of each group of inner / groups values:
 * rmsnorm(y silu(z)) w. Rows past nreal (the padding of a GPU group) do not change the state and
 * give zeros.
 *
 * GP_NM_MOE: the routed experts of the latent (x: t rows of L values): for each pair (token, slot)
 * with an expert (ids >= 0; a negative id is an expert of the GPU), up (inner x L), relu^2, down
 * (L x inner), times the weight; out = the sum over the slots, in the order of the slots (the
 * same bits for any count of threads). The pairs go by expert, so the tokens of an expert read
 * its rows once (a prompt). The products are those of kq_dot1 on int8 x (kq_quant_part). */

static inline float nm_silu(float x)
{
    return x / (1.f + expf(-x));
}

static inline float nm_softplus(float x)
{
    return x > 20.f ? x : log1pf(expf(x));
}

/* zxd: t x (inner + conv + heads); conv_state: (kern - 1) x conv; cw: conv x kern; scr: t x conv
 * floats; y: t x inner. */
static void nm_mamba_body(const float *zxd, float *conv_state, const float *cw, const float *cb,
                          const float *A_log, const float *D, const float *dt_bias, const float *nw,
                          float *ssm, float *y, float *scr, int t, int heads, int hdim, int groups,
                          int nstate, int kern, float eps, float dt_min, int nreal)
{
    const int inner = heads * hdim, conv = inner + 2 * groups * nstate;
    const size_t rs = (size_t)inner + conv + heads;
    if (nreal <= 0 || nreal > t) {
        nreal = t;
    }
#if defined(__AVX2__)
    if (conv % 64 == 0 && kern <= 16) {
        /* the conv in blocks of 64 channels, the channels in the lanes; the state as below */
        #pragma omp for schedule(static)
        for (int c0 = 0; c0 < conv; c0 += 64) {
            float wt[16][64];
            for (int k = 0; k < kern; ++k) {
                for (int c = 0; c < 64; ++c) {
                    wt[k][c] = cw[(size_t)(c0 + c) * kern + k];
                }
            }
            for (int j = 0; j < t; ++j) {
                for (int c = 0; c < 64; c += NM_VL) {
                    nm_vf s = NM_LDF(cb + c0 + c);
                    for (int k = 0; k < kern; ++k) {
                        int r = j + k - (kern - 1);
                        const float *v = r < 0 ? conv_state + (size_t)(r + kern - 1) * conv + c0 + c
                                               : zxd + (size_t)r * rs + inner + c0 + c;
                        s = NM_FMA(NM_LDF(v), NM_LDF(&wt[k][c]), s);
                    }
                    NM_STF(scr + (size_t)j * conv + c0 + c, nm_siluv(s));
                }
            }
            for (int c = c0; c < c0 + 64; ++c) {
                float last[16];
                for (int k = 0; k < kern - 1; ++k) {
                    int r = nreal + k - (kern - 1);
                    last[k] = r < 0 ? conv_state[(size_t)(r + kern - 1) * conv + c] : zxd[(size_t)r * rs + inner + c];
                }
                for (int k = 0; k < kern - 1; ++k) {
                    conv_state[(size_t)k * conv + c] = last[k];
                }
            }
        }
    } else
#endif
    /* the conv: channel c of token j reads rows j .. j + kern - 1 of [state; xBC] */
    #pragma omp for schedule(static)
    for (int c = 0; c < conv; ++c) {
        const float *w = cw + (size_t)c * kern;
        for (int j = 0; j < t; ++j) {
            float s = cb[c];
            for (int k = 0; k < kern; ++k) {
                int r = j + k - (kern - 1);         /* < 0: a row of the state */
                float v = r < 0 ? conv_state[(size_t)(r + kern - 1) * conv + c]
                                : zxd[(size_t)r * rs + inner + c];
                s += v * w[k];
            }
            scr[(size_t)j * conv + c] = nm_silu(s);
        }
        /* the state: the last kern - 1 rows of [state; xBC of the real tokens] */
        float last[16];
        for (int k = 0; k < kern - 1; ++k) {
            int r = nreal + k - (kern - 1);
            last[k] = r < 0 ? conv_state[(size_t)(r + kern - 1) * conv + c] : zxd[(size_t)r * rs + inner + c];
        }
        for (int k = 0; k < kern - 1; ++k) {
            conv_state[(size_t)k * conv + c] = last[k];
        }
    }
    /* the recurrence, a task for each head and each 16 values of hdim */
    const int per = heads / groups, dch = 16;
#if defined(__AVX2__)
    if (hdim % dch == 0 && nstate <= 256) {
        /* the 16 values of hdim of a task in the lanes (the state of the task transposed to
         * [n][d] for its tokens, then back): h_n = dA h_n + (dt x) B_n, y += h_n C_n */
        #pragma omp for schedule(static)
        for (int x = 0; x < heads * (hdim / dch); ++x) {
            int h = x / (hdim / dch), d0 = (x % (hdim / dch)) * dch, g = h / per;
            float A = -expf(A_log[h]);
            float *S = ssm + ((size_t)h * hdim + d0) * nstate;
            float St[256][16] __attribute__((aligned(64)));
            for (int d = 0; d < dch; ++d) {
                for (int n = 0; n < nstate; ++n) {
                    St[n][d] = S[(size_t)d * nstate + n];
                }
            }
            for (int j = 0; j < t; ++j) {
                float *yr = y + (size_t)j * inner + (size_t)h * hdim + d0;
                if (j >= nreal) {
                    for (int d = 0; d < dch; ++d) {
                        yr[d] = 0.f;
                    }
                    continue;
                }
                const float *xr = scr + (size_t)j * conv;
                const float *B = xr + inner + (size_t)g * nstate;
                const float *C = xr + inner + (size_t)groups * nstate + (size_t)g * nstate;
                float dt = fmaxf(nm_softplus(zxd[(size_t)j * rs + inner + conv + h] + dt_bias[h]), dt_min);
                const nm_vf va = NM_SETF(expf(dt * A));
                nm_vf xv[16 / NM_VL], dx[16 / NM_VL], acc[16 / NM_VL];
                for (int v = 0; v < 16 / NM_VL; ++v) {
                    xv[v] = NM_LDF(xr + (size_t)h * hdim + d0 + v * NM_VL);
                    dx[v] = NM_MUL(xv[v], NM_SETF(dt));
                    acc[v] = NM_ZF();
                }
                for (int n = 0; n < nstate; ++n) {
                    const nm_vf b = NM_SETF(B[n]), c = NM_SETF(C[n]);
                    for (int v = 0; v < 16 / NM_VL; ++v) {
                        nm_vf sv = NM_FMA(NM_LDF(&St[n][v * NM_VL]), va, NM_MUL(dx[v], b));
                        NM_STF(&St[n][v * NM_VL], sv);
                        acc[v] = NM_FMA(sv, c, acc[v]);
                    }
                }
                for (int v = 0; v < 16 / NM_VL; ++v) {
                    NM_STF(yr + v * NM_VL, NM_FMA(xv[v], NM_SETF(D[h]), acc[v]));
                }
            }
            for (int d = 0; d < dch; ++d) {
                for (int n = 0; n < nstate; ++n) {
                    S[(size_t)d * nstate + n] = St[n][d];
                }
            }
        }
    } else
#endif
    #pragma omp for schedule(static)
    for (int x = 0; x < heads * (hdim / dch); ++x) {
        int h = x / (hdim / dch), d0 = (x % (hdim / dch)) * dch, g = h / per;
        float A = -expf(A_log[h]);
        float *S = ssm + ((size_t)h * hdim + d0) * nstate;
        for (int j = 0; j < t; ++j) {
            float *yr = y + (size_t)j * inner + (size_t)h * hdim + d0;
            if (j >= nreal) {
                for (int d = 0; d < dch; ++d) {
                    yr[d] = 0.f;
                }
                continue;
            }
            const float *xr = scr + (size_t)j * conv;
            const float *B = xr + inner + (size_t)g * nstate;
            const float *C = xr + inner + (size_t)groups * nstate + (size_t)g * nstate;
            float dt = fmaxf(nm_softplus(zxd[(size_t)j * rs + inner + conv + h] + dt_bias[h]), dt_min);
            float dA = expf(dt * A);
            for (int d = 0; d < dch; ++d) {
                float xv = xr[(size_t)h * hdim + d0 + d], dx = dt * xv, acc = 0.f;
                float *Sd = S + (size_t)d * nstate;
                for (int n = 0; n < nstate; ++n) {
                    float s = Sd[n] * dA + dx * B[n];
                    Sd[n] = s;
                    acc += s * C[n];
                }
                yr[d] = acc + D[h] * xv;
            }
        }
    }
    /* the gated RMSNorm of each group */
    const int gs = inner / groups;
    #pragma omp for schedule(static)
    for (int x = 0; x < t * groups; ++x) {
        int j = x / groups, g = x % groups;
        float *yr = y + (size_t)j * inner + (size_t)g * gs;
        const float *z = zxd + (size_t)j * rs + (size_t)g * gs;
        float ss = 0.f;
        int i0 = 0;
#if defined(__AVX2__)
        nm_vf sv = NM_ZF();
        for (; i0 + NM_VL <= gs; i0 += NM_VL) {
            nm_vf v = NM_MUL(NM_LDF(yr + i0), nm_siluv(NM_LDF(z + i0)));
            NM_STF(yr + i0, v);
            sv = NM_FMA(v, v, sv);
        }
        ss = NM_HSUM(sv);
#endif
        for (int i = i0; i < gs; ++i) {
            yr[i] *= nm_silu(z[i]);
            ss += yr[i] * yr[i];
        }
        float inv = 1.f / sqrtf(ss / (float)gs + eps);
        for (int i = 0; i < gs; ++i) {
            yr[i] = yr[i] * inv * nw[(size_t)g * gs + i];
        }
    }
}

/* ---------- the LiLiCorr drafter (reclank/models/nemotron/lilicorr.py) ----------
 * GP_NM_DPREP: q (or none), k, v, qn, kn, K, V, qout, t, nq, nk, hd, pos, ring, theta, eps: the rows
 * of q and k normed by head (qn, kn), NEOX rope at pos + j; q to qout, k and v to row (pos + j) % ring
 * of K and V (ring x nk x hd). */
static void nm_dprep_body(const float *q, const float *k, const float *v, const float *qn, const float *kn,
                          float *K, float *V, float *qout, int t, int nq, int nk, int hd, int64_t pos,
                          int ring, float theta, float eps)
{
    const int nh = (q != NULL ? nq : 0) + nk, half = hd / 2;
    #pragma omp for schedule(static)
    for (int x = 0; x < t * nh; ++x) {
        const int j = x / nh, h = x % nh, isq = q != NULL && h < nq, kh = isq ? 0 : h - (q != NULL ? nq : 0);
        const float *src = isq ? q + ((size_t)j * nq + h) * hd : k + ((size_t)j * nk + kh) * hd;
        const float *w = isq ? qn : kn;
        const size_t slot = (size_t)((pos + j) % ring);
        float *dst = isq ? qout + ((size_t)j * nq + h) * hd : K + (slot * nk + kh) * hd;
        float ss = 0.f, y[512];
        for (int d = 0; d < hd; ++d) {
            ss += src[d] * src[d];
        }
        const float inv = 1.f / sqrtf(ss / (float)hd + eps);
        for (int d = 0; d < hd; ++d) {
            y[d] = src[d] * inv * w[d];
        }
        for (int d = 0; d < half; ++d) {
            const double a = (double)(pos + j) * pow((double)theta, -2.0 * d / hd);
            const float c = (float)cos(a), sn = (float)sin(a);
            dst[d] = y[d] * c - y[d + half] * sn;
            dst[d + half] = y[d + half] * c + y[d] * sn;
        }
        if (!isq) {
            memcpy(V + (slot * nk + kh) * hd, v + ((size_t)j * nk + kh) * hd, (size_t)hd * 4);
        }
    }
}

/* GP_NM_DATTN: q (t x nq x hd, normed and roped), K, V (the ring), sinks (nq), out, t, nq, nk, hd, pos0,
 * ring, window: query j (position pos0 + j) over the positions max(0, pos0 + j - window + 1) .. pos0 +
 * t - 1 of the ring (the context and the whole block: not causal), scale 1 / sqrt(hd), the sink of
 * the head in the denominator. */
static void nm_dattn_body(const float *q, const float *K, const float *V, const float *sinks, float *out,
                          int t, int nq, int nk, int hd, int64_t pos0, int ring, int window, float *scr)
{
    const float scale = 1.f / sqrtf((float)hd);
    const int rep = nq / nk;
    #pragma omp for schedule(dynamic, 1)
    for (int x = 0; x < t * nq; ++x) {
        const int j = x / nq, h = x % nq, kh = h / rep;
        const int64_t p = pos0 + j, lo = p - window + 1 > 0 ? p - window + 1 : 0, hi = pos0 + t - 1;
        const float *qr = q + ((size_t)j * nq + h) * hd;
        float *s = scr + (size_t)x * (window + t);
        float m = sinks[h];
        for (int64_t pk = lo; pk <= hi; ++pk) {
            const float *kr = K + ((size_t)(pk % ring) * nk + kh) * hd;
            float d = 0.f;
            for (int i = 0; i < hd; ++i) {
                d += qr[i] * kr[i];
            }
            d *= scale;
            s[pk - lo] = d;
            m = d > m ? d : m;
        }
        float den = expf(sinks[h] - m), o[512];
        for (int i = 0; i < hd; ++i) {
            o[i] = 0.f;
        }
        for (int64_t pk = lo; pk <= hi; ++pk) {
            const float e = expf(s[pk - lo] - m);
            const float *vr = V + ((size_t)(pk % ring) * nk + kh) * hd;
            den += e;
            for (int i = 0; i < hd; ++i) {
                o[i] += e * vr[i];
            }
        }
        float *orow = out + ((size_t)j * nq + h) * hd;
        for (int i = 0; i < hd; ++i) {
            orow[i] = o[i] / den;
        }
    }
}

/* GP_NM_DCONV: h, dyn (t x 2 x kern x groups: the coefficients of each token), base (2 x kern x cols),
 * out, t, cols, kern, group, side: out[j] = sum over the taps of (base[side, tap] + dyn[j, side, tap,
 * c / group]) h[j - tap] (the rows of the block before it; none before the first). */
static void nm_dconv_body(const float *h, const float *dyn, const float *base, float *out, int t, int cols,
                          int kern, int group, int side)
{
    const int ng = cols / group;
    #pragma omp for schedule(static)
    for (int x = 0; x < t * ng; ++x) {
        const int j = x / ng, g = x % ng;
        for (int c = g * group; c < (g + 1) * group; ++c) {
            float sum = 0.f;
            for (int tap = 0; tap < kern && tap <= j; ++tap) {
                const float w = base[((size_t)side * kern + tap) * cols + c] +
                                dyn[(size_t)j * 2 * kern * ng + ((size_t)side * kern + tap) * ng + g];
                sum += w * h[(size_t)(j - tap) * cols + c];
            }
            out[(size_t)j * cols + c] = sum;
        }
    }
}

/* ---------- the LiLiCorr head (the GPU's kernels/nemotron/lattice.cu; lilicorr.py lattice) ---------- */

/* GP_NM_LTOPK: logits (S x V), S, V, k, cand (S x k int32), feat (S k x 5): the k best of each slot (by
 * logit, then index) and [log p, p, log p - log p0, rank / (k - 1), rank == 0] */
static void nm_ltopk_body(const float *x, int S, int V, int k, int *cand, float *feat)
{
    #pragma omp for schedule(static)
    for (int s = 0; s < S; ++s) {
        const float *l = x + (size_t)s * V;
        float tv[16];
        int ti[16];
        float mx = -INFINITY;
        for (int q = 0; q < k; ++q) {
            tv[q] = -INFINITY;
            ti[q] = 0x7fffffff;
        }
        for (int i = 0; i < V; ++i) {
            float cv = l[i];
            int ci = i;
            mx = cv > mx ? cv : mx;
            if (cv > tv[k - 1]) {
                for (int q = 0; q < k; ++q) {
                    if (cv > tv[q]) {
                        float t = tv[q];
                        int u = ti[q];
                        tv[q] = cv, ti[q] = ci, cv = t, ci = u;
                    }
                }
            }
        }
        float sum = 0.f;
        for (int i = 0; i < V; ++i) {
            sum += expf(l[i] - mx);
        }
        const float lse = mx + logf(sum), lp0 = logf(expf(tv[0] - lse));
        for (int q = 0; q < k; ++q) {
            const float p = expf(tv[q] - lse), lp = logf(p);
            float *f = feat + (size_t)(s * k + q) * 5;
            cand[s * k + q] = ti[q];
            f[0] = lp, f[1] = p, f[2] = lp - lp0, f[3] = (float)q / (float)(k > 1 ? k - 1 : 1), f[4] = q == 0;
        }
    }
}

/* GP_NM_LATT: qkv (n x 3H), bias (heads x n x n), out (n x H), n, heads, hd */
static void nm_latt_body(const float *qkv, const float *bias, float *out, int n, int heads, int hd)
{
    const int H = heads * hd;
    #pragma omp for schedule(static)
    for (int x = 0; x < n * heads; ++x) {
        const int i = x / heads, h = x % heads;
        float sc[64], m = -INFINITY, z = 0.f;
        const float rs = sqrtf((float)hd);
        for (int j = 0; j < n; ++j) {
            const float *q = qkv + (size_t)i * 3 * H + h * hd, *kr = qkv + (size_t)j * 3 * H + H + h * hd;
            float a = 0.f;
            for (int d = 0; d < hd; ++d) {
                a += q[d] * kr[d];
            }
            sc[j] = a / rs + bias[((size_t)h * n + i) * n + j];
            m = sc[j] > m ? sc[j] : m;
        }
        for (int j = 0; j < n; ++j) {
            sc[j] = expf(sc[j] - m);
            z += sc[j];
        }
        float *o = out + (size_t)i * H + h * hd;
        for (int d = 0; d < hd; ++d) {
            o[d] = 0.f;
        }
        for (int j = 0; j < n; ++j) {
            const float p = sc[j] / z, *v = qkv + (size_t)j * 3 * H + 2 * H + h * hd;
            for (int d = 0; d < hd; ++d) {
                o[d] += p * v[d];
            }
        }
    }
}

/* GP_NM_LROWS: x, b (or none), y (or none), out, rows, cols, rep, act: out = act(x + b + y[r / rep]) */
static void nm_lrows_body(const float *x, const float *b, const float *y, float *out, int rows, int cols, int rep,
                          int act)
{
    #pragma omp for schedule(static)
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            float v = x[(size_t)r * cols + c] + (b != NULL ? b[c] : 0.f) + (y != NULL ? y[(size_t)(r / rep) * cols + c] : 0.f);
            out[(size_t)r * cols + c] = act == 1 ? v / (1.f + expf(-v)) : v;
        }
    }
}

/* GP_NM_LCAT: x (n x H), a (H), out (n x 3H), n, H */
static void nm_lcat_body(const float *x, const float *a, float *out, int n, int H)
{
    #pragma omp for schedule(static)
    for (int r = 0; r < n; ++r) {
        for (int c = 0; c < H; ++c) {
            const float v = x[(size_t)r * H + c];
            out[(size_t)r * 3 * H + c] = v;
            out[(size_t)r * 3 * H + H + c] = a[c];
            out[(size_t)r * 3 * H + 2 * H + c] = v * a[c];
        }
    }
}

static float nm_ldot(const float *a, const float *b, int D)
{
    float s = 0.f;
    for (int d = 0; d < D; ++d) {
        s += a[d] * b[d];
    }
    return s;
}

/* GP_NM_LSCORE: ov, iv (S k x D), anc (D), scores (S x k x k), S, k, D, scale, eps */
static void nm_lscore_body(const float *ov, const float *iv, const float *anc, float *sc, int S, int k, int D,
                           float scale, float eps)
{
    #pragma omp for schedule(static)
    for (int x = 0; x < S * k * k; ++x) {
        const int s = x / (k * k), p = (x / k) % k, c = x % k;
        const float *a = s == 0 ? anc : ov + (size_t)((s - 1) * k + p) * D, *b = iv + (size_t)(s * k + c) * D;
        float v = 0.f;
        if (s > 0 || p == 0) {
            const float na = fmaxf(sqrtf(nm_ldot(a, a, D)), eps), nb = fmaxf(sqrtf(nm_ldot(b, b, D)), eps);
            v = scale * nm_ldot(a, b, D) / (na * nb);
        }
        sc[x] = v;
    }
}

static void nm_relu2_body(const float *x, float *out, int64_t n)
{
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        float r = x[i] > 0.f ? x[i] : 0.f;
        out[i] = r * r;
    }
}

/* The bytes of the scratch of GP_NM_MOE for t tokens. */
static size_t nm_moe_scratch_bytes(int t, int k, int L, int inner)
{
    size_t np = (size_t)t * k;
    return (size_t)t * L + (size_t)t * L / 32 * 4 + (size_t)t * L / 16 * 4      /* xq, xs, xm */
           + np * inner * 4 + np * inner + np * inner / 32 * 4 + np * inner / 16 * 4   /* act */
           + np * L * 4                                                    /* the pair rows */
           + np * 4 * 3 + 64 * 8                                           /* order, tasks */
           + (size_t)t * L + np * inner                                    /* the low planes (x16) */
           + ((size_t)t * L + np * inner) * 2 + np * 8 * 3 + 64;           /* int16 x, act (the tiles) */
}

/* nm_moe_body for a few pairs (a step, a small group): one barrier. Each thread quantizes x for
 * itself; a task of the up rows (a pair, 64 rows) quantizes the act of its rows (two parts of
 * 32); a task of the down rows (NM_DCH rows) sums the slots of each token in their order, so no
 * pass adds the pairs afterwards (the same bits as nm_moe_body for any count of threads, not the
 * same as it). */
#define NM_FEW 32
/* the record being run is the last of its program (interp.c gp_exec sets it in each thread): a record may
 * then end with no barrier of its own, the end of the parallel region being one */
static __thread int nm_last_record;
#define NM_MOE_TN 3
#define NM_DCH 16
static void nm_moe_few(const float *x, const int32_t *ids, const float *val, int t, int k,
                       const uint8_t *up, int up_type, const uint8_t *down, int down_type, int L,
                       int inner, uint8_t *scratch, float *out, int x16)
{
    const int np = t * k, CH = 64;
    uint8_t *p = scratch;
    p += (size_t)t * L + (size_t)t * L / 32 * 4 + (size_t)t * L / 16 * 4;     /* (the shared x) */
    float *act = (float *)p;           p += (size_t)np * inner * 4;
    int8_t *aq = (int8_t *)p;          p += (size_t)np * inner;
    float *as = (float *)p;            p += (size_t)np * inner / 32 * 4;
    float *am = (float *)p;            p += (size_t)np * inner / 16 * 4;
    p += (size_t)np * L * 4 + (size_t)np * 4 * 3 + 64 * 8 + (size_t)t * L;
    int8_t *al = (int8_t *)p;                                         /* the low planes of the act */
    const size_t urb = kq_row_bytes(up_type, L), drb = kq_row_bytes(down_type, inner);
    int8_t xq[NM_FEW][1024], xl[NM_FEW][1024];
    float xs[NM_FEW][1024 / 32], xm[NM_FEW][1024 / 16];
    if (t > NM_FEW || L > 1024) {
        return;
    }
    for (int j = 0; j < t; ++j) {
        for (int g = 0; g < L / 32; ++g) {
            if (x16) {
                kq_quant_part16(x + (size_t)j * L, g, xq[j], xl[j], xs[j], xm[j]);
            } else {
                kq_quant_part(x + (size_t)j * L, g, xq[j], xs[j], xm[j]);
            }
        }
    }
#if defined(__AVX2__)
    /* KQ_NV4 experts (Super) on int16 x: the rows 16 at a time by the rows forms of nmx2.c (kq_rows_x2_nv4v
     * with VNNI, kq_rows_x2_nv4 without: the bits of kq_dot_x2 there); x of each token as int16 once */
    int16_t xpu[4][1024] __attribute__((aligned(64)));
    static int rows_on = -1;            /* RECLANK_NM_NV4_ROWS=0: the rows one at a time (a measure) */
    if (rows_on < 0) {
        const char *e = getenv("RECLANK_NM_NV4_ROWS");
        rows_on = e == NULL || atoi(e) != 0;
    }
    const int nv4u = rows_on && x16 && up_type == KQ_NV4 && t <= 4 && L % 32 == 0;
    const int nv4d = rows_on && x16 && down_type == KQ_NV4 && t <= 4 && inner % 32 == 0 && k <= NM_FEW;
#if defined(NM_X2_VNNI)
    const int uspan = kq_nv4v_span(L), dspan = kq_nv4v_span(inner);
    int uok = 1;
#else
    const int uspan = 32, dspan = 32;
    float fdu[4][1024 / 32];
    int uok = 1;
#endif
    if (nv4u) {
        for (int j = 0; j < t; ++j) {
            kq_nv4_x16(xq[j], xl[j], L, uspan, xpu[j]);
#if !defined(NM_X2_VNNI)
            uok &= kq_nv4_fd(xs[j], L, fdu[j]);
#endif
        }
    }
    /* the int16 act of each pair (made after the up rows, before the down rows: scratch past the low
     * planes, the region of the tiles' int16 act) */
    int16_t *ap = (int16_t *)(al + (size_t)np * inner + (size_t)t * L * 2);
#if defined(NM_X2_VNNI)
    static int runs_on = -1;            /* RECLANK_NM_NV4_RUNS=0: the tasks by pair (a measure) */
    if (runs_on < 0) {
        const char *e = getenv("RECLANK_NM_NV4_RUNS");
        runs_on = e == NULL || atoi(e) != 0;
    }
    if (runs_on && nv4u && nv4d && inner % CH == 0 && CH % dspan == 0 && np <= 4 * NM_FEW) {
        /* by expert (a verify group: the tokens share experts): a task reads its rows of an expert once
         * for all the pairs of the expert (the rows of a block in the L1 for the second token on), the
         * kernels and the sums as by pair: the same bits. Each thread makes the runs (pairs by expert,
         * in the order of the pairs) for itself. The up tasks also make the int16 act of their rows
         * (the 64-spans of kq_nv4_x16 stay inside a task); the down tasks write the row of each pair
         * (po: the pair rows' region of the scratch), then the slots of each token in their order. */
        float *po = (float *)(am + (size_t)np * inner / 16);
        int rex[4 * NM_FEW], rst[4 * NM_FEW + 1], rpr[4 * NM_FEW], nr = 0, n = 0;
        for (int pr = 0; pr < np; ++pr) {
            if (ids[pr] >= 0) {
                rpr[n++] = pr;
            }
        }
        for (int a = 1; a < n; ++a) {          /* insertion sort by expert, stable: np is small */
            int v = rpr[a], b = a - 1;
            while (b >= 0 && ids[rpr[b]] > ids[v]) {
                rpr[b + 1] = rpr[b];
                --b;
            }
            rpr[b + 1] = v;
        }
        for (int a = 0; a < n; ++a) {
            if (a == 0 || ids[rpr[a]] != ids[rpr[a - 1]]) {
                rex[nr] = ids[rpr[a]];
                rst[nr++] = a;
            }
        }
        rst[nr] = n;
        const int uch = inner / CH;
        #pragma omp single nowait
        {
            int *dc = (int *)((float *)(am + (size_t)np * inner / 16) + (size_t)np * L);
            for (int b = 0; b < L / 32; ++b) {
                dc[b] = 0;
            }
        }
        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < nr * uch; ++task) {
            const int ru = task / uch, r0 = (task % uch) * CH;
            const uint8_t *w = up + (size_t)rex[ru] * inner * urb;
            for (int r = r0; r < r0 + CH; r += 16) {
                for (int a = rst[ru]; a < rst[ru + 1]; ++a) {
                    const int pr = rpr[a], j = pr / k;
                    kq_rows_x2_nv4v(w + (size_t)r * urb, urb, L, 16, xpu[j], xs[j], 1, act + (size_t)pr * inner + r, 1);
                }
            }
            for (int a = rst[ru]; a < rst[ru + 1]; ++a) {
                const int pr = rpr[a];
                float *ar = act + (size_t)pr * inner;
                for (int r = r0; r < r0 + CH; ++r) {
                    float v = ar[r] > 0.f ? ar[r] : 0.f;
                    ar[r] = v * v;
                }
                for (int g = r0 / 32; g < (r0 + CH) / 32; ++g) {
                    kq_quant_part16(ar, g, aq + (size_t)pr * inner, al + (size_t)pr * inner, as + (size_t)pr * (inner / 32),
                                    am + (size_t)pr * (inner / 16));
                }
                kq_nv4_x16(aq + (size_t)pr * inner + r0, al + (size_t)pr * inner + r0, CH, dspan, ap + (size_t)pr * inner + r0);
            }
        }
        /* the sums: the task that finishes the last run of a block of DR rows (dcnt, an atomic count of
         * each block's runs; zeroed before the barrier after the up tasks) adds that block's rows of
         * each token over its slots, in their order (no barrier between the down tasks and the sums) */
        const int DR = 32, dch = L / DR;
        int *dcnt = (int *)(po + (size_t)np * L);
        const int last = nm_last_record;
        #pragma omp for schedule(dynamic, 1) nowait
        for (int task = 0; task < nr * dch; ++task) {
            const int ru = task / dch, r0 = (task % dch) * DR;
            const uint8_t *w = down + ((size_t)rex[ru] * L + r0) * drb;
            for (int r = 0; r < DR; r += 16) {
                for (int a = rst[ru]; a < rst[ru + 1]; ++a) {
                    const int pr = rpr[a];
                    kq_rows_x2_nv4v(w + (size_t)r * drb, drb, inner, 16, ap + (size_t)pr * inner, as + (size_t)pr * (inner / 32),
                                    1, po + (size_t)pr * L + r0 + r, 1);
                }
            }
            if (__atomic_add_fetch(&dcnt[r0 / DR], 1, __ATOMIC_ACQ_REL) == nr) {
                for (int j = 0; j < t; ++j) {
                    for (int r = r0; r < r0 + DR; ++r) {
                        float s = 0.f;
                        for (int q = 0; q < k; ++q) {
                            const int pr = j * k + q;
                            if (ids[pr] < 0) {
                                continue;
                            }
                            s += val[pr] * po[(size_t)pr * L + r];
                        }
                        out[(size_t)j * L + r] = s;
                    }
                }
            }
        }
        if (nr == 0) {
            #pragma omp for schedule(static) nowait
            for (int x0 = 0; x0 < t * L; ++x0) {
                out[x0] = 0.f;
            }
        }
        if (!last) {
            #pragma omp barrier
        }
        return;
    }
#endif
#endif
    const int uch = (inner + CH - 1) / CH;
    #pragma omp for schedule(dynamic, 1)
    for (int task = 0; task < np * uch; ++task) {
        int pr = task / uch, r0 = (task % uch) * CH, r1 = r0 + CH < inner ? r0 + CH : inner, j = pr / k;
        if (ids[pr] < 0) {
            continue;
        }
        const uint8_t *w = up + (size_t)ids[pr] * inner * urb;
        float *a = act + (size_t)pr * inner;
#if defined(__AVX2__)
        if (nv4u && uok) {
            for (int r = r0; r < r1; r += 16) {
                int nr = r1 - r < 16 ? r1 - r : 16;
#if defined(NM_X2_VNNI)
                kq_rows_x2_nv4v(w + (size_t)r * urb, urb, L, nr, xpu[j], xs[j], 1, a + r, 1);
#else
                kq_rows_x2_nv4(w + (size_t)r * urb, urb, L, nr, xpu[j], fdu[j], 1, a + r, 1);
#endif
            }
            for (int r = r0; r < r1; ++r) {
                float v = a[r] > 0.f ? a[r] : 0.f;
                a[r] = v * v;
            }
        } else if (x16 && (up_type == KQ_IQ2R || up_type == KQ_IQ2_XXS)) {
            /* (the rows of the task 4 at a time: kq_rows_x2_iq2, the bits of kq_dot_x2) */
            kq_rows_x2_iq2(w + (size_t)r0 * urb, urb, up_type, L, r1 - r0, xq[j], xl[j], xs[j], 1, a + r0, 1, 0);
            for (int r = r0; r < r1; ++r) {
                float v = a[r] > 0.f ? a[r] : 0.f;
                a[r] = v * v;
            }
        } else
#endif
        for (int r = r0; r < r1; ++r) {
            float v = x16 ? kq_dot_x2(w + (size_t)r * urb, up_type, L, xq[j], xl[j], xs[j])
                          : kq_dot1(w + (size_t)r * urb, up_type, L, xq[j], xs[j], xm[j], NULL);
            v = v > 0.f ? v : 0.f;
            a[r] = v * v;
        }
        for (int g = r0 / 32; g < r1 / 32; ++g) {
            if (x16) {
                kq_quant_part16(a, g, aq + (size_t)pr * inner, al + (size_t)pr * inner, as + (size_t)pr * (inner / 32),
                                am + (size_t)pr * (inner / 16));
            } else {
                kq_quant_part(a, g, aq + (size_t)pr * inner, as + (size_t)pr * (inner / 32), am + (size_t)pr * (inner / 16));
            }
        }
    }
    const int dch = (L + NM_DCH - 1) / NM_DCH;
#if defined(__AVX2__)
    if (nv4d) {
        /* the act of each pair as int16 once, then the rows of each token 16 at a time on each of its
         * pairs, each row's sum over the slots in their order (the loop below's) */
        #pragma omp for schedule(static)
        for (int pr = 0; pr < np; ++pr) {
            if (ids[pr] >= 0) {
                kq_nv4_x16(aq + (size_t)pr * inner, al + (size_t)pr * inner, inner, dspan, ap + (size_t)pr * inner);
            }
        }
        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < t * dch; ++task) {
            int j = task / dch, r0 = (task % dch) * NM_DCH, r1 = r0 + NM_DCH < L ? r0 + NM_DCH : L;
            float dq[NM_FEW][NM_DCH];
            for (int q = 0; q < k; ++q) {
                int pr = j * k + q;
                if (ids[pr] < 0) {
                    continue;
                }
                const uint8_t *w = down + ((size_t)ids[pr] * L + r0) * drb;
                const float *asr = as + (size_t)pr * (inner / 32);
#if defined(NM_X2_VNNI)
                kq_rows_x2_nv4v(w, drb, inner, r1 - r0, ap + (size_t)pr * inner, asr, 1, dq[q], 1);
#else
                float fd[4096 / 32];
                if (inner <= 4096 && kq_nv4_fd(asr, inner, fd)) {
                    kq_rows_x2_nv4(w, drb, inner, r1 - r0, ap + (size_t)pr * inner, fd, 1, dq[q], 1);
                } else {
                    for (int r = r0; r < r1; ++r) {
                        dq[q][r - r0] = kq_dot_x2(down + ((size_t)ids[pr] * L + r) * drb, down_type, inner,
                                                  aq + (size_t)pr * inner, al + (size_t)pr * inner, asr);
                    }
                }
#endif
            }
            for (int r = r0; r < r1; ++r) {
                float s = 0.f;
                for (int q = 0; q < k; ++q) {
                    int pr = j * k + q;
                    if (ids[pr] < 0) {
                        continue;
                    }
                    s += val[pr] * dq[q][r - r0];
                }
                out[(size_t)j * L + r] = s;
            }
        }
        return;
    }
    if (x16 && (down_type == KQ_IQ2R || down_type == KQ_IQ2_XXS) && k <= NM_FEW) {
        /* the rows of each pair 4 at a time (kq_rows_x2_iq2), then each row's sum over the slots in
         * their order: the bits of the loop below */
        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < t * dch; ++task) {
            int j = task / dch, r0 = (task % dch) * NM_DCH, r1 = r0 + NM_DCH < L ? r0 + NM_DCH : L;
            float dq[NM_FEW][NM_DCH];
            for (int q = 0; q < k; ++q) {
                int pr = j * k + q;
                if (ids[pr] < 0) {
                    continue;
                }
                kq_rows_x2_iq2(down + ((size_t)ids[pr] * L + r0) * drb, drb, down_type, inner, r1 - r0,
                               aq + (size_t)pr * inner, al + (size_t)pr * inner, as + (size_t)pr * (inner / 32), 1,
                               dq[q], 1, 0);
            }
            for (int r = r0; r < r1; ++r) {
                float s = 0.f;
                for (int q = 0; q < k; ++q) {
                    int pr = j * k + q;
                    if (ids[pr] < 0) {
                        continue;
                    }
                    s += val[pr] * dq[q][r - r0];
                }
                out[(size_t)j * L + r] = s;
            }
        }
        return;
    }
#endif
    #pragma omp for schedule(dynamic, 1)
    for (int task = 0; task < t * dch; ++task) {
        int j = task / dch, r0 = (task % dch) * NM_DCH, r1 = r0 + NM_DCH < L ? r0 + NM_DCH : L;
        for (int r = r0; r < r1; ++r) {
            float s = 0.f;
            for (int q = 0; q < k; ++q) {
                int pr = j * k + q;
                if (ids[pr] < 0) {
                    continue;
                }
                const uint8_t *w = down + ((size_t)ids[pr] * L + r) * drb;
                s += val[pr] * (x16 ? kq_dot_x2(w, down_type, inner, aq + (size_t)pr * inner, al + (size_t)pr * inner,
                                                as + (size_t)pr * (inner / 32))
                                    : kq_dot1(w, down_type, inner, aq + (size_t)pr * inner, as + (size_t)pr * (inner / 32),
                                              am + (size_t)pr * (inner / 16), NULL));
            }
            out[(size_t)j * L + r] = s;
        }
    }
}

/* up1, down1 (or NULL): the copies of the stacks on node 1 (the weightkeeper's, the same bytes): the
 * threads of node 1 read them, those of node 0 up and down (the team pinned by node: run.c), so every
 * expert comes from the memory of the reading thread's node; the same bits */
static void nm_moe_body(const float *x, const int32_t *ids, const float *val, int t, int k,
                        const uint8_t *up, int up_type, const uint8_t *down, int down_type, int L,
                        int inner, uint8_t *scratch, float *out, int x16, const uint8_t *up1,
                        const uint8_t *down1)
{
    const int np = t * k, CH = 64;
    if (up1 != NULL && down1 != NULL && kq_my_node() == 1) {
        up = up1;
        down = down1;
    }
    if (t <= 4 && L <= 1024 && inner % 64 == 0) {
        nm_moe_few(x, ids, val, t, k, up, up_type, down, down_type, L, inner, scratch, out, x16);
        return;
    }
    uint8_t *p = scratch;
    int8_t *xq = (int8_t *)p;          p += (size_t)t * L;
    float *xs = (float *)p;            p += (size_t)t * L / 32 * 4;
    float *xm = (float *)p;            p += (size_t)t * L / 16 * 4;
    float *act = (float *)p;           p += (size_t)np * inner * 4;
    int8_t *aq = (int8_t *)p;          p += (size_t)np * inner;
    float *as = (float *)p;            p += (size_t)np * inner / 32 * 4;
    float *am = (float *)p;            p += (size_t)np * inner / 16 * 4;
    float *po = (float *)p;            p += (size_t)np * L * 4;
    int32_t *order = (int32_t *)p;     p += (size_t)np * 4;
    int32_t *runs = (int32_t *)p;      p += (size_t)np * 4;   /* the start of each run in order */
    int32_t *nruns = (int32_t *)p;     p += 64 * 8;
    int8_t *xl = (int8_t *)p;          p += (size_t)t * L;        /* the low planes (x16) */
    int8_t *al = (int8_t *)p;
    const size_t urb = kq_row_bytes(up_type, L), drb = kq_row_bytes(down_type, inner);
    /* x to int8 */
    #pragma omp for schedule(static)
    for (int x0 = 0; x0 < t * (L / 32); ++x0) {
        int j = x0 / (L / 32), g = x0 % (L / 32);
        if (x16) {
            kq_quant_part16(x + (size_t)j * L, g, xq + (size_t)j * L, xl + (size_t)j * L, xs + (size_t)j * (L / 32),
                            xm + (size_t)j * (L / 16));
        } else {
            kq_quant_part(x + (size_t)j * L, g, xq + (size_t)j * L, xs + (size_t)j * (L / 32),
                          xm + (size_t)j * (L / 16));
        }
    }
    /* the pairs by expert (a stable sort: the order of the tokens in each run) */
    #pragma omp single
    {
        int n = 0;
        for (int a = 0; a < np; ++a) {
            if (ids[a] >= 0) {
                order[n++] = a;
            }
        }
        for (int a = 1; a < n; ++a) {          /* insertion sort by expert: np is small */
            int v = order[a], b = a - 1;
            while (b >= 0 && ids[order[b]] > ids[v]) {
                order[b + 1] = order[b];
                --b;
            }
            order[b + 1] = v;
        }
        int r = 0;
        for (int a = 0; a < n; ++a) {
            if (a == 0 || ids[order[a]] != ids[order[a - 1]]) {
                runs[r++] = a;
            }
        }
        runs[r] = n;
        nruns[0] = r;
        /* pairs without an expert give zero rows */
        for (int a = 0; a < np; ++a) {
            if (ids[a] < 0) {
                memset(po + (size_t)a * L, 0, (size_t)L * 4);
            }
        }
    }
    const int nr = nruns[0];
#if defined(__AVX2__)
    if (x16 && t >= NM_TILE_T && inner % 64 == 0 && L % 64 == 0 && nm_tiled_ok(up_type, inner, L) &&
        nm_tiled_ok(down_type, L, inner)) {
        /* the tiled form (nmx2.c) for the runs of NM_MOE_TN tokens or more, the rows of a pair for the
         * others: x and the act as int16 */
        p = (uint8_t *)al + (size_t)np * inner;
        int16_t *x16 = (int16_t *)p;       p += (size_t)t * L * 2;
        int16_t *a16 = (int16_t *)p;       p += (size_t)np * inner * 2;
        p = (uint8_t *)(((uintptr_t)p + 63) & ~(uintptr_t)63);
        const void **xr = (const void **)p, **sr = xr + np;
        void **orow = (void **)(sr + np);
        #pragma omp for schedule(static)
        for (int j = 0; j < t; ++j) {
            nm_x16_row(xq + (size_t)j * L, xl + (size_t)j * L, L, x16 + (size_t)j * L);
        }
        #pragma omp for schedule(static)
        for (int a = 0; a < runs[nr]; ++a) {
            int pr = order[a], j = pr / k;
            xr[a] = x16 + (size_t)j * L;
            sr[a] = xs + (size_t)j * (L / 32);
            orow[a] = act + (size_t)pr * inner;
        }
        const int R = 64, uch = inner / R;
        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < nr * uch; ++task) {
            const int ru = task / uch, r0 = (task % uch) * R, a0 = runs[ru], n = runs[ru + 1] - a0;
            const uint8_t *w = up + (size_t)ids[order[a0]] * inner * urb;
            if (n >= NM_MOE_TN) {
                nm_task_tiled(w, urb, up_type, L, r0, r0 + R, n, (const int16_t *const *)(xr + a0),
                              (const float *const *)(sr + a0), (float *const *)(orow + a0));
            } else {
                for (int a = a0; a < a0 + n; ++a) {
                    int pr = order[a], j = pr / k;
                    for (int r = r0; r < r0 + R; ++r) {
                        act[(size_t)pr * inner + r] = kq_dot_x2(w + (size_t)r * urb, up_type, L, xq + (size_t)j * L,
                                                                xl + (size_t)j * L, xs + (size_t)j * (L / 32));
                    }
                }
            }
            for (int a = a0; a < a0 + n; ++a) {
                float *v = act + (size_t)order[a] * inner;
                for (int r = r0; r < r0 + R; ++r) {
                    float y = v[r] > 0.f ? v[r] : 0.f;
                    v[r] = y * y;
                }
            }
        }
        /* the act to int16 (two planes, then 128 hi + lo) */
        #pragma omp for schedule(static)
        for (int x0 = 0; x0 < np * (inner / 32); ++x0) {
            int pr = x0 / (inner / 32), g = x0 % (inner / 32);
            if (ids[pr] < 0) {
                continue;
            }
            kq_quant_part16(act + (size_t)pr * inner, g, aq + (size_t)pr * inner, al + (size_t)pr * inner,
                            as + (size_t)pr * (inner / 32), am + (size_t)pr * (inner / 16));
            nm_x16_row(aq + (size_t)pr * inner + 32 * g, al + (size_t)pr * inner + 32 * g, 32, a16 + (size_t)pr * inner + 32 * g);
        }
        #pragma omp for schedule(static)
        for (int a = 0; a < runs[nr]; ++a) {
            int pr = order[a];
            xr[a] = a16 + (size_t)pr * inner;
            sr[a] = as + (size_t)pr * (inner / 32);
            orow[a] = po + (size_t)pr * L;
        }
        const int dch = L / R;
        #pragma omp for schedule(dynamic, 1)
        for (int task = 0; task < nr * dch; ++task) {
            const int ru = task / dch, r0 = (task % dch) * R, a0 = runs[ru], n = runs[ru + 1] - a0;
            const uint8_t *w = down + (size_t)ids[order[a0]] * L * drb;
            if (n >= NM_MOE_TN) {
                nm_task_tiled(w, drb, down_type, inner, r0, r0 + R, n, (const int16_t *const *)(xr + a0),
                              (const float *const *)(sr + a0), (float *const *)(orow + a0));
                for (int a = a0; a < a0 + n; ++a) {
                    int pr = order[a];
                    for (int r = r0; r < r0 + R; ++r) {
                        po[(size_t)pr * L + r] *= val[pr];
                    }
                }
            } else {
                for (int a = a0; a < a0 + n; ++a) {
                    int pr = order[a];
                    for (int r = r0; r < r0 + R; ++r) {
                        po[(size_t)pr * L + r] = val[pr] * kq_dot_x2(w + (size_t)r * drb, down_type, inner, aq + (size_t)pr * inner,
                                                                     al + (size_t)pr * inner, as + (size_t)pr * (inner / 32));
                    }
                }
            }
        }
        #pragma omp for schedule(static)
        for (int x0 = 0; x0 < t * L; ++x0) {
            int j = x0 / L, r = x0 % L;
            float s = 0.f;
            for (int q = 0; q < k; ++q) {
                s += po[((size_t)j * k + q) * L + r];
            }
            out[x0] = s;
        }
        return;
    }
#endif
    /* up, relu^2: a task for each run and each CH rows */
    const int uch = (inner + CH - 1) / CH;
    #pragma omp for schedule(dynamic, 1)
    for (int task = 0; task < nr * uch; ++task) {
        int ru = task / uch, r0 = (task % uch) * CH, r1 = r0 + CH < inner ? r0 + CH : inner;
        const uint8_t *w = up + (size_t)ids[order[runs[ru]]] * inner * urb;
        for (int r = r0; r < r1; ++r) {
            for (int a = runs[ru]; a < runs[ru + 1]; ++a) {
                int pr = order[a], j = pr / k;
                float v = x16 ? kq_dot_x2(w + (size_t)r * urb, up_type, L, xq + (size_t)j * L, xl + (size_t)j * L,
                                          xs + (size_t)j * (L / 32))
                              : kq_dot1(w + (size_t)r * urb, up_type, L, xq + (size_t)j * L,
                                        xs + (size_t)j * (L / 32), xm + (size_t)j * (L / 16), NULL);
                v = v > 0.f ? v : 0.f;
                act[(size_t)pr * inner + r] = v * v;
            }
        }
    }
    /* the act to int8 */
    #pragma omp for schedule(static)
    for (int x0 = 0; x0 < np * (inner / 32); ++x0) {
        int pr = x0 / (inner / 32), g = x0 % (inner / 32);
        if (ids[pr] < 0) {
            continue;
        }
        if (x16) {
            kq_quant_part16(act + (size_t)pr * inner, g, aq + (size_t)pr * inner, al + (size_t)pr * inner,
                            as + (size_t)pr * (inner / 32), am + (size_t)pr * (inner / 16));
        } else {
            kq_quant_part(act + (size_t)pr * inner, g, aq + (size_t)pr * inner, as + (size_t)pr * (inner / 32),
                          am + (size_t)pr * (inner / 16));
        }
    }
    /* down, times the weight */
    const int dch = (L + CH - 1) / CH;
    #pragma omp for schedule(dynamic, 1)
    for (int task = 0; task < nr * dch; ++task) {
        int ru = task / dch, r0 = (task % dch) * CH, r1 = r0 + CH < L ? r0 + CH : L;
        const uint8_t *w = down + (size_t)ids[order[runs[ru]]] * L * drb;
        for (int r = r0; r < r1; ++r) {
            for (int a = runs[ru]; a < runs[ru + 1]; ++a) {
                int pr = order[a];
                po[(size_t)pr * L + r] = val[pr] * (x16 ? kq_dot_x2(w + (size_t)r * drb, down_type, inner,
                                                                     aq + (size_t)pr * inner, al + (size_t)pr * inner,
                                                                     as + (size_t)pr * (inner / 32))
                                                          : kq_dot1(w + (size_t)r * drb, down_type, inner,
                                                                    aq + (size_t)pr * inner, as + (size_t)pr * (inner / 32),
                                                                    am + (size_t)pr * (inner / 16), NULL));
            }
        }
    }
    /* the sum over the slots */
    #pragma omp for schedule(static)
    for (int x0 = 0; x0 < t * L; ++x0) {
        int j = x0 / L, r = x0 % L;
        float s = 0.f;
        for (int q = 0; q < k; ++q) {
            s += po[((size_t)j * k + q) * L + r];
        }
        out[x0] = s;
    }
}

size_t rc_nm_moe_scratch_bytes(int t, int k, int L, int inner)
{
    return nm_moe_scratch_bytes(t, k, L, inner);
}
