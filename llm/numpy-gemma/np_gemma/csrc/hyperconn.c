/* The gated residual (hyper-connections) and the n-gram layer (PLE) of
 * qwen4exp (Qwen3.8-Flash-Next; np_gemma/qwen4.py has the math in NumPy).
 *
 * This file is part of the cops library: bf16_linear.c includes it. The
 * products (down, up, inject, key, value) are KQ_LINEAR records; these
 * records do the rest. The residual H has t rows of hc streams of hid
 * values (t x hc * hid).
 *
 *     HC_NORM   the grouped RMS norm: each stream of each row, times its
 *               weights (hc * hid values, with the 1 in them)
 *     HC_ACT    out = silu(x * scale)
 *     HC_MIX    the input of a block: the mean over the streams of
 *               sigmoid(g) * hn
 *     HC_ADD    H[s] += out * 2 sigmoid(inject[s] * scale)
 *     PLE_GATE  the gate of each stream from the key and the normed
 *               streams, times the value
 *     PLE_CONV  the dilated depthwise convolution of the normed gated
 *               values (its last inputs in the cache), plus the gated
 *               values, added to H
 */

static inline float hc_sigmoid(float v)
{
    return 1.f / (1.f + expf(-v));
}

/* HC_NORM: x (t x groups * hid), w (groups * hid), out, t, groups, hid, eps. */
static void hc_norm_body(const float *x, const float *w, float *out, int t, int groups, int hid,
                         float eps)
{
    #pragma omp for schedule(static)
    for (int rg = 0; rg < t * groups; ++rg) {
        const float *xr = x + (size_t)rg * hid;
        const float *wr = w + (size_t)(rg % groups) * hid;
        float *o = out + (size_t)rg * hid;
        float ss = 0.f;
        for (int c = 0; c < hid; ++c) {
            ss += xr[c] * xr[c];
        }
        float inv = 1.f / sqrtf(ss / (float)hid + eps);
        for (int c = 0; c < hid; ++c) {
            o[c] = xr[c] * inv * wr[c];
        }
    }
}

/* HC_ACT: x, out, n, scale. out = silu(x * scale). */
static void hc_act_body(const float *x, float *out, int64_t n, float scale)
{
    #pragma omp for schedule(static)
    for (int64_t i = 0; i < n; ++i) {
        float v = x[i] * scale;
        out[i] = v * hc_sigmoid(v);
    }
}

/* HC_MIX: hn, g, out, t, hc, hid. out (t x hid) = the mean over the
 * streams of sigmoid(g) * hn. */
static void hc_mix_body(const float *hn, const float *g, float *out, int t, int hc, int hid)
{
    int nb = hid / 64;
    #pragma omp for schedule(static)
    for (int x = 0; x < t * nb; ++x) {
        int j = x / nb, c0 = (x % nb) * 64;
        float *o = out + (size_t)j * hid;
        for (int c = c0; c < c0 + 64; ++c) {
            float s = 0.f;
            for (int k = 0; k < hc; ++k) {
                size_t i = ((size_t)j * hc + k) * hid + c;
                s += hc_sigmoid(g[i]) * hn[i];
            }
            o[c] = s / (float)hc;
        }
    }
}

/* HC_ADD: H, out, inject, t, hc, hid, scale. Each stream s of each row
 * gets out * 2 sigmoid(inject[s] * scale). */
static void hc_add_body(float *H, const float *out, const float *inject, int t, int hc, int hid,
                        float scale)
{
    int nb = hid / 64;
    #pragma omp for schedule(static)
    for (int x = 0; x < t * nb; ++x) {
        int j = x / nb, c0 = (x % nb) * 64;
        for (int k = 0; k < hc; ++k) {
            float w = 2.f * hc_sigmoid(inject[(size_t)j * hc + k] * scale);
            float *h = H + ((size_t)j * hc + k) * hid;
            const float *o = out + (size_t)j * hid;
            for (int c = c0; c < c0 + 64; ++c) {
                h[c] += o[c] * w;
            }
        }
    }
}

/* PLE_GATE: keyn, qn, value, gated, t, hc, hid. For each stream s: the gate
 * s = sum(keyn * qn) / sqrt(hid), then sigmoid(sign(s) sqrt(max(|s|, 1e-6))),
 * and gated[s] = gate * value (t x hc * hid). */
static void ple_gate_body(const float *keyn, const float *qn, const float *value, float *gated,
                          int t, int hc, int hid)
{
    #pragma omp for schedule(static)
    for (int rg = 0; rg < t * hc; ++rg) {
        const float *k = keyn + (size_t)rg * hid, *q = qn + (size_t)rg * hid;
        float s = 0.f;
        for (int c = 0; c < hid; ++c) {
            s += k[c] * q[c];
        }
        s /= sqrtf((float)hid);
        float mag = sqrtf(fabsf(s) > 1e-6f ? fabsf(s) : 1e-6f);
        float g = hc_sigmoid(s < 0.f ? -mag : (s > 0.f ? mag : 0.f));
        const float *v = value + (size_t)(rg / hc) * hid;
        float *o = gated + (size_t)rg * hid;
        for (int c = 0; c < hid; ++c) {
            o[c] = g * v[c];
        }
    }
}

/* PLE_CONV: gn, gated, H, state, w, t, channels, kernel, dilation.
 * out[c] = silu(sum_k w[c][k] gn[t - (kernel - 1 - k) dilation][c]), with
 * the last (kernel - 1) dilation rows of gn of the steps before in state;
 * H += gated + out. The call moves the new rows of gn to state. */
static void ple_conv_body(const float *gn, const float *gated, float *H, float *state,
                          const float *w, int t, int channels, int kernel, int dil)
{
    int hist = (kernel - 1) * dil;
    #pragma omp for schedule(static)
    for (int c0 = 0; c0 < channels; c0 += 64) {
        int c1 = c0 + 64 < channels ? c0 + 64 : channels;
        for (int j = 0; j < t; ++j) {
            for (int c = c0; c < c1; ++c) {
                float v = 0.f;
                for (int k = 0; k < kernel; ++k) {
                    int back = (kernel - 1 - k) * dil;
                    /* row j - back of the history: the state, then gn */
                    int r = j - back;
                    float x = r >= 0 ? gn[(size_t)r * channels + c]
                                     : state[(size_t)(hist + r) * channels + c];
                    v += w[(size_t)c * kernel + k] * x;
                }
                size_t i = (size_t)j * channels + c;
                H[i] += gated[i] + v * hc_sigmoid(v);
            }
        }
        /* the new state: the last hist rows of (state, gn) */
        for (int r = 0; r < hist; ++r) {
            int src = t - hist + r;         /* a row of gn, or of the old state */
            for (int c = c0; c < c1; ++c) {
                float x = src >= 0 ? gn[(size_t)src * channels + c]
                                   : state[(size_t)(hist + src) * channels + c];
                /* rows of the old state move up; the copy goes in order, and a
                 * source row of the state is always after its target */
                state[(size_t)r * channels + c] = x;
            }
        }
    }
}

/* HC_CAT: e (t x hid), hn (t x hc * hid), out (t x hc x 2 hid), t, hc, hid.
 * For each stream: the row of e, then the stream (the input of eh_proj of
 * the MTP layer: embedding first, as the converter joins fc_embedding and
 * fc_hidden). */
static void hc_cat_body(const float *e, const float *hn, float *out, int t, int hc, int hid)
{
    #pragma omp for schedule(static)
    for (int rg = 0; rg < t * hc; ++rg) {
        float *o = out + (size_t)rg * 2 * hid;
        memcpy(o, e + (size_t)(rg / hc) * hid, (size_t)hid * 4);
        memcpy(o + hid, hn + (size_t)rg * hid, (size_t)hid * 4);
    }
}
