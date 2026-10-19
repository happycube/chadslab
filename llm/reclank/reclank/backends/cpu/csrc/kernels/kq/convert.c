/* ---------- the conversions of the safetensors loader (reclank/formats/nvfp4_st.py) ---------- */

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

/* lrintf for the values of the encoders (well inside int): the conversion
 * rounds by MXCSR, to the nearest even as lrintf, inline (lrintf went
 * through the PLT) */
static inline int kq_lrint(float v)
{
#if defined(__SSE2__)
    return _mm_cvtss_si32(_mm_set_ss(v));
#else
    return (int)lrintf(v);
#endif
}

/* One row of cols float values to Q8_0 blocks (as quantize_row_q8_0 of ggml). */
static void kq_row_to_q8_0(const float *x, int cols, uint8_t *dst)
{
    for (int b = 0; b < cols / 32; ++b) {
        const float *v = x + 32 * b;
        float amax = 0.f;
        for (int i = 0; i < 32; ++i) {
            amax = fabsf(v[i]) > amax ? fabsf(v[i]) : amax;
        }
        float d = amax / 127.f, id = d != 0.f ? 1.f / d : 0.f;
        uint8_t *blk = dst + (size_t)b * 34;
        uint16_t hd = _cvtss_sh(d, 0);
        memcpy(blk, &hd, 2);
        for (int i = 0; i < 32; ++i) {
            blk[2 + i] = (uint8_t)(int8_t)lrintf(v[i] * id);
        }
    }
}

/* rows x cols values (bf16 if bf16, else float32) to Q8_0 rows. */
void rc_kq_to_q8_0(const void *src, int bf16, int64_t rows, int cols, uint8_t *dst)
{
    #pragma omp parallel
    {
        float *tmp = (float *)malloc((size_t)cols * 4);
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            const float *x;
            if (bf16) {
                const uint16_t *h = (const uint16_t *)src + (size_t)r * cols;
                for (int c = 0; c < cols; ++c) {
                    tmp[c] = kq_bf16(h[c]);
                }
                x = tmp;
            } else {
                x = (const float *)src + (size_t)r * cols;
            }
            kq_row_to_q8_0(x, cols, dst + (size_t)r * kq_row_bytes(KQ_Q8_0, cols));
        }
        free(tmp);
    }
}

/* make_qx_quants of ggml (rmse_type 1: the weights x^2) for n values: the
 * codes + nmax in L, and the scale. Rounding to the nearest even (lrintf) as
 * nearest_int of ggml. */
static float kq_qx_quants(int n, int nmax, const float *x, uint8_t *L)
{
    float max = 0.f, amax = 0.f;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            max = x[i];
        }
    }
    if (amax < 1e-15f) {
        memset(L, 0, (size_t)n);
        return 0.f;
    }
    float iscale = -nmax / max, sumlx = 0.f, suml2 = 0.f;
    for (int i = 0; i < n; ++i) {
        int l = kq_lrint(iscale * x[i]);
        l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
        L[i] = (uint8_t)(l + nmax);
        float w = x[i] * x[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 > 0.f ? sumlx / suml2 : 0.f;
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f * is) / max;
        sumlx = suml2 = 0.f;
        for (int i = 0; i < n; ++i) {
            int l = kq_lrint(iscale * x[i]);
            l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
            float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0.f && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = kq_lrint(iscale * x[i]);
                l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
                L[i] = (uint8_t)(l + nmax);
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    return scale;
}

/* One Q6_K block of 256 values (quantize_row_q6_K_ref of ggml): 16 scales of
 * make_qx_quants, as int8 of a float16 d, then the 6-bit codes. */
static void kq_block_to_q6_k(const float *x, uint8_t *y)
{
    uint8_t L[256];
    float scales[16], max_scale = 0.f, max_abs = 0.f;
    for (int ib = 0; ib < 16; ++ib) {
        float sc = kq_qx_quants(16, 32, x + 16 * ib, L + 16 * ib);
        scales[ib] = sc;
        if (fabsf(sc) > max_abs) {
            max_abs = fabsf(sc);
            max_scale = sc;
        }
    }
    if (max_abs < 1e-15f) {
        memset(y, 0, 210);
        return;
    }
    float iscale = -128.f / max_scale;
    uint16_t dh = kq_f32_to_f16(1.f / iscale);
    float d = kq_h((const uint8_t *)&dh);
    int8_t *sc = (int8_t *)(y + 192);
    for (int ib = 0; ib < 16; ++ib) {
        int v = kq_lrint(iscale * scales[ib]);
        sc[ib] = (int8_t)(v > 127 ? 127 : v);
    }
    for (int j = 0; j < 256; ++j) {
        float dd = d * sc[j / 16];
        if (dd == 0.f) {
            continue;
        }
        int l = kq_lrint(x[j] / dd);
        l = l < -32 ? -32 : (l > 31 ? 31 : l);
        L[j] = (uint8_t)(l + 32);
    }
    uint8_t *ql = y, *qh = y + 128;
    for (int h = 0; h < 2; ++h) {
        const uint8_t *Lh = L + 128 * h;
        for (int l = 0; l < 32; ++l) {
            int q1 = Lh[l] & 15, q2 = Lh[l + 32] & 15, q3 = Lh[l + 64] & 15, q4 = Lh[l + 96] & 15;
            ql[64 * h + l] = (uint8_t)(q1 | (q3 << 4));
            ql[64 * h + l + 32] = (uint8_t)(q2 | (q4 << 4));
            qh[32 * h + l] = (uint8_t)((Lh[l] >> 4) | ((Lh[l + 32] >> 4) << 2) |
                                       ((Lh[l + 64] >> 4) << 4) | ((Lh[l + 96] >> 4) << 6));
        }
    }
    memcpy(y + 208, &dh, 2);
}

/* Q6_K rows (cols a multiple of 256) of a matrix: bfloat16 (bf16) or
 * float32. */
void rc_kq_to_q6_k(const void *src, int bf16, int64_t rows, int cols, uint8_t *dst)
{
    #pragma omp parallel
    {
        float *tmp = (float *)malloc((size_t)cols * 4);
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            const float *x;
            if (bf16) {
                const uint16_t *h = (const uint16_t *)src + (size_t)r * cols;
                for (int c = 0; c < cols; ++c) {
                    tmp[c] = kq_bf16(h[c]);
                }
                x = tmp;
            } else {
                x = (const float *)src + (size_t)r * cols;
            }
            uint8_t *o = dst + (size_t)r * kq_row_bytes(KQ_Q6_K, cols);
            for (int b = 0; b < cols / 256; ++b) {
                kq_block_to_q6_k(x + 256 * b, o + 210 * b);
            }
        }
        free(tmp);
    }
}

/* make_qkx2_quants of ggml: the codes 0..nmax (L) of n values with a scale
 * and a min (*the_min, the negated offset), the weighted squared error
 * least of nstep + 1 trials of the scale around nmax / (max - min). */
static float kq_qkx2_quants(int n, int nmax, const float *x, const float *weights, uint8_t *L,
                            float *the_min, uint8_t *Laux, float rmin, float rdelta, int nstep)
{
    float min = x[0], max = x[0], sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        min = x[i] < min ? x[i] : min;
        max = x[i] > max ? x[i] : max;
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max == min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min), scale = 1 / iscale, best = 0;
    for (int i = 0; i < n; ++i) {
        int l = kq_lrint(iscale * (x[i] - min));
        L[i] = (uint8_t)(l < 0 ? 0 : (l > nmax ? nmax : l));
        float diff = scale * L[i] + min - x[i];
        best += weights[i] * diff * diff;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = kq_lrint(iscale * (x[i] - min));
            l = l < 0 ? 0 : (l > nmax ? nmax : l);
            Laux[i] = (uint8_t)l;
            float w = weights[i];
            sum_l += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float err = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                err += weights[i] * diff * diff;
            }
            if (err < best) {
                memcpy(L, Laux, (size_t)n);
                best = err;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}

#if defined(__AVX512F__)
/* kq_qkx2_quants of 32 values with unit weights (the better encoder), the
 * trials of the scale in the lanes: lane k sums its values in the order of
 * the scalar loop, with the same float steps, so each trial gives the same
 * sums and errors, and the trials are taken in the same order. */
#ifndef KQ_XL_FMA
#define KQ_XL_FMA 1
#endif
static float kq_qkx2_unit(int nmax, const float *x, uint8_t *L, float *the_min, float rmin, float rdelta,
                          int nstep)
{
    const int n = 32;
    float weights[32];
    for (int i = 0; i < n; ++i) {
        weights[i] = 1.f;
    }
    float min = x[0], max = x[0], sum_w = weights[0], sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        min = x[i] < min ? x[i] : min;
        max = x[i] > max ? x[i] : max;
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if (min > 0) {
        min = 0;
    }
    if (max == min) {
        memset(L, 0, (size_t)n);
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax / (max - min), scale = 1 / iscale, best = 0;
    for (int i = 0; i < n; ++i) {
        int l = kq_lrint(iscale * (x[i] - min));
        L[i] = (uint8_t)(l < 0 ? 0 : (l > nmax ? nmax : l));
        float diff = scale * L[i] + min - x[i];
        best += weights[i] * diff * diff;
    }
    /* The trials from start on, with the min of now; at the first that wins,
     * the min changes (as in the scalar loop, where the later trials use
     * the new min and max - min), so the trials after it run again. */
    const int ns = nstep + 1;
    const __m512i top = _mm512_set1_epi32(nmax), zero = _mm512_setzero_si512();
    float isc[32], sl[32], sl2[32], sxl[32], ts[32], tm[32], er[32];
    int ok[32];
    for (int start = 0; start < ns;) {
        for (int is = 0; is < 32; ++is) {
            isc[is] = is >= start && is < ns ? (rmin + rdelta * is + nmax) / (max - min) : 0.f;
        }
        const int v0 = start / 16, v1 = (ns + 15) / 16;
        for (int v = v0; v < v1; ++v) {
            const __m512 vi = _mm512_loadu_ps(isc + 16 * v);
            __m512 a_l = _mm512_setzero_ps(), a_l2 = _mm512_setzero_ps(), a_xl = _mm512_setzero_ps();
            for (int i = 0; i < n; ++i) {
                const __m512 xm = _mm512_set1_ps(x[i] - min), xi = _mm512_set1_ps(x[i]);
                __m512i l = _mm512_cvtps_epi32(_mm512_mul_ps(vi, xm));
                __m512 lf = _mm512_cvtepi32_ps(_mm512_min_epi32(_mm512_max_epi32(l, zero), top));
                a_l = _mm512_add_ps(a_l, lf);
                a_l2 = _mm512_fmadd_ps(lf, lf, a_l2);
#if KQ_XL_FMA
                a_xl = _mm512_fmadd_ps(lf, xi, a_xl);
#else
                a_xl = _mm512_add_ps(a_xl, _mm512_mul_ps(lf, xi));
#endif
            }
            _mm512_storeu_ps(sl + 16 * v, a_l);
            _mm512_storeu_ps(sl2 + 16 * v, a_l2);
            _mm512_storeu_ps(sxl + 16 * v, a_xl);
        }
        for (int is = 16 * v0; is < 16 * v1; ++is) {
            ok[is] = 0;
            ts[is] = tm[is] = 0.f;
            if (is < start || is >= ns) {
                continue;
            }
            float sum_l = sl[is], sum_l2 = sl2[is], sum_xl = sxl[is];
            float D = sum_w * sum_l2 - sum_l * sum_l;
            if (D > 0) {
                float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
                float this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D;
                if (this_min > 0) {
                    this_min = 0;
                    this_scale = sum_xl / sum_l2;
                }
                ok[is] = 1;
                ts[is] = this_scale;
                tm[is] = this_min;
            }
        }
        for (int v = v0; v < v1; ++v) {
            const __m512 vi = _mm512_loadu_ps(isc + 16 * v), vs = _mm512_loadu_ps(ts + 16 * v),
                         vm = _mm512_loadu_ps(tm + 16 * v);
            __m512 err = _mm512_setzero_ps();
            for (int i = 0; i < n; ++i) {
                const __m512 xm = _mm512_set1_ps(x[i] - min), xi = _mm512_set1_ps(x[i]);
                __m512i l = _mm512_cvtps_epi32(_mm512_mul_ps(vi, xm));
                __m512 lf = _mm512_cvtepi32_ps(_mm512_min_epi32(_mm512_max_epi32(l, zero), top));
                __m512 diff = _mm512_sub_ps(_mm512_fmadd_ps(vs, lf, vm), xi);
                err = _mm512_fmadd_ps(diff, diff, err);
            }
            _mm512_storeu_ps(er + 16 * v, err);
        }
        int win = -1;
        for (int is = start; is < ns; ++is) {
            if (ok[is] && er[is] < best) {
                win = is;
                break;
            }
        }
        if (win < 0) {
            break;
        }
        for (int i = 0; i < n; ++i) {            /* the codes of the winner, with the min it ran with */
            int l = kq_lrint(isc[win] * (x[i] - min));
            L[i] = (uint8_t)(l < 0 ? 0 : (l > nmax ? nmax : l));
        }
        best = er[win];
        scale = ts[win];
        min = tm[win];
        start = win + 1;
    }
    *the_min = -min;
    return scale;
}
#endif

/* get_scale_min_k4 of ggml: the 6-bit scale and min of sub-block j. */
static void kq_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m)
{
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        *m = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

/* One Q4_K (bits 4: 144 bytes) or Q5_K (bits 5: 176 bytes) block of 256
 * values (quantize_row_q4_K_ref, quantize_row_q5_K_ref of ggml): d, dmin,
 * the 12 bytes of the 6-bit scales and mins of the 8 sub-blocks of 32, (Q5_K:
 * the high bits, 32 bytes), the low 4 bits of the codes. */
static void kq_block_to_qk(const float *x, uint8_t *y, int bits)
{
    const int nmax = bits == 4 ? 15 : 31;
    uint8_t L[256], Laux[32], scales_q[12];
    float weights[32], mins[8], scales[8], max_scale = 0.f, max_min = 0.f;
    for (int j = 0; j < 8; ++j) {
        float sum_x2 = 0.f;
        for (int l = 0; l < 32; ++l) {
            sum_x2 += x[32 * j + l] * x[32 * j + l];
        }
        float av_x = sqrtf(sum_x2 / 32);
        for (int l = 0; l < 32; ++l) {
            weights[l] = av_x + fabsf(x[32 * j + l]);
        }
        scales[j] = bits == 4
            ? kq_qkx2_quants(32, nmax, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20)
            : kq_qkx2_quants(32, nmax, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.5f, 0.1f, 15);
        max_scale = scales[j] > max_scale ? scales[j] : max_scale;
        max_min = mins[j] > max_min ? mins[j] : max_min;
    }
    float inv_scale = max_scale > 0 ? 63.f / max_scale : 0.f;
    float inv_min = max_min > 0 ? 63.f / max_min : 0.f;
    memset(scales_q, 0, 12);
    for (int j = 0; j < 8; ++j) {
        int ls = kq_lrint(inv_scale * scales[j]), lm = kq_lrint(inv_min * mins[j]);
        ls = ls > 63 ? 63 : ls;
        lm = lm > 63 ? 63 : lm;
        if (j < 4) {
            scales_q[j] = (uint8_t)ls;
            scales_q[j + 4] = (uint8_t)lm;
        } else {
            scales_q[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            scales_q[j - 4] |= (uint8_t)((ls >> 4) << 6);
            scales_q[j] |= (uint8_t)((lm >> 4) << 6);
        }
    }
    uint16_t dh = kq_f32_to_f16(max_scale / 63.f), mh = kq_f32_to_f16(max_min / 63.f);
    float d0 = kq_h((const uint8_t *)&dh), m0 = kq_h((const uint8_t *)&mh);
    for (int j = 0; j < 8; ++j) {
        uint8_t sc, m;
        kq_scale_min_k4(j, scales_q, &sc, &m);
        const float d = d0 * sc;
        if (d == 0.f) {
            continue;
        }
        const float dm = m0 * m;
        for (int ii = 0; ii < 32; ++ii) {
            int l = kq_lrint((x[32 * j + ii] + dm) / d);
            L[32 * j + ii] = (uint8_t)(l < 0 ? 0 : (l > nmax ? nmax : l));
        }
    }
    memcpy(y, &dh, 2);
    memcpy(y + 2, &mh, 2);
    memcpy(y + 4, scales_q, 12);
    if (bits == 4) {
        uint8_t *q = y + 16;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
            }
            q += 32;
        }
        return;
    }
    uint8_t *qh = y + 16, *ql = y + 48;
    memset(qh, 0, 32);
    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < 256; n += 64) {
        for (int j = 0; j < 32; ++j) {
            int l1 = L[n + j], l2 = L[n + j + 32];
            if (l1 > 15) {
                l1 -= 16;
                qh[j] |= m1;
            }
            if (l2 > 15) {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[j] = (uint8_t)(l1 | (l2 << 4));
        }
        m1 = (uint8_t)(m1 << 2);
        m2 = (uint8_t)(m2 << 2);
        ql += 32;
    }
}

/* Q4_K (bits 4) or Q5_K (bits 5) rows (cols a multiple of 256) of a matrix:
 * bfloat16 (bf16) or float32. */
void rc_kq_to_qk(const void *src, int bf16, int64_t rows, int cols, int bits, uint8_t *dst)
{
    const int bb = bits == 4 ? 144 : 176;
    #pragma omp parallel
    {
        float *tmp = (float *)malloc((size_t)cols * 4);
        #pragma omp for schedule(static)
        for (int64_t r = 0; r < rows; ++r) {
            const float *x;
            if (bf16) {
                const uint16_t *h = (const uint16_t *)src + (size_t)r * cols;
                for (int c = 0; c < cols; ++c) {
                    tmp[c] = kq_bf16(h[c]);
                }
                x = tmp;
            } else {
                x = (const float *)src + (size_t)r * cols;
            }
            uint8_t *o = dst + (size_t)r * (cols / 256) * bb;
            for (int b = 0; b < cols / 256; ++b) {
                kq_block_to_qk(x + 256 * b, o + (size_t)bb * b, bits);
            }
        }
        free(tmp);
    }
}

/* ---- Q4_K / Q5_K with a better encoder (rc_kq_to_qk_opt) ----
 *
 * The same blocks as kq_block_to_qk (any reader takes them), from a better
 * search (the RQ4 study, work/rq4best4: -0.21 dB on the weights of LFM2.5,
 * -0.18 at the outputs, against the ggml encoder):
 * - the scale and min of each sub-block from kq_qkx2_quants with unit
 *   weights (the squared error itself, not ggml's av_x + |x|);
 * - d and dmin among f16(base (1 + 0.01 k)), |k| <= 3, each pair with the
 *   6-bit scales and mins rounded through 1 / d, 1 / dmin;
 * - each sub-block also tries the 6-bit neighbours (+-1) of its scale and
 *   min; the least squared error wins.
 * dsel (or null): one value for each super-block; its low dbits go to the
 * low mantissa bits of d and the next mbits to those of dmin (each rounded
 * up to them): a selector that costs no byte (the free rotation choice of
 * the study: 4 rotations a super-block). err (or null) gets the squared
 * error of each super-block. */

/* the squared error of a sub-block of 32 with step d0 sc and offset m0 m;
 * out (or null) gets the values */
static double kq_qk_sub_err(const float *x, float d0, float m0, int sc, int m, int nmax, float *out)
{
    const float d = d0 * sc, dm = m0 * m;
    double e = 0;
    for (int i = 0; i < 32; ++i) {
        int l = 0;
        if (d != 0.f) {
            l = kq_lrint((x[i] + dm) / d);
            l = l < 0 ? 0 : (l > nmax ? nmax : l);
        }
        float v = d * (float)l - dm;
        if (out) {
            out[i] = v;
        }
        double df = (double)v - x[i];
        e += df * df;
    }
    return e;
}

static int kq_q6_round(float v)
{
    int r = kq_lrint(v);
    return r > 63 ? 63 : (r < 0 ? 0 : r);
}

/* f16 h with its low bits set to sel, not smaller in magnitude */
static uint16_t kq_f16_up(uint16_t h, int sel, int bits)
{
    if (bits == 0) {
        return h;
    }
    uint32_t sign = h & 0x8000, mag = h & 0x7fff, mk = (1u << bits) - 1;
    uint32_t m2 = (mag & ~mk) | (uint32_t)sel;
    if (m2 < mag) {
        m2 += 1u << bits;
    }
    return (uint16_t)(sign | m2);
}

#if defined(__AVX512F__)
/* the values d l - dm of 16 values t (already x + dm) for the step d, l =
 * lrintf(t / d) clamped to 0..nmax, as kq_qk_sub_err (the rounding and the
 * clamp in float give the codes of the conversion: |t / d| stays far below
 * 2^31) */
static inline __m512 kq_vals16(__m512 t, __m512 vd, __m512 vtop, __m512 vdm)
{
    __m512 l = _mm512_roundscale_ps(_mm512_div_ps(t, vd), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    l = _mm512_min_ps(_mm512_max_ps(l, _mm512_setzero_ps()), vtop);
    return _mm512_sub_ps(_mm512_mul_ps(vd, l), vdm);
}

/* the squared errors of a sub-block of 32 for the 9 pairs (ls + a, lm + b),
 * a, b in -1..1 (e[3 (a + 1) + (b + 1)]), as kq_qk_sub_err: the same codes
 * and values. A step d of 0 (ls + a = 0) gives the codes 0 as there. The
 * squares of the float differences are exact in double, and so are their
 * sums here, so the order of the sums does not change the errors. */
static inline void kq_qk_sub_err9(const float *x, float d0, float m0, int ls, int lm, int nmax, double *e)
{
    const __m512 xa = _mm512_loadu_ps(x), xb = _mm512_loadu_ps(x + 16);
    const __m512 vtop = _mm512_set1_ps((float)nmax);
    __m512d xd[4];
    xd[0] = _mm512_cvtps_pd(_mm512_castps512_ps256(xa));
    xd[1] = _mm512_cvtps_pd(_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(xa), 1)));
    xd[2] = _mm512_cvtps_pd(_mm512_castps512_ps256(xb));
    xd[3] = _mm512_cvtps_pd(_mm256_castsi256_ps(_mm512_extracti64x4_epi64(_mm512_castps_si512(xb), 1)));
    __m256d s4[9];
    for (int a = 0; a < 3; ++a) {
        const float d = d0 * (ls + a - 1);
        const __m512 vd = _mm512_set1_ps(d);
        for (int b = 0; b < 3; ++b) {
            const __m512 vdm = _mm512_set1_ps(m0 * (lm + b - 1));
            __m512 va, vb;
            if (d != 0.f) {
                va = kq_vals16(_mm512_add_ps(xa, vdm), vd, vtop, vdm);
                vb = kq_vals16(_mm512_add_ps(xb, vdm), vd, vtop, vdm);
            } else {
                va = vb = _mm512_sub_ps(_mm512_setzero_ps(), vdm);
            }
            __m512i ia = _mm512_castps_si512(va), ib = _mm512_castps_si512(vb);
            __m512d f0 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(_mm512_castsi512_si256(ia))), xd[0]);
            __m512d f1 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(_mm512_extracti64x4_epi64(ia, 1))), xd[1]);
            __m512d f2 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(_mm512_castsi512_si256(ib))), xd[2]);
            __m512d f3 = _mm512_sub_pd(_mm512_cvtps_pd(_mm256_castsi256_ps(_mm512_extracti64x4_epi64(ib, 1))), xd[3]);
            __m512d acc = _mm512_mul_pd(f0, f0);
            acc = _mm512_fmadd_pd(f1, f1, acc);
            acc = _mm512_fmadd_pd(f2, f2, acc);
            acc = _mm512_fmadd_pd(f3, f3, acc);
            s4[3 * a + b] = _mm256_add_pd(_mm512_castpd512_pd256(acc), _mm512_extractf64x4_pd(acc, 1));
        }
    }
    for (int k = 0; k < 8; k += 4) {         /* 4 sums at a time: [e k, e k+1, e k+2, e k+3] */
        __m256d h01 = _mm256_hadd_pd(s4[k], s4[k + 1]), h23 = _mm256_hadd_pd(s4[k + 2], s4[k + 3]);
        __m256d r4 = _mm256_add_pd(_mm256_permute2f128_pd(h01, h23, 0x20), _mm256_permute2f128_pd(h01, h23, 0x31));
        _mm256_storeu_pd(e + k, r4);
    }
    __m128d h = _mm_add_pd(_mm256_castpd256_pd128(s4[8]), _mm256_extractf128_pd(s4[8], 1));
    e[8] = _mm_cvtsd_f64(_mm_add_sd(h, _mm_unpackhi_pd(h, h)));
}
#endif

/* the 6-bit scales and mins of the 8 sub-blocks for d0, m0, each the best of
 * its rounding and the 8 neighbours; the total squared error */
static double kq_qk_choose(const float *x, const float *scales, const float *mins, float d0, float m0,
                           int nmax, int *LS, int *LM)
{
    float is = d0 > 0 ? 1.f / d0 : 0.f, im = m0 > 0 ? 1.f / m0 : 0.f;
    double tot = 0;
    for (int j = 0; j < 8; ++j) {
        int ls = kq_q6_round(is * scales[j]), lm = kq_q6_round(im * mins[j]), l0 = ls, m0i = lm;
#if defined(__AVX512F__)
        double e9[9];
        kq_qk_sub_err9(x + 32 * j, d0, m0, l0, m0i, nmax, e9);
        double best = e9[4];
#else
        double best = kq_qk_sub_err(x + 32 * j, d0, m0, ls, lm, nmax, NULL);
#endif
        for (int a = -1; a <= 1; ++a) {
            for (int b = -1; b <= 1; ++b) {
                int s2 = l0 + a, m2 = m0i + b;
                if ((a == 0 && b == 0) || s2 < 0 || s2 > 63 || m2 < 0 || m2 > 63) {
                    continue;
                }
#if defined(__AVX512F__)
                double e = e9[3 * (a + 1) + (b + 1)];
#else
                double e = kq_qk_sub_err(x + 32 * j, d0, m0, s2, m2, nmax, NULL);
#endif
                if (e < best) {
                    best = e;
                    ls = s2;
                    lm = m2;
                }
            }
        }
        LS[j] = ls;
        LM[j] = lm;
        tot += best;
    }
    return tot;
}

static double kq_block_to_qk_opt(const float *x, uint8_t *y, int bits, int sel, int dbits, int mbits)
{
    const int nmax = bits == 4 ? 15 : 31;
    uint8_t L[256], Laux[32];
    float weights[32], mins[8], scales[8], max_scale = 0.f, max_min = 0.f;
    for (int l = 0; l < 32; ++l) {
        weights[l] = 1.f;
    }
    for (int j = 0; j < 8; ++j) {
#if defined(__AVX512F__)
        (void)weights;
        (void)Laux;
        scales[j] = bits == 4 ? kq_qkx2_unit(nmax, x + 32 * j, L + 32 * j, &mins[j], -1.f, 0.1f, 20)
                              : kq_qkx2_unit(nmax, x + 32 * j, L + 32 * j, &mins[j], -0.5f, 0.1f, 15);
#else
        scales[j] = bits == 4
            ? kq_qkx2_quants(32, nmax, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -1.f, 0.1f, 20)
            : kq_qkx2_quants(32, nmax, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.5f, 0.1f, 15);
#endif
        max_scale = scales[j] > max_scale ? scales[j] : max_scale;
        max_min = mins[j] > max_min ? mins[j] : max_min;
    }
    int LS[8], LM[8], bLS[8], bLM[8];
    double best = 1e300;
    uint16_t bdh = 0, bmh = 0;
    for (int ka = -3; ka <= 3; ++ka) {
        for (int kb = -3; kb <= 3; ++kb) {
            uint16_t d2 = kq_f32_to_f16(max_scale / 63.f * (1.f + 0.01f * ka));
            uint16_t m2 = kq_f32_to_f16(max_min / 63.f * (1.f + 0.01f * kb));
            if (sel >= 0) {
                d2 = kq_f16_up(d2, sel & ((1 << dbits) - 1), dbits);
                m2 = kq_f16_up(m2, (sel >> dbits) & ((1 << mbits) - 1), mbits);
            }
            double e = kq_qk_choose(x, scales, mins, kq_h((const uint8_t *)&d2), kq_h((const uint8_t *)&m2),
                                    nmax, LS, LM);
            if (e < best) {
                best = e;
                bdh = d2;
                bmh = m2;
                memcpy(bLS, LS, sizeof LS);
                memcpy(bLM, LM, sizeof LM);
            }
        }
    }
    float d0 = kq_h((const uint8_t *)&bdh), m0 = kq_h((const uint8_t *)&bmh);
    uint8_t sq[12];
    memset(sq, 0, 12);
    for (int j = 0; j < 8; ++j) {
        int ls = bLS[j], lm = bLM[j];
        if (j < 4) {
            sq[j] = (uint8_t)ls;
            sq[j + 4] = (uint8_t)lm;
        } else {
            sq[j + 4] = (uint8_t)((ls & 0xF) | ((lm & 0xF) << 4));
            sq[j - 4] |= (uint8_t)((ls >> 4) << 6);
            sq[j] |= (uint8_t)((lm >> 4) << 6);
        }
        const float d = d0 * ls, dm = m0 * lm;
        if (d == 0.f) {
            continue;       /* (the codes of kq_qkx2_quants stay) */
        }
        for (int i = 0; i < 32; ++i) {
            int l = kq_lrint((x[32 * j + i] + dm) / d);
            L[32 * j + i] = (uint8_t)(l < 0 ? 0 : (l > nmax ? nmax : l));
        }
    }
    memcpy(y, &bdh, 2);
    memcpy(y + 2, &bmh, 2);
    memcpy(y + 4, sq, 12);
    if (bits == 4) {
        uint8_t *q = y + 16;
        for (int j = 0; j < 256; j += 64) {
            for (int l = 0; l < 32; ++l) {
                q[l] = (uint8_t)(L[j + l] | (L[j + l + 32] << 4));
            }
            q += 32;
        }
    } else {
        uint8_t *qh = y + 16, *ql = y + 48;
        memset(qh, 0, 32);
        uint8_t m1 = 1, m2 = 2;
        for (int n = 0; n < 256; n += 64) {
            for (int j = 0; j < 32; ++j) {
                int l1 = L[n + j], l2 = L[n + j + 32];
                if (l1 > 15) {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                if (l2 > 15) {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[j] = (uint8_t)(l1 | (l2 << 4));
            }
            m1 = (uint8_t)(m1 << 2);
            m2 = (uint8_t)(m2 << 2);
            ql += 32;
        }
    }
    return best;
}

/* Q4_K / Q5_K rows (float32, cols a multiple of 256) with the better encoder;
 * dsel (rows x cols / 256, or null), dbits, mbits: the selector of each
 * super-block in the low bits of d / dmin; err (or null): the squared error
 * of each super-block. */
void rc_kq_to_qk_opt(const float *src, int64_t rows, int cols, int bits, const int8_t *dsel, int dbits,
                     int mbits, uint8_t *dst, double *err)
{
    const int bb = bits == 4 ? 144 : 176, nb = cols / 256;
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < rows * nb; ++x) {
        double e = kq_block_to_qk_opt(src + (size_t)x * 256, dst + (size_t)x * bb, bits,
                                      dsel ? dsel[x] : -1, dbits, mbits);
        if (err) {
            err[x] = e;
        }
    }
}

/* The NVFP4 matrices of n experts (ModelOpt: w, rows x cols / 2 bytes, value
 * 2i in the low 4 bits of byte i; s, rows x cols / 16 E4M3 scales; g, the
 * float32 scale) to KQ_NV4 rows: dst gets n matrices of rows rows. wp and sp
 * hold the addresses of w and s of each expert. */
void rc_kq_nv4_pack(const int64_t *wp, const int64_t *sp, const float *g, int n, int rows, int cols,
                 uint8_t *dst)
{
    size_t rb = kq_row_bytes(KQ_NV4, cols);
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < (int64_t)n * rows; ++x) {
        int e = (int)(x / rows), r = (int)(x % rows);
        const uint8_t *w = (const uint8_t *)(intptr_t)wp[e] + (size_t)r * (cols / 2);
        const uint8_t *s = (const uint8_t *)(intptr_t)sp[e] + (size_t)r * (cols / 16);
        uint8_t *o = dst + (size_t)x * rb;
        memset(o, 0, rb);
        memcpy(o + (size_t)cols / 2, s, (size_t)cols / 16);       /* the scales, in order */
        memcpy(o + (size_t)cols / 2 + (size_t)cols / 16, &g[e], 4);
        for (int b = 0; b < cols / 32; ++b) {
            const uint8_t *src = w + (size_t)b * 16;     /* 32 values, 2 in each byte */
            uint8_t *q = o + (size_t)16 * b;
            for (int j = 0; j < 16; ++j) {
                int lo = (src[j / 2] >> (4 * (j % 2))) & 15;             /* value j */
                int hi = (src[8 + j / 2] >> (4 * (j % 2))) & 15;         /* value j + 16 */
                q[j] = (uint8_t)(lo | (hi << 4));
            }
        }
    }
}

/* The NVFP4 matrices of n experts (as rc_kq_nv4_pack; rows a multiple of 16)
 * to KQ_NVX groups: dst gets n matrices of rows / 16 groups. */
void rc_kq_nvx_pack(const int64_t *wp, const int64_t *sp, const float *g, int n, int rows, int cols,
                 uint8_t *dst)
{
    size_t gb = 16 * kq_row_bytes(KQ_NVX, cols);
    int ng = rows / 16;
    #pragma omp parallel for schedule(static)
    for (int64_t x = 0; x < (int64_t)n * ng; ++x) {
        int e = (int)(x / ng), g0 = 16 * (int)(x % ng);
        const uint8_t *w = (const uint8_t *)(intptr_t)wp[e];
        const uint8_t *s = (const uint8_t *)(intptr_t)sp[e];
        uint8_t *o = dst + (size_t)x * gb;
        memset(o, 0, gb);
        memcpy(o, &g[e], 4);
        for (int b = 0; b < cols / 32; ++b) {
            uint8_t *blk = o + 16 + (size_t)b * KQ_NVX_BB;
            for (int r = 0; r < 16; ++r) {
                const uint8_t *wr = w + (size_t)(g0 + r) * (cols / 2) + 16 * b;
                for (int v = 0; v < 32; ++v) {
                    int c = (wr[v / 2] >> (4 * (v % 2))) & 15;      /* value 2i: the low 4 bits */
                    blk[32 * (v / 4) + 4 * (r % 8) + v % 4] |= (uint8_t)(c << (r < 8 ? 0 : 4));
                }
                const uint8_t *sr = s + (size_t)(g0 + r) * (cols / 16) + 2 * b;
                blk[256 + r] = sr[0];
                blk[272 + r] = sr[1];
            }
        }
    }
}

/* The E4M3 block scales of an NVFP4 matrix (rows x c scales) from the CSF
 * encoding "csf" of Qwen3.8-Flash-Next-FP4-Local (byte-window4-fixed-stream-
 * u24-exceptions/1; flashinfer scripts/convert_csf_blocks.py _logical_source):
 * fixed holds rows / 16 slabs of 16 row bases, then the 4-bit codes of the 16
 * rows (c / 2 bytes a row, the low nibble the even column); a scale is its
 * row's base plus its code. Then each exception word w puts the byte w >> 24
 * at the row-major position w & 0xFFFFFF. stats (6 int64): [0] bases above
 * 240, [1] exceptions out of order or out of range, [2] exceptions inside
 * their row's window (redundant), [3] the largest base, [4] zero scales,
 * [5] E4M3 subnormal scales (codes 1-7) after the exceptions. */
void rc_csf_decode(const uint8_t *fixed, int rows, int c, const uint32_t *exc, int64_t n, uint8_t *out,
                   int64_t *stats)
{
    int rb = c / 2, sb = 16 * (1 + rb);
    int64_t bad_base = 0, bad_exc = 0, inside = 0, maxb = 0, zeros = 0, sub = 0;
    for (int r = 0; r < rows; ++r) {
        const uint8_t *slab = fixed + (size_t)(r / 16) * sb;
        int base = slab[r % 16];
        const uint8_t *codes = slab + 16 + (size_t)(r % 16) * rb;
        bad_base += base > 240;
        maxb = base > maxb ? base : maxb;
        uint8_t *o = out + (size_t)r * c;
        for (int j = 0; j < c; ++j) {
            o[j] = (uint8_t)(base + ((codes[j >> 1] >> (4 * (j & 1))) & 15));
        }
    }
    int64_t prev = -1, total = (int64_t)rows * c;
    for (int64_t i = 0; i < n; ++i) {
        int64_t pos = exc[i] & 0xFFFFFF;
        uint8_t v = (uint8_t)(exc[i] >> 24);
        if (pos <= prev || pos >= total) {
            ++bad_exc;
            continue;
        }
        prev = pos;
        int base = fixed[(size_t)(pos / c / 16) * sb + (pos / c) % 16];
        inside += v >= base && v <= base + 15;
        out[pos] = v;
    }
    for (int64_t i = 0; i < total; ++i) {
        zeros += (out[i] & 0x7F) == 0;
        sub += (out[i] & 0x78) == 0 && (out[i] & 7) != 0;
    }
    stats[0] = bad_base, stats[1] = bad_exc, stats[2] = inside, stats[3] = maxb, stats[4] = zeros, stats[5] = sub;
}

/* Copy n rows of bytes bytes from the addresses addrs (rows of a memory map
 * of a file: the n-gram table of the safetensors checkpoint). Many threads
 * take the page faults at the same time, so the disk has many reads in flight
 * (a loop of one thread waits for each read). */
/* Copy n bytes from src to dst on all the threads, in chunks of 4 MB (the
 * experts from the map of a file into memory of the process: a file on a
 * DAX mount, Optane). The pages of dst go where its policy says (mbind). */
void rc_kq_memcpy_par(uint8_t *dst, const uint8_t *src, int64_t n)
{
    const int64_t chunk = 4 << 20;
    int64_t nc = (n + chunk - 1) / chunk;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t c = 0; c < nc; ++c) {
        int64_t off = c * chunk, len = n - off < chunk ? n - off : chunk;
        memcpy(dst + off, src + off, (size_t)len);
    }
}

void rc_kq_gather(const int64_t *addrs, int64_t n, int bytes, uint8_t *out)
{
    /* On the calling thread: a team of 64 threads for the rows of a verify
     * group (16 for each token) spun on the cores of the CPU experts of a GPU
     * step (Qwen3.8, MTP drafts 30 ms a round in place of 22), and one for
     * the 480 rows of a short prompt (0.04 ms alone) is no team of the plan
     * (rc_team_warn). The 65536 rows of a mixed group of 4096 tokens from
     * the DAX module: 25 ms on one thread (9 with 64), 0.4% of the group. */
    for (int64_t i = 0; i < n; ++i) {
        memcpy(out + (size_t)i * bytes, (const void *)(intptr_t)addrs[i], (size_t)bytes);
    }
}

/* float32 or bfloat16 rows (bf) to KQ_F32X16 or KQ_BF16X16: dst gets
 * (rows + 15) / 16 groups; the rows past rows are zeros. */
void rc_kq_pack_x16f(const uint8_t *src, int bf, int64_t rows, int cols, uint8_t *dst)
{
    int es = bf ? 2 : 4;
    int64_t ng = (rows + 15) / 16;
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < ng; ++g) {
        uint8_t *o = dst + (size_t)g * 16 * cols * es;
        for (int r = 0; r < 16; ++r) {
            int64_t row = 16 * g + r;
            for (int c = 0; c < cols; ++c) {
                if (row < rows) {
                    memcpy(o + ((size_t)16 * c + r) * es, src + ((size_t)row * cols + c) * es, es);
                } else {
                    memset(o + ((size_t)16 * c + r) * es, 0, es);
                }
            }
        }
    }
}

/* BF12 rows to KQ_BF12X16 (kq_x16f_col): groups of 16 rows, the last one
 * padded with rows of zeros. */
void rc_kq_pack_bf12x16(const uint8_t *src, int64_t rows, int cols, uint8_t *dst)
{
    size_t rb = kq_row_bytes(KQ_BF12, cols), gbytes = (size_t)16 * kq_row_bytes(KQ_BF12X16, cols);
    int64_t ng = (rows + 15) / 16;
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < ng; ++g) {
        uint8_t *o = dst + (size_t)g * gbytes;
        memset(o, 0, gbytes);
        for (int r = 0; r < 16; ++r) {
            int64_t row = 16 * g + r;
            if (row >= rows) {
                continue;
            }
            const uint8_t *w = src + (size_t)row * rb;
            for (int b = 0; b < cols / 32; ++b) {
                uint8_t *blk = o + (size_t)b * 784;
                for (int cc = 0; cc < 32; ++cc) {
                    blk[16 * cc + r] = w[32 * b + cc];
                }
                for (int j = 0; j < 16; ++j) {
                    blk[512 + 16 * j + r] = w[cols + 16 * b + j];
                }
                blk[768 + r] = w[cols + cols / 2 + b];
            }
        }
    }
}

/* Q8_0 rows (rows a multiple of 16) to KQ_Q8X16 (see the formats). */
void rc_kq_pack_q8x16(const uint8_t *q8, int64_t rows, int cols, uint8_t *dst)
{
    int nb = cols / 32;
    size_t rb = (size_t)nb * 34;
    #pragma omp parallel for schedule(static)
    for (int64_t g = 0; g < rows / 16; ++g) {
        for (int b = 0; b < nb; ++b) {
            uint8_t *o = dst + ((size_t)g * nb + b) * 576;
            for (int r = 0; r < 16; ++r) {
                const uint8_t *blk = q8 + (size_t)(16 * g + r) * rb + (size_t)b * 34;
                memcpy(o + 2 * r, blk, 2);
                int16_t sum = 0;
                for (int k = 0; k < 8; ++k) {
                    for (int u = 0; u < 4; ++u) {
                        o[64 + 64 * k + 4 * r + u] = blk[2 + 4 * k + u];
                        sum += (int8_t)blk[2 + 4 * k + u];
                    }
                }
                memcpy(o + 32 + 2 * r, &sum, 2);
            }
        }
    }
}

/* The float values of rows ids of a matrix (the embeddings). */
void rc_kq_rows(const uint8_t *w, int type, int cols, const int64_t *ids, int n, float *out)
{
    size_t rb = kq_row_bytes(type, cols);
    int bv = kq_block32(type) ? 32 : 256;
    /* on the calling thread, as rc_kq_gather: no team outside the planned ones
     * (rc_team_warn). A team of 40 for the 64 n-gram rows of a verify
     * group waited 5.7 ms for threads that shared the cores of the CPU
     * experts (37 us alone); the embeddings of a mixed group of 4096 rows
     * take 22 ms on one thread (5 with 40), 0.4% of the group. */
    for (int i = 0; i < n; ++i) {
        const uint8_t *row = w + (size_t)ids[i] * rb;
        if (type == KQ_NVX) {
            kq_nvx_values(w + (size_t)(ids[i] & ~(int64_t)15) * rb, (int)(ids[i] & 15), cols,
                          out + (size_t)i * cols);
            continue;
        }
        if (type == KQ_Q4X) {
            const uint8_t *wg = w + (size_t)(ids[i] & ~(int64_t)15) * rb;
            int rin = (int)(ids[i] & 15);
            for (int b = 0; b < cols / 32; ++b) {
                const uint8_t *blk = wg + (size_t)b * KQ_Q4X_BB;
                float d = kq_h(blk + 256 + 2 * rin);
                for (int v = 0; v < 32; ++v) {
                    int c = (blk[32 * (v / 4) + 4 * (rin % 8) + v % 4] >> (rin < 8 ? 0 : 4)) & 15;
                    out[(size_t)i * cols + 32 * b + v] = d * (float)(c - 8);
                }
            }
            continue;
        }
        if (type == KQ_F32) {
            memcpy(out + (size_t)i * cols, row, (size_t)cols * 4);
            continue;
        }
        if (type == KQ_BF12) {
            for (int j = 0; j < cols; ++j) {
                uint32_t u = (uint32_t)kq_bf12_bits(row, cols, j) << 16;
                memcpy(out + (size_t)i * cols + j, &u, 4);
            }
            continue;
        }
        for (int b = 0; b < cols / bv; ++b) {
            kq_block_values(type, row, cols, b, out + (size_t)i * cols + (size_t)b * bv);
        }
    }
}

/* bfloat16 rows (uint16, rows x cols, cols a multiple of 32) to BF12 rows
 * (dst: rows x kq_row_bytes(KQ_BF12, cols)). A value more than 15 binades
 * under the largest of its group (and a zero, a denormal) gets the zero code;
 * so does -2^(E - 15) (a collision: the zero code takes its bits). rep gets
 * the report (BF12_PLAN.md step 1): [0] the values zeroed (nonzero values
 * that decode to 0), [1] the collisions, [2] the Inf and NaN values, [3] the
 * largest zeroed value in absolute value, [4] the sum of the squares of the
 * values, [5] the count of the worst kept (at most 10), then for each (by
 * size): [6 + 4 i] the value, its row, its column, the largest absolute
 * value of its group. */
static float kq_bf16f(uint16_t h)
{
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, 4);
    return f;
}

void rc_kq_bf16_to_bf12(const uint16_t *src, int64_t rows, int cols, uint8_t *dst, double *rep)
{
    size_t rb = kq_row_bytes(KQ_BF12, cols);
    int ng = cols / 32;
    double nz = 0, ncol = 0, nbad = 0, big = 0, ss = 0;
    int nt = omp_get_max_threads();
    double *worst = (double *)calloc((size_t)nt * 40, sizeof(double));
    int *nw = (int *)calloc((size_t)nt, sizeof(int));
    #pragma omp parallel for schedule(static) reduction(+ : nz, ncol, nbad, ss) reduction(max : big)
    for (int64_t r = 0; r < rows; ++r) {
        int th = omp_get_thread_num();
        const uint16_t *w = src + (size_t)r * cols;
        uint8_t *o = dst + (size_t)r * rb;
        memset(o, 0, rb);
        for (int g = 0; g < ng; ++g) {
            int E = 0;
            float gmax = 0.f;
            for (int j = 0; j < 32; ++j) {
                int e = (w[32 * g + j] >> 7) & 255;
                E = e > E ? e : E;
                float a = fabsf(kq_bf16f(w[32 * g + j]));
                if (a == a && a > gmax) {
                    gmax = a;
                }
            }
            o[cols + cols / 2 + g] = (uint8_t)E;
            for (int j = 0; j < 32; ++j) {
                uint16_t h = w[32 * g + j];
                int sgn = h >> 15, e = (h >> 7) & 255, m = h & 127;
                float f = kq_bf16f(h);
                if (e == 255) {
                    nbad += 1;
                } else {
                    ss += (double)f * f;
                }
                int gap = E - e;
                if (gap > 15) {
                    /* out of range (or a zero, a denormal): the zero code */
                    if (f != 0.f) {
                        nz += 1;
                        double a = fabs((double)f);
                        big = a > big ? a : big;
                        /* the worst 10 of this thread, by size */
                        double *ws = worst + (size_t)th * 40;
                        int k = nw[th];
                        if (k < 10 || a > fabs(ws[4 * (k - 1)])) {
                            int at = k < 10 ? k++ : 9;
                            while (at > 0 && fabs(ws[4 * (at - 1)]) < a) {
                                memcpy(ws + 4 * at, ws + 4 * (at - 1), 4 * sizeof(double));
                                --at;
                            }
                            ws[4 * at] = f;
                            ws[4 * at + 1] = (double)r;
                            ws[4 * at + 2] = 32.0 * g + j;
                            ws[4 * at + 3] = gmax;
                            nw[th] = k;
                        }
                    }
                    gap = 15;
                    sgn = 1;
                    m = 0;
                } else if (gap == 15 && sgn == 1 && m == 0) {
                    ncol += 1;
                }
                o[32 * g + j] = (uint8_t)(sgn << 7 | m);
                o[cols + 16 * g + (j & 15)] |= (uint8_t)(gap << (4 * (j >> 4)));
            }
        }
    }
    rep[0] = nz;
    rep[1] = ncol;
    rep[2] = nbad;
    rep[3] = big;
    rep[4] = ss;
    /* the worst 10 of all the threads, by size (a selection over at most
     * 10 per thread) */
    int k = 0;
    for (int pass = 0; pass < 10; ++pass) {
        int bt = -1, bi = -1;
        double ba = 0.0;            /* (a zeroed value is not 0) */
        for (int th = 0; th < nt; ++th) {
            for (int i = 0; i < nw[th]; ++i) {
                double a = fabs(worst[(size_t)th * 40 + 4 * i]);
                if (a > ba) {
                    ba = a;
                    bt = th;
                    bi = i;
                }
            }
        }
        if (bt < 0) {
            break;
        }
        double *x = worst + (size_t)bt * 40 + 4 * bi;
        memcpy(rep + 6 + 4 * k, x, 4 * sizeof(double));
        x[0] = 0.0;                 /* taken (abs 0 is below every zeroed value) */
        ++k;
    }
    rep[5] = k;
    free(worst);
    free(nw);
}

/* BF12 rows to their bfloat16 bits (dst: rows x cols uint16). */
void rc_kq_bf12_to_bf16(const uint8_t *src, int64_t rows, int cols, uint16_t *dst)
{
    size_t rb = kq_row_bytes(KQ_BF12, cols);
    #pragma omp parallel for schedule(static)
    for (int64_t r = 0; r < rows; ++r) {
        kq_bf12_row_bits(src + (size_t)r * rb, cols, dst + (size_t)r * cols);
    }
}
