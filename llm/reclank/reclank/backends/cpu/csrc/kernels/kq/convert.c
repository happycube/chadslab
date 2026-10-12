/* ---------- the conversions of the safetensors loader (reclank/formats/nvfp4_st.py) ---------- */

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
        int l = (int)lrintf(iscale * x[i]);
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
            int l = (int)lrintf(iscale * x[i]);
            l = l < -nmax ? -nmax : (l > nmax - 1 ? nmax - 1 : l);
            float w = x[i] * x[i];
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0.f && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                int l = (int)lrintf(iscale * x[i]);
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
        int v = (int)lrintf(iscale * scales[ib]);
        sc[ib] = (int8_t)(v > 127 ? 127 : v);
    }
    for (int j = 0; j < 256; ++j) {
        float dd = d * sc[j / 16];
        if (dd == 0.f) {
            continue;
        }
        int l = (int)lrintf(x[j] / dd);
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
        int l = (int)lrintf(iscale * (x[i] - min));
        L[i] = (uint8_t)(l < 0 ? 0 : (l > nmax ? nmax : l));
        float diff = scale * L[i] + min - x[i];
        best += weights[i] * diff * diff;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = (int)lrintf(iscale * (x[i] - min));
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
        int ls = (int)lrintf(inv_scale * scales[j]), lm = (int)lrintf(inv_min * mins[j]);
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
            int l = (int)lrintf((x[32 * j + ii] + dm) / d);
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
