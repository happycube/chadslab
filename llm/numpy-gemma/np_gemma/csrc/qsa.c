/* QSA (Qwen Sparse Attention) of qwen4exp: the indexer that selects the keys
 * of each query, and the attention over the int16 cache on those keys.
 * np_gemma/qwen4.py (Qwen4.qsa_mask) has the math in NumPy.
 *
 * This file is part of the cops library: bf16_linear.c includes it. It uses
 * attn_i16_head_rows of bf16_linear.c.
 *
 * QSA_SELECT keeps the raw key of the indexer of each token (idxk, one row
 * of d values for each position) and the key of each complete block of
 * `ratio` tokens (blk: the mean of the raw keys of the block, the norm, and
 * RoPE at its first position). Both keep float16 values (the scores only
 * rank the blocks); the sums use float32. A block key is made one time, when its block
 * is complete. For each query, if more than budget blocks are complete, the
 * query scores each block with the sum over the heads of relu(q . k) /
 * sqrt(d), keeps the best budget blocks (of equal scores, the most recent),
 * and the tail. sel gets the positions in order, and cnt their count; cnt
 * -1 means all the positions before the query.
 *
 * ATTN_QSA is the attention of each query (the int16 cache of the 26B) on
 * the positions of sel, or on all the positions before it. A step and each
 * query of a group use the same code, so a group gives the bits of steps.
 */

/* float16 (round to nearest even) and back: the keys of the indexer */
static inline uint16_t qsa_h(float x)
{
#if GEMMA_X86
    return (uint16_t)_cvtss_sh(x, 0);
#else
    __fp16 h = (__fp16)x;
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
#endif
}

static inline float qsa_f(uint16_t u)
{
#if GEMMA_X86
    return _cvtsh_ss(u);
#else
    __fp16 h;
    memcpy(&h, &u, 2);
    return (float)h;
#endif
}

/* RMS norm of d values times w, then RoPE on the first rot values (the two
 * halves of rot), with the angles of position p. */
static void qsa_norm_rope(float *x, const float *w, int d, float eps, int rot, const float *c,
                          const float *s)
{
    float ss = 0.f;
    for (int i = 0; i < d; ++i) {
        ss += x[i] * x[i];
    }
    float inv = 1.f / sqrtf(ss / (float)d + eps);
    for (int i = 0; i < d; ++i) {
        x[i] = x[i] * inv * w[i];
    }
    int half = rot / 2;
    for (int i = 0; i < half; ++i) {
        float a = x[i], b = x[i + half];
        x[i] = a * c[i] - b * s[i];
        x[i + half] = b * c[i + half] + a * s[i + half];
    }
}

/* The top k of n keys, largest first: a partial quickselect of uint64 keys
 * (the bits of a score >= 0 above, the block below), so equal scores keep
 * the most recent block. keys is scratch of n values. */
static void qsa_topk(uint64_t *keys, int n, int k)
{
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        uint64_t pv = keys[(lo + hi) / 2];
        int i = lo, j = hi;
        while (i <= j) {
            while (keys[i] > pv) {
                ++i;
            }
            while (keys[j] < pv) {
                --j;
            }
            if (i <= j) {
                uint64_t tmp = keys[i];
                keys[i] = keys[j];
                keys[j] = tmp;
                ++i;
                --j;
            }
        }
        if (k - 1 <= j) {
            hi = j;
        } else if (k - 1 >= i) {
            lo = i;
        } else {
            break;
        }
    }
}

/* The position of the frequency pair i of the row p for RoPE: p, or with
 * qpos (the M-RoPE positions (time, row, column) of each row, after an
 * image) the row position if i % 3 == 1 and i < 3 s1, the column position
 * if i % 3 == 2 and i < 3 s2, else the time position. sec = s1 | s2 << 8. */
static inline int64_t qsa_rpos(const int32_t *qpos, int sec, int64_t p, int i)
{
    if (qpos == NULL) {
        return p;
    }
    int a = (i % 3 == 1 && i < 3 * (sec & 255)) ? 1 : (i % 3 == 2 && i < 3 * (sec >> 8)) ? 2 : 0;
    return qpos[p * 3 + a];
}

/* QSA_SELECT: iq, ik, idxk, blk, qn, kn, cos, sin, pos, t, heads, d, ratio,
 * budget, rot, theta, eps, sel, cnt, maxsel, scratch (for each thread:
 * (max blocks) * 3 words), nbmax, qpos (or 0), sec. */
static void qsa_select_body(const float *iq, const float *ik, uint16_t *idxk, uint16_t *blk,
                            const float *qn, const float *kn, const float *qcos,
                            const float *qsin, int64_t pos, int t, int heads, int d, int ratio,
                            int budget, int rot, float theta, float eps, int32_t *sel,
                            int32_t *cnt, int maxsel, uint8_t *scratch, int64_t nbmax,
                            const int32_t *qpos, int sec)
{
    int64_t n = pos + t;
    #pragma omp for schedule(static)
    for (int j = 0; j < t; ++j) {
        for (int i = 0; i < d; ++i) {
            idxk[(size_t)(pos + j) * d + i] = qsa_h(ik[(size_t)j * d + i]);
        }
    }
    /* the keys of the blocks that are complete now */
    #pragma omp for schedule(static)
    for (int64_t b = pos / ratio; b < n / ratio; ++b) {
        float o[256];
        for (int i = 0; i < d; ++i) {
            float sum = 0.f;
            for (int r = 0; r < ratio; ++r) {
                sum += qsa_f(idxk[(size_t)(b * ratio + r) * d + i]);
            }
            o[i] = sum / (float)ratio;
        }
        float c[256], s[256];
        for (int i = 0; i < rot / 2; ++i) {
            double f = (double)qsa_rpos(qpos, sec, b * ratio, i)
                       / pow((double)theta, (double)(2 * i) / (double)rot);
            c[i] = c[i + rot / 2] = (float)cos(f);
            s[i] = s[i + rot / 2] = (float)sin(f);
        }
        qsa_norm_rope(o, kn, d, eps, rot, c, s);
        for (int i = 0; i < d; ++i) {
            blk[(size_t)b * d + i] = qsa_h(o[i]);
        }
    }
    #pragma omp for schedule(dynamic, 1)
    for (int j = 0; j < t; ++j) {
        int64_t pj = pos + j, nb = (pj + 1) / ratio;
        if (nb <= budget) {
            cnt[j] = -1;
            continue;
        }
        uint8_t *sp = scratch + (size_t)omp_get_thread_num() * (size_t)nbmax * 12;
        uint64_t *keys = (uint64_t *)sp;
        uint8_t *keep = sp + (size_t)nbmax * 8;
        float q[8 * 256];
        for (int h = 0; h < heads; ++h) {
            memcpy(q + h * d, iq + ((size_t)j * heads + h) * d, (size_t)d * 4);
            qsa_norm_rope(q + h * d, qn, d, eps, rot, qcos + (size_t)j * rot,
                          qsin + (size_t)j * rot);
        }
        float scale = 1.f / sqrtf((float)d);
        for (int64_t b = 0; b < nb; ++b) {
            float kb[256];
            for (int i = 0; i < d; ++i) {
                kb[i] = qsa_f(blk[(size_t)b * d + i]);
            }
            float score = 0.f;
            for (int h = 0; h < heads; ++h) {
                float dot = 0.f;
                for (int i = 0; i < d; ++i) {
                    dot += q[h * d + i] * kb[i];
                }
                score += dot > 0.f ? dot : 0.f;
            }
            score *= scale;
            uint32_t bits;
            memcpy(&bits, &score, 4);
            keys[b] = ((uint64_t)bits << 32) | (uint64_t)b;
        }
        qsa_topk(keys, (int)nb, budget);
        memset(keep, 0, (size_t)nb);
        for (int k = 0; k < budget; ++k) {
            keep[keys[k] & 0xffffffffu] = 1;
        }
        int32_t *out = sel + (size_t)j * maxsel;
        int c2 = 0;
        for (int64_t b = 0; b < nb; ++b) {
            if (keep[b]) {
                for (int r = 0; r < ratio; ++r) {
                    out[c2++] = (int32_t)(b * ratio + r);
                }
            }
        }
        for (int64_t p = nb * ratio; p <= pj; ++p) {
            out[c2++] = (int32_t)p;
        }
        cnt[j] = c2;
    }
}

/* The address of value o of a row of the cache: 2 bytes (i8 0), 1 (i8 1),
 * or 3 bytes for 4 values (i8 2, TQ6). */
static inline const void *qsa_at(const void *p, size_t o, int i8)
{
    return (const char *)p + (i8 == 2 ? o * 3 / 4 : o * (i8 ? 1 : 2));
}

/* attn_i16_head_rows for the int8 forms of the cache (NP_GEMMA_QWEN_KV int8
 * or k16v8): ki8, vi8 say which of the keys and the values are int8 (2: TQ6,
 * the rotated form). A row goes to float32 times its scales (as_row), then
 * one dot product. */
static void attn_q_head_rows(const float *qh, const void *kq, const float *ks, const void *vq,
                             const float *vs, float *sc, float *oh, int kv, int hd,
                             size_t kv_stride, size_t ks_stride, int n, const int32_t *rows,
                             int ki8, int vi8)
{
    float row[1024];
    int g = hd / 32;
    for (int j = 0; j < n; ++j) {
        size_t r = rows ? (size_t)rows[j] : (size_t)j;
        size_t o = r * kv_stride + (size_t)kv * hd;
        as_row(qsa_at(kq, o, ki8), ks + r * ks_stride + (size_t)kv * g, hd,
               row, ki8);
        sc[j] = as_dot(qh, row, hd);
    }
    float m = sc[0];
    for (int j = 1; j < n; ++j) {
        m = sc[j] > m ? sc[j] : m;
    }
    float l = 0.0f;
    for (int j = 0; j < n; ++j) {
        sc[j] = expf(sc[j] - m);
        l += sc[j];
    }
    float inv = 1.0f / l;
    for (int d = 0; d < hd; ++d) {
        oh[d] = 0.0f;
    }
    for (int j = 0; j < n; ++j) {
        size_t r = rows ? (size_t)rows[j] : (size_t)j;
        size_t o = r * kv_stride + (size_t)kv * hd;
        as_row(qsa_at(vq, o, vi8), vs + r * ks_stride + (size_t)kv * g, hd,
               row, vi8);
        as_axpy(oh, sc[j] * inv, row, hd);
    }
}

/* attn_i16_head_rows for float32 rows: head kv of K and V starts at kv hs,
 * a row is rs floats (gp_kv_rs; the first full layer of the cache of
 * Qwen3.8: NP_GEMMA_QWEN_KV_FIRST f32). */
static void attn_f32_head_rows(const float *qh, const float *K, const float *V, float *sc,
                               float *oh, int kv, int hd, size_t hs, size_t rs, int n,
                               const int32_t *rows)
{
    const float *Kh = K + (size_t)kv * hs, *Vh = V + (size_t)kv * hs;
    for (int j = 0; j < n; ++j) {
        size_t r = rows ? (size_t)rows[j] : (size_t)j;
        sc[j] = as_dot(qh, Kh + r * rs, hd);
    }
    float m = sc[0];
    for (int j = 1; j < n; ++j) {
        m = sc[j] > m ? sc[j] : m;
    }
    float l = 0.0f;
    for (int j = 0; j < n; ++j) {
        sc[j] = expf(sc[j] - m);
        l += sc[j];
    }
    float inv = 1.0f / l;
    for (int d = 0; d < hd; ++d) {
        oh[d] = 0.0f;
    }
    for (int j = 0; j < n; ++j) {
        size_t r = rows ? (size_t)rows[j] : (size_t)j;
        as_axpy(oh, sc[j] * inv, Vh + r * rs, hd);
    }
}

/* ATTN_QSA: q (t x nq x hd), kq, ks, vq, vs, scores, out, nq, nk, hd, t, pos,
 * sel, cnt, maxsel, and (or none) form: 0 the int16 cache, 1 int8, 2 int16
 * keys and int8 values, 3 float32 rows (kq and vq the float K and V of
 * (heads, positions, hd), the head stride hs), 4 TQ6 (q and out in the
 * rotated form: GP_TQ_ROT). scores has (the positions)
 * floats for each thread. */
static void attn_qsa_body(const float *q, const void *kq, const float *ks, const void *vq,
                          const float *vs, float *scores, float *out, int nq, int nk, int hd,
                          int t, int64_t pos, const int32_t *sel, const int32_t *cnt,
                          int maxsel, int form, int64_t hs)
{
    int rep = nq / nk;
    size_t kv_stride = (size_t)nk * hd, ks_stride = (size_t)nk * (hd / 32);
    #pragma omp for schedule(static)
    for (int u = 0; u < t * nq; ++u) {
        int j = u / nq, h = u % nq;
        int n = cnt[j] < 0 ? (int)(pos + j + 1) : cnt[j];
        const int32_t *rows = cnt[j] < 0 ? NULL : sel + (size_t)j * maxsel;
        float *sc = scores + (size_t)omp_get_thread_num() * (size_t)(pos + t);
        if (form == 3) {
            attn_f32_head_rows(q + (size_t)u * hd, (const float *)kq, (const float *)vq, sc,
                               out + (size_t)u * hd, h / rep, hd, (size_t)hs,
                               gp_kv_rs((size_t)hs, nk, hd), n, rows);
        } else if (form == 0) {
            attn_i16_head_rows(q + (size_t)u * hd, (const int16_t *)kq, ks, (const int16_t *)vq,
                               vs, sc, out + (size_t)u * hd, h / rep, hd, kv_stride, ks_stride,
                               n, rows);
        } else {
            attn_q_head_rows(q + (size_t)u * hd, kq, ks, vq, vs, sc, out + (size_t)u * hd,
                             h / rep, hd, kv_stride, ks_stride, n, rows,
                             form == 4 ? 2 : form == 1, form == 4 ? 2 : 1);
        }
    }
}
