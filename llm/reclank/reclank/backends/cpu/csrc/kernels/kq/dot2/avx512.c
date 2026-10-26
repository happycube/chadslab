/* The dots of dot2/avx2.c, and on AVX-512BW without VNNI (the tier avx512: Skylake-SP and -X,
 * the Xeon W-2100) four rows of Q4_K or Q5_K at a time on n tokens (kq2_rows4_q45k, from
 * kq_rows4): the cold experts of a decode step.
 *
 * kq2_q45k a row at a time took about 50 instructions for each 64 values of a row, many of them
 * scalar (kq2_scale_min's branch and byte loads, the scale d s xs of each part, the mins): a
 * thread of the W-2133 read 4.2 GB/s, the cold experts of a Qwen3.6 step 24 GB/s on 6 threads
 * against 60 GB/s of reads. Here two rows share each 512-bit vector (a row in each half, the 8
 * int32 lanes of kq2_q45k in each), the scales and mins of 4 rows come from the 12 bytes of each
 * block in vectors, and the mins of the 4 rows add in the lanes of a vector. Each row has the
 * operations of kq2_q45k in its order (the int32 sums of 4 products, the FMA chain of its 8
 * lanes, kq2_hsum, the chain of the mins as GCC contracts it there): the same bits. */
#include "avx2.c"

#if defined(__AVX512BW__)
#define KQ2_ROWS4 1
#define KQ2_R4_NB 64            /* the most blocks of 256 values of a row */

/* RECLANK_KQ2_ROWS4 (1): kq2_rows4_q45k for 4 rows of Q4_K or Q5_K; 0: kq_dot1 a row at a time */
static int kq2_rows4_on(void)
{
    static int on = -1;
    if (on < 0) {
        const char *v = getenv("RECLANK_KQ2_ROWS4");
        on = !(v && v[0] == '0');
    }
    return on;
}

/* RECLANK_KQ2_PF (1): kq2_rows4_q45k prefetches (to L1) the 4 rows this many groups of 4 after
 * its own (the next task of the thread, mostly), spread over its blocks; 0: none. The hardware
 * prefetcher did not follow the 4 rows read a block of each at a time: one thread read 5.7 GB/s
 * of the memory, 18.6 in L1; with it 9.8. Three cold experts of Qwen3.6 on 6 threads: 218 -> 142
 * us a call; 2 or 3 groups ahead, or to L2 (T1, T2), no better; NTA 243. */
static int kq2_pf_groups(void)
{
    static int n = -1;
    if (n < 0) {
        const char *v = getenv("RECLANK_KQ2_PF");
        n = v ? atoi(v) : 1;
    }
    return n;
}

static void kq2_rows4_q45k(const uint8_t *w, size_t rb, int five, int cols, const int8_t *xq, const float *xs,
                           const float *xm, int n, float *out, size_t ostride)
{
    const int nb = cols / 256;
    const size_t bs = five ? 176 : 144;
    const uint8_t *wr[4] = {w, w + rb, w + 2 * rb, w + 3 * rb};
    __m512 dsP[2][KQ2_R4_NB];               /* d sc of rows 0, 1 (2, 3) in the halves: lanes 0-7, 8-15 */
    __m512 mE[KQ2_R4_NB], mO[KQ2_R4_NB];    /* lane 4c + r: the min of part 2c (mO: 2c + 1) of row r */
    __m128 dmn[KQ2_R4_NB];                  /* the dmin of the 4 rows */
    {
        const __m512i k1 = _mm512_set1_epi32(0x3f3f3f3f), k2 = _mm512_set1_epi32(0x0f0f0f0f);
        const __m512i k3 = _mm512_set1_epi32(0x03030303);
        const __m512i sv = _mm512_set_epi32(4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0);
        const __m512i iSC = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23);
        const __m512i iM = _mm512_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15, 24, 25, 26, 27, 28, 29, 30, 31);
        const __m512i iE = _mm512_setr_epi32(0, 8, 16, 24, 2, 10, 18, 26, 4, 12, 20, 28, 6, 14, 22, 30);
        const __m512i iO = _mm512_add_epi32(iE, _mm512_set1_epi32(1));
        const __m512i iD0 = _mm512_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2);
        const __m512i iD1 = _mm512_add_epi32(iD0, _mm512_set1_epi32(4));
        const __m256i iDm = _mm256_setr_epi32(1, 3, 5, 7, 1, 3, 5, 7);
        for (int b = 0; b < nb; ++b) {
            const size_t o = (size_t)b * bs;
            /* the 12 bytes of scales and mins of each row in a 128-bit lane (dwords u0 u1 u2): the
             * words of kq_row_scales (sc 0-7 in u0' u1', m 0-7 in u2' u3') without a branch */
            __m512i v = _mm512_castsi128_si512(_mm_loadu_si128((const __m128i *)(wr[0] + o + 4)));
            v = _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i *)(wr[1] + o + 4)), 1);
            v = _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i *)(wr[2] + o + 4)), 2);
            v = _mm512_inserti32x4(v, _mm_loadu_si128((const __m128i *)(wr[3] + o + 4)), 3);
            const __m512i P = _mm512_shuffle_epi32(v, _MM_SHUFFLE(2, 1, 2, 0));     /* u0 u2 u1 u2 */
            const __m512i Q = _mm512_shuffle_epi32(v, _MM_SHUFFLE(1, 1, 0, 0));     /* u0 u0 u1 u1 */
            const __m512i hi = _mm512_or_si512(_mm512_and_si512(_mm512_srlv_epi32(P, sv), k2),
                                               _mm512_slli_epi32(_mm512_and_si512(_mm512_srli_epi32(Q, 6), k3), 4));
            const __m512i u = _mm512_mask_blend_epi32(0xaaaa, _mm512_and_si512(P, k1), hi);
            __m512 f[4];        /* row r: sc 0-7, m 0-7 */
            f[0] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_castsi512_si128(u)));
            f[1] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u, 1)));
            f[2] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u, 2)));
            f[3] = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm512_extracti32x4_epi32(u, 3)));
            uint32_t h[4];
            for (int r = 0; r < 4; ++r) {
                memcpy(&h[r], wr[r] + o, 4);
            }
            /* d, dmin of each row (kq_h: the same conversion) */
            const __m256 dd = _mm256_cvtph_ps(_mm_setr_epi32((int)h[0], (int)h[1], (int)h[2], (int)h[3]));
            const __m512 dd5 = _mm512_castps256_ps512(dd);
            dsP[0][b] = _mm512_mul_ps(_mm512_permutexvar_ps(iD0, dd5), _mm512_permutex2var_ps(f[0], iSC, f[1]));
            dsP[1][b] = _mm512_mul_ps(_mm512_permutexvar_ps(iD1, dd5), _mm512_permutex2var_ps(f[2], iSC, f[3]));
            const __m512 m01 = _mm512_permutex2var_ps(f[0], iM, f[1]), m23 = _mm512_permutex2var_ps(f[2], iM, f[3]);
            mE[b] = _mm512_permutex2var_ps(m01, iE, m23);
            mO[b] = _mm512_permutex2var_ps(m01, iO, m23);
            dmn[b] = _mm256_castps256_ps128(_mm256_permutevar8x32_ps(dd, iDm));
        }
    }
    const __m512i m4 = _mm512_set1_epi8(15), one = _mm512_set1_epi8(1), ones16 = _mm512_set1_epi16(1);
    /* lanes 0-7 part k of the first row of a pair, 8-15 of the second */
    __m512i iS[8];
    for (int k = 0; k < 8; ++k) {
        iS[k] = _mm512_mask_blend_epi32(0xff00, _mm512_set1_epi32(k), _mm512_set1_epi32(8 + k));
    }
    /* lane 4c + r: xm 4c (iA), 4c + 1 (iB), 4c + 2, 4c + 3 of a block */
    const __m512i iA = _mm512_setr_epi32(0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12);
    const __m512i iB = _mm512_add_epi32(iA, _mm512_set1_epi32(1));
    const __m512i iC = _mm512_add_epi32(iA, _mm512_set1_epi32(2)), iD = _mm512_add_epi32(iA, _mm512_set1_epi32(3));
    const int pfg = kq2_pf_groups();
    const size_t gb = 4 * rb;           /* the bytes of a group of 4 rows */
    const uintptr_t pfa = (uintptr_t)w + gb * (size_t)pfg;
    for (int j = 0; j < n; ++j) {
        const int8_t *x = xq + (size_t)j * cols;
        const float *xsj = xs + (size_t)j * (cols / 32), *xmj = xm + (size_t)j * (cols / 16);
        __m512 a01 = _mm512_setzero_ps(), a23 = _mm512_setzero_ps();
        __m128 mins = _mm_setzero_ps();
        for (int b = 0; b < nb; ++b) {
            const size_t o = (size_t)b * bs;
            if (j == 0 && pfg > 0) {
                for (uintptr_t p = (pfa + gb * b / nb) & ~(uintptr_t)63; p < pfa + gb * (b + 1) / nb; p += 64) {
                    _mm_prefetch((const char *)p, _MM_HINT_T0);
                }
            }
            /* the scale of each part: (d sc) xs, as kq2_q45k's d * (float)s * xs */
            const __m512 xs2 = _mm512_castpd_ps(_mm512_broadcast_f64x4(_mm256_loadu_pd((const double *)(xsj + 8 * b))));
            const __m512 s01 = _mm512_mul_ps(dsP[0][b], xs2), s23 = _mm512_mul_ps(dsP[1][b], xs2);
            /* the mins: dmin ((float)m0 (xm + xm) + (float)m1 (xm + xm)), added to the chain of each row */
            const __m512 X = _mm512_loadu_ps(xmj + 16 * b);
            const __m512 Te = _mm512_add_ps(_mm512_permutexvar_ps(iA, X), _mm512_permutexvar_ps(iB, X));
            const __m512 To = _mm512_add_ps(_mm512_permutexvar_ps(iC, X), _mm512_permutexvar_ps(iD, X));
#if KQ2_R4_INNER == 1
            const __m512 inner = _mm512_fmadd_ps(mO[b], To, _mm512_mul_ps(mE[b], Te));
#elif KQ2_R4_INNER == 2
            const __m512 inner = _mm512_add_ps(_mm512_mul_ps(mE[b], Te), _mm512_mul_ps(mO[b], To));
#else
            const __m512 inner = _mm512_fmadd_ps(mE[b], Te, _mm512_mul_ps(mO[b], To));
#endif
            const __m128 dm = dmn[b];
#if KQ2_R4_MINS == 1
#define KQ2_MIN(I) mins = _mm_add_ps(mins, _mm_mul_ps(dm, _mm512_extractf32x4_ps(inner, I)))
#else
#define KQ2_MIN(I) mins = _mm_fmadd_ps(dm, _mm512_extractf32x4_ps(inner, I), mins)
#endif
            KQ2_MIN(0);
            KQ2_MIN(1);
            KQ2_MIN(2);
            KQ2_MIN(3);
#undef KQ2_MIN
            const uint8_t *q0 = wr[0] + o + (five ? 48 : 16), *q1 = wr[1] + o + (five ? 48 : 16);
            const uint8_t *q2 = wr[2] + o + (five ? 48 : 16), *q3 = wr[3] + o + (five ? 48 : 16);
            __m512i qh01 = _mm512_setzero_si512(), qh23 = qh01;
            if (five) {
                qh01 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(wr[0] + o + 16))),
                                          _mm256_loadu_si256((const __m256i *)(wr[1] + o + 16)), 1);
                qh23 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(wr[2] + o + 16))),
                                          _mm256_loadu_si256((const __m256i *)(wr[3] + o + 16)), 1);
            }
            for (int c = 0; c < 4; ++c) {
                const int8_t *xc = x + 256 * b + 64 * c;
                const __m512i xa = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)xc));
                const __m512i xb = _mm512_broadcast_i64x4(_mm256_loadu_si256((const __m256i *)(xc + 32)));
/* rows A, B (qA, qB) of a pair: the 32 bytes of part pair c of each, its fifth bits in qh, the scales s */
#define KQ2_PAIR(ACC, QA, QB, QH, S) { \
                __m512i q = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(QA + 32 * c))), \
                                               _mm256_loadu_si256((const __m256i *)(QB + 32 * c)), 1); \
                __m512i lo = _mm512_and_si512(q, m4), hi = _mm512_and_si512(_mm512_srli_epi16(q, 4), m4); \
                if (five) { \
                    lo = _mm512_or_si512(lo, _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(QH, 2 * c), one), 4)); \
                    hi = _mm512_or_si512(hi, _mm512_slli_epi16(_mm512_and_si512(_mm512_srli_epi16(QH, 2 * c + 1), one), 4)); \
                } \
                __m512i ia = _mm512_madd_epi16(_mm512_maddubs_epi16(lo, xa), ones16); \
                __m512i ie = _mm512_madd_epi16(_mm512_maddubs_epi16(hi, xb), ones16); \
                ACC = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ia), _mm512_permutexvar_ps(iS[2 * c], S), ACC); \
                ACC = _mm512_fmadd_ps(_mm512_cvtepi32_ps(ie), _mm512_permutexvar_ps(iS[2 * c + 1], S), ACC); }
                KQ2_PAIR(a01, q0, q1, qh01, s01)
                KQ2_PAIR(a23, q2, q3, qh23, s23)
#undef KQ2_PAIR
            }
        }
        float mn[4];
        _mm_storeu_ps(mn, mins);
        const __m256 h[4] = {_mm512_castps512_ps256(a01), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a01), 1)),
                             _mm512_castps512_ps256(a23), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a23), 1))};
        for (int r = 0; r < 4; ++r) {
            out[(size_t)j * ostride + r] = kq2_hsum(h[r]) - mn[r];
        }
    }
}
#endif
