/* The AVX2 dots of dot2/avx2.c with vpdpbusd (AVX-VNNI, the tier avx2_vnni:
 * Alder Lake and later, Zen 5): one instruction in place of vpmaddubsw and
 * vpmaddwd, the same int32 lanes and the same results.
 *
 * RC_VNNI_EVEX (a test build on a CPU with AVX-512 VNNI and VL, not
 * AVX-VNNI) takes the 256-bit EVEX form of the instruction instead. */
#if defined(__AVXVNNI__)
#define KQ2_DOT(acc, u, s) _mm256_dpbusd_avx_epi32(acc, u, s)
#elif defined(RC_VNNI_EVEX)
__attribute__((target("avx512vnni,avx512vl"))) static __m256i kq2_dot_evex(__m256i acc, __m256i u, __m256i s)
{
    return _mm256_dpbusd_epi32(acc, u, s);
}
#define KQ2_DOT(acc, u, s) kq2_dot_evex(acc, u, s)
#endif
#include "avx2.c"
