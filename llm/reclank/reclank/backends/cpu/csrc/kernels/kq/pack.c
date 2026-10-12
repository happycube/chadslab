/* The int4 matrices of the Gemma 4 26B (Q4_0 blocks of 18 bytes, the codes in
 * bytes 2 .. 17: value j in the low 4 bits of byte j, j + 16 in the high 4
 * bits; float32 scales, one for each block) to KQ_Q4X: rows a multiple of 16.
 * n matrices of rows rows (stride of the blocks: bstride matrices of rows at
 * the addresses of each). Return 0, or -1 when a scale is not exact in
 * float16. */
int rc_kq_q4x_pack(const uint8_t *packed, const float *scales, int64_t rows, int cols, uint8_t *dst)
{
    int nb = cols / 32, bad = 0;
    int64_t ng = rows / 16;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for (int64_t g = 0; g < ng; ++g) {
        uint8_t *o = dst + (size_t)g * nb * KQ_Q4X_BB;
        for (int b = 0; b < nb; ++b) {
            uint8_t *blk = o + (size_t)b * KQ_Q4X_BB;
            memset(blk, 0, 256);
            for (int r = 0; r < 16; ++r) {
                int64_t row = 16 * g + r;
                const uint8_t *src = packed + ((size_t)row * nb + b) * 18 + 2;
                for (int v = 0; v < 32; ++v) {
                    int c = v < 16 ? src[v] & 15 : src[v - 16] >> 4;
                    blk[32 * (v / 4) + 4 * (r % 8) + v % 4] |= (uint8_t)(c << (r < 8 ? 0 : 4));
                }
                float sc = scales[(size_t)row * nb + b];
                uint16_t h = kq_f32_to_f16(sc);
                bad |= kq_h((const uint8_t *)&h) != sc;
                memcpy(blk + 256 + 2 * r, &h, 2);
            }
        }
    }
    return bad ? -1 : 0;
}
