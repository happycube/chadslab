/* The GGUF block formats of llama.cpp: products for one token, a small
 * group, and the experts of a MoE layer.
 *
 * This file is part of the cops library: bf16_linear.c includes it, and it
 * uses the helpers there (fp16_to_f32) and in moe.c.
 *
 * The formats (the numbers are the ggml types):
 *
 *     KQ_F32   0   float32 rows.
 *     KQ_Q8_0  8   blocks of 32: d (fp16), 32 int8. w = d * q.
 *     KQ_Q4_K 12   blocks of 256 (144 bytes): d, dmin (fp16), 12 bytes of
 *                  6-bit scales and mins for 8 parts of 32, 128 bytes of
 *                  4-bit values. w = d * sc * q - dmin * m.
 *     KQ_Q5_K 13   Q4_K with a fifth bit (32 more bytes, qh).
 *     KQ_Q5_1  7   blocks of 32 (24 bytes): d, m (fp16), 32 fifth bits,
 *                  16 bytes of 4-bit values. w = d * q + m.
 *     KQ_IQ4_NL 20 blocks of 32 (18 bytes): d, 16 bytes of 4-bit codes into
 *                  a table of 16 values (only rc_kq_rows: the n-gram table).
 *     KQ_Q6_K 14   blocks of 256 (210 bytes): 128 bytes of the low 4 bits,
 *                  64 bytes of the high 2 bits, 16 int8 scales (one for
 *                  each 16 values), d. w = d * sc * (q - 32).
 *     KQ_BF16 30   bfloat16 rows (the safetensors checkpoints).
 *     KQ_Q8X16 60  Q8_0 in groups of 16 rows (this runtime; rc_kq_pack_q8x16 makes
 *                  it from Q8_0 for the CPU): for each group and each block
 *                  of 32 columns, 576 bytes: the 16 fp16 scales of the rows,
 *                  the 16 sums of the int8 values of the block of each row
 *                  (int16), then 8 steps of 64 bytes; step k holds values
 *                  4k to 4k + 3 of the 16 rows (row r in bytes 4r to 4r + 3).
 *                  A lane of vpdpbusd is a row: no sums across lanes, and an
 *                  int32 sum for each block (see kq_x16_body).
 *     KQ_NV4  51   NVFP4 in rows for this runtime (reclank/formats/nvfp4_st.py
 *                  repacks the checkpoint of NVIDIA ModelOpt). A row: the
 *                  4-bit E2M1 codes (cols / 2 bytes; block b of 32 values in
 *                  bytes 16b .. 16b + 15, value j in the low 4 bits of byte
 *                  j, value j + 16 in the high 4 bits), the E4M3 scales
 *                  (cols / 16 bytes; 2 for each block: values 0-15, 16-31),
 *                  the float32 scale of the matrix, and zeros to a multiple
 *                  of 16 bytes. So the codes of a row start at a multiple of
 *                  16 bytes, and the scales at a multiple of 4 (the loads of
 *                  the GPU; kq_nv4_* give the places). w = g * s *
 *                  e2m1(code); 2 e2m1 is an integer (0, 1, 2, 3, 4, 6, 8,
 *                  12 and the negatives).
 *     KQ_BF16X16 61, KQ_F32X16 62  bfloat16 or float32 in groups of 16 rows
 *                  (this runtime, only in memory; rc_kq_pack_x16f makes them
 *                  for the CPU): for each column, the values of the 16 rows
 *                  (32 or 64 bytes). The rows go to a multiple of 16 with
 *                  zeros. A lane is a row, x is float32 (not int8), and a
 *                  step is one fma for each token (kq_x16f_body). Only for
 *                  64 rows or more (Qwen4CPU.KP): the smaller matrices stay
 *                  in rows, on tasks of a row and 16 tokens.
 *     KQ_Q4X  54   Q4_0 in groups of 16 rows (this runtime, only in memory;
 *                  rc_kq_q4x_pack makes it from the int4 matrices of the Gemma 4
 *                  26B: Q4_0 blocks and float32 scales). For each block of 32
 *                  columns, 288 bytes: the 8 steps of the codes of KQ_NVX (q
 *                  = w + 8, 0 to 15) and the float16 scales of the 16 rows.
 *                  A "row" is cols / 32 * 18 bytes.
 *     KQ_NVX  53   NVFP4 in groups of 16 rows (this runtime; the GGUF of
 *                  scripts/convert_nvfp4_gguf.py; rc_kq_nvx_pack). A group: 16
 *                  bytes (the float32 scale of the matrix, zeros), then for
 *                  each block of 32 columns 288 bytes: 8 steps of 32 bytes
 *                  and 32 E4M3 scales. Step s holds values 4s to 4s + 3 of
 *                  the 16 rows: byte 4r + u (r < 8) has value 4s + u of row
 *                  r in its low 4 bits, and that of row r + 8 in its high 4
 *                  bits. The scales: values 0-15 of rows 0 to 15, then
 *                  values 16-31. The CPU: a step is one vpdpbusd, a lane for
 *                  each row (kq_nvx_rows). The GPU: the 4 bytes of a lane
 *                  are a fragment of the tensor cores for 2 rows. A "row" is
 *                  cols / 32 * 18 + 1 bytes (a group is 16 of them).
 *
 * The products quantize x to int8 in its natural order, with one scale xs
 * for each 32 values and xm, xs times the sum of the int8 values of each 16
 * values
 * (rc_kq_quant_x). The same x serves all the formats. For each part of a
 * block:
 *
 *     Q8_0:  y += d * xs * sum(q * xq)
 *     Q4_K:  y += d * sc * xs * sum(q * xq) - dmin * m * xs * sum(xq)
 *     Q6_K:  y += d * sc * xs * (sum(q * xq) - 32 * sum(xq))
 *
 * The values of the K formats are not negative, which is the operation of
 * the VNNI instruction vpdpbusd (u8 x s8 -> s32). Q8_0 is signed: the
 * product uses |q| and x with the sign of q.
 *
 * All the products of one row use one function for one token, so a token
 * gives the same bits alone and in a group (an MTP verify group).
 */
#define KQ_F32 0
#define KQ_Q8_0 8
#define KQ_Q4_K 12
#define KQ_Q5_K 13
#define KQ_Q6_K 14
#define KQ_Q5_1 7
#define KQ_IQ4_NL 20
#define KQ_BF16 30
#define KQ_NV4 51
#define KQ_Q8X16 60
#define KQ_NVX 53
#define KQ_Q4X 54
#define KQ_Q4X_BB 288
#ifndef KQ_Q4X_TB
#define KQ_Q4X_TB 16
#endif
#define KQ_BF16X16 61
#define KQ_F32X16 62
#define KQ_NVX_BB 288
/* BF12 (plan-scripts/BF12_PLAN.md; gguf.BF12): the bfloat16 values of a row
 * in groups of 32: lo[cols] (sign << 7 | the 7 bits of the mantissa), hi[cols
 * / 2] (the exponent gap E - e, 0 .. 15: byte 16 g + j has the gap of value
 * 32 g + j in its low half, of 32 g + j + 16 in its high half), E[cols / 32]
 * (the largest exponent of group g), padded to 16 bytes. Sign 1, gap 15,
 * mantissa 0 decodes to 0 (the zero code); bits sign << 15 | (E - gap) << 7
 * | mantissa else. */
#define KQ_BF12 57
#define KQ_BF12X16 63
/* DeepSeek V4.1 Flash (mxx.c): KQ_MXX, MXFP4 in the groups of 16 rows of
 * KQ_NVX with an E8M0 scale of each row and block (272 bytes for 32
 * columns); KQ_FP8B32, FP8 rows with E8M0 scales of 32 x 32 blocks. */
#define KQ_MXX 64
#define KQ_FP8B32 65
#define KQ_MXX_BB 272
