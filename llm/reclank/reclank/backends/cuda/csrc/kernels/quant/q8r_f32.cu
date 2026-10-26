/* KQ_Q8_R rows of a step or a few tokens on float x, for a GPU without dp4a (Maxwell: NemotronGPU's Q8_0
 * matrices, repacked at the load; models/nemotron/gpu.py Q8R). The generic k_kq_linear took a warp a row
 * and read x from global memory for each (4 times the bytes of the weights): on the Quadro M4000 the head
 * of 131072 x 4096 14.3 ms (40 GB/s), the 4096 x 1024 latent projections 0.2 ms (20 GB/s). Here a warp
 * takes R rows, lane l the 16 values 16 l + 512 k of each (one 16-byte load, the scale of its 32), every
 * load of a pass first; x staged in shared memory once a block, each weight made a float once for the
 * NT tokens. The M4000, one token (bench in the scratch of the change): 100-153 GB/s, the head 3.7 ms.
 * The sums are float32 in an order of their own. */
#define Q8G_SW 8192         /* the columns of x in shared memory a slice: NT x SW / NT floats, 32 KB */

/* x staged with float4 u of each lane's 16 at 4 l + (u + l / 2) % 4: the 8 lanes of a quarter warp
 * read 8 different 16-byte banks. A slice of x after the first adds its sums to out. */
template <int R, int NT>
__global__ void __launch_bounds__(256) k_q8r_gemv(const uint8_t *w, int rows, int cols, const float *x, int t,
                                                 float *out)
{
    extern __shared__ __align__(16) float q8xs[];     /* NT x SW, rotated */
    const int lane = threadIdx.x % 32, nw = blockDim.x / 32;
    const size_t rb = kq_row_bytes(KQ_Q8_R, cols);
    const int ngr = (rows + R - 1) / R;
    const int SW = cols < Q8G_SW / NT ? cols : Q8G_SW / NT;
    for (int s0 = 0; s0 < cols; s0 += SW) {
        if (s0 > 0) {
            __syncthreads();
        }
        for (int i = threadIdx.x; i < t * SW / 4; i += blockDim.x) {
            const int j = i / (SW / 4), q = i % (SW / 4), L = q / 4, u = q % 4;
            ((float4 *)q8xs)[(size_t)j * (SW / 4) + 4 * L + (u + (L >> 1)) % 4] =
                ((const float4 *)(x + (size_t)j * cols + s0))[q];
        }
        __syncthreads();
        for (int gr = blockIdx.x * nw + threadIdx.x / 32; gr < ngr; gr += gridDim.x * nw) {
            const int row0 = gr * R;
            float acc[R][NT];
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    acc[r][j] = 0.f;
                }
            }
            for (int c0 = 16 * lane; c0 < SW; c0 += 512) {
                uint4 q[R];
                float d[R];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint8_t *wr = w + (size_t)(row0 + r < rows ? row0 + r : row0) * rb;
                    q[r] = *(const uint4 *)(wr + s0 + c0);
                    d[r] = __half2float(((const __half *)(wr + cols))[(s0 + c0) / 32]);
                }
                const int L = c0 / 16;
                float wv[R][16];
#pragma unroll
                for (int r = 0; r < R; ++r) {
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const uint32_t b = v == 0 ? q[r].x : v == 1 ? q[r].y : v == 2 ? q[r].z : q[r].w;
                        wv[r][4 * v] = (float)(int8_t)(b & 255);
                        wv[r][4 * v + 1] = (float)(int8_t)((b >> 8) & 255);
                        wv[r][4 * v + 2] = (float)(int8_t)((b >> 16) & 255);
                        wv[r][4 * v + 3] = (float)(int8_t)(b >> 24);
                    }
                }
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    if (j < t) {
                        float p[R];
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            p[r] = 0.f;
                        }
#pragma unroll
                        for (int v = 0; v < 4; ++v) {
                            const float4 xv = ((const float4 *)q8xs)[(size_t)j * (SW / 4) + 4 * L + (v + (L >> 1)) % 4];
#pragma unroll
                            for (int r = 0; r < R; ++r) {
                                p[r] += wv[r][4 * v] * xv.x + wv[r][4 * v + 1] * xv.y + wv[r][4 * v + 2] * xv.z +
                                        wv[r][4 * v + 3] * xv.w;
                            }
                        }
#pragma unroll
                        for (int r = 0; r < R; ++r) {
                            acc[r][j] += d[r] * p[r];
                        }
                    }
                }
            }
#pragma unroll
            for (int r = 0; r < R; ++r) {
#pragma unroll
                for (int j = 0; j < NT; ++j) {
                    float v = acc[r][j];
                    for (int o = 16; o > 0; o >>= 1) {
                        v += __shfl_xor_sync(RC_FULL_MASK, v, o);
                    }
                    if (lane == 0 && j < t && row0 + r < rows) {
                        float *o = out + (size_t)j * rows + row0 + r;
                        *o = s0 == 0 ? v : *o + v;
                    }
                }
            }
        }
    }
}

/* GP_KQ_LINEAR of KQ_Q8_R rows for t <= 8 (else 1: the generic kernels). The grid 4 blocks an SM for one
 * token, 3 for more (32 KB of x each); 2 rows a warp, 4 for 3 or 4 tokens (the M4000: 4096 x 4096,
 * 4 tokens 338 us against 349 for 2 rows; 8 tokens 934 us, the slices' sums added to out) */
static int q8r_linear_launch(const float *x, const uint8_t *w, int rows, int cols, int t, float *out, cudaStream_t s)
{
    if (t > 8 || cols % 512 != 0) {
        return 1;
    }
    const int sms = gg_caps.sm_count > 0 ? gg_caps.sm_count : 32;
    const int nt = t <= 1 ? 1 : t <= 2 ? 2 : t <= 4 ? 4 : 8, R = nt == 4 ? 4 : 2;
    const int sw = cols < Q8G_SW / nt ? cols : Q8G_SW / nt;
    const size_t sm = (size_t)nt * sw * 4;
    unsigned nblk = (unsigned)((rows + 8 * R - 1) / (8 * R)), cap = (unsigned)(sms * (t <= 1 ? 4 : 3));
    nblk = nblk < cap ? nblk : cap;
    if (nt == 1) {
        k_q8r_gemv<2, 1><<<nblk, 256, sm, s>>>(w, rows, cols, x, t, out);
    } else if (nt == 2) {
        k_q8r_gemv<2, 2><<<nblk, 256, sm, s>>>(w, rows, cols, x, t, out);
    } else if (nt == 4) {
        k_q8r_gemv<4, 4><<<nblk, 256, sm, s>>>(w, rows, cols, x, t, out);
    } else {
        k_q8r_gemv<2, 8><<<nblk, 256, sm, s>>>(w, rows, cols, x, t, out);
    }
    return 0;
}
