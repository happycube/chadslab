// The decode attention of a global layer of the Gemma 4 26B (16 query heads,
// 2 key heads, head_dim 512) on the tensor cores, k_attn_fdtc of
// np_gemma/csrc/gpu.cu, alone: its time and rate over N random rows of the
// cache (FORM 0 int16, 1 int8, 2 int16 keys and int8 values) for T queries
// (an MTP verify group: query j sees N + j rows). Each query of the group
// must give the same bits as a record of one query at its position (MTP
// verifies with the numbers of the decode). With CHECK 1, the error against
// a float64 attention on the host (small N). WIN > 0: a sliding layer
// (k_attn_fdts: head_dim 256, 8 key heads; query j sees rows N + j - WIN to
// N + j - 1). The GPU rests at a low clock: the first runs bring it up.
//
//     nvcc -O3 -arch=native -o bench_attn_fdtc scripts/bench_attn_fdtc.cu
//     ./bench_attn_fdtc [N 100000] [FORM 0] [CHECK 0] [T 1] [WIN 0] [BASE 0]
#include "../np_gemma/csrc/gpu.cu"
#include <vector>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <random>

int main(int argc, char **argv)
{
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    int form = argc > 2 ? atoi(argv[2]) : 0;     // 0 qc, 1 q8, 2 v8
    int check = argc > 3 ? atoi(argv[3]) : 0;
    int T = argc > 4 ? atoi(argv[4]) : 1;
    int win = argc > 5 ? atoi(argv[5]) : 0;
    int base0 = argc > 6 ? atoi(argv[6]) : 0;      // the position of row 0 (a window)
    const int qh = 16, kvh = win ? 8 : 2, hd = win ? 256 : 512, rs = kvh * hd;
    int rows = n + T - 1;
    std::mt19937 rng(1);
    std::normal_distribution<float> nd(0.f, 1.f);
    size_t nv = (size_t)rows * rs;
    std::vector<int16_t> k16(nv), v16(nv);
    std::vector<int8_t> k8(nv), v8(nv);
    std::vector<float> ks(nv / 32), vs(nv / 32), q((size_t)T * qh * hd);
    for (size_t i = 0; i < nv / 32; ++i) { ks[i] = 1e-4f * (1 + (i % 7)); vs[i] = 1e-4f * (1 + (i % 5)); }
    std::vector<float> ks8 = ks, vs8 = vs;
    for (size_t i = 0; i < nv / 32; ++i) { ks8[i] *= 32767.f / 127.f; vs8[i] *= 32767.f / 127.f; }
    for (size_t i = 0; i < nv; ++i) {
        float a = nd(rng) * 8000.f, b = nd(rng) * 8000.f;
        k16[i] = (int16_t)fmaxf(-32767.f, fminf(32767.f, a));
        v16[i] = (int16_t)fmaxf(-32767.f, fminf(32767.f, b));
        k8[i] = (int8_t)lrintf(k16[i] * 127.f / 32767.f);
        v8[i] = (int8_t)lrintf(v16[i] * 127.f / 32767.f);
    }
    for (auto &x : q) x = nd(rng) * 0.06f;
    void *dk, *dv, *dks, *dvs, *dq, *dout, *dout1, *dpart, *dsc;
    size_t kb = form == 1 ? nv : 2 * nv, vb = form == 0 ? 2 * nv : nv;
    cudaMalloc(&dk, kb); cudaMalloc(&dv, vb);
    cudaMalloc(&dks, nv / 8); cudaMalloc(&dvs, nv / 8);
    cudaMalloc(&dq, q.size() * 4); cudaMalloc(&dout, q.size() * 4); cudaMalloc(&dout1, q.size() * 4);
    cudaMalloc(&dsc, 64);
    cudaMalloc(&dpart, GG_PART_FLOATS * 4);
    cudaMemcpy(dk, form == 1 ? (void *)k8.data() : (void *)k16.data(), kb, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, form == 0 ? (void *)v16.data() : (void *)v8.data(), vb, cudaMemcpyHostToDevice);
    cudaMemcpy(dks, (form == 1 ? ks8 : ks).data(), nv / 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dvs, (form == 0 ? vs : vs8).data(), nv / 8, cudaMemcpyHostToDevice);
    cudaMemcpy(dq, q.data(), q.size() * 4, cudaMemcpyHostToDevice);
    int op = form == 0 ? GP_ATTN_QC : form == 1 ? GP_ATTN_Q8 : GP_ATTN_V8;
    // the record of t queries from query j0 (its n0 keys) into out; the
    // buffer from row sh (a window: the rows before it dropped, base sh)
    size_t kes = form == 1 ? 1 : 2, ves = form == 0 ? 2 : 1;
    auto record = [&](int t, int j0, int n0, void *out, int sh = 0) {
        gp_rec rec;
        memset(&rec, 0, sizeof(rec));
        rec.op = op;
        size_t o = (size_t)sh * rs;
        int64_t vals[14] = {(int64_t)((float *)dq + (size_t)j0 * qh * hd), (int64_t)((char *)dk + o * kes),
                            (int64_t)((float *)dks + o / 32), (int64_t)((char *)dv + o * ves),
                            (int64_t)((float *)dvs + o / 32), (int64_t)dsc,
                            (int64_t)((float *)out + (size_t)j0 * qh * hd), qh, kvh, hd, n0 - sh, t, win,
                            base0 + sh};
        for (int i = 0; i < (win ? 14 : t > 1 ? 12 : 11); ++i) { rec.tag[i] = GP_T_INT; rec.v[i] = vals[i]; }
        gp_rec *dr;
        cudaMalloc(&dr, sizeof(rec));
        cudaMemcpy(dr, &rec, sizeof(rec), cudaMemcpyHostToDevice);
        return dr;
    };
    int64_t *de;
    cudaMalloc(&de, 64);
    int nb = win ? (win + FT_TMAX - 1 + FS_L - 1) / FS_L + 1 : fdtc_chunks(kvh);
    dim3 grid(kvh, nb);
    auto launch = [&](int t, gp_rec *dr) {
        if (win) {
            if (t == 1) fdts_run<1>(op, grid, dr, de, (float *)dpart);
            else if (t == 2) fdts_run<2>(op, grid, dr, de, (float *)dpart);
            else fdts_run<3>(op, grid, dr, de, (float *)dpart);
        } else if (t == 1) fdtc_run<1>(op, grid, dr, de, (float *)dpart);
        else if (t == 2) fdtc_run<2>(op, grid, dr, de, (float *)dpart);
        else fdtc_run<3>(op, grid, dr, de, (float *)dpart);
    };
    auto run = [&](int t, gp_rec *dr) {
        launch(t, dr);
        if (win) k_attn_fdts_join<<<dim3(t * qh, 2), 128>>>(dr, de, (const float *)dpart, nb);
        else k_attn_join<<<dim3(t * qh, 4), 128>>>(dr, de, (const float *)dpart, nb);
    };
    gp_rec *drT = record(T, 0, n, dout);
    run(T, drT);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("error %s\n", cudaGetErrorString(err)); return 1; }
    std::vector<float> out((size_t)T * qh * hd), out1(out.size());
    cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost);
    // each query alone at its position (a window: a buffer without its
    // first rows, as after a drop of old rows)
    int same = 1;
    for (int j = 0; j < T && T > 1; ++j) {
        int sh = win ? std::max(0, std::min(n + j - win, 13 * j + 5)) : 0;
        run(1, record(1, j, n + j, dout1, sh));
    }
    if (T > 1) {
        cudaDeviceSynchronize();
        cudaMemcpy(out1.data(), dout1, out1.size() * 4, cudaMemcpyDeviceToHost);
        same = memcmp(out.data(), out1.data(), out.size() * 4) == 0;
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    int reps = 50;
    float ms = 0.f;
    for (int w = 0; w < 20; ++w) {            /* the clock up (P8 at rest) */
        cudaEventRecord(e0);
        for (int i = 0; i < reps; ++i) launch(T, drT);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        cudaEventElapsedTime(&ms, e0, e1);
        if (ms > 400.f) break;
        reps = ms < 50.f ? reps * 4 : reps * 2;
    }
    cudaEventRecord(e0);
    for (int i = 0; i < reps; ++i) launch(T, drT);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    cudaEventElapsedTime(&ms, e0, e1);
    ms /= reps;
    double mb = (win ? (double)(win + T - 1) / rows : 1.0) * (kb + vb + nv / 4) / 1e6;
    printf("n %d form %d, %d queries, %d chunks: %.3f ms, %.0f GB/s%s\n", n, form, T, nb, ms, mb / ms,
           T > 1 ? (same ? "; each query as alone: same bits" : "; each query as alone: DIFFERENT") : "");
    if (check) {
        double emax = 0, rmax = 0;
        for (int j = 0; j < T; ++j) {
            int nj = n + j, lj = win ? std::max(0, nj - win) : 0;
            for (int h = 0; h < qh; ++h) {
                int kh = h / (qh / kvh);
                std::vector<double> s(nj, -1e300), o(hd, 0.0);
                double m = -1e300, l = 0;
                const float *qv = &q[((size_t)j * qh + h) * hd];
                for (int x = lj; x < nj; ++x) {
                    double t = 0;
                    for (int d = 0; d < hd; ++d) {
                        size_t ix = (size_t)x * rs + kh * hd + d;
                        double kvl = form == 1 ? k8[ix] * (double)ks8[ix / 32] : k16[ix] * (double)ks[ix / 32];
                        t += kvl * qv[d];
                    }
                    s[x] = t; m = fmax(m, t);
                }
                for (int x = lj; x < nj; ++x) {
                    double p = exp(s[x] - m); l += p;
                    for (int d = 0; d < hd; ++d) {
                        size_t ix = (size_t)x * rs + kh * hd + d;
                        o[d] += p * (form == 0 ? v16[ix] * (double)vs[ix / 32] : v8[ix] * (double)vs8[ix / 32]);
                    }
                }
                for (int d = 0; d < hd; ++d) {
                    double rr = o[d] / l;
                    emax = fmax(emax, fabs(rr - out[((size_t)j * qh + h) * hd + d]));
                    rmax = fmax(rmax, fabs(rr));
                }
            }
        }
        printf("  max err %.3e of max %.3e (rel %.2e)\n", emax, rmax, emax / rmax);
    }
    return same ? 0 : 1;
}
