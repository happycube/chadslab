/* ---------- the capabilities of the GPU ----------
 * REFACTOR_PLAN.md, section 4.2. rc_gpu_init fills gg_caps from the device, and
 * the runner picks each kernel from them: a kernel that needs the tensor
 * cores runs only with caps.tc. The environment switches of the kernels
 * (RECLANK_GPU_*_TC=0) only turn kernels off. Before rc_gpu_init every field
 * allows its kernels, as before the caps.
 *
 * RECLANK_GPU_CC=61 makes the GPU act as one of that compute capability
 * (the kernels of a Pascal GPU on a newer one), to test those paths. */
typedef struct {
    int cc;              /* the compute capability: 86 for 8.6 (or RECLANK_GPU_CC) */
    int tc;              /* mma.sync (f16, bf16, s8, tf32), ldmatrix, cp.async: sm_80 */
    int fp8_mma;         /* mma with e4m3 inputs: sm_89 */
    int fp4_mma;         /* block-scaled FP4 mma: sm_120a (an arch-specific build) */
    int pdl;             /* programmatic dependent launch: sm_90 */
    int nanosleep;       /* __nanosleep: sm_70 */
    int smem_block_max;  /* the most shared memory of a block (opt-in), bytes */
    int warp;            /* the threads of a warp */
    int sm_count;
    int unified;         /* the GPU reads host memory as its own (an integrated GPU) */
} rc_caps;

static rc_caps gg_caps = {0, 1, 0, 0, 1, 1, 48 << 10, 32, 0, 0};

static void gg_caps_init(int device)
{
    int major = 0, minor = 0, smem = 0, sms = 0, integrated = 0, warp = 32;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
    cudaDeviceGetAttribute(&smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
    cudaDeviceGetAttribute(&integrated, cudaDevAttrIntegrated, device);
    cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, device);
    int cc = major * 10 + minor;
    const char *v = getenv("RECLANK_GPU_CC");
    if (v && atoi(v) > 0 && atoi(v) < cc) {
        cc = atoi(v);
    }
    gg_caps.cc = cc;
    gg_caps.tc = cc >= 80;
    gg_caps.fp8_mma = cc >= 89;
#if defined(RC_FP4_MMA)
    gg_caps.fp4_mma = cc >= 120;
#else
    gg_caps.fp4_mma = 0;
#endif
    gg_caps.pdl = cc >= 90;
    gg_caps.nanosleep = cc >= 70;
    gg_caps.smem_block_max = cc < 70 ? (48 << 10) : smem;
    gg_caps.warp = warp;
    gg_caps.sm_count = sms;
    gg_caps.unified = integrated;
}
